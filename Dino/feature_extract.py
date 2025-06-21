# Copyright (c) Facebook, Inc. and its affiliates.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import sys
import argparse
sys.path.append('..')
import torch
from torch import nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torchvision import datasets
from torchvision import transforms as pth_transforms
from torchvision import models as torchvision_models
import logging
import Dino.utils as utils
import Dino.vision_transformer as vits

def get_labels(data_path):
    dataset_val = ReturnIndexDataset(os.path.join(data_path, "val"))
    test_labels = torch.tensor([s[-1] for s in dataset_val.samples]).long()
    return test_labels

def get_filenames(data_path):
    dataset_val = ReturnIndexDataset(os.path.join(data_path, "val"))
    print(dataset_val)
    file_paths = [s[0] for s in dataset_val.samples]
    result_tuples = [(os.path.basename(os.path.dirname(path)), os.path.splitext(os.path.basename(path))[0]) for path in file_paths]
    return result_tuples
    
def extract_feature_pipeline(model,data_path,out_path,use_cuda=False):
    # ============ preparing data ... ============
    #Standard image transformations for network
    transform = pth_transforms.Compose([
        pth_transforms.Resize(256, interpolation=3),
        pth_transforms.CenterCrop(224),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    dataset_val = ReturnIndexDataset(os.path.join(data_path, "val"), transform=transform)
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=128,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
    )
    print(f"Data loaded with {len(dataset_val)} val imgs.")
    model.eval()
    model.cuda()
    # os.environ['MASTER_ADDR'] = '127.0.0.1'
    # os.environ['MASTER_PORT'] = '29500'
    test_labels = torch.tensor([s[-1] for s in dataset_val.samples]).long()
    # ============ extract features ... ============
    print("Extracting features for val set...")
    test_features = extract_features(model, data_loader_val,use_cuda)

    if utils.get_rank() == 0:
        test_features = nn.functional.normalize(test_features, dim=1, p=2)

    
    torch.save(test_features.cpu(), os.path.join(out_path, "testfeat.pth"))
    return test_features,test_labels

def init_group():
    dist.init_process_group(
        backend="gloo",
        init_method='tcp://127.0.0.1:23456',
        world_size=1,
        rank=0,
    )
    
@torch.no_grad()
def extract_features(model, data_loader, use_cuda, multiscale=False):
    metric_logger = utils.MetricLogger(delimiter="  ")
    features = None


# Configure logging to write messages to a file (example.log)
    logging.basicConfig(filename='logs/feature_extract.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    for samples, index in metric_logger.log_every(data_loader, 1):
        if use_cuda:
            samples = samples.cuda(non_blocking=True)
            index = index.cuda(non_blocking=True)
        # print(samples)
        # print(index)
        logging.info(f'Currently working on {index}')
        if multiscale:
            feats = utils.multi_scale(samples, model)
        else:
            feats = model(samples).clone()

        # init storage feature matrix
        if dist.get_rank() == 0 and features is None:
            features = torch.zeros(len(data_loader.dataset), feats.shape[-1])
            if use_cuda:
                features = features.cuda(non_blocking=True)
            print(f"Storing features into tensor of shape {features.shape}")

        # get indexes from all processes
        y_all = torch.empty(1, index.size(0), dtype=index.dtype, device=index.device)
        y_l = list(y_all.unbind(0))
        y_all_reduce = torch.distributed.all_gather(y_l, index, async_op=True)
        y_all_reduce.wait()
        index_all = torch.cat(y_l)

        # share features between processes
        feats_all = torch.empty(
            1,
            feats.size(0),
            feats.size(1),
            dtype=feats.dtype,
            device=feats.device,
        )
        output_l = list(feats_all.unbind(0))
        output_all_reduce = torch.distributed.all_gather(output_l, feats, async_op=True)
        output_all_reduce.wait()
        # update storage feature matrix
        if dist.get_rank() == 0:
            features.index_copy_(0, index_all, torch.cat(output_l))
            # else:
            #     features.index_copy_(0, index_all.cpu(), torch.cat(output_l).cpu())
    return features


@torch.no_grad()
def knn_classifier(train_features, train_labels, test_features, test_labels, k, T, num_classes=1000):
    top1, top5, total = 0.0, 0.0, 0
    train_features = train_features.t()
    num_test_images, num_chunks = test_labels.shape[0], 100
    imgs_per_chunk = num_test_images // num_chunks
    retrieval_one_hot = torch.zeros(k, num_classes).to(train_features.device)
    for idx in range(0, num_test_images, imgs_per_chunk):
        # get the features for test images
        features = test_features[
            idx : min((idx + imgs_per_chunk), num_test_images), :
        ]
        targets = test_labels[idx : min((idx + imgs_per_chunk), num_test_images)]
        batch_size = targets.shape[0]

        # calculate the dot product and compute top-k neighbors
        similarity = torch.mm(features, train_features)
        distances, indices = similarity.topk(k, largest=True, sorted=True)
        candidates = train_labels.view(1, -1).expand(batch_size, -1)
        retrieved_neighbors = torch.gather(candidates, 1, indices)

        retrieval_one_hot.resize_(batch_size * k, num_classes).zero_()
        retrieval_one_hot.scatter_(1, retrieved_neighbors.view(-1, 1), 1)
        distances_transform = distances.clone().div_(T).exp_()
        probs = torch.sum(
            torch.mul(
                retrieval_one_hot.view(batch_size, -1, num_classes),
                distances_transform.view(batch_size, -1, 1),
            ),
            1,
        )
        _, predictions = probs.sort(1, True)

        # find the predictions that match the target
        correct = predictions.eq(targets.data.view(-1, 1))
        top1 = top1 + correct.narrow(1, 0, 1).sum().item()
        top5 = top5 + correct.narrow(1, 0, min(5, k)).sum().item()  # top5 does not make sense if k < 5
        total += targets.size(0)
    top1 = top1 * 100.0 / total
    top5 = top5 * 100.0 / total
    return top1, top5


class ReturnIndexDataset(datasets.ImageFolder):
    def __getitem__(self, idx):
        img, lab = super(ReturnIndexDataset, self).__getitem__(idx)
        return img, idx




def main(model_string: str, data_path: str, out_path: str, use_cuda: bool = True):
    """Initialize and run the feature extraction pipeline."""
    init_group()
    model = torch.hub.load('facebookresearch/dinov2', model_string)
    extract_feature_pipeline(model, data_path, out_path, use_cuda=use_cuda)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DINOv2 feature extraction pipeline')
    parser.add_argument('--model_string', type=str, default='dinov2_vits14',
                        help='Name of the DINOv2 model to load')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to the input images directory')
    parser.add_argument('--out_path', type=str, default=None,
                        help='Directory where features will be written')
    parser.add_argument('--no_cuda', dest='use_cuda', action='store_false', default=True,
                        help='Disable CUDA even if available')
    args = parser.parse_args()

    # If out_path not provided, default to <data_path>_features/<model_string>
    if args.out_path is None:
        args.out_path = f"{args.data_path.rstrip('/')}_features/{args.model_string}"

    main(args.model_string, args.data_path, args.out_path, use_cuda=args.use_cuda)
