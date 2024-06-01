print('Starting imports')
import argparse
import random
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import torch
import pickle
from network_utils import GCN, test_acc
from GraphDataset import GraphDataset, InMemoryGraphDataset
from torch_geometric.loader import DataLoader
import glob
import sys
import time
from tqdm import tqdm
from torch_geometric.utils.convert import to_networkx, from_networkx
from sklearn.metrics import mean_squared_error
import numpy as np

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def create_one_hot_mask(source_list, target_element):
    """
    Create a one-hot encoded NumPy array for the first occurrence of the target_element in the source_list.

    Parameters:
    - source_list: The list to search in.
    - target_element: The element to encode as 1.

    Returns:
    - A NumPy array where all elements are 0 except for a 1 at the index of the target_element.
    """
    # Initialize a NumPy array of zeros with the same length as the source list
    one_hot_array = torch.zeros(len(source_list), dtype=torch.bool)
    # Attempt to find the index of the target element
    try:
        # target_index = source_list.index(target_element)
        target_index = (source_list == target_element).nonzero(as_tuple=True)[0]
        # Set the value at the target index to 1
        one_hot_array[target_index] = True
    except ValueError:
        # If the target element is not found, the one_hot_array remains all zeros
        print('Error!')
    return one_hot_array


def predict_val(model, data_loader, eval_mask):
    deltas = []
    with torch.no_grad():  # Disable gradient computation
        for i, data in enumerate(data_loader):
            try:
                data = data.to(device)
            except:
                print('didnt work')
            mask = eval_mask[i]
            try:
                out = model(data)
            except RuntimeError as e:
                if 'CUDA' in str(e):
                    time.sleep(1)
                    try:
                        out = model(data)
                    except RuntimeError:
                        deltas.append((torch.tensor([0]), torch.tensor([0])))
            y = out[mask].squeeze(0)
            x = data.y[mask]
            deltas.append((y, x))
    return deltas


# Function to calculate RMSE for a given set of predictions and true values
def calculate_performance(model, data_loader, criterion, eval_mask=None):
    total_model_loss = 0
    total_r_loss = 0

    with torch.no_grad():  # Disable gradient computation
        for i, data in enumerate(data_loader):

            try:
                data = data.to(device)
            except:
                print('didnt work')
            if not eval_mask:
                mask = data['test_mask'] & data.collection
            else:
                mask = eval_mask[i]
            out = model(data)
            new_tensor = torch.full_like(out, 50).to(device)
            total_model_loss += criterion(out[mask], data.y[mask].unsqueeze(1))
            total_r_loss += criterion(new_tensor[mask], data.y[mask].unsqueeze(1))
        
    return total_model_loss, total_r_loss

def calculate_rmse(model, data_loader, criterion, eval_mask=None):
    model_loss, baseline_loss = calculate_performance(model, data_loader, criterion, eval_mask)
    return np.sqrt(model_loss), np.sqrt(baseline_loss)


def test_performance(model, batch_size, test_dataset, eval_mask=None):
    dataloader = get_dataloader(test_dataset, batch_size)
    deltas = predict_val(model, dataloader, eval_mask)
    return deltas

def mask_tensor_instances(tensor1, tensor2, mask_value=0):
    """
    Masks instances of tensor2 from tensor1 by setting matching elements to mask_value.

    Parameters:
    - tensor1 (torch.Tensor): The first 2-row tensor.
    - tensor2 (torch.Tensor): The second 2-row tensor, elements of which to mask in tensor1.
    - mask_value (int, optional): The value to assign to masked elements. Defaults to 0.

    Returns:
    - torch.Tensor: A new tensor with elements masked.
    """
    # Ensure tensor1 is cloned to avoid modifying the original tensor
    result_tensor = tensor1.clone()

    # Iterate through each element in tensor2 and mask it in tensor1
    for value in tensor2.view(-1):  # Flatten tensor2 for iteration
        # Use broadcasting for comparison and mask matching elements
        mask = result_tensor == value
        result_tensor[mask] = mask_value

    return result_tensor


def load_model(model_path):
    print('loading model')
    # Determine the model type and batch size based on the file name
    if '_w_centroids' in model_path:
        model_class = "GCN with centroids"
    else:
        model_class = "GCN no centroids"
    

    # if '4' in model_path:
    #     batch_size = 4
    #     model_class += " batch size 4"
    # elif '3' in model_path:
    #     batch_size = 3
    #     model_class += " batch size 3"
    # else:
    # Default batch size if neither 3 nor 4 is in the filepath
    batch_size = 1
    model_class += " batch size 1"
    
    if '_pfp_only' in model_path:
        model_class += " on only pfps"
    else:
        model_class += " on all collections"
    
    # Load the model
    model = GCN()
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    try:
        model = model.to(device)
    except:
        pass
    
    model.eval()
    return model, model_class, batch_size

def get_dataloader(test_dataset, batch_size):
    # Create DataLoader
    return DataLoader(test_dataset, batch_size=batch_size, pin_memory=True)


def update_edges(test_datasets, num_edges, wallet_probs):
    eval_masks = []
    datasets = []
    
    if num_edges > 0:
        for i, dataset in enumerate(test_datasets):
            collection_node = np.random.choice(dataset.label[dataset.collection & dataset['test_mask']])
            
            mask_1 = collection_node == dataset.edge_index[0]
            wallet_neighbors = torch.unique(dataset.edge_index[1][mask_1])

            remaining_wallets = 2*num_edges
            wallet_indices = torch.tensor([], dtype=torch.int32)
            while(True):
                # sample wallet labels
                sampled_wallet_indices = wallet_probs[i].multinomial(num_samples=min(remaining_wallets, len(wallet_probs[i])), replacement=False)
                wallet_indices = torch.cat((wallet_indices, sampled_wallet_indices), dim=0)
                wallet_indices = wallet_indices.long()
                
                # create a boolean mask of wallet labels
                sampled_wallet_mask = torch.zeros(dataset.label.size(0), dtype=torch.bool)
                sampled_wallet_mask[wallet_indices] = True

                # create a boolean mask of length number of nodes setting trues on the index
                neighbor_mask = torch.zeros(dataset.label.size(0), dtype=torch.bool)
                neighbor_mask[wallet_neighbors] = True

                # compare neighbor_mask against sampled wallet labels 
                non_neighbor_mask = torch.logical_xor(neighbor_mask, sampled_wallet_mask)
                non_neighbor_mask = non_neighbor_mask.bool()
                num_trues = non_neighbor_mask.sum()

                # non_neighbor_mask is length of nodes so need it in length of wallet nodes
                non_neighbor_mask = non_neighbor_mask[~dataset.collection]
                
                # if there are enough non-neighbors in the drawn sample
                if num_trues >= num_edges:
                    wallet_indices = np.random.choice(dataset.wallet_label[non_neighbor_mask], num_edges)
                    break
                wallet_indices = dataset.wallet_label[non_neighbor_mask]

            sampled_wallets = dataset.wallet_label[wallet_indices]

            constant_tensor = torch.full_like(sampled_wallets, collection_node)

            # Stack the constant_tensor and tensor_1d to create a 2D tensor
            # We use torch.stack and specify dim=0 to stack them vertically (along rows)
            two_d_tensor = torch.stack((constant_tensor, sampled_wallets), dim=0)
            dataset.edge_index = torch.cat((dataset.edge_index, two_d_tensor), dim=1)

            two_d_tensor = torch.stack((sampled_wallets, constant_tensor), dim=0)
            dataset.edge_index = torch.cat((dataset.edge_index, two_d_tensor), dim=1)
            eval_mask = create_one_hot_mask(dataset.label, collection_node)

            eval_masks.append(eval_mask)
            datasets.append(dataset)
    elif num_edges < 0:
        def remove_edges(target_tensor, source_tensor):
            mask = torch.zeros(target_tensor.size(1), dtype=torch.bool)

            # Iterate over each column in tensor_2xM and update the mask
            for m_col in source_tensor.t():
                # Expand dimensions for broadcasting and compare
                matches = (target_tensor == m_col.unsqueeze(1)).all(dim=0)
                # Logical OR to update the mask
                mask |= matches

            # Use the mask to filter out matching columns
            filtered_target_tensor = target_tensor[:, ~mask]
            return filtered_target_tensor

        for i, dataset in enumerate(test_datasets):
            collection_node = np.random.choice(dataset.label[dataset.collection & dataset['test_mask']])
            
            # mask is only for the collection
            mask_1 = collection_node == dataset.edge_index[0]
            # NOTE: wallet_labels are indices into dataset.label
            wallet_labels = torch.unique(dataset.edge_index[1][mask_1])

            # indices 
            try:
                cur_probs = torch.sub(1, wallet_probs[i][wallet_labels])
            except:
                print('got a too large wallet index')
                cur_probs = torch.sub(1, wallet_probs[i])
            
            # sample wallet indices according to their inverse importance
            wallet_indices = cur_probs.multinomial(num_samples=min(-num_edges, len(wallet_probs[i])), replacement=True)
            # ensure that tensor is correct type
            wallet_indices = wallet_indices.long()
            
            # NOTE: wallet_indices are indices into dataset.label
            sampled_wallets = dataset.label[wallet_indices]
            
            # Stack the constant_tensor and tensor_1d to create a 2D tensor
            # We use torch.stack and specify dim=0 to stack them vertically (along rows)
            constant_tensor = torch.full_like(sampled_wallets, collection_node)
            two_d_tensor = torch.stack((constant_tensor, sampled_wallets), dim=0)
            
            # remove edges corresponding to the two_d_tensor
            dataset.edge_index = remove_edges(dataset.edge_index, two_d_tensor)

            two_d_tensor = torch.stack((sampled_wallets, constant_tensor), dim=0)
            dataset.edge_index = remove_edges(dataset.edge_index, two_d_tensor)

            eval_mask = create_one_hot_mask(dataset.label, collection_node)
            eval_masks.append(eval_mask)

            datasets.append(dataset)

    return eval_masks, InMemoryGraphDataset(datasets)



def run_experiment(model_name, num_runs, test_dataset, wallet_probs, add2graph, outpath, performance_vals):
    # Path to your 'models/' directory
    models_dir = 'models/'
    if model_name == 'all':
        pattern = models_dir + '*.pt'
    else:
        pattern = models_dir + f'best_model_{model_name}.pt'

    
    # add2graph = 100
    performance = [] #np.empty((num_runs, 50))
    # sub_performance = []

    model, model_class, batch_size = load_model(glob.glob(pattern)[0])
    print(f'loaded Model: {model_class}')

    remaining_runs = num_runs #- len(performance_vals)

    for run_num in tqdm(range(remaining_runs), desc='Rollout'):
        # Iterate over each model file in the directory and eval on expanded graph 

        # generates a mask for each graph and a modified GraphDataset object
        eval_mask, new_dataset = update_edges(test_dataset, add2graph, wallet_probs)
        print(f'Difference after adding {add2graph} edges in each graph:')
        
        reg_performance = test_performance(model, batch_size, test_dataset, eval_mask)
        
        modified_performance = test_performance(model, batch_size, new_dataset, eval_mask)
        pos_count, pos_val, neg_count, neg_val = 0, 0, 0, 0

        # iterating each number in list
        for y_mod, y_reg in zip(modified_performance, reg_performance):
            # checking condition
            if y_mod[0]-y_reg[0] > 0:
                pos_count += 1
                pos_val += y_mod[0]-y_reg[0]
            else:
                neg_count += 1
                neg_val += y_mod[0]-y_reg[0]
        
        print()
        print('======================')
        try:
            print(f"Number Above: {pos_count}, Number Below: {neg_count}, Amount Above: {pos_val.item()}, Amount Below: {neg_val.item()}")
            print(f"Modified Value: {modified_performance},  Regular Value: {reg_performance}")
        except:
            print(f"Number Above: {pos_count}, Number Below: {neg_count}, Amount Above: {pos_val}, Amount Below: {neg_val}")
            print(f"Modified Value: {modified_performance},  Regular Value: {reg_performance}")
        print('======================')
        print()

        performance.append((modified_performance, reg_performance))

        if run_num % 25 == 0:
            torch.save(performance.cpu().detach(), outpath)
        del new_dataset
    
    return performance


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name', '-m', type=str)
    parser.add_argument('--num-runs', '-n', type=int)
    parser.add_argument('--num-add', '-a', type=int)
    parser.add_argument('--num-sub', '-s', type=int)
    parser.add_argument('--type', '-t', type=str)
    parser.add_argument('-chunk', '-c', type=int)
    parser.add_argument('-resume', action='store_true')
    args = parser.parse_args()


    outpath = '/home/narunram/scratch/eval_results_fixed/'
    if args.num_add is not None:
        outpath += f'eval_{args.type}_for_chunk_{args.chunk}_add_{args.num_add}_edges.pt'
    else:
        outpath += f'eval_{args.type}_for_chunk_{args.chunk}_sub_{args.num_sub}_edges.pt'

    if args.resume:
        if os.path.exists(outpath):
            performance_vals = torch.load(outpath)
        else:
            performance_vals = []    
    else:
        performance_vals = []

    print('Loading dataset')
    test_dataset = GraphDataset('/home/narunram/scratch/dataset_stor/graph_dataset_4_processed/modified_graphs')
    wallet_probs = []
    datasets = []

    for i, dataset in enumerate(tqdm(test_dataset)):
        try:
            if args.type is None:
                wallet_nodes = torch.mul(dataset.wealth[~dataset.collection], dataset.affinity[~dataset.collection])
            elif 'affinity' in args.type:
                wallet_nodes = dataset.affinity[~dataset.collection]
            elif 'wealth' in args.type:
                wallet_nodes = dataset.wealth[~dataset.collection]
            elif 'uniform' in args.type:
                wallet_nodes = torch.ones(len(dataset.label[~dataset.collection]))
            wallet_nodes = torch.div(wallet_nodes, torch.sum(wallet_nodes))
            wallet_probs.append(wallet_nodes)
        except:
            print('Error in creating wallet probs')
            pass
        
        dataset.wallet_label =  dataset.label[~dataset.collection]
        # wallet labels are the wallet indices for masking into affinity and wealth
        # dataset.wallet_label = torch.where(~dataset.collection, indices, torch.tensor(0))
        
        if 'centroids' in args.model_name:
            dataset.x = pickle.load(open(f'/home/narunram/scratch/dataset_stor/graph_dataset_4_processed/features_{i+1}.pkl', 'rb'))
        else:
            dataset.x = torch.zeros((len(dataset.label), 1))
        datasets.append(dataset)
        # datasets[i] = dataset
        del dataset['overlap']
        del dataset['floor_price']
    test_dataset = InMemoryGraphDataset(datasets)
    print('Loaded dataset')

    if args.num_add is not None:
        num_update = args.num_add
    else:
        num_update = -args.num_sub
    print('wallet_probs length', len(wallet_probs))
    total_performance = run_experiment(model_name=args.model_name, num_runs=args.num_runs, test_dataset=test_dataset, wallet_probs=wallet_probs, add2graph=num_update, outpath=outpath, performance_vals=performance_vals)
    torch.save(total_performance, outpath)

if __name__ == "__main__":
    main()