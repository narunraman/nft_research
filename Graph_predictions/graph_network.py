print('Starting imports')

import sys
sys.path.append("..")
import torch
from torch_geometric.loader import DataLoader
from graph_utils import *
from network_utils import *
from GraphDataset import GraphDataset

print('Starting script')
dataset = GraphDataset('dataset_stor/graph_dataset_4/graphs')



device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

gcn = GCN().to(device)

loader = DataLoader(dataset, batch_size=1)
optimizer_gcn = torch.optim.Adam(gcn.parameters(), lr=0.01, weight_decay=5e-4)
criterion = torch.nn.MSELoss()
print('Starting training')
gcn = train_node_classifier(gcn, loader, optimizer_gcn, criterion,2500,print_every=100,bin=False,best_val=30000,outpath='models/train_model_slurm.pt')
torch.save(gcn, 'models/final_model.pt')
test_acc(gcn,loader,criterion,bin=False)
