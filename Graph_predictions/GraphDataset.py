import os.path as osp
from random import Random 
import torch
from torch_geometric.data import Dataset, download_url
import pickle
import os
import numpy as np

class GraphDataset(Dataset):


        
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None,training=70,test=20,val=10,normalize=False):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.root = root
   
    @property
    def processed_file_names(self):
        file_names = [f for f in os.listdir(self.root) if os.path.isfile(os.path.join(self.root, f))]
        # Filter file names containing the substring 'graph'
        filtered_file_names = [f for f in file_names if 'graph' in f]
        return filtered_file_names


    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(osp.join(self.root, f'graph_{idx+1}.pt'))
        return data