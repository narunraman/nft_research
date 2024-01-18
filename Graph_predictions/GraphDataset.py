import os.path as osp
import random
import torch
from torch_geometric.data import Dataset, download_url
import pickle
import os
import numpy as np

class GraphDataset(Dataset):

    def create_mask_dict(self,training,test,val):
        percentages = [training, test, val]
        # Total number of elements in the list
        total_elements = len(self.label_list)  # You can adjust this number based on your needs
        # Calculate the number of elements for each value based on percentages
        counts = [int(total_elements * (percentage / sum(percentages))) for percentage in percentages]
        # Generate the list with the desired distribution
        result_list = [0] * counts[0] + [1] * counts[1] + [2] * counts[2] +[0] *100
        random.shuffle(result_list)
        print(len(result_list))
        print(len(self.label_list))
        mask_dict = {i+1:result_list[i] for i,label in enumerate(self.label_list)}
        mask_dict[0] = 5
        return mask_dict
        
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None,training=70,test=20,val=10,normalize=False):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.root = root
        self.normalize = normalize
        with open(osp.join(self.root, f'label_list.pkl'),'rb') as f:
            self.label_list = pickle.load(f)
        self.mask_dict = self.create_mask_dict(training,test,val)
        if normalize:
            with open(osp.join(self.root, f'total_stats.pkl'),'rb') as f:
                self.total_stats = pickle.load(f)
            values = [x[0] for x in self.total_stats.values() if x[0] is not None]
            self.quantiles = self.make_quantiles(values,100)
    def make_quantiles(self,values, num_bins):
        # Use numpy's percentile function to calculate quantiles
        quantiles = np.percentile(values, np.linspace(0, 100, num_bins + 1))
        return quantiles
    
    def bin_value(self,values,quantiles):
        # Bin the values based on quantiles
        bins = np.digitize(values, quantiles, right=True)
        return bins      

    
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
        if self.normalize:
            data.y = torch.FloatTensor(self.bin_value(data.y,self.quantiles))
        data.training_mask = torch.BoolTensor([self.mask_dict[i]==0 for i in data.label.tolist()])
        data.test_mask = torch.BoolTensor([self.mask_dict[i]==1 for i in data.label.tolist()])
        data.val_mask = torch.BoolTensor([self.mask_dict[i]==2 for i in data.label.tolist()])
        return data