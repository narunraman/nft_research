import sys
sys.path.append("..")
import torch
from itertools import combinations
import itertools
from Openseas_Methods import *
from alchemy_methods import *
import networkx
from tqdm import tqdm
from torch_geometric.utils.convert import from_networkx
import matplotlib.pyplot as plt
from graph_utils import *
from network_utils import *
import random
import os
from collections import defaultdict
import pickle
import os.path as osp
from random import Random

def generate_dataset(run_name, max_graph_size=2000,owner_sample=100,num_graphs=20,process=True):
    # Specify the directory path
    directory_path = f'dataset_stor/{run_name}'
    graph_path = f'dataset_stor/{run_name}/graphs'
    # Create the directory
    os.makedirs(directory_path,exist_ok=True)
    os.makedirs(graph_path,exist_ok=True)
    with open('dataset_stor/label_to_owners_dec_15.pkl','rb') as f:
        glo_label_to_owners = pickle.load(f)
    label_list = []
    total_owners = {}
    total_stats = {}
    label_map = defaultdict(lambda: len(label_map) + 1)
    wallets = list(glo_label_to_owners.values())
    merged = list(itertools.chain.from_iterable(wallets))
    """A technical note on the random sampling: Grabbing a large list of collection slugsis not straightforward. There is no API endpoint that contains a master list afaik. One can monitor the 'Event' api but this is slow and biased towards frequently traded NFTs. Another option is to get a large list of 'wallets' sample wallets randomly and then grab all collection owned by them. We use this second method. In order to get large list of wallets we begin with a small number of popular collections via webcrawling, we get a list of all owners of these this small set, we then get a list of all NFTs owned by these wallets, and finally get a large list of wallets who own this new set of NFTs. This set of wallets is now too large to expand out completely so we repeatedly sub sample wallets and build out their owned NFTs. This results in our master list of NFTs used for the large dataset. This tevhnically biases away from NFT colelction which are owned by few wallets in the "central cluster"."""
    for x in range(0,num_graphs):
        #Randomly select owner_sample num wallets
        select_wallets = random.sample(merged, owner_sample)
        #Find all NFTs owned by random sample and transform to set
        NFT_list = owners_to_NFT(select_wallets)
        NFT_set = list(set(NFT_list))
        print(f'Number of NFTs found {len(NFT_set)}')
        #Compute Sales stats for all new NFTs and retrieve sales for old NFTs
        Stats_to_process = [x[0] for x in NFT_set if x[0] not in total_stats.keys()]
        Stats_completed =  [x[0] for x in NFT_set if x[0] in total_stats.keys()]
        already_completed_stats = {x:total_stats[x] for x in Stats_completed}
        label_stats = pull_nft_stats(Stats_to_process,no_save=True)
        label_to_stats = {}
        for stat in label_stats:
            # if stat['slug'] in label_to_owners.keys():
                #Sometimes Floor prices are recorded as 0 and we discard these
                #but save that we have seen them
            if stat['floor_price']>0:
                label_to_stats[stat['slug']] = (stat['floor_price'],stat['num_owners'],stat['average_price'])
            else:
                label_to_stats[stat['slug']] = (None,None,None)
        #Update the dictionaries
        label_to_stats.update(already_completed_stats)
        total_stats.update(label_to_stats)
        #Nodes that will be kept in the graph are ones with valid floor prices
        valid_set = [key for key,val in label_to_stats.items() if val[0] is not None]
        NFT_set = [x for x in NFT_set if x[0] in valid_set]
        print(f'Number of valid NFTs in Graph {len(NFT_set)}')
        if len(NFT_set)>max_graph_size:
            NFT_set = random.sample(NFT_set, max_graph_size)
        #Filter out NFTS after computing stats as many NFTs fail the stats call
        print(f'Number of valid NFTs in Graph post limit {len(NFT_set)}')
        new_NFTs = [x[0] for x in NFT_set]
        label_list = list(set(label_list+new_NFTs))
        NFTs_to_process = [x for x in NFT_set if x[0] not in total_owners.keys()]
        NFTs_completed =  [x for x in NFT_set if x[0] in total_owners.keys()]
        print(f'{len(NFTs_to_process)} new NFTs to process and {len(NFTs_completed)} NFTs already seen.')
        already_completed_owners = {x[0]:total_owners[x[0]] for x in NFTs_completed}
        label_to_owners = contracts_to_owners(NFTs_to_process)
        label_to_owners.update(already_completed_owners)
        total_owners.update(label_to_owners)

        G = make_bipartite_graph(label_to_owners,label_to_stats,label_map,bin=False)
        graph = input_from_networkx(G)
        del G
        out_path = f'dataset_stor/{run_name}/graphs/graph_{x+1}.pt'
        torch.save(graph,out_path)
    with open(f'dataset_stor/{run_name}/total_owners.pkl','wb') as f:
        pickle.dump(total_owners,f)
    with open(f'dataset_stor/{run_name}/total_stats.pkl','wb') as f:
        pickle.dump(total_stats,f)
    with open(f'dataset_stor/{run_name}/label_list.pkl','wb') as f:
        pickle.dump(label_list,f)
    with open(f'dataset_stor/{run_name}/label_map.pkl','wb') as f:
        pickle.dump(dict(label_map),f)
    if process:
        process_dataset(f'{run_name}_processed',run_name)

def make_quantiles(values, num_bins):
    # Use numpy's percentile function to calculate quantiles
    quantiles = np.percentile(values, np.linspace(0, 100, num_bins + 1))
    return quantiles

def bin_value(values,quantiles):
    # Bin the values based on quantiles
    bins = np.digitize(values, quantiles, right=True)
    return bins   

def create_mask_dict(label_list,training,test,val,seed):
    percentages = [training, test, val]
    # Total number of elements in the list
    total_elements = len(label_list)  # You can adjust this number based on your needs
    # Calculate the number of elements for each value based on percentages
    counts = [int(total_elements * (percentage / sum(percentages))) for percentage in percentages]
    # Generate the list with the desired distribution
    result_list = [0] * counts[0] + [1] * counts[1] + [2] * counts[2] +[0] *100
    Random(seed).shuffle(result_list)
    mask_dict = {i+1:result_list[i] for i,label in enumerate(label_list)}
    mask_dict[0] = 5
    return mask_dict

def process_dataset(source_name,dest_name,normalize=True,num_bins=100,training=80,test=20,val=10,seed=123):
    source_path = f'dataset_stor/{source_name}/'
    dest_path = f'dataset_stor/{dest_name}/'
    os.makedirs(dest_path,exist_ok=True)
    # Iterate through all files in the specified directory
    graph_path = os.path.join(source_path, 'graphs')
    dest_graph_path = os.path.join(dest_path, 'graphs')
    os.makedirs(dest_graph_path,exist_ok=True)
    with open(osp.join(source_path, f'label_list.pkl'),'rb') as f:
        label_list = pickle.load(f)
    mask_dict = create_mask_dict(label_list,training,test,val,seed)
    if normalize:
        with open(osp.join(source_path, f'total_stats.pkl'),'rb') as f:
            total_stats = pickle.load(f)
        values = [x[0] for x in total_stats.values() if x[0] is not None]
        quantiles = make_quantiles(values,num_bins)
    for filename in os.listdir(graph_path):
        # Construct the full file path
        file_path = os.path.join(graph_path, filename)
        # Check if it's a regular file (not a directory)
        if os.path.isfile(file_path):
            data = torch.load(osp.join(file_path))
            if normalize:
                data.y = torch.FloatTensor(bin_value(data.y,quantiles))
            data.training_mask = torch.BoolTensor([mask_dict[i]==0 for i in data.label.tolist()])
            data.test_mask = torch.BoolTensor([mask_dict[i]==1 for i in data.label.tolist()])
            data.val_mask = torch.BoolTensor([mask_dict[i]==2 for i in data.label.tolist()])
            out_path = os.path.join(dest_graph_path, filename)
            torch.save(data,out_path)

        