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

def generate_dataset(run_name, max_graph_size=2000,owner_sample=100,num_graphs=20):
    # Specify the directory path
    directory_path = f'dataset_stor/{run_name}'
    # Create the directory
    os.makedirs(directory_path,exist_ok=True)
    with open('dataset_stor/label_to_owners_dec_15.pkl','rb') as f:
        glo_label_to_owners = pickle.load(f)
    label_list = []
    total_owners = {}
    total_stats = {}
    label_map = defaultdict(lambda: len(label_map) + 1)
    wallets = list(glo_label_to_owners.values())
    merged = list(itertools.chain.from_iterable(wallets))
    for x in range(0,num_graphs):
        #Randomly select owner_sample num wallets
        select_wallets = random.sample(merged, owner_sample)
        #Find all NFTs owned by random sample and transform toset
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
        out_path = f'dataset_stor/{run_name}/graph_{x+1}.pt'
        torch.save(graph,out_path)
    with open(f'dataset_stor/{run_name}/total_owners.pkl','wb') as f:
        pickle.dump(total_owners,f)
    with open(f'dataset_stor/{run_name}/total_stats.pkl','wb') as f:
        pickle.dump(total_stats,f)
    with open(f'dataset_stor/{run_name}/label_list.pkl','wb') as f:
        pickle.dump(label_list,f)
    with open(f'dataset_stor/{run_name}/label_map.pkl','wb') as f:
        pickle.dump(dict(label_map),f)