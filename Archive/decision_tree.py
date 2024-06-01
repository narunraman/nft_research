from queue import Empty
import requests
from pymongo import MongoClient
from parsing_helpers import parse_sale_data, parse_nft_list, parse_listing_data
import time
import json
import networkx as nx
from collections import defaultdict
import datetime
from tqdm import tqdm
import urllib
from Openseas_Methods import make_nft_graph
import numpy as np
import datetime as dt
import pickle
from datetime import datetime,timedelta

with open('date_to_price.pkl','rb') as f:
    date_to_price = pickle.load(f)

def compute_graph_features(combo_perc,skip_list=None,G=None):
    sorted_edges = {}
    #Edge weight averages
    edge_weight_averages = {}
    feature_dict = {}
    if G is None:
        G = make_nft_graph(list(combo_perc.items()),skip_list=skip_list)
        weight = 'weight'
    else:
        weight=None
    print("Computing Centrality metrics")
    # eigen_centrality = nx.eigenvector_centrality(G,weight=weight)
    clustering_coeff = nx.clustering(G,weight=weight)       
    for node in G.nodes():
        feature_dict[node] = {}
        # feature_dict[node][f'eigen_centrality'] = eigen_centrality[node]
        feature_dict[node][f'clustering_coeff'] = clustering_coeff[node]

    print("Computing Average Edge Weights")
    for node in G.nodes():
        neighbors = G.edges(node, data=True)
        sorted_edges[node] = sorted(neighbors, key=lambda x: x[2]['weight'], reverse=True)
        for x in range(0,30,5):
            s_edges = sorted_edges[node][:x]
            if s_edges:
                weights= [x[2]['weight'] for x in s_edges]
                feature_dict[node][f'average_edge_weight_{x}'] = sum(weights)/len(weights)
            else:
                feature_dict[node][f'average_edge_weight_{x}'] = None
    print("Done Computing Average Edge Weights")
    #Centrality measure
    #Remove low weight edge features
    print("Computing Community")
    comms = nx.community.greedy_modularity_communities(G,weight=None)
    for i,comm_nodes in enumerate(comms):
        comm = G.subgraph(comm_nodes)
        comm_weight = [x[1] for x in list(nx.get_edge_attributes(comm,name='weight').items())]
        if comm_weight:
            avg_comm_weight = sum(comm_weight)/len(comm_weight)
        else:
            avg_comm_weight = None
        for node in comm.nodes():
            feature_dict[node][f'community_index'] = i
            feature_dict[node][f'community_size'] = len(comm.nodes())
            feature_dict[node][f'community_edge_size'] = len(comm.edges())
            feature_dict[node][f'community_average_edge_weight'] = avg_comm_weight
    print("Done Computing Community")
    print("Computing Restricted Graphs")
    if weight is None:
        return feature_dict
    for x in np.arange(0.1,0.8,0.05):
        combo_perc_sub = {key:val for key, val in combo_perc.items() if val >x}
        G = make_nft_graph(list(combo_perc_sub.items()),skip_list=skip_list)
        Gcc = sorted(nx.connected_components(G), key=len, reverse=True)
        for G0_nodes in Gcc:
            G0 = G.subgraph(G0_nodes)
            G0_weight = [x[1] for x in list(nx.get_edge_attributes(G0,name='weight').items())]
            avg_G0_weight = sum(comm_weight)/len(comm_weight)
            for node in G0.nodes():
                feature_dict[node][f'CC_size_{x}'] = len(G0.nodes())
                feature_dict[node][f'CC_edge_size_{x}'] = len(G0.edges())
                feature_dict[node][f'CC_average_edge_weight_{x}'] = avg_G0_weight
    print("Done Computing Restricted Graphs")
    return feature_dict

def db_date_to_geck(date,offset):
    dt = datetime.strptime(date,'%Y-%m-%dT%H:%M:%S')
    #hacky cause its late just fix the dict
    geck_dat = f"{dt.day}-{dt.month}-{dt.year}"
    return geck_dat

def get_eth_price(db_date,offset=0):
    try:
        result = date_to_price[db_date_to_geck(db_date,offset)]
    except:
        result = date_to_price[f"{10}-{10}-{2023}"]
    return date_to_price[db_date_to_geck(db_date,offset)]

    
