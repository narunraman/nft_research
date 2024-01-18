
import torch
from itertools import combinations
import networkx as nx
from tqdm import tqdm
import numpy
import multiprocessing
import numpy as np

def compute_overlap(edge):
    addresses_one = label_dict[edge[0]]
    addresses_two = label_dict[edge[1]]
    try:
        weight = 1-len(set(addresses_one+addresses_two))/(len(set(addresses_one))+len(set(addresses_two)))
    except:
        return None
    return (edge[0],edge[1],weight)

def compute_edge_weights(label_to_owners):
    comm_edges = combinations(label_to_owners.keys(),2)
    comm_edges = list(comm_edges)
    edge_list = []
    global label_dict
    label_dict = label_to_owners
    num_processes = multiprocessing.cpu_count()
    with multiprocessing.Pool(processes=num_processes) as pool: # Use all cores   
        for result in pool.map(compute_overlap, comm_edges):
            edge_list.append(result)
    return edge_list

def make_graph_from_stats(label_to_stats,edge_list):
    #Currently just expects a dict mapping label to tuple (floor,num_owners)
    G = nx.Graph()
    for node_name, stats in label_to_stats.items():
        if floor_price is not None:
            G.add_node(node_name,y = float(stats[0]),x=[float(stats[1])])
    G.add_weighted_edges_from(edge_list)
    edges_to_remove = [(u, v) for u, v, weight in G.edges(data='weight') if weight == 0]
    G.remove_edges_from(edges_to_remove)
    # Define the attribute you want to check
    attribute_to_check = 'y'
    # Remove nodes without the specified attribute
    nodes_to_remove = [node for node, data in G.nodes(data=True) if data[attribute_to_check] is None]
    G.remove_nodes_from(nodes_to_remove)
    return G

def make_quantiles(values, num_bins):
    # Use numpy's percentile function to calculate quantiles
    quantiles = np.percentile(values, np.linspace(0, 100, num_bins + 1))
    return quantiles

def bin_value(value,quantiles):
    # Bin the values based on quantiles
    bin = np.digitize(value, quantiles, right=True)
    return bin
def make_bipartite_graph(label_to_owners,label_to_stats,label_map,bin=True,num_bins=10):
    if bin:
        stats = label_to_stats.values()
        floors = [x[0] for x in stats if x[0] is not None]
        quantiles = make_quantiles(floors,num_bins)
    G = nx.DiGraph()
    for label in tqdm(label_to_owners.keys()):
        try:
            stats = label_to_stats[label]
        except:
            continue
        if stats[0] is None:
            continue
        if bin:
            G.add_node(label,y=int(bin_value(stats[0],quantiles)),x=[float(stats[1])],collection=True,label=label_map[label])
        else:
            G.add_node(label,y=float(stats[0]),x=[float(stats[1])],collection=True,label=label_map[label])
        # except:
        #     continue
        for wall in label_to_owners[label]:
            G.add_node(wall,y=0,x=[0],collection=False,label=0)
            G.add_edge(label,wall)
            G.add_edge(wall,label)
    return G


        