import sys
sys.path.append("..")
import time
import pickle
from torch_geometric.utils.convert import to_networkx
from Openseas_Methods import *
from graph_utils import *
import networkx as nx
from network_utils import *
from GraphDataset import GraphDataset
import argparse
from tqdm import tqdm


def compute_metrics(graph_idx):
    dataset = GraphDataset('dataset_stor/graph_dataset_3',normalize=True)

    graph = to_networkx(dataset.get(graph_idx), node_attrs=['collection', 'label', 'training_mask', 'test_mask', 'val_mask'])

    # Compute graph-wide metrics once
    eigenvector = nx.eigenvector_centrality_numpy(graph)
    print('eigenvector centrality completed')

    # Function to compute node-specific metrics
    def compute_node_specific_metrics(G, node):
        metrics = {}
        metrics_time = {}

        # Computing each metric and recording the time taken
        start_time = time.time()
        metrics["degree"] = G.degree(node)
        metrics_time['degree'] = time.time() - start_time

        start_time = time.time()
        metrics["clustering_coefficient"] = nx.clustering(G, node)
        metrics_time["clustering coefficient"] = time.time() - start_time

        # start_time = time.time()
        # NOTE: closeness is super slow
        # metrics["closeness_centrality"] = nx.closeness_centrality(G, node)
        # print("closeness centrality", time.time() - start_time)

        start_time = time.time()
        metrics["square_clustering"] = nx.square_clustering(G, node)
        metrics_time['square_clustering'] = time.time() - start_time

        # Adding eigenvalue
        metrics["eigenvalue"] = max(eigenvector.values())  # Eigenvalue corresponding to the eigenvector
        return metrics, metrics_time

    # Assigning metrics as node attributes for the subset of collection nodes
    collection_nodes = [node for node in graph if graph.nodes[node]['collection']]
    for i, node in tqdm(enumerate(collection_nodes), total=len(collection_nodes)):
        if graph.nodes[node]['collection']:
            node_metrics, time_metrics = compute_node_specific_metrics(graph, node)
        
        # node_metrics["harmonic_centrality"] = harmonic[node]
        node_metrics["eigenvector_centrality"] = eigenvector[node]
        # node_metrics["pagerank"] = pagerank[node]
        graph.nodes[node]['metrics'] = node_metrics

        graph.nodes[node]['time_metrics'] = time_metrics
        if i % 100 == 0:
            pickle.dump(graph, open(f"graph_metrics/graphs/graph_{graph_idx}.gpickle", 'wb'))

    return graph
    


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--graph', '-g', type=int, required=True)
    args = parser.parse_args()
    graph = compute_metrics(args.graph)
    pickle.dump(graph, open(f"graph_metrics/graphs/graph_{args.graph}.gpickle", 'wb'))


main()