import sys
sys.path.append("..")
from psql_methods import execute_commands
from opensea_methods import *
import networkx as nx
import matplotlib.pyplot as plt
import pickle
import random
from tqdm import tqdm
import os
import pandas as pd
from scipy.spatial import distance
import itertools
import numpy as np

def normalize_vector(vector):
    magnitude = np.linalg.norm(vector)
    if magnitude != 0:
        return vector / magnitude
    else:
        return vector

def compute_average_vector(df,column='Label'):
    grouped = df.groupby('Label')
    # Compute average feature vector for each label
    average_feature_vectors = []
    for label, group in tqdm(grouped):
        # Compute average feature vector for the current label
        avg_feature_vector = np.mean(group['Features'].tolist(), axis=0)
        average_feature_vectors.append((label, avg_feature_vector))
    average_features_df = pd.DataFrame(average_feature_vectors, columns=['Label', 'AverageFeatureVector'])
    merged_df2 = pd.merge(df, average_features_df, on='Label')
    return merged_df2

def euclidean_distance(feature1, feature2,normed=False):
    if normed:
        feature_vector1 = feature1/np.linalg.norm(feature1)
        feature_vector2 = feature2/np.linalg.norm(feature2)
    return distance.euclidean(feature1, feature2)

def dot_distance(feature1,feature2):
    return np.dot(feature1, feature2)
# Assuming merged_df is your pandas DataFrame with 'Label' and 'AverageFeatureVector' columns



def pairwise_distances(df1,df2,feature_col = 'AverageFeatureVector'):
    grouped1 = df1.groupby('Collection')
    grouped2 = df2.groupby('Collection')

    # Compute distances between average feature vectors for every pair of labels
    pairwise_distances = []
    for label1,group1 in tqdm(grouped1):
        for label2,group2 in grouped2:
            avg_feat1 = group1[feature_col].iloc[0]
            avg_feat2 = group2[feature_col].iloc[0]
            feature_vector1 = avg_feat1/np.linalg.norm(avg_feat1)
            feature_vector2 = avg_feat2/np.linalg.norm(avg_feat2)
            # print(feature_vector1)
            dist1 = euclidean_distance(feature_vector1, feature_vector2)
            dist2 = dot_distance(feature_vector1, feature_vector2)
            pairwise_distances.append((label1, label2, dist1,dist2))
    
    # Create a DataFrame for pairwise distances
    pairwise_distances_df = pd.DataFrame(pairwise_distances, columns=['Top_100', 'Alt', 'Euc_Distance','Dot_Distance'])

    # Print the DataFrame with pairwise distances
    return pairwise_distances_df

def get_smallest_values(group, n=9, column='Euc_Distance'):
    return group.nsmallest(n, column)