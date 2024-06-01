import sys
sys.path.append("..")
import pandas as pd
import psql_methods as psql
import feature_extract
import numpy as np
import torch 
from tqdm import tqdm

def create_snob_df(img_path,feat_path):
    model_string = 'dinov2_vits14'
    data_path = f'/global/scratch/tlundy/NFT_Research/nft_research/Dino/{img_path}'
    out_path = f'/global/scratch/tlundy/NFT_Research/nft_research/Dino/{feat_path}/{model_string}'
    feature_path = out_path+'/testfeat.pth'
    features = torch.load(feature_path)
    labels = feature_extract.get_labels(data_path)
    file_names = feature_extract.get_filenames(data_path)
    #Build first df of features and info
    commands = ["SELECT * from collectiontoaddress"]
    data = psql.execute_commands(commands)
    slug_to_contract = {x[0]:x[1] for x in data}
    features_list = features.tolist()
    # Create a pandas DataFrame
    data = {'Label': labels.tolist(), 'Features': features_list,'Collection':[x[0] for x in file_names],
            'NFT_num':[x[1] for x in file_names], 'Contract': [slug_to_contract.get(x[0],None) for x in file_names]}
    df = pd.DataFrame(data)
    #clean up for memory
    del data
    del features
    del features_list
    command = "Select * from nfttosales_2"
    sales = psql.execute_commands([command])
    # Column names for the DataFrame
    columns = ['Contract', 'NFT_num', 'sale_price']
    # Create a DataFrame from the list of tuples
    df_2 = pd.DataFrame(sales, columns=columns)
    merged_df = pd.merge(df, df_2, on=['Contract','NFT_num'])
    del df 
    del df_2
    df_filtered = merged_df.groupby('Collection').filter(lambda group: len(group) >= 100)
    merged_df2 = add_average_feat(df_filtered)
    merged_df2['distance'] = [np.linalg.norm(np.array(row['Features']) - np.array(row['AverageFeatureVector'])) for _, row in merged_df2.iterrows()]
    merged_df_no_feat = merged_df2.drop(['Features','AverageFeatureVector'], axis=1)
    return merged_df_no_feat

def add_average_feat(df_filtered):
    grouped = df_filtered.groupby('Label')
    # Compute average feature vector for each label
    average_feature_vectors = []
    for label, group in tqdm(grouped):
        # Compute average feature vector for the current label
        avg_feature_vector = np.mean(group['Features'].tolist(), axis=0)
        average_feature_vectors.append((label, avg_feature_vector))
    average_features_df = pd.DataFrame(average_feature_vectors, columns=['Label', 'AverageFeatureVector'])
    merged_df2 = pd.merge(df_filtered, average_features_df, on='Label')
    return merged_df2