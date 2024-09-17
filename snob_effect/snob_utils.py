import sys
sys.path.append("..")
import pandas as pd
import data_retrieval.psql_methods as psql
import data_retrieval.opensea_methods as opse
import Dino.feature_extract as feature_extract
import numpy as np
import torch 
from tqdm import tqdm
import pickle
from utils.image_utils import pull_image_from_url
import multiprocessing
from more_itertools import chunked

def assemble_data_and_features(img_path,feat_path):
    model_string = 'dinov2_vits14'
    data_path = f'/global/scratch/tlundy/NFT_Research/nft_research/Dino/images_features/{img_path}'
    out_path = f'/global/scratch/tlundy/NFT_Research/nft_research/Dino/images_features/{feat_path}/{model_string}'
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
    return df

def add_distances(df):
    df_filtered = df.groupby('Collection').filter(lambda group: len(group) >= 100)
    merged_df2 = add_average_feat(df_filtered)
    merged_df2['distance'] = [np.linalg.norm(np.array(row['Features']) - np.array(row['AverageFeatureVector'])) for _, row in merged_df2.iterrows()]
    merged_df_no_feat = merged_df2.drop(['Features','AverageFeatureVector'], axis=1)
    return merged_df_no_feat

def create_snob_df(img_path,feat_path,case_study=False):
    df = assemble_data_and_features(img_path,feat_path)
    merged_df = add_sales(df,case_study)
    del df 
    merged_df_no_feat = add_distances(merged_df)
    return merged_df_no_feat

def add_sales(df,case_study=False):
    if case_study:
        command = f"SELECT slug,token_id,sale_price from cf_sales where slug in {valid_tuple}"
        temp_stats_tuples = psql.execute_commands([command])
        columns = ['slug','token_id', 'sale_price']
        sales_df = pd.DataFrame(temp_stats_tuples,columns=columns)
        avg_sales = sales_df.groupby(['slug','token_id']).mean().reset_index()
        avg_sales.sort_values(by=['slug','token_id'],ascending=False,inplace=True)
        merged_df = pd.merge(df, avg_sales, left_on=['slug','token_id'],right_on=['Collection','NFT_num'])
    else:
        command = "Select * from nfttosales_2"
        sales = psql.execute_commands([command])
        # Column names for the DataFrame
        columns = ['Contract', 'NFT_num', 'sale_price']
        # Create a DataFrame from the list of tuples
        df_2 = pd.DataFrame(sales, columns=columns)
        merged_df = pd.merge(df, df_2, on=['Contract','NFT_num'])
    return merged_df

def add_rarity_ranks_to_df(df):
    command = "Select contract,token_id,rank from nft_to_rarity_2"
    rarities = psql.execute_commands([command])
    # print(rarities)
    # Column names for the DataFrame
    columns = ['Contract', 'NFT_num', 'rarity_rank']
    # Create a DataFrame from the list of tuples
    df_rare = pd.DataFrame(rarities, columns=columns)
    merged_df_rare = pd.merge(df_rare, df, on=['Contract','NFT_num'])
    return merged_df_rare

#Creates total dataframe with both expanded and original images and adds rarity ranks
def create_master_snob_df():
    df1 = create_snob_df('expanded_images','expanded_features')
    df2 = create_snob_df('images','features')
    total_df =pd.concat([df1, df2], ignore_index=True)
    command = "Select contract,token_id,rank from nft_to_rarity_2"
    rarities = psql.execute_commands([command])
    # print(rarities)
    # Column names for the DataFrame
    columns = ['Contract', 'NFT_num', 'rarity_rank']
    # Create a DataFrame from the list of tuples
    df_rare = pd.DataFrame(rarities, columns=columns)
    merged_df_rare = pd.merge(df_rare, total_df, on=['Contract','NFT_num'])
    return merged_df_rare

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

# ----------------- Data Retrieval Methods -----------------
def create_case_study_images(filepath = 'valid_slugs.pkl'):
    #load pickle file containing tuple of slugs
    with open(filepath, 'rb') as f:
        valid_tuple = pickle.load(f)
    command = [f"select * from nfttoimage where slug in {valid_tuple}"]
    rows  = psql.execute_commands(command)
    # Column names for the DataFrame
    columns = ['slug', 'token_id', 'url']

    # Create a DataFrame from the list of tuples
    df_images = pd.DataFrame(rows, columns=columns)
    grouped_data = df_images.groupby('slug').apply(lambda x: (x['slug'].iloc[0], list(zip(x['token_id'], x['url']))))
    args = list(grouped_data)
    records = []
    num_processes = multiprocessing.cpu_count()
    with multiprocessing.Pool(processes=num_processes) as pool: # Use all cores   
        for result in pool.starmap(pull_image_from_url, args):
            records.append(result)

def get_nft_rarities(df):
    selected_columns = sorted(df[['Contract', 'NFT_num']].to_numpy().tolist())
    command = "Select contract,token_id from nft_to_rarity_2"
    rarities = psql.execute_commands([command])
    print(len(selected_columns))
    selected_columns_trim = [tuple(x) for x in selected_columns]
    rarities_set = set(rarities)
    selected_columns = [list(sublist) for sublist in selected_columns_trim if sublist not in rarities_set]
    print(len(selected_columns))
    command = "INSERT INTO nft_to_rarity_2 (contract, token_id, rare_score, rank) VALUES (%s, %s, %s,%s) returning token_id"
    rares_to_grab  = list(chunked(selected_columns,5000))
    rarities = []
    print(len(rares_to_grab))
    bad_contracts = []
    for rare_chunk in rares_to_grab:
        for contract,token_id in tqdm(rare_chunk):
            if contract in bad_contracts:
                continue
            rare_dict = opse.pull_nft_rarity(contract,token_id)
            if rare_dict is None:
                print("Bad token")
                continue
            rarities.append((contract,token_id,rare_dict.get('score',''),rare_dict.get('rank',-1)))
        psql.batch_insert(command,rarities)
        rarities = []