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
from datetime import datetime
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
import matplotlib as mpl, font_manager
from matplotlib.ticker import MaxNLocator
from scipy import stats
from sklearn.metrics import r2_score
from sklearn import datasets, linear_model

mpl.rcParams.update(mpl.rcParamsDefault)
plt.style.use('../Dino/narunraman.mplstyle')

def save_data_and_features(img_path,feat_path,feat):
    df = assemble_data_and_features(img_path,feat_path,feat)
    df.to_pickle(f'/global/scratch/tlundy/NFT_Research/nft_research/snob_effect/dataframes/{feat_path}_df.pkl')
    return df

def load_data_and_features(feat_path):
    return pd.read_pickle(f'/global/scratch/tlundy/NFT_Research/nft_research/snob_effect/dataframes/{feat_path}_df.pkl')

def assemble_data_and_features(img_path,feat_path,feat):
    model_string = 'dinov2_vits14'
    data_path = f'/global/scratch/tlundy/NFT_Research/nft_research/Dino/images_features/{img_path}'
    out_path = f'/global/scratch/tlundy/NFT_Research/nft_research/Dino/images_features/{feat_path}/{model_string}'
    feature_path = out_path+'/testfeat.pth'
    print('before torch load')
    if feat:
        features = torch.load(feature_path)
    print('after torch load')
    labels = feature_extract.get_labels(data_path)
    file_names = feature_extract.get_filenames(data_path)
    #Build first df of features and info
    commands = ["SELECT * from collectiontoaddress"]
    print('before psql')
    data = psql.execute_commands(commands)
    slug_to_contract = {x[0]:x[1] for x in data}
    print('after psql')
    if feat:
        features_list = features.tolist()
    else:
        features_list = [None] * len(file_names)
    
    # Create a pandas DataFrame
    print('after feature list')
    data = {'Label': labels.tolist(), 'Features': features_list,'Collection':[x[0] for x in file_names],
            'NFT_num':[x[1] for x in file_names], 'Contract': [slug_to_contract.get(x[0],None) for x in file_names]}
    print('after dict')
    df = pd.DataFrame(data)
    return df

def add_distances(df):
    df_filtered = df.groupby('Collection').filter(lambda group: len(group) >= 100)
    merged_df2 = add_average_feat(df_filtered)
    merged_df2['distance'] = [np.linalg.norm(np.array(row['Features']) - np.array(row['AverageFeatureVector'])) for _, row in merged_df2.iterrows()]
    merged_df_no_feat = merged_df2.drop(['Features','AverageFeatureVector'], axis=1)
    return merged_df_no_feat

def create_snob_df(img_path,feat_path,case_study=False,normalize=False,feat=True):
    #check if df exists
    try:
        print('Loading Data')
        df = load_data_and_features(feat_path)
        print('Loaded Data')
    except:
        print('Creating Data')
        df = save_data_and_features(img_path,feat_path,feat)
    merged_df = add_sales(df,case_study,normalize)
    del df 
    if feat:
        merged_df_no_feat = add_distances(merged_df)
    else:
        return merged_df
    return merged_df_no_feat

def add_sales(df,case_study=False,normalize=False):
    if case_study:
        valid_tuple = tuple(df['Collection'].unique().tolist())
        if normalize:
            command = f"SELECT slug,token_id,sale_price,timestamp from cf_sales where slug in {valid_tuple} and (payment_token='WETH' or payment_token='ETH')"
            temp_stats_tuples = psql.execute_commands([command])
            columns = ['slug','token_id', 'sale_price','timestamp']
            sales_df = pd.DataFrame(temp_stats_tuples,columns=columns)
            sales_df['day'] = [datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d') for timestamp in sales_df['timestamp']]
            # Calculate the average 'saleprice' for each day
            average_saleprice_per_day = sales_df.groupby('day')['sale_price'].transform('median')
            # Add a new column 'normalized_sale_price' to the DataFrame
            sales_df['normalized_sale_price'] = sales_df['sale_price'] / average_saleprice_per_day
            sales_df.drop(columns=['sale_price','timestamp','day'], inplace=True)
            avg_sales = sales_df.groupby(['slug','token_id']).median().reset_index()
        else:
            command = f"SELECT slug,token_id,sale_price,transaction from cf_sales where slug in {valid_tuple} and (payment_token='WETH' or payment_token='ETH')"
            temp_stats_tuples = psql.execute_commands([command])
            columns = ['slug','token_id', 'sale_price','transaction']
            sales_df = pd.DataFrame(temp_stats_tuples,columns=columns)
            sales_df = sales_df.drop_duplicates(subset='transaction')
            sales_df.drop(columns=['transaction'], inplace=True)
            grouped = sales_df.groupby(['slug','token_id'])
            avg_sales = grouped.median().reset_index()
            avg_sales['count'] = grouped.size().reset_index(drop=True)
        #print the types of the columns
        print(avg_sales.dtypes)
        avg_sales['token_id'] = avg_sales['token_id'].astype(str)
        merged_df = pd.merge(df, avg_sales, right_on=['slug','token_id'],left_on=['Collection','NFT_num'])
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
    try:
        df = pd.read_pickle('/global/scratch/tlundy/NFT_Research/nft_research/snob_effect/dataframes/total_df.pkl')
        return df
    except:
        print('could not load total df')
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
        merged_df_rare = pd.merge(df_rare, total_df, on=['Contract','NFT_num'],how='right')
        return merged_df_rare

def load_master_df():
    df = pd.read_pickle('/global/scratch/tlundy/NFT_Research/nft_research/snob_effect/dataframes/total_df.pkl')
    return df

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
# ----------------- Analysis Methods -----------------
def compute_global_corrs(total_df,norm_type='additive',slug_list = None):
    total_df = total_df.copy()
    total_df = total_df.groupby('Collection').filter(lambda group: len(group) >= 100)
    #Keep only rows with rarity ranks
    df_filtered_rare = total_df.dropna(subset=['rarity_rank'],inplace=False).groupby('Collection').filter(lambda group: len(group) >= 100)
    if slug_list:
        df_filtered_rare = df_filtered_rare[df_filtered_rare['Collection'].isin(slug_list)]
    #if norm type additive normalize each collections sale price by subtracting the average sale price
    if norm_type == 'additive':
        df_filtered_rare['sale_price_norm'] = df_filtered_rare['sale_price'] - df_filtered_rare.groupby('Collection')['sale_price'].transform('mean')
    #if norm type multiplicative normalize each collections sale price by dividing by the average sale price
    elif norm_type == 'multiplicative':
        df_filtered_rare['sale_price_norm'] = df_filtered_rare['sale_price'] / df_filtered_rare.groupby('Collection')['sale_price'].transform('mean')
    else:
        df_filtered_rare['sale_price_norm'] = df_filtered_rare['sale_price']
    #compute global correlation
    x = df_filtered_rare['rarity_rank']
    y = df_filtered_rare['sale_price_norm']
    correlation, p_value = pearsonr(x, y)
    return correlation, p_value

        
    
def compute_all_corrs(total_df):
    #deep copy total_df
    total_df = total_df.copy()
    command = "select distinct slug from slug_to_type where type='pfps'"
    rows = psql.execute_commands([command])
    #Get slug filters: pfps, non_pfps and cases where visual has better r2
    pfp_rows = [x[0] for x in rows]
    non_pfp_rows = [x for x in total_df['Collection'].unique() if x not in pfp_rows]
    #process dfs
    total_df = total_df.groupby('Collection').filter(lambda group: len(group) >= 100)
    #Keep only rows with rarity ranks
    rares = total_df.dropna(subset=['rarity_rank'],inplace=False).groupby('Collection').filter(lambda group: len(group) >= 100)
    #Keep only rows without raarity ranks
    no_rares = total_df[total_df['rarity_rank'].isna()].groupby('Collection').filter(lambda group: len(group) >= 100)
    param_list = [(rares,'rarity_rank',None),(rares,'rarity_rank',pfp_rows),(rares,'rarity_rank',non_pfp_rows),(total_df,'distance',None),(rares,'distance',None),(no_rares,'distance',None),(total_df,'distance',pfp_rows),(total_df,'distance',non_pfp_rows)]
    result_list = []
    for param in param_list:
        grouped = param[0].groupby('Collection')
        result = compute_labels_pearson(grouped,param[1],label_filt=param[2],metric='pearson')
        result_list.append(result)
    #make result df
    result_df = pd.DataFrame(result_list,columns=['column','metric','Correlations','Pos Correlations','Neg Correlations','Total'])
    return result_df

def compute_labels_pearson(grouped,column,metric='pearson',label_filt=None,slug_list = False):
    pos_count = 0
    neg_count = 0
    total_count = 0
    correlations = []
    slug_list = []
    if label_filt is not None:
        rows = set(label_filt)
    else:
        rows = set(grouped.groups.keys())
    for label, group in grouped:
        if label in rows:
            total_count+=1
            x = group[column]
            y = group['sale_price']
            if metric == 'pearson':
                correlation, p_value = pearsonr(x, y)
            else:
                correlation, p_value = spearmanr(x, y)
            if p_value<0.05 and correlation>0:
                correlations.append(correlation)
                pos_count+=1
            elif p_value<0.05 and correlation<0:
                neg_count+=1
                slug_list.append(label)
    if slug_list:
        return slug_list
    return column,metric,correlations,pos_count,neg_count,total_count

def compare_r2(total_df,return_slugs=False):
    total_df = total_df.copy()
    total_df = total_df.groupby('Collection').filter(lambda group: len(group) >= 100)
    #Keep only rows with rarity ranks
    df_filtered_rare = total_df.dropna(subset=['rarity_rank'],inplace=False).groupby('Collection').filter(lambda group: len(group) >= 100)
    grouped = df_filtered_rare.groupby('Collection')
    count_dist = 0
    count_rare = 0
    rare_r2s = []
    dist_r2s = []
    dist_slugs = []
    for label, group in grouped:
        x_1 = -np.asarray(group['rarity_rank'])
        x_1 = x_1.reshape(-1, 1)
        x_2 = np.asarray(group['distance'])
        x_2 = x_2.reshape(-1, 1)
        y = np.asarray(group['sale_price'])
        y = y.reshape(-1, 1)
        regr1 = linear_model.LinearRegression(positive=True)
        regr1.fit(x_1, y)
        rarity_pred = regr1.predict(x_1)
        # Create linear regression object for rarity
        regr2 = linear_model.LinearRegression(positive=True)
        regr2.fit(x_2, y)
        visual_pred = regr2.predict(x_2)
        # Compute correlation coefficient and p-value
        rare_r2 = r2_score(y,rarity_pred)
        dist_r2= r2_score(y,visual_pred)
        if max(rare_r2,dist_r2) ==0:
            continue
        if rare_r2>dist_r2:
            count_rare+=1
            rare_r2s.append(rare_r2)
        else:
            count_dist+=1
            dist_slugs.append(label)
            dist_r2s.append(dist_r2)
    if return_slugs:
        return dist_slugs
    else:
        return (count_rare,count_dist,np.mean(rare_r2s),np.mean(dist_r2s))

def compare_spearman(total_df):
    total_df = total_df.copy()
    total_df = total_df.groupby('Collection').filter(lambda group: len(group) >= 100)
    df_filtered_rare = total_df.dropna(subset=['rarity_rank'],inplace=False).groupby('Collection').filter(lambda group: len(group) >= 100)
    grouped = df_filtered_rare.groupby('Collection')
    dist_slugs = compare_r2(total_df,return_slugs=True)
    # Create a scatter plot using Matplotlib
    count=0
    count_dist = 0
    count_rare = 0
    count_dist_neg = 0
    count_rare_neg = 0
    for label, group in tqdm(grouped):
        if label in dist_slugs:
            x_1 = group['rarity_rank']
            x_2 = group['distance']
            y = group['sale_price']
            dof = dof = len(x_1)-2
            def statistic(x):  # explore all possible pairings by permuting `x`
                rs = stats.spearmanr(x, y).statistic  # ignore pvalue
                transformed = rs * np.sqrt(dof / ((rs+1.0)*(1.0-rs)))
                return transformed
            count+=1
            correlation, p_value = spearmanr(x_1, y)
            correlation_2, p_value_2 = spearmanr(x_2, y)
            p_value = stats.permutation_test((x_1,), statistic, alternative='less',permutation_type='pairings').pvalue
            p_value_2 = stats.permutation_test((x_2,), statistic, alternative='greater',permutation_type='pairings').pvalue
            if np.abs(correlation)>np.abs(correlation_2) and p_value<0.05:
                if correlation<-0:
                    print(f"(Rarity) Collection: {label}, Correlation: {correlation}, P-Value:{p_value}")
                    count_rare_neg+=1
                else:
                    print(f"(Rarity) Collection: {label}, Correlation: {correlation}, P-Value:{p_value}")
                    count_rare+=1
            elif p_value_2<0.05:
                if correlation_2<0:
                    count_dist_neg+=1
                else:
                    count_dist+=1
    print(count)
    print(f"Rare: {count_rare}, Rare Neg: {count_rare_neg}, Dist: {count_dist}, Dist Neg:{count_dist_neg}")

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

#----------------- Case Study Methods -----------------

def plot_all_rarity(df,num_quants=20,y_axis='sale_price'):
    df_grouped = df.groupby('Collection')
    plt.figure(figsize=(10, 6))
    for name,group in df_grouped:
        group['rarity_bin'] = pd.qcut(group['rarity_rank'], num_quants,labels=False)
        # Group by the bins and calculate the average 'saleprice' in each bin
        average_sale_price_in_bins = group.groupby('rarity_bin')[y_axis].mean()
        # Plot the average sale prices
        
        average_sale_price_in_bins.plot(kind='line',label=name)
        plt.xlabel('Rarity Bin')
        plt.ylabel('Average Sale Price')
        plt.xticks(rotation=45)
        plt.legend()
    plt.semilogy()
    plt.show()

def plot_one_rarity(df,slug,num_quants=20,y_axis='sale_price'):
    group = df[df['Collection'] == slug].copy()
    group['rarity_bin'] = pd.qcut(df['rarity_rank'], num_quants,labels=False)
    group['rarity_bin'] = group['rarity_bin'].max() - group['rarity_bin']
    # Group by the bins and calculate the average 'saleprice' in each bin
    average_sale_price_in_bins = group.groupby('rarity_bin')[y_axis].mean()
    average_sale_price_in_bins = average_sale_price_in_bins.sort_index(ascending=False)
    # Plot the average sale prices
    plt.figure(figsize=(10, 6))
    average_sale_price_in_bins.plot(kind='line')
    plt.xlabel('Rarity Bin')
    plt.ylabel('Average Sale Price')
    plt.xticks(rotation=45)
    plt.show()

def plot_all_distance(df,num_quants=20,y_axis='sale_price'):
    df_grouped = df.groupby('Collection')
    plt.figure(figsize=(10, 6))
    for name,group in df_grouped:
        group['distance_bin'] = pd.qcut(group['distance'], num_quants)
        # Group by the bins and calculate the average 'saleprice' in each bin
        average_sale_price_in_bins = group.groupby('distance_bin')[y_axis].mean()
        # Plot the average sale prices
        average_sale_price_in_bins.plot(kind='line',label=name)
        plt.xlabel('Distance Bin')
        plt.ylabel('Average Sale Price')
        plt.xticks(rotation=45)
        plt.legend()
    plt.show()

def plot_one_distance(df,slug,num_quants=20,y_axis='sale_price'):
    df = df.query("Collection == @slug")
    df['distance_bin'] = pd.qcut(df['distance'], num_quants)
    # Group by the bins and calculate the average 'saleprice' in each bin
    average_sale_price_in_bins = df.groupby('distance_bin')[y_axis].mean()
    # Plot the average sale prices
    plt.figure(figsize=(10, 6))
    average_sale_price_in_bins.plot(kind='line')
    plt.xlabel('Distance Bin')
    plt.ylabel('Average Sale Price')
    plt.xticks(rotation=45)
    plt.show()


def compute_rarity_and_distance_bins(df, slug, num_quants, y_axis = 'sale_price', metric = 'mean'):
    group = df[df['Collection'] == slug].copy()
    group['rarity_bin'] = pd.qcut(group['rarity_rank'], num_quants,labels=False)
    group['rarity_bin'] = group['rarity_bin'].max() - group['rarity_bin']
    group['distance_bin'] = pd.qcut(group['distance'], num_quants,labels=False)
    if metric == 'mean':
        average_sale_price_in_rare_bins = group.groupby('rarity_bin')[y_axis].mean()
        average_sale_price_in_dist_bins = group.groupby('distance_bin')[y_axis].mean()
    elif metric == 'variance':
        average_sale_price_in_rare_bins = group.groupby('rarity_bin')[y_axis].var()
        average_sale_price_in_dist_bins = group.groupby('distance_bin')[y_axis].var()
    elif metric == 'maxmin':
        average_sale_price_in_rare_bins = group.groupby('rarity_bin')[y_axis].max() - group.groupby('rarity_bin')[y_axis].min()
        average_sale_price_in_dist_bins = group.groupby('distance_bin')[y_axis].max() - group.groupby('distance_bin')[y_axis].min()
    return average_sale_price_in_rare_bins, average_sale_price_in_dist_bins
    


def plot_one_distance_and_rarity(df,slug,num_quants=20,y_axis='sale_price',metric='mean'):
    average_sale_price_in_rare_bins, average_sale_price_in_dist_bins = compute_rarity_and_distance_bins(df, slug, num_quants, y_axis, metric)
    if y_axis == 'sale_price':
        y_label = 'Average Sale Price'
    else:
        y_label = 'Average Number of Sales'
    # Plot the average sale prices
    plt.figure(figsize=(10, 6))
    average_sale_price_in_rare_bins.plot(kind='line',label='Rarity')
    average_sale_price_in_dist_bins.plot(kind='line',label='Distance')
    plt.legend(fontsize=18, facecolor='white', edgecolor='none', frameon=True,framealpha=1)
    slug_name_map = {
        'boredapeyachtclub':'Bored Ape Yacht Club',
        'cool-cats-nft':'Cool Cats',
        'proof-moonbirds': 'Moonbirds',
        'beanzofficial':'BEANZ Official',
        'pudgypenguins':'Pudgy Penguins',
        'doodles-official':'Doodles',
        'azuki':'Azuki',
        'clonex':'Clone X',
        'mutant-ape-yacht-club':'Mutant Ape Yacht Club',
    }
    plt.title(slug_name_map[slug], fontsize=18)
    # not bold font
    plt.xlabel('Bin Number', fontsize=20, fontweight='normal')
    plt.ylabel(y_label, fontsize=15, fontweight='normal')
    # plt.xticks(rotation=45)
    # plt.xticks(range(1, num_quants + 1))
    # plt.xlim(1, num_quants)  # Adjust x-axis limits to ensure 0 is at the y-axis
    # plt.ylim(ymin=0)
    # plt.show()
    plt.xticks(range(num_quants), range(1, num_quants + 1))  # Set custom labels for x-ticks
    plt.xlim(0, num_quants)  # Adjust x-axis limits to ensure 0 is at the y-axis
    plt.ylim(ymin=0)
    plt.savefig(f'figures/{slug}_distance_rarity_{metric}_{y_axis}.pdf')
    plt.show()

def plot_distance_and_rarity(group, slug, num_quants, y_axis, metric, ax):
    average_sale_price_in_rare_bins, average_sale_price_in_dist_bins = compute_rarity_and_distance_bins(group, slug, num_quants, y_axis, metric)
    if y_axis == 'sale_price':
        y_label = 'Average Sale Price'
    else:
        y_label = 'Average Number of Sales'
    # Plot the average sale prices
    average_sale_price_in_rare_bins.plot(kind='line', label='Rarity', ax=ax)
    average_sale_price_in_dist_bins.plot(kind='line', label='Distance', ax=ax)
    ax.legend(fontsize=18, facecolor='white', edgecolor='none', frameon=True, framealpha=1)
    
    slug_name_map = {
        'boredapeyachtclub': 'Bored Ape Yacht Club',
        'cool-cats-nft': 'Cool Cats',
        'proof-moonbirds': 'Moonbirds',
        'beanzofficial': 'BEANZ Official',
        'pudgypenguins': 'Pudgy Penguins',
        'doodles-official': 'Doodles',
        'azuki': 'Azuki',
        'clonex': 'Clone X',
        'mutant-ape-yacht-club': 'Mutant Ape Yacht Club',
    }
    ax.set_title(slug_name_map.get(slug, slug), fontsize=18)
    ax.set_xlabel('Bin Number', fontsize=20, fontweight='normal')
    ax.set_ylabel(y_label, fontsize=15, fontweight='normal')
    ax.set_xticks([0, 4, 9, 14, 19], [1, 5, 10, 15, 20])
    # ax.set_xticklabels(range(1, num_quants + 1, 5))  # Set custom labels for x-ticks
    ax.set_xlim(0, num_quants)  # Adjust x-axis limits to ensure 0 is at the y-axis
    ax.set_ylim(ymin=0)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))  # Ensure y-ticks are integers

def plot_all_slugs(df, num_quants, y_axis, metric):
    slugs = sorted(df['Collection'].unique())
    num_plots = len(slugs) - 1  # Exclude 'meebits'
    rows = 3
    cols = 3

    fig, axes = plt.subplots(rows, cols, figsize=(15, 15))
    axes = axes.flatten()

    plot_index = 0
    for slug in slugs:
        if slug == 'meebits':
            continue
        ax = axes[plot_index]
        plot_distance_and_rarity(df[df['Collection'] == slug], slug, num_quants, y_axis, metric, ax)
        plot_index += 1

    # Hide any unused subplots
    for i in range(plot_index, rows * cols):
        fig.delaxes(axes[i])

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.5)  # Increase vertical space between subplots
    plt.savefig(f'figures/all_slugs_distance_rarity_{metric}_{y_axis}.pdf')
    plt.show()

def make_case_study_result_df(df):
    rows = []
    for name,group in df.groupby('Collection'):
        result = {}
        distance = group['distance']
        rarity = group['rarity_rank']
        price = group['sale_price']
        count = group['count']
        result['Slug'] = name
        #Price vs rarity and distance 
        correlation_1, p_value_1 = pearsonr(distance, price)
        correlation_2, p_value_2 = pearsonr(rarity, price)
        spear_1, s_p_value_1 = spearmanr(distance, price)
        spear_2, s_p_value_2 = spearmanr(rarity, price)
        result['Distance Correlation'] = correlation_1
        result['Rarity Correlation'] = correlation_2
        result['Distance Spearman'] = spear_1
        result['Rarity Spearman'] = spear_2
        result['Distance P Value'] = p_value_1
        result['Rarity P Value'] = p_value_2
        result['Distance Spearman P Value'] = s_p_value_1
        result['Rarity Spearman P Value'] = s_p_value_2
        #Count vs rarity and distance
        correlation_1, p_value_1 = pearsonr(distance, count)
        correlation_2, p_value_2 = pearsonr(rarity, count)
        spear_1, s_p_value_1 = spearmanr(distance, count)
        spear_2, s_p_value_2 = spearmanr(rarity, count)
        result['Distance Correlation (Count)'] = correlation_1
        result['Rarity Correlation (Count)'] = correlation_2
        result['Distance Spearman (Count)'] = spear_1
        result['Rarity Spearman (Count)'] = spear_2
        result['Distance P Value (Count)'] = p_value_1
        result['Rarity P Value (Count)'] = p_value_2
        result['Distance Spearman P Value (Count)'] = s_p_value_1
        result['Rarity Spearman P Value (Count)'] = s_p_value_2
        #Correlation between rarity and distance
        dist_rare_corr, dist_rare_p = pearsonr(distance, rarity)
        dist_rare_spear, dist_rare_spear_p = spearmanr(distance, rarity)
        result['Distance Rarity Correlation'] = dist_rare_corr
        result['Distance Rarity P Value'] = dist_rare_p
        
        # Filter the DataFrame to only include rows where 'rarity_rank' is less than or equal to the 90th percentile
        #Important to remember that rarity needs to bottom end filtered
        percentile_10 = group['rarity_rank'].quantile(0.1)
        group_rare_filt = group[group['rarity_rank'] >= percentile_10]
        rarity = group_rare_filt['rarity_rank']
        distance = group_rare_filt['distance']
        price = group_rare_filt['sale_price']
        correlation_2, p_value_2 = pearsonr(rarity, price)
        spear_2, s_p_value_2 = spearmanr(rarity, price)
        result['Filt Rarity Correlation'] = correlation_2
        result['Filt Rarity P Value'] = p_value_2
        result['Filt Rarity Spearman'] = spear_2
        result['Filt Rarity Spearman P Value'] = s_p_value_2
        
        # Filter the DataFrame to only include rows where 'distance' is less than or equal to the 90th percentile
        percentile_90 = group['distance'].quantile(0.9)
        group = group[group['distance'] <= percentile_90]
        group_rare_filt = group[group['distance'] <= percentile_90]
        rarity = group_rare_filt['rarity_rank']
        distance = group_rare_filt['distance']
        price = group_rare_filt['sale_price']
        correlation_1, p_value_1 = pearsonr(distance, price)
        spear_1, s_p_value_1 = spearmanr(distance, price)
        result['Filt Distance Correlation'] = correlation_1
        result['Filt Distance P Value'] = p_value_1
        result['Filt Distance Spearman'] = spear_1
        result['Filt Distance Spearman P Value'] = s_p_value_1
        rows.append(result)
    result_df = pd.DataFrame(rows)
    return result_df
        