import pandas as pd
import numpy as np
import sys
sys.path.append("..")
import data_retrieval.opensea_methods as opse
import os
import matplotlib.pyplot as plt
import data_retrieval.psql_methods as psql
import plotly.express as px
import seaborn as sns
from datetime import datetime
from tqdm import tqdm
import random
from matplotlib.cm import get_cmap
from scipy.signal import medfilt
from sklearn.neighbors import KernelDensity
# plt.style.use('https://raw.githubusercontent.com/gregdeon/plots/main/style.mplstyle')
# plt.style.use('fivethirtyeight')
# plt.matplotlib.rcParams['figure.dpi'] = 300
# plt.matplotlib.rcParams['font.size'] = 6

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager
mpl.rcParams.update(mpl.rcParamsDefault)
# plt.style.use('narunraman.mplstyle')

# font_path = 'cmunss.ttf'  # Your font path goes here
# font_manager.fontManager.addfont(font_path)
# prop = font_manager.FontProperties(fname=font_path)

# plt.rcParams['font.family'] = 'sans-serif'
# plt.rcParams['font.sans-serif'] = prop.get_name()
#CONSTANTS
DB_NAME = 'objective_cf_num'
CUT_OFF = 5
# pw_dists = pd.read_pickle('pw_dists_counterfeit.pkl')
def get_dists():
    #get local file path
    path = os.path.abspath(os.path.dirname(__file__))
    pw_dists = pd.read_pickle(f'{path}/pw_dists_counterfeit.pkl')
    return pw_dists
    
def get_slug_owners(slug):
    command  = f"Select distinct address from slug_to_token where slug='{slug}'"
    data = psql.execute_commands([command])
    addresses = tuple([x[0] for x in data])
    command = f"Select * from slug_to_token where address in {addresses}"
    data = psql.execute_commands([command])
    columns = ['slug','address','token_id']
    df = pd.DataFrame(data,columns=columns)
    return df

def get_all_owners(slugs = None):
    if slugs:
        slugs = tuple(slugs)
        command = f"Select * from slug_to_token where slug in {slugs}"
    else:
        command = f"Select * from slug_to_token"
    data = psql.execute_commands([command])
    columns = ['slug','address','token_id']
    df = pd.DataFrame(data,columns=columns)
    del data
    return df

def get_overlaps(top_slug):
    df = get_slug_owners(top_slug)
    pw_dists = get_dists()
    pw_dists_no_dupe = pw_dists.query('Top_100!=Alt')
    top_df = pw_dists_no_dupe.query(f"Top_100=='{top_slug}'")
    sorted_df = top_df.sort_values(by='Euc_Distance').reset_index(drop=True)
    sorted_df['sorted_order'] = sorted_df.index + 1
    sorted_df = sorted_df.rename(columns={'Alt': 'slug'})
    # Get the sorted labels as a list
    result_df = sorted_df[['slug', 'sorted_order']]
    merged_df = pd.merge(df,result_df,  on='slug',how='right')
    sorted_merged_df = merged_df.sort_values(by='sorted_order')
    return sorted_merged_df

def count_overlaps(top_slug,filter=None):
    df = get_overlaps(top_slug)
    df = df.query("address!='0x000000000000000000000000000000000000dEaD'")
    if filter=='one_token':
        df = df.drop_duplicates(subset=['slug','address'])
    elif filter=='owners':
        df = df.drop_duplicates(subset=['address'])
    row_counts = df.groupby(['slug', 'sorted_order'])['address'].count().reset_index(name='row_count').sort_values(by='sorted_order')
    return row_counts
    
def own_list_from_db(slug):
    command = f"Select * from owner_lists where slug='{slug}'"
    data = psql.execute_commands([command])
    # Define column names
    columns = ['Alt', 'wallet', 'Top_100']
    # Create a DataFrame
    pulled_data = pd.DataFrame(data, columns=columns)
    result = pulled_data.groupby('wallet')['Alt'].agg(list).reset_index()
    result_list = result.values.tolist()
    own_slugs = [x[1] for x in result_list]
    return own_slugs


def get_counterfeit_db(slug=None,db_name='slug_to_counterfeit_num'):
    # db_args = [(slug, num, counterfeit_slug)]
    if slug:
        command = f"SELECT num FROM {db_name} where slug='{slug}'"
    else:
        command = f"SELECT slug,num FROM {db_name} where type='pfps'"
    data = psql.execute_commands([command])
    return data

def der_list_from_db(slug=None):
    if slug == None:
        command = f"Select * from slug_to_der"
        data = psql.execute_commands([command])
        return data
    command = f"Select der from slug_to_der where slug='{slug}'"
    data = psql.execute_commands([command])
    der_list = [x[0] for x in data]
    return der_list

def plot_price_chart(slug):
    date_price = sales_from_db(slug)
    date_price = sorted(date_price)
    min_date = min(date_price, key=lambda x: x[0])[0]
    date_price_zero = [(x[0]-min_date,x[1]) for x in date_price]
    x,y = zip(*date_price_zero)
    plt.plot(x,y)
    plt.show()

def plot_price_chart_smooth(slug,ko_list = [],date_list=[], filename='',pretty_slug=None):
    if not pretty_slug:
        pretty_slug = slug
    date_price = sales_from_db(slug)
    date_price = sorted(date_price)
    min_date = min(date_price, key=lambda x: x[0])[0]
    date_price_zero = [(x[0]-min_date,x[1]) for x in date_price if x[1] is not None]
    x,y = zip(*date_price_zero)
    x = [j/86400 for j in x]
    window_size = 15
    y_values_smoothed = np.convolve(y, np.ones(window_size)/window_size, mode='valid')
    y_values_smoothed = medfilt(y, kernel_size=window_size)

    # plt.figure(figsize=(8, 6))
    # ax = plt.gca()  # Get current axes to customize
    # if slug == 'goblintownwtf':
    #     slug = 'Goblin Town'
    plt.plot(x[:len(y_values_smoothed):15], y_values_smoothed[::15], linewidth=1.8)
    cmap = get_cmap('viridis')
    colors = cmap(np.linspace(0, 1, len(ko_list)))
    colors = get_cmap('Set1', len(ko_list))
    if date_list:
        ax2 = ax.twinx()
        secs = [(date-min_date)/86400 for date in date_list]
        ax2.hist(secs, bins=100, alpha=0.3, color='blue', label='Number of Sales')
    for ko,color in zip(ko_list,colors.colors):
        sec = (creation_sec_from_db(ko)-min_date)/86400
        plt.axvline(x=sec,color='r', linestyle='--',linewidth=1.5,label=ko)
    # plt.xlabel(f"Days Since {pretty_slug} Mint", fontsize=20)
    # plt.ylabel(f"{pretty_slug} Price in ETH", fontsize=20, fontweight='normal')
    plt.xticks(fontsize=20)
    plt.yticks(np.arange(0, 9, 2), fontsize=20)
    # plt.yticks([0, 2, 4, 6, 8])
    plt.ylim(bottom=0,top=10)
    # plt.grid(False)
    # plt.xlim(left=1,right=800)
    # plt.xlim(left=150,right=250)
    # plt.savefig(f'../paper_plots/{filename}.png', facecolor='white', edgecolor='none')
    # plt.savefig(f'../paper_plots/{filename}.pdf', format="pdf", bbox_inches="tight", facecolor='white', edgecolor='none')
    # plt.legend()
    # plt.show()

def timestamps_to_dates(date_prices,raw_timestamps=False):
    if raw_timestamps:
        day_prices = [datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d') for timestamp in date_prices]
    else:
        if len(date_prices[0])==3:
            day_prices = [(datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d'), data,slug) for timestamp, data,slug in date_prices]
        else:
            day_prices = [(datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d'), data) for timestamp, data in date_prices]
    return day_prices
    




def creation_sec_from_db(slug):
    command = f"select creation_date from slug_to_creation_date where slug='{slug}'"
    data = psql.execute_commands([command])
    date_string = data[0][0]
    parsed_date = datetime.strptime(date_string, "%Y-%m-%d")
    # Convert to timestamp (seconds since epoch)
    timestamp_seconds = int(parsed_date.timestamp())
    return timestamp_seconds

def creation_date_from_db(slug):
    command = f"select creation_date from slug_to_creation_date where slug='{slug}'"
    data = psql.execute_commands([command])
    date_string = data[0][0]
    return date_string

def volume_from_db(slug):
    command = f"select volume from collection_total_stats where slug='{slug}'"
    data = psql.execute_commands([command])
    try:
        volume = data[0][0]
    except:
        print("No volume data for slug")
        volume=0
    return volume

def look_sim_volume_from_db(look_sim):
    command = f"SELECT sum(sale_price) from look_sim_sales where slug = {look_sim} and (payment_token='WETH' or payment_token='ETH')"
    data = psql.execute_commands([command])
    try:
        volume = data[0][0]
    except:
        print("No volume data for slug")
        volume=0
    return volume

def find_earliest_ownership_date(wallet,slug):
    command = f"select token_id from slug_to_token where address='{wallet}' and slug='{slug}'"
    data = psql.execute_commands([command])
    tokens = [x[0] for x in data]
    command = f"select address from collectiontoaddress where slug='{slug}'"
    data = psql.execute_commands([command])
    contract = data[0][0]
    timestamp = 100000000000000
    for token in tokens:
        events = opse.pull_nft_events(contract,token)
        if events:
            for event in events:
                if event[3]==wallet.lower() and event[4]<timestamp:
                    timestamp=event[4]
    return timestamp

def store_ownership_dates(top_slug,db_name='objective_cf_num',destination='ownership_dates'):
    command = f"Select num from {db_name} where slug='{top_slug}'"
    data = psql.execute_commands([command])
    num_cf =data[0][0]
    command = "Insert into {destination} (wallet, slug, timestamp) values (%s, %s, %s)"
    drop_dead = get_overlaps(top_slug).query("address!='0x000000000000000000000000000000000000dEaD' and sorted_order<={num_cf}")
    peeps_to_check = drop_dead[['slug', 'address']].drop_duplicates()
    og_wallets = drop_dead[['address']].drop_duplicates()['address'].to_list()
    dates = []
    for address in tqdm(og_wallets):
        timestamp = find_earliest_ownership_date(address,top_slug)
        dates.append((address,top_slug,timestamp))
    psql.batch_insert(command,dates)
    dates = []
    for index, row in tqdm(peeps_to_check.iterrows()):
        timestamp1 = find_earliest_ownership_date(row['address'],row['slug'])
        dates.append((row['address'],row['slug'],timestamp1))
    psql.batch_insert(command,dates)

def get_all_ownershipdates():
    command = "Select * from ownership_dates"
    data = psql.execute_commands([command])
    columns = ['wallet','slug','timestamp']
    df = pd.DataFrame(data,columns=columns)
    return df
    


def compute_interval(interval_days,start,df,symetric=False):
    # Filter the DataFrame for the specified date and interval
    target_date = pd.to_datetime(start)
    selected_data = df[(df['day'] >= target_date) & (df['day'] < target_date + pd.Timedelta(days=interval_days))]
    # Calculate the average value for the selected period
    av_forward = selected_data['price'].median()
    if symetric:
        selected_data = df[(df['day'] < target_date) & (df['day'] > target_date - pd.Timedelta(days=interval_days+1))]
    else:
        selected_data = df[(df['day'] < target_date) & (df['day'] > target_date - pd.Timedelta(days=3))]
    # Calculate the average value for the selected period
    av_backward = selected_data['price'].median()
    return (av_backward,av_forward)

def find_cf_days(top_slug,db_name,remove_ders = True):
    command = f"Select num from {db_name} where slug='{top_slug}'"
    data = psql.execute_commands([command])
    num_cf =data[0][0]
    df = get_overlaps(top_slug).query(f"address!='0x000000000000000000000000000000000000dEaD' and sorted_order<={num_cf}")
    if remove_ders:
        der_list = der_list_from_db(top_slug)
        df = df.query(f"slug not in {der_list}")
    slugs = df[['slug']].drop_duplicates()['slug'].to_list()
    intervals = []
    for slug in slugs:
        try:
            date =  creation_date_from_db(slug)
            volume = volume_from_db(slug)
            intervals.append((slug,volume,date))
        except:
            print(f'No creation data for slug {slug}')
    return intervals
    
def compute_all_intervals(top_slug,interval,db_name,remove_ders = True):
    command = f"Select num from {db_name} where slug='{top_slug}'"
    data = psql.execute_commands([command])
    num_cf =data[0][0]
    # df = get_overlaps(top_slug).query(f"address!='0x000000000000000000000000000000000000dEaD' and sorted_order<={num_cf}")
    # if remove_ders:
    #     der_list = der_list_from_db(top_slug)
    #     df = df.query(f"slug not in {der_list}")
    #Remove code above after testing
    slugs = get_look_sims(top_slug,remove_ders=remove_ders)
    sale_df = day_sales_from_db(top_slug)
    intervals = []
    for slug in slugs:
        try:
            date =  creation_date_from_db(slug)
            volume = volume_from_db(slug)
            intervals.append((slug,volume,date,compute_interval(interval,date,sale_df)))
        except:
            print(f'No creation data for slug {slug}')
    return (intervals,num_cf)

def sample_intervals(top_slug,start,stop,num_intervals,interval,num_samples=1):
    df = day_sales_from_db(top_slug)
    start_date = pd.to_datetime(start)
    stop_date = pd.to_datetime(stop)
    df = df[(df['day'] < stop_date) & (df['day'] > start_date)]
    tot_intervals = []
    for x in range(0,num_samples):
        intervals = []
        for x in range(0,num_intervals):
            unique_values = df['day'].unique()
            # Select a random value from the unique values
            try:
                random_date = random.choice(unique_values)
            except:
                continue
            intervals.append((random_date,compute_interval(interval,random_date,df)))
        tot_intervals.append(intervals)
    return tot_intervals

def day_sales_from_db(slug):
    sales = sales_from_db(slug)
    day_sales = timestamps_to_dates(sales)
    df_sales = pd.DataFrame(day_sales,columns=['day','price'])
    df_sales['day'] = pd.to_datetime(df_sales['day'])
    return df_sales

def sales_from_db(slug=None):
    if slug:
        command = f"select timestamp,sale_price from cf_sales where slug='{slug}' and (payment_token='WETH' or payment_token='ETH')"
    else:
        command = f"select timestamp,sale_price,slug from cf_sales where (payment_token='WETH' or payment_token='ETH')"
    data = psql.execute_commands([command])
    return data

def sample_intervals_kde(top_slug,intervals,window_size,num_samples=1,bandwidth=10):
    tot_intervals = []
    df = day_sales_from_db(top_slug)
    unique_values = np.array(sorted(df['day'].unique()))
    ko_days = [pd.to_datetime(x[2]) for x in intervals]
    indices = []
    print("Number of KO Days: ",len(ko_days))
    print("Number of Unique Values: ",len(unique_values))
    for specific_datetime in ko_days:
        loc = np.where(unique_values == specific_datetime)[0]
        if loc:
            index = loc[0]
            indices.append(index)
    # Creating and fitting the KDE
    print(indices)
    if bandwidth:
        bw_scale = np.power(len(indices), -1/5)
        bw = bandwidth * bw_scale
        kde = KernelDensity(bandwidth=bw,kernel='gaussian')
        data = np.array(indices).reshape(-1, 1)
        if len(indices)<5:
            return None
        kde.fit(data)  # KDE expects data in 2D array format
    # Sampling from the KDE
    for x in range(0,num_samples):
        if bandwidth is None:
            samples = np.random.uniform(0, len(unique_values)-1, len(ko_days))
            round_samples = [round(x) for x in samples]
        else:
            samples = kde.sample(len(ko_days))
            round_samples = [round(x[0]) for x in samples]
        rand_intervals = []
        for x in round_samples:
            x = min(x,len(unique_values)-1)
            x = max(x,0)
            random_date = unique_values[x]
            rand_intervals.append((random_date,compute_interval(window_size,random_date,df)))
        tot_intervals.append(rand_intervals)
    return tot_intervals

def get_all_value_comparisons(interval_length,num_intervals,num_samples,cut_off,db_name='objective_cf_num',plot=False,bandwidth=10,logger=None):
    top_slugs = get_top_slugs(cut_off,db_name)
    results = []
    for top_slug in tqdm(top_slugs): 
        result = value_comparison(top_slug,interval_length,num_intervals,num_samples,cut_off,db_name,plot,bandwidth,logger)
        if result:
            results.append(result)
    return results

def value_comparison(slug,interval_length,num_samples,db_name='objective_cf_num',bandwidth=10,logger=None):
    if logger:
            logger.info(slug)
    intervals,num_cf = compute_all_intervals(slug,interval_length,db_name)
    y = [j[3][1]-j[3][0] for j in intervals]
    x = [j[1] for j in intervals]
    # sampled_intervals = sample_intervals(top_slug,min_date,max_date,num_cf,interval_length,num_samples = num_samples)
    sampled_intervals = sample_intervals_kde(slug,intervals,interval_length,num_samples = num_samples,bandwidth=bandwidth)
    if sampled_intervals is None:
        print(f"Skipping slug {slug} due to lack of sampled intervals")
        return None

    mean_samples = build_sample_dist(sampled_intervals)
    data = [x for x in y if not np.isnan(x)]
    mean_data = np.mean(data)
    return (slug,mean_samples,mean_data)
    
def plot_paired_cdfs(top_slug,sampled_intervals,real_intervals,interval_length,saved=True):
    data = [j[1][1]-j[1][0] for j in sampled_intervals[0]]
    fig, ax = plt.subplots()
    sns.ecdfplot(data, color='blue', ax=ax, label='Sampled Changes')
    sns.ecdfplot(real_intervals, color='orange', ax=ax, label='Changes at Imitations')
    data1 = [x for x in data if not np.isnan(x)]
    data2 = [x for x in real_intervals if not np.isnan(x)]
    mean_samples = build_sample_dist(sampled_intervals)
    mean_data2 = np.mean(data2)
    # Add vertical lines for the means
    ax2 = ax.twinx()
    sns.kdeplot(mean_samples,color='green', label='PDF Data', ax=ax2)
    plt.axvline(mean_data2, color='orange', linestyle='--', label='Counterfeit Average')
    ax.legend()
    plt.xlabel("Change in Median Price")
    plt.ylabel("Cumalitive Density")
    plt.title(f"Cumalitive Density Functions for {top_slug} and {interval_length} Wide Windows")
    if saved:
        plt.savefig(f"plots/cdfs/{top_slug}_cdfs.png")
    plt.close()
    return (top_slug,mean_samples,mean_data2)

def build_sample_dist(sampled_intervals):
    mean_samples= []
    for i,intervals in enumerate(sampled_intervals):
        data = [j[1][1]-j[1][0] for j in intervals]
        data1 = [x for x in data if not np.isnan(x)]
        mean_data1 = np.mean(data1)
        mean_samples.append(mean_data1)
    return mean_samples

"""
-------------- Code for retreiving top slugs and their look sims ----------------
"""
def get_top_slugs(cut_off,db_name):
    command = f"Select slug from {db_name} where num>={cut_off} and type='pfps'"
    data = psql.execute_commands([command])
    slugs = [x[0] for x in data]
    return slugs

def get_all_look_sims():
    top_slugs = get_top_slugs(CUT_OFF,DB_NAME)
    unique_look_sims = set()
    for slug in top_slugs:
        look_sims = get_look_sims(slug)
        unique_look_sims.update(look_sims)
    return unique_look_sims

def get_look_sims(top_slug,remove_ders=True):
    command = f"Select num from {DB_NAME} where slug='{top_slug}'"
    data = psql.execute_commands([command])
    num_cf =data[0][0]
    df = get_overlaps(top_slug).query(f"address!='0x000000000000000000000000000000000000dEaD' and sorted_order<={num_cf}")
    if remove_ders:
        der_list = der_list_from_db(top_slug)
        df = df.query(f"slug not in {der_list}")
    slugs = df[['slug']].drop_duplicates()['slug'].to_list()
    return slugs
