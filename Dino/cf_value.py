import sys
sys.path.append("..")
import data_retrieval.opensea_methods as opse
import data_retrieval.psql_methods as psql
import pandas as pd
import counterfeit_utils as cfu
import pickle
import logging
from scipy import stats

#For all slugs for which we dont already have time-series data this retrieves it using Openseas API 
def retreive_full_time_series(slugs,table_name='cf_sales',log_file='full_time_series.log'):
    #set up logging
    logging.basicConfig(filename=log_file, level=logging.INFO)
    command = f"select distinct slug from {table_name}"
    rows = psql.execute_commands([command])
    done_slugs = [row[0] for row in rows]
    slugs_left = [x for x in slugs if x not in done_slugs]
    logging.info(f"Slugs left: {len(slugs_left)}")
    sales = []
    command = f"Insert into {table_name} (slug,token_id,seller,buyer,timestamp,sale_price,payment_token,transaction) Values (%s,%s,%s,%s,%s,%s,%s,%s)"
    for slug in slugs_left:
        try:
            sales = opse.pull_sales_data(collection_slug=slug)
            psql.batch_insert(command,sales)
            logging.info(f"Completed {slug}")
        except:
            logging.info(f"Failed to pull sales data for {slug}")
            continue

def build_ko_rankings():
    pw_dists = pd.read_pickle('pw_dists_counterfeit.pkl')
    with open("../Graph_predictions/dataset_stor/graph_dataset_4/total_stats.pkl",'rb') as f:
        total_stats = pickle.load(f)
    command = "SELECT * from collection_total_stats"
    temp_stats_tuples = psql.execute_commands([command])
    new_df = pd.DataFrame({'Top_100':pw_dists['Top_100'],'Alt': pw_dists['Alt'], 'Distance': pw_dists['Euc_Distance'],'sale_price': pw_dists['Alt'].map(lambda x: total_stats.get(x,[None,None,None])[2])})
    columns = ['volume','sales', 'avg_price','num_owners','market_cap','floor','symbol','Alt']
    stats_df = pd.DataFrame(temp_stats_tuples,columns=columns)
    result_df = pd.merge(new_df, stats_df[['Alt', 'volume']], on='Alt', how='left')
    # Keep the rows with the 100 smallest 'Euc_Distance' for each 'Top_100'
    result_df = add_date_and_volume(result_df,stats_df)
    result_df['Top_ratio'] = result_df['volume']/result_df['Top_100_volume']
    result_filt = result_df.sort_values(by='Top_ratio',ascending=False).query('Alt_date>Top_100_date')
    return result_filt

def split_ders(result_df):
    der_lists = cfu.der_list_from_db()
    no_ders = result_df[~result_df.apply(lambda row: (row['Top_100'], row['Alt']) in der_lists, axis=1)]
    no_ders = no_ders.groupby('Top_100', group_keys=False).apply(keep_knocks)
    with_ders = result_df[result_df.apply(lambda row: (row['Top_100'], row['Alt']) in der_lists, axis=1)]
    return no_ders,with_ders
    
def keep_knocks(group):
    try:
        slug = group['Top_100'].iloc[0]
        command = f"select num from objective_cf_num where slug='{slug}' and type='pfps'"
        rows = psql.execute_commands([command])
        n_smallest = rows[0][0]
        return group.nsmallest(n_smallest,'Distance')
    except:
        return
        


def keep_top(group,n_smallest=2):
    try:
        return group.nlargest(n_smallest,'Top_ratio')
    except:
        return

def filter_one_top(result_df,num_top=2):
    # Keep the rows with the 100 smallest 'Euc_Distance' for each 'Top_100'
    tops = result_df.groupby('Top_100', group_keys=False).apply(keep_top,n_smallest=2)
    tops = tops.sort_values(by='Top_100_volume',ascending=False)
    return tops
    
def add_date_and_volume(result_df,stats_df):
    unique_alts = result_df['Alt'].unique()
    alt_date_map = {}
    for alt in unique_alts:
        try:
            alt_date_map[alt] = cfu.creation_sec_from_db(alt)
        except:
            continue
    unique_top_100s = result_df['Top_100'].unique()
    top_100_map = {}
    for top_100 in unique_top_100s:
        try:
            top_100_map[top_100] = stats_df.query(f"Alt=='{top_100}'")['volume'].iloc[0]
        except:
            continue
    top_100_date_map = {}
    for top_100 in unique_top_100s:
        try:
            top_100_date_map[top_100] = cfu.creation_sec_from_db(top_100)
        except:
            continue
    result_df['Top_100_volume'] = result_df['Top_100'].map(top_100_map)
    result_df['Alt_date'] = result_df['Alt'].map(alt_date_map)
    result_df['Top_100_date'] = result_df['Top_100'].map(top_100_date_map)
    return result_df

def all_sales_from_db():
    sales = sales_from_db()
    day_sales = cfu.timestamps_to_dates(sales)
    df_sales = pd.DataFrame(day_sales,columns=['day','price','slug'])
    df_sales['day'] = pd.to_datetime(df_sales['day'])
    return df_sales
    
def sales_from_db(slug=None):
    if slug:
        command = f"select timestamp,sale_price from cf_sales where slug='{slug}'"
    else:
        command = f"select timestamp,sale_price,slug from cf_sales"
    data = psql.execute_commands([command])
    return data

def get_sale_slugs():
    command = f"select distinct slug from cf_sales"
    data = psql.execute_commands([command])
    slugs = [x[0] for x in data]
    return slugs
    
def day_sales_from_db(slug):
    sales = sales_from_db(slug)
    day_sales = cfu.timestamps_to_dates(sales)
    df_sales = pd.DataFrame(day_sales,columns=['day','price'])
    df_sales['day'] = pd.to_datetime(df_sales['day'])
    return df_sales

def calculate_percentile(row):
    return stats.percentileofscore(row['mean_samples'], row['mean_data'])

def show_cf_val_results():
    result_df = pd.DataFrame()
    for res_id in range(1,270):
        outpath = f'val_comp/result_df' + str(res_id) + '.pkl'
        try:
            df = pd.read_pickle(outpath)
            result_df = pd.concat([result_df,df])
        except:
            print(res_id)
            continue
    #iterate through rows of result df
    result_df['percentile'] = result_df.apply(calculate_percentile, axis=1)
    return result_df