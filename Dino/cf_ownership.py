import counterfeit_utils as cfu
import pandas as pd
import sys
sys.path.append("..")
import data_retrieval.opensea_methods as opse
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import tqdm as tqdm
import data_retrieval.psql_methods as psql
import pickle


def generate_all_ownership_stats(db_name='objective_cf_num',cutoff=5):
    percentiles_total,percentiles_tokens = plot_all_overlap_cdfs(db_name,cutoff)
    #Adds number of counterfeits to the list of results
    with_nums_total = sorted([(x[0],x[1],cfu.get_counterfeit_db(slug=x[0],db_name='objective_cf_num')[0][0]) for x in percentiles_total])
    with_nums_tokens = sorted([(x[0],x[1],cfu.get_counterfeit_db(slug=x[0],db_name='objective_cf_num')[0][0]) for x in percentiles_tokens])
    #Computes averages
    avgs_total = [(x[0],((x[1] < x[2]).sum())/len(x[1]),(x[2]/10976)) for x in with_nums_total]
    avgs_tokens = [(x[0],((x[1] < x[2]).sum())/len(x[1]),(x[2]/10976)) for x in with_nums_tokens]
    #Stores result in Dataframe
    df_1 = pd.DataFrame(avgs_total,columns=['Collection','% Knockoffs','Expected % Knockoffs'])
    df_1['percentage'] = df_1['% Knockoffs']/df_1['Expected % Knockoffs']
    
    df_2 = pd.DataFrame(avgs_tokens,columns=['Collection','% Knockoffs','Expected % Knockoffs'])
    df_2['percentage'] = df_2['% Knockoffs']/df_2['Expected % Knockoffs']
    print(df_1['percentage'].mean())
    print(df_2['percentage'].mean())
    return df_1,df_2

def plot_all_overlap_cdfs(db_name='objective_cf_num',cutoff=5,xlim=100,ylim=0.04):
    der_lists = cfu.der_list_from_db()
    columns = ['slug','derivative']
    df1 = pd.DataFrame(der_lists,columns=columns)
    cf_nums = cfu.get_counterfeit_db(slug=None,db_name=db_name)
    columns = ['slug','cf_num']
    df2 = pd.DataFrame(cf_nums,columns=columns)
    merged_df = pd.merge(df1,df2, how='right',on='slug').query(f"cf_num>={cutoff}")
    grouped = merged_df.groupby('slug')
    percentiles_total = []
    percentiles_tokens = []
    for slug,group in grouped:
        print(slug)
        ders = group['derivative'].to_list()
        print(ders)
        percentiles_total.append((slug,compute_overlap_cdf(slug,der_list = ders,xlim=xlim,ylim=ylim,show=False,filter='')))
        percentiles_tokens.append((slug,compute_overlap_cdf(slug,der_list = ders,xlim=xlim,ylim=ylim,show=False,filter='one_token')))
    return percentiles_total,percentiles_tokens

#First finds overlaps using cf utils and then formats it for make overlap cdf
def compute_overlap_cdf(slug,der_list = None,xlim=200,ylim=0.02,show=True,filter=None):
    count_df = cfu.count_overlaps(slug,filter=filter)
    column_order = ['sorted_order', 'row_count', 'slug']
    new_df = count_df[column_order]
    list_of_tuples = [tuple(row) for row in new_df.itertuples(index=False)]
    return make_overlap_cdf(slug,list_of_tuples,der_list = der_list,xlim=xlim,ylim=ylim,show=False)

#Takes in a slug and its overlap df to produce a plot of the cdf of overlaps
def make_overlap_cdf(slug,count_to_overlap,der_list = None,xlim=2000,ylim=0.3,show=True,plot=False):
    #assumes overlap is sorted
    # count_to_overlap = sorted(count_to_overlap)
    count_to_overlap = [x for x in count_to_overlap if x[2] not in opse.SKIP_LIST]
    if der_list:
        nonder_overlap = [x for x in count_to_overlap if x[2] not in der_list]
    else:
        nonder_overlap = count_to_overlap
    # Unpack the tuples into separate lists for values and counts
    values, counts,_ = zip(*nonder_overlap)
    # Transform the data to individual values based on counts
    individual_values = np.repeat(values, counts)
    series = pd.Series(individual_values)
    
    # Create a CDF plot using Seaborn
    # plt.figure(figsize=(10, 6))
    if plot:
        sns.ecdfplot(data=individual_values)
        for line in plt.gca().get_lines():  # gca = get current axis
            line.set_linewidth(1.5)
        x = list(np.arange(0,1,0.01))
        y = list(np.arange(0,values[-1],values[-1]/100))
        # Add labels and title
        try:
            plt.plot(y,x, linewidth=5, color='k')
        except:
            print('error')
        plt.xlabel('Ranked Visual Distance from Collection', fontsize=20)
        plt.ylabel('Cumaltive Percent of Total NFTs', fontsize=20)
        plt.xlim(left=0,right=xlim)
        plt.ylim(bottom=0,top=ylim)
        # plt.title(f'Cumulative Distribution of NFTs owned by Top Collection owners')
        # plt.legend()
        # Show the plot
        if show:
            plt.show()
    percentiles = [0.001,0.002,0.01,0.1,.25, .5, .75]
    print(series.describe(percentiles=percentiles))
    return series

def make_overlap_scatter(slug, der_list=None):
    overlap_df = cfu.count_overlaps(slug)
    count_to_overlap = sorted([tuple(x) for x in overlap_df.to_records(index=False)], key=lambda x: x[1])
    count_to_overlap = [x for x in count_to_overlap if x[0] not in opse.SKIP_LIST]
    if der_list:
        nonder_overlap = [x for x in count_to_overlap if x[0] not in der_list]
    else:
        nonder_overlap = count_to_overlap
    print(nonder_overlap)
    labels, values, counts = zip(*nonder_overlap)
    total_count = sum(counts)
    percent_overlap = [count / total_count * 100 for count in counts]
    df = pd.DataFrame({'Visual Distance from Collection': values, 
                       'Percent Overlap': percent_overlap, 
                       'Label': labels})
    fig = px.scatter(df, x='Visual Distance from Collection', y='Percent Overlap', 
                     hover_data=['Label'], title=f'Distribution of NFTs owned by {slug} owners')
    fig.show()
    print(overlap_df.sort_values('row_count', ascending=False).head(20))

def make_owner_latex(df):
    df.set_index('Collection', inplace=True)
    latex_code = df.to_latex()
    return latex_code

#----------------METHODS FOR OWNERSHIP DIRECTION --------------------------

def find_all_owner_direction(db_name='objective_cf_num',dead_slugs = ['goblintownwtf','invisiblefriends'],cutoff=5):
    cf_nums = cfu.get_counterfeit_db(slug=None,db_name=db_name)
    columns = ['slug','cf_num']
    df = pd.DataFrame(cf_nums,columns=columns)
    merged_df = df.query(f"cf_num>={cutoff}")
    slugs = merged_df['slug'].unique()
    owner_dates = cfu.get_all_ownershipdates()
    owner_dates.rename(columns={'wallet': 'address'}, inplace=True)
    date_map = {}
    for slug in slugs:
        try:
            date_map[slug] = cfu.creation_sec_from_db(slug)
        except:
            print(slug)
    #merge with the ownership dates
    owner_dates['creation_date'] = owner_dates['slug'].map(date_map)
    owner_dates['creation_date'] = owner_dates['creation_date'].fillna(owner_dates['creation_date'].max())
    owner_dates['creation_date'] = owner_dates['creation_date'].astype(int)
    counts = []
    for slug in slugs:
        if slug in dead_slugs:
            continue
        counts.append(find_owner_direction(slug,owner_dates))
    return counts

#returns a triplet of the slug, number of owners who owned a look-alike first and total num owners    
def find_owner_direction(slug,owner_dates,creation_date=True):
    owner_dates = owner_dates.drop_duplicates(subset=['slug','address'])
    overlap = cfu.get_overlaps(slug).drop_duplicates(subset=['slug','address'])
    merged_overlaps = pd.merge(owner_dates,overlap,on=['address','slug'])
    der_list = cfu.der_list_from_db(slug)
    main_dates = owner_dates.query(f"slug=='{slug}'")
    complete_df = pd.merge(main_dates,merged_overlaps,on='address',suffixes=['_orig','']).query("sorted_order<100")
    complete_df = complete_df.query(f'slug not in {der_list}')
    complete_df['timestamp'] = complete_df['timestamp'].astype(int)
    complete_df['timestamp_orig'] = complete_df['timestamp_orig'].astype(int)
    complete_df = complete_df.query('creation_date > creation_date_orig')
    filtered_df = complete_df[complete_df['timestamp'] < complete_df['timestamp_orig']]
    print(slug,len(filtered_df),len(complete_df))
    # display(complete_df)
    #TEMPORARY RETURN STATEMENT TO BE REMOVED
    # return complete_df
    return (slug,len(filtered_df),len(complete_df))


def store_all_ownership_dates(destination='ownership_dates_2'):
    top_slugs = cfu.get_top_slugs(cut_off=5,db_name="objective_cf_num")
    print(top_slugs)
    for slug in top_slugs:
        try:
            cfu.store_ownership_dates(slug,destination=destination)
        except:
            pass

def get_ownership_sales(top_slug,valid_tuple):
    command = f"SELECT slug,token_id,buyer,timestamp,transaction,sale_price from look_sim_sales where slug in {valid_tuple} and (payment_token='WETH' or payment_token='ETH')"
    temp_stats_tuples = psql.execute_commands([command])
    columns = ['slug','token_id', 'buyer','timestamp','transaction','sale_price']
    sales_df = pd.DataFrame(temp_stats_tuples,columns=columns)
    sales_df = sales_df.drop_duplicates(subset='transaction')
    sales_df.drop(columns=['transaction'], inplace=True)
    #seperate query for top_slug
    command = f"SELECT slug,token_id,buyer,timestamp,transaction,sale_price from cf_sales where slug='{top_slug}' and (payment_token='WETH' or payment_token='ETH')"
    temp_stats_tuples = psql.execute_commands([command])
    top_sales_df = pd.DataFrame(temp_stats_tuples,columns=columns)
    top_sales_df = top_sales_df.drop_duplicates(subset='transaction')
    top_sales_df.drop(columns=['transaction'], inplace=True)
    sales_df = pd.concat([sales_df,top_sales_df])
    return sales_df

def get_ownership_sales_for_wallets(wallets):
    command = f"SELECT slug,token_id,buyer,timestamp,transaction,sale_price from look_sim_sales where  (payment_token='WETH' or payment_token='ETH')"
    temp_stats_tuples = psql.execute_commands([command])
    columns = ['slug','token_id', 'buyer','timestamp','transaction','sale_price']
    sales_df = pd.DataFrame(temp_stats_tuples,columns=columns)
    del temp_stats_tuples
    sales_df = sales_df.drop_duplicates(subset='transaction')
    sales_df.drop(columns=['transaction'], inplace=True)
    #seperate query for top_slug
    command = f"SELECT slug,token_id,buyer,timestamp,transaction,sale_price from cf_sales where (payment_token='WETH' or payment_token='ETH')"
    temp_stats_tuples = psql.execute_commands([command])
    top_sales_df = pd.DataFrame(temp_stats_tuples,columns=columns)
    del temp_stats_tuples
    top_sales_df = top_sales_df.drop_duplicates(subset='transaction')
    top_sales_df.drop(columns=['transaction'], inplace=True)
    sales_df = pd.concat([sales_df,top_sales_df])
    sales_df = sales_df[sales_df['buyer'].isin(wallets)]
    return sales_df

def get_ownership_dates(slug,sales_df):
    #Step 1: For each token id in the sales_df flitered to slug in question, find the earliest date it was sold
    top_df = sales_df.query(f"slug=='{slug}'")
    top_df['timestamp'] = top_df['timestamp'].astype(int)
    top_df = top_df.sort_values('timestamp')
    top_df = top_df.drop_duplicates(subset=['buyer'])
    top_owners = top_df['buyer'].unique()
    #Step 2: For every other slug in the sales find the earliest date it was sold to the top owners
    all_dates = []
    all_slugs = sales_df['slug'].unique()
    sales_df['timestamp'] = sales_df['timestamp'].astype(int)
    for other_slug in all_slugs:
        if other_slug == slug:
            continue
        other_df = sales_df.query(f"slug=='{other_slug}'")
        other_owners = other_df['buyer'].unique()
        for owner in top_owners:
            if owner in other_owners:
                earliest_date = other_df.query(f"buyer=='{owner}'")['timestamp'].min()
                earliest_sale_price = other_df.query(f"buyer=='{owner}' and timestamp=={earliest_date}")['sale_price'].values[0]
                all_dates.append((other_slug,owner,earliest_date,earliest_sale_price))
    #Step 3 make df
    columns = ['slug','buyer','timestamp','sale_price']
    df = pd.DataFrame(all_dates,columns=columns)
    #Step 4 merge with top_df
    merged_df = pd.merge(df,top_df,on=['buyer'],suffixes=['_orig',''])
    return merged_df

def make_ownership_direction_df():
    top_slugs = cfu.get_top_slugs(cut_off=5,db_name="objective_cf_num")
    rows = []
    for slug in top_slugs:
        date_df = make_one_date_df(slug)
        for name,group in date_df.groupby('slug_orig'):
            rows.append([slug,name,len(group),len(group.query('timestamp_orig<timestamp')),len(group.query('timestamp_orig>timestamp'))])
    columns = ['slug','look_sim','num_owners','ll_first','reference_first']
    df = pd.DataFrame(rows,columns=columns)
    return df

def make_one_date_df(slug):
    look_sims = cfu.get_look_sims(slug,remove_ders=True)
    sales_df = get_ownership_sales(slug,tuple(look_sims))
    date_df = get_ownership_dates(slug,sales_df)
    date_df['look_sim'] = date_df['slug_orig']
    date_df = add_creation_dates(date_df)
    date_df = date_df.query('slug_creation_date<look_sim_creation_date')
    return date_df

def make_ownership_sale_price_df():
    top_slugs = cfu.get_top_slugs(cut_off=5,db_name="objective_cf_num")
    rows = []
    sale_ratios_ll = []
    sale_ratios_ref = []
    for slug in top_slugs:
        look_sims = cfu.get_look_sims(slug,remove_ders=True)
        sales_df = get_ownership_sales(slug,tuple(look_sims))
        date_df = get_ownership_dates(slug,sales_df)
        date_df['look_sim'] = date_df['slug_orig']
        date_df = add_creation_dates(date_df)
        date_df = date_df.query('slug_creation_date<look_sim_creation_date')
        date_df_filt_ll = date_df.query('timestamp_orig<timestamp')
        date_df_filt_ref = date_df.query('timestamp_orig>timestamp')
        sale_ratios_ll.append((slug,date_df_filt_ll['sale_price'],date_df_filt_ll['sale_price_orig']))
        sale_ratios_ref.append((slug,date_df_filt_ref['sale_price'],date_df_filt_ref['sale_price_orig']))
    df_ll = pd.DataFrame(sale_ratios_ll,columns=['slug','sale_price_ref','sale_price_ll']).reset_index(drop=True)
    df_ll['sale_price_ref'] = df_ll['sale_price_ref'].apply(lambda x: list(x))
    df_ll['sale_price_ll'] = df_ll['sale_price_ll'].apply(lambda x: list(x))
    df_ll = df_ll.reset_index()
    df_ll = df_ll.apply(pd.Series.explode)
    df_ref = pd.DataFrame(sale_ratios_ref,columns=['slug','sale_price_ref','sale_price_ll']).reset_index(drop=True)
    df_ref['sale_price_ref'] = df_ref['sale_price_ref'].apply(lambda x: list(x))
    df_ref['sale_price_ll'] = df_ref['sale_price_ll'].apply(lambda x: list(x))
    df_ref = df_ref.reset_index()
    df_ref = df_ref.apply(pd.Series.explode)
    return df_ll,df_ref

def add_creation_dates(df):
    slugs = list(df['slug'].unique())+list(df['look_sim'].unique())
    date_map = {}
    for slug in slugs:
        try:
            date_map[slug] = cfu.creation_sec_from_db(slug)
        except:
            print(slug)
    df['slug_creation_date'] = df['slug'].map(date_map)
    df['slug_creation_date'] = df['slug_creation_date'].fillna(df['slug_creation_date'].max())
    df['slug_creation_date'] = df['slug_creation_date'].astype(int)
    df['look_sim_creation_date'] = df['look_sim'].map(date_map)
    df['look_sim_creation_date'] = df['look_sim_creation_date'].fillna(df['look_sim_creation_date'].max())
    df['look_sim_creation_date'] = df['look_sim_creation_date'].astype(int)
    return df

def get_wallet_sales_distro(slug,direction='ll_first',norm=True):
    date_df = make_one_date_df(slug)
    date_df = add_creation_dates(date_df)
    date_df = date_df.query('slug_creation_date<look_sim_creation_date')
    if direction == 'll_first':
        date_df = date_df.query('timestamp_orig<timestamp')
        wallets = date_df['buyer'].unique()
    elif direction == 'ref_only':
        date_df = date_df.query('timestamp_orig<timestamp')
        ll_wallets = date_df['buyer'].unique()
        ref_sales = get_ownership_sales(slug,('',))
        ref_sales = ref_sales.sort_values('timestamp')
        date_df = ref_sales.drop_duplicates(subset=['buyer'])
        print(ref_sales['buyer'].nunique())
        wallets = set(ref_sales['buyer'].unique())-set(ll_wallets)
        print(len(wallets))
    else:
        date_df = date_df.query('timestamp_orig>timestamp')
    
    sales_df = get_ownership_sales_for_wallets(tuple(wallets))
    #group by wallet and filter sales after timestamp and normalize by sale_price
    wallet_sales = []
    for wallet in wallets:
        wallet_df = sales_df.query(f"buyer=='{wallet}'")
        last_timestamp = date_df.query(f"buyer=='{wallet}'")['timestamp_orig'].min()
        wallet_df = wallet_df.query(f"timestamp<{last_timestamp}")
        ref_sale = date_df.query(f"buyer=='{wallet}'")['sale_price'].values[0]
        if norm:
            wallet_sales.append(wallet_df['sale_price']/ref_sale)
        else:
            wallet_sales.append(wallet,wallet_df['sale_price'])
    return wallet_sales

def save_all_wallet_sales(slugs):
    for slug in slugs:
        wallet_sales_1 = get_wallet_sales_distro(slug,'ll_first')
        sales_tuple = (slug,wallet_sales_1)
        with open(f'ownership_out/{slug}_wallet_sales.pkl','wb') as f:
            pickle.dump(sales_tuple,f)

def num_lls_to_purchase_prob(slug):
    df_ll_first = make_one_date_df(slug)
    df_ll_first = df_ll_first.query('timestamp_orig<timestamp')
    ll_group = df_ll_first.groupby('buyer').agg({'slug':'count'})
    #turn  buyer group into df with one column buyer and one column num_pruchases
    ll_group = ll_group.reset_index()
    ll_group.columns = ['buyer','num_purchases']
    look_sims = cfu.get_look_sims(slug,remove_ders=True)
    sales_df = get_ownership_sales(slug,tuple(look_sims))
    sales_df = sales_df.sort_values('timestamp')
    sales_df_no_ref = sales_df.query('slug!=@slug')
    sales_df_ref = sales_df.query('slug==@slug')
    sales_df_no_ref = sales_df_no_ref.drop_duplicates(subset=['slug','buyer'],keep='first')
    #group by buyer and count how many unique slugs they have bought
    buyer_group = sales_df_no_ref.groupby('buyer').agg({'slug':'count'})
    #turn  buyer group into df with one column buyer and one column num_pruchases
    buyer_group = buyer_group.reset_index()
    buyer_group.columns = ['buyer','num_purchases']
    #Filter out buyers who appear in sales_df_ref
    buyer_group = buyer_group[~buyer_group['buyer'].isin(sales_df_ref['buyer'])]
    # Step 1: Count the occurrences of each num_purchases value in both DataFrames
    count_df1 = ll_group['num_purchases'].value_counts().sort_index()
    count_df2 = buyer_group['num_purchases'].value_counts().sort_index()

    # Step 2: Combine the counts into a single DataFrame
    counts_combined = pd.DataFrame({'df1': count_df1, 'df2': count_df2}).fillna(0)

    # Step 3: Calculate the ratio for each num_purchases value
    counts_combined['total'] = counts_combined['df1'] + counts_combined['df2']
    counts_combined['ratio'] = counts_combined['df1'] / counts_combined['total']
    return counts_combined