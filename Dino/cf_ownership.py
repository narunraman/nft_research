import counterfeit_utils as cfu
import pandas as pd
import opensea_methods as opse
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def generate_all_ownerhip_stats(db_name='objective_cf_num',cutoff=5):
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
def make_overlap_cdf(slug,count_to_overlap,der_list = None,xlim=2000,ylim=0.3,show=True):
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
    overlap_df = count_overlaps(slug)
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

def find_all_owner_direction(db_name='objective_cf_num',dead_slugs = ['golbintownwtf','invisiblefriends'],cutoff=5):
    cf_nums = cfu.get_counterfeit_db(slug=None,db_name=db_name)
    columns = ['slug','cf_num']
    df = pd.DataFrame(cf_nums,columns=columns)
    merged_df = df.query(f"cf_num>={cutoff}")
    slugs = merged_df['slug'].unique()
    owner_dates = cfu.get_all_ownershipdates()
    owner_dates.rename(columns={'wallet': 'address'}, inplace=True)
    counts = []
    for slug in slugs:
        if slug in dead_slugs:
            continue
        counts.append(find_owner_direction(slug,owner_dates))
    return counts
    
def find_owner_direction(slug,owner_dates):
    overlap = cfu.get_overlaps(slug).drop_duplicates(subset=['slug','address'])
    merged_overlaps = pd.merge(owner_dates,overlap,on=['address','slug'])
    der_list = cfu.der_list_from_db(slug)
    main_dates = owner_dates.query(f"slug=='{slug}'")
    complete_df = pd.merge(main_dates,merged_overlaps,on='address',suffixes=['_orig','']).query("sorted_order<100")
    complete_df = complete_df.query(f'slug not in {der_list}')
    complete_df['timestamp'] = complete_df['timestamp'].astype(int)
    complete_df['timestamp_orig'] = complete_df['timestamp_orig'].astype(int)
    filtered_df = complete_df[complete_df['timestamp'] < complete_df['timestamp_orig']]
    print(slug,len(filtered_df),len(complete_df))
    # display(complete_df)
    return (slug,len(filtered_df),len(complete_df))