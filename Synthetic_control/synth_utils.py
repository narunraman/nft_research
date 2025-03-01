import sys
sys.path.append("..")
import Dino.counterfeit_utils as cfu
import Dino.cf_value as cfv
import pandas as pd
from SyntheticControlMethods import Synth,DiffSynth
from matplotlib import pyplot as plt
import pickle
first_day = 0

def build_synth_df(seed=1234,normalize=True,cutoff=5,all_slugs=True,save=False):
    if all_slugs:
        df = cfv.all_sales_from_db()
    else:
        top_slugs = cfu.get_top_slugs(cutoff,'objective_cf_num')
        df = cfu.day_sales_from_db(top_slugs[0])
        df['slug'] = top_slugs[0]
        for slug in top_slugs[1:]:
            df2 = cfu.day_sales_from_db(slug)
            df2['slug'] = slug
            df = pd.concat([df,df2])
    first_day = min(df['day'])
    df['day_id'] = df['day'].apply(days_since_first,first_day=first_day)
    df['OG_data'] = True
    result = df.groupby(['slug', 'day_id','OG_data']).agg({'price': ['median', 'count']}).reset_index()
    # Rename columns for clarity
    result.columns = ['slug', 'day_id','OG_data', 'price', 'occurrences']
    multi_index = pd.MultiIndex.from_product([result['slug'].unique(), result['day_id'].unique()], names=['slug', 'day_id'])
    # Reindex the DataFrame using the MultiIndex
    result = result.set_index(['slug', 'day_id']).reindex(multi_index).reset_index()
    result = result.groupby('slug').apply(interpolate_missing_days).reset_index(drop=True)
    if normalize:
        # Calculate the average price for each group
        average_prices = result.groupby('slug')['price'].transform('mean')
        
        # Divide each price by the average price for its group
        result['price_normalized'] = result['price'] / average_prices
    if save:
        with open('synth_df_cache.pkl','wb') as f:
            pickle.dump((result,first_day),f)
    return (result,first_day)

def days_since_first(date,first_day):
    if isinstance(first_day, int):
        first_day = pd.to_datetime(first_day, unit='D', origin='2018-01-01')
    delta = date - first_day
    return delta.days

def interpolate_missing_days(df):
    # Ensure day_id is sorted before interpolation
    df = df.sort_values('day_id')
    # Interpolate missing values
    df['price'] = df['price'].interpolate(method='linear')
    df['price'] = df['price'].fillna(method='bfill').fillna(method='ffill')
    df['occurrences'] = df['occurrences'].fillna(0)
    df['OG_data'] = df['OG_data'].fillna(False)
    return df

def filter_synth_df(slug,df):
    q_df = df.query(f"slug=='{slug}' and OG_data")
    merge_df = pd.merge(df,q_df,on='day_id',how='right',suffixes=('', '_df2'))
    merge_df = merge_df.drop(columns=['slug_df2','price_df2','occurrences_df2','OG_data','OG_data_df2']).sort_values(by=['slug','day_id'])
    slug_day_counts = merge_df.groupby('slug')['day_id'].nunique()
    valid_slugs = slug_day_counts[slug_day_counts == len(q_df)].index
    # Filter the DataFrame to keep only rows with slugs in valid_slugs
    filtered_df = merge_df[merge_df['slug'].isin(valid_slugs)].sort_values(by=['slug','day_id'])
    return filtered_df

# def reindex_group(group):
#     new_index = pd.RangeIndex(group['day_id'].min(), group['day_id'].max() + 1)
#     reindexed_df = group.set_index('day_id').reindex(new_index).reset_index().rename(columns={'index': 'day_id'})
#     return reindexed_df

def query_day(slug,day,df,pre_start=3,offset=0):
    day = day-pre_start
    df = df.query(f"{day-60}<day_id<{day+15+offset}")
    df = df.query(f"day_id<={day} or day_id>{day+offset}")
    df = df.query("slug!='dope-shibas'")
    unique_slugs = df['slug'].unique()
    if offset>0:
        df['day_id'] = df['day_id'].apply(lambda x: x-offset if x > day else x) 
    for u_slug in unique_slugs:
        u_slug_df = df.query(f"slug=='{u_slug}'")
        total_occurences = u_slug_df['occurrences'].sum()
        if total_occurences<100:
            unique_slugs = unique_slugs[unique_slugs!=u_slug]
    filt_df = df.query(f"slug in {tuple(unique_slugs)}")
    if df.empty:
        return None
    sc = Synth(filt_df, "price", "slug", "day_id", day, slug, n_optim=5)
    return sc

def diff_query_day(slug,day,df):
    df = df.query(f"day_id<{day+20}")
    if df.empty:
        return None
    sc = DiffSynth(df, "price", "slug", "day_id", day, slug, n_optim=10)
    return sc

def get_ko_days(slug,first_day,remove_ders=False):
    intervals = cfu.find_cf_days(slug,'objective_cf_num',remove_ders = remove_ders)
    ko_days = [(x[0],days_since_first(pd.to_datetime(x[2]),first_day)) for x in intervals]
    return ko_days

def get_ko_day(alt,first_day):
    date = cfu.creation_date_from_db(alt)
    day = days_since_first(pd.to_datetime(date),first_day)
    ko_days = (alt,day,first_day)
    return ko_days

#Find all Look sims for slug, sort by volume and return non-overlapping KOs
#Need to pickle the result 
def create_average_synth_job_df(slug,overlap=15,return_dates=False):
    #Get all looksims
    look_sims = cfu.get_look_sims(slug,remove_ders=True)
    alt_tuples = []
    for alt in look_sims:
        alt_tuples.append((alt,cfu.volume_from_db(alt),get_ko_day(alt,0)[1]))
    
    alt_df = pd.DataFrame(alt_tuples,columns=['slug','volume','ko_day'])
    df_sorted = alt_df.sort_values(by="volume", ascending=False).reset_index(drop=True)

    selected_rows = []
    while not df_sorted.empty:
        # -- Pick the top row (highest volume)
        chosen_row = df_sorted.iloc[0]
        
        # -- Keep that row in our final result
        selected_rows.append(chosen_row)
        chosen_ko = chosen_row["ko_day"]
        mask_no_overlap = (
            (df_sorted["ko_day"] < chosen_ko - overlap)
            | (df_sorted["ko_day"] > chosen_ko + overlap)
        )
        
        # -- Keep only rows that do NOT overlap:
        df_sorted = df_sorted[mask_no_overlap].reset_index(drop=True)

    # Turn the selected rows into a DataFrame
    result_df = pd.DataFrame(selected_rows).query("volume>0")
    slug_and_alts = [(slug,alt) for alt in result_df['slug']]
    slug_and_alts_df = pd.DataFrame(slug_and_alts,columns=['Top_100','Alt'])
    if return_dates:
        slug_and_alts_df['alt_date'] = slug_and_alts_df['Alt'].apply(lambda x: cfu.creation_date_from_db(x))
    return slug_and_alts_df


    