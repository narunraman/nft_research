import synth_utils as su
import pickle
import pandas as pd
import logging
import os

def create_all_synths(cached=True):
    logging.basicConfig(filename=f'synth_creation.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    if cached:
        with open('synth_df_cache.pkl','rb') as f:
            df,first_day = pickle.load(f)
    else:
        df,first_day = su.build_synth_df()
    slugs = df['slug'].unique().tolist()
    synths = []
    for slug in slugs:
        logging.info(f"Creating synths for '{slug}'.")
        filt_df = su.filter_synth_df(slug,df)
        ko_days = su.get_ko_days(slug,first_day)
        for ko_day in ko_days:
            try:
                sc = su.query_day(slug,ko_day[1],filt_df)
            except:
                logging.info(f"Error encountered on Knock-off'{ko_day[0]}'")
                continue
            if sc is None:
                continue
            if sc.original_data.rmspe_df['post/pre'].item()>1:
                logging.info(f"Knock-off'{ko_day[0]}' had pre/post ratio of {sc.original_data.rmspe_df['post/pre'].item()} saving for placebo.")
                synths.append((slug,ko_day[0],sc))
    with open('synth_out.pkl','wb') as f:
        pickle.dump(synths,f)

def create_synth(slug,alt,norm=False,cached=True,alt_day=None,offset=0):
    if cached:
        with open('synth_df_cache.pkl','rb') as f:
            df,first_day = pickle.load(f)
        # df.rename(columns={'price':'price_unnormalized','price_normalized':'price'},inplace=True)
    else:
        df,first_day = su.build_synth_df(normalize=norm,cutoff=0)
    
    # filt_df = su.filter_synth_df(slug,df)
    filt_df = df.drop(columns=['OG_data']).sort_values(by=['slug','day_id'])
    
    if alt_day is None:
        alt_day = su.get_ko_day(alt,first_day)[1]
    sc = su.query_day(slug,alt_day,filt_df,offset=offset)
    return sc,first_day

#Must have a cached synth_df to run this function
def compute_valid_placebos(slug,alt):
    with open('synth_df_cache.pkl','rb') as f:
        df,first_day = pickle.load(f)
    alt_day = su.get_ko_day(alt,first_day)[1]
    print(alt_day)
    df_valid_placebos = df.query(f"OG_data and slug!='{slug}' and day_id<{alt_day-60}")
    unique_slugs = df_valid_placebos['slug'].unique()
    result_df = pd.DataFrame()
    print(unique_slugs)
    for plac_slug in unique_slugs:
        print(plac_slug,alt)
        try:
            plac_sc = create_synth(plac_slug,alt,cached=True)[0]
            result_row = plac_sc.original_data.rmspe_df
            #append result_row to result_df
            result_df = pd.concat([result_df,result_row])
        except:
            continue
    return result_df

#Slurm version of the above function with no database calls
def compute_valid_placebos_slurm(slug,alt,offset=0,filename='alt_day_dict.pkl'):
    with open('synth_df_cache.pkl','rb') as f:
        df,first_day = pickle.load(f)
    with open(filename,'rb') as f:
        alt_day_dict = pickle.load(f)
    alt_day = alt_day_dict[alt]
    # df.rename(columns={'price':'price_unnormalized','price_normalized':'price'},inplace=True)
    print(alt_day)
    df_valid_placebos = df.query(f"OG_data and slug!='{slug}' and day_id<{alt_day-60}")
    unique_slugs = df_valid_placebos['slug'].unique()
    for u_slug in unique_slugs:
        u_slug_df = df.query(f"slug=='{u_slug}' and OG_data and day_id>{alt_day-60} and day_id<{alt_day+15}")
        valid_days = u_slug_df['day_id'].nunique()
        total_occurences = u_slug_df['occurrences'].sum()
        if valid_days<35 or total_occurences<100:
            unique_slugs = unique_slugs[unique_slugs!=u_slug]
    result_df = pd.DataFrame()
    print(len(unique_slugs))
    plac_sc = create_synth(slug,alt,cached=True,alt_day=alt_day,offset=offset)[0]
    result_row = plac_sc.original_data.rmspe_df 
    #append result_row to result_df
    result_df = pd.concat([result_df,result_row])
    for plac_slug in unique_slugs:
        print(plac_slug,alt)
        try:
            plac_sc = create_synth(plac_slug,alt,cached=True,alt_day=alt_day)[0]
            result_row = plac_sc.original_data.rmspe_df
            #append result_row to result_df
            result_df = pd.concat([result_df,result_row])
        except:
            continue
    return result_df

def get_all_placebo_results(job_file,out_file,offset=False):
    result_df = pd.DataFrame()
    for slug_id in range(0,50):
        tops = pd.read_pickle(f"job_files/{job_file}")
        try:
            column1_data = tops['Top_100'].tolist()
            column2_data = tops['Alt'].tolist()
            # Combine them into a list of tuples
            slug,alt = list(zip(column1_data, column2_data))[slug_id]
            if offset:
                outpath = f'synth_outs/placebo_output_{out_file}_offset/result_df' + str(slug_id) + '.pkl'
            else:
                outpath = f'synth_outs/placebo_output_{out_file}/result_df' + str(slug_id) + '.pkl'

            df = pd.read_pickle(outpath).sort_values(by='post/pre',ascending=False)
            row = show_placebo_results(slug,alt,df,slug_id,job_file)
            result_df = pd.concat([result_df,row])
        except:
            continue
    return result_df

def show_placebo_results(slug,alt,df,result_id,dir_name):
    # print(f"Results for pair {slug} and {alt}")
    num_samples = len(df)
    base_error = df.query(f"unit=='{slug}'").iloc[0]['post/pre']
    p_value = (sum(df['post/pre']>base_error)+1)/num_samples
    #make df row out of results
    row = pd.DataFrame({'slug':slug,'alt':alt,'num_samples':num_samples,'base_error':base_error,'p_value':p_value,'result_id':result_id,'experiment':dir_name},index=[0])
    return row

def make_and_save_alt_day_dict(slugs,filename='alt_day_dict.pkl'):
    with open('synth_df_cache.pkl','rb') as f:
        _,first_day = pickle.load(f)
    alt_day_dict = {}
    for slug in slugs:
        alt_day_dict[slug] = su.get_ko_day(slug,first_day)[1]
    with open(filename,'wb') as f:
        pickle.dump(alt_day_dict,f)


def write_slurm_script(slug, array_size):
    """
    Generates a Slurm script for the given slug and array size, then
    writes it to slurm_scripts/slurm_{slug}.sh.
    """
    # Create output directory if it doesn't exist
    os.makedirs("slurm_scripts", exist_ok=True)

    slurm_script = f"""#!/bin/bash
#SBATCH --time=5-00:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=20G
#SBATCH --partition=ada_cpu_long
#SBATCH --job-name=placebo_{slug}
#SBATCH --output=/global/scratch/tlundy/NFT_Research/nft_research/Synthetic_control/synth_outs/placebo_output_{slug}/placebo_%a.out
#SBATCH --error=/global/scratch/tlundy/NFT_Research/nft_research/Synthetic_control/synth_errs/placebo_output_{slug}/placebo_%a.err
#SBATCH --account=rrg-kevinlb
#SBATCH --array=0-{array_size}

cd /global/scratch/tlundy/NFT_Research
source nft_venv2/bin/activate
echo "loaded environment"
cd nft_research/Synthetic_control
python -u placebo_jobs.py $SLURM_ARRAY_TASK_ID {slug} 0 {slug}_avg_synth_df.pkl
"""

    # Write the script to a file in slurm_scripts directory
    filename = f"slurm_scripts/slurm_{slug}.sh"
    with open(filename, "w") as f:
        f.write(slurm_script)

    print(f"Slurm script generated and saved to: {filename}")
