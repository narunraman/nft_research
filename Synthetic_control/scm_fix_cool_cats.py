import synth_job as sj
slug = 'cool-cats-nft'
alt = 'alienfrensnft'
df = sj.compute_valid_placebos_slurm(slug,alt,offset=15)
#save df
df.to_pickle('cool_cats_result_df_offset.pkl')