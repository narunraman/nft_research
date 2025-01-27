import synth_job as sj
slug = 'world-of-women-nft'
alt = 'womenrise'
df = sj.compute_valid_placebos_slurm(slug,alt,offset=15)
#save df
df.to_pickle('world_of_women_result_df_offset.pkl')