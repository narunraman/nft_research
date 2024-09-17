import synth_job as sj
slug = 'beanzofficial'
alt = 'azukielementals'
df = sj.compute_valid_placebos_slurm(slug,alt,offset=5)
#save df
df.to_pickle('beanz_azukielementals_result_df.pkl')