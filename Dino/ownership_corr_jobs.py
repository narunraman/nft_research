import sys
import cf_ownership as cfo
job_id = int(sys.argv[1])
num_jobs = int(sys.argv[2])
result_df = cfo.compute_all_ownership_corrs(job_id=job_id,num_jobs=num_jobs)
outpath = 'own_corr/result_df' + str(job_id) + '.pkl'
result_df.to_pickle(outpath)