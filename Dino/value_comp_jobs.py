
import pandas as pd
import sys
import synth_job as sj
import counterfeit_utils as cfu
import itertools
job_id = int(sys.argv[1])
cut_off = 5
db_name = 'objective_cf_num'
top_slugs = cfu.get_top_slugs(cut_off,db_name)
bandwidths = [None,25,10,5,1]
combinations = list(itertools.product(top_slugs,bandwidths))
slug,bandwidth = combinations[job_id]
print(len(combinations))
window_size = [1,5,10,15]
num_samples = 10000
results_w_param = []
for interval_length in window_size:
    print(f"Beginning Interval Length: {interval_length} and Bandwidth: {bandwidth} and Slug: {slug}")
    slug,mean_samples,mean_data = cfu.value_comparison(slug,interval_length,num_samples,db_name='objective_cf_num',bandwidth=bandwidth,logger=None)
    results_w_param.append((slug,mean_samples,mean_data,interval_length,bandwidth))
result_df = pd.DataFrame(results_w_param,columns=['slug','mean_samples','mean_data','interval_length','bandwidth'])
outpath = 'val_comp/result_df' + str(job_id) + '.pkl'
result_df.to_pickle(outpath)