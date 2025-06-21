import pandas as pd
import sys
sys.path.append("..")
import counterfeit_utils as cfu
job_id = int(sys.argv[1])
cut_off = 5
db_name = 'objective_cf_num'
top_slugs = cfu.get_top_slugs(cut_off,db_name)
slug = top_slugs[job_id]
window_size = [1,5,10,15]
results_w_param = []
for interval_length in window_size:
    for no_overlap in [True,False]:
        print(f"Beginning Interval Length: {interval_length} and Slug: {slug}")
        slug,no_ko_days,ko_days = cfu.value_comparison_no_overlap(slug,interval_length,db_name='objective_cf_num',logger=None,no_overlap=no_overlap)
        results_w_param.append((slug,no_ko_days,ko_days,interval_length,no_overlap))
result_df = pd.DataFrame(results_w_param,columns=['slug','no_ko_days','ko_days','interval_length','no_overlap'])
outpath = 'val_comp_ms/result_df' + str(job_id) + '.pkl'
result_df.to_pickle(outpath)