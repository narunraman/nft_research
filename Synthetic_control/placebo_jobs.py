
import pandas as pd
import sys
import synth_job as sj
import os

slug_id = int(sys.argv[1])
out_file = sys.argv[2]
offset = int(sys.argv[3])
job_file = sys.argv[4]
df_id = slug_id
tops = pd.read_pickle(f"job_files/{job_file}")
column1_data = tops['Top_100'].tolist()
column2_data = tops['Alt'].tolist()
# Combine them into a list of tuples
if slug_id >= len(column1_data):
    print(f"Slug id {slug_id} out of range {len(column1_data)}")
    sys.exit(1)
slug,alt = list(zip(column1_data, column2_data))[slug_id]
print(f"Starting pair {slug} and {alt}")
if offset>0:
    out_file = out_file + "_offset"
result_df = sj.compute_valid_placebos_slurm(slug,alt,offset=offset,filename=f'job_files/{out_file}_alt_day_dict.pkl')
#make directory if it doesn't exist
try:
    os.mkdir(f'synth_outs/placebo_output_{out_file}')
except:
    pass
outpath = f'synth_outs/placebo_output_{out_file}/result_df' + str(df_id) + '.pkl'
result_df.to_pickle(outpath)