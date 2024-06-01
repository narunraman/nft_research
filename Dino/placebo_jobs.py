
import pandas as pd
import sys
import synth_job as sj

slug_id = int(sys.argv[1])
filename = sys.argv[2]
offset = int(sys.argv[3])
df_id = slug_id
tops = pd.read_pickle(f"high_vol_{filename}.pkl")
column1_data = tops['Top_100'].tolist()
column2_data = tops['Alt'].tolist()
# Combine them into a list of tuples
slug,alt = list(zip(column1_data, column2_data))[slug_id]
print(f"Starting pair {slug} and {alt}")
if offset>0:
    filename = filename + "_offset"
result_df = sj.compute_valid_placebos_slurm(slug,alt,offset=offset)
outpath = f'placebo_output_{filename}/result_df' + str(df_id) + '.pkl'
result_df.to_pickle(outpath)