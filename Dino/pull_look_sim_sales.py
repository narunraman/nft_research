import counterfeit_utils as cfu
import cf_value as cfv
import sys
sys.path.append('..')
import data_retrieval.psql_methods as psql
import pandas as pd

look_sims = cfu.get_all_look_sims()
cfv.retreive_full_time_series(look_sims,table_name='look_sim_sales',log_file='look_sim_series.log')