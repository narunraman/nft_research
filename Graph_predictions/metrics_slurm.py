import os
from collections import defaultdict
import argparse
import json



def construct_bash_script(memory,job_name,graph_idx):
    return f"""#!/bin/bash\n#SBATCH --time=5-00:00:00\n#SBATCH --nodes=1\n#SBATCH --cpus-per-task=1\n#SBATCH --partition=ada_cpu_long\n#SBATCH --mem={memory}G\n#SBATCH --job-name={job_name}\n#SBATCH --output=/global/scratch/tlundy/NFT_Research/nft_research/Graph_predictions/graph_metrics/slurm_scripts/outputs/{job_name}.out\n#SBATCH --error=/global/scratch/tlundy/NFT_Research/nft_research/Graph_predictions/graph_metrics/slurm_scripts/outputs/{job_name}.err\n#SBATCH --account=rrg-kevinlb\n\ncd /global/scratch/tlundy/NFT_Research\nsource nft_venv/bin/activate\ncd nft_research/Graph_predictions\npython -u graph_metrics.py --graph {graph_idx}"""


def gen_script():
    from GraphDataset import GraphDataset
    num_graphs = len(GraphDataset('dataset_stor/graph_dataset_3',normalize=True))
    for idx in range(num_graphs):
        script_text = construct_bash_script(20, f'compute_graph_{idx}', idx)
        with open(f'graph_metrics/slurm_scripts/slurm_script_{idx}.sh', 'w') as f:
            f.write(script_text)
        print('\nGraph:', idx)

def run_scripts():
    for idx in range(30):
        os.system('sbatch graph_metrics/slurm_scripts/slurm_script_{idx}.sh'.format(idx = idx))

# gen_script()
run_scripts()