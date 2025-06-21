#!/bin/bash
#SBATCH --time=5-00:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=5G
#SBATCH --partition=ada_cpu_long
#SBATCH --job-name=value_comp
#SBATCH --output=/global/scratch/tlundy/NFT_Research/nft_research/Dino/val_comp_ms/comp_%a.out
#SBATCH --error=/global/scratch/tlundy/NFT_Research/nft_research/Dino/val_comp_ms/comp_%a.err
#SBATCH --account=rrg-kevinlb
#SBATCH --array=0-70

cd /global/scratch/tlundy/NFT_Research
source nft_venv2/bin/activate
echo "loaded environment"
cd nft_research/Dino/
python -u value_comp_jobs.py $SLURM_ARRAY_TASK_ID