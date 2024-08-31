#!/bin/bash
#SBATCH --time=5-00:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=20G
#SBATCH --partition=ada_cpu_long
#SBATCH --job-name=placebo_test
#SBATCH --output=/global/scratch/tlundy/NFT_Research/nft_research/Dino/placebo_output_ko/placebo_%a.out
#SBATCH --error=/global/scratch/tlundy/NFT_Research/nft_research/Dino/placebo_output_ko/placebo_%a.err
#SBATCH --account=rrg-kevinlb
#SBATCH --array=0-49

cd /global/scratch/tlundy/NFT_Research
source nft_venv/bin/activate
echo "loaded environment"
cd nft_research/Dino
python -u placebo_jobs.py $SLURM_ARRAY_TASK_ID ko 0