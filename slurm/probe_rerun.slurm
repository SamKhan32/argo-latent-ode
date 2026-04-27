#!/bin/bash
#SBATCH --job-name=ocean_probe_rerun
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err
#SBATCH --time=6:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sphill86@charlotte.edu
echo "======================================================"
echo "Start Time  : $(date)"
echo "Submit Dir  : $SLURM_SUBMIT_DIR"
echo "Job ID/Name : $SLURM_JOBID / $SLURM_JOB_NAME"
echo "Partition   : $SLURM_JOB_PARTITION"
echo "======================================================"
set -euo pipefail
cd $SLURM_SUBMIT_DIR
mkdir -p logs checkpoints results
source ~/envs/ocean/bin/activate
export PYTHONPATH=$SLURM_SUBMIT_DIR${PYTHONPATH:+:$PYTHONPATH}

python -u run/main.py --stage probe
python -u run/main.py --stage gru_probe
python -u run/main.py --stage probe_baseline

echo "======================================================"
echo "End Time : $(date)"
echo "======================================================"
