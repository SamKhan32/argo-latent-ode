#!/bin/bash
#SBATCH --job-name=ocean_latent_ode
#SBATCH --output=results/slurm_%j/slurm.out
#SBATCH --error=results/slurm_%j/slurm.err
#SBATCH --time=20:00:00
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

# SLURM already created results/slurm_$JOBID/ for the log files.
# We rename it to a timestamp dir and leave a symlink so the log handle stays valid.
RUN_DIR="results/$(date +%Y%m%d_%H%M%S)"
mv results/slurm_${SLURM_JOBID} $RUN_DIR
ln -s $(realpath $RUN_DIR) results/slurm_${SLURM_JOBID}

source ~/envs/ocean/bin/activate
export PYTHONPATH=$SLURM_SUBMIT_DIR${PYTHONPATH:+:$PYTHONPATH}

python -u run/main.py --stage all --results_dir $RUN_DIR
python -u run/main.py --stage extrapolation --results_dir $RUN_DIR

echo "======================================================"
echo "End Time : $(date)"
echo "======================================================"
