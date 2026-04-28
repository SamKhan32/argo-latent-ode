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

# ── Results Summary ────────────────────────────────────────
echo ""
echo "======================================================"
echo "  RESULTS SUMMARY"
echo "======================================================"

summarize_losses() {
    local label=$1
    local file=$2
    if [[ -f "$file" ]]; then
        # Find row with minimum val_loss (skip header)
        local best
        best=$(tail -n +2 "$file" | awk -F',' 'NR==1 || $3<min {min=$3; row=$0} END {print row}')
        local epoch=$(echo "$best" | cut -d',' -f1)
        local train=$(echo "$best" | cut -d',' -f2)
        local val=$(echo "$best"   | cut -d',' -f3)
        printf "  %-28s  epoch=%-4s  train=%-10s  val=%s\n" "$label" "$epoch" "$train" "$val"
    else
        printf "  %-28s  [not found]\n" "$label"
    fi
}

summarize_losses "Encoder"           "$RUN_DIR/encoder_losses.csv"
summarize_losses "Neural ODE"        "$RUN_DIR/node_curriculum_losses.csv"
summarize_losses "GRU Baseline"      "$RUN_DIR/gru_losses.csv"
summarize_losses "Probe (ODE)"       "$RUN_DIR/probe_losses.csv"
summarize_losses "Probe (GRU)"       "$RUN_DIR/gru_probe_losses.csv"

echo ""
echo "  Extrapolation (final MSE by model):"
if [[ -f "$RUN_DIR/extrapolation_results.csv" ]]; then
    # Print header + one best row per model
    tail -n +2 "$RUN_DIR/extrapolation_results.csv" | awk -F',' '
        {
            model=$1
            mse=$4
            if (!(model in best) || mse < best[model]) {
                best[model]=mse; horizon_days[model]=$3
            }
        }
        END {
            for (m in best)
                printf "  %-20s  best_horizon=%-6s days  final_mse=%s\n", m, horizon_days[m], best[m]
        }
    '
else
    echo "  [extrapolation_results.csv not found]"
fi

echo "======================================================"