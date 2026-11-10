#!/bin/bash
# Kernel ablations as a batch job, so a dropped ssh connection cannot kill it.
# From hw1-asr, on the head node (mlp):
#     sbatch ablation/ablate.sbatch                 # check + sweep + e2e (about 1-1.5 h)
#     sbatch ablation/ablate.sbatch --stage check   # correctness only (about 10 min)
#     squeue -u "$USER"                             # running? (R) waiting? (PD)
#     tail -f ablate-JOB_ID.out                     # live progress, Ctrl-C stops watching only
# Pinned to the RTX 2080 Ti so every number is comparable with the E03 baseline.
#SBATCH -p Teaching
#SBATCH --gres=gpu:nvidia_geforce_rtx_2080_ti:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=ablate-%j.out

export PATH=/opt/cuda-12.8.0/bin:$PATH
source "${CONDA_BASE:-$HOME/miniconda3}/etc/profile.d/conda.sh"
conda activate mls-asr
cd "$SLURM_SUBMIT_DIR"

echo "node: $(hostname)"
nvidia-smi -L | sed 's/ (UUID:.*)//'
python -u ablation/run_ablations.py "$@"
