#!/bin/bash
# The hand-in run required by the hw1 README: `make test` and `./run.sh --profile` from the SAME GPU node,
# plus `nvidia-smi -L`, on the configuration we submit. From hw1-asr on the head node:
#     sbatch ablation/handin.sbatch
#     tail -n 80 handin-JOB_ID.out
#SBATCH -p Teaching
#SBATCH --gres=gpu:nvidia_geforce_rtx_2080_ti:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:45:00
#SBATCH --output=handin-%j.out

export PATH=/opt/cuda-12.8.0/bin:$PATH
source "${CONDA_BASE:-$HOME/miniconda3}/etc/profile.d/conda.sh"
conda activate mls-asr
cd "$SLURM_SUBMIT_DIR"

echo "node: $(hostname)"
nvidia-smi -L | sed 's/ (UUID:.*)//'
grep -h -E "^#define ASR_(TILE|ATTN_WARPS|ROPE_THREADS|REDUCE_THREADS) " csrc/*.cu
make clean >/dev/null
make test
./run.sh --profile
