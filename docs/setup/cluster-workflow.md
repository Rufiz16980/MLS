# Cluster workflow: what persists

- **Kept between GPU sessions:** the cluster home (`~`, e.g. `/home/<UUN>`): your git clone, conda env, downloaded model, results. It is a shared filesystem, so the head node and all GPU nodes see the same files; `exit` from a GPU session deletes nothing there.
- **Deleted:** anything in `/tmp` or `/disk/scratch` on a GPU node (node-local).
- **Not backed up** and reclaimed when your access ends: push code to GitHub regularly and copy results (logs, report numbers) to your laptop.
- The gateway (`ssh.inf.ed.ac.uk`) shows a different (AFS) home, not the cluster home. Copy files via the gateway to `mlp`; exact command to be added once we have results.
- One-time setup needs about 20 GB in the home. Run `setup_env.sh` once on the head node in the background (`nohup`/`tmux`), because the downloads are long.
- The ELM key never goes on the cluster; ASR does not need it. Keep large data, caches and `.env` out of git.

Each session:

```bash
ssh <UUN>@ssh.inf.ed.ac.uk      # then: ssh mlp.inf.ed.ac.uk
cd ~/edin-mls-26-autumn && git pull
srun -p Teaching --gres=gpu:1 --cpus-per-task=2 --mem=16G --time=01:00:00 --pty bash
export PATH=/opt/cuda-12.8.0/bin:$PATH && conda activate mls-asr
```

## Teammate onboarding (each person has their own cluster home)

Cluster homes are per user and private, so nothing one person installs is visible to the other. Each person needs, on the head node (via the gateway):

1. `git clone https://github.com/ed-aisys/edin-mls-26-autumn.git ~/edin-mls-26-autumn` and `git clone https://github.com/Rufiz16980/MLS.git ~/MLS`
2. `cd ~/edin-mls-26-autumn/hw1-asr && (yes | nohup bash setup_env.sh > ~/setup_env.log 2>&1 &)` (about 20 GB, background; needed for kernels and tests)
3. `chmod 700 ~` and check `stat -c '%a' ~`

The RAG data (`rag-assets.zip`, 196 MB) is only needed on the cluster by whoever runs the one-off ASR transcription of the 475 audio queries. The other person does not need to download it there: the cached transcripts (small text files) are shared instead, and RAG runs on each laptop via `scripts/setup_rag_local.sh`.
Do not share homes or loosen permissions to avoid repeating setup; keep it private and repeat the script.
