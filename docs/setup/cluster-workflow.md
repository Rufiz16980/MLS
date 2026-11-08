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
