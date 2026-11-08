# GPU cluster access (Informatics Teaching partition)

Do this once to confirm it works, and each time you want a GPU. Wherever you see `s1234567`, use your own DICE login.

**Which login?** The username is your UUN, the `sNNNNNNN` part of your University email login. The password for both hops is your **DICE password**, which is different from your email/EASE password. If you have not set one, set it at https://pp.inf.ed.ac.uk. SSH public keys are blocked on these gateways, so use the password. On eduroam no VPN is needed; off campus use the University VPN.

1. Log in to the gateway (use your email-login username, then your DICE password):

```bash
ssh s1234567@ssh.inf.ed.ac.uk
```

- If you see `Are you sure you want to continue connecting (yes/no/[fingerprint])?`, type `yes` and press Enter.
- At `s1234567@ssh.inf.ed.ac.uk's password:`, type your DICE password. Nothing shows while you type. Press Enter.

2. Hop to the cluster head node (now you are on the gateway):

```bash
ssh mlp.inf.ed.ac.uk
```

- Same two prompts: `yes` if asked, then your DICE password again.
- If it cannot resolve the host or times out, run `ssh icf.inf.ed.ac.uk` instead.

3. Ask for a GPU for 10 minutes:

```bash
srun -p Teaching --gres=gpu:1 --cpus-per-task=1 --mem=2G --time=00:10:00 --pty bash
```

- Your prompt changes to a new machine name once you have a GPU. It can queue for a minute or two.

4. See which GPU you got:

```bash
hostname; nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu,temperature.gpu,driver_version --format=csv; nvidia-smi
```

5. Compile and run a tiny CUDA program:

```bash
export PATH=/opt/cuda-12.8.0/bin:$PATH; printf '#include <cstdio>\n__global__ void k(){printf("hello from GPU thread %%d\\n",threadIdx.x);}\nint main(){k<<<1,4>>>();cudaDeviceSynchronize();}\n' > /tmp/h_$USER.cu && nvcc -o /tmp/h_$USER /tmp/h_$USER.cu && /tmp/h_$USER
```

You should see four lines: `hello from GPU thread 0` to `3`.

6. Make a virtual environment, use it, then delete it:

```bash
python3 -m venv /tmp/venv_$USER && source /tmp/venv_$USER/bin/activate && python -c "import sys,os;print(sys.version.split()[0], os.cpu_count(),'CPUs')"; deactivate; rm -rf /tmp/venv_$USER /tmp/h_$USER /tmp/h_$USER.cu
```

7. Give the GPU back and log out:

```bash
exit    # leaves the GPU node; the GPU goes back to the pool
exit    # leaves the head node
exit    # leaves the gateway
```

If something goes wrong, paste the output to the team:
- `Permission denied (publickey,...)` usually means the DICE password was wrong.
- If `srun` says invalid partition or account, your course access is not enabled yet; this needs a ticket to Informatics computing support.

Course rules: one GPU per person (`--gres=gpu:1`), one GPU shell or job at a time, `exit` when finished. Compare timings only on the same GPU type; our reference GPU is the RTX 2080 Ti (`--gres=gpu:nvidia_geforce_rtx_2080_ti:1`). GPU nodes may have no internet: download things on the head node.
