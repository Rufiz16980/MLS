#!/usr/bin/env python3
"""Bulk kernel ablations for hw1-asr. Run from hw1-asr/ on a GPU node:

    python ablation/run_ablations.py --stage check      # correctness: make test, sanitizer, full model
    python ablation/run_ablations.py --stage sweep      # every variant below: build + test + time
    python ablation/run_ablations.py --stage e2e        # full-model runs with --profile
    python ablation/run_ablations.py --stage best       # combined best config + its parts, full model (about 10 min)
    python ablation/run_ablations.py --stage race       # barrier-removal ablations under racecheck
    python ablation/run_ablations.py --stage all        # check, sweep, e2e (race is opt-in)

Each variant is the same five task files built with different -D flags into its own
build/ablation/<name>/ directory, so variants never mix and the normal build/sm_XX is
untouched (the e2e stage swaps the library in and always restores it).
Nothing outside build/ and ablation/results/ is written. No course file is modified.

Output: ablation/results/<UTC time>/  raw logs, results.csv, summary.md (paste that one).
"""
import argparse, csv, datetime, os, re, shutil, statistics, subprocess, sys
from pathlib import Path

HW = Path(__file__).resolve().parent.parent          # hw1-asr/
BASE_FLAGS = "-O3 -std=c++17 -lineinfo -diag-suppress 177"  # the Makefile's default NVCCFLAGS
TEST = {1: "test_task1_rope", 2: "test_task2_reduce", 3: "test_task3_gemm", 4: "test_task4_attention"}

# name: (task, extra nvcc flags, what it tests). Defaults: ROPE 256, REDUCE 256, TILE 16,
# ws pad 1, coalesced W load, ATTN_WARPS 4, k pad 1, accurate maths.
VARIANTS = {
    "default":          (0, "", "the submitted configuration"),
    "rope_t32":         (1, "-DASR_ROPE_THREADS=32", "T1 Try 1: block size"),
    "rope_t64":         (1, "-DASR_ROPE_THREADS=64", "T1 Try 1"),
    "rope_t128":        (1, "-DASR_ROPE_THREADS=128", "T1 Try 1"),
    "rope_t512":        (1, "-DASR_ROPE_THREADS=512", "T1 Try 1"),
    "rope_t1024":       (1, "-DASR_ROPE_THREADS=1024", "T1 Try 1"),
    "rope_fastmath":    (1, "--use_fast_math", "T1 Try 3: fast sin/cos (a case should FAIL)"),
    "reduce_t32":       (2, "-DASR_REDUCE_THREADS=32", "T2 Try 1: block size"),
    "reduce_t64":       (2, "-DASR_REDUCE_THREADS=64", "T2 Try 1"),
    "reduce_t128":      (2, "-DASR_REDUCE_THREADS=128", "T2 Try 1"),
    "reduce_t512":      (2, "-DASR_REDUCE_THREADS=512", "T2 Try 1"),
    "reduce_t1024":     (2, "-DASR_REDUCE_THREADS=1024", "T2 Try 1"),
    "rms_atomic":       (2, "-DASR_ABL_RMS_ATOMIC", "T2 Try 4: atomicAdd instead of shuffles"),
    "softmax_3pass":    (2, "-DASR_ABL_SOFTMAX_3PASS", "T2 Q3: textbook 3-read softmax"),
    "gemm_tile8":       (3, "-DASR_TILE=8", "T3 Try 1: tile size"),
    "gemm_tile32":      (3, "-DASR_TILE=32", "T3 Try 1"),
    "gemm_nopad":       (3, "-DASR_WS_PAD=0", "T3 Try 2: shared-memory bank conflicts"),
    "gemm_wstrided":    (3, "-DASR_ABL_W_STRIDED", "T3 Try 3: uncoalesced W load"),
    "gemm_tile32_nopad":(3, "-DASR_TILE=32 -DASR_WS_PAD=0", "T3 Try 1+2 interaction"),
    "attn_w1":          (4, "-DASR_ATTN_WARPS=1", "T4 Try 1: queries per block"),
    "attn_w2":          (4, "-DASR_ATTN_WARPS=2", "T4 Try 1"),
    "attn_w8":          (4, "-DASR_ATTN_WARPS=8", "T4 Try 1"),
    "attn_w16":         (4, "-DASR_ATTN_WARPS=16", "T4 Try 1"),
    "attn_nopad":       (4, "-DASR_K_PAD=0", "T4 Try 2: bank conflicts in k_tile"),
    "best_t32_w16":     (0, "-DASR_TILE=32 -DASR_ATTN_WARPS=16", "combined: GEMM tile 32 + 16 attention warps"),
}
RACE = {  # correctness ablations: expected to break; only run with --stage race
    "race_no_final_sync":  (2, "-DASR_ABL_NO_FINAL_SYNC", "T2 Try 3"),
    "race_no_second_sync": (3, "-DASR_ABL_NO_SECOND_SYNC", "T3 Try 4"),
}
# Full-model runs: (label, variant whose library is used, --cuda groups, profile?)
E2E = [
    ("torch_only", None, "none", True),
    ("task1", "default", "task1", False),
    ("task1-2", "default", "task1,task2", False),
    ("task1-3", "default", "task1,task2,task3", True),
    ("all_default", "default", "all", True),
    ("all_tile8", "gemm_tile8", "all", False),
    ("all_tile32", "gemm_tile32", "all", False),
    ("all_attn_w8", "attn_w8", "all", False),
]
# Extra full-model runs for --stage best (the combination of the best single changes, plus its parts)
E2E_BEST = [
    ("all_default_repeat", "default", "all", False),
    ("all_attn_w16", "attn_w16", "all", False),
    ("all_tile32_again", "gemm_tile32", "all", False),
    ("all_best_t32_w16", "best_t32_w16", "all", True),
]
LINE = re.compile(r"^(?P<case>.+?)\s+(?P<status>PASS|FAIL)\s+max_err=(?P<err>\S+)"
                  r"(?:\s+(?P<ms>[\d.]+) ms\s+(?P<speed>.*))?$")


def sh(cmd, log, timeout=None, env=None):
    """Run cmd (list), append its output to log, return (returncode, output)."""
    try:
        p = subprocess.run(cmd, cwd=HW, capture_output=True, text=True, timeout=timeout, env=env)
        out, rc = p.stdout + p.stderr, p.returncode
    except subprocess.TimeoutExpired as e:
        out, rc = (e.stdout or "") if isinstance(e.stdout, str) else "", 124
        out += f"\n[timeout after {timeout}s]"
    with open(log, "a") as f:
        f.write(f"$ {' '.join(cmd)}\n{out}\n[exit {rc}]\n\n")
    return rc, out


def sm():
    out = subprocess.run(["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.strip().splitlines()
    if not out:
        sys.exit("No GPU found: run this inside an srun GPU shell.")
    return out[0].replace(".", "")


def build(name, flags, targets, log, jobs):
    bdir = f"build/ablation/{name}"
    t = [f"{bdir}/{x}" for x in targets]
    rc, _ = sh(["make", f"-j{jobs}", f"SM={SM}", f"BUILD={bdir}", f"NVCCFLAGS={BASE_FLAGS} {flags}".strip(), *t], log)
    return rc == 0, HW / bdir


def parse(out):
    rows = []
    for line in out.splitlines():
        m = LINE.match(line.rstrip())
        if m:
            rows.append(m.groupdict())
    return rows


def run_tests(name, task_list, flags, desc, reps, outdir, jobs, writer):
    log = outdir / f"{name}.log"
    ok, bdir = build(name, flags, [TEST[t] for t in task_list], log, jobs)
    if not ok:
        print(f"  {name}: BUILD FAILED (see {log.name})")
        writer.writerow([name, desc, "", "BUILD_FAILED", "", "", "", ""])
        return {}
    res = {}
    for t in task_list:
        per_case = {}
        for r in range(reps):
            rc, out = sh([str(bdir / TEST[t])], log, timeout=900)
            for row in parse(out):
                per_case.setdefault(row["case"], []).append(row)
        for case, rows in per_case.items():
            mss = [float(x["ms"]) for x in rows if x["ms"]]
            status = "PASS" if all(x["status"] == "PASS" for x in rows) else "FAIL"
            med = statistics.median(mss) if mss else None
            spread = (max(mss) - min(mss)) if len(mss) > 1 else 0.0
            writer.writerow([name, desc, t, status, case, f"{med:.4f}" if med is not None else "",
                             f"{spread:.4f}" if mss else "", rows[-1]["speed"] or ""])
            res[(t, case)] = (status, med, rows[-1]["speed"] or "")
        n_fail = sum(1 for (tt, _), v in res.items() if tt == t and v[0] == "FAIL")
        print(f"  {name}: task {t}: {len(per_case)} cases, {n_fail} FAIL")
    return res


def stage_check(outdir, jobs):
    log = outdir / "check.log"
    print("[check] make test (default build)")
    rc, out = sh(["make", f"-j{jobs}", "test"], log, timeout=1800)
    for line in out.splitlines():
        if "ALL PASS" in line or "SOME CASES FAILED" in line or line.startswith("GPU:"):
            print("   ", line)
    print("[check] compute-sanitizer memcheck + racecheck on the small cases")
    for t, b in TEST.items():
        for tool in ("memcheck", "racecheck"):
            rc2, o = sh(["compute-sanitizer", "--tool", tool, f"build/sm_{SM}/{b}", "small"], log, timeout=900)
            summ = [l for l in o.splitlines() if "ERROR SUMMARY" in l or "RACECHECK SUMMARY" in l]
            print(f"    task{t} {tool}: {summ[-1].strip() if summ else f'exit {rc2}'}")
    print("[check] full model on your kernels: ./run.sh")
    rc3, o = sh(["./run.sh"], log, timeout=1800)
    for line in o.splitlines():
        if line.startswith(("Time:", "Transcription:", "Accuracy:", "Status:")):
            print("   ", line)
    return rc == 0 and rc3 == 0


def stage_e2e(outdir, jobs, reps, runs=None):
    log = outdir / "e2e.log"
    lib = HW / f"build/sm_{SM}/libasr_kernels.so"
    sh(["make", "-s", f"-j{jobs}", "lib"], log)
    backup = lib.with_suffix(".so.bak")
    shutil.copy2(lib, backup)
    rows = []
    try:
        for label, variant, groups, prof in (runs or E2E):
            if variant and variant != "default":
                flags = VARIANTS[variant][1]
                ok, bdir = build(variant, flags, ["libasr_kernels.so"], log, jobs)
                if not ok:
                    print(f"  {label}: build failed"); continue
                shutil.copy2(bdir / "libasr_kernels.so", lib)
            else:
                shutil.copy2(backup, lib)
            cmd = [sys.executable, "benchmark.py", "--cuda", groups, "--runs", str(reps)] + (["--profile"] if prof else [])
            print(f"[e2e] {label}: {' '.join(cmd[1:])}")
            rc, out = sh(cmd, log, timeout=1800)
            (outdir / f"e2e_{label}.txt").write_text(out)
            t = re.search(r"Time: ([\d.]+) ms per transcription \(\+/- ([\d.]+) ms\), ([\d.]+) ms per token", out)
            st = re.search(r"Status: (\w+)", out)
            acc = re.search(r"Accuracy: ([\d.]+)%", out)
            row = [label, variant or "-", groups, t.group(1) if t else "", t.group(2) if t else "",
                   t.group(3) if t else "", acc.group(1) if acc else "", st.group(1) if st else f"exit {rc}"]
            rows.append(row)
            print("    ", " | ".join(row))
    finally:
        shutil.copy2(backup, lib)   # the normal build is restored whatever happened
        backup.unlink()
    return rows


def summary(outdir, results, e2e_rows, gpu_line):
    md = [f"# Kernel ablations\n", f"{gpu_line}\n", f"Median of reps; ms per call from the course tests.\n"]
    for t in (1, 2, 3, 4):
        names = ["default"] + [n for n, v in VARIANTS.items() if v[0] == t]
        cases = [c for (tt, c) in results.get("default", {}) if tt == t]
        if not cases:
            continue
        md.append(f"\n## Task {t}\n")
        md.append("| case | " + " | ".join(names) + " |")
        md.append("|---" * (len(names) + 1) + "|")
        for c in cases:
            cells = []
            for n in names:
                v = results.get(n, {}).get((t, c))
                cells.append("-" if v is None or (v[0] == "PASS" and v[1] is None) else ("FAIL" if v[0] == "FAIL" else f"{v[1]:.3f}"))
            md.append(f"| {c.strip()} | " + " | ".join(cells) + " |")
        md.append("\nSpeed column of the default build (from the test's own output):\n")
        for c in cases:
            md.append(f"- {c.strip()}: {results['default'][(t, c)][2]}")
    if e2e_rows:
        md.append("\n## Full model (benchmark.py)\n")
        md.append("| run | library | --cuda | ms/transcription | +/- | ms/token | accuracy % | status |")
        md.append("|---|---|---|---|---|---|---|---|")
        for r in e2e_rows:
            md.append("| " + " | ".join(r) + " |")
        md.append("\nProfiles: e2e_torch_only.txt, e2e_task1-3.txt, e2e_all_default.txt in this folder.")
    (outdir / "summary.md").write_text("\n".join(md) + "\n")


def main():
    global SM
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--stage", choices=["check", "sweep", "e2e", "best", "race", "all"], default="all")
    ap.add_argument("--only", help="comma-separated variant names for the sweep (default: all)")
    ap.add_argument("--reps", type=int, default=3, help="repeats of each test binary / model run")
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 2)
    a = ap.parse_args()
    SM = sm()
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    outdir = HW / "ablation" / "results" / stamp
    outdir.mkdir(parents=True)
    gpu_line = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout.strip()
    gpu_line = re.sub(r"\s*\(UUID:[^)]*\)", "", gpu_line)  # drop the device UUID
    (outdir / "gpu.txt").write_text(gpu_line + "\n")
    print(f"GPU: {gpu_line}  (sm_{SM})\nResults: {outdir.relative_to(HW)}")
    stages = ["check", "sweep", "e2e"] if a.stage == "all" else [a.stage]
    results, e2e_rows = {}, []
    with open(outdir / "results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["variant", "what", "task", "status", "case", "median_ms", "range_ms", "speed"])
        if "check" in stages:
            if not stage_check(outdir, a.jobs):
                print("check stage FAILED: fix correctness before timing anything (see check.log)")
                if a.stage == "all":
                    return 1
        if "sweep" in stages:
            only = set(a.only.split(",")) if a.only else None
            for name, (t, flags, desc) in VARIANTS.items():
                if only and name not in only and name != "default":
                    continue
                tasks = [1, 2, 3, 4] if t == 0 else [t]
                print(f"[sweep] {name}  ({flags or 'default flags'})")
                results[name] = run_tests(name, tasks, flags, desc, a.reps, outdir, a.jobs, w)
                f.flush()
        if "race" in stages:
            for name, (t, flags, desc) in RACE.items():
                log = outdir / f"{name}.log"
                ok, bdir = build(name, flags, [TEST[t]], log, a.jobs)
                if not ok:
                    print(f"  {name}: build failed"); continue
                rc, out = sh([str(bdir / TEST[t])], log, timeout=900)
                fails = [l for l in out.splitlines() if " FAIL " in l or "SOME CASES" in l or "ALL PASS" in l]
                rc2, o2 = sh(["compute-sanitizer", "--tool", "racecheck", str(bdir / TEST[t]), "small"], log, timeout=900)
                summ = [l for l in o2.splitlines() if "RACECHECK SUMMARY" in l or "ERROR SUMMARY" in l]
                print(f"[race] {name}: test -> {fails[-1].strip() if fails else rc}; racecheck -> {summ[-1].strip() if summ else rc2}")
                w.writerow([name, desc, t, "see log", "", "", "", (fails[-1] if fails else "") + " / " + (summ[-1] if summ else "")])
    if "e2e" in stages:
        e2e_rows = stage_e2e(outdir, a.jobs, a.reps)
    if "best" in stages:
        e2e_rows = stage_e2e(outdir, a.jobs, a.reps, E2E_BEST)
    if results or e2e_rows:
        summary(outdir, results, e2e_rows, gpu_line)
        print(f"\nDone. Paste this file:  cat {(outdir / 'summary.md').relative_to(HW)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
