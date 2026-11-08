#!/usr/bin/env python3
"""Frozen-file guard for the course code. Lives OUTSIDE the course repo; never edits it.

  guard.py check    --course DIR     verify frozen files + task-file interfaces
  guard.py run      --course DIR -- CMD...   check first, then run CMD (refuses if check fails)
  guard.py manifest --course DIR --commit SHA   (maintainers) regenerate frozen.json from git objects
  guard.py selftest                  validate frozen.json and the overlay (no course clone needed)

Rules checked (from the coursework handout):
  * every frozen file is byte-identical to the pristine course commit;
  * the five task files keep every original __global__/__device__/ASR_API function
    name and signature (launch_* entry points included);
  * task files do not use cuBLAS/cuDNN/CUTLASS/Thrust/PyTorch.
Exit code 0 = intact, 1 = violation, 2 = usage error. No file contents are ever printed.
"""
import argparse, hashlib, json, os, re, subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "frozen.json"

TASK_FILES = [
    "hw1-asr/csrc/task1_rope.cu", "hw1-asr/csrc/task2_reduce.cuh", "hw1-asr/csrc/task2_reduce.cu",
    "hw1-asr/csrc/task3_gemm.cu", "hw1-asr/csrc/task4_attention.cu",
]
# Frozen = everything under these prefixes, minus the task files and exclusions.
FROZEN_PREFIXES = [
    "hw1-asr/", "rag/src/", "rag/scripts/", "rag/tests/", "rag/requirements.txt",
    "rag/.env.example", "rag/.gitignore", "report/format_checker/", "report/latex_template/",
    "cuda-tutorial/Makefile", "LICENSE",
]
EXCLUDE = set(TASK_FILES) | {"report/latex_template/report.tex", "report/latex_template/references.bib"}
EXCLUDE_PREFIXES = ["hw1-asr/docs/"]
BANNED = re.compile(r'#\s*include\s*[<"](cublas|cublasLt|cudnn|cutlass|thrust|torch|ATen)[^>"]*[>"]', re.I)
HEADER = re.compile(r'(?:ASR_API|__global__|__device__)[^;{}]*?\)\s*(?=\{)', re.S)

def sha(b): return hashlib.sha256(b).hexdigest()
def norm(s): return re.sub(r"\s+", " ", s).strip()
def strip_comments(t): return re.sub(r"//[^\n]*|/\*.*?\*/", "", t, flags=re.S)
def headers(text): return sorted({norm(m.group(0)) for m in HEADER.finditer(strip_comments(text))})

def git(course, *args):
    return subprocess.run(["git", "-c", "safe.directory=*", "-c", "core.hooksPath=/dev/null", "-C", str(course), *args],
                          check=True, capture_output=True).stdout

def is_frozen(p):
    return (p not in EXCLUDE and not any(p.startswith(x) for x in EXCLUDE_PREFIXES)
            and any(p.startswith(x) for x in FROZEN_PREFIXES))

def cmd_manifest(a):
    course = Path(a.course)
    commit = git(course, "rev-parse", a.commit).decode().strip()
    names = git(course, "ls-tree", "-r", "--name-only", commit).decode().split("\n")
    files = {n: sha(git(course, "show", f"{commit}:{n}")) for n in names if n and is_frozen(n)}
    tasks = {n: headers(git(course, "show", f"{commit}:{n}").decode()) for n in TASK_FILES}
    MANIFEST.write_text(json.dumps({"course_repo": "https://github.com/ed-aisys/edin-mls-26-autumn",
                                    "commit": commit, "frozen_files": files, "task_headers": tasks}, indent=1, sort_keys=True) + "\n")
    print(f"manifest: {len(files)} frozen files, {len(tasks)} task files, commit {commit[:7]}")

def load():
    return json.loads(MANIFEST.read_text())

def check_task(text, want):
    bad = []
    have = headers(text)
    bad += [f"missing/changed function header: {h[:90]}" for h in want if h not in have]
    bad += ["banned library include: " + m.group(1) for m in BANNED.finditer(strip_comments(text))]
    return bad

def cmd_check(a):
    m, course, bad = load(), Path(a.course), []
    if not (course / "hw1-asr").is_dir():
        print(f"guard: {course} is not a course checkout"); return 2
    try:
        head = git(course, "rev-parse", "HEAD").decode().strip()
        if head != m["commit"]:
            print(f"guard: note: course HEAD {head[:7]} != pinned {m['commit'][:7]} (frozen hashes are from the pinned commit)")
    except Exception:
        pass
    for rel, want in m["frozen_files"].items():
        p = course / rel
        if not p.is_file(): bad.append(f"MISSING {rel}")
        elif sha(p.read_bytes()) != want: bad.append(f"MODIFIED {rel}")
    for rel, want in m["task_headers"].items():
        p = course / rel
        if not p.is_file(): bad.append(f"MISSING {rel}"); continue
        bad += [f"{rel}: {x}" for x in check_task(p.read_text(errors="replace"), want)]
    if bad:
        print("GUARD FAILED:\n  " + "\n  ".join(bad)); return 1
    print(f"guard OK: {len(m['frozen_files'])} frozen files intact, {len(m['task_headers'])} task files conform")
    return 0

def cmd_run(a):
    if not a.cmd: print("guard run: no command"); return 2
    if a.cmd[0] == "--": a.cmd = a.cmd[1:]
    rc = cmd_check(a)
    if rc: print("guard: refusing to run the command"); return rc
    r = subprocess.run(a.cmd)
    rc = cmd_check(a)  # re-check afterwards: the command must not have touched frozen files
    return r.returncode or rc

def cmd_selftest(a):
    m = load(); bad = []
    if not m["frozen_files"] or len(m["task_headers"]) != len(TASK_FILES): bad.append("manifest incomplete")
    root = HERE.parent.parent
    for rel in TASK_FILES:
        p = root / "overlay" / rel
        if p.is_file():
            bad += [f"overlay/{rel}: {x}" for x in check_task(p.read_text(errors="replace"), m["task_headers"][rel])]
        else:
            bad.append(f"overlay/{rel} missing")
    for p in (root / "overlay").rglob("*"):
        rel = p.relative_to(root / "overlay").as_posix()
        if p.is_file() and rel in m["frozen_files"]:
            bad.append(f"overlay contains frozen file {rel}")
    if bad: print("SELFTEST FAILED:\n  " + "\n  ".join(bad)); return 1
    print("selftest OK"); return 0

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="c", required=True)
    for name in ("check", "run", "manifest"):
        s = sp.add_parser(name); s.add_argument("--course", default=os.environ.get("MLS_COURSE_DIR", "edin-mls-26-autumn"))
        if name == "manifest": s.add_argument("--commit", default="HEAD")
        if name == "run": s.add_argument("cmd", nargs=argparse.REMAINDER)
    sp.add_parser("selftest")
    a = ap.parse_args()
    sys.exit({"check": cmd_check, "run": cmd_run, "selftest": cmd_selftest}.get(a.c, cmd_manifest)(a))

if __name__ == "__main__":
    main()
