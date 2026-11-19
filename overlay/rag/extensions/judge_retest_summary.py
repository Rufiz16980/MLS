"""Agreement between the original judge scores and a second judge pass on the same stored answers (no ELM calls).

  python rag/extensions/judge_retest_summary.py --csv outputs/rag/judge_retest.csv
The CSV (outputs/rag/judge_retest.csv) has one row per re-judged (query, k) answer: 36 answers that scored 0
although a relevant record was in the prompt, and 20 non-zero answers drawn at random (seed 0).
Pairs for which the judge returned malformed JSON are 'unscorable' and left out of the statistics.
"""
import argparse
import csv


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", required=True)
    args = ap.parse_args()
    rows = list(csv.DictReader(open(args.csv, encoding="utf-8")))
    print(f"{len(rows)} pairs, {sum(r['status'] == 'scored' for r in rows)} scored, "
          f"unscorable: {[(r['query_id'], r['k']) for r in rows if r['status'] != 'scored']}")
    subs = ("correctness", "grounding", "completeness")
    for group in ("zero", "nonzero", "all"):
        sel = [r for r in rows if r["status"] == "scored" and group in (r["group"], "all")]
        if not sel:
            continue
        old = [float(r["old_total"]) for r in sel]
        new = [float(r["new_total"]) for r in sel]
        same = sum(abs(a - b) < 1e-3 for a, b in zip(old, new))
        mad = sum(abs(a - b) for a, b in zip(old, new)) / len(sel)
        line = f"{group:8s} n={len(sel):2d}  total identical {same}/{len(sel)}  mean |change| {mad:.3f}  " \
               f"mean {sum(old) / len(sel):.3f} -> {sum(new) / len(sel):.3f}"
        for s in subs:
            ident = sum(r[f"old_{s}"] == r[f"new_{s}"] for r in sel)
            line += f"  {s} {ident}/{len(sel)}"
        print(line)
        up = [r for r in sel if float(r["new_total"]) > float(r["old_total"]) + 1e-3 and group == "zero"]
        if group == "zero":
            print(f"         of the stored zeros, {len(up)} came back above 0: "
                  + ", ".join(f"{r['query_id'][:10]} k={r['k']} {float(r['new_total']):.2f}" for r in up))


if __name__ == "__main__":
    main()
