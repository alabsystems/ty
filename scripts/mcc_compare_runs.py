#!/usr/bin/env python3
# Copyright 2026 Andrew Yates
# Compare a ty-mcc full-benchmark run against the official MCC summary CSV.
#
#   scripts/mcc_compare_runs.py <baseline_summary.csv> <new_run.csv>
#
# baseline = official summary_TY.csv (14 cols; results=col7, status=col13).
# new      = scripts/mcc_full_benchmark.sh output (cols: tool,Input,Examination,
#            results,wall_ms,status) — OR another official-format summary.
#
# Reports, over the (model, examination) cells present in BOTH:
#   recovered  : baseline CC/timeout  -> new ANSWER/PARTIAL
#   improved-? : baseline PARTIAL '?' -> new full ANSWER
#   lost       : baseline ANSWER      -> new CC/timeout
#   REGRESSION : both definite single-verdict answers that DISAGREE (soundness!)
# plus per-examination deltas and the net answer count change. Exit non-zero if
# any value regression is found.
import csv, sys

def classify(results, status):
    r = (results or "").strip()
    s = (status or "").strip().lower()
    if s in ("timeout", "dnf"): return "TIMEOUT"
    if "DO_NOT_COMPETE" in r: return "DNC"
    if r == "CC" or "CANNOT_COMPUTE" in r: return "CC"
    if not r or r == "NONE": return "EMPTY"
    if ":?" in r or " ?" in r: return "PARTIAL"
    if r[0].isdigit() or r.startswith(("BOOL", "NUM", "FORMULA", "STATE_SPACE")):
        return "ANSWER"
    return "OTHER"

def load(path):
    cells = {}
    with open(path) as f:
        for row in csv.reader(f):
            if not row or row[0] != "TY" or len(row) < 4:
                continue
            model, exam, results = row[1], row[2], row[3]
            status = row[12] if len(row) >= 13 else (row[5] if len(row) >= 6 else "")
            # official summary: results in col7 (idx6), status col13 (idx12)
            if len(row) >= 14:
                results, status = row[6], row[12]
            cells[(model, exam)] = (results.strip(), classify(results, status))
    return cells

def verdict(s):
    u = (s or "").upper()
    if "BOOL:TRUE" in u or " TRUE" in u: return "TRUE"
    if "BOOL:FALSE" in u or " FALSE" in u: return "FALSE"
    return None

SINGLE = {"ReachabilityDeadlock","OneSafe","QuasiLiveness","StableMarking","Liveness"}

def main():
    if len(sys.argv) != 3:
        print("usage: mcc_compare_runs.py <baseline.csv> <new.csv>"); sys.exit(2)
    base, new = load(sys.argv[1]), load(sys.argv[2])
    common = sorted(set(base) & set(new))
    rec = imp = lost = reg = same = 0
    per_ex = {}
    recoveries, regressions = [], []
    for key in common:
        m, e = key
        b_res, b_cls = base[key]; n_res, n_cls = new[key]
        d = per_ex.setdefault(e, {"rec":0,"lost":0,"reg":0})
        if b_cls in ("CC","TIMEOUT","EMPTY") and n_cls in ("ANSWER","PARTIAL"):
            rec += 1; d["rec"] += 1; recoveries.append((m,e,b_cls,"->",n_cls))
        elif b_cls == "PARTIAL" and n_cls == "ANSWER":
            imp += 1
        elif b_cls in ("ANSWER","PARTIAL") and n_cls in ("CC","TIMEOUT","EMPTY"):
            lost += 1; d["lost"] += 1
        elif b_cls == "ANSWER" and n_cls == "ANSWER":
            same += 1
            if e in SINGLE and "?" not in b_res:
                bv, nv = verdict(b_res), verdict(n_res)
                if bv and nv and bv != nv:
                    reg += 1; d["reg"] += 1; regressions.append((m,e,b_res,n_res))

    print(f"cells compared (in both runs): {len(common)}")
    print(f"  recovered (CC/timeout -> answer): {rec}")
    print(f"  partial '?' -> full answer:       {imp}")
    print(f"  lost (answer -> CC/timeout):      {lost}")
    print(f"  unchanged answers:                {same}")
    print(f"  VALUE REGRESSIONS (soundness):    {reg}")
    print("\nper-examination (recovered / lost / regressions):")
    for e in sorted(per_ex):
        d = per_ex[e]
        if d["rec"] or d["lost"] or d["reg"]:
            print(f"  {e:26} +{d['rec']:<4} -{d['lost']:<4} reg={d['reg']}")
    if regressions:
        print("\n!!! VALUE REGRESSIONS (investigate — possible wrong answers) !!!")
        for r in regressions: print("   ", r)
    net = rec + imp - lost
    print(f"\nNET answer change vs baseline: {net:+d}  (recovered+improved {rec+imp}, lost {lost})")
    sys.exit(1 if reg else 0)

if __name__ == "__main__":
    main()
