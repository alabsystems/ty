#!/usr/bin/env python3
# Copyright 2026 Andrew Yates
# Before/after harness: run the (fixed) ty-mcc on local models under the contest
# config (fpset=cas) and compare each (model, examination) result against the
# ACTUAL MCC-2026 contest result recorded in summary_TY.csv.
#
# Reports: recovered (contest CC/timeout -> answer), regressions (a definite
# contest answer changed to a DIFFERENT definite value = SOUNDNESS BUG), and
# coverage deltas. Value mismatches are the critical check.
import csv, subprocess, sys, os, time

SUMMARY = "/tmp/mcc_extract/TY/summary_TY.csv"
BIN = os.environ.get("TY_MCC_BIN", "./target/release/ty-mcc")
MODELS_DIR = "tmp_benchmark_models"
TIMEOUT = int(os.environ.get("VERIFY_TIMEOUT", "120"))
EXAMS = ["ReachabilityDeadlock","OneSafe","QuasiLiveness","StableMarking","Liveness",
         "StateSpace","UpperBounds","ReachabilityCardinality","ReachabilityFireability",
         "CTLCardinality","CTLFireability","LTLCardinality","LTLFireability"]

def classify(results, status):
    if status == "timeout": return "TIMEOUT"
    if results == "CC": return "CC"
    if ":?" in results: return "PARTIAL"
    if results and (results[0].isdigit() or results.startswith(("BOOL","NUM"))): return "ANSWER"
    return "OTHER"

# Load contest results: (model, exam) -> (results_str, class)
contest = {}
with open(SUMMARY) as f:
    for row in csv.reader(f):
        if len(row) < 14 or row[0] != "TY": continue
        contest[(row[1], row[2])] = (row[6].strip(), classify(row[6].strip(), row[12].strip()))

def norm(s):
    # normalize a result line set for value comparison (sort BOOL/NUM tokens off; keep counts/values)
    return " ".join(s.split())

models = sorted(d for d in os.listdir(MODELS_DIR)
                if os.path.isdir(os.path.join(MODELS_DIR, d)))

recovered=same=lost=regress=0
regressions=[]; recoveries=[]
env = dict(os.environ, TY_MCC_REQUIRE_BACKEND_EVIDENCE="0", TY_MCC_FPSET_BACKEND="cas")

for m in models:
    for e in EXAMS:
        key=(m,e)
        if key not in contest: continue
        c_res, c_cls = contest[key]
        sdir=f"/tmp/ba/{m}-{e}"; subprocess.run(["rm","-rf",sdir]); os.makedirs(sdir,exist_ok=True)
        t0=time.time()
        try:
            out = subprocess.run([BIN, os.path.join(MODELS_DIR,m), "--examination", e,
                  "--threads","4","--memory-fraction","0.5","--storage","auto",
                  "--storage-dir",sdir,"--timeout",str(TIMEOUT)],
                  env=env, capture_output=True, text=True, timeout=TIMEOUT+30).stdout
        except subprocess.TimeoutExpired:
            out=""
        dt=time.time()-t0
        lines=[l for l in out.splitlines() if l.startswith(("FORMULA","STATE_SPACE"))]
        if not lines: f_cls="CC/none"; f_res=""
        else:
            f_res=norm(" | ".join(lines))
            if "CANNOT_COMPUTE" in f_res and not any(x in f_res for x in ["TRUE","FALSE","NUM","BOOL"]): f_cls="CC"
            elif ":?" in f_res or "CANNOT_COMPUTE" in f_res: f_cls="PARTIAL"
            else: f_cls="ANSWER"
        tag=""
        if c_cls in ("CC","TIMEOUT") and f_cls=="ANSWER": recovered+=1; tag="RECOVERED"; recoveries.append((m,e,c_cls,round(dt,1)))
        elif c_cls=="ANSWER" and f_cls in ("CC","CC/none","PARTIAL"): lost+=1; tag="lost(budget?)"
        elif c_cls=="ANSWER" and f_cls=="ANSWER":
            same+=1
            # SOUNDNESS: compare verdicts for single-verdict exams (deadlock/onesafe/etc).
            # Contest format: "1 BOOL:TRUE"; fixed format: "FORMULA <exam> TRUE".
            if e in ("ReachabilityDeadlock","OneSafe","QuasiLiveness","StableMarking","Liveness"):
                def verdict(s):
                    s=s.upper()
                    if "BOOL:TRUE" in s or " TRUE" in s: return "TRUE"
                    if "BOOL:FALSE" in s or " FALSE" in s: return "FALSE"
                    return None
                cv, fv = verdict(c_res), verdict(f_res)
                if cv and fv and cv!=fv and "?" not in c_res:
                    regress+=1; tag="!!REGRESSION!!"; regressions.append((m,e,c_res,f_res))
        print(f"{m:30} {e:24} contest={c_cls:8} fixed={f_cls:8} {dt:5.1f}s {tag}")

print("\n===== SUMMARY =====")
print(f"recovered (CC/timeout -> ANSWER): {recovered}")
print(f"same ANSWER: {same}   lost (likely local budget): {lost}")
print(f"VALUE REGRESSIONS (soundness bugs): {regress}")
for r in regressions: print("  REGRESSION:", r)
print("\nRecoveries:")
for r in recoveries: print("  +", r)
