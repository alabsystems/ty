#!/usr/bin/env python3
# Copyright 2026 Andrew Yates
# Licensed under the Apache License, Version 2.0
#
# Turn one or more mcc_oracle_eval.py per-cell CSVs (columns:
# model,examination,grade,detail,wall_s) into an MCC-style scorecard:
# per-examination CORRECT / WRONG / CC, solve%, wrong-rate, and the MCC points
# contribution (ScoreValue * CORRECT - 2 * ScoreValue * WRONG) using the
# per-examination ScoreValue from docs/mcc-2026/scoring-formula.md.
#
#   scripts/mcc_retro_scorecard.py runs1.csv [runs2.csv ...]
#
# This is a SAMPLE scorecard (the oracle eval runs a stratified subset), so the
# headline numbers are SOLVE% and WRONG-RATE (comparable to the official 2026
# confidence/coverage), not absolute grand-total points (which need the full
# 1855-model corpus). It exists to answer the retrospective question: at HEAD,
# on held-out models, is TY (a) SOUND (wrong-rate -> 0, vs the submitted 0.33%)
# and (b) recovering coverage.
import csv, sys, collections

# ScoreValue = 16 / (values per examination instance). GlobalProperties sub-exams
# are single-value (N=1 -> 16); StateSpace N=4 -> 4; the 16-formula exams N=16 -> 1.
SCORE_VALUE = {
    "StateSpace": 4.0,
    "ReachabilityDeadlock": 16.0, "OneSafe": 16.0, "QuasiLiveness": 16.0,
    "StableMarking": 16.0, "Liveness": 16.0,
    "UpperBounds": 1.0,
    "ReachabilityCardinality": 1.0, "ReachabilityFireability": 1.0,
    "CTLCardinality": 1.0, "CTLFireability": 1.0,
    "LTLCardinality": 1.0, "LTLFireability": 1.0,
}
CATEGORY = {
    "StateSpace": "StateSpace",
    "ReachabilityDeadlock": "GlobalProperties", "OneSafe": "GlobalProperties",
    "QuasiLiveness": "GlobalProperties", "StableMarking": "GlobalProperties",
    "Liveness": "GlobalProperties",
    "UpperBounds": "UpperBounds",
    "ReachabilityCardinality": "Reachability", "ReachabilityFireability": "Reachability",
    "CTLCardinality": "CTL", "CTLFireability": "CTL",
    "LTLCardinality": "LTL", "LTLFireability": "LTL",
}

def main(paths):
    tally = collections.defaultdict(lambda: collections.Counter())
    for p in paths:
        with open(p) as f:
            r = csv.reader(f)
            next(r, None)  # header
            for row in r:
                if len(row) < 3: continue
                exam, grade = row[1], row[2]
                tally[exam][grade] += 1

    print(f"{'examination':26} {'CORR':>5} {'WRONG':>5} {'CC':>5} {'graded':>6} "
          f"{'solve%':>7} {'wrong%':>7} {'points':>9}")
    print("-" * 86)
    cat_pts = collections.defaultdict(float)
    cat_corr = collections.defaultdict(int); cat_wrong = collections.defaultdict(int)
    cat_graded = collections.defaultdict(int)
    tot = collections.Counter()
    for exam in sorted(tally, key=lambda e: (CATEGORY[e], e)):
        c = tally[exam]
        corr, wrong, cc = c["CORRECT"], c["WRONG"], c["CC"] + c["UNJUDGED"]
        graded = corr + wrong + cc
        if not graded: continue
        sv = SCORE_VALUE[exam]
        pts = sv * corr - 2 * sv * wrong
        solve = 100.0 * corr / graded if graded else 0.0
        wr = 100.0 * wrong / (corr + wrong) if (corr + wrong) else 0.0
        print(f"{exam:26} {corr:5} {wrong:5} {cc:5} {graded:6} {solve:6.1f}% {wr:6.2f}% {pts:9.0f}")
        cat = CATEGORY[exam]
        cat_pts[cat] += pts; cat_corr[cat] += corr; cat_wrong[cat] += wrong
        cat_graded[cat] += graded
        tot["CORRECT"] += corr; tot["WRONG"] += wrong; tot["CC"] += cc

    print("-" * 86)
    print(f"{'BY CATEGORY':26} {'CORR':>5} {'WRONG':>5} {'':>5} {'graded':>6} {'solve%':>7} {'wrong%':>7} {'points':>9}")
    for cat in ["StateSpace", "GlobalProperties", "UpperBounds", "Reachability", "CTL", "LTL"]:
        if not cat_graded[cat]: continue
        corr, wrong, graded = cat_corr[cat], cat_wrong[cat], cat_graded[cat]
        solve = 100.0 * corr / graded
        wr = 100.0 * wrong / (corr + wrong) if (corr + wrong) else 0.0
        print(f"{cat:26} {corr:5} {wrong:5} {'':5} {graded:6} {solve:6.1f}% {wr:6.2f}% {cat_pts[cat]:9.0f}")

    g = tot["CORRECT"] + tot["WRONG"] + tot["CC"]
    print("-" * 86)
    print(f"TOTAL graded cells: {g}   CORRECT {tot['CORRECT']}   WRONG {tot['WRONG']}   "
          f"CC {tot['CC']}   solve {100.0*tot['CORRECT']/g:.1f}%   "
          f"wrong-rate {100.0*tot['WRONG']/(tot['CORRECT']+tot['WRONG'] or 1):.3f}%")
    print(f"\nSubmitted-2026 reference: 99.674% confidence (0.326% wrong-rate), 147 wrong answers.")
    print(f"TARGET (T1 soundness parity): wrong-rate 0.000%.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: mcc_retro_scorecard.py runs1.csv [runs2.csv ...]"); sys.exit(2)
    main(sys.argv[1:])
