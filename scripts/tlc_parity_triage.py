#!/usr/bin/env python3
# Copyright 2026 Andrew Yates
# Author: Andrew Yates <andrewyates.name@gmail.com>
# Licensed under the Apache License, Version 2.0
"""Apples-to-apples TY-vs-TLC triage over the eligible strict corpus.

This is a **triage** sweep, not strict evidence. It exists to answer the
question the retained evidence cannot: *which of the 141 eligible rows is TY
actually still losing?* Roughly 100 of them have no recorded TY performance
number anywhere in the repo, so the loser list is currently unknown.

It deliberately runs the **count-verification arm** --
``--bfs-only --no-reduction --workers 1`` -- because that is the arm where TY
gets no reduction TLC does not also get. Winning here is an *algorithmic*
claim; winning only in the production arm would mean acceleration is masking a
fundamental inefficiency. See docs/perf/adversarial-acceptance-standard.md A5.

What it is NOT (and must never be presented as):
  * cgroup process-tree memory -- this samples direct-process max RSS;
  * six paired repetitions -- default is 2, enough to rank, not to publish;
  * a clean-room machine contract.

Strict evidence comes from ``ty supremacy matrix-campaign-plan`` and the
fail-closed Linux launcher. This just tells us where to aim.

Usage:
    scripts/tlc_parity_triage.py --out reports/triage.csv [--filter Foo]
                                 [--runs 2] [--timeout 900] [--cpu 9]
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
MANIFEST = REPO / "tests/tlc_comparison/strict_corpus_manifest.json"
VARIANTS = REPO / "tests/tlc_comparison/parity_variants"
CORPUS = Path(os.environ.get("TLAPLUS_EXAMPLES", Path.home() / "tlaplus-examples"))
if CORPUS.name != "specifications":
    CORPUS = CORPUS / "specifications"

TLC_JAR = Path(os.environ.get("TLC_JAR", Path.home() / "tlaplus/tytools.jar"))
COMMUNITY = Path.home() / "tlaplus/CommunityModules.jar"
# Upstream proof library first; 25 eligible rows do not parse without one.
PROOF_LIB = Path.home() / "tlaplus/tla-library"
STUB_LIB = REPO / "test_specs/tla_library"
TY_BIN = REPO / "target/release/ty"


def tla_library() -> Path | None:
    for cand in (PROOF_LIB, STUB_LIB):
        if cand.is_dir():
            return cand
    return None


def eligible_rows(filt: str | None) -> list[dict]:
    manifest = json.loads(MANIFEST.read_text())
    excluded = set(manifest["eligibility"]["exclusions"])
    rows = [r for r in manifest["rows"] if r["cfg_path"] not in excluded]
    if filt:
        rows = [r for r in rows if filt in r["name"] or filt in r["cfg_path"]]
    return rows


def cfg_for(row: dict) -> Path:
    """Frozen symmetry-free variant when one exists, else the corpus cfg.

    Two rows (MCKVsnap, BufferedRandomAccessFile) declare SYMMETRY alongside a
    property needing the genuine liveness checker. TY soundly refuses that orbit
    quotient while TLC applies it, so on the stock cfg the tools do *different
    work* and no comparison is meaningful. Both tools run the identical frozen
    variant instead.
    """
    variant = VARIANTS / f"{row['name']}.nosym.cfg"
    return variant if variant.is_file() else CORPUS / row["cfg_path"]


def run_timed(argv: list[str], cwd: Path, timeout: int, cpu: int | None):
    """Run one tool, returning (wall_seconds, max_rss_kb, stdout+stderr, rc)."""
    wrapped = ["/usr/bin/time", "-v"]
    if cpu is not None:
        wrapped = ["taskset", "-c", str(cpu)] + wrapped
    start = time.monotonic()
    try:
        proc = subprocess.run(
            wrapped + argv, cwd=cwd, timeout=timeout,
            capture_output=True, text=True,
        )
    except subprocess.TimeoutExpired:
        return None, None, "TIMEOUT", 124
    wall = time.monotonic() - start
    combined = (proc.stdout or "") + (proc.stderr or "")
    rss = None
    m = re.search(r"Maximum resident set size \(kbytes\): (\d+)", combined)
    if m:
        rss = int(m.group(1))
    return wall, rss, combined, proc.returncode


def tlc_argv(tla: Path, cfg: Path) -> list[str]:
    cp = str(TLC_JAR)
    if COMMUNITY.is_file():
        cp += ":" + str(COMMUNITY)
    argv = ["java"]
    lib = tla_library()
    if lib:
        argv.append(f"-DTLA-Library={lib}")
    # ActiveProcessorCount=1 stops the JVM parallelising GC/JIT behind a
    # -workers 1 flag, which otherwise makes TLC's wall clock artificially low.
    # Heap policy is left at TLC's DEFAULT on purpose: pinning -Xmx or forcing
    # SerialGC would be us choosing a pessimal configuration for TLC, which the
    # acceptance standard forbids (A3).
    argv += ["-XX:ActiveProcessorCount=1", "-cp", cp, "tlc2.TLC",
             "-workers", "1", "-config", str(cfg), tla.name]
    return argv


def ty_argv(tla: Path, cfg: Path) -> list[str]:
    return [str(TY_BIN), "check", tla.name, "-c", str(cfg),
            "--bfs-only", "--no-reduction", "--workers", "1", "--force"]


# TLC's FINAL summary line begins with the count at column 0:
#     131072 states generated, 65536 distinct states found, 0 states left on queue.
# Its in-flight progress lines carry the same words but are prefixed:
#     Progress(1) at ...: 82,387 states generated (82,387 s/min), 65,536 distinct ...
# Anchoring at `^` therefore structurally excludes progress lines. A naive
# unanchored `re.search` matches the FIRST progress line instead and silently
# reports partial counts as final ones -- which manufactured four bogus
# "parity mismatches" before this was caught (GameOfLife's real generated count
# is 131072, exactly matching TY, not the 85727 a progress line reported).
_TLC_FINAL = re.compile(
    r"^\s*([\d,]+) states generated, ([\d,]+) distinct states found", re.M
)


def parse_tlc(out: str) -> dict:
    d = {}
    matches = _TLC_FINAL.findall(out)
    if matches:
        gen, dist = matches[-1]  # last = final summary, even if format shifts
        d["generated"] = int(gen.replace(",", ""))
        d["distinct"] = int(dist.replace(",", ""))
    if re.search(r"^Error:", out, re.M) or "Errors: 1" in out:
        d["error"] = True
    # TLC's seen-set is fingerprint-only, so it can silently conflate distinct
    # states. Record when it says so -- this is exactly the soundness premium
    # TY pays for with collision-proof payload witnesses.
    if "two distinct states had the same fingerprint" in out:
        d["fp_collision_warning"] = True
    return d


def parse_ty(out: str) -> dict:
    d = {}
    m = re.search(r"States found:\s+(\d+)", out)
    if m:
        d["distinct"] = int(m.group(1))
    m = re.search(r"States generated:\s+(\d+)", out)
    if m:
        d["generated"] = int(m.group(1))
    if re.search(r"^Error", out, re.M):
        d["error"] = True
    return d


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--filter")
    ap.add_argument("--runs", type=int, default=2, help="paired repetitions")
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--cpu", type=int, default=None,
                    help="pin both tools to this logical CPU (use ONE core class)")
    args = ap.parse_args()

    if not TY_BIN.is_file():
        print(f"error: {TY_BIN} missing; build it first", file=sys.stderr)
        return 2
    if not TLC_JAR.is_file():
        print(f"error: {TLC_JAR} missing; run `ty install-tlc install`", file=sys.stderr)
        return 2
    lib = tla_library()
    if lib == STUB_LIB:
        print("warning: using the repo's FIRST-PARTY stub TLA library; run "
              "`ty install-tlc proof-library` for upstream", file=sys.stderr)
    elif lib is None:
        print("warning: no TLA library resolved; 25 rows will fail to parse",
              file=sys.stderr)

    rows = eligible_rows(args.filter)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["spec", "cfg", "variant", "verdict", "parity",
              "ty_distinct", "tlc_distinct", "ty_generated", "tlc_generated",
              "ty_wall", "tlc_wall", "time_ratio", "ty_rss_kb", "tlc_rss_kb",
              "mem_ratio", "both_win", "tlc_fp_collision", "note"]

    with out_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for idx, row in enumerate(rows, 1):
            tla = CORPUS / row["tla_path"]
            cfg = cfg_for(row)
            cwd = tla.parent
            rec = {"spec": row["name"], "cfg": row["cfg_path"],
                   "variant": "yes" if cfg.parent == VARIANTS else "no"}
            print(f"[{idx}/{len(rows)}] {row['name']}", flush=True)

            ty_walls, tlc_walls, ty_rsses, tlc_rsses = [], [], [], []
            ty_counts = tlc_counts = {}
            note = ""
            for rep in range(args.runs):
                # Alternate order so neither tool always runs cold.
                order = ["tlc", "ty"] if rep % 2 == 0 else ["ty", "tlc"]
                for tool in order:
                    argv = tlc_argv(tla, cfg) if tool == "tlc" else ty_argv(tla, cfg)
                    wall, rss, out, rc = run_timed(argv, cwd, args.timeout, args.cpu)
                    if wall is None:
                        note = f"{tool}_timeout"
                        continue
                    if tool == "tlc":
                        tlc_walls.append(wall); tlc_rsses.append(rss)
                        tlc_counts = parse_tlc(out) or tlc_counts
                    else:
                        ty_walls.append(wall); ty_rsses.append(rss)
                        ty_counts = parse_ty(out) or ty_counts

            def med(xs):
                xs = [x for x in xs if x is not None]
                return sorted(xs)[len(xs) // 2] if xs else None

            ty_w, tlc_w = med(ty_walls), med(tlc_walls)
            ty_r, tlc_r = med(ty_rsses), med(tlc_rsses)
            rec["ty_distinct"] = ty_counts.get("distinct")
            rec["tlc_distinct"] = tlc_counts.get("distinct")
            rec["ty_generated"] = ty_counts.get("generated")
            rec["tlc_generated"] = tlc_counts.get("generated")
            rec["ty_wall"] = f"{ty_w:.3f}" if ty_w else ""
            rec["tlc_wall"] = f"{tlc_w:.3f}" if tlc_w else ""
            rec["ty_rss_kb"] = ty_r or ""
            rec["tlc_rss_kb"] = tlc_r or ""
            parity = (rec["ty_distinct"] is not None
                      and rec["ty_distinct"] == rec["tlc_distinct"]
                      and rec["ty_generated"] == rec["tlc_generated"])
            rec["parity"] = "exact" if parity else "MISMATCH"
            rec["verdict"] = "error" if (ty_counts.get("error") or tlc_counts.get("error")) else "ok"
            if ty_w and tlc_w:
                # TLC/TY speedup: >1 means TY is faster.
                rec["time_ratio"] = f"{tlc_w / ty_w:.4f}"
            if ty_r and tlc_r:
                # TY/TLC memory: <1 means TY uses less.
                rec["mem_ratio"] = f"{ty_r / tlc_r:.4f}"
            rec["both_win"] = ("yes" if parity and ty_w and tlc_w and ty_r and tlc_r
                               and tlc_w / ty_w > 1.05 and ty_r / tlc_r < 0.95 else "no")
            rec["tlc_fp_collision"] = "yes" if tlc_counts.get("fp_collision_warning") else ""
            rec["note"] = note
            writer.writerow(rec)
            fh.flush()
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
