#!/usr/bin/env bash
# Copyright 2026 Andrew Yates.
# Author: Andrew Yates
# Licensed under the Apache License, Version 2.0
#
# span_sync_check.sh — design artifact A6 of
# docs/design/trust-verification-atoms-2026-06-17.md.
#
# The self-liveness obligation (crates/tla-check/src/selfliveness/mod.rs) labels
# each engine-selection control point with a HAND-MAINTAINED `file:line:symbol`
# span. Those labels are NOT mechanically bound to the source (true binding is
# roadmap step 4, MIR extraction). This script is the mechanical guard against
# DRIFT: for every load-bearing span it asserts the cited source LINE still
# contains the expected source token. If a line drifts, this fails LOUD with the
# file:line, forcing a re-sync of selfliveness/mod.rs rather than silently
# trusting a stale label (and silently lying in any counterexample lasso).
#
# The `symbol` component of a ProgressSpan is a descriptive label, not an
# identifier on the line, so the table below maps each span to the REAL source
# substring that must appear (symbol-anchored check, stronger than line-only).
#
# Exit 0 = all spans resolve; nonzero = drift detected. Wire into CI.

set -uo pipefail

REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
MC="$REPO/crates/tla-check/src/check/model_checker"

# Each row: "<relfile>:<line>:<expected-substring>"
# Rows are the physical (line-bearing) spans referenced by selfliveness/mod.rs
# and its doc comments. Logical spans (run_bfs_loop:drain_interp etc.) have no
# single line and are checked by function-existence instead (see below).
SPANS=(
  "trust_cg_dispatch.rs:271:TRUST_CG_LAZY_COMPILE_WORK_THRESHOLD_DEFAULT"
  "run_helpers.rs:6765:trust_cg_lazy_compile_gate_fires"
  "run_helpers.rs:6762:trust_cg_lazy_compile_work_threshold"
  "run_helpers.rs:6738:maybe_trigger_trust_cg_lazy_compile"
  "run_bfs_notrace.rs:861:trust_cg_should_hot_swap_to_compiled_bfs"
  "bfs/transport_seq.rs:236:maybe_trigger_trust_cg_lazy_compile"
  "run_helpers.rs:7145:compiled_bfs_step"
  "run_bfs_notrace.rs:781:should_use_compiled_bfs"
)

# Logical control points: assert the function still exists somewhere in MC.
FUNCS=(
  "fn run_bfs_loop"
  "fn run_compiled_bfs_loop"
)

fail=0

for row in "${SPANS[@]}"; do
  f="${row%%:*}"; rest="${row#*:}"; ln="${rest%%:*}"; sub="${rest#*:}"
  path="$MC/$f"
  if [ ! -f "$path" ]; then
    echo "DRIFT: file missing: $path" >&2; fail=1; continue
  fi
  line="$(sed -n "${ln}p" "$path")"
  if printf '%s' "$line" | grep -qF -- "$sub"; then
    printf 'OK    %-26s:%-5s contains %s\n' "$f" "$ln" "$sub"
  else
    printf 'DRIFT %-26s:%-5s MISSING %q\n      got: %s\n' "$f" "$ln" "$sub" "$(printf '%s' "$line" | sed 's/^[[:space:]]*//')" >&2
    fail=1
  fi
done

for fn in "${FUNCS[@]}"; do
  if grep -rqF -- "$fn" "$MC"; then
    printf 'OK    function present: %s\n' "$fn"
  else
    printf 'DRIFT function missing in %s: %s\n' "$MC" "$fn" >&2
    fail=1
  fi
done

if [ "$fail" -ne 0 ]; then
  echo "" >&2
  echo "span-sync FAILED: a self-liveness span drifted from its source line." >&2
  echo "Re-sync crates/tla-check/src/selfliveness/mod.rs (and this table) to the" >&2
  echo "current line numbers/tokens, then re-run." >&2
  exit 1
fi
echo ""
echo "span-sync OK: all self-liveness spans resolve to live source."
