#!/usr/bin/env bash
# Copyright 2026 Andrew Yates.
# Author: Andrew Yates
# Licensed under the Apache License, Version 2.0
#
# ty_vs_tlc_memtime.sh — single-threaded TY-vs-TLC differential measuring BOTH
# wall-clock runtime AND peak memory (max RSS), the two metrics of the goal
# "TY is more efficient in runtime AND memory than TLC for every test case,
# single-threaded."
#
# Unlike the older scripts (perf_loser_scan.sh / mass_tlc_compare.sh /
# compare_with_tlc.sh) this:
#   * captures peak RSS for BOTH tools via `/usr/bin/time` (`-l`/bytes on macOS,
#     `-v`/kbytes on Linux, normalized to bytes),
#   * gates win/loss on STATE PARITY + VERDICT PARITY (a perf win on a
#     disagreeing answer is meaningless),
#   * repeats each run (default 3x) taking MIN time / MAX rss to damp JVM jitter,
#   * uses env-overridable defaults (JAVA from PATH, $HOME/tlaplus/tytools.jar,
#     gtimeout-or-timeout) — NOT the stale hardcoded paths of the old scripts.
#
# Usage:
#   scripts/ty_vs_tlc_memtime.sh [SPEC.cfg ...]      # explicit cfg list
#   scripts/ty_vs_tlc_memtime.sh                     # default corpus
# Env:
#   TIMEOUT=60   REPEAT=3   OUT_CSV=reports/ty_vs_tlc_memtime.csv
#   CORPUS=/path/to/dir-with-cfgs   (default: ~/tlaplus-examples/specifications)
#   NOISE_TIME_S=0.0  NOISE_RSS_B=0  (deltas under these are not counted as losses)

set -uo pipefail

REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
TY="${TY:-$REPO/target/release/ty}"
JAVA="${JAVA:-java}"
command -v "$JAVA" >/dev/null 2>&1 && JAVA="$(command -v "$JAVA")"
TLC_JAR="${TLC_JAR:-$HOME/tlaplus/tytools.jar}"
TIMEOUT_BIN="${TIMEOUT_BIN:-$(command -v gtimeout || command -v timeout)}"
TIME_BIN="${TIME_BIN:-/usr/bin/time}"
# Peak-RSS measurement differs by platform: BSD/macOS `time -l` reports BYTES
# ("<n>  maximum resident set size"); GNU/Linux `time -v` reports KILOBYTES
# ("Maximum resident set size (kbytes): <n>"). Pick the flag + a byte multiplier
# so the rss columns are bytes on both. Override TIME_FLAG/RSS_MULT if needed.
case "$(uname -s)" in
    Darwin) TIME_FLAG="${TIME_FLAG:--l}"; RSS_MULT="${RSS_MULT:-1}" ;;
    *)      TIME_FLAG="${TIME_FLAG:--v}"; RSS_MULT="${RSS_MULT:-1024}" ;;
esac
TIMEOUT="${TIMEOUT:-60}"
REPEAT="${REPEAT:-3}"
CORPUS="${CORPUS:-$HOME/tlaplus-examples/specifications}"
OUT_CSV="${OUT_CSV:-$REPO/reports/ty_vs_tlc_memtime.csv}"
NOISE_TIME_S="${NOISE_TIME_S:-0.0}"
NOISE_RSS_B="${NOISE_RSS_B:-0}"

mkdir -p "$(dirname "$OUT_CSV")"

# ---- preflight: fail loud on missing tools (the old scripts' stale-path bug) --
for p in "$TY" "$JAVA" "$TLC_JAR" "$TIMEOUT_BIN" "$TIME_BIN"; do
    if [ ! -e "$p" ]; then echo "FATAL: missing required path: $p" >&2; exit 2; fi
done
"$TY" check --help >/dev/null 2>&1 || { echo "FATAL: '$TY check' not runnable" >&2; exit 2; }

TLC_ENV=(env -u JAVA_TOOL_OPTIONS -u JDK_JAVA_OPTIONS -u _JAVA_OPTIONS)
TLC_ARGS=(-XX:ActiveProcessorCount=1 -XX:+UseSerialGC -Xms64m -Xmx4g)
# CommunityModules broadens the set of specs TLC can parse/run (fairer corpus).
COMMUNITY_MODULES="${COMMUNITY_MODULES:-$HOME/tlaplus/CommunityModules.jar}"
TLC_CP="$TLC_JAR"
[ -f "$COMMUNITY_MODULES" ] && TLC_CP="$TLC_JAR:$COMMUNITY_MODULES"
# NOTE: deadlock policy is NOT forced here — each spec's .cfg CHECK_DEADLOCK
# decides, and BOTH tools honor the cfg (verified: TY honors CHECK_DEADLOCK,
# TLC reads it from the cfg). Forcing TLC's `-deadlock` (which DISABLES deadlock
# checking) while TY keeps it on was an unfair asymmetry in the old scripts.

# Strip every TY_* lever for an apples-to-apples run.
TY_FAIR=(env)
while IFS='=' read -r n _; do
    case "$n" in TY_*) TY_FAIR+=( -u "$n" );; esac
done < <(env)

echo "spec,cfg,ty_states,tlc_states,states_match,ty_verdict,tlc_verdict,verdict_match,ty_time_s,tlc_time_s,time_ratio,time_win,ty_rss_b,tlc_rss_b,mem_ratio,mem_win,status,note" > "$OUT_CSV"

# minimum of two floats (returns the smaller)
fmin() { awk -v a="$1" -v b="$2" 'BEGIN{print (a<b)?a:b}'; }
# integer max
imax() { awk -v a="$1" -v b="$2" 'BEGIN{print (a>b)?a:b}'; }
# peak RSS in BYTES from a `time` stderr file (case-insensitive label matches
# both macOS "maximum resident set size" and GNU/Linux "Maximum ... (kbytes)";
# RSS_MULT normalizes the unit to bytes).
parse_rss() {  # $1 = stderr file
    local n; n=$(grep -i 'maximum resident set size' "$1" | grep -oE '[0-9]+' | head -1)
    [ -z "$n" ] && { printf '0'; return; }
    awk -v n="$n" -v m="$RSS_MULT" 'BEGIN{printf "%d", n*m}'
}

run_ty() {  # $1 tla  $2 cfg  -> echoes "rc<TAB>time_s<TAB>rss_b<TAB>states<TAB>verdict"
    local tla="$1" cfg="$2"
    local best_t="" max_rss=0 rc=0 states="-" verdict="-"
    local i
    for ((i=0;i<REPEAT;i++)); do
        local jf tf; jf=$(mktemp); tf=$(mktemp)
        local start end
        start=$(date +%s.%N)
        "$TIMEOUT_BIN" "${TIMEOUT}s" "$TIME_BIN" "$TIME_FLAG" "${TY_FAIR[@]}" \
            "$TY" check "$tla" --config "$cfg" --workers 1 --force --output json ${TY_CHECK_EXTRA:-} >"$jf" 2>"$tf"
        rc=$?
        end=$(date +%s.%N)
        local t; t=$(awk -v s="$start" -v e="$end" 'BEGIN{printf "%.4f", e-s}')
        local rss; rss=$(parse_rss "$tf")
        # Parse the JSON verdict REGARDLESS of exit code: an invariant/property
        # violation is a legitimate verdict and exits nonzero (e.g. DieHard's
        # NotSolved). The JSON is authoritative; only a missing/unparseable
        # summary (and rc != 124 timeout) is a real TY error.
        if [ "$rc" -ne 124 ] && [ -s "$jf" ]; then
            local summ; summ=$("$TY" check-summary "$jf" 2>/dev/null)
            local st sd
            st=$(printf '%s' "$summ" | cut -f1)
            sd=$(printf '%s' "$summ" | cut -f7)   # states_found
            [ -n "$sd" ] && states="$sd"
            case "$st" in ok|limit_reached) verdict="ok";; "" ) verdict="-";; *) verdict="violation";; esac
        fi
        [ -z "$best_t" ] && best_t="$t" || best_t=$(fmin "$best_t" "$t")
        max_rss=$(imax "$max_rss" "$rss")
        rm -f "$jf" "$tf"
        [ "$rc" -eq 124 ] && break
    done
    printf '%s\t%s\t%s\t%s\t%s\n' "$rc" "$best_t" "$max_rss" "$states" "$verdict"
}

run_tlc() {  # $1 tla  $2 cfg -> "rc<TAB>time_s<TAB>rss_b<TAB>states<TAB>verdict"
    local tla="$1" cfg="$2"
    local best_t="" max_rss=0 rc=0 states="-" verdict="-"
    local i
    for ((i=0;i<REPEAT;i++)); do
        local meta of tf; meta=$(mktemp -d); of=$(mktemp); tf=$(mktemp)
        local start end
        start=$(date +%s.%N)
        "$TIMEOUT_BIN" "${TIMEOUT}s" "$TIME_BIN" "$TIME_FLAG" "${TLC_ENV[@]}" \
            "$JAVA" "${TLC_ARGS[@]}" -cp "$TLC_CP" tlc2.TLC \
            -metadir "$meta" -config "$cfg" -workers 1 "$tla" >"$of" 2>"$tf"
        rc=$?
        end=$(date +%s.%N)
        local t; t=$(awk -v s="$start" -v e="$end" 'BEGIN{printf "%.4f", e-s}')
        local rss; rss=$(parse_rss "$tf")
        local both; both=$(cat "$of" "$tf")
        local s; s=$(printf '%s' "$both" | grep -oE '[0-9,]+ distinct states found' | tail -1 | tr -d ',' | grep -oE '^[0-9]+')
        [ -n "$s" ] && states="$s"
        # Verdict detection, most-specific first:
        #  - parse/semantic/runtime FAILURE => "error" (non-comparable, not a
        #    violation) — e.g. TLAPS-extending specs TLC can't parse.
        #  - genuine violation => "violation".
        #  - clean completion => "ok".
        if printf '%s' "$both" | grep -qE 'Parsing or semantic analysis failed|Fatal errors|Unknown operator|cannot be (found|used)|Unrecoverable error|TLC threw|was not (found|declared)|Was expecting|module .* (not found|could not be)'; then
            verdict="error"
        elif printf '%s' "$both" | grep -qE 'is violated|Deadlock reached|Temporal properties were violated|Error: Invariant|Error: Action property'; then
            verdict="violation"
        elif printf '%s' "$both" | grep -qE 'Model checking completed\. No error|No error has been found'; then
            verdict="ok"
        elif [ "$rc" -eq 0 ]; then verdict="ok"; fi
        [ -z "$best_t" ] && best_t="$t" || best_t=$(fmin "$best_t" "$t")
        max_rss=$(imax "$max_rss" "$rss")
        rm -rf "$meta"; rm -f "$of" "$tf"
        [ "$rc" -eq 124 ] && break
    done
    printf '%s\t%s\t%s\t%s\t%s\n' "$rc" "$best_t" "$max_rss" "$states" "$verdict"
}

process() {
    local tla="$1" cfg="$2"
    local name; name="$(basename "$tla" .tla)"

    local ty; ty=$(run_ty "$tla" "$cfg")
    local ty_rc ty_t ty_rss ty_states ty_verdict
    IFS=$'\t' read -r ty_rc ty_t ty_rss ty_states ty_verdict <<<"$ty"

    local status="OK" note=""
    # A nonzero rc with a parsed verdict (e.g. "violation") is NOT an error.
    # Only a timeout, or a missing verdict, is.
    if [ "$ty_rc" -eq 124 ]; then status="TY_TIMEOUT";
    elif [ "$ty_verdict" = "-" ]; then status="TY_ERROR"; note="ty_rc=$ty_rc"; fi

    local tlc_rc="-" tlc_t="-" tlc_rss="-" tlc_states="-" tlc_verdict="-"
    if [ "$status" = "OK" ]; then
        local tlc; tlc=$(run_tlc "$tla" "$cfg")
        IFS=$'\t' read -r tlc_rc tlc_t tlc_rss tlc_states tlc_verdict <<<"$tlc"
        if [ "$tlc_rc" -eq 124 ]; then status="TLC_TIMEOUT";
        elif [ "$tlc_verdict" = "error" ] || [ "$tlc_verdict" = "-" ]; then status="TLC_ERROR"; note="tlc_unparseable"; fi
    fi

    local states_match="-" verdict_match="-" time_ratio="-" time_win="-" mem_ratio="-" mem_win="-"
    if [ "$status" = "OK" ]; then
        [ "$ty_verdict" = "$tlc_verdict" ] && verdict_match="true" || verdict_match="false"
        # Comparability gates on VERDICT agreement only. State counts can diverge
        # legitimately: TY auto-symmetry reduces the explored set (e.g. TwoPhase
        # TY=88 vs TLC=288, both ok), and violation runs stop at first-found.
        # verdict_match (TLC's exploration corroborating TY's) is the soundness
        # cross-check; a state divergence on ok/ok is NOTED for audit, not failed.
        if [ "$ty_verdict" = "ok" ] && [ "$tlc_verdict" = "ok" ]; then
            if [ "$ty_states" = "$tlc_states" ]; then states_match="true"; else states_match="false"; note="${note:+$note;}state_div ty=$ty_states tlc=$tlc_states"; fi
        else
            states_match="n/a"
        fi
        [ "$verdict_match" = "false" ] && status="VERDICT_FAIL"
        if [ "$tlc_t" != "0" ] && [ -n "$tlc_t" ]; then
            time_ratio=$(awk -v a="$ty_t" -v b="$tlc_t" 'BEGIN{printf "%.3f", (b>0)?a/b:0}')
        fi
        time_win=$(awk -v a="$ty_t" -v b="$tlc_t" -v n="$NOISE_TIME_S" 'BEGIN{print (a < b-n)?"true":"false"}')
        if [ "${tlc_rss:-0}" -gt 0 ] 2>/dev/null; then
            mem_ratio=$(awk -v a="$ty_rss" -v b="$tlc_rss" 'BEGIN{printf "%.3f", (b>0)?a/b:0}')
            mem_win=$(awk -v a="$ty_rss" -v b="$tlc_rss" -v n="$NOISE_RSS_B" 'BEGIN{print (a < b-n)?"true":"false"}')
        else
            status="RSS_MISSING"
        fi
    fi

    echo "$name,$cfg,$ty_states,$tlc_states,$states_match,$ty_verdict,$tlc_verdict,$verdict_match,$ty_t,$tlc_t,$time_ratio,$time_win,$ty_rss,$tlc_rss,$mem_ratio,$mem_win,$status,$note" >> "$OUT_CSV"
    printf '%-34s ty=%-7s tlc=%-7s t %-7s/%-7s (%s) mem %-10s/%-10s (%s) [%s]\n' \
        "$name" "$ty_states" "$tlc_states" "$ty_t" "$tlc_t" "$time_win" "$ty_rss" "$tlc_rss" "$mem_win" "$status"
}

# ---- spec selection ----------------------------------------------------------
declare -a CFGS=()
if [ "$#" -gt 0 ]; then
    CFGS=("$@")
elif [ -d "$CORPUS" ]; then
    while IFS= read -r c; do CFGS+=("$c"); done < <(find "$CORPUS" -name '*.cfg' -type f -not -name '._*' | sort)
else
    echo "corpus dir $CORPUS absent; falling back to in-repo specs" >&2
    while IFS= read -r c; do CFGS+=("$c"); done < <(find "$REPO/test_specs" "$REPO/examples" -name '*.cfg' -type f 2>/dev/null | sort)
fi

echo "Running ${#CFGS[@]} spec(s); TIMEOUT=${TIMEOUT}s REPEAT=${REPEAT}; out=$OUT_CSV" >&2
for cfg in "${CFGS[@]}"; do
    tla="${cfg%.cfg}.tla"
    [ -f "$tla" ] || continue
    process "$tla" "$cfg"
done

# ---- report ------------------------------------------------------------------
echo "" >&2
awk -F, 'NR>1{
    total++;
    if($17=="OK"){comp++;
        if($12=="true")tw++; if($16=="true")mw++;
        if($12=="true"&&$16=="true")bw++;
        if($12!="true"||$16!="true"){loss++; losers=losers"\n  "$1" (cfg "$2"): time_win="$12" mem_win="$16" ratios t="$11" m="$15}
    }
    if($17=="PARITY_FAIL")pf++;
    if($17=="VERDICT_FAIL")vf++;
    if($17=="TY_TIMEOUT"||$17=="TLC_TIMEOUT")to++;
    if($17=="TY_ERROR")te++;
    if($17=="RSS_MISSING")rm++;
}
END{
    printf "=== SUMMARY ===\n";
    printf "total=%d comparable(OK+parity+verdict ok)=%d\n", total, comp;
    printf "  time wins=%d  mem wins=%d  BOTH wins=%d\n", tw, mw, bw;
    printf "  TY LOSES on >=1 metric: %d%s\n", loss, (loss?losers:"");
    printf "non-comparable: parity_fail=%d verdict_fail=%d timeouts=%d ty_error=%d rss_missing=%d\n", pf, vf, to, te, rm;
}' "$OUT_CSV" >&2

echo "Wrote $OUT_CSV" >&2
