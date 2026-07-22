#!/usr/bin/env bash
# Copyright 2026 Andrew Yates.
# Author: Andrew Yates
# Licensed under the Apache License, Version 2.0
#
# ty_vs_apalache_memtime.sh — single-threaded TY-vs-Apalache differential measuring
# BOTH wall-clock runtime AND peak memory (max RSS), the companion to
# ty_vs_tlc_memtime.sh. Apalache is the SYMBOLIC (SMT/BMC) TLA+ checker, so this
# closes the "what about the other symbolic tool?" critique: TY should win against
# both TLC (explicit-state) and Apalache (symbolic).
#
# HONEST CAVEATS (Apalache differs from TLC in ways the gate must respect):
#   * Apalache is BOUNDED (--length=LEN): an Apalache "no error" proves no
#     violation only WITHIN LEN steps, NOT exhaustively. So an Apalache-ok only
#     CORROBORATES a TY-ok up to the bound; TY may find a deeper violation that
#     Apalache@LEN misses (recorded, never a TY failure). LEN is in every row.
#   * Apalache is SYMBOLIC and does not enumerate distinct states, so there is NO
#     state-count parity — the gate is VERDICT parity only.
#   * Apalache needs operator NAMES via --init/--next/--inv; it does NOT honor a
#     TLC `SPECIFICATION Spec` directive. Specs with no INIT/NEXT in the .cfg are
#     recorded APALACHE_SKIP (set INIT_OP/NEXT_OP to opt in).
#   * Apalache requires `\* @type:` variable annotations most corpus specs lack;
#     such specs report a type error -> verdict="error" (non-comparable), exactly
#     like the TLC harness treats unparseable specs.
#
# Usage:
#   scripts/ty_vs_apalache_memtime.sh [SPEC.cfg ...]
#   scripts/ty_vs_apalache_memtime.sh
# Env:
#   TIMEOUT=120  REPEAT=3  LEN=10  OUT_CSV=reports/ty_vs_apalache_memtime.csv
#   CORPUS=/path/to/dir-with-cfgs  (default ~/tlaplus-examples/specifications)
#   APALACHE=~/apalache/bin/apalache-mc   APALACHE_JAVA_HOME=<jdk> (default: derived from PATH java)
#   APALACHE_EXTRA="--features=no-rows"   (extra flags for every apalache check;
#       needed for corpus specs whose @type record annotations predate TS 1.2)
#   INIT_OP / NEXT_OP / INV   (override the operator names parsed from the .cfg)

set -uo pipefail

REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
TY="${TY:-$REPO/target/release/ty}"
APALACHE="${APALACHE:-$HOME/apalache/bin/apalache-mc}"
# Apalache needs a JDK via JAVA_HOME; derive from PATH `java` (or $JAVA_HOME) if unset.
if [ -z "${APALACHE_JAVA_HOME:-}" ]; then
    if [ -n "${JAVA_HOME:-}" ]; then APALACHE_JAVA_HOME="$JAVA_HOME"
    else
        _j="$(command -v java 2>/dev/null || true)"
        [ -n "$_j" ] && APALACHE_JAVA_HOME="$(cd "$(dirname "$(readlink -f "$_j" 2>/dev/null || echo "$_j")")/.." && pwd)"
    fi
fi
APALACHE_JAVA_HOME="${APALACHE_JAVA_HOME:-}"
TIMEOUT_BIN="${TIMEOUT_BIN:-$(command -v gtimeout || command -v timeout)}"
TIME_BIN="${TIME_BIN:-/usr/bin/time}"
# Peak-RSS unit differs by platform (macOS `time -l` bytes / Linux `time -v`
# kbytes); pick the flag + byte multiplier so rss columns are bytes on both.
case "$(uname -s)" in
    Darwin) TIME_FLAG="${TIME_FLAG:--l}"; RSS_MULT="${RSS_MULT:-1}" ;;
    *)      TIME_FLAG="${TIME_FLAG:--v}"; RSS_MULT="${RSS_MULT:-1024}" ;;
esac
TIMEOUT="${TIMEOUT:-120}"
REPEAT="${REPEAT:-3}"
LEN="${LEN:-10}"
# Extra flags appended to every `apalache-mc check` (word-split). Use this to
# pass legitimate Apalache options the corpus needs, e.g. APALACHE_EXTRA="--features=no-rows"
# for specs whose `\* @type:` record annotations predate Type System 1.2.
APALACHE_EXTRA="${APALACHE_EXTRA:-}"
read -r -a APALACHE_EXTRA_ARGS <<<"$APALACHE_EXTRA"
CORPUS="${CORPUS:-$HOME/tlaplus-examples/specifications}"
OUT_CSV="${OUT_CSV:-$REPO/reports/ty_vs_apalache_memtime.csv}"
NOISE_TIME_S="${NOISE_TIME_S:-0.0}"
NOISE_RSS_B="${NOISE_RSS_B:-0}"

mkdir -p "$(dirname "$OUT_CSV")"

# ---- preflight ---------------------------------------------------------------
for p in "$TY" "$TIMEOUT_BIN" "$TIME_BIN"; do
    if [ ! -e "$p" ]; then echo "FATAL: missing required path: $p" >&2; exit 2; fi
done
if [ ! -x "$APALACHE" ] && ! command -v "$APALACHE" >/dev/null 2>&1; then
    echo "FATAL: Apalache not found at '$APALACHE'." >&2
    echo "  install: VER=0.58.0; cd ~ && curl -fLO https://github.com/apalache-mc/apalache/releases/download/v\$VER/apalache-\$VER.tgz && tar xzf apalache-\$VER.tgz && mv apalache-\$VER apalache" >&2
    echo "  (needs a JDK; set APALACHE_JAVA_HOME, default $APALACHE_JAVA_HOME)" >&2
    exit 2
fi
"$TY" check --help >/dev/null 2>&1 || { echo "FATAL: '$TY check' not runnable" >&2; exit 2; }

# Apalache wants java on PATH / JAVA_HOME.
APALACHE_ENV=(env -u JAVA_TOOL_OPTIONS -u JDK_JAVA_OPTIONS -u _JAVA_OPTIONS
              "JAVA_HOME=$APALACHE_JAVA_HOME" "PATH=$APALACHE_JAVA_HOME/bin:$PATH")

# Strip every TY_* lever for an apples-to-apples run.
TY_FAIR=(env)
while IFS='=' read -r n _; do case "$n" in TY_*) TY_FAIR+=( -u "$n" );; esac; done < <(env)

echo "spec,cfg,len,ty_verdict,apalache_verdict,verdict_match,ty_time_s,apa_time_s,time_ratio,time_win,ty_rss_b,apa_rss_b,mem_ratio,mem_win,status,note" > "$OUT_CSV"

fmin() { awk -v a="$1" -v b="$2" 'BEGIN{print (a<b)?a:b}'; }
imax() { awk -v a="$1" -v b="$2" 'BEGIN{print (a>b)?a:b}'; }
# peak RSS in BYTES from a `time` stderr file (case-insensitive label matches
# both macOS and GNU/Linux; RSS_MULT normalizes the unit to bytes).
parse_rss() {  # $1 = stderr file
    local n; n=$(grep -i 'maximum resident set size' "$1" | grep -oE '[0-9]+' | head -1)
    [ -z "$n" ] && { printf '0'; return; }
    awk -v n="$n" -v m="$RSS_MULT" 'BEGIN{printf "%d", n*m}'
}

# parse an operator name from a .cfg directive (INIT / NEXT); first match.
cfg_op() { grep -iE "^[[:space:]]*$1[[:space:]]+" "$2" | head -1 | awk '{print $2}'; }
# first INVARIANT operator from a .cfg.
cfg_inv() { grep -iE "^[[:space:]]*INVARIANTS?[[:space:]]+" "$1" | head -1 | awk '{print $2}'; }

run_ty() {  # $1 tla  $2 cfg -> "rc<TAB>time<TAB>rss<TAB>verdict"
    local tla="$1" cfg="$2"
    local best_t="" max_rss=0 rc=0 verdict="-" i
    for ((i=0;i<REPEAT;i++)); do
        local jf tf start end; jf=$(mktemp); tf=$(mktemp); start=$(date +%s.%N)
        "$TIMEOUT_BIN" "${TIMEOUT}s" "$TIME_BIN" "$TIME_FLAG" "${TY_FAIR[@]}" \
            "$TY" check "$tla" --config "$cfg" --workers 1 --force --output json >"$jf" 2>"$tf"
        rc=$?; end=$(date +%s.%N)
        local t; t=$(awk -v s="$start" -v e="$end" 'BEGIN{printf "%.4f", e-s}')
        local rss; rss=$(parse_rss "$tf")
        if [ "$rc" -ne 124 ] && [ -s "$jf" ]; then
            local st; st=$("$TY" check-summary "$jf" 2>/dev/null | cut -f1)
            case "$st" in ok|limit_reached) verdict="ok";; "") verdict="-";; *) verdict="violation";; esac
        fi
        [ -z "$best_t" ] && best_t="$t" || best_t=$(fmin "$best_t" "$t")
        max_rss=$(imax "$max_rss" "$rss"); rm -f "$jf" "$tf"; [ "$rc" -eq 124 ] && break
    done
    printf '%s\t%s\t%s\t%s\n' "$rc" "$best_t" "$max_rss" "$verdict"
}

run_apalache() {  # $1 tla  $2 cfg -> "rc<TAB>time<TAB>rss<TAB>verdict"
    # Use --config to let Apalache read the TLC config DIRECTLY (INIT/NEXT/
    # SPECIFICATION/CONSTANTS/INVARIANTS) — the FAIR invocation. Apalache's real
    # barrier is its TYPE-ANNOTATION requirement (a type error => verdict=error,
    # non-comparable), NOT the config format.
    local tla="$1" cfg="$2"
    local best_t="" max_rss=0 rc=0 verdict="-" i
    for ((i=0;i<REPEAT;i++)); do
        local od of tf start end; od=$(mktemp -d); of=$(mktemp); tf=$(mktemp); start=$(date +%s.%N)
        "$TIMEOUT_BIN" "${TIMEOUT}s" "$TIME_BIN" "$TIME_FLAG" "${APALACHE_ENV[@]}" \
            "$APALACHE" check "${APALACHE_EXTRA_ARGS[@]}" --config="$cfg" --length="$LEN" --out-dir="$od" "$tla" >"$of" 2>"$tf"
        rc=$?; end=$(date +%s.%N)
        local t; t=$(awk -v s="$start" -v e="$end" 'BEGIN{printf "%.4f", e-s}')
        local rss; rss=$(parse_rss "$tf")
        local both; both=$(cat "$of" "$tf")
        # most-specific first: parse/type/config FAILURE => error (non-comparable);
        # violation/deadlock => violation; no-error => ok.
        if printf '%s' "$both" | grep -qiE 'type input error|Type checker error|Configuration error|parser error|syntax error|Unexpected|EXITCODE: ERROR \(255\)|Mismatch|cannot (find|be)'; then
            verdict="error"
        elif printf '%s' "$both" | grep -qiE 'The outcome is: (Error|Deadlock)|Checker has found|Counterexample|EXITCODE: ERROR \(12\)'; then
            verdict="violation"
        elif printf '%s' "$both" | grep -qiE 'The outcome is: NoError|Checker reports no error|EXITCODE: OK'; then
            verdict="ok"
        elif printf '%s' "$both" | grep -qiE 'EXITCODE: ERROR'; then
            verdict="error"
        fi
        [ -z "$best_t" ] && best_t="$t" || best_t=$(fmin "$best_t" "$t")
        max_rss=$(imax "$max_rss" "$rss"); rm -rf "$od"; rm -f "$of" "$tf"; [ "$rc" -eq 124 ] && break
    done
    printf '%s\t%s\t%s\t%s\n' "$rc" "$best_t" "$max_rss" "$verdict"
}

process() {
    local tla="$1" cfg="$2"
    local name; name="$(basename "$tla" .tla)"

    local ty; ty=$(run_ty "$tla" "$cfg")
    local ty_rc ty_t ty_rss ty_verdict
    IFS=$'\t' read -r ty_rc ty_t ty_rss ty_verdict <<<"$ty"

    local status="OK" note=""
    if [ "$ty_rc" -eq 124 ]; then status="TY_TIMEOUT";
    elif [ "$ty_verdict" = "-" ]; then status="TY_ERROR"; note="ty_rc=$ty_rc"; fi

    # Apalache reads the TLC .cfg via --config (INIT/NEXT/SPECIFICATION/CONSTANTS/
    # INVARIANTS). The non-comparable case is a TYPE error (Apalache needs @type
    # annotations the TLC corpus lacks) -> APALACHE_ERROR (untyped), like the TLC
    # harness's unparseable branch.
    local apa_rc="-" apa_t="-" apa_rss="-" apa_verdict="-"
    if [ "$status" = "OK" ]; then
        local apa; apa=$(run_apalache "$tla" "$cfg")
        IFS=$'\t' read -r apa_rc apa_t apa_rss apa_verdict <<<"$apa"
        if [ "$apa_rc" -eq 124 ]; then status="APALACHE_TIMEOUT";
        elif [ "$apa_verdict" = "error" ] || [ "$apa_verdict" = "-" ]; then status="APALACHE_ERROR"; note="${note:+$note;}apalache_untyped_or_unsupported"; fi
    fi

    local verdict_match="-" time_ratio="-" time_win="-" mem_ratio="-" mem_win="-"
    if [ "$status" = "OK" ]; then
        # VERDICT parity only (Apalache is symbolic; no state count). Bounded:
        # an apalache "ok" corroborates TY "ok" only up to LEN. A TY violation
        # MUST match an apalache violation (within LEN); a TY-ok with apalache-ok
        # is corroborated up to LEN.
        [ "$ty_verdict" = "$apa_verdict" ] && verdict_match="true" || verdict_match="false"
        [ "$verdict_match" = "false" ] && { status="VERDICT_FAIL"; note="${note:+$note;}within_bound_len=$LEN"; }
        if [ "$apa_t" != "0" ] && [ -n "$apa_t" ]; then
            time_ratio=$(awk -v a="$ty_t" -v b="$apa_t" 'BEGIN{printf "%.3f",(b>0)?a/b:0}')
        fi
        time_win=$(awk -v a="$ty_t" -v b="$apa_t" -v n="$NOISE_TIME_S" 'BEGIN{print (a<b-n)?"true":"false"}')
        if [ "${apa_rss:-0}" -gt 0 ] 2>/dev/null; then
            mem_ratio=$(awk -v a="$ty_rss" -v b="$apa_rss" 'BEGIN{printf "%.3f",(b>0)?a/b:0}')
            mem_win=$(awk -v a="$ty_rss" -v b="$apa_rss" -v n="$NOISE_RSS_B" 'BEGIN{print (a<b-n)?"true":"false"}')
        else status="RSS_MISSING"; fi
    fi

    echo "$name,$cfg,$LEN,$ty_verdict,$apa_verdict,$verdict_match,$ty_t,$apa_t,$time_ratio,$time_win,$ty_rss,$apa_rss,$mem_ratio,$mem_win,$status,$note" >> "$OUT_CSV"
    printf '%-34s ty=%-9s apa=%-9s t %-7s/%-7s (%s) mem %-10s/%-10s (%s) [%s]\n' \
        "$name" "$ty_verdict" "$apa_verdict" "$ty_t" "$apa_t" "$time_win" "$ty_rss" "$apa_rss" "$mem_win" "$status"
}

# ---- spec selection ----------------------------------------------------------
declare -a CFGS=()
if [ "$#" -gt 0 ]; then CFGS=("$@")
elif [ -d "$CORPUS" ]; then while IFS= read -r c; do CFGS+=("$c"); done < <(find "$CORPUS" -name '*.cfg' -type f | sort)
else
    echo "corpus dir $CORPUS absent; falling back to in-repo specs" >&2
    while IFS= read -r c; do CFGS+=("$c"); done < <(find "$REPO/test_specs" "$REPO/examples" -name '*.cfg' -type f 2>/dev/null | sort)
fi

echo "Running ${#CFGS[@]} spec(s) vs Apalache@len=$LEN; TIMEOUT=${TIMEOUT}s REPEAT=$REPEAT; out=$OUT_CSV" >&2
for cfg in "${CFGS[@]}"; do tla="${cfg%.cfg}.tla"; [ -f "$tla" ] || continue; process "$tla" "$cfg"; done

# ---- report ------------------------------------------------------------------
echo "" >&2
awk -F, 'NR>1{
    total++;
    if($15=="OK"){comp++;
        if($10=="true")tw++; if($14=="true")mw++;
        if($10=="true"&&$14=="true")bw++;
        if($10!="true"||$14!="true"){loss++; losers=losers"\n  "$1" (cfg "$2"): time_win="$10" mem_win="$14" ratios t="$9" m="$13}
    }
    if($15=="VERDICT_FAIL")vf++;
    if($15=="APALACHE_SKIP")sk++;
    if($15=="APALACHE_ERROR")ae++;
    if($15=="TY_TIMEOUT"||$15=="APALACHE_TIMEOUT")to++;
    if($15=="TY_ERROR")te++;
    if($15=="RSS_MISSING")rm++;
}
END{
    printf "=== SUMMARY (TY vs Apalache, symbolic, bounded) ===\n";
    printf "total=%d comparable(verdict ok, within bound)=%d\n", total, comp;
    printf "  time wins=%d  mem wins=%d  BOTH wins=%d\n", tw, mw, bw;
    printf "  TY LOSES on >=1 metric: %d%s\n", loss, (loss?losers:"");
    printf "non-comparable: verdict_fail=%d apalache_skip(no INIT/NEXT)=%d apalache_error(untyped/unparseable)=%d timeouts=%d ty_error=%d rss_missing=%d\n", vf, sk, ae, to, te, rm;
    printf "NOTE: Apalache is bounded (len) + symbolic; ok corroborates TY-ok only up to len; needs @type annotations + INIT/NEXT.\n";
}' "$OUT_CSV" >&2

echo "Wrote $OUT_CSV" >&2
