// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Hybrid per-action native dispatch — ty-side M0 (wishlist item 4).
//!
//! This module hosts the model-checker-side wiring for the hybrid flat-view
//! projection ([`crate::state::HybridFlatView`]):
//!
//! 1. **Per-action classification** ([`ModelChecker::ensure_hybrid_dispatch_ready`]):
//!    reuse ty's existing static action-footprint analysis
//!    (`extract_detected_action_dependencies`, the same reads/writes/opaque
//!    footprints POR/coverage compute) and mark each detected action
//!    *hybrid-eligible* iff it is not opaque and its entire read/write footprint
//!    lies inside the flat-admissible variable subset. This replaces the
//!    whole-state veto (`StateLayout::supports_flat_primary`) with a per-action
//!    gate.
//!
//! 2. **Hybrid successor routing** ([`ModelChecker::hybrid_route_successor`]):
//!    for a hybrid-eligible action, route the interpreter successor through the
//!    flat-view projection (project parent → native/shadow → reconstruct
//!    against the compound parent) and, when it matches the interpreter
//!    successor exactly, use the reconstructed state.
//!
//! 3. **Native flat-view dispatch** (item 4 M0, `TY_HYBRID_NATIVE=1`):
//!    [`ModelChecker::hybrid_native_candidates_for_action`] executes the
//!    per-action artifacts compiled against the HYBRID layout
//!    (`maybe_initialize_trust_cg_hybrid_action_cache`) on the projected
//!    parent buffer; `hybrid_route_successor` consumes the resulting successor
//!    views by byte-exact buffer match against each projected interpreter
//!    successor, keeping the per-successor value-equality differential fully
//!    intact. Without the native switch (or on any per-action-instance
//!    admission decline) the routing body stays the validated
//!    interpreter-through-projection shadow
//!    ([`hybrid_shadow_flat_view_dispatch`]).
//!
//! 4. **Authoritative native dispatch after burn-in** (WP-14,
//!    `TY_HYBRID_NATIVE_AUTHORITATIVE=1`, subordinate to every switch above):
//!    once an ACTION has accumulated `TY_HYBRID_BURN_IN` (default 4096)
//!    consecutive fully-clean native/interpreter differentials, its native
//!    candidate set is enqueued directly and the interpreter enumeration is
//!    SKIPPED for that action instance
//!    ([`ModelChecker::hybrid_reconstruct_all_native_candidates`] + the
//!    authoritative branch in `run_gen`). This is sound because the native
//!    path is a complete per-instance ENUMERATOR, not a filter over
//!    interpreter output: every resolved binding-specialization key executes
//!    against the projected parent buffer alone
//!    (`try_trust_cg_hybrid_action_by_keys`), key resolution fails closed if
//!    ANY instance of the action lacks a compiled key, and each key is
//!    strictly single-successor ABI. A deterministic 1-in-`TY_HYBRID_SAMPLE`
//!    (default 64) sample of post-flip instances — keyed by a fixed hash of
//!    (projected parent buffer, action index), not a mod-counter, so reruns
//!    reproduce the same sample — keeps running the full differential; ANY
//!    observed divergence once any action has flipped trips a PERMANENT
//!    whole-run fail-back to interpreter-authoritative dispatch
//!    ([`HybridAuthoritativeMachine`]). The state that mismatched keeps the
//!    interpreter successors, so the reachable set stays correct even in the
//!    run that trips.
//!
//! Everything here is inert unless the env switch `TY_HYBRID_FLAT_VIEW` is set,
//! so the default build is byte-identical to prior behavior. The routing is
//! **fail-closed and differential**: any projection failure, footprint slip, or
//! reconstructed/interpreter divergence falls back to the interpreter successor
//! and increments a counter, so the reachable-state set can never change. In
//! M0 the interpreter remains authoritative even under `TY_HYBRID_NATIVE=1`
//! (validated shadow/burn-in): a native successor is enqueued only when it is
//! byte- and value-identical to the interpreter's. Only WP-14's
//! `TY_HYBRID_NATIVE_AUTHORITATIVE=1` (default OFF) retires that per-instance
//! differential, per action, after burn-in.

use std::sync::Arc;

use rustc_hash::FxHashSet;

use crate::coverage::{detect_actions, DetectedAction};
use crate::state::{ArrayState, FlatState, HybridFlatView};
use crate::var_index::{VarIndex, VarRegistry};

use super::ModelChecker;

/// Whether the hybrid NATIVE dispatch is switched on: `TY_HYBRID_NATIVE=1` in
/// addition to the master `TY_HYBRID_FLAT_VIEW=1` switch. Without it the
/// hybrid path stays the validated interpreter-through-projection shadow.
pub(in crate::check) fn hybrid_native_enabled() -> bool {
    std::env::var_os("TY_HYBRID_FLAT_VIEW").is_some_and(|v| v == "1")
        && std::env::var_os("TY_HYBRID_NATIVE").is_some_and(|v| v == "1")
}

/// WP-14: whether burn-in-passed actions may dispatch natively WITHOUT the
/// interpreter (`TY_HYBRID_NATIVE_AUTHORITATIVE=1`, default OFF), subordinate
/// to `TY_HYBRID_FLAT_VIEW=1` + `TY_HYBRID_NATIVE=1`. Unset, the whole M0/M1
/// validated-shadow behavior is byte-identical.
pub(in crate::check) fn hybrid_native_authoritative_enabled() -> bool {
    hybrid_native_enabled()
        && std::env::var_os("TY_HYBRID_NATIVE_AUTHORITATIVE").is_some_and(|v| v == "1")
}

const ROUTER_PILOT_PARENTS: u64 = 16_384;
const ROUTER_TRIAL_PARENTS: u32 = 64;
const ROUTER_MIN_SKIP_PERCENT: u128 = 80;
// The trial compares local successor generation, while the pre-router route
// may additionally save materialization through inline streaming dedup. Keep a
// large margin before changing the end-to-end route; a small local win is not
// evidence that batch consumption will also win.
const ROUTER_MIN_GENERATION_SPEEDUP_PERCENT: u128 = 40;
const ROUTER_PARITY_SAMPLE_STRIDE: u64 = 4_096;
const ROUTER_MAX_SUCCESSORS_PER_PARENT: usize = 4_096;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RouterRequest {
    Disabled,
    Auto,
    Forced,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
enum RouterPhase {
    #[default]
    Disabled,
    Pilot,
    Trial,
    Active,
    Forced,
    Declined,
    Failback,
}

/// Resolve the standalone router's tri-state policy.
///
/// `TY_ROUTER=1` preserves the explicit force-on diagnostic. Any other
/// present value is a stable opt-out. With the variable absent, only CLI AUTO
/// engine selection may run the conservative online pilot; library callers
/// and explicitly selected backends are unchanged.
fn router_request_from(value: Option<&std::ffi::OsStr>, auto_select: bool) -> RouterRequest {
    match value {
        Some(value) if value.to_str().is_some_and(tla_backend::env_flag_enabled) => {
            RouterRequest::Forced
        }
        Some(_) => RouterRequest::Disabled,
        None if auto_select => RouterRequest::Auto,
        None => RouterRequest::Disabled,
    }
}

fn router_request() -> RouterRequest {
    let value = std::env::var_os("TY_ROUTER");
    router_request_from(
        value.as_deref(),
        crate::check::debug::trust_cg_auto_select_enabled(),
    )
}

#[inline]
fn router_skip_rate_admitted(skips: u64, checks: u64) -> bool {
    checks != 0
        && u128::from(skips).saturating_mul(100)
            >= u128::from(checks).saturating_mul(ROUTER_MIN_SKIP_PERCENT)
}

#[inline]
fn router_timing_admitted(batch_ns: u128, whole_next_ns: u128) -> bool {
    if whole_next_ns == 0 {
        return false;
    }
    let retained_percent = 100_u128.saturating_sub(ROUTER_MIN_GENERATION_SPEEDUP_PERCENT);
    let allowed_batch_ns =
        (whole_next_ns / 100) * retained_percent + ((whole_next_ns % 100) * retained_percent) / 100;
    batch_ns <= allowed_batch_ns
}

/// WP-29 lever 1: whether the per-(parent, action) enabling PRE-CHECK runs
/// before the batch path's interpreter enumeration (default ON;
/// `TY_HYBRID_ACTION_GUARD_PRECHECK=0` (or
/// `TY_ROUTER_GUARD_PRECHECK=0` for the standalone router) restores the
/// pre-WP-29 behaviour where every instance enters the enumerator).
///
/// The pre-check is a sound UNDER-approximation of enabledness: it only ever
/// reports "definitely disabled", and only when a syntactically extracted
/// state-only conjunct of the action evaluates to `FALSE` in the parent. Every
/// other outcome (no extractable guard, a non-boolean value, ANY evaluation
/// error) falls through to the full enumeration unchanged.
pub(in crate::check) fn action_guard_precheck_enabled() -> bool {
    static FLAG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *FLAG.get_or_init(|| {
        !matches!(
            std::env::var("TY_HYBRID_ACTION_GUARD_PRECHECK").as_deref(),
            Ok("0")
        )
    })
}

/// Router-local guard-precheck kill switch. It is consulted only when the
/// standalone router is the sole route owner, so router diagnostics cannot
/// change POR, coverage, or hybrid behavior.
pub(in crate::check) fn router_guard_precheck_enabled() -> bool {
    static FLAG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *FLAG.get_or_init(|| {
        !std::env::var("TY_ROUTER_GUARD_PRECHECK")
            .is_ok_and(|value| tla_backend::env_flag_disabled(&value))
    })
}

/// WP-34: whether the enabling pre-check also runs while POR is engaged
/// (default ON; `TY_HYBRID_GUARD_PRECHECK_POR=0` restores WP-29's exclusion).
///
/// POR is the default path for real users, and the pre-check is admissible to
/// it: a provably-empty per-action enumeration is invisible to the ample-set
/// computation and to the whole-Next parity self-check (see the call site in
/// `run_gen.rs`). The parity self-check remains the fail-closed net.
pub(in crate::check) fn guard_precheck_under_por_enabled() -> bool {
    static FLAG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *FLAG.get_or_init(|| {
        !matches!(
            std::env::var("TY_HYBRID_GUARD_PRECHECK_POR").as_deref(),
            Ok("0")
        )
    })
}

/// WP-29 lever 2: whether authoritative native successors are reconstructed
/// DELTA-wise — decoding only the flat-admissible variables whose slots differ
/// from the parent's projected buffer and `Arc`-sharing the rest straight off
/// the parent (default ON; `TY_HYBRID_DELTA_RECONSTRUCT=0` restores the
/// whole-buffer decode).
pub(in crate::check) fn delta_reconstruct_enabled() -> bool {
    static FLAG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *FLAG.get_or_init(|| {
        !matches!(
            std::env::var("TY_HYBRID_DELTA_RECONSTRUCT").as_deref(),
            Ok("0")
        )
    })
}

/// WP-29 lever 2 self-check (`TY_HYBRID_DELTA_RECONSTRUCT_VERIFY=1`, default
/// OFF): reconstruct every authoritative native successor BOTH ways — delta
/// and whole-buffer — and compare them variable by variable. A disagreement
/// counts into `mismatch_fallback` (the loud alarm) and the whole-buffer state
/// is used, so the arm stays fail-closed while the check is on.
fn delta_reconstruct_verify_enabled() -> bool {
    static FLAG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *FLAG.get_or_init(|| {
        std::env::var_os("TY_HYBRID_DELTA_RECONSTRUCT_VERIFY").is_some_and(|v| v == "1")
    })
}

/// Maximum operator/quantifier/LET nesting the guard extractor descends.
const GUARD_EXTRACT_MAX_DEPTH: usize = 8;

/// Maximum AST node count of an accepted guard. A guard is evaluated on EVERY
/// (parent, action) instance, so an expensive one would just move the cost
/// rather than remove it.
const GUARD_MAX_NODES: usize = 64;

/// WP-34 lever 1: maximum number of state-only conjuncts collected into ONE
/// action guard. Conjuncts evaluate left-to-right with `/\` short-circuit, so
/// the leading (usually most selective) one still decides most instances; the
/// cap bounds the tail cost on the instances it does not decide.
const GUARD_MAX_TERMS: usize = 6;

/// WP-34 lever 1: node budget across the WHOLE synthesized guard (the sum over
/// its accepted conjuncts), on top of the per-conjunct [`GUARD_MAX_NODES`].
const GUARD_TOTAL_MAX_NODES: usize = 192;

/// WP-34 lever 1: bound on extractor work (AST nodes visited) for one action.
/// Extraction runs ONCE per action, but operator unfolding under a conjunctive
/// descent is exponential in the worst case; exhausting the budget declines.
const GUARD_EXTRACT_MAX_VISITS: usize = 4096;

/// WP-34 lever 1: one-shot dump of what the guard extractor accepted for each
/// detected action (`TY_HYBRID_GUARD_DEBUG=1`). Pure diagnostics.
pub(in crate::check) fn hybrid_guard_debug_enabled() -> bool {
    static FLAG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *FLAG.get_or_init(|| std::env::var_os("TY_HYBRID_GUARD_DEBUG").is_some_and(|v| v == "1"))
}

/// WP-34 lever 2: whether the batch consumer normalizes lazy values against
/// the parent (walking only the variables a successor actually rewrote) instead
/// of walking every variable of every successor. Default ON;
/// `TY_HYBRID_CONSUME_DELTA=0` restores the unconditional whole-state pass for
/// A/B measurement. Both produce the identical state — the delta path only
/// elides scans of variables proven identical to a lazy-free parent variable.
pub(in crate::check) fn consume_delta_materialize_enabled() -> bool {
    static FLAG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *FLAG.get_or_init(|| !matches!(std::env::var("TY_HYBRID_CONSUME_DELTA").as_deref(), Ok("0")))
}

/// WP-34 lever 1: whether the extractor may collect MORE than the single
/// leading conjunct (default ON; `TY_HYBRID_GUARD_WIDE=0` restores WP-29's
/// leading-conjunct-only extractor, depth accounting included, for A/B
/// measurement).
fn guard_wide_extract_enabled() -> bool {
    static FLAG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *FLAG.get_or_init(|| !matches!(std::env::var("TY_HYBRID_GUARD_WIDE").as_deref(), Ok("0")))
}

/// Per-action consecutive-clean threshold before native becomes authoritative
/// for that action (`TY_HYBRID_BURN_IN`, default 4096).
const HYBRID_BURN_IN_DEFAULT: u64 = 4096;

/// Post-flip deterministic sampling period: 1-in-K authoritative-eligible
/// instances keep the full differential (`TY_HYBRID_SAMPLE`, default 64).
/// `0` disables sampling (never sample); `1` samples every instance (native
/// then never actually skips the interpreter — a debugging arm).
const HYBRID_SAMPLE_DEFAULT: u64 = 64;

fn hybrid_env_u64(name: &str, default: u64) -> u64 {
    std::env::var(name)
        .ok()
        .and_then(|v| v.trim().parse::<u64>().ok())
        .unwrap_or(default)
}

/// TEST-ONLY fault injection (`TY_HYBRID_AUTHORITATIVE_INJECT_SAMPLED_CORRUPTION=1`):
/// corrupt one slot of the first SAMPLED instance's first native successor
/// buffer, proving end-to-end that a sampled divergence trips the permanent
/// global fail-back while the run stays state-exact (the sampled instance
/// keeps the interpreter successors). Inert unless explicitly set; the
/// corruption can only ever make dispatch MORE conservative.
fn hybrid_inject_sampled_corruption_enabled() -> bool {
    std::env::var_os("TY_HYBRID_AUTHORITATIVE_INJECT_SAMPLED_CORRUPTION").is_some_and(|v| v == "1")
}

/// WP-25 diagnosis surface (`TY_HYBRID_DIVERGENCE_DUMP=N`): dump the first `N`
/// native/interpreter divergences with the parent state, the interpreter
/// successor, and every native candidate — decoded per variable AND as raw
/// buffer slots. Pure stderr diagnostics behind an unset-by-default env var;
/// the routing decision is identical either way.
fn hybrid_divergence_dump_budget() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| hybrid_env_u64("TY_HYBRID_DIVERGENCE_DUMP", 0) as usize)
}

static HYBRID_DIVERGENCE_DUMPED: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

/// Dump one divergent (parent, action) instance (see
/// [`hybrid_divergence_dump_budget`]). Never called unless the env var is set.
fn hybrid_dump_divergence(
    view: &HybridFlatView,
    parent: &ArrayState,
    interp_succ: &ArrayState,
    interp_view: &FlatState,
    registry: &VarRegistry,
    candidates: &HybridNativeCandidates,
) {
    let budget = hybrid_divergence_dump_budget();
    if budget == 0 {
        return;
    }
    let seq = HYBRID_DIVERGENCE_DUMPED.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    if seq >= budget {
        return;
    }
    let n = view.var_count();
    let names = registry.names();
    let show = |s: &ArrayState| -> String {
        (0..n)
            .map(|i| format!("{}={:?}", names[i], s.get(VarIndex::new(i))))
            .collect::<Vec<_>>()
            .join(" ")
    };
    eprintln!(
        "[hybrid-div] #{seq} action_idx={} mode={:?} candidates={}",
        candidates.action_idx,
        candidates.mode,
        candidates.views.len()
    );
    eprintln!("[hybrid-div]   parent      : {}", show(parent));
    eprintln!("[hybrid-div]   interp_succ : {}", show(interp_succ));
    for (i, cand) in candidates.views.iter().enumerate() {
        let slots: Vec<usize> = (0..interp_view.num_slots())
            .filter(|&s| cand.buffer().get(s) != interp_view.buffer().get(s))
            .collect();
        match view.reconstruct(parent, cand, registry) {
            Some(arr) => {
                let diffs: Vec<String> = (0..n)
                    .filter(|&i| arr.get(VarIndex::new(i)) != interp_succ.get(VarIndex::new(i)))
                    .map(|i| {
                        format!(
                            "{}: interp={:?} native={:?}",
                            names[i],
                            interp_succ.get(VarIndex::new(i)),
                            arr.get(VarIndex::new(i))
                        )
                    })
                    .collect();
                eprintln!(
                    "[hybrid-div]   cand#{i} consumed={} slot_diffs={:?} var_diffs=[{}]",
                    candidates.consumed[i],
                    slots,
                    diffs.join(" | ")
                );
            }
            None => eprintln!(
                "[hybrid-div]   cand#{i} consumed={} slot_diffs={:?} (reconstruct declined)",
                candidates.consumed[i], slots
            ),
        }
    }
}

/// Deterministic sample key: fixed-seed FxHash over (action index, projected
/// parent buffer). Reproducible across reruns of the same binary — unlike a
/// mod-counter, which drifts with dispatch order — so a sampled instance is
/// sampled again on every rerun.
fn hybrid_sample_hash(parent_buffer: &[i64], action_idx: usize) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = rustc_hash::FxHasher::default();
    action_idx.hash(&mut h);
    parent_buffer.hash(&mut h);
    h.finish()
}

/// Fixed-seed FxHash over the resolved dispatch key set (in resolution order).
/// Burn-in evidence is only valid against the exact compiled key set it was
/// accumulated for; any drift restarts the count.
fn hybrid_keys_hash(keys: &[String]) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = rustc_hash::FxHasher::default();
    for key in keys {
        key.hash(&mut h);
    }
    h.finish()
}

/// Whether per-action eligibility uses the **M1** rule
/// (`writes ⊆ flat-admissible`, compound reads serviced by the compound-read
/// callout) rather than M0's strict `reads ∪ writes ⊆ flat-admissible`.
///
/// Default ON whenever the hybrid path itself is enabled — the whole module is
/// already behind `TY_HYBRID_FLAT_VIEW`, so the default build is untouched
/// either way. `TY_HYBRID_M1_READS=0` pins the strict M0 rule for A/B
/// comparison against the M0 baselines.
///
/// Relaxing this is sound for the routing itself: reconstruction Arc-shares
/// every compound var from the parent, which reproduces the successor exactly
/// as long as no compound var is WRITTEN — and that is precisely the
/// `writes ⊆ flat-admissible` half, which M1 keeps. Compound READS are the
/// half M0 was over-strict about. Native admission is a separate, stricter
/// gate ([`ModelChecker::hybrid_native_footprint_admitted`]): a compiled
/// artifact may only read a compound var it explicitly DECLARED as a callout
/// read.
fn hybrid_m1_read_rule_enabled() -> bool {
    !std::env::var_os("TY_HYBRID_M1_READS").is_some_and(|v| v == "0")
}

/// WP-17: coarse per-bucket wall-clock accumulators for the hybrid dispatch
/// path (`TY_HYBRID_PERF_DEBUG=1`). Pure diagnostics: nanosecond sums printed
/// with the end-of-run `[hybrid]` summary; when off, every probe is a single
/// branch on `enabled` and no clock is read.
#[derive(Debug, Clone, Copy, Default)]
pub(super) struct HybridPerfTimers {
    pub(super) enabled: bool,
    /// WP-34: arm the batch consumer's inner split as well
    /// (`TY_HYBRID_CONSUME_PERF=1`). Separate from `enabled` so the coarse
    /// bucket table stays free of the extra per-successor clock reads.
    pub(super) consume_detail: bool,
    /// Per-parent hybrid projection (`hybrid_project_parent_for_dispatch`).
    pub(super) project_parent_ns: u64,
    /// Per-successor projections of interpreter successors (route + shadow).
    pub(super) project_succ_ns: u64,
    /// Key resolution + footprint dual gate (memoized; first hit per action).
    pub(super) key_resolve_ns: u64,
    /// Compiled-artifact execution (`try_trust_cg_hybrid_action_by_keys`).
    pub(super) native_exec_ns: u64,
    /// Flat-view -> ArrayState reconstruction (route + authoritative).
    pub(super) reconstruct_ns: u64,
    /// Value-equality differential (`states_value_equal`).
    pub(super) value_eq_ns: u64,
    /// State/action-constraint checks in the per-action loop (all paths).
    pub(super) constraints_ns: u64,
    /// `ensure_incremental_fp_cache_from` + `to_state` (all per-action paths).
    pub(super) fp_to_state_ns: u64,
    /// Interpreter successor enumeration (`enumerate_successors_body`).
    pub(super) interp_enum_ns: u64,
    /// WP-26: per-successor `ArrayState` construction on the interpreter branch
    /// of the per-action loop (legacy: `ArrayState::from_state`; diff:
    /// `DiffSuccessor::into_array_state`).
    pub(super) interp_succ_build_ns: u64,
    /// WP-26: per-parent batch-path setup inside
    /// `enumerate_per_action_successor_sets` — `ArrayState::from_state(state)`
    /// plus the parent fingerprint warm.
    pub(super) parent_setup_ns: u64,
    /// WP-26: total wall inside `generate_successors_array_raw` as seen by the
    /// full-state batch consumer (the per-action path's grand total; anything
    /// not covered by the finer buckets is per-action-loop overhead).
    pub(super) batch_gen_ns: u64,
    /// WP-26: streaming diff-BFS Phase A (enumeration + inline fingerprint +
    /// inline dedup). The plain arm's counterpart to `interp_enum_ns`.
    pub(super) stream_phase_a_ns: u64,
    /// WP-26: streaming diff-BFS Phase B (materialize + invariants + enqueue
    /// for post-dedup NEW successors only).
    pub(super) stream_phase_b_ns: u64,
    /// WP-26: the batch consumer's fused per-successor pass in
    /// `process_full_state_successors` (materialize -> fingerprint -> dedup ->
    /// invariant -> enqueue), paid over EVERY successor. The streaming engine's
    /// counterpart is split across `stream_phase_a` (fingerprint + dedup, no
    /// materialization) and `stream_phase_b` (post-dedup survivors only).
    pub(super) batch_consume_ns: u64,
    /// WP-34 lever 2: the `batch_consume` split, so the consumer's cost is
    /// attributable rather than a single lump. `materialize` is the lazy-value
    /// normalization pass, `fp` the fingerprint (usually a cached-hit),
    /// `observe` the candidate observers, `dedup` the seen-set probe, and
    /// `finish` the post-dedup survivor path (invariants + enqueue).
    pub(super) consume_materialize_ns: u64,
    pub(super) consume_fp_ns: u64,
    pub(super) consume_observe_ns: u64,
    pub(super) consume_dedup_ns: u64,
    pub(super) consume_finish_ns: u64,
    /// WP-34 lever 2: successors entering / surviving the batch consumer, and
    /// the ones whose lazy-value scan was skipped because every variable was
    /// bit-identical to a parent variable already proven lazy-free.
    pub(super) consume_succ: u64,
    pub(super) consume_survivors: u64,
    pub(super) consume_lazy_scan_skipped: u64,
    pub(super) consume_lazy_vars_scanned: u64,
    pub(super) consume_lazy_vars_total: u64,
    /// WP-26: interpreter successors materialized on the per-action branch.
    pub(super) interp_succ_count: u64,
    /// WP-26: per-(parent, action) interpreter enumerations entered.
    pub(super) interp_enum_calls: u64,
    /// WP-26: of those, the ones that produced NO successor — the action was
    /// disabled in this parent, so the whole call is enumerator entry overhead
    /// plus guard evaluation. The streaming engine pays this once per parent
    /// for all disjuncts together.
    pub(super) interp_enum_empty_calls: u64,
    /// WP-26: wall spent inside those zero-successor enumerations.
    pub(super) interp_enum_empty_ns: u64,
    /// WP-26: successors seen by the streaming diff path (pre-dedup).
    pub(super) stream_succ_count: u64,
    /// WP-26: parents expanded through the per-action batch path.
    pub(super) batch_parents: u64,
    /// WP-26: parents expanded through the streaming diff path.
    pub(super) stream_parents: u64,
    /// WP-29 lever 1: wall spent in the per-(parent, action) enabling
    /// pre-check, on BOTH outcomes (a decided "definitely disabled" and an
    /// undecided fall-through).
    pub(super) guard_precheck_ns: u64,
    /// WP-29 lever 1: pre-checks evaluated.
    pub(super) guard_precheck_calls: u64,
    /// WP-29 lever 1: pre-checks that decided "definitely disabled" — each one
    /// is a whole interpreter enumeration (enumerator entry + quantifier domain
    /// construction + guard evaluation) that never ran.
    pub(super) guard_precheck_skips: u64,
    /// WP-29 lever 2: authoritative native successors reconstructed
    /// delta-wise, and the flat-admissible variables actually decoded for them
    /// (vs `delta_reconstruct_vars_total`, the whole-buffer decode's count).
    pub(super) delta_reconstructed: u64,
    pub(super) delta_reconstruct_vars_decoded: u64,
    pub(super) delta_reconstruct_vars_total: u64,
}

impl HybridPerfTimers {
    /// Start a probe: `Some(now)` only when the perf switch is on.
    #[inline]
    pub(super) fn start(&self) -> Option<std::time::Instant> {
        self.enabled.then(std::time::Instant::now)
    }

    /// WP-34: start a batch-consumer inner-split probe. Off unless
    /// `TY_HYBRID_CONSUME_PERF=1` armed it alongside `TY_HYBRID_PERF_DEBUG=1`.
    #[inline]
    pub(super) fn start_consume(&self) -> Option<std::time::Instant> {
        (self.enabled && self.consume_detail).then(std::time::Instant::now)
    }
}

/// Accumulate a probe started with [`HybridPerfTimers::start`].
#[inline]
pub(super) fn perf_acc(bucket: &mut u64, t0: Option<std::time::Instant>) {
    if let Some(t0) = t0 {
        *bucket += t0.elapsed().as_nanos() as u64;
    }
}

/// Runtime counters for the hybrid dispatch path (load-independent evidence).
#[derive(Debug, Clone, Copy, Default)]
pub(super) struct HybridDispatchStats {
    /// Hybrid-eligible successors that round-tripped through the projection and
    /// matched the interpreter successor exactly (the "native share" that will
    /// dispatch through trust-cg once its M0 lands). In M0 these still run the
    /// interpreter under the hood (stub) — the enqueued state is byte-identical.
    ///
    /// Counts BOTH the per-action interpreter path (where the reconstructed
    /// successor is actually enqueued in place of the interpreter's) AND the
    /// diff-BFS paths (where it is a validating shadow — computed, compared,
    /// counted, but the original successor is kept so state counts cannot move).
    pub(super) routed: u64,
    /// Fail-closed fallbacks: the reconstructed successor diverged from the
    /// interpreter successor (a footprint slip or a projection round-trip
    /// loss), OR — on the native path — a native/interpreter successor-set
    /// divergence (an interpreter successor no native successor matched, or a
    /// native successor the interpreter never produced). MUST stay 0 on a
    /// sound run — a nonzero value is the loud alarm.
    pub(super) mismatch_fallback: u64,
    /// project/reconstruct declined (fixed-layout could not encode a value);
    /// fell back to the interpreter successor. Sound, just not routed.
    pub(super) projection_declined: u64,
    /// Native executions: (parent, action) pairs whose compiled hybrid
    /// artifacts actually ran (TY_HYBRID_NATIVE=1).
    pub(super) native_dispatched: u64,
    /// Interpreter successors matched byte-exactly by a native successor
    /// buffer (the strongest form of the differential: buffer equality is
    /// checked BEFORE the value-equality reconstruct differential).
    pub(super) native_matched: u64,
    /// Interpreter successors of a natively-executed action that NO native
    /// successor matched (each also counts in `mismatch_fallback`).
    pub(super) native_unmatched_interp: u64,
    /// Native successors the interpreter never produced, detected at
    /// action end (each also counts in `mismatch_fallback`).
    pub(super) native_residue: u64,
    /// Native execution declined at admission (artifact/key missing, footprint
    /// dual-gate mismatch, width mismatch, loop-kernel ABI). Sound: the action
    /// instance stays on the interpreter + shadow path.
    pub(super) native_declined: u64,
    /// Native runtime errors (JitStatus::RuntimeError / fallback statuses).
    /// Sound (interpreter stays authoritative), but tracked loudly.
    pub(super) native_errors: u64,
    /// WP-21: typed `TypeMismatch` shape-guard declines — the compiled code
    /// hit a fail-closed runtime shape guard (union-arm read, capacity /
    /// universe confinement, member not-found), the canonical case being a
    /// LET def touching a union var on a parent whose enabling guard is
    /// false (args-NIL). A cheap per-parent "not applicable", NOT an alarm:
    /// the instance falls back to the interpreter exactly like a decline.
    /// Kept OUT of `native_errors` for alarm hygiene.
    pub(super) native_guard_declined: u64,
    /// WP-14: action instances whose native successor set was enqueued
    /// AUTHORITATIVELY — the interpreter enumeration was skipped entirely.
    pub(super) authoritative_dispatched: u64,
    /// WP-14: post-flip instances deterministically sampled back into the
    /// full interpreter differential.
    pub(super) sampled_checks: u64,
    /// WP-14: sampled instances whose differential diverged. ANY nonzero
    /// value trips the permanent whole-run fail-back. MUST stay 0.
    pub(super) sampled_mismatches: u64,
}

/// WP-14: how one (parent, action) native execution participates in the run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum HybridInstanceMode {
    /// Interpreter authoritative, full differential (authoritative gate off,
    /// burn-in still accumulating, or permanent fail-back tripped).
    Shadow,
    /// Action has flipped, but this instance was deterministically sampled
    /// back into the full differential — interpreter stays authoritative.
    Sampled,
    /// Action has flipped and this instance was not sampled: the native
    /// candidates ARE the successors; the interpreter enumeration is skipped.
    Authoritative,
}

/// WP-14: the per-action burn-in / sampling / permanent-fail-back state
/// machine. Pure bookkeeping — no env reads, no I/O — so the flip/reset/
/// fail-back contract is directly unit-testable. One slot per detected
/// action, aligned with [`HybridDispatchState::eligible`].
#[derive(Debug, Clone, Default)]
pub(super) struct HybridAuthoritativeMachine {
    /// Consecutive clean differentials required before an action flips.
    burn_in_n: u64,
    /// Post-flip sampling period (`0` = never sample, `1` = always).
    sample_k: u64,
    /// Permanent whole-run fail-back: once set, every instance is `Shadow`
    /// and no action ever flips (back) for the rest of the run.
    failback: bool,
    slots: Vec<HybridActionBurnIn>,
}

#[derive(Debug, Clone, Default)]
struct HybridActionBurnIn {
    /// Consecutive fully-clean differentials since the last reset.
    consecutive_clean: u64,
    /// Whether native is currently authoritative for this action.
    authoritative: bool,
    /// Whether this action ever executed natively (for `burn_in_pending`).
    dispatched_any: bool,
    /// The resolved-key-set hash the current evidence was accumulated
    /// against; drift restarts the count.
    keys_hash: Option<u64>,
}

impl HybridAuthoritativeMachine {
    pub(super) fn new(burn_in_n: u64, sample_k: u64, action_count: usize) -> Self {
        Self {
            burn_in_n,
            sample_k,
            failback: false,
            slots: vec![HybridActionBurnIn::default(); action_count],
        }
    }

    fn slot_mut(&mut self, action_idx: usize) -> &mut HybridActionBurnIn {
        if action_idx >= self.slots.len() {
            self.slots
                .resize_with(action_idx + 1, HybridActionBurnIn::default);
        }
        &mut self.slots[action_idx]
    }

    /// Decide how one just-executed native instance participates. Called only
    /// when the authoritative gate is on and the native execution succeeded.
    pub(super) fn decide_mode(
        &mut self,
        action_idx: usize,
        keys_hash: u64,
        sample_hash: u64,
    ) -> HybridInstanceMode {
        let failback = self.failback;
        let sample_k = self.sample_k;
        let slot = self.slot_mut(action_idx);
        slot.dispatched_any = true;
        if failback {
            return HybridInstanceMode::Shadow;
        }
        // Burn-in evidence is only valid for the exact compiled key set it
        // was accumulated against.
        if slot.keys_hash != Some(keys_hash) {
            slot.keys_hash = Some(keys_hash);
            slot.consecutive_clean = 0;
            slot.authoritative = false;
        }
        if !slot.authoritative {
            return HybridInstanceMode::Shadow;
        }
        if sample_k > 0 && sample_hash % sample_k == 0 {
            HybridInstanceMode::Sampled
        } else {
            HybridInstanceMode::Authoritative
        }
    }

    /// Record the differential outcome of one Shadow/Sampled instance.
    ///
    /// `clean` = the FULL differential completed with zero divergence and
    /// zero declines (every interpreter successor byte-matched a native
    /// candidate, value equality held, no residue, no projection/reconstruct
    /// decline). `dirty` = a SEMANTIC divergence was observed (unmatched
    /// interpreter successor, value inequality after a byte match, or native
    /// residue) — strictly stronger than `!clean`, which also covers benign
    /// differential-incomplete declines.
    ///
    /// Returns `true` iff this outcome newly tripped the permanent fail-back.
    pub(super) fn record_result(
        &mut self,
        action_idx: usize,
        mode: HybridInstanceMode,
        clean: bool,
        dirty: bool,
    ) -> bool {
        if self.failback {
            // The fail-back is PERMANENT: no evidence accumulates after it,
            // so no action can ever re-flip within this run.
            return false;
        }
        match mode {
            HybridInstanceMode::Authoritative => false, // no differential ran
            HybridInstanceMode::Sampled => {
                if dirty || !clean {
                    // A flipped action failed its sampled differential:
                    // permanent whole-run fail-back (fail closed on the
                    // incomplete-differential case too — an authoritative
                    // action whose sampled check cannot complete has lost its
                    // ongoing evidence stream).
                    self.trip_failback()
                } else {
                    false
                }
            }
            HybridInstanceMode::Shadow => {
                let burn_in_n = self.burn_in_n;
                let any_flipped = self.slots.iter().any(|s| s.authoritative);
                let slot = self.slot_mut(action_idx);
                if clean {
                    if !slot.authoritative {
                        slot.consecutive_clean += 1;
                        if slot.consecutive_clean >= burn_in_n {
                            slot.authoritative = true;
                        }
                    }
                    false
                } else {
                    slot.consecutive_clean = 0;
                    // A SEMANTIC divergence observed anywhere after ANY action
                    // has flipped invalidates trust in the shared native
                    // machinery: fail back globally (strictly more
                    // conservative than failing back only on sampled
                    // mismatches). A benign incomplete differential only
                    // resets this action's count.
                    if dirty && any_flipped {
                        self.trip_failback()
                    } else {
                        false
                    }
                }
            }
        }
    }

    /// Returns `true` iff the fail-back was newly tripped by this call.
    fn trip_failback(&mut self) -> bool {
        let newly = !self.failback;
        self.failback = true;
        for slot in &mut self.slots {
            slot.authoritative = false;
            slot.consecutive_clean = 0;
        }
        newly
    }

    pub(super) fn failback(&self) -> bool {
        self.failback
    }

    /// Actions currently dispatching authoritatively.
    pub(super) fn authoritative_action_count(&self) -> usize {
        self.slots.iter().filter(|s| s.authoritative).count()
    }

    /// Actions that executed natively but have not (or no longer) flipped.
    pub(super) fn burn_in_pending_count(&self) -> usize {
        self.slots
            .iter()
            .filter(|s| s.dispatched_any && !s.authoritative)
            .count()
    }
}

// ===================== WP-29 lever 1: the enabling pre-check ================
//
// The batch (per-action) engine enters the unified enumerator ONCE PER
// (parent, action). On btree that is ~8 entries per parent of which ~88%
// produce zero successors: the action is disabled and the whole entry —
// working-state clone, undo stack, sink, `EnumParams`, provenance scope,
// operator resolution, and for a quantified action the FULL construction and
// traversal of its quantifier domains — is paid to learn nothing.
//
// The streaming engine never pays that: all disjuncts live inside ONE
// enumeration, so a disabled branch costs a guard evaluation and nothing else.
// This is the structural engine gap WP-26 measured.
//
// The pre-check closes it by evaluating, per (parent, action), a syntactically
// extracted STATE-ONLY conjunct `g` of the action with `action => g`. `g` FALSE
// therefore proves the action has no successors in this parent. It is a strict
// UNDER-approximation of enabledness: `g` TRUE, a non-boolean `g`, ANY
// evaluation error, or no extractable `g` at all all fall through to the
// unchanged full enumeration, so the reachable-state set cannot move.
//
// This mirrors machinery the enumerator already carries one level deeper —
// `try_let_guard_first_shortcircuit` (btree's LET-wrapped disabled actions),
// `PcGuardHoist` (PlusCal `pc = "label"` Or-branch pruning) and
// `first_guard_sched` — hoisted out to where it also elides the enumerator
// entry itself.
//
// ------------------------------- WP-34 -------------------------------------
//
// WP-29 shipped this covering 4 of btree's 7 interpreter actions per parent,
// leaving 1,147,713 zero-successor enumerations (7,523 ms) unskipped, and read
// the gap as "the rest have a leading conjunct the whitelist does not cover".
// MEASURED, that diagnosis was wrong: all 11 actions lead with a state-only
// `state = <CONSTANT>` conjunct, which the whitelist already accepted. What
// stopped them was the extractor's own DEPTH budget — it charged
// `GUARD_EXTRACT_MAX_DEPTH` for every `/\` level while walking to the leftmost
// conjunct, and a bulleted conjunction list of n conjuncts nests n deep. The
// three declining actions were exactly the long ones (`SplitRootLeaf` 10
// conjuncts, `SplitLeaf` / `SplitRootInner` 9), plus `UpdateReq` under its
// `\E` + `Apply` + `LET` wrappers.
//
// WP-34 therefore charges `depth` only for the descents that can recurse
// without bound (operator unfolds and scope introductions) and bounds the
// finite propositional walk with a separate visit budget. Coverage went
// 4.0 -> 7.0 actions/parent, zero-successor enumerations 1,147,713 -> 32,738,
// and `interp_enum_empty` 7,523 ms -> 97 ms.
//
// The widening on top of that — collecting EVERY state-only conjunct of the
// top-level conjunction instead of the leading one, admitting `\/` and `IF`,
// and extending the purity whitelist to applications, `CASE`, `LET`, `EXCEPT`,
// records and the bounded binders — buys nothing on btree (whose actions each
// have exactly one state-only conjunct) but is what lets specs that lead with
// a primed conjunct, or discriminate on a non-leading one, reach the lever at
// all. `TY_HYBRID_GUARD_WIDE=0` restores the WP-29 extractor verbatim.

/// Whether `expr` is a pure STATE predicate the pre-check may evaluate on its
/// own against the parent state: no primed reference, no action content, no
/// binder (so the free-name check below is exact), no INSTANCE / module
/// indirection, and bounded in size.
///
/// The whitelist stays a whitelist. Anything not listed — `Prime`,
/// `Unchanged`, `Enabled`, the temporal operators, `ModuleRef`,
/// `InstanceExpr`, `SubstIn`, `Lambda`, `OpRef`, `Choose` — makes the guard
/// unusable and the action enumerates as before.
///
/// WP-34 widens it to the remaining STATE-LEVEL shapes: applications of
/// module-scope operators and of a fixed pure-builtin set, `CASE`, `LET`,
/// `EXCEPT`, records/record sets/function sets, and the bounded binders
/// (`\A`, `\E`, `{x \in S : P}`, `{e : x \in S}`, `[x \in S |-> e]`). Every one
/// is a total function of the CURRENT state — none can name a primed variable
/// (`Prime` is still rejected everywhere, including under an unfolded operator
/// body), so the `action => guard` implication the pre-check needs is
/// unaffected. Binders only ever ADD names, which can only make the free-name
/// rejection in [`guard_candidate`] fire more often (it tests mentions, not
/// free occurrences), so admitting them cannot widen what the pre-check
/// decides on a bound name.
///
/// The budget stays the same and is the cost control: at most
/// [`GUARD_MAX_NODES`] nodes and [`GUARD_EXTRACT_MAX_DEPTH`] operator unfolds,
/// checked BEFORE the shape match, so a recursive operator (btree's
/// `FindLeafNode`) exhausts the depth and the whole candidate is rejected.
fn guard_expr_is_pure(
    ctx: &crate::eval::EvalCtx,
    op_defs: &rustc_hash::FxHashMap<String, tla_core::ast::OperatorDef>,
    expr: &tla_core::ast::Expr,
    nodes: &mut usize,
    depth: usize,
) -> bool {
    use tla_core::ast::Expr as E;

    *nodes += 1;
    if *nodes > GUARD_MAX_NODES || depth > GUARD_EXTRACT_MAX_DEPTH {
        return false;
    }

    match expr {
        E::Bool(_) | E::Int(_) | E::String(_) | E::StateVar(_, _, _) => true,
        // A bare name is admissible when it is NOT a module operator (a
        // CONSTANT, a model value, a config-substituted constant), or when it
        // is a zero-arg operator whose body is itself pure. Anything else
        // could hide primes or action content behind the name.
        E::Ident(name, _) => {
            let resolved = ctx.resolve_op_name(name.as_str());
            match op_defs.get(resolved) {
                None => true,
                Some(def) => {
                    def.params.is_empty()
                        && !crate::eval::should_prefer_builtin_override(resolved, def, 0, ctx)
                        && guard_expr_is_pure(ctx, op_defs, &def.body.node, nodes, depth + 1)
                }
            }
        }
        E::Label(label) => guard_expr_is_pure(ctx, op_defs, &label.body.node, nodes, depth),
        E::Not(a)
        | E::Neg(a)
        | E::Domain(a)
        | E::Powerset(a)
        | E::BigUnion(a)
        | E::RecordAccess(a, _) => guard_expr_is_pure(ctx, op_defs, &a.node, nodes, depth),
        E::And(a, b)
        | E::Or(a, b)
        | E::Implies(a, b)
        | E::Equiv(a, b)
        | E::Eq(a, b)
        | E::Neq(a, b)
        | E::Lt(a, b)
        | E::Leq(a, b)
        | E::Gt(a, b)
        | E::Geq(a, b)
        | E::In(a, b)
        | E::NotIn(a, b)
        | E::Subseteq(a, b)
        | E::Union(a, b)
        | E::Intersect(a, b)
        | E::SetMinus(a, b)
        | E::FuncApply(a, b)
        | E::Add(a, b)
        | E::Sub(a, b)
        | E::Mul(a, b)
        | E::Div(a, b)
        | E::IntDiv(a, b)
        | E::Mod(a, b)
        | E::Pow(a, b)
        | E::Range(a, b) => {
            guard_expr_is_pure(ctx, op_defs, &a.node, nodes, depth)
                && guard_expr_is_pure(ctx, op_defs, &b.node, nodes, depth)
        }
        E::If(cond, then_branch, else_branch) => {
            guard_expr_is_pure(ctx, op_defs, &cond.node, nodes, depth)
                && guard_expr_is_pure(ctx, op_defs, &then_branch.node, nodes, depth)
                && guard_expr_is_pure(ctx, op_defs, &else_branch.node, nodes, depth)
        }
        E::SetEnum(items) | E::Tuple(items) | E::Times(items) => items
            .iter()
            .all(|item| guard_expr_is_pure(ctx, op_defs, &item.node, nodes, depth)),

        // ---------------- WP-34 lever 1 widening ----------------
        // Applied operator. Two admissible heads:
        //  * a module-scope definition of matching arity — unfold its body
        //    (depth + 1, so recursion self-limits to a rejection) and require
        //    the body AND every argument to be pure. The body's references to
        //    its own parameters resolve to no `op_defs` entry, so they are
        //    treated as the pure locals they are;
        //  * a name with NO module definition, restricted to the pure-builtin
        //    whitelist below. Every other undefined head (`TLCGet`,
        //    `RandomElement`, a CONSTANT operator supplied by the config, an
        //    unknown builtin) is rejected: it could be stateful, nondeterministic
        //    or plain unresolvable.
        E::Apply(op_expr, args) => {
            let E::Ident(op_name, _) = &op_expr.node else {
                return false;
            };
            if !args
                .iter()
                .all(|a| guard_expr_is_pure(ctx, op_defs, &a.node, nodes, depth))
            {
                return false;
            }
            let resolved = ctx.resolve_op_name(op_name.as_str());
            match op_defs.get(resolved) {
                Some(def) => {
                    if crate::eval::should_prefer_builtin_override(resolved, def, args.len(), ctx) {
                        crate::enumerate::is_replay_stable_named_builtin(resolved, args.len())
                    } else {
                        def.params.len() == args.len()
                            && guard_expr_is_pure(ctx, op_defs, &def.body.node, nodes, depth + 1)
                    }
                }
                None => guard_builtin_is_pure(resolved),
            }
        }
        E::Case(arms, other) => {
            arms.iter().all(|arm| {
                guard_expr_is_pure(ctx, op_defs, &arm.guard.node, nodes, depth)
                    && guard_expr_is_pure(ctx, op_defs, &arm.body.node, nodes, depth)
            }) && other
                .as_ref()
                .is_none_or(|d| guard_expr_is_pure(ctx, op_defs, &d.node, nodes, depth))
        }
        E::Let(defs, body) => {
            defs.iter()
                .all(|def| guard_expr_is_pure(ctx, op_defs, &def.body.node, nodes, depth))
                && guard_expr_is_pure(ctx, op_defs, &body.node, nodes, depth)
        }
        E::Except(base, specs) => {
            guard_expr_is_pure(ctx, op_defs, &base.node, nodes, depth)
                && specs.iter().all(|spec| {
                    guard_expr_is_pure(ctx, op_defs, &spec.value.node, nodes, depth)
                        && spec.path.iter().all(|elem| match elem {
                            tla_core::ast::ExceptPathElement::Index(idx) => {
                                guard_expr_is_pure(ctx, op_defs, &idx.node, nodes, depth)
                            }
                            tla_core::ast::ExceptPathElement::Field(_) => true,
                        })
                })
        }
        E::Record(fields) | E::RecordSet(fields) => fields
            .iter()
            .all(|(_, v)| guard_expr_is_pure(ctx, op_defs, &v.node, nodes, depth)),
        E::FuncSet(a, b) => {
            guard_expr_is_pure(ctx, op_defs, &a.node, nodes, depth)
                && guard_expr_is_pure(ctx, op_defs, &b.node, nodes, depth)
        }
        // Bounded binders only. An unbounded `\A x : P` / `\E x : P` has no
        // domain to range over and is rejected, exactly as the extractor's
        // `\E` descent rejects it.
        E::Forall(bounds, body) | E::Exists(bounds, body) | E::SetBuilder(body, bounds) => {
            guard_bounds_are_pure(ctx, op_defs, bounds, nodes, depth)
                && guard_expr_is_pure(ctx, op_defs, &body.node, nodes, depth)
        }
        E::FuncDef(bounds, body) => {
            guard_bounds_are_pure(ctx, op_defs, bounds, nodes, depth)
                && guard_expr_is_pure(ctx, op_defs, &body.node, nodes, depth)
        }
        E::SetFilter(bound_var, body) => {
            guard_bounds_are_pure(ctx, op_defs, std::slice::from_ref(bound_var), nodes, depth)
                && guard_expr_is_pure(ctx, op_defs, &body.node, nodes, depth)
        }
        _ => false,
    }
}

/// WP-34: every bound variable carries an explicit, pure domain.
fn guard_bounds_are_pure(
    ctx: &crate::eval::EvalCtx,
    op_defs: &rustc_hash::FxHashMap<String, tla_core::ast::OperatorDef>,
    bounds: &[tla_core::ast::BoundVar],
    nodes: &mut usize,
    depth: usize,
) -> bool {
    bounds.iter().all(|b| match &b.domain {
        None => false,
        Some(domain) => guard_expr_is_pure(ctx, op_defs, &domain.node, nodes, depth),
    })
}

/// WP-34: the standard-module operators the pre-check may evaluate when no
/// module definition shadows them. Every one is a pure, deterministic function
/// of its arguments. Deliberately excludes everything with an evaluation-order,
/// randomness or I/O effect (`TLCGet`, `TLCSet`, `RandomElement`, `Print`,
/// `Assert`, `CHOOSE`-flavoured helpers) and every higher-order operator
/// (`SelectSeq`, `FoldSet`, ...) whose operator argument the whitelist cannot
/// inspect.
fn guard_builtin_is_pure(name: &str) -> bool {
    matches!(
        name,
        "Cardinality" | "IsFiniteSet" | "Len" | "Head" | "Tail" | "Append"
    )
}

/// Accept `candidate` as this action's enabling guard, or reject it.
///
/// Rejects anything impure (above) and anything mentioning a name bound on the
/// path from the action root — a quantified variable or an operator parameter —
/// because such a guard's truth depends on a binding the pre-check does not
/// establish. With no bound name mentioned, `\E x \in S : body` is FALSE
/// whenever the guard is FALSE (for EVERY x, including the empty-domain case),
/// so the implication the pre-check needs holds unconditionally.
fn guard_candidate(
    ctx: &crate::eval::EvalCtx,
    op_defs: &rustc_hash::FxHashMap<String, tla_core::ast::OperatorDef>,
    candidate: &tla_core::Spanned<tla_core::ast::Expr>,
    bound: &[String],
    budget: &mut GuardBudget,
) -> Option<tla_core::Spanned<tla_core::ast::Expr>> {
    use tla_core::ast::Expr as E;

    if budget.terms == 0 {
        return None;
    }
    // `TRUE` is a valid but useless guard: it decides nothing and would be
    // evaluated on every instance. Drop it rather than pay for it.
    if matches!(candidate.node, E::Bool(true)) {
        return None;
    }
    let mut nodes = 0usize;
    if !guard_expr_is_pure(ctx, op_defs, &candidate.node, &mut nodes, 0) {
        return None;
    }
    if nodes > budget.nodes {
        return None;
    }
    if bound
        .iter()
        .any(|name| tla_core::expr_mentions_name_v(&candidate.node, name))
    {
        return None;
    }
    budget.nodes -= nodes;
    budget.terms -= 1;
    Some(candidate.clone())
}

/// WP-34 lever 1: the shared cost ceiling for ONE action's synthesized guard.
///
/// `nodes` / `terms` bound what the guard costs to EVALUATE (it runs on every
/// (parent, action) instance); `visits` bounds what the extraction itself
/// costs, since the conjunctive descent unfolds operators on both branches and
/// is exponential in the worst case. Exhausting any of them stops collecting —
/// whatever was already accepted stays a sound guard, because each conjunct is
/// independently implied by the action.
struct GuardBudget {
    nodes: usize,
    terms: usize,
    visits: usize,
}

impl GuardBudget {
    fn new() -> Self {
        Self {
            nodes: GUARD_TOTAL_MAX_NODES,
            terms: GUARD_MAX_TERMS,
            visits: GUARD_EXTRACT_MAX_VISITS,
        }
    }

    /// Charge one extractor step; `false` = budget exhausted, stop descending.
    fn step(&mut self) -> bool {
        if self.visits == 0 {
            return false;
        }
        self.visits -= 1;
        true
    }
}

/// Conjoin two extracted guards. `a /\ b` is implied by any action that
/// implies both, and `/\` short-circuits left-to-right so the leading (source
/// order) conjunct still decides first.
fn guard_conj(
    a: Option<tla_core::Spanned<tla_core::ast::Expr>>,
    b: Option<tla_core::Spanned<tla_core::ast::Expr>>,
) -> Option<tla_core::Spanned<tla_core::ast::Expr>> {
    match (a, b) {
        (Some(a), Some(b)) => {
            if a.node == b.node {
                return Some(a);
            }
            let span = a.span;
            Some(tla_core::Spanned::new(
                tla_core::ast::Expr::And(Box::new(a), Box::new(b)),
                span,
            ))
        }
        (Some(a), None) => Some(a),
        (None, b) => b,
    }
}

/// Disjoin two extracted guards. Sound only when BOTH branches yield one:
/// `A \/ B` with `A => ga` and `B => gb` implies `ga \/ gb`, but a branch with
/// no guard could hold while the other's guard is FALSE.
fn guard_disj(
    a: Option<tla_core::Spanned<tla_core::ast::Expr>>,
    b: Option<tla_core::Spanned<tla_core::ast::Expr>>,
) -> Option<tla_core::Spanned<tla_core::ast::Expr>> {
    let (a, b) = (a?, b?);
    if a.node == b.node {
        return Some(a);
    }
    let span = a.span;
    Some(tla_core::Spanned::new(
        tla_core::ast::Expr::Or(Box::new(a), Box::new(b)),
        span,
    ))
}

/// Extract a state-only guard `g` with `action => g` from a detected action's
/// expression, or `None` when no such guard is recognizable.
///
/// Descends through the shapes that preserve the implication: labels, `/\`
/// (WP-34: BOTH sides, conjoining every state-only conjunct found anywhere in
/// the top-level conjunction rather than only the leading one), `\/` (both
/// branches must yield, and the results are disjoined), `IF c THEN a ELSE b`,
/// `LET ... IN` (the LET-defined names join the forbidden set, so a guard is
/// accepted iff it mentions none of them — the same side condition
/// `try_let_guard_first_shortcircuit` uses, now applied per conjunct), `\E`
/// (recording the bound names), and module-scope operator references /
/// applications (recording the parameter names).
///
/// WP-34: every OTHER shape now falls through to [`guard_candidate`] on the
/// expression itself instead of declining outright — a pure state predicate
/// trivially implies itself, and an action-carrying shape (`Prime`,
/// `UNCHANGED`, `ENABLED`, a temporal operator) is still rejected there by the
/// purity whitelist. That is what lets a conjunct anywhere in the action body,
/// not just the syntactically first one, become the guard.
///
/// SOUNDNESS: each accepted conjunct `g_i` satisfies `action => g_i`
/// independently, so their conjunction is implied too, and a FALSE conjunction
/// still proves only "definitely disabled". No shape here can turn an
/// undecidable instance into a decided one: on any decline the caller runs the
/// unchanged full enumeration.
fn extract_action_state_guard(
    ctx: &crate::eval::EvalCtx,
    op_defs: &rustc_hash::FxHashMap<String, tla_core::ast::OperatorDef>,
    expr: &tla_core::Spanned<tla_core::ast::Expr>,
    bound: &mut Vec<String>,
    depth: usize,
) -> Option<tla_core::Spanned<tla_core::ast::Expr>> {
    let mut budget = GuardBudget::new();
    extract_guard_rec(ctx, op_defs, expr, bound, depth, &mut budget)
}

/// Extract only an evaluation-order-safe enabling prefix for the standalone
/// router.
///
/// The broad hybrid precheck may combine state-only conjuncts found anywhere
/// in an action. That preserves the successor relation, but a later false
/// conjunct could otherwise hide an evaluation error in an earlier conjunct.
/// AUTO has a stricter contract: it descends only through wrappers that occur
/// before the action body and then selects the syntactically first conjunct.
/// Quantifier domains are retained around the guard so their evaluation/error
/// order is preserved. LET definitions and operator arguments are call-by-name;
/// accepted guards may not mention their bound names, so no skipped expression
/// would have been forced before the prefix in the canonical action.
fn extract_router_prefix_state_guard(
    ctx: &crate::eval::EvalCtx,
    op_defs: &rustc_hash::FxHashMap<String, tla_core::ast::OperatorDef>,
    expr: &tla_core::Spanned<tla_core::ast::Expr>,
    bound: &mut Vec<String>,
    depth: usize,
) -> Option<tla_core::Spanned<tla_core::ast::Expr>> {
    let mut budget = GuardBudget::new();
    extract_router_prefix_guard_rec(ctx, op_defs, expr, bound, depth, &mut budget)
}

fn extract_router_prefix_guard_rec(
    ctx: &crate::eval::EvalCtx,
    op_defs: &rustc_hash::FxHashMap<String, tla_core::ast::OperatorDef>,
    expr: &tla_core::Spanned<tla_core::ast::Expr>,
    bound: &mut Vec<String>,
    depth: usize,
    budget: &mut GuardBudget,
) -> Option<tla_core::Spanned<tla_core::ast::Expr>> {
    use tla_core::ast::{BoundPattern, Expr as E};

    if depth > GUARD_EXTRACT_MAX_DEPTH || !budget.step() {
        return None;
    }
    match &expr.node {
        E::Label(label) => {
            extract_router_prefix_guard_rec(ctx, op_defs, &label.body, bound, depth, budget)
        }
        // TLA+ conjunction evaluation is left-to-right. Only the first
        // conjunct (recursing through a left-nested conjunction spine) may be
        // tested without suppressing an earlier error.
        E::And(left, _) => {
            extract_router_prefix_guard_rec(ctx, op_defs, left, bound, depth, budget)
        }
        E::Let(defs, body) => {
            let mark = bound.len();
            for def in defs {
                bound.push(def.name.node.clone());
            }
            let guard =
                extract_router_prefix_guard_rec(ctx, op_defs, body, bound, depth + 1, budget);
            bound.truncate(mark);
            guard
        }
        E::Exists(bounds, body) => {
            if bounds.iter().any(|bound_var| bound_var.domain.is_none()) {
                return None;
            }
            // The synthesized guard is moved outside any enclosing LET or
            // unfolded-operator scope. A retained domain that mentions one of
            // those outer bound names could resolve a different global symbol
            // during the precheck and no longer be implied by the action.
            if bounds.iter().any(|bound_var| {
                bound_var.domain.as_ref().is_some_and(|domain| {
                    bound
                        .iter()
                        .any(|name| tla_core::expr_mentions_name_v(&domain.node, name.as_str()))
                })
            }) {
                return None;
            }
            // The retained domains execute before the extracted body guard.
            // They must themselves be deterministic current-state expressions;
            // otherwise this standalone precheck could observe a stale/missing
            // next state or replay an effect before canonical enumeration.
            let mut domain_nodes = 0usize;
            if !guard_bounds_are_pure(ctx, op_defs, bounds, &mut domain_nodes, depth + 1) {
                return None;
            }
            let mark = bound.len();
            for bound_var in bounds {
                bound.push(bound_var.name.node.clone());
                match &bound_var.pattern {
                    None => {}
                    Some(BoundPattern::Var(name)) => bound.push(name.node.clone()),
                    Some(BoundPattern::Tuple(names)) => {
                        bound.extend(names.iter().map(|name| name.node.clone()));
                    }
                }
            }
            let guard =
                extract_router_prefix_guard_rec(ctx, op_defs, body, bound, depth + 1, budget);
            bound.truncate(mark);
            let guard = guard?;
            Some(tla_core::Spanned::new(
                E::Exists(bounds.clone(), Box::new(guard)),
                expr.span,
            ))
        }
        E::Ident(name, _) => {
            if bound.iter().any(|bound_name| bound_name == name)
                || ctx.has_local_binding(name)
                || ctx.name_in_local_scope(name)
                || ctx.is_config_constant(name)
                || ctx.resolve_op_name(name.as_str()) != name
                || ctx.instance_substitutions().is_some()
                || ctx.call_by_name_subs().is_some()
                || ctx.local_ops().is_some()
            {
                return None;
            }
            let resolved = ctx.resolve_op_name(name.as_str());
            match op_defs.get(resolved) {
                Some(def) if def.params.is_empty() => {
                    if crate::eval::should_prefer_builtin_override(resolved, def, 0, ctx) {
                        return None;
                    }
                    extract_router_prefix_guard_rec(
                        ctx,
                        op_defs,
                        &def.body,
                        bound,
                        depth + 1,
                        budget,
                    )
                }
                _ => guard_candidate(ctx, op_defs, expr, bound, budget),
            }
        }
        E::Apply(op_expr, args) => {
            let E::Ident(op_name, _) = &op_expr.node else {
                return guard_candidate(ctx, op_defs, expr, bound, budget);
            };
            if bound.iter().any(|bound_name| bound_name == op_name)
                || ctx.has_local_binding(op_name)
                || ctx.name_in_local_scope(op_name)
                || ctx.is_config_constant(op_name)
                || ctx.resolve_op_name(op_name.as_str()) != op_name
                || ctx.instance_substitutions().is_some()
                || ctx.call_by_name_subs().is_some()
                || ctx.local_ops().is_some()
            {
                return None;
            }
            let resolved = ctx.resolve_op_name(op_name.as_str());
            match op_defs.get(resolved) {
                Some(def) if def.params.len() == args.len() => {
                    if crate::eval::should_prefer_builtin_override(resolved, def, args.len(), ctx) {
                        return None;
                    }
                    let mark = bound.len();
                    bound.extend(def.params.iter().map(|param| param.name.node.clone()));
                    let guard = extract_router_prefix_guard_rec(
                        ctx,
                        op_defs,
                        &def.body,
                        bound,
                        depth + 1,
                        budget,
                    );
                    bound.truncate(mark);
                    guard
                }
                _ => guard_candidate(ctx, op_defs, expr, bound, budget),
            }
        }
        // The first expression is itself the prefix only when it is a pure
        // state predicate. Prime/action/temporal shapes fail closed here.
        _ => guard_candidate(ctx, op_defs, expr, bound, budget),
    }
}

fn extract_guard_rec(
    ctx: &crate::eval::EvalCtx,
    op_defs: &rustc_hash::FxHashMap<String, tla_core::ast::OperatorDef>,
    expr: &tla_core::Spanned<tla_core::ast::Expr>,
    bound: &mut Vec<String>,
    depth: usize,
    budget: &mut GuardBudget,
) -> Option<tla_core::Spanned<tla_core::ast::Expr>> {
    use tla_core::ast::{BoundPattern, Expr as E};

    if depth > GUARD_EXTRACT_MAX_DEPTH || !budget.step() {
        return None;
    }
    let wide = guard_wide_extract_enabled();

    match &expr.node {
        // `depth` charges SCOPE-introducing and operator-UNFOLDING descents
        // only (`LET`, `\E`, `Ident`, `Apply`) — those are what can recurse
        // without bound. The propositional spine (`/\`, `\/`, `IF`, labels) is
        // a finite walk of one AST, bounded by `budget.visits`, and must NOT
        // consume depth: a bulleted `/\` list of n conjuncts nests n deep, so
        // charging it would reject exactly the long actions (btree's
        // `SplitLeaf`, `SplitRootLeaf`, `SplitRootInner`, `UpdateReq`) whose
        // leading conjunct WP-29 already extracted.
        E::Label(label) => {
            let d = if wide { depth } else { depth + 1 };
            extract_guard_rec(ctx, op_defs, &label.body, bound, d, budget)
        }
        // Conjunction. `/\` trees are produced in both associativities, and
        // WP-34 descends into BOTH children, so every state-only conjunct of
        // the flattened top-level conjunction is collected regardless of shape.
        E::And(left, right) => {
            if !wide {
                // WP-29 behaviour verbatim (`TY_HYBRID_GUARD_WIDE=0`): leading
                // conjunct only, AND charging `depth` for every `/\` level —
                // which is what actually capped WP-29's coverage at 4 of
                // btree's 11 actions, since a bulleted list of n conjuncts
                // nests n deep and `GUARD_EXTRACT_MAX_DEPTH` is 8.
                if matches!(left.node, E::And(_, _) | E::Label(_)) {
                    return extract_guard_rec(ctx, op_defs, left, bound, depth + 1, budget);
                }
                return guard_candidate(ctx, op_defs, left, bound, budget);
            }
            let l = extract_guard_rec(ctx, op_defs, left, bound, depth, budget);
            let r = extract_guard_rec(ctx, op_defs, right, bound, depth, budget);
            guard_conj(l, r)
        }
        E::Or(left, right) if wide => {
            let l = extract_guard_rec(ctx, op_defs, left, bound, depth, budget);
            let r = extract_guard_rec(ctx, op_defs, right, bound, depth, budget);
            guard_disj(l, r)
        }
        // `IF c THEN a ELSE b`: taking either branch implies that branch's
        // guard, so the same `IF` over the two guards is implied. `c` itself
        // must be an admissible state predicate — it is re-evaluated here,
        // outside the action.
        E::If(cond, then_branch, else_branch) if wide => {
            let t = extract_guard_rec(ctx, op_defs, then_branch, bound, depth, budget)?;
            let e = extract_guard_rec(ctx, op_defs, else_branch, bound, depth, budget)?;
            let c = guard_candidate(ctx, op_defs, cond, bound, budget)?;
            let span = c.span;
            Some(tla_core::Spanned::new(
                E::If(Box::new(c), Box::new(t), Box::new(e)),
                span,
            ))
        }
        E::Let(defs, body) => {
            // The guard is evaluated OUTSIDE the LET scope, so no accepted
            // conjunct may reference a LET-defined name. Pushing the names onto
            // `bound` applies that condition per conjunct instead of rejecting
            // the whole guard when any one of them mentions a LET name.
            let mark = bound.len();
            for def in defs {
                bound.push(def.name.node.clone());
            }
            let guard = extract_guard_rec(ctx, op_defs, body, bound, depth + 1, budget);
            bound.truncate(mark);
            guard
        }
        E::Exists(bounds, body) => {
            let mark = bound.len();
            for bound_var in bounds {
                if bound_var.domain.is_none() {
                    bound.truncate(mark);
                    return None;
                }
                bound.push(bound_var.name.node.clone());
                match &bound_var.pattern {
                    None => {}
                    Some(BoundPattern::Var(name)) => bound.push(name.node.clone()),
                    Some(BoundPattern::Tuple(names)) => {
                        for name in names {
                            bound.push(name.node.clone());
                        }
                    }
                }
            }
            let guard = extract_guard_rec(ctx, op_defs, body, bound, depth + 1, budget);
            bound.truncate(mark);
            guard
        }
        E::Ident(name, _) => {
            let resolved = ctx.resolve_op_name(name.as_str());
            let unfolded = match op_defs.get(resolved) {
                Some(def) if def.params.is_empty() => {
                    extract_guard_rec(ctx, op_defs, &def.body, bound, depth + 1, budget)
                }
                _ => None,
            };
            // A bare name that is itself a pure state predicate (a zero-arg
            // state operator, a CONSTANT) is its own guard.
            unfolded.or_else(|| guard_candidate(ctx, op_defs, expr, bound, budget))
        }
        E::Apply(op_expr, args) => {
            let unfolded = match &op_expr.node {
                E::Ident(op_name, _) => {
                    let resolved = ctx.resolve_op_name(op_name.as_str());
                    match op_defs.get(resolved) {
                        Some(def) if def.params.len() == args.len() => {
                            let mark = bound.len();
                            for param in &def.params {
                                bound.push(param.name.node.clone());
                            }
                            let guard = extract_guard_rec(
                                ctx,
                                op_defs,
                                &def.body,
                                bound,
                                depth + 1,
                                budget,
                            );
                            bound.truncate(mark);
                            guard
                        }
                        _ => None,
                    }
                }
                _ => None,
            };
            unfolded.or_else(|| guard_candidate(ctx, op_defs, expr, bound, budget))
        }
        // WP-34: any other shape is a guard exactly when it is itself an
        // admissible pure state predicate. `Prime`, `UNCHANGED`, `ENABLED` and
        // the temporal operators are rejected by the purity whitelist.
        _ => guard_candidate(ctx, op_defs, expr, bound, budget),
    }
}

/// Return `true` only when evaluating an extracted state-only guard proves the
/// action disabled in `parent`. Every other outcome, including evaluation
/// errors and non-boolean values, is deliberately undecided so canonical
/// action enumeration remains authoritative.
fn guard_proves_disabled(
    ctx: &mut crate::eval::EvalCtx,
    guard: &tla_core::Spanned<tla_core::ast::Expr>,
    parent: &ArrayState,
) -> bool {
    let verdict = {
        let _state_guard = ctx.bind_state_env_guard(parent.env_ref());
        crate::eval::eval(ctx, guard)
    };
    matches!(verdict, Ok(crate::Value::Bool(false)))
}

/// WP-34 diagnostics: one-line rendering of an accepted guard
/// (`TY_HYBRID_GUARD_DEBUG=1` only, never on a hot path).
fn guard_debug_render(expr: &tla_core::ast::Expr, depth: usize) -> String {
    use tla_core::ast::Expr as E;
    if depth > 4 {
        return "...".to_string();
    }
    let bin = |op: &str, a: &tla_core::Spanned<E>, b: &tla_core::Spanned<E>| {
        format!(
            "({} {op} {})",
            guard_debug_render(&a.node, depth + 1),
            guard_debug_render(&b.node, depth + 1)
        )
    };
    match expr {
        E::Bool(b) => b.to_string(),
        E::Int(i) => i.to_string(),
        E::String(s) => format!("\"{s}\""),
        E::Ident(n, _) => n.clone(),
        E::StateVar(n, _, _) => n.clone(),
        E::And(a, b) => bin("/\\", a, b),
        E::Or(a, b) => bin("\\/", a, b),
        E::Eq(a, b) => bin("=", a, b),
        E::Neq(a, b) => bin("#", a, b),
        E::In(a, b) => bin("\\in", a, b),
        E::NotIn(a, b) => bin("\\notin", a, b),
        E::Lt(a, b) => bin("<", a, b),
        E::Leq(a, b) => bin("<=", a, b),
        E::Gt(a, b) => bin(">", a, b),
        E::Geq(a, b) => bin(">=", a, b),
        E::FuncApply(a, b) => format!(
            "{}[{}]",
            guard_debug_render(&a.node, depth + 1),
            guard_debug_render(&b.node, depth + 1)
        ),
        E::Not(a) => format!("~{}", guard_debug_render(&a.node, depth + 1)),
        E::Apply(op, args) => format!(
            "{}({} args)",
            guard_debug_render(&op.node, depth + 1),
            args.len()
        ),
        other => format!("<{}>", guard_expr_variant(other)),
    }
}

/// WP-34 diagnostics: the AST variant name of an expression, so a declined
/// action reports WHICH shape stopped the extractor.
fn guard_expr_variant(expr: &tla_core::ast::Expr) -> &'static str {
    use tla_core::ast::Expr as E;
    match expr {
        E::Bool(_) => "Bool",
        E::Int(_) => "Int",
        E::String(_) => "String",
        E::Ident(_, _) => "Ident",
        E::StateVar(_, _, _) => "StateVar",
        E::Apply(_, _) => "Apply",
        E::OpRef(_) => "OpRef",
        E::ModuleRef(_, _, _) => "ModuleRef",
        E::InstanceExpr(_, _) => "InstanceExpr",
        E::Lambda(_, _) => "Lambda",
        E::Label(_) => "Label",
        E::And(_, _) => "And",
        E::Or(_, _) => "Or",
        E::Not(_) => "Not",
        E::Implies(_, _) => "Implies",
        E::Equiv(_, _) => "Equiv",
        E::Forall(_, _) => "Forall",
        E::Exists(_, _) => "Exists",
        E::Choose(_, _) => "Choose",
        E::SetEnum(_) => "SetEnum",
        E::SetBuilder(_, _) => "SetBuilder",
        E::SetFilter(_, _) => "SetFilter",
        E::In(_, _) => "In",
        E::NotIn(_, _) => "NotIn",
        E::Subseteq(_, _) => "Subseteq",
        E::Union(_, _) => "Union",
        E::Intersect(_, _) => "Intersect",
        E::SetMinus(_, _) => "SetMinus",
        E::Powerset(_) => "Powerset",
        E::BigUnion(_) => "BigUnion",
        E::FuncDef(_, _) => "FuncDef",
        E::FuncApply(_, _) => "FuncApply",
        E::Domain(_) => "Domain",
        E::Except(_, _) => "Except",
        E::FuncSet(_, _) => "FuncSet",
        E::Record(_) => "Record",
        E::RecordAccess(_, _) => "RecordAccess",
        E::RecordSet(_) => "RecordSet",
        E::Tuple(_) => "Tuple",
        E::Times(_) => "Times",
        E::Prime(_) => "Prime",
        E::Always(_) => "Always",
        E::Eventually(_) => "Eventually",
        E::LeadsTo(_, _) => "LeadsTo",
        E::WeakFair(_, _) => "WeakFair",
        E::StrongFair(_, _) => "StrongFair",
        E::Enabled(_) => "Enabled",
        E::Unchanged(_) => "Unchanged",
        E::If(_, _, _) => "If",
        E::Case(_, _) => "Case",
        E::Let(_, _) => "Let",
        E::SubstIn(_, _) => "SubstIn",
        E::Eq(_, _) => "Eq",
        E::Neq(_, _) => "Neq",
        E::Lt(_, _) => "Lt",
        E::Leq(_, _) => "Leq",
        E::Gt(_, _) => "Gt",
        E::Geq(_, _) => "Geq",
        E::Add(_, _) => "Add",
        E::Sub(_, _) => "Sub",
        E::Mul(_, _) => "Mul",
        E::Div(_, _) => "Div",
        E::IntDiv(_, _) => "IntDiv",
        E::Mod(_, _) => "Mod",
        E::Pow(_, _) => "Pow",
        E::Neg(_) => "Neg",
        E::Range(_, _) => "Range",
    }
}

// ===================== WP-29 lever 2: delta reconstruction ==================
//
// `HybridFlatView::reconstruct` decodes EVERY flat-admissible variable out of a
// native successor buffer, rebuilding btree's 32-entry `childOf` / `valOf`
// function values from scratch on all ~1.46M authoritative successors — even
// though a hybrid-eligible action writes only a handful of variables and the
// rest are byte-identical to the parent's projected buffer.
//
// The delta path compares each admissible variable's slot range against the
// parent's projection and decodes ONLY the ones that differ; the rest are
// `Arc`-shared straight off the parent, exactly as the compound (non-admissible)
// variables already are.
//
// SOUNDNESS: `parent_view` is `project(parent)`, so its slots for variable `v`
// are `encode(parent[v])`. Flat-admissibility is exactly the contract that
// `decode(encode(x)) == x` (`supports_flat_primary`, the same contract the
// whole-state flat path rests on and which this module's G2 tests assert). So
// equal slots imply an equal decoded value, and taking the parent's is the same
// value the whole-buffer decode would have produced — bit-for-bit, including
// its fingerprint. Any shape the delta path cannot handle returns `None` and
// falls back to the whole-buffer decode.

/// Reconstruct one native successor by decoding only the changed
/// flat-admissible variables. `None` = decline (caller falls back).
#[allow(clippy::too_many_arguments)]
fn hybrid_reconstruct_delta(
    view: &HybridFlatView,
    var_layouts: &[Option<Arc<crate::state::StateLayout>>],
    parent: &ArrayState,
    parent_view: &FlatState,
    candidate: &FlatState,
    registry: &VarRegistry,
    decoded_vars: &mut u64,
    total_vars: &mut u64,
) -> Option<ArrayState> {
    let var_count = view.var_count();
    if var_layouts.len() != var_count
        || parent_view.num_slots() != candidate.num_slots()
        || registry.len() != var_count
    {
        return None;
    }

    let mut values: Vec<crate::Value> = Vec::with_capacity(var_count);
    for var_idx in 0..var_count {
        let idx = VarIndex::new(var_idx);
        if !view.is_var_flat_admissible(var_idx) {
            // Compound variable: a hybrid-eligible action's writes are strictly
            // flat-admissible (enforced by the eligibility + dual admission
            // gates), so the parent's payload IS the successor's.
            values.push(parent.get(idx));
            continue;
        }
        *total_vars += 1;
        let succ_slots = candidate.get_var_slots(var_idx)?;
        let base_slots = parent_view.get_var_slots(var_idx)?;
        if succ_slots == base_slots {
            values.push(parent.get(idx));
            continue;
        }
        let var_layout = var_layouts.get(var_idx)?.as_ref()?;
        *decoded_vars += 1;
        let single = FlatState::try_from_buffer(
            succ_slots.to_vec().into_boxed_slice(),
            Arc::clone(var_layout),
        )
        .ok()?;
        let decoded = single.try_to_array_state(registry).ok()?;
        if decoded.len() != 1 {
            return None;
        }
        values.push(decoded.get(VarIndex::new(0)));
    }
    Some(ArrayState::from_values(values))
}

/// Per-action AST footprint snapshot (from ty's own static analysis,
/// `extract_detected_action_dependencies`) kept for the native admission dual
/// gate: the compiled bytecode footprint must be contained in it, or the
/// action instance declines native dispatch (item 4 M0-G4).
#[derive(Debug, Clone, Default)]
pub(super) struct HybridAstFootprint {
    reads: FxHashSet<usize>,
    writes: FxHashSet<usize>,
    opaque: bool,
}

/// Model-checker-side state for hybrid per-action dispatch. Lazily initialized
/// on the first per-action successor generation; inert until then.
#[derive(Debug, Clone, Default)]
pub(super) struct HybridDispatchState {
    /// Master switch (`TY_HYBRID_FLAT_VIEW`), read once at first init.
    enabled: bool,
    /// Native-dispatch switch (`TY_HYBRID_NATIVE`, only meaningful when
    /// `enabled`), read once at first init.
    native_enabled: bool,
    /// Whether lazy init has run (classification + view build).
    initialized: bool,
    /// The flat-view projection over the run's layout. `None` when disabled or
    /// when no variable is flat-admissible.
    view: Option<HybridFlatView>,
    /// Per-detected-action hybrid eligibility, aligned 1:1 with
    /// `coverage.actions`. Empty when disabled / not yet initialized.
    eligible: Vec<bool>,
    /// Per-detected-action AST footprints (same alignment as `eligible`),
    /// kept for the native admission dual gate.
    ast_footprints: Vec<HybridAstFootprint>,
    /// Runtime routing counters.
    stats: HybridDispatchStats,
    /// WP-14: authoritative-dispatch switch (`TY_HYBRID_NATIVE_AUTHORITATIVE`,
    /// only meaningful when `native_enabled`), read once at first init.
    authoritative_enabled: bool,
    /// WP-14: per-action burn-in / sampling / fail-back state machine.
    machine: HybridAuthoritativeMachine,
    /// WP-14 test-only fault injection latch: at most one sampled buffer is
    /// ever corrupted per run.
    corruption_injected: bool,
    /// WP-17: per-action memoized native admission. Key resolution
    /// (`trust_cg_hybrid_action_dispatch_keys`) and the footprint dual gate
    /// (`hybrid_native_footprint_admitted`) are pure functions of run-fixed
    /// state — the hybrid cache (assigned once, before the BFS), the
    /// split-action meta, the AST footprints, and the view — so their outcome
    /// is resolved ONCE per action instead of once per (parent, action).
    /// `None` = unresolved; `Some(None)` = permanent decline (counted in
    /// `native_declined` exactly once, at resolution); `Some(Some(keys))` =
    /// the admitted dispatch key set.
    admitted_keys: Vec<Option<Option<Arc<Vec<String>>>>>,
    /// WP-17: coarse per-bucket wall timers (`TY_HYBRID_PERF_DEBUG=1`).
    pub(super) perf: HybridPerfTimers,
    /// WP-26: use the diff-native interpreter branch in the per-action loop
    /// (default ON; `TY_HYBRID_INTERP_DIFF=0` restores the legacy round trip).
    pub(super) interp_diff_path: bool,
    /// WP-29 lever 1: per-detected-action memoized enabling guard. The
    /// extraction is a pure function of run-fixed state (the action AST, the
    /// module operator table, the config operator-replacement map), so it is
    /// resolved ONCE per action. `None` = unresolved; `Some(None)` = no usable
    /// guard for this action (every instance enumerates, as before);
    /// `Some(Some(g))` = a state-only conjunct `g` with `action => g`, held for
    /// the whole run so its node addresses stay stable for the evaluator's
    /// pointer-keyed caches.
    action_state_guards: Vec<Option<Option<Arc<tla_core::Spanned<tla_core::ast::Expr>>>>>,
    /// Standalone-router variant of `action_state_guards`: only the first
    /// evaluation-order-safe state predicate is retained, with any enclosing
    /// quantifier domains preserved. This prevents a later false conjunct from
    /// suppressing an earlier canonical evaluation error.
    router_prefix_state_guards: Vec<Option<Option<Arc<tla_core::Spanned<tla_core::ast::Expr>>>>>,
    /// WP-29 lever 2: per-variable single-variable layouts over the hybrid
    /// layout's flat-admissible variables, used to decode ONE changed variable
    /// out of a native successor buffer without rebuilding the whole state.
    /// Built once, on first delta reconstruction. `None` = not admissible (the
    /// variable is always taken from the parent).
    delta_var_layouts: Vec<Option<Arc<crate::state::StateLayout>>>,
    /// Whether `delta_var_layouts` has been built (it can legitimately end up
    /// all-`None`, which must not retrigger the build).
    delta_var_layouts_built: bool,
    /// WP-34 lever 1 diagnostics: whether the one-shot per-action guard dump
    /// (`TY_HYBRID_GUARD_DEBUG=1`) has already run.
    pub(super) guards_dumped: bool,
    /// Whether the standalone-router arming pass has run. This is independent
    /// of hybrid initialization because full-state consumers may consult route
    /// selection before touching the hybrid dispatcher.
    router_armed: bool,
    /// Conservative AUTO lifecycle, or an explicit forced route.
    router_phase: RouterPhase,
    /// Stable detected-action decomposition retained after AUTO's delayed
    /// static admission, or immediately for a forced route.
    router_actions: Option<Arc<Vec<DetectedAction>>>,
    /// Whether the router is the sole reason action boundaries matter. Other
    /// engines use this fence to keep the router's blast radius on the
    /// interpreter array/diff route even when setup had already retained a
    /// dormant detected-action vector for bytecode.
    router_sole_route_owner: bool,
    /// Whether activation copied router actions into `coverage.actions`.
    router_installed_actions: bool,
    /// Expanded-parent delay and exact-trial guard-decision evidence.
    router_pilot_parents: u64,
    router_trial_checks: u64,
    router_trial_skips: u64,
    router_trial_checks_start: u64,
    router_trial_skips_start: u64,
    /// Exact-parity/timing trial evidence.
    router_parity_checked: u32,
    router_batch_ns: u128,
    router_whole_next_ns: u128,
    /// Routed parents after trial, used for sparse parity sampling.
    router_active_parents: u64,
    /// Whether the first parent presented to the admitted router passed the
    /// recursive, read-only TLC token-closure check. Static action/source
    /// admission then preserves that property inductively for successors.
    router_parent_tokens_checked: bool,
    /// Stable diagnostic reason for a conservative decline/failback.
    router_decision_reason: Option<String>,
}

/// Native successor candidates for one (parent, hybrid-eligible action)
/// execution: the updated flat views the compiled hybrid artifacts produced.
///
/// The per-successor differential consumes candidates by byte-exact buffer
/// match against the projected interpreter successor
/// ([`ModelChecker::hybrid_route_successor`]); anything left unconsumed at
/// action end is a native/interpreter divergence
/// ([`ModelChecker::hybrid_finish_native_action`]).
pub(in crate::check) struct HybridNativeCandidates {
    views: Vec<FlatState>,
    consumed: Vec<bool>,
    /// Coverage action index this execution belongs to (burn-in slot key).
    action_idx: usize,
    /// WP-14: how this instance participates (shadow / sampled / authoritative).
    mode: HybridInstanceMode,
    /// WP-14: a SEMANTIC native/interpreter divergence was observed on this
    /// instance (unmatched interpreter successor, value inequality after a
    /// byte match; residue is derived separately at action end).
    dirty: bool,
    /// WP-14: the differential could not complete on this instance (a
    /// projection/reconstruct decline) — resets burn-in without implying a
    /// semantic divergence.
    unproven: bool,
}

impl HybridNativeCandidates {
    fn new(views: Vec<FlatState>, action_idx: usize, mode: HybridInstanceMode) -> Self {
        let consumed = vec![false; views.len()];
        Self {
            views,
            consumed,
            action_idx,
            mode,
            dirty: false,
            unproven: false,
        }
    }

    /// WP-14: whether this instance dispatches authoritatively (native
    /// successors enqueued, interpreter enumeration skipped).
    #[inline]
    pub(in crate::check) fn is_authoritative(&self) -> bool {
        self.mode == HybridInstanceMode::Authoritative
    }

    fn mark_dirty(&mut self) {
        self.dirty = true;
    }

    fn mark_unproven(&mut self) {
        self.unproven = true;
    }

    fn mark_all_consumed(&mut self) {
        for c in &mut self.consumed {
            *c = true;
        }
    }

    /// Consume and return the first unconsumed candidate whose hybrid buffer
    /// equals `target`'s byte-for-byte (multiset semantics: duplicate
    /// successors each consume one candidate).
    fn take_matching(&mut self, target: &FlatState) -> Option<FlatState> {
        for (idx, view) in self.views.iter().enumerate() {
            if !self.consumed[idx] && view.buffer() == target.buffer() {
                self.consumed[idx] = true;
                return Some(view.clone());
            }
        }
        None
    }

    /// Native successors the interpreter never matched.
    fn unconsumed(&self) -> u64 {
        self.consumed.iter().filter(|c| !**c).count() as u64
    }
}

impl HybridDispatchState {
    /// `true` once initialized AND the switch is on AND a projection exists.
    #[inline]
    fn active(&self) -> bool {
        self.enabled && self.initialized && self.view.is_some()
    }
}

/// Structural value equality between two states over their first `var_count`
/// variables.
///
/// This is the differential gate: it is robust to `Arc` identity (a compound
/// var reconstructed from the parent shares the parent's `Arc`, while the
/// interpreter successor may hold a distinct but equal `Arc`) because it
/// compares decoded `Value`s, not `CompactValue` pointers. Ruling out even a
/// (astronomically unlikely) fingerprint collision, it guarantees the routed
/// successor is semantically identical to the interpreter successor.
fn states_value_equal(a: &ArrayState, b: &ArrayState, var_count: usize) -> bool {
    (0..var_count).all(|i| {
        let idx = VarIndex::new(i);
        a.get(idx) == b.get(idx)
    })
}

/// Validating-shadow projection for hybrid-eligible successors WITHOUT a
/// native execution (TY_HYBRID_NATIVE off, or the action instance declined
/// native admission).
///
/// The REAL trust-cg native flat-view dispatch landed with item 4 M0
/// (TY_HYBRID_NATIVE=1): compiled hybrid-layout artifacts execute per
/// (parent, action) in [`ModelChecker::hybrid_native_candidates_for_action`]
/// — project parent → run the compiled action against the hybrid buffer →
/// collect updated flat views — and [`ModelChecker::hybrid_route_successor`]
/// consumes those candidates by byte-exact buffer match, keeping the
/// per-successor value-equality differential against the interpreter successor
/// fully intact (M0 stays a validated shadow/burn-in: the interpreter is
/// authoritative on any divergence).
///
/// This function remains the interpreter-through-projection SHADOW body: the
/// updated flat view is obtained by projecting the interpreter-computed
/// successor, exercising the exact projection + reconstruction + routing
/// machinery the native path uses.
#[inline]
fn hybrid_shadow_flat_view_dispatch(
    view: &HybridFlatView,
    _parent_flat_view: &FlatState,
    interp_succ: &ArrayState,
) -> Option<FlatState> {
    view.project(interp_succ)
}

impl<'a> ModelChecker<'a> {
    /// Whether an observer or specialized successor engine already owns
    /// routing. AUTO never steals from one of these routes. This includes
    /// implicit dead-action coverage: it already selects the same per-action
    /// batch route, so router activation could add parity work but no speedup.
    fn router_has_existing_route_owner(&self) -> bool {
        let explicit_tir = self
            .tir_parity
            .as_ref()
            .is_some_and(|parity| !parity.is_implicit_default_eval_mode());
        self.coverage.collect
            || self.coverage.display
            || self.coverage.coverage_guided
            || self.hybrid_dispatch.enabled
            || (!self.hybrid_dispatch.initialized
                && std::env::var_os("TY_HYBRID_FLAT_VIEW").is_some_and(|value| value == "1"))
            || std::env::var_os("TY_HYBRID_INTERP_DIFF").is_some_and(|value| value == "0")
            || self.por.independence.is_some()
            || self.jit_hybrid_ready()
            || self.jit_monolithic_ready()
            || self.compiled.pc_dispatch.is_some()
            || self.compiled.pc_var_idx.is_some()
            || self.trust_cg_action_dispatch_ready()
            || self.trust_cg_hybrid_action_dispatch_ready()
            || self.value_action_vm.is_armed()
            || self.value_action_vm.auto_selected()
            || self.flat_state_primary
            || self.uses_compiled_bfs_fingerprint_domain()
            || self.nested_set_slide_arm.is_some()
            || self.liveness_cache.cache_for_liveness
            || self.inline_liveness_active()
            || self.should_run_on_the_fly_liveness()
            || !self.compiled.eval_implied_actions.is_empty()
            || !self.config.constraints.is_empty()
            || !self.config.action_constraints.is_empty()
            || !self.config.trace_invariants.is_empty()
            || self.config.terminal.is_some()
            || self.config.postcondition.is_some()
            || self.config.alias.is_some()
            || self.compiled.cached_view_name.is_some()
            || !self.symmetry.perms.is_empty()
            || explicit_tir
    }

    fn decline_router(&mut self, phase: RouterPhase, reason: impl Into<String>) {
        if self.hybrid_dispatch.router_installed_actions {
            if self.router_has_existing_route_owner() {
                // A later route owner (for example lazy native compilation)
                // needs the same stable action Arc. Transfer ownership rather
                // than clearing it underneath that engine.
            } else {
                let old = std::mem::replace(&mut self.coverage.actions, Arc::new(Vec::new()));
                self.coverage.retired_actions.push(old);
            }
            self.hybrid_dispatch.router_installed_actions = false;
        }
        self.hybrid_dispatch.router_phase = phase;
        self.hybrid_dispatch.router_decision_reason = Some(reason.into());
    }

    fn resolve_router_actions(&self) -> Result<(Arc<Vec<DetectedAction>>, bool), &'static str> {
        if !self.coverage.actions.is_empty() {
            return Ok((Arc::clone(&self.coverage.actions), true));
        }
        let next_name = self
            .trace
            .cached_resolved_next_name
            .clone()
            .or_else(|| self.trace.cached_next_name.clone());
        let Some(next_def) = next_name.and_then(|name| self.module.op_defs.get(&name)) else {
            return Err("Next could not be resolved");
        };
        let actions = detect_actions(next_def);
        if actions.is_empty() {
            return Err("Next has no detected actions");
        }
        Ok((Arc::new(actions), false))
    }

    fn router_expr_replay_safe(
        &self,
        kind: &str,
        name: &str,
        expr: &tla_core::Spanned<tla_core::ast::Expr>,
    ) -> bool {
        let (raw_slots, expanded_slots, raw_context, expanded_context) =
            crate::checker_ops::action_expr_replay_safety_components(&self.ctx, expr);
        let safe = raw_slots && expanded_slots && raw_context && expanded_context;
        if !safe
            && std::env::var("TY_ROUTER_DIAG")
                .is_ok_and(|value| tla_backend::env_flag_enabled(&value))
        {
            let (raw_reason, expanded_reason) =
                crate::checker_ops::action_expr_replay_context_rejections(&self.ctx, expr);
            eprintln!(
                "[router] replay admission rejected kind={kind} name={name} raw_slots={} \
                 expanded_slots={} raw_context={} expanded_context={} \
                 raw_reason={:?} expanded_reason={:?}",
                raw_slots,
                expanded_slots,
                raw_context,
                expanded_context,
                raw_reason,
                expanded_reason,
            );
        }
        safe
    }

    fn router_actions_replay_safe(&self, actions: &[DetectedAction]) -> bool {
        actions
            .iter()
            .all(|action| self.router_expr_replay_safe("action", &action.name, &action.expr))
    }

    /// Prove that state sources and every BFS state observer remain free of
    /// evaluation-order-visible effects. Per-action routing preserves the
    /// successor multiset, but it can change which successor is observed
    /// first; an invariant that assigns a first-seen TLC token (or reads live
    /// checker context) could otherwise change later CHOOSE/set ordering.
    fn router_observers_replay_safe(&self) -> bool {
        // Initial states have already been generated when routing begins. Walk
        // their source anyway: it certifies that record fields or other values
        // reachable through a state variable cannot hide an unassigned TLC
        // ordering token from the action/invariant expression walks below.
        if let Some(init_name) = self
            .trace
            .cached_init_name
            .as_deref()
            .or(self.config.init.as_deref())
        {
            let resolved = self.ctx.resolve_op_name(init_name);
            let Some(def) = self.ctx.get_op(resolved) else {
                return false;
            };
            if !def.params.is_empty()
                || !self.router_expr_replay_safe("initial-state source", resolved, &def.body)
            {
                return false;
            }
        }

        // Name-based invariants execute through eval_op(raw_name), which does
        // not resolve the root through config operator replacement. Inspect
        // that exact body; the shared walker resolves calls inside it.
        for name in &self.config.invariants {
            let Some(def) = self.ctx.get_op(name) else {
                return false;
            };
            if !def.params.is_empty() || !self.router_expr_replay_safe("invariant", name, &def.body)
            {
                return false;
            }
        }
        for (name, expr) in &self.compiled.eval_state_invariants {
            if !self.router_expr_replay_safe("property state invariant", name, expr) {
                return false;
            }
        }
        true
    }

    /// Arm the standalone interpreter action router once per run.
    ///
    /// Forced mode publishes the detected-action decomposition immediately.
    /// AUTO delays even detection/guard extraction until a run has expanded
    /// enough parents to amortize that one-time work, so short runs retain the
    /// canonical engine's allocation and startup profile.
    pub(in crate::check) fn ensure_router_ready(&mut self) {
        if self.hybrid_dispatch.router_armed {
            return;
        }
        self.hybrid_dispatch.router_armed = true;
        let request = router_request();
        if matches!(request, RouterRequest::Disabled) {
            return;
        }

        if matches!(request, RouterRequest::Forced) {
            if self.router_has_existing_route_owner() {
                self.decline_router(
                    RouterPhase::Declined,
                    "forced routing cannot co-own coverage/POR/native/flat/VM/liveness/constraint dispatch",
                );
                return;
            }
            let (actions, had_actions) = match self.resolve_router_actions() {
                Ok(resolved) => resolved,
                Err(reason) => {
                    self.decline_router(RouterPhase::Declined, reason);
                    return;
                }
            };
            if !self.router_actions_replay_safe(&actions) {
                self.decline_router(
                    RouterPhase::Declined,
                    "an action body is context-dependent, side-effecting, or not replay-safe",
                );
                return;
            }
            if !self.router_observers_replay_safe() {
                self.decline_router(
                    RouterPhase::Declined,
                    "an initial-state source or state observer is context-dependent, side-effecting, or not replay-safe",
                );
                return;
            }
            if !had_actions {
                self.coverage.actions = Arc::clone(&actions);
                self.hybrid_dispatch.router_installed_actions = true;
            }
            self.hybrid_dispatch.router_actions = Some(actions);
            self.hybrid_dispatch.router_sole_route_owner = true;
            self.hybrid_dispatch.router_phase = RouterPhase::Forced;
            self.hybrid_dispatch.router_decision_reason = Some("forced by TY_ROUTER=1".into());
            return;
        }

        if self.router_has_existing_route_owner() {
            self.decline_router(
                RouterPhase::Declined,
                "another coverage/POR/native/flat/VM/liveness route already owns dispatch",
            );
            return;
        }
        if !action_guard_precheck_enabled() || !router_guard_precheck_enabled() {
            self.decline_router(
                RouterPhase::Declined,
                "an action-guard precheck kill switch disables the only expected benefit",
            );
            return;
        }
        self.hybrid_dispatch.router_sole_route_owner = true;
        self.hybrid_dispatch.router_phase = RouterPhase::Pilot;
        self.hybrid_dispatch.router_decision_reason =
            Some("16,384-parent allocation-free delay pending".into());
    }

    /// Whether the standalone router is selecting the interpreter batch path.
    #[inline]
    pub(in crate::check) fn router_active(&self) -> bool {
        matches!(
            self.hybrid_dispatch.router_phase,
            RouterPhase::Trial | RouterPhase::Active | RouterPhase::Forced
        )
    }

    /// Whether the standalone router is the sole action-boundary route owner.
    ///
    /// Flat/native engines use this fence so the standalone router changes only
    /// the interpreter array/diff route that its parity guard validates.
    #[inline]
    pub(in crate::check) fn router_only_detected_actions(&self) -> bool {
        self.router_active() && self.hybrid_dispatch.router_sole_route_owner
    }

    /// Check the actual frontier root once before the router evaluates it.
    ///
    /// Walking Init is the static source proof, but resumed/checkpointed and
    /// specialized bulk-init paths can hand the BFS a concrete payload through
    /// a different representation. This closes that boundary without mutating
    /// either token registry. Once it passes, the admitted action and observer
    /// expressions cannot manufacture an unassigned token, so every successor
    /// preserves the certificate and no per-parent rescan is needed.
    pub(in crate::check) fn router_parent_tokens_replay_safe(
        &mut self,
        state: &crate::state::State,
    ) -> bool {
        if !self.router_active() || self.hybrid_dispatch.router_parent_tokens_checked {
            return true;
        }
        if let Some((name, _)) = state.vars().find(|(_, value)| {
            !value.is_concrete_data() || !value.has_preassigned_tlc_order_tokens()
        }) {
            self.decline_router(
                RouterPhase::Failback,
                format!(
                    "frontier variable `{name}` contains executable data or an unassigned TLC ordering token"
                ),
            );
            return false;
        }
        self.hybrid_dispatch.router_parent_tokens_checked = true;
        true
    }

    /// Count one expanded parent toward AUTO's delayed trial.
    ///
    /// No action is detected, retained, evaluated, or enumerated during this
    /// wait. Runs ending below the threshold therefore stay on their existing
    /// successor engine with only a parent counter and phase check. Static
    /// admission, guard rate, parity, and timings are all deferred until the
    /// run is large enough to plausibly repay them.
    pub(in crate::check) fn maybe_advance_router_pilot(&mut self) {
        self.ensure_router_ready();
        if matches!(
            self.hybrid_dispatch.router_phase,
            RouterPhase::Trial | RouterPhase::Active
        ) && self.router_has_existing_route_owner()
        {
            self.decline_router(
                RouterPhase::Declined,
                "a higher-priority route became active after router admission",
            );
            return;
        }
        if !matches!(self.hybrid_dispatch.router_phase, RouterPhase::Pilot) {
            return;
        }
        if self.hybrid_dispatch.router_pilot_parents < ROUTER_PILOT_PARENTS {
            self.hybrid_dispatch.router_pilot_parents += 1;
            return;
        }

        // A lazy native/VM route can appear during the delay. There is no need
        // to poll the full ownership predicate on every warm-up parent because
        // the router has not changed dispatch yet; re-check once immediately
        // before publishing its private action decomposition.
        if self.router_has_existing_route_owner() {
            self.decline_router(
                RouterPhase::Declined,
                "an existing route became active during the delayed pilot",
            );
            return;
        }
        if !router_guard_precheck_enabled() {
            self.decline_router(
                RouterPhase::Declined,
                "the router guard-precheck kill switch disables the only expected benefit",
            );
            return;
        }
        let (actions, _) = match self.resolve_router_actions() {
            Ok(resolved) => resolved,
            Err(reason) => {
                self.decline_router(RouterPhase::Declined, reason);
                return;
            }
        };
        if actions.len() < 2 {
            self.decline_router(RouterPhase::Declined, "fewer than two detected actions");
            return;
        }
        for (idx, action) in actions.iter().enumerate() {
            if self.router_prefix_state_guard(idx, action).is_none() {
                self.decline_router(
                    RouterPhase::Declined,
                    "not every action has an evaluation-order-safe state-only prefix guard",
                );
                return;
            }
        }
        if !self.router_actions_replay_safe(&actions) {
            self.decline_router(
                RouterPhase::Declined,
                "an action body is context-dependent, side-effecting, or not replay-safe",
            );
            return;
        }
        if !self.router_observers_replay_safe() {
            self.decline_router(
                RouterPhase::Declined,
                "an initial-state source or state observer is context-dependent, side-effecting, or not replay-safe",
            );
            return;
        }

        if self.coverage.actions.is_empty() {
            self.coverage.actions = Arc::clone(&actions);
            self.hybrid_dispatch.router_installed_actions = true;
        }
        self.hybrid_dispatch.router_sole_route_owner = true;
        self.hybrid_dispatch.router_actions = Some(actions);
        self.hybrid_dispatch.router_trial_checks_start =
            self.hybrid_dispatch.perf.guard_precheck_calls;
        self.hybrid_dispatch.router_trial_skips_start =
            self.hybrid_dispatch.perf.guard_precheck_skips;
        self.hybrid_dispatch.router_phase = RouterPhase::Trial;
        self.hybrid_dispatch.router_decision_reason =
            Some("64-parent exact parity and timing trial pending".into());
    }

    /// Whether this routed parent must also run canonical whole-`Next`.
    pub(in crate::check) fn router_parity_check_due(&mut self) -> bool {
        if !self.router_active() {
            return false;
        }
        self.hybrid_dispatch.router_active_parents += 1;
        match self.hybrid_dispatch.router_phase {
            RouterPhase::Trial => true,
            RouterPhase::Forced
                if self.hybrid_dispatch.router_parity_checked < ROUTER_TRIAL_PARENTS =>
            {
                true
            }
            RouterPhase::Active | RouterPhase::Forced => {
                self.hybrid_dispatch.router_active_parents % ROUTER_PARITY_SAMPLE_STRIDE == 0
            }
            _ => false,
        }
    }

    /// Bound the batch route's live per-parent successor set. Streaming remains
    /// preferable for unusually high-fanout parents even when guard sparsity is
    /// high, and this cap prevents AUTO from turning that into an unbounded
    /// memory-policy change.
    pub(in crate::check) fn router_fanout_admitted(&self, successors: usize) -> bool {
        successors <= ROUTER_MAX_SUCCESSORS_PER_PARENT
    }

    /// Whether AUTO currently owns a batch whose fanout is memory-gated.
    /// Compatible coverage co-ownership does not remove the trial's duplicate
    /// whole-Next materialization, so it must obey the same cap.
    #[inline]
    pub(in crate::check) fn router_auto_memory_cap_active(&self) -> bool {
        matches!(
            self.hybrid_dispatch.router_phase,
            RouterPhase::Trial | RouterPhase::Active
        )
    }

    /// Effective raw-successor materialization budget for the split router.
    ///
    /// The configured cap is a whole-parent contract, not a per-action cap.
    /// AUTO also promises a tighter 4,096-successor working-set bound. The
    /// per-action enumerator consumes this budget cumulatively so it cannot
    /// transiently materialize `actions × cap` before the post-sum failback.
    pub(in crate::check) fn router_raw_successor_cap(&self) -> Option<usize> {
        if !self.router_active() {
            return None;
        }
        let configured = self.ctx.shared().per_state_successor_cap;
        if self.router_auto_memory_cap_active() {
            Some(
                configured
                    .unwrap_or(ROUTER_MAX_SUCCESSORS_PER_PARENT)
                    .min(ROUTER_MAX_SUCCESSORS_PER_PARENT),
            )
        } else {
            configured
        }
    }

    /// Record a successful exact parity sample and complete AUTO's timing gate.
    pub(in crate::check) fn note_router_parity_match(
        &mut self,
        batch_ns: u128,
        whole_next_ns: u128,
    ) {
        self.hybrid_dispatch.router_parity_checked += 1;
        if !matches!(self.hybrid_dispatch.router_phase, RouterPhase::Trial) {
            return;
        }
        self.hybrid_dispatch.router_trial_checks = self
            .hybrid_dispatch
            .perf
            .guard_precheck_calls
            .saturating_sub(self.hybrid_dispatch.router_trial_checks_start);
        self.hybrid_dispatch.router_trial_skips = self
            .hybrid_dispatch
            .perf
            .guard_precheck_skips
            .saturating_sub(self.hybrid_dispatch.router_trial_skips_start);
        self.hybrid_dispatch.router_batch_ns += batch_ns;
        self.hybrid_dispatch.router_whole_next_ns += whole_next_ns;
        if self.hybrid_dispatch.router_parity_checked < ROUTER_TRIAL_PARENTS {
            return;
        }
        if !router_skip_rate_admitted(
            self.hybrid_dispatch.router_trial_skips,
            self.hybrid_dispatch.router_trial_checks,
        ) {
            self.decline_router(
                RouterPhase::Declined,
                "exact trial skipped fewer than 80% of action instances",
            );
            return;
        }
        if router_timing_admitted(
            self.hybrid_dispatch.router_batch_ns,
            self.hybrid_dispatch.router_whole_next_ns,
        ) {
            self.hybrid_dispatch.router_phase = RouterPhase::Active;
            self.hybrid_dispatch.router_decision_reason =
                Some("exact trial passed with at least 40% local generation speedup".into());
        } else {
            self.decline_router(
                RouterPhase::Declined,
                "exact trial did not show a 40% local generation speedup over whole-Next",
            );
        }
    }

    /// Permanently fail the standalone router back to canonical whole-`Next`.
    pub(in crate::check) fn failback_router(&mut self, reason: &str) {
        if self.router_active() {
            self.decline_router(RouterPhase::Failback, reason);
        }
    }

    /// Lazily classify actions and build the hybrid flat view (once per run).
    ///
    /// Self-contained so it can run from ANY successor-generation entry point
    /// (per-action, full-state, diff, streaming) — the flagship compound specs
    /// (Disruptor, btree) route through the diff BFS path, which never touches
    /// the per-action dispatcher. Actions are taken from the coverage decomposition
    /// when populated (so the eligibility vector aligns 1:1 with the per-action
    /// loop's `action_idx`), otherwise re-detected from `Next` (the diff path
    /// clears `coverage.actions`, but detection is deterministic and order-stable).
    pub(in crate::check) fn ensure_hybrid_dispatch_ready(&mut self) {
        self.ensure_router_ready();
        if self.hybrid_dispatch.initialized {
            return;
        }
        self.hybrid_dispatch.initialized = true;

        let enabled = std::env::var_os("TY_HYBRID_FLAT_VIEW").is_some_and(|v| v == "1");
        self.hybrid_dispatch.enabled = enabled;
        self.hybrid_dispatch.native_enabled = enabled && hybrid_native_enabled();
        // WP-17: coarse per-bucket wall timers, read once (diagnostics only).
        // WP-26: armed INDEPENDENTLY of `TY_HYBRID_FLAT_VIEW` so the plain
        // (streaming diff-BFS) arm reports the same buckets as the hybrid batch
        // arm — the two engines are only comparable if both are measured.
        self.hybrid_dispatch.perf.enabled =
            std::env::var_os("TY_HYBRID_PERF_DEBUG").is_some_and(|v| v == "1");
        // WP-34: the batch consumer's INNER split (materialize / fp / observe /
        // dedup / finish) costs five extra clock reads per successor, which is
        // itself visible inside `batch_consume`. Arming it separately keeps the
        // WP-26/WP-29 bucket table measured under `TY_HYBRID_PERF_DEBUG=1`
        // alone directly comparable across waves.
        self.hybrid_dispatch.perf.consume_detail =
            std::env::var_os("TY_HYBRID_CONSUME_PERF").is_some_and(|v| v == "1");
        // WP-26: interpreter branch of the per-action loop keeps successors in
        // their `DiffSuccessor` form instead of round-tripping
        // `DiffSuccessor -> ArrayState -> State -> ArrayState`. Escape hatch
        // `TY_HYBRID_INTERP_DIFF=0` restores the pre-WP-26 round trip verbatim.
        self.hybrid_dispatch.interp_diff_path =
            !std::env::var_os("TY_HYBRID_INTERP_DIFF").is_some_and(|v| v == "0");
        // WP-14: authoritative dispatch is subordinate to every switch above
        // and default OFF — unset, the validated shadow is byte-identical.
        self.hybrid_dispatch.authoritative_enabled =
            self.hybrid_dispatch.native_enabled && hybrid_native_authoritative_enabled();
        if !enabled {
            return;
        }

        // Build the projection from the run's inferred layout. Without a layout
        // (or with no flat-admissible var) the hybrid path stays inert.
        let registry = self.ctx.var_registry().clone();
        let view = self
            .flat_state_layout
            .as_deref()
            .and_then(|layout| HybridFlatView::from_layout(layout, &registry));

        // Detected actions, aligned with the per-action loop when available.
        let actions: Arc<Vec<DetectedAction>> = if !self.coverage.actions.is_empty() {
            Arc::clone(&self.coverage.actions)
        } else {
            let next_name = self
                .trace
                .cached_resolved_next_name
                .clone()
                .or_else(|| self.trace.cached_next_name.clone());
            match next_name.and_then(|n| self.module.op_defs.get(&n)) {
                Some(next_def) => Arc::new(detect_actions(next_def)),
                None => Arc::new(Vec::new()),
            }
        };

        let Some(view) = view else {
            self.emit_hybrid_classification_report(actions.len(), 0, 0, None, 0);
            return;
        };

        // Per-action footprints via the SAME static analysis POR/coverage use
        // (fail-closed on opacity). Aligned 1:1 with `actions`.
        let footprints = crate::por::extract_detected_action_dependencies(&self.ctx, &actions);
        let mut eligible = Vec::with_capacity(actions.len());
        let mut ast_footprints = Vec::with_capacity(actions.len());
        let mut eligible_count = 0usize;
        let m1_reads = hybrid_m1_read_rule_enabled();
        let mut compound_reading_eligible = 0usize;
        for deps in &footprints {
            // Opaque = reads/writes every variable → never hybrid-eligible.
            //
            // M0: reads ∪ writes ⊆ flat-admissible.
            // M1: writes ⊆ flat-admissible (reads may touch compound vars,
            //     which a compiled action services through the compound-read
            //     callout and the interpreter services natively). Sound
            //     because reconstruction Arc-shares compound vars from the
            //     parent, which is exact exactly when no compound var is
            //     written.
            let writes_ok = view.footprint_all_admissible(deps.writes.iter().map(|v| v.as_usize()));
            let reads_ok = view.footprint_all_admissible(deps.reads.iter().map(|v| v.as_usize()));
            let ok = !deps.opaque && writes_ok && (reads_ok || m1_reads);
            if ok {
                eligible_count += 1;
                if !reads_ok {
                    compound_reading_eligible += 1;
                }
            }
            // TY_HYBRID_ELIGIBILITY_DEBUG=1: per-action reason dump. The
            // headline metric (hybrid_eligible) is a single number; when it
            // does not move, THIS is what says which half of the rule blocked
            // each action and exactly which variable did it.
            if std::env::var_os("TY_HYBRID_ELIGIBILITY_DEBUG").is_some_and(|v| v == "1") {
                let name = actions
                    .get(ast_footprints.len())
                    .map(|a| a.name.clone())
                    .unwrap_or_else(|| format!("#{}", ast_footprints.len()));
                let blocking = |set: &FxHashSet<VarIndex>| -> Vec<usize> {
                    let mut v: Vec<usize> = set
                        .iter()
                        .map(|x| x.as_usize())
                        .filter(|i| !view.is_var_flat_admissible(*i))
                        .collect();
                    v.sort_unstable();
                    v
                };
                eprintln!(
                    "[hybrid-elig] action={name} eligible={ok} opaque={} writes_ok={writes_ok} \
                     reads_ok={reads_ok} blocking_writes={:?} blocking_reads={:?} \
                     unchanged={:?} opaque_reason={:?}",
                    deps.opaque,
                    blocking(&deps.writes),
                    blocking(&deps.reads),
                    blocking(&deps.unchanged),
                    deps.opaque_reason.as_deref().unwrap_or(""),
                );
            }
            eligible.push(ok);
            // Snapshot the AST footprint for the native admission dual gate
            // (M0-G4): compiled bytecode accesses must stay inside it.
            ast_footprints.push(HybridAstFootprint {
                reads: deps.reads.iter().map(|v| v.as_usize()).collect(),
                writes: deps.writes.iter().map(|v| v.as_usize()).collect(),
                opaque: deps.opaque,
            });
        }

        let flat_admissible = view.flat_admissible_count();
        let var_count = view.var_count();
        self.hybrid_dispatch.view = Some(view);
        self.hybrid_dispatch.eligible = eligible;
        self.hybrid_dispatch.ast_footprints = ast_footprints;
        if self.hybrid_dispatch.authoritative_enabled {
            self.hybrid_dispatch.machine = HybridAuthoritativeMachine::new(
                hybrid_env_u64("TY_HYBRID_BURN_IN", HYBRID_BURN_IN_DEFAULT),
                hybrid_env_u64("TY_HYBRID_SAMPLE", HYBRID_SAMPLE_DEFAULT),
                actions.len(),
            );
        }
        self.emit_hybrid_classification_report(
            actions.len(),
            eligible_count,
            flat_admissible,
            Some(var_count),
            compound_reading_eligible,
        );
    }

    /// One-shot G4 classification report (stderr) — the load-independent
    /// "native share" that will dispatch through trust-cg once its M0 lands.
    fn emit_hybrid_classification_report(
        &self,
        action_count: usize,
        eligible_count: usize,
        flat_admissible_vars: usize,
        var_count: Option<usize>,
        compound_reading_eligible: usize,
    ) {
        let vars = var_count
            .map(|v| v.to_string())
            .unwrap_or_else(|| "n/a".to_string());
        eprintln!(
            "[hybrid] TY_HYBRID_FLAT_VIEW on: spec={} vars={} flat_admissible_vars={} actions={} \
             hybrid_eligible={} interpreter={} m1_rule={} compound_reading_eligible={}",
            self.module.root_name,
            vars,
            flat_admissible_vars,
            action_count,
            eligible_count,
            action_count.saturating_sub(eligible_count),
            hybrid_m1_read_rule_enabled(),
            compound_reading_eligible,
        );
    }

    /// `true` when hybrid routing is engaged (switch on + projection built).
    #[inline]
    pub(in crate::check) fn hybrid_dispatch_active(&self) -> bool {
        self.hybrid_dispatch.active()
    }

    /// Whether the detected action at `action_idx` is hybrid-eligible.
    #[inline]
    pub(in crate::check) fn hybrid_action_eligible(&self, action_idx: usize) -> bool {
        self.hybrid_dispatch
            .eligible
            .get(action_idx)
            .copied()
            .unwrap_or(false)
    }

    /// WP-17: project the parent into its hybrid flat view ONCE per parent.
    ///
    /// The projection is invariant across the per-action loop (and across
    /// every successor of every action), so the caller computes it here once
    /// and passes it by reference to `hybrid_native_candidates_for_action`
    /// and `hybrid_route_successor`. `None` = hybrid inactive, no eligible
    /// action (nothing would consume it), or the parent does not project —
    /// each consumer keeps its own fail-closed decline accounting, exactly
    /// as when it projected the parent itself.
    pub(in crate::check) fn hybrid_project_parent_for_dispatch(
        &mut self,
        parent: &ArrayState,
    ) -> Option<FlatState> {
        if !self.hybrid_dispatch.active() {
            return None;
        }
        if !self.hybrid_dispatch.eligible.iter().any(|e| *e) {
            return None;
        }
        let t0 = self.hybrid_dispatch.perf.start();
        let out = self.hybrid_dispatch.view.as_ref()?.project(parent);
        perf_acc(&mut self.hybrid_dispatch.perf.project_parent_ns, t0);
        out
    }

    /// WP-17: memoized native key resolution + footprint dual-gate admission.
    ///
    /// Both inputs are run-fixed (the hybrid cache is assigned once before
    /// the BFS and never rebuilt; the split-action meta, AST footprints, and
    /// view are built once), so the per-action outcome cannot change within a
    /// run: resolve it on first use and replay the memo afterwards. An
    /// eligible action with no compiled key set (or one the dual gate
    /// rejects) therefore declines ONCE per run — `native_declined` counts
    /// actions, not (parent, action) instances.
    fn hybrid_admitted_keys(
        &mut self,
        action_idx: usize,
        action_name: &str,
    ) -> Option<Arc<Vec<String>>> {
        if let Some(resolved) = self
            .hybrid_dispatch
            .admitted_keys
            .get(action_idx)
            .and_then(|slot| slot.as_ref())
        {
            return resolved.clone();
        }
        let t0 = self.hybrid_dispatch.perf.start();
        let admitted = match self.trust_cg_hybrid_action_dispatch_keys(action_name) {
            // M0-G4 dual gate: compiled footprint vs AST footprint vs
            // flat-admissibility. ANY mismatch = permanent decline.
            Some(keys) if self.hybrid_native_footprint_admitted(action_idx, &keys) => {
                Some(Arc::new(keys))
            }
            _ => {
                self.hybrid_dispatch.stats.native_declined += 1;
                None
            }
        };
        perf_acc(&mut self.hybrid_dispatch.perf.key_resolve_ns, t0);
        if self.hybrid_dispatch.admitted_keys.len() <= action_idx {
            self.hybrid_dispatch
                .admitted_keys
                .resize(action_idx + 1, None);
        }
        self.hybrid_dispatch.admitted_keys[action_idx] = Some(admitted.clone());
        admitted
    }

    /// WP-17: whether the per-successor shadow route pays for this action.
    ///
    /// Shadow validation advances burn-in evidence (or checks a sample) only
    /// when a native artifact can actually execute. An eligible action whose
    /// admission memo resolved to a permanent decline gets nothing from the
    /// per-successor project/reconstruct/compare — its successors keep the
    /// interpreter states untouched either way (the route is fail-closed and
    /// value-identical when it succeeds), so skipping it cannot change the
    /// reachable-state set. With native dispatch off entirely the route IS
    /// the product (the validated projection shadow), so it always pays; an
    /// unresolved memo (e.g. no hybrid cache was ever built) also keeps the
    /// pre-WP-17 routing behavior.
    #[inline]
    pub(in crate::check) fn hybrid_shadow_route_worthwhile(&self, action_idx: usize) -> bool {
        if !self.hybrid_dispatch.native_enabled {
            return true;
        }
        match self
            .hybrid_dispatch
            .admitted_keys
            .get(action_idx)
            .and_then(|slot| slot.as_ref())
        {
            Some(admitted) => admitted.is_some(),
            None => true,
        }
    }

    /// Execute the compiled hybrid-layout artifacts for one (parent,
    /// hybrid-eligible action) pair and collect the native successor flat
    /// views (item 4 M0, TY_HYBRID_NATIVE=1).
    ///
    /// Fail-closed admission, checked per action instance in order:
    /// 1. hybrid dispatch active + native switch on + hybrid cache built;
    /// 2. compiled-footprint dual gate ([`Self::hybrid_native_footprint_admitted`]):
    ///    every dispatch key's declared bytecode footprint (entry + transitive
    ///    chunk callees) must be contained in ty's AST footprint AND entirely
    ///    flat-admissible — ANY mismatch declines;
    /// 3. buffer width parity between the projected parent view and the layout
    ///    the cache was compiled against;
    /// 4. every expansion key present and single-successor ABI; any native
    ///    runtime error declines the whole action instance.
    ///
    /// Returns `None` on any decline — the action instance stays on the
    /// interpreter + validating-shadow path, so the reachable-state set cannot
    /// change.
    pub(in crate::check) fn hybrid_native_candidates_for_action(
        &mut self,
        action_idx: usize,
        action_name: &str,
        parent: &ArrayState,
        parent_view: Option<&FlatState>,
    ) -> Option<HybridNativeCandidates> {
        if !self.hybrid_dispatch.active() || !self.hybrid_dispatch.native_enabled {
            return None;
        }
        if !self.hybrid_action_eligible(action_idx) {
            return None;
        }
        if !self.trust_cg_hybrid_action_dispatch_ready() {
            return None;
        }

        // WP-17: memoized key resolution + M0-G4 footprint dual gate (both are
        // run-fixed per action). `None` = permanent decline for this action.
        let keys = self.hybrid_admitted_keys(action_idx, action_name)?;

        // WP-17: the parent was projected ONCE per parent by the caller
        // (`hybrid_project_parent_for_dispatch`); a parent that does not
        // project declines exactly like the old per-instance projection did.
        let Some(parent_view) = parent_view else {
            self.hybrid_dispatch.stats.projection_declined += 1;
            return None;
        };
        let hybrid_layout = Arc::clone(self.hybrid_dispatch.view.as_ref()?.hybrid_layout());

        // Width parity vs the compiled layout (G5 was asserted at build; this
        // guards a stale cache against a re-inferred run layout).
        let width_ok = self
            .trust_cg_hybrid_jit_layout
            .as_ref()
            .is_some_and(|jit| jit.compact_slot_count() == parent_view.num_slots());
        if !width_ok {
            self.hybrid_dispatch.stats.native_declined += 1;
            return None;
        }

        // Run the compiled artifacts against the projected parent buffer, with
        // the parent's compound values published for the duration of the call
        // (item 4 M1). The guard BORROWS `parent.values()` — no copy, no
        // deserialization — and unpublishes on drop, including on every early
        // return below, so a compiled action can never reach a parent that is
        // no longer live. Callouts against an unpublished or stale context
        // return a typed status rather than dereferencing anything.
        let t_exec = self.hybrid_dispatch.perf.start();
        let (buffers, callout_status) = {
            let _ctx = tla_trust_cg::runtime_abi::compound_read::publish_compound_read_context(
                parent.values(),
            );
            let buffers = self.try_trust_cg_hybrid_action_by_keys(&keys, parent_view.buffer());
            // Read the sticky status INSIDE the publication scope: it is reset
            // on publish, so it describes exactly this action's callouts.
            let status = tla_trust_cg::runtime_abi::compound_read::compound_read_take_error();
            (buffers, status)
        };
        perf_acc(&mut self.hybrid_dispatch.perf.native_exec_ns, t_exec);
        // Any failed compound read means a `0` placeholder flowed into the
        // native computation. The result is void — discard the whole action
        // instance and let the interpreter produce the successors.
        if callout_status != tla_trust_cg::runtime_abi::compound_read::CR_OK {
            self.hybrid_dispatch.stats.native_errors += 1;
            return None;
        }
        let (mut buffer_slots, buffer_count) = match buffers {
            Some(Ok(output)) => output,
            Some(Err(())) => {
                // WP-21: a typed TypeMismatch is the fail-closed shape-guard
                // class (canonically: a LET def reading a union arm on a
                // parent whose enabling guard is false). Count it as a
                // recoverable per-parent decline, not an alarm-adjacent
                // error. Every other kind stays in `native_errors`.
                if super::trust_cg_dispatch::last_native_action_error_was_shape_guard() {
                    self.hybrid_dispatch.stats.native_guard_declined += 1;
                } else {
                    self.hybrid_dispatch.stats.native_errors += 1;
                }
                return None;
            }
            None => {
                self.hybrid_dispatch.stats.native_declined += 1;
                return None;
            }
        };

        // WP-14: decide how this instance participates — deterministically
        // from (projected parent buffer, action) and the resolved key set, so
        // reruns reproduce the exact same shadow/sampled/authoritative split.
        // With the gate off this is pure `Shadow`, byte-identical to M0/M1.
        let mode = if self.hybrid_dispatch.authoritative_enabled {
            let keys_hash = hybrid_keys_hash(&keys);
            let sample_hash = hybrid_sample_hash(parent_view.buffer(), action_idx);
            self.hybrid_dispatch
                .machine
                .decide_mode(action_idx, keys_hash, sample_hash)
        } else {
            HybridInstanceMode::Shadow
        };

        // WP-14 test-only fault injection: corrupt one slot of the first
        // sampled instance's first successor buffer (see
        // [`hybrid_inject_sampled_corruption_enabled`]). The sampled
        // differential must catch it and trip the permanent fail-back.
        if mode == HybridInstanceMode::Sampled
            && !self.hybrid_dispatch.corruption_injected
            && hybrid_inject_sampled_corruption_enabled()
        {
            if let Some(slot0) = buffer_slots.first_mut() {
                *slot0 ^= 0x5a5a;
                self.hybrid_dispatch.corruption_injected = true;
                eprintln!(
                    "[hybrid] TEST: injected corruption into a sampled native successor buffer \
                     (action_idx={action_idx})"
                );
            }
        }

        let state_len = parent_view.num_slots();
        let expected_slots = buffer_count.checked_mul(state_len)?;
        if buffer_slots.len() != expected_slots {
            self.hybrid_dispatch.stats.native_errors += 1;
            return None;
        }
        let mut slots = buffer_slots.into_iter();
        let mut views = Vec::with_capacity(buffer_count);
        for _ in 0..buffer_count {
            let buffer: Box<[i64]> = slots.by_ref().take(state_len).collect();
            match FlatState::try_from_buffer(buffer, Arc::clone(&hybrid_layout)) {
                Ok(view) => views.push(view),
                Err(_) => {
                    // Width drift in a native output buffer: decline the whole
                    // action instance (fail closed).
                    self.hybrid_dispatch.stats.native_errors += 1;
                    return None;
                }
            }
        }
        self.hybrid_dispatch.stats.native_dispatched += 1;
        Some(HybridNativeCandidates::new(views, action_idx, mode))
    }

    /// WP-14: reconstruct EVERY native candidate into a full successor state
    /// against the parent (flat-admissible vars from the native buffer,
    /// compound vars Arc-shared from the parent — writes are strictly
    /// flat-admissible by the eligibility + dual admission gates, so sharing
    /// is exact). All-or-nothing and fail-closed: `None` (any candidate
    /// declines reconstruction) consumes nothing, and the caller falls back
    /// to the interpreter + full-differential shadow for this instance.
    pub(in crate::check) fn hybrid_reconstruct_all_native_candidates(
        &mut self,
        parent: &ArrayState,
        native: &HybridNativeCandidates,
        registry: &VarRegistry,
        parent_view: Option<&FlatState>,
    ) -> Option<Vec<ArrayState>> {
        // WP-29 lever 2: decode ONLY the flat-admissible variables whose slots
        // actually differ from the parent's projected buffer.
        self.ensure_delta_var_layouts(registry);
        let t0 = self.hybrid_dispatch.perf.start();
        let mut failed = false;
        let out = {
            let view = self.hybrid_dispatch.view.as_ref()?;
            let mut out = Vec::with_capacity(native.views.len());
            let delta_base = (delta_reconstruct_enabled()
                && !self.hybrid_dispatch.delta_var_layouts.is_empty())
            .then_some(parent_view)
            .flatten();
            let mut decoded_vars = 0u64;
            let mut total_vars = 0u64;
            let mut delta_hits = 0u64;
            let mut delta_verify_mismatches = 0u64;
            let verify = delta_reconstruct_verify_enabled();
            let var_count = view.var_count();
            for candidate in &native.views {
                let rebuilt = match delta_base {
                    Some(base) => {
                        let delta = hybrid_reconstruct_delta(
                            view,
                            &self.hybrid_dispatch.delta_var_layouts,
                            parent,
                            base,
                            candidate,
                            registry,
                            &mut decoded_vars,
                            &mut total_vars,
                        );
                        if delta.is_some() {
                            delta_hits += 1;
                        }
                        // WP-29 lever 2 self-check: with the verify switch on,
                        // reconstruct the SAME candidate the whole-buffer way
                        // and compare variable by variable. A disagreement is a
                        // loud alarm and the whole-buffer state wins.
                        if verify {
                            if let Some(delta_state) = delta.as_ref() {
                                let whole = view.reconstruct(parent, candidate, registry);
                                match whole {
                                    Some(whole_state)
                                        if states_value_equal(
                                            delta_state,
                                            &whole_state,
                                            var_count,
                                        ) => {}
                                    other => {
                                        delta_verify_mismatches += 1;
                                        // Fail closed: keep the whole-buffer
                                        // reconstruction (or decline if it too
                                        // could not decode).
                                        match other {
                                            Some(whole_state) => {
                                                out.push(whole_state);
                                                continue;
                                            }
                                            None => {
                                                failed = true;
                                                break;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        // Fail-open to the whole-buffer decode: the delta path
                        // declines (returning `None`) only on a shape it cannot
                        // handle, never on a semantic disagreement.
                        delta.or_else(|| view.reconstruct(parent, candidate, registry))
                    }
                    None => view.reconstruct(parent, candidate, registry),
                };
                match rebuilt {
                    Some(arr) => out.push(arr),
                    None => {
                        failed = true;
                        break;
                    }
                }
            }
            let hd = &mut self.hybrid_dispatch;
            hd.perf.delta_reconstructed += delta_hits;
            hd.perf.delta_reconstruct_vars_decoded += decoded_vars;
            hd.perf.delta_reconstruct_vars_total += total_vars;
            if delta_verify_mismatches > 0 {
                hd.stats.mismatch_fallback += delta_verify_mismatches;
                eprintln!(
                    "[hybrid] ERROR: delta/whole-buffer reconstruction disagreement on {} \
                     native successor(s) — the whole-buffer state was used (fail closed)",
                    delta_verify_mismatches,
                );
            }
            out
        };
        perf_acc(&mut self.hybrid_dispatch.perf.reconstruct_ns, t0);
        if failed {
            self.hybrid_dispatch.stats.projection_declined += 1;
            return None;
        }
        Some(out)
    }

    /// WP-29 lever 2: build (once) the per-variable single-variable layouts the
    /// delta reconstruction decodes a changed variable through.
    ///
    /// Each entry is the hybrid layout's kind for that variable, re-laid out on
    /// its own at offset 0 — so the variable's slot range, lifted verbatim out
    /// of a native successor buffer, decodes through exactly the same code the
    /// whole-buffer path uses for it. Non-admissible (`Dynamic`) variables get
    /// `None`: they are always taken from the parent.
    fn ensure_delta_var_layouts(&mut self, registry: &VarRegistry) {
        if self.hybrid_dispatch.delta_var_layouts_built || !delta_reconstruct_enabled() {
            return;
        }
        self.hybrid_dispatch.delta_var_layouts_built = true;
        let Some(view) = self.hybrid_dispatch.view.as_ref() else {
            return;
        };
        let hybrid_layout = view.hybrid_layout();
        let var_count = view.var_count();
        if hybrid_layout.var_count() != var_count || registry.len() != var_count {
            return;
        }
        let mut layouts: Vec<Option<Arc<crate::state::StateLayout>>> =
            Vec::with_capacity(var_count);
        for var_idx in 0..var_count {
            if !view.is_var_flat_admissible(var_idx) {
                layouts.push(None);
                continue;
            }
            let Some(var_layout) = hybrid_layout.var_layout(var_idx) else {
                layouts.push(None);
                continue;
            };
            // Fail closed on the two kinds whose single-variable decode is NOT
            // the whole-buffer decode: `Dynamic` and `Bitmask` are taken from
            // the compound original by `try_to_array_state_with_fallback`,
            // while the standalone decode would synthesize a placeholder. Both
            // are non-flat-admissible today (so unreachable here); pinning it
            // keeps a future admissibility widening from silently diverging.
            if matches!(
                var_layout.kind,
                crate::state::VarLayoutKind::Dynamic | crate::state::VarLayoutKind::Bitmask { .. }
            ) {
                layouts.push(None);
                continue;
            }
            let name = registry.name(VarIndex::new(var_idx));
            let single_registry = VarRegistry::from_names([name].into_iter());
            let single =
                crate::state::StateLayout::new(&single_registry, vec![var_layout.kind.clone()]);
            // Width parity with the slice we will hand it: anything else is a
            // layout the delta path must not use.
            if single.total_slots() == var_layout.slot_count {
                layouts.push(Some(Arc::new(single)));
            } else {
                layouts.push(None);
            }
        }
        self.hybrid_dispatch.delta_var_layouts = layouts;
    }

    /// WP-29 lever 1: the memoized enabling guard for one detected action.
    ///
    /// Extraction is a pure function of run-fixed state (the action AST, the
    /// module operator table, the config operator-replacement map), so it is
    /// resolved ONCE per action and replayed afterwards. The accepted guard is
    /// held in an `Arc` for the whole run, so the node addresses handed to the
    /// evaluator's pointer-keyed caches stay live and unique — the same
    /// run-stability contract `enumerate_successors_body` documents.
    fn action_state_guard(
        &mut self,
        action_idx: usize,
        action: &DetectedAction,
    ) -> Option<Arc<tla_core::Spanned<tla_core::ast::Expr>>> {
        if let Some(resolved) = self
            .hybrid_dispatch
            .action_state_guards
            .get(action_idx)
            .and_then(|slot| slot.as_ref())
        {
            return resolved.clone();
        }
        let extracted = {
            let mut bound: Vec<String> = Vec::new();
            extract_action_state_guard(&self.ctx, &self.module.op_defs, &action.expr, &mut bound, 0)
                .map(Arc::new)
        };
        if hybrid_guard_debug_enabled() {
            match extracted.as_ref() {
                Some(guard) => eprintln!(
                    "[hybrid] guard idx={action_idx} action={} EXTRACTED {}",
                    action.name,
                    guard_debug_render(&guard.node, 0)
                ),
                None => eprintln!(
                    "[hybrid] guard idx={action_idx} action={} NONE root={}",
                    action.name,
                    guard_expr_variant(&action.expr.node)
                ),
            }
        }
        if self.hybrid_dispatch.action_state_guards.len() <= action_idx {
            self.hybrid_dispatch
                .action_state_guards
                .resize(action_idx + 1, None);
        }
        self.hybrid_dispatch.action_state_guards[action_idx] = Some(extracted.clone());
        extracted
    }

    /// Evaluation-order-safe guard used by every interpreter-skipping
    /// precheck, including the standalone router.
    fn router_prefix_state_guard(
        &mut self,
        action_idx: usize,
        action: &DetectedAction,
    ) -> Option<Arc<tla_core::Spanned<tla_core::ast::Expr>>> {
        if let Some(resolved) = self
            .hybrid_dispatch
            .router_prefix_state_guards
            .get(action_idx)
            .and_then(|slot| slot.as_ref())
        {
            return resolved.clone();
        }
        let extracted = {
            let mut bound = Vec::new();
            extract_router_prefix_state_guard(
                &self.ctx,
                &self.module.op_defs,
                &action.expr,
                &mut bound,
                0,
            )
            .map(Arc::new)
        };
        if self.hybrid_dispatch.router_prefix_state_guards.len() <= action_idx {
            self.hybrid_dispatch
                .router_prefix_state_guards
                .resize(action_idx + 1, None);
        }
        self.hybrid_dispatch.router_prefix_state_guards[action_idx] = Some(extracted.clone());
        extracted
    }

    /// WP-34 diagnostics: resolve (and therefore dump, under
    /// `TY_HYBRID_GUARD_DEBUG=1`) the extracted guard of EVERY detected action,
    /// including the ones the pre-check never reaches because they dispatch
    /// natively. Extraction is memoized and pure, so this only pre-populates
    /// the cache the pre-check would fill lazily — no behavioural change.
    pub(in crate::check) fn hybrid_dump_action_guards(&mut self, actions: &[DetectedAction]) {
        if !hybrid_guard_debug_enabled() {
            return;
        }
        for (idx, action) in actions.iter().enumerate() {
            let _ = self.action_state_guard(idx, action);
        }
    }

    /// WP-29 lever 1: `true` only when this (parent, action) instance is
    /// PROVABLY disabled — the action's extracted state-only guard evaluated to
    /// `FALSE` against the parent, and `action => guard`, so the enumeration
    /// would return the empty set.
    ///
    /// Every other outcome returns `false` (undecided) and the caller runs the
    /// full enumeration exactly as before: no guard for this action, a
    /// non-boolean value, `TRUE`, or ANY evaluation error. An error is
    /// deliberately swallowed rather than propagated — the enumerator remains
    /// the sole authority on error semantics for the action.
    pub(in crate::check) fn action_definitely_disabled_in_parent(
        &mut self,
        action_idx: usize,
        action: &DetectedAction,
        parent: &ArrayState,
    ) -> bool {
        let t0 = self.hybrid_dispatch.perf.start();
        // Always use the leftmost prefix. A broad later conjunct can prove the
        // successor relation empty, but using it here could suppress an error
        // from an earlier canonical conjunct.
        let guard = self.router_prefix_state_guard(action_idx, action);
        let Some(guard) = guard else {
            perf_acc(&mut self.hybrid_dispatch.perf.guard_precheck_ns, t0);
            return false;
        };
        let disabled = guard_proves_disabled(&mut self.ctx, guard.as_ref(), parent);
        let perf = &mut self.hybrid_dispatch.perf;
        perf.guard_precheck_calls += 1;
        if disabled {
            perf.guard_precheck_skips += 1;
        }
        perf_acc(&mut perf.guard_precheck_ns, t0);
        disabled
    }

    /// WP-14: commit one authoritative instance — its native successor set
    /// was enqueued and the interpreter enumeration skipped. Marks every
    /// candidate consumed (the set was used in full, so no residue applies).
    pub(in crate::check) fn hybrid_commit_authoritative_instance(
        &mut self,
        native: &mut HybridNativeCandidates,
    ) {
        native.mark_all_consumed();
        self.hybrid_dispatch.stats.authoritative_dispatched += 1;
    }

    /// WP-14: demote an authoritative instance that could not be consumed
    /// (reconstruction or constraint evaluation failed) back to the
    /// interpreter + full-differential shadow. Nothing was consumed yet, so
    /// the shadow differential runs over the complete candidate set; the
    /// incomplete consumption marks the instance unproven (resets burn-in
    /// without implying a semantic divergence).
    pub(in crate::check) fn hybrid_demote_authoritative_instance(
        &mut self,
        native: &mut HybridNativeCandidates,
    ) {
        native.mode = HybridInstanceMode::Shadow;
        native.mark_unproven();
    }

    /// M0-G4 dual gate: every compiled dispatch key's declared bytecode
    /// footprint must be contained in ty's AST footprint for the action AND be
    /// entirely flat-admissible. Compiled reads may fall in AST reads OR
    /// writes (prime-mode re-reads of written vars); compiled writes must be
    /// AST writes. ANY variable outside those bounds — or an opaque AST
    /// footprint, or a key with no declared footprint — declines.
    fn hybrid_native_footprint_admitted(&self, action_idx: usize, keys: &[String]) -> bool {
        let hd = &self.hybrid_dispatch;
        let Some(view) = hd.view.as_ref() else {
            return false;
        };
        let Some(ast) = hd.ast_footprints.get(action_idx) else {
            return false;
        };
        if ast.opaque {
            return false;
        }
        let Some(cache) = self.trust_cg_hybrid_cache.as_ref() else {
            return false;
        };
        keys.iter().all(|key| {
            let Some((reads, writes)) = cache.action_declared_footprint(key) else {
                return false;
            };
            // M1: a compiled read of a NON-flat-admissible var is admitted only
            // when the artifact explicitly DECLARED it as a compound-read
            // callout var. The declaration is what makes the read serviceable
            // by the published parent context; an undeclared read of a
            // placeholder slot would be reading an inert zero.
            //
            // While the lowering's callout emission is still switched off this
            // set is always empty, so the condition degrades EXACTLY to M0's
            // "every compiled read must be flat-admissible" — the relaxed AST
            // rule above can widen the shadow, never the native path.
            let declared_compound_reads = cache.action_declared_compound_read_vars(key);
            reads.iter().all(|&v| {
                let raw = v;
                let v = usize::from(v);
                (ast.reads.contains(&v) || ast.writes.contains(&v))
                    && (view.is_var_flat_admissible(v) || declared_compound_reads.contains(&raw))
            }) && writes.iter().all(|&v| {
                let v = usize::from(v);
                // Writes stay strictly flat in M1 — the compound-read callout
                // is READ-ONLY, and reconstruction Arc-shares compound vars
                // from the parent, which a compound write would invalidate.
                ast.writes.contains(&v) && view.is_var_flat_admissible(v)
            })
        })
    }

    /// The declared compound-read footprint for one compiled action key: the
    /// compound (hybrid-placeholder) vars the artifact reads through the
    /// callout. Empty for every artifact compiled without the callout.
    #[must_use]
    pub fn hybrid_declared_compound_read_vars_for_testing(&self, key: &str) -> Vec<u16> {
        self.trust_cg_hybrid_cache
            .as_ref()
            .map(|cache| cache.action_declared_compound_read_vars(key).to_vec())
            .unwrap_or_default()
    }

    /// Consume the native candidate matching a CONSTRAINT-FILTERED interpreter
    /// successor (item 4 M0).
    ///
    /// Native execution is constraint-blind: it enumerates the action's raw
    /// successors, while the per-action loop applies state/action constraints
    /// before routing. A filtered interpreter successor therefore has a
    /// legitimate native counterpart that would otherwise be misreported as
    /// residue at action end. Consuming it here (match only — nothing is
    /// enqueued or counted as routed) keeps the divergence accounting exact.
    pub(in crate::check) fn hybrid_consume_native_match_for_filtered_successor(
        &mut self,
        native: Option<&mut HybridNativeCandidates>,
        interp_succ: &ArrayState,
    ) {
        let Some(candidates) = native else {
            return;
        };
        let Some(view) = self.hybrid_dispatch.view.as_ref() else {
            return;
        };
        let Some(interp_view) = view.project(interp_succ) else {
            return;
        };
        let _ = candidates.take_matching(&interp_view);
    }

    /// Close out one (parent, action) native execution: any native successor
    /// the interpreter never matched is a native/interpreter divergence —
    /// count it in `mismatch_fallback` (the loud alarm). The interpreter's
    /// successors were enqueued regardless (except on WP-14 authoritative
    /// instances, whose candidates were enqueued in full and marked consumed
    /// at commit), so the reachable-state set is unchanged.
    ///
    /// WP-14: this is also where the burn-in machine consumes the instance's
    /// differential outcome — consecutive-clean advance/flip on a fully-clean
    /// Shadow instance, reset on any non-clean one, permanent whole-run
    /// fail-back on any sampled mismatch (or any semantic divergence observed
    /// once any action has flipped). With the authoritative gate off the
    /// accounting below is byte-identical to M0/M1.
    pub(in crate::check) fn hybrid_finish_native_action(
        &mut self,
        native: Option<HybridNativeCandidates>,
    ) {
        let Some(native) = native else {
            return;
        };
        let residue = native.unconsumed();
        if residue > 0 {
            let stats = &mut self.hybrid_dispatch.stats;
            stats.native_residue += residue;
            stats.mismatch_fallback += residue;
        }
        if !self.hybrid_dispatch.authoritative_enabled {
            return;
        }
        match native.mode {
            HybridInstanceMode::Authoritative => {
                // No differential ran; `authoritative_dispatched` was counted
                // at commit and consumption is total by construction.
            }
            HybridInstanceMode::Sampled | HybridInstanceMode::Shadow => {
                let dirty = native.dirty || residue > 0;
                let clean = !dirty && !native.unproven;
                if native.mode == HybridInstanceMode::Sampled {
                    let stats = &mut self.hybrid_dispatch.stats;
                    stats.sampled_checks += 1;
                    if !clean {
                        stats.sampled_mismatches += 1;
                    }
                }
                let tripped = self.hybrid_dispatch.machine.record_result(
                    native.action_idx,
                    native.mode,
                    clean,
                    dirty,
                );
                if tripped {
                    eprintln!(
                        "[hybrid] ERROR: native/interpreter divergence on action_idx={} \
                         (mode={:?} dirty={} residue={residue} unproven={}); permanently \
                         failing back to interpreter-authoritative dispatch for ALL actions \
                         for the rest of this run",
                        native.action_idx, native.mode, native.dirty, native.unproven,
                    );
                }
            }
        }
    }

    /// Route a hybrid-eligible action's successor through the flat-view
    /// projection. Returns `Some(routed_successor)` when the reconstruction
    /// matches the interpreter successor exactly (safe to enqueue in its place,
    /// byte-for-byte identical), or `None` to fall back to the interpreter
    /// successor (fail-closed on any projection failure or divergence).
    ///
    /// With `native` candidates present (TY_HYBRID_NATIVE=1 and the action
    /// instance passed native admission), the updated flat view is the NATIVE
    /// successor buffer that byte-exactly equals the projected interpreter
    /// successor — consumed from the candidates so each native successor can
    /// match at most one interpreter successor. No byte-exact native candidate
    /// = `mismatch_fallback` (the loud alarm) + interpreter successor kept.
    /// Without candidates, the updated view is the validating-shadow
    /// projection of the interpreter successor (the original M0 stub).
    ///
    /// Either way the per-successor value-equality differential (step 4) stays
    /// intact and authoritative — this never changes the reachable-state set:
    /// the returned state, when `Some`, is value-identical to `interp_succ`;
    /// when `None`, the caller keeps `interp_succ`.
    pub(in crate::check) fn hybrid_route_successor(
        &mut self,
        parent: &ArrayState,
        interp_succ: &ArrayState,
        registry: &VarRegistry,
        native: Option<&mut HybridNativeCandidates>,
        parent_view: Option<&FlatState>,
    ) -> Option<ArrayState> {
        if !self.hybrid_dispatch.active() {
            return None;
        }
        // Disjoint field borrows of `self.hybrid_dispatch`: `view` (shared),
        // `stats` and `perf` (unique). The shadow body is a free function
        // taking `view` by reference, so no `&self` borrow conflicts.
        let hd = &mut self.hybrid_dispatch;
        let view = hd.view.as_ref()?;
        let stats = &mut hd.stats;
        let perf = &mut hd.perf;

        // WP-14: keep the &mut candidates borrow alive past the match so the
        // later value-equality verdict can mark the INSTANCE (not just the
        // global counters) — the burn-in machine consumes per-instance flags.
        let mut native_matched_candidates: Option<&mut HybridNativeCandidates> = None;
        let updated_view = if let Some(candidates) = native {
            // NATIVE path: match the projected interpreter successor against
            // the native successor buffers, byte-for-byte.
            let t0 = perf.start();
            let interp_view = view.project(interp_succ);
            perf_acc(&mut perf.project_succ_ns, t0);
            let Some(interp_view) = interp_view else {
                stats.projection_declined += 1;
                // An interpreter successor of a natively-executed action that
                // cannot even PROJECT can never have a native counterpart:
                // the divergence surfaces as residue at action end, which
                // already marks the instance non-clean for burn-in.
                candidates.mark_unproven();
                return None;
            };
            match candidates.take_matching(&interp_view) {
                Some(native_view) => {
                    stats.native_matched += 1;
                    native_matched_candidates = Some(candidates);
                    native_view
                }
                None => {
                    // The native execution produced no successor matching this
                    // interpreter successor: divergence. Keep the interpreter
                    // successor (authoritative) and raise the alarm.
                    stats.native_unmatched_interp += 1;
                    stats.mismatch_fallback += 1;
                    candidates.mark_dirty();
                    hybrid_dump_divergence(
                        view,
                        parent,
                        interp_succ,
                        &interp_view,
                        registry,
                        candidates,
                    );
                    return None;
                }
            }
        } else {
            // SHADOW path (no native execution for this action instance).
            // 1. The parent's flat view was projected ONCE per parent by the
            //    caller (WP-17); a parent that does not project declines per
            //    successor exactly like the old per-successor projection did.
            let Some(parent_view) = parent_view else {
                stats.projection_declined += 1;
                return None;
            };
            // 2. Interpreter-through-projection shadow body.
            let t0 = perf.start();
            let updated_view = hybrid_shadow_flat_view_dispatch(view, parent_view, interp_succ);
            perf_acc(&mut perf.project_succ_ns, t0);
            let Some(updated_view) = updated_view else {
                stats.projection_declined += 1;
                return None;
            };
            updated_view
        };
        // 3. Reconstruct the successor: admissible vars from the updated flat
        //    view, compound vars Arc-shared from the parent.
        let t0 = perf.start();
        let routed = view.reconstruct(parent, &updated_view, registry);
        perf_acc(&mut perf.reconstruct_ns, t0);
        let Some(routed) = routed else {
            stats.projection_declined += 1;
            // The differential could not complete for this successor: the
            // instance's burn-in evidence is void (no semantic divergence
            // implied — the candidate byte-matched before reconstruction).
            if let Some(candidates) = native_matched_candidates {
                candidates.mark_unproven();
            }
            return None;
        };

        // 4. Differential (fail-closed): the routed successor MUST equal the
        //    interpreter successor. On any divergence, keep the interpreter
        //    successor and raise the alarm counter.
        let t0 = perf.start();
        let value_equal = states_value_equal(&routed, interp_succ, view.var_count());
        perf_acc(&mut perf.value_eq_ns, t0);
        if value_equal {
            stats.routed += 1;
            Some(routed)
        } else {
            stats.mismatch_fallback += 1;
            if let Some(candidates) = native_matched_candidates {
                candidates.mark_dirty();
            }
            None
        }
    }

    /// Validating shadow for the diff BFS paths (Disruptor's batch path, btree's
    /// streaming path).
    ///
    /// The diff paths enumerate the monolithic `Next` with no per-action
    /// boundaries, so instead of per-ACTION routing this validates per
    /// SUCCESSOR: a successor whose changed-variable footprint is entirely
    /// flat-admissible is exactly a successor a hybrid-native action would
    /// produce (the un-flattenable vars are untouched). We project it, reconstruct
    /// against the compound parent, and assert value-equality with the
    /// interpreter-materialized successor — proving the hybrid representation +
    /// projection + reconstruction are exact on the flagship compound specs at
    /// scale.
    ///
    /// This is a **pure shadow**: it NEVER changes the enqueued successor
    /// (`succ` is materialized and consumed by the caller regardless), so the
    /// reachable-state count is provably unchanged. It only reads `(parent,
    /// succ, changes)` and bumps counters. `mismatch_fallback` MUST stay 0.
    pub(in crate::check) fn hybrid_shadow_validate_diff(
        &mut self,
        parent: &ArrayState,
        succ: &ArrayState,
        changed_vars: &[VarIndex],
        registry: &VarRegistry,
    ) {
        if !self.hybrid_dispatch.active() {
            return;
        }
        // Only successors whose CHANGED variables are all flat-admissible are
        // hybrid-eligible: then every un-flattenable var is untouched, so
        // reconstructing from the compound parent reproduces the successor.
        let hd = &mut self.hybrid_dispatch;
        let Some(view) = hd.view.as_ref() else {
            return;
        };
        let stats = &mut hd.stats;
        if !view.footprint_all_admissible(changed_vars.iter().map(|vi| vi.as_usize())) {
            // Touches a compound var — routed to the interpreter (not eligible).
            return;
        }
        let Some(routed) = view.project_then_reconstruct(parent, succ, registry) else {
            stats.projection_declined += 1;
            return;
        };
        if states_value_equal(&routed, succ, view.var_count()) {
            stats.routed += 1;
        } else {
            stats.mismatch_fallback += 1;
        }
    }

    /// End-of-run routing summary (stderr), only when the switch is on. G3
    /// evidence: `mismatch_fallback` MUST be 0.
    pub(in crate::check) fn report_hybrid_dispatch_summary(&self) {
        self.report_router_summary();
        if !self.hybrid_dispatch.enabled {
            // WP-26: the bucket split is still printed on the plain arm, so the
            // streaming engine and the per-action batch engine are directly
            // comparable on the same spec. Routing counters stay suppressed —
            // there is no routing when the master switch is off.
            self.report_hybrid_perf_buckets();
            return;
        }
        let s = &self.hybrid_dispatch.stats;
        let m = &self.hybrid_dispatch.machine;
        eprintln!(
            "[hybrid] routing summary: routed={} mismatch_fallback={} projection_declined={} \
             native_dispatched={} native_matched={} native_unmatched_interp={} native_residue={} \
             native_declined={} native_errors={} native_guard_declined={} \
             authoritative_actions={} \
             authoritative_dispatched={} sampled_checks={} sampled_mismatches={} \
             burn_in_pending={} authoritative_failback={}",
            s.routed,
            s.mismatch_fallback,
            s.projection_declined,
            s.native_dispatched,
            s.native_matched,
            s.native_unmatched_interp,
            s.native_residue,
            s.native_declined,
            s.native_errors,
            s.native_guard_declined,
            m.authoritative_action_count(),
            s.authoritative_dispatched,
            s.sampled_checks,
            s.sampled_mismatches,
            m.burn_in_pending_count(),
            m.failback(),
        );
        self.report_hybrid_perf_buckets();
    }

    /// End-of-run standalone-router summary. Default AUTO stays silent; the
    /// explicit force-on mode and `TY_ROUTER_DIAG=1` request diagnostics.
    fn report_router_summary(&self) {
        let diagnostic = std::env::var("TY_ROUTER_DIAG")
            .is_ok_and(|value| tla_backend::env_flag_enabled(&value));
        if !diagnostic && !matches!(router_request(), RouterRequest::Forced) {
            return;
        }
        let perf = &self.hybrid_dispatch.perf;
        let trial_skip_rate = if self.hybrid_dispatch.router_trial_checks == 0 {
            0.0
        } else {
            100.0 * self.hybrid_dispatch.router_trial_skips as f64
                / self.hybrid_dispatch.router_trial_checks as f64
        };
        let measured_speedup = if self.hybrid_dispatch.router_whole_next_ns == 0 {
            0.0
        } else {
            100.0
                * (1.0
                    - self.hybrid_dispatch.router_batch_ns as f64
                        / self.hybrid_dispatch.router_whole_next_ns as f64)
        };
        eprintln!(
            "[router] summary: phase={:?} active={} sole_owner={} installed_actions={} \
             actions={} pilot_parents={} trial_checks={} trial_skips={} \
             trial_skip_rate={trial_skip_rate:.1}% parity_checks={} \
             local_generation_speedup={measured_speedup:.1}% batch_parents={} guard_calls={} \
             guard_skips={} reason={}",
            self.hybrid_dispatch.router_phase,
            self.router_active(),
            self.hybrid_dispatch.router_sole_route_owner,
            self.hybrid_dispatch.router_installed_actions,
            self.hybrid_dispatch
                .router_actions
                .as_ref()
                .map_or(0, |actions| actions.len()),
            self.hybrid_dispatch.router_pilot_parents,
            self.hybrid_dispatch.router_trial_checks,
            self.hybrid_dispatch.router_trial_skips,
            self.hybrid_dispatch.router_parity_checked,
            perf.batch_parents,
            perf.guard_precheck_calls,
            perf.guard_precheck_skips,
            self.hybrid_dispatch
                .router_decision_reason
                .as_deref()
                .unwrap_or("none"),
        );
    }

    /// WP-17/WP-26: coarse per-bucket wall split (`TY_HYBRID_PERF_DEBUG=1`).
    ///
    /// Printed on BOTH engines: the `batch_*`/`interp_*` buckets describe the
    /// per-action batch path that hybrid dispatch forces, the `stream_*`
    /// buckets describe the streaming diff-BFS path the plain arm uses. Running
    /// the same spec with and without the hybrid gates therefore yields a
    /// like-for-like decomposition of the interpreted remainder.
    fn report_hybrid_perf_buckets(&self) {
        let p = &self.hybrid_dispatch.perf;
        if !p.enabled {
            return;
        }
        let ms = |ns: u64| ns as f64 / 1e6;
        eprintln!(
            "[hybrid] perf(ms): project_parent={:.0} project_succ={:.0} key_resolve={:.0} \
             native_exec={:.0} reconstruct={:.0} value_eq={:.0} constraints={:.0} \
             fp_to_state={:.0} interp_enum={:.0} interp_enum_empty={:.0} \
             interp_succ_build={:.0} parent_setup={:.0} \
             batch_gen={:.0} batch_consume={:.0} stream_phase_a={:.0} stream_phase_b={:.0} \
             guard_precheck={:.0}",
            ms(p.project_parent_ns),
            ms(p.project_succ_ns),
            ms(p.key_resolve_ns),
            ms(p.native_exec_ns),
            ms(p.reconstruct_ns),
            ms(p.value_eq_ns),
            ms(p.constraints_ns),
            ms(p.fp_to_state_ns),
            ms(p.interp_enum_ns),
            ms(p.interp_enum_empty_ns),
            ms(p.interp_succ_build_ns),
            ms(p.parent_setup_ns),
            ms(p.batch_gen_ns),
            ms(p.batch_consume_ns),
            ms(p.stream_phase_a_ns),
            ms(p.stream_phase_b_ns),
            ms(p.guard_precheck_ns),
        );
        eprintln!(
            "[hybrid] perf(consume ms): materialize={:.0} fp={:.0} observe={:.0} dedup={:.0} \
             finish={:.0} | succ={} survivors={} lazy_scan_skipped={} lazy_vars_scanned={} \
             lazy_vars_total={}",
            ms(p.consume_materialize_ns),
            ms(p.consume_fp_ns),
            ms(p.consume_observe_ns),
            ms(p.consume_dedup_ns),
            ms(p.consume_finish_ns),
            p.consume_succ,
            p.consume_survivors,
            p.consume_lazy_scan_skipped,
            p.consume_lazy_vars_scanned,
            p.consume_lazy_vars_total,
        );
        eprintln!(
            "[hybrid] perf(counts): batch_parents={} interp_succ={} interp_enum_calls={} \
             interp_enum_empty_calls={} stream_parents={} stream_succ={} interp_diff_path={} \
             guard_precheck_calls={} guard_precheck_skips={} delta_reconstructed={} \
             delta_vars_decoded={} delta_vars_total={}",
            p.batch_parents,
            p.interp_succ_count,
            p.interp_enum_calls,
            p.interp_enum_empty_calls,
            p.stream_parents,
            p.stream_succ_count,
            self.hybrid_dispatch.interp_diff_path,
            p.guard_precheck_calls,
            p.guard_precheck_skips,
            p.delta_reconstructed,
            p.delta_reconstruct_vars_decoded,
            p.delta_reconstruct_vars_total,
        );
    }

    /// Test accessor: hybrid routing counters as
    /// `(routed, mismatch_fallback, projection_declined, native_dispatched,
    /// native_matched, native_declined, native_errors)`.
    #[must_use]
    pub fn hybrid_dispatch_stats_for_testing(&self) -> (u64, u64, u64, u64, u64, u64, u64) {
        let s = &self.hybrid_dispatch.stats;
        (
            s.routed,
            s.mismatch_fallback,
            s.projection_declined,
            s.native_dispatched,
            s.native_matched,
            s.native_declined,
            s.native_errors,
        )
    }

    /// Test accessor (WP-29 lever 1 / WP-34): the enabling pre-check's
    /// `(calls, skips)`. Both are counted unconditionally, so the accessor is
    /// usable without `TY_HYBRID_PERF_DEBUG`.
    #[must_use]
    pub fn hybrid_guard_precheck_counters_for_testing(&self) -> (u64, u64) {
        let p = &self.hybrid_dispatch.perf;
        (p.guard_precheck_calls, p.guard_precheck_skips)
    }

    /// Test accessor (WP-34 lever 2): the batch consumer's
    /// `(successors, lazy_vars_scanned, lazy_vars_total)`.
    #[must_use]
    pub fn hybrid_consume_lazy_counters_for_testing(&self) -> (u64, u64, u64) {
        let p = &self.hybrid_dispatch.perf;
        (
            p.consume_succ,
            p.consume_lazy_vars_scanned,
            p.consume_lazy_vars_total,
        )
    }

    /// Test accessor (WP-21): typed shape-guard declines — recoverable
    /// per-parent "not applicable" outcomes kept out of `native_errors`.
    #[must_use]
    pub fn hybrid_guard_declined_for_testing(&self) -> u64 {
        self.hybrid_dispatch.stats.native_guard_declined
    }

    /// Test accessor (WP-14): authoritative-dispatch counters as
    /// `(authoritative_actions, authoritative_dispatched, sampled_checks,
    /// sampled_mismatches, burn_in_pending, failback)`.
    #[must_use]
    pub fn hybrid_authoritative_stats_for_testing(&self) -> (u64, u64, u64, u64, u64, bool) {
        let s = &self.hybrid_dispatch.stats;
        let m = &self.hybrid_dispatch.machine;
        (
            m.authoritative_action_count() as u64,
            s.authoritative_dispatched,
            s.sampled_checks,
            s.sampled_mismatches,
            m.burn_in_pending_count() as u64,
            m.failback(),
        )
    }

    /// Test accessor: number of hybrid-eligible detected actions.
    #[must_use]
    pub fn hybrid_eligible_action_count_for_testing(&self) -> usize {
        self.hybrid_dispatch.eligible.iter().filter(|e| **e).count()
    }

    /// Test accessor: whether a hybrid-layout native action cache was built
    /// with at least one compiled action (item 4 M0-G1).
    #[must_use]
    pub fn hybrid_native_cache_ready_for_testing(&self) -> bool {
        self.trust_cg_hybrid_action_dispatch_ready()
    }
}

/// WP-14 burn-in state-machine unit tests: consecutive-clean counting resets
/// on any non-match, the flip happens exactly at N, sampling is a
/// deterministic pure function of the hash, and the fail-back is permanent
/// and global. The machine is pure bookkeeping, so these pin the contract
#[cfg(test)]
mod router_policy_tests {
    use super::{
        router_request_from, router_skip_rate_admitted, router_timing_admitted, RouterRequest,
    };
    use std::ffi::OsStr;

    fn request(raw: Option<&str>, auto: bool) -> RouterRequest {
        router_request_from(raw.map(OsStr::new), auto)
    }

    #[test]
    fn router_request_truth_table() {
        assert_eq!(request(None, false), RouterRequest::Disabled);
        assert_eq!(request(None, true), RouterRequest::Auto);
        assert_eq!(request(Some("1"), false), RouterRequest::Forced);
        assert_eq!(request(Some(" 1 "), true), RouterRequest::Forced);
        for raw in ["0", "false", "true", "yes", ""] {
            assert_eq!(request(Some(raw), true), RouterRequest::Disabled, "{raw:?}");
        }
    }

    #[test]
    fn router_skip_rate_boundary_is_exact() {
        assert!(!router_skip_rate_admitted(0, 0));
        assert!(!router_skip_rate_admitted(79, 100));
        assert!(router_skip_rate_admitted(80, 100));
        assert!(router_skip_rate_admitted(4, 5));
    }

    #[test]
    fn router_timing_boundary_is_exact() {
        assert!(!router_timing_admitted(0, 0));
        assert!(router_timing_admitted(60, 100));
        assert!(!router_timing_admitted(61, 100));
        assert!(!router_timing_admitted(100, 100));
        assert!(!router_timing_admitted(u128::MAX, u128::MAX));
    }
}

/// Soundness boundaries for the enabling-guard extractor. An extractor false
/// positive can silently drop successors, so action/scope constructs must
/// fail closed while ordinary state predicates remain useful.
#[cfg(test)]
mod guard_extract_tests {
    use super::{
        extract_action_state_guard, extract_router_prefix_state_guard, guard_proves_disabled,
    };
    use crate::eval::EvalCtx;
    use crate::state::ArrayState;
    use crate::Value;
    use rustc_hash::FxHashMap;
    use std::sync::Arc;

    const SHAPES: &str = r#"
---- MODULE guardshapes ----
EXTENDS Naturals, Sequences
VARIABLES x, y

Bumped(a) == a + 1

D == {}
S == {}

SetToSeq(s) == /\ x = 99
               /\ x' = x + 1
               /\ y' = y

Primed == /\ x' = x + 1
          /\ y' = y

EnabledLead == /\ ENABLED (x' = 1)
               /\ x' = x + 1
               /\ y' = y

ChooseLead == /\ x = (CHOOSE v \in 0..3 : v > 1)
              /\ x' = x + 1
              /\ y' = y

OpaqueCall == /\ Oracle(x) > 2
              /\ x' = x + 1
              /\ y' = y

StringSubSeqGuard == /\ SubSeq("router_guard_fresh_string", 2, 8) = "outer_g"
                     /\ x' = x + 1
                     /\ y' = y

UnboundedQuant == /\ \E v : v = x
                  /\ x' = x + 1
                  /\ y' = y

LetBound == LET k == x + 1 IN
              /\ k > 3
              /\ x' = k
              /\ y' = y

NonConjunct == x' = x + 1

LiteralTrue == /\ TRUE
               /\ x' = x + 1
               /\ y' = y

TrueGuard == /\ x >= 0
             /\ x' = x + 1
             /\ y' = y

FalseGuard == /\ x = 99
              /\ x' = x + 1
              /\ y' = y

DefinedCall == /\ Bumped(x) > 2
               /\ x' = x + 1
               /\ y' = y

LateFalseAfterOpaque == /\ x > 0
                        /\ Oracle(x) > 2
                        /\ x = 99
                        /\ x' = x + 1
                        /\ y' = y

QuantifiedPrefix == \E k \in 1..2:
                       /\ x = 99
                       /\ x' = k
                       /\ y' = y

PrimedDomain == \E k \in {x'}:
                    /\ x = 99
                    /\ x' = k
                    /\ y' = y

LetScopedDomain == LET D == {1} IN
                      \E k \in D:
                         /\ x = 99
                         /\ x' = k
                         /\ y' = y

FormalScopedDomain(S) == \E k \in S:
                           /\ x = 99
                           /\ x' = k
                           /\ y' = y

FormalScopedDomainCall == FormalScopedDomain({1})

BuiltinOverridePrefix == SetToSeq({1})

Init == x = 1 /\ y = 0
Next == Primed \/ EnabledLead \/ ChooseLead \/ OpaqueCall \/ UnboundedQuant
          \/ LetBound \/ NonConjunct \/ LiteralTrue \/ TrueGuard \/ FalseGuard
          \/ DefinedCall
====
"#;

    type OpDefs = FxHashMap<String, tla_core::ast::OperatorDef>;

    fn setup() -> (EvalCtx, OpDefs) {
        let tree = tla_core::parse_to_syntax_tree(SHAPES);
        let lowered = tla_core::lower(tla_core::FileId(0), &tree);
        assert!(
            lowered.errors.is_empty(),
            "lowering errors: {:?}",
            lowered.errors
        );
        let module = lowered.module.expect("module lowering produced None");
        let mut ctx = EvalCtx::new();
        ctx.load_module(&module);
        let mut op_defs = OpDefs::default();
        for unit in &module.units {
            match &unit.node {
                tla_core::ast::Unit::Variable(names) => {
                    for name in names {
                        ctx.register_var(Arc::from(name.node.as_str()));
                    }
                }
                tla_core::ast::Unit::Operator(def) => {
                    op_defs.insert(def.name.node.clone(), def.clone());
                }
                _ => {}
            }
        }
        (ctx, op_defs)
    }

    fn guard_for(name: &str) -> Option<tla_core::Spanned<tla_core::ast::Expr>> {
        let (ctx, op_defs) = setup();
        let def = op_defs
            .get(name)
            .unwrap_or_else(|| panic!("no operator {name}"));
        let mut bound = Vec::new();
        extract_action_state_guard(&ctx, &op_defs, &def.body, &mut bound, 0)
    }

    fn parent() -> ArrayState {
        ArrayState::from_values(vec![Value::int(1), Value::int(0)])
    }

    fn precheck_says_disabled(name: &str) -> bool {
        let (mut ctx, op_defs) = setup();
        let def = op_defs
            .get(name)
            .unwrap_or_else(|| panic!("no operator {name}"));
        let mut bound = Vec::new();
        let Some(guard) = extract_action_state_guard(&ctx, &op_defs, &def.body, &mut bound, 0)
        else {
            return false;
        };
        guard_proves_disabled(&mut ctx, &guard, &parent())
    }

    fn router_precheck_says_disabled(name: &str) -> bool {
        let (mut ctx, op_defs) = setup();
        let def = op_defs
            .get(name)
            .unwrap_or_else(|| panic!("no operator {name}"));
        let mut bound = Vec::new();
        let Some(guard) =
            extract_router_prefix_state_guard(&ctx, &op_defs, &def.body, &mut bound, 0)
        else {
            return false;
        };
        guard_proves_disabled(&mut ctx, &guard, &parent())
    }

    fn router_guard_for(name: &str) -> Option<tla_core::Spanned<tla_core::ast::Expr>> {
        let (ctx, op_defs) = setup();
        let def = op_defs
            .get(name)
            .unwrap_or_else(|| panic!("no operator {name}"));
        let mut bound = Vec::new();
        extract_router_prefix_state_guard(&ctx, &op_defs, &def.body, &mut bound, 0)
    }

    #[test]
    fn action_or_scope_shapes_never_skip() {
        for name in [
            "Primed",
            "EnabledLead",
            "ChooseLead",
            "OpaqueCall",
            "StringSubSeqGuard",
            "UnboundedQuant",
            "LetBound",
            "NonConjunct",
            "LiteralTrue",
        ] {
            assert!(
                guard_for(name).is_none(),
                "{name} unexpectedly yielded a guard"
            );
            assert!(
                !precheck_says_disabled(name),
                "{name} was unsafely declared disabled"
            );
        }
    }

    #[test]
    fn accepted_guards_skip_only_on_definite_false() {
        for (name, expected_disabled) in [
            ("DefinedCall", true),
            ("TrueGuard", false),
            ("FalseGuard", true),
        ] {
            assert!(guard_for(name).is_some(), "{name} should yield a guard");
            assert_eq!(precheck_says_disabled(name), expected_disabled, "{name}");
        }
    }

    #[test]
    fn router_prefix_does_not_skip_past_an_earlier_expression() {
        // The broad guard can ignore the opaque middle conjunct and combine
        // the later `x = 99`. The router must retain only the leading `x > 0`,
        // which is true in `parent`, so canonical enumeration still reaches
        // (and remains authoritative for) the opaque call/error.
        assert!(!router_precheck_says_disabled("LateFalseAfterOpaque"));
        assert!(router_precheck_says_disabled("QuantifiedPrefix"));

        // Runtime replaces this same-named TLA definition with the SetToSeq
        // builtin. A guard extracted from the ignored body would therefore
        // not be implied by the expression that canonical evaluation sees.
        let (ctx, op_defs) = setup();
        let set_to_seq = op_defs.get("SetToSeq").expect("operator exists");
        assert!(crate::eval::should_prefer_builtin_override(
            "SetToSeq", set_to_seq, 1, &ctx,
        ));
        assert!(router_guard_for("BuiltinOverridePrefix").is_none());

        // A retained quantifier domain executes before the body. Primed (and
        // likewise effectful) domains are not current-state prechecks.
        assert!(router_guard_for("PrimedDomain").is_none());

        // Retaining either domain outside its lexical scope would resolve the
        // same-named global empty set and manufacture a false guard.
        assert!(router_guard_for("LetScopedDomain").is_none());
        assert!(router_guard_for("FormalScopedDomainCall").is_none());

        // Characterize why production must not use the broad extractor for
        // an interpreter-skipping decision: it reaches past Oracle and finds
        // the later false conjunct, suppressing canonical error semantics.
        assert!(precheck_says_disabled("LateFalseAfterOpaque"));
    }
}

/// directly — no model checker, no env.
#[cfg(test)]
mod authoritative_machine_tests {
    use super::{HybridAuthoritativeMachine, HybridInstanceMode};

    const KEYS: u64 = 0xfeed;
    /// A sample hash that is NOT ≡ 0 (mod 2): post-flip instances using it
    /// dispatch authoritatively under `sample_k = 2`.
    const UNSAMPLED: u64 = 3;
    /// A sample hash that IS ≡ 0 (mod 2): post-flip instances using it are
    /// sampled back into the full differential.
    const SAMPLED: u64 = 4;

    fn shadow_clean(m: &mut HybridAuthoritativeMachine, action: usize) {
        assert_eq!(
            m.decide_mode(action, KEYS, UNSAMPLED),
            HybridInstanceMode::Shadow
        );
        assert!(!m.record_result(action, HybridInstanceMode::Shadow, true, false));
    }

    #[test]
    fn flip_happens_exactly_at_n() {
        let mut m = HybridAuthoritativeMachine::new(4, 2, 1);
        for _ in 0..3 {
            shadow_clean(&mut m, 0);
        }
        // Three cleans: still shadow, still pending.
        assert_eq!(m.authoritative_action_count(), 0);
        assert_eq!(m.burn_in_pending_count(), 1);
        // The 4th clean flips — and the NEXT decide is post-flip.
        shadow_clean(&mut m, 0);
        assert_eq!(m.authoritative_action_count(), 1);
        assert_eq!(m.burn_in_pending_count(), 0);
        assert_eq!(
            m.decide_mode(0, KEYS, UNSAMPLED),
            HybridInstanceMode::Authoritative
        );
    }

    #[test]
    fn consecutive_clean_resets_on_any_non_match() {
        let mut m = HybridAuthoritativeMachine::new(4, 2, 1);
        for _ in 0..3 {
            shadow_clean(&mut m, 0);
        }
        // A dirty differential (no action flipped yet): reset, no fail-back.
        assert_eq!(
            m.decide_mode(0, KEYS, UNSAMPLED),
            HybridInstanceMode::Shadow
        );
        assert!(!m.record_result(0, HybridInstanceMode::Shadow, false, true));
        assert!(!m.failback());
        // Three more cleans do NOT flip (count restarted at 0)…
        for _ in 0..3 {
            shadow_clean(&mut m, 0);
        }
        assert_eq!(m.authoritative_action_count(), 0);
        // …the 4th consecutive clean does.
        shadow_clean(&mut m, 0);
        assert_eq!(m.authoritative_action_count(), 1);
    }

    #[test]
    fn incomplete_differential_also_resets_but_never_trips() {
        let mut m = HybridAuthoritativeMachine::new(2, 2, 1);
        shadow_clean(&mut m, 0);
        // clean=false, dirty=false: a projection/reconstruct decline.
        assert!(!m.record_result(0, HybridInstanceMode::Shadow, false, false));
        assert!(!m.failback());
        shadow_clean(&mut m, 0);
        assert_eq!(m.authoritative_action_count(), 0);
        shadow_clean(&mut m, 0);
        assert_eq!(m.authoritative_action_count(), 1);
    }

    #[test]
    fn sampling_is_a_deterministic_function_of_the_hash() {
        let mut m = HybridAuthoritativeMachine::new(1, 2, 1);
        shadow_clean(&mut m, 0);
        assert_eq!(m.authoritative_action_count(), 1);
        // Same hash, same verdict, every time — no hidden mod-counter.
        for _ in 0..5 {
            assert_eq!(m.decide_mode(0, KEYS, SAMPLED), HybridInstanceMode::Sampled);
            assert!(!m.record_result(0, HybridInstanceMode::Sampled, true, false));
            assert_eq!(
                m.decide_mode(0, KEYS, UNSAMPLED),
                HybridInstanceMode::Authoritative
            );
        }
        // sample_k = 0: never sample.
        let mut never = HybridAuthoritativeMachine::new(1, 0, 1);
        shadow_clean(&mut never, 0);
        assert_eq!(
            never.decide_mode(0, KEYS, SAMPLED),
            HybridInstanceMode::Authoritative
        );
        // sample_k = 1: always sample (native never actually skips).
        let mut always = HybridAuthoritativeMachine::new(1, 1, 1);
        shadow_clean(&mut always, 0);
        assert_eq!(
            always.decide_mode(0, KEYS, UNSAMPLED),
            HybridInstanceMode::Sampled
        );
    }

    #[test]
    fn sampled_mismatch_trips_permanent_global_failback() {
        let mut m = HybridAuthoritativeMachine::new(1, 2, 2);
        shadow_clean(&mut m, 0);
        shadow_clean(&mut m, 1);
        assert_eq!(m.authoritative_action_count(), 2);
        // Action 0's sampled differential diverges: permanent, global.
        assert_eq!(m.decide_mode(0, KEYS, SAMPLED), HybridInstanceMode::Sampled);
        assert!(m.record_result(0, HybridInstanceMode::Sampled, false, true));
        assert!(m.failback());
        assert_eq!(m.authoritative_action_count(), 0);
        // Every action — including the innocent action 1 — is shadow forever.
        assert_eq!(
            m.decide_mode(1, KEYS, UNSAMPLED),
            HybridInstanceMode::Shadow
        );
        // No amount of clean evidence re-flips after fail-back…
        for _ in 0..10 {
            assert!(!m.record_result(1, HybridInstanceMode::Shadow, true, false));
            assert_eq!(
                m.decide_mode(1, KEYS, UNSAMPLED),
                HybridInstanceMode::Shadow
            );
        }
        assert_eq!(m.authoritative_action_count(), 0);
        // …and a second trip reports "already tripped", not "newly tripped".
        assert!(!m.record_result(1, HybridInstanceMode::Sampled, false, true));
    }

    #[test]
    fn shadow_divergence_after_any_flip_trips_globally() {
        let mut m = HybridAuthoritativeMachine::new(1, 2, 2);
        shadow_clean(&mut m, 0);
        assert_eq!(m.authoritative_action_count(), 1);
        // Action 1 is still burning in; its SEMANTIC divergence invalidates
        // trust in the shared machinery — global fail-back.
        assert_eq!(
            m.decide_mode(1, KEYS, UNSAMPLED),
            HybridInstanceMode::Shadow
        );
        assert!(m.record_result(1, HybridInstanceMode::Shadow, false, true));
        assert!(m.failback());
        assert_eq!(m.authoritative_action_count(), 0);
    }

    #[test]
    fn keys_hash_drift_restarts_burn_in() {
        let mut m = HybridAuthoritativeMachine::new(1, 2, 1);
        shadow_clean(&mut m, 0);
        assert_eq!(m.authoritative_action_count(), 1);
        // The compiled key set changed: evidence void, back to shadow.
        assert_eq!(
            m.decide_mode(0, KEYS + 1, UNSAMPLED),
            HybridInstanceMode::Shadow
        );
        assert_eq!(m.authoritative_action_count(), 0);
        assert!(!m.record_result(0, HybridInstanceMode::Shadow, true, false));
        // One clean against the NEW key set re-flips (N = 1).
        assert_eq!(
            m.decide_mode(0, KEYS + 1, UNSAMPLED),
            HybridInstanceMode::Authoritative
        );
    }

    #[test]
    fn burn_in_is_independent_per_action() {
        let mut m = HybridAuthoritativeMachine::new(2, 2, 2);
        shadow_clean(&mut m, 0);
        shadow_clean(&mut m, 0);
        assert_eq!(m.authoritative_action_count(), 1);
        // Action 1 has one clean — not flipped, still pending.
        shadow_clean(&mut m, 1);
        assert_eq!(m.authoritative_action_count(), 1);
        assert_eq!(m.burn_in_pending_count(), 1);
        assert_eq!(
            m.decide_mode(0, KEYS, UNSAMPLED),
            HybridInstanceMode::Authoritative
        );
        assert_eq!(
            m.decide_mode(1, KEYS, UNSAMPLED),
            HybridInstanceMode::Shadow
        );
    }
}
