// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Site-ranked Value lifecycle churn instrumentation (`TY_CHURN_STATS=1`).
//!
//! The validated btree production profile attributes ~18% of interpreter CPU to
//! Value lifecycle churn: `Value::clone` (~4.3%), `drop_in_place`/`Arc::drop_slow`
//! (~7.8%), and mimalloc malloc/free (~6.3%). This module ranks WHERE that churn
//! comes from: named counters at the candidate hot sites (state-var reads,
//! binding-chain hits, function-apply clone-outs, temporary set/tuple
//! materializations, operator-apply argument buffers, ...).
//!
//! Mirrors the proven env-gated static-counter pattern (`feature_flag!` /
//! `TY_PROFILE_EVAL`): counters are only incremented when `TY_CHURN_STATS` is
//! set, and the guard is a cached `OnceLock<bool>` load + predictable branch,
//! so the instrumentation is effectively free when off.
//!
//! Counters are `Relaxed` atomics — the measurement workflow is single-threaded
//! (`--workers 1`), and cross-site ordering is irrelevant for ranking.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;

/// Named churn sites. Keep in sync with `SITE_NAMES`.
#[derive(Clone, Copy)]
#[repr(usize)]
pub enum ChurnSite {
    // --- Clone-out sites (Value::clone; heap variants = Arc refcount churn) ---
    /// StateEnvRef::get_value — every state-variable read materializes an owned Value.
    StateVarRead = 0,
    /// ... of which the slot held a heap-backed (compound) value → Arc clone.
    StateVarReadHeap,
    /// Binding-chain lookup hit returning a heap-backed value (quantifier/LET/param).
    BindingChainHitHeap,
    /// Function-application result clone-out (Func/IntFunc/Seq/Tuple/Record apply).
    FuncApplyResult,
    /// ... of which the result value was heap-backed.
    FuncApplyResultHeap,
    /// Eager per-parameter Value clone when binding operator/closure arguments.
    OpApplyArgClone,
    /// ... of which heap-backed.
    OpApplyArgCloneHeap,
    /// TIR Const node clone-out of a heap-backed constant value.
    TirConstCloneHeap,
    /// TIR hoist-cache hit clone-out.
    HoistCacheHit,
    /// Zero-arg operator cache hit clone-out.
    ZeroArgCacheHit,
    /// N-ary operator result cache hit clone-out.
    NaryCacheHit,
    /// CHOOSE cache (shallow or deep) hit clone-out.
    ChooseCacheHit,
    /// Precomputed-constant clone-out (eval_ident tiers).
    PrecomputedConstClone,

    // --- Temporary-materialization sites (alloc + subsequent drop) ---
    /// New Value::Set built via Value::from_sorted_set (non-empty; Arc<SortedSet> alloc).
    SetValueBuild,
    /// SortedSet::without — S \ {x} singleton subtraction materialization.
    SetWithout,
    /// SortedSet::difference — general S \ T materialization.
    SetDifference,
    /// SortedSet::union merge materialization.
    SetUnion,
    /// SortedSet::intersection materialization.
    SetIntersection,
    /// intern_set_array probes (small-set dedup table).
    SetInternProbe,
    /// ... of which hits (alloc avoided / immediately dropped fresh array).
    SetInternHit,
    /// Set enumeration literal builds ({a, b, ...} incl. singletons like {x}).
    SetEnumBuild,
    /// Tuple/Seq literal builds (<<...>> and multi-arg f[a, b] key tuples).
    TupleBuild,
    /// Record literal builds.
    RecordBuild,
    /// Eager function definition builds ([x \in S |-> e]).
    FuncDefBuild,
    /// FuncValue::except calls (EXCEPT update; COW/overlay may avoid deep clone).
    FuncExcept,
    /// Temporary Vec<Value> argument buffers for TIR operator application.
    OpApplyArgsVec,

    // --- Volume / structural counters (context for the ratios) ---
    /// User-operator / closure applications with pre-evaluated argument values.
    OpApplyWithValues,
    /// Quantifier/CHOOSE loop iterations (elements pushed into the binding).
    QuantIterations,
    /// Binding-chain node pushes served by the BFS eval arena (no heap alloc).
    ChainConsArena,
    /// Binding-chain node pushes that heap-allocate an Arc node.
    ChainConsHeap,

    // --- Tuple-key consumer split (diagnostic; virtual-tuple elimination) ---
    /// TIR FuncApply `f[<<a,b>>]` answered by the dense-2D index (no build).
    TupleKeyApplyTirDenseHit,
    /// TIR FuncApply tuple-literal subscript materialized (fall-through).
    TupleKeyApplyTirBuild,
    /// ... of which the base was a sparse (non-dense-2D) FuncValue.
    TupleKeyApplyTirSparse,
    /// AST FuncApply with a tuple-literal subscript (materialized).
    TupleKeyApplyAstBuild,
    /// TIR EXCEPT `![a,b]` path-key tuple materialized.
    TupleKeyExceptTirBuild,
    /// AST EXCEPT `![a,b]` path-key tuple materialized.
    TupleKeyExceptAstBuild,
    /// Membership `<<a,b>> \in S` answered component-wise (no build).
    TupleKeyInFused,
    /// Membership `<<a,b>> \in S` fell through to a materialized candidate.
    TupleKeyInBuild,
    /// Tuple literal built by the AST evaluator (subset of TupleBuild).
    TupleBuildAst,
    /// TIR UNCHANGED <<..>> generic path evaluated the tuple literal.
    TupleKeyUnchangedTirBuild,
    /// TIR Cmp Eq/Neq with a tuple-literal operand (x' = <<..>> and friends).
    TupleKeyCmpTirBuild,

    // --- Caller split for StateVarRead (diagnostic sub-counters) ---
    /// TIR Name(StateVar) arm reads.
    StateVarReadTirName,
    /// AST eval_ident state_env reads (resolve.rs).
    StateVarReadAstIdent,
    /// AST Expr::StateVar reads (eval_state_var_lookup.rs).
    StateVarReadAstStateVar,
    /// Primed reads (eval_prime + TIR prime).
    StateVarReadPrime,
    /// UNCHANGED comparisons (dep-tracking path only).
    StateVarReadUnchanged,
    /// ctx.lookup()/env snapshot reads (eval_ctx_state, eval_ctx_ops).
    StateVarReadCtxLookup,
    /// State-var reads ELIDED by the borrowed fast paths (no clone, no drop).
    StateVarReadElided,

    /// Number of sites (must be last).
    Count,
}

const NUM_SITES: usize = ChurnSite::Count as usize;

static SITE_NAMES: [&str; NUM_SITES] = [
    "clone: state_var_read (StateEnvRef::get_value)",
    "clone:   ... heap-backed (Arc bump + later drop)",
    "clone: binding_chain_hit heap-backed",
    "clone: func_apply result clone-out",
    "clone:   ... func-apply result heap-backed",
    "clone: op_apply eager arg->binding clone",
    "clone:   ... op-apply arg heap-backed",
    "clone: TIR Const heap-backed clone-out",
    "clone: TIR hoist-cache hit clone-out",
    "clone: zero-arg op cache hit clone-out",
    "clone: n-ary op cache hit clone-out",
    "clone: CHOOSE cache hit clone-out",
    "clone: precomputed-constant clone-out",
    "alloc: Value::Set build (Arc<SortedSet>)",
    "alloc: SortedSet::without (S \\ {x})",
    "alloc: SortedSet::difference (S \\ T)",
    "alloc: SortedSet::union",
    "alloc: SortedSet::intersection",
    "alloc: set intern probe (small sets)",
    "alloc:   ... intern hit (fresh array dropped)",
    "alloc: set enum literal build {..}",
    "alloc: tuple/seq literal build <<..>>",
    "alloc: record literal build",
    "alloc: eager func-def build [x \\in S |-> e]",
    "alloc: FuncValue::except (EXCEPT update)",
    "alloc: TIR op-apply arg Vec<Value>",
    "vol  : op applies (with values)",
    "vol  : quantifier/CHOOSE iterations",
    "vol  : binding cons (arena)",
    "vol  : binding cons (heap Arc)",
    "tkey : TIR f[<<a,b>>] dense-2D hit (no build)",
    "tkey : TIR f[<<a,b>>] materialized build",
    "tkey :   ... sparse FuncValue base",
    "tkey : AST f[<<a,b>>] materialized build",
    "tkey : TIR EXCEPT ![a,b] key build",
    "tkey : AST EXCEPT ![a,b] key build",
    "tkey : <<a,b>> \\in S fused (no build)",
    "tkey : <<a,b>> \\in S materialized build",
    "tkey : tuple builds via AST eval (subset of <<..>>)",
    "tkey : TIR UNCHANGED <<..>> generic-path build",
    "tkey : TIR Cmp Eq/Neq tuple-literal operand build",
    "svar : state-var read via TIR Name(StateVar)",
    "svar : state-var read via AST eval_ident",
    "svar : state-var read via AST Expr::StateVar",
    "svar : state-var read via primed lookup",
    "svar : state-var read via UNCHANGED (dep-tracked)",
    "svar : state-var read via ctx lookup/env snapshot",
    "svar : state-var read ELIDED (borrowed, no clone)",
];

// `const` block repeat-expr initializer: AtomicU64 is not Copy, so use the
// inline-const array form (stable since 1.79).
static COUNTERS: [AtomicU64; NUM_SITES] = [const { AtomicU64::new(0) }; NUM_SITES];

/// Whether churn-stat collection is enabled (TY_CHURN_STATS env var, cached).
#[inline(always)]
pub fn churn_stats_enabled() -> bool {
    static FLAG: OnceLock<bool> = OnceLock::new();
    *FLAG.get_or_init(|| {
        std::env::var("TY_CHURN_STATS").map_or(false, |v| !v.is_empty() && v != "0")
    })
}

/// Count one event at `site`. No-op (cached-flag branch) when disabled.
#[inline(always)]
pub fn churn_count(site: ChurnSite) {
    if churn_stats_enabled() {
        COUNTERS[site as usize].fetch_add(1, Ordering::Relaxed);
    }
}

/// Count one event at `site`, plus `heap_site` when `value` is heap-backed
/// (anything but Bool/SmallInt — i.e. clone = Arc refcount churn, drop may free).
#[inline(always)]
pub fn churn_count_value(site: ChurnSite, heap_site: ChurnSite, value: &crate::Value) {
    if churn_stats_enabled() {
        COUNTERS[site as usize].fetch_add(1, Ordering::Relaxed);
        if !matches!(value, crate::Value::Bool(_) | crate::Value::SmallInt(_)) {
            COUNTERS[heap_site as usize].fetch_add(1, Ordering::Relaxed);
        }
    }
}

/// Count one event at `site` only when `value` is heap-backed.
#[inline(always)]
pub fn churn_count_if_heap(site: ChurnSite, value: &crate::Value) {
    if churn_stats_enabled() && !matches!(value, crate::Value::Bool(_) | crate::Value::SmallInt(_))
    {
        COUNTERS[site as usize].fetch_add(1, Ordering::Relaxed);
    }
}

/// Print the ranked site table to stderr (no-op when disabled or all-zero).
/// Counters are drained so repeated runs in one process don't double-count.
pub fn print_churn_stats() {
    if !churn_stats_enabled() {
        return;
    }
    let mut rows: Vec<(u64, &'static str)> = COUNTERS
        .iter()
        .zip(SITE_NAMES.iter())
        .map(|(c, name)| (c.swap(0, Ordering::Relaxed), *name))
        .filter(|(count, _)| *count > 0)
        .collect();
    if rows.is_empty() {
        return;
    }
    rows.sort_by(|a, b| b.0.cmp(&a.0));
    eprintln!("\n=== Value Churn Stats (TY_CHURN_STATS) — ranked by count ===");
    for (count, name) in rows {
        eprintln!("  {count:>12}  {name}");
    }
    eprintln!("=== end churn stats ===");
}
