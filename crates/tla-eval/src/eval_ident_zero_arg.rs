// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Unified zero-arg operator caching for identifier evaluation.
//!
//! Consolidates the 10-step caching pattern that was duplicated across three
//! call sites in `eval_ident.rs`: fast-path inline, `eval_zero_arg_local_op`,
//! and `eval_ident_shared_zero_arg_op`.
//!
//! Part of #2669.
//! Fix #2462: ZERO_ARG_OP_CONST_CACHE removed. All entries go through
//! ZERO_ARG_OP_CACHE with dep validation via op_cache_entry_valid().

#[cfg(debug_assertions)]
use super::debug_zero_cache;
use tla_value::Rp;
use super::{
    build_lazy_func_from_ctx, current_state_lookup_mode, eval, eval_builtin,
    materialize_setpred_to_vec, no_local_ops_cache, op_cache_entry_valid, propagate_cached_deps,
    should_prefer_builtin_override, zero_arg_insert, zero_arg_lookup, CachedOpResult, EvalCtx,
    EvalError, EvalResult, Expr, LazyDomain, OpDepGuard, OpEvalDeps, Span, StateLookupMode, Value,
};
use crate::cache::zero_arg_cache::{
    record_zero_arg_canonical_hit, record_zero_arg_constant_fallback_hit, record_zero_arg_miss,
    record_zero_arg_primary_hit, zero_arg_cache_key,
};
use std::sync::Arc;
use tla_core::ast::{BoundVar, OperatorDef};
use tla_core::expr_mentions_name_spanned_v;
use tla_core::name_intern::intern_name;
use tla_core::Spanned;

/// Unified zero-arg operator caching. Consolidates the 10-step caching pattern
/// from three call sites in `eval_ident.rs`.
///
/// Fix #2462: All entries (both constant and state-dependent) are stored in
/// ZERO_ARG_OP_CACHE with dep validation. The separate unvalidated
/// ZERO_ARG_OP_CONST_CACHE has been removed.
///
/// # Parameters
///
/// * `eval_ctx` — Context for evaluation (may be `outer_ctx` for shared ops).
/// * `propagate_ctx` — Context for dep propagation on cache hit (always the
///   original calling `ctx`). Also used for `current_state_lookup_mode` and
///   `should_prefer_builtin_override` checks.
/// * `name` — Operator name.
/// * `def` — Operator definition.
/// * `span` — Source location for error reporting.
/// * `env_pre_check` — `true` for local ops per Fix #100 (check env for
///   existing LazyFunc before constructing a new one).
pub(super) fn eval_zero_arg_cached(
    eval_ctx: &EvalCtx,
    propagate_ctx: &EvalCtx,
    name: &str,
    def: &Arc<OperatorDef>,
    span: Option<Span>,
    env_pre_check: bool,
) -> EvalResult<Value> {
    let action_eval_ctx;
    let eval_ctx = if eval_ctx.in_tlc_action_scope() && eval_ctx.tlc_action_context().is_none() {
        action_eval_ctx = {
            let mut ctx = eval_ctx.clone();
            ctx.install_outermost_tlc_action_context(def);
            ctx
        };
        &action_eval_ctx
    } else {
        eval_ctx
    };

    // Step 1: Builtin override preference check.
    if should_prefer_builtin_override(name, def, 0, propagate_ctx) {
        if let Some(result) = eval_builtin(eval_ctx, name, &[], span)? {
            return Ok(result);
        }
    }

    // Steps 2-5: Self-referential FuncDef detection + cache + construction.
    if let Expr::FuncDef(bounds, func_body) = &def.body.node {
        if expr_mentions_name_spanned_v(func_body, name) {
            return eval_self_referential_func(
                eval_ctx,
                propagate_ctx,
                name,
                def,
                bounds,
                func_body.as_ref(),
                span,
                env_pre_check,
            );
        }
    }

    // Steps 6-10: General zero-arg caching.
    eval_general_zero_arg(eval_ctx, propagate_ctx, name, def, span)
}

/// Handle self-referential FuncDef operators (e.g., `nat2node[i \in S] == ... nat2node[i-1] ...`).
///
/// These need special treatment: a `LazyFunc` is constructed with the domain and body,
/// and the cached LazyFunc preserves its memo across calls.
#[allow(clippy::too_many_arguments)]
fn eval_self_referential_func(
    eval_ctx: &EvalCtx,
    propagate_ctx: &EvalCtx,
    name: &str,
    def: &Arc<OperatorDef>,
    bounds: &[BoundVar],
    func_body: &Spanned<Expr>,
    span: Option<Span>,
    env_pre_check: bool,
) -> EvalResult<Value> {
    // Fix #100: For local ops, check env for existing LazyFunc first (shared memoization).
    if env_pre_check {
        if let Some(existing) = propagate_ctx.env.get(name) {
            if matches!(existing, Value::LazyFunc(_)) {
                return Ok(existing.clone());
            }
        }
    }

    // Check cache for existing LazyFunc with populated memo.
    // Fix #2462: All lookups go through ZERO_ARG_OP_CACHE with dep validation.
    // Part of #3100: Use zero_arg_lookup to probe both state and persistent partitions.
    if !no_local_ops_cache() {
        let is_next_state = current_state_lookup_mode(propagate_ctx) == StateLookupMode::Next;
        let def_loc = def.body.span.start;
        let name_id = intern_name(name);
        let cache_key = zero_arg_cache_key(eval_ctx, name_id, def_loc, is_next_state);

        if let Some(entry) = zero_arg_lookup(&cache_key, |entry| {
            matches!(entry.value, Value::LazyFunc(_)) && op_cache_entry_valid(eval_ctx, entry)
        }) {
            propagate_cached_deps(propagate_ctx, &entry.deps);
            return Ok(entry.value);
        }

        // Canonical key fallback for self-referential LazyFunc (same rationale as
        // Step 6b in eval_general_zero_arg — unstable local_ops_id from recursive ops).
        {
            let canonical_key = crate::cache::zero_arg_cache::zero_arg_canonical_key(
                eval_ctx.shared.id,
                name_id,
                def_loc,
                is_next_state,
            );
            if let Some(entry) =
                crate::cache::zero_arg_cache::zero_arg_canonical_lookup(&canonical_key)
            {
                if matches!(entry.value, Value::LazyFunc(_)) {
                    propagate_cached_deps(propagate_ctx, &entry.deps);
                    return Ok(entry.value);
                }
            }
        }

        // Part of #3109: During ENABLED scope, retry with flipped is_next_state
        // for constant LazyFunc entries (empty deps).
        // Part of #4027: Use shadow via in_enabled_scope_ctx for consistency.
        if crate::cache::lifecycle::in_enabled_scope_ctx(eval_ctx) {
            if let Some(entry) =
                crate::cache::zero_arg_cache::zero_arg_constant_fallback(&cache_key)
            {
                if matches!(entry.value, Value::LazyFunc(_)) {
                    propagate_cached_deps(propagate_ctx, &entry.deps);
                    return Ok(entry.value.clone());
                }
            }
        }
    }

    // Cache miss: construct LazyFunc.
    let domain_val = eval_ident_func_domain(eval_ctx, bounds, span)?;
    if !domain_val.is_set() {
        return Err(EvalError::type_error(
            "Set",
            &domain_val,
            Some(def.body.span),
        ));
    }
    let op_name = Arc::from(name);
    let lazy = build_lazy_func_from_ctx(
        eval_ctx,
        Some(Arc::clone(&op_name)),
        LazyDomain::General(Box::new(domain_val)),
        bounds,
        func_body.clone(),
    );
    let result = Value::LazyFunc(Rp::new(lazy));

    // Fix #4145: Self-referential LazyFuncs MUST NOT be placed in the persistent
    // partition. The LazyFunc captures state arrays via `snapshot_state_envs()` and
    // accumulates state-dependent results in its memo table. If persisted across
    // state boundaries, the stale captured state and populated memo cause incorrect
    // results (VoteProof: 4180 states instead of 6962).
    //
    // Set `instance_lazy_read = true` to route to the state partition, which is
    // cleared on every state boundary. This ensures each state gets a fresh
    // LazyFunc with empty memo and correct captured state. Unlike `inconsistent`,
    // `instance_lazy_read` still allows cache hits within the same state via
    // `op_cache_entry_valid()` (which does not check `instance_lazy_read`).
    //
    // Do NOT store under canonical key (canonical is for persistent/constant
    // entries only).
    //
    // Prior code used `OpEvalDeps::default()` (empty deps → persistent partition),
    // which was incorrect: "LazyFunc creation is pure" ignored that the LazyFunc's
    // memo and captured state are state-dependent at application time.
    if !no_local_ops_cache() {
        let is_next_state = current_state_lookup_mode(propagate_ctx) == StateLookupMode::Next;
        let def_loc = def.body.span.start;
        let name_id = intern_name(name);
        let cache_key = zero_arg_cache_key(eval_ctx, name_id, def_loc, is_next_state);

        let deps = OpEvalDeps {
            instance_lazy_read: true,
            ..OpEvalDeps::default()
        };
        zero_arg_insert(
            cache_key,
            CachedOpResult {
                value: result.clone(),
                deps,
            },
        );
    }
    Ok(result)
}

/// General zero-arg operator caching (non-self-referential operators).
///
/// Steps 6-10 of the unified caching pattern: dep-validated cache scan,
/// cache-miss evaluation with dep tracking, SetPred materialization,
/// and cache store.
///
/// Fix #2462: ZERO_ARG_OP_CONST_CACHE removed. All entries (including those
/// with empty deps) go through ZERO_ARG_OP_CACHE with dep validation.
fn eval_general_zero_arg(
    eval_ctx: &EvalCtx,
    propagate_ctx: &EvalCtx,
    name: &str,
    def: &Arc<OperatorDef>,
    _span: Option<Span>,
) -> EvalResult<Value> {
    // Issue #284: TY_NO_LOCAL_OPS_CACHE disables caching to match TLC semantics.
    if no_local_ops_cache() {
        return eval(eval_ctx, &def.body);
    }

    let is_next_state = current_state_lookup_mode(propagate_ctx) == StateLookupMode::Next;
    let def_loc = def.body.span.start;
    let name_id = intern_name(name);
    // Part of #3097: Use shared helper for scope-discriminated key.
    let cache_key = zero_arg_cache_key(eval_ctx, name_id, def_loc, is_next_state);
    // Canonical key (Step 6b rationale below) — hoisted so both the merged
    // and the kill-switch probe paths use the identical key.
    let canonical_key = crate::cache::zero_arg_cache::zero_arg_canonical_key(
        eval_ctx.shared.id,
        name_id,
        def_loc,
        is_next_state,
    );
    // Fingerprint-keyed transition cache participation (implied-action
    // checking only — `transition_fp` is `Some` only while the checker has an
    // implied-action fingerprint scope installed). Memoizes state-level
    // zero-arg operators (refinement mappings like `token` / `pending`)
    // across transitions: the per-transition eval-scope clear wipes the state
    // partition, but the derived value of a state function depends only
    // on the state it reads, so it is keyed by that state's fingerprint and
    // revalidated by dep values on every hit.
    let transition_fp = implied_transition_fp(eval_ctx, is_next_state);
    if debug_implied_fp() {
        eprintln!(
            "[IMPLIED_FP] probe name={name} next={is_next_state} fp={transition_fp:?} pair={:?}",
            eval_ctx.runtime_state.implied_fp_pair.get()
        );
    }

    if crate::cache::zero_arg_cache::no_zero_arg_probe_merge() {
        // ---- Kill switch (TY_NO_ZERO_ARG_PROBE_MERGE=1): original probe
        // ---- sequence, one TLS access per probe. Kept verbatim for
        // ---- differential validation of the merged path below.
        // Step 6-7: Part of #3100 — probe both state and persistent partitions.
        if let Some(entry) =
            zero_arg_lookup(&cache_key, |entry| op_cache_entry_valid(eval_ctx, entry))
        {
            record_zero_arg_primary_hit();
            propagate_cached_deps(propagate_ctx, &entry.deps);
            return Ok(entry.value);
        }
        // Step 6a: Fingerprint-keyed transition cache probe.
        if let Some(fp) = transition_fp {
            let tkey = crate::cache::zero_arg_cache::zero_arg_transition_key(
                eval_ctx, name_id, def_loc, fp,
            );
            if let Some(entry) =
                crate::cache::zero_arg_cache::zero_arg_transition_lookup(&tkey, |entry| {
                    transition_entry_valid(eval_ctx, entry, is_next_state)
                })
            {
                record_zero_arg_primary_hit();
                propagate_cached_deps(
                    propagate_ctx,
                    &transition_deps_for_mode(&entry.deps, is_next_state),
                );
                return Ok(entry.value);
            }
        }
        // Step 6b: Canonical key fallback for constant operators.
        //
        // When `local_ops` contains recursive operators (e.g., RECURSIVE
        // PublicKeyOf in INSTANCE Nano), `compute_local_ops_scope_id` falls
        // back to Arc pointer identity, producing a different `local_ops_id`
        // each time `with_outer_resolution_scope()` is called. This makes
        // scope-discriminated keys unstable for shared operators accessed
        // through INSTANCE modules.
        //
        // Constant operators (empty deps) produce the same value regardless
        // of scope, so a scope-normalized "canonical" key (local_ops_id=0,
        // instance_subs_id=0) allows persistent entries to be found across
        // scope changes.
        if let Some(entry) = crate::cache::zero_arg_cache::zero_arg_canonical_lookup(&canonical_key)
        {
            record_zero_arg_canonical_hit();
            propagate_cached_deps(propagate_ctx, &entry.deps);
            return Ok(entry.value);
        }
    } else {
        // Lever 2 (#EWD998PCal): Steps 6/6a/6b merged into ONE TLS access.
        // Probe order, validators, and short-circuiting are identical to the
        // kill-switch branch above; only the `thread_local!` access count
        // changes (3 → 1). During implied-action checking this probe sequence
        // runs for every leaf refinement-operator evaluation of every checked
        // transition, so the per-access LocalKey overhead is hot-path cost.
        let tkey = transition_fp.map(|fp| {
            crate::cache::zero_arg_cache::zero_arg_transition_key(eval_ctx, name_id, def_loc, fp)
        });
        if let Some(hit) = crate::cache::zero_arg_cache::zero_arg_probe_all(
            &cache_key,
            tkey.as_ref(),
            &canonical_key,
            |entry| op_cache_entry_valid(eval_ctx, entry),
            |entry| transition_entry_valid(eval_ctx, entry, is_next_state),
        ) {
            use crate::cache::zero_arg_cache::ZeroArgProbeHit;
            match hit {
                ZeroArgProbeHit::Primary(entry) => {
                    record_zero_arg_primary_hit();
                    propagate_cached_deps(propagate_ctx, &entry.deps);
                    return Ok(entry.value);
                }
                ZeroArgProbeHit::Transition(entry) => {
                    record_zero_arg_primary_hit();
                    propagate_cached_deps(
                        propagate_ctx,
                        &transition_deps_for_mode(&entry.deps, is_next_state),
                    );
                    return Ok(entry.value);
                }
                ZeroArgProbeHit::Canonical(entry) => {
                    record_zero_arg_canonical_hit();
                    propagate_cached_deps(propagate_ctx, &entry.deps);
                    return Ok(entry.value);
                }
            }
        }
    }

    // Part of #3109: During ENABLED scope, retry with flipped is_next_state.
    // Part of #4027: Use shadow via in_enabled_scope_ctx for consistency.
    if crate::cache::lifecycle::in_enabled_scope_ctx(eval_ctx) {
        if let Some(entry) = crate::cache::zero_arg_cache::zero_arg_constant_fallback(&cache_key) {
            record_zero_arg_constant_fallback_hit();
            propagate_cached_deps(propagate_ctx, &entry.deps);
            return Ok(entry.value.clone());
        }
    }

    // Step 8: Cache miss — evaluate with dep tracking + SetPred materialization.
    debug_eprintln!(
        debug_zero_cache(),
        "[ZERO_CACHE] MISS: {} (next={})",
        name,
        is_next_state
    );
    let base_stack_len = eval_ctx.binding_depth;
    let guard = OpDepGuard::from_ctx(eval_ctx, base_stack_len);
    let val = eval(eval_ctx, &def.body)?;

    // Step 9: Materialize SetPred within dep scope (Fix #1894), except for the
    // direct RecordSet filter-chain fragment shared with Init bulk streaming.
    // Flattening that fragment here would erase the SetPred predicate/source
    // structure before Init extraction can stream it. Other SetPred values keep
    // the historical eager behavior required by callers such as FoldSet.
    let val = match &val {
        Value::SetPred(spv) if !spv.source().is_streamable_record_filter_domain() => {
            let elems = materialize_setpred_to_vec(eval_ctx, spv)?;
            Value::set(elems)
        }
        _ => val,
    };

    // Step 9b: Eagerly materialize small finite lazy set types (SetCup,
    // TupleSet) into flat Value::Set(SortedSet) for efficient membership checking.
    //
    // Constant operators like `Block == GenesisBlock \cup SendBlock \cup ...` produce
    // SetCup(RecordSet, SetCup(RecordSet, ...)) trees. Each membership check traverses
    // this tree, calling RecordSet::contains at every node. For MCNanoMedium, this
    // makes TypeInvariant checking ~2x slower than necessary.
    //
    // Materializing to a flat SortedSet converts tree-traversal membership (O(depth *
    // field_checks)) to binary search (O(log n * comparison_cost)). For a 74-element
    // Block set, this is ~6 comparisons vs ~5 RecordSet field-check chains.
    //
    // Only materialize when: (1) cardinality is known and <= 10000, (2) the value
    // is a lazy set type that benefits from materialization. Skip for Value::Set
    // (already flat), Value::RecordSet (init extraction needs filtered record
    // domains to stay lazy), Value::Subset (SUBSET S is exponential),
    // Value::FuncSet ([S -> T] is exponential), Value::BigUnion, Value::SeqSet
    // (may be infinite).
    let val = match &val {
        Value::SetCup(_)
        | Value::TupleSet(_)
        | Value::SetCap(_)
        | Value::SetDiff(_)
        | Value::KSubset(_) => {
            use num_traits::ToPrimitive;
            if let Some(len) = val.set_len() {
                if let Some(n) = len.to_u64() {
                    if n <= 10_000 {
                        if let Some(sorted) = val.to_sorted_set() {
                            Value::Set(Rp::new(sorted))
                        } else {
                            val
                        }
                    } else {
                        val
                    }
                } else {
                    val
                }
            } else {
                val
            }
        }
        _ => val,
    };

    // Step 9c: Implied-action transition-cache eligibility for lazy funcs.
    //
    // Refinement mappings like EWD998PCal's `pending == [n \in Node |-> ...]`
    // evaluate to a `LazyFunc` that captures the bound state arrays, which
    // makes the value ineligible for cross-state reuse (Fix #4145/#4158
    // rationale: stale captured arrays + memo). During implied-action checking
    // the term forces the function anyway (UNCHANGED / equality enumerate the
    // whole domain each transition), so eagerly materialize small finite
    // lazy funcs to a plain `Value::Func` INSIDE the dep-tracking scope (the
    // body's state reads land in `deps`, mirroring the Step 9 SetPred rule).
    // Fail-closed: any materialization error keeps the original lazy value
    // (the lazy path may legitimately never force the failing entry).
    //
    // Gated on the implied-action fingerprint scope, so all other evaluation
    // paths are untouched.
    let val = match (&val, transition_fp) {
        (Value::LazyFunc(f), Some(_))
            if (f.captured_state().is_some() || f.captured_next_state().is_some())
                && lazy_domain_is_small_finite(f.domain()) =>
        {
            match crate::helpers::materialize_lazy_func_to_func(eval_ctx, f) {
                Ok(materialized) => materialized,
                Err(_) => val,
            }
        }
        _ => val,
    };

    let mut deps = match guard.try_take_deps() {
        Some(mut deps) => {
            // Fix #2991: Strip local deps that are internal to this operator evaluation.
            // Internal locals (quantifier iteration variables, comprehension bindings) leak
            // into deps via Instance binding propagation, bypassing record_local_read's
            // depth filter. This causes false `inconsistent=true`, preventing caching.
            deps.strip_internal_locals(&eval_ctx.bindings, base_stack_len);
            record_zero_arg_miss(name, &deps);
            propagate_cached_deps(eval_ctx, &deps);
            deps
        }
        None => {
            return Err(EvalError::Internal {
                message: "dep tracking stack empty after zero-arg op eval".into(),
                span: Some(def.body.span),
            });
        }
    };

    // Part of #3964: Check if LazyFunc name is already set via read-only Arc
    // access before calling Arc::make_mut. This avoids a deep clone of the
    // LazyFuncValue when the name is already set (common case for cached results).
    let mut val = val;
    if let Value::LazyFunc(ref mut f) = val {
        if f.name().is_none() {
            Rp::make_mut(f).set_name_if_missing(Arc::from(name));
        }
    }
    let result = val;
    let state_capturing_set_pred = matches!(
        &result,
        Value::SetPred(value)
            if value.captured_state().is_some() || value.captured_next_state().is_some()
    );

    // Part of #4158: Deferred values that capture state MUST NOT be placed in
    // the persistent partition. LazyFunc and SetPred construction snapshot the
    // state arrays without reading them, so dependency tracking alone would
    // misclassify either wrapper as constant. Reusing it in a subsequent state
    // would evaluate against stale captured bindings.
    //
    // Examples: `F == [x \in Nat |-> x + state_var]` and
    // `S == {r \in RecordSet : r.a = state_var}`.
    //
    // Same pattern as Fix #4145 for self-referential functions and the shared
    // guard in eval_let/zero_arg_cache.rs for the LET cache path.
    if result.captures_state_environment() {
        deps.instance_lazy_read = true;
    }

    // Step 10: Cache store.
    // Part of #3100: zero_arg_insert routes to persistent or state partition by deps.
    // Part of #3109: Skip insertion during ENABLED scope — dep tracking unreliable.
    // Exception: persistent-qualified deps (empty state/local/next, not inconsistent,
    // not instance_lazy_read) are truly constant — their results are identical whether
    // evaluated inside or outside ENABLED scope. Caching these during ENABLED prevents
    // catastrophic re-evaluation of constant operators like SpanTreeRandom's `Edges`
    // (which calls RandomElement(SUBSET ...) and takes O(2^n) fingerprinting per call).
    // TLC equivalent: constant LazyValues are shared across EvalControl contexts.
    // Part of #4027: Use shadow via in_enabled_scope_ctx for consistency.
    let in_enabled = crate::cache::lifecycle::in_enabled_scope_ctx(eval_ctx);
    let is_persistent = crate::cache::zero_arg_cache::deps_are_persistent(&deps);
    // Step 11 (below) needs the deps after Step 10 moves them; clone only in
    // implied-action contexts so the global zero-arg miss path pays nothing.
    let deps_for_transition = if transition_fp.is_some() {
        Some(deps.clone())
    } else {
        None
    };
    // A preserved SetPred has not evaluated its predicate yet, so there are no
    // precise state-slot dependencies with which to validate a cache hit. Do
    // not cache a state-capturing wrapper at all: the state partition is
    // normally cleared at evaluation boundaries, but EvalCtx also supports
    // nested/direct state rebinding within one boundary. LazyFunc retains its
    // established state-partition policy; this stricter rule applies only to
    // the newly preserved RecordSet-filter shape.
    let store_primary = !deps.inconsistent
        && !state_capturing_set_pred
        && (!in_enabled || is_persistent);

    // Step 11 eligibility: Transition-partition store (implied-action
    // checking only). Strictly narrower than the state partition's:
    //   * deps confined to a single side matching the evaluation mode
    //     (`Current` → state reads only, `Next` → next reads only), so the
    //     value is a pure function of exactly the array identified by
    //     `transition_fp`;
    //   * no local reads, no TLCGet("level") dependence, no inconsistency,
    //     no INSTANCE lazy-read taint, not inside ENABLED scope (#3109: dep
    //     tracking unreliable there);
    //   * the value must not capture state arrays (LazyFunc — Fix #4145/#4158).
    // Deps are normalized to the `state` side so the entry can be reused in
    // either evaluation mode against the same state (mode translation happens
    // at hit time in `transition_deps_for_mode`).
    if let (Some(fp), Some(tdeps)) = (transition_fp, deps_for_transition.as_ref()) {
        if debug_implied_fp() {
            eprintln!(
                "[IMPLIED_FP] store? name={name} next={is_next_state} fp={fp:#x} \
                 in_enabled={in_enabled} eligible={} captures={} state_deps={} \
                 next_deps={} locals={} incons={} sni={} taint={} lvl={:?} val={}",
                transition_deps_eligible(tdeps, is_next_state),
                transition_value_captures_state(&result),
                tdeps.state.len(),
                tdeps.next.len(),
                tdeps.local.len(),
                tdeps.inconsistent,
                tdeps.state_next_inconsistent,
                tdeps.instance_lazy_read,
                tdeps.tlc_level,
                result.type_name()
            );
        }
    }
    let transition_store = match (transition_fp, deps_for_transition) {
        (Some(fp), Some(tdeps))
            if !in_enabled
                && transition_deps_eligible(&tdeps, is_next_state)
                && !transition_value_captures_state(&result) =>
        {
            let tkey = crate::cache::zero_arg_cache::zero_arg_transition_key(
                eval_ctx, name_id, def_loc, fp,
            );
            Some((
                tkey,
                CachedOpResult {
                    value: result.clone(),
                    deps: normalize_transition_deps(tdeps, is_next_state),
                },
            ))
        }
        _ => None,
    };

    if crate::cache::zero_arg_cache::no_zero_arg_probe_merge() {
        // ---- Kill switch: original store sequence (one TLS access each). ----
        if store_primary {
            // Also store under canonical key for constant operators so future
            // lookups from different scope contexts (different Arc<OpEnv>
            // pointers for recursive local_ops) can find the cached result.
            if is_persistent {
                crate::cache::zero_arg_cache::zero_arg_canonical_insert(
                    canonical_key,
                    CachedOpResult {
                        value: result.clone(),
                        deps: deps.clone(),
                    },
                );
            }
            zero_arg_insert(
                cache_key,
                CachedOpResult {
                    value: result.clone(),
                    deps,
                },
            );
        }
        if let Some((tkey, tentry)) = transition_store {
            crate::cache::zero_arg_cache::zero_arg_transition_insert(tkey, tentry);
        }
    } else {
        // Lever 2 (#EWD998PCal): Steps 10+11 merged into ONE TLS access.
        // Same canonical→primary→transition order, routing, and eviction as
        // the kill-switch branch; only the TLS access count changes.
        let canonical_store = (store_primary && is_persistent).then(|| {
            (
                canonical_key,
                CachedOpResult {
                    value: result.clone(),
                    deps: deps.clone(),
                },
            )
        });
        let primary_store = store_primary.then(|| {
            (
                cache_key,
                CachedOpResult {
                    value: result.clone(),
                    deps,
                },
            )
        });
        crate::cache::zero_arg_cache::zero_arg_store_all(
            primary_store,
            canonical_store,
            transition_store,
        );
    }

    Ok(result)
}

/// Evaluate a resolved zero-argument operator with the same cache and scope
/// boundaries used by `eval_ident`.
pub(crate) fn eval_resolved_zero_arg_op(
    ctx: &EvalCtx,
    resolved_name: &str,
    def: &Arc<OperatorDef>,
    span: Option<Span>,
    shared_scope: bool,
) -> EvalResult<Value> {
    if shared_scope {
        if ctx.local_ops().is_some() || ctx.instance_substitutions().is_some() {
            let outer_ctx = ctx.with_outer_resolution_scope();
            return eval_zero_arg_cached(&outer_ctx, ctx, resolved_name, def, span, false);
        }
        return eval_zero_arg_cached(ctx, ctx, resolved_name, def, span, false);
    }

    eval_zero_arg_cached(ctx, ctx, resolved_name, def, span, true)
}

/// Evaluate a zero-argument local operator (from `local_ops` / INSTANCE module).
///
/// Delegates to [`eval_zero_arg_cached`] with:
/// - `env_pre_check = true` (Fix #100: check env for existing LazyFunc)
pub(super) fn eval_zero_arg_local_op(
    ctx: &EvalCtx,
    name: &str,
    def: &Arc<OperatorDef>,
    span: Option<Span>,
) -> EvalResult<Value> {
    eval_resolved_zero_arg_op(ctx, name, def, span, false)
}

/// Evaluate a shared (outer-module) zero-argument operator.
///
/// Exits the current INSTANCE scope before evaluation. This is necessary because
/// `eval_module_ref` replaces the binding chain with instance substitution entries
/// (via `build_binding_chain_from_eager`/`build_lazy_subst_bindings`). If the
/// shared operator's body references a name that matches an instance substitution
/// (e.g., `INSTANCE M WITH x <- y` makes `x` resolve to `y`), the instance
/// binding would incorrectly shadow the outer-scope variable. Restoring the
/// pre-scope chain ensures the body evaluates in its definition scope.
///
/// Part of #3056 Phase 5: uses `with_outer_resolution_scope()` to rewind to the
/// pre-INSTANCE binding chain. Cannot use `without_instance_resolution_scope()`
/// because the chain is non-empty at this point (reachable from `with_module_scope`).
///
/// Delegates to [`eval_zero_arg_cached`] with `env_pre_check = false`.
pub(super) fn eval_ident_shared_zero_arg_op(
    ctx: &EvalCtx,
    resolved_name: &str,
    def: &Arc<OperatorDef>,
    span: Option<Span>,
) -> EvalResult<Value> {
    eval_resolved_zero_arg_op(ctx, resolved_name, def, span, true)
}

// ---------------------------------------------------------------------------
// Fingerprint-keyed transition cache helpers (implied-action checking)
// ---------------------------------------------------------------------------

feature_flag!(pub(crate) debug_implied_fp, "TY_DEBUG_IMPLIED_FP");

/// Fingerprint of the state array the current zero-arg evaluation reads, or
/// `None` when the fingerprint-keyed transition cache must stay inactive.
///
/// `Some` only when ALL of:
///   * the checker installed an implied-action fingerprint scope
///     (`implied_fp_scope`), i.e. we are evaluating an `[][A]_v` term with the
///     parent bound as `state_env` (fp = pair.0) and the successor bound as
///     `next_state_env` (fp = pair.1);
///   * the kill switch `TY_NO_IMPLIED_FP_CACHE` is not set;
///   * we are not inside ENABLED scope (#3109: state reads route through
///     sparse overlays there — both dep tracking and array-slot validation
///     would be unreliable).
#[inline]
pub(crate) fn implied_transition_fp(eval_ctx: &EvalCtx, is_next_state: bool) -> Option<u64> {
    if crate::cache::zero_arg_cache::no_implied_fp_cache() {
        return None;
    }
    let (parent_fp, succ_fp) = eval_ctx.runtime_state.implied_fp_pair.get()?;
    if crate::cache::lifecycle::in_enabled_scope_ctx(eval_ctx) {
        return None;
    }
    Some(if is_next_state { succ_fp } else { parent_fp })
}

/// Validate a transition-partition entry against the currently bound arrays.
///
/// Entries store deps normalized to the `state` side; in `Next` mode they are
/// validated against `next_state_env` instead. Every recorded dep value must
/// match the corresponding slot of the bound array — this makes the cache
/// sound independent of the fingerprint key (collisions degrade to misses).
/// Fail-closed: no bound array for the mode → invalid.
pub(crate) fn transition_entry_valid(
    ctx: &EvalCtx,
    entry: &CachedOpResult,
    is_next_state: bool,
) -> bool {
    debug_assert!(
        entry.deps.local.is_empty()
            && entry.deps.next.is_empty()
            && entry.deps.tlc_level.is_none()
            && !entry.deps.inconsistent
            && !entry.deps.instance_lazy_read,
        "transition cache entries must be state-side normalized"
    );
    // Mirror `op_cache_entry_valid`'s mode-aware validation branches exactly.
    // `eval_prime` may evaluate Next-mode expressions with SWAPPED arrays (the
    // successor bound as `state_env`, `next_state_env` unbound) or through
    // sparse overlays / binding-chain fast bindings, so a fixed
    // `ctx.next_state_env` slot compare would spuriously reject valid entries.
    // The cascades resolve each dep the same way an actual read would, so
    // validation compares the recorded value against exactly the value a
    // re-evaluation would observe. Fail-closed on every unresolvable dep.
    if is_next_state {
        if let Some(next_env) = ctx.next_state_env {
            for (idx, expected) in entry.deps.state.iter() {
                if idx.as_usize() >= next_env.env_len() {
                    return false;
                }
                // SAFETY: index bounds-checked against the bound array above.
                if !unsafe { next_env.slot_matches_compact(idx.as_usize(), expected) } {
                    return false;
                }
            }
        } else if let Some(next_state) = &ctx.next_state {
            for (idx, expected) in entry.deps.state.iter() {
                let name = ctx.var_registry().name(idx);
                let Some(actual) = next_state.get(name) else {
                    return false;
                };
                if !expected.matches_value(actual) {
                    return false;
                }
            }
        } else if current_state_lookup_mode(ctx) == StateLookupMode::Next {
            for (idx, expected) in entry.deps.state.iter() {
                if !crate::cache::op_result_cache::next_mode_dep_matches_compact(ctx, idx, expected)
                {
                    return false;
                }
            }
        } else {
            return false;
        }
    } else {
        if let Some(state_env) = ctx.state_env {
            for (idx, expected) in entry.deps.state.iter() {
                if idx.as_usize() >= state_env.env_len() {
                    return false;
                }
                // SAFETY: index bounds-checked against the bound array above.
                if !unsafe { state_env.slot_matches_compact(idx.as_usize(), expected) } {
                    return false;
                }
            }
        } else {
            for (idx, expected) in entry.deps.state.iter() {
                if !crate::cache::op_result_cache::current_mode_dep_matches_compact(
                    ctx, idx, expected,
                ) {
                    return false;
                }
            }
        }
    }
    true
}

/// Validate-and-refresh variant of [`transition_entry_valid`] for the lean
/// implied-action probe: identical acceptance decision, but in the bound-env
/// branches a successful NON-bitwise dep match refreshes the stored snapshot
/// to a clone of the live slot (validated equal, so entry semantics are
/// unchanged). The next probe of the same entry under the same binding then
/// matches on bits / inner-pointer equality instead of deep structural
/// comparison — the dominant validation cost when states are rebuilt across
/// generations. Branches without a directly bound env (sparse overlays,
/// legacy maps) validate exactly as before, without refresh.
pub(crate) fn transition_entry_valid_refresh(
    ctx: &EvalCtx,
    entry: &mut CachedOpResult,
    is_next_state: bool,
) -> bool {
    debug_assert!(
        entry.deps.local.is_empty()
            && entry.deps.next.is_empty()
            && entry.deps.tlc_level.is_none()
            && !entry.deps.inconsistent
            && !entry.deps.instance_lazy_read,
        "transition cache entries must be state-side normalized"
    );
    if is_next_state {
        if let Some(next_env) = ctx.next_state_env {
            for (idx, expected) in entry.deps.state.iter_mut() {
                if idx.as_usize() >= next_env.env_len() {
                    return false;
                }
                // SAFETY: index bounds-checked against the bound array above.
                if !unsafe { next_env.slot_matches_compact_refresh(idx.as_usize(), expected) } {
                    return false;
                }
            }
            return true;
        }
    } else if let Some(state_env) = ctx.state_env {
        for (idx, expected) in entry.deps.state.iter_mut() {
            if idx.as_usize() >= state_env.env_len() {
                return false;
            }
            // SAFETY: index bounds-checked against the bound array above.
            if !unsafe { state_env.slot_matches_compact_refresh(idx.as_usize(), expected) } {
                return false;
            }
        }
        return true;
    }
    // No directly bound env for the mode: fall back to the read-only
    // validator's remaining branches (no refresh possible or needed).
    transition_entry_valid(ctx, entry, is_next_state)
}

/// Translate a state-side-normalized dep set into the deps observed by the
/// current evaluation mode, for propagation into the enclosing dep frame.
/// In `Next` mode the recorded reads were actually next-state reads.
pub(crate) fn transition_deps_for_mode(entry_deps: &OpEvalDeps, is_next_state: bool) -> OpEvalDeps {
    if is_next_state {
        OpEvalDeps {
            next: entry_deps.state.clone(),
            ..OpEvalDeps::default()
        }
    } else {
        entry_deps.clone()
    }
}

/// Transition-partition store eligibility: deps must be exclusively
/// single-sided state-variable reads matching the evaluation mode.
pub(crate) fn transition_deps_eligible(deps: &OpEvalDeps, is_next_state: bool) -> bool {
    if deps.inconsistent
        || deps.state_next_inconsistent
        || deps.instance_lazy_read
        || !deps.local.is_empty()
        || deps.tlc_level.is_some()
    {
        return false;
    }
    if is_next_state {
        deps.state.is_empty() && !deps.next.is_empty()
    } else {
        deps.next.is_empty() && !deps.state.is_empty()
    }
}

/// Normalize eligible deps to the `state` side for mode-independent storage.
pub(crate) fn normalize_transition_deps(deps: OpEvalDeps, is_next_state: bool) -> OpEvalDeps {
    if is_next_state {
        OpEvalDeps {
            state: deps.next.clone(),
            ..OpEvalDeps::default()
        }
    } else {
        OpEvalDeps {
            state: deps.state.clone(),
            ..OpEvalDeps::default()
        }
    }
}

/// Values that capture state arrays must never cross state boundaries
/// (Fix #4145/#4158 rationale). Mirrors `value_captures_state` in
/// eval_let/zero_arg_cache.rs.
#[inline]
pub(crate) fn transition_value_captures_state(val: &Value) -> bool {
    val.captures_state_environment()
}

/// Small finite lazy-func domain check for eager materialization during
/// implied-action checking (Step 9c). Conservative: only `General` domains
/// with a known cardinality <= 64 qualify.
pub(crate) fn lazy_domain_is_small_finite(domain: &LazyDomain) -> bool {
    use num_traits::ToPrimitive;
    match domain {
        LazyDomain::General(v) => v
            .set_len()
            .and_then(|n| n.to_u64())
            .is_some_and(|n| n <= 64),
        _ => false,
    }
}

/// Evaluate the domain for a recursive function definition in identifier context.
/// Handles both single-bound and multi-bound function signatures.
pub(super) fn eval_ident_func_domain(
    ctx: &EvalCtx,
    bounds: &[BoundVar],
    span: Option<Span>,
) -> EvalResult<Value> {
    if bounds.len() == 1 {
        let domain_expr = bounds[0]
            .domain
            .as_ref()
            .ok_or_else(|| EvalError::Internal {
                message: "Function definition requires bounded variable".into(),
                span,
            })?;
        eval(ctx, domain_expr)
    } else {
        let mut components = Vec::with_capacity(bounds.len());
        for b in bounds {
            let domain_expr = b.domain.as_ref().ok_or_else(|| EvalError::Internal {
                message: "Function definition requires bounded variable".into(),
                span,
            })?;
            components.push(eval(ctx, domain_expr)?);
        }
        Ok(Value::tuple_set(components))
    }
}

// ---------------------------------------------------------------------------
// Lean implied-action transition-memo probe (hit-path fast lane).
// ---------------------------------------------------------------------------

feature_flag!(pub(crate) no_implied_lean_probe, "TY_NO_IMPLIED_LEAN_PROBE");

/// Lean hit-path probe of the fingerprint-keyed implied-action transition
/// memo for a zero-arg external operator reference.
///
/// During implied-action checking, every transition re-references the pinned
/// refinement operators (`token` / `pending`, primed and unprimed). Those
/// references are transition-memo HITS in the steady state, but the full
/// `eval_zero_arg_cached` path pays, per probe, the primary-partition miss,
/// the canonical-key build, and — for primed references — an `EvalCtx` clone
/// plus mode/hoist guards that only the (rare) MISS evaluation needs. This
/// function is the probe-only prefix: it builds the exact same
/// transition-partition key the store side uses and returns the entry when —
/// and only when — the same mode-aware dep validation the full path applies
/// (`transition_entry_valid`, fail-closed, collision-degrades-to-miss)
/// accepts it. On `None` the caller MUST fall through to the full path,
/// which re-probes and (on a real miss) evaluates and stores — behavior is
/// byte-identical to a run without this fast lane, minus redundant work.
///
/// Parity notes:
/// * Key components (`shared.id`, scope ids, `name_id`, `def_loc`, side fp)
///   are built with the same helpers on the same `ctx`; the full primed
///   path's swapped clone differs from `ctx` only in its state-env bindings,
///   which are not key components.
/// * Validation for primed references runs against `ctx.next_state_env`
///   directly — `transition_entry_valid` is mode-aware, so no env swap (and
///   therefore no `EvalCtx` clone) is needed on the hit path.
/// * Dep propagation (`transition_deps_for_mode`) matches the full path; the
///   runtime dep-frame state is shared between `ctx` and the full path's
///   swapped clone, so the propagation target is identical.
/// * Restricted to module-namespace (shared-scope) operators: a local-scope
///   name must resolve through the local overlay, which only the full path
///   does.
///
/// Kill switch: `TY_NO_IMPLIED_LEAN_PROBE=1` (falls back to the full probe
/// sequence for every reference). The memo itself remains governed by
/// `TY_NO_IMPLIED_FP_CACHE` (checked inside `implied_transition_fp`).
pub fn implied_zero_arg_transition_probe(ctx: &EvalCtx, name: &str, prime: bool) -> Option<Value> {
    if no_implied_lean_probe() || no_local_ops_cache() {
        return None;
    }
    let is_next = prime || current_state_lookup_mode(ctx) == StateLookupMode::Next;
    let fp = implied_transition_fp(ctx, is_next)?;
    let resolved = ctx.resolve_op_name(name);
    if ctx.name_in_local_scope(resolved) {
        return None;
    }
    let def = ctx.get_op(resolved)?;
    if !def.params.is_empty() || def.is_recursive {
        return None;
    }
    let name_id = intern_name(resolved);
    let def_loc = def.body.span.start;
    let tkey = crate::cache::zero_arg_cache::zero_arg_transition_key(ctx, name_id, def_loc, fp);
    let entry = crate::cache::zero_arg_cache::zero_arg_transition_lookup_refresh(&tkey, |e| {
        transition_entry_valid_refresh(ctx, e, is_next)
    })?;
    crate::cache::zero_arg_cache::record_zero_arg_primary_hit();
    // Parity with the full path's `propagate_cached_deps` call, minus the
    // unconditional deps clone: `transition_deps_for_mode` clones the dep
    // map, but propagation is a no-op unless dep tracking is active — the
    // clone is pure waste on the (dominant) inactive hot path.
    if crate::cache::dep_tracking::is_dep_tracking_active(ctx) {
        propagate_cached_deps(ctx, &transition_deps_for_mode(&entry.deps, is_next));
    }
    Some(entry.value)
}

/// Parent-side (unprimed, `Current`-mode) variant of
/// [`implied_zero_arg_transition_probe`] for the implied-action checker's
/// per-parent external seeding.
///
/// Returns a validated transition-memo value ONLY when the ambient state
/// lookup mode is `Current`: the seed is consumed by VM executions whose
/// unprimed `CallExternal` evaluations read the parent side exactly when the
/// ambient mode is `Current`, so seeds must never be built under a different
/// mode. The transition partition's store-side eligibility
/// (`transition_deps_eligible`) guarantees the returned value depends
/// exclusively on parent-side state reads, and validation ran against the
/// currently bound parent array — so the value is exact for every evaluation
/// under the same parent binding.
pub fn implied_parent_side_external_probe(ctx: &EvalCtx, name: &str) -> Option<Value> {
    if current_state_lookup_mode(ctx) != StateLookupMode::Current {
        return None;
    }
    implied_zero_arg_transition_probe(ctx, name, false)
}

/// `true` iff the ambient state lookup mode is `Current` (see
/// [`implied_parent_side_external_probe`]; used by the checker to gate seed
/// CONSUMPTION the same way construction is gated).
pub fn state_lookup_mode_is_current(ctx: &EvalCtx) -> bool {
    current_state_lookup_mode(ctx) == StateLookupMode::Current
}
