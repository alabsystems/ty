// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

use super::fallback::eval_state_leaves_with_cached_successors_fallback;
use super::{eval_bool_expr_with_entry_mode, InlineLeafEntryMode};
use crate::error::{EvalError, EvalResult};
use crate::eval::{BindingChain, EvalCtx};
use crate::liveness::live_expr::LiveExpr;
use crate::state::{ArrayState, Fingerprint, State};
use rustc_hash::FxHashMap;
use std::sync::Arc;
use tla_core::ast::Expr;
use tla_core::Spanned;

pub(super) fn state_leaf_preserves_batch_boundary(expr: &LiveExpr) -> bool {
    match expr {
        LiveExpr::Bool(_) | LiveExpr::StatePred { .. } => true,
        LiveExpr::Not(inner)
        | LiveExpr::Always(inner)
        | LiveExpr::Eventually(inner)
        | LiveExpr::Next(inner) => state_leaf_preserves_batch_boundary(inner),
        LiveExpr::And(parts) | LiveExpr::Or(parts) => {
            parts.iter().all(state_leaf_preserves_batch_boundary)
        }
        LiveExpr::Enabled { .. } | LiveExpr::ActionPred { .. } | LiveExpr::StateChanged { .. } => {
            false
        }
    }
}

/// Lazily-converted `State` views of the current state and its cached
/// successors, shared across all ENABLED leaves of one state-leaf batch.
///
/// Part of #liveness-leaf-memo: the ENABLED delegate consumes `State`s, and the
/// previous code converted `current_array` plus EVERY successor array fresh on
/// each ENABLED cache miss — 115 redundant conversion sweeps per state on
/// AllocatorImplementation. The conversion is a pure function of the arrays
/// (no eval state), so sharing one conversion per batch is observationally
/// identical.
pub(super) struct EnabledStateViews {
    states: std::cell::OnceCell<(State, Vec<State>)>,
}

impl EnabledStateViews {
    pub(super) fn new() -> Self {
        Self {
            states: std::cell::OnceCell::new(),
        }
    }

    fn get(
        &self,
        ctx: &EvalCtx,
        current_array: &ArrayState,
        successors: &[(&ArrayState, Fingerprint)],
    ) -> &(State, Vec<State>) {
        self.states.get_or_init(|| {
            let registry = ctx.var_registry().clone();
            (
                current_array.to_state(&registry),
                successors
                    .iter()
                    .map(|(arr, _)| arr.to_state(&registry))
                    .collect(),
            )
        })
    }
}

#[allow(clippy::too_many_arguments)]
fn eval_enabled_with_array_successors(
    ctx: &mut EvalCtx,
    current_fp: Fingerprint,
    current_array: &ArrayState,
    action: &Arc<Spanned<Expr>>,
    bindings: Option<&BindingChain>,
    require_state_change: bool,
    subscript: Option<&Arc<Spanned<Expr>>>,
    tag: u32,
    successors: &[(&ArrayState, Fingerprint)],
    state_views: &EnabledStateViews,
    // Part of #liveness-enabled-batch-ctx: shared per-state prepared ENABLED
    // context (built once per state by `eval_enabled_leaves_batched`). When
    // `Some`, it is used as `ctx_current` with `state_prepared = true`, so the
    // per-leaf state-variable snapshot+rebind is amortized across the batch.
    // `None` (single-leaf path, or kill switch) preserves the legacy per-leaf
    // preparation from `ctx` exactly.
    prepared: Option<&EvalCtx>,
) -> EvalResult<bool> {
    super::super::checker::eval_enabled_cached_mut(ctx, current_fp, tag, |ctx| {
        // TRUE-only ENABLED provenance (#3208 redo of #3100): the BFS successor
        // generation of THIS state already witnessed a state-changing successor
        // produced by exactly this leaf's (operator, argument-values) action —
        // ENABLED is true, no from-scratch enumeration needed. Fingerprint-
        // checked, TRUE-only (a miss means "no claim", never "false"), and a
        // full-evaluation fallback would compute the same `true` (the witness
        // is a genuine emission of the same enumerator the fallback runs), so
        // caching the result in the (fp, tag) ENABLED cache is consistent.
        // Kill switch: TY_DISABLE_ENABLED_PROVENANCE=1 empties the
        // registration, so this probe never hits and every leaf takes the
        // full path below.
        if super::super::enabled_provenance::witnessed_true(current_fp, tag) {
            // #liveness-witness-pair-pop: the witness decides ENABLED = true,
            // but the skipped enumeration was dual-purpose — its complete
            // successor set fed `populate_pred_results_from_enumeration`,
            // letting the paired ActionPred leaf answer per-transition from
            // membership instead of one full AST predicate evaluation per
            // (state × successor) — measured at ~77% of Huang's wall. Run
            // that same enumeration here PURELY for population, gated on:
            //   - the pair exists and carries the all-vars pinning proof
            //     (`full_population_tag`) so the FALSE side pays — the exact
            //     gate the enum-first population path already trusts;
            //   - the leaf is not whole-Next (its pair is decided by
            //     edge provenance / the whole-Next fast paths, and
            //     re-enumerating the full Next per state is a measured
            //     regression on whole-Next specs);
            //   - the `TY_DISABLE_WITNESS_PAIR_POPULATION=1` kill switch is
            //     off (differential proves verdict identity).
            // The ENABLED verdict below is the witnessed `true` regardless of
            // what the population enumeration does (error/capped/empty all
            // fail closed to per-transition evaluation — the pre-change
            // behavior, see `populate_witnessed_pair`).
            if super::super::enabled_eval::witness_pair_population_enabled()
                && !super::super::checker::whole_next_enabled_tag(tag)
            {
                if let Some(pair_tag) = super::super::checker::enabled_action_pred_pair(tag) {
                    if super::super::checker::full_population_tag(pair_tag) {
                        // #frame-fp-pop: FIRST try populating the pair from
                        // the per-frame emitted-successor fingerprints the
                        // armed BFS generation recorded — ZERO re-enumeration
                        // (the population enumeration below re-computed
                        // successors the generation had just emitted,
                        // measured at 59% of Huang's wall). All soundness
                        // gates (static Next-shape certificate, all-vars
                        // pinning, un-truncated arm, un-poisoned record) live
                        // inside `populate_pair_from_frame_fps`; ANY of them
                        // failing answers `false` and the landed
                        // re-enumeration below runs unchanged. Kill switch:
                        // TY_DISABLE_FRAME_FP_POPULATION=1 (falls back to the
                        // landed path, which has its own
                        // TY_DISABLE_WITNESS_PAIR_POPULATION=1 switch).
                        let populated =
                            super::super::enabled_provenance::populate_pair_from_frame_fps(
                                current_fp,
                                tag,
                                pair_tag,
                                successors.iter().map(|&(_, fp)| fp),
                            );
                        if !populated {
                            let (current_state, cached) =
                                state_views.get(ctx, current_array, successors);
                            let (ctx_current, state_prepared): (&EvalCtx, bool) = match prepared {
                                Some(p) => (p, true),
                                None => (&*ctx, false),
                            };
                            super::super::enabled_eval::populate_witnessed_pair(
                                ctx_current,
                                state_prepared,
                                current_state,
                                action,
                                bindings,
                                cached,
                                pair_tag,
                            );
                        }
                    }
                }
            }
            return Ok(true);
        }
        // Whole-Next fast path: `ENABLED(<<Next>>_vars)` where the action is the
        // config's COMPLETE next-state relation. For the full relation (unlike a
        // sub-action) the BFS successor slice IS the complete successor set, so
        // "some successor changes the subscript" is an EXACT, sound decision — no
        // from-scratch Next re-enumeration. Registered at plan build
        // (`record_whole_next_enabled_tag` → `extend_whole_next_enabled_tags`).
        if super::super::checker::whole_next_enabled_tag(tag) {
            let (current_state, cached) = state_views.get(ctx, current_array, successors);
            for succ_state in cached {
                let changed = match subscript {
                    Some(sub) => super::super::checker::eval_subscript_changed_state_cached(
                        ctx,
                        current_state,
                        succ_state,
                        sub,
                        tag,
                    )?,
                    None => current_state.fingerprint() != succ_state.fingerprint(),
                };
                if changed {
                    return Ok(true);
                }
            }
            return Ok(false);
        }
        // SOUND absence side (guard-prefix refutation): a prime-free,
        // operator-safe leading conjunct of the leaf's action operator body —
        // registered fail-closed at plan build — evaluating to FALSE in the
        // current state proves the action relation empty, hence
        // ENABLED <<A>>_e = false for ANY subscript. `false` from the probe
        // means "no claim" (guards true / eval error / no plan) and falls
        // through to the full evaluation below. Kill-switched together with
        // the TRUE side (TY_DISABLE_ENABLED_PROVENANCE=1 registers nothing);
        // the #guard-in-memo refinements inside are additionally covered by
        // TY_DISABLE_ENABLED_GUARD_MEMO=1. `current_fp` keys the per-state
        // shared RHS memo — the same (fp ↔ ctx state) pairing this
        // eval_enabled_cached_mut call itself relies on.
        if super::super::enabled_provenance::guard_prefix_refutes(ctx, current_fp, tag) {
            // The refutation proves the action relation EMPTY from this state
            // (a state-level necessary condition of A is FALSE, so
            // ∀t: ¬A(s, t)) — share per-transition FALSE with the paired
            // ActionPred leaf, exactly like the empty-set case of
            // `populate_pred_results_from_enumeration` does after a complete
            // enumeration. Unlike enumeration-based FALSE population this
            // needs NO pinning proof: emptiness comes from the refuted
            // conjunct, not from enumeration completeness, so it holds for
            // every candidate successor (including the stuttering pair, which
            // `successors` already carries when stuttering is allowed).
            // Without this, a guard-refuted leaf would skip the enumeration
            // that used to feed the recorder and push the paired action leaf
            // back onto full per-transition AST evaluation.
            if let Some(pair_tag) = super::super::checker::enabled_action_pred_pair(tag) {
                let (current_state, cached) = state_views.get(ctx, current_array, successors);
                let cur_fp = current_state.fingerprint();
                for succ in cached {
                    super::super::checker::insert_scan_pred_result(
                        cur_fp,
                        succ.fingerprint(),
                        pair_tag,
                        false,
                    );
                }
            }
            return Ok(false);
        }
        // Soundness (#liveness-wf): ENABLED of a subscripted fairness action
        // (`WF_e(A)`/`SF_e(A)`) must NOT be decided by scanning only the explored
        // behavior-graph successors. That set can be incomplete for a given
        // action (its non-stuttering witness successor may be merged/deduplicated
        // or simply absent from the array slice handed to this leaf batch), and
        // the array-native `eval_entry_inline` scan additionally suffers from
        // next-state cache pollution across successors — both yield a spurious
        // `false` (action reported disabled when it is in fact enabled). That
        // flips `~ENABLED` to true and makes WF/SF trivially satisfiable,
        // producing an unsound liveness counterexample (e.g. SingleLaneBridge).
        //
        // Delegate to the shared `eval_enabled_uncached` (the same algorithm the
        // post-BFS SCC and consistency paths use), which checks the action as a
        // *predicate* over each candidate successor via the clean `eval_live_entry`
        // boundary (no inline cache pollution) and, when the cached set has no
        // witness, enumerates the action's own successors from scratch. We pass
        // the explored successors as the cached set so the common (enabled) case
        // is still resolved cheaply by the first matching successor.
        let (current_state, cached) = state_views.get(ctx, current_array, successors);
        // Part of #liveness-enabled-batch-ctx: prefer the shared per-state
        // prepared context when available; otherwise fall back to the model
        // checker ctx with per-leaf preparation (`state_prepared = false`).
        let (ctx_current, state_prepared): (&EvalCtx, bool) = match prepared {
            Some(p) => (p, true),
            None => (&*ctx, false),
        };
        let result = super::super::enabled_eval::eval_enabled_uncached(
            super::super::enabled_eval::EnabledEvalRequest {
                ctx_current,
                current_state,
                action,
                bindings,
                require_state_change,
                subscript,
                cached_successors: cached,
                // Part of #liveness-leaf-memo: share scan predicate results
                // with the paired ActionPred leaf (same resolved action).
                pred_cache_tag: super::super::checker::enabled_action_pred_pair(tag),
                state_prepared,
            },
            // Part of #liveness-leaf-memo: probe subscript changes through the
            // (Fingerprint, tag)-keyed subscript value cache instead of two
            // full AST evaluations per (leaf × successor) probe. Same key
            // shape, lifecycle, and fp64-trust argument as the SCC-phase
            // evaluator (checker/eval.rs) and the inline action-leaf
            // short-circuit; see eval_subscript_changed_state_cached.
            |eval_ctx, s1, s2, sub_expr| {
                super::super::checker::eval_subscript_changed_state_cached(
                    eval_ctx, s1, s2, sub_expr, tag,
                )
            },
        )?;
        // TRUE-only ENABLED provenance: diagnostics-only outcome counter for
        // the full (provenance-miss) evaluations.
        super::super::enabled_provenance::note_full_eval(result);
        Ok(result)
    })
}

#[allow(clippy::too_many_arguments)]
fn eval_state_leaf_array(
    ctx: &mut EvalCtx,
    expr: &LiveExpr,
    current_fp: Fingerprint,
    current_array: &ArrayState,
    successors: &[(&ArrayState, Fingerprint)],
    entry_mode: InlineLeafEntryMode,
    state_views: &EnabledStateViews,
) -> EvalResult<bool> {
    match expr {
        LiveExpr::Bool(b) => Ok(*b),
        LiveExpr::StatePred { expr, bindings, .. } => match bindings {
            // `StatePred` leaves are intentionally NOT result-cached: the inline
            // state-leaf bitmask already dedups state-predicate evaluation per
            // source fingerprint, so a `(current_fp, tag)` result cache measured
            // zero hits across the corpus and was pure overhead. See
            // leaf_result_cache.rs.
            Some(chain) => {
                let eval_ctx = ctx.with_liveness_bindings(chain);
                eval_bool_expr_with_entry_mode(&eval_ctx, expr, entry_mode)
            }
            None => eval_bool_expr_with_entry_mode(ctx, expr, entry_mode),
        },
        LiveExpr::Enabled {
            action,
            bindings,
            require_state_change,
            subscript,
            tag,
        } => eval_enabled_with_array_successors(
            ctx,
            current_fp,
            current_array,
            action,
            bindings.as_ref(),
            *require_state_change,
            subscript.as_ref(),
            *tag,
            successors,
            state_views,
            // Single ENABLED leaf: no per-state batch to amortize, use the
            // legacy per-leaf preparation path.
            None,
        ),
        LiveExpr::Not(inner) => Ok(!eval_state_leaf_array(
            ctx,
            inner,
            current_fp,
            current_array,
            successors,
            entry_mode,
            state_views,
        )?),
        LiveExpr::And(parts) => {
            for part in parts {
                if !eval_state_leaf_array(
                    ctx,
                    part,
                    current_fp,
                    current_array,
                    successors,
                    entry_mode,
                    state_views,
                )? {
                    return Ok(false);
                }
            }
            Ok(true)
        }
        LiveExpr::Or(parts) => {
            for part in parts {
                if eval_state_leaf_array(
                    ctx,
                    part,
                    current_fp,
                    current_array,
                    successors,
                    entry_mode,
                    state_views,
                )? {
                    return Ok(true);
                }
            }
            Ok(false)
        }
        LiveExpr::ActionPred { expr, .. }
        | LiveExpr::StateChanged {
            subscript: Some(expr),
            ..
        } => Err(EvalError::Internal {
            message: format!(
                "array-native state leaf evaluation received action-level expression {:?}",
                expr.node
            ),
            span: Some(expr.span),
        }),
        LiveExpr::StateChanged {
            subscript: None, ..
        }
        | LiveExpr::Always(_)
        | LiveExpr::Eventually(_)
        | LiveExpr::Next(_) => Err(EvalError::Internal {
            message: "array-native state leaf evaluation received unsupported liveness node"
                .to_string(),
            span: None,
        }),
    }
}

pub(crate) fn eval_state_leaves_with_array_successors(
    ctx: &mut EvalCtx,
    leaves: &[&LiveExpr],
    current_fp: Fingerprint,
    current_array: &ArrayState,
    successors: &[(&ArrayState, Fingerprint)],
) -> EvalResult<Vec<(u32, bool)>> {
    if leaves.is_empty() {
        return Ok(Vec::new());
    }

    let registry = ctx.var_registry().clone();
    if registry.is_empty() {
        let current_state = current_array.to_state(&registry);
        let cached_successors: Vec<_> = successors
            .iter()
            .map(|(state, _)| state.to_state(&registry))
            .collect();
        return eval_state_leaves_with_cached_successors_fallback(
            ctx,
            leaves,
            &current_state,
            &cached_successors,
        );
    }

    let mut enabled_indices = Vec::new();
    let mut non_enabled_indices = Vec::new();
    for (index, leaf) in leaves.iter().enumerate() {
        if matches!(leaf, LiveExpr::Enabled { .. }) {
            enabled_indices.push(index);
        } else {
            non_enabled_indices.push(index);
        }
    }
    debug_assert_eq!(
        enabled_indices.len() + non_enabled_indices.len(),
        leaves.len()
    );

    let _state_guard = ctx.bind_state_env_guard(current_array.env_ref());
    let _next_state_guard = ctx.take_next_state_guard();
    let _next_guard = ctx.take_next_state_env_guard();

    // Part of #liveness-leaf-memo: one shared (lazy) State conversion of the
    // current state + successors for ALL ENABLED leaves in this batch.
    let state_views = EnabledStateViews::new();

    let enabled_results_map: Option<FxHashMap<u32, bool>> = if enabled_indices.len() > 1 {
        let enabled_leaves: Vec<_> = enabled_indices.iter().map(|&i| leaves[i]).collect();
        let results = eval_enabled_leaves_batched(
            ctx,
            current_fp,
            current_array,
            &enabled_leaves,
            successors,
            &state_views,
        )?;
        Some(results.into_iter().collect())
    } else {
        None
    };

    let mut out = Vec::with_capacity(leaves.len());
    let mut entry_mode = InlineLeafEntryMode::Boundary;
    for expr in leaves {
        if let Some(tag) = expr.tag() {
            if let Some(ref map) = enabled_results_map {
                if let Some(&result) = map.get(&tag) {
                    out.push((tag, result));
                    entry_mode = InlineLeafEntryMode::Boundary;
                    continue;
                }
            }
            let result = eval_state_leaf_array(
                ctx,
                expr,
                current_fp,
                current_array,
                successors,
                entry_mode,
                &state_views,
            )?;
            out.push((tag, result));
            entry_mode = if state_leaf_preserves_batch_boundary(expr) {
                InlineLeafEntryMode::Inline
            } else {
                InlineLeafEntryMode::Boundary
            };
        }
    }

    Ok(out)
}

fn eval_enabled_leaves_batched(
    ctx: &mut EvalCtx,
    current_fp: Fingerprint,
    current_array: &ArrayState,
    enabled_leaves: &[&LiveExpr],
    successors: &[(&ArrayState, Fingerprint)],
    state_views: &EnabledStateViews,
) -> EvalResult<Vec<(u32, bool)>> {
    // Soundness (#liveness-wf): each ENABLED leaf is decided by the shared,
    // authoritative `eval_enabled_with_array_successors`, which delegates to
    // `eval_enabled_uncached` (predicate-over-successor scan via the clean
    // `eval_live_entry` boundary plus a from-scratch action enumeration
    // fallback). This replaces the prior bespoke per-successor `eval_entry_inline`
    // batch loop, which both depended on the (sometimes-incomplete) explored
    // successor set and suffered next-state cache pollution — yielding spurious
    // `ENABLED = false` for genuinely-enabled fairness actions and thus unsound
    // WF/SF satisfaction (e.g. SingleLaneBridge). Per-leaf ENABLED caching
    // (keyed by `(fp, tag)`) is preserved inside the delegate, so repeated leaves
    // and repeated states are still resolved cheaply.
    let _enabled_guard = crate::eval::enter_enabled_scope_with_ctx(ctx);
    // Part of #liveness-enabled-batch-ctx: the current-state variable snapshot +
    // rebind that `eval_enabled_uncached` performs before enumeration is
    // invariant across every ENABLED leaf of this state. Build that prepared
    // context ONCE here and share it across the batch, so only the per-leaf
    // quantifier bindings (and the enumeration itself) run per leaf. Gated by
    // the `TY_DISABLE_LIVENESS_ENABLED_BATCH` kill switch (default on); when
    // disabled, `prepared` is `None` and every leaf uses the legacy per-leaf
    // preparation, which the kill switch verifies is verdict-identical.
    let prepared = if super::super::enabled_eval::enabled_batch_ctx_enabled() {
        Some(super::super::enabled_eval::prepare_enabled_ctx(ctx))
    } else {
        None
    };
    let mut out = Vec::with_capacity(enabled_leaves.len());
    for leaf in enabled_leaves {
        if let LiveExpr::Enabled {
            action,
            bindings,
            require_state_change,
            subscript,
            tag,
        } = leaf
        {
            let result = eval_enabled_with_array_successors(
                ctx,
                current_fp,
                current_array,
                action,
                bindings.as_ref(),
                *require_state_change,
                subscript.as_ref(),
                *tag,
                successors,
                state_views,
                prepared.as_ref(),
            )?;
            out.push((*tag, result));
        }
    }
    Ok(out)
}
