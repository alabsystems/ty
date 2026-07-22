// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

#[cfg(debug_assertions)]
use super::debug_enum;
use super::unified::EnumParams;
use super::unified_types::DiffSink;
use super::{unified, EvalCtx, EvalError, Spanned, State};
use crate::state::{ArrayState, DiffSuccessor, UndoEntry};
use crate::var_index::{VarIndex, VarRegistry};
use std::ops::ControlFlow;
use std::sync::Arc;
use tla_core::ast::Expr;

debug_flag!(debug_liveness_enum, "TY_DEBUG_LIVENESS_ENUM");

/// Enumerate successors from an arbitrary action expression.
///
/// This is used for <code>ENABLED&lt;&lt;A&gt;&gt;_vars</code> evaluation in liveness checking, where we need
/// to check if the action produces any non-stuttering successor (vars' ≠ vars).
///
/// Unlike `enumerate_successors` which takes an `OperatorDef`, this function takes
/// a raw expression, making it suitable for evaluating sub-actions from fairness
/// constraints.
///
/// # Arguments
/// * `ctx` - Evaluation context with current state bound
/// * `action` - The action expression to enumerate
/// * `current_state` - The current state
/// * `vars` - Variable names for the spec
///
/// # Returns
/// Vector of successor states (may include stuttering - caller should filter if needed).
pub fn enumerate_action_successors(
    ctx: &mut EvalCtx,
    action: &Spanned<Expr>,
    current_state: &State,
    vars: &[Arc<str>],
) -> Result<Vec<State>, EvalError> {
    enumerate_action_successors_capped(ctx, action, current_state, vars, usize::MAX)
        .map(|(successors, _capped)| successors)
}

/// Witness watch for ENABLED enumeration early-exit
/// (#liveness-enabled-witness-exit).
///
/// Describes the fairness subscript `e` of `ENABLED <<A>>_e` in a form the
/// enumeration sink can decide per generated diff WITHOUT an `EvalCtx`:
///
/// - `AnyVar`: no subscript (`require_state_change` against the full state).
///   A diff is a witness iff some assigned value genuinely differs from the
///   base state (the enumerator UNCHANGED-completes unassigned variables, so
///   "some change entry differs" ⇔ "successor state ≠ current state"). This is
///   an EXACT value-equality test — at least as precise as the fingerprint
///   comparison `has_state_change` performs on materialized states.
/// - `Vars(indices)`: the subscript is statically proven (see
///   `subscript_watch_vars` in `liveness::enabled_eval`) to be a (nested)
///   tuple of exactly these state variables, so the subscript value changes
///   between two states iff one of the watched variables' values differs.
///   A diff is a witness iff some assigned WATCHED value genuinely differs.
#[derive(Clone, Copy)]
pub(crate) enum SubscriptWatch<'a> {
    AnyVar,
    Vars(&'a [VarIndex]),
}

/// Outcome of a witness-watched ENABLED enumeration
/// (#liveness-enabled-witness-exit).
pub(crate) enum EnabledEnumOutcome {
    /// Early exit: a generated successor changed the watched subscript. The
    /// vector holds every successor generated up to AND INCLUDING the witness
    /// (a PREFIX of the full set — each member is a genuine successor, but the
    /// set must not be treated as complete). Only produced when a watch was
    /// supplied.
    Witness(Vec<State>),
    /// Enumeration ran to completion: the set is COMPLETE (identical to
    /// `enumerate_action_successors_capped` returning `(set, false)`). When a
    /// watch was supplied, NO generated successor changed the watched
    /// subscript (exhaustive refutation of the witness test).
    Complete(Vec<State>),
    /// The generation cap was hit first: the set is an incomplete prefix
    /// (identical to `enumerate_action_successors_capped` returning
    /// `(prefix, true)`). When a watch was supplied, no member of the prefix
    /// changed the watched subscript.
    Capped(Vec<State>),
}

/// Capping sink: collects up to `cap` generated successors, then signals
/// `Break` to short-circuit the remaining enumeration (#liveness-enabled-enum-first).
///
/// Used by the inline-liveness ENABLED evaluator so a pathological fairness
/// action with a huge successor set cannot make enumeration-first slower than
/// the legacy scan order; a capped result tells the caller to fall back.
///
/// Part of #liveness-enabled-witness-exit: optionally carries a
/// [`SubscriptWatch`] — when a generated diff changes the watched subscript
/// (decided by exact value comparison against the base state, see
/// `diff_is_witness`), the sink records the witness and breaks immediately.
/// ENABLED only needs ONE such witness, so generating the remaining successor
/// set is pure waste. Without a watch the behavior is bit-identical to the
/// pre-witness-exit sink.
struct CappedDiffSink<'a> {
    diffs: Vec<DiffSuccessor>,
    cap: usize,
    capped: bool,
    witness: bool,
    watch: Option<SubscriptWatch<'a>>,
    base: &'a ArrayState,
}

/// Decide whether `diff` changes the watched subscript, by exact value
/// comparison against the base (current) state.
///
/// Soundness (#liveness-enabled-witness-exit): `diff.changes` holds every
/// variable the action assigns for this successor (unassigned variables are
/// UNCHANGED-completed by the enumerator and thus provably unchanged). An
/// entry's value may still EQUAL the base value (`InSet` domains are not
/// pre-filtered), so each candidate entry is compared with full `Value`
/// equality — the SAME comparison `emit_successor`'s fast path uses to filter
/// unchanged assignments (exact structural equality, no fingerprint trust;
/// deliberately NOT `CompactValue::matches_value`, which is incomplete for
/// inline string/model tags and would fabricate witnesses for equal values).
/// Therefore:
///   - `AnyVar`: returns true iff the successor state differs from the base.
///   - `Vars(w)`: returns true iff some watched variable's value differs,
///     which (per the static subscript proof) holds iff the subscript tuple's
///     value differs between base and successor.
#[inline]
fn diff_is_witness(diff: &DiffSuccessor, base: &ArrayState, watch: SubscriptWatch<'_>) -> bool {
    diff.changes.iter().any(|(idx, value)| {
        let watched = match watch {
            SubscriptWatch::AnyVar => true,
            SubscriptWatch::Vars(indices) => indices.contains(idx),
        };
        watched && base.get(*idx) != *value
    })
}

impl DiffSink for CappedDiffSink<'_> {
    #[inline]
    fn push(&mut self, diff: DiffSuccessor) -> ControlFlow<()> {
        if self.diffs.len() >= self.cap {
            self.capped = true;
            return ControlFlow::Break(());
        }
        let is_witness = self
            .watch
            .is_some_and(|watch| diff_is_witness(&diff, self.base, watch));
        self.diffs.push(diff);
        if is_witness {
            self.witness = true;
            return ControlFlow::Break(());
        }
        ControlFlow::Continue(())
    }

    #[inline]
    fn count(&self) -> usize {
        self.diffs.len()
    }

    #[inline]
    fn is_stopped(&self) -> bool {
        self.capped || self.witness
    }
}

/// Enumerate successors from an arbitrary action expression, generating at
/// most `cap` successors.
///
/// Identical to [`enumerate_action_successors`] except that enumeration stops
/// (soundly, via the sink `Break` protocol the streaming BFS already uses)
/// once `cap` successors have been GENERATED. Returns `(successors, capped)`;
/// when `capped` is true the successor set is a PREFIX of the full set — any
/// member is still a genuine successor (sound as an ENABLED witness), but the
/// set must not be treated as complete.
pub fn enumerate_action_successors_capped(
    ctx: &mut EvalCtx,
    action: &Spanned<Expr>,
    current_state: &State,
    vars: &[Arc<str>],
    cap: usize,
) -> Result<(Vec<State>, bool), EvalError> {
    match enumerate_action_successors_witness_capped(ctx, action, current_state, vars, cap, None)? {
        // Witness is only produced when a watch is supplied.
        EnabledEnumOutcome::Witness(_) => unreachable!("witness outcome without a watch"),
        EnabledEnumOutcome::Complete(successors) => Ok((successors, false)),
        EnabledEnumOutcome::Capped(successors) => Ok((successors, true)),
    }
}

/// Enumerate successors from an action expression with an optional
/// subscript-witness watch (#liveness-enabled-witness-exit).
///
/// Identical to [`enumerate_action_successors_capped`] except that when
/// `watch` is `Some`, generation stops (soundly, via the sink `Break`
/// protocol) as soon as a generated successor CHANGES the watched subscript —
/// the exact witness `ENABLED <<A>>_e` needs. See [`EnabledEnumOutcome`] for
/// the three result shapes and their completeness guarantees. With
/// `watch = None` the behavior is bit-identical to the plain capped
/// enumeration.
pub(crate) fn enumerate_action_successors_witness_capped(
    ctx: &mut EvalCtx,
    action: &Spanned<Expr>,
    current_state: &State,
    vars: &[Arc<str>],
    cap: usize,
    watch: Option<SubscriptWatch<'_>>,
) -> Result<EnabledEnumOutcome, EvalError> {
    // Part of #575: Debug liveness enumeration
    debug_block!(debug_liveness_enum(), {
        eprintln!("[DEBUG LIVENESS ENUM] Starting enumerate_action_successors");
        eprintln!("[DEBUG LIVENESS ENUM] action={:?}", action.node);
    });
    // Bind current state variables to ctx.env so they can be accessed during enumeration.
    // This is required for correctness: enumerate_unified/extract_symbolic_assignments
    // evaluate IF conditions using `eval(ctx, cond)`, which looks up state vars in ctx.env.
    // Without this binding, guard expressions like `light = \"off\"` fail to find state vars.
    //
    // Note: enumerate_successors() already does this binding, but enumerate_action_successors
    // was missing it, causing bugs in liveness checking (#575) where fairness constraints
    // failed because IF conditions couldn't find state variable values.
    for (name, value) in current_state.vars() {
        ctx.bind_mut(Arc::clone(name), value.clone());
    }

    // Part of #1262: Use unified enumeration path instead of legacy next_rec.
    // Build ArrayState from current State, run enumerate_unified, convert results.
    let registry = if !ctx.var_registry().is_empty() {
        ctx.var_registry().clone()
    } else {
        VarRegistry::from_names(vars.iter().cloned())
    };
    let current_array = ArrayState::from_state(current_state, &registry);

    let mut base = current_array.clone();
    base.ensure_fp_cache_with_value_fps(&registry);

    let mut working = current_array.clone_for_working();
    let mut undo_stack: Vec<UndoEntry> = Vec::with_capacity(vars.len() * 2);
    let mut sink = CappedDiffSink {
        diffs: Vec::with_capacity(8),
        cap,
        capped: false,
        witness: false,
        watch,
        base: &current_array,
    };

    let params = EnumParams::new(vars, &registry, &base);
    let mut rec = unified::RecState {
        working: &mut working,
        undo: &mut undo_stack,
        results: &mut sink,
    };
    // TRUE-only ENABLED provenance (#3208 redo of #3100): this entry is the
    // per-leaf ENABLED enumeration itself — the depth guard ensures that when
    // it runs nested inside an armed BFS generation (or in any later phase),
    // its emissions are never recorded as BFS witnesses.
    let _prov_scope = crate::liveness::enabled_provenance::enum_scope();
    let enum_result = unified::enumerate_unified(ctx, action, &params, &mut rec);

    match enum_result {
        Ok(()) => {
            debug_eprintln!(
                debug_enum(),
                "enumerate_action_successors (unified): produced {} diffs (capped={}, witness={}), converting to State",
                sink.diffs.len(),
                sink.capped,
                sink.witness
            );
            let capped = sink.capped;
            let witness = sink.witness;
            let successors: Vec<State> = sink
                .diffs
                .into_iter()
                .map(|d| {
                    let fp = d.fingerprint;
                    d.into_array_state(&current_array, &registry, Some(fp))
                        .to_state(&registry)
                })
                .collect();
            Ok(if witness {
                EnabledEnumOutcome::Witness(successors)
            } else if capped {
                EnabledEnumOutcome::Capped(successors)
            } else {
                EnabledEnumOutcome::Complete(successors)
            })
        }
        Err(e) => {
            // Fix #1552: TLC propagates ALL errors from action evaluation fatally.
            // No suppression at the action level — errors terminate model checking.
            Err(e)
        }
    }
}
