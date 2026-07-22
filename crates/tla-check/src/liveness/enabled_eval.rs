// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Shared ENABLED evaluation logic for liveness.
//!
//! Both `consistency.rs` (BFS consistency checks) and `checker/eval.rs`
//! (SCC/PEM checks) must run the same ENABLED algorithm to avoid drift.

use crate::enumerate::is_disabled_action_error;
use crate::error::EvalResult;
use crate::eval::{BindingChain, Env, EvalCtx};
use crate::liveness::boolean_contract::expect_live_bool;
use crate::state::State;
use std::sync::Arc;
use tla_core::{ast::Expr, Spanned};

type ExprRef = Arc<Spanned<Expr>>;

/// Part of #liveness-enabled-enum-first: cap on the number of successors the
/// enumeration-first ENABLED evaluation may GENERATE before falling back to
/// the legacy scan-then-uncapped-enumeration order.
///
/// Fairness sub-actions are small (per-binding Next disjuncts: typically 0-6
/// successors), so the cap exists purely to bound the pathological case of an
/// always-enabled action with a huge successor set, where the legacy scan
/// would have found a witness in 1-2 predicate evaluations. The cap changes
/// only WHICH (equivalent) decision procedure runs, never the result.
const ENABLED_ENUM_CAP: usize = 128;

/// Part of #liveness-enabled-enum-first: number of consecutive ENABLED=true
/// outcomes after which a paired leaf switches to the legacy scan-first order
/// (see the adaptive-ordering note in `eval_enabled_uncached`).
const SCAN_FIRST_STREAK: u32 = 8;

/// Part of #liveness-leaf-memo: cap on the number of full action-predicate
/// evaluations an OPTIMISTIC explored-successor scan may spend probing for an
/// ENABLED witness. Enabled actions' witnesses are overwhelmingly found in
/// the first couple of state-changing successors, so a capped probe skims the
/// cheap wins without paying for a full scan on the disabled majority
/// (measured at ~38% of inline-liveness CPU when the pre-enumeration scan was
/// uncapped).
///
/// SOUNDNESS INVARIANT: this cap may be passed to
/// `cached_successor_satisfies_action` ONLY at call sites where a `false`
/// result is provably followed by a COMPLETE authoritative ENABLED decision
/// (the uncapped-enumeration match or `eval_enabled_cp`). Everywhere the scan
/// result is FINAL — or its follow-up is itself incomplete (e.g. the
/// enumeration of an action that leaves primed variables unpinned) — the scan
/// MUST run uncapped (`cap = None`): a capped `false` there declares a
/// genuinely-enabled action disabled, drops its WF/SF obligation, and turns a
/// HOLDING liveness property into a fabricated violation.
const MAX_SCAN_PREDICATE_EVALS: usize = 2;

/// Part of #liveness-enabled-enum-first: share the per-transition results of
/// the paired `ActionPred` leaf (see `EnabledEvalRequest::pred_cache_tag`)
/// derived from a COMPLETE action enumeration.
///
/// `enumerated` is the full successor set of the resolved fairness action
/// `A` from `current` (uncapped, error-free). For any state `t`,
/// `A(current, t)` holds iff `t` is a member of that set — the same
/// enumeration-completeness property the BFS successor set itself is built
/// on, compared via the same fp64 fingerprints the BFS dedup (and the
/// existing `(cur_fp, next_fp, tag)` leaf-result sharing from the leaf-memo
/// wave) already trusts for state identity. The inline action-leaf recorder
/// (`eval_action_leaf_array`) consults these per-state scratchpad entries
/// before falling back to a full AST evaluation of the predicate, so each
/// enabled fairness action is evaluated ONCE per state (the enumeration)
/// instead of once per (state × explored-successor) transition.
///
/// Only the explored successors of `current` (plus the stuttering pair the
/// recorder also queries — `cached` includes it when stuttering is allowed)
/// are populated: exactly the keys the recorder can ask for at this state
/// boundary. The scratchpad is cleared at every inline state boundary and at
/// property-check start, and its keys carry full fingerprints, so entries can
/// never alias another state's transitions (same lifecycle argument as the
/// scan-result sharing this generalizes).
fn populate_pred_results_from_enumeration(
    current: &State,
    cached: &[State],
    enumerated: &[State],
    tag: u32,
) {
    let cur_fp = current.fingerprint();
    // TRUE entries (membership) are unconditionally sound: an emitted
    // successor satisfies the action relation by construction. FALSE entries
    // (non-membership, including the all-false entries of an EMPTY set)
    // require the action to PIN every state variable's primed value on every
    // satisfying branch (statically proven at pair registration; see
    // `action_pins_all_vars`): for an under-specified action, the
    // enumeration's UNCHANGED-completion of free primes can omit
    // predicate-true transitions — and the degenerate no-primed-assignment
    // case (action `TRUE`) enumerates to the EMPTY set despite the relation
    // being total — so non-membership proves nothing there.
    let full = super::checker::full_population_tag(tag);
    if enumerated.is_empty() {
        // Proven-pinning + zero emissions = the enumeration refuted the
        // action outright (a guard was false or a binder domain was empty):
        // A(current, t) = false for every t.
        if full {
            for succ in cached {
                super::checker::insert_scan_pred_result(cur_fp, succ.fingerprint(), tag, false);
            }
        }
        return;
    }
    if enumerated.len() == 1 {
        // Common case (deterministic per-binding action): avoid the set build.
        let enum_fp = enumerated[0].fingerprint();
        for succ in cached {
            let succ_fp = succ.fingerprint();
            if succ_fp == enum_fp {
                super::checker::insert_scan_pred_result(cur_fp, succ_fp, tag, true);
            } else if full {
                super::checker::insert_scan_pred_result(cur_fp, succ_fp, tag, false);
            }
        }
        return;
    }
    let enum_fps: rustc_hash::FxHashSet<crate::state::Fingerprint> =
        enumerated.iter().map(State::fingerprint).collect();
    for succ in cached {
        let succ_fp = succ.fingerprint();
        if enum_fps.contains(&succ_fp) {
            super::checker::insert_scan_pred_result(cur_fp, succ_fp, tag, true);
        } else if full {
            super::checker::insert_scan_pred_result(cur_fp, succ_fp, tag, false);
        }
    }
}

/// TRUE-only predicate-result sharing from an INCOMPLETE enumeration prefix
/// (#liveness-enabled-witness-exit).
///
/// When the witness early-exit stops the ENABLED enumeration, `prefix` holds a
/// PREFIX of the action's successor set. TRUE entries (membership) remain
/// unconditionally sound — each prefix member is a genuine successor
/// satisfying the action relation by construction (the same argument as the
/// TRUE entries in [`populate_pred_results_from_enumeration`]). FALSE entries
/// are NEVER written here: non-membership in a prefix proves nothing, for
/// pinning-proven and unproven actions alike. A missing entry merely sends the
/// inline recorder down its normal full-evaluation path.
fn populate_pred_results_prefix_true(
    current: &State,
    cached: &[State],
    prefix: &[State],
    tag: u32,
) {
    if prefix.is_empty() || cached.is_empty() {
        return;
    }
    let cur_fp = current.fingerprint();
    let prefix_fps: smallvec::SmallVec<[crate::state::Fingerprint; 4]> =
        prefix.iter().map(State::fingerprint).collect();
    for succ in cached {
        let succ_fp = succ.fingerprint();
        if prefix_fps.contains(&succ_fp) {
            super::checker::insert_scan_pred_result(cur_fp, succ_fp, tag, true);
        }
    }
}

// ── Static fairness-subscript analysis (#liveness-enabled-witness-exit) ──

/// Statically resolve a fairness subscript `e` (of `ENABLED <<A>>_e`) to the
/// set of state variables whose values it is composed of.
///
/// Returns `Some(indices)` exactly when `e` provably denotes a (nested) tuple
/// whose leaves are precisely the state variables `indices` — then the
/// subscript's VALUE differs between two states iff one of those variables'
/// values differs, so a per-diff exact value comparison
/// (`SubscriptWatch::Vars`) decides "subscript changed" identically to
/// evaluating `e` on both states and comparing (`has_state_change`), modulo
/// only the fingerprint collisions the subscript value cache already trusts.
/// (The exact-value test is strictly MORE precise than the fp-based compare.)
///
/// Recognized forms, all fail-closed (`None` = keep the current evaluation
/// path, never a wrong verdict):
///   - `<<e1, ..., en>>` where every element resolves recursively (nested
///     tuples flatten: a tuple value differs iff some leaf differs);
///   - a state variable reference (`Ident`/`StateVar` naming a registry var),
///     provided the name is NOT shadowed by a quantifier binding of the leaf
///     (`BindingChain::lookup_by_name`) — a bound name makes the subscript
///     state-dependent through the binding, not the variable;
///   - `LET defs IN body`: definitions are tracked as a name→body scope
///     (innermost wins) so `vars`-style operator references resolve through
///     them; a LET name shadowing a registry variable resolves to the LET body
///     (matching evaluator scoping);
///   - a zero-argument operator reference resolving through the LET scope or
///     `ctx.get_op` (e.g. the classic `vars == <<x, y, z>>`);
///   - `Instance!op` (`ModuleRef`) resolved fail-closed via
///     `resolve_module_ref_body_ast` (e.g. a refinement property's
///     `Sched!vars`); the resolved body is analyzed in the INSTANCED module's
///     scope with a FRESH LET scope (outer LET names must not capture).
///
/// Depth-capped against cyclic operator references; anything else — applies,
/// primed/complex expressions, non-variable leaves — returns `None`.
fn subscript_watch_vars(
    ctx: &EvalCtx,
    bindings: Option<&BindingChain>,
    subscript: &Spanned<Expr>,
) -> Option<super::checker::WatchVarSet> {
    const MAX_DEPTH: u32 = 16;

    fn collect(
        ctx: &EvalCtx,
        bindings: Option<&BindingChain>,
        expr: &Spanned<Expr>,
        let_scope: &mut Vec<(String, Spanned<Expr>)>,
        out: &mut super::checker::WatchVarSet,
        depth: u32,
    ) -> bool {
        if depth >= MAX_DEPTH {
            return false;
        }
        match &expr.node {
            Expr::Tuple(elems) => elems
                .iter()
                .all(|e| collect(ctx, bindings, e, let_scope, out, depth + 1)),
            Expr::Let(defs, body) => {
                if defs.iter().any(|d| !d.params.is_empty()) {
                    // Parameterized locals cannot appear as bare tuple leaves;
                    // rather than reason about partial scopes, fail closed.
                    return false;
                }
                let base_len = let_scope.len();
                for def in defs.iter() {
                    let_scope.push((def.name.node.clone(), def.body.clone()));
                }
                let ok = collect(ctx, bindings, body, let_scope, out, depth + 1);
                let_scope.truncate(base_len);
                ok
            }
            Expr::Ident(name, _) | Expr::StateVar(name, _, _) => {
                let name = name.as_str();
                // A quantifier binding shadowing the name makes the reference
                // non-static — fail closed.
                if bindings.is_some_and(|chain| chain.lookup_by_name(name).is_some()) {
                    return false;
                }
                // LET scope wins over everything (evaluator scoping);
                // innermost definition first.
                if let Some((_, body)) = let_scope.iter().rev().find(|(n, _)| n == name) {
                    let body = body.clone();
                    return collect(ctx, bindings, &body, let_scope, out, depth + 1);
                }
                let registry = &ctx.shared().var_registry;
                if let Some(idx) = registry.get(name) {
                    out.push(idx);
                    return true;
                }
                // Zero-arg operator (e.g. `vars == <<x, y, z>>`): resolve its
                // body in a FRESH LET scope (a module-level body must not see
                // this leaf's LET names).
                let resolved = ctx.resolve_op_name(name);
                if let Some(def) = ctx.get_op(resolved) {
                    if def.params.is_empty() {
                        let body = def.body.clone();
                        let mut fresh_scope = Vec::new();
                        return collect(ctx, bindings, &body, &mut fresh_scope, out, depth + 1);
                    }
                }
                false
            }
            Expr::ModuleRef(target, op_name, args) => {
                // Fail-closed INSTANCE resolution (see resolve_module_ref_body_ast):
                // the body is substitution-applied, so its variable references
                // are the instantiating module's — resolved against the SAME
                // registry — while nested references resolve in the instanced
                // scope ctx.
                match crate::enabled::resolve_module_ref_body_ast(ctx, target, op_name, args) {
                    Some((inner_ctx, body)) => {
                        let mut fresh_scope = Vec::new();
                        collect(
                            &inner_ctx,
                            bindings,
                            &body,
                            &mut fresh_scope,
                            out,
                            depth + 1,
                        )
                    }
                    None => false,
                }
            }
            _ => false,
        }
    }

    let mut out = super::checker::WatchVarSet::new();
    let mut let_scope = Vec::new();
    if collect(ctx, bindings, subscript, &mut let_scope, &mut out, 0) {
        Some(out)
    } else {
        None
    }
}

/// Subscript-support pinning proof (#liveness-enum-exact): decide statically
/// whether the action's own enumeration decides `ENABLED <<A>>_e` EXACTLY, so
/// a complete-enumeration FALSE needs no under-specification rescue scan.
///
/// This is the partial-pinning generalization of the all-vars proof behind
/// FULL_POPULATION_TAGS. Sufficient conditions, both fail-closed:
///
///   1. The subscript `e` statically resolves (`subscript_watch_vars`) to a
///      (nested) tuple of exactly the state variables `W` (non-empty) — so
///      `e`'s value between two states differs iff some variable in `W`
///      differs, and `e` reads nothing outside `W`.
///   2. `action_pins_all_vars(A, W)` — every conjunct of `A`, on every
///      satisfying branch, is either a recognized pin (`x' = v` / `x' \in S` /
///      `UNCHANGED x`) of a variable IN `W`, or a prime-free guard; and every
///      variable of `W` is pinned on every branch. (Called with `vars = W`,
///      the proof REJECTS any primed reference to a variable outside `W` —
///      over-strict for actions that also pin non-subscript variables, but
///      strictly fail-closed.)
///
/// Soundness: by (2), `A`'s primed references constrain exactly variables in
/// `W`; any variable without a primed occurrence is FREE (its next value is
/// unconstrained by `A`). For any satisfying pair `(s, t)`, the
/// UNCHANGED-completion `t̂ = (t|W, s|rest)` therefore also satisfies `A`, is
/// emitted by the enumeration (per-branch pin completeness — the same
/// enumeration-completeness foundation the all-vars proof rests on), and by
/// (1) `e(t̂) = e(t)`. Hence
///   `∃t: A(s,t) ∧ e(t) ≠ e(s)`  ⇔  `∃ emitted t̂: e(t̂) ≠ e(s)`,
/// which is exactly the `any_state_change` scan over the complete enumerated
/// set — a FALSE there is authoritative and the rescue scan is redundant.
///
/// This does NOT authorize FALSE predicate population
/// (`populate_pred_results_from_enumeration`): a BFS transition `(s, t)` may
/// differ from every enumerated successor only OUTSIDE `W` and still satisfy
/// `A` — non-membership proves nothing about the predicate. FALSE population
/// stays gated on the all-vars proof (FULL_POPULATION_TAGS).
pub(crate) fn enabled_enum_decides_exactly(
    action: &Spanned<Expr>,
    subscript: Option<&ExprRef>,
    bindings: Option<&BindingChain>,
    ctx: &EvalCtx,
) -> bool {
    let Some(sub) = subscript else {
        // No subscript: "state change" reads EVERY variable — the required
        // support is the full registry, i.e. exactly the all-vars proof that
        // FULL_POPULATION_TAGS already covers. Nothing weaker to prove here.
        return false;
    };
    let Some(watch) = subscript_watch_vars(ctx, bindings, sub) else {
        return false;
    };
    if watch.is_empty() {
        // Degenerate constant subscript — the `TRUE`-action caveat applies
        // (an action with no primed conjuncts passes pins_rec with empty
        // coverage); fail closed.
        return false;
    }
    let registry = &ctx.shared().var_registry;
    let names: Vec<Arc<str>> = watch
        .iter()
        .map(|idx| Arc::clone(&registry.names()[idx.as_usize()]))
        .collect();
    action_pins_all_vars(action, &names, Some(ctx))
}

/// TRUE-only ENABLED provenance (#3208 redo of #3100): decide statically
/// whether a fairness subscript `e` provably covers EVERY state variable, so
/// "the emitted successor differs from the base state on some variable"
/// (exact per-value comparison) decides "the subscript changed" exactly.
///
/// Delegates to [`subscript_watch_vars`] (fail-closed static resolution of the
/// subscript to a set of state variables) and requires the resolved watch set
/// to cover the FULL registry. Any unresolvable subscript, empty registry, or
/// partial coverage returns `false` — the leaf is then never registered for
/// provenance and keeps the full evaluation path.
pub(crate) fn subscript_covers_all_vars(
    ctx: &EvalCtx,
    bindings: Option<&BindingChain>,
    subscript: &Spanned<Expr>,
) -> bool {
    let Some(watch) = subscript_watch_vars(ctx, bindings, subscript) else {
        return false;
    };
    let num_vars = ctx.shared().var_registry.names().len();
    if num_vars == 0 {
        return false;
    }
    let mut covered = vec![false; num_vars];
    for idx in &watch {
        covered[idx.as_usize()] = true;
    }
    covered.iter().all(|c| *c)
}

/// Opt-in switch for the ENABLED enumeration witness early-exit
/// (#liveness-enabled-witness-exit). See the gating note at the use site for
/// why this defaults OFF.
fn witness_exit_enabled() -> bool {
    static FLAG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *FLAG.get_or_init(|| std::env::var("TY_ENABLED_WITNESS_EXIT").is_ok_and(|v| v == "1"))
}

/// Kill switch for the witnessed-TRUE pair population
/// (#liveness-witness-pair-pop, default ON).
///
/// `TY_DISABLE_WITNESS_PAIR_POPULATION=1` restores the pre-change behavior
/// exactly: a provenance-witnessed ENABLED leaf returns `true` immediately and
/// its paired `ActionPred` leaf falls back to one full AST predicate
/// evaluation per (state × successor) transition. Used by the verdict-identity
/// differential.
pub(super) fn witness_pair_population_enabled() -> bool {
    static FLAG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *FLAG.get_or_init(|| {
        !std::env::var("TY_DISABLE_WITNESS_PAIR_POPULATION").is_ok_and(|v| v == "1")
    })
}

/// #liveness-witness-pair-pop: complete-enumeration predicate population for a
/// provenance-witnessed ENABLED leaf.
///
/// When BFS successor generation already witnessed `ENABLED <<A>>_e = true`
/// for this leaf (see `enabled_provenance::witnessed_true`), the early return
/// skips the from-scratch action enumeration — and with it the per-transition
/// predicate sharing (`populate_pred_results_from_enumeration`) that the
/// enum-first path feeds the paired `ActionPred` leaf with. The paired leaf
/// then re-evaluates the full action predicate once per (state × explored
/// successor) transition: measured at ~77% of Huang's wall clock (10 WF
/// leaves × ~8.5 transitions/state × DyadicRationals arithmetic per eval).
///
/// This helper runs the SAME capped enumeration the enum-first path runs
/// (`enumerate_action_successors_witness_capped` with no watch — bit-identical
/// to the plain capped enumeration), from a context prepared EXACTLY like
/// `eval_enabled_uncached` prepares it (same state snapshot/rebind, same eager
/// binding materialization), and feeds the SAME population routines under the
/// SAME proofs:
///
///   * `Complete` set → [`populate_pred_results_from_enumeration`]: TRUE
///     entries by membership (unconditionally sound — an emitted successor
///     satisfies the action relation by construction); FALSE entries only for
///     tags whose all-vars pinning proof (`action_pins_all_vars`,
///     FULL_POPULATION_TAGS) makes the enumeration's successor set the exact
///     action relation. The caller additionally gates THIS call on
///     `full_population_tag(pred_tag)`, so the enumeration is only spent where
///     the FALSE side pays.
///   * `Capped`/`Witness` prefix → [`populate_pred_results_prefix_true`]:
///     TRUE-only (non-membership in a prefix proves nothing).
///   * Empty complete set or enumeration error → populate NOTHING (fail
///     closed). A genuine witness implies a non-empty relation, so an empty
///     set here would indicate an upstream inconsistency; leaving the
///     scratchpad empty sends every transition down the canonical
///     per-transition evaluation — exactly the pre-change behavior. An
///     enumeration error is likewise swallowed: the witnessed ENABLED verdict
///     is unaffected (identical to today's early return), and any genuine
///     evaluation error surfaces through the canonical per-transition
///     `ActionPred` evaluation, exactly as it does today.
///
/// The ENABLED verdict itself is NEVER derived here — the caller returns the
/// witnessed `true` regardless. This function can only add scratchpad entries
/// whose values are exactly what `eval_action_pred_cached` would compute
/// (`TY_DISABLE_WITNESS_PAIR_POPULATION=1` proves verdict-identity).
///
/// Recording-scope note: this enumeration runs OUTSIDE the BFS arm window
/// (the `arm_state_guard` RAII disarmed when successor generation finished),
/// so its own emissions can never be recorded as provenance witnesses; the
/// `enum_scope` depth guard additionally covers any nested-arm scenario.
#[allow(clippy::too_many_arguments)]
pub(super) fn populate_witnessed_pair(
    ctx_current: &EvalCtx,
    state_prepared: bool,
    current_state: &State,
    action: &ExprRef,
    bindings: Option<&BindingChain>,
    cached_successors: &[State],
    pred_tag: u32,
) {
    let vars: &[Arc<str>] = ctx_current.shared().var_registry.names();
    if vars.is_empty() || cached_successors.is_empty() {
        return;
    }
    let mut eval_ctx = ctx_current.clone();
    // Same per-leaf preparation as `eval_enabled_uncached` (skipped when the
    // shared per-state prepared context is already in that form).
    if !state_prepared {
        *eval_ctx.next_state_mut() = None;
        let state_pairs = eval_ctx.snapshot_state_pairs(vars);
        let _ = eval_ctx.take_state_env();
        let _ = eval_ctx.take_next_state_env();
        for (name, value) in &state_pairs {
            eval_ctx.bind_mut(Arc::clone(name), value.clone());
        }
    }
    if let Some(chain) = bindings {
        for (_name_id, name_str, value) in chain.iter_eager() {
            eval_ctx.bind_mut(Arc::clone(&name_str), value.clone());
        }
        eval_ctx = eval_ctx.with_liveness_bindings(chain);
    }
    match crate::enumerate::enumerate_action_successors_witness_capped(
        &mut eval_ctx,
        action,
        current_state,
        vars,
        ENABLED_ENUM_CAP,
        None,
    ) {
        Ok(crate::enumerate::EnabledEnumOutcome::Complete(successors)) => {
            if !successors.is_empty() {
                populate_pred_results_from_enumeration(
                    current_state,
                    cached_successors,
                    &successors,
                    pred_tag,
                );
            }
        }
        Ok(crate::enumerate::EnabledEnumOutcome::Capped(prefix))
        | Ok(crate::enumerate::EnabledEnumOutcome::Witness(prefix)) => {
            populate_pred_results_prefix_true(current_state, cached_successors, &prefix, pred_tag);
        }
        Err(_) => {} // fail closed: no claim, per-transition evaluation decides
    }
}

/// Resolve (and memoize, keyed by the paired tag) the witness watch for one
/// ENABLED leaf request (#liveness-enabled-witness-exit). `None` = opaque
/// subscript, keep the exact current evaluation path.
fn subscript_watch_for_request(
    request: &EnabledEvalRequest<'_>,
) -> Option<super::checker::WatchVarSet> {
    let sub = request.subscript?;
    match request.pred_cache_tag {
        Some(tag) => super::checker::subscript_watch_cached(tag, || {
            subscript_watch_vars(request.ctx_current, request.bindings, sub)
        }),
        None => subscript_watch_vars(request.ctx_current, request.bindings, sub),
    }
}

// ── Static prime-pinning proof (#liveness-enabled-enum-first) ────────────

/// Conservative static proof that `action`, as an action relation, PINS every
/// state variable's primed value on every satisfying branch — i.e. each
/// satisfying assignment determines a unique successor state, so the
/// enumeration's successor set IS the action relation and non-membership is
/// a sound predicate-FALSE.
///
/// Recognized pinning forms (per conjunct, unioned across `And`):
///   - `x' = e` / `x' \in S` with prime-free `e`/`S`
///   - `UNCHANGED x` / `UNCHANGED <<x, y, ...>>` of direct state variables
///   - `LET local == ... IN body` whose definitions are zero-arg, prime-free,
///     and do not shadow a tracked state var: the proof descends into `body`
///     (the conjuncts live there; see the `Expr::Let` arm and #t1-instance-pinning)
///   - `Instance!Op(args)` whose substitution-applied body pins the var (see
///     the `Expr::ModuleRef` arm and #t1-instance-pinning)
///
/// Branching (`Or`, `IF` with prime-free condition) requires EVERY branch to
/// independently pin all variables. `\E x \in D : body` requires prime-free
/// domains. Every other conjunct must be prime-free (a state-level guard);
/// anything unrecognized — operator references that could hide primes are
/// invisible to `expr_contains_any_prime`, `CASE`, an unsafe/unresolvable `LET`,
/// an unresolvable `ModuleRef`, primed guards — fails the proof. Fail closed:
/// the only consequence is that the leaf keeps per-transition predicate
/// evaluation and the under-specification ENABLED rescue.
///
/// INSTANCE/`ModuleRef` widening (#t1-instance-pinning): a fairness action
/// written through an instance selector (e.g. `Buffer!BeginRead`) is an
/// `Expr::ModuleRef` whose pinning conjuncts live in the *instanced* module's
/// AST, not in this `action` tree. To let such actions become full_population
/// tags, the recursion consults the resolver hook for the resolved
/// (substitution-applied) body and recurses the pinning proof through it.
/// The resolution hook is fail-closed: when it cannot fully resolve the
/// reference (no `ctx`, no body, or the body would itself reach an unresolvable
/// reference) it returns `None`, the `ModuleRef` arm contributes NO coverage,
/// and the surrounding `vars.iter().all(covered)` gate then refuses the proof
/// — exactly the pre-widening outcome. Resolution may therefore only ADD
/// coverage (authorizing FALSE scratchpad entries the enumeration already
/// computes); it can never manufacture a satisfying branch the action does
/// not have, so it can never flip a TRUE/membership result.
///
/// `ctx` is the evaluation context in which `action` was resolved (the
/// production `EvalCtx` of the model checker, carrying instance metadata and
/// operator definitions). When `None` is passed (callers without a ctx), the
/// `ModuleRef` arm cannot resolve and the proof stays exactly as fail-closed as
/// the pre-widening code.
pub(crate) fn action_pins_all_vars(
    action: &Spanned<Expr>,
    vars: &[Arc<str>],
    ctx: Option<&EvalCtx>,
) -> bool {
    /// Depth cap on transitive INSTANCE/ModuleRef body resolution, to bound the
    /// pathological case of a cyclic operator reference (fail-closed: hitting
    /// the cap refuses the proof for the offending reference, never asserts
    /// coverage). Real fairness actions nest at most a couple of INSTANCE
    /// levels (e.g. `Sched!Allocator`'s `Sched!Schedule`).
    const MAX_MODULE_REF_DEPTH: u32 = 16;

    fn var_name(expr: &Expr) -> Option<&str> {
        match expr {
            Expr::Ident(name, _) => Some(name.as_str()),
            Expr::StateVar(name, _, _) => Some(name.as_str()),
            _ => None,
        }
    }

    fn unchanged_targets<'a>(
        expr: &'a Expr,
        vars: &rustc_hash::FxHashSet<&'a str>,
        covered: &mut rustc_hash::FxHashSet<&'a str>,
    ) -> bool {
        match expr {
            Expr::Tuple(elems) => elems
                .iter()
                .all(|e| unchanged_targets(&e.node, vars, covered)),
            _ => match var_name(expr) {
                Some(name) if vars.contains(name) => {
                    covered.insert(name);
                    true
                }
                // UNCHANGED of an operator reference (e.g. `UNCHANGED vars`)
                // hides its targets — fail closed.
                _ => false,
            },
        }
    }

    fn pins_rec<'a>(
        expr: &'a Expr,
        vars: &rustc_hash::FxHashSet<&'a str>,
        covered: &mut rustc_hash::FxHashSet<&'a str>,
        ctx: Option<&EvalCtx>,
        depth: u32,
    ) -> bool {
        match expr {
            Expr::And(a, b) => {
                pins_rec(&a.node, vars, covered, ctx, depth)
                    && pins_rec(&b.node, vars, covered, ctx, depth)
            }
            Expr::Exists(bounds, body) => {
                bounds.iter().all(|b| {
                    b.domain
                        .as_ref()
                        .is_some_and(|d| !crate::enumerate::expr_contains_any_prime(&d.node))
                }) && pins_rec(&body.node, vars, covered, ctx, depth)
            }
            Expr::Or(a, b) => {
                if branch_pins_all(&a.node, vars, ctx, depth)
                    && branch_pins_all(&b.node, vars, ctx, depth)
                {
                    covered.extend(vars.iter().copied());
                    true
                } else {
                    false
                }
            }
            Expr::If(c, t, e) => {
                if !crate::enumerate::expr_contains_any_prime(&c.node)
                    && branch_pins_all(&t.node, vars, ctx, depth)
                    && branch_pins_all(&e.node, vars, ctx, depth)
                {
                    covered.extend(vars.iter().copied());
                    true
                } else {
                    false
                }
            }
            Expr::Eq(l, r) | Expr::In(l, r) => {
                if let Expr::Prime(inner) = &l.node {
                    match var_name(&inner.node) {
                        Some(name)
                            if vars.contains(name)
                                && !crate::enumerate::expr_contains_any_prime(&r.node) =>
                        {
                            covered.insert(name);
                            true
                        }
                        _ => false,
                    }
                } else {
                    // Plain guard: must be prime-free on both sides.
                    !crate::enumerate::expr_contains_any_prime(&l.node)
                        && !crate::enumerate::expr_contains_any_prime(&r.node)
                }
            }
            Expr::Unchanged(inner) => unchanged_targets(&inner.node, vars, covered),
            // LET widening (#t1-instance-pinning): a fairness action body is
            // frequently a `LET local == ... IN /\ <pinning conjuncts>` (e.g.
            // Disruptor's `BeginRead(reader) == LET next == ... index == ... IN
            // ...`). The pinning conjuncts live in the LET BODY, so the proof
            // must descend into it. This is sound to do — contributing the body's
            // coverage — only when the LET cannot affect which state variables
            // are pinned:
            //   - every LET definition is a zero-arg, PRIME-FREE local operator
            //     (it can introduce no `x' = e` of its own and appears only in
            //     prime-free guard/RHS positions), and
            //   - no LET name SHADOWS a tracked state variable in `vars` (a
            //     shadow would make `x` resolve to the local op, so `x' = e`
            //     coverage attributed to the state var would be unsound).
            // Any LET that violates either condition falls through to the
            // prime-free fail-closed test below (no coverage), exactly the
            // pre-widening outcome.
            Expr::Let(defs, body) => {
                let safe = defs.iter().all(|d| {
                    d.params.is_empty()
                        && !d.contains_prime
                        && !crate::enumerate::expr_contains_any_prime(&d.body.node)
                        && !vars.contains(d.name.node.as_str())
                });
                if safe {
                    pins_rec(&body.node, vars, covered, ctx, depth)
                } else {
                    false
                }
            }
            // INSTANCE/ModuleRef widening (#t1-instance-pinning): the pinning
            // conjuncts of `Instance!Op` live in the instanced module's AST, not
            // in this `expr` tree, so the prime-free fall-through below would
            // contribute NO coverage and fail the proof. Try to resolve the
            // reference's substitution-applied body and recurse the pinning
            // proof through it instead. Resolution is fail-closed (see
            // `resolve_module_ref_body`): an unresolved reference yields `None`,
            // we fall through to the prime-free guard test (no coverage), and
            // the `vars.iter().all(covered)` gate then refuses the proof — the
            // exact pre-widening outcome. A resolved body can only ADD coverage,
            // never invent a satisfying branch, so this stays additive and can
            // never flip a TRUE/membership result.
            Expr::ModuleRef(..) => {
                if let Some((inner_ctx, body)) = resolve_module_ref_body(expr, ctx, depth) {
                    // The resolved body is freshly owned (shorter lifetime than
                    // `'a`), so run a self-contained proof that reports its
                    // covered var names by value, then re-key them into `'a`
                    // via the caller's `vars` set so they merge into `covered`.
                    // Nested ModuleRefs in `body` resolve against `inner_ctx`
                    // (the instanced-module scope), and `depth + 1` bounds the
                    // transitive resolution.
                    let body_covered =
                        body_pins_cover(&body.node, vars, Some(&inner_ctx), depth + 1);
                    for name in vars.iter().copied() {
                        if body_covered.iter().any(|c| c.as_str() == name) {
                            covered.insert(name);
                        }
                    }
                    // Conjuncts unrelated to pinning (e.g. an instanced guard)
                    // are not separately validated here; the body proof already
                    // required the whole body to pin every var it claims to
                    // cover, and any var it did NOT cover is left to the
                    // surrounding all-covered gate to reject. Treat the arm as a
                    // recognized form so the prime-free fall-through does not
                    // also (mis)classify the reference as a bare guard.
                    true
                } else {
                    // Unresolvable reference: keep the historical fail-closed
                    // behavior (a prime-free-by-args ModuleRef passes as a guard
                    // contributing no coverage; a primed-arg one fails outright).
                    !crate::enumerate::expr_contains_any_prime(expr)
                }
            }
            other => !crate::enumerate::expr_contains_any_prime(other),
        }
    }

    fn branch_pins_all(
        expr: &Expr,
        vars: &rustc_hash::FxHashSet<&str>,
        ctx: Option<&EvalCtx>,
        depth: u32,
    ) -> bool {
        let mut covered = rustc_hash::FxHashSet::default();
        pins_rec(expr, vars, &mut covered, ctx, depth) && vars.iter().all(|v| covered.contains(v))
    }

    /// Run the pinning proof over a resolved INSTANCE/ModuleRef body (a
    /// freshly-owned AST whose lifetime is shorter than `vars`'), returning the
    /// set of `vars` names the body provably pins. Allocates owned `String`
    /// keys so the proof is independent of `vars`' borrow lifetime; the caller
    /// re-keys the result back into `vars`.
    ///
    /// This delegates to `pins_rec` against a local var set built from the same
    /// names, so the recognized-form set and fail-closed semantics are
    /// identical to the top-level proof (including transitively recursing into
    /// any nested INSTANCE references via `resolve_module_ref_body`).
    fn body_pins_cover(
        body: &Expr,
        vars: &rustc_hash::FxHashSet<&str>,
        ctx: Option<&EvalCtx>,
        depth: u32,
    ) -> Vec<String> {
        let local_set: rustc_hash::FxHashSet<&str> = vars.iter().copied().collect();
        let mut local_covered: rustc_hash::FxHashSet<&str> = rustc_hash::FxHashSet::default();
        if pins_rec(body, &local_set, &mut local_covered, ctx, depth) {
            local_covered.iter().map(|s| (*s).to_owned()).collect()
        } else {
            Vec::new()
        }
    }

    /// Resolution hook for INSTANCE operator references
    /// (#t1-instance-pinning). Given an `Expr::ModuleRef` action leaf and the
    /// `EvalCtx` it was resolved in, return its substitution-applied body
    /// (paired with the instanced-module scope `EvalCtx` for resolving any
    /// nested reference) so the pinning proof can recurse through it, or `None`
    /// when the reference cannot be FULLY resolved here.
    ///
    /// FAIL-CLOSED CONTRACT: returning `None` is always sound — it preserves
    /// the pre-widening behavior (no coverage from the reference). The hook may
    /// only return `Some(body)` for a body that is the genuine, fully
    /// substitution-applied relation of the reference; any residual
    /// unresolvable selector inside it must make the hook return `None` for the
    /// whole reference, so a partially-resolved body can never claim coverage.
    /// `resolve_module_ref_body_ast` upholds this (unknown instance, op-not-
    /// found, arity mismatch, or a substitution-only fallback all yield `None`).
    ///
    /// Without a `ctx`, or once the transitive depth cap is reached, returns
    /// `None`, leaving the proof exactly as fail-closed as the pre-widening code.
    fn resolve_module_ref_body(
        expr: &Expr,
        ctx: Option<&EvalCtx>,
        depth: u32,
    ) -> Option<(EvalCtx, Spanned<Expr>)> {
        debug_assert!(matches!(expr, Expr::ModuleRef(..)));
        if depth >= MAX_MODULE_REF_DEPTH {
            return None;
        }
        let ctx = ctx?;
        let Expr::ModuleRef(target, op_name, args) = expr else {
            return None;
        };
        crate::enabled::resolve_module_ref_body_ast(ctx, target, op_name, args)
    }

    if vars.is_empty() {
        return false;
    }
    let var_set: rustc_hash::FxHashSet<&str> = vars.iter().map(|v| v.as_ref()).collect();
    branch_pins_all(&action.node, &var_set, ctx, 0)
}

pub(super) struct EnabledEvalRequest<'a> {
    pub(super) ctx_current: &'a EvalCtx,
    pub(super) current_state: &'a State,
    pub(super) action: &'a ExprRef,
    pub(super) bindings: Option<&'a BindingChain>,
    pub(super) require_state_change: bool,
    pub(super) subscript: Option<&'a ExprRef>,
    pub(super) cached_successors: &'a [State],
    /// Part of #liveness-leaf-memo: when `Some(tag)`, `action` is the SAME
    /// resolved expression (with the same bindings) as the `ActionPred` leaf
    /// carrying `tag` (paired by `EnabledActionGroup::action_pred_tag`), and
    /// the successor scan's per-(current, successor) predicate results are
    /// shared with the inline action-leaf recorder through the
    /// `(cur_fp, next_fp, tag)` leaf result cache.
    pub(super) pred_cache_tag: Option<u32>,
    /// Part of #liveness-enabled-batch-ctx: when `true`, `ctx_current` is
    /// already the per-state ENABLED evaluation context produced by
    /// [`prepare_enabled_ctx`] (current-state variables bound into `env`,
    /// `state_env`/`next_state_env`/`next_state` cleared). The per-leaf
    /// state-variable snapshot+rebind (708-716) is then skipped, since it would
    /// reproduce a context that is already in that exact form. Verdict-neutral:
    /// the enumeration sees the identical context either way (proven by the
    /// `TY_DISABLE_LIVENESS_ENABLED_BATCH=1` kill switch, which forces the
    /// per-leaf path). `false` on every non-batched call site preserves the
    /// legacy per-leaf preparation exactly.
    pub(super) state_prepared: bool,
}

/// Kill switch for the per-state batched ENABLED context (default ON).
///
/// `TY_DISABLE_LIVENESS_ENABLED_BATCH=1` forces every ENABLED leaf back onto
/// the per-leaf `snapshot_state_pairs`+rebind preparation, proving the batched
/// path is verdict-identical.
pub(super) fn enabled_batch_ctx_enabled() -> bool {
    static FLAG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *FLAG
        .get_or_init(|| !std::env::var("TY_DISABLE_LIVENESS_ENABLED_BATCH").is_ok_and(|v| v == "1"))
}

/// Part of #liveness-enabled-batch-ctx: build the shared per-state ENABLED
/// evaluation context ONCE for a batch of ENABLED leaves at the same current
/// state.
///
/// This reproduces EXACTLY the context transformation `eval_enabled_uncached`
/// applies per leaf before enumeration (see lines under `if !state_prepared`):
/// clear `next_state`, snapshot the current-state variable values (from the
/// `state_env` overlay or `env`), clear the array-backed `state_env` /
/// `next_state_env` overlays, and rebind those values directly into `env`.
///
/// Passing the result as `ctx_current` with `state_prepared = true` is
/// therefore observationally identical to the per-leaf preparation, while the
/// state-variable snapshot+rebind (which is invariant across all ENABLED leaves
/// of one state) runs ONCE instead of once per leaf. Only the per-leaf
/// quantifier bindings (`c`/`m`/...) differ and are still applied per leaf.
pub(super) fn prepare_enabled_ctx(ctx: &EvalCtx) -> EvalCtx {
    let mut eval_ctx = ctx.clone();
    let vars: Vec<Arc<str>> = ctx.shared().var_registry.names().to_vec();
    if vars.is_empty() {
        return eval_ctx;
    }
    *eval_ctx.next_state_mut() = None;
    let state_pairs = eval_ctx.snapshot_state_pairs(&vars);
    let _ = eval_ctx.take_state_env();
    let _ = eval_ctx.take_next_state_env();
    for (name, value) in &state_pairs {
        eval_ctx.bind_mut(Arc::clone(name), value.clone());
    }
    eval_ctx
}

pub(super) fn eval_enabled_uncached<F>(
    request: EnabledEvalRequest<'_>,
    mut eval_subscript_changed: F,
) -> EvalResult<bool>
where
    F: FnMut(&EvalCtx, &State, &State, &ExprRef) -> EvalResult<bool>,
{
    // Borrow the registry's var-name slice directly instead of cloning it into a
    // fresh `Vec<Arc<str>>` on every call. `eval_enabled_uncached` runs once per
    // (state, fairness-leaf) — ~2M times on AllocatorImplementation — and the
    // owned copy allocated a Vec + bumped every name's Arc refcount for a value
    // that is constant for the whole run. The slice borrows `request.ctx_current`
    // (immutable, alive for the call); the mutated `eval_ctx` below is a separate
    // clone, so there is no aliasing. Verdict-neutral (same names, same order).
    let vars: &[Arc<str>] = request.ctx_current.shared().var_registry.names();
    if !vars.is_empty() {
        let mut eval_ctx = request.ctx_current.clone();
        // Part of #liveness-enabled-batch-ctx: when `ctx_current` is already the
        // per-state prepared context (state vars in `env`, overlays cleared;
        // built once per state by `prepare_enabled_ctx`), the per-leaf snapshot
        // + rebind below would only reproduce that same form, so skip it. The
        // clone above still isolates this leaf's quantifier bindings + the
        // enumeration's transient `next_state` mutations from the shared
        // prepared context. Kill switch `TY_DISABLE_LIVENESS_ENABLED_BATCH=1`
        // never sets `state_prepared`, forcing this block for every leaf.
        if !request.state_prepared {
            *eval_ctx.next_state_mut() = None;
            // Part of #2755: snapshot state pairs BEFORE clearing state_env so that
            // eval_enabled_cp's snapshot_state_pairs finds them via env.
            let state_pairs = eval_ctx.snapshot_state_pairs(vars);
            let _ = eval_ctx.take_state_env();
            let _ = eval_ctx.take_next_state_env();
            for (name, value) in &state_pairs {
                eval_ctx.bind_mut(Arc::clone(name), value.clone());
            }
        }
        // Part of #2895: Apply liveness bindings via BindingChain (shadows state vars).
        // BindingChain bindings survive closure/function entry, no write_to_env needed.
        //
        // Soundness (#liveness-wf): also materialize the quantifier bindings (e.g.
        // the `car` in `WF_vars(MoveOutsideBridge(car))`) into the ctx ENV. The
        // from-scratch enumeration fallback below routes through the BFS
        // `enumerate_unified` engine, which resolves free identifiers via the env
        // (not the liveness BindingChain). Without this, a quantifier-bound action
        // enumerates zero successors and is wrongly reported disabled, flipping
        // `~ENABLED` to true and making WF/SF trivially satisfiable.
        if let Some(chain) = request.bindings {
            for (_name_id, name_str, value) in chain.iter_eager() {
                eval_ctx.bind_mut(Arc::clone(&name_str), value.clone());
            }
            eval_ctx = eval_ctx.with_liveness_bindings(chain);
        }

        if request.require_state_change {
            // Part of #liveness-enabled-enum-first: ADAPTIVE ordering between
            // two EXACT decision procedures (only performance can differ):
            //
            //   scan-first  = the legacy order (explored-successor predicate
            //                 scan, then authoritative enumeration), fastest
            //                 for persistently-ENABLED leaves like
            //                 `WF_vars(Next)` where the first probe witnesses
            //                 and a full Next re-enumeration per state would
            //                 roughly double successor-generation work
            //                 (measured: TokenRing +14%, cf1s_folklore +12%
            //                 under unconditional enumeration-first);
            //   enum-first  = enumeration directly (below), fastest for the
            //                 disabled majority on fairness-heavy specs AND
            //                 the only order that yields the complete
            //                 successor set for predicate-result sharing.
            //
            // A paired tag switches to scan-first after SCAN_FIRST_STREAK
            // consecutive ENABLED=true outcomes; a failed probe or a false
            // outcome resets it to enum-first.
            let scan_first_tag = request
                .pred_cache_tag
                .filter(|&tag| super::checker::enabled_true_streak(tag) >= SCAN_FIRST_STREAK);
            if let Some(tag) = scan_first_tag {
                // Capped probe: SOUND here (and only on paths like this one) —
                // a `false` result always falls through to the authoritative
                // enumeration match below; the cap can never be the final
                // ENABLED verdict.
                if cached_successor_satisfies_action(
                    &eval_ctx,
                    request.current_state,
                    request.action,
                    request.bindings,
                    true,
                    request.subscript,
                    request.cached_successors,
                    request.pred_cache_tag,
                    Some(MAX_SCAN_PREDICATE_EVALS),
                    &mut eval_subscript_changed,
                )? {
                    return Ok(true);
                }
                // Probe failed to witness: stop paying for it on this tag and
                // fall through to the authoritative enumeration-first path.
                super::checker::reset_enabled_streak(tag);
            }

            // Part of #liveness-enabled-enum-first: enumeration-FIRST.
            //
            // The from-scratch action enumeration is the authoritative ENABLED
            // algorithm (it already ran for every leaf the explored-successor
            // scan failed to witness — i.e. for the disabled majority on
            // fairness-heavy specs). Profiling (AllocatorImplementation,
            // symbol-rich `sample`) attributed ~34% of BFS worker CPU to the
            // scan's per-(leaf × successor) action-predicate AST evaluations,
            // versus ~10% for the enumeration itself. Running the enumeration
            // FIRST removes the scan from the hot path entirely:
            //
            //   - DISABLED leaves: the enumeration refutes via the action guard
            //     (zero successors) — the capped scan's predicate evals were
            //     pure pre-waste.
            //   - ENABLED leaves: the enumeration's own successors witness
            //     satisfiability directly (`any_state_change`), replacing the
            //     scan witness.
            //
            // ENABLED-exactness: the decision procedure is byte-identical to
            // the previous authoritative fallback (`enumerate_action_successors`
            // + `any_state_change`, including the disabled-action-error =>
            // `false` mapping). The removed scan was a pure optimization for
            // the ENABLED-true case whose positive answers the enumeration
            // subsumes (a scan witness is an explored successor satisfying the
            // action with the subscript changed; enumeration completeness —
            // the property the BFS successor set itself is built on — yields
            // the same witness). The scan is preserved on the two cold paths
            // where the enumeration result is unavailable: a capped successor
            // set and a non-disabled enumeration error (matching the legacy
            // order where a scan witness short-circuited before the
            // enumeration could fail).
            // Part of #liveness-enabled-witness-exit: when the fairness
            // subscript statically resolves to a (nested) tuple of state
            // variables (or there is no subscript), the enumeration sink can
            // decide "this successor changes the subscript" per generated
            // diff by exact value comparison — and ENABLED needs only ONE
            // such witness, so generation can stop there.
            //
            // DEFAULT OFF (opt-in via TY_ENABLED_WITNESS_EXIT=1): measured on
            // the liveness cluster, the early exit is a net LOSS (Huang 2x,
            // AllocatorImplementation +7%). The complete enumeration is
            // dual-purpose: its full successor set feeds
            // `populate_pred_results_from_enumeration`, whose TRUE+FALSE
            // per-transition entries (FALSE authorized by the pinning proof)
            // let the inline action-leaf recorder skip a full AST predicate
            // evaluation per (leaf x transition). Cutting the enumeration at
            // the first witness forfeits that coverage, and the recorder's
            // per-transition re-evaluations cost far more than the saved
            // enumeration tail. The mechanism is kept (soundness-complete,
            // verdict-identical) for enumeration-dominated shapes and for
            // future pairing with a cheaper recorder feed.
            let watch_vars = if witness_exit_enabled() {
                subscript_watch_for_request(&request)
            } else {
                None
            };
            let watch = if !witness_exit_enabled() {
                None
            } else {
                match (request.subscript, watch_vars.as_ref()) {
                    (None, _) => Some(crate::enumerate::SubscriptWatch::AnyVar),
                    (Some(_), Some(indices)) => {
                        Some(crate::enumerate::SubscriptWatch::Vars(indices.as_slice()))
                    }
                    (Some(_), None) => None,
                }
            };
            let watch_active = watch.is_some();
            match crate::enumerate::enumerate_action_successors_witness_capped(
                &mut eval_ctx,
                request.action,
                request.current_state,
                vars,
                ENABLED_ENUM_CAP,
                watch,
            ) {
                Ok(crate::enumerate::EnabledEnumOutcome::Witness(prefix)) => {
                    // Sound early exit: a generated successor changed the
                    // watched subscript — a genuine `<<A>>_e` witness, the
                    // same verdict `any_state_change` reaches on the complete
                    // set (it, too, stops at the first witness). Only TRUE
                    // membership entries are shared with the paired leaf; the
                    // prefix is incomplete, so FALSE entries are forbidden
                    // (see populate_pred_results_prefix_true).
                    if let Some(tag) = request.pred_cache_tag {
                        populate_pred_results_prefix_true(
                            request.current_state,
                            request.cached_successors,
                            &prefix,
                            tag,
                        );
                        super::checker::note_enabled_outcome(tag, true);
                    }
                    return Ok(true);
                }
                Ok(crate::enumerate::EnabledEnumOutcome::Complete(successors)) => {
                    // Complete successor set: share per-transition action-predicate
                    // results with the paired ActionPred leaf (fp membership).
                    // An EMPTY set is populated too — membership in the empty
                    // set is `false` for every transition (the action relation
                    // is unsatisfiable from this state), which spares the
                    // property-plan recorder a full AST evaluation per
                    // transition for every DISABLED fairness action.
                    if let Some(tag) = request.pred_cache_tag {
                        populate_pred_results_from_enumeration(
                            request.current_state,
                            request.cached_successors,
                            &successors,
                            tag,
                        );
                    }
                    let mut result = if watch_active {
                        // The sink applied the witness test to EVERY generated
                        // diff of this complete set and refuted it; by the
                        // watch equivalence (see subscript_watch_vars) the
                        // any_state_change scan over the same set is false.
                        false
                    } else {
                        any_state_change(
                            &eval_ctx,
                            request.current_state,
                            request.subscript,
                            &successors,
                            &mut eval_subscript_changed,
                        )?
                    };
                    // #liveness-enum-exact: the rescue is skipped both for
                    // all-vars pinning (full_population_tag) and for the
                    // weaker subscript-support pinning (enum_exact_tag) —
                    // either proof makes the complete enumeration's FALSE
                    // authoritative for ENABLED (see
                    // enabled_enum_decides_exactly).
                    let pinning_proven = request.pred_cache_tag.is_some_and(|t| {
                        super::checker::full_population_tag(t) || super::checker::enum_exact_tag(t)
                    });
                    if !result && !pinning_proven {
                        // Under-specification rescue: an action that leaves
                        // some primed variables FREE is not fully decided by
                        // its enumeration — free primes are completed as
                        // UNCHANGED (or, for the degenerate `TRUE`-like case
                        // with NO primed assignments at all, the enumeration
                        // emits NOTHING), hiding genuine state-changing
                        // witnesses. The legacy explored-successor predicate
                        // scan evaluates the action over CONCRETE candidate
                        // pairs and is exact for such actions; running it on
                        // every unproven FALSE outcome restores the
                        // pre-reorder decision envelope (scan ∪ enumeration).
                        // Actions whose prime-pinning is statically proven
                        // (see action_pins_all_vars) need no rescue: their
                        // enumeration IS the relation.
                        //
                        // UNCAPPED: this rescue is the FINAL decision for the
                        // unpinnable action — nothing authoritative follows a
                        // `false` here, so every explored successor must be
                        // probed (a capped `false` would declare an enabled
                        // action disabled and fabricate a WF/SF violation).
                        result = cached_successor_satisfies_action(
                            &eval_ctx,
                            request.current_state,
                            request.action,
                            request.bindings,
                            true,
                            request.subscript,
                            request.cached_successors,
                            request.pred_cache_tag,
                            None,
                            &mut eval_subscript_changed,
                        )?;
                    }
                    if let Some(tag) = request.pred_cache_tag {
                        super::checker::note_enabled_outcome(tag, result);
                    }
                    return Ok(result);
                }
                Ok(crate::enumerate::EnabledEnumOutcome::Capped(successors)) => {
                    // Capped: the prefix is incomplete, but each member is a
                    // genuine successor — a state-changing one is a sound
                    // ENABLED witness. No predicate-result sharing (the set
                    // must not be treated as complete).
                    //
                    // With an active watch the sink already tested every
                    // prefix diff and found no witness, so the re-scan is
                    // skipped as refuted (watch equivalence).
                    let prefix_witness = if watch_active {
                        false
                    } else {
                        any_state_change(
                            &eval_ctx,
                            request.current_state,
                            request.subscript,
                            &successors,
                            &mut eval_subscript_changed,
                        )?
                    };
                    if prefix_witness {
                        if let Some(tag) = request.pred_cache_tag {
                            super::checker::note_enabled_outcome(tag, true);
                        }
                        return Ok(true);
                    }
                    // Legacy order for the undecided remainder: scan, then
                    // UNCAPPED authoritative enumeration.
                    //
                    // UNCAPPED scan: the follow-up enumeration below is only
                    // authoritative for prime-pinning actions — for an
                    // unpinnable action (free primes UNCHANGED-completed) it
                    // can miss genuine witnesses, so this scan is part of the
                    // final decision envelope and must probe every explored
                    // successor.
                    if cached_successor_satisfies_action(
                        &eval_ctx,
                        request.current_state,
                        request.action,
                        request.bindings,
                        true,
                        request.subscript,
                        request.cached_successors,
                        request.pred_cache_tag,
                        None,
                        &mut eval_subscript_changed,
                    )? {
                        if let Some(tag) = request.pred_cache_tag {
                            super::checker::note_enabled_outcome(tag, true);
                        }
                        return Ok(true);
                    }
                    let successors = match crate::enumerate::enumerate_action_successors(
                        &mut eval_ctx,
                        request.action,
                        request.current_state,
                        vars,
                    ) {
                        Ok(successors) => successors,
                        Err(e) if is_disabled_action_error(&e) => {
                            if let Some(tag) = request.pred_cache_tag {
                                super::checker::note_enabled_outcome(tag, false);
                            }
                            return Ok(false);
                        }
                        Err(e) => return Err(e),
                    };
                    let result = any_state_change(
                        &eval_ctx,
                        request.current_state,
                        request.subscript,
                        &successors,
                        &mut eval_subscript_changed,
                    )?;
                    if let Some(tag) = request.pred_cache_tag {
                        super::checker::note_enabled_outcome(tag, result);
                    }
                    return Ok(result);
                }
                Err(e) if is_disabled_action_error(&e) => {
                    // Same pre-reorder envelope as the Ok(false) path: the
                    // legacy scan ran BEFORE the enumeration could report
                    // disabled, so give it its chance on unproven leaves.
                    // #liveness-enum-exact: either pinning proof (all-vars or
                    // subscript-support) makes the disabled refutation
                    // authoritative; see enabled_enum_decides_exactly.
                    let pinning_proven = request.pred_cache_tag.is_some_and(|t| {
                        super::checker::full_population_tag(t) || super::checker::enum_exact_tag(t)
                    });
                    let result = if pinning_proven {
                        false
                    } else {
                        // UNCAPPED: like the Ok(false) rescue above, this scan
                        // is the FINAL decision for the unproven leaf — a
                        // capped `false` would be returned as the verdict.
                        cached_successor_satisfies_action(
                            &eval_ctx,
                            request.current_state,
                            request.action,
                            request.bindings,
                            true,
                            request.subscript,
                            request.cached_successors,
                            request.pred_cache_tag,
                            None,
                            &mut eval_subscript_changed,
                        )?
                    };
                    if let Some(tag) = request.pred_cache_tag {
                        super::checker::note_enabled_outcome(tag, result);
                    }
                    return Ok(result);
                }
                Err(e) => {
                    // Preserve the legacy behavior on enumeration errors: a
                    // scan witness used to short-circuit BEFORE the enumeration
                    // could fail, so give the scan its chance before
                    // propagating the error.
                    //
                    // UNCAPPED: no authoritative decision follows a `false`
                    // here (the error propagates), so probe every explored
                    // successor — a late witness resolves the leaf instead of
                    // failing the check.
                    if cached_successor_satisfies_action(
                        &eval_ctx,
                        request.current_state,
                        request.action,
                        request.bindings,
                        true,
                        request.subscript,
                        request.cached_successors,
                        request.pred_cache_tag,
                        None,
                        &mut eval_subscript_changed,
                    )? {
                        return Ok(true);
                    }
                    return Err(e);
                }
            }
        }

        // Capped probe: SOUND — a `false` always falls through to
        // `eval_enabled_cp` below, the complete authoritative ENABLED
        // decision (constraint propagation; exact witness-or-refute).
        if cached_successor_satisfies_action(
            &eval_ctx,
            request.current_state,
            request.action,
            request.bindings,
            false,
            request.subscript,
            request.cached_successors,
            request.pred_cache_tag,
            Some(MAX_SCAN_PREDICATE_EVALS),
            &mut eval_subscript_changed,
        )? {
            return Ok(true);
        }

        return crate::enabled::eval_enabled_cp(&mut eval_ctx, request.action, vars);
    }

    eval_enabled_fallback(&request, &mut eval_subscript_changed)
}

/// Scan the explored successors of `current_state` for one that satisfies the
/// action predicate (an ENABLED witness).
///
/// `cap` bounds the number of full action-predicate evaluations. Per the
/// SOUNDNESS INVARIANT on [`MAX_SCAN_PREDICATE_EVALS`], callers may pass
/// `Some(..)` ONLY when a `false` result is always followed by a complete
/// authoritative ENABLED decision; whenever the scan result is final (or its
/// follow-up is incomplete), `cap` MUST be `None` so every explored successor
/// is probed before the action is declared disabled.
#[allow(clippy::too_many_arguments)]
fn cached_successor_satisfies_action<F>(
    eval_ctx: &EvalCtx,
    current_state: &State,
    action: &ExprRef,
    bindings: Option<&BindingChain>,
    require_state_change: bool,
    subscript: Option<&ExprRef>,
    cached_successors: &[State],
    pred_cache_tag: Option<u32>,
    cap: Option<usize>,
    eval_subscript_changed: &mut F,
) -> EvalResult<bool>
where
    F: FnMut(&EvalCtx, &State, &State, &ExprRef) -> EvalResult<bool>,
{
    if cached_successors.is_empty() {
        return Ok(false);
    }

    // The action-predicate context (base ctx + current-state vars + quantifier
    // bindings) is identical for every candidate successor — build it ONCE per
    // call instead of once per successor. Only the next-state env varies per
    // successor; the SUBST_CACHE clear in `transition_satisfies_action`
    // (next-state-keyed, see the soundness note there) is preserved per
    // successor.
    let action_ctx = build_action_predicate_ctx(eval_ctx, current_state, bindings);
    let current_fp = current_state.fingerprint();
    let mut predicate_evals = 0usize;

    for succ_state in cached_successors {
        if cap.is_some_and(|c| predicate_evals >= c) {
            return Ok(false);
        }
        if require_state_change
            && !has_state_change(
                eval_ctx,
                current_state,
                succ_state,
                subscript,
                eval_subscript_changed,
            )?
        {
            continue;
        }

        // Part of #liveness-leaf-memo: when this predicate is a paired
        // ActionPred leaf (see `EnabledEvalRequest::pred_cache_tag`), record
        // the per-(current, successor) result in the per-state scan
        // scratchpad so the inline action-leaf recorder skips the duplicate
        // AST evaluation. Only genuine Ok results are recorded — the
        // historical "disabled action error ⇒ skip this successor" handling
        // below is preserved bit-for-bit and an error is never laundered
        // into a recorded boolean.
        predicate_evals += 1;
        let result = transition_satisfies_action(&action_ctx, succ_state, action);
        if let (Some(tag), Ok(value)) = (pred_cache_tag, &result) {
            super::checker::insert_scan_pred_result(
                current_fp,
                succ_state.fingerprint(),
                tag,
                *value,
            );
        }
        match result {
            Ok(true) => return Ok(true),
            Ok(false) => {}
            Err(e) if is_disabled_action_error(&e) => {}
            Err(e) => return Err(e),
        }
    }

    Ok(false)
}

fn any_state_change<F>(
    eval_ctx: &EvalCtx,
    current_state: &State,
    subscript: Option<&ExprRef>,
    successors: &[State],
    eval_subscript_changed: &mut F,
) -> EvalResult<bool>
where
    F: FnMut(&EvalCtx, &State, &State, &ExprRef) -> EvalResult<bool>,
{
    for succ in successors {
        if has_state_change(
            eval_ctx,
            current_state,
            succ,
            subscript,
            eval_subscript_changed,
        )? {
            return Ok(true);
        }
    }
    Ok(false)
}

fn eval_enabled_fallback<F>(
    request: &EnabledEvalRequest<'_>,
    eval_subscript_changed: &mut F,
) -> EvalResult<bool>
where
    F: FnMut(&EvalCtx, &State, &State, &ExprRef) -> EvalResult<bool>,
{
    let mut eval_ctx = request.ctx_current.clone();
    *eval_ctx.next_state_mut() = None;
    let _ = eval_ctx.take_state_env();
    let _ = eval_ctx.take_next_state_env();
    // Part of #2895: Apply liveness bindings via BindingChain.
    if let Some(chain) = request.bindings {
        eval_ctx = eval_ctx.with_liveness_bindings(chain);
    }

    cached_successor_satisfies_action(
        &eval_ctx,
        request.current_state,
        request.action,
        request.bindings,
        request.require_state_change,
        request.subscript,
        request.cached_successors,
        // Empty-registry fallback: fingerprint-keyed caches stay off (no
        // stable production fp domain), matching the other liveness caches.
        None,
        // UNCAPPED: this scan result is returned directly as the verdict.
        None,
        eval_subscript_changed,
    )
}

fn has_state_change<F>(
    ctx: &EvalCtx,
    current_state: &State,
    successor_state: &State,
    subscript: Option<&ExprRef>,
    eval_subscript_changed: &mut F,
) -> EvalResult<bool>
where
    F: FnMut(&EvalCtx, &State, &State, &ExprRef) -> EvalResult<bool>,
{
    if let Some(sub_expr) = subscript {
        return eval_subscript_changed(ctx, current_state, successor_state, sub_expr);
    }
    Ok(successor_state.fingerprint() != current_state.fingerprint())
}

/// Build the evaluation context for action-predicate checks over candidate
/// successors of `current_state`.
///
/// Part of #liveness-leaf-memo: extracted from `transition_satisfies_action`
/// so the (current-state, bindings) setup — which does not depend on the
/// successor — is performed once per scan instead of once per successor.
fn build_action_predicate_ctx(
    eval_ctx: &EvalCtx,
    current_state: &State,
    bindings: Option<&BindingChain>,
) -> EvalCtx {
    let mut action_ctx = eval_ctx.clone();
    for (name, value) in current_state.vars() {
        action_ctx.env_mut().insert(Arc::clone(name), value.clone());
    }
    // Part of #2895: Apply liveness bindings via BindingChain (shadows state vars).
    // Soundness (#liveness-wf): also materialize the quantifier bindings into the
    // env by NAME. Resolved fairness actions carry their bound variables as
    // `Ident(name, NameId::INVALID)` (e.g. the `car` in
    // `MoveOutsideBridge(car)`); a chain keyed by interned NameId is only hit on
    // NameId equality, so an INVALID-NameId reference can miss the chain and be
    // resolved inconsistently. Binding by name as well guarantees the action's
    // free quantifier variables resolve to the intended value.
    if let Some(chain) = bindings {
        for (_name_id, name_str, value) in chain.iter_eager() {
            action_ctx
                .env_mut()
                .insert(Arc::clone(&name_str), value.clone());
        }
        action_ctx = action_ctx.with_liveness_bindings(chain);
    }
    action_ctx
}

/// Evaluate the action predicate over one `(current, successor)` pair, where
/// `action_ctx` was prepared by [`build_action_predicate_ctx`] (current-state
/// vars + quantifier bindings already bound).
fn transition_satisfies_action(
    action_ctx: &EvalCtx,
    successor_state: &State,
    action: &ExprRef,
) -> EvalResult<bool> {
    let mut action_ctx = action_ctx.clone();
    let mut next_env = Env::new();
    for (name, value) in successor_state.vars() {
        next_env.insert(Arc::clone(name), value.clone());
    }
    action_ctx.set_next_state(next_env);

    // Soundness (#liveness-wf): clear the next-state-keyed substitution cache
    // before evaluating the action over this (current, successor) pair. This
    // function is called in a loop over candidate successors with a fresh
    // `next_env` per call; the SUBST_CACHE is invalidated by next-state-env
    // *pointer* identity, and a freshly-allocated `Env` can land on the same
    // address as a previous successor's `Env`, producing a stale cache hit that
    // makes a genuine action transition spuriously evaluate to `false` (an action
    // wrongly reported NOT taken / disabled — the unsound WF/SF direction).
    crate::eval::clear_subst_cache();
    let value = super::eval_live_entry(&action_ctx, action)?;
    expect_live_bool(&value, Some(action.span))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::liveness::test_helpers::spanned;
    use crate::Value;
    use tla_core::name_intern::NameId;

    fn identity_action() -> ExprRef {
        let x = spanned(Expr::Ident("x".to_string(), NameId::INVALID));
        let x_prime = spanned(Expr::Prime(Box::new(x.clone())));
        Arc::new(spanned(Expr::Eq(Box::new(x_prime), Box::new(x))))
    }

    #[test]
    fn test_cached_successor_satisfies_action_accepts_self_loop_without_state_change() {
        let eval_ctx = EvalCtx::new();
        let current = State::from_pairs([("x", Value::int(5))]);
        let mut eval_subscript_changed = |_: &EvalCtx, _: &State, _: &State, _: &ExprRef| {
            unreachable!("no subscript check when require_state_change=false")
        };

        let enabled = cached_successor_satisfies_action(
            &eval_ctx,
            &current,
            &identity_action(),
            None,
            false,
            None,
            std::slice::from_ref(&current),
            None,
            None,
            &mut eval_subscript_changed,
        )
        .expect("self-loop successor should evaluate without error");

        assert!(
            enabled,
            "require_state_change=false should accept a cached self-loop that satisfies the action"
        );
    }

    #[test]
    fn test_cached_successor_satisfies_action_skips_self_loop_with_state_change() {
        let eval_ctx = EvalCtx::new();
        let current = State::from_pairs([("x", Value::int(5))]);
        let mut eval_subscript_changed = |_: &EvalCtx, _: &State, _: &State, _: &ExprRef| {
            unreachable!("no subscript check without a subscript expression")
        };

        let enabled = cached_successor_satisfies_action(
            &eval_ctx,
            &current,
            &identity_action(),
            None,
            true,
            None,
            std::slice::from_ref(&current),
            None,
            None,
            &mut eval_subscript_changed,
        )
        .expect("self-loop successor should evaluate without error");

        assert!(
            !enabled,
            "require_state_change=true must skip a cached self-loop successor"
        );
    }

    // ── #liveness-witness-pair-pop: populate_witnessed_pair ─────────────

    /// Build an EvalCtx with a registered var space for the enumeration path.
    fn ctx_with_vars(names: &[&str]) -> EvalCtx {
        let mut ctx = EvalCtx::new();
        ctx.register_vars(names.iter().map(|s| (*s).to_string()));
        ctx
    }

    /// Action `x' = 1` (pins `x`, prime-free RHS) from `x = 0`.
    fn assign_one_action() -> ExprRef {
        let x = spanned(Expr::Ident("x".to_string(), NameId::INVALID));
        let x_prime = spanned(Expr::Prime(Box::new(x)));
        let one = spanned(Expr::Int(1.into()));
        Arc::new(spanned(Expr::Eq(Box::new(x_prime), Box::new(one))))
    }

    /// Positive: a COMPLETE enumeration populates TRUE (membership) for the
    /// enumerated successor and — with the all-vars pinning proof registered
    /// (`full_population_tag`) — FALSE (non-membership) for the rest.
    #[test]
    fn witness_pair_pop_complete_populates_true_and_false() {
        const TAG: u32 = 9101;
        let ctx = ctx_with_vars(&["x"]);
        let current = State::from_pairs([("x", Value::int(0))]);
        let member = State::from_pairs([("x", Value::int(1))]);
        let non_member = State::from_pairs([("x", Value::int(2))]);
        let cached = [member.clone(), non_member.clone()];
        super::super::checker::clear_scan_pred_results();
        super::super::checker::extend_full_population_tags([TAG]);

        populate_witnessed_pair(
            &ctx,
            false,
            &current,
            &assign_one_action(),
            None,
            &cached,
            TAG,
        );

        let cur_fp = current.fingerprint();
        assert_eq!(
            super::super::checker::get_scan_pred_result(cur_fp, member.fingerprint(), TAG),
            Some(true),
            "the enumerated successor is a membership-TRUE predicate result"
        );
        assert_eq!(
            super::super::checker::get_scan_pred_result(cur_fp, non_member.fingerprint(), TAG),
            Some(false),
            "non-membership is FALSE for an all-vars pinning-proven tag"
        );
        super::super::checker::clear_scan_pred_results();
        super::super::checker::clear_leaf_result_cache();
    }

    /// Fail-closed: WITHOUT the pinning proof registered, only TRUE membership
    /// entries are written — non-membership makes no claim.
    #[test]
    fn witness_pair_pop_unproven_tag_populates_true_only() {
        const TAG: u32 = 9102;
        let ctx = ctx_with_vars(&["x"]);
        let current = State::from_pairs([("x", Value::int(0))]);
        let member = State::from_pairs([("x", Value::int(1))]);
        let non_member = State::from_pairs([("x", Value::int(2))]);
        let cached = [member.clone(), non_member.clone()];
        super::super::checker::clear_scan_pred_results();

        populate_witnessed_pair(
            &ctx,
            false,
            &current,
            &assign_one_action(),
            None,
            &cached,
            TAG,
        );

        let cur_fp = current.fingerprint();
        assert_eq!(
            super::super::checker::get_scan_pred_result(cur_fp, member.fingerprint(), TAG),
            Some(true),
        );
        assert_eq!(
            super::super::checker::get_scan_pred_result(cur_fp, non_member.fingerprint(), TAG),
            None,
            "an unproven tag must never receive a FALSE (no-claim) entry"
        );
        super::super::checker::clear_scan_pred_results();
    }

    /// Fail-closed: an erroring enumeration populates NOTHING (the canonical
    /// per-transition evaluation then decides — and surfaces the canonical
    /// error — exactly like the pre-change early return).
    #[test]
    fn witness_pair_pop_enum_error_populates_nothing() {
        const TAG: u32 = 9103;
        let ctx = ctx_with_vars(&["x"]);
        let current = State::from_pairs([("x", Value::int(0))]);
        // `x' = undefined_name` errors during enumeration.
        let x = spanned(Expr::Ident("x".to_string(), NameId::INVALID));
        let x_prime = spanned(Expr::Prime(Box::new(x)));
        let bad = spanned(Expr::Ident("undefined_name".to_string(), NameId::INVALID));
        let action: ExprRef = Arc::new(spanned(Expr::Eq(Box::new(x_prime), Box::new(bad))));
        let succ = State::from_pairs([("x", Value::int(1))]);
        let cached = [succ.clone()];
        super::super::checker::clear_scan_pred_results();
        super::super::checker::extend_full_population_tags([TAG]);

        populate_witnessed_pair(&ctx, false, &current, &action, None, &cached, TAG);

        assert_eq!(
            super::super::checker::get_scan_pred_result(
                current.fingerprint(),
                succ.fingerprint(),
                TAG
            ),
            None,
            "an erroring enumeration must make no claim"
        );
        super::super::checker::clear_scan_pred_results();
        super::super::checker::clear_leaf_result_cache();
    }

    /// Defense-in-depth: an EMPTY complete enumeration (impossible for a
    /// genuine witness) populates NOTHING — every transition keeps the
    /// canonical per-transition evaluation.
    #[test]
    fn witness_pair_pop_empty_enum_populates_nothing() {
        const TAG: u32 = 9104;
        let ctx = ctx_with_vars(&["x"]);
        let current = State::from_pairs([("x", Value::int(0))]);
        // `FALSE /\ x' = 1` enumerates to the empty set.
        let f = spanned(Expr::Bool(false));
        let action: ExprRef = Arc::new(spanned(Expr::And(
            Box::new(f),
            Box::new(assign_one_action().as_ref().clone()),
        )));
        let succ = State::from_pairs([("x", Value::int(1))]);
        let cached = [succ.clone()];
        super::super::checker::clear_scan_pred_results();
        super::super::checker::extend_full_population_tags([TAG]);

        populate_witnessed_pair(&ctx, false, &current, &action, None, &cached, TAG);

        assert_eq!(
            super::super::checker::get_scan_pred_result(
                current.fingerprint(),
                succ.fingerprint(),
                TAG
            ),
            None,
            "an empty enumeration under a witnessed-TRUE leaf must make no claim"
        );
        super::super::checker::clear_scan_pred_results();
        super::super::checker::clear_leaf_result_cache();
    }

    /// Regression (#wfcap): the cap must be an optional PROBE bound, not the
    /// decision procedure. With the witness sitting behind `cap`
    /// non-witnessing successors, `Some(cap)` returns `false` (sound only
    /// when an authoritative decision follows) while `None` must keep
    /// scanning and find the witness.
    #[test]
    fn test_scan_cap_none_finds_witness_past_capped_prefix() {
        let eval_ctx = EvalCtx::new();
        let current = State::from_pairs([("x", Value::int(0))]);
        // Action `x' = 5`: only the THIRD successor witnesses it.
        let five = spanned(Expr::Int(5.into()));
        let x = spanned(Expr::Ident("x".to_string(), NameId::INVALID));
        let x_prime = spanned(Expr::Prime(Box::new(x)));
        let action: ExprRef = Arc::new(spanned(Expr::Eq(Box::new(x_prime), Box::new(five))));
        let successors = [
            State::from_pairs([("x", Value::int(1))]),
            State::from_pairs([("x", Value::int(2))]),
            State::from_pairs([("x", Value::int(5))]),
        ];
        let mut eval_subscript_changed = |_: &EvalCtx, _: &State, _: &State, _: &ExprRef| {
            unreachable!("no subscript expression in this test")
        };

        let capped = cached_successor_satisfies_action(
            &eval_ctx,
            &current,
            &action,
            None,
            true,
            None,
            &successors,
            None,
            Some(MAX_SCAN_PREDICATE_EVALS),
            &mut eval_subscript_changed,
        )
        .expect("capped scan should evaluate without error");
        assert!(
            !capped,
            "cap=Some(2) must stop before the third successor (probe semantics)"
        );

        let uncapped = cached_successor_satisfies_action(
            &eval_ctx,
            &current,
            &action,
            None,
            true,
            None,
            &successors,
            None,
            None,
            &mut eval_subscript_changed,
        )
        .expect("uncapped scan should evaluate without error");
        assert!(
            uncapped,
            "cap=None must scan every explored successor and find the x'=5 witness \
             behind the capped prefix"
        );
    }
}
