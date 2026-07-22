// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! LTL model checking for MCC LTLCardinality and LTLFireability examinations.
//!
//! Uses Buchi automaton product construction: negate the formula, convert to
//! a Generalized Buchi Automaton, build the product with the system's state
//! graph, and check for accepting cycles.

use std::collections::{HashMap, HashSet};
use std::time::{Duration, Instant};

use crate::buchi::{check_ltl_on_full_graph, check_ltl_on_the_fly, to_nnf, LtlNnf, PorContext};
use crate::explorer::explore_full;
use crate::explorer::ExplorationConfig;
use crate::model::PropertyAliases;
use crate::output::{Techniques, Verdict};
use crate::petri_net::{PetriNet, PlaceIdx, TransitionIdx};
use crate::property_xml::{
    CtlFormula, Formula, LtlFormula, PathQuantifier, Property, ReachabilityFormula, StatePredicate,
};
use crate::query_slice::{build_query_local_slice, build_query_slice, QuerySlice};
use crate::reduction::{
    reduce_iterative_structural_with_mode, ReducedNet, ReductionMode, ReductionReport,
};
use crate::resolved_predicate::{
    count_unresolved_with_aliases, eval_predicate, resolve_predicate_with_aliases,
};
use crate::stubborn::DependencyGraph;

use super::ltl_lasso_bmc::find_ltl_lasso_counterexample;
use super::ltl_por::{formula_contains_next, ltl_visible_reduced_transitions};
use super::ltl_walk::try_ltl_witness_walk;
use super::query_support::{
    closure_on_reduced_net, ltl_property_support_with_aliases, relevance_cone_on_reduced_net,
};
use super::reachability::check_reachability_properties_with_aliases;
use super::smt_encoding::{DEPTH_LADDER, PER_DEPTH_TIMEOUT};

const LTL_PREFILTER_PHASE_CAP: Duration = Duration::from_secs(1);
const LTL_PREFILTER_MIN_BUDGET: Duration = Duration::from_millis(100);
/// Reserve one virtual lane for the complete Buchi product whenever a shallow
/// prefilter takes a slice from the shared LTL deadline.
const LTL_PREFILTER_BUCHI_VIRTUAL_LANES: usize = 1;
/// Minimum solver budget below which lasso BMC is skipped — the encoding has
/// a non-trivial fixed cost and we want a meaningful attempt or no attempt.
const LTL_LASSO_BMC_MIN_BUDGET: Duration = Duration::from_millis(250);
/// Reserve one virtual lane for the complete Buchi product after the optional
/// lasso-BMC counterexample prefilter.
const LTL_LASSO_BUCHI_VIRTUAL_LANES: usize = 1;

fn ltl_lasso_bmc_phase_cap() -> Duration {
    let solver_queries = DEPTH_LADDER.len().saturating_add(1);
    PER_DEPTH_TIMEOUT * (solver_queries.clamp(1, u32::MAX as usize) as u32)
}
fn env_flag_enabled(key: &str) -> bool {
    env_flag_setting(key).unwrap_or(false)
}

fn env_flag_setting(key: &str) -> Option<bool> {
    std::env::var(key)
        .ok()
        .map(|value| value.trim().to_ascii_lowercase())
        .filter(|value| !value.is_empty())
        .map(|value| matches!(value.as_str(), "1" | "true" | "yes" | "on"))
}

/// Bounded lasso BMC is on by default — it is sound (replay-validated lasso
/// witness) and only converts timeouts into FALSE verdicts for safety-shaped
/// counterexamples within the bounded depth ladder. Set
/// `TY_MCC_DISABLE_LTL_LASSO_BMC=1` to opt out (e.g. when the SMT solver is
/// unavailable or for debugging the Büchi-only baseline).
fn ltl_lasso_bmc_enabled() -> bool {
    !env_flag_enabled("TY_MCC_DISABLE_LTL_LASSO_BMC")
}

/// `TY_LTL_DEBUG_LANES=1` — dev-only diagnostic: print which LTL pipeline lane
/// (phase-1 invariant / 2a AG / 2b EF / 2c CTL-fallback / 2d CTL-true-
/// sufficient / Büchi) decided each property, plus the resolved unfolded
/// instance set of every `is-fireable` atom. Output only; never changes
/// routing or verdicts.
fn ltl_lane_debug_enabled() -> bool {
    env_flag_enabled("TY_LTL_DEBUG_LANES")
}

/// `TY_MCC_DISABLE_LTL_CTL_FALLBACK=1` — investigation-only kill switch for
/// the Batch 2c/2d LTL→CTL routing, so the exact full-graph/Büchi lanes can
/// be exercised in isolation. Default off (no behavior change).
fn ltl_ctl_fallback_disabled() -> bool {
    env_flag_enabled("TY_MCC_DISABLE_LTL_CTL_FALLBACK")
}

/// Lane-debug helper: print `id -> verdict` for every entry of a lane map.
fn debug_dump_lane(label: &str, map: &HashMap<String, Verdict>) {
    if !ltl_lane_debug_enabled() {
        return;
    }
    let mut entries: Vec<_> = map.iter().collect();
    entries.sort_by(|a, b| a.0.cmp(b.0));
    for (id, verdict) in entries {
        eprintln!("LTL lane-debug [{label}]: {id} = {verdict:?}");
    }
}

/// Lane-debug helper: dump the resolved unfolded instance set of every
/// `is-fireable` atom in each property (colored nets: one colored transition
/// name resolves to ALL of its unfolded binding instances; see
/// `unfold/transitions.rs` alias construction and
/// `resolved_predicate::resolve_predicate_with_aliases`).
fn debug_dump_fireability_atoms(
    properties: &[Property],
    aliases: &PropertyAliases,
    net: &PetriNet,
) {
    if !ltl_lane_debug_enabled() {
        return;
    }

    fn collect_pred_names(pred: &StatePredicate, out: &mut Vec<String>) {
        match pred {
            StatePredicate::And(children) | StatePredicate::Or(children) => {
                for child in children {
                    collect_pred_names(child, out);
                }
            }
            StatePredicate::Not(inner) => collect_pred_names(inner, out),
            StatePredicate::IsFireable(names) => out.extend(names.iter().cloned()),
            StatePredicate::IntLe(..) | StatePredicate::True | StatePredicate::False => {}
        }
    }

    fn collect_ltl_names(formula: &LtlFormula, out: &mut Vec<String>) {
        match formula {
            LtlFormula::Atom(pred) => collect_pred_names(pred, out),
            LtlFormula::Not(inner)
            | LtlFormula::Next(inner)
            | LtlFormula::Finally(inner)
            | LtlFormula::Globally(inner) => collect_ltl_names(inner, out),
            LtlFormula::And(children) | LtlFormula::Or(children) => {
                for child in children {
                    collect_ltl_names(child, out);
                }
            }
            LtlFormula::Until(left, right) => {
                collect_ltl_names(left, out);
                collect_ltl_names(right, out);
            }
        }
    }

    for prop in properties {
        let Formula::Ltl(ltl) = &prop.formula else {
            continue;
        };
        let mut names = Vec::new();
        collect_ltl_names(ltl, &mut names);
        names.sort();
        names.dedup();
        for name in names {
            match aliases.resolve_transitions(&name) {
                Some(indices) => {
                    let instances: Vec<&str> = indices
                        .iter()
                        .map(|t| net.transitions[t.0 as usize].id.as_str())
                        .collect();
                    eprintln!(
                        "LTL lane-debug [atom]: {} is-fireable({name}) -> {} unfolded \
                         instance(s): {instances:?} (indices {:?})",
                        prop.id,
                        instances.len(),
                        indices.iter().map(|t| t.0).collect::<Vec<_>>()
                    );
                }
                None => eprintln!(
                    "LTL lane-debug [atom]: {} is-fireable({name}) -> UNRESOLVED",
                    prop.id
                ),
            }
        }
    }
}

/// Compute the deadline passed to the lasso BMC depth-ladder.
///
/// Lasso BMC is an opportunistic FALSE-witness lane. The Büchi product is the
/// complete fallback, so with a finite per-formula deadline we preserve a small
/// tail budget for Büchi and skip lasso when only that tail plus the minimum
/// useful solver budget remains.
fn lasso_bmc_deadline_at(global_deadline: Option<Instant>, now: Instant) -> Option<Instant> {
    global_deadline.map(|deadline| {
        let remaining = deadline.saturating_duration_since(now);
        if remaining.is_zero() {
            return now;
        }
        let budget = ltl_lasso_bmc_phase_cap().min(fair_share_budget_with_virtual_lanes(
            remaining,
            1,
            LTL_LASSO_BUCHI_VIRTUAL_LANES,
        ));
        now + budget
    })
}

fn lasso_bmc_deadline(global_deadline: Option<Instant>) -> Option<Instant> {
    lasso_bmc_deadline_at(global_deadline, Instant::now())
}

/// Whether the bounded lasso BMC has enough remaining budget to attempt at
/// least one depth. Used to fail-closed when the global deadline already
/// leaves less than [`LTL_LASSO_BMC_MIN_BUDGET`] — running the encoder for a
/// fraction of a second wastes wall budget that the Büchi solver still needs.
fn lasso_bmc_has_budget(deadline: Option<Instant>) -> bool {
    match deadline {
        None => true,
        Some(deadline) => {
            deadline.saturating_duration_since(Instant::now()) >= LTL_LASSO_BMC_MIN_BUDGET
        }
    }
}

/// Default-on rolling-residual per-formula Buchi budget for the shared MCC LTL
/// batch path.
///
/// Historical behavior was "give every formula the full global deadline" so
/// easy formulas resolved under generous time. The downside is that any one
/// formula that runs to the deadline starves every later formula (yields
/// `CannotCompute` for all of them). On a 16-formula examination this can
/// convert 15 TRUE/FALSE verdicts into 15 `CannotCompute`.
///
/// The LTL examination entry points in this crate do not currently carry a
/// public-vs-MCC mode bit: MCC collection, model execution, and tests share this
/// same internal batch path. Until that distinction exists in
/// [`ExplorationConfig`], default to the MCC-safe policy here and keep explicit
/// kill switches for conservative reruns: set
/// `TY_MCC_DISABLE_LTL_ROLLING_BUDGET=1`, or set the old opt-in variable to an
/// explicit false value such as `TY_MCC_ENABLE_LTL_ROLLING_BUDGET=0`.
///
/// The enabled policy is a rolling fair-share with one virtual lane reserved
/// for exact full-system-graph retry: each formula gets
/// `(deadline - now) / (remaining_count + 1)`. Formulas that finish
/// under-budget leave their unused time in the pool for the remaining queue, so
/// the policy is strictly residual.
fn ltl_rolling_budget_enabled() -> bool {
    #[cfg(test)]
    if let Some(enabled) = ltl_rolling_budget_test_override() {
        return enabled;
    }

    if env_flag_enabled("TY_MCC_DISABLE_LTL_ROLLING_BUDGET") {
        return false;
    }

    if matches!(
        env_flag_setting("TY_MCC_ENABLE_LTL_ROLLING_BUDGET"),
        Some(false)
    ) {
        return false;
    }

    true
}

/// `TY_MCC_LTL_REDUCTION_CROSSCHECK=1` — the reduction cross-check ORACLE. When
/// structural reduction is enabled (`TY_MCC_ENABLE_LTL_REDUCTION`), compute each
/// Büchi verdict on BOTH the reduced net and the identity net; on any difference,
/// log it and RETURN THE IDENTITY VERDICT (sound). This is the prerequisite the
/// audit names before re-enabling the reduction: it runs the reduction against a
/// trustworthy oracle over the real corpus so the historical Sudoku-PT wrong
/// answer (and any other) is surfaced with its exact net/property, while the run
/// itself stays sound (identity is the safety net, never wrong).
fn ltl_reduction_crosscheck_enabled() -> bool {
    env_flag_enabled("TY_MCC_LTL_REDUCTION_CROSSCHECK")
}

/// Compute a single formula's Büchi verdict, optionally cross-checked against the
/// identity net (see [`ltl_reduction_crosscheck_enabled`]). Without the flag, or
/// when the reduction removed nothing, this is exactly `check_single_ltl_buchi`.
#[allow(clippy::too_many_arguments)]
fn buchi_verdict_crosschecked(
    prop: &Property,
    ltl: &LtlFormula,
    class: Option<&ShallowLtl>,
    net: &PetriNet,
    reduced: &ReducedNet,
    identity_reduced: &ReducedNet,
    aliases: &PropertyAliases,
    config: &ExplorationConfig,
) -> Verdict {
    let verdict = check_single_ltl_buchi(prop, ltl, class, net, reduced, aliases, config);
    let reduction_changed =
        reduced.report.places_removed() + reduced.report.transitions_removed() > 0;
    if !reduction_changed || !ltl_reduction_crosscheck_enabled() {
        return verdict;
    }
    let identity_verdict =
        check_single_ltl_buchi(prop, ltl, class, net, identity_reduced, aliases, config);
    if verdict != identity_verdict {
        // True-vs-False is an UNSOUND reduction (the Sudoku-PT class); a
        // CannotCompute difference is only a precision change. Flag both, loudly
        // for the dangerous one, and defer to the sound identity verdict.
        let dangerous =
            verdict != Verdict::CannotCompute && identity_verdict != Verdict::CannotCompute;
        eprintln!(
            "LTL reduction CROSS-CHECK {} on {}: reduced={verdict} identity={identity_verdict} \
             — using identity (sound)",
            if dangerous {
                "DIVERGENCE (unsound reduction!)"
            } else {
                "precision-diff"
            },
            prop.id,
        );
    }
    identity_verdict
}

#[cfg(test)]
thread_local! {
    static LTL_ROLLING_BUDGET_TEST_OVERRIDE: std::cell::Cell<Option<bool>> =
        const { std::cell::Cell::new(None) };
}

#[cfg(test)]
fn ltl_rolling_budget_test_override() -> Option<bool> {
    LTL_ROLLING_BUDGET_TEST_OVERRIDE.with(std::cell::Cell::get)
}

#[cfg(test)]
struct LtlRollingBudgetTestOverrideGuard {
    prev: Option<bool>,
}

#[cfg(test)]
impl Drop for LtlRollingBudgetTestOverrideGuard {
    fn drop(&mut self) {
        LTL_ROLLING_BUDGET_TEST_OVERRIDE.with(|slot| slot.set(self.prev));
    }
}

#[cfg(test)]
fn set_ltl_rolling_budget_test_override(enabled: bool) -> LtlRollingBudgetTestOverrideGuard {
    LTL_ROLLING_BUDGET_TEST_OVERRIDE.with(|slot| {
        let prev = slot.get();
        slot.set(Some(enabled));
        LtlRollingBudgetTestOverrideGuard { prev }
    })
}

#[cfg(test)]
pub(super) fn with_ltl_rolling_budget_for_test<T>(enabled: bool, f: impl FnOnce() -> T) -> T {
    let _guard = set_ltl_rolling_budget_test_override(enabled);
    f()
}

/// Count unresolved place/transition names in an LTL formula's atoms.
///
/// Returns `(total_names, unresolved_count)`. If `unresolved_count > 0`,
/// the formula references names absent from the model — the resolved formula
/// is degenerate and evaluation may produce wrong answers.
fn count_unresolved_ltl(formula: &LtlFormula, aliases: &PropertyAliases) -> (usize, usize) {
    match formula {
        LtlFormula::Atom(pred) => count_unresolved_with_aliases(pred, aliases),
        LtlFormula::Not(inner)
        | LtlFormula::Next(inner)
        | LtlFormula::Finally(inner)
        | LtlFormula::Globally(inner) => count_unresolved_ltl(inner, aliases),
        LtlFormula::And(children) | LtlFormula::Or(children) => {
            children.iter().fold((0, 0), |(t, u), c| {
                let (ct, cu) = count_unresolved_ltl(c, aliases);
                (t + ct, u + cu)
            })
        }
        LtlFormula::Until(phi, psi) => {
            let (pt, pu) = count_unresolved_ltl(phi, aliases);
            let (qt, qu) = count_unresolved_ltl(psi, aliases);
            (pt + qt, pu + qu)
        }
    }
}

/// An LTL property reducible to or pre-filterable by reachability.
#[derive(Debug)]
#[cfg_attr(test, derive(Clone, PartialEq, Eq))]
pub(crate) enum ShallowLtl {
    /// G(pred): AG(pred) — fully routable through reachability.
    Invariant(StatePredicate),
    /// F(pred): AF(pred) — pre-filterable but not fully routable.
    /// Try AG(pred) for quick TRUE, EF(pred) for quick FALSE.
    Eventually(StatePredicate),
    /// Liveness shape that may benefit from bounded lasso-BMC, but still
    /// falls through to the Buchi pipeline unless a replay-validated lasso
    /// counterexample is found.
    LassoBmcLivenessCandidate,
}

/// Classify an LTL formula as shallow (reachability-relevant) or deep.
///
/// Returns `Some` for formulas that can be fully routed through, or
/// pre-filtered by, the reachability pipeline.
pub(crate) fn classify_shallow_ltl(formula: &LtlFormula) -> Option<ShallowLtl> {
    if lasso_bmc_liveness_candidate(formula) {
        return Some(ShallowLtl::LassoBmcLivenessCandidate);
    }

    match formula {
        // G(atom) = AG(atom) — invariant
        LtlFormula::Globally(inner) => extract_state_pred_ltl(inner)
            .map(ShallowLtl::Invariant)
            .or_else(|| match inner.as_ref() {
                // G(G(atom)) = AG(atom) — idempotent
                LtlFormula::Globally(_) => classify_shallow_ltl(inner),
                _ => None,
            }),
        // F(atom) = AF(atom) — pre-filterable
        LtlFormula::Finally(inner) => extract_state_pred_ltl(inner)
            .map(ShallowLtl::Eventually)
            .or_else(|| match inner.as_ref() {
                // F(F(atom)) = AF(atom) — idempotent
                LtlFormula::Finally(_) => classify_shallow_ltl(inner),
                _ => None,
            }),
        // Not(F(...)) = G(Not(...)); Not(G(...)) = F(Not(...)).
        // Reuse the shallow classifier so idempotent F(F p)/G(G p) and
        // double-negated shallow wrappers still take the reachability lane.
        LtlFormula::Not(inner) => classify_shallow_ltl(inner).map(negate_shallow_ltl),
        _ => None,
    }
}

fn negate_shallow_ltl(class: ShallowLtl) -> ShallowLtl {
    match class {
        ShallowLtl::Invariant(pred) => ShallowLtl::Eventually(negate_state_predicate(pred)),
        ShallowLtl::Eventually(pred) => ShallowLtl::Invariant(negate_state_predicate(pred)),
        ShallowLtl::LassoBmcLivenessCandidate => ShallowLtl::LassoBmcLivenessCandidate,
    }
}

fn initial_marking_shallow_verdict(
    net: &PetriNet,
    aliases: &PropertyAliases,
    class: Option<&ShallowLtl>,
) -> Option<Verdict> {
    let pred = match class? {
        ShallowLtl::Invariant(pred) | ShallowLtl::Eventually(pred) => pred,
        ShallowLtl::LassoBmcLivenessCandidate => return None,
    };
    let (_, unresolved) = count_unresolved_with_aliases(pred, aliases);
    if unresolved > 0 {
        return None;
    }

    let resolved = resolve_predicate_with_aliases(pred, aliases);
    let initial_holds = eval_predicate(&resolved, &net.initial_marking, net);
    match class? {
        ShallowLtl::Invariant(_) if !initial_holds => Some(Verdict::False),
        ShallowLtl::Eventually(_) if initial_holds => Some(Verdict::True),
        _ => None,
    }
}

fn initial_marking_forced_verdict(
    net: &PetriNet,
    aliases: &PropertyAliases,
    formula: &LtlFormula,
) -> Option<Verdict> {
    initial_marking_forced_bool(net, aliases, formula).map(|holds| {
        if holds {
            Verdict::True
        } else {
            Verdict::False
        }
    })
}

fn initial_marking_forced_bool(
    net: &PetriNet,
    aliases: &PropertyAliases,
    formula: &LtlFormula,
) -> Option<bool> {
    match formula {
        LtlFormula::Atom(pred) => {
            let (_, unresolved) = count_unresolved_with_aliases(pred, aliases);
            if unresolved > 0 {
                return None;
            }
            let resolved = resolve_predicate_with_aliases(pred, aliases);
            Some(eval_predicate(&resolved, &net.initial_marking, net))
        }
        LtlFormula::Not(inner) => initial_marking_forced_bool(net, aliases, inner).map(|v| !v),
        LtlFormula::And(children) => {
            let mut all_true = true;
            for child in children {
                match initial_marking_forced_bool(net, aliases, child) {
                    Some(false) => return Some(false),
                    Some(true) => {}
                    None => all_true = false,
                }
            }
            all_true.then_some(true)
        }
        LtlFormula::Or(children) => {
            let mut all_false = true;
            for child in children {
                match initial_marking_forced_bool(net, aliases, child) {
                    Some(true) => return Some(true),
                    Some(false) => {}
                    None => all_false = false,
                }
            }
            all_false.then_some(false)
        }
        LtlFormula::Finally(inner) => match initial_marking_forced_bool(net, aliases, inner) {
            Some(true) => Some(true),
            _ => None,
        },
        LtlFormula::Globally(inner) => match initial_marking_forced_bool(net, aliases, inner) {
            Some(false) => Some(false),
            _ => None,
        },
        LtlFormula::Until(left, right) => {
            let right_now = initial_marking_forced_bool(net, aliases, right);
            if right_now == Some(true) {
                return Some(true);
            }
            let left_now = initial_marking_forced_bool(net, aliases, left);
            if right_now == Some(false) && left_now == Some(false) {
                Some(false)
            } else {
                None
            }
        }
        LtlFormula::Next(_) => None,
    }
}

fn lasso_bmc_liveness_candidate(formula: &LtlFormula) -> bool {
    match formula {
        // G F p
        LtlFormula::Globally(inner) => match inner.as_ref() {
            LtlFormula::Finally(body) => extract_state_pred_ltl(body).is_some(),
            LtlFormula::Or(children) => response_pattern_body(children),
            _ => false,
        },
        // F G p
        LtlFormula::Finally(inner) => match inner.as_ref() {
            LtlFormula::Globally(body) => extract_state_pred_ltl(body).is_some(),
            _ => false,
        },
        LtlFormula::And(children) | LtlFormula::Or(children) => {
            children.iter().all(lasso_bmc_liveness_candidate)
        }
        _ => false,
    }
}

fn response_pattern_body(children: &[LtlFormula]) -> bool {
    if children.len() != 2 {
        return false;
    }
    let left_not = response_negated_state_pred(&children[0]);
    let left_eventual = response_eventual_state_pred(&children[0]);
    let right_not = response_negated_state_pred(&children[1]);
    let right_eventual = response_eventual_state_pred(&children[1]);
    (left_not && right_eventual) || (right_not && left_eventual)
}

fn response_negated_state_pred(formula: &LtlFormula) -> bool {
    matches!(formula, LtlFormula::Not(inner) if extract_state_pred_ltl(inner).is_some())
}

fn response_eventual_state_pred(formula: &LtlFormula) -> bool {
    matches!(formula, LtlFormula::Finally(inner) if extract_state_pred_ltl(inner).is_some())
}

fn negate_state_predicate(predicate: StatePredicate) -> StatePredicate {
    match predicate {
        StatePredicate::Not(inner) => *inner,
        other => StatePredicate::Not(Box::new(other)),
    }
}

/// Extract a pure state predicate from an LTL formula.
///
/// Returns `Some` only if the formula contains no temporal operators.
fn extract_state_pred_ltl(formula: &LtlFormula) -> Option<StatePredicate> {
    match formula {
        LtlFormula::Atom(pred) => Some(pred.clone()),
        LtlFormula::Not(inner) => {
            extract_state_pred_ltl(inner).map(|p| StatePredicate::Not(Box::new(p)))
        }
        LtlFormula::And(children) => {
            let preds: Option<Vec<_>> = children.iter().map(extract_state_pred_ltl).collect();
            preds.map(StatePredicate::And)
        }
        LtlFormula::Or(children) => {
            let preds: Option<Vec<_>> = children.iter().map(extract_state_pred_ltl).collect();
            preds.map(StatePredicate::Or)
        }
        // Any temporal operator means not a pure state predicate.
        LtlFormula::Next(_)
        | LtlFormula::Finally(_)
        | LtlFormula::Globally(_)
        | LtlFormula::Until(_, _) => None,
    }
}

/// Build a `ReducedNet` that composes a query slice with the upstream
/// structural reduction: sliced-net indices map directly to original-net
/// indices, so `expand_marking_into` works in a single step.
fn compose_slice_and_reduction(slice: &QuerySlice, reduced: &ReducedNet) -> ReducedNet {
    let place_map = slice.compose_place_map(&reduced.place_map);
    let transition_map = slice.compose_transition_map(&reduced.transition_map);

    let place_unmap: Vec<PlaceIdx> = slice
        .place_unmap
        .iter()
        .map(|&reduced_idx| reduced.place_unmap[reduced_idx.0 as usize])
        .collect();
    let transition_unmap: Vec<TransitionIdx> = slice
        .transition_unmap
        .iter()
        .map(|&reduced_idx| reduced.transition_unmap[reduced_idx.0 as usize])
        .collect();

    ReducedNet {
        net: slice.net.clone(),
        place_map,
        place_unmap,
        place_scales: reduced.place_scales.clone(),
        transition_map,
        transition_unmap,
        constant_values: reduced.constant_values.clone(),
        reconstructions: reduced.reconstructions.clone(),
        report: ReductionReport::default(),
    }
}

/// Run LTL model checking for a set of properties.
///
/// Uses the on-the-fly product construction: system successors are computed
/// lazily by firing transitions on the reduced net, eliminating the need to
/// build and store the full system reachability graph.
///
/// Returns `(property_id, verdict)` for each property.
#[cfg(test)]
pub(crate) fn check_ltl_properties(
    net: &PetriNet,
    properties: &[Property],
    config: &ExplorationConfig,
) -> Vec<(String, Verdict)> {
    let aliases = PropertyAliases::identity(net);
    check_ltl_properties_inner(
        net,
        properties,
        &aliases,
        config,
        false,
        &Techniques::default(),
    )
}

pub(crate) fn check_ltl_properties_with_aliases(
    net: &PetriNet,
    properties: &[Property],
    aliases: &PropertyAliases,
    config: &ExplorationConfig,
) -> Vec<(String, Verdict)> {
    check_ltl_properties_inner(
        net,
        properties,
        aliases,
        config,
        false,
        &Techniques::default(),
    )
}

pub(crate) fn check_ltl_properties_with_flush(
    net: &PetriNet,
    properties: &[Property],
    aliases: &PropertyAliases,
    config: &ExplorationConfig,
) -> Vec<(String, Verdict)> {
    check_ltl_properties_with_flush_and_techniques(
        net,
        properties,
        aliases,
        config,
        &Techniques::default(),
    )
}

pub(crate) fn check_ltl_properties_with_flush_and_techniques(
    net: &PetriNet,
    properties: &[Property],
    aliases: &PropertyAliases,
    config: &ExplorationConfig,
    techniques: &Techniques,
) -> Vec<(String, Verdict)> {
    check_ltl_properties_inner(net, properties, aliases, config, true, techniques)
}

fn check_ltl_properties_inner(
    net: &PetriNet,
    properties: &[Property],
    aliases: &PropertyAliases,
    config: &ExplorationConfig,
    flush: bool,
    techniques: &Techniques,
) -> Vec<(String, Verdict)> {
    let unresolved_original_ids: HashSet<String> = properties
        .iter()
        .filter_map(|prop| {
            let Formula::Ltl(ltl) = &prop.formula else {
                return None;
            };
            let (total, unresolved) = count_unresolved_ltl(ltl, aliases);
            if unresolved == 0 {
                return None;
            }
            eprintln!(
                "LTL resolution guard: {} has {unresolved}/{total} \
                 unresolved names before simplification -> CANNOT_COMPUTE",
                prop.id
            );
            Some(prop.id.clone())
        })
        .collect();

    // Simplify formulas using structural facts and LP proofs.
    //
    // NOTE: the LTL lane intentionally keeps the un-deadlined simplifier entry
    // point. A simplifier sub-budget here was tried and reverted: while it
    // bounds the simplifier's own trap/state-equation LP, the freed budget is
    // then handed to the downstream reachability "original-net full BFS" routing
    // (G/F-atom shapes), which has its own pre-existing deadline-hygiene gap and
    // overran on SharedMemory-PT-000020 LTLFireability. The deadline-hygiene
    // work in this change set targets the CTL simplifier (the headline 4x
    // overrun) and the Liveness mu loop; the LTL exploration phases are left
    // exactly as before to avoid perturbing their resolution/overrun balance.
    let simplified =
        crate::formula_simplify::simplify_properties_with_aliases(net, properties, aliases);
    let properties = simplified.as_slice();
    let mut flushed_ids = HashSet::new();

    debug_dump_fireability_atoms(properties, aliases, net);

    // ── Phase 0: classify shallow LTL properties ──
    let classifications: Vec<Option<ShallowLtl>> = properties
        .iter()
        .map(|prop| match &prop.formula {
            Formula::Ltl(ltl) => classify_shallow_ltl(ltl),
            _ => None,
        })
        .collect();

    let initial_map: HashMap<String, Verdict> = properties
        .iter()
        .zip(&classifications)
        .filter_map(|(prop, class)| match &prop.formula {
            Formula::Ltl(ltl) => initial_marking_forced_verdict(net, aliases, ltl)
                .or_else(|| initial_marking_shallow_verdict(net, aliases, class.as_ref()))
                .map(|verdict| (prop.id.clone(), verdict)),
            _ => initial_marking_shallow_verdict(net, aliases, class.as_ref())
                .map(|verdict| (prop.id.clone(), verdict)),
        })
        .collect();

    // ── Phase 1: route G(atom) invariants through reachability as AG ──
    // Only definitive verdicts (True/False) short-circuit Buchi. CannotCompute
    // falls through so the Buchi path still gets a chance (#1370 design).
    let invariant_props: Vec<Property> = properties
        .iter()
        .zip(&classifications)
        .filter_map(|(prop, class)| match class {
            Some(ShallowLtl::Invariant(pred)) => Some(Property {
                id: prop.id.clone(),
                formula: Formula::Reachability(ReachabilityFormula {
                    quantifier: PathQuantifier::AG,
                    predicate: pred.clone(),
                }),
            }),
            _ => None,
        })
        .collect();

    let mut invariant_map: HashMap<String, Verdict> = if !invariant_props.is_empty() {
        let prefilter_config = config
            .clone()
            .with_deadline(ltl_prefilter_deadline(config.deadline(), 4));
        check_reachability_properties_with_aliases(
            net,
            &invariant_props,
            aliases,
            &prefilter_config,
        )
        .into_iter()
        .filter(|(_, v)| *v != Verdict::CannotCompute)
        .collect()
    } else {
        HashMap::new()
    };
    invariant_map.extend(
        initial_map
            .iter()
            .filter(|(_, verdict)| **verdict == Verdict::False)
            .map(|(id, verdict)| (id.clone(), *verdict)),
    );

    // Inject formulas that simplified to constant verdicts. The simplifier
    // may collapse e.g. G(trivially_true) → Atom(True), removing the temporal
    // wrapper that classify_shallow_ltl needs.
    for prop in properties {
        match &prop.formula {
            Formula::Ltl(LtlFormula::Atom(StatePredicate::True)) => {
                invariant_map.insert(prop.id.clone(), Verdict::True);
            }
            Formula::Ltl(LtlFormula::Atom(StatePredicate::False)) => {
                invariant_map.insert(prop.id.clone(), Verdict::False);
            }
            _ => {}
        }
    }
    invariant_map.extend(
        unresolved_original_ids
            .iter()
            .map(|id| (id.clone(), Verdict::CannotCompute)),
    );

    // Phase 1 streaming flush: emit invariant verdicts as soon as Phase 1
    // completes, BEFORE Phase 2 begins. If the external deadline triggers
    // SIGTERM during a later phase, every Phase 1 verdict is already in
    // stdout and survives termination. Batching the flush at the end of
    // Phase 2 (the prior behavior) discarded all per-formula progress when
    // a SIGTERM landed mid-pipeline — empty stdout = 0 MCC points even when
    // we had verdicts in hand. (Only definitive True/False are in the map;
    // CannotCompute entries injected for unresolved_original_ids are also
    // present but emitting them early is harmless — they are re-emitted
    // identically if no deeper verdict is found.)
    debug_dump_lane("phase1-invariant", &invariant_map);
    if flush {
        flush_resolved_verdicts(properties, &invariant_map, &mut flushed_ids, techniques);
    }

    // ── Phase 2: pre-filter F(atom) via batched reachability shortcuts ──
    // Batch 2a: AG(pred) for all eventual properties — quick TRUE shortcut.
    let eventually_ag_props: Vec<Property> = properties
        .iter()
        .zip(&classifications)
        .filter_map(|(prop, class)| match class {
            Some(ShallowLtl::Eventually(pred)) => Some(Property {
                id: prop.id.clone(),
                formula: Formula::Reachability(ReachabilityFormula {
                    quantifier: PathQuantifier::AG,
                    predicate: pred.clone(),
                }),
            }),
            _ => None,
        })
        .collect();

    let mut prefilter_map: HashMap<String, Verdict> = HashMap::new();
    prefilter_map.extend(
        initial_map
            .iter()
            .filter(|(_, verdict)| **verdict == Verdict::True)
            .map(|(id, verdict)| (id.clone(), *verdict)),
    );
    if !eventually_ag_props.is_empty() {
        let prefilter_config = config
            .clone()
            .with_deadline(ltl_prefilter_deadline(config.deadline(), 4));
        let ag_results = check_reachability_properties_with_aliases(
            net,
            &eventually_ag_props,
            aliases,
            &prefilter_config,
        );
        for (id, verdict) in ag_results {
            if verdict == Verdict::True {
                prefilter_map.insert(id, Verdict::True);
            }
        }
    }

    // Phase 2a streaming flush: emit any new TRUE verdicts (AG shortcut)
    // before the remaining, longer-running phases begin (see Phase 1 flush
    // rationale). These are definitive TRUE verdicts and never superseded.
    debug_dump_lane("phase2a-AG-true", &prefilter_map);
    if flush {
        flush_resolved_verdicts(properties, &prefilter_map, &mut flushed_ids, techniques);
    }

    // Batch 2b: EF(pred) for still-unresolved eventual properties — quick FALSE.
    let eventually_ef_props: Vec<Property> = properties
        .iter()
        .zip(&classifications)
        .filter_map(|(prop, class)| match class {
            Some(ShallowLtl::Eventually(pred)) if !prefilter_map.contains_key(&prop.id) => {
                Some(Property {
                    id: prop.id.clone(),
                    formula: Formula::Reachability(ReachabilityFormula {
                        quantifier: PathQuantifier::EF,
                        predicate: pred.clone(),
                    }),
                })
            }
            _ => None,
        })
        .collect();

    if !eventually_ef_props.is_empty() {
        let prefilter_config = config
            .clone()
            .with_deadline(ltl_prefilter_deadline(config.deadline(), 4));
        let ef_results = check_reachability_properties_with_aliases(
            net,
            &eventually_ef_props,
            aliases,
            &prefilter_config,
        );
        for (id, verdict) in ef_results {
            if verdict == Verdict::False {
                prefilter_map.insert(id, Verdict::False);
            }
        }
    }

    // Phase 2b streaming flush: emit any new FALSE verdicts (EF shortcut)
    // before the CTL-fallback batches and the Buchi loop — the longest-
    // running phases — begin. Landing each prefilter verdict in stdout now
    // means a later timeout still preserves them.
    debug_dump_lane("phase2ab-prefilter", &prefilter_map);
    if flush {
        flush_resolved_verdicts(properties, &prefilter_map, &mut flushed_ids, techniques);
    }

    // Batch 2c: route a conservative universal-LTL fragment through CTL.
    //
    // This is intentionally narrower than a general ACTL translation. In
    // particular, formulas such as A(F(G(p))) are not equivalent to AF(AG(p))
    // on branching systems. Restrict the exact fallback to direct universal
    // state-temporal forms that match the CTL checker's maximal-path
    // interpretation.
    let ctl_fallback_props: Vec<Property> = if ltl_ctl_fallback_disabled() {
        Vec::new()
    } else {
        properties
            .iter()
            .filter_map(|prop| {
                if invariant_map.contains_key(&prop.id) || prefilter_map.contains_key(&prop.id) {
                    return None;
                }
                let Formula::Ltl(ltl) = &prop.formula else {
                    return None;
                };
                ltl_universal_ctl_fallback(ltl).map(|ctl| Property {
                    id: prop.id.clone(),
                    formula: Formula::Ctl(ctl),
                })
            })
            .collect()
    };

    let ctl_fallback_map: HashMap<String, Verdict> =
        if !ctl_fallback_props.is_empty() && ltl_prefilter_has_budget(config.deadline(), 4) {
            let prefilter_config = config
                .clone()
                .with_deadline(ltl_prefilter_deadline(config.deadline(), 4));
            super::ctl::check_ctl_properties_with_aliases(
                net,
                &ctl_fallback_props,
                aliases,
                &prefilter_config,
            )
            .into_iter()
            .filter(|(_, verdict)| *verdict != Verdict::CannotCompute)
            .collect()
        } else {
            HashMap::new()
        };

    // Batch 2d: one-sided broad universal-LTL proof through CTL.
    //
    // This translation is only used as a sufficient condition for TRUE. For
    // example, AF(AG(p)) implies A(F(G(p))) but is not equivalent to it on
    // branching systems, so FALSE/CC results must fall through to Büchi.
    debug_dump_lane("phase2c-ctl-fallback", &ctl_fallback_map);

    let ctl_true_sufficient_props: Vec<Property> = if ltl_ctl_fallback_disabled() {
        Vec::new()
    } else {
        properties
            .iter()
            .filter_map(|prop| {
                if invariant_map.contains_key(&prop.id)
                    || prefilter_map.contains_key(&prop.id)
                    || ctl_fallback_map.contains_key(&prop.id)
                {
                    return None;
                }
                let Formula::Ltl(ltl) = &prop.formula else {
                    return None;
                };
                if ltl_universal_ctl_fallback(ltl).is_some() {
                    return None;
                }
                ltl_universal_ctl_true_sufficient(ltl).map(|ctl| Property {
                    id: prop.id.clone(),
                    formula: Formula::Ctl(ctl),
                })
            })
            .collect()
    };

    let ctl_true_sufficient_map: HashMap<String, Verdict> = if !ctl_true_sufficient_props.is_empty()
        && ltl_prefilter_has_budget(config.deadline(), 4)
    {
        let prefilter_config = config
            .clone()
            .with_deadline(ltl_prefilter_deadline(config.deadline(), 4));
        super::ctl::check_ctl_properties_with_aliases(
            net,
            &ctl_true_sufficient_props,
            aliases,
            &prefilter_config,
        )
        .into_iter()
        .filter(|(_, verdict)| *verdict == Verdict::True)
        .collect()
    } else {
        HashMap::new()
    };

    debug_dump_lane("phase2d-ctl-true-sufficient", &ctl_true_sufficient_map);

    if flush {
        for prop in properties {
            if flushed_ids.contains(&prop.id) {
                continue;
            }
            let verdict = invariant_map
                .get(&prop.id)
                .or_else(|| prefilter_map.get(&prop.id))
                .or_else(|| ctl_fallback_map.get(&prop.id))
                .or_else(|| ctl_true_sufficient_map.get(&prop.id));
            if let Some(verdict) = verdict {
                // Route through canonical formula_line so the FORMULA /
                // TECHNIQUES keywords come from mcc_keywords.rs. Closes
                // codex audit finding #4.
                crate::output::print_mcc_line(crate::output::formula_line_with_techniques(
                    "", &prop.id, *verdict, techniques,
                ));
                flushed_ids.insert(prop.id.clone());
            }
        }
    }

    let unresolved_indices: Vec<usize> = properties
        .iter()
        .enumerate()
        .filter_map(|(index, prop)| {
            (!invariant_map.contains_key(&prop.id)
                && !prefilter_map.contains_key(&prop.id)
                && !ctl_fallback_map.contains_key(&prop.id)
                && !ctl_true_sufficient_map.contains_key(&prop.id))
            .then_some(index)
        })
        .collect();

    let mut buchi_map: HashMap<String, Verdict> = HashMap::new();

    // ── Phase 2.5: ADDITIVE symbolic-BDD LTL prefilter (fail-closed). ──
    //
    // Runs BEFORE the expensive explicit Phase 3a (which materializes the full
    // reachable graph and can burn the whole budget on large nets the explicit
    // product cannot complete — the measured LTL root cause). For each residual
    // formula it builds the system × GBA(¬φ) product SYMBOLICALLY and answers
    // the fair-cycle (accepting-lasso) question via an Emerson–Lei fixpoint,
    // attacking both throughput AND the large-net memory wall.
    //
    // SOUND by construction (same GBA, guards, deadlock-stutter convention as
    // the explicit checker — `ltl_symbolic`/`ltl::symbolic_tests` differentials,
    // 0 disagreements vs the exhaustive full-graph decision). FAIL-CLOSED:
    // every per-formula attempt is budget-bounded and returns `None` on any
    // decline (unsupported atom / over-budget / OOM / net not DD-eligible),
    // leaving the formula for the existing exact lanes below — never a guessed
    // verdict. ADDITIVE: a per-formula sub-budget (a fraction of the shared
    // deadline) means it cannot starve Phase 3/3a (the CTL DD-lane drain
    // lesson). Env kill-switch: `TY_MCC_DISABLE_LTL_SYMBOLIC=1`.
    if !unresolved_indices.is_empty() && !ltl_symbolic_disabled() {
        for &index in &unresolved_indices {
            if config.deadline().is_some_and(|dl| Instant::now() >= dl) {
                break;
            }
            let prop = &properties[index];
            let Formula::Ltl(ltl) = &prop.formula else {
                continue;
            };
            let (total, unresolved) = count_unresolved_ltl(ltl, aliases);
            if unresolved > 0 {
                let _ = total;
                continue; // resolution guard; handled elsewhere
            }
            let mut atom_preds = Vec::new();
            let nnf = to_nnf(ltl, &mut atom_preds);
            let resolved_atoms: Vec<_> = atom_preds
                .iter()
                .map(|p| crate::buchi::resolve_atom_with_aliases(p, aliases))
                .collect();
            // Per-formula sub-budget: never let the symbolic prefilter consume
            // the whole shared deadline (4 virtual lanes ⇒ ≤ 1/4 each), so the
            // exact explicit lanes keep the majority of the budget on any net
            // the symbolic engine declines.
            let sub_deadline = ltl_prefilter_deadline(config.deadline(), 4);
            if let Some(verdict) =
                try_symbolic_ltl_verdict(&nnf, net, &resolved_atoms, sub_deadline)
            {
                if ltl_lane_debug_enabled() {
                    eprintln!(
                        "LTL lane-debug [symbolic-prefilter]: {} = {verdict:?}",
                        prop.id
                    );
                }
                buchi_map.insert(prop.id.clone(), verdict);
            }
        }
        if flush {
            flush_resolved_verdicts(properties, &buchi_map, &mut flushed_ids, techniques);
        }
    }

    // Recompute the residual after the symbolic prefilter so the explicit
    // Phase 3 lanes only work the formulas the symbolic lane could not decide.
    let unresolved_indices: Vec<usize> = unresolved_indices
        .into_iter()
        .filter(|&index| !buchi_map.contains_key(&properties[index].id))
        .collect();

    // ── Phase 2.6: ADDITIVE FALSE-witness random-walk lane (witness-only). ──
    //
    // Runs BEFORE the exact explicit Phase 3/3a and only ADDS verdicts: for each
    // still-unresolved `A(φ)` it random-walks the RAW net and asks the trusted,
    // differentially-tested oracle `accepting_lasso_exists` whether a reachable
    // marking trace contains a fair accepting lasso of GBA(¬φ). Such a lasso is a
    // concrete counterexample ⇒ `A(φ)` is FALSE. SOUND: emits ONLY
    // `Verdict::False`, ONLY on an oracle-confirmed reachable accepting lasso
    // (every visited marking is reachable by construction); never TRUE; a
    // miss/timeout leaves the formula unresolved for the exact lanes below.
    //
    // ADDITIVE via a SMALL, DOUBLY-BOUNDED budget (the prior `under_approx_lane_
    // deadline` = min(remaining/4, 8s) slice robbed enough wall time to flip
    // AutonomousCar-PT-01b LTLFireability's time-sensitive explicit Phase 3 from
    // 16/16 to 14/16). Two caps keep the robbery tiny:
    //   • per-formula: min(remaining/16, 2s) — a lasso that exists is found in
    //     well under a second, so each attempt is short;
    //   • global pass: min(initial_remaining/8, 8s) — once this elapses, the
    //     remaining pending formulas fall through to the exact lanes UNTOUCHED.
    // Env kill-switch: `TY_MCC_DISABLE_LTL_WALK=1`.
    if !unresolved_indices.is_empty() && !ltl_walk_disabled() {
        // Absolute global ceiling for the WHOLE pass, computed once up front.
        let pass_start = Instant::now();
        let global_deadline = config.deadline().map(|dl| {
            let remaining = dl.saturating_duration_since(pass_start);
            let capped = (remaining / LTL_WALK_GLOBAL_DIVISOR).min(LTL_WALK_GLOBAL_CAP);
            pass_start + capped
        });
        for &index in &unresolved_indices {
            // Stop the whole pass once the global cap (or the overall deadline)
            // is hit so the exact lanes keep the rest of the budget.
            if config.deadline().is_some_and(|dl| Instant::now() >= dl)
                || global_deadline.is_some_and(|dl| Instant::now() >= dl)
            {
                break;
            }
            let prop = &properties[index];
            let Formula::Ltl(ltl) = &prop.formula else {
                continue;
            };
            let (total, unresolved) = count_unresolved_ltl(ltl, aliases);
            if unresolved > 0 {
                let _ = total;
                continue; // resolution guard; handled elsewhere
            }
            // This formula's slice: min(remaining/16, 2s), never past the global
            // cap. Skip below the floor so a near-exhausted budget goes straight
            // to the exact lanes.
            let now = Instant::now();
            let lane_deadline = config.deadline().map(|dl| {
                let remaining = dl.saturating_duration_since(now);
                let capped =
                    (remaining / LTL_WALK_PER_FORMULA_DIVISOR).min(LTL_WALK_PER_FORMULA_CAP);
                let formula_dl = now + capped;
                match global_deadline {
                    Some(gdl) => formula_dl.min(gdl),
                    None => formula_dl,
                }
            });
            if lane_deadline
                .is_some_and(|dl| dl.saturating_duration_since(now) < LTL_WALK_MIN_BUDGET)
            {
                continue;
            }
            let mut atom_preds = Vec::new();
            let nnf = to_nnf(ltl, &mut atom_preds);
            let resolved_atoms: Vec<_> = atom_preds
                .iter()
                .map(|p| crate::buchi::resolve_atom_with_aliases(p, aliases))
                .collect();
            if let Some(verdict) = try_ltl_witness_walk(net, &nnf, &resolved_atoms, lane_deadline) {
                if ltl_lane_debug_enabled() {
                    eprintln!("LTL lane-debug [walk]: {} = {verdict:?}", prop.id);
                }
                buchi_map.insert(prop.id.clone(), verdict);
            }
        }
        if flush {
            flush_resolved_verdicts(properties, &buchi_map, &mut flushed_ids, techniques);
        }
    }

    // Recompute the residual after the walk lane so the explicit Phase 3 lanes
    // only work the formulas neither the symbolic prefilter nor the walk decided.
    let unresolved_indices: Vec<usize> = unresolved_indices
        .into_iter()
        .filter(|&index| !buchi_map.contains_key(&properties[index].id))
        .collect();

    // ── Phase 3: Buchi pipeline for remaining (deep + unresolved) ──
    //
    // Phase 3a (size-gated EXACT full-graph LTL, mirrors the CTL full-graph
    // budget fix af95cc86): before grinding the on-the-fly Büchi product per
    // formula, try to materialize the full reachable state graph ONCE on the
    // original net under a bounded wall fraction. When the net is enumerable
    // the build completes quickly and the exact automata-theoretic verdict
    // (negate -> GBA -> product accepting-cycle) is computed for every residual
    // formula at once — converting the deadline-starved CANNOT_COMPUTE that the
    // late retry produced (verified on Anderson-PT-04 LTLCardinality, 29641
    // states, 6 formulas were CC under starvation) into definite verdicts.
    //
    // This is SOUND: `check_ltl_on_full_graph` returns `None` -> CannotCompute
    // whenever the graph did not complete or the product overflowed/timed out,
    // so a non-enumerable net resolves nothing here and falls through to Büchi
    // exactly as before. Definite verdicts are exact (same algorithm as the
    // on-the-fly path, over a materialized COMPLETE graph). The build fraction
    // bounds the wall cost on non-enumerable nets so Büchi keeps the majority
    // of the budget; on enumerable nets the build returns early and the gate is
    // irrelevant.
    let early_full_graph_resolved = try_ltl_on_full_graph_early(
        net,
        properties,
        aliases,
        &unresolved_indices,
        &mut buchi_map,
        config,
    );
    if flush && !early_full_graph_resolved.is_empty() {
        flush_resolved_verdicts(properties, &buchi_map, &mut flushed_ids, techniques);
    }

    let identity_reduced = ReducedNet::identity(net);
    // SOUNDNESS: merge the FULL support (places AND transitions) and protect
    // the pre/post places of every `is-fireable` transition via
    // `protected_places_for_prefire`. Accumulating `support.places` alone left
    // fireability-only formulas with an EMPTY protected set — the same defect
    // that gutted the net in the CTL lane (AirplaneLD-COL-0010-LTLFireability-08
    // FALSE against a TRUE consensus). This lane is env-gated
    // (TY_MCC_ENABLE_LTL_REDUCTION), but must be verdict-preserving when on.
    let mut merged_support = crate::examinations::query_support::QuerySupport::new(
        net.num_places(),
        net.num_transitions(),
    );
    let mut unresolved_has_next = false;
    // INV-MASK completeness certificate. The structure-removing StutterInsensitiveLTL
    // rules are sound ONLY when `protected_places` is a COMPLETE over-approximation of
    // every place/transition any surviving formula atom observes. If ANY formula's
    // support cannot be resolved (removed-transition `IsFireable`, unreconstructable
    // `TokensCount`) — or a non-LTL survivor sneaks into the unresolved set — the mask
    // is under-approximated, and removing an observed place would flip a definite LTL
    // verdict (the MCC 2025 Sudoku-PT wrong-answer class). Track completeness and fail
    // closed to the marking-only safe subset below, rather than the previous behavior
    // of silently dropping a `None`/non-LTL formula's observation.
    let mut mask_complete = true;
    for &index in &unresolved_indices {
        let prop = &properties[index];
        if let Formula::Ltl(ltl) = &prop.formula {
            let mut atom_preds = Vec::new();
            let nnf = to_nnf(ltl, &mut atom_preds);
            unresolved_has_next |= formula_contains_next(&nnf);

            if let Some(support) =
                ltl_property_support_with_aliases(&identity_reduced, prop, aliases)
            {
                for (merged, keep) in merged_support.places.iter_mut().zip(support.places) {
                    *merged |= keep;
                }
                for (merged, keep) in merged_support
                    .transitions
                    .iter_mut()
                    .zip(support.transitions)
                {
                    *merged |= keep;
                }
            } else {
                // Support unresolvable ⇒ the mask cannot be certified complete ⇒ fail closed.
                mask_complete = false;
            }
        } else {
            // A non-LTL survivor ⇒ the LTL-only mask may not cover its observation.
            mask_complete = false;
        }
    }
    let protected_places =
        crate::examinations::reachability::protected_places_for_prefire(net, &merged_support);

    // Structural reduction before Buchi is still experimental. Historical
    // MCC 2025 Sudoku PT cases expose wrong answers when the stutter-sensitive
    // lane removes structure before the product construction. Keep the
    // competition default on the identity net; opt in only for investigations.
    let reduced = if env_flag_enabled("TY_MCC_ENABLE_LTL_REDUCTION") {
        let ltl_mode = if unresolved_has_next || !mask_complete {
            // X present (not stutter-invariant), OR the protected mask is uncertified ⇒
            // the marking-only StutterSensitiveLTL subset, which admits NONE of the
            // source/sink/isolated structure-removing rules and is universally sound.
            // Only an X-free formula set with a fully-certified complete mask gets the
            // full StutterInsensitiveLTL reduction.
            ReductionMode::StutterSensitiveLTL
        } else {
            ReductionMode::StutterInsensitiveLTL
        };
        match reduce_iterative_structural_with_mode(net, &protected_places, ltl_mode, None) {
            Ok(r) => {
                let removed = r.report.places_removed() + r.report.transitions_removed();
                if removed > 0 {
                    eprintln!(
                        "LTL {mode:?} reduction: removed {removed} elements \
                         ({p} places, {t} transitions)",
                        mode = ltl_mode,
                        p = r.report.places_removed(),
                        t = r.report.transitions_removed(),
                    );
                }
                r
            }
            Err(error) => {
                eprintln!("LTL reduction error: {error} — falling back to identity");
                ReducedNet::identity(net)
            }
        }
    } else {
        ReducedNet::identity(net)
    };

    let buchi_order = sorted_ltl_buchi_indices(properties, &unresolved_indices);
    for (position, &index) in buchi_order.iter().enumerate() {
        let prop = &properties[index];
        // Skip formulas the early exact full-graph pass already resolved.
        if early_full_graph_resolved.contains(&prop.id) {
            continue;
        }
        let remaining_count = buchi_order.len() - position;
        let buchi_config = config.clone().with_deadline(buchi_per_formula_deadline(
            config.deadline(),
            remaining_count,
        ));
        let verdict = match &prop.formula {
            Formula::Ltl(ltl) => buchi_verdict_crosschecked(
                prop,
                ltl,
                classifications[index].as_ref(),
                net,
                &reduced,
                &identity_reduced,
                aliases,
                &buchi_config,
            ),
            _ => Verdict::CannotCompute,
        };
        if ltl_lane_debug_enabled() {
            eprintln!("LTL lane-debug [buchi]: {} = {verdict:?}", prop.id);
        }
        buchi_map.insert(prop.id.clone(), verdict);
    }

    retry_ltl_on_full_graph(
        net,
        properties,
        aliases,
        &buchi_order,
        &mut buchi_map,
        config,
    );

    if ltl_rolling_budget_enabled() {
        let retry_indices: Vec<usize> = buchi_order
            .iter()
            .copied()
            .filter(|&index| {
                let prop = &properties[index];
                buchi_map.get(&prop.id) == Some(&Verdict::CannotCompute)
            })
            .collect();

        for &index in &retry_indices {
            if config
                .deadline()
                .is_some_and(|deadline| deadline <= Instant::now())
            {
                break;
            }

            let prop = &properties[index];
            let retry_config = config
                .clone()
                .with_deadline(buchi_property_deadline(config.deadline(), 1));
            let verdict = match &prop.formula {
                Formula::Ltl(ltl) => buchi_verdict_crosschecked(
                    prop,
                    ltl,
                    classifications[index].as_ref(),
                    net,
                    &reduced,
                    &identity_reduced,
                    aliases,
                    &retry_config,
                ),
                _ => Verdict::CannotCompute,
            };
            if verdict != Verdict::CannotCompute {
                buchi_map.insert(prop.id.clone(), verdict);
            }
        }
    }

    if flush {
        for &index in &unresolved_indices {
            let prop = &properties[index];
            // Skip ids already flushed by the early full-graph pass — re-emitting
            // here produces duplicate FORMULA lines for the same property id.
            if flushed_ids.contains(&prop.id) {
                continue;
            }
            let verdict = buchi_map
                .get(&prop.id)
                .copied()
                .unwrap_or(Verdict::CannotCompute);
            // Route through canonical formula_line — see codex #4.
            crate::output::print_mcc_line(crate::output::formula_line_with_techniques(
                "", &prop.id, verdict, techniques,
            ));
            flushed_ids.insert(prop.id.clone());
        }
    }

    // ── Phase 4: merge all results preserving original property order ──
    properties
        .iter()
        .filter_map(|prop| {
            if flushed_ids.contains(&prop.id) {
                return None;
            }
            let verdict = invariant_map
                .get(&prop.id)
                .or_else(|| prefilter_map.get(&prop.id))
                .or_else(|| ctl_fallback_map.get(&prop.id))
                .or_else(|| ctl_true_sufficient_map.get(&prop.id))
                .or_else(|| buchi_map.get(&prop.id))
                .copied()
                .unwrap_or(Verdict::CannotCompute);
            Some((prop.id.clone(), verdict))
        })
        .collect()
}

/// Fraction of the remaining wall budget handed to the EARLY exact full-graph
/// LTL build. The full-graph path is the all-formulas-at-once exact oracle:
/// when the reachable graph is enumerable it builds quickly and returns every
/// residual verdict, so a generous fraction costs nothing on the nets it
/// solves. The reserved `1 - FRACTION` protects the on-the-fly Büchi loop on
/// non-enumerable nets, where the build cannot complete anyway. Mirrors
/// `CTL_FULL_GRAPH_BUDGET_FRACTION` but is set more conservatively (0.5) because
/// Büchi — not a per-property local checker — is LTL's complete fallback lane.
const LTL_EARLY_FULL_GRAPH_BUDGET_FRACTION: f64 = 0.5;

/// Minimum remaining wall budget below which the early full-graph build is
/// skipped entirely. Building the marking-annotated graph has a fixed cost;
/// below this we let Büchi have the whole tail rather than truncating a build
/// that cannot finish.
const LTL_EARLY_FULL_GRAPH_MIN_BUDGET: Duration = Duration::from_millis(500);

fn ltl_early_full_graph_deadline_at(
    global_deadline: Option<Instant>,
    now: Instant,
) -> Option<Instant> {
    global_deadline.map(|deadline| {
        let remaining = deadline.saturating_duration_since(now);
        now + remaining.mul_f64(LTL_EARLY_FULL_GRAPH_BUDGET_FRACTION)
    })
}

fn ltl_early_full_graph_has_budget_at(global_deadline: Option<Instant>, now: Instant) -> bool {
    match global_deadline {
        None => true,
        Some(deadline) => {
            deadline.saturating_duration_since(now) >= LTL_EARLY_FULL_GRAPH_MIN_BUDGET
        }
    }
}

/// Phase 3a: try the EXACT full-graph LTL path early, size-gated.
///
/// Builds the full reachable state graph once on the original net under a
/// bounded wall fraction. If the graph completes (the net is enumerable),
/// resolves every unresolved LTL formula exactly and seeds `buchi_map` with the
/// definite verdicts. Returns the set of property ids resolved here so the
/// Büchi loop can skip them. If the graph does not complete, resolves nothing
/// (the net is not enumerable in the allotted budget) and returns an empty set;
/// every formula then falls through to the Büchi pipeline exactly as before.
///
/// SOUND: `check_ltl_on_full_graph` only returns a verdict for a COMPLETE graph
/// whose product did not overflow or time out; otherwise it returns `None`,
/// which we drop (no `buchi_map` entry), so we never emit a verdict we cannot
/// justify.
fn try_ltl_on_full_graph_early(
    net: &PetriNet,
    properties: &[Property],
    aliases: &PropertyAliases,
    unresolved_indices: &[usize],
    buchi_map: &mut HashMap<String, Verdict>,
    config: &ExplorationConfig,
) -> HashSet<String> {
    let mut resolved: HashSet<String> = HashSet::new();
    if unresolved_indices.is_empty() {
        return resolved;
    }

    let now = Instant::now();
    if !ltl_early_full_graph_has_budget_at(config.deadline(), now) {
        return resolved;
    }

    // Bound the BUILD with a fraction of the remaining budget; per-formula
    // evaluation below still uses the global deadline (cheap on a materialized
    // graph). `refitted_for_full_graph` keeps the memory-sized state cap.
    let build_config = config
        .refitted_for_full_graph(net)
        .with_deadline(ltl_early_full_graph_deadline_at(config.deadline(), now));
    let full = explore_full(net, &build_config);
    if !full.graph.completed {
        eprintln!(
            "LTL early full-graph skipped: exact graph incomplete after {} states",
            full.graph.num_states
        );
        return resolved;
    }

    for &index in unresolved_indices {
        if config
            .deadline()
            .is_some_and(|deadline| deadline <= Instant::now())
        {
            eprintln!("LTL early full-graph stopped: global deadline reached");
            break;
        }

        let prop = &properties[index];
        let Formula::Ltl(ltl) = &prop.formula else {
            continue;
        };
        let mut atom_preds = Vec::new();
        let nnf = to_nnf(ltl, &mut atom_preds);
        let resolved_atoms: Vec<_> = atom_preds
            .iter()
            .map(|pred| crate::buchi::resolve_atom_with_aliases(pred, aliases))
            .collect();

        let verdict =
            match check_ltl_on_full_graph(&nnf, &full, net, &resolved_atoms, config.deadline()) {
                Some(true) => Verdict::True,
                Some(false) => Verdict::False,
                None => Verdict::CannotCompute,
            };
        if verdict != Verdict::CannotCompute {
            eprintln!("LTL early full-graph: {} = {verdict:?}", prop.id);
            buchi_map.insert(prop.id.clone(), verdict);
            resolved.insert(prop.id.clone());
        }
    }

    resolved
}

fn retry_ltl_on_full_graph(
    net: &PetriNet,
    properties: &[Property],
    aliases: &PropertyAliases,
    buchi_order: &[usize],
    buchi_map: &mut HashMap<String, Verdict>,
    config: &ExplorationConfig,
) {
    let full_graph_retry_indices: Vec<usize> = buchi_order
        .iter()
        .copied()
        .filter(|&index| {
            let prop = &properties[index];
            buchi_map.get(&prop.id) == Some(&Verdict::CannotCompute)
        })
        .collect();

    if full_graph_retry_indices.is_empty() {
        return;
    }

    if !ltl_full_graph_retry_has_budget(config.deadline()) {
        eprintln!(
            "LTL full-graph retry skipped: insufficient remaining budget \
             for {} residual formulas",
            full_graph_retry_indices.len()
        );
        return;
    }

    // Refit the config for full-graph exploration so `max_states` is budgeted
    // against the per-state marking width (num_places × 8B) that `explore_full`
    // stores, not the fingerprint-only sizing of the incoming config. Without
    // this, a wide net retries with a state cap sized as if each state cost a
    // fingerprint, and the dense `Vec<Vec<u64>>` markings OOM. Mirrors the CTL
    // pipeline and the LTL early full-graph path. The deadline is preserved.
    let build_config = config.refitted_for_full_graph(net);
    let full = explore_full(net, &build_config);
    if !full.graph.completed {
        eprintln!(
            "LTL full-graph retry skipped: exact graph incomplete after {} states",
            full.graph.num_states
        );
        return;
    }

    for &index in &full_graph_retry_indices {
        if config
            .deadline()
            .is_some_and(|deadline| deadline <= Instant::now())
        {
            eprintln!("LTL full-graph retry stopped: global deadline reached");
            break;
        }

        let prop = &properties[index];
        let Formula::Ltl(ltl) = &prop.formula else {
            continue;
        };
        let mut atom_preds = Vec::new();
        let nnf = to_nnf(ltl, &mut atom_preds);
        let resolved_atoms: Vec<_> = atom_preds
            .iter()
            .map(|pred| crate::buchi::resolve_atom_with_aliases(pred, aliases))
            .collect();

        let verdict =
            match check_ltl_on_full_graph(&nnf, &full, net, &resolved_atoms, config.deadline()) {
                Some(true) => Verdict::True,
                Some(false) => Verdict::False,
                None => Verdict::CannotCompute,
            };
        if verdict != Verdict::CannotCompute {
            eprintln!("LTL full-graph retry: {} = {verdict:?}", prop.id);
            buchi_map.insert(prop.id.clone(), verdict);
        }
    }
}

/// Recognize the desugared response body `X ∨ F q` inside a `G(...)` — the
/// finite-state-exact form of the MCC response card `G(p → F q)` (X = ¬p) —
/// and return its CTL characterization `¬EF(¬X ∧ EG¬q)`. Both `X` and `q`
/// must be pure state predicates and there must be exactly one `F`-child;
/// any other shape returns `None` (the formula then falls through to the CPU
/// Büchi lane, unchanged). See the caller for the soundness argument.
fn ltl_response_globally_to_ctl(globally_body: &LtlFormula) -> Option<CtlFormula> {
    let LtlFormula::Or(children) = globally_body else {
        return None;
    };
    let mut finally_q: Option<StatePredicate> = None;
    let mut antecedent: Vec<StatePredicate> = Vec::new();
    for child in children {
        if let LtlFormula::Finally(body) = child {
            if finally_q.is_some() {
                return None; // two eventualities — not a plain response shape
            }
            finally_q = Some(extract_state_pred_ltl(body)?);
        } else {
            antecedent.push(extract_state_pred_ltl(child)?);
        }
    }
    let q = finally_q?;
    if antecedent.is_empty() {
        return None; // no state-predicate disjunct → plain G(F q), handled above
    }
    // The non-`F` disjuncts OR together to `X = ¬p`; so `p = ¬X`.
    let x = if antecedent.len() == 1 {
        antecedent.pop().expect("len == 1")
    } else {
        StatePredicate::Or(antecedent)
    };
    let p = negate_state_predicate(x);
    let neg_q = negate_state_predicate(q);
    Some(CtlFormula::Not(Box::new(CtlFormula::EF(Box::new(
        CtlFormula::And(vec![
            CtlFormula::Atom(p),
            CtlFormula::EG(Box::new(CtlFormula::Atom(neg_q))),
        ]),
    )))))
}

/// Recognize the persistence fragment `F(G p)` (the LTL persistence property
/// `A(FG p)`) with `p` a pure state predicate, and return its exact finite-path
/// characterization `¬EGF(¬p)`.
///
/// `A(FG p)` fails exactly when some path visits `¬p` infinitely often, which on
/// a finite maximal-path structure is precisely the fair-cycle `EGF(¬p)`; both
/// directions hold. The `CtlFormula::EGF` node lowers to the GPU `CtlOp::EGF`
/// (so `Not(EGF(Atom(¬p)))` is exactly `CtlOp::afg(p)`) and to the CPU
/// `CtlEngine::gfp_egf`, both with the deadlock-stutter successor — a deadlocked
/// `¬p`-state counts as a `p`-avoiding path on both, matching the Büchi lane's
/// deadlock self-loop. Any temporal nesting inside `G` makes
/// `extract_state_pred_ltl` fail and returns `None`, so only the pure
/// persistence shape is routed here (everything else falls through unchanged).
fn ltl_persistence_to_ctl(globally_body: &LtlFormula) -> Option<CtlFormula> {
    let p = extract_state_pred_ltl(globally_body)?;
    Some(CtlFormula::Not(Box::new(CtlFormula::EGF(Box::new(
        CtlFormula::Atom(negate_state_predicate(p)),
    )))))
}

fn ltl_universal_ctl_fallback(formula: &LtlFormula) -> Option<CtlFormula> {
    if let Some(predicate) = extract_state_pred_ltl(formula) {
        return Some(CtlFormula::Atom(predicate));
    }

    match formula {
        LtlFormula::Not(inner) => match inner.as_ref() {
            LtlFormula::Not(double) => ltl_universal_ctl_fallback(double),
            LtlFormula::Finally(body) => extract_state_pred_ltl(body).map(|predicate| {
                CtlFormula::AG(Box::new(CtlFormula::Atom(negate_state_predicate(
                    predicate,
                ))))
            }),
            LtlFormula::Globally(body) => extract_state_pred_ltl(body).map(|predicate| {
                CtlFormula::AF(Box::new(CtlFormula::Atom(negate_state_predicate(
                    predicate,
                ))))
            }),
            _ => None,
        },
        LtlFormula::And(children) => children
            .iter()
            .map(ltl_universal_ctl_fallback)
            .collect::<Option<Vec<_>>>()
            .map(CtlFormula::And),
        LtlFormula::Finally(inner) => {
            if let Some(predicate) = extract_state_pred_ltl(inner) {
                Some(CtlFormula::AF(Box::new(CtlFormula::Atom(predicate))))
            } else if let LtlFormula::Globally(body) = inner.as_ref() {
                // Persistence A(FG p) — F(G p) with p a pure state predicate.
                // Exact finite-path characterization A(FG p) ≡ ¬EGF(¬p); rides
                // the GPU/CPU CTL fair-cycle lane (CtlOp::afg / gfp_egf). Any
                // temporal nesting inside G returns None (falls through to the
                // existing bounded lasso BMC / Büchi persistence lanes).
                ltl_persistence_to_ctl(body)
            } else {
                None
            }
        }
        LtlFormula::Globally(inner) => {
            if let Some(predicate) = extract_state_pred_ltl(inner) {
                Some(CtlFormula::AG(Box::new(CtlFormula::Atom(predicate))))
            } else if let LtlFormula::Finally(body) = inner.as_ref() {
                extract_state_pred_ltl(body).map(|predicate| {
                    CtlFormula::AG(Box::new(CtlFormula::AF(Box::new(CtlFormula::Atom(
                        predicate,
                    )))))
                })
            } else {
                // Response shape A(G(X ∨ F q)) — the desugaring of the common
                // MCC response card A(G(p → F q)) with X = ¬p. Exact CTL
                // characterization on finite maximal-path structures:
                //   A(G(X ∨ F q))  ≡  ¬EF(¬X ∧ EG¬q)
                // (G(X∨Fq) fails at a position where ¬X holds and q never
                // follows, i.e. some reachable ¬X-state has a maximal path
                // avoiding q forever). Sound both ways; the CTL engine's EG
                // accepts a deadlocked ¬q state, matching the Büchi lane's
                // deadlock self-loop, so a deadlock in ¬q counts as a
                // q-avoiding path on both. Rides the GPU CTL lane via Batch 2c.
                ltl_response_globally_to_ctl(inner)
            }
        }
        LtlFormula::Until(left, right) => {
            let left_predicate = extract_state_pred_ltl(left)?;
            let right_predicate = extract_state_pred_ltl(right)?;
            Some(CtlFormula::AU(
                Box::new(CtlFormula::Atom(left_predicate)),
                Box::new(CtlFormula::Atom(right_predicate)),
            ))
        }
        LtlFormula::Next(inner) => ltl_universal_ctl_fallback(inner).map(ctl_stutter_ax),
        LtlFormula::Atom(_) | LtlFormula::Or(_) => None,
    }
}

fn ltl_universal_ctl_true_sufficient(formula: &LtlFormula) -> Option<CtlFormula> {
    if let Some(exact) = ltl_universal_ctl_fallback(formula) {
        return Some(exact);
    }

    ltl_universal_ctl_true_sufficient_syntax(formula).or_else(|| {
        let mut atoms = Vec::new();
        let nnf = to_nnf(formula, &mut atoms);
        ltl_nnf_universal_ctl_true_sufficient(&nnf, &atoms)
    })
}

fn ltl_universal_ctl_true_sufficient_syntax(formula: &LtlFormula) -> Option<CtlFormula> {
    match formula {
        LtlFormula::Not(inner) => extract_state_pred_ltl(inner)
            .map(negate_state_predicate)
            .map(CtlFormula::Atom),
        LtlFormula::And(children) => children
            .iter()
            .map(ltl_universal_ctl_true_sufficient)
            .collect::<Option<Vec<_>>>()
            .map(CtlFormula::And),
        LtlFormula::Or(children) => children
            .iter()
            .map(ltl_universal_ctl_true_sufficient)
            .collect::<Option<Vec<_>>>()
            .map(CtlFormula::Or),
        LtlFormula::Next(inner) => ltl_universal_ctl_true_sufficient(inner).map(ctl_stutter_ax),
        LtlFormula::Finally(inner) => {
            ltl_universal_ctl_true_sufficient(inner).map(|ctl| CtlFormula::AF(Box::new(ctl)))
        }
        LtlFormula::Globally(inner) => {
            ltl_universal_ctl_true_sufficient(inner).map(|ctl| CtlFormula::AG(Box::new(ctl)))
        }
        LtlFormula::Until(left, right) => {
            let left_ctl = ltl_universal_ctl_true_sufficient(left)?;
            let right_ctl = ltl_universal_ctl_true_sufficient(right)?;
            Some(CtlFormula::AU(Box::new(left_ctl), Box::new(right_ctl)))
        }
        LtlFormula::Atom(_) => None,
    }
}

fn ltl_nnf_universal_ctl_true_sufficient(
    formula: &LtlNnf,
    atoms: &[StatePredicate],
) -> Option<CtlFormula> {
    match formula {
        LtlNnf::True => Some(CtlFormula::Atom(StatePredicate::True)),
        LtlNnf::False => Some(CtlFormula::Atom(StatePredicate::False)),
        LtlNnf::Atom(id) => atoms.get(*id).cloned().map(CtlFormula::Atom),
        LtlNnf::NegAtom(id) => atoms
            .get(*id)
            .cloned()
            .map(negate_state_predicate)
            .map(CtlFormula::Atom),
        LtlNnf::And(children) => children
            .iter()
            .map(|child| ltl_nnf_universal_ctl_true_sufficient(child, atoms))
            .collect::<Option<Vec<_>>>()
            .map(CtlFormula::And),
        LtlNnf::Or(children) => children
            .iter()
            .map(|child| ltl_nnf_universal_ctl_true_sufficient(child, atoms))
            .collect::<Option<Vec<_>>>()
            .map(CtlFormula::Or),
        LtlNnf::Next(inner) => {
            ltl_nnf_universal_ctl_true_sufficient(inner, atoms).map(ctl_stutter_ax)
        }
        LtlNnf::Until(left, right) => {
            let left_ctl = ltl_nnf_universal_ctl_true_sufficient(left, atoms)?;
            let right_ctl = ltl_nnf_universal_ctl_true_sufficient(right, atoms)?;
            Some(CtlFormula::AU(Box::new(left_ctl), Box::new(right_ctl)))
        }
        LtlNnf::Release(left, right) => {
            let left_ctl = ltl_nnf_universal_ctl_true_sufficient(left, atoms)?;
            let right_ctl = ltl_nnf_universal_ctl_true_sufficient(right, atoms)?;
            Some(ctl_ar_via_eu(left_ctl, right_ctl))
        }
    }
}

fn ctl_ar_via_eu(left: CtlFormula, right: CtlFormula) -> CtlFormula {
    CtlFormula::Not(Box::new(CtlFormula::EU(
        Box::new(CtlFormula::Not(Box::new(left))),
        Box::new(CtlFormula::Not(Box::new(right))),
    )))
}

fn ctl_stutter_ax(inner: CtlFormula) -> CtlFormula {
    CtlFormula::And(vec![
        CtlFormula::AX(Box::new(inner.clone())),
        CtlFormula::Or(vec![
            CtlFormula::EX(Box::new(CtlFormula::Atom(StatePredicate::True))),
            inner,
        ]),
    ])
}

fn sorted_ltl_buchi_indices(properties: &[Property], unresolved_indices: &[usize]) -> Vec<usize> {
    let mut indices = unresolved_indices.to_vec();
    indices.sort_by_key(|&index| {
        let cost = match &properties[index].formula {
            Formula::Ltl(ltl) => ltl_schedule_cost(ltl),
            _ => u32::MAX,
        };
        (cost, index)
    });
    indices
}

fn ltl_schedule_cost(formula: &LtlFormula) -> u32 {
    match formula {
        LtlFormula::Atom(_) => 1,
        LtlFormula::Not(inner) => 1 + ltl_schedule_cost(inner),
        LtlFormula::Next(inner) => 4 + ltl_schedule_cost(inner),
        LtlFormula::Finally(inner) | LtlFormula::Globally(inner) => 2 + ltl_schedule_cost(inner),
        LtlFormula::And(children) | LtlFormula::Or(children) => {
            2 + children.iter().map(ltl_schedule_cost).sum::<u32>()
        }
        LtlFormula::Until(left, right) => 8 + ltl_schedule_cost(left) + ltl_schedule_cost(right),
    }
}

/// Emit canonical `FORMULA <id> <verdict>` lines for any property whose
/// verdict is in `resolved` and that has not been flushed yet, then mark it
/// flushed. Each `print_mcc_line` call flushes stdout (see
/// [`crate::output::print_mcc_line`]) so a subsequent SIGTERM cannot strand
/// the line in a userspace buffer.
///
/// Called between LTL phases so partial verdicts survive deadline-triggered
/// termination: a BenchKit harness that SIGTERMs the run while a later phase
/// is mid-formula still finds earlier-phase verdicts in stdout. The MCC
/// scorer awards points per emitted FORMULA verdict line, so any
/// successfully-flushed prefix is strictly better than the prior
/// "all-or-nothing" batched flush.
fn flush_resolved_verdicts(
    properties: &[Property],
    resolved: &HashMap<String, Verdict>,
    flushed_ids: &mut HashSet<String>,
    techniques: &Techniques,
) {
    for prop in properties {
        if flushed_ids.contains(&prop.id) {
            continue;
        }
        if let Some(verdict) = resolved.get(&prop.id) {
            // Route through canonical formula_line_with_techniques so the
            // FORMULA / TECHNIQUES keywords come from mcc_keywords.rs.
            crate::output::print_mcc_line(crate::output::formula_line_with_techniques(
                "", &prop.id, *verdict, techniques,
            ));
            flushed_ids.insert(prop.id.clone());
        }
    }
}

fn fair_share_budget(remaining: Duration, remaining_count: usize) -> Duration {
    let divisor = remaining_count.clamp(1, u32::MAX as usize) as u32;
    remaining / divisor
}

fn fair_share_budget_with_virtual_lanes(
    remaining: Duration,
    active_count: usize,
    virtual_downstream_lanes: usize,
) -> Duration {
    let lanes = active_count
        .saturating_add(virtual_downstream_lanes)
        .clamp(1, u32::MAX as usize);
    fair_share_budget(remaining, lanes)
}

fn ltl_prefilter_deadline_at(
    global_deadline: Option<Instant>,
    remaining_count: usize,
    now: Instant,
) -> Option<Instant> {
    global_deadline.map(|deadline| {
        let remaining = deadline.saturating_duration_since(now);
        if remaining.is_zero() {
            return now;
        }

        let budget = LTL_PREFILTER_PHASE_CAP.min(fair_share_budget_with_virtual_lanes(
            remaining,
            remaining_count,
            LTL_PREFILTER_BUCHI_VIRTUAL_LANES,
        ));
        if budget < LTL_PREFILTER_MIN_BUDGET {
            now
        } else {
            now + budget
        }
    })
}

/// Restores the pre-568cb130 prefilter wrappers: a hard 1 s phase cap with
/// fair-share allocation that reserves one virtual lane for the downstream
/// Büchi product. The "adaptive" 2-30 s variant introduced in 568cb130
/// produced a measured -35 exact_units regression on a 26-model LTL subset
/// because the prefilter phases starved the Büchi stage of wall-clock budget
/// even when their own per-phase work would not have completed. The cap also
/// bounds prefilter cost on competition deadlines where remaining/lanes is
/// generous: any extra prefilter time beyond 1 s consistently traded off
/// against Büchi completions on hard formulas.
fn ltl_prefilter_deadline(
    global_deadline: Option<Instant>,
    remaining_count: usize,
) -> Option<Instant> {
    ltl_prefilter_deadline_at(global_deadline, remaining_count, Instant::now())
}

fn ltl_prefilter_has_budget(global_deadline: Option<Instant>, remaining_count: usize) -> bool {
    let now = Instant::now();
    match ltl_prefilter_deadline_at(global_deadline, remaining_count, now) {
        None => true,
        Some(deadline) => deadline.saturating_duration_since(now) >= LTL_PREFILTER_MIN_BUDGET,
    }
}

fn ltl_full_graph_retry_has_budget(global_deadline: Option<Instant>) -> bool {
    ltl_full_graph_retry_has_budget_at(global_deadline, Instant::now())
}

fn ltl_full_graph_retry_has_budget_at(global_deadline: Option<Instant>, now: Instant) -> bool {
    match global_deadline {
        None => true,
        Some(deadline) => deadline > now,
    }
}

fn buchi_property_deadline(
    global_deadline: Option<Instant>,
    _remaining_count: usize,
) -> Option<Instant> {
    // Buchi is the complete LTL solver path. A per-property fair-share slice can
    // turn known verdicts into CannotCompute even when the global examination
    // still has enough time for the hard formula and the remaining queue.
    global_deadline
}

/// Rolling-residual per-formula Buchi deadline.
///
/// Splits the time remaining at `now` evenly among the still-unresolved
/// formulas plus one virtual lane for exact full-system-graph retry. Formulas
/// that finish under-share contribute their slack to later formulas, because
/// subsequent calls observe both a smaller `remaining_count` and a `now` that
/// did not consume the share.
///
/// When the global deadline has already expired, returns the global deadline so
/// callers preserve fail-closed semantics (the Buchi solver still returns
/// CannotCompute for an expired deadline).
fn buchi_rolling_share_deadline_at(
    global_deadline: Option<Instant>,
    remaining_count: usize,
    now: Instant,
) -> Option<Instant> {
    global_deadline.map(|deadline| {
        let remaining = deadline.saturating_duration_since(now);
        if remaining.is_zero() {
            // Past the global deadline — propagate the (now-stale) global
            // deadline so the Buchi solver short-circuits to CannotCompute.
            return deadline;
        }
        let lane_count = remaining_count
            .saturating_add(1)
            .clamp(1, u32::MAX as usize);
        let effective = fair_share_budget(remaining, lane_count);
        now + effective
    })
}

#[cfg(test)]
fn buchi_rolling_share_deadline(
    global_deadline: Option<Instant>,
    remaining_count: usize,
) -> Option<Instant> {
    buchi_rolling_share_deadline_at(global_deadline, remaining_count, Instant::now())
}

/// Dispatch between the historical (full global deadline) and the opt-in
/// rolling-residual Buchi per-formula budget.
fn buchi_per_formula_deadline(
    global_deadline: Option<Instant>,
    remaining_count: usize,
) -> Option<Instant> {
    if ltl_rolling_budget_enabled() {
        buchi_rolling_share_deadline_at(global_deadline, remaining_count, Instant::now())
    } else {
        buchi_property_deadline(global_deadline, remaining_count)
    }
}

/// Run the Buchi product pipeline for a single LTL property.
fn check_single_ltl_buchi(
    prop: &Property,
    ltl: &LtlFormula,
    class: Option<&ShallowLtl>,
    net: &PetriNet,
    reduced: &ReducedNet,
    aliases: &PropertyAliases,
    config: &ExplorationConfig,
) -> Verdict {
    // Safety: detect formulas with unresolved names.
    let (total, unresolved) = count_unresolved_ltl(ltl, aliases);
    if unresolved > 0 {
        eprintln!(
            "LTL resolution guard: {} has {unresolved}/{total} \
             unresolved names → CANNOT_COMPUTE",
            prop.id
        );
        return Verdict::CannotCompute;
    }

    // Convert LTL formula to NNF early so we can gate on X.
    let mut atom_preds = Vec::new();
    let nnf = to_nnf(ltl, &mut atom_preds);
    let has_next = formula_contains_next(&nnf);

    // Resolve atoms to indices on the original net. The lasso-BMC lane currently
    // runs on the original net only, so predicates and model replay share one
    // marking space.
    let resolved_atoms: Vec<_> = atom_preds
        .iter()
        .map(|pred| crate::buchi::resolve_atom_with_aliases(pred, aliases))
        .collect();

    if should_run_lasso_bmc(class) {
        let lasso_deadline = lasso_bmc_deadline(config.deadline());
        if lasso_bmc_has_budget(lasso_deadline) {
            if let Some(witness) =
                find_ltl_lasso_counterexample(net, &nnf, &resolved_atoms, lasso_deadline)
            {
                eprintln!(
                    "LTL lasso BMC depth {}: {} = FALSE (validated accepting lasso)",
                    witness.depth, prop.id
                );
                return Verdict::False;
            }
        }
    }

    // Per-property query slicing on the reduced net.
    //
    // SOUNDNESS: the slice is explored as the complete system, and the Büchi
    // product adds a self-loop on any state with no successors (a deadlock).
    // If slicing drops a structurally-disjoint concurrent component that can
    // still fire in the full net, a non-deadlock state becomes a slice-deadlock
    // — manufacturing a spurious infinite stutter (or erasing a real infinite
    // run). That flips both stutter-sensitive (X) verdicts and liveness
    // (F/G/U) Büchi verdicts. Slicing only trims forward-only sink *places*
    // (every transition preserved) when the support is a single connected
    // component containing all atoms; in that case deadlock and next-step
    // structure are identical to the full net and the slice is answer-
    // preserving. Otherwise we discard the slice and explore the full reduced
    // net — exact, never a wrong Büchi verdict.
    let slice = ltl_property_support_with_aliases(reduced, prop, aliases)
        .and_then(|support| {
            if has_next {
                let closure = closure_on_reduced_net(&reduced.net, support);
                build_query_slice(&reduced.net, &closure)
            } else {
                let cone = relevance_cone_on_reduced_net(&reduced.net, support);
                build_query_local_slice(&reduced.net, &cone)
            }
        })
        .filter(|slice| {
            let safe = slice.net.num_transitions() == reduced.net.num_transitions();
            if !safe {
                eprintln!(
                    "LTL slice drops {} transition(s) (disjoint component; \
                     X/liveness Büchi unsound) -> exploring full reduced net: {}",
                    reduced.net.num_transitions() - slice.net.num_transitions(),
                    prop.id
                );
            }
            safe
        });

    // When a slice was produced, build a composed ReducedNet.
    let composed;
    let (explore_net, explore_reduced): (&PetriNet, &ReducedNet) = if let Some(ref s) = slice {
        composed = compose_slice_and_reduction(s, reduced);
        (&s.net, &composed)
    } else {
        (&reduced.net, reduced)
    };

    // Stutter-insensitive partial-order reduction on the product.
    //
    // POR is sound only for stutter-insensitive (X-free) LTL. A "visible"
    // transition is any reduced/explore-net transition that can change the
    // truth of a Büchi atom; `ltl_visible_reduced_transitions` returns a sound
    // over-approximation. The on-the-fly DFS builder (`on_the_fly.rs`) enforces
    // the ample-set conditions C0–C3 (in particular the cycle proviso C3 via the
    // DFS stack), so the reduced product is stutter-trace equivalent to the full
    // product — accepting-run existence, hence the verdict, is preserved.
    // Formulas containing Next disable POR entirely (`has_next`).
    let por = if has_next || env_flag_enabled("TY_MCC_DISABLE_LTL_POR") {
        None
    } else {
        let visible = ltl_visible_reduced_transitions(&resolved_atoms, explore_reduced);
        // POR only prunes interleavings of INVISIBLE transitions, and the DFS
        // path adds per-state stubborn-set + cycle-proviso overhead. When too
        // few transitions are invisible the reduction cannot outweigh that
        // overhead, so fall back to the exact BFS product (identical verdict,
        // no DFS cost). Threshold: require >25% of transitions to be invisible
        // (visible ≤ 75%). Purely a performance gate — both paths are sound.
        let nt = explore_net.num_transitions();
        if visible.len() * 4 >= nt * 3 {
            None
        } else {
            // Per-Büchi-state visibility (port-plan P2): the DFS additionally
            // builds per-GBA-state rows (reachability-closed guard-atom sets)
            // and tests C2 against the current node's row instead of the
            // static whole-formula set. Every row is a subset of `visible`,
            // so ample sets stay small more often; any anomaly falls back to
            // the static row per node. The perf gate above stays on the
            // static set (per-state rows are subsets — only more permissive).
            Some(PorContext {
                dep: DependencyGraph::build(explore_net),
                visible,
                per_state_visibility: !env_flag_enabled("TY_MCC_DISABLE_LTL_PER_STATE_VIS"),
            })
        }
    };

    // NOTE: the additive symbolic-BDD LTL lane runs EARLIER, as Phase 2.5 in
    // `check_ltl_properties_inner` (before the expensive explicit full-graph
    // build), so it gets the budget on large nets the explicit product cannot
    // complete. By the time a formula reaches this on-the-fly path the symbolic
    // lane has already declined it (fail-closed), so there is no symbolic
    // attempt here — the exact explicit product decides the residual.

    // On-the-fly Buchi product.
    match check_ltl_on_the_fly(
        &nnf,
        explore_net,
        explore_reduced,
        net,
        &resolved_atoms,
        por.as_ref(),
        config.max_states(),
        config.deadline(),
    ) {
        Ok(Some(true)) => Verdict::True,
        Ok(Some(false)) => Verdict::False,
        Ok(None) => Verdict::CannotCompute,
        Err(error) => {
            eprintln!("LTL: {} → CANNOT_COMPUTE ({error})", prop.id);
            Verdict::CannotCompute
        }
    }
}

fn should_run_lasso_bmc(class: Option<&ShallowLtl>) -> bool {
    ltl_lasso_bmc_enabled() && matches!(class, Some(ShallowLtl::LassoBmcLivenessCandidate))
}

/// Investigation-only kill switch (`TY_MCC_DISABLE_LTL_SYMBOLIC=1`) for the
/// additive symbolic LTL lane, so the explicit Büchi path can be exercised in
/// isolation. Default off (lane enabled).
fn ltl_symbolic_disabled() -> bool {
    env_flag_enabled("TY_MCC_DISABLE_LTL_SYMBOLIC")
}

/// `TY_MCC_DISABLE_LTL_WALK=1` — kill switch for the Phase-2.6 additive
/// FALSE-witness random-walk lane. Default off (lane runs).
fn ltl_walk_disabled() -> bool {
    env_flag_enabled("TY_MCC_DISABLE_LTL_WALK")
}

/// Per-pending-formula wall slice for the Phase-2.6 walk: `min(remaining/16,
/// LTL_WALK_PER_FORMULA_CAP)`. Random-walk lassos that exist are found in well
/// under a second, so a small per-formula cap keeps each attempt from grinding
/// while the global cap (below) bounds the whole pass.
const LTL_WALK_PER_FORMULA_CAP: Duration = Duration::from_secs(2);

/// Per-formula divisor of the remaining budget for the walk slice.
const LTL_WALK_PER_FORMULA_DIVISOR: u32 = 16;

/// Absolute ceiling on the cumulative wall time the WHOLE Phase-2.6 pass may
/// consume: `min(initial_remaining/8, LTL_WALK_GLOBAL_CAP)`. Once this elapses,
/// the remaining pending formulas fall through to the exact lanes untouched, so
/// the walk can never rob more than this from the time-sensitive explicit
/// Phase 3/3a (the AutonomousCar-PT-01b CORRECT→CC hazard).
const LTL_WALK_GLOBAL_CAP: Duration = Duration::from_secs(8);

/// Global-cap divisor of the budget remaining at the start of the Phase-2.6 pass.
const LTL_WALK_GLOBAL_DIVISOR: u32 = 8;

/// Minimum leftover budget for a single Phase-2.6 walk: below this the walk is
/// skipped so it never nibbles the slice the exact lanes need.
const LTL_WALK_MIN_BUDGET: Duration = Duration::from_millis(200);

/// SOUND, fail-closed symbolic LTL verdict for `A(formula)` (gated on
/// `dd-backend`).
///
/// Builds the system × GBA(¬formula) product symbolically over the ORIGINAL
/// `net` and asks the oracle-verified
/// [`tla_dd::symbolic_ltl::symbolic_ltl_has_accepting_cycle_ordered`] whether
/// it has a reachable accepting cycle. A fair accepting lasso of the negated
/// property is a counterexample, so:
///
/// * `Some(true)`  (accepting cycle) ⇒ `A(formula)` is **FALSE**;
/// * `Some(false)` (no accepting cycle) ⇒ `A(formula)` is **TRUE**.
///
/// Returns `None` (decline) on EVERY gate — net not DD-eligible (bound/place
/// caps), any atom unsupported by the DD encoding, GBA too large, the product
/// reachability / fixpoint over budget, OOM. The caller then falls through to
/// the EXISTING explicit lanes unchanged. Never a guessed verdict.
#[cfg(feature = "dd-backend")]
#[must_use]
fn try_symbolic_ltl_verdict(
    nnf: &LtlNnf,
    net: &PetriNet,
    resolved_atoms: &[crate::resolved_predicate::ResolvedPredicate],
    global_deadline: Option<Instant>,
) -> Option<Verdict> {
    if ltl_symbolic_disabled() {
        return None;
    }
    let dbg = env_flag_enabled("TY_LTL_SYMBOLIC_DEBUG");
    // Gate 1: sound DD spec of the original net (place indices preserved).
    let Some((spec, _bounds)) = crate::examinations::dd_spec::build_sound_dd_spec(net) else {
        if dbg {
            eprintln!("LTL symbolic: DECLINE gate1 (DD spec: bound/place cap or unbounded)");
        }
        return None;
    };
    let num_places = net.num_places();
    let num_transitions = net.num_transitions();
    // Gate 2: build the SymbolicGba, lowering each LTL atom to a DdPredicate.
    // Any unsupported atom (or out-of-range index) makes the whole conversion
    // decline → fall through to the explicit path.
    let Some(gba) = crate::buchi::gba_to_symbolic(nnf, |atom_idx| {
        let pred = resolved_atoms.get(atom_idx)?;
        crate::examinations::dd_spec::translate_predicate(pred, num_places, num_transitions)
    }) else {
        if dbg {
            eprintln!("LTL symbolic: DECLINE gate2 (GBA atom unsupported / index out of range)");
        }
        return None;
    };
    if dbg {
        eprintln!(
            "LTL symbolic: gates 1-2 OK (places={num_places}, gba_states={}, accept_sets={})",
            gba.num_states,
            gba.acceptance.len(),
        );
    }
    // Gate 3: the symbolic emptiness check on the NATIVE tla-bdd LTL engine
    // (`symbolic_ltl_has_accepting_cycle_via_bdd`), run on a detached worker
    // thread with the big DD stack + the caller's deadline (so a high-bound
    // binary-band net completes or declines, never overflows the caller stack).
    // The tla-bdd port takes the deadline directly and declines (fail-closed)
    // rather than overrun it; the worker boundary turns any panic into a clean
    // DECLINE, and a non-returning worker is bounded by `recv_timeout`. `None`
    // (decline / timeout / spawn failure / panic) ⇒ fall through to the explicit
    // lanes, sound.
    use std::sync::mpsc;
    let budget = global_deadline.map(|d| d.saturating_duration_since(Instant::now()));
    if let Some(b) = budget {
        if b.is_zero() {
            return None;
        }
    }
    let (tx, rx) = mpsc::channel();
    let handle = std::thread::Builder::new()
        .name("tla-bdd-ltl".into())
        .stack_size(tla_dd::DD_WORKER_STACK_BYTES)
        .spawn(move || {
            let r = crate::examinations::mdd_common::symbolic_ltl_has_accepting_cycle_via_bdd(
                &spec,
                &gba,
                global_deadline,
            );
            let _ = tx.send(r);
        });
    if handle.is_err() {
        if dbg {
            eprintln!("LTL symbolic: DECLINE gate3 (tla-bdd lane spawn failed)");
        }
        return None;
    }
    let recv = match budget {
        Some(b) => rx.recv_timeout(b + Duration::from_millis(1500)),
        None => rx.recv().map_err(|_| mpsc::RecvTimeoutError::Disconnected),
    };
    let has_accepting_cycle = match recv {
        Ok(Some(v)) => v,
        _ => {
            if dbg {
                eprintln!(
                    "LTL symbolic: DECLINE gate3 (product fixpoint over budget / OOM / panic)"
                );
            }
            return None;
        }
    };
    // A reachable accepting lasso of ¬formula ⇔ A(formula) is FALSE.
    Some(if has_accepting_cycle {
        Verdict::False
    } else {
        Verdict::True
    })
}

/// `dd-backend` disabled: the symbolic LTL lane is a no-op that always
/// declines, leaving the existing explicit fallback unchanged.
#[cfg(not(feature = "dd-backend"))]
#[must_use]
fn try_symbolic_ltl_verdict(
    _nnf: &LtlNnf,
    _net: &PetriNet,
    _resolved_atoms: &[crate::resolved_predicate::ResolvedPredicate],
    _global_deadline: Option<Instant>,
) -> Option<Verdict> {
    None
}

#[cfg(test)]
#[path = "ltl_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "ltl_query_slicing_tests.rs"]
mod query_slicing_tests;

#[cfg(all(test, feature = "dd-backend"))]
#[path = "ltl_symbolic_tests.rs"]
mod symbolic_tests;
