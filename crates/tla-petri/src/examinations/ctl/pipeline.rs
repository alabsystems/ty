// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0
use super::super::maximal_path_suffix::{af_holds_from_mask, eg_holds_from_mask};
use super::super::query_support::{
    closure_on_reduced_net, ctl_support_with_aliases, relevance_cone_on_reduced_net,
};
use super::super::reachability::{
    check_reachability_properties_with_aliases, protected_places_for_prefire,
};
use super::checker::CtlChecker;
use super::local_checker::LocalCtlChecker;
use super::local_edg::{solve_local_edg, solve_local_edg_capped};
use super::resolve;
use super::routing::{
    classify_shallow_ctl, classify_shallow_ctl_suffix, ctl_batch_contains_next_step, ShallowCtl,
    ShallowCtlSuffix,
};
use super::{is_known_mcc_ctl_soundness_guard, SoundnessGuardMode};
use crate::explorer::{explore_full, ExplorationConfig};
use crate::model::PropertyAliases;
use crate::output::{Techniques, Verdict};
use crate::petri_net::PetriNet;
use crate::property_xml::{CtlFormula, Formula, PathQuantifier, Property, ReachabilityFormula};
use crate::query_slice::{build_query_local_slice, build_query_slice};
use crate::reduction::{reduce_iterative_structural_with_mode, ReducedNet, ReductionMode};
use crate::resolved_predicate::{
    count_unresolved_with_aliases, eval_predicate, resolve_predicate_with_aliases,
};
use std::collections::{HashMap, HashSet};
use std::time::{Duration, Instant};

fn env_flag_enabled(key: &str) -> bool {
    env_flag_setting(key).unwrap_or(false)
}

fn env_flag_setting(key: &str) -> Option<bool> {
    let value = std::env::var(key).ok()?;
    match value.trim().to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" | "on" => Some(true),
        "0" | "false" | "no" | "off" => Some(false),
        _ => None,
    }
}

fn experimental_ctl_shortcuts_enabled() -> bool {
    #[cfg(test)]
    if let Some(enabled) = ctl_shortcuts_test_override() {
        return enabled;
    }

    !env_flag_enabled("TY_MCC_DISABLE_CTL_SHORTCUTS")
        && !matches!(env_flag_setting("TY_MCC_ENABLE_CTL_SHORTCUTS"), Some(false))
}

fn ctl_local_fallback_enabled() -> bool {
    #[cfg(test)]
    if let Some(enabled) = ctl_local_fallback_test_override() {
        return enabled;
    }

    !env_flag_enabled("TY_MCC_DISABLE_CTL_LOCAL_FALLBACK")
        && !matches!(
            env_flag_setting("TY_MCC_ENABLE_CTL_LOCAL_FALLBACK"),
            Some(false)
        )
}

#[cfg(test)]
thread_local! {
    static CTL_SHORTCUTS_TEST_OVERRIDE: std::cell::Cell<Option<bool>> =
        const { std::cell::Cell::new(None) };
    static CTL_LOCAL_FALLBACK_TEST_OVERRIDE: std::cell::Cell<Option<bool>> =
        const { std::cell::Cell::new(None) };
    /// Test-only override for the (default-off) MDD CTL lane gate. `Some(true)`
    /// forces the lane ON regardless of env so the differential / soundness
    /// battery can exercise the lane without mutating process-global env (which
    /// races parallel tests). `None` (unset) defers to the env gate.
    #[cfg(feature = "dd-backend")]
    static MDD_CTL_ENABLED_TEST_OVERRIDE: std::cell::Cell<Option<bool>> =
        const { std::cell::Cell::new(None) };
}

#[cfg(test)]
fn ctl_shortcuts_test_override() -> Option<bool> {
    CTL_SHORTCUTS_TEST_OVERRIDE.with(std::cell::Cell::get)
}

#[cfg(test)]
fn ctl_local_fallback_test_override() -> Option<bool> {
    CTL_LOCAL_FALLBACK_TEST_OVERRIDE.with(std::cell::Cell::get)
}

#[cfg(test)]
struct CtlShortcutsTestOverrideGuard {
    prev: Option<bool>,
}

#[cfg(test)]
struct CtlLocalFallbackTestOverrideGuard {
    prev: Option<bool>,
}

#[cfg(test)]
impl Drop for CtlShortcutsTestOverrideGuard {
    fn drop(&mut self) {
        CTL_SHORTCUTS_TEST_OVERRIDE.with(|slot| slot.set(self.prev));
    }
}

#[cfg(test)]
impl Drop for CtlLocalFallbackTestOverrideGuard {
    fn drop(&mut self) {
        CTL_LOCAL_FALLBACK_TEST_OVERRIDE.with(|slot| slot.set(self.prev));
    }
}

#[cfg(test)]
fn set_ctl_shortcuts_test_override(enabled: bool) -> CtlShortcutsTestOverrideGuard {
    CTL_SHORTCUTS_TEST_OVERRIDE.with(|slot| {
        let prev = slot.get();
        slot.set(Some(enabled));
        CtlShortcutsTestOverrideGuard { prev }
    })
}

#[cfg(test)]
fn set_ctl_local_fallback_test_override(enabled: bool) -> CtlLocalFallbackTestOverrideGuard {
    CTL_LOCAL_FALLBACK_TEST_OVERRIDE.with(|slot| {
        let prev = slot.get();
        slot.set(Some(enabled));
        CtlLocalFallbackTestOverrideGuard { prev }
    })
}

#[cfg(test)]
pub(super) fn with_experimental_ctl_shortcuts_for_test<T>(
    enabled: bool,
    f: impl FnOnce() -> T,
) -> T {
    let _guard = set_ctl_shortcuts_test_override(enabled);
    f()
}

#[cfg(test)]
pub(super) fn with_ctl_local_fallback_for_test<T>(enabled: bool, f: impl FnOnce() -> T) -> T {
    let _guard = set_ctl_local_fallback_test_override(enabled);
    f()
}

#[cfg(all(test, feature = "dd-backend"))]
fn mdd_ctl_enabled_test_override() -> Option<bool> {
    MDD_CTL_ENABLED_TEST_OVERRIDE.with(std::cell::Cell::get)
}

#[cfg(all(test, feature = "dd-backend"))]
struct MddCtlEnabledTestOverrideGuard {
    prev: Option<bool>,
}

#[cfg(all(test, feature = "dd-backend"))]
impl Drop for MddCtlEnabledTestOverrideGuard {
    fn drop(&mut self) {
        MDD_CTL_ENABLED_TEST_OVERRIDE.with(|slot| slot.set(self.prev));
    }
}

#[cfg(all(test, feature = "dd-backend"))]
fn set_mdd_ctl_enabled_test_override(enabled: bool) -> MddCtlEnabledTestOverrideGuard {
    MDD_CTL_ENABLED_TEST_OVERRIDE.with(|slot| {
        let prev = slot.get();
        slot.set(Some(enabled));
        MddCtlEnabledTestOverrideGuard { prev }
    })
}

/// Run `f` with the MDD CTL lane forced ON (default-off in production). Used by
/// the differential / soundness battery so the lane is actually exercised.
#[cfg(all(test, feature = "dd-backend"))]
fn with_mdd_ctl_enabled_for_test<T>(f: impl FnOnce() -> T) -> T {
    let _guard = set_mdd_ctl_enabled_test_override(true);
    f()
}

fn fair_share_budget(remaining: Duration, remaining_count: usize) -> Duration {
    let divisor = remaining_count.clamp(1, u32::MAX as usize) as u32;
    remaining / divisor
}

/// Fraction of the remaining wall budget handed to the simplifier pre-pass.
///
/// The simplifier is a cheap structural/LP front-filter whose LP deciders were
/// historically un-bounded — on a 461-place net (SharedMemory-PT-000020) the
/// trap/state-equation CEGAR accumulated to ~124 s, a 4x overrun of a 30 s
/// budget. Capping the pre-pass at a quarter of the remaining budget keeps the
/// front-filter useful while reserving the bulk for the temporal engine; an
/// LP that times out returns inconclusive, leaving the formula unchanged
/// (verdict-preserving).
const CTL_SIMPLIFY_BUDGET_FRACTION: f64 = 0.25;

fn ctl_simplify_deadline_at(global_deadline: Option<Instant>, now: Instant) -> Option<Instant> {
    global_deadline.map(|deadline| {
        let remaining = deadline.saturating_duration_since(now);
        now + remaining.mul_f64(CTL_SIMPLIFY_BUDGET_FRACTION)
    })
}

fn ctl_simplify_deadline(global_deadline: Option<Instant>) -> Option<Instant> {
    ctl_simplify_deadline_at(global_deadline, Instant::now())
}

fn fair_share_deadline_at(
    global_deadline: Option<Instant>,
    remaining_count: usize,
    now: Instant,
) -> Option<Instant> {
    global_deadline.map(|deadline| {
        now + fair_share_budget(deadline.saturating_duration_since(now), remaining_count)
    })
}

fn fair_share_deadline(
    global_deadline: Option<Instant>,
    remaining_count: usize,
) -> Option<Instant> {
    fair_share_deadline_at(global_deadline, remaining_count, Instant::now())
}

fn ctl_full_graph_deadline(
    global_deadline: Option<Instant>,
    unresolved_count: usize,
    local_fallback_enabled: bool,
) -> Option<Instant> {
    ctl_full_graph_deadline_at(
        global_deadline,
        unresolved_count,
        local_fallback_enabled,
        Instant::now(),
    )
}

/// Fraction of the remaining wall budget handed to the sound full-graph CTL
/// path. The full-graph path is the all-properties-at-once exact oracle: when
/// the reachable graph is enumerable it builds the graph and returns EARLY with
/// every property resolved, so a generous deadline costs nothing on the models
/// it can solve. The remaining `1 - FRACTION` is reserved for the per-property
/// local fallback, which only does useful work on non-enumerable nets where the
/// full-graph path cannot finish anyway.
const CTL_FULL_GRAPH_BUDGET_FRACTION: f64 = 0.75;

/// Per-property wall cap for the EDG-first pre-pass. The local
/// certain-zero EDG decides "easy" roots (EF-true, AG-false, bounded
/// AF/EG cones) in milliseconds by terminating the instant the root
/// is decided either way; properties that exhaust the node cap abort
/// well under this limit, so the cap only guards pathological
/// successor-expansion stalls.
const CTL_EDG_PREPASS_MAX_PER_PROP: Duration = Duration::from_millis(500);

/// Fraction of the remaining wall budget the whole EDG pre-pass may
/// consume across all properties. Keeps the pre-pass a cheap
/// front-filter: the full-graph batch path still receives the bulk
/// of the budget on enumerable nets.
const CTL_EDG_PREPASS_BUDGET_FRACTION: f64 = 0.15;

/// EDG node cap for the pre-pass. Small enough that a property whose
/// local search cannot early-exit aborts in well under
/// [`CTL_EDG_PREPASS_MAX_PER_PROP`]; large enough to fully exhaust
/// small state spaces (where exhaustion itself decides the root).
const CTL_EDG_PREPASS_NODE_CAP: usize = 200_000;

fn ctl_full_graph_deadline_at(
    global_deadline: Option<Instant>,
    unresolved_count: usize,
    local_fallback_enabled: bool,
    now: Instant,
) -> Option<Instant> {
    if !local_fallback_enabled {
        return global_deadline;
    }

    // Previously this handed the full-graph path only a fair-share
    // `remaining/(unresolved+1)` slice (~1/17 of the budget for a 16-property
    // bundle), which starved it on mid-size enumerable nets: the graph build
    // timed out, `full.graph.completed` stayed false, and EVERY property was
    // forced through the slower per-property local fallback — producing
    // CANNOT_COMPUTE on nets that are trivially enumerable (verified:
    // Anderson-PT-05 CTLCardinality returned 10 CC under the old split).
    // Give the full-graph path the majority of the remaining budget instead;
    // the reserve below still protects the per-property fallback.
    let _ = unresolved_count;
    global_deadline.map(|deadline| {
        let remaining = deadline.saturating_duration_since(now);
        now + remaining.mul_f64(CTL_FULL_GRAPH_BUDGET_FRACTION)
    })
}

#[cfg(test)]
pub(super) fn ctl_full_graph_deadline_for_test(
    global_deadline: Option<Instant>,
    unresolved_count: usize,
    local_fallback_enabled: bool,
    now: Instant,
) -> Option<Instant> {
    ctl_full_graph_deadline_at(
        global_deadline,
        unresolved_count,
        local_fallback_enabled,
        now,
    )
}

fn flush_shallow_results(
    properties: &[Property],
    shallow_map: &HashMap<String, Verdict>,
    flushed_ids: &mut HashSet<String>,
    flush: bool,
    techniques: &Techniques,
) {
    if !flush {
        return;
    }

    for prop in properties {
        if flushed_ids.contains(&prop.id) {
            continue;
        }
        if let Some(verdict) = shallow_map.get(&prop.id) {
            // Route through the canonical formula_line emitter so all
            // FORMULA / TECHNIQUES keywords come from mcc_keywords.rs
            // and Verdict::Display. Closes codex audit finding #4.
            crate::output::print_mcc_line(crate::output::formula_line_with_techniques(
                "", &prop.id, *verdict, techniques,
            ));
            flushed_ids.insert(prop.id.clone());
        }
    }
}

fn flush_property_result(
    property_id: &str,
    verdict: Verdict,
    flushed_ids: &mut HashSet<String>,
    flush: bool,
    techniques: &Techniques,
) -> bool {
    if !flush || flushed_ids.contains(property_id) {
        return false;
    }

    // Route through the canonical formula_line emitter — see codex #4
    // comment above.
    crate::output::print_mcc_line(crate::output::formula_line_with_techniques(
        "",
        property_id,
        verdict,
        techniques,
    ));
    flushed_ids.insert(property_id.to_string());
    true
}

fn collect_shallow_or_cannot_compute(
    properties: &[Property],
    shallow_map: &HashMap<String, Verdict>,
    flushed_ids: &HashSet<String>,
) -> Vec<(String, Verdict)> {
    properties
        .iter()
        .filter_map(|prop| {
            if flushed_ids.contains(&prop.id) {
                return None;
            }
            let verdict = shallow_map
                .get(&prop.id)
                .copied()
                .unwrap_or(Verdict::CannotCompute);
            Some((prop.id.clone(), verdict))
        })
        .collect()
}

fn collect_pending_expensive_ctl_properties(
    properties: &[Property],
    shallow_map: &HashMap<String, Verdict>,
    flushed_ids: &HashSet<String>,
    guard_mode: SoundnessGuardMode,
) -> Vec<Property> {
    properties
        .iter()
        .filter(|prop| !flushed_ids.contains(&prop.id))
        .filter(|prop| !shallow_map.contains_key(&prop.id))
        .filter(|prop| matches!(prop.formula, Formula::Ctl(_)))
        .filter(|prop| {
            guard_mode != SoundnessGuardMode::Enforce || !is_known_mcc_ctl_soundness_guard(&prop.id)
        })
        .cloned()
        .collect()
}

#[cfg(test)]
pub(super) fn pending_expensive_ctl_properties_for_test(
    properties: &[Property],
    shallow_ids: &[&str],
) -> Vec<Property> {
    let shallow_map: HashMap<String, Verdict> = shallow_ids
        .iter()
        .map(|id| ((*id).to_string(), Verdict::CannotCompute))
        .collect();
    collect_pending_expensive_ctl_properties(
        properties,
        &shallow_map,
        &HashSet::new(),
        SoundnessGuardMode::Ignore,
    )
}

fn collect_suffix_ctl_properties(
    properties: &[Property],
    shallow_map: &HashMap<String, Verdict>,
    shortcuts_enabled: bool,
) -> Vec<Property> {
    if !shortcuts_enabled {
        return Vec::new();
    }

    properties
        .iter()
        .filter(|prop| !shallow_map.contains_key(&prop.id))
        .filter(|prop| {
            let Formula::Ctl(ctl) = &prop.formula else {
                return false;
            };
            classify_shallow_ctl_suffix(ctl).is_some()
        })
        .cloned()
        .collect()
}

#[cfg(test)]
pub(super) fn suffix_ctl_properties_for_test(
    properties: &[Property],
    shallow_ids: &[&str],
) -> Vec<Property> {
    let shallow_map: HashMap<String, Verdict> = shallow_ids
        .iter()
        .map(|id| ((*id).to_string(), Verdict::CannotCompute))
        .collect();
    collect_suffix_ctl_properties(properties, &shallow_map, true)
}

fn initial_marking_forced_verdict(
    formula: &CtlFormula,
    aliases: &PropertyAliases,
    net: &PetriNet,
) -> Option<Verdict> {
    initial_marking_forced_bool(formula, aliases, net).map(|holds| {
        if holds {
            Verdict::True
        } else {
            Verdict::False
        }
    })
}

fn initial_marking_forced_bool(
    formula: &CtlFormula,
    aliases: &PropertyAliases,
    net: &PetriNet,
) -> Option<bool> {
    match formula {
        CtlFormula::Atom(predicate) => {
            let resolved = resolve_predicate_with_aliases(predicate, aliases);
            Some(eval_predicate(&resolved, &net.initial_marking, net))
        }
        CtlFormula::Not(inner) => initial_marking_forced_bool(inner, aliases, net).map(|v| !v),
        CtlFormula::And(children) => {
            let mut all_true = true;
            for child in children {
                match initial_marking_forced_bool(child, aliases, net) {
                    Some(true) => {}
                    Some(false) => return Some(false),
                    None => all_true = false,
                }
            }
            all_true.then_some(true)
        }
        CtlFormula::Or(children) => {
            let mut all_false = true;
            for child in children {
                match initial_marking_forced_bool(child, aliases, net) {
                    Some(true) => return Some(true),
                    Some(false) => {}
                    None => all_false = false,
                }
            }
            all_false.then_some(false)
        }
        CtlFormula::EF(inner) | CtlFormula::AF(inner) => {
            matches!(initial_marking_forced_bool(inner, aliases, net), Some(true)).then_some(true)
        }
        CtlFormula::EG(inner) | CtlFormula::AG(inner) => matches!(
            initial_marking_forced_bool(inner, aliases, net),
            Some(false)
        )
        .then_some(false),
        CtlFormula::EU(left, right) | CtlFormula::AU(left, right) => {
            match (
                initial_marking_forced_bool(left, aliases, net),
                initial_marking_forced_bool(right, aliases, net),
            ) {
                (_, Some(true)) => Some(true),
                (Some(false), Some(false)) => Some(false),
                _ => None,
            }
        }
        // EX/AX (next-step) and EGF (fair-cycle) cannot be forced from the
        // initial marking alone without exploration — decline to model check.
        CtlFormula::EX(_) | CtlFormula::AX(_) | CtlFormula::EGF(_) => None,
    }
}

/// A query slice is "stutter/max-path safe" iff it drops NO transition of the
/// net it was carved from.
///
/// Slicing keeps only the closure/cone of the query support and then explores
/// the carved subnet *as if it were the complete system*. For stutter-sensitive
/// (EX/AX) and maximal-path / gfp shapes (EG/AF/AG/AU/EU over a non-atomic
/// inner) this is UNSOUND when a structurally-disjoint concurrent sub-component
/// (a livelock / self-loop disconnected from the support) is dropped: a state
/// that deadlocks in the slice may still fire the dropped component in the full
/// net, which flips EG/AF maximal-path verdicts and next-step structure.
///
/// When every transition survives (the slice only trimmed forward-only sink
/// *places*, as `build_query_local_slice` can), deadlock and next-step structure
/// are preserved exactly, so the slice is answer-preserving for all shapes.
///
/// Returns `true` when the slice is safe to explore for stutter/max-path shapes.
fn slice_keeps_all_transitions(slice: &crate::query_slice::QuerySlice, net: &PetriNet) -> bool {
    slice.net.num_transitions() == net.num_transitions()
}

/// Whether every CTL property in the batch is a slice-invariant *reachability*
/// shape: `EF(atom)`, `AG(atom)`, `E[true U atom]`, or their negated duals
/// (anything `classify_shallow_ctl` recognises).
///
/// For these shapes the verdict at the initial state depends only on which
/// predicate states are reachable. Query slicing keeps the entire closure of
/// every predicate atom, so dropping a structurally-disjoint component (which
/// can neither make a predicate state reachable nor unreachable) cannot change
/// the answer — the slice optimisation stays sound even when it drops
/// transitions. Stutter-sensitive (EX/AX) and maximal-path (EG/AF/AG/AU/EU over
/// a non-atomic inner) shapes are deliberately excluded.
fn batch_is_slice_invariant_reachability(properties: &[Property]) -> bool {
    properties.iter().all(|prop| {
        let Formula::Ctl(ctl) = &prop.formula else {
            return false;
        };
        classify_shallow_ctl(ctl).is_some()
    })
}

fn reduce_ctl_batch_for_mode(
    net: &PetriNet,
    properties: &[Property],
    aliases: &PropertyAliases,
    has_next_step: bool,
) -> ReducedNet {
    // Fair-cycle (EGF) batches stay on the identity net (defense-in-depth
    // alongside the slice guard). `EGF` is a maximal-path property whose truth
    // depends on the infinite-suffix / deadlock-stutter structure; a
    // dead/isolated/agglomeration rule can move or drop the terminal-marking
    // structure the fair-cycle fixpoint observes and flip the verdict. Treated
    // like the stutter-sensitive EX/AX case — verdict-preserving.
    if crate::examinations::ctl::routing::ctl_batch_contains_fair_cycle(properties) {
        return ReducedNet::identity(net);
    }
    let support = ctl_support_with_aliases(&ReducedNet::identity(net), properties, aliases);
    let protected = match support {
        // SOUNDNESS: the protected set must cover the full observation basis of
        // every atom. `support.places` alone misses `is-fireable` atoms, whose
        // truth is a function of the referenced transitions' INPUT places (and
        // whose evolution flows through the output places). Protecting the
        // pre/post places of every supported transition — exactly the
        // reachability pipeline's `protected_places_for_prefire` contract —
        // keeps those markings (and hence every fireability atom) exact in the
        // reduced net. Previously this cloned `s.places` only, so a
        // fireability-only batch (e.g. LTLFireability routed through the
        // A¬G(φ) → AF(¬φ) CTL lane) yielded an EMPTY protected set: the
        // reduction gutted the entire net to a single stuttering state whose
        // expanded marking satisfied φ, flipping AF(¬φ) to FALSE
        // (AirplaneLD-COL-0010-LTLFireability-08 answered FALSE against a
        // TRUE consensus).
        Some(ref s) => protected_places_for_prefire(net, s),
        None => {
            // No usable predicate support: for an EX/AX batch we cannot bound
            // which places the next-step structure observes, so stay on the
            // identity net (verdict-preserving). The next-free path keeps its
            // historical empty-protected behaviour.
            if has_next_step {
                return ReducedNet::identity(net);
            }
            vec![]
        }
    };

    // Mode selection follows the soundness hierarchy in `ReductionMode`:
    //   * EX/AX present  -> `CTLWithNext`: only the marking-preserving
    //     dead/constant/isolated rules, which leave the exact successor
    //     structure (and therefore every next-step verdict) intact. This was
    //     previously `identity` — a single EX/AX anywhere in a 16-property
    //     bundle disabled reduction for the whole batch.
    //   * next-free      -> `NextFreeCTL`: the declared safe mode for the CTL
    //     examinations (most rules except interleaving-changing agglomeration),
    //     strictly stronger than the `CTLWithNext` used before.
    // Both modes are answer-preserving by the reduction module's contract; the
    // explored markings are expanded back through `reduced.expand_marking`.
    let (mode, label) = if has_next_step {
        (ReductionMode::CTLWithNext, "CTL with-next reduction")
    } else {
        (ReductionMode::NextFreeCTL, "CTL next-free reduction")
    };

    match reduce_iterative_structural_with_mode(net, &protected, mode, None) {
        Ok(r) => {
            let removed = r.report.places_removed() + r.report.transitions_removed();
            if removed > 0 {
                eprintln!(
                    "{label}: removed {removed} elements \
                     ({p} places, {t} transitions)",
                    p = r.report.places_removed(),
                    t = r.report.transitions_removed(),
                );
            }
            r
        }
        Err(error) => {
            eprintln!("CTL reduction error: {error} — falling back to identity");
            ReducedNet::identity(net)
        }
    }
}

#[cfg(test)]
pub(super) fn reduce_ctl_batch_for_test(net: &PetriNet, properties: &[Property]) -> ReducedNet {
    let aliases = PropertyAliases::identity(net);
    reduce_ctl_batch_for_mode(
        net,
        properties,
        &aliases,
        ctl_batch_contains_next_step(properties),
    )
}

/// Count unresolved names in a CTL formula's atoms.
pub(super) fn count_unresolved_ctl_with_aliases(
    formula: &CtlFormula,
    aliases: &PropertyAliases,
) -> (usize, usize) {
    match formula {
        CtlFormula::Atom(pred) => count_unresolved_with_aliases(pred, aliases),
        CtlFormula::Not(inner)
        | CtlFormula::EX(inner)
        | CtlFormula::AX(inner)
        | CtlFormula::EF(inner)
        | CtlFormula::AF(inner)
        | CtlFormula::EG(inner)
        | CtlFormula::AG(inner)
        | CtlFormula::EGF(inner) => count_unresolved_ctl_with_aliases(inner, aliases),
        CtlFormula::And(children) | CtlFormula::Or(children) => {
            children.iter().fold((0, 0), |(t, u), c| {
                let (ct, cu) = count_unresolved_ctl_with_aliases(c, aliases);
                (t + ct, u + cu)
            })
        }
        CtlFormula::EU(phi, psi) | CtlFormula::AU(phi, psi) => {
            let (pt, pu) = count_unresolved_ctl_with_aliases(phi, aliases);
            let (qt, qu) = count_unresolved_ctl_with_aliases(psi, aliases);
            (pt + qt, pu + qu)
        }
    }
}

/// Fixpoint polarity of a CTL temporal operator.
///
/// In the Emerson-Clarke μ-calculus encoding the path operators split into
/// two fixpoint classes: least (μ) and greatest (ν). `EX`/`AX` are pure
/// next-step modalities with no fixpoint, so they carry no polarity.
#[derive(Clone, Copy, PartialEq, Eq)]
enum FixpointPolarity {
    /// Least fixpoint (μ): `EF`, `AF`, `EU`, `AU`.
    Least,
    /// Greatest fixpoint (ν): `EG`, `AG`.
    Greatest,
}

/// Whether `formula` is free of μ/ν fixpoint *alternation*: no least-fixpoint
/// temporal operator nested (at any depth) inside a greatest-fixpoint operator,
/// and vice versa.
///
/// SOUNDNESS GATE for the recursive [`LocalCtlChecker`] fallback. That checker
/// is a single-pass DFS with per-node cycle assumptions (assume-false on a μ
/// back-edge, assume-true on a ν back-edge) and write-once `Ready`
/// memoization. That is a *correct* local fixpoint computation only for an
/// alternation-free formula: with a single fixpoint class the cycle assumption
/// is the genuine fixpoint default and the result is exact once the relevant
/// reachable subgraph is explored (an incomplete exploration aborts with `Err`
/// instead, never a verdict). When μ and ν alternate (e.g. `AG(EF p)` =
/// ν(μ), `EF(AG p)` = μ(ν)), the inner fixpoint's `Ready` value is committed
/// under the outer fixpoint's transient cycle assumption with no outer
/// iteration to revise it, so the checker can terminate early and return a
/// confident-but-WRONG verdict — it answered `AG(EF(fireable tredo4)) = FALSE`
/// on Kanban (true answer TRUE) after interning a few hundred of the reachable
/// states (Kanban-PT-00200-CTLFireability-2025-05, -CTLCardinality-2025-09).
///
/// `Not` is treated conservatively: because the recursive checker negates by
/// direct value flip rather than pushing negation to a positive normal form,
/// any opposite-class temporal operator nested under another is treated as
/// alternation regardless of intervening `Not`/`EX`/`AX`/boolean nodes.
fn ctl_is_alternation_free(formula: &resolve::ResolvedCtl) -> bool {
    fn polarity_of(formula: &resolve::ResolvedCtl) -> Option<FixpointPolarity> {
        match formula {
            resolve::ResolvedCtl::EF(_)
            | resolve::ResolvedCtl::AF(_)
            | resolve::ResolvedCtl::EU(_, _)
            | resolve::ResolvedCtl::AU(_, _) => Some(FixpointPolarity::Least),
            resolve::ResolvedCtl::EG(_) | resolve::ResolvedCtl::AG(_) => {
                Some(FixpointPolarity::Greatest)
            }
            _ => None,
        }
    }

    // `enclosing` is the polarity of the nearest temporal ancestor (if any).
    // Encountering a temporal operator of the opposite class while inside one
    // is alternation.
    fn walk(formula: &resolve::ResolvedCtl, enclosing: Option<FixpointPolarity>) -> bool {
        let here = polarity_of(formula);
        if let (Some(outer), Some(inner)) = (enclosing, here) {
            if outer != inner {
                return false;
            }
        }
        // The polarity threaded into the children is this node's polarity when
        // it is a temporal operator; otherwise the inherited one.
        let child_pol = here.or(enclosing);
        match formula {
            resolve::ResolvedCtl::Atom(_) => true,
            resolve::ResolvedCtl::Not(inner)
            | resolve::ResolvedCtl::EX(inner)
            | resolve::ResolvedCtl::AX(inner)
            | resolve::ResolvedCtl::EF(inner)
            | resolve::ResolvedCtl::AF(inner)
            | resolve::ResolvedCtl::EG(inner)
            | resolve::ResolvedCtl::AG(inner) => walk(inner, child_pol),
            resolve::ResolvedCtl::And(children) | resolve::ResolvedCtl::Or(children) => {
                children.iter().all(|c| walk(c, child_pol))
            }
            resolve::ResolvedCtl::EU(phi, psi) | resolve::ResolvedCtl::AU(phi, psi) => {
                walk(phi, child_pol) && walk(psi, child_pol)
            }
            // EGF (E(GF a)) is the Emerson–Lei νZ.μY. fair-cycle — it alternates
            // a greatest over a least fixpoint WITHIN a single node, so it is
            // never alternation-free. Reporting it as alternating keeps the
            // single-pass LocalCtlChecker (unsound under alternation) from ever
            // being handed an EGF formula; the sound GPU/CPU CtlEngine fair-cycle
            // evaluators handle it instead.
            resolve::ResolvedCtl::EGF(_) => false,
        }
    }

    walk(formula, None)
}

#[cfg(test)]
pub(super) fn ctl_is_alternation_free_for_test(formula: &resolve::ResolvedCtl) -> bool {
    ctl_is_alternation_free(formula)
}

/// STRICT single-fixpoint-layer gate for the recursive [`LocalCtlChecker`].
///
/// `ctl_is_alternation_free` alone is INSUFFICIENT: it admits nested SAME-class
/// fixpoints (ν-in-ν like `EG(… ∨ AG p)`, μ-in-μ like `EF(EF p)`) and fixpoints
/// nested under `EX`/`AX`. The `LocalCtlChecker` is a single-pass DFS whose
/// per-node cycle assumption and write-once `Ready` memoization are a valid
/// fixpoint computation for exactly ONE fixpoint layer. With a second (even
/// same-class) fixpoint nested inside, an inner node can be committed `Ready`
/// under a still-`Active` outer assumption that later resolves the OTHER way,
/// poisoning the cache and yielding a confident-but-WRONG verdict (confirmed:
/// the 4-state `pr,p0,p1,p3` net on `EG(AG(p3≤0) ∨ (p1+p3≤0))` — the local
/// checker disagrees with the full-graph oracle).
///
/// This predicate admits a formula ONLY when at most one fixpoint operator can
/// be entered before reaching an atom, and never under another temporal
/// operator. It walks with `inside_temporal`: boolean connectives are
/// transparent (recurse with the SAME flag); every temporal/path operator marks
/// its children `inside_temporal = true`; a FIXPOINT operator
/// (`EF`/`AF`/`EG`/`AG`/`EU`/`AU`) reached while `inside_temporal` fails the
/// gate. Any single-fixpoint-layer formula is a strict subset of the
/// alternation-free formulas, so this only ever NARROWS the local lane.
fn ctl_has_single_fixpoint_layer(formula: &resolve::ResolvedCtl) -> bool {
    use resolve::ResolvedCtl as C;

    fn walk(formula: &C, inside_temporal: bool) -> bool {
        match formula {
            // Atoms carry no temporal structure.
            C::Atom(_) => true,
            // Boolean connectives are transparent: recurse with the SAME flag —
            // they do not open a fixpoint layer.
            C::Not(inner) => walk(inner, inside_temporal),
            C::And(children) | C::Or(children) => children.iter().all(|c| walk(c, inside_temporal)),
            // Pure next-step modalities: no fixpoint themselves, but a fixpoint
            // nested UNDER them is a second layer for the single-pass assumption
            // cache, so they mark their child `inside_temporal`.
            C::EX(inner) | C::AX(inner) => walk(inner, true),
            // Fixpoint operators. Reaching one while already inside a temporal
            // operator is a SECOND fixpoint layer — reject. Otherwise this is the
            // sole (first) layer; descend with `inside_temporal = true` so any
            // further fixpoint below is rejected.
            C::EF(inner) | C::AF(inner) | C::EG(inner) | C::AG(inner) => {
                !inside_temporal && walk(inner, true)
            }
            C::EU(phi, psi) | C::AU(phi, psi) => {
                !inside_temporal && walk(phi, true) && walk(psi, true)
            }
            // EGF (E G F) is the Emerson–Lei νμ fair cycle: an alternating
            // fixpoint WITHIN a single node, never a single layer. The
            // `LocalCtlChecker` fails closed on it anyway; reject unconditionally.
            C::EGF(_) => false,
        }
    }

    walk(formula, false)
}

#[cfg(test)]
pub(super) fn ctl_has_single_fixpoint_layer_for_test(formula: &resolve::ResolvedCtl) -> bool {
    ctl_has_single_fixpoint_layer(formula)
}

/// SOUND symbolic-BDD CTL verdict (fail-closed), gated behind `dd-backend`.
///
/// Builds the exact symbolic reachable set of the **original** `net` via the
/// shared, sound DD spec builder
/// ([`crate::examinations::dd_spec::build_sound_dd_spec`]), translates the
/// resolved CTL formula into `tla_dd::symbolic_ctl::CtlFormula` (declining if
/// any atom is unsupported), and asks the oracle-verified
/// [`tla_dd::symbolic_ctl::symbolic_ctl_holds`] evaluator whether the formula
/// holds at the initial marking.
///
/// Returns:
/// - `Some(true)` / `Some(false)` — a SOUND verdict, used in place of the
///   budget-limited local solvers.
/// - `None` — declined on ANY gate (net not DD-eligible: per-place bound > 16
///   or place count over cap; any atom unsupported; the reachability fixpoint
///   went over budget / OOM). The caller falls through to the EXISTING
///   behavior unchanged. Never a guessed verdict.
#[cfg(feature = "dd-backend")]
#[must_use]
fn try_symbolic_ctl_verdict(
    net: &PetriNet,
    resolved: &resolve::ResolvedCtl,
    global_deadline: Option<Instant>,
) -> Option<bool> {
    // Gate 1: sound DD spec (place indices preserved; `None` ⇒ not eligible).
    let (spec, _bounds) = crate::examinations::dd_spec::build_sound_dd_spec(net)?;
    // Gate 2: lower the formula to the shared `CtlFormulaTemplate<DdPredicate>` —
    // the SAME converter the MDD CTL lane uses (every atom must convert, else
    // decline). oxidd's `symbolic_ctl::CtlFormula` is gone.
    let template = resolved_ctl_to_mdd_template(resolved, net.num_places(), net.num_transitions())?;
    // Gate 3: evaluate on the NATIVE tla-bdd CTL engine (`evaluate_ctl_via_bdd`) —
    // ≡ the MDD CTL lane (`bdd_ctl_matches_mdd_lane`, full EX/AX/EF/AF/EG/AG/EU/AU)
    // and zero CTL disagreements in the 46-model corpus A/B. Budget-bounded via
    // `reachable_within` ⇒ a clean DECLINE (`None`) on timeout / ineligibility;
    // the caller falls through to the existing local solvers unchanged. The oxidd
    // BDD engine has been removed from this lane.
    crate::examinations::mdd_common::evaluate_ctl_via_bdd(&spec, &template, global_deadline)
}

/// `dd-backend` disabled: the symbolic CTL lane is a no-op that always
/// declines, leaving the existing fallback behavior unchanged.
#[cfg(not(feature = "dd-backend"))]
#[must_use]
fn try_symbolic_ctl_verdict(
    _net: &PetriNet,
    _resolved: &resolve::ResolvedCtl,
    _global_deadline: Option<Instant>,
) -> Option<bool> {
    None
}

// (`resolved_ctl_to_symbolic` — the resolved→oxidd-`symbolic_ctl::CtlFormula`
// converter — was removed when the CTL lane migrated to the native tla-bdd engine;
// `try_symbolic_ctl_verdict` now lowers via `resolved_ctl_to_mdd_template` →
// `CtlFormulaTemplate<DdPredicate>` and evaluates with `evaluate_ctl_via_bdd`.)

/// Gate for the MDD CTL lane. The lane is OPT-IN; DEFAULT OFF pending a broader
/// additive-gain measurement (it showed 0 cell gains on the measured subset and
/// a budget-stealing regression on Philosophers-PT-000050: it spends the budget
/// building/attempting a ~7e23-state reachable set per formula, DECLINES, and
/// steals time the EDG / LocalCtlChecker would otherwise use to decide the
/// formula). Enable it explicitly with `TY_MCC_ENABLE_MDD_CTL` set to a truthy
/// value (`1`/`on`/`true`/`yes`). `TY_MCC_DISABLE_MDD_CTL` is still honored and
/// WINS: if disable is truthy the lane stays off even when enable is set.
///
/// The lane's code, tests, and the `crosscheck_ctl` battery remain intact — this
/// is ONLY a default-gating change. When the lane runs it is SOUND (ship gate
/// passed); when it is off, formulas simply fall through to the other (also
/// sound) lanes. So the gate is SOUNDNESS-NEUTRAL: it never changes a published
/// verdict, it only changes which lane decides.
///
/// Returns `true` (lane disabled / declines) unless explicitly enabled.
#[cfg(feature = "dd-backend")]
/// Is the gated native-ROBDD CTL lane enabled? OFF by default; set
/// `TY_MCC_BDD_CTL=1` to route the symbolic CTL verdict through `tla-bdd`
/// (`evaluate_ctl_via_bdd`) instead of the MDD lane (for the per-examination
/// coverage A/B). Empty / `0` ⇒ disabled.
fn bdd_ctl_lane_enabled() -> bool {
    std::env::var("TY_MCC_BDD_CTL")
        .map(|v| !v.is_empty() && v != "0")
        .unwrap_or(false)
}

#[cfg(feature = "dd-backend")]
fn mdd_ctl_disabled() -> bool {
    // Test override: force the lane ON for the differential / soundness battery.
    #[cfg(test)]
    if let Some(enabled) = mdd_ctl_enabled_test_override() {
        return !enabled;
    }

    let disable = std::env::var("TY_MCC_DISABLE_MDD_CTL").ok();
    let enable = std::env::var("TY_MCC_ENABLE_MDD_CTL").ok();
    mdd_ctl_disabled_from(disable, enable)
}

/// Pure (env-free) gate decision so the precedence can be unit-tested without
/// mutating process-global environment (which races across parallel tests).
///
/// Precedence: `TY_MCC_DISABLE_MDD_CTL` truthy WINS (lane disabled). Otherwise
/// the lane is enabled ONLY if `TY_MCC_ENABLE_MDD_CTL` is truthy. Default (both
/// unset / non-truthy) is DISABLED.
///
/// NOTE: the CTL lane is opt-in for a PERFORMANCE reason, not soundness — it is
/// sound when it runs, but it showed 0 cell gains and a budget-stealing
/// regression (it spends the per-formula budget attempting a huge reachable set,
/// declines, and starves the EDG/LocalCtlChecker that would decide the formula).
/// So — unlike the StateSpace/Reachability MDD lanes — it stays default-OFF.
#[cfg(feature = "dd-backend")]
fn mdd_ctl_disabled_from(disable: Option<String>, enable: Option<String>) -> bool {
    if is_truthy_flag(disable.as_deref()) {
        return true; // explicit disable wins
    }
    !is_truthy_flag(enable.as_deref()) // opt-in: off unless explicitly enabled
}

/// Truthiness check for an MDD-CTL gate flag (`1`/`on`/`true`/`yes`,
/// case-insensitive, trimmed). `None`/anything else is not truthy.
#[cfg(feature = "dd-backend")]
fn is_truthy_flag(v: Option<&str>) -> bool {
    v.is_some_and(|v| {
        let v = v.trim();
        v == "1"
            || v.eq_ignore_ascii_case("on")
            || v.eq_ignore_ascii_case("true")
            || v.eq_ignore_ascii_case("yes")
    })
}

/// Lower a [`resolve::ResolvedCtl`] to the MDD CTL evaluator's
/// [`tla_mdd::CtlFormulaTemplate`] (atom = [`tla_dd::DdPredicate`]), reusing the
/// SAME `translate_predicate` converter the BDD lane uses. Returns `None`
/// (fail-closed) if any atom does not convert.
#[cfg(feature = "dd-backend")]
fn resolved_ctl_to_mdd_template(
    formula: &resolve::ResolvedCtl,
    num_places: usize,
    num_transitions: usize,
) -> Option<tla_mdd::CtlFormulaTemplate<tla_dd::DdPredicate>> {
    use resolve::ResolvedCtl as R;
    use tla_mdd::CtlFormulaTemplate as T;
    let rec =
        |f: &resolve::ResolvedCtl| resolved_ctl_to_mdd_template(f, num_places, num_transitions);
    Some(match formula {
        R::Atom(pred) => T::Atom(crate::examinations::dd_spec::translate_predicate(
            pred,
            num_places,
            num_transitions,
        )?),
        R::Not(inner) => T::Not(Box::new(rec(inner)?)),
        R::And(children) => {
            let mut out = Vec::with_capacity(children.len());
            for c in children {
                out.push(rec(c)?);
            }
            T::And(out)
        }
        R::Or(children) => {
            let mut out = Vec::with_capacity(children.len());
            for c in children {
                out.push(rec(c)?);
            }
            T::Or(out)
        }
        R::EX(inner) => T::EX(Box::new(rec(inner)?)),
        R::AX(inner) => T::AX(Box::new(rec(inner)?)),
        R::EF(inner) => T::EF(Box::new(rec(inner)?)),
        R::AF(inner) => T::AF(Box::new(rec(inner)?)),
        R::EG(inner) => T::EG(Box::new(rec(inner)?)),
        R::AG(inner) => T::AG(Box::new(rec(inner)?)),
        R::EU(phi, psi) => T::EU(Box::new(rec(phi)?), Box::new(rec(psi)?)),
        R::AU(phi, psi) => T::AU(Box::new(rec(phi)?), Box::new(rec(psi)?)),
        // EGF routes to the MDD evaluator's Emerson–Lei fair-cycle gfp with the
        // DEADLOCK-STUTTER successor — verdict-identical to CtlEngine::gfp_egf
        // (zero-disagreement differential in tla-mdd's crosscheck_ctl). Enabled
        // after the CTL routing rework (synced from main) fixed the inherited
        // persistence wrong-TRUE this routing was previously barred behind
        // (test_persistence_deadlock_stutter_false_on_trap_net — now green in
        // BOTH flag configurations). The LocalCtlChecker alternation gate still
        // keeps the single-pass checker away from EGF.
        R::EGF(inner) => T::EGF(Box::new(rec(inner)?)),
    })
}

/// SOUND symbolic-MDD CTL verdict (fail-closed), gated behind `dd-backend`.
///
/// The MDD counterpart of [`try_symbolic_ctl_verdict`], targeting the
/// counter / conserved / high-bound nets the bit-blasted BDD lane blows up on
/// (the same families where the MDD made StateSpace effective). It builds the
/// exact reachable set of the **original** `net` via the shared sound DD spec
/// builder, lowers the resolved CTL formula's atoms to characteristic MDD sets
/// (EXACT — see [`crate::examinations::mdd_common::lower_dd_predicate_to_mdd`]),
/// and evaluates the
/// oracle-verified [`tla_mdd::evaluate_at_initial`] (the `CtlEngine`-pinned
/// fixpoint convention) at the initial marking — all on a worker thread with the
/// big DD stack + the caller's deadline.
///
/// Returns `Some(true)`/`Some(false)` for a SOUND verdict, or `None` on ANY
/// gate (net not DD-eligible, atom unsupported, reachability/fixpoint over
/// budget, deadline, overflow, spawn failure, panic). Never a guessed verdict.
#[cfg(feature = "dd-backend")]
#[must_use]
fn try_mdd_ctl_verdict(
    net: &PetriNet,
    resolved: &resolve::ResolvedCtl,
    deadline: Option<Instant>,
) -> Option<bool> {
    if mdd_ctl_disabled() {
        return None;
    }
    // Gate 1: the MDD's own admission gate (sound LP bounds + structural gates +
    // edge-width cap), decoupled from the BDD lane's 127-var cap. Shared with the
    // reachability MDD fast-path via `mdd_common` (single source of truth).
    let spec = crate::examinations::mdd_common::build_mdd_spec_for_net(net)?;
    // Gate 2: lower the formula (every atom must convert, else decline).
    let template = resolved_ctl_to_mdd_template(resolved, net.num_places(), net.num_transitions())?;

    // Gated tla-bdd CTL lane (TY_MCC_BDD_CTL): the native ROBDD twin — component
    // verdict-validated (≡ MDD CTL lane, bdd_ctl_matches_mdd_lane) + budget-bounded
    // (evaluate_ctl_within). OFF by default ⇒ the MDD lane below runs UNCHANGED
    // (this branch moves spec+template and returns only when enabled). Decline /
    // timeout / panic falls through soundly, exactly like the MDD lane.
    if bdd_ctl_lane_enabled() {
        use std::sync::mpsc;
        let budget = deadline.map(|d| d.saturating_duration_since(Instant::now()));
        if let Some(b) = budget {
            if b.is_zero() {
                return None;
            }
        }
        let (tx, rx) = mpsc::channel();
        let handle = std::thread::Builder::new()
            .name("tla-bdd-ctl".into())
            .stack_size(tla_dd::DD_WORKER_STACK_BYTES)
            .spawn(move || {
                let r = crate::examinations::mdd_common::evaluate_ctl_via_bdd(
                    &spec, &template, deadline,
                );
                let _ = tx.send(r);
            });
        if handle.is_err() {
            eprintln!("CTL: tla-bdd lane spawn failed — falling through");
            return None;
        }
        let recv = match budget {
            Some(b) => rx.recv_timeout(b + Duration::from_millis(1500)),
            None => rx.recv().map_err(|_| mpsc::RecvTimeoutError::Disconnected),
        };
        return match recv {
            Ok(Some(v)) => Some(v),
            _ => None, // decline / timeout / panic → fall through (sound)
        };
    }

    let mdd_net = crate::examinations::mdd_common::dd_spec_to_mdd_net(&spec);

    // Run on a detached worker thread with the big DD stack + the caller's
    // deadline (mirrors `run_mdd_metrics_timed`). The MDD engine takes the
    // deadline directly and declines (fail-closed) rather than overrun it; the
    // worker boundary turns any panic into a clean DECLINE.
    use std::sync::mpsc;
    let budget = deadline.map(|d| d.saturating_duration_since(Instant::now()));
    // A non-positive remaining budget ⇒ skip (decline) rather than spawn.
    if let Some(b) = budget {
        if b.is_zero() {
            return None;
        }
    }
    let (tx, rx) = mpsc::channel();
    let handle = std::thread::Builder::new()
        .name("tla-mdd-ctl".into())
        .stack_size(tla_dd::DD_WORKER_STACK_BYTES)
        .spawn(move || {
            let inner_deadline = deadline;
            let r = tla_mdd::evaluate_at_initial(
                &mdd_net,
                &template,
                inner_deadline,
                crate::examinations::mdd_common::lower_dd_predicate_to_mdd,
            );
            let _ = tx.send(r);
        });
    let Ok(_thread) = handle else {
        eprintln!("CTL: MDD lane thread spawn failed — falling through");
        return None;
    };
    // Wait at most until the deadline (+ a small grace), else treat as a
    // timeout decline. With no deadline, block on the worker (it will finish or
    // decline on its own node budget).
    let recv = match budget {
        Some(b) => rx.recv_timeout(b + Duration::from_millis(1500)),
        None => rx.recv().map_err(|_| mpsc::RecvTimeoutError::Disconnected),
    };
    match recv {
        Ok(Ok(v)) => Some(v),
        Ok(Err(err)) => {
            eprintln!("CTL: MDD lane declined ({err:?}) — falling through");
            None
        }
        Err(mpsc::RecvTimeoutError::Timeout) => {
            eprintln!("CTL: MDD lane exceeded budget — falling through");
            None
        }
        Err(mpsc::RecvTimeoutError::Disconnected) => {
            eprintln!("CTL: MDD lane worker panicked — falling through");
            None
        }
    }
}

/// `dd-backend` disabled: the MDD CTL lane is a no-op that always declines.
#[cfg(not(feature = "dd-backend"))]
#[must_use]
fn try_mdd_ctl_verdict(
    _net: &PetriNet,
    _resolved: &resolve::ResolvedCtl,
    _deadline: Option<Instant>,
) -> Option<bool> {
    None
}

pub(super) fn check_ctl_properties_inner(
    net: &PetriNet,
    properties: &[Property],
    aliases: &PropertyAliases,
    config: &ExplorationConfig,
    guard_mode: SoundnessGuardMode,
    flush: bool,
    techniques: &Techniques,
) -> Vec<(String, Verdict)> {
    let shortcuts_enabled = experimental_ctl_shortcuts_enabled();
    let unresolved_original_ids: HashSet<String> = properties
        .iter()
        .filter_map(|prop| {
            let Formula::Ctl(ctl) = &prop.formula else {
                return None;
            };
            let (total, unresolved) = count_unresolved_ctl_with_aliases(ctl, aliases);
            if unresolved == 0 {
                return None;
            }
            eprintln!(
                "CTL resolution guard: {} has {unresolved}/{total} \
                 unresolved names before simplification -> CANNOT_COMPUTE",
                prop.id
            );
            Some(prop.id.clone())
        })
        .collect();

    // CTL shortcuts are default-on for MCC qualification throughput, but still
    // narrow: only syntactic EF/AG reachability equivalences, AF/EG suffix
    // cases on a completed graph, and simplifier constants may short-circuit.
    // Set TY_MCC_DISABLE_CTL_SHORTCUTS=1 (or
    // TY_MCC_ENABLE_CTL_SHORTCUTS=0) to force the full-graph baseline.
    let simplified = if shortcuts_enabled {
        Some(crate::formula_simplify::simplify_properties_with_deadline(
            net,
            properties,
            aliases,
            ctl_simplify_deadline(config.deadline()),
        ))
    } else {
        None
    };
    let properties = simplified.as_deref().unwrap_or(properties);
    let mut flushed_ids = HashSet::new();

    let mut shallow_map: HashMap<String, Verdict> = HashMap::new();
    if shortcuts_enabled {
        for prop in properties {
            if let Formula::Ctl(ctl) = &prop.formula {
                if let Some(verdict) = initial_marking_forced_verdict(ctl, aliases, net) {
                    shallow_map.insert(prop.id.clone(), verdict);
                }
            }
        }
    }

    {
        let shallow_reachability_props: Vec<Property> = if shortcuts_enabled {
            properties
                .iter()
                .filter(|prop| !shallow_map.contains_key(&prop.id))
                .filter_map(|prop| {
                    let Formula::Ctl(ctl) = &prop.formula else {
                        return None;
                    };
                    let class = classify_shallow_ctl(ctl)?;
                    let (quantifier, predicate) = match class {
                        ShallowCtl::ExistsFinally(pred) => (PathQuantifier::EF, pred),
                        ShallowCtl::AlwaysGlobally(pred) => (PathQuantifier::AG, pred),
                    };
                    Some(Property {
                        id: prop.id.clone(),
                        formula: Formula::Reachability(ReachabilityFormula {
                            quantifier,
                            predicate,
                        }),
                    })
                })
                .collect()
        } else {
            Vec::new()
        };

        // EARLY raw-net random-walk witness pass for the shallow EF(atom)/AG(atom)
        // reachability fragment.
        //
        // The shallow props below ARE routed to the reachability pipeline (which
        // has its own early raw-net walk), but on LARGE nets the CTL pipeline's
        // own heavyweight front-matter (full-graph CTL engine, slicing,
        // simplification) plus the reachability call's fair-share clamp can mean
        // the walk never gets a chance to seed the cheap witnesses before the
        // budget is gone — so EF(atom)/AG(atom) CTL formulas a walk could resolve
        // stay CANNOT_COMPUTE (e.g. BlocksWorld-PT-08 CTLFireability EF(is-fireable),
        // consensus TRUE). This pass runs the SAME walk engine on the raw `net`
        // up front, mirroring how the reachability pipeline builds trackers +
        // validation for these props, so cheap witnesses are caught first. Hits
        // resolve into `shallow_map` (skipped downstream); misses are left
        // untouched and flow into the unchanged `check_reachability_properties_*`
        // call and then the full CTL engine — nothing is lost.
        //
        // ADDITIVE / no-starvation: this lane is clamped to
        // `under_approx_lane_deadline(net, ...)` — a SLICE of the remaining budget
        // (`min(remaining/4, 8s)`) that reserves the exhaustive-CTL/BFS tail. It
        // does NOT take budget from the full CTL engine or the downstream
        // reachability call (those keep their existing deadlines unchanged).
        //
        // SOUND: `run_random_walk_seeding` only emits replay-validated witnesses
        // from reachable markings — EF(atom)=TRUE (a reachable marking satisfying
        // the atom) and AG(atom)=FALSE (it "Also seeds AG(φ)=FALSE by finding
        // counterexamples", reachability_walk.rs:39, i.e. a reachable ¬atom
        // marking). It can never produce the universal verdicts (EF=FALSE /
        // AG=TRUE), which remain the exclusive job of the full CTL engine / BFS.
        if shortcuts_enabled && !shallow_reachability_props.is_empty() {
            let (_prepared, mut trackers, validation_targets) =
                crate::examinations::reachability::prepare_trackers_with_aliases(
                    &shallow_reachability_props,
                    &shallow_reachability_props,
                    aliases,
                );
            if trackers.iter().any(|t| t.verdict.is_none()) {
                let validation =
                    crate::examinations::reachability_witness::WitnessValidationContext::new(
                        net,
                        &validation_targets,
                    );
                crate::examinations::reachability_walk::run_random_walk_seeding(
                    net,
                    &mut trackers,
                    &validation,
                    crate::examinations::reachability::under_approx_lane_deadline(
                        net,
                        config.deadline(),
                    ),
                );
                let mut early_walk_seeded = 0usize;
                for tracker in &trackers {
                    let Some(holds) = tracker.verdict else {
                        continue;
                    };
                    // The walk's witness contract: EF=TRUE / AG=FALSE only.
                    // Map the tracker's bool verdict to the CTL verdict and skip
                    // the (impossible-here) universal side defensively.
                    let verdict = match (tracker.quantifier, holds) {
                        (PathQuantifier::EF, true) => Verdict::True,
                        (PathQuantifier::AG, false) => Verdict::False,
                        _ => continue,
                    };
                    if shallow_map.insert(tracker.id.clone(), verdict).is_none() {
                        early_walk_seeded += 1;
                    }
                }
                if early_walk_seeded > 0 {
                    eprintln!(
                        "CTL shallow early raw-net random walk seeded {early_walk_seeded}/{} \
                         reachability verdicts",
                        shallow_reachability_props.len(),
                    );
                }
            }
        }

        // Misses fall through: re-filter the shallow props so the heavyweight
        // reachability call only sees the ones the early walk did not resolve.
        let shallow_reachability_props: Vec<Property> = shallow_reachability_props
            .into_iter()
            .filter(|prop| !shallow_map.contains_key(&prop.id))
            .collect();

        if !shallow_reachability_props.is_empty() {
            let shallow_count = shallow_reachability_props.len();
            let total = properties.len();
            eprintln!(
                "CTL shallow routing: {shallow_count}/{total} properties routed to reachability"
            );
            let shallow_config = config
                .clone()
                .with_deadline(fair_share_deadline(config.deadline(), 4));
            check_reachability_properties_with_aliases(
                net,
                &shallow_reachability_props,
                aliases,
                &shallow_config,
            )
            .into_iter()
            .filter(|(_, verdict)| *verdict != Verdict::CannotCompute)
            .filter(|(id, verdict)| {
                let Some(prop) = shallow_reachability_props
                    .iter()
                    .find(|prop| &prop.id == id)
                else {
                    return true;
                };
                let Formula::Reachability(rf) = &prop.formula else {
                    return true;
                };
                let resolved = resolve_predicate_with_aliases(&rf.predicate, aliases);
                let init_holds = eval_predicate(&resolved, &net.initial_marking, net);
                match (rf.quantifier, *verdict) {
                    (PathQuantifier::EF, Verdict::False) if init_holds => {
                        eprintln!(
                            "CTL shallow sanity: {} EF verdict FALSE contradicts \
                             initial state -> falling through to full CTL",
                            id
                        );
                        false
                    }
                    (PathQuantifier::AG, Verdict::True) if !init_holds => {
                        eprintln!(
                            "CTL shallow sanity: {} AG verdict TRUE contradicts \
                             initial state -> falling through to full CTL",
                            id
                        );
                        false
                    }
                    _ => true,
                }
            })
            .for_each(|(id, verdict)| {
                shallow_map.insert(id, verdict);
            });
        }
    }
    shallow_map.extend(
        unresolved_original_ids
            .iter()
            .map(|id| (id.clone(), Verdict::CannotCompute)),
    );

    flush_shallow_results(
        properties,
        &shallow_map,
        &mut flushed_ids,
        flush,
        techniques,
    );

    if properties
        .iter()
        .all(|prop| shallow_map.contains_key(&prop.id))
    {
        return properties
            .iter()
            .filter_map(|prop| {
                if flushed_ids.contains(&prop.id) {
                    return None;
                }
                let verdict = shallow_map
                    .get(&prop.id)
                    .copied()
                    .unwrap_or(Verdict::CannotCompute);
                Some((prop.id.clone(), verdict))
            })
            .collect();
    }

    let suffix_properties =
        collect_suffix_ctl_properties(properties, &shallow_map, shortcuts_enabled);
    let suffix_props: Vec<(String, ShallowCtlSuffix)> = suffix_properties
        .iter()
        .filter_map(|prop| {
            let Formula::Ctl(ctl) = &prop.formula else {
                return None;
            };
            classify_shallow_ctl_suffix(ctl).map(|class| (prop.id.clone(), class))
        })
        .collect();

    if !suffix_props.is_empty() {
        let suffix_count = suffix_props.len();
        let total = properties.len();
        eprintln!(
            "CTL suffix routing: {suffix_count}/{total} properties routed to suffix analysis"
        );

        // SOUNDNESS (suffix EG/AF max-path): the suffix lane evaluates
        // `eg_holds_from_mask` / `af_holds_from_mask`, both of which depend on
        // full-net deadlock structure (a state with no successors witnesses EG).
        // A slice that drops a structurally-disjoint concurrent component turns
        // a non-deadlock into a slice-deadlock and flips the verdict. So the
        // slice optimisation is kept ONLY when it preserves every transition
        // (it merely trimmed forward-only sink places); otherwise we fall
        // through to the full net — exact, never a wrong verdict.
        let identity = ReducedNet::identity(net);
        let slice = ctl_support_with_aliases(&identity, &suffix_properties, aliases)
            .and_then(|support| {
                let closure = closure_on_reduced_net(net, support);
                build_query_slice(net, &closure)
            })
            .filter(|slice| {
                let safe = slice_keeps_all_transitions(slice, net);
                if !safe {
                    eprintln!(
                        "CTL suffix slice drops {} transition(s) (max-path \
                         EG/AF unsound) -> exploring full net",
                        net.num_transitions() - slice.net.num_transitions()
                    );
                }
                safe
            });

        let explore_net = slice.as_ref().map_or(net, |slice| &slice.net);
        let suffix_config = config
            .refitted_for_full_graph(explore_net)
            .with_deadline(fair_share_deadline(config.deadline(), 4));
        let full = explore_full(explore_net, &suffix_config);

        if full.graph.completed {
            let markings: Vec<Vec<u64>> = if let Some(ref slice) = slice {
                (0..full.markings.len())
                    .map(|s| {
                        let marking = full.markings.unpack(s);
                        let mut original = vec![0u64; net.num_places()];
                        for (sliced_idx, &tokens) in marking.iter().enumerate() {
                            original[slice.place_unmap[sliced_idx].0 as usize] = tokens;
                        }
                        original
                    })
                    .collect()
            } else {
                (0..full.markings.len())
                    .map(|s| full.markings.unpack(s))
                    .collect()
            };

            for (prop_id, class) in &suffix_props {
                let predicate = match class {
                    ShallowCtlSuffix::ExistsGlobally(predicate) => predicate,
                    ShallowCtlSuffix::AlwaysFinally(predicate) => predicate,
                };
                let resolved = resolve_predicate_with_aliases(predicate, aliases);

                let sat: Vec<bool> = markings
                    .iter()
                    .map(|marking| eval_predicate(&resolved, marking, net))
                    .collect();

                let verdict = match class {
                    ShallowCtlSuffix::ExistsGlobally(_) => {
                        if eg_holds_from_mask(&full.graph, &sat) {
                            Verdict::True
                        } else {
                            Verdict::False
                        }
                    }
                    ShallowCtlSuffix::AlwaysFinally(_) => {
                        if af_holds_from_mask(&full.graph, &sat) {
                            Verdict::True
                        } else {
                            Verdict::False
                        }
                    }
                };

                let init_holds = eval_predicate(&resolved, &net.initial_marking, net);
                let sane = !matches!(
                    (class, verdict),
                    (ShallowCtlSuffix::ExistsGlobally(_), Verdict::True) if !init_holds
                ) && !matches!(
                    (class, verdict),
                    (ShallowCtlSuffix::AlwaysFinally(_), Verdict::False) if init_holds
                );

                if sane {
                    shallow_map.insert(prop_id.clone(), verdict);
                }
            }
        }
    }

    flush_shallow_results(
        properties,
        &shallow_map,
        &mut flushed_ids,
        flush,
        techniques,
    );

    if properties
        .iter()
        .all(|prop| shallow_map.contains_key(&prop.id))
    {
        return properties
            .iter()
            .filter_map(|prop| {
                if flushed_ids.contains(&prop.id) {
                    return None;
                }
                let verdict = shallow_map
                    .get(&prop.id)
                    .copied()
                    .unwrap_or(Verdict::CannotCompute);
                Some((prop.id.clone(), verdict))
            })
            .collect();
    }

    // Query-aware reduction: default next-free CTL batches get only the
    // universally safe temporal projection subset (dead transitions, constant
    // places, isolated places). EX/AX batches keep the identity graph because
    // next-step operators observe exact successor structure.
    let mut pending_expensive_properties = collect_pending_expensive_ctl_properties(
        properties,
        &shallow_map,
        &flushed_ids,
        guard_mode,
    );
    if pending_expensive_properties.is_empty() {
        return collect_shallow_or_cannot_compute(properties, &shallow_map, &flushed_ids);
    }

    // EDG-FIRST PRE-PASS (certain-zero early exit). The local
    // Liu-Smolka/certain-zero EDG terminates the moment the root
    // configuration is decided EITHER way, so "easy" properties
    // (EF-true, AG-false, small relevant cones) resolve in
    // milliseconds without paying for the batch full-graph build.
    // Properties it cannot decide within a tight node/time cap abort
    // (`Err` — fail-closed, never a verdict) and fall through to the
    // full-graph path and, after that, the remaining-budget EDG
    // fallback exactly as before.
    //
    // SOUNDNESS: budget routing only. The pre-pass invokes the same
    // exact engine the post-full-graph fallback already trusts
    // (`solve_local_edg`), on the same unreduced net and resolved
    // formula; `Ok` verdicts are exact, all aborts fall through.
    let local_fallback_enabled = ctl_local_fallback_enabled();
    if local_fallback_enabled {
        let prepass_start = Instant::now();
        let prop_count = pending_expensive_properties.len().max(1) as u32;
        let per_prop_budget = config
            .deadline()
            .map(|deadline| {
                (deadline
                    .saturating_duration_since(prepass_start)
                    .mul_f64(CTL_EDG_PREPASS_BUDGET_FRACTION)
                    / prop_count)
                    .min(CTL_EDG_PREPASS_MAX_PER_PROP)
            })
            .unwrap_or(CTL_EDG_PREPASS_MAX_PER_PROP);

        let mut decided = 0usize;
        for prop in &pending_expensive_properties {
            let Formula::Ctl(ctl) = &prop.formula else {
                continue;
            };
            let (_, unresolved) = count_unresolved_ctl_with_aliases(ctl, aliases);
            if unresolved > 0 {
                continue;
            }
            let resolved = resolve::resolve_ctl_with_aliases(ctl, aliases);
            let prepass_config = config
                .clone()
                .with_deadline(Some(Instant::now() + per_prop_budget));
            match solve_local_edg_capped(net, &resolved, &prepass_config, CTL_EDG_PREPASS_NODE_CAP)
            {
                Ok(value) => {
                    let verdict = if value { Verdict::True } else { Verdict::False };
                    shallow_map.insert(prop.id.clone(), verdict);
                    decided += 1;
                }
                Err(_) => {}
            }
        }

        if decided > 0 {
            eprintln!(
                "CTL EDG pre-pass: decided {decided}/{} properties in {:.2}s",
                pending_expensive_properties.len(),
                prepass_start.elapsed().as_secs_f64()
            );
            flush_shallow_results(
                properties,
                &shallow_map,
                &mut flushed_ids,
                flush,
                techniques,
            );
            pending_expensive_properties.retain(|prop| !shallow_map.contains_key(&prop.id));
            if pending_expensive_properties.is_empty() {
                return collect_shallow_or_cannot_compute(properties, &shallow_map, &flushed_ids);
            }
        }
    }

    // GPU deep-CTL tier (feature `gpu`): evaluate the still-pending batch
    // over the retained reachable set of the RAW net — no reduction, no
    // slice, so none of the slice-soundness caveats below apply. Semantics
    // match the exhaustive CPU checker (maximal-path EG, standard duals);
    // the engine self-declines past the configured exploration bound, on
    // any inadmissible atom, and at the deadline — every decline leaves the
    // pending batch untouched for the existing pipeline. Properties under
    // the MCC soundness-guard skip list stay with the CPU path.
    #[cfg(feature = "gpu")]
    if !pending_expensive_properties.is_empty() && crate::gpu_state_space::gpu_lane_enabled(net) {
        let mut gpu_batch: Vec<(String, resolve::ResolvedCtl)> = Vec::new();
        for prop in &pending_expensive_properties {
            if guard_mode == SoundnessGuardMode::Enforce
                && is_known_mcc_ctl_soundness_guard(&prop.id)
            {
                continue;
            }
            if let Formula::Ctl(ctl) = &prop.formula {
                let (_, unresolved) = count_unresolved_ctl_with_aliases(ctl, aliases);
                if unresolved == 0 {
                    gpu_batch.push((
                        prop.id.clone(),
                        resolve::resolve_ctl_with_aliases(ctl, aliases),
                    ));
                }
            }
        }
        if !gpu_batch.is_empty() {
            let formulas: Vec<resolve::ResolvedCtl> =
                gpu_batch.iter().map(|(_, f)| f.clone()).collect();
            if let Some(verdicts) = crate::gpu_state_space::ctl_check_gpu(
                net,
                &formulas,
                config.max_states(),
                config.deadline(),
            ) {
                for ((id, _), verdict) in gpu_batch.iter().zip(verdicts) {
                    shallow_map.insert(
                        id.clone(),
                        if verdict {
                            Verdict::True
                        } else {
                            Verdict::False
                        },
                    );
                }
                flush_shallow_results(
                    properties,
                    &shallow_map,
                    &mut flushed_ids,
                    flush,
                    techniques,
                );
                pending_expensive_properties.retain(|prop| !shallow_map.contains_key(&prop.id));
                if pending_expensive_properties.is_empty() {
                    return collect_shallow_or_cannot_compute(
                        properties,
                        &shallow_map,
                        &flushed_ids,
                    );
                }
            }
        }
    }

    let has_next_step = ctl_batch_contains_next_step(&pending_expensive_properties);
    let reduced =
        reduce_ctl_batch_for_mode(net, &pending_expensive_properties, aliases, has_next_step);
    // SOUNDNESS (deep/closure slice): the carved subnet is explored as the
    // complete system. That is answer-preserving for two cases:
    //   1. Slice-invariant *reachability* shapes (EF/AG-atom and duals): the
    //      verdict depends only on which predicate states are reachable, which
    //      the closure of the support fully captures — dropping a disjoint
    //      component is harmless.
    //   2. Any shape where the slice drops NO transition: deadlock and
    //      next-step structure are then identical to the full net, so even
    //      stutter-sensitive (EX/AX) and maximal-path (EG/AF/AG/AU/EU over a
    //      non-atomic inner) verdicts are preserved.
    // For all other shapes, dropping a structurally-disjoint livelock/self-loop
    // can flip the verdict, so we discard the slice and explore the full
    // reduced net — exact, never wrong.
    let slice_invariant_reachability =
        batch_is_slice_invariant_reachability(&pending_expensive_properties);
    // Fair-cycle (EGF) batches must NOT be query-sliced. A slice can keep every
    // transition yet drop an OUTPUT place, which changes the marking valuation
    // the fair-cycle fixpoint observes at deadlock / cycle states — the
    // `slice_keeps_all_transitions` guard checks transitions, not places, so it
    // does not catch this. Concretely, the LTL persistence lane
    // `A(FG p) ≡ ¬EGF(¬p)` on a net that deadlocks in the `¬p` state had the
    // deep relevance-cone slice corrupt the deadlock marking (the `¬p` atom read
    // false everywhere), flipping the verdict. Explore the full reduced net for
    // these — exact, never wrong; the structural reduction is likewise forced to
    // identity for fair-cycle batches (see `reduce_ctl_batch_for_mode`).
    let batch_has_fair_cycle = crate::examinations::ctl::routing::ctl_batch_contains_fair_cycle(
        &pending_expensive_properties,
    );
    let slice = if shortcuts_enabled && !batch_has_fair_cycle {
        let use_deep_slice = !has_next_step;
        ctl_support_with_aliases(&reduced, &pending_expensive_properties, aliases)
            .and_then(|support| {
                if use_deep_slice {
                    let cone = relevance_cone_on_reduced_net(&reduced.net, support);
                    build_query_local_slice(&reduced.net, &cone)
                } else {
                    let closure = closure_on_reduced_net(&reduced.net, support);
                    build_query_slice(&reduced.net, &closure)
                }
            })
            .filter(|slice| {
                let safe = slice_invariant_reachability
                    || slice_keeps_all_transitions(slice, &reduced.net);
                if !safe {
                    eprintln!(
                        "CTL slice drops {} transition(s) for a stutter/max-path \
                         batch (EX/AX/EG/AF/AG/AU/EU non-atomic-inner) -> \
                         exploring full reduced net",
                        reduced.net.num_transitions() - slice.net.num_transitions()
                    );
                }
                safe
            })
    } else {
        None
    };

    let explore_net = slice.as_ref().map_or(&reduced.net, |slice| &slice.net);
    let unresolved_for_local_fallback = pending_expensive_properties.len();
    let explore_config =
        config
            .refitted_for_full_graph(explore_net)
            .with_deadline(ctl_full_graph_deadline(
                config.deadline(),
                unresolved_for_local_fallback,
                local_fallback_enabled,
            ));
    let mut full = explore_full(explore_net, &explore_config);
    if !full.graph.completed {
        if !local_fallback_enabled {
            eprintln!("CTL local fallback disabled after incomplete graph -> CANNOT_COMPUTE");
            return collect_shallow_or_cannot_compute(properties, &shallow_map, &flushed_ids);
        }

        let mut results = Vec::new();
        let mut unresolved_count = pending_expensive_properties.len();

        for prop in properties {
            if flushed_ids.contains(&prop.id) {
                continue;
            }
            if let Some(&verdict) = shallow_map.get(&prop.id) {
                if !flush_property_result(&prop.id, verdict, &mut flushed_ids, flush, techniques) {
                    results.push((prop.id.clone(), verdict));
                }
                continue;
            }

            if guard_mode == SoundnessGuardMode::Enforce
                && is_known_mcc_ctl_soundness_guard(&prop.id)
            {
                let verdict = Verdict::CannotCompute;
                if !flush_property_result(&prop.id, verdict, &mut flushed_ids, flush, techniques) {
                    results.push((prop.id.clone(), verdict));
                }
                unresolved_count = unresolved_count.saturating_sub(1);
                continue;
            }

            let remaining_count = unresolved_count.max(1);
            let verdict = match &prop.formula {
                Formula::Ctl(ctl) => {
                    let (total, unresolved) = count_unresolved_ctl_with_aliases(ctl, aliases);
                    if unresolved > 0 {
                        eprintln!(
                            "CTL resolution guard: {} has {unresolved}/{total} \
                             unresolved names -> CANNOT_COMPUTE",
                            prop.id
                        );
                        Verdict::CannotCompute
                    } else {
                        let resolved = resolve::resolve_ctl_with_aliases(ctl, aliases);
                        // SOUND symbolic-BDD CTL lane (fail-closed). Before
                        // resorting to the budget-limited local solvers, try
                        // the oracle-verified symbolic CTL evaluator over the
                        // exact reachable set of the original net. It is used
                        // ONLY when it returns `Some(_)` (every gate passed:
                        // DD-eligible net, every atom converted, in-budget
                        // reachability fixpoint); on ANY decline (`None`) we
                        // fall through to the EXISTING behavior unchanged.
                        // BUDGET CAP (fixes the cap-raise GLOBAL-DRAIN regression):
                        // the DD lane previously ran on the FULL `config.deadline()`,
                        // so on a non-converging high-bound binary BDD it drained the
                        // global wall-clock across the per-formula loop and starved the
                        // explicit lanes' fair share on LATER formulas (Kanban
                        // CTLFireability 2/16 -> 1/16, decided cell CTLFireability-2023-13
                        // timing out). Cap the DD lane at 1/4 of the per-formula fair
                        // share (the same fraction the EDG pre-pass uses): a converging
                        // net decides in well under that, while a non-converging net
                        // bails early and leaves the explicit lanes their budget intact.
                        // Fail-closed either way (None -> explicit lanes -> CANNOT_COMPUTE).
                        let dd_lane_deadline = fair_share_deadline(
                            config.deadline(),
                            remaining_count.saturating_mul(4),
                        );
                        let bdd_verdict =
                            try_symbolic_ctl_verdict(net, &resolved, dd_lane_deadline);
                        // CROSS-VALIDATED MDD CTL lane (fail-closed). The MDD
                        // counterpart targets the counter / conserved / high-bound
                        // nets the bit-blasted BDD lane blows up on. Its verdict is
                        // published ONLY via the cross-validated gate:
                        //   (a) BDD returned Some(v): the BDD verdict is itself
                        //       SOUND and is published. We additionally run the MDD
                        //       as a cross-check — on a disagreement we LOG it and
                        //       keep the (oracle-verified) BDD verdict, NEVER a
                        //       wrong MDD verdict.
                        //   (b) BDD DECLINED (the target case): adopt the MDD
                        //       verdict (exact-by-construction; the evaluator is
                        //       pinned to `tla_mc_core::CtlEngine` on every
                        //       reachable marking by the soaked differential
                        //       battery). Fail-closed: an MDD decline falls through
                        //       to solve_local_edg / LocalCtlChecker as today.
                        // The MDD lane gets its OWN fair-share slice so it cannot
                        // drain the explicit lanes' budget on later formulas.
                        let mdd_lane_deadline = fair_share_deadline(
                            config.deadline(),
                            remaining_count.saturating_mul(4),
                        );
                        let published = match bdd_verdict {
                            Some(v) => {
                                // Case (a): publish the sound BDD verdict; the MDD
                                // is a cross-check only.
                                let mdd = try_mdd_ctl_verdict(net, &resolved, mdd_lane_deadline);
                                if let Some(m) = mdd {
                                    if m != v {
                                        eprintln!(
                                            "CTL: MDD lane disagreed with BDD lane on {} \
                                             (MDD={m} BDD={v}) — keeping the oracle-verified \
                                             BDD verdict (fail-closed)",
                                            prop.id
                                        );
                                    }
                                }
                                eprintln!(
                                    "CTL local fallback: {} -> {} (decided by symbolic BDD CTL lane)",
                                    prop.id,
                                    if v { "TRUE" } else { "FALSE" }
                                );
                                Some(v)
                            }
                            None => {
                                // Case (b): BDD declined ⇒ adopt the MDD verdict.
                                match try_mdd_ctl_verdict(net, &resolved, mdd_lane_deadline) {
                                    Some(m) => {
                                        eprintln!(
                                            "CTL local fallback: {} -> {} (decided by symbolic \
                                             MDD CTL lane — BDD declined)",
                                            prop.id,
                                            if m { "TRUE" } else { "FALSE" }
                                        );
                                        Some(m)
                                    }
                                    None => None,
                                }
                            }
                        };
                        if let Some(symbolic) = published {
                            if !flush_property_result(
                                &prop.id,
                                if symbolic {
                                    Verdict::True
                                } else {
                                    Verdict::False
                                },
                                &mut flushed_ids,
                                flush,
                                techniques,
                            ) {
                                results.push((
                                    prop.id.clone(),
                                    if symbolic {
                                        Verdict::True
                                    } else {
                                        Verdict::False
                                    },
                                ));
                            }
                            unresolved_count = unresolved_count.saturating_sub(1);
                            continue;
                        }
                        let local_config = config
                            .clone()
                            .with_deadline(fair_share_deadline(config.deadline(), remaining_count));
                        match solve_local_edg(net, &resolved, &local_config) {
                            Ok(true) => Verdict::True,
                            Ok(false) => Verdict::False,
                            Err(edg_error) => {
                                // SOUNDNESS GATE: the recursive `LocalCtlChecker`
                                // is a single-pass DFS with per-node cycle
                                // assumptions and write-once memoization; it is a
                                // correct local fixpoint computation only for a
                                // SINGLE fixpoint layer. `ctl_is_alternation_free`
                                // is too weak here — it admits nested SAME-class
                                // fixpoints (ν-in-ν, μ-in-μ) and fixpoints under
                                // `EX`/`AX`, where the write-once cache can be
                                // poisoned by an inner node committed `Ready` under
                                // a still-`Active` outer assumption that later
                                // resolves the other way (confirmed on a 4-state
                                // net with `EG(AG(p3≤0) ∨ (p1+p3≤0))`). It also
                                // returned a WRONG `AG(EF(fireable tredo4)) = FALSE`
                                // on Kanban (true answer TRUE) after interning a few
                                // hundred states (Kanban-PT-00200-CTLFireability-
                                // 2025-05, -CTLCardinality-2025-09). A wrong CTL
                                // verdict is catastrophic, so we consult the
                                // recursive checker ONLY for a strict single
                                // fixpoint layer (which is a subset of the
                                // alternation-free formulas); everything else
                                // returns CANNOT_COMPUTE.
                                if ctl_is_alternation_free(&resolved)
                                    && ctl_has_single_fixpoint_layer(&resolved)
                                {
                                    let mut checker = LocalCtlChecker::new(net, &local_config);
                                    match checker.eval_root(&resolved) {
                                        Ok(true) => Verdict::True,
                                        Ok(false) => Verdict::False,
                                        Err(recursive_error) => {
                                            eprintln!(
                                                "CTL local fallback: {} -> CANNOT_COMPUTE \
                                                 (EDG: {edg_error}; recursive: {recursive_error})",
                                                prop.id
                                            );
                                            Verdict::CannotCompute
                                        }
                                    }
                                } else {
                                    eprintln!(
                                        "CTL local fallback: {} -> CANNOT_COMPUTE (EDG: \
                                         {edg_error}; recursive checker skipped: formula is not a \
                                         single fixpoint layer — single-pass cycle assumption \
                                         unsound)",
                                        prop.id
                                    );
                                    Verdict::CannotCompute
                                }
                            }
                        }
                    }
                }
                _ => Verdict::CannotCompute,
            };
            if !flush_property_result(&prop.id, verdict, &mut flushed_ids, flush, techniques) {
                results.push((prop.id.clone(), verdict));
            }
            unresolved_count = unresolved_count.saturating_sub(1);
        }

        return results;
    }

    // Expand each explored marking back to the FULL net's place-space (undoing the
    // structural reduction / slice) and re-pack under the full net's config — the
    // subsequent `CtlChecker::new(&full, net)` evaluates full-net predicates. Kept
    // packed throughout (only a transient unpack scratch), preserving the memory win.
    let full_config = crate::explorer::ExplorationSetup::analyze(net).marking_config;
    let rebuilt = full.markings.try_rebuild_with(full_config, |marking| {
        if let Some(ref slice) = slice {
            let mut reduced_marking = vec![0u64; reduced.net.num_places()];
            for (sliced_idx, &tokens) in marking.iter().enumerate() {
                reduced_marking[slice.place_unmap[sliced_idx].0 as usize] = tokens;
            }
            reduced.expand_marking(&reduced_marking)
        } else {
            reduced.expand_marking(marking)
        }
    });
    match rebuilt {
        Ok(rebuilt) => full.markings = rebuilt,
        Err(error) => {
            eprintln!("CTL: CANNOT_COMPUTE ({error})");
            return collect_shallow_or_cannot_compute(properties, &shallow_map, &flushed_ids);
        }
    }

    let checker = CtlChecker::new(&full, net);

    properties
        .iter()
        .filter_map(|prop| {
            if flushed_ids.contains(&prop.id) {
                return None;
            }
            if let Some(&verdict) = shallow_map.get(&prop.id) {
                return Some((prop.id.clone(), verdict));
            }

            if guard_mode == SoundnessGuardMode::Enforce
                && is_known_mcc_ctl_soundness_guard(&prop.id)
            {
                return Some((prop.id.clone(), Verdict::CannotCompute));
            }

            let verdict = match &prop.formula {
                Formula::Ctl(ctl) => {
                    let (total, unresolved) = count_unresolved_ctl_with_aliases(ctl, aliases);
                    if unresolved > 0 {
                        eprintln!(
                            "CTL resolution guard: {} has {unresolved}/{total} \
                             unresolved names -> CANNOT_COMPUTE",
                            prop.id
                        );
                        return Some((prop.id.clone(), Verdict::CannotCompute));
                    }

                    let resolved = resolve::resolve_ctl_with_aliases(ctl, aliases);
                    if checker.eval_root(&resolved) {
                        Verdict::True
                    } else {
                        Verdict::False
                    }
                }
                _ => Verdict::CannotCompute,
            };
            Some((prop.id.clone(), verdict))
        })
        .collect()
}

/// Differential battery for the SOUND symbolic-BDD CTL lane.
///
/// For every DD-eligible small net on which `explore_full` completes, the
/// verdict produced by the wired symbolic lane
/// ([`try_symbolic_ctl_verdict`], the EXACT code path the production fallback
/// runs) must equal the exhaustive [`CtlChecker`] verdict at the initial
/// marking. Zero disagreements is the ship gate.
///
/// The atom battery exercises **real atoms** mandated by the contract:
/// per-place `TokensCount` comparisons, a multi-place sum-of-places
/// `TokensCount` comparison, and `IsFireable`. The formula battery exercises
/// every operator plus the required nested temporal shapes — `AG(EF p)`,
/// `EF(AG p)`, `EU`, `AU` (and nested forms).
#[cfg(all(test, feature = "dd-backend"))]
mod symbolic_lane_diff_tests {
    use super::*;
    use crate::examinations::ctl::checker::CtlChecker;
    use crate::examinations::ctl::resolve::ResolvedCtl;
    use crate::explorer::{explore_full, ExplorationConfig};
    use crate::petri_net::{Arc, PetriNet, PlaceIdx, PlaceInfo, TransitionIdx, TransitionInfo};
    use crate::resolved_predicate::{ResolvedIntExpr, ResolvedPredicate};

    fn place(i: usize) -> PlaceInfo {
        PlaceInfo {
            id: format!("p{i}"),
            name: None,
        }
    }

    /// Build a real `PetriNet` from per-transition `(pre, post)` weight
    /// vectors over `np` places.
    fn net(np: usize, init: Vec<u64>, transitions: &[(Vec<u64>, Vec<u64>)]) -> PetriNet {
        let places = (0..np).map(place).collect();
        let transitions = transitions
            .iter()
            .enumerate()
            .map(|(ti, (pre, post))| {
                let inputs = pre
                    .iter()
                    .enumerate()
                    .filter(|(_, &w)| w > 0)
                    .map(|(p, &w)| Arc {
                        place: PlaceIdx(p as u32),
                        weight: w,
                    })
                    .collect();
                let outputs = post
                    .iter()
                    .enumerate()
                    .filter(|(_, &w)| w > 0)
                    .map(|(p, &w)| Arc {
                        place: PlaceIdx(p as u32),
                        weight: w,
                    })
                    .collect();
                TransitionInfo {
                    id: format!("t{ti}"),
                    name: None,
                    inputs,
                    outputs,
                }
            })
            .collect();
        PetriNet {
            name: Some("diff".into()),
            places,
            transitions,
            initial_marking: init,
        }
    }

    /// A battery of bounded, DD-eligible (per-place bound <= 16) real Petri
    /// nets covering: hard deadlock, cycles, no-transition deadlock,
    /// branch-into-loop-and-deadlock, bounded producer/consumer, source/sink,
    /// two independent tokens (cycle + drain), self-loop, counter-to-deadlock.
    fn battery_nets() -> Vec<(&'static str, PetriNet)> {
        vec![
            (
                "drain_to_deadlock",
                net(2, vec![1, 0], &[(vec![1, 0], vec![0, 1])]),
            ),
            (
                "ping_pong_cycle",
                net(
                    2,
                    vec![1, 0],
                    &[(vec![1, 0], vec![0, 1]), (vec![0, 1], vec![1, 0])],
                ),
            ),
            ("no_transitions_deadlock", net(2, vec![1, 1], &[])),
            (
                "branch_loop_and_deadlock",
                net(
                    3,
                    vec![2, 0, 0],
                    &[
                        (vec![1, 0, 0], vec![0, 1, 0]),
                        (vec![0, 1, 0], vec![0, 0, 1]),
                        (vec![0, 0, 1], vec![0, 0, 1]),
                        (vec![0, 1, 0], vec![1, 0, 0]),
                    ],
                ),
            ),
            (
                "bounded_buffer",
                net(
                    3,
                    vec![3, 0, 0],
                    &[
                        (vec![1, 0, 0], vec![0, 1, 0]),
                        (vec![0, 1, 0], vec![0, 0, 1]),
                    ],
                ),
            ),
            (
                "mixed_two_tokens",
                net(
                    4,
                    vec![1, 0, 1, 0],
                    &[
                        (vec![1, 0, 0, 0], vec![0, 1, 0, 0]),
                        (vec![0, 1, 0, 0], vec![1, 0, 0, 0]),
                        (vec![0, 0, 1, 0], vec![0, 0, 0, 1]),
                    ],
                ),
            ),
            ("single_self_loop", net(1, vec![1], &[(vec![1], vec![1])])),
            (
                // Bounded counter: 4 tokens drain p0 -> p1 one at a time, then
                // deadlock (conserved at 4, both places <= 4 = DD-eligible).
                "counter_to_deadlock",
                net(2, vec![4, 0], &[(vec![1, 0], vec![0, 1])]),
            ),
        ]
    }

    // ---- Resolved-atom shorthands (real atoms) ----

    fn r_ge(place: usize, c: u64) -> ResolvedPredicate {
        ResolvedPredicate::IntLe(
            ResolvedIntExpr::Constant(c),
            ResolvedIntExpr::TokensCount(vec![PlaceIdx(place as u32)]),
        )
    }
    fn r_le(place: usize, c: u64) -> ResolvedPredicate {
        ResolvedPredicate::IntLe(
            ResolvedIntExpr::TokensCount(vec![PlaceIdx(place as u32)]),
            ResolvedIntExpr::Constant(c),
        )
    }
    fn r_sum_le(np: usize, c: u64) -> ResolvedPredicate {
        // Sum-of-places TokensCount comparison (multi-place).
        ResolvedPredicate::IntLe(
            ResolvedIntExpr::TokensCount((0..np).map(|p| PlaceIdx(p as u32)).collect()),
            ResolvedIntExpr::Constant(c),
        )
    }
    fn r_fire(t: usize) -> ResolvedPredicate {
        ResolvedPredicate::IsFireable(vec![TransitionIdx(t as u32)])
    }
    fn a(p: ResolvedPredicate) -> ResolvedCtl {
        ResolvedCtl::Atom(p)
    }

    /// A net-shape-aware battery of resolved CTL formulas exercising every
    /// operator, the dualities, and the contract's nested temporal shapes.
    fn formula_battery(np: usize, nt: usize) -> Vec<ResolvedCtl> {
        let mut atoms: Vec<ResolvedPredicate> =
            vec![ResolvedPredicate::True, ResolvedPredicate::False];
        for p in 0..np {
            atoms.push(r_ge(p, 1));
            atoms.push(r_le(p, 0));
        }
        for t in 0..nt {
            atoms.push(r_fire(t));
        }
        if np >= 2 {
            atoms.push(r_sum_le(np, 1)); // sum-of-places TokensCount compared
        }

        let mut fs: Vec<ResolvedCtl> = Vec::new();
        for at in &atoms {
            let f = a(at.clone());
            fs.push(ResolvedCtl::Not(Box::new(f.clone())));
            fs.push(ResolvedCtl::EX(Box::new(f.clone())));
            fs.push(ResolvedCtl::AX(Box::new(f.clone())));
            fs.push(ResolvedCtl::EF(Box::new(f.clone())));
            fs.push(ResolvedCtl::AF(Box::new(f.clone())));
            fs.push(ResolvedCtl::EG(Box::new(f.clone())));
            fs.push(ResolvedCtl::AG(Box::new(f.clone())));
            // Contract-mandated nested temporal shapes.
            fs.push(ResolvedCtl::AG(Box::new(ResolvedCtl::EF(Box::new(
                f.clone(),
            )))));
            fs.push(ResolvedCtl::EF(Box::new(ResolvedCtl::AG(Box::new(
                f.clone(),
            )))));
            fs.push(ResolvedCtl::Not(Box::new(ResolvedCtl::EF(Box::new(
                ResolvedCtl::AG(Box::new(f.clone())),
            )))));
            fs.push(ResolvedCtl::AG(Box::new(ResolvedCtl::AF(Box::new(
                f.clone(),
            )))));
            fs.push(ResolvedCtl::EF(Box::new(ResolvedCtl::EG(Box::new(
                f.clone(),
            )))));
        }

        fs.push(ResolvedCtl::EG(Box::new(a(ResolvedPredicate::True))));
        fs.push(ResolvedCtl::AG(Box::new(a(ResolvedPredicate::True))));
        fs.push(ResolvedCtl::AF(Box::new(a(ResolvedPredicate::False))));
        fs.push(ResolvedCtl::EF(Box::new(a(ResolvedPredicate::True))));

        // Pairwise + nested EU / AU.
        for i in 0..atoms.len().min(4) {
            for j in 0..atoms.len().min(4) {
                let phi = a(atoms[i].clone());
                let psi = a(atoms[j].clone());
                fs.push(ResolvedCtl::EU(
                    Box::new(phi.clone()),
                    Box::new(psi.clone()),
                ));
                fs.push(ResolvedCtl::AU(
                    Box::new(phi.clone()),
                    Box::new(psi.clone()),
                ));
                fs.push(ResolvedCtl::EU(
                    Box::new(phi.clone()),
                    Box::new(ResolvedCtl::EF(Box::new(psi.clone()))),
                ));
                fs.push(ResolvedCtl::AU(
                    Box::new(ResolvedCtl::AX(Box::new(phi.clone()))),
                    Box::new(psi.clone()),
                ));
            }
        }
        fs
    }

    /// Exhaustive ground-truth verdict at the initial marking via the
    /// completed explicit `CtlChecker` (the same checker the completed-graph
    /// path uses).
    fn exhaustive_verdict(petri: &PetriNet, formula: &ResolvedCtl) -> bool {
        let config = ExplorationConfig::default();
        let full = explore_full(petri, &config);
        assert!(
            full.graph.completed,
            "battery net must explore to completion for the differential oracle"
        );
        let checker = CtlChecker::new(&full, petri);
        checker.eval_root(formula)
    }

    /// THE differential test: wired symbolic lane verdict == exhaustive
    /// CtlChecker verdict, zero disagreements, on DD-eligible nets.
    #[test]
    fn symbolic_ctl_lane_matches_exhaustive_ctlchecker_zero_disagreements() {
        let mut total = 0usize;
        let mut decided = 0usize;
        let mut disagreements = 0usize;
        for (name, petri) in battery_nets() {
            let np = petri.num_places();
            let nt = petri.num_transitions();
            for f in formula_battery(np, nt) {
                let oracle = exhaustive_verdict(&petri, &f);
                // EXACT production lane code path.
                match try_symbolic_ctl_verdict(&petri, &f, None) {
                    Some(symbolic) => {
                        decided += 1;
                        if symbolic != oracle {
                            disagreements += 1;
                            eprintln!(
                                "DISAGREEMENT net='{name}' formula={f:?} \
                                 symbolic={symbolic} oracle={oracle}"
                            );
                        }
                    }
                    None => {
                        // Declined (fail-closed) — acceptable; not a verdict.
                    }
                }
                total += 1;
            }
        }
        eprintln!(
            "symbolic-CTL-lane differential: {total} checks, {decided} decided by lane, \
             {disagreements} disagreements"
        );
        assert_eq!(
            disagreements, 0,
            "symbolic CTL lane disagreed with exhaustive CtlChecker"
        );
        // The lane must actually decide a large fraction (these nets are all
        // DD-eligible with convertible atoms), proving it is not dormant.
        assert!(
            decided >= 500,
            "differential battery decided too few ({decided}) — lane may be dormant"
        );
    }

    /// GPU retained-graph CTL engine differential: the SAME battery, zero
    /// disagreements against the exhaustive CtlChecker. Batched per net (the
    /// engine evaluates a formula batch over one retained reachable set).
    /// Skipped on CUDA-less hosts; a decline is only tolerated for the
    /// zero-transition net (the lane's empty-net gate).
    #[cfg(feature = "gpu")]
    #[test]
    fn gpu_ctl_lane_matches_exhaustive_ctlchecker_zero_disagreements() {
        if tla_gpu::probe().is_err() {
            eprintln!("skipping GPU CTL differential: no usable CUDA device");
            return;
        }
        let mut total = 0usize;
        let mut disagreements = 0usize;
        for (name, petri) in battery_nets() {
            let formulas = formula_battery(petri.num_places(), petri.num_transitions());
            let Some(gpu) =
                crate::gpu_state_space::ctl_check_gpu(&petri, &formulas, 1_000_000, None)
            else {
                assert_eq!(
                    petri.num_transitions(),
                    0,
                    "GPU CTL lane declined admissible battery net '{name}'"
                );
                continue;
            };
            assert_eq!(gpu.len(), formulas.len());
            for (f, g) in formulas.iter().zip(gpu) {
                let oracle = exhaustive_verdict(&petri, f);
                if g != oracle {
                    disagreements += 1;
                    eprintln!(
                        "GPU DISAGREEMENT net='{name}' formula={f:?} gpu={g} oracle={oracle}"
                    );
                }
                total += 1;
            }
        }
        eprintln!("gpu-CTL differential: {total} checks, {disagreements} disagreements");
        assert!(total >= 500, "battery too small ({total})");
        assert_eq!(
            disagreements, 0,
            "GPU CTL engine disagreed with exhaustive CtlChecker"
        );
    }

    /// A net with a per-place bound > 16 (binary band) is now ADMITTED and
    /// DECIDED exactly via the binary (log-encoded) DD path (it used to
    /// fail-closed). A conserved 17-token shuttle has reachable markings
    /// {(17,0),(16,1),...,(0,17)}, so `EF(p1>=1)` is reachable (TRUE) and
    /// `EF(p1>=18)` is impossible — the conserved sum caps p1 at 17 (FALSE).
    #[test]
    fn lane_decides_high_bound_net_via_binary_path() {
        let petri = net(
            2,
            vec![17, 0],
            &[(vec![1, 0], vec![0, 1]), (vec![0, 1], vec![1, 0])],
        );
        assert_eq!(
            try_symbolic_ctl_verdict(&petri, &ResolvedCtl::EF(Box::new(a(r_ge(1, 1)))), None),
            Some(true),
            "bound-17 net is now decided by the binary DD path: EF(p1>=1) is reachable",
        );
        assert_eq!(
            try_symbolic_ctl_verdict(&petri, &ResolvedCtl::EF(Box::new(a(r_ge(1, 18)))), None),
            Some(false),
            "bound-17 net: conserved sum caps p1 at 17 ⇒ EF(p1>=18) is impossible",
        );
    }

    /// Fail-closed: a net whose per-place bound exceeds the binary cap
    /// (MAX_BINARY_PLACE_BOUND = 2^20) is genuinely unrepresentable, so the
    /// spec builder declines and the lane DECLINES (`None`), never guessing —
    /// the always-sound explicit fallback stays in charge.
    #[test]
    fn lane_declines_above_binary_cap() {
        let n = tla_dd::MAX_BINARY_PLACE_BOUND + 1;
        let petri = net(
            2,
            vec![n, 0],
            &[(vec![1, 0], vec![0, 1]), (vec![0, 1], vec![1, 0])],
        );
        let f = ResolvedCtl::EF(Box::new(a(r_ge(1, 1))));
        assert_eq!(
            try_symbolic_ctl_verdict(&petri, &f, None),
            None,
            "bound above MAX_BINARY_PLACE_BOUND must decline (build_sound_dd_spec gates it out)"
        );
    }

    /// Fail-closed: an unsupported atom (here a deliberately out-of-range
    /// transition index) must make the formula conversion DECLINE, so the
    /// whole lane declines rather than guessing.
    #[test]
    fn lane_declines_unconvertible_atom() {
        let petri = net(2, vec![1, 0], &[(vec![1, 0], vec![0, 1])]);
        // transition index 5 does not exist (net has 1 transition).
        let f = ResolvedCtl::EF(Box::new(a(ResolvedPredicate::IsFireable(vec![
            TransitionIdx(5),
        ]))));
        assert!(
            resolved_ctl_to_mdd_template(&f, petri.num_places(), petri.num_transitions()).is_none(),
            "out-of-range atom must make conversion decline"
        );
        assert_eq!(
            try_symbolic_ctl_verdict(&petri, &f, None),
            None,
            "unconvertible atom must make the lane decline"
        );
    }

    // ====================================================================
    // MDD CTL lane (the new lane) — same production seam, same oracle.
    // ====================================================================

    /// THE MDD differential: the wired `try_mdd_ctl_verdict` lane verdict ==
    /// exhaustive `CtlChecker` verdict, zero disagreements, on the SAME
    /// DD-eligible battery + formula battery the BDD lane uses. The lane must
    /// actually decide a large fraction (proving it is not dormant).
    #[test]
    fn mdd_ctl_lane_matches_exhaustive_ctlchecker_zero_disagreements() {
        // The lane is opt-in (default off); force it ON to exercise it here.
        with_mdd_ctl_enabled_for_test(|| {
            let mut total = 0usize;
            let mut decided = 0usize;
            let mut disagreements = 0usize;
            for (name, petri) in battery_nets() {
                let np = petri.num_places();
                let nt = petri.num_transitions();
                for f in formula_battery(np, nt) {
                    let oracle = exhaustive_verdict(&petri, &f);
                    match try_mdd_ctl_verdict(&petri, &f, None) {
                        Some(mdd) => {
                            decided += 1;
                            if mdd != oracle {
                                disagreements += 1;
                                eprintln!(
                                    "DISAGREEMENT (MDD) net='{name}' formula={f:?} \
                                 mdd={mdd} oracle={oracle}"
                                );
                            }
                        }
                        None => { /* declined (fail-closed) — acceptable */ }
                    }
                    total += 1;
                }
            }
            eprintln!(
                "MDD-CTL-lane differential: {total} checks, {decided} decided by lane, \
             {disagreements} disagreements"
            );
            assert_eq!(
                disagreements, 0,
                "MDD CTL lane disagreed with exhaustive CtlChecker"
            );
            assert!(
                decided >= 500,
                "MDD differential battery decided too few ({decided}) — lane may be dormant"
            );
        });
    }

    /// The MDD lane DECIDES a conserved high-bound net (the target family). On a
    /// conserved 17-token shuttle the reachable markings are
    /// {(17,0),(16,1),...,(0,17)}, so `EF(p1>=1)` is reachable (TRUE) and the
    /// conserved sum caps p1 at 17, making `EF(p1>=18)` impossible (FALSE).
    #[test]
    fn mdd_lane_decides_high_bound_conserved_net() {
        // The lane is opt-in (default off); force it ON to exercise it here.
        with_mdd_ctl_enabled_for_test(|| {
            let petri = net(
                2,
                vec![17, 0],
                &[(vec![1, 0], vec![0, 1]), (vec![0, 1], vec![1, 0])],
            );
            assert_eq!(
                try_mdd_ctl_verdict(&petri, &ResolvedCtl::EF(Box::new(a(r_ge(1, 1)))), None),
                Some(true),
                "MDD lane decides EF(p1>=1) TRUE on the conserved shuttle",
            );
            assert_eq!(
                try_mdd_ctl_verdict(&petri, &ResolvedCtl::EF(Box::new(a(r_ge(1, 18)))), None),
                Some(false),
                "MDD lane decides EF(p1>=18) FALSE (conserved sum caps p1 at 17)",
            );
            // AG(sum<=17) holds (conserved); AG(p0<=17) holds (bound).
            assert_eq!(
                try_mdd_ctl_verdict(&petri, &ResolvedCtl::AG(Box::new(a(r_sum_le(2, 17)))), None),
                Some(true),
                "MDD lane decides AG(sum<=17) TRUE (token-conserving)",
            );
        });
    }

    /// DEFAULT-OFF gating (opt-in — for PERFORMANCE, not soundness; see
    /// `mdd_ctl_disabled_from`). The lane is DISABLED unless
    /// `TY_MCC_ENABLE_MDD_CTL` is truthy; `TY_MCC_DISABLE_MDD_CTL` truthy WINS.
    /// Tested via the pure (env-free) decision so it does not race parallel tests.
    #[test]
    fn mdd_lane_default_off_gating() {
        // Default: both unset ⇒ DISABLED (opt-in).
        assert!(
            mdd_ctl_disabled_from(None, None),
            "both unset ⇒ lane DISABLED (default off)"
        );

        // Enable truthy (and disable unset) ⇒ ENABLED.
        for v in ["1", "on", "ON", "true", "TRUE", "yes", "Yes"] {
            assert!(
                !mdd_ctl_disabled_from(None, Some(v.to_string())),
                "TY_MCC_ENABLE_MDD_CTL='{v}' must ENABLE the lane"
            );
        }
        // Enable non-truthy ⇒ still DISABLED.
        for v in ["0", "off", "false", "no", ""] {
            assert!(
                mdd_ctl_disabled_from(None, Some(v.to_string())),
                "TY_MCC_ENABLE_MDD_CTL='{v}' is not truthy ⇒ lane DISABLED"
            );
        }

        // Disable WINS over enable.
        assert!(
            mdd_ctl_disabled_from(Some("1".to_string()), Some("1".to_string())),
            "DISABLE truthy must win even when ENABLE is truthy"
        );
        assert!(
            mdd_ctl_disabled_from(Some("yes".to_string()), None),
            "DISABLE truthy ⇒ lane DISABLED"
        );
        // Disable non-truthy + enable unset ⇒ still default OFF.
        assert!(
            mdd_ctl_disabled_from(Some("0".to_string()), None),
            "DISABLE non-truthy + no ENABLE ⇒ default OFF"
        );
    }

    /// Fail-closed: an unconvertible atom (out-of-range transition) makes the
    /// MDD lowering decline, so the whole MDD lane declines.
    #[test]
    fn mdd_lane_declines_unconvertible_atom() {
        // Force the lane ON so the decline is genuinely from the unconvertible
        // atom (not from the default-off gate).
        with_mdd_ctl_enabled_for_test(|| {
            let petri = net(2, vec![1, 0], &[(vec![1, 0], vec![0, 1])]);
            let f = ResolvedCtl::EF(Box::new(a(ResolvedPredicate::IsFireable(vec![
                TransitionIdx(5),
            ]))));
            assert!(
                resolved_ctl_to_mdd_template(&f, petri.num_places(), petri.num_transitions())
                    .is_none(),
                "out-of-range atom must make the MDD template conversion decline"
            );
            assert_eq!(
                try_mdd_ctl_verdict(&petri, &f, None),
                None,
                "unconvertible atom must make the MDD lane decline"
            );
        });
    }

    /// The default (no test override, no env) leaves the MDD CTL lane OFF, so a
    /// fully-convertible formula on a DD-eligible net declines through the gate.
    /// This pins the production default-off behavior.
    #[test]
    fn mdd_lane_default_off_declines_even_convertible_formula() {
        if std::env::var("TY_MCC_ENABLE_MDD_CTL").is_ok() {
            eprintln!("SKIP: default-off assertion is vacuous with the lane explicitly enabled");
            return;
        }
        let petri = net(
            2,
            vec![17, 0],
            &[(vec![1, 0], vec![0, 1]), (vec![0, 1], vec![1, 0])],
        );
        // No `with_mdd_ctl_enabled_for_test` wrapper ⇒ the lane is OFF by default
        // and must decline regardless of how decidable the formula is.
        assert_eq!(
            try_mdd_ctl_verdict(&petri, &ResolvedCtl::EF(Box::new(a(r_ge(1, 1)))), None),
            None,
            "default-off MDD CTL lane must decline (opt-in only)"
        );
    }
}
