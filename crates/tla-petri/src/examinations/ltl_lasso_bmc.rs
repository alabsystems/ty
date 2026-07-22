// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Counterexample-only LTL bounded-lasso search via the shared Petri BMC SMT path.

use std::collections::HashSet;
use std::time::{Duration, Instant};

use crate::buchi::{
    build_ltl_counterexample_gba, ltl_counterexample_contains_release, Gba, GbaStateId,
    GbaTransition, LtlNnf,
};
use crate::petri_net::{PetriNet, TransitionIdx};
use crate::resolved_predicate::{eval_predicate, ResolvedPredicate};

use super::bmc_runner::emit_bmc_preamble_deadlock_stutter;
use super::smt_encoding::{
    encode_predicate, find_ay, run_ay, run_ay_bool_model, SolverBoolModel, SolverOutcome,
    DEPTH_LADDER, PER_DEPTH_TIMEOUT,
};

const LTL_LASSO_MIN_SOLVER_BUDGET: Duration = Duration::from_millis(100);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct LtlLassoBmcWitness {
    pub(crate) depth: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LtlBmcTraceStep {
    Stay,
    Fire(TransitionIdx),
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct RunState {
    state: GbaStateId,
    loop_state: Option<GbaStateId>,
    accepted: Vec<bool>,
}

pub(crate) fn find_ltl_lasso_counterexample(
    net: &PetriNet,
    formula: &LtlNnf,
    atoms: &[ResolvedPredicate],
    deadline: Option<Instant>,
) -> Option<LtlLassoBmcWitness> {
    if ltl_counterexample_contains_release(formula) {
        return None;
    }

    let ay_path = find_ay()?;
    let gba = build_ltl_counterexample_gba(formula);
    if gba.num_states == 0 || gba.initial_transitions.is_empty() {
        return None;
    }

    for &depth in DEPTH_LADDER {
        if depth == 0 {
            continue;
        }

        let timeout = solver_timeout(deadline)?;
        let script = encode_lasso_check_script(net, &gba, atoms, depth);
        let outcome = run_ay(&ay_path, &script, 1, timeout)
            .and_then(|outcomes| outcomes.first().copied())
            .unwrap_or(SolverOutcome::Unknown);

        match outcome {
            SolverOutcome::Sat => {
                let timeout = solver_timeout(deadline)?;
                let model_script = encode_lasso_model_script(net, &gba, atoms, depth);
                let Some(model) = run_ay_bool_model(&ay_path, &model_script, timeout) else {
                    eprintln!(
                        "LTL lasso BMC depth {depth}: SAT model was not parseable; falling through"
                    );
                    return None;
                };

                match validate_lasso_model(net, &gba, atoms, depth, &model) {
                    Ok(()) => return Some(LtlLassoBmcWitness { depth }),
                    Err(reason) => {
                        eprintln!(
                            "LTL lasso BMC depth {depth}: SAT model rejected ({reason}); \
                             falling through"
                        );
                        return None;
                    }
                }
            }
            SolverOutcome::Unsat => {}
            SolverOutcome::Unknown => return None,
        }
    }

    None
}

fn solver_timeout(deadline: Option<Instant>) -> Option<Duration> {
    let timeout = deadline
        .map(|deadline| PER_DEPTH_TIMEOUT.min(deadline.saturating_duration_since(Instant::now())))
        .unwrap_or(PER_DEPTH_TIMEOUT);
    (timeout >= LTL_LASSO_MIN_SOLVER_BUDGET).then_some(timeout)
}

fn encode_lasso_check_script(
    net: &PetriNet,
    gba: &Gba,
    atoms: &[ResolvedPredicate],
    depth: usize,
) -> String {
    let mut script = String::with_capacity(8192);
    emit_bmc_preamble_deadlock_stutter(&mut script, net, depth);
    append_lasso_constraints(&mut script, net, gba, atoms, depth);
    script.push_str("(check-sat)\n");
    script.push_str("(exit)\n");
    script
}

fn encode_lasso_model_script(
    net: &PetriNet,
    gba: &Gba,
    atoms: &[ResolvedPredicate],
    depth: usize,
) -> String {
    let mut script = String::with_capacity(8192);
    script.push_str("(set-option :produce-models true)\n");
    emit_bmc_preamble_deadlock_stutter(&mut script, net, depth);
    append_lasso_constraints(&mut script, net, gba, atoms, depth);
    script.push_str("(check-sat)\n");
    append_bmc_step_value_query(&mut script, net, depth);
    script.push_str("(exit)\n");
    script
}

fn append_lasso_constraints(
    script: &mut String,
    net: &PetriNet,
    gba: &Gba,
    atoms: &[ResolvedPredicate],
    depth: usize,
) {
    append_gba_state_declarations(script, gba, depth);
    append_loop_declarations(script, depth);
    append_gba_state_one_hot(script, gba, depth);
    append_initial_gba_constraint(script, net, gba, atoms);
    append_gba_transition_constraints(script, net, gba, atoms, depth);
    append_loop_constraints(script, net, gba, depth);
    append_acceptance_constraints(script, net, gba, atoms, depth);
}

fn append_gba_state_declarations(script: &mut String, gba: &Gba, depth: usize) {
    for step in 0..=depth {
        for state in 0..gba.num_states {
            script.push_str(&format!(
                "(declare-const {} Bool)\n",
                gba_state_var(step, state)
            ));
        }
    }
}

fn append_loop_declarations(script: &mut String, depth: usize) {
    for loop_start in 0..depth {
        script.push_str(&format!("(declare-const {} Bool)\n", loop_var(loop_start)));
    }
}

fn append_gba_state_one_hot(script: &mut String, gba: &Gba, depth: usize) {
    for step in 0..=depth {
        let vars = (0..gba.num_states)
            .map(|state| gba_state_var(step, state))
            .collect();
        append_assert(script, exactly_one_expr(vars));
    }
}

fn append_initial_gba_constraint(
    script: &mut String,
    net: &PetriNet,
    gba: &Gba,
    atoms: &[ResolvedPredicate],
) {
    let choices = gba
        .initial_transitions
        .iter()
        .map(|transition| {
            and_expr(vec![
                gba_state_var(0, transition.successor),
                guard_expr(transition, 0, atoms, net),
            ])
        })
        .collect();
    append_assert(script, or_expr(choices));
}

fn append_gba_transition_constraints(
    script: &mut String,
    net: &PetriNet,
    gba: &Gba,
    atoms: &[ResolvedPredicate],
    depth: usize,
) {
    for step in 0..depth {
        let mut choices = Vec::new();
        for source in 0..gba.num_states {
            let Some(transitions) = gba.transitions.get(source as usize) else {
                continue;
            };
            for transition in transitions {
                choices.push(and_expr(vec![
                    gba_state_var(step, source),
                    gba_state_var(step + 1, transition.successor),
                    guard_expr(transition, step + 1, atoms, net),
                ]));
            }
        }
        append_assert(script, or_expr(choices));
    }
}

fn append_loop_constraints(script: &mut String, net: &PetriNet, gba: &Gba, depth: usize) {
    let loop_vars = (0..depth).map(loop_var).collect();
    append_assert(script, exactly_one_expr(loop_vars));

    for loop_start in 0..depth {
        let mut constraints = Vec::with_capacity(net.num_places() + gba.num_states as usize);
        for place in 0..net.num_places() {
            constraints.push(format!(
                "(= m_{}_{} m_{}_{})",
                depth, place, loop_start, place
            ));
        }
        for state in 0..gba.num_states {
            constraints.push(format!(
                "(= {} {})",
                gba_state_var(depth, state),
                gba_state_var(loop_start, state)
            ));
        }
        append_assert(
            script,
            format!("(=> {} {})", loop_var(loop_start), and_expr(constraints)),
        );
    }
}

fn append_acceptance_constraints(
    script: &mut String,
    net: &PetriNet,
    gba: &Gba,
    atoms: &[ResolvedPredicate],
    depth: usize,
) {
    for accept_index in 0..gba.acceptance.len() {
        let choices = (0..depth)
            .map(|loop_start| {
                and_expr(vec![
                    loop_var(loop_start),
                    accepted_on_loop_expr(net, gba, atoms, depth, loop_start, accept_index),
                ])
            })
            .collect();
        append_assert(script, or_expr(choices));
    }
}

fn accepted_on_loop_expr(
    net: &PetriNet,
    gba: &Gba,
    atoms: &[ResolvedPredicate],
    depth: usize,
    loop_start: usize,
    accept_index: usize,
) -> String {
    let mut clauses = Vec::new();

    for step in loop_start..depth {
        let mut states: Vec<_> = gba.acceptance[accept_index].iter().copied().collect();
        states.sort_unstable();
        for state in states {
            clauses.push(gba_state_var(step, state));
        }
    }

    for step in loop_start..depth {
        for source in 0..gba.num_states {
            let Some(transitions) = gba.transitions.get(source as usize) else {
                continue;
            };
            for transition in transitions {
                if transition
                    .edge_accept
                    .get(accept_index)
                    .copied()
                    .unwrap_or(false)
                {
                    clauses.push(and_expr(vec![
                        gba_state_var(step, source),
                        gba_state_var(step + 1, transition.successor),
                        guard_expr(transition, step + 1, atoms, net),
                    ]));
                }
            }
        }
    }

    or_expr(clauses)
}

fn append_bmc_step_value_query(script: &mut String, net: &PetriNet, depth: usize) {
    if depth == 0 {
        return;
    }

    script.push_str("(get-value (");
    let mut first = true;
    for step in 0..depth {
        append_get_value_symbol(script, &mut first, &format!("stay_{step}"));
        for transition in 0..net.num_transitions() {
            append_get_value_symbol(script, &mut first, &format!("fire_{}_{}", step, transition));
        }
    }
    script.push_str("))\n");
}

fn append_get_value_symbol(script: &mut String, first: &mut bool, symbol: &str) {
    if *first {
        *first = false;
    } else {
        script.push(' ');
    }
    script.push_str(symbol);
}

fn validate_lasso_model(
    net: &PetriNet,
    gba: &Gba,
    atoms: &[ResolvedPredicate],
    depth: usize,
    model: &SolverBoolModel,
) -> Result<(), String> {
    let trace = decode_lasso_model_trace(net, depth, model)?;
    let markings = replay_lasso_trace(net, &trace)?;
    if accepting_lasso_exists(gba, atoms, net, &markings) {
        Ok(())
    } else {
        Err("replayed trace does not contain an accepting GBA lasso".to_string())
    }
}

fn decode_lasso_model_trace(
    net: &PetriNet,
    depth: usize,
    model: &SolverBoolModel,
) -> Result<Vec<LtlBmcTraceStep>, String> {
    let mut trace = Vec::with_capacity(depth);
    for step in 0..depth {
        let stay_name = format!("stay_{step}");
        let stay = model
            .bool_value(&stay_name)
            .ok_or_else(|| format!("missing {stay_name}"))?;
        let mut fired = Vec::new();
        for transition in 0..net.num_transitions() {
            let fire_name = format!("fire_{}_{}", step, transition);
            let selected = model
                .bool_value(&fire_name)
                .ok_or_else(|| format!("missing {fire_name}"))?;
            if selected {
                fired.push(TransitionIdx(transition as u32));
            }
        }

        match (stay, fired.as_slice()) {
            (true, []) => trace.push(LtlBmcTraceStep::Stay),
            (false, [transition]) => trace.push(LtlBmcTraceStep::Fire(*transition)),
            (false, []) => return Err(format!("step {step} selects no transition or stutter")),
            _ => return Err(format!("step {step} selects multiple transition choices")),
        }
    }
    Ok(trace)
}

fn replay_lasso_trace(net: &PetriNet, trace: &[LtlBmcTraceStep]) -> Result<Vec<Vec<u64>>, String> {
    let mut marking = net.initial_marking.clone();
    let mut markings = Vec::with_capacity(trace.len() + 1);
    markings.push(marking.clone());

    for (step, action) in trace.iter().enumerate() {
        match *action {
            LtlBmcTraceStep::Stay => {
                // Defense-in-depth: a stutter is sound only at a genuine deadlock
                // marking (no enabled transition), matching the on-the-fly Büchi
                // self-loop semantics. Reject a stutter at a live marking so a
                // spurious non-deadlock self-loop is dropped even if the SMT
                // encoding ever regresses to free stutter.
                if (0..net.num_transitions())
                    .any(|t| net.is_enabled(&marking, TransitionIdx(t as u32)))
                {
                    return Err(format!(
                        "step {step} stutters at a non-deadlock marking (an enabled \
                         transition exists)"
                    ));
                }
            }
            LtlBmcTraceStep::Fire(transition) => {
                if !net.is_enabled(&marking, transition) {
                    return Err(format!(
                        "step {step} fires disabled transition {}",
                        transition.0
                    ));
                }
                // Fail-closed (#22): token-count overflow means the trace is not
                // representable — reject it rather than wrap into a wrong marking.
                net.apply_delta(&mut marking, transition)
                    .map_err(|e| format!("step {step} overflows place token count: {e}"))?;
            }
        }
        markings.push(marking.clone());
    }

    Ok(markings)
}

/// Compact, dynamically-sized bitset over atom indices.
///
/// One `u64` word per 64 atoms. `|atoms|` is typically tiny (a handful of
/// predicates), so a single word covers the common case with no allocation
/// growth, while larger formulas degrade gracefully without a `64`-atom cap.
#[derive(Clone, Debug, PartialEq, Eq)]
struct AtomMask {
    words: Vec<u64>,
}

impl AtomMask {
    fn with_capacity(num_atoms: usize) -> Self {
        let len = num_atoms.div_ceil(64).max(1);
        AtomMask {
            words: vec![0; len],
        }
    }

    #[inline]
    fn set(&mut self, atom: usize) {
        let word = atom / 64;
        if word >= self.words.len() {
            self.words.resize(word + 1, 0);
        }
        self.words[word] |= 1u64 << (atom % 64);
    }

    /// `(self & other) == self`, i.e. every bit of `self` is also set in `other`.
    #[inline]
    fn is_subset_of(&self, other: &AtomMask) -> bool {
        self.words
            .iter()
            .enumerate()
            .all(|(i, &w)| (w & other.words.get(i).copied().unwrap_or(0)) == w)
    }

    /// `(self & other) == 0`, i.e. no bit is set in both.
    #[inline]
    fn is_disjoint_from(&self, other: &AtomMask) -> bool {
        self.words
            .iter()
            .enumerate()
            .all(|(i, &w)| (w & other.words.get(i).copied().unwrap_or(0)) == 0)
    }
}

/// Precomputed positive/negative atom-index masks for one GBA transition.
struct TransitionMasks {
    pos: AtomMask,
    neg: AtomMask,
    /// Set to `false` when this transition references an atom index that is out
    /// of bounds for `atoms` (i.e. `>= num_atoms`). The reference oracle
    /// `guard_satisfied_at_marking` uses `atoms.get(atom).is_some_and(..)`, so a
    /// missing positive OR negative atom makes the guard unsatisfiable. We mirror
    /// that exactly: an out-of-bounds reference forces `satisfied_by` to return
    /// `false` regardless of the step mask. Without this, a missing *negative*
    /// atom would spuriously pass the disjointness test and diverge from the
    /// oracle.
    valid: bool,
}

impl TransitionMasks {
    fn from_transition(transition: &GbaTransition, num_atoms: usize) -> Self {
        let mut valid = true;
        let mut pos = AtomMask::with_capacity(num_atoms);
        for &atom in &transition.pos_atoms {
            if atom >= num_atoms {
                valid = false;
            }
            pos.set(atom);
        }
        let mut neg = AtomMask::with_capacity(num_atoms);
        for &atom in &transition.neg_atoms {
            if atom >= num_atoms {
                valid = false;
            }
            neg.set(atom);
        }
        TransitionMasks { pos, neg, valid }
    }

    /// Guard check against a per-step atom-truth mask. Byte-identical in result
    /// to `guard_satisfied_at_marking`: every positive atom must be true at the
    /// step, every negative atom must be false, and any out-of-bounds atom
    /// reference makes the guard unsatisfiable.
    #[inline]
    fn satisfied_by(&self, step_mask: &AtomMask) -> bool {
        self.valid && self.pos.is_subset_of(step_mask) && self.neg.is_disjoint_from(step_mask)
    }
}

/// Precomputed pos/neg masks for an entire GBA, mirroring its transition layout
/// so the inner Büchi-product loop can index masks instead of re-deriving them.
struct GbaMasks {
    initial: Vec<TransitionMasks>,
    /// `transitions[state]` parallels `gba.transitions[state]` 1:1.
    transitions: Vec<Vec<TransitionMasks>>,
}

impl GbaMasks {
    fn build(gba: &Gba, num_atoms: usize) -> Self {
        let initial = gba
            .initial_transitions
            .iter()
            .map(|t| TransitionMasks::from_transition(t, num_atoms))
            .collect();
        let transitions = gba
            .transitions
            .iter()
            .map(|row| {
                row.iter()
                    .map(|t| TransitionMasks::from_transition(t, num_atoms))
                    .collect()
            })
            .collect();
        GbaMasks {
            initial,
            transitions,
        }
    }
}

/// Per-step atom-truth bitmask for a markings trace: `step_masks[step]` has bit
/// `i` set iff `eval_predicate(atoms[i], markings[step], net)`. Computed ONCE
/// per trace ((depth+1) * |atoms| evaluations total).
fn build_step_masks(
    atoms: &[ResolvedPredicate],
    net: &PetriNet,
    markings: &[Vec<u64>],
) -> Vec<AtomMask> {
    markings
        .iter()
        .map(|marking| {
            let mut mask = AtomMask::with_capacity(atoms.len());
            for (index, predicate) in atoms.iter().enumerate() {
                if eval_predicate(predicate, marking, net) {
                    mask.set(index);
                }
            }
            mask
        })
        .collect()
}

pub(crate) fn accepting_lasso_exists(
    gba: &Gba,
    atoms: &[ResolvedPredicate],
    net: &PetriNet,
    markings: &[Vec<u64>],
) -> bool {
    if markings.len() < 2 {
        return false;
    }

    let depth = markings.len() - 1;
    // Hoisted ONCE above the per-loop_start loop: per-step atom truth + per-GBA
    // transition pos/neg masks. The same (atom, marking) pair is therefore
    // evaluated exactly once across all `loop_start` iterations instead of being
    // recomputed inside every `accepting_lasso_exists_for_loop` call.
    let step_masks = build_step_masks(atoms, net, markings);
    let gba_masks = GbaMasks::build(gba, atoms.len());

    (0..depth).any(|loop_start| {
        markings[loop_start] == markings[depth]
            && accepting_lasso_exists_for_loop(gba, &gba_masks, &step_masks, markings, loop_start)
    })
}

fn accepting_lasso_exists_for_loop(
    gba: &Gba,
    gba_masks: &GbaMasks,
    step_masks: &[AtomMask],
    markings: &[Vec<u64>],
    loop_start: usize,
) -> bool {
    let depth = markings.len() - 1;
    let num_accept = gba.acceptance.len();
    let mut runs = HashSet::new();

    for (transition, masks) in gba.initial_transitions.iter().zip(&gba_masks.initial) {
        if !masks.satisfied_by(&step_masks[0]) {
            continue;
        }
        let mut accepted = vec![false; num_accept];
        let mut loop_state = None;
        if loop_start == 0 {
            loop_state = Some(transition.successor);
            record_state_acceptance(gba, transition.successor, &mut accepted);
        }
        runs.insert(RunState {
            state: transition.successor,
            loop_state,
            accepted,
        });
    }

    for step in 0..depth {
        let mut next_runs = HashSet::new();
        for run in &runs {
            let Some(transitions) = gba.transitions.get(run.state as usize) else {
                continue;
            };
            let masks_row = gba_masks
                .transitions
                .get(run.state as usize)
                .map(Vec::as_slice)
                .unwrap_or(&[]);
            for (transition, masks) in transitions.iter().zip(masks_row) {
                if !masks.satisfied_by(&step_masks[step + 1]) {
                    continue;
                }

                let mut accepted = run.accepted.clone();
                if step >= loop_start {
                    record_edge_acceptance(transition, &mut accepted);
                }

                let mut loop_state = run.loop_state;
                let successor = transition.successor;
                if step + 1 == loop_start {
                    loop_state = Some(successor);
                    record_state_acceptance(gba, successor, &mut accepted);
                } else if step + 1 > loop_start && step + 1 < depth {
                    record_state_acceptance(gba, successor, &mut accepted);
                }

                next_runs.insert(RunState {
                    state: successor,
                    loop_state,
                    accepted,
                });
            }
        }

        if next_runs.is_empty() {
            return false;
        }
        runs = next_runs;
    }

    runs.iter().any(|run| {
        run.loop_state == Some(run.state) && run.accepted.iter().all(|accepted| *accepted)
    })
}

fn record_state_acceptance(gba: &Gba, state: GbaStateId, accepted: &mut [bool]) {
    for (index, accept_set) in gba.acceptance.iter().enumerate() {
        if accept_set.contains(&state) {
            accepted[index] = true;
        }
    }
}

fn record_edge_acceptance(transition: &GbaTransition, accepted: &mut [bool]) {
    for (index, edge_accepts) in transition.edge_accept.iter().copied().enumerate() {
        if edge_accepts && index < accepted.len() {
            accepted[index] = true;
        }
    }
}

/// Reference per-atom guard oracle. The Büchi-product hot path now uses the
/// precomputed `TransitionMasks`/`step_masks` instead (pure caching), but this
/// function is retained as the differential-test oracle that pins the cached
/// guard to byte-identical results. Dead in non-test builds by design.
#[cfg_attr(not(test), allow(dead_code))]
fn guard_satisfied_at_marking(
    transition: &GbaTransition,
    atoms: &[ResolvedPredicate],
    marking: &[u64],
    net: &PetriNet,
) -> bool {
    transition.pos_atoms.iter().all(|&atom| {
        atoms
            .get(atom)
            .is_some_and(|predicate| eval_predicate(predicate, marking, net))
    }) && transition.neg_atoms.iter().all(|&atom| {
        atoms
            .get(atom)
            .is_some_and(|predicate| !eval_predicate(predicate, marking, net))
    })
}

fn guard_expr(
    transition: &GbaTransition,
    step: usize,
    atoms: &[ResolvedPredicate],
    net: &PetriNet,
) -> String {
    let mut parts = Vec::with_capacity(transition.pos_atoms.len() + transition.neg_atoms.len());
    for &atom in &transition.pos_atoms {
        let Some(predicate) = atoms.get(atom) else {
            return "false".to_string();
        };
        parts.push(encode_predicate(predicate, step, net));
    }
    for &atom in &transition.neg_atoms {
        let Some(predicate) = atoms.get(atom) else {
            return "false".to_string();
        };
        parts.push(format!("(not {})", encode_predicate(predicate, step, net)));
    }
    and_expr(parts)
}

fn append_assert(script: &mut String, expr: String) {
    script.push_str(&format!("(assert {})\n", expr));
}

fn exactly_one_expr(vars: Vec<String>) -> String {
    if vars.is_empty() {
        return "false".to_string();
    }

    let mut parts = Vec::with_capacity(1 + vars.len().saturating_mul(vars.len()) / 2);
    parts.push(or_expr(vars.clone()));
    for left in 0..vars.len() {
        for right in left + 1..vars.len() {
            parts.push(format!("(not (and {} {}))", vars[left], vars[right]));
        }
    }
    and_expr(parts)
}

fn and_expr(parts: Vec<String>) -> String {
    match parts.len() {
        0 => "true".to_string(),
        1 => parts.into_iter().next().expect("one part"),
        _ => format!("(and {})", parts.join(" ")),
    }
}

fn or_expr(parts: Vec<String>) -> String {
    match parts.len() {
        0 => "false".to_string(),
        1 => parts.into_iter().next().expect("one part"),
        _ => format!("(or {})", parts.join(" ")),
    }
}

fn gba_state_var(step: usize, state: GbaStateId) -> String {
    format!("ltlq_{step}_{state}")
}

fn loop_var(loop_start: usize) -> String {
    format!("ltlloop_{loop_start}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    #[cfg(unix)]
    use std::os::unix::fs::PermissionsExt;
    use std::path::{Path, PathBuf};
    use std::time::{Duration, Instant};

    use tempfile::TempDir;

    use crate::petri_net::{Arc, PlaceIdx, PlaceInfo, TransitionIdx, TransitionInfo};
    use crate::resolved_predicate::ResolvedIntExpr;

    fn arc(place: u32, weight: u64) -> Arc {
        Arc {
            place: PlaceIdx(place),
            weight,
        }
    }

    fn place(id: &str) -> PlaceInfo {
        PlaceInfo {
            id: id.to_string(),
            name: None,
        }
    }

    fn transition(id: &str, inputs: Vec<Arc>, outputs: Vec<Arc>) -> TransitionInfo {
        TransitionInfo {
            id: id.to_string(),
            name: None,
            inputs,
            outputs,
        }
    }

    fn cyclic_net() -> PetriNet {
        PetriNet {
            name: Some("ltl-lasso-bmc-cycle".to_string()),
            places: vec![place("p0"), place("p1")],
            transitions: vec![
                transition("t0", vec![arc(0, 1)], vec![arc(1, 1)]),
                transition("t1", vec![arc(1, 1)], vec![arc(0, 1)]),
            ],
            initial_marking: vec![1, 0],
        }
    }

    /// Net that fires t0 (p0 -> p1) once and then deadlocks at `[0, 1]`.
    ///
    /// At `[0, 1]` no transition is enabled, so a `stay` self-loop there is a
    /// genuine deadlock stutter. Used to exercise a sound (genuine) accepting
    /// lasso for `G (p0 >= 1)` whose counterexample stutters only at a deadlock.
    fn deadlocking_net() -> PetriNet {
        PetriNet {
            name: Some("ltl-lasso-bmc-deadlock".to_string()),
            places: vec![place("p0"), place("p1")],
            transitions: vec![transition("t0", vec![arc(0, 1)], vec![arc(1, 1)])],
            initial_marking: vec![1, 0],
        }
    }

    /// Net mirroring the constructed soundness repro: places `p_a`, `p_b`,
    /// initial `[1, 0]`, single transition `t0: p_a -> p_b`. Reaches a genuine
    /// deadlock at `[0, 1]`. The repro property is `F G p` with `p = (p_b >= 1)`,
    /// which is TRUE — so NO accepting lasso witnessing its negation may exist.
    fn repro_net() -> PetriNet {
        PetriNet {
            name: Some("ltl-lasso-bmc-repro".to_string()),
            places: vec![place("p_a"), place("p_b")],
            transitions: vec![transition("t0", vec![arc(0, 1)], vec![arc(1, 1)])],
            initial_marking: vec![1, 0],
        }
    }

    fn pb_ge_one() -> ResolvedPredicate {
        ResolvedPredicate::IntLe(
            ResolvedIntExpr::Constant(1),
            ResolvedIntExpr::TokensCount(vec![PlaceIdx(1)]),
        )
    }

    fn p0_ge_one() -> ResolvedPredicate {
        ResolvedPredicate::IntLe(
            ResolvedIntExpr::Constant(1),
            ResolvedIntExpr::TokensCount(vec![PlaceIdx(0)]),
        )
    }

    fn globally_p0_ge_one() -> LtlNnf {
        LtlNnf::Release(Box::new(LtlNnf::False), Box::new(LtlNnf::Atom(0)))
    }

    fn globally_finally_p0_ge_one() -> LtlNnf {
        LtlNnf::Release(
            Box::new(LtlNnf::False),
            Box::new(LtlNnf::Until(
                Box::new(LtlNnf::True),
                Box::new(LtlNnf::Atom(0)),
            )),
        )
    }

    fn write_fake_solver_script(dir: &Path, name: &str, body: &str) -> PathBuf {
        let path = dir.join(name);
        let script = format!("#!/bin/sh\nset -eu\n{body}\n");
        fs::write(&path, script).expect("failed to write fake solver script");
        #[cfg(unix)]
        {
            let mut perms = fs::metadata(&path)
                .expect("script metadata should exist")
                .permissions();
            perms.set_mode(0o755);
            fs::set_permissions(&path, perms).expect("failed to mark fake solver executable");
        }
        path
    }

    fn with_ay_path<T>(path: &Path, f: impl FnOnce() -> T) -> T {
        let _guard = crate::examinations::smt_encoding::ay_env_lock();
        let previous = std::env::var("AY_PATH").ok();
        crate::env_guard::set_var("AY_PATH", path);
        let result = f();
        match previous {
            Some(value) => crate::env_guard::set_var("AY_PATH", value),
            None => crate::env_guard::remove_var("AY_PATH"),
        }
        result
    }

    fn fake_lasso_ay_for_depth2_model(tempdir: &TempDir, name: &str, model: &str) -> PathBuf {
        let input_path = tempdir.path().join(format!("{name}.smt2"));
        write_fake_solver_script(
            tempdir.path(),
            name,
            &format!(
                "input=\"{}\"\n\
                 cat > \"$input\"\n\
                 if grep -Fq '(get-value' \"$input\"; then\n\
                 \tprintf 'sat\\n'\n\
                 \tcat <<'MODEL'\n\
                 {}\
                 MODEL\n\
                 \texit 0\n\
                 fi\n\
                 if grep -Fq 'm_2_' \"$input\"; then\n\
                 \tprintf 'sat\\n'\n\
                 else\n\
                 \tprintf 'unsat\\n'\n\
                 fi",
                input_path.display(),
                model
            ),
        )
    }

    /// Genuine depth-2 lasso for the single-transition `deadlocking_net`:
    /// fire t0 (`[1,0]` -> `[0,1]`) then stutter at the genuine deadlock `[0,1]`.
    /// This is a sound accepting lasso for `G (p0 >= 1)` (atom0 becomes false and
    /// the system genuinely deadlocks), so it must still be accepted.
    fn valid_depth2_lasso_model() -> &'static str {
        "((stay_0 false)\n\
          (fire_0_0 true)\n\
          (stay_1 true)\n\
          (fire_1_0 false))\n"
    }

    /// Depth-2 model that stutters from the very start (`[1,0]`, which is NOT a
    /// deadlock — t0 is enabled). With sound deadlock-only stutter this never
    /// closes an accepting lasso, so the replay validator must reject it.
    fn non_counterexample_depth2_model() -> &'static str {
        "((stay_0 true)\n\
          (fire_0_0 false)\n\
          (stay_1 true)\n\
          (fire_1_0 false))\n"
    }

    #[test]
    fn lasso_declines_counterexample_release_obligations() {
        let net = cyclic_net();
        let atoms = vec![p0_ge_one()];
        let tempdir = TempDir::new().expect("tempdir should create");
        let calls_path = tempdir.path().join("calls.log");
        let solver = write_fake_solver_script(
            tempdir.path(),
            "fake-ay-ltl-release-fallback",
            &format!(
                "printf 'called\\n' >> \"{}\"\n\
                 cat >/dev/null\n\
                 printf 'sat\\n'",
                calls_path.display()
            ),
        );

        assert!(ltl_counterexample_contains_release(
            &globally_finally_p0_ge_one()
        ));
        let result = with_ay_path(&solver, || {
            find_ltl_lasso_counterexample(&net, &globally_finally_p0_ge_one(), &atoms, None)
        });
        assert_eq!(result, None);
        assert!(
            !calls_path.exists(),
            "Release counterexample automata must fall through without invoking lasso BMC"
        );
    }

    #[test]
    fn lasso_script_contains_loop_gba_and_bmc_step_symbols() {
        let net = cyclic_net();
        let atoms = vec![p0_ge_one()];
        let gba = build_ltl_counterexample_gba(&globally_p0_ge_one());
        let script = encode_lasso_check_script(&net, &gba, &atoms, 2);

        assert!(script.contains("(declare-const ltlloop_0 Bool)"));
        assert!(script.contains("(declare-const ltlloop_1 Bool)"));
        assert!(script.contains("(declare-const ltlq_0_0 Bool)"));
        assert!(script.contains("(= m_2_0 m_0_0)"));
        assert!(script.contains("fire_0_0"));
        assert!(script.contains("(check-sat)"));
    }

    #[test]
    fn lasso_sat_model_replays_to_counterexample_witness() {
        // Genuine FALSE preserved: a real deadlock self-loop at `[0,1]` (where
        // atom0 `p0 >= 1` is false forever) is a sound accepting lasso for
        // `G (p0 >= 1)` and must still yield a witness under deadlock-only stutter.
        let net = deadlocking_net();
        let atoms = vec![p0_ge_one()];
        let tempdir = TempDir::new().expect("tempdir should create");
        let solver = fake_lasso_ay_for_depth2_model(
            &tempdir,
            "fake-ay-ltl-valid-lasso",
            valid_depth2_lasso_model(),
        );

        let result = with_ay_path(&solver, || {
            find_ltl_lasso_counterexample(
                &net,
                &globally_p0_ge_one(),
                &atoms,
                Some(Instant::now() + Duration::from_secs(5)),
            )
        });

        assert_eq!(result, Some(LtlLassoBmcWitness { depth: 2 }));
    }

    #[test]
    fn lasso_sat_status_with_non_counterexample_model_falls_through() {
        let net = deadlocking_net();
        let atoms = vec![p0_ge_one()];
        let tempdir = TempDir::new().expect("tempdir should create");
        let solver = fake_lasso_ay_for_depth2_model(
            &tempdir,
            "fake-ay-ltl-invalid-lasso",
            non_counterexample_depth2_model(),
        );

        let result = with_ay_path(&solver, || {
            find_ltl_lasso_counterexample(
                &net,
                &globally_p0_ge_one(),
                &atoms,
                Some(Instant::now() + Duration::from_secs(5)),
            )
        });

        assert_eq!(
            result, None,
            "SAT status must not produce FALSE without a replay-validated lasso"
        );
    }

    #[test]
    fn local_validation_finds_accepting_lasso_for_replayed_g_p0_counterexample() {
        let net = cyclic_net();
        let atoms = vec![p0_ge_one()];
        let gba = build_ltl_counterexample_gba(&globally_p0_ge_one());
        let markings = vec![vec![1, 0], vec![0, 1], vec![0, 1]];

        assert!(accepting_lasso_exists(&gba, &atoms, &net, &markings));
    }

    #[test]
    fn local_validation_requires_marking_loop() {
        let net = cyclic_net();
        let atoms = vec![p0_ge_one()];
        let gba = build_ltl_counterexample_gba(&globally_p0_ge_one());
        let non_loop_markings = vec![vec![1, 0], vec![0, 1]];

        assert!(!accepting_lasso_exists(
            &gba,
            &atoms,
            &net,
            &non_loop_markings
        ));
    }

    #[test]
    fn replay_rejects_disabled_transition() {
        let net = cyclic_net();
        let trace = vec![LtlBmcTraceStep::Fire(TransitionIdx(1))];

        assert!(replay_lasso_trace(&net, &trace).is_err());
    }

    // --- Soundness regression: deadlock-only stutter for the LTL lasso BMC. ---

    /// Defense-in-depth validator: a `Stay` at the live initial marking `[1,0]`
    /// (t0 enabled) is the spurious depth-1 stutter that the OLD encoding admitted
    /// to witness `¬(F G p)` on a TRUE property. The replay validator must reject
    /// it independently of the SMT encoding.
    #[test]
    fn replay_rejects_stutter_at_non_deadlock_marking() {
        let net = repro_net();
        assert!(
            net.is_enabled(&net.initial_marking, TransitionIdx(0)),
            "t0 must be enabled at the live initial marking"
        );
        let spurious = vec![LtlBmcTraceStep::Stay];

        let result = replay_lasso_trace(&net, &spurious);
        assert!(
            result.is_err(),
            "a stutter at a non-deadlock marking must be rejected, got {result:?}"
        );
    }

    /// A `Stay` at a genuine deadlock (`[0,1]` after firing t0, where nothing is
    /// enabled) is sound and must be accepted by the replay validator. Genuine
    /// FALSE verdicts depend on this self-loop being allowed.
    #[test]
    fn replay_accepts_stutter_at_genuine_deadlock() {
        let net = repro_net();
        let genuine = vec![
            LtlBmcTraceStep::Fire(TransitionIdx(0)),
            LtlBmcTraceStep::Stay,
        ];

        let markings =
            replay_lasso_trace(&net, &genuine).expect("genuine deadlock stutter is sound");
        assert_eq!(markings, vec![vec![1, 0], vec![0, 1], vec![0, 1]]);
        assert!(
            (0..net.num_transitions())
                .all(|t| !net.is_enabled(&markings[2], TransitionIdx(t as u32))),
            "the stutter marking must be a genuine deadlock"
        );
    }

    /// The lasso preamble must constrain stutter to deadlock states: for every
    /// step it asserts `(=> stay_step (not <t0 enabled at step>))`. On the repro
    /// net `t0` consumes `p_a` (place 0), so the guard is `(>= m_step_0 1)`.
    #[test]
    fn lasso_preamble_constrains_stutter_to_deadlock() {
        let net = repro_net();
        let gba = build_ltl_counterexample_gba(&globally_p0_ge_one());
        let script = encode_lasso_check_script(&net, &gba, &net_atoms(), 2);

        // Step 0 deadlock-stutter guard (only transition t0 needs p_a >= 1).
        assert!(
            script.contains("(assert (=> stay_0 (not (>= m_0_0 1))))"),
            "lasso preamble must forbid stutter while t0 is enabled at step 0;\n{script}"
        );
        assert!(
            script.contains("(assert (=> stay_1 (not (>= m_1_0 1))))"),
            "lasso preamble must forbid stutter while t0 is enabled at step 1;\n{script}"
        );
    }

    /// The shared `emit_bmc_preamble` (reachability BMC / k-induction) must keep
    /// FREE stutter: it must NOT emit any deadlock-stutter constraint. This pins
    /// the #1 invariant — non-lasso callers stay byte-identical.
    #[test]
    fn non_lasso_preamble_keeps_free_stutter() {
        use crate::examinations::bmc_runner::{
            emit_bmc_preamble, emit_bmc_preamble_deadlock_stutter,
        };
        let net = repro_net();

        let mut free = String::new();
        emit_bmc_preamble(&mut free, &net, 2);
        assert!(
            !free.contains("(=> stay_0 (not"),
            "reachability/k-induction preamble must keep free stutter; got:\n{free}"
        );

        let mut lasso = String::new();
        emit_bmc_preamble_deadlock_stutter(&mut lasso, &net, 2);
        assert!(
            lasso.contains("(=> stay_0 (not (>= m_0_0 1)))"),
            "lasso preamble must add the deadlock-stutter constraint; got:\n{lasso}"
        );

        // The lasso preamble is the free preamble plus the extra stutter guards:
        // stripping those guards must recover the free preamble byte-for-byte.
        let stripped: String = lasso
            .lines()
            .filter(|line| !line.starts_with("(assert (=> stay_") || !line.contains("(not "))
            .map(|line| format!("{line}\n"))
            .collect();
        assert_eq!(
            stripped, free,
            "deadlock-stutter constraints must be the ONLY difference vs the free preamble"
        );
    }

    fn net_atoms() -> Vec<ResolvedPredicate> {
        vec![pb_ge_one()]
    }

    // --- Pure-caching differential: memoized mask guard == per-atom oracle. ---

    /// Net with enough places to give each atom an independently toggleable
    /// truth value. Atom `i` is `place_i >= 1`, so a marking is just a bit-vector
    /// over atom truths. The net's transitions are irrelevant to guard
    /// evaluation; we only ever read markings.
    fn masks_diff_net(num_places: usize) -> PetriNet {
        PetriNet {
            name: Some("ltl-lasso-bmc-masks-diff".to_string()),
            places: (0..num_places).map(|i| place(&format!("p{i}"))).collect(),
            transitions: vec![transition("t0", vec![arc(0, 1)], vec![arc(1, 1)])],
            initial_marking: vec![0; num_places],
        }
    }

    fn atom_ge_one(place_index: u32) -> ResolvedPredicate {
        ResolvedPredicate::IntLe(
            ResolvedIntExpr::Constant(1),
            ResolvedIntExpr::TokensCount(vec![PlaceIdx(place_index)]),
        )
    }

    fn gba_transition(pos: Vec<usize>, neg: Vec<usize>) -> GbaTransition {
        GbaTransition {
            pos_atoms: pos,
            neg_atoms: neg,
            successor: 0,
            edge_accept: Vec::new(),
        }
    }

    /// PURE-CACHING CONTRACT: the precomputed pos/neg-mask guard check must
    /// return the SAME boolean as the original per-atom `guard_satisfied_at_marking`
    /// oracle for a battery of (GbaTransition, atoms, marking) cases — including
    /// out-of-bounds atom indices and pos/neg overlap, which are the corner cases
    /// where a naive mask check would diverge from `atoms.get(..).is_some_and(..)`.
    #[test]
    fn memoized_guard_matches_per_atom_oracle_battery() {
        let num_atoms = 4usize;
        // 1 extra place so an in-bounds atom index can read a place that has no
        // matching atom; the net just needs >= num_atoms places here.
        let net = masks_diff_net(num_atoms + 1);
        let atoms: Vec<ResolvedPredicate> = (0..num_atoms as u32).map(atom_ge_one).collect();

        // A representative battery of transitions: empty guard, single pos,
        // single neg, multi pos+neg, pos/neg overlap on the same atom, and
        // out-of-bounds references (index == num_atoms) on both pos and neg.
        let transitions = vec![
            gba_transition(vec![], vec![]),
            gba_transition(vec![0], vec![]),
            gba_transition(vec![], vec![0]),
            gba_transition(vec![0, 1], vec![2, 3]),
            gba_transition(vec![1, 2], vec![3]),
            gba_transition(vec![0], vec![0]), // contradictory: same atom pos+neg
            gba_transition(vec![1, 3], vec![0, 2]),
            gba_transition(vec![num_atoms], vec![]), // out-of-bounds positive
            gba_transition(vec![], vec![num_atoms]), // out-of-bounds negative
            gba_transition(vec![0, num_atoms], vec![1]), // mixed OOB positive
            gba_transition(vec![0], vec![1, num_atoms]), // mixed OOB negative
        ];

        let mut total_cases = 0usize;
        // Enumerate every truth assignment over the `num_atoms` atoms; a marking
        // sets place_i to 1 iff atom_i should be true.
        for assignment in 0u32..(1 << num_atoms) {
            let mut marking = vec![0u64; net.num_places()];
            for atom in 0..num_atoms {
                if assignment & (1 << atom) != 0 {
                    marking[atom] = 1;
                }
            }
            let step_mask = {
                let masks = build_step_masks(&atoms, &net, std::slice::from_ref(&marking));
                masks.into_iter().next().expect("one step mask")
            };

            for transition in &transitions {
                let oracle = guard_satisfied_at_marking(transition, &atoms, &marking, &net);
                let memoized = TransitionMasks::from_transition(transition, atoms.len())
                    .satisfied_by(&step_mask);
                assert_eq!(
                    oracle, memoized,
                    "memoized mask guard diverged from per-atom oracle for transition \
                     pos={:?} neg={:?} at marking {:?}",
                    transition.pos_atoms, transition.neg_atoms, marking
                );
                total_cases += 1;
            }
        }

        assert_eq!(
            total_cases,
            transitions.len() * (1 << num_atoms),
            "battery must cover every (transition, assignment) pair"
        );
    }

    /// Whole-trace differential: the hoisted `accepting_lasso_exists` (which now
    /// builds the step masks once and indexes precomputed transition masks) must
    /// agree with a fully independent reference that re-evaluates every guard via
    /// the per-atom oracle for every loop_start — i.e. the exact pre-caching
    /// algorithm. Run over several real GBAs and traces.
    #[test]
    fn accepting_lasso_exists_matches_uncached_reference() {
        let net = cyclic_net();
        let atoms = vec![p0_ge_one()];
        let gbas = [
            build_ltl_counterexample_gba(&globally_p0_ge_one()),
            build_ltl_counterexample_gba(&globally_finally_p0_ge_one()),
        ];
        let traces: Vec<Vec<Vec<u64>>> = vec![
            vec![vec![1, 0], vec![0, 1], vec![0, 1]],
            vec![vec![1, 0], vec![0, 1]],
            vec![vec![1, 0], vec![0, 1], vec![1, 0], vec![0, 1]],
            vec![vec![1, 0], vec![1, 0]],
            vec![vec![1, 0]],
            vec![vec![0, 1], vec![1, 0], vec![0, 1], vec![1, 0], vec![0, 1]],
        ];

        for gba in &gbas {
            for markings in &traces {
                let cached = accepting_lasso_exists(gba, &atoms, &net, markings);
                let reference = uncached_accepting_lasso_exists(gba, &atoms, &net, markings);
                assert_eq!(
                    cached, reference,
                    "cached accepting_lasso_exists diverged from uncached reference on trace {markings:?}"
                );
            }
        }
    }

    /// Pre-caching reference: byte-for-byte the original algorithm with the
    /// per-atom `guard_satisfied_at_marking` oracle and NO hoisting/memoization.
    fn uncached_accepting_lasso_exists(
        gba: &Gba,
        atoms: &[ResolvedPredicate],
        net: &PetriNet,
        markings: &[Vec<u64>],
    ) -> bool {
        if markings.len() < 2 {
            return false;
        }
        let depth = markings.len() - 1;
        (0..depth).any(|loop_start| {
            markings[loop_start] == markings[depth]
                && uncached_for_loop(gba, atoms, net, markings, loop_start)
        })
    }

    fn uncached_for_loop(
        gba: &Gba,
        atoms: &[ResolvedPredicate],
        net: &PetriNet,
        markings: &[Vec<u64>],
        loop_start: usize,
    ) -> bool {
        let depth = markings.len() - 1;
        let num_accept = gba.acceptance.len();
        let mut runs = HashSet::new();
        for transition in &gba.initial_transitions {
            if !guard_satisfied_at_marking(transition, atoms, &markings[0], net) {
                continue;
            }
            let mut accepted = vec![false; num_accept];
            let mut loop_state = None;
            if loop_start == 0 {
                loop_state = Some(transition.successor);
                record_state_acceptance(gba, transition.successor, &mut accepted);
            }
            runs.insert(RunState {
                state: transition.successor,
                loop_state,
                accepted,
            });
        }
        for step in 0..depth {
            let mut next_runs = HashSet::new();
            for run in &runs {
                let Some(transitions) = gba.transitions.get(run.state as usize) else {
                    continue;
                };
                for transition in transitions {
                    if !guard_satisfied_at_marking(transition, atoms, &markings[step + 1], net) {
                        continue;
                    }
                    let mut accepted = run.accepted.clone();
                    if step >= loop_start {
                        record_edge_acceptance(transition, &mut accepted);
                    }
                    let mut loop_state = run.loop_state;
                    let successor = transition.successor;
                    if step + 1 == loop_start {
                        loop_state = Some(successor);
                        record_state_acceptance(gba, successor, &mut accepted);
                    } else if step + 1 > loop_start && step + 1 < depth {
                        record_state_acceptance(gba, successor, &mut accepted);
                    }
                    next_runs.insert(RunState {
                        state: successor,
                        loop_state,
                        accepted,
                    });
                }
            }
            if next_runs.is_empty() {
                return false;
            }
            runs = next_runs;
        }
        runs.iter().any(|run| {
            run.loop_state == Some(run.state) && run.accepted.iter().all(|accepted| *accepted)
        })
    }

    /// Before/after micro-measurement of guard *atom evaluations* on a
    /// non-trivial trace. The pre-caching path evaluates one atom per
    /// (loop_start, step, candidate-transition, atom); the cached path evaluates
    /// each (step, atom) exactly once. This pins the eliminated blowup and shows
    /// the speedup is real, not just refactored away.
    #[test]
    fn guard_evaluation_count_before_after_measurement() {
        // Two scenarios, each a non-trivial deep lasso trace. We report the
        // before/after guard *atom evaluation* counts and assert pure-caching
        // identity of the verdict in both.
        let scenarios: [(&str, Gba); 2] = [
            ("G p0", build_ltl_counterexample_gba(&globally_p0_ge_one())),
            (
                "G F p0",
                build_ltl_counterexample_gba(&globally_finally_p0_ge_one()),
            ),
        ];
        let net = cyclic_net();
        let atoms = vec![p0_ge_one()];

        for (label, gba) in &scenarios {
            // A non-trivial trace: depth 16 lasso over the cyclic net.
            let mut markings = Vec::new();
            for step in 0..=16usize {
                markings.push(if step % 2 == 0 {
                    vec![1, 0]
                } else {
                    vec![0, 1]
                });
            }
            let depth = markings.len() - 1;

            // BEFORE: per-atom evaluations the uncached algorithm performs.
            let before = count_uncached_atom_evals(gba, &atoms, &net, &markings);
            // AFTER: the cached path evaluates each (step, atom) exactly once.
            let after = (depth + 1) * atoms.len();

            // Sanity: both compute the SAME verdict (pure caching).
            assert_eq!(
                accepting_lasso_exists(gba, &atoms, &net, &markings),
                uncached_accepting_lasso_exists(gba, &atoms, &net, &markings),
                "cached vs uncached verdict mismatch on scenario {label}"
            );

            assert!(
                before > after,
                "expected the uncached path to perform strictly more atom evals on {label} \
                 (before={before}, after={after})"
            );
            eprintln!(
                "[ltl_lasso_bmc guard-eval measurement] scenario={label} states={} depth={depth} \
                 atoms={} before(uncached per-atom evals)={before} after(cached evals)={after} \
                 reduction={:.1}x",
                gba.num_states,
                atoms.len(),
                before as f64 / after.max(1) as f64
            );
        }
    }

    /// Counts the number of `eval_predicate` calls the *uncached* guard path
    /// would perform across the whole `accepting_lasso_exists` computation,
    /// counting one eval per atom consulted by `guard_satisfied_at_marking`
    /// (with short-circuiting matching the oracle's `all(..)` semantics).
    fn count_uncached_atom_evals(
        gba: &Gba,
        atoms: &[ResolvedPredicate],
        net: &PetriNet,
        markings: &[Vec<u64>],
    ) -> usize {
        if markings.len() < 2 {
            return 0;
        }
        let depth = markings.len() - 1;
        let mut count = 0usize;
        for loop_start in 0..depth {
            if markings[loop_start] != markings[depth] {
                // The cached/uncached driver both skip via the marking-loop
                // guard *before* touching atoms, so no atom evals occur here.
                continue;
            }
            // initial transitions, evaluated at markings[0]
            for transition in &gba.initial_transitions {
                count += count_guard_atom_evals(transition, atoms, &markings[0], net);
            }
            // We over-count vs the run-pruned reachable set, but the BEFORE
            // figure is an upper bound on guard atom work; what matters for the
            // measurement is that it dwarfs the (depth+1)*|atoms| cached figure.
            for step in 0..depth {
                for source in 0..gba.num_states {
                    if let Some(transitions) = gba.transitions.get(source as usize) {
                        for transition in transitions {
                            count +=
                                count_guard_atom_evals(transition, atoms, &markings[step + 1], net);
                        }
                    }
                }
            }
        }
        count
    }

    /// Mirrors the short-circuiting atom-eval count of `guard_satisfied_at_marking`.
    fn count_guard_atom_evals(
        transition: &GbaTransition,
        atoms: &[ResolvedPredicate],
        marking: &[u64],
        net: &PetriNet,
    ) -> usize {
        let mut count = 0usize;
        for &atom in &transition.pos_atoms {
            count += 1;
            match atoms.get(atom) {
                Some(predicate) if eval_predicate(predicate, marking, net) => {}
                _ => return count, // `.all` short-circuits on first false
            }
        }
        for &atom in &transition.neg_atoms {
            count += 1;
            match atoms.get(atom) {
                Some(predicate) if !eval_predicate(predicate, marking, net) => {}
                _ => return count,
            }
        }
        count
    }
}
