// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Local AIG refactoring for distributive OR-of-cube shapes.
//!
//! This is a bounded `rf` slice: recognize the common AIG encoding of
//! `(a | b) & (a | c)` and replace it with `a | (b & c)` when the supporting
//! OR gates are not shared enough to erase the win. ABC's full refactor pass
//! considers larger cuts and many decompositions; this module starts with the
//! high-frequency distributive factor that DAG rewrite does not reliably see.

use rustc_hash::{FxHashMap, FxHashSet};

use crate::sat_types::{Lit, Var};
use crate::transys::Transys;

use super::substitution::{apply_substitution, fold_and, rebuild_trans_clauses, sorted_and_defs};

type ExistingDefs = FxHashMap<(u32, u32), Vec<Var>>;

const MAX_REFACTOR_ROUNDS: usize = 4;

fn compute_fanout(ts: &Transys) -> FxHashMap<Var, usize> {
    let mut fanout: FxHashMap<Var, usize> = FxHashMap::default();
    for &(rhs0, rhs1) in ts.and_defs.values() {
        *fanout.entry(rhs0.var()).or_insert(0) += 1;
        *fanout.entry(rhs1.var()).or_insert(0) += 1;
    }
    for &lit in &ts.bad_lits {
        *fanout.entry(lit.var()).or_insert(0) += 1;
    }
    for &lit in &ts.constraint_lits {
        *fanout.entry(lit.var()).or_insert(0) += 1;
    }
    for &lit in ts.next_state.values() {
        *fanout.entry(lit.var()).or_insert(0) += 1;
    }
    fanout
}

#[inline]
fn canonical_pair(lhs: Lit, rhs: Lit) -> (Lit, Lit) {
    if lhs.code() <= rhs.code() {
        (lhs, rhs)
    } else {
        (rhs, lhs)
    }
}

fn and_key(lhs: Lit, rhs: Lit) -> (u32, u32) {
    let (lhs, rhs) = canonical_pair(lhs, rhs);
    (lhs.code(), rhs.code())
}

fn negated_and_as_or(lit: Lit, and_defs: &FxHashMap<Var, (Lit, Lit)>) -> Option<(Lit, Lit)> {
    if lit.is_positive() {
        return None;
    }

    let &(lhs, rhs) = and_defs.get(&lit.var())?;
    Some(canonical_pair(!lhs, !rhs))
}

fn shared_or_factor(lhs: (Lit, Lit), rhs: (Lit, Lit)) -> Option<(Lit, Lit, Lit)> {
    let (lhs0, lhs1) = lhs;
    let (rhs0, rhs1) = rhs;

    if lhs0 == rhs0 {
        Some((lhs0, lhs1, rhs1))
    } else if lhs0 == rhs1 {
        Some((lhs0, lhs1, rhs0))
    } else if lhs1 == rhs0 {
        Some((lhs1, lhs0, rhs1))
    } else if lhs1 == rhs1 {
        Some((lhs1, lhs0, rhs0))
    } else {
        None
    }
}

fn push_and_gate(
    lhs: Lit,
    rhs: Lit,
    next_var: &mut u32,
    new_and_defs: &mut Vec<(Var, Lit, Lit)>,
    existing_defs: &mut ExistingDefs,
    forbidden_reuse: &FxHashSet<Var>,
) -> Lit {
    if let Some(folded) = fold_and(lhs, rhs) {
        return folded;
    }

    let key = and_key(lhs, rhs);
    if let Some(existing) = existing_defs.get(&key) {
        if let Some(&existing) = existing
            .iter()
            .find(|&&candidate| !forbidden_reuse.contains(&candidate))
        {
            return Lit::pos(existing);
        }
    }

    let out = Var(*next_var);
    *next_var += 1;
    let (lhs, rhs) = canonical_pair(lhs, rhs);
    new_and_defs.push((out, lhs, rhs));
    existing_defs.entry(key).or_default().push(out);
    Lit::pos(out)
}

fn build_factored_or(
    shared: Lit,
    lhs_only: Lit,
    rhs_only: Lit,
    next_var: &mut u32,
    existing_defs: &mut ExistingDefs,
    forbidden_reuse: &FxHashSet<Var>,
) -> (Lit, Vec<(Var, Lit, Lit)>) {
    let mut new_and_defs = Vec::new();
    let inner = push_and_gate(
        lhs_only,
        rhs_only,
        next_var,
        &mut new_and_defs,
        existing_defs,
        forbidden_reuse,
    );

    let root = if shared == Lit::TRUE || inner == Lit::TRUE {
        Lit::TRUE
    } else if shared == Lit::FALSE {
        inner
    } else if inner == Lit::FALSE {
        shared
    } else if shared == inner {
        shared
    } else if shared == !inner {
        Lit::TRUE
    } else {
        !push_and_gate(
            !shared,
            !inner,
            next_var,
            &mut new_and_defs,
            existing_defs,
            forbidden_reuse,
        )
    };

    (root, new_and_defs)
}

fn refactor_candidate(
    rhs0: Lit,
    rhs1: Lit,
    and_defs: &FxHashMap<Var, (Lit, Lit)>,
) -> Option<(Lit, Lit, Lit)> {
    if rhs0.var() == rhs1.var() {
        return None;
    }

    let lhs_or = negated_and_as_or(rhs0, and_defs)?;
    let rhs_or = negated_and_as_or(rhs1, and_defs)?;
    shared_or_factor(lhs_or, rhs_or)
}

/// Apply one bounded AIG refactoring round.
///
/// Returns the refactored transition system and estimated eliminated gate
/// count. The estimate only credits the root plus unshared support gates that
/// become dead after substitution.
fn refactor_once(ts: &Transys) -> (Transys, usize) {
    if ts.and_defs.len() < 3 {
        return (ts.clone(), 0);
    }

    let fanout = compute_fanout(ts);
    let mut subst: FxHashMap<Var, Lit> = FxHashMap::default();
    let mut new_and_defs = Vec::new();
    let mut next_var = ts.max_var + 1;
    let mut eliminated = 0usize;
    let mut existing_defs: ExistingDefs = FxHashMap::default();
    for (out, rhs0, rhs1) in sorted_and_defs(ts) {
        existing_defs
            .entry(and_key(rhs0, rhs1))
            .or_default()
            .push(out);
    }

    for (out, rhs0, rhs1) in sorted_and_defs(ts) {
        if subst.contains_key(&out) {
            continue;
        }

        let Some((shared, lhs_only, rhs_only)) = refactor_candidate(rhs0, rhs1, &ts.and_defs)
        else {
            continue;
        };

        let orig_gates = 1
            + usize::from(fanout.get(&rhs0.var()).copied().unwrap_or(0) <= 1)
            + usize::from(fanout.get(&rhs1.var()).copied().unwrap_or(0) <= 1);
        let mut forbidden_reuse = FxHashSet::default();
        forbidden_reuse.insert(out);
        if fanout.get(&rhs0.var()).copied().unwrap_or(0) <= 1 {
            forbidden_reuse.insert(rhs0.var());
        }
        if fanout.get(&rhs1.var()).copied().unwrap_or(0) <= 1 {
            forbidden_reuse.insert(rhs1.var());
        }

        let trial_next_var = next_var;
        let mut trial_existing_defs = existing_defs.clone();
        let (replacement, candidate_defs) = build_factored_or(
            shared,
            lhs_only,
            rhs_only,
            &mut next_var,
            &mut trial_existing_defs,
            &forbidden_reuse,
        );
        let new_gates = candidate_defs.len();
        if new_gates >= orig_gates {
            next_var = trial_next_var;
            continue;
        }

        eliminated += orig_gates - new_gates;
        existing_defs = trial_existing_defs;
        new_and_defs.extend(candidate_defs);
        subst.insert(out, replacement);
    }

    if subst.is_empty() {
        return (ts.clone(), 0);
    }

    let mut result = ts.clone();
    result.max_var = next_var.saturating_sub(1).max(ts.max_var);
    for (out, lhs, rhs) in &new_and_defs {
        result.and_defs.insert(*out, (*lhs, *rhs));
    }
    result.trans_clauses = rebuild_trans_clauses(&result.and_defs);

    (apply_substitution(&result, &subst), eliminated)
}

/// Apply bounded AIG refactoring to a small fixed point.
///
/// Chained distributive opportunities often expose the next candidate only
/// after the previous substitution is cleaned up. Limit the number of rounds so
/// this stays a local `rf` slice rather than a full cut-enumerating refactor.
pub(crate) fn refactor(ts: &Transys) -> (Transys, usize) {
    let mut current = ts.clone();
    let mut total_eliminated = 0usize;

    for _ in 0..MAX_REFACTOR_ROUNDS {
        let before_gates = current.and_defs.len();
        let (next, eliminated) = refactor_once(&current);
        if eliminated == 0 {
            break;
        }

        total_eliminated += eliminated;
        current = next;

        if current.and_defs.len() >= before_gates {
            break;
        }
    }

    (current, total_eliminated)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::preprocess::substitution::rebuild_trans_clauses;
    use crate::preprocess::{preprocess_with_config, PreprocessConfig};
    use crate::sat_types::Clause;

    fn build_ts(
        max_var: u32,
        input_vars: Vec<Var>,
        bad_lits: Vec<Lit>,
        constraint_lits: Vec<Lit>,
        and_defs: FxHashMap<Var, (Lit, Lit)>,
    ) -> Transys {
        Transys {
            max_var,
            num_latches: 0,
            num_inputs: input_vars.len(),
            latch_vars: Vec::new(),
            input_vars,
            next_state: FxHashMap::default(),
            init_clauses: Vec::<Clause>::new(),
            trans_clauses: rebuild_trans_clauses(&and_defs),
            bad_lits,
            constraint_lits,
            and_defs,
            internal_signals: Vec::new(),
        }
    }

    fn eval_lit(
        lit: Lit,
        assignment: &FxHashMap<Var, bool>,
        and_defs: &FxHashMap<Var, (Lit, Lit)>,
    ) -> bool {
        if lit == Lit::FALSE {
            return false;
        }
        if lit == Lit::TRUE {
            return true;
        }

        let value = if let Some(&(lhs, rhs)) = and_defs.get(&lit.var()) {
            eval_lit(lhs, assignment, and_defs) && eval_lit(rhs, assignment, and_defs)
        } else {
            *assignment.get(&lit.var()).unwrap_or(&false)
        };

        if lit.is_negated() {
            !value
        } else {
            value
        }
    }

    #[test]
    fn test_refactor_factors_shared_or_literal() {
        // root = (a | b) & (a | c), encoded as:
        // n_ab = !a & !b; n_ac = !a & !c; root = !n_ab & !n_ac.
        // Refactor to a | (b & c), which needs two AND gates.
        let a = Var(1);
        let b = Var(2);
        let c = Var(3);
        let n_ab = Var(4);
        let n_ac = Var(5);
        let root = Var(6);

        let mut and_defs = FxHashMap::default();
        and_defs.insert(n_ab, (Lit::neg(a), Lit::neg(b)));
        and_defs.insert(n_ac, (Lit::neg(a), Lit::neg(c)));
        and_defs.insert(root, (Lit::neg(n_ab), Lit::neg(n_ac)));

        let ts = build_ts(6, vec![a, b, c], vec![Lit::pos(root)], Vec::new(), and_defs);

        let (result, eliminated) = refactor(&ts);
        assert_eq!(eliminated, 1);
        assert_eq!(result.and_defs.len(), 2);

        for mask in 0..8 {
            let mut assignment = FxHashMap::default();
            assignment.insert(a, (mask & 1) != 0);
            assignment.insert(b, (mask & 2) != 0);
            assignment.insert(c, (mask & 4) != 0);

            let before = eval_lit(ts.bad_lits[0], &assignment, &ts.and_defs);
            let after = eval_lit(result.bad_lits[0], &assignment, &result.and_defs);
            assert_eq!(before, after, "assignment mask {mask}");
        }
    }

    #[test]
    fn test_refactor_reuses_existing_factored_inner_gate() {
        // root = (a | b) & (a | c), with an existing live bc = b & c.
        // Refactor should reuse bc and only add the OR encoding for a | bc.
        let a = Var(1);
        let b = Var(2);
        let c = Var(3);
        let n_ab = Var(4);
        let n_ac = Var(5);
        let bc = Var(6);
        let root = Var(7);

        let mut and_defs = FxHashMap::default();
        and_defs.insert(n_ab, (Lit::neg(a), Lit::neg(b)));
        and_defs.insert(n_ac, (Lit::neg(a), Lit::neg(c)));
        and_defs.insert(bc, (Lit::pos(b), Lit::pos(c)));
        and_defs.insert(root, (Lit::neg(n_ab), Lit::neg(n_ac)));

        let ts = build_ts(
            7,
            vec![a, b, c],
            vec![Lit::pos(root)],
            vec![Lit::pos(bc)],
            and_defs,
        );

        let (result, eliminated) = refactor(&ts);
        assert_eq!(eliminated, 2);
        assert_eq!(result.and_defs.len(), 2);
        assert_eq!(
            result
                .and_defs
                .values()
                .filter(|&&(lhs, rhs)| and_key(lhs, rhs) == and_key(Lit::pos(b), Lit::pos(c)))
                .count(),
            1,
            "refactor should reuse the existing b & c gate instead of duplicating it",
        );

        let config = PreprocessConfig {
            enable_scorr: false,
            enable_frts: false,
            enable_bve: false,
            enable_rewrite: false,
            enable_dag_rewrite: false,
            enable_synthesis: false,
            enable_ternary_sim: false,
            ..PreprocessConfig::default()
        };
        let (_pipeline_result, stats) = preprocess_with_config(&ts, &config);
        assert_eq!(stats.refactor_eliminated, 2);

        for mask in 0..8 {
            let mut assignment = FxHashMap::default();
            assignment.insert(a, (mask & 1) != 0);
            assignment.insert(b, (mask & 2) != 0);
            assignment.insert(c, (mask & 4) != 0);

            let before_bad = eval_lit(ts.bad_lits[0], &assignment, &ts.and_defs);
            let after_bad = eval_lit(result.bad_lits[0], &assignment, &result.and_defs);
            assert_eq!(before_bad, after_bad, "bad literal mask {mask}");

            let before_constraint = eval_lit(ts.constraint_lits[0], &assignment, &ts.and_defs);
            let after_constraint =
                eval_lit(result.constraint_lits[0], &assignment, &result.and_defs);
            assert_eq!(
                before_constraint, after_constraint,
                "constraint literal mask {mask}",
            );
        }
    }

    #[test]
    fn test_refactor_does_not_overcredit_reused_original_support_gate() {
        // root = (a | b) & (a | b), encoded with two duplicate support gates.
        // Reusing one original support gate for the replacement would preserve
        // semantics but over-credit both support gates as eliminated.
        let a = Var(1);
        let b = Var(2);
        let n_ab0 = Var(3);
        let n_ab1 = Var(4);
        let root = Var(5);

        let mut and_defs = FxHashMap::default();
        and_defs.insert(n_ab0, (Lit::neg(a), Lit::neg(b)));
        and_defs.insert(n_ab1, (Lit::neg(a), Lit::neg(b)));
        and_defs.insert(root, (Lit::neg(n_ab0), Lit::neg(n_ab1)));

        let ts = build_ts(5, vec![a, b], vec![Lit::pos(root)], Vec::new(), and_defs);

        let (result, eliminated) = refactor(&ts);
        assert_eq!(eliminated, 2);
        assert_eq!(result.and_defs.len(), 1);
        assert_eq!(ts.and_defs.len() - result.and_defs.len(), eliminated);

        for mask in 0..4 {
            let mut assignment = FxHashMap::default();
            assignment.insert(a, (mask & 1) != 0);
            assignment.insert(b, (mask & 2) != 0);

            let before = eval_lit(ts.bad_lits[0], &assignment, &ts.and_defs);
            let after = eval_lit(result.bad_lits[0], &assignment, &result.and_defs);
            assert_eq!(before, after, "assignment mask {mask}");
        }
    }

    #[test]
    fn test_refactor_reuses_nonforbidden_duplicate_support_gate() {
        // root = (a | b) & (a | b), plus a separate live copy of !a & !b.
        // The first two duplicate support gates are removable, but the live
        // copy can be reused for the replacement instead of allocating another
        // structurally equivalent gate.
        let a = Var(1);
        let b = Var(2);
        let n_ab0 = Var(3);
        let n_ab1 = Var(4);
        let live_n_ab = Var(5);
        let root = Var(6);

        let mut and_defs = FxHashMap::default();
        and_defs.insert(n_ab0, (Lit::neg(a), Lit::neg(b)));
        and_defs.insert(n_ab1, (Lit::neg(a), Lit::neg(b)));
        and_defs.insert(live_n_ab, (Lit::neg(a), Lit::neg(b)));
        and_defs.insert(root, (Lit::neg(n_ab0), Lit::neg(n_ab1)));

        let ts = build_ts(
            6,
            vec![a, b],
            vec![Lit::pos(root)],
            vec![Lit::pos(live_n_ab)],
            and_defs,
        );

        let (result, eliminated) = refactor(&ts);
        assert_eq!(eliminated, 3);
        assert_eq!(result.and_defs.len(), 1);
        assert_eq!(ts.and_defs.len() - result.and_defs.len(), eliminated);
        assert!(result.and_defs.contains_key(&live_n_ab));
        assert_eq!(result.bad_lits, vec![Lit::neg(live_n_ab)]);
        assert_eq!(result.constraint_lits, vec![Lit::pos(live_n_ab)]);

        for mask in 0..4 {
            let mut assignment = FxHashMap::default();
            assignment.insert(a, (mask & 1) != 0);
            assignment.insert(b, (mask & 2) != 0);

            let before_bad = eval_lit(ts.bad_lits[0], &assignment, &ts.and_defs);
            let after_bad = eval_lit(result.bad_lits[0], &assignment, &result.and_defs);
            assert_eq!(before_bad, after_bad, "bad literal mask {mask}");

            let before_constraint = eval_lit(ts.constraint_lits[0], &assignment, &ts.and_defs);
            let after_constraint =
                eval_lit(result.constraint_lits[0], &assignment, &result.and_defs);
            assert_eq!(
                before_constraint, after_constraint,
                "constraint literal mask {mask}",
            );
        }
    }

    #[test]
    fn test_refactor_skips_when_support_is_shared() {
        let a = Var(1);
        let b = Var(2);
        let c = Var(3);
        let n_ab = Var(4);
        let n_ac = Var(5);
        let root = Var(6);

        let mut and_defs = FxHashMap::default();
        and_defs.insert(n_ab, (Lit::neg(a), Lit::neg(b)));
        and_defs.insert(n_ac, (Lit::neg(a), Lit::neg(c)));
        and_defs.insert(root, (Lit::neg(n_ab), Lit::neg(n_ac)));

        let ts = build_ts(
            6,
            vec![a, b, c],
            vec![Lit::pos(root)],
            vec![Lit::pos(n_ab)],
            and_defs,
        );

        let (result, eliminated) = refactor(&ts);
        assert_eq!(eliminated, 0);
        assert_eq!(result.and_defs.len(), ts.and_defs.len());
    }

    #[test]
    fn test_refactor_factors_negated_shared_and_terms() {
        // bad = (a & b) | (a & c), encoded with an inverted AIG root:
        // ab = a & b; ac = a & c; n_or = !ab & !ac; bad = !n_or.
        // Refactor n_or to !(a & (b | c)), preserving the negated root use.
        let a = Var(1);
        let b = Var(2);
        let c = Var(3);
        let ab = Var(4);
        let ac = Var(5);
        let n_or = Var(6);

        let mut and_defs = FxHashMap::default();
        and_defs.insert(ab, (Lit::pos(a), Lit::pos(b)));
        and_defs.insert(ac, (Lit::pos(a), Lit::pos(c)));
        and_defs.insert(n_or, (Lit::neg(ab), Lit::neg(ac)));

        let ts = build_ts(6, vec![a, b, c], vec![Lit::neg(n_or)], Vec::new(), and_defs);

        let (result, eliminated) = refactor(&ts);
        assert_eq!(eliminated, 1);
        assert_eq!(result.and_defs.len(), 2);

        for mask in 0..8 {
            let mut assignment = FxHashMap::default();
            assignment.insert(a, (mask & 1) != 0);
            assignment.insert(b, (mask & 2) != 0);
            assignment.insert(c, (mask & 4) != 0);

            let before = eval_lit(ts.bad_lits[0], &assignment, &ts.and_defs);
            let after = eval_lit(result.bad_lits[0], &assignment, &result.and_defs);
            assert_eq!(before, after, "assignment mask {mask}");
        }
    }

    #[test]
    fn test_refactor_pipeline_attributes_synthetic_reduction() {
        let a = Var(1);
        let b = Var(2);
        let c = Var(3);
        let n_ab = Var(4);
        let n_ac = Var(5);
        let root = Var(6);

        let mut and_defs = FxHashMap::default();
        and_defs.insert(n_ab, (Lit::neg(a), Lit::neg(b)));
        and_defs.insert(n_ac, (Lit::neg(a), Lit::neg(c)));
        and_defs.insert(root, (Lit::neg(n_ab), Lit::neg(n_ac)));

        let ts = build_ts(6, vec![a, b, c], vec![Lit::pos(root)], Vec::new(), and_defs);
        let config = PreprocessConfig {
            enable_scorr: false,
            enable_frts: false,
            enable_bve: false,
            enable_rewrite: false,
            enable_dag_rewrite: false,
            enable_synthesis: false,
            enable_ternary_sim: false,
            ..PreprocessConfig::default()
        };

        let (result, stats) = preprocess_with_config(&ts, &config);
        assert_eq!(stats.refactor_eliminated, 1);
        assert_eq!(stats.synthesis_gate_reduction, 0);
        assert_eq!(result.and_defs.len(), 2);

        for mask in 0..8 {
            let mut assignment = FxHashMap::default();
            assignment.insert(a, (mask & 1) != 0);
            assignment.insert(b, (mask & 2) != 0);
            assignment.insert(c, (mask & 4) != 0);

            let before = eval_lit(ts.bad_lits[0], &assignment, &ts.and_defs);
            let after = eval_lit(result.bad_lits[0], &assignment, &result.and_defs);
            assert_eq!(before, after, "assignment mask {mask}");
        }
    }

    #[test]
    fn test_refactor_iterates_chained_shared_or_clauses() {
        // root = ((a | b) & (a | c)) & (a | d). The second factoring
        // opportunity only appears after the first root substitution.
        let a = Var(1);
        let b = Var(2);
        let c = Var(3);
        let d = Var(4);
        let n_ab = Var(5);
        let n_ac = Var(6);
        let n_ad = Var(7);
        let pair = Var(8);
        let root = Var(9);

        let mut and_defs = FxHashMap::default();
        and_defs.insert(n_ab, (Lit::neg(a), Lit::neg(b)));
        and_defs.insert(n_ac, (Lit::neg(a), Lit::neg(c)));
        and_defs.insert(n_ad, (Lit::neg(a), Lit::neg(d)));
        and_defs.insert(pair, (Lit::neg(n_ab), Lit::neg(n_ac)));
        and_defs.insert(root, (Lit::pos(pair), Lit::neg(n_ad)));

        let ts = build_ts(
            9,
            vec![a, b, c, d],
            vec![Lit::pos(root)],
            vec![Lit::neg(root)],
            and_defs,
        );

        let (result, eliminated) = refactor(&ts);
        assert_eq!(eliminated, 2);
        assert_eq!(result.and_defs.len(), 3);
        assert_eq!(ts.and_defs.len() - result.and_defs.len(), eliminated);

        let config = PreprocessConfig {
            enable_scorr: false,
            enable_frts: false,
            enable_bve: false,
            enable_rewrite: false,
            enable_dag_rewrite: false,
            enable_synthesis: false,
            enable_ternary_sim: false,
            ..PreprocessConfig::default()
        };
        let (pipeline_result, stats) = preprocess_with_config(&ts, &config);
        assert_eq!(stats.refactor_eliminated, 2);
        assert_eq!(pipeline_result.and_defs.len(), 3);

        for mask in 0..16 {
            let mut assignment = FxHashMap::default();
            assignment.insert(a, (mask & 1) != 0);
            assignment.insert(b, (mask & 2) != 0);
            assignment.insert(c, (mask & 4) != 0);
            assignment.insert(d, (mask & 8) != 0);

            let before_bad = eval_lit(ts.bad_lits[0], &assignment, &ts.and_defs);
            let after_bad = eval_lit(result.bad_lits[0], &assignment, &result.and_defs);
            assert_eq!(before_bad, after_bad, "bad literal mask {mask}");

            let before_constraint = eval_lit(ts.constraint_lits[0], &assignment, &ts.and_defs);
            let after_constraint =
                eval_lit(result.constraint_lits[0], &assignment, &result.and_defs);
            assert_eq!(
                before_constraint, after_constraint,
                "constraint literal mask {mask}",
            );
        }
    }
}
