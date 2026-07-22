// Copyright 2026 Andrew Yates
// Licensed under the Apache License, Version 2.0

//! Random-net differential battery for the native ROBDD engine. Generates
//! arbitrary small bounded P/T nets and asserts the symbolic `tla-bdd` verdicts
//! match an independent explicit BFS over the reachable graph — on ALL four
//! StateSpace metrics, on EF/AG reachability (fireability + cardinality), and on
//! CTL `EF`/`EG`/`EX`. This is the validation gate that the engine is
//! verdict-faithful on arbitrary nets before `tla-dd` is routed onto it.

use std::collections::{BTreeMap, HashSet};
use tla_bdd::petri::{
    evaluate_ctl, evaluate_reachability, fair_cycle_exists, fair_cycle_exists_generalized,
    reachable_count_bounded, reachable_count_bounded_cedge, reachable_is_sound_inductive_invariant,
    state_space_metrics_bounded, state_space_metrics_bounded_checked, upper_bounds_bounded,
    BoundedNet, BoundedTransition, Ctl, Pred, Query, StateSpaceMetrics,
};

use proptest::prelude::*;

/// A random bounded net: 1..=4 places (bounds 1..=3), 1..=5 transitions.
fn arb_net() -> impl Strategy<Value = BoundedNet> {
    (1usize..=4).prop_flat_map(|np| {
        let bounds = prop::collection::vec(1u64..=3, np);
        bounds.prop_flat_map(move |bounds| {
            let init = bounds.iter().map(|&b| 0u64..=b).collect::<Vec<_>>();
            let trans = prop::collection::vec(
                (
                    prop::collection::vec(0u64..=2, np),
                    prop::collection::vec(0u64..=2, np),
                ),
                1..=5,
            );
            (Just(bounds), init, trans).prop_map(|(bounds, init, trans)| BoundedNet {
                bounds,
                init,
                transitions: trans
                    .into_iter()
                    .map(|(pre, post)| BoundedTransition { pre, post })
                    .collect(),
            })
        })
    })
}

/// Explicit reachable markings + successor map (the oracle).
fn reach_graph(net: &BoundedNet) -> (Vec<Vec<u64>>, BTreeMap<Vec<u64>, Vec<Vec<u64>>>) {
    let np = net.bounds.len();
    let mut seen: HashSet<Vec<u64>> = HashSet::new();
    seen.insert(net.init.clone());
    let mut frontier = vec![net.init.clone()];
    let mut succ: BTreeMap<Vec<u64>, Vec<Vec<u64>>> = BTreeMap::new();
    while let Some(m) = frontier.pop() {
        let mut ss = Vec::new();
        for t in &net.transitions {
            if !(0..np).all(|p| m[p] >= t.pre[p]) {
                continue;
            }
            let mut n = m.clone();
            let mut ok = true;
            for p in 0..np {
                let v = m[p] - t.pre[p] + t.post[p];
                if v > net.bounds[p] {
                    ok = false;
                    break;
                }
                n[p] = v;
            }
            if ok {
                ss.push(n.clone());
                if seen.insert(n.clone()) {
                    frontier.push(n);
                }
            }
        }
        succ.insert(m, ss);
    }
    (seen.into_iter().collect(), succ)
}

fn explicit_metrics(
    _net: &BoundedNet,
    reach: &[Vec<u64>],
    succ: &BTreeMap<Vec<u64>, Vec<Vec<u64>>>,
) -> StateSpaceMetrics {
    let mut edges: u128 = 0;
    let mut max_in_place = 0u64;
    let mut max_sum = 0u64;
    for m in reach {
        edges += succ[m].len() as u128;
        max_in_place = max_in_place.max(*m.iter().max().unwrap_or(&0));
        max_sum = max_sum.max(m.iter().sum());
    }
    StateSpaceMetrics {
        states: reach.len() as u128,
        edges,
        max_token_in_place: max_in_place,
        max_token_sum: max_sum,
    }
}

fn eval_pred(net: &BoundedNet, m: &[u64], p: &Pred) -> bool {
    match p {
        Pred::Fireable(t) => {
            let tr = &net.transitions[*t];
            (0..net.bounds.len()).all(|pl| m[pl] >= tr.pre[pl])
        }
        Pred::TokenLe { coeffs, k } => {
            let s: i128 = coeffs.iter().zip(m).map(|(&c, &mv)| c * mv as i128).sum();
            s <= *k
        }
        Pred::And(cs) => cs.iter().all(|c| eval_pred(net, m, c)),
        Pred::Or(cs) => cs.iter().any(|c| eval_pred(net, m, c)),
        Pred::Not(c) => !eval_pred(net, m, c),
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(2000))]

    #[test]
    fn statespace_metrics_match_bfs(net in arb_net()) {
        let (reach, succ) = reach_graph(&net);
        let want = explicit_metrics(&net, &reach, &succ);
        let got = state_space_metrics_bounded_checked(&net);
        prop_assert_eq!(got, Some(want));
        // The legacy saturating shims must agree wherever the count is exact.
        prop_assert_eq!(state_space_metrics_bounded(&net), want);
        prop_assert_eq!(reachable_count_bounded(&net), want.states);
        // The complemented-edge core must reach the SAME state count on the same
        // real spec (end-to-end validation of the ~2x-denser engine).
        prop_assert_eq!(reachable_count_bounded_cedge(&net), Some(want.states));
        // The proof-carrying soundness certificate must hold for EVERY net:
        // R is an inductive invariant containing init ⇒ R ⊇ reachable.
        prop_assert!(reachable_is_sound_inductive_invariant(&net));
        // UpperBounds: max weighted token-sum over R for a sample coefficient
        // vector ([1,2,3,...]) must equal the explicit max over reachable markings.
        let coeffs: Vec<i128> = (0..net.bounds.len() as i128).map(|i| i + 1).collect();
        let ub = upper_bounds_bounded(&net, &[coeffs.clone()])[0];
        let want_ub = reach
            .iter()
            .map(|m| coeffs.iter().zip(m).map(|(&c, &v)| c * v as i128).sum::<i128>())
            .max()
            .unwrap_or(0);
        prop_assert_eq!(ub, want_ub, "UB max-weighted-sum must match explicit max over R");
    }

    #[test]
    fn reachability_ef_ag_match_bfs(net in arb_net(), tsel in 0usize..5, kk in 0i128..=6) {
        let (reach, _succ) = reach_graph(&net);
        let nt = net.transitions.len();
        // a fireability atom and a cardinality atom (sum of all places <= kk)
        let fire = Pred::Fireable(tsel % nt);
        let card = Pred::TokenLe { coeffs: vec![1; net.bounds.len()], k: kk };
        let preds = [fire, card];
        let queries: Vec<Query> = preds.iter().map(|p| {
            let c = clone_pred(p);
            Query::Ef(c)
        }).chain(preds.iter().map(|p| Query::Ag(clone_pred(p)))).collect();
        let got = evaluate_reachability(&net, &queries);
        let mut want = Vec::new();
        for p in &preds { want.push(reach.iter().any(|m| eval_pred(&net, m, p))); }
        for p in &preds { want.push(reach.iter().all(|m| eval_pred(&net, m, p))); }
        prop_assert_eq!(got, want);
    }

    #[test]
    fn ctl_and_fair_cycle_match_bfs(net in arb_net(), kk in 0i128..=6) {
        let (reach, succ) = reach_graph(&net);
        // atom: sum of tokens <= kk
        let atom = || Pred::TokenLe { coeffs: vec![1; net.bounds.len()], k: kk };
        // EF atom at init
        let got_ef = evaluate_ctl(&net, &Ctl::Ef(Box::new(Ctl::Atom(atom()))));
        let want_ef = explicit_ctl(&net, &reach, &succ, &Ctl::Ef(Box::new(Ctl::Atom(atom())))).contains(&net.init);
        prop_assert_eq!(got_ef, want_ef);
        // EG atom at init
        let got_eg = evaluate_ctl(&net, &Ctl::Eg(Box::new(Ctl::Atom(atom()))));
        let want_eg = explicit_ctl(&net, &reach, &succ, &Ctl::Eg(Box::new(Ctl::Atom(atom())))).contains(&net.init);
        prop_assert_eq!(got_eg, want_eg);
        // E[true U atom] at init (EU operator).
        let eu = || Ctl::Eu(Box::new(Ctl::Atom(Pred::TokenLe { coeffs: vec![0; net.bounds.len()], k: 0 })), Box::new(Ctl::Atom(atom())));
        let got_eu = evaluate_ctl(&net, &eu());
        let want_eu = explicit_ctl(&net, &reach, &succ, &eu()).contains(&net.init);
        prop_assert_eq!(got_eu, want_eu, "EU must match explicit CTL");
        // fair cycle through the atom (FG pattern)
        let got_fc = fair_cycle_exists(&net, &atom(), None);
        let want_fc = explicit_fair_cycle(&net, &reach, &succ, &atom());
        prop_assert_eq!(got_fc, want_fc);
        // cycle entirely within the atom (GF pattern, domain-restricted)
        let got_w = fair_cycle_exists(&net, &atom(), Some(&atom()));
        let want_w = explicit_within_cycle(&net, &reach, &succ, &atom());
        prop_assert_eq!(got_w, want_w);
        // GENERALIZED fair-cycle (the GBA-emptiness core): a cycle hitting BOTH
        // (Σtokens ≤ kk) AND (Σtokens ≥ kk) infinitely often, vs the explicit
        // SCC-based generalized-Büchi oracle.
        let atom2 = || Pred::TokenLe { coeffs: vec![-1; net.bounds.len()], k: -kk }; // Σ ≥ kk
        let got_gen = fair_cycle_exists_generalized(&net, &[atom(), atom2()], None);
        let want_gen = explicit_generalized_fair_cycle(&net, &reach, &succ, &[atom(), atom2()]);
        prop_assert_eq!(got_gen, want_gen, "generalized fair-cycle (2 sets) must match explicit");
    }
}

fn clone_pred(p: &Pred) -> Pred {
    match p {
        Pred::Fireable(t) => Pred::Fireable(*t),
        Pred::TokenLe { coeffs, k } => Pred::TokenLe {
            coeffs: coeffs.clone(),
            k: *k,
        },
        Pred::And(cs) => Pred::And(cs.iter().map(clone_pred).collect()),
        Pred::Or(cs) => Pred::Or(cs.iter().map(clone_pred).collect()),
        Pred::Not(c) => Pred::Not(Box::new(clone_pred(c))),
    }
}

fn explicit_ctl(
    net: &BoundedNet,
    reach: &[Vec<u64>],
    succ: &BTreeMap<Vec<u64>, Vec<Vec<u64>>>,
    f: &Ctl,
) -> HashSet<Vec<u64>> {
    let reach_set: HashSet<Vec<u64>> = reach.iter().cloned().collect();
    match f {
        Ctl::Atom(p) => reach
            .iter()
            .filter(|m| eval_pred(net, m, p))
            .cloned()
            .collect(),
        Ctl::Not(g) => {
            let s = explicit_ctl(net, reach, succ, g);
            reach_set.difference(&s).cloned().collect()
        }
        Ctl::And(g, h) => {
            let a = explicit_ctl(net, reach, succ, g);
            let b = explicit_ctl(net, reach, succ, h);
            a.intersection(&b).cloned().collect()
        }
        Ctl::Or(g, h) => {
            let a = explicit_ctl(net, reach, succ, g);
            let b = explicit_ctl(net, reach, succ, h);
            a.union(&b).cloned().collect()
        }
        Ctl::Ex(g) => {
            let s = explicit_ctl(net, reach, succ, g);
            reach
                .iter()
                .filter(|m| succ[*m].iter().any(|n| s.contains(n)))
                .cloned()
                .collect()
        }
        Ctl::Ef(g) => {
            let mut s = explicit_ctl(net, reach, succ, g);
            loop {
                let mut grew = false;
                for m in reach {
                    if !s.contains(m) && succ[m].iter().any(|n| s.contains(n)) {
                        s.insert(m.clone());
                        grew = true;
                    }
                }
                if !grew {
                    return s;
                }
            }
        }
        Ctl::Eg(g) => {
            // MAXIMAL-PATH EG (non-totalized MCC convention): keep `m` if it is a
            // DEADLOCK (no successors at all — a maximal finite path) OR has a
            // successor still in the set. Only drop `m` when it has successors but
            // none remain in the set. (A deadlock where φ holds satisfies EG φ.)
            let phi = explicit_ctl(net, reach, succ, g);
            let mut s = phi.clone();
            loop {
                let mut shrunk = false;
                for m in s.clone() {
                    if !succ[&m].is_empty() && !succ[&m].iter().any(|n| s.contains(n)) {
                        s.remove(&m);
                        shrunk = true;
                    }
                }
                if !shrunk {
                    return s;
                }
            }
        }
        Ctl::Eu(g, h) => {
            let phi = explicit_ctl(net, reach, succ, g);
            let psi = explicit_ctl(net, reach, succ, h);
            let mut s = psi.clone();
            loop {
                let mut grew = false;
                for m in reach {
                    if !s.contains(m) && phi.contains(m) && succ[m].iter().any(|n| s.contains(n)) {
                        s.insert(m.clone());
                        grew = true;
                    }
                }
                if !grew {
                    return s;
                }
            }
        }
    }
}

/// Explicit GENERALIZED-Büchi emptiness oracle: is there a reachable non-trivial
/// SCC that intersects EVERY accepting predicate? (Within an SCC a cycle visiting
/// all its states exists, so hitting each set ⇒ a single cycle hits all.) Ground
/// truth for `fair_cycle_exists_generalized`.
fn explicit_generalized_fair_cycle(
    net: &BoundedNet,
    reach: &[Vec<u64>],
    succ: &BTreeMap<Vec<u64>, Vec<Vec<u64>>>,
    accs: &[Pred],
) -> bool {
    let fwd = |s: &Vec<u64>| -> HashSet<Vec<u64>> {
        let mut seen = HashSet::new();
        seen.insert(s.clone());
        let mut stack = vec![s.clone()];
        while let Some(m) = stack.pop() {
            for n in &succ[&m] {
                if seen.insert(n.clone()) {
                    stack.push(n.clone());
                }
            }
        }
        seen
    };
    for s in reach {
        let fs = fwd(s);
        // SCC(s) = { t reachable from s : s reachable from t }.
        let scc: Vec<Vec<u64>> = fs.iter().filter(|t| fwd(t).contains(s)).cloned().collect();
        let nontrivial = scc.len() > 1 || succ[s].contains(s);
        if !nontrivial {
            continue;
        }
        if accs
            .iter()
            .all(|a| scc.iter().any(|t| eval_pred(net, t, a)))
        {
            return true;
        }
    }
    false
}

fn explicit_fair_cycle(
    net: &BoundedNet,
    reach: &[Vec<u64>],
    succ: &BTreeMap<Vec<u64>, Vec<Vec<u64>>>,
    accepting: &Pred,
) -> bool {
    for s in reach {
        if !eval_pred(net, s, accepting) {
            continue;
        }
        let mut seen: HashSet<Vec<u64>> = HashSet::new();
        let mut frontier: Vec<Vec<u64>> = succ[s].clone(); // ≥1 step
        while let Some(m) = frontier.pop() {
            if &m == s {
                return true;
            }
            if seen.insert(m.clone()) {
                frontier.extend(succ[&m].clone());
            }
        }
    }
    false
}

/// Explicit: a reachable cycle ENTIRELY WITHIN `within`-states (the GF pattern).
fn explicit_within_cycle(
    net: &BoundedNet,
    reach: &[Vec<u64>],
    succ: &BTreeMap<Vec<u64>, Vec<Vec<u64>>>,
    within: &Pred,
) -> bool {
    for s in reach {
        if !eval_pred(net, s, within) {
            continue;
        }
        let mut seen: HashSet<Vec<u64>> = HashSet::new();
        let mut frontier: Vec<Vec<u64>> = succ[s]
            .iter()
            .filter(|m| eval_pred(net, m, within))
            .cloned()
            .collect();
        while let Some(m) = frontier.pop() {
            if &m == s {
                return true;
            }
            if seen.insert(m.clone()) {
                frontier.extend(
                    succ[&m]
                        .iter()
                        .filter(|x| eval_pred(net, x, within))
                        .cloned(),
                );
            }
        }
    }
    false
}
