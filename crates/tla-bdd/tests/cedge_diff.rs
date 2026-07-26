// Copyright 2026 Andrew Yates
// Licensed under the Apache License, Version 2.0

//! Differential battery: the complemented-edge core [`tla_bdd::cedge::CBdd`] must
//! agree with the already-validated [`tla_bdd::Bdd`] on `sat_count` for arbitrary
//! Boolean formulas. Since `Bdd` is itself cross-checked against explicit BFS on
//! 6000 nets, this pins the new complement core (especially the subtle
//! complement-aware count) to the same ground truth before the live core is
//! migrated onto it.

use proptest::prelude::*;
use tla_bdd::cedge::CBdd;
use tla_bdd::petri::{
    reachable_count_bounded_cedge, reachable_count_only_bdd, BoundedNet, BoundedTransition,
};
use tla_bdd::Bdd;

/// Perf data point (run with `TY_RUN_CEDGE_PERF=1` and `--nocapture`):
/// complemented-edge core vs plain Bdd on a reachability fixpoint. Quantifies
/// whether the ~2× node reduction translates to wall-clock — informs the
/// live-engine migration.
#[test]
fn perf_cedge_vs_bdd_reachability() {
    if !std::env::var_os("TY_RUN_CEDGE_PERF").is_some_and(|value| value == "1") {
        eprintln!(
            "SKIP perf_cedge_vs_bdd_reachability: set TY_RUN_CEDGE_PERF=1 \
             to authorize the timing campaign"
        );
        return;
    }
    use std::time::Instant;
    // 8 independent 0..=3 counters: |R| = 4^8 = 65536.
    let np = 8;
    let net = BoundedNet {
        bounds: vec![3u64; np],
        init: vec![0u64; np],
        transitions: (0..np)
            .map(|p| {
                let mut post = vec![0u64; np];
                post[p] = 1;
                BoundedTransition {
                    pre: vec![0u64; np],
                    post,
                }
            })
            .collect(),
    };
    // FAIR comparison: both count-only (no metrics), same encoding + fixpoint.
    let t0 = Instant::now();
    let cb = reachable_count_bounded_cedge(&net).expect("count fits u128");
    let t_c = t0.elapsed();
    let t1 = Instant::now();
    let bb = reachable_count_only_bdd(&net).expect("count fits u128");
    let t_b = t1.elapsed();
    assert_eq!(cb, bb);
    eprintln!(
        "PERF-CEDGE |R|={} : cedge {:?} vs bdd {:?} (cedge/bdd {:.2}x)",
        cb,
        t_c,
        t_b,
        t_c.as_secs_f64() / t_b.as_secs_f64().max(1e-9)
    );
}

const NVARS: u32 = 5;

#[derive(Debug, Clone)]
enum F {
    Var(u32),
    Not(Box<F>),
    And(Box<F>, Box<F>),
    Or(Box<F>, Box<F>),
    Xor(Box<F>, Box<F>),
}

fn arb_formula() -> impl Strategy<Value = F> {
    let leaf = (0..NVARS).prop_map(F::Var);
    leaf.prop_recursive(5, 48, 2, |inner| {
        prop_oneof![
            inner.clone().prop_map(|f| F::Not(Box::new(f))),
            (inner.clone(), inner.clone()).prop_map(|(a, b)| F::And(Box::new(a), Box::new(b))),
            (inner.clone(), inner.clone()).prop_map(|(a, b)| F::Or(Box::new(a), Box::new(b))),
            (inner.clone(), inner).prop_map(|(a, b)| F::Xor(Box::new(a), Box::new(b))),
        ]
    })
}

fn build_c(b: &mut CBdd, f: &F) -> u32 {
    match f {
        F::Var(i) => b.var(*i),
        F::Not(g) => CBdd::not(build_c(b, g)),
        F::And(g, h) => {
            let x = build_c(b, g);
            let y = build_c(b, h);
            b.and(x, y)
        }
        F::Or(g, h) => {
            let x = build_c(b, g);
            let y = build_c(b, h);
            b.or(x, y)
        }
        F::Xor(g, h) => {
            let x = build_c(b, g);
            let y = build_c(b, h);
            b.xor(x, y)
        }
    }
}

fn build_b(b: &mut Bdd, f: &F) -> u32 {
    match f {
        F::Var(i) => b.var(*i),
        F::Not(g) => {
            let x = build_b(b, g);
            b.not(x)
        }
        F::And(g, h) => {
            let x = build_b(b, g);
            let y = build_b(b, h);
            b.and(x, y)
        }
        F::Or(g, h) => {
            let x = build_b(b, g);
            let y = build_b(b, h);
            b.or(x, y)
        }
        F::Xor(g, h) => {
            let x = build_b(b, g);
            let y = build_b(b, h);
            b.xor(x, y)
        }
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(3000))]

    #[test]
    fn cedge_satcount_matches_bdd(f in arb_formula()) {
        let mut cb = CBdd::new();
        let ec = build_c(&mut cb, &f);
        let mut bb = Bdd::new();
        let eb = build_b(&mut bb, &f);
        prop_assert_eq!(
            cb.sat_count(ec, NVARS),
            bb.sat_count(eb, NVARS),
            "complemented-edge sat_count must match the validated Bdd on {:?}", f
        );
        // Native variable reordering is answer-preserving: rebuilding under ANY
        // permutation leaves the model count unchanged (only the node count moves).
        let perm: Vec<u32> = {
            // deterministic permutation derived from the formula shape (rotate).
            let k = (bb.node_count(eb) as u32) % NVARS;
            (0..NVARS).map(|i| (i + k) % NVARS).collect()
        };
        let reordered = bb.reorder(eb, &perm);
        prop_assert_eq!(
            bb.sat_count(reordered, NVARS),
            bb.sat_count(eb, NVARS),
            "reorder must preserve the model count under permutation {:?}", perm
        );
        // VAR-PRESERVING reorder: the SAME function on EVERY assignment (strong).
        let (nb, root) = bb.reorder_into(eb, &perm);
        prop_assert_eq!(nb.sat_count(root, NVARS), bb.sat_count(eb, NVARS), "reorder_into count");
        for bits in 0u32..(1 << NVARS) {
            let a: Vec<bool> = (0..NVARS).map(|i| bits & (1 << i) != 0).collect();
            prop_assert_eq!(
                nb.eval(root, &a),
                bb.eval(eb, &a),
                "reorder_into changed the function under {:?} at {:?}", perm, a
            );
        }
        // Benefit invariant: complement sharing only MERGES nodes (f and ¬f share
        // one), so the complement core never uses more nodes than the plain Bdd.
        prop_assert!(
            cb.node_count(ec) <= bb.node_count(eb),
            "complement core must not exceed Bdd node count: {} > {} on {:?}",
            cb.node_count(ec), bb.node_count(eb), f
        );
    }

    /// exists/and_exists on the complement core must match the validated Bdd.
    #[test]
    fn cedge_quantification_matches_bdd(f in arb_formula(), g in arb_formula(), qmask in 0u32..32) {
        // Quantify the variables whose bit is set in qmask (over the 5 vars).
        let vars: Vec<u32> = (0..NVARS).filter(|i| qmask & (1 << i) != 0).collect();
        // ∃vars. f
        let mut cb = CBdd::new();
        let cf = build_c(&mut cb, &f);
        let ce = cb.exists(cf, &vars);
        let mut bb = Bdd::new();
        let bf = build_b(&mut bb, &f);
        let be = bb.exists(bf, &vars);
        prop_assert_eq!(cb.sat_count(ce, NVARS), bb.sat_count(be, NVARS), "exists mismatch");
        // ∃vars. (f ∧ g)
        let cg = build_c(&mut cb, &g);
        let cae = cb.and_exists(cf, cg, &vars);
        let bg = build_b(&mut bb, &g);
        let bae = bb.and_exists(bf, bg, &vars);
        prop_assert_eq!(cb.sat_count(cae, NVARS), bb.sat_count(bae, NVARS), "and_exists mismatch");
    }
}
