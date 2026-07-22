// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! BDD symbolic reachability engine — the decision-diagram lane for the
//! hardware frontend.
//!
//! This wires the workspace's general symbolic engine (`tla-bdd`, the same
//! GC'd + sifting ROBDD core the Petri/MCC examinations run on) into the AIGER
//! safety portfolio. It computes the EXACT forward reachable set of the latch
//! state space and therefore decides UNBOUNDED safety in both directions:
//!
//! - **Safe**: the least fixpoint converges with no bad state reached — an
//!   exact, complete proof (unlike BMC's bounded "no bad ≤ k"). Before the
//!   verdict is surfaced the engine re-checks its own fixpoint *inductively*
//!   (`init ⊆ R`, `post(R) ⊆ R`, `R ∩ bad = ∅`) so an engine bug degrades to a
//!   decline, never a wrong Safe.
//! - **Bad reachable at depth k**: the fixpoint intersects the bad set at
//!   frontier ring `k` (minimal transition depth). The portfolio runner then
//!   re-derives the counterexample through the CPU BMC engine at depth `k` —
//!   the identical protocol the GPU exhaustive-BMC lane uses — so every
//!   `Unsafe` carries a portfolio-verifiable trace and the BDD engine itself
//!   never has to be trusted for witnesses.
//!
//! # Fail-closed admission and budgets
//!
//! The lane DECLINES (→ the other portfolio engines proceed unaffected) on:
//! environment constraints (v1 does not model them), relational/gate-referencing
//! init clauses, latch/gate counts over the configured caps, any variable the
//! functional view cannot resolve, node-budget/memory/deadline exhaustion
//! (via `tla-bdd`'s cooperative `BddAbort`), and a failed inductive self-check.
//! It never guesses: every decline is `Unknown`, both verdicts are exact.
//!
//! # Encoding
//!
//! Latch `i` ↦ BDD var `2i` (current) / `2i+1` (next) — the standard
//! interleaved order, which keeps `x_i ↔ f_i` conjuncts local; primary input
//! `j` ↦ BDD var `2L + j`, existentially quantified at every image round. The
//! transition relation is the monolithic `⋀_i (x_i' ↔ f_i(x, in))` built
//! functionally from the AIG (`Transys::and_defs`), not from the Tseitin CNF,
//! so no auxiliary gate variables enter the relation.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;

use tla_bdd::{Bdd, NodeId, ReachOutcome};

use crate::sat_types::{Lit, Var};
use crate::transys::Transys;

/// Admission caps and budgets for the BDD reachability lane.
#[derive(Debug, Clone)]
pub struct BddReachConfig {
    /// Decline circuits with more latches than this. BDD reachability's sweet
    /// spot is small-to-medium control logic; past this, the node budget
    /// would almost always fire anyway — declining early is cheaper.
    pub max_latches: usize,
    /// Decline circuits with more AND gates than this (relation-build cost).
    pub max_ands: usize,
    /// BDD node-store budget (cooperative abort inside every operation).
    /// `None` derives the adaptive default from available memory.
    pub node_budget: Option<usize>,
}

impl Default for BddReachConfig {
    fn default() -> Self {
        Self {
            max_latches: 256,
            max_ands: 50_000,
            node_budget: None,
        }
    }
}

/// Outcome of the BDD reachability engine.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BddReachOutcome {
    /// Exact fixpoint, no bad state reachable — full (unbounded) safety,
    /// re-verified inductively inside the engine before being surfaced.
    Safe,
    /// A bad state is reachable in exactly `depth` transition steps (minimal).
    /// The caller re-derives a verifiable trace via CPU BMC at this depth.
    BadReachable {
        /// Minimal number of transition steps from an initial state to a bad
        /// state (0 = an initial state is bad).
        depth: usize,
    },
    /// The lane declined (admission gate, budget, or self-check failure) —
    /// fail-closed, no verdict.
    Declined {
        /// Human-readable decline reason (diagnostics only).
        reason: String,
    },
}

fn declined(reason: impl Into<String>) -> BddReachOutcome {
    BddReachOutcome::Declined {
        reason: reason.into(),
    }
}

/// How a Transys variable maps into the BDD state space.
#[derive(Clone, Copy)]
enum VarRole {
    /// Latch with index `i` → current-state BDD var `2i`.
    Latch(usize),
    /// Primary input with index `j` → BDD var `2L + j`.
    Input(usize),
}

/// The functional AIG→BDD lowering context.
struct Lowering<'a> {
    ts: &'a Transys,
    roles: HashMap<Var, VarRole>,
    memo: HashMap<Var, NodeId>,
    num_latches: usize,
}

impl<'a> Lowering<'a> {
    fn new(ts: &'a Transys) -> Self {
        let mut roles = HashMap::new();
        for (i, &v) in ts.latch_vars.iter().enumerate() {
            roles.insert(v, VarRole::Latch(i));
        }
        for (j, &v) in ts.input_vars.iter().enumerate() {
            roles.insert(v, VarRole::Input(j));
        }
        Self {
            ts,
            roles,
            memo: HashMap::new(),
            num_latches: ts.latch_vars.len(),
        }
    }

    fn bdd_var_of_role(&self, role: VarRole) -> u32 {
        match role {
            VarRole::Latch(i) => (2 * i) as u32,
            VarRole::Input(j) => (2 * self.num_latches + j) as u32,
        }
    }

    /// BDD of a Transys variable as a function of current-state latch vars and
    /// input vars, resolving AND gates through `and_defs`. Iterative (explicit
    /// stack) so deep combinational cones cannot overflow the call stack.
    /// `None` = an unresolvable variable (fail-closed decline upstream).
    fn var_node(&mut self, bdd: &mut Bdd, v: Var) -> Option<NodeId> {
        if let Some(&n) = self.memo.get(&v) {
            return Some(n);
        }
        // Cycle guard (audit 2026-07-10): a well-formed AIG is acyclic, so
        // each var is visited a bounded number of times (push + once per
        // child-ready re-check). A malformed cyclic `and_defs` would otherwise
        // spin this loop forever without reaching an aborting `bdd` op —
        // decline instead (fail-closed), never hang.
        let mut fuel = 8 * (self.ts.and_defs.len() + self.roles.len() + 2);
        let mut stack = vec![v];
        while let Some(&top) = stack.last() {
            fuel = fuel.checked_sub(1)?;
            if self.memo.contains_key(&top) {
                stack.pop();
                continue;
            }
            if top == Var::CONST {
                // AIGER var 0: positive literal is constant FALSE.
                self.memo.insert(top, Bdd::FALSE);
                stack.pop();
                continue;
            }
            if let Some(&role) = self.roles.get(&top) {
                let node = bdd.var(self.bdd_var_of_role(role));
                self.memo.insert(top, node);
                stack.pop();
                continue;
            }
            let Some(&(a, b)) = self.ts.and_defs.get(&top) else {
                // Not const / latch / input / AND — unresolvable.
                return None;
            };
            let need_a = !self.memo.contains_key(&a.var());
            let need_b = !self.memo.contains_key(&b.var());
            if need_a {
                stack.push(a.var());
            }
            if need_b {
                stack.push(b.var());
            }
            if !need_a && !need_b {
                let an = self.lit_node_memoized(bdd, a)?;
                let bn = self.lit_node_memoized(bdd, b)?;
                let node = bdd.and(an, bn);
                self.memo.insert(top, node);
                stack.pop();
            }
        }
        self.memo.get(&v).copied()
    }

    /// Literal → BDD, assuming the literal's variable is already memoized
    /// (used inside the iterative gate walk).
    fn lit_node_memoized(&self, bdd: &mut Bdd, lit: Lit) -> Option<NodeId> {
        let base = self.memo.get(&lit.var()).copied()?;
        Some(if lit.is_negated() {
            bdd.not(base)
        } else {
            base
        })
    }

    /// Literal → BDD, resolving through gates as needed.
    fn lit_node(&mut self, bdd: &mut Bdd, lit: Lit) -> Option<NodeId> {
        let base = self.var_node(bdd, lit.var())?;
        Some(if lit.is_negated() {
            bdd.not(base)
        } else {
            base
        })
    }
}

/// Run BDD symbolic reachability on `ts`.
///
/// `deadline` bounds the whole run (build + fixpoint + self-check);
/// `cancelled` is polled between build phases so a portfolio winner stops the
/// lane early. Every exit path other than the two exact verdicts is a
/// [`BddReachOutcome::Declined`].
pub fn bdd_reach_check(
    ts: &Transys,
    config: &BddReachConfig,
    deadline: Option<Instant>,
    cancelled: &Arc<AtomicBool>,
) -> BddReachOutcome {
    // ---- Admission gates (all fail-closed declines) ----
    if !ts.constraint_lits.is_empty() {
        return declined("environment constraints not modeled by the BDD lane (v1)");
    }
    if ts.num_latches > config.max_latches {
        return declined(format!(
            "latch count {} over the BDD admission cap {}",
            ts.num_latches, config.max_latches
        ));
    }
    if ts.and_defs.len() > config.max_ands {
        return declined(format!(
            "AND-gate count {} over the BDD admission cap {}",
            ts.and_defs.len(),
            config.max_ands
        ));
    }
    // Init clauses must be over latch variables only (constant resets).
    // Gate-referencing / input-referencing init (relational reset) is v1-out.
    for clause in &ts.init_clauses {
        for lit in &clause.lits {
            let v = lit.var();
            let is_latch = ts.latch_vars.contains(&v);
            if !is_latch && v != Var::CONST {
                return declined("relational init clause (non-latch literal) — BDD lane v1");
            }
        }
    }

    let num_latches = ts.latch_vars.len();
    let num_inputs = ts.input_vars.len();

    let mut bdd = Bdd::new();
    let node_budget = config
        .node_budget
        .unwrap_or_else(tla_bdd::default_abort_node_budget);
    bdd.set_abort_limits(Some(node_budget), deadline);
    // Cooperative cancellation (audit 2026-07-10): when another portfolio
    // engine wins, the shared flag aborts the lane in-operation (next fresh
    // node insertion / next fixpoint round) instead of running to deadline.
    bdd.set_abort_flag(Some(cancelled.clone()));

    let mut lowering = Lowering::new(ts);

    // Variable lists for the fixpoint.
    let current: Vec<u32> = (0..num_latches).map(|i| (2 * i) as u32).collect();
    let next: Vec<u32> = (0..num_latches).map(|i| (2 * i + 1) as u32).collect();
    let inputs: Vec<u32> = (0..num_inputs)
        .map(|j| (2 * num_latches + j) as u32)
        .collect();
    let mut quantify = current.clone();
    quantify.extend_from_slice(&inputs);

    // ---- Build phase (each step under the cooperative abort) ----
    let build = tla_bdd::catch_abort(|| {
        // Init: conjunction of the (latch-only) init clauses.
        let mut init = Bdd::TRUE;
        for clause in &ts.init_clauses {
            let mut c = Bdd::FALSE;
            for lit in &clause.lits {
                let ln = lowering.lit_node(&mut bdd, *lit)?;
                c = bdd.or(c, ln);
            }
            init = bdd.and(init, c);
        }

        // Bad target: OR over all bad literals (functions of latches+inputs;
        // inputs stay free in the target — a state is bad-capable iff SOME
        // input valuation makes a bad literal true at it, which is exactly
        // the trace semantics of observing bad at that step).
        let mut target = Bdd::FALSE;
        for &b in &ts.bad_lits {
            let bn = lowering.lit_node(&mut bdd, b)?;
            target = bdd.or(target, bn);
        }

        if cancelled.load(Ordering::Relaxed) {
            return None;
        }

        // Transition relation: ⋀_i (x_i' ↔ f_i(x, in)).
        let mut trans = Bdd::TRUE;
        for (i, &latch) in ts.latch_vars.iter().enumerate() {
            let &next_lit = ts.next_state.get(&latch)?;
            let f = lowering.lit_node(&mut bdd, next_lit)?;
            let xn = bdd.var((2 * i + 1) as u32);
            let x = bdd.xor(xn, f);
            let iff = bdd.not(x);
            trans = bdd.and(trans, iff);
            if cancelled.load(Ordering::Relaxed) {
                return None;
            }
        }
        Some((init, target, trans))
    });
    let Some((init, target, trans)) = build else {
        return declined("BDD build phase declined (budget/deadline/unresolvable var/cancel)");
    };

    if cancelled.load(Ordering::Relaxed) {
        return declined("cancelled before fixpoint");
    }

    // ---- Exact forward fixpoint with early bad-exit ----
    let outcome =
        bdd.reachable_target_within(init, trans, &quantify, &current, &next, target, deadline);
    match outcome {
        None => declined("BDD fixpoint declined (node budget / memory / deadline)"),
        Some(ReachOutcome::TargetHit { depth }) => BddReachOutcome::BadReachable { depth },
        Some(ReachOutcome::Fixpoint { reached, .. }) => {
            // Inductive self-check — defense-in-depth so an engine bug can
            // only decline, never produce a wrong Safe. All three checks are
            // cheap relative to the fixpoint, and run under the same abort.
            let verified = tla_bdd::catch_abort(|| {
                let n2c: std::collections::HashMap<u32, u32> =
                    next.iter().copied().zip(current.iter().copied()).collect();
                let init_in = bdd.subset(init, reached);
                let img = bdd.post_image(trans, reached, &quantify, &n2c);
                let inductive = bdd.subset(img, reached);
                let bad_free = bdd.and(reached, target) == Bdd::FALSE;
                Some(init_in && inductive && bad_free)
            });
            match verified {
                Some(true) => BddReachOutcome::Safe,
                Some(false) => declined("BDD inductive self-check FAILED (engine bug guard)"),
                None => declined("BDD self-check declined (budget/deadline)"),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sat_types::Clause;
    use rustc_hash::FxHashMap;

    /// Hand-built Transys: an `n`-bit binary counter with an enable input.
    /// Latches b0..b_{n-1}, input `en`; when `en` is 1 the counter increments
    /// (wrapping), when 0 it holds. All latches reset to 0.
    ///
    /// next(b_i) = b_i XOR (en AND b_0 AND ... AND b_{i-1})   [carry chain]
    ///
    /// Built directly over Transys fields with explicit AND gates for the
    /// carry chain and the XOR (x ^ y = ¬(¬(x∧¬y) ∧ ¬(¬x∧y))).
    struct CounterBuilder {
        next_var: u32,
        and_defs: FxHashMap<Var, (Lit, Lit)>,
    }

    impl CounterBuilder {
        fn new(first_free: u32) -> Self {
            Self {
                next_var: first_free,
                and_defs: FxHashMap::default(),
            }
        }
        fn fresh_and(&mut self, a: Lit, b: Lit) -> Lit {
            let v = Var(self.next_var);
            self.next_var += 1;
            self.and_defs.insert(v, (a, b));
            v.lit()
        }
        fn xor(&mut self, a: Lit, b: Lit) -> Lit {
            // a^b = ¬(¬(a∧¬b) ∧ ¬(¬a∧b))
            let t1 = self.fresh_and(a, !b);
            let t2 = self.fresh_and(!a, b);
            let n = self.fresh_and(!t1, !t2);
            !n
        }
    }

    fn counter_transys(bits: usize, bad_at: Option<u64>) -> Transys {
        // Vars: 1..=bits are latches; bits+1 is the input; gates after that.
        let latch_vars: Vec<Var> = (1..=bits as u32).map(Var).collect();
        let input_var = Var(bits as u32 + 1);
        let mut cb = CounterBuilder::new(bits as u32 + 2);

        let mut next_state = FxHashMap::default();
        let mut carry: Lit = input_var.lit(); // en
        for (i, &lv) in latch_vars.iter().enumerate() {
            let bit = lv.lit();
            let nxt = cb.xor(bit, carry);
            next_state.insert(lv, nxt);
            if i + 1 < bits {
                carry = cb.fresh_and(carry, bit);
            }
        }

        // bad: counter value == bad_at (AND over bit polarities).
        let mut bad_lits = Vec::new();
        if let Some(val) = bad_at {
            let mut acc = Lit::TRUE;
            for (i, &lv) in latch_vars.iter().enumerate() {
                let want_one = (val >> i) & 1 == 1;
                let bit = if want_one { lv.lit() } else { !lv.lit() };
                acc = cb.fresh_and(acc, bit);
            }
            bad_lits.push(acc);
        }

        // All latches reset to 0.
        let init_clauses: Vec<Clause> = latch_vars
            .iter()
            .map(|&lv| Clause::unit(!lv.lit()))
            .collect();

        let max_var = cb.next_var - 1;
        Transys {
            max_var,
            num_latches: bits,
            num_inputs: 1,
            latch_vars,
            input_vars: vec![input_var],
            next_state,
            init_clauses,
            trans_clauses: Vec::new(), // unused by the BDD lane
            bad_lits,
            constraint_lits: Vec::new(),
            and_defs: cb.and_defs,
            internal_signals: Vec::new(),
        }
    }

    fn no_cancel() -> Arc<AtomicBool> {
        Arc::new(AtomicBool::new(false))
    }

    #[test]
    fn counter_reaches_max_at_exact_depth() {
        // 3-bit counter from 0: value 7 is first reached after 7 increments.
        let ts = counter_transys(3, Some(7));
        let out = bdd_reach_check(&ts, &BddReachConfig::default(), None, &no_cancel());
        assert_eq!(out, BddReachOutcome::BadReachable { depth: 7 });
    }

    #[test]
    fn counter_bad_at_init_is_depth_zero() {
        // bad = value 0 = the initial state.
        let ts = counter_transys(3, Some(0));
        let out = bdd_reach_check(&ts, &BddReachConfig::default(), None, &no_cancel());
        assert_eq!(out, BddReachOutcome::BadReachable { depth: 0 });
    }

    #[test]
    fn counter_without_bad_is_safe() {
        let ts = counter_transys(4, None);
        let out = bdd_reach_check(&ts, &BddReachConfig::default(), None, &no_cancel());
        assert_eq!(out, BddReachOutcome::Safe);
    }

    #[test]
    fn unreachable_bad_is_safe_exactly() {
        // 3-bit counter whose top bit's next-state is forced to 0 (a stuck
        // bit): value with bit2=1 is unreachable ⇒ Safe, and the fixpoint
        // must PROVE it (exactly), not merely not-find it.
        let mut ts = counter_transys(3, Some(4)); // bad = 0b100
                                                  // Stuck top bit: next(b2) = FALSE.
        let top = ts.latch_vars[2];
        ts.next_state.insert(top, Lit::FALSE);
        let out = bdd_reach_check(&ts, &BddReachConfig::default(), None, &no_cancel());
        assert_eq!(out, BddReachOutcome::Safe);
    }

    #[test]
    fn constraints_decline_fail_closed() {
        let mut ts = counter_transys(3, Some(7));
        ts.constraint_lits.push(ts.latch_vars[0].lit());
        let out = bdd_reach_check(&ts, &BddReachConfig::default(), None, &no_cancel());
        assert!(matches!(out, BddReachOutcome::Declined { .. }));
    }

    #[test]
    fn latch_cap_declines() {
        let ts = counter_transys(9, Some(1));
        let cfg = BddReachConfig {
            max_latches: 8,
            ..BddReachConfig::default()
        };
        let out = bdd_reach_check(&ts, &cfg, None, &no_cancel());
        assert!(matches!(out, BddReachOutcome::Declined { .. }));
    }

    #[test]
    fn tiny_node_budget_declines_not_wrong() {
        let ts = counter_transys(6, Some(63));
        let cfg = BddReachConfig {
            node_budget: Some(64), // absurdly small — must abort, not lie
            ..BddReachConfig::default()
        };
        let out = bdd_reach_check(&ts, &cfg, None, &no_cancel());
        assert!(matches!(out, BddReachOutcome::Declined { .. }));
    }

    #[test]
    fn cyclic_and_defs_declines_instead_of_hanging() {
        // Malformed circuit: gate g depends on itself through h. A well-formed
        // AIG is acyclic; the lowering's fuel guard must decline, not spin.
        let mut ts = counter_transys(2, None);
        let g = Var(100);
        let h = Var(101);
        ts.and_defs.insert(g, (h.lit(), ts.latch_vars[0].lit()));
        ts.and_defs.insert(h, (g.lit(), ts.latch_vars[1].lit()));
        ts.next_state.insert(ts.latch_vars[0], g.lit());
        let out = bdd_reach_check(&ts, &BddReachConfig::default(), None, &no_cancel());
        assert!(matches!(out, BddReachOutcome::Declined { .. }));
    }

    #[test]
    fn pre_set_cancellation_declines_immediately() {
        // The cancelled flag aborts the lane in-operation (mk + fixpoint round
        // checks), not just at the between-phase polls.
        let ts = counter_transys(6, Some(63));
        let cancelled = Arc::new(AtomicBool::new(true));
        let out = bdd_reach_check(&ts, &BddReachConfig::default(), None, &cancelled);
        assert!(matches!(out, BddReachOutcome::Declined { .. }));
    }

    #[test]
    fn expired_deadline_declines() {
        let ts = counter_transys(5, Some(31));
        let out = bdd_reach_check(
            &ts,
            &BddReachConfig::default(),
            Some(Instant::now() - std::time::Duration::from_secs(1)),
            &no_cancel(),
        );
        assert!(matches!(out, BddReachOutcome::Declined { .. }));
    }
}
