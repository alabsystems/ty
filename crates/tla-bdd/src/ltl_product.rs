// Copyright 2026 Andrew Yates
// Licensed under the Apache License, Version 2.0

//! Native symbolic LTL emptiness (system × GBA accepting-cycle) on `tla-bdd`.
//!
//! Faithful port of `tla_dd::symbolic_ltl` (the oxidd version) onto the native
//! ROBDD engine, so the LTL lane can drop oxidd. Given a bounded net and a GBA
//! for the NEGATED property, [`ltl_has_accepting_run`] answers: does
//! `system × GBA` have a reachable accepting cycle (a fair lasso)?
//!
//! Conventions matched EXACTLY to the explicit `tla_petri::buchi::on_the_fly`
//! checker (the production reference):
//! - augmented net = system places + one synthetic GBA-state "qbit" place
//!   (bound `num_states-1`);
//! - product edge = fire a system transition OR the DEADLOCK STUTTER self-loop
//!   (`sys_deadlock ∧ next_sys = cur_sys`, maximal-path semantics), then a GBA
//!   edge whose guard holds at the SUCCESSOR marking (GPVW next-side guards);
//! - generalized Büchi with mixed state- + edge-based acceptance;
//! - emptiness = Emerson–Lei fair-states gfp then backward reach from initial.
//!
//! Differential-tested to 0 disagreements vs an independent brute-force explicit
//! product-SCC oracle (below), including deadlock nets and edge-acceptance GBAs.

use crate::petri::{BoundedNet, Pred};
use crate::{Bdd, NodeId};
use std::collections::HashMap as StdHashMap;
use std::time::Instant;

fn bits_for(bound: u64) -> usize {
    (64 - bound.leading_zeros()).max(1) as usize
}

/// A GBA transition lowered for the symbolic product. Guard = ⋀ pos hold ∧
/// ⋀ neg ¬hold, evaluated on the SUCCESSOR marking (GPVW). `edge_accept[i]` ⇔
/// this edge discharges acceptance set `i`.
#[derive(Debug, Clone)]
pub struct LtlGbaTransition {
    /// Atom indices that must hold at the successor marking.
    pub pos_atoms: Vec<usize>,
    /// Atom indices that must NOT hold at the successor marking.
    pub neg_atoms: Vec<usize>,
    /// The GBA state entered by taking this edge.
    pub successor: u32,
    /// `edge_accept[i]` ⇔ taking this edge discharges acceptance set `i`.
    pub edge_accept: Vec<bool>,
}

/// A Generalized Büchi Automaton for the negated property (engine-agnostic;
/// mirror of `buchi::gba::Gba` / `tla_dd::symbolic_ltl::SymbolicGba`).
#[derive(Clone)]
pub struct LtlGba {
    /// Number of GBA states (states are `0..num_states`).
    pub num_states: u32,
    /// Atom predicates over the SYSTEM marking, referenced by index.
    pub atoms: Vec<Pred>,
    /// Initial transitions (guards read the INITIAL marking).
    pub initial_transitions: Vec<LtlGbaTransition>,
    /// `transitions[q]` = outgoing edges of state q (guards read the successor).
    pub transitions: Vec<Vec<LtlGbaTransition>>,
    /// State-based acceptance: `acceptance[i]` = the accepting states for set i.
    pub acceptance: Vec<Vec<u32>>,
}

/// Does `net × gba` have a reachable accepting cycle? `Some(true)` = yes (the
/// property `A(φ)` is violated). `None` = decline (fail-closed) — an empty
/// automaton or (via [`ltl_has_accepting_run_within`]) the budget was exceeded.
#[must_use]
pub fn ltl_has_accepting_run(net: &BoundedNet, gba: &LtlGba) -> Option<bool> {
    ltl_has_accepting_run_within(net, gba, None)
}

/// Deadline-aware [`ltl_has_accepting_run`]: every symbolic fixpoint (product
/// reachability, the Emerson–Lei fair-states gfp, the confined `E[·U·]`, and the
/// backward reach) polls `deadline` at each iteration and returns `None`
/// (fail-closed DECLINE) the moment it is exceeded — so the caller falls through
/// to the explicit lane rather than overrunning its budget. `None` deadline runs
/// to convergence.
#[must_use]
pub fn ltl_has_accepting_run_within(
    net: &BoundedNet,
    gba: &LtlGba,
    deadline: Option<Instant>,
) -> Option<bool> {
    if gba.num_states == 0 {
        return Some(false); // empty automaton ⇒ no accepting run
    }
    // `catch_abort` folds an in-operation `BddAbort` (node budget or
    // mid-round deadline, audit 2026-07-02) into the same fail-closed
    // decline as the per-round deadline polls.
    crate::catch_abort(move || {
        let prod = Prod::build(net, gba, deadline)?;
        prod.has_accepting_cycle()
    })
}

#[inline]
fn expired(deadline: Option<Instant>) -> bool {
    deadline.is_some_and(|d| Instant::now() >= d)
}

/// The symbolic product, all BDDs over one manager. Variable layout: system
/// places `0..S` then the qbit place `S`; per place `p`, cur bits at
/// `offset[p]+i`, next bits at `total+offset[p]+i`.
struct Prod {
    b: Bdd,
    /// current→next rename map and its inverse.
    c2n: StdHashMap<u32, u32>,
    next_vars: Vec<u32>,
    trans_rel: NodeId,
    edge_accept_rel: Vec<NodeId>,
    initial: NodeId,
    reachable: NodeId,
    num_accept: usize,
    /// state-based acceptance chars: `state_acc_chars[i]` = qbit ∈ acceptance[i].
    state_acc_chars: Vec<NodeId>,
    /// wall-clock budget for the emptiness fixpoints (`None` ⇒ unbounded).
    deadline: Option<Instant>,
}

impl Prod {
    fn build(net: &BoundedNet, gba: &LtlGba, deadline: Option<Instant>) -> Option<Prod> {
        let ns = net.bounds.len();
        // Augmented bounds: system + qbit.
        let qbound = u64::from(gba.num_states - 1);
        let mut bounds = net.bounds.clone();
        bounds.push(qbound);
        let np = bounds.len();
        let qplace = ns;
        let widths: Vec<usize> = bounds.iter().map(|&x| bits_for(x)).collect();
        let mut offset = vec![0usize; np];
        let mut acc = 0usize;
        for p in 0..np {
            offset[p] = acc;
            acc += widths[p];
        }
        let total = acc;
        let cur_bit = |p: usize, i: usize| (offset[p] + i) as u32;
        let nxt_bit = |p: usize, i: usize| (total + offset[p] + i) as u32;
        let mut b = Bdd::new();
        // Cooperative abort (audit 2026-07-02): bounds product construction
        // and every downstream fixpoint on this manager; the entry point
        // (`ltl_has_accepting_run_within`) folds the abort into a decline.
        b.set_abort_limits(Some(crate::default_abort_node_budget()), deadline);

        // place p == v on the cur (in_next=false) or next side.
        let value = |b: &mut Bdd, p: usize, v: u64, in_next: bool| -> NodeId {
            let mut a = Bdd::TRUE;
            for i in 0..widths[p] {
                let var = if in_next {
                    nxt_bit(p, i)
                } else {
                    cur_bit(p, i)
                };
                let vb = b.var(var);
                let lit = if (v >> i) & 1 == 1 { vb } else { b.not(vb) };
                a = b.and(a, lit);
            }
            a
        };

        // charset of a Pred over the SYSTEM places, on cur (in_next=false) or
        // next side. Pred atoms reference system places 0..ns only.
        fn pred_charset(
            b: &mut Bdd,
            net: &BoundedNet,
            p: &Pred,
            in_next: bool,
            value: &dyn Fn(&mut Bdd, usize, u64, bool) -> NodeId,
            widths: &[usize],
            offset: &[usize],
            total: usize,
        ) -> NodeId {
            match p {
                Pred::Fireable(t) => {
                    let tr = &net.transitions[*t];
                    let mut g = Bdd::TRUE;
                    for pl in 0..net.bounds.len() {
                        let mut gp = Bdd::FALSE;
                        for v in tr.pre[pl]..=net.bounds[pl] {
                            let cv = value(b, pl, v, in_next);
                            gp = b.or(gp, cv);
                        }
                        g = b.and(g, gp);
                    }
                    g
                }
                Pred::TokenLe { coeffs, k } => {
                    let mut terms: Vec<(u32, i128)> = Vec::new();
                    for (pl, &c) in coeffs.iter().enumerate() {
                        for i in 0..widths[pl] {
                            let bit = if in_next {
                                (total + offset[pl] + i) as u32
                            } else {
                                (offset[pl] + i) as u32
                            };
                            terms.push((bit, c * (1i128 << i)));
                        }
                    }
                    terms.sort_by_key(|&(v, _)| v);
                    b.linear_le(&terms, *k)
                }
                Pred::And(cs) => {
                    let mut a = Bdd::TRUE;
                    for c in cs {
                        let cc = pred_charset(b, net, c, in_next, value, widths, offset, total);
                        a = b.and(a, cc);
                    }
                    a
                }
                Pred::Or(cs) => {
                    let mut o = Bdd::FALSE;
                    for c in cs {
                        let cc = pred_charset(b, net, c, in_next, value, widths, offset, total);
                        o = b.or(o, cc);
                    }
                    o
                }
                Pred::Not(c) => {
                    let cc = pred_charset(b, net, c, in_next, value, widths, offset, total);
                    b.not(cc)
                }
            }
        }

        // guard(e) on a given side: ⋀ pos ∧ ⋀ ¬neg over gba.atoms.
        let guard = |b: &mut Bdd, e: &LtlGbaTransition, in_next: bool| -> NodeId {
            let mut g = Bdd::TRUE;
            for &ai in &e.pos_atoms {
                let a = pred_charset(
                    b,
                    net,
                    &gba.atoms[ai],
                    in_next,
                    &value,
                    &widths,
                    &offset,
                    total,
                );
                g = b.and(g, a);
            }
            for &ai in &e.neg_atoms {
                let a = pred_charset(
                    b,
                    net,
                    &gba.atoms[ai],
                    in_next,
                    &value,
                    &widths,
                    &offset,
                    total,
                );
                let na = b.not(a);
                g = b.and(g, na);
            }
            g
        };

        // T_sys(cur_sys, next_sys): system transition relation over system
        // places; qbit bits unconstrained. Plus deadlock stutter.
        // has_succ(cur_sys) = ∃next_sys. T_sys. sys_identity: cur_sys==next_sys.
        let mut t_sys = Bdd::FALSE;
        for t in &net.transitions {
            let mut rel = Bdd::TRUE;
            for p in 0..ns {
                let mut rel_p = Bdd::FALSE;
                for v in 0..=net.bounds[p] {
                    if v < t.pre[p] || v - t.pre[p] + t.post[p] > net.bounds[p] {
                        continue;
                    }
                    let cv = value(&mut b, p, v, false);
                    let nvb = value(&mut b, p, v - t.pre[p] + t.post[p], true);
                    let pair = b.and(cv, nvb);
                    rel_p = b.or(rel_p, pair);
                }
                rel = b.and(rel, rel_p);
            }
            t_sys = b.or(t_sys, rel);
        }
        // system-side next vars (for the has_succ projection).
        let mut sys_next: Vec<u32> = Vec::new();
        for p in 0..ns {
            for i in 0..widths[p] {
                sys_next.push((total + offset[p] + i) as u32);
            }
        }
        let has_succ = b.exists(t_sys, &sys_next);
        let no_succ = b.not(has_succ);
        // sys_identity over system places: cur_sys == next_sys.
        let mut sys_id = Bdd::TRUE;
        for p in 0..ns {
            for i in 0..widths[p] {
                let cv = b.var(cur_bit(p, i));
                let nv = b.var(nxt_bit(p, i));
                let eq = {
                    let a = b.and(cv, nv);
                    let ncv = b.not(cv);
                    let nnv = b.not(nv);
                    let a2 = b.and(ncv, nnv);
                    b.or(a, a2)
                };
                sys_id = b.and(sys_id, eq);
            }
        }
        let stutter = b.and(no_succ, sys_id);
        let t_sys_or_dead = b.or(t_sys, stutter);

        // T_prod = ⋁_q [cur.qbit=q] ∧ ⋁_{e∈trans[q]} t_sys_or_dead ∧ guard(e)(next)
        //          ∧ [next.qbit=e.successor].
        // edge_accept_rel[i] = the same but only e with edge_accept[i].
        let num_accept = gba.acceptance.len();
        let mut trans_rel = Bdd::FALSE;
        let mut edge_accept_rel = vec![Bdd::FALSE; num_accept];
        for q in 0..gba.num_states as usize {
            let cur_q = value(&mut b, qplace, q as u64, false);
            for e in &gba.transitions[q] {
                let g = guard(&mut b, e, true); // successor-side guard
                let nq = value(&mut b, qplace, u64::from(e.successor), true);
                let mut edge = b.and(cur_q, t_sys_or_dead);
                edge = b.and(edge, g);
                edge = b.and(edge, nq);
                trans_rel = b.or(trans_rel, edge);
                for i in 0..num_accept {
                    if e.edge_accept.get(i).copied().unwrap_or(false) {
                        edge_accept_rel[i] = b.or(edge_accept_rel[i], edge);
                    }
                }
            }
        }

        // initial: ⋁_{e∈initial_transitions, guard holds at init}
        //          [cur_sys=init] ∧ [cur.qbit=e.successor].
        // Initial guards read the INITIAL marking (cur side, at the init values).
        let mut init_sys = Bdd::TRUE;
        for p in 0..ns {
            let m = value(&mut b, p, net.init[p], false);
            init_sys = b.and(init_sys, m);
        }
        let mut initial = Bdd::FALSE;
        for e in &gba.initial_transitions {
            // Evaluate guard on the concrete initial marking (cur side); it is a
            // function of cur_sys, so AND with init_sys pins it to the init value
            // — equivalent to a concrete check.
            let g = guard(&mut b, e, false);
            let holds = b.and(init_sys, g); // init_sys ∧ guard(cur) ; nonempty iff holds at init
            if holds == Bdd::FALSE {
                continue;
            }
            let nq = value(&mut b, qplace, u64::from(e.successor), false);
            let st = b.and(init_sys, nq);
            initial = b.or(initial, st);
        }

        // rename map cur→next over all product vars.
        let c2n: StdHashMap<u32, u32> = (0..total as u32).map(|v| (v, total as u32 + v)).collect();
        let next_vars: Vec<u32> = (total as u32..2 * total as u32).collect();

        // reachable product set: forward fixpoint from initial via trans_rel.
        let reachable = {
            let mut r = initial;
            let mut frontier = initial;
            loop {
                if expired(deadline) {
                    return None;
                }
                let img_next =
                    b.and_exists(frontier, trans_rel, &(0..total as u32).collect::<Vec<_>>());
                let img = b.rename(img_next, &{
                    let mut m = StdHashMap::default();
                    for v in 0..total as u32 {
                        m.insert(total as u32 + v, v);
                    }
                    m
                });
                let not_r = b.not(r);
                let new = b.and(img, not_r);
                if new == Bdd::FALSE {
                    break r;
                }
                r = b.or(r, new);
                frontier = new;
            }
        };

        // state-based acceptance chars: qbit ∈ acceptance[i], on the cur side.
        let mut state_acc_chars = Vec::with_capacity(num_accept);
        for set in &gba.acceptance {
            let mut c = Bdd::FALSE;
            for &q in set {
                let qc = value(&mut b, qplace, u64::from(q), false);
                c = b.or(c, qc);
            }
            state_acc_chars.push(c);
        }

        Some(Prod {
            b,
            c2n,
            next_vars,
            trans_rel,
            edge_accept_rel,
            initial,
            reachable,
            num_accept,
            state_acc_chars,
            deadline,
        })
    }

    /// pre_e(S) = ∃next. T_prod ∧ S[cur→next] — states with a T_prod successor in S.
    fn pre_e(&mut self, s: NodeId) -> NodeId {
        let s_next = self.b.rename(s, &self.c2n);
        self.b.and_exists(self.trans_rel, s_next, &self.next_vars)
    }

    /// pre_e_in(S, y) = pre_e(S) ∧ y.
    fn pre_e_in(&mut self, s: NodeId, y: NodeId) -> NodeId {
        let p = self.pre_e(s);
        self.b.and(p, y)
    }

    /// E[y U target] confined to y: μZ. target ∨ (y ∧ pre_e(Z)).
    /// `None` on budget exhaustion.
    fn eu_within(&mut self, y: NodeId, target: NodeId) -> Option<NodeId> {
        let mut z = target;
        loop {
            if expired(self.deadline) {
                return None;
            }
            let pre = self.pre_e(z);
            let step = self.b.and(y, pre);
            let nz = self.b.or(target, step);
            if nz == z {
                return Some(z);
            }
            z = nz;
        }
    }

    /// pre via a sub-relation `rel`, into `region`, staying in `region`:
    /// region ∧ ∃next. (rel ∧ region[next]).
    fn pre_via(&mut self, rel: NodeId, region: NodeId) -> NodeId {
        let region_next = self.b.rename(region, &self.c2n);
        let step = self.b.and_exists(rel, region_next, &self.next_vars);
        self.b.and(region, step)
    }

    /// accept_step(i, region) = (qbit∈acc[i] ∧ region) ∨ edge-accepting move into region.
    fn accept_step(&mut self, i: usize, region: NodeId, state_acc: &[NodeId]) -> NodeId {
        let f_state = self.b.and(state_acc[i], region);
        let edge_pre = self.pre_via(self.edge_accept_rel[i], region);
        self.b.or(f_state, edge_pre)
    }

    /// fair_states: νY. ⋀_i EX_Y(E[Y U (Y ∧ accept_step_i(Y))]), starting at
    /// reachable. `None` on budget exhaustion.
    fn fair_states(&mut self, state_acc: &[NodeId]) -> Option<NodeId> {
        let mut y = self.reachable;
        loop {
            if expired(self.deadline) {
                return None;
            }
            if self.num_accept == 0 {
                let ex = self.pre_e_in(y, y);
                let next = self.b.and(y, ex);
                if next == y {
                    return Some(y);
                }
                y = next;
                continue;
            }
            let mut conj = y;
            for i in 0..self.num_accept {
                let target = self.accept_step(i, y, state_acc);
                let reach_target = self.eu_within(y, target)?;
                let ex = self.pre_e_in(reach_target, y);
                conj = self.b.and(conj, ex);
            }
            if conj == y {
                return Some(y);
            }
            y = conj;
        }
    }

    /// `Some(true)`/`Some(false)` = sound verdict; `None` = budget exhausted.
    fn has_accepting_cycle(mut self) -> Option<bool> {
        if self.initial == Bdd::FALSE {
            return Some(false);
        }
        let state_acc = self.state_acc_chars.clone();
        let fair = self.fair_states(&state_acc)?;
        if fair == Bdd::FALSE {
            return Some(false);
        }
        // backward reach from fair, confined to reachable.
        let can_reach_fair = {
            let mut z = self.b.and(fair, self.reachable);
            loop {
                if expired(self.deadline) {
                    return None;
                }
                let pre = self.pre_e(z);
                let nz = self.b.or(z, pre);
                if nz == z {
                    break z;
                }
                z = nz;
            }
        };
        let hit = self.b.and(self.initial, can_reach_fair);
        Some(hit != Bdd::FALSE)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::petri::BoundedTransition;
    use std::collections::HashMap;

    // ---- Independent brute-force explicit product-SCC oracle -------------
    // A completely separate algorithm (explicit graph + O(n²) mutual-reach
    // SCC) from the symbolic gfp, encoding the SAME conventions literally:
    // deadlock-stutter, next-side (GPVW) guards, init guards at init, mixed
    // state+edge generalized-Büchi acceptance.

    fn eval_pred(net: &BoundedNet, p: &Pred, m: &[u64]) -> bool {
        match p {
            Pred::Fireable(t) => {
                let tr = &net.transitions[*t];
                (0..net.bounds.len()).all(|pl| m[pl] >= tr.pre[pl])
            }
            Pred::TokenLe { coeffs, k } => {
                let s: i128 = coeffs
                    .iter()
                    .enumerate()
                    .map(|(pl, &c)| c * i128::from(m[pl] as u64))
                    .sum();
                s <= *k
            }
            Pred::And(cs) => cs.iter().all(|c| eval_pred(net, c, m)),
            Pred::Or(cs) => cs.iter().any(|c| eval_pred(net, c, m)),
            Pred::Not(c) => !eval_pred(net, c, m),
        }
    }

    fn guard_holds(net: &BoundedNet, gba: &LtlGba, e: &LtlGbaTransition, m: &[u64]) -> bool {
        e.pos_atoms
            .iter()
            .all(|&a| eval_pred(net, &gba.atoms[a], m))
            && e.neg_atoms
                .iter()
                .all(|&a| !eval_pred(net, &gba.atoms[a], m))
    }

    fn sys_successors(net: &BoundedNet, m: &[u64]) -> Vec<Vec<u64>> {
        let np = net.bounds.len();
        let mut succ: Vec<Vec<u64>> = Vec::new();
        for tr in &net.transitions {
            let enabled = (0..np)
                .all(|p| m[p] >= tr.pre[p] && m[p] - tr.pre[p] + tr.post[p] <= net.bounds[p]);
            if enabled {
                let m2: Vec<u64> = (0..np).map(|p| m[p] - tr.pre[p] + tr.post[p]).collect();
                if !succ.contains(&m2) {
                    succ.push(m2);
                }
            }
        }
        succ
    }

    fn brute_force_has_accepting_run(net: &BoundedNet, gba: &LtlGba) -> bool {
        if gba.num_states == 0 {
            return false;
        }
        let num_accept = gba.acceptance.len();
        // Reachable product graph.  node = (marking, qbit).
        let mut nodes: Vec<(Vec<u64>, u32)> = Vec::new();
        let mut idx: HashMap<(Vec<u64>, u32), usize> = HashMap::new();
        let mut adj: Vec<Vec<(usize, Vec<bool>)>> = Vec::new();
        let mut stack: Vec<usize> = Vec::new();

        // Initial product states: init marking × successor of each init edge
        // whose guard holds at the initial marking.
        for e in &gba.initial_transitions {
            if guard_holds(net, gba, e, &net.init) {
                let key = (net.init.clone(), e.successor);
                if !idx.contains_key(&key) {
                    let i = nodes.len();
                    nodes.push(key.clone());
                    adj.push(Vec::new());
                    idx.insert(key, i);
                    stack.push(i);
                }
            }
        }
        let init_ids: Vec<usize> = (0..nodes.len()).collect();

        while let Some(u) = stack.pop() {
            let (m, q) = nodes[u].clone();
            let ss = sys_successors(net, &m);
            let succ_markings: Vec<Vec<u64>> = if ss.is_empty() { vec![m.clone()] } else { ss };
            for e in &gba.transitions[q as usize] {
                for m2 in &succ_markings {
                    if guard_holds(net, gba, e, m2) {
                        let key = (m2.clone(), e.successor);
                        let v = match idx.get(&key) {
                            Some(&i) => i,
                            None => {
                                let i = nodes.len();
                                nodes.push(key.clone());
                                adj.push(Vec::new());
                                idx.insert(key, i);
                                stack.push(i);
                                i
                            }
                        };
                        adj[u].push((v, e.edge_accept.clone()));
                    }
                }
            }
        }

        let n = nodes.len();
        if n == 0 {
            return false;
        }
        // forward[u][v] = v reachable from u in >=1 steps.
        let mut forward = vec![vec![false; n]; n];
        for u in 0..n {
            let mut st: Vec<usize> = adj[u].iter().map(|&(w, _)| w).collect();
            for &w in &st.clone() {
                forward[u][w] = true;
            }
            while let Some(x) = st.pop() {
                for &(w, _) in &adj[x] {
                    if !forward[u][w] {
                        forward[u][w] = true;
                        st.push(w);
                    }
                }
            }
        }

        // A reachable cyclic SCC that is generalized-Büchi fair ⇒ accepting run.
        for u in 0..n {
            // component of u under mutual reachability.
            let comp: Vec<usize> = (0..n)
                .filter(|&v| v == u || (forward[u][v] && forward[v][u]))
                .collect();
            let cyclic = comp.len() > 1 || forward[u][u];
            if !cyclic {
                continue;
            }
            let mut all_ok = true;
            for i in 0..num_accept {
                let state_ok = comp
                    .iter()
                    .any(|&nd| gba.acceptance[i].contains(&nodes[nd].1));
                let edge_ok = comp.iter().any(|&a| {
                    adj[a]
                        .iter()
                        .any(|(bb, ea)| comp.contains(bb) && ea.get(i).copied().unwrap_or(false))
                });
                if !(state_ok || edge_ok) {
                    all_ok = false;
                    break;
                }
            }
            if all_ok {
                // reachable from an initial product state (built that way).
                let _ = &init_ids;
                return true;
            }
        }
        false
    }

    // ---- Deterministic PRNG (splitmix64) --------------------------------
    struct Rng(u64);
    impl Rng {
        fn next_u64(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }
        fn below(&mut self, n: u64) -> u64 {
            if n == 0 {
                0
            } else {
                self.next_u64() % n
            }
        }
        fn flip(&mut self) -> bool {
            self.next_u64() & 1 == 1
        }
    }

    fn gen_edge(
        rng: &mut Rng,
        natoms: usize,
        num_states: u32,
        num_accept: usize,
    ) -> LtlGbaTransition {
        let mut pos = Vec::new();
        let mut neg = Vec::new();
        for a in 0..natoms {
            match rng.below(3) {
                0 => pos.push(a),
                1 => neg.push(a),
                _ => {}
            }
        }
        let successor = rng.below(u64::from(num_states)) as u32;
        let edge_accept = (0..num_accept).map(|_| rng.flip()).collect();
        LtlGbaTransition {
            pos_atoms: pos,
            neg_atoms: neg,
            successor,
            edge_accept,
        }
    }

    fn gen_case(seed: u64) -> (BoundedNet, LtlGba) {
        let mut rng = Rng(seed ^ 0xDEAD_BEEF_CAFE_F00D);
        let nplaces = 1 + rng.below(3) as usize; // 1..=3
        let bounds: Vec<u64> = (0..nplaces).map(|_| 1 + rng.below(2)).collect(); // 1..=2
        let init: Vec<u64> = bounds.iter().map(|&b| rng.below(b + 1)).collect();
        let ntrans = rng.below(4) as usize; // 0..=3
        let transitions: Vec<BoundedTransition> = (0..ntrans)
            .map(|_| {
                let pre = (0..nplaces).map(|_| rng.below(2)).collect(); // 0..=1
                let post = (0..nplaces).map(|_| rng.below(3)).collect(); // 0..=2
                BoundedTransition { pre, post }
            })
            .collect();
        let net = BoundedNet {
            bounds,
            init,
            transitions,
        };

        let natoms = 1 + rng.below(2) as usize; // 1..=2
        let atoms: Vec<Pred> = (0..natoms)
            .map(|_| {
                if ntrans > 0 && rng.flip() {
                    Pred::Fireable(rng.below(ntrans as u64) as usize)
                } else {
                    let coeffs: Vec<i128> =
                        (0..nplaces).map(|_| i128::from(rng.below(3)) - 1).collect(); // -1..=1
                    let k = i128::from(rng.below(4)) - 1; // -1..=2
                    Pred::TokenLe { coeffs, k }
                }
            })
            .collect();

        let num_states = 1 + rng.below(3) as u32; // 1..=3
        let num_accept = rng.below(3) as usize; // 0..=2
        let acceptance: Vec<Vec<u32>> = (0..num_accept)
            .map(|_| (0..num_states).filter(|_| rng.flip()).collect())
            .collect();
        let initial_transitions: Vec<LtlGbaTransition> = (0..(1 + rng.below(2)))
            .map(|_| gen_edge(&mut rng, natoms, num_states, num_accept))
            .collect();
        let transitions_gba: Vec<Vec<LtlGbaTransition>> = (0..num_states)
            .map(|_| {
                (0..rng.below(3))
                    .map(|_| gen_edge(&mut rng, natoms, num_states, num_accept))
                    .collect()
            })
            .collect();
        let gba = LtlGba {
            num_states,
            atoms,
            initial_transitions,
            transitions: transitions_gba,
            acceptance,
        };
        (net, gba)
    }

    #[test]
    fn differential_symbolic_vs_bruteforce_zero_disagreements() {
        let mut disagreements = 0usize;
        let mut first_bad: Option<u64> = None;
        let iters = 20_000u64;
        for seed in 0..iters {
            let (net, gba) = gen_case(seed);
            let symbolic = ltl_has_accepting_run(&net, &gba).expect("small bounds never decline");
            let brute = brute_force_has_accepting_run(&net, &gba);
            if symbolic != brute {
                disagreements += 1;
                if first_bad.is_none() {
                    first_bad = Some(seed);
                }
            }
        }
        assert_eq!(
            disagreements, 0,
            "symbolic LTL port disagreed with brute-force oracle on {disagreements}/{iters} cases; first failing seed = {first_bad:?}"
        );
    }

    // ---- Hand-checked cases guarding against a shared-convention bug -----

    fn always_true_atom() -> Pred {
        // 0 <= 0 for a single place: holds at every marking.
        Pred::TokenLe {
            coeffs: vec![0],
            k: 0,
        }
    }

    #[test]
    fn edge_accepting_self_loop_is_accepting() {
        // 1 place, bound 1, init 0; one always-enabled self-loop transition.
        let net = BoundedNet {
            bounds: vec![1],
            init: vec![0],
            transitions: vec![BoundedTransition {
                pre: vec![0],
                post: vec![0],
            }],
        };
        let e = LtlGbaTransition {
            pos_atoms: vec![0],
            neg_atoms: vec![],
            successor: 0,
            edge_accept: vec![true],
        };
        let gba = LtlGba {
            num_states: 1,
            atoms: vec![always_true_atom()],
            initial_transitions: vec![e.clone()],
            transitions: vec![vec![e]],
            acceptance: vec![vec![]], // one set; satisfied only by the accepting edge
        };
        assert_eq!(ltl_has_accepting_run(&net, &gba), Some(true));
        assert!(brute_force_has_accepting_run(&net, &gba));
    }

    #[test]
    fn no_accepting_edge_or_state_is_rejecting() {
        let net = BoundedNet {
            bounds: vec![1],
            init: vec![0],
            transitions: vec![BoundedTransition {
                pre: vec![0],
                post: vec![0],
            }],
        };
        let e = LtlGbaTransition {
            pos_atoms: vec![0],
            neg_atoms: vec![],
            successor: 0,
            edge_accept: vec![false],
        };
        let gba = LtlGba {
            num_states: 1,
            atoms: vec![always_true_atom()],
            initial_transitions: vec![e.clone()],
            transitions: vec![vec![e]],
            acceptance: vec![vec![]], // one set, never satisfied
        };
        assert_eq!(ltl_has_accepting_run(&net, &gba), Some(false));
        assert!(!brute_force_has_accepting_run(&net, &gba));
    }

    #[test]
    fn deadlock_stutter_carries_accepting_edge() {
        // No transitions ⇒ init is a system deadlock; the stutter self-loop
        // must still carry the accepting GBA edge.
        let net = BoundedNet {
            bounds: vec![1],
            init: vec![0],
            transitions: vec![],
        };
        let e = LtlGbaTransition {
            pos_atoms: vec![0],
            neg_atoms: vec![],
            successor: 0,
            edge_accept: vec![true],
        };
        let gba = LtlGba {
            num_states: 1,
            atoms: vec![always_true_atom()],
            initial_transitions: vec![e.clone()],
            transitions: vec![vec![e]],
            acceptance: vec![vec![]],
        };
        assert_eq!(ltl_has_accepting_run(&net, &gba), Some(true));
        assert!(brute_force_has_accepting_run(&net, &gba));
    }

    #[test]
    fn state_accepting_cycle_is_accepting() {
        // Acceptance by STATE membership (qbit 0 accepting), no edge acceptance.
        let net = BoundedNet {
            bounds: vec![1],
            init: vec![0],
            transitions: vec![BoundedTransition {
                pre: vec![0],
                post: vec![0],
            }],
        };
        let e = LtlGbaTransition {
            pos_atoms: vec![0],
            neg_atoms: vec![],
            successor: 0,
            edge_accept: vec![false],
        };
        let gba = LtlGba {
            num_states: 1,
            atoms: vec![always_true_atom()],
            initial_transitions: vec![e.clone()],
            transitions: vec![vec![e]],
            acceptance: vec![vec![0]], // state 0 is accepting
        };
        assert_eq!(ltl_has_accepting_run(&net, &gba), Some(true));
        assert!(brute_force_has_accepting_run(&net, &gba));
    }
}
