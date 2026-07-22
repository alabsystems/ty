// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Product graph emptiness checking: system × GBA accepting cycle detection.
//!
//! Uses both state-based and edge-based acceptance to correctly handle
//! `G(X(F(...)))` patterns where the `Release` operator re-introduces
//! `Until` obligations at every step.

use rustc_hash::FxHashMap;
use std::time::Instant;

use super::atoms::LtlContext;
use super::gba::{accept_bit, AcceptanceMasks, Gba, GbaStateId};
use super::on_the_fly::CsrProductAdj;
use crate::scc::tarjan_scc_slices;

const PRODUCT_STATE_LIMIT: usize = 50_000_000;

/// Check if the product of system × GBA has an accepting cycle reachable from
/// the initial system state.
///
/// Returns `Some(true)` if an accepting cycle exists, `Some(false)` if none,
/// or `None` if the product graph exceeded the size limit or deadline.
pub(super) fn product_has_accepting_cycle(
    gba: &Gba,
    ctx: &LtlContext<'_>,
    deadline: Option<Instant>,
) -> Option<bool> {
    product_has_accepting_cycle_impl(gba, ctx, PRODUCT_STATE_LIMIT, deadline)
}

#[cfg(test)]
pub(super) fn product_has_accepting_cycle_with_limit(
    gba: &Gba,
    ctx: &LtlContext<'_>,
    product_state_limit: usize,
) -> Option<bool> {
    product_has_accepting_cycle_impl(gba, ctx, product_state_limit, None)
}

fn product_has_accepting_cycle_impl(
    gba: &Gba,
    ctx: &LtlContext<'_>,
    product_state_limit: usize,
    deadline: Option<Instant>,
) -> Option<bool> {
    if gba.num_states == 0 || ctx.full.graph.num_states == 0 {
        return Some(false);
    }

    // No acceptance sets means any cycle is accepting.
    let num_accept = gba.acceptance.len();
    // Packed per-GBA-state / per-GBA-edge acceptance words (audit S3):
    // per-product-state and per-product-edge acceptance is stored as
    // `num_words`-strided u64 words, not per-state/per-edge `Vec<bool>`s.
    let acc = AcceptanceMasks::from_gba(gba);
    let nw = acc.num_words;

    // Build the product graph on-the-fly and check for accepting SCCs.
    // Product state = (system_state, gba_state).
    let mut product_ids: FxHashMap<(u32, GbaStateId), u32> = FxHashMap::default();
    let mut product_adj: Vec<Vec<u32>> = Vec::new();
    // Packed state acceptance, strided: words of pid = [pid*nw, (pid+1)*nw).
    let mut product_accept: Vec<u64> = Vec::new();
    // Edge-based acceptance: per product state, the outgoing edges in
    // generation order (`edge_succ`, NOT deduped) parallel to their packed
    // edge_accept words (`edge_words`, strided by `nw`).
    let mut product_edge_succ: Vec<Vec<u32>> = Vec::new();
    let mut product_edge_words: Vec<Vec<u64>> = Vec::new();
    let mut queue: std::collections::VecDeque<u32> = std::collections::VecDeque::new();

    // Initial product states: check each initial GBA transition's guard
    // against system state 0. Only add product states where guard is satisfied.
    for init_trans in &gba.initial_transitions {
        if init_trans.guard_satisfied(ctx, 0) {
            let gba_state = init_trans.successor;
            let product_key = (0u32, gba_state);
            if product_ids.contains_key(&product_key) {
                continue; // Already added
            }
            let pid = product_ids.len() as u32;
            product_ids.insert(product_key, pid);
            product_adj.push(Vec::new());
            product_edge_succ.push(Vec::new());
            product_edge_words.push(Vec::new());
            product_accept.extend_from_slice(acc.state(gba_state));
            queue.push_back(pid);
        }
    }

    // Reverse lookup: product_id → (sys_state, gba_state)
    let mut product_keys: Vec<(u32, GbaStateId)> = vec![(0, 0); product_ids.len()];
    for (&key, &pid) in &product_ids {
        product_keys[pid as usize] = key;
    }
    // Initial product states form a prefix of the id space (interned above).
    let num_roots = product_ids.len() as u32;

    // BFS to build product graph. One adaptive probe (deadline + memory): the
    // 50M-product-state item cap says nothing about bytes (nested adjacency
    // rows, edge lists, packed acceptance words, keys). `None` = inconclusive.
    let mut probe = crate::memory::explorer_probe(deadline);
    while let Some(pid) = queue.pop_front() {
        if probe.over_budget() {
            return None;
        }

        let (sys, gba_state) = product_keys[pid as usize];

        // Generation-order edge list (parallel to `edge_word_buf`); copied
        // into `product_edge_succ` before being sorted/deduped into the adjacency.
        let mut successors: Vec<u32> = Vec::new();
        let mut edge_word_buf: Vec<u64> = Vec::new();

        // For each system transition from sys
        let sys_succs: Vec<u32> = if ctx.full.graph.adj[sys as usize].is_empty() {
            vec![sys] // Deadlock self-loop
        } else {
            ctx.full.graph.adj[sys as usize]
                .iter()
                .map(|&(s, _)| s)
                .collect()
        };

        // GBA guards are evaluated against the SUCCESSOR system state.
        //
        // In the GPVW on-the-fly construction, the initial expansion
        // "reads" the first letter (L(s0)), so GBA state obligations
        // are one step ahead of the product state's system position.
        // Product state (si, qi) pairs system state si with GBA state qi
        // whose obligations correspond to step i+1. When qi transitions,
        // the guard describes what must hold at step i+1 = L(si+1).
        //
        // Historical note: #1246 changed this from sys_succ to sys,
        // which fixed 5 wrong answers that were actually caused by the
        // XML parser bug (first_element → only_element_child). Reverting
        // to sys_succ with the parser fix resolves all 9 properties.
        for &sys_succ in &sys_succs {
            for (e, trans) in gba.transitions[gba_state as usize].iter().enumerate() {
                if trans.guard_satisfied(ctx, sys_succ) {
                    let product_key = (sys_succ, trans.successor);
                    let succ_pid = if let Some(&existing) = product_ids.get(&product_key) {
                        existing
                    } else {
                        let new_pid = product_ids.len() as u32;
                        product_ids.insert(product_key, new_pid);
                        product_adj.push(Vec::new());
                        product_edge_succ.push(Vec::new());
                        product_edge_words.push(Vec::new());
                        product_keys.push(product_key);
                        product_accept.extend_from_slice(acc.state(trans.successor));
                        queue.push_back(new_pid);
                        new_pid
                    };
                    successors.push(succ_pid);
                    edge_word_buf.extend_from_slice(acc.edge(gba_state, e));
                }
            }
        }

        product_edge_succ[pid as usize] = successors.clone();
        product_edge_words[pid as usize] = edge_word_buf;
        successors.sort_unstable();
        successors.dedup();
        product_adj[pid as usize] = successors;

        // Safety: limit product size
        if product_ids.len() > product_state_limit {
            // Product too large — inconclusive
            return None;
        }
    }

    if deadline.is_some_and(|deadline| Instant::now() >= deadline) {
        return None;
    }

    // Find accepting SCCs in the product graph.
    // An SCC is accepting if it's non-trivial AND for each acceptance set,
    // at least one state in the SCC is state-accepting for that set, OR
    // at least one edge within the SCC is edge-accepting for that set.
    if product_adj.is_empty() {
        return Some(false);
    }

    // Flatten the adjacency to CSR (and free the nested rows) before the
    // long-lived Tarjan/SCC phase (audit S5). `None` (u32 edge-offset
    // overflow) makes the run inconclusive, like the size limit above.
    let product_adj = super::on_the_fly::flatten_adjacency(&mut product_adj)?;

    let sccs = tarjan_scc_slices(
        product_adj.state_count(),
        |v| product_adj.neighbors(v),
        |&w| w,
    );

    for scc in &sccs {
        // Non-trivial: has a cycle (size > 1, or size 1 with self-loop)
        let is_nontrivial = if scc.len() > 1 {
            true
        } else {
            let s = scc[0];
            product_adj.neighbors(s as usize).contains(&s)
        };
        if !is_nontrivial {
            continue;
        }

        // Check all acceptance conditions
        if num_accept == 0 {
            // Any non-trivial cycle is accepting.
            dump_full_graph_lasso_if_enabled(ctx, &product_adj, &product_keys, num_roots, scc);
            return Some(true);
        }

        // Build set of SCC members for fast lookup
        let scc_set: rustc_hash::FxHashSet<u32> = scc.iter().copied().collect();

        let all_accepted = (0..num_accept).all(|i| {
            // State-based acceptance: some state in SCC is accepting
            let state_accepted = scc.iter().any(|&s| {
                let s = s as usize;
                accept_bit(&product_accept[s * nw..(s + 1) * nw], i)
            });
            if state_accepted {
                return true;
            }
            // Edge-based acceptance: some edge WITHIN the SCC is accepting
            // (both source and target must be in the SCC)
            scc.iter().any(|&s| {
                let s = s as usize;
                let words = &product_edge_words[s];
                product_edge_succ[s].iter().enumerate().any(|(e, succ)| {
                    scc_set.contains(succ) && accept_bit(&words[e * nw..(e + 1) * nw], i)
                })
            })
        });
        if all_accepted {
            dump_full_graph_lasso_if_enabled(ctx, &product_adj, &product_keys, num_roots, scc);
            return Some(true);
        }
    }

    Some(false)
}

/// Dev-only (`TY_LTL_DUMP_LASSO=1`): print a concrete accepting lasso found by
/// the full-graph product oracle — stem (BFS from the initial product states)
/// plus a cycle inside the accepting SCC. Each position prints the product
/// state, system state id, GBA state, the fired system transition INDEX (this
/// legacy context does not carry the net, so names are not available here —
/// the on-the-fly dump in `on_the_fly.rs` prints names), per-atom truth
/// values, and the non-zero places of the marking as `place_idx=tokens`.
/// Pure diagnostics; never changes verdicts.
fn dump_full_graph_lasso_if_enabled(
    ctx: &LtlContext<'_>,
    product_adj: &CsrProductAdj,
    product_keys: &[(u32, GbaStateId)],
    num_roots: u32,
    scc: &[u32],
) {
    if !std::env::var_os("TY_LTL_DUMP_LASSO").is_some_and(|v| !v.is_empty() && v != "0") {
        return;
    }
    let scc_set: rustc_hash::FxHashSet<u32> = scc.iter().copied().collect();

    // Stem: BFS from the initial product states (id prefix 0..num_roots).
    let mut parent: FxHashMap<u32, u32> = FxHashMap::default();
    let mut seen: rustc_hash::FxHashSet<u32> = rustc_hash::FxHashSet::default();
    let mut queue: std::collections::VecDeque<u32> = std::collections::VecDeque::new();
    let mut entry: Option<u32> = None;
    for r in 0..num_roots {
        if scc_set.contains(&r) {
            entry = Some(r);
        }
        if seen.insert(r) {
            queue.push_back(r);
        }
    }
    'bfs: while let Some(u) = queue.pop_front() {
        if entry.is_some() {
            break;
        }
        for &v in product_adj.neighbors(u as usize) {
            if seen.insert(v) {
                parent.insert(v, u);
                if scc_set.contains(&v) {
                    entry = Some(v);
                    break 'bfs;
                }
                queue.push_back(v);
            }
        }
    }
    let Some(entry) = entry else {
        eprintln!("TY_LTL_DUMP_LASSO[full-graph]: accepting SCC unreachable from roots (?)");
        return;
    };
    let mut stem = vec![entry];
    while let Some(&p) = parent.get(stem.last().expect("non-empty")) {
        stem.push(p);
    }
    stem.reverse();

    // Cycle: shortest SCC-internal path entry → … → entry.
    let mut cyc_parent: FxHashMap<u32, u32> = FxHashMap::default();
    let mut cyc_seen: rustc_hash::FxHashSet<u32> = rustc_hash::FxHashSet::default();
    let mut cyc_queue: std::collections::VecDeque<u32> = std::collections::VecDeque::new();
    cyc_seen.insert(entry);
    cyc_queue.push_back(entry);
    let mut last_before_entry: Option<u32> = None;
    'cyc: while let Some(u) = cyc_queue.pop_front() {
        if product_adj.neighbors(u as usize).contains(&entry) {
            last_before_entry = Some(u);
            break 'cyc;
        }
        for &v in product_adj.neighbors(u as usize) {
            if scc_set.contains(&v) && cyc_seen.insert(v) {
                cyc_parent.insert(v, u);
                cyc_queue.push_back(v);
            }
        }
    }
    let mut cycle: Vec<u32> = Vec::new();
    if let Some(u) = last_before_entry {
        cycle.push(entry);
        let mut back = vec![u];
        let mut cur = u;
        while cur != entry {
            cur = cyc_parent[&cur];
            back.push(cur);
        }
        for &n in back.iter().rev().skip(1) {
            cycle.push(n);
        }
        cycle.push(entry);
    }

    let fired = |u: u32, v: u32| -> String {
        let sys_u = product_keys[u as usize].0;
        let sys_v = product_keys[v as usize].0;
        let sys_adj = &ctx.full.graph.adj[sys_u as usize];
        if sys_adj.is_empty() && sys_u == sys_v {
            return "(deadlock stutter)".to_string();
        }
        sys_adj
            .iter()
            .find(|&&(succ, _)| succ == sys_v)
            .map_or("(no system edge?)".to_string(), |&(_, t)| format!("t#{t}"))
    };
    let print_state = |label: &str, step: usize, pid: u32, fired: &str| {
        let (sys, gba_state) = product_keys[pid as usize];
        let atoms: Vec<bool> = (0..ctx.num_atoms())
            .map(|a| ctx.atom_holds(a, sys))
            .collect();
        let marked: Vec<String> = ctx
            .full
            .markings
            .unpack(sys as usize)
            .iter()
            .enumerate()
            .filter(|&(_, &tokens)| tokens > 0)
            .map(|(p, &tokens)| format!("p{p}={tokens}"))
            .collect();
        eprintln!(
            "  {label}[{step}] pid={pid} sys={sys} gba={gba_state} fired={fired} \
             atoms={atoms:?} marked=[{}]",
            marked.join(", ")
        );
    };

    eprintln!(
        "TY_LTL_DUMP_LASSO[full-graph]: accepting SCC of {} product state(s)",
        scc.len()
    );
    for (i, &pid) in stem.iter().enumerate() {
        let f = if i == 0 {
            "(init)".to_string()
        } else {
            fired(stem[i - 1], pid)
        };
        print_state("stem", i, pid, &f);
    }
    if cycle.is_empty() {
        eprintln!("  cycle: (none reconstructed — SCC cycle search failed?)");
    } else {
        for i in 1..cycle.len() {
            let f = fired(cycle[i - 1], cycle[i]);
            print_state("cycle", i, cycle[i], &f);
        }
        eprintln!(
            "  cycle closes back to stem entry pid={entry} (cycle length {})",
            cycle.len() - 1
        );
    }
}

#[cfg(test)]
#[path = "product_tests.rs"]
mod product_tests;
