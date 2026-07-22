// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Deadlock-existence preservation for the `ReachabilityDeadlock` reduction
//! mode (GlobalProperties P1 pre-reduction).
//!
//! Regression coverage for three confirmed wrong-verdict mechanisms in the
//! Rule B / blocked-by-constant / LP-redundancy materialization cluster:
//!
//! 1. `blocked_by_constant` deleted LIVE consumers of Rule B parallel
//!    duplicates (a merged place is replenished, not frozen at its initial
//!    marking) — manufactured deadlocks (wrong TRUE).
//! 2. `remap_and_combine_arcs` SUMMED the duplicate's arcs onto the
//!    canonical (w -> 2w) while `new_initial` kept the canonical's single
//!    initial marking — enabling `m0 + 2k >= 2w` diverged from the
//!    original's `m0 + k >= w` for any `m0 > 0` (wrong TRUE/FALSE).
//! 3. `find_redundant_places` removed MUTUALLY-dependent implicit places in
//!    one pass: each LP certificate cited the other place's precondition,
//!    so neither certificate survived the joint removal (wrong FALSE).
//!
//! The fix: Rule B drops the duplicate's arcs entirely (the canonical's
//! identical arc already carries the exact constraint), the blocked check
//! exempts every `place_map`-remapped place, and each LP certificate is
//! proven without leaning on places already slated for removal.

use std::collections::{HashSet, VecDeque};

use crate::petri_net::{PetriNet, TransitionIdx};
use crate::reduction::{reduce_iterative_structural_with_mode, ReductionMode};

use super::support::{arc, place, trans};

/// Bounded exhaustive BFS deadlock check.
///
/// Returns `Some(true)` if a reachable marking with no enabled transition
/// exists, `Some(false)` if the full reachable set was explored without
/// finding one, and `None` if the exploration bound was exceeded
/// (inconclusive).
fn has_deadlock_bfs(net: &PetriNet, max_states: usize) -> Option<bool> {
    let mut seen: HashSet<Vec<u64>> = HashSet::new();
    let mut queue: VecDeque<Vec<u64>> = VecDeque::new();
    let init = net.initial_marking.clone();
    seen.insert(init.clone());
    queue.push_back(init);

    while let Some(marking) = queue.pop_front() {
        let mut any_enabled = false;
        for t in 0..net.num_transitions() {
            let tidx = TransitionIdx(t as u32);
            if net.is_enabled(&marking, tidx) {
                any_enabled = true;
                let next = net.fire(&marking, tidx).expect("fire (test)");
                if seen.insert(next.clone()) {
                    if seen.len() > max_states {
                        return None;
                    }
                    queue.push_back(next);
                }
            }
        }
        if !any_enabled {
            return Some(true);
        }
    }
    Some(false)
}

fn reduce_deadlock(net: &PetriNet) -> crate::reduction::ReducedNet {
    reduce_iterative_structural_with_mode(net, &[], ReductionMode::ReachabilityDeadlock, None)
        .expect("ReachabilityDeadlock reduction must not fail")
}

/// Mechanism 1 traced repro: places A,B (m0=0) form a Rule B parallel pair,
/// C (m0=1); t1: A+B -> C; t2: C -> A+B. The original alternates t2/t1
/// forever — deadlock-free (MCC answer FALSE).
///
/// Unsound materialization: B merged into A, then `blocked_by_constant` saw
/// t1's input arc on removed B with m0(B)=0 < 1 and DELETED the live t1;
/// the reduced net fired t2 once and deadlocked -> wrong TRUE.
#[test]
fn test_rule_b_zero_marked_parallel_pair_preserves_deadlock_freedom() {
    let net = PetriNet {
        name: Some("rule-b-zero-marked-pair".into()),
        places: vec![place("A"), place("B"), place("C")],
        transitions: vec![
            trans("t1", vec![arc(0, 1), arc(1, 1)], vec![arc(2, 1)]),
            trans("t2", vec![arc(2, 1)], vec![arc(0, 1), arc(1, 1)]),
        ],
        initial_marking: vec![0, 0, 1],
    };
    assert_eq!(
        has_deadlock_bfs(&net, 10_000),
        Some(false),
        "ground truth: the original net is deadlock-free"
    );

    let reduced = reduce_deadlock(&net);

    // The parallel pair must actually merge (Rule B stays admitted).
    assert!(
        reduced.place_map[1].is_some() && reduced.place_map[1] == reduced.place_map[0],
        "B must merge into A (Rule B admitted for ReachabilityDeadlock)"
    );
    // The live consumer t1 must NOT be deleted by the blocked-by-constant
    // pass: the merged duplicate is replenished, not frozen.
    assert_eq!(
        reduced.net.num_transitions(),
        2,
        "both live transitions must survive; deleting t1 manufactured a deadlock"
    );
    assert_eq!(
        has_deadlock_bfs(&reduced.net, 10_000),
        Some(false),
        "reduced net must remain deadlock-free (original has no deadlock)"
    );
}

/// Mechanism 2 traced repro: A,B (m0=1) parallel pair, C (m0=0); same
/// t1/t2 as above. The original alternates forever (MCC answer FALSE).
///
/// Unsound materialization: the duplicate's arcs were SUMMED onto the
/// canonical (t1 needed A >= 2) while the merged initial marking stayed 1,
/// so the reduced INITIAL marking was already dead -> wrong TRUE.
#[test]
fn test_rule_b_marked_parallel_pair_keeps_arc_weights_and_deadlock_freedom() {
    let net = PetriNet {
        name: Some("rule-b-marked-pair".into()),
        places: vec![place("A"), place("B"), place("C")],
        transitions: vec![
            trans("t1", vec![arc(0, 1), arc(1, 1)], vec![arc(2, 1)]),
            trans("t2", vec![arc(2, 1)], vec![arc(0, 1), arc(1, 1)]),
        ],
        initial_marking: vec![1, 1, 0],
    };
    assert_eq!(
        has_deadlock_bfs(&net, 10_000),
        Some(false),
        "ground truth: the original net is deadlock-free"
    );

    let reduced = reduce_deadlock(&net);

    // Arc weights on the merged canonical must stay at 1 (the duplicate's
    // arcs are dropped, not summed).
    if let Some(canonical) = reduced.place_map[0] {
        for t in &reduced.net.transitions {
            for a in t.inputs.iter().chain(t.outputs.iter()) {
                if a.place == canonical {
                    assert_eq!(
                        a.weight, 1,
                        "Rule B must DROP the duplicate's arcs, not sum them \
                         (doubled weight against an un-doubled initial marking \
                         manufactured a deadlock at the initial marking)"
                    );
                }
            }
        }
    }
    assert_eq!(
        has_deadlock_bfs(&reduced.net, 10_000),
        Some(false),
        "reduced net must remain deadlock-free (original has no deadlock)"
    );

    // Marking expansion must be exact: m(dup) == m(canonical).
    let expanded = reduced
        .expand_marking(&reduced.net.initial_marking)
        .expect("expansion");
    assert_eq!(
        expanded, net.initial_marking,
        "expanding the reduced initial marking must reproduce the original"
    );
}

/// Mechanism 3 traced repro (finding 5): two implicit places `a` and `b`
/// in DIFFERENT P-invariants, each LP certificate citing the other's
/// precondition on the shared consumer `t`.
///
///   u: {s:1, a':2, b':1} -> {a:2, b:1}
///   t: {a:2, b:1} -> {a':2, b':1}
///   M0 = (a=0, b=0, a'=2, b'=1, s=1)
///
/// Ground truth: fire u then t -> (0,0,2,1,0) where u (s=0) and t (a=0)
/// are both disabled — a REAL reachable deadlock (MCC answer TRUE).
///
/// Unsound joint removal: `a` certified via M(b)>=1, then `b` certified via
/// M(a)>=2 in the SAME pass; both removed left `t` with no inputs in the
/// reduced net (always enabled, unbounded growth) -> wrong FALSE.
#[test]
fn test_mutually_implicit_places_not_jointly_removed() {
    let net = PetriNet {
        name: Some("mutually-implicit-pair".into()),
        places: vec![
            place("a"),
            place("b"),
            place("a_comp"),
            place("b_comp"),
            place("s"),
        ],
        transitions: vec![
            trans(
                "u",
                vec![arc(4, 1), arc(2, 2), arc(3, 1)],
                vec![arc(0, 2), arc(1, 1)],
            ),
            trans("t", vec![arc(0, 2), arc(1, 1)], vec![arc(2, 2), arc(3, 1)]),
        ],
        initial_marking: vec![0, 0, 2, 1, 1],
    };
    assert_eq!(
        has_deadlock_bfs(&net, 10_000),
        Some(true),
        "ground truth: the original net reaches a deadlock"
    );

    let reduced = reduce_deadlock(&net);

    let removed_a = reduced.place_map[0].is_none();
    let removed_b = reduced.place_map[1].is_none();
    assert!(
        !(removed_a && removed_b),
        "mutually-implicit places a and b must not BOTH be removed: each \
         LP certificate cites the other's precondition on t"
    );
    assert_eq!(
        has_deadlock_bfs(&reduced.net, 10_000),
        Some(true),
        "the real reachable deadlock must be preserved by the reduction"
    );
}

/// k>=3-way merge: three identical zero-marked buffer places. All
/// duplicates' arcs must be dropped; the canonical's arc alone carries the
/// constraint, and behavior is preserved.
#[test]
fn test_rule_b_three_way_merge_preserves_deadlock_freedom() {
    let net = PetriNet {
        name: Some("rule-b-three-way".into()),
        places: vec![place("A"), place("B"), place("C"), place("buf")],
        transitions: vec![
            trans(
                "fill",
                vec![arc(3, 1)],
                vec![arc(0, 1), arc(1, 1), arc(2, 1)],
            ),
            trans(
                "drain",
                vec![arc(0, 1), arc(1, 1), arc(2, 1)],
                vec![arc(3, 1)],
            ),
        ],
        initial_marking: vec![0, 0, 0, 1],
    };
    assert_eq!(has_deadlock_bfs(&net, 10_000), Some(false));

    let reduced = reduce_deadlock(&net);
    assert_eq!(
        has_deadlock_bfs(&reduced.net, 10_000),
        Some(false),
        "3-way Rule B merge must preserve deadlock-freedom"
    );
}

/// Differential sweep over small MCC 2025 PT models: the full
/// `ReachabilityDeadlock` reduction must be deadlock-equivalent to the
/// original net under bounded exhaustive BFS.
///
/// Skips silently when the local MCC corpus is not present (the corpus is
/// machine-local; see `ty-mccctl fetch`). Models whose original state space
/// exceeds the exploration bound are reported as inconclusive and do not
/// count toward the equivalence assertion.
#[test]
fn test_differential_deadlock_sweep_small_mcc_models() {
    let Some(home) = std::env::var_os("HOME") else {
        eprintln!("differential sweep: $HOME not set; skipping");
        return;
    };
    // Prefer the extracted `ty-corpus` cache (the canonical layout produced
    // by `ty-corpus ensure`); fall back to a hand-extracted INPUTS dir. The
    // INPUTS dir may hold only the unextracted per-model `.tgz` archives, in
    // which case there is nothing scannable — treat that as corpus-absent.
    let cache = std::path::Path::new(&home)
        .join(".cache")
        .join("ty")
        .join("corpus")
        .join("2025");
    let inputs_raw = std::path::Path::new(&home)
        .join("mcc-benchmarks")
        .join("2025")
        .join("inputs")
        .join("INPUTS-2025");
    let inputs = if cache.is_dir() {
        cache
    } else if inputs_raw.is_dir() {
        inputs_raw
    } else {
        eprintln!(
            "differential sweep: no corpus at {} or {}; skipping",
            std::path::Path::new(&home)
                .join(".cache/ty/corpus/2025")
                .display(),
            inputs_raw.display()
        );
        return;
    };

    // Deterministic selection: PT models with model.pnml < 150KB, the
    // smallest instance per model FAMILY (diverse structure beats many
    // size-variants of one net), 12 smallest families overall.
    let mut by_family: std::collections::BTreeMap<String, (u64, std::path::PathBuf)> =
        std::collections::BTreeMap::new();
    let entries = std::fs::read_dir(&inputs).expect("read corpus dir");
    for entry in entries.flatten() {
        let dir = entry.path();
        let name = entry.file_name().to_string_lossy().into_owned();
        let Some(family) = name.split("-PT-").next().filter(|_f| name.contains("-PT-")) else {
            continue;
        };
        if !dir.is_dir() {
            continue;
        }
        let pnml = dir.join("model.pnml");
        let Ok(meta) = std::fs::metadata(&pnml) else {
            continue;
        };
        if meta.len() >= 150 * 1024 {
            continue;
        }
        let candidate = (meta.len(), dir);
        by_family
            .entry(family.to_string())
            .and_modify(|best| {
                if candidate.0 < best.0 {
                    *best = candidate.clone();
                }
            })
            .or_insert(candidate);
    }
    let mut candidates: Vec<(u64, std::path::PathBuf)> = by_family.into_values().collect();
    candidates.sort();
    candidates.truncate(12);
    if candidates.is_empty() {
        // A corpus dir holding only unextracted `.tgz` archives (or no small
        // PT instances) has nothing to sweep — corpus-absent semantics.
        eprintln!(
            "differential sweep: {} has no extracted small PT models; skipping",
            inputs.display()
        );
        return;
    }

    const BFS_BOUND: usize = 1_000_000;
    let mut conclusive = 0usize;
    let mut mismatches: Vec<String> = Vec::new();

    for (size, dir) in &candidates {
        let model = dir.file_name().unwrap().to_string_lossy().into_owned();
        let net = match crate::parser::parse_pnml_dir(dir) {
            Ok(net) => net,
            Err(err) => {
                eprintln!("differential sweep: {model}: parse error {err:?}; skipping");
                continue;
            }
        };
        let reduced = reduce_deadlock(&net);
        let original_dl = has_deadlock_bfs(&net, BFS_BOUND);
        let reduced_dl = has_deadlock_bfs(&reduced.net, BFS_BOUND);
        eprintln!(
            "differential sweep: {model} ({size}B): places {}->{}, transitions {}->{}, \
             original={original_dl:?} reduced={reduced_dl:?}",
            net.num_places(),
            reduced.net.num_places(),
            net.num_transitions(),
            reduced.net.num_transitions(),
        );
        if let (Some(o), Some(r)) = (original_dl, reduced_dl) {
            conclusive += 1;
            if o != r {
                mismatches.push(format!("{model}: original={o} reduced={r}"));
            }
        }
    }

    assert!(
        mismatches.is_empty(),
        "deadlock-equivalence violated by ReachabilityDeadlock reduction:\n{}",
        mismatches.join("\n")
    );
    assert!(
        conclusive > 0,
        "no model was exhaustively explorable within {BFS_BOUND} states; \
         the sweep proved nothing"
    );
    eprintln!("differential sweep: {conclusive} conclusive equivalences, 0 mismatches");
}
