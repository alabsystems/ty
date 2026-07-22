// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

use super::{
    on_the_fly_product_emptiness, on_the_fly_product_emptiness_por_memo_toggle,
    on_the_fly_product_emptiness_por_with_size, on_the_fly_product_emptiness_with_limit,
    on_the_fly_product_emptiness_with_limit_memo_toggle, MarkingTable, PorContext,
};
use crate::buchi::gba::build_gba;
use crate::buchi::nnf::negate;
use crate::buchi::LtlNnf;
use crate::examinations::ltl_por::{ltl_visible_per_gba_state, ltl_visible_reduced_transitions};
use crate::petri_net::{Arc, PetriNet, PlaceIdx, PlaceInfo, TransitionInfo};
use crate::reduction::ReducedNet;
use crate::resolved_predicate::{ResolvedIntExpr, ResolvedPredicate};
use crate::stubborn::DependencyGraph;
use std::time::{Duration, Instant};

/// Identity ReducedNet: no reductions applied, marking expansion is passthrough.
fn identity_reduced(net: &PetriNet) -> ReducedNet {
    ReducedNet::identity(net)
}

fn alternating_net() -> PetriNet {
    PetriNet {
        name: Some("alternating".to_string()),
        places: vec![
            PlaceInfo {
                id: "p0".to_string(),
                name: None,
            },
            PlaceInfo {
                id: "p1".to_string(),
                name: None,
            },
        ],
        transitions: vec![
            TransitionInfo {
                id: "t0".to_string(),
                name: None,
                inputs: vec![Arc {
                    place: PlaceIdx(0),
                    weight: 1,
                }],
                outputs: vec![Arc {
                    place: PlaceIdx(1),
                    weight: 1,
                }],
            },
            TransitionInfo {
                id: "t1".to_string(),
                name: None,
                inputs: vec![Arc {
                    place: PlaceIdx(1),
                    weight: 1,
                }],
                outputs: vec![Arc {
                    place: PlaceIdx(0),
                    weight: 1,
                }],
            },
        ],
        initial_marking: vec![1, 0],
    }
}

fn deadlock_net(initial_marking: Vec<u64>) -> PetriNet {
    PetriNet {
        name: Some("deadlock".to_string()),
        places: initial_marking
            .iter()
            .enumerate()
            .map(|(index, _)| PlaceInfo {
                id: format!("p{index}"),
                name: None,
            })
            .collect(),
        transitions: Vec::new(),
        initial_marking,
    }
}

fn tokens_at_least(place: PlaceIdx, value: u64) -> ResolvedPredicate {
    ResolvedPredicate::IntLe(
        ResolvedIntExpr::Constant(value),
        ResolvedIntExpr::TokensCount(vec![place]),
    )
}

/// Helper: run on-the-fly product emptiness with identity reduction.
fn check_on_the_fly(
    gba_formula: &LtlNnf,
    net: &PetriNet,
    atoms: &[ResolvedPredicate],
) -> Option<bool> {
    let reduced = identity_reduced(net);
    let neg = negate(gba_formula);
    let gba = build_gba(&neg);
    on_the_fly_product_emptiness_with_limit(
        &gba,
        net,
        &reduced,
        net,
        atoms,
        usize::MAX,
        50_000_000,
        None,
    )
    .expect("on-the-fly product should expand markings safely")
}

/// N independent token shuttles (place pairs `p_{2k}` ⇄ `p_{2k+1}`). Each
/// shuttle's two transitions are independent of every other shuttle, so an atom
/// referencing only one shuttle's place makes the other shuttles' transitions
/// invisible — the canonical case where stutter-insensitive POR must prune.
fn independent_shuttles(n: usize) -> PetriNet {
    let mut places = Vec::new();
    let mut transitions = Vec::new();
    let mut initial = Vec::new();
    for k in 0..n {
        let a = PlaceIdx((2 * k) as u32);
        let b = PlaceIdx((2 * k + 1) as u32);
        places.push(PlaceInfo {
            id: format!("p{}", 2 * k),
            name: None,
        });
        places.push(PlaceInfo {
            id: format!("p{}", 2 * k + 1),
            name: None,
        });
        initial.push(1);
        initial.push(0);
        transitions.push(TransitionInfo {
            id: format!("t{}_fwd", k),
            name: None,
            inputs: vec![Arc {
                place: a,
                weight: 1,
            }],
            outputs: vec![Arc {
                place: b,
                weight: 1,
            }],
        });
        transitions.push(TransitionInfo {
            id: format!("t{}_bwd", k),
            name: None,
            inputs: vec![Arc {
                place: b,
                weight: 1,
            }],
            outputs: vec![Arc {
                place: a,
                weight: 1,
            }],
        });
    }
    PetriNet {
        name: Some("independent_shuttles".to_string()),
        places,
        transitions,
        initial_marking: initial,
    }
}

/// Cross-check: the POR (DFS + cycle-proviso) product emptiness verdict MUST
/// equal the exhaustive (full-expansion BFS) verdict — in BOTH visibility
/// modes (static whole-formula AND per-Büchi-state). A disagreement would be
/// a soundness bug. Returns the (full, por) pair so callers can also assert
/// the concrete verdict.
fn assert_por_matches_full(
    formula: &LtlNnf,
    net: &PetriNet,
    atoms: &[ResolvedPredicate],
) -> (Option<bool>, Option<bool>) {
    let reduced = identity_reduced(net);
    let neg = negate(formula);
    let gba = build_gba(&neg);

    let full = on_the_fly_product_emptiness_with_limit(
        &gba,
        net,
        &reduced,
        net,
        atoms,
        usize::MAX,
        50_000_000,
        None,
    )
    .expect("full product expands safely");

    let mut last_por = None;
    for per_state_visibility in [false, true] {
        let por_ctx = PorContext {
            dep: DependencyGraph::build(net),
            visible: ltl_visible_reduced_transitions(atoms, &reduced),
            per_state_visibility,
        };
        let por = on_the_fly_product_emptiness(
            &gba,
            net,
            &reduced,
            net,
            atoms,
            Some(&por_ctx),
            usize::MAX,
            None,
        )
        .expect("POR product expands safely");

        assert_eq!(
            full, por,
            "POR (per_state_visibility={per_state_visibility}) verdict must \
             match exhaustive (formula soundness)"
        );
        last_por = Some(por);
    }
    (full, last_por.expect("both POR modes ran"))
}

/// Net whose POR product contains a **self-loop** (length-1 cycle): a pure
/// Δ=0 "noop" transition (`p_idle` → `p_idle`) on a *private* input place,
/// alongside a live `p0 ⇄ p1` shuttle plus a sink `p0 → p2`.
///
/// At the initial marking `[p0=1, p1=0, p_idle=1, p2=0]` the enabled set is
/// {a, c, noop}. `noop`'s input place `p_idle` is private, so its
/// `interferes_with` set is empty → it is the min-interference stubborn seed →
/// the deadlock-preserving stubborn set is the singleton `{noop}`. `noop` has
/// Δ=0, hence touches no atom (invisible), so C2 accepts `{noop}` as the ample
/// set. Firing `noop` reproduces the *same* (system marking, GBA state) product
/// node — a product self-loop.
///
/// This is the C3-cycle-proviso regression: the node currently being expanded
/// (`pid`) is marked `on_stack` only AFTER `expand_product_node` returns, so an
/// `on_stack`-only `closes_cycle` test misses the self-loop, leaving the cycle
/// `{pid}` with NO fully-expanded state (a C3 violation) — the POR product then
/// loops on `noop` forever, never fires `a`, never reaches `p1 ≥ 1`, and reports
/// a WRONG "no accepting run". The applied fix detects `id == pid` explicitly,
/// forcing full expansion of self-looping product states.
fn self_loop_net() -> PetriNet {
    let place = |i: usize| PlaceInfo {
        id: format!("p{i}"),
        name: None,
    };
    let arc = |p: u32| Arc {
        place: PlaceIdx(p),
        weight: 1,
    };
    // Places: p0=0, p1=1, p_idle=2, p2=3.
    PetriNet {
        name: Some("self_loop".to_string()),
        places: vec![
            place(0),
            place(1),
            PlaceInfo {
                id: "p_idle".to_string(),
                name: None,
            },
            place(2),
        ],
        transitions: vec![
            // a: p0 → p1  (visible: touches the atom place p1)
            TransitionInfo {
                id: "a".to_string(),
                name: None,
                inputs: vec![arc(0)],
                outputs: vec![arc(1)],
            },
            // c: p0 → p2  (sink; invisible)
            TransitionInfo {
                id: "c".to_string(),
                name: None,
                inputs: vec![arc(0)],
                outputs: vec![arc(3)],
            },
            // b: p1 → p0  (visible: touches the atom place p1)
            TransitionInfo {
                id: "b".to_string(),
                name: None,
                inputs: vec![arc(1)],
                outputs: vec![arc(0)],
            },
            // noop: p_idle → p_idle  (Δ=0 self-loop on a private place; invisible)
            TransitionInfo {
                id: "noop".to_string(),
                name: None,
                inputs: vec![arc(2)],
                outputs: vec![arc(2)],
            },
        ],
        initial_marking: vec![1, 0, 1, 0],
    }
}

#[test]
fn test_por_matches_full_with_product_self_loop() {
    // Atom 0 = (p1 >= 1), referencing the atom place p1 (PlaceIdx(1)).
    let atoms = vec![tokens_at_least(PlaceIdx(1), 1)];
    let net = self_loop_net();

    // Sanity: the visibility over-approximation is exactly the p1-touching
    // transitions {a, b}; `noop` (and `c`) are invisible, which is what lets
    // the stubborn singleton {noop} pass C2 and create the self-loop.
    let reduced = identity_reduced(&net);
    let visible = ltl_visible_reduced_transitions(&atoms, &reduced);
    assert_eq!(
        visible.len(),
        2,
        "exactly the p1-touching transitions (a, b) must be visible; \
         noop/c stay invisible (self-loop trigger)"
    );

    // Formula: G(p1 < 1)  ==  Release(False, ¬atom0).
    // `assert_por_matches_full` negates internally, so the GBA is built for
    // negate(G(¬a)) = F(a) = F(p1 >= 1). The exhaustive product finds the
    // accepting run (fire a → reach p1 >= 1); a C3-unsound POR that ignores the
    // `noop` self-loop never fires `a` and would (wrongly) report no run.
    let g_not_a = LtlNnf::Release(Box::new(LtlNnf::False), Box::new(LtlNnf::NegAtom(0)));

    let (full, por) = assert_por_matches_full(&g_not_a, &net, &atoms);

    // Guard against a vacuous pass: the accepting run must genuinely exist, so
    // `full` is a definite Some(true). With the old `on_stack`-only check the
    // POR builder would return Some(false) here (self-loop on noop, never fires
    // a) and `assert_por_matches_full` would fail on the full/por mismatch.
    assert_eq!(
        full,
        Some(true),
        "exhaustive product must find the accepting run for F(p1>=1)"
    );
    assert_eq!(
        por,
        Some(true),
        "POR product must also find it — self-loops force full expansion (C3)"
    );
}

#[test]
fn test_por_matches_full_on_concurrent_liveness() {
    // Atom references only shuttle 0's place p0 → all other shuttles invisible.
    let atom = tokens_at_least(PlaceIdx(0), 1);
    let a = LtlNnf::Atom(0);
    let ff = || Box::new(LtlNnf::False);
    let tt = || Box::new(LtlNnf::True);

    for n in 1..=3usize {
        let net = independent_shuttles(n);
        let atoms = vec![atom.clone()];

        // F p0
        assert_por_matches_full(&LtlNnf::Until(tt(), Box::new(a.clone())), &net, &atoms);
        // G p0  == Release(False, p0)
        assert_por_matches_full(&LtlNnf::Release(ff(), Box::new(a.clone())), &net, &atoms);
        // G F p0  == Release(False, F p0)
        assert_por_matches_full(
            &LtlNnf::Release(ff(), Box::new(LtlNnf::Until(tt(), Box::new(a.clone())))),
            &net,
            &atoms,
        );
        // F G p0  == Until(True, Release(False, p0))
        assert_por_matches_full(
            &LtlNnf::Until(tt(), Box::new(LtlNnf::Release(ff(), Box::new(a.clone())))),
            &net,
            &atoms,
        );
    }
}

#[test]
fn test_por_matches_full_multi_atom() {
    // Atoms on two different shuttles; shuttle 2 stays invisible.
    let atoms = vec![
        tokens_at_least(PlaceIdx(0), 1),
        tokens_at_least(PlaceIdx(2), 1),
    ];
    let net = independent_shuttles(3);
    let a0 = LtlNnf::Atom(0);
    let a1 = LtlNnf::Atom(1);
    let tt = || Box::new(LtlNnf::True);
    let ff = || Box::new(LtlNnf::False);

    // G(p0 ∨ p2)
    assert_por_matches_full(
        &LtlNnf::Release(ff(), Box::new(LtlNnf::Or(vec![a0.clone(), a1.clone()]))),
        &net,
        &atoms,
    );
    // G F (p0 ∧ p2)
    assert_por_matches_full(
        &LtlNnf::Release(
            ff(),
            Box::new(LtlNnf::Until(
                tt(),
                Box::new(LtlNnf::And(vec![a0.clone(), a1.clone()])),
            )),
        ),
        &net,
        &atoms,
    );
    // p0 U p2
    assert_por_matches_full(&LtlNnf::Until(Box::new(a0), Box::new(a1)), &net, &atoms);
}

// ── Per-Büchi-state visibility (P2) regression nets ──
//
// These two nets pin the SOUND per-state design (reachability-closed atom
// sets, retarding edges included) against the two unsound shrinkings:
// progressing-edge-atoms-only and current-state-atoms-only. Under either
// unsound variant the POR product loses the (only) accepting run and the
// `assert_por_matches_full` cross-check fails with a wrong "no accepting run".

/// CE-1 — retarding atoms matter: ψ (negated property) = `r U p`.
///
/// GBA state q0 = {r U p} has the progressing edge guarded by `p` and the
/// RETARDING (self-loop) edge guarded by `r`. Net: `t` (index 0) silently
/// kills `r`'s place; `u` (index 1) independently sets `p`'s place. The
/// stubborn seed is `t` (lowest index, empty interference), so the candidate
/// ample is `{t}`. Under progressing-only visibility ({p}) `t` is invisible
/// → ample accepted → firing `t` makes BOTH of q0's guards false at the
/// successor → product dead end, the accepting run via `u` is erased, and C3
/// never fires (the broken reduced graph has no cycle to close). With
/// retarding atoms included (`r` ∈ row(q0)) `t` is visible → full expansion
/// → the accepting run survives.
fn retarding_kill_net() -> PetriNet {
    let place = |id: &str| PlaceInfo {
        id: id.to_string(),
        name: None,
    };
    let arc = |p: u32| Arc {
        place: PlaceIdx(p),
        weight: 1,
    };
    // Places: pr=0 (atom r), pp=1 (atom p), ps=2 (private source for u).
    PetriNet {
        name: Some("retarding_kill".to_string()),
        places: vec![place("pr"), place("pp"), place("ps")],
        transitions: vec![
            // t: pr → ∅  (kills r; touches no progressing-edge atom place)
            TransitionInfo {
                id: "t".to_string(),
                name: None,
                inputs: vec![arc(0)],
                outputs: vec![],
            },
            // u: ps → pp  (sets p; independent of t — disjoint places)
            TransitionInfo {
                id: "u".to_string(),
                name: None,
                inputs: vec![arc(2)],
                outputs: vec![arc(1)],
            },
        ],
        initial_marking: vec![1, 0, 1],
    }
}

#[test]
fn test_por_matches_full_retarding_guard_kill() {
    // Atom 0 = r = (pr >= 1), atom 1 = p = (pp >= 1).
    let atoms = vec![
        tokens_at_least(PlaceIdx(0), 1),
        tokens_at_least(PlaceIdx(1), 1),
    ];
    let net = retarding_kill_net();

    // Property φ = ¬(r U p) = Release(¬r, ¬p); the helper negates internally,
    // so the GBA is built for ψ = r U p.
    let phi = LtlNnf::Release(Box::new(LtlNnf::NegAtom(0)), Box::new(LtlNnf::NegAtom(1)));

    let (full, por) = assert_por_matches_full(&phi, &net, &atoms);

    // Guard against a vacuous pass: the accepting run (fire u, discharge the
    // Until via p, loop in the accepting sink) genuinely exists.
    assert_eq!(
        full,
        Some(true),
        "exhaustive product must find the r U p run"
    );
    assert_eq!(
        por,
        Some(true),
        "POR must keep it (retarding atoms visible)"
    );
}

/// CE-2 — the crossing problem: even ALL outgoing-edge atoms of the CURRENT
/// state are not enough; reachability closure is required.
///
/// ψ (negated property) = (z R (z∧y)) ∧ (b U (¬z∧c)). In the GBA state
/// q = {zR(z∧y), bU(¬z∧c)} every surviving expansion branch forces `z` (the
/// ¬z branch dies by contradiction before recording its atoms), so atom `c`
/// appears on NO edge of q — only on the edges of the reachable state
/// q1 = {bU(¬z∧c)}. Net: `t_c` (index 0) silently clears `c`'s place
/// (place-disjoint from everything), `tau` is a neutral private self-loop,
/// `u_z` clears `z`'s place. Under current-state-only visibility ({z,y,b})
/// the stubborn seed {t_c} is invisible at q → ample accepted → `c` is dead
/// before q1 can ever discharge ¬z∧c → the accepting run (fire u_z while c
/// still holds) is erased. The reachability-closed row(q) ⊇ atoms(q1) ∋ c
/// makes t_c visible → full expansion → correct verdict.
fn crossing_net() -> PetriNet {
    let place = |id: &str| PlaceInfo {
        id: id.to_string(),
        name: None,
    };
    let arc = |p: u32| Arc {
        place: PlaceIdx(p),
        weight: 1,
    };
    // Places: pz=0, py=1, pb=2, pc=3, ptau=4.
    PetriNet {
        name: Some("crossing".to_string()),
        places: vec![
            place("pz"),
            place("py"),
            place("pb"),
            place("pc"),
            place("ptau"),
        ],
        transitions: vec![
            // t_c: pc → ∅  (clears c; invisible to q's OWN edge atoms {z,y,b})
            TransitionInfo {
                id: "t_c".to_string(),
                name: None,
                inputs: vec![arc(3)],
                outputs: vec![],
            },
            // tau: ptau → ptau  (neutral; keeps the net live)
            TransitionInfo {
                id: "tau".to_string(),
                name: None,
                inputs: vec![arc(4)],
                outputs: vec![arc(4)],
            },
            // u_z: pz → ∅  (clears z, enabling the ¬z∧c discharge)
            TransitionInfo {
                id: "u_z".to_string(),
                name: None,
                inputs: vec![arc(0)],
                outputs: vec![],
            },
        ],
        initial_marking: vec![1, 1, 1, 1, 1],
    }
}

#[test]
fn test_por_matches_full_atom_crossing() {
    // Atoms: z=0 (pz>=1), y=1 (py>=1), b=2 (pb>=1), c=3 (pc>=1).
    let atoms = vec![
        tokens_at_least(PlaceIdx(0), 1),
        tokens_at_least(PlaceIdx(1), 1),
        tokens_at_least(PlaceIdx(2), 1),
        tokens_at_least(PlaceIdx(3), 1),
    ];
    let net = crossing_net();

    // φ = ¬ψ in NNF, so that negate(φ) = ψ = (z R (z∧y)) ∧ (b U (¬z∧c)).
    let z = || LtlNnf::Atom(0);
    let nz = || LtlNnf::NegAtom(0);
    let phi = LtlNnf::Or(vec![
        // ¬(z R (z∧y)) = ¬z U (¬z ∨ ¬y)
        LtlNnf::Until(
            Box::new(nz()),
            Box::new(LtlNnf::Or(vec![nz(), LtlNnf::NegAtom(1)])),
        ),
        // ¬(b U (¬z∧c)) = ¬b R (z ∨ ¬c)
        LtlNnf::Release(
            Box::new(LtlNnf::NegAtom(2)),
            Box::new(LtlNnf::Or(vec![z(), LtlNnf::NegAtom(3)])),
        ),
    ]);
    // Sanity: the negation the helper builds is exactly ψ.
    assert_eq!(
        negate(&phi),
        LtlNnf::And(vec![
            LtlNnf::Release(
                Box::new(z()),
                Box::new(LtlNnf::And(vec![z(), LtlNnf::Atom(1)])),
            ),
            LtlNnf::Until(
                Box::new(LtlNnf::Atom(2)),
                Box::new(LtlNnf::And(vec![nz(), LtlNnf::Atom(3)])),
            ),
        ]),
        "test setup: negate(φ) must be the intended ψ"
    );

    let (full, por) = assert_por_matches_full(&phi, &net, &atoms);

    // The accepting run (fire u_z while c is still marked) genuinely exists.
    assert_eq!(full, Some(true), "exhaustive product must find the ψ run");
    assert_eq!(por, Some(true), "POR must keep it (c visible via closure)");
}

/// Differential: per-Büchi-state visibility prunes STRICTLY more product
/// states than the static whole-formula set — with an identical verdict.
///
/// ψ (negated property) = g U h with g = (p_g >= 1) FALSE initially and
/// h = (p_h >= 1) true initially: the Until discharges at step 0, so the only
/// product root is the no-obligation sink state q∅, whose reachability-closed
/// atom set is EMPTY (its True self-loop carries no atoms) — every transition
/// is per-state invisible there. The four independent one-shot transitions
/// w_k (a_k → b_k + p_g) PRODUCE into g's place, so they are visible to the
/// static whole-formula set (full expansion of the 2^4 marking cube) but
/// invisible to row(q∅) (singleton ample chains, and the w-subnet is acyclic
/// so C3 never forces re-expansion).
#[test]
fn test_per_state_visibility_prunes_more_than_static() {
    let place = |id: String| PlaceInfo { id, name: None };
    let arc = |p: u32| Arc {
        place: PlaceIdx(p),
        weight: 1,
    };
    // Places: p_g=0 (atom g, initially 0), p_h=1 (atom h, initially 1),
    // then a_k/b_k pairs for k = 0..4.
    let n = 4u32;
    let mut places = vec![place("p_g".to_string()), place("p_h".to_string())];
    let mut transitions = Vec::new();
    let mut initial = vec![0u64, 1];
    for k in 0..n {
        let a = 2 + 2 * k;
        let b = 3 + 2 * k;
        places.push(place(format!("a{k}")));
        places.push(place(format!("b{k}")));
        initial.push(1);
        initial.push(0);
        transitions.push(TransitionInfo {
            id: format!("w{k}"),
            name: None,
            inputs: vec![arc(a)],
            // Produces into p_g: statically visible (touches atom g's place),
            // but per-state invisible at q∅. Output arcs create no stubborn
            // interference, so {w_k} stays a singleton ample candidate.
            outputs: vec![arc(b), arc(0)],
        });
    }
    let net = PetriNet {
        name: Some("per_state_prunes_more".to_string()),
        places,
        transitions,
        initial_marking: initial,
    };

    let atoms = vec![
        tokens_at_least(PlaceIdx(0), 1), // g
        tokens_at_least(PlaceIdx(1), 1), // h
    ];
    // φ = ¬(g U h) = ¬g R ¬h, so the helper's negation is ψ = g U h.
    let phi = LtlNnf::Release(Box::new(LtlNnf::NegAtom(0)), Box::new(LtlNnf::NegAtom(1)));

    let reduced = identity_reduced(&net);
    let neg = negate(&phi);
    let gba = build_gba(&neg);

    // Row-level proof that per-state visibility is strictly smaller: the
    // static set is non-empty (the w_k touch p_g) while some per-state row
    // (the no-obligation sink q∅) is all-invisible; and every row is a
    // subset of the static set.
    let static_visible = ltl_visible_reduced_transitions(&atoms, &reduced);
    assert!(!static_visible.is_empty(), "w_k must be statically visible");
    let rows = ltl_visible_per_gba_state(&gba, &atoms, &reduced)
        .expect("per-state rows must build without anomaly");
    let mut static_row = vec![false; net.num_transitions()];
    for &t in &static_visible {
        static_row[t.0 as usize] = true;
    }
    for row in &rows {
        for (t, &v) in row.iter().enumerate() {
            assert!(!v || static_row[t], "per-state row must be ⊆ static row");
        }
    }
    assert!(
        rows.iter().any(|row| row.iter().all(|&v| !v)),
        "the accepting sink's row must make every transition invisible"
    );

    // Exhaustive oracle.
    let full = on_the_fly_product_emptiness_with_limit(
        &gba,
        &net,
        &reduced,
        &net,
        &atoms,
        usize::MAX,
        50_000_000,
        None,
    )
    .expect("full product expands safely");
    assert_eq!(full, Some(true), "ψ = g U h discharges at step 0");

    // POR with static vs per-state visibility: identical verdict, strictly
    // fewer product states under per-state rows.
    let run = |per_state_visibility: bool| {
        let por_ctx = PorContext {
            dep: DependencyGraph::build(&net),
            visible: ltl_visible_reduced_transitions(&atoms, &reduced),
            per_state_visibility,
        };
        on_the_fly_product_emptiness_por_with_size(
            &gba,
            &net,
            &reduced,
            &net,
            &atoms,
            &por_ctx,
            usize::MAX,
            None,
        )
        .expect("POR product expands safely")
    };
    let (verdict_static, size_static) = run(false);
    let (verdict_per_state, size_per_state) = run(true);

    assert_eq!(verdict_static, full, "static POR must match exhaustive");
    assert_eq!(
        verdict_per_state, full,
        "per-state POR must match exhaustive"
    );
    eprintln!("product states: static={size_static} per_state={size_per_state}");
    assert!(
        size_per_state < size_static,
        "per-state visibility must prune strictly more product states \
         (per_state={size_per_state}, static={size_static})"
    );
}

// ── Equivalence tests: on-the-fly must match pre-built product ──

#[test]
fn test_on_the_fly_trivial_true() {
    // GBA for True has an accepting cycle on any non-empty system.
    let net = alternating_net();
    let result = check_on_the_fly(&LtlNnf::True, &net, &[]);
    // check_ltl negates, so True → neg → False → no accepting cycle → Some(true).
    // But we're calling product emptiness directly: GBA for neg(True)=False → no states → false.
    // Actually, let me think about this correctly.
    // We negate the formula in check_on_the_fly, so:
    // formula = True → neg = False → GBA has 0 states → product empty → no cycle → Some(false)
    assert_eq!(result, Some(false));
}

#[test]
fn test_on_the_fly_alternating_has_cycle() {
    // Build GBA directly for True (not negated) — any cycle is accepting.
    let net = alternating_net();
    let reduced = identity_reduced(&net);
    let gba = build_gba(&LtlNnf::True);
    let result = on_the_fly_product_emptiness_with_limit(
        &gba,
        &net,
        &reduced,
        &net,
        &[],
        usize::MAX,
        50_000_000,
        None,
    )
    .expect("on-the-fly product should expand markings safely");
    assert_eq!(result, Some(true));
}

#[test]
fn test_on_the_fly_deadlock_self_loop_has_cycle() {
    let net = deadlock_net(vec![1]);
    let reduced = identity_reduced(&net);
    let gba = build_gba(&LtlNnf::True);
    let result = on_the_fly_product_emptiness_with_limit(
        &gba,
        &net,
        &reduced,
        &net,
        &[],
        usize::MAX,
        50_000_000,
        None,
    )
    .expect("on-the-fly product should expand markings safely");
    assert_eq!(result, Some(true));
}

#[test]
fn test_on_the_fly_until_no_accepting_cycle_on_deadlock() {
    // F(p0 >= 1) on a deadlock net with p0=0: never satisfied.
    let net = deadlock_net(vec![0]);
    let reduced = identity_reduced(&net);
    let atoms = vec![tokens_at_least(PlaceIdx(0), 1)];
    let formula = LtlNnf::Until(Box::new(LtlNnf::True), Box::new(LtlNnf::Atom(0)));
    let gba = build_gba(&formula);
    let result = on_the_fly_product_emptiness_with_limit(
        &gba,
        &net,
        &reduced,
        &net,
        &atoms,
        usize::MAX,
        50_000_000,
        None,
    )
    .expect("on-the-fly product should expand markings safely");
    assert_eq!(result, Some(false));
}

#[test]
fn test_on_the_fly_size_limit_returns_none() {
    let net = alternating_net();
    let reduced = identity_reduced(&net);
    let gba = build_gba(&LtlNnf::True);
    let result = on_the_fly_product_emptiness_with_limit(
        &gba,
        &net,
        &reduced,
        &net,
        &[],
        usize::MAX,
        0,
        None,
    )
    .expect("size-limit path should not hit expansion overflow");
    assert_eq!(result, None);
}

#[test]
fn test_on_the_fly_system_marking_limit_returns_none() {
    let net = alternating_net();
    let reduced = identity_reduced(&net);
    let gba = build_gba(&LtlNnf::True);
    let result = on_the_fly_product_emptiness_with_limit(
        &gba,
        &net,
        &reduced,
        &net,
        &[],
        1,
        50_000_000,
        None,
    )
    .expect("system-marking limit should not hit expansion overflow");
    assert_eq!(result, None);
}

#[test]
fn test_on_the_fly_expired_deadline_returns_none() {
    let net = alternating_net();
    let reduced = identity_reduced(&net);
    let gba = build_gba(&LtlNnf::True);
    // Deliberately-expired deadline: `now - 1s` cannot underflow (the process has
    // been up far longer than 1s), so the unchecked sub is safe.
    #[allow(clippy::unchecked_time_subtraction)]
    let expired_deadline = Instant::now() - Duration::from_secs(1);
    let result = on_the_fly_product_emptiness_with_limit(
        &gba,
        &net,
        &reduced,
        &net,
        &[],
        usize::MAX,
        50_000_000,
        Some(expired_deadline),
    )
    .expect("deadline path should not hit expansion overflow");
    assert_eq!(result, None);
}

// ── Atom-satisfaction memo: unit + differential tests ──
//
// Note: in debug builds (i.e. under `cargo test`) every memoized guard
// decision is ALSO cross-checked per edge against the direct eval path inside
// `for_satisfied_edges` (assert), so every memo-on run below doubles as a
// guard-decision differential.

/// Unit: the interned atom bitmask must equal `eval_predicate` per atom on
/// the expanded ORIGINAL-net marking, for every reachable marking.
#[test]
fn test_atom_mask_memo_matches_direct_eval() {
    use crate::explorer::ExplorationSetup;
    use crate::marking::pack_marking_config;
    use crate::petri_net::TransitionIdx;
    use crate::resolved_predicate::eval_predicate;
    use rustc_hash::FxHashSet;

    let net = independent_shuttles(3);
    let reduced = identity_reduced(&net);
    let atoms = vec![
        tokens_at_least(PlaceIdx(0), 1),
        tokens_at_least(PlaceIdx(2), 1),
        tokens_at_least(PlaceIdx(4), 1),
    ];
    let setup = ExplorationSetup::analyze(&net);
    let mut table = MarkingTable::new(
        &atoms,
        &reduced,
        &net,
        &setup.marking_config,
        usize::MAX,
        true,
    );

    // Enumerate all reachable markings (tiny: 2^3) and check each.
    let mut seen: FxHashSet<Vec<u64>> = FxHashSet::default();
    let mut stack = vec![net.initial_marking.clone()];
    let mut pack_buf = Vec::new();
    while let Some(tokens) = stack.pop() {
        if !seen.insert(tokens.clone()) {
            continue;
        }
        pack_marking_config(&tokens, &setup.marking_config, &mut pack_buf);
        let mid = table
            .intern_marking(&pack_buf)
            .expect("expansion is safe")
            .expect("budget is unlimited");
        let expanded = reduced.expand_marking(&tokens).expect("identity expand");
        for (i, atom) in atoms.iter().enumerate() {
            let direct = eval_predicate(atom, &expanded, &net);
            let memoized = (table.atom_masks[mid as usize] >> i) & 1 == 1;
            assert_eq!(memoized, direct, "atom {i} bit mismatch at {tokens:?}");
        }
        // Re-intern must return the same id (hit path).
        assert_eq!(
            table.intern_marking(&pack_buf).unwrap(),
            Some(mid),
            "re-intern must be a stable hit"
        );
        for t in 0..net.num_transitions() {
            let t = TransitionIdx(t as u32);
            if net.is_enabled(&tokens, t) {
                let mut succ = tokens.clone();
                net.apply_delta(&mut succ, t).expect("apply_delta");
                stack.push(succ);
            }
        }
    }
    assert_eq!(seen.len(), 8, "3 independent shuttles → 2^3 markings");
}

/// Deterministic LCG (no rand dependency) for the random-net differential.
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0 >> 33
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

/// Random token-preserving net (every transition is 1-in/1-out, weight 1, so
/// the total token count is invariant and the state space stays tiny).
fn random_conservative_net(rng: &mut Lcg) -> PetriNet {
    let num_places = 3 + rng.below(4) as usize; // 3..=6
    let num_transitions = 3 + rng.below(5) as usize; // 3..=7
    let places = (0..num_places)
        .map(|i| PlaceInfo {
            id: format!("p{i}"),
            name: None,
        })
        .collect();
    let transitions = (0..num_transitions)
        .map(|i| {
            let src = rng.below(num_places as u64) as u32;
            let dst = rng.below(num_places as u64) as u32;
            TransitionInfo {
                id: format!("t{i}"),
                name: None,
                inputs: vec![Arc {
                    place: PlaceIdx(src),
                    weight: 1,
                }],
                outputs: vec![Arc {
                    place: PlaceIdx(dst),
                    weight: 1,
                }],
            }
        })
        .collect();
    let mut initial: Vec<u64> = (0..num_places).map(|_| rng.below(2)).collect();
    if initial.iter().all(|&v| v == 0) {
        initial[0] = 1; // keep at least one token so something can fire
    }
    PetriNet {
        name: Some("random_conservative".to_string()),
        places,
        transitions,
        initial_marking: initial,
    }
}

/// Differential: on random nets × a family of LTL formulas, the memoized
/// product verdict must equal the direct-eval (memo-disabled) verdict, on
/// BOTH builders (BFS exact and DFS+POR), and POR must match BFS.
#[test]
fn test_memo_vs_direct_differential_random_nets() {
    let mut rng = Lcg(0x5eed_2026);

    for round in 0..40 {
        let net = random_conservative_net(&mut rng);
        let reduced = identity_reduced(&net);
        let num_places = net.num_places();
        let num_atoms = 1 + rng.below(3) as usize; // 1..=3
        let atoms: Vec<ResolvedPredicate> = (0..num_atoms)
            .map(|_| {
                tokens_at_least(
                    PlaceIdx(rng.below(num_places as u64) as u32),
                    1 + rng.below(2),
                )
            })
            .collect();

        let a0 = || Box::new(LtlNnf::Atom(0));
        let n0 = || Box::new(LtlNnf::NegAtom(0));
        let tt = || Box::new(LtlNnf::True);
        let ff = || Box::new(LtlNnf::False);
        let mut formulas = vec![
            LtlNnf::Until(tt(), a0()),                                  // F a0
            LtlNnf::Release(ff(), a0()),                                // G a0
            LtlNnf::Release(ff(), Box::new(LtlNnf::Until(tt(), a0()))), // G F a0
            LtlNnf::Until(tt(), Box::new(LtlNnf::Release(ff(), n0()))), // F G ¬a0
        ];
        if num_atoms >= 2 {
            formulas.push(LtlNnf::Until(a0(), Box::new(LtlNnf::Atom(1)))); // a0 U a1
            formulas.push(LtlNnf::Release(
                ff(),
                Box::new(LtlNnf::Or(vec![
                    LtlNnf::NegAtom(0),
                    LtlNnf::Until(tt(), Box::new(LtlNnf::Atom(1))),
                ])),
            )); // G(a0 → F a1)
        }

        for (fi, formula) in formulas.iter().enumerate() {
            let neg = negate(formula);
            let gba = build_gba(&neg);

            let run_bfs = |disable_memo: bool| {
                on_the_fly_product_emptiness_with_limit_memo_toggle(
                    &gba,
                    &net,
                    &reduced,
                    &net,
                    &atoms,
                    usize::MAX,
                    50_000_000,
                    None,
                    disable_memo,
                )
                .expect("BFS product expands safely")
            };
            let bfs_memo = run_bfs(false);
            let bfs_direct = run_bfs(true);
            assert_eq!(
                bfs_memo, bfs_direct,
                "round {round} formula {fi}: BFS memo verdict diverged from direct eval"
            );

            for per_state_visibility in [false, true] {
                let por_ctx = PorContext {
                    dep: DependencyGraph::build(&net),
                    visible: ltl_visible_reduced_transitions(&atoms, &reduced),
                    per_state_visibility,
                };
                let run_por = |disable_memo: bool| {
                    on_the_fly_product_emptiness_por_memo_toggle(
                        &gba,
                        &net,
                        &reduced,
                        &net,
                        &atoms,
                        &por_ctx,
                        usize::MAX,
                        None,
                        disable_memo,
                    )
                    .expect("POR product expands safely")
                };
                let por_memo = run_por(false);
                let por_direct = run_por(true);
                assert_eq!(
                    por_memo, por_direct,
                    "round {round} formula {fi} psv={per_state_visibility}: \
                     POR memo verdict diverged from direct eval"
                );
                assert_eq!(
                    por_memo, bfs_memo,
                    "round {round} formula {fi} psv={per_state_visibility}: \
                     POR verdict diverged from exhaustive BFS"
                );
            }
        }
    }
}

#[test]
fn test_on_the_fly_matches_legacy_on_alternating_mutex() {
    // A(G(¬(p0>=1 ∧ p1>=1))) — mutual exclusion on alternating net.
    // The on-the-fly path should give the same result as the legacy path.
    use crate::examinations::ltl::check_ltl_properties;
    use crate::explorer::ExplorationConfig;
    use crate::output::Verdict;
    use crate::property_xml::{Formula, IntExpr, LtlFormula, Property, StatePredicate};

    let net = alternating_net();
    let props = vec![Property {
        id: "mutex".to_string(),
        formula: Formula::Ltl(LtlFormula::Globally(Box::new(LtlFormula::Not(Box::new(
            LtlFormula::And(vec![
                LtlFormula::Atom(StatePredicate::IntLe(
                    IntExpr::Constant(1),
                    IntExpr::TokensCount(vec!["p0".to_string()]),
                )),
                LtlFormula::Atom(StatePredicate::IntLe(
                    IntExpr::Constant(1),
                    IntExpr::TokensCount(vec!["p1".to_string()]),
                )),
            ]),
        ))))),
    }];

    let results = check_ltl_properties(&net, &props, &ExplorationConfig::default());
    assert_eq!(results[0].1, Verdict::True, "mutual exclusion should hold");
}
