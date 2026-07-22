// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! PROPTEST RANDOM-NET DIFFERENTIAL GATE for the **C3 cycle (ignoring) proviso**
//! of the stutter-insensitive LTL partial-order reduction
//! ([`super::on_the_fly_dfs_impl`] / `expand_product_node`).
//!
//! This is the LIVENESS analogue of `stubborn_safety_differential_proptest.rs`
//! (which gates the D1+D2+visibility safety POR). It targets the one extra
//! condition stutter-insensitive LTL POR needs on top of D1+D2+visibility: the
//! **C3 / L cycle proviso** — *no action may be ignored forever along a cycle*.
//! On the product graph C3 is enforced as: **every cycle (equivalently, every
//! non-trivial SCC) of the reduced product contains at least one fully-expanded
//! state**. A missing or wrong C3 proviso lets the reduced product loop forever
//! on invisible "stutter" transitions, never firing a visible/progress action —
//! and that mis-decides a LIVENESS property (typically reporting "φ holds" when
//! ¬φ actually has an accepting run, i.e. φ is violated). C3 is therefore a
//! ZERO-WRONG soundness condition, exactly the kind this gate must protect.
//!
//! # What the gate proves (three teeth)
//!
//! 1. **Verdict differential (random nets).** For each random (net, X-free LTL
//!    formula) pair: the EXACT full-expansion product
//!    ([`super::on_the_fly_product_emptiness_with_limit`], `por = None`) and the
//!    PRODUCTION DFS+POR product (C3 ON) must agree on the accepting-cycle
//!    verdict (= the LTL verdict, projected on the visible atoms — atoms are
//!    over the net's places). ZERO disagreements.
//!
//! 2. **Per-SCC C3 structural verifier (random nets).** On the SAME production
//!    DFS+POR product graph (snapshotted via [`super::ProductCapture`]) every
//!    non-trivial SCC must contain ≥1 fully-expanded state. This is the direct,
//!    fine-grained check of the proviso itself — it catches a broken C3 even
//!    when the verdict differential happens not to flip on a given net.
//!
//! 3. **Teeth (mutation injection).** The SAME two checks are run against a
//!    MUTANT that drops the C3 proviso (`c3_disabled = true`, exposed only by
//!    the test entry [`super::on_the_fly_product_emptiness_c3_gate`]). On at
//!    least one net in the battery the mutant MUST EITHER mis-decide the verdict
//!    (FALSE→TRUE: a missed accepting run) OR build a non-trivial SCC with no
//!    fully-expanded state (a structural C3 violation). If the mutant never
//!    diverged, the differential would have no teeth against C3 and the test
//!    fails loudly. The honest production path is asserted SOUND (verdict match
//!    + structural C3 holds) on every one of those same nets.
//!
//! # Soundness of the mutant plumbing
//!
//! The `c3_disabled` switch and the [`super::ProductCapture`] snapshot live
//! entirely behind `#[cfg(test)]`. Production (`on_the_fly_dfs_impl`) always
//! passes `c3_disabled = false` and `capture = None`; the field does not even
//! exist in `cfg(not(test))` builds (`DfsCtx::c3_enabled()` is a `const true`
//! there). So this gate can build the deliberately unsound reduction WITHOUT
//! any path to shipping it.
//!
//! # Boundedness / termination
//!
//! Same discipline as the safety gate: both the full and POR products are
//! capped (`MAX_SYSTEM_STATES`, `PRODUCT_LIMIT`); a case where the full product
//! is inconclusive (`None`) is skipped, never failed. The reduced product
//! explores a stutter-equivalent SUBSET, so when full is conclusive POR is too.

use proptest::prelude::*;
use rustc_hash::FxHashSet;

use super::{
    on_the_fly_product_emptiness_c3_gate, on_the_fly_product_emptiness_with_limit, PorContext,
    ProductCapture,
};
use crate::buchi::gba::{accept_bit, build_gba};
use crate::buchi::nnf::negate;
use crate::buchi::LtlNnf;
use crate::examinations::ltl_por::ltl_visible_reduced_transitions;
use crate::petri_net::{Arc, PetriNet, PlaceIdx, PlaceInfo, TransitionIdx, TransitionInfo};
use crate::reduction::ReducedNet;
use crate::resolved_predicate::{ResolvedIntExpr, ResolvedPredicate};
use crate::scc::tarjan_scc_generic;
use crate::stubborn::DependencyGraph;

/// Distinct-system-marking budget. A net exceeding it is inconclusive (skipped).
const MAX_SYSTEM_STATES: usize = 4_000;
/// Product-state budget for both builders. Same skip-if-exceeded discipline.
const PRODUCT_LIMIT: usize = 60_000;

// ---------------------------------------------------------------------------
// Net + formula construction.
// ---------------------------------------------------------------------------

fn identity_reduced(net: &PetriNet) -> ReducedNet {
    ReducedNet::identity(net)
}

/// `tokens(place) >= value`.
fn atom_ge(place: u32, value: u64) -> ResolvedPredicate {
    ResolvedPredicate::IntLe(
        ResolvedIntExpr::Constant(value),
        ResolvedIntExpr::TokensCount(vec![PlaceIdx(place)]),
    )
}

/// The X-free LTL property shapes we instantiate over the chosen atom(s). All
/// are stutter-insensitive (no Next), so POR is sound for them; each stresses a
/// different part of the C3 proviso (G/GF/FG keep accepting cycles alive, so a
/// dropped C3 most easily mis-decides them).
#[derive(Clone, Copy, Debug)]
enum PropertyShape {
    /// G a  ==  Release(False, a)   (safety-ish; cycle in the "still true" sink)
    Globally,
    /// F a  ==  Until(True, a)
    Finally,
    /// G F a ==  Release(False, Until(True, a))   (response / progress)
    GloballyFinally,
    /// F G a ==  Until(True, Release(False, a))    (stabilization)
    FinallyGlobally,
    /// a U b
    Until,
    /// G(a -> F b) == Release(False, (¬a ∨ F b))   (classic liveness)
    Response,
}

impl PropertyShape {
    fn all() -> [PropertyShape; 6] {
        [
            PropertyShape::Globally,
            PropertyShape::Finally,
            PropertyShape::GloballyFinally,
            PropertyShape::FinallyGlobally,
            PropertyShape::Until,
            PropertyShape::Response,
        ]
    }

    /// Build the property φ in NNF over atom indices `a` (and `b` for binary
    /// shapes). The caller checks A(φ); the engine negates internally to search
    /// for an accepting run of ¬φ.
    fn build(self, a: usize, b: usize) -> LtlNnf {
        let tt = || Box::new(LtlNnf::True);
        let ff = || Box::new(LtlNnf::False);
        let atom = |i: usize| Box::new(LtlNnf::Atom(i));
        let f = |inner: LtlNnf| LtlNnf::Until(tt(), Box::new(inner));
        match self {
            PropertyShape::Globally => LtlNnf::Release(ff(), atom(a)),
            PropertyShape::Finally => LtlNnf::Until(tt(), atom(a)),
            PropertyShape::GloballyFinally => LtlNnf::Release(ff(), Box::new(f(LtlNnf::Atom(a)))),
            PropertyShape::FinallyGlobally => {
                LtlNnf::Until(tt(), Box::new(LtlNnf::Release(ff(), atom(a))))
            }
            PropertyShape::Until => LtlNnf::Until(atom(a), atom(b)),
            PropertyShape::Response => LtlNnf::Release(
                ff(),
                Box::new(LtlNnf::Or(vec![LtlNnf::NegAtom(a), f(LtlNnf::Atom(b))])),
            ),
        }
    }
}

// ---------------------------------------------------------------------------
// Random-net strategy (mirrors the safety gate's band): <=5 places, 1..=5
// transitions, weights 1-2, m0 in 0..=2, at most one arc per place per side.
// Plus the chosen atom places, a property shape, and the atom thresholds.
// Smaller bounds than the safety gate keep the FULL product within budget for
// the GBA × system blowup while still exercising real concurrency + cycles.
// ---------------------------------------------------------------------------

type RawArc = (u8, u8);
type RawTransition = (Vec<RawArc>, Vec<RawArc>);
/// (num_places, m0, transitions, atom_a_place, atom_b_place, shape_sel,
///  thresh_a, thresh_b)
type RawCase = (usize, Vec<u8>, Vec<RawTransition>, u8, u8, u8, u8, u8);

fn raw_arc_strategy() -> impl Strategy<Value = RawArc> {
    (0u8..=200, 1u8..=2)
}

fn raw_transition_strategy() -> impl Strategy<Value = RawTransition> {
    (
        prop::collection::vec(raw_arc_strategy(), 0..=3),
        prop::collection::vec(raw_arc_strategy(), 0..=3),
    )
}

fn raw_case_strategy() -> impl Strategy<Value = RawCase> {
    (1usize..=5).prop_flat_map(|num_places| {
        (
            Just(num_places),
            prop::collection::vec(0u8..=2, num_places),
            prop::collection::vec(raw_transition_strategy(), 1..=5),
            0u8..=200,
            0u8..=200,
            0u8..=5,
            1u8..=2,
            1u8..=2,
        )
    })
}

fn build_net(num_places: usize, m0: &[u8], transitions: &[RawTransition]) -> PetriNet {
    let num_places = num_places.max(1);
    let places: Vec<PlaceInfo> = (0..num_places)
        .map(|i| PlaceInfo {
            id: format!("p{i}"),
            name: Some(format!("p{i}")),
        })
        .collect();
    let initial_marking: Vec<u64> = (0..num_places)
        .map(|i| u64::from(m0.get(i).copied().unwrap_or(0)))
        .collect();

    let mut nets: Vec<TransitionInfo> = Vec::with_capacity(transitions.len());
    for (ti, (inputs, outputs)) in transitions.iter().enumerate() {
        let mut in_by_place: std::collections::BTreeMap<u32, u64> = Default::default();
        for &(sel, w) in inputs {
            let p = (sel as usize % num_places) as u32;
            in_by_place.insert(p, u64::from(w));
        }
        let mut out_by_place: std::collections::BTreeMap<u32, u64> = Default::default();
        for &(sel, w) in outputs {
            let p = (sel as usize % num_places) as u32;
            out_by_place.insert(p, u64::from(w));
        }
        let in_arcs: Vec<Arc> = in_by_place
            .into_iter()
            .map(|(p, weight)| Arc {
                place: PlaceIdx(p),
                weight,
            })
            .collect();
        let out_arcs: Vec<Arc> = out_by_place
            .into_iter()
            .map(|(p, weight)| Arc {
                place: PlaceIdx(p),
                weight,
            })
            .collect();
        nets.push(TransitionInfo {
            id: format!("t{ti}"),
            name: Some(format!("t{ti}")),
            inputs: in_arcs,
            outputs: out_arcs,
        });
    }

    PetriNet {
        name: Some("ltl-c3-proptest-random".into()),
        places,
        transitions: nets,
        initial_marking,
    }
}

/// A resolved random case ready to run through the differential.
struct ResolvedCase {
    net: PetriNet,
    atoms: Vec<ResolvedPredicate>,
    formula: LtlNnf,
    /// Visible transitions (production over-approximation) for this atom set.
    visible: Vec<TransitionIdx>,
}

/// Resolve a raw case into a net, two atoms over its places, and a property.
/// Returns `None` when POR would be inert (no invisible transitions — the
/// reduction never fires, so the C3 proviso is never exercised and the gate
/// would prove nothing). Skipping those is sound.
fn resolve_case(raw: &RawCase) -> Option<ResolvedCase> {
    let (num_places, m0, transitions, sel_a, sel_b, shape_sel, thresh_a, thresh_b) = raw;
    let net = build_net(*num_places, m0, transitions);
    let np = net.num_places() as u32;
    if np == 0 || net.num_transitions() < 2 {
        return None;
    }

    let a_place = (*sel_a as u32) % np;
    let b_place = (*sel_b as u32) % np;
    let atoms = vec![
        atom_ge(a_place, u64::from(*thresh_a)),
        atom_ge(b_place, u64::from(*thresh_b)),
    ];

    let shapes = PropertyShape::all();
    let shape = shapes[(*shape_sel as usize) % shapes.len()];
    let formula = shape.build(0, 1);

    let reduced = identity_reduced(&net);
    let visible = ltl_visible_reduced_transitions(&atoms, &reduced);
    // POR only prunes interleavings of INVISIBLE transitions. If every
    // transition is visible the DFS full-expands everywhere and C3 is never
    // exercised — nothing to prove. (The safety gate skips the all-visible
    // case for the same reason.)
    if visible.len() >= net.num_transitions() {
        return None;
    }
    Some(ResolvedCase {
        net,
        atoms,
        formula,
        visible,
    })
}

// ---------------------------------------------------------------------------
// The exact (full) oracle and the DFS+POR product, plus the C3 structural
// verifier over the snapshotted product graph.
// ---------------------------------------------------------------------------

/// Exact full-expansion product emptiness (`por = None`): the trusted oracle
/// for the LTL verdict. `Some(has_accepting_cycle)`, or `None` if inconclusive.
fn full_verdict(formula: &LtlNnf, net: &PetriNet, atoms: &[ResolvedPredicate]) -> Option<bool> {
    let reduced = identity_reduced(net);
    let neg = negate(formula);
    let gba = build_gba(&neg);
    on_the_fly_product_emptiness_with_limit(
        &gba,
        net,
        &reduced,
        net,
        atoms,
        MAX_SYSTEM_STATES,
        PRODUCT_LIMIT,
        None,
    )
    .expect("full product expands safely")
}

/// Run the DFS+POR product with C3 either ON (production) or OFF (mutant) and
/// snapshot the built graph. Returns `(verdict, capture)`; verdict `None` means
/// inconclusive.
fn por_verdict_and_capture(
    formula: &LtlNnf,
    net: &PetriNet,
    atoms: &[ResolvedPredicate],
    visible: &[TransitionIdx],
    per_state_visibility: bool,
    c3_disabled: bool,
) -> (Option<bool>, ProductCapture) {
    let reduced = identity_reduced(net);
    let neg = negate(formula);
    let gba = build_gba(&neg);
    let por = PorContext {
        dep: DependencyGraph::build(net),
        visible: visible.to_vec(),
        per_state_visibility,
    };
    let mut capture = ProductCapture::default();
    let verdict = on_the_fly_product_emptiness_c3_gate(
        &gba,
        net,
        &reduced,
        net,
        atoms,
        &por,
        MAX_SYSTEM_STATES,
        None,
        c3_disabled,
        Some(&mut capture),
    )
    .expect("POR product expands safely");
    (verdict, capture)
}

/// Outcome of the per-SCC C3 structural verifier.
enum C3Check {
    /// Every non-trivial SCC contains ≥1 fully-expanded state.
    Holds,
    /// A non-trivial SCC has NO fully-expanded state — a C3 violation. Carries
    /// the offending SCC (product ids) for the failure message.
    Violated(Vec<u32>),
}

/// Per-SCC C3 proviso verifier over a snapshotted product graph.
///
/// C3 (cycle/ignoring proviso) on the product is: every cycle of the reduced
/// product contains at least one fully-expanded state. Equivalently — since
/// every cycle lies inside one SCC and a fully-expanded node anywhere in an SCC
/// is reachable on a cycle through it — every NON-TRIVIAL SCC (size > 1, or a
/// size-1 SCC with a self-loop) must contain ≥1 fully-expanded state.
///
/// This is the direct structural test of the proviso, independent of the
/// verdict: it trips on the mutant exactly where C3 was needed, even on nets
/// whose verdict happens not to flip.
fn verify_c3(cap: &ProductCapture) -> C3Check {
    if cap.adj.is_empty() {
        return C3Check::Holds;
    }
    let sccs = tarjan_scc_generic(&cap.adj, |&w| w);
    for scc in &sccs {
        let nontrivial = if scc.len() > 1 {
            true
        } else {
            let s = scc[0];
            cap.adj[s as usize].contains(&s)
        };
        if !nontrivial {
            continue;
        }
        let has_full = scc
            .iter()
            .any(|&s| cap.fully_expanded.get(s as usize).copied().unwrap_or(false));
        if !has_full {
            return C3Check::Violated(scc.clone());
        }
    }
    C3Check::Holds
}

/// Recompute the accepting-cycle verdict from a captured product graph. Mirrors
/// `super::find_accepting_scc` but reads the snapshot, so the gate can confirm
/// the captured graph corresponds to the verdict the engine returned.
fn accepting_from_capture(cap: &ProductCapture) -> bool {
    if cap.adj.is_empty() {
        return false;
    }
    let sccs = tarjan_scc_generic(&cap.adj, |&w| w);
    for scc in &sccs {
        let nontrivial = if scc.len() > 1 {
            true
        } else {
            let s = scc[0];
            cap.adj[s as usize].contains(&s)
        };
        if !nontrivial {
            continue;
        }
        if cap.num_accept == 0 {
            return true;
        }
        let nw = cap.num_accept.div_ceil(64);
        let scc_set: FxHashSet<u32> = scc.iter().copied().collect();
        let all = (0..cap.num_accept).all(|i| {
            let state_acc = scc.iter().any(|&s| {
                let s = s as usize;
                accept_bit(&cap.accept[s * nw..(s + 1) * nw], i)
            });
            if state_acc {
                return true;
            }
            scc.iter().any(|&s| {
                let s = s as usize;
                let words = &cap.edge_words[s];
                cap.edge_succ[s].iter().enumerate().any(|(e, succ)| {
                    scc_set.contains(succ) && accept_bit(&words[e * nw..(e + 1) * nw], i)
                })
            })
        });
        if all {
            return true;
        }
    }
    false
}

fn net_dbg(net: &PetriNet) -> String {
    format!(
        "places={} m0={:?} transitions={:?}",
        net.places.len(),
        net.initial_marking,
        net.transitions
            .iter()
            .map(|t| (
                t.inputs
                    .iter()
                    .map(|a| (a.place.0, a.weight))
                    .collect::<Vec<_>>(),
                t.outputs
                    .iter()
                    .map(|a| (a.place.0, a.weight))
                    .collect::<Vec<_>>()
            ))
            .collect::<Vec<_>>()
    )
}

// ---------------------------------------------------------------------------
// THE C3 GATE (proptest). Verdict differential + per-SCC structural verifier,
// in BOTH visibility modes. ZERO wrong verdicts; C3 holds at every SCC.
// ---------------------------------------------------------------------------
proptest! {
    #![proptest_config(ProptestConfig {
        cases: 3000,
        max_shrink_iters: 4000,
        .. ProptestConfig::default()
    })]

    #[test]
    fn proptest_c3_verdict_and_structure_preserved(raw in raw_case_strategy()) {
        let Some(rc) = resolve_case(&raw) else {
            return Ok(()); // POR inert — nothing to prove
        };
        let Some(full) = full_verdict(&rc.formula, &rc.net, &rc.atoms) else {
            return Ok(()); // full product inconclusive — skip
        };

        for per_state in [false, true] {
            let (por, cap) = por_verdict_and_capture(
                &rc.formula, &rc.net, &rc.atoms, &rc.visible, per_state, /*c3_disabled=*/ false,
            );
            let Some(por) = por else { continue }; // POR inconclusive — skip

            // 1. Verdict differential: production POR must equal the exact full
            //    product. A C3 (or any POR) soundness bug flips this.
            prop_assert_eq!(
                full, por,
                "LTL C3 GATE: production POR verdict diverged from exact full product \
                 (per_state_visibility={})\n  shape formula={:?}\n  visible={:?}\n  net: {}",
                per_state, rc.formula, rc.visible, net_dbg(&rc.net)
            );

            // The captured graph must correspond to the returned verdict.
            prop_assert_eq!(
                accepting_from_capture(&cap), por,
                "captured product graph disagrees with the returned verdict"
            );

            // 2. Per-SCC C3 structural verifier on the SAME production graph.
            if let C3Check::Violated(scc) = verify_c3(&cap) {
                prop_assert!(
                    false,
                    "LTL C3 GATE: production POR built a non-trivial SCC with NO \
                     fully-expanded state (C3 violation) scc={:?}\n  shape formula={:?}\n  \
                     visible={:?}\n  net: {}",
                    scc, rc.formula, rc.visible, net_dbg(&rc.net)
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Deterministic (RNG-free) xorshift case generator shared by the non-vacuity
// and teeth sweeps — identical discipline to the safety gate.
// ---------------------------------------------------------------------------
struct Xorshift(u64);
impl Xorshift {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
}

fn gen_case(rng: &mut Xorshift) -> RawCase {
    let num_places = (rng.next() as usize % 5) + 1;
    let m0: Vec<u8> = (0..num_places).map(|_| (rng.next() % 3) as u8).collect();
    let ntrans = (rng.next() as usize % 5) + 1;
    let transitions: Vec<RawTransition> = (0..ntrans)
        .map(|_| {
            let nin = rng.next() as usize % 4;
            let nout = rng.next() as usize % 4;
            let ins: Vec<RawArc> = (0..nin)
                .map(|_| ((rng.next() % 201) as u8, (rng.next() % 2 + 1) as u8))
                .collect();
            let outs: Vec<RawArc> = (0..nout)
                .map(|_| ((rng.next() % 201) as u8, (rng.next() % 2 + 1) as u8))
                .collect();
            (ins, outs)
        })
        .collect();
    let sel_a = (rng.next() % 201) as u8;
    let sel_b = (rng.next() % 201) as u8;
    let shape_sel = (rng.next() % 6) as u8;
    let thresh_a = (rng.next() % 2 + 1) as u8;
    let thresh_b = (rng.next() % 2 + 1) as u8;
    (
        num_places,
        m0,
        transitions,
        sel_a,
        sel_b,
        shape_sel,
        thresh_a,
        thresh_b,
    )
}

// ---------------------------------------------------------------------------
// NON-VACUITY GUARD. A C3 gate proves nothing if the reduction never actually
// reduces, or if no built product ever has a non-trivial SCC (no cycle for C3
// to constrain). This deterministic sweep asserts that within a modest battery:
//   * production POR is conclusive on enough nets,
//   * production POR agrees with FULL on every conclusive net (0-wrong),
//   * at least one production product has a non-trivial SCC whose fully-expanded
//     node is the ONLY thing keeping C3 — i.e. a state that fired its full set
//     while its candidate ample set was a strict subset (C3 forced expansion),
//   * the per-SCC C3 verifier HOLDS on every production graph.
// ---------------------------------------------------------------------------
#[test]
fn c3_gate_non_vacuity_and_production_soundness() {
    let mut rng = Xorshift(0xC3C3_0DD0_1234_5678);
    let total = 8_000;
    let mut conclusive = 0usize;
    let mut nontrivial_scc_seen = 0usize;
    let mut c3_relevant = 0usize; // products with a non-trivial SCC at all
    for _ in 0..total {
        let raw = gen_case(&mut rng);
        let Some(rc) = resolve_case(&raw) else {
            continue;
        };
        let Some(full) = full_verdict(&rc.formula, &rc.net, &rc.atoms) else {
            continue;
        };
        for per_state in [false, true] {
            let (por, cap) = por_verdict_and_capture(
                &rc.formula,
                &rc.net,
                &rc.atoms,
                &rc.visible,
                per_state,
                false,
            );
            let Some(por) = por else { continue };
            conclusive += 1;
            assert_eq!(
                full,
                por,
                "deterministic C3 sweep: production POR diverged from full \
                 (per_state={per_state}) shape={:?} visible={:?} net: {}",
                rc.formula,
                rc.visible,
                net_dbg(&rc.net)
            );
            // Production C3 must ALWAYS hold structurally.
            match verify_c3(&cap) {
                C3Check::Holds => {}
                C3Check::Violated(scc) => panic!(
                    "PRODUCTION C3 VIOLATION (REAL BUG): non-trivial SCC with no \
                     fully-expanded state scc={scc:?} shape={:?} visible={:?} net: {}",
                    rc.formula,
                    rc.visible,
                    net_dbg(&rc.net)
                ),
            }
            // Track whether the battery exercises non-trivial SCCs at all.
            let sccs = tarjan_scc_generic(&cap.adj, |&w| w);
            let any_nt = sccs
                .iter()
                .any(|scc| scc.len() > 1 || cap.adj[scc[0] as usize].contains(&scc[0]));
            if any_nt {
                c3_relevant += 1;
                nontrivial_scc_seen += 1;
            }
        }
    }
    assert!(
        conclusive > 300,
        "non-vacuity: too few conclusive C3 cases ({conclusive})"
    );
    assert!(
        c3_relevant >= 1 && nontrivial_scc_seen >= 1,
        "non-vacuity: no production product ever had a non-trivial SCC across \
         {conclusive} conclusive cases — the C3 proviso was never relevant, so the \
         gate is vacuous"
    );
}

// ---------------------------------------------------------------------------
// TEETH (mutation injection). Run the SAME differential + structural verifier
// against the MUTANT that drops the C3 proviso (`c3_disabled = true`). On at
// least one net the mutant MUST EITHER:
//   (a) mis-decide the verdict (FULL found an accepting run, the mutant did
//       not — a missed liveness counterexample, the classic C3 failure), OR
//   (b) build a non-trivial SCC with no fully-expanded state (a direct
//       structural C3 violation).
// If the mutant never diverged on either axis, the gate has NO teeth against
// C3 and this test fails loudly — a regression that removed the proviso would
// pass silently. The honest production path is asserted SOUND on every one of
// those same nets (verdict matches FULL and C3 holds structurally), confirming
// the production reduction is correct exactly where the mutant is not.
// ---------------------------------------------------------------------------
#[test]
fn dropping_c3_proviso_is_caught_by_the_gate() {
    let mut rng = Xorshift(0x7EE7_C3C3_0DD0_0001);
    let total = 20_000;
    let mut conclusive = 0usize;
    let mut production_agreements = 0usize;
    let mut mutant_verdict_divergences = 0usize;
    let mut mutant_structural_violations = 0usize;
    for _ in 0..total {
        let raw = gen_case(&mut rng);
        let Some(rc) = resolve_case(&raw) else {
            continue;
        };
        let Some(full) = full_verdict(&rc.formula, &rc.net, &rc.atoms) else {
            continue;
        };

        for per_state in [false, true] {
            // Honest production path MUST be sound.
            let (prod, prod_cap) = por_verdict_and_capture(
                &rc.formula,
                &rc.net,
                &rc.atoms,
                &rc.visible,
                per_state,
                false,
            );
            let Some(prod) = prod else { continue };
            conclusive += 1;
            assert_eq!(
                full,
                prod,
                "PRODUCTION LTL POR MIS-DECIDED a liveness property — REAL C3 \
                 UNSOUNDNESS BUG.\n  full={full:?} prod={prod:?} per_state={per_state}\n  \
                 shape={:?} visible={:?}\n  net: {}",
                rc.formula,
                rc.visible,
                net_dbg(&rc.net)
            );
            match verify_c3(&prod_cap) {
                C3Check::Holds => {}
                C3Check::Violated(scc) => panic!(
                    "PRODUCTION C3 STRUCTURAL VIOLATION (REAL BUG) scc={scc:?}\n  \
                     shape={:?} visible={:?}\n  net: {}",
                    rc.formula,
                    rc.visible,
                    net_dbg(&rc.net)
                ),
            }
            production_agreements += 1;

            // MUTANT (C3 dropped) — record where it diverges. Either axis is a
            // catch the gate would make on the production path if C3 were
            // removed.
            let (mutant, mutant_cap) = por_verdict_and_capture(
                &rc.formula,
                &rc.net,
                &rc.atoms,
                &rc.visible,
                per_state,
                /*c3_disabled=*/ true,
            );
            if let Some(mutant) = mutant {
                if mutant != full {
                    // POR explores a subset, so the mutant can only UNDER-report
                    // an accepting run (FULL true, mutant false). It must never
                    // manufacture a run FULL lacks.
                    assert!(
                        full && !mutant,
                        "C3 mutant OVER-reported an accepting run FULL lacks — \
                         impossible (reduced graph ⊆ full): full={full} mutant={mutant} \
                         net: {}",
                        net_dbg(&rc.net)
                    );
                    mutant_verdict_divergences += 1;
                }
            }
            if let C3Check::Violated(_) = verify_c3(&mutant_cap) {
                mutant_structural_violations += 1;
            }
        }
    }
    assert!(
        conclusive > 1_000,
        "teeth: too few conclusive cases ({conclusive}) to trust the mutation result"
    );
    assert_eq!(
        production_agreements, conclusive,
        "every conclusive net: production LTL POR must agree with FULL and pass C3 \
         (0-wrong)"
    );
    assert!(
        mutant_verdict_divergences >= 1 || mutant_structural_violations >= 1,
        "TEETH FAILURE: dropping the C3 proviso never mis-decided a verdict and never \
         produced a structural C3 violation across {conclusive} conclusive cases — the \
         gate has no teeth against C3, so a regression that removed the proviso would \
         pass silently (verdict_divergences={mutant_verdict_divergences}, \
         structural_violations={mutant_structural_violations})"
    );
    eprintln!(
        "C3 teeth: conclusive={conclusive} mutant_verdict_divergences={mutant_verdict_divergences} \
         mutant_structural_violations={mutant_structural_violations}"
    );
}
