// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Exact Petri-net class membership and the exact per-class liveness
//! certificates for state machines and marked graphs.
//!
//! The classifiers are O(arcs) and use **distinct-place semantics**: a
//! transition's preset/postset is the set of *distinct* places it touches.
//! Parallel unit arcs (the same place appearing twice in a transition's
//! inputs or outputs) have effective weight ≥ 2 under the firing semantics
//! ([`PetriNet::fire`] subtracts/adds per arc), so a net containing them is
//! reported as NOT ordinary — every class below requires ordinariness, hence
//! every certificate declines on such nets.
//!
//! ## Theorems (exact, both directions)
//!
//! - **Marked graph** (ordinary; every place has exactly one producer and
//!   one consumer transition): live ⟺ every directed circuit contains an
//!   initially marked place [Commoner, Holt, Even, Pnueli, "Marked directed
//!   graphs", JCSS 5(5):511–523, 1971; Murata 1989, Theorem 19].
//!   Equivalently: the transition digraph with one edge `•p → p•` per
//!   initially EMPTY place `p` is acyclic (a circuit with all places empty
//!   is exactly a cycle of this digraph, and circuit token counts are
//!   invariant under firing).
//! - **State machine** (ordinary; every transition has exactly one input
//!   and one output place): per weakly-connected component containing at
//!   least one transition, live ⟺ the component is strongly connected and
//!   holds ≥ 1 token [Murata 1989, §VI]. Tokens are conserved and an SC
//!   component can route any token to any input place; conversely a
//!   non-SC component's source SCC can be drained (or starts empty)
//!   permanently disabling its consumers, and a token-free component's
//!   transitions are dead outright.
//!
//! The free-choice Commoner–Hack certificate lives in
//! [`crate::structural`] (it reuses the complete minimal-siphon
//! enumeration there); this module supplies its class gate.

use crate::petri_net::PetriNet;

/// Net-class membership flags, computed once in O(arcs) by [`classify`].
///
/// All shape classes use distinct-place semantics and imply `ordinary`
/// (a non-ordinary net has every class flag false).
pub(crate) struct NetClass {
    /// Every arc weight is 1 AND no transition has parallel arcs to the
    /// same place (which would make the effective weight ≥ 2).
    pub ordinary: bool,
    /// Ordinary + simple free choice: every place with ≥ 2 (distinct)
    /// consumers only feeds transitions whose whole preset is that place.
    /// Equivalently: every transition with ≥ 2 distinct input places has
    /// exclusive ownership of each of them (`|p•| = 1` for each).
    /// This matches `structural.rs::is_ordinary_free_choice` exactly.
    pub free_choice: bool,
    /// Ordinary + every place has exactly one producer and exactly one
    /// consumer transition (distinct counts; self-loops allowed).
    pub marked_graph: bool,
    /// Ordinary + every transition has exactly one distinct input place
    /// and exactly one distinct output place (self-loops allowed).
    pub state_machine: bool,
    /// Every place is touched by at least one arc. An isolated unmarked
    /// place is a vacuous uncovered singleton siphon that falsifies the
    /// only-if direction of Commoner's theorem, so the free-choice
    /// certificate (and, belt-and-braces, the state-machine certificate)
    /// must decline when this is false.
    pub all_places_incident: bool,
    /// Some transition has an empty preset. Source transitions are outside
    /// the textbook free-choice "system" setting, so the Commoner
    /// certificate declines on them (belt-and-braces; they cannot feed any
    /// siphon, but the theorem statements we cite do not cover them).
    pub has_source_transition: bool,
}

/// Classify `net` in O(arcs). See [`NetClass`] for the exact definitions.
pub(crate) fn classify(net: &PetriNet) -> NetClass {
    let np = net.num_places();
    let nt = net.num_transitions();

    let mut producers = vec![0usize; np]; // distinct producing transitions
    let mut consumers = vec![0usize; np]; // distinct consuming transitions
    let mut incident = vec![false; np];
    // Epoch markers for per-transition duplicate detection (O(arcs) total).
    let mut mark = vec![0u64; np];
    let mut epoch = 0u64;

    let mut ordinary = true;
    let mut has_source_transition = false;
    let mut state_machine = nt > 0;

    for transition in &net.transitions {
        if transition.inputs.is_empty() {
            has_source_transition = true;
        }
        epoch += 1;
        let mut distinct_inputs = 0usize;
        for arc in &transition.inputs {
            let p = arc.place.0 as usize;
            incident[p] = true;
            if arc.weight != 1 {
                ordinary = false;
            }
            if mark[p] == epoch {
                // Parallel arc: effective weight ≥ 2 → not ordinary.
                ordinary = false;
            } else {
                mark[p] = epoch;
                distinct_inputs += 1;
                consumers[p] += 1;
            }
        }
        epoch += 1;
        let mut distinct_outputs = 0usize;
        for arc in &transition.outputs {
            let p = arc.place.0 as usize;
            incident[p] = true;
            if arc.weight != 1 {
                ordinary = false;
            }
            if mark[p] == epoch {
                ordinary = false;
            } else {
                mark[p] = epoch;
                distinct_outputs += 1;
                producers[p] += 1;
            }
        }
        if distinct_inputs != 1 || distinct_outputs != 1 {
            state_machine = false;
        }
    }

    // Free choice: ∀t with |•t| ≥ 2: ∀p ∈ •t: |p•| = 1. Equivalent to the
    // place-side definition (|p•| > 1 ⇒ ∀t ∈ p•: •t = {p}): a violation of
    // either is a pair (p, t) with p ∈ •t, |•t| ≥ 2, |p•| ≥ 2. Under
    // `ordinary` there are no duplicate arcs, so raw arc counts are
    // distinct-place counts.
    let free_choice = ordinary
        && net.transitions.iter().all(|transition| {
            transition.inputs.len() < 2
                || transition
                    .inputs
                    .iter()
                    .all(|arc| consumers[arc.place.0 as usize] == 1)
        });

    let marked_graph =
        ordinary && nt > 0 && (0..np).all(|p| producers[p] == 1 && consumers[p] == 1);

    NetClass {
        ordinary,
        free_choice,
        marked_graph,
        state_machine: ordinary && state_machine,
        all_places_incident: incident.iter().all(|&seen| seen),
        has_source_transition,
    }
}

/// Exact liveness of a **marked graph** (caller must have verified
/// `classify(net).marked_graph`).
///
/// **Theorem** [Commoner–Holt–Even–Pnueli 1971; Murata 1989, Thm 19]:
/// a marked graph is live ⟺ every directed circuit contains an initially
/// marked place ⟺ the transition digraph with one edge `•p → p•` per
/// initially empty place `p` is acyclic.
///
/// Soundness of both directions:
/// - A circuit whose places are all empty keeps token count 0 forever
///   (each circuit place's unique producer and consumer are its circuit
///   neighbours, so no other transition touches it), hence every circuit
///   transition is dead → not live.
/// - If every circuit is marked, the empty-place digraph is acyclic at M0
///   and stays acyclic at every reachable marking (circuit counts are
///   invariant); a transition with an empty input place has its (unique)
///   producing predecessor strictly earlier in the topological order, so by
///   induction every transition can always be re-enabled → live.
///
/// O(places + transitions + arcs) via Kahn's algorithm.
pub(crate) fn marked_graph_live(net: &PetriNet) -> bool {
    let np = net.num_places();
    let nt = net.num_transitions();
    let mut producer = vec![usize::MAX; np];
    let mut consumer = vec![usize::MAX; np];
    for (tidx, transition) in net.transitions.iter().enumerate() {
        for arc in &transition.outputs {
            producer[arc.place.0 as usize] = tidx;
        }
        for arc in &transition.inputs {
            consumer[arc.place.0 as usize] = tidx;
        }
    }

    // Edge •p → p• for each initially empty place p.
    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); nt];
    let mut indegree = vec![0usize; nt];
    for p in 0..np {
        if net.initial_marking[p] == 0 {
            let (from, to) = (producer[p], consumer[p]);
            debug_assert!(
                from != usize::MAX && to != usize::MAX,
                "marked-graph shape guarantees one producer and one consumer per place"
            );
            adj[from].push(to);
            indegree[to] += 1;
        }
    }

    // Kahn topological sort: acyclic ⟺ every node drained.
    let mut queue: Vec<usize> = (0..nt).filter(|&t| indegree[t] == 0).collect();
    let mut drained = 0usize;
    while let Some(t) = queue.pop() {
        drained += 1;
        for &succ in &adj[t] {
            indegree[succ] -= 1;
            if indegree[succ] == 0 {
                queue.push(succ);
            }
        }
    }
    drained == nt
}

/// Exact liveness of a **state machine** (caller must have verified
/// `classify(net).state_machine && classify(net).all_places_incident`).
///
/// **Theorem** [Murata 1989, §VI]: per weakly-connected component with at
/// least one transition, live ⟺ the component's place digraph (one edge
/// `•t → t•` per transition) is strongly connected AND the component holds
/// ≥ 1 token initially. The net is live ⟺ every such component is.
///
/// Soundness of both directions (token count per component is invariant):
/// - SC + marked ⇒ live: a token can be routed from wherever it sits to any
///   transition's input place (each routing step `a → b` needs only `a`
///   marked, so the walk is always fireable).
/// - Not SC ⇒ not live: the condensation has a source SCC with an outgoing
///   edge (weak connectivity); its tokens can be drained out and can never
///   re-enter, after which every transition consuming inside it is dead —
///   and such a transition exists (an internal edge, or the outgoing one).
/// - SC + token-free ⇒ not live: every place stays empty forever, so every
///   transition of the component (it has one, and its input place is in the
///   component) is dead.
///
/// O(places + transitions) via union-find (weak components) + per-component
/// forward/backward BFS (strong connectivity).
pub(crate) fn state_machine_live(net: &PetriNet) -> bool {
    let np = net.num_places();

    let mut fwd: Vec<Vec<usize>> = vec![Vec::new(); np];
    let mut bwd: Vec<Vec<usize>> = vec![Vec::new(); np];
    let mut parent: Vec<usize> = (0..np).collect();
    fn find(parent: &mut [usize], x: usize) -> usize {
        let mut root = x;
        while parent[root] != root {
            root = parent[root];
        }
        let mut cur = x;
        while parent[cur] != root {
            let next = parent[cur];
            parent[cur] = root;
            cur = next;
        }
        root
    }

    let mut component_has_transition = vec![false; np];
    for transition in &net.transitions {
        let a = transition.inputs[0].place.0 as usize;
        let b = transition.outputs[0].place.0 as usize;
        fwd[a].push(b);
        bwd[b].push(a);
        let (ra, rb) = (find(&mut parent, a), find(&mut parent, b));
        parent[ra] = rb;
    }
    for transition in &net.transitions {
        let root = find(&mut parent, transition.inputs[0].place.0 as usize);
        component_has_transition[root] = true;
    }

    // Group component members.
    let mut members: Vec<Vec<usize>> = vec![Vec::new(); np];
    for p in 0..np {
        let root = find(&mut parent, p);
        members[root].push(p);
    }

    let mut seen = vec![0u32; np];
    let mut stamp = 0u32;
    let mut stack: Vec<usize> = Vec::new();
    let mut reach = |adj: &[Vec<usize>], start: usize, seen: &mut [u32], stamp: u32| -> usize {
        let mut count = 1usize;
        seen[start] = stamp;
        stack.clear();
        stack.push(start);
        while let Some(p) = stack.pop() {
            for &q in &adj[p] {
                if seen[q] != stamp {
                    seen[q] = stamp;
                    count += 1;
                    stack.push(q);
                }
            }
        }
        count
    };

    for root in 0..np {
        if !component_has_transition[root] || members[root].is_empty() {
            // Transition-free components (only possible without the
            // all-places-incident gate) impose no liveness constraint.
            continue;
        }
        let component = &members[root];
        // ≥ 1 token in the component (token count is invariant).
        if component.iter().all(|&p| net.initial_marking[p] == 0) {
            return false;
        }
        // Strong connectivity: forward AND backward reachability from any
        // member must cover the whole (weakly-connected) member set.
        let start = component[0];
        stamp += 1;
        if reach(&fwd, start, &mut seen, stamp) != component.len() {
            return false;
        }
        stamp += 1;
        if reach(&bwd, start, &mut seen, stamp) != component.len() {
            return false;
        }
    }

    true
}

#[cfg(test)]
#[path = "net_class_tests.rs"]
mod tests;
