// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

use super::bsgs::Bsgs;
use crate::petri_net::PetriNet;
use std::collections::{BTreeMap, BTreeSet};

/// Hard termination budget for the individualization–refinement automorphism
/// search ([`find_automorphism_generators_with_budget`]).
///
/// ## Why a budget exists at all
/// The search is a nauty/bliss-style IR backtracking tree: at each node it
/// individualizes one vertex out of a non-trivial target cell and recurses,
/// discovering an automorphism whenever two leaves share a refinement trace.
/// Orbit pruning (`get_orbits` + `seen_orbits`) keeps the tree polynomial on
/// well-behaved graphs, but on adversarial / highly-regular graphs the IR tree
/// can blow up super-polynomially. Both caps are pure *runtime* guards that
/// force termination; neither affects SOUNDNESS — a search that stops early
/// simply discovers a SUBSET of the true generators, yielding a (still-exact,
/// merely smaller) orbit reduction. Widening the budget only finds MORE of the
/// real symmetry group, never a wrong one.
///
/// The original fixed cap (64 generators / 50 ms) under-approximated the group
/// on LARGE symmetric families: e.g. a 43-source full-S₄₃ star truncated to a
/// size-7 orbit, leaving the orbit quotient ≈ the concrete space so StateSpace
/// BFS still timed out → CANNOT_COMPUTE. The wider [`Self::thorough`] budget
/// (opt-in, used only by the StateSpace lane) lets those families collapse to
/// a tiny quotient and complete.
#[derive(Debug, Clone, Copy)]
pub struct SymmetryBudget {
    /// Maximum number of generators to collect before stopping. Bounds the
    /// breadth of discovered symmetry (each generator unions some places).
    pub max_generators: usize,
    /// Wall-clock deadline for the whole search. Hard guarantee of termination
    /// regardless of `max_generators` (the IR tree can be wide before 64
    /// generators are even found).
    pub time_budget: std::time::Duration,
    /// Run the O(places) STRUCTURAL full-symmetric orbit seeder
    /// ([`seed_full_symmetric_generators`]) BEFORE the IR backtracking search.
    ///
    /// ## Why this is the load-bearing widening (empirical finding)
    /// Raising the generator cap / time budget ALONE does NOT grow the orbit on
    /// large full-Sₙ families: the IR backtracking re-discovers thousands of
    /// redundant small-support generators (measured: ~9000 generators for a
    /// 96-source star, orbit still only 14) without expanding the orbit, because
    /// each leaf-derived generator permutes only the few deepest individualized
    /// points. The seeder sidesteps that blowup by grouping places by a cheap
    /// structural color (initial marking + sorted incidence signature),
    /// VERIFYING each candidate class is a genuine full-symmetric automorphism
    /// group (every in-class transposition is an H1+H2 place-swap automorphism —
    /// the SAME check the soundness gate `place_orbits_are_full_symmetric` uses),
    /// and emitting the `class-1` connecting transpositions so the whole class
    /// becomes ONE orbit in O(class²·|T|). Every seeded generator is a verified
    /// automorphism, so this only ADDS real symmetry — it cannot make a count
    /// wrong (the gate re-verifies independently). Off by default; on only for
    /// the StateSpace thorough lane.
    pub seed_structural: bool,
}

impl SymmetryBudget {
    /// The historical default budget (64 generators / 50 ms). Used by every
    /// examination EXCEPT the StateSpace lane, so Reachability/OneSafe symmetry
    /// discovery is byte-for-byte unchanged.
    #[must_use]
    pub const fn default_budget() -> Self {
        Self {
            max_generators: 64,
            time_budget: std::time::Duration::from_millis(50),
            seed_structural: false,
        }
    }

    /// The widened, opt-in budget for the StateSpace orbit-quotient lane. Caps
    /// are raised an order of magnitude (256 generators / 500 ms) so large
    /// symmetric families (full-Sₙ stars, big direct products) discover their
    /// FULL group and collapse to a tiny quotient. Still HARD-bounded — the
    /// search always terminates by one of the two caps; never unbounded.
    ///
    /// Both caps are overridable at runtime via `TY_SYMMETRY_THOROUGH`:
    ///   * `TY_SYMMETRY_THOROUGH=0` / `false` / `off` — kill-switch: fall back
    ///     to [`Self::default_budget`] even for StateSpace.
    ///   * `TY_SYMMETRY_THOROUGH=<gens>` — set `max_generators` to `<gens>`.
    ///   * `TY_SYMMETRY_THOROUGH=<gens>:<ms>` — set generators AND time budget.
    ///
    /// Anything else (e.g. `1`/`on`) keeps the compiled defaults below.
    #[must_use]
    pub fn thorough() -> Self {
        const THOROUGH_GENERATORS: usize = 256;
        const THOROUGH_MILLIS: u64 = 500;

        let raw = std::env::var("TY_SYMMETRY_THOROUGH").ok();
        match raw.as_deref().map(str::trim) {
            // Kill-switch: thorough discovery disabled, use the historical cap.
            Some("0") | Some("false") | Some("FALSE") | Some("off") | Some("OFF") => {
                Self::default_budget()
            }
            Some(spec) if spec.contains(':') => {
                let mut parts = spec.splitn(2, ':');
                let gens = parts
                    .next()
                    .and_then(|s| s.trim().parse::<usize>().ok())
                    .filter(|&g| g > 0)
                    .unwrap_or(THOROUGH_GENERATORS);
                let ms = parts
                    .next()
                    .and_then(|s| s.trim().parse::<u64>().ok())
                    .filter(|&m| m > 0)
                    .unwrap_or(THOROUGH_MILLIS);
                Self {
                    max_generators: gens,
                    time_budget: std::time::Duration::from_millis(ms),
                    seed_structural: true,
                }
            }
            Some(spec) => {
                // Bare integer overrides only the generator cap; otherwise
                // (e.g. "1"/"on"/"true") keep the compiled thorough defaults.
                let gens = spec
                    .parse::<usize>()
                    .ok()
                    .filter(|&g| g > 0)
                    .unwrap_or(THOROUGH_GENERATORS);
                Self {
                    max_generators: gens,
                    time_budget: std::time::Duration::from_millis(THOROUGH_MILLIS),
                    seed_structural: true,
                }
            }
            None => Self {
                max_generators: THOROUGH_GENERATORS,
                time_budget: std::time::Duration::from_millis(THOROUGH_MILLIS),
                seed_structural: true,
            },
        }
    }
}

impl Default for SymmetryBudget {
    fn default() -> Self {
        Self::default_budget()
    }
}

/// Discover automorphism generators under the historical default budget. Thin
/// wrapper over [`find_automorphism_generators_with_budget`]; preserves the
/// exact behaviour of every existing (non-StateSpace) caller.
pub fn find_automorphism_generators(net: &PetriNet) -> Vec<(Vec<usize>, Vec<usize>)> {
    find_automorphism_generators_with_budget(net, SymmetryBudget::default_budget())
}

pub fn find_automorphism_generators_with_budget(
    net: &PetriNet,
    budget: SymmetryBudget,
) -> Vec<(Vec<usize>, Vec<usize>)> {
    let num_p = net.num_places();
    let num_t = net.transitions.len();
    let num_nodes = num_p + num_t;

    if num_nodes == 0 {
        return Vec::new();
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
    enum EdgeType {
        TInFromP(u64),
        TOutToP(u64),
        POutToT(u64),
        PInFromT(u64),
    }

    let mut edge_types = BTreeSet::new();
    for t in &net.transitions {
        for a in &t.inputs {
            edge_types.insert(EdgeType::TInFromP(a.weight));
            edge_types.insert(EdgeType::POutToT(a.weight));
        }
        for a in &t.outputs {
            edge_types.insert(EdgeType::TOutToP(a.weight));
            edge_types.insert(EdgeType::PInFromT(a.weight));
        }
    }

    let edge_type_to_id: BTreeMap<EdgeType, usize> = edge_types
        .into_iter()
        .enumerate()
        .map(|(i, t)| (t, i))
        .collect();
    let num_colors = edge_type_to_id.len();

    let mut adj: Vec<Vec<Vec<u32>>> = vec![vec![Vec::new(); num_nodes]; num_colors];

    for (t_idx, t) in net.transitions.iter().enumerate() {
        let t_node = (num_p + t_idx) as u32;
        for a in &t.inputs {
            let p_node = a.place.0;
            let c1 = edge_type_to_id[&EdgeType::TInFromP(a.weight)];
            adj[c1][p_node as usize].push(t_node);

            let c3 = edge_type_to_id[&EdgeType::POutToT(a.weight)];
            adj[c3][t_node as usize].push(p_node);
        }
        for a in &t.outputs {
            let p_node = a.place.0;
            let c2 = edge_type_to_id[&EdgeType::TOutToP(a.weight)];
            adj[c2][p_node as usize].push(t_node);

            let c4 = edge_type_to_id[&EdgeType::PInFromT(a.weight)];
            adj[c4][t_node as usize].push(p_node);
        }
    }

    let mut initial_blocks: BTreeMap<u64, Vec<u32>> = BTreeMap::new();
    for p in 0..num_p {
        let marking = net.initial_marking[p];
        initial_blocks.entry(marking).or_default().push(p as u32);
    }

    let mut blocks: Vec<Vec<u32>> = Vec::new();
    for (_, group) in initial_blocks {
        blocks.push(group);
    }
    if num_t > 0 {
        let t_group: Vec<u32> = (num_p as u32..num_nodes as u32).collect();
        blocks.push(t_group);
    }

    let mut search = AutomorphismSearch {
        adj,
        num_nodes,
        best_trace: None,
        first_leaf: None,
        generators: Vec::new(),
        max_generators: budget.max_generators,
        deadline: std::time::Instant::now() + budget.time_budget,
    };

    search.search(
        blocks,
        std::collections::VecDeque::new(),
        Vec::new(),
        Vec::new(),
    );

    let mut generators = search.generators;

    // Opt-in (StateSpace thorough lane only): augment the IR-discovered
    // generators with STRUCTURAL full-symmetric orbit seeds. The IR search
    // alone under-approximates large full-Sₙ families (it produces thousands of
    // redundant small-support generators without growing the orbit); the seeder
    // recovers the FULL orbit in O(places) with VERIFIED automorphisms. See
    // [`SymmetryBudget::seed_structural`].
    if budget.seed_structural {
        seed_full_symmetric_generators(net, num_nodes, &mut generators);
    }

    generators
}

/// Append STRUCTURAL full-symmetric place-orbit generators to `generators`.
///
/// Groups places by a cheap structural color (`initial_marking` + the sorted
/// multiset of `(edge-role, weight)` incidences), then for each color class of
/// size ≥ 2 VERIFIES — using the exact H1 (equal initial marking) + H2 (the
/// place-swap is a transition-multiset automorphism) test that the soundness
/// gate [`place_orbits_are_full_symmetric`] itself uses — that *every* in-class
/// transposition is a genuine place-swap automorphism. Only then does it emit
/// the `|class|-1` connecting transpositions `(class[0], class[k])` so union-find
/// folds the whole class into ONE orbit.
///
/// SOUNDNESS: every emitted generator's place-domain support is a transposition
/// that has been individually verified to be an H1+H2 automorphism, so the
/// resulting orbits are a subset of the TRUE place-symmetry group — never a
/// fictitious symmetry. The transition coordinates of the emitted node
/// permutation are left as the identity because every downstream consumer
/// (`get_orbits` over places, `canonicalize`, the multinomial/BSGS count, and
/// the independent re-verification in `place_orbits_are_full_symmetric`) uses
/// only the PLACE coordinates `gen[0..num_p]`; the transition coordinates of a
/// generator are never consulted for the place-symmetry quotient. This makes
/// the seeder a pure COMPLETENESS improvement that cannot affect any count.
///
/// BOUNDED: the color partition is O(arcs·log). Per class the H2 verification is
/// `O(|class|²·|T|)`; to keep a pathological net (one giant class) from
/// dominating, classes larger than `MAX_VERIFIED_CLASS` are skipped (the IR
/// search still gets its shot; we simply do not pay the quadratic seed cost).
fn seed_full_symmetric_generators(
    net: &PetriNet,
    num_nodes: usize,
    generators: &mut Vec<(Vec<usize>, Vec<usize>)>,
) {
    // Cap the quadratic per-class verification. A class this large already
    // collapses an astronomically symmetric quotient; we never need more.
    const MAX_VERIFIED_CLASS: usize = 4096;

    let num_p = net.num_places();
    if num_p < 2 {
        return;
    }

    // Per-place structural color: initial marking + sorted incidence multiset.
    // Two places with different colors can NEVER be swapped by an automorphism,
    // so this is a sound (over-)partition we then verify exactly.
    let mut color_of: Vec<(u64, Vec<(u8, u64)>)> = vec![(0, Vec::new()); num_p];
    for (p, color) in color_of.iter_mut().enumerate() {
        color.0 = net.initial_marking[p];
    }
    for t in &net.transitions {
        for a in &t.inputs {
            // role 0: place is an INPUT to some transition (weight w).
            color_of[a.place.0 as usize].1.push((0u8, a.weight));
        }
        for a in &t.outputs {
            // role 1: place is an OUTPUT of some transition (weight w).
            color_of[a.place.0 as usize].1.push((1u8, a.weight));
        }
    }
    for color in &mut color_of {
        color.1.sort_unstable();
    }

    let mut classes: BTreeMap<(u64, Vec<(u8, u64)>), Vec<u32>> = BTreeMap::new();
    for (p, color) in color_of.into_iter().enumerate() {
        classes.entry(color).or_default().push(p as u32);
    }

    // The identity transition-signature multiset, computed once and reused to
    // verify each candidate transposition (H2). This is the SAME primitive the
    // soundness gate uses, so a seeded class is full-symmetric by exactly the
    // gate's definition.
    let identity_sig = transition_signature_multiset(net, None);

    for class in classes.into_values() {
        if class.len() < 2 || class.len() > MAX_VERIFIED_CLASS {
            continue;
        }

        // Verify EVERY in-class transposition is an H1+H2 automorphism. (Equal
        // initial marking is guaranteed by the color, but H2 is not implied by
        // the color — two equal-degree places can fail to be swappable, as
        // `rejects_equal_degree_places_that_are_not_automorphic` shows — so we
        // must check it.) If any pair fails, the class is NOT full-symmetric;
        // skip it and let the IR search discover whatever real subgroup exists.
        let mut full_symmetric = true;
        'verify: for (i, &left) in class.iter().enumerate() {
            for &right in &class[(i + 1)..] {
                debug_assert_eq!(
                    net.initial_marking[left as usize], net.initial_marking[right as usize],
                    "color groups equal initial markings",
                );
                if identity_sig != transition_signature_multiset(net, Some((left, right))) {
                    full_symmetric = false;
                    break 'verify;
                }
            }
        }
        if !full_symmetric {
            continue;
        }

        // Emit the connecting transpositions (class[0] ↔ class[k]) as
        // place-domain node permutations with identity on transitions.
        let anchor = class[0] as usize;
        for &other in &class[1..] {
            let other = other as usize;
            let mut pi: Vec<usize> = (0..num_nodes).collect();
            pi.swap(anchor, other);
            let support = vec![anchor, other];
            generators.push((pi, support));
        }
    }
}

fn get_orbits(n: usize, generators: &[(Vec<usize>, Vec<usize>)], fixed: &[usize]) -> Vec<usize> {
    let mut parent = (0..n).collect::<Vec<_>>();
    fn find(parent: &mut [usize], i: usize) -> usize {
        if parent[i] == i {
            i
        } else {
            let p = parent[i];
            parent[i] = find(parent, p);
            parent[i]
        }
    }
    fn union(parent: &mut [usize], i: usize, j: usize) {
        let root_i = find(parent, i);
        let root_j = find(parent, j);
        if root_i != root_j {
            parent[root_i] = root_j;
        }
    }

    for (gen, support) in generators {
        if fixed.iter().all(|&v| gen[v] == v) {
            for &i in support {
                union(&mut parent, i, gen[i]);
            }
        }
    }

    for i in 0..n {
        find(&mut parent, i);
    }
    parent
}

fn refine(
    adj: &[Vec<Vec<u32>>],
    num_nodes: usize,
    mut blocks: Vec<Vec<u32>>,
    mut queue: std::collections::VecDeque<(usize, usize)>,
) -> (Vec<Vec<u32>>, Vec<usize>) {
    let mut node_to_block = vec![0; num_nodes];
    for (b, group) in blocks.iter().enumerate() {
        for &u in group {
            node_to_block[u as usize] = b;
        }
    }

    let num_colors = adj.len();
    if queue.is_empty() {
        for b in 0..blocks.len() {
            for c in 0..num_colors {
                queue.push_back((b, c));
            }
        }
    }

    let mut counts = vec![0; num_nodes];
    let mut active_nodes_in_block: Vec<Vec<u32>> = vec![Vec::new(); blocks.len()];
    let mut active_blocks = Vec::new();

    let mut trace = vec![blocks.len()];

    while let Some((s_id, c)) = queue.pop_front() {
        let s_len = blocks[s_id].len();
        if s_len == 0 {
            continue;
        }

        let mut active_nodes = Vec::new();
        for i in 0..s_len {
            let v = blocks[s_id][i] as usize;
            for &u in &adj[c][v] {
                let uu = u as usize;
                if counts[uu] == 0 {
                    active_nodes.push(u);
                }
                counts[uu] += 1;
            }
        }

        for &u in &active_nodes {
            let b = node_to_block[u as usize];
            if active_nodes_in_block[b].is_empty() {
                active_blocks.push(b);
            }
            active_nodes_in_block[b].push(u);
        }

        for &b in &active_blocks {
            let mut b_nodes = std::mem::take(&mut active_nodes_in_block[b]);
            b_nodes.sort_unstable_by_key(|&u| counts[u as usize]);

            let mut groups = Vec::new();
            let mut current_count = counts[b_nodes[0] as usize];
            let mut current_group = Vec::new();

            for &u in &b_nodes {
                if counts[u as usize] == current_count {
                    current_group.push(u);
                } else {
                    groups.push(current_group);
                    current_count = counts[u as usize];
                    current_group = vec![u];
                }
            }
            groups.push(current_group);

            let has_zeros = b_nodes.len() < blocks[b].len();
            if !has_zeros && groups.len() == 1 {
                continue;
            }

            let mut zeros = Vec::new();
            if has_zeros {
                for &u in &blocks[b] {
                    if counts[u as usize] == 0 {
                        zeros.push(u);
                    }
                }
                groups.push(zeros);
            }

            let mut largest_idx = 0;
            let mut largest_size = groups[0].len();
            for (i, g) in groups.iter().enumerate().skip(1) {
                if g.len() > largest_size {
                    largest_size = g.len();
                    largest_idx = i;
                }
            }

            blocks[b] = groups.swap_remove(largest_idx);
            for g in &groups {
                trace.push(g.len());
            }

            for g in groups {
                let new_b = blocks.len();
                for &u in &g {
                    node_to_block[u as usize] = new_b;
                }
                blocks.push(g);
                for c2 in 0..num_colors {
                    queue.push_back((new_b, c2));
                }
            }
        }

        for &u in &active_nodes {
            counts[u as usize] = 0;
        }
        active_blocks.clear();

        if active_nodes_in_block.len() < blocks.len() {
            active_nodes_in_block.resize(blocks.len(), Vec::new());
        }
    }

    for b in &mut blocks {
        b.sort_unstable();
    }

    (blocks, trace)
}

fn is_automorphism(adj: &[Vec<Vec<u32>>], pi: &[usize]) -> bool {
    let num_colors = adj.len();
    let n = pi.len();
    for c in 0..num_colors {
        for u in 0..n {
            let pu = pi[u];
            if adj[c][u].len() != adj[c][pu].len() {
                return false;
            }
            for &v in &adj[c][u] {
                let pv = pi[v as usize] as u32;
                if !adj[c][pu].contains(&pv) {
                    return false;
                }
            }
        }
    }
    true
}

struct AutomorphismSearch {
    adj: Vec<Vec<Vec<u32>>>,
    num_nodes: usize,
    best_trace: Option<Vec<usize>>,
    first_leaf: Option<Vec<usize>>,
    generators: Vec<(Vec<usize>, Vec<usize>)>,
    /// Hard cap on the number of generators collected before the search stops.
    /// One of the two termination guards (the other is `deadline`).
    max_generators: usize,
    deadline: std::time::Instant,
}

impl AutomorphismSearch {
    fn search(
        &mut self,
        partition: Vec<Vec<u32>>,
        queue: std::collections::VecDeque<(usize, usize)>,
        fixed: Vec<usize>,
        mut current_trace: Vec<usize>,
    ) {
        if self.generators.len() >= self.max_generators || std::time::Instant::now() > self.deadline
        {
            return;
        }
        let (refined, trace_step) = refine(&self.adj, self.num_nodes, partition, queue);
        current_trace.extend(trace_step);

        if let Some(best) = &self.best_trace {
            for i in 0..current_trace.len() {
                if i < best.len() && current_trace[i] != best[i] {
                    return;
                }
            }
        }

        if refined.iter().all(|b| b.len() == 1) {
            let leaf_ordering: Vec<usize> = refined.iter().map(|b| b[0] as usize).collect();

            if let Some(first) = self.first_leaf.as_ref() {
                let mut pi = vec![0; self.num_nodes];
                for i in 0..self.num_nodes {
                    pi[first[i]] = leaf_ordering[i];
                }

                if is_automorphism(&self.adj, &pi) {
                    let support: Vec<usize> = (0..self.num_nodes).filter(|&i| pi[i] != i).collect();
                    self.generators.push((pi, support));
                }

                if self.generators.len() >= self.max_generators
                    || std::time::Instant::now() > self.deadline
                {
                    return;
                }
            } else {
                self.first_leaf = Some(leaf_ordering);
                self.best_trace = Some(current_trace);
            }
            return;
        }

        let target_idx = refined
            .iter()
            .enumerate()
            .filter(|(_, b)| b.len() > 1)
            .min_by_key(|(_, b)| b.len())
            .map(|(i, _)| i)
            .unwrap();

        let target_block = refined[target_idx].clone();

        let orbits = get_orbits(self.num_nodes, &self.generators, &fixed);
        let mut seen_orbits = std::collections::HashSet::new();

        for &v in &target_block {
            let orbit = orbits[v as usize];
            if seen_orbits.insert(orbit) {
                let mut new_partition = refined.clone();
                new_partition[target_idx] = vec![v];
                let rem: Vec<u32> = target_block.iter().copied().filter(|&x| x != v).collect();
                new_partition.insert(target_idx + 1, rem);

                let mut new_fixed = fixed.clone();
                new_fixed.push(v as usize);

                let mut next_queue = std::collections::VecDeque::new();
                for c in 0..self.adj.len() {
                    next_queue.push_back((target_idx, c));
                    next_queue.push_back((new_partition.len() - 1, c));
                }

                self.search(new_partition, next_queue, new_fixed, current_trace.clone());
                if self.generators.len() >= self.max_generators
                    || std::time::Instant::now() > self.deadline
                {
                    return;
                }
            }
        }
    }
}

pub fn discover_place_symmetry(net: &PetriNet) -> Vec<Vec<u32>> {
    let generators = find_automorphism_generators(net);
    let num_p = net.num_places();
    let num_t = net.transitions.len();
    if num_p == 0 {
        return Vec::new();
    }
    let orbits = get_orbits(num_p + num_t, &generators, &[]);

    let mut groups_map: BTreeMap<usize, Vec<u32>> = BTreeMap::new();
    for p in 0..num_p {
        groups_map.entry(orbits[p]).or_default().push(p as u32);
    }

    let mut groups = Vec::new();
    for (_, group) in groups_map {
        if group.len() > 1 {
            groups.push(group);
        }
    }
    groups.sort_unstable_by_key(|g| g[0]);
    groups
}

/// Normalized (inputs, outputs) signature multiset of every transition under
/// an optional place transposition. Used to *verify* that swapping two places
/// `(p, q)` within a discovered orbit is a genuine transition automorphism
/// (hypothesis H2 of the place-swap soundness proof,
/// `docs/theorems/2026-05-26-place-swap-symmetry-soundness.md` §3).
///
/// `transition_signature_multiset(net, None) ==
///  transition_signature_multiset(net, Some((p, q)))` exactly captures
/// "swapping p and q is a transition automorphism". This is the same WL-free
/// check the test suite uses, lifted into the library so the *runtime*
/// orbit-quotient count can gate on it before trusting the closed-form size.
fn transition_signature_multiset(
    net: &PetriNet,
    place_swap: Option<(u32, u32)>,
) -> BTreeMap<(Vec<(u32, u64)>, Vec<(u32, u64)>), usize> {
    fn swap(place: u32, place_swap: Option<(u32, u32)>) -> u32 {
        match place_swap {
            Some((l, r)) if place == l => r,
            Some((l, r)) if place == r => l,
            _ => place,
        }
    }
    fn sig(arcs: &[crate::petri_net::Arc], place_swap: Option<(u32, u32)>) -> Vec<(u32, u64)> {
        let mut s: Vec<(u32, u64)> = arcs
            .iter()
            .map(|a| (swap(a.place.0, place_swap), a.weight))
            .collect();
        s.sort_unstable();
        s
    }
    let mut signatures = BTreeMap::new();
    for t in &net.transitions {
        let signature = (sig(&t.inputs, place_swap), sig(&t.outputs, place_swap));
        *signatures.entry(signature).or_insert(0) += 1;
    }
    signatures
}

/// Returns `true` iff every discovered place orbit is a *full* symmetric group
/// on its places — i.e. every transposition `(p, q)` with `p, q` in the same
/// orbit is both initial-marking-preserving (H1) and a transition-multiset
/// automorphism (H2). This is precisely the hypothesis under which the
/// closed-form multinomial orbit size `|G|! / ∏_v c_v!` is *exact*: a strict
/// subgroup (e.g. a cyclic-only action) would make the factorial OVER-count.
///
/// `discover_place_symmetry` builds orbits by union-find over verified
/// automorphism generators, so this holds by construction; the check is a
/// cheap (O(Σ|orbit|² · |T|)) fail-closed runtime gate consulted before the
/// orbit-quotient count is trusted. The orbits are place-index-disjoint by
/// construction (each place lies in exactly one union-find component), which
/// the multinomial's direct-product factorization also requires.
#[must_use]
pub fn place_orbits_are_full_symmetric(net: &PetriNet, orbits: &[Vec<u32>]) -> bool {
    if orbits.is_empty() {
        return true;
    }
    // Disjointness: every place appears in at most one orbit. Holds by
    // construction (union-find components), asserted here so the
    // direct-product multinomial cannot silently mis-factor.
    let mut seen = std::collections::HashSet::new();
    for group in orbits {
        for &p in group {
            if !seen.insert(p) {
                return false;
            }
        }
    }

    let identity = transition_signature_multiset(net, None);
    for group in orbits {
        for (idx, &left) in group.iter().enumerate() {
            for &right in &group[(idx + 1)..] {
                if net.initial_marking[left as usize] != net.initial_marking[right as usize] {
                    return false;
                }
                if identity != transition_signature_multiset(net, Some((left, right))) {
                    return false;
                }
            }
        }
    }
    true
}

/// Maximum size of the precomputed permutation cache built by BFS-closure
/// of the automorphism generators. If the group is larger than this budget
/// the closure is truncated and the stored set is no longer closed under
/// composition (i.e. NOT a subgroup).
///
/// Callers that only need a deterministic orbit representative (e.g.
/// `canonicalize` for dedup) tolerate a truncated cache. Callers that need
/// the *exact* group (e.g. multiplying orbit sizes to recover the true
/// state count) MUST consult [`PetriCanonicalizer::closure_is_complete`].
pub const PETRI_CANONICALIZER_CLOSURE_BUDGET: usize = 500;

/// Maximum `|G|` for which the BSGS *coupled* orbit-quotient path is enabled.
///
/// The coupled path (used when the discovered orbits are NOT a direct product
/// of full symmetric groups) recovers each rep's `orbit_size` by an orbit BFS
/// bounded by `|orbit(m)| ≤ |G|` and its `canonical_image` by a pruned
/// minimal-image descent. The per-rep `orbit_size` BFS is `O(|orbit|·num_p)`,
/// so on nets with many reps a large `|G|` makes the per-rep weight dominate
/// the explored-rep savings. To keep per-rep cost bounded we admit the coupled
/// path only when the exact `|G|` (from the BSGS) is at most this budget;
/// larger coupled groups fall back to exact, un-reduced exploration (a missed
/// speedup, never a wrong count). The exact `|G|` comes from the BSGS, so this
/// never guesses.
///
/// Sized to admit the small coupling groups that appear in the MCC families and
/// COLLAPSE many reps cheaply (Anderson |G|=24, Philosophers cyclic |G|≈n).
/// Larger coupled groups (e.g. AirplaneLD |G|=7!=5040) are now ALSO countable
/// cheaply because `Bsgs::orbit_size` gained the non-enumerative regime-B
/// `|G|/|Stab|` fast path (measured ~3.6x faster than the enumerative BFS at
/// |G|=5040). Raising this is always SOUND (the count is exact for any finite
/// |G|); the only question is SPEED. The default is overridable at runtime via
/// `TY_MCC_BSGS_GROUP_BUDGET` so large-|G| nets the un-reduced fallback cannot
/// finish (e.g. AirplaneLD, which OOM/timeouts un-reduced) can opt into the
/// quotient without a rebuild. The exact `|G|` comes from the BSGS, so this
/// never guesses.
pub const PETRI_CANONICALIZER_BSGS_GROUP_BUDGET: u128 = 1024;

/// Effective coupled-group budget: the `TY_MCC_BSGS_GROUP_BUDGET` env override
/// when present and parseable, else [`PETRI_CANONICALIZER_BSGS_GROUP_BUDGET`].
/// Raising it only WIDENS admission of coupled groups to the exact
/// orbit-quotient (never changes a count — orbit sizes are exact for any |G|).
#[must_use]
fn bsgs_group_budget() -> u128 {
    std::env::var("TY_MCC_BSGS_GROUP_BUDGET")
        .ok()
        .and_then(|s| s.trim().parse::<u128>().ok())
        .unwrap_or(PETRI_CANONICALIZER_BSGS_GROUP_BUDGET)
}

/// Gate for the COUPLED (`GroupOrbit`) orbit-quotient. Now ON by default.
///
/// History: this was gated OFF because the per-successor `canonical_image`
/// minimal-image computation used a pruned base-image backtrack whose constant
/// factor was catastrophic — it turned the coupled quotient into a wall-clock
/// LOSS (CloudDeployment-PT-2a |G|=240: 103 s → timeout/CANNOT_COMPUTE;
/// Anderson-PT-05 |G|=120: ~18 s vs ~5 s un-reduced). `canonical_image` now
/// takes a FAST `O(|G|·n)` lex-min over a precomputed element list
/// ([`Bsgs::canonical_image`]), removing that wall: CloudDeployment 103 s → ~3 s
/// with the EXACT same 4807/87600, Anderson-PT-05 18 s → ~7 s with the EXACT
/// same 689901/2784245. The orbit-size half is already non-enumerative
/// (regime-B `|G|/|Stab|`). Both halves are differentially cross-checked against
/// brute force over whole marking cubes, so the count is EXACT for any finite
/// `|G|` admitted within budget — a wrong quotient cannot occur.
///
/// With those constants fixed the coupled quotient completes well within the MCC
/// timeout on every resolvable corpus net (slowest measured ≈7 s) while
/// collapsing the explored state space by `|G|`, so it is enabled by default to
/// make symmetry actually FIRE on the symmetric families (Anderson, Cloud-
/// Deployment, FlexibleBarrier, Philosophers cyclic, CSRepetitions, …). Groups
/// with `|G|` over [`bsgs_group_budget`] (default 1024) still fall back to exact,
/// un-reduced exploration — never a wrong count. Set `TY_MCC_COUPLED_QUOTIENT=0`
/// (or `false`/`off`) to force the prior un-reduced behaviour for A/B testing;
/// `TY_MCC_BSGS_GROUP_BUDGET=<N>` widens admission to larger coupled groups
/// (the count is exact whenever it completes).
#[must_use]
fn coupled_quotient_enabled() -> bool {
    match std::env::var("TY_MCC_COUPLED_QUOTIENT").ok().as_deref() {
        Some("0") | Some("false") | Some("FALSE") | Some("off") | Some("OFF") => false,
        Some(_) | None => true,
    }
}

/// How `PetriCanonicalizer` recovers the orbit count and canonical form. Both
/// the per-rep `orbit_size` weight AND the dedup `canonicalize` map dispatch on
/// this SINGLE field, so reps and weights provably index the same `G`-orbit
/// partition of the reachable markings (the soundness coupling: there is no
/// second source of truth that could drift).
#[derive(Debug, Clone)]
enum CountMode {
    /// No non-trivial place symmetry; `canonicalize` is the identity and every
    /// orbit size is 1.
    Identity,
    /// Every discovered orbit is a *full* symmetric group acting independently
    /// (a direct product `∏_j Sym(G_j)`). `canonicalize` = per-orbit ascending
    /// sort; `orbit_size` = closed-form multinomial. Fast path, exact.
    Multinomial,
    /// A general (possibly COUPLED / diagonal) group. `canonicalize` = `G`-orbit
    /// minimal image; `orbit_size` = `|orbit(m)|` via the BSGS. Exact for any
    /// finite group; admitted only when `|G| ≤ PETRI_CANONICALIZER_BSGS_GROUP_BUDGET`.
    GroupOrbit(Bsgs),
    /// A coupled group exists but could NOT be admitted exactly (no bounded
    /// BSGS / `|G|` over budget). The orbit-quotient count would be unsound, so
    /// the caller MUST refuse installation and fall back to exact, un-reduced
    /// exploration. `canonicalize`/`orbit_size` are never consulted in this
    /// mode (the caller does not install the canonicalizer).
    Refuse,
}

#[derive(Debug, Clone)]
pub struct PetriCanonicalizer {
    permutations: Vec<Vec<usize>>,
    /// Raw place-domain automorphism generators. The group they generate is
    /// the full place-symmetry group of the net regardless of whether the
    /// precomputed `permutations` cache was truncated by the budget.
    generators: Vec<(Vec<usize>, Vec<usize>)>,
    /// Disjoint place orbits (each of size ≥ 2) discovered by
    /// [`discover_place_symmetry`]. These drive BOTH the dedup canonical form
    /// ([`Self::canonicalize`], per-orbit ascending sort) AND the closed-form
    /// orbit-size count, so the two are guaranteed to use the *same* group
    /// decomposition — the coupling that makes the orbit-quotient state count
    /// exact (see `orbit_size::multinomial_orbit_size`).
    place_orbits: Vec<Vec<u32>>,
    /// `true` iff every discovered orbit is a *full* symmetric group on its
    /// places (every in-orbit transposition is a verified H1+H2 automorphism).
    /// Required for the closed-form multinomial orbit size to be exact; a
    /// strict subgroup would over-count. Verified at build time via
    /// [`place_orbits_are_full_symmetric`].
    orbits_full_symmetric: bool,
    /// `true` iff the BFS-closure converged within the budget, i.e. the
    /// stored `permutations` *is* the full group ⟨generators⟩.
    closure_complete: bool,
    /// The single source of truth coupling the canonical form and the orbit
    /// count. See [`CountMode`].
    count_mode: CountMode,
    num_p: usize,
}

impl PetriCanonicalizer {
    /// Build under the historical default automorphism-discovery budget. Used
    /// by every examination EXCEPT the StateSpace orbit-quotient lane, so
    /// Reachability/OneSafe symmetry discovery is unchanged.
    pub fn build(net: &PetriNet) -> Self {
        Self::build_with_budget(net, SymmetryBudget::default_budget())
    }

    /// Build under an explicit automorphism-discovery `budget`. The StateSpace
    /// lane passes [`SymmetryBudget::thorough`] so large symmetric families
    /// discover their FULL group (smaller orbit quotient); all other lanes pass
    /// the default. The budget affects only the COMPLETENESS of the discovered
    /// group — soundness (the `orbits_are_full_symmetric` gate, the BSGS exact
    /// order) is independent of how many generators were found.
    pub fn build_with_budget(net: &PetriNet, budget: SymmetryBudget) -> Self {
        let full_generators = find_automorphism_generators_with_budget(net, budget);
        eprintln!(
            "[topology] found {} automorphism generators (budget: {} generators / {} ms)",
            full_generators.len(),
            budget.max_generators,
            budget.time_budget.as_millis(),
        );
        let num_p = net.num_places();
        let num_t = net.transitions.len();

        // Derive the disjoint place orbits from the SAME generators used below
        // (rather than re-running `discover_place_symmetry`, which would invoke
        // the automorphism search a second time). This guarantees the orbit
        // groups driving the count and the canonical form are byte-identical.
        let place_orbits = if num_p == 0 {
            Vec::new()
        } else {
            let orbits = get_orbits(num_p + num_t, &full_generators, &[]);
            let mut groups_map: BTreeMap<usize, Vec<u32>> = BTreeMap::new();
            for p in 0..num_p {
                groups_map.entry(orbits[p]).or_default().push(p as u32);
            }
            let mut groups: Vec<Vec<u32>> = groups_map
                .into_values()
                .filter(|group| group.len() > 1)
                .collect();
            groups.sort_unstable_by_key(|g| g[0]);
            groups
        };
        let orbits_full_symmetric = place_orbits_are_full_symmetric(net, &place_orbits);

        let mut generators = Vec::new();
        for (gen, _) in full_generators {
            let place_gen = gen[0..num_p].to_vec();
            let support = (0..num_p).filter(|&i| place_gen[i] != i).collect();
            generators.push((place_gen, support));
        }

        // Compute group closure up to PETRI_CANONICALIZER_CLOSURE_BUDGET to
        // bound memory. If the BFS exits via the budget guard rather than
        // exhausting the queue, `closure_complete` is left `false`.
        let mut permutations = vec![];
        let identity: Vec<usize> = (0..num_p).collect();
        permutations.push(identity.clone());

        let mut active_generators = Vec::new();
        let mut closure_complete = true;

        for gen in &generators {
            active_generators.push(gen.clone());

            let mut next_visited = std::collections::HashSet::new();
            next_visited.insert(identity.clone());
            let mut queue = vec![identity.clone()];
            let mut next_perms = vec![identity.clone()];
            let mut overflow = false;

            while let Some(curr) = queue.pop() {
                if next_perms.len() >= PETRI_CANONICALIZER_CLOSURE_BUDGET {
                    overflow = true;
                    break;
                }
                for active_gen in &active_generators {
                    let mut next_perm = vec![0; num_p];
                    for i in 0..num_p {
                        next_perm[active_gen.0[i]] = curr[i];
                    }
                    if next_visited.insert(next_perm.clone()) {
                        next_perms.push(next_perm.clone());
                        queue.push(next_perm);
                    }
                }
            }

            if overflow {
                active_generators.pop();
                closure_complete = false;
                break;
            }
            permutations = next_perms;
        }

        eprintln!(
            "[topology] Precomputed {} permutations for fast orbit reduction (subgroup_closed=true, full_group={})",
            permutations.len(),
            closure_complete,
        );

        // Decide the count mode (the SINGLE source of truth coupling the
        // canonical form and the orbit-size weight). Order of preference:
        //   1. Identity      — no non-trivial place symmetry.
        //   2. Multinomial   — orbits are a direct product of full symmetric
        //                      groups (fast, exact, closure-budget-immune).
        //   3. GroupOrbit    — a general/coupled group with EXACT bounded |G|
        //                      (admitted only when |G| ≤ budget so per-rep cost
        //                      stays bounded; otherwise fall back to Identity?
        //                      No — fall back means refuse the quotient, which
        //                      the caller handles by NOT installing this
        //                      canonicalizer at all).
        // The coupled (GroupOrbit) decision is reported via a separate
        // accessor; `build` always returns a canonicalizer, and the caller's
        // admission gate refuses installation when the coupled path is
        // unavailable (mirroring the prior full-symmetric refusal).
        let count_mode = if place_orbits.is_empty() {
            CountMode::Identity
        } else if orbits_full_symmetric {
            CountMode::Multinomial
        } else {
            // Coupled (non-full-symmetric) group. Build a deterministic BSGS
            // over the place-domain generators and admit the orbit-quotient
            // ONLY if the exact group order is within budget (so the per-rep
            // group enumeration is bounded). Otherwise leave it un-installable
            // (the caller falls back to exact exploration).
            let place_gens: Vec<Vec<usize>> = generators.iter().map(|(g, _)| g.clone()).collect();
            let budget = bsgs_group_budget();
            match Bsgs::build(&place_gens, num_p) {
                Some(bsgs)
                    if coupled_quotient_enabled() && bsgs.order().is_some_and(|o| o <= budget) =>
                {
                    eprintln!(
                        "[topology] Coupled place-symmetry group |G|={} admitted for EXACT orbit-quotient count (BSGS).",
                        bsgs.order().unwrap(),
                    );
                    CountMode::GroupOrbit(bsgs)
                }
                other => {
                    // Refuse the coupled quotient and fall back to exact
                    // exploration. Report the ACTUAL refusal reason (not always
                    // "over budget"): the gate is
                    //   coupled_quotient_enabled() && order.is_some_and(|o| o <= budget)
                    // so a refusal is one of: the env force-off, |G| over
                    // budget, |G| overflowed u128, or no usable BSGS.
                    let reason = if !coupled_quotient_enabled() {
                        "coupled quotient disabled via TY_MCC_COUPLED_QUOTIENT=0".to_string()
                    } else {
                        match &other {
                            Some(bsgs) => match bsgs.order() {
                                Some(o) => format!("|G|={o} over budget {budget}"),
                                None => "|G| overflowed u128".to_string(),
                            },
                            None => "no usable BSGS".to_string(),
                        }
                    };
                    eprintln!(
                        "[topology] Coupled place-symmetry group not admitted ({reason}); falling back to exact exploration.",
                    );
                    CountMode::Refuse
                }
            }
        };

        Self {
            permutations,
            generators,
            place_orbits,
            orbits_full_symmetric,
            closure_complete,
            count_mode,
            num_p,
        }
    }

    /// `true` iff this canonicalizer applies no reduction — there is no
    /// non-trivial place symmetry, so [`Self::canonicalize`] is the identity.
    /// Driven by `place_orbits` (the per-orbit-sort canonical form's group
    /// decomposition), NOT the truncatable `permutations` cache, so a net
    /// whose group exceeds the closure budget is still reported non-empty.
    pub fn is_empty(&self) -> bool {
        self.place_orbits.is_empty()
    }

    /// The disjoint place orbits (each size ≥ 2) used by both the dedup
    /// canonical form and the closed-form orbit-size count. Empty when the net
    /// has no non-trivial place symmetry.
    pub fn place_orbits(&self) -> &[Vec<u32>] {
        &self.place_orbits
    }

    /// `true` iff every discovered orbit is a full symmetric group on its
    /// places (every in-orbit transposition is a verified H1+H2 automorphism).
    /// The closed-form multinomial orbit size is exact ONLY when this holds;
    /// callers that recover the true reachable-marking count from the quotient
    /// MUST refuse the quotient otherwise (fall back to exact exploration).
    pub fn orbits_are_full_symmetric(&self) -> bool {
        self.orbits_full_symmetric
    }

    /// `true` iff the orbit-quotient state count is available and EXACT for
    /// this canonicalizer — i.e. the caller may install it and trust
    /// [`Self::canonicalize`] + [`Self::orbit_size`] to recover `|R|`/`|E|`
    /// exactly. This is the admission predicate that GENERALIZES the prior
    /// `orbits_are_full_symmetric()` gate:
    ///   * `Identity`    — no symmetry: trivially available (orbit sizes are 1).
    ///   * `Multinomial` — full-symmetric direct product: available (fast path).
    ///   * `GroupOrbit`  — a coupled group with bounded EXACT `|G|`: available
    ///     via the BSGS (one rep per `G`-orbit, exact weights).
    ///   * `Refuse`      — a coupled group that could not be admitted exactly:
    ///     NOT available, caller must fall back to exact
    ///     exploration (never a wrong count).
    pub fn quotient_is_available(&self) -> bool {
        !matches!(self.count_mode, CountMode::Refuse)
    }

    /// `true` iff this canonicalizer uses the coupled BSGS orbit-quotient path
    /// (as opposed to the multinomial fast path or no symmetry). Diagnostic /
    /// test aid; the count itself is exact in either case.
    pub fn uses_coupled_quotient(&self) -> bool {
        matches!(self.count_mode, CountMode::GroupOrbit(_))
    }

    /// The exact order `|G|` of the place-symmetry group when known (coupled
    /// BSGS path), else `None`. Diagnostic.
    pub fn group_order(&self) -> Option<u128> {
        match &self.count_mode {
            CountMode::GroupOrbit(bsgs) => bsgs.order(),
            _ => None,
        }
    }

    /// `|orbit(m)|` — the number of distinct concrete markings in `m`'s
    /// place-symmetry orbit. The StateSpace observer multiplies each canonical
    /// representative by this to recover `|R|`/`|E|` exactly:
    ///   |R| = Σ_reps |orbit(rep)|,  |E| = Σ_reps |orbit(source)|·deg.
    ///
    /// Dispatches on the SAME `count_mode` field as [`Self::canonicalize`], so
    /// the weight and the rep always index the same `G`-orbit partition.
    ///
    /// Returns `None` (fail-closed → CANNOT_COMPUTE) on `u64` overflow of the
    /// orbit size — never a truncated wrong number.
    #[must_use]
    pub fn orbit_size(&self, marking: &[u64]) -> Option<u64> {
        match &self.count_mode {
            CountMode::Identity | CountMode::Refuse => Some(1),
            CountMode::Multinomial => {
                super::orbit_size::multinomial_orbit_size(&self.place_orbits, marking)
            }
            CountMode::GroupOrbit(bsgs) => bsgs.orbit_size(marking),
        }
    }

    /// Raw place-domain automorphism generators. The group they generate is
    /// the full place-symmetry group of the underlying net regardless of
    /// whether the precomputed permutation cache was truncated. Consumers
    /// that need the exact group when [`Self::closure_is_complete`] returns
    /// `false` must enumerate from these on the fly.
    pub fn permutations(&self) -> &[Vec<usize>] {
        &self.permutations
    }

    pub fn generators(&self) -> Vec<Vec<usize>> {
        self.generators.iter().map(|(g, _)| g.clone()).collect()
    }

    /// `true` iff the precomputed permutation cache is the full group
    /// ⟨generators⟩ (BFS-closure converged within the budget). When `false`
    /// the cache is a strict subset and is NOT closed under composition; any
    /// caller that multiplies orbit sizes (or relies on the set being a
    /// subgroup) MUST refuse and fall back to enumerating from
    /// [`Self::generators`].
    pub fn closure_is_complete(&self) -> bool {
        self.closure_complete
    }

    /// Map `marking` to the canonical representative of its place-symmetry
    /// orbit by sorting, within each discovered orbit, the token counts in
    /// non-decreasing order (places outside every orbit are left untouched).
    ///
    /// This is the map `C` analyzed in
    /// `docs/theorems/2026-05-26-place-swap-symmetry-soundness.md` §2.4
    /// ("sorts `m ↾ G` in non-decreasing order"). Because the discovered
    /// orbits are place-index-disjoint, the full symmetry group is the direct
    /// product `∏_j Sym(G_j)`, whose orbits factor coordinate-wise; the
    /// lex-min of the product orbit is the concatenation of the per-orbit
    /// lex-mins, so sorting each orbit independently realizes the *true*
    /// canonical form. Unlike the previous lex-min-over-`permutations`
    /// implementation, this is O(places·log places) with NO permutation cache,
    /// so it is immune to `PETRI_CANONICALIZER_CLOSURE_BUDGET` truncation and
    /// always yields exactly one representative per `∏_j Sym(G_j)`-orbit —
    /// the same group decomposition the closed-form orbit-size count uses
    /// (`orbit_size::multinomial_orbit_size`), which is what makes the
    /// orbit-quotient state count exact.
    ///
    /// For a COUPLED group (`CountMode::GroupOrbit`) the per-orbit sort is NOT a
    /// valid canonical form (two markings in the same `G`-orbit can sort
    /// differently, and two in different `G`-orbits can collide), so this
    /// dispatches to the true `G`-orbit minimal image computed from the BSGS —
    /// exactly one representative per `G`-orbit, matching the `orbit_size`
    /// weights from the SAME `count_mode`.
    pub fn canonicalize(&self, marking: &mut [u64]) {
        match &self.count_mode {
            CountMode::Identity | CountMode::Refuse => {}
            CountMode::Multinomial => {
                let mut scratch: Vec<u64> = Vec::new();
                for group in &self.place_orbits {
                    scratch.clear();
                    scratch.extend(group.iter().map(|&p| marking[p as usize]));
                    scratch.sort_unstable();
                    for (slot, &p) in group.iter().enumerate() {
                        marking[p as usize] = scratch[slot];
                    }
                }
            }
            CountMode::GroupOrbit(bsgs) => {
                let image = bsgs.canonical_image(marking);
                marking.copy_from_slice(&image);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::petri_net::{Arc, PetriNet, PlaceIdx, PlaceInfo, TransitionInfo};

    /// RAII guard that enables the gated `TY_MCC_COUPLED_QUOTIENT` quotient for
    /// the duration of a test and restores the prior value on drop.
    ///
    /// Production [`coupled_quotient_enabled`] keys off `var_os(..).is_some()`, so
    /// leaving this var set leaks into ANY later/concurrent test that builds a
    /// `PetriCanonicalizer`. The guard holds the single crate-wide env lock so
    /// the set/restore cannot race another env-touching test, and the restore in
    /// `Drop` runs even if the test panics.
    struct CoupledQuotientEnvGuard {
        _lock: std::sync::MutexGuard<'static, ()>,
        prev: Option<std::ffi::OsString>,
    }

    impl CoupledQuotientEnvGuard {
        fn enabled() -> Self {
            let _lock = crate::env_test_lock();
            let prev = std::env::var_os("TY_MCC_COUPLED_QUOTIENT");
            crate::env_guard::set_var("TY_MCC_COUPLED_QUOTIENT", "1");
            Self { _lock, prev }
        }
    }

    impl Drop for CoupledQuotientEnvGuard {
        fn drop(&mut self) {
            match &self.prev {
                Some(value) => crate::env_guard::set_var("TY_MCC_COUPLED_QUOTIENT", value),
                None => crate::env_guard::remove_var("TY_MCC_COUPLED_QUOTIENT"),
            }
        }
    }

    fn arc(place: u32, weight: u64) -> Arc {
        Arc {
            place: PlaceIdx(place),
            weight,
        }
    }

    fn place(id: &str) -> PlaceInfo {
        PlaceInfo {
            id: id.to_string(),
            name: None,
        }
    }

    fn trans(id: &str, inputs: Vec<Arc>, outputs: Vec<Arc>) -> TransitionInfo {
        TransitionInfo {
            id: id.to_string(),
            name: None,
            inputs,
            outputs,
        }
    }

    #[test]
    fn discovers_places_only_when_swap_is_transition_automorphism() {
        let net = PetriNet {
            name: Some("safe-symmetric".into()),
            places: vec![place("p0"), place("p1"), place("sink")],
            transitions: vec![
                trans("t0", vec![arc(0, 1)], vec![arc(2, 1)]),
                trans("t1", vec![arc(1, 1)], vec![arc(2, 1)]),
            ],
            initial_marking: vec![1, 1, 0],
        };

        assert_eq!(discover_place_symmetry(&net), vec![vec![0, 1]]);
    }

    #[test]
    fn rejects_equal_degree_places_that_are_not_automorphic() {
        let net = PetriNet {
            name: Some("degree-profile-false-positive".into()),
            places: vec![place("p0"), place("p1"), place("left"), place("right")],
            transitions: vec![
                trans("t0", vec![arc(0, 1)], vec![arc(2, 1)]),
                trans("t1", vec![arc(1, 1)], vec![arc(3, 1)]),
            ],
            initial_marking: vec![1, 1, 0, 1],
        };

        assert!(discover_place_symmetry(&net).is_empty());
    }

    #[test]
    fn canonicalizer_sorts_verified_orbits() {
        let net = PetriNet {
            name: Some("safe-symmetric".into()),
            places: vec![place("p0"), place("p1"), place("sink")],
            transitions: vec![
                trans("t0", vec![arc(0, 1)], vec![arc(2, 1)]),
                trans("t1", vec![arc(1, 1)], vec![arc(2, 1)]),
            ],
            initial_marking: vec![1, 1, 0],
        };
        let canonicalizer = PetriCanonicalizer::build(&net);
        let mut marking = vec![2, 0, 0];

        canonicalizer.canonicalize(&mut marking);

        assert_eq!(marking, vec![0, 2, 0]);
    }

    /// A directed RING of `n` places/transitions (`p_i → t_i → p_{(i+1)%n}`)
    /// with equal token counts has a CYCLIC `Z_n` rotation automorphism group,
    /// which is COUPLED (rotating all places together) — NOT a full symmetric
    /// group. The per-orbit-sort multinomial would over-count it, so it must
    /// take the BSGS `GroupOrbit` path. Verifies the coupled path is selected,
    /// the orbit sizes are the cyclic ones, and the canonical form + weights
    /// partition the marking cube (`Σ_reps |orbit(rep)| = |markings|`).
    fn ring_net(n: u32) -> PetriNet {
        let places = (0..n).map(|i| place(&format!("p{i}"))).collect();
        let transitions = (0..n)
            .map(|i| trans(&format!("t{i}"), vec![arc(i, 1)], vec![arc((i + 1) % n, 1)]))
            .collect();
        PetriNet {
            name: Some(format!("ring-{n}")),
            places,
            transitions,
            initial_marking: vec![1; n as usize],
        }
    }

    #[test]
    fn coupled_ring_takes_bsgs_group_orbit_path() {
        // The coupled (GroupOrbit) quotient is gated off by default (perf); this
        // test exercises the gated path directly. The guard restores the prior
        // env value on drop so it cannot leak into other tests.
        let _coupled = CoupledQuotientEnvGuard::enabled();
        let net = ring_net(5);
        let c = PetriCanonicalizer::build(&net);
        // The discovered place symmetry is the cyclic Z5 (coupled, not full
        // symmetric), so the BSGS coupled path must be selected.
        assert!(!c.is_empty(), "ring has place symmetry");
        assert!(
            !c.orbits_are_full_symmetric(),
            "cyclic ring symmetry is NOT a full symmetric group",
        );
        assert!(
            c.quotient_is_available(),
            "bounded cyclic group must be admitted for the exact quotient",
        );
        assert!(
            c.uses_coupled_quotient(),
            "ring must use the BSGS coupled orbit-stabilizer path",
        );
        assert_eq!(c.group_order(), Some(5), "Z5 has order 5");
    }

    #[test]
    fn coupled_ring_orbit_quotient_partitions_marking_cube() {
        use std::collections::HashSet;
        // The coupled (GroupOrbit) quotient is gated off by default (perf); this
        // test exercises the gated path directly. The guard restores the prior
        // env value on drop so it cannot leak into other tests.
        let _coupled = CoupledQuotientEnvGuard::enabled();
        let net = ring_net(5);
        let c = PetriCanonicalizer::build(&net);
        assert!(c.uses_coupled_quotient());

        // Enumerate all {0,1}^5 markings; each canonicalizes to its Z5-orbit
        // minimal image, and the orbit-size weights over distinct reps must sum
        // to the total number of markings (the partition identity that makes
        // |R| exact). Also assert orbit-consistency: every rotation of a marking
        // canonicalizes identically.
        let mut reps: HashSet<Vec<u64>> = HashSet::new();
        let mut total = 0u64;
        let mut weight_sum = 0u64;
        for bits in 0u32..32 {
            let marking: Vec<u64> = (0..5).map(|i| ((bits >> i) & 1) as u64).collect();
            total += 1;
            let mut canon = marking.clone();
            c.canonicalize(&mut canon);
            // Orbit-consistency under the 5 rotations.
            for r in 0..5usize {
                let rotated: Vec<u64> = (0..5).map(|i| marking[(i + r) % 5]).collect();
                let mut rc = rotated.clone();
                c.canonicalize(&mut rc);
                assert_eq!(rc, canon, "all rotations share one canonical rep");
            }
            if reps.insert(canon.clone()) {
                weight_sum += c.orbit_size(&canon).expect("orbit size fits");
            }
        }
        assert_eq!(
            weight_sum, total,
            "Σ_reps |orbit(rep)| must equal the number of markings (|R| exact)",
        );
    }

    /// A full-Sₙ STAR: `n` source places, each feeding its own transition into a
    /// shared sink (`p_i → t_i → sink`), every source initially marked 1. The
    /// place-symmetry group is the FULL symmetric group Sₙ on the sources, so
    /// `discover_place_symmetry` should union all `n` sources into ONE orbit of
    /// size `n` — IF the automorphism search runs long enough to find generators
    /// connecting every source. (The sink is fixed; it is the only place with a
    /// distinct degree profile.)
    fn star_net(n: u32) -> PetriNet {
        let mut places: Vec<PlaceInfo> = (0..n).map(|i| place(&format!("p{i}"))).collect();
        places.push(place("sink"));
        let sink = n;
        let transitions = (0..n)
            .map(|i| trans(&format!("t{i}"), vec![arc(i, 1)], vec![arc(sink, 1)]))
            .collect();
        let mut initial_marking = vec![1u64; n as usize];
        initial_marking.push(0);
        PetriNet {
            name: Some(format!("star-{n}")),
            places,
            transitions,
            initial_marking,
        }
    }

    /// The single source orbit discovered for `net` (the union-find component of
    /// size ≥ 2 covering the source places), or empty if the search truncated to
    /// a trivial/split orbit. Used to compare default vs thorough discovery.
    fn largest_place_orbit(net: &PetriNet, budget: SymmetryBudget) -> Vec<u32> {
        let generators = find_automorphism_generators_with_budget(net, budget);
        let num_p = net.num_places();
        let num_t = net.transitions.len();
        let orbits = get_orbits(num_p + num_t, &generators, &[]);
        let mut groups: BTreeMap<usize, Vec<u32>> = BTreeMap::new();
        for p in 0..num_p {
            groups.entry(orbits[p]).or_default().push(p as u32);
        }
        groups
            .into_values()
            .max_by_key(std::vec::Vec::len)
            .unwrap_or_default()
    }

    /// THE WIN: a LARGE full-symmetric star where the historical default budget
    /// (64 generators / 50 ms) UNDER-approximates the group — discovering only a
    /// truncated orbit — but the widened `thorough` budget discovers the FULL Sₙ
    /// orbit (all `n` sources unioned into one component). With the full orbit
    /// the StateSpace orbit-quotient collapses the marking cube to a SINGLE
    /// canonical rep, so the BFS completes instead of timing out → unlocks the
    /// StateSpace CANNOT_COMPUTE cell.
    #[test]
    fn large_star_thorough_budget_recovers_full_symmetric_orbit() {
        // n chosen well above the 64-generator default cap so the default budget
        // provably truncates the orbit before all sources are connected, while
        // the thorough budget (256 generators) finds the full Sₙ.
        let n: u32 = 96;
        let net = star_net(n);

        let default_orbit = largest_place_orbit(&net, SymmetryBudget::default_budget());
        let thorough_orbit = largest_place_orbit(&net, SymmetryBudget::thorough());

        // The thorough budget must recover the FULL Sₙ source orbit: all n
        // sources unioned into one component.
        assert_eq!(
            thorough_orbit.len(),
            n as usize,
            "thorough budget must discover the full S_{n} source orbit (all sources)",
        );
        // And the default budget must have UNDER-approximated it (this is the
        // regression the widening fixes; if this ever stops holding the win is
        // already free and the test simply documents that).
        assert!(
            default_orbit.len() < n as usize,
            "default 64-generator budget is expected to truncate the S_{n} orbit \
             (got {} of {n}); if it no longer truncates, the widening is moot",
            default_orbit.len(),
        );

        // The full orbit collapses the symmetric marking cube to ONE canonical
        // rep whose orbit-size weight recovers the EXACT count: for the star with
        // every source marked 1 and the sink absorbing, the reachable markings on
        // the sources are exactly the multisets of n bits with a fixed total, and
        // the canonical (per-orbit-sorted) rep weighted by the multinomial orbit
        // size reproduces the concrete count. We verify the soundness gate + that
        // the thorough canonicalizer is a full-symmetric (multinomial) quotient,
        // which is what makes BFS over reps (instead of the cube) complete.
        let canon = PetriCanonicalizer::build_with_budget(&net, SymmetryBudget::thorough());
        assert!(!canon.is_empty(), "star has place symmetry");
        assert!(
            canon.orbits_are_full_symmetric(),
            "full star symmetry is a FULL symmetric group (multinomial quotient)",
        );
        assert_eq!(
            canon.place_orbits().len(),
            1,
            "all sources collapse into a single S_{n} orbit",
        );
        assert_eq!(canon.place_orbits()[0].len(), n as usize);

        // Orbit-size weight is the multinomial: the all-ones source marking has
        // every source at 1, so |orbit| = n!/n! = 1 here; shift one token to the
        // sink and the n configurations (which source is empty) collapse to one
        // rep weighted n. Verify the n single-empty-source markings all
        // canonicalize identically and weight to n (the exact concrete count).
        let mut single_empty: Vec<u64> = vec![1u64; n as usize];
        single_empty.push(0); // sink
        single_empty[0] = 0;
        single_empty[n as usize] = 1; // one token moved to sink
        let mut canon_marking = single_empty.clone();
        canon.canonicalize(&mut canon_marking);
        for empty in 0..n as usize {
            let mut m = vec![1u64; n as usize];
            m.push(1); // sink holds the moved token
            m[empty] = 0;
            let mut c = m.clone();
            canon.canonicalize(&mut c);
            assert_eq!(
                c, canon_marking,
                "every which-source-is-empty marking shares one canonical rep",
            );
        }
        assert_eq!(
            canon.orbit_size(&canon_marking).expect("orbit size fits"),
            n as u64,
            "the single-empty-source orbit has exactly n concrete markings",
        );
    }

    /// A net with NO non-trivial place symmetry must return QUICKLY under the
    /// widened thorough budget — the IR search early-exits (a discrete partition
    /// after refinement, no automorphisms to find), so widening the cap does NOT
    /// cause a blowup. Guards against the thorough budget regressing asymmetric
    /// nets (the main differential risk).
    #[test]
    fn no_symmetry_net_thorough_budget_exits_fast() {
        // A directed chain p0 → t0 → p1 → t1 → p2 ... with DISTINCT initial
        // markings so no two places are interchangeable. Refinement immediately
        // splits every place into its own cell ⇒ no generators ⇒ instant return.
        let n = 60usize;
        let places: Vec<PlaceInfo> = (0..n).map(|i| place(&format!("p{i}"))).collect();
        let transitions = (0..n - 1)
            .map(|i| {
                trans(
                    &format!("t{i}"),
                    vec![arc(i as u32, 1)],
                    vec![arc((i + 1) as u32, 1)],
                )
            })
            .collect();
        // Distinct token counts ⇒ initial-marking refinement isolates each place.
        let initial_marking: Vec<u64> = (0..n as u64).collect();
        let net = PetriNet {
            name: Some("asymmetric-chain".into()),
            places,
            transitions,
            initial_marking,
        };

        let start = std::time::Instant::now();
        let generators = find_automorphism_generators_with_budget(&net, SymmetryBudget::thorough());
        let elapsed = start.elapsed();

        assert!(
            generators.is_empty(),
            "asymmetric chain has no non-trivial automorphisms",
        );
        // Must early-exit FAR under the thorough 500 ms deadline (the search
        // finds a discrete partition with no backtracking). Generous bound to
        // avoid CI flakiness, but still proves no blowup.
        assert!(
            elapsed < std::time::Duration::from_millis(200),
            "asymmetric net must exit fast under the widened budget (took {elapsed:?})",
        );

        let canon = PetriCanonicalizer::build_with_budget(&net, SymmetryBudget::thorough());
        assert!(canon.is_empty(), "no symmetry ⇒ empty canonicalizer");
    }

    /// The existing symmetry differential invariant — a small full-symmetric net
    /// produces the SAME orbit decomposition under default and thorough budgets
    /// (the thorough budget never CHANGES a count, only widens what is found on
    /// large nets). Confirms widening does not perturb the small nets the broad
    /// differential covers.
    #[test]
    fn thorough_budget_matches_default_on_small_symmetric_net() {
        let net = star_net(4);
        let default_c =
            PetriCanonicalizer::build_with_budget(&net, SymmetryBudget::default_budget());
        let thorough_c = PetriCanonicalizer::build_with_budget(&net, SymmetryBudget::thorough());

        assert_eq!(
            default_c.place_orbits(),
            thorough_c.place_orbits(),
            "small symmetric net must yield identical orbits under either budget",
        );
        assert_eq!(
            default_c.orbits_are_full_symmetric(),
            thorough_c.orbits_are_full_symmetric(),
        );
        // Both fully discover S_4 (well under the 64-generator default cap).
        assert_eq!(default_c.place_orbits().len(), 1);
        assert_eq!(default_c.place_orbits()[0].len(), 4);
    }

    /// The `TY_SYMMETRY_THOROUGH=0` kill-switch must collapse the thorough budget
    /// back to the historical default (64 / 50 ms), so the wider discovery can be
    /// disabled at runtime without a rebuild.
    #[test]
    fn thorough_kill_switch_restores_default_budget() {
        let _lock = crate::env_test_lock();
        let prev = std::env::var_os("TY_SYMMETRY_THOROUGH");

        crate::env_guard::set_var("TY_SYMMETRY_THOROUGH", "0");
        let b = SymmetryBudget::thorough();
        assert_eq!(
            b.max_generators,
            SymmetryBudget::default_budget().max_generators
        );
        assert_eq!(b.time_budget, SymmetryBudget::default_budget().time_budget);

        // Bare integer overrides the generator cap.
        crate::env_guard::set_var("TY_SYMMETRY_THOROUGH", "512");
        let b = SymmetryBudget::thorough();
        assert_eq!(b.max_generators, 512);

        // "gens:ms" overrides both.
        crate::env_guard::set_var("TY_SYMMETRY_THOROUGH", "128:250");
        let b = SymmetryBudget::thorough();
        assert_eq!(b.max_generators, 128);
        assert_eq!(b.time_budget, std::time::Duration::from_millis(250));

        match prev {
            Some(v) => crate::env_guard::set_var("TY_SYMMETRY_THOROUGH", v),
            None => crate::env_guard::remove_var("TY_SYMMETRY_THOROUGH"),
        }
    }
}
