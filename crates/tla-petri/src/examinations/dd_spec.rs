// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Shared, **sound** Decision-Diagram spec builder.
//!
//! Both the StateSpace verdict path
//! (`examination_non_property::state_space`) and the UpperBounds fast-path
//! (`examinations::upper_bounds::dd_fastpath`) need to turn a [`PetriNet`]
//! into a `tla_dd::DdNetSpec` whose **unary per-place encoding cannot
//! silently lose a reachable marking**. The encoding drops any firing
//! whose successor would exceed a place's encoded value range, so an
//! *under*-estimated per-place bound yields a wrong (too-small) reachable
//! set. To stay sound we must bound each place by a value that is
//! provably `>=` its true maximum over all reachable markings.
//!
//! This module is the single source of truth for that soundness gate, so
//! the two call sites cannot drift apart. The gate is **reject-not-clamp**:
//! any condition we cannot prove sound returns `None`, and the caller
//! falls back to explicit BFS (always sound).
//!
//! # Soundness argument
//!
//! For each place `p` we take `lp_upper_bound(net, &[p])`, the LP
//! relaxation of the state equation `M = M0 + C·x, x >= 0`. The integer
//! reachable set is contained in that LP polytope, so the LP optimum
//! (rounded up via `ceil`) is `>=` the true reachable max tokens in `p`.
//! Encoding every place with its LP bound therefore makes the encoded
//! value range a *superset* of the reachable projection on every place,
//! so no reachable marking is dropped and the BDD reachable set is
//! **exact**. Every metric read off an exact reachable set (state count,
//! edge count, per-place max, token-sum max, per-query weighted max) is
//! itself exact.
//!
//! # Gates
//!
//! 1. `1 <= num_places <= MAX_PLACES` (the MVP encoding cap).
//! 2. Every place's LP bound is finite and `<= MAX_PER_PLACE_BOUND`
//!    (unbounded or oversized ⇒ decline; the encoding cannot represent
//!    it). Places with bound `<= 16` use the unary one-hot encoding;
//!    places with `16 < bound <= MAX_PER_PLACE_BOUND` use the binary
//!    (log-encoded) encoding inside `tla-dd` — both are exact (see the
//!    `tla_dd::tests::test_binary_encoding_*` differentials).
//! 3. The initial marking fits every per-place bound (implied by LP
//!    soundness, re-checked defensively).
//! 4. No transition's total output weight on a place exceeds that
//!    place's bound (a `post[p] > bound[p]` firing from an enabled
//!    marking would push the successor out of the encoded range and be
//!    silently dropped).

use crate::nupn::NupnStructure;
use crate::petri_net::{PetriNet, PlaceIdx};
use crate::resolved_predicate::{ResolvedIntExpr, ResolvedPredicate};

/// Cap on per-place token count admitted into the production DD lane.
///
/// **Raised to the binary (log-encoded) range
/// (`tla_dd::MAX_BINARY_PLACE_BOUND` = 2^20).** A place with `bound <= 16`
/// is still encoded with the byte-for-byte-unchanged unary one-hot
/// encoding; a place with `16 < bound <= 2^20` is encoded with the binary
/// (log) field inside `tla-dd` (`ceil(log2(bound+1))` BDD variables holding
/// the value's bits). Both encodings are exact on the nets they answer (see
/// the `tla_dd::tests::test_binary_encoding_*` and
/// `tla_dd::crosscheck` differentials against the exhaustive oracle), so
/// admitting the binary band only changes *which* engine decides a net, not
/// *what* it decides.
///
/// The earlier OxiDD `apply` non-termination that pinned this to the unary
/// range was fixed by running every DD computation on an isolated worker
/// thread with a 512 MiB stack and a wall-clock deadline
/// (`run_isolated`/`DD_WORKER_STACK_BYTES`): deep recursion now completes
/// or is cut off by the deadline into a clean **decline**, never a
/// `SIGABRT`. The high-bound (>16) binary fixtures up to bound 256+ now
/// compute correctly.
///
/// Soundness floor is unchanged: a net whose binary BDD node-blows-up or
/// does not converge inside its budget still DECLINES (`None` here, or a
/// `CANNOT_COMPUTE`/timeout from the engine) and falls back to the
/// always-sound explicit/structural engine — never a wrong answer, never a
/// crash. The remaining hard limits (`MAX_PLACES`, the engine's
/// `MAX_TOTAL_BDD_VARS` memory guard, and the per-net wall-clock deadline)
/// continue to bound resource use.
pub(crate) const MAX_PER_PLACE_BOUND: u64 = tla_dd::MAX_BINARY_PLACE_BOUND;

/// Cap on number of places, matching the engine's `tla_dd::MAX_PLACES` (1024).
/// This is a defensive guard (bounding the up-front BDD-variable allocation and
/// the per-place LP precompute below), NOT a soundness boundary — the real
/// engine ceiling is `tla_dd::MAX_TOTAL_BDD_VARS` (2^16), and every DD consumer
/// runs the heavy reachable-set computation under a wall-clock budget, falling
/// back to explicit search on timeout/OOM/non-convergence. Raised 256→1024 (it
/// was stricter than the engine cap it claimed to mirror) so more conserved
/// nets in the 256<np≤1024 band reach the symbolic lane — pairing with the
/// P-invariant DD-admission + ordering increments.
pub(crate) const MAX_PLACES: usize = 1024;

/// Build a sound `tla_dd::DdNetSpec` for `net`, or `None` if any
/// soundness gate fails. On success also returns the per-place bound
/// vector (callers validate DD results against it).
///
/// Soundness: the returned spec encodes a value range that is a superset
/// of every place's reachable projection, so the DD reachable set built
/// from it is exact. See the module docs for the full argument.
#[must_use]
/// Choose the per-place DD value bound from the two sound upper-bound sources:
/// the LP state-equation relaxation and the structural P-invariant bound. Both
/// are proven upper bounds on the place's reachable token count, so either keeps
/// the encoded range a SUPERSET of reachable (the DD soundness invariant).
///
/// Zero-regression by construction: an in-range LP bound is used as-is (so every
/// net the LP path already admitted encodes identically, and Gate 4 is
/// unaffected). The structural bound is a pure FALLBACK used only when the LP
/// declined (its size gate `np + nt > MAX_LP_VARIABLES`, or unbounded/oversized
/// reported as `None`) or returned an out-of-range value — admitting large
/// conserved nets that were previously rejected, gating the whole symbolic
/// pillar off. Returns `None` (decline, fail-closed) when neither source yields
/// an in-range bound.
fn sound_place_bound(lp: Option<u64>, structural: Option<u64>) -> Option<u64> {
    match lp {
        Some(l) if l <= MAX_PER_PLACE_BOUND => Some(l),
        _ => match structural {
            Some(s) if s <= MAX_PER_PLACE_BOUND => Some(s),
            _ => None,
        },
    }
}

pub(crate) fn build_sound_dd_spec(net: &PetriNet) -> Option<(tla_dd::DdNetSpec, Vec<u64>)> {
    // Gate 1: place-count cap.
    let num_places = net.num_places();
    if num_places == 0 || num_places > MAX_PLACES {
        return None;
    }

    // Gate 2: per-place sound upper bound. Each place needs a proven upper bound
    // on its reachable token count so the DD encodes a value range that is a
    // SUPERSET of the reachable projection (keeping the symbolic reachable set
    // exact). Two independent sound sources:
    //   - the LP state-equation relaxation (`lp_upper_bound`), and
    //   - the structural P-invariant bound (`structural_place_bound`,
    //     `min_y ⌊token_count / y_p⌋` over covering invariants — exact by the
    //     P-invariant theorem yᵀm = yᵀm₀).
    // The LP solver DECLINES (`None`) on large nets (`np + nt > MAX_LP_VARIABLES`)
    // and reports oversized/unbounded as `None` / `> 1e15`, which previously made
    // `build_sound_dd_spec` reject the whole net — gating OFF every symbolic lane
    // (StateSpace / UpperBounds / Reachability / Deadlock / Liveness / CTL / LTL)
    // that consumes this single admission gate. A P-invariant computation is
    // cheap (no LP) and bounds conserved places regardless of net size, so we
    // ADMIT such nets using the structural bound.
    //
    // Zero-regression by construction: when the LP bound is in range we use it
    // unchanged (identical encoding + Gate 4 behaviour for every net the old
    // path already admitted). We fall back to the structural bound ONLY when the
    // LP declined or its bound is oversized — a pure admission gain. We do not
    // tighten an in-range LP bound with a smaller structural one, which could
    // otherwise make Gate 4 needlessly reject a net carrying a structurally-dead
    // high-weight transition.
    let invariants = crate::invariant::compute_p_invariants(net);
    let mut per_place_bounds: Vec<u64> = Vec::with_capacity(num_places);
    for p in 0..num_places {
        let lp = crate::lp_state_equation::lp_upper_bound(net, &[PlaceIdx(p as u32)]);
        let structural = crate::invariant::structural_place_bound(&invariants, p);
        let bound = sound_place_bound(lp, structural)?;
        per_place_bounds.push(bound);
    }

    // Gate 3: initial marking fits the encoded range (defensive; implied
    // by LP soundness since the initial marking is reachable).
    if net.initial_marking.len() != num_places {
        return None;
    }
    for (p, &v) in net.initial_marking.iter().enumerate() {
        if v > per_place_bounds[p] {
            return None;
        }
    }

    // Build per-place pre/post vectors, applying Gate 4.
    let mut transitions: Vec<tla_dd::DdTransition> = Vec::with_capacity(net.transitions.len());
    for t in &net.transitions {
        let mut pre = vec![0u64; num_places];
        let mut post = vec![0u64; num_places];
        for arc in &t.inputs {
            let idx = arc.place.0 as usize;
            if idx >= num_places {
                return None;
            }
            pre[idx] = pre[idx].saturating_add(arc.weight);
        }
        for arc in &t.outputs {
            let idx = arc.place.0 as usize;
            if idx >= num_places {
                return None;
            }
            post[idx] = post[idx].saturating_add(arc.weight);
        }
        // Gate 4: a transition that could push a place past its bound
        // would be silently dropped by the encoding ⇒ decline.
        for p in 0..num_places {
            if post[p] > per_place_bounds[p] {
                return None;
            }
        }
        transitions.push(tla_dd::DdTransition { pre, post });
    }

    let spec = tla_dd::DdNetSpec {
        bounds: per_place_bounds.clone(),
        initial_marking: net.initial_marking.clone(),
        transitions,
    };
    Some((spec, per_place_bounds))
}

/// Translate a [`ResolvedIntExpr`] into the DD crate's expression type,
/// validating every place index against the net. Returns `None` if any index
/// is out of range (fail-closed; resolved predicates index the same net, so
/// this should not normally happen).
///
/// This is the single source of truth for resolved-predicate → DD-predicate
/// conversion shared by the reachability DD fast-path
/// (`examinations::reachability::dd_fastpath`) and the symbolic CTL lane
/// (`examinations::ctl::pipeline`), so the two cannot drift apart.
#[must_use]
pub(crate) fn translate_int_expr(
    expr: &ResolvedIntExpr,
    num_places: usize,
) -> Option<tla_dd::DdIntExpr> {
    match expr {
        ResolvedIntExpr::Constant(v) => Some(tla_dd::DdIntExpr::Constant(*v)),
        ResolvedIntExpr::TokensCount(places) => {
            let mut idxs = Vec::with_capacity(places.len());
            for p in places {
                let idx = p.0 as usize;
                if idx >= num_places {
                    return None;
                }
                idxs.push(idx);
            }
            Some(tla_dd::DdIntExpr::TokensCount(idxs))
        }
    }
}

/// Translate a [`ResolvedPredicate`] into the DD crate's predicate type.
/// Returns `None` (fail-closed) if any place/transition index is out of range
/// or the predicate uses a shape the DD encoding does not support.
///
/// Shared single source of truth — see [`translate_int_expr`].
#[must_use]
pub(crate) fn translate_predicate(
    pred: &ResolvedPredicate,
    num_places: usize,
    num_transitions: usize,
) -> Option<tla_dd::DdPredicate> {
    match pred {
        ResolvedPredicate::True => Some(tla_dd::DdPredicate::True),
        ResolvedPredicate::False => Some(tla_dd::DdPredicate::False),
        ResolvedPredicate::Not(inner) => Some(tla_dd::DdPredicate::Not(Box::new(
            translate_predicate(inner, num_places, num_transitions)?,
        ))),
        ResolvedPredicate::And(children) => {
            let mut out = Vec::with_capacity(children.len());
            for c in children {
                out.push(translate_predicate(c, num_places, num_transitions)?);
            }
            Some(tla_dd::DdPredicate::And(out))
        }
        ResolvedPredicate::Or(children) => {
            let mut out = Vec::with_capacity(children.len());
            for c in children {
                out.push(translate_predicate(c, num_places, num_transitions)?);
            }
            Some(tla_dd::DdPredicate::Or(out))
        }
        ResolvedPredicate::IntLe(left, right) => Some(tla_dd::DdPredicate::IntLe(
            translate_int_expr(left, num_places)?,
            translate_int_expr(right, num_places)?,
        )),
        ResolvedPredicate::IsFireable(transitions) => {
            let mut idxs = Vec::with_capacity(transitions.len());
            for t in transitions {
                let idx = t.0 as usize;
                if idx >= num_transitions {
                    return None;
                }
                idxs.push(idx);
            }
            Some(tla_dd::DdPredicate::IsFireable(idxs))
        }
    }
}

/// Kill-switch for the NUPN-seeded DD variable order: set
/// `TY_DD_NUPN_ORDER=0` (or `off`/`false`) to disable seeding and fall back
/// to today's unseeded FORCE ordering everywhere. PERFORMANCE-ONLY either
/// way — the seed can never change a represented set/count/bound (any
/// permutation is answer-preserving; see `tla_dd::order`), so the switch
/// exists for triage/benchmarking, not soundness.
fn nupn_order_seed_disabled() -> bool {
    std::env::var("TY_DD_NUPN_ORDER")
        .is_ok_and(|v| v == "0" || v.eq_ignore_ascii_case("off") || v.eq_ignore_ascii_case("false"))
}

/// Derive a DD variable-order *seed* from the NUPN unit hierarchy: a
/// pre-order DFS from the root unit emitting each unit's local places
/// contiguously, then its subunits recursively (pnmc's hierarchical-order
/// recursion, `make_order.cc:30-74`). Units unreachable from the root are
/// appended in declaration order; places not covered by any unit are
/// appended in ascending index order.
///
/// The NUPN units partition the covered places into nested groups of
/// mutually exclusive places — exactly the locality structure a good DD
/// order wants — so the block order is offered to
/// `tla_dd::force_place_order_seeded` both as-is and as a FORCE seed, where
/// it competes under the span guard (it is used only when it strictly
/// improves on today's choice).
///
/// Returns `None` when the seed cannot help: NUPN disabled via
/// [`nupn_order_seed_disabled`], no units, the result is not a permutation
/// of `0..num_places` (defensive — the parser already rejects duplicate
/// ownership), or the seed equals the identity (pointless work).
///
/// PERFORMANCE-ONLY: the returned order never affects any reported value
/// (see the soundness note on `tla_dd::force_place_order_seeded`).
#[must_use]
pub(crate) fn nupn_order_seed(nupn: &NupnStructure, num_places: usize) -> Option<Vec<usize>> {
    if nupn_order_seed_disabled() {
        return None;
    }
    let units = nupn.units();
    if units.is_empty() || num_places == 0 {
        return None;
    }

    let mut order: Vec<usize> = Vec::with_capacity(num_places);
    let mut covered = vec![false; num_places];
    let mut visited = vec![false; units.len()];

    // Pre-order DFS emitting a unit's local places, then its subunits.
    // Iterative (explicit stack) so a deep hierarchy cannot overflow; the
    // `visited` set makes malformed (shared/cyclic) subunit references
    // harmless — each unit is emitted at most once.
    let mut emit_from = |start: usize, order: &mut Vec<usize>, covered: &mut Vec<bool>| {
        let mut stack = vec![start];
        while let Some(u) = stack.pop() {
            if visited[u] {
                continue;
            }
            visited[u] = true;
            for place in units[u].places() {
                let idx = place.0 as usize;
                if idx >= num_places || covered[idx] {
                    // Out-of-range or duplicated ownership: not a valid
                    // permutation source — abort the whole seed.
                    return false;
                }
                covered[idx] = true;
                order.push(idx);
            }
            // Reverse push so subunits are visited in declaration order.
            for &sub in units[u].subunits().iter().rev() {
                if sub < units.len() && !visited[sub] {
                    stack.push(sub);
                }
            }
        }
        true
    };

    // Root first (the hierarchy proper), then any units unreachable from
    // the root in declaration order.
    let root = nupn
        .root_unit()
        .and_then(|r| units.iter().position(|u| std::ptr::eq(u, r)));
    if let Some(root) = root {
        if !emit_from(root, &mut order, &mut covered) {
            return None;
        }
    }
    for u in 0..units.len() {
        if !emit_from(u, &mut order, &mut covered) {
            return None;
        }
    }

    // Uncovered places keep their relative (ascending) PNML order.
    for (idx, &is_covered) in covered.iter().enumerate() {
        if !is_covered {
            order.push(idx);
        }
    }

    // Defensive permutation check + skip the no-op seed.
    if order.len() != num_places {
        return None;
    }
    if order.iter().enumerate().all(|(rank, &p)| rank == p) {
        return None; // identity — already a candidate, skip pointless work
    }
    Some(order)
}

/// `TY_DD_PINV_ORDER=0` (or `off`/`false`) disables the P-invariant block seed.
fn pinv_order_seed_disabled() -> bool {
    std::env::var("TY_DD_PINV_ORDER").is_ok_and(|v| {
        let v = v.trim().to_ascii_lowercase();
        v == "0" || v == "off" || v == "false"
    })
}

/// Structural variable-order seed for conserved nets that carry **no NUPN**
/// annotation (the case [`nupn_order_seed`] cannot serve). Emit places block by
/// block, one block per P-invariant support, so the FORCE heuristic starts from
/// an order that keeps each conserved place cluster contiguous in the BDD — which
/// narrows the per-level saturation relations and converts admitted-but-timing-
/// out large conserved nets into solved cells. Blocks are ordered by ascending
/// minimum support index; uncovered places keep ascending PNML order.
///
/// SOUNDNESS-NEUTRAL: this only produces a *candidate* place permutation. A
/// variable order is applied as a pure relabeling before the BDD is built, so
/// every reachable marking and every metric is invariant under it
/// (`order.rs` module note). `force_place_order_seeded` adopts the seed **only
/// when it strictly reduces transition span** vs the identity incumbent (its
/// span guard), so it can never worsen an order — only change which cells
/// converge. A `None` seed is bit-identical to today's behaviour. The P-invariant
/// basis is already computed inside [`build_sound_dd_spec`], so this is cheap.
pub(crate) fn p_invariant_order_seed(net: &PetriNet, num_places: usize) -> Option<Vec<usize>> {
    if pinv_order_seed_disabled() || num_places == 0 {
        return None;
    }
    let invariants = crate::invariant::compute_p_invariants(net);
    if invariants.is_empty() {
        return None;
    }
    // Each invariant's in-range support place indices, ascending + deduped.
    let mut blocks: Vec<Vec<usize>> = invariants
        .iter()
        .map(|inv| {
            let mut places: Vec<usize> = inv
                .support()
                .map(|(p, _w)| p)
                .filter(|&p| p < num_places)
                .collect();
            places.sort_unstable();
            places.dedup();
            places
        })
        .filter(|b| !b.is_empty())
        .collect();
    // Deterministic block order: ascending minimum support index.
    blocks.sort_by_key(|b| b[0]);

    let mut order: Vec<usize> = Vec::with_capacity(num_places);
    let mut covered = vec![false; num_places];
    for block in &blocks {
        for &p in block {
            if !covered[p] {
                covered[p] = true;
                order.push(p);
            }
        }
    }
    // Places no invariant covers keep their relative (ascending) PNML order.
    for (idx, &is_covered) in covered.iter().enumerate() {
        if !is_covered {
            order.push(idx);
        }
    }

    // Defensive permutation check + skip the no-op seed (same guards as
    // `nupn_order_seed`; by construction `order` is already a permutation).
    if order.len() != num_places {
        return None;
    }
    if order.iter().enumerate().all(|(rank, &p)| rank == p) {
        return None; // identity — already a candidate, skip pointless work
    }
    Some(order)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::petri_net::{Arc, PetriNet, PlaceInfo, TransitionInfo};

    /// Serializes tests that read/write the process-global `TY_DD_*_ORDER` env
    /// vars, so a kill-switch test setting one cannot race a parallel test that
    /// reads it (cargo runs tests multi-threaded). Each such test holds this for
    /// its whole body.
    static SEED_ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    fn seed_env_guard() -> std::sync::MutexGuard<'static, ()> {
        SEED_ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Measurement harness: load a real PNML model directory from
    /// `TY_DD_BENCH_DIR`, build the sound DD spec, and time the ordered DD
    /// StateSpace dispatch. Run with, e.g.:
    ///
    /// ```text
    /// TY_DD_BENCH_DIR=tmp_benchmark_models/CSRepetitions-PT-03 \
    ///   cargo test -p tla-petri --features dd-backend dd_bench -- --nocapture
    /// ```
    #[test]
    fn dd_bench_statespace_from_env() {
        let Ok(dir) = std::env::var("TY_DD_BENCH_DIR") else {
            eprintln!("TY_DD_BENCH_DIR not set; skipping");
            return;
        };
        let net =
            crate::parser::parse_pnml_dir(std::path::Path::new(&dir)).expect("parse PNML dir");
        eprintln!(
            "[dd-bench] {dir}: places={} transitions={}",
            net.num_places(),
            net.transitions.len()
        );
        let t0 = std::time::Instant::now();
        let built = build_sound_dd_spec(&net);
        eprintln!("[dd-bench] spec build/gate took {:?}", t0.elapsed());
        let Some((spec, bounds)) = built else {
            eprintln!("[dd-bench] spec DECLINED (gate failed) — DD inapplicable");
            return;
        };
        eprintln!(
            "[dd-bench] gate passed: max per-place bound = {}",
            bounds.iter().copied().max().unwrap_or(0)
        );
        eprintln!("[dd-bench] running native tla-bdd metrics");
        let t1 = std::time::Instant::now();
        let m = crate::examinations::mdd_common::state_space_metrics_via_bdd(&spec);
        eprintln!(
            "[dd-bench] native tla-bdd metrics in {:?}: states={} edges={} \
             max_in_place={} max_sum={} iters={}",
            t1.elapsed(),
            m.state_count,
            m.edge_count,
            m.max_token_in_place,
            m.max_token_sum,
            m.iterations,
        );
    }

    /// Manual probe: which order candidate does the span guard
    /// pick on a real PNML model, with and without the NUPN seed? Run with:
    ///
    /// ```text
    /// TY_DD_BENCH_DIR=tmp_benchmark_models/TokenRing-PT-010 \
    ///   cargo test -p tla-petri --release dd_bench_nupn_seed -- --nocapture
    /// ```
    #[test]
    fn dd_bench_nupn_seed_from_env() {
        let Ok(dir) = std::env::var("TY_DD_BENCH_DIR") else {
            eprintln!("TY_DD_BENCH_DIR not set; skipping");
            return;
        };
        let dir = std::path::Path::new(&dir);
        let net = crate::parser::parse_pnml_dir(dir).expect("parse PNML dir");
        let model = crate::model::load_model_dir(dir).expect("load model dir");
        let Some(nupn) = model.nupn() else {
            eprintln!("[nupn-seed] model has no NUPN annotation");
            return;
        };
        let Some((spec, _bounds)) = build_sound_dd_spec(&net) else {
            eprintln!("[nupn-seed] spec DECLINED (gate failed)");
            return;
        };
        let n = net.num_places();
        let span = |order: &[usize]| -> u64 {
            let inv = tla_dd::invert_order(order);
            spec.transitions
                .iter()
                .map(|t| {
                    let lvls: Vec<usize> = (0..n)
                        .filter(|&p| t.pre[p] != 0 || t.post[p] != 0)
                        .map(|p| inv[p])
                        .collect();
                    match (lvls.iter().min(), lvls.iter().max()) {
                        (Some(lo), Some(hi)) => (hi - lo) as u64,
                        _ => 0,
                    }
                })
                .sum()
        };
        let identity: Vec<usize> = (0..n).collect();
        let unseeded = tla_dd::force_place_order(&spec);
        let seed = nupn_order_seed(nupn, n);
        let seeded = tla_dd::force_place_order_seeded(&spec, seed.as_deref());
        eprintln!("[nupn-seed] places={n}");
        eprintln!("[nupn-seed] span(identity)        = {}", span(&identity));
        eprintln!("[nupn-seed] span(unseeded choice) = {}", span(&unseeded));
        if let Some(s) = &seed {
            eprintln!("[nupn-seed] span(nupn block)      = {}", span(s));
            eprintln!("[nupn-seed] span(seeded choice)   = {}", span(&seeded));
            eprintln!(
                "[nupn-seed] seeded choice == nupn block: {}; == unseeded: {}",
                &seeded == s,
                seeded == unseeded,
            );
        } else {
            eprintln!("[nupn-seed] no usable seed (identity/disabled/non-permutation)");
        }
    }

    /// Manual probe: the DIRECT MDD-size measurement — reachable-set
    /// interior node count (the quantity the 8M cap bounds) under identity vs
    /// FORCE vs NUPN-seeded place order, on a real PNML model. This is the
    /// conclusive evidence (beyond the transition-span proxy) that the variable
    /// ordering shrinks the actual MDD. Run with:
    ///
    /// ```text
    /// TY_DD_BENCH_DIR=~/.cache/ty/corpus/2025/Philosophers-PT-000010 \
    ///   cargo test -p tla-petri --features dd-backend mdd_node_count_from_env -- --nocapture
    /// ```
    #[cfg(feature = "dd-backend")]
    #[test]
    fn mdd_node_count_from_env() {
        let Ok(dir) = std::env::var("TY_DD_BENCH_DIR") else {
            eprintln!("TY_DD_BENCH_DIR not set; skipping");
            return;
        };
        let dir = std::path::Path::new(&dir);
        let net = crate::parser::parse_pnml_dir(dir).expect("parse PNML dir");
        let Some((spec, _bounds)) = build_sound_dd_spec(&net) else {
            eprintln!("[mdd-nodes] spec DECLINED (gate failed)");
            return;
        };
        let n = net.num_places();
        let seed = crate::model::load_model_dir(dir)
            .ok()
            .and_then(|m| m.nupn().and_then(|nupn| nupn_order_seed(nupn, n)));

        let to_mdd = |s: &tla_dd::DdNetSpec| tla_mdd::MddNet {
            bounds: s.bounds.clone(),
            initial_marking: s.initial_marking.clone(),
            transitions: s
                .transitions
                .iter()
                .map(|t| tla_mdd::MddTransition {
                    pre: t.pre.clone(),
                    post: t.post.clone(),
                })
                .collect(),
        };
        // A generous per-build deadline so a bad order does not stall the probe;
        // it declines (Err) rather than blow memory, which is itself the result.
        let deadline = || Some(std::time::Instant::now() + std::time::Duration::from_secs(90));
        let count = |order: &[usize]| -> String {
            let permuted = tla_dd::permute_spec(&spec, order);
            match to_mdd(&permuted).reachable_set_node_count(deadline()) {
                Ok(nodes) => format!("{nodes} nodes"),
                Err(e) => format!("DECLINED ({e:?})"),
            }
        };

        let identity: Vec<usize> = (0..n).collect();
        let forced = tla_dd::force_place_order(&spec);
        eprintln!("[mdd-nodes] places={n}");
        eprintln!("[mdd-nodes] identity order   : {}", count(&identity));
        eprintln!("[mdd-nodes] FORCE order      : {}", count(&forced));
        if seed.is_some() {
            let seeded = tla_dd::force_place_order_seeded(&spec, seed.as_deref());
            eprintln!("[mdd-nodes] NUPN-seeded order : {}", count(&seeded));
        } else {
            eprintln!("[mdd-nodes] (no NUPN seed available)");
        }
    }

    fn place(id: &str) -> PlaceInfo {
        PlaceInfo {
            id: id.into(),
            name: None,
        }
    }

    /// 2-place swap net: p0+p1 conserved at 1, per-place max 1.
    fn swap_net() -> PetriNet {
        PetriNet {
            name: Some("swap".into()),
            places: vec![place("p0"), place("p1")],
            transitions: vec![
                TransitionInfo {
                    id: "t01".into(),
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
                    id: "t10".into(),
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

    #[test]
    fn swap_net_admits_with_tight_bounds() {
        let net = swap_net();
        let (spec, bounds) = build_sound_dd_spec(&net).expect("swap net is DD-eligible");
        // LP bound on each place of a 1-token conserved net is 1.
        assert_eq!(bounds, vec![1, 1]);
        assert_eq!(spec.bounds, vec![1, 1]);
        assert_eq!(spec.initial_marking, vec![1, 0]);
        assert_eq!(spec.transitions.len(), 2);
    }

    /// Non-conservative net that the OLD crude gate rejected (a transition
    /// with post_sum > pre_sum), but which is genuinely bounded: t doubles
    /// p1 from p0 once, then is dead (p0 starts at 1, never refilled).
    /// LP must bound it finitely for the spec to build.
    #[test]
    fn non_conservative_but_bounded_net_admits() {
        // p0 --t--> p1 (x2 output). p0 init 1, p1 init 0.
        // Reachable: {p0=1,p1=0} and {p0=0,p1=2}. Per-place max: p0=1, p1=2.
        let net = PetriNet {
            name: Some("doubler".into()),
            places: vec![place("p0"), place("p1")],
            transitions: vec![TransitionInfo {
                id: "t".into(),
                name: None,
                inputs: vec![Arc {
                    place: PlaceIdx(0),
                    weight: 1,
                }],
                outputs: vec![Arc {
                    place: PlaceIdx(1),
                    weight: 2,
                }],
            }],
            initial_marking: vec![1, 0],
        };
        let (_, bounds) = build_sound_dd_spec(&net)
            .expect("non-conservative-but-bounded net is DD-eligible under LP bounds");
        // LP upper bound must be sound: >= true per-place max (p0>=1, p1>=2).
        assert!(
            bounds[0] >= 1,
            "p0 bound {} must cover true max 1",
            bounds[0]
        );
        assert!(
            bounds[1] >= 2,
            "p1 bound {} must cover true max 2",
            bounds[1]
        );
    }

    #[test]
    fn unbounded_place_declines() {
        // A source transition with no input perpetually adds to p0 ⇒
        // p0 is unbounded ⇒ LP is unbounded ⇒ decline.
        let net = PetriNet {
            name: Some("source".into()),
            places: vec![place("p0")],
            transitions: vec![TransitionInfo {
                id: "gen".into(),
                name: None,
                inputs: vec![],
                outputs: vec![Arc {
                    place: PlaceIdx(0),
                    weight: 1,
                }],
            }],
            initial_marking: vec![0],
        };
        assert!(
            build_sound_dd_spec(&net).is_none(),
            "unbounded place must make the spec decline (fail-closed)",
        );
    }

    #[test]
    fn oversized_net_declines() {
        let places: Vec<PlaceInfo> = (0..=MAX_PLACES).map(|i| place(&format!("p{i}"))).collect();
        let net = PetriNet {
            name: Some("oversized".into()),
            places,
            transitions: vec![],
            initial_marking: vec![0; MAX_PLACES + 1],
        };
        assert!(
            build_sound_dd_spec(&net).is_none(),
            "> MAX_PLACES net must decline",
        );
    }

    /// Binary band ADMISSION: a conserved 2-place shuttle of 17 tokens has a
    /// per-place LP bound of 17 (> 16, the unary cap), which the binary
    /// (log-encoded) field represents exactly. The production gate now admits
    /// it (cap raised to `tla_dd::MAX_BINARY_PLACE_BOUND`); the spec builds
    /// with the tight LP bounds.
    #[test]
    fn high_bound_conserved_net_admits_via_binary_band() {
        let net = PetriNet {
            name: Some("shuttle17".into()),
            places: vec![place("p0"), place("p1")],
            transitions: vec![
                TransitionInfo {
                    id: "t01".into(),
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
                    id: "t10".into(),
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
            initial_marking: vec![17, 0],
        };
        // LP bound is 17 <= MAX_PER_PLACE_BOUND (now 2^20), so the gate
        // admits — the conserved sum bounds each place at 17.
        let (spec, bounds) = build_sound_dd_spec(&net)
            .expect("17-token conserved shuttle is now DD-eligible (binary band)");
        assert!(
            bounds
                .iter()
                .all(|&b| (17..=MAX_PER_PLACE_BOUND).contains(&b)),
            "per-place bounds {bounds:?} must cover the conserved 17 and stay <= cap",
        );
        assert_eq!(spec.bounds, bounds);
        assert_eq!(spec.initial_marking, vec![17, 0]);
        assert_eq!(
            MAX_PER_PLACE_BOUND,
            tla_dd::MAX_BINARY_PLACE_BOUND,
            "production gate is the binary band",
        );
    }

    /// Fail-closed above the binary cap: a conserved shuttle of
    /// `MAX_BINARY_PLACE_BOUND + 1` tokens has an LP bound that the binary
    /// field cannot represent, so the gate declines (fall back to the
    /// always-sound explicit engine), never a wrong answer.
    #[test]
    fn above_binary_cap_net_declines() {
        let n = MAX_PER_PLACE_BOUND + 1; // 2^20 + 1
        let net = PetriNet {
            name: Some("shuttle-over-cap".into()),
            places: vec![place("p0"), place("p1")],
            transitions: vec![
                TransitionInfo {
                    id: "t01".into(),
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
                    id: "t10".into(),
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
            initial_marking: vec![n, 0],
        };
        assert!(
            build_sound_dd_spec(&net).is_none(),
            "net with bound {n} > binary cap must decline (fail-closed)",
        );
    }

    /// 6-place net whose PNML index order interleaves two independent
    /// 3-place token rings; the NUPN hierarchy groups each ring into a
    /// unit, so the block order de-interleaves them.
    fn nupn_seed_fixture() -> (PetriNet, crate::nupn::NupnStructure) {
        let net = PetriNet {
            name: Some("nupn-seed".into()),
            places: (0..6)
                .map(|i| PlaceInfo {
                    id: format!("P{i}"),
                    name: None,
                })
                .collect(),
            transitions: vec![],
            initial_marking: vec![1, 1, 0, 0, 0, 0],
        };
        // Ring A = {P0, P2, P4}, Ring B = {P1, P3, P5}; root u0 has no
        // local places and two subunits in order uA, uB.
        let pnml = r#"<?xml version="1.0"?>
<pnml xmlns="http://www.pnml.org/version-2009/grammar/pnml">
  <net id="nupn-seed" type="http://www.pnml.org/version-2009/grammar/ptnet">
    <page id="page0">
      <toolspecific tool="nupn" version="1.1">
        <structure units="3" root="u0" safe="true">
          <unit id="u0"><places/><subunits>uA uB</subunits></unit>
          <unit id="uA"><places>P0 P2 P4</places><subunits/></unit>
          <unit id="uB"><places>P1 P3 P5</places><subunits/></unit>
        </structure>
      </toolspecific>
    </page>
  </net>
</pnml>"#;
        let nupn = crate::nupn::parse_nupn(pnml, &net)
            .expect("fixture NUPN parses")
            .expect("fixture NUPN present");
        (net, nupn)
    }

    #[test]
    fn nupn_order_seed_emits_preorder_unit_blocks() {
        let _env = seed_env_guard();
        let (net, nupn) = nupn_seed_fixture();
        let seed = nupn_order_seed(&nupn, net.num_places())
            .expect("hierarchical NUPN yields a non-identity seed");
        // Pre-order DFS from root: uA's places, then uB's places.
        assert_eq!(seed, vec![0, 2, 4, 1, 3, 5]);
    }

    #[test]
    fn nupn_order_seed_skips_identity_and_respects_kill_switch() {
        let _env = seed_env_guard();
        let (net, nupn) = nupn_seed_fixture();
        // Kill-switch: TY_DD_NUPN_ORDER=0 must suppress the seed entirely.
        // (Benign even under test parallelism: the seed is performance-only,
        // so a concurrent reader merely falls back to the unseeded order.)
        crate::env_guard::set_var("TY_DD_NUPN_ORDER", "0");
        assert_eq!(
            nupn_order_seed(&nupn, net.num_places()),
            None,
            "kill-switch must disable NUPN seeding",
        );
        crate::env_guard::remove_var("TY_DD_NUPN_ORDER");
        assert!(nupn_order_seed(&nupn, net.num_places()).is_some());

        // An identity block order must be skipped (pointless work).
        let identity_net = PetriNet {
            name: Some("nupn-identity".into()),
            places: vec![
                PlaceInfo {
                    id: "P0".into(),
                    name: None,
                },
                PlaceInfo {
                    id: "P1".into(),
                    name: None,
                },
            ],
            transitions: vec![],
            initial_marking: vec![1, 0],
        };
        let pnml = r#"<?xml version="1.0"?>
<pnml xmlns="http://www.pnml.org/version-2009/grammar/pnml">
  <net id="nupn-identity" type="http://www.pnml.org/version-2009/grammar/ptnet">
    <page id="page0">
      <toolspecific tool="nupn" version="1.1">
        <structure units="1" root="u0" safe="true">
          <unit id="u0"><places>P0 P1</places><subunits/></unit>
        </structure>
      </toolspecific>
    </page>
  </net>
</pnml>"#;
        let nupn = crate::nupn::parse_nupn(pnml, &identity_net)
            .expect("identity NUPN parses")
            .expect("identity NUPN present");
        assert_eq!(
            nupn_order_seed(&nupn, identity_net.num_places()),
            None,
            "identity seed must be skipped",
        );
    }

    #[test]
    fn nupn_order_seed_covers_unrooted_units_and_uncovered_places() {
        let _env = seed_env_guard();
        // 5 places; the NUPN covers P3 P1 (one unit, no root link to it)
        // under a root covering P4. P0/P2 are uncovered. Expect: root DFS
        // (P4), then remaining units in declaration order (P3 P1), then
        // uncovered ascending (P0, P2).
        let net = PetriNet {
            name: Some("nupn-partial".into()),
            places: (0..5)
                .map(|i| PlaceInfo {
                    id: format!("P{i}"),
                    name: None,
                })
                .collect(),
            transitions: vec![],
            initial_marking: vec![0; 5],
        };
        let pnml = r#"<?xml version="1.0"?>
<pnml xmlns="http://www.pnml.org/version-2009/grammar/pnml">
  <net id="nupn-partial" type="http://www.pnml.org/version-2009/grammar/ptnet">
    <page id="page0">
      <toolspecific tool="nupn" version="1.1">
        <structure units="2" root="u0" safe="false">
          <unit id="u0"><places>P4</places><subunits/></unit>
          <unit id="u1"><places>P3 P1</places><subunits/></unit>
        </structure>
      </toolspecific>
    </page>
  </net>
</pnml>"#;
        let nupn = crate::nupn::parse_nupn(pnml, &net)
            .expect("partial NUPN parses")
            .expect("partial NUPN present");
        let seed = nupn_order_seed(&nupn, net.num_places()).expect("partial cover yields seed");
        assert_eq!(seed, vec![4, 3, 1, 0, 2]);
        // Must be a permutation of 0..n by construction.
        let mut sorted = seed.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, vec![0, 1, 2, 3, 4]);
    }

    // Pillar increment #1: `sound_place_bound` picks the DD value bound from the
    // LP and structural P-invariant sources. Pins the zero-regression contract
    // (in-range LP used as-is, NOT tightened by a smaller structural bound) and
    // the admission win (structural fallback when the LP declined/oversized).
    #[test]
    fn sound_place_bound_prefers_in_range_lp_then_structural_fallback() {
        const MAX: u64 = MAX_PER_PLACE_BOUND;
        // In-range LP is used verbatim — even when structural is tighter — so no
        // net the LP path already admitted changes its encoding (no Gate-4 churn).
        assert_eq!(sound_place_bound(Some(100), Some(50)), Some(100));
        assert_eq!(sound_place_bound(Some(100), None), Some(100));
        assert_eq!(sound_place_bound(Some(MAX), Some(7)), Some(MAX));
        // LP declined (size gate / unbounded): admit via the structural bound.
        assert_eq!(sound_place_bound(None, Some(50)), Some(50));
        // LP oversized: fall back to an in-range structural bound (admission).
        assert_eq!(sound_place_bound(Some(MAX + 1), Some(50)), Some(50));
        // Neither source yields an in-range bound: decline (fail-closed).
        assert_eq!(sound_place_bound(None, None), None);
        assert_eq!(sound_place_bound(Some(MAX + 1), None), None);
        assert_eq!(sound_place_bound(None, Some(MAX + 1)), None);
        assert_eq!(sound_place_bound(Some(MAX + 1), Some(MAX + 1)), None);
    }

    /// Pillar increment #2: `p_invariant_order_seed` groups conserved place
    /// blocks for the FORCE variable order. Soundness is guaranteed generically
    /// (any place permutation is answer-preserving; cross-checked in
    /// `tla-dd/crosscheck.rs`), so this pins the *seed-production* contract:
    /// a non-NUPN conserved net with interleaved blocks yields a valid
    /// non-identity permutation; a structureless net yields `None`; the kill
    /// switch yields `None`.
    fn arc(p: u32, w: u64) -> Arc {
        Arc {
            place: PlaceIdx(p),
            weight: w,
        }
    }
    fn shuttle(id: &str, from: u32, to: u32) -> TransitionInfo {
        TransitionInfo {
            id: id.into(),
            name: None,
            inputs: vec![arc(from, 1)],
            outputs: vec![arc(to, 1)],
        }
    }

    #[test]
    fn p_invariant_order_seed_groups_interleaved_conserved_blocks() {
        let _env = seed_env_guard();
        // Two independent token-conserving blocks, interleaved in PNML index
        // order: block A conserves p0+p2, block B conserves p1+p3.
        let net = PetriNet {
            name: Some("two-blocks".into()),
            places: vec![place("p0"), place("p1"), place("p2"), place("p3")],
            transitions: vec![
                shuttle("a02", 0, 2),
                shuttle("a20", 2, 0),
                shuttle("b13", 1, 3),
                shuttle("b31", 3, 1),
            ],
            initial_marking: vec![1, 1, 1, 1],
        };

        let seed = p_invariant_order_seed(&net, 4).expect("conserved net yields a block seed");
        // Valid permutation of 0..4.
        let mut sorted = seed.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, vec![0, 1, 2, 3], "seed must be a permutation");
        // Non-identity (the interleaved PNML order is de-interleaved).
        assert_ne!(
            seed,
            vec![0, 1, 2, 3],
            "interleaved blocks must be reordered"
        );
        // Each conserved block is contiguous in the seed (the whole point).
        let pos = |p: usize| seed.iter().position(|&x| x == p).unwrap();
        assert_eq!(
            (pos(0) as i64 - pos(2) as i64).abs(),
            1,
            "block A {{p0,p2}} must be contiguous",
        );
        assert_eq!(
            (pos(1) as i64 - pos(3) as i64).abs(),
            1,
            "block B {{p1,p3}} must be contiguous",
        );

        // Kill switch suppresses the seed entirely.
        crate::env_guard::set_var("TY_DD_PINV_ORDER", "0");
        assert!(
            p_invariant_order_seed(&net, 4).is_none(),
            "kill switch → None"
        );
        crate::env_guard::remove_var("TY_DD_PINV_ORDER");
    }

    #[test]
    fn p_invariant_order_seed_declines_without_conservation() {
        let _env = seed_env_guard();
        // A single unbounded source place has NO P-invariant (an isolated place
        // would itself be a trivial constant invariant, so use a lone source).
        let net = PetriNet {
            name: Some("source".into()),
            places: vec![place("p0")],
            transitions: vec![TransitionInfo {
                id: "gen".into(),
                name: None,
                inputs: vec![],
                outputs: vec![arc(0, 1)],
            }],
            initial_marking: vec![0],
        };
        assert!(
            p_invariant_order_seed(&net, 1).is_none(),
            "no conservation structure → no block seed",
        );
    }
}
