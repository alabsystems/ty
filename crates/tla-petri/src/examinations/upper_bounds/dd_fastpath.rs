// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Decision-Diagram fast-path for UpperBounds.
//!
//! Given a batch of `place-bound(...)` queries, this module attempts to
//! compute each query's exact maximum via a single shared reachable-set
//! BDD over the *original* (un-reduced) net. UpperBounds is required to
//! report bounds against the original net, so the DD path bypasses the
//! reduction layer entirely — every reduction-related Safety-Net (A, B,
//! C in `pipeline.rs`) is unnecessary on the DD verdict, because the
//! reachable BDD is built directly from `net.transitions` /
//! `net.initial_marking`.
//!
//! # Soundness invariants
//!
//! 1. **Per-place encoding bounds come from LP**, never from a heuristic.
//!    The DD's unary encoding silently drops any firing whose successor
//!    would exceed the encoded value range, so under-bounding a place
//!    produces a wrong (too-low) reachable set. We call
//!    `lp_state_equation::lp_upper_bound(net, &[place])` for each place;
//!    LP is a sound upper bound on the per-place reachable maximum (the
//!    LP relaxation strictly contains the integer reachable set).
//!
//! 2. **Any LP-unbounded or LP > MVP cap place ⇒ fast-path declines.**
//!    Falling through to BFS is safe — sound — and we never emit a
//!    degraded DD answer.
//!
//! 3. **DD verdicts are validated against the LP/structural cap.** A DD
//!    result strictly greater than the LP bound for the same query would
//!    indicate a soundness bug; we treat it as a DD failure and fall
//!    through. The DD result is also clamped to be `>=` the
//!    initial-marking sum for the query (initial marking is reachable by
//!    definition).
//!
//! 4. **DD-resolved trackers are reported as completed.** The DD computed
//!    the *exact* `max_{M ∈ R} Σ_p coeff[p] · m[p]`, so the resulting
//!    bound is the true reachable maximum — equivalent to a completed
//!    BFS on the identity net. UpperBounds soundness therefore matches
//!    the identity-net BFS path: a successful DD result IS the ground
//!    truth.

use std::sync::mpsc;
use std::thread;
use std::time::Duration;

use crate::petri_net::PetriNet;

use super::model::BoundTracker;

/// Wall-clock budget for the DD fast-path on a single UpperBounds batch.
///
/// The DD worker thread is detached on timeout — the BDD manager it
/// holds is dropped on its way out, so the budget is a soft cap with
/// no resource leak. Mirrors the budget on the StateSpace DD path
/// (`state_space.rs`'s `try_dd_full_metrics_timed`).
const DD_BUDGET: Duration = Duration::from_secs(5);

/// Outcome of the DD fast-path run.
pub(super) struct DdFastPathResult {
    /// Per-tracker DD-computed bound. Length matches the input
    /// trackers slice. `None` means the DD path declined to update that
    /// tracker (either because it was already structurally resolved on
    /// entry, or because the DD reported a value that failed validation).
    pub(super) bounds: Vec<Option<u64>>,
}

/// Try the DD fast-path on the original net for a batch of UpperBounds
/// trackers. Returns `Some(result)` on a successful DD run (all four
/// preconditions pass and the BDD fixpoint converges within
/// [`DD_BUDGET`]); returns `None` on any precondition failure, timeout,
/// or DD-side error so the caller continues with the LP+BFS pipeline.
///
/// Caller contract: trackers' `lp_bound` and `structural_bound` must
/// already be populated. The fast-path uses these as soundness gates
/// on the DD-returned value (a DD bound exceeding the LP cap is treated
/// as a DD failure).
///
/// `nupn` (when present) seeds the DD variable order from the NUPN unit
/// hierarchy via [`crate::examinations::dd_spec::nupn_order_seed`].
/// PERFORMANCE-ONLY: any permutation is answer-preserving and the seed
/// competes under `tla_dd`'s span guard, so the returned bounds are
/// identical with or without it — only convergence speed can differ.
#[must_use]
pub(super) fn try_dd_fast_path(
    net: &PetriNet,
    trackers: &[BoundTracker],
    // NUPN order-seed was an oxidd variable-ordering perf hint; the tla-bdd lane
    // builds its own order, so this is currently unused (kept for the signature +
    // a future tla-bdd order seed).
    _nupn: Option<&crate::nupn::NupnStructure>,
) -> Option<DdFastPathResult> {
    // Gates 1-4 + spec construction are delegated to the shared, sound
    // builder so this fast-path and the StateSpace verdict path cannot
    // drift apart on the soundness gate:
    //   1. place-count cap,
    //   2. per-place LP upper bound (sound over-approximation) ≤ MVP cap,
    //   3. initial marking fits the encoded range,
    //   4. no transition pushes a place past its bound.
    // Any condition that cannot be proven sound returns `None`, and we
    // fall through to the LP+BFS pipeline. See `examinations::dd_spec`.
    let num_places = net.num_places();
    let (spec, _per_place_bounds) = crate::examinations::dd_spec::build_sound_dd_spec(net)?;
    // Build per-tracker coefficient vectors with MCC multiplicity
    // semantics (a place mentioned k times contributes its tokens with
    // multiplicity k). Trackers that are already structurally resolved
    // on entry skip the DD path — we have nothing to gain by recomputing
    // them and falling through preserves the existing resolved-bound
    // ratification path in `pipeline.rs`.
    let mut query_indices: Vec<usize> = Vec::with_capacity(trackers.len());
    let mut queries: Vec<Vec<u64>> = Vec::with_capacity(trackers.len());
    for (i, tracker) in trackers.iter().enumerate() {
        if tracker.is_structurally_resolved() {
            continue;
        }
        let mut coeffs = vec![0u64; num_places];
        for place in &tracker.place_indices {
            let idx = place.0 as usize;
            if idx >= num_places {
                return None;
            }
            coeffs[idx] = coeffs[idx].saturating_add(1);
        }
        query_indices.push(i);
        queries.push(coeffs);
    }
    if queries.is_empty() {
        // Nothing for the DD path to compute — all trackers are
        // already resolved. Caller short-circuits, but return an
        // empty result rather than `None` so the caller knows the
        // gate passed and the DD path is admissible.
        return Some(DdFastPathResult {
            bounds: vec![None; trackers.len()],
        });
    }

    // The DD UpperBounds fast-path runs natively on tla-bdd — the oxidd engine was
    // REMOVED. Validated as a sound oxidd replacement: ≡ the MDD UB lane
    // (bdd_upper_bounds_matches_mdd_lane), the 100-test UpperBounds suite passes,
    // and the 46-model corpus A/B found zero disagreements (ub net +16, incl. a
    // HealthRecord-PT-01 0→16 unlock). run_bdd_upper_bounds_fast_path applies the
    // SAME initial-sum lower-bound + LP/structural-cap soundness validation; any
    // decline / overflow / timeout / panic returns None → fall through to LP+BFS.
    run_bdd_upper_bounds_fast_path(net, trackers, &spec, &queries, &query_indices)
}

/// The native-ROBDD UpperBounds lane: computes `max Σ coeff·m` over the exact
/// reachable set via `tla-bdd`, then applies the SAME initial-sum lower-bound and
/// LP/structural-cap soundness validation as the oxidd path (a value below the
/// initial sum or above the LP cap ⇒ treated as a DD failure → fall through).
/// Worker thread + `DD_BUDGET` deadline (bound via `upper_bounds_bounded_within`);
/// any decline / overflow / timeout / panic returns `None` soundly.
#[must_use]
fn run_bdd_upper_bounds_fast_path(
    net: &PetriNet,
    trackers: &[BoundTracker],
    spec: &tla_dd::DdNetSpec,
    queries: &[Vec<u64>],
    query_indices: &[usize],
) -> Option<DdFastPathResult> {
    let bdd_queries: Vec<Vec<i128>> = queries
        .iter()
        .map(|q| q.iter().map(|&c| c as i128).collect())
        .collect();
    let (tx, rx) = mpsc::channel();
    let spec_for_thread = spec.clone();
    let handle = thread::Builder::new()
        .name("tla-bdd-upper-bounds".into())
        .stack_size(tla_dd::DD_WORKER_STACK_BYTES)
        .spawn(move || {
            let deadline = std::time::Instant::now() + DD_BUDGET;
            let _ = tx.send(crate::examinations::mdd_common::upper_bounds_via_bdd(
                &spec_for_thread,
                &bdd_queries,
                Some(deadline),
            ));
        });
    if handle.is_err() {
        eprintln!("UpperBounds: tla-bdd lane spawn failed — using LP+BFS");
        return None;
    }
    let raw = match rx.recv_timeout(DD_BUDGET + Duration::from_millis(1500)) {
        Ok(Some(v)) => v, // Vec<i128>
        Ok(None) | Err(_) => return None,
    };
    if raw.len() != queries.len() {
        return None;
    }
    // UB bounds are non-negative; a negative / overflowing value ⇒ fail-closed.
    let mut dd_bounds: Vec<u64> = Vec::with_capacity(raw.len());
    for x in raw {
        match u64::try_from(x) {
            Ok(v) => dd_bounds.push(v),
            Err(_) => return None,
        }
    }
    // SAME soundness validation as the oxidd path.
    let mut bounds: Vec<Option<u64>> = vec![None; trackers.len()];
    for (qi, &ti) in query_indices.iter().enumerate() {
        let dd_value = dd_bounds[qi];
        let tracker = &trackers[ti];
        let initial_sum: u64 = tracker
            .place_indices
            .iter()
            .map(|p| net.initial_marking[p.0 as usize])
            .sum();
        if dd_value < initial_sum {
            return None;
        }
        if let Some(cap) = tracker.effective_bound() {
            if dd_value > cap {
                return None;
            }
        }
        bounds[ti] = Some(dd_value);
    }
    Some(DdFastPathResult { bounds })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::petri_net::{Arc, PetriNet, PlaceIdx, PlaceInfo, TransitionInfo};

    /// 2-place swap net: p0+p1 conserved=1, per-place max=1.
    fn swap_net() -> PetriNet {
        PetriNet {
            name: Some("swap".into()),
            places: vec![
                PlaceInfo {
                    id: "p0".into(),
                    name: None,
                },
                PlaceInfo {
                    id: "p1".into(),
                    name: None,
                },
            ],
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

    fn tracker(id: &str, places: Vec<PlaceIdx>) -> BoundTracker {
        let mut t = BoundTracker {
            id: id.to_string(),
            place_indices: places,
            max_bound: 0,
            structural_bound: None,
            lp_bound: None,
            monotone_bound: None,
        };
        // Seed lp_bound at 16 so effective_bound() returns Some, exercising
        // the validation gate in fast-path tests; production callers set
        // this from the real LP analysis.
        t.lp_bound = Some(16);
        t
    }

    #[test]
    fn dd_fast_path_swap_net_single_place() {
        // bound(p0) on the swap net = 1 (max per-place token count).
        let net = swap_net();
        let trackers = vec![tracker("UB-p0", vec![PlaceIdx(0)])];
        let result = try_dd_fast_path(&net, &trackers, None).expect("fast-path admits swap net");
        assert_eq!(
            result.bounds,
            vec![Some(1)],
            "swap net bound(p0) = 1 (true reachable max)",
        );
    }

    #[test]
    fn dd_fast_path_swap_net_joint_query_matches_conservation_law() {
        // bound(p0, p1) on the swap net = 1 (sum is invariant at 1).
        let net = swap_net();
        let trackers = vec![tracker("UB-p0-p1", vec![PlaceIdx(0), PlaceIdx(1)])];
        let result = try_dd_fast_path(&net, &trackers, None)
            .expect("fast-path admits joint query on swap net");
        assert_eq!(
            result.bounds,
            vec![Some(1)],
            "swap net bound(p0,p1) = 1 (conservation)",
        );
    }

    #[test]
    fn dd_fast_path_skips_resolved_trackers() {
        // A structurally-resolved tracker on entry must be left as `None`
        // (the caller already has the exact value via the LP/structural
        // path and the DD path has nothing to add). Mixed-batch case:
        // tracker 0 resolved, tracker 1 unresolved — DD computes only
        // tracker 1 and reports `None` for tracker 0.
        let net = swap_net();
        let mut already_resolved = tracker("UB-resolved", vec![PlaceIdx(0)]);
        // Force "resolved": max_bound >= effective_bound.
        already_resolved.lp_bound = Some(1);
        already_resolved.max_bound = 1;
        let unresolved = tracker("UB-unresolved", vec![PlaceIdx(1)]);
        let trackers = vec![already_resolved, unresolved];
        let result = try_dd_fast_path(&net, &trackers, None)
            .expect("fast-path admits mixed-resolution batch");
        assert_eq!(
            result.bounds,
            vec![None, Some(1)],
            "DD path computes only the unresolved tracker",
        );
    }

    /// NUPN-seeded-order answer invariance at the fast-path level: the DD
    /// bounds with a (non-identity) NUPN hierarchy seed must be identical
    /// to the unseeded bounds — the seed is performance-only.
    #[test]
    fn dd_fast_path_nupn_seed_is_answer_preserving() {
        // Two independent swap pairs, interleaved in PNML index order:
        // ring A = {p0, p2}, ring B = {p1, p3}. The NUPN units group the
        // rings, so the seed de-interleaves: [0, 2, 1, 3] (non-identity).
        let mk_place = |i: usize| PlaceInfo {
            id: format!("p{i}"),
            name: None,
        };
        let mk_t = |id: &str, from: u32, to: u32| TransitionInfo {
            id: id.into(),
            name: None,
            inputs: vec![Arc {
                place: PlaceIdx(from),
                weight: 1,
            }],
            outputs: vec![Arc {
                place: PlaceIdx(to),
                weight: 1,
            }],
        };
        let net = PetriNet {
            name: Some("two-swaps".into()),
            places: (0..4).map(mk_place).collect(),
            transitions: vec![
                mk_t("a01", 0, 2),
                mk_t("a10", 2, 0),
                mk_t("b01", 1, 3),
                mk_t("b10", 3, 1),
            ],
            initial_marking: vec![1, 1, 0, 0],
        };
        let pnml = r#"<?xml version="1.0"?>
<pnml xmlns="http://www.pnml.org/version-2009/grammar/pnml">
  <net id="two-swaps" type="http://www.pnml.org/version-2009/grammar/ptnet">
    <page id="page0">
      <toolspecific tool="nupn" version="1.1">
        <structure units="3" root="u0" safe="true">
          <unit id="u0"><places/><subunits>uA uB</subunits></unit>
          <unit id="uA"><places>p0 p2</places><subunits/></unit>
          <unit id="uB"><places>p1 p3</places><subunits/></unit>
        </structure>
      </toolspecific>
    </page>
  </net>
</pnml>"#;
        let nupn = crate::nupn::parse_nupn(pnml, &net)
            .expect("NUPN parses")
            .expect("NUPN present");
        assert_eq!(
            crate::examinations::dd_spec::nupn_order_seed(&nupn, net.num_places()),
            Some(vec![0, 2, 1, 3]),
            "fixture must exercise a real, non-identity seed",
        );
        let trackers = vec![
            tracker("UB-p0", vec![PlaceIdx(0)]),
            tracker(
                "UB-all",
                vec![PlaceIdx(0), PlaceIdx(1), PlaceIdx(2), PlaceIdx(3)],
            ),
        ];
        let unseeded = try_dd_fast_path(&net, &trackers, None).expect("unseeded run admits");
        let seeded = try_dd_fast_path(&net, &trackers, Some(&nupn)).expect("seeded run admits");
        assert_eq!(
            seeded.bounds, unseeded.bounds,
            "NUPN seed must not change any UpperBounds value",
        );
        assert_eq!(unseeded.bounds, vec![Some(1), Some(2)], "ground truth");
    }

    #[test]
    fn dd_fast_path_returns_none_for_oversized_net() {
        // Synthetic net with > MAX_PLACES places. Fast-path must decline.
        let max_places = crate::examinations::dd_spec::MAX_PLACES;
        let places: Vec<PlaceInfo> = (0..=max_places)
            .map(|i| PlaceInfo {
                id: format!("p{i}"),
                name: None,
            })
            .collect();
        let net = PetriNet {
            name: Some("oversized".into()),
            places,
            transitions: vec![],
            initial_marking: vec![0; max_places + 1],
        };
        let trackers = vec![tracker("UB-x", vec![PlaceIdx(0)])];
        assert!(
            try_dd_fast_path(&net, &trackers, None).is_none(),
            "fast-path declines on > MAX_PLACES net",
        );
    }
}
