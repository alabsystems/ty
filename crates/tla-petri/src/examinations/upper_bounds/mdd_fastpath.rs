// Copyright 2026 Andrew Yates
// Licensed under the Apache License, Version 2.0

//! Native MDD UpperBounds fast-path — the MDD twin of [`super::dd_fastpath`].
//!
//! Computes the EXACT `max_{M ∈ R} Σ_p coeff[p]·m[p]` for every unresolved
//! tracker over a single shared reachable MDD `R`, via
//! [`tla_mdd::MddNet::upper_bounds`] (which builds `R` once by saturation and
//! evaluates each query with `tla_mdd::max_weighted_sum_of`). Same admission
//! gate as the StateSpace/Reachability MDD lanes
//! ([`crate::examinations::mdd_common::build_mdd_spec_for_net`]: sound per-place
//! LP bounds + structural gates + edge-width cap), so the encoded range is a
//! superset of every place's reachable projection ⇒ `R` and every query maximum
//! are EXACT.
//!
//! Fail-closed: a `None` admission, a build `Err` (overflow / node cap /
//! deadline), spawn failure, or a per-query `i128`→`u64` overflow leaves the
//! affected tracker(s) unresolved; the pipeline falls through to LP+BFS. A wrong
//! bound is impossible — the lane can only withhold.

use std::sync::mpsc;
use std::thread;
use std::time::{Duration, Instant};

use crate::examinations::mdd_common::{
    build_mdd_spec_for_net, dd_spec_to_ordered_mdd_net, order_mdd_net,
};
use crate::petri_net::PetriNet;

use super::model::BoundTracker;

/// Wall-clock budget when no caller deadline is supplied (mirrors the BDD twin's
/// `DD_BUDGET`). Production MCC always supplies a deadline upstream.
const MDD_BUDGET: Duration = Duration::from_secs(5);

/// Permute UpperBounds coefficient vectors into the level coordinates of an
/// ordered MDD net (`inv[place] = level`, from
/// [`crate::examinations::mdd_common::dd_spec_to_ordered_mdd_net`]). The
/// per-query maximum `max_{M ∈ R} Σ_p coeff[p]·m[p]` is invariant — the products
/// are merely reindexed — so every bound is unchanged; only the saturation order
/// (hence MDD size / feasibility) improves. `inv.len() == q.len() == num_places`,
/// so the reindex is in-bounds.
fn permute_ub_queries(queries: &[Vec<i128>], inv: &[usize]) -> Vec<Vec<i128>> {
    queries
        .iter()
        .map(|q| {
            let mut out = vec![0i128; q.len()];
            for (orig, &c) in q.iter().enumerate() {
                out[inv[orig]] = c;
            }
            out
        })
        .collect()
}

/// Per-tracker MDD-computed exact bound (`None` = declined / already resolved).
pub(super) struct MddFastPathResult {
    pub(super) bounds: Vec<Option<u64>>,
}

/// Run the exact MDD UpperBounds fast-path over `trackers`. `Some(result)` on a
/// successful run (gate passed); `None` if the lane declines (gate / spawn /
/// budget), leaving the caller to fall through unchanged.
pub(super) fn try_mdd_fast_path(
    net: &PetriNet,
    trackers: &[BoundTracker],
    deadline: Option<Instant>,
    colored: Option<&crate::hlpnml::ColoredNet>,
    nupn: Option<&crate::nupn::NupnStructure>,
) -> Option<MddFastPathResult> {
    // Net-building strategy. For a COLORED net we prefer the compact colored MDD
    // (`build_colored_mdd_net`: binding-level transitions, no full unfold — it
    // uses the SAME (place,color) slot encoding as `unfold_to_pt`, so the
    // per-unfolded-place query coeffs below align directly). The P/T spec is the
    // fallback (when the colored path declines / is out-of-sub-class) and the
    // ONLY path for non-colored nets. Built lazily in the worker (big stack).
    let pt_spec = build_mdd_spec_for_net(net);
    if colored.is_none() && pt_spec.is_none() {
        return None; // non-colored net, P/T admission gate declined
    }
    let num_places = net.num_places();
    // NUPN unit-hierarchy seed for the P/T MDD order (span-guarded; `None` ⇒ base
    // FORCE). Computed here (owned) so it can move into the worker thread. Only
    // the P/T branch uses it; the colored branch keeps its own slot order.
    let order_seed: Option<Vec<usize>> =
        nupn.and_then(|n| crate::examinations::dd_spec::nupn_order_seed(n, num_places));

    // Build per-tracker coefficient vectors with MCC multiplicity (a place
    // listed k times contributes its tokens with coefficient k). Already-resolved
    // trackers skip the lane (nothing to gain).
    let mut query_indices: Vec<usize> = Vec::with_capacity(trackers.len());
    let mut queries: Vec<Vec<i128>> = Vec::with_capacity(trackers.len());
    for (i, tracker) in trackers.iter().enumerate() {
        if tracker.is_structurally_resolved() {
            continue;
        }
        let mut coeffs = vec![0i128; num_places];
        for place in &tracker.place_indices {
            let idx = place.0 as usize;
            if idx >= num_places {
                return None;
            }
            coeffs[idx] += 1;
        }
        query_indices.push(i);
        queries.push(coeffs);
    }
    if queries.is_empty() {
        return Some(MddFastPathResult {
            bounds: vec![None; trackers.len()],
        });
    }

    // Worker thread with the big DD stack + a wall-clock budget (mirrors the BDD
    // twin and the reachability MDD lane). The MDD saturation respects the
    // deadline internally; the recv timeout is a backstop.
    let worker_deadline = {
        let cap = Instant::now() + MDD_BUDGET;
        Some(match deadline {
            Some(d) if d < cap => d,
            _ => cap,
        })
    };
    let (tx, rx) = mpsc::channel();
    // Own the inputs for the worker. The colored net (if any) is cloned so the
    // compact `build_colored_mdd_net` can run on the big stack; it falls back to
    // the P/T spec if it declines (out-of-sub-class), and `None` colored uses
    // the P/T spec directly.
    let colored_owned = colored.cloned();
    let handle = thread::Builder::new()
        .name("tla-mdd-upper-bounds".into())
        .stack_size(tla_dd::DD_WORKER_STACK_BYTES)
        .spawn(move || {
            // Build the net + align the query coefficient vectors to its level
            // order. The P/T branch reorders the net for saturation locality
            // (`dd_spec_to_ordered_mdd_net`, the scale lever the BDD lane already
            // pulls) and permutes each coeff vector into the same levels via
            // `inv` — SOUND: an isomorphic relabeling, the per-query maximum is
            // unchanged. The colored branch keeps its own (place,color) slot
            // order, with which the coeffs already align (the FREE query mapping).
            let (mdd_net, aligned_queries) = match colored_owned {
                Some(c) => match crate::symbolic_colored::build_colored_mdd_net(&c) {
                    Ok(net) => {
                        // Order the colored MDD too (its (place,color) slots are
                        // as order-sensitive as P/T places); permute the coeff
                        // vectors into the same slot order. SOUND: isomorphic
                        // relabel, per-query maxima invariant.
                        let (net, inv) = order_mdd_net(net);
                        let q = permute_ub_queries(&queries, &inv);
                        (net, q)
                    }
                    Err(_) => match &pt_spec {
                        Some(s) => {
                            let (net, inv) = dd_spec_to_ordered_mdd_net(s, order_seed.as_deref());
                            let q = permute_ub_queries(&queries, &inv);
                            (net, q)
                        }
                        None => return, // both declined ⇒ recv error ⇒ lane declines
                    },
                },
                None => match &pt_spec {
                    Some(s) => {
                        let (net, inv) = dd_spec_to_ordered_mdd_net(s, order_seed.as_deref());
                        let q = permute_ub_queries(&queries, &inv);
                        (net, q)
                    }
                    None => return,
                },
            };
            let _ = tx.send(mdd_net.upper_bounds(&aligned_queries, worker_deadline));
        });
    if handle.is_err() {
        eprintln!("UpperBounds: MDD fast-path thread spawn failed — using LP+BFS");
        return None;
    }
    let per_query = match rx.recv_timeout(MDD_BUDGET + Duration::from_millis(1500)) {
        Ok(Ok(bounds)) => bounds,
        Ok(Err(err)) => {
            eprintln!("UpperBounds: MDD fast-path fell through ({err:?}) — using LP+BFS");
            return None;
        }
        Err(_) => {
            eprintln!(
                "UpperBounds: MDD fast-path exceeded {}s budget — using LP+BFS",
                MDD_BUDGET.as_secs()
            );
            return None;
        }
    };

    // Map per-query bounds back to per-tracker slots. UpperBounds coefficients
    // are non-negative, so the maximum is non-negative; an `i128`→`u64` overflow
    // (or a per-query decline) leaves that tracker unresolved (fail-closed).
    let mut bounds = vec![None; trackers.len()];
    for (qi, b) in per_query.iter().enumerate() {
        if let Some(v) = b {
            if let Ok(u) = u64::try_from(*v) {
                bounds[query_indices[qi]] = Some(u);
            }
        }
    }
    Some(MddFastPathResult { bounds })
}
