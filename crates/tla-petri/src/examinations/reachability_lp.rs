// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! LP state-equation pre-seeding for reachability properties.
//!
//! Runs between BMC witness seeding and BFS exploration. Uses the LP
//! relaxation of the Petri net state equation to prove *unreachability*:
//!
//! - if LP proves `φ` always false → seed `FALSE`
//! - if LP proves `φ` always true → seed `TRUE`
//!
//! Handles pure marking predicates and conservative `IsFireable` cases where
//! LP/P-invariant reasoning is decisive. Non-decisive formulas are left
//! unresolved for BFS.

use std::time::Instant;

use crate::lp_state_equation::lp_predicate_truth;
use crate::petri_net::PetriNet;

use super::reachability::{resolve_tracker, PropertyTracker, ReachabilityResolutionSource};

/// Run LP state-equation seeding on unresolved reachability trackers.
///
/// For each tracker without a verdict:
/// - if `φ` is LP-proved always false, seed the property `FALSE`
/// - if `φ` is LP-proved always true, seed the property `TRUE`
///
/// `deadline` bounds the phase. A pure-`IntLe` (Cardinality) formula issues only
/// a couple of LP solves, but a fireability formula can trigger one LP solve per
/// listed transition per atom (hundreds on unfolded nets), so the cumulative
/// time can run far past an MCC deadline. The deadline is therefore polled both
/// between formulas (here) AND inside `lp_predicate_truth` / `lp_fireability_truth`
/// so a single heavy formula cannot overrun the reserved budget: on expiry the
/// in-flight formula returns `None` and falls through to the exhaustive BFS,
/// leaving every verdict resolved so far in place (`resolve_tracker` is
/// first-writer-wins). Verdict-preserving and progress-preserving.
pub(crate) fn run_lp_seeding(
    net: &PetriNet,
    trackers: &mut [PropertyTracker],
    deadline: Option<Instant>,
) {
    for tracker in trackers.iter_mut() {
        if deadline.is_some_and(|limit| Instant::now() >= limit) {
            break;
        }
        if tracker.verdict.is_some() {
            continue;
        }
        if let Some(truth) = lp_predicate_truth(net, &tracker.predicate, deadline) {
            resolve_tracker(tracker, truth, ReachabilityResolutionSource::Lp, None);
        }
    }
}

#[cfg(test)]
#[path = "reachability_lp_tests.rs"]
mod tests;
