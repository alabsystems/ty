// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! PDR/IC3 reachability seeding on Petri nets.
//!
//! This phase can resolve both reachability directions:
//! - `AG(phi)` by proving safety of `phi`
//! - `EF(phi)` by disproving safety of `not phi`

use std::time::{Duration, Instant};

use crate::petri_net::PetriNet;
use crate::property_xml::PathQuantifier;
use crate::resolved_predicate::ResolvedPredicate;

use super::pdr_encoding::{solve_petri_net_pdr, PdrCheckResult, PdrConfig};
use super::reachability::{resolve_tracker, PropertyTracker, ReachabilityResolutionSource};

const PDR_SEED_TIMEOUT: Duration = Duration::from_secs(5);
const ENABLE_REACHABILITY_PDR_ENV: &str = "TY_MCC_ENABLE_REACHABILITY_PDR";

fn env_flag_enabled(key: &str, default: bool) -> bool {
    env_flag_value_enabled(std::env::var(key).ok().as_deref(), default)
}

fn env_flag_value_enabled(value: Option<&str>, default: bool) -> bool {
    match value.map(str::trim) {
        Some("1" | "true" | "TRUE" | "yes" | "on" | "ON") => true,
        Some("0" | "false" | "FALSE" | "no" | "off" | "OFF") => false,
        Some("") | None => default,
        Some(_) => default,
    }
}

/// PDR seeding for reachability is sound by construction: `solve_petri_net_pdr`
/// only resolves trackers when it returns a definite Safe/Unsafe verdict, and
/// `predicate_contains_fireability` strips out the predicate class with
/// historical wrong-answer risk before the solver is even invoked. Defaulting
/// on lets short cardinality properties resolve in 5 s without burning BFS
/// budget. Set `TY_MCC_ENABLE_REACHABILITY_PDR=0` (or `false`, `off`) to
/// force-disable.
fn reachability_pdr_enabled() -> bool {
    env_flag_enabled(ENABLE_REACHABILITY_PDR_ENV, true)
}

fn predicate_contains_fireability(predicate: &ResolvedPredicate) -> bool {
    match predicate {
        ResolvedPredicate::And(children) | ResolvedPredicate::Or(children) => {
            children.iter().any(predicate_contains_fireability)
        }
        ResolvedPredicate::Not(inner) => predicate_contains_fireability(inner),
        ResolvedPredicate::IsFireable(_) => true,
        ResolvedPredicate::IntLe(..) | ResolvedPredicate::True | ResolvedPredicate::False => false,
    }
}

pub(crate) fn run_pdr_seeding(
    net: &PetriNet,
    trackers: &mut [PropertyTracker],
    deadline: Option<Instant>,
) {
    // Reachability PDR currently has known historical wrong-answer risk on
    // fireability predicates, so keep it opt-in for competition runs and avoid
    // fireability atoms even when the phase is explicitly enabled.
    if !reachability_pdr_enabled() {
        return;
    }

    for tracker in trackers
        .iter_mut()
        .filter(|tracker| tracker.verdict.is_none())
    {
        if predicate_contains_fireability(&tracker.predicate) {
            continue;
        }

        let timeout = deadline
            .map(|limit| PDR_SEED_TIMEOUT.min(limit.saturating_duration_since(Instant::now())))
            .unwrap_or(PDR_SEED_TIMEOUT);
        if timeout.is_zero() {
            break;
        }
        let exact_fallback_budget = deadline.map(|_| timeout);

        let safety_property = match tracker.quantifier {
            PathQuantifier::AG => tracker.predicate.clone(),
            PathQuantifier::EF => ResolvedPredicate::Not(Box::new(tracker.predicate.clone())),
        };
        let result = solve_petri_net_pdr(
            net,
            &safety_property,
            &PdrConfig {
                time_budget: timeout,
                exact_fallback_budget,
                exact_fallback_deadline: deadline,
                ..PdrConfig::default()
            },
            None,
        );
        if deadline.is_some_and(|limit| Instant::now() >= limit) {
            break;
        }

        match (tracker.quantifier, result) {
            (PathQuantifier::AG, PdrCheckResult::Safe) => {
                resolve_tracker(tracker, true, ReachabilityResolutionSource::Pdr, None)
            }
            (PathQuantifier::AG, PdrCheckResult::Unsafe) => {
                resolve_tracker(tracker, false, ReachabilityResolutionSource::Pdr, None)
            }
            (PathQuantifier::EF, PdrCheckResult::Safe) => {
                resolve_tracker(tracker, false, ReachabilityResolutionSource::Pdr, None)
            }
            (PathQuantifier::EF, PdrCheckResult::Unsafe) => {
                resolve_tracker(tracker, true, ReachabilityResolutionSource::Pdr, None)
            }
            (_, PdrCheckResult::Unknown) => {}
        }
    }
}

#[cfg(test)]
#[path = "reachability_pdr_tests.rs"]
mod tests;
