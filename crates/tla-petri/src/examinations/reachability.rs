// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! ReachabilityCardinality and ReachabilityFireability examinations.
//!
//! Both examinations share the same observer: each property is an EF(φ) or
//! AG(φ) formula where φ is a boolean predicate over markings (integer
//! comparisons on token counts) and transition enablement.
//!
//! - EF(φ): TRUE if any reachable state satisfies φ.
//! - AG(φ): TRUE if all reachable states satisfy φ.
//!
//! The observer evaluates all properties simultaneously during a single BFS
//! pass and early-terminates when all properties have definitive answers.
//!
//! Before BFS, bounded model checking (BMC) via ay is attempted on the
//! original net. BMC can seed witness verdicts (EF=TRUE, AG=FALSE) without
//! full exploration. BFS then handles the remaining unresolved properties.

#[cfg(feature = "dd-backend")]
mod dd_fastpath;
#[cfg(feature = "dd-backend")]
mod mdd_fastpath;
mod observer;
mod pipeline;
mod reduction;
mod types;

pub(crate) use pipeline::check_reachability_properties_with_aliases;
pub(crate) use pipeline::check_reachability_properties_with_aliases_and_nupn;
pub(crate) use pipeline::check_reachability_properties_with_flush_and_nupn;
pub(crate) use reduction::protected_places_for_prefire;
pub(crate) use types::{resolve_tracker, PropertyTracker, ReachabilityResolutionSource};

// Re-exported (crate-wide) so under-approximation witness lanes outside this
// module — the CTL pipeline's shallow EF/AG raw-net walk, and the OneSafe
// FALSE-witness walk in `examination_non_property::deadlock_one_safe` — can
// clamp their walk to the same reserve-preserving leftover-budget slice that
// keeps them from starving the exhaustive BFS tail.
pub(crate) use pipeline::under_approx_lane_deadline;
pub(in crate::examinations) use types::prepare_trackers_with_aliases;

#[cfg(test)]
pub(crate) use observer::ReachabilityObserver;
#[cfg(test)]
pub(crate) use pipeline::check_reachability_properties;
#[cfg(test)]
pub(crate) use types::prepare_trackers;

// Test-visible re-exports: the test sidecars use `super::*` and expect items
// that previously lived directly in this file.
#[cfg(test)]
pub(super) use reduction::{explore_reachability_on_reduced_net, reduce_reachability_queries};
#[cfg(test)]
pub(super) use types::{assemble_results, finalize_exhaustive_completion};

// LP state equation seeding is implemented in reachability_lp module.
// It handles both EF (direct infeasibility) and AG (conjunction decomposition).

#[cfg(test)]
#[path = "reachability_tests/mod.rs"]
mod tests;

#[cfg(test)]
#[path = "reachability_por_tests.rs"]
mod por_tests;
