// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Colored-to-P/T net unfolding engine.
//!
//! Takes a [`ColoredNet`] and produces a standard [`PetriNet`] plus
//! [`PropertyAliases`] mapping colored names to unfolded indices.
//!
//! Phase 2: handles `CyclicEnum`, `Dot`, and `Product` sorts.

mod context;
mod places;
#[cfg(feature = "dd-backend")]
pub(crate) mod symbolic_build;
mod transitions;

use std::time::Instant;

use crate::error::PnmlError;
use crate::hlpnml::ColoredNet;
use crate::model::PropertyAliases;
use crate::petri_net::PetriNet;

use context::UnfoldContext;
use places::unfold_places;
use transitions::unfold_transitions;

/// Maximum number of unfolded places before aborting.
pub(crate) const MAX_UNFOLDED_PLACES: usize = 100_000;
/// FLOOR of the unfolded-transition cap — the historical fixed value; kept as
/// the lower clamp of [`max_unfolded_transitions`] so no environment admits
/// less than before the cap became memory-aware.
pub(crate) const MAX_UNFOLDED_TRANSITIONS: usize = 500_000;
/// CEILING of the memory-aware unfolded-transition cap. Sized to admit the
/// verified in-band colored families (PhilosophersDyn-COL-80 ≈ 1.04M,
/// TokenRing-COL-100 ≈ 1.03M, GlobalResAllocation-COL-09 ≈ 1.0M unfolded
/// transitions) while keeping the materialized net far below the runtime
/// engines' own adaptive memory guards. Under the MCC 16 GB confinement the
/// derived cap lands around ~1.6M — the larger GlobalResAllocation sizes
/// (COL-10/11 ≈ 1.7–2.7M) only admit on hosts with ≳28 GB effective-available
/// memory (audit 2026-07-11: the cap is a LIVE first-call reading, so heavy
/// concurrent memory load can pin it at the 500k floor — always fail-closed).
pub(crate) const MAX_UNFOLDED_TRANSITIONS_CEIL: usize = 3_000_000;

/// Memory-aware unfolded-transition cap (v8 diagnosis, 2026-07-10).
///
/// The fixed 500k cap instant-declined whole colored families by hair-widths —
/// PhilosophersDyn-COL-80's guardless 3-variable transitions yield exactly
/// 80³ = 512,000 bindings, 2.4% over — while the COL-50 sibling (125k
/// bindings) runs the FULL pipeline in seconds, proving the only blocker was
/// the number. Scale the cap with effective-available memory (~10% at a
/// conservative ~1 KB per materialized transition), clamped to
/// [500k (the historical floor), 3M]. On the MCC 16 GB confinement this
/// yields ~1.6M (admits PhilosophersDyn-COL-80 / TokenRing-COL-100); on
/// larger hosts it tops out at 3M.
///
/// Fail-closed all the way down: the unfolding arithmetic is exact and
/// unchanged (guards fail closed on unresolvable operands), the 50M
/// binding-iteration pre-check and the UnfoldBudget deadline polls stay
/// verbatim, and every downstream engine keeps its own adaptive memory
/// guard — a bigger cap changes only WHEN we decline, never WHAT we answer.
pub(crate) fn max_unfolded_transitions() -> usize {
    static CAP: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *CAP.get_or_init(|| {
        const EST_BYTES_PER_UNFOLDED_TRANSITION: usize = 1024;
        tla_resource::platform::effective_available_bytes()
            .map(|bytes| ((bytes as f64 * 0.10) as usize) / EST_BYTES_PER_UNFOLDED_TRANSITION)
            .unwrap_or(MAX_UNFOLDED_TRANSITIONS)
            .clamp(MAX_UNFOLDED_TRANSITIONS, MAX_UNFOLDED_TRANSITIONS_CEIL)
    })
}
/// Maximum number of variable-binding combinations a single colored
/// transition may enumerate (the Cartesian product of its variables' sort
/// cardinalities). Bounds both the odometer iteration count and the binding
/// Vec so a transition over large sorts declines cleanly instead of OOMing /
/// stalling. Generous relative to [`MAX_UNFOLDED_TRANSITIONS`] so a restrictive
/// guard that filters a large product down to a small set is still unfolded.
pub(crate) const MAX_BINDING_ITERATIONS: usize = 50_000_000;

/// Cooperative-abort budget for colored unfolding.
///
/// Carries an optional wall-clock deadline. The two unfolding hot loops
/// (place expansion and transition/binding expansion) poll
/// [`UnfoldBudget::check`] every few thousand iterations and abort cleanly
/// with [`PnmlError::ColoredUnfoldUnavailable`] once the deadline passes,
/// turning what was an unbounded load-time stall (OS kill / hard timeout)
/// into a recoverable CANNOT_COMPUTE.
#[derive(Clone, Copy, Default)]
pub(crate) struct UnfoldBudget {
    deadline: Option<Instant>,
}

impl UnfoldBudget {
    /// Construct a budget with an optional wall-clock deadline.
    pub(crate) fn new(deadline: Option<Instant>) -> Self {
        Self { deadline }
    }

    /// Return `Err(ColoredUnfoldUnavailable)` if the deadline has passed.
    /// Cheap: a single `Instant::now()` comparison, called only on the
    /// throttled boundary so steady-state cost is negligible.
    #[inline]
    pub(super) fn check(&self, what: &str) -> Result<(), PnmlError> {
        if let Some(deadline) = self.deadline {
            if Instant::now() >= deadline {
                return Err(PnmlError::ColoredUnfoldUnavailable {
                    reason: format!("{what} exceeded load-time unfolding deadline"),
                });
            }
        }
        Ok(())
    }
}

/// Result of unfolding a colored net.
pub(crate) struct UnfoldedNet {
    pub net: PetriNet,
    pub aliases: PropertyAliases,
}

/// A concrete color value (index into the sort's constant list).
type ColorValue = usize;

/// Unfold a colored net into a standard P/T net.
///
/// Returns the unfolded net and property aliases mapping colored names
/// to their unfolded P/T indices.
///
/// # Errors
///
/// Returns `PnmlError` if the unfolded net exceeds size limits or if
/// the colored net contains unsupported constructs.
pub(crate) fn unfold_to_pt(colored: &ColoredNet) -> Result<UnfoldedNet, PnmlError> {
    unfold_to_pt_with_budget(colored, &UnfoldBudget::default())
}

/// Unfold a colored net into a standard P/T net under a cooperative budget.
///
/// Identical to [`unfold_to_pt`] but polls `budget` in the two hot loops so a
/// load-time deadline produces a clean [`PnmlError::ColoredUnfoldUnavailable`]
/// instead of an unbounded stall.
pub(crate) fn unfold_to_pt_with_budget(
    colored: &ColoredNet,
    budget: &UnfoldBudget,
) -> Result<UnfoldedNet, PnmlError> {
    let ctx = UnfoldContext::new(colored)?;

    // Phase 1: Unfold places.
    let pu = unfold_places(&ctx, colored, budget)?;

    // Phase 2: Unfold transitions.
    let tu = unfold_transitions(&ctx, colored, &pu, budget)?;

    let mut net = PetriNet {
        name: colored.name.clone(),
        places: pu.places,
        transitions: tu.transitions,
        initial_marking: pu.initial_marking,
    };
    // Merge any parallel arcs the unfolding produced (a colored inscription can
    // contribute the same (place,color) twice — e.g. via a subtract that leaves
    // overlapping multiset terms), so the unfolded P/T net has the same
    // is_enabled/apply_delta/DD-consistent single-arc-per-place form the native
    // parser now guarantees. No-op when the unfolder already emits one arc per
    // (place,color).
    net.canonicalize_parallel_arcs();

    let aliases = PropertyAliases {
        place_aliases: pu.place_aliases,
        transition_aliases: tu.transition_aliases,
        colored_place_group_aliases: pu.colored_place_group_aliases,
    };

    Ok(UnfoldedNet { net, aliases })
}

#[cfg(test)]
#[path = "../unfold_tests.rs"]
mod tests;
