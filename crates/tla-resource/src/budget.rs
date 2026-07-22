// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Derived, three-valued memory budget — the single place the "how much is too
//! much" decision lives.
//!
//! A [`MemoryBudget`] is computed once per run from live platform memory (not
//! scattered magic constants). Its thresholds are explicit named inputs, so the
//! two engines with different working-set profiles — the tla-petri explicit
//! explorers (one self-ceiling) and the tla-check checker policy (warning +
//! critical tiers) — share the SAME decision function while keeping their own
//! tuning, instead of one shared magic constant that would over- or under-fit
//! both.

use std::sync::OnceLock;

use crate::platform;

/// Fraction of the effective machine/container size to keep free machine-wide.
const FLOOR_FRACTION: f64 = 0.18;

/// Absolute minimum collective free-memory floor when total RAM is unknown, and
/// the cap the fractional floor's minimum is clamped to half of (so a small
/// container is never handed a floor it can never clear).
const FLOOR_MIN: usize = 4 * 1024 * 1024 * 1024; // 4 GiB

/// Memory-pressure level. Three-valued because the checker distinguishes a
/// `Warning` (checkpoint if configured) from a `Critical` (graceful stop); the
/// explicit explorers collapse this to "Critical ⇒ decline".
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Pressure {
    /// Below the warning threshold — no action.
    Normal,
    /// At/above warning but below critical — checkpoint if configured.
    Warning,
    /// At/above critical, or the machine is collectively low — stop/decline.
    Critical,
}

/// A resource-aware memory budget. Every threshold is derived from live memory
/// or passed explicitly; there are no per-call-site magic numbers.
#[derive(Debug, Clone, Copy)]
pub struct MemoryBudget {
    /// Critical self-footprint threshold (bytes). `None` ⇒ no self ceiling.
    critical: Option<usize>,
    /// Warning self-footprint threshold (bytes), `< critical`. `None` ⇒ no
    /// warning tier (the explicit explorers, which have one ceiling only).
    warning: Option<usize>,
    /// Collective host-free floor (bytes): the machine is under pressure when
    /// host free memory drops below this.
    floor: usize,
    /// Footprint at/above which this process participates in the collective
    /// backoff. Below it, a process cannot free meaningful memory by declining
    /// and is not the pressure source, so it ignores the floor (prevents a
    /// lone/small checker on a busy host from declining before doing any work).
    /// `0` ⇒ always participate (the tla-petri explorer semantics).
    gate: usize,
}

/// The SINGLE source of truth for the symbolic-engine live-footprint ceiling
/// fraction. The tla-bdd and tla-mdd fixpoint engines both charge this fraction
/// of effective-available memory as their in-operation abort-probe budget, so
/// they back off at the SAME pressure — previously a `0.65` duplicated in each
/// crate behind a fragile "keep in sync" comment; now enforced by construction
/// via [`MemoryBudget::symbolic_explorer`].
pub const SYMBOLIC_MEMORY_FRACTION: f64 = 0.65;

impl MemoryBudget {
    /// The symbolic-engine explorer budget: [`Self::explorer`] at the shared
    /// [`SYMBOLIC_MEMORY_FRACTION`]. Both symbolic lanes (tla-bdd, tla-mdd) use
    /// this so their live-footprint back-off pressure is identical BY
    /// CONSTRUCTION (no cross-crate constant to drift).
    #[must_use]
    pub fn symbolic_explorer() -> Self {
        Self::explorer(SYMBOLIC_MEMORY_FRACTION)
    }

    /// Explicit-explorer budget: a single self-footprint ceiling at
    /// `fraction * effective_available`, plus the collective floor with no
    /// footprint gate (matches tla-petri's historical `exceeds_memory_budget`).
    #[must_use]
    pub fn explorer(fraction: f64) -> Self {
        Self {
            critical: platform::effective_available_bytes().map(|a| (a as f64 * fraction) as usize),
            warning: None,
            floor: collective_floor_bytes(),
            gate: 0,
        }
    }

    /// Checker-policy budget: `warn_frac`/`crit_frac` of an explicit `limit`,
    /// the collective floor, and a footprint `gate` below which the collective
    /// floor is ignored (matches tla-check's `MemoryPolicy`).
    #[must_use]
    pub fn checker(limit: usize, warn_frac: f64, crit_frac: f64, gate: usize) -> Self {
        Self {
            critical: Some((limit as f64 * crit_frac) as usize),
            warning: Some((limit as f64 * warn_frac) as usize),
            floor: collective_floor_bytes(),
            gate,
        }
    }

    /// Construct from explicit thresholds (used by callers with bespoke tuning
    /// and by tests). `warning`/`critical` are self-footprint byte thresholds.
    #[must_use]
    pub fn from_thresholds(
        critical: Option<usize>,
        warning: Option<usize>,
        floor: usize,
        gate: usize,
    ) -> Self {
        Self {
            critical,
            warning,
            floor,
            gate,
        }
    }

    /// The pure decision: pressure level for a measured `footprint`, given
    /// current `host_free` (the collective arm; `None` ⇒ probe failed, treat as
    /// "not low" — fail-soft). This is the one function every guard consults.
    #[must_use]
    pub fn pressure(&self, footprint: usize, host_free: Option<usize>) -> Pressure {
        let collectively_low = footprint >= self.gate && host_free.is_some_and(|f| f < self.floor);
        if collectively_low || self.critical.is_some_and(|c| footprint > c) {
            return Pressure::Critical;
        }
        if self.warning.is_some_and(|w| footprint > w) {
            return Pressure::Warning;
        }
        Pressure::Normal
    }

    /// Single-ceiling convenience for the explicit explorers: `true` ⇔ the
    /// budget is exceeded (pressure is `Critical`).
    #[must_use]
    pub fn over_budget(&self, footprint: usize, host_free: Option<usize>) -> bool {
        matches!(self.pressure(footprint, host_free), Pressure::Critical)
    }

    /// The critical self-footprint ceiling, if any — used by the adaptive probe
    /// to size its cadence from the remaining headroom.
    #[must_use]
    pub fn ceiling(&self) -> Option<usize> {
        self.critical
    }
}

/// Collective machine free-memory floor: a fraction of the effective
/// machine/container size ([`platform::effective_total_bytes`], cgroup-capped),
/// with the absolute minimum clamped to half the base so a small container is
/// never handed an unclearable floor. Cached in a `OnceLock` — total RAM is a
/// stable machine property and the live poll must not re-probe it.
#[must_use]
pub fn collective_floor_bytes() -> usize {
    static FLOOR: OnceLock<usize> = OnceLock::new();
    *FLOOR.get_or_init(|| {
        platform::effective_total_bytes()
            .map(|base| {
                let fractional = (base as f64 * FLOOR_FRACTION) as usize;
                let abs_min = FLOOR_MIN.min(base / 2);
                fractional.max(abs_min)
            })
            .unwrap_or(FLOOR_MIN)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn explorer_budget_single_ceiling_no_gate() {
        // ceiling 1000, floor 500, no warning, no gate.
        let b = MemoryBudget::from_thresholds(Some(1000), None, 500, 0);
        // Under ceiling, host healthy ⇒ Normal.
        assert_eq!(b.pressure(900, Some(10_000)), Pressure::Normal);
        // Over ceiling ⇒ Critical.
        assert_eq!(b.pressure(1001, Some(10_000)), Pressure::Critical);
        // Collective floor fires regardless of footprint (gate=0).
        assert_eq!(b.pressure(1, Some(400)), Pressure::Critical);
        // No warning tier for explorers.
        assert!(b.warning.is_none());
        assert!(b.over_budget(1001, Some(10_000)));
        assert!(!b.over_budget(900, Some(10_000)));
    }

    #[test]
    fn checker_budget_three_valued_with_gate() {
        // limit 1000, warn 0.70, crit 0.85, gate 600.
        let b = MemoryBudget::checker(1000, 0.70, 0.85, 600);
        assert_eq!(b.pressure(500, Some(usize::MAX)), Pressure::Normal);
        assert_eq!(b.pressure(750, Some(usize::MAX)), Pressure::Warning);
        assert_eq!(b.pressure(900, Some(usize::MAX)), Pressure::Critical);
    }

    #[test]
    fn footprint_gate_prevents_lone_small_process_decline() {
        // Machine collectively low (host_free 100 < floor 500), but this process
        // is small (footprint 50 < gate 600) ⇒ must NOT decline.
        let b = MemoryBudget::from_thresholds(Some(10_000), Some(7_000), 500, 600);
        assert_eq!(b.pressure(50, Some(100)), Pressure::Normal);
        // A genuinely large process on the same low machine DOES back off.
        assert_eq!(b.pressure(700, Some(100)), Pressure::Critical);
    }

    #[test]
    fn host_free_probe_failure_is_fail_soft() {
        let b = MemoryBudget::from_thresholds(Some(1000), None, 500, 0);
        // host_free = None must not trip the collective arm.
        assert_eq!(b.pressure(900, None), Pressure::Normal);
    }
}
