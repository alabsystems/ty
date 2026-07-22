// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! StateSpace examination.
//!
//! Computes state space statistics: number of reachable markings,
//! number of explored transition edges, maximum tokens in any place,
//! and maximum token sum.

use crate::explorer::{
    CheckpointableObserver, ExplorationObserver, ParallelExplorationObserver,
    ParallelExplorationSummary,
};
use crate::petri_net::TransitionIdx;
use serde::{Deserialize, Serialize};
use tla_bignum::{BigUint, Zero};

/// Serde shim: a [`BigUint`] is (de)serialized as its decimal string. Avoids
/// enabling num-bigint's own serde feature surface; the checkpoint stays a
/// plain JSON string, round-tripping the exact value at any magnitude.
mod biguint_decimal_serde {
    use super::BigUint;
    use serde::{Deserialize, Deserializer, Serializer};

    pub(super) fn serialize<S: Serializer>(v: &BigUint, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(&v.to_str_radix(10))
    }

    pub(super) fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<BigUint, D::Error> {
        let s = String::deserialize(d)?;
        BigUint::parse_bytes(s.as_bytes(), 10)
            .ok_or_else(|| serde::de::Error::custom("invalid BigUint decimal string"))
    }
}

/// Observer that collects state space statistics during exploration.
///
/// `states_count` / `transition_edges` are arbitrary-precision (`BigUint`): the
/// orbit-quotient explorer weights each canonical representative by its orbit
/// size, so the SUM of orbit weights — the true `|R|` / `|E|` — can exceed
/// `u128` on a wide symmetric net even though each enumerated rep fits `usize`.
/// A `BigUint` accumulator carries that EXACT total so the cell is reported
/// instead of declining on the carrier; the value is unchanged on every net
/// whose count fits the old `usize`/`u64` carriers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct StateSpaceObserver {
    #[serde(with = "biguint_decimal_serde")]
    states_count: BigUint,
    #[serde(with = "biguint_decimal_serde")]
    transition_edges: BigUint,
    max_token_in_place: u64,
    max_token_sum: u64,
    #[serde(default)]
    overflowed: bool,
}

impl StateSpaceObserver {
    #[must_use]
    pub(crate) fn new(initial_marking: &[u64]) -> Self {
        let sum = checked_token_sum(initial_marking);
        let max_place = initial_marking.iter().copied().max().unwrap_or(0);
        Self {
            states_count: BigUint::zero(),
            transition_edges: BigUint::zero(),
            max_token_in_place: max_place,
            max_token_sum: sum.unwrap_or(0),
            overflowed: sum.is_none(),
        }
    }

    /// Returns the computed state space statistics.
    #[must_use]
    pub(crate) fn stats(&self) -> StateSpaceStats {
        StateSpaceStats {
            states: self.states_count.clone(),
            edges: self.transition_edges.clone(),
            max_token_in_place: self.max_token_in_place,
            max_token_sum: self.max_token_sum,
        }
    }
}

fn checked_token_sum(marking: &[u64]) -> Option<u64> {
    marking
        .iter()
        .try_fold(0u64, |acc, tokens| acc.checked_add(*tokens))
}

/// State space statistics computed during exploration.
#[derive(Debug, Clone)]
pub(crate) struct StateSpaceStats {
    /// Total number of unique reachable markings, EXACT as a [`BigUint`].
    ///
    /// Carries arbitrary precision so a structurally-computable count BEYOND
    /// `u128` (orbit-quotient Σ orbit sizes, disconnected-component product, or
    /// MDD-compact reachable set — e.g. FMS ≈1e47, Kanban/Philosophers ≈1e238)
    /// is REPORTED at full precision rather than declining on the carrier. The
    /// explicit BFS / DD / MDD lanes produce the SAME value as before on every
    /// net whose count fit the old `usize`/`u64`/`u128` carriers.
    pub(crate) states: BigUint,
    /// Total number of transition firings explored across all reachable states,
    /// EXACT as a [`BigUint`] (widened with [`Self::states`]).
    pub(crate) edges: BigUint,
    /// Maximum number of tokens seen in any single place.
    pub(crate) max_token_in_place: u64,
    /// Maximum sum of tokens across all places in any marking.
    pub(crate) max_token_sum: u64,
}

impl ExplorationObserver for StateSpaceObserver {
    fn on_new_state(&mut self, marking: &[u64]) -> bool {
        self.on_new_state_with_orbit(marking, 1)
    }

    fn on_new_state_with_orbit(&mut self, marking: &[u64], orbit_size: u64) -> bool {
        if self.overflowed {
            return false;
        }
        // The count accumulator is `BigUint`, so a u64 orbit weight adds
        // EXACTLY with no carrier overflow (the prior `usize` saturation that
        // could force CANNOT_COMPUTE on a large symmetric net is gone). A
        // single-orbit weight that exceeds `u64` is still fail-closed upstream
        // (BSGS `orbit_size` → `None` → `on_orbit_overflow`).
        self.states_count += BigUint::from(orbit_size);
        let Some(sum) = checked_token_sum(marking) else {
            self.overflowed = true;
            return false;
        };
        let max_place = marking.iter().copied().max().unwrap_or(0);
        if max_place > self.max_token_in_place {
            self.max_token_in_place = max_place;
        }
        if sum > self.max_token_sum {
            self.max_token_sum = sum;
        }
        true
    }

    fn on_transition_fire(&mut self, _trans: TransitionIdx) -> bool {
        self.on_transition_fire_with_orbit(&[], _trans, 1)
    }

    fn on_transition_fire_with_orbit(
        &mut self,
        _source: &[u64],
        _trans: TransitionIdx,
        orbit_size: u64,
    ) -> bool {
        if self.overflowed {
            return false;
        }
        self.transition_edges += BigUint::from(orbit_size);
        true
    }

    fn on_deadlock(&mut self, _marking: &[u64]) {}

    fn on_orbit_overflow(&mut self) {
        // An orbit-size computation overflowed u64. Fail closed: the exact
        // reachable-marking count can no longer be recovered from the quotient,
        // so mark overflowed and the examination reports CANNOT_COMPUTE rather
        // than a truncated wrong number.
        self.overflowed = true;
    }

    fn is_done(&self) -> bool {
        self.overflowed
    }
}

#[derive(Default)]
pub(crate) struct StateSpaceSummary {
    states_count: BigUint,
    transition_edges: BigUint,
    max_token_in_place: u64,
    max_token_sum: u64,
    overflowed: bool,
}

impl ParallelExplorationSummary for StateSpaceSummary {
    fn on_new_state(&mut self, marking: &[u64]) {
        self.on_new_state_with_orbit(marking, 1);
    }

    fn on_new_state_with_orbit(&mut self, marking: &[u64], orbit_size: u64) {
        if self.overflowed {
            return;
        }
        // `BigUint` accumulator: a u64 orbit weight adds exactly, no carrier
        // overflow (see `StateSpaceObserver::on_new_state_with_orbit`).
        self.states_count += BigUint::from(orbit_size);
        let Some(sum) = checked_token_sum(marking) else {
            self.overflowed = true;
            return;
        };
        let max_place = marking.iter().copied().max().unwrap_or(0);
        self.max_token_in_place = self.max_token_in_place.max(max_place);
        self.max_token_sum = self.max_token_sum.max(sum);
    }

    fn on_transition_fire(&mut self, _trans: TransitionIdx) {
        self.on_transition_fire_with_orbit(_trans, 1);
    }

    fn on_transition_fire_with_orbit(&mut self, _trans: TransitionIdx, orbit_size: u64) {
        if self.overflowed {
            return;
        }
        self.transition_edges += BigUint::from(orbit_size);
    }

    fn on_orbit_overflow(&mut self) {
        self.overflowed = true;
    }

    fn on_deadlock(&mut self, _marking: &[u64]) {}

    fn stop_requested(&self) -> bool {
        self.overflowed
    }
}

impl ParallelExplorationObserver for StateSpaceObserver {
    type Summary = StateSpaceSummary;

    fn new_summary(&self) -> Self::Summary {
        StateSpaceSummary::default()
    }

    fn merge_summary(&mut self, summary: Self::Summary) {
        if summary.overflowed {
            self.overflowed = true;
            return;
        }
        // EXACT bignum merge — never overflows the carrier.
        self.states_count += summary.states_count;
        self.transition_edges += summary.transition_edges;
        self.max_token_in_place = self.max_token_in_place.max(summary.max_token_in_place);
        self.max_token_sum = self.max_token_sum.max(summary.max_token_sum);
    }
}

impl CheckpointableObserver for StateSpaceObserver {
    type Snapshot = Self;

    const CHECKPOINT_KIND: &'static str = "StateSpaceObserver";

    fn snapshot(&self) -> Self::Snapshot {
        self.clone()
    }

    fn restore_from_snapshot(&mut self, snapshot: Self::Snapshot) {
        *self = snapshot;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn token_sum_overflow_requests_stop() {
        let mut observer = StateSpaceObserver::new(&[u64::MAX, 1]);

        assert!(observer.is_done());
        assert!(!observer.on_new_state(&[u64::MAX, 1]));
    }

    #[test]
    fn edge_count_past_u64_does_not_overflow_carrier() {
        // The edge accumulator is now `BigUint`: starting at u64::MAX and firing
        // once more no longer overflows / stops — it counts exactly to
        // u64::MAX + 1. (The old `u64` carrier set `overflowed` here.)
        let mut observer = StateSpaceObserver::new(&[0]);
        observer.transition_edges = BigUint::from(u64::MAX);
        assert!(observer.on_transition_fire(TransitionIdx(0)));
        assert!(!observer.is_done());
        assert_eq!(observer.stats().edges, BigUint::from(u64::MAX) + 1u32);
    }

    #[test]
    fn orbit_weighted_sum_exceeds_u128_exactly() {
        // The orbit-quotient explorer weights each rep by its orbit size. Here
        // we drive the observer directly with a large per-rep weight (u64::MAX)
        // repeated enough times that the running Σ exceeds u128::MAX — the exact
        // total the `BigUint` accumulator must carry without declining. The old
        // `usize`/`u64` carrier would have set `overflowed` (CANNOT_COMPUTE).
        let mut observer = StateSpaceObserver::new(&[0]);
        // u64::MAX × 2^65 > u128::MAX (since u64::MAX ≈ 2^64, product ≈ 2^129).
        let reps = 1u128 << 65;
        let mut expected = BigUint::zero();
        // Apply the weight `reps` times via direct accumulation (the observer
        // adds BigUint::from(orbit_size) per call; we mirror that here, then
        // assert one real call composes into the same exact running total).
        for _ in 0..3 {
            assert!(observer.on_new_state_with_orbit(&[0], u64::MAX));
            expected += BigUint::from(u64::MAX);
        }
        // Now jump the accumulator forward to the > u128 regime and add once
        // more through the real path, proving no carrier overflow.
        let big_base = BigUint::from(u64::MAX) * BigUint::from(reps);
        observer.states_count = big_base.clone();
        assert!(observer.on_new_state_with_orbit(&[0], 7));
        assert!(
            !observer.is_done(),
            "BigUint count never overflows the carrier"
        );
        assert_eq!(observer.stats().states, big_base + 7u32);
        assert!(
            observer.stats().states > BigUint::from(u128::MAX),
            "the orbit-weighted Σ is genuinely > u128::MAX and reported exactly",
        );
    }
}
