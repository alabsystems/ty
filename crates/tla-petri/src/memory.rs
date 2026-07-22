// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! System memory detection for dynamic exploration sizing.
//!
//! Queries available system memory to compute optimal `max_states` based on
//! per-state memory footprint. This converts many `CANNOT_COMPUTE` results
//! into correct answers by allowing the explorer to use available RAM rather
//! than a hardcoded 10M state limit.

use crate::marking::TokenWidth;
use std::mem::size_of;

/// Bytes per state when using fingerprint-based dedup.
///
/// 16 bytes for the u128 fingerprint + 16 bytes for FxHashSet entry overhead.
pub(crate) const FINGERPRINT_BYTES_PER_STATE: usize = 32;

/// `explore_full` stores the unpacked `Vec<u64>` marking for every state.
const FULL_GRAPH_MARKING_BYTES_PER_PLACE: usize = size_of::<u64>();

/// Base per-state heap/header overhead for full-graph CTL storage:
/// fingerprint registry entry + marking Vec header + adjacency Vec header.
const FULL_GRAPH_BASE_BYTES_PER_STATE: usize =
    FINGERPRINT_BYTES_PER_STATE + size_of::<Vec<u64>>() + size_of::<Vec<(u32, u32)>>();

/// Conservative amortized edge payload allowance per state.
///
/// A forward edge costs 8 bytes (`(u32, u32)` in `adj`) and the CTL lane
/// adds a predecessor CSR at 4 bytes per edge on top, so charge 12 bytes per
/// edge for an assumed average out-degree of 8 (audit 2026-07-02: the former
/// 2-edge/16-byte allowance under-sized dense narrow-marking nets; the live
/// footprint guard backstopped it, but the cap should be byte-honest).
const FULL_GRAPH_EDGE_BYTES_PER_STATE_ESTIMATE: usize = (size_of::<(u32, u32)>() + 4) * 8;

/// Default max_states when memory detection fails.
const FALLBACK_MAX_STATES: usize = 10_000_000;

/// Minimum max_states to preserve exploration of the initial marking.
const MIN_MAX_STATES: usize = 1;

/// Absolute primary in-memory fingerprint-set capacity for unbounded
/// disk-backed runs.
const PRIMARY_CAPACITY_ABSOLUTE_CEILING: usize = u32::MAX as usize;

/// Effective memory this process may address for state-space sizing: host free
/// memory capped by the cgroup limit and `BK_MEMORY_CONFINEMENT`. Delegates to
/// the single-source [`tla_resource::platform`] probes (the unsafe OS code
/// lives there now, not duplicated here). Returns `None` if detection fails.
pub(crate) fn available_memory_bytes() -> Option<usize> {
    tla_resource::platform::effective_available_bytes()
}

/// Fraction of available memory the explicit explorer may consume (resident
/// set) before it must stop and report the exploration incomplete
/// (`CANNOT_COMPUTE`) rather than risk an OOM kill.
///
/// The BFS queue and the collision-guard resident arena each retain a full
/// packed marking per state; on wide nets (tens of thousands of places) those
/// markings — *not* the 32-byte fingerprints the state budget is sized for —
/// dominate RAM. The fingerprint-only "unbounded" auto path runs with
/// `max_states = usize::MAX`, so the only existing in-loop guard is the
/// wall-clock deadline; this ceiling adds the symmetric memory backstop.
///
/// Conservative (0.65) so the ~35% headroom absorbs a single expansion's worth
/// of successor pushes (a wide net's initial marking can enable tens of
/// thousands of transitions, each adding a ~100+ KB marking to the queue *and*
/// the collision-guard arena before the next RSS poll), the pre-allocated
/// fingerprint table, allocator slack, and OS overhead. Measured: AirplaneLD-
/// PT-4000 under a 16 GB confinement peaks ~13.9 GB at 0.75; 0.65 keeps the
/// peak comfortably clear of the cap.
pub(crate) const EXPLORER_MEMORY_GUARD_FRACTION: f64 = 0.65;

/// An adaptive memory/deadline probe for the explicit explorers — the ONE
/// replacement for the former scattered `explorer_memory_ceiling_bytes()` +
/// per-loop `counter.is_multiple_of(N) && exceeds_memory_budget()` idiom.
///
/// The probe owns the derived budget (self-footprint ceiling at
/// [`EXPLORER_MEMORY_GUARD_FRACTION`] of effective-available memory, plus the
/// collective free-memory floor) and self-tunes its poll cadence to the loop's
/// speed and remaining headroom. Call [`tla_resource::MemoryProbe::over_budget`]
/// once per loop iteration and map `true` to the loop's decline value. Carry
/// the wall-clock `deadline` so both resources are checked on one cadence.
pub(crate) fn explorer_probe(deadline: Option<std::time::Instant>) -> tla_resource::MemoryProbe {
    tla_resource::MemoryProbe::new(explorer_budget(), deadline)
}

/// The explorer's derived memory budget (self-footprint ceiling at
/// [`EXPLORER_MEMORY_GUARD_FRACTION`] of effective-available memory + the
/// collective free-memory floor), derived from available memory AT CALL TIME.
///
/// Compute this ONCE per exploration and reuse it for every probe — the ceiling
/// must be pinned to the memory available at the start (like the former
/// one-shot `explorer_memory_ceiling_bytes()`), NOT re-derived from the
/// now-shrunken free memory a running exploration has itself consumed (that
/// would collapse the ceiling and trip prematurely). The single-loop explorers
/// get this for free via one [`explorer_probe`] before their loop; the parallel
/// fingerprint explorer, which spawns fresh per-level workers, must hoist this
/// out of the level loop and hand each worker a clone.
pub(crate) fn explorer_budget() -> tla_resource::MemoryBudget {
    tla_resource::MemoryBudget::explorer(EXPLORER_MEMORY_GUARD_FRACTION)
}

/// Compute optimal max_states from available memory and per-state cost.
///
/// Each explored state consumes:
/// - `num_places * width.bytes()` for the packed marking (hash key)
/// - ~48 bytes overhead per state (Box, FxHashSet entry, queue amortization)
///
/// The `memory_fraction` controls what fraction of available memory to budget
/// for state storage (default 0.25 — conservative to leave room for queue,
/// graph adjacency lists, and marking vectors in `explore_full`).
///
/// When `fingerprint_dedup` is true, uses 32 bytes per state (u128 hash +
/// overhead) instead of `packed_size + 48`.
///
/// Returns `FALLBACK_MAX_STATES` (10M) if memory detection fails.
pub(crate) fn compute_max_states(
    num_places: usize,
    width: TokenWidth,
    memory_fraction: f64,
    fingerprint_dedup: bool,
) -> usize {
    compute_max_states_for_packed_bytes(
        num_places.saturating_mul(width.bytes()),
        memory_fraction,
        fingerprint_dedup,
    )
}

pub(crate) fn compute_max_states_for_packed_bytes(
    packed_bytes: usize,
    memory_fraction: f64,
    fingerprint_dedup: bool,
) -> usize {
    let available = match available_memory_bytes() {
        Some(bytes) => bytes,
        None => return FALLBACK_MAX_STATES,
    };

    compute_max_states_from_available_memory(
        available,
        packed_bytes,
        memory_fraction,
        fingerprint_dedup,
    )
}

/// Compute optimal `max_states` for full-graph CTL/LTL exploration.
///
/// Unlike plain exploration, `explore_full` stores both the packed frontier
/// state and the unpacked `Vec<u64>` marking for every state, plus adjacency.
/// Auto-sized CTL runs therefore need a tighter budget than the fingerprint-
/// only sizing used by observer-mode exploration.
pub(crate) fn compute_max_states_for_full_graph(
    num_places: usize,
    packed_places: usize,
    width: TokenWidth,
    memory_fraction: f64,
) -> usize {
    compute_max_states_for_full_graph_packed_bytes(
        num_places,
        packed_places.saturating_mul(width.bytes()),
        memory_fraction,
    )
}

pub(crate) fn compute_max_states_for_full_graph_packed_bytes(
    num_places: usize,
    packed_bytes: usize,
    memory_fraction: f64,
) -> usize {
    let available = match available_memory_bytes() {
        Some(bytes) => bytes,
        None => return FALLBACK_MAX_STATES,
    };

    compute_max_states_for_full_graph_from_available_memory(
        available,
        num_places,
        packed_bytes,
        memory_fraction,
    )
}

fn compute_max_states_from_available_memory(
    available: usize,
    packed_bytes: usize,
    memory_fraction: f64,
    fingerprint_dedup: bool,
) -> usize {
    let fraction = sanitize_memory_fraction(memory_fraction);
    let budget = (available as f64 * fraction) as usize;
    let bytes_per_state = if fingerprint_dedup {
        // Fingerprint mode: u128 (16) + hash entry overhead (~16) — PLUS the
        // packed marking that the explorers on this sizing path actually
        // retain per admitted state (the fingerprint-only explorer's resident
        // collision-guard arena, and the observer/checkpoint BFS queues).
        // Sizing to bare fingerprints was a ~packed_bytes/32× underestimate
        // on wide nets (audit 2026-07-02); the live footprint ceiling
        // backstopped it, but the cap should be byte-honest.
        FINGERPRINT_BYTES_PER_STATE
            .saturating_add(packed_bytes)
            .max(1)
    } else {
        // Full marking mode: Box<[u8]> (16) + hash entry (~16) + queue (~16)
        packed_bytes.saturating_add(48).max(1)
    };

    let max = budget / bytes_per_state;

    // Clamp: at least the initial marking, at most u32::MAX (ReachabilityGraph uses u32 IDs)
    max.clamp(MIN_MAX_STATES, u32::MAX as usize)
}

fn compute_max_states_for_full_graph_from_available_memory(
    available: usize,
    num_places: usize,
    packed_bytes: usize,
    memory_fraction: f64,
) -> usize {
    let fraction = sanitize_memory_fraction(memory_fraction);
    let budget = (available as f64 * fraction) as usize;
    let marking_bytes = num_places.saturating_mul(FULL_GRAPH_MARKING_BYTES_PER_PLACE);
    let bytes_per_state = FULL_GRAPH_BASE_BYTES_PER_STATE
        .saturating_add(FULL_GRAPH_EDGE_BYTES_PER_STATE_ESTIMATE)
        .saturating_add(packed_bytes)
        .saturating_add(marking_bytes)
        .max(1);
    let max = budget / bytes_per_state;

    max.clamp(MIN_MAX_STATES, u32::MAX as usize)
}

fn sanitize_memory_fraction(memory_fraction: f64) -> f64 {
    if !memory_fraction.is_finite() {
        return 0.0;
    }
    memory_fraction.clamp(0.0, 1.0)
}

fn cap_primary_capacity_for_unbounded(requested: usize) -> usize {
    requested
        .max(MIN_MAX_STATES)
        .min(PRIMARY_CAPACITY_ABSOLUTE_CEILING)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_max_states_small_markings() {
        // With fingerprint: 32 bytes/state, 4GB budget → ~134M states
        let max = compute_max_states(50, TokenWidth::U8, 0.25, true);
        // Should always explore at least the initial marking.
        assert!(max >= MIN_MAX_STATES);
        // On any machine with ≥1GB RAM, should exceed fallback 10M
        if available_memory_bytes().unwrap_or(0) >= 4_000_000_000 {
            assert!(max > FALLBACK_MAX_STATES);
        }
    }

    #[test]
    fn test_compute_max_states_large_markings() {
        // 1000 places × 8 bytes (U64) = 8000 bytes + 48 overhead = 8048 bytes/state (non-fingerprint)
        // Even with large markings, should return at least the initial marking.
        let max = compute_max_states(1000, TokenWidth::U64, 0.25, false);
        assert!(max >= MIN_MAX_STATES);
    }

    #[test]
    fn test_compute_max_states_zero_places() {
        // Edge case: no places means 48 bytes overhead only (non-fingerprint)
        let max = compute_max_states(0, TokenWidth::U8, 0.25, false);
        assert!(max >= MIN_MAX_STATES);
    }

    #[test]
    fn test_compute_max_states_respects_low_memory_budget() {
        let max = compute_max_states_from_available_memory(
            64 * 1024 * 1024,
            1000 * TokenWidth::U64.bytes(),
            0.25,
            false,
        );
        assert_eq!(max, 2084);
    }

    #[test]
    fn test_compute_max_states_fingerprint_increases_budget() {
        // Same memory, fingerprint mode gives more states because 32 bytes/state
        // instead of 8048 bytes/state (1000 places × U64)
        let max_full = compute_max_states_from_available_memory(
            64 * 1024 * 1024,
            1000 * TokenWidth::U64.bytes(),
            0.25,
            false,
        );
        let max_fp = compute_max_states_from_available_memory(
            64 * 1024 * 1024,
            1000 * TokenWidth::U64.bytes(),
            0.25,
            true,
        );
        assert!(
            max_fp > max_full,
            "fingerprint mode should allow more states: {max_fp} vs {max_full}"
        );
    }

    #[test]
    fn test_compute_max_states_for_full_graph_accounts_for_markings_and_edges() {
        let max = compute_max_states_for_full_graph_from_available_memory(
            64 * 1024 * 1024,
            1_000,
            1_000 * TokenWidth::U64.bytes(),
            0.25,
        );
        // budget = 16 MiB; bytes/state = 80 (base) + 96 (8 edges × 12 B,
        // audit 2026-07-02) + 8000 (packed) + 8000 (marking) = 16176.
        assert_eq!(max, 16 * 1024 * 1024 / 16_176);
        assert_eq!(max, 1_037);
    }

    #[test]
    fn test_compute_max_states_for_full_graph_is_tighter_than_fingerprint_budget() {
        let max_fp = compute_max_states_from_available_memory(
            64 * 1024 * 1024,
            1_000 * TokenWidth::U64.bytes(),
            0.25,
            true,
        );
        let max_full = compute_max_states_for_full_graph_from_available_memory(
            64 * 1024 * 1024,
            1_000,
            1_000 * TokenWidth::U64.bytes(),
            0.25,
        );
        assert!(
            max_full < max_fp,
            "full-graph budget should be tighter than fingerprint-only sizing: \
             {max_full} vs {max_fp}"
        );
    }

    #[test]
    fn test_compute_max_states_zero_fraction_keeps_initial_state_only() {
        let max = compute_max_states_from_available_memory(
            16 * 1024 * 1024,
            50 * TokenWidth::U8.bytes(),
            0.0,
            false,
        );
        assert_eq!(max, 1);
    }

    #[test]
    fn test_compute_max_states_clamped_to_u32_max() {
        // Even with tiny states and huge memory, should not exceed u32::MAX
        let max = compute_max_states_from_available_memory(
            usize::MAX,
            TokenWidth::U8.bytes(),
            1.0,
            false,
        );
        assert!(u32::try_from(max).is_ok());
    }

    #[test]
    fn test_available_memory_returns_some() {
        // On supported platforms (Linux/macOS), this should succeed
        let mem = available_memory_bytes();
        if cfg!(any(target_os = "linux", target_os = "macos")) {
            assert!(
                mem.is_some(),
                "memory detection should work on this platform"
            );
            assert!(mem.unwrap() > 0);
        }
    }

    #[test]
    fn test_sanitize_memory_fraction_nan_returns_zero() {
        assert_eq!(sanitize_memory_fraction(f64::NAN), 0.0);
    }

    #[test]
    fn test_sanitize_memory_fraction_infinity_clamped() {
        assert_eq!(sanitize_memory_fraction(f64::INFINITY), 0.0);
        assert_eq!(sanitize_memory_fraction(f64::NEG_INFINITY), 0.0);
    }

    #[test]
    fn test_sanitize_memory_fraction_negative_clamped_to_zero() {
        assert_eq!(sanitize_memory_fraction(-0.5), 0.0);
    }

    #[test]
    fn test_sanitize_memory_fraction_above_one_clamped() {
        assert_eq!(sanitize_memory_fraction(2.0), 1.0);
    }

    #[test]
    fn test_sanitize_memory_fraction_valid_passthrough() {
        assert_eq!(sanitize_memory_fraction(0.5), 0.5);
        assert_eq!(sanitize_memory_fraction(0.0), 0.0);
        assert_eq!(sanitize_memory_fraction(1.0), 1.0);
    }

    #[test]
    fn test_compute_max_states_nan_fraction_returns_initial_only() {
        // NaN fraction → sanitize returns 0.0 → budget = 0 → clamp to MIN_MAX_STATES
        let max = compute_max_states_from_available_memory(
            16 * 1024 * 1024,
            50 * TokenWidth::U8.bytes(),
            f64::NAN,
            false,
        );
        assert_eq!(max, MIN_MAX_STATES);
    }

    #[test]
    fn test_compute_max_states_negative_fraction_returns_initial_only() {
        let max = compute_max_states_from_available_memory(
            16 * 1024 * 1024,
            50 * TokenWidth::U8.bytes(),
            -1.0,
            false,
        );
        assert_eq!(max, MIN_MAX_STATES);
    }

    #[test]
    fn test_cap_primary_capacity_for_unbounded_respects_absolute_ceiling() {
        // Requesting more than the absolute ceiling clamps to the ceiling.
        let capped = cap_primary_capacity_for_unbounded(usize::MAX);
        assert!(
            capped <= PRIMARY_CAPACITY_ABSOLUTE_CEILING,
            "unbounded request must not exceed the absolute ceiling: {capped}"
        );
        assert!(capped >= MIN_MAX_STATES);
    }

    #[test]
    fn test_cap_primary_capacity_for_unbounded_preserves_small_requests() {
        // Small requests (well below both ceilings) pass through unchanged.
        let capped = cap_primary_capacity_for_unbounded(1024);
        assert_eq!(capped, 1024);
    }

    #[test]
    fn test_cap_primary_capacity_for_unbounded_always_returns_at_least_one() {
        // Zero collapses to MIN_MAX_STATES so the initial marking always fits.
        let capped = cap_primary_capacity_for_unbounded(0);
        assert_eq!(capped, MIN_MAX_STATES);
    }
}
