// Licensed under the Apache License, Version 2.0

// Licensed under the Apache License, Version 2.0

// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// Licensed under the Apache License, Version 2.0

//! BFS submodule: storage-mode abstractions and exploration engine.
//!
//! Part of #2351: boundary-aligned split from `run_bfs_common.rs`.
//! Separates storage-mode contracts (trait + implementations) from the
//! BFS exploration algorithm, mirroring TLC's Worker/IStateQueue split.

mod checkpoint_view;
pub(crate) mod core_step;
// core_step_seq removed in #2963: both diff and full-state paths now do
// inline admit → invariant → enqueue (no need for SequentialBfsAdapter
// or run_core_step_batch).
mod diff_successors;
mod diff_successors_streaming;
pub(crate) mod engine;
mod full_state_successors;
mod full_state_successors_streaming;
mod iter_state;
mod observer;
pub(in crate::check::model_checker) mod storage_modes;
mod successor_processing;
pub(crate) mod transport;
pub(in crate::check::model_checker) mod transport_seq;
pub(crate) mod worker_loop;
// Dialect trace bridge: emits `tla_dialect::verif::VerifBfsStep` ops when
// `TY_DIALECT_TRACE=1` is set. Part of #4253 Wave 14 consumer wiring.
pub(crate) mod dialect_trace;
// Arena-backed BFS frontier for FlatState buffers (Part of #4126)
pub(in crate::check::model_checker) mod flat_frontier;
// Compact witness-backed frontier for fingerprint-only ArrayState BFS.
pub(in crate::check::model_checker) mod witness_frontier;
// Compiled BFS level loop for JIT-compiled frontier processing (Part of #3988)
mod compiled_bfs_loop;
// Backend-agnostic compiled BFS step/level traits (Part of #4171 / #4267 Stage 2d)
pub(in crate::check::model_checker) mod compiled_step_trait;
// Frontier sampling for Cooperative Dual-Engine Model Checking (Part of #3768)
#[cfg(feature = "ay")]
pub(crate) mod frontier_sampler;
pub mod topology;

use crate::var_index::VarRegistry;

/// Release the ModelChecker-owned compiled-flat collision-witness arena only
/// after a proven, unbounded frontier exhaustion.
///
/// This helper is shared by compiled BFS and compiled-to-standard fallback so
/// limits and portfolio exits cannot drift between the two terminal paths.
pub(super) fn release_compiled_payload_witnesses_after_terminal_bfs(
    witnesses: &mut crate::storage::FingerprintPayloadWitnesses,
    frontier_exhausted: bool,
    limit_reached: Option<super::LimitType>,
) -> bool {
    if frontier_exhausted && limit_reached.is_none() {
        witnesses.release_storage();
        true
    } else {
        false
    }
}

/// Value parameters shared by all BFS successor-processing functions.
///
/// Bundles the read-only depth/level/registry arguments that are threaded
/// identically through `process_{diff,full_state}_successors{,_streaming}`,
/// reducing their argument count from 9 to 6.
pub(super) struct BfsStepParams<'a> {
    pub registry: &'a VarRegistry,
    pub current_depth: usize,
    pub succ_depth: usize,
    /// TLC level of the current (parent) state: `depth_to_tlc_level(current_depth)`.
    /// ACTION_CONSTRAINT expressions that reference `TLCGet("level")` should see
    /// this value, not `succ_level`.  Part of #1281.
    pub current_level: u32,
    pub succ_level: u32,
}

// Re-export storage types for sibling modules (run_bfs_full, run_bfs_notrace, run_resume).
pub(in crate::check::model_checker) use self::storage_modes::{
    FingerprintOnlyStorage, FullStateStorage, NoTraceQueueEntry,
};
pub(in crate::check::model_checker) use self::worker_loop::BfsLoopOutcome;

#[cfg(test)]
mod release_tests {
    use super::*;
    use crate::state::Fingerprint;
    use crate::storage::FingerprintPayloadWitnesses;

    fn seeded_witnesses() -> FingerprintPayloadWitnesses {
        let mut witnesses = FingerprintPayloadWitnesses::new();
        for n in 0..128 {
            witnesses.record_flat_i64_slots_if_absent(Fingerprint(n), &[n as i64]);
        }
        assert!(witnesses.estimated_memory_bytes() > 0);
        witnesses
    }

    #[test]
    fn normal_terminal_completion_releases_compiled_payload_witnesses() {
        let fresh_census = FingerprintPayloadWitnesses::new().census();
        let fresh_bytes = FingerprintPayloadWitnesses::new().estimated_memory_bytes();
        let mut witnesses = seeded_witnesses();

        assert!(release_compiled_payload_witnesses_after_terminal_bfs(
            &mut witnesses,
            true,
            None,
        ));

        assert_eq!(witnesses.census(), fresh_census);
        assert_eq!(witnesses.estimated_memory_bytes(), fresh_bytes);
    }

    #[test]
    fn truncated_or_early_completion_retains_compiled_payload_witnesses() {
        for (frontier_exhausted, limit_reached) in [
            (true, Some(super::super::LimitType::Depth)),
            (true, Some(super::super::LimitType::States)),
            (false, None),
        ] {
            let mut witnesses = seeded_witnesses();
            let retained_bytes = witnesses.estimated_memory_bytes();

            assert!(!release_compiled_payload_witnesses_after_terminal_bfs(
                &mut witnesses,
                frontier_exhausted,
                limit_reached,
            ));

            assert_eq!(
                witnesses.confirm_flat_i64_slots(Fingerprint(7), &[7]),
                Some(true)
            );
            assert_eq!(witnesses.estimated_memory_bytes(), retained_bytes);
        }
    }
}
