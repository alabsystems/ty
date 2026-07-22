// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Compiled BFS level loop integration for the model checker.
//!
//! When the compiled BFS step is available (all actions AND all invariants
//! JIT-compiled, fully-flat state layout), this module provides a level-based
//! BFS loop that processes entire frontiers from the contiguous `FlatBfsFrontier`
//! arena. Each parent state is fed through `CompiledBfsStep::step_flat()` which
//! performs action dispatch, inline fingerprinting, first-level dedup via
//! `AtomicFpSet`, and invariant checking in native compiled code.
//!
//! New successors are then passed through the model checker's global seen set
//! for second-level dedup and enqueued as `NoTraceQueueEntry::Flat` entries
//! into the frontier arena.
//!
//! # Design
//!
//! The compiled level loop replaces `run_bfs_loop` for specs that qualify:
//!
//! ```text
//! for each BFS level:
//!   get raw arena slice from FlatBfsFrontier
//!   for each parent in arena:
//!     compiled_step.step_flat(parent) -> FlatBfsStepOutput
//!     for each new successor:
//!       traditional_fingerprint via FlatBfsAdapter
//!       dedup against global seen set
//!       if new: enqueue as Flat entry + record trace
//!     handle invariant violations, deadlock
//!   advance read cursor (bulk consume)
//! ```
//!
//! Part of #3988: JIT V2 Phase 5 compiled BFS step.

use super::super::fingerprint::BfsFingerprintDomain;
use super::super::trace_invariant::TraceInvariantOutcome;
use super::super::{ArrayState, CheckResult, LimitType, ModelChecker, Trace};
use super::compiled_step_trait::{
    BfsStepError, CompiledBfsStepScratch, CompiledSuccessorParentIndices,
};
use super::flat_frontier::FlatBfsFrontier;
use super::storage_modes::{BfsStorage, FingerprintOnlyStorage};
use crate::check::model_checker::frontier::BfsFrontier;
use crate::shared_verdict::Verdict;
use crate::state::Fingerprint;
use crate::storage::{
    BatchInsertedIndexAdmission, FingerprintSet as ModelFingerprintSet, StorageFault,
    TraceLocationStorage,
};
use crate::RuntimeCheckError;
use tla_mc_core::{
    CheckerSourceKind, PreparedFingerprintAdmissionPlan, PreparedFingerprintPayloadWitnessKind,
    PreparedProgramPayloadKind, PreparedStorageKind, SetupTraceLaneKind, SharedCollisionPolicy,
    SharedDedupIdentity, SharedDedupScope, SharedDedupStorageKind, SharedDuplicateAuthorization,
    SharedFingerprintAlgorithm, SharedFingerprintIdentity, SharedFingerprintValueKind,
    ValidatedPreparedFingerprintAdmissionPlan,
};

/// Minimum number of parents between progress/memory pressure checks
/// during compiled BFS level processing. Matches the
/// `MEMORY_CHECK_INTERVAL` used by the standard BFS transport to ensure
/// OOM safety checks are not delayed until level-end.
///
/// Part of #4203.
const COMPILED_BFS_PROGRESS_INTERVAL: u64 = 4096;

/// 2026-07 OOM audit: cadence (in fused successors processed) for the live
/// memory poll INSIDE a fused compiled BFS level. The fused/native path
/// previously polled the memory policy only at LEVEL boundaries, making one
/// wide level an unbounded overshoot window (each admitted successor grows
/// the fingerprint store AND the flat frontier). Tunable: coarse enough that
/// the poll (one OS probe) is amortized over tens of thousands of successor
/// admissions, fine enough to bound the overshoot to tens of MB.
///
/// Must be a multiple of `COMPILED_BFS_PROGRESS_INTERVAL`: the poll
/// piggybacks on the existing per-successor cadence blocks.
const COMPILED_BFS_IN_LEVEL_MEMORY_CHECK_INTERVAL: u64 = 65_536;
const _: () = assert!(
    COMPILED_BFS_IN_LEVEL_MEMORY_CHECK_INTERVAL % COMPILED_BFS_PROGRESS_INTERVAL == 0,
    "in-level memory poll must fire on the progress-interval cadence"
);

const COMPILED_BFS_INTERPRETER_CROSSCHECK_ENV: &str = "TY_COMPILED_BFS_INTERPRETER_CROSSCHECK";

/// Default parent stride for the sampled fused-arena native↔interpreter equivalence check (P0).
const FUSED_NATIVE_EQUIV_SAMPLE_STRIDE: usize = 64;

/// Target number of parents to sample per crosschecked level (all of them on small levels).
const FUSED_NATIVE_EQUIV_TARGET_SAMPLES: usize = 16;

/// Crosscheck the fused arena only on the first N levels (the native CODE is level-invariant, so
/// early real states validate it), bounding the cost of the extra peek-arena run per check.
const FUSED_NATIVE_EQUIV_MAX_LEVELS: usize = 4;

/// Skip the crosscheck on any level wider than this, so the whole-frontier peek-arena run the
/// check performs stays cheap even if an early level is unexpectedly large.
const FUSED_NATIVE_EQUIV_MAX_LEVEL_PARENTS: usize = 8192;

/// Env override (in BYTES) for the transient successor-arena target when sub-batching
/// wide fused BFS levels. `0`/`off` disables sub-batching entirely.
const COMPILED_BFS_SUBBATCH_TARGET_ENV: &str = "TY_COMPILED_BFS_SUBBATCH_TARGET_BYTES";

/// Env override (in BYTES) for the predicted whole-level transient-arena size at/above
/// which sub-batching engages. Narrow levels stay a single native call.
const COMPILED_BFS_SUBBATCH_ENGAGE_ENV: &str = "TY_COMPILED_BFS_SUBBATCH_ENGAGE_BYTES";

/// Default sub-batch target: cap the transient native successor arena to ~this many
/// BYTES per fused call. The whole widest BFS level's arena (records × state slots × 8)
/// dominated peak RSS on wide-record specs (MCLamportMutex: ~255k records × 89 i64
/// slots ≈ 180MB). Reducing the parents fed to one native call shrinks the
/// predictively-sized arena proportionally; ~64MiB keeps the transient peak modest.
const COMPILED_BFS_SUBBATCH_TARGET_BYTES: usize = 64 * 1024 * 1024;

/// Default engage threshold: only sub-batch a level whose PREDICTED whole-level
/// transient arena is at least this many BYTES (≈ two+ sub-batches). Levels below this
/// run whole. Sizing in bytes (not records) makes the decision track real memory: a
/// spec with small per-record cost (few slots) needs a much wider level before its
/// arena crosses the bound, so narrow-record fused specs stay a single native call.
const COMPILED_BFS_SUBBATCH_ENGAGE_BYTES: usize = 96 * 1024 * 1024;

/// Read the sub-batch target-bytes config (env override, else the default).
/// A value of `0` (or `off`/`false`) disables sub-batching.
fn compiled_bfs_subbatch_target_bytes() -> usize {
    match std::env::var(COMPILED_BFS_SUBBATCH_TARGET_ENV) {
        Ok(v) => {
            let v = v.trim();
            if v.eq_ignore_ascii_case("off") || v.eq_ignore_ascii_case("false") {
                0
            } else {
                v.parse::<usize>()
                    .unwrap_or(COMPILED_BFS_SUBBATCH_TARGET_BYTES)
            }
        }
        Err(_) => COMPILED_BFS_SUBBATCH_TARGET_BYTES,
    }
}

/// Read the sub-batch engage-threshold config (env override, else the default).
fn compiled_bfs_subbatch_engage_bytes() -> usize {
    match std::env::var(COMPILED_BFS_SUBBATCH_ENGAGE_ENV) {
        Ok(v) => v
            .trim()
            .parse::<usize>()
            .unwrap_or(COMPILED_BFS_SUBBATCH_ENGAGE_BYTES),
        Err(_) => COMPILED_BFS_SUBBATCH_ENGAGE_BYTES,
    }
}

/// Count the leading run of frontier states that share `front_depth`, capped at `cap`.
///
/// BFS is strictly FIFO here (successors are appended to the tail, parents consumed
/// from the head), so every state at the current depth forms a contiguous prefix of the
/// remaining frontier ahead of any deeper successors already appended this level. A
/// sub-batch drawn from this run therefore never straddles a depth boundary, so the
/// successor depth (`first_depth + 1`) stays uniform and same-depth duplicate
/// tie-breaking is by ascending parent index exactly as the whole-level native call
/// would resolve it. Scans at most `cap + 1` metadata entries.
fn compiled_bfs_leading_same_depth_run(
    flat_queue: &FlatBfsFrontier,
    front_depth: usize,
    cap: usize,
) -> usize {
    let mut n = 0usize;
    while n < cap {
        match flat_queue.meta_at_offset(n) {
            Some((_, depth, _)) if depth == front_depth => n += 1,
            _ => break,
        }
    }
    n
}

#[derive(Debug)]
struct CompiledBfsInterpreterActionSuccessors {
    action_idx: usize,
    action_name: String,
    successor_arena: Vec<i64>,
    successor_count: usize,
    state_len: usize,
}

impl CompiledBfsInterpreterActionSuccessors {
    fn len(&self) -> usize {
        self.successor_count
    }

    fn iter_successors(&self) -> CompiledBfsInterpreterSuccessorIter<'_> {
        if self.state_len == 0 {
            return CompiledBfsInterpreterSuccessorIter::Empty(self.successor_count);
        }

        debug_assert_eq!(
            self.successor_arena.len(),
            self.successor_count * self.state_len,
            "interpreter crosscheck successor arena length mismatch",
        );
        CompiledBfsInterpreterSuccessorIter::Chunked(
            self.successor_arena.chunks_exact(self.state_len),
        )
    }
}

enum CompiledBfsInterpreterSuccessorIter<'a> {
    Chunked(std::slice::ChunksExact<'a, i64>),
    Empty(usize),
}

impl<'a> Iterator for CompiledBfsInterpreterSuccessorIter<'a> {
    type Item = &'a [i64];

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            CompiledBfsInterpreterSuccessorIter::Chunked(chunks) => chunks.next(),
            CompiledBfsInterpreterSuccessorIter::Empty(remaining) => {
                if *remaining == 0 {
                    None
                } else {
                    *remaining -= 1;
                    Some(&[])
                }
            }
        }
    }
}

fn fused_successor_needs_pre_seen_lookup(
    regular_invariants_checked_by_backend: bool,
    regular_invariant_count: usize,
    level_may_skip_global_pre_seen_lookup: bool,
) -> bool {
    if regular_invariant_count == 0 {
        return false;
    }
    if !regular_invariants_checked_by_backend {
        return true;
    }
    !level_may_skip_global_pre_seen_lookup
}

fn fused_successor_needs_rust_regular_invariant_check(
    regular_invariants_checked_by_backend: bool,
    regular_invariant_count: usize,
) -> bool {
    !regular_invariants_checked_by_backend && regular_invariant_count > 0
}

fn fused_successor_has_flat_frontier_admission_proof(
    flat_state_primary: bool,
    native_fused_flat_frontier_admission_active: bool,
) -> bool {
    flat_state_primary || native_fused_flat_frontier_admission_active
}

fn fused_level_may_skip_global_pre_seen_lookup(
    level_skip_global_pre_seen_lookup: bool,
    flat_state_primary: bool,
    native_fused_flat_frontier_admission_active: bool,
) -> bool {
    level_skip_global_pre_seen_lookup
        && fused_successor_has_flat_frontier_admission_proof(
            flat_state_primary,
            native_fused_flat_frontier_admission_active,
        )
}

// Backend fingerprint sidecars are a complete admission proof when the raw
// flat buffer is the primary state representation, or when strict native-fused
// flat-frontier admission has proved the non-primary compact layout safe.
fn fused_successor_trusts_backend_fingerprint_sidecars(
    flat_state_primary: bool,
    native_fused_flat_frontier_admission_active: bool,
) -> bool {
    fused_successor_has_flat_frontier_admission_proof(
        flat_state_primary,
        native_fused_flat_frontier_admission_active,
    )
}

fn compiled_fused_batch_fingerprint_identity(
    domain: BfsFingerprintDomain,
) -> (
    SharedFingerprintIdentity,
    PreparedFingerprintPayloadWitnessKind,
    &'static str,
) {
    match domain {
        BfsFingerprintDomain::CompiledFlat => (
            SharedFingerprintIdentity::new(
                "tla compiled-flat fingerprint-only state",
                SharedFingerprintAlgorithm::Xxh3U64,
                SharedFingerprintValueKind::State,
                "flat-i64-state-v1",
                "tla-compiled-flat-state",
                64,
            )
            .with_canonical_domain("tla-flat-i64-state", "v1")
            .with_seed_identity("xxh3-64"),
            PreparedFingerprintPayloadWitnessKind::CompiledFlatXxh3,
            "tla-check-compiled-flat-fingerprint-only-dyn-fingerprint-set-v1",
        ),
        BfsFingerprintDomain::View => (
            SharedFingerprintIdentity::new(
                "compiled fused BFS VIEW state",
                SharedFingerprintAlgorithm::TlaFingerprint64,
                SharedFingerprintValueKind::State,
                "tla-view-value-v1",
                "tla-view-state",
                64,
            )
            .with_canonical_domain("tla-view-value", "v1")
            .with_seed_identity("tlc-fp64"),
            PreparedFingerprintPayloadWitnessKind::TlaArrayFp64,
            "tla-check-view-fingerprint-only-dyn-fingerprint-set-v1",
        ),
        BfsFingerprintDomain::SymmetryCanonical => (
            SharedFingerprintIdentity::new(
                "compiled fused BFS symmetry state",
                SharedFingerprintAlgorithm::TlaFingerprint64,
                SharedFingerprintValueKind::State,
                "array-state-v1",
                "tla-symmetry-canonical-state",
                64,
            )
            .with_canonical_domain("tla-symmetry-canonical-array-state", "v1")
            .with_seed_identity("tlc-fp64"),
            PreparedFingerprintPayloadWitnessKind::TlaArrayFp64,
            "tla-check-symmetry-fingerprint-only-dyn-fingerprint-set-v1",
        ),
        // WP-11 slice 2: defensive-only — the compiled BFS loop is not
        // admitted for the flat-symmetry canonical domain while
        // `flat_symmetry_native_veto_relaxed()` is fail-closed (its successor
        // paths hash RAW buffers). The identity strings match the
        // state_tracking flat-symmetry dedup identity so admission stays
        // domain-exact if the native increment lands.
        BfsFingerprintDomain::FlatSymmetryCanonical => (
            SharedFingerprintIdentity::new(
                "compiled fused BFS flat-symmetry state",
                SharedFingerprintAlgorithm::Xxh3U64,
                SharedFingerprintValueKind::State,
                "flat-i64-state-v1",
                "tla-flat-symmetry-canonical-state",
                64,
            )
            .with_canonical_domain("tla-flat-symmetry-canonical-flat-i64-state", "v1")
            .with_seed_identity("xxh3-64"),
            PreparedFingerprintPayloadWitnessKind::CompiledFlatXxh3,
            "tla-check-flat-symmetry-fingerprint-only-dyn-fingerprint-set-v1",
        ),
        BfsFingerprintDomain::FullStateFp64 | BfsFingerprintDomain::ArrayFp64 => (
            SharedFingerprintIdentity::new(
                "compiled fused BFS TLA fp64 state",
                SharedFingerprintAlgorithm::TlaFingerprint64,
                SharedFingerprintValueKind::State,
                "array-state-v1",
                "tla-explicit-state",
                64,
            )
            .with_canonical_domain("tla-array-state", "v1")
            .with_seed_identity("tlc-fp64"),
            PreparedFingerprintPayloadWitnessKind::TlaArrayFp64,
            "tla-check-fingerprint-only-dyn-fingerprint-set-v1",
        ),
    }
}

fn compiled_fused_batch_admission_plan(
    domain: BfsFingerprintDomain,
) -> PreparedFingerprintAdmissionPlan {
    let (fingerprint, payload_witness, storage_config_identity) =
        compiled_fused_batch_fingerprint_identity(domain);
    let dedup = SharedDedupIdentity::new(
        format!(
            "compiled fused BFS {} state-space dedup",
            domain.diagnostic_name()
        ),
        fingerprint,
        SharedDedupScope::StateSpace,
        SharedDedupStorageKind::External,
        SetupTraceLaneKind::Fingerprint,
    )
    .with_collision_policy(SharedCollisionPolicy::CanonicalPayloadEquality)
    .with_storage_config_identity(storage_config_identity);

    PreparedFingerprintAdmissionPlan::new(
        format!(
            "compiled fused BFS {} state-space batch admission",
            domain.diagnostic_name()
        ),
        CheckerSourceKind::Tla,
        PreparedProgramPayloadKind::Tla,
        PreparedStorageKind::TlaStateSlots,
        SetupTraceLaneKind::Fingerprint,
        dedup,
        SharedDuplicateAuthorization::CanonicalPayloadEquality,
        payload_witness,
    )
}

#[derive(Debug)]
struct CompiledBfsLivenessSuccessors {
    storage: CompiledBfsLivenessSuccessorStorage,
}

#[derive(Debug)]
enum CompiledBfsLivenessSuccessorStorage {
    Runs {
        successors: Vec<Fingerprint>,
        run_parent_indices: Vec<usize>,
        run_lengths: Vec<usize>,
    },
    Sparse {
        edges: Vec<(usize, Fingerprint)>,
    },
}

impl CompiledBfsLivenessSuccessors {
    fn with_successor_capacity(successor_count: usize) -> Self {
        Self {
            storage: CompiledBfsLivenessSuccessorStorage::Runs {
                successors: Vec::with_capacity(successor_count),
                run_parent_indices: Vec::new(),
                run_lengths: Vec::new(),
            },
        }
    }

    fn push(&mut self, parent_idx: usize, successor_fp: Fingerprint) {
        let out_of_order = match &self.storage {
            CompiledBfsLivenessSuccessorStorage::Runs {
                run_parent_indices, ..
            } => run_parent_indices
                .last()
                .is_some_and(|last_parent_idx| parent_idx < *last_parent_idx),
            CompiledBfsLivenessSuccessorStorage::Sparse { .. } => false,
        };
        if out_of_order {
            self.convert_runs_to_sparse();
        }

        match &mut self.storage {
            CompiledBfsLivenessSuccessorStorage::Runs {
                successors,
                run_parent_indices,
                run_lengths,
            } => {
                if run_parent_indices.last().copied() != Some(parent_idx) {
                    run_parent_indices.push(parent_idx);
                    run_lengths.push(0);
                }
                successors.push(successor_fp);
                *run_lengths
                    .last_mut()
                    .expect("liveness successor run exists after parent push") += 1;
            }
            CompiledBfsLivenessSuccessorStorage::Sparse { edges } => {
                edges.push((parent_idx, successor_fp));
            }
        }
    }

    fn try_for_each_parent_successors(
        self,
        parent_count: usize,
        mut f: impl FnMut(usize, Vec<Fingerprint>) -> Result<(), crate::CheckError>,
    ) -> Result<(), crate::CheckError> {
        match self.storage {
            CompiledBfsLivenessSuccessorStorage::Runs {
                successors,
                run_parent_indices,
                run_lengths,
            } => {
                let mut successor_iter = successors.into_iter();
                let mut run_iter = run_parent_indices.into_iter().zip(run_lengths).peekable();
                for parent_idx in 0..parent_count {
                    let successors = match run_iter.peek().copied() {
                        Some((run_parent_idx, _run_len)) if run_parent_idx < parent_idx => {
                            return Err(compiled_bfs_liveness_parent_index_error(
                                run_parent_idx,
                                parent_count,
                            ));
                        }
                        Some((run_parent_idx, run_len)) if run_parent_idx == parent_idx => {
                            run_iter.next();
                            successor_iter.by_ref().take(run_len).collect()
                        }
                        _ => Vec::new(),
                    };
                    f(parent_idx, successors)?;
                }
                if let Some((run_parent_idx, _)) = run_iter.next() {
                    return Err(compiled_bfs_liveness_parent_index_error(
                        run_parent_idx,
                        parent_count,
                    ));
                }
                Ok(())
            }
            CompiledBfsLivenessSuccessorStorage::Sparse { mut edges } => {
                edges.sort_unstable_by_key(|(parent_idx, _)| *parent_idx);
                let mut edge_iter = edges.into_iter().peekable();
                for parent_idx in 0..parent_count {
                    let mut successors = Vec::new();
                    while edge_iter
                        .peek()
                        .is_some_and(|(edge_parent_idx, _)| *edge_parent_idx == parent_idx)
                    {
                        let (_, successor_fp) = edge_iter
                            .next()
                            .expect("peeked liveness successor edge exists");
                        successors.push(successor_fp);
                    }
                    f(parent_idx, successors)?;
                }
                if let Some((edge_parent_idx, _)) = edge_iter.next() {
                    return Err(compiled_bfs_liveness_parent_index_error(
                        edge_parent_idx,
                        parent_count,
                    ));
                }
                Ok(())
            }
        }
    }

    fn convert_runs_to_sparse(&mut self) {
        let storage = std::mem::replace(
            &mut self.storage,
            CompiledBfsLivenessSuccessorStorage::Sparse { edges: Vec::new() },
        );
        match storage {
            CompiledBfsLivenessSuccessorStorage::Runs {
                successors,
                run_parent_indices,
                run_lengths,
            } => {
                let mut successor_iter = successors.into_iter();
                let mut edges = Vec::with_capacity(run_lengths.iter().sum());
                for (parent_idx, run_len) in run_parent_indices.into_iter().zip(run_lengths) {
                    edges.extend(
                        successor_iter
                            .by_ref()
                            .take(run_len)
                            .map(|successor_fp| (parent_idx, successor_fp)),
                    );
                }
                self.storage = CompiledBfsLivenessSuccessorStorage::Sparse { edges };
            }
            sparse @ CompiledBfsLivenessSuccessorStorage::Sparse { .. } => {
                self.storage = sparse;
            }
        }
    }

    #[cfg(test)]
    fn is_run_encoded_for_testing(&self) -> bool {
        matches!(
            self.storage,
            CompiledBfsLivenessSuccessorStorage::Runs { .. }
        )
    }

    #[cfg(test)]
    fn into_successor_entries_for_testing(
        self,
        parent_count: usize,
    ) -> Vec<(usize, Vec<Fingerprint>)> {
        let mut entries = Vec::new();
        self.try_for_each_parent_successors(parent_count, |parent_idx, successors| {
            entries.push((parent_idx, successors));
            Ok::<(), crate::CheckError>(())
        })
        .expect("test liveness successor capture should be valid");
        entries
    }
}

fn compiled_bfs_liveness_parent_index_error(
    parent_idx: usize,
    parent_count: usize,
) -> crate::CheckError {
    RuntimeCheckError::Internal(format!(
        "fused compiled BFS state-graph capture reported parent index {parent_idx} for \
         {parent_count} parents",
    ))
    .into()
}

/// Statistics from a compiled BFS level loop run.
#[derive(Debug, Default)]
pub(in crate::check::model_checker) struct CompiledBfsLoopStats {
    /// Wall-clock instant when compiled BFS execution starts, after cold JIT build/setup.
    pub(in crate::check::model_checker) execution_started_at: Option<std::time::Instant>,
    /// Total parents processed across all levels.
    pub(in crate::check::model_checker) parents_processed: u64,
    /// Total successors generated (before global dedup).
    pub(in crate::check::model_checker) successors_generated: u64,
    /// Total new successors after global dedup.
    pub(in crate::check::model_checker) successors_new: u64,
    /// BFS levels completed.
    pub(in crate::check::model_checker) levels_completed: usize,
    /// Per-parent compiled step outputs that borrowed reusable successor scratch.
    pub(in crate::check::model_checker) step_outputs_borrowed: u64,
    /// Per-parent compiled step outputs that owned a freshly materialized arena.
    pub(in crate::check::model_checker) step_outputs_owned: u64,
    /// Fused successors that still needed a pre-admission seen-set probe.
    pub(in crate::check::model_checker) fused_pre_seen_lookups: u64,
    /// Fused successors admitted through a single insert/test-and-set.
    pub(in crate::check::model_checker) fused_pre_seen_lookups_skipped: u64,
    /// Per-successor raw admission validations avoided after level-level checks.
    pub(in crate::check::model_checker) fused_admission_validations_elided: u64,
    /// Fused batch admissions routed through the prepared runtime callsite.
    pub(in crate::check::model_checker) fused_prepared_batch_admission_calls: u64,
    /// Prepared admission descriptor validations performed while binding the runtime handle.
    pub(in crate::check::model_checker) fused_prepared_batch_admission_descriptor_validations: u64,
    /// Hot descriptor validations performed by fused prepared batch admission.
    pub(in crate::check::model_checker) fused_prepared_batch_admission_hot_descriptor_validations:
        u64,
    /// Fingerprints presented through the prepared fused batch admission callsite.
    pub(in crate::check::model_checker) fused_prepared_batch_admission_fingerprints: u64,
    /// Duplicate authorization checks performed through the validated prepared handle.
    pub(in crate::check::model_checker) fused_prepared_batch_duplicate_authorization_checks: u64,
    /// Observability-only (un-darkening STEP 1): per-action runtime native-vs-interp
    /// wall-clock telemetry, populated ONLY when `TY_TRUST_CG_TELEMETRY` is set.
    /// When the flag is off these stay default and no clock is ever read, so a
    /// default run is byte-identical with zero timing overhead.
    pub(in crate::check::model_checker) telemetry_enabled: bool,
    /// Native fused-level callouts (`run_level_fused_arena`) that returned a
    /// result (whole-frontier native kernel invocations).
    pub(in crate::check::model_checker) native_fused_call_count: u64,
    /// Total wall-clock ns spent inside native fused-level callouts.
    pub(in crate::check::model_checker) native_fused_total_ns: u128,
    /// Native per-parent step callouts (`step_flat_scoped`).
    pub(in crate::check::model_checker) native_step_call_count: u64,
    /// Total wall-clock ns spent inside native per-parent step callouts.
    pub(in crate::check::model_checker) native_step_total_ns: u128,
}

struct CompiledFusedBatchAdmissionRuntime {
    handle: ValidatedPreparedFingerprintAdmissionPlan,
}

impl CompiledFusedBatchAdmissionRuntime {
    fn new(domain: BfsFingerprintDomain) -> Result<Self, StorageFault> {
        let plan = compiled_fused_batch_admission_plan(domain);
        let plan_id = plan.id.clone();
        let handle = plan.into_validated_runtime_handle().map_err(|rejection| {
            StorageFault::new(
                "compiled_bfs_prepared_batch_admission",
                "validate_runtime_handle",
                format!(
                    "status_code=rejected reason_code={} fail_closed=true plan_id={} \
                     fingerprint_domain={} detail={}",
                    rejection.reason_code,
                    plan_id,
                    domain.diagnostic_name(),
                    rejection.detail
                ),
            )
        })?;
        Ok(Self { handle })
    }

    fn record_setup_validation(&self, stats: &mut CompiledBfsLoopStats) {
        let evidence = self.handle.validation_evidence();
        stats.fused_prepared_batch_admission_descriptor_validations +=
            evidence.setup_descriptor_validation_count;
        stats.fused_prepared_batch_admission_hot_descriptor_validations +=
            evidence.hot_descriptor_validation_count;
    }

    fn insert_batch_fingerprint_values_inserted_indices_checked_into(
        &self,
        seen_fps: &dyn ModelFingerprintSet,
        fingerprint_values: &[u64],
        admission: &mut BatchInsertedIndexAdmission,
        stats: &mut CompiledBfsLoopStats,
    ) {
        stats.fused_prepared_batch_admission_calls += 1;
        stats.fused_prepared_batch_admission_fingerprints = stats
            .fused_prepared_batch_admission_fingerprints
            .checked_add(fingerprint_values.len() as u64)
            .expect("compiled BFS prepared batch admission fingerprint count overflow");
        seen_fps.insert_batch_fingerprint_values_inserted_indices_checked_into(
            fingerprint_values,
            admission,
        );
    }

    fn enforce_batch_duplicate_authorization_checked(
        &self,
        attempted: usize,
        inserted_count: usize,
        fault_present: bool,
        duplicate_payload_confirmed: bool,
        stats: &mut CompiledBfsLoopStats,
    ) -> Result<(), StorageFault> {
        if fault_present || inserted_count >= attempted {
            return Ok(());
        }
        stats.fused_prepared_batch_duplicate_authorization_checks += 1;
        self.handle
            .enforce_batch_duplicate_with_canonical_payload_comparison(
                attempted,
                inserted_count,
                fault_present,
                || Ok(duplicate_payload_confirmed),
            )
    }
}

fn compiled_fused_batch_admission_runtime<'a>(
    domain: BfsFingerprintDomain,
    runtime: &'a mut Option<CompiledFusedBatchAdmissionRuntime>,
    stats: &mut CompiledBfsLoopStats,
) -> Result<&'a CompiledFusedBatchAdmissionRuntime, StorageFault> {
    if runtime.is_none() {
        let initialized = CompiledFusedBatchAdmissionRuntime::new(domain)?;
        initialized.record_setup_validation(stats);
        *runtime = Some(initialized);
    }
    Ok(runtime
        .as_ref()
        .expect("compiled fused batch admission runtime initialized"))
}

fn cached_fingerprint_only_admission_handle(
    checker: &ModelChecker<'_>,
    handle: &mut Option<&'static ValidatedPreparedFingerprintAdmissionPlan>,
) -> Result<&'static ValidatedPreparedFingerprintAdmissionPlan, StorageFault> {
    if let Some(handle) = *handle {
        return Ok(handle);
    }
    let initialized = checker.fingerprint_only_prepared_admission_handle_for_current_domain()?;
    *handle = Some(initialized);
    Ok(initialized)
}

impl ModelChecker<'_> {
    /// 2026-07 OOM audit: live memory poll INSIDE a fused compiled BFS level.
    ///
    /// The fused path previously polled the memory policy only at level
    /// boundaries (the post-level check in the fused loop), so one wide level
    /// was an unbounded overshoot window. Called on the
    /// [`COMPILED_BFS_IN_LEVEL_MEMORY_CHECK_INTERVAL`] successor cadence;
    /// returns `true` when the policy reports `Critical`, in which case the
    /// caller performs the SAME graceful stop the level boundary uses
    /// (`LimitReached { Memory }` — fail-closed: an incomplete run, never a
    /// verdict). No policy configured or a failed probe returns `false`.
    fn compiled_bfs_in_level_memory_critical(&self, fused_succ_processed: u64) -> bool {
        let Some(ref policy) = self.exploration.memory_policy else {
            return false;
        };
        if policy.check() != crate::memory::MemoryPressure::Critical {
            return false;
        }
        let rss_mb = crate::memory::current_rss_bytes()
            .map(|b| b / (1024 * 1024))
            .unwrap_or(0);
        let limit_mb = policy.limit_bytes() / (1024 * 1024);
        eprintln!(
            "[compiled-bfs] memory critical ({rss_mb} MB / {limit_mb} MB limit) \
             after {fused_succ_processed} fused successors within level — stopping."
        );
        true
    }

    fn compiled_bfs_interpreter_action_successors_for_parent(
        &mut self,
        parent_slice: &[i64],
        registry: &tla_core::VarRegistry,
    ) -> Result<Vec<CompiledBfsInterpreterActionSuccessors>, CheckResult> {
        let parent_state = self.try_reconstruct_state_from_flat(parent_slice, registry)?;
        let actions = self.coverage.actions.clone();
        if actions.is_empty() {
            return Ok(Vec::new());
        }

        let flat_layout = self
            .flat_bfs_adapter
            .as_ref()
            .map(|adapter| adapter.layout().clone())
            .or_else(|| self.flat_state_layout.clone())
            .ok_or_else(|| {
                CheckResult::from_error(
                    RuntimeCheckError::Internal(
                        "compiled BFS interpreter crosscheck missing flat state layout".to_string(),
                    )
                    .into(),
                    self.stats.clone(),
                )
            })?;

        let next_name = self
            .trace
            .cached_resolved_next_name
            .as_deref()
            .or(self.config.next.as_deref())
            .ok_or_else(|| {
                CheckResult::from_error(
                    RuntimeCheckError::Internal(
                        "compiled BFS interpreter crosscheck missing resolved NEXT name"
                            .to_string(),
                    )
                    .into(),
                    self.stats.clone(),
                )
            })?;

        if !self.module.op_defs.contains_key(next_name) {
            return Err(CheckResult::from_error(
                RuntimeCheckError::Internal(format!(
                    "compiled BFS interpreter crosscheck could not find NEXT operator {next_name}"
                ))
                .into(),
                self.stats.clone(),
            ));
        }

        let current_arr = ArrayState::from_state(&parent_state, registry);
        let mut action_successors = Vec::with_capacity(actions.len());

        for (action_idx, action) in actions.iter().enumerate() {
            // SOUNDNESS: pass the run-stable `&action.expr` (Arc-shared with
            // `self.coverage.actions`) — never a per-state clone — to keep the
            // unified enumerator's pointer-keyed caches valid.
            let successors = crate::enumerate::enumerate_successors_body(
                &mut self.ctx,
                &action.expr,
                &parent_state,
                &self.module.vars,
            )
            .map_err(|error| {
                CheckResult::from_error(
                    RuntimeCheckError::Internal(format!(
                        "compiled BFS interpreter crosscheck failed while enumerating {}: {error}",
                        action.name
                    ))
                    .into(),
                    self.stats.clone(),
                )
            })?;

            let state_len = flat_layout.total_slots();
            let mut successor_arena =
                Vec::with_capacity(successors.len().saturating_mul(state_len));
            let mut successor_count = 0;
            for successor in successors {
                let succ_arr = ArrayState::from_state(&successor, registry);
                let state_ok = self
                    .check_state_constraints_array_uncached(&succ_arr)
                    .map_err(|error| CheckResult::from_error(error, self.stats.clone()))?;
                let action_ok = self
                    .check_action_constraints_array(&current_arr, &succ_arr)
                    .map_err(|error| CheckResult::from_error(error, self.stats.clone()))?;
                if state_ok && action_ok {
                    // Graceful flat-overflow handling: typed error (never
                    // panic) when the successor cannot be encoded in the
                    // fixed flat layout; the CLI retries without flat.
                    let succ_flat = crate::state::FlatState::try_from_array_state(
                        &succ_arr,
                        flat_layout.clone(),
                    )
                    .map_err(|err| {
                        CheckResult::from_error(
                            crate::CheckError::flat_layout_unsupported_value(err.to_string()),
                            self.stats.clone(),
                        )
                    })?;
                    debug_assert_eq!(
                        succ_flat.buffer().len(),
                        state_len,
                        "interpreter crosscheck flat successor width mismatch",
                    );
                    successor_arena.extend_from_slice(succ_flat.buffer());
                    successor_count += 1;
                }
            }
            action_successors.push(CompiledBfsInterpreterActionSuccessors {
                action_idx,
                action_name: action.name.clone(),
                successor_arena,
                successor_count,
                state_len,
            });
        }

        Ok(action_successors)
    }

    /// Run the compiled BFS level loop, processing flat frontiers through
    /// the compiled BFS step.
    ///
    /// This is the Phase 5 alternative to `run_bfs_loop` that uses the
    /// contiguous arena in `FlatBfsFrontier` and the `CompiledBfsStep` for
    /// native-code action dispatch + fingerprinting + invariant checking.
    ///
    /// Falls back to `run_bfs_loop` if any prerequisite is not met.
    ///
    /// Part of #3988: JIT V2 Phase 5 compiled BFS step.
    // The inner successor loop indexes into multiple parallel arrays
    // (parent_indices, batch_fp_values) under early-return control flow that
    // mutates outer-scope counters; iter::enumerate is not a clean fit.
    #[allow(clippy::needless_range_loop)]
    pub(in crate::check::model_checker) fn run_compiled_bfs_loop(
        &mut self,
        storage: &mut FingerprintOnlyStorage,
        flat_queue: &mut FlatBfsFrontier,
    ) -> CheckResult {
        // Native lane: on ANY exit, drain a sampled native↔interpreter crosscheck (if one was
        // captured this run) and build the KERNEL-CHECKED relation-inclusion certificate offline —
        // surfacing that the JIT's sampled successors are interpreter-reachable, kernel-proven.
        //
        // 2026-07 memory audit: OPT-IN via `TY_NATIVE_LANE_CERT=1`. Building this
        // single-sample certificate forces construction of the full Clean kernel
        // environment (`ck0_bridge::build_env` → prelude typechecking): ~37 MB of
        // retained heap and ~1 GB of allocation churn — 61% of ty's check-RSS
        // floor on a 1-state spec. The runtime native↔interpreter crosscheck
        // itself still runs unconditionally during BFS (this guard only upgrades
        // one already-validated sample to a kernel-proven witness and prints a
        // diagnostic); skipping it is verdict-neutral by construction.
        #[cfg(feature = "clean-cic")]
        struct NativeLaneCertGuard;
        #[cfg(feature = "clean-cic")]
        impl Drop for NativeLaneCertGuard {
            fn drop(&mut self) {
                if !crate::check::debug::native_lane_cert_enabled() {
                    return;
                }
                if let Some(cert) = crate::cleancic::take_native_lane_cert() {
                    eprintln!(
                        "[compiled-bfs] native lane KERNEL-CERTIFIED: a sampled JIT successor is \
                         interpreter-reachable, witnessed by a {}-byte Clean-kernel-checked CIC term",
                        cert.len()
                    );
                }
            }
        }
        #[cfg(feature = "clean-cic")]
        let _native_lane_cert_guard = NativeLaneCertGuard;

        // Validate prerequisites -- if any fail, fall back to standard loop.
        if !self.compiled_bfs_level_eligible() {
            if self.trust_cg_native_fused_strict_enabled() {
                return self.strict_native_fused_fallback_result(
                    "compiled BFS level is not eligible for this run",
                );
            }
            if !self.compiled_bfs_step_usable_for_current_graph_mode() {
                self.disable_compiled_bfs_for_standard_fallback();
            }
            return self.run_bfs_loop(storage, flat_queue);
        }

        // Part of #4215: Seal the fingerprint algorithm before compiled BFS processing.
        #[cfg(debug_assertions)]
        {
            self.fp_algorithm_sealed = true;
        }

        // Freeze the fingerprint domain (idempotent). In eager AUTO / forced
        // trust-cg mode this is the first BFS loop entered, so capture the
        // domain here; in the interpreter->compiled hot-swap it was already
        // frozen by the interpreter entry and stays put.
        self.freeze_bfs_fingerprint_domain();

        let registry = self.ctx.var_registry().clone();
        let bfs_fingerprint_domain = self.bfs_fingerprint_domain();
        // Non-native implied-action handling on the per-parent STEP path.
        //
        // When `compiled_bfs_step_evaluates_interpreter_implied_actions()` holds,
        // the STEP path generates successors natively but a non-native implied
        // action (e.g. `[][B!Next]_B!vars`) must still be evaluated per edge in
        // the interpreter. We clone the eval-implied-action list once here so
        // the STEP emit closure can call
        // `check_eval_implied_actions_for_transition` with `&mut self.ctx`
        // without a disjoint-borrow conflict against `self.compiled`. The clone
        // is small (one term for the INSTANCE-refinement case) and happens once
        // per loop invocation, never per state.
        let step_eval_implied_actions: Vec<crate::checker_ops::EvalImpliedActionTerm> =
            if self.compiled_bfs_step_evaluates_interpreter_implied_actions() {
                self.compiled.eval_implied_actions.clone()
            } else {
                Vec::new()
            };
        let step_eval_implied_actions_active = !step_eval_implied_actions.is_empty();
        if step_eval_implied_actions_active {
            eprintln!(
                "[compiled-bfs] per-parent STEP path will evaluate {} non-native implied action(s) \
                 per edge via the interpreter hook (native successor generation engaged)",
                step_eval_implied_actions.len(),
            );
        }
        // Observability-only (un-darkening STEP 1): read the telemetry flag ONCE
        // here. When off, every per-callout `Instant::now()` below is skipped, so
        // a default run pays zero clock overhead and is byte-identical.
        let mut stats = CompiledBfsLoopStats {
            telemetry_enabled: super::super::trust_cg_dispatch::trust_cg_runtime_telemetry_enabled(
            ),
            ..Default::default()
        };
        let mut compiled_fused_batch_admission = None;
        let mut fingerprint_only_admission_handle = None;
        let mut limit_reached: Option<LimitType> = None;
        // Set only by the explicit empty-frontier branch below. Portfolio,
        // cooperative, defensive, and limit exits must not be inferred as
        // exhaustion from the queue length observed after the loop.
        let mut frontier_exhausted = false;
        let mut raw_inserted_index_admission = BatchInsertedIndexAdmission::default();
        let mut raw_inserted_successors = Vec::new();

        // Initialize eval arena for sequential BFS.
        tla_eval::eval_arena::init_thread_arena();
        crate::arena::init_worker_arena();

        let mut use_fused = self.fused_bfs_level_available();
        // Soundness guard for non-native implied actions: when an implied
        // action requires interpreter per-transition evaluation, only the
        // per-parent STEP path (which surfaces every raw edge before dedup) may
        // run — the fused LEVEL path may locally dedup successors and hide
        // duplicate edges from `check_eval_implied_actions_for_transition`.
        // `compiled_bfs_step_evaluates_interpreter_implied_actions()` already
        // requires `compiled_bfs_level.is_none()`, so `use_fused` should be
        // false here; if it is not, fail closed to the interpreter loop rather
        // than risk an unchecked implied-action edge.
        if use_fused && self.compiled_bfs_step_evaluates_interpreter_implied_actions() {
            eprintln!(
                "[compiled-bfs] fused level present alongside interpreter-evaluated implied actions; \
                 falling back to interpreter loop to keep every edge checked"
            );
            self.disable_compiled_bfs_for_standard_fallback();
            return self.run_bfs_loop(storage, flat_queue);
        }
        if self.trust_cg_native_fused_strict_enabled() && !self.native_fused_bfs_level_available() {
            return self
                .strict_native_fused_fallback_result("native fused BFS level is unavailable");
        }
        if flat_queue.has_fallback_entries() || flat_queue.remaining_flat_count() == 0 {
            eprintln!("[compiled-bfs] compiled BFS disabled: frontier has no flat parents ready");
            if self.trust_cg_native_fused_strict_enabled() {
                return self
                    .strict_native_fused_fallback_result("frontier has no flat parents ready");
            }
            if !self.compiled_bfs_step_usable_for_current_graph_mode() {
                self.disable_compiled_bfs_for_standard_fallback();
            }
            return self.run_bfs_loop(storage, flat_queue);
        }
        if use_fused {
            if let Some(ref level) = self.compiled_bfs_level {
                let parent_count = flat_queue.remaining_flat_count();
                let preflight_result = match flat_queue.remaining_arena() {
                    Some((arena, count)) if count >= parent_count => {
                        level.preflight_fused_arena(arena, parent_count)
                    }
                    _ => Err(BfsStepError::RuntimeError),
                };
                if let Err(error) = preflight_result {
                    eprintln!("[compiled-bfs] fused level preflight fatal error: {error}");
                    return CheckResult::from_error(
                        RuntimeCheckError::Internal(format!(
                            "compiled BFS fatal error: fused preflight failed: {error}"
                        ))
                        .into(),
                        self.stats.clone(),
                    );
                }
            }
        }
        telemetry_eprintln!(
            "[compiled-bfs] starting compiled BFS level loop ({} initial states in arena, fused={})",
            flat_queue.len(),
            use_fused,
        );
        stats.execution_started_at = Some(std::time::Instant::now());
        telemetry_eprintln!(
            "[compiled-bfs] compiled_bfs_step_active={} compiled_bfs_level_active={use_fused}",
            self.compiled_bfs_step.is_some(),
        );

        // P0: monotonic level index for sampling the fused-arena native↔interpreter crosscheck.
        let mut fused_crosscheck_level_index = 0usize;

        // Wide-level parent SUB-BATCHING state. When a fused BFS level's predicted
        // transient successor arena would be large, feed the level's parents to the
        // native fused call in bounded sub-batches (each a contiguous, same-depth
        // slice of the level's parents), draining each sub-batch's successors into
        // the SAME global dedup set + frontier before the next. This caps the
        // transient arena to the sub-batch (not the whole widest level), the peak
        // that dominated RSS on record-set specs, while producing the exact same
        // reachable set / dedup / invariant results (BFS is FIFO; the global seen
        // set is persistent across sub-batches; cross-sub-batch duplicates are
        // caught by that set and confirmed via eagerly-recorded payload witnesses).
        //
        // Config (gates that never change mid-run are folded in here so a
        // non-eligible run pays nothing): disabled for trace-file / checkpoint /
        // liveness-graph capture and for depth/state-limited runs, where the drain
        // or completion semantics differ from the plain reachable-set sweep — those
        // fall back to the whole-level native call (fail-safe, never a wrong result).
        let subbatch_target_bytes = compiled_bfs_subbatch_target_bytes();
        let subbatch_engage_bytes = compiled_bfs_subbatch_engage_bytes();
        // Per-record transient-arena cost (i64 slots × 8B). Used to convert the
        // byte-denominated target/engage thresholds into a parent count.
        let subbatch_record_bytes = flat_queue.slots_per_state().saturating_mul(8).max(1);
        // A trace file is fine (auto-created for normal runs): sub-batching records
        // the SAME first-discoverer parent→successor edges (dedup keeps the lowest
        // parent index either way), just interleaved with the drain. Checkpointing,
        // liveness-graph capture, and depth/state limits ARE disabled — their
        // whole-level snapshot / edge-capture / early-stop semantics are out of scope
        // for this change and fall back to the whole-level call (fail-safe).
        let subbatch_enabled = subbatch_target_bytes > 0
            && self.checkpoint.dir.is_none()
            && !self.liveness_cache.cache_for_liveness
            && self.exploration.max_states.is_none()
            && self.exploration.max_depth.is_none();
        // Per-level sub-batch decision + carried learning (spp = successor arena
        // FILL per parent, learned from committed yield so the predicted arena
        // tracks the real per-level allocation, not the raw pre-dedup edge count).
        let mut subbatch_level_depth: Option<usize> = None;
        let mut subbatch_active = false;
        let mut subbatch_spp_estimate: usize = 1;
        let use_compiled_fingerprint =
            matches!(bfs_fingerprint_domain, BfsFingerprintDomain::CompiledFlat);

        // Main level loop: process levels until frontier is empty or violation found.
        loop {
            // Wall-clock deadline backstop (Part of #3: overrun/OOM hardening).
            // Checked once per BFS level — the same granularity as the existing
            // portfolio and depth-limit checks below. Unlike the normal loop
            // `break` (which finalizes as a complete/Success exploration), a
            // deadline-triggered stop must report `LimitReached { Time }`: the
            // frontier is non-empty, so reporting success would be unsound. This
            // mirrors the unified worker loop's `on_deadline_exceeded`.
            if let Some(deadline) = self.exploration.deadline {
                if std::time::Instant::now() >= deadline {
                    self.report_compiled_bfs_stats(&stats);
                    return self.finalize_terminal_result_with_storage(CheckResult::LimitReached {
                        limit_type: super::super::LimitType::Time,
                        stats: self.stats.clone(),
                    });
                }
            }
            // Check if another portfolio lane has resolved the verdict.
            if let Some(ref sv) = self.portfolio_verdict {
                if sv.is_resolved() {
                    break;
                }
            }
            #[cfg(feature = "ay")]
            if let Some(ref coop) = self.cooperative {
                if coop.verdict.is_resolved() {
                    break;
                }
            }

            // Snapshot the current-level parent count. Successors are appended
            // to the same frontier arena below, so do not keep a borrowed arena
            // slice across enqueue operations.
            if flat_queue.has_fallback_entries() {
                eprintln!(
                    "[compiled-bfs] non-flat fallback entries remain in frontier, \
                     falling back to standard BFS loop"
                );
                if stats.levels_completed > 0
                    || stats.parents_processed > 0
                    || stats.successors_generated > 0
                    || stats.successors_new > 0
                {
                    self.report_compiled_bfs_stats(&stats);
                }
                if !self.compiled_bfs_step_usable_for_current_graph_mode() {
                    self.disable_compiled_bfs_for_standard_fallback();
                }
                if self.trust_cg_native_fused_strict_enabled() {
                    return self.strict_native_fused_fallback_result(
                        "non-flat fallback entries remain in the frontier",
                    );
                }
                return self.run_bfs_loop(storage, flat_queue);
            }

            let total_remaining = flat_queue.remaining_flat_count();
            if total_remaining == 0 {
                frontier_exhausted = true;
                break; // Frontier empty -- BFS complete
            }

            // Sub-batch bookkeeping. `front_depth` is the depth shared by the
            // leading (FIFO) run of frontier states; a change in it marks a fresh
            // BFS level (all `total_remaining` states are then this new depth, since
            // the prior level fully drained before its successors were exposed as
            // parents). On a fresh level, re-decide whether to sub-batch this level
            // and snapshot its whole size for the once-per-level witness seeding.
            let front_depth = flat_queue.meta_at_offset(0).map(|(_, depth, _)| depth);
            let fresh_level = subbatch_level_depth != front_depth;
            // Parents per sub-batch = target_bytes / (per-parent successor bytes),
            // where per-parent bytes ≈ spp × record_bytes. Bounds the transient arena
            // (records × record_bytes) to ~target_bytes.
            let subbatch_parent_cap = (subbatch_target_bytes
                / subbatch_spp_estimate
                    .saturating_mul(subbatch_record_bytes)
                    .max(1))
            .max(1);
            if fresh_level {
                subbatch_level_depth = front_depth;
                // Predicted whole-level transient arena in bytes.
                let predicted_bytes = total_remaining
                    .saturating_mul(subbatch_spp_estimate)
                    .saturating_mul(subbatch_record_bytes);
                subbatch_active = subbatch_enabled
                    && use_fused
                    && front_depth.is_some()
                    // Native fused level present AND NOT a record-set loop kernel.
                    // The loop-kernel path self-sizes its arena (prev committed
                    // yield × 2), so it lacks the fixed parents×action_count
                    // over-reservation sub-batching targets; leaving it whole keeps
                    // the loop-kernel flagship (PaxosCommit) byte-for-byte unaffected.
                    && self
                        .compiled_bfs_level
                        .as_ref()
                        .is_some_and(|level| !level.native_fused_has_loop_actions())
                    // Predicted whole-level transient arena is large enough to split...
                    && predicted_bytes >= subbatch_engage_bytes
                    // ...and the cap actually splits the level into >1 native call.
                    && subbatch_parent_cap < total_remaining;
                if std::env::var_os("TY_DEBUG_SUBBATCH").is_some() {
                    eprintln!(
                        "[subbatch-decide] depth={front_depth:?} total_remaining={total_remaining} \
                         spp={subbatch_spp_estimate} record_bytes={subbatch_record_bytes} \
                         predicted_mb={} engage_mb={} cap={subbatch_parent_cap} \
                         enabled={subbatch_enabled} use_fused={use_fused} -> active={subbatch_active}",
                        predicted_bytes / (1024 * 1024),
                        subbatch_engage_bytes / (1024 * 1024),
                    );
                }
            }
            let whole_level_count = total_remaining;
            // The count of parents fed to THIS native fused call. Sub-batched levels
            // take a bounded, same-depth prefix; every drain path below consumes and
            // advances by exactly `parent_count`, so successors accumulate into the
            // shared frontier + global dedup set and the next iteration resumes at
            // the following parent — preserving BFS FIFO order and dedup exactly.
            let (parent_count, is_final_subbatch_of_level) = if subbatch_active {
                let front_depth = front_depth.expect("subbatch_active implies a front depth");
                let batch = compiled_bfs_leading_same_depth_run(
                    flat_queue,
                    front_depth,
                    subbatch_parent_cap,
                );
                // Final sub-batch of the level iff the next state (if any) is deeper.
                let is_final = flat_queue
                    .meta_at_offset(batch)
                    .map_or(true, |(_, depth, _)| depth != front_depth);
                (batch, is_final)
            } else {
                (total_remaining, true)
            };
            if parent_count == 0 {
                break; // Defensive: no same-depth parents to process.
            }

            // P0: default-on SAMPLED native↔interpreter equivalence on the fused arena fast path.
            // Validates the native compiled-BFS execution against the interpreter oracle on the
            // FIRST few levels (the native callout + arena CODE is level-invariant, so early real
            // states exercise it; later/large levels are skipped to bound the extra peek-arena
            // cost). `&mut self` is free here (no `level` borrow held), the peek does not advance
            // the read cursor, and on any divergence we fail CLOSED by recomputing this level via
            // the interpreter-backed loop — sound + non-breaking. FN_003: `cfg!(not(test))` is a
            // COMPILE-TIME gate that is FALSE only for this crate's own unit tests (whose mock
            // levels deliberately diverge to exercise the fallback paths). Integration tests and
            // every production binary compile with `cfg!(test)=false`, so the crosscheck RUNS there
            // — it is disabled for unit-test mocks only, not for real runs.
            let crosscheck_this_level =
                fused_crosscheck_level_index < FUSED_NATIVE_EQUIV_MAX_LEVELS;
            fused_crosscheck_level_index = fused_crosscheck_level_index.saturating_add(1);
            if cfg!(not(test)) && crosscheck_this_level {
                if let Some(stride) = self.fused_native_equiv_stride() {
                    // Batteries-on: cover BOTH native execution paths by default — the fused arena
                    // (peek-run) AND the per-parent compiled step (step_flat_scoped). Either fails
                    // CLOSED to the interpreter-backed loop on divergence (cursor intact here, so the
                    // fallback re-sees every parent — sound + non-breaking). The env-gated full
                    // per-parent crosscheck below remains an opt-in "exact, every-parent" mode.
                    let diverged = if self.fused_bfs_level_available() {
                        self.fused_arena_native_equiv_diverged(flat_queue, parent_count, stride)
                    } else if self.compiled_bfs_step.is_some() {
                        self.fused_step_native_equiv_diverged(flat_queue, parent_count, stride)
                    } else {
                        false
                    };
                    if diverged {
                        self.report_compiled_bfs_stats(&stats);
                        return self.run_bfs_loop(storage, flat_queue);
                    }
                }
            }

            // Deferred native fused-level promotion. Setup skipped the
            // expensive fused parent-loop compile (`should_defer_fused_level_build`)
            // and the per-parent step path has driven the loop so far. Once the
            // cumulative distinct-state count proves the run is large enough to
            // amortize the compile, build and adopt the fused level here — a
            // level boundary, exactly the granularity at which the loop already
            // switches between the fused and step paths on fused runtime
            // errors, so the verdict and state counts are unaffected. The
            // promotion is one-shot: if the build declines, the step path
            // simply continues.
            if self.deferred_fused_level_build
                && self.compiled_bfs_level.is_none()
                && self.states_count()
                    >= super::super::trust_cg_dispatch::trust_cg_fused_level_defer_threshold()
            {
                self.deferred_fused_level_build = false;
                let promotion_started = std::time::Instant::now();
                // Deferred step-path promotion fuses invariants as before (the
                // default builder is ungated); the invariant size-gate applies
                // only to the eager coverage-skippable build.
                let promoted = self
                    .trust_cg_cache
                    .as_ref()
                    .and_then(|cache| self.try_build_trust_cg_compiled_bfs_level(cache));
                if let Some(level) = promoted {
                    self.compiled_bfs_level = Some(Box::new(level));
                }
                if self.fused_bfs_level_available() {
                    if let Some(ref level) = self.compiled_bfs_level {
                        // Mirror the pre-loop arena preflight before first use.
                        // Slice to exactly this (sub-)batch's parents (see the fused
                        // callout below for why `remaining_arena()` may be wider).
                        let preflight_result = match flat_queue.remaining_arena() {
                            Some((arena, count)) if count >= parent_count => {
                                let slots = arena.len() / count;
                                level.preflight_fused_arena(
                                    &arena[..parent_count * slots],
                                    parent_count,
                                )
                            }
                            _ => Err(BfsStepError::RuntimeError),
                        };
                        if let Err(error) = preflight_result {
                            eprintln!("[compiled-bfs] fused level preflight fatal error: {error}");
                            self.report_compiled_bfs_stats(&stats);
                            return CheckResult::from_error(
                                RuntimeCheckError::Internal(format!(
                                    "compiled BFS fatal error: fused preflight failed: {error}"
                                ))
                                .into(),
                                self.stats.clone(),
                            );
                        }
                    }
                    use_fused = true;
                    eprintln!(
                        "[compiled-bfs] deferred native fused level adopted at {} states \
                         (compiled in {}ms at a level boundary)",
                        self.states_count(),
                        promotion_started.elapsed().as_millis(),
                    );
                } else {
                    eprintln!(
                        "[compiled-bfs] deferred native fused level declined at {} states; \
                         per-parent compiled step path continues",
                        self.states_count(),
                    );
                }
            }

            // Deferred native fused-INVARIANT promotion. An action-only fused
            // level was installed (setup eager build or the level promotion
            // above) because the state space had not yet crossed the invariant
            // fusion floor (`TY_FUSED_INVARIANT_MIN_STATES`); invariants have
            // been checked per successor by the interpreter so far. Now that the
            // cumulative distinct-state count proves the run large enough to
            // amortize the (large) invariant-fusion compile, rebuild the level
            // with the invariants fused into the native parent loop. Best-effort
            // upgrade: on any decline the existing action-only level is restored
            // and the interpreter invariant check simply continues — verdict- and
            // count-identical either way (the fingerprint domain is frozen at BFS
            // start, and invariant checking is the same predicate native or
            // interpreted). Fires at a level boundary, exactly the granularity at
            // which the loop already swaps fused/step paths.
            if self.deferred_fused_invariant_build
                && self.states_count()
                    >= super::super::trust_cg_dispatch::trust_cg_fused_invariant_min_states()
            {
                self.deferred_fused_invariant_build = false;
                let refuse_started = std::time::Instant::now();
                // `try_build_..._with_invariant_fusion` early-returns when a level
                // is already installed, so take the current action-only level out
                // first and restore it if the invariant-fused rebuild declines.
                let previous_level = self.compiled_bfs_level.take();
                let refused = self.trust_cg_cache.as_ref().and_then(|cache| {
                    self.try_build_trust_cg_compiled_bfs_level_with_invariant_fusion(cache, true)
                });
                match refused {
                    Some(level) => {
                        self.compiled_bfs_level = Some(Box::new(level));
                        if self.fused_bfs_level_available() {
                            if let Some(ref level) = self.compiled_bfs_level {
                                let preflight_result = match flat_queue.remaining_arena() {
                                    Some((arena, count)) if count >= parent_count => {
                                        let slots = arena.len() / count;
                                        level.preflight_fused_arena(
                                            &arena[..parent_count * slots],
                                            parent_count,
                                        )
                                    }
                                    _ => Err(BfsStepError::RuntimeError),
                                };
                                if let Err(error) = preflight_result {
                                    eprintln!(
                                        "[compiled-bfs] re-fused invariant level preflight fatal error: {error}"
                                    );
                                    self.report_compiled_bfs_stats(&stats);
                                    return CheckResult::from_error(
                                        RuntimeCheckError::Internal(format!(
                                            "compiled BFS fatal error: re-fused invariant preflight failed: {error}"
                                        ))
                                        .into(),
                                        self.stats.clone(),
                                    );
                                }
                            }
                            use_fused = true;
                            eprintln!(
                                "[compiled-bfs] native fused invariant level adopted at {} states \
                                 (re-fused in {}ms at a level boundary; interpreter invariant checks \
                                 handled the run below the TY_FUSED_INVARIANT_MIN_STATES floor)",
                                self.states_count(),
                                refuse_started.elapsed().as_millis(),
                            );
                        }
                    }
                    None => {
                        // Restore the action-only level; interpreter invariant
                        // checks continue unchanged.
                        self.compiled_bfs_level = previous_level;
                        eprintln!(
                            "[compiled-bfs] native fused invariant re-fusion declined at {} states; \
                             action-only level retained, interpreter invariant checks continue",
                            self.states_count(),
                        );
                    }
                }
            }

            if !self.compiled_bfs_step_usable_for_current_graph_mode()
                && !self.fused_bfs_level_available()
            {
                eprintln!(
                    "[compiled-bfs] compiled step and fused level became unavailable, \
                     falling back to standard BFS loop"
                );
                if self.trust_cg_native_fused_strict_enabled() {
                    return self.strict_native_fused_fallback_result(
                        "compiled step and fused level became unavailable",
                    );
                }
                self.disable_compiled_bfs_for_standard_fallback();
                return self.run_bfs_loop(storage, flat_queue);
            }

            // Determine successor depth from the first parent's metadata.
            // All parents in this level share the same depth (BFS invariant).
            let (_first_fp, first_depth, _first_trace_loc) = match flat_queue.meta_at_offset(0) {
                Some(m) => m,
                None => {
                    self.report_compiled_bfs_stats(&stats);
                    return CheckResult::from_error(
                        RuntimeCheckError::Internal(format!(
                            "compiled BFS frontier has {parent_count} parents without first-parent metadata"
                        ))
                        .into(),
                        self.stats.clone(),
                    );
                }
            };

            // Depth limit check (all parents share same depth in BFS).
            if let Some(max_depth) = self.exploration.max_depth {
                if first_depth >= max_depth {
                    limit_reached = Some(LimitType::Depth);
                    flat_queue.advance_read_cursor(parent_count);
                    stats.levels_completed += 1;
                    continue;
                }
            }

            let succ_depth = match first_depth.checked_add(1) {
                Some(d) => d,
                None => {
                    let error = crate::checker_ops::depth_overflow_error(first_depth);
                    self.report_compiled_bfs_stats(&stats);
                    return CheckResult::from_error(error, self.stats.clone());
                }
            };

            let succ_level = match crate::checker_ops::depth_to_tlc_level(succ_depth) {
                Ok(level) => level,
                Err(error) => {
                    self.report_compiled_bfs_stats(&stats);
                    return CheckResult::from_error(error, self.stats.clone());
                }
            };
            self.ctx.set_tlc_level(succ_level);
            // Seed frontier-root witnesses for the WHOLE current level exactly once
            // (on its first sub-batch). Sub-batching splits a level across native
            // calls, so a successor drained in sub-batch 1 may revisit a level root
            // that only appears as a parent in a later sub-batch; recording every
            // root up front (identical to the whole-level path, which records all
            // `parent_count == whole_level_count` roots before draining) keeps that
            // duplicate confirmable. For non-sub-batched levels `fresh_level` is true
            // every iteration and `whole_level_count == parent_count`, so this is
            // byte-identical to the previous per-level recording.
            if fresh_level {
                if use_compiled_fingerprint {
                    self.record_compiled_flat_frontier_payload_witnesses(
                        flat_queue,
                        whole_level_count,
                    );
                } else if step_eval_implied_actions_active {
                    // ArrayFp64 STEP path: seed canonical frontier witnesses so
                    // successors that revisit a frontier root are confirmed as
                    // duplicates instead of failing closed.
                    self.record_compiled_flat_frontier_payload_witnesses_array_state(
                        flat_queue,
                        whole_level_count,
                        &registry,
                    );
                }
            }

            // Part of #4171: Fused BFS level path — process entire frontier
            // in a single native function call when the fused level is available.
            // Falls back to the per-parent step loop on error.
            if use_fused {
                if let Some(ref level) = self.compiled_bfs_level {
                    // Observability-only (un-darkening STEP 1): time the native
                    // fused-level kernel callout. The `Instant::now()` reads are
                    // gated behind `stats.telemetry_enabled`, so a default run
                    // never touches the clock here.
                    let fused_callout_start = stats.telemetry_enabled.then(std::time::Instant::now);
                    let fused_result = match flat_queue.remaining_arena() {
                        Some((arena, count)) if count >= parent_count => {
                            // Feed EXACTLY this (sub-)batch's parents: the native
                            // parent-arena ABI requires the slice length to equal
                            // `parent_count * state_len`. `remaining_arena()` returns
                            // the whole remaining window (which under sub-batching
                            // holds later parents + already-appended successors), so
                            // slice its leading `parent_count` states.
                            let slots = arena.len() / count;
                            level
                                .run_level_fused_arena(&arena[..parent_count * slots], parent_count)
                        }
                        _ => Some(Err(BfsStepError::RuntimeError)),
                    };
                    if let Some(start) = fused_callout_start {
                        stats.native_fused_call_count += 1;
                        stats.native_fused_total_ns += start.elapsed().as_nanos();
                    }

                    match fused_result {
                        Some(Ok(mut level_result)) => {
                            // Learn the per-parent successor-arena FILL rate from this
                            // call's committed yield (max-tracked so the predicted
                            // whole-level arena, and the sub-batch parent cap derived
                            // from it, upper-bound the real allocation). Rate is
                            // batch-size-independent, so a sub-batch's ratio is a valid
                            // whole-level estimate. Seeds the engage/cap decision for
                            // subsequent levels (wide levels always follow narrower
                            // ones that prime this).
                            if parent_count > 0 {
                                let observed = (level_result.successor_count())
                                    .div_ceil(parent_count)
                                    .max(1);
                                subbatch_spp_estimate = subbatch_spp_estimate.max(observed);
                            }
                            let native_fused_flat_frontier_admission_active =
                                self.native_fused_flat_frontier_admission_active();
                            let level_may_skip_global_pre_seen_lookup =
                                fused_level_may_skip_global_pre_seen_lookup(
                                    level.skip_global_pre_seen_lookup(),
                                    self.flat_state_primary,
                                    native_fused_flat_frontier_admission_active,
                                );
                            if self.exploration.check_deadlock
                                && level_result.raw_successor_metadata_complete()
                            {
                                if let Some(deadlocked_parent_idx) =
                                    level_result.first_parent_without_raw_successors()
                                {
                                    let expected_processed =
                                        deadlocked_parent_idx.saturating_add(1);
                                    let valid_processed = level_result.parents_processed
                                        == expected_processed
                                        || level_result.parents_processed == parent_count;
                                    if deadlocked_parent_idx >= parent_count || !valid_processed {
                                        self.report_compiled_bfs_stats(&stats);
                                        return CheckResult::from_error(
                                            RuntimeCheckError::Internal(format!(
                                                "fused compiled BFS deadlock metadata reported \
                                                 parent {deadlocked_parent_idx} with {} parents \
                                                 processed for current level of {parent_count}",
                                                level_result.parents_processed,
                                            ))
                                            .into(),
                                            self.stats.clone(),
                                        );
                                    }
                                    self.record_transitions(
                                        usize::try_from(level_result.total_generated)
                                            .unwrap_or(usize::MAX),
                                    );
                                    if level_result.total_generated > 0 {
                                        self.stats.max_depth = self.stats.max_depth.max(succ_depth);
                                    }
                                    let parent_slice =
                                        flat_queue.remaining_state_at_offset(deadlocked_parent_idx);
                                    stats.parents_processed +=
                                        level_result.parents_processed as u64;
                                    stats.successors_generated += level_result.total_generated;
                                    let Some(parent_slice) = parent_slice else {
                                        self.report_compiled_bfs_stats(&stats);
                                        return CheckResult::from_error(
                                            RuntimeCheckError::Internal(
                                                "fused deadlock reported without a parent state"
                                                    .to_string(),
                                            )
                                            .into(),
                                            self.stats.clone(),
                                        );
                                    };
                                    let state = match self
                                        .try_reconstruct_state_from_flat(parent_slice, &registry)
                                    {
                                        Ok(state) => state,
                                        Err(result) => {
                                            self.report_compiled_bfs_stats(&stats);
                                            return result;
                                        }
                                    };
                                    self.report_compiled_bfs_stats(&stats);

                                    // Reconstruct the full parent chain ending at
                                    // the deadlocked state, mirroring the
                                    // interpreter path (run_helpers.rs
                                    // check_deadlock -> reconstruct_trace(fp)). The
                                    // deadlocked state's own fingerprint is already
                                    // admitted to the trace file via
                                    // mark_state_seen_fp_only_*, so reconstruct_trace
                                    // walks back to Init. Fall back to the single
                                    // state when no trace file is configured
                                    // (reconstruct_trace returns an empty Trace).
                                    let trace = match flat_queue
                                        .meta_at_offset(deadlocked_parent_idx)
                                    {
                                        Some((deadlock_fp, _, _)) => {
                                            let reconstructed = self.reconstruct_trace(deadlock_fp);
                                            if reconstructed.is_empty() {
                                                Trace::from_states(vec![state])
                                            } else {
                                                reconstructed
                                            }
                                        }
                                        None => Trace::from_states(vec![state]),
                                    };

                                    return CheckResult::Deadlock {
                                        trace,
                                        stats: self.stats.clone(),
                                    };
                                }
                            }

                            if level_result.invariant_ok
                                && level_result.parents_processed != parent_count
                            {
                                self.report_compiled_bfs_stats(&stats);
                                return CheckResult::from_error(
                                    RuntimeCheckError::Internal(format!(
                                        "fused compiled BFS processed {} parents for current \
                                         level of {parent_count}",
                                        level_result.parents_processed,
                                    ))
                                    .into(),
                                    self.stats.clone(),
                                );
                            }

                            if self.exploration.check_deadlock
                                && !level_result.raw_successor_metadata_complete()
                            {
                                if !self.config.constraints.is_empty() {
                                    self.report_compiled_bfs_stats(&stats);
                                    return CheckResult::from_error(
                                        RuntimeCheckError::Internal(
                                            "state-constrained compiled BFS fused level failed \
                                             closed: missing raw-successor metadata for deadlock \
                                             checking"
                                                .to_string(),
                                        )
                                        .into(),
                                        self.stats.clone(),
                                    );
                                }
                                eprintln!(
                                    "[compiled-bfs] fused level lacks raw-successor metadata \
                                     for deadlock checking, falling back to per-parent step"
                                );
                                if self.trust_cg_native_fused_strict_enabled() {
                                    self.report_compiled_bfs_stats(&stats);
                                    return self.strict_native_fused_fallback_result(
                                        "fused level lacks raw-successor metadata for deadlock checking",
                                    );
                                }
                                self.compiled_bfs_level = None;
                                if !self.compiled_bfs_step_usable_for_current_graph_mode() {
                                    self.disable_compiled_bfs_for_standard_fallback();
                                    if self.trust_cg_native_fused_strict_enabled() {
                                        self.report_compiled_bfs_stats(&stats);
                                        return self.strict_native_fused_fallback_result(
                                            "per-parent compiled step is unavailable after fused metadata fallback",
                                        );
                                    }
                                    return self.run_bfs_loop(storage, flat_queue);
                                }
                                // Fall through to per-parent loop below.
                            } else if self.liveness_cache.cache_for_liveness
                                && !level_result.state_graph_successors_complete()
                            {
                                if !self.config.constraints.is_empty() {
                                    self.report_compiled_bfs_stats(&stats);
                                    return CheckResult::from_error(
                                        RuntimeCheckError::Internal(
                                            "state-constrained compiled BFS fused level failed \
                                             closed: missing complete state-graph successor \
                                             metadata for liveness capture"
                                                .to_string(),
                                        )
                                        .into(),
                                        self.stats.clone(),
                                    );
                                }
                                eprintln!(
                                    "[compiled-bfs] fused level lacks complete successor edge metadata \
                                     for state-graph capture, falling back to per-parent step"
                                );
                                if self.trust_cg_native_fused_strict_enabled() {
                                    self.report_compiled_bfs_stats(&stats);
                                    return self.strict_native_fused_fallback_result(
                                        "fused level lacks complete successor edge metadata for state-graph capture",
                                    );
                                }
                                self.compiled_bfs_level = None;
                                if !self.compiled_bfs_step_usable_for_current_graph_mode() {
                                    self.disable_compiled_bfs_for_standard_fallback();
                                    if self.trust_cg_native_fused_strict_enabled() {
                                        self.report_compiled_bfs_stats(&stats);
                                        return self.strict_native_fused_fallback_result(
                                            "per-parent compiled step is unavailable after liveness metadata fallback",
                                        );
                                    }
                                    return self.run_bfs_loop(storage, flat_queue);
                                }
                                // Fall through to per-parent loop below.
                            } else {
                                self.record_transitions(
                                    usize::try_from(level_result.total_generated)
                                        .unwrap_or(usize::MAX),
                                );
                                if level_result.total_generated > 0 {
                                    self.stats.max_depth = self.stats.max_depth.max(succ_depth);
                                }
                                // Handle invariant violation from fused level.
                                if !level_result.invariant_ok {
                                    let Some(inv_idx) = level_result.failed_invariant_idx else {
                                        flat_queue.advance_read_cursor(parent_count);
                                        stats.parents_processed +=
                                            level_result.parents_processed as u64;
                                        stats.successors_generated += level_result.total_generated;
                                        self.report_compiled_bfs_stats(&stats);
                                        return CheckResult::from_error(
                                            RuntimeCheckError::Internal(
                                                "fused compiled BFS reported invariant failure \
                                                 without failed invariant metadata"
                                                    .to_string(),
                                            )
                                            .into(),
                                            self.stats.clone(),
                                        );
                                    };
                                    let Some(failed_parent_idx) = level_result.failed_parent_idx
                                    else {
                                        self.report_compiled_bfs_stats(&stats);
                                        return CheckResult::from_error(
                                            RuntimeCheckError::Internal(
                                                "fused compiled BFS reported invariant failure \
                                                 without failed parent metadata"
                                                    .to_string(),
                                            )
                                            .into(),
                                            self.stats.clone(),
                                        );
                                    };
                                    let expected_processed = failed_parent_idx.saturating_add(1);
                                    if failed_parent_idx >= parent_count
                                        || level_result.parents_processed != expected_processed
                                    {
                                        self.report_compiled_bfs_stats(&stats);
                                        return CheckResult::from_error(
                                            RuntimeCheckError::Internal(format!(
                                                "fused compiled BFS invariant metadata reported \
                                                 parent {failed_parent_idx} with {} parents \
                                                 processed for current level of {parent_count}",
                                                level_result.parents_processed,
                                            ))
                                            .into(),
                                            self.stats.clone(),
                                        );
                                    }
                                    let (inv_name, action_property_violation) =
                                        self.compiled_bfs_failure_name_and_kind(inv_idx);

                                    let Some(failed_successor) = level_result.failed_successor()
                                    else {
                                        self.report_compiled_bfs_stats(&stats);
                                        return CheckResult::from_error(
                                            RuntimeCheckError::Internal(
                                                "fused compiled BFS reported invariant failure \
                                                 without failed successor metadata"
                                                    .to_string(),
                                            )
                                            .into(),
                                            self.stats.clone(),
                                        );
                                    };
                                    let Some((parent_fp, _parent_depth, parent_trace_loc)) =
                                        flat_queue.meta_at_offset(failed_parent_idx)
                                    else {
                                        self.report_compiled_bfs_stats(&stats);
                                        return CheckResult::from_error(
                                            RuntimeCheckError::Internal(format!(
                                                "fused compiled BFS invariant failure missing \
                                                 parent metadata at index {failed_parent_idx}",
                                            ))
                                            .into(),
                                            self.stats.clone(),
                                        );
                                    };
                                    let result = self.compiled_bfs_flat_invariant_violation_result(
                                        parent_fp,
                                        parent_trace_loc,
                                        failed_successor,
                                        inv_name,
                                        action_property_violation,
                                        succ_depth,
                                        &registry,
                                    );

                                    flat_queue.advance_read_cursor(parent_count);
                                    stats.parents_processed +=
                                        level_result.parents_processed as u64;
                                    stats.successors_generated += level_result.total_generated;
                                    self.report_compiled_bfs_stats(&stats);
                                    return result;
                                }

                                // Process the fused level's successors through global dedup.
                                // Clone the bridge to avoid holding an immutable borrow on
                                // `self.flat_bfs_adapter` while calling mutable methods below.
                                let bridge = self
                                    .flat_bfs_adapter
                                    .as_ref()
                                    .map(|adapter| adapter.bridge().clone());

                                debug_assert_eq!(
                                    level_result.successor_count() as u64,
                                    level_result.total_new,
                                    "fused level arena count must match backend new count",
                                );
                                let trust_backend_fingerprint_sidecars =
                                    fused_successor_trusts_backend_fingerprint_sidecars(
                                        self.flat_state_primary,
                                        native_fused_flat_frontier_admission_active,
                                    );
                                let mut compiled_fingerprint_successors_validated_once =
                                    if use_compiled_fingerprint {
                                        let Some(bridge) = bridge.as_ref() else {
                                            flat_queue.advance_read_cursor(parent_count);
                                            stats.parents_processed +=
                                                level_result.parents_processed as u64;
                                            stats.successors_generated +=
                                                level_result.total_generated;
                                            self.report_compiled_bfs_stats(&stats);
                                            return CheckResult::from_error(
                                                RuntimeCheckError::Internal(
                                                    "compiled BFS flat fingerprinting requested \
                                                     without a FlatBfsAdapter"
                                                        .to_string(),
                                                )
                                                .into(),
                                                self.stats.clone(),
                                            );
                                        };
                                        if level_result.state_len() != bridge.num_slots() {
                                            flat_queue.advance_read_cursor(parent_count);
                                            stats.parents_processed +=
                                                level_result.parents_processed as u64;
                                            stats.successors_generated +=
                                                level_result.total_generated;
                                            self.report_compiled_bfs_stats(&stats);
                                            return self.flat_reconstruction_error_result(
                                                "fused compiled BFS successor width mismatch",
                                                format!(
                                                    "backend returned {} i64 slots per state, \
                                                     adapter expects {}",
                                                    level_result.state_len(),
                                                    bridge.num_slots(),
                                                ),
                                            );
                                        }
                                        trust_backend_fingerprint_sidecars
                                    } else {
                                        false
                                    };
                                let mut traditional_fingerprint_successors_validated_once = false;
                                if let Some(bridge) = bridge.as_ref() {
                                    if level_result.state_len() == bridge.num_slots() {
                                        if bridge.raw_admission_validation_required() {
                                            let mut successors_needing_canonicalization =
                                                Vec::new();
                                            for (successor_idx, succ_buf) in
                                                level_result.iter_successors().enumerate()
                                            {
                                                if bridge
                                                    .validate_raw_buffer_for_admission(succ_buf)
                                                    .is_err()
                                                {
                                                    successors_needing_canonicalization
                                                        .push(successor_idx);
                                                }
                                            }
                                            if !successors_needing_canonicalization.is_empty() {
                                                let mut successor_idx = 0usize;
                                                let mut canonicalization_idx = 0usize;
                                                if let Err(error) = level_result
                                                    .for_each_successor_mut(|succ_buf| {
                                                        let current_idx = successor_idx;
                                                        successor_idx += 1;
                                                        if successors_needing_canonicalization
                                                            .get(canonicalization_idx)
                                                            .is_some_and(|idx| *idx == current_idx)
                                                        {
                                                            canonicalization_idx += 1;
                                                            bridge
                                                                .canonicalize_raw_buffer_for_admission(
                                                                    succ_buf,
                                                                )
                                                        } else {
                                                            Ok(())
                                                        }
                                                    })
                                                {
                                                    flat_queue.advance_read_cursor(parent_count);
                                                    stats.parents_processed +=
                                                        level_result.parents_processed as u64;
                                                    stats.successors_generated +=
                                                        level_result.total_generated;
                                                    self.report_compiled_bfs_stats(&stats);
                                                    return self.flat_reconstruction_error_result(
                                                        "failed to canonicalize fused compiled BFS \
                                                         successor buffer",
                                                        error,
                                                    );
                                                }
                                                compiled_fingerprint_successors_validated_once =
                                                    false;
                                            }
                                        }
                                        traditional_fingerprint_successors_validated_once = true;
                                    }
                                }
                                let mut global_new: u64 = 0;
                                let mut fused_succ_processed: u64 = 0;
                                let mut liveness_successors =
                                    self.liveness_cache.cache_for_liveness.then(|| {
                                        CompiledBfsLivenessSuccessors::with_successor_capacity(
                                            level_result.successor_count(),
                                        )
                                    });
                                let needs_rust_regular_invariant_check =
                                    fused_successor_needs_rust_regular_invariant_check(
                                        level_result.regular_invariants_checked_by_backend(),
                                        self.config.invariants.len(),
                                    );
                                let needs_pre_seen_lookup = fused_successor_needs_pre_seen_lookup(
                                    level_result.regular_invariants_checked_by_backend(),
                                    self.config.invariants.len(),
                                    level_may_skip_global_pre_seen_lookup,
                                );
                                let needs_trace_invariant_check =
                                    !self.config.trace_invariants.is_empty();
                                let needs_successor_parent_provenance = liveness_successors
                                    .is_some()
                                    || self.trace.trace_file.is_some()
                                    || self.checkpoint.dir.is_some();
                                let successor_parent_indices_complete =
                                    level_result.successor_parent_indices_complete();
                                let use_batch_admission = level_result.successor_count() > 0
                                    && !needs_pre_seen_lookup
                                    && !needs_rust_regular_invariant_check
                                    && !needs_trace_invariant_check
                                    && successor_parent_indices_complete;
                                if use_batch_admission {
                                    let borrowed_batch_inputs = if use_compiled_fingerprint
                                        && compiled_fingerprint_successors_validated_once
                                    {
                                        level_result.successor_fingerprint_values()
                                    } else {
                                        None
                                    };
                                    if let Some(batch_fp_values) = borrowed_batch_inputs {
                                        let successor_count = level_result.successor_count();
                                        let parent_indices = if needs_successor_parent_provenance {
                                            match level_result.successor_parent_indices() {
                                                Some(parent_indices) => Some(parent_indices),
                                                None => {
                                                    flat_queue.advance_read_cursor(parent_count);
                                                    stats.parents_processed +=
                                                        level_result.parents_processed as u64;
                                                    stats.successors_generated +=
                                                        level_result.total_generated;
                                                    stats.successors_new += global_new;
                                                    self.report_compiled_bfs_stats(&stats);
                                                    return CheckResult::from_error(
                                                        RuntimeCheckError::Internal(
                                                            "fused compiled BFS batch admission requested \
                                                             without complete successor parent metadata"
                                                                .to_string(),
                                                        )
                                                        .into(),
                                                        self.stats.clone(),
                                                    );
                                                }
                                            }
                                        } else {
                                            None
                                        };

                                        if let Some(parent_indices) = parent_indices {
                                            if parent_indices.len() != successor_count {
                                                flat_queue.advance_read_cursor(parent_count);
                                                stats.parents_processed +=
                                                    level_result.parents_processed as u64;
                                                stats.successors_generated +=
                                                    level_result.total_generated;
                                                stats.successors_new += global_new;
                                                self.report_compiled_bfs_stats(&stats);
                                                return CheckResult::from_error(
                                                    RuntimeCheckError::Internal(
                                                        "fused compiled BFS batch admission requested \
                                                         without complete successor parent metadata"
                                                            .to_string(),
                                                    )
                                                    .into(),
                                                    self.stats.clone(),
                                                );
                                            }
                                            if let Some((_, parent_idx)) =
                                                parent_indices.first_out_of_bounds(parent_count)
                                            {
                                                flat_queue.advance_read_cursor(parent_count);
                                                stats.parents_processed +=
                                                    level_result.parents_processed as u64;
                                                stats.successors_generated +=
                                                    level_result.total_generated;
                                                stats.successors_new += global_new;
                                                self.report_compiled_bfs_stats(&stats);
                                                return CheckResult::from_error(
                                                    RuntimeCheckError::Internal(format!(
                                                        "fused compiled BFS batch admission reported \
                                                         parent index {parent_idx} for {parent_count} \
                                                         parents"
                                                    ))
                                                    .into(),
	                                                    self.stats.clone(),
	                                                );
                                            }
                                            for successor_idx in 0..successor_count {
                                                let Some(parent_idx) =
                                                    parent_indices.get(successor_idx)
                                                else {
                                                    flat_queue.advance_read_cursor(parent_count);
                                                    stats.parents_processed +=
                                                        level_result.parents_processed as u64;
                                                    stats.successors_generated +=
                                                        level_result.total_generated;
                                                    stats.successors_new += global_new;
                                                    self.report_compiled_bfs_stats(&stats);
                                                    return CheckResult::from_error(
	                                                        RuntimeCheckError::Internal(
	                                                            "fused compiled BFS batch admission \
	                                                             requested without successor parent metadata"
	                                                                .to_string(),
	                                                        )
	                                                        .into(),
	                                                        self.stats.clone(),
	                                                    );
                                                };
                                                if flat_queue.meta_at_offset(parent_idx).is_none() {
                                                    flat_queue.advance_read_cursor(parent_count);
                                                    stats.parents_processed +=
                                                        level_result.parents_processed as u64;
                                                    stats.successors_generated +=
                                                        level_result.total_generated;
                                                    stats.successors_new += global_new;
                                                    self.report_compiled_bfs_stats(&stats);
                                                    return CheckResult::from_error(
                                                        RuntimeCheckError::Internal(format!(
                                                            "fused compiled BFS batch admission \
	                                                             missing parent metadata at index \
	                                                             {parent_idx}",
                                                        ))
                                                        .into(),
                                                        self.stats.clone(),
                                                    );
                                                }
                                            }

                                            if let Some(ref mut successors) = liveness_successors {
                                                for successor_idx in 0..successor_count {
                                                    // Part of #4203: Periodic state_count update within the
                                                    // fused successor dedup loop. The fused path processes all
                                                    // parents in native code, so per-parent updates are not
                                                    // possible. Instead, update every
                                                    // COMPILED_BFS_PROGRESS_INTERVAL successors to keep stats
                                                    // fresh for memory pressure checks.
                                                    fused_succ_processed += 1;
                                                    if fused_succ_processed
                                                        % COMPILED_BFS_PROGRESS_INTERVAL
                                                        == 0
                                                    {
                                                        self.stats.states_found =
                                                            self.states_count();
                                                    }
                                                    // 2026-07 OOM audit: in-level memory
                                                    // poll — same graceful stop as the
                                                    // level-boundary check.
                                                    if fused_succ_processed
                                                        % COMPILED_BFS_IN_LEVEL_MEMORY_CHECK_INTERVAL
                                                        == 0
                                                        && self
                                                            .compiled_bfs_in_level_memory_critical(
                                                                fused_succ_processed,
                                                            )
                                                    {
                                                        flat_queue
                                                            .advance_read_cursor(parent_count);
                                                        stats.parents_processed +=
                                                            level_result.parents_processed as u64;
                                                        stats.successors_generated +=
                                                            level_result.total_generated;
                                                        stats.successors_new += global_new;
                                                        self.report_compiled_bfs_stats(&stats);
                                                        return CheckResult::LimitReached {
                                                            limit_type:
                                                                super::super::LimitType::Memory,
                                                            stats: self.stats.clone(),
                                                        };
                                                    }

                                                    let Some(parent_idx) =
                                                        parent_indices.get(successor_idx)
                                                    else {
                                                        flat_queue
                                                            .advance_read_cursor(parent_count);
                                                        stats.parents_processed +=
                                                            level_result.parents_processed as u64;
                                                        stats.successors_generated +=
                                                            level_result.total_generated;
                                                        stats.successors_new += global_new;
                                                        self.report_compiled_bfs_stats(&stats);
                                                        return CheckResult::from_error(
                                                            RuntimeCheckError::Internal(
                                                                "fused compiled BFS batch admission \
                                                                 requested without successor parent metadata"
                                                                    .to_string(),
                                                            )
                                                            .into(),
                                                            self.stats.clone(),
                                                        );
                                                    };
                                                    if flat_queue
                                                        .meta_at_offset(parent_idx)
                                                        .is_none()
                                                    {
                                                        flat_queue
                                                            .advance_read_cursor(parent_count);
                                                        stats.parents_processed +=
                                                            level_result.parents_processed as u64;
                                                        stats.successors_generated +=
                                                            level_result.total_generated;
                                                        stats.successors_new += global_new;
                                                        self.report_compiled_bfs_stats(&stats);
                                                        return CheckResult::from_error(
                                                            RuntimeCheckError::Internal(format!(
                                                                "fused compiled BFS batch admission \
                                                                 missing parent metadata at index \
                                                                 {parent_idx}",
                                                            ))
                                                            .into(),
                                                            self.stats.clone(),
                                                        );
                                                    }
                                                    successors.push(
                                                        parent_idx,
                                                        Fingerprint(batch_fp_values[successor_idx]),
                                                    );
                                                }
                                            } else {
                                                let previous_fused_succ_processed =
                                                    fused_succ_processed;
                                                fused_succ_processed = fused_succ_processed
                                                    .checked_add(successor_count as u64)
                                                    .expect(
                                                        "compiled BFS fused successor count overflow",
                                                    );
                                                if previous_fused_succ_processed
                                                    / COMPILED_BFS_PROGRESS_INTERVAL
                                                    != fused_succ_processed
                                                        / COMPILED_BFS_PROGRESS_INTERVAL
                                                {
                                                    self.stats.states_found = self.states_count();
                                                }
                                                // 2026-07 OOM audit: in-level memory poll
                                                // on the interval crossing (the bulk skip
                                                // jumps the counter by a whole chunk),
                                                // BEFORE batch admission materializes the
                                                // chunk into the frontier.
                                                if previous_fused_succ_processed
                                                    / COMPILED_BFS_IN_LEVEL_MEMORY_CHECK_INTERVAL
                                                    != fused_succ_processed
                                                        / COMPILED_BFS_IN_LEVEL_MEMORY_CHECK_INTERVAL
                                                    && self.compiled_bfs_in_level_memory_critical(
                                                        fused_succ_processed,
                                                    )
                                                {
                                                    flat_queue.advance_read_cursor(parent_count);
                                                    stats.parents_processed +=
                                                        level_result.parents_processed as u64;
                                                    stats.successors_generated +=
                                                        level_result.total_generated;
                                                    stats.successors_new += global_new;
                                                    self.report_compiled_bfs_stats(&stats);
                                                    return CheckResult::LimitReached {
                                                        limit_type:
                                                            super::super::LimitType::Memory,
                                                        stats: self.stats.clone(),
                                                    };
                                                }
                                            }
                                        } else {
                                            let previous_fused_succ_processed =
                                                fused_succ_processed;
                                            fused_succ_processed = fused_succ_processed
                                                .checked_add(successor_count as u64)
                                                .expect(
                                                    "compiled BFS fused successor count overflow",
                                                );
                                            if previous_fused_succ_processed
                                                / COMPILED_BFS_PROGRESS_INTERVAL
                                                != fused_succ_processed
                                                    / COMPILED_BFS_PROGRESS_INTERVAL
                                            {
                                                self.stats.states_found = self.states_count();
                                            }
                                            // 2026-07 OOM audit: in-level memory poll on
                                            // the interval crossing (the bulk skip jumps
                                            // the counter by a whole chunk), BEFORE batch
                                            // admission materializes the chunk into the
                                            // frontier.
                                            if previous_fused_succ_processed
                                                / COMPILED_BFS_IN_LEVEL_MEMORY_CHECK_INTERVAL
                                                != fused_succ_processed
                                                    / COMPILED_BFS_IN_LEVEL_MEMORY_CHECK_INTERVAL
                                                && self.compiled_bfs_in_level_memory_critical(
                                                    fused_succ_processed,
                                                )
                                            {
                                                flat_queue.advance_read_cursor(parent_count);
                                                stats.parents_processed +=
                                                    level_result.parents_processed as u64;
                                                stats.successors_generated +=
                                                    level_result.total_generated;
                                                stats.successors_new += global_new;
                                                self.report_compiled_bfs_stats(&stats);
                                                return CheckResult::LimitReached {
                                                    limit_type: super::super::LimitType::Memory,
                                                    stats: self.stats.clone(),
                                                };
                                            }
                                        }

                                        if compiled_fingerprint_successors_validated_once {
                                            stats.fused_admission_validations_elided +=
                                                successor_count as u64;
                                        }
                                        stats.fused_pre_seen_lookups_skipped +=
                                            successor_count as u64;

                                        if let Err(error) = flat_queue
                                            .try_reserve_raw_buffers(batch_fp_values.len())
                                        {
                                            flat_queue.advance_read_cursor(parent_count);
                                            stats.parents_processed +=
                                                level_result.parents_processed as u64;
                                            stats.successors_generated +=
                                                level_result.total_generated;
                                            stats.successors_new += global_new;
                                            self.report_compiled_bfs_stats(&stats);
                                            return CheckResult::from_error(
                                                RuntimeCheckError::Internal(format!(
                                                    "compiled BFS batch admission could not reserve \
                                                     successor frontier capacity before global \
                                                     admission: {error}"
                                                ))
                                                .into(),
                                                self.stats.clone(),
                                            );
                                        }

                                        let batch_admission_runtime =
                                            match compiled_fused_batch_admission_runtime(
                                                bfs_fingerprint_domain,
                                                &mut compiled_fused_batch_admission,
                                                &mut stats,
                                            ) {
                                                Ok(runtime) => runtime,
                                                Err(fault) => {
                                                    flat_queue.advance_read_cursor(parent_count);
                                                    stats.parents_processed +=
                                                        level_result.parents_processed as u64;
                                                    stats.successors_generated +=
                                                        level_result.total_generated;
                                                    stats.successors_new += global_new;
                                                    self.report_compiled_bfs_stats(&stats);
                                                    return self.storage_fault_result(fault);
                                                }
                                            };
                                        batch_admission_runtime
                                            .insert_batch_fingerprint_values_inserted_indices_checked_into(
                                                self.state_storage.seen_fps.as_ref(),
                                                batch_fp_values,
                                                &mut raw_inserted_index_admission,
                                                &mut stats,
                                            );
                                        let admission = &raw_inserted_index_admission;
                                        if admission.attempted > batch_fp_values.len() {
                                            flat_queue.advance_read_cursor(parent_count);
                                            stats.parents_processed +=
                                                level_result.parents_processed as u64;
                                            stats.successors_generated +=
                                                level_result.total_generated;
                                            stats.successors_new += global_new;
                                            self.report_compiled_bfs_stats(&stats);
                                            return CheckResult::from_error(
                                                RuntimeCheckError::Internal(format!(
                                                    "compiled BFS batch admission attempted {} \
                                                     for {} fingerprints",
                                                    admission.attempted,
                                                    batch_fp_values.len(),
                                                ))
                                                .into(),
                                                self.stats.clone(),
                                            );
                                        }
                                        if admission.fault.is_none()
                                            && admission.attempted != batch_fp_values.len()
                                        {
                                            flat_queue.advance_read_cursor(parent_count);
                                            stats.parents_processed +=
                                                level_result.parents_processed as u64;
                                            stats.successors_generated +=
                                                level_result.total_generated;
                                            stats.successors_new += global_new;
                                            self.report_compiled_bfs_stats(&stats);
                                            return CheckResult::from_error(
                                                RuntimeCheckError::Internal(format!(
                                                    "compiled BFS batch admission attempted {} \
                                                     fingerprints without reporting a fault for \
                                                     {} inputs",
                                                    admission.attempted,
                                                    batch_fp_values.len(),
                                                ))
                                                .into(),
                                                self.stats.clone(),
                                            );
                                        }
                                        let successor_arena = level_result.successor_arena_slice();
                                        let successor_state_len = level_result.state_len();
                                        let successor_count = level_result.successor_count();
                                        assert_eq!(
                                            successor_state_len,
                                            flat_queue.slots_per_state(),
                                            "compiled BFS borrowed batch successor state has {} slots, expected {}",
                                            successor_state_len,
                                            flat_queue.slots_per_state()
                                        );
                                        let expected_successor_slots = successor_count
                                            .checked_mul(successor_state_len)
                                            .expect(
                                                "compiled BFS borrowed batch successor slot count overflow",
                                            );
                                        assert!(
                                            successor_arena.len() >= expected_successor_slots,
                                            "compiled BFS borrowed batch successor arena has {} slots, expected at least {}",
                                            successor_arena.len(),
                                            expected_successor_slots
                                        );
                                        let mut last_inserted_successor_idx: Option<usize> = None;
                                        for &successor_idx in &admission.inserted_indices {
                                            if successor_idx >= admission.attempted
                                                || successor_idx >= batch_fp_values.len()
                                            {
                                                flat_queue.advance_read_cursor(parent_count);
                                                stats.parents_processed +=
                                                    level_result.parents_processed as u64;
                                                stats.successors_generated +=
                                                    level_result.total_generated;
                                                stats.successors_new += global_new;
                                                self.report_compiled_bfs_stats(&stats);
                                                return CheckResult::from_error(
                                                    RuntimeCheckError::Internal(format!(
                                                        "compiled BFS batch admission returned \
                                                         inserted index {successor_idx} outside \
                                                         attempted prefix {} for {} fingerprints",
                                                        admission.attempted,
                                                        batch_fp_values.len(),
                                                    ))
                                                    .into(),
                                                    self.stats.clone(),
                                                );
                                            }
                                            if last_inserted_successor_idx
                                                .is_some_and(|last| successor_idx <= last)
                                            {
                                                flat_queue.advance_read_cursor(parent_count);
                                                stats.parents_processed +=
                                                    level_result.parents_processed as u64;
                                                stats.successors_generated +=
                                                    level_result.total_generated;
                                                stats.successors_new += global_new;
                                                self.report_compiled_bfs_stats(&stats);
                                                return CheckResult::from_error(
                                                    RuntimeCheckError::Internal(format!(
                                                        "compiled BFS batch admission returned \
                                                         non-increasing inserted index \
                                                         {successor_idx}"
                                                    ))
                                                    .into(),
                                                    self.stats.clone(),
                                                );
                                            }
                                            last_inserted_successor_idx = Some(successor_idx);
                                        }
                                        let duplicate_payload_confirmed = admission.fault.is_none()
                                            && admission.inserted_indices.len()
                                                < admission.attempted
                                            && match level_result.successor_parent_indices() {
                                                Some(parent_indices) => self
                                                    .fp_only_batch_duplicate_payloads_confirmed(
                                                        &registry,
                                                        flat_queue,
                                                        parent_indices,
                                                        batch_fp_values,
                                                        &admission.inserted_indices,
                                                        successor_arena,
                                                        successor_state_len,
                                                        admission.attempted,
                                                    ),
                                                None => false,
                                            };
                                        if let Err(fault) = batch_admission_runtime
                                            .enforce_batch_duplicate_authorization_checked(
                                                admission.attempted,
                                                admission.inserted_indices.len(),
                                                admission.fault.is_some(),
                                                duplicate_payload_confirmed,
                                                &mut stats,
                                            )
                                        {
                                            self.stats.states_found = self.states_count();
                                            flat_queue.advance_read_cursor(parent_count);
                                            stats.parents_processed +=
                                                level_result.parents_processed as u64;
                                            stats.successors_generated +=
                                                level_result.total_generated;
                                            stats.successors_new += global_new;
                                            self.report_compiled_bfs_stats(&stats);
                                            return self.storage_fault_result(fault);
                                        }
                                        // Fused-level admission has already processed the whole
                                        // current frontier. Inserted successors are copied into the
                                        // flat frontier below and will get durable flat-payload
                                        // witnesses when that next frontier is recorded before
                                        // expansion, avoiding an extra witness copy while queued.
                                        if self.trace.trace_file.is_none()
                                            && self.checkpoint.dir.is_none()
                                        {
                                            let inserted_count = admission.inserted_indices.len();
                                            flat_queue
                                                .push_prevalidated_raw_buffers_from_arena_indices(
                                                    successor_arena,
                                                    successor_state_len,
                                                    successor_count,
                                                    &admission.inserted_indices,
                                                    batch_fp_values,
                                                    succ_depth,
                                                    self.trace.last_inserted_trace_loc,
                                                );
                                            global_new = global_new
                                                .checked_add(inserted_count as u64)
                                                .expect(
                                                    "compiled BFS borrowed batch new-state count overflow",
                                                );
                                            if inserted_count > 0 {
                                                self.stats.max_depth =
                                                    self.stats.max_depth.max(succ_depth);
                                            }
                                            // Sub-batch soundness: eagerly witness the
                                            // states this sub-batch just admitted so a
                                            // LATER sub-batch of the SAME level that
                                            // regenerates one (a cross-parent duplicate
                                            // the batch path would otherwise leave
                                            // unwitnessed until expansion) can confirm it
                                            // as a real duplicate instead of failing
                                            // closed. No-op off the sub-batch path, so the
                                            // whole-level fast path is byte-identical
                                            // (witness arena is append-only + first-writer-
                                            // wins, so this only records earlier, never
                                            // extra).
                                            if subbatch_active {
                                                self.record_subbatch_inserted_flat_witnesses(
                                                    successor_arena,
                                                    successor_state_len,
                                                    &admission.inserted_indices,
                                                    batch_fp_values,
                                                );
                                            }
                                        } else {
                                            let parent_indices = parent_indices.expect(
                                                "batch parent provenance required for trace/checkpoint bookkeeping",
                                            );
                                            if let Err(result) = self
                                                .record_fp_only_batch_admission_bookkeeping_for_indices(
                                                    flat_queue,
                                                    parent_indices,
                                                    batch_fp_values,
                                                    &admission.inserted_indices,
                                                    succ_depth,
                                                    &mut raw_inserted_successors,
                                                )
                                            {
                                                if !raw_inserted_successors.is_empty() {
                                                    let inserted_count =
                                                        raw_inserted_successors.len();
                                                    flat_queue.push_raw_buffers_from_arena_indices(
                                                        successor_arena,
                                                        successor_state_len,
                                                        successor_count,
                                                        raw_inserted_successors.iter(),
                                                    );
                                                    raw_inserted_successors.clear();
                                                    global_new = global_new
                                                        .checked_add(inserted_count as u64)
                                                        .expect(
                                                            "compiled BFS borrowed batch new-state count overflow",
                                                        );
                                                    self.stats.max_depth =
                                                        self.stats.max_depth.max(succ_depth);
                                                }
                                                flat_queue.advance_read_cursor(parent_count);
                                                stats.parents_processed +=
                                                    level_result.parents_processed as u64;
                                                stats.successors_generated +=
                                                    level_result.total_generated;
                                                stats.successors_new += global_new;
                                                self.report_compiled_bfs_stats(&stats);
                                                return result;
                                            }
                                            if !raw_inserted_successors.is_empty() {
                                                let inserted_count = raw_inserted_successors.len();
                                                flat_queue.push_raw_buffers_from_arena_indices(
                                                    successor_arena,
                                                    successor_state_len,
                                                    successor_count,
                                                    raw_inserted_successors.iter(),
                                                );
                                                raw_inserted_successors.clear();
                                                global_new = global_new
                                                    .checked_add(inserted_count as u64)
                                                    .expect(
                                                        "compiled BFS borrowed batch new-state count overflow",
                                                    );
                                                self.stats.max_depth =
                                                    self.stats.max_depth.max(succ_depth);
                                            }
                                            // Sub-batch soundness (trace/checkpoint
                                            // bookkeeping variant of the borrowed-
                                            // fingerprint batch path — the branch normal
                                            // auto-trace runs take): eagerly witness the
                                            // admitted states for cross-sub-batch dedup.
                                            // See the no-trace branch above.
                                            if subbatch_active {
                                                self.record_subbatch_inserted_flat_witnesses(
                                                    successor_arena,
                                                    successor_state_len,
                                                    &admission.inserted_indices,
                                                    batch_fp_values,
                                                );
                                            }
                                        }

                                        if let Some(fault) = admission.fault.clone() {
                                            self.stats.states_found = self.states_count();
                                            flat_queue.advance_read_cursor(parent_count);
                                            stats.parents_processed +=
                                                level_result.parents_processed as u64;
                                            stats.successors_generated +=
                                                level_result.total_generated;
                                            stats.successors_new += global_new;
                                            self.report_compiled_bfs_stats(&stats);
                                            return self.storage_fault_result(fault);
                                        }
                                    } else {
                                        let successor_count = level_result.successor_count();
                                        let Some(parent_indices) =
                                            level_result.successor_parent_indices()
                                        else {
                                            flat_queue.advance_read_cursor(parent_count);
                                            stats.parents_processed +=
                                                level_result.parents_processed as u64;
                                            stats.successors_generated +=
                                                level_result.total_generated;
                                            stats.successors_new += global_new;
                                            self.report_compiled_bfs_stats(&stats);
                                            return CheckResult::from_error(
                                                RuntimeCheckError::Internal(
                                                    "fused compiled BFS batch admission requested \
                                                     without complete successor parent metadata"
                                                        .to_string(),
                                                )
                                                .into(),
                                                self.stats.clone(),
                                            );
                                        };
                                        let mut batch_fingerprint_values =
                                            Vec::with_capacity(successor_count);

                                        for (successor_idx, succ_buf) in
                                            level_result.iter_successors().enumerate()
                                        {
                                            // Part of #4203: Periodic state_count update within the
                                            // fused successor dedup loop. The fused path processes all
                                            // parents in native code, so per-parent updates are not
                                            // possible. Instead, update every
                                            // COMPILED_BFS_PROGRESS_INTERVAL successors to keep stats
                                            // fresh for memory pressure checks.
                                            fused_succ_processed += 1;
                                            if fused_succ_processed % COMPILED_BFS_PROGRESS_INTERVAL
                                                == 0
                                            {
                                                self.stats.states_found = self.states_count();
                                            }
                                            // 2026-07 OOM audit: in-level memory poll —
                                            // same graceful stop as the level-boundary
                                            // check.
                                            if fused_succ_processed
                                                % COMPILED_BFS_IN_LEVEL_MEMORY_CHECK_INTERVAL
                                                == 0
                                                && self.compiled_bfs_in_level_memory_critical(
                                                    fused_succ_processed,
                                                )
                                            {
                                                flat_queue.advance_read_cursor(parent_count);
                                                stats.parents_processed +=
                                                    level_result.parents_processed as u64;
                                                stats.successors_generated +=
                                                    level_result.total_generated;
                                                stats.successors_new += global_new;
                                                self.report_compiled_bfs_stats(&stats);
                                                return CheckResult::LimitReached {
                                                    limit_type: super::super::LimitType::Memory,
                                                    stats: self.stats.clone(),
                                                };
                                            }

                                            if compiled_fingerprint_successors_validated_once {
                                                stats.fused_admission_validations_elided += 1;
                                            }

                                            // Part of #3987: Use compiled xxh3 when active —
                                            // single SIMD hash of raw i64 buffer, no per-variable
                                            // type dispatch. Falls back to FP64 otherwise.
                                            let succ_fp = if use_compiled_fingerprint {
                                                level_result
                                                .successor_fingerprint_at(successor_idx)
                                                .filter(|_| {
                                                    compiled_fingerprint_successors_validated_once
                                                })
                                                .unwrap_or_else(|| {
                                                    super::super::invariants::fingerprint_flat_compiled(
                                                        succ_buf,
                                                    )
                                                })
                                            } else {
                                                let Some(bridge) = bridge.as_ref() else {
                                                    flat_queue.advance_read_cursor(parent_count);
                                                    stats.parents_processed +=
                                                        level_result.parents_processed as u64;
                                                    stats.successors_generated +=
                                                        level_result.total_generated;
                                                    stats.successors_new += global_new;
                                                    self.report_compiled_bfs_stats(&stats);
                                                    return CheckResult::from_error(
                                                    RuntimeCheckError::Internal(
                                                        "compiled BFS traditional fingerprinting \
                                                         requested without a FlatBfsAdapter"
                                                            .to_string(),
                                                    )
                                                    .into(),
                                                    self.stats.clone(),
                                                );
                                                };
                                                let fingerprint_result =
                                                    if traditional_fingerprint_successors_validated_once
                                                    {
                                                        bridge.try_traditional_fingerprint_from_validated_buffer(
                                                            succ_buf, &registry,
                                                        )
                                                    } else {
                                                        bridge.try_traditional_fingerprint_from_buffer(
                                                            succ_buf, &registry,
                                                        )
                                                    };
                                                match fingerprint_result {
                                                    Ok(fp) => fp,
                                                    Err(error) => {
                                                        flat_queue
                                                            .advance_read_cursor(parent_count);
                                                        stats.parents_processed +=
                                                            level_result.parents_processed as u64;
                                                        stats.successors_generated +=
                                                            level_result.total_generated;
                                                        stats.successors_new += global_new;
                                                        self.report_compiled_bfs_stats(&stats);
                                                        return self.flat_reconstruction_error_result(
                                                        "failed to fingerprint fused compiled BFS \
                                                         successor buffer",
                                                        error,
                                                    );
                                                    }
                                                }
                                            };

                                            stats.fused_pre_seen_lookups_skipped += 1;
                                            let Some(parent_idx) =
                                                parent_indices.get(successor_idx)
                                            else {
                                                flat_queue.advance_read_cursor(parent_count);
                                                stats.parents_processed +=
                                                    level_result.parents_processed as u64;
                                                stats.successors_generated +=
                                                    level_result.total_generated;
                                                stats.successors_new += global_new;
                                                self.report_compiled_bfs_stats(&stats);
                                                return CheckResult::from_error(
                                                RuntimeCheckError::Internal(
                                                    "fused compiled BFS batch admission requested \
                                                     without successor parent metadata"
                                                        .to_string(),
                                                )
                                                .into(),
                                                self.stats.clone(),
                                            );
                                            };
                                            if parent_idx >= parent_count {
                                                flat_queue.advance_read_cursor(parent_count);
                                                stats.parents_processed +=
                                                    level_result.parents_processed as u64;
                                                stats.successors_generated +=
                                                    level_result.total_generated;
                                                stats.successors_new += global_new;
                                                self.report_compiled_bfs_stats(&stats);
                                                return CheckResult::from_error(
                                                    RuntimeCheckError::Internal(format!(
                                                        "fused compiled BFS batch admission reported \
                                                         parent index {parent_idx} for {parent_count} \
                                                         parents"
                                                    ))
                                                    .into(),
                                                    self.stats.clone(),
                                                );
                                            }
                                            if flat_queue.meta_at_offset(parent_idx).is_none() {
                                                flat_queue.advance_read_cursor(parent_count);
                                                stats.parents_processed +=
                                                    level_result.parents_processed as u64;
                                                stats.successors_generated +=
                                                    level_result.total_generated;
                                                stats.successors_new += global_new;
                                                self.report_compiled_bfs_stats(&stats);
                                                return CheckResult::from_error(
                                                    RuntimeCheckError::Internal(format!(
                                                        "fused compiled BFS batch admission reported \
                                                         parent index {parent_idx} for {parent_count} \
                                                         parents"
                                                    ))
                                                    .into(),
                                                    self.stats.clone(),
                                                );
                                            }
                                            if let Some(ref mut successors) = liveness_successors {
                                                successors.push(parent_idx, succ_fp);
                                            }
                                            batch_fingerprint_values.push(succ_fp.0);
                                        }

                                        if let Err(error) =
                                            flat_queue.try_reserve_raw_buffers(successor_count)
                                        {
                                            flat_queue.advance_read_cursor(parent_count);
                                            stats.parents_processed +=
                                                level_result.parents_processed as u64;
                                            stats.successors_generated +=
                                                level_result.total_generated;
                                            stats.successors_new += global_new;
                                            self.report_compiled_bfs_stats(&stats);
                                            return CheckResult::from_error(
                                                RuntimeCheckError::Internal(format!(
                                                    "compiled BFS batch admission could not reserve \
                                                     successor frontier capacity before global \
                                                     admission: {error}"
                                                ))
                                                .into(),
                                                self.stats.clone(),
                                            );
                                        }

                                        let batch_admission_runtime =
                                            match compiled_fused_batch_admission_runtime(
                                                bfs_fingerprint_domain,
                                                &mut compiled_fused_batch_admission,
                                                &mut stats,
                                            ) {
                                                Ok(runtime) => runtime,
                                                Err(fault) => {
                                                    flat_queue.advance_read_cursor(parent_count);
                                                    stats.parents_processed +=
                                                        level_result.parents_processed as u64;
                                                    stats.successors_generated +=
                                                        level_result.total_generated;
                                                    stats.successors_new += global_new;
                                                    self.report_compiled_bfs_stats(&stats);
                                                    return self.storage_fault_result(fault);
                                                }
                                            };
                                        batch_admission_runtime
                                            .insert_batch_fingerprint_values_inserted_indices_checked_into(
                                                self.state_storage.seen_fps.as_ref(),
                                                &batch_fingerprint_values,
                                                &mut raw_inserted_index_admission,
                                                &mut stats,
                                            );
                                        let admission = &raw_inserted_index_admission;
                                        if admission.attempted > batch_fingerprint_values.len() {
                                            flat_queue.advance_read_cursor(parent_count);
                                            stats.parents_processed +=
                                                level_result.parents_processed as u64;
                                            stats.successors_generated +=
                                                level_result.total_generated;
                                            stats.successors_new += global_new;
                                            self.report_compiled_bfs_stats(&stats);
                                            return CheckResult::from_error(
                                                RuntimeCheckError::Internal(format!(
                                                    "compiled BFS batch admission attempted {} \
                                                 for {} fingerprints",
                                                    admission.attempted,
                                                    batch_fingerprint_values.len(),
                                                ))
                                                .into(),
                                                self.stats.clone(),
                                            );
                                        }
                                        if admission.fault.is_none()
                                            && admission.attempted != batch_fingerprint_values.len()
                                        {
                                            flat_queue.advance_read_cursor(parent_count);
                                            stats.parents_processed +=
                                                level_result.parents_processed as u64;
                                            stats.successors_generated +=
                                                level_result.total_generated;
                                            stats.successors_new += global_new;
                                            self.report_compiled_bfs_stats(&stats);
                                            return CheckResult::from_error(
                                                RuntimeCheckError::Internal(format!(
                                                    "compiled BFS batch admission attempted {} \
                                                     fingerprints without reporting a fault for \
                                                     {} inputs",
                                                    admission.attempted,
                                                    batch_fingerprint_values.len(),
                                                ))
                                                .into(),
                                                self.stats.clone(),
                                            );
                                        }
                                        let successor_arena = level_result.successor_arena_slice();
                                        let successor_state_len = level_result.state_len();
                                        let successor_count = level_result.successor_count();
                                        raw_inserted_successors.clear();
                                        let mut last_inserted_successor_idx: Option<usize> = None;
                                        for &successor_idx in &admission.inserted_indices {
                                            if successor_idx >= admission.attempted
                                                || successor_idx >= batch_fingerprint_values.len()
                                            {
                                                flat_queue.advance_read_cursor(parent_count);
                                                stats.parents_processed +=
                                                    level_result.parents_processed as u64;
                                                stats.successors_generated +=
                                                    level_result.total_generated;
                                                stats.successors_new += global_new;
                                                self.report_compiled_bfs_stats(&stats);
                                                return CheckResult::from_error(
                                                    RuntimeCheckError::Internal(format!(
                                                        "compiled BFS batch admission returned \
                                                         inserted index {successor_idx} outside \
                                                         attempted prefix {} for {} fingerprints",
                                                        admission.attempted,
                                                        batch_fingerprint_values.len(),
                                                    ))
                                                    .into(),
                                                    self.stats.clone(),
                                                );
                                            }
                                            if last_inserted_successor_idx
                                                .is_some_and(|last| successor_idx <= last)
                                            {
                                                flat_queue.advance_read_cursor(parent_count);
                                                stats.parents_processed +=
                                                    level_result.parents_processed as u64;
                                                stats.successors_generated +=
                                                    level_result.total_generated;
                                                stats.successors_new += global_new;
                                                self.report_compiled_bfs_stats(&stats);
                                                return CheckResult::from_error(
                                                    RuntimeCheckError::Internal(format!(
                                                        "compiled BFS batch admission returned \
                                                         non-increasing inserted index \
                                                         {successor_idx}"
                                                    ))
                                                    .into(),
                                                    self.stats.clone(),
                                                );
                                            }
                                            last_inserted_successor_idx = Some(successor_idx);
                                        }
                                        let duplicate_payload_confirmed = admission.fault.is_none()
                                            && admission.inserted_indices.len()
                                                < admission.attempted
                                            && self.fp_only_batch_duplicate_payloads_confirmed(
                                                &registry,
                                                flat_queue,
                                                parent_indices,
                                                &batch_fingerprint_values,
                                                &admission.inserted_indices,
                                                successor_arena,
                                                successor_state_len,
                                                admission.attempted,
                                            );
                                        if let Err(fault) = batch_admission_runtime
                                            .enforce_batch_duplicate_authorization_checked(
                                                admission.attempted,
                                                admission.inserted_indices.len(),
                                                admission.fault.is_some(),
                                                duplicate_payload_confirmed,
                                                &mut stats,
                                            )
                                        {
                                            self.stats.states_found = self.states_count();
                                            flat_queue.advance_read_cursor(parent_count);
                                            stats.parents_processed +=
                                                level_result.parents_processed as u64;
                                            stats.successors_generated +=
                                                level_result.total_generated;
                                            stats.successors_new += global_new;
                                            self.report_compiled_bfs_stats(&stats);
                                            return self.storage_fault_result(fault);
                                        }
                                        // Fused-level admission has already processed the whole
                                        // current frontier. Inserted successors are copied into the
                                        // flat frontier below and will get durable flat-payload
                                        // witnesses when that next frontier is recorded before
                                        // expansion, avoiding an extra witness copy while queued.
                                        if let Err(result) = self
                                            .record_fp_only_batch_admission_bookkeeping_for_indices(
                                                flat_queue,
                                                parent_indices,
                                                &batch_fingerprint_values,
                                                &admission.inserted_indices,
                                                succ_depth,
                                                &mut raw_inserted_successors,
                                            )
                                        {
                                            if !raw_inserted_successors.is_empty() {
                                                let inserted_count = raw_inserted_successors.len();
                                                flat_queue.push_raw_buffers_from_arena_indices(
                                                    successor_arena,
                                                    successor_state_len,
                                                    successor_count,
                                                    raw_inserted_successors.iter(),
                                                );
                                                raw_inserted_successors.clear();
                                                global_new = global_new
                                                    .checked_add(inserted_count as u64)
                                                    .expect(
                                                        "compiled BFS batch new-state count overflow",
                                                    );
                                                self.stats.max_depth =
                                                    self.stats.max_depth.max(succ_depth);
                                            }
                                            flat_queue.advance_read_cursor(parent_count);
                                            stats.parents_processed +=
                                                level_result.parents_processed as u64;
                                            stats.successors_generated +=
                                                level_result.total_generated;
                                            stats.successors_new += global_new;
                                            self.report_compiled_bfs_stats(&stats);
                                            return result;
                                        }
                                        if !raw_inserted_successors.is_empty() {
                                            global_new = global_new
                                                .checked_add(raw_inserted_successors.len() as u64)
                                                .expect(
                                                    "compiled BFS batch new-state count overflow",
                                                );
                                            self.stats.max_depth =
                                                self.stats.max_depth.max(succ_depth);
                                        }

                                        if let Some(fault) = admission.fault.clone() {
                                            self.stats.states_found = self.states_count();
                                            if !raw_inserted_successors.is_empty() {
                                                flat_queue.push_raw_buffers_from_arena_indices(
                                                    successor_arena,
                                                    successor_state_len,
                                                    successor_count,
                                                    raw_inserted_successors.iter(),
                                                );
                                                raw_inserted_successors.clear();
                                            }
                                            flat_queue.advance_read_cursor(parent_count);
                                            stats.parents_processed +=
                                                level_result.parents_processed as u64;
                                            stats.successors_generated +=
                                                level_result.total_generated;
                                            stats.successors_new += global_new;
                                            self.report_compiled_bfs_stats(&stats);
                                            return self.storage_fault_result(fault);
                                        }

                                        if !raw_inserted_successors.is_empty() {
                                            flat_queue.push_raw_buffers_from_arena_indices(
                                                successor_arena,
                                                successor_state_len,
                                                successor_count,
                                                raw_inserted_successors.iter(),
                                            );
                                            raw_inserted_successors.clear();
                                        }
                                        // Sub-batch soundness (recomputed-fingerprint
                                        // batch variant): eagerly witness this
                                        // sub-batch's admitted states so a later
                                        // sub-batch of the same level can confirm a
                                        // cross-parent duplicate. See the borrowed-
                                        // fingerprint variant above for the rationale.
                                        if subbatch_active {
                                            self.record_subbatch_inserted_flat_witnesses(
                                                successor_arena,
                                                successor_state_len,
                                                &admission.inserted_indices,
                                                &batch_fingerprint_values,
                                            );
                                        }
                                    }
                                } else {
                                    for (successor_idx, (parent_idx, succ_buf)) in level_result
                                        .iter_successors_with_parent_indices()
                                        .enumerate()
                                    {
                                        // Part of #4203: Periodic state_count update within the
                                        // fused successor dedup loop. The fused path processes all
                                        // parents in native code, so per-parent updates are not
                                        // possible. Instead, update every
                                        // COMPILED_BFS_PROGRESS_INTERVAL successors to keep stats
                                        // fresh for memory pressure checks.
                                        fused_succ_processed += 1;
                                        if fused_succ_processed % COMPILED_BFS_PROGRESS_INTERVAL
                                            == 0
                                        {
                                            self.stats.states_found = self.states_count();
                                        }
                                        // 2026-07 OOM audit: in-level memory poll — same
                                        // graceful stop as the level-boundary check.
                                        if fused_succ_processed
                                            % COMPILED_BFS_IN_LEVEL_MEMORY_CHECK_INTERVAL
                                            == 0
                                            && self.compiled_bfs_in_level_memory_critical(
                                                fused_succ_processed,
                                            )
                                        {
                                            flat_queue.advance_read_cursor(parent_count);
                                            stats.parents_processed +=
                                                level_result.parents_processed as u64;
                                            stats.successors_generated +=
                                                level_result.total_generated;
                                            stats.successors_new += global_new;
                                            self.report_compiled_bfs_stats(&stats);
                                            return CheckResult::LimitReached {
                                                limit_type: super::super::LimitType::Memory,
                                                stats: self.stats.clone(),
                                            };
                                        }

                                        if compiled_fingerprint_successors_validated_once {
                                            stats.fused_admission_validations_elided += 1;
                                        }

                                        // Part of #3987: Use compiled xxh3 when active —
                                        // single SIMD hash of raw i64 buffer, no per-variable
                                        // type dispatch. Falls back to FP64 otherwise.
                                        let succ_fp = if use_compiled_fingerprint {
                                            level_result
                                            .successor_fingerprint_at(successor_idx)
                                            .filter(|_| {
                                                compiled_fingerprint_successors_validated_once
                                            })
                                            .unwrap_or_else(|| {
                                                super::super::invariants::fingerprint_flat_compiled(
                                                    succ_buf,
                                                )
                                            })
                                        } else {
                                            let Some(bridge) = bridge.as_ref() else {
                                                flat_queue.advance_read_cursor(parent_count);
                                                stats.parents_processed +=
                                                    level_result.parents_processed as u64;
                                                stats.successors_generated +=
                                                    level_result.total_generated;
                                                stats.successors_new += global_new;
                                                self.report_compiled_bfs_stats(&stats);
                                                return CheckResult::from_error(
                                                    RuntimeCheckError::Internal(
                                                        "compiled BFS traditional fingerprinting \
                                                     requested without a FlatBfsAdapter"
                                                            .to_string(),
                                                    )
                                                    .into(),
                                                    self.stats.clone(),
                                                );
                                            };
                                            let fingerprint_result =
                                                if traditional_fingerprint_successors_validated_once
                                                {
                                                    bridge.try_traditional_fingerprint_from_validated_buffer(
                                                        succ_buf, &registry,
                                                    )
                                                } else {
                                                    bridge.try_traditional_fingerprint_from_buffer(
                                                        succ_buf, &registry,
                                                    )
                                                };
                                            match fingerprint_result {
                                                Ok(fp) => fp,
                                                Err(error) => {
                                                    flat_queue.advance_read_cursor(parent_count);
                                                    stats.parents_processed +=
                                                        level_result.parents_processed as u64;
                                                    stats.successors_generated +=
                                                        level_result.total_generated;
                                                    stats.successors_new += global_new;
                                                    self.report_compiled_bfs_stats(&stats);
                                                    return self.flat_reconstruction_error_result(
                                                        "failed to fingerprint fused compiled BFS \
                                                     successor buffer",
                                                        error,
                                                    );
                                                }
                                            }
                                        };

                                        let parent_idx = match parent_idx {
                                            Some(parent_idx) if parent_idx < parent_count => {
                                                parent_idx
                                            }
                                            Some(parent_idx) => {
                                                flat_queue.advance_read_cursor(parent_count);
                                                stats.parents_processed +=
                                                    level_result.parents_processed as u64;
                                                stats.successors_generated +=
                                                    level_result.total_generated;
                                                stats.successors_new += global_new;
                                                self.report_compiled_bfs_stats(&stats);
                                                return CheckResult::from_error(
                                                    RuntimeCheckError::Internal(format!(
                                                        "fused compiled BFS successor reported \
                                                         parent index {parent_idx} for \
                                                         {parent_count} parents",
                                                    ))
                                                    .into(),
                                                    self.stats.clone(),
                                                );
                                            }
                                            None => {
                                                flat_queue.advance_read_cursor(parent_count);
                                                stats.parents_processed +=
                                                    level_result.parents_processed as u64;
                                                stats.successors_generated +=
                                                    level_result.total_generated;
                                                stats.successors_new += global_new;
                                                self.report_compiled_bfs_stats(&stats);
                                                return CheckResult::from_error(
                                                    RuntimeCheckError::Internal(
                                                        "fused compiled BFS successor admission \
                                                         requested without successor parent metadata"
                                                            .to_string(),
                                                    )
                                                    .into(),
                                                    self.stats.clone(),
                                                );
                                            }
                                        };
                                        let Some((parent_fp, _depth, parent_trace_loc)) =
                                            flat_queue.meta_at_offset(parent_idx)
                                        else {
                                            flat_queue.advance_read_cursor(parent_count);
                                            stats.parents_processed +=
                                                level_result.parents_processed as u64;
                                            stats.successors_generated +=
                                                level_result.total_generated;
                                            stats.successors_new += global_new;
                                            self.report_compiled_bfs_stats(&stats);
                                            return CheckResult::from_error(
                                                RuntimeCheckError::Internal(format!(
                                                    "fused compiled BFS successor admission \
                                                     missing parent metadata at index {parent_idx}",
                                                ))
                                                .into(),
                                                self.stats.clone(),
                                            );
                                        };

                                        if let Some(ref mut successors) = liveness_successors {
                                            successors.push(parent_idx, succ_fp);
                                        }

                                        // Global dedup check.
                                        if needs_pre_seen_lookup {
                                            stats.fused_pre_seen_lookups += 1;
                                            match self.is_state_seen_checked(succ_fp) {
                                                Ok(true) => {
                                                    // The fingerprint is already resident. Treat
                                                    // this as a confirmed duplicate (and skip) ONLY
                                                    // when we can positively prove the flat payloads
                                                    // are identical — either because the successor
                                                    // is a parent self-loop, or because a recorded
                                                    // compiled-flat payload witness for this
                                                    // fingerprint matches `succ_buf` byte-for-byte.
                                                    // This mirrors the canonical-payload
                                                    // confirmation the post-invariant admission path
                                                    // performs below (see
                                                    // `mark_state_seen_fp_only_with_prepared_admission_checked`),
                                                    // and is required for soundness when regular
                                                    // invariants are checked Rust-side (so the
                                                    // borrowed-batch path with its own witness
                                                    // confirmation is unavailable). Without it, a
                                                    // legitimate cross-path state revisit would be
                                                    // rejected fail-closed as a
                                                    // `canonical_payload_mismatch`. If the payload
                                                    // cannot be confirmed equal here, we still fall
                                                    // through and fail closed.
                                                    let payload_confirmed_duplicate = (parent_fp
                                                        == succ_fp
                                                        && flat_queue
                                                            .remaining_state_at_offset(parent_idx)
                                                            .is_some_and(|parent_buf| {
                                                                parent_buf == succ_buf
                                                            }))
                                                        || self
                                                            .compiled_flat_payload_witness_confirms(
                                                                succ_fp, succ_buf,
                                                            )
                                                            .unwrap_or(false);
                                                    if payload_confirmed_duplicate {
                                                        continue;
                                                    }
                                                    flat_queue.advance_read_cursor(parent_count);
                                                    stats.parents_processed +=
                                                        level_result.parents_processed as u64;
                                                    self.report_compiled_bfs_stats(&stats);
                                                    let admission_handle =
                                                        match cached_fingerprint_only_admission_handle(
                                                            self,
                                                            &mut fingerprint_only_admission_handle,
                                                        ) {
                                                            Ok(handle) => handle,
                                                            Err(fault) => {
                                                                return self
                                                                    .storage_fault_result(fault);
                                                            }
                                                        };
                                                    return self
                                                        .fingerprint_only_duplicate_rejection_result_with_prepared_admission(
                                                            admission_handle,
                                                        );
                                                }
                                                Ok(false) => {}
                                                Err(result) => {
                                                    flat_queue.advance_read_cursor(parent_count);
                                                    stats.parents_processed +=
                                                        level_result.parents_processed as u64;
                                                    self.report_compiled_bfs_stats(&stats);
                                                    return result;
                                                }
                                            }
                                        } else {
                                            stats.fused_pre_seen_lookups_skipped += 1;
                                            debug_assert!(
                                                !needs_rust_regular_invariant_check,
                                                "pre-seen lookup may be skipped only when no Rust-side \
                                             regular invariant check is needed before admission"
                                            );
                                        }

                                        let parent_fp_for_invariant = parent_fp;

                                        let mut succ_array_for_invariants = None;
                                        if needs_rust_regular_invariant_check
                                            || needs_trace_invariant_check
                                        {
                                            succ_array_for_invariants = Some(
                                                match self.try_reconstruct_array_state_from_flat(
                                                    succ_buf, &registry,
                                                ) {
                                                    Ok(succ_array) => succ_array,
                                                    Err(result) => {
                                                        flat_queue
                                                            .advance_read_cursor(parent_count);
                                                        stats.parents_processed +=
                                                            level_result.parents_processed as u64;
                                                        stats.successors_generated +=
                                                            level_result.total_generated;
                                                        stats.successors_new += global_new;
                                                        self.report_compiled_bfs_stats(&stats);
                                                        return result;
                                                    }
                                                },
                                            );
                                        }

                                        if needs_rust_regular_invariant_check {
                                            let succ_array =
                                                succ_array_for_invariants.as_ref().expect(
                                                    "successor reconstructed for invariant check",
                                                );
                                            match self.check_successor_invariant(
                                            parent_fp_for_invariant,
                                            succ_array,
                                            succ_fp,
                                            succ_level,
                                        ) {
                                            crate::checker_ops::InvariantOutcome::Ok
                                            | crate::checker_ops::InvariantOutcome::ViolationContinued => {}
                                            crate::checker_ops::InvariantOutcome::Violation {
                                                invariant,
                                                ..
                                            } => {
                                                let previous_parent_trace_loc =
                                                    self.trace.current_parent_trace_loc;
                                                self.trace.current_parent_trace_loc =
                                                    Some(parent_trace_loc);
                                                let stage_result =
                                                    self.stage_successor_for_terminal_trace(
                                                        parent_fp_for_invariant,
                                                        succ_array,
                                                        succ_fp,
                                                        succ_depth,
                                                    );
                                                self.trace.current_parent_trace_loc =
                                                    previous_parent_trace_loc;
                                                if let Err(result) = stage_result {
                                                    flat_queue.advance_read_cursor(parent_count);
                                                    stats.parents_processed +=
                                                        level_result.parents_processed as u64;
                                                    stats.successors_generated +=
                                                        level_result.total_generated;
                                                    stats.successors_new += global_new;
                                                    self.report_compiled_bfs_stats(&stats);
                                                    return self
                                                        .finalize_terminal_result_with_storage(result);
                                                }
                                                if let Some(result) = self
                                                    .handle_invariant_violation(
                                                        invariant, succ_fp, succ_depth,
                                                    )
                                                {
                                                    flat_queue.advance_read_cursor(parent_count);
                                                    stats.parents_processed +=
                                                        level_result.parents_processed as u64;
                                                    stats.successors_generated +=
                                                        level_result.total_generated;
                                                    stats.successors_new += global_new;
                                                    self.report_compiled_bfs_stats(&stats);
                                                    return result;
                                                }
                                            }
                                            crate::checker_ops::InvariantOutcome::Error(error) => {
                                                flat_queue.advance_read_cursor(parent_count);
                                                stats.parents_processed +=
                                                    level_result.parents_processed as u64;
                                                stats.successors_generated +=
                                                    level_result.total_generated;
                                                stats.successors_new += global_new;
                                                self.report_compiled_bfs_stats(&stats);
                                                return CheckResult::from_error(
                                                    error,
                                                    self.stats.clone(),
                                                );
                                            }
                                        }
                                        }

                                        if needs_trace_invariant_check {
                                            let succ_array = succ_array_for_invariants
                                                .as_ref()
                                                .expect(
                                                "successor reconstructed for trace invariant check",
                                            );
                                            match self.check_trace_invariants(
                                                parent_fp_for_invariant,
                                                succ_array,
                                                succ_fp,
                                            ) {
                                                TraceInvariantOutcome::Ok => {}
                                                TraceInvariantOutcome::Violation {
                                                    invariant,
                                                    trace,
                                                } => {
                                                    let previous_parent_trace_loc =
                                                        self.trace.current_parent_trace_loc;
                                                    self.trace.current_parent_trace_loc =
                                                        Some(parent_trace_loc);
                                                    let stage_result = self
                                                        .stage_successor_for_terminal_trace(
                                                            parent_fp_for_invariant,
                                                            succ_array,
                                                            succ_fp,
                                                            succ_depth,
                                                        );
                                                    self.trace.current_parent_trace_loc =
                                                        previous_parent_trace_loc;
                                                    if let Err(result) = stage_result {
                                                        flat_queue
                                                            .advance_read_cursor(parent_count);
                                                        stats.parents_processed +=
                                                            level_result.parents_processed as u64;
                                                        stats.successors_generated +=
                                                            level_result.total_generated;
                                                        stats.successors_new += global_new;
                                                        self.report_compiled_bfs_stats(&stats);
                                                        return self
                                                            .finalize_terminal_result_with_storage(
                                                                result,
                                                            );
                                                    }
                                                    self.stats.max_depth =
                                                        self.stats.max_depth.max(succ_depth);
                                                    self.stats.states_found = self.states_count();
                                                    self.update_coverage_totals();
                                                    flat_queue.advance_read_cursor(parent_count);
                                                    stats.parents_processed +=
                                                        level_result.parents_processed as u64;
                                                    stats.successors_generated +=
                                                        level_result.total_generated;
                                                    stats.successors_new += global_new;
                                                    self.report_compiled_bfs_stats(&stats);
                                                    let result = CheckResult::InvariantViolation {
                                                        invariant,
                                                        trace,
                                                        stats: self.stats.clone(),
                                                    };
                                                    return self.finalize_terminal_result(result);
                                                }
                                                TraceInvariantOutcome::Error(error) => {
                                                    flat_queue.advance_read_cursor(parent_count);
                                                    stats.parents_processed +=
                                                        level_result.parents_processed as u64;
                                                    stats.successors_generated +=
                                                        level_result.total_generated;
                                                    stats.successors_new += global_new;
                                                    self.report_compiled_bfs_stats(&stats);
                                                    return CheckResult::from_error(
                                                        error,
                                                        self.stats.clone(),
                                                    );
                                                }
                                            }
                                        }

                                        let previous_parent_trace_loc =
                                            self.trace.current_parent_trace_loc;
                                        self.trace.current_parent_trace_loc =
                                            Some(parent_trace_loc);
                                        let duplicate_payload_confirmed = (parent_fp == succ_fp
                                            && flat_queue
                                                .remaining_state_at_offset(parent_idx)
                                                .is_some_and(|parent_buf| parent_buf == succ_buf))
                                            || self
                                                .compiled_flat_payload_witness_confirms(
                                                    succ_fp, succ_buf,
                                                )
                                                .unwrap_or(false);
                                        let admission_handle =
                                            match cached_fingerprint_only_admission_handle(
                                                self,
                                                &mut fingerprint_only_admission_handle,
                                            ) {
                                                Ok(handle) => handle,
                                                Err(fault) => {
                                                    self.trace.current_parent_trace_loc =
                                                        previous_parent_trace_loc;
                                                    flat_queue.advance_read_cursor(parent_count);
                                                    stats.parents_processed +=
                                                        level_result.parents_processed as u64;
                                                    self.report_compiled_bfs_stats(&stats);
                                                    return self.storage_fault_result(fault);
                                                }
                                            };
                                        let mark_result = self
                                            .mark_state_seen_fp_only_with_prepared_admission_checked(
                                                succ_fp,
                                                Some(parent_fp),
                                                succ_depth,
                                                duplicate_payload_confirmed,
                                                admission_handle,
                                            );
                                        self.trace.current_parent_trace_loc =
                                            previous_parent_trace_loc;

                                        match mark_result {
                                            Ok(true) => {}
                                            Ok(false) => continue,
                                            Err(result) => {
                                                flat_queue.advance_read_cursor(parent_count);
                                                stats.parents_processed +=
                                                    level_result.parents_processed as u64;
                                                self.report_compiled_bfs_stats(&stats);
                                                return result;
                                            }
                                        }

                                        global_new += 1;
                                        self.record_compiled_flat_payload_witness_if_absent(
                                            succ_fp, succ_buf,
                                        );
                                        let trace_loc = self.trace.last_inserted_trace_loc;
                                        self.stats.max_depth = self.stats.max_depth.max(succ_depth);
                                        flat_queue.push_raw_buffer(
                                            succ_buf, succ_fp, succ_depth, trace_loc,
                                        );
                                    }
                                }

                                if let Some(successors) = liveness_successors {
                                    if let Err(error) = self
                                        .insert_compiled_bfs_liveness_successors_for_level(
                                            flat_queue,
                                            parent_count,
                                            successors,
                                        )
                                    {
                                        flat_queue.advance_read_cursor(parent_count);
                                        stats.parents_processed +=
                                            level_result.parents_processed as u64;
                                        stats.successors_generated += level_result.total_generated;
                                        stats.successors_new += global_new;
                                        self.report_compiled_bfs_stats(&stats);
                                        return CheckResult::from_error(error, self.stats.clone());
                                    }
                                }

                                flat_queue.advance_read_cursor(parent_count);
                                stats.parents_processed += level_result.parents_processed as u64;
                                stats.successors_generated += level_result.total_generated;
                                stats.successors_new += global_new;
                                // A sub-batched level completes only on its final
                                // sub-batch; count one level per BFS depth (telemetry).
                                if is_final_subbatch_of_level {
                                    stats.levels_completed += 1;
                                }

                                // State limit check after level.
                                if let Some(max_states) = self.exploration.max_states {
                                    if self.states_count() >= max_states {
                                        limit_reached = Some(LimitType::States);
                                    }
                                }

                                // Update model checker stats and progress.
                                let total_states = self.states_count();
                                self.stats.states_found = total_states;
                                if crate::check::debug::compiled_bfs_progress_enabled()
                                    && stats.levels_completed % 5 == 0
                                {
                                    eprintln!(
                                        "[compiled-bfs] fused level {}: {} states, {} generated, {} new, queue={}",
                                        stats.levels_completed,
                                        total_states,
                                        stats.successors_generated,
                                        stats.successors_new,
                                        flat_queue.len(),
                                    );
                                }

                                // Part of #4203: Memory pressure check after fused level.
                                // Fused levels process many parents at once; check OOM
                                // before starting the next level.
                                if let Some(ref policy) = self.exploration.memory_policy {
                                    use crate::memory::MemoryPressure;
                                    if policy.check() == MemoryPressure::Critical {
                                        let rss_mb = crate::memory::current_rss_bytes()
                                            .map(|b| b / (1024 * 1024))
                                            .unwrap_or(0);
                                        let limit_mb = policy.limit_bytes() / (1024 * 1024);
                                        eprintln!(
                                            "[compiled-bfs] memory critical ({rss_mb} MB / {limit_mb} MB limit) \
                                         after fused level {} — stopping.",
                                            stats.levels_completed,
                                        );
                                        self.report_compiled_bfs_stats(&stats);
                                        return CheckResult::LimitReached {
                                            limit_type: super::super::LimitType::Memory,
                                            stats: self.stats.clone(),
                                        };
                                    }
                                }

                                crate::arena::worker_arena_reset();

                                if limit_reached.is_some() {
                                    break;
                                }
                                continue;
                            }
                        }
                        Some(Err(e @ BfsStepError::FatalRuntimeError)) => {
                            eprintln!("[compiled-bfs] fused level fatal error: {e}");
                            self.report_compiled_bfs_stats(&stats);
                            return CheckResult::from_error(
                                RuntimeCheckError::Internal(format!(
                                    "compiled BFS fatal error: {e}"
                                ))
                                .into(),
                                self.stats.clone(),
                            );
                        }
                        Some(Err(e)) => {
                            eprintln!(
                                "[compiled-bfs] fused level error: {e} — \
                                 falling back to per-parent step"
                            );
                            if self.trust_cg_native_fused_strict_enabled() {
                                self.report_compiled_bfs_stats(&stats);
                                return self.strict_native_fused_fallback_result(format!(
                                    "fused level returned recoverable error: {e}"
                                ));
                            }
                            if !self.config.constraints.is_empty() {
                                self.report_compiled_bfs_stats(&stats);
                                return CheckResult::from_error(
                                    RuntimeCheckError::Internal(format!(
                                        "state-constrained compiled BFS fused level failed \
                                         closed: {e}"
                                    ))
                                    .into(),
                                    self.stats.clone(),
                                );
                            }
                            if !self.compiled_bfs_step_usable_for_current_graph_mode() {
                                self.disable_compiled_bfs_for_standard_fallback();
                                if self.trust_cg_native_fused_strict_enabled() {
                                    self.report_compiled_bfs_stats(&stats);
                                    return self.strict_native_fused_fallback_result(
                                        "per-parent compiled step is unavailable after fused error",
                                    );
                                }
                                return self.run_bfs_loop(storage, flat_queue);
                            }
                            // Fall through to per-parent loop below.
                        }
                        None => {
                            // Fused function not available (shouldn't happen
                            // since use_fused checked has_fused_level).
                            if !self.config.constraints.is_empty() {
                                self.report_compiled_bfs_stats(&stats);
                                return CheckResult::from_error(
                                    RuntimeCheckError::Internal(
                                        "state-constrained compiled BFS fused level became \
                                         unavailable after eligibility"
                                            .to_string(),
                                    )
                                    .into(),
                                    self.stats.clone(),
                                );
                            }
                            if !self.compiled_bfs_step_usable_for_current_graph_mode() {
                                self.disable_compiled_bfs_for_standard_fallback();
                                if self.trust_cg_native_fused_strict_enabled() {
                                    self.report_compiled_bfs_stats(&stats);
                                    return self.strict_native_fused_fallback_result(
                                        "per-parent compiled step is unavailable after fused level disappeared",
                                    );
                                }
                                return self.run_bfs_loop(storage, flat_queue);
                            }
                            eprintln!(
                                "[compiled-bfs] fused level returned None after availability \
                                 check; falling back to per-parent step"
                            );
                            if self.trust_cg_native_fused_strict_enabled() {
                                self.report_compiled_bfs_stats(&stats);
                                return self.strict_native_fused_fallback_result(
                                    "fused level returned None after availability check",
                                );
                            }
                        }
                    }
                }
            }

            // Per-parent step loop (fallback when fused path not available or failed).
            let mut level_generated: u64 = 0;
            let mut level_new: u64 = 0;
            let mut level_parents: u64 = 0;
            let mut early_break = false;
            let mut consumed_count = parent_count; // default: consume all
            let crosscheck_with_interpreter =
                std::env::var_os(COMPILED_BFS_INTERPRETER_CROSSCHECK_ENV).is_some();
            if crosscheck_with_interpreter && parent_count > 0 {
                // POSITIVE marker: the per-parent loop (the ONLY path that runs the
                // native-vs-interpreter crosscheck) is active this level. The native
                // FUSED fast path does NOT crosscheck, so a consumer must require THIS
                // marker — not merely "compiled BFS active" — before claiming the
                // native backend was validated against the interpreter. (`ty selfcheck`
                // keys its "native ≡ interpreter" row off this; gated on the env, so
                // there is zero cost in normal runs.)
                telemetry_eprintln!(
                    "[compiled-bfs-xcheck] active: comparing native vs interpreter successors \
                     for {parent_count} parent(s) this level"
                );
            }
            let mut step_scratch = {
                let step = self
                    .compiled_bfs_step
                    .as_ref()
                    .expect("invariant: compiled_bfs_step present in compiled BFS loop");
                CompiledBfsStepScratch::new(step.state_len())
            };

            for parent_idx in 0..parent_count {
                // State limit check.
                if let Some(max_states) = self.exploration.max_states {
                    if self.states_count() >= max_states {
                        limit_reached = Some(LimitType::States);
                        consumed_count = parent_idx;
                        early_break = true;
                        break;
                    }
                }

                // Execute the compiled BFS step.
                let mut crosscheck_parent = None;
                let output = {
                    let parent_slice = match flat_queue.remaining_state_at_offset(parent_idx) {
                        Some(parent_slice) => parent_slice,
                        None => {
                            eprintln!(
                                "[compiled-bfs] missing parent {parent_idx} of current level, \
                                 falling back to standard BFS loop"
                            );
                            if self.trust_cg_native_fused_strict_enabled() {
                                flat_queue.advance_read_cursor(parent_idx);
                                stats.parents_processed += level_parents;
                                stats.successors_generated += level_generated;
                                stats.successors_new += level_new;
                                self.report_compiled_bfs_stats(&stats);
                                return self.strict_native_fused_fallback_result(format!(
                                    "missing parent {parent_idx} of current level"
                                ));
                            }
                            flat_queue.advance_read_cursor(parent_idx);
                            stats.parents_processed += level_parents;
                            stats.successors_generated += level_generated;
                            stats.successors_new += level_new;
                            self.report_compiled_bfs_stats(&stats);
                            return self.run_bfs_loop(storage, flat_queue);
                        }
                    };
                    if crosscheck_with_interpreter {
                        crosscheck_parent = Some(parent_slice.to_vec());
                    }
                    let step = self
                        .compiled_bfs_step
                        .as_ref()
                        .expect("invariant: compiled_bfs_step present in compiled BFS loop");
                    // Observability-only (un-darkening STEP 1): time the native
                    // per-parent step callout. Gated behind `telemetry_enabled`,
                    // so a default run reads the clock zero times per state.
                    let step_callout_start = stats.telemetry_enabled.then(std::time::Instant::now);
                    let step_output = step.step_flat_scoped(parent_slice, &mut step_scratch);
                    if let Some(start) = step_callout_start {
                        stats.native_step_call_count += 1;
                        stats.native_step_total_ns += start.elapsed().as_nanos();
                    }
                    step_output
                };

                let output = match output {
                    Ok(output) => output,
                    Err(e @ BfsStepError::FatalRuntimeError) => {
                        eprintln!(
                            "[compiled-bfs] step fatal error at depth {first_depth}, \
                             parent {parent_idx}: {e}"
                        );
                        stats.parents_processed += level_parents;
                        stats.successors_generated += level_generated;
                        stats.successors_new += level_new;
                        self.report_compiled_bfs_stats(&stats);
                        return CheckResult::from_error(
                            RuntimeCheckError::Internal(format!("compiled BFS fatal error: {e}"))
                                .into(),
                            self.stats.clone(),
                        );
                    }
                    Err(e) => {
                        eprintln!(
                            "[compiled-bfs] step error at depth {first_depth}, \
                             parent {parent_idx}: {e} -- disabling"
                        );
                        if self.trust_cg_native_fused_strict_enabled() {
                            stats.parents_processed += level_parents;
                            stats.successors_generated += level_generated;
                            stats.successors_new += level_new;
                            self.report_compiled_bfs_stats(&stats);
                            return self.strict_native_fused_fallback_result(format!(
                                "per-parent compiled step returned recoverable error: {e}"
                            ));
                        }
                        self.jit_monolithic_disabled = true;
                        self.compiled_bfs_step = None;
                        self.compiled_bfs_level = None;
                        // Advance cursor past processed parents, fall back for rest.
                        flat_queue.advance_read_cursor(parent_idx);
                        stats.parents_processed += level_parents;
                        stats.successors_generated += level_generated;
                        stats.successors_new += level_new;
                        self.report_compiled_bfs_stats(&stats);
                        return self.run_bfs_loop(storage, flat_queue);
                    }
                };
                if output.is_borrowed() {
                    stats.step_outputs_borrowed += 1;
                } else {
                    stats.step_outputs_owned += 1;
                }

                level_parents += 1;
                let output_generated_count = output.generated_count();
                self.record_transitions(
                    usize::try_from(output_generated_count).unwrap_or(usize::MAX),
                );
                level_generated += u64::from(output_generated_count);

                if crosscheck_with_interpreter {
                    let parent_slice = crosscheck_parent.as_deref().ok_or_else(|| {
                        CheckResult::from_error(
                            RuntimeCheckError::Internal(
                                "compiled BFS interpreter crosscheck missing parent snapshot"
                                    .to_string(),
                            )
                            .into(),
                            self.stats.clone(),
                        )
                    });
                    let parent_slice = match parent_slice {
                        Ok(parent_slice) => parent_slice,
                        Err(result) => {
                            flat_queue.advance_read_cursor(parent_idx + 1);
                            stats.parents_processed += level_parents;
                            stats.successors_generated += level_generated;
                            stats.successors_new += level_new;
                            self.report_compiled_bfs_stats(&stats);
                            return result;
                        }
                    };
                    let interpreter_actions = match self
                        .compiled_bfs_interpreter_action_successors_for_parent(
                            parent_slice,
                            &registry,
                        ) {
                        Ok(action_successors) => action_successors,
                        Err(result) => {
                            flat_queue.advance_read_cursor(parent_idx + 1);
                            stats.parents_processed += level_parents;
                            stats.successors_generated += level_generated;
                            stats.successors_new += level_new;
                            self.report_compiled_bfs_stats(&stats);
                            return result;
                        }
                    };
                    let interpreter_action_counts: Vec<_> = interpreter_actions
                        .iter()
                        .map(|action| {
                            (action.action_idx, action.action_name.as_str(), action.len())
                        })
                        .collect();
                    let interpreter_generated: usize = interpreter_actions
                        .iter()
                        .map(CompiledBfsInterpreterActionSuccessors::len)
                        .sum();

                    let compiled_returned = output.successor_count();
                    let mut compiled_successor_matched = vec![false; compiled_returned];

                    let mut first_missing = None;
                    'actions: for action in &interpreter_actions {
                        for (successor_idx, successor) in action.iter_successors().enumerate() {
                            let mut matched_idx = None;
                            for (compiled_idx, matched) in
                                compiled_successor_matched.iter_mut().enumerate()
                            {
                                if *matched {
                                    continue;
                                }
                                if output.successor_at(compiled_idx) == Some(successor) {
                                    *matched = true;
                                    matched_idx = Some(compiled_idx);
                                    break;
                                }
                            }
                            if matched_idx.is_none() {
                                first_missing = Some((
                                    action.action_idx,
                                    action.action_name.as_str(),
                                    successor_idx,
                                    successor,
                                ));
                                break 'actions;
                            }
                        }
                    }

                    if let Some((action_idx, action_name, successor_idx, missing_successor)) =
                        first_missing
                    {
                        eprintln!(
                            "[compiled-bfs-crosscheck] missing-edge depth={first_depth} parent={parent_idx}: action_idx={action_idx} action={action_name} action_successor_idx={successor_idx} compiled_generated={} compiled_returned={} interpreter_generated={} interpreter_action_counts={interpreter_action_counts:?} parent_flat={parent_slice:?} missing_successor_flat={missing_successor:?}",
                            output_generated_count,
                            compiled_returned,
                            interpreter_generated,
                        );
                        flat_queue.advance_read_cursor(parent_idx + 1);
                        stats.parents_processed += level_parents;
                        stats.successors_generated += level_generated;
                        stats.successors_new += level_new;
                        self.report_compiled_bfs_stats(&stats);
                        return CheckResult::from_error(
                            RuntimeCheckError::Internal(
                                "compiled BFS interpreter crosscheck found a missing successor edge"
                                    .to_string(),
                            )
                            .into(),
                            self.stats.clone(),
                        );
                    }

                    if interpreter_generated != compiled_returned
                        || compiled_successor_matched.iter().any(|matched| !matched)
                    {
                        let first_extra = compiled_successor_matched.iter().enumerate().find_map(
                            |(compiled_idx, matched)| {
                                (!matched)
                                    .then(|| output.successor_at(compiled_idx))
                                    .flatten()
                            },
                        );
                        eprintln!(
                            "[compiled-bfs-crosscheck] extra-edge depth={first_depth} parent={parent_idx}: compiled_generated={} compiled_returned={} interpreter_generated={} interpreter_action_counts={interpreter_action_counts:?} parent_flat={parent_slice:?} extra_successor_flat={first_extra:?}",
                            output_generated_count,
                            compiled_returned,
                            interpreter_generated,
                        );
                        flat_queue.advance_read_cursor(parent_idx + 1);
                        stats.parents_processed += level_parents;
                        stats.successors_generated += level_generated;
                        stats.successors_new += level_new;
                        self.report_compiled_bfs_stats(&stats);
                        return CheckResult::from_error(
                            RuntimeCheckError::Internal(
                                "compiled BFS interpreter crosscheck found an extra compiled successor edge"
                                    .to_string(),
                            )
                            .into(),
                            self.stats.clone(),
                        );
                    }
                }

                let had_raw_successors = output_generated_count > 0;
                if had_raw_successors {
                    self.stats.max_depth = self.stats.max_depth.max(succ_depth);
                }

                // Handle invariant violation from compiled step.
                if !output.invariant_ok() {
                    let Some(inv_idx) = output.failed_invariant_idx() else {
                        flat_queue.advance_read_cursor(parent_idx + 1);
                        stats.parents_processed += level_parents;
                        stats.successors_generated += level_generated;
                        stats.successors_new += level_new;
                        self.report_compiled_bfs_stats(&stats);
                        return CheckResult::from_error(
                            RuntimeCheckError::Internal(
                                "compiled BFS reported invariant failure without failed invariant metadata"
                                    .to_string(),
                            )
                            .into(),
                            self.stats.clone(),
                        );
                    };
                    let (inv_name, action_property_violation) =
                        self.compiled_bfs_failure_name_and_kind(inv_idx);

                    let Some(failed_succ_idx) = output.failed_successor_idx() else {
                        flat_queue.advance_read_cursor(parent_idx + 1);
                        stats.parents_processed += level_parents;
                        stats.successors_generated += level_generated;
                        stats.successors_new += level_new;
                        self.report_compiled_bfs_stats(&stats);
                        return CheckResult::from_error(
                            RuntimeCheckError::Internal(
                                "compiled BFS reported invariant failure without failed successor metadata"
                                    .to_string(),
                            )
                            .into(),
                            self.stats.clone(),
                        );
                    };

                    let Some(succ_slice) = output.successor_at(failed_succ_idx as usize) else {
                        flat_queue.advance_read_cursor(parent_idx + 1);
                        stats.parents_processed += level_parents;
                        stats.successors_generated += level_generated;
                        stats.successors_new += level_new;
                        self.report_compiled_bfs_stats(&stats);
                        return CheckResult::from_error(
                            RuntimeCheckError::Internal(format!(
                                "compiled BFS reported invariant failure successor index {failed_succ_idx} outside {} successors",
                                output.successor_count()
                            ))
                            .into(),
                            self.stats.clone(),
                        );
                    };

                    let (parent_fp, parent_trace_loc) = match flat_queue.meta_at_offset(parent_idx)
                    {
                        Some((fp, _depth, trace_loc)) => (fp, trace_loc),
                        None => {
                            flat_queue.advance_read_cursor(parent_idx + 1);
                            stats.parents_processed += level_parents;
                            stats.successors_generated += level_generated;
                            stats.successors_new += level_new;
                            self.report_compiled_bfs_stats(&stats);
                            return CheckResult::from_error(
                                RuntimeCheckError::Internal(format!(
                                    "compiled BFS invariant failure missing parent metadata at \
                                     index {parent_idx}",
                                ))
                                .into(),
                                self.stats.clone(),
                            );
                        }
                    };
                    let result = self.compiled_bfs_flat_invariant_violation_result(
                        parent_fp,
                        parent_trace_loc,
                        succ_slice,
                        inv_name,
                        action_property_violation,
                        succ_depth,
                        &registry,
                    );

                    flat_queue.advance_read_cursor(parent_idx + 1);
                    stats.parents_processed += level_parents;
                    stats.successors_generated += level_generated;
                    stats.successors_new += level_new;
                    self.report_compiled_bfs_stats(&stats);
                    return result;
                }

                // Process new successors: second-level dedup against global seen set.
                // Clone the bridge to avoid holding an immutable borrow on
                // `self.flat_bfs_adapter` while calling mutable methods below.
                // Part of #3986: Phase 3 zero-alloc compiled BFS.
                let bridge = self
                    .flat_bfs_adapter
                    .as_ref()
                    .map(|adapter| adapter.bridge().clone());
                let (parent_fp, parent_trace_loc) = match flat_queue.meta_at_offset(parent_idx) {
                    Some((fp, _depth, trace_loc)) => (fp, trace_loc),
                    None => {
                        flat_queue.advance_read_cursor(parent_idx + 1);
                        stats.parents_processed += level_parents;
                        stats.successors_generated += level_generated;
                        stats.successors_new += level_new;
                        self.report_compiled_bfs_stats(&stats);
                        return CheckResult::from_error(
                            RuntimeCheckError::Internal(format!(
                                "compiled BFS missing parent metadata at index {parent_idx}",
                            ))
                            .into(),
                            self.stats.clone(),
                        );
                    }
                };
                let mut liveness_successors = self.liveness_cache.cache_for_liveness.then(Vec::new);

                // Reconstruct the parent ArrayState once per parent when a
                // non-native implied action must be evaluated per edge. The
                // parent is constant across this parent's successors, so we do
                // this outside the successor closure. The flat parent buffer is
                // copied to an owned Vec first to release the `flat_queue`
                // borrow before reconstruction (which borrows `&self`).
                let step_parent_array_for_implied = if step_eval_implied_actions_active {
                    let parent_buf = match flat_queue.remaining_state_at_offset(parent_idx) {
                        Some(buf) => buf.to_vec(),
                        None => {
                            flat_queue.advance_read_cursor(parent_idx + 1);
                            stats.parents_processed += level_parents;
                            stats.successors_generated += level_generated;
                            stats.successors_new += level_new;
                            self.report_compiled_bfs_stats(&stats);
                            return CheckResult::from_error(
                                RuntimeCheckError::Internal(format!(
                                    "compiled BFS implied-action eval missing parent buffer at \
                                     index {parent_idx}",
                                ))
                                .into(),
                                self.stats.clone(),
                            );
                        }
                    };
                    match self.try_reconstruct_array_state_from_flat(&parent_buf, &registry) {
                        Ok(parent_array) => Some(parent_array),
                        Err(result) => {
                            flat_queue.advance_read_cursor(parent_idx + 1);
                            stats.parents_processed += level_parents;
                            stats.successors_generated += level_generated;
                            stats.successors_new += level_new;
                            self.report_compiled_bfs_stats(&stats);
                            return result;
                        }
                    }
                } else {
                    None
                };

                let successor_result = output.for_each_successor(|flat_succ| {
                    if use_compiled_fingerprint {
                        let Some(bridge) = bridge.as_ref() else {
                            flat_queue.advance_read_cursor(parent_idx + 1);
                            stats.parents_processed += level_parents;
                            stats.successors_generated += level_generated;
                            stats.successors_new += level_new;
                            self.report_compiled_bfs_stats(&stats);
                            return Err(CheckResult::from_error(
                                RuntimeCheckError::Internal(
                                    "compiled BFS flat fingerprinting requested without \
                                     a FlatBfsAdapter"
                                        .to_string(),
                                )
                                .into(),
                                self.stats.clone(),
                            ));
                        };
                        if let Err(error) = bridge.validate_raw_buffer_slot_count(flat_succ) {
                            flat_queue.advance_read_cursor(parent_idx + 1);
                            stats.parents_processed += level_parents;
                            stats.successors_generated += level_generated;
                            stats.successors_new += level_new;
                            self.report_compiled_bfs_stats(&stats);
                            return Err(self.flat_reconstruction_error_result(
                                "failed to validate compiled BFS successor buffer",
                                error,
                            ));
                        }
                        if bridge.raw_admission_validation_required() {
                            if let Err(error) = bridge.validate_raw_buffer_for_admission(flat_succ)
                            {
                                flat_queue.advance_read_cursor(parent_idx + 1);
                                stats.parents_processed += level_parents;
                                stats.successors_generated += level_generated;
                                stats.successors_new += level_new;
                                self.report_compiled_bfs_stats(&stats);
                                return Err(self.flat_reconstruction_error_result(
                                    "failed to validate compiled BFS successor buffer",
                                    error,
                                ));
                            }
                        }
                    }

                    // Zero-allocation fast path: compute fingerprint directly
                    // from the raw &[i64] buffer without constructing FlatState
                    // (avoids Box<[i64]> heap allocation per successor).
                    // Part of #3986: Phase 3 zero-alloc compiled BFS.
                    // Part of #3987/#4356: Use the same fingerprint domain
                    // selected for queued init states. `flat_state_primary`
                    // can activate compiled-flat fingerprints even when
                    // `jit_compiled_fp_active` is false for fully-flat
                    // non-scalar layouts.
                    let succ_fp = if use_compiled_fingerprint {
                        super::super::invariants::fingerprint_flat_compiled(flat_succ)
                    } else {
                        let Some(bridge) = bridge.as_ref() else {
                            flat_queue.advance_read_cursor(parent_idx + 1);
                            stats.parents_processed += level_parents;
                            stats.successors_generated += level_generated;
                            stats.successors_new += level_new;
                            self.report_compiled_bfs_stats(&stats);
                            return Err(CheckResult::from_error(
                                RuntimeCheckError::Internal(
                                    "compiled BFS traditional fingerprinting requested without \
                                     a FlatBfsAdapter"
                                        .to_string(),
                                )
                                .into(),
                                self.stats.clone(),
                            ));
                        };
                        match bridge.try_traditional_fingerprint_from_buffer(flat_succ, &registry) {
                            Ok(fp) => fp,
                            Err(error) => {
                                flat_queue.advance_read_cursor(parent_idx + 1);
                                stats.parents_processed += level_parents;
                                stats.successors_generated += level_generated;
                                stats.successors_new += level_new;
                                self.report_compiled_bfs_stats(&stats);
                                return Err(self.flat_reconstruction_error_result(
                                    "failed to fingerprint compiled BFS successor buffer",
                                    error,
                                ));
                            }
                        }
                    };

                    // --- Eval-based (non-native) implied actions, per edge ---
                    //
                    // SOUNDNESS ANCHOR (option B): when a non-native implied
                    // action (e.g. `[][B!Next]_B!vars` from an INSTANCE
                    // refinement) must be evaluated by the interpreter, the
                    // native successor generator still produces every
                    // `(parent, successor)` edge here. We surface each such
                    // edge to the SAME validated interpreter hook used by the
                    // full-state path
                    // (`check_eval_implied_actions_for_transition`) BEFORE any
                    // dedup/admission below. This mirrors
                    // full_state_successors.rs:482-502 exactly: fingerprint ->
                    // implied-action hook (gated on a non-stuttering edge) ->
                    // dedup. Because the STEP carrier
                    // (`preserves_state_graph_successor_edges() == true`) never
                    // collapses or batches edges before this closure runs, the
                    // hook is authoritative over every generated edge.
                    //
                    // This path always runs in a *canonical* fingerprint domain
                    // (ArrayFp64): `bfs_fingerprint_domain()` only selects
                    // `CompiledFlat` when no implied action requires interpreter
                    // evaluation, so when `step_eval_implied_actions_active` the
                    // successor fp is the canonicalizing ArrayFp64 fp. We
                    // therefore reconstruct the successor ArrayState once here
                    // and reuse it for (a) the implied-action hook and (b)
                    // canonical duplicate-payload confirmation below. The raw
                    // flat-slot witness used by the compiled-flat STEP path is
                    // unsound in this domain: two distinct raw flat encodings can
                    // represent the SAME logical state (identical ArrayFp64), so
                    // raw-slot comparison would spuriously fail closed with
                    // `canonical_payload_mismatch`. Canonical ArrayState equality
                    // matches the ArrayFp64 admission plan's own check.
                    let step_succ_array_for_implied: Option<crate::state::ArrayState> =
                        if step_eval_implied_actions_active {
                            match self.try_reconstruct_array_state_from_flat(flat_succ, &registry) {
                                Ok(succ_array) => Some(succ_array),
                                Err(result) => {
                                    flat_queue.advance_read_cursor(parent_idx + 1);
                                    stats.parents_processed += level_parents;
                                    stats.successors_generated += level_generated;
                                    stats.successors_new += level_new;
                                    self.report_compiled_bfs_stats(&stats);
                                    return Err(result);
                                }
                            }
                        } else {
                            None
                        };

                    // Part of #3140: skip stuttering transitions (succ == parent).
                    if step_eval_implied_actions_active && succ_fp != parent_fp {
                        let succ_array = step_succ_array_for_implied
                            .as_ref()
                            .expect("step implied-action successor state reconstructed above");
                        // `step_parent_array_for_implied` is `Some` whenever
                        // `step_eval_implied_actions_active` is true (both are
                        // gated by the same predicate), so the parent ArrayState
                        // reconstructed once-per-parent above is available here.
                        let parent_array = step_parent_array_for_implied
                            .as_ref()
                            .expect("step implied-action parent state reconstructed per parent");
                        let outcome = crate::checker_ops::check_eval_implied_actions_for_transition(
                            &mut self.ctx,
                            &step_eval_implied_actions,
                            parent_array,
                            parent_fp,
                            succ_array,
                            succ_fp,
                        );
                        match outcome {
                            crate::checker_ops::InvariantOutcome::Ok
                            | crate::checker_ops::InvariantOutcome::ViolationContinued => {}
                            crate::checker_ops::InvariantOutcome::Violation {
                                invariant, ..
                            } => {
                                // Implied actions come from PROPERTY entries, so
                                // a violation is an action-level PropertyViolation
                                // (matches handle_implied_action_outcome and TLC).
                                let result = self.compiled_bfs_flat_invariant_violation_result(
                                    parent_fp,
                                    parent_trace_loc,
                                    flat_succ,
                                    invariant,
                                    /* action_property_violation = */ true,
                                    succ_depth,
                                    &registry,
                                );
                                flat_queue.advance_read_cursor(parent_idx + 1);
                                stats.parents_processed += level_parents;
                                stats.successors_generated += level_generated;
                                stats.successors_new += level_new;
                                self.report_compiled_bfs_stats(&stats);
                                return Err(result);
                            }
                            crate::checker_ops::InvariantOutcome::Error(error) => {
                                flat_queue.advance_read_cursor(parent_idx + 1);
                                stats.parents_processed += level_parents;
                                stats.successors_generated += level_generated;
                                stats.successors_new += level_new;
                                self.report_compiled_bfs_stats(&stats);
                                return Err(CheckResult::from_error(error, self.stats.clone()));
                            }
                        }
                    }

                    if let Some(ref mut successors) = liveness_successors {
                        successors.push(succ_fp);
                    }

                    // Prepared admission combines the global test-and-set with
                    // duplicate authorization, avoiding a separate hot
                    // contains() probe on the compiled per-parent path.
                    //
                    // Two confirmation channels are valid here:
                    //   1. Self-loop fast path: parent buffer equals successor
                    //      buffer (same fingerprint by construction).
                    //   2. Witness lookup: a previously admitted successor with
                    //      the same fingerprint recorded its payload; compare
                    //      against that witness so cross-parent duplicates (e.g.,
                    //      x=0→x=2 and x=1→x=2) do not fail closed as
                    //      canonical_payload_mismatch.
                    //
                    // For the interpreter-implied-action STEP path the domain is
                    // the canonicalizing ArrayFp64 domain (see comment above), so
                    // confirmation channel 2 uses the *canonical* ArrayState
                    // witness keyed by fp — the reconstructed `succ_array` is
                    // compared by `ArrayState` value equality, exactly as the
                    // ArrayFp64 admission plan does. The raw flat-slot witness is
                    // used only on the compiled-flat STEP path where the raw
                    // buffer is itself canonical.
                    let duplicate_payload_confirmed =
                        if step_eval_implied_actions_active {
                            let succ_array = step_succ_array_for_implied
                                .as_ref()
                                .expect("step implied-action successor state reconstructed above");
                            (parent_fp == succ_fp
                                && step_parent_array_for_implied.as_ref().is_some_and(
                                    |parent_array| parent_array.values() == succ_array.values(),
                                ))
                                || self
                                    .compiled_flat_payload_witness_confirms_array_state(
                                        succ_fp, succ_array,
                                    )
                                    .unwrap_or(false)
                        } else {
                            (parent_fp == succ_fp
                                && flat_queue
                                    .remaining_state_at_offset(parent_idx)
                                    .is_some_and(|parent_buf| parent_buf == flat_succ))
                                || self
                                    .compiled_flat_payload_witness_confirms(succ_fp, flat_succ)
                                    .unwrap_or(false)
                        };
                    let admission_handle = match cached_fingerprint_only_admission_handle(
                        self,
                        &mut fingerprint_only_admission_handle,
                    ) {
                        Ok(handle) => handle,
                        Err(fault) => {
                            flat_queue.advance_read_cursor(parent_idx + 1);
                            stats.parents_processed += level_parents;
                            self.report_compiled_bfs_stats(&stats);
                            return Err(self.storage_fault_result(fault));
                        }
                    };
                    let previous_parent_trace_loc = self.trace.current_parent_trace_loc;
                    self.trace.current_parent_trace_loc = Some(parent_trace_loc);
                    let mark_result = self.mark_state_seen_fp_only_with_prepared_admission_checked(
                        succ_fp,
                        Some(parent_fp),
                        succ_depth,
                        duplicate_payload_confirmed,
                        admission_handle,
                    );
                    self.trace.current_parent_trace_loc = previous_parent_trace_loc;

                    match mark_result {
                        Ok(true) => {}              // Successfully inserted
                        Ok(false) => return Ok(()), // Race condition (shouldn't happen in sequential)
                        Err(result) => {
                            flat_queue.advance_read_cursor(parent_idx + 1);
                            stats.parents_processed += level_parents;
                            self.report_compiled_bfs_stats(&stats);
                            return Err(result);
                        }
                    }

                    level_new += 1;
                    // Record the duplicate-confirmation witness for this fp. The
                    // interpreter-implied-action STEP path runs in the
                    // canonicalizing ArrayFp64 domain, so it records the
                    // *canonical* ArrayState witness (matching its confirmation
                    // channel above); the compiled-flat STEP path records the raw
                    // flat-slot witness (raw == canonical there).
                    if step_eval_implied_actions_active {
                        if let Some(succ_array) = step_succ_array_for_implied.as_ref() {
                            self.record_compiled_flat_payload_witness_array_state_if_absent(
                                succ_fp, succ_array,
                            );
                        }
                    } else {
                        self.record_compiled_flat_payload_witness_if_absent(succ_fp, flat_succ);
                    }
                    self.stats.max_depth = self.stats.max_depth.max(succ_depth);

                    // Get trace_loc from the last inserted state.
                    let trace_loc = self.trace.last_inserted_trace_loc;

                    // Zero-allocation enqueue: push raw buffer directly into
                    // the arena without constructing FlatState.
                    // Part of #3986: Phase 3 zero-alloc compiled BFS.
                    flat_queue.push_raw_buffer(flat_succ, succ_fp, succ_depth, trace_loc);
                    Ok(())
                });
                if let Err(result) = successor_result {
                    return result;
                }

                if let Some(successors) = liveness_successors {
                    if let Err(error) = self.liveness_cache.successors.insert(parent_fp, successors)
                    {
                        flat_queue.advance_read_cursor(parent_idx + 1);
                        stats.parents_processed += level_parents;
                        stats.successors_generated += level_generated;
                        stats.successors_new += level_new;
                        self.report_compiled_bfs_stats(&stats);
                        return CheckResult::from_error(error, self.stats.clone());
                    }
                }

                // Deadlock detection: if no successors were generated.
                if self.exploration.check_deadlock && !had_raw_successors {
                    let state_result = match flat_queue.remaining_state_at_offset(parent_idx) {
                        Some(parent_slice) => {
                            self.try_reconstruct_state_from_flat(parent_slice, &registry)
                        }
                        None => Err(CheckResult::from_error(
                            RuntimeCheckError::Internal(
                                "compiled BFS deadlock reported without a parent state".to_string(),
                            )
                            .into(),
                            self.stats.clone(),
                        )),
                    };
                    flat_queue.advance_read_cursor(parent_idx + 1);
                    stats.parents_processed += level_parents;
                    stats.successors_generated += level_generated;
                    stats.successors_new += level_new;
                    let state = match state_result {
                        Ok(state) => state,
                        Err(result) => {
                            self.report_compiled_bfs_stats(&stats);
                            return result;
                        }
                    };
                    self.report_compiled_bfs_stats(&stats);

                    // Reconstruct the full parent chain ending at the deadlocked
                    // state, mirroring the interpreter path (run_helpers.rs
                    // check_deadlock -> reconstruct_trace(fp)). `parent_fp` (from
                    // meta_at_offset(parent_idx) at the top of this loop body) is the
                    // deadlocked state's own fingerprint, already admitted to the
                    // trace file via mark_state_seen_fp_only_*, so reconstruct_trace
                    // walks back to Init. Fall back to the single state when no trace
                    // file is configured (reconstruct_trace returns an empty Trace).
                    let reconstructed = self.reconstruct_trace(parent_fp);
                    let trace = if reconstructed.is_empty() {
                        Trace::from_states(vec![state])
                    } else {
                        reconstructed
                    };

                    return CheckResult::Deadlock {
                        trace,
                        stats: self.stats.clone(),
                    };
                }

                // Part of #4203: Update state_count after each parent so that
                // progress reporting, memory pressure checks, and state limit
                // checks see fresh data during level processing — not just at
                // level-end. This matches the per-dequeue update pattern in the
                // standard BFS worker loop (transport_seq.rs report_progress).
                self.stats.states_found = self.states_count();

                // Part of #4203: Periodic memory/progress check within the level.
                // Without this, memory pressure is only checked at level
                // boundaries, which can be arbitrarily far apart on wide
                // frontiers. Check every COMPILED_BFS_PROGRESS_INTERVAL parents.
                if level_parents % COMPILED_BFS_PROGRESS_INTERVAL == 0 {
                    if let Some(ref policy) = self.exploration.memory_policy {
                        use crate::memory::MemoryPressure;
                        if policy.check() == MemoryPressure::Critical {
                            let rss_mb = crate::memory::current_rss_bytes()
                                .map(|b| b / (1024 * 1024))
                                .unwrap_or(0);
                            let limit_mb = policy.limit_bytes() / (1024 * 1024);
                            eprintln!(
                                "[compiled-bfs] memory critical ({rss_mb} MB / {limit_mb} MB limit) \
                                 at parent {parent_idx} of level — stopping."
                            );
                            flat_queue.advance_read_cursor(parent_idx + 1);
                            stats.parents_processed += level_parents;
                            stats.successors_generated += level_generated;
                            stats.successors_new += level_new;
                            self.report_compiled_bfs_stats(&stats);
                            return CheckResult::LimitReached {
                                limit_type: super::super::LimitType::Memory,
                                stats: self.stats.clone(),
                            };
                        }
                    }
                }
            }

            // Advance past all parents we processed.
            flat_queue.advance_read_cursor(consumed_count);
            stats.parents_processed += level_parents;
            stats.successors_generated += level_generated;
            stats.successors_new += level_new;
            // A sub-batched level completes only on its final sub-batch; count one
            // level per BFS depth (telemetry). `early_break` (state-limit) never
            // co-occurs with sub-batching (disabled under a state limit).
            if is_final_subbatch_of_level {
                stats.levels_completed += 1;
            }

            // Update model checker stats.
            let total_states = self.states_count();
            self.stats.states_found = total_states;

            // Progress reporting.
            if crate::check::debug::compiled_bfs_progress_enabled()
                && stats.levels_completed % 5 == 0
            {
                eprintln!(
                    "[compiled-bfs] level {}: {} states, {} generated, {} new, queue={}",
                    stats.levels_completed,
                    total_states,
                    stats.successors_generated,
                    stats.successors_new,
                    flat_queue.len(),
                );
            }

            // Arena reset at level boundaries.
            crate::arena::worker_arena_reset();

            if early_break {
                break;
            }
        }

        self.report_compiled_bfs_stats(&stats);
        // The compiled loop can also stop for a state limit or because another
        // portfolio lane resolved the verdict. Release only after the explicit
        // empty-frontier branch, matching the standard worker's exhaustion bit.
        if super::release_compiled_payload_witnesses_after_terminal_bfs(
            &mut self.state_storage.compiled_flat_payload_witnesses,
            frontier_exhausted,
            limit_reached,
        ) {
            storage.release_after_complete_bfs();
            flat_queue.release_after_complete_bfs();
        }
        let active_payload_witness_bytes = storage.payload_witness_memory_bytes();
        let result =
            self.finish_check_after_bfs(limit_reached, false, active_payload_witness_bytes);

        // Publish verdict to portfolio/cooperative.
        if let Some(ref sv) = self.portfolio_verdict {
            let verdict = match &result {
                CheckResult::Success(_) => Verdict::Satisfied,
                CheckResult::InvariantViolation { .. }
                | CheckResult::PropertyViolation { .. }
                | CheckResult::LivenessViolation { .. } => Verdict::Violated,
                _ => Verdict::Unknown,
            };
            sv.publish(verdict);
        }
        #[cfg(feature = "ay")]
        if let Some(ref coop) = self.cooperative {
            let verdict = match &result {
                CheckResult::Success(_) => Verdict::Satisfied,
                CheckResult::InvariantViolation { .. }
                | CheckResult::PropertyViolation { .. }
                | CheckResult::LivenessViolation { .. } => Verdict::Violated,
                _ => Verdict::Unknown,
            };
            coop.verdict.publish(verdict);
            coop.mark_bfs_complete();
        }

        result
    }

    /// Check whether the compiled BFS level loop is eligible for this run.
    ///
    /// Part of #3988: JIT V2 Phase 5 compiled BFS step.
    /// Part of #4171: End-to-end compiled BFS wiring (config/env controls).
    ///
    /// Also consulted by `should_defer_fused_level_build` (lever L1): the
    /// deferred fused-level promotion point lives inside
    /// `run_compiled_bfs_loop`, so deferral is only meaningful when this
    /// predicate holds at setup.
    #[must_use]
    pub(in crate::check::model_checker) fn compiled_bfs_level_eligible(&self) -> bool {
        macro_rules! dbg_veto {
            ($why:expr) => {
                if std::env::var_os("TY_DEBUG_LEVEL_ELIGIBLE").is_some() {
                    eprintln!("[level-eligible-debug] veto: {}", $why);
                }
            };
        }
        // Force-disable via config or env var.
        if self.config.use_compiled_bfs == Some(false) {
            dbg_veto!("use_compiled_bfs=false");
            return false;
        }
        if crate::check::debug::compiled_bfs_disabled() {
            dbg_veto!("compiled_bfs_disabled env");
            return false;
        }
        if !self.compiled_bfs_flat_frontier_admitted() {
            dbg_veto!("flat frontier not admitted");
            return false;
        }
        if self.compiled_bfs_step.is_none() && !self.fused_bfs_level_available() {
            dbg_veto!("no step and no fused level");
            return false;
        }
        if self.jit_monolithic_disabled {
            dbg_veto!("jit_monolithic_disabled");
            return false;
        }
        if self.implied_actions_require_interpreter_eval()
            && !self.compiled_bfs_step_evaluates_interpreter_implied_actions()
        {
            // Non-native implied actions require interpreter per-transition
            // evaluation. They normally disable the compiled BFS loop entirely.
            // The exception is the per-parent STEP path with full edge
            // preservation: the loop's STEP emit closure routes every
            // (parent, successor) edge through
            // `check_eval_implied_actions_for_transition` before dedup, so the
            // property stays checked on every edge. The fused LEVEL is not
            // admitted in that configuration (it may locally dedup edges), as
            // guaranteed by `compiled_bfs_step_evaluates_interpreter_implied_actions`.
            dbg_veto!("implied actions require interpreter eval");
            return false;
        }
        if !self.config.action_constraints.is_empty() {
            dbg_veto!("action constraints");
            return false;
        }
        if self.config.terminal.is_some() {
            // Compiled BFS deadlock reporting does not evaluate TerminalSpec:
            // both compiled deadlock sites (the fused-level metadata branch and
            // the per-parent "no successors" branch) return
            // `CheckResult::Deadlock` directly, without the
            // `is_terminal_state_array` exemption the interpreter path applies
            // in `check_deadlock` (run_helpers). A genuine terminal state would
            // therefore be misreported as a deadlock. Route TERMINAL configs to
            // the interpreter loop, which evaluates the terminal predicate.
            dbg_veto!("terminal spec");
            return false;
        }
        if !self.config.constraints.is_empty()
            && !self.state_constrained_native_fused_admission_active()
        {
            dbg_veto!("state constraints without native fused admission");
            return false;
        }
        if self.por.independence.is_some() {
            dbg_veto!("POR independence");
            return false;
        }
        if self.coverage.collect && !self.coverage.actions.is_empty() {
            dbg_veto!("coverage collection");
            return false;
        }
        if !self.config.trace_invariants.is_empty() {
            dbg_veto!("trace invariants");
            return false;
        }
        if !self.symmetry.perms.is_empty() && !self.flat_symmetry_native_veto_relaxed() {
            // WP-11 slice 2 veto relaxation: symmetry runs may enter the
            // compiled loop ONLY under the flat-symmetry admission token AND
            // the (still fail-closed) native canonicalization hook — see
            // `flat_symmetry_native_veto_relaxed` for why this stays vetoed
            // until the loop's successor hashes route through the
            // canonicalizer. Every other path keeps the unconditional veto.
            dbg_veto!("symmetry perms");
            return false;
        }
        if self.compiled.cached_view_name.is_some() {
            dbg_veto!("cached view");
            return false;
        }
        if self.inline_liveness_active() {
            dbg_veto!("inline liveness");
            return false;
        }
        true
    }

    /// Parent stride for the default-on SAMPLED fused-arena native↔interpreter equivalence check
    /// (P0), or `None` when disabled. ON by default for native runs (batteries-on); opt out via
    /// `TY_NATIVE_EQUIV_CHECK=0/off/false`. The explicit per-state crosscheck env forces stride 1.
    fn fused_native_equiv_stride(&self) -> Option<usize> {
        if std::env::var_os(COMPILED_BFS_INTERPRETER_CROSSCHECK_ENV).is_some() {
            return Some(1);
        }
        match std::env::var("TY_NATIVE_EQUIV_CHECK") {
            Ok(v) => {
                let v = v.trim();
                if v == "0" || v.eq_ignore_ascii_case("off") || v.eq_ignore_ascii_case("false") {
                    None
                } else {
                    Some(FUSED_NATIVE_EQUIV_SAMPLE_STRIDE)
                }
            }
            Err(_) => Some(FUSED_NATIVE_EQUIV_SAMPLE_STRIDE),
        }
    }

    /// Pure comparison core of the fused-arena native↔interpreter equivalence check.
    ///
    /// `successors` lists `(parent_index, successor_state)` for each successor the native fused
    /// arena produced; `expected` maps each SAMPLED parent index to the interpreter's successor
    /// set for that parent. Returns `Some(reason)` on a SOUND divergence:
    /// - EXTRA-EDGE (always sound, even under cross-parent dedup): every native successor whose
    ///   parent was sampled must be an interpreter successor of that parent. A native successor
    ///   the interpreter can't produce is a miscompile (it would over-approximate the state space
    ///   and could report a spurious violation).
    /// - MISSING-EDGE (only sound when `graph_complete` — no cross-parent dedup): every
    ///   interpreter successor must appear in the native set for that parent. Skipped when the
    ///   arena deduplicated successors across parents, since a real edge may legitimately be
    ///   attributed to a different parent.
    ///
    /// Returns `None` (no divergence) when attribution is unreliable, so the check never
    /// false-positives on a backend that did not supply a complete parent sidecar.
    ///
    /// Two audit notes on what this still covers:
    /// - SOUNDNESS_001 (state/action constraints): the interpreter `expected` set is the
    ///   constraint-FILTERED successor set. So if the native arena emits a successor that
    ///   violates a state/action constraint (which the interpreter would have filtered), it is
    ///   not in `expected` and the EXTRA-EDGE check flags it — exactly the over-approximation we
    ///   must catch. (A native backend that legitimately defers constraints to admission would
    ///   trip the same path, costing only a sound interpreter fallback, never correctness.)
    /// - SOUNDNESS_003 (trusting `graph_complete`): a backend that falsely claims `complete` while
    ///   it actually deduped would make a real edge "missing" for its parent, which the
    ///   MISSING-EDGE check flags → sound fallback. A backend that falsely claims INcomplete only
    ///   suppresses the missing-edge direction (coverage loss, never unsoundness). Either way a
    ///   wrong completeness flag cannot turn a real divergence into a false pass.
    fn fused_arena_native_equiv_divergence(
        successors: &[(usize, Vec<i64>)],
        attribution_complete: bool,
        graph_complete: bool,
        expected: &std::collections::BTreeMap<usize, std::collections::BTreeSet<Vec<i64>>>,
    ) -> Option<String> {
        if !attribution_complete {
            return None;
        }
        let mut native_by_parent: std::collections::BTreeMap<
            usize,
            std::collections::BTreeSet<Vec<i64>>,
        > = std::collections::BTreeMap::new();
        for (parent, succ) in successors {
            let Some(exp) = expected.get(parent) else {
                continue; // parent not in this round's sample
            };
            if !exp.contains(succ) {
                return Some(format!(
                    "native produced a successor not interpreter-reachable from sampled parent {parent}"
                ));
            }
            // The native successor IS interpreter-reachable: capture the first single-var nonneg
            // sample so the native lane can be KERNEL-certified offline (native_membership_cert),
            // upgrading this sampled crosscheck from runtime-trusted to kernel-proven. Cheap
            // (first-wins; the build is bounded to the sampled successors and gated to clean-cic).
            //
            // SCOPE NOTE: this `succ.len() == 1 && succ[0] >= 0` guard (and the matching filter on
            // `exp` below) is a CERTIFICATION-scope filter ONLY. Multi-variable or negative-valued
            // successors are still fully covered by the runtime crosscheck above (`exp.contains(succ)`)
            // — they are merely NOT kernel-certified. So the emitted "native lane KERNEL-CERTIFIED"
            // attests the nonneg single-var PROJECTION of `exp`, not full native-lane coverage.
            #[cfg(feature = "clean-cic")]
            if succ.len() == 1 && succ[0] >= 0 && crate::check::debug::native_lane_cert_enabled() {
                let interp_vals: Vec<u64> = exp
                    .iter()
                    .filter_map(|v| (v.len() == 1 && v[0] >= 0).then_some(v[0] as u64))
                    .collect();
                crate::cleancic::capture_native_sample(succ[0] as u64, interp_vals);
            }
            native_by_parent
                .entry(*parent)
                .or_default()
                .insert(succ.clone());
        }
        if graph_complete {
            for (parent, exp) in expected {
                let native = native_by_parent.get(parent);
                for edge in exp {
                    if native.map_or(true, |n| !n.contains(edge)) {
                        return Some(format!(
                            "interpreter successor missing from native for sampled parent {parent} (complete graph)"
                        ));
                    }
                }
            }
        }
        None
    }

    /// P0 — default-on SAMPLED native↔interpreter equivalence on the FUSED arena fast path
    /// (the common all-scalar `Prototype`/JIT path, which has no per-parent step).
    ///
    /// Phase 1 (needs `&mut self.ctx`, run BEFORE any fused-level borrow): enumerate the
    /// interpreter's successor set for a stride-sample of this level's parents. Phase 2: PEEK-run
    /// the fused arena — `remaining_arena()` only peeks; the read cursor advances at admission,
    /// not here — collect native successors per parent into OWNED data, release the level borrow,
    /// and compare via [`Self::fused_arena_native_equiv_divergence`]. Returns `true` on a sound
    /// divergence; the caller then fails CLOSED by recomputing this level via the interpreter-
    /// backed loop (the peek did not advance the cursor, so that fallback re-sees every parent —
    /// SOUND and NON-BREAKING). Bounded cost: one extra arena run + ~ceil(parents/stride)
    /// interpreter enumerations, only on sampled levels.
    fn fused_arena_native_equiv_diverged(
        &mut self,
        flat_queue: &FlatBfsFrontier,
        parent_count: usize,
        stride: usize,
    ) -> bool {
        if self.compiled_bfs_level.is_none() || parent_count == 0 {
            return false;
        }
        // FN_001: a level too wide for the whole-frontier peek-run is SKIPPED for cost — but
        // logged, so it is never confused with "checked and clean": its native execution is
        // simply not oracle-validated this run (the native CODE is still validated on the
        // earlier, smaller levels, which is the point of the level-invariance bound).
        //
        // Explicit opt-in override: `TY_COMPILED_BFS_INTERPRETER_CROSSCHECK` is the
        // "exact, every-parent" validation mode (stride 1); when it is set the caller has
        // deliberately accepted the whole-frontier interpreter re-run cost, so honor it even
        // on a wide level (e.g. GameOfLife's single 65536-parent level, which is otherwise
        // above the cost cap). Zero effect on normal runs (the env is unset).
        if parent_count > FUSED_NATIVE_EQUIV_MAX_LEVEL_PARENTS
            && std::env::var_os(COMPILED_BFS_INTERPRETER_CROSSCHECK_ENV).is_none()
        {
            eprintln!(
                "[compiled-bfs-crosscheck] level too wide ({parent_count} parents > {}) — fused-arena \
                 native↔interpreter crosscheck SKIPPED for cost (this level not oracle-validated)",
                FUSED_NATIVE_EQUIV_MAX_LEVEL_PARENTS
            );
            return false;
        }
        let registry = self.ctx.var_registry().clone();
        // `stride == 1` (the explicit per-state crosscheck env) means full coverage; otherwise
        // sample ~FUSED_NATIVE_EQUIV_TARGET_SAMPLES parents evenly across the level (all of them
        // on the small early levels this check targets).
        let stride = if stride <= 1 {
            1
        } else {
            (parent_count / FUSED_NATIVE_EQUIV_TARGET_SAMPLES).max(1)
        };

        // Phase 1: interpreter expected successors for the sampled parents.
        let mut expected: std::collections::BTreeMap<usize, std::collections::BTreeSet<Vec<i64>>> =
            std::collections::BTreeMap::new();
        let mut idx = 0usize;
        while idx < parent_count {
            let Some(parent) = flat_queue
                .remaining_state_at_offset(idx)
                .map(<[i64]>::to_vec)
            else {
                break;
            };
            match self.compiled_bfs_interpreter_action_successors_for_parent(&parent, &registry) {
                Ok(actions) => {
                    let set: std::collections::BTreeSet<Vec<i64>> = actions
                        .iter()
                        .flat_map(|a| a.iter_successors().map(<[i64]>::to_vec))
                        .collect();
                    expected.insert(idx, set);
                }
                // Batteries-on / FAIL-CLOSED: if the interpreter oracle cannot enumerate a SAMPLED
                // parent we cannot certify native≡interpreter for it, so fall back to the
                // interpreter-backed loop rather than silently trusting native (FALSE_NEGATIVE_002).
                Err(_) => {
                    eprintln!(
                        "[compiled-bfs-crosscheck] interpreter could not enumerate sampled parent \
                         {idx} — failing closed to the interpreter-backed loop"
                    );
                    return true;
                }
            }
            idx = idx.saturating_add(stride);
        }
        if expected.is_empty() {
            return false;
        }
        let sampled = expected.len();

        // Phase 2: peek-run the fused arena, collect owned native successors, compare.
        let divergence = {
            let Some(level) = self.compiled_bfs_level.as_ref() else {
                return false;
            };
            let Some((arena, count)) = flat_queue.remaining_arena() else {
                return false;
            };
            if count < parent_count {
                return false;
            }
            // Slice to exactly `parent_count` states: the native parent-arena ABI
            // requires an exact-length slice, and `remaining_arena()` may be wider
            // than the (sub-)batch being crosschecked.
            let slots = arena.len() / count;
            let arena = &arena[..parent_count * slots];
            let level_result = match level.run_level_fused_arena(arena, parent_count) {
                Some(Ok(result)) => result,
                // SOUNDNESS_002: the peek-run could not produce a level result — skip (the real
                // run hits the same path and handles the error). Logged so the skip is not silent.
                _ => {
                    eprintln!(
                        "[compiled-bfs-crosscheck] fused-arena peek-run produced no level result — \
                         crosscheck skipped for this level"
                    );
                    return false;
                }
            };
            let graph_complete = level_result.state_graph_successors_complete();
            let parent_indices = level_result.successor_parent_indices();
            let attribution_complete =
                level_result.successor_parent_indices_complete() && parent_indices.is_some();
            let mut successors: Vec<(usize, Vec<i64>)> = Vec::new();
            let mut misaligned = false;
            if let Some(parent_indices) = parent_indices.as_ref() {
                for i in 0..level_result.successor_count() {
                    match (parent_indices.get(i), level_result.successor_at(i)) {
                        (Some(parent), Some(succ)) => successors.push((parent, succ.to_vec())),
                        // Attribution claimed complete but an index is missing -> backend arena
                        // inconsistency; fail closed rather than silently dropping the successor
                        // (INCOMPLETENESS_001).
                        _ => {
                            misaligned = true;
                            break;
                        }
                    }
                }
            }
            if attribution_complete && misaligned {
                Some(
                    "native arena successor/parent-index sidecar misaligned (count vs reconstructed)"
                        .to_string(),
                )
            } else {
                Self::fused_arena_native_equiv_divergence(
                    &successors,
                    attribution_complete,
                    graph_complete,
                    &expected,
                )
            }
        };

        match divergence {
            Some(reason) => {
                eprintln!(
                    "[compiled-bfs-crosscheck] fused arena native↔interpreter DIVERGENCE \
                     ({sampled} parent(s) sampled): {reason} — failing closed to the \
                     interpreter-backed loop"
                );
                true
            }
            None => {
                telemetry_eprintln!(
                    "[compiled-bfs-xcheck] active: sampled {sampled} fused-arena parent(s) — native \
                     matched interpreter (extra-edge; missing-edge when graph complete)"
                );
                false
            }
        }
    }

    /// P0 (per-parent path) — default-on SAMPLED native↔interpreter equivalence for the per-parent
    /// compiled-step native path (used when there is no fused level). Mirrors
    /// [`Self::fused_arena_native_equiv_diverged`] but drives `step_flat_scoped` per parent: the
    /// per-parent step performs NO cross-parent dedup, so the comparison is EXACT in both directions
    /// (every native successor is an interpreter successor and vice-versa). Returns `true` on a sound
    /// divergence — or, FAIL-CLOSED, if the interpreter oracle or the native step cannot be evaluated
    /// for a sampled parent — and the caller then recomputes the level via the interpreter loop.
    fn fused_step_native_equiv_diverged(
        &mut self,
        flat_queue: &FlatBfsFrontier,
        parent_count: usize,
        stride: usize,
    ) -> bool {
        if self.compiled_bfs_step.is_none() || parent_count == 0 {
            return false;
        }
        // FN_001: too-wide level skipped for cost, but logged (not silently "clean").
        if parent_count > FUSED_NATIVE_EQUIV_MAX_LEVEL_PARENTS {
            eprintln!(
                "[compiled-bfs-crosscheck] level too wide ({parent_count} parents > {}) — per-parent \
                 step native↔interpreter crosscheck SKIPPED for cost (this level not oracle-validated)",
                FUSED_NATIVE_EQUIV_MAX_LEVEL_PARENTS
            );
            return false;
        }
        let registry = self.ctx.var_registry().clone();
        let stride = if stride <= 1 {
            1
        } else {
            (parent_count / FUSED_NATIVE_EQUIV_TARGET_SAMPLES).max(1)
        };

        // Phase 1: interpreter expected successors for the sampled parents (&mut self).
        let mut sampled: Vec<(usize, Vec<i64>, std::collections::BTreeSet<Vec<i64>>)> = Vec::new();
        let mut idx = 0usize;
        while idx < parent_count {
            let Some(parent) = flat_queue
                .remaining_state_at_offset(idx)
                .map(<[i64]>::to_vec)
            else {
                break;
            };
            match self.compiled_bfs_interpreter_action_successors_for_parent(&parent, &registry) {
                Ok(actions) => {
                    let set: std::collections::BTreeSet<Vec<i64>> = actions
                        .iter()
                        .flat_map(|a| a.iter_successors().map(<[i64]>::to_vec))
                        .collect();
                    sampled.push((idx, parent, set));
                }
                Err(_) => {
                    eprintln!(
                        "[compiled-bfs-crosscheck] interpreter could not enumerate sampled parent \
                         {idx} (per-parent path) — failing closed to the interpreter-backed loop"
                    );
                    return true;
                }
            }
            idx = idx.saturating_add(stride);
        }
        if sampled.is_empty() {
            return false;
        }
        let count = sampled.len();

        // Phase 2: native per-parent step for each sampled parent, EXACT set compare.
        let step = self
            .compiled_bfs_step
            .as_ref()
            .expect("compiled_bfs_step present (checked above)");
        let mut scratch = CompiledBfsStepScratch::new(step.state_len());
        for (pidx, parent, expected) in &sampled {
            let native: std::collections::BTreeSet<Vec<i64>> =
                match step.step_flat_scoped(parent, &mut scratch) {
                    Ok(out) => (0..out.successor_count())
                        .filter_map(|i| out.successor_at(i).map(<[i64]>::to_vec))
                        .collect(),
                    // Native step error — fail closed.
                    Err(_) => return true,
                };
            if &native != expected {
                eprintln!(
                    "[compiled-bfs-crosscheck] per-parent step native↔interpreter DIVERGENCE at \
                     parent={pidx}: native_successors={} interpreter_successors={} — failing closed \
                     to the interpreter-backed loop",
                    native.len(),
                    expected.len()
                );
                return true;
            }
        }
        telemetry_eprintln!(
            "[compiled-bfs-xcheck] active: sampled {count} per-parent native step(s) — native \
             matched interpreter (exact)"
        );
        false
    }

    /// Whether the fused BFS level function is available and should be used.
    ///
    /// The fused path processes entire frontiers in a single native call,
    /// eliminating per-parent Rust-to-JIT boundary crossings. Falls back to
    /// the per-parent `CompiledBfsStep` path when unavailable.
    ///
    /// Part of #4171: End-to-end compiled BFS wiring.
    #[must_use]
    fn fused_bfs_level_available(&self) -> bool {
        if !self.config.trace_invariants.is_empty() {
            return false;
        }
        self.compiled_bfs_level
            .as_ref()
            .is_some_and(|level| level.has_fused_level())
    }

    /// Whether the fused BFS level is backed by a native generated parent loop.
    #[must_use]
    fn native_fused_bfs_level_available(&self) -> bool {
        if !self.config.trace_invariants.is_empty() {
            return false;
        }
        self.compiled_bfs_level
            .as_ref()
            .is_some_and(|level| level.has_native_fused_level())
    }

    /// Whether the per-parent compiled step can be used without corrupting
    /// liveness state-graph capture in the current checker mode.
    #[must_use]
    fn compiled_bfs_step_usable_for_current_graph_mode(&self) -> bool {
        if !self.config.constraints.is_empty() {
            return false;
        }
        self.compiled_bfs_step.as_ref().is_some_and(|step| {
            !self.liveness_cache.cache_for_liveness || step.preserves_state_graph_successor_edges()
        })
    }

    fn disable_compiled_bfs_for_standard_fallback(&mut self) {
        self.compiled_bfs_level = None;
        self.compiled_bfs_step = None;
        // The interpreter loop has no fused-promotion point; drop any pending
        // deferred fused-level build along with the compiled artifacts. The
        // interpreter loop performs its own invariant checks, so any pending
        // deferred invariant re-fusion is moot here too.
        self.deferred_fused_level_build = false;
        self.deferred_fused_invariant_build = false;
    }

    fn insert_compiled_bfs_liveness_successors_for_level(
        &mut self,
        flat_queue: &FlatBfsFrontier,
        parent_count: usize,
        liveness_successors: CompiledBfsLivenessSuccessors,
    ) -> Result<(), crate::CheckError> {
        liveness_successors.try_for_each_parent_successors(
            parent_count,
            |parent_idx, successors| {
                let Some((parent_fp, _depth, _trace_loc)) = flat_queue.meta_at_offset(parent_idx)
                else {
                    return Err(RuntimeCheckError::Internal(format!(
                        "fused compiled BFS state-graph capture missing parent metadata at index \
                     {parent_idx}",
                    ))
                    .into());
                };
                self.liveness_cache.successors.insert(parent_fp, successors)
            },
        )
    }

    #[must_use]
    fn trust_cg_native_fused_strict_enabled(&self) -> bool {
        crate::check::debug::trust_cg_native_fused_strict()
    }

    fn strict_native_fused_fallback_result(&self, reason: impl std::fmt::Display) -> CheckResult {
        CheckResult::from_error(
            RuntimeCheckError::Internal(format!(
                "strict trust-codegen native fused requirement failed: {reason}"
            ))
            .into(),
            self.stats.clone(),
        )
    }

    /// Reconstruct a TLA+ `State` from a flat i64 buffer.
    ///
    /// Uses the `FlatBfsAdapter` to convert flat -> ArrayState -> State.
    /// This is the cold path used only for error reporting (invariant
    /// violations, deadlock).
    ///
    /// Part of #3988.
    fn flat_reconstruction_error_result(
        &self,
        context: &str,
        error: impl std::fmt::Display,
    ) -> CheckResult {
        CheckResult::from_error(
            RuntimeCheckError::Internal(format!("{context}: {error}")).into(),
            self.stats.clone(),
        )
    }

    fn compiled_bfs_flat_invariant_violation_result(
        &mut self,
        parent_fp: Fingerprint,
        parent_trace_loc: u64,
        failed_successor: &[i64],
        invariant: String,
        action_property_violation: bool,
        succ_depth: usize,
        registry: &crate::var_index::VarRegistry,
    ) -> CheckResult {
        let mut succ_array =
            match self.try_reconstruct_array_state_from_flat(failed_successor, registry) {
                Ok(succ_array) => succ_array,
                Err(result) => return result,
            };
        let succ_state = succ_array.to_state(registry);
        let succ_fp = match self.array_state_fingerprint(&mut succ_array) {
            Ok(fp) => fp,
            Err(error) => return CheckResult::from_error(error, self.stats.clone()),
        };

        let previous_parent_trace_loc = self.trace.current_parent_trace_loc;
        self.trace.current_parent_trace_loc = Some(parent_trace_loc);
        let stage_result =
            self.stage_successor_for_terminal_trace(parent_fp, &succ_array, succ_fp, succ_depth);
        self.trace.current_parent_trace_loc = previous_parent_trace_loc;
        if let Err(result) = stage_result {
            return self.finalize_terminal_result_with_storage(result);
        }

        self.stats.max_depth = self.stats.max_depth.max(succ_depth);
        self.stats.states_found = self.states_count();
        if action_property_violation {
            let _should_stop = self.record_action_property_violation(invariant.clone(), succ_fp);
        } else {
            let _should_stop = self.record_invariant_violation(invariant.clone(), succ_fp);
        }
        self.update_coverage_totals();

        let trace =
            self.reconstruct_trace_for_staged_compiled_successor(parent_fp, succ_fp, succ_state);
        let candidate = if action_property_violation {
            CheckResult::PropertyViolation {
                property: invariant,
                kind: crate::check::api::PropertyViolationKind::ActionLevel,
                trace,
                stats: self.stats.clone(),
            }
        } else if self
            .compiled
            .state_property_violation_names
            .contains(&invariant)
        {
            CheckResult::PropertyViolation {
                property: invariant,
                kind: crate::check::api::PropertyViolationKind::StateLevel,
                trace,
                stats: self.stats.clone(),
            }
        } else {
            CheckResult::InvariantViolation {
                invariant,
                trace,
                stats: self.stats.clone(),
            }
        };
        self.finalize_terminal_result(candidate)
    }

    fn compiled_bfs_failure_name_and_kind(&self, idx: u32) -> (String, bool) {
        let idx = idx as usize;
        if let Some(name) = self.config.invariants.get(idx) {
            return (name.clone(), false);
        }
        let implied_idx = idx.saturating_sub(self.config.invariants.len());
        self.compiled
            .native_implied_actions
            .get(implied_idx)
            .map(|term| (term.name.clone(), true))
            .unwrap_or_else(|| (format!("invariant_{idx}"), false))
    }

    fn reconstruct_trace_for_staged_compiled_successor(
        &mut self,
        parent_fp: Fingerprint,
        succ_fp: Fingerprint,
        succ_state: crate::State,
    ) -> Trace {
        let trace = self.reconstruct_trace(succ_fp);
        if parent_fp == succ_fp || trace.len() > 1 {
            if trace.is_empty() {
                return Trace::from_states(vec![succ_state]);
            }
            return trace;
        }

        let parent_trace = self.reconstruct_trace(parent_fp);
        if parent_trace.is_empty() {
            if trace.is_empty() {
                return Trace::from_states(vec![succ_state]);
            }
            return trace;
        }

        let mut states = parent_trace.states;
        states.push(succ_state);
        let labels = self.identify_action_labels(&states);
        Trace::from_states_with_labels(states, labels)
    }

    fn record_fp_only_batch_admission_bookkeeping_for_indices(
        &mut self,
        flat_queue: &FlatBfsFrontier,
        parent_indices: CompiledSuccessorParentIndices<'_>,
        fingerprint_values: &[u64],
        inserted_indices: &[usize],
        depth: usize,
        raw_inserted_successors: &mut Vec<(usize, Fingerprint, usize, u64)>,
    ) -> Result<(), CheckResult> {
        debug_assert!(!self.state_storage.store_full_states);

        raw_inserted_successors.clear();
        raw_inserted_successors.reserve(inserted_indices.len());

        let mut trace_records = self
            .trace
            .trace_file
            .is_some()
            .then(|| Vec::with_capacity(inserted_indices.len()));
        let checkpoint_active = self.checkpoint.dir.is_some();

        for &successor_idx in inserted_indices {
            let Some(&fp_value) = fingerprint_values.get(successor_idx) else {
                return Err(CheckResult::from_error(
                    RuntimeCheckError::Internal(format!(
                        "compiled BFS batch bookkeeping received inserted index \
                         {successor_idx} for {} fingerprints",
                        fingerprint_values.len(),
                    ))
                    .into(),
                    self.stats.clone(),
                ));
            };
            let succ_fp = Fingerprint(fp_value);
            let Some(parent_idx) = parent_indices.get(successor_idx) else {
                return Err(CheckResult::from_error(
                    RuntimeCheckError::Internal(
                        "compiled BFS batch bookkeeping missing successor parent metadata"
                            .to_string(),
                    )
                    .into(),
                    self.stats.clone(),
                ));
            };
            let Some((_parent_fp, _depth, parent_trace_loc)) =
                flat_queue.meta_at_offset(parent_idx)
            else {
                return Err(CheckResult::from_error(
                    RuntimeCheckError::Internal(format!(
                        "compiled BFS batch bookkeeping missing parent metadata at index \
                         {parent_idx}",
                    ))
                    .into(),
                    self.stats.clone(),
                ));
            };

            if let Some(ref mut records) = trace_records {
                records.push((parent_trace_loc, succ_fp));
            }
            if checkpoint_active {
                self.trace.depths.insert(succ_fp, depth);
            }
            raw_inserted_successors.push((
                successor_idx,
                succ_fp,
                depth,
                self.trace.last_inserted_trace_loc,
            ));
        }

        if let Some(records) = trace_records {
            let (locations, error) = {
                let trace_file = self
                    .trace
                    .trace_file
                    .as_mut()
                    .expect("trace_records exists only when trace_file is installed");
                trace_file.write_states_batch_until_error(records.iter().copied())
            };
            let written_count = locations.len();
            for ((_, fp, _, trace_loc), loc) in raw_inserted_successors.iter_mut().zip(locations) {
                *trace_loc = loc;
                self.trace.last_inserted_trace_loc = loc;
                if !self.trace.lazy_trace_index && !self.trace.trace_locs.insert(*fp, loc) {
                    self.trace.trace_degraded = true;
                }
            }
            if let Some(error) = error {
                self.mark_trace_degraded(&error);
            }

            let fallback_trace_loc = self.trace.last_inserted_trace_loc;
            for (_, _, _, trace_loc) in raw_inserted_successors.iter_mut().skip(written_count) {
                *trace_loc = fallback_trace_loc;
            }
        }

        Ok(())
    }

    fn successor_arena_state(
        successor_arena: &[i64],
        successor_state_len: usize,
        successor_idx: usize,
    ) -> Option<&[i64]> {
        let start = successor_idx.checked_mul(successor_state_len)?;
        let end = start.checked_add(successor_state_len)?;
        successor_arena.get(start..end)
    }

    fn record_compiled_flat_payload_witness_if_absent(&mut self, fp: Fingerprint, slots: &[i64]) {
        self.state_storage
            .compiled_flat_payload_witnesses
            .record_flat_i64_slots_if_absent(fp, slots);
    }

    /// Eagerly record flat-payload witnesses for a batch of just-admitted
    /// successors (indexed into `successor_arena` by `inserted_indices`, with
    /// fingerprints in `fingerprint_values`).
    ///
    /// The deferred batch-admission fast paths do NOT witness inserted successors
    /// (they are witnessed later, when expanded as frontier roots). That is sound
    /// for a whole-level native call — every duplicate reaching global admission is
    /// a collision with a PRIOR level, whose root is already witnessed. Under
    /// parent sub-batching, a state admitted by an earlier sub-batch can be
    /// regenerated by a later sub-batch of the SAME level (a cross-parent duplicate
    /// the per-call native local-dedup no longer catches across calls); that
    /// duplicate hits the global set with no witness yet and would fail closed.
    /// Recording the witness here — first-writer-wins into an append-only arena, so
    /// the state is still witnessed exactly once over the run, just earlier — lets
    /// the later sub-batch confirm it as a real duplicate. Callers gate on
    /// sub-batch mode so the whole-level path is unaffected.
    fn record_subbatch_inserted_flat_witnesses(
        &mut self,
        successor_arena: &[i64],
        successor_state_len: usize,
        inserted_indices: &[usize],
        fingerprint_values: &[u64],
    ) {
        for &idx in inserted_indices {
            if let (Some(slots), Some(&fp)) = (
                Self::successor_arena_state(successor_arena, successor_state_len, idx),
                fingerprint_values.get(idx),
            ) {
                self.record_compiled_flat_payload_witness_if_absent(Fingerprint(fp), slots);
            }
        }
    }

    fn compiled_flat_payload_witness_confirms(
        &self,
        fp: Fingerprint,
        candidate: &[i64],
    ) -> Option<bool> {
        self.state_storage
            .compiled_flat_payload_witnesses
            .confirm_flat_i64_slots(fp, candidate)
    }

    /// Record a *canonical* ArrayState payload witness for `fp` if absent.
    ///
    /// Used by the interpreter-implied-action STEP path, which dedups in the
    /// canonicalizing ArrayFp64 domain. The witness stores `ArrayState` values
    /// (not raw flat slots) so that two distinct raw flat encodings of the same
    /// logical state are confirmed as duplicates rather than failing closed.
    fn record_compiled_flat_payload_witness_array_state_if_absent(
        &mut self,
        fp: Fingerprint,
        state: &crate::state::ArrayState,
    ) {
        self.state_storage
            .compiled_flat_payload_witnesses
            .record_array_state_if_absent(fp, state);
    }

    /// Confirm an ArrayState candidate against the canonical witness for `fp`.
    fn compiled_flat_payload_witness_confirms_array_state(
        &self,
        fp: Fingerprint,
        candidate: &crate::state::ArrayState,
    ) -> Option<bool> {
        self.state_storage
            .compiled_flat_payload_witnesses
            .confirm_array_state(fp, candidate)
    }

    fn record_compiled_flat_frontier_payload_witnesses(
        &mut self,
        flat_queue: &FlatBfsFrontier,
        parent_count: usize,
    ) {
        for parent_idx in 0..parent_count {
            if let (Some((fp, _, _)), Some(parent_buf)) = (
                flat_queue.meta_at_offset(parent_idx),
                flat_queue.remaining_state_at_offset(parent_idx),
            ) {
                self.record_compiled_flat_payload_witness_if_absent(fp, parent_buf);
            }
        }
    }

    /// Pre-record *canonical* ArrayState witnesses for every frontier state.
    ///
    /// The interpreter-implied-action STEP path dedups in the ArrayFp64 domain,
    /// so a successor that revisits a frontier root must be confirmed against a
    /// canonical witness for that root's fingerprint. Frontier roots are queued
    /// before they are expanded as parents, so we seed their canonical witnesses
    /// up front (mirrors `record_compiled_flat_frontier_payload_witnesses` for
    /// the raw compiled-flat domain). Buffers that fail to reconstruct are
    /// skipped: a missing witness only makes the per-edge confirmation fail
    /// closed, which is sound.
    fn record_compiled_flat_frontier_payload_witnesses_array_state(
        &mut self,
        flat_queue: &FlatBfsFrontier,
        parent_count: usize,
        registry: &crate::var_index::VarRegistry,
    ) {
        for parent_idx in 0..parent_count {
            let Some((fp, _, _)) = flat_queue.meta_at_offset(parent_idx) else {
                continue;
            };
            let Some(parent_buf) = flat_queue.remaining_state_at_offset(parent_idx) else {
                continue;
            };
            let parent_buf = parent_buf.to_vec();
            if let Ok(array) = self.try_reconstruct_array_state_from_flat(&parent_buf, registry) {
                self.record_compiled_flat_payload_witness_array_state_if_absent(fp, &array);
            }
        }
    }

    fn fp_only_batch_duplicate_payloads_confirmed(
        &mut self,
        registry: &crate::var_index::VarRegistry,
        flat_queue: &FlatBfsFrontier,
        parent_indices: CompiledSuccessorParentIndices<'_>,
        fingerprint_values: &[u64],
        inserted_indices: &[usize],
        successor_arena: &[i64],
        successor_state_len: usize,
        attempted: usize,
    ) -> bool {
        let mut batch_payload_witnesses: rustc_hash::FxHashMap<u64, usize> =
            rustc_hash::FxHashMap::default();
        batch_payload_witnesses.reserve(inserted_indices.len().min(attempted));
        let mut next_inserted_index = 0usize;

        for successor_idx in 0..attempted {
            let Some(&fp_value) = fingerprint_values.get(successor_idx) else {
                return false;
            };
            let Some(candidate) =
                Self::successor_arena_state(successor_arena, successor_state_len, successor_idx)
            else {
                return false;
            };

            if inserted_indices
                .get(next_inserted_index)
                .is_some_and(|inserted_idx| *inserted_idx == successor_idx)
            {
                batch_payload_witnesses
                    .entry(fp_value)
                    .or_insert(successor_idx);
                next_inserted_index += 1;
                continue;
            }

            let parent_confirms = parent_indices
                .get(successor_idx)
                .and_then(|parent_idx| {
                    flat_queue
                        .meta_at_offset(parent_idx)
                        .zip(flat_queue.remaining_state_at_offset(parent_idx))
                })
                .is_some_and(|((parent_fp, _, _), parent_buf)| {
                    parent_fp.0 == fp_value && parent_buf == candidate
                });
            if parent_confirms {
                batch_payload_witnesses
                    .entry(fp_value)
                    .or_insert(successor_idx);
                continue;
            }

            if let Some(&previous_idx) = batch_payload_witnesses.get(&fp_value) {
                let Some(previous) =
                    Self::successor_arena_state(successor_arena, successor_state_len, previous_idx)
                else {
                    return false;
                };
                if previous == candidate {
                    continue;
                }
            }

            match self.compiled_flat_payload_witness_confirms(Fingerprint(fp_value), candidate) {
                Some(true) => {
                    batch_payload_witnesses
                        .entry(fp_value)
                        .or_insert(successor_idx);
                    continue;
                }
                Some(false) => return false,
                None => {}
            }

            let Ok(candidate_array) =
                self.try_reconstruct_array_state_from_flat(candidate, registry)
            else {
                return false;
            };
            if self
                .enforce_seen_state_duplicate_with_payload(
                    Fingerprint(fp_value),
                    &candidate_array,
                    None,
                )
                .is_ok()
            {
                batch_payload_witnesses
                    .entry(fp_value)
                    .or_insert(successor_idx);
                continue;
            }

            return false;
        }

        true
    }

    fn try_reconstruct_state_from_flat(
        &self,
        flat_buf: &[i64],
        registry: &crate::var_index::VarRegistry,
    ) -> Result<crate::State, CheckResult> {
        self.try_reconstruct_array_state_from_flat(flat_buf, registry)
            .map(|array| array.to_state(registry))
    }

    fn try_reconstruct_array_state_from_flat(
        &self,
        flat_buf: &[i64],
        registry: &crate::var_index::VarRegistry,
    ) -> Result<crate::state::ArrayState, CheckResult> {
        if let Some(ref adapter) = self.flat_bfs_adapter {
            adapter
                .bridge()
                .try_to_array_state_from_buffer(flat_buf, registry)
                .map_err(|error| {
                    self.flat_reconstruction_error_result(
                        "failed to reconstruct compiled BFS flat buffer",
                        error,
                    )
                })
        } else {
            // Fallback: treat each i64 as SmallInt.
            let values: Vec<_> = flat_buf
                .iter()
                .map(|&v| crate::Value::SmallInt(v))
                .collect();
            Ok(crate::state::ArrayState::from_values(values))
        }
    }

    /// Report compiled BFS loop statistics.
    fn report_compiled_bfs_stats(&self, stats: &CompiledBfsLoopStats) {
        let execution = stats
            .execution_started_at
            .map(|started_at| started_at.elapsed());
        let execution_nanos = execution
            .map(|duration| duration.as_nanos())
            .unwrap_or_default();
        let execution_seconds = execution
            .map(|duration| duration.as_secs_f64())
            .unwrap_or_default();
        if let (Some(duration), Some(trace)) = (execution, self.setup_trace.as_ref()) {
            if let Ok(mut trace) = trace.try_borrow_mut() {
                trace.record_duration(tla_mc_core::SetupTracePhase::HotExecution, duration);
                if super::super::trust_cg_dispatch::trust_cg_setup_timing_enabled() {
                    for row in trace.render_evidence_rows("TY") {
                        if row.contains("phase=hot_execution ") {
                            eprintln!("[trust_cg-setup-trace] {row}");
                        }
                    }
                }
            }
        }
        telemetry_eprintln!(
            "[compiled-bfs] completed: {} levels, {} parents, {} generated, {} new, {} total states, step_outputs_borrowed={}, step_outputs_owned={}, fused_pre_seen_lookups={}, fused_pre_seen_skipped={}, fused_admission_validations_elided={}, fused_prepared_batch_admission_calls={}, fused_prepared_batch_admission_descriptor_validations={}, fused_prepared_batch_admission_hot_descriptor_validations={}, fused_prepared_batch_admission_fingerprints={}, fused_prepared_batch_duplicate_authorization_checks={}, compiled_bfs_execution_nanos={}, compiled_bfs_execution_seconds={:.6}, execution_time_ns={}, execution_time_seconds={:.6}",
            stats.levels_completed,
            stats.parents_processed,
            stats.successors_generated,
            stats.successors_new,
            self.states_count(),
            stats.step_outputs_borrowed,
            stats.step_outputs_owned,
            stats.fused_pre_seen_lookups,
            stats.fused_pre_seen_lookups_skipped,
            stats.fused_admission_validations_elided,
            stats.fused_prepared_batch_admission_calls,
            stats.fused_prepared_batch_admission_descriptor_validations,
            stats.fused_prepared_batch_admission_hot_descriptor_validations,
            stats.fused_prepared_batch_admission_fingerprints,
            stats.fused_prepared_batch_duplicate_authorization_checks,
            execution_nanos,
            execution_seconds,
            execution_nanos,
            execution_seconds,
        );
        if stats.telemetry_enabled {
            self.report_compiled_bfs_native_vs_interp_telemetry(stats);
        }
        // --profile-enum: the compiled BFS loop bypasses the interpreter
        // transport (`transport_seq.rs`) where `BfsProfile` normally
        // accumulates, so synthesize the enumeration-profile snapshot from the
        // loop's own counters here. The measured native execution time is
        // attributed to successor generation — the fused kernel performs
        // generation, fingerprinting, dedup, and invariant checks in one
        // native call, so a finer per-phase split does not exist on this path.
        // `BfsProfile::new` gates on `profile_enum()` internally and
        // `output_bfs_profile` is a no-op when profiling is disabled.
        let mut prof = super::super::run_helpers::BfsProfile::new(
            stats
                .execution_started_at
                .unwrap_or_else(std::time::Instant::now),
        );
        if prof.do_profile {
            prof.total_successors = stats.successors_generated;
            prof.new_states = stats.successors_new;
            prof.succ_gen_us = (execution_nanos / 1_000) as u64;
            prof.snapshot_arena_stats();
            Self::output_bfs_profile(&prof);
        }
    }

    /// Observability-only (un-darkening STEP 1): emit the per-path native-vs-interp
    /// runtime wall-clock summary plus the lazy-compile promotion point. Called
    /// from `report_compiled_bfs_stats` only when `TY_TRUST_CG_TELEMETRY` is set,
    /// so a default run produces no extra output and reads no clocks.
    ///
    /// NOTE on grain: the native kernel runs ALL actions fused per level (or per
    /// parent on the step path), so the wall-clock here is the aggregate
    /// native-callout time, not a per-individual-action decomposition (that would
    /// require instrumenting the JIT kernel, which would perturb the very runtime
    /// being measured). The per-individual-action native-vs-interpreter
    /// *admission* breakdown — which action compiled native vs fell back to the
    /// interpreter, and why — is surfaced at cache-build time by
    /// `TY_TRUST_CG_DUMP_NATIVE_ADMISSION_FAILURES`. Together they answer "which
    /// ops block which specs" (admission dump) and "is native faster at runtime"
    /// (this table).
    fn report_compiled_bfs_native_vs_interp_telemetry(&self, stats: &CompiledBfsLoopStats) {
        let mean = |total_ns: u128, count: u64| -> f64 {
            if count == 0 {
                0.0
            } else {
                total_ns as f64 / count as f64
            }
        };
        let native_total_ns = stats
            .native_fused_total_ns
            .saturating_add(stats.native_step_total_ns);
        let native_call_count = stats
            .native_fused_call_count
            .saturating_add(stats.native_step_call_count);
        eprintln!(
            "[trust_cg-telemetry] native-vs-interp runtime summary (TY_TRUST_CG_TELEMETRY): \
             native_call_count={native_call_count} native_total_ns={native_total_ns} \
             native_mean_ns={:.1} | fused_level: call_count={} total_ns={} mean_ns={:.1} | \
             step_per_parent: call_count={} total_ns={} mean_ns={:.1}",
            mean(native_total_ns, native_call_count),
            stats.native_fused_call_count,
            stats.native_fused_total_ns,
            mean(stats.native_fused_total_ns, stats.native_fused_call_count),
            stats.native_step_call_count,
            stats.native_step_total_ns,
            mean(stats.native_step_total_ns, stats.native_step_call_count),
        );
        eprintln!(
            "[trust_cg-telemetry] interp note: when this compiled BFS loop falls back \
             to `run_bfs_loop`, the remaining states are generated by the interpreter \
             tree-walker; native counts above cover only the compiled-path callouts. \
             Per-action native-vs-interpreter ADMISSION (which actions compiled native, \
             and the rejection reason for those that did not) is reported at cache-build \
             time under TY_TRUST_CG_DUMP_NATIVE_ADMISSION_FAILURES."
        );
        // Lazy-compile promotion point (level/states/transitions/compile_ms). The
        // compile_ms is tracked by trust_cg_dispatch; surface what the loop can
        // see here (the cumulative distinct-state count and the configured
        // deferral threshold) so the promotion point is observable alongside the
        // runtime split.
        eprintln!(
            "[trust_cg-telemetry] lazy-compile promotion: levels_completed={} states={} \
             transitions(successors_generated)={} fused_level_defer_threshold={}",
            stats.levels_completed,
            self.states_count(),
            stats.successors_generated,
            super::super::trust_cg_dispatch::trust_cg_fused_level_defer_threshold(),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::{
        compiled_bfs_leading_same_depth_run, compiled_fused_batch_admission_plan,
        fused_level_may_skip_global_pre_seen_lookup, fused_successor_needs_pre_seen_lookup,
        fused_successor_needs_rust_regular_invariant_check,
        fused_successor_trusts_backend_fingerprint_sidecars, BfsFingerprintDomain,
        CompiledBfsLivenessSuccessors, CompiledBfsLoopStats, CompiledFusedBatchAdmissionRuntime,
    };
    use crate::arena::BulkStateStorage;
    use crate::check::model_checker::bfs::compiled_step_trait::{
        BfsStepError, CompiledBfsLevel, CompiledBfsStep, CompiledLevelResult,
        CompiledSuccessorParentIndices, FlatBfsStepOutput,
    };
    use crate::check::model_checker::bfs::flat_frontier::FlatBfsFrontier;
    use crate::check::model_checker::bfs::storage_modes::FingerprintOnlyStorage;
    use crate::check::model_checker::frontier::BfsFrontier;
    use crate::check::model_checker::{CheckResult, ModelChecker};
    use crate::config::Config;
    use crate::state::{ArrayState, Fingerprint, StateLayout, VarLayoutKind};
    use crate::state::{FlatValueLayout, SequenceBoundEvidence, SlotType};
    use crate::storage::TraceLocationStorage;
    use crate::storage::{BatchInsertedIndexAdmission, FingerprintSet, FingerprintStorage};
    use crate::storage::{CapacityStatus, InsertOutcome, LookupOutcome};
    use crate::test_support::parse_module;
    use crate::{CheckError, InfraCheckError, TraceFile, Value};
    use std::collections::HashSet;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::sync::Arc;
    use std::sync::Mutex;
    use tla_mc_core::PreparedFingerprintPayloadWitnessKind;

    // ---- P0: fused-arena native↔interpreter equivalence comparison (pure core) ----------------

    fn expected_map(
        entries: &[(usize, &[&[i64]])],
    ) -> std::collections::BTreeMap<usize, std::collections::BTreeSet<Vec<i64>>> {
        entries
            .iter()
            .map(|(p, succs)| {
                (
                    *p,
                    succs
                        .iter()
                        .map(|s| s.to_vec())
                        .collect::<std::collections::BTreeSet<_>>(),
                )
            })
            .collect()
    }

    // ---- Wide-level parent sub-batching -------------------------------------------------------

    fn subbatch_scalar_frontier() -> FlatBfsFrontier {
        let registry = crate::var_index::VarRegistry::from_names(["x"]);
        let layout = Arc::new(StateLayout::new(&registry, vec![VarLayoutKind::Scalar]));
        FlatBfsFrontier::new(layout)
    }

    /// The leading-same-depth-run split must (a) never cross a BFS depth boundary
    /// (so a sub-batch's successors all share `first_depth + 1`) and (b) honor the
    /// records-derived parent cap. This is the invariant that keeps a sub-batched
    /// level's successor set — and its ascending-parent-index duplicate
    /// tie-breaking — identical to the whole-level native call.
    #[test]
    fn subbatch_run_stops_at_depth_boundary_and_cap() {
        let mut frontier = subbatch_scalar_frontier();
        // Five depth-7 parents (the current level) followed by three depth-8
        // successors already appended to the tail (the next level).
        for i in 0..5 {
            frontier.push_raw_buffer(&[i], Fingerprint(1000 + i as u64), 7, i as u64);
        }
        for i in 0..3 {
            frontier.push_raw_buffer(&[100 + i], Fingerprint(2000 + i as u64), 8, i as u64);
        }

        // A generous cap stops exactly at the depth-7/8 boundary (never straddles).
        assert_eq!(compiled_bfs_leading_same_depth_run(&frontier, 7, 1_000), 5);
        // A tight cap bounds the sub-batch below the same-depth run length.
        assert_eq!(compiled_bfs_leading_same_depth_run(&frontier, 7, 2), 2);
        assert_eq!(compiled_bfs_leading_same_depth_run(&frontier, 7, 5), 5);

        // Consume the depth-7 level in two sub-batches (cap 2, then 2, then 1); the
        // run shrinks and the boundary detection tracks it exactly.
        frontier.advance_read_cursor(2);
        assert_eq!(compiled_bfs_leading_same_depth_run(&frontier, 7, 1_000), 3);
        frontier.advance_read_cursor(2);
        assert_eq!(compiled_bfs_leading_same_depth_run(&frontier, 7, 1_000), 1);
        frontier.advance_read_cursor(1);
        // Now the frontier head is the depth-8 level.
        assert_eq!(frontier.meta_at_offset(0).map(|(_, d, _)| d), Some(8));
        assert_eq!(compiled_bfs_leading_same_depth_run(&frontier, 8, 1_000), 3);
    }

    /// Cross-sub-batch dedup soundness: a state admitted by an EARLIER sub-batch and
    /// regenerated by a LATER sub-batch of the same level is a global-set duplicate
    /// whose witness the deferred batch-admission path would not otherwise record
    /// until expansion. Eagerly recording it (as `record_subbatch_inserted_flat_
    /// witnesses` does) makes the later sub-batch confirm it as a real duplicate —
    /// exactly (same fingerprint AND same payload) — while still rejecting a
    /// hash-collision (same fingerprint, different payload). This reproduces the
    /// whole-level guarantee, where such a duplicate is instead removed by the
    /// native per-call local dedup, so the admitted set is identical either way.
    #[test]
    fn subbatch_cross_batch_duplicate_confirms_via_eager_witness() {
        use crate::storage::FingerprintPayloadWitnesses;
        let mut witnesses = FingerprintPayloadWitnesses::new();
        let fp = Fingerprint(0xABCD);
        let payload = [7i64, -3, 42];

        // Before sub-batch 1 admits the state there is no witness: a later sub-batch
        // could not confirm the duplicate (would fail closed) — the very gap the
        // eager recording closes.
        assert_eq!(witnesses.confirm_flat_i64_slots(fp, &payload), None);

        // Sub-batch 1 admits the state → record its witness eagerly.
        witnesses.record_flat_i64_slots_if_absent(fp, &payload);

        // Sub-batch 2 regenerates the SAME state: confirmed as a true duplicate.
        assert_eq!(witnesses.confirm_flat_i64_slots(fp, &payload), Some(true));
        // A genuine hash collision (same fp, different payload) is still rejected.
        assert_eq!(
            witnesses.confirm_flat_i64_slots(fp, &[7, -3, 43]),
            Some(false)
        );
        // First-writer-wins keeps the arena append-once: re-recording is a no-op, so
        // eager witnessing during sub-batching never double-stores a state.
        witnesses.record_flat_i64_slots_if_absent(fp, &payload);
        assert_eq!(witnesses.len(), 1);
    }

    #[test]
    fn fused_equiv_agrees_when_native_matches_interpreter() {
        let expected = expected_map(&[(0, &[&[1], &[2]]), (1, &[&[3]])]);
        let native = vec![(0usize, vec![1i64]), (0, vec![2]), (1, vec![3])];
        // Exact match, complete graph → no divergence.
        assert!(
            ModelChecker::fused_arena_native_equiv_divergence(&native, true, true, &expected)
                .is_none()
        );
    }

    #[test]
    fn fused_equiv_detects_extra_native_edge() {
        let expected = expected_map(&[(0, &[&[1], &[2]])]);
        // Native produced [9] for parent 0, which the interpreter cannot reach → divergence,
        // EVEN with an incomplete (deduped) graph, because extra-edge is always sound.
        let native = vec![(0usize, vec![1i64]), (0, vec![9])];
        assert!(
            ModelChecker::fused_arena_native_equiv_divergence(&native, true, false, &expected)
                .is_some()
        );
    }

    #[test]
    fn fused_equiv_missing_edge_only_flags_when_graph_complete() {
        let expected = expected_map(&[(0, &[&[1], &[2]])]);
        // Native is missing [2] for parent 0.
        let native = vec![(0usize, vec![1i64])];
        // Complete graph (no cross-parent dedup) → missing edge IS a divergence.
        assert!(
            ModelChecker::fused_arena_native_equiv_divergence(&native, true, true, &expected)
                .is_some()
        );
        // Incomplete graph (dedup possible) → missing edge is NOT flagged (sound: the edge may
        // be attributed to another parent).
        assert!(
            ModelChecker::fused_arena_native_equiv_divergence(&native, true, false, &expected)
                .is_none()
        );
    }

    #[test]
    fn fused_equiv_skips_when_attribution_incomplete() {
        let expected = expected_map(&[(0, &[&[1]])]);
        // Even a blatant extra edge is NOT flagged when the backend gave no reliable parent
        // sidecar — the check must never false-positive on missing attribution.
        let native = vec![(0usize, vec![999i64])];
        assert!(
            ModelChecker::fused_arena_native_equiv_divergence(&native, false, true, &expected)
                .is_none()
        );
    }

    #[test]
    fn fused_equiv_ignores_unsampled_parents() {
        // Parent 1 was not sampled this round; its native successors must not affect the verdict.
        let expected = expected_map(&[(0, &[&[1]])]);
        let native = vec![(0usize, vec![1i64]), (1, vec![777])];
        assert!(
            ModelChecker::fused_arena_native_equiv_divergence(&native, true, true, &expected)
                .is_none()
        );
    }

    fn two_parent_liveness_module() -> tla_core::ast::Module {
        parse_module(
            r#"
---- MODULE CompiledBfsLivenessTwoParentTest ----
VARIABLE x
Init == x = 0 \/ x = 1
Next == \/ /\ x = 0 /\ x' = 2
        \/ /\ x = 1 /\ x' = 2
Inv == TRUE
====
"#,
        )
    }

    fn seed_two_parent_flat_frontier(
        checker: &mut ModelChecker<'_>,
    ) -> (
        FlatBfsFrontier,
        FingerprintOnlyStorage,
        Fingerprint,
        Fingerprint,
    ) {
        checker.trace.cached_next_name = Some("Next".to_string());
        checker.trace.cached_resolved_next_name = Some("Next".to_string());

        let layout = Arc::new(StateLayout::new(
            checker.ctx.var_registry(),
            vec![VarLayoutKind::Scalar],
        ));
        checker.flat_bfs_adapter = Some(crate::state::FlatBfsAdapter::from_layout(layout.clone()));

        let fp0 = crate::check::model_checker::invariants::fingerprint_flat_compiled(&[0]);
        let fp1 = crate::check::model_checker::invariants::fingerprint_flat_compiled(&[1]);
        checker
            .mark_state_seen_fp_only_checked(fp0, None, 0)
            .expect("seed x=0");
        let loc0 = checker.trace.last_inserted_trace_loc;
        checker
            .mark_state_seen_fp_only_checked(fp1, None, 0)
            .expect("seed x=1");
        let loc1 = checker.trace.last_inserted_trace_loc;

        let mut flat_queue = FlatBfsFrontier::new(layout);
        flat_queue.push_raw_buffer(&[0], fp0, 0, loc0);
        flat_queue.push_raw_buffer(&[1], fp1, 0, loc1);

        let storage = FingerprintOnlyStorage::new(
            BulkStateStorage::empty(checker.ctx.var_registry().len()),
            checker.ctx.var_registry().len(),
        );
        (flat_queue, storage, fp0, fp1)
    }

    fn assert_both_parent_liveness_edges(
        checker: &ModelChecker<'_>,
        fp0: Fingerprint,
        fp1: Fingerprint,
    ) {
        let graph = checker
            .liveness_cache
            .successors
            .as_inner_map()
            .expect("test uses in-memory successor graph");
        let s0 = graph.get(&fp0).expect("x=0 liveness parent entry");
        let s1 = graph.get(&fp1).expect("x=1 liveness parent entry");

        assert_eq!(s0.len(), 1, "x=0 should retain its edge to x=2");
        assert_eq!(s1.len(), 1, "x=1 duplicate child edge must not be lost");
        assert_eq!(s0[0], s1[0], "both parents should point at x=2");
    }

    #[test]
    fn compiled_bfs_liveness_successor_capture_keeps_monotonic_runs_sparse_by_parent() {
        let mut capture = CompiledBfsLivenessSuccessors::with_successor_capacity(3);
        capture.push(0, Fingerprint(10));
        capture.push(0, Fingerprint(11));
        capture.push(2, Fingerprint(20));

        assert!(capture.is_run_encoded_for_testing());
        assert_eq!(
            capture.into_successor_entries_for_testing(4),
            vec![
                (0, vec![Fingerprint(10), Fingerprint(11)]),
                (1, vec![]),
                (2, vec![Fingerprint(20)]),
                (3, vec![]),
            ]
        );
    }

    #[test]
    fn compiled_bfs_liveness_successor_capture_handles_out_of_order_edges() {
        let mut capture = CompiledBfsLivenessSuccessors::with_successor_capacity(2);
        capture.push(2, Fingerprint(20));
        capture.push(0, Fingerprint(10));

        assert!(!capture.is_run_encoded_for_testing());
        assert_eq!(
            capture.into_successor_entries_for_testing(3),
            vec![
                (0, vec![Fingerprint(10)]),
                (1, vec![]),
                (2, vec![Fingerprint(20)]),
            ]
        );
    }

    #[test]
    fn compiled_bfs_batch_bookkeeping_preserves_trace_degradation_and_checkpoint_depth() {
        let module = parse_module(
            r#"
---- MODULE CompiledBfsBatchBookkeepingTest ----
VARIABLE x
Init == x = 0
Next == x' = x + 1
====
"#,
        );
        let config = Config {
            init: Some("Init".to_string()),
            next: Some("Next".to_string()),
            check_deadlock: false,
            ..Default::default()
        };
        let mut checker = ModelChecker::new(&module, &config);
        checker.set_trace_file(TraceFile::create_temp().expect("trace file should initialize"));
        checker.set_trace_locations_storage(
            crate::storage::TraceLocationsStorage::mmap(1, None)
                .expect("trace location mmap should initialize"),
        );
        let checkpoint_dir = tempfile::tempdir().expect("checkpoint temp dir");
        checker.set_checkpoint(
            checkpoint_dir.path().to_path_buf(),
            std::time::Duration::from_secs(60),
        );

        let layout = Arc::new(StateLayout::new(
            checker.ctx.var_registry(),
            vec![VarLayoutKind::Scalar],
        ));
        let mut flat_queue = FlatBfsFrontier::new(layout);
        let parent_fp = Fingerprint(10);
        checker
            .mark_state_seen_fp_only_checked(parent_fp, None, 0)
            .expect("seed parent fingerprint");
        let parent_loc = checker.trace.last_inserted_trace_loc;
        flat_queue.push_raw_buffer(&[0], parent_fp, 0, parent_loc);

        let successor_fp = Fingerprint(20);
        let mut raw_inserted_successors = Vec::new();
        checker
            .record_fp_only_batch_admission_bookkeeping_for_indices(
                &flat_queue,
                CompiledSuccessorParentIndices::Usize(&[0]),
                &[successor_fp.0],
                &[0],
                1,
                &mut raw_inserted_successors,
            )
            .expect("batch bookkeeping should complete despite trace index degradation");

        assert!(checker.trace.trace_degraded);
        assert_eq!(
            raw_inserted_successors,
            vec![(0, successor_fp, 1, parent_loc + 16)]
        );
        assert_eq!(checker.trace.depths.get(&successor_fp), Some(&1));
        let parents = checker
            .trace
            .trace_file
            .as_mut()
            .expect("trace file still installed")
            .build_parents_map()
            .expect("parents map should scan");
        assert_eq!(parents.get(&successor_fp), Some(&parent_fp));
    }

    #[test]
    fn compiled_fused_batch_admission_runtime_uses_validated_handle_once_for_batch_calls() {
        let runtime = CompiledFusedBatchAdmissionRuntime::new(BfsFingerprintDomain::CompiledFlat)
            .expect("compiled fused batch admission descriptor should validate once");
        let storage = FingerprintStorage::in_memory();
        let mut stats = CompiledBfsLoopStats::default();
        let mut admission = BatchInsertedIndexAdmission::default();
        runtime.record_setup_validation(&mut stats);

        let fp1 = crate::check::model_checker::invariants::fingerprint_flat_compiled(&[1]).0;
        let fp2 = crate::check::model_checker::invariants::fingerprint_flat_compiled(&[2]).0;
        runtime.insert_batch_fingerprint_values_inserted_indices_checked_into(
            &storage,
            &[fp1, fp2],
            &mut admission,
            &mut stats,
        );

        assert_eq!(admission.attempted, 2);
        assert_eq!(admission.inserted_indices, vec![0, 1]);

        runtime.insert_batch_fingerprint_values_inserted_indices_checked_into(
            &storage,
            &[fp1],
            &mut admission,
            &mut stats,
        );
        runtime
            .enforce_batch_duplicate_authorization_checked(
                admission.attempted,
                admission.inserted_indices.len(),
                admission.fault.is_some(),
                true,
                &mut stats,
            )
            .expect("payload-confirmed duplicate batch should suppress");

        assert_eq!(admission.attempted, 1);
        assert!(admission.inserted_indices.is_empty());
        assert_eq!(stats.fused_prepared_batch_admission_calls, 2);
        assert_eq!(
            stats.fused_prepared_batch_admission_descriptor_validations, 1,
            "compiled/fused batch admission must bind the prepared descriptor once"
        );
        assert_eq!(
            stats.fused_prepared_batch_admission_hot_descriptor_validations, 0,
            "compiled/fused batch admission must not validate descriptors per fingerprint"
        );
        assert_eq!(stats.fused_prepared_batch_admission_fingerprints, 3);
        assert_eq!(
            stats.fused_prepared_batch_duplicate_authorization_checks,
            1,
            "duplicate authorization should be checked once per duplicate batch, not per fingerprint"
        );

        let mismatch = runtime
            .enforce_batch_duplicate_authorization_checked(
                admission.attempted,
                admission.inserted_indices.len(),
                admission.fault.is_some(),
                false,
                &mut stats,
            )
            .expect_err("mismatched duplicate batch must fail through fused prepared handle");
        assert_eq!(mismatch.backend, "prepared_fingerprint_admission");
        assert!(mismatch
            .detail
            .contains("reason_code=canonical_payload_mismatch"));
        assert_eq!(stats.fused_prepared_batch_duplicate_authorization_checks, 2);
    }

    #[test]
    fn compiled_fused_batch_admission_compiled_flat_identity_matches_scalar_domain() {
        let plan = compiled_fused_batch_admission_plan(BfsFingerprintDomain::CompiledFlat);

        assert_eq!(
            plan.dedup.fingerprint.id,
            "tla compiled-flat fingerprint-only state"
        );
        assert_eq!(plan.dedup.fingerprint.namespace, "tla-compiled-flat-state");
        assert_eq!(
            plan.dedup.fingerprint.canonical_domain.id,
            "tla-flat-i64-state"
        );
        assert_eq!(plan.dedup.fingerprint.canonical_domain.version, "v1");
        assert_eq!(
            plan.dedup.storage_policy_identity(),
            "dedup_storage:external:state_space:tla-check-compiled-flat-fingerprint-only-dyn-fingerprint-set-v1"
        );
        assert_eq!(
            plan.payload_witness,
            PreparedFingerprintPayloadWitnessKind::CompiledFlatXxh3
        );
    }

    fn recursive_sequence_layout(checker: &ModelChecker<'_>) -> Arc<StateLayout> {
        Arc::new(StateLayout::new(
            checker.ctx.var_registry(),
            vec![VarLayoutKind::Recursive {
                layout: FlatValueLayout::Sequence {
                    bound: SequenceBoundEvidence::ProvenInvariant {
                        invariant: Arc::from("BoundedSeq"),
                    },
                    max_len: 2,
                    element_layout: Box::new(FlatValueLayout::Scalar(SlotType::Int)),
                },
            }],
        ))
    }

    struct DuplicateOnlyFusedLevel;

    impl CompiledBfsLevel for DuplicateOnlyFusedLevel {
        fn has_fused_level(&self) -> bool {
            true
        }

        fn run_level_fused_arena(
            &self,
            _arena: &[i64],
            parent_count: usize,
        ) -> Option<Result<CompiledLevelResult, BfsStepError>> {
            assert_eq!(parent_count, 1);
            Some(Ok(CompiledLevelResult::from_successor_arena(
                vec![],
                0,
                1,
                1,
                1,
                0,
                true,
                None,
                None,
                None,
                true,
            )))
        }
    }

    struct DuplicateSuccessorFusedLevel;

    impl CompiledBfsLevel for DuplicateSuccessorFusedLevel {
        fn has_fused_level(&self) -> bool {
            true
        }

        fn run_level_fused_arena(
            &self,
            _arena: &[i64],
            parent_count: usize,
        ) -> Option<Result<CompiledLevelResult, BfsStepError>> {
            assert_eq!(parent_count, 1);
            Some(Ok(
                CompiledLevelResult::from_successor_arena_with_parent_indices(
                    vec![0],
                    Some(vec![0]),
                    1,
                    1,
                    1,
                    1,
                    1,
                    true,
                    None,
                    None,
                    None,
                    true,
                ),
            ))
        }
    }

    struct DuplicateCollisionBatchAdmissionFusedLevel;

    impl CompiledBfsLevel for DuplicateCollisionBatchAdmissionFusedLevel {
        fn has_fused_level(&self) -> bool {
            true
        }

        fn run_level_fused_arena(
            &self,
            _arena: &[i64],
            parent_count: usize,
        ) -> Option<Result<CompiledLevelResult, BfsStepError>> {
            assert_eq!(parent_count, 1);
            Some(Ok(
                CompiledLevelResult::from_successor_arena_with_parent_indices(
                    vec![1],
                    Some(vec![0]),
                    1,
                    1,
                    1,
                    1,
                    1,
                    true,
                    None,
                    None,
                    None,
                    true,
                ),
            ))
        }
    }

    struct ShortSuccessfulFusedLevel;

    impl CompiledBfsLevel for ShortSuccessfulFusedLevel {
        fn has_fused_level(&self) -> bool {
            true
        }

        fn run_level_fused_arena(
            &self,
            _arena: &[i64],
            parent_count: usize,
        ) -> Option<Result<CompiledLevelResult, BfsStepError>> {
            assert_eq!(parent_count, 2);
            Some(Ok(
                CompiledLevelResult::from_successor_arena_with_parent_indices(
                    vec![],
                    Some(Vec::new()),
                    0,
                    1,
                    1,
                    0,
                    0,
                    true,
                    None,
                    None,
                    None,
                    true,
                ),
            ))
        }
    }

    struct CompleteDeadlockMetadataFusedLevel;

    impl CompiledBfsLevel for CompleteDeadlockMetadataFusedLevel {
        fn has_fused_level(&self) -> bool {
            true
        }

        fn run_level_fused_arena(
            &self,
            _arena: &[i64],
            parent_count: usize,
        ) -> Option<Result<CompiledLevelResult, BfsStepError>> {
            assert_eq!(parent_count, 2);
            let mut arena = tla_trust_cg::TrustCgSuccessorArena::with_capacity(1, 1);
            let mut abi = arena
                .prepare_abi(1)
                .expect("prepare trust-codegen arena ABI");
            unsafe {
                *abi.states = 2;
                *abi.parent_index = 1;
                *abi.fingerprints =
                    crate::check::model_checker::invariants::fingerprint_flat_compiled(&[2]).0;
            }
            abi.state_count = 1;
            abi.generated = 1;
            abi.parents_processed = 2;
            abi.first_zero_generated_parent_idx = 0;
            abi.raw_successor_metadata_complete = 1;
            let outcome =
                unsafe { arena.commit_abi(&abi) }.expect("commit trust-codegen arena ABI");

            Some(Ok(CompiledLevelResult::from_trust_cg_successor_arena(
                arena,
                outcome.parents_processed,
                outcome.total_generated,
                outcome.total_new,
                outcome.invariant.is_ok(),
                None,
                None,
                None,
                true,
            )))
        }
    }

    struct ZeroGeneratedDeadlockMetadataFusedLevel {
        first_zero_generated_parent_idx: u32,
    }

    impl CompiledBfsLevel for ZeroGeneratedDeadlockMetadataFusedLevel {
        fn has_fused_level(&self) -> bool {
            true
        }

        fn run_level_fused_arena(
            &self,
            _arena: &[i64],
            parent_count: usize,
        ) -> Option<Result<CompiledLevelResult, BfsStepError>> {
            assert_eq!(parent_count, 2);
            let mut arena = tla_trust_cg::TrustCgSuccessorArena::with_capacity(1, 0);
            let mut abi = arena
                .prepare_abi(0)
                .expect("prepare empty trust-codegen arena ABI");
            abi.state_count = 0;
            abi.generated = 0;
            abi.parents_processed = 2;
            abi.first_zero_generated_parent_idx = self.first_zero_generated_parent_idx;
            abi.raw_successor_metadata_complete = 1;
            let outcome =
                unsafe { arena.commit_abi(&abi) }.expect("commit empty trust-codegen arena ABI");

            Some(Ok(CompiledLevelResult::from_trust_cg_successor_arena(
                arena,
                outcome.parents_processed,
                outcome.total_generated,
                outcome.total_new,
                outcome.invariant.is_ok(),
                None,
                None,
                None,
                true,
            )))
        }
    }

    struct MalformedInvariantFusedLevel;

    impl CompiledBfsLevel for MalformedInvariantFusedLevel {
        fn has_fused_level(&self) -> bool {
            true
        }

        fn run_level_fused_arena(
            &self,
            _arena: &[i64],
            parent_count: usize,
        ) -> Option<Result<CompiledLevelResult, BfsStepError>> {
            assert_eq!(parent_count, 2);
            Some(Ok(CompiledLevelResult::from_successor_arena(
                vec![],
                0,
                1,
                1,
                0,
                0,
                false,
                Some(1),
                Some(0),
                None,
                true,
            )))
        }
    }

    struct InvariantFailureTraceFusedLevel;

    impl CompiledBfsLevel for InvariantFailureTraceFusedLevel {
        fn has_fused_level(&self) -> bool {
            true
        }

        fn run_level_fused_arena(
            &self,
            _arena: &[i64],
            parent_count: usize,
        ) -> Option<Result<CompiledLevelResult, BfsStepError>> {
            assert_eq!(parent_count, 2);
            Some(Ok(
                CompiledLevelResult::from_successor_arena_with_parent_indices(
                    vec![2],
                    Some(vec![1]),
                    1,
                    1,
                    2,
                    1,
                    1,
                    false,
                    Some(1),
                    Some(0),
                    Some(vec![2]),
                    true,
                ),
            ))
        }
    }

    struct BatchAdmissionFusedLevel {
        calls: std::sync::atomic::AtomicUsize,
    }

    impl CompiledBfsLevel for BatchAdmissionFusedLevel {
        fn has_fused_level(&self) -> bool {
            true
        }

        fn run_level_fused_arena(
            &self,
            _arena: &[i64],
            parent_count: usize,
        ) -> Option<Result<CompiledLevelResult, BfsStepError>> {
            let call = self.calls.fetch_add(1, Ordering::SeqCst);
            if call == 0 {
                assert_eq!(parent_count, 1);
                Some(Ok(
                    CompiledLevelResult::from_successor_arena_with_parent_indices(
                        vec![1, 2],
                        Some(vec![0, 0]),
                        2,
                        1,
                        1,
                        2,
                        2,
                        true,
                        None,
                        None,
                        None,
                        true,
                    ),
                ))
            } else {
                Some(Ok(
                    CompiledLevelResult::from_successor_arena_with_parent_indices(
                        vec![],
                        Some(Vec::new()),
                        0,
                        1,
                        parent_count,
                        0,
                        0,
                        true,
                        None,
                        None,
                        None,
                        true,
                    ),
                ))
            }
        }
    }

    struct TrustCgBorrowedBatchAdmissionFusedLevel {
        calls: AtomicUsize,
    }

    impl CompiledBfsLevel for TrustCgBorrowedBatchAdmissionFusedLevel {
        fn has_fused_level(&self) -> bool {
            true
        }

        fn run_level_fused_arena(
            &self,
            _arena: &[i64],
            parent_count: usize,
        ) -> Option<Result<CompiledLevelResult, BfsStepError>> {
            let call = self.calls.fetch_add(1, Ordering::SeqCst);
            let mut arena = tla_trust_cg::TrustCgSuccessorArena::new(1);
            if call == 0 {
                assert_eq!(parent_count, 1);
                arena.push_successor(0, &[1]).unwrap();
                arena.push_successor(0, &[2]).unwrap();
                Some(Ok(CompiledLevelResult::from_trust_cg_successor_arena(
                    arena, 1, 2, 2, true, None, None, None, true,
                )))
            } else if call == 1 {
                assert_eq!(parent_count, 2);
                arena.push_successor(0, &[3]).unwrap();
                Some(Ok(CompiledLevelResult::from_trust_cg_successor_arena(
                    arena, 2, 1, 1, true, None, None, None, true,
                )))
            } else {
                Some(Ok(CompiledLevelResult::from_trust_cg_successor_arena(
                    arena,
                    parent_count,
                    0,
                    0,
                    true,
                    None,
                    None,
                    None,
                    true,
                )))
            }
        }
    }

    struct TrustCgRecursiveCanonicalBatchAdmissionFusedLevel {
        calls: AtomicUsize,
    }

    impl CompiledBfsLevel for TrustCgRecursiveCanonicalBatchAdmissionFusedLevel {
        fn has_fused_level(&self) -> bool {
            true
        }

        fn run_level_fused_arena(
            &self,
            arena: &[i64],
            parent_count: usize,
        ) -> Option<Result<CompiledLevelResult, BfsStepError>> {
            let call = self.calls.fetch_add(1, Ordering::SeqCst);
            let mut successor_arena = tla_trust_cg::TrustCgSuccessorArena::new(3);
            if call == 0 {
                assert_eq!(arena, &[0, 0, 0]);
                assert_eq!(parent_count, 1);
                successor_arena.push_successor(0, &[1, 10, 0]).unwrap();
                successor_arena.push_successor(0, &[2, 10, 20]).unwrap();
                Some(Ok(CompiledLevelResult::from_trust_cg_successor_arena(
                    successor_arena,
                    1,
                    2,
                    2,
                    true,
                    None,
                    None,
                    None,
                    true,
                )))
            } else {
                assert_eq!(arena, &[1, 10, 0, 2, 10, 20]);
                assert_eq!(parent_count, 2);
                Some(Ok(CompiledLevelResult::from_trust_cg_successor_arena(
                    successor_arena,
                    parent_count,
                    0,
                    0,
                    true,
                    None,
                    None,
                    None,
                    true,
                )))
            }
        }
    }

    struct TrustCgRecursiveNoncanonicalTailFusedLevel {
        calls: AtomicUsize,
    }

    impl CompiledBfsLevel for TrustCgRecursiveNoncanonicalTailFusedLevel {
        fn has_fused_level(&self) -> bool {
            true
        }

        fn run_level_fused_arena(
            &self,
            arena: &[i64],
            parent_count: usize,
        ) -> Option<Result<CompiledLevelResult, BfsStepError>> {
            let call = self.calls.fetch_add(1, Ordering::SeqCst);
            let mut successor_arena = tla_trust_cg::TrustCgSuccessorArena::new(3);
            if call == 0 {
                assert_eq!(arena, &[0, 0, 0]);
                assert_eq!(parent_count, 1);
                successor_arena.push_successor(0, &[1, 10, 99]).unwrap();
                Some(Ok(CompiledLevelResult::from_trust_cg_successor_arena(
                    successor_arena,
                    1,
                    1,
                    1,
                    true,
                    None,
                    None,
                    None,
                    true,
                )))
            } else {
                assert_eq!(
                    arena,
                    &[1, 10, 0],
                    "inactive recursive sequence tail must be canonicalized before enqueue",
                );
                assert_eq!(parent_count, 1);
                Some(Ok(CompiledLevelResult::from_trust_cg_successor_arena(
                    successor_arena,
                    parent_count,
                    0,
                    0,
                    true,
                    None,
                    None,
                    None,
                    true,
                )))
            }
        }
    }

    struct TrustCgBorrowedBatchStorageFaultFusedLevel;

    impl CompiledBfsLevel for TrustCgBorrowedBatchStorageFaultFusedLevel {
        fn has_fused_level(&self) -> bool {
            true
        }

        fn run_level_fused_arena(
            &self,
            _arena: &[i64],
            parent_count: usize,
        ) -> Option<Result<CompiledLevelResult, BfsStepError>> {
            assert_eq!(parent_count, 1);
            let mut arena = tla_trust_cg::TrustCgSuccessorArena::new(1);
            arena.push_successor(0, &[1]).unwrap();
            arena.push_successor(0, &[2]).unwrap();
            arena.push_successor(0, &[3]).unwrap();
            arena.push_successor(0, &[4]).unwrap();
            Some(Ok(CompiledLevelResult::from_trust_cg_successor_arena(
                arena, 1, 4, 4, true, None, None, None, true,
            )))
        }
    }

    struct TrustCgBorrowedBatchTraceProvenanceFusedLevel {
        calls: AtomicUsize,
    }

    impl CompiledBfsLevel for TrustCgBorrowedBatchTraceProvenanceFusedLevel {
        fn has_fused_level(&self) -> bool {
            true
        }

        fn run_level_fused_arena(
            &self,
            _arena: &[i64],
            parent_count: usize,
        ) -> Option<Result<CompiledLevelResult, BfsStepError>> {
            let call = self.calls.fetch_add(1, Ordering::SeqCst);
            let mut arena = tla_trust_cg::TrustCgSuccessorArena::new(1);
            if call == 0 {
                assert_eq!(parent_count, 2);
                arena.push_successor(0, &[10]).unwrap();
                arena.push_successor(1, &[20]).unwrap();
                Some(Ok(CompiledLevelResult::from_trust_cg_successor_arena(
                    arena, 2, 2, 2, true, None, None, None, true,
                )))
            } else {
                Some(Ok(CompiledLevelResult::from_trust_cg_successor_arena(
                    arena,
                    parent_count,
                    0,
                    0,
                    true,
                    None,
                    None,
                    None,
                    true,
                )))
            }
        }
    }

    struct BorrowedBatchOnlyFingerprintSet {
        seen: Mutex<HashSet<Fingerprint>>,
        value_batch_calls: AtomicUsize,
        scratch_ptr: AtomicUsize,
        scratch_reused_calls: AtomicUsize,
        fault_at_value_batch_index: Option<usize>,
        dropped: AtomicUsize,
    }

    impl BorrowedBatchOnlyFingerprintSet {
        fn new() -> Self {
            Self {
                seen: Mutex::new(HashSet::new()),
                value_batch_calls: AtomicUsize::new(0),
                scratch_ptr: AtomicUsize::new(0),
                scratch_reused_calls: AtomicUsize::new(0),
                fault_at_value_batch_index: None,
                dropped: AtomicUsize::new(0),
            }
        }

        fn with_fault_at_value_batch_index(index: usize) -> Self {
            Self {
                fault_at_value_batch_index: Some(index),
                ..Self::new()
            }
        }

        fn insert_one(&self, fp: Fingerprint) -> InsertOutcome {
            if self.seen.lock().unwrap().insert(fp) {
                InsertOutcome::Inserted
            } else {
                InsertOutcome::AlreadyPresent
            }
        }
    }

    impl tla_mc_core::FingerprintSet<Fingerprint> for BorrowedBatchOnlyFingerprintSet {
        fn insert_checked(&self, fingerprint: Fingerprint) -> InsertOutcome {
            self.insert_one(fingerprint)
        }

        fn contains_checked(&self, fingerprint: Fingerprint) -> LookupOutcome {
            if self.seen.lock().unwrap().contains(&fingerprint) {
                LookupOutcome::Present
            } else {
                LookupOutcome::Absent
            }
        }

        fn len(&self) -> usize {
            self.seen.lock().unwrap().len()
        }

        fn has_errors(&self) -> bool {
            self.dropped.load(Ordering::SeqCst) > 0
        }

        fn dropped_count(&self) -> usize {
            self.dropped.load(Ordering::SeqCst)
        }

        fn capacity_status(&self) -> CapacityStatus {
            CapacityStatus::Normal
        }
    }

    impl FingerprintSet for BorrowedBatchOnlyFingerprintSet {
        fn insert_batch_checked(&self, _fingerprints: &[Fingerprint]) -> Vec<InsertOutcome> {
            panic!("borrowed trust-codegen batch admission should not stage Fingerprint values")
        }

        fn insert_batch_fingerprint_values_checked(
            &self,
            _fingerprint_values: &[u64],
        ) -> Vec<InsertOutcome> {
            panic!("borrowed trust-codegen batch admission should use caller-owned outcome scratch")
        }

        fn insert_batch_fingerprint_values_checked_into(
            &self,
            _fingerprint_values: &[u64],
            _outcomes: &mut Vec<InsertOutcome>,
        ) {
            panic!("borrowed trust-codegen batch admission should use inserted-index scratch");
        }

        fn insert_batch_fingerprint_values_inserted_indices_checked_into(
            &self,
            fingerprint_values: &[u64],
            admission: &mut BatchInsertedIndexAdmission,
        ) {
            self.value_batch_calls.fetch_add(1, Ordering::SeqCst);
            admission.inserted_indices.clear();
            admission.inserted_indices.reserve(fingerprint_values.len());
            admission.attempted = 0;
            admission.fault = None;
            for (idx, fp) in fingerprint_values.iter().enumerate() {
                admission.attempted += 1;
                if self.fault_at_value_batch_index == Some(idx) {
                    self.dropped.fetch_add(1, Ordering::SeqCst);
                    admission.fault = Some(crate::storage::StorageFault::new(
                        "borrowed_trust_cg_test",
                        "insert_batch_fingerprint_values_inserted_indices_checked_into",
                        "synthetic prefix fault",
                    ));
                    break;
                }
                match self.insert_one(Fingerprint(*fp)) {
                    InsertOutcome::Inserted => admission.inserted_indices.push(idx),
                    InsertOutcome::AlreadyPresent => {}
                    InsertOutcome::StorageFault(fault) => {
                        admission.fault = Some(fault);
                        break;
                    }
                    _ => unreachable!(),
                }
            }

            let ptr = admission.inserted_indices.as_ptr() as usize;
            let previous = self.scratch_ptr.load(Ordering::SeqCst);
            if previous == 0 {
                self.scratch_ptr.store(ptr, Ordering::SeqCst);
            } else if previous == ptr {
                self.scratch_reused_calls.fetch_add(1, Ordering::SeqCst);
            } else {
                panic!(
                    "borrowed trust-codegen batch admission replaced outcome scratch allocation"
                );
            }
        }
    }

    struct StagedBatchOnlyFingerprintSet {
        seen: Mutex<HashSet<Fingerprint>>,
        staged_batch_calls: AtomicUsize,
    }

    impl StagedBatchOnlyFingerprintSet {
        fn new() -> Self {
            Self {
                seen: Mutex::new(HashSet::new()),
                staged_batch_calls: AtomicUsize::new(0),
            }
        }

        fn insert_one(&self, fp: Fingerprint) -> InsertOutcome {
            if self.seen.lock().unwrap().insert(fp) {
                InsertOutcome::Inserted
            } else {
                InsertOutcome::AlreadyPresent
            }
        }
    }

    impl tla_mc_core::FingerprintSet<Fingerprint> for StagedBatchOnlyFingerprintSet {
        fn insert_checked(&self, fingerprint: Fingerprint) -> InsertOutcome {
            self.insert_one(fingerprint)
        }

        fn contains_checked(&self, fingerprint: Fingerprint) -> LookupOutcome {
            if self.seen.lock().unwrap().contains(&fingerprint) {
                LookupOutcome::Present
            } else {
                LookupOutcome::Absent
            }
        }

        fn len(&self) -> usize {
            self.seen.lock().unwrap().len()
        }

        fn has_errors(&self) -> bool {
            false
        }

        fn dropped_count(&self) -> usize {
            0
        }

        fn capacity_status(&self) -> CapacityStatus {
            CapacityStatus::Normal
        }
    }

    impl FingerprintSet for StagedBatchOnlyFingerprintSet {
        fn insert_batch_checked(&self, _fingerprints: &[Fingerprint]) -> Vec<InsertOutcome> {
            panic!("staged trust-codegen batch admission should use inserted-index value admission")
        }

        fn insert_batch_fingerprint_values_checked(
            &self,
            _fingerprint_values: &[u64],
        ) -> Vec<InsertOutcome> {
            panic!("canonicalizing trust-codegen fallback must not use stale fingerprint sidecars")
        }

        fn insert_batch_fingerprint_values_checked_into(
            &self,
            _fingerprint_values: &[u64],
            _outcomes: &mut Vec<InsertOutcome>,
        ) {
            panic!("canonicalizing trust-codegen fallback must not use stale fingerprint sidecars")
        }

        fn insert_batch_fingerprint_values_inserted_indices_checked_into(
            &self,
            fingerprint_values: &[u64],
            admission: &mut BatchInsertedIndexAdmission,
        ) {
            self.staged_batch_calls.fetch_add(1, Ordering::SeqCst);
            admission.inserted_indices.clear();
            admission.inserted_indices.reserve(fingerprint_values.len());
            admission.attempted = 0;
            admission.fault = None;
            for (idx, fp) in fingerprint_values.iter().enumerate() {
                admission.attempted += 1;
                match self.insert_one(Fingerprint(*fp)) {
                    InsertOutcome::Inserted => admission.inserted_indices.push(idx),
                    InsertOutcome::AlreadyPresent => {}
                    InsertOutcome::StorageFault(fault) => {
                        admission.fault = Some(fault);
                        break;
                    }
                    _ => unreachable!(),
                }
            }
        }
    }

    struct MalformedInsertedIndexFingerprintSet {
        value_batch_calls: AtomicUsize,
    }

    impl MalformedInsertedIndexFingerprintSet {
        fn new() -> Self {
            Self {
                value_batch_calls: AtomicUsize::new(0),
            }
        }
    }

    impl tla_mc_core::FingerprintSet<Fingerprint> for MalformedInsertedIndexFingerprintSet {
        fn insert_checked(&self, _fingerprint: Fingerprint) -> InsertOutcome {
            InsertOutcome::Inserted
        }

        fn contains_checked(&self, _fingerprint: Fingerprint) -> LookupOutcome {
            LookupOutcome::Absent
        }

        fn len(&self) -> usize {
            0
        }

        fn has_errors(&self) -> bool {
            false
        }

        fn dropped_count(&self) -> usize {
            0
        }

        fn capacity_status(&self) -> CapacityStatus {
            CapacityStatus::Normal
        }
    }

    impl FingerprintSet for MalformedInsertedIndexFingerprintSet {
        fn insert_batch_fingerprint_values_inserted_indices_checked_into(
            &self,
            fingerprint_values: &[u64],
            admission: &mut BatchInsertedIndexAdmission,
        ) {
            assert!(
                fingerprint_values.len() >= 2,
                "malformed inserted-index test needs at least two successors"
            );
            self.value_batch_calls.fetch_add(1, Ordering::SeqCst);
            admission.clear();
            admission.attempted = fingerprint_values.len();
            admission.inserted_indices.extend([1, 0]);
        }
    }

    struct MixedBatchAdmissionFusedLevel {
        calls: AtomicUsize,
    }

    struct SiblingDuplicateBatchAdmissionFusedLevel;

    impl CompiledBfsLevel for SiblingDuplicateBatchAdmissionFusedLevel {
        fn has_fused_level(&self) -> bool {
            true
        }

        fn run_level_fused_arena(
            &self,
            arena: &[i64],
            parent_count: usize,
        ) -> Option<Result<CompiledLevelResult, BfsStepError>> {
            assert_eq!(arena, &[0, 1]);
            assert_eq!(parent_count, 2);
            Some(Ok(
                CompiledLevelResult::from_successor_arena_with_parent_indices(
                    vec![0],
                    Some(vec![1]),
                    1,
                    1,
                    parent_count,
                    1,
                    1,
                    true,
                    None,
                    None,
                    None,
                    true,
                ),
            ))
        }
    }

    struct IntraBatchDuplicateFusedLevel;

    impl CompiledBfsLevel for IntraBatchDuplicateFusedLevel {
        fn has_fused_level(&self) -> bool {
            true
        }

        fn run_level_fused_arena(
            &self,
            arena: &[i64],
            parent_count: usize,
        ) -> Option<Result<CompiledLevelResult, BfsStepError>> {
            assert_eq!(arena, &[0]);
            assert_eq!(parent_count, 1);
            Some(Ok(
                CompiledLevelResult::from_successor_arena_with_parent_indices(
                    vec![1, 1],
                    Some(vec![0, 0]),
                    2,
                    1,
                    parent_count,
                    2,
                    2,
                    true,
                    None,
                    None,
                    None,
                    true,
                ),
            ))
        }
    }

    impl CompiledBfsLevel for MixedBatchAdmissionFusedLevel {
        fn has_fused_level(&self) -> bool {
            true
        }

        fn run_level_fused_arena(
            &self,
            arena: &[i64],
            parent_count: usize,
        ) -> Option<Result<CompiledLevelResult, BfsStepError>> {
            let call = self.calls.fetch_add(1, Ordering::SeqCst);
            if call == 0 {
                assert_eq!(arena, &[0]);
                assert_eq!(parent_count, 1);
                Some(Ok(
                    CompiledLevelResult::from_successor_arena_with_parent_indices(
                        vec![1, 2],
                        Some(vec![0, 0]),
                        2,
                        1,
                        1,
                        2,
                        2,
                        true,
                        None,
                        None,
                        None,
                        true,
                    ),
                ))
            } else {
                assert_eq!(arena, &[2]);
                assert_eq!(parent_count, 1);
                Some(Ok(
                    CompiledLevelResult::from_successor_arena_with_parent_indices(
                        vec![],
                        Some(Vec::new()),
                        0,
                        1,
                        parent_count,
                        0,
                        0,
                        true,
                        None,
                        None,
                        None,
                        true,
                    ),
                ))
            }
        }
    }

    struct BatchAdmissionMissingParentMetadataFusedLevel {
        calls: AtomicUsize,
    }

    impl CompiledBfsLevel for BatchAdmissionMissingParentMetadataFusedLevel {
        fn has_fused_level(&self) -> bool {
            true
        }

        fn has_native_fused_level(&self) -> bool {
            true
        }

        fn fused_level_state_len(&self) -> Option<usize> {
            Some(1)
        }

        fn native_fused_state_constraint_count(&self) -> usize {
            1
        }

        fn native_fused_regular_invariants_checked_by_backend(&self) -> bool {
            true
        }

        fn run_level_fused_arena(
            &self,
            _arena: &[i64],
            parent_count: usize,
        ) -> Option<Result<CompiledLevelResult, BfsStepError>> {
            let call = self.calls.fetch_add(1, Ordering::SeqCst);
            if call == 0 {
                assert_eq!(parent_count, 1);
                Some(Ok(CompiledLevelResult::from_successor_arena(
                    vec![1],
                    1,
                    1,
                    1,
                    1,
                    1,
                    true,
                    None,
                    None,
                    None,
                    true,
                )))
            } else {
                Some(Ok(CompiledLevelResult::from_successor_arena(
                    vec![],
                    0,
                    1,
                    parent_count,
                    0,
                    0,
                    true,
                    None,
                    None,
                    None,
                    true,
                )))
            }
        }
    }

    struct NonBatchOutOfRangeParentAfterAppendFusedLevel;

    impl CompiledBfsLevel for NonBatchOutOfRangeParentAfterAppendFusedLevel {
        fn has_fused_level(&self) -> bool {
            true
        }

        fn run_level_fused_arena(
            &self,
            _arena: &[i64],
            parent_count: usize,
        ) -> Option<Result<CompiledLevelResult, BfsStepError>> {
            if parent_count == 1 {
                Some(Ok(
                    CompiledLevelResult::from_successor_arena_with_parent_indices(
                        vec![1, 2],
                        Some(vec![0, 1]),
                        2,
                        1,
                        1,
                        2,
                        2,
                        true,
                        None,
                        None,
                        None,
                        false,
                    ),
                ))
            } else {
                Some(Ok(
                    CompiledLevelResult::from_successor_arena_with_parent_indices(
                        vec![],
                        Some(Vec::new()),
                        0,
                        1,
                        parent_count,
                        0,
                        0,
                        true,
                        None,
                        None,
                        None,
                        false,
                    ),
                ))
            }
        }
    }

    struct BufferOverflowFusedLevel;

    impl CompiledBfsLevel for BufferOverflowFusedLevel {
        fn has_fused_level(&self) -> bool {
            true
        }

        fn has_native_fused_level(&self) -> bool {
            true
        }

        fn fused_level_state_len(&self) -> Option<usize> {
            Some(1)
        }

        fn native_fused_state_constraint_count(&self) -> usize {
            1
        }

        fn native_fused_regular_invariants_checked_by_backend(&self) -> bool {
            true
        }

        fn run_level_fused_arena(
            &self,
            _arena: &[i64],
            _parent_count: usize,
        ) -> Option<Result<CompiledLevelResult, BfsStepError>> {
            Some(Err(BfsStepError::BufferOverflow { partial_count: 0 }))
        }
    }

    struct CountingCompiledStep {
        calls: Arc<AtomicUsize>,
    }

    impl CompiledBfsStep for CountingCompiledStep {
        fn state_len(&self) -> usize {
            1
        }

        fn preserves_state_graph_successor_edges(&self) -> bool {
            true
        }

        fn step_flat(&self, state: &[i64]) -> Result<FlatBfsStepOutput, BfsStepError> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            let successors = if state == [0] { vec![1] } else { vec![] };
            let successor_count = successors.len();
            Ok(FlatBfsStepOutput::from_parts(
                successors,
                1,
                successor_count,
                successor_count as u32,
                true,
                None,
                None,
            ))
        }
    }

    struct CompleteTwoParentCompiledStep {
        calls: Arc<AtomicUsize>,
    }

    impl CompiledBfsStep for CompleteTwoParentCompiledStep {
        fn state_len(&self) -> usize {
            1
        }

        fn preserves_state_graph_successor_edges(&self) -> bool {
            true
        }

        fn step_flat(&self, state: &[i64]) -> Result<FlatBfsStepOutput, BfsStepError> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            let (successors, generated) = match state {
                [0] | [1] => (vec![2], 1),
                _ => (Vec::new(), 0),
            };
            Ok(FlatBfsStepOutput::from_parts(
                successors,
                1,
                generated,
                generated as u32,
                true,
                None,
                None,
            ))
        }
    }

    struct UnsafeLocalDedupFusedLevel {
        calls: Arc<AtomicUsize>,
    }

    impl CompiledBfsLevel for UnsafeLocalDedupFusedLevel {
        fn has_fused_level(&self) -> bool {
            true
        }

        fn has_native_fused_level(&self) -> bool {
            true
        }

        fn fused_level_state_len(&self) -> Option<usize> {
            Some(1)
        }

        fn native_fused_state_constraint_count(&self) -> usize {
            1
        }

        fn native_fused_regular_invariants_checked_by_backend(&self) -> bool {
            true
        }

        fn run_level_fused_arena(
            &self,
            _arena: &[i64],
            parent_count: usize,
        ) -> Option<Result<CompiledLevelResult, BfsStepError>> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            let result = if parent_count == 2 {
                CompiledLevelResult::from_successor_arena_with_parent_indices(
                    vec![2],
                    Some(vec![0]),
                    1,
                    1,
                    2,
                    2,
                    1,
                    true,
                    None,
                    None,
                    None,
                    true,
                )
                .with_state_graph_successors_complete(true)
            } else {
                CompiledLevelResult::from_successor_arena_with_parent_indices(
                    vec![],
                    Some(vec![]),
                    0,
                    1,
                    parent_count,
                    0,
                    0,
                    true,
                    None,
                    None,
                    None,
                    true,
                )
                .with_state_graph_successors_complete(true)
            };
            Some(Ok(result))
        }
    }

    struct UnsafeDedupingCompiledStep {
        calls: Arc<AtomicUsize>,
    }

    impl CompiledBfsStep for UnsafeDedupingCompiledStep {
        fn state_len(&self) -> usize {
            1
        }

        fn step_flat(&self, state: &[i64]) -> Result<FlatBfsStepOutput, BfsStepError> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            match state {
                [0] => Ok(FlatBfsStepOutput::from_parts(
                    vec![2],
                    1,
                    1,
                    1,
                    true,
                    None,
                    None,
                )),
                [1] => Ok(FlatBfsStepOutput::from_parts(
                    vec![],
                    1,
                    0,
                    1,
                    true,
                    None,
                    None,
                )),
                _ => Ok(FlatBfsStepOutput::from_parts(
                    vec![],
                    1,
                    0,
                    0,
                    true,
                    None,
                    None,
                )),
            }
        }
    }

    struct FatalFusedLevel;

    impl CompiledBfsLevel for FatalFusedLevel {
        fn has_fused_level(&self) -> bool {
            true
        }

        fn run_level_fused_arena(
            &self,
            _arena: &[i64],
            _parent_count: usize,
        ) -> Option<Result<CompiledLevelResult, BfsStepError>> {
            Some(Err(BfsStepError::FatalRuntimeError))
        }
    }

    struct TraceProvenanceCompiledStep;

    impl CompiledBfsStep for TraceProvenanceCompiledStep {
        fn state_len(&self) -> usize {
            1
        }

        fn step_flat(&self, state: &[i64]) -> Result<FlatBfsStepOutput, BfsStepError> {
            let successors = match state {
                [0] => vec![10],
                [1] => vec![20],
                _ => Vec::new(),
            };
            let successor_count = successors.len();
            Ok(FlatBfsStepOutput::from_parts(
                successors,
                1,
                successor_count,
                successor_count as u32,
                true,
                None,
                None,
            ))
        }
    }

    struct InvariantFailureTraceCompiledStep;

    impl CompiledBfsStep for InvariantFailureTraceCompiledStep {
        fn state_len(&self) -> usize {
            1
        }

        fn step_flat(&self, state: &[i64]) -> Result<FlatBfsStepOutput, BfsStepError> {
            match state {
                [0] => Ok(FlatBfsStepOutput::from_parts(
                    vec![],
                    1,
                    0,
                    0,
                    true,
                    None,
                    None,
                )),
                [1] => Ok(FlatBfsStepOutput::from_parts(
                    vec![2],
                    1,
                    1,
                    1,
                    false,
                    Some(0),
                    Some(0),
                )),
                _ => Ok(FlatBfsStepOutput::from_parts(
                    vec![],
                    1,
                    0,
                    0,
                    true,
                    None,
                    None,
                )),
            }
        }
    }

    struct PreflightFatalFusedLevel {
        preflight_seen: Arc<AtomicBool>,
        run_called: Arc<AtomicBool>,
    }

    impl CompiledBfsLevel for PreflightFatalFusedLevel {
        fn has_fused_level(&self) -> bool {
            true
        }

        fn preflight_fused_arena(
            &self,
            arena: &[i64],
            parent_count: usize,
        ) -> Result<(), BfsStepError> {
            assert_eq!(arena, &[0]);
            assert_eq!(parent_count, 1);
            self.preflight_seen.store(true, Ordering::SeqCst);
            Err(BfsStepError::FatalRuntimeError)
        }

        fn run_level_fused_arena(
            &self,
            _arena: &[i64],
            _parent_count: usize,
        ) -> Option<Result<CompiledLevelResult, BfsStepError>> {
            self.run_called.store(true, Ordering::SeqCst);
            Some(Err(BfsStepError::RuntimeError))
        }
    }

    #[test]
    fn compiled_bfs_fused_batch_admission_enqueues_new_successors() {
        let module = parse_module(
            r#"
---- MODULE CompiledBfsBatchAdmissionTest ----
VARIABLE x
Init == x = 0
Next == x' = x + 1
====
"#,
        );
        let config = Config {
            init: Some("Init".to_string()),
            next: Some("Next".to_string()),
            check_deadlock: false,
            ..Default::default()
        };
        let mut checker = ModelChecker::new(&module, &config);
        checker.flat_state_primary = true;
        checker.compiled_bfs_level = Some(Box::new(BatchAdmissionFusedLevel {
            calls: std::sync::atomic::AtomicUsize::new(0),
        }));
        checker.compiled_bfs_step = None;

        let layout = Arc::new(StateLayout::new(
            checker.ctx.var_registry(),
            vec![VarLayoutKind::Scalar],
        ));
        checker.flat_bfs_adapter = Some(crate::state::FlatBfsAdapter::from_layout(layout.clone()));
        let mut flat_queue = FlatBfsFrontier::new(layout);
        flat_queue.push_raw_buffer(&[0], Fingerprint(1), 0, 0);
        let mut storage = FingerprintOnlyStorage::new(
            BulkStateStorage::empty(checker.ctx.var_registry().len()),
            1,
        );

        let result = checker.run_compiled_bfs_loop(&mut storage, &mut flat_queue);

        match result {
            CheckResult::Success(stats) => {
                assert_eq!(stats.transitions, 2);
                assert_eq!(stats.max_depth, 1);
                assert_eq!(stats.states_found, 2);
            }
            other => panic!("expected batch-admission fused level success, got {other:?}"),
        }
        assert_eq!(checker.test_seen_fps_len(), 2);
        assert_eq!(flat_queue.len(), 0);
        assert_eq!(flat_queue.flat_pushed(), 3);
    }

    #[test]
    fn compiled_bfs_fused_batch_admission_rejects_non_increasing_inserted_indices() {
        let module = parse_module(
            r#"
---- MODULE CompiledBfsMalformedInsertedIndexAdmissionTest ----
VARIABLE x
Init == x = 0
Next == x' = x + 1
====
"#,
        );
        let config = Config {
            init: Some("Init".to_string()),
            next: Some("Next".to_string()),
            check_deadlock: false,
            ..Default::default()
        };
        let mut checker = ModelChecker::new(&module, &config);
        checker.flat_state_primary = true;
        checker.compiled_bfs_level = Some(Box::new(BatchAdmissionFusedLevel {
            calls: AtomicUsize::new(0),
        }));
        checker.compiled_bfs_step = None;

        let seen = Arc::new(MalformedInsertedIndexFingerprintSet::new());
        checker.set_fingerprint_storage(seen.clone() as Arc<dyn FingerprintSet>);

        let layout = Arc::new(StateLayout::new(
            checker.ctx.var_registry(),
            vec![VarLayoutKind::Scalar],
        ));
        checker.flat_bfs_adapter = Some(crate::state::FlatBfsAdapter::from_layout(layout.clone()));
        let mut flat_queue = FlatBfsFrontier::new(layout);
        flat_queue.push_raw_buffer(&[0], Fingerprint(1), 0, 0);
        let mut storage = FingerprintOnlyStorage::new(
            BulkStateStorage::empty(checker.ctx.var_registry().len()),
            1,
        );

        let result = checker.run_compiled_bfs_loop(&mut storage, &mut flat_queue);

        match result {
            CheckResult::Error { error, .. } => {
                let message = error.to_string();
                assert!(
                    message.contains("non-increasing inserted index 0"),
                    "malformed inserted-index admission should fail closed, got: {message}"
                );
            }
            other => panic!("expected malformed inserted-index admission error, got {other:?}"),
        }
        assert_eq!(seen.value_batch_calls.load(Ordering::SeqCst), 1);
        assert_eq!(flat_queue.flat_pushed(), 1);
    }

    #[test]
    fn compiled_bfs_fused_batch_admission_uses_borrowed_trust_cg_sidecars() {
        let module = parse_module(
            r#"
---- MODULE CompiledBfsBorrowedBatchAdmissionTest ----
VARIABLE x
Init == x = 0
Next == x' = x + 1
====
"#,
        );
        let config = Config {
            init: Some("Init".to_string()),
            next: Some("Next".to_string()),
            check_deadlock: false,
            ..Default::default()
        };
        let mut checker = ModelChecker::new(&module, &config);
        checker.flat_state_primary = true;
        checker.compiled_bfs_level = Some(Box::new(TrustCgBorrowedBatchAdmissionFusedLevel {
            calls: AtomicUsize::new(0),
        }));
        checker.compiled_bfs_step = None;

        let seen = Arc::new(BorrowedBatchOnlyFingerprintSet::new());
        checker.set_fingerprint_storage(seen.clone() as Arc<dyn FingerprintSet>);

        let layout = Arc::new(StateLayout::new(
            checker.ctx.var_registry(),
            vec![VarLayoutKind::Scalar],
        ));
        checker.flat_bfs_adapter = Some(crate::state::FlatBfsAdapter::from_layout(layout.clone()));
        let mut flat_queue = FlatBfsFrontier::new(layout);
        flat_queue.push_raw_buffer(&[0], Fingerprint(1), 0, 0);
        let mut storage = FingerprintOnlyStorage::new(
            BulkStateStorage::empty(checker.ctx.var_registry().len()),
            1,
        );

        let result = checker.run_compiled_bfs_loop(&mut storage, &mut flat_queue);

        match result {
            CheckResult::Success(stats) => {
                assert_eq!(stats.transitions, 3);
                assert_eq!(stats.max_depth, 2);
                assert_eq!(stats.states_found, 3);
            }
            other => panic!("expected borrowed batch-admission fused level success, got {other:?}"),
        }
        assert!(
            seen.value_batch_calls.load(Ordering::SeqCst) > 0,
            "borrowed trust-codegen sidecar batch path should use the compiled/fused prepared batch admission callsite"
        );
        assert!(
            seen.scratch_reused_calls.load(Ordering::SeqCst) > 0,
            "borrowed trust-codegen sidecar batch path should reuse caller-owned outcome scratch across levels"
        );
        assert_eq!(checker.test_seen_fps_len(), 3);
        assert_eq!(flat_queue.len(), 0);
        assert_eq!(flat_queue.flat_pushed(), 4);
    }

    #[test]
    fn compiled_bfs_preserves_borrowed_trust_cg_sidecars_for_recursive_canonical_successors() {
        let module = parse_module(
            r#"
---- MODULE CompiledBfsRecursiveBorrowedBatchAdmissionTest ----
VARIABLE q
Init == q = <<>>
Next == q' = q
====
"#,
        );
        let config = Config {
            init: Some("Init".to_string()),
            next: Some("Next".to_string()),
            check_deadlock: false,
            ..Default::default()
        };
        let mut checker = ModelChecker::new(&module, &config);
        checker.flat_state_primary = true;
        checker.compiled_bfs_level = Some(Box::new(
            TrustCgRecursiveCanonicalBatchAdmissionFusedLevel {
                calls: AtomicUsize::new(0),
            },
        ));
        checker.compiled_bfs_step = None;

        let seen = Arc::new(BorrowedBatchOnlyFingerprintSet::new());
        checker.set_fingerprint_storage(seen.clone() as Arc<dyn FingerprintSet>);

        let layout = recursive_sequence_layout(&checker);
        checker.flat_bfs_adapter = Some(crate::state::FlatBfsAdapter::from_layout(layout.clone()));
        let mut flat_queue = FlatBfsFrontier::new(layout);
        flat_queue.push_raw_buffer(&[0, 0, 0], Fingerprint(1), 0, 0);
        let mut storage = FingerprintOnlyStorage::new(
            BulkStateStorage::empty(checker.ctx.var_registry().len()),
            checker.ctx.var_registry().len(),
        );

        let result = checker.run_compiled_bfs_loop(&mut storage, &mut flat_queue);

        match result {
            CheckResult::Success(stats) => {
                assert_eq!(stats.transitions, 2);
                assert_eq!(stats.max_depth, 1);
                assert_eq!(stats.states_found, 2);
            }
            other => panic!("expected recursive borrowed batch-admission success, got {other:?}"),
        }
        assert!(
            seen.value_batch_calls.load(Ordering::SeqCst) > 0,
            "canonical recursive successors should preserve trust-codegen fingerprint sidecars"
        );
        assert_eq!(checker.test_seen_fps_len(), 2);
        assert_eq!(flat_queue.len(), 0);
        assert_eq!(flat_queue.flat_pushed(), 3);
    }

    #[test]
    fn compiled_bfs_canonicalizes_recursive_successors_before_recomputing_batch_fingerprints() {
        let module = parse_module(
            r#"
---- MODULE CompiledBfsRecursiveCanonicalizingFallbackTest ----
VARIABLE q
Init == q = <<>>
Next == q' = q
====
"#,
        );
        let config = Config {
            init: Some("Init".to_string()),
            next: Some("Next".to_string()),
            check_deadlock: false,
            ..Default::default()
        };
        let mut checker = ModelChecker::new(&module, &config);
        checker.flat_state_primary = true;
        checker.compiled_bfs_level = Some(Box::new(TrustCgRecursiveNoncanonicalTailFusedLevel {
            calls: AtomicUsize::new(0),
        }));
        checker.compiled_bfs_step = None;

        let seen = Arc::new(StagedBatchOnlyFingerprintSet::new());
        checker.set_fingerprint_storage(seen.clone() as Arc<dyn FingerprintSet>);

        let layout = recursive_sequence_layout(&checker);
        checker.flat_bfs_adapter = Some(crate::state::FlatBfsAdapter::from_layout(layout.clone()));
        let mut flat_queue = FlatBfsFrontier::new(layout);
        flat_queue.push_raw_buffer(&[0, 0, 0], Fingerprint(1), 0, 0);
        let mut storage = FingerprintOnlyStorage::new(
            BulkStateStorage::empty(checker.ctx.var_registry().len()),
            checker.ctx.var_registry().len(),
        );

        let result = checker.run_compiled_bfs_loop(&mut storage, &mut flat_queue);

        match result {
            CheckResult::Success(stats) => {
                assert_eq!(stats.transitions, 1);
                assert_eq!(stats.max_depth, 1);
                assert_eq!(stats.states_found, 1);
            }
            other => panic!("expected recursive canonicalizing fallback success, got {other:?}"),
        }
        assert!(
            seen.staged_batch_calls.load(Ordering::SeqCst) > 0,
            "canonicalizing fallback should clear sidecars and stage recomputed fingerprints"
        );
        assert_eq!(checker.test_seen_fps_len(), 1);
        assert_eq!(flat_queue.len(), 0);
        assert_eq!(flat_queue.flat_pushed(), 2);
    }

    #[test]
    fn compiled_bfs_borrowed_trust_cg_batch_admission_preserves_inserted_prefix_on_storage_fault() {
        let module = parse_module(
            r#"
---- MODULE CompiledBfsBorrowedBatchAdmissionPrefixFaultTest ----
VARIABLE x
Init == x = 0
Next == x' = x + 1
====
"#,
        );
        let config = Config {
            init: Some("Init".to_string()),
            next: Some("Next".to_string()),
            check_deadlock: false,
            ..Default::default()
        };
        let mut checker = ModelChecker::new(&module, &config);
        checker.flat_state_primary = true;
        checker.compiled_bfs_level = Some(Box::new(TrustCgBorrowedBatchStorageFaultFusedLevel));
        checker.compiled_bfs_step = None;

        let seen = Arc::new(BorrowedBatchOnlyFingerprintSet::with_fault_at_value_batch_index(3));
        checker.set_fingerprint_storage(seen.clone() as Arc<dyn FingerprintSet>);

        let layout = Arc::new(StateLayout::new(
            checker.ctx.var_registry(),
            vec![VarLayoutKind::Scalar],
        ));
        checker.flat_bfs_adapter = Some(crate::state::FlatBfsAdapter::from_layout(layout.clone()));
        let mut flat_queue = FlatBfsFrontier::new(layout);
        let parent_fp = crate::check::model_checker::invariants::fingerprint_flat_compiled(&[0]);
        checker
            .mark_state_seen_checked_with_current(
                parent_fp,
                &ArrayState::from_values(vec![Value::int(0)]),
                None,
                0,
                None,
            )
            .expect("seed parent fingerprint");
        let duplicate_fp = crate::check::model_checker::invariants::fingerprint_flat_compiled(&[2]);
        checker
            .mark_state_seen_checked_with_current(
                duplicate_fp,
                &ArrayState::from_values(vec![Value::int(2)]),
                None,
                1,
                None,
            )
            .expect("seed duplicate successor fingerprint");
        flat_queue.push_raw_buffer(&[0], parent_fp, 0, 0);
        let mut storage = FingerprintOnlyStorage::new(
            BulkStateStorage::empty(checker.ctx.var_registry().len()),
            1,
        );

        let result = checker.run_compiled_bfs_loop(&mut storage, &mut flat_queue);

        match result {
            CheckResult::Error {
                error: CheckError::Infra(InfraCheckError::FingerprintOverflow { dropped, detail }),
                stats,
                ..
            } => {
                assert_eq!(dropped, 1);
                assert!(
                    detail.contains("synthetic prefix fault"),
                    "storage fault detail should be preserved, got: {detail}"
                );
                assert_eq!(stats.states_found, checker.test_seen_fps_len());
                assert_eq!(stats.states_found, 4);
            }
            other => panic!("expected borrowed trust-codegen prefix storage fault, got {other:?}"),
        }

        assert_eq!(
            seen.value_batch_calls.load(Ordering::SeqCst),
            1,
            "borrowed trust-codegen path should use the compiled/fused prepared batch admission callsite"
        );
        assert_eq!(checker.test_seen_fps_len(), 4);
        assert_eq!(flat_queue.len(), 2);
        assert_eq!(flat_queue.flat_pushed(), 3);
        let (remaining, count) = flat_queue
            .remaining_arena()
            .expect("inserted prefix should remain queued after storage fault");
        assert_eq!(count, 2);
        assert_eq!(remaining, &[1, 3]);
    }

    #[test]
    fn compiled_bfs_borrowed_trust_cg_batch_admission_records_parent_trace_provenance() {
        let module = two_parent_liveness_module();
        let config = Config {
            init: Some("Init".to_string()),
            next: Some("Next".to_string()),
            check_deadlock: false,
            ..Default::default()
        };
        let mut checker = ModelChecker::new(&module, &config);
        checker.flat_state_primary = true;
        checker.set_trace_file(TraceFile::create_temp().expect("trace file should initialize"));
        let checkpoint_dir = tempfile::tempdir().expect("checkpoint temp dir");
        checker.set_checkpoint(
            checkpoint_dir.path().to_path_buf(),
            std::time::Duration::from_secs(60),
        );
        checker.compiled_bfs_level =
            Some(Box::new(TrustCgBorrowedBatchTraceProvenanceFusedLevel {
                calls: AtomicUsize::new(0),
            }));
        checker.compiled_bfs_step = None;

        let seen = Arc::new(BorrowedBatchOnlyFingerprintSet::new());
        checker.set_fingerprint_storage(seen.clone() as Arc<dyn FingerprintSet>);

        checker.trace.cached_next_name = Some("Next".to_string());
        checker.trace.cached_resolved_next_name = Some("Next".to_string());
        let layout = Arc::new(StateLayout::new(
            checker.ctx.var_registry(),
            vec![VarLayoutKind::Scalar],
        ));
        checker.flat_bfs_adapter = Some(crate::state::FlatBfsAdapter::from_layout(layout.clone()));

        let fp0 = crate::check::model_checker::invariants::fingerprint_flat_compiled(&[0]);
        checker
            .mark_state_seen_fp_only_checked(fp0, None, 0)
            .expect("seed x=0");
        let loc0 = checker.trace.last_inserted_trace_loc;
        let fp1 = crate::check::model_checker::invariants::fingerprint_flat_compiled(&[1]);
        checker
            .mark_state_seen_fp_only_checked(fp1, None, 0)
            .expect("seed x=1");
        let loc1 = checker.trace.last_inserted_trace_loc;

        let mut flat_queue = FlatBfsFrontier::new(layout);
        flat_queue.push_raw_buffer(&[0], fp0, 0, loc0);
        flat_queue.push_raw_buffer(&[1], fp1, 0, loc1);
        let mut storage = FingerprintOnlyStorage::new(
            BulkStateStorage::empty(checker.ctx.var_registry().len()),
            checker.ctx.var_registry().len(),
        );

        let result = checker.run_compiled_bfs_loop(&mut storage, &mut flat_queue);

        assert!(matches!(result, CheckResult::Success(_)), "got {result:?}");
        assert!(
            seen.value_batch_calls.load(Ordering::SeqCst) > 0,
            "borrowed trust-codegen path should use the compiled/fused prepared batch admission callsite"
        );
        let successor_10_fp =
            crate::check::model_checker::invariants::fingerprint_flat_compiled(&[10]);
        let successor_20_fp =
            crate::check::model_checker::invariants::fingerprint_flat_compiled(&[20]);
        let parents = checker
            .trace
            .trace_file
            .as_mut()
            .expect("trace file still installed")
            .build_parents_map()
            .expect("parents map should scan");
        assert_eq!(parents.get(&successor_10_fp), Some(&fp0));
        assert_eq!(parents.get(&successor_20_fp), Some(&fp1));
        assert_eq!(checker.trace.trace_locs.get(&successor_10_fp), Some(32));
        assert_eq!(checker.trace.trace_locs.get(&successor_20_fp), Some(48));
        assert_eq!(checker.trace.depths.get(&successor_10_fp), Some(&1));
        assert_eq!(checker.trace.depths.get(&successor_20_fp), Some(&1));
    }

    #[test]
    fn compiled_bfs_fused_batch_admission_uses_frontier_payload_witness_for_fp_only_duplicate() {
        let module = two_parent_liveness_module();
        let config = Config {
            init: Some("Init".to_string()),
            next: Some("Next".to_string()),
            check_deadlock: false,
            ..Default::default()
        };
        let mut checker = ModelChecker::new(&module, &config);
        checker.flat_state_primary = true;
        checker.compiled_bfs_level = Some(Box::new(SiblingDuplicateBatchAdmissionFusedLevel));
        checker.compiled_bfs_step = None;
        let fresh_witness_census = crate::storage::FingerprintPayloadWitnesses::new().census();

        let (mut flat_queue, mut storage, fp0, fp1) = seed_two_parent_flat_frontier(&mut checker);
        assert_eq!(checker.test_seen_fps_len(), 2);

        let result = checker.run_compiled_bfs_loop(&mut storage, &mut flat_queue);

        match result {
            CheckResult::Success(stats) => {
                assert_eq!(stats.transitions, 1);
                assert_eq!(stats.states_found, 2);
                assert_eq!(stats.max_depth, 1);
            }
            other => panic!(
                "expected sibling duplicate to be admitted through compiled-flat payload witness, got {other:?}"
            ),
        }
        assert_eq!(checker.test_seen_fps_len(), 2);
        assert_eq!(flat_queue.len(), 0);
        assert_eq!(
            checker
                .state_storage
                .compiled_flat_payload_witnesses
                .confirm_flat_i64_slots(fp0, &[0]),
            None,
            "normal compiled completion must release the global witness arena"
        );
        assert_eq!(
            checker
                .state_storage
                .compiled_flat_payload_witnesses
                .confirm_flat_i64_slots(fp1, &[1]),
            None,
            "normal compiled completion must release the global witness arena"
        );
        assert_eq!(
            checker
                .state_storage
                .compiled_flat_payload_witnesses
                .census(),
            fresh_witness_census
        );
    }

    #[test]
    fn compiled_bfs_fused_batch_duplicate_uses_local_payload_witness_without_global_copy() {
        let module = parse_module(
            r#"
---- MODULE CompiledBfsLocalBatchPayloadWitnessTest ----
VARIABLE x
Init == x = 0
Next == x' = x + 1
====
"#,
        );
        let config = Config {
            init: Some("Init".to_string()),
            next: Some("Next".to_string()),
            check_deadlock: false,
            ..Default::default()
        };
        let mut checker = ModelChecker::new(&module, &config);
        checker.flat_state_primary = true;

        let layout = Arc::new(StateLayout::new(
            checker.ctx.var_registry(),
            vec![VarLayoutKind::Scalar],
        ));
        checker.flat_bfs_adapter = Some(crate::state::FlatBfsAdapter::from_layout(layout.clone()));
        let mut flat_queue = FlatBfsFrontier::new(layout);
        let parent_fp = crate::check::model_checker::invariants::fingerprint_flat_compiled(&[0]);
        flat_queue.push_raw_buffer(&[0], parent_fp, 0, 0);

        let duplicate_fp = crate::check::model_checker::invariants::fingerprint_flat_compiled(&[7]);
        let parent_indices = [0usize, 0usize];
        let fingerprint_values = [duplicate_fp.0, duplicate_fp.0];
        let inserted_indices = [0usize];
        let successor_arena = [7i64, 7i64];
        let registry = checker.ctx.var_registry().clone();

        assert_eq!(
            checker
                .state_storage
                .compiled_flat_payload_witnesses
                .confirm_flat_i64_slots(duplicate_fp, &[7]),
            None
        );
        assert!(checker.fp_only_batch_duplicate_payloads_confirmed(
            &registry,
            &flat_queue,
            CompiledSuccessorParentIndices::Usize(&parent_indices),
            &fingerprint_values,
            &inserted_indices,
            &successor_arena,
            1,
            2,
        ));
        assert_eq!(
            checker
                .state_storage
                .compiled_flat_payload_witnesses
                .confirm_flat_i64_slots(duplicate_fp, &[7]),
            None,
            "same-batch duplicate confirmation must use the local batch witness map"
        );
    }

    #[test]
    fn compiled_bfs_fused_batch_admission_skips_already_present_and_enqueues_later_inserted() {
        let module = parse_module(
            r#"
---- MODULE CompiledBfsBatchAdmissionMixedTest ----
VARIABLE x
Init == x = 0
Next == x' = x + 1
====
"#,
        );
        let config = Config {
            init: Some("Init".to_string()),
            next: Some("Next".to_string()),
            check_deadlock: false,
            ..Default::default()
        };
        let mut checker = ModelChecker::new(&module, &config);
        checker.flat_state_primary = true;
        checker.compiled_bfs_level = Some(Box::new(MixedBatchAdmissionFusedLevel {
            calls: AtomicUsize::new(0),
        }));
        checker.compiled_bfs_step = None;

        let layout = Arc::new(StateLayout::new(
            checker.ctx.var_registry(),
            vec![VarLayoutKind::Scalar],
        ));
        checker.flat_bfs_adapter = Some(crate::state::FlatBfsAdapter::from_layout(layout.clone()));
        let mut flat_queue = FlatBfsFrontier::new(layout);
        let parent_fp = crate::check::model_checker::invariants::fingerprint_flat_compiled(&[0]);
        flat_queue.push_raw_buffer(&[0], parent_fp, 0, 0);
        checker
            .mark_state_seen_checked_with_current(
                parent_fp,
                &ArrayState::from_values(vec![Value::int(0)]),
                None,
                0,
                None,
            )
            .expect("seed parent fingerprint");
        let duplicate_fp = crate::check::model_checker::invariants::fingerprint_flat_compiled(&[1]);
        checker
            .mark_state_seen_checked_with_current(
                duplicate_fp,
                &ArrayState::from_values(vec![Value::int(1)]),
                None,
                1,
                None,
            )
            .expect("seed duplicate successor fingerprint");
        let mut storage = FingerprintOnlyStorage::new(
            BulkStateStorage::empty(checker.ctx.var_registry().len()),
            1,
        );

        let result = checker.run_compiled_bfs_loop(&mut storage, &mut flat_queue);

        match result {
            CheckResult::Success(stats) => {
                assert_eq!(stats.transitions, 2);
                assert_eq!(stats.max_depth, 1);
                assert_eq!(stats.states_found, 3);
            }
            other => panic!("expected mixed batch-admission fused level success, got {other:?}"),
        }
        assert_eq!(checker.test_seen_fps_len(), 3);
        assert_eq!(flat_queue.len(), 0);
        assert_eq!(flat_queue.flat_pushed(), 2);
    }

    #[test]
    fn compiled_bfs_fused_batch_admission_enqueues_inserted_prefix_before_storage_fault() {
        let module = parse_module(
            r#"
---- MODULE CompiledBfsBatchAdmissionPrefixFaultTest ----
VARIABLE x
Init == x = 0
Next == x' = x + 1
====
"#,
        );
        let config = Config {
            init: Some("Init".to_string()),
            next: Some("Next".to_string()),
            check_deadlock: false,
            ..Default::default()
        };
        let mut checker = ModelChecker::new(&module, &config);
        checker.flat_state_primary = true;
        checker.compiled_bfs_level = Some(Box::new(BatchAdmissionFusedLevel {
            calls: std::sync::atomic::AtomicUsize::new(0),
        }));
        checker.compiled_bfs_step = None;
        let storage = FingerprintStorage::mmap(2, None)
            .expect("test mmap fingerprint storage should initialize");
        checker.set_fingerprint_storage(Arc::new(storage) as Arc<dyn FingerprintSet>);

        let layout = Arc::new(StateLayout::new(
            checker.ctx.var_registry(),
            vec![VarLayoutKind::Scalar],
        ));
        checker.flat_bfs_adapter = Some(crate::state::FlatBfsAdapter::from_layout(layout.clone()));
        let mut flat_queue = FlatBfsFrontier::new(layout);
        let parent_fp = crate::check::model_checker::invariants::fingerprint_flat_compiled(&[0]);
        flat_queue.push_raw_buffer(&[0], parent_fp, 0, 0);
        checker
            .mark_state_seen_fp_only_checked(parent_fp, None, 0)
            .expect("seed parent fingerprint");
        let mut storage = FingerprintOnlyStorage::new(
            BulkStateStorage::empty(checker.ctx.var_registry().len()),
            1,
        );

        let result = checker.run_compiled_bfs_loop(&mut storage, &mut flat_queue);

        match result {
            CheckResult::Error {
                error: CheckError::Infra(InfraCheckError::FingerprintOverflow { dropped, .. }),
                stats,
                ..
            } => {
                assert_eq!(dropped, 1);
                assert_eq!(stats.states_found, checker.test_seen_fps_len());
                assert_eq!(stats.states_found, 2);
            }
            other => panic!("expected prefix storage fault, got {other:?}"),
        }

        assert_eq!(checker.test_seen_fps_len(), 2);
        assert_eq!(flat_queue.len(), 1);
        assert_eq!(flat_queue.flat_pushed(), 2);
    }

    #[test]
    fn compiled_bfs_fused_batch_admission_treats_duplicate_at_mmap_load_limit_as_already_present() {
        let module = parse_module(
            r#"
---- MODULE CompiledBfsBatchAdmissionMmapDuplicateLimitTest ----
VARIABLE x
Init == x = 0
Next == x' = x
====
"#,
        );
        let config = Config {
            init: Some("Init".to_string()),
            next: Some("Next".to_string()),
            check_deadlock: false,
            ..Default::default()
        };
        let mut checker = ModelChecker::new(&module, &config);
        checker.flat_state_primary = true;
        checker.compiled_bfs_level = Some(Box::new(DuplicateSuccessorFusedLevel));
        checker.compiled_bfs_step = None;
        let storage = FingerprintStorage::mmap(1, None)
            .expect("test mmap fingerprint storage should initialize");
        checker.set_fingerprint_storage(Arc::new(storage) as Arc<dyn FingerprintSet>);

        let layout = Arc::new(StateLayout::new(
            checker.ctx.var_registry(),
            vec![VarLayoutKind::Scalar],
        ));
        checker.flat_bfs_adapter = Some(crate::state::FlatBfsAdapter::from_layout(layout.clone()));
        let mut flat_queue = FlatBfsFrontier::new(layout);
        let parent_fp = crate::check::model_checker::invariants::fingerprint_flat_compiled(&[0]);
        checker
            .mark_state_seen_fp_only_checked(parent_fp, None, 0)
            .expect("seed parent fingerprint");
        flat_queue.push_raw_buffer(&[0], parent_fp, 0, 0);
        let mut storage = FingerprintOnlyStorage::new(
            BulkStateStorage::empty(checker.ctx.var_registry().len()),
            1,
        );

        let result = checker.run_compiled_bfs_loop(&mut storage, &mut flat_queue);

        match result {
            CheckResult::Success(stats) => {
                assert_eq!(stats.transitions, 1);
                assert_eq!(stats.states_found, 1);
                assert_eq!(stats.max_depth, 1);
            }
            other => panic!("expected duplicate-at-load-limit success, got {other:?}"),
        }
        assert_eq!(checker.test_seen_fps_len(), 1);
        assert_eq!(flat_queue.len(), 0);
        assert_eq!(flat_queue.flat_pushed(), 1);
    }

    #[test]
    fn compiled_bfs_fused_batch_admission_rejects_compiled_flat_payload_collision() {
        let module = parse_module(
            r#"
---- MODULE CompiledBfsBatchAdmissionCollisionTest ----
VARIABLE x
Init == x = 0
Next == x' = x + 1
====
"#,
        );
        let config = Config {
            init: Some("Init".to_string()),
            next: Some("Next".to_string()),
            check_deadlock: false,
            ..Default::default()
        };
        let mut checker = ModelChecker::new(&module, &config);
        checker.flat_state_primary = true;
        checker.compiled_bfs_level = Some(Box::new(DuplicateCollisionBatchAdmissionFusedLevel));
        checker.compiled_bfs_step = None;

        let layout = Arc::new(StateLayout::new(
            checker.ctx.var_registry(),
            vec![VarLayoutKind::Scalar],
        ));
        checker.flat_bfs_adapter = Some(crate::state::FlatBfsAdapter::from_layout(layout.clone()));
        let mut flat_queue = FlatBfsFrontier::new(layout);
        let parent_fp = crate::check::model_checker::invariants::fingerprint_flat_compiled(&[0]);
        checker
            .mark_state_seen_fp_only_checked(parent_fp, None, 0)
            .expect("seed parent fingerprint");
        let duplicate_fp = crate::check::model_checker::invariants::fingerprint_flat_compiled(&[1]);
        checker
            .mark_state_seen_fp_only_checked(duplicate_fp, None, 0)
            .expect("seed duplicate fingerprint without payload witness");
        flat_queue.push_raw_buffer(&[0], parent_fp, 0, 0);
        let mut storage = FingerprintOnlyStorage::new(
            BulkStateStorage::empty(checker.ctx.var_registry().len()),
            1,
        );

        let result = checker.run_compiled_bfs_loop(&mut storage, &mut flat_queue);

        match result {
            CheckResult::Error { error, .. } => {
                let message = error.to_string();
                assert!(
                    message.contains("prepared_fingerprint_admission")
                        && message.contains("reason_code=canonical_payload_mismatch")
                        && message.contains("payload_witness=compiled_flat_xxh3"),
                    "compiled-flat batch duplicate collision must fail closed, got: {message}"
                );
            }
            other => panic!("expected compiled-flat batch collision error, got {other:?}"),
        }
        assert_eq!(checker.test_seen_fps_len(), 2);
        assert_eq!(flat_queue.flat_pushed(), 1);
    }

    #[test]
    fn compiled_bfs_fused_success_rejects_short_parents_processed() {
        let module = two_parent_liveness_module();
        let config = Config {
            init: Some("Init".to_string()),
            next: Some("Next".to_string()),
            check_deadlock: false,
            ..Default::default()
        };
        let mut checker = ModelChecker::new(&module, &config);
        checker.flat_state_primary = true;
        checker.compiled_bfs_level = Some(Box::new(ShortSuccessfulFusedLevel));
        checker.compiled_bfs_step = None;

        let (mut flat_queue, mut storage, _fp0, _fp1) = seed_two_parent_flat_frontier(&mut checker);

        let result = checker.run_compiled_bfs_loop(&mut storage, &mut flat_queue);

        match result {
            CheckResult::Error { error, .. } => {
                let message = error.to_string();
                assert!(
                    message.contains("processed 1 parents for current level of 2"),
                    "short parents_processed should fail closed, got: {message}"
                );
            }
            other => panic!("expected short parents_processed error, got {other:?}"),
        }
        assert_eq!(flat_queue.len(), 2);
    }

    #[test]
    fn compiled_bfs_fused_deadlock_accepts_complete_level_metadata() {
        let module = two_parent_liveness_module();
        let config = Config {
            init: Some("Init".to_string()),
            next: Some("Next".to_string()),
            check_deadlock: true,
            ..Default::default()
        };
        let mut checker = ModelChecker::new(&module, &config);
        checker.flat_state_primary = true;
        checker.compiled_bfs_level = Some(Box::new(CompleteDeadlockMetadataFusedLevel));
        checker.compiled_bfs_step = None;

        let (mut flat_queue, mut storage, _fp0, _fp1) = seed_two_parent_flat_frontier(&mut checker);

        let result = checker.run_compiled_bfs_loop(&mut storage, &mut flat_queue);

        match result {
            CheckResult::Deadlock { trace, stats } => {
                assert_eq!(trace.len(), 1);
                assert_eq!(stats.transitions, 1);
            }
            other => panic!("expected complete-metadata fused deadlock, got {other:?}"),
        }
    }

    #[test]
    fn compiled_bfs_fused_zero_generated_without_raw_gap_is_not_deadlock() {
        let module = two_parent_liveness_module();
        let config = Config {
            init: Some("Init".to_string()),
            next: Some("Next".to_string()),
            check_deadlock: true,
            ..Default::default()
        };
        let mut checker = ModelChecker::new(&module, &config);
        checker.flat_state_primary = true;
        checker.compiled_bfs_level = Some(Box::new(ZeroGeneratedDeadlockMetadataFusedLevel {
            first_zero_generated_parent_idx: tla_trust_cg::TRUST_CG_BFS_NO_INDEX,
        }));
        checker.compiled_bfs_step = None;

        let (mut flat_queue, mut storage, _fp0, _fp1) = seed_two_parent_flat_frontier(&mut checker);

        let result = checker.run_compiled_bfs_loop(&mut storage, &mut flat_queue);

        match result {
            CheckResult::Success(stats) => {
                assert_eq!(stats.transitions, 0);
            }
            other => panic!("expected zero-generated fused success without raw gap, got {other:?}"),
        }
    }

    #[test]
    fn compiled_bfs_fused_zero_generated_with_raw_gap_is_deadlock() {
        let module = two_parent_liveness_module();
        let config = Config {
            init: Some("Init".to_string()),
            next: Some("Next".to_string()),
            check_deadlock: true,
            ..Default::default()
        };
        let mut checker = ModelChecker::new(&module, &config);
        checker.flat_state_primary = true;
        checker.compiled_bfs_level = Some(Box::new(ZeroGeneratedDeadlockMetadataFusedLevel {
            first_zero_generated_parent_idx: 0,
        }));
        checker.compiled_bfs_step = None;

        let (mut flat_queue, mut storage, _fp0, _fp1) = seed_two_parent_flat_frontier(&mut checker);

        let result = checker.run_compiled_bfs_loop(&mut storage, &mut flat_queue);

        match result {
            CheckResult::Deadlock { trace, stats } => {
                assert_eq!(trace.len(), 1);
                assert_eq!(stats.transitions, 0);
            }
            other => panic!("expected zero-generated fused deadlock with raw gap, got {other:?}"),
        }
    }

    /// Module whose two initial states are BOTH terminal per the TERMINAL
    /// config (no enabled Next transition). A correct checker must report
    /// Success — the interpreter deadlock check (`check_deadlock` ->
    /// `is_terminal_state_array`) exempts them. The compiled BFS deadlock
    /// branches never evaluate TerminalSpec, so routing this config through
    /// the compiled loop misreported the terminal states as a deadlock.
    fn two_terminal_parent_module() -> tla_core::ast::Module {
        parse_module(
            r#"
---- MODULE CompiledBfsTerminalVetoTest ----
VARIABLE x
Init == x = 2 \/ x = 3
Next == x = 0 /\ x' = 1
IsTerminal == x = 2 \/ x = 3
====
"#,
        )
    }

    /// Seed a flat frontier with the two terminal states x=2 and x=3
    /// (mirrors `seed_two_parent_flat_frontier`, which seeds x=0/x=1).
    fn seed_two_terminal_parent_flat_frontier(
        checker: &mut ModelChecker<'_>,
    ) -> (FlatBfsFrontier, FingerprintOnlyStorage) {
        checker.trace.cached_next_name = Some("Next".to_string());
        checker.trace.cached_resolved_next_name = Some("Next".to_string());

        let layout = Arc::new(StateLayout::new(
            checker.ctx.var_registry(),
            vec![VarLayoutKind::Scalar],
        ));
        checker.flat_bfs_adapter = Some(crate::state::FlatBfsAdapter::from_layout(layout.clone()));

        let fp2 = crate::check::model_checker::invariants::fingerprint_flat_compiled(&[2]);
        let fp3 = crate::check::model_checker::invariants::fingerprint_flat_compiled(&[3]);
        checker
            .mark_state_seen_fp_only_checked(fp2, None, 0)
            .expect("seed x=2");
        let loc2 = checker.trace.last_inserted_trace_loc;
        checker
            .mark_state_seen_fp_only_checked(fp3, None, 0)
            .expect("seed x=3");
        let loc3 = checker.trace.last_inserted_trace_loc;

        let mut flat_queue = FlatBfsFrontier::new(layout);
        flat_queue.push_raw_buffer(&[2], fp2, 0, loc2);
        flat_queue.push_raw_buffer(&[3], fp3, 0, loc3);

        let storage = FingerprintOnlyStorage::new(
            BulkStateStorage::empty(checker.ctx.var_registry().len()),
            checker.ctx.var_registry().len(),
        );
        (flat_queue, storage)
    }

    /// Soundness pin: a TERMINAL config must veto compiled-BFS level
    /// eligibility. Neither compiled deadlock site (the fused-level metadata
    /// branch nor the per-parent no-successors branch) evaluates TerminalSpec,
    /// so an otherwise-eligible run with `config.terminal` set must be routed
    /// to the interpreter loop, whose `check_deadlock` applies the
    /// `is_terminal_state_array` exemption.
    #[test]
    fn compiled_bfs_terminal_config_vetoes_level_eligibility() {
        use crate::config::TerminalSpec;

        let module = two_terminal_parent_module();

        // Baseline: identical harness WITHOUT a terminal spec is eligible,
        // proving the veto below is caused by `config.terminal` alone.
        let config_no_terminal = Config {
            init: Some("Init".to_string()),
            next: Some("Next".to_string()),
            check_deadlock: true,
            ..Default::default()
        };
        let mut checker = ModelChecker::new(&module, &config_no_terminal);
        checker.flat_state_primary = true;
        checker.compiled_bfs_level = Some(Box::new(ZeroGeneratedDeadlockMetadataFusedLevel {
            first_zero_generated_parent_idx: 0,
        }));
        checker.compiled_bfs_step = None;
        let _seeded = seed_two_terminal_parent_flat_frontier(&mut checker);
        assert!(
            checker.compiled_bfs_level_eligible(),
            "harness must be compiled-BFS eligible without TERMINAL, or this test pins nothing"
        );

        let config_terminal = Config {
            init: Some("Init".to_string()),
            next: Some("Next".to_string()),
            check_deadlock: true,
            terminal: Some(TerminalSpec::Operator("IsTerminal".to_string())),
            ..Default::default()
        };
        let mut checker = ModelChecker::new(&module, &config_terminal);
        checker.flat_state_primary = true;
        checker.compiled_bfs_level = Some(Box::new(ZeroGeneratedDeadlockMetadataFusedLevel {
            first_zero_generated_parent_idx: 0,
        }));
        checker.compiled_bfs_step = None;
        let _seeded = seed_two_terminal_parent_flat_frontier(&mut checker);
        assert!(
            !checker.compiled_bfs_level_eligible(),
            "TERMINAL config must veto compiled BFS: its deadlock reporting never \
             evaluates TerminalSpec, so terminal states would be misreported as deadlocks"
        );
    }

    /// End-to-end soundness pin for the terminal-vs-deadlock verdict: with a
    /// TERMINAL config, `run_compiled_bfs_loop` must NOT trust a fused level
    /// that truthfully reports zero successors for a terminal parent — that
    /// used to surface as `CheckResult::Deadlock`. The TerminalSpec veto in
    /// `compiled_bfs_level_eligible` routes the run to the interpreter loop,
    /// which exempts terminal states and completes with Success.
    #[test]
    fn compiled_bfs_terminal_state_not_misreported_as_deadlock() {
        use crate::config::TerminalSpec;

        let module = two_terminal_parent_module();
        let config = Config {
            init: Some("Init".to_string()),
            next: Some("Next".to_string()),
            check_deadlock: true,
            terminal: Some(TerminalSpec::Operator("IsTerminal".to_string())),
            ..Default::default()
        };
        let mut checker = ModelChecker::new(&module, &config);
        checker.flat_state_primary = true;
        checker.compiled_bfs_level = Some(Box::new(ZeroGeneratedDeadlockMetadataFusedLevel {
            first_zero_generated_parent_idx: 0,
        }));
        checker.compiled_bfs_step = None;

        let (mut flat_queue, mut storage) = seed_two_terminal_parent_flat_frontier(&mut checker);

        let result = checker.run_compiled_bfs_loop(&mut storage, &mut flat_queue);

        match result {
            CheckResult::Success(_) => {}
            CheckResult::Deadlock { .. } => panic!(
                "terminal states misreported as deadlock: compiled BFS deadlock reporting \
                 does not evaluate TerminalSpec and must not run for TERMINAL configs"
            ),
            other => panic!("expected Success for all-terminal frontier, got {other:?}"),
        }
    }

    #[test]
    fn compiled_bfs_fused_invariant_failure_rejects_bad_parent_prefix() {
        let module = two_parent_liveness_module();
        let config = Config {
            init: Some("Init".to_string()),
            next: Some("Next".to_string()),
            invariants: vec!["Inv".to_string()],
            check_deadlock: false,
            ..Default::default()
        };
        let mut checker = ModelChecker::new(&module, &config);
        checker.flat_state_primary = true;
        checker.compiled_bfs_level = Some(Box::new(MalformedInvariantFusedLevel));
        checker.compiled_bfs_step = None;

        let (mut flat_queue, mut storage, _fp0, _fp1) = seed_two_parent_flat_frontier(&mut checker);

        let result = checker.run_compiled_bfs_loop(&mut storage, &mut flat_queue);

        match result {
            CheckResult::Error { error, .. } => {
                let message = error.to_string();
                assert!(
                    message.contains("invariant metadata reported parent 1 with 1 parents"),
                    "bad invariant parent prefix should fail closed, got: {message}"
                );
            }
            other => panic!("expected malformed invariant metadata error, got {other:?}"),
        }
        assert_eq!(flat_queue.len(), 2);
    }

    #[test]
    fn compiled_bfs_fused_invariant_failure_reconstructs_parent_trace() {
        let module = two_parent_liveness_module();
        let config = Config {
            init: Some("Init".to_string()),
            next: Some("Next".to_string()),
            invariants: vec!["Inv".to_string()],
            check_deadlock: false,
            ..Default::default()
        };
        let mut checker = ModelChecker::new(&module, &config);
        checker.flat_state_primary = true;
        checker.set_trace_file(TraceFile::create_temp().expect("trace file should initialize"));
        checker.trace.cached_init_name = Some("Init".to_string());
        checker.compiled_bfs_level = Some(Box::new(InvariantFailureTraceFusedLevel));
        checker.compiled_bfs_step = None;

        let (mut flat_queue, mut storage, _fp0, _fp1) = seed_two_parent_flat_frontier(&mut checker);

        let result = checker.run_compiled_bfs_loop(&mut storage, &mut flat_queue);

        match result {
            CheckResult::InvariantViolation {
                invariant, trace, ..
            } => {
                assert_eq!(invariant, "Inv");
                assert_eq!(trace.len(), 2);
                assert_eq!(trace.states[0].get("x"), Some(&crate::Value::int(1)));
                assert_eq!(trace.states[1].get("x"), Some(&crate::Value::int(2)));
            }
            other => panic!("expected fused invariant violation trace, got {other:?}"),
        }
    }

    #[test]
    fn compiled_bfs_per_parent_invariant_failure_reconstructs_parent_trace() {
        let module = two_parent_liveness_module();
        let config = Config {
            init: Some("Init".to_string()),
            next: Some("Next".to_string()),
            invariants: vec!["Inv".to_string()],
            check_deadlock: false,
            ..Default::default()
        };
        let mut checker = ModelChecker::new(&module, &config);
        checker.flat_state_primary = true;
        checker.set_trace_file(TraceFile::create_temp().expect("trace file should initialize"));
        checker.trace.cached_init_name = Some("Init".to_string());
        checker.compiled_bfs_level = None;
        checker.compiled_bfs_step = Some(Box::new(InvariantFailureTraceCompiledStep));

        let (mut flat_queue, mut storage, _fp0, _fp1) = seed_two_parent_flat_frontier(&mut checker);

        let result = checker.run_compiled_bfs_loop(&mut storage, &mut flat_queue);

        match result {
            CheckResult::InvariantViolation {
                invariant, trace, ..
            } => {
                assert_eq!(invariant, "Inv");
                assert_eq!(trace.len(), 2);
                assert_eq!(trace.states[0].get("x"), Some(&crate::Value::int(1)));
                assert_eq!(trace.states[1].get("x"), Some(&crate::Value::int(2)));
            }
            other => panic!("expected per-parent invariant violation trace, got {other:?}"),
        }
    }

    #[test]
    fn compiled_bfs_fused_non_batch_admission_rejects_missing_parent_metadata() {
        let module = parse_module(
            r#"
	---- MODULE CompiledBfsBatchAdmissionMissingParentTest ----
VARIABLE x
Init == x = 0
Next == x' = x + 1
====
"#,
        );
        let config = Config {
            init: Some("Init".to_string()),
            next: Some("Next".to_string()),
            check_deadlock: false,
            ..Default::default()
        };
        let mut checker = ModelChecker::new(&module, &config);
        checker.flat_state_primary = true;
        checker.compiled_bfs_level =
            Some(Box::new(BatchAdmissionMissingParentMetadataFusedLevel {
                calls: AtomicUsize::new(0),
            }));
        checker.compiled_bfs_step = None;

        let layout = Arc::new(StateLayout::new(
            checker.ctx.var_registry(),
            vec![VarLayoutKind::Scalar],
        ));
        checker.flat_bfs_adapter = Some(crate::state::FlatBfsAdapter::from_layout(layout.clone()));
        let mut flat_queue = FlatBfsFrontier::new(layout);
        flat_queue.push_raw_buffer(&[0], Fingerprint(1), 0, 0);
        let mut storage = FingerprintOnlyStorage::new(
            BulkStateStorage::empty(checker.ctx.var_registry().len()),
            1,
        );

        let result = checker.run_compiled_bfs_loop(&mut storage, &mut flat_queue);

        match result {
            CheckResult::Error { error, .. } => {
                let message = error.to_string();
                assert!(
                    message.contains("without successor parent metadata"),
                    "missing parent metadata should fail closed, got: {message}"
                );
            }
            other => panic!("expected missing-parent-metadata error, got {other:?}"),
        }
        assert_eq!(checker.test_seen_fps_len(), 0);
        assert_eq!(flat_queue.len(), 0);
        assert_eq!(flat_queue.flat_pushed(), 1);
    }

    #[test]
    fn compiled_bfs_fused_non_batch_admission_rejects_parent_index_outside_current_level_after_append(
    ) {
        let module = parse_module(
            r#"
	---- MODULE CompiledBfsNonBatchOutOfRangeParentTest ----
	VARIABLE x
	Init == x = 0
	Next == x' = x + 1
	Inv == TRUE
	====
	"#,
        );
        let config = Config {
            init: Some("Init".to_string()),
            next: Some("Next".to_string()),
            invariants: vec!["Inv".to_string()],
            check_deadlock: false,
            ..Default::default()
        };
        let mut checker = ModelChecker::new(&module, &config);
        checker.flat_state_primary = true;
        checker.compiled_bfs_level = Some(Box::new(NonBatchOutOfRangeParentAfterAppendFusedLevel));
        checker.compiled_bfs_step = None;

        let layout = Arc::new(StateLayout::new(
            checker.ctx.var_registry(),
            vec![VarLayoutKind::Scalar],
        ));
        checker.flat_bfs_adapter = Some(crate::state::FlatBfsAdapter::from_layout(layout.clone()));
        let mut flat_queue = FlatBfsFrontier::new(layout);
        flat_queue.push_raw_buffer(&[0], Fingerprint(1), 0, 0);
        let mut storage = FingerprintOnlyStorage::new(
            BulkStateStorage::empty(checker.ctx.var_registry().len()),
            1,
        );

        let result = checker.run_compiled_bfs_loop(&mut storage, &mut flat_queue);

        match result {
            CheckResult::Error { error, .. } => {
                let message = error.to_string();
                assert!(
                    message.contains("parent index 1 for 1 parents"),
                    "out-of-range parent metadata should fail closed, got: {message}"
                );
            }
            other => panic!("expected out-of-range parent metadata error, got {other:?}"),
        }
        assert_eq!(checker.test_seen_fps_len(), 1);
        assert_eq!(flat_queue.len(), 1);
        assert_eq!(flat_queue.flat_pushed(), 2);
    }

    #[test]
    fn compiled_bfs_state_constrained_fused_buffer_overflow_returns_error_without_per_parent_fallback(
    ) {
        let module = parse_module(
            r#"
	---- MODULE CompiledBfsStateConstrainedOverflowTest ----
	VARIABLE x
	Init == x = 0
	Next == x' = x + 1
	Constraint == x <= 0
	====
	"#,
        );
        let config = Config {
            init: Some("Init".to_string()),
            next: Some("Next".to_string()),
            constraints: vec!["Constraint".to_string()],
            check_deadlock: false,
            ..Default::default()
        };
        let step_calls = Arc::new(AtomicUsize::new(0));
        let mut checker = ModelChecker::new(&module, &config);
        checker.flat_state_primary = true;
        checker.compiled_bfs_level = Some(Box::new(BufferOverflowFusedLevel));
        checker.compiled_bfs_step = Some(Box::new(CountingCompiledStep {
            calls: step_calls.clone(),
        }));

        let layout = Arc::new(StateLayout::new(
            checker.ctx.var_registry(),
            vec![VarLayoutKind::Scalar],
        ));
        checker.flat_bfs_adapter = Some(crate::state::FlatBfsAdapter::from_layout(layout.clone()));
        let mut flat_queue = FlatBfsFrontier::new(layout);
        flat_queue.push_raw_buffer(&[0], Fingerprint(1), 0, 0);
        let mut storage = FingerprintOnlyStorage::new(
            BulkStateStorage::empty(checker.ctx.var_registry().len()),
            1,
        );

        let result = checker.run_compiled_bfs_loop(&mut storage, &mut flat_queue);

        match result {
            CheckResult::Error { error, .. } => {
                let message = error.to_string();
                assert!(
                    message.contains("state-constrained compiled BFS fused level failed closed"),
                    "state-constrained fused overflow should fail closed, got: {message}"
                );
            }
            other => panic!("expected state-constrained overflow error, got {other:?}"),
        }
        assert_eq!(
            step_calls.load(Ordering::SeqCst),
            0,
            "state-constrained fused overflow must not retry through per-parent compiled step"
        );
    }

    #[test]
    fn compiled_bfs_state_constrained_missing_raw_metadata_fails_closed_without_fallback() {
        let module = parse_module(
            r#"
---- MODULE CompiledBfsStateConstrainedMetadataTest ----
VARIABLE x
Init == x = 0
Next == x' = x + 1
Constraint == x <= 1
====
"#,
        );
        let config = Config {
            init: Some("Init".to_string()),
            next: Some("Next".to_string()),
            constraints: vec!["Constraint".to_string()],
            check_deadlock: true,
            ..Default::default()
        };
        let step_calls = Arc::new(AtomicUsize::new(0));
        let mut checker = ModelChecker::new(&module, &config);
        checker.flat_state_primary = true;
        checker.compiled_bfs_level =
            Some(Box::new(BatchAdmissionMissingParentMetadataFusedLevel {
                calls: AtomicUsize::new(0),
            }));
        checker.compiled_bfs_step = Some(Box::new(CountingCompiledStep {
            calls: step_calls.clone(),
        }));

        let layout = Arc::new(StateLayout::new(
            checker.ctx.var_registry(),
            vec![VarLayoutKind::Scalar],
        ));
        checker.flat_bfs_adapter = Some(crate::state::FlatBfsAdapter::from_layout(layout.clone()));
        let mut flat_queue = FlatBfsFrontier::new(layout);
        flat_queue.push_raw_buffer(&[0], Fingerprint(1), 0, 0);
        let mut storage = FingerprintOnlyStorage::new(
            BulkStateStorage::empty(checker.ctx.var_registry().len()),
            1,
        );

        let result = checker.run_compiled_bfs_loop(&mut storage, &mut flat_queue);

        match result {
            CheckResult::Error { error, .. } => {
                let message = error.to_string();
                assert!(
                    message.contains("state-constrained compiled BFS fused level failed closed")
                        && message.contains("raw-successor metadata"),
                    "state-constrained metadata gap should fail closed, got: {message}"
                );
            }
            other => panic!("expected state-constrained metadata error, got {other:?}"),
        }
        assert_eq!(
            step_calls.load(Ordering::SeqCst),
            0,
            "state-constrained metadata gap must not retry through per-parent compiled step"
        );
    }

    #[test]
    fn compiled_bfs_per_parent_step_records_successor_parent_trace_loc() {
        let module = two_parent_liveness_module();
        let config = Config {
            init: Some("Init".to_string()),
            next: Some("Next".to_string()),
            check_deadlock: false,
            ..Default::default()
        };
        let mut checker = ModelChecker::new(&module, &config);
        checker.flat_state_primary = true;
        checker.set_trace_file(TraceFile::create_temp().expect("trace file should initialize"));
        checker.compiled_bfs_level = None;
        checker.compiled_bfs_step = Some(Box::new(TraceProvenanceCompiledStep));

        checker.trace.cached_next_name = Some("Next".to_string());
        checker.trace.cached_resolved_next_name = Some("Next".to_string());
        let layout = Arc::new(StateLayout::new(
            checker.ctx.var_registry(),
            vec![VarLayoutKind::Scalar],
        ));
        checker.flat_bfs_adapter = Some(crate::state::FlatBfsAdapter::from_layout(layout.clone()));

        let fp0 = crate::check::model_checker::invariants::fingerprint_flat_compiled(&[0]);
        checker
            .mark_state_seen_fp_only_checked(fp0, None, 0)
            .expect("seed x=0");
        let loc0 = checker.trace.last_inserted_trace_loc;
        let fp1 = crate::check::model_checker::invariants::fingerprint_flat_compiled(&[1]);
        checker
            .mark_state_seen_fp_only_checked(fp1, None, 0)
            .expect("seed x=1");
        let loc1 = checker.trace.last_inserted_trace_loc;

        let mut flat_queue = FlatBfsFrontier::new(layout);
        flat_queue.push_raw_buffer(&[0], fp0, 0, loc0);
        flat_queue.push_raw_buffer(&[1], fp1, 0, loc1);
        let mut storage = FingerprintOnlyStorage::new(
            BulkStateStorage::empty(checker.ctx.var_registry().len()),
            checker.ctx.var_registry().len(),
        );

        let result = checker.run_compiled_bfs_loop(&mut storage, &mut flat_queue);

        assert!(matches!(result, CheckResult::Success(_)), "got {result:?}");
        let successor_10_fp =
            crate::check::model_checker::invariants::fingerprint_flat_compiled(&[10]);
        let successor_20_fp =
            crate::check::model_checker::invariants::fingerprint_flat_compiled(&[20]);
        let parents = checker
            .trace
            .trace_file
            .as_mut()
            .expect("trace file still installed")
            .build_parents_map()
            .expect("parents map should scan");
        assert_eq!(parents.get(&successor_10_fp), Some(&fp0));
        assert_eq!(parents.get(&successor_20_fp), Some(&fp1));
    }

    #[test]
    fn compiled_bfs_liveness_falls_back_when_fused_parent_metadata_missing() {
        let module = parse_module(
            r#"
---- MODULE CompiledBfsLivenessMissingParentTest ----
VARIABLE x
Init == x = 0
Next == x' = x + 1
====
"#,
        );
        let config = Config {
            init: Some("Init".to_string()),
            next: Some("Next".to_string()),
            check_deadlock: false,
            ..Default::default()
        };
        let step_calls = Arc::new(AtomicUsize::new(0));
        let mut checker = ModelChecker::new(&module, &config);
        checker.flat_state_primary = true;
        checker.liveness_cache.cache_for_liveness = true;
        checker.compiled_bfs_level =
            Some(Box::new(BatchAdmissionMissingParentMetadataFusedLevel {
                calls: AtomicUsize::new(0),
            }));
        checker.compiled_bfs_step = Some(Box::new(CountingCompiledStep {
            calls: step_calls.clone(),
        }));

        let layout = Arc::new(StateLayout::new(
            checker.ctx.var_registry(),
            vec![VarLayoutKind::Scalar],
        ));
        checker.flat_bfs_adapter = Some(crate::state::FlatBfsAdapter::from_layout(layout.clone()));
        let mut flat_queue = FlatBfsFrontier::new(layout);
        flat_queue.push_raw_buffer(&[0], Fingerprint(1), 0, 0);
        let mut storage = FingerprintOnlyStorage::new(
            BulkStateStorage::empty(checker.ctx.var_registry().len()),
            1,
        );

        let result = checker.run_compiled_bfs_loop(&mut storage, &mut flat_queue);

        match result {
            CheckResult::Success(stats) => {
                assert_eq!(stats.transitions, 1);
                assert_eq!(stats.max_depth, 1);
                assert_eq!(stats.states_found, 1);
            }
            other => panic!("expected liveness fallback success, got {other:?}"),
        }
        assert!(step_calls.load(Ordering::SeqCst) > 0);
        assert_eq!(checker.test_seen_fps_len(), 1);
    }

    #[test]
    fn compiled_bfs_liveness_falls_back_when_fused_state_graph_edges_incomplete() {
        let module = two_parent_liveness_module();
        let config = Config {
            init: Some("Init".to_string()),
            next: Some("Next".to_string()),
            check_deadlock: false,
            ..Default::default()
        };
        let fused_calls = Arc::new(AtomicUsize::new(0));
        let step_calls = Arc::new(AtomicUsize::new(0));
        let mut checker = ModelChecker::new(&module, &config);
        checker.flat_state_primary = true;
        checker.liveness_cache.cache_for_liveness = true;
        checker.compiled_bfs_level = Some(Box::new(UnsafeLocalDedupFusedLevel {
            calls: fused_calls.clone(),
        }));
        checker.compiled_bfs_step = Some(Box::new(CompleteTwoParentCompiledStep {
            calls: step_calls.clone(),
        }));

        let (mut flat_queue, mut storage, fp0, fp1) = seed_two_parent_flat_frontier(&mut checker);

        let result = checker.run_compiled_bfs_loop(&mut storage, &mut flat_queue);

        assert!(matches!(result, CheckResult::Success(_)), "got {result:?}");
        assert_eq!(fused_calls.load(Ordering::SeqCst), 1);
        assert!(step_calls.load(Ordering::SeqCst) > 0);
        assert!(checker.compiled_bfs_level.is_none());
        assert_both_parent_liveness_edges(&checker, fp0, fp1);
    }

    #[test]
    fn compiled_bfs_state_constrained_liveness_metadata_gap_fails_closed() {
        let module = parse_module(
            r#"
---- MODULE CompiledBfsStateConstrainedLivenessMetadataTest ----
VARIABLE x
Init == x = 0 \/ x = 1
Next == \/ /\ x = 0 /\ x' = 2
        \/ /\ x = 1 /\ x' = 2
Constraint == x <= 1
====
"#,
        );
        let config = Config {
            init: Some("Init".to_string()),
            next: Some("Next".to_string()),
            constraints: vec!["Constraint".to_string()],
            check_deadlock: false,
            ..Default::default()
        };
        let fused_calls = Arc::new(AtomicUsize::new(0));
        let step_calls = Arc::new(AtomicUsize::new(0));
        let mut checker = ModelChecker::new(&module, &config);
        checker.flat_state_primary = true;
        checker.liveness_cache.cache_for_liveness = true;
        checker.compiled_bfs_level = Some(Box::new(UnsafeLocalDedupFusedLevel {
            calls: fused_calls.clone(),
        }));
        checker.compiled_bfs_step = Some(Box::new(CompleteTwoParentCompiledStep {
            calls: step_calls.clone(),
        }));

        let (mut flat_queue, mut storage, _fp0, _fp1) = seed_two_parent_flat_frontier(&mut checker);

        let result = checker.run_compiled_bfs_loop(&mut storage, &mut flat_queue);

        match result {
            CheckResult::Error { error, .. } => {
                let message = error.to_string();
                assert!(
                    message.contains("state-constrained compiled BFS fused level failed closed")
                        && message.contains("state-graph successor")
                        && message.contains("liveness capture"),
                    "state-constrained liveness metadata gap should fail closed, got: {message}"
                );
            }
            other => panic!("expected state-constrained liveness metadata error, got {other:?}"),
        }
        assert_eq!(fused_calls.load(Ordering::SeqCst), 1);
        assert_eq!(
            step_calls.load(Ordering::SeqCst),
            0,
            "state-constrained liveness metadata gap must not retry through per-parent compiled step"
        );
        assert!(
            checker.compiled_bfs_level.is_some(),
            "fail-closed liveness metadata error must not clear the fused level for fallback"
        );
    }

    #[test]
    fn compiled_bfs_liveness_falls_back_when_per_parent_step_dedups_edges() {
        let module = two_parent_liveness_module();
        let config = Config {
            init: Some("Init".to_string()),
            next: Some("Next".to_string()),
            check_deadlock: false,
            ..Default::default()
        };
        let step_calls = Arc::new(AtomicUsize::new(0));
        let mut checker = ModelChecker::new(&module, &config);
        checker.flat_state_primary = true;
        checker.liveness_cache.cache_for_liveness = true;
        checker.compiled_bfs_level = None;
        checker.compiled_bfs_step = Some(Box::new(UnsafeDedupingCompiledStep {
            calls: step_calls.clone(),
        }));

        let (mut flat_queue, mut storage, fp0, fp1) = seed_two_parent_flat_frontier(&mut checker);

        let result = checker.run_compiled_bfs_loop(&mut storage, &mut flat_queue);

        assert!(matches!(result, CheckResult::Success(_)), "got {result:?}");
        assert_eq!(step_calls.load(Ordering::SeqCst), 0);
        assert_both_parent_liveness_edges(&checker, fp0, fp1);
    }

    #[test]
    fn compiled_bfs_liveness_clears_unsafe_step_when_prerequisite_falls_back() {
        let module = two_parent_liveness_module();
        let config = Config {
            init: Some("Init".to_string()),
            next: Some("Next".to_string()),
            check_deadlock: false,
            ..Default::default()
        };
        let step_calls = Arc::new(AtomicUsize::new(0));
        let mut checker = ModelChecker::new(&module, &config);
        checker.flat_state_primary = false;
        checker.liveness_cache.cache_for_liveness = true;
        checker.compiled_bfs_level = None;
        checker.compiled_bfs_step = Some(Box::new(UnsafeDedupingCompiledStep {
            calls: step_calls.clone(),
        }));

        let (mut flat_queue, mut storage, fp0, fp1) = seed_two_parent_flat_frontier(&mut checker);

        let result = checker.run_compiled_bfs_loop(&mut storage, &mut flat_queue);

        assert!(matches!(result, CheckResult::Success(_)), "got {result:?}");
        assert_eq!(step_calls.load(Ordering::SeqCst), 0);
        assert!(checker.compiled_bfs_step.is_none());
        assert!(checker.compiled_bfs_level.is_none());
        assert_both_parent_liveness_edges(&checker, fp0, fp1);
    }

    #[test]
    fn fused_successor_pre_seen_lookup_needed_for_rust_invariant_filtering() {
        assert!(fused_successor_needs_pre_seen_lookup(false, 2, false));
    }

    #[test]
    fn fused_successor_pre_seen_lookup_needed_without_level_skip_proof() {
        assert!(fused_successor_needs_pre_seen_lookup(true, 2, false));
    }

    #[test]
    fn fused_successor_pre_seen_lookup_skipped_after_backend_invariant_checking_with_proof() {
        assert!(!fused_successor_needs_pre_seen_lookup(true, 2, true));
    }

    #[test]
    fn fused_successor_pre_seen_lookup_skipped_when_no_regular_invariants_exist() {
        assert!(!fused_successor_needs_pre_seen_lookup(false, 0, false));
    }

    #[test]
    fn fused_level_pre_seen_skip_accepts_flat_primary_or_native_fused_admission_proof() {
        assert!(fused_level_may_skip_global_pre_seen_lookup(
            true, true, false
        ));
        assert!(fused_level_may_skip_global_pre_seen_lookup(
            true, false, true
        ));
        assert!(!fused_level_may_skip_global_pre_seen_lookup(
            true, false, false
        ));
        assert!(!fused_level_may_skip_global_pre_seen_lookup(
            false, true, true
        ));
    }

    #[test]
    fn fused_successor_backend_fingerprint_sidecars_trust_flat_or_native_fused_admission_proof() {
        assert!(fused_successor_trusts_backend_fingerprint_sidecars(
            true, false
        ));
        assert!(fused_successor_trusts_backend_fingerprint_sidecars(
            false, true
        ));
        assert!(!fused_successor_trusts_backend_fingerprint_sidecars(
            false, false
        ));
    }

    #[test]
    fn fused_successor_rust_regular_invariant_check_skipped_when_none_configured() {
        assert!(!fused_successor_needs_rust_regular_invariant_check(
            false, 0,
        ));
    }

    #[test]
    fn fused_successor_rust_regular_invariant_check_skipped_when_backend_checked() {
        assert!(!fused_successor_needs_rust_regular_invariant_check(true, 2));
    }

    #[test]
    fn fused_successor_rust_regular_invariant_check_needed_for_action_only_backend() {
        assert!(fused_successor_needs_rust_regular_invariant_check(false, 2,));
    }

    #[test]
    fn compiled_bfs_fused_duplicate_only_level_counts_transitions_and_depth() {
        let module = parse_module(
            r#"
---- MODULE CompiledBfsDuplicateOnlyStatsTest ----
VARIABLE x
Init == x = 0
Next == x' = x
====
"#,
        );
        let config = Config {
            init: Some("Init".to_string()),
            next: Some("Next".to_string()),
            check_deadlock: false,
            ..Default::default()
        };
        let mut checker = ModelChecker::new(&module, &config);
        checker.flat_state_primary = true;
        checker.compiled_bfs_level = Some(Box::new(DuplicateOnlyFusedLevel));
        checker.compiled_bfs_step = None;

        let layout = Arc::new(StateLayout::new(
            checker.ctx.var_registry(),
            vec![VarLayoutKind::Scalar],
        ));
        checker.flat_bfs_adapter = Some(crate::state::FlatBfsAdapter::from_layout(layout.clone()));
        let mut flat_queue = FlatBfsFrontier::new(layout);
        flat_queue.push_raw_buffer(&[0], Fingerprint(1), 0, 0);
        let mut storage = FingerprintOnlyStorage::new(
            BulkStateStorage::empty(checker.ctx.var_registry().len()),
            1,
        );

        let result = checker.run_compiled_bfs_loop(&mut storage, &mut flat_queue);

        // This synthetic duplicate-only level yields 0 distinct states with a
        // declared Init/Next, so the V1 vacuity gate now reports
        // `Vacuous(EmptyReachableSet)`. Both variants carry the same CheckStats;
        // this test only asserts the transition/depth counting.
        match result {
            CheckResult::Success(stats) | CheckResult::Vacuous { stats, .. } => {
                assert_eq!(stats.transitions, 1);
                assert_eq!(stats.max_depth, 1);
            }
            other => panic!("expected duplicate-only fused level success/vacuous, got {other:?}"),
        }
    }

    #[test]
    fn compiled_bfs_loop_returns_fatal_error_without_fallback() {
        let module = parse_module(
            r#"
---- MODULE CompiledBfsFatalRuntimeErrorTest ----
VARIABLE x
Init == x = 0
Next == x' = x
====
"#,
        );
        let config = Config {
            init: Some("Init".to_string()),
            next: Some("Next".to_string()),
            ..Default::default()
        };
        let mut checker = ModelChecker::new(&module, &config);
        checker.flat_state_primary = true;
        checker.compiled_bfs_level = Some(Box::new(FatalFusedLevel));
        checker.compiled_bfs_step = None;

        let layout = Arc::new(StateLayout::new(
            checker.ctx.var_registry(),
            vec![VarLayoutKind::Scalar],
        ));
        let mut flat_queue = FlatBfsFrontier::new(layout);
        flat_queue.push_raw_buffer(&[0], Fingerprint(1), 0, 0);
        let mut storage = FingerprintOnlyStorage::new(
            BulkStateStorage::empty(checker.ctx.var_registry().len()),
            1,
        );

        let result = checker.run_compiled_bfs_loop(&mut storage, &mut flat_queue);

        match result {
            CheckResult::Error { error, .. } => {
                let message = error.to_string();
                assert!(
                    message.contains("compiled BFS fatal error"),
                    "fatal compiled BFS error should be returned directly, got: {message}"
                );
            }
            other => panic!("expected fatal compiled BFS error, got {other:?}"),
        }
        assert!(
            checker.compiled_bfs_level.is_some(),
            "fatal fused-level errors must not clear the level and fall back"
        );
        assert!(
            !checker.jit_monolithic_disabled,
            "fatal fused-level errors must not disable compiled BFS and retry through fallback"
        );
    }

    #[test]
    fn compiled_bfs_loop_returns_preflight_fatal_error_without_fallback_or_level_call() {
        let module = parse_module(
            r#"
---- MODULE CompiledBfsPreflightFatalRuntimeErrorTest ----
VARIABLE x
Init == x = 0
Next == x' = x
====
"#,
        );
        let config = Config {
            init: Some("Init".to_string()),
            next: Some("Next".to_string()),
            ..Default::default()
        };
        let preflight_seen = Arc::new(AtomicBool::new(false));
        let run_called = Arc::new(AtomicBool::new(false));
        let mut checker = ModelChecker::new(&module, &config);
        checker.flat_state_primary = true;
        checker.compiled_bfs_level = Some(Box::new(PreflightFatalFusedLevel {
            preflight_seen: preflight_seen.clone(),
            run_called: run_called.clone(),
        }));
        checker.compiled_bfs_step = None;

        let layout = Arc::new(StateLayout::new(
            checker.ctx.var_registry(),
            vec![VarLayoutKind::Scalar],
        ));
        let mut flat_queue = FlatBfsFrontier::new(layout);
        flat_queue.push_raw_buffer(&[0], Fingerprint(1), 0, 0);
        let mut storage = FingerprintOnlyStorage::new(
            BulkStateStorage::empty(checker.ctx.var_registry().len()),
            1,
        );

        let result = checker.run_compiled_bfs_loop(&mut storage, &mut flat_queue);

        assert!(
            preflight_seen.load(Ordering::SeqCst),
            "fused level preflight should run before the native level call"
        );
        assert!(
            !run_called.load(Ordering::SeqCst),
            "fatal preflight must stop before run_level_fused_arena"
        );
        match result {
            CheckResult::Error { error, .. } => {
                let message = error.to_string();
                assert!(
                    message.contains("compiled BFS fatal error"),
                    "fatal compiled BFS preflight error should be returned directly, got: {message}"
                );
            }
            other => panic!("expected fatal compiled BFS preflight error, got {other:?}"),
        }
        assert!(
            checker.compiled_bfs_level.is_some(),
            "fatal fused-level preflight errors must not clear the level and fall back"
        );
        assert!(
            !checker.jit_monolithic_disabled,
            "fatal fused-level preflight errors must not disable compiled BFS"
        );
    }
}
