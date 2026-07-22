// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Fingerprint-only BFS storage backend.
//!
//! Part of #3436: extracted from `storage_modes.rs`.

use super::super::super::fingerprint::BfsFingerprintDomain;
use super::super::super::frontier::BfsFrontier;
use super::super::super::mem_census::mem_census_enabled;
use super::super::super::{
    check_error_to_result, ArrayState, BulkStateHandle, BulkStateStorage, CheckResult, Fingerprint,
    ModelChecker, State, VecDeque,
};
use super::super::checkpoint_view;
use super::{BfsAdmissionCandidate, BfsAdmittedEntry, BfsBatchAdmissionResult, BfsStorage};
use crate::state::{fp_hashmap, FlatState, FpHashMap};
use crate::storage::{BatchInsertedIndexAdmission, FingerprintPayloadWitnesses};
use crate::EvalCheckError;
use crate::TraceLocationStorage;
use crate::Value;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PayloadWitnessDomain {
    ArrayState,
    FlatI64,
    View,
}

fn array_witness_frontier_enabled() -> bool {
    !std::env::var("TY_NO_ARRAY_WITNESS_FRONTIER").is_ok_and(|value| {
        let value = value.trim();
        value == "1" || value.eq_ignore_ascii_case("true")
    })
}

fn diff_witness_prefilter_enabled() -> bool {
    !std::env::var("TY_NO_DIFF_WITNESS_PREFILTER").is_ok_and(|value| {
        let value = value.trim();
        value == "1" || value.eq_ignore_ascii_case("true")
    })
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct DiffWitnessPrefilterTestTelemetry {
    checks: u64,
    confirmed: u64,
    fallbacks: u64,
}

#[cfg(test)]
std::thread_local! {
    /// Test-only view of the most recent fingerprint-only run on this thread.
    ///
    /// `FingerprintOnlyStorage` is owned inside `ModelChecker::check()`, so an
    /// end-to-end test cannot otherwise prove that the virtual-payload fast path
    /// ran. Thread-local storage keeps the probe isolated from concurrently
    /// executing model-checker tests and is compiled out of production builds.
    static DIFF_WITNESS_PREFILTER_TEST_TELEMETRY:
        std::cell::Cell<DiffWitnessPrefilterTestTelemetry> =
            const { std::cell::Cell::new(DiffWitnessPrefilterTestTelemetry {
                checks: 0,
                confirmed: 0,
                fallbacks: 0,
            }) };
}

#[cfg(test)]
fn reset_diff_witness_prefilter_test_telemetry() {
    DIFF_WITNESS_PREFILTER_TEST_TELEMETRY.with(|telemetry| {
        telemetry.set(DiffWitnessPrefilterTestTelemetry::default());
    });
}

#[cfg(test)]
fn diff_witness_prefilter_test_telemetry() -> DiffWitnessPrefilterTestTelemetry {
    DIFF_WITNESS_PREFILTER_TEST_TELEMETRY.with(std::cell::Cell::get)
}

#[cfg(test)]
fn record_diff_witness_prefilter_test_telemetry(checks: u64, confirmed: u64, fallbacks: u64) {
    DIFF_WITNESS_PREFILTER_TEST_TELEMETRY.with(|telemetry| {
        telemetry.set(DiffWitnessPrefilterTestTelemetry {
            checks,
            confirmed,
            fallbacks,
        });
    });
}

/// Queue entry type for no-trace (fingerprint-only) BFS mode.
///
/// Bulk entries reference initial states stored in contiguous `BulkStateStorage`;
/// owned entries hold successor states materialized during BFS exploration.
/// Flat entries hold states as contiguous `[i64]` buffers, auto-detected for
/// specs where all state variables are scalar, or force-enabled via
/// `Config::use_flat_state = Some(true)` (Part of #4126: Tier 0 interpreter
/// sandwich).
#[derive(Clone)]
pub(in crate::check::model_checker) enum NoTraceQueueEntry {
    Bulk(BulkStateHandle),
    Owned {
        state: ArrayState,
        fp: Fingerprint,
    },
    /// Exact handle into `FingerprintPayloadWitnesses` for non-flat ArrayFp64.
    ///
    /// The witness already retains the admitted payload needed for collision
    /// authorization, so retaining a second deep ArrayState in the BFS frontier
    /// is redundant. `combined_xor` restores the incremental fingerprint base
    /// when the state becomes a parent.
    Witness {
        fp: Fingerprint,
        combined_xor: u64,
    },
    /// Flat i64 buffer representation for the interpreter sandwich.
    ///
    /// Auto-activated when the state layout is fully flat (all vars scalar)
    /// and roundtrip verification passes. Can also be force-enabled via
    /// `Config::use_flat_state = Some(true)`.
    ///
    /// On dequeue, the `FlatState` is converted to `ArrayState` via the
    /// `FlatBfsAdapter` for interpreter evaluation. On successor admission,
    /// `ArrayState` successors are converted to `FlatState` before enqueue.
    ///
    /// Part of #4126: Tier 0 interpreter sandwich.
    Flat {
        flat: FlatState,
        fp: Fingerprint,
    },
}

/// Fingerprint-only BFS storage: states are not kept in a HashMap.
///
/// Queue entries carry the actual `ArrayState` (for successors) or a handle
/// to `BulkStateStorage` (for initial states). Traces are reconstructed from
/// a disk-based trace file when needed.
pub(in crate::check::model_checker) struct FingerprintOnlyStorage {
    pub(in crate::check::model_checker) bulk_initial: BulkStateStorage,
    num_vars: usize,
    payload_witnesses: FingerprintPayloadWitnesses,
    view_payload_witnesses: FpHashMap<Value>,
    batch_fingerprints: Vec<Fingerprint>,
    batch_admission: BatchInsertedIndexAdmission,
    diff_witness_prefilter_enabled: bool,
    use_array_witness_frontier: bool,
    diff_witness_prefilter_checks: u64,
    diff_witness_prefilter_confirmed: u64,
    diff_witness_prefilter_fallbacks: u64,
}

impl FingerprintOnlyStorage {
    pub(in crate::check::model_checker) fn new(
        bulk_initial: BulkStateStorage,
        num_vars: usize,
    ) -> Self {
        Self {
            bulk_initial,
            num_vars,
            payload_witnesses: FingerprintPayloadWitnesses::new(),
            view_payload_witnesses: fp_hashmap(),
            batch_fingerprints: Vec::new(),
            batch_admission: BatchInsertedIndexAdmission::default(),
            diff_witness_prefilter_enabled: diff_witness_prefilter_enabled(),
            use_array_witness_frontier: false,
            diff_witness_prefilter_checks: 0,
            diff_witness_prefilter_confirmed: 0,
            diff_witness_prefilter_fallbacks: 0,
        }
    }

    /// Configure witness-backed queue handles once the production frontier
    /// representation has been selected.
    ///
    /// Keep the lane deliberately narrow: only safety-only ArrayFp64 runs have
    /// an exact ArrayState witness that can reconstruct the explored state, and
    /// flat runs must retain Owned entries long enough for FlatBfsFrontier to
    /// write them into its arena.
    pub(in crate::check::model_checker) fn configure_array_witness_frontier(
        &mut self,
        mc: &ModelChecker,
        flat_frontier_active: bool,
    ) -> bool {
        self.use_array_witness_frontier = array_witness_frontier_enabled()
            && !flat_frontier_active
            && Self::compact_payload_witness_domain(mc) == Some(PayloadWitnessDomain::ArrayState);
        self.use_array_witness_frontier
    }

    fn compact_payload_witness_domain(mc: &ModelChecker) -> Option<PayloadWitnessDomain> {
        // Keep the production slice narrow: liveness and flat-fp64 paths use
        // domains where ArrayState payload equality is not the canonical
        // duplicate witness.
        if mc.liveness_cache.cache_for_liveness {
            // Liveness normally forces full ArrayState retention in `seen` (the
            // full-retention `mark_state_seen_checked_with_current` branch),
            // because most liveness layouts need the payload as the duplicate
            // witness. For fully-flat, roundtrip-verified, no-VIEW/no-SYMMETRY
            // specs (`flat_state_primary`) the raw flat i64 buffer is itself a
            // lossless, fail-closed duplicate witness (identical to safety
            // mode). And the fp-only liveness verdict path never reads `seen`
            // payloads — it rebuilds full States from `init_states` + the
            // successor graph via Next-relation replay (see
            // `build_fp_only_liveness_state_cache`); seen-payload readers run
            // only under SYMMETRY, which is force-disabled for temporal
            // properties. So retaining the deep compound `Box<Value>` payload
            // per reachable state is pure waste here (it dominated liveness peak
            // RSS, e.g. nbacg_guer01 — `seen` held ~156 MB of compound
            // sequence-valued ArrayStates at peak).
            //
            // This covers BOTH flat dedup domains a `flat_state_primary` spec
            // can land on:
            //   * `CompiledFlat`: dedup is already on the raw flat buffer, so
            //     the flat witness is the canonical key.
            //   * `ArrayFp64`: dedup is on a content FP64 over the ArrayState
            //     (e.g. specs that run the Tier-0 interpreter because implied
            //     actions require interpreter eval, such as nbacg_guer01). The
            //     flat buffer is still a *complete* witness here because
            //     `flat_state_primary` guarantees `fully_flat` +
            //     roundtrip-verified losslessness, so two distinct states cannot
            //     share a flat buffer. The general-ArrayFp64 warning below (about
            //     variable-length `Dom -> Seq` layouts merging distinct states)
            //     does NOT apply: those layouts are not `fully_flat`, hence not
            //     `flat_state_primary`. And `duplicate_flat_payload_confirmed`
            //     fail-closes — if `try_from_array_state_lossless` ever returns
            //     `None` for a state it reports "not a confirmed duplicate", so
            //     a collision is kept (overcount, sound), never merged.
            //
            // Fail closed: any spec that is not provably flat-primary keeps the
            // full-retention path unchanged.
            if mc.flat_state_primary
                && matches!(
                    mc.bfs_fingerprint_domain(),
                    BfsFingerprintDomain::CompiledFlat | BfsFingerprintDomain::ArrayFp64
                )
            {
                return Some(PayloadWitnessDomain::FlatI64);
            }
            return None;
        }

        match mc.bfs_fingerprint_domain() {
            // For a `flat_state_primary` spec the raw flat i64 buffer is itself a
            // complete, collision-free duplicate witness — `fully_flat` +
            // roundtrip-verified losslessness guarantee two distinct states
            // cannot share a flat buffer (the same soundness argument the
            // liveness branch above relies on at lines 112-128). Use it here so a
            // safety-only run does NOT retain a full deep compound `Box<Value>`
            // ArrayState per distinct state. For large sequence/function-valued
            // state that deep copy dominated peak RSS: lamport_mutex
            // (`network = [Proc -> [Proc -> Seq]]`) cost ~22.9 KB/state ×
            // 724,274 states ≈ 16 GB, vs ~40 B/state for the flat witness.
            // `flat_payload_slots` fail-closes: if `try_from_array_state_lossless`
            // returns None the state is kept as a non-duplicate (sound overcount),
            // never merged. (Previously this cheap witness was gated inside the
            // `cache_for_liveness` branch only, so safety-mode flat-primary specs
            // wrongly paid the ArrayState witness.)
            BfsFingerprintDomain::ArrayFp64 if mc.flat_state_primary => {
                Some(PayloadWitnessDomain::FlatI64)
            }
            // Non-flat-primary ArrayFp64 dedups on a content-based FP64
            // fingerprint, which can alias distinct states (e.g. variable-length
            // `Dom -> Seq` layouts that are not `fully_flat`). The ArrayState
            // payload witness is the collision-disambiguation mechanism that
            // makes this dedup sound and must stay active. It is keyed off the
            // dedup *domain* — never off flat-frontier storage, which is only a
            // storage optimization and does not change the dedup key.
            BfsFingerprintDomain::ArrayFp64 => Some(PayloadWitnessDomain::ArrayState),
            // CompiledFlat dedups on the raw flat i64 buffer, so the flat buffer
            // itself is the canonical witness.
            BfsFingerprintDomain::CompiledFlat => Some(PayloadWitnessDomain::FlatI64),
            // WP-11 slice 2: the duplicate witness is the CANONICAL flat i64
            // buffer (`flat_payload_slots` routes through the flat-symmetry
            // lexmin when this domain is active). Two states witness-confirm
            // IFF their canonical buffers are byte-equal — exactly the orbit
            // relation the fingerprint domain dedups on — so symmetric
            // duplicates confirm and genuine hash collisions stay separate.
            BfsFingerprintDomain::FlatSymmetryCanonical => Some(PayloadWitnessDomain::FlatI64),
            BfsFingerprintDomain::View => Some(PayloadWitnessDomain::View),
            BfsFingerprintDomain::SymmetryCanonical | BfsFingerprintDomain::FullStateFp64 => None,
        }
    }

    fn flat_payload_slots(candidate: &ArrayState, mc: &ModelChecker) -> Option<Box<[i64]>> {
        // WP-11 slice 2: under the flat-symmetry canonical domain the witness
        // is the CANONICAL buffer; `flat_symmetry_canonical_slots` fails closed
        // (None → caller keeps the state as a non-duplicate, a sound
        // overcount) and is inert (None) for every other domain.
        if mc.flat_symmetry_fingerprint_active() {
            return mc.flat_symmetry_canonical_slots(candidate);
        }
        let layout = mc
            .flat_bfs_adapter
            .as_ref()
            .map(|adapter| adapter.layout().clone())
            .or_else(|| mc.flat_state_layout().cloned())?;
        let flat = FlatState::try_from_array_state_lossless(candidate, layout)?;
        Some(flat.into_buffer())
    }

    fn current_array_payload_confirms(
        fp: Fingerprint,
        candidate: &ArrayState,
        current: Option<(Fingerprint, &ArrayState)>,
    ) -> bool {
        current
            .filter(|(current_fp, _)| *current_fp == fp)
            .is_some_and(|(_, current_state)| current_state.values() == candidate.values())
    }

    fn current_flat_payload_confirms(
        fp: Fingerprint,
        candidate_slots: &[i64],
        current: Option<(Fingerprint, &ArrayState)>,
        mc: &ModelChecker,
    ) -> bool {
        current
            .filter(|(current_fp, _)| *current_fp == fp)
            .and_then(|(_, current_state)| Self::flat_payload_slots(current_state, mc))
            .is_some_and(|current_slots| current_slots.as_ref() == candidate_slots)
    }

    fn duplicate_array_payload_confirmed(
        &mut self,
        fp: Fingerprint,
        candidate: &ArrayState,
        current: Option<(Fingerprint, &ArrayState)>,
        mc: &mut ModelChecker,
    ) -> bool {
        if let Some(confirmed) = self.payload_witnesses.confirm_array_state(fp, candidate) {
            return confirmed;
        }

        if Self::current_array_payload_confirms(fp, candidate, current) {
            return true;
        }

        if let Some(resident) = mc.state_storage.seen.get(&fp) {
            self.payload_witnesses
                .record_array_state_if_absent(fp, resident);
            return self
                .payload_witnesses
                .confirm_array_state(fp, candidate)
                .unwrap_or(false);
        }

        false
    }

    fn duplicate_flat_payload_confirmed(
        &mut self,
        fp: Fingerprint,
        candidate: &ArrayState,
        current: Option<(Fingerprint, &ArrayState)>,
        mc: &ModelChecker,
    ) -> bool {
        let Some(candidate_slots) = Self::flat_payload_slots(candidate, mc) else {
            return false;
        };

        if let Some(confirmed) = self
            .payload_witnesses
            .confirm_flat_i64_slots(fp, &candidate_slots)
        {
            return confirmed;
        }

        if Self::current_flat_payload_confirms(fp, &candidate_slots, current, mc) {
            return true;
        }

        if let Some(resident) = mc.state_storage.seen.get(&fp) {
            let Some(resident_slots) = Self::flat_payload_slots(resident, mc) else {
                return false;
            };
            self.payload_witnesses
                .record_flat_i64_slots_if_absent(fp, &resident_slots);
            return self
                .payload_witnesses
                .confirm_flat_i64_slots(fp, &candidate_slots)
                .unwrap_or(false);
        }

        false
    }

    fn duplicate_payload_confirmed(
        &mut self,
        fp: Fingerprint,
        candidate: &ArrayState,
        current: Option<(Fingerprint, &ArrayState)>,
        witness_domain: PayloadWitnessDomain,
        mc: &mut ModelChecker,
    ) -> Result<bool, CheckResult> {
        match witness_domain {
            PayloadWitnessDomain::ArrayState => {
                Ok(self.duplicate_array_payload_confirmed(fp, candidate, current, mc))
            }
            PayloadWitnessDomain::FlatI64 => {
                Ok(self.duplicate_flat_payload_confirmed(fp, candidate, current, mc))
            }
            PayloadWitnessDomain::View => Ok(self
                .duplicate_view_payload_confirmed(fp, candidate, current, mc)?
                .0),
        }
    }

    fn view_payload_value(
        state: &ArrayState,
        mc: &mut ModelChecker,
    ) -> Result<Option<Value>, CheckResult> {
        let Some(view_name) = mc.compiled.cached_view_name.clone() else {
            return Ok(None);
        };
        let bfs_level = mc.ctx.get_tlc_level();
        crate::checker_ops::compute_view_value_array(&mut mc.ctx, state, &view_name, bfs_level)
            .map(Some)
            .map_err(|error| check_error_to_result(error, &mc.stats))
    }

    fn duplicate_view_payload_confirmed(
        &mut self,
        fp: Fingerprint,
        candidate: &ArrayState,
        current: Option<(Fingerprint, &ArrayState)>,
        mc: &mut ModelChecker,
    ) -> Result<(bool, Option<Value>), CheckResult> {
        let Some(candidate_value) = Self::view_payload_value(candidate, mc)? else {
            return Ok((false, None));
        };

        if let Some(witness) = self.view_payload_witnesses.get(&fp) {
            return Ok((witness == &candidate_value, Some(candidate_value)));
        }

        if let Some((_, current_state)) = current.filter(|(current_fp, _)| *current_fp == fp) {
            if let Some(current_value) = Self::view_payload_value(current_state, mc)? {
                let confirmed = current_value == candidate_value;
                self.view_payload_witnesses
                    .entry(fp)
                    .or_insert(current_value);
                return Ok((confirmed, Some(candidate_value)));
            }
        }

        if let Some(resident_value) = Self::resident_view_payload_value(fp, mc)? {
            let confirmed = resident_value == candidate_value;
            self.view_payload_witnesses
                .entry(fp)
                .or_insert(resident_value);
            return Ok((confirmed, Some(candidate_value)));
        }

        Ok((false, Some(candidate_value)))
    }

    fn resident_view_payload_value(
        fp: Fingerprint,
        mc: &mut ModelChecker,
    ) -> Result<Option<Value>, CheckResult> {
        let Some(view_name) = mc.compiled.cached_view_name.clone() else {
            return Ok(None);
        };
        let Some(resident) = mc.state_storage.seen.get(&fp) else {
            return Ok(None);
        };
        let bfs_level = mc.ctx.get_tlc_level();
        crate::checker_ops::compute_view_value_array(&mut mc.ctx, resident, &view_name, bfs_level)
            .map(Some)
            .map_err(|error| check_error_to_result(error, &mc.stats))
    }

    fn record_view_payload_witness_value_if_absent(&mut self, fp: Fingerprint, value: Value) {
        self.view_payload_witnesses.entry(fp).or_insert(value);
    }

    fn record_payload_witness_if_absent(
        &mut self,
        fp: Fingerprint,
        state: &ArrayState,
        witness_domain: PayloadWitnessDomain,
        mc: &ModelChecker,
    ) {
        match witness_domain {
            PayloadWitnessDomain::ArrayState => self
                .payload_witnesses
                .record_array_state_if_absent(fp, state),
            PayloadWitnessDomain::FlatI64 => {
                if let Some(slots) = Self::flat_payload_slots(state, mc) {
                    self.payload_witnesses
                        .record_flat_i64_slots_if_absent(fp, &slots);
                }
            }
            PayloadWitnessDomain::View => {}
        }
    }

    fn record_debug_seen_state(
        fp: Fingerprint,
        state: &ArrayState,
        depth: usize,
        mc: &mut ModelChecker,
    ) {
        mc.debug_record_seen_state_array(fp, state, depth);
    }

    fn write_trace_for_inserted(
        fp: Fingerprint,
        parent_fp: Option<Fingerprint>,
        depth: usize,
        mc: &mut ModelChecker,
    ) -> u64 {
        if let Some(ref mut trace_file) = mc.trace.trace_file {
            let loc = if let Some(parent_fp) = parent_fp {
                let parent_loc = if let Some(cached) = mc.trace.current_parent_trace_loc {
                    cached
                } else {
                    match mc.trace.trace_locs.get(&parent_fp) {
                        Some(loc) => loc,
                        None => {
                            if !mc.trace.trace_degraded {
                                eprintln!(
                                    "WARNING: parent fingerprint {parent_fp:?} not found in trace location index (using root as fallback)"
                                );
                            }
                            0
                        }
                    }
                };
                match trace_file.write_state(parent_loc, fp) {
                    Ok(loc) => Some(loc),
                    Err(e) => {
                        mc.mark_trace_degraded(&e);
                        None
                    }
                }
            } else {
                match trace_file.write_initial(fp) {
                    Ok(loc) => Some(loc),
                    Err(e) => {
                        mc.mark_trace_degraded(&e);
                        None
                    }
                }
            };

            if let Some(loc) = loc {
                mc.trace.last_inserted_trace_loc = loc;
                if !mc.trace.lazy_trace_index && !mc.trace.trace_locs.insert(fp, loc) {
                    mc.trace.trace_degraded = true;
                }
            }
        }

        if mc.checkpoint.dir.is_some() {
            mc.trace.depths.insert(fp, depth);
        }

        mc.trace.last_inserted_trace_loc
    }

    fn queue_entry_for_inserted(
        &self,
        fp: Fingerprint,
        state: ArrayState,
        depth: usize,
        trace_loc: u64,
    ) -> (NoTraceQueueEntry, usize, u64) {
        // Keep the admitted successor as an ArrayState until it reaches
        // FlatBfsFrontier. The frontier can write it directly into its arena,
        // avoiding a temporary boxed FlatState and an immediate second copy.
        // The witness frontier instead retains an exact witness handle when
        // doing so would not discard a complete per-slot fingerprint cache.
        // All other non-flat frontier entries remain Owned.
        let entry = if self.use_array_witness_frontier && !state.has_complete_fp_cache() {
            NoTraceQueueEntry::Witness {
                fp,
                combined_xor: state.cached_combined_xor().unwrap_or(0),
            }
        } else {
            NoTraceQueueEntry::Owned { state, fp }
        };

        (entry, depth, trace_loc)
    }
}

impl BfsStorage for FingerprintOnlyStorage {
    /// Queue entries carry `(NoTraceQueueEntry, depth, trace_loc)`.
    ///
    /// Part of #2881 Step 3: the `trace_loc` (u64) is the trace file offset of
    /// this state, carried on the queue entry to eliminate per-successor HashMap
    /// reads when processing this state's successors. Matches TLC's pattern of
    /// embedding per-state metadata on the state/queue object.
    type QueueEntry = (NoTraceQueueEntry, usize, u64);

    fn release_after_complete_bfs(&mut self) {
        // All Bulk queue handles have been consumed once BFS completes, and
        // collision witnesses plus batch buffers are only consulted during
        // admission. Replace rather than clear so their backing allocations are
        // returned before potentially long-running post-BFS liveness checking.
        if mem_census_enabled() {
            let (unique, hits, fingerprint_fallbacks, estimated_bytes) =
                self.payload_witnesses.compact_value_pool_census();
            eprintln!(
                "[mem-census:fp-only-pre-release] compact-pool(unique={unique} hits={hits} \
                 fingerprint-fallbacks={fingerprint_fallbacks} estimated-bytes={estimated_bytes}) \
                 diff-prefilter(checks={} confirmed={} fallbacks={})",
                self.diff_witness_prefilter_checks,
                self.diff_witness_prefilter_confirmed,
                self.diff_witness_prefilter_fallbacks,
            );
        }
        self.bulk_initial = BulkStateStorage::empty(self.num_vars);
        self.payload_witnesses.release_storage();
        self.view_payload_witnesses = fp_hashmap();
        self.batch_fingerprints = Vec::new();
        self.batch_admission = BatchInsertedIndexAdmission::default();
    }

    fn payload_witness_memory_bytes(&self) -> usize {
        let view_witness_bytes = crate::memory::estimate_hashmap_bytes::<Fingerprint, Value>(
            self.view_payload_witnesses.capacity(),
        );
        self.payload_witnesses
            .estimated_memory_bytes()
            .saturating_add(view_witness_bytes)
    }

    fn dequeue(
        &mut self,
        (entry, depth, trace_loc): (NoTraceQueueEntry, usize, u64),
        mc: &mut ModelChecker,
    ) -> Result<Option<(Fingerprint, ArrayState, usize)>, CheckResult> {
        // Part of #2881 Step 3: cache the parent's trace_loc for successor admission.
        mc.trace.current_parent_trace_loc = Some(trace_loc);
        match entry {
            NoTraceQueueEntry::Bulk(handle) => {
                let mut state = ArrayState::new(self.num_vars);
                state.overwrite_from_slice(self.bulk_initial.get_state(handle.index));

                // Re-materialize lazy values after bulk reload.
                if let Err(e) = crate::materialize::materialize_array_state(
                    &mc.ctx,
                    &mut state,
                    mc.compiled.spec_may_produce_lazy,
                ) {
                    return Err(check_error_to_result(
                        EvalCheckError::Eval(e).into(),
                        &mc.stats,
                    ));
                }

                let fp = match handle.fingerprint {
                    Some(fp) => fp,
                    None => mc
                        .array_state_fingerprint(&mut state)
                        .map_err(|e| check_error_to_result(e, &mc.stats))?,
                };
                if Self::compact_payload_witness_domain(mc) == Some(PayloadWitnessDomain::View) {
                    if let Some(view_payload_value) = Self::view_payload_value(&state, mc)? {
                        self.record_view_payload_witness_value_if_absent(fp, view_payload_value);
                    }
                } else if let Some(witness_domain) = Self::compact_payload_witness_domain(mc) {
                    self.record_payload_witness_if_absent(fp, &state, witness_domain, mc);
                }
                Ok(Some((fp, state, depth)))
            }
            NoTraceQueueEntry::Owned { state, fp } => Ok(Some((fp, state, depth))),
            NoTraceQueueEntry::Witness { fp, combined_xor } => {
                let Some(mut state) = self.payload_witnesses.materialize_array_state(fp) else {
                    return Err(check_error_to_result(
                        crate::RuntimeCheckError::Internal(format!(
                            "missing or malformed ArrayState payload witness for queued fingerprint {fp:?}"
                        ))
                        .into(),
                        &mc.stats,
                    ));
                };
                state.set_cached_fingerprint_with_xor(fp, combined_xor);
                Ok(Some((fp, state, depth)))
            }
            NoTraceQueueEntry::Flat { flat, fp } => {
                if Self::compact_payload_witness_domain(mc) == Some(PayloadWitnessDomain::FlatI64) {
                    self.payload_witnesses
                        .record_flat_i64_slots_if_absent(fp, flat.buffer());
                }
                // Part of #4126: Tier 0 interpreter sandwich — convert FlatState
                // to ArrayState for interpreter evaluation.
                let registry = mc.ctx.var_registry().clone();
                let state = match mc.flat_bfs_adapter.as_mut() {
                    Some(adapter) => adapter.flat_to_array(&flat, &registry, None),
                    None => {
                        // Adapter not initialized — should not happen when Flat entries exist.
                        return Err(check_error_to_result(
                            crate::ConfigCheckError::Setup(
                                "FlatBfsAdapter not initialized for Flat queue entry".to_string(),
                            )
                            .into(),
                            &mc.stats,
                        ));
                    }
                };
                // Record the ArrayState collision witness for the reconstructed
                // state. Init states enter the flat arena before admit_successor
                // runs, so the witness for the ArrayFp64 domain must be captured
                // here to keep content-based dedup sound under the flat frontier.
                if Self::compact_payload_witness_domain(mc)
                    == Some(PayloadWitnessDomain::ArrayState)
                {
                    self.record_payload_witness_if_absent(
                        fp,
                        &state,
                        PayloadWitnessDomain::ArrayState,
                        mc,
                    );
                }
                Ok(Some((fp, state, depth)))
            }
        }
    }

    fn return_current(&mut self, _fp: Fingerprint, _state: ArrayState, _mc: &mut ModelChecker) {
        // No-op: fingerprint-only mode doesn't store states in a HashMap.
    }

    fn admit_successor(
        &mut self,
        fp: Fingerprint,
        state: ArrayState,
        parent_fp: Option<Fingerprint>,
        current: Option<(Fingerprint, &ArrayState)>,
        depth: usize,
        mc: &mut ModelChecker,
    ) -> Result<Option<(NoTraceQueueEntry, usize, u64)>, CheckResult> {
        let witness_domain = Self::compact_payload_witness_domain(mc);
        let view_payload_value = if witness_domain == Some(PayloadWitnessDomain::View) {
            Self::record_debug_seen_state(fp, &state, depth, mc);
            let (duplicate_payload_confirmed, view_payload_value) =
                self.duplicate_view_payload_confirmed(fp, &state, current, mc)?;
            let inserted = mc.mark_state_seen_fp_only_with_duplicate_payload_checked(
                fp,
                parent_fp,
                depth,
                duplicate_payload_confirmed,
            )?;
            if inserted {
                if let Some(ref detector) = mc.collision_detector {
                    detector.record_state(fp, &state);
                }
            }
            Some((inserted, view_payload_value))
        } else {
            None
        };

        let inserted = if let Some((inserted, _)) = view_payload_value.as_ref() {
            *inserted
        } else if let Some(witness_domain) = witness_domain {
            Self::record_debug_seen_state(fp, &state, depth, mc);
            let duplicate_payload_confirmed =
                self.duplicate_payload_confirmed(fp, &state, current, witness_domain, mc)?;
            mc.mark_state_seen_fp_only_with_duplicate_payload_checked(
                fp,
                parent_fp,
                depth,
                duplicate_payload_confirmed,
            )?
        } else {
            mc.mark_state_seen_checked_with_current(fp, &state, parent_fp, depth, current)?
        };

        if inserted {
            if let Some((_, Some(view_payload_value))) = view_payload_value {
                self.record_view_payload_witness_value_if_absent(fp, view_payload_value);
            } else if let Some(witness_domain) = Self::compact_payload_witness_domain(mc) {
                if let Some(ref detector) = mc.collision_detector {
                    detector.record_state(fp, &state);
                }
                self.record_payload_witness_if_absent(fp, &state, witness_domain, mc);
            }
            // Part of #2881 Step 3: carry the new state's trace_loc on the queue
            // entry so successors can use it without a HashMap read.
            let trace_loc = mc.trace.last_inserted_trace_loc;

            // Part of #4126: When flat BFS is active (auto-detected for scalar
            // specs or force-enabled via use_flat_state=Some(true)), keep the
            // ArrayState intact until FlatBfsFrontier writes it directly into
            // the arena.
            // This avoids constructing a temporary boxed FlatState that would
            // be copied again immediately by the frontier.
            //
            // `should_use_flat_bfs()` encapsulates the full decision hierarchy:
            // env var overrides, config overrides, and auto-detection for fully-flat
            // layouts with verified roundtrip.
            Ok(Some(
                self.queue_entry_for_inserted(fp, state, depth, trace_loc),
            ))
        } else {
            Ok(None)
        }
    }

    fn admit_successor_batch_into(
        &mut self,
        candidates: &mut Vec<BfsAdmissionCandidate>,
        current: Option<(Fingerprint, &ArrayState)>,
        result: &mut BfsBatchAdmissionResult<(NoTraceQueueEntry, usize, u64)>,
        mc: &mut ModelChecker,
    ) {
        result.clear_for_capacity(candidates.len());
        if candidates.is_empty() {
            return;
        }

        let Some(witness_domain) = Self::compact_payload_witness_domain(mc) else {
            for candidate in candidates.drain(..) {
                let BfsAdmissionCandidate {
                    fp,
                    state,
                    parent_fp,
                    depth,
                } = candidate;
                let current = current.filter(|(current_fp, _)| Some(*current_fp) == parent_fp);
                let entry = match self.admit_successor(fp, state, parent_fp, current, depth, mc) {
                    Ok(entry) => entry,
                    Err(error) => {
                        result.fault = Some(error);
                        break;
                    }
                };
                result.entries.push(BfsAdmittedEntry { depth, entry });
            }
            return;
        };

        for candidate in candidates.iter() {
            Self::record_debug_seen_state(candidate.fp, &candidate.state, candidate.depth, mc);
        }

        self.batch_fingerprints.clear();
        self.batch_fingerprints.reserve(candidates.len());
        self.batch_fingerprints
            .extend(candidates.iter().map(|candidate| candidate.fp));
        mc.state_storage
            .seen_fps
            .insert_prechecked_absent_batch_inserted_indices_checked_into(
                &self.batch_fingerprints,
                &mut self.batch_admission,
            );

        let attempted = if self.batch_admission.fault.is_some() {
            self.batch_admission.attempted.saturating_sub(1)
        } else {
            self.batch_admission.attempted
        };
        let mut inserted_cursor = 0usize;

        for (idx, mut candidate) in candidates.drain(..).enumerate() {
            if idx >= attempted {
                break;
            }

            if self
                .batch_admission
                .inserted_indices
                .get(inserted_cursor)
                .copied()
                == Some(idx)
            {
                inserted_cursor += 1;
                if let Some(ref detector) = mc.collision_detector {
                    detector.record_state(candidate.fp, &candidate.state);
                }
                if witness_domain == PayloadWitnessDomain::View {
                    match Self::view_payload_value(&candidate.state, mc) {
                        Ok(Some(value)) => {
                            self.record_view_payload_witness_value_if_absent(candidate.fp, value)
                        }
                        Ok(None) => {}
                        Err(error) => {
                            result.fault = Some(error);
                            break;
                        }
                    }
                } else {
                    self.record_payload_witness_if_absent(
                        candidate.fp,
                        &candidate.state,
                        witness_domain,
                        mc,
                    );
                }

                let trace_loc = Self::write_trace_for_inserted(
                    candidate.fp,
                    candidate.parent_fp,
                    candidate.depth,
                    mc,
                );
                let entry = self.queue_entry_for_inserted(
                    candidate.fp,
                    std::mem::replace(&mut candidate.state, ArrayState::new(0)),
                    candidate.depth,
                    trace_loc,
                );
                result.entries.push(BfsAdmittedEntry {
                    depth: candidate.depth,
                    entry: Some(entry),
                });
            } else {
                let current =
                    current.filter(|(current_fp, _)| Some(*current_fp) == candidate.parent_fp);
                let duplicate_payload_confirmed = match self.duplicate_payload_confirmed(
                    candidate.fp,
                    &candidate.state,
                    current,
                    witness_domain,
                    mc,
                ) {
                    Ok(confirmed) => confirmed,
                    Err(error) => {
                        result.fault = Some(error);
                        break;
                    }
                };
                if let Err(error) = mc.enforce_fingerprint_only_duplicate_with_payload_confirmation(
                    duplicate_payload_confirmed,
                ) {
                    result.fault = Some(error);
                    break;
                }
                result.entries.push(BfsAdmittedEntry {
                    depth: candidate.depth,
                    entry: None,
                });
            }
        }

        if result.fault.is_none() {
            if let Some(fault) = self.batch_admission.fault.take() {
                result.fault = Some(mc.storage_fault_result(fault));
            }
        }
    }

    fn enforce_seen_successor_duplicate(
        &mut self,
        fp: Fingerprint,
        candidate: &ArrayState,
        current: Option<(Fingerprint, &ArrayState)>,
        mc: &mut ModelChecker,
    ) -> Result<(), CheckResult> {
        let Some(witness_domain) = Self::compact_payload_witness_domain(mc) else {
            return mc.enforce_seen_state_duplicate_with_payload(fp, candidate, current);
        };
        let duplicate_payload_confirmed =
            self.duplicate_payload_confirmed(fp, candidate, current, witness_domain, mc)?;
        mc.enforce_fingerprint_only_duplicate_with_payload_confirmation(duplicate_payload_confirmed)
    }

    fn confirm_seen_array_diff_duplicate(
        &mut self,
        fp: Fingerprint,
        base: &ArrayState,
        changes: &[(crate::var_index::VarIndex, Value)],
    ) -> Option<bool> {
        if !self.diff_witness_prefilter_enabled {
            return None;
        }

        self.diff_witness_prefilter_checks = self.diff_witness_prefilter_checks.saturating_add(1);
        let result = self
            .payload_witnesses
            .confirm_array_state_diff(fp, base, changes);
        if result == Some(true) {
            self.diff_witness_prefilter_confirmed =
                self.diff_witness_prefilter_confirmed.saturating_add(1);
        } else {
            self.diff_witness_prefilter_fallbacks =
                self.diff_witness_prefilter_fallbacks.saturating_add(1);
        }
        #[cfg(test)]
        record_diff_witness_prefilter_test_telemetry(
            self.diff_witness_prefilter_checks,
            self.diff_witness_prefilter_confirmed,
            self.diff_witness_prefilter_fallbacks,
        );
        result
    }

    fn use_diffs(&self, mc: &ModelChecker) -> bool {
        // Nested-set A6: the diff path is NO LONGER disabled when a monitor is
        // installed. The per-successor escape monitor is hooked INTO the streaming
        // ClosureSink (`observe_diff_monitors_escape`), so it still sees EVERY
        // successor's board and fails closed on escape — without forcing the slow
        // batch path. The diff path's `compute_diff_fingerprint_with_xor` already
        // produces `value_fingerprint(board)`, which byte-matches the monitored
        // `dedup_fp`, so the verdict is identical and the monitor stays unbypassable.
        mc.compiled.cached_view_name.is_none()
            && mc.symmetry.perms.is_empty()
            && !mc.liveness_cache.cache_for_liveness
    }

    fn checkpoint_frontier(
        &self,
        current: &ArrayState,
        queue: &impl BfsFrontier<Entry = (NoTraceQueueEntry, usize, u64)>,
        registry: &crate::var_index::VarRegistry,
        mc: &mut ModelChecker,
    ) -> VecDeque<State> {
        checkpoint_view::build_checkpoint_frontier(current, queue, registry, |(entry, _, _)| {
            Some(match entry {
                NoTraceQueueEntry::Bulk(handle) => {
                    State::from_indexed(self.bulk_initial.get_state(handle.index), registry)
                }
                NoTraceQueueEntry::Owned { state, .. } => state.to_state(registry),
                NoTraceQueueEntry::Witness { fp, .. } => self
                    .payload_witnesses
                    .materialize_array_state(*fp)
                    .unwrap_or_else(|| {
                        panic!(
                            "missing or malformed ArrayState payload witness for checkpoint fingerprint {fp:?}"
                        )
                    })
                    .to_state(registry),
                NoTraceQueueEntry::Flat { flat, .. } => {
                    // Part of #4126: Convert FlatState back to State for checkpoint.
                    match mc.flat_bfs_bridge() {
                        Some(bridge) => {
                            let arr = bridge.to_array_state(flat, registry);
                            arr.to_state(registry)
                        }
                        None => {
                            // Shouldn't happen: Flat entries require the bridge.
                            // Return a placeholder state to avoid panicking in checkpoint.
                            State::new()
                        }
                    }
                }
            })
        })
    }

    fn cache_diff_liveness(
        &self,
        _parent_fp: Fingerprint,
        _succ_fps: Option<Vec<Fingerprint>>,
        _mc: &mut ModelChecker,
    ) -> Result<(), crate::check::CheckError> {
        // No-op: use_diffs() returns false when cache_for_liveness is true,
        // so this method is never called on the diff path.
        Ok(())
    }

    fn cache_full_liveness(
        &self,
        parent_fp: Fingerprint,
        successors: &[(ArrayState, Fingerprint)],
        mc: &mut ModelChecker,
    ) -> Result<(), crate::check::CheckError> {
        if !mc.liveness_cache.cache_for_liveness {
            return Ok(());
        }
        let succ_fps: Vec<Fingerprint> = successors.iter().map(|(_, fp)| *fp).collect();
        mc.liveness_cache.successors.insert(parent_fp, succ_fps)?;
        // No-trace mode does not cache symmetry witness states — symmetry
        // liveness requires full-state mode for witness reconstruction.
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arena::BulkStateStorage;
    use crate::config::Config;
    use crate::test_support::parse_module;
    use crate::Value;

    fn minimal_module() -> tla_core::ast::Module {
        parse_module(
            r#"
---- MODULE FingerprintOnlyPayloadWitnessTest ----
VARIABLE x
Init == x = 0
Next == x' = x + 1
====
"#,
        )
    }

    fn minimal_config() -> Config {
        Config {
            init: Some("Init".to_string()),
            next: Some("Next".to_string()),
            ..Default::default()
        }
    }

    fn successful_count_tuple(result: CheckResult, label: &str) -> (usize, usize, usize, usize) {
        match result {
            CheckResult::Success(stats) => (
                stats.initial_states,
                stats.states_found,
                stats.transitions,
                stats.max_depth,
            ),
            other => panic!("{label} should succeed, got {other:?}"),
        }
    }

    fn assert_default_tir_eval(checker: &ModelChecker<'_>) {
        assert!(
            checker
                .tir_parity
                .as_ref()
                .is_some_and(super::super::super::super::tir_parity::TirParityState::is_eval_mode),
            "the regression must exercise the sequential checker's default TIR Eval mode",
        );
    }

    /// End-to-end regression for the constrained batch-diff route used by specs
    /// such as Sailfish. The compound `payload` keeps the test in ArrayFp64,
    /// while `x' \in 0..2` produces both exact duplicates and a new candidate
    /// rejected by the state constraint.
    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn exact_diff_prefilter_reuses_only_pure_state_constraints() {
        let _quiescence = crate::test_utils::acquire_model_check_quiescence_lock();
        let module = parse_module(
            r#"
---- MODULE ExactDiffPureConstraintE2E ----
EXTENDS Integers, Sequences, TLC

VARIABLE x, payload

Init == x = 0 /\ payload = <<42>>
Next == /\ x' \in 0..2
        /\ UNCHANGED payload

PureConstraint == x \in 0..1
ContextConstraint == /\ x \in 0..1
                     /\ TLCGet("level") < 100
====
"#,
        );
        let pure_config = Config {
            init: Some("Init".to_string()),
            next: Some("Next".to_string()),
            constraints: vec!["PureConstraint".to_string()],
            check_deadlock: false,
            auto_por: Some(false),
            use_flat_state: Some(false),
            use_compiled_bfs: Some(false),
            ..Default::default()
        };

        // Full-state storage cannot use the ArrayState virtual-payload proof and
        // is the result/count oracle for the optimized fingerprint-only run.
        let baseline_counts = {
            let mut checker = ModelChecker::new(&module, &pure_config);
            assert_default_tir_eval(&checker);
            checker.set_store_states(true);
            successful_count_tuple(checker.check(), "full-state baseline")
        };

        reset_diff_witness_prefilter_test_telemetry();
        let optimized_counts = {
            let mut checker = ModelChecker::new(&module, &pure_config);
            assert_default_tir_eval(&checker);
            checker.set_store_states(false);
            let result = checker.check();
            assert!(
                checker
                    .compiled
                    .state_constraints_reusable_on_exact_duplicate,
                "setup should certify the pure, unprimed state constraint",
            );
            successful_count_tuple(result, "fingerprint-only optimized run")
        };
        let optimized_telemetry = diff_witness_prefilter_test_telemetry();

        assert_eq!(
            optimized_counts, baseline_counts,
            "virtual exact-duplicate confirmation must preserve the full-state verdict/counts",
        );
        assert_eq!(
            optimized_counts,
            (1, 2, 4, 1),
            "two admitted states should each have two constraint-admitted successors",
        );
        assert!(
            optimized_telemetry.confirmed > 0,
            "the end-to-end run must actually confirm an exact duplicate before materialization: {optimized_telemetry:?}",
        );
        assert!(
            optimized_telemetry.fallbacks > 0,
            "new and constraint-rejected candidates must retain the canonical fallback: {optimized_telemetry:?}",
        );
        assert_eq!(
            optimized_telemetry.checks,
            optimized_telemetry.confirmed + optimized_telemetry.fallbacks,
        );

        // A context-dependent constraint has the same verdict on this bounded
        // model, but TLCGet makes its result non-reusable. It must therefore
        // keep every duplicate on the materialized constraint-evaluation path.
        let impure_config = Config {
            constraints: vec!["ContextConstraint".to_string()],
            ..pure_config.clone()
        };
        reset_diff_witness_prefilter_test_telemetry();
        let impure_counts = {
            let mut checker = ModelChecker::new(&module, &impure_config);
            assert_default_tir_eval(&checker);
            checker.set_store_states(false);
            let result = checker.check();
            assert!(
                !checker
                    .compiled
                    .state_constraints_reusable_on_exact_duplicate,
                "TLCGet-dependent state constraints must fail closed",
            );
            successful_count_tuple(result, "context-dependent constraint run")
        };

        assert_eq!(impure_counts, baseline_counts);
        assert_eq!(
            diff_witness_prefilter_test_telemetry(),
            DiffWitnessPrefilterTestTelemetry::default(),
            "an uncertified state constraint must not reach the exact-duplicate prefilter",
        );
    }

    #[test]
    fn diff_witness_prefilter_requires_an_exact_virtual_payload() {
        let bulk_initial = BulkStateStorage::empty(2);
        let mut storage = FingerprintOnlyStorage::new(bulk_initial, 2);
        let fp = Fingerprint(19);
        let stored =
            ArrayState::from_values(vec![Value::set((0..10).map(Value::int)), Value::int(2)]);
        let base =
            ArrayState::from_values(vec![Value::set((0..10).map(Value::int)), Value::int(1)]);
        storage
            .payload_witnesses
            .record_array_state_if_absent(fp, &stored);

        assert_eq!(
            storage.confirm_seen_array_diff_duplicate(
                fp,
                &base,
                &[(crate::var_index::VarIndex::new(1), Value::int(2))],
            ),
            Some(true)
        );
        assert_eq!(
            storage.confirm_seen_array_diff_duplicate(
                fp,
                &base,
                &[(crate::var_index::VarIndex::new(1), Value::int(3))],
            ),
            Some(false)
        );
        assert_eq!(storage.diff_witness_prefilter_checks, 2);
        assert_eq!(storage.diff_witness_prefilter_confirmed, 1);
        assert_eq!(storage.diff_witness_prefilter_fallbacks, 1);

        storage.diff_witness_prefilter_enabled = false;
        assert_eq!(
            storage.confirm_seen_array_diff_duplicate(
                fp,
                &base,
                &[(crate::var_index::VarIndex::new(1), Value::int(2))],
            ),
            None,
            "the kill switch must restore full materialization"
        );
        assert_eq!(storage.diff_witness_prefilter_checks, 2);
    }

    #[test]
    fn witness_queue_entry_roundtrips_payload_and_incremental_cache() {
        let module = minimal_module();
        let config = Config {
            use_flat_state: Some(false),
            ..minimal_config()
        };
        let mut mc = ModelChecker::new(&module, &config);
        let registry = mc.ctx.var_registry().clone();
        let mut state = ArrayState::from_values(vec![Value::seq([Value::int(7), Value::int(11)])]);
        let fp = state.fingerprint(&registry);
        let combined_xor = state
            .cached_combined_xor()
            .expect("fingerprinted state has an incremental base");
        let expected = state.values().to_vec();
        state = ArrayState::from_compact_values(state.compact_values_arc());
        state.set_cached_fingerprint_with_xor(fp, combined_xor);

        let mut storage = FingerprintOnlyStorage::new(BulkStateStorage::empty(1), 1);
        storage.use_array_witness_frontier = true;
        storage
            .payload_witnesses
            .record_array_state_if_absent(fp, &state);
        let queued = storage.queue_entry_for_inserted(fp, state, 3, 91);
        assert!(matches!(
            &queued.0,
            NoTraceQueueEntry::Witness {
                fp: queued_fp,
                combined_xor: queued_xor,
            } if *queued_fp == fp && *queued_xor == combined_xor
        ));

        let (restored_fp, restored, depth) = storage
            .dequeue(queued, &mut mc)
            .expect("witness dequeue succeeds")
            .expect("witness dequeue is never phantom");
        assert_eq!(restored_fp, fp);
        assert_eq!(depth, 3);
        assert_eq!(mc.trace.current_parent_trace_loc, Some(91));
        assert_eq!(restored.values(), expected.as_slice());
        assert_eq!(restored.cached_fingerprint(), Some(fp));
        assert_eq!(restored.cached_combined_xor(), Some(combined_xor));
    }

    #[test]
    fn witness_queue_preserves_owned_states_with_complete_fingerprint_caches() {
        let module = minimal_module();
        let config = Config {
            use_flat_state: Some(false),
            ..minimal_config()
        };
        let mc = ModelChecker::new(&module, &config);
        let registry = mc.ctx.var_registry().clone();
        let mut state = ArrayState::from_values(vec![Value::seq([Value::int(7)])]);
        let fp = state.fingerprint(&registry);
        assert!(state.has_complete_fp_cache());

        let mut storage = FingerprintOnlyStorage::new(BulkStateStorage::empty(1), 1);
        storage.use_array_witness_frontier = true;
        storage
            .payload_witnesses
            .record_array_state_if_absent(fp, &state);
        let queued = storage.queue_entry_for_inserted(fp, state, 2, 17);

        assert!(matches!(queued.0, NoTraceQueueEntry::Owned { .. }));
    }

    #[test]
    fn missing_witness_queue_payload_fails_closed() {
        let module = minimal_module();
        let config = minimal_config();
        let mut mc = ModelChecker::new(&module, &config);
        let mut storage = FingerprintOnlyStorage::new(BulkStateStorage::empty(1), 1);
        let entry = (
            NoTraceQueueEntry::Witness {
                fp: Fingerprint(0xdead_beef),
                combined_xor: 7,
            },
            4,
            92,
        );

        assert!(storage.dequeue(entry, &mut mc).is_err());
        assert_eq!(mc.trace.current_parent_trace_loc, Some(92));
    }

    fn compiled_flat_checker<'a>(
        module: &'a tla_core::ast::Module,
        config: &'a Config,
    ) -> ModelChecker<'a> {
        let mut mc = ModelChecker::new(module, config);
        let init = ArrayState::from_values(vec![Value::int(0)]);
        mc.infer_flat_state_layout(&init);
        mc.flat_state_primary = true;
        mc
    }

    #[test]
    fn compiled_flat_payload_witness_records_flat_slots() {
        let module = minimal_module();
        let config = minimal_config();
        let mc = compiled_flat_checker(&module, &config);
        let mut storage = FingerprintOnlyStorage::new(BulkStateStorage::empty(1), 1);
        let fp = Fingerprint(77);
        let state = ArrayState::from_values(vec![Value::int(7)]);

        storage.record_payload_witness_if_absent(fp, &state, PayloadWitnessDomain::FlatI64, &mc);

        assert_eq!(
            storage.payload_witnesses.confirm_flat_i64_slots(fp, &[7]),
            Some(true)
        );
        assert_eq!(
            storage.payload_witnesses.confirm_array_state(fp, &state),
            Some(false),
            "compiled-flat admission must not authorize duplicates with ArrayState payload witnesses"
        );
    }

    #[test]
    fn payload_witness_memory_includes_view_witness_map() {
        let mut storage = FingerprintOnlyStorage::new(BulkStateStorage::empty(1), 1);
        let empty_bytes = storage.payload_witness_memory_bytes();

        storage.record_view_payload_witness_value_if_absent(
            Fingerprint(78),
            Value::seq([Value::int(1), Value::int(2)]),
        );

        assert!(storage.payload_witness_memory_bytes() > empty_bytes);
    }

    #[test]
    fn release_after_complete_bfs_drops_fp_only_allocations() {
        let mut bulk_initial = BulkStateStorage::new(1, 32);
        bulk_initial.push_state(&[Value::int(1)]);
        let mut storage = FingerprintOnlyStorage::new(bulk_initial, 1);
        let fresh_payload_witness_bytes = storage.payload_witness_memory_bytes();
        let fp = Fingerprint(17);
        storage
            .payload_witnesses
            .record_flat_i64_slots_if_absent(fp, &[1]);
        let pooled_fp = Fingerprint(18);
        let pooled_state =
            ArrayState::from_values(vec![Value::set((0..10).map(Value::int)), Value::int(2)]);
        storage
            .payload_witnesses
            .record_array_state_if_absent(pooled_fp, &pooled_state);
        storage.view_payload_witnesses.insert(fp, Value::int(1));
        storage.batch_fingerprints.reserve(64);
        storage.batch_admission.inserted_indices.reserve(64);

        assert!(storage.bulk_initial.memory_usage() > 0);
        assert!(!storage.payload_witnesses.is_empty());
        assert_eq!(storage.payload_witnesses.compact_value_pool_stats().0, 2);
        assert!(!storage.view_payload_witnesses.is_empty());
        assert!(storage.batch_fingerprints.capacity() >= 64);
        assert!(storage.batch_admission.inserted_indices.capacity() >= 64);

        storage.release_after_complete_bfs();

        assert!(storage.bulk_initial.is_empty());
        assert_eq!(storage.bulk_initial.memory_usage(), 0);
        assert!(storage.payload_witnesses.is_empty());
        assert_eq!(
            storage.payload_witnesses.compact_value_pool_stats(),
            (0, 0, 0)
        );
        assert!(storage.view_payload_witnesses.is_empty());
        assert_eq!(storage.view_payload_witnesses.capacity(), 0);
        assert_eq!(storage.batch_fingerprints.capacity(), 0);
        assert_eq!(storage.batch_admission.inserted_indices.capacity(), 0);
        assert_eq!(
            storage.payload_witness_memory_bytes(),
            fresh_payload_witness_bytes
        );
    }

    #[test]
    fn memory_breakdown_includes_active_and_compiled_payload_witnesses() {
        let module = minimal_module();
        let config = minimal_config();
        let mut mc = compiled_flat_checker(&module, &config);
        mc.state_storage
            .compiled_flat_payload_witnesses
            .record_flat_i64_slots_if_absent(Fingerprint(79), &[7, 8]);
        let compiled_bytes = mc
            .state_storage
            .compiled_flat_payload_witnesses
            .estimated_memory_bytes();
        assert!(compiled_bytes > 0);

        let active_bytes = 4096;
        let breakdown = mc.memory_breakdown(0, active_bytes);
        assert_eq!(
            breakdown.payload_witness_bytes,
            active_bytes.saturating_add(compiled_bytes)
        );
    }

    #[test]
    fn compiled_flat_duplicate_with_different_flat_payload_fails_closed() {
        let module = minimal_module();
        let config = minimal_config();
        let mut mc = compiled_flat_checker(&module, &config);
        let mut storage = FingerprintOnlyStorage::new(BulkStateStorage::empty(1), 1);
        let fp = Fingerprint(88);

        let first = storage
            .admit_successor(
                fp,
                ArrayState::from_values(vec![Value::int(7)]),
                None,
                None,
                0,
                &mut mc,
            )
            .unwrap();
        assert!(first.is_some());

        let err = match storage.admit_successor(
            fp,
            ArrayState::from_values(vec![Value::int(8)]),
            None,
            None,
            1,
            &mut mc,
        ) {
            Err(err) => err,
            Ok(_) => panic!("compiled-flat duplicate with mismatched payload must fail closed"),
        };
        match err {
            CheckResult::Error { error, .. } => {
                let rendered = error.to_string();
                // Part of #4451 follow-up: the prepared admission backend renders
                // the fail-closed collision with `reason_code=canonical_payload_mismatch`
                // and `payload_witness=compiled_flat_xxh3` for the flat-i64 domain.
                assert!(
                    rendered.contains("reason_code=canonical_payload_mismatch"),
                    "unexpected error: {rendered}"
                );
            }
            other => panic!("expected CheckResult::Error, got {other:?}"),
        }
    }
}
