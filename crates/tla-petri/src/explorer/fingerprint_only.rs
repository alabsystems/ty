// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Fingerprint-only BFS exploration for Petri nets.
//!
//! Uses a lock-free `CasFingerprintSet` from `tla-mc-core` for state
//! deduplication, admitting states through the shared collision-checked
//! fingerprint contract. The BFS queue still carries packed markings needed to
//! compute successors; an exact packed-marking guard authorizes duplicate
//! fingerprints and fails closed on collisions.
//!
//! For trace reconstruction on counterexample (e.g., when a deadlock
//! invariant is violated), a disk-based trace log records
//! `(fingerprint, parent_fingerprint, transition_id)` triples that can be
//! walked backward from the violating state to the initial state.
//!
//! Part of #3721.

use std::collections::VecDeque;
use std::io::{BufWriter, Write};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::time::Instant;

use rustc_hash::FxHashMap;
use tla_mc_core::{
    CasFingerprintSet, CheckerSourceKind, FingerprintAdmission, FingerprintSet, LookupOutcome,
    PreparedFingerprintAdmissionPlan, PreparedFingerprintPayloadWitnessKind,
    PreparedProgramPayloadKind, PreparedStorageKind, SetupTraceLaneKind, SharedDedupIdentity,
    SharedDedupScope, SharedDedupStorageKind, SharedDuplicateAuthorization,
    SharedFingerprintAlgorithm, SharedFingerprintIdentity, SharedFingerprintValueKind,
    StorageFault, ValidatedPreparedFingerprintAdmissionPlan,
};

use super::config::{ExplorationConfig, ExplorationObserver, ExplorationResult};
use super::fingerprint::fingerprint_marking;
use super::setup::ExplorationSetup;
use super::successors::{
    EnabledCarry, InterpretedSuccessorProvider, PetriSuccessorProvider, SuccessorVisit,
};
use crate::marking::unpack_marking_config;
use crate::petri_net::PetriNet;
use crate::stubborn::{DependencyGraph, PorStrategy};

const PETRI_MARKING_DEDUP_ID: &str = "state-space-dedup-v1";
const PETRI_MARKING_PREPARED_FINGERPRINT_ADMISSION_PLAN_ID: &str =
    "mcc_petri.marking_vector.fingerprint_only.prepared_fingerprint_admission.v1";
const PETRI_MARKING_FINGERPRINT_ID_U64_LOW: &str = "marking-low64-v1";
const PETRI_MARKING_CANONICALIZATION_VERSION_U64_LOW: &str = "place-token-marking-u64-low-v1";
const PETRI_MARKING_FINGERPRINT_NAMESPACE_U64_LOW: &str = "place-token-marking-low64";
const PETRI_MARKING_CANONICAL_DOMAIN: &str = "place-token-marking";
const PETRI_MARKING_CANONICAL_DOMAIN_VERSION: &str = "u64-vector-v1";
const PETRI_FINGERPRINT_ONLY_STORAGE_CONFIG: &str = "fingerprint-only-cas-fingerprint-set-v1";
const PETRI_FINGERPRINT_ONLY_ADMISSION_CALLSITE: &str =
    "tla_petri::explorer::fingerprint_only::PackedMarkingCollisionGuard::admit";
/// Bound unbounded MCC sentinel configs before sizing the resident CAS table.
/// This yields a 64M-slot table, about 512 MiB for the current AtomicU64 slots.
const FINGERPRINT_ONLY_MAX_CAPACITY_STATES: usize = 32 * 1024 * 1024;

/// Stack size for each parallel BFS worker thread (64 MiB).
///
/// Scoped worker threads (`spawn_scoped`) otherwise inherit the platform
/// default (~2 MiB on macOS/Linux), far smaller than the main thread's ~8 MiB.
/// On WIDE nets (hundreds of places / thousands of transitions, e.g.
/// NoC3x3-PT-8A: 317 places, 4293 transitions) the per-state successor /
/// guard-evaluation machinery recurses deep enough to overflow a 2 MiB worker
/// stack — aborting the WHOLE process with SIGABRT instead of declining
/// fail-closed. The work is net-size-bounded (a 64 MiB worker stack lets the
/// very same run finish and emit a clean `CANNOT_COMPUTE`), so we give workers
/// a generous stack that comfortably exceeds the main thread's. The reservation
/// is virtual (lazily backed), so 64 MiB × workers costs no real memory until
/// touched. Fail-closed: a worker must never abort the process.
const BFS_WORKER_STACK_BYTES: usize = 64 * 1024 * 1024;

/// Poll the process RSS for the memory-pressure guard once per this many
/// successor visits. Each visit may enqueue one packed marking; at ~512 visits
/// between polls the worst-case growth between checks is bounded (e.g. 512 ×
/// 128 KiB ≈ 64 MiB on the widest nets), well inside the headroom left by
/// [`crate::memory::EXPLORER_MEMORY_GUARD_FRACTION`]. An RSS query is a cheap
/// syscall (mach `task_info` / `/proc/self/statm`), so this stays off the hot
/// path while keeping memory-overshoot tightly bounded.
const MEMORY_PROBE_INTERVAL: u32 = 512;
const FINGERPRINT_ONLY_MIN_TABLE_CAPACITY: usize = 4096;

/// Statistics from fingerprint-only BFS exploration.
#[derive(Debug, Clone)]
pub(crate) struct FingerprintOnlyStats {
    /// Number of unique states visited.
    pub(crate) states_visited: usize,
    /// Maximum BFS depth reached.
    pub(crate) max_depth: usize,
    /// Memory bytes used by the CAS fingerprint set.
    pub(crate) fp_set_memory_bytes: usize,
    /// Estimated bytes used by resident packed markings for collision checks.
    pub(crate) collision_guard_memory_bytes: usize,
    /// Runtime attempts to consume the prepared fingerprint admission handle.
    pub(crate) admission_attempted: usize,
    /// Runtime admissions that inserted a new fingerprint.
    pub(crate) admission_new: usize,
    /// Runtime admissions that observed an authorized duplicate fingerprint.
    pub(crate) admission_duplicate: usize,
    /// Runtime admissions that failed closed on a storage/collision fault.
    pub(crate) admission_fault: usize,
}

/// Disk-based trace log entry for counterexample reconstruction.
///
/// Each entry is 20 bytes: `(state_fp: u64, parent_fp: u64, transition_id: u32)`.
/// Walk backward from the violating state's fingerprint to the initial state
/// (whose `parent_fp` is 0) to reconstruct the error trace.
#[repr(C, packed)]
#[derive(Debug, Clone, Copy)]
struct TraceEntry {
    state_fp: u64,
    parent_fp: u64,
    transition_id: u32,
}

/// Optional disk-based trace logger for counterexample reconstruction.
struct TraceLogger {
    writer: BufWriter<std::fs::File>,
}

impl TraceLogger {
    fn new(path: &std::path::Path) -> std::io::Result<Self> {
        let file = std::fs::File::create(path)?;
        Ok(Self {
            writer: BufWriter::new(file),
        })
    }

    fn log_entry(&mut self, state_fp: u64, parent_fp: u64, transition_id: u32) {
        // Write as raw bytes for compact, fast logging.
        let entry = TraceEntry {
            state_fp,
            parent_fp,
            transition_id,
        };
        let bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(
                (&entry as *const TraceEntry).cast::<u8>(),
                std::mem::size_of::<TraceEntry>(),
            )
        };
        // Best-effort: trace logging failures are non-fatal.
        let _ = self.writer.write_all(bytes);
    }

    fn flush(&mut self) {
        let _ = self.writer.flush();
    }
}

/// BFS entry carrying packed marking and its precomputed 64-bit fingerprint.
///
/// Trace data (parent fingerprint, transition ID) is logged immediately at
/// successor generation time by `TraceLogger`, so the queue entry only needs
/// the packed marking, fingerprint, and depth for BFS traversal.
struct QueueEntry {
    packed: Box<[u8]>,
    fp: u64,
    depth: usize,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct PreparedFingerprintAdmissionRuntimeCounters {
    pub(crate) attempted: usize,
    pub(crate) new: usize,
    pub(crate) duplicate: usize,
    pub(crate) fault: usize,
}

/// Arena-backed packed-marking residency map for collision-guard duplicate
/// authorization.
///
/// Per-marking heap allocations (the original `FxHashMap<u64, Box<[u8]>>`
/// shape) carried 24+ bytes of allocator overhead per entry on glibc/jemalloc
/// in addition to the hashmap node, which dominated steady-state memory on
/// large Philosophers-COL runs (audit R-2, ~76 GB observed). The arena layout
/// packs every resident marking byte into one contiguous `Vec<u8>` and stores
/// only `(offset, len)` slices in the hashmap. Membership semantics are
/// identical: equality is `arena[offset..offset+len] == candidate`.
///
/// `MarkingSlice::offset` is `u64` to allow arenas larger than 4 GiB
/// (FamilyReunion-COL at scale spills past u32 by ~5x), while `len` stays
/// `u32` — even the widest MCC marking (Petri net hard-cap ~16M places ×
/// 8-byte tokens) fits in ~128 MB. The struct is 16 bytes vs the original
/// 24-byte `Box<[u8]>` header + 16-byte malloc overhead, while leaving the
/// guard sound for arenas up to 16 EiB.
#[derive(Debug, Clone, Copy)]
struct MarkingSlice {
    offset: u64,
    len: u32,
}

#[derive(Debug, Default)]
struct ResidentMarkingArena {
    arena: Vec<u8>,
    index: FxHashMap<u64, MarkingSlice>,
}

impl ResidentMarkingArena {
    fn lookup<'a>(&'a self, fingerprint: u64) -> Option<&'a [u8]> {
        self.index.get(&fingerprint).map(|slice| {
            let start = slice.offset as usize;
            let end = start + slice.len as usize;
            &self.arena[start..end]
        })
    }

    fn insert(&mut self, fingerprint: u64, packed: &[u8]) -> Option<MarkingSlice> {
        // Per-marking length is bounded by the Petri net's packed-marking
        // byte budget. `u32::MAX` (4 GiB) is well above any plausible MCC
        // marking and gives us a hard ceiling that the caller can detect
        // (insert returns None) without entering a corrupting overflow.
        let offset = self.arena.len() as u64;
        let len = packed.len();
        if len > u32::MAX as usize {
            return None;
        }
        self.arena.extend_from_slice(packed);
        let slice = MarkingSlice {
            offset,
            len: len as u32,
        };
        self.index.insert(fingerprint, slice)
    }

    fn memory_bytes(&self) -> usize {
        let index_bytes = self
            .index
            .capacity()
            .saturating_mul(std::mem::size_of::<u64>() + std::mem::size_of::<MarkingSlice>());
        let arena_bytes = self.arena.capacity();
        index_bytes.saturating_add(arena_bytes)
    }

    fn entries(&self) -> usize {
        self.index.len()
    }
}

struct PackedMarkingCollisionGuard {
    admission_handle: ValidatedPreparedFingerprintAdmissionPlan,
    admission_counters: PreparedFingerprintAdmissionRuntimeCounters,
    residents: ResidentMarkingArena,
}

impl PackedMarkingCollisionGuard {
    fn new(admission_plan: PreparedFingerprintAdmissionPlan) -> Self {
        let admission_handle = admission_plan
            .into_validated_runtime_handle()
            .expect("fingerprint-only prepared admission plan must validate at setup");
        Self {
            admission_handle,
            admission_counters: PreparedFingerprintAdmissionRuntimeCounters::default(),
            residents: ResidentMarkingArena::default(),
        }
    }

    fn admission_plan(&self) -> &PreparedFingerprintAdmissionPlan {
        self.admission_handle.plan()
    }

    fn admission_counters(&self) -> PreparedFingerprintAdmissionRuntimeCounters {
        self.admission_counters
    }

    fn admit(
        &mut self,
        fp_set: &impl FingerprintSet<u64>,
        fingerprint: u64,
        packed: &[u8],
    ) -> Result<FingerprintAdmission, StorageFault> {
        self.admission_counters.attempted += 1;
        let admission_result = {
            let residents = &self.residents;
            self.admission_handle
                .admit_fingerprint_with_canonical_payload_comparison(fp_set, fingerprint, || {
                    Ok(residents
                        .lookup(fingerprint)
                        .is_some_and(|resident| resident == packed))
                })
        };
        match &admission_result {
            Ok(admission) if admission.is_new() => self.admission_counters.new += 1,
            Ok(admission) if admission.is_duplicate() => self.admission_counters.duplicate += 1,
            Ok(_) => {}
            Err(_) => self.admission_counters.fault += 1,
        }
        let admission = admission_result?;

        if admission.is_new() {
            let previous = self.residents.insert(fingerprint, packed);
            debug_assert!(
                previous.is_none(),
                "new fingerprint admission should not replace a resident marking"
            );
        }
        Ok(admission)
    }

    fn memory_bytes(&self) -> usize {
        self.residents.memory_bytes()
    }
}

fn fingerprint_only_marking_dedup_identity() -> SharedDedupIdentity {
    let fingerprint = SharedFingerprintIdentity::new(
        PETRI_MARKING_FINGERPRINT_ID_U64_LOW,
        SharedFingerprintAlgorithm::CanonicalBytesSha256,
        SharedFingerprintValueKind::MarkingVector,
        PETRI_MARKING_CANONICALIZATION_VERSION_U64_LOW,
        PETRI_MARKING_FINGERPRINT_NAMESPACE_U64_LOW,
        64,
    )
    .with_canonical_domain(
        PETRI_MARKING_CANONICAL_DOMAIN,
        PETRI_MARKING_CANONICAL_DOMAIN_VERSION,
    );
    SharedDedupIdentity::new(
        PETRI_MARKING_DEDUP_ID,
        fingerprint,
        SharedDedupScope::StateSpace,
        SharedDedupStorageKind::Cas,
        SetupTraceLaneKind::Fingerprint,
    )
    .with_storage_config_identity(PETRI_FINGERPRINT_ONLY_STORAGE_CONFIG)
}

fn fingerprint_only_marking_admission_plan() -> PreparedFingerprintAdmissionPlan {
    let dedup_identity = fingerprint_only_marking_dedup_identity();
    PreparedFingerprintAdmissionPlan::new(
        PETRI_MARKING_PREPARED_FINGERPRINT_ADMISSION_PLAN_ID,
        CheckerSourceKind::MccPetri,
        PreparedProgramPayloadKind::MccPetri,
        PreparedStorageKind::PetriMarking,
        SetupTraceLaneKind::Fingerprint,
        dedup_identity,
        SharedDuplicateAuthorization::CanonicalPayloadEquality,
        PreparedFingerprintPayloadWitnessKind::PetriMarkingCas,
    )
}

fn fingerprint_only_capacity_states(config: &ExplorationConfig) -> usize {
    let requested = config
        .storage_primary_capacity
        .unwrap_or(config.max_states())
        .max(1);
    let capacity_states = requested.min(FINGERPRINT_ONLY_MAX_CAPACITY_STATES);
    if requested > capacity_states {
        eprintln!(
            "fingerprint-only CAS resident target capped from {requested} to {capacity_states} states"
        );
    }
    capacity_states
}

fn fingerprint_only_table_capacity(config: &ExplorationConfig) -> usize {
    fingerprint_only_capacity_states(config)
        .saturating_mul(2)
        .max(FINGERPRINT_ONLY_MIN_TABLE_CAPACITY)
}

fn fingerprint_only_stats_from_memory(
    fp_set_memory_bytes: usize,
    collision_guard: &PackedMarkingCollisionGuard,
    states_visited: usize,
    max_depth: usize,
) -> FingerprintOnlyStats {
    FingerprintOnlyStats {
        states_visited,
        max_depth,
        fp_set_memory_bytes,
        collision_guard_memory_bytes: collision_guard.memory_bytes(),
        admission_attempted: collision_guard.admission_counters.attempted,
        admission_new: collision_guard.admission_counters.new,
        admission_duplicate: collision_guard.admission_counters.duplicate,
        admission_fault: collision_guard.admission_counters.fault,
    }
}

fn fingerprint_only_stats(
    fp_set: &CasFingerprintSet,
    collision_guard: &PackedMarkingCollisionGuard,
    states_visited: usize,
    max_depth: usize,
) -> FingerprintOnlyStats {
    fingerprint_only_stats_from_memory(
        FingerprintSet::stats(fp_set).memory_bytes,
        collision_guard,
        states_visited,
        max_depth,
    )
}

fn record_fingerprint_only_admission_runtime_consumption(
    collision_guard: &PackedMarkingCollisionGuard,
) {
    let admission_counters = collision_guard.admission_counters();
    crate::mcc_backend_evidence::record_mcc_prepared_fingerprint_admission_runtime_consumption(
        crate::mcc_backend_evidence::MccPreparedFingerprintAdmissionRuntimeConsumption::from_plan(
            collision_guard.admission_plan(),
            PETRI_FINGERPRINT_ONLY_ADMISSION_CALLSITE,
            admission_counters.attempted,
            admission_counters.new,
            admission_counters.duplicate,
            admission_counters.fault,
        ),
    );
}

fn finish_fingerprint_only(
    fp_set: &CasFingerprintSet,
    collision_guard: &PackedMarkingCollisionGuard,
    states_visited: usize,
    max_depth: usize,
    result: ExplorationResult,
) -> (ExplorationResult, FingerprintOnlyStats) {
    let stats = fingerprint_only_stats(fp_set, collision_guard, states_visited, max_depth);
    record_fingerprint_only_admission_runtime_consumption(collision_guard);
    (result, stats)
}

fn finish_fingerprint_only_without_fp_set(
    collision_guard: &PackedMarkingCollisionGuard,
    states_visited: usize,
    max_depth: usize,
    result: ExplorationResult,
) -> (ExplorationResult, FingerprintOnlyStats) {
    let stats = fingerprint_only_stats_from_memory(0, collision_guard, states_visited, max_depth);
    record_fingerprint_only_admission_runtime_consumption(collision_guard);
    (result, stats)
}

/// Fingerprint-only BFS exploration of a Petri net state space.
///
/// Uses a lock-free `CasFingerprintSet` for fingerprint admission and an exact
/// packed-marking guard for duplicate authorization. The BFS queue still
/// carries packed markings for successor computation.
///
/// # Arguments
///
/// * `net` - The Petri net to explore.
/// * `config` - Exploration configuration (max_states, deadline, POR).
/// * `observer` - Observer for state/transition/deadlock callbacks.
/// * `trace_dir` - Optional directory for disk-based trace logging.
///
/// # Returns
///
/// `(ExplorationResult, FingerprintOnlyStats)` with exploration outcome and
/// memory usage statistics.
pub(crate) fn explore_fingerprint_only(
    net: &PetriNet,
    config: &ExplorationConfig,
    observer: &mut dyn ExplorationObserver,
    trace_dir: Option<&std::path::Path>,
) -> (ExplorationResult, FingerprintOnlyStats) {
    let ExplorationSetup {
        marking_config,
        pack_capacity,
        num_places,
        num_transitions,
        initial_packed,
    } = ExplorationSetup::analyze(net);

    let dep_graph = match &config.por_strategy {
        PorStrategy::None => None,
        _ => Some(DependencyGraph::build(net)),
    };

    let mut collision_guard =
        PackedMarkingCollisionGuard::new(fingerprint_only_marking_admission_plan());
    // Size the CAS table to ~2x the expected resident primary tier. Unbounded
    // MCC configs use usize::MAX as a sentinel, so allocation sizing must be
    // capped before multiplying or rounding up in CasFingerprintSet.
    let table_capacity = fingerprint_only_table_capacity(config);
    let fp_set = match CasFingerprintSet::try_new(table_capacity) {
        Ok(set) => set,
        Err(err) => {
            eprintln!(
                "fingerprint-only CAS table allocation failed ({err}); \
                 exploration incomplete; caller will emit CANNOT_COMPUTE"
            );
            return finish_fingerprint_only_without_fp_set(
                &collision_guard,
                0,
                0,
                ExplorationResult::new(false, 0, false),
            );
        }
    };

    let mut trace_logger = trace_dir.and_then(|dir| {
        std::fs::create_dir_all(dir).ok()?;
        let path = dir.join("trace.bin");
        TraceLogger::new(&path).ok()
    });

    // Each queue entry optionally carries the full enabled bitmap of its
    // marking (the parent carry for the O(Δ) incremental enabled-set update on
    // the next BFS step). Sequential-only wrapper so the shared `QueueEntry`
    // (also used by the parallel path) is unchanged.
    let mut queue: VecDeque<(QueueEntry, Option<EnabledCarry>)> = VecDeque::new();
    let mut max_depth: usize = 0;

    // Compute initial fingerprint (truncated to u64 for CAS table).
    let initial_fp128 = fingerprint_marking(&initial_packed);
    let initial_fp = initial_fp128 as u64;

    // Admit initial state.
    let initial_admission = match collision_guard.admit(&fp_set, initial_fp, &initial_packed) {
        Ok(admission) => admission,
        Err(_) => {
            return finish_fingerprint_only(
                &fp_set,
                &collision_guard,
                0,
                0,
                ExplorationResult::new(false, 0, false),
            );
        }
    };
    let mut successor_provider = InterpretedSuccessorProvider::new(
        net,
        &marking_config,
        pack_capacity,
        num_transitions,
        dep_graph.as_ref(),
        &config.por_strategy,
        None,
    );
    // Incremental enabled-set path: seed the BFS root with a one-time full-scan
    // enabled bitmap when admissible (no POR, no canonicalizer here, kill-switch
    // off). Every subsequent state's carry is then produced by the O(Δ)
    // incremental update keyed off its parent. When inadmissible, carries stay
    // `None` and the provider full-scans each state (unchanged behaviour).
    let incremental_enabled = successor_provider.incremental_enabled_admissible();

    if initial_admission.is_new() {
        if !observer.on_new_state(&net.initial_marking) {
            return finish_fingerprint_only(
                &fp_set,
                &collision_guard,
                1,
                0,
                ExplorationResult::new(false, 1, true),
            );
        }

        if let Some(ref mut logger) = trace_logger {
            logger.log_entry(initial_fp, 0, 0);
        }

        let initial_carry: Option<EnabledCarry> = if incremental_enabled {
            Some(EnabledCarry::from(
                successor_provider
                    .full_enabled_bitmap(&net.initial_marking)
                    .into_boxed_slice(),
            ))
        } else {
            None
        };
        queue.push_back((
            QueueEntry {
                packed: initial_packed,
                fp: initial_fp,
                depth: 0,
            },
            initial_carry,
        ));
    }

    let mut stopped_by_observer = false;
    let mut current_tokens = Vec::with_capacity(num_places);
    // One adaptive probe (deadline + memory): the frontier queue and the
    // collision-guard resident arena each retain a full packed marking per
    // state, so on wide nets RAM — not the deadline — binds first. Ticked per
    // pop AND per successor.
    let mut probe = crate::memory::explorer_probe(config.deadline());

    while let Some((entry, parent_carry)) = queue.pop_front() {
        if probe.over_budget() {
            let visited = FingerprintSet::len(&fp_set);
            return finish_fingerprint_only(
                &fp_set,
                &collision_guard,
                visited,
                max_depth,
                ExplorationResult::new(false, visited, false),
            );
        }

        if observer.is_done() {
            stopped_by_observer = true;
            break;
        }

        max_depth = max_depth.max(entry.depth);

        unpack_marking_config(&entry.packed, &marking_config, &mut current_tokens);

        let mut early_result = None;
        let parent_enabled: Option<&[bool]> = parent_carry.as_deref();
        successor_provider.for_each_successor_with_enabled(
            &mut current_tokens,
            parent_enabled,
            &mut |successor, child_enabled| {
                // Per-successor tick bounds the byte-overshoot window within one
                // wide expansion (the per-pop tick cannot). Same adaptive probe.
                if probe.over_budget() {
                    let visited = FingerprintSet::len(&fp_set);
                    early_result = Some((visited, ExplorationResult::new(false, visited, false)));
                    return SuccessorVisit::Stop;
                }

                if !observer.on_transition_fire(successor.transition) {
                    stopped_by_observer = true;
                    return SuccessorVisit::Stop;
                }

                let succ_fp = successor.fingerprint as u64;
                if FingerprintSet::len(&fp_set) >= config.max_states() {
                    let stop_at_limit = match fp_set.contains_checked(succ_fp) {
                        LookupOutcome::Present => false,
                        LookupOutcome::Absent | LookupOutcome::StorageFault(_) => true,
                        _ => true,
                    };
                    if stop_at_limit {
                        let visited = FingerprintSet::len(&fp_set);
                        early_result =
                            Some((visited, ExplorationResult::new(false, visited, false)));
                        return SuccessorVisit::Stop;
                    }
                }

                let admission = match collision_guard.admit(&fp_set, succ_fp, successor.packed) {
                    Ok(admission) => admission,
                    Err(_) => {
                        // Storage fault -- treat as state limit reached.
                        let visited = FingerprintSet::len(&fp_set);
                        early_result =
                            Some((visited, ExplorationResult::new(false, visited, false)));
                        return SuccessorVisit::Stop;
                    }
                };
                if admission.is_duplicate() {
                    return SuccessorVisit::Continue;
                }

                if !observer.on_new_state(successor.marking) {
                    stopped_by_observer = true;
                    return SuccessorVisit::Stop;
                }

                if let Some(ref mut logger) = trace_logger {
                    logger.log_entry(succ_fp, entry.fp, successor.transition.0);
                }

                let succ_packed: Box<[u8]> = successor.packed.into();
                // Carry the child's incrementally-derived full enabled bitmap so the
                // next BFS step skips the O(T) scan. Only retained when the
                // incremental path is active (otherwise `child_enabled` is a
                // full-scan bitmap we don't need to keep).
                let succ_carry: Option<EnabledCarry> = if incremental_enabled {
                    Some(EnabledCarry::from(child_enabled))
                } else {
                    None
                };
                queue.push_back((
                    QueueEntry {
                        packed: succ_packed,
                        fp: succ_fp,
                        depth: entry.depth + 1,
                    },
                    succ_carry,
                ));

                SuccessorVisit::Continue
            },
        );

        // Fail-closed (#22): a token-count overflow aborted successor
        // enumeration — report the exploration incomplete (CANNOT_COMPUTE)
        // rather than a complete-but-wrong fingerprint frontier.
        if successor_provider.token_overflow_declined() {
            let visited = FingerprintSet::len(&fp_set);
            return finish_fingerprint_only(
                &fp_set,
                &collision_guard,
                visited,
                max_depth,
                ExplorationResult::new(false, visited, false),
            );
        }

        if let Some((visited, result)) = early_result {
            return finish_fingerprint_only(&fp_set, &collision_guard, visited, max_depth, result);
        }

        if stopped_by_observer {
            break;
        }

        if !successor_provider.has_enabled_successors() {
            observer.on_deadlock(&current_tokens);
            if observer.is_done() {
                stopped_by_observer = true;
                break;
            }
        }
    }

    if let Some(ref mut logger) = trace_logger {
        logger.flush();
    }

    let visited = FingerprintSet::len(&fp_set);
    finish_fingerprint_only(
        &fp_set,
        &collision_guard,
        visited,
        max_depth,
        ExplorationResult::new(
            !stopped_by_observer && queue.is_empty(),
            visited,
            stopped_by_observer,
        ),
    )
}

/// Returns the recommended collision-guard shard count for a parallel
/// fingerprint-only run.
///
/// Sharding amortizes mutex contention on the resident-marking arena across
/// `4 * workers` independent locks. The fingerprint-routing is by
/// `fingerprint % shards`, so doubling the shard count halves expected
/// per-shard contention. We cap at 256 to keep the per-shard hashmap large
/// enough to amortize per-shard memory headers; below that, contention is
/// the dominant cost.
fn collision_guard_shard_count(workers: usize) -> usize {
    (workers.max(1) * 4).next_power_of_two().min(256)
}

/// Per-shard collision guard wrapper for the parallel BFS path.
struct ParallelCollisionShard {
    inner: RwLock<PackedMarkingCollisionGuard>,
}

impl ParallelCollisionShard {
    fn new() -> Self {
        Self {
            inner: RwLock::new(PackedMarkingCollisionGuard::new(
                fingerprint_only_marking_admission_plan(),
            )),
        }
    }
}

/// Frontier-parallel BFS exploration of a Petri net state space using the
/// fingerprint-only storage layout.
///
/// Architecture (R-1 from the perf audit):
/// - **Shared** `Arc<CasFingerprintSet>` is the atomic dedup oracle. Workers
///   race on `insert_checked`; the unique winner is the discoverer of a
///   fingerprint, the others observe `AlreadyPresent` and skip.
/// - **Sharded** collision guards (one `Mutex<PackedMarkingCollisionGuard>`
///   per `4 * workers` shards) hold resident packed markings for duplicate
///   authorization. Fingerprint routes to one shard, so contention is
///   bounded by hot fingerprints alone. The shard lock spans the
///   `admit` call to keep "insert resident marking" atomic with the CAS
///   admission decision (otherwise two workers could read the resident map
///   between the other's CAS-success and resident-insert, mis-authorizing a
///   duplicate as a collision).
/// - **Frontier** is processed level-by-level. Each level becomes a `Vec<_>`
///   chunked across workers; each worker accumulates its own next-frontier
///   contribution and merges it after all workers drain the current level.
///   This avoids the harder termination protocol of a single shared deque
///   and keeps observer/queue mutation off the hot path.
/// - **Observer** lives behind a `Mutex` shared across workers. Observer
///   methods are only called on transition firings, new-state discoveries,
///   and deadlocks — the hot per-successor work (unpack/fire/fingerprint)
///   is lock-free.
///
/// Soundness invariant — first-seen wins: a fingerprint is "new" iff it
/// transitions the CAS table from absent to present atomically. Only the
/// admitting worker calls `observer.on_new_state` and enqueues the marking.
/// Workers observing `Duplicate` skip both. This preserves the exact state
/// count and observer event multiset of the sequential
/// [`explore_fingerprint_only`] path.
#[allow(clippy::too_many_lines)]
fn explore_fingerprint_only_parallel<O>(
    net: &PetriNet,
    config: &ExplorationConfig,
    observer: &mut O,
    trace_dir: Option<&std::path::Path>,
    workers: usize,
) -> (ExplorationResult, FingerprintOnlyStats)
where
    O: ExplorationObserver + Send,
{
    debug_assert!(workers >= 2, "parallel path requires at least 2 workers");
    let ExplorationSetup {
        marking_config,
        pack_capacity,
        num_transitions,
        initial_packed,
        ..
    } = ExplorationSetup::analyze(net);

    let dep_graph: Option<Arc<DependencyGraph>> = match &config.por_strategy {
        PorStrategy::None => None,
        _ => Some(Arc::new(DependencyGraph::build(net))),
    };

    let table_capacity = fingerprint_only_table_capacity(config);
    let fp_set = match CasFingerprintSet::try_new(table_capacity) {
        Ok(set) => Arc::new(set),
        Err(err) => {
            eprintln!(
                "fingerprint-only CAS table allocation failed ({err}); \
                 exploration incomplete; caller will emit CANNOT_COMPUTE"
            );
            let empty_guard =
                PackedMarkingCollisionGuard::new(fingerprint_only_marking_admission_plan());
            return finish_fingerprint_only_without_fp_set(
                &empty_guard,
                0,
                0,
                ExplorationResult::new(false, 0, false),
            );
        }
    };

    let shard_count = collision_guard_shard_count(workers);
    let shards: Arc<Vec<ParallelCollisionShard>> = Arc::new(
        (0..shard_count)
            .map(|_| ParallelCollisionShard::new())
            .collect(),
    );

    let trace_logger = trace_dir.and_then(|dir| {
        std::fs::create_dir_all(dir).ok()?;
        let path = dir.join("trace.bin");
        TraceLogger::new(&path).ok()
    });
    let trace_logger: Option<Arc<Mutex<TraceLogger>>> =
        trace_logger.map(|t| Arc::new(Mutex::new(t)));

    // Admit the initial state through shard 0 so the resident-marking
    // contract holds even before workers start.
    let initial_fp128 = fingerprint_marking(&initial_packed);
    let initial_fp = initial_fp128 as u64;
    let initial_shard_idx = (initial_fp as usize) % shard_count;
    let initial_admission = {
        let mut guard = shards[initial_shard_idx]
            .inner
            .write()
            .expect("collision shard mutex poisoned");
        guard.admit(fp_set.as_ref(), initial_fp, &initial_packed)
    };
    let initial_admission = match initial_admission {
        Ok(adm) => adm,
        Err(_) => {
            let drained = drain_shards(&shards);
            return finish_fingerprint_only_parallel(
                fp_set.as_ref(),
                &drained,
                0,
                0,
                ExplorationResult::new(false, 0, false),
            );
        }
    };

    let mut current_frontier: Vec<QueueEntry> = Vec::new();
    if initial_admission.is_new() {
        if !observer.on_new_state(&net.initial_marking) {
            let drained = drain_shards(&shards);
            return finish_fingerprint_only_parallel(
                fp_set.as_ref(),
                &drained,
                1,
                0,
                ExplorationResult::new(false, 1, true),
            );
        }
        if let Some(logger) = trace_logger.as_ref() {
            let mut l = logger.lock().expect("trace logger mutex poisoned");
            l.log_entry(initial_fp, 0, 0);
        }
        current_frontier.push(QueueEntry {
            packed: initial_packed,
            fp: initial_fp,
            depth: 0,
        });
    }

    let mut stopped_by_observer = false;
    let mut state_limit_reached = false;
    let mut deadline_reached = false;
    let mut memory_reached = false;
    let mut max_depth: usize = 0;
    let max_states_for_limit = config.max_states();
    // Derive the explorer memory budget ONCE, before the level loop: the
    // self-footprint ceiling must be pinned to the memory available at the
    // START of the exploration (like the former one-shot `mem_ceiling`), not
    // re-derived each level from the free memory this very exploration has
    // already consumed — that would shrink the ceiling as the run grows and
    // trip prematurely (spurious CANNOT_COMPUTE). Workers clone it per level.
    let explorer_budget = crate::memory::explorer_budget();

    let observer_mutex: Arc<Mutex<&mut O>> = Arc::new(Mutex::new(observer));

    while !current_frontier.is_empty()
        && !stopped_by_observer
        && !state_limit_reached
        && !deadline_reached
        && !memory_reached
    {
        max_depth = max_depth.max(current_frontier.last().map(|e| e.depth).unwrap_or(0));

        // Check deadline at level boundary.
        if config.deadline().is_some_and(|d| Instant::now() >= d) {
            deadline_reached = true;
            break;
        }

        // Check is_done at level boundary while we still hold the observer
        // mutex exclusively (no workers in flight).
        {
            let obs = observer_mutex.lock().expect("observer mutex poisoned");
            if obs.is_done() {
                stopped_by_observer = true;
                break;
            }
        }

        // Distribute the level across workers. Each worker pulls
        // `frontier_cursor` chunks until exhausted. Per-worker next-frontier
        // contributions are collected lock-free, then concatenated.
        let frontier = Arc::new(current_frontier);
        let frontier_cursor = Arc::new(AtomicUsize::new(0));
        let chunk_size = ((frontier.len() / workers).max(1)).min(64);
        let stop_flag = Arc::new(AtomicBool::new(false));
        let state_limit_flag = Arc::new(AtomicBool::new(false));
        let fault_flag = Arc::new(AtomicBool::new(false));
        let deadline_flag = Arc::new(AtomicBool::new(false));
        let memory_flag = Arc::new(AtomicBool::new(false));
        let deadline_for_workers = config.deadline();

        let next_frontier = std::thread::scope(|scope| -> Vec<QueueEntry> {
            let mut handles: Vec<std::thread::ScopedJoinHandle<'_, Vec<QueueEntry>>> =
                Vec::with_capacity(workers);
            for _ in 0..workers {
                let frontier = Arc::clone(&frontier);
                let cursor = Arc::clone(&frontier_cursor);
                let shards = Arc::clone(&shards);
                let fp_set = Arc::clone(&fp_set);
                let stop_flag = Arc::clone(&stop_flag);
                let state_limit_flag = Arc::clone(&state_limit_flag);
                let fault_flag = Arc::clone(&fault_flag);
                let deadline_flag = Arc::clone(&deadline_flag);
                let memory_flag = Arc::clone(&memory_flag);
                let observer_mutex = Arc::clone(&observer_mutex);
                let trace_logger = trace_logger.as_ref().map(Arc::clone);
                let marking_config = marking_config.clone();
                let dep_graph_clone = dep_graph.as_ref().map(Arc::clone);
                let por_strategy = config.por_strategy.clone();
                let net_clone = net.clone();
                let pack_capacity_local = pack_capacity;
                let num_transitions_local = num_transitions;
                let max_states_local = max_states_for_limit;
                let deadline_local = deadline_for_workers;
                // Clone the ONCE-derived budget so every worker's ceiling is
                // pinned to start-of-exploration availability (see explorer_budget).
                let budget_local = explorer_budget.clone();

                let handle =
                    std::thread::Builder::new()
                        .stack_size(BFS_WORKER_STACK_BYTES)
                        .spawn_scoped(scope, move || -> Vec<QueueEntry> {
                            let mut local_next: Vec<QueueEntry> = Vec::new();
                            let mut current_tokens: Vec<u64> =
                                Vec::with_capacity(net_clone.num_places());
                            let dep_graph_ref: Option<&DependencyGraph> =
                                dep_graph_clone.as_ref().map(|a| a.as_ref());
                            let mut successor_provider = InterpretedSuccessorProvider::new(
                                &net_clone,
                                &marking_config,
                                pack_capacity_local,
                                num_transitions_local,
                                dep_graph_ref,
                                &por_strategy,
                                None,
                            );

                            // Per-worker adaptive probe for BOTH the deadline and the
                            // memory budget, built from the ONCE-derived budget (stable
                            // ceiling). Each worker owns its own probe; the process
                            // footprint it reads is a whole-process signal, so the
                            // workers coordinate implicitly. A trip sets the matching
                            // shared flag so ALL workers stop and the level loop finishes
                            // incomplete (CANNOT_COMPUTE), never a verdict. Ticked per
                            // chunk (the probe self-amortizes the clock/syscall further).
                            let mut probe =
                                tla_resource::MemoryProbe::new(budget_local, deadline_local);

                            loop {
                                if stop_flag.load(Ordering::Acquire)
                                    || state_limit_flag.load(Ordering::Acquire)
                                    || fault_flag.load(Ordering::Acquire)
                                    || deadline_flag.load(Ordering::Acquire)
                                    || memory_flag.load(Ordering::Acquire)
                                {
                                    return local_next;
                                }

                                match probe.check() {
                                    Some(tla_resource::Trip::Deadline) => {
                                        deadline_flag.store(true, Ordering::Release);
                                        return local_next;
                                    }
                                    Some(tla_resource::Trip::Memory) => {
                                        memory_flag.store(true, Ordering::Release);
                                        return local_next;
                                    }
                                    None => {}
                                }

                                let start = cursor.fetch_add(chunk_size, Ordering::AcqRel);
                                if start >= frontier.len() {
                                    return local_next;
                                }
                                let end = (start + chunk_size).min(frontier.len());

                                for entry in &frontier[start..end] {
                                    if stop_flag.load(Ordering::Acquire)
                                        || state_limit_flag.load(Ordering::Acquire)
                                        || fault_flag.load(Ordering::Acquire)
                                        || deadline_flag.load(Ordering::Acquire)
                                        || memory_flag.load(Ordering::Acquire)
                                    {
                                        return local_next;
                                    }

                                    unpack_marking_config(
                                        &entry.packed,
                                        &marking_config,
                                        &mut current_tokens,
                                    );

                                    let mut stop_now = false;
                                    successor_provider.for_each_successor(
                                &mut current_tokens,
                                &mut |successor| {
                                    // Observer: transition firing. Holding the
                                    // mutex for a single virtual-call is fine —
                                    // this matches sequential ordering.
                                    let go = {
                                        let mut obs =
                                            observer_mutex.lock().expect("observer mutex poisoned");
                                        obs.on_transition_fire(successor.transition)
                                    };
                                    if !go {
                                        stop_flag.store(true, Ordering::Release);
                                        stop_now = true;
                                        return SuccessorVisit::Stop;
                                    }

                                    let succ_fp = successor.fingerprint as u64;
                                    let shard_idx = (succ_fp as usize) % shards.len();

                                    let admission = {
                                        let len_before = FingerprintSet::len(fp_set.as_ref());
                                        if len_before >= max_states_local {
                                            match fp_set.contains_checked(succ_fp) {
                                                LookupOutcome::Present => {
                                                    return SuccessorVisit::Continue
                                                }
                                                _ => {
                                                    state_limit_flag.store(true, Ordering::Release);
                                                    return SuccessorVisit::Stop;
                                                }
                                            }
                                        }

                                        // Lock-free fast path for duplicates using read lock
                                        if fp_set.contains_checked(succ_fp)
                                            == LookupOutcome::Present
                                        {
                                            let guard = shards[shard_idx]
                                                .inner
                                                .read()
                                                .expect("collision shard rwlock poisoned");
                                            let is_match = guard
                                                .residents
                                                .lookup(succ_fp)
                                                .is_some_and(|r| r == successor.packed);
                                            if is_match {
                                                Ok(tla_mc_core::FingerprintAdmission::Duplicate)
                                            } else {
                                                drop(guard);
                                                let mut write_guard = shards[shard_idx]
                                                    .inner
                                                    .write()
                                                    .expect("collision shard rwlock poisoned");
                                                write_guard.admit(
                                                    fp_set.as_ref(),
                                                    succ_fp,
                                                    successor.packed,
                                                )
                                            }
                                        } else {
                                            let mut write_guard = shards[shard_idx]
                                                .inner
                                                .write()
                                                .expect("collision shard rwlock poisoned");
                                            write_guard.admit(
                                                fp_set.as_ref(),
                                                succ_fp,
                                                successor.packed,
                                            )
                                        }
                                    };

                                    let admission = match admission {
                                        Ok(a) => a,
                                        Err(_) => {
                                            fault_flag.store(true, Ordering::Release);
                                            return SuccessorVisit::Stop;
                                        }
                                    };

                                    if admission.is_duplicate() {
                                        return SuccessorVisit::Continue;
                                    }

                                    let go = {
                                        let mut obs =
                                            observer_mutex.lock().expect("observer mutex poisoned");
                                        obs.on_new_state(successor.marking)
                                    };
                                    if !go {
                                        stop_flag.store(true, Ordering::Release);
                                        stop_now = true;
                                        return SuccessorVisit::Stop;
                                    }

                                    if let Some(logger) = trace_logger.as_ref() {
                                        let mut l =
                                            logger.lock().expect("trace logger mutex poisoned");
                                        l.log_entry(succ_fp, entry.fp, successor.transition.0);
                                    }

                                    let packed: Box<[u8]> = successor.packed.into();
                                    local_next.push(QueueEntry {
                                        packed,
                                        fp: succ_fp,
                                        depth: entry.depth + 1,
                                    });
                                    SuccessorVisit::Continue
                                },
                            );

                                    if stop_now {
                                        return local_next;
                                    }

                                    if !successor_provider.has_enabled_successors() {
                                        let go = {
                                            let mut obs = observer_mutex
                                                .lock()
                                                .expect("observer mutex poisoned");
                                            obs.on_deadlock(&current_tokens);
                                            !obs.is_done()
                                        };
                                        if !go {
                                            stop_flag.store(true, Ordering::Release);
                                            return local_next;
                                        }
                                    }
                                }
                            }
                        })
                        .expect("spawn FingerprintOnly BFS worker thread");
                handles.push(handle);
            }

            let mut next_frontier: Vec<QueueEntry> = Vec::new();
            for handle in handles {
                match handle.join() {
                    Ok(local) => next_frontier.extend(local),
                    Err(_) => {
                        // Worker panic — mark as faulted and degrade.
                        fault_flag.store(true, Ordering::Release);
                    }
                }
            }
            next_frontier
        });

        if stop_flag.load(Ordering::Acquire) {
            stopped_by_observer = true;
        }
        if state_limit_flag.load(Ordering::Acquire) {
            state_limit_reached = true;
        }
        if deadline_flag.load(Ordering::Acquire) {
            deadline_reached = true;
        }
        if memory_flag.load(Ordering::Acquire) {
            memory_reached = true;
        }
        if fault_flag.load(Ordering::Acquire) {
            // Bubble fault up as incomplete run.
            let drained = drain_shards(&shards);
            let visited = FingerprintSet::len(fp_set.as_ref());
            return finish_fingerprint_only_parallel(
                fp_set.as_ref(),
                &drained,
                visited,
                max_depth,
                ExplorationResult::new(false, visited, false),
            );
        }

        current_frontier = next_frontier;
    }

    if let Some(logger) = trace_logger.as_ref() {
        let mut l = logger.lock().expect("trace logger mutex poisoned");
        l.flush();
    }

    let drained = drain_shards(&shards);
    let visited = FingerprintSet::len(fp_set.as_ref());
    let completed = !stopped_by_observer
        && !state_limit_reached
        && !deadline_reached
        && !memory_reached
        && current_frontier.is_empty();
    finish_fingerprint_only_parallel(
        fp_set.as_ref(),
        &drained,
        visited,
        max_depth,
        ExplorationResult::new(completed, visited, stopped_by_observer),
    )
}

/// Aggregate counter+memory snapshot from every per-shard collision guard
/// at the end of a parallel run.
///
/// Critically, this does NOT copy marking bytes from the shards into a
/// single aggregate arena: doing so doubled steady-state memory at drain
/// time (on FamilyReunion-COL-L00010 with 5.8M states, that meant a 64 GB
/// over-allocation, triggering CANNOT_COMPUTE). The caller observes only
/// the per-counter sums and a synthesized `memory_bytes` total, so a
/// snapshot suffices.
struct ParallelGuardSnapshot {
    plan: PreparedFingerprintAdmissionPlan,
    counters: PreparedFingerprintAdmissionRuntimeCounters,
    memory_bytes: usize,
}

impl ParallelGuardSnapshot {
    fn admission_plan(&self) -> &PreparedFingerprintAdmissionPlan {
        &self.plan
    }

    fn admission_counters(&self) -> PreparedFingerprintAdmissionRuntimeCounters {
        self.counters
    }

    fn memory_bytes(&self) -> usize {
        self.memory_bytes
    }
}

fn drain_shards(shards: &Arc<Vec<ParallelCollisionShard>>) -> ParallelGuardSnapshot {
    let mut counters = PreparedFingerprintAdmissionRuntimeCounters::default();
    let mut memory_bytes: usize = 0;
    for shard in shards.iter() {
        let g = shard.inner.write().expect("collision shard mutex poisoned");
        counters.attempted += g.admission_counters.attempted;
        counters.new += g.admission_counters.new;
        counters.duplicate += g.admission_counters.duplicate;
        counters.fault += g.admission_counters.fault;
        memory_bytes = memory_bytes.saturating_add(g.memory_bytes());
    }
    ParallelGuardSnapshot {
        plan: fingerprint_only_marking_admission_plan(),
        counters,
        memory_bytes,
    }
}

/// Variant of [`finish_fingerprint_only`] for the parallel path that takes a
/// drained-shards snapshot instead of a single guard. Mirrors the evidence
/// reporting contract.
fn finish_fingerprint_only_parallel(
    fp_set: &CasFingerprintSet,
    snapshot: &ParallelGuardSnapshot,
    states_visited: usize,
    max_depth: usize,
    result: ExplorationResult,
) -> (ExplorationResult, FingerprintOnlyStats) {
    let stats = FingerprintOnlyStats {
        states_visited,
        max_depth,
        fp_set_memory_bytes: FingerprintSet::stats(fp_set).memory_bytes,
        collision_guard_memory_bytes: snapshot.memory_bytes(),
        admission_attempted: snapshot.counters.attempted,
        admission_new: snapshot.counters.new,
        admission_duplicate: snapshot.counters.duplicate,
        admission_fault: snapshot.counters.fault,
    };
    let admission_counters = snapshot.admission_counters();
    crate::mcc_backend_evidence::record_mcc_prepared_fingerprint_admission_runtime_consumption(
        crate::mcc_backend_evidence::MccPreparedFingerprintAdmissionRuntimeConsumption::from_plan(
            snapshot.admission_plan(),
            PETRI_FINGERPRINT_ONLY_ADMISSION_CALLSITE,
            admission_counters.attempted,
            admission_counters.new,
            admission_counters.duplicate,
            admission_counters.fault,
        ),
    );
    (result, stats)
}

/// Dispatch entry point: choose between the sequential and parallel
/// fingerprint-only BFS paths based on configured worker count.
pub(crate) fn explore_fingerprint_only_dispatch<O>(
    net: &PetriNet,
    config: &ExplorationConfig,
    observer: &mut O,
    trace_dir: Option<&std::path::Path>,
) -> (ExplorationResult, FingerprintOnlyStats)
where
    O: ExplorationObserver + Send,
{
    let workers = config.workers();
    if workers >= 2 {
        explore_fingerprint_only_parallel(net, config, observer, trace_dir, workers)
    } else {
        explore_fingerprint_only(net, config, observer, trace_dir)
    }
}

/// Reconstruct a counterexample trace from the disk log.
///
/// Reads the binary trace log and walks backward from `violating_fp` to the
/// initial state (parent_fp == 0), returning the sequence of transition IDs
/// in forward order.
///
/// Returns `None` if the trace file doesn't exist or the fingerprint chain
/// is broken.
///
/// # Usage
///
/// Called by examination code when a violation (e.g., deadlock) is detected
/// during fingerprint-only BFS and `trace_dir` was provided. The examination
/// must track the violating state's 64-bit fingerprint to pass here.
///
/// # Hash collision note
///
/// With 64-bit fingerprints, birthday-paradox collision probability reaches
/// ~50% at ~2^32 (~4 billion) states. For state spaces below 10^8, the
/// probability is <10^-3. For larger explorations, use full-state mode.
#[allow(dead_code)] // Infrastructure for examination-level trace reconstruction.
pub(crate) fn reconstruct_trace(
    trace_dir: &std::path::Path,
    violating_fp: u64,
) -> Option<Vec<u32>> {
    use std::collections::HashMap;
    use std::io::Read;

    let path = trace_dir.join("trace.bin");
    let mut file = std::fs::File::open(&path).ok()?;
    let mut data = Vec::new();
    file.read_to_end(&mut data).ok()?;

    let entry_size = std::mem::size_of::<TraceEntry>();
    if data.len() % entry_size != 0 {
        return None;
    }

    // Build parent map: state_fp -> (parent_fp, transition_id)
    let mut parent_map: HashMap<u64, (u64, u32)> = HashMap::new();
    for chunk in data.chunks_exact(entry_size) {
        let entry: TraceEntry =
            unsafe { std::ptr::read_unaligned(chunk.as_ptr().cast::<TraceEntry>()) };
        parent_map.insert(entry.state_fp, (entry.parent_fp, entry.transition_id));
    }

    // Walk backward from violating state to initial state.
    let mut trace = Vec::new();
    let mut current = violating_fp;
    let mut steps = 0;
    let max_steps = parent_map.len();

    loop {
        let (parent, trans_id) = parent_map.get(&current)?;
        if *parent == 0 {
            // Reached initial state.
            break;
        }
        trace.push(*trans_id);
        current = *parent;
        steps += 1;
        if steps > max_steps {
            // Cycle detection -- broken trace.
            return None;
        }
    }

    trace.reverse();
    Some(trace)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::explorer::config::ExplorationConfig;
    use crate::petri_net::{Arc, PetriNet, PlaceIdx, PlaceInfo, TransitionIdx, TransitionInfo};
    use tla_mc_core::SharedEngineFrontendFamily;

    fn simple_linear_net() -> PetriNet {
        PetriNet {
            name: Some("linear".into()),
            places: vec![
                PlaceInfo {
                    id: "P0".into(),
                    name: None,
                },
                PlaceInfo {
                    id: "P1".into(),
                    name: None,
                },
            ],
            transitions: vec![TransitionInfo {
                id: "T0".into(),
                name: None,
                inputs: vec![Arc {
                    place: PlaceIdx(0),
                    weight: 1,
                }],
                outputs: vec![Arc {
                    place: PlaceIdx(1),
                    weight: 1,
                }],
            }],
            initial_marking: vec![1, 0],
        }
    }

    fn cyclic_net() -> PetriNet {
        PetriNet {
            name: Some("cycle".into()),
            places: vec![
                PlaceInfo {
                    id: "P0".into(),
                    name: None,
                },
                PlaceInfo {
                    id: "P1".into(),
                    name: None,
                },
            ],
            transitions: vec![
                TransitionInfo {
                    id: "T0".into(),
                    name: None,
                    inputs: vec![Arc {
                        place: PlaceIdx(0),
                        weight: 1,
                    }],
                    outputs: vec![Arc {
                        place: PlaceIdx(1),
                        weight: 1,
                    }],
                },
                TransitionInfo {
                    id: "T1".into(),
                    name: None,
                    inputs: vec![Arc {
                        place: PlaceIdx(1),
                        weight: 1,
                    }],
                    outputs: vec![Arc {
                        place: PlaceIdx(0),
                        weight: 1,
                    }],
                },
            ],
            initial_marking: vec![1, 0],
        }
    }

    fn deadlock_net() -> PetriNet {
        PetriNet {
            name: Some("deadlock".into()),
            places: vec![PlaceInfo {
                id: "P0".into(),
                name: None,
            }],
            transitions: vec![],
            initial_marking: vec![1],
        }
    }

    fn counting_net(initial_tokens: u64) -> PetriNet {
        PetriNet {
            name: Some("counting".into()),
            places: vec![
                PlaceInfo {
                    id: "P0".into(),
                    name: None,
                },
                PlaceInfo {
                    id: "P1".into(),
                    name: None,
                },
            ],
            transitions: vec![TransitionInfo {
                id: "T0".into(),
                name: None,
                inputs: vec![Arc {
                    place: PlaceIdx(0),
                    weight: 1,
                }],
                outputs: vec![Arc {
                    place: PlaceIdx(1),
                    weight: 1,
                }],
            }],
            initial_marking: vec![initial_tokens, 0],
        }
    }

    struct CountingObserver {
        states: usize,
        deadlocks: usize,
        firings: usize,
    }

    impl CountingObserver {
        fn new() -> Self {
            Self {
                states: 0,
                deadlocks: 0,
                firings: 0,
            }
        }
    }

    impl ExplorationObserver for CountingObserver {
        fn on_new_state(&mut self, _marking: &[u64]) -> bool {
            self.states += 1;
            true
        }

        fn on_transition_fire(&mut self, _trans: TransitionIdx) -> bool {
            self.firings += 1;
            true
        }

        fn on_deadlock(&mut self, _marking: &[u64]) {
            self.deadlocks += 1;
        }

        fn is_done(&self) -> bool {
            false
        }
    }

    #[test]
    fn fingerprint_only_dedup_identity_is_fail_closed() {
        let dedup_identity = fingerprint_only_marking_dedup_identity();

        dedup_identity
            .require_fail_closed()
            .expect("fingerprint-only dedup identity must fail closed");
        assert_eq!(dedup_identity.id, PETRI_MARKING_DEDUP_ID);
        assert_eq!(dedup_identity.storage, SharedDedupStorageKind::Cas);
        assert_eq!(dedup_identity.lane, SetupTraceLaneKind::Fingerprint);
        assert_eq!(
            dedup_identity.fingerprint.algorithm,
            SharedFingerprintAlgorithm::CanonicalBytesSha256
        );
        assert_eq!(
            dedup_identity.fingerprint.value_kind,
            SharedFingerprintValueKind::MarkingVector
        );
        assert_eq!(dedup_identity.fingerprint.digest_bits, 64);
        assert_eq!(
            dedup_identity.storage_config_identity.as_deref(),
            Some(PETRI_FINGERPRINT_ONLY_STORAGE_CONFIG)
        );
        assert!(dedup_identity
            .fingerprint
            .reusable_frontend_families()
            .contains(&SharedEngineFrontendFamily::MccPetri));
    }

    #[test]
    fn fingerprint_only_marking_admission_plan_binds_shared_runtime_contract() {
        let plan = fingerprint_only_marking_admission_plan();

        plan.validate_runtime_admission()
            .expect("marking admission must use a valid shared runtime contract");
        assert_eq!(
            plan.id,
            PETRI_MARKING_PREPARED_FINGERPRINT_ADMISSION_PLAN_ID
        );
        assert_eq!(plan.source_kind, CheckerSourceKind::MccPetri);
        assert_eq!(plan.payload_kind, PreparedProgramPayloadKind::MccPetri);
        assert_eq!(plan.storage_kind, PreparedStorageKind::PetriMarking);
        assert_eq!(plan.lane, SetupTraceLaneKind::Fingerprint);
        assert_eq!(
            plan.payload_witness,
            PreparedFingerprintPayloadWitnessKind::PetriMarkingCas
        );
        assert_eq!(
            plan.duplicate_authorization,
            SharedDuplicateAuthorization::CanonicalPayloadEquality
        );
        assert_eq!(plan.dedup.storage, SharedDedupStorageKind::Cas);
        assert_eq!(
            plan.dedup.collision_policy,
            tla_mc_core::SharedCollisionPolicy::RejectOnCollision
        );
    }

    #[test]
    fn fingerprint_only_table_capacity_caps_unbounded_state_limit() {
        let config = ExplorationConfig::new(usize::MAX);

        assert_eq!(
            fingerprint_only_table_capacity(&config),
            FINGERPRINT_ONLY_MAX_CAPACITY_STATES * 2
        );
    }

    #[test]
    fn fingerprint_only_table_capacity_uses_primary_capacity_override() {
        let config = ExplorationConfig::new(usize::MAX).with_storage_primary_capacity(5_000);

        assert_eq!(fingerprint_only_table_capacity(&config), 10_000);
    }

    #[test]
    fn collision_guard_authorizes_exact_duplicate_payload() {
        let fp_set = CasFingerprintSet::new(16);
        let mut guard = PackedMarkingCollisionGuard::new(fingerprint_only_marking_admission_plan());
        assert_eq!(
            guard.admission_plan().id,
            PETRI_MARKING_PREPARED_FINGERPRINT_ADMISSION_PLAN_ID
        );

        let first = guard
            .admit(&fp_set, 42, b"packed-marking")
            .expect("first marking should be admitted");
        let second = guard
            .admit(&fp_set, 42, b"packed-marking")
            .expect("exact duplicate should be authorized");

        assert_eq!(first, FingerprintAdmission::New);
        assert_eq!(second, FingerprintAdmission::Duplicate);
        assert_eq!(FingerprintSet::len(&fp_set), 1);
        assert_eq!(
            guard.admission_counters(),
            PreparedFingerprintAdmissionRuntimeCounters {
                attempted: 2,
                new: 1,
                duplicate: 1,
                fault: 0,
            }
        );
    }

    #[test]
    fn collision_guard_rejects_same_fingerprint_different_payload() {
        let fp_set = CasFingerprintSet::new(16);
        let mut guard = PackedMarkingCollisionGuard::new(fingerprint_only_marking_admission_plan());

        let first = guard
            .admit(&fp_set, 42, b"resident")
            .expect("resident marking should be admitted");
        assert_eq!(first, FingerprintAdmission::New);

        let error = guard
            .admit(&fp_set, 42, b"candidate")
            .expect_err("same fingerprint with different bytes must fail closed");

        assert_eq!(error.backend, "prepared_fingerprint_admission");
        assert_eq!(error.operation, "admit");
        assert!(error.detail.contains("status_code=rejected"));
        assert!(error
            .detail
            .contains("reason_code=canonical_payload_mismatch"));
        assert!(error.detail.contains("fail_closed=true"));
        assert!(error
            .detail
            .contains("collision_policy=reject_on_collision"));
        assert!(error.detail.contains("payload_witness=petri_marking_cas"));
        assert!(error.detail.contains("frontend_family=mcc_petri"));
        assert_eq!(FingerprintSet::len(&fp_set), 1);
        assert_eq!(
            guard.admission_counters(),
            PreparedFingerprintAdmissionRuntimeCounters {
                attempted: 2,
                new: 1,
                duplicate: 0,
                fault: 1,
            }
        );
    }

    /// A WIDE net with a SMALL reachable set — the exact shape the incremental
    /// enabled update targets (high T, tiny |R|, O(T) full-scan dominates).
    ///
    /// Structure: ONE bounded counter chain `c_0 -> c_1 -> ... -> c_len` (a single
    /// token walking `len+1` places, then `recycle` returns it to `c_0`), giving a
    /// reachable set of exactly `len+1` markings. Plus `spinners` always-enabled
    /// self-loop transitions on a shared, always-marked place — each is enabled at
    /// EVERY state and never changes the marking, so they inflate the transition
    /// count `T` (and thus the per-state enabled-scan cost) WITHOUT inflating |R|.
    ///
    /// So `T = spinners + len + 1` (wide) while `|R| = len + 1` (tiny). The
    /// incremental update re-evaluates only the ~2 transitions touching the chain
    /// place that moved, never re-scanning the `spinners` always-enabled loops.
    fn wide_small_state_net(spinners: usize, len: usize) -> PetriNet {
        // Places: [hub, c_0 .. c_len]
        let mut places = vec![PlaceInfo {
            id: "hub".into(),
            name: None,
        }];
        for p in 0..=len {
            places.push(PlaceInfo {
                id: format!("c{p}"),
                name: None,
            });
        }
        let chain = |p: usize| -> u32 { (1 + p) as u32 };

        let mut transitions = Vec::new();
        // Spinners: self-loops on the always-marked hub. Always enabled, never
        // change the marking ⇒ widen T, keep |R| fixed.
        for s in 0..spinners {
            transitions.push(TransitionInfo {
                id: format!("spin{s}"),
                name: None,
                inputs: vec![Arc {
                    place: PlaceIdx(0),
                    weight: 1,
                }],
                outputs: vec![Arc {
                    place: PlaceIdx(0),
                    weight: 1,
                }],
            });
        }
        // The single counter chain c_0 -> c_1 -> ... -> c_len.
        for p in 0..len {
            transitions.push(TransitionInfo {
                id: format!("step{p}"),
                name: None,
                inputs: vec![Arc {
                    place: PlaceIdx(chain(p)),
                    weight: 1,
                }],
                outputs: vec![Arc {
                    place: PlaceIdx(chain(p + 1)),
                    weight: 1,
                }],
            });
        }
        // Recycle c_len -> c_0 so the state space is a cycle of len+1 markings
        // (no deadlock; both paths must agree on 0 deadlocks).
        transitions.push(TransitionInfo {
            id: "recycle".into(),
            name: None,
            inputs: vec![Arc {
                place: PlaceIdx(chain(len)),
                weight: 1,
            }],
            outputs: vec![Arc {
                place: PlaceIdx(chain(0)),
                weight: 1,
            }],
        });

        let mut initial_marking = vec![0u64; places.len()];
        initial_marking[0] = 1; // hub token (keeps every spinner enabled)
        initial_marking[chain(0) as usize] = 1; // counter token at c_0
        PetriNet {
            name: Some(format!("wide-small-{spinners}s-{len}len")),
            places,
            transitions,
            initial_marking,
        }
    }

    /// Wide-net capability/parity: the incremental enabled path (default) and the
    /// kill-switched full-scan path explore the IDENTICAL state space (same |R|,
    /// firings, deadlocks). The per-state differential assertion (always on under
    /// `debug_assertions`) additionally verifies bit-for-bit enabled-set equality
    /// at every state during the incremental run. Env is serialized to avoid
    /// cross-test races on the process-global kill-switch.
    #[test]
    fn fingerprint_only_wide_net_incremental_matches_full_scan() {
        use std::sync::Mutex;
        static ENV_LOCK: Mutex<()> = Mutex::new(());
        let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());

        // 4000 always-enabled spinners + a 12-state counter cycle: T = 4013,
        // |R| = 13. Full-scan does 4013 is_enabled per state; incremental does ~2.
        let net = wide_small_state_net(4000, 12);
        let config = ExplorationConfig::new(usize::MAX);

        // Incremental ON (default).
        crate::env_guard::remove_var("TY_MCC_DISABLE_INCREMENTAL_ENABLED");
        let mut inc_obs = CountingObserver::new();
        let (inc_result, inc_stats) = explore_fingerprint_only(&net, &config, &mut inc_obs, None);

        // Full-scan via the kill-switch.
        crate::env_guard::set_var("TY_MCC_DISABLE_INCREMENTAL_ENABLED", "1");
        let mut full_obs = CountingObserver::new();
        let (full_result, full_stats) =
            explore_fingerprint_only(&net, &config, &mut full_obs, None);
        crate::env_guard::remove_var("TY_MCC_DISABLE_INCREMENTAL_ENABLED");

        assert!(inc_result.completed, "incremental run must decide");
        assert!(full_result.completed, "full-scan run must decide");
        // IDENTICAL counts both ways — the optimization is behavior-preserving.
        assert_eq!(
            inc_stats.states_visited, full_stats.states_visited,
            "|R| must be identical incremental vs full-scan"
        );
        assert_eq!(inc_obs.states, full_obs.states);
        assert_eq!(inc_obs.firings, full_obs.firings);
        assert_eq!(inc_obs.deadlocks, full_obs.deadlocks);
        // |R| is the 13-marking counter cycle; no deadlock (recycle closes it).
        assert_eq!(inc_stats.states_visited, 13);
        assert_eq!(inc_obs.deadlocks, 0);
        // Non-vacuous on the WIDTH axis: every state fires all 4000 spinners +
        // the chain transition, so the per-state enabled scan is genuinely wide
        // (this is exactly the O(T) cost the incremental path elides). firings =
        // 13 states × 4001 enabled = 52013.
        assert!(
            inc_obs.firings > 50_000,
            "firings={} should reflect the wide per-state enabled set",
            inc_obs.firings
        );
    }

    /// Weighted-arc wide net: every chain transition uses a weight-2 threshold,
    /// so the incremental re-eval must compare against `arc.weight`, not assume 1.
    /// Incremental and full-scan agree end-to-end (and the always-on per-state
    /// differential checks every state during the incremental run).
    #[test]
    fn fingerprint_only_weighted_arcs_incremental_matches_full_scan() {
        use std::sync::Mutex;
        static ENV_LOCK: Mutex<()> = Mutex::new(());
        let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());

        // ONE bounded weighted chain (small |R|) + many WEIGHTED spinners (wide).
        // Places: [hub, p0, p1, p2]. The hub holds 2 tokens; each weighted spinner
        // is a self-loop consuming/producing 2 from the hub (always enabled, never
        // changes the marking — its weight-2 threshold is always met). The chain
        // p0 --(w2)--> p1 --(w2)--> p2 --(w2 recycle)--> p0 walks a single pool of
        // 2 tokens, so |R| is the tiny set of weighted distributions of 2 tokens
        // across {p0,p1,p2}. The incremental re-eval MUST honour the weight-2
        // thresholds (a place with 1 token does NOT enable a w2 consumer).
        let spinners = 2000usize;
        let places = vec![
            PlaceInfo {
                id: "hub".into(),
                name: None,
            },
            PlaceInfo {
                id: "p0".into(),
                name: None,
            },
            PlaceInfo {
                id: "p1".into(),
                name: None,
            },
            PlaceInfo {
                id: "p2".into(),
                name: None,
            },
        ];
        let mut transitions = Vec::new();
        for s in 0..spinners {
            transitions.push(TransitionInfo {
                id: format!("wspin{s}"),
                name: None,
                inputs: vec![Arc {
                    place: PlaceIdx(0),
                    weight: 2,
                }],
                outputs: vec![Arc {
                    place: PlaceIdx(0),
                    weight: 2,
                }],
            });
        }
        // Weighted chain over the 2-token pool: p0 -(w2)-> p1 -(w2)-> p2 -(w2)-> p0.
        for (src, dst) in [(1u32, 2u32), (2, 3), (3, 1)] {
            transitions.push(TransitionInfo {
                id: format!("wstep{src}_{dst}"),
                name: None,
                inputs: vec![Arc {
                    place: PlaceIdx(src),
                    weight: 2,
                }],
                outputs: vec![Arc {
                    place: PlaceIdx(dst),
                    weight: 2,
                }],
            });
        }
        // hub = 2 (every weight-2 spinner enabled), pool = 2 tokens on p0.
        let initial_marking = vec![2, 2, 0, 0];
        let net = PetriNet {
            name: Some("weighted-wide".into()),
            places,
            transitions,
            initial_marking,
        };
        let config = ExplorationConfig::new(usize::MAX);

        crate::env_guard::remove_var("TY_MCC_DISABLE_INCREMENTAL_ENABLED");
        let mut inc_obs = CountingObserver::new();
        let (inc_result, inc_stats) = explore_fingerprint_only(&net, &config, &mut inc_obs, None);

        crate::env_guard::set_var("TY_MCC_DISABLE_INCREMENTAL_ENABLED", "1");
        let mut full_obs = CountingObserver::new();
        let (full_result, full_stats) =
            explore_fingerprint_only(&net, &config, &mut full_obs, None);
        crate::env_guard::remove_var("TY_MCC_DISABLE_INCREMENTAL_ENABLED");

        assert!(inc_result.completed && full_result.completed);
        assert_eq!(inc_stats.states_visited, full_stats.states_visited);
        assert_eq!(inc_obs.firings, full_obs.firings);
        assert_eq!(inc_obs.deadlocks, full_obs.deadlocks);
        // |R| = 3 weighted distributions of the 2-token pool ({p0:2},{p1:2},{p2:2}).
        assert_eq!(inc_stats.states_visited, 3);
        // Wide: each state fires 2000 spinners + 1 chain step. Confirms the
        // weight-2 incremental re-eval matched full-scan across the whole run.
        assert!(inc_obs.firings > 5_000);
    }

    #[test]
    fn fingerprint_only_linear_net() {
        let net = simple_linear_net();
        let config = ExplorationConfig::default();
        let mut observer = CountingObserver::new();
        let (result, stats) = explore_fingerprint_only(&net, &config, &mut observer, None);

        assert!(result.completed);
        assert_eq!(stats.states_visited, 2);
        assert_eq!(stats.admission_attempted, 2);
        assert_eq!(stats.admission_new, 2);
        assert_eq!(stats.admission_duplicate, 0);
        assert_eq!(stats.admission_fault, 0);
        assert_eq!(observer.states, 2);
        assert_eq!(observer.deadlocks, 1);
        assert_eq!(observer.firings, 1);
    }

    #[test]
    fn fingerprint_only_cyclic_net() {
        let net = cyclic_net();
        let config = ExplorationConfig::default();
        let mut observer = CountingObserver::new();
        let (result, stats) = explore_fingerprint_only(&net, &config, &mut observer, None);

        assert!(result.completed);
        assert_eq!(stats.states_visited, 2);
        assert_eq!(observer.deadlocks, 0);
    }

    #[test]
    fn fingerprint_only_deadlock_net() {
        let net = deadlock_net();
        let config = ExplorationConfig::default();
        let mut observer = CountingObserver::new();
        let (result, stats) = explore_fingerprint_only(&net, &config, &mut observer, None);

        assert!(result.completed);
        assert_eq!(stats.states_visited, 1);
        assert_eq!(observer.deadlocks, 1);
    }

    #[test]
    fn fingerprint_only_state_limit() {
        let net = counting_net(100);
        let config = ExplorationConfig::new(10);
        let mut observer = CountingObserver::new();
        let (result, stats) = explore_fingerprint_only(&net, &config, &mut observer, None);

        assert!(!result.completed);
        assert_eq!(result.states_visited, 10);
        assert_eq!(stats.states_visited, 10);
        assert_eq!(observer.states, 10);
    }

    #[test]
    fn fingerprint_only_reports_memory_stats() {
        let net = cyclic_net();
        let config = ExplorationConfig::default();
        let mut observer = CountingObserver::new();
        let (_result, stats) = explore_fingerprint_only(&net, &config, &mut observer, None);

        // CAS table and exact collision guard should have allocated memory.
        assert!(stats.fp_set_memory_bytes > 0);
        assert!(stats.collision_guard_memory_bytes > 0);
    }

    #[test]
    fn fingerprint_only_matches_full_bfs_state_count() {
        use crate::explorer::observer::explore;

        let nets = vec![
            simple_linear_net(),
            cyclic_net(),
            deadlock_net(),
            counting_net(5),
        ];

        for net in nets {
            let config = ExplorationConfig::default();

            let mut full_observer = CountingObserver::new();
            let full_result = explore(&net, &config, &mut full_observer);

            let mut fp_observer = CountingObserver::new();
            let (fp_result, _stats) =
                explore_fingerprint_only(&net, &config, &mut fp_observer, None);

            assert_eq!(
                full_observer.states, fp_observer.states,
                "state count mismatch for net {:?}",
                net.name
            );
            assert_eq!(
                full_observer.deadlocks, fp_observer.deadlocks,
                "deadlock count mismatch for net {:?}",
                net.name
            );
            assert_eq!(
                full_result.completed, fp_result.completed,
                "completion mismatch for net {:?}",
                net.name
            );
        }
    }

    #[test]
    fn fingerprint_only_with_trace_logging() {
        let dir = tempfile::tempdir().expect("create temp dir");
        let net = simple_linear_net();
        let config = ExplorationConfig::default();
        let mut observer = CountingObserver::new();
        let (result, _stats) =
            explore_fingerprint_only(&net, &config, &mut observer, Some(dir.path()));

        assert!(result.completed);

        // Trace file should exist and be non-empty.
        let trace_path = dir.path().join("trace.bin");
        assert!(trace_path.exists());
        let metadata = std::fs::metadata(&trace_path).expect("trace metadata");
        assert!(metadata.len() > 0);
    }

    #[test]
    fn fingerprint_only_trace_reconstruction() {
        // counting_net(3): P0=3 -> fire T0 three times -> P0=0,P1=3
        // States: [3,0] -> [2,1] -> [1,2] -> [0,3]  (4 states, linear chain)
        let dir = tempfile::tempdir().expect("create temp dir");
        let net = counting_net(3);
        let config = ExplorationConfig::default();

        // Track the fingerprint of the last discovered (deadlock) state.
        struct DeadlockFpObserver {
            last_fp: u64,
        }
        impl ExplorationObserver for DeadlockFpObserver {
            fn on_new_state(&mut self, marking: &[u64]) -> bool {
                // Compute fingerprint the same way the explorer does.
                use crate::marking::{pack_marking_config, PreparedMarking};
                let net = counting_net(3);
                let prepared = PreparedMarking::analyze(&net);
                let mut buf = Vec::new();
                pack_marking_config(marking, &prepared.config, &mut buf);
                let fp128 = fingerprint_marking(&buf);
                self.last_fp = fp128 as u64;
                true
            }
            fn on_transition_fire(&mut self, _trans: TransitionIdx) -> bool {
                true
            }
            fn on_deadlock(&mut self, _marking: &[u64]) {}
            fn is_done(&self) -> bool {
                false
            }
        }

        let mut observer = DeadlockFpObserver { last_fp: 0 };
        let (result, stats) =
            explore_fingerprint_only(&net, &config, &mut observer, Some(dir.path()));

        assert!(result.completed);
        assert_eq!(stats.states_visited, 4);

        // Reconstruct trace from the final state (deadlock at [0,3]).
        let trace = reconstruct_trace(dir.path(), observer.last_fp);
        assert!(
            trace.is_some(),
            "trace reconstruction should succeed for a linear chain"
        );
        let transitions = trace.expect("verified Some above");
        // The chain [3,0]->[2,1]->[1,2]->[0,3] requires 3 firings of T0.
        assert_eq!(transitions.len(), 3, "should have 3 transition firings");
        // All firings should be T0 (index 0).
        assert!(
            transitions.iter().all(|&t| t == 0),
            "all transitions should be T0"
        );
    }

    #[test]
    fn fingerprint_only_trace_reconstruction_returns_none_without_trace() {
        let dir = tempfile::tempdir().expect("create temp dir");
        // No trace file written -- reconstruction should return None.
        assert!(reconstruct_trace(dir.path(), 12345).is_none());
    }

    #[test]
    fn fingerprint_only_early_termination() {
        struct StopAfterTwo {
            count: usize,
        }

        impl ExplorationObserver for StopAfterTwo {
            fn on_new_state(&mut self, _marking: &[u64]) -> bool {
                self.count += 1;
                self.count < 2
            }

            fn on_transition_fire(&mut self, _trans: TransitionIdx) -> bool {
                true
            }

            fn on_deadlock(&mut self, _marking: &[u64]) {}

            fn is_done(&self) -> bool {
                self.count >= 2
            }
        }

        let net = counting_net(100);
        let config = ExplorationConfig::default();
        let mut observer = StopAfterTwo { count: 0 };
        let (result, _stats) = explore_fingerprint_only(&net, &config, &mut observer, None);

        assert!(result.stopped_by_observer);
        assert!(!result.completed);
    }

    // -----------------------------------------------------------------
    // R-1 / R-2 regression tests (perf audit 3c7abf31)
    // -----------------------------------------------------------------

    fn fanout_net(fanout: usize) -> PetriNet {
        // P0 has `fanout` tokens; one transition per output place. Each
        // firing distributes one token to a different sink, producing
        // `fanout + 1` reachable markings (initial + one per fire). Useful
        // for synthetic races where two workers could race on the same
        // successor fingerprint when chunk boundaries align.
        let mut places = vec![PlaceInfo {
            id: "P0".into(),
            name: None,
        }];
        let mut transitions = Vec::with_capacity(fanout);
        for i in 0..fanout {
            places.push(PlaceInfo {
                id: format!("S{i}"),
                name: None,
            });
            transitions.push(TransitionInfo {
                id: format!("T{i}"),
                name: None,
                inputs: vec![Arc {
                    place: PlaceIdx(0),
                    weight: 1,
                }],
                outputs: vec![Arc {
                    place: PlaceIdx((i + 1) as u32),
                    weight: 1,
                }],
            });
        }
        let mut initial_marking = vec![fanout as u64];
        initial_marking.extend(std::iter::repeat(0u64).take(fanout));
        PetriNet {
            name: Some("fanout".into()),
            places,
            transitions,
            initial_marking,
        }
    }

    #[test]
    fn test_parallel_fingerprint_only_matches_sequential() {
        // R-1: parallel exploration must produce identical state counts
        // and observer event totals as sequential.
        let nets = vec![
            ("linear", simple_linear_net()),
            ("cyclic", cyclic_net()),
            ("counting-5", counting_net(5)),
            ("fanout-3", fanout_net(3)),
        ];

        for (name, net) in nets {
            let seq_config = ExplorationConfig::default().with_workers(1);
            let mut seq_obs = CountingObserver::new();
            let (seq_result, seq_stats) =
                explore_fingerprint_only_dispatch(&net, &seq_config, &mut seq_obs, None);

            let par_config = ExplorationConfig::default().with_workers(4);
            let mut par_obs = CountingObserver::new();
            let (par_result, par_stats) =
                explore_fingerprint_only_dispatch(&net, &par_config, &mut par_obs, None);

            assert_eq!(
                seq_stats.states_visited, par_stats.states_visited,
                "{name}: states_visited mismatch (seq={} par={})",
                seq_stats.states_visited, par_stats.states_visited
            );
            assert_eq!(
                seq_obs.states, par_obs.states,
                "{name}: observer.states mismatch"
            );
            assert_eq!(
                seq_obs.deadlocks, par_obs.deadlocks,
                "{name}: observer.deadlocks mismatch"
            );
            assert_eq!(
                seq_result.completed, par_result.completed,
                "{name}: completed flag mismatch"
            );
            assert_eq!(
                seq_stats.admission_new, par_stats.admission_new,
                "{name}: admission_new mismatch"
            );
        }
    }

    #[test]
    fn test_parallel_fingerprint_only_no_double_count() {
        // R-1 soundness: workers racing on the same successor fingerprint
        // must result in exactly one admission (first-seen wins). The
        // `cyclic` net forces both transitions to converge back on the
        // initial fingerprint; with 4 workers, multiple chunks may try to
        // re-enqueue the initial state simultaneously. The admission
        // counters detect any double-count by reporting more `new`
        // admissions than there are distinct states.
        for _ in 0..16 {
            let net = cyclic_net();
            let config = ExplorationConfig::default().with_workers(4);
            let mut obs = CountingObserver::new();
            let (result, stats) = explore_fingerprint_only_dispatch(&net, &config, &mut obs, None);
            assert!(result.completed);
            assert_eq!(stats.states_visited, 2, "exactly two distinct states");
            assert_eq!(stats.admission_new, 2, "exactly two `new` admissions");
            assert_eq!(obs.states, 2, "observer sees exactly two new states");
        }
    }

    #[test]
    fn test_arena_collision_guard_membership_identical_to_hashmap() {
        // R-2: arena-backed collision guard must produce identical
        // insert/contains semantics as the original Box<[u8]> hashmap.
        // We exercise it with a 1000-entry corpus of synthetic
        // (fingerprint, payload) pairs and assert the admission decisions
        // match a reference oracle (the resident hashmap of known bytes).
        let fp_set = CasFingerprintSet::new(4096);
        let mut guard = PackedMarkingCollisionGuard::new(fingerprint_only_marking_admission_plan());
        let mut oracle: FxHashMap<u64, Vec<u8>> = FxHashMap::default();

        for i in 0..1000u64 {
            // Use varied payload lengths to exercise arena packing.
            let payload: Vec<u8> = (0..=(i % 17) as u8)
                .map(|j| ((i.wrapping_mul(31).wrapping_add(j as u64)) & 0xff) as u8)
                .collect();
            let admission = guard
                .admit(&fp_set, i, &payload)
                .expect("synthetic admission should not fault");
            if let Some(prev) = oracle.insert(i, payload.clone()) {
                assert_eq!(prev, payload, "oracle must match arena resident");
                assert!(admission.is_duplicate(), "oracle predicts duplicate");
            } else {
                assert!(admission.is_new(), "oracle predicts new");
            }
        }

        // Re-admit every entry; all should be duplicates and the arena
        // must report the exact same bytes for each fingerprint.
        for (fp, payload) in &oracle {
            let admission = guard
                .admit(&fp_set, *fp, payload)
                .expect("re-admit should not fault");
            assert!(admission.is_duplicate());
            let resident = guard
                .residents
                .lookup(*fp)
                .expect("resident slice must be present");
            assert_eq!(resident, payload.as_slice());
        }
    }

    #[test]
    fn test_arena_collision_guard_memory_lower() {
        // R-2 win: arena overhead is ~16 bytes/entry (8 key + 8 slice)
        // plus payload, vs. the original ~32 bytes/entry (8 key + 24
        // Box<[u8]> header) plus a 16-byte malloc header per payload. For
        // 10K small entries that is ~480 KB (original) → ~240 KB + payload
        // (arena). Assert the arena overhead per entry stays under the
        // original's allocator-attributed lower bound.
        let fp_set = CasFingerprintSet::new(1 << 15);
        let mut guard = PackedMarkingCollisionGuard::new(fingerprint_only_marking_admission_plan());
        const N: u64 = 10_000;
        let payload = [0xAAu8; 8];

        for i in 0..N {
            guard
                .admit(&fp_set, i, &payload)
                .expect("admit should not fault");
        }

        let bytes = guard.memory_bytes();
        let entries = guard.residents.entries() as u64;
        assert_eq!(entries, N, "all entries should be admitted");

        // Original layout: 8 (key) + 24 (Box<[u8]> header) per entry +
        // 16-byte malloc header + 8 payload bytes ≈ 56 B/entry minimum,
        // plus hashmap headroom often pushing it to >80 B/entry.
        // Arena layout: 8 (key) + 16 (MarkingSlice = u64+u32+pad) + 8
        // payload = 32 B/entry plus per-entry hashmap headroom (~50%).
        // Expect <56 B amortized per entry (>30% reduction vs original).
        let bytes_per_entry = bytes as u64 / entries;
        assert!(
            bytes_per_entry < 56,
            "arena should average <56 B/entry, got {bytes_per_entry} (total {bytes} B / {entries} entries)"
        );
    }

    #[test]
    fn test_parallel_fingerprint_only_fanout_no_lost_states() {
        // Synthetic race amplifier: fanout-8 with workers=4 produces
        // multiple distinct successors per current state. The shared
        // CAS oracle and per-shard arena must produce identical state
        // counts as sequential.
        for fanout in [2usize, 5, 8] {
            let net = fanout_net(fanout);
            let seq_config = ExplorationConfig::default().with_workers(1);
            let mut seq_obs = CountingObserver::new();
            let (seq_result, seq_stats) =
                explore_fingerprint_only_dispatch(&net, &seq_config, &mut seq_obs, None);

            let par_config = ExplorationConfig::default().with_workers(4);
            let mut par_obs = CountingObserver::new();
            let (par_result, par_stats) =
                explore_fingerprint_only_dispatch(&net, &par_config, &mut par_obs, None);

            assert_eq!(
                seq_stats.states_visited, par_stats.states_visited,
                "fanout={fanout}: state count mismatch"
            );
            assert_eq!(
                seq_obs.states, par_obs.states,
                "fanout={fanout}: observer state count mismatch"
            );
            assert!(par_result.completed);
            assert!(seq_result.completed);
        }
    }
}
