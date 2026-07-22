// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0
#![forbid(unsafe_code)]
#![deny(missing_docs)]

//! Generic explicit-state model checking traits and reusable algorithms.
//!
//! `tla-mc-core` is the domain-agnostic core shared by the `ty` checker lanes.
//! It is deliberately dependency-light (no AY or frontend crates) so TLA+,
//! MCC/Petri, hardware (AIGER/BTOR2), VMT, symbolic, and replay frontends can
//! all build on the same engine without dependency cycles.
//!
//! # What it provides
//!
//! - **Transition-system API and exploration** — the [`TransitionSystem`]
//!   trait plus the sequential and work-stealing parallel BFS engines
//!   ([`explore_bfs`], [`explore_bfs_parallel`]) driven by
//!   [`ExplorationObserver`]s.
//! - **Fingerprint storage** — the [`FingerprintSet`] contract and in-memory,
//!   sharded, and CAS-backed implementations for state deduplication.
//! - **Graph algorithms** — iterative Tarjan SCC detection ([`tarjan_scc`])
//!   and a CTL model-checking engine ([`check_ctl`]).
//! - **Frontend-neutral contracts and evidence** — the prepared-program,
//!   native-contract, capability, analytical-solve, fingerprint-identity,
//!   setup-trace, validation-receipt, and shared-engine-adoption record types
//!   that let every frontend describe and gate work as data (string/checksum
//!   based), and render stable evidence rows for parity tooling.
//!
//! Most evidence/record types share three conventions: a `code()` returning a
//! stable lowercase wire token for each enum, `render_evidence_row`/`render_*`
//! producing a stable space-separated key=value row, and a `validate`/
//! `validate_*_evidence_row` checking the fail-closed contract.
//!
//! # Example
//!
//! ```rust
//! use tla_mc_core::{explore_bfs, ExplorationObserver, NoopObserver, TransitionSystem};
//!
//! #[derive(Clone)]
//! struct Counter;
//!
//! impl TransitionSystem for Counter {
//!     type State = u8;
//!     type Action = &'static str;
//!     type Fingerprint = u8;
//!
//!     fn initial_states(&self) -> Vec<Self::State> {
//!         vec![0]
//!     }
//!
//!     fn successors(&self, state: &Self::State) -> Vec<(Self::Action, Self::State)> {
//!         if *state < 2 {
//!             vec![("inc", state + 1)]
//!         } else {
//!             Vec::new()
//!         }
//!     }
//!
//!     fn fingerprint(&self, state: &Self::State) -> Self::Fingerprint {
//!         *state
//!     }
//! }
//!
//! let mut observer = NoopObserver::<Counter>::default();
//! let outcome = explore_bfs(&Counter, &mut observer).unwrap();
//! assert!(outcome.completed);
//! assert_eq!(outcome.states_discovered, 3);
//! ```

mod analytical_solve;
mod backend_capability;
mod backend_evidence;
mod bfs;
mod cas_fpset;
mod ctl;
mod evidence_row;
mod fingerprint_identity;
mod hardware_replay_evidence;
mod native_contract;
mod observer;
mod prepared_fingerprint_admission;
mod prepared_program;
mod prepared_successor_batch;
mod random_walk;
mod scc;
mod setup_trace;
mod shared_engine_adoption;
mod storage;
mod traits;
mod validation_receipt;

pub use analytical_solve::{
    choose_analytical_solve_route, AnalyticalSolveDecision, AnalyticalSolveDecisionReason,
    AnalyticalSolveDecisionStatus, AnalyticalSolvePortfolioLifecycle, AnalyticalSolveRoute,
    AnalyticalSolveRoutingDecision, AnalyticalSolveValidationReceiptReadiness,
};
pub use backend_capability::{
    ay_chc_capability, ay_sat_capability, ay_smt_capability, ay_symbolic_execution_capability,
    external_ay_binary_capability, find_ay_binary, local_symbolic_execution_capability,
    preferred_ay_backend_for_symbolic_execution, AYBinaryAvailability, AYBinarySource,
    BackendCapability, BackendDomain, BackendKind, CapabilityReport, CapabilityRole,
    CapabilityStatus, ProblemKind, ProductionRoutingStatus, SolverDelegation,
    SolverDelegationTarget, SolverFacet, SolverLimits, SymbolicExecutionDetection,
    SymbolicExecutionReason, SymbolicExecutionStatus, UnsupportedReason,
};
pub use backend_evidence::{
    render_capability_lane_evidence, render_capability_lane_status_evidence,
    render_symbolic_execution_detection_evidence, CapabilityLaneDecision, NO_REASON_CODE,
};
pub use bfs::{
    explore_bfs, explore_bfs_parallel, explore_bfs_parallel_with_options,
    explore_bfs_parallel_with_storage, explore_bfs_with_options, explore_bfs_with_storage,
    BfsError, BfsOptions, BfsOutcome,
};
pub use cas_fpset::{CasFingerprintSet, PartitionedCasFingerprintSet};
pub use ctl::{
    build_predecessor_adjacency, build_predecessor_csr, check_ctl, CsrAdjacency, CtlAtomEvaluator,
    CtlEdge, CtlEngine, CtlFormula, IndexedCtlGraph,
};
pub use fingerprint_identity::{
    FingerprintCanonicalPayload, FingerprintDomainKey, FingerprintDomainKeyBuilder,
    FingerprintDomainProjection, FingerprintDomainStoragePolicy, SharedCollisionPolicy,
    SharedDedupIdentity, SharedDedupScope, SharedDedupStorageKind, SharedDuplicateAuthorization,
    SharedFingerprintAlgorithm, SharedFingerprintCanonicalDomain, SharedFingerprintIdentity,
    SharedFingerprintIdentityRejection, SharedFingerprintValueKind, SharedNativeCacheReusePolicy,
    SharedNativePlanningIdentity, SHARED_FINGERPRINT_IDENTITY_SCHEMA,
    SHARED_FINGERPRINT_IDENTITY_SCHEMA_VERSION,
    SHARED_FINGERPRINT_REJECTION_DIGEST_BITS_EXCEED_ALGORITHM,
    SHARED_FINGERPRINT_REJECTION_EMPTY_CANONICALIZATION_VERSION,
    SHARED_FINGERPRINT_REJECTION_EMPTY_CANONICAL_DOMAIN, SHARED_FINGERPRINT_REJECTION_EMPTY_ID,
    SHARED_FINGERPRINT_REJECTION_EMPTY_NAMESPACE, SHARED_FINGERPRINT_REJECTION_INVALID_DIGEST_BITS,
    SHARED_FINGERPRINT_REJECTION_NON_FAIL_CLOSED_COLLISION_POLICY,
    SHARED_NATIVE_CACHE_REUSE_DISABLED, SHARED_NATIVE_CACHE_REUSE_FRONTEND_LOCAL_ONLY,
    SHARED_NATIVE_CACHE_REUSE_FRONTEND_REUSABLE,
    SHARED_NATIVE_PLANNING_REJECTION_DUPLICATE_FRONTEND_FAMILY,
    SHARED_NATIVE_PLANNING_REJECTION_EMPTY_FRONTEND_FAMILY_SCOPE,
    SHARED_NATIVE_PLANNING_REJECTION_FRONTEND_REUSABLE_REQUIRES_CACHE_IDENTITY,
    SHARED_NATIVE_PLANNING_REJECTION_FRONTEND_REUSABLE_REQUIRES_COMPATIBLE_FAMILIES,
    SHARED_NATIVE_PLANNING_REJECTION_FRONTEND_REUSABLE_REQUIRES_FINGERPRINT_DOMAIN,
    SHARED_NATIVE_PLANNING_REJECTION_FRONTEND_REUSABLE_REQUIRES_PLAN_REUSE_MANIFEST,
    SHARED_NATIVE_PLANNING_REJECTION_INCOMPLETE_PLAN_REUSE_MANIFEST,
    SHARED_NATIVE_PLANNING_REJECTION_INVALID_CACHE_REUSE_POLICY,
    SHARED_NATIVE_PLANNING_REJECTION_MISSING_SOURCE_FRONTEND_FAMILY,
};
pub use hardware_replay_evidence::{
    aiger_hardware_proof_replay_boundary_status, btor2_hardware_proof_replay_boundary_status,
    hardware_replay_decision_accepts_replay_primitive,
    runtime_blocked_hardware_replay_decision_statuses,
    runtime_hardware_proof_replay_boundary_statuses,
    validate_hardware_proof_replay_boundary_evidence_row,
    validate_hardware_replay_decision_evidence_row, HardwareProofReplayBoundaryEvidenceError,
    HardwareProofReplayBoundaryStatus, HardwareReplayDecisionEvidenceError,
    HardwareReplayDecisionStatus, HardwareReplayPrimitiveAssignmentStatus,
    HardwareReplayPrimitiveConsumerStatus, HardwareReplayPrimitiveDecisionStatus,
    HardwareReplayPrimitiveRejectionReason, HardwareReplayPrimitiveStatus,
    HARDWARE_PROOF_REPLAY_BOUNDARY_REQUIRED_FIELDS, HARDWARE_PROOF_REPLAY_BOUNDARY_ROW_KIND,
    HARDWARE_PROOF_REPLAY_BOUNDARY_SCHEMA, HARDWARE_PROOF_REPLAY_BOUNDARY_SCHEMA_VERSION,
    HARDWARE_REPLAY_DECISION_AY_PROOF_FIELDS, HARDWARE_REPLAY_DECISION_AY_REPLAY_FIELDS,
    HARDWARE_REPLAY_DECISION_REQUIRED_FIELDS, HARDWARE_REPLAY_DECISION_ROW_KIND,
    HARDWARE_REPLAY_DECISION_SCHEMA, HARDWARE_REPLAY_DECISION_SCHEMA_VERSION,
    HARDWARE_REPLAY_PRIMITIVE_SCHEMA,
};
pub use native_contract::{
    current_native_frontend_families, SharedNativeAbiParam, SharedNativeAbiSignature,
    SharedNativeAbiValueKind, SharedNativeAdmission, SharedNativeAdmissionDisposition,
    SharedNativeAdmissionReason, SharedNativeAdmissionStatus, SharedNativeContract,
    SharedNativeContractIdentity, SharedNativeContractKind, SharedNativeContractValidationError,
    SharedNativeEvidenceKind, SharedNativeEvidencePolicy, SharedNativeEvidenceRequirement,
    SharedNativeInstallAuthority, SharedNativeLayoutContract, SharedNativeLayoutKind,
    SharedNativeVectorContract, SHARED_NATIVE_CONTRACT_FRONTEND_FAMILIES,
    SHARED_NATIVE_CONTRACT_REQUIRED_FIELDS, SHARED_NATIVE_CONTRACT_ROW_KIND,
    SHARED_NATIVE_CONTRACT_SCHEMA, SHARED_NATIVE_CONTRACT_SCHEMA_VERSION,
};
pub use observer::{
    CompositeObserver, ExplorationObserver, NoopObserver, ParallelObserver, ParallelObserverSummary,
};
pub use prepared_fingerprint_admission::{
    PreparedFingerprintAdmissionCounters, PreparedFingerprintAdmissionOutcome,
    PreparedFingerprintAdmissionPlan, PreparedFingerprintAdmissionValidationEvidence,
    PreparedFingerprintBatchAdmissionOutcome, PreparedFingerprintDuplicateAuthorizationEvidence,
    PreparedFingerprintPayloadWitnessKind, ValidatedPreparedFingerprintAdmissionPlan,
};
pub use prepared_program::{
    validate_prepared_candidate_lane_evidence_row, validate_prepared_checker_program_evidence_row,
    validate_prepared_frontend_extension_evidence_row, validate_prepared_payload_default_use,
    validate_prepared_validation_plan_evidence_row, PreparedAnalyticalSolveDescriptor,
    PreparedAnalyticalSolveKind, PreparedBackendFamilyDescriptor, PreparedCandidateLaneDescriptor,
    PreparedCanonicalIdentityDescriptor, PreparedCanonicalIdentityKind, PreparedCheckerProgram,
    PreparedFingerprintDescriptor, PreparedFingerprintScheme, PreparedFrontendExtensionDescriptor,
    PreparedFrontendExtensionKind, PreparedPayloadIdentityDescriptor, PreparedProgramPayloadKind,
    PreparedPropertyDescriptor, PreparedPropertyKind, PreparedStorageKind,
    PreparedSymbolicProofDescriptor, PreparedSymbolicProofKind, PreparedTransitionDescriptor,
    PreparedTransitionKind, PreparedValidationKind, PreparedValidationPlanDescriptor,
    PREPARED_CANDIDATE_LANE_REQUIRED_FIELDS, PREPARED_CANDIDATE_LANE_ROW_KIND,
    PREPARED_CHECKER_PROGRAM_REQUIRED_FIELDS, PREPARED_CHECKER_PROGRAM_ROW_KIND,
    PREPARED_FRONTEND_EXTENSION_REQUIRED_FIELDS, PREPARED_FRONTEND_EXTENSION_ROW_KIND,
    PREPARED_FUTURE_IMPORTER_RESERVED_PAYLOAD_CODE, PREPARED_VALIDATION_PLAN_REQUIRED_FIELDS,
    PREPARED_VALIDATION_PLAN_ROW_KIND,
};
pub use prepared_successor_batch::{
    PreparedSuccessorAdmissionKind, PreparedSuccessorBatch, PreparedSuccessorBatchDescriptor,
    PreparedSuccessorBatchDescriptorError, PreparedSuccessorPayloadKind, PreparedSuccessorProvider,
    PreparedSuccessorRef, PREPARED_SUCCESSOR_BATCH_REQUIRED_FIELDS,
    PREPARED_SUCCESSOR_BATCH_ROW_KIND, PREPARED_SUCCESSOR_BATCH_SCHEMA,
    PREPARED_SUCCESSOR_BATCH_SCHEMA_VERSION,
};
pub use random_walk::{
    deadline_poll, random_walk_witness, RandomWalkBudget, RandomWalkPoll, RandomWalkStep,
    RandomWalkStepper, DEFAULT_MAX_STEPS as RANDOM_WALK_DEFAULT_MAX_STEPS,
    DEFAULT_POLL_INTERVAL as RANDOM_WALK_DEFAULT_POLL_INTERVAL,
    DEFAULT_WALKS as RANDOM_WALK_DEFAULT_WALKS,
};
pub use scc::{bottom_sccs, tarjan_scc, Scc, TarjanResult};
pub use setup_trace::{
    CheckerArtifactIdentityFields, CheckerSourceKind, SetupTrace, SetupTraceKey,
    SetupTraceLaneKind, SetupTracePhase, SetupTraceTiming, SetupTraceValidationStatus,
    SETUP_TRACE_FINGERPRINT_PHASES, SETUP_TRACE_REQUIRED_FIELDS, SETUP_TRACE_ROW_KIND,
    SETUP_TRACE_SCHEMA, SETUP_TRACE_SCHEMA_VERSION,
};
pub use shared_engine_adoption::{
    validate_shared_engine_adoption_evidence_row, SharedEngineAdoptionBlockerStatus,
    SharedEngineAdoptionEvidence, SharedEngineAdoptionEvidenceError,
    SharedEngineAdoptionFamilyBlocker, SharedEngineAdoptionLevel, SharedEngineFrontendFamily,
    SHARED_ENGINE_ADOPTION_BLOCKER_STATUSES, SHARED_ENGINE_ADOPTION_DEFAULT_FAMILY_BLOCKER,
    SHARED_ENGINE_ADOPTION_EXTRACTION_STATUSES, SHARED_ENGINE_ADOPTION_FRONTEND_FAMILIES,
    SHARED_ENGINE_ADOPTION_LEVELS, SHARED_ENGINE_ADOPTION_REQUIRED_FIELDS,
    SHARED_ENGINE_ADOPTION_ROW_KIND, SHARED_ENGINE_ADOPTION_SCHEMA,
    SHARED_ENGINE_ADOPTION_SCHEMA_VERSION,
};
pub use storage::{
    CapacityStatus, FingerprintAdmission, FingerprintBatchAdmission, FingerprintSet,
    InMemoryFingerprintSet, InsertOutcome, LocalFingerprintSet, LookupOutcome,
    ShardedFingerprintSet, StorageFault, StorageStats,
};
pub use traits::{AtomEvaluator, PorPropertyClass, PorProvider, TransitionSystem};
pub use validation_receipt::{
    validate_validation_receipt, validate_validation_receipt_evidence_row, ValidationReceipt,
    ValidationReceiptArtifactKind, ValidationReceiptStatus, ValidationReceiptValidationError,
    ValidationReceiptValidatorKind, VALIDATION_RECEIPT_REQUIRED_FIELDS,
    VALIDATION_RECEIPT_ROW_KIND, VALIDATION_RECEIPT_SCHEMA, VALIDATION_RECEIPT_SCHEMA_VERSION,
};
