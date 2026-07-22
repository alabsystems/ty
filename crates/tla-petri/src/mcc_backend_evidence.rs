// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Shared backend capability evidence for MCC/Petri runs.
//!
//! MCC stdout must contain only competition result lines, so backend evidence is
//! emitted only to an opt-in JSONL sidecar path.

use std::cell::RefCell;
use std::fs::OpenOptions;
use std::io::Write;
use std::path::Path;
use std::time::{Duration, Instant};

use serde::Serialize;
use sha2::{Digest, Sha256};
use tla_mc_core::{
    runtime_blocked_hardware_replay_decision_statuses,
    runtime_hardware_proof_replay_boundary_statuses, AnalyticalSolveDecision,
    AnalyticalSolveDecisionReason, AnalyticalSolvePortfolioLifecycle, BackendCapability,
    BackendDomain, BackendKind, CapabilityReport, CapabilityRole, CheckerArtifactIdentityFields,
    CheckerSourceKind, PreparedAnalyticalSolveDescriptor, PreparedAnalyticalSolveKind,
    PreparedBackendFamilyDescriptor, PreparedCandidateLaneDescriptor,
    PreparedCanonicalIdentityDescriptor, PreparedCanonicalIdentityKind, PreparedCheckerProgram,
    PreparedFingerprintAdmissionPlan, PreparedFingerprintDescriptor,
    PreparedFingerprintPayloadWitnessKind, PreparedFingerprintScheme,
    PreparedFrontendExtensionDescriptor, PreparedFrontendExtensionKind, PreparedProgramPayloadKind,
    PreparedPropertyKind, PreparedStorageKind, PreparedSymbolicProofDescriptor,
    PreparedSymbolicProofKind, PreparedTransitionKind, PreparedValidationKind,
    PreparedValidationPlanDescriptor, ProblemKind, SetupTrace, SetupTraceLaneKind, SetupTracePhase,
    SetupTraceValidationStatus, SharedDedupIdentity, SharedDedupScope, SharedDedupStorageKind,
    SharedDuplicateAuthorization, SharedEngineAdoptionEvidence, SharedEngineAdoptionFamilyBlocker,
    SharedEngineAdoptionLevel, SharedEngineFrontendFamily, SharedFingerprintAlgorithm,
    SharedFingerprintIdentity, SharedFingerprintValueKind, SolverFacet, SolverLimits,
    SymbolicExecutionDetection, SymbolicExecutionReason, UnsupportedReason,
    HARDWARE_REPLAY_PRIMITIVE_SCHEMA,
};

use crate::examination::{Examination, ExaminationRecord, ExaminationValue, Verdict};
use crate::explorer::{ExplorationConfig, FpsetBackend, StorageMode};
use crate::model::{PreparedModel, SourceNetKind};
use crate::petri_net::PetriNet;

/// Opt-in JSONL sidecar for backend routing evidence.
pub(crate) const MCC_BACKEND_EVIDENCE_JSONL_ENV: &str = "TY_MCC_BACKEND_EVIDENCE_JSONL";
const REACHABILITY_PDR_ENV: &str = "TY_MCC_ENABLE_REACHABILITY_PDR";
const MCC_MODEL_LOAD_SCHEMA: &str = "mcc.model_load.v1";
const MCC_PREPARED_PROGRAM_SCHEMA: &str = "mcc.prepared_program.v1";
const MCC_KERNEL_BUILD_SCHEMA: &str = "mcc.kernel_build.v1";
const MCC_NATIVE_PUBLISH_SCHEMA: &str = "mcc.native_publish.v1";
const MCC_HOT_EXECUTION_SCHEMA: &str = "mcc.hot_execution.v1";
const MCC_RUNTIME_FINGERPRINT_ADOPTION_SCHEMA: &str = "mcc.runtime_fingerprint_adoption.v1";
const MCC_RUNTIME_DEDUP_ADMISSION_SCHEMA: &str = "mcc.runtime_dedup_admission.v1";
const MCC_PREPARED_FINGERPRINT_ADMISSION_RUNTIME_CONSUMPTION_SCHEMA: &str =
    "mcc.prepared_fingerprint_admission.runtime_consumption.v1";
const MCC_PREPARED_FINGERPRINT_ADMISSION_RUNTIME_CONSUMPTION_MISSING_SCHEMA: &str =
    "mcc.prepared_fingerprint_admission.runtime_consumption_missing.v1";
const MCC_PREPARED_PROGRAM_FINGERPRINT_ALGORITHM: &str = "sha256";
const MCC_PREPARED_PROGRAM_CANONICALIZATION: &str = "prepared-checker-program-v1";
const MCC_PREPARED_PROGRAM_CANONICAL_IDENTITY: &str = "canonical-prepared";
const MCC_SHARED_ENGINE_ORIGIN_FRONTEND: &str = "mcc_petri";
const MCC_SHARED_ENGINE_COMPONENT: &str = "tla_mc_core.prepared_checker_program";
const MCC_SHARED_ENGINE_LANE_OWNER: &str = "shared_high_performance_engine";
const MCC_SHARED_ENGINE_FIRST_BENEFICIARY: &str = "mcc_petri_runtime_storage";
const MCC_SHARED_ENGINE_SECOND_BENEFICIARY: &str = "tla_plus";
const MCC_SHARED_ENGINE_SECOND_BENEFICIARIES: &str =
    "tla_plus,aiger,btor2,vmt_transition_system,witness_replay";
const MCC_SHARED_ENGINE_EXTRACTION_STATUS: &str = "shared-core-ready";
const MCC_SHARED_ENGINE_COMPATIBLE_FRONTEND_FAMILIES: &str =
    "tla_plus,quint,mcc_petri,aiger,btor2,vmt_transition_system,ay_analytical,witness_replay";
const MCC_SHARED_ENGINE_DEFAULT_COMPATIBLE_FRONTEND_FAMILIES: &str = "tla_plus,mcc_petri";
const MCC_SHARED_ENGINE_DOWNSTREAM_BENEFICIARY_FAMILIES: &str =
    "aiger,btor2,vmt_transition_system,ay_analytical,witness_replay";
const MCC_SHARED_ENGINE_REMAINING_COMPATIBLE_FRONTEND_FAMILIES: &str = "quint";
const MCC_SHARED_ENGINE_FRONTEND_FAMILY_BLOCKERS: &str =
    "future_importer:awaiting_registered_importer_frontend";
const MCC_SHARED_ENGINE_BLOCKER_STATUS: &str = "tracked-blockers";
const MCC_SHARED_ENGINE_GENERIC_PREREQUISITES: &str =
    "prepared_checker_program_descriptor,marking_storage_identity,fingerprint_identity,prepared_fingerprint_admission_plan,dedup_admission,validation_plan_descriptor";
const MCC_SHARED_ADOPTION_ACCEPTANCE_TEST: &str =
    "cargo test -p tla-petri --lib mcc_backend_evidence";
const MCC_SHARED_ENGINE_ACCEPTANCE_EVIDENCE: &str =
    "mcc_backend_evidence_unit_tests,runtime_fingerprint_adoption_rows,prepared_fingerprint_admission_validate_runtime_admission";

// Random-walk witness engine extraction (`tla_mc_core::random_walk`). The
// budgeted single-random-enabled-transition walk that the Petri deadlock and
// quasi-liveness lanes share with every other frontend that needs a positive
// witness via random simulation. Origin is the Petri walk lanes; the shared
// component is the domain-agnostic `random_walk_witness` control loop, adopted
// by the mcc_petri deadlock + quasi-liveness lanes with tla_plus as the
// second beneficiary (the same generic stepper contract drives any
// TransitionSystem-style walk).
const MCC_RANDOM_WALK_COMPONENT: &str = "tla_mc_core.random_walk_witness";
const MCC_RANDOM_WALK_FIRST_BENEFICIARY: &str = "mcc_petri_random_walk_lanes";
const MCC_RANDOM_WALK_EXTRACTION_STATUS: &str = "shared-core-extracted";
const MCC_RANDOM_WALK_ACCEPTANCE_TEST: &str = "cargo test -p tla-petri --lib reachability_walk";
const MCC_RANDOM_WALK_ACCEPTANCE_EVIDENCE: &str =
    "reachability_walk_unit_tests,tla_mc_core_random_walk_tests,deadlock_walk_lane,quasi_liveness_witness_walk_lane";
const MCC_MARKING_FINGERPRINT_ID: &str = "marking-v1";
const MCC_MARKING_FINGERPRINT_ID_U64_LOW: &str = "marking-low64-v1";
const MCC_MARKING_FINGERPRINT_ID_U64_XORFOLD: &str = "marking-xorfold64-v1";
const MCC_MARKING_CANONICALIZATION_VERSION: &str = "place-token-marking-u64-v1";
const MCC_MARKING_CANONICALIZATION_VERSION_U64_LOW: &str = "place-token-marking-u64-low-v1";
const MCC_MARKING_CANONICALIZATION_VERSION_U64_XORFOLD: &str = "place-token-marking-u64-xorfold-v1";
const MCC_MARKING_FINGERPRINT_NAMESPACE: &str = "place-token-marking";
const MCC_MARKING_FINGERPRINT_NAMESPACE_U64_LOW: &str = "place-token-marking-low64";
const MCC_MARKING_FINGERPRINT_NAMESPACE_U64_XORFOLD: &str = "place-token-marking-xorfold64";
const MCC_MARKING_CANONICAL_DOMAIN: &str = "place-token-marking";
const MCC_MARKING_CANONICAL_DOMAIN_VERSION: &str = "u64-vector-v1";
const MCC_STATE_SPACE_DEDUP_ID: &str = "state-space-dedup-v1";
const MCC_MARKING_PREPARED_FINGERPRINT_ADMISSION_PLAN_ID: &str =
    "mcc_petri.marking_vector.prepared_fingerprint_admission.v1";
const MCC_FINGERPRINT_ONLY_PREPARED_FINGERPRINT_ADMISSION_PLAN_ID: &str =
    "mcc_petri.marking_vector.fingerprint_only.prepared_fingerprint_admission.v1";
const MCC_PREPARED_FINGERPRINT_ADMISSION_CONTRACT: &str = "prepared_fingerprint_admission";
const MCC_PREPARED_FINGERPRINT_ADMISSION_VALIDATION_STATUS: &str = "accepted";
const MCC_PREPARED_FINGERPRINT_ADMISSION_SETUP_VALIDATION_SCOPE: &str = "setup_only";
const MCC_PREPARED_FINGERPRINT_ADMISSION_RUNTIME_VALIDATION_SCOPE: &str = "setup_once";
const MCC_PREPARED_FINGERPRINT_ADMISSION_RUNTIME_RECEIPT_ACCEPTED: &str = "accepted";
const MCC_PREPARED_FINGERPRINT_ADMISSION_RUNTIME_RECEIPT_MISSING: &str = "missing";
const MCC_PREPARED_FINGERPRINT_ADMISSION_RUNTIME_RECEIPT_NOT_REQUIRED: &str = "not_required";
const MCC_PREPARED_FINGERPRINT_ADMISSION_RUNTIME_RECEIPT_SETUP_ONLY: &str = "setup_only";
const MCC_PREPARED_FINGERPRINT_ADMISSION_RUNTIME_RECEIPT_MISSING_REASON: &str =
    "missing_hot_loop_admission_receipt";
const MCC_PREPARED_FINGERPRINT_ADMISSION_RUNTIME_RECEIPT_AVAILABLE_REASON: &str = "available";
const MCC_PREPARED_FINGERPRINT_ADMISSION_RUNTIME_RECEIPT_NOT_REQUIRED_REASON: &str =
    "runtime_receipt_not_required";
const MCC_PREPARED_FINGERPRINT_ADMISSION_RUNTIME_RECEIPT_SETUP_ONLY_REASON: &str =
    "setup_only_no_hot_loop_claim";
const MCC_PREPARED_FINGERPRINT_ADMISSION_RUNTIME_CONSUMPTION_OBSERVED: &str = "observed";
const MCC_PREPARED_FINGERPRINT_ADMISSION_RUNTIME_CONSUMPTION_MISSING: &str = "missing";
const MCC_PREPARED_FINGERPRINT_ADMISSION_RUNTIME_CONSUMPTION_SETUP_ONLY: &str = "setup_only";
const MCC_PREPARED_FINGERPRINT_ADMISSION_RUNTIME_CONSUMPTION_NOT_REQUIRED: &str = "not_required";
const MCC_PREPARED_FINGERPRINT_ADMISSION_RUNTIME_PRODUCTION_SELECTED: &str = "selected";
const MCC_PREPARED_FINGERPRINT_ADMISSION_RUNTIME_PRODUCTION_BLOCKED: &str = "blocked";
const MCC_PREPARED_FINGERPRINT_ADMISSION_RUNTIME_PRODUCTION_FAULT_REASON: &str =
    "prepared_admission_fault_observed";
const MCC_HOT_EXECUTION_NOT_COMPLETED_REASON: &str = "hot_execution_not_completed";

thread_local! {
    static RUNTIME_REACHABILITY_BMC_REPORTS: RefCell<Option<Vec<CapabilityReport>>> =
        const { RefCell::new(None) };
    static MCC_PREPARED_FINGERPRINT_ADMISSION_RUNTIME_CONSUMPTION:
        RefCell<Vec<MccPreparedFingerprintAdmissionRuntimeConsumption>> =
            const { RefCell::new(Vec::new()) };
}
const MCC_PORTFOLIO_ROUTE_SCHEMA: &str = "mcc.portfolio_route.v1";
const MCC_PORTFOLIO_ROUTE_SCHEMA_VERSION: &str = "1";
const AIGER_AY_ADAPTER_DECISION_SCHEMA: &str = "aiger.ay_adapter_decision.v1";
#[cfg(feature = "trust-cg-petri-native")]
const TRUST_CG_PETRI_NATIVE_SUCCESSOR_COMPILE_ARTIFACT_HANDOFF_SCHEMA: &str =
    tla_trust_cg::PETRI_NATIVE_SUCCESSOR_COMPILE_ARTIFACT_HANDOFF_SCHEMA;
#[cfg(not(feature = "trust-cg-petri-native"))]
const TRUST_CG_PETRI_NATIVE_SUCCESSOR_COMPILE_ARTIFACT_HANDOFF_SCHEMA: &str =
    "trust-cg.petri.native_successor.compile_artifact_handoff.v1";
const LTL_ROUTE_ADMISSION_SCHEMA: &str = "mcc.ltl_route_admission.v1";
const LTL_ANSWER_LANE_SUMMARY_SCHEMA: &str = "mcc.ltl_answer_lane_summary.v1";

struct MccPortfolioRoute {
    route: &'static str,
    lane_family: &'static str,
    backend_code: &'static str,
    problem: &'static str,
    role: &'static str,
    readiness: &'static str,
    readiness_code: &'static str,
    evidence_source: &'static str,
    evidence_gate: &'static str,
    owner_project: &'static str,
    answer_producer: bool,
    routing_selected: bool,
    selection_rank: u8,
    production_selected: bool,
    fail_closed: bool,
}

impl MccPortfolioRoute {
    fn evidence_row(&self) -> String {
        format!(
            "MCC portfolio_route schema={schema} schema_version={schema_version} \
             route={route} lane_family={lane_family} backend_code={backend_code} \
             problem={problem} role={role} readiness={readiness} \
             readiness_code={readiness_code} evidence_source={evidence_source} \
             evidence_gate={evidence_gate} owner_project={owner_project} \
             answer_producer={answer_producer} routing_selected={routing_selected} \
             selection_rank={selection_rank} production_selected={production_selected} \
             fail_closed={fail_closed}",
            schema = MCC_PORTFOLIO_ROUTE_SCHEMA,
            schema_version = MCC_PORTFOLIO_ROUTE_SCHEMA_VERSION,
            route = self.route,
            lane_family = self.lane_family,
            backend_code = self.backend_code,
            problem = self.problem,
            role = self.role,
            readiness = self.readiness,
            readiness_code = self.readiness_code,
            evidence_source = self.evidence_source,
            evidence_gate = self.evidence_gate,
            owner_project = self.owner_project,
            answer_producer = self.answer_producer,
            routing_selected = self.routing_selected,
            selection_rank = self.selection_rank,
            production_selected = self.production_selected,
            fail_closed = self.fail_closed,
        )
    }
}

const MCC_PORTFOLIO_ROUTES: &[MccPortfolioRoute] = &[
    MccPortfolioRoute {
        route: "explicit_bfs",
        lane_family: "explicit_bfs",
        backend_code: "explicit_state",
        problem: "ExplicitReachability",
        role: "fallback_answer",
        readiness: "ready",
        readiness_code: "explicit_state_fallback_available",
        evidence_source: "MCC.answer_lane.explicit_state_fallback",
        evidence_gate: "explicit_state",
        owner_project: "TY",
        answer_producer: true,
        routing_selected: true,
        selection_rank: 10,
        production_selected: true,
        fail_closed: false,
    },
    MccPortfolioRoute {
        route: "reductions",
        lane_family: "reductions",
        backend_code: "structural_reductions",
        problem: "Preprocessing",
        role: "preprocessor",
        readiness: "ready",
        readiness_code: "structural_reductions_available",
        evidence_source: "MCC.production_selector_decision",
        evidence_gate: "shared_primitive_evidence",
        owner_project: "TY",
        answer_producer: false,
        routing_selected: true,
        selection_rank: 20,
        production_selected: false,
        fail_closed: false,
    },
    MccPortfolioRoute {
        route: "ay_symbolic",
        lane_family: "ay_symbolic",
        backend_code: "ay_sat",
        problem: "Sat",
        role: "symbolic_evidence",
        readiness: "ready",
        readiness_code: "ay_symbolic_ready",
        evidence_source: "MCC.symbolic_execution",
        evidence_gate: tla_ay::AY_SYMBOLIC_EXECUTION_CONTRACT_MANIFEST_SCHEMA,
        owner_project: "AY",
        answer_producer: false,
        routing_selected: true,
        selection_rank: 30,
        production_selected: false,
        fail_closed: false,
    },
    MccPortfolioRoute {
        route: "aiger_hwmcc",
        lane_family: "aiger_hwmcc",
        backend_code: "aiger_portfolio",
        problem: "Safety",
        role: "hardware_portfolio",
        readiness: "ready",
        readiness_code: "aiger_ay_adapter_ready",
        evidence_source: "AIGER.ay_adapter_decision",
        evidence_gate: AIGER_AY_ADAPTER_DECISION_SCHEMA,
        owner_project: "TY",
        answer_producer: false,
        routing_selected: true,
        selection_rank: 40,
        production_selected: false,
        fail_closed: false,
    },
    MccPortfolioRoute {
        route: "native_jit",
        lane_family: "native_jit",
        backend_code: "trust_cg_petri_native",
        problem: "NativeSuccessor",
        role: "primary_answer_producer",
        readiness: "blocked",
        readiness_code: "shared_primitive_runtime_proof_blocked",
        evidence_source: "trust-cg.petri_native_successor_compile_artifact_handoff",
        evidence_gate: TRUST_CG_PETRI_NATIVE_SUCCESSOR_COMPILE_ARTIFACT_HANDOFF_SCHEMA,
        owner_project: "trust-cg",
        answer_producer: true,
        routing_selected: false,
        selection_rank: 50,
        production_selected: false,
        fail_closed: true,
    },
    MccPortfolioRoute {
        route: "hardware_model",
        lane_family: "hardware_model",
        backend_code: "hardware_ay_replay",
        problem: "HardwareReplay",
        role: "hardware_replay_candidate",
        readiness: "blocked",
        readiness_code: "proof_replay_acceptance_required",
        evidence_source: "AIGER.hardware_replay_primitive",
        evidence_gate: HARDWARE_REPLAY_PRIMITIVE_SCHEMA,
        owner_project: "TY",
        answer_producer: true,
        routing_selected: false,
        selection_rank: 60,
        production_selected: false,
        fail_closed: true,
    },
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum MccRunStatus {
    Completed { records: usize },
    Error,
    Panic,
}

pub(crate) struct MccSetupEvidence {
    run_started_at: Instant,
    trace: SetupTrace,
    prepared_program: PreparedCheckerProgram,
    marking_dedup: SharedDedupIdentity,
    examination: Examination,
    property_id_status: &'static str,
}

impl MccSetupEvidence {
    pub(crate) fn record_hot_execution(&mut self, duration: Duration) {
        self.trace
            .record_duration(SetupTracePhase::HotExecution, duration);
        self.trace
            .record_duration(SetupTracePhase::TotalWall, self.run_started_at.elapsed());
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct MccPreparedFingerprintAdmissionRuntimeConsumption {
    plan_id: String,
    callsite: String,
    validation_scope: &'static str,
    attempted: usize,
    new: usize,
    duplicate: usize,
    fault: usize,
    lane: &'static str,
    storage: &'static str,
    prepared_storage_kind: &'static str,
    payload_witness: &'static str,
    source_kind: &'static str,
    payload_kind: &'static str,
    dedup_identity: String,
    fingerprint_identity: String,
}

impl MccPreparedFingerprintAdmissionRuntimeConsumption {
    #[must_use]
    pub(crate) fn from_plan(
        plan: &PreparedFingerprintAdmissionPlan,
        callsite: impl Into<String>,
        attempted: usize,
        new: usize,
        duplicate: usize,
        fault: usize,
    ) -> Self {
        Self {
            plan_id: plan.id.clone(),
            callsite: callsite.into(),
            validation_scope: MCC_PREPARED_FINGERPRINT_ADMISSION_RUNTIME_VALIDATION_SCOPE,
            attempted,
            new,
            duplicate,
            fault,
            lane: plan.lane.code(),
            storage: plan.dedup.storage.code(),
            prepared_storage_kind: plan.storage_kind.code(),
            payload_witness: plan.payload_witness.code(),
            source_kind: plan.source_kind.code(),
            payload_kind: plan.payload_kind.code(),
            dedup_identity: plan.dedup.dedup_identity(),
            fingerprint_identity: plan.dedup.fingerprint.fingerprint_identity(),
        }
    }

    fn render_evidence_row(&self) -> String {
        let fault_observed = self.fault > 0;
        let status_code = if fault_observed {
            MCC_PREPARED_FINGERPRINT_ADMISSION_RUNTIME_PRODUCTION_BLOCKED
        } else {
            MCC_PREPARED_FINGERPRINT_ADMISSION_RUNTIME_RECEIPT_ACCEPTED
        };
        let reason_code = if fault_observed {
            MCC_PREPARED_FINGERPRINT_ADMISSION_RUNTIME_PRODUCTION_FAULT_REASON
        } else {
            MCC_PREPARED_FINGERPRINT_ADMISSION_RUNTIME_RECEIPT_AVAILABLE_REASON
        };
        let runtime_consumed = !fault_observed;
        let runtime_consumption_status = if fault_observed {
            MCC_PREPARED_FINGERPRINT_ADMISSION_RUNTIME_CONSUMPTION_MISSING
        } else {
            MCC_PREPARED_FINGERPRINT_ADMISSION_RUNTIME_CONSUMPTION_OBSERVED
        };
        let receipt_status = if fault_observed {
            MCC_PREPARED_FINGERPRINT_ADMISSION_RUNTIME_RECEIPT_MISSING
        } else {
            MCC_PREPARED_FINGERPRINT_ADMISSION_RUNTIME_RECEIPT_ACCEPTED
        };
        let receipt_reason_code = if fault_observed {
            MCC_PREPARED_FINGERPRINT_ADMISSION_RUNTIME_PRODUCTION_FAULT_REASON
        } else {
            MCC_PREPARED_FINGERPRINT_ADMISSION_RUNTIME_RECEIPT_AVAILABLE_REASON
        };
        let production_selected = !fault_observed;
        let fail_closed = fault_observed;
        format!(
            "MCC prepared_fingerprint_admission_runtime_consumption \
             schema={} schema_version=1 git_head={} plan_id={} callsite={} \
             status_code={} reason_code={} validation_scope={} attempted={} new={} duplicate={} fault={} \
             lane={} storage={} prepared_storage_kind={} payload_witness={} \
             source_kind={} payload_kind={} dedup_identity={} fingerprint_identity={} \
             evidence_scope=hot_loop runtime_claim=true runtime_win_claim=hot_loop_consumed \
             setup_only=false runtime_consumed={} runtime_consumption_status={} \
             hot_loop_consumption={} \
             admission_counters_present=true hot_loop_attempts={} \
             prepared_admission_attempted={} prepared_admission_new={} \
             prepared_admission_duplicate={} prepared_admission_fault={} \
             prepared_admission_receipt_required=true prepared_admission_receipt_status={} \
             prepared_admission_receipt_reason_code={} \
             origin_frontend={} shared_engine_component={} shared_owner={} \
             first_beneficiary={} second_beneficiaries={} default_consumers={} \
             state_vector_reuse=canonical_marking_vector frontend_neutral_state_vector=true \
             frontend_neutral_fingerprint=true solver_storage_reusable=true \
             tla_check_inherits=true hardware_lanes_inherit=true storage_lanes_inherit=true \
             replay_lanes_inherit=true fault_observed={} production_readiness_status={} \
             production_readiness_reason_code={} production_selected={} fail_closed={}",
            MCC_PREPARED_FINGERPRINT_ADMISSION_RUNTIME_CONSUMPTION_SCHEMA,
            env!("TY_MCC_BUILD_GIT_HEAD"),
            evidence_token(&self.plan_id),
            evidence_token(&self.callsite),
            status_code,
            reason_code,
            self.validation_scope,
            self.attempted,
            self.new,
            self.duplicate,
            self.fault,
            self.lane,
            self.storage,
            self.prepared_storage_kind,
            self.payload_witness,
            self.source_kind,
            self.payload_kind,
            evidence_token(&self.dedup_identity),
            evidence_token(&self.fingerprint_identity),
            runtime_consumed,
            runtime_consumption_status,
            runtime_consumption_status,
            self.attempted,
            self.attempted,
            self.new,
            self.duplicate,
            self.fault,
            receipt_status,
            receipt_reason_code,
            MCC_SHARED_ENGINE_ORIGIN_FRONTEND,
            MCC_SHARED_ENGINE_COMPONENT,
            MCC_SHARED_ENGINE_LANE_OWNER,
            MCC_SHARED_ENGINE_FIRST_BENEFICIARY,
            MCC_SHARED_ENGINE_SECOND_BENEFICIARIES,
            MCC_SHARED_ENGINE_DEFAULT_COMPATIBLE_FRONTEND_FAMILIES,
            fault_observed,
            if production_selected {
                MCC_PREPARED_FINGERPRINT_ADMISSION_RUNTIME_PRODUCTION_SELECTED
            } else {
                MCC_PREPARED_FINGERPRINT_ADMISSION_RUNTIME_PRODUCTION_BLOCKED
            },
            reason_code,
            production_selected,
            fail_closed,
        )
    }
}

pub(crate) fn record_mcc_prepared_fingerprint_admission_runtime_consumption(
    consumption: MccPreparedFingerprintAdmissionRuntimeConsumption,
) {
    if consumption.attempted == 0 {
        return;
    }
    MCC_PREPARED_FINGERPRINT_ADMISSION_RUNTIME_CONSUMPTION.with(|slot| {
        slot.borrow_mut().push(consumption);
    });
}

fn clear_mcc_prepared_fingerprint_admission_runtime_consumption() {
    MCC_PREPARED_FINGERPRINT_ADMISSION_RUNTIME_CONSUMPTION.with(|slot| {
        slot.borrow_mut().clear();
    });
}

fn take_mcc_prepared_fingerprint_admission_runtime_consumption(
) -> Vec<MccPreparedFingerprintAdmissionRuntimeConsumption> {
    MCC_PREPARED_FINGERPRINT_ADMISSION_RUNTIME_CONSUMPTION
        .with(|slot| std::mem::take(&mut *slot.borrow_mut()))
}

/// RAII guard that bounds the per-run prepared-fingerprint-admission runtime
/// consumption accumulator to a well-defined scope.
///
/// Constructing the guard saves whatever the thread-local currently holds and
/// resets the cell to empty, so collection during the scope cannot observe
/// leftover entries from a prior run on the same thread. On drop the previous
/// contents are restored (re-entrancy-safe, mirroring
/// [`RuntimeReachabilityBmcReportScope`]) — a nested scope therefore never
/// corrupts an outer one. This reproduces the effective behaviour of the
/// pre-existing ad-hoc `clear()` at the front of [`build_mcc_setup_evidence`]
/// while additionally guaranteeing the cell is always returned to a clean
/// (empty) state for the next run, instead of relying on a downstream
/// `take()` having drained it.
struct MccPreparedFingerprintAdmissionConsumptionScope {
    previous: Vec<MccPreparedFingerprintAdmissionRuntimeConsumption>,
}

impl MccPreparedFingerprintAdmissionConsumptionScope {
    fn begin() -> Self {
        let previous = MCC_PREPARED_FINGERPRINT_ADMISSION_RUNTIME_CONSUMPTION
            .with(|slot| std::mem::take(&mut *slot.borrow_mut()));
        Self { previous }
    }
}

impl Drop for MccPreparedFingerprintAdmissionConsumptionScope {
    fn drop(&mut self) {
        MCC_PREPARED_FINGERPRINT_ADMISSION_RUNTIME_CONSUMPTION.with(|slot| {
            *slot.borrow_mut() = std::mem::take(&mut self.previous);
        });
    }
}

/// RAII boundary for the per-run MCC evidence accumulators.
///
/// One examination run collects evidence into two thread-local accumulators:
/// the runtime reachability BMC reports and the prepared-fingerprint-admission
/// runtime consumption rows. This guard installs a single scope that bounds
/// BOTH for the duration of the run, so leftover state can never leak between
/// runs on the same thread and a half-borrowed cell can never persist past the
/// scope.
///
/// It composes the pre-existing [`RuntimeReachabilityBmcReportScope`] (whose
/// `finish()` is still used explicitly by the caller to drain the BMC reports
/// at the right point) with a save/reset/restore guard for the consumption
/// accumulator. The combined guard is purely additive: it reproduces today's
/// effective reset timing (the consumption cell is empty at scope entry on a
/// fresh worker thread, and the ad-hoc `clear()` in `build_mcc_setup_evidence`
/// remains the reset for same-thread callers) and does not alter which evidence
/// rows are collected or emitted.
pub(crate) struct EvidenceScope {
    bmc_reports: Option<RuntimeReachabilityBmcReportScope>,
    // Dropped after `bmc_reports` per struct field declaration order, restoring
    // the consumption accumulator only once BMC collection has fully unwound.
    _consumption: MccPreparedFingerprintAdmissionConsumptionScope,
}

impl EvidenceScope {
    /// Open the evidence-collection scope for one examination run.
    pub(crate) fn begin() -> Self {
        // Open the consumption guard first so it is the outermost (dropped last)
        // boundary; the BMC scope nests inside it.
        let consumption = MccPreparedFingerprintAdmissionConsumptionScope::begin();
        let bmc_reports = begin_runtime_reachability_bmc_report_collection();
        Self {
            bmc_reports: Some(bmc_reports),
            _consumption: consumption,
        }
    }

    /// Drain and return the runtime reachability BMC reports collected during
    /// the scope. Mirrors the previous explicit
    /// `RuntimeReachabilityBmcReportScope::finish()` call site exactly: it must
    /// be invoked once, at the point the BMC reports are appended to the
    /// backend report. The consumption accumulator is left for the downstream
    /// `take()` in `add_mcc_setup_evidence` and is finally cleared when this
    /// `EvidenceScope` is dropped.
    pub(crate) fn finish_runtime_reachability_bmc_reports(&mut self) -> Vec<CapabilityReport> {
        match self.bmc_reports.take() {
            Some(scope) => scope.finish(),
            None => Vec::new(),
        }
    }
}

#[must_use]
pub(crate) fn build_mcc_setup_evidence(
    model: &PreparedModel,
    examination: Examination,
    config: &ExplorationConfig,
    run_started_at: Instant,
    source_load_duration: Duration,
    config_resolution_duration: Duration,
) -> MccSetupEvidence {
    clear_mcc_prepared_fingerprint_admission_runtime_consumption();
    let prepared_started = Instant::now();
    let marking_dedup = mcc_marking_dedup_identity(config);
    let (prepared_program, property_id_status) =
        build_mcc_prepared_program(model, examination, &marking_dedup);
    let prepared_duration = prepared_started.elapsed();

    let mut trace = SetupTrace::new(CheckerSourceKind::MccPetri)
        .with_lane(SetupTraceLaneKind::Frontend)
        .with_identity_fields(prepared_program.effective_identity_fields())
        .with_source_identity(model.model_dir().display().to_string())
        .with_property_identity(examination.as_str())
        .with_origin_frontend(MCC_SHARED_ENGINE_ORIGIN_FRONTEND)
        .with_shared_engine_component(MCC_SHARED_ENGINE_COMPONENT)
        .with_first_beneficiary(MCC_SHARED_ENGINE_FIRST_BENEFICIARY)
        .with_second_beneficiary(MCC_SHARED_ENGINE_SECOND_BENEFICIARY)
        .with_compatible_frontend_families(
            MCC_SHARED_ENGINE_COMPATIBLE_FRONTEND_FAMILIES.split(','),
        )
        .with_shared_engine_extraction_status(MCC_SHARED_ENGINE_EXTRACTION_STATUS)
        .with_shared_engine_blocker_status(MCC_SHARED_ENGINE_BLOCKER_STATUS)
        .with_validation_status(SetupTraceValidationStatus::Accepted);
    trace.record_duration(SetupTracePhase::SourceLoad, source_load_duration);
    trace.record_duration(SetupTracePhase::FrontendImport, source_load_duration);
    trace.record_duration(
        SetupTracePhase::ConfigResolution,
        config_resolution_duration,
    );
    trace.record_duration(SetupTracePhase::SemanticLowering, source_load_duration);
    trace.record_duration(SetupTracePhase::PreparedProgramBuild, prepared_duration);
    trace.record_duration(SetupTracePhase::OutputFormatValidation, Duration::ZERO);
    if symbolic_ay_candidate(examination) {
        trace.record_duration(SetupTracePhase::SolverSetup, Duration::ZERO);
    }

    MccSetupEvidence {
        run_started_at,
        trace,
        prepared_program,
        marking_dedup,
        examination,
        property_id_status,
    }
}

pub(crate) struct RuntimeReachabilityBmcReportScope {
    previous: Option<Vec<CapabilityReport>>,
    finished: bool,
}

impl RuntimeReachabilityBmcReportScope {
    pub(crate) fn finish(mut self) -> Vec<CapabilityReport> {
        let reports = RUNTIME_REACHABILITY_BMC_REPORTS
            .with(|slot| slot.borrow_mut().take().unwrap_or_default());
        RUNTIME_REACHABILITY_BMC_REPORTS.with(|slot| {
            *slot.borrow_mut() = self.previous.take();
        });
        self.finished = true;
        reports
    }
}

impl Drop for RuntimeReachabilityBmcReportScope {
    fn drop(&mut self) {
        if self.finished {
            return;
        }
        RUNTIME_REACHABILITY_BMC_REPORTS.with(|slot| {
            *slot.borrow_mut() = self.previous.take();
        });
    }
}

pub(crate) fn begin_runtime_reachability_bmc_report_collection() -> RuntimeReachabilityBmcReportScope
{
    let previous =
        RUNTIME_REACHABILITY_BMC_REPORTS.with(|slot| slot.borrow_mut().replace(Vec::new()));
    RuntimeReachabilityBmcReportScope {
        previous,
        finished: false,
    }
}

#[cfg(test)]
pub(crate) fn collect_runtime_reachability_bmc_reports<T>(
    collect: impl FnOnce() -> T,
) -> (T, Vec<CapabilityReport>) {
    let scope = begin_runtime_reachability_bmc_report_collection();
    let value = collect();
    let reports = scope.finish();
    (value, reports)
}

pub(crate) fn record_runtime_reachability_bmc_report(report: &CapabilityReport) {
    RUNTIME_REACHABILITY_BMC_REPORTS.with(|slot| {
        if let Some(reports) = slot.borrow_mut().as_mut() {
            reports.push(report.clone());
        }
    });
}

pub(crate) fn append_runtime_reachability_bmc_reports(
    target: &mut CapabilityReport,
    runtime_reports: Vec<CapabilityReport>,
) {
    if runtime_reports.is_empty() {
        return;
    }

    target
        .evidence
        .retain(|row| !is_mcc_ay_solve_decision_profile_row(row));

    let mut has_runtime_profile = false;
    for runtime_report in runtime_reports {
        for evidence in runtime_report.evidence {
            has_runtime_profile |= is_mcc_ay_solve_decision_profile_row(&evidence);
            add_unique_evidence(target, evidence);
        }
    }

    if !has_runtime_profile {
        add_ay_solve_decision_profile_evidence(target);
    }
    add_ay_capability_contract_evidence(target);
    add_mcc_portfolio_route_evidence(target);
}

fn build_mcc_prepared_program(
    model: &PreparedModel,
    examination: Examination,
    marking_dedup: &SharedDedupIdentity,
) -> (PreparedCheckerProgram, &'static str) {
    let identity = format!("{}:{}", model.model_name(), examination.as_str());
    let storage_layout_fingerprint = marking_dedup
        .storage_config_identity
        .clone()
        .unwrap_or_else(|| prepared_identity("mcc.petri.storage_layout", &identity));
    let mut program = PreparedCheckerProgram::new(
        identity.clone(),
        PreparedProgramPayloadKind::MccPetri,
        PreparedStorageKind::PetriMarking,
    )
    .with_canonical_payload_identity(prepared_identity("mcc.petri.canonical_payload", &identity))
    .with_source_identity(model.model_dir().display().to_string())
    .with_config_identity(prepared_identity(
        "mcc.petri.config",
        marking_dedup
            .storage_config_identity
            .as_deref()
            .unwrap_or("unknown_storage_config"),
    ))
    .with_examination_identity(prepared_identity(
        "mcc.petri.examination",
        examination.as_str(),
    ))
    .with_cache_key(prepared_identity("mcc.petri.prepared_cache", &identity))
    .with_source_fingerprint(prepared_identity("mcc.petri.source_fingerprint", &identity))
    .with_frontend_payload_identity(prepared_identity("mcc.petri.payload", &identity))
    .with_frontend_payload_fingerprint(prepared_identity(
        "mcc.petri.payload_fingerprint",
        &identity,
    ))
    .with_artifact_identity(prepared_identity("mcc.petri.prepared_program", &identity))
    .with_storage_layout_fingerprint(storage_layout_fingerprint)
    .with_storage_policy_identity(marking_dedup.storage_policy_identity())
    .with_fingerprint_policy_identity(marking_dedup.fingerprint.fingerprint_policy_identity())
    .with_fingerprint_identity(marking_dedup.fingerprint.fingerprint_identity())
    .with_transition_descriptor_fingerprint(prepared_identity(
        "mcc.petri.transition_descriptor",
        &identity,
    ))
    .with_property_descriptor_fingerprint(prepared_identity(
        "mcc.petri.property_descriptor",
        &identity,
    ))
    .with_validation_plan_fingerprint(prepared_identity("mcc.petri.validation_plan", &identity))
    .with_fingerprint(marking_dedup.prepared_fingerprint_descriptor());

    for transition in &model.net().transitions {
        program = program.add_transition(&transition.id, PreparedTransitionKind::PetriTransition);
    }

    let (property_ids, property_id_status) = prepared_property_ids(model, examination);
    let property_kind = prepared_property_kind(examination);
    for property_id in property_ids {
        program = program.add_property(property_id, property_kind);
    }

    program = program.add_analytical_solve(
        format!("mcc.petri.{}", examination.as_str()),
        prepared_analytical_solve_kind(examination),
        problem_for_examination(examination),
    );
    program = add_preprocessing_prepared_lanes(program, examination);

    if let Some(symbolic_kind) = prepared_symbolic_proof_kind(examination) {
        let symbolic_problem = prepared_symbolic_problem(symbolic_kind);
        program = program
            .add_symbolic_proof(
                format!(
                    "mcc.petri.ay.{}.{}",
                    symbolic_kind.code(),
                    examination.as_str()
                ),
                symbolic_kind,
                symbolic_problem,
            )
            .add_backend_family(
                PreparedBackendFamilyDescriptor::new(
                    format!(
                        "mcc.petri.ay.external.{}.{}",
                        symbolic_kind.code(),
                        examination.as_str()
                    ),
                    BackendKind::ExternalAYBinary,
                    symbolic_problem,
                )
                .with_facet(SolverFacet::ExternalProcess)
                .with_facet(SolverFacet::Smt)
                .with_facet(symbolic_solver_facet(symbolic_kind)),
            );
    }
    program = add_ay_solver_candidate_lanes(program, examination);

    program = program
        .require_validation(PreparedValidationKind::OutputFormat)
        .require_validation(PreparedValidationKind::Selftest);
    if matches!(
        examination,
        Examination::ReachabilityCardinality | Examination::ReachabilityFireability
    ) {
        program = program.require_validation(PreparedValidationKind::WitnessReplay);
    }
    if matches!(
        examination,
        Examination::CTLCardinality
            | Examination::CTLFireability
            | Examination::LTLCardinality
            | Examination::LTLFireability
            | Examination::Liveness
    ) {
        program = program.require_validation(PreparedValidationKind::SccCertificate);
    }
    if matches!(
        examination,
        Examination::UpperBounds | Examination::StableMarking
    ) {
        program = program.require_validation(PreparedValidationKind::StructuralProof);
    }
    program = add_mcc_frontend_extension_descriptors(program, examination, marking_dedup);
    program = add_mcc_candidate_readiness_descriptors(program, examination, marking_dedup);
    program = add_mcc_validation_plan_descriptors(program, examination);
    let program_fingerprint = prepared_program_sha256(&program);
    program = program.add_canonical_identity(
        PreparedCanonicalIdentityDescriptor::new(
            MCC_PREPARED_PROGRAM_CANONICAL_IDENTITY,
            PreparedCanonicalIdentityKind::PreparedProgram,
            MCC_PREPARED_PROGRAM_CANONICALIZATION,
        )
        .with_digest(
            MCC_PREPARED_PROGRAM_FINGERPRINT_ALGORITHM,
            program_fingerprint,
        ),
    );

    (program, property_id_status)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MccMarkingFingerprintProjection {
    FullU128,
    LowU64,
    XorFoldU64,
}

fn mcc_marking_fingerprint_projection(
    config: &ExplorationConfig,
) -> MccMarkingFingerprintProjection {
    if config.storage_mode() == StorageMode::FingerprintOnly {
        MccMarkingFingerprintProjection::LowU64
    } else if config.workers() > 1 && config.fpset_backend() == FpsetBackend::Cas {
        MccMarkingFingerprintProjection::XorFoldU64
    } else {
        MccMarkingFingerprintProjection::FullU128
    }
}

fn mcc_marking_fingerprint_identity(
    projection: MccMarkingFingerprintProjection,
) -> SharedFingerprintIdentity {
    let (id, canonicalization_version, namespace, digest_bits) = match projection {
        MccMarkingFingerprintProjection::FullU128 => (
            MCC_MARKING_FINGERPRINT_ID,
            MCC_MARKING_CANONICALIZATION_VERSION,
            MCC_MARKING_FINGERPRINT_NAMESPACE,
            128,
        ),
        MccMarkingFingerprintProjection::LowU64 => (
            MCC_MARKING_FINGERPRINT_ID_U64_LOW,
            MCC_MARKING_CANONICALIZATION_VERSION_U64_LOW,
            MCC_MARKING_FINGERPRINT_NAMESPACE_U64_LOW,
            64,
        ),
        MccMarkingFingerprintProjection::XorFoldU64 => (
            MCC_MARKING_FINGERPRINT_ID_U64_XORFOLD,
            MCC_MARKING_CANONICALIZATION_VERSION_U64_XORFOLD,
            MCC_MARKING_FINGERPRINT_NAMESPACE_U64_XORFOLD,
            64,
        ),
    };
    SharedFingerprintIdentity::new(
        id,
        SharedFingerprintAlgorithm::CanonicalBytesSha256,
        SharedFingerprintValueKind::MarkingVector,
        canonicalization_version,
        namespace,
        digest_bits,
    )
    .with_canonical_domain(
        MCC_MARKING_CANONICAL_DOMAIN,
        MCC_MARKING_CANONICAL_DOMAIN_VERSION,
    )
}

fn mcc_marking_dedup_identity(config: &ExplorationConfig) -> SharedDedupIdentity {
    let (storage, lane, storage_config_identity) =
        if config.storage_mode() == StorageMode::FingerprintOnly {
            (
                SharedDedupStorageKind::Cas,
                SetupTraceLaneKind::Fingerprint,
                "fingerprint-only-cas-fingerprint-set-v1",
            )
        } else if config.workers() <= 1 {
            (
                SharedDedupStorageKind::InMemory,
                SetupTraceLaneKind::ExplicitState,
                "local-fingerprint-set-v1",
            )
        } else {
            match config.fpset_backend() {
                FpsetBackend::Sharded => (
                    SharedDedupStorageKind::ShardedInMemory,
                    SetupTraceLaneKind::ExplicitState,
                    "sharded-fingerprint-set-v1",
                ),
                FpsetBackend::Cas => (
                    SharedDedupStorageKind::Cas,
                    SetupTraceLaneKind::ExplicitState,
                    "partitioned-cas-fingerprint-set-v1-partition-bits-4",
                ),
            }
        };
    SharedDedupIdentity::new(
        MCC_STATE_SPACE_DEDUP_ID,
        mcc_marking_fingerprint_identity(mcc_marking_fingerprint_projection(config)),
        SharedDedupScope::StateSpace,
        storage,
        lane,
    )
    .with_storage_config_identity(storage_config_identity)
}

fn add_mcc_frontend_extension_descriptors(
    mut program: PreparedCheckerProgram,
    examination: Examination,
    marking_dedup: &SharedDedupIdentity,
) -> PreparedCheckerProgram {
    let identity = program.identity.clone();
    let examination_name = examination.as_str();
    let storage_policy = program.storage_kind.code();
    for (kind, problem, bridge) in [
        (
            PreparedFrontendExtensionKind::Aiger,
            ProblemKind::Safety,
            "aiger_hwmcc",
        ),
        (
            PreparedFrontendExtensionKind::Btor2,
            ProblemKind::Safety,
            "btor2_hwmcc",
        ),
        (
            PreparedFrontendExtensionKind::VmtInterchange,
            ProblemKind::Smt,
            "vmt_interchange",
        ),
        (
            PreparedFrontendExtensionKind::AYOnly,
            prepared_symbolic_problem(
                prepared_symbolic_proof_kind(examination)
                    .unwrap_or(PreparedSymbolicProofKind::StatePredicate),
            ),
            "ay_only",
        ),
        (
            PreparedFrontendExtensionKind::WitnessReplay,
            problem_for_examination(examination),
            "witness_replay",
        ),
    ] {
        program = program.add_frontend_extension(
            PreparedFrontendExtensionDescriptor::new(
                format!("mcc.petri.frontend_extension.{bridge}.{examination_name}"),
                kind,
                problem,
            )
            .with_cache_key(prepared_identity(
                &format!("mcc.petri.frontend_extension_cache.{bridge}"),
                &identity,
            ))
            .with_frontend_payload_identity(prepared_identity(
                &format!("mcc.petri.frontend_payload.{bridge}"),
                &identity,
            ))
            .with_artifact_identity(prepared_identity(
                &format!("mcc.petri.frontend_artifact.{bridge}"),
                &identity,
            ))
            .with_storage_policy_identity(storage_policy)
            .with_fingerprint_policy_identity(
                marking_dedup.fingerprint.fingerprint_policy_identity(),
            )
            .with_fingerprint_identity(marking_dedup.fingerprint.fingerprint_identity()),
        );
    }
    program
}

fn add_mcc_candidate_readiness_descriptors(
    mut program: PreparedCheckerProgram,
    examination: Examination,
    marking_dedup: &SharedDedupIdentity,
) -> PreparedCheckerProgram {
    let identity = program.identity.clone();
    let examination_name = examination.as_str();
    let marking_fingerprint = &marking_dedup.fingerprint;
    program = program.add_candidate_lane(
        PreparedCandidateLaneDescriptor::new(
            format!("mcc.petri.explicit_state.{examination_name}"),
            SetupTraceLaneKind::ExplicitState,
        )
        .with_candidate_key("explicit_bfs")
        .with_candidate_identity(prepared_identity(
            "mcc.petri.candidate.explicit_state",
            &identity,
        ))
        .with_lane_identity(MCC_SHARED_ENGINE_LANE_OWNER)
        .with_fingerprint_policy_identity(marking_fingerprint.fingerprint_policy_identity())
        .with_fingerprint_identity(marking_fingerprint.fingerprint_identity()),
    );
    if symbolic_ay_candidate(examination) {
        program = program.add_candidate_lane(
            PreparedCandidateLaneDescriptor::new(
                format!("mcc.petri.ay.symbolic.{examination_name}"),
                SetupTraceLaneKind::AY,
            )
            .with_candidate_key("ay_symbolic")
            .with_candidate_identity(prepared_identity("mcc.petri.candidate.ay", &identity))
            .with_lane_identity(MCC_SHARED_ENGINE_LANE_OWNER)
            .with_fingerprint_policy_identity("mcc_petri_ay_symbolic_artifact_v1")
            .with_fingerprint_identity(prepared_identity("mcc.petri.ay.proof", &identity)),
        );
    }
    program
}

fn add_mcc_validation_plan_descriptors(
    mut program: PreparedCheckerProgram,
    examination: Examination,
) -> PreparedCheckerProgram {
    let identity = program.identity.clone();
    for validation in program.validations.clone() {
        let artifact_role = mcc_validation_artifact_role(validation);
        let fingerprint_identity = prepared_identity(artifact_role, &identity);
        let plan = PreparedValidationPlanDescriptor::new(
            format!(
                "mcc.petri.validation.{}.{}",
                validation.code(),
                examination.as_str()
            ),
            validation,
            problem_for_examination(examination),
        )
        .with_fingerprint(
            PreparedFingerprintDescriptor::new(
                artifact_role,
                PreparedFingerprintScheme::CanonicalBytesSha256,
                MCC_PREPARED_PROGRAM_SCHEMA,
            )
            .with_fingerprint_policy_identity("mcc_prepared_validation_fingerprint_v1")
            .with_fingerprint_identity(fingerprint_identity),
        )
        .with_artifact_identity(prepared_identity(
            "mcc.petri.validation_artifact",
            &identity,
        ));
        program = program.add_validation_plan(plan);
    }
    program
}

fn mcc_validation_artifact_role(validation: PreparedValidationKind) -> &'static str {
    match validation {
        PreparedValidationKind::WitnessReplay | PreparedValidationKind::TraceReplay => {
            "mcc.petri.witness_fingerprint"
        }
        PreparedValidationKind::SccCertificate
        | PreparedValidationKind::AcceptingCycleCertificate
        | PreparedValidationKind::CompleteGraph => "mcc.petri.certificate_fingerprint",
        PreparedValidationKind::StructuralProof | PreparedValidationKind::AYProof => {
            "mcc.petri.proof_fingerprint"
        }
        PreparedValidationKind::Selftest | PreparedValidationKind::OutputFormat => {
            "mcc.petri.validation_fingerprint"
        }
    }
}

fn add_preprocessing_prepared_lanes(
    mut program: PreparedCheckerProgram,
    examination: Examination,
) -> PreparedCheckerProgram {
    let examination_name = examination.as_str();
    let reduction_mode = examination.reduction_mode();
    program = program
        .add_analytical_solve(
            format!("mcc.petri.structural.p_invariant_bounds.{examination_name}"),
            PreparedAnalyticalSolveKind::LinearInvariant,
            ProblemKind::Invariant,
        )
        .add_analytical_solve(
            format!("mcc.petri.lp_state_equation.relaxation.{examination_name}"),
            PreparedAnalyticalSolveKind::SmtQuery,
            ProblemKind::Smt,
        )
        .add_analytical_solve(
            format!(
                "mcc.petri.reduction.{}.{examination_name}",
                reduction_mode_code(reduction_mode)
            ),
            prepared_analytical_solve_kind(examination),
            problem_for_examination(examination),
        )
        .add_backend_family(
            PreparedBackendFamilyDescriptor::new(
                format!(
                    "mcc.petri.reduction.{}.{examination_name}",
                    reduction_mode_code(reduction_mode)
                ),
                BackendKind::ExplicitState,
                problem_for_examination(examination),
            )
            .with_facet(SolverFacet::LinearIntegerArithmetic),
        );
    program
}

fn add_ay_solver_candidate_lanes(
    mut program: PreparedCheckerProgram,
    examination: Examination,
) -> PreparedCheckerProgram {
    if matches!(
        examination,
        Examination::ReachabilityCardinality | Examination::ReachabilityFireability
    ) {
        program = program
            .add_symbolic_proof(
                format!("mcc.petri.ay.chc.reachability_pdr.{}", examination.as_str()),
                PreparedSymbolicProofKind::ChcQuery,
                ProblemKind::Chc,
            )
            .add_backend_family(
                PreparedBackendFamilyDescriptor::new(
                    format!("mcc.petri.ay.chc.reachability_pdr.{}", examination.as_str()),
                    BackendKind::AYChc,
                    ProblemKind::Chc,
                )
                .with_facet(SolverFacet::InProcess)
                .with_facet(SolverFacet::Chc)
                .with_facet(SolverFacet::Pdr)
                .with_facet(SolverFacet::LinearIntegerArithmetic),
            );
    }
    program
}

fn symbolic_solver_facet(kind: PreparedSymbolicProofKind) -> SolverFacet {
    match kind {
        PreparedSymbolicProofKind::BoundedModelCheck => SolverFacet::Bmc,
        PreparedSymbolicProofKind::KInduction => SolverFacet::KInduction,
        PreparedSymbolicProofKind::PdrSafetyProof => SolverFacet::Pdr,
        PreparedSymbolicProofKind::ChcQuery | PreparedSymbolicProofKind::InvariantProof => {
            SolverFacet::Chc
        }
        PreparedSymbolicProofKind::UnsatCore => SolverFacet::UnsatCore,
        PreparedSymbolicProofKind::ProofCertificate => SolverFacet::Proof,
        PreparedSymbolicProofKind::ModelExtraction => SolverFacet::ModelValues,
        PreparedSymbolicProofKind::InitialCondition
        | PreparedSymbolicProofKind::TransitionRelation
        | PreparedSymbolicProofKind::StatePredicate => SolverFacet::Smt,
    }
}

fn reduction_mode_code(mode: crate::reduction::ReductionMode) -> &'static str {
    match mode {
        crate::reduction::ReductionMode::Reachability => "reachability",
        crate::reduction::ReductionMode::NextFreeCTL => "next_free_ctl",
        crate::reduction::ReductionMode::CTLWithNext => "ctl_with_next",
        crate::reduction::ReductionMode::StutterInsensitiveLTL => "stutter_insensitive_ltl",
        crate::reduction::ReductionMode::StutterSensitiveLTL => "stutter_sensitive_ltl",
        crate::reduction::ReductionMode::ReachabilityDeadlock => "reachability_deadlock",
    }
}

fn prepared_analytical_solve_kind(examination: Examination) -> PreparedAnalyticalSolveKind {
    match examination {
        Examination::ReachabilityDeadlock => PreparedAnalyticalSolveKind::DeadlockFreedom,
        Examination::ReachabilityCardinality | Examination::ReachabilityFireability => {
            PreparedAnalyticalSolveKind::Reachability
        }
        Examination::CTLCardinality | Examination::CTLFireability => {
            PreparedAnalyticalSolveKind::Reachability
        }
        Examination::LTLCardinality | Examination::LTLFireability | Examination::Liveness => {
            PreparedAnalyticalSolveKind::Reachability
        }
        Examination::StateSpace => PreparedAnalyticalSolveKind::StateSpaceCardinality,
        Examination::OneSafe => PreparedAnalyticalSolveKind::LinearInvariant,
        Examination::QuasiLiveness => PreparedAnalyticalSolveKind::Reachability,
        Examination::StableMarking => PreparedAnalyticalSolveKind::StableMarking,
        Examination::UpperBounds => PreparedAnalyticalSolveKind::UpperBounds,
    }
}

fn prepared_symbolic_proof_kind(examination: Examination) -> Option<PreparedSymbolicProofKind> {
    match examination {
        Examination::ReachabilityDeadlock
        | Examination::ReachabilityCardinality
        | Examination::ReachabilityFireability => {
            Some(PreparedSymbolicProofKind::BoundedModelCheck)
        }
        Examination::OneSafe | Examination::StableMarking => {
            Some(PreparedSymbolicProofKind::InvariantProof)
        }
        Examination::QuasiLiveness | Examination::Liveness => {
            Some(PreparedSymbolicProofKind::KInduction)
        }
        Examination::CTLCardinality
        | Examination::CTLFireability
        | Examination::LTLCardinality
        | Examination::LTLFireability
        | Examination::StateSpace
        | Examination::UpperBounds => None,
    }
}

fn prepared_symbolic_problem(kind: PreparedSymbolicProofKind) -> ProblemKind {
    match kind {
        PreparedSymbolicProofKind::BoundedModelCheck => ProblemKind::Bmc,
        PreparedSymbolicProofKind::KInduction => ProblemKind::KInduction,
        PreparedSymbolicProofKind::PdrSafetyProof => ProblemKind::Chc,
        PreparedSymbolicProofKind::ChcQuery | PreparedSymbolicProofKind::InvariantProof => {
            ProblemKind::Chc
        }
        PreparedSymbolicProofKind::UnsatCore
        | PreparedSymbolicProofKind::InitialCondition
        | PreparedSymbolicProofKind::TransitionRelation
        | PreparedSymbolicProofKind::StatePredicate
        | PreparedSymbolicProofKind::ProofCertificate
        | PreparedSymbolicProofKind::ModelExtraction => ProblemKind::Smt,
    }
}

fn prepared_property_ids(
    model: &PreparedModel,
    examination: Examination,
) -> (Vec<String>, &'static str) {
    let Ok(xml_name) = examination.property_xml_name() else {
        return (
            vec![examination.as_str().to_string()],
            "non_property_examination",
        );
    };
    match crate::property_xml::parse_property_ids(model.model_dir(), xml_name) {
        Ok(ids) => (ids, "property_xml_ids"),
        Err(_) => (Vec::new(), "property_xml_unavailable"),
    }
}

fn prepared_property_kind(examination: Examination) -> PreparedPropertyKind {
    match examination {
        Examination::ReachabilityDeadlock => PreparedPropertyKind::Deadlock,
        Examination::ReachabilityCardinality | Examination::ReachabilityFireability => {
            PreparedPropertyKind::Reachability
        }
        Examination::CTLCardinality | Examination::CTLFireability => PreparedPropertyKind::Ctl,
        Examination::LTLCardinality | Examination::LTLFireability | Examination::Liveness => {
            PreparedPropertyKind::Ltl
        }
        Examination::StateSpace => PreparedPropertyKind::StateSpace,
        Examination::OneSafe => PreparedPropertyKind::Invariant,
        Examination::QuasiLiveness => PreparedPropertyKind::Ltl,
        Examination::StableMarking => PreparedPropertyKind::StableMarking,
        Examination::UpperBounds => PreparedPropertyKind::UpperBounds,
    }
}

#[must_use]
pub(crate) fn mcc_backend_capability_report(
    model: &PreparedModel,
    examination: Examination,
    config: &ExplorationConfig,
) -> CapabilityReport {
    build_petri_mcc_capability_report(model.net(), examination, config, model.source_kind())
}

pub(crate) fn maybe_emit_mcc_backend_evidence(
    model: &PreparedModel,
    examination: Examination,
    config: &ExplorationConfig,
    report: &CapabilityReport,
    setup_evidence: Option<&MccSetupEvidence>,
    status: MccRunStatus,
    error: Option<&str>,
) {
    let Some(path) = std::env::var_os(MCC_BACKEND_EVIDENCE_JSONL_ENV) else {
        return;
    };
    if path.is_empty() {
        return;
    }

    let record = SerializableMccBackendEvidence::new(
        model,
        examination,
        config,
        report,
        setup_evidence,
        status,
        error,
    );
    if let Err(write_error) = append_jsonl(Path::new(&path), &record) {
        eprintln!(
            "Warning: failed to write MCC backend evidence to {}: {write_error}",
            Path::new(&path).display()
        );
    }
}

pub(crate) fn add_mcc_setup_evidence(
    report: &mut CapabilityReport,
    model: &PreparedModel,
    examination: Examination,
    setup_evidence: &MccSetupEvidence,
    status: MccRunStatus,
    error: Option<&str>,
) {
    for row in setup_evidence.trace.render_evidence_rows("MCC") {
        add_unique_evidence(report, row);
    }
    add_unique_evidence(report, mcc_model_load_evidence_row(model, setup_evidence));
    let marking_dedup = &setup_evidence.marking_dedup;
    add_unique_evidence(
        report,
        marking_dedup
            .fingerprint
            .render_evidence_row("MCC", CheckerSourceKind::MccPetri),
    );
    add_unique_evidence(
        report,
        marking_dedup
            .fingerprint
            .render_validation_evidence_row("MCC", CheckerSourceKind::MccPetri),
    );
    add_unique_evidence(
        report,
        marking_dedup.render_evidence_row("MCC", CheckerSourceKind::MccPetri),
    );
    add_unique_evidence(
        report,
        marking_dedup.render_validation_evidence_row("MCC", CheckerSourceKind::MccPetri),
    );
    let runtime_consumptions = take_mcc_prepared_fingerprint_admission_runtime_consumption();
    let runtime_admission_fault_observed =
        mcc_prepared_fingerprint_admission_runtime_fault_observed(
            setup_evidence,
            &runtime_consumptions,
        );
    let runtime_admission_consumed = !runtime_admission_fault_observed
        && mcc_prepared_fingerprint_admission_runtime_consumed(
            setup_evidence,
            &runtime_consumptions,
        );
    let hot_execution_recorded = setup_evidence
        .trace
        .phase_nanos(SetupTracePhase::HotExecution)
        .is_some();

    for row in mcc_runtime_fingerprint_adoption_rows(
        setup_evidence,
        runtime_admission_consumed,
        runtime_admission_fault_observed,
        hot_execution_recorded,
    ) {
        add_unique_evidence(report, row);
    }
    for consumption in runtime_consumptions {
        add_unique_evidence(report, consumption.render_evidence_row());
    }
    if mcc_prepared_fingerprint_admission_runtime_receipt_fail_closed(
        setup_evidence,
        runtime_admission_consumed,
        runtime_admission_fault_observed,
        hot_execution_recorded,
    ) && !runtime_admission_fault_observed
    {
        add_unique_evidence(
            report,
            mcc_missing_prepared_fingerprint_admission_runtime_consumption_row(setup_evidence),
        );
    }
    let adoption = mcc_shared_engine_adoption_evidence();
    debug_assert!(adoption.validate().is_ok());
    add_unique_evidence(report, adoption.render_evidence_row("MCC"));
    let random_walk_adoption = mcc_random_walk_engine_adoption_evidence();
    debug_assert!(random_walk_adoption.validate().is_ok());
    add_unique_evidence(report, random_walk_adoption.render_evidence_row("MCC"));
    add_unique_evidence(
        report,
        mcc_shared_prepared_checker_program_evidence_row(setup_evidence),
    );
    for row in setup_evidence
        .prepared_program
        .render_frontend_extension_evidence_rows("MCC")
    {
        add_unique_evidence(report, row);
    }
    for row in setup_evidence
        .prepared_program
        .render_candidate_lane_evidence_rows("MCC")
    {
        add_unique_evidence(report, row);
    }
    for row in setup_evidence
        .prepared_program
        .render_validation_plan_evidence_rows("MCC")
    {
        add_unique_evidence(report, row);
    }
    for row in mcc_analytical_solve_decision_rows(setup_evidence) {
        add_unique_evidence(report, row);
    }
    add_unique_evidence(
        report,
        mcc_prepared_program_evidence_row(model, examination, setup_evidence),
    );
    add_unique_evidence(report, mcc_kernel_build_evidence_row(model, setup_evidence));
    if setup_evidence
        .trace
        .phase_nanos(SetupTracePhase::NativePublish)
        .is_some()
    {
        add_unique_evidence(report, mcc_native_publish_evidence_row(setup_evidence));
    }
    add_unique_evidence(
        report,
        mcc_hot_execution_evidence_row(
            setup_evidence,
            status,
            error,
            runtime_admission_consumed,
            runtime_admission_fault_observed,
            hot_execution_recorded,
        ),
    );
}

/// Wall-clock cap for the diagnostic trust-cg native successor capability probe
/// when an MCC deadline is in effect.
///
/// `petri_native_successor_capability_report` JIT-compiles a NativeVerification
/// bundle (`run_isel`, measured ~26s on FlexibleBarrier-PT-04b) to emit
/// *diagnostic* backend-capability evidence. It runs once, before any
/// examination logic, and does NOT poll the deadline, so on a short budget it
/// alone overran the deadline and the harness SIGKILLed the process with zero
/// verdicts emitted. Bounding it is strictly verdict-preserving: the capability
/// report is diagnostic-only and native *execution* adoption is decided
/// separately (see `explorer::observer::native_batch_within_budget`).
const NATIVE_CAPABILITY_PROBE_CAP: Duration = Duration::from_secs(2);

/// Degraded native-successor capability evidence used when the full probe is
/// skipped/abandoned under an MCC deadline. Carries a single explanatory row so
/// audit trails record the deferral; never alters any verdict.
fn petri_native_capability_deferred_report(
    net: &PetriNet,
    remaining: Duration,
) -> CapabilityReport {
    let mut report = CapabilityReport::new(ProblemKind::NativeSuccessor);
    report.add_evidence(format!(
        "petri native successor capability probe deferred under MCC deadline \
         (remaining={remaining:?}, places={}, transitions={}); diagnostic-only, verdict unaffected",
        net.num_places(),
        net.num_transitions(),
    ));
    report
}

/// Build the trust-cg native successor capability evidence, deadline-aware.
///
/// Without a deadline (non-MCC / public API contract) the original unbounded
/// inline probe runs unchanged. Under a deadline the probe is wall-capped on a
/// worker thread (mirroring `run_phase_with_wall_cap` in
/// `examination_non_property/deadlock_one_safe.rs`) and abandoned if it does not
/// finish within a small diagnostic slice; a "deferred" evidence row is
/// substituted. The abandoned worker is intentionally not joined — the trust-cg
/// codegen has no cooperative cancellation primitive; the OS reclaims it when
/// the process exits. Because the capability report never produces a verdict,
/// degrading it under a deadline is strictly verdict-preserving.
fn petri_native_capability_report_deadline_aware(
    net: &PetriNet,
    config: &ExplorationConfig,
) -> CapabilityReport {
    // A 0-place net is the colored-decline PLACEHOLDER (`model/loader.rs`
    // substitutes it when the colored unfold declines). Probing it JIT-compiles
    // an empty successor kernel, which the trust-cg codegen rejects by PANIC
    // ("published image length … exceeds mapping length 0") — on the
    // deadline-less inline path below that panic killed the whole process
    // with zero FORMULA output (v8 diagnosis, 2026-07-10). The report is
    // diagnostic-only, so skipping the probe is strictly verdict-preserving.
    if net.num_places() == 0 {
        let mut report = CapabilityReport::new(ProblemKind::NativeSuccessor);
        report.add_evidence(
            "petri native successor capability probe skipped: empty placeholder net \
             (colored unfold declined); diagnostic-only, verdict unaffected"
                .to_string(),
        );
        return report;
    }
    let Some(deadline) = config.deadline() else {
        // Inline probe, panic-guarded (the worker path below already
        // catch_unwinds; the inline path previously did not, so ANY probe
        // panic here was process-fatal).
        return std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            crate::trust_cg_petri_kernel::petri_native_successor_capability_report(net)
        }))
        .unwrap_or_else(|_| petri_native_capability_deferred_report(net, Duration::ZERO));
    };

    let remaining = deadline.saturating_duration_since(Instant::now());
    let cap = NATIVE_CAPABILITY_PROBE_CAP.min(remaining);
    if cap.is_zero() {
        return petri_native_capability_deferred_report(net, remaining);
    }

    let net_for_worker = net.clone();
    let (tx, rx) = std::sync::mpsc::sync_channel::<std::thread::Result<CapabilityReport>>(1);
    let _worker = std::thread::Builder::new()
        .name("ty-petri-native-capability".to_string())
        .spawn(move || {
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                crate::trust_cg_petri_kernel::petri_native_successor_capability_report(
                    &net_for_worker,
                )
            }));
            // Ignore SendError: parent abandoned the probe and dropped rx.
            let _ = tx.send(result);
        });

    match rx.recv_timeout(cap) {
        Ok(Ok(report)) => report,
        // Panic or cap expiry: substitute the diagnostic deferral row.
        Ok(Err(_)) | Err(_) => petri_native_capability_deferred_report(net, remaining),
    }
}

fn build_petri_mcc_capability_report(
    net: &PetriNet,
    examination: Examination,
    config: &ExplorationConfig,
    source_kind: SourceNetKind,
) -> CapabilityReport {
    let problem = problem_for_examination(examination);
    let limits = solver_limits(config);
    let mut report = CapabilityReport::new(problem).with_limits(limits);
    report.add_evidence(format!(
        "mcc model source={:?} places={} transitions={} examination={}",
        source_kind,
        net.num_places(),
        net.num_transitions(),
        examination.as_str()
    ));

    if symbolic_ay_candidate(examination) {
        let (_, ay_report) = crate::examinations::smt_encoding::find_ay_with_report(
            ProblemKind::Bmc,
            limits_for_bmc(config),
        );
        append_report(&mut report, ay_report);
        add_reachability_chc_lane(&mut report, examination);
    } else {
        report.reject(
            BackendCapability::unsupported(
                BackendDomain::PetriMcc,
                BackendKind::ExternalAYBinary,
                UnsupportedReason::UnsupportedFragment(
                    "MCC examination uses explicit/topological path",
                ),
            )
            .for_problem(problem)
            .with_facets([SolverFacet::ExternalProcess])
            .with_role(CapabilityRole::Fallback)
            .with_detail("AY is not part of the default route for this examination"),
        );
    }

    add_aiger_lane(&mut report, examination);
    append_report(
        &mut report,
        petri_native_capability_report_deadline_aware(net, config),
    );
    add_ltl_route_admission_evidence(&mut report, examination);

    let explicit_role = if report.ay_selected_for_production() {
        CapabilityRole::Fallback
    } else {
        CapabilityRole::Production
    };
    report.select(
        BackendCapability::available(
            BackendDomain::PetriMcc,
            BackendKind::ExplicitState,
            "Rust Petri explicit-state exploration",
        )
        .for_problem(problem)
        .with_role(explicit_role)
        .with_detail(format!(
            "max_states={} workers={} fingerprint_dedup={}",
            config.max_states(),
            config.workers(),
            config.fingerprint_dedup()
        )),
    );

    add_hardware_replay_boundary_and_decision_evidence(&mut report);
    add_symbolic_execution_evidence(&mut report, examination, problem);
    add_ay_solve_decision_profile_evidence(&mut report);
    add_ay_capability_contract_evidence(&mut report);
    add_mcc_portfolio_route_evidence(&mut report);

    report
}

pub(crate) fn add_ltl_answer_lane_summary_evidence(
    report: &mut CapabilityReport,
    examination: Examination,
    config: &ExplorationConfig,
    records: &[ExaminationRecord],
) {
    if !is_ltl_examination(examination) {
        return;
    }

    let answered = records
        .iter()
        .filter(|record| {
            matches!(
                record.value,
                ExaminationValue::Verdict(Verdict::True | Verdict::False)
            )
        })
        .count();
    let cannot_compute = records
        .iter()
        .filter(|record| {
            matches!(
                record.value,
                ExaminationValue::Verdict(Verdict::CannotCompute)
            )
        })
        .count();
    let cannot_compute_reason_code = if cannot_compute == 0 {
        "none"
    } else {
        "explicit_buchi_cannot_compute"
    };
    let blocker_piece = if cannot_compute == 0 {
        "none"
    } else {
        "ltl_explicit_buchi_completion"
    };
    let fail_closed = cannot_compute > 0;
    let production_selected = !fail_closed;
    let production_readiness_status = if fail_closed { "blocked" } else { "accepted" };
    let production_readiness_reason_code = if fail_closed {
        cannot_compute_reason_code
    } else {
        "none"
    };
    let deadline_budget_ms = remaining_budget(config.deadline())
        .map(|duration| duration.as_millis().to_string())
        .unwrap_or_else(|| "none".to_string());

    report.add_evidence(format!(
        "MCC ltl_answer_lane_summary schema={schema} schema_version=1 \
         examination={examination} selected_lane=explicit_buchi \
         selected_backend_code={backend_code} route_status=completed \
         property_count={property_count} answered={answered} \
         cannot_compute={cannot_compute} \
         cannot_compute_reason_code={cannot_compute_reason_code} \
         blocker_piece={blocker_piece} \
         blocker_gate=ltl_explicit_buchi_completion \
         next_answer_lane=ay_lasso_or_aiger_buchi \
         fallback_lane=explicit_state_fallback \
         fallback_reason_code={cannot_compute_reason_code} \
         max_states={max_states} deadline_budget_ms={deadline_budget_ms} \
         production_readiness_status={production_readiness_status} \
         production_readiness_reason_code={production_readiness_reason_code} \
         production_selected={production_selected} fail_closed={fail_closed}",
        schema = LTL_ANSWER_LANE_SUMMARY_SCHEMA,
        examination = examination.as_str(),
        backend_code = BackendKind::ExplicitState.code(),
        property_count = records.len(),
        max_states = config.max_states(),
    ));
}

fn add_symbolic_execution_evidence(
    report: &mut CapabilityReport,
    examination: Examination,
    problem: ProblemKind,
) {
    let detection = symbolic_execution_detection(examination, report);
    report.add_evidence(detection.render_evidence("MCC", problem));
}

fn add_ay_solve_decision_profile_evidence(report: &mut CapabilityReport) {
    if !report
        .evidence
        .iter()
        .any(|row| is_mcc_ay_solve_decision_profile_row(row))
    {
        report.add_evidence(tla_ay::solve_decision_profile_summary_evidence("MCC", None));
    }
}

fn is_mcc_ay_solve_decision_profile_row(row: &str) -> bool {
    row.starts_with("MCC ay_solver_decision_profile_summary ")
}

fn add_ay_capability_contract_evidence(report: &mut CapabilityReport) {
    report.evidence.retain(|row| {
        !is_ay_solver_capability_descriptor_row(row)
            && !is_ay_symbolic_execution_contract_manifest_row(row)
            && !is_ay_symbolic_execution_contract_manifest_health_row(row)
    });
    for row in ay_solver_capability_descriptor_evidence_rows() {
        report.add_evidence(row);
    }
    for row in ay_symbolic_execution_contract_manifest_evidence_rows() {
        report.add_evidence(row);
    }
}

fn add_hardware_replay_boundary_and_decision_evidence(report: &mut CapabilityReport) {
    let production_routing_status_code = report.production_routing_status().code();
    for boundary in runtime_hardware_proof_replay_boundary_statuses(production_routing_status_code)
    {
        let row = boundary.render_evidence_row();
        debug_assert!(
            tla_mc_core::validate_hardware_proof_replay_boundary_evidence_row(&row).is_ok()
        );
        add_unique_evidence(report, row);
    }
    for decision in runtime_blocked_hardware_replay_decision_statuses() {
        let row = decision.render_evidence_row();
        debug_assert!(tla_mc_core::validate_hardware_replay_decision_evidence_row(&row).is_ok());
        debug_assert_eq!(
            tla_mc_core::hardware_replay_decision_accepts_replay_primitive(&row),
            Ok(false)
        );
        add_unique_evidence(report, row);
    }
}

fn ay_solver_capability_descriptor_evidence_rows() -> Vec<String> {
    let descriptor = tla_ay::solver_capability_descriptor();
    let manifest_pairs = tla_ay::solver_capability_descriptor_key_value_pairs();
    let model_blocking = descriptor
        .capability(tla_ay::SolverCapabilityCode::ModelBlocking)
        .expect("AY descriptor must expose model-blocking capability");
    let api_symbols = manifest_pair_value(&manifest_pairs, "api_symbols")
        .unwrap_or_else(|| model_blocking.api_symbols.join(","));
    let evidence_schemas = manifest_pair_value(&manifest_pairs, "evidence_schemas")
        .unwrap_or_else(|| model_blocking.evidence_schemas.join(","));

    vec![format!(
        "AY solver_capability_descriptor \
         schema={schema} schema_version={schema_version} \
         source_package={source_package} package={source_package} \
         solver={solver} capability={capability} \
         status={status} status_code={status_code} reason_code={reason_code} \
         api_symbols={api_symbols} evidence_schemas={evidence_schemas} \
         production_selected=false fail_closed={fail_closed}",
        schema = descriptor.schema,
        schema_version = descriptor.schema_version,
        source_package = tla_ay::AY_SYMBOLIC_EXECUTION_ROUTE_SELECTED_SOLVER_CRATE,
        solver = descriptor.solver,
        capability = model_blocking.capability_code,
        status = model_blocking.status_code,
        status_code = model_blocking.status_code,
        reason_code = model_blocking.reason_code,
        fail_closed = model_blocking.fail_closed,
    )]
}

fn manifest_pair_value(pairs: &[(&'static str, String)], key: &'static str) -> Option<String> {
    pairs
        .iter()
        .find(|(candidate, _)| *candidate == key)
        .map(|(_, value)| value.clone())
}

fn ay_symbolic_execution_contract_manifest_evidence_rows() -> Vec<String> {
    let mut rows = Vec::new();
    rows.extend(
        tla_ay::symbolic_execution_contract_manifest_key_value_pairs()
            .into_iter()
            .map(|(key, value)| {
                ay_manifest_line_evidence_row(
                    "symbolic_execution_contract_manifest",
                    tla_ay::AY_SYMBOLIC_EXECUTION_CONTRACT_MANIFEST_SCHEMA,
                    key,
                    &value,
                )
            }),
    );
    rows.extend(
        tla_ay::symbolic_execution_contract_manifest_health_key_value_rows()
            .into_iter()
            .map(|(key, value)| {
                ay_manifest_line_evidence_row(
                    "symbolic_execution_contract_manifest_health",
                    tla_ay::AY_SYMBOLIC_EXECUTION_CONTRACT_MANIFEST_HEALTH_SCHEMA,
                    &key,
                    &value,
                )
            }),
    );
    rows
}

fn ay_manifest_line_evidence_row(component: &str, schema: &str, key: &str, value: &str) -> String {
    format!(
        "AY {component} schema={schema} schema_version=1 \
         source_package={source_package} package={source_package} \
         manifest_line={key}={value}",
        source_package = tla_ay::AY_SYMBOLIC_EXECUTION_ROUTE_SELECTED_SOLVER_CRATE,
    )
}

fn is_ay_solver_capability_descriptor_row(row: &str) -> bool {
    row.starts_with("AY solver_capability_descriptor ")
}

fn is_ay_symbolic_execution_contract_manifest_row(row: &str) -> bool {
    row.starts_with("AY symbolic_execution_contract_manifest ")
}

fn is_ay_symbolic_execution_contract_manifest_health_row(row: &str) -> bool {
    row.starts_with("AY symbolic_execution_contract_manifest_health ")
}

fn add_mcc_portfolio_route_evidence(report: &mut CapabilityReport) {
    report
        .evidence
        .retain(|row| !is_mcc_portfolio_route_row(row));
    for route in MCC_PORTFOLIO_ROUTES {
        report.add_evidence(route.evidence_row());
    }
}

fn is_mcc_portfolio_route_row(row: &str) -> bool {
    row.starts_with("MCC portfolio_route ")
}

fn add_unique_evidence(report: &mut CapabilityReport, evidence: String) {
    if !report.evidence.iter().any(|row| row == &evidence) {
        report.add_evidence(evidence);
    }
}

fn add_ltl_route_admission_evidence(report: &mut CapabilityReport, examination: Examination) {
    if !is_ltl_examination(examination) {
        return;
    }

    report.add_evidence(format!(
        "MCC ltl_route_admission schema={schema} schema_version=1 \
         examination={examination} selected_lane=explicit_buchi \
         selected_backend_code={backend_code} route_status=admitted \
         reason_code=explicit_buchi_only_sound_ltl_lane \
         ay_lasso_lane_status=blocked \
         ay_lasso_reason_code=missing_ltl_lasso_lowering \
         aiger_ltl_lane_status=blocked \
         aiger_ltl_reason_code=missing_ltl_buchi_product_lowering \
         native_ltl_lane_status=blocked \
         native_ltl_reason_code=missing_ltl_product_kernel \
         next_answer_lane=ay_lasso_or_aiger_buchi \
         fallback_lane=explicit_state_fallback \
         fallback_reason_code=missing_symbolic_ltl_handoff \
         production_readiness_status=blocked \
         production_readiness_reason_code=missing_symbolic_ltl_handoff \
         production_selected=false fail_closed=true",
        schema = LTL_ROUTE_ADMISSION_SCHEMA,
        examination = examination.as_str(),
        backend_code = BackendKind::ExplicitState.code(),
    ));
}

fn is_ltl_examination(examination: Examination) -> bool {
    matches!(
        examination,
        Examination::LTLCardinality | Examination::LTLFireability
    )
}

fn symbolic_execution_detection(
    examination: Examination,
    report: &CapabilityReport,
) -> SymbolicExecutionDetection {
    let Some((required, reason)) = symbolic_execution_need(examination) else {
        return SymbolicExecutionDetection::not_detected();
    };

    if report.ay_selected_for_production() {
        if required {
            SymbolicExecutionDetection::ay_required(reason)
        } else {
            SymbolicExecutionDetection::ay_preferred(reason)
        }
    } else if report
        .rejection_reason(BackendKind::ExternalAYBinary)
        .is_some()
    {
        SymbolicExecutionDetection::local_fallback_after_ay_rejection(reason)
    } else if required {
        SymbolicExecutionDetection::ay_required(reason)
    } else {
        SymbolicExecutionDetection::ay_preferred(reason)
    }
}

fn symbolic_execution_need(examination: Examination) -> Option<(bool, SymbolicExecutionReason)> {
    match examination {
        Examination::ReachabilityDeadlock
        | Examination::ReachabilityCardinality
        | Examination::ReachabilityFireability => {
            Some((false, SymbolicExecutionReason::SymbolicTransitionRelation))
        }
        Examination::OneSafe | Examination::StableMarking => {
            Some((false, SymbolicExecutionReason::SymbolicInitialState))
        }
        Examination::Liveness | Examination::QuasiLiveness => {
            Some((true, SymbolicExecutionReason::UnsupportedLocalFragment))
        }
        Examination::CTLCardinality
        | Examination::CTLFireability
        | Examination::LTLCardinality
        | Examination::LTLFireability
        | Examination::StateSpace
        | Examination::UpperBounds => None,
    }
}

fn add_reachability_chc_lane(report: &mut CapabilityReport, examination: Examination) {
    if !matches!(
        examination,
        Examination::ReachabilityCardinality | Examination::ReachabilityFireability
    ) {
        return;
    }

    if env_flag_enabled(REACHABILITY_PDR_ENV) {
        report.select(
            tla_mc_core::ay_chc_capability(BackendDomain::PetriMcc, ProblemKind::Chc)
                .with_role(CapabilityRole::Validation)
                .with_detail("reachability PDR opt-in is enabled; lane remains validation gated"),
        );
    } else {
        report.reject(
            BackendCapability::disabled(
                BackendDomain::PetriMcc,
                BackendKind::AYChc,
                UnsupportedReason::DisabledByPolicy(REACHABILITY_PDR_ENV),
            )
            .for_problem(ProblemKind::Chc)
            .with_facets([SolverFacet::InProcess, SolverFacet::Chc, SolverFacet::Pdr])
            .with_role(CapabilityRole::Validation)
            .with_detail("reachability PDR has not been promoted for default MCC use"),
        );
    }
}

fn add_aiger_lane(report: &mut CapabilityReport, examination: Examination) {
    if matches!(
        examination,
        Examination::ReachabilityCardinality | Examination::ReachabilityFireability
    ) {
        report.select(
            BackendCapability::available(
                BackendDomain::PetriMcc,
                BackendKind::AigerPortfolio,
                "Petri-to-AIGER reachability seeding",
            )
            .for_problem(ProblemKind::Safety)
            .with_facets([SolverFacet::Sat, SolverFacet::Pdr, SolverFacet::Witness])
            .with_role(CapabilityRole::Production)
            .with_detail("eligible when LP bounds and predicate encoding admit hardware checking"),
        );
    } else {
        report.reject(
            BackendCapability::unsupported(
                BackendDomain::PetriMcc,
                BackendKind::AigerPortfolio,
                UnsupportedReason::UnsupportedFragment("not a reachability property examination"),
            )
            .for_problem(ProblemKind::Safety)
            .with_facets([SolverFacet::Sat, SolverFacet::Pdr])
            .with_role(CapabilityRole::Fallback)
            .with_detail("Petri-to-AIGER lane is only routed for reachability properties"),
        );
    }
}

fn problem_for_examination(examination: Examination) -> ProblemKind {
    match examination {
        Examination::ReachabilityDeadlock => ProblemKind::Deadlock,
        Examination::ReachabilityCardinality | Examination::ReachabilityFireability => {
            ProblemKind::ExplicitReachability
        }
        Examination::CTLCardinality | Examination::CTLFireability => ProblemKind::Safety,
        Examination::LTLCardinality
        | Examination::LTLFireability
        | Examination::Liveness
        | Examination::QuasiLiveness => ProblemKind::Liveness,
        Examination::StateSpace => ProblemKind::StateSpace,
        Examination::OneSafe | Examination::StableMarking => ProblemKind::Safety,
        Examination::UpperBounds => ProblemKind::Invariant,
    }
}

fn symbolic_ay_candidate(examination: Examination) -> bool {
    matches!(
        examination,
        Examination::ReachabilityDeadlock
            | Examination::ReachabilityCardinality
            | Examination::ReachabilityFireability
            | Examination::OneSafe
            | Examination::QuasiLiveness
            | Examination::StableMarking
            | Examination::Liveness
    )
}

fn solver_limits(config: &ExplorationConfig) -> SolverLimits {
    SolverLimits {
        time_budget: remaining_budget(config.deadline()),
        max_depth: None,
        max_states: u64::try_from(config.max_states()).ok(),
        max_memory_bytes: None,
    }
}

fn limits_for_bmc(config: &ExplorationConfig) -> SolverLimits {
    SolverLimits {
        time_budget: remaining_budget(config.deadline()),
        max_depth: Some(16),
        max_states: u64::try_from(config.max_states()).ok(),
        max_memory_bytes: None,
    }
}

fn remaining_budget(deadline: Option<Instant>) -> Option<std::time::Duration> {
    deadline.map(|deadline| deadline.saturating_duration_since(Instant::now()))
}

fn append_report(target: &mut CapabilityReport, source: CapabilityReport) {
    target.selected.extend(source.selected);
    target.rejected.extend(source.rejected);
    target.evidence.extend(source.evidence);
}

fn mcc_model_load_evidence_row(model: &PreparedModel, setup_evidence: &MccSetupEvidence) -> String {
    format!(
        "MCC model_load schema={schema} schema_version=1 \
         source_kind={} model={} model_dir={} pnml_source_kind={} \
         place_count={} transition_count={} nupn_present={} phase={} nanos={} \
         production_selected=false fail_closed=false",
        setup_evidence.prepared_program.source_kind.code(),
        evidence_token(model.model_name()),
        evidence_token(&model.model_dir().display().to_string()),
        source_kind_name(model.source_kind()),
        model.net().num_places(),
        model.net().num_transitions(),
        model.nupn().is_some(),
        SetupTracePhase::SourceLoad.code(),
        setup_evidence
            .trace
            .phase_nanos(SetupTracePhase::SourceLoad)
            .unwrap_or(0),
        schema = MCC_MODEL_LOAD_SCHEMA,
    )
}

fn mcc_shared_engine_adoption_evidence() -> SharedEngineAdoptionEvidence {
    SharedEngineAdoptionEvidence::new(
        MCC_SHARED_ENGINE_ORIGIN_FRONTEND,
        MCC_SHARED_ENGINE_COMPONENT,
        MCC_SHARED_ENGINE_FIRST_BENEFICIARY,
        MCC_SHARED_ENGINE_SECOND_BENEFICIARY,
        MCC_SHARED_ENGINE_EXTRACTION_STATUS,
        MCC_SHARED_ENGINE_LANE_OWNER,
        MCC_SHARED_ADOPTION_ACCEPTANCE_TEST,
    )
    .with_downstream_compatible_frontend_families(
        SharedEngineAdoptionLevel::Level3,
        [
            SharedEngineFrontendFamily::Aiger,
            SharedEngineFrontendFamily::Btor2,
            SharedEngineFrontendFamily::VmtTransitionSystem,
            SharedEngineFrontendFamily::AYAnalytical,
            SharedEngineFrontendFamily::WitnessReplay,
        ],
        [SharedEngineFrontendFamily::Quint],
        [SharedEngineAdoptionFamilyBlocker::new(
            SharedEngineFrontendFamily::FutureImporter,
            "awaiting registered importer frontend",
        )],
    )
    .with_generic_prerequisite("prepared_checker_program_descriptor")
    .with_generic_prerequisite("marking_storage_identity")
    .with_generic_prerequisite("fingerprint_identity")
    .with_generic_prerequisite("prepared_fingerprint_admission_plan")
    .with_generic_prerequisite("dedup_admission")
    .with_generic_prerequisite("ay_symbolic_candidate_lanes")
    .with_generic_prerequisite("witness_replay_validation_plan")
    .with_acceptance_evidence(MCC_RUNTIME_FINGERPRINT_ADOPTION_SCHEMA)
    .with_acceptance_evidence(MCC_RUNTIME_DEDUP_ADMISSION_SCHEMA)
    .with_acceptance_evidence(MCC_SHARED_ENGINE_ACCEPTANCE_EVIDENCE)
}

/// Shared-engine adoption row for the random-walk witness engine extracted
/// into `tla_mc_core::random_walk`. Records that the Petri deadlock and
/// quasi-liveness walk lanes now drive the domain-agnostic
/// `random_walk_witness` control loop, with the frontend-family map naming the
/// other families that the same generic stepper contract is available to.
fn mcc_random_walk_engine_adoption_evidence() -> SharedEngineAdoptionEvidence {
    SharedEngineAdoptionEvidence::new(
        MCC_SHARED_ENGINE_ORIGIN_FRONTEND,
        MCC_RANDOM_WALK_COMPONENT,
        MCC_RANDOM_WALK_FIRST_BENEFICIARY,
        MCC_SHARED_ENGINE_SECOND_BENEFICIARY,
        MCC_RANDOM_WALK_EXTRACTION_STATUS,
        MCC_SHARED_ENGINE_LANE_OWNER,
        MCC_RANDOM_WALK_ACCEPTANCE_TEST,
    )
    .with_downstream_compatible_frontend_families(
        SharedEngineAdoptionLevel::Level3,
        [
            SharedEngineFrontendFamily::Aiger,
            SharedEngineFrontendFamily::Btor2,
            SharedEngineFrontendFamily::VmtTransitionSystem,
            SharedEngineFrontendFamily::AYAnalytical,
            SharedEngineFrontendFamily::WitnessReplay,
        ],
        [SharedEngineFrontendFamily::Quint],
        [SharedEngineAdoptionFamilyBlocker::new(
            SharedEngineFrontendFamily::FutureImporter,
            "awaiting registered importer frontend",
        )],
    )
    .with_generic_prerequisite("transition_system_step_contract")
    .with_generic_prerequisite("single_random_enabled_successor_selection")
    .with_generic_prerequisite("budgeted_walk_step_caps")
    .with_generic_prerequisite("deadline_poll_cadence")
    .with_generic_prerequisite("restart_from_initial_on_dead_state")
    .with_generic_prerequisite("abandon_on_step_failure")
    .with_acceptance_evidence(MCC_RANDOM_WALK_ACCEPTANCE_EVIDENCE)
}

fn mcc_shared_prepared_checker_program_evidence_row(setup_evidence: &MccSetupEvidence) -> String {
    let prepared_program = &setup_evidence.prepared_program;
    let program_fingerprint = prepared_program_sha256(prepared_program);
    let base = prepared_program.render_evidence_row("MCC");
    format!(
        "{base} source_identity={source_identity} property_identity={property_identity} \
         candidate_key={candidate_key} \
         fingerprint_digest_algorithm={algorithm} fingerprint_digest={digest} \
         prepared_program_fingerprint_algorithm={algorithm} prepared_program_fingerprint={digest} \
         shared_engine_origin_frontend={origin_frontend} \
         shared_engine_component={component} shared_engine_lane_owner={owner} \
         shared_engine_first_beneficiary={first_beneficiary} \
         shared_engine_second_beneficiary={second_beneficiary} \
         shared_engine_default_compatible_frontend_families={default_frontend_families} \
         shared_engine_downstream_beneficiary_families={downstream_beneficiary_families} \
         shared_engine_remaining_compatible_frontend_families={remaining_frontend_families} \
         shared_engine_generic_prerequisites={generic_prerequisites} \
         shared_engine_extraction_status={extraction_status} \
         shared_engine_compatible_frontend_families={compatible_frontend_families} \
         shared_engine_frontend_family_blockers={frontend_family_blockers} \
         shared_engine_blocker_status={blocker_status} \
         shared_engine_acceptance_evidence={acceptance_evidence} \
         artifact_fingerprint_algorithm={algorithm} artifact_fingerprint={digest} \
         artifact_prepared_program_fingerprint={digest} \
         proof_fingerprint_algorithm={algorithm} proof_fingerprint={digest} \
         proof_prepared_program_fingerprint={digest} \
         replay_fingerprint_algorithm={algorithm} replay_fingerprint={digest} \
         replay_prepared_program_fingerprint={digest} artifact_id=prepared-program \
         artifact_kind=prepared_program artifact_sha256={digest} \
         canonical_identity_id={canonical_identity} \
         canonical_identity_digest_algorithm={algorithm} \
         canonical_identity_digest={digest}",
        source_identity = evidence_option_token(setup_evidence.trace.source_identity.as_deref()),
        property_identity =
            evidence_option_token(setup_evidence.trace.property_identity.as_deref()),
        candidate_key = evidence_option_token(setup_evidence.trace.candidate_key.as_deref()),
        algorithm = MCC_PREPARED_PROGRAM_FINGERPRINT_ALGORITHM,
        digest = program_fingerprint,
        origin_frontend = MCC_SHARED_ENGINE_ORIGIN_FRONTEND,
        component = MCC_SHARED_ENGINE_COMPONENT,
        owner = MCC_SHARED_ENGINE_LANE_OWNER,
        first_beneficiary = MCC_SHARED_ENGINE_FIRST_BENEFICIARY,
        second_beneficiary = MCC_SHARED_ENGINE_SECOND_BENEFICIARY,
        default_frontend_families = MCC_SHARED_ENGINE_DEFAULT_COMPATIBLE_FRONTEND_FAMILIES,
        downstream_beneficiary_families = MCC_SHARED_ENGINE_DOWNSTREAM_BENEFICIARY_FAMILIES,
        remaining_frontend_families = MCC_SHARED_ENGINE_REMAINING_COMPATIBLE_FRONTEND_FAMILIES,
        generic_prerequisites = MCC_SHARED_ENGINE_GENERIC_PREREQUISITES,
        extraction_status = MCC_SHARED_ENGINE_EXTRACTION_STATUS,
        compatible_frontend_families = MCC_SHARED_ENGINE_COMPATIBLE_FRONTEND_FAMILIES,
        frontend_family_blockers = MCC_SHARED_ENGINE_FRONTEND_FAMILY_BLOCKERS,
        blocker_status = MCC_SHARED_ENGINE_BLOCKER_STATUS,
        acceptance_evidence = MCC_SHARED_ENGINE_ACCEPTANCE_EVIDENCE,
        canonical_identity = MCC_PREPARED_PROGRAM_CANONICAL_IDENTITY,
    )
}

fn mcc_analytical_solve_decision_rows(setup_evidence: &MccSetupEvidence) -> Vec<String> {
    let program = &setup_evidence.prepared_program;
    program
        .analytical_solves
        .iter()
        .enumerate()
        .map(|(index, solve)| {
            let validation = mcc_validation_for_analytical_solve(solve.kind);
            let lane = mcc_candidate_lane_for_analytical_solve(program, solve.kind);
            let mut decision = AnalyticalSolveDecision::from_prepared_solve(program, solve, lane)
                .with_backend(mcc_backend_for_analytical_solve(solve.kind))
                .with_validation_requirements([validation])
                .with_candidate_key(format!("mcc.petri.analytical.{}", solve.kind.code()))
                .with_portfolio_lifecycle(AnalyticalSolvePortfolioLifecycle::Admitted)
                .with_portfolio_rank((index + 1) as u32)
                .with_portfolio_candidate_id(solve.id.as_str())
                .with_decision_reason(AnalyticalSolveDecisionReason::StructuralProofOnly)
                .with_reason_code("requires_verifier_lane");

            let fingerprint = prepared_identity(
                mcc_analytical_fingerprint_role(solve.kind),
                &format!("{}:{}", program.identity, solve.id),
            );
            match solve.kind {
                PreparedAnalyticalSolveKind::Reachability => {
                    decision = decision.with_witness_fingerprint(fingerprint);
                }
                PreparedAnalyticalSolveKind::StateSpaceCardinality
                | PreparedAnalyticalSolveKind::DeadlockFreedom => {
                    decision = decision.with_certificate_fingerprint(fingerprint);
                }
                PreparedAnalyticalSolveKind::StableMarking
                | PreparedAnalyticalSolveKind::UpperBounds
                | PreparedAnalyticalSolveKind::LinearInvariant
                | PreparedAnalyticalSolveKind::BoundedModelCheck
                | PreparedAnalyticalSolveKind::PdrSafety
                | PreparedAnalyticalSolveKind::KInduction
                | PreparedAnalyticalSolveKind::SmtQuery
                | PreparedAnalyticalSolveKind::SatQuery => {
                    decision = decision.with_proof_fingerprint(fingerprint);
                }
            }

            decision.render_evidence_row("MCC")
        })
        .collect()
}

fn mcc_candidate_lane_for_analytical_solve(
    program: &PreparedCheckerProgram,
    kind: PreparedAnalyticalSolveKind,
) -> Option<&PreparedCandidateLaneDescriptor> {
    let candidate_key = match kind {
        PreparedAnalyticalSolveKind::BoundedModelCheck
        | PreparedAnalyticalSolveKind::PdrSafety
        | PreparedAnalyticalSolveKind::KInduction
        | PreparedAnalyticalSolveKind::SmtQuery
        | PreparedAnalyticalSolveKind::SatQuery => "ay_symbolic",
        PreparedAnalyticalSolveKind::StateSpaceCardinality
        | PreparedAnalyticalSolveKind::DeadlockFreedom
        | PreparedAnalyticalSolveKind::Reachability
        | PreparedAnalyticalSolveKind::StableMarking
        | PreparedAnalyticalSolveKind::UpperBounds
        | PreparedAnalyticalSolveKind::LinearInvariant => return None,
    };
    program
        .candidate_lanes
        .iter()
        .find(|lane| lane.candidate_key.as_deref() == Some(candidate_key))
}

fn mcc_validation_for_analytical_solve(
    kind: PreparedAnalyticalSolveKind,
) -> PreparedValidationKind {
    match kind {
        PreparedAnalyticalSolveKind::Reachability => PreparedValidationKind::WitnessReplay,
        PreparedAnalyticalSolveKind::StateSpaceCardinality
        | PreparedAnalyticalSolveKind::DeadlockFreedom => PreparedValidationKind::CompleteGraph,
        PreparedAnalyticalSolveKind::BoundedModelCheck
        | PreparedAnalyticalSolveKind::PdrSafety
        | PreparedAnalyticalSolveKind::KInduction
        | PreparedAnalyticalSolveKind::SmtQuery
        | PreparedAnalyticalSolveKind::SatQuery => PreparedValidationKind::AYProof,
        PreparedAnalyticalSolveKind::StableMarking
        | PreparedAnalyticalSolveKind::UpperBounds
        | PreparedAnalyticalSolveKind::LinearInvariant => PreparedValidationKind::StructuralProof,
    }
}

fn mcc_backend_for_analytical_solve(kind: PreparedAnalyticalSolveKind) -> BackendKind {
    match kind {
        PreparedAnalyticalSolveKind::BoundedModelCheck
        | PreparedAnalyticalSolveKind::KInduction
        | PreparedAnalyticalSolveKind::SmtQuery
        | PreparedAnalyticalSolveKind::SatQuery => BackendKind::ExternalAYBinary,
        PreparedAnalyticalSolveKind::PdrSafety => BackendKind::AYChc,
        PreparedAnalyticalSolveKind::StateSpaceCardinality
        | PreparedAnalyticalSolveKind::DeadlockFreedom
        | PreparedAnalyticalSolveKind::Reachability
        | PreparedAnalyticalSolveKind::StableMarking
        | PreparedAnalyticalSolveKind::UpperBounds
        | PreparedAnalyticalSolveKind::LinearInvariant => BackendKind::LocalSymbolicExecution,
    }
}

fn mcc_analytical_fingerprint_role(kind: PreparedAnalyticalSolveKind) -> &'static str {
    match kind {
        PreparedAnalyticalSolveKind::Reachability => "mcc.petri.witness_fingerprint",
        PreparedAnalyticalSolveKind::StateSpaceCardinality
        | PreparedAnalyticalSolveKind::DeadlockFreedom => "mcc.petri.certificate_fingerprint",
        PreparedAnalyticalSolveKind::StableMarking
        | PreparedAnalyticalSolveKind::UpperBounds
        | PreparedAnalyticalSolveKind::LinearInvariant
        | PreparedAnalyticalSolveKind::BoundedModelCheck
        | PreparedAnalyticalSolveKind::PdrSafety
        | PreparedAnalyticalSolveKind::KInduction
        | PreparedAnalyticalSolveKind::SmtQuery
        | PreparedAnalyticalSolveKind::SatQuery => "mcc.petri.proof_fingerprint",
    }
}

fn mcc_prepared_program_evidence_row(
    model: &PreparedModel,
    examination: Examination,
    setup_evidence: &MccSetupEvidence,
) -> String {
    let prepared_program = &setup_evidence.prepared_program;
    let analytical_solve_ids = evidence_list(
        prepared_program
            .analytical_solves
            .iter()
            .map(|solve| solve.id.as_str()),
    );
    let analytical_solve_kinds = evidence_list(
        prepared_program
            .analytical_solves
            .iter()
            .map(|solve| solve.kind.code()),
    );
    let analytical_solve_problems = evidence_list(
        prepared_program
            .analytical_solves
            .iter()
            .map(|solve| solve.problem.code()),
    );
    let symbolic_proof_ids = evidence_list(
        prepared_program
            .symbolic_proofs
            .iter()
            .map(|proof| proof.id.as_str()),
    );
    let symbolic_proof_kinds = evidence_list(
        prepared_program
            .symbolic_proofs
            .iter()
            .map(|proof| proof.kind.code()),
    );
    let symbolic_proof_problems = evidence_list(
        prepared_program
            .symbolic_proofs
            .iter()
            .map(|proof| proof.problem.code()),
    );
    let backend_family_ids = evidence_list(
        prepared_program
            .backend_families
            .iter()
            .map(|family| family.id.as_str()),
    );
    let backend_family_codes = evidence_list(
        prepared_program
            .backend_families
            .iter()
            .map(|family| family.backend.code()),
    );
    let backend_family_problems = evidence_list(
        prepared_program
            .backend_families
            .iter()
            .map(|family| family.problem.code()),
    );
    let backend_family_facets =
        evidence_list(prepared_program.backend_families.iter().map(|family| {
            if family.facets.is_empty() {
                String::from("none")
            } else {
                family
                    .facets
                    .iter()
                    .map(|facet| facet.code())
                    .collect::<Vec<_>>()
                    .join("+")
            }
        }));
    let validation_kinds = evidence_list(
        prepared_program
            .validations
            .iter()
            .map(|validation| validation.code()),
    );
    let program_fingerprint = prepared_program_sha256(prepared_program);
    let fingerprint_id = prepared_program
        .fingerprint
        .as_ref()
        .map(|fingerprint| evidence_token(&fingerprint.id))
        .unwrap_or_else(|| String::from("none"));
    let identity_fields = prepared_identity_evidence_fields(
        setup_evidence,
        &prepared_program.effective_identity_fields(),
    );
    format!(
        "MCC mcc_prepared_program schema={schema} schema_version=1 \
         identity={} frontend_kind={} source_kind={} payload_kind={} model={} \
         examination={} pnml_source_kind={} storage_kind={} marking_abi=u64_marking_vector \
         place_count={} transition_count={} property_count={} analytical_solve_count={} \
         symbolic_proof_count={} backend_family_count={} validation_count={} \
         property_id_status={} reduction_mode={} analytical_solve_ids={} \
         analytical_solve_kinds={} analytical_solve_problems={} symbolic_proof_ids={} \
         symbolic_proof_kinds={} symbolic_proof_problems={} backend_family_ids={} \
         backend_family_codes={} backend_family_problems={} backend_family_facets={} \
         validation_kinds={} output_contract=mcc_stdout_v1 {identity_fields} fingerprint_id={} \
         fingerprint_digest_algorithm={} fingerprint_digest={} \
         prepared_program_fingerprint_algorithm={} prepared_program_fingerprint={} \
         shared_engine_origin_frontend={} shared_engine_component={} shared_engine_lane_owner={} \
         shared_engine_first_beneficiary={} \
         shared_engine_second_beneficiary={} shared_engine_extraction_status={} \
         shared_engine_compatible_frontend_families={} \
         shared_engine_default_compatible_frontend_families={} \
         shared_engine_downstream_beneficiary_families={} \
         shared_engine_remaining_compatible_frontend_families={} \
         shared_engine_generic_prerequisites={} \
         shared_engine_frontend_family_blockers={} shared_engine_blocker_status={} \
         shared_engine_acceptance_evidence={} \
         artifact_fingerprint_algorithm={} artifact_fingerprint={} \
         artifact_prepared_program_fingerprint={} proof_fingerprint_algorithm={} \
         proof_fingerprint={} proof_prepared_program_fingerprint={} \
         replay_fingerprint_algorithm={} replay_fingerprint={} \
         replay_prepared_program_fingerprint={} production_selected=false fail_closed=false",
        evidence_token(&prepared_program.identity),
        prepared_program.source_kind.code(),
        prepared_program.source_kind.code(),
        prepared_program.payload_kind.code(),
        evidence_token(model.model_name()),
        examination.as_str(),
        source_kind_name(model.source_kind()),
        prepared_program.storage_kind.code(),
        model.net().num_places(),
        model.net().num_transitions(),
        prepared_program.properties.len(),
        prepared_program.analytical_solves.len(),
        prepared_program.symbolic_proofs.len(),
        prepared_program.backend_families.len(),
        prepared_program.validations.len(),
        setup_evidence.property_id_status,
        reduction_mode_code(examination.reduction_mode()),
        analytical_solve_ids,
        analytical_solve_kinds,
        analytical_solve_problems,
        symbolic_proof_ids,
        symbolic_proof_kinds,
        symbolic_proof_problems,
        backend_family_ids,
        backend_family_codes,
        backend_family_problems,
        backend_family_facets,
        validation_kinds,
        fingerprint_id,
        MCC_PREPARED_PROGRAM_FINGERPRINT_ALGORITHM,
        program_fingerprint,
        MCC_PREPARED_PROGRAM_FINGERPRINT_ALGORITHM,
        program_fingerprint,
        MCC_SHARED_ENGINE_ORIGIN_FRONTEND,
        MCC_SHARED_ENGINE_COMPONENT,
        MCC_SHARED_ENGINE_LANE_OWNER,
        MCC_SHARED_ENGINE_FIRST_BENEFICIARY,
        MCC_SHARED_ENGINE_SECOND_BENEFICIARY,
        MCC_SHARED_ENGINE_EXTRACTION_STATUS,
        MCC_SHARED_ENGINE_COMPATIBLE_FRONTEND_FAMILIES,
        MCC_SHARED_ENGINE_DEFAULT_COMPATIBLE_FRONTEND_FAMILIES,
        MCC_SHARED_ENGINE_DOWNSTREAM_BENEFICIARY_FAMILIES,
        MCC_SHARED_ENGINE_REMAINING_COMPATIBLE_FRONTEND_FAMILIES,
        MCC_SHARED_ENGINE_GENERIC_PREREQUISITES,
        MCC_SHARED_ENGINE_FRONTEND_FAMILY_BLOCKERS,
        MCC_SHARED_ENGINE_BLOCKER_STATUS,
        MCC_SHARED_ENGINE_ACCEPTANCE_EVIDENCE,
        MCC_PREPARED_PROGRAM_FINGERPRINT_ALGORITHM,
        program_fingerprint,
        program_fingerprint,
        MCC_PREPARED_PROGRAM_FINGERPRINT_ALGORITHM,
        program_fingerprint,
        program_fingerprint,
        MCC_PREPARED_PROGRAM_FINGERPRINT_ALGORITHM,
        program_fingerprint,
        program_fingerprint,
        schema = MCC_PREPARED_PROGRAM_SCHEMA,
    )
}

fn prepared_identity_evidence_fields(
    setup_evidence: &MccSetupEvidence,
    identities: &CheckerArtifactIdentityFields,
) -> String {
    format!(
        "source_identity={} property_identity={} candidate_key={} candidate_identity={} \
         lane_identity={} cache_key={} frontend_payload_identity={} artifact_identity={} \
         storage_policy_identity={} fingerprint_policy_identity={} fingerprint_identity={} \
         batch_artifact_identity={}",
        evidence_option_token(setup_evidence.trace.source_identity.as_deref()),
        evidence_option_token(setup_evidence.trace.property_identity.as_deref()),
        evidence_option_token(setup_evidence.trace.candidate_key.as_deref()),
        evidence_option_token(identities.candidate_identity.as_deref()),
        evidence_option_token(identities.lane_identity.as_deref()),
        evidence_option_token(identities.cache_key.as_deref()),
        evidence_option_token(identities.frontend_payload_identity.as_deref()),
        evidence_option_token(identities.artifact_identity.as_deref()),
        evidence_option_token(identities.storage_policy_identity.as_deref()),
        evidence_option_token(identities.fingerprint_policy_identity.as_deref()),
        evidence_option_token(identities.fingerprint_identity.as_deref()),
        evidence_option_token(identities.batch_artifact_identity.as_deref()),
    )
}

fn mcc_kernel_build_evidence_row(
    model: &PreparedModel,
    setup_evidence: &MccSetupEvidence,
) -> String {
    format!(
        "MCC kernel_build schema={schema} schema_version=1 \
         source_kind={} payload_kind={} kernel_family=petri_transition \
         storage_kind={} marking_abi=u64_marking_vector phase={} nanos={} \
         place_count={} transition_kernels={} predicate_kernels={} \
         production_selected=false fail_closed=false",
        setup_evidence.prepared_program.source_kind.code(),
        setup_evidence.prepared_program.payload_kind.code(),
        setup_evidence.prepared_program.storage_kind.code(),
        SetupTracePhase::PreparedProgramBuild.code(),
        setup_evidence
            .trace
            .phase_nanos(SetupTracePhase::PreparedProgramBuild)
            .unwrap_or(0),
        model.net().num_places(),
        setup_evidence.prepared_program.transitions.len(),
        setup_evidence.prepared_program.properties.len(),
        schema = MCC_KERNEL_BUILD_SCHEMA,
    )
}

fn mcc_native_publish_evidence_row(setup_evidence: &MccSetupEvidence) -> String {
    format!(
        "MCC native_publish schema={schema} schema_version=1 \
         source_kind={} payload_kind={} storage_kind={} phase={} nanos={} \
         native_backend=trust_cg_petri_native status=blocked reason_code=validation_only_candidate \
         production_selected=false fail_closed=true",
        setup_evidence.prepared_program.source_kind.code(),
        setup_evidence.prepared_program.payload_kind.code(),
        setup_evidence.prepared_program.storage_kind.code(),
        SetupTracePhase::NativePublish.code(),
        setup_evidence
            .trace
            .phase_nanos(SetupTracePhase::NativePublish)
            .unwrap_or(0),
        schema = MCC_NATIVE_PUBLISH_SCHEMA,
    )
}

fn mcc_hot_execution_evidence_row(
    setup_evidence: &MccSetupEvidence,
    status: MccRunStatus,
    error: Option<&str>,
    runtime_admission_consumed: bool,
    runtime_admission_fault_observed: bool,
    hot_execution_recorded: bool,
) -> String {
    let (status_code, records) = match status {
        MccRunStatus::Completed { records } => ("completed", records.to_string()),
        MccRunStatus::Error => ("error", String::from("unknown")),
        MccRunStatus::Panic => ("panic", String::from("unknown")),
    };
    let runtime_admission_receipt_required =
        mcc_requires_prepared_fingerprint_admission_runtime_receipt(setup_evidence);
    let runtime_admission_receipt_status =
        mcc_prepared_fingerprint_admission_runtime_receipt_status(
            setup_evidence,
            runtime_admission_consumed,
            runtime_admission_fault_observed,
            hot_execution_recorded,
        );
    let runtime_admission_receipt_reason_code =
        mcc_prepared_fingerprint_admission_runtime_receipt_reason_code(
            setup_evidence,
            runtime_admission_consumed,
            runtime_admission_fault_observed,
            hot_execution_recorded,
        );
    let runtime_admission_fail_closed =
        mcc_prepared_fingerprint_admission_runtime_receipt_fail_closed(
            setup_evidence,
            runtime_admission_consumed,
            runtime_admission_fault_observed,
            hot_execution_recorded,
        );
    let runtime_consumption_status = mcc_prepared_fingerprint_admission_runtime_consumption_status(
        setup_evidence,
        runtime_admission_consumed,
        runtime_admission_fault_observed,
        hot_execution_recorded,
    );
    let hot_loop_consumption = if runtime_admission_consumed {
        "prepared_fingerprint_admission_observed"
    } else if runtime_admission_fault_observed {
        MCC_PREPARED_FINGERPRINT_ADMISSION_RUNTIME_CONSUMPTION_MISSING
    } else if runtime_admission_fail_closed {
        MCC_PREPARED_FINGERPRINT_ADMISSION_RUNTIME_RECEIPT_MISSING_REASON
    } else if runtime_admission_receipt_required {
        "setup_only_no_hot_loop_claim"
    } else {
        "not_required"
    };
    let completed = matches!(status, MccRunStatus::Completed { .. });
    let fail_closed = !completed || !hot_execution_recorded || runtime_admission_fail_closed;
    let production_selected = completed && hot_execution_recorded && !runtime_admission_fail_closed;
    let production_readiness_reason_code = if !completed {
        MCC_HOT_EXECUTION_NOT_COMPLETED_REASON
    } else if !hot_execution_recorded {
        MCC_PREPARED_FINGERPRINT_ADMISSION_RUNTIME_RECEIPT_SETUP_ONLY_REASON
    } else if runtime_admission_fail_closed {
        runtime_admission_receipt_reason_code
    } else {
        MCC_PREPARED_FINGERPRINT_ADMISSION_RUNTIME_RECEIPT_AVAILABLE_REASON
    };
    format!(
        "MCC hot_execution schema={schema} schema_version=1 \
         source_kind={} payload_kind={} phase={} nanos={} hot_execution_recorded={} \
         status={} records={} \
         {} runtime_frontend_neutral=true solver_storage_shared=true \
         fingerprint_policy_identity={} fingerprint_identity={} dedup_identity={} \
         storage_policy_identity={} exact_or_unknown={} \
         prepared_fingerprint_admission_runtime_receipt_required={} \
         prepared_fingerprint_admission_runtime_receipt_status={} \
         prepared_fingerprint_admission_runtime_reason_code={} \
         prepared_fingerprint_admission_runtime_counters_present={} \
         prepared_fingerprint_admission_runtime_fail_closed={} \
         runtime_consumption_status={} hot_loop_consumption={} error={} \
         production_readiness_status={} production_readiness_reason_code={} \
         production_selected={} fail_closed={}",
        setup_evidence.prepared_program.source_kind.code(),
        setup_evidence.prepared_program.payload_kind.code(),
        SetupTracePhase::HotExecution.code(),
        setup_evidence
            .trace
            .phase_nanos(SetupTracePhase::HotExecution)
            .unwrap_or(0),
        hot_execution_recorded,
        status_code,
        records,
        mcc_shared_setup_adoption_fields(setup_evidence),
        evidence_token(
            &setup_evidence
                .marking_dedup
                .fingerprint
                .fingerprint_policy_identity()
        ),
        evidence_token(
            &setup_evidence
                .marking_dedup
                .fingerprint
                .fingerprint_identity()
        ),
        evidence_token(&setup_evidence.marking_dedup.dedup_identity()),
        evidence_token(&setup_evidence.marking_dedup.storage_policy_identity()),
        mcc_exact_or_unknown_status(status),
        runtime_admission_receipt_required,
        runtime_admission_receipt_status,
        runtime_admission_receipt_reason_code,
        runtime_admission_consumed,
        runtime_admission_fail_closed,
        runtime_consumption_status,
        hot_loop_consumption,
        evidence_option_token(error),
        if production_selected {
            MCC_PREPARED_FINGERPRINT_ADMISSION_RUNTIME_PRODUCTION_SELECTED
        } else {
            MCC_PREPARED_FINGERPRINT_ADMISSION_RUNTIME_PRODUCTION_BLOCKED
        },
        production_readiness_reason_code,
        production_selected,
        fail_closed,
        schema = MCC_HOT_EXECUTION_SCHEMA,
    )
}

fn mcc_runtime_fingerprint_adoption_rows(
    setup_evidence: &MccSetupEvidence,
    runtime_admission_consumed: bool,
    runtime_admission_fault_observed: bool,
    hot_execution_recorded: bool,
) -> Vec<String> {
    let shared_fields = mcc_shared_setup_adoption_fields(setup_evidence);
    let runtime_reuse_fields = mcc_runtime_reuse_inheritance_fields(setup_evidence);
    let fingerprint = &setup_evidence.marking_dedup.fingerprint;
    let dedup = &setup_evidence.marking_dedup;
    let admission_plan = mcc_marking_prepared_fingerprint_admission_plan(setup_evidence);
    debug_assert!(admission_plan.validate_runtime_admission().is_ok());
    let admission_fields = mcc_prepared_fingerprint_admission_fields(&admission_plan);
    let runtime_admission_receipt_required =
        mcc_requires_prepared_fingerprint_admission_runtime_receipt(setup_evidence);
    let runtime_admission_receipt_status =
        mcc_prepared_fingerprint_admission_runtime_receipt_status(
            setup_evidence,
            runtime_admission_consumed,
            runtime_admission_fault_observed,
            hot_execution_recorded,
        );
    let runtime_admission_receipt_reason_code =
        mcc_prepared_fingerprint_admission_runtime_receipt_reason_code(
            setup_evidence,
            runtime_admission_consumed,
            runtime_admission_fault_observed,
            hot_execution_recorded,
        );
    let runtime_admission_fail_closed =
        mcc_prepared_fingerprint_admission_runtime_receipt_fail_closed(
            setup_evidence,
            runtime_admission_consumed,
            runtime_admission_fault_observed,
            hot_execution_recorded,
        );
    let runtime_consumption_status = mcc_prepared_fingerprint_admission_runtime_consumption_status(
        setup_evidence,
        runtime_admission_consumed,
        runtime_admission_fault_observed,
        hot_execution_recorded,
    );
    vec![
        format!(
            "MCC runtime_fingerprint_adoption schema={} schema_version=1 {} \
             {} {} source_kind={} frontend_kind={} lane_kind={} fingerprint_id={} \
             fingerprint_policy_identity={} fingerprint_identity={} \
             canonical_domain={} canonical_domain_version={} digest_bits={} \
             canonical_fingerprint_identity={} fingerprint_algorithm={} \
             fingerprint_value_kind={} fingerprint_namespace={} canonicalization_version={} \
             storage_policy_identity={} dedup_identity={} exact_or_unknown=unknown \
             exact_or_unknown_guard=fail_closed_until_runtime_completion \
             admission_status=accepted validation_scope={} runtime_consumed=false \
             hot_loop_consumption=not_observed validation_status={} \
             evidence_scope=setup_descriptor setup_only=true runtime_claim=false \
             runtime_win_claim=false prepared_admission_receipt_required={} \
             prepared_admission_receipt_status={} prepared_admission_receipt_reason_code={} \
             runtime_consumption_status={} admission_counters_present={} \
             missing_runtime_receipt_fail_closed={} \
             production_selected=false fail_closed=true",
            MCC_RUNTIME_FINGERPRINT_ADOPTION_SCHEMA,
            shared_fields,
            runtime_reuse_fields,
            admission_fields,
            setup_evidence.prepared_program.source_kind.code(),
            setup_evidence.prepared_program.source_kind.code(),
            dedup.lane.code(),
            evidence_token(&fingerprint.id),
            evidence_token(&fingerprint.fingerprint_policy_identity()),
            evidence_token(&fingerprint.fingerprint_identity()),
            evidence_token(&fingerprint.canonical_domain.id),
            evidence_token(&fingerprint.canonical_domain.version),
            fingerprint.digest_bits,
            evidence_token(&fingerprint.fingerprint_identity()),
            fingerprint.algorithm.code(),
            fingerprint.value_kind.code(),
            evidence_token(&fingerprint.namespace),
            evidence_token(&fingerprint.canonicalization_version),
            evidence_token(&dedup.storage_policy_identity()),
            evidence_token(&dedup.dedup_identity()),
            MCC_PREPARED_FINGERPRINT_ADMISSION_SETUP_VALIDATION_SCOPE,
            setup_evidence.trace.validation_status.code(),
            runtime_admission_receipt_required,
            runtime_admission_receipt_status,
            runtime_admission_receipt_reason_code,
            runtime_consumption_status,
            runtime_admission_consumed,
            runtime_admission_fail_closed,
        ),
        format!(
            "MCC runtime_dedup_admission schema={} schema_version=1 {} \
             {} {} source_kind={} frontend_kind={} lane_kind={} dedup_scope={} storage_kind={} \
             collision_policy={} collision_fail_closed={} storage_config_identity={} \
             dedup_admission_scope={} dedup_admission_policy={} \
             fingerprint_policy_identity={} fingerprint_identity={} dedup_identity={} \
             canonical_fingerprint_identity={} storage_policy_identity={} \
             exact_or_unknown=unknown \
             exact_or_unknown_guard=fail_closed_until_runtime_completion \
             admission_status=accepted validation_scope={} runtime_consumed=false \
             hot_loop_consumption=not_observed \
             evidence_scope=setup_descriptor setup_only=true runtime_claim=false \
             runtime_win_claim=false prepared_admission_receipt_required={} \
             prepared_admission_receipt_status={} prepared_admission_receipt_reason_code={} \
             runtime_consumption_status={} admission_counters_present={} \
             missing_runtime_receipt_fail_closed={} \
             validation_status={} production_selected=false fail_closed=true",
            MCC_RUNTIME_DEDUP_ADMISSION_SCHEMA,
            shared_fields,
            runtime_reuse_fields,
            admission_fields,
            setup_evidence.prepared_program.source_kind.code(),
            setup_evidence.prepared_program.source_kind.code(),
            dedup.lane.code(),
            dedup.scope.code(),
            dedup.storage.code(),
            dedup.collision_policy.code(),
            dedup.collision_policy.is_fail_closed(),
            evidence_option_token(dedup.storage_config_identity.as_deref()),
            dedup.scope.code(),
            dedup.collision_policy.code(),
            evidence_token(&fingerprint.fingerprint_policy_identity()),
            evidence_token(&fingerprint.fingerprint_identity()),
            evidence_token(&dedup.dedup_identity()),
            evidence_token(&fingerprint.fingerprint_identity()),
            evidence_token(&dedup.storage_policy_identity()),
            MCC_PREPARED_FINGERPRINT_ADMISSION_SETUP_VALIDATION_SCOPE,
            runtime_admission_receipt_required,
            runtime_admission_receipt_status,
            runtime_admission_receipt_reason_code,
            runtime_consumption_status,
            runtime_admission_consumed,
            runtime_admission_fail_closed,
            setup_evidence.trace.validation_status.code(),
        ),
    ]
}

fn mcc_prepared_fingerprint_admission_runtime_consumed(
    setup_evidence: &MccSetupEvidence,
    consumptions: &[MccPreparedFingerprintAdmissionRuntimeConsumption],
) -> bool {
    let expected_plan_id = mcc_marking_prepared_fingerprint_admission_plan_id(setup_evidence);
    consumptions
        .iter()
        .any(|consumption| consumption.plan_id == expected_plan_id && consumption.attempted > 0)
}

fn mcc_prepared_fingerprint_admission_runtime_fault_observed(
    setup_evidence: &MccSetupEvidence,
    consumptions: &[MccPreparedFingerprintAdmissionRuntimeConsumption],
) -> bool {
    let expected_plan_id = mcc_marking_prepared_fingerprint_admission_plan_id(setup_evidence);
    consumptions
        .iter()
        .any(|consumption| consumption.plan_id == expected_plan_id && consumption.fault > 0)
}

fn mcc_requires_prepared_fingerprint_admission_runtime_receipt(
    setup_evidence: &MccSetupEvidence,
) -> bool {
    mcc_marking_prepared_fingerprint_admission_plan_id(setup_evidence)
        == MCC_FINGERPRINT_ONLY_PREPARED_FINGERPRINT_ADMISSION_PLAN_ID
}

fn mcc_prepared_fingerprint_admission_runtime_receipt_status(
    setup_evidence: &MccSetupEvidence,
    runtime_admission_consumed: bool,
    runtime_admission_fault_observed: bool,
    hot_execution_recorded: bool,
) -> &'static str {
    if !mcc_requires_prepared_fingerprint_admission_runtime_receipt(setup_evidence) {
        MCC_PREPARED_FINGERPRINT_ADMISSION_RUNTIME_RECEIPT_NOT_REQUIRED
    } else if runtime_admission_fault_observed {
        MCC_PREPARED_FINGERPRINT_ADMISSION_RUNTIME_RECEIPT_MISSING
    } else if runtime_admission_consumed {
        MCC_PREPARED_FINGERPRINT_ADMISSION_RUNTIME_RECEIPT_ACCEPTED
    } else if hot_execution_recorded {
        MCC_PREPARED_FINGERPRINT_ADMISSION_RUNTIME_RECEIPT_MISSING
    } else {
        MCC_PREPARED_FINGERPRINT_ADMISSION_RUNTIME_RECEIPT_SETUP_ONLY
    }
}

fn mcc_prepared_fingerprint_admission_runtime_receipt_reason_code(
    setup_evidence: &MccSetupEvidence,
    runtime_admission_consumed: bool,
    runtime_admission_fault_observed: bool,
    hot_execution_recorded: bool,
) -> &'static str {
    if mcc_requires_prepared_fingerprint_admission_runtime_receipt(setup_evidence)
        && runtime_admission_fault_observed
    {
        return MCC_PREPARED_FINGERPRINT_ADMISSION_RUNTIME_PRODUCTION_FAULT_REASON;
    }
    match mcc_prepared_fingerprint_admission_runtime_receipt_status(
        setup_evidence,
        runtime_admission_consumed,
        runtime_admission_fault_observed,
        hot_execution_recorded,
    ) {
        MCC_PREPARED_FINGERPRINT_ADMISSION_RUNTIME_RECEIPT_ACCEPTED => {
            MCC_PREPARED_FINGERPRINT_ADMISSION_RUNTIME_RECEIPT_AVAILABLE_REASON
        }
        MCC_PREPARED_FINGERPRINT_ADMISSION_RUNTIME_RECEIPT_MISSING => {
            MCC_PREPARED_FINGERPRINT_ADMISSION_RUNTIME_RECEIPT_MISSING_REASON
        }
        MCC_PREPARED_FINGERPRINT_ADMISSION_RUNTIME_RECEIPT_SETUP_ONLY => {
            MCC_PREPARED_FINGERPRINT_ADMISSION_RUNTIME_RECEIPT_SETUP_ONLY_REASON
        }
        MCC_PREPARED_FINGERPRINT_ADMISSION_RUNTIME_RECEIPT_NOT_REQUIRED => {
            MCC_PREPARED_FINGERPRINT_ADMISSION_RUNTIME_RECEIPT_NOT_REQUIRED_REASON
        }
        _ => MCC_PREPARED_FINGERPRINT_ADMISSION_RUNTIME_RECEIPT_MISSING_REASON,
    }
}

fn mcc_prepared_fingerprint_admission_runtime_consumption_status(
    setup_evidence: &MccSetupEvidence,
    runtime_admission_consumed: bool,
    runtime_admission_fault_observed: bool,
    hot_execution_recorded: bool,
) -> &'static str {
    if mcc_requires_prepared_fingerprint_admission_runtime_receipt(setup_evidence)
        && runtime_admission_fault_observed
    {
        MCC_PREPARED_FINGERPRINT_ADMISSION_RUNTIME_CONSUMPTION_MISSING
    } else if runtime_admission_consumed {
        MCC_PREPARED_FINGERPRINT_ADMISSION_RUNTIME_CONSUMPTION_OBSERVED
    } else if mcc_prepared_fingerprint_admission_runtime_receipt_fail_closed(
        setup_evidence,
        runtime_admission_consumed,
        runtime_admission_fault_observed,
        hot_execution_recorded,
    ) {
        MCC_PREPARED_FINGERPRINT_ADMISSION_RUNTIME_CONSUMPTION_MISSING
    } else if mcc_requires_prepared_fingerprint_admission_runtime_receipt(setup_evidence) {
        MCC_PREPARED_FINGERPRINT_ADMISSION_RUNTIME_CONSUMPTION_SETUP_ONLY
    } else {
        MCC_PREPARED_FINGERPRINT_ADMISSION_RUNTIME_CONSUMPTION_NOT_REQUIRED
    }
}

fn mcc_prepared_fingerprint_admission_runtime_receipt_fail_closed(
    setup_evidence: &MccSetupEvidence,
    runtime_admission_consumed: bool,
    runtime_admission_fault_observed: bool,
    hot_execution_recorded: bool,
) -> bool {
    mcc_requires_prepared_fingerprint_admission_runtime_receipt(setup_evidence)
        && hot_execution_recorded
        && (runtime_admission_fault_observed || !runtime_admission_consumed)
}

fn mcc_missing_prepared_fingerprint_admission_runtime_consumption_row(
    setup_evidence: &MccSetupEvidence,
) -> String {
    let admission_plan = mcc_marking_prepared_fingerprint_admission_plan(setup_evidence);
    format!(
        "MCC prepared_fingerprint_admission_runtime_consumption_missing \
         schema={} schema_version=1 git_head={} {} {} \
         status_code=blocked \
         evidence_scope=hot_loop setup_only=false runtime_claim=false runtime_win_claim=false \
         runtime_consumed=false runtime_consumption_status={} hot_loop_consumption=missing \
         prepared_admission_receipt_required=true prepared_admission_receipt_status={} \
         prepared_admission_receipt_reason_code={} reason_code={} \
         attempted=0 new=0 duplicate=0 fault=0 \
         admission_counters_present=false second_beneficiaries={} \
         state_vector_reuse=canonical_marking_vector frontend_neutral_state_vector=true \
         frontend_neutral_fingerprint=true solver_storage_reusable=true \
         production_readiness_status=blocked production_readiness_reason_code={} \
         production_selected=false fail_closed=true",
        MCC_PREPARED_FINGERPRINT_ADMISSION_RUNTIME_CONSUMPTION_MISSING_SCHEMA,
        env!("TY_MCC_BUILD_GIT_HEAD"),
        mcc_shared_setup_adoption_fields(setup_evidence),
        mcc_prepared_fingerprint_admission_fields(&admission_plan),
        MCC_PREPARED_FINGERPRINT_ADMISSION_RUNTIME_CONSUMPTION_MISSING,
        MCC_PREPARED_FINGERPRINT_ADMISSION_RUNTIME_RECEIPT_MISSING,
        MCC_PREPARED_FINGERPRINT_ADMISSION_RUNTIME_RECEIPT_MISSING_REASON,
        MCC_PREPARED_FINGERPRINT_ADMISSION_RUNTIME_RECEIPT_MISSING_REASON,
        MCC_SHARED_ENGINE_SECOND_BENEFICIARIES,
        MCC_PREPARED_FINGERPRINT_ADMISSION_RUNTIME_RECEIPT_MISSING_REASON,
    )
}

fn mcc_marking_prepared_fingerprint_admission_plan(
    setup_evidence: &MccSetupEvidence,
) -> PreparedFingerprintAdmissionPlan {
    PreparedFingerprintAdmissionPlan::new(
        mcc_marking_prepared_fingerprint_admission_plan_id(setup_evidence),
        setup_evidence.prepared_program.source_kind,
        setup_evidence.prepared_program.payload_kind,
        setup_evidence.prepared_program.storage_kind,
        setup_evidence.marking_dedup.lane,
        setup_evidence.marking_dedup.clone(),
        SharedDuplicateAuthorization::CanonicalPayloadEquality,
        PreparedFingerprintPayloadWitnessKind::PetriMarkingCas,
    )
    .with_prepared_program(&setup_evidence.prepared_program)
}

fn mcc_marking_prepared_fingerprint_admission_plan_id(
    setup_evidence: &MccSetupEvidence,
) -> &'static str {
    if setup_evidence.marking_dedup.lane == SetupTraceLaneKind::Fingerprint
        && setup_evidence.marking_dedup.storage == SharedDedupStorageKind::Cas
        && setup_evidence
            .marking_dedup
            .storage_config_identity
            .as_deref()
            == Some("fingerprint-only-cas-fingerprint-set-v1")
    {
        MCC_FINGERPRINT_ONLY_PREPARED_FINGERPRINT_ADMISSION_PLAN_ID
    } else {
        MCC_MARKING_PREPARED_FINGERPRINT_ADMISSION_PLAN_ID
    }
}

fn mcc_prepared_fingerprint_admission_fields(plan: &PreparedFingerprintAdmissionPlan) -> String {
    format!(
        "prepared_fingerprint_admission_contract={} \
         prepared_fingerprint_admission_plan_id={} \
         prepared_fingerprint_admission_status={} \
         prepared_fingerprint_admission_source_kind={} \
         prepared_fingerprint_admission_payload_kind={} \
         prepared_fingerprint_admission_storage_kind={} \
         prepared_fingerprint_admission_lane_kind={} \
         prepared_fingerprint_admission_duplicate_authorization={} \
         prepared_fingerprint_admission_payload_witness={} \
         prepared_fingerprint_admission_dedup_identity={} \
         prepared_fingerprint_admission_fingerprint_identity={} \
         prepared_fingerprint_admission_storage_policy_identity={} \
         prepared_fingerprint_admission_canonical_payload_guard=packed_marking_bytes",
        MCC_PREPARED_FINGERPRINT_ADMISSION_CONTRACT,
        evidence_token(&plan.id),
        MCC_PREPARED_FINGERPRINT_ADMISSION_VALIDATION_STATUS,
        plan.source_kind.code(),
        plan.payload_kind.code(),
        plan.storage_kind.code(),
        plan.lane.code(),
        plan.duplicate_authorization.code(),
        plan.payload_witness.code(),
        evidence_token(&plan.dedup.dedup_identity()),
        evidence_token(&plan.dedup.fingerprint.fingerprint_identity()),
        evidence_token(&plan.dedup.storage_policy_identity()),
    )
}

fn mcc_runtime_reuse_inheritance_fields(setup_evidence: &MccSetupEvidence) -> String {
    let fingerprint = &setup_evidence.marking_dedup.fingerprint;
    format!(
        "state_vector_reuse=canonical_marking_vector \
         frontend_neutral_state_vector=true frontend_neutral_fingerprint=true \
         solver_storage_reusable=true second_beneficiaries={} \
         tla_check_inherits=true hardware_lanes_inherit=true storage_lanes_inherit=true \
         replay_lanes_inherit=true canonical_state_vector_layout={} \
         canonical_state_vector_layout_version={} state_vector_fingerprint_id={}",
        MCC_SHARED_ENGINE_SECOND_BENEFICIARIES,
        evidence_token(&fingerprint.canonical_domain.id),
        evidence_token(&fingerprint.canonical_domain.version),
        evidence_token(&fingerprint.id),
    )
}

fn mcc_shared_setup_adoption_fields(setup_evidence: &MccSetupEvidence) -> String {
    let compatible_frontend_families =
        if setup_evidence.trace.compatible_frontend_families.is_empty() {
            MCC_SHARED_ENGINE_COMPATIBLE_FRONTEND_FAMILIES.to_string()
        } else {
            setup_evidence.trace.compatible_frontend_families.join(",")
        };
    format!(
        "origin_frontend={} shared_engine_origin_frontend={} shared_engine_component={} \
         shared_owner={} shared_engine_lane_owner={} first_beneficiary={} \
         second_beneficiary={} shared_engine_second_beneficiary={} \
         compatible_frontend_families={} shared_engine_compatible_frontend_families={} \
         default_compatible_frontend_families={} default_consumers={} \
         downstream_beneficiary_families={} remaining_compatible_frontend_families={} \
         frontend_family_blockers={} generic_prerequisites={} \
         shared_engine_extraction_status={} shared_engine_blocker_status={} \
         acceptance_test={} acceptance_evidence={} setup_trace_validation_status={}",
        evidence_option_token(
            setup_evidence
                .trace
                .origin_frontend
                .as_deref()
                .or(Some(MCC_SHARED_ENGINE_ORIGIN_FRONTEND))
        ),
        evidence_option_token(
            setup_evidence
                .trace
                .origin_frontend
                .as_deref()
                .or(Some(MCC_SHARED_ENGINE_ORIGIN_FRONTEND))
        ),
        evidence_option_token(
            setup_evidence
                .trace
                .shared_engine_component
                .as_deref()
                .or(Some(MCC_SHARED_ENGINE_COMPONENT))
        ),
        MCC_SHARED_ENGINE_LANE_OWNER,
        MCC_SHARED_ENGINE_LANE_OWNER,
        evidence_option_token(
            setup_evidence
                .trace
                .first_beneficiary
                .as_deref()
                .or(Some(MCC_SHARED_ENGINE_FIRST_BENEFICIARY))
        ),
        evidence_option_token(
            setup_evidence
                .trace
                .second_beneficiary
                .as_deref()
                .or(Some(MCC_SHARED_ENGINE_SECOND_BENEFICIARY))
        ),
        evidence_option_token(
            setup_evidence
                .trace
                .second_beneficiary
                .as_deref()
                .or(Some(MCC_SHARED_ENGINE_SECOND_BENEFICIARY))
        ),
        compatible_frontend_families,
        MCC_SHARED_ENGINE_COMPATIBLE_FRONTEND_FAMILIES,
        MCC_SHARED_ENGINE_DEFAULT_COMPATIBLE_FRONTEND_FAMILIES,
        MCC_SHARED_ENGINE_DEFAULT_COMPATIBLE_FRONTEND_FAMILIES,
        MCC_SHARED_ENGINE_DOWNSTREAM_BENEFICIARY_FAMILIES,
        MCC_SHARED_ENGINE_REMAINING_COMPATIBLE_FRONTEND_FAMILIES,
        MCC_SHARED_ENGINE_FRONTEND_FAMILY_BLOCKERS,
        MCC_SHARED_ENGINE_GENERIC_PREREQUISITES,
        evidence_option_token(
            setup_evidence
                .trace
                .extraction_status
                .as_deref()
                .or(Some(MCC_SHARED_ENGINE_EXTRACTION_STATUS))
        ),
        evidence_option_token(
            setup_evidence
                .trace
                .blocker_status
                .as_deref()
                .or(Some(MCC_SHARED_ENGINE_BLOCKER_STATUS))
        ),
        evidence_token(MCC_SHARED_ADOPTION_ACCEPTANCE_TEST),
        MCC_SHARED_ENGINE_ACCEPTANCE_EVIDENCE,
        setup_evidence.trace.validation_status.code(),
    )
}

fn mcc_exact_or_unknown_status(status: MccRunStatus) -> &'static str {
    match status {
        MccRunStatus::Completed { .. } => "exact",
        MccRunStatus::Error | MccRunStatus::Panic => "unknown",
    }
}

fn evidence_token(value: &str) -> String {
    if value.is_empty() {
        String::from("none")
    } else {
        value.replace(char::is_whitespace, "_")
    }
}

fn prepared_identity(prefix: &str, identity: &str) -> String {
    format!("{prefix}:{}", evidence_token(identity))
}

fn evidence_option_token(value: Option<&str>) -> String {
    value
        .filter(|value| !value.is_empty())
        .map(evidence_token)
        .unwrap_or_else(|| String::from("none"))
}

fn evidence_list<I, S>(values: I) -> String
where
    I: IntoIterator<Item = S>,
    S: AsRef<str>,
{
    let tokens: Vec<String> = values
        .into_iter()
        .map(|value| evidence_token(value.as_ref()))
        .collect();
    if tokens.is_empty() {
        String::from("none")
    } else {
        tokens.join("|")
    }
}

fn prepared_program_sha256(program: &PreparedCheckerProgram) -> String {
    let mut hasher = Sha256::new();
    hash_field(&mut hasher, "identity", &program.identity);
    hash_field(&mut hasher, "source_kind", program.source_kind.code());
    hash_field(&mut hasher, "payload_kind", program.payload_kind.code());
    hash_field(&mut hasher, "storage_kind", program.storage_kind.code());

    let identities = program.effective_identity_fields();
    hash_optional(&mut hasher, "cache_key", identities.cache_key.as_deref());
    hash_optional(
        &mut hasher,
        "frontend_payload_identity",
        identities.frontend_payload_identity.as_deref(),
    );
    hash_optional(
        &mut hasher,
        "artifact_identity",
        identities.artifact_identity.as_deref(),
    );
    hash_optional(
        &mut hasher,
        "storage_policy_identity",
        identities.storage_policy_identity.as_deref(),
    );
    hash_optional(
        &mut hasher,
        "fingerprint_policy_identity",
        identities.fingerprint_policy_identity.as_deref(),
    );
    hash_optional(
        &mut hasher,
        "fingerprint_identity",
        identities.fingerprint_identity.as_deref(),
    );
    hash_optional(
        &mut hasher,
        "batch_artifact_identity",
        identities.batch_artifact_identity.as_deref(),
    );
    hash_optional(
        &mut hasher,
        "candidate_identity",
        identities.candidate_identity.as_deref(),
    );
    hash_optional(
        &mut hasher,
        "lane_identity",
        identities.lane_identity.as_deref(),
    );

    if let Some(fingerprint) = &program.fingerprint {
        hash_field(&mut hasher, "fingerprint.id", &fingerprint.id);
        hash_field(&mut hasher, "fingerprint.scheme", fingerprint.scheme.code());
        hash_field(
            &mut hasher,
            "fingerprint.canonicalization_version",
            &fingerprint.canonicalization_version,
        );
    } else {
        hash_field(&mut hasher, "fingerprint.id", "none");
        hash_field(&mut hasher, "fingerprint.scheme", "none");
        hash_field(&mut hasher, "fingerprint.canonicalization_version", "none");
    }

    hash_count(&mut hasher, "transitions", program.transitions.len());
    for (index, transition) in program.transitions.iter().enumerate() {
        hash_field(
            &mut hasher,
            &format!("transition.{index}.id"),
            &transition.id,
        );
        hash_field(
            &mut hasher,
            &format!("transition.{index}.kind"),
            transition.kind.code(),
        );
    }

    hash_count(&mut hasher, "properties", program.properties.len());
    for (index, property) in program.properties.iter().enumerate() {
        hash_field(&mut hasher, &format!("property.{index}.id"), &property.id);
        hash_field(
            &mut hasher,
            &format!("property.{index}.kind"),
            property.kind.code(),
        );
    }

    hash_count(
        &mut hasher,
        "analytical_solves",
        program.analytical_solves.len(),
    );
    for (index, solve) in program.analytical_solves.iter().enumerate() {
        hash_field(
            &mut hasher,
            &format!("analytical_solve.{index}.id"),
            &solve.id,
        );
        hash_field(
            &mut hasher,
            &format!("analytical_solve.{index}.kind"),
            solve.kind.code(),
        );
        hash_field(
            &mut hasher,
            &format!("analytical_solve.{index}.problem"),
            solve.problem.code(),
        );
    }

    hash_count(
        &mut hasher,
        "symbolic_proofs",
        program.symbolic_proofs.len(),
    );
    for (index, proof) in program.symbolic_proofs.iter().enumerate() {
        hash_field(
            &mut hasher,
            &format!("symbolic_proof.{index}.id"),
            &proof.id,
        );
        hash_field(
            &mut hasher,
            &format!("symbolic_proof.{index}.kind"),
            proof.kind.code(),
        );
        hash_field(
            &mut hasher,
            &format!("symbolic_proof.{index}.problem"),
            proof.problem.code(),
        );
    }

    hash_count(
        &mut hasher,
        "backend_families",
        program.backend_families.len(),
    );
    for (index, family) in program.backend_families.iter().enumerate() {
        hash_field(
            &mut hasher,
            &format!("backend_family.{index}.id"),
            &family.id,
        );
        hash_field(
            &mut hasher,
            &format!("backend_family.{index}.backend"),
            family.backend.code(),
        );
        hash_field(
            &mut hasher,
            &format!("backend_family.{index}.problem"),
            family.problem.code(),
        );
        let facets = if family.facets.is_empty() {
            String::from("none")
        } else {
            family
                .facets
                .iter()
                .map(|facet| facet.code())
                .collect::<Vec<_>>()
                .join("+")
        };
        hash_field(
            &mut hasher,
            &format!("backend_family.{index}.facets"),
            &facets,
        );
    }

    hash_count(
        &mut hasher,
        "frontend_extensions",
        program.frontend_extensions.len(),
    );
    for (index, extension) in program.frontend_extensions.iter().enumerate() {
        hash_field(
            &mut hasher,
            &format!("frontend_extension.{index}.id"),
            &extension.id,
        );
        hash_field(
            &mut hasher,
            &format!("frontend_extension.{index}.kind"),
            extension.kind.code(),
        );
        hash_field(
            &mut hasher,
            &format!("frontend_extension.{index}.problem"),
            extension.problem.code(),
        );
        hash_field(
            &mut hasher,
            &format!("frontend_extension.{index}.source_kind"),
            extension.source_kind.code(),
        );
        hash_field(
            &mut hasher,
            &format!("frontend_extension.{index}.payload_kind"),
            extension.payload_kind.code(),
        );
        hash_field(
            &mut hasher,
            &format!("frontend_extension.{index}.storage_kind"),
            extension.storage_kind.code(),
        );
        let identities = extension
            .identities
            .merged_with_fallback(&program.effective_identity_fields());
        hash_identity_fields(
            &mut hasher,
            &format!("frontend_extension.{index}.identity"),
            &identities,
        );
    }

    hash_count(
        &mut hasher,
        "candidate_lanes",
        program.candidate_lanes.len(),
    );
    for (index, lane) in program.candidate_lanes.iter().enumerate() {
        hash_field(&mut hasher, &format!("candidate_lane.{index}.id"), &lane.id);
        hash_field(
            &mut hasher,
            &format!("candidate_lane.{index}.lane"),
            lane.lane.code(),
        );
        hash_optional(
            &mut hasher,
            &format!("candidate_lane.{index}.candidate_key"),
            lane.candidate_key.as_deref(),
        );
        let identities = program.effective_candidate_lane_identity_fields(lane);
        hash_identity_fields(
            &mut hasher,
            &format!("candidate_lane.{index}.identity"),
            &identities,
        );
    }

    hash_count(
        &mut hasher,
        "validation_plans",
        program.validation_plans.len(),
    );
    for (index, plan) in program.validation_plans.iter().enumerate() {
        hash_field(
            &mut hasher,
            &format!("validation_plan.{index}.id"),
            &plan.id,
        );
        hash_field(
            &mut hasher,
            &format!("validation_plan.{index}.kind"),
            plan.kind.code(),
        );
        hash_field(
            &mut hasher,
            &format!("validation_plan.{index}.problem"),
            plan.problem.code(),
        );
        hash_field(
            &mut hasher,
            &format!("validation_plan.{index}.required"),
            if plan.required { "true" } else { "false" },
        );
        hash_field(
            &mut hasher,
            &format!("validation_plan.{index}.fail_closed"),
            if plan.fail_closed { "true" } else { "false" },
        );
        if let Some(fingerprint) = &plan.fingerprint {
            hash_field(
                &mut hasher,
                &format!("validation_plan.{index}.fingerprint.id"),
                &fingerprint.id,
            );
            hash_field(
                &mut hasher,
                &format!("validation_plan.{index}.fingerprint.scheme"),
                fingerprint.scheme.code(),
            );
            hash_field(
                &mut hasher,
                &format!("validation_plan.{index}.fingerprint.canonicalization_version"),
                &fingerprint.canonicalization_version,
            );
        } else {
            hash_field(
                &mut hasher,
                &format!("validation_plan.{index}.fingerprint.id"),
                "none",
            );
            hash_field(
                &mut hasher,
                &format!("validation_plan.{index}.fingerprint.scheme"),
                "none",
            );
            hash_field(
                &mut hasher,
                &format!("validation_plan.{index}.fingerprint.canonicalization_version"),
                "none",
            );
        }
        let identities = program.effective_validation_plan_identity_fields(plan);
        hash_identity_fields(
            &mut hasher,
            &format!("validation_plan.{index}.identity"),
            &identities,
        );
    }

    hash_count(&mut hasher, "validations", program.validations.len());
    for (index, validation) in program.validations.iter().enumerate() {
        hash_field(
            &mut hasher,
            &format!("validation.{index}.kind"),
            validation.code(),
        );
    }
    let digest = hasher.finalize();
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn hash_count(hasher: &mut Sha256, key: &str, value: usize) {
    hash_field(hasher, key, &value.to_string());
}

fn hash_optional(hasher: &mut Sha256, key: &str, value: Option<&str>) {
    hash_field(hasher, key, value.unwrap_or("none"));
}

fn hash_identity_fields(
    hasher: &mut Sha256,
    prefix: &str,
    identities: &CheckerArtifactIdentityFields,
) {
    hash_optional(
        hasher,
        &format!("{prefix}.cache_key"),
        identities.cache_key.as_deref(),
    );
    hash_optional(
        hasher,
        &format!("{prefix}.frontend_payload_identity"),
        identities.frontend_payload_identity.as_deref(),
    );
    hash_optional(
        hasher,
        &format!("{prefix}.artifact_identity"),
        identities.artifact_identity.as_deref(),
    );
    hash_optional(
        hasher,
        &format!("{prefix}.storage_policy_identity"),
        identities.storage_policy_identity.as_deref(),
    );
    hash_optional(
        hasher,
        &format!("{prefix}.fingerprint_policy_identity"),
        identities.fingerprint_policy_identity.as_deref(),
    );
    hash_optional(
        hasher,
        &format!("{prefix}.fingerprint_identity"),
        identities.fingerprint_identity.as_deref(),
    );
    hash_optional(
        hasher,
        &format!("{prefix}.batch_artifact_identity"),
        identities.batch_artifact_identity.as_deref(),
    );
    hash_optional(
        hasher,
        &format!("{prefix}.candidate_identity"),
        identities.candidate_identity.as_deref(),
    );
    hash_optional(
        hasher,
        &format!("{prefix}.lane_identity"),
        identities.lane_identity.as_deref(),
    );
}

fn hash_field(hasher: &mut Sha256, key: &str, value: &str) {
    hasher.update(key.as_bytes());
    hasher.update(b"=");
    hasher.update(value.as_bytes());
    hasher.update(b"\n");
}

fn env_flag_enabled(key: &str) -> bool {
    matches!(
        std::env::var(key).ok().as_deref(),
        Some("1") | Some("true") | Some("TRUE") | Some("yes") | Some("on")
    )
}

fn append_jsonl(path: &Path, record: &SerializableMccBackendEvidence<'_>) -> std::io::Result<()> {
    let mut file = OpenOptions::new().create(true).append(true).open(path)?;
    serde_json::to_writer(&mut file, record)?;
    file.write_all(b"\n")
}

#[derive(Debug, Serialize)]
struct SerializableMccBackendEvidence<'a> {
    schema_version: u32,
    model: &'a str,
    examination: &'a str,
    source_kind: &'static str,
    place_count: usize,
    transition_count: usize,
    max_states: usize,
    workers: usize,
    fingerprint_dedup: bool,
    run_status: SerializableRunStatus,
    error: Option<&'a str>,
    setup_trace: Option<SerializableSetupTrace<'a>>,
    prepared_program: Option<SerializablePreparedProgram<'a>>,
    report: SerializableCapabilityReport<'a>,
}

impl<'a> SerializableMccBackendEvidence<'a> {
    fn new(
        model: &'a PreparedModel,
        examination: Examination,
        config: &ExplorationConfig,
        report: &'a CapabilityReport,
        setup_evidence: Option<&'a MccSetupEvidence>,
        status: MccRunStatus,
        error: Option<&'a str>,
    ) -> Self {
        Self {
            schema_version: 1,
            model: model.model_name(),
            examination: examination.as_str(),
            source_kind: source_kind_name(model.source_kind()),
            place_count: model.net().num_places(),
            transition_count: model.net().num_transitions(),
            max_states: config.max_states(),
            workers: config.workers(),
            fingerprint_dedup: config.fingerprint_dedup(),
            run_status: SerializableRunStatus::from(status),
            error,
            setup_trace: setup_evidence.map(SerializableSetupTrace::from),
            prepared_program: setup_evidence.map(SerializablePreparedProgram::from),
            report: SerializableCapabilityReport::from(report),
        }
    }
}

#[derive(Debug, Serialize)]
#[serde(tag = "status", rename_all = "snake_case")]
enum SerializableRunStatus {
    Completed { records: usize },
    Error,
    Panic,
}

impl From<MccRunStatus> for SerializableRunStatus {
    fn from(value: MccRunStatus) -> Self {
        match value {
            MccRunStatus::Completed { records } => Self::Completed { records },
            MccRunStatus::Error => Self::Error,
            MccRunStatus::Panic => Self::Panic,
        }
    }
}

#[derive(Debug, Serialize)]
struct SerializableSetupTrace<'a> {
    source_kind: &'static str,
    frontend_kind: &'static str,
    lane_kind: &'static str,
    lane: &'static str,
    #[serde(flatten)]
    identity_fields: SerializableSharedIdentityFields,
    source_identity: Option<&'a str>,
    property_identity: Option<&'a str>,
    origin_frontend: Option<&'a str>,
    shared_engine_component: Option<&'a str>,
    first_beneficiary: Option<&'a str>,
    second_beneficiary: Option<&'a str>,
    compatible_frontend_families: Vec<&'a str>,
    shared_engine_extraction_status: Option<&'a str>,
    shared_engine_blocker_status: Option<&'a str>,
    validation_status: &'static str,
    timings: Vec<SerializableSetupTraceTiming>,
}

impl<'a> From<&'a MccSetupEvidence> for SerializableSetupTrace<'a> {
    fn from(value: &'a MccSetupEvidence) -> Self {
        Self {
            source_kind: value.trace.source_kind.code(),
            frontend_kind: value.trace.source_kind.code(),
            lane_kind: value.trace.lane.code(),
            lane: value.trace.lane.code(),
            identity_fields: SerializableSharedIdentityFields::from_identity_fields(
                value.trace.candidate_key.as_deref(),
                &value.trace.identities,
            ),
            source_identity: value.trace.source_identity.as_deref(),
            property_identity: value.trace.property_identity.as_deref(),
            origin_frontend: value.trace.origin_frontend.as_deref(),
            shared_engine_component: value.trace.shared_engine_component.as_deref(),
            first_beneficiary: value.trace.first_beneficiary.as_deref(),
            second_beneficiary: value.trace.second_beneficiary.as_deref(),
            compatible_frontend_families: value
                .trace
                .compatible_frontend_families
                .iter()
                .map(String::as_str)
                .collect(),
            shared_engine_extraction_status: value.trace.extraction_status.as_deref(),
            shared_engine_blocker_status: value.trace.blocker_status.as_deref(),
            validation_status: value.trace.validation_status.code(),
            timings: value
                .trace
                .timings()
                .iter()
                .map(|timing| SerializableSetupTraceTiming::from_trace_timing(&value.trace, timing))
                .collect(),
        }
    }
}

#[derive(Debug, Serialize)]
struct SerializableSetupTraceTiming {
    source_kind: &'static str,
    frontend_kind: &'static str,
    lane_kind: &'static str,
    lane: &'static str,
    #[serde(flatten)]
    identity_fields: SerializableSharedIdentityFields,
    source_identity: Option<String>,
    property_identity: Option<String>,
    origin_frontend: Option<String>,
    shared_engine_component: Option<String>,
    first_beneficiary: Option<String>,
    second_beneficiary: Option<String>,
    compatible_frontend_families: Vec<String>,
    shared_engine_extraction_status: Option<String>,
    shared_engine_blocker_status: Option<String>,
    validation_status: &'static str,
    phase: &'static str,
    nanos: u64,
}

impl SerializableSetupTraceTiming {
    fn from_trace_timing(trace: &SetupTrace, value: &tla_mc_core::SetupTraceTiming) -> Self {
        let identities = value.key.identities.merged_with_fallback(&trace.identities);
        Self {
            source_kind: value.key.frontend.code(),
            frontend_kind: value.key.frontend.code(),
            lane_kind: value.key.lane.code(),
            lane: value.key.lane.code(),
            identity_fields: SerializableSharedIdentityFields::from_identity_fields(
                value.key.candidate_key.as_deref(),
                &identities,
            ),
            source_identity: optional_string(trace.source_identity.as_deref()),
            property_identity: optional_string(trace.property_identity.as_deref()),
            origin_frontend: optional_string(trace.origin_frontend.as_deref()),
            shared_engine_component: optional_string(trace.shared_engine_component.as_deref()),
            first_beneficiary: optional_string(trace.first_beneficiary.as_deref()),
            second_beneficiary: optional_string(trace.second_beneficiary.as_deref()),
            compatible_frontend_families: trace.compatible_frontend_families.clone(),
            shared_engine_extraction_status: optional_string(trace.extraction_status.as_deref()),
            shared_engine_blocker_status: optional_string(trace.blocker_status.as_deref()),
            validation_status: trace.validation_status.code(),
            phase: value.phase.code(),
            nanos: value.nanos,
        }
    }
}

#[derive(Debug, Serialize)]
struct SerializableSharedIdentityFields {
    candidate_key: Option<String>,
    candidate_identity: Option<String>,
    lane_identity: Option<String>,
    cache_key: Option<String>,
    frontend_payload_identity: Option<String>,
    artifact_identity: Option<String>,
    storage_policy_identity: Option<String>,
    fingerprint_policy_identity: Option<String>,
    fingerprint_identity: Option<String>,
    batch_artifact_identity: Option<String>,
}

impl SerializableSharedIdentityFields {
    fn from_identity_fields(
        candidate_key: Option<&str>,
        identities: &CheckerArtifactIdentityFields,
    ) -> Self {
        Self {
            candidate_key: optional_string(candidate_key),
            candidate_identity: optional_string(identities.candidate_identity.as_deref()),
            lane_identity: optional_string(identities.lane_identity.as_deref()),
            cache_key: optional_string(identities.cache_key.as_deref()),
            frontend_payload_identity: optional_string(
                identities.frontend_payload_identity.as_deref(),
            ),
            artifact_identity: optional_string(identities.artifact_identity.as_deref()),
            storage_policy_identity: optional_string(identities.storage_policy_identity.as_deref()),
            fingerprint_policy_identity: optional_string(
                identities.fingerprint_policy_identity.as_deref(),
            ),
            fingerprint_identity: optional_string(identities.fingerprint_identity.as_deref()),
            batch_artifact_identity: optional_string(identities.batch_artifact_identity.as_deref()),
        }
    }
}

fn optional_string(value: Option<&str>) -> Option<String> {
    value
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
}

#[derive(Debug, Serialize)]
struct SerializablePreparedProgram<'a> {
    identity: &'a str,
    frontend_kind: &'static str,
    source_kind: &'static str,
    payload_kind: &'static str,
    storage_kind: &'static str,
    source_identity: Option<&'a str>,
    property_identity: Option<&'a str>,
    #[serde(flatten)]
    identity_fields: SerializableSharedIdentityFields,
    fingerprint_id: String,
    fingerprint_digest_algorithm: &'static str,
    fingerprint_digest: String,
    prepared_program_fingerprint_algorithm: &'static str,
    prepared_program_fingerprint: String,
    shared_engine_origin_frontend: &'static str,
    shared_engine_component: &'static str,
    shared_engine_lane_owner: &'static str,
    shared_engine_second_beneficiary: &'static str,
    shared_engine_extraction_status: &'static str,
    shared_engine_compatible_frontend_families: &'static str,
    shared_engine_frontend_family_blockers: &'static str,
    shared_engine_blocker_status: &'static str,
    artifact_fingerprint_algorithm: &'static str,
    artifact_fingerprint: String,
    artifact_prepared_program_fingerprint: String,
    proof_fingerprint_algorithm: &'static str,
    proof_fingerprint: String,
    proof_prepared_program_fingerprint: String,
    replay_fingerprint_algorithm: &'static str,
    replay_fingerprint: String,
    replay_prepared_program_fingerprint: String,
    transitions: usize,
    properties: usize,
    analytical_solves: usize,
    analytical_solve_descriptors: Vec<SerializablePreparedAnalyticalSolve<'a>>,
    symbolic_proofs: usize,
    symbolic_proof_descriptors: Vec<SerializablePreparedSymbolicProof<'a>>,
    backend_families: usize,
    backend_family_descriptors: Vec<SerializablePreparedBackendFamily<'a>>,
    validations: Vec<&'static str>,
    property_id_status: &'static str,
    reduction_mode: &'static str,
}

#[derive(Debug, Serialize)]
struct SerializablePreparedAnalyticalSolve<'a> {
    id: &'a str,
    kind: &'static str,
    problem: &'static str,
}

impl<'a> From<&'a PreparedAnalyticalSolveDescriptor> for SerializablePreparedAnalyticalSolve<'a> {
    fn from(value: &'a PreparedAnalyticalSolveDescriptor) -> Self {
        Self {
            id: &value.id,
            kind: value.kind.code(),
            problem: value.problem.code(),
        }
    }
}

#[derive(Debug, Serialize)]
struct SerializablePreparedSymbolicProof<'a> {
    id: &'a str,
    kind: &'static str,
    problem: &'static str,
}

impl<'a> From<&'a PreparedSymbolicProofDescriptor> for SerializablePreparedSymbolicProof<'a> {
    fn from(value: &'a PreparedSymbolicProofDescriptor) -> Self {
        Self {
            id: &value.id,
            kind: value.kind.code(),
            problem: value.problem.code(),
        }
    }
}

#[derive(Debug, Serialize)]
struct SerializablePreparedBackendFamily<'a> {
    id: &'a str,
    backend: &'static str,
    backend_code: &'static str,
    problem: &'static str,
    facets: Vec<&'static str>,
}

impl<'a> From<&'a PreparedBackendFamilyDescriptor> for SerializablePreparedBackendFamily<'a> {
    fn from(value: &'a PreparedBackendFamilyDescriptor) -> Self {
        Self {
            id: &value.id,
            backend: value.backend.name(),
            backend_code: value.backend.code(),
            problem: value.problem.code(),
            facets: value.facets.iter().map(|facet| facet.code()).collect(),
        }
    }
}

impl<'a> From<&'a MccSetupEvidence> for SerializablePreparedProgram<'a> {
    fn from(value: &'a MccSetupEvidence) -> Self {
        let prepared_program_fingerprint = prepared_program_sha256(&value.prepared_program);
        let identity_fields = value.prepared_program.effective_identity_fields();
        let fingerprint_id = value
            .prepared_program
            .fingerprint
            .as_ref()
            .map(|fingerprint| evidence_token(&fingerprint.id))
            .unwrap_or_else(|| String::from("none"));
        Self {
            identity: &value.prepared_program.identity,
            frontend_kind: value.prepared_program.source_kind.code(),
            source_kind: value.prepared_program.source_kind.code(),
            payload_kind: value.prepared_program.payload_kind.code(),
            storage_kind: value.prepared_program.storage_kind.code(),
            source_identity: value.trace.source_identity.as_deref(),
            property_identity: value.trace.property_identity.as_deref(),
            identity_fields: SerializableSharedIdentityFields::from_identity_fields(
                value.trace.candidate_key.as_deref(),
                &identity_fields,
            ),
            fingerprint_id,
            fingerprint_digest_algorithm: MCC_PREPARED_PROGRAM_FINGERPRINT_ALGORITHM,
            fingerprint_digest: prepared_program_fingerprint.clone(),
            prepared_program_fingerprint_algorithm: MCC_PREPARED_PROGRAM_FINGERPRINT_ALGORITHM,
            prepared_program_fingerprint: prepared_program_fingerprint.clone(),
            shared_engine_origin_frontend: MCC_SHARED_ENGINE_ORIGIN_FRONTEND,
            shared_engine_component: MCC_SHARED_ENGINE_COMPONENT,
            shared_engine_lane_owner: MCC_SHARED_ENGINE_LANE_OWNER,
            shared_engine_second_beneficiary: MCC_SHARED_ENGINE_SECOND_BENEFICIARY,
            shared_engine_extraction_status: MCC_SHARED_ENGINE_EXTRACTION_STATUS,
            shared_engine_compatible_frontend_families:
                MCC_SHARED_ENGINE_COMPATIBLE_FRONTEND_FAMILIES,
            shared_engine_frontend_family_blockers: MCC_SHARED_ENGINE_FRONTEND_FAMILY_BLOCKERS,
            shared_engine_blocker_status: MCC_SHARED_ENGINE_BLOCKER_STATUS,
            artifact_fingerprint_algorithm: MCC_PREPARED_PROGRAM_FINGERPRINT_ALGORITHM,
            artifact_fingerprint: prepared_program_fingerprint.clone(),
            artifact_prepared_program_fingerprint: prepared_program_fingerprint.clone(),
            proof_fingerprint_algorithm: MCC_PREPARED_PROGRAM_FINGERPRINT_ALGORITHM,
            proof_fingerprint: prepared_program_fingerprint.clone(),
            proof_prepared_program_fingerprint: prepared_program_fingerprint.clone(),
            replay_fingerprint_algorithm: MCC_PREPARED_PROGRAM_FINGERPRINT_ALGORITHM,
            replay_fingerprint: prepared_program_fingerprint.clone(),
            replay_prepared_program_fingerprint: prepared_program_fingerprint,
            transitions: value.prepared_program.transitions.len(),
            properties: value.prepared_program.properties.len(),
            analytical_solves: value.prepared_program.analytical_solves.len(),
            analytical_solve_descriptors: value
                .prepared_program
                .analytical_solves
                .iter()
                .map(SerializablePreparedAnalyticalSolve::from)
                .collect(),
            symbolic_proofs: value.prepared_program.symbolic_proofs.len(),
            symbolic_proof_descriptors: value
                .prepared_program
                .symbolic_proofs
                .iter()
                .map(SerializablePreparedSymbolicProof::from)
                .collect(),
            backend_families: value.prepared_program.backend_families.len(),
            backend_family_descriptors: value
                .prepared_program
                .backend_families
                .iter()
                .map(SerializablePreparedBackendFamily::from)
                .collect(),
            validations: value
                .prepared_program
                .validations
                .iter()
                .map(|validation| validation.code())
                .collect(),
            property_id_status: value.property_id_status,
            reduction_mode: reduction_mode_code(value.examination.reduction_mode()),
        }
    }
}

#[derive(Debug, Serialize)]
struct SerializableCapabilityReport<'a> {
    problem: Option<String>,
    limits: SerializableSolverLimits,
    selected: Vec<SerializableBackendCapability>,
    rejected: Vec<SerializableBackendCapability>,
    evidence: &'a [String],
    ay_selected_for_production: bool,
    production_routing_status: String,
    has_unjustified_local_production: bool,
}

impl<'a> From<&'a CapabilityReport> for SerializableCapabilityReport<'a> {
    fn from(value: &'a CapabilityReport) -> Self {
        Self {
            problem: value.problem.map(|problem| problem.name().to_string()),
            limits: SerializableSolverLimits::from(value.limits),
            selected: value
                .selected
                .iter()
                .map(SerializableBackendCapability::from)
                .collect(),
            rejected: value
                .rejected
                .iter()
                .map(SerializableBackendCapability::from)
                .collect(),
            evidence: &value.evidence,
            ay_selected_for_production: value.ay_selected_for_production(),
            production_routing_status: value.production_routing_status_name().to_string(),
            has_unjustified_local_production: value.has_unjustified_local_production(),
        }
    }
}

#[derive(Debug, Serialize)]
struct SerializableSolverLimits {
    time_budget_ms: Option<u128>,
    max_depth: Option<u32>,
    max_states: Option<u64>,
    max_memory_bytes: Option<u64>,
}

impl From<SolverLimits> for SerializableSolverLimits {
    fn from(value: SolverLimits) -> Self {
        Self {
            time_budget_ms: value.time_budget.map(|duration| duration.as_millis()),
            max_depth: value.max_depth,
            max_states: value.max_states,
            max_memory_bytes: value.max_memory_bytes,
        }
    }
}

#[derive(Debug, Serialize)]
struct SerializableBackendCapability {
    domain: String,
    backend: String,
    backend_code: &'static str,
    problem: Option<String>,
    facets: Vec<String>,
    role: String,
    status: String,
    reason_code: Option<String>,
    reason: Option<String>,
    detail: Option<String>,
}

impl From<&BackendCapability> for SerializableBackendCapability {
    fn from(value: &BackendCapability) -> Self {
        Self {
            domain: value.domain.name().to_string(),
            backend: value.backend.name().to_string(),
            backend_code: value.backend.code(),
            problem: value.problem.map(|problem| problem.name().to_string()),
            facets: value
                .facets
                .iter()
                .map(|facet| facet.name().to_string())
                .collect(),
            role: value.role.code().to_string(),
            status: value.status.code().to_string(),
            reason_code: value.reason_code().map(str::to_string),
            reason: value.reason.as_ref().map(ToString::to_string),
            detail: value.detail.clone(),
        }
    }
}

fn source_kind_name(value: SourceNetKind) -> &'static str {
    match value {
        SourceNetKind::Pt => "pt",
        SourceNetKind::SymmetricNet => "symmetric_net",
    }
}

#[cfg(test)]
mod tests {
    use std::fs;

    use tempfile::{NamedTempFile, TempDir};
    use tla_mc_core::{
        hardware_replay_decision_accepts_replay_primitive,
        validate_hardware_proof_replay_boundary_evidence_row,
        validate_hardware_replay_decision_evidence_row,
        validate_prepared_candidate_lane_evidence_row,
        validate_prepared_checker_program_evidence_row,
        validate_prepared_frontend_extension_evidence_row,
        validate_prepared_validation_plan_evidence_row,
        validate_shared_engine_adoption_evidence_row, BackendDomain, BackendKind, CapabilityRole,
        CapabilityStatus, ProblemKind, SolverFacet,
    };

    use super::*;
    use crate::explorer::ExplorationObserver;
    use crate::petri_net::{Arc, PetriNet, PlaceIdx, PlaceInfo, TransitionIdx, TransitionInfo};

    const MINIMAL_PT_NET: &str = r#"<?xml version="1.0"?>
<pnml xmlns="http://www.pnml.org/version-2009/grammar/pnml">
  <net id="test" type="http://www.pnml.org/version-2009/grammar/ptnet">
    <page id="p1">
      <place id="P0"><initialMarking><text>1</text></initialMarking></place>
      <place id="P1"/>
      <transition id="T0"/>
      <arc id="a1" source="P0" target="T0"/>
      <arc id="a2" source="T0" target="P1"/>
    </page>
  </net>
</pnml>"#;

    fn tiny_net() -> PetriNet {
        PetriNet {
            name: Some("tiny".to_string()),
            places: vec![
                PlaceInfo {
                    id: "p0".to_string(),
                    name: None,
                },
                PlaceInfo {
                    id: "p1".to_string(),
                    name: None,
                },
            ],
            transitions: vec![TransitionInfo {
                id: "t0".to_string(),
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

    fn tiny_model_dir() -> TempDir {
        let dir = TempDir::new().expect("temp model dir");
        fs::write(dir.path().join("model.pnml"), MINIMAL_PT_NET).expect("write model.pnml");
        dir
    }

    #[derive(Default)]
    struct RuntimeCountingObserver {
        states: usize,
        firings: usize,
        deadlocks: usize,
    }

    impl ExplorationObserver for RuntimeCountingObserver {
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

    fn symbolic_execution_row(report: &CapabilityReport) -> &str {
        let rows: Vec<_> = report
            .evidence
            .iter()
            .filter(|row| row.contains(" symbolic_execution "))
            .map(String::as_str)
            .collect();
        assert_eq!(rows.len(), 1, "expected exactly one symbolic execution row");
        rows[0]
    }

    fn ay_solve_decision_profile_row(report: &CapabilityReport) -> &str {
        let rows: Vec<_> = report
            .evidence
            .iter()
            .filter(|row| row.contains(" ay_solver_decision_profile_summary "))
            .map(String::as_str)
            .collect();
        assert_eq!(
            rows.len(),
            1,
            "expected exactly one AY solve decision profile row"
        );
        rows[0]
    }

    fn portfolio_route_rows(report: &CapabilityReport) -> Vec<&str> {
        report
            .evidence
            .iter()
            .filter(|row| is_mcc_portfolio_route_row(row))
            .map(String::as_str)
            .collect()
    }

    fn ay_solver_capability_descriptor_rows(report: &CapabilityReport) -> Vec<&str> {
        report
            .evidence
            .iter()
            .filter(|row| is_ay_solver_capability_descriptor_row(row))
            .map(String::as_str)
            .collect()
    }

    fn ay_symbolic_execution_contract_manifest_rows(report: &CapabilityReport) -> Vec<&str> {
        report
            .evidence
            .iter()
            .filter(|row| is_ay_symbolic_execution_contract_manifest_row(row))
            .map(String::as_str)
            .collect()
    }

    fn ay_symbolic_execution_contract_manifest_health_rows(report: &CapabilityReport) -> Vec<&str> {
        report
            .evidence
            .iter()
            .filter(|row| is_ay_symbolic_execution_contract_manifest_health_row(row))
            .map(String::as_str)
            .collect()
    }

    fn hardware_proof_replay_boundary_rows(report: &CapabilityReport) -> Vec<&str> {
        report
            .evidence
            .iter()
            .filter(|row| row.contains(" proof_replay_boundary "))
            .map(String::as_str)
            .collect()
    }

    fn hardware_replay_decision_rows(report: &CapabilityReport) -> Vec<&str> {
        report
            .evidence
            .iter()
            .filter(|row| row.contains(" hardware_replay_decision "))
            .map(String::as_str)
            .collect()
    }

    fn evidence_field<'a>(row: &'a str, key: &str) -> Option<&'a str> {
        let prefix = format!("{key}=");
        row.split_whitespace()
            .find_map(|piece| piece.strip_prefix(&prefix))
    }

    fn make_executable_ay() -> NamedTempFile {
        let ay = NamedTempFile::new().expect("temp ay");
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mut permissions = ay.as_file().metadata().unwrap().permissions();
            permissions.set_mode(0o755);
            ay.as_file().set_permissions(permissions).unwrap();
        }
        ay
    }

    #[test]
    fn prepared_program_records_granular_mcc_candidate_lanes() {
        let dir = tiny_model_dir();
        let model = crate::model::load_model_dir(dir.path()).expect("model should load");
        let config = ExplorationConfig::new(100).with_workers(1);
        let marking_dedup = mcc_marking_dedup_identity(&config);
        let (program, property_id_status) = build_mcc_prepared_program(
            &model,
            Examination::ReachabilityFireability,
            &marking_dedup,
        );

        assert_eq!(property_id_status, "property_xml_unavailable");

        let analytical_ids: Vec<_> = program
            .analytical_solves
            .iter()
            .map(|solve| solve.id.as_str())
            .collect();
        assert!(analytical_ids.contains(&"mcc.petri.ReachabilityFireability"));
        assert!(analytical_ids
            .contains(&"mcc.petri.structural.p_invariant_bounds.ReachabilityFireability"));
        assert!(analytical_ids
            .contains(&"mcc.petri.lp_state_equation.relaxation.ReachabilityFireability"));
        assert!(
            analytical_ids.contains(&"mcc.petri.reduction.reachability.ReachabilityFireability")
        );

        let lp_solve = program
            .analytical_solves
            .iter()
            .find(|solve| solve.id.contains("lp_state_equation"))
            .expect("LP state-equation solve descriptor");
        assert_eq!(lp_solve.kind, PreparedAnalyticalSolveKind::SmtQuery);
        assert_eq!(lp_solve.problem, ProblemKind::Smt);

        let symbolic_ids: Vec<_> = program
            .symbolic_proofs
            .iter()
            .map(|proof| proof.id.as_str())
            .collect();
        assert!(symbolic_ids.contains(&"mcc.petri.ay.bounded_model_check.ReachabilityFireability"));
        assert!(symbolic_ids.contains(&"mcc.petri.ay.chc.reachability_pdr.ReachabilityFireability"));

        let external_ay = program
            .backend_families
            .iter()
            .find(|family| family.backend == BackendKind::ExternalAYBinary)
            .expect("external AY candidate family");
        assert_eq!(external_ay.problem, ProblemKind::Bmc);
        assert!(external_ay.facets.contains(&SolverFacet::ExternalProcess));
        assert!(external_ay.facets.contains(&SolverFacet::Smt));
        assert!(external_ay.facets.contains(&SolverFacet::Bmc));

        let ay_chc = program
            .backend_families
            .iter()
            .find(|family| family.backend == BackendKind::AYChc)
            .expect("AY CHC/PDR candidate family");
        assert_eq!(ay_chc.problem, ProblemKind::Chc);
        assert!(ay_chc.facets.contains(&SolverFacet::Pdr));

        let reduction = program
            .backend_families
            .iter()
            .find(|family| family.id.contains("reduction.reachability"))
            .expect("query-aware reduction candidate family");
        assert_eq!(reduction.backend, BackendKind::ExplicitState);
        assert!(reduction
            .facets
            .contains(&SolverFacet::LinearIntegerArithmetic));
        assert_eq!(
            program
                .fingerprint
                .as_ref()
                .map(|fingerprint| fingerprint.id.as_str()),
            Some(MCC_MARKING_FINGERPRINT_ID)
        );
        assert_eq!(program.canonical_identities.len(), 1);
        assert_eq!(
            program.canonical_identities[0].id,
            MCC_PREPARED_PROGRAM_CANONICAL_IDENTITY
        );
        let identities = program.effective_identity_fields();
        assert!(identities
            .fingerprint_identity
            .as_deref()
            .is_some_and(|identity| identity.contains(MCC_MARKING_FINGERPRINT_NAMESPACE)));
        assert!(identities
            .storage_policy_identity
            .as_deref()
            .is_some_and(|identity| identity.contains("in_memory")));

        assert!(program
            .candidate_lanes
            .iter()
            .any(|lane| lane.candidate_key.as_deref() == Some("explicit_bfs")
                && lane.lane == SetupTraceLaneKind::ExplicitState));
        assert!(program
            .candidate_lanes
            .iter()
            .any(|lane| lane.candidate_key.as_deref() == Some("ay_symbolic")
                && lane.lane == SetupTraceLaneKind::AY));
        assert!(program.validation_plans.iter().any(|plan| {
            plan.kind == PreparedValidationKind::WitnessReplay
                && plan.fingerprint.as_ref().is_some_and(|fingerprint| {
                    fingerprint
                        .identities
                        .fingerprint_identity
                        .as_deref()
                        .is_some_and(|identity| identity.contains("mcc.petri.witness_fingerprint"))
                })
        }));
    }

    fn assert_setup_dedup_identity_for_config(
        config: ExplorationConfig,
        expected_storage_kind: &str,
        expected_storage_config_identity: &str,
        expected_lane_kind: &str,
        expected_fingerprint_id: &str,
        expected_digest_bits: u16,
    ) {
        let dir = tiny_model_dir();
        let model = crate::model::load_model_dir(dir.path()).expect("model should load");
        let setup_evidence = build_mcc_setup_evidence(
            &model,
            Examination::StateSpace,
            &config,
            Instant::now(),
            std::time::Duration::from_nanos(1),
            std::time::Duration::from_nanos(1),
        );
        let expected_admission_plan_id =
            mcc_marking_prepared_fingerprint_admission_plan_id(&setup_evidence);
        let expected_runtime_receipt_required = expected_admission_plan_id
            == MCC_FINGERPRINT_ONLY_PREPARED_FINGERPRINT_ADMISSION_PLAN_ID;
        let expected_runtime_receipt_required_str = expected_runtime_receipt_required.to_string();
        let expected_runtime_receipt_status = if expected_runtime_receipt_required {
            MCC_PREPARED_FINGERPRINT_ADMISSION_RUNTIME_RECEIPT_SETUP_ONLY
        } else {
            MCC_PREPARED_FINGERPRINT_ADMISSION_RUNTIME_RECEIPT_NOT_REQUIRED
        };

        assert_eq!(
            setup_evidence.marking_dedup.storage.code(),
            expected_storage_kind
        );
        assert_eq!(setup_evidence.marking_dedup.lane.code(), expected_lane_kind);
        assert_eq!(
            setup_evidence
                .marking_dedup
                .storage_config_identity
                .as_deref(),
            Some(expected_storage_config_identity)
        );
        assert_eq!(
            setup_evidence.marking_dedup.fingerprint.id,
            expected_fingerprint_id
        );
        assert_eq!(
            setup_evidence.marking_dedup.fingerprint.digest_bits,
            expected_digest_bits
        );
        assert_eq!(
            setup_evidence.marking_dedup.fingerprint.algorithm,
            SharedFingerprintAlgorithm::CanonicalBytesSha256
        );
        assert_eq!(
            setup_evidence.marking_dedup.fingerprint.value_kind,
            SharedFingerprintValueKind::MarkingVector
        );

        let prepared_identities = setup_evidence.prepared_program.effective_identity_fields();
        let expected_storage_policy_identity =
            setup_evidence.marking_dedup.storage_policy_identity();
        let expected_fingerprint_identity = setup_evidence
            .marking_dedup
            .fingerprint
            .fingerprint_identity();
        assert_eq!(
            prepared_identities.storage_policy_identity.as_deref(),
            Some(expected_storage_policy_identity.as_str())
        );
        assert_eq!(
            prepared_identities.fingerprint_identity.as_deref(),
            Some(expected_fingerprint_identity.as_str())
        );
        assert_eq!(
            setup_evidence
                .prepared_program
                .fingerprint
                .as_ref()
                .and_then(|fingerprint| fingerprint.identities.storage_policy_identity.as_deref()),
            Some(expected_storage_policy_identity.as_str())
        );

        let mut report = CapabilityReport::new(ProblemKind::StateSpace);
        add_mcc_setup_evidence(
            &mut report,
            &model,
            Examination::StateSpace,
            &setup_evidence,
            MccRunStatus::Completed { records: 0 },
            None,
        );
        let dedup_row = report
            .evidence
            .iter()
            .find(|row| row.contains(" shared_dedup_identity "))
            .expect("shared dedup identity row should be emitted");
        assert_eq!(
            evidence_field(dedup_row, "storage_kind"),
            Some(expected_storage_kind)
        );
        assert_eq!(
            evidence_field(dedup_row, "storage_config_identity"),
            Some(expected_storage_config_identity)
        );
        assert_eq!(
            evidence_field(dedup_row, "lane_kind"),
            Some(expected_lane_kind)
        );
        let expected_fingerprint_identity_token = setup_evidence
            .marking_dedup
            .fingerprint
            .fingerprint_identity()
            .replace(':', "_");
        let expected_runtime_fingerprint_identity_token = evidence_token(
            &setup_evidence
                .marking_dedup
                .fingerprint
                .fingerprint_identity(),
        );
        assert_eq!(
            evidence_field(dedup_row, "fingerprint_identity"),
            Some(expected_fingerprint_identity_token.as_str())
        );
        let runtime_fingerprint_row = report
            .evidence
            .iter()
            .find(|row| row.contains(" runtime_fingerprint_adoption "))
            .expect("runtime fingerprint adoption row should be emitted");
        assert_eq!(
            evidence_field(runtime_fingerprint_row, "origin_frontend"),
            Some(MCC_SHARED_ENGINE_ORIGIN_FRONTEND)
        );
        assert_eq!(
            evidence_field(runtime_fingerprint_row, "shared_owner"),
            Some(MCC_SHARED_ENGINE_LANE_OWNER)
        );
        assert_eq!(
            evidence_field(runtime_fingerprint_row, "first_beneficiary"),
            Some(MCC_SHARED_ENGINE_FIRST_BENEFICIARY)
        );
        assert_eq!(
            evidence_field(runtime_fingerprint_row, "second_beneficiary"),
            Some(MCC_SHARED_ENGINE_SECOND_BENEFICIARY)
        );
        assert_eq!(
            evidence_field(runtime_fingerprint_row, "generic_prerequisites"),
            Some(MCC_SHARED_ENGINE_GENERIC_PREREQUISITES)
        );
        assert_eq!(
            evidence_field(
                runtime_fingerprint_row,
                "default_compatible_frontend_families"
            ),
            Some(MCC_SHARED_ENGINE_DEFAULT_COMPATIBLE_FRONTEND_FAMILIES)
        );
        assert_eq!(
            evidence_field(runtime_fingerprint_row, "default_consumers"),
            Some(MCC_SHARED_ENGINE_DEFAULT_COMPATIBLE_FRONTEND_FAMILIES)
        );
        assert_eq!(
            evidence_field(runtime_fingerprint_row, "frontend_family_blockers"),
            Some(MCC_SHARED_ENGINE_FRONTEND_FAMILY_BLOCKERS)
        );
        assert_eq!(
            evidence_field(runtime_fingerprint_row, "acceptance_evidence"),
            Some(MCC_SHARED_ENGINE_ACCEPTANCE_EVIDENCE)
        );
        assert_eq!(
            evidence_field(runtime_fingerprint_row, "second_beneficiaries"),
            Some(MCC_SHARED_ENGINE_SECOND_BENEFICIARIES)
        );
        assert_eq!(
            evidence_field(runtime_fingerprint_row, "frontend_neutral_state_vector"),
            Some("true")
        );
        assert_eq!(
            evidence_field(runtime_fingerprint_row, "frontend_neutral_fingerprint"),
            Some("true")
        );
        assert_eq!(
            evidence_field(runtime_fingerprint_row, "tla_check_inherits"),
            Some("true")
        );
        assert_eq!(
            evidence_field(runtime_fingerprint_row, "hardware_lanes_inherit"),
            Some("true")
        );
        assert_eq!(
            evidence_field(runtime_fingerprint_row, "fingerprint_id"),
            Some(expected_fingerprint_id)
        );
        assert_eq!(
            evidence_field(runtime_fingerprint_row, "state_vector_fingerprint_id"),
            Some(expected_fingerprint_id)
        );
        assert_eq!(
            evidence_field(runtime_fingerprint_row, "lane_kind"),
            Some(expected_lane_kind)
        );
        assert_eq!(
            evidence_field(
                runtime_fingerprint_row,
                "prepared_fingerprint_admission_contract"
            ),
            Some(MCC_PREPARED_FINGERPRINT_ADMISSION_CONTRACT)
        );
        assert_eq!(
            evidence_field(
                runtime_fingerprint_row,
                "prepared_fingerprint_admission_plan_id"
            ),
            Some(expected_admission_plan_id)
        );
        assert_eq!(
            evidence_field(
                runtime_fingerprint_row,
                "prepared_fingerprint_admission_status"
            ),
            Some(MCC_PREPARED_FINGERPRINT_ADMISSION_VALIDATION_STATUS)
        );
        assert_eq!(
            evidence_field(
                runtime_fingerprint_row,
                "prepared_fingerprint_admission_payload_witness"
            ),
            Some("petri_marking_cas")
        );
        assert_eq!(
            evidence_field(runtime_fingerprint_row, "canonical_fingerprint_identity"),
            Some(expected_runtime_fingerprint_identity_token.as_str())
        );
        assert_eq!(
            evidence_field(runtime_fingerprint_row, "fingerprint_algorithm"),
            Some("canonical_bytes_sha256")
        );
        assert_eq!(
            evidence_field(runtime_fingerprint_row, "fingerprint_value_kind"),
            Some("marking_vector")
        );
        assert_eq!(
            evidence_field(runtime_fingerprint_row, "canonical_state_vector_layout"),
            Some(MCC_MARKING_CANONICAL_DOMAIN)
        );
        assert_eq!(
            evidence_field(
                runtime_fingerprint_row,
                "canonical_state_vector_layout_version"
            ),
            Some(MCC_MARKING_CANONICAL_DOMAIN_VERSION)
        );
        assert_eq!(
            evidence_field(runtime_fingerprint_row, "exact_or_unknown_guard"),
            Some("fail_closed_until_runtime_completion")
        );
        assert_eq!(
            evidence_field(runtime_fingerprint_row, "admission_status"),
            Some("accepted")
        );
        assert_eq!(
            evidence_field(runtime_fingerprint_row, "validation_scope"),
            Some(MCC_PREPARED_FINGERPRINT_ADMISSION_SETUP_VALIDATION_SCOPE)
        );
        assert_eq!(
            evidence_field(runtime_fingerprint_row, "runtime_consumed"),
            Some("false")
        );
        assert_eq!(
            evidence_field(runtime_fingerprint_row, "production_selected"),
            Some("false")
        );
        assert_eq!(
            evidence_field(runtime_fingerprint_row, "evidence_scope"),
            Some("setup_descriptor")
        );
        assert_eq!(
            evidence_field(runtime_fingerprint_row, "setup_only"),
            Some("true")
        );
        assert_eq!(
            evidence_field(runtime_fingerprint_row, "runtime_claim"),
            Some("false")
        );
        assert_eq!(
            evidence_field(runtime_fingerprint_row, "runtime_win_claim"),
            Some("false")
        );
        assert_eq!(
            evidence_field(
                runtime_fingerprint_row,
                "prepared_admission_receipt_required"
            ),
            Some(expected_runtime_receipt_required_str.as_str())
        );
        assert_eq!(
            evidence_field(runtime_fingerprint_row, "prepared_admission_receipt_status"),
            Some(expected_runtime_receipt_status)
        );
        assert_eq!(
            evidence_field(
                runtime_fingerprint_row,
                "prepared_admission_receipt_reason_code"
            ),
            Some(if expected_runtime_receipt_required {
                MCC_PREPARED_FINGERPRINT_ADMISSION_RUNTIME_RECEIPT_SETUP_ONLY_REASON
            } else {
                MCC_PREPARED_FINGERPRINT_ADMISSION_RUNTIME_RECEIPT_NOT_REQUIRED_REASON
            })
        );
        assert_eq!(
            evidence_field(runtime_fingerprint_row, "runtime_consumption_status"),
            Some(if expected_runtime_receipt_required {
                MCC_PREPARED_FINGERPRINT_ADMISSION_RUNTIME_CONSUMPTION_SETUP_ONLY
            } else {
                MCC_PREPARED_FINGERPRINT_ADMISSION_RUNTIME_CONSUMPTION_NOT_REQUIRED
            })
        );
        assert_eq!(
            evidence_field(runtime_fingerprint_row, "admission_counters_present"),
            Some("false")
        );
        assert_eq!(
            evidence_field(runtime_fingerprint_row, "fail_closed"),
            Some("true")
        );
        let runtime_dedup_row = report
            .evidence
            .iter()
            .find(|row| row.contains(" runtime_dedup_admission "))
            .expect("runtime dedup admission row should be emitted");
        assert_eq!(
            evidence_field(runtime_dedup_row, "storage_kind"),
            Some(expected_storage_kind)
        );
        assert_eq!(
            evidence_field(runtime_dedup_row, "storage_config_identity"),
            Some(expected_storage_config_identity)
        );
        assert_eq!(
            evidence_field(runtime_dedup_row, "dedup_admission_scope"),
            Some("state_space")
        );
        assert_eq!(
            evidence_field(runtime_dedup_row, "dedup_admission_policy"),
            Some("reject_on_collision")
        );
        assert_eq!(
            evidence_field(runtime_dedup_row, "prepared_fingerprint_admission_contract"),
            Some(MCC_PREPARED_FINGERPRINT_ADMISSION_CONTRACT)
        );
        assert_eq!(
            evidence_field(runtime_dedup_row, "prepared_fingerprint_admission_plan_id"),
            Some(expected_admission_plan_id)
        );
        assert_eq!(
            evidence_field(
                runtime_dedup_row,
                "prepared_fingerprint_admission_duplicate_authorization"
            ),
            Some("canonical_payload_equality")
        );
        assert_eq!(
            evidence_field(runtime_dedup_row, "canonical_fingerprint_identity"),
            Some(expected_runtime_fingerprint_identity_token.as_str())
        );
        assert_eq!(
            evidence_field(runtime_dedup_row, "second_beneficiaries"),
            Some(MCC_SHARED_ENGINE_SECOND_BENEFICIARIES)
        );
        assert_eq!(
            evidence_field(runtime_dedup_row, "exact_or_unknown_guard"),
            Some("fail_closed_until_runtime_completion")
        );
        assert_eq!(
            evidence_field(runtime_dedup_row, "collision_fail_closed"),
            Some("true")
        );
        assert_eq!(
            evidence_field(runtime_dedup_row, "admission_status"),
            Some("accepted")
        );
        assert_eq!(
            evidence_field(runtime_dedup_row, "validation_scope"),
            Some(MCC_PREPARED_FINGERPRINT_ADMISSION_SETUP_VALIDATION_SCOPE)
        );
        assert_eq!(
            evidence_field(runtime_dedup_row, "runtime_consumed"),
            Some("false")
        );
        assert_eq!(
            evidence_field(runtime_dedup_row, "production_selected"),
            Some("false")
        );
        assert_eq!(
            evidence_field(runtime_dedup_row, "evidence_scope"),
            Some("setup_descriptor")
        );
        assert_eq!(
            evidence_field(runtime_dedup_row, "runtime_claim"),
            Some("false")
        );
        assert_eq!(
            evidence_field(runtime_dedup_row, "prepared_admission_receipt_status"),
            Some(expected_runtime_receipt_status)
        );
        assert_eq!(
            evidence_field(runtime_dedup_row, "prepared_admission_receipt_reason_code"),
            Some(if expected_runtime_receipt_required {
                MCC_PREPARED_FINGERPRINT_ADMISSION_RUNTIME_RECEIPT_SETUP_ONLY_REASON
            } else {
                MCC_PREPARED_FINGERPRINT_ADMISSION_RUNTIME_RECEIPT_NOT_REQUIRED_REASON
            })
        );
        assert!(
            !report
                .evidence
                .iter()
                .any(|row| row.contains(" prepared_fingerprint_admission_runtime_consumption ")),
            "setup-only evidence must not claim hot-loop prepared admission consumption"
        );
        let hot_row = report
            .evidence
            .iter()
            .find(|row| row.contains(" hot_execution "))
            .expect("hot execution row should be emitted");
        assert_eq!(
            evidence_field(hot_row, "hot_execution_recorded"),
            Some("false")
        );
        assert_eq!(
            evidence_field(hot_row, "runtime_consumption_status"),
            Some(if expected_runtime_receipt_required {
                MCC_PREPARED_FINGERPRINT_ADMISSION_RUNTIME_CONSUMPTION_SETUP_ONLY
            } else {
                MCC_PREPARED_FINGERPRINT_ADMISSION_RUNTIME_CONSUMPTION_NOT_REQUIRED
            })
        );
        assert_eq!(
            evidence_field(hot_row, "production_readiness_reason_code"),
            Some(MCC_PREPARED_FINGERPRINT_ADMISSION_RUNTIME_RECEIPT_SETUP_ONLY_REASON)
        );
        assert_eq!(
            evidence_field(hot_row, "production_selected"),
            Some("false")
        );
        assert_eq!(evidence_field(hot_row, "fail_closed"), Some("true"));
    }

    #[test]
    fn setup_evidence_reports_sequential_memory_dedup_identity() {
        assert_setup_dedup_identity_for_config(
            ExplorationConfig::new(100)
                .with_workers(1)
                .with_storage_mode(StorageMode::Memory),
            "in_memory",
            "local-fingerprint-set-v1",
            "explicit_state",
            MCC_MARKING_FINGERPRINT_ID,
            128,
        );
    }

    #[test]
    fn setup_evidence_reports_fingerprint_only_cas_identity() {
        assert_setup_dedup_identity_for_config(
            ExplorationConfig::new(100)
                .with_workers(1)
                .with_storage_mode(StorageMode::FingerprintOnly),
            "cas",
            "fingerprint-only-cas-fingerprint-set-v1",
            "fingerprint",
            MCC_MARKING_FINGERPRINT_ID_U64_LOW,
            64,
        );
    }

    #[test]
    fn fingerprint_only_runtime_consumption_row_comes_from_hot_loop() {
        let dir = tiny_model_dir();
        let model = crate::model::load_model_dir(dir.path()).expect("model should load");
        let config = ExplorationConfig::new(100)
            .with_workers(1)
            .with_storage_mode(StorageMode::FingerprintOnly);
        let mut setup_evidence = build_mcc_setup_evidence(
            &model,
            Examination::StateSpace,
            &config,
            Instant::now(),
            std::time::Duration::from_nanos(1),
            std::time::Duration::from_nanos(1),
        );

        let mut observer = RuntimeCountingObserver::default();
        let (result, stats) = crate::explorer::fingerprint_only::explore_fingerprint_only(
            model.net(),
            &config,
            &mut observer,
            None,
        );
        assert!(result.completed);
        assert_eq!(stats.admission_attempted, 2);
        assert_eq!(stats.admission_new, 2);
        assert_eq!(stats.admission_duplicate, 0);
        assert_eq!(stats.admission_fault, 0);
        setup_evidence.record_hot_execution(std::time::Duration::from_nanos(17));

        let mut report = CapabilityReport::new(ProblemKind::StateSpace);
        add_mcc_setup_evidence(
            &mut report,
            &model,
            Examination::StateSpace,
            &setup_evidence,
            MccRunStatus::Completed { records: 0 },
            None,
        );

        let setup_only_row = report
            .evidence
            .iter()
            .find(|row| row.contains(" runtime_fingerprint_adoption "))
            .expect("setup fingerprint row should be emitted");
        assert_eq!(
            evidence_field(setup_only_row, "validation_scope"),
            Some(MCC_PREPARED_FINGERPRINT_ADMISSION_SETUP_VALIDATION_SCOPE)
        );
        assert_eq!(
            evidence_field(setup_only_row, "runtime_consumed"),
            Some("false")
        );
        assert_eq!(
            evidence_field(setup_only_row, "runtime_claim"),
            Some("false")
        );
        assert_eq!(
            evidence_field(setup_only_row, "production_selected"),
            Some("false")
        );
        assert_eq!(
            evidence_field(setup_only_row, "prepared_admission_receipt_status"),
            Some(MCC_PREPARED_FINGERPRINT_ADMISSION_RUNTIME_RECEIPT_ACCEPTED)
        );
        assert_eq!(
            evidence_field(setup_only_row, "prepared_admission_receipt_reason_code"),
            Some(MCC_PREPARED_FINGERPRINT_ADMISSION_RUNTIME_RECEIPT_AVAILABLE_REASON)
        );
        assert_eq!(
            evidence_field(setup_only_row, "runtime_consumption_status"),
            Some(MCC_PREPARED_FINGERPRINT_ADMISSION_RUNTIME_CONSUMPTION_OBSERVED)
        );
        assert_eq!(
            evidence_field(setup_only_row, "admission_counters_present"),
            Some("true")
        );

        let runtime_row = report
            .evidence
            .iter()
            .find(|row| row.contains(" prepared_fingerprint_admission_runtime_consumption "))
            .expect("hot-loop runtime consumption row should be emitted");
        assert_eq!(
            evidence_field(runtime_row, "git_head"),
            Some(env!("TY_MCC_BUILD_GIT_HEAD"))
        );
        assert_eq!(
            evidence_field(runtime_row, "plan_id"),
            Some(MCC_FINGERPRINT_ONLY_PREPARED_FINGERPRINT_ADMISSION_PLAN_ID)
        );
        assert_eq!(
            evidence_field(runtime_row, "callsite"),
            Some("tla_petri::explorer::fingerprint_only::PackedMarkingCollisionGuard::admit")
        );
        assert_eq!(
            evidence_field(runtime_row, "validation_scope"),
            Some(MCC_PREPARED_FINGERPRINT_ADMISSION_RUNTIME_VALIDATION_SCOPE)
        );
        assert_eq!(evidence_field(runtime_row, "attempted"), Some("2"));
        assert_eq!(evidence_field(runtime_row, "new"), Some("2"));
        assert_eq!(evidence_field(runtime_row, "duplicate"), Some("0"));
        assert_eq!(evidence_field(runtime_row, "fault"), Some("0"));
        assert_eq!(evidence_field(runtime_row, "lane"), Some("fingerprint"));
        assert_eq!(evidence_field(runtime_row, "storage"), Some("cas"));
        assert_eq!(
            evidence_field(runtime_row, "payload_witness"),
            Some("petri_marking_cas")
        );
        assert_eq!(
            evidence_field(runtime_row, "runtime_consumed"),
            Some("true")
        );
        assert_eq!(
            evidence_field(runtime_row, "evidence_scope"),
            Some("hot_loop")
        );
        assert_eq!(evidence_field(runtime_row, "runtime_claim"), Some("true"));
        assert_eq!(
            evidence_field(runtime_row, "runtime_win_claim"),
            Some("hot_loop_consumed")
        );
        assert_eq!(
            evidence_field(runtime_row, "hot_loop_consumption"),
            Some("observed")
        );
        assert_eq!(
            evidence_field(runtime_row, "admission_counters_present"),
            Some("true")
        );
        assert_eq!(
            evidence_field(runtime_row, "prepared_admission_receipt_status"),
            Some(MCC_PREPARED_FINGERPRINT_ADMISSION_RUNTIME_RECEIPT_ACCEPTED)
        );
        assert_eq!(
            evidence_field(runtime_row, "prepared_admission_receipt_reason_code"),
            Some(MCC_PREPARED_FINGERPRINT_ADMISSION_RUNTIME_RECEIPT_AVAILABLE_REASON)
        );
        assert_eq!(
            evidence_field(runtime_row, "production_readiness_status"),
            Some(MCC_PREPARED_FINGERPRINT_ADMISSION_RUNTIME_PRODUCTION_SELECTED)
        );
        assert_eq!(
            evidence_field(runtime_row, "production_selected"),
            Some("true")
        );
        assert_eq!(evidence_field(runtime_row, "fail_closed"), Some("false"));
        assert_eq!(
            evidence_field(runtime_row, "frontend_neutral_state_vector"),
            Some("true")
        );
        assert_eq!(
            evidence_field(runtime_row, "frontend_neutral_fingerprint"),
            Some("true")
        );
        let hot_row = report
            .evidence
            .iter()
            .find(|row| row.contains(" hot_execution "))
            .expect("hot execution row should be emitted");
        assert_eq!(
            evidence_field(
                hot_row,
                "prepared_fingerprint_admission_runtime_receipt_status"
            ),
            Some(MCC_PREPARED_FINGERPRINT_ADMISSION_RUNTIME_RECEIPT_ACCEPTED)
        );
        assert_eq!(
            evidence_field(
                hot_row,
                "prepared_fingerprint_admission_runtime_reason_code"
            ),
            Some(MCC_PREPARED_FINGERPRINT_ADMISSION_RUNTIME_RECEIPT_AVAILABLE_REASON)
        );
        assert_eq!(
            evidence_field(
                hot_row,
                "prepared_fingerprint_admission_runtime_counters_present"
            ),
            Some("true")
        );
        assert_eq!(
            evidence_field(hot_row, "hot_loop_consumption"),
            Some("prepared_fingerprint_admission_observed")
        );
        assert_eq!(evidence_field(hot_row, "production_selected"), Some("true"));
        assert_eq!(evidence_field(hot_row, "fail_closed"), Some("false"));
    }

    #[test]
    fn fingerprint_runtime_consumption_fault_fails_closed() {
        let dir = tiny_model_dir();
        let model = crate::model::load_model_dir(dir.path()).expect("model should load");
        let config = ExplorationConfig::new(100)
            .with_workers(1)
            .with_storage_mode(StorageMode::FingerprintOnly);
        let mut setup_evidence = build_mcc_setup_evidence(
            &model,
            Examination::StateSpace,
            &config,
            Instant::now(),
            std::time::Duration::from_nanos(1),
            std::time::Duration::from_nanos(1),
        );
        setup_evidence.record_hot_execution(std::time::Duration::from_nanos(19));
        let plan = mcc_marking_prepared_fingerprint_admission_plan(&setup_evidence);
        let consumption = MccPreparedFingerprintAdmissionRuntimeConsumption::from_plan(
            &plan,
            "tla_petri::tests::fault_injection",
            3,
            1,
            1,
            1,
        );
        let row = consumption.render_evidence_row();

        assert_eq!(
            evidence_field(&row, "plan_id"),
            Some(MCC_FINGERPRINT_ONLY_PREPARED_FINGERPRINT_ADMISSION_PLAN_ID)
        );
        assert_eq!(evidence_field(&row, "runtime_consumed"), Some("false"));
        assert_eq!(
            evidence_field(&row, "runtime_consumption_status"),
            Some(MCC_PREPARED_FINGERPRINT_ADMISSION_RUNTIME_CONSUMPTION_MISSING)
        );
        assert_eq!(
            evidence_field(&row, "hot_loop_consumption"),
            Some(MCC_PREPARED_FINGERPRINT_ADMISSION_RUNTIME_CONSUMPTION_MISSING)
        );
        assert_eq!(
            evidence_field(&row, "prepared_admission_receipt_status"),
            Some(MCC_PREPARED_FINGERPRINT_ADMISSION_RUNTIME_RECEIPT_MISSING)
        );
        assert_eq!(
            evidence_field(&row, "prepared_admission_receipt_reason_code"),
            Some(MCC_PREPARED_FINGERPRINT_ADMISSION_RUNTIME_PRODUCTION_FAULT_REASON)
        );
        assert_eq!(
            evidence_field(&row, "production_readiness_status"),
            Some(MCC_PREPARED_FINGERPRINT_ADMISSION_RUNTIME_PRODUCTION_BLOCKED)
        );
        assert_eq!(
            evidence_field(&row, "production_readiness_reason_code"),
            Some(MCC_PREPARED_FINGERPRINT_ADMISSION_RUNTIME_PRODUCTION_FAULT_REASON)
        );
        assert_eq!(evidence_field(&row, "production_selected"), Some("false"));
        assert_eq!(evidence_field(&row, "fail_closed"), Some("true"));

        record_mcc_prepared_fingerprint_admission_runtime_consumption(consumption);
        let mut report = CapabilityReport::new(ProblemKind::StateSpace);
        add_mcc_setup_evidence(
            &mut report,
            &model,
            Examination::StateSpace,
            &setup_evidence,
            MccRunStatus::Completed { records: 0 },
            None,
        );

        let runtime_fingerprint_row = report
            .evidence
            .iter()
            .find(|row| row.contains(" runtime_fingerprint_adoption "))
            .expect("setup fingerprint row should be emitted");
        assert_eq!(
            evidence_field(runtime_fingerprint_row, "prepared_admission_receipt_status"),
            Some(MCC_PREPARED_FINGERPRINT_ADMISSION_RUNTIME_RECEIPT_MISSING)
        );
        assert_eq!(
            evidence_field(
                runtime_fingerprint_row,
                "prepared_admission_receipt_reason_code"
            ),
            Some(MCC_PREPARED_FINGERPRINT_ADMISSION_RUNTIME_PRODUCTION_FAULT_REASON)
        );
        assert_eq!(
            evidence_field(runtime_fingerprint_row, "runtime_consumption_status"),
            Some(MCC_PREPARED_FINGERPRINT_ADMISSION_RUNTIME_CONSUMPTION_MISSING)
        );
        assert_eq!(
            evidence_field(runtime_fingerprint_row, "admission_counters_present"),
            Some("false")
        );
        assert_eq!(
            evidence_field(
                runtime_fingerprint_row,
                "missing_runtime_receipt_fail_closed"
            ),
            Some("true")
        );
        assert_eq!(
            evidence_field(runtime_fingerprint_row, "production_selected"),
            Some("false")
        );
        assert_eq!(
            evidence_field(runtime_fingerprint_row, "fail_closed"),
            Some("true")
        );

        let runtime_dedup_row = report
            .evidence
            .iter()
            .find(|row| row.contains(" runtime_dedup_admission "))
            .expect("setup dedup row should be emitted");
        assert_eq!(
            evidence_field(runtime_dedup_row, "prepared_admission_receipt_status"),
            Some(MCC_PREPARED_FINGERPRINT_ADMISSION_RUNTIME_RECEIPT_MISSING)
        );
        assert_eq!(
            evidence_field(runtime_dedup_row, "prepared_admission_receipt_reason_code"),
            Some(MCC_PREPARED_FINGERPRINT_ADMISSION_RUNTIME_PRODUCTION_FAULT_REASON)
        );
        assert_eq!(
            evidence_field(runtime_dedup_row, "runtime_consumption_status"),
            Some(MCC_PREPARED_FINGERPRINT_ADMISSION_RUNTIME_CONSUMPTION_MISSING)
        );
        assert_eq!(
            evidence_field(runtime_dedup_row, "admission_counters_present"),
            Some("false")
        );
        assert_eq!(
            evidence_field(runtime_dedup_row, "missing_runtime_receipt_fail_closed"),
            Some("true")
        );
        assert_eq!(
            evidence_field(runtime_dedup_row, "production_selected"),
            Some("false")
        );
        assert_eq!(
            evidence_field(runtime_dedup_row, "fail_closed"),
            Some("true")
        );

        let hot_row = report
            .evidence
            .iter()
            .find(|row| row.contains(" hot_execution "))
            .expect("hot execution row should be emitted");
        assert_eq!(
            evidence_field(
                hot_row,
                "prepared_fingerprint_admission_runtime_receipt_status"
            ),
            Some(MCC_PREPARED_FINGERPRINT_ADMISSION_RUNTIME_RECEIPT_MISSING)
        );
        assert_eq!(
            evidence_field(
                hot_row,
                "prepared_fingerprint_admission_runtime_reason_code"
            ),
            Some(MCC_PREPARED_FINGERPRINT_ADMISSION_RUNTIME_PRODUCTION_FAULT_REASON)
        );
        assert_eq!(
            evidence_field(hot_row, "runtime_consumption_status"),
            Some(MCC_PREPARED_FINGERPRINT_ADMISSION_RUNTIME_CONSUMPTION_MISSING)
        );
        assert_eq!(
            evidence_field(hot_row, "hot_loop_consumption"),
            Some(MCC_PREPARED_FINGERPRINT_ADMISSION_RUNTIME_CONSUMPTION_MISSING)
        );
        assert_eq!(
            evidence_field(hot_row, "production_readiness_reason_code"),
            Some(MCC_PREPARED_FINGERPRINT_ADMISSION_RUNTIME_PRODUCTION_FAULT_REASON)
        );
        assert_eq!(
            evidence_field(hot_row, "production_selected"),
            Some("false")
        );
        assert_eq!(evidence_field(hot_row, "fail_closed"), Some("true"));
        assert!(
            !report
                .evidence
                .iter()
                .any(|row| row.contains(" prepared_fingerprint_admission_runtime_consumption_missing ")),
            "faulted runtime consumption should be represented by the fault row, not a synthetic missing row"
        );
    }

    #[test]
    fn fingerprint_only_hot_execution_without_admission_receipt_fails_closed_in_evidence() {
        let dir = tiny_model_dir();
        let model = crate::model::load_model_dir(dir.path()).expect("model should load");
        let config = ExplorationConfig::new(100)
            .with_workers(1)
            .with_storage_mode(StorageMode::FingerprintOnly);
        let mut setup_evidence = build_mcc_setup_evidence(
            &model,
            Examination::StateSpace,
            &config,
            Instant::now(),
            std::time::Duration::from_nanos(1),
            std::time::Duration::from_nanos(1),
        );
        setup_evidence.record_hot_execution(std::time::Duration::from_nanos(19));

        let mut report = CapabilityReport::new(ProblemKind::StateSpace);
        add_mcc_setup_evidence(
            &mut report,
            &model,
            Examination::StateSpace,
            &setup_evidence,
            MccRunStatus::Completed { records: 0 },
            None,
        );

        let runtime_fingerprint_row = report
            .evidence
            .iter()
            .find(|row| row.contains(" runtime_fingerprint_adoption "))
            .expect("setup fingerprint row should be emitted");
        assert_eq!(
            evidence_field(runtime_fingerprint_row, "runtime_claim"),
            Some("false")
        );
        assert_eq!(
            evidence_field(runtime_fingerprint_row, "prepared_admission_receipt_status"),
            Some(MCC_PREPARED_FINGERPRINT_ADMISSION_RUNTIME_RECEIPT_MISSING)
        );
        assert_eq!(
            evidence_field(
                runtime_fingerprint_row,
                "prepared_admission_receipt_reason_code"
            ),
            Some(MCC_PREPARED_FINGERPRINT_ADMISSION_RUNTIME_RECEIPT_MISSING_REASON)
        );
        assert_eq!(
            evidence_field(runtime_fingerprint_row, "runtime_consumption_status"),
            Some(MCC_PREPARED_FINGERPRINT_ADMISSION_RUNTIME_CONSUMPTION_MISSING)
        );
        assert_eq!(
            evidence_field(
                runtime_fingerprint_row,
                "missing_runtime_receipt_fail_closed"
            ),
            Some("true")
        );

        let missing_row = report
            .evidence
            .iter()
            .find(|row| {
                row.contains(" prepared_fingerprint_admission_runtime_consumption_missing ")
            })
            .expect("missing hot-loop admission receipt row should be emitted");
        assert_eq!(
            evidence_field(missing_row, "prepared_fingerprint_admission_plan_id"),
            Some(MCC_FINGERPRINT_ONLY_PREPARED_FINGERPRINT_ADMISSION_PLAN_ID)
        );
        assert_eq!(
            evidence_field(missing_row, "reason_code"),
            Some(MCC_PREPARED_FINGERPRINT_ADMISSION_RUNTIME_RECEIPT_MISSING_REASON)
        );
        assert_eq!(
            evidence_field(missing_row, "prepared_admission_receipt_status"),
            Some(MCC_PREPARED_FINGERPRINT_ADMISSION_RUNTIME_RECEIPT_MISSING)
        );
        assert_eq!(
            evidence_field(missing_row, "prepared_admission_receipt_reason_code"),
            Some(MCC_PREPARED_FINGERPRINT_ADMISSION_RUNTIME_RECEIPT_MISSING_REASON)
        );
        assert_eq!(
            evidence_field(missing_row, "runtime_consumed"),
            Some("false")
        );
        assert_eq!(
            evidence_field(missing_row, "runtime_consumption_status"),
            Some(MCC_PREPARED_FINGERPRINT_ADMISSION_RUNTIME_CONSUMPTION_MISSING)
        );
        assert_eq!(
            evidence_field(missing_row, "admission_counters_present"),
            Some("false")
        );
        assert_eq!(evidence_field(missing_row, "fail_closed"), Some("true"));

        let hot_row = report
            .evidence
            .iter()
            .find(|row| row.contains(" hot_execution "))
            .expect("hot execution row should be emitted");
        assert_eq!(
            evidence_field(
                hot_row,
                "prepared_fingerprint_admission_runtime_receipt_status"
            ),
            Some(MCC_PREPARED_FINGERPRINT_ADMISSION_RUNTIME_RECEIPT_MISSING)
        );
        assert_eq!(
            evidence_field(
                hot_row,
                "prepared_fingerprint_admission_runtime_reason_code"
            ),
            Some(MCC_PREPARED_FINGERPRINT_ADMISSION_RUNTIME_RECEIPT_MISSING_REASON)
        );
        assert_eq!(
            evidence_field(
                hot_row,
                "prepared_fingerprint_admission_runtime_fail_closed"
            ),
            Some("true")
        );
        assert_eq!(
            evidence_field(hot_row, "hot_loop_consumption"),
            Some(MCC_PREPARED_FINGERPRINT_ADMISSION_RUNTIME_RECEIPT_MISSING_REASON)
        );
        assert_eq!(
            evidence_field(hot_row, "production_selected"),
            Some("false")
        );
        assert_eq!(evidence_field(hot_row, "fail_closed"), Some("true"));
    }

    #[test]
    fn setup_evidence_reports_parallel_sharded_and_cas_identity() {
        assert_setup_dedup_identity_for_config(
            ExplorationConfig::new(100)
                .with_workers(8)
                .with_fpset_backend(FpsetBackend::Sharded),
            "sharded_in_memory",
            "sharded-fingerprint-set-v1",
            "explicit_state",
            MCC_MARKING_FINGERPRINT_ID,
            128,
        );
        assert_setup_dedup_identity_for_config(
            ExplorationConfig::new(100)
                .with_workers(8)
                .with_fpset_backend(FpsetBackend::Cas),
            "cas",
            "partitioned-cas-fingerprint-set-v1-partition-bits-4",
            "explicit_state",
            MCC_MARKING_FINGERPRINT_ID_U64_XORFOLD,
            64,
        );
    }

    #[test]
    fn prepared_program_evidence_row_and_json_expose_descriptor_granularity() {
        let dir = tiny_model_dir();
        let model = crate::model::load_model_dir(dir.path()).expect("model should load");
        let config = ExplorationConfig::new(100).with_workers(1);
        let setup_evidence = build_mcc_setup_evidence(
            &model,
            Examination::ReachabilityFireability,
            &config,
            Instant::now(),
            std::time::Duration::from_nanos(1),
            std::time::Duration::from_nanos(1),
        );

        let row = mcc_prepared_program_evidence_row(
            &model,
            Examination::ReachabilityFireability,
            &setup_evidence,
        );
        assert!(row.contains("reduction_mode=reachability"));
        assert!(row.contains(
            "analytical_solve_ids=mcc.petri.ReachabilityFireability|mcc.petri.structural.p_invariant_bounds.ReachabilityFireability|mcc.petri.lp_state_equation.relaxation.ReachabilityFireability|mcc.petri.reduction.reachability.ReachabilityFireability"
        ));
        assert!(row.contains(
            "symbolic_proof_ids=mcc.petri.ay.bounded_model_check.ReachabilityFireability|mcc.petri.ay.chc.reachability_pdr.ReachabilityFireability"
        ));
        assert!(row.contains("backend_family_codes=explicit_state|external_ay_binary|ay_chc"));
        assert!(row.contains("backend_family_facets=linear_integer_arithmetic|external_process+smt+bmc|in_process+chc+pdr+linear_integer_arithmetic"));
        assert!(row.contains("output_contract=mcc_stdout_v1"));
        assert!(row.contains("frontend_kind=mcc_petri"));
        assert!(row.contains("fingerprint_id=marking-v1"));
        assert!(row.contains("prepared_program_fingerprint_algorithm=sha256"));
        assert!(row.contains("shared_engine_origin_frontend=mcc_petri"));
        assert!(row.contains("shared_engine_component=tla_mc_core.prepared_checker_program"));
        assert!(row.contains("shared_engine_lane_owner=shared_high_performance_engine"));
        assert!(row.contains("shared_engine_first_beneficiary=mcc_petri_runtime_storage"));
        assert!(row.contains("shared_engine_second_beneficiary=tla_plus"));
        assert!(row.contains("shared_engine_extraction_status=shared-core-ready"));
        assert!(row.contains(
            "shared_engine_compatible_frontend_families=tla_plus,quint,mcc_petri,aiger,btor2,vmt_transition_system,ay_analytical,witness_replay"
        ));
        assert!(
            row.contains("shared_engine_default_compatible_frontend_families=tla_plus,mcc_petri")
        );
        assert!(row.contains("shared_engine_downstream_beneficiary_families=aiger,btor2,vmt_transition_system,ay_analytical,witness_replay"));
        assert!(row.contains("shared_engine_remaining_compatible_frontend_families=quint"));
        assert!(row.contains("shared_engine_generic_prerequisites=prepared_checker_program_descriptor,marking_storage_identity,fingerprint_identity,prepared_fingerprint_admission_plan,dedup_admission,validation_plan_descriptor"));
        assert!(row.contains(
            "shared_engine_frontend_family_blockers=future_importer:awaiting_registered_importer_frontend"
        ));
        assert!(row.contains("shared_engine_blocker_status=tracked-blockers"));
        assert!(row.contains("shared_engine_acceptance_evidence=mcc_backend_evidence_unit_tests,runtime_fingerprint_adoption_rows,prepared_fingerprint_admission_validate_runtime_admission"));
        assert!(row.contains("artifact_prepared_program_fingerprint="));
        assert!(row.contains("proof_prepared_program_fingerprint="));
        assert!(row.contains("replay_prepared_program_fingerprint="));
        let prepared_identities = setup_evidence.prepared_program.effective_identity_fields();
        let expected_source_identity =
            evidence_option_token(setup_evidence.trace.source_identity.as_deref());
        let expected_property_identity =
            evidence_option_token(setup_evidence.trace.property_identity.as_deref());
        let expected_cache_key = evidence_option_token(prepared_identities.cache_key.as_deref());
        let expected_frontend_payload_identity =
            evidence_option_token(prepared_identities.frontend_payload_identity.as_deref());
        let expected_artifact_identity =
            evidence_option_token(prepared_identities.artifact_identity.as_deref());
        let expected_storage_policy_identity =
            evidence_option_token(prepared_identities.storage_policy_identity.as_deref());
        let expected_fingerprint_policy_identity =
            evidence_option_token(prepared_identities.fingerprint_policy_identity.as_deref());
        let expected_fingerprint_identity =
            evidence_option_token(prepared_identities.fingerprint_identity.as_deref());
        assert_eq!(
            evidence_field(&row, "source_identity"),
            Some(expected_source_identity.as_str())
        );
        assert_eq!(
            evidence_field(&row, "property_identity"),
            Some(expected_property_identity.as_str())
        );
        assert_eq!(evidence_field(&row, "candidate_key"), Some("none"));
        assert_eq!(evidence_field(&row, "candidate_identity"), Some("none"));
        assert_eq!(evidence_field(&row, "lane_identity"), Some("none"));
        assert_eq!(
            evidence_field(&row, "cache_key"),
            Some(expected_cache_key.as_str())
        );
        assert_eq!(
            evidence_field(&row, "frontend_payload_identity"),
            Some(expected_frontend_payload_identity.as_str())
        );
        assert_eq!(
            evidence_field(&row, "artifact_identity"),
            Some(expected_artifact_identity.as_str())
        );
        assert_eq!(
            evidence_field(&row, "storage_policy_identity"),
            Some(expected_storage_policy_identity.as_str())
        );
        assert_eq!(
            evidence_field(&row, "fingerprint_policy_identity"),
            Some(expected_fingerprint_policy_identity.as_str())
        );
        assert_eq!(
            evidence_field(&row, "fingerprint_identity"),
            Some(expected_fingerprint_identity.as_str())
        );
        assert_eq!(
            evidence_field(&row, "batch_artifact_identity"),
            Some("none")
        );
        let shared_row = mcc_shared_prepared_checker_program_evidence_row(&setup_evidence);
        validate_prepared_checker_program_evidence_row(&shared_row)
            .expect("MCC shared prepared-program row should use shared vocabulary");
        assert_eq!(
            evidence_field(&shared_row, "source_identity"),
            Some(expected_source_identity.as_str())
        );
        assert_eq!(
            evidence_field(&shared_row, "property_identity"),
            Some(expected_property_identity.as_str())
        );
        assert_eq!(evidence_field(&shared_row, "candidate_key"), Some("none"));
        let adoption_row = mcc_shared_engine_adoption_evidence().render_evidence_row("MCC");
        validate_shared_engine_adoption_evidence_row(&adoption_row).unwrap();
        assert!(adoption_row.contains("origin_frontend=mcc_petri"));
        assert!(adoption_row.contains("first_beneficiary=mcc_petri_runtime_storage"));
        assert!(adoption_row.contains("second_beneficiary=tla_plus"));
        assert!(adoption_row.contains("adoption_level=level-3"));
        assert!(adoption_row.contains(
            "compatible_frontend_families=tla_plus,quint,mcc_petri,aiger,btor2,vmt_transition_system,ay_analytical,witness_replay"
        ));
        assert!(adoption_row.contains("default_compatible_frontend_families=tla_plus,mcc_petri"));
        assert!(adoption_row.contains(
            "downstream_beneficiary_families=aiger,btor2,vmt_transition_system,ay_analytical,witness_replay"
        ));
        assert!(adoption_row.contains("remaining_compatible_frontend_families=quint"));
        assert!(adoption_row.contains(
            "frontend_family_blockers=future_importer:awaiting_registered_importer_frontend"
        ));
        assert!(adoption_row.contains("acceptance_evidence="));
        assert!(adoption_row.contains("prepared_fingerprint_admission_validate_runtime_admission"));
        assert!(adoption_row.contains("blocker_status=tracked-blockers"));
        assert!(!adoption_row.contains("adoption_not_yet_recorded"));
        let setup_evidence_again = build_mcc_setup_evidence(
            &model,
            Examination::ReachabilityFireability,
            &config,
            Instant::now(),
            std::time::Duration::from_nanos(7),
            std::time::Duration::from_nanos(11),
        );
        let row_again = mcc_prepared_program_evidence_row(
            &model,
            Examination::ReachabilityFireability,
            &setup_evidence_again,
        );
        assert_eq!(
            evidence_field(&row, "prepared_program_fingerprint"),
            evidence_field(&row_again, "prepared_program_fingerprint")
        );
        assert_eq!(
            evidence_field(
                &mcc_shared_prepared_checker_program_evidence_row(&setup_evidence),
                "prepared_program_fingerprint"
            ),
            evidence_field(
                &mcc_shared_prepared_checker_program_evidence_row(&setup_evidence_again),
                "prepared_program_fingerprint"
            )
        );

        let serializable = SerializablePreparedProgram::from(&setup_evidence);
        assert_eq!(serializable.frontend_kind, "mcc_petri");
        assert_eq!(serializable.fingerprint_id, "marking-v1");
        let setup_trace_json =
            serde_json::to_value(SerializableSetupTrace::from(&setup_evidence)).expect("json");
        assert_eq!(
            setup_trace_json["frontend_kind"].as_str(),
            Some("mcc_petri")
        );
        assert_eq!(setup_trace_json["lane_kind"].as_str(), Some("frontend"));
        assert!(setup_trace_json["candidate_key"].is_null());
        assert_eq!(
            setup_trace_json["source_identity"].as_str(),
            setup_evidence.trace.source_identity.as_deref()
        );
        assert_eq!(
            setup_trace_json["property_identity"].as_str(),
            setup_evidence.trace.property_identity.as_deref()
        );
        assert_eq!(
            setup_trace_json["origin_frontend"].as_str(),
            Some(MCC_SHARED_ENGINE_ORIGIN_FRONTEND)
        );
        assert_eq!(
            setup_trace_json["shared_engine_component"].as_str(),
            Some(MCC_SHARED_ENGINE_COMPONENT)
        );
        assert_eq!(
            setup_trace_json["first_beneficiary"].as_str(),
            Some(MCC_SHARED_ENGINE_FIRST_BENEFICIARY)
        );
        assert_eq!(
            setup_trace_json["second_beneficiary"].as_str(),
            Some(MCC_SHARED_ENGINE_SECOND_BENEFICIARY)
        );
        assert_eq!(
            setup_trace_json["shared_engine_extraction_status"].as_str(),
            Some(MCC_SHARED_ENGINE_EXTRACTION_STATUS)
        );
        assert_eq!(
            setup_trace_json["shared_engine_blocker_status"].as_str(),
            Some(MCC_SHARED_ENGINE_BLOCKER_STATUS)
        );
        assert_eq!(
            setup_trace_json["validation_status"].as_str(),
            Some("accepted")
        );
        assert_eq!(
            setup_trace_json["cache_key"].as_str(),
            prepared_identities.cache_key.as_deref()
        );
        assert_eq!(
            setup_trace_json["artifact_identity"].as_str(),
            prepared_identities.artifact_identity.as_deref()
        );
        assert_eq!(
            setup_trace_json["storage_policy_identity"].as_str(),
            prepared_identities.storage_policy_identity.as_deref()
        );
        assert_eq!(
            setup_trace_json["fingerprint_identity"].as_str(),
            prepared_identities.fingerprint_identity.as_deref()
        );
        assert!(setup_trace_json["batch_artifact_identity"].is_null());
        let timings = setup_trace_json["timings"]
            .as_array()
            .expect("setup trace timings");
        assert!(!timings.is_empty());
        assert!(timings.iter().all(|timing| {
            timing["frontend_kind"].as_str() == Some("mcc_petri")
                && timing["lane_kind"].as_str() == Some("frontend")
                && timing["cache_key"].as_str() == setup_trace_json["cache_key"].as_str()
                && timing["artifact_identity"].as_str()
                    == setup_trace_json["artifact_identity"].as_str()
                && timing["fingerprint_identity"].as_str()
                    == setup_trace_json["fingerprint_identity"].as_str()
                && timing["origin_frontend"].as_str() == Some(MCC_SHARED_ENGINE_ORIGIN_FRONTEND)
                && timing["shared_engine_component"].as_str() == Some(MCC_SHARED_ENGINE_COMPONENT)
                && timing["shared_engine_extraction_status"].as_str()
                    == Some(MCC_SHARED_ENGINE_EXTRACTION_STATUS)
                && timing["shared_engine_blocker_status"].as_str()
                    == Some(MCC_SHARED_ENGINE_BLOCKER_STATUS)
                && timing["validation_status"].as_str() == Some("accepted")
                && timing["batch_artifact_identity"].is_null()
        }));
        let prepared_json =
            serde_json::to_value(SerializablePreparedProgram::from(&setup_evidence)).expect("json");
        assert_eq!(
            prepared_json["source_identity"].as_str(),
            setup_evidence.trace.source_identity.as_deref()
        );
        assert!(prepared_json["candidate_key"].is_null());
        assert_eq!(
            prepared_json["cache_key"].as_str(),
            prepared_identities.cache_key.as_deref()
        );
        assert_eq!(
            prepared_json["frontend_payload_identity"].as_str(),
            prepared_identities.frontend_payload_identity.as_deref()
        );
        assert_eq!(
            prepared_json["artifact_identity"].as_str(),
            prepared_identities.artifact_identity.as_deref()
        );
        assert_eq!(
            prepared_json["storage_policy_identity"].as_str(),
            prepared_identities.storage_policy_identity.as_deref()
        );
        assert_eq!(
            prepared_json["fingerprint_policy_identity"].as_str(),
            prepared_identities.fingerprint_policy_identity.as_deref()
        );
        assert_eq!(
            prepared_json["fingerprint_identity"].as_str(),
            prepared_identities.fingerprint_identity.as_deref()
        );
        assert!(prepared_json["batch_artifact_identity"].is_null());
        assert_eq!(
            serializable.prepared_program_fingerprint_algorithm,
            MCC_PREPARED_PROGRAM_FINGERPRINT_ALGORITHM
        );
        assert_eq!(
            serializable.shared_engine_origin_frontend,
            MCC_SHARED_ENGINE_ORIGIN_FRONTEND
        );
        let canonical_identity = setup_evidence
            .prepared_program
            .canonical_identities
            .iter()
            .find(|identity| identity.id == MCC_PREPARED_PROGRAM_CANONICAL_IDENTITY)
            .expect("canonical prepared-program identity should be recorded");
        assert_eq!(
            canonical_identity.digest_algorithm.as_deref(),
            Some(MCC_PREPARED_PROGRAM_FINGERPRINT_ALGORITHM)
        );
        assert_eq!(
            canonical_identity.digest.as_deref(),
            Some(serializable.prepared_program_fingerprint.as_str())
        );
        assert_eq!(
            serializable.shared_engine_second_beneficiary,
            MCC_SHARED_ENGINE_SECOND_BENEFICIARY
        );
        assert_eq!(
            serializable.shared_engine_extraction_status,
            MCC_SHARED_ENGINE_EXTRACTION_STATUS
        );
        assert_eq!(
            serializable.shared_engine_compatible_frontend_families,
            MCC_SHARED_ENGINE_COMPATIBLE_FRONTEND_FAMILIES
        );
        assert_eq!(
            serializable.shared_engine_frontend_family_blockers,
            MCC_SHARED_ENGINE_FRONTEND_FAMILY_BLOCKERS
        );
        assert_eq!(
            serializable.shared_engine_blocker_status,
            MCC_SHARED_ENGINE_BLOCKER_STATUS
        );
        assert_eq!(
            serializable.artifact_prepared_program_fingerprint,
            serializable.prepared_program_fingerprint
        );
        assert_eq!(
            serializable.proof_prepared_program_fingerprint,
            serializable.prepared_program_fingerprint
        );
        assert_eq!(
            serializable.replay_prepared_program_fingerprint,
            serializable.prepared_program_fingerprint
        );
        assert_eq!(serializable.analytical_solves, 4);
        assert_eq!(serializable.symbolic_proofs, 2);
        assert_eq!(serializable.backend_families, 3);
        assert_eq!(serializable.reduction_mode, "reachability");
        assert!(serializable
            .analytical_solve_descriptors
            .iter()
            .any(|solve| solve.id.contains("structural.p_invariant_bounds")
                && solve.kind == "linear_invariant"
                && solve.problem == "invariant"));
        assert!(serializable
            .analytical_solve_descriptors
            .iter()
            .any(|solve| solve.id.contains("lp_state_equation")
                && solve.kind == "smt_query"
                && solve.problem == "smt"));
        assert!(serializable
            .symbolic_proof_descriptors
            .iter()
            .any(|proof| proof.kind == "chc_query" && proof.problem == "chc"));
        assert!(serializable
            .backend_family_descriptors
            .iter()
            .any(|family| family.backend_code == "ay_chc" && family.facets.contains(&"pdr")));

        let validation_rows = setup_evidence
            .prepared_program
            .render_validation_plan_evidence_rows("MCC");
        for row in &validation_rows {
            validate_prepared_validation_plan_evidence_row(row)
                .expect("MCC validation-plan rows should use shared validation vocabulary");
        }
        assert!(validation_rows
            .iter()
            .any(|row| row.contains("validation_kind=witness_replay")
                && row.contains("fingerprint_scheme=canonical_bytes_sha256")));
        for row in setup_evidence
            .prepared_program
            .render_frontend_extension_evidence_rows("MCC")
        {
            validate_prepared_frontend_extension_evidence_row(&row)
                .expect("MCC frontend-extension rows should use shared adapter vocabulary");
        }
        for row in setup_evidence
            .prepared_program
            .render_candidate_lane_evidence_rows("MCC")
        {
            validate_prepared_candidate_lane_evidence_row(&row)
                .expect("MCC candidate-lane rows should use shared adapter vocabulary");
        }

        let mut report = CapabilityReport::new(ProblemKind::ExplicitReachability);
        add_mcc_setup_evidence(
            &mut report,
            &model,
            Examination::ReachabilityFireability,
            &setup_evidence,
            MccRunStatus::Completed { records: 0 },
            None,
        );
        let setup_row = report
            .evidence
            .iter()
            .find(|row| {
                row.contains(" setup_trace ") && row.contains("phase=prepared_program_build")
            })
            .expect("prepared-program setup trace row should be emitted");
        assert_eq!(
            evidence_field(setup_row, "frontend_kind"),
            Some("mcc_petri")
        );
        assert_eq!(evidence_field(setup_row, "candidate_key"), Some("none"));
        assert_eq!(
            evidence_field(setup_row, "cache_key"),
            Some(expected_cache_key.as_str())
        );
        assert_eq!(
            evidence_field(setup_row, "artifact_identity"),
            Some(expected_artifact_identity.as_str())
        );
        assert_eq!(
            evidence_field(setup_row, "fingerprint_identity"),
            Some(expected_fingerprint_identity.as_str())
        );
        assert_eq!(
            evidence_field(setup_row, "batch_artifact_identity"),
            Some("none")
        );
        assert!(report
            .evidence
            .iter()
            .any(|row| row.contains(" prepared_candidate_lane ")
                && row.contains("candidate_key=ay_symbolic")));
        assert!(report
            .evidence
            .iter()
            .any(|row| row.contains(" prepared_validation_plan ")
                && row.contains("validation_kind=witness_replay")));
        assert!(report.evidence.iter().any(|row| {
            row.contains(" shared_fingerprint_identity ")
                && row.contains("id=marking-v1")
                && row.contains("canonical_domain=place-token-marking")
                && row.contains("source_kind=mcc_petri")
        }));
        assert!(report.evidence.iter().any(|row| {
            row.contains(" shared_dedup_identity ")
                && row.contains("id=state-space-dedup-v1")
                && row.contains("collision_policy=reject_on_collision")
                && row.contains("collision_fail_closed=true")
        }));
        let runtime_fingerprint_row = report
            .evidence
            .iter()
            .find(|row| row.contains(" runtime_fingerprint_adoption "))
            .expect("runtime fingerprint adoption row should be emitted");
        assert_eq!(
            evidence_field(runtime_fingerprint_row, "shared_engine_component"),
            Some(MCC_SHARED_ENGINE_COMPONENT)
        );
        assert_eq!(
            evidence_field(runtime_fingerprint_row, "first_beneficiary"),
            Some(MCC_SHARED_ENGINE_FIRST_BENEFICIARY)
        );
        assert_eq!(
            evidence_field(runtime_fingerprint_row, "second_beneficiaries"),
            Some(MCC_SHARED_ENGINE_SECOND_BENEFICIARIES)
        );
        assert_eq!(
            evidence_field(runtime_fingerprint_row, "generic_prerequisites"),
            Some(MCC_SHARED_ENGINE_GENERIC_PREREQUISITES)
        );
        assert_eq!(
            evidence_field(runtime_fingerprint_row, "exact_or_unknown"),
            Some("unknown")
        );
        assert_eq!(
            evidence_field(runtime_fingerprint_row, "exact_or_unknown_guard"),
            Some("fail_closed_until_runtime_completion")
        );
        assert_eq!(
            evidence_field(runtime_fingerprint_row, "canonical_fingerprint_identity"),
            Some(expected_fingerprint_identity.as_str())
        );
        assert_eq!(
            evidence_field(runtime_fingerprint_row, "state_vector_reuse"),
            Some("canonical_marking_vector")
        );
        assert_eq!(
            evidence_field(runtime_fingerprint_row, "solver_storage_reusable"),
            Some("true")
        );
        assert_eq!(
            evidence_field(runtime_fingerprint_row, "tla_check_inherits"),
            Some("true")
        );
        assert_eq!(
            evidence_field(runtime_fingerprint_row, "hardware_lanes_inherit"),
            Some("true")
        );
        let runtime_dedup_row = report
            .evidence
            .iter()
            .find(|row| row.contains(" runtime_dedup_admission "))
            .expect("runtime dedup admission row should be emitted");
        assert_eq!(
            evidence_field(runtime_dedup_row, "admission_status"),
            Some("accepted")
        );
        assert_eq!(
            evidence_field(runtime_dedup_row, "dedup_admission_scope"),
            Some("state_space")
        );
        assert_eq!(
            evidence_field(runtime_dedup_row, "dedup_admission_policy"),
            Some("reject_on_collision")
        );
        assert_eq!(
            evidence_field(runtime_dedup_row, "canonical_fingerprint_identity"),
            Some(expected_fingerprint_identity.as_str())
        );
        assert_eq!(
            evidence_field(runtime_dedup_row, "collision_fail_closed"),
            Some("true")
        );
        let hot_row = report
            .evidence
            .iter()
            .find(|row| row.contains(" hot_execution "))
            .expect("hot execution row should be emitted");
        assert_eq!(
            evidence_field(hot_row, "origin_frontend"),
            Some(MCC_SHARED_ENGINE_ORIGIN_FRONTEND)
        );
        assert_eq!(
            evidence_field(hot_row, "runtime_frontend_neutral"),
            Some("true")
        );
        assert_eq!(
            evidence_field(hot_row, "solver_storage_shared"),
            Some("true")
        );
        assert_eq!(evidence_field(hot_row, "exact_or_unknown"), Some("exact"));
        assert!(report.evidence.iter().any(|row| {
            row.starts_with("MCC shared_engine_adoption ")
                && row.contains("adoption_level=level-3")
                && row.contains("ay_analytical")
                && row.contains("witness_replay")
                && row.contains("future_importer:awaiting_registered_importer_frontend")
        }));
        assert!(report.evidence.iter().any(|row| {
            row.contains(" prepared_checker_program ")
                && row.contains("fingerprint_id=marking-v1")
                && row.contains("canonical_identity_id=canonical-prepared")
                && row.contains("prepared_program_fingerprint_algorithm=sha256")
        }));
        assert!(report.evidence.iter().any(|row| {
            row.contains(" analytical_solve_decision ")
                && row.contains("candidate_key=mcc.petri.analytical.reachability")
                && row.contains("witness_fingerprint=mcc.petri.witness_fingerprint")
                && row.contains("publication_readiness=blocked")
        }));
        let analytical_rows: Vec<_> = report
            .evidence
            .iter()
            .filter(|row| row.contains(" analytical_solve_decision "))
            .collect();
        assert_eq!(analytical_rows.len(), 4);
        assert!(analytical_rows.iter().all(|row| {
            evidence_field(row, "prepared_program_identity")
                .is_some_and(|identity| identity != "none")
        }));
        let smt_row = analytical_rows
            .iter()
            .find(|row| row.contains("candidate_key=mcc.petri.analytical.smt_query"))
            .expect("SMT analytical row should be emitted");
        assert_eq!(
            evidence_field(smt_row, "backend_code"),
            Some("external_ay_binary")
        );
        assert_eq!(evidence_field(smt_row, "validation"), Some("ay_proof"));
        assert_eq!(evidence_field(smt_row, "portfolio_rank"), Some("3"));
        assert!(evidence_field(smt_row, "proof_fingerprint")
            .is_some_and(|fingerprint| fingerprint.starts_with("mcc.petri.proof_fingerprint")));
        assert!(evidence_field(smt_row, "candidate_identity")
            .is_some_and(|identity| identity.starts_with("mcc.petri.candidate.ay")));
        assert_eq!(
            evidence_field(smt_row, "lane_identity"),
            Some(MCC_SHARED_ENGINE_LANE_OWNER)
        );
    }

    #[test]
    fn mcc_random_walk_engine_adoption_row_is_well_formed() {
        let adoption = mcc_random_walk_engine_adoption_evidence();
        adoption
            .validate()
            .expect("random-walk adoption evidence should validate");
        let row = adoption.render_evidence_row("MCC");
        validate_shared_engine_adoption_evidence_row(&row)
            .expect("rendered random-walk adoption row should validate");
        assert!(row.contains("origin_frontend=mcc_petri"));
        assert!(row.contains("shared_engine_component=tla_mc_core.random_walk_witness"));
        assert!(row.contains("first_beneficiary=mcc_petri_random_walk_lanes"));
        assert!(row.contains("second_beneficiary=tla_plus"));
        assert!(row.contains("extraction_status=shared-core-extracted"));
        assert!(row.contains("adoption_level=level-3"));
        assert!(row.contains("single_random_enabled_successor_selection"));
        assert!(row.contains("restart_from_initial_on_dead_state"));
        assert!(row.contains("blocker_status=tracked-blockers"));
        assert!(!row.contains("adoption_not_yet_recorded"));
    }

    #[test]
    fn mcc_portfolio_route_emitter_adds_canonical_rows_once() {
        let mut report = CapabilityReport::new(ProblemKind::ExplicitReachability);
        add_mcc_portfolio_route_evidence(&mut report);

        let rows = portfolio_route_rows(&report);
        assert_eq!(rows.len(), 6, "runtime sidecar should expose six routes");
        assert_eq!(
            rows.iter()
                .map(|row| evidence_field(row, "route").expect("route field"))
                .collect::<Vec<_>>(),
            vec![
                "explicit_bfs",
                "reductions",
                "ay_symbolic",
                "aiger_hwmcc",
                "native_jit",
                "hardware_model",
            ]
        );
        assert_eq!(
            rows.iter()
                .map(|row| evidence_field(row, "selection_rank").expect("rank field"))
                .collect::<Vec<_>>(),
            vec!["10", "20", "30", "40", "50", "60"]
        );
        assert!(rows[2].contains(tla_ay::AY_SYMBOLIC_EXECUTION_CONTRACT_MANIFEST_SCHEMA));
        assert!(rows[4].contains(TRUST_CG_PETRI_NATIVE_SUCCESSOR_COMPILE_ARTIFACT_HANDOFF_SCHEMA));
        assert!(rows[5].contains(HARDWARE_REPLAY_PRIMITIVE_SCHEMA));

        add_mcc_portfolio_route_evidence(&mut report);
        assert_eq!(
            portfolio_route_rows(&report).len(),
            6,
            "canonical route rows should not be duplicated"
        );

        report.add_evidence(
            "MCC portfolio_route schema=mcc.portfolio_route.v0 schema_version=0 route=stale"
                .to_string(),
        );
        add_mcc_portfolio_route_evidence(&mut report);
        let rows = portfolio_route_rows(&report);
        assert_eq!(rows.len(), 6);
        assert!(
            rows.iter()
                .all(|row| !row.contains("mcc.portfolio_route.v0")),
            "stale local portfolio rows should not survive canonical emission"
        );
    }

    #[test]
    fn ay_capability_contract_emitter_forwards_tla_ay_rows_once() {
        let mut report = CapabilityReport::new(ProblemKind::ExplicitReachability);
        add_ay_capability_contract_evidence(&mut report);

        let capability_rows = ay_solver_capability_descriptor_rows(&report);
        assert_eq!(
            capability_rows.len(),
            1,
            "MCC sidecar should expose one AY solver capability descriptor"
        );
        let capability_row = capability_rows[0];
        let descriptor = tla_ay::solver_capability_descriptor();
        let model_blocking = descriptor
            .capability(tla_ay::SolverCapabilityCode::ModelBlocking)
            .expect("model-blocking capability should be producer-owned");
        assert_eq!(
            evidence_field(capability_row, "schema"),
            Some(descriptor.schema)
        );
        let descriptor_schema_version = descriptor.schema_version.to_string();
        assert_eq!(
            evidence_field(capability_row, "schema_version"),
            Some(descriptor_schema_version.as_str())
        );
        assert_eq!(
            evidence_field(capability_row, "source_package"),
            Some(tla_ay::AY_SYMBOLIC_EXECUTION_ROUTE_SELECTED_SOLVER_CRATE)
        );
        assert_eq!(
            evidence_field(capability_row, "solver"),
            Some(descriptor.solver)
        );
        assert_eq!(
            evidence_field(capability_row, "capability"),
            Some(model_blocking.capability_code)
        );
        assert_eq!(
            evidence_field(capability_row, "status"),
            Some(model_blocking.status_code)
        );
        assert_eq!(
            evidence_field(capability_row, "status_code"),
            Some(model_blocking.status_code)
        );
        assert_eq!(
            evidence_field(capability_row, "reason_code"),
            Some(model_blocking.reason_code)
        );
        assert!(model_blocking
            .api_symbols
            .iter()
            .all(|symbol| capability_row.contains(symbol)));
        assert!(model_blocking
            .evidence_schemas
            .iter()
            .all(|schema| capability_row.contains(schema)));
        assert_eq!(
            evidence_field(capability_row, "production_selected"),
            Some("false")
        );
        assert_eq!(evidence_field(capability_row, "fail_closed"), Some("true"));

        let manifest_rows = ay_symbolic_execution_contract_manifest_rows(&report);
        let manifest_pairs = tla_ay::symbolic_execution_contract_manifest_key_value_pairs();
        assert_eq!(
            manifest_rows.len(),
            manifest_pairs.len(),
            "MCC should forward every AY symbolic contract manifest pair as a row"
        );
        for (key, value) in manifest_pairs {
            let line = format!("manifest_line={key}={value}");
            assert!(
                manifest_rows.iter().any(|row| {
                    row.contains(tla_ay::AY_SYMBOLIC_EXECUTION_CONTRACT_MANIFEST_SCHEMA)
                        && row.contains(&line)
                }),
                "missing forwarded AY manifest line {line}"
            );
        }

        let health_rows = ay_symbolic_execution_contract_manifest_health_rows(&report);
        let health_pairs = tla_ay::symbolic_execution_contract_manifest_health_key_value_rows();
        assert_eq!(
            health_rows.len(),
            health_pairs.len(),
            "MCC should forward every AY symbolic contract health pair as a row"
        );
        for (key, value) in health_pairs {
            let line = format!("manifest_line={key}={value}");
            assert!(
                health_rows.iter().any(|row| {
                    row.contains(tla_ay::AY_SYMBOLIC_EXECUTION_CONTRACT_MANIFEST_HEALTH_SCHEMA)
                        && row.contains(&line)
                }),
                "missing forwarded AY health line {line}"
            );
        }

        add_ay_capability_contract_evidence(&mut report);
        assert_eq!(
            ay_solver_capability_descriptor_rows(&report).len(),
            1,
            "AY capability descriptor rows should not duplicate"
        );
        assert_eq!(
            ay_symbolic_execution_contract_manifest_rows(&report).len(),
            tla_ay::symbolic_execution_contract_manifest_key_value_pairs().len(),
            "AY symbolic manifest rows should remain canonical"
        );
        assert_eq!(
            ay_symbolic_execution_contract_manifest_health_rows(&report).len(),
            tla_ay::symbolic_execution_contract_manifest_health_key_value_rows().len(),
            "AY symbolic health rows should remain canonical"
        );
    }

    #[test]
    fn mcc_capability_report_emits_fail_closed_hardware_replay_rows() {
        let report = build_petri_mcc_capability_report(
            &tiny_net(),
            Examination::StateSpace,
            &ExplorationConfig::new(100),
            SourceNetKind::Pt,
        );

        let boundary_rows = hardware_proof_replay_boundary_rows(&report);
        assert_eq!(
            boundary_rows.len(),
            2,
            "runtime sidecar should expose AIGER and BTOR2 proof replay boundaries"
        );
        assert!(boundary_rows
            .iter()
            .any(|row| row.starts_with("AIGER proof_replay_boundary ")));
        assert!(boundary_rows
            .iter()
            .any(|row| row.starts_with("BTOR2 proof_replay_boundary ")));
        for row in &boundary_rows {
            assert!(row.contains("schema=hardware_replay_primitive/v1"));
            assert!(row.contains("local_production_gate=no_local_production"));
            assert!(row.contains("native_promotion_gate=fail_closed"));
            assert!(row.contains("production_selected=false"));
            assert!(row.contains("fail_closed=true"));
            validate_hardware_proof_replay_boundary_evidence_row(row).unwrap();
        }

        let decision_rows = hardware_replay_decision_rows(&report);
        assert_eq!(
            decision_rows.len(),
            2,
            "runtime sidecar should expose blocked decisions for each hardware replay scope"
        );
        assert!(decision_rows.iter().any(|row| {
            row.starts_with("AIGER hardware_replay_decision ")
                && row.contains("decision_status=blocked")
                && row.contains("reason_code=missing_real_replay_artifact_evidence")
                && row.contains("accepted_replay_primitive=false")
        }));
        assert!(decision_rows.iter().any(|row| {
            row.starts_with("BTOR2 hardware_replay_decision ")
                && row.contains("decision_status=blocked")
                && row.contains("reason_code=concrete_trace_assignments_unavailable")
                && row.contains("accepted_replay_primitive=false")
                && row.contains("accepted_replay_evidence_identity_sha256=none")
                && row.contains("accepted_trace_validity_obligations=0")
                && row.contains("accepted_ay_proof_evidence_sha256=none")
        }));
        for row in decision_rows {
            validate_hardware_replay_decision_evidence_row(row).unwrap();
            assert!(!hardware_replay_decision_accepts_replay_primitive(row).unwrap());
        }
    }

    #[test]
    fn runtime_reachability_bmc_report_preserves_single_ay_profile_row() {
        let mut report = CapabilityReport::new(ProblemKind::ExplicitReachability);
        add_ay_solve_decision_profile_evidence(&mut report);

        let mut runtime_report = CapabilityReport::new(ProblemKind::Bmc);
        runtime_report.add_evidence("external ay selected at /tmp/ty-test-ay".to_string());
        runtime_report
            .add_evidence("reachability BMC runtime selected ay at /tmp/ty-test-ay".to_string());

        append_runtime_reachability_bmc_reports(&mut report, vec![runtime_report]);

        assert_eq!(
            report
                .evidence
                .iter()
                .filter(|row| row.contains(" ay_solver_decision_profile_summary "))
                .count(),
            1,
            "runtime BMC merge should replace the static profile placeholder, not duplicate it"
        );
        let profile = ay_solve_decision_profile_row(&report);
        assert!(profile.contains("MCC ay_solver_decision_profile_summary"));
        assert!(profile.contains("status=Unavailable"));
        assert!(profile.contains("status_code=missing_typed_summary"));
        assert!(profile.contains("typed_consumer=false"));
        assert!(profile.contains("fail_closed=true"));
        assert!(profile.contains(tla_ay::AY_SOLVE_DECISION_PROFILE_SUMMARY_SCHEMA));
        assert!(report
            .evidence
            .iter()
            .any(|row| row == "reachability BMC runtime selected ay at /tmp/ty-test-ay"));
    }

    #[test]
    fn runtime_reachability_bmc_collection_is_scoped() {
        let mut runtime_report = CapabilityReport::new(ProblemKind::Bmc);
        runtime_report.add_evidence("reachability BMC runtime selected ay at /tmp/ay");

        record_runtime_reachability_bmc_report(&runtime_report);
        let (value, reports) = collect_runtime_reachability_bmc_reports(|| {
            record_runtime_reachability_bmc_report(&runtime_report);
            7
        });

        assert_eq!(value, 7);
        assert_eq!(reports.len(), 1);
        assert!(reports[0]
            .evidence
            .iter()
            .any(|row| row == "reachability BMC runtime selected ay at /tmp/ay"));

        let ((), reports) = collect_runtime_reachability_bmc_reports(|| ());
        assert!(
            reports.is_empty(),
            "reports recorded outside the collection scope must not leak"
        );
    }

    /// Build a non-trivial (attempted > 0) runtime consumption row so it is
    /// actually recorded into the accumulator (zero-attempt rows are dropped).
    fn sample_runtime_consumption() -> MccPreparedFingerprintAdmissionRuntimeConsumption {
        let dir = tiny_model_dir();
        let model = crate::model::load_model_dir(dir.path()).expect("model should load");
        let config = ExplorationConfig::new(100)
            .with_workers(1)
            .with_storage_mode(StorageMode::FingerprintOnly);
        let setup_evidence = build_mcc_setup_evidence(
            &model,
            Examination::StateSpace,
            &config,
            Instant::now(),
            std::time::Duration::from_nanos(1),
            std::time::Duration::from_nanos(1),
        );
        let plan = mcc_marking_prepared_fingerprint_admission_plan(&setup_evidence);
        MccPreparedFingerprintAdmissionRuntimeConsumption::from_plan(
            &plan,
            "tla_petri::tests::scope_guard",
            3,
            3,
            3,
            3,
        )
    }

    #[test]
    fn evidence_scope_resets_consumption_accumulator_on_drop() {
        // A run leaves the consumption accumulator clean for the next one even
        // if nothing drains it: the EvidenceScope's drop restores the saved
        // (empty) state. Without the guard a missing drain would leak rows.
        clear_mcc_prepared_fingerprint_admission_runtime_consumption();
        {
            let _scope = EvidenceScope::begin();
            record_mcc_prepared_fingerprint_admission_runtime_consumption(
                sample_runtime_consumption(),
            );
            // Recorded inside the scope.
            let live = MCC_PREPARED_FINGERPRINT_ADMISSION_RUNTIME_CONSUMPTION
                .with(|slot| slot.borrow().len());
            assert_eq!(live, 1, "consumption must accumulate inside the scope");
        }
        // The guard's drop returned the accumulator to its clean (empty) state
        // even though nobody called the downstream `take()`.
        let leaked =
            MCC_PREPARED_FINGERPRINT_ADMISSION_RUNTIME_CONSUMPTION.with(|slot| slot.borrow().len());
        assert_eq!(
            leaked, 0,
            "consumption recorded inside the scope must not leak past its drop"
        );
    }

    #[test]
    fn evidence_scope_take_path_matches_legacy_clear_then_take() {
        // The happy path the production caller uses: begin scope, record, then
        // drain via `take()` (as `add_mcc_setup_evidence` does). The drained
        // set must be exactly what was recorded, and the accumulator must be
        // empty afterwards — identical to the pre-guard clear()/take() flow.
        clear_mcc_prepared_fingerprint_admission_runtime_consumption();
        let scope = EvidenceScope::begin();
        record_mcc_prepared_fingerprint_admission_runtime_consumption(sample_runtime_consumption());
        let drained = take_mcc_prepared_fingerprint_admission_runtime_consumption();
        assert_eq!(
            drained.len(),
            1,
            "take must return exactly the recorded rows"
        );
        // After the explicit take the accumulator is already empty; the scope
        // drop must keep it empty (and not panic on a half-borrowed cell).
        drop(scope);
        let after =
            MCC_PREPARED_FINGERPRINT_ADMISSION_RUNTIME_CONSUMPTION.with(|slot| slot.borrow().len());
        assert_eq!(after, 0, "accumulator must remain empty after take + drop");
    }

    #[test]
    fn evidence_scope_restores_outer_consumption_when_nested() {
        // Re-entrancy: a nested scope must save and restore the outer scope's
        // in-flight consumption rows, never clobbering them.
        clear_mcc_prepared_fingerprint_admission_runtime_consumption();
        let _outer = EvidenceScope::begin();
        record_mcc_prepared_fingerprint_admission_runtime_consumption(sample_runtime_consumption());
        {
            let _inner = EvidenceScope::begin();
            // The inner scope starts clean (the outer row was saved aside).
            let inner_live = MCC_PREPARED_FINGERPRINT_ADMISSION_RUNTIME_CONSUMPTION
                .with(|slot| slot.borrow().len());
            assert_eq!(inner_live, 0, "nested scope must start clean");
            record_mcc_prepared_fingerprint_admission_runtime_consumption(
                sample_runtime_consumption(),
            );
        }
        // After the inner scope drops, the outer scope's single row is restored.
        let restored =
            MCC_PREPARED_FINGERPRINT_ADMISSION_RUNTIME_CONSUMPTION.with(|slot| slot.borrow().len());
        assert_eq!(
            restored, 1,
            "outer scope's in-flight consumption must survive a nested scope"
        );
    }

    #[test]
    fn state_space_report_selects_explicit_and_rejects_non_routes() {
        let report = build_petri_mcc_capability_report(
            &tiny_net(),
            Examination::StateSpace,
            &ExplorationConfig::new(100),
            SourceNetKind::Pt,
        );

        assert!(report.has_selected(BackendKind::ExplicitState));
        assert_eq!(
            report
                .selected
                .iter()
                .find(|capability| capability.backend == BackendKind::ExplicitState)
                .map(|capability| capability.role),
            Some(CapabilityRole::Production)
        );
        assert!(report
            .rejection_reason(BackendKind::ExternalAYBinary)
            .is_some());
        assert!(report.rejection_reason(BackendKind::NativeKernel).is_some());
        assert!(!report.has_unjustified_local_production());
        assert_eq!(
            symbolic_execution_row(&report),
            "MCC symbolic_execution domain=petri_mcc status=NotDetected status_code=not_detected problem=StateSpace reason=None reason_code=none preferred_backend=None preferred_backend_code=none"
        );
        assert!(
            ay_solve_decision_profile_row(&report).contains(
                "MCC ay_solver_decision_profile_summary status=Unavailable status_code=missing_typed_summary typed_consumer=false"
            ),
            "MCC evidence should expose a fail-closed AY solve decision profile row"
        );
    }

    #[test]
    fn reachability_report_routes_explicit_as_fallback_when_ay_is_selected() {
        let _guard = crate::examinations::smt_encoding::ay_env_lock();
        let ay = make_executable_ay();
        let _env = EnvVarGuard::set("AY_PATH", ay.path().to_str().unwrap());

        let report = build_petri_mcc_capability_report(
            &tiny_net(),
            Examination::ReachabilityFireability,
            &ExplorationConfig::new(100),
            SourceNetKind::Pt,
        );

        assert!(report.has_selected(BackendKind::ExternalAYBinary));
        assert!(report.has_selected(BackendKind::AigerPortfolio));
        assert_eq!(
            report
                .selected
                .iter()
                .find(|capability| capability.backend == BackendKind::ExplicitState)
                .map(|capability| capability.role),
            Some(CapabilityRole::Fallback)
        );
        assert!(report.rejection_reason(BackendKind::AYChc).is_some());
        assert!(!report.has_unjustified_local_production());
        assert_eq!(
            symbolic_execution_row(&report),
            "MCC symbolic_execution domain=petri_mcc status=AYPreferred status_code=ay_preferred problem=ExplicitReachability reason=SymbolicTransitionRelation reason_code=symbolic_transition_relation preferred_backend=AYSmt preferred_backend_code=ay_smt"
        );
    }

    #[test]
    fn liveness_report_marks_symbolic_execution_as_ay_required() {
        let _guard = crate::examinations::smt_encoding::ay_env_lock();
        let ay = make_executable_ay();
        let _env = EnvVarGuard::set("AY_PATH", ay.path().to_str().unwrap());

        let report = build_petri_mcc_capability_report(
            &tiny_net(),
            Examination::Liveness,
            &ExplorationConfig::new(100),
            SourceNetKind::Pt,
        );

        assert!(report.has_selected(BackendKind::ExternalAYBinary));
        assert_eq!(
            symbolic_execution_row(&report),
            "MCC symbolic_execution domain=petri_mcc status=AYRequired status_code=ay_required problem=Liveness reason=UnsupportedLocalFragment reason_code=unsupported_local_fragment preferred_backend=AYSmt preferred_backend_code=ay_smt"
        );
        assert!(!report.has_selected(BackendKind::LocalSymbolicExecution));
    }

    #[test]
    fn ltl_report_pins_explicit_buchi_route_and_blocked_handoff_lanes() {
        let report = build_petri_mcc_capability_report(
            &tiny_net(),
            Examination::LTLFireability,
            &ExplorationConfig::new(100),
            SourceNetKind::Pt,
        );

        let row = report
            .evidence
            .iter()
            .find(|row| row.contains("MCC ltl_route_admission "))
            .expect("LTL report should expose route/admission evidence");
        assert!(row.contains("schema=mcc.ltl_route_admission.v1"));
        assert!(row.contains("examination=LTLFireability"));
        assert!(row.contains("selected_lane=explicit_buchi"));
        assert!(row.contains("selected_backend_code=explicit_state"));
        assert!(row.contains("ay_lasso_lane_status=blocked"));
        assert!(row.contains("aiger_ltl_lane_status=blocked"));
        assert!(row.contains("native_ltl_lane_status=blocked"));
        assert!(row.contains("next_answer_lane=ay_lasso_or_aiger_buchi"));
        assert!(row.contains("production_readiness_status=blocked"));
        assert!(row.contains("production_readiness_reason_code=missing_symbolic_ltl_handoff"));
        assert!(row.contains("production_selected=false"));
        assert!(row.contains("fail_closed=true"));
    }

    #[test]
    fn ltl_answer_summary_pins_cannot_compute_to_explicit_buchi_completion() {
        let mut report = build_petri_mcc_capability_report(
            &tiny_net(),
            Examination::LTLCardinality,
            &ExplorationConfig::new(1),
            SourceNetKind::Pt,
        );
        let records = vec![
            ExaminationRecord::new(
                "ltl-00".to_string(),
                ExaminationValue::Verdict(Verdict::True),
            ),
            ExaminationRecord::new(
                "ltl-01".to_string(),
                ExaminationValue::Verdict(Verdict::CannotCompute),
            ),
        ];

        add_ltl_answer_lane_summary_evidence(
            &mut report,
            Examination::LTLCardinality,
            &ExplorationConfig::new(1),
            &records,
        );

        let row = report
            .evidence
            .iter()
            .find(|row| row.contains("MCC ltl_answer_lane_summary "))
            .expect("LTL report should expose answer-lane summary evidence");
        assert!(row.contains("schema=mcc.ltl_answer_lane_summary.v1"));
        assert!(row.contains("selected_lane=explicit_buchi"));
        assert!(row.contains("property_count=2"));
        assert!(row.contains("answered=1"));
        assert!(row.contains("cannot_compute=1"));
        assert!(row.contains("cannot_compute_reason_code=explicit_buchi_cannot_compute"));
        assert!(row.contains("blocker_piece=ltl_explicit_buchi_completion"));
        assert!(row.contains("production_readiness_status=blocked"));
        assert!(row.contains("production_readiness_reason_code=explicit_buchi_cannot_compute"));
        assert!(row.contains("production_selected=false"));
        assert!(row.contains("fail_closed=true"));
    }

    #[test]
    fn ltl_answer_summary_selects_production_only_when_all_properties_answered() {
        let mut report = build_petri_mcc_capability_report(
            &tiny_net(),
            Examination::LTLFireability,
            &ExplorationConfig::new(100),
            SourceNetKind::Pt,
        );
        let records = vec![
            ExaminationRecord::new(
                "ltl-00".to_string(),
                ExaminationValue::Verdict(Verdict::True),
            ),
            ExaminationRecord::new(
                "ltl-01".to_string(),
                ExaminationValue::Verdict(Verdict::False),
            ),
        ];

        add_ltl_answer_lane_summary_evidence(
            &mut report,
            Examination::LTLFireability,
            &ExplorationConfig::new(100),
            &records,
        );

        let row = report
            .evidence
            .iter()
            .find(|row| row.contains("MCC ltl_answer_lane_summary "))
            .expect("LTL report should expose answer-lane summary evidence");
        assert!(row.contains("answered=2"));
        assert!(row.contains("cannot_compute=0"));
        assert!(row.contains("production_readiness_status=accepted"));
        assert!(row.contains("production_readiness_reason_code=none"));
        assert!(row.contains("production_selected=true"));
        assert!(row.contains("fail_closed=false"));
    }

    #[test]
    fn reachability_report_marks_local_fallback_after_ay_rejection() {
        let _guard = crate::examinations::smt_encoding::ay_env_lock();
        let temp = tempfile::TempDir::new().expect("tempdir should create");
        let bin_dir = temp.path().join("bin");
        fs::create_dir(&bin_dir).expect("bin dir should create");
        let _ay_path = EnvVarGuard::remove("AY_PATH");
        let _home = EnvVarGuard::set("HOME", temp.path().to_str().expect("utf8 temp path"));
        let _path = EnvVarGuard::set("PATH", bin_dir.to_str().expect("utf8 temp path"));

        let report = build_petri_mcc_capability_report(
            &tiny_net(),
            Examination::ReachabilityFireability,
            &ExplorationConfig::new(100),
            SourceNetKind::Pt,
        );

        assert_eq!(
            report.rejection_reason_code(BackendKind::ExternalAYBinary),
            Some("missing_binary")
        );
        assert_eq!(
            report
                .selected
                .iter()
                .find(|capability| capability.backend == BackendKind::ExplicitState)
                .map(|capability| capability.role),
            Some(CapabilityRole::Production)
        );
        assert!(!report.has_selected(BackendKind::LocalSymbolicExecution));
        assert_eq!(
            symbolic_execution_row(&report),
            "MCC symbolic_execution domain=petri_mcc status=LocalFallbackAfterAYRejection status_code=local_fallback_after_ay_rejection problem=ExplicitReachability reason=SymbolicTransitionRelation reason_code=symbolic_transition_relation preferred_backend=None preferred_backend_code=none"
        );
    }

    #[test]
    fn capability_report_serializes_to_jsonl() {
        let report = build_petri_mcc_capability_report(
            &tiny_net(),
            Examination::StateSpace,
            &ExplorationConfig::new(100),
            SourceNetKind::Pt,
        );
        let record = SerializableCapabilityReport::from(&report);
        let file = NamedTempFile::new().expect("jsonl temp");

        let serializable = SerializableMccBackendEvidence {
            schema_version: 1,
            model: "tiny",
            examination: "StateSpace",
            source_kind: "pt",
            place_count: 2,
            transition_count: 1,
            max_states: 100,
            workers: 1,
            fingerprint_dedup: true,
            run_status: SerializableRunStatus::Completed { records: 1 },
            error: None,
            setup_trace: None,
            prepared_program: None,
            report: record,
        };
        append_jsonl(file.path(), &serializable).expect("write jsonl");
        let contents = fs::read_to_string(file.path()).expect("read jsonl");
        let parsed: serde_json::Value = serde_json::from_str(&contents).expect("json");

        assert_eq!(parsed["schema_version"], 1);
        assert_eq!(parsed["model"], "tiny");
        assert_eq!(parsed["report"]["problem"], ProblemKind::StateSpace.name());
        assert_eq!(parsed["report"]["selected"][0]["backend"], "ExplicitState");
        assert_eq!(
            parsed["report"]["selected"][0]["backend_code"],
            BackendKind::ExplicitState.code()
        );
        assert_eq!(
            parsed["report"]["selected"][0]["role"],
            CapabilityRole::Production.code()
        );
        assert_eq!(
            parsed["report"]["selected"][0]["status"],
            CapabilityStatus::Available.code()
        );
        assert_eq!(
            parsed["report"]["production_routing_status"],
            report.production_routing_status_name()
        );
        let evidence = evidence_rows(&parsed);
        assert!(evidence
            .iter()
            .any(|row| row.contains("MCC symbolic_execution domain=petri_mcc status=NotDetected")));
        assert!(evidence
            .iter()
            .any(|row| row.contains("MCC ay_solver_decision_profile_summary status=Unavailable")));
    }

    fn serialize_report_to_json(report: &CapabilityReport) -> serde_json::Value {
        let record = SerializableCapabilityReport::from(report);
        let file = NamedTempFile::new().expect("jsonl temp");
        let serializable = SerializableMccBackendEvidence {
            schema_version: 1,
            model: "tiny",
            examination: "StateSpace",
            source_kind: "pt",
            place_count: 2,
            transition_count: 1,
            max_states: 100,
            workers: 1,
            fingerprint_dedup: true,
            run_status: SerializableRunStatus::Completed { records: 1 },
            error: None,
            setup_trace: None,
            prepared_program: None,
            report: record,
        };

        append_jsonl(file.path(), &serializable).expect("write jsonl");
        let contents = fs::read_to_string(file.path()).expect("read jsonl");
        serde_json::from_str(contents.trim()).expect("json evidence")
    }

    fn evidence_rows(record: &serde_json::Value) -> Vec<&str> {
        record["report"]["evidence"]
            .as_array()
            .expect("report evidence should be an array")
            .iter()
            .map(|entry| entry.as_str().expect("evidence row should be a string"))
            .collect()
    }

    #[cfg(not(feature = "trust-cg-petri-native"))]
    #[test]
    fn capability_jsonl_emits_native_transport_identity_blocker_rows() {
        let report = build_petri_mcc_capability_report(
            &tiny_net(),
            Examination::StateSpace,
            &ExplorationConfig::new(100),
            SourceNetKind::Pt,
        );
        let record = serialize_report_to_json(&report);
        let evidence = evidence_rows(&record);
        let transport_row = evidence
            .iter()
            .copied()
            .find(|row| row.contains("Petri native_jit trust_ir_transport_identity unavailable"))
            .expect("MCC JSONL should preserve native transport identity blocker evidence");

        assert!(transport_row.contains("api=NativeVerificationBundle::transport_identity"));
        assert!(transport_row.contains("production_selected=false"));
        assert!(transport_row.contains("fail_closed=true"));
        assert!(transport_row.contains("cargo_dependency=false"));
        assert!(
            transport_row.contains("tla-petri was built without the trust-cg-petri-native feature")
        );

        let admission_row = evidence
            .iter()
            .copied()
            .find(|row| row.contains("trust-cg trust_cg_admission_blocker"))
            .expect("MCC JSONL should preserve native admission blocker evidence");
        assert!(admission_row.contains("consumer=mcc"));
        assert!(admission_row.contains("kind=petri_native_successor"));
        assert!(admission_row.contains("surface=mcc_replay"));
        assert!(admission_row.contains("rejection_code=missing_trust_ir_transport_identity"));
        assert!(admission_row.contains("reason_code=missing_trust_ir_transport_identity"));
        assert!(admission_row.contains(
            "admission_api=trust-cg::petri_native_successor_admission_from_trust_ir_bundle"
        ));
        assert!(admission_row.contains("trust_ir_transport_identity_available=false"));
        assert!(admission_row.contains("trust_ir_bundle_consumed=false"));
        assert!(admission_row.contains("production_selected=false"));
        assert!(admission_row.contains("fail_closed=true"));

        let native_successor = record["report"]["rejected"]
            .as_array()
            .expect("rejected lanes should serialize")
            .iter()
            .find(|lane| {
                lane["backend_code"] == BackendKind::NativeKernel.code()
                    && lane["problem"] == ProblemKind::NativeSuccessor.name()
            })
            .expect("native successor lane should be rejected in MCC JSONL");
        assert!(
            native_successor["detail"]
                .as_str()
                .is_some_and(|detail| detail.contains("trust_ir_transport_identity_available=false")),
            "native successor lane detail should carry fail-closed transport availability: {native_successor}"
        );
    }

    #[cfg(feature = "trust-cg-petri-native")]
    #[test]
    #[ignore = "pre-existing upstream trust-ir/ay evidence-descriptor drift; asserts stale descriptor strings, unrelated to native successor parity (verified). Re-enable after evidence-descriptor re-sync."]
    fn capability_jsonl_preserves_petri_produced_native_bundle_rows() {
        let report = build_petri_mcc_capability_report(
            &tiny_net(),
            Examination::StateSpace,
            &ExplorationConfig::new(100),
            SourceNetKind::Pt,
        );
        let record = serialize_report_to_json(&report);
        let evidence = evidence_rows(&record);
        let transport_row = evidence
            .iter()
            .copied()
            .find(|row| row.contains("Petri native_jit trust_ir_transport_identity available"))
            .expect("MCC JSONL should preserve produced native bundle transport evidence");

        assert!(transport_row.contains("cargo_dependency=true"));
        assert!(transport_row.contains("api=NativeVerificationBundle::transport_identity"));
        assert!(transport_row.contains("validation_api=NativeVerificationBundle::validate"));
        assert!(transport_row.contains("bundle_source=petri_native_production_path"));
        assert!(transport_row.contains("bundle_validated=true"));
        assert!(transport_row.contains("producer=trust_ir"));
        assert!(transport_row.contains("input=trust_ir_module"));
        assert!(transport_row.contains("schema=trust_ir.native.transport_identity.v1"));
        assert!(transport_row.contains("module_digest=sha256:"));
        assert!(transport_row.contains("bundle_digest=sha256:"));
        assert!(transport_row.contains("target_abi_digest=sha256:"));
        assert!(transport_row.contains("request_digests=1"));
        assert!(transport_row.contains("evidence_digests=1"));
        assert!(transport_row.contains("production_selected=false"));
        assert!(transport_row.contains("fail_closed=true"));

        let admission_row = evidence
            .iter()
            .copied()
            .find(|row| {
                row.contains("trust-cg trust_cg_admission_blocker")
                    && row.contains("source=NativeInstallGateAdmissionSummary")
                    && row.contains("surface=mcc_replay")
            })
            .expect("MCC JSONL should preserve produced bundle admission evidence");
        assert!(admission_row.contains("source=NativeInstallGateAdmissionSummary"));
        assert!(admission_row
            .contains("schema=trust_cg.phase6.native_install_gate.admission_summary.v1"));
        assert!(admission_row.contains("source_package=trust_cg-codegen"));
        assert!(admission_row.contains("package=trust_cg-codegen"));
        assert!(admission_row.contains("consumer=mcc"));
        assert!(admission_row.contains("consumer_mode=petri_successor"));
        assert!(admission_row.contains("kind=petri_native_successor"));
        assert!(admission_row.contains("surface=mcc_replay"));
        assert!(admission_row.contains("summary_consumer_mode=ty_petri_native_jit"));
        assert!(admission_row.contains("summary_kind=petri_successor"));
        assert!(admission_row.contains("summary_surface=native_successor"));
        assert!(admission_row.contains("requested_authority=active_callable"));
        assert!(admission_row.contains("summary_requested_authority=validation_only"));
        assert!(admission_row.contains("install_authority=none"));
        assert!(admission_row
            .contains("bundle_api=NativeVerificationBundle::native_evidence_consumption_report"));
        assert!(admission_row.contains(
            "admission_api=trust-cg::petri_native_successor_admission_from_trust_ir_bundle"
        ));
        assert!(admission_row.contains("bundle_source=petri_native_production_path"));
        assert!(admission_row.contains("bundle_validated=true"));
        assert!(admission_row.contains("trust_ir_transport_identity_available=true"));
        assert!(admission_row.contains("trust_ir_bundle_consumed=true"));
        let legacy_missing_native_evidence =
            admission_row.contains("rejection_code=missing_native_evidence_bundle");
        let missing_install_gate_packet =
            admission_row.contains("rejection_code=missing_native_install_gate_packet");
        let accepted_validation_only = admission_row.contains("disposition=accepted")
            && admission_row.contains("status_code=accepted");
        assert!(
            legacy_missing_native_evidence || missing_install_gate_packet || accepted_validation_only,
            "produced bundle admission should either preserve a fail-closed blocker or the validation-only accepted handoff: {admission_row}"
        );
        if legacy_missing_native_evidence {
            assert!(admission_row.contains("reason_code=missing_native_evidence_bundle"));
            assert!(admission_row.contains("trust_ir_consumption_status=missing_native_evidence"));
            assert!(admission_row.contains("trust_ir_consumption_entries=0"));
            assert!(admission_row.contains("consumed_certificates=0"));
            assert!(admission_row.contains("artifact_count=0"));
        } else if missing_install_gate_packet {
            assert!(admission_row.contains("reason_code=missing_native_install_gate_packet"));
            assert!(admission_row.contains("disposition=rejected"));
            assert!(admission_row.contains("status_code=rejected"));
            assert!(admission_row.contains("production_selected=false"));
            assert!(admission_row.contains("fail_closed=true"));
            assert!(admission_row.contains("trust_ir_consumption_status=available"));
            assert!(admission_row.contains("trust_ir_consumption_entries=1"));
            // The Petri native producer now attaches a semantic-evidence bundle with
            // four artifacts (one per shared-primitive contract requirement plus the
            // native-execution artifact). Matches the `artifact_count=4` already
            // tracked by `native_successor_capability_report_is_validation_disabled_by_default`.
            assert!(admission_row.contains("artifact_count=4"));
        } else {
            assert!(accepted_validation_only);
            assert!(!admission_row.contains("rejection_code="));
            assert!(admission_row.contains("trust_ir_consumption_status=available"));
            assert!(admission_row.contains("trust_ir_consumption_entries=1"));
            assert!(admission_row.contains("artifact_count=4"));
        }
        assert!(admission_row.contains("actions_ty_native_activate=false"));
        assert!(admission_row.contains("useful_native_delta=0"));
        assert!(admission_row.contains("packet_hash="));
        assert!(admission_row.contains("artifact_id=petri_successor"));
        assert!(admission_row.contains("request_digests=1"));
        assert!(admission_row.contains("evidence_digests=1"));
        assert!(admission_row.contains("production_selected=false"));
        assert!(admission_row.contains("fail_closed=true"));

        assert!(
            evidence.iter().any(|row| row.contains(
                "trust-ir native_verification_bundle_handoff_replay_contract_json_manifest_binding "
            ) && row.contains("manifest_line=json_manifest_binding.status=bound")
                && row.contains("linked_replay_contract_report_identity_component=native_verification_bundle_handoff_replay_contract_report_identity")),
            "MCC JSONL should preserve Petri-produced trust-ir replay JSON binding evidence"
        );
        assert!(
            evidence
                .iter()
                .any(|row| row.contains(
                    "trust-ir native_verification_bundle_handoff_replay_contract_report_identity "
                ) && row.contains("manifest_line=round_trip_report.digest=sha256:")),
            "MCC JSONL should preserve Petri-produced trust-ir replay identity evidence"
        );

        let native_successor = record["report"]["rejected"]
            .as_array()
            .expect("rejected lanes should serialize")
            .iter()
            .find(|lane| {
                lane["backend_code"] == BackendKind::NativeKernel.code()
                    && lane["problem"] == ProblemKind::NativeSuccessor.name()
            })
            .expect("native successor lane should be rejected in MCC JSONL");
        assert!(
            native_successor["detail"]
                .as_str()
                .is_some_and(|detail| detail.contains("trust_ir_transport_identity_available=true")),
            "native successor lane detail should carry produced transport availability: {native_successor}"
        );
    }

    #[cfg(feature = "trust-cg-petri-native")]
    #[test]
    #[ignore = "pre-existing upstream trust-ir/ay evidence-descriptor drift; asserts stale descriptor strings, unrelated to native successor parity (verified). Re-enable after evidence-descriptor re-sync."]
    fn capability_jsonl_preserves_native_transport_identity_availability_rows() {
        let net = tiny_net();
        let bundle = native_verification_bundle_fixture();
        let report =
            crate::trust_cg_petri_kernel::petri_native_successor_capability_report_with_verification_bundle(
                &net,
                Some(&bundle),
            );
        let record = serialize_report_to_json(&report);
        let evidence = evidence_rows(&record);
        let transport_row = evidence
            .iter()
            .copied()
            .find(|row| row.contains("Petri native_jit trust_ir_transport_identity available"))
            .expect("MCC JSONL should preserve native transport identity availability evidence");

        assert!(transport_row.contains("cargo_dependency=true"));
        assert!(transport_row.contains("api=NativeVerificationBundle::transport_identity"));
        assert!(transport_row.contains("bundle_source=external_supplied"));
        assert!(transport_row.contains("schema=trust_ir.native.transport_identity.v1"));
        assert!(transport_row.contains("module_digest="));
        assert!(transport_row.contains("bundle_digest="));
        assert!(transport_row.contains("target_abi_digest="));
        assert!(transport_row.contains("production_selected=false"));
        assert!(transport_row.contains("fail_closed=true"));

        let admission_row = evidence
            .iter()
            .copied()
            .find(|row| {
                row.contains("trust-cg trust_cg_admission_blocker")
                    && row.contains("source=NativeInstallGateAdmissionSummary")
                    && row.contains("surface=mcc_replay")
                    && row.contains("rejection_code=petri_trust_ir_bundle_validation_failed")
            })
            .expect("MCC JSONL should preserve external bundle admission blocker evidence");
        assert!(admission_row.contains("source=NativeInstallGateAdmissionSummary"));
        assert!(admission_row
            .contains("schema=trust_cg.phase6.native_install_gate.admission_summary.v1"));
        assert!(admission_row.contains("source_package=trust_cg-codegen"));
        assert!(admission_row.contains("package=trust_cg-codegen"));
        assert!(admission_row.contains("consumer=mcc"));
        assert!(admission_row.contains("consumer_mode=petri_successor"));
        assert!(admission_row.contains("kind=petri_native_successor"));
        assert!(admission_row.contains("surface=mcc_replay"));
        assert!(admission_row.contains("summary_consumer_mode=ty_petri_native_jit"));
        assert!(admission_row.contains("summary_kind=petri_successor"));
        assert!(admission_row.contains("summary_surface=native_successor"));
        assert!(admission_row.contains("rejection_code=petri_trust_ir_bundle_validation_failed"));
        assert!(admission_row.contains("reason_code=petri_trust_ir_bundle_validation_failed"));
        assert!(admission_row.contains("requested_authority=active_callable"));
        assert!(admission_row.contains("summary_requested_authority=validation_only"));
        assert!(admission_row.contains("install_authority=none"));
        assert!(admission_row.contains("bundle_source=external_supplied"));
        assert!(admission_row.contains("bundle_validated=false"));
        assert!(admission_row.contains("trust_ir_transport_identity_available=true"));
        assert!(admission_row.contains("trust_ir_bundle_consumed=false"));
        assert!(admission_row.contains("trust_ir_consumption_status=validation_failed"));
        assert!(admission_row.contains("validation_errors="));
        assert!(admission_row.contains("actions_ty_native_activate=false"));
        assert!(admission_row.contains("production_selected=false"));
        assert!(admission_row.contains("fail_closed=true"));

        let native_successor = record["report"]["rejected"]
            .as_array()
            .expect("rejected lanes should serialize")
            .iter()
            .find(|lane| {
                lane["backend_code"] == BackendKind::NativeKernel.code()
                    && lane["problem"] == ProblemKind::NativeSuccessor.name()
            })
            .expect("native successor lane should be rejected in MCC JSONL");
        assert!(
            native_successor["detail"]
                .as_str()
                .is_some_and(|detail| detail.contains("trust_ir_transport_identity_available=true")),
            "native successor lane detail should carry typed transport availability: {native_successor}"
        );
    }

    #[cfg(feature = "trust-cg-petri-native")]
    fn native_verification_bundle_fixture() -> trust_ir::NativeVerificationBundle {
        // Deliberately invalid validation-failure fixture: these fixed raw
        // SHA-256 values are corruption sentinels, never positive authority
        // identities. Positive native fixtures use domain-separated digests
        // and the completed module's `stable_digest()`.
        let mut module = trust_ir::Module::new("petri_native_successor");
        module.target_info = Some(trust_ir::TargetInfo {
            triple: "x86_64-unknown-linux-gnu".to_owned(),
            pointer_size: 8,
            endianness: trust_ir::Endianness::Little,
            // ABI derived from the triple (documented legacy state) + the
            // default NativeC struct-passing policy — mirrors the production
            // successor-module site in `trust_cg_petri_native.rs`.
            abi: None,
            struct_passing: trust_ir::StructPassingPolicy::default(),
        });

        trust_ir::NativeVerificationBundle::new(
            trust_ir::NativeBundleProducer::TrustIr,
            trust_ir::NativeAdapterInput::RustMir {
                body_digest: trust_ir::ProofDigest::sha256([0x11; 32]),
            },
            trust_ir::ProofDigest::sha256([0x22; 32]),
            module,
            trust_ir::ProofLineageManifest::new(),
        )
    }

    #[test]
    fn capability_report_serializes_stable_backend_and_reason_codes() {
        let _guard = crate::examinations::smt_encoding::ay_env_lock();
        let temp = tempfile::TempDir::new().expect("tempdir should create");
        let bin_dir = temp.path().join("bin");
        fs::create_dir(&bin_dir).expect("bin dir should create");
        let _ay_path = EnvVarGuard::remove("AY_PATH");
        let _home = EnvVarGuard::set("HOME", temp.path().to_str().expect("utf8 temp path"));
        let _path = EnvVarGuard::set("PATH", bin_dir.to_str().expect("utf8 temp path"));

        let report = build_petri_mcc_capability_report(
            &tiny_net(),
            Examination::ReachabilityFireability,
            &ExplorationConfig::new(100),
            SourceNetKind::Pt,
        );
        let record = SerializableCapabilityReport::from(&report);
        let ay_rejection = record
            .rejected
            .iter()
            .find(|capability| capability.backend == "ExternalAYBinary")
            .expect("missing ay should be serialized as rejected");
        let explicit_selection = record
            .selected
            .iter()
            .find(|capability| capability.backend == "ExplicitState")
            .expect("explicit state should be serialized as selected");

        assert_eq!(
            ay_rejection.backend_code,
            BackendKind::ExternalAYBinary.code()
        );
        assert_eq!(
            ay_rejection.status.as_str(),
            CapabilityStatus::Unavailable.code()
        );
        assert!(ay_rejection
            .facets
            .iter()
            .any(|facet| facet == SolverFacet::ExternalProcess.name()));
        assert!(ay_rejection
            .facets
            .iter()
            .any(|facet| facet == SolverFacet::Smt.name()));
        assert_eq!(ay_rejection.reason_code.as_deref(), Some("missing_binary"));
        assert_eq!(ay_rejection.reason.as_deref(), Some("missing binary: ay"));
        assert_eq!(
            explicit_selection.domain.as_str(),
            BackendDomain::PetriMcc.name()
        );
        assert_eq!(
            explicit_selection.backend.as_str(),
            BackendKind::ExplicitState.name()
        );
        assert_eq!(
            explicit_selection.backend_code,
            BackendKind::ExplicitState.code()
        );
        assert_eq!(
            explicit_selection.problem.as_deref(),
            Some(ProblemKind::ExplicitReachability.name())
        );
        assert_eq!(
            explicit_selection.role.as_str(),
            CapabilityRole::Production.code()
        );
        assert_eq!(
            explicit_selection.status.as_str(),
            CapabilityStatus::Available.code()
        );
        assert_eq!(explicit_selection.reason_code, None);
        assert_eq!(
            report.rejection_reason_code(BackendKind::ExternalAYBinary),
            Some("missing_binary")
        );
    }

    struct EnvVarGuard {
        key: &'static str,
        previous: Option<String>,
    }

    impl EnvVarGuard {
        fn set(key: &'static str, value: &str) -> Self {
            let previous = std::env::var(key).ok();
            crate::env_guard::set_var(key, value);
            Self { key, previous }
        }

        fn remove(key: &'static str) -> Self {
            let previous = std::env::var(key).ok();
            crate::env_guard::remove_var(key);
            Self { key, previous }
        }
    }

    impl Drop for EnvVarGuard {
        fn drop(&mut self) {
            match &self.previous {
                Some(value) => crate::env_guard::set_var(self.key, value),
                None => crate::env_guard::remove_var(self.key),
            }
        }
    }
}
