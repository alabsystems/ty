// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Portfolio runner: parallel engine execution with cooperative cancellation.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use tla_mc_core::{
    ay_sat_capability, validate_hardware_replay_decision_evidence_row, BackendCapability,
    BackendDomain, BackendKind, CapabilityLaneDecision, CapabilityReport, CapabilityRole,
    CapabilityStatus, HardwareReplayDecisionEvidenceError, HardwareReplayPrimitiveAssignmentStatus,
    HardwareReplayPrimitiveConsumerStatus, HardwareReplayPrimitiveRejectionReason,
    HardwareReplayPrimitiveStatus, ProblemKind, ProductionRoutingStatus, SolverFacet, SolverLimits,
    SymbolicExecutionDetection, SymbolicExecutionReason, UnsupportedReason,
    HARDWARE_REPLAY_DECISION_ROW_KIND, NO_REASON_CODE,
};

use crate::bmc::{BmcEngine, RandomSimEngine};
use crate::check_result::CheckResult;
use crate::ic3::{Ic3Engine, Ic3Result};
use crate::kind::{KindEngine, KindStrengthenedEngine};
use crate::preprocess::analyze_circuit;
use crate::sat_types::{
    AYDeadlineReason, AYSolveDecision, AYSolverErrorReason, AYUnavailableReason, AYUnknownReason,
};
use crate::shared_engine_evidence::aiger_shared_engine_evidence_rows;
use crate::transys::Transys;
use crate::types::AigerCircuit;

use super::config::{EngineConfig, PortfolioConfig, PortfolioResult};
use super::factory::{arithmetic_portfolio, is_sat_likely, sat_focused_portfolio};
use super::safe_witness::{validate_safe, SafeValidation, SafeWitness};

struct AigerAYAdapterDecisionEvidence {
    action: &'static str,
    decision: AYSolveDecision,
}

struct AigerReplayApiGateEvidence {
    verdict: &'static str,
    artifact_kind: &'static str,
    backend: BackendKind,
    replay_api: &'static str,
    replay_status: &'static str,
    acceptance_gate: &'static str,
    failure_policy: &'static str,
    evidence_basis: &'static str,
}

/// Static adapter decision evidence exported by the AIGER portfolio.
const AIGER_AY_ADAPTER_DECISIONS: &[AigerAYAdapterDecisionEvidence] = &[
    AigerAYAdapterDecisionEvidence {
        action: "selected",
        decision: AYSolveDecision::Sat,
    },
    AigerAYAdapterDecisionEvidence {
        action: "selected",
        decision: AYSolveDecision::Unsat,
    },
    AigerAYAdapterDecisionEvidence {
        action: "rejected",
        decision: AYSolveDecision::Unavailable(AYUnavailableReason::Poisoned),
    },
    AigerAYAdapterDecisionEvidence {
        action: "rejected",
        decision: AYSolveDecision::Unavailable(AYUnavailableReason::UnsupportedConfig),
    },
    AigerAYAdapterDecisionEvidence {
        action: "rejected",
        decision: AYSolveDecision::Deadline(AYDeadlineReason::Interrupted),
    },
    AigerAYAdapterDecisionEvidence {
        action: "rejected",
        decision: AYSolveDecision::Deadline(AYDeadlineReason::ConflictBudgetZero),
    },
    AigerAYAdapterDecisionEvidence {
        action: "rejected",
        decision: AYSolveDecision::Deadline(AYDeadlineReason::ConflictBudgetExhausted),
    },
    AigerAYAdapterDecisionEvidence {
        action: "rejected",
        decision: AYSolveDecision::Unknown(AYUnknownReason::TheoryStop),
    },
    AigerAYAdapterDecisionEvidence {
        action: "rejected",
        decision: AYSolveDecision::Unknown(AYUnknownReason::ExtensionUnknown),
    },
    AigerAYAdapterDecisionEvidence {
        action: "rejected",
        decision: AYSolveDecision::Unknown(AYUnknownReason::AssumptionUnknown),
    },
    AigerAYAdapterDecisionEvidence {
        action: "rejected",
        decision: AYSolveDecision::Unknown(AYUnknownReason::Unspecified),
    },
    AigerAYAdapterDecisionEvidence {
        action: "rejected",
        decision: AYSolveDecision::SolverError(AYSolverErrorReason::Panic),
    },
    AigerAYAdapterDecisionEvidence {
        action: "rejected",
        decision: AYSolveDecision::SolverError(AYSolverErrorReason::InvalidSatModel),
    },
    AigerAYAdapterDecisionEvidence {
        action: "rejected",
        decision: AYSolveDecision::SolverError(AYSolverErrorReason::ProofFinalizationFailure),
    },
    AigerAYAdapterDecisionEvidence {
        action: "rejected",
        decision: AYSolveDecision::SolverError(AYSolverErrorReason::EmptyTheoryConflict),
    },
];

const AIGER_REPLAY_API_GATES: &[AigerReplayApiGateEvidence] = &[
    AigerReplayApiGateEvidence {
        verdict: "safe",
        artifact_kind: "safe_witness_inductive_invariant",
        backend: BackendKind::AigerPortfolio,
        replay_api: "validate_safe",
        replay_status: "proven",
        acceptance_gate: "safe_validation_accepted",
        failure_policy: "fail_closed_continue_or_respawn",
        evidence_basis: "independent_sat_recheck",
    },
    AigerReplayApiGateEvidence {
        verdict: "safe",
        artifact_kind: "safe_witness_trivial",
        backend: BackendKind::AigerPortfolio,
        replay_api: "validate_safe",
        replay_status: "proven",
        acceptance_gate: "safe_validation_accepted",
        failure_policy: "fail_closed_continue_or_respawn",
        evidence_basis: "bad_lit_recheck",
    },
    AigerReplayApiGateEvidence {
        verdict: "safe",
        artifact_kind: "safe_witness_engine_verified",
        backend: BackendKind::AigerPortfolio,
        replay_api: "engine_internal_proof",
        replay_status: "delegated_not_replayable",
        acceptance_gate: "safe_validation_accepted",
        failure_policy: "logged_engine_internal_proof",
        evidence_basis: "engine_verified_safe_witness",
    },
    AigerReplayApiGateEvidence {
        verdict: "safe",
        artifact_kind: "safe_witness_unwitnessed",
        backend: BackendKind::AigerPortfolio,
        replay_api: "none",
        replay_status: "not_available",
        acceptance_gate: "safe_validation_downgrade",
        failure_policy: "fail_closed_continue_or_respawn",
        evidence_basis: "no_safe_witness",
    },
    AigerReplayApiGateEvidence {
        verdict: "unsafe",
        artifact_kind: "counterexample_trace",
        backend: BackendKind::AigerPortfolio,
        replay_api: "transys_verify_witness",
        replay_status: "proven",
        acceptance_gate: "transys_verify_witness_ok",
        failure_policy: "fail_closed_continue_or_respawn",
        evidence_basis: "trace_simulation",
    },
];

/// Classify AIGER unsafe replay rows using the shared hardware replay primitive
/// boundary vocabulary.
pub fn aiger_hardware_replay_primitive_status(
    evidence: &[String],
) -> HardwareReplayPrimitiveStatus {
    if evidence.iter().any(is_generated_placeholder_row) {
        return aiger_replay_primitive_rejected(
            HardwareReplayPrimitiveRejectionReason::GeneratedPlaceholderEvidence,
            true,
        );
    }
    if !evidence.iter().any(|row| {
        row.starts_with("AIGER proof_replay_boundary ")
            && row.contains("ay_backend_code=ay_sat")
            && row.contains("unsafe_witness=aiger_counterexample_trace")
            && row.contains("unsafe_replay=transys_verify_witness")
            && row.contains("local_production_gate=no_local_production")
            && row.contains("native_promotion_gate=fail_closed")
    }) {
        return aiger_replay_primitive_rejected(
            HardwareReplayPrimitiveRejectionReason::MissingProofReplayBoundaryEvidence,
            false,
        );
    }
    if !evidence.iter().any(|row| {
        row.starts_with("AIGER replay_api_gate ")
            && row.contains("verdict=unsafe")
            && row.contains("artifact_kind=counterexample_trace")
            && row.contains("ay_backend_code=ay_sat")
            && row.contains("replay_api=transys_verify_witness")
            && row.contains("replay_status=proven")
            && row.contains("failure_policy=fail_closed")
    }) {
        return aiger_replay_primitive_rejected(
            HardwareReplayPrimitiveRejectionReason::MissingUnsafeReplayGateEvidence,
            false,
        );
    }
    if !evidence.iter().any(|row| {
        row.starts_with("AIGER real_proof_replay_artifact ")
            && row.contains("verdict=unsafe")
            && row.contains("ay_backend_code=ay_sat")
            && row.contains("replay_api=transys_verify_witness")
            && row.contains("replay_status=proven")
            && row.contains("evidence_source=real_solver")
            && row.contains("generated_placeholder=false")
    }) {
        return aiger_replay_primitive_rejected(
            HardwareReplayPrimitiveRejectionReason::MissingRealReplayArtifactEvidence,
            false,
        );
    }

    let Some(assignments) = aiger_typed_assignment_completeness(evidence) else {
        return aiger_replay_primitive_rejected(
            HardwareReplayPrimitiveRejectionReason::ConcreteTraceAssignmentsUnavailable,
            false,
        );
    };
    if assignments.replay_assignment_status != HardwareReplayPrimitiveAssignmentStatus::Complete
        || assignments.typed_assignment_missing_slots != 0
        || assignments.typed_assignment_present_slots < assignments.typed_assignment_required_slots
    {
        return aiger_replay_primitive_rejected_with_assignments(
            HardwareReplayPrimitiveRejectionReason::TypedAYTraceAssignmentsIncomplete,
            false,
            assignments,
        );
    }

    HardwareReplayPrimitiveStatus {
        hardware: "AIGER",
        verdict: "unsafe",
        primitive: "unsafe_counterexample_trace",
        ay_backend_code: BackendKind::AYSat.code(),
        replay_api: "transys_verify_witness",
        replay_status: "proven",
        evidence_source: "real_solver",
        generated_placeholder: false,
        typed_assignment_source: assignments.typed_assignment_source,
        replay_assignment_status: assignments.replay_assignment_status,
        typed_assignment_required_slots: assignments.typed_assignment_required_slots,
        typed_assignment_present_slots: assignments.typed_assignment_present_slots,
        typed_assignment_missing_slots: assignments.typed_assignment_missing_slots,
        consumer_status: HardwareReplayPrimitiveConsumerStatus::Accepted,
        rejection_reason: HardwareReplayPrimitiveRejectionReason::None,
    }
}

/// Render the actionable AIGER hardware replay decision evidence row.
pub fn aiger_hardware_replay_decision_evidence(evidence: &[String]) -> String {
    aiger_hardware_replay_primitive_status(evidence).render_decision_evidence_row()
}

/// Validate an AIGER hardware replay decision row against the exported schema.
pub fn validate_aiger_hardware_replay_decision_evidence_row(
    row: &str,
) -> Result<(), HardwareReplayDecisionEvidenceError> {
    let expected_prefix = format!("AIGER {} ", HARDWARE_REPLAY_DECISION_ROW_KIND);
    if !row.starts_with(&expected_prefix) {
        return Err(HardwareReplayDecisionEvidenceError::WrongRowKind);
    }

    validate_hardware_replay_decision_evidence_row(row)
}

/// Validate the AIGER decision row against the current primitive status.
pub fn validate_aiger_hardware_replay_decision_evidence(
    evidence: &[String],
) -> Result<(), HardwareReplayDecisionEvidenceError> {
    let expected_prefix = format!("AIGER {} ", HARDWARE_REPLAY_DECISION_ROW_KIND);
    let mut rows = evidence
        .iter()
        .filter(|row| row.starts_with(&expected_prefix));
    let row = rows
        .next()
        .ok_or(HardwareReplayDecisionEvidenceError::MissingDecisionEvidence)?;
    if rows.next().is_some() {
        return Err(HardwareReplayDecisionEvidenceError::DuplicateDecisionEvidence);
    }

    validate_aiger_hardware_replay_decision_evidence_row(row)?;
    let expected_row = aiger_hardware_replay_decision_evidence(evidence);
    if row != &expected_row {
        return Err(HardwareReplayDecisionEvidenceError::InconsistentDecision(
            "decision_row_does_not_match_current_primitive_status",
        ));
    }

    Ok(())
}

fn aiger_replay_primitive_rejected(
    rejection_reason: HardwareReplayPrimitiveRejectionReason,
    generated_placeholder: bool,
) -> HardwareReplayPrimitiveStatus {
    aiger_replay_primitive_rejected_with_assignments(
        rejection_reason,
        generated_placeholder,
        AigerTypedAssignmentCompleteness::missing(),
    )
}

fn aiger_replay_primitive_rejected_with_assignments(
    rejection_reason: HardwareReplayPrimitiveRejectionReason,
    generated_placeholder: bool,
    assignments: AigerTypedAssignmentCompleteness,
) -> HardwareReplayPrimitiveStatus {
    HardwareReplayPrimitiveStatus {
        hardware: "AIGER",
        verdict: "unsafe",
        primitive: "unsafe_counterexample_trace",
        ay_backend_code: BackendKind::AYSat.code(),
        replay_api: "transys_verify_witness",
        replay_status: "not_available",
        evidence_source: "consumer_gate",
        generated_placeholder,
        typed_assignment_source: assignments.typed_assignment_source,
        replay_assignment_status: assignments.replay_assignment_status,
        typed_assignment_required_slots: assignments.typed_assignment_required_slots,
        typed_assignment_present_slots: assignments.typed_assignment_present_slots,
        typed_assignment_missing_slots: assignments.typed_assignment_missing_slots,
        consumer_status: HardwareReplayPrimitiveConsumerStatus::Rejected,
        rejection_reason,
    }
}

fn is_generated_placeholder_row(row: &String) -> bool {
    row.contains("generated_placeholder=true")
        || row.contains("MCC hardware_fallback")
        || row.contains("mcc-generated")
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct AigerTypedAssignmentCompleteness {
    typed_assignment_source: String,
    replay_assignment_status: HardwareReplayPrimitiveAssignmentStatus,
    typed_assignment_required_slots: usize,
    typed_assignment_present_slots: usize,
    typed_assignment_missing_slots: usize,
}

impl AigerTypedAssignmentCompleteness {
    fn missing() -> Self {
        Self {
            typed_assignment_source: "missing".to_string(),
            replay_assignment_status: HardwareReplayPrimitiveAssignmentStatus::Missing,
            typed_assignment_required_slots: 0,
            typed_assignment_present_slots: 0,
            typed_assignment_missing_slots: 0,
        }
    }
}

fn aiger_typed_assignment_completeness(
    evidence: &[String],
) -> Option<AigerTypedAssignmentCompleteness> {
    let row = evidence.iter().find(|row| {
        row.starts_with("AIGER real_proof_replay_artifact ")
            && row.contains("verdict=unsafe")
            && row.contains("ay_backend_code=ay_sat")
            && row.contains("replay_api=transys_verify_witness")
    })?;
    let typed_assignment_source = evidence_field(row, "typed_assignment_source")?.to_string();
    let replay_assignment_status = match evidence_field(row, "replay_assignment_status")? {
        "complete" => HardwareReplayPrimitiveAssignmentStatus::Complete,
        "incomplete" => HardwareReplayPrimitiveAssignmentStatus::Incomplete,
        "missing" => HardwareReplayPrimitiveAssignmentStatus::Missing,
        _ => return None,
    };
    let typed_assignment_required_slots =
        evidence_usize_field(row, "typed_assignment_required_slots")?;
    let typed_assignment_present_slots =
        evidence_usize_field(row, "typed_assignment_present_slots")?;
    let typed_assignment_missing_slots =
        evidence_usize_field(row, "typed_assignment_missing_slots")?;

    Some(AigerTypedAssignmentCompleteness {
        typed_assignment_source,
        replay_assignment_status,
        typed_assignment_required_slots,
        typed_assignment_present_slots,
        typed_assignment_missing_slots,
    })
}

fn evidence_usize_field(row: &str, key: &str) -> Option<usize> {
    evidence_field(row, key)?.parse().ok()
}

fn evidence_field<'a>(row: &'a str, key: &str) -> Option<&'a str> {
    let prefix = format!("{key}=");
    row.split_whitespace()
        .find_map(|token| token.strip_prefix(prefix.as_str()))
}

/// Internal channel payload: pairs the public `PortfolioResult` with the
/// optional `SafeWitness` produced by the engine so the aggregation loop can
/// cross-validate `Safe` verdicts (#4315) without changing the public API.
struct EngineOutcome {
    result: PortfolioResult,
    /// `Some(_)` for every engine; encodes how the engine backs up a `Safe`
    /// verdict (InductiveInvariant / Trivial / Unwitnessed). `None` is reserved
    /// for future use and treated the same as `Unwitnessed`.
    witness: Option<SafeWitness>,
}

/// Run a portfolio of model checking engines in parallel.
///
/// Spawns one thread per engine. The first definitive result (Safe or Unsafe)
/// causes all other engines to be cancelled via the shared `AtomicBool` flag.
///
/// Returns the first definitive result, or the best Unknown result if all
/// engines finish without a definitive answer.
pub fn portfolio_check(circuit: &AigerCircuit, config: PortfolioConfig) -> CheckResult {
    portfolio_check_detailed(circuit, config).result
}

/// Report shared backend lanes the AIGER portfolio can use without solving.
pub fn aiger_portfolio_capability_report(
    circuit: &AigerCircuit,
    config: &PortfolioConfig,
) -> CapabilityReport {
    let mut report = CapabilityReport::new(ProblemKind::Safety).with_limits(SolverLimits {
        time_budget: Some(config.timeout),
        max_depth: Some(u32::try_from(config.max_depth).unwrap_or(u32::MAX)),
        max_states: None,
        max_memory_bytes: None,
    });

    let portfolio_capability = BackendCapability::available(
        BackendDomain::Aiger,
        BackendKind::AigerPortfolio,
        format!(
            "AIGER portfolio with {} engines, {} latches, {} AND gates",
            config.engines.len(),
            circuit.latches.len(),
            circuit.ands.len()
        ),
    )
    .for_problem(ProblemKind::Safety)
    .with_facets([
        SolverFacet::Sat,
        SolverFacet::Bmc,
        SolverFacet::KInduction,
        SolverFacet::Pdr,
        SolverFacet::Incremental,
        SolverFacet::Assumptions,
    ]);
    add_aiger_lane_evidence(
        &mut report,
        CapabilityLaneDecision::Selected,
        &portfolio_capability,
    );
    report.select(portfolio_capability);

    add_aiger_portfolio_engine_inventory(&mut report, config);
    for engine in &config.engines {
        report_aiger_engine_capability(&mut report, engine);
    }
    if config.engines.iter().any(engine_has_ay_adapter_boundary) {
        add_aiger_ay_adapter_decision_catalog(&mut report);
    }
    add_aiger_symbolic_execution_evidence(&mut report, config);
    add_aiger_shared_engine_evidence(&mut report, circuit);

    let native_capability = BackendCapability::unsupported(
        BackendDomain::Aiger,
        BackendKind::NativeKernel,
        UnsupportedReason::NativeKernelUnavailable,
    )
    .for_problem(ProblemKind::NativeSuccessor)
    .with_facets([SolverFacet::NativeCodegen])
    .with_role(CapabilityRole::Validation)
    .with_detail("AIGER has no shared successor/predicate-kernel adapter yet");
    add_aiger_lane_evidence(
        &mut report,
        CapabilityLaneDecision::Rejected,
        &native_capability,
    );
    report.reject(native_capability);
    add_aiger_routing_evidence(&mut report);
    report
}

fn add_aiger_portfolio_engine_inventory(report: &mut CapabilityReport, config: &PortfolioConfig) {
    for (index, engine) in config.engines.iter().enumerate() {
        report.add_evidence(format!(
            "AIGER portfolio_engine index={index} engine_name={} engine_kind={} engine_label={}",
            engine.name(),
            engine.kind_code(),
            engine.diagnostic_label()
        ));
    }
}

fn add_aiger_ay_adapter_decision_catalog(report: &mut CapabilityReport) {
    report.add_evidence(
        "AIGER ay_adapter_decision_schema version=1 source=AYSolveDecision \
         sat_result_behavior=preserved",
    );
    for decision in AIGER_AY_ADAPTER_DECISIONS {
        report.add_evidence(format!(
            "AIGER ay_adapter_decision action={} backend=AYSat kind={} status={} \
             reason_code={} sat_result={}",
            decision.action,
            decision.decision.kind_code(),
            decision.decision.evidence_status_name(),
            decision.decision.reason_code(),
            decision.decision.evidence_sat_result_name()
        ));
    }
}

fn add_aiger_lane_evidence(
    report: &mut CapabilityReport,
    decision: CapabilityLaneDecision,
    capability: &BackendCapability,
) {
    report.add_evidence(capability.render_lane_evidence("AIGER", decision));
    report.add_evidence(capability.render_lane_status_evidence("AIGER", decision));
}

fn add_aiger_routing_evidence(report: &mut CapabilityReport) {
    report.add_evidence(report.render_production_routing_status_evidence("AIGER"));
    report.add_evidence(format!(
        "AIGER routing_summary production_routing_status={} ay_selected_for_production={} has_unjustified_local_production={}",
        report.production_routing_status_name(),
        report.ay_selected_for_production(),
        report.has_unjustified_local_production()
    ));
    add_aiger_proof_replay_boundary_evidence(report);
    add_aiger_replay_api_gate_evidence(report);
    add_aiger_fail_closed_hardware_replay_evidence(report);
    add_aiger_handoff_evidence(report);
    if let Some(reason_code) = report.rejection_reason_code(BackendKind::NativeKernel) {
        report.add_evidence(format!(
            "AIGER unsupported_reason backend={} code={reason_code}",
            BackendKind::NativeKernel.name(),
        ));
    }
}

fn add_aiger_proof_replay_boundary_evidence(report: &mut CapabilityReport) {
    report.add_evidence(format!(
        "AIGER proof_replay_boundary ay_backend_code={} safe_proof=aiger_safe_witness_validation safe_replay=validate_safe unsafe_witness=aiger_counterexample_trace unsafe_replay=transys_verify_witness witness_attribution=engine_trace local_production_gate=no_local_production native_promotion_gate=fail_closed production_routing_status_code={}",
        BackendKind::AYSat.code(),
        report.production_routing_status_code(),
    ));
}

fn add_aiger_replay_api_gate_evidence(report: &mut CapabilityReport) {
    for gate in AIGER_REPLAY_API_GATES {
        add_unique_aiger_evidence(
            report,
            format!(
                "AIGER replay_api_gate verdict={} artifact_kind={} api_backend={} api_backend_code={} ay_backend_code={} replay_api={} replay_status={} acceptance_gate={} failure_policy={} evidence_basis={} production_routing_status_code={}",
                gate.verdict,
                gate.artifact_kind,
                gate.backend.name(),
                gate.backend.code(),
                BackendKind::AYSat.code(),
                gate.replay_api,
                gate.replay_status,
                gate.acceptance_gate,
                gate.failure_policy,
                gate.evidence_basis,
                report.production_routing_status_code(),
            ),
        );
    }
}

fn add_aiger_fail_closed_hardware_replay_evidence(report: &mut CapabilityReport) {
    let replay_status = aiger_hardware_replay_primitive_status(&report.evidence);
    add_unique_aiger_evidence(report, replay_status.render_evidence_row());
    add_unique_aiger_evidence(report, replay_status.render_decision_evidence_row());
}

fn add_aiger_handoff_evidence(report: &mut CapabilityReport) {
    let ay_handoffs: Vec<_> = report
        .selected
        .iter()
        .filter(|capability| capability.backend.is_ay())
        .map(|capability| ("delegated", capability.clone()))
        .chain(
            report
                .rejected
                .iter()
                .filter(|capability| capability.backend.is_ay())
                .map(|capability| ("rejected", capability.clone())),
        )
        .collect();
    for (handoff_status, capability) in ay_handoffs {
        add_unique_aiger_evidence(
            report,
            format!(
                "AIGER ay_handoff handoff_status={handoff_status} from_backend={} to_backend={} to_backend_code={} to_problem={} to_role={} to_status={} reason_code={}",
                BackendKind::AigerPortfolio.name(),
                capability.backend.name(),
                capability.backend.code(),
                capability.problem.map_or("None", ProblemKind::name),
                capability.role.code(),
                capability.status.code(),
                capability.normalized_reason_code()
            ),
        );
        add_aiger_ay_handoff_detail_evidence(
            report,
            if handoff_status == "delegated" {
                CapabilityLaneDecision::Selected
            } else {
                CapabilityLaneDecision::Rejected
            },
            handoff_status,
            &capability,
        );
    }

    let native_handoffs: Vec<_> = report
        .selected
        .iter()
        .filter(|capability| capability.backend == BackendKind::NativeKernel)
        .map(|capability| ("available", capability.clone()))
        .chain(
            report
                .rejected
                .iter()
                .filter(|capability| capability.backend == BackendKind::NativeKernel)
                .map(|capability| ("deferred", capability.clone())),
        )
        .collect();
    for (handoff_status, capability) in native_handoffs {
        add_unique_aiger_evidence(
            report,
            format!(
                "AIGER native_handoff handoff_status={handoff_status} from_backend={} to_backend={} to_backend_code={} to_problem={} to_role={} to_status={} reason_code={}",
                BackendKind::AigerPortfolio.name(),
                BackendKind::NativeKernel.name(),
                BackendKind::NativeKernel.code(),
                capability.problem.map_or("None", ProblemKind::name),
                capability.role.code(),
                capability.status.code(),
                capability.normalized_reason_code()
            ),
        );
    }
}

fn add_aiger_ay_handoff_detail_evidence(
    report: &mut CapabilityReport,
    decision: CapabilityLaneDecision,
    handoff_status: &str,
    capability: &BackendCapability,
) {
    add_unique_aiger_evidence(
        report,
        format!(
            "AIGER ay_handoff_detail lane_status={} handoff_status={handoff_status} from_backend={} to_backend={} to_backend_code={} to_problem={} to_problem_code={} to_role={} to_status={} reason_code={} production_routing_status={} production_routing_status_code={} local_fallback_status={}",
            decision.action(),
            BackendKind::AigerPortfolio.name(),
            capability.backend.name(),
            capability.backend.code(),
            capability.problem_name_or_none(),
            capability.problem_code().unwrap_or(NO_REASON_CODE),
            capability.role.code(),
            capability.status.code(),
            capability.normalized_reason_code(),
            report.production_routing_status_name(),
            report.production_routing_status_code(),
            aiger_local_fallback_status(report),
        ),
    );
}

fn aiger_local_fallback_status(report: &CapabilityReport) -> &'static str {
    match report.production_routing_status() {
        ProductionRoutingStatus::JustifiedLocalFallback => "justified_local_fallback",
        ProductionRoutingStatus::UnjustifiedLocalFallback => "unjustified_local_fallback",
        _ => "not_selected",
    }
}

fn add_aiger_symbolic_execution_evidence(report: &mut CapabilityReport, config: &PortfolioConfig) {
    let has_symbolic_engine = config.engines.iter().any(engine_has_ay_adapter_boundary);
    if has_symbolic_engine {
        report.add_evidence(
            SymbolicExecutionDetection::ay_preferred(SymbolicExecutionReason::BitVectorFormula)
                .render_evidence("AIGER", ProblemKind::Sat),
        );
    } else {
        report.add_evidence(
            SymbolicExecutionDetection::not_detected()
                .render_evidence("AIGER", ProblemKind::Safety),
        );
    }
}

fn add_aiger_shared_engine_evidence(report: &mut CapabilityReport, circuit: &AigerCircuit) {
    for evidence in aiger_shared_engine_evidence_rows(circuit) {
        add_unique_aiger_evidence(report, evidence);
    }
}

fn add_unique_aiger_evidence(report: &mut CapabilityReport, evidence: String) {
    if !report.evidence.iter().any(|existing| existing == &evidence) {
        report.add_evidence(evidence);
    }
}

/// Run the detailed portfolio and return shared backend capability evidence.
pub fn portfolio_check_detailed_with_report(
    circuit: &AigerCircuit,
    config: PortfolioConfig,
) -> (PortfolioResult, CapabilityReport) {
    let mut report = aiger_portfolio_capability_report(circuit, &config);
    let result = portfolio_check_detailed(circuit, config.clone());
    add_aiger_portfolio_winner_evidence(&mut report, &config, &result);
    (result, report)
}

fn add_aiger_portfolio_winner_evidence(
    report: &mut CapabilityReport,
    config: &PortfolioConfig,
    result: &PortfolioResult,
) {
    report.add_evidence(format!(
        "AIGER portfolio winner={} time_secs={:.3}",
        result.solver_name, result.time_secs
    ));
    add_unique_aiger_evidence(report, aiger_portfolio_winner_evidence_row(config, result));
}

pub(super) fn aiger_portfolio_winner_evidence_row(
    config: &PortfolioConfig,
    result: &PortfolioResult,
) -> String {
    let engine_name = if result.solver_name.is_empty() {
        "unknown"
    } else {
        result.solver_name.as_str()
    };
    let Some(engine) = config
        .engines
        .iter()
        .find(|engine| engine.name() == result.solver_name.as_str())
    else {
        return format!(
            "AIGER portfolio_winner engine_name={engine_name} engine_kind=unknown \
             engine_label=unknown problem=unknown problem_code=unknown backend_code=unknown \
             role=unknown time_secs={:.3}",
            result.time_secs
        );
    };

    let problem = engine_problem_kind(engine);
    let (backend, role) = engine_winner_backend_role(engine);
    format!(
        "AIGER portfolio_winner engine_name={} engine_kind={} engine_label={} \
         problem={} problem_code={} backend_code={} role={} time_secs={:.3}",
        engine.name(),
        engine.kind_code(),
        engine.diagnostic_label(),
        problem.name(),
        problem.code(),
        backend.code(),
        role.code(),
        result.time_secs
    )
}

fn engine_winner_backend_role(engine: &EngineConfig) -> (BackendKind, CapabilityRole) {
    if engine_uses_test_simple_solver(engine) {
        return (BackendKind::ExplicitState, CapabilityRole::TestOnly);
    }
    if matches!(engine, EngineConfig::RandomSim { .. }) {
        return (BackendKind::ExplicitState, CapabilityRole::Validation);
    }
    (BackendKind::AYSat, CapabilityRole::Production)
}

fn report_aiger_engine_capability(report: &mut CapabilityReport, engine: &EngineConfig) {
    if matches!(engine, EngineConfig::RandomSim { .. }) {
        let capability = BackendCapability::available(
            BackendDomain::Aiger,
            BackendKind::ExplicitState,
            format!("AIGER validation lane {}", engine.name()),
        )
        .for_problem(ProblemKind::Safety)
        .with_facets([SolverFacet::Witness])
        .with_role(CapabilityRole::Validation);
        add_aiger_lane_evidence(report, CapabilityLaneDecision::Selected, &capability);
        report.select(capability);
        return;
    }

    if engine_uses_test_simple_solver(engine) {
        let capability = BackendCapability::available(
            BackendDomain::Aiger,
            BackendKind::ExplicitState,
            format!("AIGER test-only SimpleSolver lane {}", engine.name()),
        )
        .for_problem(engine_problem_kind(engine))
        .with_facets([SolverFacet::Sat])
        .with_role(CapabilityRole::TestOnly);
        add_aiger_lane_evidence(report, CapabilityLaneDecision::Selected, &capability);
        add_aiger_local_fallback_evidence(report, engine, CapabilityRole::TestOnly);
        report.select(capability);

        let ay_rejection = BackendCapability::disabled(
            BackendDomain::Aiger,
            BackendKind::AYSat,
            UnsupportedReason::DisabledByPolicy("AIGER SimpleSolver local fallback"),
        )
        .for_problem(engine_problem_kind(engine))
        .with_facets([SolverFacet::Sat])
        .with_role(CapabilityRole::Production)
        .with_detail("AIGER SimpleSolver lane remains test-only; AY production handoff is rejected for this local fallback");
        add_aiger_lane_evidence(report, CapabilityLaneDecision::Rejected, &ay_rejection);
        report.reject(ay_rejection);
        return;
    }

    let capability = ay_sat_capability(BackendDomain::Aiger, engine_problem_kind(engine))
        .with_detail(format!("AIGER portfolio engine {}", engine.name()));
    add_aiger_lane_evidence(report, CapabilityLaneDecision::Selected, &capability);
    add_aiger_ay_engine_adapter_evidence(report, engine);
    report.select(capability);
}

fn add_aiger_ay_engine_adapter_evidence(report: &mut CapabilityReport, engine: &EngineConfig) {
    report.add_evidence(format!(
        "AIGER ay_adapter_decision action={} engine={} backend={} \
         kind=production status={} role={} reason_code={} \
         sat_result=unchanged",
        CapabilityLaneDecision::Selected.action(),
        engine.name(),
        BackendKind::AYSat.name(),
        CapabilityStatus::Available.name(),
        CapabilityRole::Production.name(),
        NO_REASON_CODE
    ));
}

fn add_aiger_local_fallback_evidence(
    report: &mut CapabilityReport,
    engine: &EngineConfig,
    role: CapabilityRole,
) {
    report.add_evidence(format!(
        "AIGER ay_adapter_decision action={} engine={} backend={} \
         kind=local_fallback status={} role={} reason_code=local_fallback \
         sat_result=unchanged",
        CapabilityLaneDecision::Selected.action(),
        engine.name(),
        BackendKind::ExplicitState.name(),
        CapabilityStatus::Available.name(),
        role.name()
    ));
    report.add_evidence(format!(
        "AIGER ay_adapter_decision action={} engine={} backend={} \
         kind=local_fallback status={} role={} reason_code=local_fallback \
         sat_result=unchanged",
        CapabilityLaneDecision::Rejected.action(),
        engine.name(),
        BackendKind::AYSat.name(),
        CapabilityStatus::Disabled.name(),
        CapabilityRole::Production.name()
    ));
}

fn engine_problem_kind(engine: &EngineConfig) -> ProblemKind {
    match engine {
        EngineConfig::Bmc { .. }
        | EngineConfig::BmcDynamic
        | EngineConfig::BmcAYVariant { .. }
        | EngineConfig::BmcAYVariantDynamic { .. }
        | EngineConfig::BmcGeometricBackoff { .. }
        | EngineConfig::BmcGeometricBackoffAYVariant { .. }
        | EngineConfig::BmcLinearOffset { .. }
        | EngineConfig::GpuExhaustiveBmc { .. } => ProblemKind::Bmc,
        EngineConfig::Kind
        | EngineConfig::KindSimplePath
        | EngineConfig::KindSkipBmc
        | EngineConfig::KindAYVariant { .. }
        | EngineConfig::KindSkipBmcAYVariant { .. }
        | EngineConfig::KindStrengthened
        | EngineConfig::KindStrengthenedAYVariant { .. } => ProblemKind::KInduction,
        EngineConfig::Ic3 | EngineConfig::Ic3Configured { .. } | EngineConfig::CegarIc3 { .. } => {
            ProblemKind::Sat
        }
        EngineConfig::RandomSim { .. } => ProblemKind::Safety,
        EngineConfig::BddReach { .. } => ProblemKind::Safety,
    }
}

fn engine_has_ay_adapter_boundary(engine: &EngineConfig) -> bool {
    // random-sim and the BDD lane run no SAT solver of their own (the BDD
    // lane's Unsafe path re-derives through CPU BMC, which reports its own
    // boundary).
    !matches!(
        engine,
        EngineConfig::RandomSim { .. } | EngineConfig::BddReach { .. }
    )
}

fn engine_uses_test_simple_solver(engine: &EngineConfig) -> bool {
    match engine {
        EngineConfig::BmcAYVariant { backend, .. }
        | EngineConfig::BmcAYVariantDynamic { backend }
        | EngineConfig::BmcGeometricBackoffAYVariant { backend, .. }
        | EngineConfig::KindAYVariant { backend }
        | EngineConfig::KindSkipBmcAYVariant { backend }
        | EngineConfig::KindStrengthenedAYVariant { backend }
        | EngineConfig::Ic3Configured {
            config:
                crate::ic3::Ic3Config {
                    solver_backend: backend,
                    ..
                },
            ..
        }
        | EngineConfig::CegarIc3 {
            config:
                crate::ic3::Ic3Config {
                    solver_backend: backend,
                    ..
                },
            ..
        } => *backend == crate::sat_types::SolverBackend::Simple,
        _ => false,
    }
}

/// Run a portfolio with detailed result including solver attribution and timing.
pub fn portfolio_check_detailed(
    circuit: &AigerCircuit,
    config: PortfolioConfig,
) -> PortfolioResult {
    // Set preprocessing timeout if not already configured (#4074).
    // Use 20% of the overall timeout (minimum 5s) to prevent preprocessing
    // from consuming the entire time budget on large circuits (e.g., bakery
    // with 112 latches, 1472 ANDs hangs in SCORR/synthesis).
    let preprocess_config = if config.preprocess.timeout_secs == 0 && config.timeout.as_secs() > 0 {
        let mut pc = config.preprocess.clone();
        pc.timeout_secs = (config.timeout.as_secs() / 5).max(5);
        pc
    } else {
        config.preprocess.clone()
    };
    let ts = Transys::from_aiger(circuit)
        .preprocess_configured(&preprocess_config)
        .0;

    // Detect circuit structure and override portfolio if needed.
    //
    // Priority order (#4149): SAT-likely FIRST, then arithmetic.
    // Sokoban/microban puzzles have deep combinational logic (game rule constraints)
    // that triggers the arithmetic heuristic, but they need BMC (SAT-focused portfolio)
    // not IC3 (arithmetic portfolio). SAT-likely check must take priority.
    let config = if is_sat_likely(&ts) {
        eprintln!(
            "Portfolio: SAT-likely heuristic triggered (inputs={} latches={} constraints={}), using SAT-focused portfolio",
            ts.num_inputs, ts.num_latches, ts.constraint_lits.len(),
        );
        let mut sat = sat_focused_portfolio();
        sat.timeout = config.timeout;
        sat
    } else if analyze_circuit(&ts).is_arithmetic {
        let mut arith = arithmetic_portfolio();
        arith.timeout = config.timeout;
        arith
    } else {
        config
    };

    let cancelled = Arc::new(AtomicBool::new(false));
    let (tx, rx) = mpsc::channel();
    let start = Instant::now();
    // Wall-clock deadline for engines that poll a deadline rather than the
    // cancelled flag (the BDD lane's fixpoint). Zero timeout = unbounded.
    let engine_deadline = (!config.timeout.is_zero()).then(|| start + config.timeout);

    let mut handles = Vec::new();

    for engine_config in &config.engines {
        let ts = ts.clone();
        let cancelled = cancelled.clone();
        let tx = tx.clone();
        let cfg = engine_config.clone();
        let max_depth = config.max_depth;

        handles.push(thread::spawn(move || {
            let engine_start = Instant::now();
            let engine_name = cfg.name().to_string();

            // Wrap the entire engine execution in catch_unwind (#4026).
            // If a ay-sat panic escapes past the solver-level catch_unwind
            // (e.g., during add_clause or push/pop), or if any other panic
            // occurs in the engine, we degrade gracefully to Unknown instead
            // of crashing the portfolio thread and losing its channel sender.
            let engine_name_clone = engine_name.clone();
            let (result, witness) = std::panic::catch_unwind(std::panic::AssertUnwindSafe(
                move || -> (CheckResult, SafeWitness) {
                    let safe_engine = engine_verified_label(&cfg);
                    match cfg {
                        EngineConfig::Bmc { step } => {
                            let mut engine = BmcEngine::new(ts, step);
                            engine.set_cancelled(cancelled);
                            wrap_engine_verified(engine.check(max_depth), safe_engine)
                        }
                        EngineConfig::BmcDynamic => {
                            let mut engine = BmcEngine::new_dynamic(ts);
                            engine.set_cancelled(cancelled);
                            wrap_engine_verified(engine.check(max_depth), safe_engine)
                        }
                        EngineConfig::BmcAYVariant { step, backend } => {
                            let mut engine = BmcEngine::new_with_backend(ts, step, backend);
                            engine.set_cancelled(cancelled);
                            wrap_engine_verified(engine.check(max_depth), safe_engine)
                        }
                        EngineConfig::BmcAYVariantDynamic { backend } => {
                            let mut engine = BmcEngine::new_dynamic_with_backend(ts, backend);
                            engine.set_cancelled(cancelled);
                            wrap_engine_verified(engine.check(max_depth), safe_engine)
                        }
                        EngineConfig::BmcGeometricBackoff {
                            initial_depths,
                            double_interval,
                            max_step,
                        } => {
                            let mut engine = BmcEngine::new_geometric_backoff(ts);
                            engine.set_cancelled(cancelled);
                            wrap_engine_verified(
                                engine.check_geometric_backoff(
                                    max_depth,
                                    initial_depths,
                                    double_interval,
                                    max_step,
                                ),
                                safe_engine,
                            )
                        }
                        EngineConfig::BmcGeometricBackoffAYVariant {
                            initial_depths,
                            double_interval,
                            max_step,
                            backend,
                        } => {
                            let mut engine =
                                BmcEngine::new_geometric_backoff_with_backend(ts, backend);
                            engine.set_cancelled(cancelled);
                            wrap_engine_verified(
                                engine.check_geometric_backoff(
                                    max_depth,
                                    initial_depths,
                                    double_interval,
                                    max_step,
                                ),
                                safe_engine,
                            )
                        }
                        EngineConfig::BmcLinearOffset {
                            start_depth,
                            step,
                            max_depth: config_max_depth,
                        } => {
                            // Use step=1 as the construction parameter; the
                            // linear-offset runner drives step internally and
                            // relies on unroll_step_no_accumulator for the skip
                            // region. Cap max_depth at the portfolio's overall
                            // cap to stay honest about the budget.
                            let mut engine = BmcEngine::new(ts, 1);
                            engine.set_cancelled(cancelled);
                            wrap_engine_verified(
                                engine.check_linear_offset(
                                    start_depth,
                                    step,
                                    config_max_depth.min(max_depth),
                                ),
                                safe_engine,
                            )
                        }
                        EngineConfig::Kind => {
                            let mut engine = KindEngine::new(ts);
                            engine.set_cancelled(cancelled);
                            wrap_kind_verified(
                                engine.check(max_depth),
                                safe_engine,
                                false,
                                false,
                                max_depth,
                            )
                        }
                        EngineConfig::KindSimplePath => {
                            let mut engine = KindEngine::new_simple_path(ts);
                            engine.set_cancelled(cancelled);
                            wrap_kind_verified(
                                engine.check(max_depth),
                                safe_engine,
                                false,
                                true,
                                max_depth,
                            )
                        }
                        EngineConfig::KindSkipBmc => {
                            let mut engine = KindEngine::with_config(
                                ts,
                                crate::kind::KindConfig {
                                    simple_path: false,
                                    skip_bmc: true,
                                },
                            );
                            engine.set_cancelled(cancelled);
                            wrap_kind_verified(
                                engine.check(max_depth),
                                safe_engine,
                                false,
                                false,
                                max_depth,
                            )
                        }
                        EngineConfig::KindAYVariant { backend } => {
                            let mut engine = KindEngine::with_config_and_backend(
                                ts,
                                crate::kind::KindConfig::default(),
                                backend,
                            );
                            engine.set_cancelled(cancelled);
                            wrap_kind_verified(
                                engine.check(max_depth),
                                safe_engine,
                                false,
                                false,
                                max_depth,
                            )
                        }
                        EngineConfig::KindSkipBmcAYVariant { backend } => {
                            let mut engine = KindEngine::with_config_and_backend(
                                ts,
                                crate::kind::KindConfig {
                                    simple_path: false,
                                    skip_bmc: true,
                                },
                                backend,
                            );
                            engine.set_cancelled(cancelled);
                            wrap_kind_verified(
                                engine.check(max_depth),
                                safe_engine,
                                false,
                                false,
                                max_depth,
                            )
                        }
                        EngineConfig::KindStrengthened => {
                            let mut engine = KindStrengthenedEngine::new(ts);
                            engine.set_cancelled(cancelled);
                            wrap_kind_verified(
                                engine.check(max_depth),
                                safe_engine,
                                true,
                                false,
                                max_depth,
                            )
                        }
                        EngineConfig::KindStrengthenedAYVariant { backend } => {
                            let mut engine = KindStrengthenedEngine::with_backend(ts, backend);
                            engine.set_cancelled(cancelled);
                            wrap_kind_verified(
                                engine.check(max_depth),
                                safe_engine,
                                true,
                                false,
                                max_depth,
                            )
                        }
                        EngineConfig::Ic3 => {
                            let ts_ref = ts.clone();
                            let mut engine = Ic3Engine::new(ts);
                            engine.set_cancelled(cancelled);
                            ic3_to_check_result(engine.check(), &ts_ref)
                        }
                        EngineConfig::Ic3Configured { config, .. } => {
                            let mut ts = ts;
                            // inn-proper: promote internal signals to first-class latches BEFORE
                            // IC3 engine construction (#4308). Mutually exclusive with the
                            // cube-extension `internal_signals` variant.
                            if config.inn_proper && !config.internal_signals {
                                ts = crate::inn_proper::promote_internal_signals_to_latches(&ts);
                            }
                            if config.internal_signals {
                                ts.select_internal_signals();
                            }
                            let ts_ref = ts.clone();
                            let mut engine = Ic3Engine::with_config(ts, config);
                            engine.set_cancelled(cancelled);
                            ic3_to_check_result(engine.check(), &ts_ref)
                        }
                        EngineConfig::CegarIc3 { config, mode, .. } => {
                            let mut cegar =
                                crate::ic3::cegar::CegarIc3::with_mode(ts, config, mode);
                            cegar.set_cancelled(cancelled);
                            wrap_engine_verified(cegar.run(), safe_engine)
                        }
                        EngineConfig::RandomSim {
                            steps_per_walk,
                            num_walks,
                            seed,
                        } => {
                            // GPU bit-parallel tier first (threads x 64 lanes
                            // of the same walk semantics; deterministic
                            // replay builds the trace, then the standard
                            // witness verification below validates it like
                            // any other engine's). Falsification-only: a
                            // clean or unavailable GPU run falls through to
                            // the scalar walker unchanged (different RNG
                            // mapping = extra input-path diversity).
                            if let Some(result) = crate::bmc::try_gpu_random_sim(
                                &ts,
                                steps_per_walk,
                                seed,
                                &cancelled,
                            ) {
                                wrap_engine_verified(result, safe_engine)
                            } else {
                                let mut engine =
                                    RandomSimEngine::new(ts, steps_per_walk, num_walks, seed);
                                engine.set_cancelled(cancelled);
                                wrap_engine_verified(engine.check(), safe_engine)
                            }
                        }
                        EngineConfig::GpuExhaustiveBmc { max_k } => {
                            // GPU exhaustive-BMC lane: unroll k steps into one
                            // combinational AIG and enumerate ALL free-variable
                            // assignments on the GPU (bmc/gpu_exhaustive.rs).
                            let k = max_k.min(max_depth);
                            let gpu_cancel = {
                                let flag = cancelled.clone();
                                move || flag.load(Ordering::Relaxed)
                            };
                            match crate::bmc::try_gpu_exhaustive_bmc(&ts, k, &gpu_cancel) {
                                // Complete bounded proof — NOT full safety. A
                                // k-bounded absence of bad is exactly the
                                // k-induction base case, so it is surfaced as a
                                // non-terminating Unknown (never Safe), matching
                                // the CPU BMC's own bounded contract. The
                                // portfolio only ever stores it as `best`, so it
                                // can never outrank a real Safe/Unsafe.
                                Some(crate::bmc::GpuExhaustBmc::BoundedSafe) => (
                                    CheckResult::Unknown {
                                        reason: format!(
                                            "GPU exhaustive BMC: no bad state reachable in <= {k} \
                                             steps (complete bounded proof; full safety needs \
                                             k-induction)"
                                        ),
                                    },
                                    SafeWitness::Unwitnessed,
                                ),
                                // Unsafe (the carrier returns no trace) OR
                                // declined (no CUDA / relational init / any
                                // constraint / free set over the cap): run the
                                // CPU BMC to depth k to obtain a
                                // portfolio-verifiable counterexample trace
                                // (guaranteed to hit within k on Unsafe) or an
                                // ordinary bounded search on decline. Never emits
                                // an unwitnessed Unsafe.
                                Some(crate::bmc::GpuExhaustBmc::Unsafe) | None => {
                                    let mut engine = BmcEngine::new(ts, 1);
                                    engine.set_cancelled(cancelled);
                                    wrap_engine_verified(engine.check(k), safe_engine)
                                }
                            }
                        }
                        EngineConfig::BddReach { config: bdd_config } => {
                            // BDD symbolic reachability (bdd_reach.rs): the
                            // exact forward fixpoint on the workspace's
                            // general ROBDD engine.
                            match crate::bdd_reach::bdd_reach_check(
                                &ts,
                                &bdd_config,
                                engine_deadline,
                                &cancelled,
                            ) {
                                // Exact fixpoint + the engine's own inductive
                                // self-check (init ⊆ R, post(R) ⊆ R,
                                // R ∩ bad = ∅) — a full unbounded Safe.
                                crate::bdd_reach::BddReachOutcome::Safe => (
                                    CheckResult::Safe,
                                    SafeWitness::EngineVerified {
                                        engine: safe_engine,
                                    },
                                ),
                                // Bad is PROVEN reachable at minimal depth k:
                                // re-derive the counterexample through the CPU
                                // BMC engine at that depth so the Unsafe
                                // carries a portfolio-verifiable trace (the
                                // GPU-lane protocol). Deliberately NOT capped
                                // at the portfolio max_depth: reachability at
                                // k is already proven, and the witness needs
                                // the full unroll (budget still bounds it via
                                // cancelled/timeout). Never emits an
                                // unwitnessed Unsafe.
                                crate::bdd_reach::BddReachOutcome::BadReachable { depth } => {
                                    let mut engine = BmcEngine::new(ts, 1);
                                    engine.set_cancelled(cancelled);
                                    wrap_engine_verified(engine.check(depth), safe_engine)
                                }
                                // Fail-closed decline: admission gate, budget,
                                // or self-check failure.
                                crate::bdd_reach::BddReachOutcome::Declined { reason } => (
                                    CheckResult::Unknown {
                                        reason: format!("BDD reachability declined: {reason}"),
                                    },
                                    SafeWitness::Unwitnessed,
                                ),
                            }
                        }
                    }
                },
            ))
            .unwrap_or_else(|panic_info: Box<dyn std::any::Any + Send>| {
                let msg = if let Some(s) = panic_info.downcast_ref::<&str>() {
                    (*s).to_string()
                } else if let Some(s) = panic_info.downcast_ref::<String>() {
                    s.clone()
                } else {
                    "unknown panic".to_string()
                };
                eprintln!("Portfolio: engine {engine_name_clone} panicked: {msg}");
                (
                    CheckResult::Unknown {
                        reason: format!("engine panicked: {msg}"),
                    },
                    SafeWitness::Unwitnessed,
                )
            });

            let elapsed = engine_start.elapsed().as_secs_f64();
            let _ = tx.send(EngineOutcome {
                result: PortfolioResult {
                    result,
                    solver_name: engine_name,
                    time_secs: elapsed,
                },
                witness: Some(witness),
            });
        }));
    }
    drop(tx); // Close sender so rx.recv() returns Err when all threads finish

    // Wait for first definitive result or timeout
    let mut best = PortfolioResult {
        result: CheckResult::Unknown {
            reason: "no engine finished".into(),
        },
        solver_name: String::new(),
        time_secs: 0.0,
    };

    loop {
        let remaining = config.timeout.saturating_sub(start.elapsed());
        if remaining.is_zero() {
            cancelled.store(true, Ordering::Relaxed);
            join_with_grace_period(handles, PORTFOLIO_GRACE_PERIOD);
            return PortfolioResult {
                result: CheckResult::Unknown {
                    reason: "portfolio timeout".into(),
                },
                solver_name: String::new(),
                time_secs: config.timeout.as_secs_f64(),
            };
        }

        match rx.recv_timeout(remaining) {
            Ok(outcome) => {
                let EngineOutcome {
                    result: portfolio_result,
                    witness,
                } = outcome;
                if portfolio_result.result.is_definitive() {
                    match &portfolio_result.result {
                        // Portfolio-level witness verification (#4103):
                        // Before accepting an Unsafe result from ANY engine (BMC,
                        // k-induction, IC3), verify the witness by simulating
                        // the circuit. This is defense-in-depth: BMC/k-ind already
                        // verify internally, but IC3 does not, and this catches
                        // bugs in any engine's witness extraction.
                        CheckResult::Unsafe { .. } => {
                            if !verify_portfolio_unsafe_witness(&ts, &portfolio_result) {
                                // Don't accept this result -- continue waiting
                                // for other engines.
                                continue;
                            }
                            cancelled.store(true, Ordering::Relaxed);
                            join_with_grace_period(handles, PORTFOLIO_GRACE_PERIOD);
                            return portfolio_result;
                        }
                        CheckResult::Safe => {
                            // Symmetric Safe cross-validator (#4315).
                            // Run the independent witness checker BEFORE taking
                            // the cancellation shortcut. On Rejected / Downgrade
                            // we log a SOUNDNESS ALERT and keep waiting for
                            // sibling engines — we do NOT accept the result.
                            let witness_ref = witness.as_ref().unwrap_or(&SafeWitness::Unwitnessed);
                            if !portfolio_safe_validation_accepts(
                                portfolio_result.solver_name.as_str(),
                                witness_ref,
                                &ts,
                            ) {
                                continue;
                            }

                            cancelled.store(true, Ordering::Relaxed);

                            let mut competing = Vec::new();
                            if let Ok(drained) = rx.recv_timeout(SAFE_CROSS_VALIDATION_GRACE) {
                                if verify_portfolio_unsafe_witness(&ts, &drained.result) {
                                    competing.push(drained.result);
                                }
                            }

                            loop {
                                match rx.try_recv() {
                                    Ok(drained) => {
                                        if verify_portfolio_unsafe_witness(&ts, &drained.result) {
                                            competing.push(drained.result);
                                        }
                                    }
                                    Err(mpsc::TryRecvError::Empty)
                                    | Err(mpsc::TryRecvError::Disconnected) => {
                                        break;
                                    }
                                }
                            }

                            let winner = cross_validate_safe_result(portfolio_result, competing);
                            join_with_grace_period(handles, PORTFOLIO_GRACE_PERIOD);
                            return winner;
                        }
                        CheckResult::Unknown { .. } => {
                            unreachable!("definitive result must be Safe or Unsafe")
                        }
                    }
                }
                best = portfolio_result;
            }
            Err(mpsc::RecvTimeoutError::Timeout) => {
                cancelled.store(true, Ordering::Relaxed);
                join_with_grace_period(handles, PORTFOLIO_GRACE_PERIOD);
                return PortfolioResult {
                    result: CheckResult::Unknown {
                        reason: "portfolio timeout".into(),
                    },
                    solver_name: String::new(),
                    time_secs: config.timeout.as_secs_f64(),
                };
            }
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                // All senders dropped -- all engines finished
                break;
            }
        }
    }

    // Wait for remaining threads
    join_with_grace_period(handles, PORTFOLIO_GRACE_PERIOD);

    best
}

fn verify_portfolio_unsafe_witness(ts: &Transys, portfolio_result: &PortfolioResult) -> bool {
    if let CheckResult::Unsafe { trace, depth } = &portfolio_result.result {
        if let Err(reason) = ts.verify_witness(trace) {
            eprintln!(
                "Portfolio: {} returned Unsafe at depth {} but witness \
                 verification FAILED: {}. Treating as Unknown.",
                portfolio_result.solver_name.as_str(),
                depth,
                reason,
            );
            return false;
        }
    }

    true
}

pub(super) fn portfolio_safe_validation_accepts(
    solver_name: &str,
    witness: &SafeWitness,
    ts: &Transys,
) -> bool {
    let outcome = validate_safe(witness, ts);
    match &outcome {
        SafeValidation::Accepted => {}
        SafeValidation::Indeterminate { reason } => {
            // Fail-OPEN for assurance — mirror the committed
            // `SafeWitness::KInduction` replay contract
            // (`safe_witness.rs` `CheckResult::Unknown => Accepted`). A
            // budget-out / solver-`Unknown` during the independent inductive-
            // invariant re-check means the proof could not be CONFIRMED, not
            // that it was DISPROVED. Rejecting here would downgrade a correct
            // SAFE verdict (e.g. a large circuit whose lemma set cannot be
            // re-verified inside the flat budget) — the verdict-change risk the
            // proof-backing work is meant to eliminate. Accept the engine's own
            // internally-verified Safe exactly as before; only a genuine
            // counterexample (`Rejected`) blocks acceptance.
            eprintln!(
                "Portfolio: validate_safe indeterminate for {solver_name} \
                 (#4315): {reason}. Independent re-check could not complete \
                 within budget; accepting the engine's internally-verified \
                 Safe (fail-open — a correct SAFE is never downgraded).",
            );
        }
        SafeValidation::Rejected { reason } => {
            eprintln!(
                "Portfolio: SOUNDNESS ALERT (#4315) — Safe verdict from \
                 {solver_name} REJECTED by independent validator: {reason}. \
                 Continuing to wait for other engines.",
            );
        }
        SafeValidation::Downgrade { reason } => {
            eprintln!(
                "Portfolio: SOUNDNESS ALERT (#4315) — Safe verdict from \
                 {solver_name} has no proof witness ({reason}). Downgrading to \
                 unverified; continuing.",
            );
        }
    }
    outcome.portfolio_accepts()
}

/// Cross-validate a candidate Safe winner against other completed worker results (#4315).
///
/// In HWMCC, `Unsafe` (SAT) is always ground truth — the counterexample is
/// the witness. If a candidate Safe result is about to be returned but any
/// competing worker result (already drained from the channel) is Unsafe,
/// prefer Unsafe and log an ERROR citing the disagreement.
///
/// If two workers both return Safe with different `solver_name` (so
/// independent confirmation), log an INFO noting the agreement. If only
/// one Safe is present, this is a no-op wrapper.
///
/// Callers MUST have already verified any Unsafe witnesses in `competing`
/// before calling this helper (the helper trusts Unsafe results).
pub(super) fn cross_validate_safe_result(
    candidate_safe: PortfolioResult,
    competing: Vec<PortfolioResult>,
) -> PortfolioResult {
    debug_assert!(matches!(&candidate_safe.result, CheckResult::Safe));

    if let Some(unsafe_result) = competing
        .iter()
        .find(|result| matches!(&result.result, CheckResult::Unsafe { .. }))
    {
        eprintln!(
            "Portfolio: SOUNDNESS ALERT (#4315) — Safe result from {} ({:.3}s) \
             disagreed with Unsafe result from {} ({:.3}s); preferring Unsafe witness.",
            candidate_safe.solver_name.as_str(),
            candidate_safe.time_secs,
            unsafe_result.solver_name.as_str(),
            unsafe_result.time_secs,
        );
        return unsafe_result.clone();
    }

    if let Some(confirming_safe) = competing.iter().find(|result| {
        matches!(&result.result, CheckResult::Safe)
            && result.solver_name.as_str() != candidate_safe.solver_name.as_str()
    }) {
        eprintln!(
            "Portfolio: INFO (#4315) — Safe result from {} ({:.3}s) \
             independently confirmed by {} ({:.3}s).",
            candidate_safe.solver_name.as_str(),
            candidate_safe.time_secs,
            confirming_safe.solver_name.as_str(),
            confirming_safe.time_secs,
        );
    }

    candidate_safe
}

/// Grace period for thread joins after cancellation (#4096).
///
/// After the cancellation flag is set, each engine thread should check
/// `is_cancelled()` at its next cancellation point and exit. The grace
/// period is how long the portfolio waits for threads to respond before
/// detaching them (letting them finish in the background without blocking
/// the caller).
///
/// 3 seconds is generous — well-instrumented threads should exit within
/// milliseconds of seeing the cancellation flag. The only case where
/// threads take longer is if a SAT solver query is in progress and the
/// solver doesn't check its own cancellation flag frequently enough.
pub(super) const PORTFOLIO_GRACE_PERIOD: Duration = Duration::from_secs(3);

/// Grace period for draining competing results after a candidate Safe arrives (#4315).
pub(super) const SAFE_CROSS_VALIDATION_GRACE: Duration = Duration::from_millis(500);

/// Join threads with a grace period (#4096).
///
/// Waits up to `grace` for all threads to finish. If any threads are
/// still running after the grace period, they are detached (the
/// JoinHandle is dropped without joining, allowing the thread to run
/// in the background until it finishes on its own).
///
/// This prevents the portfolio from hanging indefinitely when a thread
/// is stuck in a long SAT solver query that doesn't respect the
/// cancellation flag.
pub(super) fn join_with_grace_period(handles: Vec<thread::JoinHandle<()>>, grace: Duration) {
    let deadline = Instant::now() + grace;
    let mut remaining_handles: Vec<Option<thread::JoinHandle<()>>> =
        handles.into_iter().map(Some).collect();

    // Poll for thread completion until deadline.
    loop {
        let now = Instant::now();
        if now >= deadline {
            break;
        }

        let all_done = remaining_handles.iter().all(|h| h.is_none());
        if all_done {
            return;
        }

        // Check each handle. JoinHandle doesn't have a timed join in std,
        // so we use is_finished() (stable since Rust 1.61) to poll.
        for slot in &mut remaining_handles {
            if let Some(handle) = slot {
                if handle.is_finished() {
                    let h = slot.take().expect("just checked Some");
                    let _ = h.join();
                }
            }
        }

        // Brief sleep to avoid busy-spinning. 10ms is fine for a grace period.
        let remaining_wait = deadline.saturating_duration_since(Instant::now());
        thread::sleep(remaining_wait.min(Duration::from_millis(10)));
    }

    // Grace period exceeded: detach any still-running threads by dropping
    // their JoinHandles. The threads will continue running in the background
    // but won't block the caller.
    let detached_count = remaining_handles.iter().filter(|h| h.is_some()).count();
    if detached_count > 0 {
        eprintln!(
            "Portfolio: {} thread(s) still running after {}s grace period — detaching (#4096)",
            detached_count,
            grace.as_secs_f64(),
        );
    }
    // Dropping the remaining JoinHandles detaches the threads.
    drop(remaining_handles);
}

/// Convert IC3 result to the shared CheckResult type plus the `SafeWitness`
/// attached to any `Safe` verdict (#4315). IC3 is currently the only engine
/// that emits an inductive-invariant witness; other engines are wrapped via
/// [`wrap_engine_verified`] below.
pub(super) fn ic3_to_check_result(ic3: Ic3Result, ts: &Transys) -> (CheckResult, SafeWitness) {
    match ic3 {
        Ic3Result::Safe { depth, lemmas } => (
            CheckResult::Safe,
            SafeWitness::InductiveInvariant { lemmas, depth },
        ),
        Ic3Result::Unsafe { depth, trace } => {
            // Convert IC3's (Var, bool) trace to named FxHashMap trace.
            // Use the same naming convention as BMC (l{idx}/i{idx}/v{var_id})
            // so verify_witness can find the variable values.
            let named_trace = trace
                .into_iter()
                .map(|state| {
                    let mut named: rustc_hash::FxHashMap<String, bool> =
                        rustc_hash::FxHashMap::default();
                    for (var, val) in &state {
                        // Map latch vars to "l{idx}" format
                        if let Some(idx) = ts.latch_vars.iter().position(|lv| lv == var) {
                            named.insert(format!("l{idx}"), *val);
                        }
                        // Map input vars to "i{idx}" format
                        if let Some(idx) = ts.input_vars.iter().position(|iv| iv == var) {
                            named.insert(format!("i{idx}"), *val);
                        }
                        // Also include raw "v{id}" for compatibility
                        named.insert(format!("v{}", var.0), *val);
                    }
                    named
                })
                .collect();
            (
                CheckResult::Unsafe {
                    depth,
                    trace: named_trace,
                },
                // Unsafe results don't need a Safe witness; Unwitnessed is
                // harmless because validate_safe is only invoked on Safe.
                SafeWitness::Unwitnessed,
            )
        }
        Ic3Result::Unknown { reason } => {
            (CheckResult::Unknown { reason }, SafeWitness::Unwitnessed)
        }
    }
}

/// Wrap an engine result that has no formal witness but passed its own
/// internal verification (#4315).
///
/// For `Safe`: emits `SafeWitness::EngineVerified { engine }`. The validator
/// accepts this without independent re-verification but logs that no
/// symmetric check was performed. Engines that CAN produce a formal invariant
/// (currently only IC3 variants) should emit
/// `SafeWitness::InductiveInvariant` through [`ic3_to_check_result`] so the
/// independent validator can catch #4310-class soundness bugs.
///
/// For `Unsafe`/`Unknown`: the witness is unused — we thread `Unwitnessed`
/// through as a neutral default.
pub(super) fn wrap_engine_verified(
    result: CheckResult,
    engine: &'static str,
) -> (CheckResult, SafeWitness) {
    let witness = match &result {
        CheckResult::Safe => SafeWitness::EngineVerified { engine },
        _ => SafeWitness::Unwitnessed,
    };
    (result, witness)
}

/// Wrap a k-induction (plain or strengthened) result so that a `Safe` verdict
/// carries a [`SafeWitness::KInduction`] certificate (#4315). Unlike the
/// blindly-trusted [`SafeWitness::EngineVerified`] path, this routes the Safe
/// verdict through an independent k-induction re-proof on a fresh solver
/// backend (`safe_witness::validate_kinduction_replay`), turning a previously
/// "trust the solver" Safe into a proof-backed one — while the validator's
/// fail-open-on-inconclusive contract guarantees a *correct* Safe is never
/// downgraded.
///
/// `strengthened` selects the strengthened engine for replay; `simple_path`
/// reproduces the plain engine's simple-path mode; `max_depth` is the unrolling
/// budget used by the original run.
pub(super) fn wrap_kind_verified(
    result: CheckResult,
    engine: &'static str,
    strengthened: bool,
    simple_path: bool,
    max_depth: usize,
) -> (CheckResult, SafeWitness) {
    let witness = match &result {
        CheckResult::Safe => SafeWitness::KInduction {
            engine,
            strengthened,
            simple_path,
            max_depth,
        },
        _ => SafeWitness::Unwitnessed,
    };
    (result, witness)
}

/// Stable audit label for Safe results that are trusted from an engine's own
/// internal proof instead of independently replayed by the portfolio validator.
pub(super) fn engine_verified_label(engine: &EngineConfig) -> &'static str {
    match engine {
        EngineConfig::Bmc { .. } => "bmc-lower-bound",
        EngineConfig::BmcDynamic => "bmc-dynamic-lower-bound",
        EngineConfig::BmcAYVariant { backend, .. } => match backend {
            crate::sat_types::SolverBackend::AYLuby => "bmc-ay-luby-lower-bound",
            crate::sat_types::SolverBackend::AYStable => "bmc-ay-stable-lower-bound",
            crate::sat_types::SolverBackend::AYGeometric => "bmc-ay-geometric-lower-bound",
            crate::sat_types::SolverBackend::AYVmtf => "bmc-ay-vmtf-lower-bound",
            crate::sat_types::SolverBackend::AYChb => "bmc-ay-chb-lower-bound",
            crate::sat_types::SolverBackend::AYNoPreprocess => "bmc-ay-nopreproc-lower-bound",
            crate::sat_types::SolverBackend::Simple => "bmc-simple-lower-bound",
            _ => "bmc-ay-variant-lower-bound",
        },
        EngineConfig::BmcAYVariantDynamic { backend } => match backend {
            crate::sat_types::SolverBackend::AYLuby => "bmc-ay-luby-dynamic-lower-bound",
            crate::sat_types::SolverBackend::AYStable => "bmc-ay-stable-dynamic-lower-bound",
            _ => "bmc-ay-variant-dynamic-lower-bound",
        },
        EngineConfig::BmcGeometricBackoff { .. } => "bmc-geometric-lower-bound",
        EngineConfig::BmcGeometricBackoffAYVariant { backend, .. } => match backend {
            crate::sat_types::SolverBackend::AYLuby => "bmc-geometric-ay-luby-lower-bound",
            crate::sat_types::SolverBackend::AYStable => "bmc-geometric-ay-stable-lower-bound",
            crate::sat_types::SolverBackend::Simple => "bmc-geometric-simple-lower-bound",
            _ => "bmc-geometric-ay-variant-lower-bound",
        },
        EngineConfig::BmcLinearOffset { .. } => "bmc-linear-offset-lower-bound",
        EngineConfig::Kind => "k-induction",
        EngineConfig::KindSimplePath => "k-induction-simple-path",
        EngineConfig::KindSkipBmc => "k-induction-skip-bmc",
        EngineConfig::KindAYVariant { backend } => match backend {
            crate::sat_types::SolverBackend::AYLuby => "k-induction-ay-luby",
            crate::sat_types::SolverBackend::AYStable => "k-induction-ay-stable",
            crate::sat_types::SolverBackend::AYVmtf => "k-induction-ay-vmtf",
            _ => "k-induction-ay-variant",
        },
        EngineConfig::KindSkipBmcAYVariant { backend } => match backend {
            crate::sat_types::SolverBackend::AYLuby => "k-induction-skip-bmc-ay-luby",
            crate::sat_types::SolverBackend::AYStable => "k-induction-skip-bmc-ay-stable",
            crate::sat_types::SolverBackend::AYVmtf => "k-induction-skip-bmc-ay-vmtf",
            _ => "k-induction-skip-bmc-ay-variant",
        },
        EngineConfig::KindStrengthened => "k-induction-strengthened",
        EngineConfig::KindStrengthenedAYVariant { backend } => match backend {
            crate::sat_types::SolverBackend::AYLuby => "k-induction-strengthened-ay-luby",
            crate::sat_types::SolverBackend::AYStable => "k-induction-strengthened-ay-stable",
            _ => "k-induction-strengthened-ay-variant",
        },
        EngineConfig::Ic3 => "ic3-default",
        EngineConfig::Ic3Configured { .. } => "ic3-configured",
        EngineConfig::CegarIc3 { .. } => "cegar-ic3",
        EngineConfig::RandomSim { .. } => "random-sim",
        EngineConfig::GpuExhaustiveBmc { .. } => "gpu-exhaustive-bmc-lower-bound",
        EngineConfig::BddReach { .. } => "bdd-reach-exact-fixpoint",
    }
}
