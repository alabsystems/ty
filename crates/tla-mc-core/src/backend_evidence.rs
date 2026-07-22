// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Shared evidence rendering helpers for backend capability decisions.
//!
//! Capability reports carry structured data. This module provides the stable
//! string vocabulary used by operational evidence while backend-specific
//! consumers migrate away from local formatting helpers.

use crate::backend_capability::{
    BackendCapability, BackendDomain, BackendKind, CapabilityReport, CapabilityRole,
    CapabilityStatus, ProblemKind, ProductionRoutingStatus, SolverFacet,
    SymbolicExecutionDetection, SymbolicExecutionReason, SymbolicExecutionStatus,
};

/// Reason-code text used when a lane has no unsupported reason.
pub const NO_REASON_CODE: &str = "none";

/// Whether a capability lane was selected or rejected by a routing decision.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CapabilityLaneDecision {
    /// Lane was selected for the report.
    Selected,
    /// Lane was rejected for the report.
    Rejected,
}

impl CapabilityLaneDecision {
    /// Stable action token used in evidence text.
    pub fn action(self) -> &'static str {
        match self {
            Self::Selected => "selected",
            Self::Rejected => "rejected",
        }
    }

    /// Stable evidence key used in human-readable capability rows.
    pub fn evidence_key(self) -> &'static str {
        match self {
            Self::Selected => "selected_lane",
            Self::Rejected => "rejected_lane",
        }
    }
}

impl BackendCapability {
    /// Stable snake_case problem code, when this lane is tied to one problem.
    pub fn problem_code(&self) -> Option<&'static str> {
        optional_problem_code(self.problem)
    }

    /// Stable display problem name, when this lane is tied to one problem.
    pub fn problem_name(&self) -> Option<&'static str> {
        optional_problem_name(self.problem)
    }

    /// Stable display problem name, including the shared no-problem sentinel.
    pub fn problem_name_or_none(&self) -> &'static str {
        problem_name_or_none(self.problem)
    }

    /// Stable reason code, including the shared no-reason sentinel.
    pub fn normalized_reason_code(&self) -> &'static str {
        self.reason_code().unwrap_or(NO_REASON_CODE)
    }

    /// Render this capability as one selected/rejected lane evidence line.
    pub fn render_lane_evidence(&self, scope: &str, decision: CapabilityLaneDecision) -> String {
        render_capability_lane_evidence(scope, decision, self)
    }

    /// Render this capability as one machine-readable lane-status evidence line.
    pub fn render_lane_status_evidence(
        &self,
        scope: &str,
        decision: CapabilityLaneDecision,
    ) -> String {
        render_capability_lane_status_evidence(scope, decision, self)
    }
}

impl CapabilityReport {
    /// Stable snake_case problem code, when this report is tied to one problem.
    pub fn problem_code(&self) -> Option<&'static str> {
        optional_problem_code(self.problem)
    }

    /// Stable display problem name, when this report is tied to one problem.
    pub fn problem_name(&self) -> Option<&'static str> {
        optional_problem_name(self.problem)
    }

    /// Stable routing-status name for this report.
    pub fn production_routing_status_name(&self) -> &'static str {
        self.production_routing_status().name()
    }

    /// Stable routing-status code for this report.
    pub fn production_routing_status_code(&self) -> &'static str {
        self.production_routing_status().code()
    }

    /// Render this report's production routing status as one evidence line.
    pub fn render_production_routing_status_evidence(&self, scope: &str) -> String {
        format!(
            "{} production_routing_status={}",
            scope,
            self.production_routing_status_name()
        )
    }
}

impl SymbolicExecutionDetection {
    /// Stable snake_case status code for JSON and table evidence.
    pub fn status_code(self) -> &'static str {
        self.status.code()
    }

    /// Stable display status name matching Rust enum naming.
    pub fn status_name(self) -> &'static str {
        self.status.name()
    }

    /// Stable snake_case reason code, when symbolic execution was detected.
    pub fn reason_code(self) -> Option<&'static str> {
        self.reason.map(SymbolicExecutionReason::code)
    }

    /// Stable display reason name, when symbolic execution was detected.
    pub fn reason_name(self) -> Option<&'static str> {
        self.reason.map(SymbolicExecutionReason::name)
    }

    /// Stable display reason name, including the shared no-reason sentinel.
    pub fn reason_name_or_none(self) -> &'static str {
        self.reason_name().unwrap_or("None")
    }

    /// Stable reason code, including the shared no-reason sentinel.
    pub fn normalized_reason_code(self) -> &'static str {
        self.reason_code().unwrap_or(NO_REASON_CODE)
    }

    /// Render this detection as one shared evidence line.
    pub fn render_evidence(self, scope: &str, problem: ProblemKind) -> String {
        render_symbolic_execution_detection_evidence(scope, problem, self)
    }
}

impl BackendDomain {
    /// Stable snake_case code for JSON and table evidence.
    pub fn code(self) -> &'static str {
        match self {
            Self::Shared => "shared",
            Self::Tla => "tla",
            Self::PetriMcc => "petri_mcc",
            Self::Aiger => "aiger",
            Self::Btor2 => "btor2",
            Self::TrustIr => "trust_ir",
            Self::TrustCg => "trust-cg",
            Self::AY => "ay",
        }
    }

    /// Stable display name matching the existing capability evidence style.
    pub fn name(self) -> &'static str {
        match self {
            Self::Shared => "Shared",
            Self::Tla => "Tla",
            Self::PetriMcc => "PetriMcc",
            Self::Aiger => "Aiger",
            Self::Btor2 => "Btor2",
            Self::TrustIr => "TrustIr",
            Self::TrustCg => "TrustCg",
            Self::AY => "AY",
        }
    }
}

impl BackendKind {
    /// Stable snake_case backend code for JSON and table evidence.
    pub fn code(self) -> &'static str {
        match self {
            Self::ExplicitState => "explicit_state",
            Self::NativeKernel => "native_kernel",
            Self::LocalSymbolicExecution => "local_symbolic_execution",
            Self::ExternalAYBinary => "external_ay_binary",
            Self::AYSmt => "ay_smt",
            Self::AYSat => "ay_sat",
            Self::AYChc => "ay_chc",
            Self::AigerPortfolio => "aiger_portfolio",
            Self::Btor2Portfolio => "btor2_portfolio",
        }
    }

    /// Stable display name matching the existing capability evidence style.
    pub fn name(self) -> &'static str {
        match self {
            Self::ExplicitState => "ExplicitState",
            Self::NativeKernel => "NativeKernel",
            Self::LocalSymbolicExecution => "LocalSymbolicExecution",
            Self::ExternalAYBinary => "ExternalAYBinary",
            Self::AYSmt => "AYSmt",
            Self::AYSat => "AYSat",
            Self::AYChc => "AYChc",
            Self::AigerPortfolio => "AigerPortfolio",
            Self::Btor2Portfolio => "Btor2Portfolio",
        }
    }
}

impl ProblemKind {
    /// Stable snake_case problem code for JSON and table evidence.
    pub fn code(self) -> &'static str {
        match self {
            Self::ExplicitReachability => "explicit_reachability",
            Self::Safety => "safety",
            Self::Liveness => "liveness",
            Self::Deadlock => "deadlock",
            Self::StateSpace => "state_space",
            Self::SymbolicExecution => "symbolic_execution",
            Self::Invariant => "invariant",
            Self::Bmc => "bmc",
            Self::KInduction => "k_induction",
            Self::Chc => "chc",
            Self::Sat => "sat",
            Self::Smt => "smt",
            Self::NativeSuccessor => "native_successor",
        }
    }

    /// Stable display name matching the existing capability evidence style.
    pub fn name(self) -> &'static str {
        match self {
            Self::ExplicitReachability => "ExplicitReachability",
            Self::Safety => "Safety",
            Self::Liveness => "Liveness",
            Self::Deadlock => "Deadlock",
            Self::StateSpace => "StateSpace",
            Self::SymbolicExecution => "SymbolicExecution",
            Self::Invariant => "Invariant",
            Self::Bmc => "Bmc",
            Self::KInduction => "KInduction",
            Self::Chc => "Chc",
            Self::Sat => "Sat",
            Self::Smt => "Smt",
            Self::NativeSuccessor => "NativeSuccessor",
        }
    }
}

impl SolverFacet {
    /// Stable snake_case facet code for JSON and table evidence.
    pub fn code(self) -> &'static str {
        match self {
            Self::ExternalProcess => "external_process",
            Self::InProcess => "in_process",
            Self::Sat => "sat",
            Self::Smt => "smt",
            Self::SymbolicExecution => "symbolic_execution",
            Self::Chc => "chc",
            Self::Bmc => "bmc",
            Self::KInduction => "k_induction",
            Self::Pdr => "pdr",
            Self::AllSat => "all_sat",
            Self::Incremental => "incremental",
            Self::Assumptions => "assumptions",
            Self::UnsatCore => "unsat_core",
            Self::ModelValues => "model_values",
            Self::Cancellation => "cancellation",
            Self::Proof => "proof",
            Self::Witness => "witness",
            Self::BitVector => "bit_vector",
            Self::LinearIntegerArithmetic => "linear_integer_arithmetic",
            Self::NativeCodegen => "native_codegen",
        }
    }

    /// Stable display name matching Rust enum naming.
    pub fn name(self) -> &'static str {
        match self {
            Self::ExternalProcess => "ExternalProcess",
            Self::InProcess => "InProcess",
            Self::Sat => "Sat",
            Self::Smt => "Smt",
            Self::SymbolicExecution => "SymbolicExecution",
            Self::Chc => "Chc",
            Self::Bmc => "Bmc",
            Self::KInduction => "KInduction",
            Self::Pdr => "Pdr",
            Self::AllSat => "AllSat",
            Self::Incremental => "Incremental",
            Self::Assumptions => "Assumptions",
            Self::UnsatCore => "UnsatCore",
            Self::ModelValues => "ModelValues",
            Self::Cancellation => "Cancellation",
            Self::Proof => "Proof",
            Self::Witness => "Witness",
            Self::BitVector => "BitVector",
            Self::LinearIntegerArithmetic => "LinearIntegerArithmetic",
            Self::NativeCodegen => "NativeCodegen",
        }
    }
}

impl CapabilityRole {
    /// Stable snake_case role code for JSON and table evidence.
    pub fn code(self) -> &'static str {
        match self {
            Self::Production => "production",
            Self::Fallback => "fallback",
            Self::Validation => "validation",
            Self::TestOnly => "test_only",
        }
    }

    /// Stable display name matching the existing capability evidence style.
    pub fn name(self) -> &'static str {
        match self {
            Self::Production => "Production",
            Self::Fallback => "Fallback",
            Self::Validation => "Validation",
            Self::TestOnly => "TestOnly",
        }
    }
}

impl CapabilityStatus {
    /// Stable snake_case status code for JSON and table evidence.
    pub fn code(self) -> &'static str {
        match self {
            Self::Available => "available",
            Self::Unavailable => "unavailable",
            Self::Unsupported => "unsupported",
            Self::Disabled => "disabled",
            Self::Experimental => "experimental",
        }
    }

    /// Stable display name matching the existing capability evidence style.
    pub fn name(self) -> &'static str {
        match self {
            Self::Available => "Available",
            Self::Unavailable => "Unavailable",
            Self::Unsupported => "Unsupported",
            Self::Disabled => "Disabled",
            Self::Experimental => "Experimental",
        }
    }
}

impl ProductionRoutingStatus {
    /// Stable snake_case routing-status code for JSON and table evidence.
    pub fn code(self) -> &'static str {
        match self {
            Self::AYFirst => "ay_first",
            Self::JustifiedLocalFallback => "justified_local_fallback",
            Self::UnjustifiedLocalFallback => "unjustified_local_fallback",
            Self::OtherProduction => "other_production",
            Self::NoProductionSelection => "no_production_selection",
        }
    }

    /// Stable display name matching the current evidence payloads.
    pub fn name(self) -> &'static str {
        match self {
            Self::AYFirst => "AYFirst",
            Self::JustifiedLocalFallback => "JustifiedLocalFallback",
            Self::UnjustifiedLocalFallback => "UnjustifiedLocalFallback",
            Self::OtherProduction => "OtherProduction",
            Self::NoProductionSelection => "NoProductionSelection",
        }
    }
}

impl SymbolicExecutionStatus {
    /// Stable snake_case symbolic-execution status code.
    pub fn code(self) -> &'static str {
        match self {
            Self::NotDetected => "not_detected",
            Self::AYPreferred => "ay_preferred",
            Self::AYRequired => "ay_required",
            Self::LocalFallbackAfterAYRejection => "local_fallback_after_ay_rejection",
        }
    }

    /// Stable display name matching Rust enum naming.
    pub fn name(self) -> &'static str {
        match self {
            Self::NotDetected => "NotDetected",
            Self::AYPreferred => "AYPreferred",
            Self::AYRequired => "AYRequired",
            Self::LocalFallbackAfterAYRejection => "LocalFallbackAfterAYRejection",
        }
    }
}

impl SymbolicExecutionReason {
    /// Stable snake_case symbolic-execution reason code.
    pub fn code(self) -> &'static str {
        match self {
            Self::SymbolicInitialState => "symbolic_initial_state",
            Self::SymbolicTransitionRelation => "symbolic_transition_relation",
            Self::StateSpaceExplosion => "state_space_explosion",
            Self::UnsupportedLocalFragment => "unsupported_local_fragment",
            Self::BitVectorFormula => "bit_vector_formula",
            Self::ModelEnumeration => "model_enumeration",
            Self::NativeKernelUnsupported => "native_kernel_unsupported",
        }
    }

    /// Stable display name matching Rust enum naming.
    pub fn name(self) -> &'static str {
        match self {
            Self::SymbolicInitialState => "SymbolicInitialState",
            Self::SymbolicTransitionRelation => "SymbolicTransitionRelation",
            Self::StateSpaceExplosion => "StateSpaceExplosion",
            Self::UnsupportedLocalFragment => "UnsupportedLocalFragment",
            Self::BitVectorFormula => "BitVectorFormula",
            Self::ModelEnumeration => "ModelEnumeration",
            Self::NativeKernelUnsupported => "NativeKernelUnsupported",
        }
    }
}

/// Render a selected/rejected backend lane using the shared evidence format.
pub fn render_capability_lane_evidence(
    scope: &str,
    decision: CapabilityLaneDecision,
    capability: &BackendCapability,
) -> String {
    format!(
        "{} {} backend={} role={} problem={} status={} reason_code={}",
        scope,
        decision.evidence_key(),
        capability.backend.name(),
        capability.role.name(),
        capability_problem_name(capability.problem),
        capability.status.name(),
        capability.normalized_reason_code()
    )
}

/// Render a selected/rejected backend lane with stable machine-code columns.
pub fn render_capability_lane_status_evidence(
    scope: &str,
    decision: CapabilityLaneDecision,
    capability: &BackendCapability,
) -> String {
    format!(
        "{} shared_lane lane_status={} backend={} backend_code={} backend_role={} problem={} capability_status={} reason_code={}",
        scope,
        decision.action(),
        capability.backend.name(),
        capability.backend.code(),
        capability.role.code(),
        capability.problem_name_or_none(),
        capability.status.code(),
        capability.normalized_reason_code()
    )
}

/// Render a symbolic-execution detection with stable machine-code columns.
pub fn render_symbolic_execution_detection_evidence(
    scope: &str,
    problem: ProblemKind,
    detection: SymbolicExecutionDetection,
) -> String {
    let domain_code = symbolic_execution_scope_domain_code(scope);
    let preferred_backend = detection.preferred_ay_backend(problem);
    let preferred_backend_name = preferred_backend.map(BackendKind::name).unwrap_or("None");
    let preferred_backend_code = preferred_backend
        .map(BackendKind::code)
        .unwrap_or(NO_REASON_CODE);
    format!(
        "{} symbolic_execution domain={} status={} status_code={} problem={} reason={} reason_code={} preferred_backend={} preferred_backend_code={}",
        scope,
        domain_code,
        detection.status_name(),
        detection.status_code(),
        problem.name(),
        detection.reason_name_or_none(),
        detection.normalized_reason_code(),
        preferred_backend_name,
        preferred_backend_code
    )
}

fn symbolic_execution_scope_domain_code(scope: &str) -> &'static str {
    match scope {
        "AIGER" => BackendDomain::Aiger.code(),
        "BTOR2" => BackendDomain::Btor2.code(),
        "MCC" => BackendDomain::PetriMcc.code(),
        "TLA" => BackendDomain::Tla.code(),
        "AY" => BackendDomain::AY.code(),
        _ => BackendDomain::Shared.code(),
    }
}

fn capability_problem_name(problem: Option<ProblemKind>) -> &'static str {
    problem_name_or_none(problem)
}

fn optional_problem_code(problem: Option<ProblemKind>) -> Option<&'static str> {
    problem.map(ProblemKind::code)
}

fn optional_problem_name(problem: Option<ProblemKind>) -> Option<&'static str> {
    problem.map(ProblemKind::name)
}

fn problem_name_or_none(problem: Option<ProblemKind>) -> &'static str {
    optional_problem_name(problem).unwrap_or("None")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend_capability::{
        ay_chc_capability, ay_smt_capability, SolverLimits, UnsupportedReason,
    };

    #[test]
    fn lane_decision_tokens_are_stable() {
        assert_eq!(CapabilityLaneDecision::Selected.action(), "selected");
        assert_eq!(
            CapabilityLaneDecision::Selected.evidence_key(),
            "selected_lane"
        );
        assert_eq!(CapabilityLaneDecision::Rejected.action(), "rejected");
        assert_eq!(
            CapabilityLaneDecision::Rejected.evidence_key(),
            "rejected_lane"
        );
    }

    #[test]
    fn enum_codes_and_names_are_stable_for_backend_evidence() {
        assert_eq!(BackendDomain::PetriMcc.code(), "petri_mcc");
        assert_eq!(BackendDomain::PetriMcc.name(), "PetriMcc");
        assert_eq!(BackendKind::ExternalAYBinary.code(), "external_ay_binary");
        assert_eq!(BackendKind::ExternalAYBinary.name(), "ExternalAYBinary");
        assert_eq!(
            BackendKind::LocalSymbolicExecution.code(),
            "local_symbolic_execution"
        );
        assert_eq!(
            BackendKind::LocalSymbolicExecution.name(),
            "LocalSymbolicExecution"
        );
        assert_eq!(ProblemKind::KInduction.code(), "k_induction");
        assert_eq!(ProblemKind::KInduction.name(), "KInduction");
        assert_eq!(ProblemKind::SymbolicExecution.code(), "symbolic_execution");
        assert_eq!(ProblemKind::SymbolicExecution.name(), "SymbolicExecution");
        assert_eq!(SolverFacet::NativeCodegen.code(), "native_codegen");
        assert_eq!(SolverFacet::NativeCodegen.name(), "NativeCodegen");
        assert_eq!(SolverFacet::SymbolicExecution.code(), "symbolic_execution");
        assert_eq!(SolverFacet::SymbolicExecution.name(), "SymbolicExecution");
        assert_eq!(CapabilityRole::TestOnly.code(), "test_only");
        assert_eq!(CapabilityRole::TestOnly.name(), "TestOnly");
        assert_eq!(CapabilityStatus::Experimental.code(), "experimental");
        assert_eq!(CapabilityStatus::Experimental.name(), "Experimental");
    }

    #[test]
    fn production_routing_status_codes_and_names_are_stable() {
        assert_eq!(ProductionRoutingStatus::AYFirst.code(), "ay_first");
        assert_eq!(ProductionRoutingStatus::AYFirst.name(), "AYFirst");
        assert_eq!(
            ProductionRoutingStatus::JustifiedLocalFallback.code(),
            "justified_local_fallback"
        );
        assert_eq!(
            ProductionRoutingStatus::JustifiedLocalFallback.name(),
            "JustifiedLocalFallback"
        );
        assert_eq!(
            ProductionRoutingStatus::UnjustifiedLocalFallback.code(),
            "unjustified_local_fallback"
        );
        assert_eq!(
            ProductionRoutingStatus::UnjustifiedLocalFallback.name(),
            "UnjustifiedLocalFallback"
        );
        assert_eq!(
            ProductionRoutingStatus::OtherProduction.code(),
            "other_production"
        );
        assert_eq!(
            ProductionRoutingStatus::OtherProduction.name(),
            "OtherProduction"
        );
        assert_eq!(
            ProductionRoutingStatus::NoProductionSelection.code(),
            "no_production_selection"
        );
        assert_eq!(
            ProductionRoutingStatus::NoProductionSelection.name(),
            "NoProductionSelection"
        );
    }

    #[test]
    fn symbolic_execution_codes_and_names_are_stable() {
        assert_eq!(SymbolicExecutionStatus::NotDetected.code(), "not_detected");
        assert_eq!(SymbolicExecutionStatus::NotDetected.name(), "NotDetected");
        assert_eq!(SymbolicExecutionStatus::AYPreferred.code(), "ay_preferred");
        assert_eq!(SymbolicExecutionStatus::AYPreferred.name(), "AYPreferred");
        assert_eq!(SymbolicExecutionStatus::AYRequired.code(), "ay_required");
        assert_eq!(SymbolicExecutionStatus::AYRequired.name(), "AYRequired");
        assert_eq!(
            SymbolicExecutionStatus::LocalFallbackAfterAYRejection.code(),
            "local_fallback_after_ay_rejection"
        );
        assert_eq!(
            SymbolicExecutionStatus::LocalFallbackAfterAYRejection.name(),
            "LocalFallbackAfterAYRejection"
        );

        assert_eq!(
            SymbolicExecutionReason::SymbolicInitialState.code(),
            "symbolic_initial_state"
        );
        assert_eq!(
            SymbolicExecutionReason::SymbolicInitialState.name(),
            "SymbolicInitialState"
        );
        assert_eq!(
            SymbolicExecutionReason::SymbolicTransitionRelation.code(),
            "symbolic_transition_relation"
        );
        assert_eq!(
            SymbolicExecutionReason::SymbolicTransitionRelation.name(),
            "SymbolicTransitionRelation"
        );
        assert_eq!(
            SymbolicExecutionReason::StateSpaceExplosion.code(),
            "state_space_explosion"
        );
        assert_eq!(
            SymbolicExecutionReason::StateSpaceExplosion.name(),
            "StateSpaceExplosion"
        );
        assert_eq!(
            SymbolicExecutionReason::UnsupportedLocalFragment.code(),
            "unsupported_local_fragment"
        );
        assert_eq!(
            SymbolicExecutionReason::UnsupportedLocalFragment.name(),
            "UnsupportedLocalFragment"
        );
        assert_eq!(
            SymbolicExecutionReason::BitVectorFormula.code(),
            "bit_vector_formula"
        );
        assert_eq!(
            SymbolicExecutionReason::BitVectorFormula.name(),
            "BitVectorFormula"
        );
        assert_eq!(
            SymbolicExecutionReason::ModelEnumeration.code(),
            "model_enumeration"
        );
        assert_eq!(
            SymbolicExecutionReason::ModelEnumeration.name(),
            "ModelEnumeration"
        );
        assert_eq!(
            SymbolicExecutionReason::NativeKernelUnsupported.code(),
            "native_kernel_unsupported"
        );
        assert_eq!(
            SymbolicExecutionReason::NativeKernelUnsupported.name(),
            "NativeKernelUnsupported"
        );
    }

    #[test]
    fn normalized_reason_code_uses_shared_none_sentinel() {
        let available = ay_smt_capability(BackendDomain::PetriMcc, ProblemKind::Bmc);
        assert_eq!(available.normalized_reason_code(), NO_REASON_CODE);

        let unsupported = BackendCapability::unsupported(
            BackendDomain::PetriMcc,
            BackendKind::NativeKernel,
            UnsupportedReason::NativeKernelUnavailable,
        );
        assert_eq!(
            unsupported.normalized_reason_code(),
            "native_kernel_unavailable"
        );
    }

    #[test]
    fn capability_problem_helpers_expose_optional_codes_and_names() {
        let without_problem = BackendCapability::available(
            BackendDomain::Shared,
            BackendKind::ExplicitState,
            "shared explicit-state lane",
        );
        assert_eq!(without_problem.problem_code(), None);
        assert_eq!(without_problem.problem_name(), None);
        assert_eq!(without_problem.problem_name_or_none(), "None");

        let with_problem = ay_chc_capability(BackendDomain::Btor2, ProblemKind::Chc);
        assert_eq!(with_problem.problem_code(), Some("chc"));
        assert_eq!(with_problem.problem_name(), Some("Chc"));
        assert_eq!(with_problem.problem_name_or_none(), "Chc");

        let report = CapabilityReport::new(ProblemKind::Safety);
        assert_eq!(report.problem_code(), Some("safety"));
        assert_eq!(report.problem_name(), Some("Safety"));

        let empty_report = CapabilityReport::default();
        assert_eq!(empty_report.problem_code(), None);
        assert_eq!(empty_report.problem_name(), None);
    }

    #[test]
    fn selected_lane_rendering_matches_existing_hardware_evidence_shape() {
        let capability = ay_chc_capability(BackendDomain::Btor2, ProblemKind::Chc);

        assert_eq!(
            render_capability_lane_evidence(
                "BTOR2",
                CapabilityLaneDecision::Selected,
                &capability
            ),
            "BTOR2 selected_lane backend=AYChc role=Production problem=Chc status=Available reason_code=none"
        );
        assert_eq!(
            capability.render_lane_evidence("BTOR2", CapabilityLaneDecision::Selected),
            "BTOR2 selected_lane backend=AYChc role=Production problem=Chc status=Available reason_code=none"
        );
    }

    #[test]
    fn rejected_lane_rendering_matches_existing_hardware_evidence_shape() {
        let capability = BackendCapability::unsupported(
            BackendDomain::Aiger,
            BackendKind::NativeKernel,
            UnsupportedReason::NativeKernelUnavailable,
        )
        .for_problem(ProblemKind::NativeSuccessor)
        .with_role(CapabilityRole::Validation);

        assert_eq!(
            render_capability_lane_evidence(
                "AIGER",
                CapabilityLaneDecision::Rejected,
                &capability
            ),
            "AIGER rejected_lane backend=NativeKernel role=Validation problem=NativeSuccessor status=Unsupported reason_code=native_kernel_unavailable"
        );
    }

    #[test]
    fn lane_status_rendering_uses_shared_machine_codes() {
        let selected = BackendCapability::available(
            BackendDomain::Btor2,
            BackendKind::Btor2Portfolio,
            "BTOR2 local orchestration",
        )
        .for_problem(ProblemKind::Safety)
        .with_role(CapabilityRole::Validation);

        assert_eq!(
            render_capability_lane_status_evidence(
                "BTOR2",
                CapabilityLaneDecision::Selected,
                &selected
            ),
            "BTOR2 shared_lane lane_status=selected backend=Btor2Portfolio backend_code=btor2_portfolio backend_role=validation problem=Safety capability_status=available reason_code=none"
        );
        assert_eq!(
            selected.render_lane_status_evidence("BTOR2", CapabilityLaneDecision::Selected),
            "BTOR2 shared_lane lane_status=selected backend=Btor2Portfolio backend_code=btor2_portfolio backend_role=validation problem=Safety capability_status=available reason_code=none"
        );

        let rejected = BackendCapability::unsupported(
            BackendDomain::Aiger,
            BackendKind::NativeKernel,
            UnsupportedReason::NativeKernelUnavailable,
        )
        .for_problem(ProblemKind::NativeSuccessor)
        .with_role(CapabilityRole::Validation);

        assert_eq!(
            rejected.render_lane_status_evidence("AIGER", CapabilityLaneDecision::Rejected),
            "AIGER shared_lane lane_status=rejected backend=NativeKernel backend_code=native_kernel backend_role=validation problem=NativeSuccessor capability_status=unsupported reason_code=native_kernel_unavailable"
        );
    }

    #[test]
    fn lane_status_rendering_preserves_no_problem_sentinel() {
        let unavailable = BackendCapability::unavailable(
            BackendDomain::PetriMcc,
            BackendKind::ExternalAYBinary,
            UnsupportedReason::MissingBinary("ay"),
        )
        .with_role(CapabilityRole::Fallback);

        assert_eq!(
            unavailable.render_lane_status_evidence("MCC", CapabilityLaneDecision::Rejected),
            "MCC shared_lane lane_status=rejected backend=ExternalAYBinary backend_code=external_ay_binary backend_role=fallback problem=None capability_status=unavailable reason_code=missing_binary"
        );
    }

    #[test]
    fn routing_status_evidence_uses_shared_status_name() {
        let mut report = CapabilityReport::new(ProblemKind::Bmc).with_limits(SolverLimits {
            time_budget: None,
            max_depth: Some(8),
            max_states: None,
            max_memory_bytes: None,
        });
        report.select(ay_smt_capability(BackendDomain::PetriMcc, ProblemKind::Bmc));

        assert_eq!(report.production_routing_status_name(), "AYFirst");
        assert_eq!(report.production_routing_status_code(), "ay_first");
        assert_eq!(
            report.render_production_routing_status_evidence("MCC"),
            "MCC production_routing_status=AYFirst"
        );
    }

    #[test]
    fn symbolic_execution_detection_rendering_uses_shared_codes() {
        let detection =
            SymbolicExecutionDetection::ay_preferred(SymbolicExecutionReason::ModelEnumeration);

        assert_eq!(detection.status_code(), "ay_preferred");
        assert_eq!(detection.status_name(), "AYPreferred");
        assert_eq!(detection.reason_code(), Some("model_enumeration"));
        assert_eq!(detection.reason_name(), Some("ModelEnumeration"));
        assert_eq!(detection.normalized_reason_code(), "model_enumeration");
        assert_eq!(
            render_symbolic_execution_detection_evidence(
                "TLA",
                ProblemKind::SymbolicExecution,
                detection
            ),
            "TLA symbolic_execution domain=tla status=AYPreferred status_code=ay_preferred problem=SymbolicExecution reason=ModelEnumeration reason_code=model_enumeration preferred_backend=AYSmt preferred_backend_code=ay_smt"
        );
        assert_eq!(
            detection.render_evidence("AIGER", ProblemKind::Sat),
            "AIGER symbolic_execution domain=aiger status=AYPreferred status_code=ay_preferred problem=Sat reason=ModelEnumeration reason_code=model_enumeration preferred_backend=AYSat preferred_backend_code=ay_sat"
        );
    }

    #[test]
    fn symbolic_execution_not_detected_rendering_uses_none_sentinels() {
        let detection = SymbolicExecutionDetection::not_detected();

        assert_eq!(detection.reason_code(), None);
        assert_eq!(detection.reason_name(), None);
        assert_eq!(detection.reason_name_or_none(), "None");
        assert_eq!(detection.normalized_reason_code(), NO_REASON_CODE);
        assert_eq!(
            detection.render_evidence("MCC", ProblemKind::Safety),
            "MCC symbolic_execution domain=petri_mcc status=NotDetected status_code=not_detected problem=Safety reason=None reason_code=none preferred_backend=None preferred_backend_code=none"
        );
    }
}
