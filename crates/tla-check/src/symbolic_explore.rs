// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Symbolic exploration engine for interactive state-space exploration.
//!
//! Wraps a [`BmcTranslator`] with push/pop scoping to enable interactive
//! symbolic exploration of TLA+ specs. Unlike concrete exploration (which
//! enumerates explicit states), symbolic exploration uses SMT solving to
//! find satisfying assignments, supports rollback via solver scopes, and
//! can enumerate alternate models via blocking clauses.
//!
//! Part of #3751: Apalache Gap 3 — interactive symbolic exploration API.

use tla_ay::{
    AYError, BmcState, BmcTranslator, BmcValue, SolveDecisionProfileSummary, SolveResult, TlaSort,
};
use tla_core::ast::{Expr, Module};
use tla_core::Spanned;
use tla_mc_core::{
    ProblemKind, SymbolicExecutionDetection, SymbolicExecutionReason, NO_REASON_CODE,
};

use crate::ay_pdr::expand_operators_for_chc;
use crate::ay_shared;
use crate::config::Config;
use crate::eval::EvalCtx;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors from symbolic exploration operations.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum SymbolicExploreError {
    /// Missing specification component (Init, Next, etc.).
    #[error("missing specification: {0}")]
    MissingSpec(String),
    /// SMT solver error.
    #[error("solver error: {0}")]
    SolverError(String),
    /// Translation error.
    #[error("translation error: {0}")]
    TranslationError(String),
    /// No satisfying assignment (UNSAT).
    #[error("no satisfying assignment: {0}")]
    #[cfg_attr(not(test), allow(dead_code))]
    Unsatisfiable(String),
    /// Scope stack underflow (pop without push).
    #[error("scope stack underflow: no scope to pop")]
    ScopeUnderflow,
    /// Invariant not found.
    #[error("invariant not found: {0}")]
    #[cfg_attr(not(test), allow(dead_code))]
    InvariantNotFound(String),
}

impl From<AYError> for SymbolicExploreError {
    fn from(err: AYError) -> Self {
        match err {
            AYError::Solver(inner) => SymbolicExploreError::SolverError(inner.to_string()),
            other => SymbolicExploreError::TranslationError(other.to_string()),
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Check if any variable sort requires array-based SMT encoding.
fn needs_array_logic(var_sorts: &[(String, TlaSort)]) -> bool {
    var_sorts.iter().any(|(_, sort)| {
        matches!(
            sort,
            TlaSort::Set { .. } | TlaSort::Function { .. } | TlaSort::Sequence { .. }
        )
    })
}

/// Create a BMC translator with the appropriate logic for the given variable sorts.
fn make_translator(
    var_sorts: &[(String, TlaSort)],
    depth: usize,
) -> Result<BmcTranslator, SymbolicExploreError> {
    let translator = if needs_array_logic(var_sorts) {
        BmcTranslator::new_with_arrays(depth)?
    } else {
        BmcTranslator::new(depth)?
    };
    Ok(translator)
}

/// Extract a concrete state from a BMC model at a given step.
fn extract_state_at_step(
    translator: &BmcTranslator,
    model: &tla_ay::Model,
    step: usize,
) -> BmcState {
    let trace = translator.extract_trace(model);
    trace
        .into_iter()
        .find(|s| s.step == step)
        .unwrap_or(BmcState {
            step,
            assignments: std::collections::HashMap::new(),
        })
}

/// TLA-check boundary around AY's typed solve decision/profile summary.
///
/// This keeps the summarizer-ready evidence row, but exposes the consumer
/// acceptance fields directly so callers do not parse strings or apply a weaker
/// policy than AY's public acceptance boundary. For typed summaries, the
/// consumer boundary is AY's owned facade decision.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct AYSolveDecisionProfileEvidence {
    evidence: String,
    status: &'static str,
    status_code: &'static str,
    decision: &'static str,
    unknown_reason_code: &'static str,
    unknown_limit_code: &'static str,
    typed_consumer: bool,
    consumer_boundary: AYSolveDecisionProfileConsumerBoundary,
    model_blocking_capability: AYModelBlockingCapabilityEvidence,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum AYSolveDecisionProfileConsumerBoundary {
    Typed(tla_ay::SolveDecisionProfileModelConsumerDecision),
    MissingTypedSummary,
    #[cfg(test)]
    SyntheticForTesting {
        decision_code: &'static str,
        solve_accepted_for_consumer: bool,
        solve_consumer_rejection_code: &'static str,
        model_consumer_status_code: &'static str,
        model_consumer_reason_code: &'static str,
        model_consumer_accepted: bool,
        model_validated: bool,
        verification_level_code: &'static str,
        fail_closed: bool,
    },
}

impl AYSolveDecisionProfileConsumerBoundary {
    fn decision_code(&self) -> &'static str {
        match self {
            Self::Typed(decision) => decision.decision_code,
            Self::MissingTypedSummary => NO_REASON_CODE,
            #[cfg(test)]
            Self::SyntheticForTesting { decision_code, .. } => decision_code,
        }
    }

    fn solve_accepted_for_consumer(&self) -> bool {
        match self {
            Self::Typed(decision) => decision.solve_accepted_for_consumer,
            Self::MissingTypedSummary => false,
            #[cfg(test)]
            Self::SyntheticForTesting {
                solve_accepted_for_consumer,
                ..
            } => *solve_accepted_for_consumer,
        }
    }

    fn solve_consumer_rejection_code(&self) -> &'static str {
        match self {
            Self::Typed(decision) => decision
                .solve_consumer_rejection_code
                .unwrap_or(NO_REASON_CODE),
            Self::MissingTypedSummary => "missing_typed_summary",
            #[cfg(test)]
            Self::SyntheticForTesting {
                solve_consumer_rejection_code,
                ..
            } => solve_consumer_rejection_code,
        }
    }

    fn model_consumer_status_code(&self) -> &'static str {
        match self {
            Self::Typed(decision) => decision.status_code,
            Self::MissingTypedSummary => "rejected",
            #[cfg(test)]
            Self::SyntheticForTesting {
                model_consumer_status_code,
                ..
            } => model_consumer_status_code,
        }
    }

    fn model_consumer_reason_code(&self) -> &'static str {
        match self {
            Self::Typed(decision) => decision.reason_code,
            Self::MissingTypedSummary => "missing_typed_summary",
            #[cfg(test)]
            Self::SyntheticForTesting {
                model_consumer_reason_code,
                ..
            } => model_consumer_reason_code,
        }
    }

    fn model_consumer_accepted(&self) -> bool {
        match self {
            Self::Typed(decision) => decision.accepted_for_consumer,
            Self::MissingTypedSummary => false,
            #[cfg(test)]
            Self::SyntheticForTesting {
                model_consumer_accepted,
                ..
            } => *model_consumer_accepted,
        }
    }

    fn model_validated(&self) -> bool {
        match self {
            Self::Typed(decision) => decision.model_validated,
            Self::MissingTypedSummary => false,
            #[cfg(test)]
            Self::SyntheticForTesting {
                model_validated, ..
            } => *model_validated,
        }
    }

    fn verification_level_code(&self) -> &'static str {
        match self {
            Self::Typed(decision) => decision.verification_level_code,
            Self::MissingTypedSummary => NO_REASON_CODE,
            #[cfg(test)]
            Self::SyntheticForTesting {
                verification_level_code,
                ..
            } => verification_level_code,
        }
    }

    fn fail_closed(&self) -> bool {
        match self {
            Self::Typed(decision) => decision.fail_closed,
            Self::MissingTypedSummary => true,
            #[cfg(test)]
            Self::SyntheticForTesting { fail_closed, .. } => *fail_closed,
        }
    }

    fn to_json(&self) -> serde_json::Value {
        match self {
            Self::Typed(decision) => decision.to_json_value(),
            Self::MissingTypedSummary => serde_json::json!({
                "status": "rejected",
                "reason": "missing_typed_summary",
                "accepted_for_consumer": false,
                "fail_closed": true,
                "decision": NO_REASON_CODE,
                "solve_accepted_for_consumer": false,
                "solve_consumer_rejection_code": "missing_typed_summary",
                "model_validated": false,
                "verification_level_code": NO_REASON_CODE,
            }),
            #[cfg(test)]
            Self::SyntheticForTesting {
                decision_code,
                solve_accepted_for_consumer,
                solve_consumer_rejection_code,
                model_consumer_status_code,
                model_consumer_reason_code,
                model_consumer_accepted,
                model_validated,
                verification_level_code,
                fail_closed,
            } => serde_json::json!({
                "status": model_consumer_status_code,
                "reason": model_consumer_reason_code,
                "accepted_for_consumer": model_consumer_accepted,
                "fail_closed": fail_closed,
                "decision": decision_code,
                "solve_accepted_for_consumer": solve_accepted_for_consumer,
                "solve_consumer_rejection_code": solve_consumer_rejection_code,
                "model_validated": model_validated,
                "verification_level_code": verification_level_code,
            }),
        }
    }
}

/// TLA-check boundary around AY's public symbolic-execution contract manifest.
///
/// The availability/status fields come from AY's typed aggregate manifest.
/// TLA-check only records and exposes that contract for downstream diagnostics.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct AYModelBlockingCapabilityEvidence {
    evidence: String,
    route_admission_evidence: String,
    capability_route_readiness_evidence: String,
    manifest: tla_ay::SymbolicExecutionContractManifest,
    health_report: tla_ay::SymbolicExecutionContractManifestHealthReport,
    diagnostic_summary: tla_ay::SymbolicExecutionContractManifestDiagnosticSummary,
    route_admission: tla_ay::SymbolicExecutionRouteAdmissionDecision,
    symbolic_contracts: Vec<tla_ay::SymbolicExecutionCapabilityRouteReadiness>,
    solver: &'static str,
    capability: &'static str,
    status_code: &'static str,
    reason_code: &'static str,
    api_symbols: Vec<&'static str>,
    evidence_schemas: Vec<&'static str>,
    consumer_responsibilities: Vec<&'static str>,
    typed_consumer: bool,
    manifest_valid: bool,
    fail_closed: bool,
}

fn capability_readiness<'a>(
    symbolic_contracts: &'a [tla_ay::SymbolicExecutionCapabilityRouteReadiness],
    capability: tla_ay::SolverCapabilityCode,
) -> Option<&'a tla_ay::SymbolicExecutionCapabilityRouteReadiness> {
    symbolic_contracts
        .iter()
        .find(|readiness| readiness.capability == capability)
}

fn capability_readiness_json(
    symbolic_contracts: &[tla_ay::SymbolicExecutionCapabilityRouteReadiness],
    capability: tla_ay::SolverCapabilityCode,
    route_admission: &tla_ay::SymbolicExecutionRouteAdmissionDecision,
) -> serde_json::Value {
    capability_readiness(symbolic_contracts, capability)
        .cloned()
        .unwrap_or_else(|| {
            tla_ay::symbolic_execution_capability_route_readiness_for_decision(
                capability,
                route_admission,
            )
        })
        .to_json_value()
}

fn capability_route_ready(readiness: &tla_ay::SymbolicExecutionCapabilityRouteReadiness) -> bool {
    readiness.accepted_for_consumer && readiness.fail_closed
}

fn all_supported_capability_route_readiness_ready(
    symbolic_contracts: &[tla_ay::SymbolicExecutionCapabilityRouteReadiness],
) -> bool {
    !symbolic_contracts.is_empty() && symbolic_contracts.iter().all(capability_route_ready)
}

fn render_evidence_value(value: &str) -> &str {
    if value.is_empty() {
        NO_REASON_CODE
    } else {
        value
    }
}

fn render_capability_route_readiness_evidence(
    scope: &str,
    symbolic_contracts: &[tla_ay::SymbolicExecutionCapabilityRouteReadiness],
) -> String {
    let fields = symbolic_contracts
        .iter()
        .flat_map(|readiness| {
            let capability = readiness.capability_code;
            readiness
                .to_key_value_rows()
                .into_iter()
                .map(move |(key, value)| {
                    format!("{capability}_{key}={}", render_evidence_value(&value))
                })
        })
        .collect::<Vec<_>>();
    format!(
        "{} ay_symbolic_execution_all_supported_capability_route_readiness {}",
        scope,
        fields.join(" ")
    )
}

fn render_route_admission_evidence(
    scope: &str,
    route_admission: &tla_ay::SymbolicExecutionRouteAdmissionDecision,
) -> String {
    let fields = route_admission
        .to_key_value_rows()
        .into_iter()
        .map(|(key, value)| format!("{key}={}", render_evidence_value(&value)))
        .collect::<Vec<_>>();
    format!(
        "{} ay_symbolic_execution_route_admission {}",
        scope,
        fields.join(" ")
    )
}

impl AYModelBlockingCapabilityEvidence {
    fn fail_closed(
        scope: &str,
        manifest: tla_ay::SymbolicExecutionContractManifest,
        manifest_pairs: Vec<(&'static str, String)>,
        health_report: tla_ay::SymbolicExecutionContractManifestHealthReport,
        diagnostic_summary: tla_ay::SymbolicExecutionContractManifestDiagnosticSummary,
        route_admission: tla_ay::SymbolicExecutionRouteAdmissionDecision,
        symbolic_contracts: Vec<tla_ay::SymbolicExecutionCapabilityRouteReadiness>,
        model_blocking_contract: tla_ay::SolverCapabilityContract,
        manifest_valid: bool,
    ) -> Self {
        let capability = model_blocking_contract.capability_code;
        let model_blocking_readiness = capability_readiness(
            &symbolic_contracts,
            tla_ay::SolverCapabilityCode::ModelBlocking,
        );
        let status_code = model_blocking_readiness
            .map(|readiness| readiness.status_code)
            .unwrap_or(tla_ay::SymbolicExecutionCapabilityRouteReadinessStatus::Blocked.code());
        let reason_code = model_blocking_readiness
            .map(|readiness| readiness.reason_code)
            .unwrap_or(
                tla_ay::SymbolicExecutionCapabilityRouteReadinessReason::MissingReadinessRow.code(),
            );
        let api_symbols = model_blocking_readiness
            .map(|readiness| readiness.api_symbols.to_vec())
            .unwrap_or_else(|| model_blocking_contract.api_symbols.to_vec());
        let evidence_schemas = model_blocking_readiness
            .map(|readiness| readiness.evidence_schemas.to_vec())
            .unwrap_or_else(|| model_blocking_contract.evidence_schemas.to_vec());
        let consumer_responsibilities = model_blocking_readiness
            .map(|readiness| readiness.consumer_responsibilities.to_vec())
            .unwrap_or_else(|| model_blocking_contract.consumer_responsibilities.to_vec());
        Self {
            evidence: render_model_blocking_capability_evidence(
                scope,
                &manifest_pairs,
                &health_report,
                &diagnostic_summary,
                &route_admission,
                &symbolic_contracts,
                &model_blocking_contract,
                capability,
                status_code,
                reason_code,
                false,
                true,
                manifest_valid,
            ),
            route_admission_evidence: render_route_admission_evidence(scope, &route_admission),
            capability_route_readiness_evidence: render_capability_route_readiness_evidence(
                scope,
                &symbolic_contracts,
            ),
            solver: manifest.solver,
            manifest,
            health_report,
            diagnostic_summary,
            route_admission,
            symbolic_contracts,
            capability,
            status_code,
            reason_code,
            api_symbols,
            evidence_schemas,
            consumer_responsibilities,
            typed_consumer: false,
            fail_closed: true,
            manifest_valid,
        }
    }
}

/// Local adapter for AY-derived symbolic evidence consumed by TLA-check.
#[derive(Debug, Clone)]
struct AYSymbolicEvidenceAdapter {
    manifest: tla_ay::SymbolicExecutionContractManifest,
    manifest_pairs: Vec<(&'static str, String)>,
    health_report: tla_ay::SymbolicExecutionContractManifestHealthReport,
    diagnostic_summary: tla_ay::SymbolicExecutionContractManifestDiagnosticSummary,
    route_admission: tla_ay::SymbolicExecutionRouteAdmissionDecision,
    model_blocking_contract: tla_ay::SolverCapabilityContract,
}

impl AYSymbolicEvidenceAdapter {
    fn from_current_ay() -> Self {
        let manifest = tla_ay::symbolic_execution_contract_manifest();
        let manifest_pairs = tla_ay::symbolic_execution_contract_manifest_key_value_pairs();
        let health_report = tla_ay::validate_symbolic_execution_contract_manifest_round_trip(
            &manifest,
            &manifest_pairs,
        );
        let diagnostic_summary =
            tla_ay::symbolic_execution_contract_manifest_diagnostic_summary_for_round_trip(
                &manifest,
                &manifest_pairs,
            );
        let route_admission =
            tla_ay::symbolic_execution_route_admission_decision_for_summary(&diagnostic_summary);
        Self {
            manifest,
            manifest_pairs,
            health_report,
            diagnostic_summary,
            route_admission,
            model_blocking_contract: tla_ay::model_blocking_symbolic_execution_contract(),
        }
    }

    #[cfg(test)]
    fn from_parts_for_testing(
        manifest: tla_ay::SymbolicExecutionContractManifest,
        manifest_pairs: Vec<(&'static str, String)>,
    ) -> Self {
        let health_report = tla_ay::validate_symbolic_execution_contract_manifest_round_trip(
            &manifest,
            &manifest_pairs,
        );
        let diagnostic_summary =
            tla_ay::symbolic_execution_contract_manifest_diagnostic_summary_for_round_trip(
                &manifest,
                &manifest_pairs,
            );
        let route_admission =
            tla_ay::symbolic_execution_route_admission_decision_for_summary(&diagnostic_summary);
        Self {
            manifest,
            manifest_pairs,
            health_report,
            diagnostic_summary,
            route_admission,
            model_blocking_contract: tla_ay::model_blocking_symbolic_execution_contract(),
        }
    }

    fn manifest_valid(&self) -> bool {
        self.route_admission.accepted_for_consumer && self.route_admission.fail_closed
    }

    fn symbolic_contracts(&self) -> Vec<tla_ay::SymbolicExecutionCapabilityRouteReadiness> {
        tla_ay::symbolic_execution_all_supported_capability_route_readiness_for_decision(
            &self.route_admission,
        )
    }

    fn model_blocking_capability_report(&self, scope: &str) -> AYModelBlockingCapabilityEvidence {
        let manifest_pairs = self.manifest_pairs.clone();
        let manifest_valid = self.manifest_valid();
        let symbolic_contracts = self.symbolic_contracts();
        let all_supported_readiness_ready =
            all_supported_capability_route_readiness_ready(&symbolic_contracts);
        let model_blocking_readiness = capability_readiness(
            &symbolic_contracts,
            tla_ay::SolverCapabilityCode::ModelBlocking,
        )
        .cloned();

        match (
            manifest_valid && all_supported_readiness_ready,
            model_blocking_readiness,
        ) {
            (true, Some(readiness)) if capability_route_ready(&readiness) => {
                AYModelBlockingCapabilityEvidence {
                    evidence: render_model_blocking_capability_evidence(
                        scope,
                        &manifest_pairs,
                        &self.health_report,
                        &self.diagnostic_summary,
                        &self.route_admission,
                        &symbolic_contracts,
                        &self.model_blocking_contract,
                        readiness.capability_code,
                        readiness.status_code,
                        readiness.reason_code,
                        true,
                        readiness.fail_closed,
                        true,
                    ),
                    route_admission_evidence: render_route_admission_evidence(
                        scope,
                        &self.route_admission,
                    ),
                    capability_route_readiness_evidence: render_capability_route_readiness_evidence(
                        scope,
                        &symbolic_contracts,
                    ),
                    solver: self.manifest.solver,
                    manifest: self.manifest,
                    health_report: self.health_report.clone(),
                    diagnostic_summary: self.diagnostic_summary.clone(),
                    route_admission: self.route_admission.clone(),
                    symbolic_contracts,
                    capability: readiness.capability_code,
                    status_code: readiness.status_code,
                    reason_code: readiness.reason_code,
                    api_symbols: readiness.api_symbols.to_vec(),
                    evidence_schemas: readiness.evidence_schemas.to_vec(),
                    consumer_responsibilities: readiness.consumer_responsibilities.to_vec(),
                    typed_consumer: true,
                    fail_closed: readiness.fail_closed,
                    manifest_valid: true,
                }
            }
            _ => AYModelBlockingCapabilityEvidence::fail_closed(
                scope,
                self.manifest,
                manifest_pairs,
                self.health_report.clone(),
                self.diagnostic_summary.clone(),
                self.route_admission.clone(),
                symbolic_contracts,
                self.model_blocking_contract,
                manifest_valid,
            ),
        }
    }

    fn symbolic_evidence_report(
        &self,
        scope: &str,
        summary: Option<&SolveDecisionProfileSummary>,
    ) -> AYSymbolicEvidenceReport {
        let model_blocking_capability = self.model_blocking_capability_report(scope);
        let solver_decision_profile = AYSolveDecisionProfileEvidence::from_summary_with_capability(
            scope,
            summary,
            model_blocking_capability.clone(),
        );
        AYSymbolicEvidenceReport {
            solver_decision_profile,
            model_blocking_capability,
        }
    }
}

/// Combined symbolic evidence generated from one adapter snapshot.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct AYSymbolicEvidenceReport {
    solver_decision_profile: AYSolveDecisionProfileEvidence,
    model_blocking_capability: AYModelBlockingCapabilityEvidence,
}

impl AYSymbolicEvidenceReport {
    pub(crate) fn solver_decision_profile(&self) -> &AYSolveDecisionProfileEvidence {
        &self.solver_decision_profile
    }

    pub(crate) fn model_blocking_capability(&self) -> &AYModelBlockingCapabilityEvidence {
        &self.model_blocking_capability
    }

    pub(crate) fn route_admission_evidence_row(&self) -> &str {
        self.model_blocking_capability
            .route_admission_evidence_row()
    }

    pub(crate) fn route_admission_json(&self) -> serde_json::Value {
        self.model_blocking_capability.route_admission_json()
    }

    pub(crate) fn capability_route_readiness_evidence_row(&self) -> &str {
        self.model_blocking_capability
            .capability_route_readiness_evidence_row()
    }

    pub(crate) fn all_supported_capability_route_readiness_json(&self) -> serde_json::Value {
        self.model_blocking_capability
            .all_supported_capability_route_readiness_json()
    }

    pub(crate) fn model_blocking_capability_route_readiness_json(&self) -> serde_json::Value {
        self.model_blocking_capability
            .model_blocking_capability_route_readiness_json()
    }
}

impl AYModelBlockingCapabilityEvidence {
    pub(crate) fn evidence_row(&self) -> &str {
        &self.evidence
    }

    pub(crate) fn route_admission_evidence_row(&self) -> &str {
        &self.route_admission_evidence
    }

    pub(crate) fn capability_route_readiness_evidence_row(&self) -> &str {
        &self.capability_route_readiness_evidence
    }

    pub(crate) fn route_admission_json(&self) -> serde_json::Value {
        self.route_admission.to_json_value()
    }

    pub(crate) fn all_supported_capability_route_readiness_json(&self) -> serde_json::Value {
        serde_json::json!(self
            .symbolic_contracts
            .iter()
            .map(tla_ay::SymbolicExecutionCapabilityRouteReadiness::to_json_value)
            .collect::<Vec<_>>())
    }

    pub(crate) fn model_blocking_capability_route_readiness_json(&self) -> serde_json::Value {
        capability_readiness_json(
            &self.symbolic_contracts,
            tla_ay::SolverCapabilityCode::ModelBlocking,
            &self.route_admission,
        )
    }

    pub(crate) fn to_json(&self) -> serde_json::Value {
        let contract_capabilities = self
            .manifest
            .contracts
            .iter()
            .map(|contract| contract.capability_code)
            .collect::<Vec<_>>();
        let symbolic_contracts = self
            .symbolic_contracts
            .iter()
            .map(tla_ay::SymbolicExecutionCapabilityRouteReadiness::to_json_value)
            .collect::<Vec<_>>();

        let mut object = serde_json::Map::new();
        object.insert("evidence".to_string(), serde_json::json!(self.evidence));
        object.insert(
            "manifest_schema".to_string(),
            serde_json::json!(self.manifest.schema),
        );
        object.insert(
            "manifest_schema_version".to_string(),
            serde_json::json!(self.manifest.schema_version),
        );
        object.insert("solver".to_string(), serde_json::json!(self.solver));
        object.insert(
            "contract_count".to_string(),
            serde_json::json!(self.manifest.contracts.len()),
        );
        object.insert(
            "contract_capabilities".to_string(),
            serde_json::json!(contract_capabilities),
        );
        object.insert(
            "symbolic_execution_contract_manifest".to_string(),
            self.manifest.to_json_value(),
        );
        object.insert(
            "symbolic_execution_contract_manifest_health".to_string(),
            self.health_report.to_json_value(),
        );
        object.insert(
            "symbolic_execution_contract_diagnostic_summary".to_string(),
            self.diagnostic_summary.to_json_value(),
        );
        object.insert(
            "symbolic_execution_route_admission".to_string(),
            self.route_admission.to_json_value(),
        );
        object.insert(
            "symbolic_execution_route_admission_evidence".to_string(),
            serde_json::json!(&self.route_admission_evidence),
        );
        object.insert(
            "health_schema".to_string(),
            serde_json::json!(self.health_report.schema),
        );
        object.insert(
            "health_schema_version".to_string(),
            serde_json::json!(self.health_report.schema_version),
        );
        object.insert(
            "health_status_code".to_string(),
            serde_json::json!(self.health_report.status_code),
        );
        object.insert(
            "health_reason_code".to_string(),
            serde_json::json!(self.health_report.reason_code),
        );
        object.insert(
            "health_diagnostic_code".to_string(),
            serde_json::json!(self.health_report.diagnostic_code()),
        );
        object.insert(
            "accepted_for_consumer".to_string(),
            serde_json::json!(self.contract_ready(tla_ay::SolverCapabilityCode::ModelBlocking)),
        );
        object.insert(
            "route_admission_schema".to_string(),
            serde_json::json!(self.route_admission.schema),
        );
        object.insert(
            "route_admission_schema_version".to_string(),
            serde_json::json!(self.route_admission.schema_version),
        );
        object.insert(
            "route_admission_status_code".to_string(),
            serde_json::json!(self.route_admission.status_code),
        );
        object.insert(
            "route_admission_reason_code".to_string(),
            serde_json::json!(self.route_admission.reason_code),
        );
        object.insert(
            "route_admission_accepted_for_consumer".to_string(),
            serde_json::json!(self.route_admission.accepted_for_consumer),
        );
        object.insert(
            "route_admission_fail_closed".to_string(),
            serde_json::json!(self.route_admission.fail_closed),
        );
        object.insert(
            "route_capabilities".to_string(),
            serde_json::json!(&self.route_admission.route_capabilities),
        );
        object.insert(
            "route_authorities".to_string(),
            serde_json::json!(&self.route_admission.route_authorities),
        );
        object.insert(
            "route_validators".to_string(),
            serde_json::json!(self.route_admission.validators),
        );
        object.insert(
            "route_issue_field".to_string(),
            serde_json::json!(&self.route_admission.issue_field),
        );
        object.insert(
            "route_issue_expected".to_string(),
            serde_json::json!(&self.route_admission.issue_expected),
        );
        object.insert(
            "route_issue_actual".to_string(),
            serde_json::json!(&self.route_admission.issue_actual),
        );
        object.insert(
            "required_capabilities".to_string(),
            serde_json::json!(&self.health_report.required_capabilities),
        );
        object.insert(
            "present_capabilities".to_string(),
            serde_json::json!(&self.health_report.present_capabilities),
        );
        object.insert(
            "symbolic_execution_contracts".to_string(),
            serde_json::json!(symbolic_contracts),
        );
        object.insert(
            "symbolic_execution_all_supported_capability_route_readiness".to_string(),
            self.all_supported_capability_route_readiness_json(),
        );
        object.insert(
            "symbolic_execution_all_supported_capability_route_readiness_evidence".to_string(),
            serde_json::json!(&self.capability_route_readiness_evidence),
        );
        object.insert(
            "all_supported_capability_route_readiness_ready".to_string(),
            serde_json::json!(all_supported_capability_route_readiness_ready(
                &self.symbolic_contracts
            )),
        );
        object.insert(
            "model_blocking_ready".to_string(),
            serde_json::json!(self.contract_ready(tla_ay::SolverCapabilityCode::ModelBlocking)),
        );
        object.insert(
            "model_blocking_capability_route_readiness".to_string(),
            capability_readiness_json(
                &self.symbolic_contracts,
                tla_ay::SolverCapabilityCode::ModelBlocking,
                &self.route_admission,
            ),
        );
        object.insert(
            "incremental_assumptions_ready".to_string(),
            serde_json::json!(
                self.contract_ready(tla_ay::SolverCapabilityCode::IncrementalAssumptions)
            ),
        );
        object.insert(
            "incremental_assumptions_capability_route_readiness".to_string(),
            capability_readiness_json(
                &self.symbolic_contracts,
                tla_ay::SolverCapabilityCode::IncrementalAssumptions,
                &self.route_admission,
            ),
        );
        object.insert(
            "all_sat_enumeration_ready".to_string(),
            serde_json::json!(self.contract_ready(tla_ay::SolverCapabilityCode::AllSatEnumeration)),
        );
        object.insert(
            "all_sat_enumeration_capability_route_readiness".to_string(),
            capability_readiness_json(
                &self.symbolic_contracts,
                tla_ay::SolverCapabilityCode::AllSatEnumeration,
                &self.route_admission,
            ),
        );
        object.insert("capability".to_string(), serde_json::json!(self.capability));
        object.insert(
            "status_code".to_string(),
            serde_json::json!(self.status_code),
        );
        object.insert(
            "reason_code".to_string(),
            serde_json::json!(self.reason_code),
        );
        object.insert(
            "api_symbols".to_string(),
            serde_json::json!(&self.api_symbols),
        );
        object.insert(
            "evidence_schemas".to_string(),
            serde_json::json!(&self.evidence_schemas),
        );
        object.insert(
            "consumer_responsibilities".to_string(),
            serde_json::json!(&self.consumer_responsibilities),
        );
        object.insert(
            "capability_manifest".to_string(),
            self.manifest.to_json_value(),
        );
        object.insert(
            "typed_consumer".to_string(),
            serde_json::json!(self.typed_consumer),
        );
        object.insert("production_selected".to_string(), serde_json::json!(false));
        object.insert(
            "all_contracts_fail_closed".to_string(),
            serde_json::json!(self.manifest.all_contracts_fail_closed),
        );
        object.insert(
            "manifest_valid".to_string(),
            serde_json::json!(self.manifest_valid),
        );
        object.insert(
            "fail_closed".to_string(),
            serde_json::json!(self.fail_closed),
        );

        serde_json::Value::Object(object)
    }

    fn contract_ready(&self, capability: tla_ay::SolverCapabilityCode) -> bool {
        capability_readiness(&self.symbolic_contracts, capability)
            .is_some_and(capability_route_ready)
    }
}

fn render_model_blocking_capability_evidence(
    scope: &str,
    manifest_pairs: &[(&'static str, String)],
    health_report: &tla_ay::SymbolicExecutionContractManifestHealthReport,
    diagnostic_summary: &tla_ay::SymbolicExecutionContractManifestDiagnosticSummary,
    route_admission: &tla_ay::SymbolicExecutionRouteAdmissionDecision,
    symbolic_contracts: &[tla_ay::SymbolicExecutionCapabilityRouteReadiness],
    model_blocking_contract: &tla_ay::SolverCapabilityContract,
    capability: &str,
    status_code: &str,
    reason_code: &str,
    typed_consumer: bool,
    fail_closed: bool,
    manifest_valid: bool,
) -> String {
    let mut fields = manifest_pairs
        .iter()
        .map(|(key, value)| format!("{key}={}", render_evidence_value(value)))
        .collect::<Vec<_>>();
    fields.extend(
        health_report
            .to_key_value_pairs()
            .into_iter()
            .map(|(key, value)| format!("health_{key}={}", render_evidence_value(&value))),
    );
    fields.extend(
        diagnostic_summary
            .to_key_value_rows()
            .into_iter()
            .map(|(key, value)| {
                format!("diagnostic_summary_{key}={}", render_evidence_value(&value))
            }),
    );
    fields.extend(
        route_admission
            .to_key_value_rows()
            .into_iter()
            .map(|(key, value)| format!("route_admission_{key}={}", render_evidence_value(&value))),
    );
    fields.extend(
        model_blocking_contract
            .to_key_value_pairs()
            .into_iter()
            .map(|(key, value)| {
                format!(
                    "model_blocking_contract_{key}={}",
                    render_evidence_value(&value)
                )
            }),
    );
    for readiness in symbolic_contracts {
        let capability = readiness.capability_code;
        fields.push(format!(
            "{capability}_ready={}",
            capability_route_ready(readiness)
        ));
        fields.push(format!(
            "{capability}_status_code={}",
            readiness.status_code
        ));
        fields.push(format!(
            "{capability}_reason_code={}",
            readiness.reason_code
        ));
        fields.extend(
            readiness
                .to_key_value_rows()
                .into_iter()
                .map(|(key, value)| {
                    format!(
                        "{capability}_route_readiness_{key}={}",
                        render_evidence_value(&value)
                    )
                }),
        );
    }
    fields.push(format!("capability={capability}"));
    fields.push(format!("status_code={status_code}"));
    fields.push(format!("reason_code={reason_code}"));
    fields.push(format!("typed_consumer={typed_consumer}"));
    fields.push("production_selected=false".to_string());
    fields.push(format!("manifest_valid={manifest_valid}"));
    fields.push(format!("fail_closed={fail_closed}"));
    format!(
        "{} ay_symbolic_execution_contract_manifest {}",
        scope,
        fields.join(" ")
    )
}

impl AYSolveDecisionProfileEvidence {
    pub(crate) fn from_summary(scope: &str, summary: Option<&SolveDecisionProfileSummary>) -> Self {
        AYSymbolicEvidenceAdapter::from_current_ay()
            .symbolic_evidence_report(scope, summary)
            .solver_decision_profile
    }

    fn from_summary_with_capability(
        scope: &str,
        summary: Option<&SolveDecisionProfileSummary>,
        model_blocking_capability: AYModelBlockingCapabilityEvidence,
    ) -> Self {
        match summary {
            Some(summary) => {
                Self::from_typed_summary_with_capability(scope, summary, model_blocking_capability)
            }
            None => Self::missing_with_capability(scope, model_blocking_capability),
        }
    }

    pub(crate) fn missing(scope: &str) -> Self {
        Self::from_summary(scope, None)
    }

    fn missing_with_capability(
        scope: &str,
        model_blocking_capability: AYModelBlockingCapabilityEvidence,
    ) -> Self {
        Self {
            evidence: tla_ay::solve_decision_profile_summary_evidence(scope, None),
            status: "Unavailable",
            status_code: "missing_typed_summary",
            decision: "None",
            unknown_reason_code: NO_REASON_CODE,
            unknown_limit_code: NO_REASON_CODE,
            typed_consumer: false,
            consumer_boundary: AYSolveDecisionProfileConsumerBoundary::MissingTypedSummary,
            model_blocking_capability,
        }
    }

    fn from_typed_summary_with_capability(
        scope: &str,
        summary: &SolveDecisionProfileSummary,
        model_blocking_capability: AYModelBlockingCapabilityEvidence,
    ) -> Self {
        let unknown_reason_code = summary
            .unknown
            .as_ref()
            .map_or(NO_REASON_CODE, |unknown| unknown.reason_code);
        let unknown_limit_code = summary
            .unknown
            .as_ref()
            .and_then(|unknown| unknown.limit_code)
            .unwrap_or(NO_REASON_CODE);
        let model_consumer_decision = summary.model_consumer_decision();

        Self {
            evidence: tla_ay::solve_decision_profile_summary_evidence(scope, Some(summary)),
            status: "Available",
            status_code: "typed_summary_available",
            decision: summary.decision_name,
            unknown_reason_code,
            unknown_limit_code,
            typed_consumer: true,
            consumer_boundary: AYSolveDecisionProfileConsumerBoundary::Typed(
                model_consumer_decision,
            ),
            model_blocking_capability,
        }
    }

    #[cfg(test)]
    pub(crate) fn from_typed_fields_for_testing(
        decision: tla_ay::SolveDecision,
        accepted_for_consumer: bool,
        consumer_rejection_code: Option<&'static str>,
        model_validated: bool,
    ) -> Self {
        let consumer_rejection_code = consumer_rejection_code.unwrap_or(NO_REASON_CODE);
        let model_consumer_accepted = decision.is_sat() && accepted_for_consumer && model_validated;
        let (model_consumer_status_code, model_consumer_reason_code) = if model_consumer_accepted {
            ("accepted", "accepted")
        } else if !decision.is_sat() {
            ("rejected", "non_sat_decision")
        } else if !accepted_for_consumer {
            ("rejected", "consumer_rejected")
        } else {
            ("rejected", "model_not_validated")
        };
        let model_blocking_capability =
            AYSymbolicEvidenceAdapter::from_current_ay().model_blocking_capability_report("TLA");

        Self {
            evidence: format!(
                "TLA ay_solver_decision_profile_summary status=Available status_code=typed_summary_available decision={} decision_code={} accepted_for_consumer={} consumer_rejection_code={} model_validated={} unknown_reason_code=none unknown_limit_code=none typed_consumer=true production_selected=false fail_closed={}",
                decision.name(),
                decision.code(),
                accepted_for_consumer,
                consumer_rejection_code,
                model_validated,
                !model_consumer_accepted,
            ),
            status: "Available",
            status_code: "typed_summary_available",
            decision: decision.name(),
            unknown_reason_code: NO_REASON_CODE,
            unknown_limit_code: NO_REASON_CODE,
            typed_consumer: true,
            consumer_boundary: AYSolveDecisionProfileConsumerBoundary::SyntheticForTesting {
                decision_code: decision.code(),
                solve_accepted_for_consumer: accepted_for_consumer,
                solve_consumer_rejection_code: consumer_rejection_code,
                model_consumer_status_code,
                model_consumer_reason_code,
                model_consumer_accepted,
                model_validated,
                verification_level_code: "testing",
                fail_closed: !model_consumer_accepted,
            },
            model_blocking_capability,
        }
    }

    pub(crate) fn evidence_row(&self) -> &str {
        &self.evidence
    }

    pub(crate) fn fail_closed(&self) -> bool {
        self.consumer_boundary.fail_closed()
    }

    pub(crate) fn accepts_model_for_tla_boundary(&self) -> bool {
        self.consumer_boundary.model_consumer_accepted()
    }

    fn decision_code(&self) -> &'static str {
        self.consumer_boundary.decision_code()
    }

    fn accepted_for_consumer(&self) -> bool {
        self.consumer_boundary.solve_accepted_for_consumer()
    }

    fn consumer_rejection_code(&self) -> &'static str {
        self.consumer_boundary.solve_consumer_rejection_code()
    }

    fn model_consumer_status_code(&self) -> &'static str {
        self.consumer_boundary.model_consumer_status_code()
    }

    fn model_consumer_reason_code(&self) -> &'static str {
        self.consumer_boundary.model_consumer_reason_code()
    }

    fn model_consumer_accepted(&self) -> bool {
        self.consumer_boundary.model_consumer_accepted()
    }

    fn model_validated(&self) -> bool {
        self.consumer_boundary.model_validated()
    }

    fn verification_level_code(&self) -> &'static str {
        self.consumer_boundary.verification_level_code()
    }

    pub(crate) fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "evidence": self.evidence,
            "status": self.status,
            "status_code": self.status_code,
            "decision": self.decision,
            "decision_code": self.decision_code(),
            "accepted_for_consumer": self.accepted_for_consumer(),
            "consumer_rejection_code": self.consumer_rejection_code(),
            "model_consumer_status_code": self.model_consumer_status_code(),
            "model_consumer_reason_code": self.model_consumer_reason_code(),
            "model_consumer_accepted": self.model_consumer_accepted(),
            "model_validated": self.model_validated(),
            "verification_level_code": self.verification_level_code(),
            "unknown_reason_code": self.unknown_reason_code,
            "unknown_limit_code": self.unknown_limit_code,
            "typed_consumer": self.typed_consumer,
            "model_consumer_decision": self.consumer_boundary.to_json(),
            "model_blocking_capability_evidence": self.model_blocking_capability.evidence_row(),
            "model_blocking_capability": self.model_blocking_capability.to_json(),
            "production_selected": false,
            "fail_closed": self.fail_closed(),
        })
    }
}

// ---------------------------------------------------------------------------
// SymbolicExplorer
// ---------------------------------------------------------------------------

/// Symbolic exploration engine wrapping a BmcTranslator with push/pop.
///
/// Provides interactive symbolic exploration of TLA+ specs using SMT solving.
/// Supports:
/// - Step-by-step symbolic transitions with solver push/pop for rollback
/// - Blocking clauses for enumerating alternate models (`next_model`)
/// - Concrete state assertions (`assume_state`)
/// - State compaction (extract + reset solver at current depth)
pub(crate) struct SymbolicExplorer {
    /// The BMC translator holding the ay solver.
    translator: BmcTranslator,
    /// Current symbolic depth (number of transitions applied).
    depth: usize,
    /// Maximum allowed depth.
    max_depth: usize,
    /// Stack of solver scopes for rollback. Each entry is the depth at push time.
    scope_stack: Vec<usize>,
    /// Variable sorts for this spec.
    var_sorts: Vec<(String, TlaSort)>,
    /// Expanded Init expression.
    #[cfg_attr(not(test), allow(dead_code))]
    init_expanded: Spanned<Expr>,
    /// Expanded Next expression.
    next_expanded: Spanned<Expr>,
    /// Invariants: (name, expanded_body).
    #[cfg_attr(not(test), allow(dead_code))]
    invariants: Vec<(String, Spanned<Expr>)>,
    /// Last extracted state (for blocking clause generation in next_model).
    last_model_state: Option<BmcState>,
    /// Last typed AY decision/profile summary captured from a solve call.
    last_solver_decision_profile: Option<SolveDecisionProfileSummary>,
}

impl SymbolicExplorer {
    /// Shared symbolic-execution detection for model enumeration/blocking.
    ///
    /// Model enumeration is intentionally reported as AY-required because
    /// blocking clauses must be built at the AY translator boundary.
    pub(crate) fn model_enumeration_detection() -> SymbolicExecutionDetection {
        SymbolicExecutionDetection::ay_required(SymbolicExecutionReason::ModelEnumeration)
    }

    /// Render model-enumeration evidence using the shared mc-core vocabulary.
    pub(crate) fn model_enumeration_evidence(scope: &str) -> String {
        Self::model_enumeration_detection().render_evidence(scope, ProblemKind::SymbolicExecution)
    }

    /// Render fail-closed AY decision/profile summary evidence.
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn solver_decision_profile_evidence(scope: &str) -> String {
        Self::symbolic_evidence_report(scope)
            .solver_decision_profile()
            .evidence_row()
            .to_string()
    }

    /// Render the fail-closed structured AY decision/profile summary boundary.
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn solver_decision_profile_report(scope: &str) -> AYSolveDecisionProfileEvidence {
        Self::symbolic_evidence_report(scope).solver_decision_profile
    }

    /// Render AY's public model-blocking capability descriptor evidence.
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn model_blocking_capability_evidence(scope: &str) -> String {
        Self::symbolic_evidence_report(scope)
            .model_blocking_capability()
            .evidence_row()
            .to_string()
    }

    /// Return AY's public model-blocking capability descriptor boundary.
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn model_blocking_capability_report(
        scope: &str,
    ) -> AYModelBlockingCapabilityEvidence {
        Self::symbolic_evidence_report(scope).model_blocking_capability
    }

    /// Return fail-closed symbolic evidence from one AY manifest adapter snapshot.
    pub(crate) fn symbolic_evidence_report(scope: &str) -> AYSymbolicEvidenceReport {
        AYSymbolicEvidenceAdapter::from_current_ay().symbolic_evidence_report(scope, None)
    }

    /// Render the last typed AY decision/profile summary, or fail closed if no solve ran.
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn current_solver_decision_profile_evidence(&self, scope: &str) -> String {
        self.current_symbolic_evidence_report(scope)
            .solver_decision_profile()
            .evidence_row()
            .to_string()
    }

    /// Return the last typed AY decision/profile summary boundary.
    pub(crate) fn current_solver_decision_profile_report(
        &self,
        scope: &str,
    ) -> AYSolveDecisionProfileEvidence {
        self.current_symbolic_evidence_report(scope)
            .solver_decision_profile
    }

    /// Return current symbolic evidence from one AY manifest adapter snapshot.
    pub(crate) fn current_symbolic_evidence_report(&self, scope: &str) -> AYSymbolicEvidenceReport {
        AYSymbolicEvidenceAdapter::from_current_ay()
            .symbolic_evidence_report(scope, self.last_solver_decision_profile.as_ref())
    }

    /// Create a new symbolic explorer from a loaded spec.
    ///
    /// Initializes the BMC translator, declares all variables, and resolves
    /// Init/Next/invariant expressions.
    pub(crate) fn new(
        module: &Module,
        config: &Config,
        ctx: &EvalCtx,
        max_depth: usize,
    ) -> Result<Self, SymbolicExploreError> {
        let symbolic_ctx = ay_shared::symbolic_ctx_with_config(ctx, config)
            .map_err(SymbolicExploreError::MissingSpec)?;

        let vars = ay_shared::collect_state_vars(module, &symbolic_ctx);
        if vars.is_empty() {
            return Err(SymbolicExploreError::MissingSpec(
                "No state variables declared".to_string(),
            ));
        }

        let resolved = ay_shared::resolve_init_next(config, &symbolic_ctx)
            .map_err(SymbolicExploreError::MissingSpec)?;

        let init_expr = ay_shared::get_operator_body(&symbolic_ctx, &resolved.init)
            .map_err(SymbolicExploreError::MissingSpec)?;
        let next_expr = ay_shared::get_operator_body(&symbolic_ctx, &resolved.next)
            .map_err(SymbolicExploreError::MissingSpec)?;

        let init_expanded = expand_operators_for_chc(&symbolic_ctx, &init_expr, false);
        let next_expanded = expand_operators_for_chc(&symbolic_ctx, &next_expr, true);

        let var_sorts =
            ay_shared::infer_var_sorts(&vars, &init_expanded, &config.invariants, &symbolic_ctx);

        // Resolve invariant expressions.
        let mut invariants = Vec::new();
        for inv_name in &config.invariants {
            let resolved_name = symbolic_ctx.resolve_op_name(inv_name);
            if let Ok(body) = ay_shared::get_operator_body(&symbolic_ctx, resolved_name) {
                let expanded = expand_operators_for_chc(&symbolic_ctx, &body, false);
                invariants.push((inv_name.clone(), expanded));
            }
        }

        // Create translator with enough capacity for max_depth transitions.
        let mut translator = make_translator(&var_sorts, max_depth)?;
        for (name, sort) in &var_sorts {
            translator.declare_var(name, sort.clone())?;
        }

        Ok(Self {
            translator,
            depth: 0,
            max_depth,
            scope_stack: Vec::new(),
            var_sorts,
            init_expanded,
            next_expanded,
            invariants,
            last_model_state: None,
            last_solver_decision_profile: None,
        })
    }

    /// Translate Init and assert it at step 0. Solve and return concrete initial states.
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn init(&mut self) -> Result<Vec<BmcState>, SymbolicExploreError> {
        self.depth = 0;
        let init_term = self.translator.translate_init(&self.init_expanded)?;
        self.translator.assert(init_term);

        match self.check_sat_with_decision_profile()? {
            SolveResult::Sat => {
                self.require_last_sat_model_accepted()?;
                let model = self.translator.try_get_model()?;
                let state = extract_state_at_step(&self.translator, &model, 0);
                self.last_model_state = Some(state.clone());
                Ok(vec![state])
            }
            SolveResult::Unsat(_) => Err(SymbolicExploreError::Unsatisfiable(
                "Init predicate is unsatisfiable".to_string(),
            )),
            _ => Err(SymbolicExploreError::SolverError(
                "Solver returned unknown for Init".to_string(),
            )),
        }
    }

    /// Assert Next transition from current depth to depth+1. Solve and return successor states.
    pub(crate) fn next_state(&mut self) -> Result<Vec<BmcState>, SymbolicExploreError> {
        if self.depth >= self.max_depth {
            return Err(SymbolicExploreError::SolverError(format!(
                "maximum depth {} reached",
                self.max_depth
            )));
        }

        let next_term = self
            .translator
            .translate_next(&self.next_expanded, self.depth)?;
        self.translator.assert(next_term);
        self.depth += 1;

        match self.check_sat_with_decision_profile()? {
            SolveResult::Sat => {
                self.require_last_sat_model_accepted()?;
                let model = self.translator.try_get_model()?;
                let state = extract_state_at_step(&self.translator, &model, self.depth);
                self.last_model_state = Some(state.clone());
                Ok(vec![state])
            }
            SolveResult::Unsat(_) => Ok(Vec::new()), // Deadlock
            _ => Err(SymbolicExploreError::SolverError(
                "Solver returned unknown for Next".to_string(),
            )),
        }
    }

    /// Check an invariant at the current depth.
    ///
    /// Returns `true` if the invariant holds, `false` if violated.
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn check_invariant(&mut self, inv_name: &str) -> Result<bool, SymbolicExploreError> {
        let inv_expr = self
            .invariants
            .iter()
            .find(|(name, _)| name == inv_name)
            .map(|(_, expr)| expr.clone())
            .ok_or_else(|| SymbolicExploreError::InvariantNotFound(inv_name.to_string()))?;

        // Push a scope so the invariant check does not pollute the main solver state.
        self.translator.push_scope()?;

        let not_inv = self
            .translator
            .translate_not_safety_at_step(&inv_expr, self.depth)?;
        self.translator.assert(not_inv);

        let result = match self.check_sat_with_decision_profile()? {
            SolveResult::Sat => {
                if let Err(error) = self.require_last_sat_model_accepted() {
                    self.translator.pop_scope()?;
                    return Err(error);
                }
                false // Found a state where invariant is violated.
            }
            SolveResult::Unsat(_) => true, // Invariant holds at current depth.
            _ => {
                self.translator.pop_scope()?;
                return Err(SymbolicExploreError::SolverError(
                    "Solver returned unknown for invariant check".to_string(),
                ));
            }
        };

        self.translator.pop_scope()?;
        Ok(result)
    }

    /// Push a solver scope (for rollback).
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn push(&mut self) -> Result<(), SymbolicExploreError> {
        self.translator.push_scope()?;
        self.scope_stack.push(self.depth);
        Ok(())
    }

    /// Pop a solver scope (rollback to previous state).
    pub(crate) fn pop(&mut self) -> Result<(), SymbolicExploreError> {
        let saved_depth = self
            .scope_stack
            .pop()
            .ok_or(SymbolicExploreError::ScopeUnderflow)?;
        self.translator.pop_scope()?;
        self.depth = saved_depth;
        self.last_model_state = None;
        self.last_solver_decision_profile = None;
        Ok(())
    }

    /// Assert concrete state constraints at current depth.
    pub(crate) fn assume_state(
        &mut self,
        assignments: &[(String, BmcValue)],
    ) -> Result<(), SymbolicExploreError> {
        self.translator
            .assert_concrete_state(assignments, self.depth)?;
        Ok(())
    }

    /// Get next satisfying model by adding a blocking clause for the current model.
    ///
    /// After a SAT result, adds a clause that negates the conjunction of all
    /// current variable assignments at the current depth, then re-solves.
    pub(crate) fn next_model(&mut self) -> Result<Option<BmcState>, SymbolicExploreError> {
        let last_state = match self.last_model_state.clone() {
            Some(state) => state,
            None => {
                return Err(SymbolicExploreError::SolverError(
                    "no previous model to block".to_string(),
                ));
            }
        };

        self.translator.block_concrete_state(&last_state)?;
        match self.check_sat_with_decision_profile()? {
            SolveResult::Sat => {
                self.require_last_sat_model_accepted()?;
                let model = self.translator.try_get_model()?;
                let state = extract_state_at_step(&self.translator, &model, self.depth);
                self.last_model_state = Some(state.clone());
                Ok(Some(state))
            }
            SolveResult::Unsat(_) => {
                self.last_model_state = None;
                Ok(None)
            }
            SolveResult::Unknown => Err(SymbolicExploreError::SolverError(
                "Solver returned unknown after blocking previous model".to_string(),
            )),
            _ => Err(SymbolicExploreError::SolverError(
                "Solver returned unsupported result after blocking previous model".to_string(),
            )),
        }
    }

    /// Extract concrete state from current model, reset solver, re-assert only concrete state.
    ///
    /// This "compacts" the solver by extracting the current concrete state,
    /// creating a fresh translator, and asserting just that concrete state.
    /// This removes accumulated symbolic constraints, reducing solver complexity.
    pub(crate) fn compact(&mut self) -> Result<BmcState, SymbolicExploreError> {
        let current_state = match &self.last_model_state {
            Some(s) => s.clone(),
            None => {
                return Err(SymbolicExploreError::SolverError(
                    "no current model to compact from".to_string(),
                ));
            }
        };

        // Create a fresh translator.
        let mut new_translator = make_translator(&self.var_sorts, self.max_depth)?;
        for (name, sort) in &self.var_sorts {
            new_translator.declare_var(name, sort.clone())?;
        }

        // Assert the concrete state at step 0.
        let assignments: Vec<(String, BmcValue)> = current_state
            .assignments
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
        new_translator.assert_concrete_state(&assignments, 0)?;

        // Replace the translator and reset depth.
        self.translator = new_translator;
        self.depth = 0;
        self.scope_stack.clear();
        self.last_solver_decision_profile = None;

        let compacted = BmcState {
            step: 0,
            assignments: current_state.assignments,
        };
        self.last_model_state = Some(compacted.clone());
        Ok(compacted)
    }

    /// Apply transitions in sequence, returning the state after each step.
    ///
    /// Note: `action_names` is accepted for API compatibility but not used
    /// for filtering in the current implementation — all enabled Next transitions
    /// are considered at each step. Action-specific filtering would require
    /// disjunct decomposition of the Next relation.
    pub(crate) fn apply_in_order(
        &mut self,
        _action_names: &[String],
        steps: usize,
    ) -> Result<Vec<BmcState>, SymbolicExploreError> {
        let mut trace = Vec::new();

        for _ in 0..steps {
            let successors = self.next_state()?;
            if successors.is_empty() {
                break; // Deadlock.
            }
            trace.push(successors.into_iter().next().expect("non-empty checked"));
        }

        Ok(trace)
    }

    /// Get the current symbolic depth.
    pub(crate) fn current_depth(&self) -> usize {
        self.depth
    }

    fn check_sat_with_decision_profile(&mut self) -> Result<SolveResult, SymbolicExploreError> {
        self.last_solver_decision_profile = None;
        let (result, summary) = self
            .translator
            .try_check_sat_with_decision_profile_summary()?;
        self.last_solver_decision_profile = Some(summary);
        Ok(result)
    }

    fn require_last_sat_model_accepted(&self) -> Result<(), SymbolicExploreError> {
        let profile = self.current_solver_decision_profile_report("TLA");
        if profile.accepts_model_for_tla_boundary() {
            return Ok(());
        }

        Err(SymbolicExploreError::SolverError(format!(
            "AY SAT result rejected by consumer boundary: decision_code={} model_consumer_status_code={} model_consumer_reason_code={} accepted_for_consumer={} consumer_rejection_code={} model_validated={} fail_closed={}",
            profile.decision_code(),
            profile.model_consumer_status_code(),
            profile.model_consumer_reason_code(),
            profile.accepted_for_consumer(),
            profile.consumer_rejection_code(),
            profile.model_validated(),
            profile.fail_closed(),
        )))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;
    use crate::eval::EvalCtx;
    use crate::test_support::parse_module;

    fn make_config(init: &str, next: &str, invariants: &[&str]) -> Config {
        Config {
            init: Some(init.to_string()),
            next: Some(next.to_string()),
            invariants: invariants.iter().map(|s| s.to_string()).collect(),
            ..Default::default()
        }
    }

    fn ay_model_blocking_contract() -> tla_ay::SolverCapabilityContract {
        tla_ay::model_blocking_symbolic_execution_contract()
    }

    fn assert_evidence_contains_kv(evidence: &str, key: &str, value: impl ToString) {
        let value = value.to_string();
        let value = if value.is_empty() {
            NO_REASON_CODE.to_string()
        } else {
            value
        };
        assert!(
            evidence.contains(&format!("{key}={value}")),
            "evidence should forward {key} from AY manifest: {evidence}"
        );
    }

    fn assert_model_blocking_evidence_matches_ay_manifest(
        evidence: &str,
        json: &serde_json::Value,
    ) {
        let manifest = tla_ay::symbolic_execution_contract_manifest();
        let manifest_pairs = tla_ay::symbolic_execution_contract_manifest_key_value_pairs();
        let health = tla_ay::validate_symbolic_execution_contract_manifest_round_trip(
            &manifest,
            &manifest_pairs,
        );
        let diagnostic_summary =
            tla_ay::symbolic_execution_contract_manifest_diagnostic_summary_for_round_trip(
                &manifest,
                &manifest_pairs,
            );
        let route_admission =
            tla_ay::symbolic_execution_route_admission_decision_for_summary(&diagnostic_summary);
        let model_blocking_contract = ay_model_blocking_contract();
        let incremental_contract = tla_ay::incremental_assumptions_symbolic_execution_contract();
        let all_sat_contract = tla_ay::all_sat_enumeration_symbolic_execution_contract();
        let model_blocking_readiness =
            tla_ay::symbolic_execution_capability_route_readiness_for_decision(
                tla_ay::SolverCapabilityCode::ModelBlocking,
                &route_admission,
            );
        let incremental_readiness =
            tla_ay::symbolic_execution_capability_route_readiness_for_decision(
                tla_ay::SolverCapabilityCode::IncrementalAssumptions,
                &route_admission,
            );
        let all_sat_readiness = tla_ay::symbolic_execution_capability_route_readiness_for_decision(
            tla_ay::SolverCapabilityCode::AllSatEnumeration,
            &route_admission,
        );
        let all_supported_readiness =
            tla_ay::symbolic_execution_all_supported_capability_route_readiness_for_decision(
                &route_admission,
            );
        assert_eq!(all_supported_readiness.len(), 3);
        assert!(
            tla_ay::validate_symbolic_execution_all_supported_capability_route_readiness(
                &all_supported_readiness
            )
            .iter()
            .all(|readiness| readiness.accepted_for_consumer)
        );

        assert!(evidence.contains("TLA ay_symbolic_execution_contract_manifest"));
        for (key, value) in manifest_pairs {
            assert_evidence_contains_kv(evidence, key, value);
        }
        for (key, value) in health.to_key_value_pairs() {
            assert_evidence_contains_kv(evidence, &format!("health_{key}"), value);
        }
        for (key, value) in diagnostic_summary.to_key_value_rows() {
            assert_evidence_contains_kv(evidence, &format!("diagnostic_summary_{key}"), value);
        }
        for (key, value) in route_admission.to_key_value_rows() {
            assert_evidence_contains_kv(evidence, &format!("route_admission_{key}"), value);
        }
        for (key, value) in model_blocking_contract.to_key_value_pairs() {
            assert_evidence_contains_kv(evidence, &format!("model_blocking_contract_{key}"), value);
        }
        for readiness in [
            &model_blocking_readiness,
            &incremental_readiness,
            &all_sat_readiness,
        ] {
            for (key, value) in readiness.to_key_value_rows() {
                assert_evidence_contains_kv(
                    evidence,
                    &format!("{}_route_readiness_{key}", readiness.capability_code),
                    value,
                );
            }
        }
        assert_evidence_contains_kv(evidence, "model_blocking_ready", true);
        assert_evidence_contains_kv(evidence, "incremental_assumptions_ready", true);
        assert_evidence_contains_kv(evidence, "all_sat_enumeration_ready", true);
        assert_evidence_contains_kv(
            evidence,
            "capability",
            model_blocking_contract.capability_code,
        );
        assert_evidence_contains_kv(
            evidence,
            "status_code",
            model_blocking_readiness.status_code,
        );
        assert_evidence_contains_kv(
            evidence,
            "reason_code",
            model_blocking_readiness.reason_code,
        );
        assert_evidence_contains_kv(evidence, "typed_consumer", true);
        assert_evidence_contains_kv(evidence, "production_selected", false);
        assert_evidence_contains_kv(evidence, "manifest_valid", true);
        assert_evidence_contains_kv(
            evidence,
            "fail_closed",
            model_blocking_readiness.fail_closed,
        );

        assert_eq!(json["manifest_schema"], manifest.schema);
        assert_eq!(json["manifest_schema_version"], manifest.schema_version);
        assert_eq!(json["solver"], manifest.solver);
        assert_eq!(json["contract_count"], manifest.contracts.len());
        assert_eq!(json["capability"], model_blocking_contract.capability_code);
        assert_eq!(json["status_code"], model_blocking_readiness.status_code);
        assert_eq!(json["reason_code"], model_blocking_readiness.reason_code);
        assert_eq!(json["typed_consumer"], true);
        assert_eq!(json["manifest_valid"], true);
        assert_eq!(
            json["all_contracts_fail_closed"],
            manifest.all_contracts_fail_closed
        );
        assert_eq!(json["fail_closed"], model_blocking_readiness.fail_closed);
        assert_eq!(json["capability_manifest"], manifest.to_json_value());
        assert_eq!(
            json["symbolic_execution_contract_manifest"],
            manifest.to_json_value()
        );
        assert_eq!(
            json["symbolic_execution_contract_manifest_health"],
            health.to_json_value()
        );
        assert_eq!(
            json["symbolic_execution_contract_diagnostic_summary"],
            diagnostic_summary.to_json_value()
        );
        assert_eq!(
            json["symbolic_execution_route_admission"],
            route_admission.to_json_value()
        );
        assert_eq!(
            json["symbolic_execution_all_supported_capability_route_readiness"],
            serde_json::json!(all_supported_readiness
                .iter()
                .map(|readiness| readiness.to_json_value())
                .collect::<Vec<_>>())
        );
        assert_eq!(json["all_supported_capability_route_readiness_ready"], true);
        let route_readiness_evidence = json
            ["symbolic_execution_all_supported_capability_route_readiness_evidence"]
            .as_str()
            .expect("all-supported readiness evidence");
        assert!(route_readiness_evidence
            .contains("TLA ay_symbolic_execution_all_supported_capability_route_readiness"));
        assert!(route_readiness_evidence.contains("model_blocking_status=ready"));
        assert!(route_readiness_evidence
            .contains("model_blocking_reason=ay_authoritative_capability_route"));
        assert!(route_readiness_evidence.contains("model_blocking_accepted_for_consumer=true"));
        assert!(route_readiness_evidence.contains("model_blocking_fail_closed=true"));
        assert!(route_readiness_evidence.contains("incremental_assumptions_status=ready"));
        assert!(route_readiness_evidence.contains("all_sat_enumeration_status=ready"));
        assert_eq!(json["health_status_code"], health.status_code);
        assert_eq!(json["health_reason_code"], health.reason_code);
        assert_eq!(json["health_diagnostic_code"], health.diagnostic_code());
        assert_eq!(
            json["accepted_for_consumer"],
            model_blocking_readiness.accepted_for_consumer
        );
        assert_eq!(json["route_admission_schema"], route_admission.schema);
        assert_eq!(
            json["route_admission_schema_version"],
            route_admission.schema_version
        );
        assert_eq!(
            json["route_admission_status_code"],
            route_admission.status_code
        );
        assert_eq!(
            json["route_admission_reason_code"],
            route_admission.reason_code
        );
        assert_eq!(
            json["route_admission_accepted_for_consumer"],
            route_admission.accepted_for_consumer
        );
        assert_eq!(
            json["route_admission_fail_closed"],
            route_admission.fail_closed
        );
        assert_eq!(
            json["route_capabilities"],
            serde_json::json!(&route_admission.route_capabilities)
        );
        assert_eq!(
            json["route_authorities"],
            serde_json::json!(&route_admission.route_authorities)
        );
        assert_eq!(
            json["route_validators"],
            serde_json::json!(route_admission.validators)
        );
        assert_eq!(
            json["required_capabilities"],
            serde_json::json!(&health.required_capabilities)
        );
        assert_eq!(
            json["present_capabilities"],
            serde_json::json!(&health.present_capabilities)
        );
        assert_eq!(json["model_blocking_ready"], true);
        assert_eq!(json["incremental_assumptions_ready"], true);
        assert_eq!(json["all_sat_enumeration_ready"], true);
        assert_eq!(
            json["model_blocking_capability_route_readiness"],
            model_blocking_readiness.to_json_value()
        );
        assert_eq!(
            json["incremental_assumptions_capability_route_readiness"],
            incremental_readiness.to_json_value()
        );
        assert_eq!(
            json["all_sat_enumeration_capability_route_readiness"],
            all_sat_readiness.to_json_value()
        );

        let contract_capabilities = json["contract_capabilities"]
            .as_array()
            .expect("contract_capabilities array");
        for contract in [
            model_blocking_contract,
            incremental_contract,
            all_sat_contract,
        ] {
            assert!(
                contract_capabilities
                    .iter()
                    .any(|value| value == contract.capability_code),
                "missing symbolic contract capability {}",
                contract.capability_code
            );
        }

        let api_symbols = json["api_symbols"].as_array().expect("api_symbols array");
        for symbol in model_blocking_readiness.api_symbols {
            assert!(
                api_symbols.iter().any(|value| value == *symbol),
                "missing AY readiness API symbol {symbol}"
            );
        }

        let evidence_schemas = json["evidence_schemas"]
            .as_array()
            .expect("evidence_schemas array");
        for schema in model_blocking_readiness.evidence_schemas {
            assert!(
                evidence_schemas.iter().any(|value| value == *schema),
                "missing AY readiness evidence schema {schema}"
            );
        }

        let consumer_responsibilities = json["consumer_responsibilities"]
            .as_array()
            .expect("consumer_responsibilities array");
        for responsibility in model_blocking_readiness.consumer_responsibilities {
            assert!(
                consumer_responsibilities
                    .iter()
                    .any(|value| value == *responsibility),
                "missing AY readiness consumer responsibility {responsibility}"
            );
        }

        let readiness_rows = json["symbolic_execution_contracts"]
            .as_array()
            .expect("symbolic_execution_contracts array");
        for contract in [
            model_blocking_contract,
            incremental_contract,
            all_sat_contract,
        ] {
            let row = readiness_rows
                .iter()
                .find(|row| row["capability"] == contract.capability_code)
                .expect("contract readiness row");
            let expected_readiness = all_supported_readiness
                .iter()
                .find(|readiness| readiness.capability_code == contract.capability_code)
                .expect("AY all-supported readiness should include contract capability");
            assert_eq!(row, &expected_readiness.to_json_value());
            assert_eq!(row["status"], "ready");
            assert_eq!(row["reason"], "ay_authoritative_capability_route");
        }
    }

    #[test]
    fn test_symbolic_explorer_init() {
        let src = r#"
---- MODULE SymExploreInit ----
VARIABLE x
Init == x \in {0, 1, 2}
Next == x' = x
TypeOK == x \in {0, 1, 2}
====
"#;
        let module = parse_module(src);
        let config = make_config("Init", "Next", &["TypeOK"]);
        let mut ctx = EvalCtx::new();
        ctx.load_module(&module);

        let mut explorer =
            SymbolicExplorer::new(&module, &config, &ctx, 20).expect("should create explorer");

        let states = explorer.init().expect("init should succeed");
        assert!(!states.is_empty(), "should produce at least one init state");
        assert_eq!(explorer.current_depth(), 0);
    }

    #[test]
    fn test_symbolic_explorer_step() {
        let src = r#"
---- MODULE SymExploreStep ----
VARIABLE x
Init == x = 0
Next == x' = x + 1
TypeOK == x \in 0..10
====
"#;
        let module = parse_module(src);
        let config = make_config("Init", "Next", &["TypeOK"]);
        let mut ctx = EvalCtx::new();
        ctx.load_module(&module);

        let mut explorer =
            SymbolicExplorer::new(&module, &config, &ctx, 20).expect("should create explorer");

        let init_states = explorer.init().expect("init should succeed");
        assert_eq!(init_states.len(), 1);

        // Assert the init state concretely so the next step is deterministic.
        let init_assignments: Vec<(String, BmcValue)> = init_states[0]
            .assignments
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
        explorer
            .assume_state(&init_assignments)
            .expect("assume_state");

        let next_states = explorer.next_state().expect("step should succeed");
        assert!(!next_states.is_empty(), "should have successors");
        assert_eq!(explorer.current_depth(), 1);
    }

    #[test]
    fn test_symbolic_explorer_push_pop() {
        let src = r#"
---- MODULE SymExplorePushPop ----
VARIABLE x
Init == x = 0
Next == x' = x + 1
TypeOK == x \in 0..10
====
"#;
        let module = parse_module(src);
        let config = make_config("Init", "Next", &["TypeOK"]);
        let mut ctx = EvalCtx::new();
        ctx.load_module(&module);

        let mut explorer =
            SymbolicExplorer::new(&module, &config, &ctx, 20).expect("should create explorer");

        let _ = explorer.init().expect("init");
        assert_eq!(explorer.current_depth(), 0);

        explorer.push().expect("push");
        let _ = explorer.next_state().expect("step");
        assert_eq!(explorer.current_depth(), 1);

        explorer.pop().expect("pop");
        assert_eq!(explorer.current_depth(), 0);
    }

    #[test]
    fn test_symbolic_explorer_check_invariant() {
        let src = r#"
---- MODULE SymExploreInv ----
VARIABLE x
Init == x = 0
Next == x' = x + 1
TypeOK == x >= 0
====
"#;
        let module = parse_module(src);
        let config = make_config("Init", "Next", &["TypeOK"]);
        let mut ctx = EvalCtx::new();
        ctx.load_module(&module);

        let mut explorer =
            SymbolicExplorer::new(&module, &config, &ctx, 20).expect("should create explorer");

        let init_states = explorer.init().expect("init");
        let init_assignments: Vec<(String, BmcValue)> = init_states[0]
            .assignments
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
        explorer.assume_state(&init_assignments).expect("assume");

        let holds = explorer.check_invariant("TypeOK").expect("check_invariant");
        assert!(holds, "TypeOK should hold for x=0");
    }

    #[test]
    fn test_symbolic_explorer_compact() {
        let src = r#"
---- MODULE SymExploreCompact ----
VARIABLE x
Init == x = 0
Next == x' = x + 1
TypeOK == x >= 0
====
"#;
        let module = parse_module(src);
        let config = make_config("Init", "Next", &["TypeOK"]);
        let mut ctx = EvalCtx::new();
        ctx.load_module(&module);

        let mut explorer =
            SymbolicExplorer::new(&module, &config, &ctx, 20).expect("should create explorer");

        let _ = explorer.init().expect("init");
        let compacted = explorer.compact().expect("compact");
        assert_eq!(compacted.step, 0);
        assert_eq!(explorer.current_depth(), 0);
    }

    #[test]
    fn test_symbolic_explorer_scope_underflow() {
        let src = r#"
---- MODULE SymExploreUnderflow ----
VARIABLE x
Init == x = 0
Next == x' = x
====
"#;
        let module = parse_module(src);
        let config = make_config("Init", "Next", &[]);
        let mut ctx = EvalCtx::new();
        ctx.load_module(&module);

        let mut explorer =
            SymbolicExplorer::new(&module, &config, &ctx, 20).expect("should create explorer");

        let result = explorer.pop();
        assert!(result.is_err(), "pop without push should fail");
    }

    #[test]
    fn test_symbolic_explorer_next_model_blocks_previous_state() {
        let src = r#"
---- MODULE SymExploreNextModel ----
VARIABLE x
Init == x \in {0, 1}
Next == x' = x
====
"#;
        let module = parse_module(src);
        let config = make_config("Init", "Next", &[]);
        let mut ctx = EvalCtx::new();
        ctx.load_module(&module);

        let mut explorer =
            SymbolicExplorer::new(&module, &config, &ctx, 20).expect("should create explorer");

        let states = explorer.init().expect("init should succeed");
        let first = states[0]
            .assignments
            .get("x")
            .cloned()
            .expect("first model should assign x");
        let second = explorer
            .next_model()
            .expect("second model should solve")
            .expect("second model should exist");
        assert_ne!(
            second.assignments.get("x"),
            Some(&first),
            "blocking clause should exclude the previous model"
        );

        let third = explorer.next_model().expect("third model should solve");
        assert!(third.is_none(), "two-state Init should be exhausted");
    }

    #[test]
    fn test_symbolic_explorer_model_enumeration_evidence_uses_shared_vocabulary() {
        let detection = SymbolicExplorer::model_enumeration_detection();
        assert!(detection.prefers_ay());
        assert!(detection.requires_ay());
        assert_eq!(
            SymbolicExplorer::model_enumeration_evidence("TLA"),
            "TLA symbolic_execution domain=tla status=AYRequired status_code=ay_required problem=SymbolicExecution reason=ModelEnumeration reason_code=model_enumeration preferred_backend=AYSmt preferred_backend_code=ay_smt"
        );
    }

    #[test]
    fn test_symbolic_explorer_exposes_ay_model_blocking_capability() {
        let evidence = SymbolicExplorer::model_blocking_capability_evidence("TLA");
        let report = SymbolicExplorer::model_blocking_capability_report("TLA");
        let json = report.to_json();

        assert!(evidence.contains("TLA ay_symbolic_execution_contract_manifest"));
        assert_model_blocking_evidence_matches_ay_manifest(&evidence, &json);
    }

    #[test]
    fn test_symbolic_explorer_decision_profiles_use_shared_model_blocking_hook() {
        let symbolic_report = SymbolicExplorer::symbolic_evidence_report("TLA");
        let direct_report = symbolic_report.model_blocking_capability();
        let expected_evidence = direct_report.evidence_row();
        let expected_json = direct_report.to_json();

        let missing_profile = symbolic_report.solver_decision_profile().to_json();
        assert_eq!(
            missing_profile["model_blocking_capability_evidence"]
                .as_str()
                .expect("missing profile should expose model-blocking evidence"),
            expected_evidence
        );
        assert_eq!(missing_profile["model_blocking_capability"], expected_json);

        let typed_profile = AYSolveDecisionProfileEvidence::from_typed_fields_for_testing(
            tla_ay::SolveDecision::Sat,
            true,
            None,
            true,
        )
        .to_json();
        assert_eq!(
            typed_profile["model_blocking_capability_evidence"]
                .as_str()
                .expect("typed profile should expose model-blocking evidence"),
            expected_evidence
        );
        assert_eq!(typed_profile["model_blocking_capability"], expected_json);
    }

    #[test]
    fn test_symbolic_evidence_adapter_fails_closed_without_aggregate_rows() {
        let adapter = AYSymbolicEvidenceAdapter::from_parts_for_testing(
            tla_ay::symbolic_execution_contract_manifest(),
            Vec::new(),
        );
        let report = adapter.symbolic_evidence_report("TLA", None);
        let blocking_report = report.model_blocking_capability();
        let blocking_json = blocking_report.to_json();

        assert!(blocking_report
            .evidence_row()
            .contains("typed_consumer=false"));
        assert!(blocking_report
            .evidence_row()
            .contains("manifest_valid=false"));
        assert!(blocking_report
            .evidence_row()
            .contains("route_admission_status=blocked"));
        assert!(blocking_report
            .evidence_row()
            .contains("route_admission_accepted_for_consumer=false"));
        assert!(blocking_report.evidence_row().contains("fail_closed=true"));
        assert_eq!(blocking_json["status_code"], "blocked");
        assert_eq!(blocking_json["reason_code"], "route_admission_blocked");
        assert_eq!(blocking_json["typed_consumer"], false);
        assert_eq!(blocking_json["manifest_valid"], false);
        assert_eq!(blocking_json["fail_closed"], true);
        assert_eq!(blocking_json["accepted_for_consumer"], false);
        assert_eq!(blocking_json["route_admission_status_code"], "blocked");
        assert_eq!(
            blocking_json["route_admission_accepted_for_consumer"],
            false
        );
        assert_eq!(blocking_json["route_admission_fail_closed"], true);
        assert_eq!(
            blocking_json["symbolic_execution_route_admission"]["status"],
            "blocked"
        );
        assert_eq!(blocking_json["model_blocking_ready"], false);
        assert_eq!(blocking_json["incremental_assumptions_ready"], false);
        assert_eq!(blocking_json["all_sat_enumeration_ready"], false);
        assert_eq!(
            blocking_json["all_supported_capability_route_readiness_ready"],
            false
        );
        let readiness_rows = blocking_json
            ["symbolic_execution_all_supported_capability_route_readiness"]
            .as_array()
            .expect("all-supported readiness rows");
        assert_eq!(readiness_rows.len(), 3);
        for row in readiness_rows {
            assert_eq!(row["status"], "blocked");
            assert_eq!(row["reason"], "route_admission_blocked");
            assert_eq!(row["accepted_for_consumer"], false);
            assert_eq!(row["fail_closed"], true);
            assert_eq!(row["route_admission_status"], "blocked");
            assert_eq!(row["issue_field"], "route_admission_status");
        }
        let readiness_evidence = blocking_json
            ["symbolic_execution_all_supported_capability_route_readiness_evidence"]
            .as_str()
            .expect("all-supported readiness evidence");
        assert!(readiness_evidence.contains("model_blocking_status=blocked"));
        assert!(readiness_evidence.contains("model_blocking_reason=route_admission_blocked"));
        assert!(readiness_evidence.contains("model_blocking_accepted_for_consumer=false"));
        assert!(readiness_evidence.contains("incremental_assumptions_status=blocked"));
        assert!(readiness_evidence.contains("all_sat_enumeration_status=blocked"));
        assert!(report.solver_decision_profile().fail_closed());
        assert_eq!(
            report.solver_decision_profile().to_json()["model_blocking_capability"],
            blocking_json
        );
    }

    #[test]
    fn test_symbolic_evidence_adapter_fails_closed_for_stale_manifest_pairs() {
        let manifest = tla_ay::symbolic_execution_contract_manifest();
        let mut stale_pairs = tla_ay::symbolic_execution_contract_manifest_key_value_pairs();
        let schema_version = stale_pairs
            .iter_mut()
            .find(|(key, _)| *key == "schema_version")
            .expect("schema_version pair should exist");
        schema_version.1 = "0".to_string();

        let adapter = AYSymbolicEvidenceAdapter::from_parts_for_testing(manifest, stale_pairs);
        let report = adapter.symbolic_evidence_report("TLA", None);
        let blocking_report = report.model_blocking_capability();
        let blocking_json = blocking_report.to_json();

        assert!(blocking_report
            .evidence_row()
            .contains("typed_consumer=false"));
        assert!(blocking_report
            .evidence_row()
            .contains("manifest_valid=false"));
        assert!(blocking_report
            .evidence_row()
            .contains("route_admission_status=blocked"));
        assert!(blocking_report
            .evidence_row()
            .contains("route_admission_accepted_for_consumer=false"));
        assert!(blocking_report.evidence_row().contains("fail_closed=true"));
        assert_eq!(blocking_json["status_code"], "blocked");
        assert_eq!(blocking_json["reason_code"], "route_admission_blocked");
        assert_eq!(blocking_json["typed_consumer"], false);
        assert_eq!(blocking_json["manifest_valid"], false);
        assert_eq!(blocking_json["fail_closed"], true);
        assert_eq!(blocking_json["accepted_for_consumer"], false);
        assert_eq!(blocking_json["route_admission_status_code"], "blocked");
        assert_eq!(
            blocking_json["route_admission_accepted_for_consumer"],
            false
        );
        assert_eq!(blocking_json["route_admission_fail_closed"], true);
        assert_eq!(
            blocking_json["symbolic_execution_route_admission"]["status"],
            "blocked"
        );
        assert_eq!(blocking_json["model_blocking_ready"], false);
        assert_eq!(blocking_json["incremental_assumptions_ready"], false);
        assert_eq!(blocking_json["all_sat_enumeration_ready"], false);
        assert_eq!(
            blocking_json["all_supported_capability_route_readiness_ready"],
            false
        );
        let readiness_rows = blocking_json
            ["symbolic_execution_all_supported_capability_route_readiness"]
            .as_array()
            .expect("all-supported readiness rows");
        assert_eq!(readiness_rows.len(), 3);
        for row in readiness_rows {
            assert_eq!(row["status"], "blocked");
            assert_eq!(row["reason"], "route_admission_blocked");
            assert_eq!(row["accepted_for_consumer"], false);
            assert_eq!(row["fail_closed"], true);
            assert_eq!(row["route_admission_status"], "blocked");
            assert_eq!(row["issue_field"], "route_admission_status");
        }
        let readiness_evidence = blocking_json
            ["symbolic_execution_all_supported_capability_route_readiness_evidence"]
            .as_str()
            .expect("all-supported readiness evidence");
        assert!(readiness_evidence.contains("model_blocking_status=blocked"));
        assert!(readiness_evidence.contains("model_blocking_reason=route_admission_blocked"));
        assert!(readiness_evidence.contains("model_blocking_accepted_for_consumer=false"));
        assert!(readiness_evidence.contains("incremental_assumptions_status=blocked"));
        assert!(readiness_evidence.contains("all_sat_enumeration_status=blocked"));
        assert!(report.solver_decision_profile().fail_closed());
        assert!(!report
            .solver_decision_profile()
            .accepts_model_for_tla_boundary());
        assert_eq!(
            report.solver_decision_profile().to_json()["model_blocking_capability"],
            blocking_json
        );
    }

    #[test]
    fn test_symbolic_explorer_ay_decision_profile_summary_missing_evidence_fails_closed() {
        let evidence = SymbolicExplorer::solver_decision_profile_evidence("TLA");
        let report = SymbolicExplorer::solver_decision_profile_report("TLA");
        let json = report.to_json();

        assert!(evidence.contains("TLA ay_solver_decision_profile_summary"));
        assert!(evidence.contains("status_code=missing_typed_summary"));
        assert!(evidence.contains("typed_consumer=false"));
        assert!(evidence.contains(&format!(
            "expected_schema={}",
            tla_ay::AY_SOLVE_DECISION_PROFILE_SUMMARY_SCHEMA
        )));
        assert!(evidence.contains(&format!(
            "expected_schema_version={}",
            tla_ay::AY_SOLVE_DECISION_PROFILE_SUMMARY_SCHEMA_VERSION
        )));
        assert!(evidence.contains(&format!(
            "expected_fields={}",
            tla_ay::AY_SOLVE_DECISION_PROFILE_SUMMARY_EXPECTED_FIELDS
        )));
        assert!(evidence.contains("production_selected=false"));
        assert!(evidence.contains("fail_closed=true"));
        assert!(!report.accepts_model_for_tla_boundary());
        assert!(report.fail_closed());
        assert_eq!(report.status_code, "missing_typed_summary");
        assert_eq!(report.consumer_rejection_code(), "missing_typed_summary");
        assert_eq!(report.model_consumer_status_code(), "rejected");
        assert_eq!(report.model_consumer_reason_code(), "missing_typed_summary");
        assert!(!report.model_consumer_accepted());
        assert_eq!(json["model_consumer_decision"]["status"], "rejected");
        assert_eq!(
            json["model_consumer_decision"]["reason"],
            "missing_typed_summary"
        );
        assert_eq!(json["model_consumer_decision"]["fail_closed"], true);
        let blocking_evidence = json["model_blocking_capability_evidence"]
            .as_str()
            .expect("model-blocking capability evidence should be a string");
        assert_model_blocking_evidence_matches_ay_manifest(
            blocking_evidence,
            &json["model_blocking_capability"],
        );
    }

    #[test]
    fn test_symbolic_explorer_ay_decision_profile_summary_uses_typed_solve_data() {
        let src = r#"
---- MODULE SymExploreTypedProfile ----
VARIABLE x
Init == x = 0
Next == x' = x
====
"#;
        let module = parse_module(src);
        let config = make_config("Init", "Next", &[]);
        let mut ctx = EvalCtx::new();
        ctx.load_module(&module);

        let mut explorer =
            SymbolicExplorer::new(&module, &config, &ctx, 20).expect("should create explorer");

        explorer.init().expect("init should succeed");
        let evidence = explorer.current_solver_decision_profile_evidence("TLA");
        let report = explorer.current_solver_decision_profile_report("TLA");
        let json = report.to_json();
        let expected_consumer_decision = explorer
            .last_solver_decision_profile
            .as_ref()
            .expect("init should record a typed AY summary")
            .model_consumer_decision_json();

        assert!(evidence.contains("TLA ay_solver_decision_profile_summary"));
        assert!(evidence.contains("status_code=typed_summary_available"));
        assert!(evidence.contains(&format!(
            "schema={}",
            tla_ay::AY_SOLVE_DECISION_PROFILE_SUMMARY_SCHEMA
        )));
        assert!(evidence.contains(&format!(
            "schema_version={}",
            tla_ay::AY_SOLVE_DECISION_PROFILE_SUMMARY_SCHEMA_VERSION
        )));
        assert!(evidence.contains("decision=SAT"));
        assert!(evidence.contains("decision_code=sat"));
        assert!(evidence.contains("typed_consumer=true"));
        assert!(evidence.contains("production_selected=false"));
        assert!(evidence.contains("fail_closed=false"));
        assert!(report.accepts_model_for_tla_boundary());
        assert!(!report.fail_closed());
        assert!(report.accepted_for_consumer());
        assert!(report.model_validated());
        assert_eq!(report.consumer_rejection_code(), NO_REASON_CODE);
        assert_eq!(report.model_consumer_status_code(), "accepted");
        assert_eq!(report.model_consumer_reason_code(), "accepted");
        assert!(report.model_consumer_accepted());
        assert_eq!(json["model_consumer_decision"], expected_consumer_decision);
        let blocking_evidence = json["model_blocking_capability_evidence"]
            .as_str()
            .expect("model-blocking capability evidence should be a string");
        assert_model_blocking_evidence_matches_ay_manifest(
            blocking_evidence,
            &json["model_blocking_capability"],
        );
    }

    #[test]
    fn test_tla_ay_decision_profile_boundary_fails_closed_for_rejected_sat() {
        let report = AYSolveDecisionProfileEvidence::from_typed_fields_for_testing(
            tla_ay::SolveDecision::Sat,
            false,
            Some("sat_model_not_validated"),
            false,
        );

        assert!(report.fail_closed());
        assert!(!report.accepts_model_for_tla_boundary());
        assert_eq!(report.decision_code(), "sat");
        assert_eq!(report.consumer_rejection_code(), "sat_model_not_validated");
        assert!(!report.accepted_for_consumer());
        assert!(!report.model_validated());
        assert_eq!(report.model_consumer_status_code(), "rejected");
        assert_eq!(report.model_consumer_reason_code(), "consumer_rejected");
        assert!(!report.model_consumer_accepted());
        assert!(report.evidence_row().contains("fail_closed=true"));
    }

    #[test]
    fn test_tla_ay_decision_profile_boundary_fails_closed_for_unvalidated_sat() {
        let report = AYSolveDecisionProfileEvidence::from_typed_fields_for_testing(
            tla_ay::SolveDecision::Sat,
            true,
            None,
            false,
        );

        assert!(report.fail_closed());
        assert!(!report.accepts_model_for_tla_boundary());
        assert_eq!(report.decision_code(), "sat");
        assert_eq!(report.consumer_rejection_code(), NO_REASON_CODE);
        assert!(report.accepted_for_consumer());
        assert!(!report.model_validated());
        assert_eq!(report.model_consumer_status_code(), "rejected");
        assert_eq!(report.model_consumer_reason_code(), "model_not_validated");
        assert!(!report.model_consumer_accepted());
        assert!(report.evidence_row().contains("fail_closed=true"));
    }

    #[test]
    fn test_tla_ay_decision_profile_boundary_fails_closed_for_unknown_summary() {
        let mut solver = tla_ay::Solver::try_new(tla_ay::Logic::QfLia).expect("solver");
        solver.set_timeout(Some(std::time::Duration::ZERO));
        let details = solver.try_check_sat_with_details().expect("solve details");
        let summary = details.decision_profile_summary();
        let report = AYSolveDecisionProfileEvidence::from_summary("TLA", Some(&summary));
        let json = report.to_json();

        assert!(report.fail_closed());
        assert!(!report.accepts_model_for_tla_boundary());
        assert_eq!(report.decision_code(), "unknown");
        assert_eq!(report.unknown_reason_code, "timeout");
        assert_eq!(report.unknown_limit_code, "timeout");
        assert!(report.accepted_for_consumer());
        assert!(!report.model_validated());
        assert_eq!(report.model_consumer_status_code(), "rejected");
        assert_eq!(report.model_consumer_reason_code(), "non_sat_decision");
        assert!(!report.model_consumer_accepted());
        assert_eq!(
            json["model_consumer_decision"],
            summary.model_consumer_decision_json()
        );
        assert!(report.evidence_row().contains("fail_closed=true"));
    }
}
