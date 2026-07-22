// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0
#![forbid(unsafe_code)]

//! TLA+ to ay SMT Solver Integration
//!
//! This crate provides constraint-based Init state enumeration using ay,
//! our pure-Rust SMT solver. It enables TY to handle specs with complex
//! Init predicates that defeat brute-force enumeration.
//!
//! The `shared_engine` module also exposes frontend-neutral metadata for the
//! same AY BMC/CHC/PDR/k-induction lanes when other frontends lower to generic
//! transition-system and proof-obligation contracts.
//!
//! # Background (Part of #251)
//!
//! Some TLA+ specs have Init predicates too complex for direct enumeration:
//! - Einstein: `Permutation(S)` creates 120! combinations
//! - MCConsensus/MCVoting: ISpec pattern where Init is an invariant
//! - Specs with unconstrained function variables
//!
//! For these specs, SMT-based enumeration via ay's ALL-SAT solver is more
//! efficient than brute-force enumeration.
//!
//! # Architecture
//!
//! ```text
//! TLA+ Init predicate
//!         │
//!         ▼
//! ┌───────────────────┐
//! │  AYTranslator     │  TLA+ Expr → ay Formula
//! └────────┬──────────┘
//!          ▼
//! ┌───────────────────┐
//! │  ay SMT solver    │  QF_LIA / QF_AUFLIA
//! └────────┬──────────┘
//!          ▼
//! ┌───────────────────┐
//! │  Solution enum    │  ALL-SAT with blocking clauses
//! └────────┬──────────┘
//!          ▼
//! ┌───────────────────┐
//! │  Model → State    │  ay model to TY Value
//! └───────────────────┘
//! ```
//!
//! # Phases
//!
//! ## Phase 2a: Basic Bool/Int translation
//! - Boolean: TRUE, FALSE, /\, \/, ~, =>, <=>
//! - Integer: literals, +, -, *, \div, %, comparisons
//! - Finite set membership: x \in {a, b, c}
//! - Range membership: x \in lo..hi
//!
//! ## Phase 2b (Current): Function support for finite domains
//! - Function application: <code>f\[x\]</code> via ITE expansion
//! - Function set membership: f \in [D -> R] for finite D
//! - Automatic function variable declaration from constraints
//!
//! For finite domains, functions are represented as individual scalar
//! variables (f__key1, f__key2, ...) and function application is expanded
//! to ITE chains. This handles common TLA+ model checking patterns.
//!
//! ## Phase 2c (Planned): tla-check integration
//! - Connect to enumerate.rs
//! - Run blocked specs with ay path
//! - Full array theory for larger domains (requires ay API extension)
//!
//! # Example
//!
//! ```no_run
//! use num_bigint::BigInt;
//! use tla_core::ast::Expr;
//! use tla_core::span::Spanned;
//! use tla_ay::{SolveResult, TlaSort, AYTranslator};
//!
//! fn main() -> Result<(), tla_ay::AYError> {
//!     let mut trans = AYTranslator::new();
//!
//!     // Declare state variables
//!     trans.declare_var("x", TlaSort::Int)?;
//!     trans.declare_var("y", TlaSort::Int)?;
//!
//!     // Translate Init predicate: x = 0 /\\ y > 5
//!     let x_eq_0 = Spanned::dummy(Expr::Eq(
//!         Box::new(Spanned::dummy(Expr::Ident("x".to_string(), tla_core::name_intern::NameId::INVALID))),
//!         Box::new(Spanned::dummy(Expr::Int(BigInt::from(0)))),
//!     ));
//!     let y_gt_5 = Spanned::dummy(Expr::Gt(
//!         Box::new(Spanned::dummy(Expr::Ident("y".to_string(), tla_core::name_intern::NameId::INVALID))),
//!         Box::new(Spanned::dummy(Expr::Int(BigInt::from(5)))),
//!     ));
//!     let init_expr = Spanned::dummy(Expr::And(Box::new(x_eq_0), Box::new(y_gt_5)));
//!
//!     let init_term = trans.translate_bool(&init_expr)?;
//!     trans.assert(init_term);
//!
//!     // Check satisfiability
//!     if matches!(trans.check_sat(), SolveResult::Sat) {
//!         let model = trans.get_model().expect("SAT implies a model");
//!         let _x = model.int_val("x").unwrap();
//!         let _y = model.int_val("y").unwrap();
//!     }
//!     Ok(())
//! }
//! ```
//!
//! Copyright 2026 Andrew Yates
//! SPDX-License-Identifier: Apache-2.0

#![deny(missing_docs)]
// `if cond { 1 } else { 0 }` patterns are clearer than `usize::from(cond)`
// when the surrounding context is integer accumulation.
#![allow(clippy::bool_to_int_with_if)]
// Helpers retained for staged solver-encoding work (e.g. encode_symbolic_subset
// while the compound dispatch lane lands).
#![allow(dead_code)]

pub mod bmc;
pub mod chc;
pub(crate) mod dispatch_shared;
pub mod error;
pub mod shared_engine;
pub mod translate;

pub use bmc::incremental::{IncrementalBmc, IncrementalBmcResult};
pub use bmc::{BmcState, BmcTranslator, BmcValue};
// Proof-artifact surface for certifying verification (AY's own re-checkable proof).
pub use ay_dpll::api::{FarkasCertificate, ProofQuality, StrictProofVerdict, UnsatProofArtifact};
// Leg D: the portable, checker-only proof bundle + store-independent canonical
// renderer + offline strict re-checker (from ay-proof), plus the ay-core term
// primitives the verifier rebuilds the embedded store from. These let a consumer
// re-check an embedded proof WITHOUT re-running the solver.
pub use ay_core::{Constant, ProofStep, Sort as AyCoreSort, TermData, TermId, TermStore};
// For the engine-diverse Leg D part-2 binding: TY builds concrete `mk_int` probe
// constants and reads back `Constant::Bool` after `substitute`+`simplify`.
pub use ay_proof::{
    re_check_bundle_strict, render_term_canonical, BundleReCheck, ProofCheckError,
    SerializableProofBundle, PROOF_BUNDLE_SCHEMA,
};
pub use num_bigint::BigInt;
// k-Induction checker re-exports (Part of #3722)
pub use bmc::kinduction::{
    KInductionChecker, KInductionConfig as AYKInductionConfig,
    KInductionResult as AYKInductionResult,
};
pub use error::{AYError, AYResult};
pub use shared_engine::{
    ay_shared_engine_all_lane_metadata, ay_shared_engine_evidence_key_value_rows,
    ay_shared_engine_lane_metadata, render_ay_shared_engine_evidence,
    render_ay_shared_engine_lane_evidence, AYFrontendFamily, AYProofValidationReceipt,
    AYProofValidationReceiptKind, AYProofValidationReceiptStatus, AYSharedEngineLane,
    AYSharedEngineLaneMetadata, AYSharedProofLaneDescriptor,
    AY_ANALYTICAL_PROOF_SHARED_ENGINE_COMPONENT, AY_SHARED_ENGINE_FRONTEND_FAMILIES,
    AY_SHARED_ENGINE_LANES, AY_SHARED_ENGINE_METADATA_SCHEMA,
    AY_SHARED_ENGINE_METADATA_SCHEMA_VERSION, AY_SHARED_PROOF_LANE_DESCRIPTOR_SCHEMA,
    AY_SHARED_PROOF_LANE_DESCRIPTOR_SCHEMA_VERSION, AY_SHARED_PROOF_VALIDATION_RECEIPT_SCHEMA,
    AY_SHARED_PROOF_VALIDATION_RECEIPT_SCHEMA_VERSION,
};
pub use translate::finite_set::FiniteSetEncoder;
pub use translate::function_encoder::{FuncTerm, FunctionEncoder};
pub use translate::nested_powerset::{
    k_subsets, BaseElement, NestedPowersetConfig, NestedPowersetEncoder, NestedPowersetSolution,
};
pub use translate::record_encoder::RecordEncoder;
pub use translate::sequence_encoder::{SeqTerm, SequenceEncoder};
pub use translate::{AYTranslator, TlaSort};

// Re-export ay types needed by users
pub use ay_dpll::api::{
    all_sat_enumeration_symbolic_execution_contract,
    all_sat_enumeration_symbolic_execution_contract_key_value_pairs,
    incremental_assumptions_symbolic_execution_contract,
    incremental_assumptions_symbolic_execution_contract_key_value_pairs,
    model_blocking_symbolic_execution_contract,
    model_blocking_symbolic_execution_contract_key_value_pairs, solver_capability_descriptor,
    solver_capability_descriptor_json, solver_capability_descriptor_key_value_pairs,
    solver_capability_descriptor_manifest,
    symbolic_execution_all_supported_capability_route_readiness,
    symbolic_execution_all_supported_capability_route_readiness_for_decision,
    symbolic_execution_all_supported_capability_route_readiness_json,
    symbolic_execution_all_supported_capability_route_readiness_key_value_rows,
    symbolic_execution_all_supported_capability_route_readiness_text_lines,
    symbolic_execution_capability_route_readiness,
    symbolic_execution_capability_route_readiness_for_decision,
    symbolic_execution_capability_route_readiness_json,
    symbolic_execution_capability_route_readiness_key_value_rows,
    symbolic_execution_capability_route_readiness_text_lines, symbolic_execution_contract_manifest,
    symbolic_execution_contract_manifest_diagnostic_summary,
    symbolic_execution_contract_manifest_diagnostic_summary_for_round_trip,
    symbolic_execution_contract_manifest_diagnostic_summary_json,
    symbolic_execution_contract_manifest_diagnostic_summary_key_value_rows,
    symbolic_execution_contract_manifest_diagnostic_summary_text_lines,
    symbolic_execution_contract_manifest_health_diagnostic_lines,
    symbolic_execution_contract_manifest_health_key_value_rows,
    symbolic_execution_contract_manifest_health_report, symbolic_execution_contract_manifest_json,
    symbolic_execution_contract_manifest_key_value_pairs,
    symbolic_execution_contract_manifest_round_trip_health_report,
    symbolic_execution_route_admission_decision,
    symbolic_execution_route_admission_decision_for_summary,
    symbolic_execution_route_admission_decision_json,
    symbolic_execution_route_admission_decision_key_value_rows,
    symbolic_execution_route_admission_decision_text_lines,
    validate_symbolic_execution_all_supported_capability_route_readiness,
    validate_symbolic_execution_all_supported_capability_route_readiness_key_value_rows,
    validate_symbolic_execution_all_supported_capability_route_readiness_text_lines,
    validate_symbolic_execution_capability_route_readiness,
    validate_symbolic_execution_capability_route_readiness_key_value_rows,
    validate_symbolic_execution_capability_route_readiness_text_lines,
    validate_symbolic_execution_contract_manifest,
    validate_symbolic_execution_contract_manifest_diagnostic_summary,
    validate_symbolic_execution_contract_manifest_diagnostic_summary_key_value_rows,
    validate_symbolic_execution_contract_manifest_diagnostic_summary_text_lines,
    validate_symbolic_execution_contract_manifest_key_value_pairs,
    validate_symbolic_execution_contract_manifest_round_trip,
    validate_symbolic_execution_route_admission_decision,
    validate_symbolic_execution_route_admission_decision_key_value_rows,
    validate_symbolic_execution_route_admission_decision_text_lines, Logic, Model,
    ModelBlockingAssignment, ModelBlockingClause, ModelBlockingClauseEvidence, SolveDecision,
    SolveDecisionProfileModelConsumerDecision, SolveDecisionProfileModelConsumerReason,
    SolveDecisionProfileModelConsumerStatus, SolveDecisionProfileSummary, SolveDetails,
    SolveResult, Solver, SolverCapability, SolverCapabilityCode, SolverCapabilityContract,
    SolverCapabilityDescriptor, SolverCapabilityDescriptorManifest, SolverCapabilityReason,
    SolverCapabilityStatus, SolverError, Sort, SymbolicExecutionCapabilityRouteReadiness,
    SymbolicExecutionCapabilityRouteReadinessReason,
    SymbolicExecutionCapabilityRouteReadinessStatus, SymbolicExecutionContractManifest,
    SymbolicExecutionContractManifestDiagnosticSummary, SymbolicExecutionContractManifestEntry,
    SymbolicExecutionContractManifestHealthDiagnostic,
    SymbolicExecutionContractManifestHealthIssue, SymbolicExecutionContractManifestHealthReason,
    SymbolicExecutionContractManifestHealthReport, SymbolicExecutionContractManifestHealthStatus,
    SymbolicExecutionRouteAdmissionDecision, SymbolicExecutionRouteAdmissionReason,
    SymbolicExecutionRouteAdmissionStatus, Term,
    AY_ALL_SAT_ENUMERATION_SYMBOLIC_EXECUTION_CONTRACT_SCHEMA,
    AY_ALL_SAT_ENUMERATION_SYMBOLIC_EXECUTION_CONTRACT_SCHEMA_VERSION,
    AY_INCREMENTAL_ASSUMPTIONS_SYMBOLIC_EXECUTION_CONTRACT_SCHEMA,
    AY_INCREMENTAL_ASSUMPTIONS_SYMBOLIC_EXECUTION_CONTRACT_SCHEMA_VERSION,
    AY_MODEL_BLOCKING_CLAUSE_EVIDENCE_ACCEPTED_REASON,
    AY_MODEL_BLOCKING_CLAUSE_EVIDENCE_ACCEPTED_STATUS,
    AY_MODEL_BLOCKING_CLAUSE_EVIDENCE_FAIL_CLOSED_REASON,
    AY_MODEL_BLOCKING_CLAUSE_EVIDENCE_FAIL_CLOSED_STATUS, AY_MODEL_BLOCKING_CLAUSE_EVIDENCE_SCHEMA,
    AY_MODEL_BLOCKING_CLAUSE_EVIDENCE_SCHEMA_VERSION, AY_MODEL_BLOCKING_CLAUSE_SCHEMA,
    AY_MODEL_BLOCKING_CLAUSE_SCHEMA_VERSION, AY_MODEL_BLOCKING_SYMBOLIC_EXECUTION_CONTRACT_SCHEMA,
    AY_MODEL_BLOCKING_SYMBOLIC_EXECUTION_CONTRACT_SCHEMA_VERSION, AY_SOLVER_CAPABILITIES,
    AY_SOLVER_CAPABILITY_DESCRIPTOR_MANIFEST_SCHEMA,
    AY_SOLVER_CAPABILITY_DESCRIPTOR_MANIFEST_SCHEMA_VERSION,
    AY_SOLVER_CAPABILITY_DESCRIPTOR_SCHEMA, AY_SOLVER_CAPABILITY_DESCRIPTOR_SCHEMA_VERSION,
    AY_SOLVE_DECISION_PROFILE_SUMMARY_SCHEMA, AY_SOLVE_DECISION_PROFILE_SUMMARY_SCHEMA_VERSION,
    AY_SYMBOLIC_EXECUTION_CAPABILITY_ROUTE_READINESS_HELPERS,
    AY_SYMBOLIC_EXECUTION_CAPABILITY_ROUTE_READINESS_SCHEMA,
    AY_SYMBOLIC_EXECUTION_CAPABILITY_ROUTE_READINESS_SCHEMA_VERSION,
    AY_SYMBOLIC_EXECUTION_CAPABILITY_ROUTE_READINESS_VALIDATORS, AY_SYMBOLIC_EXECUTION_CONTRACTS,
    AY_SYMBOLIC_EXECUTION_CONTRACT_DIAGNOSTIC_HELPERS,
    AY_SYMBOLIC_EXECUTION_CONTRACT_DIAGNOSTIC_SUMMARY_SCHEMA,
    AY_SYMBOLIC_EXECUTION_CONTRACT_DIAGNOSTIC_SUMMARY_SCHEMA_VERSION,
    AY_SYMBOLIC_EXECUTION_CONTRACT_MANIFEST_HEALTH_SCHEMA,
    AY_SYMBOLIC_EXECUTION_CONTRACT_MANIFEST_HEALTH_SCHEMA_VERSION,
    AY_SYMBOLIC_EXECUTION_CONTRACT_MANIFEST_SCHEMA,
    AY_SYMBOLIC_EXECUTION_CONTRACT_MANIFEST_SCHEMA_VERSION,
    AY_SYMBOLIC_EXECUTION_CONTRACT_REQUIRED_CAPABILITIES,
    AY_SYMBOLIC_EXECUTION_ROUTE_ADMISSION_HELPERS, AY_SYMBOLIC_EXECUTION_ROUTE_ADMISSION_SCHEMA,
    AY_SYMBOLIC_EXECUTION_ROUTE_ADMISSION_SCHEMA_VERSION,
    AY_SYMBOLIC_EXECUTION_ROUTE_ADMISSION_VALIDATORS,
    AY_SYMBOLIC_EXECUTION_ROUTE_CURRENT_REVISION_KIND,
    AY_SYMBOLIC_EXECUTION_ROUTE_REQUIRED_CONTRACT_REVISION,
    AY_SYMBOLIC_EXECUTION_ROUTE_SELECTED_SOLVER, AY_SYMBOLIC_EXECUTION_ROUTE_SELECTED_SOLVER_CRATE,
    AY_SYMBOLIC_EXECUTION_ROUTE_SELECTED_SOLVER_PATH_KIND,
};
pub use ay_dpll::UnknownReason;

// Re-export CHC types for PDR users (Part of #642)
pub use ay_chc::PdrConfig;
// Re-export the CHC engines' cooperative cancellation token so ty's fused
// orchestrator can abort an in-flight PDR/CHC solve (all ay-chc engine main
// loops poll it) once another lane has already resolved the verdict.
pub use ay_chc::CancellationToken;

/// Expected upstream fields TY should consume once the AY pin is advanced.
pub const AY_SOLVE_DECISION_PROFILE_SUMMARY_EXPECTED_FIELDS: &str = "schema,schema_version,decision,decision_code,decision_name,accepted_for_consumer,consumer_rejection_code,model_validated,verification_level,verification_level_code,verification,unknown,profile";

const NO_SOLVE_DECISION_PROFILE_REASON_CODE: &str = "none";

/// Render typed solve decision/profile summary evidence.
#[must_use]
pub fn solve_decision_profile_summary_evidence(
    scope: &str,
    summary: Option<&SolveDecisionProfileSummary>,
) -> String {
    match summary {
        Some(summary) => render_solve_decision_profile_summary(scope, summary),
        None => render_missing_solve_decision_profile_summary(scope),
    }
}

/// Render typed solve decision/profile summary evidence from an atomic AY solve envelope.
#[must_use]
pub fn solve_details_decision_profile_summary_evidence(
    scope: &str,
    details: Option<&SolveDetails>,
) -> String {
    let summary = details.map(SolveDetails::decision_profile_summary);
    solve_decision_profile_summary_evidence(scope, summary.as_ref())
}

fn render_missing_solve_decision_profile_summary(scope: &str) -> String {
    format!(
        "{} ay_solver_decision_profile_summary status=Unavailable status_code=missing_typed_summary typed_consumer=false expected_schema={} expected_schema_version={} expected_fields={} production_selected=false fail_closed=true",
        scope,
        AY_SOLVE_DECISION_PROFILE_SUMMARY_SCHEMA,
        AY_SOLVE_DECISION_PROFILE_SUMMARY_SCHEMA_VERSION,
        AY_SOLVE_DECISION_PROFILE_SUMMARY_EXPECTED_FIELDS,
    )
}

fn render_solve_decision_profile_summary(
    scope: &str,
    summary: &SolveDecisionProfileSummary,
) -> String {
    let unknown_reason_code = summary
        .unknown
        .as_ref()
        .map_or(NO_SOLVE_DECISION_PROFILE_REASON_CODE, |unknown| {
            unknown.reason_code
        });
    let unknown_limit_code = summary
        .unknown
        .as_ref()
        .and_then(|unknown| unknown.limit_code)
        .unwrap_or(NO_SOLVE_DECISION_PROFILE_REASON_CODE);
    let consumer_rejection_code = summary
        .consumer_rejection_code
        .unwrap_or(NO_SOLVE_DECISION_PROFILE_REASON_CODE);
    let fail_closed = solve_decision_profile_summary_fail_closed(summary);

    format!(
        "{} ay_solver_decision_profile_summary status=Available status_code=typed_summary_available schema={} schema_version={} decision={} decision_code={} accepted_for_consumer={} consumer_rejection_code={} model_validated={} verification_level_code={} unknown_reason_code={} unknown_limit_code={} wall_time_ms={} conflicts={} decisions={} propagations={} restarts={} learned_clause_count={} typed_consumer=true production_selected=false fail_closed={}",
        scope,
        summary.schema,
        summary.schema_version,
        summary.decision_name,
        summary.decision_code,
        summary.accepted_for_consumer,
        consumer_rejection_code,
        summary.model_validated,
        summary.verification_level_code,
        unknown_reason_code,
        unknown_limit_code,
        summary.profile.wall_time_ms,
        summary.profile.conflicts,
        summary.profile.decisions,
        summary.profile.propagations,
        summary.profile.restarts,
        summary.profile.learned_clause_count,
        fail_closed,
    )
}

fn solve_decision_profile_summary_fail_closed(summary: &SolveDecisionProfileSummary) -> bool {
    summary.model_consumer_decision().fail_closed
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;

    #[test]
    fn decision_profile_summary_evidence_fails_closed_without_typed_summary() {
        let evidence = solve_decision_profile_summary_evidence("TLA", None);

        assert!(evidence.contains("TLA ay_solver_decision_profile_summary"));
        assert!(evidence.contains("status_code=missing_typed_summary"));
        assert!(evidence.contains("typed_consumer=false"));
        assert!(evidence.contains("expected_schema=ay.solve-decision-profile-summary.v1"));
        assert!(evidence.contains("expected_schema_version=1"));
        assert!(evidence.contains("expected_fields=schema,schema_version,decision,decision_code,decision_name,accepted_for_consumer,consumer_rejection_code,model_validated,verification_level,verification_level_code,verification,unknown,profile"));
        assert!(evidence.contains("production_selected=false"));
        assert!(evidence.contains("fail_closed=true"));
    }

    #[test]
    fn decision_profile_summary_evidence_consumes_ay_typed_summary() {
        let mut solver = Solver::try_new(Logic::QfLia).expect("solver");
        let x = solver.declare_const("x", Sort::Int);
        let five = solver.int_const(5);
        let eq = solver.try_eq(x, five).expect("eq");
        solver.try_assert_term(eq).expect("assert");

        let details = solver.try_check_sat_with_details().expect("solve details");
        let evidence = solve_details_decision_profile_summary_evidence("TLA", Some(&details));

        assert!(evidence.contains("TLA ay_solver_decision_profile_summary"));
        assert!(evidence.contains("status_code=typed_summary_available"));
        assert!(evidence.contains("schema=ay.solve-decision-profile-summary.v1"));
        assert!(evidence.contains("schema_version=1"));
        assert!(evidence.contains("decision=SAT"));
        assert!(evidence.contains("decision_code=sat"));
        assert!(evidence.contains("accepted_for_consumer=true"));
        assert!(evidence.contains("consumer_rejection_code=none"));
        assert!(evidence.contains("model_validated=true"));
        assert!(evidence.contains("verification_level_code="));
        assert!(evidence.contains("unknown_reason_code=none"));
        assert!(evidence.contains("unknown_limit_code=none"));
        assert!(evidence.contains("typed_consumer=true"));
        assert!(evidence.contains("production_selected=false"));
        assert!(evidence.contains("fail_closed=false"));
    }

    #[test]
    fn solver_capability_descriptor_exposes_model_blocking_boundary() {
        let descriptor = solver_capability_descriptor();
        let model_blocking = descriptor
            .capability(SolverCapabilityCode::ModelBlocking)
            .expect("model-blocking capability is explicit");
        let exported_capability_rows: &[SolverCapability] = AY_SOLVER_CAPABILITIES;

        assert_eq!(descriptor.schema, AY_SOLVER_CAPABILITY_DESCRIPTOR_SCHEMA);
        assert_eq!(descriptor.capabilities, exported_capability_rows);
        assert!(descriptor.supports(SolverCapabilityCode::ModelBlocking));
        assert_eq!(model_blocking.status, SolverCapabilityStatus::Available);
        assert_eq!(
            model_blocking.reason,
            SolverCapabilityReason::AYOwnedPublicApi
        );
        assert!(model_blocking
            .api_symbols
            .contains(&"ay_dpll::api::Solver::try_assert_model_blocking_clause_for_consumer"));
        assert!(model_blocking
            .evidence_schemas
            .contains(&AY_MODEL_BLOCKING_CLAUSE_SCHEMA));
        assert!(model_blocking
            .evidence_schemas
            .contains(&AY_MODEL_BLOCKING_CLAUSE_EVIDENCE_SCHEMA));
    }

    #[test]
    fn solver_capability_descriptor_manifest_reexports_forwardable_rows() {
        let manifest: SolverCapabilityDescriptorManifest = solver_capability_descriptor_manifest();
        let pairs = solver_capability_descriptor_key_value_pairs();

        assert_eq!(
            manifest.schema,
            AY_SOLVER_CAPABILITY_DESCRIPTOR_MANIFEST_SCHEMA
        );
        assert_eq!(
            manifest.schema_version,
            AY_SOLVER_CAPABILITY_DESCRIPTOR_MANIFEST_SCHEMA_VERSION
        );
        assert_eq!(
            manifest.descriptor_schema,
            AY_SOLVER_CAPABILITY_DESCRIPTOR_SCHEMA
        );
        assert!(manifest
            .available_capability_codes
            .contains(&SolverCapabilityCode::ModelBlocking.code()));
        assert!(manifest
            .api_symbols
            .contains(&"ay_dpll::api::Solver::try_assert_model_blocking_clause_for_consumer"));
        assert!(manifest
            .evidence_schemas
            .contains(&AY_MODEL_BLOCKING_CLAUSE_EVIDENCE_SCHEMA));
        assert!(manifest.all_capabilities_fail_closed);
        assert!(pairs.contains(&(
            "schema",
            AY_SOLVER_CAPABILITY_DESCRIPTOR_MANIFEST_SCHEMA.to_string()
        )));
        assert!(pairs.contains(&(
            "descriptor_schema",
            AY_SOLVER_CAPABILITY_DESCRIPTOR_SCHEMA.to_string()
        )));
        assert!(pairs
            .iter()
            .any(|(key, value)| *key == "available_capabilities"
                && value.contains(SolverCapabilityCode::ModelBlocking.code())));
    }

    #[test]
    fn symbolic_execution_contract_manifest_reexports_ay_owned_routing_contracts() {
        let manifest: SymbolicExecutionContractManifest = symbolic_execution_contract_manifest();
        let pairs = symbolic_execution_contract_manifest_key_value_pairs();

        assert_eq!(
            manifest.schema,
            AY_SYMBOLIC_EXECUTION_CONTRACT_MANIFEST_SCHEMA
        );
        assert_eq!(
            manifest.schema_version,
            AY_SYMBOLIC_EXECUTION_CONTRACT_MANIFEST_SCHEMA_VERSION
        );
        assert_eq!(manifest.contracts, AY_SYMBOLIC_EXECUTION_CONTRACTS);
        assert!(manifest.all_contracts_fail_closed);
        assert!(manifest
            .contracts
            .iter()
            .any(|contract| contract.contract_schema
                == AY_MODEL_BLOCKING_SYMBOLIC_EXECUTION_CONTRACT_SCHEMA));
        assert!(manifest
            .contracts
            .iter()
            .any(|contract| contract.contract_schema
                == AY_INCREMENTAL_ASSUMPTIONS_SYMBOLIC_EXECUTION_CONTRACT_SCHEMA));
        assert!(manifest
            .contracts
            .iter()
            .any(|contract| contract.contract_schema
                == AY_ALL_SAT_ENUMERATION_SYMBOLIC_EXECUTION_CONTRACT_SCHEMA));
        assert!(pairs.contains(&(
            "schema",
            AY_SYMBOLIC_EXECUTION_CONTRACT_MANIFEST_SCHEMA.to_string()
        )));
        assert!(pairs.iter().any(|(key, value)| {
            *key == "contract_capabilities" && value.contains("model_blocking")
        }));
    }

    #[test]
    fn symbolic_execution_contract_manifest_health_is_forwardable_without_local_checks() {
        let manifest = symbolic_execution_contract_manifest();
        let pairs = symbolic_execution_contract_manifest_key_value_pairs();
        let report: SymbolicExecutionContractManifestHealthReport =
            symbolic_execution_contract_manifest_health_report();

        assert_eq!(
            report.schema,
            AY_SYMBOLIC_EXECUTION_CONTRACT_MANIFEST_HEALTH_SCHEMA
        );
        assert_eq!(
            report.schema_version,
            AY_SYMBOLIC_EXECUTION_CONTRACT_MANIFEST_HEALTH_SCHEMA_VERSION
        );
        assert_eq!(
            report.status,
            SymbolicExecutionContractManifestHealthStatus::Complete
        );
        assert_eq!(
            report.reason,
            SymbolicExecutionContractManifestHealthReason::Complete
        );
        assert_eq!(
            report.diagnostic(),
            SymbolicExecutionContractManifestHealthDiagnostic::Healthy
        );
        assert_eq!(
            report.required_capabilities,
            AY_SYMBOLIC_EXECUTION_CONTRACT_REQUIRED_CAPABILITIES
        );
        assert!(report.accepted_for_consumer);
        assert!(report.all_contracts_fail_closed);
        assert!(report.issues.is_empty());
        assert_eq!(
            validate_symbolic_execution_contract_manifest(&manifest),
            report
        );
        assert!(
            validate_symbolic_execution_contract_manifest_key_value_pairs(&pairs)
                .accepted_for_consumer
        );
        assert!(
            validate_symbolic_execution_contract_manifest_round_trip(&manifest, &pairs)
                .accepted_for_consumer
        );
        assert!(
            symbolic_execution_contract_manifest_round_trip_health_report().accepted_for_consumer
        );
        assert!(symbolic_execution_contract_manifest_health_key_value_rows()
            .contains(&("diagnostic".to_string(), "healthy".to_string())));
        assert!(
            symbolic_execution_contract_manifest_health_diagnostic_lines()
                .contains(&"diagnostic=healthy".to_string())
        );
    }

    #[test]
    fn symbolic_execution_route_admission_reexports_ay_owned_decision() {
        let manifest = symbolic_execution_contract_manifest();
        let pairs = symbolic_execution_contract_manifest_key_value_pairs();
        let diagnostic_summary: SymbolicExecutionContractManifestDiagnosticSummary =
            symbolic_execution_contract_manifest_diagnostic_summary_for_round_trip(
                &manifest, &pairs,
            );
        let route_admission: SymbolicExecutionRouteAdmissionDecision =
            symbolic_execution_route_admission_decision_for_summary(&diagnostic_summary);
        let model_blocking_readiness: SymbolicExecutionCapabilityRouteReadiness =
            symbolic_execution_capability_route_readiness_for_decision(
                SolverCapabilityCode::ModelBlocking,
                &route_admission,
            );
        let all_supported_readiness =
            symbolic_execution_all_supported_capability_route_readiness_for_decision(
                &route_admission,
            );

        assert_eq!(
            diagnostic_summary.schema,
            AY_SYMBOLIC_EXECUTION_CONTRACT_DIAGNOSTIC_SUMMARY_SCHEMA
        );
        assert_eq!(
            diagnostic_summary.schema_version,
            AY_SYMBOLIC_EXECUTION_CONTRACT_DIAGNOSTIC_SUMMARY_SCHEMA_VERSION
        );
        assert!(
            validate_symbolic_execution_contract_manifest_diagnostic_summary(&diagnostic_summary)
                .accepted_for_consumer
        );
        assert!(
            validate_symbolic_execution_contract_manifest_diagnostic_summary_key_value_rows(
                &diagnostic_summary.to_key_value_rows()
            )
            .accepted_for_consumer
        );
        assert!(
            validate_symbolic_execution_contract_manifest_diagnostic_summary_text_lines(
                &diagnostic_summary.to_text_lines()
            )
            .accepted_for_consumer
        );
        assert!(AY_SYMBOLIC_EXECUTION_CONTRACT_DIAGNOSTIC_HELPERS
            .contains(&"ay_dpll::api::symbolic_execution_contract_manifest_diagnostic_summary"));
        assert_eq!(
            symbolic_execution_contract_manifest_diagnostic_summary().schema,
            AY_SYMBOLIC_EXECUTION_CONTRACT_DIAGNOSTIC_SUMMARY_SCHEMA
        );
        assert_eq!(
            symbolic_execution_contract_manifest_diagnostic_summary_json()["schema"],
            AY_SYMBOLIC_EXECUTION_CONTRACT_DIAGNOSTIC_SUMMARY_SCHEMA
        );
        assert!(
            symbolic_execution_contract_manifest_diagnostic_summary_key_value_rows().contains(&(
                "schema".to_string(),
                AY_SYMBOLIC_EXECUTION_CONTRACT_DIAGNOSTIC_SUMMARY_SCHEMA.to_string()
            ))
        );
        assert!(
            symbolic_execution_contract_manifest_diagnostic_summary_text_lines().contains(
                &format!(
                    "schema={}",
                    AY_SYMBOLIC_EXECUTION_CONTRACT_DIAGNOSTIC_SUMMARY_SCHEMA
                )
            )
        );

        assert_eq!(
            route_admission.schema,
            AY_SYMBOLIC_EXECUTION_ROUTE_ADMISSION_SCHEMA
        );
        assert_eq!(
            route_admission.schema_version,
            AY_SYMBOLIC_EXECUTION_ROUTE_ADMISSION_SCHEMA_VERSION
        );
        assert_eq!(
            route_admission.status,
            SymbolicExecutionRouteAdmissionStatus::Accepted
        );
        assert_eq!(
            route_admission.reason,
            SymbolicExecutionRouteAdmissionReason::AYAuthoritativeRoutes
        );
        assert!(route_admission.accepted_for_consumer);
        assert!(route_admission.fail_closed);
        assert!(
            validate_symbolic_execution_route_admission_decision(&route_admission)
                .accepted_for_consumer
        );
        assert!(
            validate_symbolic_execution_route_admission_decision_key_value_rows(
                &route_admission.to_key_value_rows()
            )
            .accepted_for_consumer
        );
        assert!(
            validate_symbolic_execution_route_admission_decision_text_lines(
                &route_admission.to_text_lines()
            )
            .accepted_for_consumer
        );
        assert!(AY_SYMBOLIC_EXECUTION_ROUTE_ADMISSION_HELPERS
            .contains(&"ay_dpll::api::symbolic_execution_route_admission_decision"));
        assert!(AY_SYMBOLIC_EXECUTION_ROUTE_ADMISSION_VALIDATORS.contains(
            &"ay_dpll::api::validate_symbolic_execution_route_admission_decision_key_value_rows"
        ));
        assert_eq!(
            symbolic_execution_route_admission_decision().schema,
            AY_SYMBOLIC_EXECUTION_ROUTE_ADMISSION_SCHEMA
        );
        assert_eq!(
            symbolic_execution_route_admission_decision_json()["schema"],
            AY_SYMBOLIC_EXECUTION_ROUTE_ADMISSION_SCHEMA
        );
        assert!(
            symbolic_execution_route_admission_decision_key_value_rows().contains(&(
                "schema".to_string(),
                AY_SYMBOLIC_EXECUTION_ROUTE_ADMISSION_SCHEMA.to_string()
            ))
        );
        assert!(
            symbolic_execution_route_admission_decision_text_lines().contains(&format!(
                "schema={}",
                AY_SYMBOLIC_EXECUTION_ROUTE_ADMISSION_SCHEMA
            ))
        );

        assert_eq!(
            model_blocking_readiness.schema,
            AY_SYMBOLIC_EXECUTION_CAPABILITY_ROUTE_READINESS_SCHEMA
        );
        assert_eq!(
            model_blocking_readiness.schema_version,
            AY_SYMBOLIC_EXECUTION_CAPABILITY_ROUTE_READINESS_SCHEMA_VERSION
        );
        assert_eq!(
            model_blocking_readiness.status,
            SymbolicExecutionCapabilityRouteReadinessStatus::Ready
        );
        assert_eq!(
            model_blocking_readiness.reason,
            SymbolicExecutionCapabilityRouteReadinessReason::AYAuthoritativeCapabilityRoute
        );
        assert_eq!(
            model_blocking_readiness.selected_solver,
            AY_SYMBOLIC_EXECUTION_ROUTE_SELECTED_SOLVER
        );
        assert_eq!(
            model_blocking_readiness.selected_solver_crate,
            AY_SYMBOLIC_EXECUTION_ROUTE_SELECTED_SOLVER_CRATE
        );
        assert_eq!(
            model_blocking_readiness.selected_solver_path_kind,
            AY_SYMBOLIC_EXECUTION_ROUTE_SELECTED_SOLVER_PATH_KIND
        );
        assert_eq!(
            model_blocking_readiness.selected_solver_path,
            "ay_dpll::api::Solver::try_assert_model_blocking_clause_for_consumer"
        );
        assert!(model_blocking_readiness.supported);
        assert_eq!(model_blocking_readiness.unsupported_reason, "none");
        assert_eq!(
            model_blocking_readiness.required_contract_revision,
            AY_SYMBOLIC_EXECUTION_ROUTE_REQUIRED_CONTRACT_REVISION
        );
        assert_eq!(
            model_blocking_readiness.current_ay_revision_kind,
            AY_SYMBOLIC_EXECUTION_ROUTE_CURRENT_REVISION_KIND
        );
        assert_ne!(model_blocking_readiness.current_ay_revision, "unknown");
        assert!(model_blocking_readiness.accepted_for_consumer);
        assert!(model_blocking_readiness.fail_closed);
        assert!(
            validate_symbolic_execution_capability_route_readiness(&model_blocking_readiness)
                .accepted_for_consumer
        );
        assert!(
            validate_symbolic_execution_capability_route_readiness_key_value_rows(
                SolverCapabilityCode::ModelBlocking,
                &model_blocking_readiness.to_key_value_rows(),
            )
            .accepted_for_consumer
        );
        assert!(
            validate_symbolic_execution_capability_route_readiness_text_lines(
                SolverCapabilityCode::ModelBlocking,
                &model_blocking_readiness.to_text_lines(),
            )
            .accepted_for_consumer
        );
        assert_eq!(
            symbolic_execution_capability_route_readiness(SolverCapabilityCode::ModelBlocking)
                .schema,
            AY_SYMBOLIC_EXECUTION_CAPABILITY_ROUTE_READINESS_SCHEMA
        );
        assert_eq!(
            symbolic_execution_capability_route_readiness_json(SolverCapabilityCode::ModelBlocking)
                ["schema"],
            AY_SYMBOLIC_EXECUTION_CAPABILITY_ROUTE_READINESS_SCHEMA
        );
        assert!(
            symbolic_execution_capability_route_readiness_key_value_rows(
                SolverCapabilityCode::ModelBlocking
            )
            .contains(&(
                "schema".to_string(),
                AY_SYMBOLIC_EXECUTION_CAPABILITY_ROUTE_READINESS_SCHEMA.to_string()
            ))
        );
        assert!(symbolic_execution_capability_route_readiness_text_lines(
            SolverCapabilityCode::ModelBlocking
        )
        .contains(&format!(
            "schema={}",
            AY_SYMBOLIC_EXECUTION_CAPABILITY_ROUTE_READINESS_SCHEMA
        )));
        assert_eq!(
            all_supported_readiness.len(),
            AY_SYMBOLIC_EXECUTION_CONTRACTS.len()
        );
        assert!(all_supported_readiness
            .iter()
            .all(|readiness| readiness.accepted_for_consumer && readiness.fail_closed));
        assert!(
            validate_symbolic_execution_all_supported_capability_route_readiness(
                &all_supported_readiness
            )
            .iter()
            .all(|readiness| readiness.accepted_for_consumer)
        );
        assert!(
            validate_symbolic_execution_all_supported_capability_route_readiness_key_value_rows(
                &symbolic_execution_all_supported_capability_route_readiness_key_value_rows()
            )
            .iter()
            .all(|readiness| readiness.accepted_for_consumer)
        );
        assert!(
            validate_symbolic_execution_all_supported_capability_route_readiness_text_lines(
                &symbolic_execution_all_supported_capability_route_readiness_text_lines()
            )
            .iter()
            .all(|readiness| readiness.accepted_for_consumer)
        );
        assert_eq!(
            symbolic_execution_all_supported_capability_route_readiness_json()
                .as_array()
                .expect("all-supported readiness JSON array")
                .len(),
            AY_SYMBOLIC_EXECUTION_CONTRACTS.len()
        );
        assert!(
            symbolic_execution_all_supported_capability_route_readiness_key_value_rows()
                .contains(&("model_blocking_status".to_string(), "ready".to_string()))
        );
        assert!(
            symbolic_execution_all_supported_capability_route_readiness_text_lines()
                .contains(&"all_sat_enumeration_fail_closed=true".to_string())
        );
        assert!(AY_SYMBOLIC_EXECUTION_CAPABILITY_ROUTE_READINESS_HELPERS
            .contains(&"ay_dpll::api::symbolic_execution_capability_route_readiness"));
        assert!(
            AY_SYMBOLIC_EXECUTION_CAPABILITY_ROUTE_READINESS_HELPERS.contains(
                &"ay_dpll::api::symbolic_execution_all_supported_capability_route_readiness"
            )
        );
        assert!(AY_SYMBOLIC_EXECUTION_CAPABILITY_ROUTE_READINESS_VALIDATORS.contains(
            &"ay_dpll::api::validate_symbolic_execution_capability_route_readiness_key_value_rows"
        ));
        assert!(AY_SYMBOLIC_EXECUTION_CAPABILITY_ROUTE_READINESS_VALIDATORS.contains(
            &"ay_dpll::api::validate_symbolic_execution_all_supported_capability_route_readiness_key_value_rows"
        ));
    }

    #[test]
    fn symbolic_execution_capability_route_readiness_reexports_ay_owned_rows() {
        let readiness: SymbolicExecutionCapabilityRouteReadiness =
            symbolic_execution_capability_route_readiness(SolverCapabilityCode::ModelBlocking);

        assert_eq!(
            readiness.schema,
            AY_SYMBOLIC_EXECUTION_CAPABILITY_ROUTE_READINESS_SCHEMA
        );
        assert_eq!(
            readiness.schema_version,
            AY_SYMBOLIC_EXECUTION_CAPABILITY_ROUTE_READINESS_SCHEMA_VERSION
        );
        assert_eq!(
            readiness.status,
            SymbolicExecutionCapabilityRouteReadinessStatus::Ready
        );
        assert_eq!(
            readiness.reason,
            SymbolicExecutionCapabilityRouteReadinessReason::AYAuthoritativeCapabilityRoute
        );
        assert_eq!(
            readiness.selected_solver,
            AY_SYMBOLIC_EXECUTION_ROUTE_SELECTED_SOLVER
        );
        assert_eq!(
            readiness.selected_solver_crate,
            AY_SYMBOLIC_EXECUTION_ROUTE_SELECTED_SOLVER_CRATE
        );
        assert_eq!(
            readiness.selected_solver_path_kind,
            AY_SYMBOLIC_EXECUTION_ROUTE_SELECTED_SOLVER_PATH_KIND
        );
        assert_eq!(
            readiness.selected_solver_path,
            "ay_dpll::api::Solver::try_assert_model_blocking_clause_for_consumer"
        );
        assert!(readiness.supported);
        assert_eq!(readiness.unsupported_reason, "none");
        assert_eq!(
            readiness.required_contract_revision,
            AY_SYMBOLIC_EXECUTION_ROUTE_REQUIRED_CONTRACT_REVISION
        );
        assert_eq!(
            readiness.current_ay_revision_kind,
            AY_SYMBOLIC_EXECUTION_ROUTE_CURRENT_REVISION_KIND
        );
        assert_ne!(readiness.current_ay_revision, "unknown");
        assert!(readiness.accepted_for_consumer);
        assert!(readiness.fail_closed);
        assert!(
            validate_symbolic_execution_capability_route_readiness(&readiness)
                .accepted_for_consumer
        );
        assert!(
            validate_symbolic_execution_capability_route_readiness_key_value_rows(
                SolverCapabilityCode::ModelBlocking,
                &readiness.to_key_value_rows()
            )
            .accepted_for_consumer
        );
        assert!(
            validate_symbolic_execution_capability_route_readiness_text_lines(
                SolverCapabilityCode::ModelBlocking,
                &readiness.to_text_lines()
            )
            .accepted_for_consumer
        );
        assert!(AY_SYMBOLIC_EXECUTION_CAPABILITY_ROUTE_READINESS_HELPERS
            .contains(&"ay_dpll::api::symbolic_execution_capability_route_readiness"));
        assert!(AY_SYMBOLIC_EXECUTION_CAPABILITY_ROUTE_READINESS_VALIDATORS.contains(
            &"ay_dpll::api::validate_symbolic_execution_capability_route_readiness_key_value_rows"
        ));

        let all_supported = symbolic_execution_all_supported_capability_route_readiness();
        assert!(all_supported
            .iter()
            .any(|readiness| readiness.capability == SolverCapabilityCode::ModelBlocking));
        assert!(
            validate_symbolic_execution_all_supported_capability_route_readiness(&all_supported)
                .iter()
                .all(|readiness| readiness.accepted_for_consumer)
        );
        assert!(
            validate_symbolic_execution_all_supported_capability_route_readiness_key_value_rows(
                &symbolic_execution_all_supported_capability_route_readiness_key_value_rows()
            )
            .iter()
            .all(|readiness| readiness.accepted_for_consumer)
        );
        assert!(
            validate_symbolic_execution_all_supported_capability_route_readiness_text_lines(
                &symbolic_execution_all_supported_capability_route_readiness_text_lines()
            )
            .iter()
            .all(|readiness| readiness.accepted_for_consumer)
        );
        assert_eq!(
            symbolic_execution_all_supported_capability_route_readiness_json()[0]["schema"],
            AY_SYMBOLIC_EXECUTION_CAPABILITY_ROUTE_READINESS_SCHEMA
        );
    }

    #[test]
    fn model_blocking_evidence_reexports_key_value_row_helpers() {
        let mut solver = Solver::try_new(Logic::QfLia).expect("solver");
        let x = solver.declare_const("x", Sort::Int);
        let five = solver.int_const(5);
        let eq = solver.try_eq(x, five).expect("eq");
        solver.try_assert_term(eq).expect("assert");
        let _details = solver.try_check_sat_with_details().expect("solve details");

        let blocking = solver
            .try_model_blocking_clause_for_consumer(&[x])
            .expect("validated SAT should produce model-blocking clause");
        let evidence: ModelBlockingClauseEvidence = blocking.evidence_descriptor();

        let pairs = evidence.to_key_value_pairs();
        assert!(pairs.contains(&(
            "schema",
            AY_MODEL_BLOCKING_CLAUSE_EVIDENCE_SCHEMA.to_string()
        )));
        assert!(pairs.contains(&("clause_schema", AY_MODEL_BLOCKING_CLAUSE_SCHEMA.to_string())));
        assert!(pairs.contains(&(
            "clause_schema_version",
            AY_MODEL_BLOCKING_CLAUSE_SCHEMA_VERSION.to_string()
        )));
        assert!(pairs.contains(&(
            "status",
            AY_MODEL_BLOCKING_CLAUSE_EVIDENCE_ACCEPTED_STATUS.to_string()
        )));
        assert!(pairs.contains(&(
            "reason",
            AY_MODEL_BLOCKING_CLAUSE_EVIDENCE_ACCEPTED_REASON.to_string()
        )));
        assert!(pairs.contains(&("assignment_count", "1".to_string())));
        assert!(pairs.contains(&("value_kinds", "Int".to_string())));
        assert!(pairs.contains(&("accepted_for_consumer", "true".to_string())));

        let json = evidence.to_json_value();
        assert_eq!(json["schema"], AY_MODEL_BLOCKING_CLAUSE_EVIDENCE_SCHEMA);
        assert_eq!(json["clause_schema"], AY_MODEL_BLOCKING_CLAUSE_SCHEMA);
        assert_eq!(
            json["clause_schema_version"],
            AY_MODEL_BLOCKING_CLAUSE_SCHEMA_VERSION
        );
        assert_eq!(json["assignment_count"], 1);
        assert_eq!(json["value_kinds"][0], "Int");
        assert_eq!(json["accepted_for_consumer"], true);
        assert_eq!(json["fail_closed"], false);
    }

    #[test]
    fn decision_profile_summary_evidence_fails_closed_for_unknown_summary() {
        let mut solver = Solver::try_new(Logic::QfLia).expect("solver");
        solver.set_timeout(Some(Duration::ZERO));

        let details = solver.try_check_sat_with_details().expect("solve details");
        let evidence = solve_details_decision_profile_summary_evidence("TLA", Some(&details));

        assert!(evidence.contains("TLA ay_solver_decision_profile_summary"));
        assert!(evidence.contains("status_code=typed_summary_available"));
        assert!(evidence.contains("decision=Unknown"));
        assert!(evidence.contains("decision_code=unknown"));
        assert!(evidence.contains("accepted_for_consumer=true"));
        assert!(evidence.contains("consumer_rejection_code=none"));
        assert!(evidence.contains("model_validated=false"));
        assert!(evidence.contains("verification_level_code="));
        assert!(evidence.contains("unknown_reason_code=timeout"));
        assert!(evidence.contains("unknown_limit_code=timeout"));
        assert!(evidence.contains("typed_consumer=true"));
        assert!(evidence.contains("production_selected=false"));
        assert!(evidence.contains("fail_closed=true"));
    }

    #[test]
    fn decision_profile_summary_evidence_fails_closed_for_rejected_sat_boundary() {
        let mut solver = Solver::try_new(Logic::QfLia).expect("solver");
        solver.set_timeout(Some(Duration::ZERO));

        let details = solver.try_check_sat_with_details().expect("solve details");
        let summary = details.decision_profile_summary();
        let decision = summary.model_consumer_decision();

        assert_eq!(decision.status_code, "rejected");
        assert_eq!(decision.reason_code, "non_sat_decision");
        assert!(decision.fail_closed);
        assert_eq!(
            solve_decision_profile_summary_fail_closed(&summary),
            decision.fail_closed
        );
    }
}
