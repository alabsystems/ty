// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! CHC/PDR result types and formatting helpers

use std::collections::HashMap;

use ay_chc::{
    ChcProofTranscriptConsumerEvidence, ChcProofTranscriptMetadata, Counterexample, InvariantModel,
    CHC_PROOF_TRANSCRIPT_SCHEMA, NORMALIZED_CHC_INPUT_SCHEMA,
};

/// Fields expected from AY's typed CHC proof/replay transcript metadata.
pub const AY_CHC_PROOF_REPLAY_BOUNDARY_EXPECTED_FIELDS: &str = "schema,schema_version,normalized_input_schema,normalized_input_sha256,normalized_input_bytes,engine,result,proof_status,accepted_as_proof,replay_status,transcript_status,trust_full_verifier_admissible,trust_full_verifier_non_admission_reason,unknown_reason";

const NO_CHC_PROOF_REPLAY_REASON_CODE: &str = "none";

/// Result of PDR safety checking
#[derive(Debug)]
pub enum PdrCheckResult {
    /// Proven safe with synthesized invariant
    Safe {
        /// String representation of the invariant
        invariant: String,
    },
    /// Found counterexample trace
    Unsafe {
        /// Counterexample trace (each step is a state assignment)
        trace: Vec<PdrState>,
    },
    /// Inconclusive (resource limits reached)
    Unknown {
        /// Reason for unknown result
        reason: String,
    },
}

/// PDR result plus the typed AY CHC proof/replay boundary evidence row.
#[derive(Debug)]
pub struct PdrProofCheckResult {
    /// Proven-safe, unsafe, or inconclusive PDR result.
    pub result: PdrCheckResult,
    /// Stable evidence row rendered from AY CHC typed transcript metadata.
    pub proof_replay_evidence: String,
    /// AY-owned typed consumer evidence for proof/witness admission.
    pub proof_consumer_evidence: Option<ChcProofTranscriptConsumerEvidence>,
}

/// A state in the PDR counterexample trace
#[derive(Debug, Clone)]
pub struct PdrState {
    /// Variable assignments: name -> value
    pub assignments: HashMap<String, i64>,
}

/// Render AY CHC proof/replay boundary evidence.
#[must_use]
pub fn render_chc_proof_replay_boundary_evidence(
    scope: &str,
    metadata: Option<&ChcProofTranscriptMetadata>,
) -> String {
    match metadata {
        Some(metadata) => render_typed_chc_proof_replay_boundary(scope, metadata),
        None => render_missing_chc_proof_replay_boundary(scope),
    }
}

fn render_missing_chc_proof_replay_boundary(scope: &str) -> String {
    format!(
        "{} ay_chc_proof_replay_boundary status=Unavailable status_code=missing_typed_chc_proof_transcript typed_consumer=false expected_schema={} expected_schema_version=1 expected_normalized_input_schema={} expected_fields={} upstream_api=ay_chc::engines::solve_pdr_proof production_selected=false fail_closed=true",
        scope,
        CHC_PROOF_TRANSCRIPT_SCHEMA,
        NORMALIZED_CHC_INPUT_SCHEMA,
        AY_CHC_PROOF_REPLAY_BOUNDARY_EXPECTED_FIELDS,
    )
}

fn render_typed_chc_proof_replay_boundary(
    scope: &str,
    metadata: &ChcProofTranscriptMetadata,
) -> String {
    let trust_reason = metadata
        .trust_full_verifier_non_admission_reason
        .as_deref()
        .unwrap_or(NO_CHC_PROOF_REPLAY_REASON_CODE);
    let unknown_reason = metadata
        .unknown_reason
        .as_deref()
        .unwrap_or(NO_CHC_PROOF_REPLAY_REASON_CODE);
    let fail_closed = !metadata.trust_full_verifier_admissible;

    format!(
        "{} ay_chc_proof_replay_boundary status=Available status_code=typed_chc_proof_transcript schema={} schema_version=1 normalized_input_schema={} normalized_input_sha256={} normalized_input_bytes={} engine={} result={} proof_status={} accepted_as_proof={} replay_status={} transcript_status={} trust_full_verifier_admissible={} trust_full_verifier_non_admission_reason={} unknown_reason={} typed_consumer=true production_selected=false fail_closed={}",
        scope,
        metadata.schema,
        metadata.normalized_input_schema,
        metadata.normalized_input_sha256,
        metadata.normalized_input_bytes,
        metadata.engine,
        metadata.result,
        metadata.proof_status,
        metadata.accepted_as_proof,
        metadata.replay_status,
        metadata.transcript_status,
        metadata.trust_full_verifier_admissible,
        trust_reason,
        unknown_reason,
        fail_closed,
    )
}

/// Format invariant model for display
pub(super) fn format_invariant(model: &InvariantModel) -> String {
    format!("{model:?}")
}

/// Format counterexample trace
pub(super) fn format_counterexample(cex: &Counterexample) -> Vec<PdrState> {
    cex.steps
        .iter()
        .map(|step| {
            let assignments = step
                .assignments
                .iter()
                .map(|(name, val)| (name.clone(), *val))
                .collect();
            PdrState { assignments }
        })
        .collect()
}
