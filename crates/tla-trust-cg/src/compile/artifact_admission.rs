// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Batch JIT artifact admission contract (fail-closed validation).
//!
//! Pure code motion out of `compile.rs`: the fail-closed admission types and
//! the [`admit_batch_jit_artifact`] entry point that gate whether a batch JIT
//! artifact carries the required fingerprints/options before it may be
//! installed. Parent types/consts (`OptLevel`, `BatchJitCompilePreset`,
//! `BatchJitStats`, the admission schema labels, the per-batch host symbol map
//! count) are pulled in via `use super::*`.

use super::*;

/// Fail-closed disposition for a batch JIT artifact admission check.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BatchJitArtifactAdmissionStatus {
    /// All required fingerprints and compile options are present.
    Accepted,
    /// One or more required fingerprints/options are missing or invalid.
    Rejected,
}

impl BatchJitArtifactAdmissionStatus {
    /// Stable status code for evidence rows.
    #[must_use]
    pub fn as_str(&self) -> &'static str {
        match self {
            BatchJitArtifactAdmissionStatus::Accepted => "accepted",
            BatchJitArtifactAdmissionStatus::Rejected => "rejected",
        }
    }
}

/// Minimal evidence required before a batch JIT artifact may be admitted.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct BatchJitArtifactAdmissionInput<'a> {
    /// Stable frontend-neutral semantic/trust-ir artifact digest.
    pub semantic_trust_ir_artifact_digest: Option<&'a str>,
    /// Process-local link digest that includes host/link discriminators.
    pub process_local_link_digest: Option<&'a str>,
    /// Compile preset requested/selected for this batch artifact.
    pub compile_preset: Option<BatchJitCompilePreset>,
    /// Effective optimization level used by the artifact.
    pub opt_level: Option<OptLevel>,
    /// Number of host symbol maps constructed for this batch.
    pub host_symbol_map_count: Option<usize>,
    /// Number of functions in the batch.
    pub function_count: Option<usize>,
}

impl<'a> BatchJitArtifactAdmissionInput<'a> {
    /// Build an admission input from returned batch stats.
    #[must_use]
    pub fn from_stats(stats: &'a BatchJitStats) -> Self {
        Self {
            semantic_trust_ir_artifact_digest: Some(&stats.artifact_identity.semantic_digest),
            process_local_link_digest: Some(&stats.artifact_identity.link_digest),
            compile_preset: Some(stats.compile_preset),
            opt_level: Some(stats.artifact_identity.opt_level),
            host_symbol_map_count: Some(stats.host_symbol_map_count),
            function_count: Some(stats.function_count),
        }
    }
}

/// Result of applying the fail-closed batch JIT artifact admission contract.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BatchJitArtifactAdmission {
    /// Stable schema label for this admission result.
    pub schema: &'static str,
    /// Stable schema version for this admission result.
    pub schema_version: u32,
    /// Admission status.
    pub status: BatchJitArtifactAdmissionStatus,
    /// Missing required fields.
    pub missing_fields: Vec<&'static str>,
    /// Stable rejection reason codes.
    pub rejection_reasons: Vec<&'static str>,
}

impl BatchJitArtifactAdmission {
    /// Return true when the artifact has the required fingerprints/options.
    #[must_use]
    pub fn is_admitted(&self) -> bool {
        self.status == BatchJitArtifactAdmissionStatus::Accepted
    }

    /// Return true when the artifact is rejected by the fail-closed contract.
    #[must_use]
    pub fn is_fail_closed(&self) -> bool {
        self.status == BatchJitArtifactAdmissionStatus::Rejected
    }
}

/// Apply the batch JIT fail-closed artifact admission contract.
#[must_use]
pub fn admit_batch_jit_artifact(
    input: BatchJitArtifactAdmissionInput<'_>,
) -> BatchJitArtifactAdmission {
    let mut missing_fields = Vec::new();
    let mut rejection_reasons = Vec::new();

    match input.semantic_trust_ir_artifact_digest {
        Some(value) if is_sha256_hex(value) => {}
        Some(_) => rejection_reasons.push("invalid_semantic_trust_ir_artifact_digest"),
        None => {
            missing_fields.push("semantic_trust_ir_artifact_digest");
            rejection_reasons.push("missing_semantic_trust_ir_artifact_digest");
        }
    }
    match input.process_local_link_digest {
        Some(value) if is_sha256_hex(value) => {}
        Some(_) => rejection_reasons.push("invalid_process_local_link_digest"),
        None => {
            missing_fields.push("process_local_link_digest");
            rejection_reasons.push("missing_process_local_link_digest");
        }
    }
    if input.compile_preset.is_none() {
        missing_fields.push("compile_preset");
        rejection_reasons.push("missing_compile_preset");
    }
    if input.opt_level.is_none() {
        missing_fields.push("opt_level");
        rejection_reasons.push("missing_opt_level");
    }
    match input.host_symbol_map_count {
        Some(TRUST_CG_BATCH_JIT_HOST_SYMBOL_MAPS_PER_BATCH) => {}
        Some(_) => rejection_reasons.push("host_symbol_map_count_must_be_one_per_batch"),
        None => {
            missing_fields.push("host_symbol_map_count");
            rejection_reasons.push("missing_host_symbol_map_count");
        }
    }
    match input.function_count {
        Some(count) if count > 0 => {}
        Some(_) => rejection_reasons.push("function_count_must_be_positive"),
        None => {
            missing_fields.push("function_count");
            rejection_reasons.push("missing_function_count");
        }
    }

    BatchJitArtifactAdmission {
        schema: TRUST_CG_BATCH_JIT_ARTIFACT_ADMISSION_SCHEMA,
        schema_version: TRUST_CG_BATCH_JIT_ARTIFACT_ADMISSION_SCHEMA_VERSION,
        status: if rejection_reasons.is_empty() {
            BatchJitArtifactAdmissionStatus::Accepted
        } else {
            BatchJitArtifactAdmissionStatus::Rejected
        },
        missing_fields,
        rejection_reasons,
    }
}

fn is_sha256_hex(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}
