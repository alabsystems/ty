// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Petri/trust-ir native admission bridge for the `tla-check` consumer.
//!
//! Petri produces a validated `trust_ir::NativeVerificationBundle`; trust-codegen owns the
//! native successor admission vocabulary. This module gives `tla-check` a small
//! consumer surface that renders trust_cg's typed `NativeInstallGateAdmissionSummary`
//! rows without parsing `Display` text.

#![allow(dead_code)]

use serde_json::{Map, Value};

const TRUST_CG_NATIVE_ADMISSION_EVIDENCE_REPORT_SCHEMA: &str =
    "ty.trust_cg.native_admission_evidence_report.v1";
const TRUST_CG_NATIVE_ADMISSION_EVIDENCE_REPORT_SCHEMA_VERSION: u32 = 1;

pub(in crate::check) const PETRI_NATIVE_ADMISSION_API: &str =
    "trust-cg::petri_native_successor_admission_from_trust_ir_bundle";
const PETRI_NATIVE_ADMISSION_KIND: &str = "petri_successor";
const PETRI_NATIVE_ADMISSION_SURFACE: &str = "native_successor";
const PETRI_NATIVE_ADMISSION_CONSUMER: &str = "tla-check";
const PETRI_NATIVE_ADMISSION_CONSUMER_MODE: &str = "ty_petri_native_jit";
const PETRI_NATIVE_ADMISSION_MISSING_TRANSPORT_REASON: &str = "missing_trust_ir_transport_identity";

#[derive(Clone, Debug, PartialEq, Eq)]
pub(in crate::check) struct PetriNativeAdmissionEvidenceReport {
    evidence_row: String,
    fields: Vec<(String, String)>,
}

impl PetriNativeAdmissionEvidenceReport {
    #[must_use]
    pub(in crate::check) fn missing_trust_ir_bundle() -> Self {
        let mut fields = Vec::new();
        push_field(&mut fields, "source", "PetriNativeSuccessorAdmissionBridge");
        push_field(
            &mut fields,
            "schema",
            TRUST_CG_NATIVE_ADMISSION_EVIDENCE_REPORT_SCHEMA,
        );
        push_field(
            &mut fields,
            "schema_version",
            TRUST_CG_NATIVE_ADMISSION_EVIDENCE_REPORT_SCHEMA_VERSION,
        );
        push_field(&mut fields, "consumer", PETRI_NATIVE_ADMISSION_CONSUMER);
        push_field(
            &mut fields,
            "consumer_mode",
            PETRI_NATIVE_ADMISSION_CONSUMER_MODE,
        );
        push_field(&mut fields, "kind", PETRI_NATIVE_ADMISSION_KIND);
        push_field(&mut fields, "surface", PETRI_NATIVE_ADMISSION_SURFACE);
        push_field(&mut fields, "disposition", "rejected");
        push_field(&mut fields, "status_code", "rejected");
        push_field(
            &mut fields,
            "rejection_code",
            PETRI_NATIVE_ADMISSION_MISSING_TRANSPORT_REASON,
        );
        push_field(
            &mut fields,
            "reason_code",
            PETRI_NATIVE_ADMISSION_MISSING_TRANSPORT_REASON,
        );
        push_field(&mut fields, "requested_authority", "validation_only");
        push_field(&mut fields, "install_authority", "none");
        push_field(
            &mut fields,
            "expected_output",
            "NativeInstallGateAdmissionSummary",
        );
        push_field(
            &mut fields,
            "bundle_type",
            "trust_ir::NativeVerificationBundle",
        );
        push_field(&mut fields, "admission_api", PETRI_NATIVE_ADMISSION_API);
        push_field(&mut fields, "trust_ir_bundle_available", false);
        push_field(&mut fields, "trust_ir_bundle_consumed", false);
        push_field(&mut fields, "production_selected", false);
        push_field(&mut fields, "fail_closed", true);

        let evidence_row = render_evidence_row("trust-cg trust_cg_admission_blocker", &fields);
        Self {
            evidence_row,
            fields,
        }
    }

    #[must_use]
    pub(in crate::check) fn from_trust_ir_bundle(
        bundle: &trust_ir::NativeVerificationBundle,
    ) -> Self {
        let summary = tla_trust_cg::petri_native_successor_admission_from_trust_ir_bundle(
            bundle,
            tla_trust_cg::PetriNativeSuccessorAdmissionExpected::validation_only(),
        );
        let trust_ir_bundle_consumed = summary.reason_code
            != Some(
                tla_trust_cg::NativeInstallGateRejectionCode::PetriTrustIrBundleValidationFailed
                    .as_str(),
            );

        Self::from_trust_cg_summary(&summary, true, trust_ir_bundle_consumed)
    }

    #[must_use]
    fn from_trust_cg_summary(
        summary: &tla_trust_cg::NativeInstallGateAdmissionSummary,
        trust_ir_bundle_available: bool,
        trust_ir_bundle_consumed: bool,
    ) -> Self {
        let reason_code = summary.reason_code.unwrap_or("none");
        let mut fields = Vec::new();
        push_field(&mut fields, "source", "NativeInstallGateAdmissionSummary");
        push_field(&mut fields, "schema", summary.schema);
        push_field(&mut fields, "schema_version", summary.schema_version);
        push_field(&mut fields, "consumer", &summary.consumer);
        push_field(&mut fields, "consumer_mode", &summary.consumer_mode);
        push_field(&mut fields, "kind", PETRI_NATIVE_ADMISSION_KIND);
        push_field(&mut fields, "surface", summary.surface);
        push_field(&mut fields, "disposition", summary.disposition);
        push_field(&mut fields, "status_code", summary.disposition);
        push_field(&mut fields, "rejection_code", reason_code);
        push_field(&mut fields, "reason_code", reason_code);
        push_field(
            &mut fields,
            "requested_authority",
            summary.requested_authority,
        );
        push_field(&mut fields, "install_authority", summary.install_authority);
        push_field(&mut fields, "admission_api", PETRI_NATIVE_ADMISSION_API);
        push_field(
            &mut fields,
            "trust_ir_bundle_available",
            trust_ir_bundle_available,
        );
        push_field(
            &mut fields,
            "trust_ir_bundle_consumed",
            trust_ir_bundle_consumed,
        );
        push_field(
            &mut fields,
            "trust_ir_bundle_available",
            trust_ir_bundle_available,
        );
        push_field(
            &mut fields,
            "trust_ir_bundle_consumed",
            trust_ir_bundle_consumed,
        );
        push_field(
            &mut fields,
            "actions_ty_native_activate",
            summary.actions.ty_native_activate,
        );
        push_field(
            &mut fields,
            "useful_native_delta",
            summary.useful_native_delta,
        );
        push_field(&mut fields, "packet_hash", summary.packet_hash);
        push_field(&mut fields, "artifact_id", &summary.artifact_id);
        push_field(&mut fields, "production_selected", false);
        push_field(&mut fields, "fail_closed", true);

        let evidence_row = render_evidence_row("trust-cg trust_cg_admission_blocker", &fields);
        Self {
            evidence_row,
            fields,
        }
    }

    #[must_use]
    pub(in crate::check) fn evidence_row(&self) -> &str {
        &self.evidence_row
    }

    #[must_use]
    pub(in crate::check) fn field(&self, key: &str) -> Option<&str> {
        self.fields
            .iter()
            .find(|(field, _)| field == key)
            .map(|(_, value)| value.as_str())
    }

    #[must_use]
    pub(in crate::check) fn to_json_value(&self) -> Value {
        let fields = self
            .fields
            .iter()
            .map(|(key, value)| (key.clone(), Value::String(value.clone())))
            .collect::<Map<_, _>>();

        serde_json::json!({
            "schema": TRUST_CG_NATIVE_ADMISSION_EVIDENCE_REPORT_SCHEMA,
            "schema_version": TRUST_CG_NATIVE_ADMISSION_EVIDENCE_REPORT_SCHEMA_VERSION,
            "backend": "trust-cg",
            "kind": PETRI_NATIVE_ADMISSION_KIND,
            "surface": PETRI_NATIVE_ADMISSION_SURFACE,
            "evidence": [self.evidence_row],
            "fields": fields,
        })
    }
}

fn push_field(fields: &mut Vec<(String, String)>, key: &str, value: impl ToString) {
    fields.push((key.to_string(), value.to_string()));
}

fn render_evidence_row(prefix: &str, fields: &[(String, String)]) -> String {
    let mut row = String::from(prefix);
    for (key, value) in fields {
        row.push(' ');
        row.push_str(key);
        row.push('=');
        row.push_str(value);
    }
    row
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    fn parse_evidence_row(row: &str) -> BTreeMap<&str, &str> {
        row.split_whitespace()
            .filter_map(|token| {
                let (key, value) = token.split_once('=')?;
                Some((key, value))
            })
            .collect()
    }

    #[test]
    fn petri_native_successor_admission_missing_trust_ir_bundle_fails_closed() {
        let report = PetriNativeAdmissionEvidenceReport::missing_trust_ir_bundle();
        let row = report.evidence_row();
        let fields = parse_evidence_row(row);

        assert!(row.starts_with("trust-cg trust_cg_admission_blocker "));
        assert_eq!(
            fields.get("source").copied(),
            Some("PetriNativeSuccessorAdmissionBridge")
        );
        assert_eq!(
            fields.get("schema").copied(),
            Some(TRUST_CG_NATIVE_ADMISSION_EVIDENCE_REPORT_SCHEMA)
        );
        assert_eq!(fields.get("consumer").copied(), Some("tla-check"));
        assert_eq!(
            fields.get("consumer_mode").copied(),
            Some("ty_petri_native_jit")
        );
        assert_eq!(fields.get("kind").copied(), Some("petri_successor"));
        assert_eq!(fields.get("surface").copied(), Some("native_successor"));
        assert_eq!(fields.get("disposition").copied(), Some("rejected"));
        assert_eq!(fields.get("status_code").copied(), Some("rejected"));
        assert_eq!(
            fields.get("rejection_code").copied(),
            Some("missing_trust_ir_transport_identity")
        );
        assert_eq!(
            fields.get("reason_code").copied(),
            Some("missing_trust_ir_transport_identity")
        );
        assert_eq!(
            fields.get("requested_authority").copied(),
            Some("validation_only")
        );
        assert_eq!(fields.get("install_authority").copied(), Some("none"));
        assert_eq!(
            fields.get("expected_output").copied(),
            Some("NativeInstallGateAdmissionSummary")
        );
        assert_eq!(
            fields.get("bundle_type").copied(),
            Some("trust_ir::NativeVerificationBundle")
        );
        assert_eq!(
            fields.get("admission_api").copied(),
            Some("trust-cg::petri_native_successor_admission_from_trust_ir_bundle")
        );
        assert_eq!(
            fields.get("trust_ir_bundle_available").copied(),
            Some("false")
        );
        assert_eq!(
            fields.get("trust_ir_bundle_consumed").copied(),
            Some("false")
        );
        assert_eq!(fields.get("production_selected").copied(), Some("false"));
        assert_eq!(fields.get("fail_closed").copied(), Some("true"));
    }

    #[test]
    fn petri_native_successor_admission_missing_trust_ir_bundle_json_is_summarizer_ready() {
        let report = PetriNativeAdmissionEvidenceReport::missing_trust_ir_bundle();
        let json = report.to_json_value();

        assert_eq!(
            json["schema"].as_str(),
            Some(TRUST_CG_NATIVE_ADMISSION_EVIDENCE_REPORT_SCHEMA)
        );
        assert_eq!(
            json["schema_version"].as_u64(),
            Some(u64::from(
                TRUST_CG_NATIVE_ADMISSION_EVIDENCE_REPORT_SCHEMA_VERSION
            ))
        );
        assert_eq!(json["backend"].as_str(), Some("trust-cg"));
        assert_eq!(json["kind"].as_str(), Some("petri_successor"));
        assert_eq!(json["surface"].as_str(), Some("native_successor"));
        assert_eq!(
            json["evidence"]
                .as_array()
                .and_then(|evidence| evidence.first())
                .and_then(Value::as_str),
            Some(report.evidence_row())
        );
        assert_eq!(
            json["fields"]["admission_api"].as_str(),
            Some("trust-cg::petri_native_successor_admission_from_trust_ir_bundle")
        );
        assert_eq!(
            json["fields"]["reason_code"].as_str(),
            Some("missing_trust_ir_transport_identity")
        );
        assert_eq!(json["fields"]["fail_closed"].as_str(), Some("true"));
    }
}
