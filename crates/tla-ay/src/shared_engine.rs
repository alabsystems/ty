// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Frontend-neutral AY engine lane metadata.
//!
//! This module describes the shared transition-system contracts consumed by
//! the AY-backed all-SAT enumeration, BMC, CHC, PDR, and k-induction lanes.
//! The metadata is meant for callers that lower from different frontends into
//! the same generic state/transition/property proof obligations.

/// Schema identifier for shared AY engine lane metadata.
pub const AY_SHARED_ENGINE_METADATA_SCHEMA: &str = "tla-ay.shared-engine-metadata/v1";

/// Schema version for shared AY engine lane metadata.
pub const AY_SHARED_ENGINE_METADATA_SCHEMA_VERSION: u32 = 1;

/// Schema identifier for frontend-neutral analytical/AY proof lane publication.
pub const AY_SHARED_PROOF_LANE_DESCRIPTOR_SCHEMA: &str = "tla-ay.shared-proof-lane-descriptor/v1";

/// Schema version for frontend-neutral analytical/AY proof lane publication.
pub const AY_SHARED_PROOF_LANE_DESCRIPTOR_SCHEMA_VERSION: u32 = 1;

/// Schema identifier for frontend-neutral analytical/AY solve contract evidence.
pub const AY_ANALYTICAL_SOLVE_CONTRACT_SCHEMA: &str = "tla-ay.analytical-solve-contract/v1";

/// Schema version for frontend-neutral analytical/AY solve contract evidence.
pub const AY_ANALYTICAL_SOLVE_CONTRACT_SCHEMA_VERSION: u32 = 1;

/// Schema identifier for validation receipts that gate proof lane publication.
pub const AY_SHARED_PROOF_VALIDATION_RECEIPT_SCHEMA: &str =
    "tla-ay.shared-proof-validation-receipt/v1";

/// Schema version for validation receipts that gate proof lane publication.
pub const AY_SHARED_PROOF_VALIDATION_RECEIPT_SCHEMA_VERSION: u32 = 1;

/// Release-critical shared-engine component code for analytical/AY proof evidence.
pub const AY_ANALYTICAL_PROOF_SHARED_ENGINE_COMPONENT: &str = "analytical_ay_proof";

/// Shared owner for analytical/AY proof-lane publication evidence.
pub const AY_SHARED_PROOF_LANE_SHARED_OWNER: &str = "shared_high_performance_engine";

/// Extraction status for analytical/AY proof-lane publication evidence.
pub const AY_SHARED_PROOF_LANE_EXTRACTION_STATUS: &str = "shared-core-ready";

/// Blocker status for analytical/AY proof-lane publication evidence.
pub const AY_SHARED_PROOF_LANE_BLOCKER_STATUS: &str = "tracked-blockers";

/// Reserved importer blocker published until future importers register a
/// canonical transition-system/fingerprint mapping for the AY lanes.
pub const AY_SHARED_PROOF_LANE_FRONTEND_FAMILY_BLOCKERS: &str =
    "future_importer:awaiting_registered_importer_frontend";

/// Frontend-neutral big-win detector used by analytical/AY solve evidence.
pub const AY_ANALYTICAL_BIG_WIN_DETECTION_RULE: &str =
    "shared_analytical_solve_replaces_frontend_specific_search_or_enumeration";

/// Stable proof that analytical big-win detection is not tied to a source frontend.
pub const AY_ANALYTICAL_BIG_WIN_DETECTION_BASIS: &str =
    "prepared_descriptor_fingerprints_validation_receipts";

/// Default-consumer policy for analytical/AY proof-lane evidence.
pub const AY_ANALYTICAL_BIG_WIN_DEFAULT_FRONTEND_POLICY: &str =
    "all_active_compatible_frontend_families_default_after_receipt_validation";

/// Fail-closed replacement policy for explicit search.
pub const AY_ANALYTICAL_BIG_WIN_EXPLICIT_SEARCH_REPLACEMENT_POLICY: &str =
    "replace_explicit_search_only_after_validator_backed_proof_and_witness_receipts";

/// Blockers that keep analytical/AY big-win publication fail-closed.
pub const AY_ANALYTICAL_BIG_WIN_FAIL_CLOSED_BLOCKERS: &str =
    "missing_proof_receipt,missing_witness_receipt,future_importer_missing_registered_payload";

/// Known frontend families that can target the shared AY engine lanes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AYFrontendFamily {
    /// TLA+ modules lowered through `tla-core` expressions.
    Tla,
    /// Quint modules lowered to transition-system expressions.
    Quint,
    /// MCC/Petri net models lowered to transition relations.
    MccPetri,
    /// AIGER Boolean transition systems.
    Aiger,
    /// BTOR2 word-level transition systems.
    Btor2,
    /// AY-native transition systems or helper queries with no source frontend.
    AYOnly,
    /// VMT inputs and replay artifacts with normalized transition/property IDs.
    VmtReplay,
    /// Witness/replay inputs with validator-backed replay obligations.
    WitnessReplay,
    /// Future importer family reserved before a dedicated variant exists.
    FutureImporter,
}

impl AYFrontendFamily {
    /// Shared frontend families in stable evidence order.
    #[must_use]
    pub fn all() -> &'static [Self] {
        &AY_SHARED_ENGINE_FRONTEND_FAMILIES
    }

    /// Stable machine-readable frontend family code.
    #[must_use]
    pub const fn code(self) -> &'static str {
        match self {
            Self::Tla => "tla",
            Self::Quint => "quint",
            Self::MccPetri => "mcc_petri",
            Self::Aiger => "aiger",
            Self::Btor2 => "btor2",
            Self::AYOnly => "ay_only",
            Self::VmtReplay => "vmt_replay",
            Self::WitnessReplay => "witness_replay",
            Self::FutureImporter => "future_importer",
        }
    }

    /// Shared-engine adoption family code aligned with the core adoption registry.
    #[must_use]
    pub const fn adoption_code(self) -> &'static str {
        match self {
            Self::Tla => "tla_plus",
            Self::Quint => "quint",
            Self::MccPetri => "mcc_petri",
            Self::Aiger => "aiger",
            Self::Btor2 => "btor2",
            Self::AYOnly => "ay_analytical",
            Self::VmtReplay => "vmt_transition_system",
            Self::WitnessReplay => "witness_replay",
            Self::FutureImporter => "future_importer",
        }
    }

    /// Human-facing frontend family name.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Tla => "TLA",
            Self::Quint => "Quint",
            Self::MccPetri => "MCC/Petri",
            Self::Aiger => "AIGER",
            Self::Btor2 => "BTOR2",
            Self::AYOnly => "AY-only",
            Self::VmtReplay => "VMT/replay",
            Self::WitnessReplay => "witness/replay",
            Self::FutureImporter => "future importer",
        }
    }
}

/// AY-backed verification lanes exposed by this crate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AYSharedEngineLane {
    /// All-SAT model enumeration over typed symbolic variables.
    AllSatEnumeration,
    /// Bounded model checking over unrolled transition systems.
    Bmc,
    /// Constrained Horn clause construction.
    Chc,
    /// PDR/IC3-style solving over CHC inputs.
    Pdr,
    /// BMC-backed k-induction over transition systems.
    KInduction,
}

impl AYSharedEngineLane {
    /// Stable machine-readable lane code.
    #[must_use]
    pub const fn code(self) -> &'static str {
        match self {
            Self::AllSatEnumeration => "all_sat_enumeration",
            Self::Bmc => "bmc",
            Self::Chc => "chc",
            Self::Pdr => "pdr",
            Self::KInduction => "k_induction",
        }
    }

    /// Human-facing lane name.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::AllSatEnumeration => "ALL-SAT enumeration",
            Self::Bmc => "BMC",
            Self::Chc => "CHC",
            Self::Pdr => "PDR",
            Self::KInduction => "k-induction",
        }
    }
}

/// Validation receipt kind required before a proof lane may be published.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AYProofValidationReceiptKind {
    /// Output-format validator for exported transition-system artifacts.
    OutputFormat,
    /// Model checker/consumer validation for satisfying assignments or traces.
    Model,
    /// Certificate validator for frontend-neutral analytical solve certificates.
    Certificate,
    /// Replay validator for witness artifacts.
    Witness,
    /// Proof transcript validator for CHC/PDR/k-induction proof artifacts.
    ProofTranscript,
}

impl AYProofValidationReceiptKind {
    /// Stable machine-readable validation kind code.
    #[must_use]
    pub const fn code(self) -> &'static str {
        match self {
            Self::OutputFormat => "output_format",
            Self::Model => "model",
            Self::Certificate => "certificate",
            Self::Witness => "witness",
            Self::ProofTranscript => "proof_transcript",
        }
    }
}

/// Validation receipt status used to gate proof lane publication.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AYProofValidationReceiptStatus {
    /// A validator checked the named artifact/fingerprint boundary.
    ValidatorBacked,
    /// A producer only observed an artifact; this is not enough to publish.
    ArtifactOnly,
    /// The required validation receipt is absent.
    Missing,
}

impl AYProofValidationReceiptStatus {
    /// Stable machine-readable validation status code.
    #[must_use]
    pub const fn code(self) -> &'static str {
        match self {
            Self::ValidatorBacked => "validator_backed",
            Self::ArtifactOnly => "artifact_only",
            Self::Missing => "missing",
        }
    }
}

/// Receipt proving a proof/model/witness fingerprint passed the required validator.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AYProofValidationReceipt {
    /// Receipt schema identifier.
    pub schema: &'static str,
    /// Receipt schema version.
    pub schema_version: u32,
    /// Stable receipt identity.
    pub receipt_identity: String,
    /// Validator class that produced this receipt.
    pub validation_kind: AYProofValidationReceiptKind,
    /// Proof obligation identity that this receipt validates.
    pub proof_obligation_identity: String,
    /// Validated proof/model/witness fingerprint identity.
    pub validated_fingerprint_identity: String,
    /// Receipt status.
    pub status: AYProofValidationReceiptStatus,
}

impl AYProofValidationReceipt {
    /// Build a validator-backed receipt for one proof obligation/fingerprint boundary.
    #[must_use]
    pub fn validator_backed(
        receipt_identity: impl Into<String>,
        validation_kind: AYProofValidationReceiptKind,
        proof_obligation_identity: impl Into<String>,
        validated_fingerprint_identity: impl Into<String>,
    ) -> Self {
        Self {
            schema: AY_SHARED_PROOF_VALIDATION_RECEIPT_SCHEMA,
            schema_version: AY_SHARED_PROOF_VALIDATION_RECEIPT_SCHEMA_VERSION,
            receipt_identity: receipt_identity.into(),
            validation_kind,
            proof_obligation_identity: proof_obligation_identity.into(),
            validated_fingerprint_identity: validated_fingerprint_identity.into(),
            status: AYProofValidationReceiptStatus::ValidatorBacked,
        }
    }

    /// Return a copy of this receipt with a different status.
    #[must_use]
    pub fn with_status(mut self, status: AYProofValidationReceiptStatus) -> Self {
        self.status = status;
        self
    }

    fn has_publishable_shape_for(&self, descriptor: &AYSharedProofLaneDescriptor) -> bool {
        self.schema == AY_SHARED_PROOF_VALIDATION_RECEIPT_SCHEMA
            && self.schema_version == AY_SHARED_PROOF_VALIDATION_RECEIPT_SCHEMA_VERSION
            && self.status == AYProofValidationReceiptStatus::ValidatorBacked
            && !self.receipt_identity.is_empty()
            && self.proof_obligation_identity == descriptor.proof_obligation_identity
            && descriptor
                .fingerprint_identities()
                .contains(&self.validated_fingerprint_identity.as_str())
    }

    fn validates_fingerprint_for(
        &self,
        descriptor: &AYSharedProofLaneDescriptor,
        fingerprint_identity: &str,
    ) -> bool {
        self.has_publishable_shape_for(descriptor)
            && self.validated_fingerprint_identity == fingerprint_identity
    }
}

/// Frontend families listed for every shared lane.
pub static AY_SHARED_ENGINE_FRONTEND_FAMILIES: [AYFrontendFamily; 9] = [
    AYFrontendFamily::Tla,
    AYFrontendFamily::Quint,
    AYFrontendFamily::MccPetri,
    AYFrontendFamily::Aiger,
    AYFrontendFamily::Btor2,
    AYFrontendFamily::AYOnly,
    AYFrontendFamily::VmtReplay,
    AYFrontendFamily::WitnessReplay,
    AYFrontendFamily::FutureImporter,
];

/// Frontend families compatible with published AY analytical/proof lanes today.
pub static AY_SHARED_ENGINE_COMPATIBLE_FRONTEND_FAMILIES: [AYFrontendFamily; 8] = [
    AYFrontendFamily::Tla,
    AYFrontendFamily::Quint,
    AYFrontendFamily::MccPetri,
    AYFrontendFamily::Aiger,
    AYFrontendFamily::Btor2,
    AYFrontendFamily::AYOnly,
    AYFrontendFamily::VmtReplay,
    AYFrontendFamily::WitnessReplay,
];

/// Core adoption frontend family codes exposed by compatible AY analytical/proof lanes.
pub static AY_SHARED_ENGINE_ADOPTION_FRONTEND_FAMILIES: [&str; 8] = [
    "tla_plus",
    "quint",
    "mcc_petri",
    "aiger",
    "btor2",
    "ay_analytical",
    "vmt_transition_system",
    "witness_replay",
];

/// Reserved adoption frontend family codes blocked until importer registration.
pub static AY_SHARED_ENGINE_BLOCKED_ADOPTION_FRONTEND_FAMILIES: [&str; 1] = ["future_importer"];

/// Shared AY engine lanes listed in stable evidence order.
pub static AY_SHARED_ENGINE_LANES: [AYSharedEngineLane; 5] = [
    AYSharedEngineLane::AllSatEnumeration,
    AYSharedEngineLane::Bmc,
    AYSharedEngineLane::Chc,
    AYSharedEngineLane::Pdr,
    AYSharedEngineLane::KInduction,
];

const ALL_SAT_PREREQUISITES: &[&str] = &[
    "typed_symbolic_variable_vector",
    "state_or_property_query_predicate",
    "typed_transition_system_or_state_predicate",
    "finite_model_projection",
    "model_blocking_clause_projection",
    "solver_logic_qf_lia_or_qf_auflia",
];

const ALL_SAT_PROOF_OBLIGATIONS: &[&str] = &[
    "assert_symbolic_enumeration_predicate",
    "query_next_satisfying_model",
    "consumer_validate_each_model",
    "assert_model_blocking_clause_after_acceptance",
    "terminate_on_unsat_or_configured_limit",
];

const BMC_PREREQUISITES: &[&str] = &[
    "typed_state_vector",
    "initial_state_predicate",
    "step_indexed_transition_relation",
    "safety_property_at_each_bound",
    "finite_bmc_bound",
    "solver_logic_qf_lia_or_qf_auflia",
];

const BMC_PROOF_OBLIGATIONS: &[&str] = &[
    "assert_init_at_step_0",
    "assert_transition_unrolling_0_to_k",
    "query_exists_violation_step_0_to_k",
    "consumer_validate_counterexample_model",
];

const CHC_PREREQUISITES: &[&str] = &[
    "typed_state_vector",
    "initial_state_predicate",
    "current_next_transition_relation",
    "safety_property",
    "normalized_horn_clause_problem",
];

const CHC_PROOF_OBLIGATIONS: &[&str] = &[
    "initiation_init_implies_inv",
    "consecution_inv_and_next_implies_inv_prime",
    "query_inv_and_not_safety_implies_false",
];

const PDR_PREREQUISITES: &[&str] = &[
    "normalized_chc_problem",
    "transition_system_encoded_as_chc",
    "invariant_predicate_signature",
    "safety_query_clause",
    "typed_chc_proof_transcript_boundary",
];

const PDR_PROOF_OBLIGATIONS: &[&str] = &[
    "safe_result_supplies_inductive_invariant",
    "unsafe_result_supplies_init_next_not_safety_trace",
    "unknown_result_is_non_proof",
    "consumer_validate_chc_proof_transcript",
];

const KIND_PREREQUISITES: &[&str] = &[
    "typed_state_vector",
    "initial_state_predicate",
    "current_next_transition_relation",
    "safety_property",
    "max_k_and_start_k",
    "solver_logic_qf_lia_or_qf_auflia",
];

const KIND_PROOF_OBLIGATIONS: &[&str] = &[
    "base_case_no_reachable_violation_0_to_k",
    "induction_hypothesis_safety_on_k_consecutive_states",
    "inductive_step_next_and_hypothesis_imply_safety",
    "sat_inductive_step_is_unknown_not_proof",
];

const TRANSITION_SYSTEM_COMPATIBILITY_NOTES: &[&str] = &[
    "frontend_must_lower_to_typed_transition_system",
    "property_must_lower_to_safety_predicate",
    "bitvector_frontends_must_preserve_word_semantics_before_entering_lane",
    "vmt_replay_must_preserve_normalized_transition_and_property_ids",
];

const CHC_COMPATIBILITY_NOTES: &[&str] = &[
    "frontend_must_lower_to_normalized_chc_or_supported_transition_system",
    "property_must_lower_to_chc_query_clause",
    "bitvector_frontends_must_preserve_word_semantics_before_entering_chc",
    "vmt_replay_must_preserve_normalized_chc_and_property_ids",
];

/// Metadata for one shared AY engine lane.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AYSharedEngineLaneMetadata {
    /// Metadata schema identifier.
    pub schema: &'static str,
    /// Metadata schema version.
    pub schema_version: u32,
    /// Engine lane described by this metadata row.
    pub lane: AYSharedEngineLane,
    /// AY backend crate or API family used by this lane.
    pub backend: &'static str,
    /// Generic, frontend-neutral prerequisites for entering this lane.
    pub generic_prerequisites: &'static [&'static str],
    /// Proof obligations or witness obligations checked by this lane.
    pub proof_obligations: &'static [&'static str],
    /// Compatible frontend families when their lowering preserves lane semantics.
    pub compatible_frontends: &'static [AYFrontendFamily],
    /// Notes that qualify semantic compatibility across frontend families.
    pub compatibility_notes: &'static [&'static str],
    /// Whether the lane contract is independent of any one source frontend.
    pub frontend_neutral: bool,
}

/// Descriptor for publishing one frontend-neutral analytical/AY proof lane row.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AYSharedProofLaneDescriptor {
    /// Descriptor schema identifier.
    pub schema: &'static str,
    /// Descriptor schema version.
    pub schema_version: u32,
    /// Engine lane used for this proof obligation.
    pub lane: AYSharedEngineLane,
    /// Proof obligation identity shared across source frontends.
    pub proof_obligation_identity: String,
    /// Source frontend family that lowered into the shared lane.
    pub source_frontend_family: AYFrontendFamily,
    /// Shared prepared-program identity for this lowered obligation.
    pub prepared_program_identity: String,
    /// Fingerprint identity for a proof transcript or proof artifact.
    pub proof_fingerprint_identity: Option<String>,
    /// Fingerprint identity for a model/counterexample artifact.
    pub model_fingerprint_identity: Option<String>,
    /// Fingerprint identity for an analytical solve certificate artifact.
    pub certificate_fingerprint_identity: Option<String>,
    /// Fingerprint identity for a replay/witness artifact.
    pub witness_fingerprint_identity: Option<String>,
    /// Required validation receipt shape before publication.
    pub validation_receipt_requirement: &'static str,
    /// Compatible frontend families when they preserve the lane contract.
    pub compatible_frontends: &'static [AYFrontendFamily],
}

#[derive(Debug, Clone, Copy)]
struct AYAnalyticalBigWinReceipts<'a> {
    proof_receipt: &'a AYProofValidationReceipt,
    model_receipt: Option<&'a AYProofValidationReceipt>,
    certificate_receipt: Option<&'a AYProofValidationReceipt>,
    witness_receipt: &'a AYProofValidationReceipt,
}

impl AYSharedProofLaneDescriptor {
    /// Build a descriptor for one prepared proof obligation.
    #[must_use]
    pub fn new(
        lane: AYSharedEngineLane,
        source_frontend_family: AYFrontendFamily,
        prepared_program_identity: impl Into<String>,
        proof_obligation_identity: impl Into<String>,
    ) -> Self {
        Self {
            schema: AY_SHARED_PROOF_LANE_DESCRIPTOR_SCHEMA,
            schema_version: AY_SHARED_PROOF_LANE_DESCRIPTOR_SCHEMA_VERSION,
            lane,
            proof_obligation_identity: proof_obligation_identity.into(),
            source_frontend_family,
            prepared_program_identity: prepared_program_identity.into(),
            proof_fingerprint_identity: None,
            model_fingerprint_identity: None,
            certificate_fingerprint_identity: None,
            witness_fingerprint_identity: None,
            validation_receipt_requirement:
                "validator_backed_receipt_for_proof_obligation_fingerprint",
            compatible_frontends: ay_shared_engine_lane_metadata(lane).compatible_frontends,
        }
    }

    /// Attach a proof transcript/artifact fingerprint identity.
    #[must_use]
    pub fn with_proof_fingerprint_identity(mut self, identity: impl Into<String>) -> Self {
        self.proof_fingerprint_identity = non_empty_string(identity.into());
        self
    }

    /// Attach a model/counterexample fingerprint identity.
    #[must_use]
    pub fn with_model_fingerprint_identity(mut self, identity: impl Into<String>) -> Self {
        self.model_fingerprint_identity = non_empty_string(identity.into());
        self
    }

    /// Attach an analytical solve certificate fingerprint identity.
    #[must_use]
    pub fn with_certificate_fingerprint_identity(mut self, identity: impl Into<String>) -> Self {
        self.certificate_fingerprint_identity = non_empty_string(identity.into());
        self
    }

    /// Attach a replay/witness fingerprint identity.
    #[must_use]
    pub fn with_witness_fingerprint_identity(mut self, identity: impl Into<String>) -> Self {
        self.witness_fingerprint_identity = non_empty_string(identity.into());
        self
    }

    /// Return true when this descriptor can be used by `frontend`.
    #[must_use]
    pub fn supports_frontend(&self, frontend: AYFrontendFamily) -> bool {
        self.compatible_frontends.contains(&frontend)
    }

    /// Fingerprint identities supplied by this descriptor.
    #[must_use]
    pub fn fingerprint_identities(&self) -> Vec<&str> {
        [
            self.proof_fingerprint_identity.as_deref(),
            self.model_fingerprint_identity.as_deref(),
            self.certificate_fingerprint_identity.as_deref(),
            self.witness_fingerprint_identity.as_deref(),
        ]
        .into_iter()
        .flatten()
        .collect()
    }

    /// Compatible frontend family codes in stable order.
    #[must_use]
    pub fn compatible_frontend_codes(&self) -> Vec<&'static str> {
        self.compatible_frontends
            .iter()
            .map(|frontend| frontend.code())
            .collect()
    }

    /// Core shared-engine adoption family codes in stable order.
    #[must_use]
    pub fn compatible_frontend_family_codes(&self) -> Vec<&'static str> {
        AY_SHARED_ENGINE_ADOPTION_FRONTEND_FAMILIES.to_vec()
    }

    /// Active frontend families that can consume this descriptor today.
    #[must_use]
    pub fn active_frontend_family_codes(&self) -> Vec<&'static str> {
        self.compatible_frontend_family_codes()
    }

    /// Frontend families that are default consumers for this descriptor row.
    #[must_use]
    pub fn default_frontend_family_codes(&self) -> Vec<&'static str> {
        self.active_frontend_family_codes()
    }

    /// Compatible frontend families that are not default consumers in this row.
    #[must_use]
    pub fn remaining_frontend_family_codes(&self) -> Vec<&'static str> {
        let defaults = self.default_frontend_family_codes();
        self.compatible_frontend_family_codes()
            .into_iter()
            .filter(|family| !defaults.contains(family))
            .collect()
    }

    /// Reserved frontend families blocked from publication until registration.
    #[must_use]
    pub fn blocked_frontend_family_codes(&self) -> Vec<&'static str> {
        AY_SHARED_ENGINE_BLOCKED_ADOPTION_FRONTEND_FAMILIES.to_vec()
    }

    /// Canonical frontend families that may claim this analytical solve as a shared-engine win.
    #[must_use]
    pub fn analytical_big_win_beneficiary_frontend_family_codes(&self) -> Vec<&'static str> {
        self.compatible_frontend_family_codes()
    }

    /// Stable frontend-neutral proof-lane identity for adoption evidence.
    #[must_use]
    pub fn shared_proof_lane_identity(&self) -> String {
        evidence_value(&format!(
            "{}:{}:{}",
            self.lane.code(),
            self.prepared_program_identity,
            self.proof_obligation_identity
        ))
    }

    /// Stable frontend-neutral analytical solve contract identity.
    #[must_use]
    pub fn analytical_solve_contract_identity(&self) -> String {
        evidence_value(&format!(
            "{}:{}",
            AY_ANALYTICAL_PROOF_SHARED_ENGINE_COMPONENT,
            self.shared_proof_lane_identity()
        ))
    }

    /// Stable frontend-neutral big-win detector identity for this solve lane.
    #[must_use]
    pub fn analytical_big_win_detection_identity(&self) -> String {
        evidence_value(&format!(
            "{}:{}:{}",
            AY_ANALYTICAL_BIG_WIN_DETECTION_RULE,
            self.lane.code(),
            self.proof_obligation_identity
        ))
    }

    /// Fingerprint identity list rendered in a stable proof/model/witness order.
    #[must_use]
    pub fn shared_fingerprint_identity_list(&self) -> String {
        self.fingerprint_identities()
            .into_iter()
            .map(evidence_value)
            .collect::<Vec<_>>()
            .join(",")
    }

    /// Validation receipt kinds required for analytical big-win publication.
    #[must_use]
    pub fn required_analytical_big_win_receipt_kind_codes(&self) -> Vec<&'static str> {
        let mut required_receipt_kinds = vec![AYProofValidationReceiptKind::ProofTranscript.code()];
        if self.model_fingerprint_identity.is_some() {
            required_receipt_kinds.push(AYProofValidationReceiptKind::Model.code());
        }
        if self.certificate_fingerprint_identity.is_some() {
            required_receipt_kinds.push(AYProofValidationReceiptKind::Certificate.code());
        }
        required_receipt_kinds.push(AYProofValidationReceiptKind::Witness.code());
        required_receipt_kinds
    }

    fn validated_receipt_for_fingerprint<'a>(
        &self,
        receipt: Option<&'a AYProofValidationReceipt>,
        validation_kind: AYProofValidationReceiptKind,
        fingerprint_identity: &str,
    ) -> Option<&'a AYProofValidationReceipt> {
        receipt.filter(|receipt| {
            receipt.validation_kind == validation_kind
                && receipt.validates_fingerprint_for(self, fingerprint_identity)
        })
    }

    fn validated_analytical_big_win_receipts<'a>(
        &self,
        proof_receipt: Option<&'a AYProofValidationReceipt>,
        model_receipt: Option<&'a AYProofValidationReceipt>,
        certificate_receipt: Option<&'a AYProofValidationReceipt>,
        witness_receipt: Option<&'a AYProofValidationReceipt>,
    ) -> Option<AYAnalyticalBigWinReceipts<'a>> {
        if !self.supports_frontend(self.source_frontend_family) {
            return None;
        }
        let proof_fingerprint = self.proof_fingerprint_identity.as_deref()?;
        let witness_fingerprint = self.witness_fingerprint_identity.as_deref()?;
        let proof_receipt = self.validated_receipt_for_fingerprint(
            proof_receipt,
            AYProofValidationReceiptKind::ProofTranscript,
            proof_fingerprint,
        )?;
        let model_receipt =
            if let Some(model_fingerprint) = self.model_fingerprint_identity.as_deref() {
                Some(self.validated_receipt_for_fingerprint(
                    model_receipt,
                    AYProofValidationReceiptKind::Model,
                    model_fingerprint,
                )?)
            } else {
                None
            };
        let certificate_receipt = if let Some(certificate_fingerprint) =
            self.certificate_fingerprint_identity.as_deref()
        {
            Some(self.validated_receipt_for_fingerprint(
                certificate_receipt,
                AYProofValidationReceiptKind::Certificate,
                certificate_fingerprint,
            )?)
        } else {
            None
        };
        let witness_receipt = self.validated_receipt_for_fingerprint(
            witness_receipt,
            AYProofValidationReceiptKind::Witness,
            witness_fingerprint,
        )?;
        Some(AYAnalyticalBigWinReceipts {
            proof_receipt,
            model_receipt,
            certificate_receipt,
            witness_receipt,
        })
    }

    /// True only when every analytical big-win artifact has a matching validator receipt.
    #[must_use]
    pub fn can_publish_analytical_big_win_with_validation_receipts(
        &self,
        proof_receipt: Option<&AYProofValidationReceipt>,
        model_receipt: Option<&AYProofValidationReceipt>,
        certificate_receipt: Option<&AYProofValidationReceipt>,
        witness_receipt: Option<&AYProofValidationReceipt>,
    ) -> bool {
        self.validated_analytical_big_win_receipts(
            proof_receipt,
            model_receipt,
            certificate_receipt,
            witness_receipt,
        )
        .is_some()
    }

    /// Render the frontend-neutral analytical solve contract before publication.
    ///
    /// This row describes the reusable solve identity, expected artifacts, and
    /// receipt requirement. Publication rows remain gated on a validator-backed
    /// receipt for one of the declared fingerprints.
    #[must_use]
    pub fn render_analytical_solve_contract_evidence(&self, scope: &str) -> String {
        let rows = vec![
            (
                "schema".to_string(),
                AY_ANALYTICAL_SOLVE_CONTRACT_SCHEMA.to_string(),
            ),
            (
                "schema_version".to_string(),
                AY_ANALYTICAL_SOLVE_CONTRACT_SCHEMA_VERSION.to_string(),
            ),
            (
                "analytical_solve_contract_identity".to_string(),
                self.analytical_solve_contract_identity(),
            ),
            (
                "shared_proof_lane_identity".to_string(),
                self.shared_proof_lane_identity(),
            ),
            (
                "analytical_big_win_detection_rule".to_string(),
                AY_ANALYTICAL_BIG_WIN_DETECTION_RULE.to_string(),
            ),
            (
                "analytical_big_win_detection_basis".to_string(),
                AY_ANALYTICAL_BIG_WIN_DETECTION_BASIS.to_string(),
            ),
            (
                "analytical_big_win_detection_identity".to_string(),
                self.analytical_big_win_detection_identity(),
            ),
            (
                "shared_engine_component".to_string(),
                AY_ANALYTICAL_PROOF_SHARED_ENGINE_COMPONENT.to_string(),
            ),
            (
                "origin_frontend".to_string(),
                self.source_frontend_family.adoption_code().to_string(),
            ),
            ("lane".to_string(), self.lane.code().to_string()),
            (
                "proof_obligation_identity".to_string(),
                evidence_value(&self.proof_obligation_identity),
            ),
            (
                "prepared_program_identity".to_string(),
                evidence_value(&self.prepared_program_identity),
            ),
            (
                "proof_fingerprint_identity".to_string(),
                evidence_optional(self.proof_fingerprint_identity.as_deref()),
            ),
            (
                "model_fingerprint_identity".to_string(),
                evidence_optional(self.model_fingerprint_identity.as_deref()),
            ),
            (
                "certificate_fingerprint_identity".to_string(),
                evidence_optional(self.certificate_fingerprint_identity.as_deref()),
            ),
            (
                "witness_fingerprint_identity".to_string(),
                evidence_optional(self.witness_fingerprint_identity.as_deref()),
            ),
            (
                "shared_fingerprint_identities".to_string(),
                self.shared_fingerprint_identity_list(),
            ),
            (
                "validation_receipt_required".to_string(),
                "true".to_string(),
            ),
            (
                "proof_and_witness_receipts_required_for_big_win".to_string(),
                "true".to_string(),
            ),
            (
                "proof_receipt_required".to_string(),
                self.proof_fingerprint_identity.is_some().to_string(),
            ),
            (
                "model_receipt_required".to_string(),
                self.model_fingerprint_identity.is_some().to_string(),
            ),
            (
                "certificate_receipt_required".to_string(),
                self.certificate_fingerprint_identity.is_some().to_string(),
            ),
            (
                "witness_receipt_required".to_string(),
                self.witness_fingerprint_identity.is_some().to_string(),
            ),
            (
                "validation_receipt_requirement".to_string(),
                self.validation_receipt_requirement.to_string(),
            ),
            (
                "compatible_frontend_families".to_string(),
                frontend_family_codes_to_evidence(self.compatible_frontend_family_codes()),
            ),
            (
                "active_frontend_families".to_string(),
                frontend_family_codes_to_evidence(self.active_frontend_family_codes()),
            ),
            (
                "default_frontend_families".to_string(),
                frontend_family_codes_to_evidence(self.default_frontend_family_codes()),
            ),
            (
                "remaining_frontend_families".to_string(),
                frontend_family_codes_to_evidence(self.remaining_frontend_family_codes()),
            ),
            (
                "blocked_frontend_families".to_string(),
                frontend_family_codes_to_evidence(self.blocked_frontend_family_codes()),
            ),
            (
                "analytical_big_win_beneficiary_frontend_families".to_string(),
                frontend_family_codes_to_evidence(
                    self.analytical_big_win_beneficiary_frontend_family_codes(),
                ),
            ),
            (
                "compatible_frontend_codes".to_string(),
                self.compatible_frontend_codes().join(","),
            ),
            (
                "default_frontend_policy".to_string(),
                AY_ANALYTICAL_BIG_WIN_DEFAULT_FRONTEND_POLICY.to_string(),
            ),
            (
                "explicit_search_replacement_policy".to_string(),
                AY_ANALYTICAL_BIG_WIN_EXPLICIT_SEARCH_REPLACEMENT_POLICY.to_string(),
            ),
            (
                "fail_closed_blockers".to_string(),
                AY_ANALYTICAL_BIG_WIN_FAIL_CLOSED_BLOCKERS.to_string(),
            ),
            (
                "extraction_status".to_string(),
                AY_SHARED_PROOF_LANE_EXTRACTION_STATUS.to_string(),
            ),
            (
                "blocker_status".to_string(),
                AY_SHARED_PROOF_LANE_BLOCKER_STATUS.to_string(),
            ),
            (
                "frontend_family_blockers".to_string(),
                AY_SHARED_PROOF_LANE_FRONTEND_FAMILY_BLOCKERS.to_string(),
            ),
            (
                "shared_owner".to_string(),
                AY_SHARED_PROOF_LANE_SHARED_OWNER.to_string(),
            ),
            ("publishable_shared_win".to_string(), "false".to_string()),
            (
                "publication_blocker".to_string(),
                "missing_validator_backed_proof_and_witness_receipts".to_string(),
            ),
        ];
        format!(
            "{} ay_analytical_solve_contract {}",
            scope,
            key_value_pairs_to_text(&rows)
        )
    }

    /// Render frontend-neutral analytical big-win evidence with distinct
    /// proof and witness receipts.
    ///
    /// This is the generic shared-engine publication row for frontends that
    /// claim a reusable analytical solve win. The row is intentionally keyed by
    /// prepared proof obligation and declared fingerprints, not by any
    /// frontend-specific artifact format.
    #[must_use]
    pub fn render_analytical_big_win_evidence_with_receipts(
        &self,
        scope: &str,
        proof_receipt: Option<&AYProofValidationReceipt>,
        witness_receipt: Option<&AYProofValidationReceipt>,
    ) -> Option<String> {
        self.render_analytical_big_win_evidence_with_artifact_receipts(
            scope,
            proof_receipt,
            None,
            witness_receipt,
        )
    }

    /// Render frontend-neutral analytical big-win evidence with distinct
    /// validator-backed receipts for every declared solve artifact.
    ///
    /// Proof and witness receipts are always required for a big-win
    /// publication. Model and certificate receipts are required when the
    /// descriptor declares those artifact fingerprints. This keeps SAT,
    /// UNSAT/proof, certificate-producing, and replay-oriented solver families
    /// on the same fail-closed evidence path.
    #[must_use]
    pub fn render_analytical_big_win_evidence_with_artifact_receipts(
        &self,
        scope: &str,
        proof_receipt: Option<&AYProofValidationReceipt>,
        model_receipt: Option<&AYProofValidationReceipt>,
        witness_receipt: Option<&AYProofValidationReceipt>,
    ) -> Option<String> {
        self.render_analytical_big_win_evidence_with_validation_receipts(
            scope,
            proof_receipt,
            model_receipt,
            None,
            witness_receipt,
        )
    }

    /// Render frontend-neutral analytical big-win evidence with the complete
    /// proof/model/certificate/witness receipt set required by this descriptor.
    #[must_use]
    pub fn render_analytical_big_win_evidence_with_validation_receipts(
        &self,
        scope: &str,
        proof_receipt: Option<&AYProofValidationReceipt>,
        model_receipt: Option<&AYProofValidationReceipt>,
        certificate_receipt: Option<&AYProofValidationReceipt>,
        witness_receipt: Option<&AYProofValidationReceipt>,
    ) -> Option<String> {
        let proof_fingerprint = self.proof_fingerprint_identity.as_deref()?;
        let witness_fingerprint = self.witness_fingerprint_identity.as_deref()?;
        let receipts = self.validated_analytical_big_win_receipts(
            proof_receipt,
            model_receipt,
            certificate_receipt,
            witness_receipt,
        )?;
        let proof_receipt = receipts.proof_receipt;
        let model_receipt = receipts.model_receipt;
        let certificate_receipt = receipts.certificate_receipt;
        let witness_receipt = receipts.witness_receipt;
        let required_receipt_kinds = self.required_analytical_big_win_receipt_kind_codes();
        let rows = vec![
            (
                "schema".to_string(),
                AY_ANALYTICAL_SOLVE_CONTRACT_SCHEMA.to_string(),
            ),
            (
                "schema_version".to_string(),
                AY_ANALYTICAL_SOLVE_CONTRACT_SCHEMA_VERSION.to_string(),
            ),
            (
                "analytical_solve_contract_identity".to_string(),
                self.analytical_solve_contract_identity(),
            ),
            (
                "shared_proof_lane_identity".to_string(),
                self.shared_proof_lane_identity(),
            ),
            (
                "analytical_big_win_detection_rule".to_string(),
                AY_ANALYTICAL_BIG_WIN_DETECTION_RULE.to_string(),
            ),
            (
                "analytical_big_win_detection_basis".to_string(),
                AY_ANALYTICAL_BIG_WIN_DETECTION_BASIS.to_string(),
            ),
            (
                "analytical_big_win_detection_identity".to_string(),
                self.analytical_big_win_detection_identity(),
            ),
            (
                "shared_engine_component".to_string(),
                AY_ANALYTICAL_PROOF_SHARED_ENGINE_COMPONENT.to_string(),
            ),
            (
                "origin_frontend".to_string(),
                self.source_frontend_family.adoption_code().to_string(),
            ),
            (
                "first_beneficiary".to_string(),
                self.source_frontend_family.adoption_code().to_string(),
            ),
            (
                "second_beneficiary".to_string(),
                ay_publication_second_beneficiary(self.source_frontend_family).to_string(),
            ),
            (
                "beneficiary_frontend_families".to_string(),
                frontend_family_codes_to_evidence(
                    self.analytical_big_win_beneficiary_frontend_family_codes(),
                ),
            ),
            (
                "active_frontend_families".to_string(),
                frontend_family_codes_to_evidence(self.active_frontend_family_codes()),
            ),
            (
                "default_frontend_families".to_string(),
                frontend_family_codes_to_evidence(self.default_frontend_family_codes()),
            ),
            (
                "remaining_frontend_families".to_string(),
                frontend_family_codes_to_evidence(self.remaining_frontend_family_codes()),
            ),
            (
                "blocked_frontend_families".to_string(),
                frontend_family_codes_to_evidence(self.blocked_frontend_family_codes()),
            ),
            ("lane".to_string(), self.lane.code().to_string()),
            (
                "proof_obligation_identity".to_string(),
                evidence_value(&self.proof_obligation_identity),
            ),
            (
                "prepared_program_identity".to_string(),
                evidence_value(&self.prepared_program_identity),
            ),
            (
                "proof_fingerprint_identity".to_string(),
                evidence_value(proof_fingerprint),
            ),
            (
                "model_fingerprint_identity".to_string(),
                evidence_optional(self.model_fingerprint_identity.as_deref()),
            ),
            (
                "certificate_fingerprint_identity".to_string(),
                evidence_optional(self.certificate_fingerprint_identity.as_deref()),
            ),
            (
                "witness_fingerprint_identity".to_string(),
                evidence_value(witness_fingerprint),
            ),
            (
                "shared_fingerprint_identities".to_string(),
                self.shared_fingerprint_identity_list(),
            ),
            (
                "validation_receipt_requirement".to_string(),
                self.validation_receipt_requirement.to_string(),
            ),
            (
                "required_receipt_kinds".to_string(),
                required_receipt_kinds.join(","),
            ),
            ("validation_receipt_backed".to_string(), "true".to_string()),
            (
                "proof_and_witness_receipts_required_for_big_win".to_string(),
                "true".to_string(),
            ),
            ("proof_receipt_required".to_string(), "true".to_string()),
            (
                "model_receipt_required".to_string(),
                self.model_fingerprint_identity.is_some().to_string(),
            ),
            (
                "certificate_receipt_required".to_string(),
                self.certificate_fingerprint_identity.is_some().to_string(),
            ),
            ("witness_receipt_required".to_string(), "true".to_string()),
            (
                "proof_receipt_identity".to_string(),
                evidence_value(&proof_receipt.receipt_identity),
            ),
            (
                "proof_receipt_schema".to_string(),
                proof_receipt.schema.to_string(),
            ),
            (
                "proof_receipt_validation_kind".to_string(),
                proof_receipt.validation_kind.code().to_string(),
            ),
            (
                "proof_receipt_status".to_string(),
                proof_receipt.status.code().to_string(),
            ),
            (
                "proof_receipt_validated_fingerprint_identity".to_string(),
                evidence_value(&proof_receipt.validated_fingerprint_identity),
            ),
            (
                "model_receipt_identity".to_string(),
                evidence_optional(model_receipt.map(|receipt| receipt.receipt_identity.as_str())),
            ),
            (
                "model_receipt_schema".to_string(),
                evidence_optional(model_receipt.map(|receipt| receipt.schema)),
            ),
            (
                "model_receipt_validation_kind".to_string(),
                evidence_optional(model_receipt.map(|receipt| receipt.validation_kind.code())),
            ),
            (
                "model_receipt_status".to_string(),
                evidence_optional(model_receipt.map(|receipt| receipt.status.code())),
            ),
            (
                "model_receipt_validated_fingerprint_identity".to_string(),
                evidence_optional(
                    model_receipt.map(|receipt| receipt.validated_fingerprint_identity.as_str()),
                ),
            ),
            (
                "certificate_receipt_identity".to_string(),
                evidence_optional(
                    certificate_receipt.map(|receipt| receipt.receipt_identity.as_str()),
                ),
            ),
            (
                "certificate_receipt_schema".to_string(),
                evidence_optional(certificate_receipt.map(|receipt| receipt.schema)),
            ),
            (
                "certificate_receipt_validation_kind".to_string(),
                evidence_optional(
                    certificate_receipt.map(|receipt| receipt.validation_kind.code()),
                ),
            ),
            (
                "certificate_receipt_status".to_string(),
                evidence_optional(certificate_receipt.map(|receipt| receipt.status.code())),
            ),
            (
                "certificate_receipt_validated_fingerprint_identity".to_string(),
                evidence_optional(
                    certificate_receipt
                        .map(|receipt| receipt.validated_fingerprint_identity.as_str()),
                ),
            ),
            (
                "witness_receipt_identity".to_string(),
                evidence_value(&witness_receipt.receipt_identity),
            ),
            (
                "witness_receipt_schema".to_string(),
                witness_receipt.schema.to_string(),
            ),
            (
                "witness_receipt_validation_kind".to_string(),
                witness_receipt.validation_kind.code().to_string(),
            ),
            (
                "witness_receipt_status".to_string(),
                witness_receipt.status.code().to_string(),
            ),
            (
                "witness_receipt_validated_fingerprint_identity".to_string(),
                evidence_value(&witness_receipt.validated_fingerprint_identity),
            ),
            (
                "compatible_frontend_families".to_string(),
                frontend_family_codes_to_evidence(self.compatible_frontend_family_codes()),
            ),
            (
                "default_compatible_frontend_families".to_string(),
                frontend_family_codes_to_evidence(self.default_frontend_family_codes()),
            ),
            (
                "remaining_compatible_frontend_families".to_string(),
                frontend_family_codes_to_evidence(self.remaining_frontend_family_codes()),
            ),
            (
                "compatible_frontend_codes".to_string(),
                self.compatible_frontend_codes().join(","),
            ),
            (
                "default_frontend_policy".to_string(),
                AY_ANALYTICAL_BIG_WIN_DEFAULT_FRONTEND_POLICY.to_string(),
            ),
            (
                "explicit_search_replacement_policy".to_string(),
                AY_ANALYTICAL_BIG_WIN_EXPLICIT_SEARCH_REPLACEMENT_POLICY.to_string(),
            ),
            (
                "explicit_search_replacement_admitted".to_string(),
                "true".to_string(),
            ),
            ("publication_blocker".to_string(), "none".to_string()),
            (
                "extraction_status".to_string(),
                AY_SHARED_PROOF_LANE_EXTRACTION_STATUS.to_string(),
            ),
            (
                "blocker_status".to_string(),
                AY_SHARED_PROOF_LANE_BLOCKER_STATUS.to_string(),
            ),
            (
                "frontend_family_blockers".to_string(),
                AY_SHARED_PROOF_LANE_FRONTEND_FAMILY_BLOCKERS.to_string(),
            ),
            (
                "shared_owner".to_string(),
                AY_SHARED_PROOF_LANE_SHARED_OWNER.to_string(),
            ),
            ("publishable_shared_win".to_string(), "true".to_string()),
        ];
        Some(format!(
            "{} ay_analytical_big_win_evidence {}",
            scope,
            key_value_pairs_to_text(&rows)
        ))
    }

    /// True only when a validator-backed receipt matches this obligation/fingerprint boundary.
    #[must_use]
    pub fn can_publish_with_receipt(&self, receipt: Option<&AYProofValidationReceipt>) -> bool {
        self.supports_frontend(self.source_frontend_family)
            && receipt.is_some_and(|receipt| receipt.has_publishable_shape_for(self))
    }

    /// Render a publication row only after receipt validation succeeds.
    #[must_use]
    pub fn render_publication_evidence(
        &self,
        scope: &str,
        receipt: Option<&AYProofValidationReceipt>,
    ) -> Option<String> {
        if !self.supports_frontend(self.source_frontend_family) {
            return None;
        }
        let receipt = receipt.filter(|receipt| receipt.has_publishable_shape_for(self))?;
        let rows = vec![
            ("schema".to_string(), self.schema.to_string()),
            (
                "schema_version".to_string(),
                self.schema_version.to_string(),
            ),
            (
                "shared_proof_lane_identity".to_string(),
                self.shared_proof_lane_identity(),
            ),
            (
                "analytical_solve_contract_identity".to_string(),
                self.analytical_solve_contract_identity(),
            ),
            (
                "analytical_big_win_detection_rule".to_string(),
                AY_ANALYTICAL_BIG_WIN_DETECTION_RULE.to_string(),
            ),
            (
                "analytical_big_win_detection_basis".to_string(),
                AY_ANALYTICAL_BIG_WIN_DETECTION_BASIS.to_string(),
            ),
            (
                "analytical_big_win_detection_identity".to_string(),
                self.analytical_big_win_detection_identity(),
            ),
            (
                "shared_engine_component".to_string(),
                AY_ANALYTICAL_PROOF_SHARED_ENGINE_COMPONENT.to_string(),
            ),
            (
                "origin_frontend".to_string(),
                self.source_frontend_family.adoption_code().to_string(),
            ),
            (
                "first_beneficiary".to_string(),
                self.source_frontend_family.adoption_code().to_string(),
            ),
            (
                "second_beneficiary".to_string(),
                ay_publication_second_beneficiary(self.source_frontend_family).to_string(),
            ),
            (
                "extraction_status".to_string(),
                AY_SHARED_PROOF_LANE_EXTRACTION_STATUS.to_string(),
            ),
            (
                "blocker_status".to_string(),
                AY_SHARED_PROOF_LANE_BLOCKER_STATUS.to_string(),
            ),
            (
                "frontend_family_blockers".to_string(),
                AY_SHARED_PROOF_LANE_FRONTEND_FAMILY_BLOCKERS.to_string(),
            ),
            (
                "shared_owner".to_string(),
                AY_SHARED_PROOF_LANE_SHARED_OWNER.to_string(),
            ),
            ("lane".to_string(), self.lane.code().to_string()),
            (
                "source_frontend_family".to_string(),
                self.source_frontend_family.adoption_code().to_string(),
            ),
            (
                "prepared_program_identity".to_string(),
                evidence_value(&self.prepared_program_identity),
            ),
            (
                "proof_obligation_identity".to_string(),
                evidence_value(&self.proof_obligation_identity),
            ),
            (
                "proof_fingerprint_identity".to_string(),
                evidence_optional(self.proof_fingerprint_identity.as_deref()),
            ),
            (
                "model_fingerprint_identity".to_string(),
                evidence_optional(self.model_fingerprint_identity.as_deref()),
            ),
            (
                "certificate_fingerprint_identity".to_string(),
                evidence_optional(self.certificate_fingerprint_identity.as_deref()),
            ),
            (
                "witness_fingerprint_identity".to_string(),
                evidence_optional(self.witness_fingerprint_identity.as_deref()),
            ),
            (
                "shared_fingerprint_identities".to_string(),
                self.shared_fingerprint_identity_list(),
            ),
            (
                "validation_receipt_requirement".to_string(),
                self.validation_receipt_requirement.to_string(),
            ),
            ("validation_receipt_backed".to_string(), "true".to_string()),
            (
                "proof_receipt_identity".to_string(),
                evidence_value(&receipt.receipt_identity),
            ),
            (
                "validation_receipt_identity".to_string(),
                evidence_value(&receipt.receipt_identity),
            ),
            (
                "validation_receipt_schema".to_string(),
                receipt.schema.to_string(),
            ),
            (
                "validation_kind".to_string(),
                receipt.validation_kind.code().to_string(),
            ),
            (
                "validation_status".to_string(),
                receipt.status.code().to_string(),
            ),
            (
                "validated_fingerprint_identity".to_string(),
                evidence_value(&receipt.validated_fingerprint_identity),
            ),
            (
                "compatible_frontend_families".to_string(),
                frontend_family_codes_to_evidence(self.compatible_frontend_family_codes()),
            ),
            (
                "active_frontend_families".to_string(),
                frontend_family_codes_to_evidence(self.active_frontend_family_codes()),
            ),
            (
                "default_frontend_families".to_string(),
                frontend_family_codes_to_evidence(self.default_frontend_family_codes()),
            ),
            (
                "remaining_frontend_families".to_string(),
                frontend_family_codes_to_evidence(self.remaining_frontend_family_codes()),
            ),
            (
                "blocked_frontend_families".to_string(),
                frontend_family_codes_to_evidence(self.blocked_frontend_family_codes()),
            ),
            (
                "compatible_frontend_codes".to_string(),
                self.compatible_frontend_codes().join(","),
            ),
            ("publishable_shared_win".to_string(), "true".to_string()),
        ];
        Some(format!(
            "{} ay_shared_proof_lane_publication {}",
            scope,
            key_value_pairs_to_text(&rows)
        ))
    }

    /// Render hardware/register-vector proof-lane adoption evidence.
    ///
    /// This is the shared path for hardware frontends that publish the AY proof
    /// lane against a prepared register-vector transition system. Publication
    /// remains gated on the same validator-backed receipt shape as the generic
    /// proof-lane row.
    #[must_use]
    pub fn render_hardware_adoption_evidence(
        &self,
        scope: &str,
        row_kind: &str,
        register_vector_identity: &str,
        first_beneficiary: &str,
        second_beneficiary: &str,
        receipt: Option<&AYProofValidationReceipt>,
    ) -> Option<String> {
        self.render_hardware_adoption_evidence_with_receipts(
            scope,
            row_kind,
            register_vector_identity,
            first_beneficiary,
            second_beneficiary,
            receipt,
            None,
        )
    }

    /// Render hardware/register-vector proof-lane adoption evidence with
    /// distinct validator-backed receipts for proof and witness fingerprints.
    ///
    /// Hardware descriptors usually publish both the solver proof fingerprint
    /// and the replay/witness fingerprint. The row is publishable only when the
    /// proof receipt validates the proof fingerprint and, when present, the
    /// witness receipt validates the witness fingerprint.
    #[must_use]
    pub fn render_hardware_adoption_evidence_with_receipts(
        &self,
        scope: &str,
        row_kind: &str,
        register_vector_identity: &str,
        first_beneficiary: &str,
        second_beneficiary: &str,
        proof_receipt: Option<&AYProofValidationReceipt>,
        witness_receipt: Option<&AYProofValidationReceipt>,
    ) -> Option<String> {
        if !self.supports_frontend(self.source_frontend_family) {
            return None;
        }
        let proof_fingerprint = self.proof_fingerprint_identity.as_deref()?;
        let proof_receipt = proof_receipt.filter(|receipt| {
            receipt.validation_kind == AYProofValidationReceiptKind::ProofTranscript
                && receipt.validates_fingerprint_for(self, proof_fingerprint)
        })?;
        let witness_receipt =
            if let Some(witness_fingerprint) = self.witness_fingerprint_identity.as_deref() {
                Some(witness_receipt.filter(|receipt| {
                    receipt.validation_kind == AYProofValidationReceiptKind::Witness
                        && receipt.validates_fingerprint_for(self, witness_fingerprint)
                })?)
            } else {
                None
            };
        let rows = vec![
            ("schema".to_string(), self.schema.to_string()),
            (
                "schema_version".to_string(),
                self.schema_version.to_string(),
            ),
            (
                "shared_proof_lane_identity".to_string(),
                self.shared_proof_lane_identity(),
            ),
            (
                "analytical_solve_contract_identity".to_string(),
                self.analytical_solve_contract_identity(),
            ),
            (
                "analytical_big_win_detection_rule".to_string(),
                AY_ANALYTICAL_BIG_WIN_DETECTION_RULE.to_string(),
            ),
            (
                "analytical_big_win_detection_basis".to_string(),
                AY_ANALYTICAL_BIG_WIN_DETECTION_BASIS.to_string(),
            ),
            (
                "analytical_big_win_detection_identity".to_string(),
                self.analytical_big_win_detection_identity(),
            ),
            (
                "origin_frontend".to_string(),
                self.source_frontend_family.adoption_code().to_string(),
            ),
            (
                "shared_engine_component".to_string(),
                AY_ANALYTICAL_PROOF_SHARED_ENGINE_COMPONENT.to_string(),
            ),
            ("lane".to_string(), self.lane.code().to_string()),
            (
                "proof_obligation_identity".to_string(),
                evidence_value(&self.proof_obligation_identity),
            ),
            (
                "prepared_program_identity".to_string(),
                evidence_value(&self.prepared_program_identity),
            ),
            (
                "register_vector_identity".to_string(),
                evidence_value(register_vector_identity),
            ),
            (
                "proof_fingerprint_identity".to_string(),
                evidence_optional(self.proof_fingerprint_identity.as_deref()),
            ),
            (
                "model_fingerprint_identity".to_string(),
                evidence_optional(self.model_fingerprint_identity.as_deref()),
            ),
            (
                "certificate_fingerprint_identity".to_string(),
                evidence_optional(self.certificate_fingerprint_identity.as_deref()),
            ),
            (
                "witness_fingerprint_identity".to_string(),
                evidence_optional(self.witness_fingerprint_identity.as_deref()),
            ),
            (
                "shared_fingerprint_identities".to_string(),
                self.shared_fingerprint_identity_list(),
            ),
            (
                "validation_receipt_requirement".to_string(),
                self.validation_receipt_requirement.to_string(),
            ),
            (
                "required_receipt_kinds".to_string(),
                if self.witness_fingerprint_identity.is_some() {
                    "proof_transcript,witness"
                } else {
                    "proof_transcript"
                }
                .to_string(),
            ),
            ("validation_receipt_backed".to_string(), "true".to_string()),
            (
                "proof_and_witness_receipts_required_for_big_win".to_string(),
                "true".to_string(),
            ),
            ("proof_receipt_required".to_string(), "true".to_string()),
            (
                "witness_receipt_required".to_string(),
                self.witness_fingerprint_identity.is_some().to_string(),
            ),
            (
                "proof_receipt_identity".to_string(),
                evidence_value(&proof_receipt.receipt_identity),
            ),
            (
                "proof_receipt_schema".to_string(),
                proof_receipt.schema.to_string(),
            ),
            (
                "proof_receipt_validation_kind".to_string(),
                proof_receipt.validation_kind.code().to_string(),
            ),
            (
                "proof_receipt_status".to_string(),
                proof_receipt.status.code().to_string(),
            ),
            (
                "proof_receipt_validated_fingerprint_identity".to_string(),
                evidence_value(&proof_receipt.validated_fingerprint_identity),
            ),
            (
                "witness_receipt_identity".to_string(),
                evidence_optional(witness_receipt.map(|receipt| receipt.receipt_identity.as_str())),
            ),
            (
                "witness_receipt_schema".to_string(),
                evidence_optional(witness_receipt.map(|receipt| receipt.schema)),
            ),
            (
                "witness_receipt_validation_kind".to_string(),
                evidence_optional(witness_receipt.map(|receipt| receipt.validation_kind.code())),
            ),
            (
                "witness_receipt_status".to_string(),
                evidence_optional(witness_receipt.map(|receipt| receipt.status.code())),
            ),
            (
                "witness_receipt_validated_fingerprint_identity".to_string(),
                evidence_optional(
                    witness_receipt.map(|receipt| receipt.validated_fingerprint_identity.as_str()),
                ),
            ),
            (
                "validation_receipt_identity".to_string(),
                evidence_value(&proof_receipt.receipt_identity),
            ),
            (
                "validation_receipt_schema".to_string(),
                proof_receipt.schema.to_string(),
            ),
            (
                "validation_kind".to_string(),
                proof_receipt.validation_kind.code().to_string(),
            ),
            (
                "validation_status".to_string(),
                proof_receipt.status.code().to_string(),
            ),
            (
                "validated_fingerprint_identity".to_string(),
                evidence_value(&proof_receipt.validated_fingerprint_identity),
            ),
            (
                "first_beneficiary".to_string(),
                evidence_value(first_beneficiary),
            ),
            (
                "second_beneficiary".to_string(),
                evidence_value(second_beneficiary),
            ),
            (
                "compatible_frontend_families".to_string(),
                frontend_family_codes_to_evidence(self.compatible_frontend_family_codes()),
            ),
            (
                "active_frontend_families".to_string(),
                frontend_family_codes_to_evidence(self.active_frontend_family_codes()),
            ),
            (
                "default_frontend_families".to_string(),
                frontend_family_codes_to_evidence(self.default_frontend_family_codes()),
            ),
            (
                "remaining_frontend_families".to_string(),
                frontend_family_codes_to_evidence(self.remaining_frontend_family_codes()),
            ),
            (
                "blocked_frontend_families".to_string(),
                frontend_family_codes_to_evidence(self.blocked_frontend_family_codes()),
            ),
            (
                "compatible_frontend_codes".to_string(),
                self.compatible_frontend_codes().join(","),
            ),
            (
                "default_frontend_policy".to_string(),
                AY_ANALYTICAL_BIG_WIN_DEFAULT_FRONTEND_POLICY.to_string(),
            ),
            (
                "explicit_search_replacement_policy".to_string(),
                AY_ANALYTICAL_BIG_WIN_EXPLICIT_SEARCH_REPLACEMENT_POLICY.to_string(),
            ),
            ("publication_blocker".to_string(), "none".to_string()),
            (
                "extraction_status".to_string(),
                AY_SHARED_PROOF_LANE_EXTRACTION_STATUS.to_string(),
            ),
            (
                "blocker_status".to_string(),
                AY_SHARED_PROOF_LANE_BLOCKER_STATUS.to_string(),
            ),
            (
                "frontend_family_blockers".to_string(),
                AY_SHARED_PROOF_LANE_FRONTEND_FAMILY_BLOCKERS.to_string(),
            ),
            (
                "shared_owner".to_string(),
                AY_SHARED_PROOF_LANE_SHARED_OWNER.to_string(),
            ),
            ("publishable_shared_win".to_string(), "true".to_string()),
        ];
        Some(format!(
            "{} {} {}",
            scope,
            evidence_value(row_kind),
            key_value_pairs_to_text(&rows)
        ))
    }
}

fn ay_publication_second_beneficiary(frontend: AYFrontendFamily) -> &'static str {
    match frontend {
        AYFrontendFamily::Tla => "mcc_petri",
        AYFrontendFamily::Quint => "mcc_petri",
        AYFrontendFamily::MccPetri => "ay_analytical",
        AYFrontendFamily::Aiger => "btor2",
        AYFrontendFamily::Btor2 => "aiger",
        AYFrontendFamily::AYOnly => "mcc_petri",
        AYFrontendFamily::VmtReplay => "ay_analytical",
        AYFrontendFamily::WitnessReplay => "ay_analytical",
        AYFrontendFamily::FutureImporter => "mcc_petri",
    }
}

impl AYSharedEngineLaneMetadata {
    /// Return true when this lane advertises compatibility for `frontend`.
    #[must_use]
    pub fn supports_frontend(&self, frontend: AYFrontendFamily) -> bool {
        self.compatible_frontends.contains(&frontend)
    }

    /// Compatible frontend family codes in stable order.
    #[must_use]
    pub fn compatible_frontend_codes(&self) -> Vec<&'static str> {
        self.compatible_frontends
            .iter()
            .map(|frontend| frontend.code())
            .collect()
    }

    /// Core shared-engine adoption family codes in stable order.
    #[must_use]
    pub fn compatible_frontend_family_codes(&self) -> Vec<&'static str> {
        AY_SHARED_ENGINE_ADOPTION_FRONTEND_FAMILIES.to_vec()
    }

    /// Compatible frontend family names in stable order.
    #[must_use]
    pub fn compatible_frontend_names(&self) -> Vec<&'static str> {
        self.compatible_frontends
            .iter()
            .map(|frontend| frontend.name())
            .collect()
    }

    /// Render this lane metadata as stable key/value rows.
    #[must_use]
    pub fn to_key_value_rows(&self) -> Vec<(String, String)> {
        vec![
            ("schema".to_string(), self.schema.to_string()),
            (
                "schema_version".to_string(),
                self.schema_version.to_string(),
            ),
            ("lane".to_string(), self.lane.code().to_string()),
            ("lane_name".to_string(), self.lane.name().to_string()),
            ("backend".to_string(), self.backend.to_string()),
            (
                "frontend_neutral".to_string(),
                self.frontend_neutral.to_string(),
            ),
            (
                "compatible_frontend_codes".to_string(),
                self.compatible_frontend_codes().join(","),
            ),
            (
                "compatible_frontend_families".to_string(),
                self.compatible_frontend_family_codes().join(","),
            ),
            (
                "compatible_frontends".to_string(),
                self.compatible_frontend_names().join(","),
            ),
            (
                "frontend_family_blockers".to_string(),
                AY_SHARED_PROOF_LANE_FRONTEND_FAMILY_BLOCKERS.to_string(),
            ),
            (
                "generic_prerequisites".to_string(),
                self.generic_prerequisites.join(","),
            ),
            (
                "proof_obligations".to_string(),
                self.proof_obligations.join(","),
            ),
            (
                "compatibility_notes".to_string(),
                self.compatibility_notes.join(","),
            ),
        ]
    }
}

/// Return metadata for one shared AY engine lane.
#[must_use]
pub fn ay_shared_engine_lane_metadata(lane: AYSharedEngineLane) -> AYSharedEngineLaneMetadata {
    match lane {
        AYSharedEngineLane::AllSatEnumeration => AYSharedEngineLaneMetadata {
            schema: AY_SHARED_ENGINE_METADATA_SCHEMA,
            schema_version: AY_SHARED_ENGINE_METADATA_SCHEMA_VERSION,
            lane,
            backend: "ay_dpll::api::Solver",
            generic_prerequisites: ALL_SAT_PREREQUISITES,
            proof_obligations: ALL_SAT_PROOF_OBLIGATIONS,
            compatible_frontends: &AY_SHARED_ENGINE_COMPATIBLE_FRONTEND_FAMILIES,
            compatibility_notes: TRANSITION_SYSTEM_COMPATIBILITY_NOTES,
            frontend_neutral: true,
        },
        AYSharedEngineLane::Bmc => AYSharedEngineLaneMetadata {
            schema: AY_SHARED_ENGINE_METADATA_SCHEMA,
            schema_version: AY_SHARED_ENGINE_METADATA_SCHEMA_VERSION,
            lane,
            backend: "ay_dpll::api::Solver",
            generic_prerequisites: BMC_PREREQUISITES,
            proof_obligations: BMC_PROOF_OBLIGATIONS,
            compatible_frontends: &AY_SHARED_ENGINE_COMPATIBLE_FRONTEND_FAMILIES,
            compatibility_notes: TRANSITION_SYSTEM_COMPATIBILITY_NOTES,
            frontend_neutral: true,
        },
        AYSharedEngineLane::Chc => AYSharedEngineLaneMetadata {
            schema: AY_SHARED_ENGINE_METADATA_SCHEMA,
            schema_version: AY_SHARED_ENGINE_METADATA_SCHEMA_VERSION,
            lane,
            backend: "ay_chc::ChcProblem",
            generic_prerequisites: CHC_PREREQUISITES,
            proof_obligations: CHC_PROOF_OBLIGATIONS,
            compatible_frontends: &AY_SHARED_ENGINE_COMPATIBLE_FRONTEND_FAMILIES,
            compatibility_notes: CHC_COMPATIBILITY_NOTES,
            frontend_neutral: true,
        },
        AYSharedEngineLane::Pdr => AYSharedEngineLaneMetadata {
            schema: AY_SHARED_ENGINE_METADATA_SCHEMA,
            schema_version: AY_SHARED_ENGINE_METADATA_SCHEMA_VERSION,
            lane,
            backend: "ay_chc::engines::solve_pdr_proof",
            generic_prerequisites: PDR_PREREQUISITES,
            proof_obligations: PDR_PROOF_OBLIGATIONS,
            compatible_frontends: &AY_SHARED_ENGINE_COMPATIBLE_FRONTEND_FAMILIES,
            compatibility_notes: CHC_COMPATIBILITY_NOTES,
            frontend_neutral: true,
        },
        AYSharedEngineLane::KInduction => AYSharedEngineLaneMetadata {
            schema: AY_SHARED_ENGINE_METADATA_SCHEMA,
            schema_version: AY_SHARED_ENGINE_METADATA_SCHEMA_VERSION,
            lane,
            backend: "tla_ay::bmc::kinduction",
            generic_prerequisites: KIND_PREREQUISITES,
            proof_obligations: KIND_PROOF_OBLIGATIONS,
            compatible_frontends: &AY_SHARED_ENGINE_COMPATIBLE_FRONTEND_FAMILIES,
            compatibility_notes: TRANSITION_SYSTEM_COMPATIBILITY_NOTES,
            frontend_neutral: true,
        },
    }
}

/// Return metadata for all shared AY engine lanes in stable order.
#[must_use]
pub fn ay_shared_engine_all_lane_metadata() -> Vec<AYSharedEngineLaneMetadata> {
    AY_SHARED_ENGINE_LANES
        .iter()
        .copied()
        .map(ay_shared_engine_lane_metadata)
        .collect()
}

/// Render one shared engine lane as a frontend-neutral evidence line.
#[must_use]
pub fn render_ay_shared_engine_lane_evidence(scope: &str, lane: AYSharedEngineLane) -> String {
    let metadata = ay_shared_engine_lane_metadata(lane);
    let rows = metadata.to_key_value_rows();
    format!(
        "{} ay_shared_engine_lane_metadata {}",
        scope,
        key_value_pairs_to_text(&rows)
    )
}

/// Render all shared engine lanes as stable key/value rows.
#[must_use]
pub fn ay_shared_engine_evidence_key_value_rows() -> Vec<(String, String)> {
    let mut rows = vec![
        (
            "schema".to_string(),
            AY_SHARED_ENGINE_METADATA_SCHEMA.to_string(),
        ),
        (
            "schema_version".to_string(),
            AY_SHARED_ENGINE_METADATA_SCHEMA_VERSION.to_string(),
        ),
        (
            "shared_engine_component".to_string(),
            AY_ANALYTICAL_PROOF_SHARED_ENGINE_COMPONENT.to_string(),
        ),
        (
            "lanes".to_string(),
            AY_SHARED_ENGINE_LANES
                .iter()
                .map(|lane| lane.code())
                .collect::<Vec<_>>()
                .join(","),
        ),
        (
            "frontend_neutral".to_string(),
            ay_shared_engine_all_lane_metadata()
                .iter()
                .all(|metadata| metadata.frontend_neutral)
                .to_string(),
        ),
        (
            "compatible_frontends".to_string(),
            AY_SHARED_ENGINE_COMPATIBLE_FRONTEND_FAMILIES
                .iter()
                .map(|frontend| frontend.name())
                .collect::<Vec<_>>()
                .join(","),
        ),
        (
            "compatible_frontend_codes".to_string(),
            AY_SHARED_ENGINE_COMPATIBLE_FRONTEND_FAMILIES
                .iter()
                .map(|frontend| frontend.code())
                .collect::<Vec<_>>()
                .join(","),
        ),
        (
            "compatible_frontend_families".to_string(),
            AY_SHARED_ENGINE_ADOPTION_FRONTEND_FAMILIES.join(","),
        ),
        (
            "known_frontend_families".to_string(),
            AY_SHARED_ENGINE_FRONTEND_FAMILIES
                .iter()
                .map(|frontend| frontend.adoption_code())
                .collect::<Vec<_>>()
                .join(","),
        ),
        (
            "frontend_family_blockers".to_string(),
            AY_SHARED_PROOF_LANE_FRONTEND_FAMILY_BLOCKERS.to_string(),
        ),
    ];

    for metadata in ay_shared_engine_all_lane_metadata() {
        let prefix = metadata.lane.code();
        rows.push((format!("{prefix}_backend"), metadata.backend.to_string()));
        rows.push((
            format!("{prefix}_compatible_frontends"),
            metadata.compatible_frontend_names().join(","),
        ));
        rows.push((
            format!("{prefix}_compatible_frontend_families"),
            metadata.compatible_frontend_family_codes().join(","),
        ));
        rows.push((
            format!("{prefix}_generic_prerequisites"),
            metadata.generic_prerequisites.join(","),
        ));
        rows.push((
            format!("{prefix}_proof_obligations"),
            metadata.proof_obligations.join(","),
        ));
    }

    rows
}

/// Render all shared engine lanes as one frontend-neutral evidence line.
#[must_use]
pub fn render_ay_shared_engine_evidence(scope: &str) -> String {
    let rows = ay_shared_engine_evidence_key_value_rows();
    format!(
        "{} ay_shared_engine_metadata {}",
        scope,
        key_value_pairs_to_text(&rows)
    )
}

fn key_value_pairs_to_text(rows: &[(String, String)]) -> String {
    rows.iter()
        .map(|(key, value)| format!("{key}={value}"))
        .collect::<Vec<_>>()
        .join(" ")
}

fn evidence_value(value: &str) -> String {
    if value.is_empty() {
        "none".to_string()
    } else {
        value.replace(char::is_whitespace, "_")
    }
}

fn evidence_optional(value: Option<&str>) -> String {
    value
        .map(evidence_value)
        .unwrap_or_else(|| "none".to_string())
}

fn frontend_family_codes_to_evidence(values: Vec<&'static str>) -> String {
    if values.is_empty() {
        "none".to_string()
    } else {
        values.join(",")
    }
}

fn non_empty_string(value: String) -> Option<String> {
    if value.is_empty() {
        None
    } else {
        Some(value)
    }
}

#[cfg(test)]
mod tests {
    use super::{
        ay_shared_engine_all_lane_metadata, ay_shared_engine_evidence_key_value_rows,
        ay_shared_engine_lane_metadata, render_ay_shared_engine_evidence,
        render_ay_shared_engine_lane_evidence, AYFrontendFamily, AYProofValidationReceipt,
        AYProofValidationReceiptKind, AYProofValidationReceiptStatus, AYSharedEngineLane,
        AYSharedProofLaneDescriptor, AY_ANALYTICAL_PROOF_SHARED_ENGINE_COMPONENT,
        AY_ANALYTICAL_SOLVE_CONTRACT_SCHEMA, AY_ANALYTICAL_SOLVE_CONTRACT_SCHEMA_VERSION,
        AY_SHARED_ENGINE_FRONTEND_FAMILIES, AY_SHARED_ENGINE_LANES,
        AY_SHARED_ENGINE_METADATA_SCHEMA, AY_SHARED_ENGINE_METADATA_SCHEMA_VERSION,
        AY_SHARED_PROOF_LANE_DESCRIPTOR_SCHEMA, AY_SHARED_PROOF_LANE_DESCRIPTOR_SCHEMA_VERSION,
        AY_SHARED_PROOF_VALIDATION_RECEIPT_SCHEMA,
    };

    fn evidence_field<'a>(row: &'a str, key: &str) -> Option<&'a str> {
        let prefix = format!("{key}=");
        row.split_whitespace()
            .find_map(|field| field.strip_prefix(&prefix))
    }

    fn family_tokens<'a>(row: &'a str, key: &str) -> Vec<&'a str> {
        match evidence_field(row, key) {
            Some("none") | None => Vec::new(),
            Some(value) => value.split(',').collect(),
        }
    }

    #[test]
    fn shared_engine_metadata_lists_frontend_neutral_lanes() {
        let rows = ay_shared_engine_evidence_key_value_rows();

        assert!(rows.contains(&(
            "schema".to_string(),
            AY_SHARED_ENGINE_METADATA_SCHEMA.to_string()
        )));
        assert!(rows.contains(&(
            "schema_version".to_string(),
            AY_SHARED_ENGINE_METADATA_SCHEMA_VERSION.to_string()
        )));
        assert!(rows.iter().any(|(key, value)| {
            key == "compatible_frontend_codes"
                && value == "tla,quint,mcc_petri,aiger,btor2,ay_only,vmt_replay,witness_replay"
        }));
        assert!(rows.iter().any(|(key, value)| {
            key == "compatible_frontend_families"
                && value == "tla_plus,quint,mcc_petri,aiger,btor2,ay_analytical,vmt_transition_system,witness_replay"
        }));
        assert!(rows.iter().any(|(key, value)| {
            key == "known_frontend_families"
                && value == "tla_plus,quint,mcc_petri,aiger,btor2,ay_analytical,vmt_transition_system,witness_replay,future_importer"
        }));
        assert!(rows.iter().any(|(key, value)| {
            key == "frontend_family_blockers"
                && value == "future_importer:awaiting_registered_importer_frontend"
        }));
        assert_eq!(AY_SHARED_ENGINE_LANES.len(), 5);
        assert_eq!(AY_SHARED_ENGINE_FRONTEND_FAMILIES.len(), 9);
        assert!(ay_shared_engine_all_lane_metadata()
            .iter()
            .all(|metadata| metadata.frontend_neutral));
    }

    #[test]
    fn shared_engine_lane_metadata_supports_required_frontends() {
        let required_frontends = [
            AYFrontendFamily::Tla,
            AYFrontendFamily::MccPetri,
            AYFrontendFamily::Aiger,
            AYFrontendFamily::Btor2,
            AYFrontendFamily::VmtReplay,
            AYFrontendFamily::WitnessReplay,
        ];

        for lane in AY_SHARED_ENGINE_LANES {
            let metadata = ay_shared_engine_lane_metadata(lane);
            for frontend in required_frontends {
                assert!(
                    metadata.supports_frontend(frontend),
                    "{lane:?} should support {frontend:?}"
                );
            }
            assert!(
                !metadata.supports_frontend(AYFrontendFamily::FutureImporter),
                "{lane:?} should keep future importers behind the registration blocker"
            );
            assert!(
                !metadata.generic_prerequisites.is_empty(),
                "{lane:?} should expose generic prerequisites"
            );
            assert!(
                !metadata.proof_obligations.is_empty(),
                "{lane:?} should expose generic lane obligations"
            );
        }
    }

    #[test]
    fn proof_lane_descriptor_carries_identity_fingerprints_and_compatible_frontends() {
        let descriptor = AYSharedProofLaneDescriptor::new(
            AYSharedEngineLane::Pdr,
            AYFrontendFamily::Aiger,
            "prepared program:aiger:counter",
            "proof obligation:aiger:safety",
        )
        .with_proof_fingerprint_identity("proof:fingerprint:aiger:safety")
        .with_model_fingerprint_identity("model:fingerprint:aiger:safety")
        .with_certificate_fingerprint_identity("certificate:fingerprint:aiger:safety")
        .with_witness_fingerprint_identity("witness:fingerprint:aiger:safety");

        assert_eq!(descriptor.schema, AY_SHARED_PROOF_LANE_DESCRIPTOR_SCHEMA);
        assert_eq!(
            descriptor.schema_version,
            AY_SHARED_PROOF_LANE_DESCRIPTOR_SCHEMA_VERSION
        );
        assert_eq!(descriptor.lane, AYSharedEngineLane::Pdr);
        assert_eq!(descriptor.source_frontend_family, AYFrontendFamily::Aiger);
        assert_eq!(
            descriptor.prepared_program_identity,
            "prepared program:aiger:counter"
        );
        assert_eq!(
            descriptor.proof_obligation_identity,
            "proof obligation:aiger:safety"
        );
        assert_eq!(
            descriptor.fingerprint_identities(),
            vec![
                "proof:fingerprint:aiger:safety",
                "model:fingerprint:aiger:safety",
                "certificate:fingerprint:aiger:safety",
                "witness:fingerprint:aiger:safety"
            ]
        );
        for frontend in [
            AYFrontendFamily::Tla,
            AYFrontendFamily::MccPetri,
            AYFrontendFamily::Aiger,
            AYFrontendFamily::Btor2,
            AYFrontendFamily::VmtReplay,
        ] {
            assert!(descriptor.supports_frontend(frontend));
        }
    }

    #[test]
    fn analytical_solve_contract_evidence_is_frontend_neutral_and_certificate_aware() {
        let descriptor = AYSharedProofLaneDescriptor::new(
            AYSharedEngineLane::Pdr,
            AYFrontendFamily::Btor2,
            "prepared_program:shared:word_safety",
            "proof_obligation:shared:inductive_safety",
        )
        .with_proof_fingerprint_identity("proof:fingerprint:shared:inductive_safety")
        .with_model_fingerprint_identity("model:fingerprint:shared:counterexample")
        .with_certificate_fingerprint_identity("certificate:fingerprint:shared:inductive_safety");

        let row = descriptor.render_analytical_solve_contract_evidence("AY");

        assert!(row.starts_with("AY ay_analytical_solve_contract "));
        assert!(row.contains(&format!("schema={}", AY_ANALYTICAL_SOLVE_CONTRACT_SCHEMA)));
        assert!(row.contains(&format!(
            "schema_version={}",
            AY_ANALYTICAL_SOLVE_CONTRACT_SCHEMA_VERSION
        )));
        assert!(row.contains(
            "analytical_solve_contract_identity=analytical_ay_proof:pdr:prepared_program:shared:word_safety:proof_obligation:shared:inductive_safety"
        ));
        assert!(row.contains(
            "analytical_big_win_detection_rule=shared_analytical_solve_replaces_frontend_specific_search_or_enumeration"
        ));
        assert!(row.contains(
            "analytical_big_win_detection_basis=prepared_descriptor_fingerprints_validation_receipts"
        ));
        assert!(row.contains(
            "analytical_big_win_detection_identity=shared_analytical_solve_replaces_frontend_specific_search_or_enumeration:pdr:proof_obligation:shared:inductive_safety"
        ));
        assert!(row.contains("origin_frontend=btor2"));
        assert!(
            row.contains("proof_fingerprint_identity=proof:fingerprint:shared:inductive_safety")
        );
        assert!(row.contains("model_fingerprint_identity=model:fingerprint:shared:counterexample"));
        assert!(row.contains(
            "certificate_fingerprint_identity=certificate:fingerprint:shared:inductive_safety"
        ));
        assert!(row.contains(
            "shared_fingerprint_identities=proof:fingerprint:shared:inductive_safety,model:fingerprint:shared:counterexample,certificate:fingerprint:shared:inductive_safety"
        ));
        assert!(row.contains("validation_receipt_required=true"));
        assert!(row.contains("proof_and_witness_receipts_required_for_big_win=true"));
        assert!(row.contains("proof_receipt_required=true"));
        assert!(row.contains("model_receipt_required=true"));
        assert!(row.contains("witness_receipt_required=false"));
        assert!(row.contains(
            "validation_receipt_requirement=validator_backed_receipt_for_proof_obligation_fingerprint"
        ));
        assert!(row.contains(
            "compatible_frontend_families=tla_plus,quint,mcc_petri,aiger,btor2,ay_analytical,vmt_transition_system,witness_replay"
        ));
        assert!(row.contains(
            "analytical_big_win_beneficiary_frontend_families=tla_plus,quint,mcc_petri,aiger,btor2,ay_analytical,vmt_transition_system,witness_replay"
        ));
        assert!(row.contains(
            "active_frontend_families=tla_plus,quint,mcc_petri,aiger,btor2,ay_analytical,vmt_transition_system,witness_replay"
        ));
        assert!(row.contains(
            "default_frontend_families=tla_plus,quint,mcc_petri,aiger,btor2,ay_analytical,vmt_transition_system,witness_replay"
        ));
        assert!(row.contains("remaining_frontend_families=none"));
        assert!(row.contains(
            "default_frontend_policy=all_active_compatible_frontend_families_default_after_receipt_validation"
        ));
        assert!(row.contains(
            "explicit_search_replacement_policy=replace_explicit_search_only_after_validator_backed_proof_and_witness_receipts"
        ));
        assert!(row.contains(
            "fail_closed_blockers=missing_proof_receipt,missing_witness_receipt,future_importer_missing_registered_payload"
        ));
        assert!(row.contains("blocker_status=tracked-blockers"));
        assert!(row.contains(
            "frontend_family_blockers=future_importer:awaiting_registered_importer_frontend"
        ));
        assert!(row.contains("shared_owner=shared_high_performance_engine"));
        assert!(
            row.contains("publication_blocker=missing_validator_backed_proof_and_witness_receipts")
        );
    }

    #[test]
    fn analytical_big_win_evidence_keeps_future_importers_fail_closed_until_registered() {
        let descriptor = AYSharedProofLaneDescriptor::new(
            AYSharedEngineLane::Pdr,
            AYFrontendFamily::FutureImporter,
            "prepared_program:shared:future_transition_system",
            "proof_obligation:shared:safety",
        )
        .with_proof_fingerprint_identity("proof:fingerprint:shared:safety")
        .with_witness_fingerprint_identity("witness:fingerprint:shared:safety");

        let proof_receipt = AYProofValidationReceipt::validator_backed(
            "receipt:proof:shared:safety",
            AYProofValidationReceiptKind::ProofTranscript,
            "proof_obligation:shared:safety",
            "proof:fingerprint:shared:safety",
        );
        let witness_receipt = AYProofValidationReceipt::validator_backed(
            "receipt:witness:shared:safety",
            AYProofValidationReceiptKind::Witness,
            "proof_obligation:shared:safety",
            "witness:fingerprint:shared:safety",
        );

        assert!(!descriptor.supports_frontend(AYFrontendFamily::FutureImporter));
        assert!(descriptor
            .render_analytical_big_win_evidence_with_receipts(
                "AY",
                Some(&proof_receipt),
                Some(&witness_receipt),
            )
            .is_none());
        assert!(descriptor
            .render_publication_evidence("AY", Some(&proof_receipt))
            .is_none());
        assert!(descriptor
            .render_hardware_adoption_evidence_with_receipts(
                "AY",
                "future_ay_proof_lane_adoption",
                "future.register_vector.v1",
                "future_importer",
                "mcc_petri",
                Some(&proof_receipt),
                Some(&witness_receipt),
            )
            .is_none());
    }

    #[test]
    fn analytical_big_win_evidence_requires_proof_and_witness_receipts_for_compatible_frontends() {
        let descriptor = AYSharedProofLaneDescriptor::new(
            AYSharedEngineLane::Pdr,
            AYFrontendFamily::Btor2,
            "prepared_program:shared:register_transition_system",
            "proof_obligation:shared:safety",
        )
        .with_proof_fingerprint_identity("proof:fingerprint:shared:safety")
        .with_witness_fingerprint_identity("witness:fingerprint:shared:safety");

        let proof_receipt = AYProofValidationReceipt::validator_backed(
            "receipt:proof:shared:safety",
            AYProofValidationReceiptKind::ProofTranscript,
            "proof_obligation:shared:safety",
            "proof:fingerprint:shared:safety",
        );
        let witness_receipt = AYProofValidationReceipt::validator_backed(
            "receipt:witness:shared:safety",
            AYProofValidationReceiptKind::Witness,
            "proof_obligation:shared:safety",
            "witness:fingerprint:shared:safety",
        );

        assert!(descriptor.supports_frontend(AYFrontendFamily::Btor2));
        assert!(descriptor
            .render_analytical_big_win_evidence_with_receipts("AY", Some(&proof_receipt), None)
            .is_none());
        assert!(descriptor
            .render_analytical_big_win_evidence_with_receipts("AY", None, Some(&witness_receipt))
            .is_none());
        let model_receipt_for_proof_fingerprint = AYProofValidationReceipt::validator_backed(
            "receipt:model:shared:safety",
            AYProofValidationReceiptKind::Model,
            "proof_obligation:shared:safety",
            "proof:fingerprint:shared:safety",
        );
        assert!(descriptor
            .render_analytical_big_win_evidence_with_receipts(
                "AY",
                Some(&model_receipt_for_proof_fingerprint),
                Some(&witness_receipt),
            )
            .is_none());

        let row = descriptor
            .render_analytical_big_win_evidence_with_receipts(
                "AY",
                Some(&proof_receipt),
                Some(&witness_receipt),
            )
            .expect("proof and witness receipts should publish analytical big-win evidence");
        assert!(row.starts_with("AY ay_analytical_big_win_evidence "));
        assert!(row.contains("origin_frontend=btor2"));
        assert!(row.contains("first_beneficiary=btor2"));
        assert!(row.contains("second_beneficiary=aiger"));
        assert!(row.contains(
            "beneficiary_frontend_families=tla_plus,quint,mcc_petri,aiger,btor2,ay_analytical,vmt_transition_system,witness_replay"
        ));
        assert!(row.contains("required_receipt_kinds=proof_transcript,witness"));
        assert!(row.contains("proof_and_witness_receipts_required_for_big_win=true"));
        assert!(row.contains("proof_receipt_required=true"));
        assert!(row.contains("witness_receipt_required=true"));
        assert!(row.contains("proof_receipt_identity=receipt:proof:shared:safety"));
        assert!(row.contains("proof_receipt_schema=tla-ay.shared-proof-validation-receipt/v1"));
        assert!(row.contains("proof_receipt_validation_kind=proof_transcript"));
        assert!(row.contains("witness_receipt_identity=receipt:witness:shared:safety"));
        assert!(row.contains("witness_receipt_schema=tla-ay.shared-proof-validation-receipt/v1"));
        assert!(row.contains("witness_receipt_validation_kind=witness"));
        assert!(row.contains(
            "shared_fingerprint_identities=proof:fingerprint:shared:safety,witness:fingerprint:shared:safety"
        ));
        assert!(row.contains(
            "compatible_frontend_families=tla_plus,quint,mcc_petri,aiger,btor2,ay_analytical,vmt_transition_system,witness_replay"
        ));
        assert!(row.contains(
            "active_frontend_families=tla_plus,quint,mcc_petri,aiger,btor2,ay_analytical,vmt_transition_system,witness_replay"
        ));
        assert!(row.contains(
            "default_frontend_families=tla_plus,quint,mcc_petri,aiger,btor2,ay_analytical,vmt_transition_system,witness_replay"
        ));
        assert!(row.contains("remaining_frontend_families=none"));
        assert!(row.contains(
            "default_compatible_frontend_families=tla_plus,quint,mcc_petri,aiger,btor2,ay_analytical,vmt_transition_system,witness_replay"
        ));
        assert!(row.contains("remaining_compatible_frontend_families=none"));
        assert!(row.contains(
            "default_frontend_policy=all_active_compatible_frontend_families_default_after_receipt_validation"
        ));
        assert!(row.contains(
            "explicit_search_replacement_policy=replace_explicit_search_only_after_validator_backed_proof_and_witness_receipts"
        ));
        assert!(row.contains("explicit_search_replacement_admitted=true"));
        assert!(row.contains("publication_blocker=none"));
        assert!(row.contains(
            "analytical_big_win_detection_rule=shared_analytical_solve_replaces_frontend_specific_search_or_enumeration"
        ));
        assert!(row.contains(
            "analytical_big_win_detection_basis=prepared_descriptor_fingerprints_validation_receipts"
        ));
        assert!(row.contains("blocker_status=tracked-blockers"));
        assert!(row.contains(
            "frontend_family_blockers=future_importer:awaiting_registered_importer_frontend"
        ));
    }

    #[test]
    fn analytical_big_win_defaults_all_active_frontends_but_not_future_importers() {
        let descriptor = AYSharedProofLaneDescriptor::new(
            AYSharedEngineLane::Pdr,
            AYFrontendFamily::MccPetri,
            "prepared_program:portable:default_consumers",
            "proof_obligation:portable:default_consumers",
        )
        .with_proof_fingerprint_identity("proof:fingerprint:portable:default_consumers")
        .with_witness_fingerprint_identity("witness:fingerprint:portable:default_consumers");
        let proof_receipt = AYProofValidationReceipt::validator_backed(
            "receipt:proof:portable:default_consumers",
            AYProofValidationReceiptKind::ProofTranscript,
            "proof_obligation:portable:default_consumers",
            "proof:fingerprint:portable:default_consumers",
        );
        let witness_receipt = AYProofValidationReceipt::validator_backed(
            "receipt:witness:portable:default_consumers",
            AYProofValidationReceiptKind::Witness,
            "proof_obligation:portable:default_consumers",
            "witness:fingerprint:portable:default_consumers",
        );

        let row = descriptor
            .render_analytical_big_win_evidence_with_receipts(
                "AY",
                Some(&proof_receipt),
                Some(&witness_receipt),
            )
            .expect("proof and witness receipts should publish active defaults");
        let current_families =
            "tla_plus,quint,mcc_petri,aiger,btor2,ay_analytical,vmt_transition_system,witness_replay";

        assert_eq!(
            evidence_field(&row, "active_frontend_families"),
            Some(current_families)
        );
        assert_eq!(
            evidence_field(&row, "default_frontend_families"),
            Some(current_families)
        );
        assert_eq!(
            evidence_field(&row, "default_compatible_frontend_families"),
            Some(current_families)
        );
        assert_eq!(
            evidence_field(&row, "remaining_frontend_families"),
            Some("none")
        );
        assert_eq!(
            evidence_field(&row, "remaining_compatible_frontend_families"),
            Some("none")
        );
        assert_eq!(
            evidence_field(&row, "blocked_frontend_families"),
            Some("future_importer")
        );
        assert!(
            !family_tokens(&row, "default_frontend_families").contains(&"future_importer"),
            "future importers need a registered payload before default consumption"
        );
        assert_eq!(evidence_field(&row, "publication_blocker"), Some("none"));
        assert_eq!(
            evidence_field(&row, "proof_and_witness_receipts_required_for_big_win"),
            Some("true")
        );
    }

    #[test]
    fn analytical_big_win_evidence_requires_declared_model_receipt_and_kind() {
        let descriptor = AYSharedProofLaneDescriptor::new(
            AYSharedEngineLane::AllSatEnumeration,
            AYFrontendFamily::AYOnly,
            "prepared_program:ay_analytical:model_search",
            "proof_obligation:ay_analytical:model_projection",
        )
        .with_proof_fingerprint_identity("proof:fingerprint:ay_analytical:model_search")
        .with_model_fingerprint_identity("model:fingerprint:ay_analytical:model_projection")
        .with_witness_fingerprint_identity("witness:fingerprint:ay_analytical:model_projection");

        let proof_receipt = AYProofValidationReceipt::validator_backed(
            "receipt:proof:ay_analytical:model_search",
            AYProofValidationReceiptKind::ProofTranscript,
            "proof_obligation:ay_analytical:model_projection",
            "proof:fingerprint:ay_analytical:model_search",
        );
        let model_receipt = AYProofValidationReceipt::validator_backed(
            "receipt:model:ay_analytical:model_projection",
            AYProofValidationReceiptKind::Model,
            "proof_obligation:ay_analytical:model_projection",
            "model:fingerprint:ay_analytical:model_projection",
        );
        let mislabeled_model_receipt = AYProofValidationReceipt::validator_backed(
            "receipt:witness:ay_analytical:model_projection",
            AYProofValidationReceiptKind::Witness,
            "proof_obligation:ay_analytical:model_projection",
            "model:fingerprint:ay_analytical:model_projection",
        );
        let witness_receipt = AYProofValidationReceipt::validator_backed(
            "receipt:witness:ay_analytical:model_projection",
            AYProofValidationReceiptKind::Witness,
            "proof_obligation:ay_analytical:model_projection",
            "witness:fingerprint:ay_analytical:model_projection",
        );

        assert!(descriptor
            .render_analytical_big_win_evidence_with_artifact_receipts(
                "AY",
                Some(&proof_receipt),
                None,
                Some(&witness_receipt),
            )
            .is_none());
        assert!(descriptor
            .render_analytical_big_win_evidence_with_artifact_receipts(
                "AY",
                Some(&proof_receipt),
                Some(&mislabeled_model_receipt),
                Some(&witness_receipt),
            )
            .is_none());
        assert!(descriptor
            .render_analytical_big_win_evidence_with_receipts(
                "AY",
                Some(&proof_receipt),
                Some(&witness_receipt),
            )
            .is_none());

        let row = descriptor
            .render_analytical_big_win_evidence_with_artifact_receipts(
                "AY",
                Some(&proof_receipt),
                Some(&model_receipt),
                Some(&witness_receipt),
            )
            .expect("proof, model, and witness receipts should publish ay analytical wins");

        assert!(row.contains("origin_frontend=ay_analytical"));
        assert!(row.contains("first_beneficiary=ay_analytical"));
        assert!(row.contains("second_beneficiary=mcc_petri"));
        assert!(row.contains("required_receipt_kinds=proof_transcript,model,witness"));
        assert!(row.contains("proof_receipt_required=true"));
        assert!(row.contains("model_receipt_required=true"));
        assert!(row.contains("witness_receipt_required=true"));
        assert!(row.contains("model_receipt_identity=receipt:model:ay_analytical:model_projection"));
        assert!(row.contains("model_receipt_schema=tla-ay.shared-proof-validation-receipt/v1"));
        assert!(row.contains("model_receipt_validation_kind=model"));
        assert!(row.contains("model_receipt_status=validator_backed"));
        assert!(row.contains(
            "model_receipt_validated_fingerprint_identity=model:fingerprint:ay_analytical:model_projection"
        ));
        assert!(row.contains(
            "shared_fingerprint_identities=proof:fingerprint:ay_analytical:model_search,model:fingerprint:ay_analytical:model_projection,witness:fingerprint:ay_analytical:model_projection"
        ));
        assert!(row.contains(
            "compatible_frontend_families=tla_plus,quint,mcc_petri,aiger,btor2,ay_analytical,vmt_transition_system,witness_replay"
        ));
    }

    #[test]
    fn analytical_big_win_evidence_requires_declared_certificate_receipt_and_kind() {
        let descriptor = AYSharedProofLaneDescriptor::new(
            AYSharedEngineLane::Chc,
            AYFrontendFamily::WitnessReplay,
            "prepared_program:witness_replay:certificate_lane",
            "proof_obligation:witness_replay:certificate_check",
        )
        .with_proof_fingerprint_identity("proof:fingerprint:witness_replay:certificate_check")
        .with_certificate_fingerprint_identity(
            "certificate:fingerprint:witness_replay:certificate_check",
        )
        .with_witness_fingerprint_identity("witness:fingerprint:witness_replay:certificate_check");

        let proof_receipt = AYProofValidationReceipt::validator_backed(
            "receipt:proof:witness_replay:certificate_check",
            AYProofValidationReceiptKind::ProofTranscript,
            "proof_obligation:witness_replay:certificate_check",
            "proof:fingerprint:witness_replay:certificate_check",
        );
        let certificate_receipt = AYProofValidationReceipt::validator_backed(
            "receipt:certificate:witness_replay:certificate_check",
            AYProofValidationReceiptKind::Certificate,
            "proof_obligation:witness_replay:certificate_check",
            "certificate:fingerprint:witness_replay:certificate_check",
        );
        let mislabeled_certificate_receipt = AYProofValidationReceipt::validator_backed(
            "receipt:model:witness_replay:certificate_check",
            AYProofValidationReceiptKind::Model,
            "proof_obligation:witness_replay:certificate_check",
            "certificate:fingerprint:witness_replay:certificate_check",
        );
        let witness_receipt = AYProofValidationReceipt::validator_backed(
            "receipt:witness:witness_replay:certificate_check",
            AYProofValidationReceiptKind::Witness,
            "proof_obligation:witness_replay:certificate_check",
            "witness:fingerprint:witness_replay:certificate_check",
        );

        assert!(descriptor
            .render_analytical_big_win_evidence_with_receipts(
                "AY",
                Some(&proof_receipt),
                Some(&witness_receipt),
            )
            .is_none());
        assert!(descriptor
            .render_analytical_big_win_evidence_with_validation_receipts(
                "AY",
                Some(&proof_receipt),
                None,
                Some(&mislabeled_certificate_receipt),
                Some(&witness_receipt),
            )
            .is_none());

        let row = descriptor
            .render_analytical_big_win_evidence_with_validation_receipts(
                "AY",
                Some(&proof_receipt),
                None,
                Some(&certificate_receipt),
                Some(&witness_receipt),
            )
            .expect("proof, certificate, and witness receipts should publish replay wins");

        assert!(row.contains("origin_frontend=witness_replay"));
        assert!(row.contains("first_beneficiary=witness_replay"));
        assert!(row.contains("second_beneficiary=ay_analytical"));
        assert!(row.contains("required_receipt_kinds=proof_transcript,certificate,witness"));
        assert!(row.contains("model_receipt_required=false"));
        assert!(row.contains("certificate_receipt_required=true"));
        assert!(row.contains("witness_receipt_required=true"));
        assert!(row.contains(
            "certificate_receipt_identity=receipt:certificate:witness_replay:certificate_check"
        ));
        assert!(
            row.contains("certificate_receipt_schema=tla-ay.shared-proof-validation-receipt/v1")
        );
        assert!(row.contains("certificate_receipt_validation_kind=certificate"));
        assert!(row.contains("certificate_receipt_status=validator_backed"));
        assert!(row.contains(
            "certificate_receipt_validated_fingerprint_identity=certificate:fingerprint:witness_replay:certificate_check"
        ));
        assert!(row.contains(
            "shared_fingerprint_identities=proof:fingerprint:witness_replay:certificate_check,certificate:fingerprint:witness_replay:certificate_check,witness:fingerprint:witness_replay:certificate_check"
        ));
    }

    #[test]
    fn analytical_big_win_evidence_uses_same_contract_for_btor2_and_tla_style_sources() {
        let tla_descriptor = AYSharedProofLaneDescriptor::new(
            AYSharedEngineLane::KInduction,
            AYFrontendFamily::Tla,
            "prepared_program:shared:counter",
            "proof_obligation:shared:invariant",
        )
        .with_proof_fingerprint_identity("proof:fingerprint:shared:invariant")
        .with_witness_fingerprint_identity("witness:fingerprint:shared:invariant");
        let btor2_descriptor = AYSharedProofLaneDescriptor::new(
            AYSharedEngineLane::KInduction,
            AYFrontendFamily::Btor2,
            "prepared_program:shared:counter",
            "proof_obligation:shared:invariant",
        )
        .with_proof_fingerprint_identity("proof:fingerprint:shared:invariant")
        .with_witness_fingerprint_identity("witness:fingerprint:shared:invariant");

        assert_eq!(
            tla_descriptor.analytical_solve_contract_identity(),
            btor2_descriptor.analytical_solve_contract_identity()
        );
        assert_eq!(
            tla_descriptor.analytical_big_win_detection_identity(),
            btor2_descriptor.analytical_big_win_detection_identity()
        );
        assert_eq!(
            tla_descriptor.compatible_frontend_family_codes(),
            btor2_descriptor.compatible_frontend_family_codes()
        );
    }

    #[test]
    fn analytical_big_win_evidence_is_publishable_for_replay_petri_tla_and_hardware_sources() {
        let proof_receipt = AYProofValidationReceipt::validator_backed(
            "receipt:proof:portable:safety",
            AYProofValidationReceiptKind::ProofTranscript,
            "proof_obligation:portable:safety",
            "proof:fingerprint:portable:safety",
        );
        let witness_receipt = AYProofValidationReceipt::validator_backed(
            "receipt:witness:portable:safety",
            AYProofValidationReceiptKind::Witness,
            "proof_obligation:portable:safety",
            "witness:fingerprint:portable:safety",
        );

        for (frontend, origin, second_beneficiary) in [
            (AYFrontendFamily::Tla, "tla_plus", "mcc_petri"),
            (AYFrontendFamily::MccPetri, "mcc_petri", "ay_analytical"),
            (
                AYFrontendFamily::VmtReplay,
                "vmt_transition_system",
                "ay_analytical",
            ),
            (AYFrontendFamily::Aiger, "aiger", "btor2"),
        ] {
            let descriptor = AYSharedProofLaneDescriptor::new(
                AYSharedEngineLane::Bmc,
                frontend,
                "prepared_program:portable:transition_system",
                "proof_obligation:portable:safety",
            )
            .with_proof_fingerprint_identity("proof:fingerprint:portable:safety")
            .with_witness_fingerprint_identity("witness:fingerprint:portable:safety");

            assert!(descriptor.supports_frontend(frontend));
            assert!(!descriptor.supports_frontend(AYFrontendFamily::FutureImporter));

            let row = descriptor
                .render_analytical_big_win_evidence_with_receipts(
                    "AY",
                    Some(&proof_receipt),
                    Some(&witness_receipt),
                )
                .expect("portable proof and witness receipts should publish a shared big-win row");

            assert!(row.contains(&format!("origin_frontend={origin}")));
            assert!(row.contains(&format!("first_beneficiary={origin}")));
            assert!(row.contains(&format!("second_beneficiary={second_beneficiary}")));
            assert!(row.contains(
                "analytical_solve_contract_identity=analytical_ay_proof:bmc:prepared_program:portable:transition_system:proof_obligation:portable:safety"
            ));
            assert!(row.contains(
                "beneficiary_frontend_families=tla_plus,quint,mcc_petri,aiger,btor2,ay_analytical,vmt_transition_system,witness_replay"
            ));
            assert!(row.contains("required_receipt_kinds=proof_transcript,witness"));
            assert!(row.contains("proof_receipt_validation_kind=proof_transcript"));
            assert!(row.contains("witness_receipt_validation_kind=witness"));
        }
    }

    #[test]
    fn analytical_big_win_evidence_is_receipt_backed_for_all_current_families() {
        let proof_receipt = AYProofValidationReceipt::validator_backed(
            "receipt:proof:portable:all_families",
            AYProofValidationReceiptKind::ProofTranscript,
            "proof_obligation:portable:all_families",
            "proof:fingerprint:portable:all_families",
        );
        let witness_receipt = AYProofValidationReceipt::validator_backed(
            "receipt:witness:portable:all_families",
            AYProofValidationReceiptKind::Witness,
            "proof_obligation:portable:all_families",
            "witness:fingerprint:portable:all_families",
        );

        for (frontend, canonical_origin, legacy_alias) in [
            (AYFrontendFamily::Tla, "tla_plus", Some("tla")),
            (AYFrontendFamily::Quint, "quint", None),
            (AYFrontendFamily::MccPetri, "mcc_petri", None),
            (AYFrontendFamily::Aiger, "aiger", None),
            (AYFrontendFamily::Btor2, "btor2", None),
            (
                AYFrontendFamily::VmtReplay,
                "vmt_transition_system",
                Some("vmt_replay"),
            ),
            (AYFrontendFamily::WitnessReplay, "witness_replay", None),
            (AYFrontendFamily::AYOnly, "ay_analytical", Some("ay_only")),
        ] {
            let descriptor = AYSharedProofLaneDescriptor::new(
                AYSharedEngineLane::KInduction,
                frontend,
                "prepared_program:portable:all_families",
                "proof_obligation:portable:all_families",
            )
            .with_proof_fingerprint_identity("proof:fingerprint:portable:all_families")
            .with_witness_fingerprint_identity("witness:fingerprint:portable:all_families");

            assert!(
                descriptor.can_publish_analytical_big_win_with_validation_receipts(
                    Some(&proof_receipt),
                    None,
                    None,
                    Some(&witness_receipt),
                )
            );

            let row = descriptor
                .render_analytical_big_win_evidence_with_validation_receipts(
                    "AY",
                    Some(&proof_receipt),
                    None,
                    None,
                    Some(&witness_receipt),
                )
                .expect("validator-backed proof and witness receipts should publish");

            assert_eq!(
                evidence_field(&row, "origin_frontend"),
                Some(canonical_origin)
            );
            assert_eq!(
                evidence_field(&row, "validation_receipt_backed"),
                Some("true")
            );
            assert_eq!(evidence_field(&row, "publishable_shared_win"), Some("true"));
            assert_eq!(
                evidence_field(&row, "blocked_frontend_families"),
                Some("future_importer")
            );
            assert_eq!(
                evidence_field(&row, "frontend_family_blockers"),
                Some("future_importer:awaiting_registered_importer_frontend")
            );
            assert_eq!(
                evidence_field(&row, "blocker_status"),
                Some("tracked-blockers")
            );
            assert!(family_tokens(&row, "compatible_frontend_families").contains(&canonical_origin));
            assert!(
                family_tokens(&row, "beneficiary_frontend_families").contains(&canonical_origin)
            );
            assert!(
                !family_tokens(&row, "compatible_frontend_families").contains(&"future_importer")
            );
            assert!(!family_tokens(&row, "default_frontend_families").contains(&"future_importer"));
            assert!(
                !family_tokens(&row, "remaining_frontend_families").contains(&"future_importer")
            );
            assert_eq!(
                descriptor.required_analytical_big_win_receipt_kind_codes(),
                vec!["proof_transcript", "witness"]
            );
            if let Some(legacy_alias) = legacy_alias {
                assert_ne!(
                    evidence_field(&row, "origin_frontend"),
                    Some(legacy_alias),
                    "legacy frontend aliases must normalize before publication evidence"
                );
            }
        }
    }

    #[test]
    fn analytical_big_win_evidence_rejects_artifact_only_or_missing_receipts() {
        let descriptor = AYSharedProofLaneDescriptor::new(
            AYSharedEngineLane::Pdr,
            AYFrontendFamily::Btor2,
            "prepared_program:portable:receipt_gate",
            "proof_obligation:portable:receipt_gate",
        )
        .with_proof_fingerprint_identity("proof:fingerprint:portable:receipt_gate")
        .with_model_fingerprint_identity("model:fingerprint:portable:receipt_gate")
        .with_certificate_fingerprint_identity("certificate:fingerprint:portable:receipt_gate")
        .with_witness_fingerprint_identity("witness:fingerprint:portable:receipt_gate");

        let proof_receipt = AYProofValidationReceipt::validator_backed(
            "receipt:proof:portable:receipt_gate",
            AYProofValidationReceiptKind::ProofTranscript,
            "proof_obligation:portable:receipt_gate",
            "proof:fingerprint:portable:receipt_gate",
        );
        let model_receipt = AYProofValidationReceipt::validator_backed(
            "receipt:model:portable:receipt_gate",
            AYProofValidationReceiptKind::Model,
            "proof_obligation:portable:receipt_gate",
            "model:fingerprint:portable:receipt_gate",
        );
        let certificate_receipt = AYProofValidationReceipt::validator_backed(
            "receipt:certificate:portable:receipt_gate",
            AYProofValidationReceiptKind::Certificate,
            "proof_obligation:portable:receipt_gate",
            "certificate:fingerprint:portable:receipt_gate",
        );
        let witness_receipt = AYProofValidationReceipt::validator_backed(
            "receipt:witness:portable:receipt_gate",
            AYProofValidationReceiptKind::Witness,
            "proof_obligation:portable:receipt_gate",
            "witness:fingerprint:portable:receipt_gate",
        );

        assert_eq!(
            descriptor.required_analytical_big_win_receipt_kind_codes(),
            vec!["proof_transcript", "model", "certificate", "witness"]
        );
        assert!(
            descriptor.can_publish_analytical_big_win_with_validation_receipts(
                Some(&proof_receipt),
                Some(&model_receipt),
                Some(&certificate_receipt),
                Some(&witness_receipt),
            )
        );
        let row = descriptor
            .render_analytical_big_win_evidence_with_validation_receipts(
                "AY",
                Some(&proof_receipt),
                Some(&model_receipt),
                Some(&certificate_receipt),
                Some(&witness_receipt),
            )
            .expect("complete validator-backed receipt set should publish");
        assert!(row.contains("validation_receipt_backed=true"));
        assert!(row.contains("publishable_shared_win=true"));

        let artifact_only_proof = proof_receipt
            .clone()
            .with_status(AYProofValidationReceiptStatus::ArtifactOnly);
        let artifact_only_model = model_receipt
            .clone()
            .with_status(AYProofValidationReceiptStatus::ArtifactOnly);
        let artifact_only_certificate = certificate_receipt
            .clone()
            .with_status(AYProofValidationReceiptStatus::ArtifactOnly);
        let artifact_only_witness = witness_receipt
            .clone()
            .with_status(AYProofValidationReceiptStatus::ArtifactOnly);

        for (proof, model, certificate, witness) in [
            (
                None,
                Some(&model_receipt),
                Some(&certificate_receipt),
                Some(&witness_receipt),
            ),
            (
                Some(&artifact_only_proof),
                Some(&model_receipt),
                Some(&certificate_receipt),
                Some(&witness_receipt),
            ),
            (
                Some(&proof_receipt),
                None,
                Some(&certificate_receipt),
                Some(&witness_receipt),
            ),
            (
                Some(&proof_receipt),
                Some(&artifact_only_model),
                Some(&certificate_receipt),
                Some(&witness_receipt),
            ),
            (
                Some(&proof_receipt),
                Some(&model_receipt),
                None,
                Some(&witness_receipt),
            ),
            (
                Some(&proof_receipt),
                Some(&model_receipt),
                Some(&artifact_only_certificate),
                Some(&witness_receipt),
            ),
            (
                Some(&proof_receipt),
                Some(&model_receipt),
                Some(&certificate_receipt),
                None,
            ),
            (
                Some(&proof_receipt),
                Some(&model_receipt),
                Some(&certificate_receipt),
                Some(&artifact_only_witness),
            ),
        ] {
            assert!(
                !descriptor.can_publish_analytical_big_win_with_validation_receipts(
                    proof,
                    model,
                    certificate,
                    witness,
                )
            );
            assert!(descriptor
                .render_analytical_big_win_evidence_with_validation_receipts(
                    "AY",
                    proof,
                    model,
                    certificate,
                    witness,
                )
                .is_none());
        }
    }

    #[test]
    fn proof_lane_publication_requires_validator_backed_receipt_shape() {
        let descriptor = AYSharedProofLaneDescriptor::new(
            AYSharedEngineLane::Bmc,
            AYFrontendFamily::MccPetri,
            "prepared_program:mcc_petri:net",
            "proof_obligation:mcc_petri:coverability",
        )
        .with_model_fingerprint_identity("model:fingerprint:mcc_petri:coverability");

        let valid_receipt = AYProofValidationReceipt::validator_backed(
            "receipt:mcc_petri:coverability",
            AYProofValidationReceiptKind::Model,
            "proof_obligation:mcc_petri:coverability",
            "model:fingerprint:mcc_petri:coverability",
        );
        assert!(descriptor.can_publish_with_receipt(Some(&valid_receipt)));

        let row = descriptor
            .render_publication_evidence("AY", Some(&valid_receipt))
            .expect("validator-backed receipt should publish the proof lane");
        assert!(row.contains(" ay_shared_proof_lane_publication "));
        assert!(row.contains(
            "shared_proof_lane_identity=bmc:prepared_program:mcc_petri:net:proof_obligation:mcc_petri:coverability"
        ));
        assert!(row.contains(
            "analytical_solve_contract_identity=analytical_ay_proof:bmc:prepared_program:mcc_petri:net:proof_obligation:mcc_petri:coverability"
        ));
        assert!(row.contains(
            "analytical_big_win_detection_rule=shared_analytical_solve_replaces_frontend_specific_search_or_enumeration"
        ));
        assert!(row.contains("shared_engine_component=analytical_ay_proof"));
        assert!(row.contains("origin_frontend=mcc_petri"));
        assert!(row.contains("first_beneficiary=mcc_petri"));
        assert!(row.contains("second_beneficiary=ay_analytical"));
        assert!(row.contains("extraction_status=shared-core-ready"));
        assert!(row.contains("blocker_status=tracked-blockers"));
        assert!(row.contains(
            "frontend_family_blockers=future_importer:awaiting_registered_importer_frontend"
        ));
        assert!(row.contains("shared_owner=shared_high_performance_engine"));
        assert!(row.contains("lane=bmc"));
        assert!(row.contains("source_frontend_family=mcc_petri"));
        assert!(row.contains("prepared_program_identity=prepared_program:mcc_petri:net"));
        assert!(row.contains("proof_obligation_identity=proof_obligation:mcc_petri:coverability"));
        assert!(row.contains("model_fingerprint_identity=model:fingerprint:mcc_petri:coverability"));
        assert!(row.contains("certificate_fingerprint_identity=none"));
        assert!(
            row.contains("shared_fingerprint_identities=model:fingerprint:mcc_petri:coverability")
        );
        assert!(row.contains("proof_receipt_identity=receipt:mcc_petri:coverability"));
        assert!(row.contains("validation_receipt_identity=receipt:mcc_petri:coverability"));
        assert!(row.contains("validation_receipt_schema=tla-ay.shared-proof-validation-receipt/v1"));
        assert!(row.contains("validation_kind=model"));
        assert!(row.contains("validation_status=validator_backed"));
        assert!(
            row.contains("validated_fingerprint_identity=model:fingerprint:mcc_petri:coverability")
        );
        assert!(row.contains(
            "compatible_frontend_families=tla_plus,quint,mcc_petri,aiger,btor2,ay_analytical,vmt_transition_system,witness_replay"
        ));
        assert!(row.contains(
            "compatible_frontend_codes=tla,quint,mcc_petri,aiger,btor2,ay_only,vmt_replay,witness_replay"
        ));

        let artifact_only_receipt =
            valid_receipt.with_status(AYProofValidationReceiptStatus::ArtifactOnly);
        assert!(!descriptor.can_publish_with_receipt(Some(&artifact_only_receipt)));
        assert!(descriptor
            .render_publication_evidence("AY", Some(&artifact_only_receipt))
            .is_none());
        assert!(!descriptor.can_publish_with_receipt(None));
        assert!(descriptor.render_publication_evidence("AY", None).is_none());
    }

    #[test]
    fn hardware_adoption_publication_uses_shared_receipt_gate_and_beneficiary_fields() {
        let descriptor = AYSharedProofLaneDescriptor::new(
            AYSharedEngineLane::Bmc,
            AYFrontendFamily::Aiger,
            "prepared_program:aiger:register_vector",
            "proof_obligation:aiger:hardware_safety",
        )
        .with_proof_fingerprint_identity("proof:fingerprint:aiger:hardware_safety")
        .with_witness_fingerprint_identity("witness:fingerprint:aiger:counterexample");

        let valid_proof_receipt = AYProofValidationReceipt::validator_backed(
            "receipt:aiger:hardware_safety",
            AYProofValidationReceiptKind::ProofTranscript,
            "proof_obligation:aiger:hardware_safety",
            "proof:fingerprint:aiger:hardware_safety",
        );
        let valid_witness_receipt = AYProofValidationReceipt::validator_backed(
            "receipt:aiger:counterexample",
            AYProofValidationReceiptKind::Witness,
            "proof_obligation:aiger:hardware_safety",
            "witness:fingerprint:aiger:counterexample",
        );
        let row = descriptor
            .render_hardware_adoption_evidence_with_receipts(
                "AIGER",
                "hardware_ay_proof_lane_adoption",
                "aiger.hardware_register_layout.v1",
                "aiger_hardware_register_vector",
                "tla_and_mcc_shared_ay_proof_lanes",
                Some(&valid_proof_receipt),
                Some(&valid_witness_receipt),
            )
            .expect("validator-backed receipts should publish hardware adoption");

        assert!(row.starts_with("AIGER hardware_ay_proof_lane_adoption "));
        assert!(row.contains("origin_frontend=aiger"));
        assert!(row.contains(
            "shared_proof_lane_identity=bmc:prepared_program:aiger:register_vector:proof_obligation:aiger:hardware_safety"
        ));
        assert!(row.contains(
            "analytical_big_win_detection_rule=shared_analytical_solve_replaces_frontend_specific_search_or_enumeration"
        ));
        assert!(row.contains("shared_engine_component=analytical_ay_proof"));
        assert!(row.contains("register_vector_identity=aiger.hardware_register_layout.v1"));
        assert!(row.contains("model_fingerprint_identity=none"));
        assert!(row.contains("certificate_fingerprint_identity=none"));
        assert!(row.contains("validation_receipt_schema=tla-ay.shared-proof-validation-receipt/v1"));
        assert!(row.contains("validation_status=validator_backed"));
        assert!(row.contains("required_receipt_kinds=proof_transcript,witness"));
        assert!(row.contains("proof_receipt_identity=receipt:aiger:hardware_safety"));
        assert!(row.contains("proof_receipt_schema=tla-ay.shared-proof-validation-receipt/v1"));
        assert!(row.contains("proof_receipt_required=true"));
        assert!(row.contains("proof_receipt_validation_kind=proof_transcript"));
        assert!(row.contains("proof_receipt_status=validator_backed"));
        assert!(row.contains(
            "proof_receipt_validated_fingerprint_identity=proof:fingerprint:aiger:hardware_safety"
        ));
        assert!(row.contains("witness_receipt_required=true"));
        assert!(row.contains("witness_receipt_identity=receipt:aiger:counterexample"));
        assert!(row.contains("witness_receipt_schema=tla-ay.shared-proof-validation-receipt/v1"));
        assert!(row.contains("witness_receipt_validation_kind=witness"));
        assert!(row.contains("witness_receipt_status=validator_backed"));
        assert!(row.contains(
            "witness_receipt_validated_fingerprint_identity=witness:fingerprint:aiger:counterexample"
        ));
        assert!(row.contains("validation_receipt_identity=receipt:aiger:hardware_safety"));
        assert!(row.contains(
            "shared_fingerprint_identities=proof:fingerprint:aiger:hardware_safety,witness:fingerprint:aiger:counterexample"
        ));
        assert!(row.contains(
            "compatible_frontend_families=tla_plus,quint,mcc_petri,aiger,btor2,ay_analytical,vmt_transition_system,witness_replay"
        ));
        assert!(row.contains("first_beneficiary=aiger_hardware_register_vector"));
        assert!(row.contains("second_beneficiary=tla_and_mcc_shared_ay_proof_lanes"));
        assert!(row.contains("extraction_status=shared-core-ready"));
        assert!(row.contains("blocker_status=tracked-blockers"));
        assert!(row.contains(
            "frontend_family_blockers=future_importer:awaiting_registered_importer_frontend"
        ));
        assert!(row.contains("shared_owner=shared_high_performance_engine"));

        let artifact_only_receipt = valid_proof_receipt
            .clone()
            .with_status(AYProofValidationReceiptStatus::ArtifactOnly);
        let artifact_only_witness_receipt = valid_witness_receipt
            .clone()
            .with_status(AYProofValidationReceiptStatus::ArtifactOnly);
        let mislabeled_proof_receipt = AYProofValidationReceipt::validator_backed(
            "receipt:aiger:hardware_safety:model",
            AYProofValidationReceiptKind::Model,
            "proof_obligation:aiger:hardware_safety",
            "proof:fingerprint:aiger:hardware_safety",
        );
        let mislabeled_witness_receipt = AYProofValidationReceipt::validator_backed(
            "receipt:aiger:counterexample:proof",
            AYProofValidationReceiptKind::ProofTranscript,
            "proof_obligation:aiger:hardware_safety",
            "witness:fingerprint:aiger:counterexample",
        );
        assert!(descriptor
            .render_hardware_adoption_evidence_with_receipts(
                "AIGER",
                "hardware_ay_proof_lane_adoption",
                "aiger.hardware_register_layout.v1",
                "aiger_hardware_register_vector",
                "tla_and_mcc_shared_ay_proof_lanes",
                Some(&artifact_only_receipt),
                Some(&valid_witness_receipt),
            )
            .is_none());
        assert!(descriptor
            .render_hardware_adoption_evidence_with_receipts(
                "AIGER",
                "hardware_ay_proof_lane_adoption",
                "aiger.hardware_register_layout.v1",
                "aiger_hardware_register_vector",
                "tla_and_mcc_shared_ay_proof_lanes",
                None,
                Some(&valid_witness_receipt),
            )
            .is_none());
        assert!(descriptor
            .render_hardware_adoption_evidence_with_receipts(
                "AIGER",
                "hardware_ay_proof_lane_adoption",
                "aiger.hardware_register_layout.v1",
                "aiger_hardware_register_vector",
                "tla_and_mcc_shared_ay_proof_lanes",
                Some(&valid_proof_receipt),
                None,
            )
            .is_none());
        assert!(descriptor
            .render_hardware_adoption_evidence_with_receipts(
                "AIGER",
                "hardware_ay_proof_lane_adoption",
                "aiger.hardware_register_layout.v1",
                "aiger_hardware_register_vector",
                "tla_and_mcc_shared_ay_proof_lanes",
                Some(&valid_proof_receipt),
                Some(&artifact_only_witness_receipt),
            )
            .is_none());
        assert!(descriptor
            .render_hardware_adoption_evidence_with_receipts(
                "AIGER",
                "hardware_ay_proof_lane_adoption",
                "aiger.hardware_register_layout.v1",
                "aiger_hardware_register_vector",
                "tla_and_mcc_shared_ay_proof_lanes",
                Some(&mislabeled_proof_receipt),
                Some(&valid_witness_receipt),
            )
            .is_none());
        assert!(descriptor
            .render_hardware_adoption_evidence_with_receipts(
                "AIGER",
                "hardware_ay_proof_lane_adoption",
                "aiger.hardware_register_layout.v1",
                "aiger_hardware_register_vector",
                "tla_and_mcc_shared_ay_proof_lanes",
                Some(&valid_proof_receipt),
                Some(&mislabeled_witness_receipt),
            )
            .is_none());
    }

    #[test]
    fn proof_lane_publication_rejects_mismatched_receipt_identity_boundary() {
        let descriptor = AYSharedProofLaneDescriptor::new(
            AYSharedEngineLane::KInduction,
            AYFrontendFamily::VmtReplay,
            "prepared_program:vmt_replay:counter",
            "proof_obligation:vmt_replay:safety",
        )
        .with_proof_fingerprint_identity("proof:fingerprint:vmt_replay:safety")
        .with_witness_fingerprint_identity("witness:fingerprint:vmt_replay:safety");

        let mismatched_obligation = AYProofValidationReceipt::validator_backed(
            "receipt:vmt_replay:safety",
            AYProofValidationReceiptKind::ProofTranscript,
            "proof_obligation:vmt_replay:other",
            "proof:fingerprint:vmt_replay:safety",
        );
        let mismatched_fingerprint = AYProofValidationReceipt::validator_backed(
            "receipt:vmt_replay:safety",
            AYProofValidationReceiptKind::ProofTranscript,
            "proof_obligation:vmt_replay:safety",
            "proof:fingerprint:vmt_replay:other",
        );

        assert!(!descriptor.can_publish_with_receipt(Some(&mismatched_obligation)));
        assert!(!descriptor.can_publish_with_receipt(Some(&mismatched_fingerprint)));
        assert!(descriptor
            .render_publication_evidence("AY", Some(&mismatched_obligation))
            .is_none());
        assert!(descriptor
            .render_publication_evidence("AY", Some(&mismatched_fingerprint))
            .is_none());
    }

    #[test]
    fn proof_lane_publication_rejects_future_importer_until_registered_even_with_certificate_receipt(
    ) {
        let descriptor = AYSharedProofLaneDescriptor::new(
            AYSharedEngineLane::Chc,
            AYFrontendFamily::FutureImporter,
            "prepared_program:future:normalized_transition_system",
            "proof_obligation:future:safety_certificate",
        )
        .with_certificate_fingerprint_identity("certificate:fingerprint:future:safety");

        let certificate_receipt = AYProofValidationReceipt::validator_backed(
            "receipt:future:safety_certificate",
            AYProofValidationReceiptKind::Certificate,
            "proof_obligation:future:safety_certificate",
            "certificate:fingerprint:future:safety",
        );

        assert!(!descriptor.can_publish_with_receipt(Some(&certificate_receipt)));
        assert!(descriptor
            .render_publication_evidence("AY", Some(&certificate_receipt))
            .is_none());
    }

    #[test]
    fn proof_lane_publication_accepts_certificate_receipt_for_compatible_frontend_family() {
        let descriptor = AYSharedProofLaneDescriptor::new(
            AYSharedEngineLane::Chc,
            AYFrontendFamily::WitnessReplay,
            "prepared_program:replay:normalized_transition_system",
            "proof_obligation:replay:safety_certificate",
        )
        .with_certificate_fingerprint_identity("certificate:fingerprint:replay:safety");

        let certificate_receipt = AYProofValidationReceipt::validator_backed(
            "receipt:replay:safety_certificate",
            AYProofValidationReceiptKind::Certificate,
            "proof_obligation:replay:safety_certificate",
            "certificate:fingerprint:replay:safety",
        );

        assert!(descriptor.can_publish_with_receipt(Some(&certificate_receipt)));
        let row = descriptor
            .render_publication_evidence("AY", Some(&certificate_receipt))
            .expect("certificate receipt should publish frontend-neutral proof lane");

        assert!(row.contains("origin_frontend=witness_replay"));
        assert!(row.contains("first_beneficiary=witness_replay"));
        assert!(row.contains("second_beneficiary=ay_analytical"));
        assert!(
            row.contains("certificate_fingerprint_identity=certificate:fingerprint:replay:safety")
        );
        assert!(row.contains("shared_fingerprint_identities=certificate:fingerprint:replay:safety"));
        assert!(row.contains("validation_kind=certificate"));
        assert!(row.contains("validation_status=validator_backed"));
        assert!(row.contains("proof_receipt_identity=receipt:replay:safety_certificate"));
        assert!(
            row.contains("validated_fingerprint_identity=certificate:fingerprint:replay:safety")
        );
        assert!(row.contains(
            "compatible_frontend_families=tla_plus,quint,mcc_petri,aiger,btor2,ay_analytical,vmt_transition_system,witness_replay"
        ));
    }

    #[test]
    fn render_helpers_emit_stable_shared_engine_rows() {
        let lane_row = render_ay_shared_engine_lane_evidence("AY", AYSharedEngineLane::Chc);
        assert!(lane_row.contains("AY ay_shared_engine_lane_metadata "));
        assert!(lane_row.contains("lane=chc"));
        assert!(lane_row.contains("frontend_neutral=true"));
        assert!(lane_row.contains(
            "compatible_frontend_codes=tla,quint,mcc_petri,aiger,btor2,ay_only,vmt_replay,witness_replay"
        ));
        assert!(lane_row.contains(
            "compatible_frontend_families=tla_plus,quint,mcc_petri,aiger,btor2,ay_analytical,vmt_transition_system,witness_replay"
        ));

        let engine_row = render_ay_shared_engine_evidence("AY");
        assert!(engine_row.contains("AY ay_shared_engine_metadata "));
        assert!(engine_row.contains("lanes=all_sat_enumeration,bmc,chc,pdr,k_induction"));
        assert!(engine_row.contains(
            "compatible_frontend_codes=tla,quint,mcc_petri,aiger,btor2,ay_only,vmt_replay,witness_replay"
        ));
        assert!(engine_row.contains(
            "compatible_frontend_families=tla_plus,quint,mcc_petri,aiger,btor2,ay_analytical,vmt_transition_system,witness_replay"
        ));
        assert!(engine_row.contains(AY_ANALYTICAL_PROOF_SHARED_ENGINE_COMPONENT));

        assert_eq!(
            AYProofValidationReceiptKind::OutputFormat.code(),
            "output_format"
        );
        assert_eq!(
            AYProofValidationReceiptKind::Certificate.code(),
            "certificate"
        );
        assert_eq!(AYProofValidationReceiptKind::Witness.code(), "witness");
        assert_eq!(AYProofValidationReceiptStatus::Missing.code(), "missing");
        assert_eq!(
            AY_SHARED_PROOF_VALIDATION_RECEIPT_SCHEMA,
            "tla-ay.shared-proof-validation-receipt/v1"
        );
    }
}
