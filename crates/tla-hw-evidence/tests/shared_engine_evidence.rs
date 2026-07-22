// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Integration tests for the frontend-neutral machinery in `tla-hw-evidence`.
//!
//! The crate is consumed by the AIGER and BTOR2 frontends, which exercise the
//! generic builder against their concrete wirings. These tests instead pin the
//! *frontend-neutral* public contract directly: the pure evidence-token helpers,
//! the stable shared-frontend-family constants, the FNV-1a prepared-program
//! identity digest, and the generic `SharedEngineEvidence` pipeline driven by a
//! minimal in-test `HardwareFrontend` implementation.

use tla_ay::{
    AYFrontendFamily, AYProofValidationReceiptKind, AYProofValidationReceiptStatus,
    AYSharedEngineLane,
};
use tla_hw_evidence::{
    ay_proof_lane_adoption_evidence_row, ay_proof_lane_descriptor, ay_proof_lane_receipt,
    ay_witness_lane_receipt, evidence_option, evidence_token, prepared_identity,
    prepared_program_identity_digest, register_vector_admission_base_plan,
    register_vector_admission_plan, shared_engine_evidence_rows, validation_plan, HardwareFrontend,
    SharedEngineEvidence, HARDWARE_REGISTER_VECTOR_BLOCKERS,
    HARDWARE_REGISTER_VECTOR_COMPATIBLE_FRONTEND_FAMILIES,
    HARDWARE_REGISTER_VECTOR_DEFAULT_CONSUMERS,
    HARDWARE_REGISTER_VECTOR_REMAINING_COMPATIBLE_FRONTEND_FAMILIES,
};
use tla_mc_core::{
    validate_prepared_candidate_lane_evidence_row, validate_prepared_checker_program_evidence_row,
    validate_prepared_validation_plan_evidence_row, validate_shared_engine_adoption_evidence_row,
    validate_validation_receipt_evidence_row, CheckerSourceKind, PreparedCandidateLaneDescriptor,
    PreparedCheckerProgram, PreparedProgramPayloadKind, PreparedPropertyKind, PreparedStorageKind,
    PreparedTransitionKind, PreparedValidationKind, ProblemKind, SetupTraceLaneKind,
    ValidationReceiptStatus,
};

// --- Minimal in-test frontend -------------------------------------------------

/// A tiny opaque "input" for the test frontend; the count varies the program
/// identity so we can pin digest content-sensitivity.
struct TestInput {
    bad_count: usize,
}

/// Minimal valid hardware frontend that drives the generic machinery.
///
/// It mirrors the shape the real AIGER/BTOR2 frontends build (a hardware
/// register-vector prepared program with a `hardware_state_fingerprint`
/// candidate lane) but is trimmed to the smallest program the neutral builder
/// requires.
struct TestHwFrontend;

const TEST_REGISTER_LAYOUT_IDENTITY: &str = "test.hardware_register_layout.v1";

impl HardwareFrontend for TestHwFrontend {
    type Input = TestInput;

    const LABEL: &'static str = "TEST";
    const ORIGIN_FRONTEND: &'static str = "btor2";
    const SHARED_ENGINE_COMPONENT: &'static str = "tla_mc_core.prepared_checker_program";
    const SHARED_ENGINE_OWNER: &'static str = "shared_high_performance_engine";
    const PORTFOLIO: &'static str = "test_portfolio";
    const SHARED_ENGINE_SECOND_BENEFICIARY: &'static str = "other_portfolio";
    const SHARED_ENGINE_EXTRACTION_STATUS: &'static str = "shared-core-ready";
    const ACCEPTANCE_TEST: &'static str = "cargo test -p tla-hw-evidence";
    const PREPARED_PROGRAM_DIGEST_ALGORITHM: &'static str = "fnv1a64";

    const REGISTER_LAYOUT_IDENTITY: &'static str = TEST_REGISTER_LAYOUT_IDENTITY;
    const STATE_CANONICALIZATION: &'static str = "test-register-vector-v1";

    const ADMISSION_DESCRIPTION: &'static str = "test register-vector prepared admission";
    const CHECKER_SOURCE_KIND: CheckerSourceKind = CheckerSourceKind::Btor2;
    const PROGRAM_PAYLOAD_KIND: PreparedProgramPayloadKind = PreparedProgramPayloadKind::Btor2;

    const AY_PROOF_RECEIPT_PREREQUISITE: &'static str = "ay proof receipt";

    const AY_SAFETY_CANDIDATE_PREFIX: &'static str = "test.ay.safety_candidate";
    const AY_PROOF_ARTIFACT_PREFIX: &'static str = "test.ay.proof_artifact";
    const AY_PROOF_FINGERPRINT_PREFIX: &'static str = "test.ay.proof";
    const REPLAY_COUNTEREXAMPLE_CANDIDATE_PREFIX: &'static str = "test.replay.candidate";
    const REPLAY_COUNTEREXAMPLE_ARTIFACT_PREFIX: &'static str = "test.replay.artifact";
    const REPLAY_COUNTEREXAMPLE_PREFIX: &'static str = "test.replay.counterexample";

    const AY_SHARED_ENGINE_LANE: AYSharedEngineLane = AYSharedEngineLane::Bmc;
    const AY_FRONTEND_FAMILY: AYFrontendFamily = AYFrontendFamily::Btor2;
    const AY_PROOF_OBLIGATION_IDENTITY: &'static str =
        "test.hardware_register_safety_obligation.v1";
    const AY_PROOF_LANE_RECEIPT_IDENTITY: &'static str = "test.ay_proof_lane.validation_receipt";
    const AY_WITNESS_LANE_RECEIPT_IDENTITY: &'static str =
        "test.ay_witness_lane.validation_receipt";
    const AY_PROOF_LANE_FIRST_BENEFICIARY: &'static str = "test_hardware_register_vector";
    const AY_PROOF_LANE_SECOND_BENEFICIARY: &'static str = "shared_ay_proof_lanes";

    fn prepared_checker_program(input: &Self::Input) -> PreparedCheckerProgram {
        let identity = format!("test:safety:bad={}", input.bad_count);
        let admission = register_vector_admission_base_plan::<TestHwFrontend>();
        let storage_policy = admission.dedup.storage_policy_identity();
        let fp_policy = admission.dedup.fingerprint.fingerprint_policy_identity();
        let fp_identity = admission.dedup.fingerprint.fingerprint_identity();

        let mut program = PreparedCheckerProgram::new(
            identity.clone(),
            PreparedProgramPayloadKind::Btor2,
            PreparedStorageKind::HardwareRegisters,
        )
        .with_canonical_payload_identity(prepared_identity("test.canonical_payload", &identity))
        .with_source_identity(prepared_identity("test.source", &identity))
        .with_config_identity(prepared_identity("test.config", "default"))
        .with_examination_identity(prepared_identity("test.examination", "safety"))
        .with_cache_key(prepared_identity("test.prepared.cache", &identity))
        .with_source_fingerprint(prepared_identity("test.source_fingerprint", &identity))
        .with_frontend_payload_identity(prepared_identity("test.payload", &identity))
        .with_frontend_payload_fingerprint(prepared_identity("test.payload_fingerprint", &identity))
        .with_artifact_identity(prepared_identity("test.prepared_program", &identity))
        .with_storage_layout_fingerprint(TEST_REGISTER_LAYOUT_IDENTITY)
        .with_storage_policy_identity(storage_policy.clone())
        .with_fingerprint_policy_identity(fp_policy.clone())
        .with_fingerprint_identity(fp_identity.clone())
        .with_transition_descriptor_fingerprint(prepared_identity(
            "test.transition_descriptor",
            &identity,
        ))
        .with_property_descriptor_fingerprint(prepared_identity(
            "test.property_descriptor",
            &identity,
        ))
        .with_validation_plan_fingerprint(prepared_identity("test.validation_plan", &identity))
        .with_fingerprint(admission.prepared_fingerprint_descriptor())
        .add_transition("test.next_state", PreparedTransitionKind::HardwareNextState);

        for index in 0..input.bad_count.max(1) {
            program =
                program.add_property(format!("test.bad.{index}"), PreparedPropertyKind::BadState);
        }

        program = program
            .add_candidate_lane(
                PreparedCandidateLaneDescriptor::new("test.ay.safety", SetupTraceLaneKind::AY)
                    .with_candidate_key("ay_safety")
                    .with_candidate_identity(prepared_identity(
                        "test.ay.safety_candidate",
                        &identity,
                    ))
                    .with_lane_identity("shared_ay_sat")
                    .with_fingerprint_policy_identity("test_proof_fingerprint.v1")
                    .with_fingerprint_identity(prepared_identity("test.ay.proof", &identity)),
            )
            .add_candidate_lane(
                PreparedCandidateLaneDescriptor::new(
                    "test.fingerprint.hardware_state",
                    SetupTraceLaneKind::Fingerprint,
                )
                .with_candidate_key("hardware_state_fingerprint")
                .with_candidate_identity(prepared_identity("test.fingerprint.state", &identity))
                .with_lane_identity("shared_hardware_state_fingerprint")
                .with_storage_policy_identity(storage_policy)
                .with_fingerprint_policy_identity(fp_policy)
                .with_fingerprint_identity(fp_identity),
            )
            .add_validation_plan(validation_plan(
                &identity,
                PreparedValidationKind::AYProof,
                ProblemKind::Sat,
                "test.validation.ay_proof",
                "test.ay.proof_fingerprint",
                "test-ay-proof-v1",
                "test_proof_fingerprint.v1",
                "test.ay.proof",
                "test.ay.proof_artifact",
            ));

        program
    }
}

fn build(bad_count: usize) -> SharedEngineEvidence<TestHwFrontend> {
    SharedEngineEvidence::<TestHwFrontend>::from_input(&TestInput { bad_count })
}

// --- Pure helper: evidence_token ---------------------------------------------

#[test]
fn evidence_token_empty_input_becomes_none_sentinel() {
    // Documented contract: empty becomes "none".
    assert_eq!(evidence_token(""), "none");
}

#[test]
fn evidence_token_preserves_allowed_characters() {
    // Documented contract: ASCII-alphanumeric plus -_.:= survive unchanged.
    let allowed = "abcXYZ0189-_.:=";
    assert_eq!(evidence_token(allowed), allowed);
}

#[test]
fn evidence_token_replaces_disallowed_with_underscore() {
    // Spaces, slashes, commas, and non-ASCII all collapse to '_'.
    assert_eq!(evidence_token("a b/c,d"), "a_b_c_d");
    assert_eq!(evidence_token("café"), "caf_");
    // A token that is non-empty but entirely disallowed stays non-empty (it
    // does NOT collapse to the "none" sentinel, which is reserved for empty).
    assert_eq!(evidence_token("   "), "___");
    assert_eq!(evidence_token("\n\t"), "__");
}

#[test]
fn evidence_token_is_idempotent_on_its_own_output() {
    // Re-tokenizing an already-tokenized value must be a no-op, otherwise the
    // value embedded in evidence rows would not be stable under re-rendering.
    for raw in ["", "x y", "a.b:c=d", "weird/value\u{1f600}", "   "] {
        let once = evidence_token(raw);
        assert_eq!(evidence_token(&once), once, "not idempotent for {raw:?}");
    }
}

// --- Pure helper: evidence_option --------------------------------------------

#[test]
fn evidence_option_none_is_none_sentinel() {
    assert_eq!(evidence_option(None), "none");
}

#[test]
fn evidence_option_some_is_tokenized() {
    assert_eq!(evidence_option(Some("ok value")), "ok_value");
    // Some("") tokenizes the empty string, which itself maps to "none".
    assert_eq!(evidence_option(Some("")), "none");
}

// --- Pure helper: prepared_identity ------------------------------------------

#[test]
fn prepared_identity_formats_prefix_colon_token() {
    assert_eq!(
        prepared_identity("aiger.source", "circuit id"),
        "aiger.source:circuit_id"
    );
}

#[test]
fn prepared_identity_tokenizes_only_the_identity_not_the_prefix() {
    // The prefix is trusted/static; only the identity is normalized.
    let out = prepared_identity("p", "a/b");
    assert_eq!(out, "p:a_b");
    // An empty identity yields the "none" sentinel after the colon.
    assert_eq!(prepared_identity("p", ""), "p:none");
}

// --- Shared frontend-family constants ----------------------------------------

#[test]
fn shared_register_vector_constants_have_stable_wire_values() {
    // These strings are emitted verbatim into evidence rows and consumed by
    // downstream validators, so their exact value is a public contract.
    assert_eq!(
        HARDWARE_REGISTER_VECTOR_COMPATIBLE_FRONTEND_FAMILIES,
        "aiger,btor2,vmt_transition_system,ay_analytical,witness_replay"
    );
    assert_eq!(HARDWARE_REGISTER_VECTOR_DEFAULT_CONSUMERS, "aiger,btor2");
    assert_eq!(
        HARDWARE_REGISTER_VECTOR_REMAINING_COMPATIBLE_FRONTEND_FAMILIES,
        "vmt_transition_system,ay_analytical,witness_replay"
    );
    assert_eq!(
        HARDWARE_REGISTER_VECTOR_BLOCKERS,
        "future_importer:awaiting_registered_importer_frontend"
    );
}

#[test]
fn default_consumers_and_remaining_partition_the_compatible_families() {
    // Invariant: default consumers + remaining families = the full compatible
    // set, with no overlap. This guards the three constants against drift.
    let compatible: Vec<&str> = HARDWARE_REGISTER_VECTOR_COMPATIBLE_FRONTEND_FAMILIES
        .split(',')
        .collect();
    let defaults: Vec<&str> = HARDWARE_REGISTER_VECTOR_DEFAULT_CONSUMERS
        .split(',')
        .collect();
    let remaining: Vec<&str> = HARDWARE_REGISTER_VECTOR_REMAINING_COMPATIBLE_FRONTEND_FAMILIES
        .split(',')
        .collect();

    let mut union: Vec<&str> = defaults.iter().chain(remaining.iter()).copied().collect();
    union.sort_unstable();
    let mut compatible_sorted = compatible.clone();
    compatible_sorted.sort_unstable();
    assert_eq!(union, compatible_sorted);

    for d in &defaults {
        assert!(
            !remaining.contains(d),
            "default consumer {d} also listed as remaining"
        );
    }
}

// --- FNV-1a prepared-program identity digest ---------------------------------

#[test]
fn prepared_program_digest_is_16_char_lowercase_hex() {
    let evidence = build(1);
    let digest = &evidence.prepared_program_digest;
    assert_eq!(digest.len(), 16, "FNV-1a/64 renders to 16 hex chars");
    assert!(
        digest
            .chars()
            .all(|c| c.is_ascii_digit() || ('a'..='f').contains(&c)),
        "digest {digest} is not lowercase hex"
    );
}

#[test]
fn prepared_program_digest_is_deterministic() {
    let a = build(2);
    let b = build(2);
    assert_eq!(a.prepared_program_digest, b.prepared_program_digest);
    // The free function and the bundled field agree.
    assert_eq!(
        prepared_program_identity_digest::<TestHwFrontend>(&a.prepared_program),
        a.prepared_program_digest
    );
}

#[test]
fn prepared_program_digest_changes_with_program_content() {
    // A different property count yields a different identity and therefore a
    // different digest; the digest is content-sensitive, not a constant.
    let one = build(1);
    let two = build(3);
    assert_ne!(one.prepared_program.identity, two.prepared_program.identity);
    assert_ne!(
        one.prepared_program_digest, two.prepared_program_digest,
        "digest must depend on prepared-program content"
    );
}

#[test]
fn prepared_program_digest_matches_independent_fnv1a_over_identity_rows() {
    // Recompute the digest independently from the public render API to pin the
    // exact hashing convention (newline-terminated identity rows, FNV-1a/64).
    let evidence = build(1);
    let program = &evidence.prepared_program;

    let mut rows: Vec<String> = Vec::new();
    rows.push(program.render_evidence_row(TestHwFrontend::LABEL));
    rows.extend(program.render_frontend_extension_evidence_rows(TestHwFrontend::LABEL));
    rows.extend(program.render_candidate_lane_evidence_rows(TestHwFrontend::LABEL));
    rows.extend(program.render_validation_plan_evidence_rows(TestHwFrontend::LABEL));

    const OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const PRIME: u64 = 0x0000_0100_0000_01b3;
    let mut hash = OFFSET;
    for row in &rows {
        for byte in row.as_bytes() {
            hash ^= u64::from(*byte);
            hash = hash.wrapping_mul(PRIME);
        }
        hash ^= u64::from(b'\n');
        hash = hash.wrapping_mul(PRIME);
    }
    assert_eq!(format!("{hash:016x}"), evidence.prepared_program_digest);
}

// --- Generic SharedEngineEvidence pipeline -----------------------------------

#[test]
fn from_input_produces_internally_consistent_accepted_receipts() {
    let evidence = build(1);

    // The adoption row and admission plan validate.
    evidence.adoption.validate().unwrap();
    evidence.register_vector_admission.validate().unwrap();

    // Both classic receipts are accepted and bound to the program digest.
    evidence.ay_proof_receipt.validate().unwrap();
    evidence.replay_receipt.validate().unwrap();
    assert_eq!(
        evidence.ay_proof_receipt.status,
        ValidationReceiptStatus::Accepted
    );
    assert_eq!(
        evidence.replay_receipt.status,
        ValidationReceiptStatus::Accepted
    );
    assert_eq!(
        evidence.ay_proof_receipt.digest,
        evidence.prepared_program_digest
    );
    assert_eq!(
        evidence.replay_receipt.digest,
        evidence.prepared_program_digest
    );

    // The validator-backed lane receipts carry the configured identities/kinds.
    assert_eq!(
        evidence.ay_proof_lane_receipt.validation_kind,
        AYProofValidationReceiptKind::ProofTranscript
    );
    assert_eq!(
        evidence.ay_witness_lane_receipt.validation_kind,
        AYProofValidationReceiptKind::Witness
    );
    assert_eq!(
        evidence.ay_proof_lane_receipt.status,
        AYProofValidationReceiptStatus::ValidatorBacked
    );
}

#[test]
fn register_vector_admission_binds_to_the_fingerprint_lane_when_present() {
    // The program built by the test frontend carries a
    // `hardware_state_fingerprint` candidate lane, so the admission plan must
    // bind to that lane (candidate_key set) rather than the program as a whole.
    let evidence = build(1);
    let admission = &evidence.register_vector_admission;
    assert_eq!(
        admission.candidate_key.as_deref(),
        Some("hardware_state_fingerprint")
    );
    assert_eq!(
        admission.prepared_program_identity.as_deref(),
        Some(evidence.prepared_program.identity.as_str())
    );
    assert_eq!(
        admission.payload_witness.code(),
        "register_vector_canonical"
    );
    assert_eq!(admission.source_kind, CheckerSourceKind::Btor2);

    // The standalone plan helper agrees with the bundled value.
    let standalone = register_vector_admission_plan::<TestHwFrontend>(&evidence.prepared_program);
    assert_eq!(standalone.candidate_key, admission.candidate_key);
    assert_eq!(standalone.id, admission.id);
}

#[test]
fn register_vector_admission_falls_back_to_whole_program_without_fingerprint_lane() {
    // A bare program with no fingerprint lane must bind the plan to the program
    // as a whole: no candidate key, but still carrying the program identity.
    let program = PreparedCheckerProgram::new(
        "test:bare".to_string(),
        PreparedProgramPayloadKind::Btor2,
        PreparedStorageKind::HardwareRegisters,
    )
    .add_transition("t", PreparedTransitionKind::HardwareNextState);

    let plan = register_vector_admission_plan::<TestHwFrontend>(&program);
    assert_eq!(plan.candidate_key, None);
    assert_eq!(plan.prepared_program_identity.as_deref(), Some("test:bare"));
    assert_eq!(plan.prepared_lane_identity, None);
}

#[test]
fn render_evidence_rows_emits_validatable_neutral_rows() {
    let evidence = build(1);
    let rows = evidence.render_evidence_rows();
    assert!(!rows.is_empty());

    // Every row is prefixed with the frontend label.
    for row in &rows {
        assert!(
            row.starts_with("TEST "),
            "row missing TEST label prefix: {row}"
        );
    }

    // The shared adoption row validates and pins the neutral family contract.
    let adoption_row = rows
        .iter()
        .find(|r| r.starts_with("TEST shared_engine_adoption "))
        .expect("adoption row");
    validate_shared_engine_adoption_evidence_row(adoption_row).unwrap();
    assert!(adoption_row.contains("origin_frontend=btor2"));
    assert!(adoption_row.contains("second_beneficiary=other_portfolio"));
    assert!(adoption_row.contains("adoption_level=level-3"));
    assert!(adoption_row.contains(&format!(
        "compatible_frontend_families={}",
        "tla_plus,quint,mcc_petri,aiger,btor2,vmt_transition_system,ay_analytical,witness_replay"
    )));

    // The prepared-program row validates.
    let prepared_row = rows
        .iter()
        .find(|r| r.starts_with("TEST prepared_checker_program "))
        .expect("prepared program row");
    validate_prepared_checker_program_evidence_row(prepared_row).unwrap();
    assert!(prepared_row.contains("payload_kind=btor2"));
    assert!(prepared_row.contains("storage_kind=hardware_registers"));

    // Every candidate-lane / validation-plan / receipt row validates.
    for row in rows
        .iter()
        .filter(|r| r.starts_with("TEST prepared_candidate_lane "))
    {
        validate_prepared_candidate_lane_evidence_row(row).unwrap();
    }
    for row in rows
        .iter()
        .filter(|r| r.starts_with("TEST prepared_validation_plan "))
    {
        validate_prepared_validation_plan_evidence_row(row).unwrap();
    }
    for row in rows
        .iter()
        .filter(|r| r.starts_with("TEST validation_receipt "))
    {
        validate_validation_receipt_evidence_row(row).unwrap();
    }

    // The neutral admission + hardware transition rows carry the shared
    // register-vector family contract verbatim.
    let admission_row = rows
        .iter()
        .find(|r| r.starts_with("TEST prepared_fingerprint_admission "))
        .expect("fingerprint admission row");
    assert!(admission_row.contains("admission_status=accepted"));
    assert!(admission_row.contains("payload_witness=register_vector_canonical"));
    assert!(admission_row.contains(&format!(
        "default_consumers={HARDWARE_REGISTER_VECTOR_DEFAULT_CONSUMERS}"
    )));
    assert!(admission_row.contains(&format!("blockers={HARDWARE_REGISTER_VECTOR_BLOCKERS}")));

    let transition_row = rows
        .iter()
        .find(|r| r.starts_with("TEST hardware_transition_system_adoption "))
        .expect("hardware transition row");
    assert!(transition_row.contains("origin_frontend=btor2"));
    assert!(transition_row.contains("transition_kind=hardware_next_state"));
    assert!(transition_row.contains("ay_analytical_lane=receipt_backed"));
    assert!(transition_row.contains("witness_replay_lane=receipt_backed"));
    assert!(transition_row.contains(&format!(
        "register_vector_identity={TEST_REGISTER_LAYOUT_IDENTITY}"
    )));

    // The generalized proof-lane adoption row is published (receipts present).
    let proof_lane_row = rows
        .iter()
        .find(|r| r.starts_with("TEST hardware_ay_proof_lane_adoption "))
        .expect("proof-lane adoption row");
    assert!(proof_lane_row.contains("lane=bmc"));
    assert!(proof_lane_row.contains("first_beneficiary=test_hardware_register_vector"));
    assert!(proof_lane_row.contains("second_beneficiary=shared_ay_proof_lanes"));
    assert!(proof_lane_row.contains("validation_status=validator_backed"));
}

#[test]
fn shared_engine_evidence_rows_free_fn_matches_bundle_render() {
    // The free convenience function is exactly `from_input(..).render_evidence_rows()`.
    let input = TestInput { bad_count: 2 };
    let via_free_fn = shared_engine_evidence_rows::<TestHwFrontend>(&input);
    let via_bundle = build(2).render_evidence_rows();
    assert_eq!(via_free_fn, via_bundle);
}

#[test]
fn proof_lane_adoption_fails_closed_without_validator_receipts() {
    // Publication is gated on validator-backed receipts; the descriptor must
    // refuse to render an adoption row otherwise.
    let evidence = build(1);
    let program = &evidence.prepared_program;

    let artifact_only_proof = evidence
        .ay_proof_lane_receipt
        .clone()
        .with_status(AYProofValidationReceiptStatus::ArtifactOnly);
    let artifact_only_witness = evidence
        .ay_witness_lane_receipt
        .clone()
        .with_status(AYProofValidationReceiptStatus::ArtifactOnly);

    // Missing proof receipt.
    assert!(ay_proof_lane_adoption_evidence_row::<TestHwFrontend>(
        program,
        None,
        Some(&evidence.ay_witness_lane_receipt)
    )
    .is_none());
    // Missing witness receipt.
    assert!(ay_proof_lane_adoption_evidence_row::<TestHwFrontend>(
        program,
        Some(&evidence.ay_proof_lane_receipt),
        None
    )
    .is_none());
    // Artifact-only proof receipt is not enough.
    assert!(ay_proof_lane_adoption_evidence_row::<TestHwFrontend>(
        program,
        Some(&artifact_only_proof),
        Some(&evidence.ay_witness_lane_receipt)
    )
    .is_none());
    // Artifact-only witness receipt is not enough.
    assert!(ay_proof_lane_adoption_evidence_row::<TestHwFrontend>(
        program,
        Some(&evidence.ay_proof_lane_receipt),
        Some(&artifact_only_witness)
    )
    .is_none());
    // Both validator-backed: publication succeeds.
    assert!(ay_proof_lane_adoption_evidence_row::<TestHwFrontend>(
        program,
        Some(&evidence.ay_proof_lane_receipt),
        Some(&evidence.ay_witness_lane_receipt)
    )
    .is_some());
}

#[test]
fn proof_lane_descriptor_can_publish_with_matching_receipts() {
    let evidence = build(1);
    let descriptor = ay_proof_lane_descriptor::<TestHwFrontend>(&evidence.prepared_program);
    assert!(descriptor.can_publish_with_receipt(Some(&evidence.ay_proof_lane_receipt)));
    assert!(descriptor.can_publish_with_receipt(Some(&evidence.ay_witness_lane_receipt)));
    // No receipt -> cannot publish (fail-closed).
    assert!(!descriptor.can_publish_with_receipt(None));

    // The standalone receipt builders agree with the bundled receipts.
    let proof = ay_proof_lane_receipt::<TestHwFrontend>(&evidence.prepared_program);
    let witness = ay_witness_lane_receipt::<TestHwFrontend>(&evidence.prepared_program);
    assert_eq!(
        proof.validation_kind,
        evidence.ay_proof_lane_receipt.validation_kind
    );
    assert_eq!(
        witness.validation_kind,
        evidence.ay_witness_lane_receipt.validation_kind
    );
}

// --- validation_plan free helper ---------------------------------------------

#[test]
fn validation_plan_builds_canonical_sha256_descriptor_with_expanded_identities() {
    let descriptor = validation_plan(
        "my-id",
        PreparedValidationKind::AYProof,
        ProblemKind::Sat,
        "plan.id",
        "fp.id",
        "canon-v1",
        "fp.policy.v1",
        "fp.prefix",
        "artifact.prefix",
    );
    // Structured fields: id/kind/problem and a canonical-bytes-sha256 fingerprint.
    assert_eq!(descriptor.id, "plan.id");
    assert_eq!(descriptor.kind, PreparedValidationKind::AYProof);
    assert_eq!(descriptor.problem, ProblemKind::Sat);
    let fingerprint = descriptor
        .fingerprint
        .as_ref()
        .expect("validation_plan attaches a fingerprint descriptor");
    assert_eq!(fingerprint.id, "fp.id");
    assert_eq!(
        fingerprint.scheme,
        tla_mc_core::PreparedFingerprintScheme::CanonicalBytesSha256
    );
    assert_eq!(fingerprint.canonicalization_version, "canon-v1");
    // The prefixes are expanded against the identity via `prepared_identity`.
    assert_eq!(
        fingerprint
            .identities
            .fingerprint_policy_identity
            .as_deref(),
        Some("fp.policy.v1")
    );
    assert_eq!(
        fingerprint.identities.fingerprint_identity.as_deref(),
        Some("fp.prefix:my-id")
    );
    assert_eq!(
        descriptor.identities.artifact_identity.as_deref(),
        Some("artifact.prefix:my-id")
    );
    // Tokenization is applied: a slash in the identity collapses to '_'.
    let slashed = validation_plan(
        "a/b",
        PreparedValidationKind::WitnessReplay,
        ProblemKind::Safety,
        "plan2.id",
        "fp2.id",
        "canon2",
        "fp2.policy",
        "fp2.prefix",
        "artifact2.prefix",
    );
    assert_eq!(
        slashed
            .fingerprint
            .as_ref()
            .and_then(|f| f.identities.fingerprint_identity.as_deref()),
        Some("fp2.prefix:a_b")
    );
    assert_eq!(slashed.kind, PreparedValidationKind::WitnessReplay);
    assert_eq!(slashed.problem, ProblemKind::Safety);
}

// --- Debug / Clone surface ----------------------------------------------------

#[test]
fn evidence_bundle_clone_is_a_faithful_copy() {
    let evidence = build(1);
    let cloned = evidence.clone();
    assert_eq!(
        cloned.prepared_program_digest,
        evidence.prepared_program_digest
    );
    assert_eq!(
        cloned.prepared_program.identity,
        evidence.prepared_program.identity
    );
    assert_eq!(
        cloned.render_evidence_rows(),
        evidence.render_evidence_rows()
    );
}

#[test]
fn evidence_bundle_debug_mentions_key_fields() {
    let evidence = build(1);
    let dbg = format!("{evidence:?}");
    assert!(dbg.contains("SharedEngineEvidence"));
    assert!(dbg.contains("prepared_program_digest"));
    assert!(dbg.contains("ay_proof_lane_receipt"));
}
