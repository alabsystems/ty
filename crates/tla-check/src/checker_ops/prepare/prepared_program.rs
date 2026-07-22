// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! TLA adapter for the frontend-neutral prepared checker program contract.
//!
//! This module is intentionally descriptor-only. It records the stable identity,
//! storage ABI, transition names, and configured property obligations that a TLA
//! checker run has prepared. It does not own runtime state, queues, bytecode,
//! native libraries, solver handles, or any execution decisions.

use crate::config::Config;
use tla_mc_core::{
    PreparedCandidateLaneDescriptor, PreparedCanonicalIdentityDescriptor,
    PreparedCanonicalIdentityKind, PreparedCheckerProgram, PreparedFrontendExtensionDescriptor,
    PreparedFrontendExtensionKind, PreparedProgramPayloadKind, PreparedPropertyKind,
    PreparedStorageKind, PreparedTransitionKind, PreparedValidationKind,
    PreparedValidationPlanDescriptor, ProblemKind, SetupTrace, SetupTraceLaneKind,
    SetupTraceValidationStatus, SharedCollisionPolicy, SharedDedupIdentity, SharedDedupScope,
    SharedDedupStorageKind, SharedEngineAdoptionEvidence, SharedEngineAdoptionFamilyBlocker,
    SharedEngineAdoptionLevel, SharedEngineFrontendFamily, SharedFingerprintAlgorithm,
    SharedFingerprintIdentity, SharedFingerprintValueKind,
};

const PREPARED_PROGRAM_CANONICALIZATION_VERSION: &str = "ty-prepared-program-v1";
const TLA_LOWERED_PAYLOAD_CANONICALIZATION_VERSION: &str = "ty-lowered-tla-ast-v1";
const QUINT_LOWERED_PAYLOAD_CANONICALIZATION_VERSION: &str = "ty-quint-json-ir-to-tla-ast-v1";
const TLA_STATE_FINGERPRINT_CANONICALIZATION_VERSION: &str = "tla-state-slots-v1";
const SHARED_PREPARED_PROGRAM_COMPONENT: &str = "tla_mc_core.prepared_checker_program";
const SHARED_PREPARED_PROGRAM_OWNER: &str = "shared_high_performance_engine";
const SHARED_PREPARED_PROGRAM_TRANSFER_RECORD: &str = "many_frontends_one_engine";
const SHARED_PREPARED_PROGRAM_EXTRACTION_STATUS: &str = "shared-core-ready";
const SHARED_PREPARED_PROGRAM_BLOCKER_STATUS: &str = "tracked-blockers";
const SHARED_PREPARED_PROGRAM_DOWNSTREAM_BLOCKERS: &str =
    "future_importer:awaiting_registered_importer_frontend";
const TLA_PREPARED_PROGRAM_STORAGE_LAYOUT_IDENTITY: &str = "tla_state_slots.storage_layout.v1";
const TLA_PREPARED_PROGRAM_TRANSITION_DESCRIPTOR_IDENTITY: &str =
    "tla_action.transition_descriptor.v1";
const TLA_PREPARED_PROGRAM_PROPERTY_DESCRIPTOR_IDENTITY: &str =
    "tla_property_obligation.descriptor.v1";
const TLA_PREPARED_PROGRAM_VALIDATION_PLAN_IDENTITY: &str = "tla_eval.validation_plan.v1";
const TLA_PREPARED_PROGRAM_REPLAY_PLAN_IDENTITY: &str = "tla_trace.replay_plan.v1";
const TLA_SHARED_PREPARED_PROGRAM_COMPATIBLE_FRONTENDS: &str =
    "tla_plus,quint,mcc_petri,aiger,btor2,vmt_transition_system,ay_analytical,witness_replay";
const TLA_SHARED_PREPARED_PROGRAM_DOWNSTREAM_BENEFICIARIES: &str =
    "quint,mcc_petri,aiger,btor2,vmt_transition_system,ay_analytical,witness_replay";
const QUINT_SHARED_ADOPTION_ACCEPTANCE_TEST: &str =
    "cargo test -p tla-check --lib tla_prepared_program_evidence_rows_publish_quint_shared_adoption";

/// Source family for a TLA prepared-program descriptor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TlaPreparedProgramSource {
    /// Native TLA+ source.
    Tla,
    /// TLA AST translated from Quint JSON IR.
    Quint,
}

impl TlaPreparedProgramSource {
    fn payload_kind(self) -> PreparedProgramPayloadKind {
        match self {
            Self::Tla => PreparedProgramPayloadKind::Tla,
            Self::Quint => PreparedProgramPayloadKind::Quint,
        }
    }

    fn lowering_path(self) -> &'static str {
        match self {
            Self::Tla => "tla_source_to_tla_ast",
            Self::Quint => "quint_json_ir_to_tla_ast",
        }
    }

    fn lowered_payload_kind(self) -> &'static str {
        "tla_ast"
    }

    fn lowered_payload_canonicalization_version(self) -> &'static str {
        match self {
            Self::Tla => TLA_LOWERED_PAYLOAD_CANONICALIZATION_VERSION,
            Self::Quint => QUINT_LOWERED_PAYLOAD_CANONICALIZATION_VERSION,
        }
    }

    fn first_beneficiary(self) -> &'static str {
        match self {
            Self::Tla => "tla_plus",
            Self::Quint => "quint",
        }
    }

    fn second_beneficiary(self) -> &'static str {
        match self {
            Self::Tla => "quint",
            Self::Quint => "tla_plus",
        }
    }

    fn frontend_family_code(self) -> &'static str {
        match self {
            Self::Tla => "tla_plus",
            Self::Quint => "quint",
        }
    }
}

/// TLA-specific wrapper around the shared prepared checker program descriptor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct TlaPreparedProgram {
    program: PreparedCheckerProgram,
    source: TlaPreparedProgramSource,
}

impl TlaPreparedProgram {
    /// Build a descriptor from already-resolved checker preparation facts.
    ///
    /// `detected_action_names` should be the same stable action names already
    /// detected from the resolved NEXT body. They are sorted for deterministic
    /// evidence rows. When it is empty, the resolved NEXT name is recorded as
    /// the single transition descriptor.
    #[must_use]
    pub(crate) fn from_config(
        identity: impl Into<String>,
        source: TlaPreparedProgramSource,
        config: &Config,
        resolved_next_name: Option<&str>,
        detected_action_names: &[String],
    ) -> Self {
        let identity = identity.into();
        let frontend_payload_identity =
            descriptor_identity("frontend_payload", source.payload_kind().code(), &identity);
        let artifact_identity =
            descriptor_identity("prepared_program", source.payload_kind().code(), &identity);
        let dedup = tla_state_dedup_identity();
        let storage_policy_identity = dedup.storage_policy_identity();
        let fingerprint = dedup.fingerprint.clone();
        let fingerprint_fields = fingerprint.identity_fields();
        let fingerprint_policy_identity = fingerprint_fields
            .fingerprint_policy_identity
            .clone()
            .unwrap_or_default();
        let fingerprint_identity = fingerprint_fields
            .fingerprint_identity
            .clone()
            .unwrap_or_default();

        let mut program = PreparedCheckerProgram::new(
            identity,
            source.payload_kind(),
            PreparedStorageKind::TlaStateSlots,
        )
        .with_frontend_payload_identity(frontend_payload_identity)
        .with_artifact_identity(artifact_identity)
        .with_storage_policy_identity(storage_policy_identity)
        .with_fingerprint_policy_identity(fingerprint_policy_identity.clone())
        .with_fingerprint_identity(fingerprint_identity.clone())
        .with_fingerprint(dedup.prepared_fingerprint_descriptor())
        .add_canonical_identity(PreparedCanonicalIdentityDescriptor::new(
            "prepared_program",
            PreparedCanonicalIdentityKind::PreparedProgram,
            PREPARED_PROGRAM_CANONICALIZATION_VERSION,
        ))
        .add_canonical_identity(PreparedCanonicalIdentityDescriptor::new(
            "lowered_payload",
            PreparedCanonicalIdentityKind::FrontendPayload,
            source.lowered_payload_canonicalization_version(),
        ))
        .add_frontend_extension(
            PreparedFrontendExtensionDescriptor::new(
                "tla-trace-replay",
                PreparedFrontendExtensionKind::WitnessReplay,
                ProblemKind::Safety,
            )
            .with_artifact_identity(TLA_PREPARED_PROGRAM_REPLAY_PLAN_IDENTITY)
            .with_storage_policy_identity(TLA_PREPARED_PROGRAM_STORAGE_LAYOUT_IDENTITY)
            .with_fingerprint_policy_identity(fingerprint_policy_identity.clone())
            .with_fingerprint_identity(fingerprint_identity.clone()),
        )
        .add_validation_plan(
            PreparedValidationPlanDescriptor::new(
                "tla-eval-selftest",
                PreparedValidationKind::Selftest,
                ProblemKind::Safety,
            )
            .with_artifact_identity(TLA_PREPARED_PROGRAM_VALIDATION_PLAN_IDENTITY)
            .with_fingerprint_policy_identity(fingerprint_policy_identity.clone())
            .with_fingerprint_identity(fingerprint_identity.clone()),
        )
        .add_validation_plan(
            PreparedValidationPlanDescriptor::new(
                "tla-trace-replay",
                PreparedValidationKind::TraceReplay,
                ProblemKind::Safety,
            )
            .with_artifact_identity(TLA_PREPARED_PROGRAM_REPLAY_PLAN_IDENTITY)
            .with_fingerprint_policy_identity(fingerprint_policy_identity)
            .with_fingerprint_identity(fingerprint_identity),
        );

        if detected_action_names.is_empty() {
            if let Some(next_name) = resolved_next_name.or(config.next.as_deref()) {
                program = program.add_transition(next_name, PreparedTransitionKind::TlaAction);
            }
        } else {
            for action_name in sorted_unique_names(detected_action_names) {
                program = program.add_transition(action_name, PreparedTransitionKind::TlaAction);
            }
        }

        program = add_named_properties(
            program,
            "invariant",
            &config.invariants,
            PreparedPropertyKind::Invariant,
        );
        program = add_named_properties(
            program,
            "trace_invariant",
            &config.trace_invariants,
            PreparedPropertyKind::Invariant,
        );
        program = add_named_properties(
            program,
            "state_constraint",
            &config.constraints,
            PreparedPropertyKind::StateConstraint,
        );
        program = add_named_properties(
            program,
            "action_constraint",
            &config.action_constraints,
            PreparedPropertyKind::ProofObligation,
        );
        program = add_named_properties(
            program,
            "property",
            &config.properties,
            PreparedPropertyKind::Ltl,
        );

        if config.check_deadlock {
            program = program.add_property("deadlock", PreparedPropertyKind::Deadlock);
        }

        Self { program, source }
    }

    /// Borrow the frontend-neutral descriptor.
    #[must_use]
    pub(crate) fn as_core_program(&self) -> &PreparedCheckerProgram {
        &self.program
    }

    /// Consume the wrapper and return the frontend-neutral descriptor.
    #[must_use]
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn into_core_program(self) -> PreparedCheckerProgram {
        self.program
    }

    #[must_use]
    pub(crate) fn origin_frontend(&self) -> &'static str {
        self.source.frontend_family_code()
    }

    #[must_use]
    pub(crate) fn shared_engine_component(&self) -> &'static str {
        SHARED_PREPARED_PROGRAM_COMPONENT
    }

    #[must_use]
    pub(crate) fn transfer_record(&self) -> &'static str {
        SHARED_PREPARED_PROGRAM_TRANSFER_RECORD
    }

    #[must_use]
    pub(crate) fn first_beneficiary(&self) -> &'static str {
        self.source.first_beneficiary()
    }

    #[must_use]
    pub(crate) fn second_beneficiary(&self) -> &'static str {
        self.source.second_beneficiary()
    }

    #[must_use]
    pub(crate) fn compatible_frontend_families(&self) -> &'static str {
        TLA_SHARED_PREPARED_PROGRAM_COMPATIBLE_FRONTENDS
    }

    #[must_use]
    pub(crate) fn downstream_beneficiaries(&self) -> &'static str {
        TLA_SHARED_PREPARED_PROGRAM_DOWNSTREAM_BENEFICIARIES
    }

    #[must_use]
    pub(crate) fn extraction_status(&self) -> &'static str {
        SHARED_PREPARED_PROGRAM_EXTRACTION_STATUS
    }

    #[must_use]
    pub(crate) fn blocker_status(&self) -> &'static str {
        SHARED_PREPARED_PROGRAM_BLOCKER_STATUS
    }

    #[must_use]
    pub(crate) fn downstream_blockers(&self) -> &'static str {
        SHARED_PREPARED_PROGRAM_DOWNSTREAM_BLOCKERS
    }

    /// Build setup/runtime evidence for any prepared-program lane.
    ///
    /// This keeps TLA+/Quint source semantics at the adapter boundary while
    /// routing adoption metadata and canonical identities through the shared
    /// prepared-program descriptor.
    #[must_use]
    pub(crate) fn setup_trace_for_lane(
        &self,
        lane: SetupTraceLaneKind,
        candidate_key: impl Into<String>,
        validation_status: SetupTraceValidationStatus,
    ) -> SetupTrace {
        let candidate_key = candidate_key.into();
        let candidate_lane = self.program.candidate_lanes.iter().find(|candidate| {
            candidate.lane == lane
                && candidate.candidate_key.as_deref() == Some(candidate_key.as_str())
        });
        let identities = candidate_lane
            .map(|candidate| {
                self.program
                    .effective_candidate_lane_identity_fields(candidate)
            })
            .unwrap_or_else(|| self.program.effective_identity_fields());

        SetupTrace::new(self.program.source_kind)
            .with_lane(lane)
            .with_candidate_key(candidate_key)
            .with_identity_fields(identities)
            .with_source_identity(self.program.identity.clone())
            .with_origin_frontend(self.origin_frontend())
            .with_shared_engine_component(self.shared_engine_component())
            .with_first_beneficiary(self.first_beneficiary())
            .with_second_beneficiary(self.second_beneficiary())
            .with_compatible_frontend_families(self.compatible_frontend_families().split(','))
            .with_shared_engine_extraction_status(self.extraction_status())
            .with_shared_engine_blocker_status(self.blocker_status())
            .with_validation_status(validation_status)
    }

    /// Render runtime evidence rows for a prepared-program lane.
    ///
    /// Descriptor rows remain non-executing declarations; setup rows carry the
    /// runtime phase and validation status for the selected shared lane.
    #[must_use]
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn runtime_evidence_rows_for_lane(
        &self,
        lane: SetupTraceLaneKind,
        candidate_key: impl Into<String>,
        phase: tla_mc_core::SetupTracePhase,
        nanos: u64,
        validation_status: SetupTraceValidationStatus,
    ) -> Vec<String> {
        let mut rows = self.describe_evidence_rows();
        let mut trace = self.setup_trace_for_lane(lane, candidate_key, validation_status);
        trace.record_duration(phase, std::time::Duration::from_nanos(nanos));
        rows.extend(trace.render_evidence_rows("TY"));
        rows
    }

    /// Add a shared-engine candidate lane without claiming it has executed.
    #[must_use]
    pub(crate) fn with_candidate_lane(
        mut self,
        id: impl Into<String>,
        lane: SetupTraceLaneKind,
        candidate_key: impl Into<String>,
    ) -> Self {
        let id = id.into();
        let candidate_identity = descriptor_identity("candidate_lane", lane.code(), &id);
        let lane_identity = descriptor_identity("lane", lane.code(), &self.program.identity);
        self.program = self.program.add_candidate_lane(
            PreparedCandidateLaneDescriptor::new(id, lane)
                .with_candidate_key(candidate_key)
                .with_candidate_identity(candidate_identity)
                .with_lane_identity(lane_identity),
        );
        self
    }

    /// Attach runtime artifact identity fields to a declared candidate lane.
    #[must_use]
    pub(crate) fn with_candidate_lane_artifact_identity_fields(
        mut self,
        lane: SetupTraceLaneKind,
        candidate_key: &str,
        cache_key: Option<&str>,
        batch_artifact_identity: Option<&str>,
    ) -> Self {
        for candidate in &mut self.program.candidate_lanes {
            if candidate.lane != lane || candidate.candidate_key.as_deref() != Some(candidate_key) {
                continue;
            }
            let mut identities = candidate.identities.clone();
            if let Some(cache_key) = cache_key {
                identities = identities.with_cache_key(cache_key);
            }
            if let Some(batch_artifact_identity) = batch_artifact_identity {
                identities = identities.with_batch_artifact_identity(batch_artifact_identity);
            }
            candidate.identities = identities;
        }
        self
    }

    /// Render the shared evidence-row summary without implying execution.
    #[must_use]
    pub(crate) fn describe(&self) -> String {
        self.program.render_evidence_row("TY")
    }

    /// Render frontend-neutral evidence rows for the prepared descriptor only.
    ///
    /// These rows declare source identity, action/property descriptors, and
    /// fingerprint/artifact placeholders. They do not claim that trust-codegen or any
    /// other engine has executed the program.
    #[must_use]
    pub(crate) fn describe_evidence_rows(&self) -> Vec<String> {
        let mut rows = Vec::new();
        rows.push(self.describe());
        rows.push(self.shared_bridge_evidence_row());
        rows.push(format!(
            "TY prepared_source_identity core_contract=prepared_checker_program identity={} source_kind={} frontend_kind={} payload_kind={} storage_kind={} lowering_path={} lowered_payload_kind={} descriptor_status=declared execution_status=not_executed",
            evidence_value(&self.program.identity),
            self.program.source_kind.code(),
            self.program.source_kind.code(),
            self.program.payload_kind.code(),
            self.program.storage_kind.code(),
            self.source.lowering_path(),
            self.source.lowered_payload_kind(),
        ));
        if self.source == TlaPreparedProgramSource::Quint {
            let adoption = quint_shared_prepared_program_adoption_evidence();
            debug_assert!(adoption.validate().is_ok());
            rows.push(adoption.render_evidence_row("TY"));
        }

        for transition in &self.program.transitions {
            rows.push(format!(
                "TY prepared_transition_descriptor core_contract=prepared_checker_program source_identity={} source_kind={} frontend_kind={} id={} kind={} payload_kind={} storage_kind={} descriptor_status=declared execution_status=not_executed",
                evidence_value(&self.program.identity),
                self.program.source_kind.code(),
                self.program.source_kind.code(),
                evidence_value(&transition.id),
                transition.kind.code(),
                self.program.payload_kind.code(),
                self.program.storage_kind.code(),
            ));
        }

        for property in &self.program.properties {
            rows.push(format!(
                "TY prepared_property_descriptor core_contract=prepared_checker_program source_identity={} source_kind={} frontend_kind={} id={} kind={} payload_kind={} storage_kind={} descriptor_status=declared execution_status=not_executed",
                evidence_value(&self.program.identity),
                self.program.source_kind.code(),
                self.program.source_kind.code(),
                evidence_value(&property.id),
                property.kind.code(),
                self.program.payload_kind.code(),
                self.program.storage_kind.code(),
            ));
        }

        if let Some(fingerprint) = &self.program.fingerprint {
            let shared_dedup = tla_state_dedup_identity();
            let shared_fingerprint = &shared_dedup.fingerprint;
            rows.push(shared_fingerprint.render_evidence_row("TY", self.program.source_kind));
            rows.push(
                shared_fingerprint.render_validation_evidence_row("TY", self.program.source_kind),
            );
            rows.push(shared_dedup.render_evidence_row("TY", self.program.source_kind));
            rows.push(shared_dedup.render_validation_evidence_row("TY", self.program.source_kind));
            rows.push(format!(
                "TY prepared_fingerprint_descriptor core_contract=prepared_checker_program source_identity={} source_kind={} frontend_kind={} id={} scheme={} canonicalization_version={} descriptor_status=declared value_status=not_computed execution_status=not_executed",
                evidence_value(&self.program.identity),
                self.program.source_kind.code(),
                self.program.source_kind.code(),
                evidence_value(&fingerprint.id),
                fingerprint.scheme.code(),
                evidence_value(&fingerprint.canonicalization_version),
            ));
        }

        for identity in &self.program.canonical_identities {
            rows.push(format!(
                "TY prepared_artifact_placeholder core_contract=prepared_checker_program source_identity={} source_kind={} frontend_kind={} id={} kind={} canonicalization_version={} digest_algorithm={} digest_status={} descriptor_status=declared execution_status=not_executed",
                evidence_value(&self.program.identity),
                self.program.source_kind.code(),
                self.program.source_kind.code(),
                evidence_value(&identity.id),
                identity.kind.code(),
                evidence_value(&identity.canonicalization_version),
                evidence_option(identity.digest_algorithm.as_deref()),
                digest_status(identity.digest.as_deref()),
            ));
        }

        rows.extend(self.program.render_frontend_extension_evidence_rows("TY"));
        rows.extend(self.program.render_candidate_lane_evidence_rows("TY"));
        rows.extend(self.program.render_validation_plan_evidence_rows("TY"));
        rows
    }

    fn shared_bridge_evidence_row(&self) -> String {
        let identities = self.program.effective_identity_fields();
        format!(
            "TY prepared_program_shared_bridge core_contract=prepared_checker_program prepared_program_identity={} source_kind={} frontend_kind={} origin_frontend={} shared_engine_component={} transfer_record={} payload_kind={} storage_kind={} storage_layout_identity={} transition_descriptor_identity={} property_descriptor_identity={} validation_plan_identity={} replay_plan_identity={} frontend_payload_identity={} artifact_identity={} storage_policy_identity={} fingerprint_policy_identity={} fingerprint_identity={} first_beneficiary={} second_beneficiary={} downstream_beneficiaries={} compatible_frontend_families={} downstream_blockers={} extraction_status={} blocker_status={} validation_status=accepted descriptor_status=declared execution_status=not_executed",
            evidence_value(&self.program.identity),
            self.program.source_kind.code(),
            self.program.source_kind.code(),
            self.origin_frontend(),
            self.shared_engine_component(),
            self.transfer_record(),
            self.program.payload_kind.code(),
            self.program.storage_kind.code(),
            TLA_PREPARED_PROGRAM_STORAGE_LAYOUT_IDENTITY,
            TLA_PREPARED_PROGRAM_TRANSITION_DESCRIPTOR_IDENTITY,
            TLA_PREPARED_PROGRAM_PROPERTY_DESCRIPTOR_IDENTITY,
            TLA_PREPARED_PROGRAM_VALIDATION_PLAN_IDENTITY,
            TLA_PREPARED_PROGRAM_REPLAY_PLAN_IDENTITY,
            evidence_option(identities.frontend_payload_identity.as_deref()),
            evidence_option(identities.artifact_identity.as_deref()),
            evidence_option(identities.storage_policy_identity.as_deref()),
            evidence_option(identities.fingerprint_policy_identity.as_deref()),
            evidence_option(identities.fingerprint_identity.as_deref()),
            self.first_beneficiary(),
            self.second_beneficiary(),
            self.downstream_beneficiaries(),
            self.compatible_frontend_families(),
            self.downstream_blockers(),
            self.extraction_status(),
            self.blocker_status(),
        )
    }
}

fn quint_shared_prepared_program_adoption_evidence() -> SharedEngineAdoptionEvidence {
    SharedEngineAdoptionEvidence::new(
        "quint",
        SHARED_PREPARED_PROGRAM_COMPONENT,
        "quint",
        "tla_plus",
        SHARED_PREPARED_PROGRAM_EXTRACTION_STATUS,
        SHARED_PREPARED_PROGRAM_OWNER,
        QUINT_SHARED_ADOPTION_ACCEPTANCE_TEST,
    )
    .with_frontend_family_contract(
        SharedEngineAdoptionLevel::Level3,
        [
            SharedEngineFrontendFamily::TlaPlus,
            SharedEngineFrontendFamily::Quint,
            SharedEngineFrontendFamily::MccPetri,
            SharedEngineFrontendFamily::Aiger,
            SharedEngineFrontendFamily::Btor2,
            SharedEngineFrontendFamily::VmtTransitionSystem,
            SharedEngineFrontendFamily::AYAnalytical,
            SharedEngineFrontendFamily::WitnessReplay,
        ],
        [SharedEngineAdoptionFamilyBlocker::new(
            SharedEngineFrontendFamily::FutureImporter,
            "awaiting registered importer frontend",
        )],
    )
    .with_generic_prerequisite("quint_json_ir_to_tla_ast")
    .with_generic_prerequisite("prepared_checker_program_descriptor")
    .with_generic_prerequisite("tla_state_slots_storage_identity")
}

fn tla_state_fingerprint_identity() -> SharedFingerprintIdentity {
    SharedFingerprintIdentity::new(
        "state_fingerprint",
        SharedFingerprintAlgorithm::TlaFingerprint64,
        SharedFingerprintValueKind::State,
        TLA_STATE_FINGERPRINT_CANONICALIZATION_VERSION,
        "tla_state_slots",
        64,
    )
    .with_canonical_domain(
        "tla_state_slots",
        TLA_STATE_FINGERPRINT_CANONICALIZATION_VERSION,
    )
}

fn tla_state_dedup_identity() -> SharedDedupIdentity {
    SharedDedupIdentity::new(
        "state_space_dedup",
        tla_state_fingerprint_identity(),
        SharedDedupScope::StateSpace,
        SharedDedupStorageKind::InMemory,
        SetupTraceLaneKind::ExplicitState,
    )
    .with_collision_policy(SharedCollisionPolicy::RejectOnCollision)
    .with_storage_config_identity("tla-state-slots-fingerprint-set-v1")
}

fn add_named_properties(
    mut program: PreparedCheckerProgram,
    prefix: &str,
    names: &[String],
    kind: PreparedPropertyKind,
) -> PreparedCheckerProgram {
    for name in sorted_unique_names(names) {
        program = program.add_property(descriptor_id(prefix, name), kind);
    }
    program
}

fn sorted_unique_names(names: &[String]) -> Vec<&str> {
    let mut names = names.iter().map(String::as_str).collect::<Vec<_>>();
    names.sort_unstable();
    names.dedup();
    names
}

fn descriptor_id(prefix: &str, name: &str) -> String {
    if name.is_empty() {
        prefix.to_string()
    } else {
        format!("{prefix}:{name}")
    }
}

fn descriptor_identity(kind: &str, family: &str, identity: &str) -> String {
    format!("{kind}:{family}:{}", evidence_value(identity))
}

fn evidence_value(value: &str) -> String {
    if value.is_empty() {
        "none".to_string()
    } else {
        value.replace(char::is_whitespace, "_")
    }
}

fn evidence_option(value: Option<&str>) -> String {
    value
        .map(evidence_value)
        .unwrap_or_else(|| "none".to_string())
}

fn digest_status(digest: Option<&str>) -> &'static str {
    if digest.is_some() {
        "available"
    } else {
        "not_computed"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tla_mc_core::PreparedFingerprintScheme;

    #[test]
    fn tla_prepared_program_describes_configured_descriptors() {
        let config = Config {
            next: Some("Next".to_string()),
            invariants: vec!["TypeOK".to_string(), "BoundsOK".to_string()],
            trace_invariants: vec!["TraceOK".to_string()],
            constraints: vec!["StateConstraint".to_string()],
            action_constraints: vec!["ActionConstraint".to_string()],
            properties: vec!["EventuallyDone".to_string()],
            ..Default::default()
        };
        let actions = vec!["Send".to_string(), "Recv".to_string(), "Send".to_string()];

        let prepared = TlaPreparedProgram::from_config(
            "Spec.cfg",
            TlaPreparedProgramSource::Tla,
            &config,
            Some("ResolvedNext"),
            &actions,
        );
        let program = prepared.as_core_program();

        assert_eq!(program.payload_kind, PreparedProgramPayloadKind::Tla);
        assert_eq!(program.storage_kind, PreparedStorageKind::TlaStateSlots);
        assert_eq!(program.transitions.len(), 2);
        assert_eq!(program.transitions[0].id, "Recv");
        assert_eq!(program.transitions[1].id, "Send");
        assert_eq!(
            program.transitions[1].kind,
            PreparedTransitionKind::TlaAction
        );
        let invariant_ids = program
            .properties
            .iter()
            .filter(|property| {
                property.kind == PreparedPropertyKind::Invariant
                    && property.id.starts_with("invariant:")
            })
            .map(|property| property.id.as_str())
            .collect::<Vec<_>>();
        assert_eq!(
            invariant_ids,
            vec!["invariant:BoundsOK", "invariant:TypeOK"]
        );
        assert!(program
            .properties
            .iter()
            .any(|property| property.id == "invariant:TypeOK"
                && property.kind == PreparedPropertyKind::Invariant));
        assert!(program
            .properties
            .iter()
            .any(|property| property.id == "trace_invariant:TraceOK"
                && property.kind == PreparedPropertyKind::Invariant));
        assert!(program
            .properties
            .iter()
            .any(|property| property.id == "state_constraint:StateConstraint"
                && property.kind == PreparedPropertyKind::StateConstraint));
        assert!(program.properties.iter().any(|property| property.id
            == "action_constraint:ActionConstraint"
            && property.kind == PreparedPropertyKind::ProofObligation));
        assert!(program
            .properties
            .iter()
            .any(|property| property.id == "property:EventuallyDone"
                && property.kind == PreparedPropertyKind::Ltl));
        assert!(program
            .properties
            .iter()
            .any(|property| property.id == "deadlock"
                && property.kind == PreparedPropertyKind::Deadlock));
        assert_eq!(
            program
                .fingerprint
                .as_ref()
                .map(|fingerprint| fingerprint.scheme),
            Some(PreparedFingerprintScheme::TlaFingerprint64)
        );
        assert_eq!(
            program.identities.fingerprint_policy_identity,
            tla_state_fingerprint_identity()
                .identity_fields()
                .fingerprint_policy_identity
        );
        assert_eq!(
            program.identities.fingerprint_identity,
            tla_state_fingerprint_identity()
                .identity_fields()
                .fingerprint_identity
        );
        let expected_storage_policy_identity = tla_state_dedup_identity().storage_policy_identity();
        assert_eq!(
            program.identities.storage_policy_identity.as_deref(),
            Some(expected_storage_policy_identity.as_str())
        );
        assert_eq!(
            program
                .fingerprint
                .as_ref()
                .and_then(|fingerprint| fingerprint.identities.storage_policy_identity.as_deref()),
            Some(expected_storage_policy_identity.as_str())
        );
        assert_eq!(program.canonical_identities.len(), 2);
        assert!(program
            .canonical_identities
            .iter()
            .any(|identity| identity.id == "prepared_program"
                && identity.kind == PreparedCanonicalIdentityKind::PreparedProgram
                && identity.digest.is_none()));
        assert!(program
            .canonical_identities
            .iter()
            .any(|identity| identity.id == "lowered_payload"
                && identity.kind == PreparedCanonicalIdentityKind::FrontendPayload
                && identity.digest.is_none()));
        assert_eq!(program.frontend_extensions.len(), 1);
        assert_eq!(program.validation_plans.len(), 2);
        assert!(program
            .validation_plans
            .iter()
            .any(|plan| plan.id == "tla-eval-selftest"
                && plan.kind == PreparedValidationKind::Selftest
                && plan.required
                && plan.fail_closed));
        assert!(program
            .validation_plans
            .iter()
            .any(|plan| plan.id == "tla-trace-replay"
                && plan.kind == PreparedValidationKind::TraceReplay
                && plan.required
                && plan.fail_closed));
        assert!(prepared.describe().contains("storage_kind=tla_state_slots"));
        assert!(prepared
            .describe()
            .contains("frontend_payload_identity=frontend_payload:tla:Spec.cfg"));
        assert!(prepared.describe().contains("frontend_extensions=1"));
        assert!(prepared.describe().contains("validation_plans=2"));
        assert!(prepared.describe().contains("validations=2"));

        let rows = prepared.describe_evidence_rows();
        assert!(rows.iter().any(|row| row
            .contains("prepared_program_shared_bridge core_contract=prepared_checker_program")
            && row.contains("prepared_program_identity=Spec.cfg")
            && row.contains("storage_layout_identity=tla_state_slots.storage_layout.v1")
            && row.contains("transition_descriptor_identity=tla_action.transition_descriptor.v1")
            && row.contains("property_descriptor_identity=tla_property_obligation.descriptor.v1")
            && row.contains("validation_plan_identity=tla_eval.validation_plan.v1")
            && row.contains("replay_plan_identity=tla_trace.replay_plan.v1")
            && row.contains("origin_frontend=tla_plus")
            && row.contains("shared_engine_component=tla_mc_core.prepared_checker_program")
            && row.contains("transfer_record=many_frontends_one_engine")
            && row.contains("frontend_payload_identity=frontend_payload:tla:Spec.cfg")
            && row.contains("artifact_identity=prepared_program:tla:Spec.cfg")
            && row.contains("fingerprint_identity=fingerprint:tla_state_slots:canonical_domain_tla_state_slots_tla-state-slots-v1:tla_fingerprint64:state:tla-state-slots-v1")
            && row.contains("first_beneficiary=tla_plus")
            && row.contains("second_beneficiary=quint")
            && row.contains("downstream_beneficiaries=quint,mcc_petri,aiger,btor2,vmt_transition_system,ay_analytical,witness_replay")
            && row.contains("compatible_frontend_families=tla_plus,quint,mcc_petri,aiger,btor2,vmt_transition_system,ay_analytical,witness_replay")
            && row.contains("downstream_blockers=future_importer:awaiting_registered_importer_frontend")
            && row.contains("validation_status=accepted")));
        assert!(rows.iter().any(|row| row
            .contains("prepared_source_identity core_contract=prepared_checker_program")
            && row.contains("frontend_kind=tla")
            && row.contains("lowering_path=tla_source_to_tla_ast")
            && row.contains("execution_status=not_executed")));
        assert!(rows.iter().any(|row| row
            .contains("prepared_transition_descriptor core_contract=prepared_checker_program")
            && row.contains("id=Send")
            && row.contains("frontend_kind=tla")
            && row.contains("kind=tla_action")));
        assert!(rows.iter().any(|row| row
            .contains("prepared_property_descriptor core_contract=prepared_checker_program")
            && row.contains("id=invariant:TypeOK")
            && row.contains("kind=invariant")));
        assert!(rows.iter().any(|row| row
            .contains("shared_fingerprint_identity schema=ty.shared.fingerprint_identity.v1")
            && row.contains("source_kind=tla")
            && row.contains("algorithm=tla_fingerprint64")
            && row.contains("canonical_domain=tla_state_slots")));
        assert!(rows
            .iter()
            .any(|row| row.contains("shared_fingerprint_identity_validation")
                && row.contains("status_code=accepted")
                && row.contains("fail_closed=true")));
        assert!(rows.iter().any(|row| row.contains("shared_dedup_identity")
            && row.contains("source_kind=tla")
            && row.contains("dedup_scope=state_space")
            && row.contains("storage_kind=in_memory")
            && row.contains("collision_policy=reject_on_collision")
            && row.contains("collision_fail_closed=true")
            && row.contains("storage_config_identity=tla-state-slots-fingerprint-set-v1")));
        assert!(rows
            .iter()
            .any(|row| row.contains("shared_dedup_identity_validation")
                && row.contains("status_code=accepted")
                && row.contains("collision_policy=reject_on_collision")
                && row.contains("fail_closed=true")));
        assert!(rows.iter().any(|row| row
            .contains("prepared_fingerprint_descriptor core_contract=prepared_checker_program")
            && row.contains("scheme=tla_fingerprint64")
            && row.contains("value_status=not_computed")));
        assert!(rows.iter().any(|row| row
            .contains("prepared_artifact_placeholder core_contract=prepared_checker_program")
            && row.contains("id=lowered_payload")
            && row.contains("digest_status=not_computed")));
        assert!(rows
            .iter()
            .any(|row| row.contains("prepared_frontend_extension")
                && row.contains("identity=tla-trace-replay")
                && row.contains("extension_kind=witness_replay")
                && row.contains("artifact_identity=tla_trace.replay_plan.v1")));
        assert!(rows
            .iter()
            .any(|row| row.contains("prepared_validation_plan")
                && row.contains("identity=tla-eval-selftest")
                && row.contains("validation_kind=selftest")
                && row.contains("artifact_identity=tla_eval.validation_plan.v1")
                && row.contains("required=true")
                && row.contains("fail_closed=true")));
        assert!(rows
            .iter()
            .any(|row| row.contains("prepared_validation_plan")
                && row.contains("identity=tla-trace-replay")
                && row.contains("validation_kind=trace_replay")
                && row.contains("artifact_identity=tla_trace.replay_plan.v1")
                && row.contains("required=true")
                && row.contains("fail_closed=true")));
    }

    #[test]
    fn tla_prepared_program_can_publish_candidate_lane_evidence() {
        let prepared = TlaPreparedProgram::from_config(
            "Spec.cfg",
            TlaPreparedProgramSource::Tla,
            &Config {
                next: Some("Next".to_string()),
                check_deadlock: false,
                ..Default::default()
            },
            None,
            &["Next".to_string()],
        )
        .with_candidate_lane("trust-cg-native", SetupTraceLaneKind::Native, "trust-cg")
        .with_candidate_lane_artifact_identity_fields(
            SetupTraceLaneKind::Native,
            "trust-cg",
            Some("trust_cg_batch_jit_cache:abc123"),
            Some("trust_cg_batch_jit:shared_high_performance_engine:abc123"),
        );
        let program = prepared.as_core_program();

        assert_eq!(program.candidate_lanes.len(), 1);
        let lane = &program.candidate_lanes[0];
        assert_eq!(lane.lane, SetupTraceLaneKind::Native);
        assert_eq!(lane.candidate_key.as_deref(), Some("trust-cg"));

        let trace_key = program.setup_trace_key_for_candidate_lane(lane);
        assert_eq!(trace_key.frontend.code(), "tla");
        assert_eq!(trace_key.lane, SetupTraceLaneKind::Native);
        assert_eq!(trace_key.candidate_key.as_deref(), Some("trust-cg"));
        assert!(trace_key.identities.candidate_identity.is_some());
        assert!(trace_key.identities.lane_identity.is_some());
        assert_eq!(
            trace_key.identities.cache_key.as_deref(),
            Some("trust_cg_batch_jit_cache:abc123")
        );
        assert_eq!(
            trace_key.identities.batch_artifact_identity.as_deref(),
            Some("trust_cg_batch_jit:shared_high_performance_engine:abc123")
        );
        assert_eq!(
            trace_key.identities.fingerprint_identity,
            program.effective_identity_fields().fingerprint_identity
        );

        let rows = prepared.describe_evidence_rows();
        let expected_fingerprint_identity = format!(
            "fingerprint_identity={}",
            tla_state_fingerprint_identity().fingerprint_identity()
        );
        assert!(rows.iter().any(
            |row| row.contains("prepared_checker_program") && row.contains("candidate_lanes=1")
        ));
        assert!(rows
            .iter()
            .any(|row| row.contains("prepared_candidate_lane")
                && row.contains("lane_kind=native")
                && row.contains("candidate_key=trust-cg")
                && row.contains("candidate_identity=candidate_lane:native:trust-cg-native")
                && row.contains("lane_identity=lane:native:Spec.cfg")
                && row.contains("cache_key=trust_cg_batch_jit_cache:abc123")
                && row.contains(
                    "batch_artifact_identity=trust_cg_batch_jit:shared_high_performance_engine:abc123"
                )
                && row.contains(&expected_fingerprint_identity)));
    }

    #[test]
    fn tla_prepared_program_falls_back_to_resolved_next_and_supports_quint() {
        let config = Config {
            next: Some("NextAlias".to_string()),
            check_deadlock: false,
            ..Default::default()
        };

        let program = TlaPreparedProgram::from_config(
            "QuintSpec",
            TlaPreparedProgramSource::Quint,
            &config,
            Some("ResolvedNext"),
            &[],
        )
        .into_core_program();

        assert_eq!(program.payload_kind, PreparedProgramPayloadKind::Quint);
        assert_eq!(program.transitions.len(), 1);
        assert_eq!(program.transitions[0].id, "ResolvedNext");
        assert!(program.properties.is_empty());
        assert_eq!(program.frontend_extensions.len(), 1);
        assert_eq!(program.validation_plans.len(), 2);
        assert_eq!(
            program.canonical_identities[1].canonicalization_version,
            QUINT_LOWERED_PAYLOAD_CANONICALIZATION_VERSION
        );
    }

    #[test]
    fn tla_prepared_program_evidence_rows_keep_quint_lowering_explicit() {
        let prepared = TlaPreparedProgram::from_config(
            "Quint Spec",
            TlaPreparedProgramSource::Quint,
            &Config {
                next: Some("step".to_string()),
                invariants: vec!["ok".to_string()],
                check_deadlock: false,
                ..Default::default()
            },
            None,
            &[],
        );

        let rows = prepared.describe_evidence_rows();

        assert!(rows.iter().any(|row| {
            row.contains("prepared_source_identity")
                && row.contains("identity=Quint_Spec")
                && row.contains("source_kind=quint")
                && row.contains("payload_kind=quint")
                && row.contains("lowering_path=quint_json_ir_to_tla_ast")
                && row.contains("lowered_payload_kind=tla_ast")
        }));
        assert!(rows.iter().any(|row| {
            row.contains("prepared_program_shared_bridge")
                && row.contains("prepared_program_identity=Quint_Spec")
                && row.contains("origin_frontend=quint")
                && row.contains("shared_engine_component=tla_mc_core.prepared_checker_program")
                && row.contains("transfer_record=many_frontends_one_engine")
                && row.contains("frontend_payload_identity=frontend_payload:quint:Quint_Spec")
                && row.contains("artifact_identity=prepared_program:quint:Quint_Spec")
                && row.contains("first_beneficiary=quint")
                && row.contains("second_beneficiary=tla_plus")
                && row.contains("compatible_frontend_families=tla_plus,quint,mcc_petri,aiger,btor2,vmt_transition_system,ay_analytical,witness_replay")
                && row.contains("downstream_blockers=future_importer:awaiting_registered_importer_frontend")
                && row.contains("validation_status=accepted")
        }));
        assert!(rows.iter().any(|row| {
            row.contains("prepared_transition_descriptor")
                && row.contains("id=step")
                && row.contains("kind=tla_action")
        }));
        assert!(rows.iter().any(|row| {
            row.contains("prepared_property_descriptor")
                && row.contains("id=invariant:ok")
                && row.contains("kind=invariant")
        }));
    }

    #[test]
    fn tla_prepared_program_setup_trace_adoption_fields_reach_prepared_rows() {
        let prepared = TlaPreparedProgram::from_config(
            "Quint Spec",
            TlaPreparedProgramSource::Quint,
            &Config {
                next: Some("step".to_string()),
                check_deadlock: false,
                ..Default::default()
            },
            None,
            &[],
        );

        let rows = prepared.describe_evidence_rows();
        let row = rows
            .iter()
            .find(|row| row.contains("prepared_program_shared_bridge"))
            .expect("prepared-program bridge row should be emitted");

        assert!(row.contains("origin_frontend=quint"));
        assert!(row.contains("shared_engine_component=tla_mc_core.prepared_checker_program"));
        assert!(row.contains("transfer_record=many_frontends_one_engine"));
        assert!(row.contains("first_beneficiary=quint"));
        assert!(row.contains("second_beneficiary=tla_plus"));
        assert!(row.contains(
            "downstream_beneficiaries=quint,mcc_petri,aiger,btor2,vmt_transition_system,ay_analytical,witness_replay"
        ));
        assert!(row.contains(
            "compatible_frontend_families=tla_plus,quint,mcc_petri,aiger,btor2,vmt_transition_system,ay_analytical,witness_replay"
        ));
        assert!(row
            .contains("downstream_blockers=future_importer:awaiting_registered_importer_frontend"));
        assert!(row.contains("extraction_status=shared-core-ready"));
        assert!(row.contains("blocker_status=tracked-blockers"));
        assert!(row.contains("validation_status=accepted"));
        assert!(row.contains("frontend_payload_identity=frontend_payload:quint:Quint_Spec"));
        assert!(row.contains("artifact_identity=prepared_program:quint:Quint_Spec"));
        assert!(row.contains(
            "storage_policy_identity=dedup_storage:in_memory:state_space:tla-state-slots-fingerprint-set-v1"
        ));
        assert!(row.contains("fingerprint_policy_identity=fingerprint_policy:tla_fingerprint64:state:canonical_domain_tla_state_slots_tla-state-slots-v1:tla-state-slots-v1:64:seedless"));
        assert!(row.contains("fingerprint_identity=fingerprint:tla_state_slots:canonical_domain_tla_state_slots_tla-state-slots-v1:tla_fingerprint64:state:tla-state-slots-v1"));
    }

    #[test]
    fn tla_prepared_program_shared_bridge_uses_audit_canonical_tla_plus_family() {
        let prepared = TlaPreparedProgram::from_config(
            "Spec.cfg",
            TlaPreparedProgramSource::Tla,
            &Config {
                next: Some("Next".to_string()),
                check_deadlock: false,
                ..Default::default()
            },
            Some("Next"),
            &[],
        );

        let rows = prepared.describe_evidence_rows();
        let row = rows
            .iter()
            .find(|row| row.contains("prepared_program_shared_bridge"))
            .expect("prepared-program bridge row should be emitted");

        assert!(row.contains("source_kind=tla"));
        assert!(row.contains("frontend_kind=tla"));
        assert!(row.contains("frontend_payload_identity=frontend_payload:tla:Spec.cfg"));
        assert!(row.contains("origin_frontend=tla_plus"));
        assert!(row.contains("transfer_record=many_frontends_one_engine"));
        assert!(row.contains("first_beneficiary=tla_plus"));
        assert!(row.contains("second_beneficiary=quint"));
        assert!(row.contains(
            "downstream_beneficiaries=quint,mcc_petri,aiger,btor2,vmt_transition_system,ay_analytical,witness_replay"
        ));
        assert!(row.contains(
            "compatible_frontend_families=tla_plus,quint,mcc_petri,aiger,btor2,vmt_transition_system,ay_analytical,witness_replay"
        ));
        assert!(row
            .contains("downstream_blockers=future_importer:awaiting_registered_importer_frontend"));
        for family in [
            "tla_plus",
            "quint",
            "mcc_petri",
            "aiger",
            "btor2",
            "vmt_transition_system",
            "ay_analytical",
            "witness_replay",
        ] {
            assert!(
                row.contains(family),
                "bridge row should advertise {family}: {row}"
            );
        }
    }

    #[test]
    fn tla_prepared_program_shared_bridge_rejects_tla_only_beneficiary_shape() {
        let prepared = TlaPreparedProgram::from_config(
            "Spec.cfg",
            TlaPreparedProgramSource::Tla,
            &Config {
                next: Some("Next".to_string()),
                check_deadlock: false,
                ..Default::default()
            },
            Some("Next"),
            &[],
        );

        let rows = prepared.describe_evidence_rows();
        let row = rows
            .iter()
            .find(|row| row.contains("prepared_program_shared_bridge"))
            .expect("prepared-program bridge row should be emitted");

        assert!(!row.contains("second_beneficiary=tla_plus"));
        assert!(!row.contains("compatible_frontend_families=tla_plus "));
        assert!(!row.contains("compatible_frontend_families=tla_plus descriptor_status"));
        assert!(row.contains("second_beneficiary=quint"));
        assert!(row.contains("downstream_beneficiaries=quint,mcc_petri,aiger,btor2,vmt_transition_system,ay_analytical,witness_replay"));
        assert!(row
            .contains("downstream_blockers=future_importer:awaiting_registered_importer_frontend"));
        assert!(row.contains("blocker_status=tracked-blockers"));
        assert!(!row.contains("compatible_frontend_families=tla_plus,quint "));
        assert!(!row.contains("compatible_frontend_families=tla_plus,quint descriptor_status"));
        assert!(row.contains("mcc_petri"));
        assert!(row.contains("aiger"));
        assert!(row.contains("btor2"));
        assert!(row.contains("witness_replay"));
    }

    #[test]
    fn tla_prepared_program_runtime_rows_share_explicit_state_adoption_metadata() {
        let prepared = TlaPreparedProgram::from_config(
            "Quint Spec",
            TlaPreparedProgramSource::Quint,
            &Config {
                next: Some("step".to_string()),
                check_deadlock: false,
                ..Default::default()
            },
            None,
            &["step".to_string()],
        )
        .with_candidate_lane(
            "explicit-bfs",
            SetupTraceLaneKind::ExplicitState,
            "explicit_state",
        );

        let rows = prepared.runtime_evidence_rows_for_lane(
            SetupTraceLaneKind::ExplicitState,
            "explicit_state",
            tla_mc_core::SetupTracePhase::HotExecution,
            17,
            SetupTraceValidationStatus::Accepted,
        );
        let row = rows
            .iter()
            .find(|row| row.contains("setup_trace") && row.contains("phase=hot_execution"))
            .expect("runtime setup row should be emitted");

        assert!(row.contains("source_kind=quint"));
        assert!(row.contains("lane_kind=explicit_state"));
        assert!(row.contains("candidate_key=explicit_state"));
        assert!(row.contains("candidate_identity=candidate_lane:explicit_state:explicit-bfs"));
        assert!(row.contains("lane_identity=lane:explicit_state:Quint_Spec"));
        assert!(row.contains("origin_frontend=quint"));
        assert!(row.contains("shared_engine_component=tla_mc_core.prepared_checker_program"));
        assert!(row.contains("first_beneficiary=quint"));
        assert!(row.contains("second_beneficiary=tla_plus"));
        assert!(row.contains("compatible_frontend_families=aiger,ay_analytical,btor2,mcc_petri,quint,tla_plus,vmt_transition_system,witness_replay"));
        assert!(row.contains("extraction_status=shared-core-ready"));
        assert!(row.contains("blocker_status=tracked-blockers"));
        assert!(!row.contains("compatible_frontend_families=quint,tla_plus "));
        assert!(row.contains("mcc_petri"));
        assert!(row.contains("aiger"));
        assert!(row.contains("btor2"));
        assert!(row.contains("witness_replay"));
        assert!(row.contains("validation_status=accepted"));
        assert!(row.contains("frontend_payload_identity=frontend_payload:quint:Quint_Spec"));
        assert!(row.contains("artifact_identity=prepared_program:quint:Quint_Spec"));
        assert!(row.contains("nanos=17"));
    }

    #[test]
    fn tla_prepared_program_evidence_rows_publish_quint_shared_adoption() {
        let prepared = TlaPreparedProgram::from_config(
            "Quint Spec",
            TlaPreparedProgramSource::Quint,
            &Config {
                next: Some("step".to_string()),
                check_deadlock: false,
                ..Default::default()
            },
            None,
            &[],
        );

        let rows = prepared.describe_evidence_rows();
        let row = rows
            .iter()
            .find(|row| row.contains("shared_engine_adoption"))
            .expect("Quint prepared-program rows should publish shared adoption evidence");

        assert!(row.starts_with("TY shared_engine_adoption "));
        assert!(row.contains("schema=ty.shared.engine_adoption.v1"));
        assert!(row.contains("origin_frontend=quint"));
        assert!(row.contains("shared_engine_component=tla_mc_core.prepared_checker_program"));
        assert!(row.contains(
            "generic_prerequisites=quint_json_ir_to_tla_ast,prepared_checker_program_descriptor,tla_state_slots_storage_identity"
        ));
        assert!(row.contains("first_beneficiary=quint"));
        assert!(row.contains("second_beneficiary=tla_plus"));
        assert!(row.contains("extraction_status=shared-core-ready"));
        assert!(row.contains("adoption_level=level-3"));
        assert!(row.contains(
            "compatible_frontend_families=tla_plus,quint,mcc_petri,aiger,btor2,vmt_transition_system,ay_analytical,witness_replay"
        ));
        assert!(row.contains(
            "frontend_family_blockers=future_importer:awaiting_registered_importer_frontend"
        ));
        assert!(!row.contains("adoption_not_yet_recorded"));
        assert!(row.contains("owner=shared_high_performance_engine"));
        assert!(row.contains(
            "acceptance_test=cargo_test_-p_tla-check_--lib_tla_prepared_program_evidence_rows_publish_quint_shared_adoption"
        ));
        tla_mc_core::validate_shared_engine_adoption_evidence_row(row).unwrap();

        let tla_prepared = TlaPreparedProgram::from_config(
            "Tla Spec",
            TlaPreparedProgramSource::Tla,
            &Config {
                next: Some("Next".to_string()),
                check_deadlock: false,
                ..Default::default()
            },
            None,
            &[],
        );
        assert!(!tla_prepared
            .describe_evidence_rows()
            .iter()
            .any(|row| row.contains("shared_engine_adoption")));
    }
}
