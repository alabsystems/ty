// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Shared-engine adoption evidence rows.
//!
//! These rows let frontends record that an optimization or checker capability
//! was extracted into a reusable shared-engine component. The evidence includes
//! the adoption ladder level plus a frontend-family map so performance rows can
//! prove many-frontend adoption instead of naming only a first and second
//! beneficiary.

use std::fmt;

use crate::evidence_row::evidence_field;

/// Stable row kind for shared-engine adoption evidence.
pub const SHARED_ENGINE_ADOPTION_ROW_KIND: &str = "shared_engine_adoption";

/// Stable schema label for shared-engine adoption evidence.
pub const SHARED_ENGINE_ADOPTION_SCHEMA: &str = "ty.shared.engine_adoption.v1";

/// Schema version for shared-engine adoption evidence rows.
pub const SHARED_ENGINE_ADOPTION_SCHEMA_VERSION: u32 = 1;

/// Fields every shared-engine adoption row publishes.
pub const SHARED_ENGINE_ADOPTION_REQUIRED_FIELDS: &[&str] = &[
    "schema",
    "schema_version",
    "origin_frontend",
    "shared_engine_component",
    "generic_prerequisites",
    "first_beneficiary",
    "second_beneficiary",
    "extraction_status",
    "adoption_level",
    "compatible_frontend_families",
    "default_compatible_frontend_families",
    "downstream_beneficiary_families",
    "remaining_compatible_frontend_families",
    "active_frontend_families",
    "frontend_family_blockers",
    "blocker_status",
    "owner",
    "acceptance_test",
    "acceptance_evidence",
];

/// Accepted extraction-state values for shared-engine adoption evidence.
pub const SHARED_ENGINE_ADOPTION_EXTRACTION_STATUSES: &[&str] = &[
    "already-shared",
    "shared-core-ready",
    "frontend-local-with-tracked-extraction",
    "shared-core-extracted",
];

/// Accepted stable adoption-ladder codes.
pub const SHARED_ENGINE_ADOPTION_LEVELS: &[&str] =
    &["level-0", "level-1", "level-2", "level-3", "level-4"];

/// Accepted stable blocker-status codes.
pub const SHARED_ENGINE_ADOPTION_BLOCKER_STATUSES: &[&str] = &["no-blockers", "tracked-blockers"];

/// Frontend-family codes tracked by the shared-engine adoption map.
pub const SHARED_ENGINE_ADOPTION_FRONTEND_FAMILIES: &[&str] = &[
    "tla_plus",
    "quint",
    "mcc_petri",
    "aiger",
    "btor2",
    "vmt_transition_system",
    "ay_analytical",
    "witness_replay",
    "future_importer",
];

/// Default blocker used when older call sites have not yet declared a family
/// adoption path. The value is explicit so validators and reports can surface
/// the adoption gap.
pub const SHARED_ENGINE_ADOPTION_DEFAULT_FAMILY_BLOCKER: &str = "adoption_not_yet_recorded";

/// Blocker code used while `future_importer` has no registered importer
/// payload identity, layout/domain mapping, fingerprints, and validation
/// receipts.
pub const SHARED_ENGINE_ADOPTION_FUTURE_IMPORTER_RESERVED_BLOCKER: &str =
    "blocked_reserved_importer_contract";

const SHARED_ENGINE_FRONTEND_FAMILY_VALUES: [SharedEngineFrontendFamily; 9] = [
    SharedEngineFrontendFamily::TlaPlus,
    SharedEngineFrontendFamily::Quint,
    SharedEngineFrontendFamily::MccPetri,
    SharedEngineFrontendFamily::Aiger,
    SharedEngineFrontendFamily::Btor2,
    SharedEngineFrontendFamily::VmtTransitionSystem,
    SharedEngineFrontendFamily::AYAnalytical,
    SharedEngineFrontendFamily::WitnessReplay,
    SharedEngineFrontendFamily::FutureImporter,
];

const SHARED_ENGINE_OPERATIONAL_FRONTEND_FAMILY_VALUES: [SharedEngineFrontendFamily; 8] = [
    SharedEngineFrontendFamily::TlaPlus,
    SharedEngineFrontendFamily::Quint,
    SharedEngineFrontendFamily::MccPetri,
    SharedEngineFrontendFamily::Aiger,
    SharedEngineFrontendFamily::Btor2,
    SharedEngineFrontendFamily::VmtTransitionSystem,
    SharedEngineFrontendFamily::AYAnalytical,
    SharedEngineFrontendFamily::WitnessReplay,
];

/// Shared-engine adoption ladder level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SharedEngineAdoptionLevel {
    /// Level 0: frontend-local work with no shared performance claim.
    Level0,
    /// Level 1: shared descriptors, fingerprints, validation plans, or
    /// capability rows are ready for another frontend to consume.
    Level1,
    /// Level 2: two-family active shared implementation.
    Level2,
    /// Level 3: many-family active across TLA-style, graph/net, and
    /// hardware/symbolic/replay families.
    Level3,
    /// Level 4: default shared-engine path for compatible frontends.
    Level4,
}

impl SharedEngineAdoptionLevel {
    /// Stable evidence code.
    #[must_use]
    pub fn code(self) -> &'static str {
        match self {
            Self::Level0 => "level-0",
            Self::Level1 => "level-1",
            Self::Level2 => "level-2",
            Self::Level3 => "level-3",
            Self::Level4 => "level-4",
        }
    }

    /// Parse a stable adoption-level code.
    #[must_use]
    pub fn from_code(code: &str) -> Option<Self> {
        match code {
            "level-0" => Some(Self::Level0),
            "level-1" => Some(Self::Level1),
            "level-2" => Some(Self::Level2),
            "level-3" => Some(Self::Level3),
            "level-4" => Some(Self::Level4),
            _ => None,
        }
    }

    fn requires_many_family_coverage(self) -> bool {
        matches!(self, Self::Level3 | Self::Level4)
    }
}

impl fmt::Display for SharedEngineAdoptionLevel {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.code())
    }
}

/// Summary of whether non-adopting frontend families have tracked blockers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SharedEngineAdoptionBlockerStatus {
    /// All frontend families covered by this schema are already compatible.
    NoBlockers,
    /// One or more non-compatible frontend families has a named blocker.
    TrackedBlockers,
}

impl SharedEngineAdoptionBlockerStatus {
    /// Stable evidence code.
    #[must_use]
    pub fn code(self) -> &'static str {
        match self {
            Self::NoBlockers => "no-blockers",
            Self::TrackedBlockers => "tracked-blockers",
        }
    }

    /// Parse a stable blocker-status code.
    #[must_use]
    pub fn from_code(code: &str) -> Option<Self> {
        match code {
            "no-blockers" => Some(Self::NoBlockers),
            "tracked-blockers" => Some(Self::TrackedBlockers),
            _ => None,
        }
    }

    fn from_blockers(frontend_family_blockers: &[SharedEngineAdoptionFamilyBlocker]) -> Self {
        if frontend_family_blockers.is_empty() {
            Self::NoBlockers
        } else {
            Self::TrackedBlockers
        }
    }
}

impl fmt::Display for SharedEngineAdoptionBlockerStatus {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.code())
    }
}

/// Frontend family tracked in shared-engine adoption evidence.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SharedEngineFrontendFamily {
    /// TLA+ source plus TLC config.
    TlaPlus,
    /// Quint IR lowered through preserved Quint identity.
    Quint,
    /// MCC/Petri PNML and HLPNML frontends.
    MccPetri,
    /// AIGER hardware transition systems.
    Aiger,
    /// BTOR2 bit-vector/register transition systems.
    Btor2,
    /// VMT or exported transition-system interchange.
    VmtTransitionSystem,
    /// AY, symbolic, structural, and analytical proof lanes.
    AYAnalytical,
    /// Witness, trace, proof, certificate, and replay-derived inputs.
    WitnessReplay,
    /// Future importers inheriting the generic shared-engine contract.
    FutureImporter,
}

impl SharedEngineFrontendFamily {
    /// Stable evidence code.
    #[must_use]
    pub fn code(self) -> &'static str {
        match self {
            Self::TlaPlus => "tla_plus",
            Self::Quint => "quint",
            Self::MccPetri => "mcc_petri",
            Self::Aiger => "aiger",
            Self::Btor2 => "btor2",
            Self::VmtTransitionSystem => "vmt_transition_system",
            Self::AYAnalytical => "ay_analytical",
            Self::WitnessReplay => "witness_replay",
            Self::FutureImporter => "future_importer",
        }
    }

    /// Parse a stable frontend-family code.
    #[must_use]
    pub fn from_code(code: &str) -> Option<Self> {
        match code {
            "tla_plus" => Some(Self::TlaPlus),
            "quint" => Some(Self::Quint),
            "mcc_petri" => Some(Self::MccPetri),
            "aiger" => Some(Self::Aiger),
            "btor2" => Some(Self::Btor2),
            "vmt_transition_system" => Some(Self::VmtTransitionSystem),
            "ay_analytical" => Some(Self::AYAnalytical),
            "witness_replay" => Some(Self::WitnessReplay),
            "future_importer" => Some(Self::FutureImporter),
            _ => None,
        }
    }

    /// Stable registry order for all shared-engine frontend families.
    #[must_use]
    pub fn all() -> &'static [Self] {
        all_frontend_families()
    }

    /// Stable registry order for currently operational frontend families.
    ///
    /// `future_importer` is intentionally excluded until a concrete importer
    /// registers payload identity, layout/domain mapping, fingerprints, and
    /// validation receipts.
    #[must_use]
    pub fn operational() -> &'static [Self] {
        operational_frontend_families()
    }

    fn is_tla_style(self) -> bool {
        matches!(self, Self::TlaPlus | Self::Quint)
    }

    fn is_graph_or_net(self) -> bool {
        matches!(self, Self::MccPetri | Self::VmtTransitionSystem)
    }

    fn is_hardware_symbolic_or_replay(self) -> bool {
        matches!(
            self,
            Self::Aiger
                | Self::Btor2
                | Self::VmtTransitionSystem
                | Self::AYAnalytical
                | Self::WitnessReplay
        )
    }
}

impl fmt::Display for SharedEngineFrontendFamily {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.code())
    }
}

/// A machine-checkable blocker for a frontend family that cannot consume the
/// shared component yet.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SharedEngineAdoptionFamilyBlocker {
    /// Frontend family that cannot yet consume the shared component.
    pub frontend_family: SharedEngineFrontendFamily,
    /// Human-readable description of what blocks adoption.
    pub blocker: String,
}

impl SharedEngineAdoptionFamilyBlocker {
    /// Create one frontend-family blocker entry.
    #[must_use]
    pub fn new(frontend_family: SharedEngineFrontendFamily, blocker: impl Into<String>) -> Self {
        Self {
            frontend_family,
            blocker: blocker.into(),
        }
    }

    /// Create the reserved blocker required for `future_importer` until a real
    /// importer payload contract exists.
    #[must_use]
    pub fn future_importer_reserved() -> Self {
        Self::new(
            SharedEngineFrontendFamily::FutureImporter,
            SHARED_ENGINE_ADOPTION_FUTURE_IMPORTER_RESERVED_BLOCKER,
        )
    }

    fn is_future_importer_reserved(&self) -> bool {
        if self.frontend_family != SharedEngineFrontendFamily::FutureImporter {
            return false;
        }
        let blocker = evidence_identity(&self.blocker);
        blocker == SHARED_ENGINE_ADOPTION_FUTURE_IMPORTER_RESERVED_BLOCKER
            || blocker.contains("registered_importer")
            || blocker.contains("reserved_importer")
            || blocker.contains("future_importer_registry")
    }
}

/// One machine-checkable record that a frontend-specific improvement has been
/// extracted into a shared engine component.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SharedEngineAdoptionEvidence {
    /// Frontend family the improvement originally came from.
    pub origin_frontend: String,
    /// Name of the shared engine component the improvement was extracted into.
    pub shared_engine_component: String,
    /// Generic prerequisites that had to hold for the extraction.
    pub generic_prerequisites: Vec<String>,
    /// First frontend family to benefit from the shared component.
    pub first_beneficiary: String,
    /// Second, distinct frontend family to benefit.
    pub second_beneficiary: String,
    /// Free-form extraction status string.
    pub extraction_status: String,
    /// Structured adoption level derived from / overriding the status.
    pub adoption_level: SharedEngineAdoptionLevel,
    /// Families currently able to consume the shared component.
    pub compatible_frontend_families: Vec<SharedEngineFrontendFamily>,
    /// Families inferred as compatible from the origin/beneficiary references.
    pub default_compatible_frontend_families: Vec<SharedEngineFrontendFamily>,
    /// Families that benefit downstream of the direct beneficiaries.
    pub downstream_beneficiary_families: Vec<SharedEngineFrontendFamily>,
    /// Compatible families still awaiting adoption.
    pub remaining_compatible_frontend_families: Vec<SharedEngineFrontendFamily>,
    /// Per-family blockers explaining why a family cannot adopt yet.
    pub frontend_family_blockers: Vec<SharedEngineAdoptionFamilyBlocker>,
    /// Owner responsible for the extraction.
    pub owner: String,
    /// Acceptance test that gates the extraction.
    pub acceptance_test: String,
    /// Collected acceptance evidence entries (starts with the acceptance test).
    pub acceptance_evidence: Vec<String>,
}

impl SharedEngineAdoptionEvidence {
    /// Build an adoption-evidence record, inferring compatible families from the
    /// origin and beneficiary references and a default adoption level from the
    /// extraction status, then filling in any missing family blockers.
    pub fn new(
        origin_frontend: impl Into<String>,
        shared_engine_component: impl Into<String>,
        first_beneficiary: impl Into<String>,
        second_beneficiary: impl Into<String>,
        extraction_status: impl Into<String>,
        owner: impl Into<String>,
        acceptance_test: impl Into<String>,
    ) -> Self {
        let origin_frontend = canonical_frontend_reference(origin_frontend.into());
        let shared_engine_component = shared_engine_component.into();
        let first_beneficiary = canonical_frontend_reference(first_beneficiary.into());
        let second_beneficiary = canonical_frontend_reference(second_beneficiary.into());
        let extraction_status = extraction_status.into();
        let owner = owner.into();
        let acceptance_test = acceptance_test.into();

        let mut compatible_frontend_families = Vec::new();
        add_inferred_frontend_families(&mut compatible_frontend_families, &origin_frontend);
        add_inferred_frontend_families(&mut compatible_frontend_families, &first_beneficiary);
        add_inferred_frontend_families(&mut compatible_frontend_families, &second_beneficiary);
        let default_compatible_frontend_families =
            canonical_frontend_family_vec(compatible_frontend_families.iter().copied());
        let adoption_level = default_adoption_level(&extraction_status);

        let mut evidence = Self {
            origin_frontend,
            shared_engine_component,
            generic_prerequisites: Vec::new(),
            first_beneficiary,
            second_beneficiary,
            extraction_status,
            adoption_level,
            compatible_frontend_families,
            default_compatible_frontend_families,
            downstream_beneficiary_families: Vec::new(),
            remaining_compatible_frontend_families: Vec::new(),
            frontend_family_blockers: Vec::new(),
            owner,
            acceptance_test: acceptance_test.clone(),
            acceptance_evidence: vec![acceptance_test],
        };
        evidence.fill_missing_family_blockers();
        evidence
    }

    /// Append a generic prerequisite (blank input is ignored).
    pub fn with_generic_prerequisite(mut self, prerequisite: impl Into<String>) -> Self {
        let prerequisite = prerequisite.into();
        if !prerequisite.trim().is_empty() {
            self.generic_prerequisites.push(prerequisite);
        }
        self
    }

    /// Override the [`adoption_level`](Self::adoption_level).
    pub fn with_adoption_level(mut self, adoption_level: SharedEngineAdoptionLevel) -> Self {
        self.adoption_level = adoption_level;
        self
    }

    /// Append an acceptance-evidence entry (blank input is ignored).
    pub fn with_acceptance_evidence(mut self, evidence: impl Into<String>) -> Self {
        let evidence = evidence.into();
        if !evidence.trim().is_empty() {
            self.acceptance_evidence.push(evidence);
        }
        self
    }

    /// Replace inferred adoption metadata with the explicit frontend-family
    /// contract published by the producing evidence boundary.
    pub fn with_frontend_family_contract(
        mut self,
        adoption_level: SharedEngineAdoptionLevel,
        compatible_frontend_families: impl IntoIterator<Item = SharedEngineFrontendFamily>,
        frontend_family_blockers: impl IntoIterator<Item = SharedEngineAdoptionFamilyBlocker>,
    ) -> Self {
        self.adoption_level = adoption_level;
        self.compatible_frontend_families =
            canonical_frontend_family_vec(compatible_frontend_families);
        self.default_compatible_frontend_families = expected_default_compatible_frontend_families(
            &self.compatible_frontend_families,
            &self.origin_frontend,
            &self.first_beneficiary,
            &self.second_beneficiary,
        );
        self.downstream_beneficiary_families = Vec::new();
        self.remaining_compatible_frontend_families =
            remaining_families_from_compatible(&self, &self.compatible_frontend_families);
        self.frontend_family_blockers =
            canonical_frontend_family_blockers(frontend_family_blockers);
        self.rebalance_role_family_sets();
        self
    }

    /// Publish the full compatible-family adoption map using frontend-neutral
    /// roles beyond the first and second concrete beneficiaries.
    ///
    /// `downstream_beneficiary_families` names additional frontend families
    /// already benefiting from the shared component. `remaining_compatible_*`
    /// names other compatible families that are not the origin, first, or
    /// second beneficiary in this row. Any non-compatible family not covered by
    /// `frontend_family_blockers` receives the default tracked gap marker.
    pub fn with_downstream_compatible_frontend_families(
        mut self,
        adoption_level: SharedEngineAdoptionLevel,
        downstream_beneficiary_families: impl IntoIterator<Item = SharedEngineFrontendFamily>,
        remaining_compatible_frontend_families: impl IntoIterator<Item = SharedEngineFrontendFamily>,
        frontend_family_blockers: impl IntoIterator<Item = SharedEngineAdoptionFamilyBlocker>,
    ) -> Self {
        self.adoption_level = adoption_level;
        let downstream_input = downstream_beneficiary_families
            .into_iter()
            .collect::<Vec<_>>();
        let remaining_input = remaining_compatible_frontend_families
            .into_iter()
            .collect::<Vec<_>>();
        let mut compatible_frontend_families = inferred_frontend_families_for_evidence(&self);
        let mut downstream_beneficiary_families = Vec::new();
        for frontend_family in downstream_input {
            add_unique_frontend_family(&mut compatible_frontend_families, frontend_family);
            add_unique_frontend_family(&mut downstream_beneficiary_families, frontend_family);
        }
        let mut remaining_compatible_frontend_families = Vec::new();
        for frontend_family in remaining_input {
            add_unique_frontend_family(&mut compatible_frontend_families, frontend_family);
            add_unique_frontend_family(
                &mut remaining_compatible_frontend_families,
                frontend_family,
            );
        }
        self.compatible_frontend_families =
            canonical_frontend_family_vec(compatible_frontend_families);
        self.default_compatible_frontend_families = expected_default_compatible_frontend_families(
            &self.compatible_frontend_families,
            &self.origin_frontend,
            &self.first_beneficiary,
            &self.second_beneficiary,
        );
        self.downstream_beneficiary_families = canonical_frontend_family_vec(
            downstream_beneficiary_families
                .into_iter()
                .filter(|family| self.compatible_frontend_families.contains(family)),
        );
        self.remaining_compatible_frontend_families = canonical_frontend_family_vec(
            remaining_compatible_frontend_families
                .into_iter()
                .filter(|family| self.compatible_frontend_families.contains(family)),
        );
        self.frontend_family_blockers =
            canonical_frontend_family_blockers(frontend_family_blockers);
        self.rebalance_role_family_sets();
        self.fill_missing_family_blockers();
        self
    }

    /// Publish an explicit Level 4 role partition.
    ///
    /// The four role inputs are intentionally separate so release evidence can
    /// distinguish default consumers, additional active/downstream consumers,
    /// remaining compatible families, and blocked compatible families.
    pub fn with_level_four_frontend_family_contract(
        mut self,
        default_compatible_frontend_families: impl IntoIterator<Item = SharedEngineFrontendFamily>,
        downstream_beneficiary_families: impl IntoIterator<Item = SharedEngineFrontendFamily>,
        remaining_compatible_frontend_families: impl IntoIterator<Item = SharedEngineFrontendFamily>,
        frontend_family_blockers: impl IntoIterator<Item = SharedEngineAdoptionFamilyBlocker>,
    ) -> Self {
        self.adoption_level = SharedEngineAdoptionLevel::Level4;

        let mut default_compatible_frontend_families = default_compatible_frontend_families
            .into_iter()
            .collect::<Vec<_>>();
        for frontend_family in inferred_frontend_families_for_evidence(&self) {
            add_unique_frontend_family(&mut default_compatible_frontend_families, frontend_family);
        }
        let default_compatible_frontend_families =
            canonical_frontend_family_vec(default_compatible_frontend_families);
        let downstream_beneficiary_families =
            canonical_frontend_family_vec(downstream_beneficiary_families);
        let remaining_compatible_frontend_families =
            canonical_frontend_family_vec(remaining_compatible_frontend_families);
        let frontend_family_blockers = canonical_frontend_family_blockers(frontend_family_blockers);

        let mut compatible_frontend_families = Vec::new();
        for frontend_family in default_compatible_frontend_families
            .iter()
            .chain(downstream_beneficiary_families.iter())
            .chain(remaining_compatible_frontend_families.iter())
            .copied()
        {
            add_unique_frontend_family(&mut compatible_frontend_families, frontend_family);
        }
        self.compatible_frontend_families =
            canonical_frontend_family_vec(compatible_frontend_families);
        self.default_compatible_frontend_families = default_compatible_frontend_families;
        self.downstream_beneficiary_families = downstream_beneficiary_families;
        self.remaining_compatible_frontend_families = remaining_compatible_frontend_families;
        self.frontend_family_blockers = frontend_family_blockers;
        self
    }

    /// Publish a Level 4 row for current operational frontends while reserving
    /// `future_importer` behind the importer-payload contract.
    pub fn with_level_four_operational_frontend_contract(
        self,
        default_compatible_frontend_families: impl IntoIterator<Item = SharedEngineFrontendFamily>,
        downstream_beneficiary_families: impl IntoIterator<Item = SharedEngineFrontendFamily>,
        remaining_compatible_frontend_families: impl IntoIterator<Item = SharedEngineFrontendFamily>,
    ) -> Self {
        self.with_level_four_frontend_family_contract(
            default_compatible_frontend_families,
            downstream_beneficiary_families,
            remaining_compatible_frontend_families,
            [SharedEngineAdoptionFamilyBlocker::future_importer_reserved()],
        )
    }

    /// Mark one additional frontend family as a downstream beneficiary.
    pub fn with_downstream_beneficiary_family(
        self,
        frontend_family: SharedEngineFrontendFamily,
    ) -> Self {
        let mut evidence = self.with_compatible_frontend_family(frontend_family);
        add_unique_frontend_family(
            &mut evidence.downstream_beneficiary_families,
            frontend_family,
        );
        evidence.rebalance_role_family_sets();
        evidence
    }

    /// Mark one additional frontend family as compatible even when it is not
    /// the origin, first beneficiary, or second beneficiary for this row.
    pub fn with_remaining_compatible_frontend_family(
        self,
        frontend_family: SharedEngineFrontendFamily,
    ) -> Self {
        let mut evidence = self.with_compatible_frontend_family(frontend_family);
        add_unique_frontend_family(
            &mut evidence.remaining_compatible_frontend_families,
            frontend_family,
        );
        evidence.rebalance_role_family_sets();
        evidence
    }

    /// Mark `frontend_family` as compatible, recording it as still-remaining
    /// when it is neither a default nor a downstream consumer, and clearing any
    /// blocker previously recorded for it.
    pub fn with_compatible_frontend_family(
        mut self,
        frontend_family: SharedEngineFrontendFamily,
    ) -> Self {
        add_unique_frontend_family(&mut self.compatible_frontend_families, frontend_family);
        if !default_compatible_frontend_families(&self).contains(&frontend_family)
            && !self
                .downstream_beneficiary_families
                .contains(&frontend_family)
        {
            add_unique_frontend_family(
                &mut self.remaining_compatible_frontend_families,
                frontend_family,
            );
        }
        self.frontend_family_blockers
            .retain(|blocker| blocker.frontend_family != frontend_family);
        self.fill_missing_family_blockers();
        self
    }

    /// Record a blocker for `frontend_family`, removing it from every
    /// compatible/active/remaining set first (a blocked family is not a
    /// consumer). A blank blocker string is ignored.
    pub fn with_frontend_family_blocker(
        mut self,
        frontend_family: SharedEngineFrontendFamily,
        blocker: impl Into<String>,
    ) -> Self {
        let blocker = blocker.into();
        self.compatible_frontend_families
            .retain(|candidate| *candidate != frontend_family);
        self.default_compatible_frontend_families
            .retain(|candidate| *candidate != frontend_family);
        self.downstream_beneficiary_families
            .retain(|candidate| *candidate != frontend_family);
        self.remaining_compatible_frontend_families
            .retain(|candidate| *candidate != frontend_family);
        if !blocker.trim().is_empty() {
            upsert_frontend_family_blocker(
                &mut self.frontend_family_blockers,
                frontend_family,
                evidence_value(&blocker),
            );
        }
        self.fill_missing_family_blockers();
        self
    }

    /// Derived blocker status for this adoption row.
    #[must_use]
    pub fn blocker_status(&self) -> SharedEngineAdoptionBlockerStatus {
        SharedEngineAdoptionBlockerStatus::from_blockers(&self.frontend_family_blockers)
    }

    /// Derived active families: default consumers plus downstream
    /// beneficiaries.
    #[must_use]
    pub fn active_frontend_families(&self) -> Vec<SharedEngineFrontendFamily> {
        active_frontend_families(self)
    }

    /// Render a stable evidence row for adoption consumers, prefixed by `scope`.
    pub fn render_evidence_row(&self, scope: &str) -> String {
        let default_compatible_frontend_families = default_compatible_frontend_families(self);
        let active_frontend_families = active_frontend_families(self);
        let downstream_beneficiary_families = canonical_frontend_family_vec(
            self.downstream_beneficiary_families
                .iter()
                .copied()
                .filter(|family| self.compatible_frontend_families.contains(family)),
        );
        let remaining_compatible_frontend_families = canonical_frontend_family_vec(
            self.remaining_compatible_frontend_families
                .iter()
                .copied()
                .filter(|family| self.compatible_frontend_families.contains(family)),
        );
        format!(
            "{} {} schema={} schema_version={} origin_frontend={} shared_engine_component={} generic_prerequisites={} first_beneficiary={} second_beneficiary={} extraction_status={} adoption_level={} compatible_frontend_families={} active_frontend_families={} default_compatible_frontend_families={} downstream_beneficiary_families={} remaining_compatible_frontend_families={} frontend_family_blockers={} blocker_status={} owner={} acceptance_test={} acceptance_evidence={}",
            scope,
            SHARED_ENGINE_ADOPTION_ROW_KIND,
            SHARED_ENGINE_ADOPTION_SCHEMA,
            SHARED_ENGINE_ADOPTION_SCHEMA_VERSION,
            evidence_value(&self.origin_frontend),
            evidence_value(&self.shared_engine_component),
            evidence_list(&self.generic_prerequisites),
            evidence_value(&self.first_beneficiary),
            evidence_value(&self.second_beneficiary),
            evidence_value(&self.extraction_status),
            self.adoption_level.code(),
            evidence_family_list(&self.compatible_frontend_families),
            evidence_family_list(&active_frontend_families),
            evidence_family_list(&default_compatible_frontend_families),
            evidence_family_list(&downstream_beneficiary_families),
            evidence_family_list(&remaining_compatible_frontend_families),
            evidence_family_blockers(&self.frontend_family_blockers),
            self.blocker_status().code(),
            evidence_value(&self.owner),
            evidence_value(&self.acceptance_test),
            evidence_list(&self.acceptance_evidence),
        )
    }

    /// Validate that this record satisfies the shared-engine adoption contract.
    ///
    /// # Errors
    ///
    /// Returns a [`SharedEngineAdoptionEvidenceError`] describing the first
    /// violated requirement: a blank required field, a non-canonical or
    /// placeholder frontend reference, a second beneficiary not distinct from
    /// the origin or first beneficiary, an unsupported extraction status, an
    /// empty prerequisite/acceptance-evidence list, or an inconsistent
    /// adoption-level / family-role partition.
    pub fn validate(&self) -> Result<(), SharedEngineAdoptionEvidenceError> {
        require_non_empty_struct_field("origin_frontend", &self.origin_frontend)?;
        require_non_empty_struct_field("shared_engine_component", &self.shared_engine_component)?;
        require_non_empty_struct_field("first_beneficiary", &self.first_beneficiary)?;
        require_non_empty_struct_field("second_beneficiary", &self.second_beneficiary)?;
        require_canonical_frontend_reference("origin_frontend", &self.origin_frontend)?;
        require_canonical_frontend_reference("first_beneficiary", &self.first_beneficiary)?;
        require_canonical_frontend_reference("second_beneficiary", &self.second_beneficiary)?;
        require_concrete_beneficiary("first_beneficiary", &self.first_beneficiary)?;
        require_concrete_beneficiary("second_beneficiary", &self.second_beneficiary)?;
        require_distinct_second_beneficiary(&self.origin_frontend, &self.second_beneficiary)?;
        require_distinct_beneficiary_families(&self.first_beneficiary, &self.second_beneficiary)?;
        require_non_empty_struct_field("extraction_status", &self.extraction_status)?;
        require_supported_extraction_status(&self.extraction_status)?;
        require_non_empty_generic_prerequisites(&self.generic_prerequisites)?;
        require_adoption_metadata(
            self.adoption_level,
            &self.compatible_frontend_families,
            &self.frontend_family_blockers,
            self.blocker_status(),
            &self.extraction_status,
            &self.origin_frontend,
            &self.first_beneficiary,
            &self.second_beneficiary,
        )?;
        let default_compatible_frontend_families = default_compatible_frontend_families(self);
        let active_frontend_families = active_frontend_families(self);
        let required_default_frontend_families = expected_default_compatible_frontend_families(
            &self.compatible_frontend_families,
            &self.origin_frontend,
            &self.first_beneficiary,
            &self.second_beneficiary,
        );
        require_frontend_family_role_partition(
            &self.compatible_frontend_families,
            &required_default_frontend_families,
            &active_frontend_families,
            &default_compatible_frontend_families,
            &self.downstream_beneficiary_families,
            &self.remaining_compatible_frontend_families,
            &self.frontend_family_blockers,
            self.adoption_level,
        )?;
        require_non_empty_struct_field("owner", &self.owner)?;
        require_non_empty_struct_field("acceptance_test", &self.acceptance_test)?;
        require_non_empty_acceptance_evidence(&self.acceptance_evidence)?;
        Ok(())
    }

    fn fill_missing_family_blockers(&mut self) {
        let unblocked_frontend_families = unblocked_compatible_frontend_families(self);
        self.frontend_family_blockers.retain(|blocker| {
            !self
                .default_compatible_frontend_families
                .contains(&blocker.frontend_family)
                && !self
                    .downstream_beneficiary_families
                    .contains(&blocker.frontend_family)
                && !self
                    .remaining_compatible_frontend_families
                    .contains(&blocker.frontend_family)
        });
        for frontend_family in all_frontend_families() {
            if unblocked_frontend_families.contains(frontend_family)
                || self
                    .frontend_family_blockers
                    .iter()
                    .any(|blocker| blocker.frontend_family == *frontend_family)
            {
                continue;
            }
            self.frontend_family_blockers
                .push(SharedEngineAdoptionFamilyBlocker::new(
                    *frontend_family,
                    SHARED_ENGINE_ADOPTION_DEFAULT_FAMILY_BLOCKER,
                ));
        }
        let frontend_family_blockers = std::mem::take(&mut self.frontend_family_blockers);
        self.frontend_family_blockers =
            canonical_frontend_family_blockers(frontend_family_blockers);
        self.rebalance_role_family_sets();
    }

    fn rebalance_role_family_sets(&mut self) {
        let default_families = default_compatible_frontend_families(self);
        self.default_compatible_frontend_families = default_families.clone();
        self.downstream_beneficiary_families = canonical_frontend_family_vec(
            self.downstream_beneficiary_families
                .iter()
                .copied()
                .filter(|family| self.compatible_frontend_families.contains(family))
                .filter(|family| !default_families.contains(family)),
        );
        self.remaining_compatible_frontend_families = canonical_frontend_family_vec(
            self.remaining_compatible_frontend_families
                .iter()
                .copied()
                .filter(|family| self.compatible_frontend_families.contains(family))
                .filter(|family| !default_families.contains(family))
                .filter(|family| !self.downstream_beneficiary_families.contains(family)),
        );
    }
}

/// Validation error for shared-engine adoption evidence.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum SharedEngineAdoptionEvidenceError {
    /// The row does not use the shared-engine adoption row kind.
    WrongRowKind,
    /// A required key/value field is absent.
    MissingField(&'static str),
    /// The row uses an unsupported schema.
    UnsupportedSchema(String),
    /// A field value is syntactically invalid for this schema.
    InvalidField {
        /// Field name.
        field: &'static str,
        /// Field value.
        value: String,
    },
}

impl SharedEngineAdoptionEvidenceError {
    /// Stable reason code for orchestration logs.
    pub fn reason_code(&self) -> &'static str {
        match self {
            Self::WrongRowKind => "wrong_shared_engine_adoption_row_kind",
            Self::MissingField(_) => "missing_shared_engine_adoption_field",
            Self::UnsupportedSchema(_) => "unsupported_shared_engine_adoption_schema",
            Self::InvalidField { .. } => "invalid_shared_engine_adoption_field",
        }
    }
}

impl fmt::Display for SharedEngineAdoptionEvidenceError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::WrongRowKind => write!(formatter, "wrong shared engine adoption row kind"),
            Self::MissingField(field) => {
                write!(formatter, "missing shared engine adoption field: {field}")
            }
            Self::UnsupportedSchema(schema) => {
                write!(
                    formatter,
                    "unsupported shared engine adoption schema: {schema}"
                )
            }
            Self::InvalidField { field, value } => {
                write!(
                    formatter,
                    "invalid shared engine adoption field {field}={value}"
                )
            }
        }
    }
}

impl std::error::Error for SharedEngineAdoptionEvidenceError {}

/// Validate one rendered shared-engine adoption evidence row.
pub fn validate_shared_engine_adoption_evidence_row(
    row: &str,
) -> Result<(), SharedEngineAdoptionEvidenceError> {
    let mut tokens = row.split_whitespace();
    if tokens.next().is_none() {
        return Err(SharedEngineAdoptionEvidenceError::WrongRowKind);
    }
    if tokens.next() != Some(SHARED_ENGINE_ADOPTION_ROW_KIND) {
        return Err(SharedEngineAdoptionEvidenceError::WrongRowKind);
    }

    for field in SHARED_ENGINE_ADOPTION_REQUIRED_FIELDS {
        required_field(row, field)?;
    }

    let schema = required_field(row, "schema")?;
    if schema != SHARED_ENGINE_ADOPTION_SCHEMA {
        return Err(SharedEngineAdoptionEvidenceError::UnsupportedSchema(
            schema.to_string(),
        ));
    }

    let schema_version = required_field(row, "schema_version")?;
    if schema_version != SHARED_ENGINE_ADOPTION_SCHEMA_VERSION.to_string() {
        return Err(SharedEngineAdoptionEvidenceError::InvalidField {
            field: "schema_version",
            value: schema_version.to_string(),
        });
    }

    for field in [
        "origin_frontend",
        "shared_engine_component",
        "first_beneficiary",
        "second_beneficiary",
        "extraction_status",
        "owner",
        "acceptance_test",
        "acceptance_evidence",
    ] {
        require_non_empty_row_field(row, field)?;
    }

    let extraction_status = required_field(row, "extraction_status")?;
    require_supported_extraction_status(extraction_status)?;
    require_non_empty_row_generic_prerequisites(row)?;
    require_non_empty_row_acceptance_evidence(row)?;
    require_canonical_frontend_reference(
        "origin_frontend",
        required_field(row, "origin_frontend")?,
    )?;
    require_canonical_frontend_reference(
        "first_beneficiary",
        required_field(row, "first_beneficiary")?,
    )?;
    require_canonical_frontend_reference(
        "second_beneficiary",
        required_field(row, "second_beneficiary")?,
    )?;
    require_distinct_second_beneficiary(
        required_field(row, "origin_frontend")?,
        required_field(row, "second_beneficiary")?,
    )?;
    require_concrete_beneficiary(
        "first_beneficiary",
        required_field(row, "first_beneficiary")?,
    )?;
    require_concrete_beneficiary(
        "second_beneficiary",
        required_field(row, "second_beneficiary")?,
    )?;
    require_distinct_beneficiary_families(
        required_field(row, "first_beneficiary")?,
        required_field(row, "second_beneficiary")?,
    )?;

    let adoption_level = parse_adoption_level(row)?;
    let compatible_frontend_families = parse_compatible_frontend_families(row)?;
    let frontend_family_blockers = parse_frontend_family_blockers(row)?;
    let blocker_status = parse_blocker_status(row)?;
    require_optional_frontend_family_role_fields(
        row,
        adoption_level,
        &compatible_frontend_families,
        &frontend_family_blockers,
        required_field(row, "origin_frontend")?,
        required_field(row, "first_beneficiary")?,
        required_field(row, "second_beneficiary")?,
    )?;
    require_adoption_metadata(
        adoption_level,
        &compatible_frontend_families,
        &frontend_family_blockers,
        blocker_status,
        extraction_status,
        required_field(row, "origin_frontend")?,
        required_field(row, "first_beneficiary")?,
        required_field(row, "second_beneficiary")?,
    )?;

    Ok(())
}

fn parse_blocker_status(
    row: &str,
) -> Result<SharedEngineAdoptionBlockerStatus, SharedEngineAdoptionEvidenceError> {
    let value = required_field(row, "blocker_status")?;
    if value.is_empty() || value == "none" {
        return Err(SharedEngineAdoptionEvidenceError::MissingField(
            "blocker_status",
        ));
    }
    SharedEngineAdoptionBlockerStatus::from_code(value).ok_or_else(|| {
        SharedEngineAdoptionEvidenceError::InvalidField {
            field: "blocker_status",
            value: value.to_string(),
        }
    })
}

fn parse_adoption_level(
    row: &str,
) -> Result<SharedEngineAdoptionLevel, SharedEngineAdoptionEvidenceError> {
    let value = required_field(row, "adoption_level")?;
    if value.is_empty() || value == "none" {
        return Err(SharedEngineAdoptionEvidenceError::MissingField(
            "adoption_level",
        ));
    }
    SharedEngineAdoptionLevel::from_code(value).ok_or_else(|| {
        SharedEngineAdoptionEvidenceError::InvalidField {
            field: "adoption_level",
            value: value.to_string(),
        }
    })
}

fn parse_compatible_frontend_families(
    row: &str,
) -> Result<Vec<SharedEngineFrontendFamily>, SharedEngineAdoptionEvidenceError> {
    let value = required_field(row, "compatible_frontend_families")?;
    parse_frontend_family_list_value("compatible_frontend_families", value, false)
}

fn parse_optional_frontend_family_list(
    row: &str,
    field: &'static str,
) -> Result<Option<Vec<SharedEngineFrontendFamily>>, SharedEngineAdoptionEvidenceError> {
    evidence_field(row, field)
        .map(|value| parse_frontend_family_list_value(field, value, true))
        .transpose()
}

fn parse_frontend_family_list_value(
    field: &'static str,
    value: &str,
    allow_none: bool,
) -> Result<Vec<SharedEngineFrontendFamily>, SharedEngineAdoptionEvidenceError> {
    if value.is_empty() || value == "none" {
        if allow_none {
            return Ok(Vec::new());
        }
        return Err(SharedEngineAdoptionEvidenceError::MissingField(field));
    }

    let mut families = Vec::new();
    for code in value.split(',') {
        if code.is_empty() || code == "none" {
            return Err(SharedEngineAdoptionEvidenceError::InvalidField {
                field,
                value: value.to_string(),
            });
        }
        let family = SharedEngineFrontendFamily::from_code(code).ok_or_else(|| {
            SharedEngineAdoptionEvidenceError::InvalidField {
                field,
                value: code.to_string(),
            }
        })?;
        if families.contains(&family) {
            return Err(SharedEngineAdoptionEvidenceError::InvalidField {
                field,
                value: code.to_string(),
            });
        }
        families.push(family);
    }

    Ok(families)
}

fn require_optional_frontend_family_role_fields(
    row: &str,
    adoption_level: SharedEngineAdoptionLevel,
    compatible_frontend_families: &[SharedEngineFrontendFamily],
    frontend_family_blockers: &[SharedEngineAdoptionFamilyBlocker],
    origin_frontend: &str,
    first_beneficiary: &str,
    second_beneficiary: &str,
) -> Result<(), SharedEngineAdoptionEvidenceError> {
    let active_families = parse_optional_frontend_family_list(row, "active_frontend_families")?;
    let default_families =
        parse_optional_frontend_family_list(row, "default_compatible_frontend_families")?;
    let downstream_families =
        parse_optional_frontend_family_list(row, "downstream_beneficiary_families")?;
    let remaining_families =
        parse_optional_frontend_family_list(row, "remaining_compatible_frontend_families")?;

    if active_families.is_none()
        && default_families.is_none()
        && downstream_families.is_none()
        && remaining_families.is_none()
    {
        return Ok(());
    }

    let Some(active_families) = active_families else {
        return Err(SharedEngineAdoptionEvidenceError::MissingField(
            "active_frontend_families",
        ));
    };
    let Some(default_families) = default_families else {
        return Err(SharedEngineAdoptionEvidenceError::MissingField(
            "default_compatible_frontend_families",
        ));
    };
    let Some(downstream_families) = downstream_families else {
        return Err(SharedEngineAdoptionEvidenceError::MissingField(
            "downstream_beneficiary_families",
        ));
    };
    let Some(remaining_families) = remaining_families else {
        return Err(SharedEngineAdoptionEvidenceError::MissingField(
            "remaining_compatible_frontend_families",
        ));
    };

    let expected_default_families = expected_default_compatible_frontend_families(
        compatible_frontend_families,
        origin_frontend,
        first_beneficiary,
        second_beneficiary,
    );
    require_frontend_family_role_partition(
        compatible_frontend_families,
        &expected_default_families,
        &active_families,
        &default_families,
        &downstream_families,
        &remaining_families,
        frontend_family_blockers,
        adoption_level,
    )
}

fn expected_default_compatible_frontend_families(
    compatible_frontend_families: &[SharedEngineFrontendFamily],
    origin_frontend: &str,
    first_beneficiary: &str,
    second_beneficiary: &str,
) -> Vec<SharedEngineFrontendFamily> {
    let mut families = Vec::new();
    add_inferred_frontend_families(&mut families, origin_frontend);
    add_inferred_frontend_families(&mut families, first_beneficiary);
    add_inferred_frontend_families(&mut families, second_beneficiary);
    canonical_frontend_family_vec(
        families
            .into_iter()
            .filter(|family| compatible_frontend_families.contains(family)),
    )
}

fn require_frontend_family_role_partition(
    compatible_frontend_families: &[SharedEngineFrontendFamily],
    required_default_families: &[SharedEngineFrontendFamily],
    active_families: &[SharedEngineFrontendFamily],
    default_families: &[SharedEngineFrontendFamily],
    downstream_families: &[SharedEngineFrontendFamily],
    remaining_families: &[SharedEngineFrontendFamily],
    frontend_family_blockers: &[SharedEngineAdoptionFamilyBlocker],
    adoption_level: SharedEngineAdoptionLevel,
) -> Result<(), SharedEngineAdoptionEvidenceError> {
    for (field, families) in [
        ("compatible_frontend_families", compatible_frontend_families),
        ("active_frontend_families", active_families),
        ("default_compatible_frontend_families", default_families),
        ("downstream_beneficiary_families", downstream_families),
        ("remaining_compatible_frontend_families", remaining_families),
    ] {
        let canonical = canonical_frontend_family_vec(families.iter().copied());
        if families != canonical {
            return Err(SharedEngineAdoptionEvidenceError::InvalidField {
                field,
                value: evidence_family_list(families),
            });
        }
    }

    for family in required_default_families {
        if !default_families.contains(family) {
            return Err(SharedEngineAdoptionEvidenceError::InvalidField {
                field: "default_compatible_frontend_families",
                value: format!("missing:{}", family.code()),
            });
        }
    }

    let expected_active_families =
        active_frontend_families_from_roles(default_families, downstream_families);
    if active_families != expected_active_families {
        return Err(SharedEngineAdoptionEvidenceError::InvalidField {
            field: "active_frontend_families",
            value: evidence_family_list(active_families),
        });
    }

    for (field, families) in [
        ("active_frontend_families", active_families),
        ("default_compatible_frontend_families", default_families),
        ("downstream_beneficiary_families", downstream_families),
        ("remaining_compatible_frontend_families", remaining_families),
    ] {
        for family in families {
            if !compatible_frontend_families.contains(family) {
                return Err(SharedEngineAdoptionEvidenceError::InvalidField {
                    field,
                    value: family.code().to_string(),
                });
            }
        }
    }

    for family in default_families {
        if !active_families.contains(family) {
            return Err(SharedEngineAdoptionEvidenceError::InvalidField {
                field: "active_frontend_families",
                value: format!("missing:{}", family.code()),
            });
        }
    }
    for family in downstream_families {
        if default_families.contains(family) {
            return Err(SharedEngineAdoptionEvidenceError::InvalidField {
                field: "downstream_beneficiary_families",
                value: family.code().to_string(),
            });
        }
    }
    for family in remaining_families {
        if active_families.contains(family) {
            return Err(SharedEngineAdoptionEvidenceError::InvalidField {
                field: "remaining_compatible_frontend_families",
                value: family.code().to_string(),
            });
        }
    }

    for blocker in frontend_family_blockers {
        let blocked_family = blocker.frontend_family;
        if active_families.contains(&blocked_family) {
            return Err(SharedEngineAdoptionEvidenceError::InvalidField {
                field: "frontend_family_blockers",
                value: format!("active:{}", blocked_family.code()),
            });
        }
        if remaining_families.contains(&blocked_family) {
            return Err(SharedEngineAdoptionEvidenceError::InvalidField {
                field: "frontend_family_blockers",
                value: format!("remaining:{}", blocked_family.code()),
            });
        }
    }

    require_future_importer_reservation(
        adoption_level,
        active_families,
        default_families,
        downstream_families,
        remaining_families,
        frontend_family_blockers,
    )?;

    let mut union = Vec::new();
    for family in active_families
        .iter()
        .chain(remaining_families.iter())
        .copied()
    {
        add_unique_frontend_family(&mut union, family);
    }
    for blocker in frontend_family_blockers
        .iter()
        .filter(|blocker| compatible_frontend_families.contains(&blocker.frontend_family))
    {
        add_unique_frontend_family(&mut union, blocker.frontend_family);
    }
    let union = canonical_frontend_family_vec(union);
    if union != compatible_frontend_families {
        return Err(SharedEngineAdoptionEvidenceError::InvalidField {
            field: "remaining_compatible_frontend_families",
            value: evidence_family_list(&union),
        });
    }

    Ok(())
}

fn require_future_importer_reservation(
    adoption_level: SharedEngineAdoptionLevel,
    active_families: &[SharedEngineFrontendFamily],
    default_families: &[SharedEngineFrontendFamily],
    downstream_families: &[SharedEngineFrontendFamily],
    remaining_families: &[SharedEngineFrontendFamily],
    frontend_family_blockers: &[SharedEngineAdoptionFamilyBlocker],
) -> Result<(), SharedEngineAdoptionEvidenceError> {
    if adoption_level != SharedEngineAdoptionLevel::Level4 {
        return Ok(());
    }

    for (field, families) in [
        ("active_frontend_families", active_families),
        ("default_compatible_frontend_families", default_families),
        ("downstream_beneficiary_families", downstream_families),
        ("remaining_compatible_frontend_families", remaining_families),
    ] {
        if families.contains(&SharedEngineFrontendFamily::FutureImporter) {
            return Err(SharedEngineAdoptionEvidenceError::InvalidField {
                field,
                value: SharedEngineFrontendFamily::FutureImporter
                    .code()
                    .to_string(),
            });
        }
    }

    if frontend_family_blockers
        .iter()
        .any(SharedEngineAdoptionFamilyBlocker::is_future_importer_reserved)
    {
        Ok(())
    } else {
        Err(SharedEngineAdoptionEvidenceError::InvalidField {
            field: "frontend_family_blockers",
            value: "future_importer_reserved".to_string(),
        })
    }
}

fn parse_frontend_family_blockers(
    row: &str,
) -> Result<Vec<SharedEngineAdoptionFamilyBlocker>, SharedEngineAdoptionEvidenceError> {
    let value = required_field(row, "frontend_family_blockers")?;
    if value.is_empty() || value == "none" {
        return Ok(Vec::new());
    }

    let mut blockers = Vec::new();
    for entry in value.split(',') {
        let Some((family_code, blocker)) = entry.split_once(':') else {
            return Err(SharedEngineAdoptionEvidenceError::InvalidField {
                field: "frontend_family_blockers",
                value: entry.to_string(),
            });
        };
        let frontend_family =
            SharedEngineFrontendFamily::from_code(family_code).ok_or_else(|| {
                SharedEngineAdoptionEvidenceError::InvalidField {
                    field: "frontend_family_blockers",
                    value: family_code.to_string(),
                }
            })?;
        if blocker.is_empty() || blocker == "none" {
            return Err(SharedEngineAdoptionEvidenceError::MissingField(
                "frontend_family_blockers",
            ));
        }
        if blockers
            .iter()
            .any(|existing: &SharedEngineAdoptionFamilyBlocker| {
                existing.frontend_family == frontend_family
            })
        {
            return Err(SharedEngineAdoptionEvidenceError::InvalidField {
                field: "frontend_family_blockers",
                value: family_code.to_string(),
            });
        }
        blockers.push(SharedEngineAdoptionFamilyBlocker::new(
            frontend_family,
            blocker,
        ));
    }

    Ok(blockers)
}

fn require_adoption_metadata(
    adoption_level: SharedEngineAdoptionLevel,
    compatible_frontend_families: &[SharedEngineFrontendFamily],
    frontend_family_blockers: &[SharedEngineAdoptionFamilyBlocker],
    blocker_status: SharedEngineAdoptionBlockerStatus,
    extraction_status: &str,
    origin_frontend: &str,
    first_beneficiary: &str,
    second_beneficiary: &str,
) -> Result<(), SharedEngineAdoptionEvidenceError> {
    require_compatible_frontend_families(compatible_frontend_families)?;
    require_frontend_family_blockers(compatible_frontend_families, frontend_family_blockers)?;
    require_tracked_frontend_family_blockers(adoption_level, frontend_family_blockers)?;
    require_blocker_status(blocker_status, frontend_family_blockers)?;
    require_named_frontend_families_are_compatible(
        "origin_frontend",
        origin_frontend,
        compatible_frontend_families,
    )?;
    require_named_frontend_families_are_compatible(
        "first_beneficiary",
        first_beneficiary,
        compatible_frontend_families,
    )?;
    require_named_frontend_families_are_compatible(
        "second_beneficiary",
        second_beneficiary,
        compatible_frontend_families,
    )?;
    require_adoption_level_requirements(
        adoption_level,
        compatible_frontend_families,
        extraction_status,
    )
}

fn require_blocker_status(
    blocker_status: SharedEngineAdoptionBlockerStatus,
    frontend_family_blockers: &[SharedEngineAdoptionFamilyBlocker],
) -> Result<(), SharedEngineAdoptionEvidenceError> {
    let expected = SharedEngineAdoptionBlockerStatus::from_blockers(frontend_family_blockers);
    if blocker_status == expected {
        Ok(())
    } else {
        Err(SharedEngineAdoptionEvidenceError::InvalidField {
            field: "blocker_status",
            value: blocker_status.code().to_string(),
        })
    }
}

fn require_adoption_level_requirements(
    adoption_level: SharedEngineAdoptionLevel,
    compatible_frontend_families: &[SharedEngineFrontendFamily],
    extraction_status: &str,
) -> Result<(), SharedEngineAdoptionEvidenceError> {
    if extraction_status == "frontend-local-with-tracked-extraction"
        && adoption_level != SharedEngineAdoptionLevel::Level0
    {
        return Err(SharedEngineAdoptionEvidenceError::InvalidField {
            field: "extraction_status",
            value: extraction_status.to_string(),
        });
    }

    if adoption_level == SharedEngineAdoptionLevel::Level0
        && extraction_status != "frontend-local-with-tracked-extraction"
    {
        return Err(SharedEngineAdoptionEvidenceError::InvalidField {
            field: "adoption_level",
            value: adoption_level.code().to_string(),
        });
    }

    if matches!(
        adoption_level,
        SharedEngineAdoptionLevel::Level1
            | SharedEngineAdoptionLevel::Level2
            | SharedEngineAdoptionLevel::Level3
            | SharedEngineAdoptionLevel::Level4
    ) && compatible_frontend_families.len() < 2
    {
        return Err(SharedEngineAdoptionEvidenceError::InvalidField {
            field: "compatible_frontend_families",
            value: "requires_two_frontend_families".to_string(),
        });
    }

    if adoption_level.requires_many_family_coverage() {
        require_many_family_coverage(compatible_frontend_families)?;
    }

    Ok(())
}

fn require_many_family_coverage(
    compatible_frontend_families: &[SharedEngineFrontendFamily],
) -> Result<(), SharedEngineAdoptionEvidenceError> {
    for tla_style in compatible_frontend_families
        .iter()
        .copied()
        .filter(|family| family.is_tla_style())
    {
        for graph_or_net in compatible_frontend_families
            .iter()
            .copied()
            .filter(|family| family.is_graph_or_net())
        {
            for hardware_symbolic_or_replay in compatible_frontend_families
                .iter()
                .copied()
                .filter(|family| family.is_hardware_symbolic_or_replay())
            {
                if tla_style != graph_or_net
                    && tla_style != hardware_symbolic_or_replay
                    && graph_or_net != hardware_symbolic_or_replay
                {
                    return Ok(());
                }
            }
        }
    }

    Err(SharedEngineAdoptionEvidenceError::InvalidField {
        field: "compatible_frontend_families",
        value: "missing_level_3_family_coverage".to_string(),
    })
}

fn require_compatible_frontend_families(
    compatible_frontend_families: &[SharedEngineFrontendFamily],
) -> Result<(), SharedEngineAdoptionEvidenceError> {
    if compatible_frontend_families.is_empty() {
        return Err(SharedEngineAdoptionEvidenceError::MissingField(
            "compatible_frontend_families",
        ));
    }

    let mut seen = Vec::new();
    for family in compatible_frontend_families {
        if seen.contains(family) {
            return Err(SharedEngineAdoptionEvidenceError::InvalidField {
                field: "compatible_frontend_families",
                value: family.code().to_string(),
            });
        }
        seen.push(*family);
    }

    Ok(())
}

fn require_frontend_family_blockers(
    compatible_frontend_families: &[SharedEngineFrontendFamily],
    frontend_family_blockers: &[SharedEngineAdoptionFamilyBlocker],
) -> Result<(), SharedEngineAdoptionEvidenceError> {
    let mut seen = Vec::new();
    for blocker in frontend_family_blockers {
        if blocker.blocker.trim().is_empty() || blocker.blocker.trim() == "none" {
            return Err(SharedEngineAdoptionEvidenceError::MissingField(
                "frontend_family_blockers",
            ));
        }
        if seen.contains(&blocker.frontend_family) {
            return Err(SharedEngineAdoptionEvidenceError::InvalidField {
                field: "frontend_family_blockers",
                value: blocker.frontend_family.code().to_string(),
            });
        }
        seen.push(blocker.frontend_family);
    }

    for frontend_family in all_frontend_families() {
        if compatible_frontend_families.contains(frontend_family)
            || frontend_family_blockers
                .iter()
                .any(|blocker| blocker.frontend_family == *frontend_family)
        {
            continue;
        }
        return Err(SharedEngineAdoptionEvidenceError::InvalidField {
            field: "frontend_family_blockers",
            value: format!("missing:{}", frontend_family.code()),
        });
    }

    Ok(())
}

fn require_tracked_frontend_family_blockers(
    adoption_level: SharedEngineAdoptionLevel,
    frontend_family_blockers: &[SharedEngineAdoptionFamilyBlocker],
) -> Result<(), SharedEngineAdoptionEvidenceError> {
    if adoption_level == SharedEngineAdoptionLevel::Level0 {
        return Ok(());
    }

    for blocker in frontend_family_blockers {
        if blocker.blocker.trim() == SHARED_ENGINE_ADOPTION_DEFAULT_FAMILY_BLOCKER {
            return Err(SharedEngineAdoptionEvidenceError::InvalidField {
                field: "frontend_family_blockers",
                value: format!("untracked:{}", blocker.frontend_family.code()),
            });
        }
    }

    Ok(())
}

fn require_named_frontend_families_are_compatible(
    field: &'static str,
    value: &str,
    compatible_frontend_families: &[SharedEngineFrontendFamily],
) -> Result<(), SharedEngineAdoptionEvidenceError> {
    for family in inferred_frontend_families(value) {
        if !compatible_frontend_families.contains(&family) {
            return Err(SharedEngineAdoptionEvidenceError::InvalidField {
                field,
                value: format!("{}:{}", evidence_value(value), family.code()),
            });
        }
    }

    Ok(())
}

fn require_supported_extraction_status(
    value: &str,
) -> Result<(), SharedEngineAdoptionEvidenceError> {
    if SHARED_ENGINE_ADOPTION_EXTRACTION_STATUSES.contains(&value) {
        Ok(())
    } else {
        Err(SharedEngineAdoptionEvidenceError::InvalidField {
            field: "extraction_status",
            value: value.to_string(),
        })
    }
}

fn evidence_value(value: &str) -> String {
    if value.trim().is_empty() {
        "none".to_string()
    } else {
        value.replace(
            |character: char| character.is_whitespace() || matches!(character, ',' | ':' | ';'),
            "_",
        )
    }
}

fn evidence_list(values: &[String]) -> String {
    let values = values
        .iter()
        .filter(|value| !value.trim().is_empty())
        .map(|value| evidence_value(value))
        .collect::<Vec<_>>();
    if values.is_empty() {
        "none".to_string()
    } else {
        values.join(",")
    }
}

fn evidence_family_list(values: &[SharedEngineFrontendFamily]) -> String {
    let values = all_frontend_families()
        .iter()
        .filter(|family| values.contains(family))
        .map(|family| family.code())
        .collect::<Vec<_>>();
    if values.is_empty() {
        "none".to_string()
    } else {
        values.join(",")
    }
}

fn evidence_family_blockers(values: &[SharedEngineAdoptionFamilyBlocker]) -> String {
    let values = all_frontend_families()
        .iter()
        .filter_map(|frontend_family| {
            values
                .iter()
                .find(|blocker| {
                    blocker.frontend_family == *frontend_family
                        && !blocker.blocker.trim().is_empty()
                })
                .map(|blocker| {
                    format!(
                        "{}:{}",
                        frontend_family.code(),
                        evidence_value(&blocker.blocker)
                    )
                })
        })
        .collect::<Vec<_>>();
    if values.is_empty() {
        "none".to_string()
    } else {
        values.join(",")
    }
}

fn required_field<'a>(
    row: &'a str,
    key: &'static str,
) -> Result<&'a str, SharedEngineAdoptionEvidenceError> {
    evidence_field(row, key).ok_or(SharedEngineAdoptionEvidenceError::MissingField(key))
}

fn require_non_empty_row_field(
    row: &str,
    key: &'static str,
) -> Result<(), SharedEngineAdoptionEvidenceError> {
    let value = required_field(row, key)?;
    if value.is_empty() || value == "none" {
        Err(SharedEngineAdoptionEvidenceError::MissingField(key))
    } else {
        Ok(())
    }
}

fn require_non_empty_row_generic_prerequisites(
    row: &str,
) -> Result<(), SharedEngineAdoptionEvidenceError> {
    let value = required_field(row, "generic_prerequisites")?;
    if value.is_empty() || value == "none" {
        return Err(SharedEngineAdoptionEvidenceError::MissingField(
            "generic_prerequisites",
        ));
    }

    for prerequisite in value.split(',') {
        if prerequisite.is_empty() || prerequisite == "none" {
            return Err(SharedEngineAdoptionEvidenceError::InvalidField {
                field: "generic_prerequisites",
                value: value.to_string(),
            });
        }
    }

    Ok(())
}

fn require_non_empty_row_acceptance_evidence(
    row: &str,
) -> Result<(), SharedEngineAdoptionEvidenceError> {
    let value = required_field(row, "acceptance_evidence")?;
    if value.is_empty() || value == "none" {
        return Err(SharedEngineAdoptionEvidenceError::MissingField(
            "acceptance_evidence",
        ));
    }

    for evidence in value.split(',') {
        if evidence.is_empty() || evidence == "none" {
            return Err(SharedEngineAdoptionEvidenceError::InvalidField {
                field: "acceptance_evidence",
                value: value.to_string(),
            });
        }
    }

    Ok(())
}

fn require_non_empty_struct_field(
    key: &'static str,
    value: &str,
) -> Result<(), SharedEngineAdoptionEvidenceError> {
    if value.trim().is_empty() || value.trim() == "none" {
        Err(SharedEngineAdoptionEvidenceError::MissingField(key))
    } else {
        Ok(())
    }
}

fn require_non_empty_generic_prerequisites(
    generic_prerequisites: &[String],
) -> Result<(), SharedEngineAdoptionEvidenceError> {
    if generic_prerequisites
        .iter()
        .any(|prerequisite| !prerequisite.trim().is_empty() && prerequisite.trim() != "none")
    {
        Ok(())
    } else {
        Err(SharedEngineAdoptionEvidenceError::MissingField(
            "generic_prerequisites",
        ))
    }
}

fn require_non_empty_acceptance_evidence(
    acceptance_evidence: &[String],
) -> Result<(), SharedEngineAdoptionEvidenceError> {
    if acceptance_evidence
        .iter()
        .any(|evidence| !evidence.trim().is_empty() && evidence.trim() != "none")
    {
        Ok(())
    } else {
        Err(SharedEngineAdoptionEvidenceError::MissingField(
            "acceptance_evidence",
        ))
    }
}

fn require_concrete_beneficiary(
    key: &'static str,
    value: &str,
) -> Result<(), SharedEngineAdoptionEvidenceError> {
    if is_placeholder_beneficiary(value) {
        Err(SharedEngineAdoptionEvidenceError::InvalidField {
            field: key,
            value: value.to_string(),
        })
    } else {
        Ok(())
    }
}

fn require_canonical_frontend_reference(
    key: &'static str,
    value: &str,
) -> Result<(), SharedEngineAdoptionEvidenceError> {
    if let Some(family) = canonical_frontend_family(value) {
        if value != family.code() {
            return Err(SharedEngineAdoptionEvidenceError::InvalidField {
                field: key,
                value: value.to_string(),
            });
        }
    }
    Ok(())
}

fn require_distinct_second_beneficiary(
    origin_frontend: &str,
    second_beneficiary: &str,
) -> Result<(), SharedEngineAdoptionEvidenceError> {
    let origin_families = inferred_frontend_families(origin_frontend);
    let second_families = inferred_frontend_families(second_beneficiary);
    if evidence_identity(origin_frontend) == evidence_identity(second_beneficiary)
        || (!origin_families.is_empty()
            && origin_families
                .iter()
                .any(|family| second_families.contains(family)))
    {
        Err(SharedEngineAdoptionEvidenceError::InvalidField {
            field: "second_beneficiary",
            value: second_beneficiary.to_string(),
        })
    } else {
        Ok(())
    }
}

fn require_distinct_beneficiary_families(
    first_beneficiary: &str,
    second_beneficiary: &str,
) -> Result<(), SharedEngineAdoptionEvidenceError> {
    let first_families = inferred_frontend_families(first_beneficiary);
    let second_families = inferred_frontend_families(second_beneficiary);
    if evidence_identity(first_beneficiary) == evidence_identity(second_beneficiary)
        || (!first_families.is_empty()
            && first_families
                .iter()
                .any(|family| second_families.contains(family)))
    {
        Err(SharedEngineAdoptionEvidenceError::InvalidField {
            field: "second_beneficiary",
            value: second_beneficiary.to_string(),
        })
    } else {
        Ok(())
    }
}

fn evidence_identity(value: &str) -> String {
    evidence_value(value.trim()).to_ascii_lowercase()
}

fn canonical_frontend_reference(value: String) -> String {
    let trimmed = value.trim();
    if let Some(family) = canonical_frontend_family(trimmed) {
        family.code().to_string()
    } else {
        value
    }
}

fn canonical_frontend_family(value: &str) -> Option<SharedEngineFrontendFamily> {
    let identity = evidence_identity(value);
    match identity.as_str() {
        "tla" | "tla_plus" | "ty" => Some(SharedEngineFrontendFamily::TlaPlus),
        "quint" => Some(SharedEngineFrontendFamily::Quint),
        "mcc" | "petri" | "mcc_petri" | "pnml" | "hlpnml" => {
            Some(SharedEngineFrontendFamily::MccPetri)
        }
        "aiger" => Some(SharedEngineFrontendFamily::Aiger),
        "btor" | "btor2" => Some(SharedEngineFrontendFamily::Btor2),
        "vmt" | "vmt_interchange" | "vmt_transition_system" => {
            Some(SharedEngineFrontendFamily::VmtTransitionSystem)
        }
        "ay" | "ay_only" | "ay_analytical" | "analytical" | "symbolic" => {
            Some(SharedEngineFrontendFamily::AYAnalytical)
        }
        "witness" | "replay" | "witness_replay" | "certificate" => {
            Some(SharedEngineFrontendFamily::WitnessReplay)
        }
        "future" | "future_importer" | "importer" => {
            Some(SharedEngineFrontendFamily::FutureImporter)
        }
        _ => None,
    }
}

fn is_placeholder_beneficiary(value: &str) -> bool {
    matches!(
        evidence_identity(value).as_str(),
        "none"
            | "unknown"
            | "origin_frontend"
            | "frontend_family"
            | "compatible_frontend_family"
            | "compatible_frontend_families"
            | "first_beneficiary"
            | "second_beneficiary"
    )
}

fn default_adoption_level(extraction_status: &str) -> SharedEngineAdoptionLevel {
    if extraction_status == "frontend-local-with-tracked-extraction" {
        SharedEngineAdoptionLevel::Level0
    } else {
        SharedEngineAdoptionLevel::Level2
    }
}

fn add_inferred_frontend_families(
    compatible_frontend_families: &mut Vec<SharedEngineFrontendFamily>,
    value: &str,
) {
    for family in inferred_frontend_families(value) {
        add_unique_frontend_family(compatible_frontend_families, family);
    }
}

fn inferred_frontend_families_for_evidence(
    evidence: &SharedEngineAdoptionEvidence,
) -> Vec<SharedEngineFrontendFamily> {
    let mut compatible_frontend_families = Vec::new();
    add_inferred_frontend_families(&mut compatible_frontend_families, &evidence.origin_frontend);
    add_inferred_frontend_families(
        &mut compatible_frontend_families,
        &evidence.first_beneficiary,
    );
    add_inferred_frontend_families(
        &mut compatible_frontend_families,
        &evidence.second_beneficiary,
    );
    compatible_frontend_families
}

fn default_compatible_frontend_families(
    evidence: &SharedEngineAdoptionEvidence,
) -> Vec<SharedEngineFrontendFamily> {
    canonical_frontend_family_vec(
        evidence
            .default_compatible_frontend_families
            .iter()
            .copied()
            .filter(|family| evidence.compatible_frontend_families.contains(family)),
    )
}

fn active_frontend_families(
    evidence: &SharedEngineAdoptionEvidence,
) -> Vec<SharedEngineFrontendFamily> {
    active_frontend_families_from_roles(
        &default_compatible_frontend_families(evidence),
        &canonical_frontend_family_vec(
            evidence
                .downstream_beneficiary_families
                .iter()
                .copied()
                .filter(|family| evidence.compatible_frontend_families.contains(family)),
        ),
    )
}

fn active_frontend_families_from_roles(
    default_families: &[SharedEngineFrontendFamily],
    downstream_families: &[SharedEngineFrontendFamily],
) -> Vec<SharedEngineFrontendFamily> {
    canonical_frontend_family_vec(
        default_families
            .iter()
            .chain(downstream_families.iter())
            .copied(),
    )
}

fn unblocked_compatible_frontend_families(
    evidence: &SharedEngineAdoptionEvidence,
) -> Vec<SharedEngineFrontendFamily> {
    canonical_frontend_family_vec(
        default_compatible_frontend_families(evidence)
            .into_iter()
            .chain(evidence.downstream_beneficiary_families.iter().copied())
            .chain(
                evidence
                    .remaining_compatible_frontend_families
                    .iter()
                    .copied(),
            ),
    )
}

fn remaining_families_from_compatible(
    evidence: &SharedEngineAdoptionEvidence,
    compatible_frontend_families: &[SharedEngineFrontendFamily],
) -> Vec<SharedEngineFrontendFamily> {
    let default_families = default_compatible_frontend_families(evidence);
    canonical_frontend_family_vec(
        compatible_frontend_families
            .iter()
            .copied()
            .filter(|family| !default_families.contains(family)),
    )
}

fn inferred_frontend_families(value: &str) -> Vec<SharedEngineFrontendFamily> {
    let normalized = value.to_ascii_lowercase();
    let mut families = Vec::new();

    let normalized_tokens = normalized
        .split(|character: char| !character.is_ascii_alphanumeric())
        .filter(|token| !token.is_empty())
        .collect::<Vec<_>>();

    if normalized == "tla"
        || normalized_tokens.contains(&"tla")
        || normalized.contains("tla+")
        || normalized.contains("ty")
        || normalized.contains("tla_")
    {
        add_unique_frontend_family(&mut families, SharedEngineFrontendFamily::TlaPlus);
    }
    if normalized.contains("quint") {
        add_unique_frontend_family(&mut families, SharedEngineFrontendFamily::Quint);
    }
    if normalized.contains("mcc")
        || normalized.contains("petri")
        || normalized.contains("pnml")
        || normalized.contains("hlpnml")
    {
        add_unique_frontend_family(&mut families, SharedEngineFrontendFamily::MccPetri);
    }
    if normalized.contains("aiger") {
        add_unique_frontend_family(&mut families, SharedEngineFrontendFamily::Aiger);
    }
    if normalized.contains("btor2") {
        add_unique_frontend_family(&mut families, SharedEngineFrontendFamily::Btor2);
    }
    if normalized.contains("vmt") {
        add_unique_frontend_family(
            &mut families,
            SharedEngineFrontendFamily::VmtTransitionSystem,
        );
    }
    if normalized.contains("ay")
        || normalized.contains("analytical")
        || normalized.contains("symbolic")
    {
        add_unique_frontend_family(&mut families, SharedEngineFrontendFamily::AYAnalytical);
    }
    if normalized.contains("witness")
        || normalized.contains("replay")
        || normalized.contains("certificate")
    {
        add_unique_frontend_family(&mut families, SharedEngineFrontendFamily::WitnessReplay);
    }
    if normalized.contains("future") || normalized.contains("importer") {
        add_unique_frontend_family(&mut families, SharedEngineFrontendFamily::FutureImporter);
    }

    families
}

fn add_unique_frontend_family(
    frontend_families: &mut Vec<SharedEngineFrontendFamily>,
    frontend_family: SharedEngineFrontendFamily,
) {
    if !frontend_families.contains(&frontend_family) {
        frontend_families.push(frontend_family);
    }
}

fn canonical_frontend_family_vec(
    frontend_families: impl IntoIterator<Item = SharedEngineFrontendFamily>,
) -> Vec<SharedEngineFrontendFamily> {
    let mut canonical = Vec::new();
    for frontend_family in frontend_families {
        add_unique_frontend_family(&mut canonical, frontend_family);
    }
    all_frontend_families()
        .iter()
        .copied()
        .filter(|frontend_family| canonical.contains(frontend_family))
        .collect()
}

fn canonical_frontend_family_blockers(
    frontend_family_blockers: impl IntoIterator<Item = SharedEngineAdoptionFamilyBlocker>,
) -> Vec<SharedEngineAdoptionFamilyBlocker> {
    let mut canonical = Vec::new();
    for blocker in frontend_family_blockers {
        if blocker.blocker.trim().is_empty() {
            continue;
        }
        upsert_frontend_family_blocker(
            &mut canonical,
            blocker.frontend_family,
            evidence_value(&blocker.blocker),
        );
    }
    all_frontend_families()
        .iter()
        .filter_map(|frontend_family| {
            canonical
                .iter()
                .find(|blocker| blocker.frontend_family == *frontend_family)
                .cloned()
        })
        .collect()
}

fn upsert_frontend_family_blocker(
    frontend_family_blockers: &mut Vec<SharedEngineAdoptionFamilyBlocker>,
    frontend_family: SharedEngineFrontendFamily,
    blocker: String,
) {
    if let Some(existing) = frontend_family_blockers
        .iter_mut()
        .find(|existing| existing.frontend_family == frontend_family)
    {
        existing.blocker = blocker;
    } else {
        frontend_family_blockers.push(SharedEngineAdoptionFamilyBlocker::new(
            frontend_family,
            blocker,
        ));
    }
}

fn all_frontend_families() -> &'static [SharedEngineFrontendFamily] {
    &SHARED_ENGINE_FRONTEND_FAMILY_VALUES
}

fn operational_frontend_families() -> &'static [SharedEngineFrontendFamily] {
    &SHARED_ENGINE_OPERATIONAL_FRONTEND_FAMILY_VALUES
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tracked_blockers_excluding(
        compatible_frontend_families: &[SharedEngineFrontendFamily],
    ) -> Vec<SharedEngineAdoptionFamilyBlocker> {
        all_frontend_families()
            .iter()
            .copied()
            .filter(|family| !compatible_frontend_families.contains(family))
            .map(|family| {
                SharedEngineAdoptionFamilyBlocker::new(
                    family,
                    format!("{} adoption blocker tracked", family.code()),
                )
            })
            .collect()
    }

    #[test]
    fn shared_engine_frontend_family_codes_match_public_registry() {
        let family_codes = all_frontend_families()
            .iter()
            .map(|family| family.code())
            .collect::<Vec<_>>();

        assert_eq!(family_codes, SHARED_ENGINE_ADOPTION_FRONTEND_FAMILIES);
        for code in SHARED_ENGINE_ADOPTION_FRONTEND_FAMILIES {
            let family = SharedEngineFrontendFamily::from_code(code).unwrap();
            assert_eq!(family.code(), *code);
        }
        assert_eq!(
            SharedEngineFrontendFamily::all(),
            &SHARED_ENGINE_FRONTEND_FAMILY_VALUES
        );
        assert_eq!(
            SharedEngineFrontendFamily::operational(),
            &SHARED_ENGINE_OPERATIONAL_FRONTEND_FAMILY_VALUES
        );
        assert!(!SharedEngineFrontendFamily::operational()
            .contains(&SharedEngineFrontendFamily::FutureImporter));
        assert_eq!(
            SharedEngineFrontendFamily::from_code("tla"),
            None,
            "shared-engine families use the canonical tla_plus code; tla remains a source-kind code"
        );
        assert_eq!(
            SharedEngineFrontendFamily::from_code("tla_plus"),
            Some(SharedEngineFrontendFamily::TlaPlus)
        );
    }

    #[test]
    fn shared_engine_adoption_row_renders_machine_checkable_fields() {
        let row = SharedEngineAdoptionEvidence::new(
            "quint",
            "prepared checker program",
            "quint cli",
            "tla cli",
            "shared-core-extracted",
            "W5",
            "cargo test -p tla-mc-core shared_engine_adoption",
        )
        .with_generic_prerequisite("frontend payload identity")
        .with_generic_prerequisite("storage policy identity")
        .with_frontend_family_contract(
            SharedEngineAdoptionLevel::Level2,
            [
                SharedEngineFrontendFamily::TlaPlus,
                SharedEngineFrontendFamily::Quint,
            ],
            tracked_blockers_excluding(&[
                SharedEngineFrontendFamily::TlaPlus,
                SharedEngineFrontendFamily::Quint,
            ]),
        )
        .render_evidence_row("CORE");

        assert!(row.starts_with("CORE shared_engine_adoption "));
        assert!(row.contains("schema=ty.shared.engine_adoption.v1"));
        assert!(row.contains("origin_frontend=quint"));
        assert!(row.contains("shared_engine_component=prepared_checker_program"));
        assert!(
            row.contains("generic_prerequisites=frontend_payload_identity,storage_policy_identity")
        );
        assert!(row.contains("first_beneficiary=quint_cli"));
        assert!(row.contains("second_beneficiary=tla_cli"));
        assert!(row.contains("extraction_status=shared-core-extracted"));
        assert!(row.contains("adoption_level=level-2"));
        assert!(row.contains("compatible_frontend_families=tla_plus,quint"));
        assert!(row.contains("active_frontend_families=tla_plus,quint"));
        assert!(row.contains("default_compatible_frontend_families=tla_plus,quint"));
        assert!(row.contains("downstream_beneficiary_families=none"));
        assert!(row.contains("remaining_compatible_frontend_families=none"));
        assert!(
            row.contains("frontend_family_blockers=mcc_petri:mcc_petri_adoption_blocker_tracked")
        );
        assert!(row.contains("blocker_status=tracked-blockers"));
        assert!(row.contains("aiger:aiger_adoption_blocker_tracked"));
        assert!(!row.contains(SHARED_ENGINE_ADOPTION_DEFAULT_FAMILY_BLOCKER));
        assert!(row.contains("owner=W5"));
        assert!(row.contains("acceptance_test=cargo_test_-p_tla-mc-core_shared_engine_adoption"));
        assert!(
            row.contains("acceptance_evidence=cargo_test_-p_tla-mc-core_shared_engine_adoption")
        );
        validate_shared_engine_adoption_evidence_row(&row).unwrap();
    }

    #[test]
    fn shared_engine_adoption_level_three_requires_many_family_coverage() {
        let row = SharedEngineAdoptionEvidence::new(
            "quint",
            "prepared checker program",
            "quint cli",
            "petri portfolio",
            "shared-core-extracted",
            "W5",
            "cargo test -p tla-mc-core shared_engine_adoption",
        )
        .with_generic_prerequisite("prepared transition descriptor")
        .with_frontend_family_contract(
            SharedEngineAdoptionLevel::Level3,
            [
                SharedEngineFrontendFamily::Quint,
                SharedEngineFrontendFamily::MccPetri,
                SharedEngineFrontendFamily::Aiger,
            ],
            tracked_blockers_excluding(&[
                SharedEngineFrontendFamily::Quint,
                SharedEngineFrontendFamily::MccPetri,
                SharedEngineFrontendFamily::Aiger,
            ]),
        )
        .render_evidence_row("CORE");

        assert!(row.contains("adoption_level=level-3"));
        assert!(row.contains("compatible_frontend_families=quint,mcc_petri,aiger"));
        assert!(row.contains("btor2:btor2_adoption_blocker_tracked"));
        assert!(row.contains("blocker_status=tracked-blockers"));
        validate_shared_engine_adoption_evidence_row(&row).unwrap();
    }

    #[test]
    fn shared_engine_adoption_transfer_record_requires_roles_and_acceptance_evidence() {
        let row = SharedEngineAdoptionEvidence::new(
            "tla",
            "prepared checker program",
            "tla cli",
            "petri portfolio",
            "shared-core-extracted",
            "W4",
            "cargo test -p tla-mc-core shared_engine_adoption",
        )
        .with_generic_prerequisite("prepared checker descriptor")
        .with_acceptance_evidence("benchmark parity gate")
        .with_frontend_family_contract(
            SharedEngineAdoptionLevel::Level2,
            [
                SharedEngineFrontendFamily::TlaPlus,
                SharedEngineFrontendFamily::MccPetri,
            ],
            tracked_blockers_excluding(&[
                SharedEngineFrontendFamily::TlaPlus,
                SharedEngineFrontendFamily::MccPetri,
            ]),
        )
        .render_evidence_row("CORE");

        assert!(row.contains("default_compatible_frontend_families=tla_plus,mcc_petri"));
        assert!(row.contains("downstream_beneficiary_families=none"));
        assert!(row.contains("remaining_compatible_frontend_families=none"));
        assert!(row.contains(
            "acceptance_evidence=cargo_test_-p_tla-mc-core_shared_engine_adoption,benchmark_parity_gate"
        ));
        validate_shared_engine_adoption_evidence_row(&row).unwrap();

        let missing_acceptance_evidence = row
            .replace(" acceptance_evidence=cargo_test_-p_tla-mc-core_shared_engine_adoption,benchmark_parity_gate", "");
        assert_eq!(
            validate_shared_engine_adoption_evidence_row(&missing_acceptance_evidence),
            Err(SharedEngineAdoptionEvidenceError::MissingField(
                "acceptance_evidence"
            ))
        );

        let legacy_without_role_fields = "CORE shared_engine_adoption schema=ty.shared.engine_adoption.v1 schema_version=1 origin_frontend=tla_plus shared_engine_component=prepared_checker_program generic_prerequisites=prepared_checker_descriptor first_beneficiary=tla_cli second_beneficiary=mcc_petri extraction_status=shared-core-extracted adoption_level=level-2 compatible_frontend_families=tla_plus,mcc_petri frontend_family_blockers=quint:not_active,aiger:not_active,btor2:not_active,vmt_transition_system:not_active,ay_analytical:not_active,witness_replay:not_active,future_importer:not_active blocker_status=tracked-blockers owner=W4 acceptance_test=core_tests acceptance_evidence=core_tests";
        assert_eq!(
            validate_shared_engine_adoption_evidence_row(legacy_without_role_fields),
            Err(SharedEngineAdoptionEvidenceError::MissingField(
                "default_compatible_frontend_families"
            ))
        );

        let mut missing_struct_acceptance_evidence = SharedEngineAdoptionEvidence::new(
            "tla",
            "prepared checker program",
            "tla cli",
            "petri portfolio",
            "shared-core-extracted",
            "W4",
            "cargo test -p tla-mc-core shared_engine_adoption",
        )
        .with_generic_prerequisite("prepared checker descriptor")
        .with_frontend_family_contract(
            SharedEngineAdoptionLevel::Level2,
            [
                SharedEngineFrontendFamily::TlaPlus,
                SharedEngineFrontendFamily::MccPetri,
            ],
            tracked_blockers_excluding(&[
                SharedEngineFrontendFamily::TlaPlus,
                SharedEngineFrontendFamily::MccPetri,
            ]),
        );
        missing_struct_acceptance_evidence
            .acceptance_evidence
            .clear();
        assert_eq!(
            missing_struct_acceptance_evidence.validate(),
            Err(SharedEngineAdoptionEvidenceError::MissingField(
                "acceptance_evidence"
            ))
        );
    }

    #[test]
    fn shared_engine_adoption_builder_canonicalizes_non_tla_frontend_families() {
        let evidence = SharedEngineAdoptionEvidence::new(
            "vmt",
            "native shared engine",
            "btor2",
            "aiger",
            "shared-core-extracted",
            "core-shared-engine",
            "cargo test -p tla-mc-core shared_engine_adoption",
        )
        .with_generic_prerequisite("transition descriptor identity")
        .with_frontend_family_contract(
            SharedEngineAdoptionLevel::Level3,
            [
                SharedEngineFrontendFamily::Aiger,
                SharedEngineFrontendFamily::Quint,
                SharedEngineFrontendFamily::VmtTransitionSystem,
                SharedEngineFrontendFamily::Btor2,
                SharedEngineFrontendFamily::Aiger,
                SharedEngineFrontendFamily::AYAnalytical,
            ],
            [
                SharedEngineAdoptionFamilyBlocker::new(
                    SharedEngineFrontendFamily::TlaPlus,
                    "not a default consumer for this hardware lane",
                ),
                SharedEngineAdoptionFamilyBlocker::new(
                    SharedEngineFrontendFamily::MccPetri,
                    "petri descriptor adapter pending",
                ),
                SharedEngineAdoptionFamilyBlocker::new(
                    SharedEngineFrontendFamily::WitnessReplay,
                    "witness replay adapter pending",
                ),
                SharedEngineAdoptionFamilyBlocker::new(
                    SharedEngineFrontendFamily::FutureImporter,
                    "future importer registration pending",
                ),
            ],
        );

        assert_eq!(evidence.origin_frontend, "vmt_transition_system");
        assert_eq!(evidence.first_beneficiary, "btor2");
        assert_eq!(evidence.second_beneficiary, "aiger");
        assert_eq!(
            evidence.compatible_frontend_families,
            vec![
                SharedEngineFrontendFamily::Quint,
                SharedEngineFrontendFamily::Aiger,
                SharedEngineFrontendFamily::Btor2,
                SharedEngineFrontendFamily::VmtTransitionSystem,
                SharedEngineFrontendFamily::AYAnalytical,
            ]
        );

        let row = evidence.render_evidence_row("CORE");
        assert!(row.contains("origin_frontend=vmt_transition_system"));
        assert!(row.contains("first_beneficiary=btor2"));
        assert!(row.contains("second_beneficiary=aiger"));
        assert!(row.contains(
            "compatible_frontend_families=quint,aiger,btor2,vmt_transition_system,ay_analytical"
        ));
        assert!(
            row.contains("default_compatible_frontend_families=aiger,btor2,vmt_transition_system")
        );
        assert!(row.contains("downstream_beneficiary_families=none"));
        assert!(row.contains("remaining_compatible_frontend_families=quint,ay_analytical"));
        assert!(!row.contains("first_beneficiary=tla"));
        validate_shared_engine_adoption_evidence_row(&row).unwrap();
    }

    #[test]
    fn shared_engine_adoption_downstream_builder_records_all_compatible_families() {
        let evidence = SharedEngineAdoptionEvidence::new(
            "btor2",
            "shared transition kernel",
            "aiger",
            "vmt",
            "shared-core-extracted",
            "core-shared-engine",
            "cargo test -p tla-mc-core shared_engine_adoption",
        )
        .with_generic_prerequisite("storage layout identity")
        .with_downstream_compatible_frontend_families(
            SharedEngineAdoptionLevel::Level3,
            [
                SharedEngineFrontendFamily::Quint,
                SharedEngineFrontendFamily::WitnessReplay,
            ],
            [
                SharedEngineFrontendFamily::AYAnalytical,
                SharedEngineFrontendFamily::Quint,
            ],
            [
                SharedEngineAdoptionFamilyBlocker::new(
                    SharedEngineFrontendFamily::TlaPlus,
                    "tla plus transition export pending",
                ),
                SharedEngineAdoptionFamilyBlocker::new(
                    SharedEngineFrontendFamily::MccPetri,
                    "petri transition adapter pending",
                ),
                SharedEngineAdoptionFamilyBlocker::new(
                    SharedEngineFrontendFamily::FutureImporter,
                    "future importer registry pending",
                ),
            ],
        );

        assert_eq!(evidence.origin_frontend, "btor2");
        assert_eq!(evidence.first_beneficiary, "aiger");
        assert_eq!(evidence.second_beneficiary, "vmt_transition_system");
        assert_eq!(
            evidence.compatible_frontend_families,
            vec![
                SharedEngineFrontendFamily::Quint,
                SharedEngineFrontendFamily::Aiger,
                SharedEngineFrontendFamily::Btor2,
                SharedEngineFrontendFamily::VmtTransitionSystem,
                SharedEngineFrontendFamily::AYAnalytical,
                SharedEngineFrontendFamily::WitnessReplay,
            ]
        );
        assert_eq!(
            evidence.frontend_family_blockers,
            vec![
                SharedEngineAdoptionFamilyBlocker::new(
                    SharedEngineFrontendFamily::TlaPlus,
                    "tla_plus_transition_export_pending",
                ),
                SharedEngineAdoptionFamilyBlocker::new(
                    SharedEngineFrontendFamily::MccPetri,
                    "petri_transition_adapter_pending",
                ),
                SharedEngineAdoptionFamilyBlocker::new(
                    SharedEngineFrontendFamily::FutureImporter,
                    "future_importer_registry_pending",
                ),
            ]
        );

        let row = evidence.render_evidence_row("CORE");
        assert!(row.contains(
            "compatible_frontend_families=quint,aiger,btor2,vmt_transition_system,ay_analytical,witness_replay"
        ));
        assert!(
            row.contains("default_compatible_frontend_families=aiger,btor2,vmt_transition_system")
        );
        assert!(row.contains("downstream_beneficiary_families=quint,witness_replay"));
        assert!(row.contains("remaining_compatible_frontend_families=ay_analytical"));
        assert!(row.contains(
            "frontend_family_blockers=tla_plus:tla_plus_transition_export_pending,mcc_petri:petri_transition_adapter_pending,future_importer:future_importer_registry_pending"
        ));
        assert!(row.contains("blocker_status=tracked-blockers"));
        assert!(!row.contains(SHARED_ENGINE_ADOPTION_DEFAULT_FAMILY_BLOCKER));
        validate_shared_engine_adoption_evidence_row(&row).unwrap();
    }

    #[test]
    fn shared_engine_adoption_downstream_builder_records_level_four_default_consumers() {
        let evidence = SharedEngineAdoptionEvidence::new(
            "ay_only",
            "shared proof replay kernel",
            "witness_replay",
            "quint",
            "shared-core-extracted",
            "core-shared-engine",
            "cargo test -p tla-mc-core shared_engine_adoption",
        )
        .with_generic_prerequisite("proof transcript identity")
        .with_level_four_operational_frontend_contract(
            SharedEngineFrontendFamily::operational().iter().copied(),
            [],
            [],
        );

        assert_eq!(evidence.origin_frontend, "ay_analytical");
        assert_eq!(evidence.first_beneficiary, "witness_replay");
        assert_eq!(
            evidence.compatible_frontend_families,
            SHARED_ENGINE_OPERATIONAL_FRONTEND_FAMILY_VALUES
        );
        assert_eq!(
            evidence.frontend_family_blockers,
            vec![SharedEngineAdoptionFamilyBlocker::future_importer_reserved()]
        );
        assert_eq!(
            evidence.blocker_status(),
            SharedEngineAdoptionBlockerStatus::TrackedBlockers
        );

        let row = evidence.render_evidence_row("CORE");
        assert!(row.contains(
            "compatible_frontend_families=tla_plus,quint,mcc_petri,aiger,btor2,vmt_transition_system,ay_analytical,witness_replay"
        ));
        assert!(
            row.contains("active_frontend_families=tla_plus,quint,mcc_petri,aiger,btor2,vmt_transition_system,ay_analytical,witness_replay")
        );
        assert!(row.contains(
            "default_compatible_frontend_families=tla_plus,quint,mcc_petri,aiger,btor2,vmt_transition_system,ay_analytical,witness_replay"
        ));
        assert!(row.contains("downstream_beneficiary_families=none"));
        assert!(row.contains("remaining_compatible_frontend_families=none"));
        assert!(row.contains(
            "frontend_family_blockers=future_importer:blocked_reserved_importer_contract"
        ));
        assert!(row.contains("blocker_status=tracked-blockers"));
        validate_shared_engine_adoption_evidence_row(&row).unwrap();
    }

    #[test]
    fn shared_engine_adoption_level_four_contract_distinguishes_family_roles() {
        let evidence = SharedEngineAdoptionEvidence::new(
            "tla",
            "shared runtime storage",
            "tla cli",
            "mcc petri portfolio",
            "shared-core-extracted",
            "core-shared-engine",
            "cargo test -p tla-mc-core shared_engine_adoption",
        )
        .with_generic_prerequisite("state vector storage layout")
        .with_level_four_frontend_family_contract(
            [
                SharedEngineFrontendFamily::TlaPlus,
                SharedEngineFrontendFamily::MccPetri,
            ],
            [
                SharedEngineFrontendFamily::Aiger,
                SharedEngineFrontendFamily::AYAnalytical,
            ],
            [
                SharedEngineFrontendFamily::Quint,
                SharedEngineFrontendFamily::Btor2,
                SharedEngineFrontendFamily::VmtTransitionSystem,
                SharedEngineFrontendFamily::WitnessReplay,
            ],
            [SharedEngineAdoptionFamilyBlocker::future_importer_reserved()],
        );

        assert_eq!(
            evidence.active_frontend_families(),
            vec![
                SharedEngineFrontendFamily::TlaPlus,
                SharedEngineFrontendFamily::MccPetri,
                SharedEngineFrontendFamily::Aiger,
                SharedEngineFrontendFamily::AYAnalytical,
            ]
        );
        assert_eq!(
            evidence.remaining_compatible_frontend_families,
            vec![
                SharedEngineFrontendFamily::Quint,
                SharedEngineFrontendFamily::Btor2,
                SharedEngineFrontendFamily::VmtTransitionSystem,
                SharedEngineFrontendFamily::WitnessReplay,
            ]
        );

        let row = evidence.render_evidence_row("CORE");
        assert!(row.contains("active_frontend_families=tla_plus,mcc_petri,aiger,ay_analytical"));
        assert!(row.contains("default_compatible_frontend_families=tla_plus,mcc_petri"));
        assert!(row.contains("downstream_beneficiary_families=aiger,ay_analytical"));
        assert!(row.contains(
            "remaining_compatible_frontend_families=quint,btor2,vmt_transition_system,witness_replay"
        ));
        assert!(row.contains(
            "frontend_family_blockers=future_importer:blocked_reserved_importer_contract"
        ));
        validate_shared_engine_adoption_evidence_row(&row).unwrap();
    }

    #[test]
    fn shared_engine_adoption_validation_checks_first_class_family_role_fields() {
        let row = "CORE shared_engine_adoption schema=ty.shared.engine_adoption.v1 schema_version=1 origin_frontend=btor2 shared_engine_component=shared_transition_kernel generic_prerequisites=storage_layout_identity first_beneficiary=aiger second_beneficiary=vmt_transition_system extraction_status=shared-core-extracted adoption_level=level-3 compatible_frontend_families=quint,aiger,btor2,vmt_transition_system,ay_analytical active_frontend_families=aiger,btor2,vmt_transition_system,ay_analytical default_compatible_frontend_families=aiger,btor2,vmt_transition_system downstream_beneficiary_families=ay_analytical remaining_compatible_frontend_families=quint frontend_family_blockers=tla_plus:not_active,mcc_petri:not_active,witness_replay:not_active,future_importer:not_active blocker_status=tracked-blockers owner=core acceptance_test=core_tests acceptance_evidence=core_tests";
        validate_shared_engine_adoption_evidence_row(row).unwrap();

        let missing_role_field = "CORE shared_engine_adoption schema=ty.shared.engine_adoption.v1 schema_version=1 origin_frontend=btor2 shared_engine_component=shared_transition_kernel generic_prerequisites=storage_layout_identity first_beneficiary=aiger second_beneficiary=vmt_transition_system extraction_status=shared-core-extracted adoption_level=level-3 compatible_frontend_families=quint,aiger,btor2,vmt_transition_system,ay_analytical active_frontend_families=aiger,btor2,vmt_transition_system,ay_analytical default_compatible_frontend_families=aiger,btor2,vmt_transition_system downstream_beneficiary_families=ay_analytical frontend_family_blockers=tla_plus:not_active,mcc_petri:not_active,witness_replay:not_active,future_importer:not_active blocker_status=tracked-blockers owner=core acceptance_test=core_tests acceptance_evidence=core_tests";
        assert_eq!(
            validate_shared_engine_adoption_evidence_row(missing_role_field),
            Err(SharedEngineAdoptionEvidenceError::MissingField(
                "remaining_compatible_frontend_families"
            ))
        );

        let overlapping_role_field = "CORE shared_engine_adoption schema=ty.shared.engine_adoption.v1 schema_version=1 origin_frontend=btor2 shared_engine_component=shared_transition_kernel generic_prerequisites=storage_layout_identity first_beneficiary=aiger second_beneficiary=vmt_transition_system extraction_status=shared-core-extracted adoption_level=level-3 compatible_frontend_families=quint,aiger,btor2,vmt_transition_system,ay_analytical active_frontend_families=aiger,btor2,vmt_transition_system,ay_analytical default_compatible_frontend_families=aiger,btor2,vmt_transition_system downstream_beneficiary_families=ay_analytical remaining_compatible_frontend_families=ay_analytical frontend_family_blockers=tla_plus:not_active,mcc_petri:not_active,witness_replay:not_active,future_importer:not_active blocker_status=tracked-blockers owner=core acceptance_test=core_tests acceptance_evidence=core_tests";
        assert_eq!(
            validate_shared_engine_adoption_evidence_row(overlapping_role_field),
            Err(SharedEngineAdoptionEvidenceError::InvalidField {
                field: "remaining_compatible_frontend_families",
                value: "ay_analytical".to_string(),
            })
        );

        let incomplete_active_field = "CORE shared_engine_adoption schema=ty.shared.engine_adoption.v1 schema_version=1 origin_frontend=btor2 shared_engine_component=shared_transition_kernel generic_prerequisites=storage_layout_identity first_beneficiary=aiger second_beneficiary=vmt_transition_system extraction_status=shared-core-extracted adoption_level=level-3 compatible_frontend_families=quint,aiger,btor2,vmt_transition_system,ay_analytical active_frontend_families=aiger,btor2,vmt_transition_system default_compatible_frontend_families=aiger,btor2,vmt_transition_system downstream_beneficiary_families=ay_analytical remaining_compatible_frontend_families=quint frontend_family_blockers=tla_plus:not_active,mcc_petri:not_active,witness_replay:not_active,future_importer:not_active blocker_status=tracked-blockers owner=core acceptance_test=core_tests acceptance_evidence=core_tests";
        assert_eq!(
            validate_shared_engine_adoption_evidence_row(incomplete_active_field),
            Err(SharedEngineAdoptionEvidenceError::InvalidField {
                field: "active_frontend_families",
                value: "aiger,btor2,vmt_transition_system".to_string(),
            })
        );
    }

    #[test]
    fn shared_engine_adoption_compatible_family_helper_records_remaining_role() {
        let evidence = SharedEngineAdoptionEvidence::new(
            "quint",
            "prepared checker program",
            "quint cli",
            "tla cli",
            "shared-core-extracted",
            "W5",
            "cargo test -p tla-mc-core shared_engine_adoption",
        )
        .with_generic_prerequisite("prepared transition descriptor")
        .with_frontend_family_contract(
            SharedEngineAdoptionLevel::Level2,
            [
                SharedEngineFrontendFamily::TlaPlus,
                SharedEngineFrontendFamily::Quint,
            ],
            tracked_blockers_excluding(&[
                SharedEngineFrontendFamily::TlaPlus,
                SharedEngineFrontendFamily::Quint,
            ]),
        )
        .with_compatible_frontend_family(SharedEngineFrontendFamily::MccPetri);

        assert_eq!(
            evidence.compatible_frontend_families,
            vec![
                SharedEngineFrontendFamily::TlaPlus,
                SharedEngineFrontendFamily::Quint,
                SharedEngineFrontendFamily::MccPetri,
            ]
        );
        assert_eq!(evidence.downstream_beneficiary_families, Vec::new());
        assert_eq!(
            evidence.remaining_compatible_frontend_families,
            vec![SharedEngineFrontendFamily::MccPetri]
        );
        evidence.validate().unwrap();

        let row = evidence.render_evidence_row("CORE");

        assert!(row.contains("compatible_frontend_families=tla_plus,quint,mcc_petri"));
        assert!(row.contains("active_frontend_families=tla_plus,quint"));
        assert!(row.contains("default_compatible_frontend_families=tla_plus,quint"));
        assert!(row.contains("downstream_beneficiary_families=none"));
        assert!(row.contains("remaining_compatible_frontend_families=mcc_petri"));
        validate_shared_engine_adoption_evidence_row(&row).unwrap();
    }

    #[test]
    fn shared_engine_adoption_struct_validation_rejects_invalid_family_role_partitions() {
        let evidence = SharedEngineAdoptionEvidence::new(
            "btor2",
            "shared transition kernel",
            "aiger",
            "vmt",
            "shared-core-extracted",
            "core-shared-engine",
            "cargo test -p tla-mc-core shared_engine_adoption",
        )
        .with_generic_prerequisite("storage layout identity")
        .with_downstream_compatible_frontend_families(
            SharedEngineAdoptionLevel::Level3,
            [SharedEngineFrontendFamily::AYAnalytical],
            [SharedEngineFrontendFamily::Quint],
            [
                SharedEngineAdoptionFamilyBlocker::new(
                    SharedEngineFrontendFamily::TlaPlus,
                    "tla plus transition export pending",
                ),
                SharedEngineAdoptionFamilyBlocker::new(
                    SharedEngineFrontendFamily::MccPetri,
                    "petri transition adapter pending",
                ),
                SharedEngineAdoptionFamilyBlocker::new(
                    SharedEngineFrontendFamily::WitnessReplay,
                    "witness replay adapter pending",
                ),
                SharedEngineAdoptionFamilyBlocker::new(
                    SharedEngineFrontendFamily::FutureImporter,
                    "future importer registry pending",
                ),
            ],
        );
        evidence.validate().unwrap();

        let mut overlapping = evidence.clone();
        overlapping
            .remaining_compatible_frontend_families
            .push(SharedEngineFrontendFamily::AYAnalytical);
        assert_eq!(
            overlapping.validate(),
            Err(SharedEngineAdoptionEvidenceError::InvalidField {
                field: "remaining_compatible_frontend_families",
                value: "ay_analytical".to_string()
            })
        );

        let mut incomplete = evidence.clone();
        incomplete.remaining_compatible_frontend_families.clear();
        assert_eq!(
            incomplete.validate(),
            Err(SharedEngineAdoptionEvidenceError::InvalidField {
                field: "remaining_compatible_frontend_families",
                value: "aiger,btor2,vmt_transition_system,ay_analytical".to_string()
            })
        );

        let mut noncanonical = evidence;
        noncanonical.compatible_frontend_families = vec![
            SharedEngineFrontendFamily::Btor2,
            SharedEngineFrontendFamily::Aiger,
            SharedEngineFrontendFamily::VmtTransitionSystem,
            SharedEngineFrontendFamily::Quint,
            SharedEngineFrontendFamily::AYAnalytical,
        ];
        assert_eq!(
            noncanonical.validate(),
            Err(SharedEngineAdoptionEvidenceError::InvalidField {
                field: "compatible_frontend_families",
                value: "quint,aiger,btor2,vmt_transition_system,ay_analytical".to_string()
            })
        );
    }

    #[test]
    fn shared_engine_adoption_validation_requires_concrete_beneficiaries() {
        let evidence = SharedEngineAdoptionEvidence::new(
            "quint",
            "prepared checker program",
            "origin_frontend",
            "compatible_frontend_family",
            "shared-core-extracted",
            "W5",
            "cargo test -p tla-mc-core shared_engine_adoption",
        )
        .with_generic_prerequisite("prepared transition descriptor")
        .with_frontend_family_contract(
            SharedEngineAdoptionLevel::Level2,
            [
                SharedEngineFrontendFamily::TlaPlus,
                SharedEngineFrontendFamily::Quint,
            ],
            tracked_blockers_excluding(&[
                SharedEngineFrontendFamily::TlaPlus,
                SharedEngineFrontendFamily::Quint,
            ]),
        );

        assert_eq!(
            evidence.validate(),
            Err(SharedEngineAdoptionEvidenceError::InvalidField {
                field: "first_beneficiary",
                value: "origin_frontend".to_string()
            })
        );

        let row = "CORE shared_engine_adoption schema=ty.shared.engine_adoption.v1 schema_version=1 origin_frontend=quint shared_engine_component=prepared_checker_program generic_prerequisites=descriptor first_beneficiary=origin_frontend second_beneficiary=compatible_frontend_family extraction_status=shared-core-extracted adoption_level=level-2 compatible_frontend_families=tla_plus,quint active_frontend_families=tla_plus,quint default_compatible_frontend_families=tla_plus,quint downstream_beneficiary_families=none remaining_compatible_frontend_families=none frontend_family_blockers=mcc_petri:not_active,aiger:not_active,btor2:not_active,vmt_transition_system:not_active,ay_analytical:not_active,witness_replay:not_active,future_importer:not_active blocker_status=tracked-blockers owner=W5 acceptance_test=core_tests acceptance_evidence=core_tests";
        assert_eq!(
            validate_shared_engine_adoption_evidence_row(row),
            Err(SharedEngineAdoptionEvidenceError::InvalidField {
                field: "first_beneficiary",
                value: "origin_frontend".to_string()
            })
        );
    }

    #[test]
    fn shared_engine_adoption_validation_rejects_noncanonical_family_aliases() {
        let row = "CORE shared_engine_adoption schema=ty.shared.engine_adoption.v1 schema_version=1 origin_frontend=vmt shared_engine_component=native_shared_engine generic_prerequisites=descriptor first_beneficiary=btor2 second_beneficiary=aiger extraction_status=shared-core-extracted adoption_level=level-3 compatible_frontend_families=quint,aiger,btor2,vmt_transition_system,ay_analytical active_frontend_families=aiger,btor2,vmt_transition_system default_compatible_frontend_families=aiger,btor2,vmt_transition_system downstream_beneficiary_families=none remaining_compatible_frontend_families=quint,ay_analytical frontend_family_blockers=tla_plus:not_active,mcc_petri:not_active,witness_replay:not_active,future_importer:not_active blocker_status=tracked-blockers owner=core acceptance_test=core_tests acceptance_evidence=core_tests";

        assert_eq!(
            validate_shared_engine_adoption_evidence_row(row),
            Err(SharedEngineAdoptionEvidenceError::InvalidField {
                field: "origin_frontend",
                value: "vmt".to_string()
            })
        );

        let mut evidence = SharedEngineAdoptionEvidence::new(
            "vmt",
            "native shared engine",
            "btor2",
            "aiger",
            "shared-core-extracted",
            "core",
            "core tests",
        )
        .with_generic_prerequisite("descriptor")
        .with_frontend_family_contract(
            SharedEngineAdoptionLevel::Level3,
            [
                SharedEngineFrontendFamily::Quint,
                SharedEngineFrontendFamily::Aiger,
                SharedEngineFrontendFamily::Btor2,
                SharedEngineFrontendFamily::VmtTransitionSystem,
            ],
            tracked_blockers_excluding(&[
                SharedEngineFrontendFamily::Quint,
                SharedEngineFrontendFamily::Aiger,
                SharedEngineFrontendFamily::Btor2,
                SharedEngineFrontendFamily::VmtTransitionSystem,
            ]),
        );
        evidence.origin_frontend = "vmt".to_string();

        assert_eq!(
            evidence.validate(),
            Err(SharedEngineAdoptionEvidenceError::InvalidField {
                field: "origin_frontend",
                value: "vmt".to_string()
            })
        );
    }

    #[test]
    fn shared_engine_adoption_explicit_contract_replaces_inferred_defaults() {
        let row = SharedEngineAdoptionEvidence::new(
            "aiger",
            "prepared checker program",
            "aiger portfolio",
            "btor2 portfolio",
            "shared-core-ready",
            "W5",
            "cargo test -p tla-mc-core shared_engine_adoption",
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
        .with_generic_prerequisite("prepared checker descriptor")
        .render_evidence_row("CORE");

        assert!(row.contains("adoption_level=level-3"));
        assert!(row.contains(
            "compatible_frontend_families=tla_plus,quint,mcc_petri,aiger,btor2,vmt_transition_system,ay_analytical,witness_replay"
        ));
        assert!(row.contains(
            "frontend_family_blockers=future_importer:awaiting_registered_importer_frontend"
        ));
        assert!(row.contains("blocker_status=tracked-blockers"));
        assert!(!row.contains(SHARED_ENGINE_ADOPTION_DEFAULT_FAMILY_BLOCKER));
        validate_shared_engine_adoption_evidence_row(&row).unwrap();
    }

    #[test]
    fn shared_engine_adoption_level_four_operational_contract_reserves_future_importer() {
        let row = SharedEngineAdoptionEvidence::new(
            "tla",
            "prepared checker program",
            "tla cli",
            "mcc petri portfolio",
            "shared-core-extracted",
            "W5",
            "cargo test -p tla-mc-core shared_engine_adoption",
        )
        .with_level_four_operational_frontend_contract(
            SharedEngineFrontendFamily::operational().iter().copied(),
            [],
            [],
        )
        .with_generic_prerequisite("prepared checker descriptor")
        .render_evidence_row("CORE");

        assert!(row.contains("adoption_level=level-4"));
        assert!(row.contains("origin_frontend=tla_plus"));
        assert!(row.contains(
            "compatible_frontend_families=tla_plus,quint,mcc_petri,aiger,btor2,vmt_transition_system,ay_analytical,witness_replay"
        ));
        assert!(row.contains(
            "active_frontend_families=tla_plus,quint,mcc_petri,aiger,btor2,vmt_transition_system,ay_analytical,witness_replay"
        ));
        assert!(row.contains(
            "frontend_family_blockers=future_importer:blocked_reserved_importer_contract"
        ));
        assert!(row.contains("blocker_status=tracked-blockers"));
        validate_shared_engine_adoption_evidence_row(&row).unwrap();
    }

    #[test]
    fn shared_engine_adoption_validation_rejects_level_three_without_graph_family() {
        let row = "CORE shared_engine_adoption schema=ty.shared.engine_adoption.v1 schema_version=1 origin_frontend=quint shared_engine_component=prepared_checker_program generic_prerequisites=descriptor first_beneficiary=quint second_beneficiary=aiger extraction_status=shared-core-extracted adoption_level=level-3 compatible_frontend_families=quint,aiger,ay_analytical active_frontend_families=quint,aiger default_compatible_frontend_families=quint,aiger downstream_beneficiary_families=none remaining_compatible_frontend_families=ay_analytical frontend_family_blockers=tla_plus:not_active,mcc_petri:not_active,btor2:not_active,vmt_transition_system:not_active,witness_replay:not_active,future_importer:not_active blocker_status=tracked-blockers owner=W5 acceptance_test=core_tests acceptance_evidence=core_tests";
        assert_eq!(
            validate_shared_engine_adoption_evidence_row(row),
            Err(SharedEngineAdoptionEvidenceError::InvalidField {
                field: "compatible_frontend_families",
                value: "missing_level_3_family_coverage".to_string()
            })
        );
    }

    #[test]
    fn shared_engine_adoption_validation_rejects_level_three_overlapping_family_only() {
        let evidence = SharedEngineAdoptionEvidence::new(
            "quint",
            "prepared checker program",
            "quint cli",
            "vmt export",
            "shared-core-extracted",
            "W5",
            "cargo test -p tla-mc-core shared_engine_adoption",
        )
        .with_generic_prerequisite("prepared transition descriptor")
        .with_frontend_family_contract(
            SharedEngineAdoptionLevel::Level3,
            [
                SharedEngineFrontendFamily::Quint,
                SharedEngineFrontendFamily::VmtTransitionSystem,
            ],
            tracked_blockers_excluding(&[
                SharedEngineFrontendFamily::Quint,
                SharedEngineFrontendFamily::VmtTransitionSystem,
            ]),
        );

        assert_eq!(
            evidence.validate(),
            Err(SharedEngineAdoptionEvidenceError::InvalidField {
                field: "compatible_frontend_families",
                value: "missing_level_3_family_coverage".to_string()
            })
        );
    }

    #[test]
    fn shared_engine_adoption_validation_rejects_missing_generic_prerequisites() {
        let evidence = SharedEngineAdoptionEvidence::new(
            "quint",
            "prepared checker program",
            "quint cli",
            "tla cli",
            "shared-core-extracted",
            "W5",
            "cargo test -p tla-mc-core shared_engine_adoption",
        )
        .with_frontend_family_contract(
            SharedEngineAdoptionLevel::Level2,
            [
                SharedEngineFrontendFamily::TlaPlus,
                SharedEngineFrontendFamily::Quint,
            ],
            tracked_blockers_excluding(&[
                SharedEngineFrontendFamily::TlaPlus,
                SharedEngineFrontendFamily::Quint,
            ]),
        );
        let row = evidence.render_evidence_row("CORE");

        assert_eq!(
            evidence.validate(),
            Err(SharedEngineAdoptionEvidenceError::MissingField(
                "generic_prerequisites"
            ))
        );
        assert_eq!(
            validate_shared_engine_adoption_evidence_row(&row),
            Err(SharedEngineAdoptionEvidenceError::MissingField(
                "generic_prerequisites"
            ))
        );
    }

    #[test]
    fn shared_engine_adoption_validation_rejects_untracked_shared_level_blockers() {
        let evidence = SharedEngineAdoptionEvidence::new(
            "quint",
            "prepared checker program",
            "quint cli",
            "tla cli",
            "shared-core-extracted",
            "W5",
            "cargo test -p tla-mc-core shared_engine_adoption",
        )
        .with_generic_prerequisite("prepared transition descriptor");

        assert_eq!(
            evidence.validate(),
            Err(SharedEngineAdoptionEvidenceError::InvalidField {
                field: "frontend_family_blockers",
                value: "untracked:mcc_petri".to_string()
            })
        );
    }

    #[test]
    fn shared_engine_adoption_validation_rejects_frontend_local_level_one() {
        let evidence = SharedEngineAdoptionEvidence::new(
            "quint",
            "prepared checker program",
            "quint cli",
            "tla cli",
            "frontend-local-with-tracked-extraction",
            "W5",
            "cargo test -p tla-mc-core shared_engine_adoption",
        )
        .with_generic_prerequisite("prepared transition descriptor")
        .with_frontend_family_contract(
            SharedEngineAdoptionLevel::Level1,
            [
                SharedEngineFrontendFamily::TlaPlus,
                SharedEngineFrontendFamily::Quint,
            ],
            tracked_blockers_excluding(&[
                SharedEngineFrontendFamily::TlaPlus,
                SharedEngineFrontendFamily::Quint,
            ]),
        );

        assert_eq!(
            evidence.validate(),
            Err(SharedEngineAdoptionEvidenceError::InvalidField {
                field: "extraction_status",
                value: "frontend-local-with-tracked-extraction".to_string()
            })
        );
    }

    #[test]
    fn shared_engine_adoption_validation_rejects_shared_status_level_zero() {
        let evidence = SharedEngineAdoptionEvidence::new(
            "quint",
            "prepared checker program",
            "quint cli",
            "tla cli",
            "shared-core-extracted",
            "W5",
            "cargo test -p tla-mc-core shared_engine_adoption",
        )
        .with_generic_prerequisite("prepared transition descriptor")
        .with_frontend_family_contract(
            SharedEngineAdoptionLevel::Level0,
            [
                SharedEngineFrontendFamily::TlaPlus,
                SharedEngineFrontendFamily::Quint,
            ],
            tracked_blockers_excluding(&[
                SharedEngineFrontendFamily::TlaPlus,
                SharedEngineFrontendFamily::Quint,
            ]),
        );

        assert_eq!(
            evidence.validate(),
            Err(SharedEngineAdoptionEvidenceError::InvalidField {
                field: "adoption_level",
                value: "level-0".to_string()
            })
        );
    }

    #[test]
    fn shared_engine_adoption_validation_rejects_level_four_future_importer_default() {
        let evidence = SharedEngineAdoptionEvidence::new(
            "aiger",
            "prepared checker program",
            "aiger portfolio",
            "btor2 portfolio",
            "shared-core-extracted",
            "W5",
            "cargo test -p tla-mc-core shared_engine_adoption",
        )
        .with_generic_prerequisite("prepared transition descriptor")
        .with_level_four_frontend_family_contract(
            [
                SharedEngineFrontendFamily::TlaPlus,
                SharedEngineFrontendFamily::Quint,
                SharedEngineFrontendFamily::MccPetri,
                SharedEngineFrontendFamily::Aiger,
                SharedEngineFrontendFamily::Btor2,
                SharedEngineFrontendFamily::VmtTransitionSystem,
                SharedEngineFrontendFamily::AYAnalytical,
                SharedEngineFrontendFamily::WitnessReplay,
                SharedEngineFrontendFamily::FutureImporter,
            ],
            [],
            [],
            [],
        );

        assert_eq!(
            evidence.validate(),
            Err(SharedEngineAdoptionEvidenceError::InvalidField {
                field: "active_frontend_families",
                value: "future_importer".to_string()
            })
        );
    }

    #[test]
    fn shared_engine_adoption_validation_rejects_level_four_unreserved_future_importer() {
        let row = "CORE shared_engine_adoption schema=ty.shared.engine_adoption.v1 schema_version=1 origin_frontend=tla_plus shared_engine_component=prepared_checker_program generic_prerequisites=descriptor first_beneficiary=tla_plus second_beneficiary=mcc_petri extraction_status=shared-core-extracted adoption_level=level-4 compatible_frontend_families=tla_plus,quint,mcc_petri,aiger,btor2,vmt_transition_system,ay_analytical,witness_replay,future_importer active_frontend_families=tla_plus,quint,mcc_petri,aiger,btor2,vmt_transition_system,ay_analytical,witness_replay default_compatible_frontend_families=tla_plus,quint,mcc_petri,aiger,btor2,vmt_transition_system,ay_analytical,witness_replay downstream_beneficiary_families=none remaining_compatible_frontend_families=none frontend_family_blockers=none blocker_status=no-blockers owner=W5 acceptance_test=core_tests acceptance_evidence=core_tests";
        assert_eq!(
            validate_shared_engine_adoption_evidence_row(row),
            Err(SharedEngineAdoptionEvidenceError::InvalidField {
                field: "frontend_family_blockers",
                value: "future_importer_reserved".to_string()
            })
        );
    }

    #[test]
    fn shared_engine_adoption_validation_rejects_origin_family_as_second_beneficiary() {
        let evidence = SharedEngineAdoptionEvidence::new(
            "quint",
            "prepared checker program",
            "tla cli",
            "quint cli",
            "shared-core-extracted",
            "W5",
            "cargo test -p tla-mc-core shared_engine_adoption",
        )
        .with_generic_prerequisite("prepared transition descriptor")
        .with_frontend_family_contract(
            SharedEngineAdoptionLevel::Level2,
            [
                SharedEngineFrontendFamily::TlaPlus,
                SharedEngineFrontendFamily::Quint,
            ],
            tracked_blockers_excluding(&[
                SharedEngineFrontendFamily::TlaPlus,
                SharedEngineFrontendFamily::Quint,
            ]),
        );

        assert_eq!(
            evidence.validate(),
            Err(SharedEngineAdoptionEvidenceError::InvalidField {
                field: "second_beneficiary",
                value: "quint cli".to_string()
            })
        );
    }

    #[test]
    fn shared_engine_adoption_validation_rejects_missing_adoption_level() {
        let row = "CORE shared_engine_adoption schema=ty.shared.engine_adoption.v1 schema_version=1 origin_frontend=quint shared_engine_component=prepared_checker_program generic_prerequisites=descriptor first_beneficiary=quint_cli second_beneficiary=tla_cli extraction_status=shared-core-extracted compatible_frontend_families=tla_plus,quint active_frontend_families=tla_plus,quint default_compatible_frontend_families=tla_plus,quint downstream_beneficiary_families=none remaining_compatible_frontend_families=none frontend_family_blockers=mcc_petri:not_active,aiger:not_active,btor2:not_active,vmt_transition_system:not_active,ay_analytical:not_active,witness_replay:not_active,future_importer:not_active blocker_status=tracked-blockers owner=W5 acceptance_test=core_tests acceptance_evidence=core_tests";
        assert_eq!(
            validate_shared_engine_adoption_evidence_row(row),
            Err(SharedEngineAdoptionEvidenceError::MissingField(
                "adoption_level"
            ))
        );
    }

    #[test]
    fn shared_engine_adoption_validation_rejects_missing_blocker_status() {
        let row = "CORE shared_engine_adoption schema=ty.shared.engine_adoption.v1 schema_version=1 origin_frontend=quint shared_engine_component=prepared_checker_program generic_prerequisites=descriptor first_beneficiary=quint_cli second_beneficiary=tla_cli extraction_status=shared-core-extracted adoption_level=level-2 compatible_frontend_families=tla_plus,quint active_frontend_families=tla_plus,quint default_compatible_frontend_families=tla_plus,quint downstream_beneficiary_families=none remaining_compatible_frontend_families=none frontend_family_blockers=mcc_petri:not_active,aiger:not_active,btor2:not_active,vmt_transition_system:not_active,ay_analytical:not_active,witness_replay:not_active,future_importer:not_active owner=W5 acceptance_test=core_tests acceptance_evidence=core_tests";
        assert_eq!(
            validate_shared_engine_adoption_evidence_row(row),
            Err(SharedEngineAdoptionEvidenceError::MissingField(
                "blocker_status"
            ))
        );
    }

    #[test]
    fn shared_engine_adoption_validation_rejects_inconsistent_blocker_status() {
        let row = "CORE shared_engine_adoption schema=ty.shared.engine_adoption.v1 schema_version=1 origin_frontend=quint shared_engine_component=prepared_checker_program generic_prerequisites=descriptor first_beneficiary=quint_cli second_beneficiary=tla_cli extraction_status=shared-core-extracted adoption_level=level-2 compatible_frontend_families=tla_plus,quint active_frontend_families=tla_plus,quint default_compatible_frontend_families=tla_plus,quint downstream_beneficiary_families=none remaining_compatible_frontend_families=none frontend_family_blockers=mcc_petri:not_active,aiger:not_active,btor2:not_active,vmt_transition_system:not_active,ay_analytical:not_active,witness_replay:not_active,future_importer:not_active blocker_status=no-blockers owner=W5 acceptance_test=core_tests acceptance_evidence=core_tests";
        assert_eq!(
            validate_shared_engine_adoption_evidence_row(row),
            Err(SharedEngineAdoptionEvidenceError::InvalidField {
                field: "blocker_status",
                value: "no-blockers".to_string()
            })
        );
    }

    #[test]
    fn shared_engine_adoption_validation_rejects_missing_family_blocker() {
        let row = "CORE shared_engine_adoption schema=ty.shared.engine_adoption.v1 schema_version=1 origin_frontend=quint shared_engine_component=prepared_checker_program generic_prerequisites=descriptor first_beneficiary=quint_cli second_beneficiary=tla_cli extraction_status=shared-core-extracted adoption_level=level-2 compatible_frontend_families=tla_plus,quint active_frontend_families=tla_plus,quint default_compatible_frontend_families=tla_plus,quint downstream_beneficiary_families=none remaining_compatible_frontend_families=none frontend_family_blockers=mcc_petri:not_active,aiger:not_active,btor2:not_active,vmt_transition_system:not_active,ay_analytical:not_active,witness_replay:not_active blocker_status=tracked-blockers owner=W5 acceptance_test=core_tests acceptance_evidence=core_tests";
        assert_eq!(
            validate_shared_engine_adoption_evidence_row(row),
            Err(SharedEngineAdoptionEvidenceError::InvalidField {
                field: "frontend_family_blockers",
                value: "missing:future_importer".to_string()
            })
        );
    }

    #[test]
    fn shared_engine_adoption_validation_rejects_blocked_compatible_family() {
        let row = "CORE shared_engine_adoption schema=ty.shared.engine_adoption.v1 schema_version=1 origin_frontend=quint shared_engine_component=prepared_checker_program generic_prerequisites=descriptor first_beneficiary=quint_cli second_beneficiary=tla_cli extraction_status=shared-core-extracted adoption_level=level-2 compatible_frontend_families=tla_plus,quint active_frontend_families=tla_plus,quint default_compatible_frontend_families=tla_plus,quint downstream_beneficiary_families=none remaining_compatible_frontend_families=none frontend_family_blockers=quint:blocked,mcc_petri:not_active,aiger:not_active,btor2:not_active,vmt_transition_system:not_active,ay_analytical:not_active,witness_replay:not_active,future_importer:not_active blocker_status=tracked-blockers owner=W5 acceptance_test=core_tests acceptance_evidence=core_tests";
        assert_eq!(
            validate_shared_engine_adoption_evidence_row(row),
            Err(SharedEngineAdoptionEvidenceError::InvalidField {
                field: "frontend_family_blockers",
                value: "active:quint".to_string()
            })
        );
    }

    #[test]
    fn shared_engine_adoption_validation_rejects_second_beneficiary_family_gap() {
        let row = "CORE shared_engine_adoption schema=ty.shared.engine_adoption.v1 schema_version=1 origin_frontend=quint shared_engine_component=prepared_checker_program generic_prerequisites=descriptor first_beneficiary=quint_cli second_beneficiary=tla_cli extraction_status=shared-core-extracted adoption_level=level-2 compatible_frontend_families=quint,aiger active_frontend_families=quint default_compatible_frontend_families=quint downstream_beneficiary_families=none remaining_compatible_frontend_families=aiger frontend_family_blockers=tla_plus:not_active,mcc_petri:not_active,btor2:not_active,vmt_transition_system:not_active,ay_analytical:not_active,witness_replay:not_active,future_importer:not_active blocker_status=tracked-blockers owner=W5 acceptance_test=core_tests acceptance_evidence=core_tests";
        assert_eq!(
            validate_shared_engine_adoption_evidence_row(row),
            Err(SharedEngineAdoptionEvidenceError::InvalidField {
                field: "second_beneficiary",
                value: "tla_cli:tla_plus".to_string()
            })
        );
    }

    #[test]
    fn shared_engine_adoption_validation_rejects_missing_second_beneficiary() {
        let row = "CORE shared_engine_adoption schema=ty.shared.engine_adoption.v1 schema_version=1 origin_frontend=quint shared_engine_component=prepared_checker_program generic_prerequisites=descriptor first_beneficiary=quint_cli extraction_status=extracted adoption_level=level-2 compatible_frontend_families=tla_plus,quint active_frontend_families=tla_plus,quint default_compatible_frontend_families=tla_plus,quint downstream_beneficiary_families=none remaining_compatible_frontend_families=none frontend_family_blockers=mcc_petri:not_active,aiger:not_active,btor2:not_active,vmt_transition_system:not_active,ay_analytical:not_active,witness_replay:not_active,future_importer:not_active blocker_status=tracked-blockers owner=W5 acceptance_test=core_tests acceptance_evidence=core_tests";
        assert_eq!(
            validate_shared_engine_adoption_evidence_row(row),
            Err(SharedEngineAdoptionEvidenceError::MissingField(
                "second_beneficiary"
            ))
        );

        let evidence = SharedEngineAdoptionEvidence::new(
            "quint",
            "prepared_checker_program",
            "quint_cli",
            "",
            "shared-core-extracted",
            "W5",
            "core_tests",
        );
        assert_eq!(
            evidence.validate(),
            Err(SharedEngineAdoptionEvidenceError::MissingField(
                "second_beneficiary"
            ))
        );
    }

    #[test]
    fn shared_engine_adoption_validation_rejects_origin_as_second_beneficiary() {
        let row = "CORE shared_engine_adoption schema=ty.shared.engine_adoption.v1 schema_version=1 origin_frontend=quint shared_engine_component=prepared_checker_program generic_prerequisites=descriptor first_beneficiary=quint_cli second_beneficiary=quint extraction_status=shared-core-extracted adoption_level=level-2 compatible_frontend_families=quint,aiger active_frontend_families=quint default_compatible_frontend_families=quint downstream_beneficiary_families=none remaining_compatible_frontend_families=aiger frontend_family_blockers=tla_plus:not_active,mcc_petri:not_active,btor2:not_active,vmt_transition_system:not_active,ay_analytical:not_active,witness_replay:not_active,future_importer:not_active blocker_status=tracked-blockers owner=W5 acceptance_test=core_tests acceptance_evidence=core_tests";
        assert_eq!(
            validate_shared_engine_adoption_evidence_row(row),
            Err(SharedEngineAdoptionEvidenceError::InvalidField {
                field: "second_beneficiary",
                value: "quint".to_string()
            })
        );

        let evidence = SharedEngineAdoptionEvidence::new(
            "quint",
            "prepared_checker_program",
            "quint_cli",
            "quint",
            "shared-core-extracted",
            "W5",
            "core_tests",
        );
        assert_eq!(
            evidence.validate(),
            Err(SharedEngineAdoptionEvidenceError::InvalidField {
                field: "second_beneficiary",
                value: "quint".to_string()
            })
        );
    }

    #[test]
    fn shared_engine_adoption_validation_rejects_missing_extraction_status() {
        let row = "CORE shared_engine_adoption schema=ty.shared.engine_adoption.v1 schema_version=1 origin_frontend=quint shared_engine_component=prepared_checker_program generic_prerequisites=descriptor first_beneficiary=quint_cli second_beneficiary=tla_cli adoption_level=level-2 compatible_frontend_families=tla_plus,quint active_frontend_families=tla_plus,quint default_compatible_frontend_families=tla_plus,quint downstream_beneficiary_families=none remaining_compatible_frontend_families=none frontend_family_blockers=mcc_petri:not_active,aiger:not_active,btor2:not_active,vmt_transition_system:not_active,ay_analytical:not_active,witness_replay:not_active,future_importer:not_active blocker_status=tracked-blockers owner=W5 acceptance_test=core_tests acceptance_evidence=core_tests";
        assert_eq!(
            validate_shared_engine_adoption_evidence_row(row),
            Err(SharedEngineAdoptionEvidenceError::MissingField(
                "extraction_status"
            ))
        );

        let evidence = SharedEngineAdoptionEvidence::new(
            "quint",
            "prepared_checker_program",
            "quint_cli",
            "tla_cli",
            "",
            "W5",
            "core_tests",
        );
        assert_eq!(
            evidence.validate(),
            Err(SharedEngineAdoptionEvidenceError::MissingField(
                "extraction_status"
            ))
        );
    }

    #[test]
    fn shared_engine_adoption_validation_rejects_unknown_extraction_status() {
        let row = "CORE shared_engine_adoption schema=ty.shared.engine_adoption.v1 schema_version=1 origin_frontend=quint shared_engine_component=prepared_checker_program generic_prerequisites=descriptor first_beneficiary=quint_cli second_beneficiary=tla_cli extraction_status=extracted adoption_level=level-2 compatible_frontend_families=tla_plus,quint active_frontend_families=tla_plus,quint default_compatible_frontend_families=tla_plus,quint downstream_beneficiary_families=none remaining_compatible_frontend_families=none frontend_family_blockers=mcc_petri:not_active,aiger:not_active,btor2:not_active,vmt_transition_system:not_active,ay_analytical:not_active,witness_replay:not_active,future_importer:not_active blocker_status=tracked-blockers owner=W5 acceptance_test=core_tests acceptance_evidence=core_tests";
        assert_eq!(
            validate_shared_engine_adoption_evidence_row(row),
            Err(SharedEngineAdoptionEvidenceError::InvalidField {
                field: "extraction_status",
                value: "extracted".to_string()
            })
        );
    }
}
