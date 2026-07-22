// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Internal certified result scaffolding for MCC examinations.
//!
//! This module is intentionally not wired into the live examination pipelines
//! yet. It provides the no-behavior-change adapter that future engines can use
//! before `ExaminationRecord` remains the public compatibility boundary.

use crate::examination::{ExaminationRecord, ExaminationValue, StateSpaceReport};
use crate::output::{Technique, Techniques, Verdict};

/// Result of an examination lane that either has an exact answer with evidence
/// or a typed reason for failing closed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum ExactOrUnknown<T> {
    Exact(Exact<T>),
    Unknown(Unknown),
}

impl<T> ExactOrUnknown<T> {
    /// Build an exact result.
    #[must_use]
    pub(crate) fn exact(value: T, evidence: EvidenceSet) -> Self {
        Self::Exact(Exact::new(value, evidence))
    }

    /// Build an unknown result.
    #[must_use]
    pub(crate) fn unknown(reason: UnknownReason) -> Self {
        Self::unknown_with_techniques(reason, Techniques::default())
    }

    /// Build an unknown result with the technique provenance that reached it.
    #[must_use]
    pub(crate) fn unknown_with_techniques(reason: UnknownReason, techniques: Techniques) -> Self {
        Self::Unknown(Unknown::new(reason, techniques))
    }

    /// Transform an exact value without changing unknowns.
    #[must_use]
    pub(crate) fn map<U>(self, f: impl FnOnce(T) -> U) -> ExactOrUnknown<U> {
        match self {
            Self::Exact(exact) => ExactOrUnknown::Exact(Exact {
                value: f(exact.value),
                evidence: exact.evidence,
            }),
            Self::Unknown(unknown) => ExactOrUnknown::Unknown(unknown),
        }
    }

    /// Whether the result contains an exact value.
    #[must_use]
    pub(crate) fn is_exact(&self) -> bool {
        matches!(self, Self::Exact(_))
    }
}

/// An exact value plus the evidence that justified emitting it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Exact<T> {
    value: T,
    evidence: EvidenceSet,
}

impl<T> Exact<T> {
    fn new(value: T, evidence: EvidenceSet) -> Self {
        Self { value, evidence }
    }
}

/// A fail-closed result with a typed reason.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Unknown {
    reason: UnknownReason,
    techniques: Techniques,
}

impl Unknown {
    fn new(reason: UnknownReason, techniques: Techniques) -> Self {
        Self { reason, techniques }
    }
}

/// Exact values that can be represented in MCC output.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum MccExactValue {
    Bool(BoolVerdict),
    UpperBound(u64),
    StateSpace(StateSpaceReport),
}

impl MccExactValue {
    fn kind(&self) -> MccValueKind {
        match self {
            Self::Bool(_) => MccValueKind::Bool,
            Self::UpperBound(_) => MccValueKind::UpperBound,
            Self::StateSpace(_) => MccValueKind::StateSpace,
        }
    }
}

/// Boolean MCC verdicts that are exact by construction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BoolVerdict {
    True,
    False,
}

impl From<BoolVerdict> for Verdict {
    fn from(value: BoolVerdict) -> Self {
        match value {
            BoolVerdict::True => Self::True,
            BoolVerdict::False => Self::False,
        }
    }
}

/// The shape of the MCC value, used to adapt `Unknown` to legacy output.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum MccValueKind {
    Bool,
    UpperBound,
    StateSpace,
}

/// Evidence attached to an exact result.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct EvidenceSet {
    pub(crate) primary: Evidence,
    pub(crate) supporting: Vec<Evidence>,
    pub(crate) techniques: Techniques,
}

impl EvidenceSet {
    /// Build evidence with a single technique tag.
    #[must_use]
    pub(crate) fn single(primary: Evidence, technique: Technique) -> Self {
        Self {
            primary,
            supporting: Vec::new(),
            techniques: Techniques::single(technique),
        }
    }

    /// Temporary migration evidence for existing trusted lanes.
    #[must_use]
    pub(crate) fn legacy_explicit(lane: &'static str) -> Self {
        Self::single(Evidence::LegacyTrusted { lane }, Technique::Explicit)
    }
}

/// Coarse evidence categories for exact answers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Evidence {
    ExplicitComplete,
    WitnessTrace { length: usize },
    InductiveInvariant,
    AigerProof,
    BoundCertificate,
    ReductionCertificate,
    CrossChecked,
    LegacyTrusted { lane: &'static str },
}

/// Typed reasons that force `CANNOT_COMPUTE`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum UnknownReason {
    UnsupportedModel,
    UnsupportedFormula,
    DisabledBySoundnessGuard,
    UnresolvedNames { total: usize, unresolved: usize },
    IncompleteExploration { visited_states: Option<usize> },
    Deadline,
    MemoryBudget,
    CheckpointError,
    SolverUnknown { solver: &'static str },
    EncodingRejected { encoding: &'static str },
    ReductionNotCertified,
    EvidenceValidationFailed { validator: &'static str },
    InternalError { component: &'static str },
}

/// Certified internal record for a single MCC formula or metric group.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct CertifiedExaminationRecord {
    formula_id: String,
    kind: MccValueKind,
    outcome: ExactOrUnknown<MccExactValue>,
}

impl CertifiedExaminationRecord {
    fn exact(formula_id: impl Into<String>, value: MccExactValue, evidence: EvidenceSet) -> Self {
        let kind = value.kind();
        Self {
            formula_id: formula_id.into(),
            kind,
            outcome: ExactOrUnknown::exact(value, evidence),
        }
    }

    fn unknown(formula_id: impl Into<String>, kind: MccValueKind, reason: UnknownReason) -> Self {
        Self::unknown_with_techniques(formula_id, kind, reason, Techniques::default())
    }

    fn unknown_with_techniques(
        formula_id: impl Into<String>,
        kind: MccValueKind,
        reason: UnknownReason,
        techniques: Techniques,
    ) -> Self {
        Self {
            formula_id: formula_id.into(),
            kind,
            outcome: ExactOrUnknown::unknown_with_techniques(reason, techniques),
        }
    }

    /// Build an exact boolean record.
    #[must_use]
    pub(crate) fn exact_bool(
        formula_id: impl Into<String>,
        value: BoolVerdict,
        evidence: EvidenceSet,
    ) -> Self {
        Self::exact(formula_id, MccExactValue::Bool(value), evidence)
    }

    /// Build an unknown boolean record.
    #[must_use]
    pub(crate) fn unknown_bool(formula_id: impl Into<String>, reason: UnknownReason) -> Self {
        Self::unknown(formula_id, MccValueKind::Bool, reason)
    }

    /// Build an unknown boolean record with technique provenance.
    #[must_use]
    pub(crate) fn unknown_bool_with_techniques(
        formula_id: impl Into<String>,
        reason: UnknownReason,
        techniques: Techniques,
    ) -> Self {
        Self::unknown_with_techniques(formula_id, MccValueKind::Bool, reason, techniques)
    }

    /// Build an exact upper-bound record.
    #[must_use]
    pub(crate) fn exact_upper_bound(
        formula_id: impl Into<String>,
        value: u64,
        evidence: EvidenceSet,
    ) -> Self {
        Self::exact(formula_id, MccExactValue::UpperBound(value), evidence)
    }

    /// Build an unknown upper-bound record.
    #[must_use]
    pub(crate) fn unknown_upper_bound(
        formula_id: impl Into<String>,
        reason: UnknownReason,
    ) -> Self {
        Self::unknown(formula_id, MccValueKind::UpperBound, reason)
    }

    /// Build an unknown upper-bound record with technique provenance.
    #[must_use]
    pub(crate) fn unknown_upper_bound_with_techniques(
        formula_id: impl Into<String>,
        reason: UnknownReason,
        techniques: Techniques,
    ) -> Self {
        Self::unknown_with_techniques(formula_id, MccValueKind::UpperBound, reason, techniques)
    }

    /// Build an exact StateSpace record.
    #[must_use]
    pub(crate) fn exact_state_space(
        evidence: EvidenceSet,
        states: usize,
        edges: u64,
        max_token_in_place: u64,
        max_token_sum: u64,
    ) -> Self {
        Self::exact(
            "StateSpace",
            MccExactValue::StateSpace(StateSpaceReport::new(
                states,
                edges,
                max_token_in_place,
                max_token_sum,
            )),
            evidence,
        )
    }

    /// Build an unknown StateSpace record.
    #[must_use]
    pub(crate) fn unknown_state_space(reason: UnknownReason) -> Self {
        Self::unknown("StateSpace", MccValueKind::StateSpace, reason)
    }

    /// Build an unknown StateSpace record with technique provenance.
    #[must_use]
    pub(crate) fn unknown_state_space_with_techniques(
        reason: UnknownReason,
        techniques: Techniques,
    ) -> Self {
        Self::unknown_with_techniques("StateSpace", MccValueKind::StateSpace, reason, techniques)
    }

    /// Adapt the certified record to the existing public record type.
    #[must_use]
    pub(crate) fn to_legacy_record(&self) -> ExaminationRecord {
        let techniques = match &self.outcome {
            ExactOrUnknown::Exact(exact) => exact.evidence.techniques.clone(),
            ExactOrUnknown::Unknown(unknown) => unknown.techniques.clone(),
        };

        let value = match (&self.outcome, self.kind) {
            (ExactOrUnknown::Exact(exact), _) => match &exact.value {
                MccExactValue::Bool(value) => ExaminationValue::Verdict((*value).into()),
                MccExactValue::UpperBound(value) => ExaminationValue::OptionalBound(Some(*value)),
                MccExactValue::StateSpace(stats) => {
                    ExaminationValue::StateSpace(Some(stats.clone()))
                }
            },
            (ExactOrUnknown::Unknown(_), MccValueKind::Bool) => {
                ExaminationValue::Verdict(Verdict::CannotCompute)
            }
            (ExactOrUnknown::Unknown(_), MccValueKind::UpperBound) => {
                ExaminationValue::OptionalBound(None)
            }
            (ExactOrUnknown::Unknown(_), MccValueKind::StateSpace) => {
                ExaminationValue::StateSpace(None)
            }
        };

        ExaminationRecord::with_techniques(self.formula_id.clone(), value, techniques)
    }

    /// Render using the existing MCC formatter through the compatibility adapter.
    #[must_use]
    pub(crate) fn to_mcc_line(&self) -> String {
        self.to_legacy_record().to_mcc_line()
    }
}
