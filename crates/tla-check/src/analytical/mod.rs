// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Analytical checker outcome scaffolding.
//!
//! Analytical engines report candidates, not portfolio verdicts. A candidate
//! violation must be replayed before it can publish `Violated`, and a candidate
//! proof must be verified before it can publish `Satisfied`. Unknown and
//! ineligible results intentionally leave the shared verdict unresolved.

pub mod affine_record_counter;
pub mod bound_context;
pub mod finite_cardinality;
pub mod interval_counter;

use crate::shared_verdict::{CertificateVerification, SharedVerdict, Verdict};

/// Analytical engine output before proof verification or trace replay.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum AnalyticalOutcome<V, P> {
    /// The engine found a possible counterexample that still requires replay.
    CandidateViolation(CandidateViolation<V>),
    /// The engine found a possible proof that still requires certificate checks.
    CandidateProof(CandidateProof<P>),
    /// The engine ran but could not produce a definitive analytical result.
    Unknown(Unknown),
    /// The property/spec shape is outside the analytical engine's scope.
    Ineligible(Ineligible),
}

impl<V, P> AnalyticalOutcome<V, P> {
    /// Return the no-publish gate action required for this raw analytical outcome.
    pub fn required_gate_action(&self) -> GateDecision {
        match self {
            Self::CandidateViolation(_) => GateDecision::NeedsReplay,
            Self::CandidateProof(_) => GateDecision::NeedsProof,
            Self::Unknown(_) => GateDecision::DeferredUnknown,
            Self::Ineligible(_) => GateDecision::SkippedIneligible,
        }
    }
}

/// Analytical output after the proof/replay admission gate.
///
/// This type is for code paths that have already performed the checker-side
/// validation needed to make an analytical result publishable. It intentionally
/// has no candidate variants: a caller that receives `VerifiedProof` or
/// `ReplayedViolation` can pass it to [`VerificationGate`], while unsupported
/// shapes stay inconclusive.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum AnalyticalAdmission<V, P> {
    /// A violation whose trace has been replayed by the normal executor.
    ReplayedViolation(ReplayedViolation<V>),
    /// A satisfaction proof whose certificate has been verified.
    VerifiedProof(VerifiedProof<P>),
    /// The analytical path ran but could not prove or replay a result.
    Unknown(Unknown),
    /// The property/spec shape is outside the analytical path's scope.
    Ineligible(Ineligible),
}

impl<V, P> AnalyticalAdmission<V, P> {
    /// Whether this admitted result is ready to publish through a gate.
    pub fn is_publishable(&self) -> bool {
        matches!(self, Self::ReplayedViolation(_) | Self::VerifiedProof(_))
    }
}

/// A possible violation emitted by an analytical engine.
///
/// This is intentionally not publishable. Convert it to a [`ReplayedViolation`]
/// only after checker replay validates the counterexample.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CandidateViolation<T> {
    candidate: T,
}

impl<T> CandidateViolation<T> {
    /// Create a new violation candidate.
    pub fn new(candidate: T) -> Self {
        Self { candidate }
    }

    /// Borrow the candidate payload for replay.
    pub fn candidate(&self) -> &T {
        &self.candidate
    }

    /// Consume this wrapper and return the candidate payload.
    pub fn into_candidate(self) -> T {
        self.candidate
    }
}

/// A possible satisfaction proof emitted by an analytical engine.
///
/// This is intentionally not publishable. Convert it to a [`VerifiedProof`]
/// only after a proof checker validates the certificate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CandidateProof<T> {
    certificate: T,
}

impl<T> CandidateProof<T> {
    /// Create a new proof candidate.
    pub fn new(certificate: T) -> Self {
        Self { certificate }
    }

    /// Borrow the candidate certificate for proof verification.
    pub fn certificate(&self) -> &T {
        &self.certificate
    }

    /// Consume this wrapper and return the candidate certificate.
    pub fn into_certificate(self) -> T {
        self.certificate
    }
}

/// Analytical engine reached an inconclusive result.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Unknown {
    reason: String,
}

impl Unknown {
    /// Create an inconclusive analytical outcome with a diagnostic reason.
    pub fn new(reason: impl Into<String>) -> Self {
        Self {
            reason: reason.into(),
        }
    }

    /// Human-readable reason the engine could not decide.
    pub fn reason(&self) -> &str {
        &self.reason
    }
}

/// Analytical engine determined that the input is outside its supported subset.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Ineligible {
    reason: String,
}

impl Ineligible {
    /// Create an ineligible analytical outcome with a diagnostic reason.
    pub fn new(reason: impl Into<String>) -> Self {
        Self {
            reason: reason.into(),
        }
    }

    /// Human-readable reason the engine declined the input.
    pub fn reason(&self) -> &str {
        &self.reason
    }
}

/// A violation candidate after checker replay has validated the trace.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReplayedViolation<T> {
    evidence: T,
}

impl<T> ReplayedViolation<T> {
    /// Mark a replay-validated violation candidate as publishable.
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn new(evidence: T) -> Self {
        Self { evidence }
    }

    /// Borrow the replay evidence.
    pub fn evidence(&self) -> &T {
        &self.evidence
    }

    /// Consume this wrapper and return the replay evidence.
    pub fn into_evidence(self) -> T {
        self.evidence
    }
}

/// A proof candidate after certificate verification has succeeded.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifiedProof<T> {
    certificate: T,
}

impl<T> VerifiedProof<T> {
    /// Mark a verified proof certificate as publishable.
    pub(crate) fn new(certificate: T) -> Self {
        Self { certificate }
    }

    /// Borrow the verified proof certificate.
    pub fn certificate(&self) -> &T {
        &self.certificate
    }

    /// Consume this wrapper and return the verified proof certificate.
    pub fn into_certificate(self) -> T {
        self.certificate
    }
}

/// Result of passing analytical output through the verification gate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GateDecision {
    /// A replayed violation or verified proof resolved the shared verdict.
    Published,
    /// A verified result reached the gate after another lane had already won.
    AlreadyResolved,
    /// A violation candidate must be replayed before publication.
    NeedsReplay,
    /// A proof candidate must be verified before publication.
    NeedsProof,
    /// Unknown analytical output leaves other lanes running.
    DeferredUnknown,
    /// Ineligible analytical output leaves other lanes running.
    SkippedIneligible,
}

impl GateDecision {
    /// Whether this decision published a portfolio verdict.
    pub fn published(self) -> bool {
        self == Self::Published
    }
}

/// Verification gate between analytical candidates and shared portfolio verdicts.
///
/// The gate has no API for publishing [`CandidateViolation`], [`CandidateProof`],
/// [`Unknown`], or [`Ineligible`] directly. Only replay-validated violations and
/// proof-verified certificates can reach [`SharedVerdict`].
#[derive(Debug)]
pub struct VerificationGate<'a> {
    verdict: &'a SharedVerdict,
}

impl<'a> VerificationGate<'a> {
    /// Create a gate backed by a shared portfolio verdict slot.
    pub fn new(verdict: &'a SharedVerdict) -> Self {
        Self { verdict }
    }

    /// Inspect a raw analytical outcome without publishing it.
    pub fn inspect<V, P>(&self, outcome: &AnalyticalOutcome<V, P>) -> GateDecision {
        outcome.required_gate_action()
    }

    /// Publish a replay-validated violation.
    pub fn publish_replayed_violation<T>(&self, _violation: ReplayedViolation<T>) -> GateDecision {
        if self
            .verdict
            .publish_analytical(Verdict::Violated, CertificateVerification::MissingVerifier)
        {
            GateDecision::Published
        } else {
            GateDecision::AlreadyResolved
        }
    }

    /// Publish a certificate-verified proof.
    pub fn publish_verified_proof<T>(&self, _proof: VerifiedProof<T>) -> GateDecision {
        if self
            .verdict
            .publish_analytical(Verdict::Satisfied, CertificateVerification::Verified)
        {
            GateDecision::Published
        } else {
            GateDecision::AlreadyResolved
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn candidate_violation_requires_replay_and_does_not_publish() {
        let shared = SharedVerdict::new();
        let gate = VerificationGate::new(&shared);
        let outcome: AnalyticalOutcome<&str, &str> =
            AnalyticalOutcome::CandidateViolation(CandidateViolation::new("trace-candidate"));

        assert_eq!(gate.inspect(&outcome), GateDecision::NeedsReplay);
        assert!(!shared.is_resolved());
        assert_eq!(shared.get(), None);
    }

    #[test]
    fn candidate_proof_requires_verification_and_does_not_publish() {
        let shared = SharedVerdict::new();
        let gate = VerificationGate::new(&shared);
        let outcome: AnalyticalOutcome<&str, &str> =
            AnalyticalOutcome::CandidateProof(CandidateProof::new("proof-candidate"));

        assert_eq!(gate.inspect(&outcome), GateDecision::NeedsProof);
        assert!(!shared.is_resolved());
        assert_eq!(shared.get(), None);
    }

    #[test]
    fn unknown_and_ineligible_do_not_publish() {
        let shared = SharedVerdict::new();
        let gate = VerificationGate::new(&shared);
        let unknown: AnalyticalOutcome<(), ()> =
            AnalyticalOutcome::Unknown(Unknown::new("timeout"));
        let ineligible: AnalyticalOutcome<(), ()> =
            AnalyticalOutcome::Ineligible(Ineligible::new("unsupported operator"));

        assert_eq!(gate.inspect(&unknown), GateDecision::DeferredUnknown);
        assert_eq!(gate.inspect(&ineligible), GateDecision::SkippedIneligible);
        assert!(!shared.is_resolved());
        assert_eq!(shared.get(), None);
    }

    #[test]
    fn replayed_violation_can_publish_violated() {
        let shared = SharedVerdict::new();
        let gate = VerificationGate::new(&shared);

        let decision = gate.publish_replayed_violation(ReplayedViolation::new("replayed-trace"));

        assert_eq!(decision, GateDecision::Published);
        assert!(decision.published());
        assert_eq!(shared.get(), Some(Verdict::Violated));
    }

    #[test]
    fn verified_proof_can_publish_satisfied() {
        let shared = SharedVerdict::new();
        let gate = VerificationGate::new(&shared);

        let decision = gate.publish_verified_proof(VerifiedProof::new("checked-certificate"));

        assert_eq!(decision, GateDecision::Published);
        assert!(decision.published());
        assert_eq!(shared.get(), Some(Verdict::Satisfied));
    }

    #[test]
    fn verified_publication_respects_first_writer_wins() {
        let shared = SharedVerdict::new();
        let gate = VerificationGate::new(&shared);

        assert_eq!(
            gate.publish_verified_proof(VerifiedProof::new("checked-certificate")),
            GateDecision::Published
        );
        assert_eq!(
            gate.publish_replayed_violation(ReplayedViolation::new("replayed-trace")),
            GateDecision::AlreadyResolved
        );
        assert_eq!(shared.get(), Some(Verdict::Satisfied));
    }
}
