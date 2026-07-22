// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Proof-gated analytical recognizer for one-variable interval counters.
//!
//! This module is intentionally not wired into the default checker. It provides
//! a narrow structural recognizer plus an independent certificate verifier. The
//! recognizer may emit [`CandidateProof`], but only the verifier can construct
//! [`VerifiedProof`].

use num_bigint::BigInt;
use num_traits::{One, Zero};
use std::collections::{BTreeMap, BTreeSet};
use tla_core::ast::{Expr, Module, OperatorDef, Unit};
use tla_core::span::Spanned;

use super::{
    AnalyticalAdmission, AnalyticalOutcome, CandidateProof, Ineligible, Unknown, VerifiedProof,
};

/// Outcome type used by the interval-counter analytical recognizer.
pub type IntervalCounterOutcome = AnalyticalOutcome<(), IntervalCounterCertificate>;

/// Outcome type used by the independent interval-counter analytical recognizer.
pub type IndependentIntervalCounterOutcome =
    AnalyticalOutcome<(), IndependentIntervalCounterCertificate>;

/// Proof/replay-gated admission result for bounded interval counters.
pub type IntervalCounterAdmissionOutcome =
    AnalyticalAdmission<(), IntervalCounterAdmissionCertificate>;

/// Proof-gated execution-model fast-path result for bounded interval counters.
pub type IntervalCounterExecutionOutcome =
    AnalyticalAdmission<(), IntervalCounterExecutionCertificate>;

/// Closed finite integer interval.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IntegerInterval {
    lower: BigInt,
    upper: BigInt,
}

impl IntegerInterval {
    /// Create a closed interval when `lower <= upper`.
    pub fn new(lower: BigInt, upper: BigInt) -> Option<Self> {
        if lower <= upper {
            Some(Self { lower, upper })
        } else {
            None
        }
    }

    /// Lower inclusive bound.
    pub fn lower(&self) -> &BigInt {
        &self.lower
    }

    /// Upper inclusive bound.
    pub fn upper(&self) -> &BigInt {
        &self.upper
    }

    /// Number of integers in the interval.
    pub fn cardinality(&self) -> BigInt {
        &self.upper - &self.lower + BigInt::one()
    }

    fn contains_interval(&self, other: &Self) -> bool {
        self.lower <= other.lower && other.upper <= self.upper
    }

    fn shift(&self, delta: &BigInt) -> Self {
        Self {
            lower: &self.lower + delta,
            upper: &self.upper + delta,
        }
    }

    fn singleton(value: BigInt) -> Self {
        Self {
            lower: value.clone(),
            upper: value,
        }
    }
}

/// A guard interval that can be intersected with the invariant interval.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct IntervalGuard {
    lower: Option<BigInt>,
    upper: Option<BigInt>,
}

impl IntervalGuard {
    /// Create an unconstrained guard.
    pub fn unconstrained() -> Self {
        Self::default()
    }

    /// Lower inclusive guard bound, if any.
    pub fn lower(&self) -> Option<&BigInt> {
        self.lower.as_ref()
    }

    /// Upper inclusive guard bound, if any.
    pub fn upper(&self) -> Option<&BigInt> {
        self.upper.as_ref()
    }

    fn intersect_finite(&self, interval: &IntegerInterval) -> Option<IntegerInterval> {
        let lower = match &self.lower {
            Some(bound) => bound.max(&interval.lower).clone(),
            None => interval.lower.clone(),
        };
        let upper = match &self.upper {
            Some(bound) => bound.min(&interval.upper).clone(),
            None => interval.upper.clone(),
        };
        IntegerInterval::new(lower, upper)
    }

    fn constrain_lower(&mut self, bound: BigInt) {
        match &mut self.lower {
            Some(existing) if *existing < bound => *existing = bound,
            None => self.lower = Some(bound),
            _ => {}
        }
    }

    fn constrain_upper(&mut self, bound: BigInt) {
        match &mut self.upper {
            Some(existing) if *existing > bound => *existing = bound,
            None => self.upper = Some(bound),
            _ => {}
        }
    }

    fn constrain_eq(&mut self, value: BigInt) {
        self.constrain_lower(value.clone());
        self.constrain_upper(value);
    }
}

/// Supported counter update.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CounterUpdate {
    /// `x' = x + delta`, with `delta = 0` representing stutter.
    Shift(BigInt),
    /// `x' = constant`.
    Set(BigInt),
    /// `x' \in lo..hi`, a finite nondeterministic update.
    Choose(IntegerInterval),
}

impl CounterUpdate {
    fn image(&self, source: &IntegerInterval) -> IntegerInterval {
        match self {
            Self::Shift(delta) => source.shift(delta),
            Self::Set(value) => IntegerInterval::singleton(value.clone()),
            Self::Choose(interval) => interval.clone(),
        }
    }
}

/// One disjunctive transition branch in the interval-counter certificate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CounterTransitionBranch {
    guard: IntervalGuard,
    update: CounterUpdate,
}

impl CounterTransitionBranch {
    /// Source-state guard recognized from this branch.
    pub fn guard(&self) -> &IntervalGuard {
        &self.guard
    }

    /// Counter update recognized from this branch.
    pub fn update(&self) -> &CounterUpdate {
        &self.update
    }
}

/// Certificate emitted by the interval-counter structural recognizer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IntervalCounterCertificate {
    variable: String,
    init_interval: IntegerInterval,
    invariant_interval: IntegerInterval,
    transition_branches: Vec<CounterTransitionBranch>,
}

/// Product certificate for structurally independent interval counters.
///
/// Each counter is verified independently. The recognizer rejects transition
/// components that mention multiple primed counters, so the product proof is the
/// conjunction of the per-counter induction obligations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IndependentIntervalCounterCertificate {
    counters: Vec<IntervalCounterCertificate>,
}

/// Verified interval-counter certificate with enough source metadata for later replay wiring.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IntervalCounterAdmissionCertificate {
    init_operator: String,
    next_operator: String,
    invariant_operator: String,
    proof: IntervalCounterAdmissionProof,
}

impl IntervalCounterAdmissionCertificate {
    /// INIT operator name supplied by the caller/config.
    pub fn init_operator(&self) -> &str {
        &self.init_operator
    }

    /// NEXT operator name supplied by the caller/config.
    pub fn next_operator(&self) -> &str {
        &self.next_operator
    }

    /// Invariant operator name supplied by the caller/config.
    pub fn invariant_operator(&self) -> &str {
        &self.invariant_operator
    }

    /// Verified proof payload.
    pub fn proof(&self) -> &IntervalCounterAdmissionProof {
        &self.proof
    }

    /// Number of independent counter dimensions covered by the proof.
    pub fn counter_count(&self) -> usize {
        self.proof.counter_count()
    }

    /// Cardinality of the proved invariant interval or product box.
    pub fn invariant_cardinality(&self) -> BigInt {
        self.proof.invariant_cardinality()
    }

    /// Whether `Init` admits exactly the full proved invariant box.
    pub fn initial_states_cover_invariant(&self) -> bool {
        self.proof.initial_states_cover_invariant()
    }

    /// Whether transition guards cover every state in the proved invariant box.
    pub fn transition_total_on_invariant(&self) -> bool {
        self.proof.transition_total_on_invariant()
    }
}

/// Verified proof that analytical execution can replace explicit exploration.
///
/// This is stronger than invariant admission. It only succeeds when the exact
/// reachable set is the proved finite invariant box. If the caller still needs
/// deadlock checking, transition guards must also cover the whole box.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IntervalCounterExecutionCertificate {
    admission: IntervalCounterAdmissionCertificate,
    state_count: BigInt,
    deadlock_free: bool,
}

impl IntervalCounterExecutionCertificate {
    /// Underlying verified invariant proof and source operator metadata.
    pub fn admission(&self) -> &IntervalCounterAdmissionCertificate {
        &self.admission
    }

    /// Exact number of states in the proved reachable box.
    pub fn state_count(&self) -> &BigInt {
        &self.state_count
    }

    /// Whether the certificate also proves no deadlocks in the proved box.
    pub fn deadlock_free(&self) -> bool {
        self.deadlock_free
    }
}

/// Verified bounded-counter proof shape admitted by this analytical path.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IntervalCounterAdmissionProof {
    /// One bounded interval counter.
    Single(IntervalCounterCertificate),
    /// Product of independent bounded interval counters.
    Independent(IndependentIntervalCounterCertificate),
}

impl IntervalCounterAdmissionProof {
    /// Number of independent counter dimensions covered by the proof.
    pub fn counter_count(&self) -> usize {
        match self {
            Self::Single(_) => 1,
            Self::Independent(certificate) => certificate.counter_count(),
        }
    }

    /// Cardinality of the proved invariant interval or product box.
    pub fn invariant_cardinality(&self) -> BigInt {
        match self {
            Self::Single(certificate) => certificate.invariant_cardinality(),
            Self::Independent(certificate) => certificate.invariant_cardinality(),
        }
    }

    fn initial_states_cover_invariant(&self) -> bool {
        match self {
            Self::Single(certificate) => certificate.initial_states_cover_invariant(),
            Self::Independent(certificate) => certificate.initial_states_cover_invariant(),
        }
    }

    fn transition_total_on_invariant(&self) -> bool {
        match self {
            Self::Single(certificate) => certificate.transition_total_on_invariant(),
            Self::Independent(certificate) => certificate.transition_total_on_invariant(),
        }
    }

    fn first_counter_without_full_init(&self) -> Option<&IntervalCounterCertificate> {
        match self {
            Self::Single(certificate) => {
                (!certificate.initial_states_cover_invariant()).then_some(certificate)
            }
            Self::Independent(certificate) => certificate
                .counters
                .iter()
                .find(|counter| !counter.initial_states_cover_invariant()),
        }
    }

    fn first_counter_without_total_transition(&self) -> Option<&IntervalCounterCertificate> {
        match self {
            Self::Single(certificate) => {
                (!certificate.transition_total_on_invariant()).then_some(certificate)
            }
            Self::Independent(certificate) => certificate
                .counters
                .iter()
                .find(|counter| !counter.transition_total_on_invariant()),
        }
    }
}

impl IndependentIntervalCounterCertificate {
    /// Per-counter certificates in source/init order.
    pub fn counters(&self) -> &[IntervalCounterCertificate] {
        &self.counters
    }

    /// Number of independent counters covered by this certificate.
    pub fn counter_count(&self) -> usize {
        self.counters.len()
    }

    /// Product cardinality of the invariant box.
    pub fn invariant_cardinality(&self) -> BigInt {
        self.counters.iter().fold(BigInt::one(), |acc, counter| {
            acc * counter.invariant_cardinality()
        })
    }

    /// Whether every counter starts with its full proved invariant interval.
    pub fn initial_states_cover_invariant(&self) -> bool {
        self.counters
            .iter()
            .all(IntervalCounterCertificate::initial_states_cover_invariant)
    }

    /// Whether every counter has total transition coverage over its invariant.
    pub fn transition_total_on_invariant(&self) -> bool {
        self.counters
            .iter()
            .all(IntervalCounterCertificate::transition_total_on_invariant)
    }
}

impl IntervalCounterCertificate {
    /// Counter variable name.
    pub fn variable(&self) -> &str {
        &self.variable
    }

    /// Initial interval for the counter.
    pub fn init_interval(&self) -> &IntegerInterval {
        &self.init_interval
    }

    /// Claimed invariant interval for the counter.
    pub fn invariant_interval(&self) -> &IntegerInterval {
        &self.invariant_interval
    }

    /// Disjunctive transition branches covered by this certificate.
    pub fn transition_branches(&self) -> &[CounterTransitionBranch] {
        &self.transition_branches
    }

    /// Size of the invariant interval covered by the certificate.
    pub fn invariant_cardinality(&self) -> BigInt {
        self.invariant_interval.cardinality()
    }

    /// Whether the initial interval is exactly the proved invariant interval.
    pub fn initial_states_cover_invariant(&self) -> bool {
        self.init_interval == self.invariant_interval
    }

    /// Whether transition guards cover every state in the invariant interval.
    pub fn transition_total_on_invariant(&self) -> bool {
        guards_cover_interval(&self.transition_branches, &self.invariant_interval)
    }
}

/// Verification error for a structurally recognized interval-counter proof.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IntervalCounterVerificationError {
    /// Initial states are not covered by the claimed invariant.
    InitOutsideInvariant {
        /// Interval of initial states.
        init: IntegerInterval,
        /// The claimed invariant interval.
        invariant: IntegerInterval,
    },
    /// A transition branch can leave the claimed invariant.
    BranchEscapesInvariant {
        /// Index of the offending transition branch.
        branch_index: usize,
        /// Source-state interval the branch was evaluated over.
        source: IntegerInterval,
        /// Interval the branch's successor can reach.
        image: IntegerInterval,
        /// The claimed invariant interval.
        invariant: IntegerInterval,
    },
}

/// Verification error for an independent interval-counter product proof.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IndependentIntervalCounterVerificationError {
    /// The product certificate contains no counters.
    EmptyProduct,
    /// A counter variable appears more than once in the product certificate.
    DuplicateVariable(String),
    /// One per-counter proof obligation failed.
    Counter {
        /// The counter variable whose proof failed.
        variable: String,
        /// The underlying per-counter verification error.
        source: IntervalCounterVerificationError,
    },
}

/// Verify an interval-counter candidate proof and return a publishable proof.
pub fn verify_interval_counter_candidate(
    candidate: CandidateProof<IntervalCounterCertificate>,
) -> Result<VerifiedProof<IntervalCounterCertificate>, IntervalCounterVerificationError> {
    let certificate = candidate.into_certificate();
    verify_certificate(&certificate)?;
    Ok(VerifiedProof::new(certificate))
}

/// Verify an independent interval-counter candidate proof and return a publishable proof.
pub fn verify_independent_interval_counter_candidate(
    candidate: CandidateProof<IndependentIntervalCounterCertificate>,
) -> Result<
    VerifiedProof<IndependentIntervalCounterCertificate>,
    IndependentIntervalCounterVerificationError,
> {
    let certificate = candidate.into_certificate();
    verify_independent_certificate(&certificate)?;
    Ok(VerifiedProof::new(certificate))
}

/// Admit a bounded interval-counter proof only after certificate verification.
///
/// This is the first proof-gated analytical admission path. It uses structural
/// AST matching only: operator names are caller/config lookup keys, not
/// shortcuts. A successful result contains a [`VerifiedProof`] that can be
/// published through [`super::VerificationGate`]. Unsupported shapes return
/// `Ineligible` or `Unknown`; raw candidates are never returned from this API.
pub fn admit_module_interval_counters(
    module: &Module,
    init_operator: &str,
    next_operator: &str,
    invariant_operator: &str,
) -> IntervalCounterAdmissionOutcome {
    match recognize_module_independent_interval_counters(
        module,
        init_operator,
        next_operator,
        invariant_operator,
    ) {
        AnalyticalOutcome::CandidateProof(candidate) => {
            return admit_independent_candidate(
                candidate,
                init_operator,
                next_operator,
                invariant_operator,
            );
        }
        AnalyticalOutcome::Unknown(reason) => return AnalyticalAdmission::Unknown(reason),
        AnalyticalOutcome::CandidateViolation(_) => {
            return AnalyticalAdmission::Unknown(Unknown::new(
                "interval-counter recognizer emitted unsupported violation candidate",
            ));
        }
        AnalyticalOutcome::Ineligible(_) => {}
    }

    match recognize_module_interval_counter(
        module,
        init_operator,
        next_operator,
        invariant_operator,
    ) {
        AnalyticalOutcome::CandidateProof(candidate) => {
            admit_single_candidate(candidate, init_operator, next_operator, invariant_operator)
        }
        AnalyticalOutcome::Unknown(reason) => AnalyticalAdmission::Unknown(reason),
        AnalyticalOutcome::Ineligible(reason) => AnalyticalAdmission::Ineligible(reason),
        AnalyticalOutcome::CandidateViolation(_) => AnalyticalAdmission::Unknown(Unknown::new(
            "interval-counter recognizer emitted unsupported violation candidate",
        )),
    }
}

/// Admit an interval-counter execution fast path after all proof obligations pass.
///
/// The invariant proof alone can prove safety but cannot replace BFS state
/// counting. This path only succeeds when `Init` covers exactly the proved
/// finite box. If the caller requires deadlock checking, branch guards must also
/// cover every state in that box.
pub fn admit_module_interval_counter_execution_model(
    module: &Module,
    init_operator: &str,
    next_operator: &str,
    invariant_operator: &str,
    require_deadlock_freedom: bool,
) -> IntervalCounterExecutionOutcome {
    let admission = match admit_module_interval_counters(
        module,
        init_operator,
        next_operator,
        invariant_operator,
    ) {
        AnalyticalAdmission::VerifiedProof(verified) => verified.into_certificate(),
        AnalyticalAdmission::Unknown(reason) => return AnalyticalAdmission::Unknown(reason),
        AnalyticalAdmission::Ineligible(reason) => return AnalyticalAdmission::Ineligible(reason),
        AnalyticalAdmission::ReplayedViolation(_) => {
            return AnalyticalAdmission::Unknown(Unknown::new(
                "interval-counter admission emitted unsupported replayed violation",
            ));
        }
    };

    if let Some(counter) = admission.proof.first_counter_without_full_init() {
        return AnalyticalAdmission::Ineligible(Ineligible::new(format!(
            "analytical execution fast path requires Init to cover the full invariant interval for counter `{}`",
            counter.variable()
        )));
    }

    if require_deadlock_freedom {
        if let Some(counter) = admission.proof.first_counter_without_total_transition() {
            return AnalyticalAdmission::Ineligible(Ineligible::new(format!(
                "analytical execution fast path cannot prove deadlock freedom for counter `{}`",
                counter.variable()
            )));
        }
    }

    let state_count = admission.invariant_cardinality();
    let deadlock_free = admission.transition_total_on_invariant();
    AnalyticalAdmission::VerifiedProof(VerifiedProof::new(IntervalCounterExecutionCertificate {
        admission,
        state_count,
        deadlock_free,
    }))
}

fn admit_single_candidate(
    candidate: CandidateProof<IntervalCounterCertificate>,
    init_operator: &str,
    next_operator: &str,
    invariant_operator: &str,
) -> IntervalCounterAdmissionOutcome {
    match verify_interval_counter_candidate(candidate) {
        Ok(verified) => {
            let certificate = IntervalCounterAdmissionCertificate {
                init_operator: init_operator.to_string(),
                next_operator: next_operator.to_string(),
                invariant_operator: invariant_operator.to_string(),
                proof: IntervalCounterAdmissionProof::Single(verified.into_certificate()),
            };
            AnalyticalAdmission::VerifiedProof(VerifiedProof::new(certificate))
        }
        Err(err) => AnalyticalAdmission::Unknown(Unknown::new(format!(
            "interval-counter certificate verification failed: {err:?}"
        ))),
    }
}

fn admit_independent_candidate(
    candidate: CandidateProof<IndependentIntervalCounterCertificate>,
    init_operator: &str,
    next_operator: &str,
    invariant_operator: &str,
) -> IntervalCounterAdmissionOutcome {
    match verify_independent_interval_counter_candidate(candidate) {
        Ok(verified) => {
            let certificate = IntervalCounterAdmissionCertificate {
                init_operator: init_operator.to_string(),
                next_operator: next_operator.to_string(),
                invariant_operator: invariant_operator.to_string(),
                proof: IntervalCounterAdmissionProof::Independent(verified.into_certificate()),
            };
            AnalyticalAdmission::VerifiedProof(VerifiedProof::new(certificate))
        }
        Err(err) => AnalyticalAdmission::Unknown(Unknown::new(format!(
            "independent interval-counter certificate verification failed: {err:?}"
        ))),
    }
}

/// Recognize a module by caller-provided operator names.
///
/// Operator names are only lookup keys supplied by the caller, typically from
/// configuration. The recognizer does not inspect the module name or use any
/// spec-name shortcut.
pub fn recognize_module_interval_counter(
    module: &Module,
    init_operator: &str,
    next_operator: &str,
    invariant_operator: &str,
) -> IntervalCounterOutcome {
    let init = match find_zero_arity_operator(module, init_operator) {
        Ok(op) => op,
        Err(reason) => return AnalyticalOutcome::Ineligible(Ineligible::new(reason)),
    };
    let next = match find_zero_arity_operator(module, next_operator) {
        Ok(op) => op,
        Err(reason) => return AnalyticalOutcome::Ineligible(Ineligible::new(reason)),
    };
    let invariant = match find_zero_arity_operator(module, invariant_operator) {
        Ok(op) => op,
        Err(reason) => return AnalyticalOutcome::Ineligible(Ineligible::new(reason)),
    };

    recognize_interval_counter_ops(init, next, invariant)
}

/// Recognize independent interval counters by caller-provided operator names.
///
/// Operator names are only lookup keys supplied by the caller. The recognizer
/// uses no module-name or spec-name shortcut.
pub fn recognize_module_independent_interval_counters(
    module: &Module,
    init_operator: &str,
    next_operator: &str,
    invariant_operator: &str,
) -> IndependentIntervalCounterOutcome {
    let init = match find_zero_arity_operator(module, init_operator) {
        Ok(op) => op,
        Err(reason) => return AnalyticalOutcome::Ineligible(Ineligible::new(reason)),
    };
    let next = match find_zero_arity_operator(module, next_operator) {
        Ok(op) => op,
        Err(reason) => return AnalyticalOutcome::Ineligible(Ineligible::new(reason)),
    };
    let invariant = match find_zero_arity_operator(module, invariant_operator) {
        Ok(op) => op,
        Err(reason) => return AnalyticalOutcome::Ineligible(Ineligible::new(reason)),
    };

    recognize_independent_interval_counter_ops(init, next, invariant)
}

/// Recognize a one-variable interval-counter proof candidate from operator bodies.
pub fn recognize_interval_counter_ops(
    init: &OperatorDef,
    next: &OperatorDef,
    invariant: &OperatorDef,
) -> IntervalCounterOutcome {
    if !init.params.is_empty() || !next.params.is_empty() || !invariant.params.is_empty() {
        return AnalyticalOutcome::Ineligible(Ineligible::new(
            "interval-counter operators must be zero-arity",
        ));
    }

    recognize_interval_counter_exprs(&init.body, &next.body, &invariant.body)
}

/// Recognize structurally independent interval counters from operator bodies.
pub fn recognize_independent_interval_counter_ops(
    init: &OperatorDef,
    next: &OperatorDef,
    invariant: &OperatorDef,
) -> IndependentIntervalCounterOutcome {
    if !init.params.is_empty() || !next.params.is_empty() || !invariant.params.is_empty() {
        return AnalyticalOutcome::Ineligible(Ineligible::new(
            "independent interval-counter operators must be zero-arity",
        ));
    }

    recognize_independent_interval_counter_exprs(&init.body, &next.body, &invariant.body)
}

/// Recognize a one-variable interval-counter proof candidate from AST bodies.
pub fn recognize_interval_counter_exprs(
    init: &Spanned<Expr>,
    next: &Spanned<Expr>,
    invariant: &Spanned<Expr>,
) -> IntervalCounterOutcome {
    let (init_var, init_interval) = match recognize_interval_membership(init) {
        RecognizeResult::Ok(value) => value,
        RecognizeResult::Ineligible(reason) => {
            return AnalyticalOutcome::Ineligible(Ineligible::new(reason));
        }
        RecognizeResult::Unknown(reason) => {
            return AnalyticalOutcome::Unknown(Unknown::new(reason))
        }
    };
    let (inv_var, invariant_interval) = match recognize_interval_membership(invariant) {
        RecognizeResult::Ok(value) => value,
        RecognizeResult::Ineligible(reason) => {
            return AnalyticalOutcome::Ineligible(Ineligible::new(reason));
        }
        RecognizeResult::Unknown(reason) => {
            return AnalyticalOutcome::Unknown(Unknown::new(reason))
        }
    };
    if init_var != inv_var {
        return AnalyticalOutcome::Ineligible(Ineligible::new(
            "init and invariant refer to different counter variables",
        ));
    }

    let transition_branches = match recognize_transition(next, &init_var) {
        RecognizeResult::Ok(branches) => branches,
        RecognizeResult::Ineligible(reason) => {
            return AnalyticalOutcome::Ineligible(Ineligible::new(reason));
        }
        RecognizeResult::Unknown(reason) => {
            return AnalyticalOutcome::Unknown(Unknown::new(reason))
        }
    };

    AnalyticalOutcome::CandidateProof(CandidateProof::new(IntervalCounterCertificate {
        variable: init_var,
        init_interval,
        invariant_interval,
        transition_branches,
    }))
}

/// Recognize a product proof candidate for independent interval counters.
pub fn recognize_independent_interval_counter_exprs(
    init: &Spanned<Expr>,
    next: &Spanned<Expr>,
    invariant: &Spanned<Expr>,
) -> IndependentIntervalCounterOutcome {
    let init_intervals = match recognize_interval_memberships(init) {
        RecognizeResult::Ok(value) => value,
        RecognizeResult::Ineligible(reason) => {
            return AnalyticalOutcome::Ineligible(Ineligible::new(reason));
        }
        RecognizeResult::Unknown(reason) => {
            return AnalyticalOutcome::Unknown(Unknown::new(reason))
        }
    };
    let invariant_intervals = match recognize_interval_memberships(invariant) {
        RecognizeResult::Ok(value) => value,
        RecognizeResult::Ineligible(reason) => {
            return AnalyticalOutcome::Ineligible(Ineligible::new(reason));
        }
        RecognizeResult::Unknown(reason) => {
            return AnalyticalOutcome::Unknown(Unknown::new(reason))
        }
    };

    if init_intervals.len() < 2 {
        return AnalyticalOutcome::Ineligible(Ineligible::new(
            "expected at least two independent interval counters",
        ));
    }

    let init_map = interval_map(&init_intervals);
    let invariant_map = interval_map(&invariant_intervals);
    if init_map.keys().ne(invariant_map.keys()) {
        return AnalyticalOutcome::Ineligible(Ineligible::new(
            "init and invariant mention different counter variables",
        ));
    }

    let variables: Vec<String> = init_intervals
        .iter()
        .map(|(variable, _)| variable.clone())
        .collect();
    let grouped_transitions = match group_independent_transition_components(next, &variables) {
        RecognizeResult::Ok(value) => value,
        RecognizeResult::Ineligible(reason) => {
            return AnalyticalOutcome::Ineligible(Ineligible::new(reason));
        }
        RecognizeResult::Unknown(reason) => {
            return AnalyticalOutcome::Unknown(Unknown::new(reason))
        }
    };

    let mut counters = Vec::with_capacity(variables.len());
    for variable in variables {
        let transition = match grouped_transitions.get(&variable) {
            Some(parts) => fold_and_components(parts),
            None => {
                return AnalyticalOutcome::Ineligible(Ineligible::new(format!(
                    "missing transition component for counter `{variable}`"
                )));
            }
        };
        let transition_branches = match recognize_transition(&transition, &variable) {
            RecognizeResult::Ok(branches) => branches,
            RecognizeResult::Ineligible(reason) => {
                return AnalyticalOutcome::Ineligible(Ineligible::new(reason));
            }
            RecognizeResult::Unknown(reason) => {
                return AnalyticalOutcome::Unknown(Unknown::new(reason))
            }
        };

        let init_interval = init_map
            .get(&variable)
            .expect("variable came from init map")
            .clone();
        let invariant_interval = invariant_map
            .get(&variable)
            .expect("variable came from invariant map")
            .clone();
        counters.push(IntervalCounterCertificate {
            variable,
            init_interval,
            invariant_interval,
            transition_branches,
        });
    }

    AnalyticalOutcome::CandidateProof(CandidateProof::new(IndependentIntervalCounterCertificate {
        counters,
    }))
}

/// Verify a recognized interval-counter certificate without publishing.
pub fn verify_certificate(
    certificate: &IntervalCounterCertificate,
) -> Result<(), IntervalCounterVerificationError> {
    if !certificate
        .invariant_interval
        .contains_interval(&certificate.init_interval)
    {
        return Err(IntervalCounterVerificationError::InitOutsideInvariant {
            init: certificate.init_interval.clone(),
            invariant: certificate.invariant_interval.clone(),
        });
    }

    for (branch_index, branch) in certificate.transition_branches.iter().enumerate() {
        let Some(source) = branch
            .guard
            .intersect_finite(&certificate.invariant_interval)
        else {
            continue;
        };
        let image = branch.update.image(&source);
        if !certificate.invariant_interval.contains_interval(&image) {
            return Err(IntervalCounterVerificationError::BranchEscapesInvariant {
                branch_index,
                source,
                image,
                invariant: certificate.invariant_interval.clone(),
            });
        }
    }

    Ok(())
}

/// Verify an independent interval-counter product certificate without publishing.
pub fn verify_independent_certificate(
    certificate: &IndependentIntervalCounterCertificate,
) -> Result<(), IndependentIntervalCounterVerificationError> {
    if certificate.counters.is_empty() {
        return Err(IndependentIntervalCounterVerificationError::EmptyProduct);
    }

    let mut seen = BTreeSet::new();
    for counter in &certificate.counters {
        if !seen.insert(counter.variable.clone()) {
            return Err(
                IndependentIntervalCounterVerificationError::DuplicateVariable(
                    counter.variable.clone(),
                ),
            );
        }
        verify_certificate(counter).map_err(|source| {
            IndependentIntervalCounterVerificationError::Counter {
                variable: counter.variable.clone(),
                source,
            }
        })?;
    }

    Ok(())
}

fn guards_cover_interval(branches: &[CounterTransitionBranch], interval: &IntegerInterval) -> bool {
    let mut covered: Vec<IntegerInterval> = branches
        .iter()
        .filter_map(|branch| branch.guard.intersect_finite(interval))
        .collect();
    covered.sort_by(|lhs, rhs| lhs.lower.cmp(&rhs.lower).then(lhs.upper.cmp(&rhs.upper)));

    let mut next_uncovered = interval.lower.clone();
    for covered_interval in covered {
        if covered_interval.upper < next_uncovered {
            continue;
        }
        if covered_interval.lower > next_uncovered {
            return false;
        }
        if covered_interval.upper >= interval.upper {
            return true;
        }
        next_uncovered = covered_interval.upper + BigInt::one();
    }

    next_uncovered > interval.upper
}

fn find_zero_arity_operator<'a>(module: &'a Module, name: &str) -> Result<&'a OperatorDef, String> {
    let op = module
        .units
        .iter()
        .find_map(|unit| match &unit.node {
            Unit::Operator(op) if op.name.node == name => Some(op),
            _ => None,
        })
        .ok_or_else(|| format!("operator `{name}` not found"))?;

    if op.params.is_empty() {
        Ok(op)
    } else {
        Err(format!("operator `{name}` is not zero-arity"))
    }
}

enum RecognizeResult<T> {
    Ok(T),
    Ineligible(String),
    Unknown(String),
}

fn recognize_interval_membership(
    expr: &Spanned<Expr>,
) -> RecognizeResult<(String, IntegerInterval)> {
    match &expr.node {
        Expr::In(lhs, rhs) => {
            let Some(variable) = variable_name(&lhs.node) else {
                return RecognizeResult::Ineligible(
                    "interval membership left side must be a variable".to_string(),
                );
            };
            recognize_range(rhs).map(|interval| (variable, interval))
        }
        Expr::Eq(lhs, rhs) => recognize_singleton_interval_equality(lhs, rhs),
        _ => RecognizeResult::Ineligible(
            "expected interval membership `x \\in lo..hi` or singleton equality `x = n`"
                .to_string(),
        ),
    }
}

fn recognize_singleton_interval_equality(
    lhs: &Spanned<Expr>,
    rhs: &Spanned<Expr>,
) -> RecognizeResult<(String, IntegerInterval)> {
    if let Some(variable) = variable_name(&lhs.node) {
        if let Some(value) = integer_literal(&rhs.node) {
            return RecognizeResult::Ok((variable, IntegerInterval::singleton(value)));
        }
    }
    if let Some(variable) = variable_name(&rhs.node) {
        if let Some(value) = integer_literal(&lhs.node) {
            return RecognizeResult::Ok((variable, IntegerInterval::singleton(value)));
        }
    }

    RecognizeResult::Ineligible(
        "singleton interval equality must compare a counter variable with an integer literal"
            .to_string(),
    )
}

fn recognize_interval_memberships(
    expr: &Spanned<Expr>,
) -> RecognizeResult<Vec<(String, IntegerInterval)>> {
    let mut conjuncts = Vec::new();
    flatten_and(expr, &mut conjuncts);

    let mut intervals = Vec::with_capacity(conjuncts.len());
    let mut seen = BTreeSet::new();
    for conjunct in conjuncts {
        let (variable, interval) = match recognize_interval_membership(conjunct) {
            RecognizeResult::Ok(value) => value,
            RecognizeResult::Ineligible(reason) => return RecognizeResult::Ineligible(reason),
            RecognizeResult::Unknown(reason) => return RecognizeResult::Unknown(reason),
        };
        if !seen.insert(variable.clone()) {
            return RecognizeResult::Ineligible(format!(
                "duplicate interval membership for counter `{variable}`"
            ));
        }
        intervals.push((variable, interval));
    }

    if intervals.is_empty() {
        RecognizeResult::Ineligible("expected interval membership conjuncts".to_string())
    } else {
        RecognizeResult::Ok(intervals)
    }
}

fn interval_map(intervals: &[(String, IntegerInterval)]) -> BTreeMap<String, IntegerInterval> {
    intervals
        .iter()
        .map(|(variable, interval)| (variable.clone(), interval.clone()))
        .collect()
}

trait MapRecognize<T> {
    fn map<U>(self, f: impl FnOnce(T) -> U) -> RecognizeResult<U>;
}

impl<T> MapRecognize<T> for RecognizeResult<T> {
    fn map<U>(self, f: impl FnOnce(T) -> U) -> RecognizeResult<U> {
        match self {
            RecognizeResult::Ok(value) => RecognizeResult::Ok(f(value)),
            RecognizeResult::Ineligible(reason) => RecognizeResult::Ineligible(reason),
            RecognizeResult::Unknown(reason) => RecognizeResult::Unknown(reason),
        }
    }
}

fn recognize_range(expr: &Spanned<Expr>) -> RecognizeResult<IntegerInterval> {
    let Expr::Range(lower, upper) = &expr.node else {
        return RecognizeResult::Ineligible("expected finite integer range `lo..hi`".to_string());
    };
    let Some(lower) = integer_literal(&lower.node) else {
        return RecognizeResult::Ineligible(
            "range lower bound must be an integer literal".to_string(),
        );
    };
    let Some(upper) = integer_literal(&upper.node) else {
        return RecognizeResult::Ineligible(
            "range upper bound must be an integer literal".to_string(),
        );
    };
    match IntegerInterval::new(lower, upper) {
        Some(interval) => RecognizeResult::Ok(interval),
        None => RecognizeResult::Unknown("empty counter interval".to_string()),
    }
}

fn recognize_transition(
    expr: &Spanned<Expr>,
    variable: &str,
) -> RecognizeResult<Vec<CounterTransitionBranch>> {
    let mut disjuncts = Vec::new();
    flatten_or(expr, &mut disjuncts);

    let mut branches = Vec::with_capacity(disjuncts.len());
    for disjunct in disjuncts {
        match recognize_branch(disjunct, variable) {
            RecognizeResult::Ok(branch) => branches.push(branch),
            RecognizeResult::Ineligible(reason) => return RecognizeResult::Ineligible(reason),
            RecognizeResult::Unknown(reason) => return RecognizeResult::Unknown(reason),
        }
    }

    if branches.is_empty() {
        RecognizeResult::Unknown("transition has no branches".to_string())
    } else {
        RecognizeResult::Ok(branches)
    }
}

fn group_independent_transition_components(
    expr: &Spanned<Expr>,
    variables: &[String],
) -> RecognizeResult<BTreeMap<String, Vec<Spanned<Expr>>>> {
    let variable_set: BTreeSet<&str> = variables.iter().map(String::as_str).collect();
    let mut grouped = BTreeMap::new();
    let mut components = Vec::new();
    flatten_and(expr, &mut components);

    for component in components {
        match group_transition_component(component, &variable_set, &mut grouped) {
            RecognizeResult::Ok(()) => {}
            RecognizeResult::Ineligible(reason) => return RecognizeResult::Ineligible(reason),
            RecognizeResult::Unknown(reason) => return RecognizeResult::Unknown(reason),
        }
    }

    for variable in variables {
        if !grouped.contains_key(variable) {
            return RecognizeResult::Ineligible(format!(
                "missing transition component for counter `{variable}`"
            ));
        }
    }

    RecognizeResult::Ok(grouped)
}

fn group_transition_component(
    expr: &Spanned<Expr>,
    variables: &BTreeSet<&str>,
    grouped: &mut BTreeMap<String, Vec<Spanned<Expr>>>,
) -> RecognizeResult<()> {
    let mut primed = BTreeSet::new();
    if let Err(reason) = collect_primed_variables(expr, &mut primed) {
        return RecognizeResult::Ineligible(reason);
    }

    if primed.len() == 1 {
        let variable = primed.iter().next().expect("len checked").clone();
        if !variables.contains(variable.as_str()) {
            return RecognizeResult::Ineligible(format!(
                "transition updates undeclared counter `{variable}`"
            ));
        }
        grouped.entry(variable).or_default().push(expr.clone());
        return RecognizeResult::Ok(());
    }

    if primed.is_empty() {
        return RecognizeResult::Ineligible(
            "transition component does not update a counter".to_string(),
        );
    }

    RecognizeResult::Ineligible("transition component couples multiple counters".to_string())
}

fn fold_and_components(parts: &[Spanned<Expr>]) -> Spanned<Expr> {
    let mut iter = parts.iter().cloned();
    let mut expr = iter
        .next()
        .expect("grouped transition components are never empty");
    for part in iter {
        expr = Spanned::dummy(Expr::And(Box::new(expr), Box::new(part)));
    }
    expr
}

fn collect_primed_variables(
    expr: &Spanned<Expr>,
    out: &mut BTreeSet<String>,
) -> Result<(), String> {
    match &expr.node {
        Expr::Bool(_)
        | Expr::Int(_)
        | Expr::String(_)
        | Expr::Ident(_, _)
        | Expr::StateVar(_, _, _) => Ok(()),
        Expr::Prime(inner) => match variable_name(&inner.node) {
            Some(variable) => {
                out.insert(variable);
                Ok(())
            }
            None => Err("unsupported primed expression in transition".to_string()),
        },
        Expr::And(lhs, rhs)
        | Expr::Or(lhs, rhs)
        | Expr::Implies(lhs, rhs)
        | Expr::Equiv(lhs, rhs)
        | Expr::In(lhs, rhs)
        | Expr::NotIn(lhs, rhs)
        | Expr::Eq(lhs, rhs)
        | Expr::Neq(lhs, rhs)
        | Expr::Lt(lhs, rhs)
        | Expr::Leq(lhs, rhs)
        | Expr::Gt(lhs, rhs)
        | Expr::Geq(lhs, rhs)
        | Expr::Add(lhs, rhs)
        | Expr::Sub(lhs, rhs)
        | Expr::Range(lhs, rhs) => {
            collect_primed_variables(lhs, out)?;
            collect_primed_variables(rhs, out)
        }
        Expr::Not(inner) | Expr::Neg(inner) => collect_primed_variables(inner, out),
        _ => Err("unsupported expression in independent-counter transition".to_string()),
    }
}

fn recognize_branch(
    expr: &Spanned<Expr>,
    variable: &str,
) -> RecognizeResult<CounterTransitionBranch> {
    let mut conjuncts = Vec::new();
    flatten_and(expr, &mut conjuncts);

    let mut guard = IntervalGuard::unconstrained();
    let mut update = None;

    for conjunct in conjuncts {
        match recognize_update(conjunct, variable) {
            RecognizeResult::Ok(Some(next_update)) => {
                if update.is_some() {
                    return RecognizeResult::Ineligible(
                        "transition branch contains multiple counter assignments".to_string(),
                    );
                }
                update = Some(next_update);
                continue;
            }
            RecognizeResult::Ok(None) => {}
            RecognizeResult::Ineligible(reason) => return RecognizeResult::Ineligible(reason),
            RecognizeResult::Unknown(reason) => return RecognizeResult::Unknown(reason),
        }

        if is_counter_assignment(conjunct, variable) {
            return RecognizeResult::Ineligible(
                "unsupported interval-counter assignment".to_string(),
            );
        }

        match recognize_guard(conjunct, variable) {
            RecognizeResult::Ok(()) => merge_guard(conjunct, variable, &mut guard),
            RecognizeResult::Ineligible(reason) => return RecognizeResult::Ineligible(reason),
            RecognizeResult::Unknown(reason) => return RecognizeResult::Unknown(reason),
        }
    }

    match update {
        Some(update) => RecognizeResult::Ok(CounterTransitionBranch { guard, update }),
        None => RecognizeResult::Ineligible(
            "transition branch has no assignment to the primed counter".to_string(),
        ),
    }
}

fn merge_guard(expr: &Spanned<Expr>, variable: &str, guard: &mut IntervalGuard) {
    match &expr.node {
        Expr::In(lhs, rhs) if variable_name(&lhs.node).as_deref() == Some(variable) => {
            if let RecognizeResult::Ok(interval) = recognize_range(rhs) {
                guard.constrain_lower(interval.lower);
                guard.constrain_upper(interval.upper);
            }
        }
        Expr::Eq(lhs, rhs) => {
            if variable_name(&lhs.node).as_deref() == Some(variable) {
                if let Some(value) = integer_literal(&rhs.node) {
                    guard.constrain_eq(value);
                }
            } else if variable_name(&rhs.node).as_deref() == Some(variable) {
                if let Some(value) = integer_literal(&lhs.node) {
                    guard.constrain_eq(value);
                }
            }
        }
        Expr::Lt(lhs, rhs) => {
            if variable_name(&lhs.node).as_deref() == Some(variable) {
                if let Some(value) = integer_literal(&rhs.node) {
                    guard.constrain_upper(value - BigInt::one());
                }
            } else if variable_name(&rhs.node).as_deref() == Some(variable) {
                if let Some(value) = integer_literal(&lhs.node) {
                    guard.constrain_lower(value + BigInt::one());
                }
            }
        }
        Expr::Leq(lhs, rhs) => {
            if variable_name(&lhs.node).as_deref() == Some(variable) {
                if let Some(value) = integer_literal(&rhs.node) {
                    guard.constrain_upper(value);
                }
            } else if variable_name(&rhs.node).as_deref() == Some(variable) {
                if let Some(value) = integer_literal(&lhs.node) {
                    guard.constrain_lower(value);
                }
            }
        }
        Expr::Gt(lhs, rhs) => {
            if variable_name(&lhs.node).as_deref() == Some(variable) {
                if let Some(value) = integer_literal(&rhs.node) {
                    guard.constrain_lower(value + BigInt::one());
                }
            } else if variable_name(&rhs.node).as_deref() == Some(variable) {
                if let Some(value) = integer_literal(&lhs.node) {
                    guard.constrain_upper(value - BigInt::one());
                }
            }
        }
        Expr::Geq(lhs, rhs) => {
            if variable_name(&lhs.node).as_deref() == Some(variable) {
                if let Some(value) = integer_literal(&rhs.node) {
                    guard.constrain_lower(value);
                }
            } else if variable_name(&rhs.node).as_deref() == Some(variable) {
                if let Some(value) = integer_literal(&lhs.node) {
                    guard.constrain_upper(value);
                }
            }
        }
        _ => {}
    }
}

fn recognize_guard(expr: &Spanned<Expr>, variable: &str) -> RecognizeResult<()> {
    match &expr.node {
        Expr::Bool(true) => RecognizeResult::Ok(()),
        Expr::In(lhs, rhs) if variable_name(&lhs.node).as_deref() == Some(variable) => {
            recognize_range(rhs).map(|_| ())
        }
        Expr::Eq(lhs, rhs) => recognize_variable_integer_comparison(lhs, rhs, variable),
        Expr::Lt(lhs, rhs) | Expr::Leq(lhs, rhs) | Expr::Gt(lhs, rhs) | Expr::Geq(lhs, rhs) => {
            recognize_variable_integer_comparison(lhs, rhs, variable)
        }
        _ => {
            RecognizeResult::Ineligible("unsupported interval-counter transition guard".to_string())
        }
    }
}

fn recognize_variable_integer_comparison(
    lhs: &Spanned<Expr>,
    rhs: &Spanned<Expr>,
    variable: &str,
) -> RecognizeResult<()> {
    if variable_name(&lhs.node).as_deref() == Some(variable) && integer_literal(&rhs.node).is_some()
    {
        return RecognizeResult::Ok(());
    }
    if integer_literal(&lhs.node).is_some() && variable_name(&rhs.node).as_deref() == Some(variable)
    {
        return RecognizeResult::Ok(());
    }
    RecognizeResult::Ineligible(
        "interval-counter guard must compare the counter with an integer literal".to_string(),
    )
}

fn recognize_update(
    expr: &Spanned<Expr>,
    variable: &str,
) -> RecognizeResult<Option<CounterUpdate>> {
    match &expr.node {
        Expr::Eq(lhs, rhs) => {
            if is_prime_of_variable(&lhs.node, variable) {
                return RecognizeResult::Ok(recognize_update_rhs(&rhs.node, variable));
            }
            if is_prime_of_variable(&rhs.node, variable) {
                return RecognizeResult::Ok(recognize_update_rhs(&lhs.node, variable));
            }
            RecognizeResult::Ok(None)
        }
        Expr::In(lhs, rhs) if is_prime_of_variable(&lhs.node, variable) => {
            recognize_range(rhs).map(|interval| Some(CounterUpdate::Choose(interval)))
        }
        _ => RecognizeResult::Ok(None),
    }
}

fn is_counter_assignment(expr: &Spanned<Expr>, variable: &str) -> bool {
    match &expr.node {
        Expr::Eq(lhs, rhs) => {
            is_prime_of_variable(&lhs.node, variable) || is_prime_of_variable(&rhs.node, variable)
        }
        Expr::In(lhs, _) => is_prime_of_variable(&lhs.node, variable),
        _ => false,
    }
}

fn recognize_update_rhs(expr: &Expr, variable: &str) -> Option<CounterUpdate> {
    if variable_name(expr).as_deref() == Some(variable) {
        return Some(CounterUpdate::Shift(BigInt::zero()));
    }
    if let Some(value) = integer_literal(expr) {
        return Some(CounterUpdate::Set(value));
    }

    match expr {
        Expr::Add(lhs, rhs) if variable_name(&lhs.node).as_deref() == Some(variable) => {
            integer_literal(&rhs.node).map(CounterUpdate::Shift)
        }
        Expr::Add(lhs, rhs) if variable_name(&rhs.node).as_deref() == Some(variable) => {
            integer_literal(&lhs.node).map(CounterUpdate::Shift)
        }
        Expr::Sub(lhs, rhs) if variable_name(&lhs.node).as_deref() == Some(variable) => {
            integer_literal(&rhs.node).map(|delta| CounterUpdate::Shift(-delta))
        }
        _ => None,
    }
}

fn flatten_or<'a>(expr: &'a Spanned<Expr>, out: &mut Vec<&'a Spanned<Expr>>) {
    match &expr.node {
        Expr::Or(lhs, rhs) => {
            flatten_or(lhs, out);
            flatten_or(rhs, out);
        }
        Expr::Bool(false) => {}
        _ => out.push(expr),
    }
}

fn flatten_and<'a>(expr: &'a Spanned<Expr>, out: &mut Vec<&'a Spanned<Expr>>) {
    match &expr.node {
        Expr::And(lhs, rhs) => {
            flatten_and(lhs, out);
            flatten_and(rhs, out);
        }
        Expr::Bool(true) => {}
        _ => out.push(expr),
    }
}

fn variable_name(expr: &Expr) -> Option<String> {
    match expr {
        Expr::Ident(name, _) | Expr::StateVar(name, _, _) => Some(name.clone()),
        _ => None,
    }
}

fn is_prime_of_variable(expr: &Expr, variable: &str) -> bool {
    match expr {
        Expr::Prime(inner) => variable_name(&inner.node).as_deref() == Some(variable),
        _ => false,
    }
}

fn integer_literal(expr: &Expr) -> Option<BigInt> {
    match expr {
        Expr::Int(value) => Some(value.clone()),
        Expr::Neg(inner) => integer_literal(&inner.node).map(|value| -value),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analytical::{AnalyticalAdmission, GateDecision, VerificationGate};
    use crate::shared_verdict::{SharedVerdict, Verdict};
    use crate::test_support::parse_module;

    fn outcome_for(
        source: &str,
        init: &str,
        next: &str,
        invariant: &str,
    ) -> IntervalCounterOutcome {
        let module = parse_module(source);
        recognize_module_interval_counter(&module, init, next, invariant)
    }

    fn candidate_for(
        source: &str,
        init: &str,
        next: &str,
        invariant: &str,
    ) -> CandidateProof<IntervalCounterCertificate> {
        match outcome_for(source, init, next, invariant) {
            AnalyticalOutcome::CandidateProof(candidate) => candidate,
            other => panic!("expected interval-counter candidate, got {other:?}"),
        }
    }

    fn independent_outcome_for(
        source: &str,
        init: &str,
        next: &str,
        invariant: &str,
    ) -> IndependentIntervalCounterOutcome {
        let module = parse_module(source);
        recognize_module_independent_interval_counters(&module, init, next, invariant)
    }

    fn independent_candidate_for(
        source: &str,
        init: &str,
        next: &str,
        invariant: &str,
    ) -> CandidateProof<IndependentIntervalCounterCertificate> {
        match independent_outcome_for(source, init, next, invariant) {
            AnalyticalOutcome::CandidateProof(candidate) => candidate,
            other => panic!("expected independent-counter candidate, got {other:?}"),
        }
    }

    fn admission_for(
        source: &str,
        init: &str,
        next: &str,
        invariant: &str,
    ) -> IntervalCounterAdmissionOutcome {
        let module = parse_module(source);
        admit_module_interval_counters(&module, init, next, invariant)
    }

    fn execution_admission_for(
        source: &str,
        init: &str,
        next: &str,
        invariant: &str,
        require_deadlock_freedom: bool,
    ) -> IntervalCounterExecutionOutcome {
        let module = parse_module(source);
        admit_module_interval_counter_execution_model(
            &module,
            init,
            next,
            invariant,
            require_deadlock_freedom,
        )
    }

    #[test]
    fn recognizes_and_verifies_cyclic_interval_counter() {
        let candidate = candidate_for(
            r#"
---- MODULE WeirdButStructural ----
EXTENDS Integers
VARIABLE x
StartHere == x \in 0..2
MoveHere == (x < 2 /\ x' = x + 1) \/ (x = 2 /\ x' = 0)
SafeHere == x \in 0..2
====
"#,
            "StartHere",
            "MoveHere",
            "SafeHere",
        );

        let certificate = candidate.certificate();
        assert_eq!(certificate.variable(), "x");
        assert_eq!(certificate.invariant_cardinality(), BigInt::from(3));
        assert_eq!(certificate.transition_branches().len(), 2);

        let shared = SharedVerdict::new();
        let gate = VerificationGate::new(&shared);
        let raw: IntervalCounterOutcome = AnalyticalOutcome::CandidateProof(candidate.clone());
        assert_eq!(gate.inspect(&raw), GateDecision::NeedsProof);
        assert!(!shared.is_resolved());

        let verified = verify_interval_counter_candidate(candidate).expect("certificate verifies");
        assert_eq!(
            gate.publish_verified_proof(verified),
            GateDecision::Published
        );
        assert_eq!(shared.get(), Some(Verdict::Satisfied));
    }

    #[test]
    fn module_and_operator_names_are_not_spec_shortcuts() {
        let candidate = candidate_for(
            r#"
---- MODULE NotACounterByName ----
EXTENDS Integers
VARIABLE pc
Alpha == pc \in -1..1
Beta == (pc < 1 /\ pc' = pc + 1) \/ (pc = 1 /\ pc' = -1)
Gamma == pc \in -1..1
====
"#,
            "Alpha",
            "Beta",
            "Gamma",
        );
        assert_eq!(candidate.certificate().variable(), "pc");

        let misleading = outcome_for(
            r#"
---- MODULE Counter ----
EXTENDS Integers
VARIABLE x
Init == x = 0
Next == x' = x * 2
Inv == x \in 0..10
====
"#,
            "Init",
            "Next",
            "Inv",
        );
        assert!(matches!(misleading, AnalyticalOutcome::Ineligible(_)));
    }

    #[test]
    fn unsupported_transition_shape_is_ineligible() {
        let outcome = outcome_for(
            r#"
---- MODULE UnsupportedTransition ----
EXTENDS Integers
VARIABLE x
Init == x \in 0..2
Next == x' = x * 2
Inv == x \in 0..2
====
"#,
            "Init",
            "Next",
            "Inv",
        );

        match outcome {
            AnalyticalOutcome::Ineligible(reason) => {
                assert!(reason.reason().contains("assignment"));
            }
            other => panic!("expected Ineligible, got {other:?}"),
        }
    }

    #[test]
    fn empty_interval_is_unknown() {
        let outcome = outcome_for(
            r#"
---- MODULE EmptyInit ----
EXTENDS Integers
VARIABLE x
Init == x \in 3..1
Next == x' = x
Inv == x \in 0..3
====
"#,
            "Init",
            "Next",
            "Inv",
        );

        match outcome {
            AnalyticalOutcome::Unknown(reason) => {
                assert!(reason.reason().contains("empty counter interval"));
            }
            other => panic!("expected Unknown, got {other:?}"),
        }
    }

    #[test]
    fn verifier_rejects_candidate_that_does_not_prove_induction() {
        let candidate = candidate_for(
            r#"
---- MODULE BadInvariant ----
EXTENDS Integers
VARIABLE x
Init == x \in 0..2
Next == x' = x
Inv == x \in 1..2
====
"#,
            "Init",
            "Next",
            "Inv",
        );

        let err = verify_interval_counter_candidate(candidate).expect_err("not an inductive proof");
        assert!(matches!(
            err,
            IntervalCounterVerificationError::InitOutsideInvariant { .. }
        ));
    }

    #[test]
    fn verifier_rejects_branch_that_can_escape_invariant() {
        let candidate = candidate_for(
            r#"
---- MODULE BadStep ----
EXTENDS Integers
VARIABLE x
Init == x \in 0..2
Next == x' = x + 1
Inv == x \in 0..2
====
"#,
            "Init",
            "Next",
            "Inv",
        );

        let err = verify_interval_counter_candidate(candidate).expect_err("step escapes invariant");
        assert!(matches!(
            err,
            IntervalCounterVerificationError::BranchEscapesInvariant { .. }
        ));
    }

    #[test]
    fn recognizes_finite_domain_transition_update() {
        let candidate = candidate_for(
            r#"
---- MODULE FiniteDomainUpdate ----
EXTENDS Integers
VARIABLE x
Init == x \in 0..2
Next == x' \in 0..2
Inv == x \in 0..2
====
"#,
            "Init",
            "Next",
            "Inv",
        );

        let certificate = candidate.certificate();
        assert_eq!(certificate.transition_branches().len(), 1);
        assert!(matches!(
            certificate.transition_branches()[0].update(),
            CounterUpdate::Choose(interval) if interval.cardinality() == BigInt::from(3)
        ));
        verify_interval_counter_candidate(candidate).expect("finite-domain update verifies");
    }

    #[test]
    fn execution_fast_path_accepts_full_init_total_finite_domain_transition() {
        let admission = execution_admission_for(
            r#"
---- MODULE FullFiniteDomainUpdate ----
EXTENDS Integers
VARIABLE x
Init == x \in 0..2
Next == x' \in 0..2
Inv == x \in 0..2
====
"#,
            "Init",
            "Next",
            "Inv",
            true,
        );

        let AnalyticalAdmission::VerifiedProof(verified) = admission else {
            panic!("expected verified execution-model certificate");
        };
        let certificate = verified.certificate();
        assert_eq!(certificate.state_count(), &BigInt::from(3));
        assert!(certificate.deadlock_free());
        assert!(certificate.admission().initial_states_cover_invariant());
        assert!(certificate.admission().transition_total_on_invariant());
    }

    #[test]
    fn execution_fast_path_rejects_partial_init_even_when_invariant_is_inductive() {
        let admission = execution_admission_for(
            r#"
---- MODULE PartialInitFiniteDomainUpdate ----
EXTENDS Integers
VARIABLE x
Init == x = 0
Next == x' \in 0..2
Inv == x \in 0..2
====
"#,
            "Init",
            "Next",
            "Inv",
            true,
        );

        match admission {
            AnalyticalAdmission::Ineligible(reason) => {
                assert!(reason.reason().contains("requires Init to cover"));
            }
            other => panic!("expected Ineligible execution model, got {other:?}"),
        }
    }

    #[test]
    fn finite_domain_transition_rejects_non_range_domain() {
        let outcome = outcome_for(
            r#"
---- MODULE UnsupportedFiniteDomainUpdate ----
EXTENDS Integers
VARIABLE x
Init == x \in 0..2
Next == x' \in {0, 1, 2}
Inv == x \in 0..2
====
"#,
            "Init",
            "Next",
            "Inv",
        );

        match outcome {
            AnalyticalOutcome::Ineligible(reason) => {
                assert!(reason.reason().contains("finite integer range"));
            }
            other => panic!("expected Ineligible, got {other:?}"),
        }
    }

    #[test]
    fn recognizes_and_verifies_two_independent_counters() {
        let candidate = independent_candidate_for(
            r#"
---- MODULE ProductShapeNotByName ----
EXTENDS Integers
VARIABLE x, y
Begin == x \in 0..2 /\ y \in 5..6
Step ==
  ((x < 2 /\ x' = x + 1) \/ (x = 2 /\ x' = 0))
  /\
  ((y < 6 /\ y' = y + 1) \/ (y = 6 /\ y' = 5))
Box == x \in 0..2 /\ y \in 5..6
====
"#,
            "Begin",
            "Step",
            "Box",
        );

        let certificate = candidate.certificate();
        assert_eq!(certificate.counter_count(), 2);
        assert_eq!(certificate.invariant_cardinality(), BigInt::from(6));
        assert_eq!(certificate.counters()[0].variable(), "x");
        assert_eq!(certificate.counters()[1].variable(), "y");

        let shared = SharedVerdict::new();
        let gate = VerificationGate::new(&shared);
        let raw: IndependentIntervalCounterOutcome =
            AnalyticalOutcome::CandidateProof(candidate.clone());
        assert_eq!(gate.inspect(&raw), GateDecision::NeedsProof);
        assert!(!shared.is_resolved());

        let verified =
            verify_independent_interval_counter_candidate(candidate).expect("product verifies");
        assert_eq!(
            gate.publish_verified_proof(verified),
            GateDecision::Published
        );
        assert_eq!(shared.get(), Some(Verdict::Satisfied));
    }

    #[test]
    fn independent_counter_rejects_coupled_transition_component() {
        let outcome = independent_outcome_for(
            r#"
---- MODULE CoupledProduct ----
EXTENDS Integers
VARIABLE x, y
Init == x \in 0..2 /\ y \in 0..2
Next == x' = y /\ y' = y
Inv == x \in 0..2 /\ y \in 0..2
====
"#,
            "Init",
            "Next",
            "Inv",
        );

        match outcome {
            AnalyticalOutcome::Ineligible(reason) => {
                assert!(reason.reason().contains("assignment"));
            }
            other => panic!("expected Ineligible, got {other:?}"),
        }
    }

    #[test]
    fn independent_counter_rejects_global_source_only_component() {
        let outcome = independent_outcome_for(
            r#"
---- MODULE GuardedProduct ----
EXTENDS Integers
VARIABLE x, y
Init == x \in 0..2 /\ y \in 0..2
Next == (x = 0) /\ (x' = x) /\ (y' = y)
Inv == x \in 0..2 /\ y \in 0..2
====
"#,
            "Init",
            "Next",
            "Inv",
        );

        match outcome {
            AnalyticalOutcome::Ineligible(reason) => {
                assert!(reason.reason().contains("does not update a counter"));
            }
            other => panic!("expected Ineligible, got {other:?}"),
        }
    }

    #[test]
    fn independent_counter_verifier_reports_escaping_counter() {
        let candidate = independent_candidate_for(
            r#"
---- MODULE ProductBadStep ----
EXTENDS Integers
VARIABLE x, y
Init == x \in 0..2 /\ y \in 0..2
Next == (x' = x + 1) /\ (y' = y)
Inv == x \in 0..2 /\ y \in 0..2
====
"#,
            "Init",
            "Next",
            "Inv",
        );

        let err = verify_independent_interval_counter_candidate(candidate)
            .expect_err("x branch escapes invariant");
        match err {
            IndependentIntervalCounterVerificationError::Counter { variable, source } => {
                assert_eq!(variable, "x");
                assert!(matches!(
                    source,
                    IntervalCounterVerificationError::BranchEscapesInvariant { .. }
                ));
            }
            other => panic!("expected per-counter error, got {other:?}"),
        }
    }

    #[test]
    fn admission_verifies_independent_counter_product_before_publishing() {
        let admission = admission_for(
            r#"
---- MODULE ProductAdmissionNotByName ----
EXTENDS Integers
VARIABLE x, y
Begin == x \in 0..2 /\ y \in 5..6
Step ==
  ((x < 2 /\ x' = x + 1) \/ (x = 2 /\ x' = 0))
  /\
  ((y < 6 /\ y' = y + 1) \/ (y = 6 /\ y' = 5))
Box == x \in 0..2 /\ y \in 5..6
====
"#,
            "Begin",
            "Step",
            "Box",
        );

        assert!(admission.is_publishable());
        let AnalyticalAdmission::VerifiedProof(verified) = admission else {
            panic!("expected verified interval-counter admission");
        };
        let certificate = verified.certificate();
        assert_eq!(certificate.init_operator(), "Begin");
        assert_eq!(certificate.next_operator(), "Step");
        assert_eq!(certificate.invariant_operator(), "Box");
        assert_eq!(certificate.counter_count(), 2);
        assert_eq!(certificate.invariant_cardinality(), BigInt::from(6));
        assert!(matches!(
            certificate.proof(),
            IntervalCounterAdmissionProof::Independent(_)
        ));

        let shared = SharedVerdict::new();
        let gate = VerificationGate::new(&shared);
        assert_eq!(
            gate.publish_verified_proof(verified),
            GateDecision::Published
        );
        assert_eq!(shared.get(), Some(Verdict::Satisfied));
    }

    #[test]
    fn admission_falls_back_to_single_interval_counter() {
        let admission = admission_for(
            r#"
---- MODULE SingleAdmission ----
EXTENDS Integers
VARIABLE pc
Alpha == pc \in -1..1
Beta == (pc < 1 /\ pc' = pc + 1) \/ (pc = 1 /\ pc' = -1)
Gamma == pc \in -1..1
====
"#,
            "Alpha",
            "Beta",
            "Gamma",
        );

        let AnalyticalAdmission::VerifiedProof(verified) = admission else {
            panic!("expected verified single-counter admission");
        };
        let certificate = verified.certificate();
        assert_eq!(certificate.counter_count(), 1);
        assert_eq!(certificate.invariant_cardinality(), BigInt::from(3));
        assert!(matches!(
            certificate.proof(),
            IntervalCounterAdmissionProof::Single(_)
        ));
    }

    #[test]
    fn admission_accepts_singleton_equality_initializers_for_independent_counters() {
        let admission = admission_for(
            r#"
---- MODULE EqualityInitProductAdmission ----
EXTENDS Integers
VARIABLE x, y
Init == (x = 0) /\ (y = 5)
Next ==
  ((x < 2 /\ x' = x + 1) \/ (x = 2 /\ x' = 0))
  /\
  ((y < 6 /\ y' = y + 1) \/ (y = 6 /\ y' = 5))
Inv == x \in 0..2 /\ y \in 5..6
====
"#,
            "Init",
            "Next",
            "Inv",
        );

        let AnalyticalAdmission::VerifiedProof(verified) = admission else {
            panic!("expected verified product admission from singleton initializers");
        };
        let certificate = verified.certificate();
        assert_eq!(certificate.counter_count(), 2);
        assert_eq!(certificate.invariant_cardinality(), BigInt::from(6));
        assert!(matches!(
            certificate.proof(),
            IntervalCounterAdmissionProof::Independent(_)
        ));
    }

    #[test]
    fn singleton_equality_admission_requires_integer_literal() {
        let admission = admission_for(
            r#"
---- MODULE EqualityInitAdversarial ----
EXTENDS Integers
VARIABLE x, y
Init == x = y
Next == x' = x
Inv == x \in 0..2
====
"#,
            "Init",
            "Next",
            "Inv",
        );

        assert!(!admission.is_publishable());
        match admission {
            AnalyticalAdmission::Ineligible(reason) => {
                assert!(reason.reason().contains("integer literal"));
            }
            other => panic!("expected Ineligible admission, got {other:?}"),
        }
    }

    #[test]
    fn admission_returns_unknown_when_certificate_verification_fails() {
        let admission = admission_for(
            r#"
---- MODULE AdmissionBadStep ----
EXTENDS Integers
VARIABLE x
Init == x \in 0..2
Next == x' = x + 1
Inv == x \in 0..2
====
"#,
            "Init",
            "Next",
            "Inv",
        );

        assert!(!admission.is_publishable());
        match admission {
            AnalyticalAdmission::Unknown(reason) => {
                assert!(reason.reason().contains("verification failed"));
            }
            other => panic!("expected Unknown admission, got {other:?}"),
        }
    }

    #[test]
    fn admission_keeps_unsupported_shapes_ineligible() {
        let admission = admission_for(
            r#"
---- MODULE AdmissionUnsupported ----
EXTENDS Integers
VARIABLE x
Init == x \in 0..2
Next == x' = x * 2
Inv == x \in 0..2
====
"#,
            "Init",
            "Next",
            "Inv",
        );

        assert!(!admission.is_publishable());
        assert!(matches!(admission, AnalyticalAdmission::Ineligible(_)));
    }

    #[test]
    fn independent_counter_requires_matching_init_and_invariant_variables() {
        let outcome = independent_outcome_for(
            r#"
---- MODULE ProductMismatchedVars ----
EXTENDS Integers
VARIABLE x, y, z
Init == x \in 0..2 /\ y \in 0..2
Next == (x' = x) /\ (y' = y)
Inv == x \in 0..2 /\ z \in 0..2
====
"#,
            "Init",
            "Next",
            "Inv",
        );

        match outcome {
            AnalyticalOutcome::Ineligible(reason) => {
                assert!(reason.reason().contains("different counter variables"));
            }
            other => panic!("expected Ineligible, got {other:?}"),
        }
    }
}
