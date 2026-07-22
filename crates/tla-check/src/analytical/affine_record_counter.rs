// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Proof-gated recognizer for finite record-valued affine counter systems.
//!
//! This module is deliberately certificate-first. The structural recognizer
//! can emit a candidate for CoffeeCan-shaped systems, but the independent
//! verifier rechecks the finite record box, affine slice, guarded record EXCEPT
//! updates, and stutter branches before a caller can admit the proof.

use num_bigint::BigInt;
use num_traits::{One, ToPrimitive, Zero};
use std::borrow::Cow;
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use tla_core::ast::{ExceptPathElement, Expr, Module, OperatorDef, Substitution, Unit};
use tla_core::span::Spanned;

use crate::eval::apply_substitutions;

use super::{
    AnalyticalAdmission, AnalyticalOutcome, CandidateProof, Ineligible, Unknown, VerifiedProof,
};

/// Outcome type used by the affine record-counter recognizer.
pub type AffineRecordCounterOutcome = AnalyticalOutcome<(), AffineRecordCounterCertificate>;

/// Proof-gated admission result for affine record counters.
pub type AffineRecordCounterAdmissionOutcome =
    AnalyticalAdmission<(), AffineRecordCounterAdmissionCertificate>;

/// Proof-gated admission result for exact reachable-state counting.
pub type AffineRecordCounterExecutionOutcome =
    AnalyticalAdmission<(), AffineRecordCounterExecutionCertificate>;

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

    fn contains(&self, other: &Self) -> bool {
        self.lower <= other.lower && other.upper <= self.upper
    }

    fn shifted(&self, delta: &BigInt) -> Self {
        Self {
            lower: &self.lower + delta,
            upper: &self.upper + delta,
        }
    }

    fn constrain_lower(&mut self, lower: BigInt) -> bool {
        if self.lower < lower {
            self.lower = lower;
            true
        } else {
            false
        }
    }

    fn constrain_upper(&mut self, upper: BigInt) -> bool {
        if self.upper > upper {
            self.upper = upper;
            true
        } else {
            false
        }
    }

    fn is_empty(&self) -> bool {
        self.lower > self.upper
    }
}

/// One integer field in the finite record box.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RecordFieldBox {
    name: String,
    interval: IntegerInterval,
}

impl RecordFieldBox {
    /// Field name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Finite integer interval assigned to this field.
    pub fn interval(&self) -> &IntegerInterval {
        &self.interval
    }
}

/// Canonical affine term over record fields.
///
/// The term denotes `constant + sum(coeff_i * field_i)`. Coefficients are kept
/// sorted by field name with zero coefficients removed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AffineTerm {
    constant: BigInt,
    coefficients: Vec<(String, BigInt)>,
}

impl AffineTerm {
    /// Constant term.
    pub fn constant(&self) -> &BigInt {
        &self.constant
    }

    /// Sorted non-zero field coefficients.
    pub fn coefficients(&self) -> &[(String, BigInt)] {
        &self.coefficients
    }

    fn zero() -> Self {
        Self {
            constant: BigInt::zero(),
            coefficients: Vec::new(),
        }
    }

    fn from_constant(value: BigInt) -> Self {
        Self {
            constant: value,
            coefficients: Vec::new(),
        }
    }

    fn field(name: String) -> Self {
        Self {
            constant: BigInt::zero(),
            coefficients: vec![(name, BigInt::one())],
        }
    }

    fn normalize(constant: BigInt, coefficients: BTreeMap<String, BigInt>) -> Self {
        Self {
            constant,
            coefficients: coefficients
                .into_iter()
                .filter(|(_, coefficient)| !coefficient.is_zero())
                .collect(),
        }
    }

    fn add(&self, other: &Self) -> Self {
        let mut coefficients = self.coefficient_map();
        for (field, coefficient) in &other.coefficients {
            *coefficients
                .entry(field.clone())
                .or_insert_with(BigInt::zero) += coefficient;
        }
        Self::normalize(&self.constant + &other.constant, coefficients)
    }

    fn neg(&self) -> Self {
        let coefficients = self
            .coefficients
            .iter()
            .map(|(field, coefficient)| (field.clone(), -coefficient))
            .collect();
        Self::normalize(-&self.constant, coefficients)
    }

    fn sub(&self, other: &Self) -> Self {
        self.add(&other.neg())
    }

    fn without_constant(&self) -> Self {
        Self {
            constant: BigInt::zero(),
            coefficients: self.coefficients.clone(),
        }
    }

    fn coefficient_map(&self) -> BTreeMap<String, BigInt> {
        self.coefficients.iter().cloned().collect()
    }

    fn same_coefficients(&self, other: &Self) -> bool {
        self.coefficients == other.coefficients
    }

    fn delta_under(&self, updates: &BTreeMap<String, BigInt>) -> BigInt {
        self.coefficients
            .iter()
            .filter_map(|(field, coefficient)| updates.get(field).map(|delta| coefficient * delta))
            .fold(BigInt::zero(), |acc, delta| acc + delta)
    }

    fn shifted_by_updates(&self, updates: &BTreeMap<String, BigInt>) -> Self {
        Self {
            constant: &self.constant + self.delta_under(updates),
            coefficients: self.coefficients.clone(),
        }
    }
}

/// Closed affine constraint over record fields.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AffineConstraint {
    term: AffineTerm,
    lower: Option<BigInt>,
    upper: Option<BigInt>,
}

impl AffineConstraint {
    /// Affine term constrained by this fact.
    pub fn term(&self) -> &AffineTerm {
        &self.term
    }

    /// Lower inclusive bound, if any.
    pub fn lower(&self) -> Option<&BigInt> {
        self.lower.as_ref()
    }

    /// Upper inclusive bound, if any.
    pub fn upper(&self) -> Option<&BigInt> {
        self.upper.as_ref()
    }

    fn new_normalized(
        term: AffineTerm,
        lower: Option<BigInt>,
        upper: Option<BigInt>,
    ) -> Option<Self> {
        let constant = term.constant.clone();
        let normalized = term.without_constant();
        let lower = lower.map(|bound| bound - &constant);
        let upper = upper.map(|bound| bound - &constant);
        if matches!((&lower, &upper), (Some(lower), Some(upper)) if lower > upper) {
            return None;
        }
        Some(Self {
            term: normalized,
            lower,
            upper,
        })
    }

    fn exact_value(&self) -> Option<&BigInt> {
        match (&self.lower, &self.upper) {
            (Some(lower), Some(upper)) if lower == upper => Some(lower),
            _ => None,
        }
    }
}

/// Initial record set recognized from `Init`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AffineRecordInit {
    constraints: Vec<AffineConstraint>,
}

impl AffineRecordInit {
    /// Affine constraints that slice the finite record box in `Init`.
    pub fn constraints(&self) -> &[AffineConstraint] {
        &self.constraints
    }
}

/// Field delta update for one branch.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FieldDeltaUpdate {
    field: String,
    delta: BigInt,
}

impl FieldDeltaUpdate {
    /// Updated field.
    pub fn field(&self) -> &str {
        &self.field
    }

    /// Additive delta applied to the old field value.
    pub fn delta(&self) -> &BigInt {
        &self.delta
    }
}

/// One guarded transition branch.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AffineRecordTransitionBranch {
    guard: Vec<AffineConstraint>,
    updates: Vec<FieldDeltaUpdate>,
    stutter: bool,
}

impl AffineRecordTransitionBranch {
    /// Source-state guard facts recognized from this branch.
    pub fn guard(&self) -> &[AffineConstraint] {
        &self.guard
    }

    /// Additive field updates. Missing fields are unchanged.
    pub fn updates(&self) -> &[FieldDeltaUpdate] {
        &self.updates
    }

    /// Whether this branch is an explicit stutter/identity update.
    pub fn is_stutter(&self) -> bool {
        self.stutter
    }

    fn update_map(&self) -> BTreeMap<String, BigInt> {
        self.updates
            .iter()
            .map(|update| (update.field.clone(), update.delta.clone()))
            .collect()
    }
}

/// Certificate emitted by the affine record-counter structural recognizer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AffineRecordCounterCertificate {
    variable: String,
    fields: Vec<RecordFieldBox>,
    init: AffineRecordInit,
    invariant_constraints: Vec<AffineConstraint>,
    transition_branches: Vec<AffineRecordTransitionBranch>,
}

impl AffineRecordCounterCertificate {
    /// Record-valued state variable.
    pub fn variable(&self) -> &str {
        &self.variable
    }

    /// Finite integer field box, sorted by field name.
    pub fn fields(&self) -> &[RecordFieldBox] {
        &self.fields
    }

    /// Recognized initial slice.
    pub fn init(&self) -> &AffineRecordInit {
        &self.init
    }

    /// Inductive affine constraints over the finite record box.
    pub fn invariant_constraints(&self) -> &[AffineConstraint] {
        &self.invariant_constraints
    }

    /// Guarded transition branches.
    pub fn transition_branches(&self) -> &[AffineRecordTransitionBranch] {
        &self.transition_branches
    }

    /// Product cardinality of the finite record box, ignoring affine slices.
    pub fn box_cardinality(&self) -> BigInt {
        self.fields.iter().fold(BigInt::one(), |acc, field| {
            acc * field.interval.cardinality()
        })
    }

    /// Whether a stutter branch covers the lower boundary of a recognized affine slice.
    pub fn terminal_stutter_covered(&self) -> bool {
        self.invariant_constraints.iter().any(|constraint| {
            constraint.lower.is_some()
                && self.transition_branches.iter().any(|branch| {
                    branch.stutter
                        && branch.guard.iter().any(|guard| {
                            guard.term.same_coefficients(&constraint.term)
                                && guard.exact_value() == constraint.lower.as_ref()
                        })
                })
        })
    }
}

/// Verified affine record-counter certificate with source metadata.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AffineRecordCounterAdmissionCertificate {
    init_operator: String,
    next_operator: String,
    invariant_operator: String,
    proof: AffineRecordCounterCertificate,
}

impl AffineRecordCounterAdmissionCertificate {
    /// INIT operator supplied by the caller/config.
    pub fn init_operator(&self) -> &str {
        &self.init_operator
    }

    /// NEXT operator supplied by the caller/config.
    pub fn next_operator(&self) -> &str {
        &self.next_operator
    }

    /// Invariant operator supplied by the caller/config.
    pub fn invariant_operator(&self) -> &str {
        &self.invariant_operator
    }

    /// Verified proof payload.
    pub fn proof(&self) -> &AffineRecordCounterCertificate {
        &self.proof
    }
}

/// Verified affine record-counter exact-count certificate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AffineRecordCounterExecutionCertificate {
    admission: AffineRecordCounterAdmissionCertificate,
    state_count: BigInt,
    initial_state_count: BigInt,
    transition_count: BigInt,
    enumerated_box_states: BigInt,
    deadlock_free: bool,
}

impl AffineRecordCounterExecutionCertificate {
    /// Verified structural admission that this exact count refines.
    pub fn admission(&self) -> &AffineRecordCounterAdmissionCertificate {
        &self.admission
    }

    /// Exact number of reachable states in the admitted affine transition system.
    pub fn state_count(&self) -> &BigInt {
        &self.state_count
    }

    /// Exact number of initial states.
    pub fn initial_state_count(&self) -> &BigInt {
        &self.initial_state_count
    }

    /// Exact number of enabled transition branches examined from reachable states.
    pub fn transition_count(&self) -> &BigInt {
        &self.transition_count
    }

    /// Number of finite record-box states enumerated while deriving the initial frontier.
    pub fn enumerated_box_states(&self) -> &BigInt {
        &self.enumerated_box_states
    }

    /// Whether every reachable state has at least one enabled transition branch.
    pub fn deadlock_free(&self) -> bool {
        self.deadlock_free
    }
}

/// Verification error for an affine record-counter certificate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AffineRecordCounterVerificationError {
    /// Certificate contains no record fields.
    EmptyFieldBox,
    /// Record fields are not sorted and unique.
    NonCanonicalFieldOrder,
    /// A field interval is empty.
    EmptyFieldInterval {
        /// The field whose interval is empty.
        field: String,
    },
    /// Affine term references a field outside the record box.
    UnknownFieldInTerm {
        /// The unknown field referenced by the term.
        field: String,
    },
    /// Transition set is empty.
    EmptyTransitionSet,
    /// A transition branch updates the same field more than once.
    DuplicateFieldUpdate {
        /// The field updated more than once.
        field: String,
    },
    /// Initial slice is inconsistent with the claimed invariant constraints.
    InitOutsideInvariant {
        /// Index of the invariant constraint the init slice violates.
        constraint_index: usize,
        /// Interval the init slice actually achieves for the constraint term.
        observed: IntegerInterval,
        /// The invariant constraint that was expected to hold.
        expected: AffineConstraint,
    },
    /// A branch can update one field outside its finite box.
    BranchEscapesFieldBox {
        /// Index of the offending transition branch.
        branch_index: usize,
        /// The field that escapes its box.
        field: String,
        /// Interval the branch's successor can reach for the field.
        image: IntegerInterval,
        /// The field's declared finite box.
        expected: IntegerInterval,
    },
    /// A branch can violate one affine invariant constraint.
    BranchViolatesAffineInvariant {
        /// Index of the offending transition branch.
        branch_index: usize,
        /// Index of the violated invariant constraint.
        constraint_index: usize,
        /// Interval the post-state term can reach.
        observed: IntegerInterval,
        /// The invariant constraint that was expected to hold.
        expected: AffineConstraint,
    },
}

/// Exact counting error for an already verified affine record-counter certificate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AffineRecordCounterExactCountError {
    /// The finite record box is larger than the configured analytical limit.
    EnumerationLimitExceeded {
        /// Configured maximum number of box states to enumerate.
        limit: usize,
        /// Product cardinality of the finite record box.
        box_cardinality: BigInt,
    },
    /// An integer bound or coefficient is outside the exact counter's native range.
    IntegerOutOfRange {
        /// The out-of-range value.
        value: BigInt,
        /// Context describing where the value came from.
        context: String,
    },
    /// Native integer arithmetic overflowed while evaluating a verified affine term.
    ArithmeticOverflow {
        /// Context describing the operation that overflowed.
        context: String,
    },
    /// A transition branch produced a state outside the verified finite record box.
    TransitionEscapesFieldBox {
        /// The field that escaped its box.
        field: String,
        /// The out-of-box value produced.
        value: i64,
    },
    /// A transition branch produced a state violating the verified invariant slice.
    TransitionEscapesInvariant,
    /// Deadlock checking was requested and a reachable state has no enabled branch.
    DeadlockFound {
        /// Number of reachable states discovered before the deadlock.
        reachable_states: BigInt,
    },
}

/// Verify an affine record-counter candidate proof and return a publishable proof.
pub fn verify_affine_record_counter_candidate(
    candidate: CandidateProof<AffineRecordCounterCertificate>,
) -> Result<VerifiedProof<AffineRecordCounterCertificate>, AffineRecordCounterVerificationError> {
    let certificate = candidate.into_certificate();
    verify_certificate(&certificate)?;
    Ok(VerifiedProof::new(certificate))
}

/// Admit an affine record-counter proof only after certificate verification.
pub fn admit_module_affine_record_counter(
    module: &Module,
    init_operator: &str,
    next_operator: &str,
    invariant_operator: &str,
) -> AffineRecordCounterAdmissionOutcome {
    match recognize_module_affine_record_counter(
        module,
        init_operator,
        next_operator,
        invariant_operator,
    ) {
        AnalyticalOutcome::CandidateProof(candidate) => {
            match verify_affine_record_counter_candidate(candidate) {
                Ok(verified) => {
                    let certificate = AffineRecordCounterAdmissionCertificate {
                        init_operator: init_operator.to_string(),
                        next_operator: next_operator.to_string(),
                        invariant_operator: invariant_operator.to_string(),
                        proof: verified.into_certificate(),
                    };
                    AnalyticalAdmission::VerifiedProof(VerifiedProof::new(certificate))
                }
                Err(err) => AnalyticalAdmission::Unknown(Unknown::new(format!(
                    "affine record-counter certificate verification failed: {err:?}"
                ))),
            }
        }
        AnalyticalOutcome::Unknown(reason) => AnalyticalAdmission::Unknown(reason),
        AnalyticalOutcome::Ineligible(reason) => AnalyticalAdmission::Ineligible(reason),
        AnalyticalOutcome::CandidateViolation(_) => AnalyticalAdmission::Unknown(Unknown::new(
            "affine record-counter recognizer emitted unsupported violation candidate",
        )),
    }
}

/// Admit an affine record-counter proof across a root module plus checker scope.
///
/// Root operators take precedence, followed by imported modules in caller order.
/// Standalone `INSTANCE M WITH ...` substitutions from the root are applied to
/// imported operator bodies before recognition.
pub fn admit_module_set_affine_record_counter(
    module: &Module,
    checker_modules: &[&Module],
    init_operator: &str,
    next_operator: &str,
    invariant_operator: &str,
) -> AffineRecordCounterAdmissionOutcome {
    match recognize_module_set_affine_record_counter(
        module,
        checker_modules,
        init_operator,
        next_operator,
        invariant_operator,
    ) {
        AnalyticalOutcome::CandidateProof(candidate) => {
            match verify_affine_record_counter_candidate(candidate) {
                Ok(verified) => {
                    let certificate = AffineRecordCounterAdmissionCertificate {
                        init_operator: init_operator.to_string(),
                        next_operator: next_operator.to_string(),
                        invariant_operator: invariant_operator.to_string(),
                        proof: verified.into_certificate(),
                    };
                    AnalyticalAdmission::VerifiedProof(VerifiedProof::new(certificate))
                }
                Err(err) => AnalyticalAdmission::Unknown(Unknown::new(format!(
                    "affine record-counter certificate verification failed: {err:?}"
                ))),
            }
        }
        AnalyticalOutcome::Unknown(reason) => AnalyticalAdmission::Unknown(reason),
        AnalyticalOutcome::Ineligible(reason) => AnalyticalAdmission::Ineligible(reason),
        AnalyticalOutcome::CandidateViolation(_) => AnalyticalAdmission::Unknown(Unknown::new(
            "affine record-counter recognizer emitted unsupported violation candidate",
        )),
    }
}

/// Admit a verified affine record-counter exact execution model.
pub fn admit_module_affine_record_counter_execution_model(
    module: &Module,
    init_operator: &str,
    next_operator: &str,
    invariant_operator: &str,
    require_deadlock_freedom: bool,
    max_enumerated_box_states: usize,
) -> AffineRecordCounterExecutionOutcome {
    admit_module_set_affine_record_counter_execution_model(
        module,
        &[],
        init_operator,
        next_operator,
        invariant_operator,
        require_deadlock_freedom,
        max_enumerated_box_states,
    )
}

/// Admit a verified affine record-counter exact execution model across checker scope.
pub fn admit_module_set_affine_record_counter_execution_model(
    module: &Module,
    checker_modules: &[&Module],
    init_operator: &str,
    next_operator: &str,
    invariant_operator: &str,
    require_deadlock_freedom: bool,
    max_enumerated_box_states: usize,
) -> AffineRecordCounterExecutionOutcome {
    let admission = match admit_module_set_affine_record_counter(
        module,
        checker_modules,
        init_operator,
        next_operator,
        invariant_operator,
    ) {
        AnalyticalAdmission::VerifiedProof(verified) => verified.into_certificate(),
        AnalyticalAdmission::ReplayedViolation(_) => {
            return AnalyticalAdmission::Unknown(Unknown::new(
                "affine record-counter admission emitted unsupported violation replay",
            ));
        }
        AnalyticalAdmission::Unknown(reason) => return AnalyticalAdmission::Unknown(reason),
        AnalyticalAdmission::Ineligible(reason) => return AnalyticalAdmission::Ineligible(reason),
    };

    match exact_count_verified_certificate(
        admission,
        require_deadlock_freedom,
        max_enumerated_box_states,
    ) {
        Ok(certificate) => AnalyticalAdmission::VerifiedProof(VerifiedProof::new(certificate)),
        Err(err) => AnalyticalAdmission::Unknown(Unknown::new(format!(
            "affine record-counter exact count failed: {err:?}"
        ))),
    }
}

/// Recognize a module by caller-provided operator names.
///
/// Operator names are lookup keys only. The recognizer does not inspect the
/// module name and does not use CoffeeCan-specific field, variable, or action
/// names.
pub fn recognize_module_affine_record_counter(
    module: &Module,
    init_operator: &str,
    next_operator: &str,
    invariant_operator: &str,
) -> AffineRecordCounterOutcome {
    let env = OperatorEnv::from_module(module);
    let init = match env.find_zero_arity(init_operator) {
        Ok(op) => op,
        Err(reason) => return AnalyticalOutcome::Ineligible(Ineligible::new(reason)),
    };
    let next = match env.find_zero_arity(next_operator) {
        Ok(op) => op,
        Err(reason) => return AnalyticalOutcome::Ineligible(Ineligible::new(reason)),
    };
    let invariant = match env.find_zero_arity(invariant_operator) {
        Ok(op) => op,
        Err(reason) => return AnalyticalOutcome::Ineligible(Ineligible::new(reason)),
    };

    recognize_affine_record_counter_exprs_with_env(init, next, invariant, &env)
}

/// Recognize a module set by caller-provided operator names.
pub fn recognize_module_set_affine_record_counter(
    module: &Module,
    checker_modules: &[&Module],
    init_operator: &str,
    next_operator: &str,
    invariant_operator: &str,
) -> AffineRecordCounterOutcome {
    let env = OperatorEnv::from_module_set(module, checker_modules);
    let init = match env.find_zero_arity(init_operator) {
        Ok(op) => op,
        Err(reason) => return AnalyticalOutcome::Ineligible(Ineligible::new(reason)),
    };
    let next = match env.find_zero_arity(next_operator) {
        Ok(op) => op,
        Err(reason) => return AnalyticalOutcome::Ineligible(Ineligible::new(reason)),
    };
    let invariant = match env.find_zero_arity(invariant_operator) {
        Ok(op) => op,
        Err(reason) => return AnalyticalOutcome::Ineligible(Ineligible::new(reason)),
    };

    recognize_affine_record_counter_exprs_with_env(init, next, invariant, &env)
}

/// Recognize an affine record-counter proof candidate from zero-arity operators.
pub fn recognize_affine_record_counter_ops(
    init: &OperatorDef,
    next: &OperatorDef,
    invariant: &OperatorDef,
) -> AffineRecordCounterOutcome {
    if !init.params.is_empty() || !next.params.is_empty() || !invariant.params.is_empty() {
        return AnalyticalOutcome::Ineligible(Ineligible::new(
            "affine record-counter operators must be zero-arity",
        ));
    }

    let env = OperatorEnv::empty();
    recognize_affine_record_counter_exprs_with_env(&init.body, &next.body, &invariant.body, &env)
}

/// Recognize an affine record-counter proof candidate from AST bodies.
pub fn recognize_affine_record_counter_exprs(
    init: &Spanned<Expr>,
    next: &Spanned<Expr>,
    invariant: &Spanned<Expr>,
) -> AffineRecordCounterOutcome {
    let env = OperatorEnv::empty();
    recognize_affine_record_counter_exprs_with_env(init, next, invariant, &env)
}

/// Verify a recognized affine record-counter certificate without publishing.
pub fn verify_certificate(
    certificate: &AffineRecordCounterCertificate,
) -> Result<(), AffineRecordCounterVerificationError> {
    if certificate.fields.is_empty() {
        return Err(AffineRecordCounterVerificationError::EmptyFieldBox);
    }
    if certificate
        .fields
        .windows(2)
        .any(|pair| pair[0].name >= pair[1].name)
    {
        return Err(AffineRecordCounterVerificationError::NonCanonicalFieldOrder);
    }

    for field in &certificate.fields {
        if field.interval.is_empty() {
            return Err(AffineRecordCounterVerificationError::EmptyFieldInterval {
                field: field.name.clone(),
            });
        }
    }
    if certificate.transition_branches.is_empty() {
        return Err(AffineRecordCounterVerificationError::EmptyTransitionSet);
    }

    let field_names: BTreeSet<_> = certificate
        .fields
        .iter()
        .map(|field| field.name.as_str())
        .collect();
    let box_intervals = field_interval_map(&certificate.fields);

    for constraint in certificate
        .init
        .constraints
        .iter()
        .chain(certificate.invariant_constraints.iter())
        .chain(
            certificate
                .transition_branches
                .iter()
                .flat_map(|branch| branch.guard.iter()),
        )
    {
        verify_term_fields(&constraint.term, &field_names)?;
    }

    for branch in &certificate.transition_branches {
        let mut seen = BTreeSet::new();
        for update in &branch.updates {
            if !field_names.contains(update.field.as_str()) {
                return Err(AffineRecordCounterVerificationError::UnknownFieldInTerm {
                    field: update.field.clone(),
                });
            }
            if !seen.insert(update.field.clone()) {
                return Err(AffineRecordCounterVerificationError::DuplicateFieldUpdate {
                    field: update.field.clone(),
                });
            }
        }
    }

    verify_init_implies_invariant(certificate, &box_intervals)?;
    verify_branches_preserve_invariant(certificate, &box_intervals)
}

fn recognize_affine_record_counter_exprs_with_env<'a>(
    init: &'a Spanned<Expr>,
    next: &'a Spanned<Expr>,
    invariant: &'a Spanned<Expr>,
    env: &'a OperatorEnv<'_>,
) -> AffineRecordCounterOutcome {
    let recognized_invariant = match recognize_record_predicate(invariant, env) {
        RecognizeResult::Ok(value) => value,
        RecognizeResult::Ineligible(reason) => {
            return AnalyticalOutcome::Ineligible(Ineligible::new(reason));
        }
        RecognizeResult::Unknown(reason) => {
            return AnalyticalOutcome::Unknown(Unknown::new(reason));
        }
    };
    let recognized_init = match recognize_record_predicate(init, env) {
        RecognizeResult::Ok(value) => value,
        RecognizeResult::Ineligible(reason) => {
            return AnalyticalOutcome::Ineligible(Ineligible::new(reason));
        }
        RecognizeResult::Unknown(reason) => {
            return AnalyticalOutcome::Unknown(Unknown::new(reason));
        }
    };

    if recognized_init.variable != recognized_invariant.variable {
        return AnalyticalOutcome::Ineligible(Ineligible::new(
            "init and invariant refer to different record variables",
        ));
    }
    if recognized_init.fields != recognized_invariant.fields {
        return AnalyticalOutcome::Ineligible(Ineligible::new(
            "init and invariant use different finite record boxes",
        ));
    }

    let transition_branches = match recognize_transition(next, &recognized_invariant.variable, env)
    {
        RecognizeResult::Ok(branches) => branches,
        RecognizeResult::Ineligible(reason) => {
            return AnalyticalOutcome::Ineligible(Ineligible::new(reason));
        }
        RecognizeResult::Unknown(reason) => {
            return AnalyticalOutcome::Unknown(Unknown::new(reason));
        }
    };

    let mut invariant_constraints = recognized_invariant.constraints;
    for derived in derive_affine_slice_constraints(
        &recognized_invariant.fields,
        &recognized_init.constraints,
        &transition_branches,
    ) {
        push_or_intersect_constraint(&mut invariant_constraints, derived);
    }

    let certificate = AffineRecordCounterCertificate {
        variable: recognized_invariant.variable,
        fields: recognized_invariant.fields,
        init: AffineRecordInit {
            constraints: recognized_init.constraints,
        },
        invariant_constraints,
        transition_branches,
    };

    AnalyticalOutcome::CandidateProof(CandidateProof::new(certificate))
}

fn verify_init_implies_invariant(
    certificate: &AffineRecordCounterCertificate,
    box_intervals: &BTreeMap<String, IntegerInterval>,
) -> Result<(), AffineRecordCounterVerificationError> {
    let init_intervals = match tighten_intervals(box_intervals, &certificate.init.constraints) {
        Some(intervals) => intervals,
        None => {
            return Err(AffineRecordCounterVerificationError::InitOutsideInvariant {
                constraint_index: 0,
                observed: IntegerInterval {
                    lower: BigInt::one(),
                    upper: BigInt::zero(),
                },
                expected: certificate
                    .invariant_constraints
                    .first()
                    .cloned()
                    .unwrap_or_else(|| AffineConstraint {
                        term: AffineTerm::zero(),
                        lower: Some(BigInt::zero()),
                        upper: Some(BigInt::zero()),
                    }),
            });
        }
    };

    for (constraint_index, constraint) in certificate.invariant_constraints.iter().enumerate() {
        let observed = bound_affine_term(
            &constraint.term,
            &init_intervals,
            &certificate.init.constraints,
        );
        let Some(observed) = observed else {
            continue;
        };
        if !constraint_contains_interval(constraint, &observed) {
            return Err(AffineRecordCounterVerificationError::InitOutsideInvariant {
                constraint_index,
                observed,
                expected: constraint.clone(),
            });
        }
    }

    Ok(())
}

fn verify_branches_preserve_invariant(
    certificate: &AffineRecordCounterCertificate,
    box_intervals: &BTreeMap<String, IntegerInterval>,
) -> Result<(), AffineRecordCounterVerificationError> {
    for (branch_index, branch) in certificate.transition_branches.iter().enumerate() {
        let mut source_constraints = certificate.invariant_constraints.clone();
        source_constraints.extend(branch.guard.clone());
        let Some(source_intervals) = tighten_intervals(box_intervals, &source_constraints) else {
            continue;
        };

        let update_map = branch.update_map();
        for field in &certificate.fields {
            let delta = update_map
                .get(&field.name)
                .cloned()
                .unwrap_or_else(BigInt::zero);
            let image = source_intervals
                .get(&field.name)
                .expect("field box map contains all fields")
                .shifted(&delta);
            if !field.interval.contains(&image) {
                return Err(
                    AffineRecordCounterVerificationError::BranchEscapesFieldBox {
                        branch_index,
                        field: field.name.clone(),
                        image,
                        expected: field.interval.clone(),
                    },
                );
            }
        }

        for (constraint_index, constraint) in certificate.invariant_constraints.iter().enumerate() {
            let post_term = constraint.term.shifted_by_updates(&update_map);
            let Some(observed) =
                bound_affine_term(&post_term, &source_intervals, &source_constraints)
            else {
                continue;
            };
            if !constraint_contains_interval(constraint, &observed) {
                return Err(
                    AffineRecordCounterVerificationError::BranchViolatesAffineInvariant {
                        branch_index,
                        constraint_index,
                        observed,
                        expected: constraint.clone(),
                    },
                );
            }
        }
    }

    Ok(())
}

type ExactRecordState = Vec<i64>;

#[derive(Debug, Clone)]
struct ExactInterval {
    lower: i64,
    upper: i64,
}

#[derive(Debug, Clone)]
struct ExactRecordContext {
    fields: Vec<(String, ExactInterval)>,
    field_indexes: BTreeMap<String, usize>,
}

fn exact_count_verified_certificate(
    admission: AffineRecordCounterAdmissionCertificate,
    require_deadlock_freedom: bool,
    max_enumerated_box_states: usize,
) -> Result<AffineRecordCounterExecutionCertificate, AffineRecordCounterExactCountError> {
    let proof = admission.proof();
    let box_cardinality = proof.box_cardinality();
    if box_cardinality > BigInt::from(max_enumerated_box_states) {
        return Err(
            AffineRecordCounterExactCountError::EnumerationLimitExceeded {
                limit: max_enumerated_box_states,
                box_cardinality,
            },
        );
    }

    let context = exact_record_context(proof)?;
    let initial_states = enumerate_initial_states(proof, &context, max_enumerated_box_states)?;
    let initial_state_count = BigInt::from(initial_states.len());
    let mut reachable = BTreeSet::new();
    let mut queue = VecDeque::new();

    for state in initial_states {
        if reachable.insert(state.clone()) {
            queue.push_back(state);
        }
    }

    let mut transition_count = BigInt::zero();
    let mut deadlock_free = true;
    while let Some(state) = queue.pop_front() {
        let mut enabled = 0usize;
        for branch in proof.transition_branches() {
            if !state_satisfies_constraints(&state, branch.guard(), &context)? {
                continue;
            }
            enabled += 1;
            transition_count += BigInt::one();
            let successor = apply_exact_branch(&state, branch, proof, &context)?;
            if reachable.insert(successor.clone()) {
                queue.push_back(successor);
            }
        }

        if enabled == 0 {
            deadlock_free = false;
            if require_deadlock_freedom {
                return Err(AffineRecordCounterExactCountError::DeadlockFound {
                    reachable_states: BigInt::from(reachable.len()),
                });
            }
        }
    }

    let state_count = BigInt::from(reachable.len());
    Ok(AffineRecordCounterExecutionCertificate {
        admission,
        state_count,
        initial_state_count,
        transition_count,
        enumerated_box_states: box_cardinality,
        deadlock_free,
    })
}

fn exact_record_context(
    certificate: &AffineRecordCounterCertificate,
) -> Result<ExactRecordContext, AffineRecordCounterExactCountError> {
    let mut fields = Vec::with_capacity(certificate.fields().len());
    let mut field_indexes = BTreeMap::new();
    for field in certificate.fields() {
        let index = fields.len();
        let lower = big_int_to_i64(
            field.interval().lower(),
            format!("lower bound for field `{}`", field.name()),
        )?;
        let upper = big_int_to_i64(
            field.interval().upper(),
            format!("upper bound for field `{}`", field.name()),
        )?;
        fields.push((field.name().to_string(), ExactInterval { lower, upper }));
        field_indexes.insert(field.name().to_string(), index);
    }
    Ok(ExactRecordContext {
        fields,
        field_indexes,
    })
}

fn enumerate_initial_states(
    certificate: &AffineRecordCounterCertificate,
    context: &ExactRecordContext,
    max_enumerated_box_states: usize,
) -> Result<Vec<ExactRecordState>, AffineRecordCounterExactCountError> {
    let mut out = Vec::new();
    let mut current = Vec::with_capacity(context.fields.len());
    let mut enumerated = 0usize;
    enumerate_initial_states_inner(
        0,
        &mut current,
        &mut enumerated,
        &mut out,
        certificate,
        context,
        max_enumerated_box_states,
    )?;
    Ok(out)
}

fn enumerate_initial_states_inner(
    field_index: usize,
    current: &mut ExactRecordState,
    enumerated: &mut usize,
    out: &mut Vec<ExactRecordState>,
    certificate: &AffineRecordCounterCertificate,
    context: &ExactRecordContext,
    max_enumerated_box_states: usize,
) -> Result<(), AffineRecordCounterExactCountError> {
    if field_index == context.fields.len() {
        *enumerated += 1;
        if *enumerated > max_enumerated_box_states {
            return Err(
                AffineRecordCounterExactCountError::EnumerationLimitExceeded {
                    limit: max_enumerated_box_states,
                    box_cardinality: certificate.box_cardinality(),
                },
            );
        }
        if state_satisfies_constraints(current, certificate.init().constraints(), context)?
            && state_satisfies_constraints(current, certificate.invariant_constraints(), context)?
        {
            out.push(current.clone());
        }
        return Ok(());
    }

    let (_, interval) = &context.fields[field_index];
    for value in interval.lower..=interval.upper {
        current.push(value);
        enumerate_initial_states_inner(
            field_index + 1,
            current,
            enumerated,
            out,
            certificate,
            context,
            max_enumerated_box_states,
        )?;
        current.pop();
    }
    Ok(())
}

fn state_satisfies_constraints(
    state: &ExactRecordState,
    constraints: &[AffineConstraint],
    context: &ExactRecordContext,
) -> Result<bool, AffineRecordCounterExactCountError> {
    for constraint in constraints {
        let value = exact_affine_value(state, constraint.term(), context)?;
        if let Some(lower) = constraint.lower() {
            let lower = big_int_to_i128(lower, "constraint lower bound")?;
            if value < lower {
                return Ok(false);
            }
        }
        if let Some(upper) = constraint.upper() {
            let upper = big_int_to_i128(upper, "constraint upper bound")?;
            if value > upper {
                return Ok(false);
            }
        }
    }
    Ok(true)
}

fn apply_exact_branch(
    state: &ExactRecordState,
    branch: &AffineRecordTransitionBranch,
    certificate: &AffineRecordCounterCertificate,
    context: &ExactRecordContext,
) -> Result<ExactRecordState, AffineRecordCounterExactCountError> {
    let mut successor = state.clone();
    for update in branch.updates() {
        let Some(index) = context.field_indexes.get(update.field()) else {
            return Err(
                AffineRecordCounterExactCountError::TransitionEscapesFieldBox {
                    field: update.field().to_string(),
                    value: 0,
                },
            );
        };
        let delta = big_int_to_i64(
            update.delta(),
            format!("delta for field `{}`", update.field()),
        )?;
        successor[*index] = successor[*index].checked_add(delta).ok_or_else(|| {
            AffineRecordCounterExactCountError::ArithmeticOverflow {
                context: format!("applying delta for field `{}`", update.field()),
            }
        })?;
    }

    for (field_index, (field, interval)) in context.fields.iter().enumerate() {
        let value = successor[field_index];
        if value < interval.lower || value > interval.upper {
            return Err(
                AffineRecordCounterExactCountError::TransitionEscapesFieldBox {
                    field: field.clone(),
                    value,
                },
            );
        }
    }
    if !state_satisfies_constraints(&successor, certificate.invariant_constraints(), context)? {
        return Err(AffineRecordCounterExactCountError::TransitionEscapesInvariant);
    }
    Ok(successor)
}

fn exact_affine_value(
    state: &ExactRecordState,
    term: &AffineTerm,
    context: &ExactRecordContext,
) -> Result<i128, AffineRecordCounterExactCountError> {
    let mut value = big_int_to_i128(term.constant(), "affine constant")?;
    for (field, coefficient) in term.coefficients() {
        let Some(index) = context.field_indexes.get(field) else {
            return Err(
                AffineRecordCounterExactCountError::TransitionEscapesFieldBox {
                    field: field.clone(),
                    value: 0,
                },
            );
        };
        let coefficient = big_int_to_i128(coefficient, format!("coefficient for field `{field}`"))?;
        let contribution = coefficient
            .checked_mul(i128::from(state[*index]))
            .ok_or_else(|| AffineRecordCounterExactCountError::ArithmeticOverflow {
                context: format!("multiplying coefficient for field `{field}`"),
            })?;
        value = value.checked_add(contribution).ok_or_else(|| {
            AffineRecordCounterExactCountError::ArithmeticOverflow {
                context: format!("accumulating coefficient for field `{field}`"),
            }
        })?;
    }
    Ok(value)
}

fn big_int_to_i64(
    value: &BigInt,
    context: impl Into<String>,
) -> Result<i64, AffineRecordCounterExactCountError> {
    value
        .to_i64()
        .ok_or_else(|| AffineRecordCounterExactCountError::IntegerOutOfRange {
            value: value.clone(),
            context: context.into(),
        })
}

fn big_int_to_i128(
    value: &BigInt,
    context: impl Into<String>,
) -> Result<i128, AffineRecordCounterExactCountError> {
    value
        .to_i128()
        .ok_or_else(|| AffineRecordCounterExactCountError::IntegerOutOfRange {
            value: value.clone(),
            context: context.into(),
        })
}

fn verify_term_fields(
    term: &AffineTerm,
    field_names: &BTreeSet<&str>,
) -> Result<(), AffineRecordCounterVerificationError> {
    for (field, _) in &term.coefficients {
        if !field_names.contains(field.as_str()) {
            return Err(AffineRecordCounterVerificationError::UnknownFieldInTerm {
                field: field.clone(),
            });
        }
    }
    Ok(())
}

fn constraint_contains_interval(constraint: &AffineConstraint, interval: &IntegerInterval) -> bool {
    constraint
        .lower
        .as_ref()
        .is_none_or(|lower| lower <= &interval.lower)
        && constraint
            .upper
            .as_ref()
            .is_none_or(|upper| &interval.upper <= upper)
}

fn field_interval_map(fields: &[RecordFieldBox]) -> BTreeMap<String, IntegerInterval> {
    fields
        .iter()
        .map(|field| (field.name.clone(), field.interval.clone()))
        .collect()
}

fn derive_affine_slice_constraints(
    fields: &[RecordFieldBox],
    init_constraints: &[AffineConstraint],
    branches: &[AffineRecordTransitionBranch],
) -> Vec<AffineConstraint> {
    let intervals = field_interval_map(fields);
    let mut derived = Vec::new();

    for init_constraint in init_constraints {
        let Some(init_value) = init_constraint.exact_value() else {
            continue;
        };
        if init_constraint.term.coefficients.is_empty() {
            continue;
        }

        let mut saw_decrease = false;
        let mut all_non_increasing = true;
        for branch in branches {
            let delta = init_constraint.term.delta_under(&branch.update_map());
            if delta > BigInt::zero() {
                all_non_increasing = false;
                break;
            }
            if delta < BigInt::zero() {
                saw_decrease = true;
            }
        }
        if !all_non_increasing || !saw_decrease {
            continue;
        }

        let lower = terminal_stutter_exact_value(branches, &init_constraint.term)
            .or_else(|| linear_min_max(&init_constraint.term, &intervals).map(|bounds| bounds.0));
        let Some(lower) = lower else {
            continue;
        };
        if lower > *init_value {
            continue;
        }
        if let Some(constraint) = AffineConstraint::new_normalized(
            init_constraint.term.clone(),
            Some(lower),
            Some(init_value.clone()),
        ) {
            derived.push(constraint);
        }
    }

    derived
}

fn terminal_stutter_exact_value(
    branches: &[AffineRecordTransitionBranch],
    term: &AffineTerm,
) -> Option<BigInt> {
    branches.iter().find_map(|branch| {
        if !branch.stutter {
            return None;
        }
        branch.guard.iter().find_map(|guard| {
            if guard.term.same_coefficients(term) {
                guard.exact_value().cloned()
            } else {
                None
            }
        })
    })
}

fn push_or_intersect_constraint(constraints: &mut Vec<AffineConstraint>, next: AffineConstraint) {
    if let Some(existing) = constraints
        .iter_mut()
        .find(|constraint| constraint.term.same_coefficients(&next.term))
    {
        if let Some(lower) = next.lower {
            match &mut existing.lower {
                Some(existing_lower) if *existing_lower < lower => *existing_lower = lower,
                None => existing.lower = Some(lower),
                _ => {}
            }
        }
        if let Some(upper) = next.upper {
            match &mut existing.upper {
                Some(existing_upper) if *existing_upper > upper => *existing_upper = upper,
                None => existing.upper = Some(upper),
                _ => {}
            }
        }
    } else {
        constraints.push(next);
    }
}

fn tighten_intervals(
    base: &BTreeMap<String, IntegerInterval>,
    constraints: &[AffineConstraint],
) -> Option<BTreeMap<String, IntegerInterval>> {
    let mut intervals = base.clone();

    loop {
        let mut changed = false;
        for constraint in constraints {
            if let Some(observed) = linear_min_max(&constraint.term, &intervals) {
                if constraint
                    .lower
                    .as_ref()
                    .is_some_and(|lower| &observed.1 < lower)
                    || constraint
                        .upper
                        .as_ref()
                        .is_some_and(|upper| upper < &observed.0)
                {
                    return None;
                }
            }

            if let Some(upper) = &constraint.upper {
                for (field, coefficient) in &constraint.term.coefficients {
                    if coefficient.is_zero() {
                        continue;
                    }
                    let others = min_contribution_except(&constraint.term, &intervals, field);
                    let remainder = upper - others;
                    if coefficient > &BigInt::zero() {
                        let new_upper = floor_div(&remainder, coefficient);
                        changed |= intervals.get_mut(field)?.constrain_upper(new_upper);
                    } else {
                        let positive = -coefficient;
                        let new_lower = ceil_div(&(-remainder), &positive);
                        changed |= intervals.get_mut(field)?.constrain_lower(new_lower);
                    }
                }
            }

            if let Some(lower) = &constraint.lower {
                for (field, coefficient) in &constraint.term.coefficients {
                    if coefficient.is_zero() {
                        continue;
                    }
                    let others = max_contribution_except(&constraint.term, &intervals, field);
                    let remainder = lower - others;
                    if coefficient > &BigInt::zero() {
                        let new_lower = ceil_div(&remainder, coefficient);
                        changed |= intervals.get_mut(field)?.constrain_lower(new_lower);
                    } else {
                        let positive = -coefficient;
                        let new_upper = floor_div(&(-remainder), &positive);
                        changed |= intervals.get_mut(field)?.constrain_upper(new_upper);
                    }
                }
            }

            if intervals.values().any(IntegerInterval::is_empty) {
                return None;
            }
        }
        if !changed {
            return Some(intervals);
        }
    }
}

fn bound_affine_term(
    term: &AffineTerm,
    intervals: &BTreeMap<String, IntegerInterval>,
    constraints: &[AffineConstraint],
) -> Option<IntegerInterval> {
    let (mut lower, mut upper) = linear_min_max(term, intervals)?;
    for constraint in constraints {
        if !constraint.term.same_coefficients(term) {
            continue;
        }
        let shift = &term.constant - &constraint.term.constant;
        if let Some(constraint_lower) = &constraint.lower {
            lower = lower.max(constraint_lower + &shift);
        }
        if let Some(constraint_upper) = &constraint.upper {
            upper = upper.min(constraint_upper + &shift);
        }
    }
    IntegerInterval::new(lower, upper)
}

fn linear_min_max(
    term: &AffineTerm,
    intervals: &BTreeMap<String, IntegerInterval>,
) -> Option<(BigInt, BigInt)> {
    let mut lower = term.constant.clone();
    let mut upper = term.constant.clone();
    for (field, coefficient) in &term.coefficients {
        let interval = intervals.get(field)?;
        if coefficient >= &BigInt::zero() {
            lower += coefficient * &interval.lower;
            upper += coefficient * &interval.upper;
        } else {
            lower += coefficient * &interval.upper;
            upper += coefficient * &interval.lower;
        }
    }
    Some((lower, upper))
}

fn min_contribution_except(
    term: &AffineTerm,
    intervals: &BTreeMap<String, IntegerInterval>,
    skip_field: &str,
) -> BigInt {
    let mut value = term.constant.clone();
    for (field, coefficient) in &term.coefficients {
        if field == skip_field {
            continue;
        }
        let interval = intervals
            .get(field)
            .expect("verified terms reference known fields");
        if coefficient >= &BigInt::zero() {
            value += coefficient * &interval.lower;
        } else {
            value += coefficient * &interval.upper;
        }
    }
    value
}

fn max_contribution_except(
    term: &AffineTerm,
    intervals: &BTreeMap<String, IntegerInterval>,
    skip_field: &str,
) -> BigInt {
    let mut value = term.constant.clone();
    for (field, coefficient) in &term.coefficients {
        if field == skip_field {
            continue;
        }
        let interval = intervals
            .get(field)
            .expect("verified terms reference known fields");
        if coefficient >= &BigInt::zero() {
            value += coefficient * &interval.upper;
        } else {
            value += coefficient * &interval.lower;
        }
    }
    value
}

fn floor_div(numerator: &BigInt, denominator: &BigInt) -> BigInt {
    debug_assert!(denominator > &BigInt::zero());
    let quotient = numerator / denominator;
    let remainder = numerator % denominator;
    if remainder < BigInt::zero() {
        quotient - BigInt::one()
    } else {
        quotient
    }
}

fn ceil_div(numerator: &BigInt, denominator: &BigInt) -> BigInt {
    -floor_div(&(-numerator), denominator)
}

#[derive(Debug, Clone)]
struct RecognizedRecordPredicate {
    variable: String,
    fields: Vec<RecordFieldBox>,
    constraints: Vec<AffineConstraint>,
}

#[derive(Debug, Clone)]
struct OperatorEnv<'a> {
    operators: BTreeMap<String, OperatorBinding<'a>>,
}

#[derive(Debug, Clone)]
struct OperatorBinding<'a> {
    params_len: usize,
    body: Cow<'a, Spanned<Expr>>,
}

impl<'a> OperatorEnv<'a> {
    fn empty() -> Self {
        Self {
            operators: BTreeMap::new(),
        }
    }

    fn from_module(module: &'a Module) -> Self {
        let mut env = Self::empty();
        env.insert_module(module, None);
        env
    }

    fn from_module_set(module: &'a Module, checker_modules: &[&'a Module]) -> Self {
        let mut env = Self::empty();
        env.insert_module(module, None);
        let instance_substitutions = standalone_instance_substitutions_by_module(module);
        for checker_module in checker_modules {
            let substitutions = instance_substitutions
                .get(&checker_module.name.node)
                .map(Vec::as_slice);
            env.insert_module(checker_module, substitutions);
        }
        env
    }

    fn insert_module(&mut self, module: &'a Module, substitutions: Option<&[Substitution]>) {
        for unit in &module.units {
            let Unit::Operator(operator) = &unit.node else {
                continue;
            };
            self.operators
                .entry(operator.name.node.clone())
                .or_insert_with(|| OperatorBinding {
                    params_len: operator.params.len(),
                    body: match substitutions {
                        Some(substitutions) if !substitutions.is_empty() => {
                            Cow::Owned(apply_substitutions(&operator.body, substitutions))
                        }
                        _ => Cow::Borrowed(&operator.body),
                    },
                });
        }
    }

    fn find_zero_arity(&self, name: &str) -> Result<&Spanned<Expr>, String> {
        let operator = self
            .operators
            .get(name)
            .ok_or_else(|| format!("operator `{name}` not found"))?;
        if operator.params_len == 0 {
            Ok(operator.body.as_ref())
        } else {
            Err(format!("operator `{name}` is not zero-arity"))
        }
    }

    fn zero_arity_body(&self, name: &str) -> RecognizeResult<Option<&Spanned<Expr>>> {
        match self.operators.get(name) {
            Some(operator) if operator.params_len == 0 => {
                RecognizeResult::Ok(Some(operator.body.as_ref()))
            }
            Some(_) => RecognizeResult::Ineligible(format!("operator `{name}` is not zero-arity")),
            None => RecognizeResult::Ok(None),
        }
    }
}

fn standalone_instance_substitutions_by_module(
    module: &Module,
) -> BTreeMap<String, Vec<Substitution>> {
    let mut substitutions = BTreeMap::new();
    for unit in &module.units {
        if let Unit::Instance(instance) = &unit.node {
            substitutions
                .entry(instance.module.node.clone())
                .or_insert_with(|| instance.substitutions.clone());
        }
    }
    substitutions
}

enum RecognizeResult<T> {
    Ok(T),
    Ineligible(String),
    Unknown(String),
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

fn recognize_record_predicate<'a>(
    expr: &'a Spanned<Expr>,
    env: &'a OperatorEnv<'_>,
) -> RecognizeResult<RecognizedRecordPredicate> {
    let mut stack = Vec::new();
    let mut conjuncts = Vec::new();
    match flatten_and(expr, env, &mut stack, &mut conjuncts) {
        RecognizeResult::Ok(()) => {}
        RecognizeResult::Ineligible(reason) => return RecognizeResult::Ineligible(reason),
        RecognizeResult::Unknown(reason) => return RecognizeResult::Unknown(reason),
    }

    let mut variable = None;
    let mut fields = None;
    let mut constraints = Vec::new();
    let mut non_membership_conjuncts = Vec::new();
    for conjunct in &conjuncts {
        match recognize_record_membership(conjunct, env) {
            RecognizeResult::Ok(Some(membership)) => {
                if variable.is_some() {
                    return RecognizeResult::Ineligible(
                        "record predicate contains multiple record memberships".to_string(),
                    );
                }
                variable = Some(membership.variable);
                fields = Some(membership.fields);
                constraints.extend(membership.constraints);
            }
            RecognizeResult::Ok(None) => non_membership_conjuncts.push(*conjunct),
            RecognizeResult::Ineligible(reason) => return RecognizeResult::Ineligible(reason),
            RecognizeResult::Unknown(reason) => return RecognizeResult::Unknown(reason),
        }
    }

    let Some(variable) = variable else {
        return RecognizeResult::Ineligible(
            "expected record membership `r \\in [field : lo..hi, ...]`".to_string(),
        );
    };
    let fields = fields.expect("membership sets fields with variable");

    for conjunct in non_membership_conjuncts {
        match recognize_constraint(conjunct, &variable, env, None) {
            RecognizeResult::Ok(Some(constraint)) => constraints.push(constraint),
            RecognizeResult::Ok(None) => {}
            RecognizeResult::Ineligible(reason) => return RecognizeResult::Ineligible(reason),
            RecognizeResult::Unknown(reason) => return RecognizeResult::Unknown(reason),
        }
    }

    RecognizeResult::Ok(RecognizedRecordPredicate {
        variable,
        fields,
        constraints,
    })
}

#[derive(Debug, Clone)]
struct RecognizedMembership {
    variable: String,
    fields: Vec<RecordFieldBox>,
    constraints: Vec<AffineConstraint>,
}

fn recognize_record_membership<'a>(
    expr: &'a Spanned<Expr>,
    env: &'a OperatorEnv<'_>,
) -> RecognizeResult<Option<RecognizedMembership>> {
    let Expr::In(lhs, rhs) = &expr.node else {
        return RecognizeResult::Ok(None);
    };
    let Some(variable) = record_variable_name(&lhs.node) else {
        return RecognizeResult::Ok(None);
    };

    let mut stack = Vec::new();
    match recognize_record_box(rhs, env, &mut stack) {
        RecognizeResult::Ok(fields) => {
            return RecognizeResult::Ok(Some(RecognizedMembership {
                variable,
                fields,
                constraints: Vec::new(),
            }));
        }
        RecognizeResult::Ineligible(_) => {}
        RecognizeResult::Unknown(reason) => return RecognizeResult::Unknown(reason),
    }

    match &rhs.node {
        Expr::SetFilter(bound, predicate) => {
            let Some(domain) = bound.domain.as_ref() else {
                return RecognizeResult::Ineligible(
                    "record init set filter must have a finite record-set domain".to_string(),
                );
            };
            let mut stack = Vec::new();
            let fields = match recognize_record_box(domain, env, &mut stack) {
                RecognizeResult::Ok(fields) => fields,
                RecognizeResult::Ineligible(reason) => return RecognizeResult::Ineligible(reason),
                RecognizeResult::Unknown(reason) => return RecognizeResult::Unknown(reason),
            };
            if bound.pattern.is_some() {
                return RecognizeResult::Ineligible(
                    "record init set filter must bind a simple record variable".to_string(),
                );
            }
            let mut constraints = Vec::new();
            let mut stack = Vec::new();
            let mut conjuncts = Vec::new();
            match flatten_and(predicate, env, &mut stack, &mut conjuncts) {
                RecognizeResult::Ok(()) => {}
                RecognizeResult::Ineligible(reason) => return RecognizeResult::Ineligible(reason),
                RecognizeResult::Unknown(reason) => return RecognizeResult::Unknown(reason),
            }
            for conjunct in conjuncts {
                match recognize_constraint(conjunct, &variable, env, Some(&bound.name.node)) {
                    RecognizeResult::Ok(Some(constraint)) => constraints.push(constraint),
                    RecognizeResult::Ok(None) => {}
                    RecognizeResult::Ineligible(reason) => {
                        return RecognizeResult::Ineligible(reason);
                    }
                    RecognizeResult::Unknown(reason) => return RecognizeResult::Unknown(reason),
                }
            }
            RecognizeResult::Ok(Some(RecognizedMembership {
                variable,
                fields,
                constraints,
            }))
        }
        _ => RecognizeResult::Ok(None),
    }
}

fn recognize_record_box<'a>(
    expr: &'a Spanned<Expr>,
    env: &'a OperatorEnv<'_>,
    stack: &mut Vec<String>,
) -> RecognizeResult<Vec<RecordFieldBox>> {
    if let Expr::Ident(name, _) = &expr.node {
        match resolve_body(name, env, stack) {
            RecognizeResult::Ok(Some(body)) => {
                let result = recognize_record_box(body, env, stack);
                stack.pop();
                return result;
            }
            RecognizeResult::Ok(None) => {}
            RecognizeResult::Ineligible(reason) => return RecognizeResult::Ineligible(reason),
            RecognizeResult::Unknown(reason) => return RecognizeResult::Unknown(reason),
        }
    }

    let Expr::RecordSet(fields) = &expr.node else {
        return RecognizeResult::Ineligible(
            "expected finite record set `[field : lo..hi, ...]`".to_string(),
        );
    };
    if fields.is_empty() {
        return RecognizeResult::Ineligible("record box must contain fields".to_string());
    }

    let mut boxes = Vec::with_capacity(fields.len());
    let mut seen = BTreeSet::new();
    for (field, value_set) in fields {
        if !seen.insert(field.node.clone()) {
            return RecognizeResult::Ineligible(format!("duplicate record field `{}`", field.node));
        }
        let interval = match recognize_integer_range(value_set, env, stack) {
            RecognizeResult::Ok(interval) => interval,
            RecognizeResult::Ineligible(reason) => return RecognizeResult::Ineligible(reason),
            RecognizeResult::Unknown(reason) => return RecognizeResult::Unknown(reason),
        };
        boxes.push(RecordFieldBox {
            name: field.node.clone(),
            interval,
        });
    }
    boxes.sort_by(|left, right| left.name.cmp(&right.name));
    RecognizeResult::Ok(boxes)
}

fn recognize_integer_range<'a>(
    expr: &'a Spanned<Expr>,
    env: &'a OperatorEnv<'_>,
    stack: &mut Vec<String>,
) -> RecognizeResult<IntegerInterval> {
    let Expr::Range(lower, upper) = &expr.node else {
        return RecognizeResult::Ineligible(
            "record field domain must be an integer range".to_string(),
        );
    };
    let lower = match recognize_integer_literal(lower, env, stack) {
        RecognizeResult::Ok(value) => value,
        RecognizeResult::Ineligible(reason) => return RecognizeResult::Ineligible(reason),
        RecognizeResult::Unknown(reason) => return RecognizeResult::Unknown(reason),
    };
    let upper = match recognize_integer_literal(upper, env, stack) {
        RecognizeResult::Ok(value) => value,
        RecognizeResult::Ineligible(reason) => return RecognizeResult::Ineligible(reason),
        RecognizeResult::Unknown(reason) => return RecognizeResult::Unknown(reason),
    };
    match IntegerInterval::new(lower, upper) {
        Some(interval) => RecognizeResult::Ok(interval),
        None => RecognizeResult::Ineligible("record field range is empty".to_string()),
    }
}

fn recognize_transition<'a>(
    expr: &'a Spanned<Expr>,
    variable: &str,
    env: &'a OperatorEnv<'_>,
) -> RecognizeResult<Vec<AffineRecordTransitionBranch>> {
    let mut stack = Vec::new();
    let mut disjuncts = Vec::new();
    match flatten_or(expr, env, &mut stack, &mut disjuncts) {
        RecognizeResult::Ok(()) => {}
        RecognizeResult::Ineligible(reason) => return RecognizeResult::Ineligible(reason),
        RecognizeResult::Unknown(reason) => return RecognizeResult::Unknown(reason),
    }

    if disjuncts.is_empty() {
        return RecognizeResult::Unknown("transition has no branches".to_string());
    }

    let mut branches = Vec::with_capacity(disjuncts.len());
    for disjunct in disjuncts {
        match recognize_branch(disjunct, variable, env) {
            RecognizeResult::Ok(branch) => branches.push(branch),
            RecognizeResult::Ineligible(reason) => return RecognizeResult::Ineligible(reason),
            RecognizeResult::Unknown(reason) => return RecognizeResult::Unknown(reason),
        }
    }
    RecognizeResult::Ok(branches)
}

fn recognize_branch<'a>(
    expr: &'a Spanned<Expr>,
    variable: &str,
    env: &'a OperatorEnv<'_>,
) -> RecognizeResult<AffineRecordTransitionBranch> {
    let mut stack = Vec::new();
    let mut conjuncts = Vec::new();
    match flatten_and(expr, env, &mut stack, &mut conjuncts) {
        RecognizeResult::Ok(()) => {}
        RecognizeResult::Ineligible(reason) => return RecognizeResult::Ineligible(reason),
        RecognizeResult::Unknown(reason) => return RecognizeResult::Unknown(reason),
    }

    let mut guard = Vec::new();
    let mut updates = None;
    let mut stutter = false;

    for conjunct in conjuncts {
        match recognize_update(conjunct, variable, env) {
            RecognizeResult::Ok(Some((branch_updates, branch_stutter))) => {
                if updates.is_some() {
                    return RecognizeResult::Ineligible(
                        "transition branch contains multiple record assignments".to_string(),
                    );
                }
                updates = Some(branch_updates);
                stutter = branch_stutter;
                continue;
            }
            RecognizeResult::Ok(None) => {}
            RecognizeResult::Ineligible(reason) => return RecognizeResult::Ineligible(reason),
            RecognizeResult::Unknown(reason) => return RecognizeResult::Unknown(reason),
        }

        match recognize_constraint(conjunct, variable, env, None) {
            RecognizeResult::Ok(Some(constraint)) => guard.push(constraint),
            RecognizeResult::Ok(None) => {}
            RecognizeResult::Ineligible(reason) => return RecognizeResult::Ineligible(reason),
            RecognizeResult::Unknown(reason) => return RecognizeResult::Unknown(reason),
        }
    }

    let Some(updates) = updates else {
        return RecognizeResult::Ineligible(
            "transition branch has no assignment to the primed record".to_string(),
        );
    };
    RecognizeResult::Ok(AffineRecordTransitionBranch {
        guard,
        updates,
        stutter,
    })
}

fn recognize_update<'a>(
    expr: &'a Spanned<Expr>,
    variable: &str,
    env: &'a OperatorEnv<'_>,
) -> RecognizeResult<Option<(Vec<FieldDeltaUpdate>, bool)>> {
    match &expr.node {
        Expr::Unchanged(inner)
            if record_variable_name(&inner.node).as_deref() == Some(variable) =>
        {
            RecognizeResult::Ok(Some((Vec::new(), true)))
        }
        Expr::Eq(lhs, rhs) if is_prime_of_record(&lhs.node, variable) => {
            recognize_update_rhs(rhs, variable, env)
        }
        Expr::Eq(lhs, rhs) if is_prime_of_record(&rhs.node, variable) => {
            recognize_update_rhs(lhs, variable, env)
        }
        _ => RecognizeResult::Ok(None),
    }
}

fn recognize_update_rhs<'a>(
    expr: &'a Spanned<Expr>,
    variable: &str,
    env: &'a OperatorEnv<'_>,
) -> RecognizeResult<Option<(Vec<FieldDeltaUpdate>, bool)>> {
    if record_variable_name(&expr.node).as_deref() == Some(variable) {
        return RecognizeResult::Ok(Some((Vec::new(), true)));
    }

    let Expr::Except(base, specs) = &expr.node else {
        return RecognizeResult::Ineligible(
            "record assignment must be identity or record EXCEPT update".to_string(),
        );
    };
    if record_variable_name(&base.node).as_deref() != Some(variable) {
        return RecognizeResult::Ineligible(
            "record EXCEPT update must use the current record as base".to_string(),
        );
    }
    if specs.is_empty() {
        return RecognizeResult::Ineligible("record EXCEPT update has no specs".to_string());
    }

    let mut updates = Vec::with_capacity(specs.len());
    let mut seen = BTreeSet::new();
    let mut stack = Vec::new();
    for spec in specs {
        let [ExceptPathElement::Field(field)] = spec.path.as_slice() else {
            return RecognizeResult::Ineligible(
                "record EXCEPT update must target direct record fields".to_string(),
            );
        };
        if !seen.insert(field.name.node.clone()) {
            return RecognizeResult::Ineligible(format!(
                "record EXCEPT updates field `{}` more than once",
                field.name.node
            ));
        }
        let delta = match recognize_at_shift(&spec.value, env, &mut stack) {
            RecognizeResult::Ok(delta) => delta,
            RecognizeResult::Ineligible(reason) => return RecognizeResult::Ineligible(reason),
            RecognizeResult::Unknown(reason) => return RecognizeResult::Unknown(reason),
        };
        updates.push(FieldDeltaUpdate {
            field: field.name.node.clone(),
            delta,
        });
    }
    updates.sort_by(|left, right| left.field.cmp(&right.field));
    let stutter = updates.iter().all(|update| update.delta.is_zero());
    RecognizeResult::Ok(Some((updates, stutter)))
}

fn recognize_at_shift<'a>(
    expr: &'a Spanned<Expr>,
    env: &'a OperatorEnv<'_>,
    stack: &mut Vec<String>,
) -> RecognizeResult<BigInt> {
    if let Expr::Ident(name, _) = &expr.node {
        if name == "@" {
            return RecognizeResult::Ok(BigInt::zero());
        }
        match resolve_body(name, env, stack) {
            RecognizeResult::Ok(Some(body)) => {
                let result = recognize_at_shift(body, env, stack);
                stack.pop();
                return result;
            }
            RecognizeResult::Ok(None) => {}
            RecognizeResult::Ineligible(reason) => return RecognizeResult::Ineligible(reason),
            RecognizeResult::Unknown(reason) => return RecognizeResult::Unknown(reason),
        }
    }

    match &expr.node {
        Expr::Add(lhs, rhs) if is_at(&lhs.node) => recognize_integer_literal(rhs, env, stack),
        Expr::Add(lhs, rhs) if is_at(&rhs.node) => recognize_integer_literal(lhs, env, stack),
        Expr::Sub(lhs, rhs) if is_at(&lhs.node) => {
            recognize_integer_literal(rhs, env, stack).map(|value| -value)
        }
        _ => RecognizeResult::Ineligible(
            "record EXCEPT value must be @, @ + n, or @ - n".to_string(),
        ),
    }
}

fn recognize_constraint<'a>(
    expr: &'a Spanned<Expr>,
    variable: &str,
    env: &'a OperatorEnv<'_>,
    alias: Option<&str>,
) -> RecognizeResult<Option<AffineConstraint>> {
    let mut stack = Vec::new();
    match &expr.node {
        Expr::Bool(true) => RecognizeResult::Ok(None),
        Expr::Eq(lhs, rhs) => recognize_comparison_constraint(
            lhs,
            rhs,
            ComparisonKind::Eq,
            variable,
            env,
            alias,
            &mut stack,
        ),
        Expr::Lt(lhs, rhs) => recognize_comparison_constraint(
            lhs,
            rhs,
            ComparisonKind::Lt,
            variable,
            env,
            alias,
            &mut stack,
        ),
        Expr::Leq(lhs, rhs) => recognize_comparison_constraint(
            lhs,
            rhs,
            ComparisonKind::Leq,
            variable,
            env,
            alias,
            &mut stack,
        ),
        Expr::Gt(lhs, rhs) => recognize_comparison_constraint(
            lhs,
            rhs,
            ComparisonKind::Gt,
            variable,
            env,
            alias,
            &mut stack,
        ),
        Expr::Geq(lhs, rhs) => recognize_comparison_constraint(
            lhs,
            rhs,
            ComparisonKind::Geq,
            variable,
            env,
            alias,
            &mut stack,
        ),
        Expr::In(lhs, rhs) => {
            let term = match recognize_affine_expr(lhs, variable, env, alias, &mut stack) {
                RecognizeResult::Ok(term) => term,
                RecognizeResult::Ineligible(reason) => return RecognizeResult::Ineligible(reason),
                RecognizeResult::Unknown(reason) => return RecognizeResult::Unknown(reason),
            };
            let interval = match recognize_integer_range(rhs, env, &mut stack) {
                RecognizeResult::Ok(interval) => interval,
                RecognizeResult::Ineligible(reason) => return RecognizeResult::Ineligible(reason),
                RecognizeResult::Unknown(reason) => return RecognizeResult::Unknown(reason),
            };
            normalized_constraint(
                term,
                Some(interval.lower().clone()),
                Some(interval.upper().clone()),
            )
        }
        _ => RecognizeResult::Ineligible(
            "unsupported affine record-counter guard or slice".to_string(),
        ),
    }
}

#[derive(Debug, Clone, Copy)]
enum ComparisonKind {
    Eq,
    Lt,
    Leq,
    Gt,
    Geq,
}

fn recognize_comparison_constraint<'a>(
    lhs: &'a Spanned<Expr>,
    rhs: &'a Spanned<Expr>,
    kind: ComparisonKind,
    variable: &str,
    env: &'a OperatorEnv<'_>,
    alias: Option<&str>,
    stack: &mut Vec<String>,
) -> RecognizeResult<Option<AffineConstraint>> {
    let lhs = match recognize_affine_expr(lhs, variable, env, alias, stack) {
        RecognizeResult::Ok(term) => term,
        RecognizeResult::Ineligible(reason) => return RecognizeResult::Ineligible(reason),
        RecognizeResult::Unknown(reason) => return RecognizeResult::Unknown(reason),
    };
    let rhs = match recognize_affine_expr(rhs, variable, env, alias, stack) {
        RecognizeResult::Ok(term) => term,
        RecognizeResult::Ineligible(reason) => return RecognizeResult::Ineligible(reason),
        RecognizeResult::Unknown(reason) => return RecognizeResult::Unknown(reason),
    };
    let difference = lhs.sub(&rhs);
    match kind {
        ComparisonKind::Eq => {
            normalized_constraint(difference, Some(BigInt::zero()), Some(BigInt::zero()))
        }
        ComparisonKind::Lt => normalized_constraint(difference, None, Some(-BigInt::one())),
        ComparisonKind::Leq => normalized_constraint(difference, None, Some(BigInt::zero())),
        ComparisonKind::Gt => normalized_constraint(difference, Some(BigInt::one()), None),
        ComparisonKind::Geq => normalized_constraint(difference, Some(BigInt::zero()), None),
    }
}

fn normalized_constraint(
    term: AffineTerm,
    lower: Option<BigInt>,
    upper: Option<BigInt>,
) -> RecognizeResult<Option<AffineConstraint>> {
    let Some(constraint) = AffineConstraint::new_normalized(term, lower, upper) else {
        return RecognizeResult::Unknown("inconsistent affine constraint".to_string());
    };
    if constraint.term.coefficients.is_empty() {
        let zero = BigInt::zero();
        if constraint.lower.as_ref().is_none_or(|lower| lower <= &zero)
            && constraint.upper.as_ref().is_none_or(|upper| &zero <= upper)
        {
            return RecognizeResult::Ok(None);
        }
        return RecognizeResult::Unknown("false constant affine constraint".to_string());
    }
    RecognizeResult::Ok(Some(constraint))
}

fn recognize_affine_expr<'a>(
    expr: &'a Spanned<Expr>,
    variable: &str,
    env: &'a OperatorEnv<'_>,
    alias: Option<&str>,
    stack: &mut Vec<String>,
) -> RecognizeResult<AffineTerm> {
    if let Expr::Ident(name, _) = &expr.node {
        match resolve_body(name, env, stack) {
            RecognizeResult::Ok(Some(body)) => {
                let result = recognize_affine_expr(body, variable, env, alias, stack);
                stack.pop();
                return result;
            }
            RecognizeResult::Ok(None) => {}
            RecognizeResult::Ineligible(reason) => return RecognizeResult::Ineligible(reason),
            RecognizeResult::Unknown(reason) => return RecognizeResult::Unknown(reason),
        }
    }

    match &expr.node {
        Expr::Int(value) => RecognizeResult::Ok(AffineTerm::from_constant(value.clone())),
        Expr::Neg(inner) => {
            recognize_affine_expr(inner, variable, env, alias, stack).map(|term| term.neg())
        }
        Expr::Add(lhs, rhs) => {
            let lhs = match recognize_affine_expr(lhs, variable, env, alias, stack) {
                RecognizeResult::Ok(term) => term,
                RecognizeResult::Ineligible(reason) => return RecognizeResult::Ineligible(reason),
                RecognizeResult::Unknown(reason) => return RecognizeResult::Unknown(reason),
            };
            recognize_affine_expr(rhs, variable, env, alias, stack).map(|rhs| lhs.add(&rhs))
        }
        Expr::Sub(lhs, rhs) => {
            let lhs = match recognize_affine_expr(lhs, variable, env, alias, stack) {
                RecognizeResult::Ok(term) => term,
                RecognizeResult::Ineligible(reason) => return RecognizeResult::Ineligible(reason),
                RecognizeResult::Unknown(reason) => return RecognizeResult::Unknown(reason),
            };
            recognize_affine_expr(rhs, variable, env, alias, stack).map(|rhs| lhs.sub(&rhs))
        }
        Expr::RecordAccess(record, field)
            if record_reference_name(&record.node).as_deref() == Some(variable)
                || alias.is_some_and(|alias| {
                    record_reference_name(&record.node).as_deref() == Some(alias)
                }) =>
        {
            RecognizeResult::Ok(AffineTerm::field(field.name.node.clone()))
        }
        _ => RecognizeResult::Ineligible(
            "expected affine expression over record integer fields".to_string(),
        ),
    }
}

fn recognize_integer_literal<'a>(
    expr: &'a Spanned<Expr>,
    env: &'a OperatorEnv<'_>,
    stack: &mut Vec<String>,
) -> RecognizeResult<BigInt> {
    if let Expr::Ident(name, _) = &expr.node {
        match resolve_body(name, env, stack) {
            RecognizeResult::Ok(Some(body)) => {
                let result = recognize_integer_literal(body, env, stack);
                stack.pop();
                return result;
            }
            RecognizeResult::Ok(None) => {}
            RecognizeResult::Ineligible(reason) => return RecognizeResult::Ineligible(reason),
            RecognizeResult::Unknown(reason) => return RecognizeResult::Unknown(reason),
        }
    }

    match &expr.node {
        Expr::Int(value) => RecognizeResult::Ok(value.clone()),
        Expr::Neg(inner) => recognize_integer_literal(inner, env, stack).map(|value| -value),
        _ => RecognizeResult::Ineligible("expected integer literal".to_string()),
    }
}

fn resolve_body<'a>(
    name: &str,
    env: &'a OperatorEnv<'_>,
    stack: &mut Vec<String>,
) -> RecognizeResult<Option<&'a Spanned<Expr>>> {
    if stack.iter().any(|entry| entry == name) {
        return RecognizeResult::Ineligible(format!(
            "recursive zero-arity operator `{name}` is unsupported"
        ));
    }
    match env.zero_arity_body(name) {
        RecognizeResult::Ok(Some(body)) => {
            stack.push(name.to_string());
            RecognizeResult::Ok(Some(body))
        }
        other => other,
    }
}

fn flatten_or<'a>(
    expr: &'a Spanned<Expr>,
    env: &'a OperatorEnv<'_>,
    stack: &mut Vec<String>,
    out: &mut Vec<&'a Spanned<Expr>>,
) -> RecognizeResult<()> {
    if let Expr::Ident(name, _) = &expr.node {
        match resolve_body(name, env, stack) {
            RecognizeResult::Ok(Some(body)) => {
                let result = flatten_or(body, env, stack, out);
                stack.pop();
                return result;
            }
            RecognizeResult::Ok(None) => {}
            RecognizeResult::Ineligible(reason) => return RecognizeResult::Ineligible(reason),
            RecognizeResult::Unknown(reason) => return RecognizeResult::Unknown(reason),
        }
    }

    match &expr.node {
        Expr::Or(lhs, rhs) => {
            match flatten_or(lhs, env, stack, out) {
                RecognizeResult::Ok(()) => {}
                RecognizeResult::Ineligible(reason) => return RecognizeResult::Ineligible(reason),
                RecognizeResult::Unknown(reason) => return RecognizeResult::Unknown(reason),
            }
            flatten_or(rhs, env, stack, out)
        }
        Expr::Bool(false) => RecognizeResult::Ok(()),
        _ => {
            out.push(expr);
            RecognizeResult::Ok(())
        }
    }
}

fn flatten_and<'a>(
    expr: &'a Spanned<Expr>,
    env: &'a OperatorEnv<'_>,
    stack: &mut Vec<String>,
    out: &mut Vec<&'a Spanned<Expr>>,
) -> RecognizeResult<()> {
    if let Expr::Ident(name, _) = &expr.node {
        match resolve_body(name, env, stack) {
            RecognizeResult::Ok(Some(body)) => {
                let result = flatten_and(body, env, stack, out);
                stack.pop();
                return result;
            }
            RecognizeResult::Ok(None) => {}
            RecognizeResult::Ineligible(reason) => return RecognizeResult::Ineligible(reason),
            RecognizeResult::Unknown(reason) => return RecognizeResult::Unknown(reason),
        }
    }

    match &expr.node {
        Expr::And(lhs, rhs) => {
            match flatten_and(lhs, env, stack, out) {
                RecognizeResult::Ok(()) => {}
                RecognizeResult::Ineligible(reason) => return RecognizeResult::Ineligible(reason),
                RecognizeResult::Unknown(reason) => return RecognizeResult::Unknown(reason),
            }
            flatten_and(rhs, env, stack, out)
        }
        Expr::Bool(true) => RecognizeResult::Ok(()),
        _ => {
            out.push(expr);
            RecognizeResult::Ok(())
        }
    }
}

fn record_variable_name(expr: &Expr) -> Option<String> {
    match expr {
        Expr::Ident(name, _) | Expr::StateVar(name, _, _) => Some(name.clone()),
        _ => None,
    }
}

fn record_reference_name(expr: &Expr) -> Option<String> {
    record_variable_name(expr)
}

fn is_prime_of_record(expr: &Expr, variable: &str) -> bool {
    match expr {
        Expr::Prime(inner) => record_variable_name(&inner.node).as_deref() == Some(variable),
        _ => false,
    }
}

fn is_at(expr: &Expr) -> bool {
    matches!(expr, Expr::Ident(name, _) if name == "@")
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
    ) -> AffineRecordCounterOutcome {
        let module = parse_module(source);
        recognize_module_affine_record_counter(&module, init, next, invariant)
    }

    fn candidate_for(
        source: &str,
        init: &str,
        next: &str,
        invariant: &str,
    ) -> CandidateProof<AffineRecordCounterCertificate> {
        match outcome_for(source, init, next, invariant) {
            AnalyticalOutcome::CandidateProof(candidate) => candidate,
            other => panic!("expected affine record-counter candidate, got {other:?}"),
        }
    }

    fn admission_for(
        source: &str,
        init: &str,
        next: &str,
        invariant: &str,
    ) -> AffineRecordCounterAdmissionOutcome {
        let module = parse_module(source);
        admit_module_affine_record_counter(&module, init, next, invariant)
    }

    #[test]
    fn recognizes_and_verifies_renamed_coffeecan_shape() {
        let candidate = candidate_for(
            r#"
---- MODULE RenamedJar ----
EXTENDS Integers
VARIABLE jar
Limit == 4
Bucket == [dark : 0..Limit, light : 0..Limit]
Mass == jar.dark + jar.light
Start == jar \in {r \in Bucket : r.dark + r.light = Limit}
SameLight ==
    /\ Mass > 1
    /\ jar.light >= 2
    /\ jar' = [jar EXCEPT !.dark = @ + 1, !.light = @ - 2]
SameDark ==
    /\ Mass > 1
    /\ jar.dark >= 2
    /\ jar' = [jar EXCEPT !.dark = @ - 1]
Different ==
    /\ Mass > 1
    /\ jar.dark >= 1
    /\ jar.light >= 1
    /\ jar' = [jar EXCEPT !.dark = @ - 1]
Done ==
    /\ Mass = 1
    /\ UNCHANGED jar
Step == SameLight \/ SameDark \/ Different \/ Done
TypeOK == jar \in Bucket
====
"#,
            "Start",
            "Step",
            "TypeOK",
        );

        let certificate = candidate.certificate();
        assert_eq!(certificate.variable(), "jar");
        assert_eq!(certificate.fields().len(), 2);
        assert_eq!(certificate.transition_branches().len(), 4);
        assert_eq!(certificate.box_cardinality(), BigInt::from(25));
        assert!(certificate.terminal_stutter_covered());
        assert_eq!(certificate.invariant_constraints().len(), 1);
        assert_eq!(
            certificate.invariant_constraints()[0].lower(),
            Some(&BigInt::from(1))
        );
        assert_eq!(
            certificate.invariant_constraints()[0].upper(),
            Some(&BigInt::from(4))
        );

        verify_affine_record_counter_candidate(candidate).expect("certificate verifies");
    }

    #[test]
    fn admission_is_proof_gated() {
        let admission = admission_for(
            r#"
---- MODULE GateJar ----
EXTENDS Integers
VARIABLE bag
N == 3
Box == [a : 0..N, b : 0..N]
Total == bag.a + bag.b
Init == bag \in {x \in Box : x.a + x.b = N}
DecA == /\ Total > 1 /\ bag.a >= 1 /\ bag' = [bag EXCEPT !.a = @ - 1]
DecB == /\ Total > 1 /\ bag.b >= 1 /\ bag' = [bag EXCEPT !.b = @ - 1]
Stop == /\ Total = 1 /\ UNCHANGED bag
Next == DecA \/ DecB \/ Stop
Inv == bag \in Box
====
"#,
            "Init",
            "Next",
            "Inv",
        );

        let AnalyticalAdmission::VerifiedProof(verified) = admission else {
            panic!("expected verified affine record-counter proof");
        };
        assert_eq!(verified.certificate().init_operator(), "Init");
        assert_eq!(verified.certificate().proof().variable(), "bag");
    }

    #[test]
    fn candidate_proof_requires_verification_gate_before_publish() {
        let candidate = candidate_for(
            r#"
---- MODULE PublishGateJar ----
EXTENDS Integers
VARIABLE r
Box == [x : 0..2, y : 0..2]
Sum == r.x + r.y
Init == r \in {v \in Box : v.x + v.y = 2}
Move == /\ Sum > 1 /\ r.x >= 1 /\ r' = [r EXCEPT !.x = @ - 1]
Stop == /\ Sum = 1 /\ UNCHANGED r
Next == Move \/ Stop
Inv == r \in Box
====
"#,
            "Init",
            "Next",
            "Inv",
        );

        let shared = SharedVerdict::new();
        let gate = VerificationGate::new(&shared);
        let raw: AffineRecordCounterOutcome = AnalyticalOutcome::CandidateProof(candidate.clone());
        assert_eq!(gate.inspect(&raw), GateDecision::NeedsProof);
        assert!(!shared.is_resolved());

        let verified =
            verify_affine_record_counter_candidate(candidate).expect("candidate verifies");
        assert_eq!(
            gate.publish_verified_proof(verified),
            GateDecision::Published
        );
        assert_eq!(shared.get(), Some(Verdict::Satisfied));
    }

    #[test]
    fn unsafe_branch_fails_closed_at_admission() {
        let admission = admission_for(
            r#"
---- MODULE UnsafeJar ----
EXTENDS Integers
VARIABLE jar
N == 4
Box == [black : 0..N, white : 0..N]
Init == jar \in Box
Next == /\ jar.white >= 2
        /\ jar' = [jar EXCEPT !.black = @ + 1, !.white = @ - 2]
Inv == jar \in Box
====
"#,
            "Init",
            "Next",
            "Inv",
        );

        match admission {
            AnalyticalAdmission::Unknown(reason) => {
                assert!(reason.reason().contains("verification failed"));
            }
            other => panic!("expected verification failure to stay Unknown, got {other:?}"),
        }
    }

    #[test]
    fn unsupported_update_shape_is_ineligible() {
        let outcome = outcome_for(
            r#"
---- MODULE UnsupportedJar ----
EXTENDS Integers
VARIABLE jar
Box == [x : 0..5]
Init == jar \in Box
Next == jar' = [jar EXCEPT !.x = @ * 2]
Inv == jar \in Box
====
"#,
            "Init",
            "Next",
            "Inv",
        );

        assert!(matches!(outcome, AnalyticalOutcome::Ineligible(_)));
    }

    #[test]
    fn record_box_fields_are_canonicalized_independent_of_source_order() {
        let candidate = candidate_for(
            r#"
---- MODULE ReorderedFields ----
EXTENDS Integers
VARIABLE r
N == 2
Box == [z : 0..N, a : 0..N]
Init == r \in Box
Next == UNCHANGED r
Inv == r \in Box
====
"#,
            "Init",
            "Next",
            "Inv",
        );

        let certificate = candidate.certificate();
        let field_names: Vec<_> = certificate
            .fields()
            .iter()
            .map(RecordFieldBox::name)
            .collect();
        assert_eq!(field_names, vec!["a", "z"]);
        verify_affine_record_counter_candidate(candidate).expect("certificate verifies");
    }
}
