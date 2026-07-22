// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Proof-gated recognizer for state-independent finite-cardinality facts.
//!
//! This module is intentionally narrow: it recognizes only invariant operators
//! made from static finite-cardinality facts over literal finite sets, integer
//! ranges, `SUBSET`, finite function sets, and zero-arity operators that resolve
//! to those forms. It never publishes a result directly; callers must pass the
//! emitted certificate through the verifier before using the proof.

use num_bigint::BigInt;
use num_traits::{One, ToPrimitive, Zero};
use std::collections::{BTreeMap, BTreeSet};
use tla_core::ast::{Expr, Module, OperatorDef, Unit};
use tla_core::span::Spanned;

use super::{
    AnalyticalAdmission, AnalyticalOutcome, CandidateProof, Ineligible, Unknown, VerifiedProof,
};

const MAX_STATIC_VALUE_SET_ELEMENTS: usize = 1024;

macro_rules! try_recognize {
    ($expr:expr) => {
        match $expr {
            RecognizeResult::Ok(value) => value,
            RecognizeResult::Ineligible(reason) => {
                return RecognizeResult::Ineligible(reason);
            }
            RecognizeResult::Unknown(reason) => {
                return RecognizeResult::Unknown(reason);
            }
        }
    };
}

/// Outcome type used by the finite-cardinality recognizer.
pub type FiniteCardinalityOutcome = AnalyticalOutcome<(), FiniteCardinalityCertificate>;

/// Proof-gated admission result for static finite-cardinality invariants.
pub type FiniteCardinalityAdmissionOutcome =
    AnalyticalAdmission<(), FiniteCardinalityAdmissionCertificate>;

/// Certificate emitted by the finite-cardinality structural recognizer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FiniteCardinalityCertificate {
    facts: Vec<FiniteCardinalityFact>,
}

impl FiniteCardinalityCertificate {
    /// Recognized finite-cardinality facts, in source conjunction order.
    pub fn facts(&self) -> &[FiniteCardinalityFact] {
        &self.facts
    }

    /// Number of static facts covered by the certificate.
    pub fn fact_count(&self) -> usize {
        self.facts.len()
    }
}

/// Verified finite-cardinality certificate with source metadata.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FiniteCardinalityAdmissionCertificate {
    invariant_operator: String,
    proof: FiniteCardinalityCertificate,
}

impl FiniteCardinalityAdmissionCertificate {
    /// Invariant operator supplied by the caller/config.
    pub fn invariant_operator(&self) -> &str {
        &self.invariant_operator
    }

    /// Verified proof payload.
    pub fn proof(&self) -> &FiniteCardinalityCertificate {
        &self.proof
    }

    /// Number of static facts covered by the proof.
    pub fn fact_count(&self) -> usize {
        self.proof.fact_count()
    }
}

/// One recognized static finite-cardinality fact.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FiniteCardinalityFact {
    /// `IsFiniteSet(S)` for a recognized finite set expression.
    IsFiniteSet {
        /// The finite set certificate for `S`.
        set: FiniteSetCertificate,
    },
    /// `Cardinality(S) <op> n` for a recognized finite set expression.
    CardinalityComparison {
        /// The finite set certificate for `S`.
        set: FiniteSetCertificate,
        /// The comparison operator.
        comparison: CardinalityComparison,
        /// The right-hand-side value `n`.
        expected: BigInt,
    },
}

impl FiniteCardinalityFact {
    /// The finite set expression proven by this fact.
    pub fn set(&self) -> &FiniteSetCertificate {
        match self {
            Self::IsFiniteSet { set } | Self::CardinalityComparison { set, .. } => set,
        }
    }

    /// Cardinality recomputed from the finite-set certificate.
    pub fn actual_cardinality(&self) -> Option<BigInt> {
        self.set().try_cardinality()
    }

    fn verify(&self) -> Result<(), FiniteCardinalityVerificationError> {
        verify_set_certificate(self.set())?;
        let actual = self.set().try_cardinality().ok_or_else(|| {
            FiniteCardinalityVerificationError::CardinalityTooLarge {
                context: "fact set".to_string(),
            }
        })?;
        match self {
            Self::IsFiniteSet { .. } => Ok(()),
            Self::CardinalityComparison {
                comparison,
                expected,
                ..
            } if comparison.holds(&actual, expected) => Ok(()),
            Self::CardinalityComparison {
                comparison,
                expected,
                ..
            } => Err(FiniteCardinalityVerificationError::FalseFact {
                actual,
                comparison: *comparison,
                expected: expected.clone(),
            }),
        }
    }
}

/// Supported cardinality comparison.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CardinalityComparison {
    /// `Cardinality(S) = n`
    Eq,
    /// `Cardinality(S) /= n`
    Neq,
    /// `Cardinality(S) < n`
    Lt,
    /// `Cardinality(S) <= n`
    Leq,
    /// `Cardinality(S) > n`
    Gt,
    /// `Cardinality(S) >= n`
    Geq,
}

impl CardinalityComparison {
    fn holds(self, actual: &BigInt, expected: &BigInt) -> bool {
        match self {
            Self::Eq => actual == expected,
            Self::Neq => actual != expected,
            Self::Lt => actual < expected,
            Self::Leq => actual <= expected,
            Self::Gt => actual > expected,
            Self::Geq => actual >= expected,
        }
    }

    fn reversed(self) -> Self {
        match self {
            Self::Eq => Self::Eq,
            Self::Neq => Self::Neq,
            Self::Lt => Self::Gt,
            Self::Leq => Self::Geq,
            Self::Gt => Self::Lt,
            Self::Geq => Self::Leq,
        }
    }
}

/// Structural proof that a set expression is finite.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FiniteSetCertificate {
    /// Literal set with canonical, sorted, deduplicated elements.
    LiteralSet {
        /// The set's elements in canonical order.
        elements: Vec<StaticValue>,
    },
    /// Integer range `lo..hi`; empty ranges are finite with cardinality zero.
    IntegerRange {
        /// Inclusive lower bound.
        lower: BigInt,
        /// Inclusive upper bound.
        upper: BigInt,
    },
    /// Powerset `SUBSET S`, finite because `S` is finite.
    Powerset {
        /// Certificate for the base set `S`.
        base: Box<FiniteSetCertificate>,
    },
    /// Function set `[S -> T]`, finite because both `S` and `T` are finite.
    FunctionSet {
        /// Certificate for the domain set `S`.
        domain: Box<FiniteSetCertificate>,
        /// Certificate for the codomain set `T`.
        codomain: Box<FiniteSetCertificate>,
    },
    /// Zero-arity operator resolved to a finite set expression.
    ResolvedConstant {
        /// The operator's name.
        name: String,
        /// Certificate for the operator's resolved body.
        body: Box<FiniteSetCertificate>,
    },
}

impl FiniteSetCertificate {
    /// Cardinality recomputed from this finite-set certificate.
    pub fn try_cardinality(&self) -> Option<BigInt> {
        match self {
            Self::LiteralSet { elements } => Some(BigInt::from(elements.len())),
            Self::IntegerRange { lower, upper } if upper < lower => Some(BigInt::zero()),
            Self::IntegerRange { lower, upper } => Some(upper - lower + BigInt::one()),
            Self::Powerset { base } => pow_bigint(&BigInt::from(2_u8), &base.try_cardinality()?),
            Self::FunctionSet { domain, codomain } => {
                pow_bigint(&codomain.try_cardinality()?, &domain.try_cardinality()?)
            }
            Self::ResolvedConstant { body, .. } => body.try_cardinality(),
        }
    }
}

/// Canonical static values supported as literal set elements.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum StaticValue {
    /// Boolean literal.
    Bool(bool),
    /// Integer literal.
    Int(BigInt),
    /// String literal.
    String(String),
    /// Tuple of static values.
    Tuple(Vec<StaticValue>),
    /// Set of canonical, sorted, deduplicated static values.
    Set(Vec<StaticValue>),
    /// Record with field names sorted lexicographically.
    Record(Vec<(String, StaticValue)>),
}

/// Verification error for a recognized finite-cardinality proof.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FiniteCardinalityVerificationError {
    /// Certificate contains no facts.
    EmptyCertificate,
    /// A literal set or static set is not sorted and deduplicated.
    NonCanonicalStaticSet,
    /// A static record contains duplicate or unsorted fields.
    NonCanonicalStaticRecord,
    /// A resolved constant certificate has an empty name.
    EmptyResolvedConstantName,
    /// A powerset/function-set exponent was too large to recompute.
    CardinalityTooLarge {
        /// Context describing which cardinality computation overflowed.
        context: String,
    },
    /// The certificate records a fact that is not true.
    FalseFact {
        /// The actual recomputed cardinality.
        actual: BigInt,
        /// The comparison the fact claimed.
        comparison: CardinalityComparison,
        /// The value the fact compared against.
        expected: BigInt,
    },
}

/// Verify a finite-cardinality candidate proof and return a publishable proof.
pub fn verify_finite_cardinality_candidate(
    candidate: CandidateProof<FiniteCardinalityCertificate>,
) -> Result<VerifiedProof<FiniteCardinalityCertificate>, FiniteCardinalityVerificationError> {
    let certificate = candidate.into_certificate();
    verify_certificate(&certificate)?;
    Ok(VerifiedProof::new(certificate))
}

/// Verify a finite-cardinality certificate without publishing.
pub fn verify_certificate(
    certificate: &FiniteCardinalityCertificate,
) -> Result<(), FiniteCardinalityVerificationError> {
    if certificate.facts.is_empty() {
        return Err(FiniteCardinalityVerificationError::EmptyCertificate);
    }
    for fact in &certificate.facts {
        fact.verify()?;
    }
    Ok(())
}

/// Recognize a module invariant by caller-provided operator name.
///
/// The operator name is only a lookup key supplied by the caller, typically from
/// configuration. The recognizer does not inspect the module name or use any
/// spec-name shortcut.
pub fn recognize_module_finite_cardinality_invariant(
    module: &Module,
    invariant_operator: &str,
) -> FiniteCardinalityOutcome {
    recognize_module_set_finite_cardinality_invariant(module, &[], invariant_operator)
}

/// Recognize an invariant from the root module or loaded checker modules.
///
/// If the invariant is owned by a checker module, that module is placed first in
/// the recognition environment so local helper operators resolve before root
/// or EXTENDS-loaded helpers with the same unqualified name.
pub fn recognize_module_set_finite_cardinality_invariant(
    module: &Module,
    checker_modules: &[&Module],
    invariant_operator: &str,
) -> FiniteCardinalityOutcome {
    let modules: Vec<&Module> = std::iter::once(module)
        .chain(checker_modules.iter().copied())
        .collect();
    let (owner_idx, invariant) =
        match find_zero_arity_operator_in_module_set(&modules, invariant_operator) {
            Ok(op) => op,
            Err(reason) => return AnalyticalOutcome::Ineligible(Ineligible::new(reason)),
        };
    let mut env_modules = Vec::with_capacity(modules.len());
    env_modules.push(modules[owner_idx]);
    for (idx, module) in modules.iter().copied().enumerate() {
        if idx != owner_idx {
            env_modules.push(module);
        }
    }
    let env = RecognitionEnv::from_modules(env_modules);
    recognize_finite_cardinality_expr(&invariant.body, &env)
}

/// Recognize, verify, and admit a module-level static finite-cardinality invariant.
///
/// This is the narrow integration point for a normal checker: after config/spec
/// resolution has selected an invariant operator, call this function and publish
/// only the returned `VerifiedProof` admission.
pub fn admit_module_finite_cardinality_invariant(
    module: &Module,
    invariant_operator: &str,
) -> FiniteCardinalityAdmissionOutcome {
    admit_module_set_finite_cardinality_invariant(module, &[], invariant_operator)
}

/// Recognize, verify, and admit a static finite-cardinality invariant from the
/// same root-plus-checker module scope used for analytical eligibility.
pub fn admit_module_set_finite_cardinality_invariant(
    module: &Module,
    checker_modules: &[&Module],
    invariant_operator: &str,
) -> FiniteCardinalityAdmissionOutcome {
    match recognize_module_set_finite_cardinality_invariant(
        module,
        checker_modules,
        invariant_operator,
    ) {
        AnalyticalOutcome::CandidateProof(candidate) => {
            let proof = candidate.into_certificate();
            match verify_certificate(&proof) {
                Ok(()) => AnalyticalAdmission::VerifiedProof(VerifiedProof::new(
                    FiniteCardinalityAdmissionCertificate {
                        invariant_operator: invariant_operator.to_string(),
                        proof,
                    },
                )),
                Err(error) => AnalyticalAdmission::Ineligible(Ineligible::new(format!(
                    "finite-cardinality certificate verification failed: {error:?}"
                ))),
            }
        }
        AnalyticalOutcome::Unknown(reason) => AnalyticalAdmission::Unknown(reason),
        AnalyticalOutcome::Ineligible(reason) => AnalyticalAdmission::Ineligible(reason),
        AnalyticalOutcome::CandidateViolation(_) => AnalyticalAdmission::Ineligible(
            Ineligible::new("finite-cardinality recognizer does not emit violation candidates"),
        ),
    }
}

/// Recognize a finite-cardinality proof candidate from an invariant expression.
pub fn recognize_finite_cardinality_expr(
    invariant: &Spanned<Expr>,
    env: &RecognitionEnv<'_>,
) -> FiniteCardinalityOutcome {
    let mut facts = Vec::new();
    let mut stack = Vec::new();
    match recognize_facts(invariant, env, &mut stack, &mut facts) {
        RecognizeResult::Ok(()) if facts.is_empty() => AnalyticalOutcome::Ineligible(
            Ineligible::new("expected at least one finite-cardinality fact"),
        ),
        RecognizeResult::Ok(()) => {
            AnalyticalOutcome::CandidateProof(CandidateProof::new(FiniteCardinalityCertificate {
                facts,
            }))
        }
        RecognizeResult::Ineligible(reason) => {
            AnalyticalOutcome::Ineligible(Ineligible::new(reason))
        }
        RecognizeResult::Unknown(reason) => AnalyticalOutcome::Unknown(Unknown::new(reason)),
    }
}

/// Immutable module context used by the recognizer.
#[derive(Debug)]
pub struct RecognitionEnv<'a> {
    operators: BTreeMap<String, &'a OperatorDef>,
    variables: BTreeSet<String>,
    constants: BTreeSet<String>,
}

impl<'a> RecognitionEnv<'a> {
    /// Build recognition context from a lowered module.
    pub fn new(module: &'a Module) -> Self {
        Self::from_modules(std::iter::once(module))
    }

    fn from_modules<I>(modules: I) -> Self
    where
        I: IntoIterator<Item = &'a Module>,
    {
        let mut operators = BTreeMap::new();
        let mut variables = BTreeSet::new();
        let mut constants = BTreeSet::new();

        for module in modules {
            for unit in &module.units {
                match &unit.node {
                    Unit::Operator(op) => {
                        operators.entry(op.name.node.clone()).or_insert(op);
                    }
                    Unit::Variable(names) => {
                        variables.extend(names.iter().map(|name| name.node.clone()));
                    }
                    Unit::Constant(decls) => {
                        constants.extend(decls.iter().map(|decl| decl.name.node.clone()));
                    }
                    _ => {}
                }
            }
        }

        Self {
            operators,
            variables,
            constants,
        }
    }
}

enum RecognizeResult<T> {
    Ok(T),
    Ineligible(String),
    Unknown(String),
}

fn recognize_facts(
    expr: &Spanned<Expr>,
    env: &RecognitionEnv<'_>,
    stack: &mut Vec<String>,
    facts: &mut Vec<FiniteCardinalityFact>,
) -> RecognizeResult<()> {
    match &expr.node {
        Expr::And(lhs, rhs) => {
            try_recognize!(recognize_facts(lhs, env, stack, facts));
            recognize_facts(rhs, env, stack, facts)
        }
        Expr::Bool(true) => RecognizeResult::Ok(()),
        Expr::Bool(false) => {
            RecognizeResult::Ineligible("finite-cardinality invariant contains FALSE".to_string())
        }
        Expr::Ident(name, _) => {
            let op = match resolve_zero_arity_operator(env, name, stack) {
                RecognizeResult::Ok(op) => op,
                RecognizeResult::Ineligible(reason) => {
                    return RecognizeResult::Ineligible(reason);
                }
                RecognizeResult::Unknown(reason) => return RecognizeResult::Unknown(reason),
            };
            stack.push(name.clone());
            let result = recognize_facts(&op.body, env, stack, facts);
            stack.pop();
            result
        }
        _ if matches_unary_operator(expr, "ConstCardinality").is_some() => {
            let inner = matches_unary_operator(expr, "ConstCardinality").expect("checked above");
            recognize_facts(inner, env, stack, facts)
        }
        Expr::Prime(_) => RecognizeResult::Ineligible(
            "finite-cardinality facts must be state-independent; primed expression found"
                .to_string(),
        ),
        Expr::StateVar(name, _, _) => RecognizeResult::Ineligible(format!(
            "finite-cardinality facts must be state-independent; state variable `{name}` found"
        )),
        _ => {
            let fact = try_recognize!(recognize_single_fact(expr, env, stack));
            facts.push(fact);
            RecognizeResult::Ok(())
        }
    }
}

fn recognize_single_fact(
    expr: &Spanned<Expr>,
    env: &RecognitionEnv<'_>,
    stack: &mut Vec<String>,
) -> RecognizeResult<FiniteCardinalityFact> {
    if let Some(arg) = matches_unary_operator(expr, "IsFiniteSet") {
        let set = try_recognize!(recognize_finite_set(arg, env, stack));
        return RecognizeResult::Ok(FiniteCardinalityFact::IsFiniteSet { set });
    }

    match &expr.node {
        Expr::Eq(lhs, rhs) => {
            recognize_cardinality_comparison(lhs, rhs, CardinalityComparison::Eq, env, stack)
        }
        Expr::Neq(lhs, rhs) => {
            recognize_cardinality_comparison(lhs, rhs, CardinalityComparison::Neq, env, stack)
        }
        Expr::Lt(lhs, rhs) => {
            recognize_cardinality_comparison(lhs, rhs, CardinalityComparison::Lt, env, stack)
        }
        Expr::Leq(lhs, rhs) => {
            recognize_cardinality_comparison(lhs, rhs, CardinalityComparison::Leq, env, stack)
        }
        Expr::Gt(lhs, rhs) => {
            recognize_cardinality_comparison(lhs, rhs, CardinalityComparison::Gt, env, stack)
        }
        Expr::Geq(lhs, rhs) => {
            recognize_cardinality_comparison(lhs, rhs, CardinalityComparison::Geq, env, stack)
        }
        _ => RecognizeResult::Ineligible(
            "expected IsFiniteSet(S) or Cardinality(S) compared with an integer".to_string(),
        ),
    }
}

fn recognize_cardinality_comparison(
    lhs: &Spanned<Expr>,
    rhs: &Spanned<Expr>,
    comparison: CardinalityComparison,
    env: &RecognitionEnv<'_>,
    stack: &mut Vec<String>,
) -> RecognizeResult<FiniteCardinalityFact> {
    if let Some(set_expr) = matches_unary_operator(lhs, "Cardinality") {
        let set = try_recognize!(recognize_finite_set(set_expr, env, stack));
        let expected = try_recognize!(recognize_integer(rhs, env, stack));
        return finish_cardinality_comparison(set, comparison, expected);
    }

    if let Some(set_expr) = matches_unary_operator(rhs, "Cardinality") {
        let expected = try_recognize!(recognize_integer(lhs, env, stack));
        let set = try_recognize!(recognize_finite_set(set_expr, env, stack));
        return finish_cardinality_comparison(set, comparison.reversed(), expected);
    }

    RecognizeResult::Ineligible("expected one side of comparison to be Cardinality(S)".to_string())
}

fn finish_cardinality_comparison(
    set: FiniteSetCertificate,
    comparison: CardinalityComparison,
    expected: BigInt,
) -> RecognizeResult<FiniteCardinalityFact> {
    let Some(actual) = set.try_cardinality() else {
        return RecognizeResult::Unknown(
            "finite-cardinality comparison exponent is too large".to_string(),
        );
    };
    if comparison.holds(&actual, &expected) {
        RecognizeResult::Ok(FiniteCardinalityFact::CardinalityComparison {
            set,
            comparison,
            expected,
        })
    } else {
        RecognizeResult::Ineligible(format!(
            "finite-cardinality comparison is false: actual cardinality {actual:?}"
        ))
    }
}

fn recognize_finite_set(
    expr: &Spanned<Expr>,
    env: &RecognitionEnv<'_>,
    stack: &mut Vec<String>,
) -> RecognizeResult<FiniteSetCertificate> {
    match &expr.node {
        Expr::SetEnum(elements) => {
            let mut values = Vec::with_capacity(elements.len());
            for element in elements {
                values.push(try_recognize!(recognize_static_value(element, env, stack)));
            }
            canonicalize_static_values(&mut values);
            RecognizeResult::Ok(FiniteSetCertificate::LiteralSet { elements: values })
        }
        Expr::Range(lower, upper) => {
            let lower = try_recognize!(recognize_integer(lower, env, stack));
            let upper = try_recognize!(recognize_integer(upper, env, stack));
            RecognizeResult::Ok(FiniteSetCertificate::IntegerRange { lower, upper })
        }
        Expr::Powerset(base) => {
            let base = try_recognize!(recognize_finite_set(base, env, stack));
            RecognizeResult::Ok(FiniteSetCertificate::Powerset {
                base: Box::new(base),
            })
        }
        Expr::FuncSet(domain, codomain) => {
            let domain = try_recognize!(recognize_finite_set(domain, env, stack));
            let codomain = try_recognize!(recognize_finite_set(codomain, env, stack));
            RecognizeResult::Ok(FiniteSetCertificate::FunctionSet {
                domain: Box::new(domain),
                codomain: Box::new(codomain),
            })
        }
        Expr::Ident(name, _) => {
            let op = match resolve_zero_arity_operator(env, name, stack) {
                RecognizeResult::Ok(op) => op,
                RecognizeResult::Ineligible(reason) => {
                    return RecognizeResult::Ineligible(reason);
                }
                RecognizeResult::Unknown(reason) => return RecognizeResult::Unknown(reason),
            };
            stack.push(name.clone());
            let body = match recognize_finite_set(&op.body, env, stack) {
                RecognizeResult::Ok(body) => body,
                RecognizeResult::Ineligible(reason) => {
                    stack.pop();
                    return RecognizeResult::Ineligible(reason);
                }
                RecognizeResult::Unknown(reason) => {
                    stack.pop();
                    return RecognizeResult::Unknown(reason);
                }
            };
            stack.pop();
            RecognizeResult::Ok(FiniteSetCertificate::ResolvedConstant {
                name: name.clone(),
                body: Box::new(body),
            })
        }
        Expr::Prime(_) => RecognizeResult::Ineligible(
            "finite-cardinality set expression must be state-independent; primed expression found"
                .to_string(),
        ),
        Expr::StateVar(name, _, _) => RecognizeResult::Ineligible(format!(
            "finite-cardinality set expression must be state-independent; state variable `{name}` found"
        )),
        _ => RecognizeResult::Ineligible(
            "unsupported finite-cardinality set expression".to_string(),
        ),
    }
}

fn recognize_static_value(
    expr: &Spanned<Expr>,
    env: &RecognitionEnv<'_>,
    stack: &mut Vec<String>,
) -> RecognizeResult<StaticValue> {
    match &expr.node {
        Expr::Bool(value) => RecognizeResult::Ok(StaticValue::Bool(*value)),
        Expr::Int(value) => RecognizeResult::Ok(StaticValue::Int(value.clone())),
        Expr::String(value) => RecognizeResult::Ok(StaticValue::String(value.clone())),
        Expr::Neg(inner) => {
            let value = try_recognize!(recognize_integer(inner, env, stack));
            RecognizeResult::Ok(StaticValue::Int(-value))
        }
        Expr::Tuple(elements) => {
            let mut values = Vec::with_capacity(elements.len());
            for element in elements {
                values.push(try_recognize!(recognize_static_value(element, env, stack)));
            }
            RecognizeResult::Ok(StaticValue::Tuple(values))
        }
        Expr::Record(fields) => {
            let mut values = Vec::with_capacity(fields.len());
            for (name, value_expr) in fields {
                values.push((
                    name.node.clone(),
                    try_recognize!(recognize_static_value(value_expr, env, stack)),
                ));
            }
            values.sort_by(|lhs, rhs| lhs.0.cmp(&rhs.0));
            if values.windows(2).any(|pair| pair[0].0 == pair[1].0) {
                return RecognizeResult::Ineligible(
                    "literal record contains duplicate fields".to_string(),
                );
            }
            RecognizeResult::Ok(StaticValue::Record(values))
        }
        Expr::SetEnum(elements) => {
            let mut values = Vec::with_capacity(elements.len());
            for element in elements {
                values.push(try_recognize!(recognize_static_value(element, env, stack)));
            }
            canonicalize_static_values(&mut values);
            RecognizeResult::Ok(StaticValue::Set(values))
        }
        Expr::Range(lower, upper) => {
            let lower = try_recognize!(recognize_integer(lower, env, stack));
            let upper = try_recognize!(recognize_integer(upper, env, stack));
            let mut values = try_recognize!(enumerate_integer_range(&lower, &upper));
            canonicalize_static_values(&mut values);
            RecognizeResult::Ok(StaticValue::Set(values))
        }
        Expr::Ident(name, _) => {
            let op = match resolve_zero_arity_operator(env, name, stack) {
                RecognizeResult::Ok(op) => op,
                RecognizeResult::Ineligible(reason) => {
                    return RecognizeResult::Ineligible(reason);
                }
                RecognizeResult::Unknown(reason) => return RecognizeResult::Unknown(reason),
            };
            stack.push(name.clone());
            let result = recognize_static_value(&op.body, env, stack);
            stack.pop();
            result
        }
        Expr::Prime(_) => RecognizeResult::Ineligible(
            "finite-cardinality literal value must be state-independent; primed expression found"
                .to_string(),
        ),
        Expr::StateVar(name, _, _) => RecognizeResult::Ineligible(format!(
            "finite-cardinality literal value must be state-independent; state variable `{name}` found"
        )),
        _ => RecognizeResult::Ineligible(
            "unsupported literal element in finite-cardinality set".to_string(),
        ),
    }
}

fn recognize_integer(
    expr: &Spanned<Expr>,
    env: &RecognitionEnv<'_>,
    stack: &mut Vec<String>,
) -> RecognizeResult<BigInt> {
    match &expr.node {
        Expr::Int(value) => RecognizeResult::Ok(value.clone()),
        Expr::Neg(inner) => {
            let value = try_recognize!(recognize_integer(inner, env, stack));
            RecognizeResult::Ok(-value)
        }
        Expr::Ident(name, _) => {
            let op = match resolve_zero_arity_operator(env, name, stack) {
                RecognizeResult::Ok(op) => op,
                RecognizeResult::Ineligible(reason) => {
                    return RecognizeResult::Ineligible(reason);
                }
                RecognizeResult::Unknown(reason) => return RecognizeResult::Unknown(reason),
            };
            stack.push(name.clone());
            let result = recognize_integer(&op.body, env, stack);
            stack.pop();
            result
        }
        Expr::Prime(_) => RecognizeResult::Ineligible(
            "finite-cardinality integer expression must be state-independent; primed expression found"
                .to_string(),
        ),
        Expr::StateVar(name, _, _) => RecognizeResult::Ineligible(format!(
            "finite-cardinality integer expression must be state-independent; state variable `{name}` found"
        )),
        _ => RecognizeResult::Ineligible(
            "expected integer literal or resolved integer constant".to_string(),
        ),
    }
}

fn resolve_zero_arity_operator<'a>(
    env: &'a RecognitionEnv<'a>,
    name: &str,
    stack: &[String],
) -> RecognizeResult<&'a OperatorDef> {
    if env.variables.contains(name) {
        return RecognizeResult::Ineligible(format!(
            "finite-cardinality facts must be state-independent; state variable `{name}` found"
        ));
    }

    if stack.iter().any(|entry| entry == name) {
        return RecognizeResult::Ineligible(format!(
            "recursive static constant `{name}` is not supported"
        ));
    }

    match env.operators.get(name) {
        Some(op) if op.params.is_empty() => RecognizeResult::Ok(op),
        Some(_) => RecognizeResult::Ineligible(format!(
            "operator `{name}` is not a resolved zero-arity constant"
        )),
        None if env.constants.contains(name) => RecognizeResult::Ineligible(format!(
            "constant `{name}` is not resolved to a static value"
        )),
        None => RecognizeResult::Ineligible(format!(
            "identifier `{name}` is not a resolved static constant"
        )),
    }
}

fn matches_unary_operator<'a>(expr: &'a Spanned<Expr>, name: &str) -> Option<&'a Spanned<Expr>> {
    match &expr.node {
        Expr::Apply(op, args) if args.len() == 1 && operator_name(op).as_deref() == Some(name) => {
            args.first()
        }
        Expr::ModuleRef(_, op_name, args) if args.len() == 1 && op_name == name => args.first(),
        _ => None,
    }
}

fn operator_name(expr: &Spanned<Expr>) -> Option<String> {
    match &expr.node {
        Expr::Ident(name, _) => Some(name.clone()),
        Expr::ModuleRef(_, name, args) if args.is_empty() => Some(name.clone()),
        _ => None,
    }
}

fn find_zero_arity_operator_in_module_set<'a>(
    modules: &[&'a Module],
    name: &str,
) -> Result<(usize, &'a OperatorDef), String> {
    let (module_idx, op) = modules
        .iter()
        .enumerate()
        .find_map(|(idx, module)| {
            module.units.iter().find_map(|unit| match &unit.node {
                Unit::Operator(op) if op.name.node == name => Some((idx, op)),
                _ => None,
            })
        })
        .ok_or_else(|| format!("operator `{name}` not found"))?;

    if op.params.is_empty() {
        Ok((module_idx, op))
    } else {
        Err(format!("operator `{name}` is not zero-arity"))
    }
}

fn canonicalize_static_values(values: &mut Vec<StaticValue>) {
    values.sort();
    values.dedup();
}

fn enumerate_integer_range(lower: &BigInt, upper: &BigInt) -> RecognizeResult<Vec<StaticValue>> {
    if upper < lower {
        return RecognizeResult::Ok(Vec::new());
    }
    let count = upper - lower + BigInt::one();
    let Some(count) = count.to_usize() else {
        return RecognizeResult::Unknown(
            "literal range is too large to canonicalize as a set element".to_string(),
        );
    };
    if count > MAX_STATIC_VALUE_SET_ELEMENTS {
        return RecognizeResult::Unknown(
            "literal range is too large to canonicalize as a set element".to_string(),
        );
    }

    let mut current = lower.clone();
    let mut values = Vec::with_capacity(count);
    while current <= *upper {
        values.push(StaticValue::Int(current.clone()));
        current += BigInt::one();
    }
    RecognizeResult::Ok(values)
}

fn pow_bigint(base: &BigInt, exponent: &BigInt) -> Option<BigInt> {
    let exponent = exponent.to_u32()?;
    Some(base.pow(exponent))
}

fn verify_set_certificate(
    certificate: &FiniteSetCertificate,
) -> Result<(), FiniteCardinalityVerificationError> {
    match certificate {
        FiniteSetCertificate::LiteralSet { elements } => {
            verify_static_values_are_canonical(elements)?;
            for element in elements {
                verify_static_value(element)?;
            }
        }
        FiniteSetCertificate::IntegerRange { .. } => {}
        FiniteSetCertificate::Powerset { base } => {
            verify_set_certificate(base)?;
        }
        FiniteSetCertificate::FunctionSet { domain, codomain } => {
            verify_set_certificate(domain)?;
            verify_set_certificate(codomain)?;
        }
        FiniteSetCertificate::ResolvedConstant { name, body } => {
            if name.is_empty() {
                return Err(FiniteCardinalityVerificationError::EmptyResolvedConstantName);
            }
            verify_set_certificate(body)?;
        }
    }

    certificate.try_cardinality().ok_or_else(|| {
        FiniteCardinalityVerificationError::CardinalityTooLarge {
            context: "finite set certificate".to_string(),
        }
    })?;
    Ok(())
}

fn verify_static_value(value: &StaticValue) -> Result<(), FiniteCardinalityVerificationError> {
    match value {
        StaticValue::Bool(_) | StaticValue::Int(_) | StaticValue::String(_) => Ok(()),
        StaticValue::Tuple(elements) => {
            for element in elements {
                verify_static_value(element)?;
            }
            Ok(())
        }
        StaticValue::Set(elements) => {
            verify_static_values_are_canonical(elements)?;
            for element in elements {
                verify_static_value(element)?;
            }
            Ok(())
        }
        StaticValue::Record(fields) => {
            if fields.windows(2).any(|pair| pair[0].0 >= pair[1].0) {
                return Err(FiniteCardinalityVerificationError::NonCanonicalStaticRecord);
            }
            for (_, value) in fields {
                verify_static_value(value)?;
            }
            Ok(())
        }
    }
}

fn verify_static_values_are_canonical(
    values: &[StaticValue],
) -> Result<(), FiniteCardinalityVerificationError> {
    if values.windows(2).any(|pair| pair[0] >= pair[1]) {
        Err(FiniteCardinalityVerificationError::NonCanonicalStaticSet)
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analytical::{AnalyticalAdmission, GateDecision, VerificationGate};
    use crate::shared_verdict::{SharedVerdict, Verdict};
    use crate::test_support::parse_module;

    fn outcome_for(source: &str, invariant: &str) -> FiniteCardinalityOutcome {
        let module = parse_module(source);
        recognize_module_finite_cardinality_invariant(&module, invariant)
    }

    fn candidate_for(
        source: &str,
        invariant: &str,
    ) -> CandidateProof<FiniteCardinalityCertificate> {
        match outcome_for(source, invariant) {
            AnalyticalOutcome::CandidateProof(candidate) => candidate,
            other => panic!("expected finite-cardinality candidate, got {other:?}"),
        }
    }

    fn admission_for(source: &str, invariant: &str) -> FiniteCardinalityAdmissionOutcome {
        let module = parse_module(source);
        admit_module_finite_cardinality_invariant(&module, invariant)
    }

    fn actual_cardinalities(certificate: &FiniteCardinalityCertificate) -> Vec<BigInt> {
        certificate
            .facts()
            .iter()
            .map(|fact| fact.actual_cardinality().expect("cardinality recomputes"))
            .collect()
    }

    #[test]
    fn recognizes_and_admits_static_literal_subset_and_function_facts() {
        let admission = admission_for(
            r#"
---- MODULE StaticCardinality ----
EXTENDS FiniteSets
S == {"a", "b"}
T == SUBSET S
F == [S -> {0, 1, 2}]
Inv == /\ IsFiniteSet({1, 2, 3})
       /\ Cardinality(S) = 2
       /\ Cardinality(T) = 4
       /\ Cardinality(F) = 9
====
"#,
            "Inv",
        );

        let AnalyticalAdmission::VerifiedProof(verified) = admission else {
            panic!("expected verified finite-cardinality proof");
        };
        let certificate = verified.certificate();
        assert_eq!(certificate.invariant_operator(), "Inv");
        assert_eq!(certificate.fact_count(), 4);
        assert_eq!(
            actual_cardinalities(certificate.proof()),
            vec![
                BigInt::from(3),
                BigInt::from(2),
                BigInt::from(4),
                BigInt::from(9)
            ]
        );
    }

    #[test]
    fn resolved_constants_are_followed_through_ranges_and_expected_values() {
        let candidate = candidate_for(
            r#"
---- MODULE ResolvedConstants ----
EXTENDS FiniteSets, Integers
Base == 1..3
Power == SUBSET Base
Expected == 8
Inv == ConstCardinality(Cardinality(Power) = Expected)
====
"#,
            "Inv",
        );

        let certificate = candidate.certificate();
        assert_eq!(certificate.fact_count(), 1);
        assert_eq!(
            certificate.facts()[0].actual_cardinality(),
            Some(BigInt::from(8))
        );
        verify_finite_cardinality_candidate(candidate).expect("candidate verifies");
    }

    #[test]
    fn admits_checker_module_invariant_with_checker_local_helpers() {
        let root = parse_module(
            r#"
---- MODULE RootCardinality ----
EXTENDS FiniteSets
S == {1}
====
"#,
        );
        let checker = parse_module(
            r#"
---- MODULE CheckerCardinality ----
EXTENDS FiniteSets
S == {1, 2, 3}
Inv == Cardinality(S) = 3
====
"#,
        );

        let admission = admit_module_set_finite_cardinality_invariant(&root, &[&checker], "Inv");

        let AnalyticalAdmission::VerifiedProof(verified) = admission else {
            panic!("expected checker-module finite-cardinality proof");
        };
        assert_eq!(
            actual_cardinalities(verified.certificate().proof()),
            vec![BigInt::from(3)]
        );
    }

    #[test]
    fn candidate_proof_requires_verification_gate_before_publish() {
        let candidate = candidate_for(
            r#"
---- MODULE ProofGate ----
EXTENDS FiniteSets
Inv == Cardinality(SUBSET {1, 2}) = 4
====
"#,
            "Inv",
        );

        let shared = SharedVerdict::new();
        let gate = VerificationGate::new(&shared);
        let raw: FiniteCardinalityOutcome = AnalyticalOutcome::CandidateProof(candidate.clone());
        assert_eq!(gate.inspect(&raw), GateDecision::NeedsProof);
        assert!(!shared.is_resolved());

        let verified =
            verify_finite_cardinality_candidate(candidate).expect("static proof verifies");
        assert_eq!(
            gate.publish_verified_proof(verified),
            GateDecision::Published
        );
        assert_eq!(shared.get(), Some(Verdict::Satisfied));
    }

    #[test]
    fn rejects_state_variables_fail_closed() {
        let outcome = outcome_for(
            r#"
---- MODULE StateDependent ----
VARIABLE x
Inv == IsFiniteSet(x)
====
"#,
            "Inv",
        );

        match outcome {
            AnalyticalOutcome::Ineligible(reason) => {
                assert!(reason.reason().contains("state variable"));
            }
            other => panic!("expected Ineligible, got {other:?}"),
        }
    }

    #[test]
    fn rejects_primed_expressions_fail_closed() {
        let outcome = outcome_for(
            r#"
---- MODULE PrimedFact ----
VARIABLE x
Inv == IsFiniteSet(x')
====
"#,
            "Inv",
        );

        match outcome {
            AnalyticalOutcome::Ineligible(reason) => {
                assert!(reason.reason().contains("primed"));
            }
            other => panic!("expected Ineligible, got {other:?}"),
        }
    }

    #[test]
    fn rejects_false_cardinality_comparison_fail_closed() {
        let outcome = outcome_for(
            r#"
---- MODULE FalseCardinality ----
EXTENDS FiniteSets
Inv == Cardinality({1, 2}) = 3
====
"#,
            "Inv",
        );

        match outcome {
            AnalyticalOutcome::Ineligible(reason) => {
                assert!(reason.reason().contains("false"));
            }
            other => panic!("expected Ineligible, got {other:?}"),
        }
    }

    #[test]
    fn rejects_false_literal_fail_closed() {
        let outcome = outcome_for(
            r#"
---- MODULE FalseLiteral ----
Inv == FALSE
====
"#,
            "Inv",
        );

        match outcome {
            AnalyticalOutcome::Ineligible(reason) => {
                assert!(reason.reason().contains("FALSE"));
            }
            other => panic!("expected Ineligible, got {other:?}"),
        }
    }

    #[test]
    fn rejects_unresolved_constants_fail_closed() {
        let outcome = outcome_for(
            r#"
---- MODULE UnresolvedConstant ----
CONSTANT S
Inv == IsFiniteSet(S)
====
"#,
            "Inv",
        );

        match outcome {
            AnalyticalOutcome::Ineligible(reason) => {
                assert!(reason.reason().contains("not resolved"));
            }
            other => panic!("expected Ineligible, got {other:?}"),
        }
    }

    #[test]
    fn verifier_rejects_noncanonical_static_certificate_values() {
        let duplicate_literal = FiniteCardinalityCertificate {
            facts: vec![FiniteCardinalityFact::IsFiniteSet {
                set: FiniteSetCertificate::LiteralSet {
                    elements: vec![
                        StaticValue::Int(BigInt::from(1)),
                        StaticValue::Int(1.into()),
                    ],
                },
            }],
        };
        assert_eq!(
            verify_certificate(&duplicate_literal),
            Err(FiniteCardinalityVerificationError::NonCanonicalStaticSet)
        );

        let unsorted_record = FiniteCardinalityCertificate {
            facts: vec![FiniteCardinalityFact::IsFiniteSet {
                set: FiniteSetCertificate::LiteralSet {
                    elements: vec![StaticValue::Record(vec![
                        ("z".to_string(), StaticValue::Bool(true)),
                        ("a".to_string(), StaticValue::Bool(false)),
                    ])],
                },
            }],
        };
        assert_eq!(
            verify_certificate(&unsorted_record),
            Err(FiniteCardinalityVerificationError::NonCanonicalStaticRecord)
        );
    }
}
