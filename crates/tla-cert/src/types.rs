// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Type definitions for the TLA+ proof certificate system.

use serde::{Deserialize, Serialize};

/// A proof certificate that can be independently verified.
///
/// A certificate records a complete derivation of [`goal`](Self::goal) from a
/// list of [`hypotheses`](Self::hypotheses): an ordered list of
/// [`steps`](Self::steps), each carrying a [`Formula`] and the
/// [`Justification`] that derives it. The certificate is *valid* when every
/// step's justification checks against earlier steps (and the hypotheses) and
/// the final step is alpha-equivalent to the goal.
///
/// Certificates are produced by automated backends (see [`Backend`]) and are
/// re-checked from scratch by [`CertificateChecker`](crate::CertificateChecker),
/// so a valid certificate is trustworthy without trusting the producer. The
/// type is `serde`-serializable so certificates can be persisted and exchanged
/// (see [`Certificate::to_json`] / [`Certificate::save_to_file`]).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Certificate {
    /// Unique identifier for this proof
    pub id: String,
    /// The goal that was proven
    pub goal: Formula,
    /// Hypotheses assumed in the proof
    pub hypotheses: Vec<Formula>,
    /// The proof steps
    pub steps: Vec<CertificateStep>,
    /// Backend that generated this certificate
    pub backend: Backend,
}

/// A single step in a proof certificate: a [`Formula`] together with the
/// [`Justification`] that establishes it.
///
/// During verification a step's [`id`](Self::id) becomes a key in the checker's
/// fact table, so later steps can cite it. Ids should be distinct and should not
/// collide with the hypothesis range `0..n` unless deliberately re-stating a
/// hypothesis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CertificateStep {
    /// Identifier other steps use to reference this step's formula.
    pub id: StepId,
    /// The formula this step claims to establish.
    pub formula: Formula,
    /// The inference rule (and operands) that derives [`formula`](Self::formula).
    pub justification: Justification,
}

/// Identifier for a proof step (and, for the first `n` ids, for a hypothesis).
pub type StepId = u32;

/// Justification for a proof step: the inference rule (and its operands) that
/// derives the step's formula.
///
/// Most variants reference earlier steps by [`StepId`]; the checker looks those
/// up and confirms the rule applies. Hypotheses occupy step ids `0..n` (their
/// position in [`Certificate::hypotheses`]), so [`Justification::Hypothesis`]
/// indexes into that prefix. Each variant corresponds to one verifier in the
/// checker; if the referenced step is missing or the rule does not apply,
/// verification fails with a [`VerificationError`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Justification {
    /// An axiom (built-in logical truth); the contained [`Axiom`] names the
    /// schema the step's formula must instantiate.
    Axiom(Axiom),
    /// A hypothesis assumed by the certificate. The `usize` indexes into
    /// [`Certificate::hypotheses`]; the step's formula must equal that
    /// hypothesis exactly.
    Hypothesis(usize),
    /// Modus ponens: from P and P => Q, derive Q
    ModusPonens {
        /// Step that established the antecedent `P`.
        premise: StepId,
        /// Step that established the implication `P => Q`.
        implication: StepId,
    },
    /// Universal instantiation: from ∀x. P(x), derive P(t)
    UniversalInstantiation {
        /// Step that established the universally quantified formula `∀x. P(x)`.
        forall: StepId,
        /// The term `t` substituted for the bound variable `x`.
        term: Term,
    },
    /// Existential introduction: from P(t), derive ∃x. P(x)
    ExistentialIntro {
        /// Step that established the witnessing instance `P(t)`.
        witness: StepId,
        /// Name of the bound variable `x` introduced by the existential.
        variable: String,
    },
    /// Definition expansion
    Definition {
        /// Name of the definition to expand; must be registered via
        /// [`CertificateChecker::add_definition`](crate::CertificateChecker::add_definition).
        name: String,
    },
    /// Conjunction introduction: from P and Q, derive P ∧ Q
    AndIntro {
        /// Step that established the left conjunct `P`.
        left: StepId,
        /// Step that established the right conjunct `Q`.
        right: StepId,
    },
    /// Conjunction elimination (left): from P ∧ Q, derive P
    AndElimLeft {
        /// Step that established the conjunction `P ∧ Q`.
        conjunction: StepId,
    },
    /// Conjunction elimination (right): from P ∧ Q, derive Q
    AndElimRight {
        /// Step that established the conjunction `P ∧ Q`.
        conjunction: StepId,
    },
    /// Disjunction introduction (left): from P, derive P ∨ Q
    OrIntroLeft {
        /// Step that established the proven disjunct `P`.
        premise: StepId,
        /// The other disjunct `Q`, supplied directly (it need not be proven).
        right: Formula,
    },
    /// Disjunction introduction (right): from Q, derive P ∨ Q
    OrIntroRight {
        /// The other disjunct `P`, supplied directly (it need not be proven).
        left: Formula,
        /// Step that established the proven disjunct `Q`.
        premise: StepId,
    },
    /// Double negation elimination: from ¬¬P, derive P
    DoubleNegElim {
        /// Step that established the doubly negated formula `¬¬P`.
        premise: StepId,
    },
    /// Rewrite using equality: from a = b and P(a), derive P(b)
    Rewrite {
        /// Step that established an equality `a = b` used to rewrite.
        equality: StepId,
        /// Step whose formula `P(a)` is rewritten (in either direction).
        target: StepId,
    },
    /// Tableau decomposition: a formula derived from a premise by a *sound
    /// single-premise* tableau rule (an α non-branching rule, γ universal
    /// instantiation, or equality symmetry — see
    /// [`is_valid_tableau_decomposition`](crate::alpha_equiv) siblings). The
    /// checker re-derives the consequence itself and rejects the step otherwise;
    /// branching (β) and existential-witness (δ) rules are NOT accepted here and
    /// must use [`Justification::CaseSplit`].
    TableauDecomposition {
        /// Step that established the formula being decomposed.
        premise: StepId,
    },
    /// Tableau case split ("proof by cases"): from a `premise` whose β-rule
    /// decomposition is a disjunction of the branch assumptions
    /// (`A ∨ B`, `A → B`, `¬(A ∧ B)`, `¬(A ↔ B)`), establish the *same*
    /// conclusion ([`CertificateStep::formula`]) in **every** branch.
    ///
    /// Each [`CaseBranch`] additionally assumes its own disjunct(s) and is
    /// verified against the outer facts **plus** those assumptions only — branch
    /// facts are scoped, so a contradiction reached in one branch cannot license
    /// a conclusion in another (or in the outer proof). This is the sound
    /// replacement for the unsound single-premise β "decomposition": deriving one
    /// disjunct of a disjunction requires discharging *all* cases.
    CaseSplit {
        /// Step establishing the formula being split; its β-decomposition
        /// determines the assumptions each branch must make.
        premise: StepId,
        /// One scoped sub-derivation per branch. The conclusion holds only if
        /// every branch derives it under its assumption.
        branches: Vec<CaseBranch>,
    },
}

/// One branch of a [`Justification::CaseSplit`]: a scoped sub-derivation that
/// assumes the branch's disjunct(s) and must establish the case-split's
/// conclusion.
///
/// The branch is checked against the outer proof's facts extended with
/// [`assumptions`](Self::assumptions); its [`steps`](Self::steps) may reference
/// outer facts and the assumptions by [`StepId`], but the facts it derives are
/// discarded when the branch closes (they never leak to sibling branches or the
/// outer proof).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CaseBranch {
    /// The disjunct(s) assumed at the start of this branch, each paired with the
    /// [`StepId`] the branch's steps use to reference it. The checker requires
    /// these to match the premise's β-decomposition for this branch (e.g. `¬A`
    /// for the left branch of `A → B`, or `A` and `¬B` for the first branch of
    /// `¬(A ↔ B)`).
    pub assumptions: Vec<(StepId, Formula)>,
    /// The scoped derivation establishing the case-split conclusion under
    /// [`assumptions`](Self::assumptions).
    pub steps: Vec<CertificateStep>,
}

/// A built-in axiom schema that an [`Justification::Axiom`] step may
/// instantiate.
///
/// Each variant denotes a logically valid schema; the checker confirms that the
/// step's formula is a concrete instance of it (it does not assume validity on
/// faith). Variants that need a specific propositional parameter carry it (e.g.
/// [`Axiom::ExcludedMiddle`] carries the `P` in `P ∨ ¬P`); schemas matched
/// purely by shape (e.g. [`Axiom::Weakening`]) carry nothing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Axiom {
    /// P ∨ ¬P (excluded middle)
    ExcludedMiddle(Formula),
    /// P → P (identity)
    Identity(Formula),
    /// P → (Q → P) (weakening)
    Weakening,
    /// a = a (reflexivity)
    EqualityRefl,
    /// a = b → b = a (symmetry)
    EqualitySym,
    /// a = b ∧ b = c → a = c (transitivity)
    EqualityTrans,
    /// Basic arithmetic axiom
    Arithmetic(ArithmeticAxiom),
    /// Set theory axiom
    SetTheory(SetAxiom),
}

/// Arithmetic axiom schemas over the `"+"` and `"*"` operators (see
/// [`Axiom::Arithmetic`]). Each is matched structurally against a step's
/// equality formula.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ArithmeticAxiom {
    /// 0 + a = a
    AddZero,
    /// a + b = b + a
    AddComm,
    /// (a + b) + c = a + (b + c)
    AddAssoc,
    /// a * 1 = a
    MulOne,
    /// a * 0 = 0
    MulZero,
}

/// Set-theory axiom schemas characterizing membership (`∈`) for the empty set,
/// singletons, union (`∪`) and intersection (`∩`) (see [`Axiom::SetTheory`]).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SetAxiom {
    /// x ∈ {} ↔ FALSE
    EmptySet,
    /// x ∈ {a} ↔ x = a
    Singleton,
    /// x ∈ S ∪ T ↔ x ∈ S ∨ x ∈ T
    Union,
    /// x ∈ S ∩ T ↔ x ∈ S ∧ x ∈ T
    Intersection,
}

/// A first-order logic formula: the proposition language of certificates.
///
/// Formulas are built from booleans, predicate applications, the propositional
/// connectives, the two quantifiers, and term equality. Connective operands are
/// boxed so the type can nest. Equality is treated as a top-level formula
/// constructor ([`Formula::Eq`]) rather than a binary predicate so the checker's
/// equality axioms and rewrite rule can match it structurally.
///
/// [`PartialEq`] is *syntactic* (capture-sensitive) equality; use
/// [`alpha_equiv`](crate::alpha_equiv) when bound-variable names should be
/// ignored.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Formula {
    /// Propositional constant
    Bool(bool),
    /// Predicate application
    Predicate(String, Vec<Term>),
    /// Negation
    Not(Box<Formula>),
    /// Conjunction
    And(Box<Formula>, Box<Formula>),
    /// Disjunction
    Or(Box<Formula>, Box<Formula>),
    /// Implication
    Implies(Box<Formula>, Box<Formula>),
    /// Equivalence
    Equiv(Box<Formula>, Box<Formula>),
    /// Universal quantification
    Forall(String, Box<Formula>),
    /// Existential quantification
    Exists(String, Box<Formula>),
    /// Equality
    Eq(Term, Term),
}

/// A first-order term: the object-level expression language that appears inside
/// [`Formula`] predicates and equalities.
///
/// Variables ([`Term::Var`]) are the names bound by quantifiers and substituted
/// during instantiation/rewriting; constants and integer literals are atomic;
/// [`Term::App`] is an uninterpreted function application whose head string also
/// encodes the built-in operators the axioms recognize (e.g. `"+"`, `"*"`,
/// `"∪"`, `"singleton"`).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Term {
    /// A variable, by name; bound by a quantifier or otherwise free.
    Var(String),
    /// A named, uninterpreted constant.
    Const(String),
    /// An integer literal.
    Int(i64),
    /// Application of a function/operator (the head string) to arguments.
    App(String, Vec<Term>),
}

/// Backend that generated the certificate.
///
/// This is provenance metadata only: the checker re-verifies every step
/// independently and never trusts the originating backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Backend {
    /// The Zenon tableau prover.
    Zenon,
    /// The Z3 SMT solver.
    Z3,
    /// The cvc5 SMT solver.
    CVC5,
    /// The Lean 4 theorem prover.
    Lean4,
}

/// Outcome of checking a [`Certificate`].
///
/// This is a verdict, not a `Result`: a well-formed but *invalid* certificate
/// is a normal outcome ([`Invalid`](Self::Invalid)) rather than an error. Use
/// [`VerificationResult::is_valid`] to test the verdict and
/// [`VerificationResult::error`] to inspect the failure.
#[derive(Debug, Serialize, Deserialize)]
pub enum VerificationResult {
    /// Every step checked and the final step matches the goal.
    Valid,
    /// A step failed to check, or the final step did not match the goal; the
    /// contained [`VerificationError`] explains the first failure encountered.
    Invalid(VerificationError),
}

/// The reason a [`Certificate`] failed verification.
#[derive(Debug, Serialize, Deserialize)]
pub enum VerificationError {
    /// A justification referenced a step id that has not been established.
    UnknownStep(StepId),
    /// Justification doesn't match formula
    InvalidJustification {
        /// The step whose justification failed to verify.
        step: StepId,
        /// Human-readable explanation of why the justification was rejected.
        reason: String,
    },
    /// The certificate's final step is not alpha-equivalent to the goal (or the
    /// certificate has no steps).
    GoalMismatch,
    /// An [`Justification::Axiom`] step's formula does not instantiate the
    /// claimed axiom schema; the string describes the specific mismatch.
    InvalidAxiom(String),
}

// ============================================================================
// Serialization
// ============================================================================

/// Error returned by [`Certificate`] file I/O ([`Certificate::save_to_file`],
/// [`Certificate::load_from_file`]), wrapping either a serialization or a
/// filesystem failure.
#[derive(Debug)]
pub enum CertificateIoError {
    /// JSON serialization/deserialization error
    Json(serde_json::Error),
    /// File I/O error
    Io(std::io::Error),
}

impl std::fmt::Display for CertificateIoError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Json(e) => write!(f, "JSON error: {}", e),
            Self::Io(e) => write!(f, "I/O error: {}", e),
        }
    }
}

impl std::error::Error for CertificateIoError {}

impl From<serde_json::Error> for CertificateIoError {
    fn from(e: serde_json::Error) -> Self {
        Self::Json(e)
    }
}

impl From<std::io::Error> for CertificateIoError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

impl Certificate {
    /// Serialize the certificate to a compact JSON string.
    ///
    /// # Errors
    ///
    /// Returns a [`serde_json::Error`] if the certificate cannot be serialized
    /// (in practice the certificate model is always serializable, so this is
    /// only reached on a downstream I/O failure of the serializer).
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string(self)
    }

    /// Serialize the certificate to a human-readable, indented JSON string.
    ///
    /// # Errors
    ///
    /// Returns a [`serde_json::Error`] if serialization fails; see
    /// [`Certificate::to_json`] for when that can happen.
    pub fn to_json_pretty(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// Deserialize a certificate from a JSON string.
    ///
    /// # Errors
    ///
    /// Returns a [`serde_json::Error`] if `json` is not valid JSON or does not
    /// match the [`Certificate`] schema (e.g. missing or mistyped fields).
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }

    /// Save the certificate to `path` as pretty-printed JSON, overwriting any
    /// existing file.
    ///
    /// # Errors
    ///
    /// Returns [`CertificateIoError::Json`] if serialization fails, or
    /// [`CertificateIoError::Io`] if the file cannot be written (e.g. the parent
    /// directory does not exist or permissions are insufficient).
    pub fn save_to_file(
        &self,
        path: impl AsRef<std::path::Path>,
    ) -> Result<(), CertificateIoError> {
        let json = self.to_json_pretty()?;
        std::fs::write(path, json)?;
        Ok(())
    }

    /// Load a certificate from a JSON file at `path`.
    ///
    /// # Errors
    ///
    /// Returns [`CertificateIoError::Io`] if the file cannot be read (e.g. it
    /// does not exist), or [`CertificateIoError::Json`] if its contents are not
    /// a valid [`Certificate`].
    pub fn load_from_file(path: impl AsRef<std::path::Path>) -> Result<Self, CertificateIoError> {
        let json = std::fs::read_to_string(path)?;
        let cert = Self::from_json(&json)?;
        Ok(cert)
    }
}

impl VerificationResult {
    /// Returns `true` if the certificate verified (i.e. the result is
    /// [`VerificationResult::Valid`]).
    pub fn is_valid(&self) -> bool {
        matches!(self, Self::Valid)
    }

    /// Returns the failure reason, or `None` if the certificate was valid.
    pub fn error(&self) -> Option<&VerificationError> {
        match self {
            Self::Valid => None,
            Self::Invalid(e) => Some(e),
        }
    }
}
