// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0
#![forbid(unsafe_code)]
#![deny(missing_docs)]

//! TLA+ proof certificate checker.
//!
//! This crate is the *trusted core* of the toolchain's deductive layer. When an
//! automated backend (a tableau prover, an SMT solver, an interactive prover)
//! claims to have proven a TLA+ proof obligation, it can emit a
//! [`Certificate`]: a self-contained, machine-checkable record of the entire
//! derivation. [`CertificateChecker`] then *re-derives* that record from
//! scratch and returns a [`VerificationResult`]. Because the checker re-checks
//! every inference itself, a valid certificate is trustworthy even if the
//! producing backend is buggy or untrusted — only this small checker, not the
//! prover, is part of the trusted computing base.
//!
//! # Design goals
//!
//! 1. **Minimal** — a small, auditable codebase with no `unsafe`.
//! 2. **Trusted** — a small trusted computing base: trust the checker, not the
//!    prover.
//! 3. **Independent** — proofs are verified without trusting the producer.
//!
//! # Certificate format
//!
//! A [`Certificate`] proves a [`goal`](Certificate::goal) from a list of
//! [`hypotheses`](Certificate::hypotheses) via an ordered list of
//! [`steps`](Certificate::steps). Each [`CertificateStep`] pairs a [`Formula`]
//! with a [`Justification`] — the inference rule that derives it — drawn from:
//!
//! - **Axioms** ([`Justification::Axiom`]): built-in, structurally checked
//!   logical/arithmetic/set-theoretic [`Axiom`] schemas.
//! - **Earlier steps**: rules such as [`Justification::ModusPonens`],
//!   conjunction/disjunction intro/elim, [`Justification::UniversalInstantiation`],
//!   [`Justification::ExistentialIntro`], [`Justification::Rewrite`], and
//!   [`Justification::TableauDecomposition`].
//! - **Definitions** ([`Justification::Definition`]): expansions registered with
//!   [`CertificateChecker::add_definition`].
//!
//! The certificate is valid iff every step checks and the final step is
//! [`alpha_equiv`]alent to the goal.
//!
//! # Example
//!
//! ```
//! use tla_cert::{
//!     Axiom, ArithmeticAxiom, Backend, Certificate, CertificateChecker,
//!     CertificateStep, Formula, Justification, Term,
//! };
//!
//! let goal = Formula::Eq(
//!     Term::App("+".into(), vec![Term::Int(0), Term::Var("a".into())]),
//!     Term::Var("a".into()),
//! );
//! let cert = Certificate {
//!     id: "demo".into(),
//!     goal: goal.clone(),
//!     hypotheses: vec![],
//!     steps: vec![CertificateStep {
//!         id: 0,
//!         formula: goal,
//!         justification: Justification::Axiom(Axiom::Arithmetic(ArithmeticAxiom::AddZero)),
//!     }],
//!     backend: Backend::Z3,
//! };
//! assert!(CertificateChecker::new().verify(&cert).is_valid());
//! ```
//!
//! # Internal structure
//!
//! Internally the crate splits into the data model (certificates, formulas,
//! terms, results — all re-exported at the crate root), the verification engine
//! ([`CertificateChecker`]), structural axiom-instance checking, and the formula
//! operations ([`alpha_equiv`], substitution, tableau checks) the engine relies
//! on.

mod axiom_check;
mod checker;
mod formula_ops;
mod types;

pub use checker::CertificateChecker;
pub use formula_ops::alpha_equiv;
pub use types::{
    ArithmeticAxiom, Axiom, Backend, CaseBranch, Certificate, CertificateIoError, CertificateStep,
    Formula, Justification, SetAxiom, StepId, Term, VerificationError, VerificationResult,
};

#[cfg(test)]
mod tests;
