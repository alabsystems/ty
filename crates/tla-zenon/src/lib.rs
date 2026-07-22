// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0
#![forbid(unsafe_code)]
#![deny(missing_docs)]

//! `tla-zenon` — first-order tableau prover for the `ty` toolchain.
//!
//! A Rust port of Zenon, the automated theorem prover used by TLAPM (the TLA+
//! Proof Manager). It implements the analytic tableau method for classical
//! first-order logic with equality, and is the leaf prover `ty` calls to
//! discharge first-order proof obligations.
//!
//! # The tableau method
//!
//! Proving proceeds by *refutation*:
//! 1. Negate the goal formula.
//! 2. Apply decomposition rules to expand the negated formula into a tree of
//!    branches, each branch being a set of formulas that must hold together.
//! 3. If every branch contains a contradiction (some `P` together with `¬P`,
//!    or `⊥`, or `¬(t = t)`), the negation is unsatisfiable, so the original
//!    goal is valid.
//!
//! # Tableau rules
//!
//! Formulas are classified (see [`rules::classify`]) by which rule applies, and
//! the prover prefers non-branching rules to keep the search tree small:
//!
//! - **Alpha rules**: conjunctive decomposition with a single successor
//!   (e.g. `A ∧ B ⊢ A, B`; `¬(A ∨ B) ⊢ ¬A, ¬B`).
//! - **Beta rules**: disjunctive decomposition that splits the branch in two
//!   (e.g. `A ∨ B ⊢ A | B`; `A → B ⊢ ¬A | B`).
//! - **Gamma rules**: universal instantiation, `∀x.P ⊢ P[t/x]`, applied for
//!   each available term up to a configurable instance limit.
//! - **Delta rules**: existential witnessing, `∃x.P ⊢ P[c/x]` for a fresh
//!   Skolem constant `c`.
//!
//! # Verifiable certificates
//!
//! A successful proof is a tree ([`Proof`]) that can be lowered to a linear,
//! independently checkable [`tla_cert::Certificate`] via [`Proof::to_certificate`]
//! (or [`proof_to_certificate`]). This lets a downstream checker re-verify the
//! result without trusting the prover's search.
//!
//! # Crate layout
//!
//! - [`formula`]: the [`Formula`] / [`Term`] / [`Subst`] FOL representation.
//! - [`rules`]: the alpha/beta/gamma/delta decomposition rules.
//! - [`prover`]: the proof-search engine ([`Prover`], [`ProverConfig`],
//!   [`ProofResult`]).
//! - [`proof`]: the proof-tree data types ([`Proof`], [`ProofNode`],
//!   [`ProofRule`]).
//! - [`certificate`]: lowering proofs to verifiable certificates.
//!
//! # Example
//!
//! ```rust
//! use tla_zenon::{Formula, Prover, ProofResult};
//!
//! // Prove: (A ∧ B) → A
//! let a = Formula::var("A");
//! let b = Formula::var("B");
//! let goal = Formula::implies(Formula::and(a.clone(), b), a);
//!
//! let mut prover = Prover::new();
//! let result = prover.prove(&goal, Default::default());
//! assert!(matches!(result, ProofResult::Valid(_)));
//! ```

pub mod certificate;
pub mod formula;
pub mod proof;
pub mod prover;
pub mod rules;

pub use certificate::{convert_formula, convert_term, proof_to_certificate};
pub use formula::{Formula, Subst, Term};
pub use proof::{Proof, ProofNode, ProofRule};
pub use prover::{ProofResult, Prover, ProverConfig};
