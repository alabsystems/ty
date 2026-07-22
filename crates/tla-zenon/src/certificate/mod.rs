// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Certificate generation from Zenon proofs.
//!
//! This module provides conversion from tableau proofs (tree-structured) to
//! proof certificates (linear step sequences) that can be independently verified.
//!
//! # Module structure
//!
//! Internally this work is split across three private submodules:
//!
//! - `convert`: type conversion between Zenon and certificate representations
//!   (re-exported as [`convert_formula`] / [`convert_term`]).
//! - `builder`: a certificate step accumulator that deduplicates derived
//!   formulas and hands out [`tla_cert::StepId`]s.
//! - `generate`: proof-tree traversal that emits certificate steps
//!   (re-exported as [`proof_to_certificate`]).

mod builder;
mod convert;
mod generate;

pub use convert::{convert_formula, convert_term};
pub use generate::proof_to_certificate;

#[cfg(test)]
mod tests;
