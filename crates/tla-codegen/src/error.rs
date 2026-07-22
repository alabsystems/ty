// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Typed error types for the TLA+ code generator.

use crate::types::TypeInferError;

/// Errors produced during TLA+ to Rust code generation.
#[derive(Debug, thiserror::Error)]
pub enum CodegenError {
    /// Type inference failed with one or more errors.
    #[error("type inference errors:\n{}", .0.iter().map(|e| e.to_string()).collect::<Vec<_>>().join("\n"))]
    TypeInference(Vec<TypeInferError>),

    /// Invalid combination of code generation options.
    #[error("checker_map requires generate_checker")]
    CheckerMapRequiresChecker,

    /// Checker map module name does not match the module being generated.
    #[error("checker map module mismatch: config spec.module={config_module:?}, generating module {actual_module:?}")]
    CheckerMapModuleMismatch {
        /// Module name the checker-map config declared it targets.
        config_module: String,
        /// Module name actually being code-generated.
        actual_module: String,
    },

    /// Checker map has no `[[impls]]` entries.
    #[error("checker map has no [[impls]] entries")]
    CheckerMapNoImpls,

    /// Checker map has a duplicate field mapping.
    #[error("checker map impls[{index}] duplicate mapping for state field {field:?}: {prev:?} vs {current:?}")]
    CheckerMapDuplicateField {
        /// Index of the offending `[[impls]]` entry in the config.
        index: usize,
        /// State field name that received two mappings.
        field: String,
        /// The first (previously seen) mapped expression.
        prev: String,
        /// The conflicting later mapped expression.
        current: String,
    },

    /// Checker map references unknown state field(s).
    #[error("checker map impls[{index}] has unknown state field(s): {unknown} (expected keys: {expected})")]
    CheckerMapUnknownFields {
        /// Index of the offending `[[impls]]` entry in the config.
        index: usize,
        /// Comma-separated unknown field name(s) the config referenced.
        unknown: String,
        /// Comma-separated set of valid state field names.
        expected: String,
    },

    /// Checker map is missing a required field mapping.
    #[error("checker map impls[{index}] missing mapping for state field {field:?}")]
    CheckerMapMissingField {
        /// Index of the offending `[[impls]]` entry in the config.
        index: usize,
        /// State field name that the config left unmapped.
        field: String,
    },

    /// Input validation failure for user-supplied Rust fragments (injection prevention).
    #[error("{context}: {reason}")]
    InvalidRustFragment {
        /// Where the fragment came from (e.g. the field being mapped).
        context: String,
        /// Why the fragment was rejected (e.g. not a single-line expression).
        reason: String,
    },
}
