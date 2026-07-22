// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Error types for the trust-codegen compilation backend.

/// Errors that can occur during trust-ir-to-native compilation via `trust_cg`.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum TrustCgError {
    /// The input trust-ir module is malformed or contains unsupported constructs.
    #[error("invalid trust-ir module: {0}")]
    InvalidModule(String),

    /// A trust-ir instruction is not yet supported by the trust-codegen lowering.
    #[error("unsupported trust-ir instruction: {0}")]
    UnsupportedInst(String),

    /// Error during trust-codegen IR construction.
    #[error("trust-cg IR emission failed: {0}")]
    Emission(String),

    /// Error during trust-codegen optimization passes.
    #[error("trust-cg optimization failed: {0}")]
    Optimization(String),

    /// Error during native code generation (register allocation, `ISel`, encoding).
    #[error("trust-cg code generation failed: {0}")]
    CodeGen(String),

    /// Error loading or linking the compiled native code.
    #[error("native code loading failed: {0}")]
    Loading(String),

    /// The trust-codegen backend is not available (feature not enabled or library missing).
    #[error("trust-cg backend not available: {0}")]
    BackendUnavailable(String),

    /// Error from the upstream trust-ir lowering phase.
    #[error("trust-ir lowering error: {source}")]
    TrustIrLowering {
        /// The underlying `tla-ir` lowering failure that produced this error.
        #[from]
        source: tla_ir::TrustIrError,
    },
}
