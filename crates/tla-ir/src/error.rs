// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Error types for trust-ir lowering.

/// Errors that can occur during bytecode-to-trust-ir lowering.
#[derive(Debug, thiserror::Error)]
pub enum TrustIrError {
    /// Error during trust-ir construction.
    #[error("trust-ir emission failed: {0}")]
    Emission(String),

    /// Opcode is not supported by the trust-ir backend.
    #[error("unsupported opcode for trust-ir backend: {0}")]
    UnsupportedOpcode(String),

    /// Bytecode function is not eligible for trust-ir lowering.
    #[error("bytecode function is not eligible for trust-ir lowering: {reason}")]
    NotEligible {
        /// Human-readable explanation of why the function was rejected.
        reason: String,
    },
}
