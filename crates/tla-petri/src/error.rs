// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Error types for PNML parsing and Petri net exploration.

use std::path::PathBuf;

/// Failure modes of PNML parsing, colored-net unfolding, and Petri net
/// exploration.
///
/// This is the error half of every fallible public entry point in the crate
/// (for example [`crate::parser::parse_pnml_dir`] and
/// [`crate::model::load_model_dir`]). It is `#[non_exhaustive]`: callers must
/// include a wildcard match arm, since new variants may be added without a
/// major-version bump.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum PnmlError {
    /// An I/O error occurred while reading the model directory or a PNML file.
    #[error("I/O error reading {path}: {source}")]
    Io {
        /// Filesystem path whose read failed.
        path: PathBuf,
        /// Underlying I/O error.
        source: std::io::Error,
    },

    /// The PNML XML was malformed or could not be parsed; the payload is a
    /// human-readable description of the parse failure.
    #[error("XML parse error: {0}")]
    XmlParse(String),

    /// The PNML input uses a net kind or construct that `tla-petri` does not support.
    ///
    /// [`crate::parser::parse_pnml_dir`] emits this for non-`ptnet` inputs.
    /// [`crate::model::load_model_dir`] may also emit it for colored-model
    /// encodings outside the supported unfolding subset or for unfold-size
    /// guardrails.
    #[error("unsupported PNML net or construct: {net_type}")]
    UnsupportedNetType {
        /// The net `type` URI or construct name that is not supported.
        net_type: String,
    },

    /// Colored unfolding could not complete within the configured size or
    /// time budget. Unlike [`PnmlError::UnsupportedNetType`], this is a
    /// *recoverable* outcome: the colored source is well-formed and supported,
    /// but materializing its P/T expansion would exceed the place/transition
    /// caps or the load-time deadline. Callers that hold the colored source
    /// (e.g. the colored OneSafe structural shortcut) may still answer some
    /// examinations; the rest fall back to CANNOT_COMPUTE rather than the
    /// whole-model collapse that `UnsupportedNetType` triggers.
    #[error("colored unfolding unavailable (over budget): {reason}")]
    ColoredUnfoldUnavailable {
        /// Why the unfolding was declined (which budget was exceeded).
        reason: String,
    },

    /// A required PNML element (e.g. `<net>`, a place's `<initialMarking>`)
    /// was absent; the payload names the missing element.
    #[error("missing required element: {0}")]
    MissingElement(String),

    /// An arc references an endpoint that does not exist, or connects two
    /// nodes of the same kind (place-to-place or transition-to-transition).
    #[error("invalid arc: source={src_id}, target={tgt_id}: {reason}")]
    InvalidArc {
        /// `id` of the arc's source node.
        src_id: String,
        /// `id` of the arc's target node.
        tgt_id: String,
        /// Why the arc is rejected.
        reason: String,
    },

    /// A place's initial-marking text could not be parsed as a non-negative
    /// integer token count; the payload is the offending text.
    #[error("invalid marking value: {0}")]
    InvalidMarking(String),

    /// A `<toolspecific>` NUPN (Nested-Unit Petri Net) annotation was
    /// structurally invalid (e.g. malformed unit list or place reference).
    #[error("invalid NUPN annotation: {reason}")]
    InvalidNupn {
        /// What made the NUPN annotation invalid.
        reason: String,
    },

    /// A structural reduction's bookkeeping arithmetic overflowed (e.g. an
    /// agglomerated arc weight exceeding the integer range); the payload names
    /// the reduction context in which the overflow occurred.
    #[error("reduction arithmetic overflow: {context}")]
    ReductionOverflow {
        /// Which reduction step overflowed.
        context: String,
    },

    /// Firing a transition would push a place's token count past `u64::MAX`
    /// (output-arc add) or below zero (input-arc subtract). The net or its
    /// initial marking is malformed/oversized; the toolchain declines rather
    /// than wrapping into a wrong marking. Mirrors the trust-cg kernel's
    /// `TokenOverflow` fail-closed contract.
    #[error(
        "marking token-count overflow at place {place}: \
         {value} {op} {weight} is not representable in u64"
    )]
    MarkingOverflow {
        /// Index of the place whose token count overflowed.
        place: u32,
        /// The place's token count before the operation.
        value: u64,
        /// The arc weight being added or subtracted.
        weight: u64,
        /// The operation that overflowed (`"+"` for an output arc, `"-"` for
        /// an input arc).
        op: &'static str,
    },

    /// The requested examination name is not one of the recognized MCC
    /// examinations; the payload is the unrecognized name.
    #[error("unknown examination: {0}")]
    UnknownExamination(String),

    /// A property-XML accessor was called for an examination that takes no
    /// property file (e.g. `StateSpace`); the payload names the examination.
    #[error("examination {examination} does not use property XML")]
    ExaminationDoesNotUsePropertyXml {
        /// Name of the examination that does not consume property XML.
        examination: String,
    },
}
