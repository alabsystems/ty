// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

// Author: Andrew Yates

//! BTOR2 parser, intermediate representation, and solver pipeline for hardware
//! model checking.
//!
//! [BTOR2] is the word-level circuit format used by the Hardware Model Checking
//! Competition (HWMCC). A BTOR2 program is a flat, line-oriented netlist of
//! bitvector/array nodes that declares state and input variables, an `init`
//! relation, a `next` (transition) relation, environment `constraint`s, and one
//! or more `bad` safety properties. The system is *safe* for a property when its
//! `bad` signal is unreachable from the initial states under the transition
//! relation.
//!
//! This crate turns such a netlist into a verification verdict.
//!
//! # Pipeline
//!
//! 1. **Parse** ([`parse`], [`parse_file`]) text BTOR2 into the [`Btor2Program`]
//!    IR ([`types`]), validating every cross-reference and rejecting bitvector
//!    widths above [`error::MAX_BV_WIDTH`].
//! 2. **Translate** ([`translate_to_chc`], [`to_chc`]) the program into a system
//!    of Constrained Horn Clauses (CHC), encoding `init`/`next`/`bad` as an
//!    inductive-invariant query for the `ay-chc` solver.
//! 3. **Solve** via either the high-level [`check_btor2_portfolio`] (COI
//!    reduction, simplification, bounded model checking, then full CHC solving)
//!    or [`bitblast()`] the netlist to a bit-level AIGER circuit for a SAT-based
//!    IC3/PDR engine on narrow benchmarks.
//!
//! Every verdict path is *fail-closed*: any condition that cannot be modeled
//! faithfully (an over-wide bitvector, an unparseable constant, an
//! independently-unverifiable safety proof) yields an error or an `Unknown`
//! verdict rather than a possibly-wrong SAFE/UNSAFE answer.
//!
//! # Verdicts
//!
//! Per `bad` property the solver returns a [`Btor2CheckResult`]: `Unsat` (the
//! property holds), `Sat` (a concrete counterexample trace was found), or
//! `Unknown` (the solver could not decide). SAFE verdicts from the adaptive
//! path are proof-backed: the discovered invariant is re-verified on a fresh
//! solver before being reported.
//!
//! # Example
//!
//! Parse a tiny BTOR2 program and check its single `bad` property:
//!
//! ```
//! use tla_btor2::{parse, check_btor2, Btor2CheckResult};
//!
//! // A 1-bit register that holds a constant 0; the `bad` signal can never fire.
//! let src = "\
//! 1 sort bitvec 1
//! 2 zero 1
//! 3 state 1 r
//! 4 init 1 3 2
//! 5 next 1 3 3
//! 6 bad 3
//! ";
//! let program = parse(src).expect("valid BTOR2");
//! let results = check_btor2(&program).expect("translation succeeds");
//! assert!(matches!(results.as_slice(), [Btor2CheckResult::Unsat]));
//! ```
//!
//! Reference: *BTOR2, BtorMC and Boolector 3.0*, Niemetz, Preiner, Wolf, Biere
//! (CAV 2018).
//!
//! [BTOR2]: https://fmv.jku.at/papers/NiemetzPreinerWolfBiere-CAV18.pdf

#![deny(missing_docs)]

// `array_battery` (four-way array differential battery) and `array_oracle`
// (K-step ground-truth enumerator) compile under `cfg(test)` ONLY — they are
// differential harnesses, never a production verdict path.
#[cfg(test)]
mod array_battery;
pub mod array_bmc;
pub mod array_cert;
pub mod array_elim;
pub mod array_ic3;
#[cfg(test)]
mod array_oracle;
pub mod bitblast;
pub(crate) mod bmc;
pub(crate) mod coi;
pub mod error;
pub(crate) mod gpu_exhaustive;
pub(crate) mod gpu_falsify;
pub mod parser;
pub mod portfolio;
pub mod shared_engine_evidence;
pub(crate) mod simplify;
pub mod to_chc;
pub mod translate;
pub mod types;
pub mod witness;
pub mod word_replay;

pub use array_bmc::{
    check_array_bmc, check_array_kinduction, ArrayBmcConfig, ArrayBmcOutcome, ArrayKindConfig,
    ArrayKindOutcome,
};
pub use array_cert::{certify_btor2_safe_independent, IndependentCertResult};
pub use array_ic3::{
    check_array_ic3, ArrayCertTier, ArrayFrameInvariant, ArrayIc3Config, ArrayIc3Outcome, InvAtom,
    InvLit,
};
pub use bitblast::{bitblast, bitblast_eligible, BitblastedCircuit};
pub use error::Btor2Error;
pub use parser::{parse, parse_btor2, parse_file};
pub use portfolio::{
    btor2_accept_concrete_trace_replay, btor2_hardware_replay_decision_evidence,
    btor2_hardware_replay_decision_status, btor2_hardware_replay_primitive_status,
    btor2_portfolio_capability_report, btor2_unsafe_proof_replay_artifact, check_btor2_portfolio,
    check_btor2_portfolio_with_report, validate_btor2_hardware_replay_decision_evidence,
    validate_btor2_hardware_replay_decision_evidence_row, Btor2ConcreteTraceReplayAcceptance,
    Btor2ConcreteTraceReplayRejection, Btor2UnsafeProofReplayArtifact, PortfolioConfig,
    PortfolioStats, ResultPhase,
};
pub use shared_engine_evidence::{
    btor2_prepared_checker_program, btor2_prepared_program_identity_digest,
    btor2_shared_engine_evidence_rows, Btor2SharedEngineEvidence,
};
pub use tla_mc_core::{
    HardwareReplayDecisionEvidenceError, HardwareReplayDecisionStatus,
    HardwareReplayPrimitiveAssignmentStatus, HardwareReplayPrimitiveConsumerStatus,
    HardwareReplayPrimitiveDecisionStatus, HardwareReplayPrimitiveRejectionReason,
    HardwareReplayPrimitiveStatus, HARDWARE_REPLAY_DECISION_REQUIRED_FIELDS,
    HARDWARE_REPLAY_DECISION_ROW_KIND, HARDWARE_REPLAY_DECISION_SCHEMA,
    HARDWARE_REPLAY_DECISION_SCHEMA_VERSION, HARDWARE_REPLAY_PRIMITIVE_SCHEMA,
};
pub use to_chc::{check_btor2_adaptive, translate_to_chc, TranslationResult};
pub use translate::{check_btor2, Btor2CheckResult};
pub use types::{Btor2Line, Btor2Node, Btor2Program, Btor2Sort};
pub use witness::{project_bitblast_witness, Btor2Witness};
pub use word_replay::{
    build_word_level_witness, word_level_replay, InitialState, InputFrame, WordLevelModel,
    WordValue,
};
