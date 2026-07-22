// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Thread-based portfolio solver for AIGER safety checking.
//!
//! Runs multiple engines (IC3 variants, BMC, k-induction) in parallel and
//! returns the first definitive result. Uses `Arc<AtomicBool>` for cooperative
//! cancellation.

pub mod adaptive;
pub mod config;
pub mod factory;
pub mod runner;
pub mod safe_witness;

#[cfg(test)]
mod tests;

// Re-export public API for backward compatibility.
pub use adaptive::{
    default_preset_pool, portfolio_check_adaptive, AdaptivePortfolioConfig, AdaptiveScheduler,
};
pub use config::{
    single_bdd_reach, single_bmc, single_ic3, EngineConfig, PortfolioConfig, PortfolioResult,
};
pub use factory::*;
pub use runner::{
    aiger_hardware_replay_decision_evidence, aiger_hardware_replay_primitive_status,
    aiger_portfolio_capability_report, portfolio_check, portfolio_check_detailed,
    portfolio_check_detailed_with_report, validate_aiger_hardware_replay_decision_evidence,
    validate_aiger_hardware_replay_decision_evidence_row,
};
pub use safe_witness::{validate_safe, validate_safe_with_budget, SafeValidation, SafeWitness};
pub use tla_mc_core::{
    hardware_replay_decision_accepts_replay_primitive,
    validate_hardware_replay_decision_evidence_row, HardwareReplayDecisionEvidenceError,
    HardwareReplayPrimitiveAssignmentStatus, HardwareReplayPrimitiveConsumerStatus,
    HardwareReplayPrimitiveDecisionStatus, HardwareReplayPrimitiveRejectionReason,
    HardwareReplayPrimitiveStatus, HARDWARE_REPLAY_DECISION_REQUIRED_FIELDS,
    HARDWARE_REPLAY_DECISION_ROW_KIND, HARDWARE_REPLAY_DECISION_SCHEMA,
    HARDWARE_REPLAY_DECISION_SCHEMA_VERSION, HARDWARE_REPLAY_PRIMITIVE_SCHEMA,
};
