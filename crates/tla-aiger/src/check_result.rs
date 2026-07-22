// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Shared result type for SAT-based AIGER model checking engines.

use rustc_hash::FxHashMap;

/// Result of a SAT-based model checking run.
#[derive(Debug, Clone)]
pub enum CheckResult {
    /// The property holds (no bad state is reachable).
    Safe,
    /// A bad state is reachable. Contains the depth and counterexample trace.
    Unsafe {
        /// Length of the counterexample (number of transition steps to the bad state).
        depth: usize,
        /// Per-step assignment of signal names to boolean values along the trace.
        trace: Vec<FxHashMap<String, bool>>,
    },
    /// Could not determine the result.
    Unknown {
        /// Why the run was inconclusive (e.g. timeout, depth limit).
        reason: String,
    },
}

impl CheckResult {
    /// Returns true if this is a definitive result (Safe or Unsafe).
    pub fn is_definitive(&self) -> bool {
        matches!(self, CheckResult::Safe | CheckResult::Unsafe { .. })
    }
}
