// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! JIT warmup / tuning gate constants.
//!
//! These thresholds drive the JIT warmup gate (`evaluate_jit_warmup_gate`) and
//! the validation cross-check sampling. They live here, separate from the BFS
//! profiling and flat-successor concerns, so the tuning knobs are easy to find.

/// Number of JIT-dispatched states to sample before making the warmup gate decision.
/// After this many states, cumulative JIT vs interpreter time is compared and JIT
/// is disabled if it's >20% slower.
///
/// Part of #4031: JIT warmup gate.
pub(in crate::check) const JIT_WARMUP_THRESHOLD: u32 = 500;

/// Ratio threshold: if JIT time / interpreter time exceeds this, JIT is disabled.
/// 1.2 means JIT must be no more than 20% slower than the interpreter.
///
/// Part of #4031: JIT warmup gate.
pub(in crate::check::model_checker::run_helpers) const JIT_SLOWDOWN_RATIO: f64 = 1.2;

/// Initial number of JIT validation cross-checks against the interpreter.
///
/// Used both as the initial value of `jit_validation_remaining` and in the
/// warmup gate calculation to compute the validation sample size. Previously
/// these were separate hardcoded `100` literals; this const ensures they stay
/// in sync.
///
/// Part of #4229: Sync hardcoded initial_validation with field default.
pub(in crate::check) const JIT_INITIAL_VALIDATION_COUNT: u32 = 100;
