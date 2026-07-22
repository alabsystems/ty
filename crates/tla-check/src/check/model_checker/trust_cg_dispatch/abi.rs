// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Native-ABI function-pointer type aliases for the trust-codegen dispatch path.
//!
//! These aliases capture the stable `extern "C"` ABI shared by the
//! trust-codegen native pipeline and the legacy JIT compatibility layer, so the
//! BFS dispatch logic can stay backend-neutral.

use tla_jit_abi::JitCallOut;

/// Type alias for the native next-state function pointer.
///
/// ABI: `extern "C" fn(out: *mut JitCallOut, state_in: *const i64, state_out: *mut i64, state_len: u32)`
///
/// - `out`: caller-allocated result struct. On success, `out.status = Ok` and
///   `out.value = 1` (enabled) or `0` (disabled). On error, status is set
///   to RuntimeError with error details.
/// - `state_in`: flat i64 array of current state variable values.
/// - `state_out`: flat i64 array for successor state (pre-allocated by caller).
/// - `state_len`: number of state variables.
pub(super) type NativeNextStateFn =
    unsafe extern "C" fn(*mut JitCallOut, *const i64, *mut i64, u32);
pub(super) type NativeImpliedActionFn =
    unsafe extern "C" fn(*mut JitCallOut, *const i64, *mut i64, u32);

/// Type alias for the native invariant function pointer.
///
/// ABI: `extern "C" fn(out: *mut JitCallOut, state: *const i64, state_len: u32)`
pub(super) type NativeInvariantFn = unsafe extern "C" fn(*mut JitCallOut, *const i64, u32);
