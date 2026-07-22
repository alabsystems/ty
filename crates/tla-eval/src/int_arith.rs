// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

use super::{EvalError, EvalResult, Span, Value};
use num_bigint::BigInt;
use num_traits::{ToPrimitive, Zero};

// Integer arithmetic helper functions extracted from eval_arith.rs as part of
// #3424.

/// Apply binary arithmetic operation with SmallInt fast path
///
/// Used for +, -, * which don't need division-by-zero checks.
/// `op_name` is the TLC operator name for TLC-compatible error messages.
/// Part of #2955: inline hot-path arithmetic inner helper.
#[inline(always)]
pub(super) fn int_arith_op(
    left: Value,
    right: Value,
    small_op: impl Fn(i64, i64) -> Option<i64>,
    big_op: impl Fn(BigInt, BigInt) -> BigInt,
    op_name: &str,
    span: Option<Span>,
) -> EvalResult<Value> {
    // SmallInt fast path
    if let (Value::SmallInt(a), Value::SmallInt(b)) = (&left, &right) {
        if let Some(result) = small_op(*a, *b) {
            return Ok(Value::SmallInt(result));
        }
        // Overflow: fall through to BigInt
    }
    // BigInt path (TLC: EC.TLC_MODULE_ARGUMENT_ERROR)
    let a = left
        .to_bigint()
        .ok_or_else(|| EvalError::argument_error("first", op_name, "integer", &left, span))?;
    let b = right
        .to_bigint()
        .ok_or_else(|| EvalError::argument_error("second", op_name, "integer", &right, span))?;
    Ok(Value::big_int(big_op(a, b)))
}

/// Apply integer comparison operation with SmallInt fast path
/// `op_name` is the TLC operator name for TLC-compatible error messages.
pub(super) fn int_cmp_op(
    left: Value,
    right: Value,
    small_cmp: impl Fn(i64, i64) -> bool,
    big_cmp: impl Fn(&BigInt, &BigInt) -> bool,
    op_name: &str,
    span: Option<Span>,
) -> EvalResult<Value> {
    if let (Value::SmallInt(a), Value::SmallInt(b)) = (&left, &right) {
        return Ok(Value::Bool(small_cmp(*a, *b)));
    }

    let a = left
        .to_bigint()
        .ok_or_else(|| EvalError::argument_error("first", op_name, "integer", &left, span))?;
    let b = right
        .to_bigint()
        .ok_or_else(|| EvalError::argument_error("second", op_name, "integer", &right, span))?;
    Ok(Value::Bool(big_cmp(&a, &b)))
}

/// The shared inexact-`/` error. Every runtime engine (AST evaluator, TIR
/// tree-walker, bytecode VM) reports inexact real division through this one
/// constructor so the error variant AND message are identical across engines.
fn inexact_real_division_error(divisor: &Value, span: Option<Span>) -> EvalError {
    EvalError::argument_error(
        "second",
        "/",
        "divisor of the first argument (`/` is real division; use \\div for integer division)",
        divisor,
        span,
    )
}

/// Apply TLA+ real division `/` with EXACT-OR-ERROR semantics.
///
/// TLA+ `/` is division on the reals; applied to integers it only has an
/// integer value when the divisor evenly divides the dividend. TY does not
/// implement reals, so an inexact quotient is an evaluation ERROR — never a
/// truncation (a truncated `7 / 2 = 3` has no TLA+ meaning). This matches the
/// native codegen (`lower_real_division` declines to this path) and the TIR
/// const-folder (`const_prop`), which folds only exact quotients.
///
/// All runtime engines route through this single helper so cross-engine
/// parity holds by construction.
pub(super) fn int_exact_div_op(left: Value, right: Value, span: Option<Span>) -> EvalResult<Value> {
    // SmallInt fast path
    if let (Value::SmallInt(a), Value::SmallInt(b)) = (&left, &right) {
        if *b == 0 {
            return Err(EvalError::DivisionByZero { span });
        }
        // i64::MIN / -1 (and i64::MIN % -1) overflow i64: decline to the
        // BigInt path below, which yields the exact 2^63 (-1 divides
        // everything, so the division IS exact).
        if !(*a == i64::MIN && *b == -1) {
            if a % b != 0 {
                return Err(inexact_real_division_error(&right, span));
            }
            return Ok(Value::SmallInt(a / b));
        }
    }
    // BigInt path (TLC: EC.TLC_MODULE_ARGUMENT_ERROR)
    let a = left
        .to_bigint()
        .ok_or_else(|| EvalError::argument_error("first", "/", "integer", &left, span))?;
    let b = right
        .to_bigint()
        .ok_or_else(|| EvalError::argument_error("second", "/", "integer", &right, span))?;
    if b.is_zero() {
        return Err(EvalError::DivisionByZero { span });
    }
    if !(&a % &b).is_zero() {
        return Err(inexact_real_division_error(&right, span));
    }
    Ok(Value::big_int(a / b))
}

/// Apply division operation with SmallInt fast path and zero check
///
/// Used for \div which needs a division-by-zero check (real division `/`
/// goes through `int_exact_div_op` instead).
/// The `small_op` returns `Option<i64>` to handle potential overflow.
/// `op_name` is the TLC operator name for TLC-compatible error messages.
pub(super) fn int_div_op(
    left: Value,
    right: Value,
    small_op: impl Fn(i64, i64) -> Option<i64>,
    big_op: impl Fn(BigInt, BigInt) -> BigInt,
    op_name: &str,
    span: Option<Span>,
) -> EvalResult<Value> {
    // SmallInt fast path
    if let (Value::SmallInt(a), Value::SmallInt(b)) = (&left, &right) {
        if *b == 0 {
            return Err(EvalError::DivisionByZero { span });
        }
        if let Some(result) = small_op(*a, *b) {
            return Ok(Value::SmallInt(result));
        }
        // Overflow (e.g., MIN / -1): fall through to BigInt
    }
    // BigInt path (TLC: EC.TLC_MODULE_ARGUMENT_ERROR)
    let a = left
        .to_bigint()
        .ok_or_else(|| EvalError::argument_error("first", op_name, "integer", &left, span))?;
    let b = right
        .to_bigint()
        .ok_or_else(|| EvalError::argument_error("second", op_name, "integer", &right, span))?;
    if b.is_zero() {
        return Err(EvalError::DivisionByZero { span });
    }
    Ok(Value::big_int(big_op(a, b)))
}

/// Apply modulus operation with positive divisor check (TLC semantics)
///
/// TLC requires the divisor to be positive (> 0), not just non-zero.
/// Returns ModulusNotPositive error if divisor <= 0.
pub(super) fn int_mod_op(left: Value, right: Value, span: Option<Span>) -> EvalResult<Value> {
    // SmallInt fast path
    if let (Value::SmallInt(a), Value::SmallInt(b)) = (&left, &right) {
        if *b <= 0 {
            return Err(EvalError::ModulusNotPositive {
                value: b.to_string(),
                span,
            });
        }
        // Euclidean modulo (always non-negative for positive divisor)
        return Ok(Value::SmallInt(a.rem_euclid(*b)));
    }
    // BigInt path (TLC: EC.TLC_MODULE_ARGUMENT_ERROR)
    let a = left
        .to_bigint()
        .ok_or_else(|| EvalError::argument_error("first", "%", "integer", &left, span))?;
    let b = right
        .to_bigint()
        .ok_or_else(|| EvalError::argument_error("second", "%", "integer", &right, span))?;
    if b <= BigInt::from(0) {
        return Err(EvalError::ModulusNotPositive {
            value: b.to_string(),
            span,
        });
    }
    // Euclidean modulo (always non-negative for positive divisor)
    let r = &a % &b;
    let result = if r < BigInt::from(0) { r + &b } else { r };
    Ok(Value::big_int(result))
}

/// Apply power operation with SmallInt fast path
///
/// Special handling: exponent must be non-negative and fit in u32.
pub(super) fn int_pow_op(left: Value, right: Value, span: Option<Span>) -> EvalResult<Value> {
    // SmallInt fast path for small exponents
    if let (Value::SmallInt(base), Value::SmallInt(exp)) = (&left, &right) {
        if *exp >= 0 && *exp <= 62 {
            if let Some(result) = base.checked_pow(*exp as u32) {
                return Ok(Value::SmallInt(result));
            }
        }
        // Overflow or large exponent: fall through to BigInt
    }
    // BigInt path (TLC: EC.TLC_MODULE_ARGUMENT_ERROR)
    let base = left
        .to_bigint()
        .ok_or_else(|| EvalError::argument_error("first", "^", "integer", &left, span))?;
    let exp = right
        .to_bigint()
        .ok_or_else(|| EvalError::argument_error("second", "^", "integer", &right, span))?;
    let exp_u32 = exp.to_u32().ok_or_else(|| EvalError::Internal {
        message: "Exponent too large or negative".into(),
        span,
    })?;

    // Fail-closed result-size bound. base^exp has ~= exp * base.bits() bits.
    // Without this, e.g. 2^4_000_000_000 (exp fits u32) tries to allocate
    // hundreds of MB and can OOM/abort. Cap the *result* magnitude BEFORE
    // calling pow so oversized inputs DECLINE instead of crashing.
    //
    // MAX_POW_RESULT_BITS = 1_000_000 bits ~= 125 KB per BigInt: ample for
    // legitimate models, but a hard ceiling against adversarial blowup.
    // ±1 never grows (bits stays small), so it is always allowed.
    const MAX_POW_RESULT_BITS: u64 = 1_000_000;
    let base_bits = base.bits(); // bits() ignores sign; 0 for base == 0
    if base_bits > 1 {
        // saturating: avoid overflow when estimating the bit budget itself.
        let result_bits = (exp_u32 as u64).saturating_mul(base_bits);
        if result_bits > MAX_POW_RESULT_BITS {
            return Err(EvalError::ArgumentError {
                position: "second",
                op: "^".to_string(),
                expected_type: "exponent within the supported magnitude limit",
                value_display: exp.to_string(),
                span,
            });
        }
    }
    Ok(Value::big_int(base.pow(exp_u32)))
}
