// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

use super::{check_arity, eval, EvalCtx, EvalError, EvalResult, Expr, Span, Spanned, Value};
use tla_value::Rp;
use crate::builtin_bags::bag_from_sorted_entries;
use num_bigint::BigInt;
use num_traits::{One, ToPrimitive, Zero};
use std::sync::Arc;

// BagsExt module operators — BagAdd, BagRemove, BagRemoveAll, FoldBag, SumBag, ProductBag

/// Value-level bag-op failure, so the AST arm can map to its exact
/// span-carrying `EvalError`s and the bytecode VM to `VmError` — while both
/// share ONE implementation (results are value- and fingerprint-identical).
pub(crate) enum BagOpError {
    /// The bag operand is not a bag/function-like value (carries the operand).
    NotBag(Value),
    /// A count entry is not an Int (carries the offending count value).
    NotInt(Value),
}

/// Value-level `BagAdd(B, e)` — shared by the interpreter arm and the
/// bytecode VM `CallBuiltin` dispatch. Semantics identical to the historical
/// AST arm: compact fast path first, then the general rebuild-and-sort path
/// promoted via `bag_from_sorted_entries`.
pub(crate) fn bag_add_value(bv: &Value, ev: &Value) -> Result<Value, BagOpError> {
    if let Value::Bag(b) = bv {
        if let Some(added) = b.bag_add(ev) {
            return Ok(Value::Bag(Rp::new(added)));
        }
    }
    let func = bv
        .to_func_coerced()
        .ok_or_else(|| BagOpError::NotBag(bv.clone()))?;
    let mut entries: Vec<(Value, Value)> = func
        .iter()
        .map(|(key, value)| (key.clone(), value.clone()))
        .collect();
    let mut found = false;
    for entry in &mut entries {
        if entry.0 == *ev {
            let n = entry
                .1
                .to_bigint()
                .ok_or_else(|| BagOpError::NotInt(entry.1.clone()))?;
            entry.1 = Value::big_int(n + BigInt::one());
            found = true;
            break;
        }
    }
    if !found {
        entries.push((ev.clone(), Value::SmallInt(1)));
        entries.sort_by(|a, b| a.0.cmp(&b.0));
    }
    Ok(bag_from_sorted_entries(entries))
}

/// Value-level `BagRemove(B, e)` — shared implementation (see
/// [`bag_add_value`]). Absent element == unchanged bag, count reaching zero
/// drops the entry, identical to the historical AST arm.
pub(crate) fn bag_remove_value(bv: &Value, ev: &Value) -> Result<Value, BagOpError> {
    if let Value::Bag(b) = bv {
        return Ok(match b.bag_remove(ev) {
            Some(removed) => Value::Bag(Rp::new(removed)),
            None => bv.clone(),
        });
    }
    let func = bv
        .to_func_coerced()
        .ok_or_else(|| BagOpError::NotBag(bv.clone()))?;
    let mut entries: Vec<(Value, Value)> = Vec::new();
    for (key, val) in func.mapping_iter() {
        if *key == *ev {
            let n = val
                .to_bigint()
                .ok_or_else(|| BagOpError::NotInt(val.clone()))?;
            let new_count = n - BigInt::one();
            if new_count > BigInt::zero() {
                entries.push((key.clone(), Value::big_int(new_count)));
            }
            // else: drop this entry (count becomes 0 or negative)
        } else {
            entries.push((key.clone(), val.clone()));
        }
    }
    Ok(bag_from_sorted_entries(entries))
}

pub(super) fn eval_builtin_bagsext(
    ctx: &EvalCtx,
    name: &str,
    args: &[Spanned<Expr>],
    span: Option<Span>,
) -> EvalResult<Option<Value>> {
    match name {
        "BagAdd" => {
            // BagAdd(B, e) - add 1 to count of e in bag B.
            // Shared value-level implementation (also the bytecode VM's
            // CallBuiltin dispatch): compact fast path + general
            // rebuild-and-sort path promoted via bag_from_sorted_entries.
            check_arity(name, args, 2, span)?;
            let bv = eval(ctx, &args[0])?;
            let ev = eval(ctx, &args[1])?;
            match bag_add_value(&bv, &ev) {
                Ok(v) => Ok(Some(v)),
                Err(BagOpError::NotBag(v)) => Err(EvalError::type_error(
                    "Bag/Function",
                    &v,
                    Some(args[0].span),
                )),
                Err(BagOpError::NotInt(v)) => Err(EvalError::type_error("Int", &v, span)),
            }
        }

        "BagRemove" => {
            // BagRemove(B, e) - remove 1 from count of e in bag B.
            // Shared value-level implementation (also the bytecode VM's
            // CallBuiltin dispatch). Absent element == unchanged bag.
            check_arity(name, args, 2, span)?;
            let bv = eval(ctx, &args[0])?;
            let ev = eval(ctx, &args[1])?;
            match bag_remove_value(&bv, &ev) {
                Ok(v) => Ok(Some(v)),
                Err(BagOpError::NotBag(v)) => Err(EvalError::type_error(
                    "Bag/Function",
                    &v,
                    Some(args[0].span),
                )),
                Err(BagOpError::NotInt(v)) => Err(EvalError::type_error("Int", &v, span)),
            }
        }

        "BagRemoveAll" => {
            // BagRemoveAll(B, e) - completely remove e from bag B
            check_arity(name, args, 2, span)?;
            let bv = eval(ctx, &args[0])?;
            let ev = eval(ctx, &args[1])?;
            // Compact fast path. Absent element (None) == unchanged bag.
            if let Value::Bag(b) = &bv {
                return Ok(Some(match b.bag_remove_all(&ev) {
                    Some(removed) => Value::Bag(Rp::new(removed)),
                    None => bv.clone(),
                }));
            }
            // Use to_func_coerced to accept Seq/IntFunc/Tuple — intern table substitution (#1713)
            let func = bv
                .to_func_coerced()
                .ok_or_else(|| EvalError::type_error("Bag/Function", &bv, Some(args[0].span)))?;

            // Copy entries excluding the element
            let entries: Vec<(Value, Value)> = func
                .mapping_iter()
                .filter(|(k, _)| *k != &ev)
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect();
            Ok(Some(bag_from_sorted_entries(entries)))
        }

        "FoldBag" => {
            // FoldBag(op, base, B) - fold a binary operator over all elements in a bag
            // Each element e with count n appears n times in the fold
            check_arity(name, args, 3, span)?;

            // Get the operator name from the first argument
            let op_name = match &args[0].node {
                Expr::Ident(name, _) => name.clone(),
                _ => {
                    return Err(EvalError::Internal {
                        message: "FoldBag requires operator name as first argument".into(),
                        span,
                    });
                }
            };

            let base = eval(ctx, &args[1])?;
            let bv = eval(ctx, &args[2])?;
            // Use to_func_coerced to accept Seq/IntFunc/Tuple — intern table substitution (#1713)
            let func = bv
                .to_func_coerced()
                .ok_or_else(|| EvalError::type_error("Bag/Function", &bv, Some(args[2].span)))?;

            // Get the operator definition
            let op_def = ctx.get_op(&op_name).ok_or_else(|| EvalError::UndefinedOp {
                name: op_name.clone(),
                span,
            })?;

            if op_def.params.len() != 2 {
                return Err(EvalError::ArityMismatch {
                    op: op_name,
                    expected: 2,
                    got: op_def.params.len(),
                    span,
                });
            }

            // Fold over the bag elements (each element appears count times)
            let mut result = base;
            for (elem, count_val) in func.mapping_iter() {
                let count = count_val
                    .to_bigint()
                    .ok_or_else(|| EvalError::type_error("Int", count_val, span))?
                    .to_i64()
                    .unwrap_or(0);
                // Apply the operator count times for this element
                for _ in 0..count {
                    let new_ctx = ctx.bind2(
                        op_def.params[0].name.node.as_str(),
                        result,
                        op_def.params[1].name.node.as_str(),
                        elem.clone(),
                    );
                    result = eval(&new_ctx, &op_def.body)?;
                }
            }

            Ok(Some(result))
        }

        "SumBag" => {
            // SumBag(B) - sum of element * count for each element in bag
            // SumBag([1 |-> 2, 3 |-> 1]) = 1*2 + 3*1 = 5
            check_arity(name, args, 1, span)?;
            let bv = eval(ctx, &args[0])?;
            // Use to_func_coerced to accept Seq/IntFunc/Tuple — intern table substitution (#1713)
            let func = bv
                .to_func_coerced()
                .ok_or_else(|| EvalError::type_error("Bag/Function", &bv, Some(args[0].span)))?;

            let mut sum = BigInt::zero();
            for (elem, count_val) in func.mapping_iter() {
                let elem_int = elem
                    .to_bigint()
                    .ok_or_else(|| EvalError::type_error("Int", elem, span))?;
                let count = count_val
                    .to_bigint()
                    .ok_or_else(|| EvalError::type_error("Int", count_val, span))?;
                sum += elem_int * count;
            }
            Ok(Some(Value::big_int(sum)))
        }

        "ProductBag" => {
            // ProductBag(B) - product of element^count for each element in bag
            // ProductBag([2 |-> 3, 3 |-> 2]) = 2^3 * 3^2 = 8 * 9 = 72
            check_arity(name, args, 1, span)?;
            let bv = eval(ctx, &args[0])?;
            // Use to_func_coerced to accept Seq/IntFunc/Tuple — intern table substitution (#1713)
            let func = bv
                .to_func_coerced()
                .ok_or_else(|| EvalError::type_error("Bag/Function", &bv, Some(args[0].span)))?;

            let mut product = BigInt::one();
            for (elem, count_val) in func.mapping_iter() {
                let elem_int = elem
                    .to_bigint()
                    .ok_or_else(|| EvalError::type_error("Int", elem, span))?;
                let count = count_val
                    .to_bigint()
                    .ok_or_else(|| EvalError::type_error("Int", count_val, span))?
                    .to_i64()
                    .unwrap_or(0);
                // Multiply product by elem^count
                for _ in 0..count {
                    product *= &elem_int;
                }
            }
            Ok(Some(Value::big_int(product)))
        }

        // Not handled by this domain
        _ => Ok(None),
    }
}
