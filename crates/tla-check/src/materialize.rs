// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! State value materialization for TLC-compatible fingerprinting.
//!
//! TLC always materializes lazy values before fingerprinting:
//! - `FcnLambdaValue.fingerPrint()` calls `toFcnRcd()` (enumerate domain, evaluate body)
//! - `SetPredValue.fingerPrint()` calls `toSetEnum()` (enumerate source, filter by predicate)
//! - `OpLambdaValue.fingerPrint()` calls `Assert.fail()` (forbidden in state variables)
//!
//! TY must match this behavior to produce deterministic, content-based fingerprints.
//! Without materialization, lazy values use process-local IDs or structural hashes
//! that cause non-determinism (#1989), state space expansion (#1865), and false-unique
//! states (#1914).
//!
//! Part of #2018: Materialize before fingerprinting.

use crate::error::{EvalError, EvalResult};
use crate::state::{ArrayState, DiffChanges};
use crate::value::IntIntervalFunc;
use crate::var_index::VarIndex;
use crate::Value;
use tla_core::ast::{Expr, Module, Unit};
use tla_core::{walk_expr, ExprVisitor};
use tla_eval::EvalCtx;
use tla_eval::{materialize_lazy_func_to_func, materialize_setpred_to_vec};
use tla_value::Rp;

/// AST visitor that detects expressions which produce lazy values at runtime.
///
/// Returns `true` (short-circuiting) when `FuncDef`, `SetFilter`, or `Lambda`
/// is found. These are the only AST nodes that produce `LazyFunc`, `SetPred`,
/// or `Closure` values respectively.
struct ContainsLazyProducers;

impl ExprVisitor for ContainsLazyProducers {
    type Output = bool;

    fn visit_node(&mut self, expr: &Expr) -> Option<bool> {
        match expr {
            Expr::FuncDef(_, _) | Expr::SetFilter(_, _) | Expr::Lambda(_, _) => Some(true),
            _ => None,
        }
    }
}

/// Determine whether a spec's AST contains any expressions that can produce
/// lazy values at runtime (`LazyFunc`, `SetPred`, `Closure`).
///
/// Scans all operator definitions in the module and its extensions for
/// `FuncDef` (`[x \in S |-> e]`), `SetFilter` (`{x \in S : P(x)}`), and
/// `Lambda` expressions. When none are present, materialization can be
/// skipped entirely during BFS because no lazy values will ever appear
/// in state variables.
///
/// Part of #4053: Skip per-successor `has_lazy_state_value` when the spec
/// cannot produce lazy values.
pub(crate) fn spec_may_produce_lazy_values(module: &Module, extended_modules: &[&Module]) -> bool {
    let modules = std::iter::once(module).chain(extended_modules.iter().copied());
    for m in modules {
        for unit in &m.units {
            if let Unit::Operator(op_def) = &unit.node {
                if walk_expr(&mut ContainsLazyProducers, &op_def.body.node) {
                    return true;
                }
            }
        }
    }
    false
}

/// Check whether a value (or any nested child) contains lazy types that
/// need materialization before fingerprinting.
///
/// Lazy types: `LazyFunc`, `SetPred`, `Closure`.
pub(crate) fn has_lazy_state_value(value: &Value) -> bool {
    match value {
        Value::LazyFunc(_) | Value::SetPred(_) | Value::Closure(_) => true,
        Value::Tuple(elems) => elems.iter().any(has_lazy_state_value),
        Value::Seq(seq) => seq.iter().any(has_lazy_state_value),
        Value::Record(rec) => rec.iter().any(|(_, v)| has_lazy_state_value(v)),
        Value::Func(func) => func
            .mapping_iter()
            .any(|(k, v)| has_lazy_state_value(k) || has_lazy_state_value(v)),
        Value::IntFunc(func) => func.values().iter().any(has_lazy_state_value),
        Value::Set(set) => set.iter().any(has_lazy_state_value),
        Value::Subset(subset) => has_lazy_state_value(subset.base()),
        Value::FuncSet(func_set) => {
            has_lazy_state_value(func_set.domain()) || has_lazy_state_value(func_set.codomain())
        }
        Value::RecordSet(record_set) => record_set
            .fields_iter()
            .any(|(_, field_set)| has_lazy_state_value(field_set)),
        Value::TupleSet(tuple_set) => tuple_set.components_iter().any(has_lazy_state_value),
        Value::SetCup(cup) => has_lazy_state_value(cup.set1()) || has_lazy_state_value(cup.set2()),
        Value::SetCap(cap) => has_lazy_state_value(cap.set1()) || has_lazy_state_value(cap.set2()),
        Value::SetDiff(diff) => {
            has_lazy_state_value(diff.set1()) || has_lazy_state_value(diff.set2())
        }
        Value::KSubset(k_subset) => has_lazy_state_value(k_subset.base()),
        Value::BigUnion(union) => has_lazy_state_value(union.set()),
        Value::SeqSet(seq_set) => has_lazy_state_value(seq_set.base()),
        _ => false,
    }
}

/// Materialize a single value: convert lazy representations to concrete data.
///
/// - `SetPred` → `Value::Set` (via predicate evaluation, matching TLC's `toSetEnum()`)
/// - `LazyFunc` → `Value::Func` (via domain enumeration + body evaluation,
///   matching TLC's `toFcnRcd()`). Returns error for non-enumerable domains.
/// - `Closure` → error (TLC forbids operator lambdas in state variables)
///
/// Recursively materializes nested lazy values within compound types
/// (tuples, sequences, records, functions, sets, and lazy set wrappers).
///
/// Part of #2018: Materialize before fingerprinting.
pub(crate) fn materialize_value(ctx: &EvalCtx, value: &Value) -> EvalResult<Value> {
    match value {
        Value::SetPred(spv) => {
            // TLC: SetPredValue.fingerPrint() → toSetEnum()
            let elements = materialize_setpred_to_vec(ctx, spv)?;
            let materialized = Value::set(elements);
            // A predicate set may itself contain lazy values. Concretizing the
            // outer set is not sufficient in that case: state values must be
            // expression-free at every depth before fingerprinting.
            if has_lazy_state_value(&materialized) {
                materialize_value(ctx, &materialized)
            } else {
                Ok(materialized)
            }
        }
        Value::LazyFunc(f) => {
            // TLC: FcnLambdaValue.fingerPrint() → toFcnRcd()
            // Enumerate domain, evaluate body for each element, produce concrete Func.
            // Returns error for non-enumerable domains (Nat, Int, Real, String).
            let materialized = materialize_lazy_func_to_func(ctx, f)?;
            // Recursively materialize any lazy values in the function's range
            if has_lazy_state_value(&materialized) {
                materialize_value(ctx, &materialized)
            } else {
                Ok(materialized)
            }
        }
        Value::Closure(_) => Err(EvalError::Internal {
            message: "TY has found a state in which the value of a variable \
                contains an operator (LAMBDA). TLC does not allow operator \
                values in state variables."
                .to_string(),
            span: None,
        }),
        // Compound types with lazy children: recursively materialize
        _ if has_lazy_state_value(value) => materialize_children(ctx, value),
        // No lazy values — return as-is
        _ => Ok(value.clone()),
    }
}

/// Recursively materialize lazy children within compound value types.
///
/// Called by `materialize_value` when a compound value (tuple, seq, record,
/// func, set, or lazy set wrapper) contains nested lazy values that need
/// materialization.
fn materialize_children(ctx: &EvalCtx, value: &Value) -> EvalResult<Value> {
    match value {
        Value::Tuple(elems) => {
            let m: Vec<Value> = elems
                .iter()
                .map(|v| materialize_value(ctx, v))
                .collect::<EvalResult<Vec<_>>>()?;
            Ok(Value::Tuple(m.into()))
        }
        Value::Seq(seq) => {
            let m: Vec<Value> = seq
                .iter()
                .map(|v| materialize_value(ctx, v))
                .collect::<EvalResult<Vec<_>>>()?;
            Ok(Value::seq(m))
        }
        Value::Record(rec) => {
            let entries: Vec<_> = rec
                .iter_str()
                .map(|(k, v)| Ok((k, materialize_value(ctx, v)?)))
                .collect::<EvalResult<Vec<_>>>()?;
            Ok(Value::record(entries))
        }
        Value::Func(func) => {
            let mut builder = crate::value::FuncBuilder::with_capacity(func.domain_len());
            for (k, v) in func.mapping_iter() {
                builder.insert(materialize_value(ctx, k)?, materialize_value(ctx, v)?);
            }
            Ok(Value::Func(Rp::new(builder.build())))
        }
        Value::IntFunc(func) => {
            let m: Vec<Value> = func
                .values()
                .iter()
                .map(|v| materialize_value(ctx, v))
                .collect::<EvalResult<Vec<_>>>()?;
            Ok(Value::IntFunc(Rp::new(crate::value::IntIntervalFunc::new(
                IntIntervalFunc::min(func),
                IntIntervalFunc::max(func),
                m,
            ))))
        }
        Value::Set(set) => {
            let m: Vec<Value> = set
                .iter()
                .map(|v| materialize_value(ctx, v))
                .collect::<EvalResult<Vec<_>>>()?;
            Ok(Value::set(m))
        }
        Value::Subset(subset) => Ok(Value::Subset(crate::value::SubsetValue::new(
            materialize_value(ctx, subset.base())?,
        ))),
        Value::FuncSet(func_set) => Ok(Value::FuncSet(crate::value::FuncSetValue::new(
            materialize_value(ctx, func_set.domain())?,
            materialize_value(ctx, func_set.codomain())?,
        ))),
        Value::RecordSet(record_set) => {
            let fields: Vec<_> = record_set
                .fields_iter()
                .map(|(name, field_set)| Ok((name.clone(), materialize_value(ctx, field_set)?)))
                .collect::<EvalResult<Vec<_>>>()?;
            Ok(Value::RecordSet(Rp::new(
                crate::value::RecordSetValue::new(fields),
            )))
        }
        Value::TupleSet(tuple_set) => {
            let components: Vec<_> = tuple_set
                .components_iter()
                .map(|component| materialize_value(ctx, component))
                .collect::<EvalResult<Vec<_>>>()?;
            Ok(Value::TupleSet(Rp::new(crate::value::TupleSetValue::new(
                components,
            ))))
        }
        Value::SetCup(cup) => Ok(Value::SetCup(crate::value::SetCupValue::new(
            materialize_value(ctx, cup.set1())?,
            materialize_value(ctx, cup.set2())?,
        ))),
        Value::SetCap(cap) => Ok(Value::SetCap(crate::value::SetCapValue::new(
            materialize_value(ctx, cap.set1())?,
            materialize_value(ctx, cap.set2())?,
        ))),
        Value::SetDiff(diff) => Ok(Value::SetDiff(crate::value::SetDiffValue::new(
            materialize_value(ctx, diff.set1())?,
            materialize_value(ctx, diff.set2())?,
        ))),
        Value::KSubset(k_subset) => Ok(Value::KSubset(crate::value::KSubsetValue::new(
            materialize_value(ctx, k_subset.base())?,
            k_subset.k(),
        ))),
        Value::BigUnion(union) => Ok(Value::BigUnion(crate::value::UnionValue::new(
            materialize_value(ctx, union.set())?,
        ))),
        Value::SeqSet(seq_set) => Ok(Value::SeqSet(crate::value::SeqSetValue::new(
            materialize_value(ctx, seq_set.base())?,
        ))),
        _ => Ok(value.clone()),
    }
}

/// Materialize lazy values in all state variables of an ArrayState.
///
/// This is the TLC-compatible normalization pass that should be called before
/// computing fingerprints. It ensures all values are concrete data without
/// process-local IDs, closures, or unevaluated expressions.
///
/// When `spec_may_produce_lazy` is `false`, the spec's AST contains no
/// `FuncDef`, `SetFilter`, or `Lambda` nodes, so lazy values are impossible
/// and the function returns immediately without scanning any values.
///
/// Part of #4053: Skip per-successor `has_lazy_state_value` scan.
///
/// Returns `true` if any values were materialized (state was modified).
pub(crate) fn materialize_array_state(
    ctx: &EvalCtx,
    state: &mut ArrayState,
    spec_may_produce_lazy: bool,
) -> EvalResult<bool> {
    if !spec_may_produce_lazy {
        return Ok(false);
    }
    let mut modified = false;
    let len = state.len();
    for i in 0..len {
        let value = state.get(VarIndex::new(i));
        if has_lazy_state_value(&value) {
            let materialized = materialize_value(ctx, &value)?;
            state.set(VarIndex::new(i), materialized);
            modified = true;
        }
    }
    Ok(modified)
}

/// Materialize lazy values in diff successor changes.
///
/// Call before computing the diff fingerprint to ensure changed values
/// are concrete data. This avoids ID-based fingerprints for lazy values
/// in the diff path.
///
/// When `spec_may_produce_lazy` is `false`, returns immediately.
///
/// Part of #4053: Skip per-successor `has_lazy_state_value` scan.
///
/// Returns `true` if any values were materialized.
pub(crate) fn materialize_diff_changes(
    ctx: &EvalCtx,
    changes: &mut DiffChanges,
    spec_may_produce_lazy: bool,
) -> EvalResult<bool> {
    if !spec_may_produce_lazy {
        return Ok(false);
    }
    let mut modified = false;
    for (_idx, value) in changes.iter_mut() {
        if has_lazy_state_value(value) {
            *value = materialize_value(ctx, value)?;
            modified = true;
        }
    }
    Ok(modified)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use tla_core::ast::{BoundVar, Expr};
    use tla_core::kani_types::HashMap;
    use tla_core::{FileId, Span, Spanned};

    fn true_set_pred(source: Value) -> Value {
        let dummy_span = Span {
            file: FileId(0),
            start: 0,
            end: 0,
        };
        Value::SetPred(Box::new(crate::value::SetPredValue::new(
            source,
            BoundVar {
                name: Spanned {
                    node: "x".to_string(),
                    span: dummy_span,
                },
                domain: None,
                pattern: None,
            },
            Spanned {
                node: Expr::Bool(true),
                span: dummy_span,
            },
            HashMap::new(),
            None,
            None,
        )))
    }

    #[test]
    fn test_has_lazy_state_value_concrete() {
        assert!(!has_lazy_state_value(&Value::SmallInt(42)));
        assert!(!has_lazy_state_value(&Value::Bool(true)));
        assert!(!has_lazy_state_value(&Value::set(vec![
            Value::SmallInt(1),
            Value::SmallInt(2),
        ])));
        assert!(!has_lazy_state_value(&Value::Tuple(
            vec![Value::SmallInt(1), Value::Bool(false)].into()
        )));
    }

    #[test]
    fn test_has_lazy_state_value_nested_setpred() {
        let setpred = true_set_pred(Value::set(vec![Value::SmallInt(1)]));

        // Direct SetPred
        assert!(has_lazy_state_value(&setpred));

        // SetPred nested in tuple
        let nested = Value::Tuple(vec![Value::SmallInt(1), setpred].into());
        assert!(has_lazy_state_value(&nested));
    }

    #[test]
    fn test_materialize_lazy_set_wrappers_removes_nested_setpred() {
        let setpred = true_set_pred(Value::set(vec![Value::SmallInt(1)]));
        let concrete = Value::set(vec![Value::SmallInt(2)]);
        let wrappers = [
            Value::Subset(crate::value::SubsetValue::new(setpred.clone())),
            Value::FuncSet(crate::value::FuncSetValue::new(
                concrete.clone(),
                setpred.clone(),
            )),
            Value::RecordSet(Rp::new(crate::value::RecordSetValue::new([(
                Arc::<str>::from("field"),
                setpred.clone(),
            )]))),
            Value::TupleSet(Rp::new(crate::value::TupleSetValue::new([setpred.clone()]))),
            Value::SetCup(crate::value::SetCupValue::new(
                setpred.clone(),
                concrete.clone(),
            )),
            Value::SetCap(crate::value::SetCapValue::new(
                concrete.clone(),
                setpred.clone(),
            )),
            Value::SetDiff(crate::value::SetDiffValue::new(
                concrete.clone(),
                setpred.clone(),
            )),
            Value::KSubset(crate::value::KSubsetValue::new(setpred.clone(), 1)),
            Value::BigUnion(crate::value::UnionValue::new(setpred.clone())),
            Value::SeqSet(crate::value::SeqSetValue::new(setpred)),
        ];

        let ctx = EvalCtx::new();
        for wrapper in wrappers {
            assert!(has_lazy_state_value(&wrapper), "missed {wrapper:?}");
            let materialized = materialize_value(&ctx, &wrapper).unwrap();
            assert!(
                !has_lazy_state_value(&materialized),
                "failed to eliminate nested lazy value from {wrapper:?}"
            );
        }
    }

    #[test]
    fn test_materialize_setcup_with_setpred_child() {
        let value = Value::SetCup(crate::value::SetCupValue::new(
            true_set_pred(Value::set(vec![Value::SmallInt(1), Value::SmallInt(2)])),
            Value::set(vec![Value::SmallInt(2), Value::SmallInt(3)]),
        ));

        assert!(has_lazy_state_value(&value));
        let materialized = materialize_value(&EvalCtx::new(), &value).unwrap();
        assert!(!has_lazy_state_value(&materialized));

        let Value::SetCup(cup) = &materialized else {
            panic!("materialized union changed shape");
        };
        assert!(matches!(cup.set1(), Value::Set(_)));
        assert!(matches!(cup.set2(), Value::Set(_)));
        let elements: Vec<_> = materialized.iter_set().unwrap().collect();
        assert_eq!(
            Value::set(elements),
            Value::set(vec![
                Value::SmallInt(1),
                Value::SmallInt(2),
                Value::SmallInt(3),
            ])
        );
    }

    #[test]
    fn test_materialize_lazy_set_wrapper_rejects_nested_closure() {
        let closure = Value::Closure(Rp::new(crate::value::ClosureValue::new(
            Vec::new(),
            Spanned::dummy(Expr::Bool(true)),
            Arc::new(HashMap::new()),
            None,
        )));
        let value = Value::SeqSet(crate::value::SeqSetValue::new(closure));

        assert!(has_lazy_state_value(&value));
        assert!(materialize_value(&EvalCtx::new(), &value).is_err());
    }
}
