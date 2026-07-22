// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Bytecode VM compound-value opcode handlers.
//!
//! Owns set operations, records, function operations, tuples, sequences,
//! strings, and cross-product types. Extracted from `execute.rs` per #3611.

use std::sync::Arc;
use tla_tir::bytecode::{BuiltinOp, ConstantPool, Opcode};
use tla_value::Rp;
use tla_value::{RecordValue, SortedSet, Value};

use super::execute::{BytecodeVm, VmError};
use super::execute_helpers::{
    func_apply, int_arith, load_prime_var, load_state_var, to_bigint, to_sorted_set, value_domain,
    value_except,
};

/// Test membership of a two-element tuple without allocating on concrete sets.
///
/// Lazy/compound set values receive the exact materialized tuple used by the
/// ordinary `TupleNew` then `SetIn` path, retaining membership and error
/// semantics.
#[inline]
pub(super) fn tuple2_set_contains(
    first: &Value,
    second: &Value,
    set: &Value,
) -> Result<bool, VmError> {
    // Stack storage only: the concrete-set path compares these elements
    // directly and never constructs an Arc-backed tuple.
    let tuple = [first.clone(), second.clone()];
    match set.try_set_contains_tuple_elements(&tuple) {
        Some(contains) => Ok(contains),
        None => {
            // Lazy/compound set representations still receive the exact
            // historical candidate value and membership dispatch.
            let candidate = Value::Tuple(tuple.into());
            set.set_contains(&candidate)
                .ok_or_else(|| VmError::TypeError {
                    expected: "enumerable set for \\in",
                    actual: format!("{set:?}"),
                })
        }
    }
}

/// Evaluate ordinary bytecode `Subseteq` semantics for two values.
#[inline]
fn subseteq_values(left: &Value, right: &Value) -> Result<bool, VmError> {
    let left = to_sorted_set(left)?;
    let right = to_sorted_set(right)?;
    Ok(left.is_subset(&right))
}

/// Test an enumerated left set without allocating it on concrete RHS sets.
///
/// Lazy and invalid RHS values take the exact historical materialization and
/// type-error path so this optimization changes neither accepted values nor
/// error text.
#[inline]
fn set_enum_subseteq_values(elements: &[Value], right: &Value) -> Result<bool, VmError> {
    // Keep the direct path bounded. `contains` is a linear raw scan for an
    // unnormalized set, so materialize and use the historical normalized
    // merge for larger manually-constructed opcode ranges.
    if elements.len() <= 2 {
        if let Value::Set(right) = right {
            return Ok(elements.iter().all(|element| right.contains(element)));
        }
    }

    let left = Value::Set(Rp::new(SortedSet::from_iter(elements.iter().cloned())));
    subseteq_values(&left, right)
}

/// Evaluate `value = <<value[1], value[2]>>` without constructing the tuple
/// when a direct tuple/sequence representation makes the verdict structural.
///
/// Lengths zero and one deliberately take the projection path: the historical
/// expression raises its first or second out-of-domain error before equality.
/// All other representations also run both projections left-to-right and then
/// use the allocation-free equivalent of comparing with the materialized tuple.
#[inline]
fn tuple2_self_eq_value(value: &Value) -> Result<bool, VmError> {
    match value {
        Value::Tuple(elements) if elements.len() >= 2 => Ok(elements.len() == 2),
        Value::Seq(elements) if elements.len() >= 2 => Ok(elements.len() == 2),
        _ => {
            let first = func_apply(value, &Value::SmallInt(1))?;
            let second = func_apply(value, &Value::SmallInt(2))?;
            Ok(value.equals_tuple_elements(&[first, second]))
        }
    }
}

/// Evaluate the complete certified Round body on one already-evaluated value.
#[inline]
fn round_apply_value(value: &Value) -> Result<Value, VmError> {
    if value.equals_empty_tuple() {
        Ok(Value::SmallInt(0))
    } else {
        func_apply(value, &Value::SmallInt(2))
    }
}

impl<'a> BytecodeVm<'a> {
    /// Execute the dynamic state read performed by an ordinary unprimed
    /// `LoadVar`. A caller's `SetPrimeMode` can redirect a precompiled callee
    /// to the next state, so VM-only fused state reads must use this exact path.
    #[inline]
    fn load_dynamic_state_var(&mut self, var_idx: u16) -> Result<Value, VmError> {
        if self.prime_mode {
            load_prime_var(self.next_state, self.next_state_cache.as_mut(), var_idx)
        } else {
            load_state_var(self.state, &mut self.state_cache, var_idx)
        }
    }

    /// Evaluate the exact tuple-self/subset conjunction while keeping the
    /// right-hand state slot strictly behind the tuple-shape guard.
    #[inline]
    fn tuple2_self_subseteq(&mut self, value: &Value, set_var_idx: u16) -> Result<bool, VmError> {
        match value {
            Value::Tuple(elements) if elements.len() >= 2 => {
                if elements.len() != 2 {
                    return Ok(false);
                }
                let right = self.load_dynamic_state_var(set_var_idx)?;
                return set_enum_subseteq_values(elements, &right);
            }
            Value::Seq(elements) if elements.len() >= 2 => {
                if elements.len() != 2 {
                    return Ok(false);
                }
                let projected = [
                    elements.get(0).expect("length checked above").clone(),
                    elements.get(1).expect("length checked above").clone(),
                ];
                let right = self.load_dynamic_state_var(set_var_idx)?;
                return set_enum_subseteq_values(&projected, &right);
            }
            _ => {}
        }

        // Preserve ordered projection errors for short direct values and all
        // non-direct function representations.
        let first = func_apply(value, &Value::SmallInt(1))?;
        let second = func_apply(value, &Value::SmallInt(2))?;
        let projected = [first, second];
        if !value.equals_tuple_elements(&projected) {
            return Ok(false);
        }
        let right = self.load_dynamic_state_var(set_var_idx)?;
        set_enum_subseteq_values(&projected, &right)
    }

    pub(super) fn execute_compound_opcode(
        &mut self,
        opcode: &Opcode,
        constants: &ConstantPool,
        regs: &mut [Value],
    ) -> Result<(), VmError> {
        match opcode {
            // === Set Operations ===
            Opcode::SetEnum { rd, start, count } => {
                let elements: Vec<Value> = (0..*count as usize)
                    .map(|i| regs[*start as usize + i].clone())
                    .collect();
                regs[*rd as usize] = Value::Set(Rp::new(SortedSet::from_iter(elements)));
            }
            Opcode::SetIn { rd, elem, set } => {
                let e = &regs[*elem as usize];
                let s = &regs[*set as usize];
                match s.set_contains(e) {
                    Some(b) => regs[*rd as usize] = Value::Bool(b),
                    None => {
                        return Err(VmError::TypeError {
                            expected: "enumerable set for \\in",
                            actual: format!("{s:?}"),
                        });
                    }
                }
            }
            Opcode::Tuple2SetIn {
                rd,
                first,
                second,
                set,
            } => {
                let contains = tuple2_set_contains(
                    &regs[*first as usize],
                    &regs[*second as usize],
                    &regs[*set as usize],
                )?;
                regs[*rd as usize] = Value::Bool(contains);
            }
            Opcode::SetEnumSubseteq {
                rd,
                start,
                count,
                set,
            } => {
                let start = *start as usize;
                let elements = &regs[start..start + *count as usize];
                let subset = set_enum_subseteq_values(elements, &regs[*set as usize])?;
                regs[*rd as usize] = Value::Bool(subset);
            }
            Opcode::Tuple2SelfEq { rd, value } => {
                let equal = tuple2_self_eq_value(&regs[*value as usize])?;
                regs[*rd as usize] = Value::Bool(equal);
            }
            Opcode::Tuple2SelfSubseteq {
                rd,
                value,
                set_var_idx,
            } => {
                let subset = self.tuple2_self_subseteq(&regs[*value as usize], *set_var_idx)?;
                regs[*rd as usize] = Value::Bool(subset);
            }
            Opcode::SetUnion { rd, r1, r2 } => {
                let a = to_sorted_set(&regs[*r1 as usize])?;
                let b = to_sorted_set(&regs[*r2 as usize])?;
                regs[*rd as usize] = Value::Set(Rp::new(a.union(&b)));
            }
            Opcode::SetIntersect { rd, r1, r2 } => {
                let a = to_sorted_set(&regs[*r1 as usize])?;
                let b = to_sorted_set(&regs[*r2 as usize])?;
                regs[*rd as usize] = Value::Set(Rp::new(a.intersection(&b)));
            }
            Opcode::SetDiff { rd, r1, r2 } => {
                // Fast path: both operands materialize to concrete sorted sets —
                // byte-identical to the historical eager behavior.
                match (
                    to_sorted_set(&regs[*r1 as usize]),
                    to_sorted_set(&regs[*r2 as usize]),
                ) {
                    (Ok(a), Ok(b)) => {
                        regs[*rd as usize] = Value::Set(Rp::new(a.difference(&b)));
                    }
                    _ => {
                        // A non-materializable operand — most importantly the
                        // infinite `Nat`/`Int` LHS in a `TypeOK`-style
                        // `x \in (Nat \ {0})`. Historically the eager
                        // `to_sorted_set` above returned a hard `TypeError`
                        // ("expected set, got @Nat"), which forced the whole
                        // enclosing invariant to abandon the VM and tree-walk
                        // EVERY state. Mirror the interpreter's lazy set algebra
                        // (`set_minus_values`) so `\ ` yields a lazy `SetDiff`
                        // whose membership is answered by `set_contains` — the
                        // VM result is then bit-identical to the interpreter's.
                        let a = regs[*r1 as usize].clone();
                        let b = regs[*r2 as usize].clone();
                        match self.eval_ctx {
                            Some(ctx) => {
                                regs[*rd as usize] =
                                    crate::eval_sets::set_minus_values(ctx, a, b, None, None, None)
                                        .map_err(VmError::Eval)?;
                            }
                            None => {
                                if self.is_action_execution() {
                                    return Err(VmError::NeedsEvalCtx(
                                        "non-materializable set difference",
                                    ));
                                }
                                // No caller ctx outside transactional action
                                // execution: keep the historical fail-closed
                                // TypeError used by isolated expression VMs.
                                return Err(VmError::TypeError {
                                    expected: "materializable set for \\",
                                    actual: format!("{:?}", regs[*r1 as usize]),
                                });
                            }
                        }
                    }
                }
            }
            Opcode::Subseteq { rd, r1, r2 } => {
                let subset = subseteq_values(&regs[*r1 as usize], &regs[*r2 as usize])?;
                regs[*rd as usize] = Value::Bool(subset);
            }
            Opcode::RoundStepEq { rd, child, parent } => {
                // Preserve the exact source order: the child Round call can
                // fail before the parent is touched.
                let child_round = round_apply_value(&regs[*child as usize])?;
                let parent_round = round_apply_value(&regs[*parent as usize])?;
                let parent_minus_one = int_arith(
                    &parent_round,
                    &Value::SmallInt(1),
                    |a, b| a.checked_sub(b),
                    |a, b| a - b,
                )?;
                let equal = self.equality_opcode_result(&child_round, &parent_minus_one)?;
                regs[*rd as usize] = Value::Bool(equal);
            }
            Opcode::Powerset { rd, rs } => {
                let base_val = &regs[*rs as usize];
                if !base_val.is_set() {
                    return Err(VmError::TypeError {
                        expected: "set for SUBSET base",
                        actual: format!("{base_val:?}"),
                    });
                }
                regs[*rd as usize] = Value::Subset(tla_value::SubsetValue::new(base_val.clone()));
            }
            Opcode::KSubset { rd, base, k } => {
                let base_val = &regs[*base as usize];
                if !base_val.is_set() {
                    return Err(VmError::TypeError {
                        expected: "set for KSubset base",
                        actual: format!("{base_val:?}"),
                    });
                }
                let k_val = to_bigint(&regs[*k as usize])?;
                use num_traits::ToPrimitive;
                let k_usize = k_val.to_usize().ok_or_else(|| VmError::TypeError {
                    expected: "non-negative integer for KSubset k",
                    actual: format!("{k_val}"),
                })?;
                regs[*rd as usize] =
                    Value::KSubset(tla_value::KSubsetValue::new(base_val.clone(), k_usize));
            }
            Opcode::BigUnion { rd, rs } => {
                let outer = to_sorted_set(&regs[*rs as usize])?;
                if outer.is_empty() {
                    regs[*rd as usize] = Value::empty_set();
                    return Ok(());
                }
                if let Some(elem) = outer.as_singleton() {
                    if !elem.is_set() {
                        return Err(VmError::TypeError {
                            expected: "set element in UNION",
                            actual: format!("{elem:?}"),
                        });
                    }
                    regs[*rd as usize] = elem.clone();
                    return Ok(());
                }

                let mut result = SortedSet::new();
                for elem in outer.iter() {
                    let inner = elem.to_sorted_set().ok_or_else(|| VmError::TypeError {
                        expected: "set element in UNION",
                        actual: format!("{elem:?}"),
                    })?;
                    result = result.union(&inner);
                }
                regs[*rd as usize] = Value::Set(Rp::new(result));
            }
            Opcode::Range { rd, lo, hi } => {
                let lo_val = to_bigint(&regs[*lo as usize])?;
                let hi_val = to_bigint(&regs[*hi as usize])?;
                regs[*rd as usize] = tla_value::range_set(&lo_val, &hi_val);
            }

            // === Records ===
            Opcode::RecordNew {
                rd,
                fields_start,
                values_start,
                count,
            } => {
                regs[*rd as usize] =
                    build_record_new(constants, regs, *fields_start, *values_start, *count)?;
            }
            Opcode::RecordGet { rd, rs, field_idx } => {
                let field_id = tla_core::NameId(constants.get_field_id(*field_idx));
                match &regs[*rs as usize] {
                    Value::Record(rec) => match rec.get_by_id(field_id) {
                        Some(v) => regs[*rd as usize] = v.clone(),
                        None => {
                            return Err(VmError::TypeError {
                                expected: "record field exists",
                                actual: format!(
                                    "field {:?} not found",
                                    tla_core::resolve_name_id(field_id)
                                ),
                            });
                        }
                    },
                    other => {
                        return Err(VmError::TypeError {
                            expected: "record",
                            actual: format!("{other:?}"),
                        });
                    }
                }
            }

            // === Function Operations ===
            Opcode::FuncApply { rd, func, arg } => {
                let f = &regs[*func as usize];
                let a = &regs[*arg as usize];
                regs[*rd as usize] = func_apply(f, a)?;
            }
            Opcode::Domain { rd, rs } => {
                regs[*rd as usize] = value_domain(&regs[*rs as usize])?;
            }
            Opcode::FuncExcept {
                rd,
                func,
                path,
                val,
            } => {
                let f = regs[*func as usize].clone();
                let p = regs[*path as usize].clone();
                let v = regs[*val as usize].clone();
                regs[*rd as usize] = value_except(f, p, v)?;
            }

            // === Tuples ===
            Opcode::TupleNew { rd, start, count } => {
                let elements: Vec<Value> = (0..*count as usize)
                    .map(|i| regs[*start as usize + i].clone())
                    .collect();
                regs[*rd as usize] = Value::tuple(elements);
            }
            Opcode::TupleGet { rd, rs, idx } => match &regs[*rs as usize] {
                Value::Tuple(elems) => {
                    let i = *idx as usize;
                    if i >= 1 && i <= elems.len() {
                        regs[*rd as usize] = elems[i - 1].clone();
                    } else {
                        return Err(VmError::TypeError {
                            expected: "valid tuple index",
                            actual: format!("index {i} out of bounds (len {})", elems.len()),
                        });
                    }
                }
                other => {
                    return Err(VmError::TypeError {
                        expected: "tuple",
                        actual: format!("{other:?}"),
                    });
                }
            },

            // === Sequences ===
            Opcode::SeqNew { rd, start, count } => {
                let elements: Vec<Value> = (0..*count as usize)
                    .map(|i| regs[*start as usize + i].clone())
                    .collect();
                regs[*rd as usize] = Value::seq(elements);
            }

            // === String ===
            Opcode::StrConcat { rd, r1, r2 } => match (&regs[*r1 as usize], &regs[*r2 as usize]) {
                (Value::String(a), Value::String(b)) => {
                    let mut s = a.to_string();
                    s.push_str(b);
                    regs[*rd as usize] = Value::string(s);
                }
                (a, b) => {
                    return Err(VmError::TypeError {
                        expected: "strings for concatenation",
                        actual: format!("{a:?} \\o {b:?}"),
                    });
                }
            },

            // === Not yet implemented / cross-product types ===
            Opcode::FuncDef { .. } => {
                return Err(VmError::Unsupported(
                    "FuncDef (non-loop variant)".to_string(),
                ));
            }
            Opcode::FuncSet { rd, domain, range } => {
                let d = regs[*domain as usize].clone();
                let r = regs[*range as usize].clone();
                regs[*rd as usize] = Value::FuncSet(tla_value::FuncSetValue::new(d, r));
            }
            Opcode::RecordSet {
                rd,
                fields_start,
                values_start,
                count,
            } => {
                let mut field_entries: Vec<(Arc<str>, Value)> = Vec::with_capacity(*count as usize);
                for i in 0..*count as usize {
                    let field_name = constants.get_value(*fields_start + i as u16);
                    let field_set = regs[*values_start as usize + i].clone();
                    let name_str: Arc<str> = match field_name {
                        Value::String(s) => s.clone().into(),
                        _ => {
                            return Err(VmError::TypeError {
                                expected: "string field name",
                                actual: format!("{field_name:?}"),
                            });
                        }
                    };
                    field_entries.push((name_str, field_set));
                }
                regs[*rd as usize] = Value::record_set(field_entries);
            }
            Opcode::Times { rd, start, count } => {
                let components: Vec<Value> = (0..*count as usize)
                    .map(|i| regs[*start as usize + i].clone())
                    .collect();
                regs[*rd as usize] = Value::tuple_set(components);
            }

            // === Closures ===
            Opcode::MakeClosure {
                rd,
                template_idx,
                captures_start,
                capture_count,
            } => {
                if self.is_action_execution() {
                    return Err(VmError::Unsupported(
                        "MakeClosure in action bytecode".to_string(),
                    ));
                }
                let template = constants.get_value(*template_idx);
                let closure_arc = match template {
                    Value::Closure(c) => c,
                    _ => {
                        return Err(VmError::TypeError {
                            expected: "closure template in MakeClosure",
                            actual: format!("{template:?}"),
                        });
                    }
                };
                // Build captured environment from consecutive constant-pool names
                // and consecutive register values.
                let mut env = tla_core::kani_types::HashMap::new();
                for i in 0..*capture_count as usize {
                    let name_val = constants.get_value(*template_idx + 1 + i as u16);
                    let name: Arc<str> = match name_val {
                        Value::String(s) => s.clone().into(),
                        _ => {
                            return Err(VmError::TypeError {
                                expected: "string capture name in MakeClosure",
                                actual: format!("{name_val:?}"),
                            });
                        }
                    };
                    let value = regs[*captures_start as usize + i].clone();
                    env.insert(name, value);
                }
                // Clone template and inject the captured env.
                let new_closure = closure_arc.as_ref().clone().with_env(Arc::new(env));
                regs[*rd as usize] = Value::Closure(Rp::new(new_closure));
            }

            // === Concat (polymorphic \o) ===
            Opcode::Concat { rd, r1, r2 } => {
                let v1 = &regs[*r1 as usize];
                let v2 = &regs[*r2 as usize];
                regs[*rd as usize] = execute_concat(v1, v2)?;
            }

            // === Standard-library builtin calls ===
            Opcode::CallBuiltin {
                rd,
                builtin,
                args_start,
                argc,
            } => {
                let args: &[Value] = &regs[*args_start as usize..][..*argc as usize];
                regs[*rd as usize] = execute_builtin(*builtin, args)?;
            }

            // === Fused Eq superinstructions (implied-action term compile) ===
            //
            // SEMANTIC CONTRACT (see the opcode docs): byte-identical to the
            // producer (FuncExcept / RecordNew) followed by Eq on its result.
            // The fast branches below decide the equality structurally on
            // same-representation shapes only — every element comparison uses
            // the same `Value` equality nested record/function comparison
            // would use. EVERY other shape (cross-representation pairs,
            // out-of-domain keys, non-function/-record operands, malformed
            // pools) falls back to literally constructing the intermediate
            // value and comparing via `equality_opcode_result`, reproducing
            // errors and verdicts exactly.
            Opcode::EqFuncExcept {
                rd,
                lhs,
                func,
                path,
                val,
            } => {
                let result = self.eq_func_except(
                    &regs[*lhs as usize],
                    &regs[*func as usize],
                    &regs[*path as usize],
                    &regs[*val as usize],
                )?;
                regs[*rd as usize] = Value::Bool(result);
            }
            Opcode::EqRecordNew {
                rd,
                lhs,
                fields_start,
                values_start,
                count,
            } => {
                let result = 'fast: {
                    if let Value::Record(g) = &regs[*lhs as usize] {
                        if g.len() != *count as usize {
                            // Duplicate constructor field names could make
                            // the built record's field count differ from
                            // `count`, so only the slow path may decide
                            // count-mismatched shapes exactly.
                            break 'fast None;
                        }
                        let mut result = Some(true);
                        for i in 0..*count as usize {
                            let Some(Value::String(name)) =
                                constants.try_get_value(*fields_start + i as u16)
                            else {
                                // Malformed pool: the constructor would
                                // error — slow path reproduces it.
                                result = None;
                                break;
                            };
                            // Duplicate constructor field names collapse in
                            // `from_entries`; only the slow path decides
                            // those shapes exactly.
                            let duplicate = (0..i).any(|j| {
                                matches!(
                                    constants.try_get_value(*fields_start + j as u16),
                                    Some(Value::String(prev)) if prev == name
                                )
                            });
                            if duplicate {
                                result = None;
                                break;
                            }
                            match g.get(name.as_ref()) {
                                Some(gv) if *gv == regs[*values_start as usize + i] => {}
                                _ => {
                                    result = Some(false);
                                    break;
                                }
                            }
                        }
                        break 'fast result;
                    }
                    None
                };
                let result = match result {
                    Some(b) => b,
                    None => {
                        // Slow path: construct + compare, byte-identical to
                        // the unfused RecordNew → Eq pair (including errors).
                        let tmp = build_record_new(
                            constants,
                            regs,
                            *fields_start,
                            *values_start,
                            *count,
                        )?;
                        self.equality_opcode_result(&regs[*lhs as usize], &tmp)?
                    }
                };
                regs[*rd as usize] = Value::Bool(result);
            }

            _ => unreachable!("non-compound opcode routed to execute_compound_opcode"),
        }

        Ok(())
    }

    /// Fused `lhs = [f EXCEPT ![k] = v]` (see `Opcode::EqFuncExcept`).
    ///
    /// Fast branches handle same-representation pairs with an in-domain key;
    /// everything else constructs the EXCEPT result and compares exactly like
    /// the unfused pair.
    fn eq_func_except(
        &self,
        lhs: &Value,
        f: &Value,
        k: &Value,
        v: &Value,
    ) -> Result<bool, VmError> {
        match (lhs, f) {
            (Value::IntFunc(g), Value::IntFunc(ff)) => {
                if let Value::SmallInt(ki) = k {
                    let (f_min, f_max) = (
                        tla_value::IntIntervalFunc::min(ff),
                        tla_value::IntIntervalFunc::max(ff),
                    );
                    if *ki >= f_min && *ki <= f_max {
                        // In-domain key: the EXCEPT result has ff's exact
                        // domain and values, with position `ki - min`
                        // replaced by `v`. Equality against `g` (same
                        // representation) is decided field-wise with plain
                        // `Value` equality — identical to `eq_same_type`'s
                        // IntFunc arm on the constructed value.
                        if tla_value::IntIntervalFunc::min(g) != f_min
                            || tla_value::IntIntervalFunc::max(g) != f_max
                        {
                            return Ok(false);
                        }
                        let pos = (*ki - f_min) as usize;
                        let gv = g.values();
                        let fv = ff.values();
                        debug_assert_eq!(gv.len(), fv.len());
                        if gv.as_ptr() == fv.as_ptr() {
                            // Same backing buffer: all non-`pos` positions
                            // trivially equal.
                            return Ok(gv[pos] == *v);
                        }
                        for i in 0..fv.len() {
                            let expected = if i == pos { v } else { &fv[i] };
                            if gv[i] != *expected {
                                return Ok(false);
                            }
                        }
                        return Ok(true);
                    }
                }
            }
            (Value::Func(g), Value::Func(ff)) => {
                // Same-representation general functions: the EXCEPT result
                // shares ff's domain when the key is found. Walk both
                // mappings in canonical (sorted) order; require identical
                // key sequences, values equal except at the key position.
                // Key not found in ff's domain → slow path (the EXCEPT
                // no-op/error semantics stay with the real constructor).
                if g.domain_len() == ff.domain_len() {
                    let mut key_seen = false;
                    let mut equal = true;
                    for ((gk, gv), (fk, fv)) in g.mapping_iter().zip(ff.mapping_iter()) {
                        if gk != fk {
                            equal = false;
                            break;
                        }
                        let is_key = !key_seen && fk == k;
                        if is_key {
                            key_seen = true;
                            if gv != v {
                                equal = false;
                                break;
                            }
                        } else if gv != fv {
                            equal = false;
                            break;
                        }
                    }
                    if key_seen {
                        return Ok(equal);
                    }
                    if equal && !key_seen {
                        // Domains matched pairwise but the key is not in the
                        // domain: EXCEPT's out-of-domain semantics belong to
                        // the constructor — fall through to the slow path.
                    } else if !equal && !key_seen {
                        // Key might appear after the mismatch position; only
                        // the constructor can decide exactly. Fall through.
                    }
                }
            }
            _ => {}
        }
        // Slow path: literal construct + compare — byte-identical to the
        // unfused FuncExcept → Eq pair (including construction errors and
        // cross-representation equality semantics).
        let tmp = value_except(f.clone(), k.clone(), v.clone())?;
        self.equality_opcode_result(lhs, &tmp)
    }
}

/// Build a record value for `RecordNew` (shared by the plain opcode and the
/// `EqRecordNew` slow path). `from_entries` sorts into canonical record
/// field order (field-name string), NOT NameId interning order.
fn build_record_new(
    constants: &ConstantPool,
    regs: &[Value],
    fields_start: u16,
    values_start: u8,
    count: u8,
) -> Result<Value, VmError> {
    let mut entries = Vec::with_capacity(count as usize);
    for i in 0..count as usize {
        let field_name = constants.get_value(fields_start + i as u16);
        let value = regs[values_start as usize + i].clone();
        let name_str = match field_name {
            Value::String(s) => s.as_ref(),
            _ => {
                return Err(VmError::TypeError {
                    expected: "string field name in record",
                    actual: format!("{field_name:?}"),
                });
            }
        };
        entries.push((tla_core::intern_name(name_str), value));
    }
    Ok(Value::Record(RecordValue::from_entries(entries)))
}

/// Execute the polymorphic `\o` (concat) operator on two values.
///
/// Handles string-string concatenation, sequence-sequence concatenation,
/// and sequence-like values (Tuple, IntFunc, Func with 1..n domains).
/// Part of #3789.
fn execute_concat(v1: &Value, v2: &Value) -> Result<Value, VmError> {
    // String concatenation
    if let (Value::String(s1), Value::String(s2)) = (v1, v2) {
        let mut s = s1.to_string();
        s.push_str(s2);
        return Ok(Value::String(crate::value::intern_string(&s)));
    }
    // Sequence concatenation (accept Seq, Tuple, and tuple-like Func/IntFunc)
    let s1 = v1
        .as_seq()
        .or_else(|| v1.to_tuple_like_elements())
        .ok_or_else(|| VmError::TypeError {
            expected: "sequence or string for \\o",
            actual: format!("{v1:?}"),
        })?;
    let s2 = v2
        .as_seq()
        .or_else(|| v2.to_tuple_like_elements())
        .ok_or_else(|| VmError::TypeError {
            expected: "sequence or string for \\o",
            actual: format!("{v2:?}"),
        })?;
    let mut result: Vec<Value> = Vec::with_capacity(s1.len() + s2.len());
    result.extend(s1.iter().cloned());
    result.extend(s2.iter().cloned());
    Ok(Value::Seq(Rp::new(result.into())))
}

/// Execute a standard-library builtin operator on already-evaluated arguments.
///
/// This is the value-level equivalent of `eval_builtin_sequences` /
/// `eval_builtin_finite_sets` / `eval_builtin_tlc` — but operates on `Value`
/// directly without needing `EvalCtx` or `Expr`.
/// Part of #3789: cross-module stdlib operator support in bytecode VM.
fn execute_builtin(op: BuiltinOp, args: &[Value]) -> Result<Value, VmError> {
    match op {
        BuiltinOp::RoundApply => round_apply_value(&args[0]),
        BuiltinOp::Len => {
            let v = &args[0];
            match v {
                Value::Seq(s) => Ok(Value::int(s.len() as i64)),
                Value::Tuple(s) => Ok(Value::int(s.len() as i64)),
                Value::String(s) => Ok(Value::int(crate::value::tlc_string_len(s.as_ref()) as i64)),
                Value::IntFunc(f) => {
                    if tla_value::IntIntervalFunc::min(f) == 1 {
                        Ok(Value::int(f.len() as i64))
                    } else {
                        Err(VmError::TypeError {
                            expected: "sequence for Len",
                            actual: format!("{v:?}"),
                        })
                    }
                }
                Value::Func(f) => {
                    // Sequences are functions with domain 1..n
                    let mut expected: i64 = 1;
                    for key in f.domain_iter() {
                        let Some(k) = key.as_i64() else {
                            return Err(VmError::TypeError {
                                expected: "sequence for Len",
                                actual: format!("{v:?}"),
                            });
                        };
                        if k != expected {
                            return Err(VmError::TypeError {
                                expected: "sequence for Len",
                                actual: format!("{v:?}"),
                            });
                        }
                        expected += 1;
                    }
                    Ok(Value::int(expected - 1))
                }
                _ => Err(VmError::TypeError {
                    expected: "sequence for Len",
                    actual: format!("{v:?}"),
                }),
            }
        }

        BuiltinOp::Head => {
            let v = &args[0];
            let seq = v
                .as_seq()
                .or_else(|| v.to_tuple_like_elements())
                .ok_or_else(|| VmError::TypeError {
                    expected: "sequence for Head",
                    actual: format!("{v:?}"),
                })?;
            seq.first().cloned().ok_or({
                VmError::Eval(tla_value::error::EvalError::ApplyEmptySeq {
                    op: "Head",
                    span: None,
                })
            })
        }

        BuiltinOp::Tail => {
            let v = &args[0];
            // Fast path: use O(log n) tail for SeqValue
            if let Some(seq_value) = v.as_seq_value() {
                if seq_value.is_empty() {
                    return Err(VmError::Eval(tla_value::error::EvalError::ApplyEmptySeq {
                        op: "Tail",
                        span: None,
                    }));
                }
                return Ok(Value::Seq(Rp::new(seq_value.tail())));
            }
            let seq = v
                .as_seq()
                .or_else(|| v.to_tuple_like_elements())
                .ok_or_else(|| VmError::TypeError {
                    expected: "sequence for Tail",
                    actual: format!("{v:?}"),
                })?;
            if seq.is_empty() {
                return Err(VmError::Eval(tla_value::error::EvalError::ApplyEmptySeq {
                    op: "Tail",
                    span: None,
                }));
            }
            Ok(Value::Seq(Rp::new(seq[1..].to_vec().into())))
        }

        BuiltinOp::Append => {
            let sv = &args[0];
            let elem = args[1].clone();
            if let Some(seq_value) = sv.as_seq_value() {
                return Ok(Value::Seq(Rp::new(seq_value.append(elem))));
            }
            let s = sv
                .as_seq()
                .or_else(|| sv.to_tuple_like_elements())
                .ok_or_else(|| VmError::TypeError {
                    expected: "sequence for Append",
                    actual: format!("{sv:?}"),
                })?;
            let mut v = s.to_vec();
            v.push(elem);
            Ok(Value::Seq(Rp::new(v.into())))
        }

        BuiltinOp::SubSeq => {
            let sv = &args[0];
            let m = args[1].as_i64().ok_or_else(|| VmError::TypeError {
                expected: "integer for SubSeq start",
                actual: format!("{:?}", args[1]),
            })?;
            let n = args[2].as_i64().ok_or_else(|| VmError::TypeError {
                expected: "integer for SubSeq end",
                actual: format!("{:?}", args[2]),
            })?;
            if m > n {
                return match sv {
                    Value::String(_) => Ok(Value::String(crate::value::intern_string(""))),
                    _ => Ok(Value::Seq(Rp::new(Vec::new().into()))),
                };
            }
            match sv {
                Value::String(s) => {
                    let len = crate::value::tlc_string_len(s.as_ref());
                    if m < 1 || (m as usize) > len || n < 1 || (n as usize) > len {
                        return Err(VmError::Eval(
                            tla_value::error::EvalError::IndexOutOfBounds {
                                index: if m < 1 || (m as usize) > len { m } else { n },
                                len,
                                value_display: None,
                                span: None,
                            },
                        ));
                    }
                    let start_off = (m - 1) as usize;
                    let end_off = n as usize;
                    let substr = crate::value::tlc_string_subseq_utf16_offsets(
                        s.as_ref(),
                        start_off..end_off,
                    );
                    Ok(Value::String(crate::value::intern_string(substr.as_ref())))
                }
                Value::Seq(seq_value) => {
                    let len = seq_value.len();
                    if m < 1 || (m as usize) > len || n < 1 || (n as usize) > len {
                        return Err(VmError::Eval(
                            tla_value::error::EvalError::IndexOutOfBounds {
                                index: if m < 1 || (m as usize) > len { m } else { n },
                                len,
                                value_display: None,
                                span: None,
                            },
                        ));
                    }
                    let start = (m - 1) as usize;
                    let end = n as usize;
                    Ok(Value::Seq(Rp::new(seq_value.subseq(start, end))))
                }
                Value::Tuple(seq) => {
                    let len = seq.len();
                    if m < 1 || (m as usize) > len || n < 1 || (n as usize) > len {
                        return Err(VmError::Eval(
                            tla_value::error::EvalError::IndexOutOfBounds {
                                index: if m < 1 || (m as usize) > len { m } else { n },
                                len,
                                value_display: None,
                                span: None,
                            },
                        ));
                    }
                    let start = (m - 1) as usize;
                    let end = n as usize;
                    Ok(Value::Seq(Rp::new(seq[start..end].to_vec().into())))
                }
                _ => Err(VmError::TypeError {
                    expected: "sequence or string for SubSeq",
                    actual: format!("{sv:?}"),
                }),
            }
        }

        BuiltinOp::RemoveAt => {
            // RemoveAt(s, i) — remove the element at 1-indexed position i.
            // Mirrors crate::builtin_sequences_ext_ops::mutation "RemoveAt".
            let sv = &args[0];
            let seq = sv
                .as_seq()
                .or_else(|| sv.to_tuple_like_elements())
                .ok_or_else(|| VmError::TypeError {
                    expected: "sequence for RemoveAt",
                    actual: format!("{sv:?}"),
                })?;
            let i = args[1].as_i64().ok_or_else(|| VmError::TypeError {
                expected: "integer for RemoveAt index",
                actual: format!("{:?}", args[1]),
            })?;
            let len = seq.len();
            if i < 1 || i > len as i64 {
                return Err(VmError::Eval(
                    tla_value::error::EvalError::IndexOutOfBounds {
                        index: i,
                        len,
                        value_display: None,
                        span: None,
                    },
                ));
            }
            let idx = (i - 1) as usize;
            let mut out: Vec<Value> = seq.to_vec();
            out.remove(idx);
            Ok(Value::Seq(Rp::new(out.into())))
        }

        BuiltinOp::Seq => {
            // Seq(S) = set of all finite sequences over S
            let base = args[0].clone();
            Ok(Value::SeqSet(tla_value::SeqSetValue::new(base)))
        }

        BuiltinOp::Cardinality => {
            let v = &args[0];
            match v.set_len() {
                Some(n) => Ok(Value::big_int(n)),
                None if v.is_set() => Err(VmError::Unsupported(
                    "Cardinality not supported for this set value".to_string(),
                )),
                None => Err(VmError::TypeError {
                    expected: "set for Cardinality",
                    actual: format!("{v:?}"),
                }),
            }
        }

        BuiltinOp::IsFiniteSet => {
            let v = &args[0];
            Ok(Value::Bool(v.is_finite_set()))
        }

        BuiltinOp::FoldFunctionOnSetSum => {
            if args.len() != 2 {
                return Err(VmError::TypeError {
                    expected: "function and set for FoldFunctionOnSet(+, 0, f, S)",
                    actual: format!("{} arguments", args.len()),
                });
            }
            let f = &args[0];
            let s = &args[1];
            let mut sum = num_bigint::BigInt::from(0);
            let iter = s.iter_set().ok_or_else(|| VmError::TypeError {
                expected: "finite set for FoldFunctionOnSet(+, 0, f, S)",
                actual: format!("{s:?}"),
            })?;
            for elem in iter {
                let value = func_apply(f, &elem)?;
                sum += to_bigint(&value)?;
            }
            Ok(Value::big_int(sum))
        }

        BuiltinOp::ToString => {
            let v = &args[0];
            Ok(Value::String(crate::value::intern_string(&format!("{v}"))))
        }

        BuiltinOp::Range => {
            // Range(f) — the set of values in the function's mapping (co-domain image),
            // i.e. `{ f[x] : x \in DOMAIN f }`. Mirrors the tree-walking interpreter in
            // `builtin_stdlib_ext.rs::eval_builtin_stdlib_ext` ("Range" arm): collect the
            // mapping values for functions/sequences/tuples, then dedup via `Value::set`.
            let fv = &args[0];
            let values: Vec<Value> = match fv {
                Value::Func(func) => func.mapping_values().cloned().collect(),
                // Compact bag: the range is the set of counts.
                Value::Bag(b) => b.counts().iter().map(|&c| Value::SmallInt(c)).collect(),
                Value::IntFunc(func) => func.values().to_vec(),
                Value::Seq(seq) => seq.iter().cloned().collect(),
                Value::Tuple(seq) => seq.iter().cloned().collect(),
                _ => {
                    return Err(VmError::TypeError {
                        expected: "Function/Seq for Range",
                        actual: format!("{fv:?}"),
                    });
                }
            };
            Ok(Value::set(values))
        }

        // Bags / BagsExt: shared value-level implementations with the AST
        // interpreter arms (builtin_bags/builtin_bagsext), so VM results are
        // value- and fingerprint-identical to interpreter results by
        // construction. Type failures propagate as VmError (fail-closed: the
        // caller falls back to interpreter evaluation).
        BuiltinOp::BagAdd => crate::builtin_bagsext::bag_add_value(&args[0], &args[1])
            .map_err(|e| bag_op_vm_error(e, "BagAdd")),
        BuiltinOp::BagRemove => crate::builtin_bagsext::bag_remove_value(&args[0], &args[1])
            .map_err(|e| bag_op_vm_error(e, "BagRemove")),
        BuiltinOp::SetToBag => {
            let s = &args[0];
            let iter = s.iter_set().ok_or_else(|| VmError::TypeError {
                expected: "finite set for SetToBag",
                actual: format!("{s:?}"),
            })?;
            // Order-insensitive: set_to_bag_from_elems sorts by Value::cmp,
            // exactly like the interpreter arm.
            Ok(crate::builtin_bags::set_to_bag_from_elems(iter))
        }
    }
}

/// Map a shared bag-op failure to the VM's error type (fail-closed).
fn bag_op_vm_error(e: crate::builtin_bagsext::BagOpError, op: &'static str) -> VmError {
    match e {
        crate::builtin_bagsext::BagOpError::NotBag(v) => VmError::TypeError {
            expected: "Bag/Function",
            actual: format!("{op}: {v:?}"),
        },
        crate::builtin_bagsext::BagOpError::NotInt(v) => VmError::TypeError {
            expected: "Int count",
            actual: format!("{op}: {v:?}"),
        },
    }
}
