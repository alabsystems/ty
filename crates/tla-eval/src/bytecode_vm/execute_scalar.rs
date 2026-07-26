// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Bytecode VM scalar/control opcode handlers.
//!
//! Owns value loading, integer arithmetic, comparisons, booleans,
//! control flow, and special opcodes. Extracted from `execute.rs` per #3611.

use num_bigint::BigInt;
use num_traits::Signed;
use tla_tir::bytecode::{ConstantPool, Opcode};
use tla_value::error::EvalError;
use tla_value::Rp;
use tla_value::Value;

use super::execute::{BytecodeVm, VmError};
use super::execute_dispatch::DispatchOutcome;
use super::execute_helpers::{
    as_bool, call_external, enter_vm_call, exit_vm_call, int_arith, int_cmp, int_div,
    int_exact_div, int_pow, load_prime_var, load_state_var, value_apply,
};

impl<'a> BytecodeVm<'a> {
    pub(super) fn equality_opcode_result(&self, lhs: &Value, rhs: &Value) -> Result<bool, VmError> {
        if !lhs.is_set() && !rhs.is_set() {
            return Ok(lhs == rhs);
        }
        if lhs.is_set() != rhs.is_set() {
            return Ok(false);
        }
        if let Some(eval_ctx) = self.eval_ctx {
            return crate::values_equal(eval_ctx, lhs, rhs, None).map_err(VmError::Eval);
        }
        if let (Value::Set(lhs_set), Value::Set(rhs_set)) = (lhs, rhs) {
            if lhs_set.iter().all(|value| !value.is_set())
                && rhs_set.iter().all(|value| !value.is_set())
            {
                return Ok(lhs_set == rhs_set);
            }
        }
        if self.is_action_execution() {
            Err(VmError::NeedsEvalCtx("extensional set equality"))
        } else {
            Err(VmError::Unsupported(
                "set equality without EvalCtx".to_string(),
            ))
        }
    }

    pub(super) fn execute_scalar_opcode(
        &mut self,
        opcode: &Opcode,
        constants: &ConstantPool,
        regs: &mut [Value],
        pc: usize,
    ) -> Result<DispatchOutcome, VmError> {
        match opcode {
            // === Value Loading ===
            Opcode::LoadImm { rd, value } => {
                regs[*rd as usize] = Value::SmallInt(*value);
            }
            Opcode::LoadBool { rd, value } => {
                regs[*rd as usize] = Value::Bool(*value);
            }
            Opcode::LoadConst { rd, idx } => {
                regs[*rd as usize] = constants.get_value(*idx).clone();
            }
            Opcode::LoadVar { rd, var_idx } => {
                regs[*rd as usize] = if self.prime_mode {
                    load_prime_var(self.next_state, self.next_state_cache.as_mut(), *var_idx)?
                } else {
                    load_state_var(self.state, &mut self.state_cache, *var_idx)?
                };
            }
            Opcode::LoadPrime { rd, var_idx } => {
                if self.is_action_execution() {
                    let Some(action_state) = self.action_state.as_ref() else {
                        return Err(VmError::Unsupported(format!(
                            "unbound primed variable {var_idx} in action bytecode"
                        )));
                    };
                    regs[*rd as usize] = match action_state.bound_overlay_value(*var_idx)?.cloned()
                    {
                        Some(value) => value,
                        None => load_state_var(self.state, &mut self.state_cache, *var_idx)?,
                    };
                } else {
                    regs[*rd as usize] =
                        load_prime_var(self.next_state, self.next_state_cache.as_mut(), *var_idx)?;
                }
            }
            Opcode::StoreVar { var_idx, rs } => {
                if !self.is_action_execution() {
                    return Err(VmError::Unsupported(
                        "StoreVar (invariant-only path)".to_string(),
                    ));
                }
                let action_state = self
                    .action_state
                    .get_or_insert_with(|| Box::new(super::execute::ActionVmState::default()));
                action_state.store(*var_idx, regs[*rs as usize].clone(), self.state.env_len())?;
            }
            Opcode::Move { rd, rs } => {
                regs[*rd as usize] = regs[*rs as usize].clone();
            }

            // === Integer Arithmetic ===
            Opcode::AddInt { rd, r1, r2 } => {
                regs[*rd as usize] = int_arith(
                    &regs[*r1 as usize],
                    &regs[*r2 as usize],
                    |a, b| a.checked_add(b),
                    |a, b| a + b,
                )?;
            }
            Opcode::SubInt { rd, r1, r2 } => {
                regs[*rd as usize] = int_arith(
                    &regs[*r1 as usize],
                    &regs[*r2 as usize],
                    |a, b| a.checked_sub(b),
                    |a, b| a - b,
                )?;
            }
            Opcode::MulInt { rd, r1, r2 } => {
                regs[*rd as usize] = int_arith(
                    &regs[*r1 as usize],
                    &regs[*r2 as usize],
                    |a, b| a.checked_mul(b),
                    |a, b| a * b,
                )?;
            }
            Opcode::DivInt { rd, r1, r2 } => {
                // `/` is TLA+ real division: EXACT-OR-ERROR on integers.
                regs[*rd as usize] = int_exact_div(&regs[*r1 as usize], &regs[*r2 as usize])?;
            }
            Opcode::IntDiv { rd, r1, r2 } => {
                regs[*rd as usize] = int_div(
                    &regs[*r1 as usize],
                    &regs[*r2 as usize],
                    |a, b| {
                        if b == 0 {
                            return None;
                        }
                        // i64::MIN / -1 overflows (panics even in release; Rust's
                        // division-overflow check is not gated by overflow-checks).
                        // Decline the SmallInt fast path so int_arith falls through
                        // to the BigInt big_op, which yields the correct 2^63.
                        if a == i64::MIN && b == -1 {
                            return None;
                        }
                        let q = a / b;
                        if (a ^ b) < 0 && a % b != 0 {
                            Some(q - 1)
                        } else {
                            Some(q)
                        }
                    },
                    |a, b| {
                        let q = &a / &b;
                        if (a.sign() != b.sign()) && (&a % &b) != BigInt::from(0) {
                            q - 1
                        } else {
                            q
                        }
                    },
                )?;
            }
            Opcode::ModInt { rd, r1, r2 } => {
                // TLC/TY semantics require a STRICTLY POSITIVE modulus divisor.
                // The AST/TIR tree-walker (int_mod_op) raises ModulusNotPositive
                // when the divisor is <= 0; mirror it here so the default-on
                // bytecode VM matches the interpreter (cross-engine parity) and so
                // the i64::MIN % -1 overflow path is never reached.
                let divisor = &regs[*r2 as usize];
                let nonpositive = match divisor {
                    Value::SmallInt(b) => *b <= 0,
                    Value::Int(n) => n.as_ref() <= &BigInt::from(0),
                    _ => false,
                };
                if nonpositive {
                    return Err(VmError::Eval(EvalError::ModulusNotPositive {
                        value: match divisor {
                            Value::SmallInt(b) => b.to_string(),
                            Value::Int(n) => n.as_ref().to_string(),
                            other => format!("{other:?}"),
                        },
                        span: None,
                    }));
                }
                regs[*rd as usize] = int_div(
                    &regs[*r1 as usize],
                    &regs[*r2 as usize],
                    |a, b| {
                        if b == 0 {
                            return None;
                        }
                        let r = a % b;
                        Some(if r < 0 { r + b.abs() } else { r })
                    },
                    |a, b| {
                        let r = &a % &b;
                        if r < BigInt::from(0) {
                            r + b.abs()
                        } else {
                            r
                        }
                    },
                )?;
            }
            Opcode::NegInt { rd, rs } => {
                regs[*rd as usize] = match &regs[*rs as usize] {
                    Value::SmallInt(n) => match n.checked_neg() {
                        Some(neg) => Value::SmallInt(neg),
                        None => Value::big_int(-BigInt::from(*n)),
                    },
                    Value::Int(n) => Value::big_int(-n.as_ref().clone()),
                    other => {
                        return Err(VmError::TypeError {
                            expected: "integer",
                            actual: format!("{other:?}"),
                        });
                    }
                };
            }
            Opcode::PowInt { rd, r1, r2 } => {
                regs[*rd as usize] = int_pow(&regs[*r1 as usize], &regs[*r2 as usize])?;
            }

            // === Comparison ===
            Opcode::Eq { rd, r1, r2 } => {
                let result =
                    self.equality_opcode_result(&regs[*r1 as usize], &regs[*r2 as usize])?;
                regs[*rd as usize] = Value::Bool(result);
            }
            Opcode::Neq { rd, r1, r2 } => {
                let result =
                    self.equality_opcode_result(&regs[*r1 as usize], &regs[*r2 as usize])?;
                regs[*rd as usize] = Value::Bool(!result);
            }
            Opcode::LtInt { rd, r1, r2 } => {
                regs[*rd as usize] = int_cmp(
                    &regs[*r1 as usize],
                    &regs[*r2 as usize],
                    |a, b| a < b,
                    |a, b| a < b,
                )?;
            }
            Opcode::LeInt { rd, r1, r2 } => {
                regs[*rd as usize] = int_cmp(
                    &regs[*r1 as usize],
                    &regs[*r2 as usize],
                    |a, b| a <= b,
                    |a, b| a <= b,
                )?;
            }
            Opcode::GtInt { rd, r1, r2 } => {
                regs[*rd as usize] = int_cmp(
                    &regs[*r1 as usize],
                    &regs[*r2 as usize],
                    |a, b| a > b,
                    |a, b| a > b,
                )?;
            }
            Opcode::GeInt { rd, r1, r2 } => {
                regs[*rd as usize] = int_cmp(
                    &regs[*r1 as usize],
                    &regs[*r2 as usize],
                    |a, b| a >= b,
                    |a, b| a >= b,
                )?;
            }

            // === Boolean Operations ===
            Opcode::And { rd, r1, r2 } => {
                let a = as_bool(&regs[*r1 as usize])?;
                let b = as_bool(&regs[*r2 as usize])?;
                regs[*rd as usize] = Value::Bool(a && b);
            }
            Opcode::Or { rd, r1, r2 } => {
                let a = as_bool(&regs[*r1 as usize])?;
                let b = as_bool(&regs[*r2 as usize])?;
                regs[*rd as usize] = Value::Bool(a || b);
            }
            Opcode::Not { rd, rs } => {
                regs[*rd as usize] = Value::Bool(!as_bool(&regs[*rs as usize])?);
            }
            Opcode::Implies { rd, r1, r2 } => {
                let a = as_bool(&regs[*r1 as usize])?;
                let b = as_bool(&regs[*r2 as usize])?;
                regs[*rd as usize] = Value::Bool(!a || b);
            }
            Opcode::Equiv { rd, r1, r2 } => {
                let a = as_bool(&regs[*r1 as usize])?;
                let b = as_bool(&regs[*r2 as usize])?;
                regs[*rd as usize] = Value::Bool(a == b);
            }

            // === Control Flow ===
            Opcode::Jump { offset } => {
                return Ok(DispatchOutcome::Jump(
                    ((pc as i64) + (*offset as i64)) as usize,
                ));
            }
            Opcode::JumpTrue { rs, offset } => {
                if as_bool(&regs[*rs as usize])? {
                    return Ok(DispatchOutcome::Jump(
                        ((pc as i64) + (*offset as i64)) as usize,
                    ));
                }
            }
            Opcode::JumpFalse { rs, offset } => {
                if !as_bool(&regs[*rs as usize])? {
                    return Ok(DispatchOutcome::Jump(
                        ((pc as i64) + (*offset as i64)) as usize,
                    ));
                }
            }
            Opcode::Call {
                rd,
                op_idx,
                args_start,
                argc,
            } => {
                let callee = self.chunk.get_function(*op_idx);
                if callee.arity != *argc {
                    return Err(VmError::TypeError {
                        expected: "matching arity",
                        actual: format!(
                            "operator {} expects {} args, got {}",
                            callee.name, callee.arity, argc
                        ),
                    });
                }
                enter_vm_call()?;
                // Lever 4 (#EWD998PCal): pooled callee registers — nested
                // calls fire many times per VM execution on the implied-action
                // hot path. The buffer is cleared + fully re-initialized here,
                // so behavior is identical to a fresh allocation.
                let mut callee_regs = super::execute::acquire_regs_buf();
                callee_regs.clear();
                callee_regs.resize((callee.max_register as usize) + 1, Value::Bool(false));
                for i in 0..(*argc as usize) {
                    callee_regs[i] = regs[*args_start as usize + i].clone();
                }
                let result =
                    self.execute_with_regs(callee, &self.chunk.constants, &mut callee_regs);
                exit_vm_call();
                super::execute::release_regs_buf(callee_regs);
                regs[*rd as usize] = result?;
            }
            Opcode::ValueApply {
                rd,
                func,
                args_start,
                argc,
            } => {
                if self.is_action_execution() {
                    return Err(VmError::Unsupported(
                        "ValueApply in action bytecode".to_string(),
                    ));
                }
                // Part of #3697: If the callable is a closure with a compiled
                // bytecode function, execute it via Call instead of tree-walking.
                if let Value::Closure(closure) = &regs[*func as usize] {
                    if let Some(bc_idx) = closure.bytecode_func_idx() {
                        if (bc_idx as usize) < self.chunk.functions.len() {
                            let callee = self.chunk.get_function(bc_idx);
                            // Collect capture values from the closure env to pass
                            // as extra arguments after the real args. Sort by key
                            // so the order matches the compiler's alphabetical
                            // capture parameter order (HashMap iteration is unordered).
                            // Part of #3697: Both sides must agree on canonical order.
                            let mut capture_entries: Vec<_> = closure
                                .env()
                                .iter()
                                .map(|(k, v)| (k.clone(), v.clone()))
                                .collect();
                            capture_entries.sort_by(|(a, _), (b, _)| a.cmp(b));
                            let capture_values: Vec<Value> =
                                capture_entries.into_iter().map(|(_, v)| v).collect();
                            let total_argc = *argc as usize + capture_values.len();
                            if callee.arity as usize == total_argc {
                                enter_vm_call()?;
                                // Lever 4 (#EWD998PCal): pooled callee
                                // registers (cleared + fully re-initialized —
                                // identical behavior to a fresh allocation).
                                let mut callee_regs = super::execute::acquire_regs_buf();
                                callee_regs.clear();
                                callee_regs
                                    .resize((callee.max_register as usize) + 1, Value::Bool(false));
                                // Copy real arguments.
                                for i in 0..(*argc as usize) {
                                    callee_regs[i] = regs[*args_start as usize + i].clone();
                                }
                                // Copy capture values as extra parameters.
                                for (i, v) in capture_values.into_iter().enumerate() {
                                    callee_regs[*argc as usize + i] = v;
                                }
                                let result = self.execute_with_regs(
                                    callee,
                                    &self.chunk.constants,
                                    &mut callee_regs,
                                );
                                exit_vm_call();
                                super::execute::release_regs_buf(callee_regs);
                                regs[*rd as usize] = result?;
                                return Ok(DispatchOutcome::Continue);
                            }
                        }
                    }
                }
                let args: Vec<Value> = (0..(*argc as usize))
                    .map(|i| regs[*args_start as usize + i].clone())
                    .collect();
                regs[*rd as usize] = value_apply(self.eval_ctx, &regs[*func as usize], &args)?;
            }
            Opcode::Ret { rs } => {
                return Ok(DispatchOutcome::Return(regs[*rs as usize].clone()));
            }

            // === External Call (INSTANCE-imported operator fallback) ===
            Opcode::CallExternal {
                rd,
                name_idx,
                args_start,
                argc,
                self_recursive: _,
            } => {
                if self.is_action_execution() {
                    return Err(VmError::Unsupported(
                        "CallExternal in action bytecode".to_string(),
                    ));
                }
                let name = match constants.get_value(*name_idx) {
                    Value::String(s) => s.clone(),
                    other => {
                        return Err(VmError::TypeError {
                            expected: "string operator name in constant pool",
                            actual: format!("{other:?}"),
                        });
                    }
                };
                // Per-execution memo for zero-arg externals (implied-action
                // fast path): the state binding is fixed for the duration of
                // one execution, so repeated references to the same pinned
                // state function (e.g. `token` / `token'`) reuse the value
                // instead of re-probing the interpreter caches. Keyed by the
                // name VALUE (the pool does not deduplicate constants); the
                // pool's `Value::String` is Arc'd — ptr-eq fast path first.
                if *argc == 0 && self.zero_arg_external_memo.is_some() {
                    let key = constants.get_value(*name_idx);
                    if let Some(memo) = self.zero_arg_external_memo.as_ref() {
                        if let Some((_, _, value)) = memo.iter().find(|(n, prime, _)| {
                            *prime == self.prime_mode
                                && match (n, key) {
                                    (Value::String(a), Value::String(b)) => {
                                        Rp::ptr_eq(a, b) || a == b
                                    }
                                    _ => false,
                                }
                        }) {
                            regs[*rd as usize] = value.clone();
                            return Ok(DispatchOutcome::Continue);
                        }
                    }
                    // Caller-provided seeds (parent-batch refinement values;
                    // see `seeded_zero_arg_externals` for the validity
                    // contract). Checked after the per-execution memo and
                    // before the interpreter callback.
                    if let Some((_, _, value)) =
                        self.seeded_zero_arg_externals.iter().find(|(n, prime, _)| {
                            *prime == self.prime_mode
                                && match (n, key) {
                                    (Value::String(a), Value::String(b)) => {
                                        Rp::ptr_eq(a, b) || a == b
                                    }
                                    _ => false,
                                }
                        })
                    {
                        regs[*rd as usize] = value.clone();
                        return Ok(DispatchOutcome::Continue);
                    }
                    let value = call_external(self.eval_ctx, &name, &[], self.prime_mode)?;
                    if let Some(memo) = self.zero_arg_external_memo.as_mut() {
                        memo.push((key.clone(), self.prime_mode, value.clone()));
                    }
                    regs[*rd as usize] = value;
                    return Ok(DispatchOutcome::Continue);
                }
                let args: Vec<Value> = (0..(*argc as usize))
                    .map(|i| regs[*args_start as usize + i].clone())
                    .collect();
                regs[*rd as usize] = call_external(self.eval_ctx, &name, &args, self.prime_mode)?;
            }

            // === Special ===
            Opcode::CondMove { rd, cond, rs } => {
                if as_bool(&regs[*cond as usize])? {
                    regs[*rd as usize] = regs[*rs as usize].clone();
                }
            }
            Opcode::Unchanged { rd, start, count } => {
                if self.is_action_execution() {
                    let action_state = self
                        .action_state
                        .get_or_insert_with(|| Box::new(super::execute::ActionVmState::default()));
                    let mut all_equal = true;
                    for i in 0..(*count as u16) {
                        let var_idx = match constants.get_value(*start + i) {
                            Value::SmallInt(idx) => {
                                u16::try_from(*idx).map_err(|_| VmError::TypeError {
                                    expected: "u16 variable index in constant pool",
                                    actual: idx.to_string(),
                                })?
                            }
                            other => {
                                return Err(VmError::TypeError {
                                    expected: "integer var index in constant pool",
                                    actual: format!("{other:?}"),
                                });
                            }
                        };
                        if let Some(written) =
                            action_state.bind_unchanged_or_written(var_idx, self.state.env_len())?
                        {
                            let current =
                                load_state_var(self.state, &mut self.state_cache, var_idx)?;
                            if written != current {
                                all_equal = false;
                            }
                        }
                    }
                    regs[*rd as usize] = Value::Bool(all_equal);
                    return Ok(DispatchOutcome::Continue);
                }
                let ns = self.next_state.ok_or_else(|| VmError::TypeError {
                    expected: "next state for UNCHANGED",
                    actual: "no next state bound".to_string(),
                })?;
                let mut all_equal = true;
                for i in 0..(*count as u16) {
                    let var_idx = match constants.get_value(*start + i) {
                        Value::SmallInt(idx) => *idx as usize,
                        other => {
                            return Err(VmError::TypeError {
                                expected: "integer var index in constant pool",
                                actual: format!("{other:?}"),
                            });
                        }
                    };
                    let cur = load_state_var(self.state, &mut self.state_cache, var_idx as u16)?;
                    let next =
                        load_prime_var(Some(ns), self.next_state_cache.as_mut(), var_idx as u16)?;
                    if cur != next {
                        all_equal = false;
                        break;
                    }
                }
                regs[*rd as usize] = Value::Bool(all_equal);
            }

            _ => unreachable!("non-scalar opcode routed to execute_scalar_opcode"),
        }

        Ok(DispatchOutcome::Continue)
    }
}
