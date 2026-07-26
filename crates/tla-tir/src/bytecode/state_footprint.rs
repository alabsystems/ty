// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Static state-footprint analysis for compiled predicate bytecode.
//!
//! [`analyze_predicate_state_footprint`] computes, for a compiled boolean
//! predicate (an implied-action / invariant-shaped term), the EXACT set of
//! state inputs its VM execution can observe:
//!
//! * **Direct state-slot reads** — `LoadVar` / `LoadPrime` / `Unchanged` and
//!   VM-only fused-op var indices (prime mode only selects WHICH side of the
//!   transition a dynamic `LoadVar` reads, so callers must key on both sides
//!   of every slot).
//! * **Zero-arg external operators** — `CallExternal { argc: 0 }` names,
//!   whose results are produced by the interpreter at run time (in current
//!   or primed mode depending on the dynamic prime flag; callers must treat
//!   both modes as observable).
//!
//! Everything else in the admitted opcode set is a pure function of
//! registers and pool constants: the VM handlers for those opcodes never
//! consult the evaluation context or the bound state arrays.
//!
//! The analysis is FAIL-CLOSED and returns `None` when the predicate's
//! execution could observe state through any other channel:
//!
//! * `MakeClosure` / `ValueApply` — closure application evaluates a stored
//!   body through the evaluation context; the body may read live state.
//! * `CallExternal { argc > 0 }` — parameterized externals evaluate operator
//!   bodies that may read arbitrary state.
//! * `StoreVar` — a predicate must not write state; a chunk that does is
//!   outside this analysis' contract.
//! * A referenced pool constant that is not pure data
//!   ([`Value::is_concrete_data`]) — e.g. a closure template: applying it
//!   evaluates expressions through the evaluation context.
//! * Malformed chunks (out-of-range function/constant references).
//!
//! The opcode match below is intentionally EXHAUSTIVE (no `_` arm): adding a
//! new opcode fails compilation here and forces an explicit classification,
//! so a future state-reading opcode can never be silently classified as pure.
//!
//! Consumer: the implied-action verdict cache (`tla-check`), which keys
//! per-transition verdicts by the values of exactly this footprint.

use rustc_hash::FxHashSet;

use super::chunk::{BytecodeChunk, ConstantPool};
use super::opcode::Opcode;

/// The state inputs a compiled predicate's VM execution can observe.
#[derive(Debug, Clone, Default)]
pub struct PredicateStateFootprint {
    /// State-variable slots read directly (`LoadVar` / `LoadPrime` /
    /// `Unchanged` or a fused equivalent), sorted and deduplicated. Callers
    /// must treat BOTH the current-state and next-state values of every listed
    /// slot as inputs.
    pub direct_slots: Vec<u16>,
    /// Zero-arg `CallExternal` operator names, sorted and deduplicated.
    /// Callers must treat the interpreter-evaluated results in BOTH prime
    /// modes as inputs.
    pub zero_arg_externals: Vec<String>,
}

/// Analyze the state footprint of the predicate rooted at `entry_func`.
///
/// Returns `None` (fail closed) when any reachable opcode could observe
/// state outside the reported footprint. See module docs for the contract.
#[must_use]
pub fn analyze_predicate_state_footprint(
    chunk: &BytecodeChunk,
    entry_func: u16,
) -> Option<PredicateStateFootprint> {
    let mut slots: FxHashSet<u16> = FxHashSet::default();
    let mut externals: FxHashSet<String> = FxHashSet::default();
    let mut visited: FxHashSet<u16> = FxHashSet::default();
    let mut worklist: Vec<u16> = vec![entry_func];

    while let Some(func_idx) = worklist.pop() {
        if !visited.insert(func_idx) {
            continue;
        }
        let func = chunk.functions.get(func_idx as usize)?;
        for op in &func.instructions {
            scan_opcode(
                op,
                &chunk.constants,
                &mut slots,
                &mut externals,
                &mut worklist,
            )?;
        }
    }

    let mut direct_slots: Vec<u16> = slots.into_iter().collect();
    direct_slots.sort_unstable();
    let mut zero_arg_externals: Vec<String> = externals.into_iter().collect();
    zero_arg_externals.sort_unstable();
    Some(PredicateStateFootprint {
        direct_slots,
        zero_arg_externals,
    })
}

/// Classify one opcode; returns `None` to fail the whole analysis closed.
#[allow(clippy::too_many_lines)]
fn scan_opcode(
    op: &Opcode,
    constants: &ConstantPool,
    slots: &mut FxHashSet<u16>,
    externals: &mut FxHashSet<String>,
    worklist: &mut Vec<u16>,
) -> Option<()> {
    match op {
        // --- Direct state access: record the slot ---
        Opcode::LoadVar { var_idx, .. } | Opcode::LoadPrime { var_idx, .. } => {
            slots.insert(*var_idx);
        }
        Opcode::Tuple2SelfSubseteq { set_var_idx, .. } => {
            // The read is conditional on the tuple-shape guard, but footprint
            // keys conservatively include every state slot execution may see.
            slots.insert(*set_var_idx);
        }
        // UNCHANGED reads the var indices listed as consecutive SmallInt pool
        // constants starting at `start` (mirrors the VM handler exactly).
        Opcode::Unchanged { start, count, .. } => {
            for i in 0..u16::from(*count) {
                let idx = start.checked_add(i)?;
                match constants.try_get_value(idx)? {
                    tla_value::Value::SmallInt(v) if *v >= 0 && *v <= i64::from(u16::MAX) => {
                        slots.insert(*v as u16);
                    }
                    _ => return None,
                }
            }
        }

        // --- Interpreter callback: zero-arg externals are reported; anything
        // --- with arguments can read arbitrary state (fail closed).
        Opcode::CallExternal { name_idx, argc, .. } => {
            if *argc != 0 {
                return None;
            }
            match constants.try_get_value(*name_idx)? {
                tla_value::Value::String(name) => {
                    externals.insert(name.to_string());
                }
                _ => return None,
            }
        }

        // --- Compiled callee: recurse ---
        Opcode::Call { op_idx, .. } => {
            worklist.push(*op_idx);
        }

        // --- Pool constants must be pure data (no closure templates etc.) ---
        Opcode::LoadConst { idx, .. } => {
            let value = constants.try_get_value(*idx)?;
            if !value.is_concrete_data() {
                return None;
            }
        }

        // --- Fail closed: execution can observe state beyond the footprint ---
        // Closure construction/application evaluates stored bodies through the
        // evaluation context; a predicate must not write state.
        Opcode::MakeClosure { .. } | Opcode::ValueApply { .. } | Opcode::StoreVar { .. } => {
            return None;
        }

        // --- Pure over registers / immediates / (already-checked) constants ---
        // Scalar loads, moves, arithmetic, comparisons, boolean ops, control
        // flow, quantifier/builder loops, and compound-value constructors:
        // their VM handlers compute strictly from operand registers and pool
        // constants and never consult the evaluation context or state arrays.
        // (`Eq`/`Neq` route through `values_equal`, which is deterministic in
        // its operand values; `CallBuiltin` dispatches to pure builtins that
        // receive only operand values.)
        Opcode::LoadImm { .. }
        | Opcode::LoadBool { .. }
        | Opcode::Move { .. }
        | Opcode::AddInt { .. }
        | Opcode::SubInt { .. }
        | Opcode::MulInt { .. }
        | Opcode::DivInt { .. }
        | Opcode::IntDiv { .. }
        | Opcode::ModInt { .. }
        | Opcode::NegInt { .. }
        | Opcode::PowInt { .. }
        | Opcode::Eq { .. }
        | Opcode::Tuple2SelfEq { .. }
        | Opcode::Neq { .. }
        | Opcode::LtInt { .. }
        | Opcode::LeInt { .. }
        | Opcode::GtInt { .. }
        | Opcode::GeInt { .. }
        | Opcode::And { .. }
        | Opcode::Or { .. }
        | Opcode::Not { .. }
        | Opcode::Implies { .. }
        | Opcode::Equiv { .. }
        | Opcode::Jump { .. }
        | Opcode::JumpTrue { .. }
        | Opcode::JumpFalse { .. }
        | Opcode::Ret { .. }
        | Opcode::SetEnum { .. }
        | Opcode::SetIn { .. }
        | Opcode::Tuple2SetIn { .. }
        | Opcode::SetEnumSubseteq { .. }
        | Opcode::SetUnion { .. }
        | Opcode::SetIntersect { .. }
        | Opcode::SetDiff { .. }
        | Opcode::Subseteq { .. }
        | Opcode::RoundStepEq { .. }
        | Opcode::EdgeFilter { .. }
        | Opcode::Powerset { .. }
        | Opcode::BigUnion { .. }
        | Opcode::KSubset { .. }
        | Opcode::Range { .. }
        | Opcode::ForallBegin { .. }
        | Opcode::ForallNext { .. }
        | Opcode::ExistsBegin { .. }
        | Opcode::ExistsNext { .. }
        | Opcode::RecordNew { .. }
        | Opcode::RecordGet { .. }
        | Opcode::FuncApply { .. }
        | Opcode::Domain { .. }
        | Opcode::FuncExcept { .. }
        | Opcode::TupleNew { .. }
        | Opcode::TupleGet { .. }
        | Opcode::FuncDef { .. }
        | Opcode::FuncSet { .. }
        | Opcode::RecordSet { .. }
        | Opcode::Times { .. }
        | Opcode::SeqNew { .. }
        | Opcode::StrConcat { .. }
        | Opcode::CondMove { .. }
        | Opcode::ChooseBegin { .. }
        | Opcode::ChooseNext { .. }
        | Opcode::SetBuilderBegin { .. }
        | Opcode::SetFilterBegin { .. }
        | Opcode::FuncDefBegin { .. }
        | Opcode::LoopNext { .. }
        | Opcode::SetPrimeMode { .. }
        | Opcode::Concat { .. }
        | Opcode::CallBuiltin { .. }
        | Opcode::Nop
        | Opcode::Halt => {}

        // Fused Eq superinstructions: pure over their operand registers and
        // (string-name) pool constants — semantically the producer
        // (FuncExcept / RecordNew) followed by Eq, neither of which touches
        // state or the evaluation context.
        Opcode::EqFuncExcept { .. } => {}
        Opcode::EqRecordNew {
            fields_start,
            count,
            ..
        } => {
            for i in 0..u16::from(*count) {
                let idx = fields_start.checked_add(i)?;
                match constants.try_get_value(idx)? {
                    tla_value::Value::String(_) => {}
                    _ => return None,
                }
            }
        }
    }
    Some(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bytecode::chunk::BytecodeFunction;
    use tla_value::Value;

    fn chunk_with(functions: Vec<BytecodeFunction>, constants: Vec<Value>) -> BytecodeChunk {
        let mut chunk = BytecodeChunk::new();
        for value in constants {
            chunk.constants.add_value(value);
        }
        chunk.functions = functions;
        chunk
    }

    fn func(name: &str, instructions: Vec<Opcode>) -> BytecodeFunction {
        let mut f = BytecodeFunction::new(name.to_string(), 0);
        f.instructions = instructions;
        f
    }

    #[test]
    fn collects_direct_slots_and_externals() {
        let chunk = chunk_with(
            vec![func(
                "p",
                vec![
                    Opcode::LoadVar { rd: 0, var_idx: 2 },
                    Opcode::LoadPrime { rd: 1, var_idx: 1 },
                    Opcode::CallExternal {
                        rd: 2,
                        name_idx: 0,
                        args_start: 0,
                        argc: 0,
                        self_recursive: false,
                    },
                    Opcode::Ret { rs: 0 },
                ],
            )],
            vec![Value::String("token".into())],
        );
        let fp = analyze_predicate_state_footprint(&chunk, 0).expect("footprint");
        assert_eq!(fp.direct_slots, vec![1, 2]);
        assert_eq!(fp.zero_arg_externals, vec!["token".to_string()]);
    }

    #[test]
    fn tuple2_self_subseteq_collects_conditional_state_slot() {
        let chunk = chunk_with(
            vec![func(
                "p",
                vec![
                    Opcode::Tuple2SelfSubseteq {
                        rd: 1,
                        value: 0,
                        set_var_idx: 7,
                    },
                    Opcode::Ret { rs: 1 },
                ],
            )],
            vec![],
        );
        let fp = analyze_predicate_state_footprint(&chunk, 0).expect("footprint");
        assert_eq!(fp.direct_slots, vec![7]);
        assert!(fp.zero_arg_externals.is_empty());
    }

    #[test]
    fn round_step_eq_is_pure_over_registers() {
        let chunk = chunk_with(
            vec![func(
                "p",
                vec![
                    Opcode::RoundStepEq {
                        rd: 2,
                        child: 0,
                        parent: 1,
                    },
                    Opcode::Ret { rs: 2 },
                ],
            )],
            vec![],
        );
        let fp = analyze_predicate_state_footprint(&chunk, 0).expect("footprint");
        assert!(fp.direct_slots.is_empty());
        assert!(fp.zero_arg_externals.is_empty());
    }

    #[test]
    fn recurses_into_callees() {
        let chunk = chunk_with(
            vec![
                func(
                    "p",
                    vec![
                        Opcode::Call {
                            rd: 0,
                            op_idx: 1,
                            args_start: 0,
                            argc: 0,
                        },
                        Opcode::Ret { rs: 0 },
                    ],
                ),
                func(
                    "q",
                    vec![Opcode::LoadVar { rd: 0, var_idx: 7 }, Opcode::Ret { rs: 0 }],
                ),
            ],
            vec![],
        );
        let fp = analyze_predicate_state_footprint(&chunk, 0).expect("footprint");
        assert_eq!(fp.direct_slots, vec![7]);
    }

    #[test]
    fn unchanged_slot_list_is_collected() {
        let chunk = chunk_with(
            vec![func(
                "p",
                vec![
                    Opcode::Unchanged {
                        rd: 0,
                        start: 0,
                        count: 2,
                    },
                    Opcode::Ret { rs: 0 },
                ],
            )],
            vec![Value::SmallInt(3), Value::SmallInt(0)],
        );
        let fp = analyze_predicate_state_footprint(&chunk, 0).expect("footprint");
        assert_eq!(fp.direct_slots, vec![0, 3]);
    }

    #[test]
    fn fails_closed_on_parameterized_external() {
        let chunk = chunk_with(
            vec![func(
                "p",
                vec![Opcode::CallExternal {
                    rd: 0,
                    name_idx: 0,
                    args_start: 0,
                    argc: 1,
                    self_recursive: false,
                }],
            )],
            vec![Value::String("op".into())],
        );
        assert!(analyze_predicate_state_footprint(&chunk, 0).is_none());
    }

    #[test]
    fn fails_closed_on_store_and_closures() {
        for bad in [
            Opcode::StoreVar { var_idx: 0, rs: 0 },
            Opcode::MakeClosure {
                rd: 0,
                template_idx: 0,
                captures_start: 0,
                capture_count: 0,
            },
            Opcode::ValueApply {
                rd: 0,
                func: 0,
                args_start: 0,
                argc: 1,
            },
        ] {
            let chunk = chunk_with(vec![func("p", vec![bad])], vec![Value::SmallInt(1)]);
            assert!(analyze_predicate_state_footprint(&chunk, 0).is_none());
        }
    }

    #[test]
    fn fails_closed_on_missing_function() {
        let chunk = chunk_with(vec![], vec![]);
        assert!(analyze_predicate_state_footprint(&chunk, 0).is_none());
    }
}
