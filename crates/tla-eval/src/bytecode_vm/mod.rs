// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Register-based bytecode VM for TLA+ expression evaluation.
//!
//! Executes flat opcode streams produced by `tla_tir::bytecode::BytecodeCompiler`.
//! The VM operates on a stack-allocated register file (256 × `Value`) and
//! returns the result of the compiled expression.
//!
//! ## Design
//!
//! - **Register file**: 256 slots, each holding a `Value`. Initialized to `Value::Bool(false)`.
//! - **State access**: Current/next state arrays passed by reference.
//! - **Quantifier loops**: Managed via an internal iterator stack.
//! - **Error handling**: Returns `EvalResult<Value>` — type errors propagate as `EvalError`.
//!
//! ## Limitations
//!
//! - Lambda closures that capture LET-scoped operators fall back to the tree-walk evaluator
//! - Choose, SetBuilder, SetFilter, FuncDef loops use iterator stack (basic impl)
//! - No ENABLED/temporal operator support; action mode accepts only certified,
//!   next-state-transformed functions

pub mod compile;
mod execute;
mod execute_compound;
mod execute_dispatch;
mod execute_helpers;
mod execute_loops;
mod execute_scalar;
#[cfg(test)]
mod tests;

pub(crate) use compile::resolved_constants_with_bytecode_stdlib;
pub use compile::{
    collect_bytecode_namespace_callees, compile_operators_to_bytecode,
    compile_operators_to_bytecode_full, compile_operators_to_bytecode_full_with_state_vars,
    compile_operators_to_bytecode_with_constants,
    compile_operators_to_bytecode_with_flat_bool_chains,
    compile_operators_to_bytecode_with_reg_recycling, CompiledBytecode,
};
pub use execute::{ActionVmOutcome, BytecodeVm, VmError};
pub use execute_helpers::eval_zero_arg_external;
