// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Bytecode representation for TLA+ TIR expressions.
//!
//! This module defines a register-based bytecode instruction set that provides
//! 3-5x speedup over tree-walking interpretation by eliminating:
//! - Recursive function call overhead per AST node
//! - Pointer chasing through `Box<Spanned<TirExpr>>` indirections
//! - Branch misprediction on large `match` arms
//! - Data dependency chains through recursion
//!
//! The bytecode is compiled from TIR in a single O(n) pass at spec load time.
//! A register-based VM executes the flat opcode stream with a stack-allocated
//! register file (256 registers × 8B = 2KB).
//!
//! ## Architecture
//!
//! - **Register-based**: 256 virtual registers (r0-r255), each holds a `CompactValue`
//! - **Constant pool**: Compile-time constants stored in a separate array
//! - **Flat instruction stream**: No recursion, pure linear execution with jumps
//! - **Native-ready**: Bytecode is the input format for trust-ir/trust_cg lowering

pub mod action_transform;
mod chunk;
mod compiler;
mod const_fold;
mod disjunction_split;
mod inner_exists_expansion;
mod normalize;
mod opcode;
mod opcode_support;
mod state_footprint;

pub use chunk::{
    specialize_bytecode_function, specialize_bytecode_function_with_values, BytecodeChunk,
    BytecodeFunction, ConstantPool,
};
pub use compiler::{BytecodeCompiler, CalleeInfo, CompileError};
pub use const_fold::{
    const_fold_count, install_const_fold_executor, reset_const_fold_count, set_const_fold_override,
    ConstFoldExecutor,
};
pub use disjunction_split::{split_top_level_disjunction, split_top_level_disjunction_general};
pub use inner_exists_expansion::{
    can_expand_inner_exists, expand_inner_exists_preserving_offsets,
    static_expansion_drops_sibling_successor, ExpandedAction, InnerExistsInfo,
    MAX_INNER_DOMAIN_SIZE,
};
pub use normalize::normalize_action_function;
pub use opcode::{BuiltinOp, Opcode, Register};
pub use state_footprint::{analyze_predicate_state_footprint, PredicateStateFootprint};
