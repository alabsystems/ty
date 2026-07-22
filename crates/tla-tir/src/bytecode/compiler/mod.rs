// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! TIR → Bytecode compiler.
//!
//! Compiles `TirExpr` trees into flat bytecode in a single recursive pass.
//! Register allocation is a simple bump allocator — each sub-expression gets
//! the next available register. This is optimal for a tree-structured IR
//! (no register pressure issues with SSA-like trees).

use std::sync::Arc;

use super::chunk::{BytecodeChunk, BytecodeFunction, ConstantPool};
use super::opcode::{ConstIdx, Opcode, Register};
use crate::nodes::{PreservedAstBody, TirExpr};
use crate::TirOperator;
use thiserror::Error;
use tla_core::{NameId, Spanned};
use tla_value::Value;

mod compile_const_fold;
mod compile_control;
mod compile_expr;
mod compile_expr_support;
mod compile_quantifiers;
mod let_guard_hoist;
#[cfg(test)]
mod tests;

/// Errors during bytecode compilation.
#[derive(Debug, Error)]
pub enum CompileError {
    /// A TIR expression has no bytecode lowering yet; the payload names the construct.
    #[error("unsupported TIR expression for bytecode compilation: {0}")]
    Unsupported(String),

    /// The expression needed more than the 256 available registers.
    #[error("register overflow: expression too deep (>256 live values)")]
    RegisterOverflow,

    /// The expression required more than the 65536 constant-pool entries.
    #[error("constant pool overflow: too many constants (>65536)")]
    ConstantPoolOverflow,

    /// An on-demand callee tried to compile a name already on the exact
    /// compilation stack. Kept distinct from generic Unsupported so the
    /// emitting call site can authenticate recursive-self provenance.
    #[error("recursive on-demand compilation of '{0}'")]
    RecursiveOnDemand(String),
}

/// Bytecode compiler state.
///
/// Compiles a set of TIR operators into a `BytecodeChunk`. Each operator
/// becomes one `BytecodeFunction`.
pub struct BytecodeCompiler {
    chunk: BytecodeChunk,
    /// Map from operator name to function index, for Call resolution.
    op_indices: std::collections::HashMap<String, u16>,
    /// Pre-resolved constant values keyed by interned name.
    resolved_constants: std::collections::HashMap<NameId, Value>,
    /// Operator replacements from .cfg `CONSTANT Foo <- Bar` directives.
    /// Maps original name → replacement name so the compiler can resolve
    /// identifiers through the replacement chain.
    op_replacements: std::collections::HashMap<String, String>,
    /// State variable name → var_index mapping for resolving Ident names
    /// that are actually state variables. The TIR lowerer from raw (unresolved)
    /// Module AST produces `TirNameKind::Ident` for state vars; this map lets
    /// the compiler emit `LoadVar`/`LoadPrime`/`Unchanged` opcodes for them.
    state_vars: std::collections::HashMap<String, u16>,
    /// Propagated to every `FnCompileState` (see its `reg_recycle` docs).
    reg_recycle: bool,
    /// Fuse `Eq(lhs, FuncExcept(..))` / `Eq(lhs, RecordNew(..))` pairs into
    /// the non-materializing `EqFuncExcept` / `EqRecordNew` superinstructions
    /// (see their opcode docs for the exact-equivalence contract). Off by
    /// default; enabled ONLY for the implied-action term compile — bytecode
    /// consumed by the trust-cg native lowering never sees these opcodes.
    eq_fusion: bool,
    /// Fuse a direct two-element tuple literal used as the left operand of
    /// membership into `Tuple2SetIn`, avoiding the temporary `Value::Tuple`.
    /// Off by default: the opcode is currently VM-only and must not leak into
    /// bytecode consumed by native lowering.
    tuple2_set_in: bool,
    /// Fuse a direct set enumeration used as the left operand of `\subseteq`
    /// into `SetEnumSubseteq`, avoiding the temporary left `Value::Set`.
    /// Off by default because the opcode is VM-only.
    set_enum_subseteq: bool,
    /// Fuse the exact bound-local shape `e = <<e[1], e[2]>>` into a VM-only
    /// tuple-shape test. Off by default so native-lowered chunks cannot see it.
    tuple2_self_eq: bool,
    /// Fuse that tuple-shape test and its exact two-projection subset check
    /// against an unprimed state variable. Off by default because the opcode
    /// is VM-only and reads the state slot internally after short-circuiting.
    tuple2_self_subseteq: bool,
    /// Hoist one exact, VM-only SetFilter projection shape out of its loop.
    /// Off by default and additionally gated on `tuple2_set_in` at the match
    /// site; see `enable_set_filter_projection_hoist` for the proof contract.
    set_filter_projection_hoist: bool,
    /// Replace a call to a complete one-parameter global operator whose body
    /// is exactly `IF p = <<>> THEN 0 ELSE p[2]` with one VM builtin dispatch.
    /// Off by default: the builtin is VM-only and native-lowered chunks must
    /// never contain it.
    round_shape_apply: bool,
    /// Fuse the exact one-binder FORALL body
    /// `Round(child) = Round(parent) - 1` into `RoundStepEq` after proving
    /// both calls target the same complete direct Round definition and the
    /// parent is an outer register binding. Off by default because the opcode
    /// is VM-only.
    round_step_eq: bool,
    /// Omit the innermost `FuncApply` that materializes EXCEPT's `@` value
    /// when every path element is an Index and the replacement expression is
    /// independently certified to compile to an already-existing, total
    /// value. Off by default because the opcode contract is VM-only; the
    /// `TirProgram` VM compiler enables it for its VM-exclusive chunks.
    except_at_free_rhs: bool,
    /// Flatten multi-arm And/Or chains into ONE single-accumulator chain
    /// (the canonical shape the top-level-disjunction splitter detects),
    /// WITHOUT register rollback: registers stay monotone-unique, so the
    /// next-state transform's interference checks are unaffected. Implied by
    /// `reg_recycle`; independently enabled for action compilation.
    flatten_bool_chains: bool,
    /// Operator names that must compile as `CallExternal` (interpreter
    /// callback) instead of inline/`Call`, even when compilable. Used by the
    /// implied-action compile for root-module state-dependent zero-arg
    /// operators (refinement mappings like `token`/`pending`): the checker's
    /// fingerprint-keyed transition memo serves them per state, and CHOOSE
    /// inside them stays interpreter-produced by construction.
    force_external: std::collections::HashSet<String>,
}

impl BytecodeCompiler {
    /// Create a new compiler.
    #[must_use]
    pub fn new() -> Self {
        Self::with_resolved_constants(std::collections::HashMap::new())
    }

    /// Create a new compiler with a snapshot of already-resolved constants.
    #[must_use]
    pub fn with_resolved_constants(
        resolved_constants: std::collections::HashMap<NameId, Value>,
    ) -> Self {
        Self {
            chunk: BytecodeChunk::new(),
            op_indices: std::collections::HashMap::new(),
            resolved_constants,
            op_replacements: std::collections::HashMap::new(),
            state_vars: std::collections::HashMap::new(),
            reg_recycle: false,
            eq_fusion: false,
            tuple2_set_in: false,
            set_enum_subseteq: false,
            tuple2_self_eq: false,
            tuple2_self_subseteq: false,
            set_filter_projection_hoist: false,
            round_shape_apply: false,
            round_step_eq: false,
            except_at_free_rhs: false,
            flatten_bool_chains: false,
            force_external: std::collections::HashSet::new(),
        }
    }

    /// Force the named zero-arg operators to compile as `CallExternal`
    /// interpreter callbacks (see the `force_external` field docs).
    pub fn set_force_external_ops(&mut self, ops: std::collections::HashSet<String>) {
        self.force_external = ops;
    }

    /// Enable register recycling for all functions compiled by this compiler
    /// (boolean short-circuit points, quantifier loop bodies, aggregate
    /// operand slots). Used for large action-level predicate compilations
    /// that would otherwise overflow the 256-register file.
    pub fn enable_reg_recycling(&mut self) {
        self.reg_recycle = true;
    }

    /// Enable the Eq-fusion peephole (`EqFuncExcept` / `EqRecordNew`). See
    /// the `eq_fusion` field docs; implied-action term compilation only.
    pub fn enable_eq_fusion(&mut self) {
        self.eq_fusion = true;
    }

    /// Enable direct two-element tuple-membership fusion.
    ///
    /// This is intentionally opt-in while `Tuple2SetIn` is VM-only. Callers
    /// whose chunks may flow to native lowering must leave it disabled.
    pub fn enable_tuple2_set_in(&mut self) {
        self.tuple2_set_in = true;
    }

    /// Enable direct set-enumeration subset fusion.
    ///
    /// This is intentionally opt-in while `SetEnumSubseteq` is VM-only.
    /// Callers whose chunks may flow to native lowering must leave it disabled.
    pub fn enable_set_enum_subseteq(&mut self) {
        self.set_enum_subseteq = true;
    }

    /// Enable the exact bound-local two-projection tuple equality fusion.
    ///
    /// This is intentionally opt-in while `Tuple2SelfEq` is VM-only. Callers
    /// whose chunks may flow to native lowering must leave it disabled.
    pub fn enable_tuple2_self_eq(&mut self) {
        self.tuple2_self_eq = true;
    }

    /// Enable the exact tuple-shape plus state-set subset conjunction fusion.
    ///
    /// This is intentionally opt-in while `Tuple2SelfSubseteq` is VM-only.
    /// Callers whose chunks may reach native lowering must leave it disabled.
    pub fn enable_tuple2_self_subseteq(&mut self) {
        self.tuple2_self_subseteq = true;
    }

    /// Enable the VM-only SetFilter projection hoist.
    ///
    /// This recognizes only `{c \in D : <<outer, c>> \in S(arg)}` where
    /// `outer` and `arg` are already-bound registers and the complete global
    /// callee body is exactly `S(p) == p[k]`. It requires
    /// [`Self::enable_tuple2_set_in`]; callers whose chunks can reach native
    /// lowering must leave both options disabled.
    pub fn enable_set_filter_projection_hoist(&mut self) {
        self.set_filter_projection_hoist = true;
    }

    /// Enable the exact, VM-only Round-shape call superinstruction.
    ///
    /// The call-site matcher requires an unreplaced, non-external, complete
    /// global callee body with one parameter and the exact body documented on
    /// `round_shape_apply`. Callers whose chunks may reach native lowering
    /// must leave this disabled.
    pub fn enable_round_shape_apply(&mut self) {
        self.round_shape_apply = true;
    }

    /// Enable the exact, VM-only Round-step FORALL fusion.
    ///
    /// The structural proof and child-before-parent evaluation contract live
    /// beside `compile_forall_nested`. Callers whose chunks may reach native
    /// lowering must leave this disabled.
    pub fn enable_round_step_eq(&mut self) {
        self.round_step_eq = true;
    }

    /// Enable the VM-only EXCEPT `@`-free replacement optimization.
    ///
    /// This only applies to all-Index paths when the replacement is a bound
    /// register, a non-closure literal/resolved constant, or a direct
    /// state-variable load. Field paths are refused because VM lowering loses
    /// the source-level Field-vs-string-Index distinction. Callers whose chunks
    /// may reach native lowering must leave it disabled. `TirProgram` enables
    /// it by default for chunks that are executed exclusively by `BytecodeVm`.
    pub fn enable_except_at_free_rhs(&mut self) {
        self.except_at_free_rhs = true;
    }

    /// Enable single-accumulator And/Or chain flattening WITHOUT register
    /// recycling (see `flatten_bool_chains`).
    pub fn enable_bool_chain_flattening(&mut self) {
        self.flatten_bool_chains = true;
    }

    /// Set operator replacement mappings from .cfg `CONSTANT Foo <- Bar`.
    pub fn set_op_replacements(&mut self, replacements: std::collections::HashMap<String, String>) {
        self.op_replacements = replacements;
    }

    /// Set state variable name → index mapping for resolving unresolved Ident
    /// names that are actually state variables in callee TIR bodies.
    pub fn set_state_vars(&mut self, state_vars: std::collections::HashMap<String, u16>) {
        self.state_vars = state_vars;
    }

    /// Compile a single TIR operator into bytecode.
    ///
    /// Returns the operator index in the chunk's function table.
    pub fn compile_operator(&mut self, op: &TirOperator) -> Result<u16, CompileError> {
        let func = {
            let mut state = FnCompileState::new(
                op.name.clone(),
                0, // TIR operators are zero-arity (params bound via BindingChain)
                &mut self.chunk.constants,
            );
            state.reg_recycle = self.reg_recycle;
            state.eq_fusion = self.eq_fusion;
            state.tuple2_set_in = self.tuple2_set_in;
            state.set_enum_subseteq = self.set_enum_subseteq;
            state.tuple2_self_eq = self.tuple2_self_eq;
            state.tuple2_self_subseteq = self.tuple2_self_subseteq;
            state.set_filter_projection_hoist = self.set_filter_projection_hoist;
            state.round_shape_apply = self.round_shape_apply;
            state.round_step_eq = self.round_step_eq;
            state.except_at_free_rhs = self.except_at_free_rhs;
            state.flatten_bool_chains = self.flatten_bool_chains;
            if !self.force_external.is_empty() {
                state.force_external = Some(&self.force_external);
            }
            state.resolved_constants = Some(&self.resolved_constants);
            state.op_indices = Some(&self.op_indices);
            if !self.op_replacements.is_empty() {
                state.op_replacements = Some(&self.op_replacements);
            }
            if !self.state_vars.is_empty() {
                state.state_vars = Some(&self.state_vars);
            }

            // Compile the body expression, placing the result in register 0.
            let result_reg = state.compile_expr(&op.body)?;

            // If the result isn't in r0, move it.
            if result_reg != 0 {
                state.func.emit(Opcode::Move {
                    rd: 0,
                    rs: result_reg,
                });
            }
            state.func.emit(Opcode::Ret { rs: 0 });
            state.func
        };
        // `state` dropped here, releasing the borrow on self.chunk.constants.

        let idx = self.chunk.add_function(func);
        self.op_indices.insert(op.name.clone(), idx);
        Ok(idx)
    }

    /// Compile a standalone expression (e.g., an invariant predicate).
    ///
    /// Returns the function index.
    pub fn compile_expression(
        &mut self,
        name: &str,
        expr: &Spanned<TirExpr>,
    ) -> Result<u16, CompileError> {
        let func = {
            let mut state = FnCompileState::new(name.to_string(), 0, &mut self.chunk.constants);
            state.reg_recycle = self.reg_recycle;
            state.eq_fusion = self.eq_fusion;
            state.tuple2_set_in = self.tuple2_set_in;
            state.set_enum_subseteq = self.set_enum_subseteq;
            state.tuple2_self_eq = self.tuple2_self_eq;
            state.tuple2_self_subseteq = self.tuple2_self_subseteq;
            state.set_filter_projection_hoist = self.set_filter_projection_hoist;
            state.round_shape_apply = self.round_shape_apply;
            state.round_step_eq = self.round_step_eq;
            state.except_at_free_rhs = self.except_at_free_rhs;
            state.flatten_bool_chains = self.flatten_bool_chains;
            if !self.force_external.is_empty() {
                state.force_external = Some(&self.force_external);
            }
            state.resolved_constants = Some(&self.resolved_constants);
            if !self.state_vars.is_empty() {
                state.state_vars = Some(&self.state_vars);
            }

            let result_reg = state.compile_expr(expr)?;

            if result_reg != 0 {
                state.func.emit(Opcode::Move {
                    rd: 0,
                    rs: result_reg,
                });
            }
            state.func.emit(Opcode::Ret { rs: 0 });
            state.func
        };

        let idx = self.chunk.add_function(func);
        Ok(idx)
    }

    /// Compile a standalone expression with cross-operator call resolution.
    ///
    /// Pre-compiles all reachable callees from `callee_bodies` (fixed-point
    /// loop), then compiles the entry-point body. Callees are resolved via
    /// `op_indices`; unresolvable `Apply` nodes cause `Unsupported` fallback.
    ///
    /// Returns the function index of the entry-point operator.
    pub fn compile_expression_with_callees(
        &mut self,
        name: &str,
        params: &[String],
        expr: &Spanned<TirExpr>,
        callee_bodies: &std::collections::HashMap<String, CalleeInfo>,
    ) -> Result<u16, CompileError> {
        // Phase 1: pre-compile all available callees (fixed-point).
        // Multiple passes handle transitive dependencies: if A calls B,
        // B's compilation fails on first pass (B not in op_indices yet),
        // but succeeds on second pass after B itself is compiled.
        let mut progress = true;
        let mut last_errors: std::collections::HashMap<String, String> =
            std::collections::HashMap::new();
        while progress {
            progress = false;
            for (callee_name, info) in callee_bodies {
                if self.op_indices.contains_key(callee_name) || callee_name == name {
                    continue;
                }
                // Try to compile callee. Failures are OK — the callee will
                // just not be in op_indices and Apply will return Unsupported.
                match self.compile_callee_body(
                    callee_name,
                    &info.params,
                    &info.body,
                    Some(callee_bodies),
                ) {
                    Ok(_) => {
                        progress = true;
                        last_errors.remove(callee_name);
                    }
                    Err(e) => {
                        last_errors.insert(callee_name.clone(), e.to_string());
                    }
                }
            }
        }
        // Log final failures for callees that never compiled.
        if matches!(std::env::var("TY_BYTECODE_VM_STATS").as_deref(), Ok("1")) {
            for (cname, err) in &last_errors {
                if callee_bodies.contains_key(cname) {
                    eprintln!("[bytecode] callee '{cname}' failed: {err}");
                }
            }
        }

        // Phase 2: compile the entry-point body with Call resolution.
        self.compile_callee_body(name, params, expr, Some(callee_bodies))
    }

    /// Compile a single callee body with Call resolution from `op_indices`.
    fn compile_callee_body(
        &mut self,
        name: &str,
        params: &[String],
        body: &Spanned<TirExpr>,
        callee_bodies: Option<&std::collections::HashMap<String, CalleeInfo>>,
    ) -> Result<u16, CompileError> {
        let arity = params.len() as u8;
        // Capture function_count before borrowing self.chunk.constants mutably.
        let next_func_idx = self.chunk.function_count() as u16;
        let (func, sub_functions) = {
            let mut state = FnCompileState::new(name.to_string(), arity, &mut self.chunk.constants);
            state.reg_recycle = self.reg_recycle;
            state.eq_fusion = self.eq_fusion;
            state.tuple2_set_in = self.tuple2_set_in;
            state.set_enum_subseteq = self.set_enum_subseteq;
            state.tuple2_self_eq = self.tuple2_self_eq;
            state.tuple2_self_subseteq = self.tuple2_self_subseteq;
            state.set_filter_projection_hoist = self.set_filter_projection_hoist;
            state.round_shape_apply = self.round_shape_apply;
            state.round_step_eq = self.round_step_eq;
            state.except_at_free_rhs = self.except_at_free_rhs;
            state.flatten_bool_chains = self.flatten_bool_chains;
            if !self.force_external.is_empty() {
                state.force_external = Some(&self.force_external);
            }
            // Bind parameters to registers 0..arity.
            for (i, param) in params.iter().enumerate() {
                state.bindings.push((param.clone(), i as u8));
            }
            state.next_reg = arity;
            state.resolved_constants = Some(&self.resolved_constants);
            // Pass op_indices snapshot for Call resolution.
            state.op_indices = Some(&self.op_indices);
            state.callee_bodies = callee_bodies;
            if !self.op_replacements.is_empty() {
                state.op_replacements = Some(&self.op_replacements);
            }
            if !self.state_vars.is_empty() {
                state.state_vars = Some(&self.state_vars);
            }
            // Set the next function index so LET sub-functions get correct indices.
            state.next_func_idx = next_func_idx;

            let result_reg = state.compile_expr(body)?;

            if result_reg != 0 {
                state.func.emit(Opcode::Move {
                    rd: 0,
                    rs: result_reg,
                });
            }
            state.func.emit(Opcode::Ret { rs: 0 });
            (state.func, state.sub_functions)
        };

        // Add LET sub-functions first so their indices are valid.
        for sub_func in sub_functions {
            self.chunk.add_function(sub_func);
        }

        let idx = self.chunk.add_function(func);
        self.op_indices.insert(name.to_string(), idx);
        Ok(idx)
    }

    /// Update the operator-name-to-function-index map used for Call resolution.
    ///
    /// Called between compilation phases so that operators compiled in an
    /// earlier phase are available as Call targets in later phases.
    pub fn set_op_indices<'i, I>(&mut self, indices: I)
    where
        I: IntoIterator<Item = (&'i String, &'i u16)>,
    {
        for (name, &idx) in indices {
            self.op_indices.entry(name.clone()).or_insert(idx);
        }
    }

    /// Finish compilation and return the bytecode chunk.
    #[must_use]
    pub fn finish(self) -> BytecodeChunk {
        self.chunk
    }

    /// Finish compilation and return the bytecode chunk along with the
    /// operator name → function index mapping.
    #[must_use]
    pub fn finish_with_indices(self) -> (BytecodeChunk, std::collections::HashMap<String, u16>) {
        (self.chunk, self.op_indices)
    }

    /// Check if an operator has been compiled (by name).
    pub fn has_operator(&self, name: &str) -> bool {
        self.op_indices.contains_key(name)
    }

    /// Number of successfully compiled operators.
    pub fn op_count(&self) -> usize {
        self.op_indices.len()
    }

    /// Get the compiled chunk (reference).
    #[must_use]
    pub fn chunk(&self) -> &BytecodeChunk {
        &self.chunk
    }
}

/// Information about a callee operator available for cross-operator call compilation.
///
/// Part of #4113: `body` is `Arc<Spanned<TirExpr>>` so cloning `CalleeInfo`
/// is O(1) refcount bump instead of deep-cloning the entire TIR expression tree.
#[derive(Clone)]
pub struct CalleeInfo {
    /// Parameter names of the callee operator, in order.
    pub params: Vec<String>,
    /// The callee's lowered body, shared by `Arc` for O(1) cloning.
    pub body: Arc<Spanned<TirExpr>>,
    /// Preserved AST body for building first-class closure values.
    ///
    /// This is required when a parameterized operator is referenced as a value
    /// (`LET F == Op IN F(...)`). The bytecode compiler can lower that to a
    /// closure constant instead of incorrectly emitting a zero-arg `Call`.
    pub ast_body: Option<PreservedAstBody>,
}

impl Default for BytecodeCompiler {
    fn default() -> Self {
        Self::new()
    }
}

/// Per-function compilation state.
struct FnCompileState<'a> {
    func: BytecodeFunction,
    constants: &'a mut ConstantPool,
    /// Next available register.
    next_reg: Register,
    /// Bound variable name → register mapping for quantifier/LET variables.
    bindings: Vec<(String, Register)>,
    /// Pre-compiled operator indices for Call resolution.
    /// When Some, Apply(Name) and zero-arg Name(Ident) resolve via this map.
    op_indices: Option<&'a std::collections::HashMap<String, u16>>,
    /// Callee metadata available to the current compilation.
    ///
    /// Used to distinguish zero-arg operators (emit `Call`) from parameterized
    /// operators referenced as first-class values (emit closure constants).
    callee_bodies: Option<&'a std::collections::HashMap<String, CalleeInfo>>,
    /// Pre-resolved constant values keyed by interned name.
    resolved_constants: Option<&'a std::collections::HashMap<NameId, Value>>,
    /// LET-scoped sub-function indices for parameterized LET defs.
    /// Checked before `op_indices` during callee resolution.
    local_op_indices: std::collections::HashMap<String, u16>,
    /// Sub-functions compiled from parameterized LET defs.
    /// Drained by the caller and added to the chunk's function table.
    sub_functions: Vec<BytecodeFunction>,
    /// Next function index for sub-functions (tracks chunk.functions.len() at compile start).
    next_func_idx: u16,
    /// Operator replacements for resolving `.cfg` `CONSTANT Foo <- Bar` directives.
    op_replacements: Option<&'a std::collections::HashMap<String, String>>,
    /// State variable name → var_index for resolving Ident names as state vars.
    state_vars: Option<&'a std::collections::HashMap<String, u16>>,
    /// Register holding the current `@` value during EXCEPT compilation.
    except_at_register: Option<Register>,
    /// When true, all state variable loads emit LoadPrime instead of LoadVar.
    /// Set during compilation of general Prime(expr) expressions.
    in_prime_context: bool,
    /// Set of callee names currently being compiled on-demand (recursion guard).
    /// Part of #3789: prevents infinite recursion when a callee body references
    /// itself or a cycle of callees.
    on_demand_compiling: std::collections::HashSet<String>,
    /// True while compiling a constant-fold scratch function (F1, lever L2).
    /// Disables nested fold attempts — the enclosing subtree already executes
    /// as a whole at compile time.
    const_folding: bool,
    /// Recycle temporary registers at boolean short-circuit points, quantifier
    /// loop bodies, and aggregate operand slots (`reg_rollback`). Off by
    /// default so existing compilations are bit-for-bit unchanged; enabled by
    /// `BytecodeCompiler::enable_reg_recycling` for compilations of large
    /// action-level predicates (implied-action terms) that would otherwise
    /// overflow the 256-register file. Sound because bytecode is linear: a
    /// rolled-back register index is only re-allocated by instructions
    /// compiled AFTER the value it held was consumed (moved into an
    /// accumulator allocated before the checkpoint), and loop-carried
    /// registers (accumulators, iterator bindings) are always allocated
    /// before the body checkpoint.
    reg_recycle: bool,
    /// See `BytecodeCompiler::eq_fusion`.
    pub(super) eq_fusion: bool,
    /// See `BytecodeCompiler::tuple2_set_in`.
    pub(super) tuple2_set_in: bool,
    /// See `BytecodeCompiler::set_enum_subseteq`.
    pub(super) set_enum_subseteq: bool,
    /// See `BytecodeCompiler::tuple2_self_eq`.
    pub(super) tuple2_self_eq: bool,
    /// See `BytecodeCompiler::tuple2_self_subseteq`.
    pub(super) tuple2_self_subseteq: bool,
    /// See `BytecodeCompiler::set_filter_projection_hoist`.
    pub(super) set_filter_projection_hoist: bool,
    /// See `BytecodeCompiler::round_shape_apply`.
    pub(super) round_shape_apply: bool,
    /// See `BytecodeCompiler::round_step_eq`.
    pub(super) round_step_eq: bool,
    /// See `BytecodeCompiler::except_at_free_rhs`.
    pub(super) except_at_free_rhs: bool,
    /// See `BytecodeCompiler::flatten_bool_chains`.
    pub(super) flatten_bool_chains: bool,
    /// Names that must compile as `CallExternal` (see
    /// `BytecodeCompiler::force_external`).
    force_external: Option<&'a std::collections::HashSet<String>>,
}

impl<'a> FnCompileState<'a> {
    fn new(name: String, arity: u8, constants: &'a mut ConstantPool) -> Self {
        Self {
            func: BytecodeFunction::new(name, arity),
            constants,
            next_reg: 0,
            bindings: Vec::new(),
            op_indices: None,
            callee_bodies: None,
            resolved_constants: None,
            local_op_indices: std::collections::HashMap::new(),
            sub_functions: Vec::new(),
            next_func_idx: 0,
            op_replacements: None,
            state_vars: None,
            except_at_register: None,
            in_prime_context: false,
            on_demand_compiling: std::collections::HashSet::new(),
            const_folding: false,
            reg_recycle: false,
            eq_fusion: false,
            tuple2_set_in: false,
            set_enum_subseteq: false,
            tuple2_self_eq: false,
            tuple2_self_subseteq: false,
            set_filter_projection_hoist: false,
            round_shape_apply: false,
            round_step_eq: false,
            except_at_free_rhs: false,
            flatten_bool_chains: false,
            force_external: None,
        }
    }

    /// True when `name`/`resolved` is pinned to `CallExternal` compilation
    /// and is not shadowed by a LET-local operator in the current scope.
    fn is_force_external(&self, name: &str, resolved: &str) -> bool {
        let Some(force) = self.force_external else {
            return false;
        };
        if self.local_op_indices.contains_key(name) || self.local_op_indices.contains_key(resolved)
        {
            return false;
        }
        force.contains(name) || force.contains(resolved)
    }

    /// Resolve an operator name through the replacement map.
    fn resolve_op_name<'b>(&'b self, name: &'b str) -> &'b str {
        self.op_replacements
            .and_then(|r| r.get(name))
            .map(String::as_str)
            .unwrap_or(name)
    }

    /// Register-file checkpoint for `reg_rollback`.
    #[inline]
    fn reg_checkpoint(&self) -> Register {
        self.next_reg
    }

    /// Roll the bump allocator back to `checkpoint`, releasing every register
    /// allocated since. No-op unless `reg_recycle` is enabled. Callers must
    /// guarantee no live value (result accumulators, loop iterators, bound
    /// variables) resides at an index >= `checkpoint`.
    #[inline]
    fn reg_rollback(&mut self, checkpoint: Register) {
        if self.reg_recycle && checkpoint <= self.next_reg {
            self.next_reg = checkpoint;
        }
    }

    /// Allocate a fresh register.
    fn alloc_reg(&mut self) -> Result<Register, CompileError> {
        if self.next_reg == 255 {
            return Err(CompileError::RegisterOverflow);
        }
        let r = self.next_reg;
        self.next_reg += 1;
        Ok(r)
    }

    /// Reserve `count` consecutive registers and return the starting register.
    fn alloc_consecutive_regs(&mut self, count: usize) -> Result<Register, CompileError> {
        let start = self.next_reg;
        for _ in 0..count {
            self.alloc_reg()?;
        }
        Ok(start)
    }

    /// Compile child expressions and pack their final results into consecutive registers.
    ///
    /// Aggregate opcodes consume contiguous operand ranges, but child expressions may
    /// allocate intermediate temporaries and leave gaps in the register file. Reserve
    /// the operand slots up front, then move each child result into its expected slot.
    fn compile_exprs_into_consecutive<'b, I>(&mut self, exprs: I) -> Result<Register, CompileError>
    where
        I: IntoIterator<Item = &'b Spanned<TirExpr>>,
    {
        let exprs: Vec<_> = exprs.into_iter().collect();
        let start = self.alloc_consecutive_regs(exprs.len())?;
        for (index, expr) in exprs.into_iter().enumerate() {
            // Each child's temporaries are dead once its result is moved into
            // its operand slot; recycle them (reg_recycle only).
            let cp = self.reg_checkpoint();
            let r_result = self.compile_expr(expr)?;
            let expected = start + index as u8;
            if r_result != expected {
                self.func.emit(Opcode::Move {
                    rd: expected,
                    rs: r_result,
                });
            }
            self.reg_rollback(cp);
        }
        Ok(start)
    }

    /// Look up a bound variable by name.
    fn lookup_binding(&self, name: &str) -> Option<Register> {
        // Search from the end (most recent binding wins).
        self.bindings
            .iter()
            .rev()
            .find(|(n, _)| n == name)
            .map(|(_, r)| *r)
    }

    /// Add a constant to the pool.
    fn add_const(&mut self, value: Value) -> Result<ConstIdx, CompileError> {
        let idx = self.constants.value_count();
        if idx > ConstIdx::MAX as usize {
            return Err(CompileError::ConstantPoolOverflow);
        }
        Ok(self.constants.add_value(value))
    }
}
