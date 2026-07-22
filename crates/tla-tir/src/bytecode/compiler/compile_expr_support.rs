// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Support helpers for `compile_expr` — name resolution, EXCEPT/record lowering,
//! and Prime/UNCHANGED special-case compilation.

use super::super::opcode::{Opcode, Register};
use super::{CalleeInfo, CompileError, FnCompileState};
use crate::nodes::{TirExceptPathElement, TirExpr, TirNameKind, TirNameRef};
use tla_core::Spanned;
use tla_value::Value;

impl<'a> FnCompileState<'a> {
    /// Resolve an unbound name as the state slot that `compile_name_expr`
    /// would load. Raw module ASTs retain state variables as `Ident`, so
    /// matchers that embed a slot must mirror the ordinary name-resolution
    /// precedence instead of requiring a pre-resolved `StateVar` node.
    pub(super) fn resolve_name_state_var(&self, name_ref: &TirNameRef) -> Option<u16> {
        if self.lookup_binding(&name_ref.name).is_some() {
            return None;
        }
        match &name_ref.kind {
            TirNameKind::StateVar { index } => Some(*index),
            TirNameKind::Ident => {
                if self.resolved_constant_value(name_ref).is_some() {
                    return None;
                }
                let resolved_name = self.resolve_op_name(&name_ref.name);
                if self.is_force_external(&name_ref.name, resolved_name) {
                    return None;
                }
                if let Some(info) = self
                    .callee_bodies
                    .and_then(|bodies| bodies.get(resolved_name))
                {
                    if !info.params.is_empty()
                        || self
                            .inlineable_zero_arg_finite_domain_body(resolved_name)
                            .is_some()
                    {
                        return None;
                    }
                }
                if self.local_op_indices.contains_key(&name_ref.name)
                    || self
                        .op_indices
                        .is_some_and(|indices| indices.contains_key(resolved_name))
                {
                    return None;
                }
                self.state_vars
                    .and_then(|state_vars| state_vars.get(&name_ref.name))
                    .copied()
            }
        }
    }

    pub(super) fn compile_name_expr(
        &mut self,
        name_ref: &TirNameRef,
    ) -> Result<Register, CompileError> {
        // Check for bound variable first.
        if let Some(reg) = self.lookup_binding(&name_ref.name) {
            return Ok(reg);
        }
        match &name_ref.kind {
            TirNameKind::StateVar { index } => {
                let rd = self.alloc_reg()?;
                if self.in_prime_context {
                    self.func.emit(Opcode::LoadPrime {
                        rd,
                        var_idx: *index,
                    });
                } else {
                    self.func.emit(Opcode::LoadVar {
                        rd,
                        var_idx: *index,
                    });
                }
                Ok(rd)
            }
            TirNameKind::Ident => {
                if let Some(resolved_constants) = self.resolved_constants {
                    let lookup_id = if name_ref.name_id != tla_core::NameId::INVALID {
                        Some(name_ref.name_id)
                    } else {
                        tla_core::name_intern::lookup_name_id(&name_ref.name)
                    };
                    if let Some(lookup_id) = lookup_id {
                        if let Some(value) = resolved_constants.get(&lookup_id) {
                            return self.compile_const(value);
                        }
                    }
                }
                // Own the resolved name so it doesn't borrow `self`.
                let resolved_name = self.resolve_op_name(&name_ref.name).to_string();
                // Pinned interpreter callback (refinement-mapping operators):
                // compile as CallExternal even though the body is compilable,
                // so the checker's transition memo serves the value and CHOOSE
                // stays interpreter-produced.
                if self.is_force_external(&name_ref.name, &resolved_name) {
                    let name_idx = self.add_const(Value::string(resolved_name.clone()))?;
                    let rd = self.alloc_reg()?;
                    self.func.emit(Opcode::CallExternal {
                        rd,
                        name_idx,
                        args_start: 0,
                        argc: 0,
                        self_recursive: false,
                    });
                    return Ok(rd);
                }
                if let Some(callee_bodies) = self.callee_bodies {
                    if let Some(info) = callee_bodies.get(&resolved_name) {
                        if !info.params.is_empty() {
                            let ast_body = info.ast_body.as_ref().ok_or_else(|| {
                                CompileError::Unsupported(format!(
                                    "parameterized operator reference '{}' is missing an AST body",
                                    resolved_name
                                ))
                            })?;
                            return self.compile_closure_const(
                                info.params.clone(),
                                (*ast_body.0).clone(),
                                Some(&*info.body),
                            );
                        }
                    }
                }
                if let Some(body) = self.inlineable_zero_arg_finite_domain_body(&resolved_name) {
                    return self.compile_expr(&body);
                }
                // Try to resolve as a zero-arg operator call.
                // Check local (LET) scope first, then global op_indices.
                if let Some(&op_idx) = self.local_op_indices.get(&name_ref.name) {
                    let rd = self.alloc_reg()?;
                    self.func.emit(Opcode::Call {
                        rd,
                        op_idx,
                        args_start: 0,
                        argc: 0,
                    });
                    return Ok(rd);
                }
                if let Some(op_indices) = self.op_indices {
                    if let Some(&op_idx) = op_indices.get(resolved_name.as_str()) {
                        let rd = self.alloc_reg()?;
                        self.func.emit(Opcode::Call {
                            rd,
                            op_idx,
                            args_start: 0,
                            argc: 0,
                        });
                        return Ok(rd);
                    }
                }
                // Check if this Ident is actually a state variable (unresolved
                // because the Module AST wasn't state-var-resolved before TIR lowering).
                if let Some(state_vars) = self.state_vars {
                    if let Some(&var_idx) = state_vars.get(&name_ref.name) {
                        let rd = self.alloc_reg()?;
                        if self.in_prime_context {
                            self.func.emit(Opcode::LoadPrime { rd, var_idx });
                        } else {
                            self.func.emit(Opcode::LoadVar { rd, var_idx });
                        }
                        return Ok(rd);
                    }
                }
                // Part of #3789: compile cross-module callee on-demand when
                // it wasn't pre-compiled during the fixed-point loop. This
                // avoids CallExternal fallback which requires EvalCtx at runtime.
                if let Some(callee_bodies) = self.callee_bodies {
                    if let Some(info) = callee_bodies.get(&resolved_name) {
                        if info.params.is_empty() {
                            let info_clone = info.clone();
                            match self.compile_callee_on_demand(&resolved_name, &info_clone) {
                                Ok(func_idx) => {
                                    let rd = self.alloc_reg()?;
                                    self.func.emit(Opcode::Call {
                                        rd,
                                        op_idx: func_idx,
                                        args_start: 0,
                                        argc: 0,
                                    });
                                    return Ok(rd);
                                }
                                Err(error) => {
                                    // On-demand compilation failed (e.g., recursive or
                                    // unsupported sub-expression). Fall back to CallExternal.
                                    let self_recursive = matches!(
                                        &error,
                                        CompileError::RecursiveOnDemand(target)
                                            if target == &resolved_name
                                                && self.func.name == resolved_name
                                                && self.func.arity == 0
                                    );
                                    let name_idx =
                                        self.add_const(Value::string(resolved_name.clone()))?;
                                    let rd = self.alloc_reg()?;
                                    self.func.emit(Opcode::CallExternal {
                                        rd,
                                        name_idx,
                                        args_start: 0,
                                        argc: 0,
                                        self_recursive,
                                    });
                                    return Ok(rd);
                                }
                            }
                        }
                    }
                }
                Err(CompileError::Unsupported(format!(
                    "unresolved identifier '{}'",
                    name_ref.name
                )))
            }
        }
    }

    fn inlineable_zero_arg_finite_domain_body(
        &self,
        resolved_name: &str,
    ) -> Option<std::sync::Arc<Spanned<TirExpr>>> {
        let callee_bodies = self.callee_bodies?;
        let info = callee_bodies.get(resolved_name)?;
        if !info.params.is_empty() || !self.is_inlineable_finite_domain_expr(&info.body) {
            return None;
        }
        Some(std::sync::Arc::clone(&info.body))
    }

    fn is_inlineable_finite_domain_expr(&self, expr: &Spanned<TirExpr>) -> bool {
        match &expr.node {
            TirExpr::Const { value, .. } => finite_domain_const_value(value),
            TirExpr::Name(name_ref) if matches!(name_ref.kind, TirNameKind::Ident) => self
                .resolved_constant_value(name_ref)
                .is_some_and(finite_domain_const_value),
            TirExpr::Range { lo, hi } => {
                self.is_inlineable_scalar_expr(lo) && self.is_inlineable_scalar_expr(hi)
            }
            TirExpr::SetEnum(elements) => elements
                .iter()
                .all(|element| self.is_inlineable_scalar_expr(element)),
            TirExpr::SetBinOp { left, right, .. } => {
                self.is_inlineable_finite_domain_expr(left)
                    && self.is_inlineable_finite_domain_expr(right)
            }
            TirExpr::Powerset(inner) => self.is_inlineable_finite_domain_expr(inner),
            TirExpr::KSubset { base, k } => {
                self.is_inlineable_finite_domain_expr(base) && self.is_inlineable_scalar_expr(k)
            }
            _ => false,
        }
    }

    fn is_inlineable_scalar_expr(&self, expr: &Spanned<TirExpr>) -> bool {
        match &expr.node {
            TirExpr::Const { value, .. } => scalar_const_value(value),
            TirExpr::Name(name_ref) if matches!(name_ref.kind, TirNameKind::Ident) => self
                .resolved_constant_value(name_ref)
                .is_some_and(scalar_const_value),
            _ => false,
        }
    }

    pub(super) fn resolved_constant_value(&self, name_ref: &TirNameRef) -> Option<&'a Value> {
        let resolved_constants = self.resolved_constants?;
        let lookup_id = if name_ref.name_id != tla_core::NameId::INVALID {
            Some(name_ref.name_id)
        } else {
            tla_core::name_intern::lookup_name_id(&name_ref.name)
        }?;
        resolved_constants.get(&lookup_id)
    }

    pub(super) fn compile_except_expr(
        &mut self,
        base: &Spanned<TirExpr>,
        specs: &[crate::nodes::TirExceptSpec],
    ) -> Result<Register, CompileError> {
        let mut r_func = self.compile_expr(base)?;
        for spec in specs {
            // For single-path EXCEPT, emit FuncExcept directly.
            if spec.path.len() == 1 {
                match &spec.path[0] {
                    TirExceptPathElement::Index(idx_expr) => {
                        let r_path = self.compile_expr(idx_expr)?;
                        let r_val = self.compile_except_rhs(r_func, r_path, &spec.value, true)?;
                        let rd = self.alloc_reg()?;
                        self.func.emit(Opcode::FuncExcept {
                            rd,
                            func: r_func,
                            path: r_path,
                            val: r_val,
                        });
                        r_func = rd;
                    }
                    TirExceptPathElement::Field(field) => {
                        let field_idx = self.constants.add_field_id(field.field_id.0);
                        let r_path = self.alloc_reg()?;
                        let idx = self.add_const(Value::String(field.name.clone().into()))?;
                        self.func.emit(Opcode::LoadConst { rd: r_path, idx });
                        // `FuncExcept` represents both Field and string-Index
                        // updates. Omitting @ here would erase the distinction:
                        // the tree-walker rejects Field on function values,
                        // while a string Index can be a valid/no-op update.
                        let r_val = self.compile_except_rhs(r_func, r_path, &spec.value, false)?;
                        let rd = self.alloc_reg()?;
                        self.func.emit(Opcode::FuncExcept {
                            rd,
                            func: r_func,
                            path: r_path,
                            val: r_val,
                        });
                        let _ = field_idx; // Used for future optimization
                        r_func = rd;
                    }
                }
            } else {
                // Multi-level EXCEPT: desugar to nested single-level.
                // [f EXCEPT ![a][b] = c] → [f EXCEPT ![a] = [f[a] EXCEPT ![b] = c]]
                let path_is_all_indices = spec
                    .path
                    .iter()
                    .all(|element| matches!(element, TirExceptPathElement::Index(_)));
                r_func = self.compile_multi_level_except(
                    r_func,
                    &spec.path,
                    &spec.value,
                    path_is_all_indices,
                )?;
            }
        }
        Ok(r_func)
    }

    /// Compile a multi-level EXCEPT path by recursive desugaring.
    ///
    /// `[f EXCEPT ![a][b] = c]` → `[f EXCEPT ![a] = [f[a] EXCEPT ![b] = c]]`
    ///
    /// The recursion peels one path element at a time. When only one element
    /// remains, we emit a single FuncExcept directly.
    fn compile_multi_level_except(
        &mut self,
        r_func: Register,
        path: &[TirExceptPathElement],
        value: &Spanned<TirExpr>,
        path_is_all_indices: bool,
    ) -> Result<Register, CompileError> {
        debug_assert!(path.len() >= 2);

        // Compile the first path element as a key.
        let r_key = self.compile_except_key(&path[0])?;

        // Get the inner value: f[a]
        let r_inner = self.alloc_reg()?;
        self.func.emit(Opcode::FuncApply {
            rd: r_inner,
            func: r_func,
            arg: r_key,
        });

        // Recursively compile the inner EXCEPT on remaining path elements.
        let r_inner_result = if path.len() == 2 {
            // Base case: single remaining element → direct FuncExcept.
            let r_inner_key = self.compile_except_key(&path[1])?;
            let r_val =
                self.compile_except_rhs(r_inner, r_inner_key, value, path_is_all_indices)?;
            let rd = self.alloc_reg()?;
            self.func.emit(Opcode::FuncExcept {
                rd,
                func: r_inner,
                path: r_inner_key,
                val: r_val,
            });
            rd
        } else {
            // Recursive case: more path elements remain.
            self.compile_multi_level_except(r_inner, &path[1..], value, path_is_all_indices)?
        };

        // Outer EXCEPT: [f EXCEPT ![a] = inner_result]
        let rd = self.alloc_reg()?;
        self.func.emit(Opcode::FuncExcept {
            rd,
            func: r_func,
            path: r_key,
            val: r_inner_result,
        });
        Ok(rd)
    }

    /// Compile an EXCEPT replacement, materializing `@` only when required.
    ///
    /// The optimized path is deliberately structural and fail-closed. It does
    /// not ask the general expression compiler whether an expression happens
    /// to lower without a call; it recognizes only the small set of values
    /// whose evaluation cannot observe the omitted function application.
    fn compile_except_rhs(
        &mut self,
        r_func: Register,
        r_path: Register,
        value: &Spanned<TirExpr>,
        path_is_all_indices: bool,
    ) -> Result<Register, CompileError> {
        if self.except_at_free_rhs && path_is_all_indices {
            if let Some(r_value) = self.try_compile_total_at_free_except_rhs(value)? {
                return Ok(r_value);
            }
        }

        // Preserve the established lowering for every refused expression:
        // compute @ = base[key], expose it while compiling the RHS, then
        // restore the enclosing EXCEPT context even if compilation fails.
        let r_at = self.alloc_reg()?;
        self.func.emit(Opcode::FuncApply {
            rd: r_at,
            func: r_func,
            arg: r_path,
        });
        let prev_at = self.except_at_register.replace(r_at);
        let result = self.compile_expr(value);
        self.except_at_register = prev_at;
        result
    }

    /// Compile a replacement proven to be independent of EXCEPT's `@` value.
    ///
    /// Accepted forms are limited to an existing bound register, a direct
    /// non-closure constant load, a direct state-variable load (under the VM's
    /// well-formed state-array invariant), or a resolved constant load.
    /// Everything else (including calls, applications, LET, lambdas/closures,
    /// external callbacks, and arithmetic) returns `None`.
    fn try_compile_total_at_free_except_rhs(
        &mut self,
        expr: &Spanned<TirExpr>,
    ) -> Result<Option<Register>, CompileError> {
        match &expr.node {
            TirExpr::Const { value, .. } if !matches!(value, Value::Closure(_)) => {
                self.compile_const(value).map(Some)
            }
            TirExpr::Name(name_ref) => {
                if let Some(reg) = self.lookup_binding(&name_ref.name) {
                    return Ok(Some(reg));
                }

                match &name_ref.kind {
                    TirNameKind::StateVar { index } => {
                        self.compile_direct_state_var_load(*index).map(Some)
                    }
                    TirNameKind::Ident => {
                        if let Some(value) = self.resolved_constant_value(name_ref) {
                            if matches!(value, Value::Closure(_)) {
                                return Ok(None);
                            }
                            // Drop the immutable borrow of the compiler's
                            // resolved-constant map before allocating/emitting.
                            let value = value.clone();
                            return self.compile_const(&value).map(Some);
                        }

                        // Unresolved module AST can leave a state variable as
                        // Ident. Accept it only if no name-resolution branch
                        // that precedes `state_vars` in `compile_name_expr`
                        // could turn it into a call, closure, inline body, or
                        // external callback.
                        let resolved_name = self.resolve_op_name(&name_ref.name);
                        if resolved_name != name_ref.name
                            || self.is_force_external(&name_ref.name, resolved_name)
                            || self.local_op_indices.contains_key(&name_ref.name)
                            || self
                                .op_indices
                                .is_some_and(|indices| indices.contains_key(resolved_name))
                            || self
                                .callee_bodies
                                .is_some_and(|bodies| bodies.contains_key(resolved_name))
                        {
                            return Ok(None);
                        }

                        let Some(index) = self
                            .state_vars
                            .and_then(|state_vars| state_vars.get(&name_ref.name))
                            .copied()
                        else {
                            return Ok(None);
                        };
                        self.compile_direct_state_var_load(index).map(Some)
                    }
                }
            }
            _ => Ok(None),
        }
    }

    fn compile_direct_state_var_load(&mut self, var_idx: u16) -> Result<Register, CompileError> {
        let rd = self.alloc_reg()?;
        if self.in_prime_context {
            self.func.emit(Opcode::LoadPrime { rd, var_idx });
        } else {
            self.func.emit(Opcode::LoadVar { rd, var_idx });
        }
        Ok(rd)
    }

    /// Compile a single EXCEPT path element (Index or Field) into a register.
    fn compile_except_key(
        &mut self,
        element: &TirExceptPathElement,
    ) -> Result<Register, CompileError> {
        match element {
            TirExceptPathElement::Index(idx_expr) => self.compile_expr(idx_expr),
            TirExceptPathElement::Field(field) => {
                let _field_idx = self.constants.add_field_id(field.field_id.0);
                let rd = self.alloc_reg()?;
                let idx = self.add_const(Value::String(field.name.clone().into()))?;
                self.func.emit(Opcode::LoadConst { rd, idx });
                Ok(rd)
            }
        }
    }

    pub(super) fn compile_record_expr(
        &mut self,
        fields: &[(crate::nodes::TirFieldName, Spanned<TirExpr>)],
    ) -> Result<Register, CompileError> {
        if fields.is_empty() {
            let rd = self.alloc_reg()?;
            self.func.emit(Opcode::RecordNew {
                rd,
                fields_start: 0,
                values_start: 0,
                count: 0,
            });
            return Ok(rd);
        }

        // Add field IDs to constant pool.
        let fields_start = self.constants.value_count() as u16;
        for (field_name, _) in fields {
            self.add_const(Value::String(field_name.name.clone().into()))?;
        }

        let values_start =
            self.compile_exprs_into_consecutive(fields.iter().map(|(_, expr)| expr))?;

        let count = fields.len().min(255) as u8;
        let rd = self.alloc_reg()?;
        self.func.emit(Opcode::RecordNew {
            rd,
            fields_start,
            values_start,
            count,
        });
        Ok(rd)
    }

    pub(super) fn compile_record_access_expr(
        &mut self,
        record: &Spanned<TirExpr>,
        field: &crate::nodes::TirFieldName,
    ) -> Result<Register, CompileError> {
        let rs = self.compile_expr(record)?;
        let field_idx = self.constants.add_field_id(field.field_id.0);
        let rd = self.alloc_reg()?;
        self.func.emit(Opcode::RecordGet { rd, rs, field_idx });
        Ok(rd)
    }

    pub(super) fn compile_record_set_expr(
        &mut self,
        fields: &[(crate::nodes::TirFieldName, Spanned<TirExpr>)],
    ) -> Result<Register, CompileError> {
        let fields_start = self.constants.value_count() as u16;
        for (field_name, _) in fields {
            self.add_const(Value::String(field_name.name.clone().into()))?;
        }
        let values_start =
            self.compile_exprs_into_consecutive(fields.iter().map(|(_, expr)| expr))?;
        let count = fields.len().min(255) as u8;
        let rd = self.alloc_reg()?;
        self.func.emit(Opcode::RecordSet {
            rd,
            fields_start,
            values_start,
            count,
        });
        Ok(rd)
    }

    pub(super) fn compile_prime_expr(
        &mut self,
        inner: &Spanned<TirExpr>,
    ) -> Result<Register, CompileError> {
        // If inner is a state variable name, use LoadPrime directly.
        if let TirExpr::Name(name_ref) = &inner.node {
            match &name_ref.kind {
                TirNameKind::StateVar { index } => {
                    let rd = self.alloc_reg()?;
                    self.func.emit(Opcode::LoadPrime {
                        rd,
                        var_idx: *index,
                    });
                    return Ok(rd);
                }
                TirNameKind::Ident => {
                    // Unresolved Ident — check if it's a state variable.
                    if let Some(state_vars) = self.state_vars {
                        if let Some(&var_idx) = state_vars.get(&name_ref.name) {
                            let rd = self.alloc_reg()?;
                            self.func.emit(Opcode::LoadPrime { rd, var_idx });
                            return Ok(rd);
                        }
                    }
                }
            }
        }
        // General case: compile inner expression in prime context so that
        // all state variable loads use LoadPrime (next-state) instead of LoadVar.
        // SetPrimeMode also needed for Call targets whose LoadVar opcodes are
        // resolved at runtime.
        self.func.emit(Opcode::SetPrimeMode { enable: true });
        let was_prime = self.in_prime_context;
        self.in_prime_context = true;
        let result = self.compile_expr(inner);
        self.in_prime_context = was_prime;
        self.func.emit(Opcode::SetPrimeMode { enable: false });
        result
    }

    pub(super) fn compile_unchanged_expr(
        &mut self,
        inner: &Spanned<TirExpr>,
    ) -> Result<Register, CompileError> {
        // Fast path: extract state variable indices directly and emit a
        // single Unchanged opcode that compares current vs next state slots.
        if let Ok(var_indices) = extract_unchanged_var_indices(
            inner,
            self.state_vars,
            self.callee_bodies,
            self.op_replacements,
        ) {
            let count = var_indices.len();
            if count > 0 && count <= 255 {
                let start = self.add_const(Value::SmallInt(var_indices[0] as i64))?;
                for &idx in &var_indices[1..] {
                    self.add_const(Value::SmallInt(idx as i64))?;
                }
                let rd = self.alloc_reg()?;
                self.func.emit(Opcode::Unchanged {
                    rd,
                    start,
                    count: count as u8,
                });
                return Ok(rd);
            }
        }

        // General fallback: UNCHANGED expr ≡ (expr = expr').
        // Compile inner in normal context (current state), then in prime
        // context (next state), and compare with Eq. Handles operator-defined
        // variable tuples like `UNCHANGED vars` where `vars == <<x, y>>`.
        //
        // Both compile-time (in_prime_context) and runtime (SetPrimeMode)
        // flags are used: in_prime_context redirects direct LoadVar→LoadPrime
        // at compile time, while SetPrimeMode causes the VM to redirect
        // LoadVar→next-state for calls to pre-compiled functions that use
        // LoadVar opcodes.
        let r_current = self.compile_expr(inner)?;
        self.func.emit(Opcode::SetPrimeMode { enable: true });
        let was_prime = self.in_prime_context;
        self.in_prime_context = true;
        let r_prime = self.compile_expr(inner)?;
        self.in_prime_context = was_prime;
        self.func.emit(Opcode::SetPrimeMode { enable: false });
        let rd = self.alloc_reg()?;
        self.func.emit(Opcode::Eq {
            rd,
            r1: r_current,
            r2: r_prime,
        });
        Ok(rd)
    }
}

fn scalar_const_value(value: &Value) -> bool {
    matches!(
        value,
        Value::Bool(_)
            | Value::SmallInt(_)
            | Value::Int(_)
            | Value::String(_)
            | Value::ModelValue(_)
    )
}

fn finite_domain_const_value(value: &Value) -> bool {
    matches!(
        value,
        Value::Set(_) | Value::Interval(_) | Value::Subset(_) | Value::KSubset(_)
    )
}

/// Extract state variable indices from an UNCHANGED inner expression.
///
/// Supports:
/// - Single state variable: `UNCHANGED x` → `[idx]`
/// - Tuple of state variables: `UNCHANGED <<x, y>>` → `[idx_x, idx_y]`
///
/// When `state_vars` is provided, unresolved `Ident` names are checked
/// against the state variable map as a fallback (for TIR bodies lowered
/// from raw Module AST without prior state-var resolution).
///
/// Returns `Err(Unsupported)` for complex inner expressions.
fn extract_unchanged_var_indices(
    inner: &Spanned<TirExpr>,
    state_vars: Option<&std::collections::HashMap<String, u16>>,
    callee_bodies: Option<&std::collections::HashMap<String, CalleeInfo>>,
    op_replacements: Option<&std::collections::HashMap<String, String>>,
) -> Result<Vec<u16>, CompileError> {
    extract_unchanged_var_indices_inner(
        inner,
        state_vars,
        callee_bodies,
        op_replacements,
        &mut Vec::new(),
    )
}

fn extract_unchanged_var_indices_inner(
    inner: &Spanned<TirExpr>,
    state_vars: Option<&std::collections::HashMap<String, u16>>,
    callee_bodies: Option<&std::collections::HashMap<String, CalleeInfo>>,
    op_replacements: Option<&std::collections::HashMap<String, String>>,
    resolving_aliases: &mut Vec<String>,
) -> Result<Vec<u16>, CompileError> {
    match &inner.node {
        TirExpr::Name(name_ref) => match &name_ref.kind {
            TirNameKind::StateVar { index } => Ok(vec![*index]),
            TirNameKind::Ident => {
                if let Some(var_idx) =
                    replacement_state_var_index(&name_ref.name, state_vars, op_replacements)
                {
                    return Ok(vec![var_idx]);
                }
                if let Some(sv) = state_vars {
                    if let Some(&var_idx) = sv.get(&name_ref.name) {
                        return Ok(vec![var_idx]);
                    }
                }
                if let Some(var_indices) = extract_zero_arg_alias_var_indices(
                    &name_ref.name,
                    state_vars,
                    callee_bodies,
                    op_replacements,
                    resolving_aliases,
                )? {
                    return Ok(var_indices);
                }
                Err(CompileError::Unsupported(
                    "UNCHANGED on non-state-variable identifier".to_string(),
                ))
            }
        },
        TirExpr::Tuple(elems) => {
            let mut indices = Vec::with_capacity(elems.len());
            for elem in elems {
                match &elem.node {
                    TirExpr::Name(name_ref) => match &name_ref.kind {
                        TirNameKind::StateVar { index } => indices.push(*index),
                        TirNameKind::Ident => {
                            if let Some(var_idx) = replacement_state_var_index(
                                &name_ref.name,
                                state_vars,
                                op_replacements,
                            ) {
                                indices.push(var_idx);
                                continue;
                            }
                            if let Some(sv) = state_vars {
                                if let Some(&var_idx) = sv.get(&name_ref.name) {
                                    indices.push(var_idx);
                                    continue;
                                }
                            }
                            return Err(CompileError::Unsupported(
                                "UNCHANGED tuple element is not a state variable".to_string(),
                            ));
                        }
                    },
                    _ => {
                        return Err(CompileError::Unsupported(
                            "UNCHANGED tuple element is not a simple variable".to_string(),
                        ));
                    }
                }
            }
            Ok(indices)
        }
        _ => Err(CompileError::Unsupported(
            "UNCHANGED on complex expression".to_string(),
        )),
    }
}

fn replacement_state_var_index(
    name: &str,
    state_vars: Option<&std::collections::HashMap<String, u16>>,
    op_replacements: Option<&std::collections::HashMap<String, String>>,
) -> Option<u16> {
    let state_vars = state_vars?;
    op_replacements
        .and_then(|replacements| replacements.get(name))
        .and_then(|replacement| state_vars.get(replacement).copied())
}

fn extract_zero_arg_alias_var_indices(
    name: &str,
    state_vars: Option<&std::collections::HashMap<String, u16>>,
    callee_bodies: Option<&std::collections::HashMap<String, CalleeInfo>>,
    op_replacements: Option<&std::collections::HashMap<String, String>>,
    resolving_aliases: &mut Vec<String>,
) -> Result<Option<Vec<u16>>, CompileError> {
    let Some(callee_bodies) = callee_bodies else {
        return Ok(None);
    };

    let resolved_name = op_replacements
        .and_then(|replacements| replacements.get(name))
        .map(String::as_str)
        .unwrap_or(name);

    if let Some(var_indices) = extract_named_zero_arg_alias_var_indices(
        resolved_name,
        state_vars,
        callee_bodies,
        op_replacements,
        resolving_aliases,
    )? {
        return Ok(Some(var_indices));
    }

    if resolved_name != name {
        return extract_named_zero_arg_alias_var_indices(
            name,
            state_vars,
            callee_bodies,
            op_replacements,
            resolving_aliases,
        );
    }

    Ok(None)
}

fn extract_named_zero_arg_alias_var_indices(
    name: &str,
    state_vars: Option<&std::collections::HashMap<String, u16>>,
    callee_bodies: &std::collections::HashMap<String, CalleeInfo>,
    op_replacements: Option<&std::collections::HashMap<String, String>>,
    resolving_aliases: &mut Vec<String>,
) -> Result<Option<Vec<u16>>, CompileError> {
    let Some(info) = callee_bodies.get(name) else {
        return Ok(None);
    };
    if !info.params.is_empty() {
        return Ok(None);
    }
    if resolving_aliases.iter().any(|alias| alias == name) {
        return Err(CompileError::Unsupported(format!(
            "cyclic UNCHANGED alias '{name}'"
        )));
    }

    resolving_aliases.push(name.to_string());
    let result = extract_unchanged_var_indices_inner(
        &info.body,
        state_vars,
        Some(callee_bodies),
        op_replacements,
        resolving_aliases,
    );
    resolving_aliases.pop();
    result.map(Some)
}
