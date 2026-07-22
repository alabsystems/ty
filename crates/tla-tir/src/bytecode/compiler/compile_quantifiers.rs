// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Quantifier and set comprehension compilation helpers.

use super::super::opcode::{Opcode, Register};
use super::{CompileError, FnCompileState};
use crate::nodes::{
    TirArithOp, TirBoundPattern, TirBoundVar, TirCmpOp, TirExpr, TirNameKind, TirNameRef,
};
use tla_core::Spanned;

/// Fully-proved inputs for the one SetFilter projection hoist.
#[derive(Clone, Copy)]
struct SetFilterProjectionHoist {
    r_outer: Register,
    r_arg: Register,
    projection_index: i64,
}

impl<'a> FnCompileState<'a> {
    /// Match the complete one-binder predicate
    ///
    /// `Round(child) = Round(parent) - 1`
    ///
    /// where `child` is exactly the current binder, `parent` is a direct
    /// outer register binding, and both calls name the same unreplaced global
    /// operator whose complete body is the certified Round shape. The exact
    /// orientation preserves the ordinary child-call, parent-call,
    /// subtraction, equality evaluation order.
    fn match_round_step_eq(
        &self,
        vars: &[TirBoundVar],
        body: &Spanned<TirExpr>,
    ) -> Option<Register> {
        if !self.round_step_eq || self.in_prime_context {
            return None;
        }
        let [var] = vars else {
            return None;
        };
        if var.pattern.is_some() {
            return None;
        }

        let TirExpr::Cmp {
            left: child_call,
            op: TirCmpOp::Eq,
            right,
        } = &body.node
        else {
            return None;
        };
        let TirExpr::ArithBinOp {
            left: parent_call,
            op: TirArithOp::Sub,
            right: one,
        } = &right.node
        else {
            return None;
        };
        if !matches!(
            &one.node,
            TirExpr::Const {
                value: tla_value::Value::SmallInt(1),
                ..
            }
        ) {
            return None;
        }

        let (child_callee, child_arg) = exact_named_unary_call(child_call)?;
        if child_callee.name == var.name || !is_exact_binder_name(child_arg, var) {
            return None;
        }
        let (parent_callee, parent_arg) = exact_named_unary_call(parent_call)?;
        if parent_callee.name != child_callee.name
            || !self.is_direct_complete_round_callee(parent_callee)
        {
            return None;
        }
        let TirExpr::Name(parent_name) = &parent_arg.node else {
            return None;
        };
        if !matches!(parent_name.kind, TirNameKind::Ident) || parent_name.name == var.name {
            return None;
        }
        let r_parent = self.lookup_binding(&parent_name.name)?;

        if !self.is_direct_complete_round_callee(child_callee) {
            return None;
        }
        Some(r_parent)
    }

    /// Prove that a call target has the same precedence and complete body as
    /// an ordinary direct global Round call.
    fn is_direct_complete_round_callee(&self, callee: &TirNameRef) -> bool {
        if !matches!(callee.kind, TirNameKind::Ident)
            || self.lookup_binding(&callee.name).is_some()
            || self.local_op_indices.contains_key(&callee.name)
            || self
                .state_vars
                .is_some_and(|state_vars| state_vars.contains_key(&callee.name))
            || super::compile_control::fixed_builtin_call(&callee.name).is_some()
            || self
                .op_replacements
                .is_some_and(|replacements| replacements.contains_key(&callee.name))
        {
            return false;
        }
        let resolved = self.resolve_op_name(&callee.name);
        if resolved != callee.name.as_str() || self.is_force_external(&callee.name, resolved) {
            return false;
        }
        let Some(info) = self.callee_bodies.and_then(|callees| callees.get(resolved)) else {
            return false;
        };
        let [param] = info.params.as_slice() else {
            return false;
        };
        super::compile_control::is_round_shape_body(&info.body, param)
    }

    /// Match exactly `{c \in D : <<outer, c>> \in S(arg)}` with
    /// `S(p) == p[k]`.
    ///
    /// Both free operands must already be register bindings before `c` is
    /// installed. This excludes state loads, constants, calls, LET operators,
    /// shadowing, and every expression that could change or error per loop
    /// iteration. The callee proof likewise accepts only the complete direct
    /// global body `p[k]`; no inlining or recursive purity analysis is used.
    fn match_set_filter_projection_hoist(
        &self,
        var: &TirBoundVar,
        body: &Spanned<TirExpr>,
    ) -> Option<SetFilterProjectionHoist> {
        if !self.set_filter_projection_hoist || !self.tuple2_set_in {
            return None;
        }

        let binder_name = match &var.pattern {
            Some(TirBoundPattern::Tuple(_)) => return None,
            Some(TirBoundPattern::Var(name, _)) => name.as_str(),
            None => var.name.as_str(),
        };

        let TirExpr::In { elem, set } = &body.node else {
            return None;
        };
        let TirExpr::Tuple(elements) = &elem.node else {
            return None;
        };
        let [outer_expr, binder_expr] = elements.as_slice() else {
            return None;
        };
        let TirExpr::Name(outer) = &outer_expr.node else {
            return None;
        };
        let TirExpr::Name(binder) = &binder_expr.node else {
            return None;
        };
        if !matches!(outer.kind, TirNameKind::Ident)
            || !matches!(binder.kind, TirNameKind::Ident)
            || binder.name != binder_name
            || outer.name == binder_name
        {
            return None;
        }
        let r_outer = self.lookup_binding(&outer.name)?;

        let TirExpr::Apply { op, args } = &set.node else {
            return None;
        };
        let [arg_expr] = args.as_slice() else {
            return None;
        };
        let TirExpr::Name(arg) = &arg_expr.node else {
            return None;
        };
        if !matches!(arg.kind, TirNameKind::Ident) || arg.name == binder_name {
            return None;
        }
        let r_arg = self.lookup_binding(&arg.name)?;

        let TirExpr::Name(callee) = &op.node else {
            return None;
        };
        if !matches!(callee.kind, TirNameKind::Ident)
            || callee.name == binder_name
            || self.lookup_binding(&callee.name).is_some()
            || self.local_op_indices.contains_key(&callee.name)
            || super::compile_control::fixed_builtin_call(&callee.name)
                .is_some_and(|(_, arity)| arity == 1)
        {
            return None;
        }
        // Config replacement can redirect the historical Call to a different
        // definition. Refuse rather than trying to prove replacement chains.
        let resolved = self.resolve_op_name(&callee.name);
        if resolved != callee.name.as_str() || self.is_force_external(&callee.name, resolved) {
            return None;
        }

        let info = self.callee_bodies?.get(resolved)?;
        let [param] = info.params.as_slice() else {
            return None;
        };
        let TirExpr::FuncApply {
            func,
            arg: projection,
        } = &info.body.node
        else {
            return None;
        };
        let TirExpr::Name(param_ref) = &func.node else {
            return None;
        };
        if !matches!(param_ref.kind, TirNameKind::Ident)
            || param_ref.name.as_str() != param.as_str()
        {
            return None;
        }
        let TirExpr::Const {
            value: tla_value::Value::SmallInt(projection_index),
            ..
        } = &projection.node
        else {
            return None;
        };

        Some(SetFilterProjectionHoist {
            r_outer,
            r_arg,
            projection_index: *projection_index,
        })
    }

    /// Push the binding(s) for a bound variable whose current element is held
    /// in `r_binding`, returning the number of `self.bindings` entries pushed
    /// (so callers can `truncate`/`pop` the exact count afterwards).
    ///
    /// For a plain binder this pushes a single `name -> r_binding` mapping.
    /// For a tuple-destructuring binder `<<a, b, ...>> \in S` it emits a
    /// `FuncApply` per component (TLA+ tuples are 1-indexed) extracting
    /// `r_binding[i]` into a fresh register, then pushes one
    /// `component_name -> r_component` mapping per component. This mirrors the
    /// element destructuring already performed for multi-variable `FuncDef`
    /// (`compile_func_def`) and matches the interpreter's tuple-pattern
    /// semantics in `tla-eval` (`push_tir_bound_var`).
    ///
    /// The fix is purely structural: it is driven by the binder carrying a
    /// `TirBoundPattern::Tuple`, never by any spec/action/variable name.
    fn push_bound_var_bindings(
        &mut self,
        var: &TirBoundVar,
        r_binding: Register,
    ) -> Result<usize, CompileError> {
        match &var.pattern {
            Some(TirBoundPattern::Tuple(components)) => {
                for (i, (name, _name_id)) in components.iter().enumerate() {
                    let r_idx = self.alloc_reg()?;
                    // TLA+ tuples are 1-indexed.
                    self.func.emit(Opcode::LoadImm {
                        rd: r_idx,
                        value: (i as i64) + 1,
                    });
                    let r_elem = self.alloc_reg()?;
                    self.func.emit(Opcode::FuncApply {
                        rd: r_elem,
                        func: r_binding,
                        arg: r_idx,
                    });
                    self.bindings.push((name.clone(), r_elem));
                }
                Ok(components.len())
            }
            Some(TirBoundPattern::Var(name, _name_id)) => {
                self.bindings.push((name.clone(), r_binding));
                Ok(1)
            }
            None => {
                self.bindings.push((var.name.clone(), r_binding));
                Ok(1)
            }
        }
    }

    /// Compile FORALL quantifier.
    ///
    /// Multi-variable `\A x \in S, y \in T: P(x, y)` is desugared to
    /// `\A x \in S: (\A y \in T: P(x, y))` via recursive nesting.
    pub(super) fn compile_forall(
        &mut self,
        vars: &[TirBoundVar],
        body: &Spanned<TirExpr>,
    ) -> Result<Register, CompileError> {
        self.compile_forall_nested(vars, body, vars.len() == 1)
    }

    fn compile_forall_nested(
        &mut self,
        vars: &[TirBoundVar],
        body: &Spanned<TirExpr>,
        allow_single_binder_fusions: bool,
    ) -> Result<Register, CompileError> {
        // Match before emitting the domain or installing the binder so every
        // refusal follows byte-for-byte historical compilation.
        let round_step_eq = allow_single_binder_fusions
            .then(|| self.match_round_step_eq(vars, body))
            .flatten();
        let var = &vars[0];
        let domain_expr = var
            .domain
            .as_ref()
            .ok_or_else(|| CompileError::Unsupported("FORALL without domain".to_string()))?;

        let r_domain = self.compile_expr(domain_expr)?;
        let rd = self.alloc_reg()?;
        let r_binding = self.alloc_reg()?;

        let begin_idx = self.func.emit(Opcode::ForallBegin {
            rd,
            r_binding,
            r_domain,
            loop_end: 0, // patched
        });

        // Push binding(s) for this variable (tuple patterns destructure into
        // one register per component).
        let pushed = self.push_bound_var_bindings(var, r_binding)?;
        // Loop-body temporaries are recomputed every iteration; recycle them
        // once the loop is closed (reg_recycle only).
        let body_cp = self.reg_checkpoint();
        // If more variables remain, nest; otherwise compile the body.
        let r_body = if let Some(parent) = round_step_eq {
            debug_assert_eq!(vars.len(), 1);
            debug_assert_eq!(self.lookup_binding(&var.name), Some(r_binding));
            let r_body = self.alloc_reg()?;
            self.func.emit(Opcode::RoundStepEq {
                rd: r_body,
                child: r_binding,
                parent,
            });
            r_body
        } else if vars.len() > 1 {
            self.compile_forall_nested(&vars[1..], body, false)?
        } else {
            self.compile_expr(body)?
        };
        let keep = self.bindings.len() - pushed;
        self.bindings.truncate(keep);

        let next_idx = self.func.emit(Opcode::ForallNext {
            rd,
            r_binding,
            r_body,
            loop_begin: 0, // patched
        });

        let end = self.func.len();
        self.func.patch_jump(begin_idx, end);
        self.func.patch_jump(next_idx, begin_idx + 1);
        self.reg_rollback(body_cp);

        Ok(rd)
    }

    /// Compile EXISTS quantifier.
    ///
    /// Multi-variable `\E x \in S, y \in T: P(x, y)` is desugared to
    /// `\E x \in S: (\E y \in T: P(x, y))` via recursive nesting.
    pub(super) fn compile_exists(
        &mut self,
        vars: &[TirBoundVar],
        body: &Spanned<TirExpr>,
    ) -> Result<Register, CompileError> {
        self.compile_exists_nested(vars, body)
    }

    fn compile_exists_nested(
        &mut self,
        vars: &[TirBoundVar],
        body: &Spanned<TirExpr>,
    ) -> Result<Register, CompileError> {
        let var = &vars[0];
        let domain_expr = var
            .domain
            .as_ref()
            .ok_or_else(|| CompileError::Unsupported("EXISTS without domain".to_string()))?;

        let r_domain = self.compile_expr(domain_expr)?;
        let rd = self.alloc_reg()?;
        let r_binding = self.alloc_reg()?;

        let begin_idx = self.func.emit(Opcode::ExistsBegin {
            rd,
            r_binding,
            r_domain,
            loop_end: 0, // patched
        });

        let pushed = self.push_bound_var_bindings(var, r_binding)?;
        // Loop-body temporaries are recomputed every iteration; recycle them
        // once the loop is closed (reg_recycle only).
        let body_cp = self.reg_checkpoint();
        let r_body = if vars.len() > 1 {
            self.compile_exists_nested(&vars[1..], body)?
        } else {
            self.compile_expr(body)?
        };
        let keep = self.bindings.len() - pushed;
        self.bindings.truncate(keep);

        let next_idx = self.func.emit(Opcode::ExistsNext {
            rd,
            r_binding,
            r_body,
            loop_begin: 0, // patched
        });

        let end = self.func.len();
        self.func.patch_jump(begin_idx, end);
        self.func.patch_jump(next_idx, begin_idx + 1);
        self.reg_rollback(body_cp);

        Ok(rd)
    }

    /// Compile CHOOSE expression: `CHOOSE x \in S : P(x)`.
    ///
    /// Iterates the domain, evaluates the predicate for each element,
    /// and returns the first element where the predicate is TRUE.
    /// If no element satisfies P, halts (TLA+ runtime error).
    pub(super) fn compile_choose(
        &mut self,
        var: &TirBoundVar,
        body: &Spanned<TirExpr>,
    ) -> Result<Register, CompileError> {
        let domain_expr = var
            .domain
            .as_ref()
            .ok_or_else(|| CompileError::Unsupported("CHOOSE without domain".to_string()))?;

        let r_domain = self.compile_expr(domain_expr)?;
        let rd = self.alloc_reg()?;
        let r_binding = self.alloc_reg()?;

        let begin_idx = self.func.emit(Opcode::ChooseBegin {
            rd,
            r_binding,
            r_domain,
            loop_end: 0, // patched
        });

        let pushed = self.push_bound_var_bindings(var, r_binding)?;
        // Loop-body temporaries are recomputed every iteration; recycle them
        // once the loop is closed (reg_recycle only).
        let body_cp = self.reg_checkpoint();
        let r_body = self.compile_expr(body)?;
        let keep = self.bindings.len() - pushed;
        self.bindings.truncate(keep);

        let next_idx = self.func.emit(Opcode::ChooseNext {
            rd,
            r_binding,
            r_body,
            loop_begin: 0, // patched
        });

        let end = self.func.len();
        self.func.patch_jump(begin_idx, end);
        self.func.patch_jump(next_idx, begin_idx + 1);
        self.reg_rollback(body_cp);

        Ok(rd)
    }

    /// Compile set filter: `{x \in S : P(x)}`.
    pub(super) fn compile_set_filter(
        &mut self,
        var: &TirBoundVar,
        body: &Spanned<TirExpr>,
    ) -> Result<Register, CompileError> {
        // Match before emitting anything so every refusal falls through to
        // byte-for-byte historical compilation.
        let projection_hoist = self.match_set_filter_projection_hoist(var, body);
        let domain_expr = var
            .domain
            .as_ref()
            .ok_or_else(|| CompileError::Unsupported("SetFilter without domain".to_string()))?;

        let r_domain = self.compile_expr(domain_expr)?;
        let rd = self.alloc_reg()?;
        let r_binding = self.alloc_reg()?;

        let begin_idx = self.func.emit(Opcode::SetFilterBegin {
            rd,
            r_binding,
            r_domain,
            loop_end: 0,
        });

        let pushed = self.push_bound_var_bindings(var, r_binding)?;
        // Loop-body temporaries are recomputed every iteration; recycle them
        // once the loop is closed (reg_recycle only).
        let body_cp = self.reg_checkpoint();
        let (r_body, loop_body_idx) = if let Some(projection_hoist) = projection_hoist {
            // SetFilterBegin validates/enumerates the domain, skips this
            // preheader for an empty domain, and installs the first binder.
            // Only then is the historically per-iteration `S(arg) == arg[k]`
            // projection evaluated. Subsequent LoopNext iterations jump past
            // these two instructions directly to Tuple2SetIn.
            let r_index = self.alloc_reg()?;
            self.func.emit(Opcode::LoadImm {
                rd: r_index,
                value: projection_hoist.projection_index,
            });
            let r_set = self.alloc_reg()?;
            self.func.emit(Opcode::FuncApply {
                rd: r_set,
                func: projection_hoist.r_arg,
                arg: r_index,
            });
            let loop_body_idx = self.func.len();
            let r_body = self.alloc_reg()?;
            self.func.emit(Opcode::Tuple2SetIn {
                rd: r_body,
                first: projection_hoist.r_outer,
                second: r_binding,
                set: r_set,
            });
            (r_body, loop_body_idx)
        } else {
            (self.compile_expr(body)?, begin_idx + 1)
        };
        let keep = self.bindings.len() - pushed;
        self.bindings.truncate(keep);

        let next_idx = self.func.emit(Opcode::LoopNext {
            r_binding,
            r_body,
            loop_begin: 0,
        });

        let end = self.func.len();
        self.func.patch_jump(begin_idx, end);
        self.func.patch_jump(next_idx, loop_body_idx);
        self.reg_rollback(body_cp);

        Ok(rd)
    }

    /// Compile set builder: `{e : x \in S}` or `{e : x \in S, y \in T}`.
    ///
    /// Multi-variable SetBuilder is desugared via UNION (BigUnion) to flatten
    /// nested iteration into a flat set:
    /// `{e : x \in S, y \in T}` → `UNION {{e : y \in T} : x \in S}`
    pub(super) fn compile_set_builder(
        &mut self,
        body: &Spanned<TirExpr>,
        vars: &[TirBoundVar],
    ) -> Result<Register, CompileError> {
        if vars.len() == 1 {
            return self.compile_set_builder_single(body, &vars[0]);
        }
        // Multi-variable: peel first var, recurse on rest, flatten with BigUnion.
        // Outer set builder iterates x ∈ S, body = {e : y ∈ T, ...}
        let var = &vars[0];
        let domain_expr = var
            .domain
            .as_ref()
            .ok_or_else(|| CompileError::Unsupported("SetBuilder without domain".to_string()))?;

        let r_domain = self.compile_expr(domain_expr)?;
        let r_outer = self.alloc_reg()?;
        let r_binding = self.alloc_reg()?;

        let begin_idx = self.func.emit(Opcode::SetBuilderBegin {
            rd: r_outer,
            r_binding,
            r_domain,
            loop_end: 0,
        });

        let pushed = self.push_bound_var_bindings(var, r_binding)?;
        // Loop-body temporaries are recomputed every iteration; recycle them
        // once the loop is closed (reg_recycle only).
        let body_cp = self.reg_checkpoint();
        // Inner set builder produces {e : remaining vars}
        let r_inner_set = self.compile_set_builder(body, &vars[1..])?;
        let keep = self.bindings.len() - pushed;
        self.bindings.truncate(keep);

        let next_idx = self.func.emit(Opcode::LoopNext {
            r_binding,
            r_body: r_inner_set,
            loop_begin: 0,
        });

        let end = self.func.len();
        self.func.patch_jump(begin_idx, end);
        self.func.patch_jump(next_idx, begin_idx + 1);
        self.reg_rollback(body_cp);

        // Flatten: UNION { {e : y ∈ T, ...} : x ∈ S }
        let rd = self.alloc_reg()?;
        self.func.emit(Opcode::BigUnion { rd, rs: r_outer });
        Ok(rd)
    }

    /// Compile a single-variable set builder: `{e : x \in S}`.
    fn compile_set_builder_single(
        &mut self,
        body: &Spanned<TirExpr>,
        var: &TirBoundVar,
    ) -> Result<Register, CompileError> {
        let domain_expr = var
            .domain
            .as_ref()
            .ok_or_else(|| CompileError::Unsupported("SetBuilder without domain".to_string()))?;

        let r_domain = self.compile_expr(domain_expr)?;
        let rd = self.alloc_reg()?;
        let r_binding = self.alloc_reg()?;

        let begin_idx = self.func.emit(Opcode::SetBuilderBegin {
            rd,
            r_binding,
            r_domain,
            loop_end: 0,
        });

        let pushed = self.push_bound_var_bindings(var, r_binding)?;
        // Loop-body temporaries are recomputed every iteration; recycle them
        // once the loop is closed (reg_recycle only).
        let body_cp = self.reg_checkpoint();
        let r_body = self.compile_expr(body)?;
        let keep = self.bindings.len() - pushed;
        self.bindings.truncate(keep);

        let next_idx = self.func.emit(Opcode::LoopNext {
            r_binding,
            r_body,
            loop_begin: 0,
        });

        let end = self.func.len();
        self.func.patch_jump(begin_idx, end);
        self.func.patch_jump(next_idx, begin_idx + 1);
        self.reg_rollback(body_cp);

        Ok(rd)
    }

    /// Compile function definition: `[x \in S |-> e]` or `[x \in S, y \in T |-> e]`.
    ///
    /// Multi-variable FuncDef is desugared to a tuple-domain function:
    /// `[x \in S, y \in T |-> e]` → `[t \in S \X T |-> LET x == t[1], y == t[2] IN e]`
    pub(super) fn compile_func_def(
        &mut self,
        vars: &[TirBoundVar],
        body: &Spanned<TirExpr>,
    ) -> Result<Register, CompileError> {
        if vars.len() == 1 {
            return self.compile_func_def_single(&vars[0], body);
        }

        // Multi-variable: compute cross-product domain S × T × ...
        let mut domain_regs = Vec::with_capacity(vars.len());
        for var in vars {
            let domain_expr = var
                .domain
                .as_ref()
                .ok_or_else(|| CompileError::Unsupported("FuncDef without domain".to_string()))?;
            domain_regs.push(self.compile_expr(domain_expr)?);
        }

        // Emit Times to compute the cross product.
        let times_start = self.next_reg;
        let count = vars.len().min(255) as u8;
        for &r in &domain_regs {
            let slot = self.alloc_reg()?;
            if r != slot {
                self.func.emit(Opcode::Move { rd: slot, rs: r });
            }
        }
        let r_domain = self.alloc_reg()?;
        self.func.emit(Opcode::Times {
            rd: r_domain,
            start: times_start,
            count,
        });

        // Iterate over the cross-product domain.
        let rd = self.alloc_reg()?;
        let r_binding = self.alloc_reg()?;

        let begin_idx = self.func.emit(Opcode::FuncDefBegin {
            rd,
            r_binding,
            r_domain,
            loop_end: 0,
        });

        // Destructure tuple: bind x = t[1], y = t[2], etc.
        let saved_bindings = self.bindings.len();
        for (i, var) in vars.iter().enumerate() {
            let r_idx = self.alloc_reg()?;
            // TLA+ tuples are 1-indexed.
            self.func.emit(Opcode::LoadImm {
                rd: r_idx,
                value: (i as i64) + 1,
            });
            let r_elem = self.alloc_reg()?;
            self.func.emit(Opcode::FuncApply {
                rd: r_elem,
                func: r_binding,
                arg: r_idx,
            });
            self.bindings.push((var.name.clone(), r_elem));
        }

        // Loop-body temporaries are recomputed every iteration; recycle them
        // once the loop is closed (reg_recycle only).
        let body_cp = self.reg_checkpoint();
        let r_body = self.compile_expr(body)?;

        // Pop all tuple-destructuring bindings.
        self.bindings.truncate(saved_bindings);

        let next_idx = self.func.emit(Opcode::LoopNext {
            r_binding,
            r_body,
            loop_begin: 0,
        });

        let end = self.func.len();
        self.func.patch_jump(begin_idx, end);
        self.func.patch_jump(next_idx, begin_idx + 1);
        self.reg_rollback(body_cp);

        Ok(rd)
    }

    /// Compile a single-variable function definition: `[x \in S |-> e]`.
    fn compile_func_def_single(
        &mut self,
        var: &TirBoundVar,
        body: &Spanned<TirExpr>,
    ) -> Result<Register, CompileError> {
        let domain_expr = var
            .domain
            .as_ref()
            .ok_or_else(|| CompileError::Unsupported("FuncDef without domain".to_string()))?;

        let r_domain = self.compile_expr(domain_expr)?;
        let rd = self.alloc_reg()?;
        let r_binding = self.alloc_reg()?;

        let begin_idx = self.func.emit(Opcode::FuncDefBegin {
            rd,
            r_binding,
            r_domain,
            loop_end: 0,
        });

        // Bind the binder to r_binding, destructuring a tuple pattern
        // (`f[<<x,y>> \in D] == ...`) into per-component bindings exactly as the
        // multi-var FuncDef path and the quantifier/set-builder binders do.
        // Previously this pushed only `var.name`, so a single tuple-pattern
        // binder left `x`/`y` unresolved and the FuncDef failed to compile to
        // bytecode (falling back to the tree-walker).
        let pushed = self.push_bound_var_bindings(var, r_binding)?;
        // Loop-body temporaries are recomputed every iteration; recycle them
        // once the loop is closed (reg_recycle only).
        let body_cp = self.reg_checkpoint();
        let r_body = self.compile_expr(body)?;
        for _ in 0..pushed {
            self.bindings.pop();
        }

        let next_idx = self.func.emit(Opcode::LoopNext {
            r_binding,
            r_body,
            loop_begin: 0,
        });

        let end = self.func.len();
        self.func.patch_jump(begin_idx, end);
        self.func.patch_jump(next_idx, begin_idx + 1);
        self.reg_rollback(body_cp);

        Ok(rd)
    }
}

fn is_exact_binder_name(expr: &Spanned<TirExpr>, var: &TirBoundVar) -> bool {
    let TirExpr::Name(name) = &expr.node else {
        return false;
    };
    matches!(name.kind, TirNameKind::Ident) && name.name == var.name && name.name_id == var.name_id
}

fn exact_named_unary_call(expr: &Spanned<TirExpr>) -> Option<(&TirNameRef, &Spanned<TirExpr>)> {
    let TirExpr::Apply { op, args } = &expr.node else {
        return None;
    };
    let [arg] = args.as_slice() else {
        return None;
    };
    let TirExpr::Name(callee) = &op.node else {
        return None;
    };
    Some((callee, arg))
}
