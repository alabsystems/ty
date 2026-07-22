// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Scalar setup helpers for model checker preparation.
//!
//! These leaf operations are called through the stable `crate::checker_ops::*`
//! surface by both sequential and parallel model checkers.

#[cfg(debug_assertions)]
use crate::check::debug::ty_debug;
use crate::check::{format_span_location, CheckError, INLINE_NEXT_NAME};
use crate::config::Config;
use crate::enumerate::{expand_operators, expand_operators_with_primes};
use crate::eval::EvalCtx;
use crate::{ConfigCheckError, EvalCheckError, RuntimeCheckError};
use tla_core::ast::{Expr, OperatorDef};
use tla_core::{Spanned, SyntaxNode, VarRegistry};

/// Validate the VIEW operator: must exist and have zero parameters.
///
/// Returns `Some(name)` if the VIEW operator is configured and valid,
/// `None` if not configured or the operator is missing/invalid.
///
/// Both sequential (`run_prepare.rs:193-224`) and parallel (`prepare.rs:166-194`)
/// checkers previously duplicated this exact logic. Extracting it prevents
/// silent divergence if validation rules change.
pub(crate) fn validate_view_operator(ctx: &EvalCtx, config: &Config) -> Option<String> {
    let view_name = config.view.as_ref()?;

    match ctx.get_op(view_name) {
        Some(def) => {
            if !def.params.is_empty() {
                eprintln!(
                    "Warning: VIEW operator '{}' has {} parameters, expected 0. \
                     Using full state fingerprints",
                    view_name,
                    def.params.len()
                );
                None
            } else {
                debug_eprintln!(
                    ty_debug(),
                    "VIEW enabled: using '{}' for fingerprinting",
                    view_name
                );
                Some(view_name.clone())
            }
        }
        None => {
            eprintln!(
                "Warning: VIEW operator '{}' not found, using full state fingerprints",
                view_name
            );
            None
        }
    }
}

/// Expand operator references in an operator definition body.
///
/// Returns a new `OperatorDef` with inlined operator definitions in the body.
/// This eliminates expensive operator lookups and expression tree cloning during
/// state enumeration by inlining all operator definitions once at startup.
///
/// Both sequential (`run_prepare.rs:379-391`) and parallel (`prepare.rs:252-264`)
/// checkers previously duplicated this exact OperatorDef reconstruction.
pub(crate) fn expand_operator_body(ctx: &EvalCtx, def: &OperatorDef) -> OperatorDef {
    let expanded_body = expand_operators(ctx, &def.body);
    OperatorDef {
        name: def.name.clone(),
        params: def.params.clone(),
        body: expanded_body,
        local: def.local,
        contains_prime: def.contains_prime,
        guards_depend_on_prime: def.guards_depend_on_prime,
        has_primed_param: def.has_primed_param,
        is_recursive: def.is_recursive,
        self_call_count: def.self_call_count,
    }
}

/// Expand operator body including operators that contain primed variables.
///
/// Used by POR dependency extraction which needs to see the full action body
/// (primed assignments, UNCHANGED) to compute read/write sets. The default
/// `expand_operator_body` skips primed operators to protect guard compilation
/// and constraint evaluation from seeing primed sub-expressions.
///
/// Part of #3354 Slice 4: replaces CompiledAction-tree dependency walk.
///
/// Since the per-coverage-action dependency extraction (audit-2026-07 #11),
/// production POR no longer expands the whole Next body with primes (doing so
/// re-split named actions into a finer decomposition than enumeration uses);
/// POR unit tests still exercise this helper directly.
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn expand_operator_body_with_primes(ctx: &EvalCtx, def: &OperatorDef) -> OperatorDef {
    let expanded_body = expand_operators_with_primes(ctx, &def.body);
    OperatorDef {
        name: def.name.clone(),
        params: def.params.clone(),
        body: expanded_body,
        local: def.local,
        contains_prime: def.contains_prime,
        guards_depend_on_prime: def.guards_depend_on_prime,
        has_primed_param: def.has_primed_param,
        is_recursive: def.is_recursive,
        self_call_count: def.self_call_count,
    }
}

/// Lower an inline NEXT CST node to an operator definition.
///
/// When the SPECIFICATION formula contains an inline NEXT expression like
/// `Init /\ [][\E n \in Node: Next(n)]_vars`, the resolved spec's `next_node`
/// contains the CST node. This function lowers it to an AST expression, wraps
/// it in a synthetic `OperatorDef` named `INLINE_NEXT_NAME`, and resolves state
/// variables in the body.
///
/// Returns `None` if `next_node` is `None` (no inline NEXT expression).
/// Returns `Some(Ok(op_def))` on success.
/// Returns `Some(Err(CheckError))` if CST lowering fails.
///
/// Both sequential (`run_prepare.rs:33-66`) and parallel (`checker.rs:347-402`)
/// checkers previously duplicated this lowering+construction logic. Each caller
/// then registers the returned `OperatorDef` according to its own storage model:
/// sequential stores in both `module.op_defs` and `ctx`, parallel stores in
/// `self.op_defs` and recomputes `uses_trace`.
pub(crate) fn lower_inline_next(
    next_node: Option<&SyntaxNode>,
    var_registry: &VarRegistry,
) -> Option<Result<OperatorDef, CheckError>> {
    let node = next_node?;

    let expr = match tla_core::lower_single_expr(tla_core::FileId(0), node) {
        Some(e) => e,
        None => {
            return Some(Err(ConfigCheckError::Specification(format!(
                "Failed to lower inline NEXT expression: {}",
                node.text()
            ))
            .into()));
        }
    };

    let mut op_def = OperatorDef {
        name: Spanned::dummy(INLINE_NEXT_NAME.to_string()),
        params: vec![],
        body: Spanned::dummy(expr),
        local: false,
        contains_prime: true,
        guards_depend_on_prime: true,
        has_primed_param: false,
        is_recursive: false,
        self_call_count: 0,
    };

    tla_eval::state_var::resolve_state_vars_in_op_def(&mut op_def, var_registry);

    Some(Ok(op_def))
}

/// Check ASSUME statements, returning the first failing error.
///
/// Evaluates each ASSUME expression in order. Returns `None` if all pass,
/// `Some(CheckError)` if any fail (false, non-boolean, or eval error).
/// Callers wrap the error in `CheckResult` with their own stats.
///
/// Both sequential (`run_checks.rs:18-46`) and parallel (`prepare.rs:128-153`)
/// checkers previously duplicated this exact match-arm pattern. Extracting it
/// prevents divergence in error classification (e.g., non-boolean handling)
/// and ensures both paths produce identical `CheckError` variants.
pub(crate) fn check_assumes(
    ctx: &EvalCtx,
    assumes: &[(String, Spanned<Expr>)],
) -> Option<CheckError> {
    check_assumes_with_modules(ctx, assumes, None)
}

/// `check_assumes` with an optional module set enabling bytecode-VM
/// evaluation of iteration-heavy constant ASSUME expressions.
///
/// ASSUME clauses are constant expressions, but quantifier-heavy ones (e.g.
/// `\A x \in 0..1000000 : Even(x+x)`) dominate whole runs when evaluated by
/// the AST tree-walker (per-element operator application: n-ary cache
/// insertion, dep tracking, binding-chain churn). When the bytecode VM is
/// enabled (production default) and the expression is structurally
/// quantifier-bearing, compile it through the same TIR bytecode pipeline the
/// checker already trusts for invariants and execute it once in the VM.
///
/// Semantics stay exact:
/// - Only a clean VM value short-circuits the tree-walker; any compile
///   failure, unsupported opcode, or VM runtime error falls back to the AST
///   evaluator, which produces the canonical result/error.
/// - A VM `FALSE`/non-boolean is reported through the identical
///   `AssumeFalse { location }` construction as the AST path.
/// - Functions that touch state buffers are rejected before execution
///   (ASSUMEs are constant-level; there is no state environment yet).
pub(crate) fn check_assumes_with_modules(
    ctx: &EvalCtx,
    assumes: &[(String, Spanned<Expr>)],
    modules: Option<(&tla_core::ast::Module, &[tla_core::ast::Module])>,
) -> Option<CheckError> {
    use crate::eval::eval_entry;
    use crate::value::Value;

    let vm_results = modules
        .filter(|_| !assumes.is_empty() && tla_eval::tir::bytecode_vm_enabled())
        .map(|(root, deps)| try_eval_assumes_via_bytecode_vm(ctx, assumes, root, deps));

    for (idx, (module_name, assume_expr)) in assumes.iter().enumerate() {
        let vm_value = vm_results
            .as_ref()
            .and_then(|results| results.get(idx).cloned().flatten());
        let evaluated = match vm_value {
            Some(value) => Ok(value),
            None => eval_entry(ctx, assume_expr),
        };
        match evaluated {
            Ok(Value::Bool(true)) => {}
            Ok(Value::Bool(false)) | Ok(_) => {
                let location = format_span_location(&assume_expr.span, module_name);
                return Some(RuntimeCheckError::AssumeFalse { location }.into());
            }
            Err(e) => {
                return Some(EvalCheckError::Eval(e).into());
            }
        }
    }
    None
}

/// Structural VM-candidacy pre-screen for an ASSUME expression.
///
/// Returns `true` only when the expression (transitively, through operator
/// definitions resolvable in `ctx`) is built EXCLUSIVELY from scalar-friendly
/// constructs — boolean logic, integer arithmetic, comparisons, membership,
/// ranges, set literals, IF, quantifiers — AND contains at least one
/// quantifier. This is the shape where the tree-walker is iteration-bound
/// (per-element operator application: n-ary cache insertion, dep tracking,
/// binding-chain churn) and the bytecode VM is measurably faster.
///
/// Everything else stays on the AST path. That is deliberately conservative:
/// set-heavy / recursive / higher-order ASSUME bodies (CHOOSE, set builders,
/// SUBSET, function sets, EXCEPT, LET, records, ...) evaluate SLOWER through
/// the VM (it materializes domains the tree-walker streams lazily and gets no
/// benefit from the AST-side operator-result caches), and quantifier-free
/// constant predicates (`N \in Nat`) are too cheap to justify a compile.
fn assume_expr_is_vm_candidate(ctx: &EvalCtx, expr: &Expr) -> bool {
    struct ScalarScan<'c> {
        ctx: &'c EvalCtx,
        visited_ops: std::collections::HashSet<usize>,
        depth: usize,
        saw_quantifier: bool,
    }

    impl ScalarScan<'_> {
        fn op_is_scalar(&mut self, name: &str) -> bool {
            if self.depth >= 64 {
                // Conservative: very deep operator chains stay on AST.
                return false;
            }
            let resolved = self.ctx.resolve_op_name(name);
            let Some(def) = self.ctx.get_op(resolved).cloned() else {
                // Not an operator: a config constant, bound model value, or
                // builtin set name. Allowed as a leaf; the bytecode compiler
                // resolves (or rejects) it.
                return true;
            };
            if def.is_recursive || def.has_primed_param || def.contains_prime {
                return false;
            }
            let key = std::sync::Arc::as_ptr(&def) as usize;
            if !self.visited_ops.insert(key) {
                // Already vetted (or in progress, which non-recursive defs
                // cannot be); treat as vetted.
                return true;
            }
            self.depth += 1;
            let result = self.expr_is_scalar(&def.body.node);
            self.depth -= 1;
            result
        }

        fn bounds_are_scalar(&mut self, bounds: &[tla_core::ast::BoundVar]) -> bool {
            bounds.iter().all(|bound| {
                bound.pattern.is_none()
                    && bound
                        .domain
                        .as_ref()
                        .is_some_and(|domain| self.expr_is_scalar(&domain.node))
            })
        }

        fn expr_is_scalar(&mut self, expr: &Expr) -> bool {
            match expr {
                Expr::Bool(_) | Expr::Int(_) | Expr::String(_) => true,
                Expr::Ident(name, _) => self.op_is_scalar(name),
                Expr::Apply(op_expr, args) => {
                    let Expr::Ident(name, _) = &op_expr.node else {
                        return false;
                    };
                    self.op_is_scalar(name) && args.iter().all(|arg| self.expr_is_scalar(&arg.node))
                }
                Expr::And(lhs, rhs)
                | Expr::Or(lhs, rhs)
                | Expr::Implies(lhs, rhs)
                | Expr::Equiv(lhs, rhs)
                | Expr::In(lhs, rhs)
                | Expr::NotIn(lhs, rhs)
                | Expr::Eq(lhs, rhs)
                | Expr::Neq(lhs, rhs)
                | Expr::Lt(lhs, rhs)
                | Expr::Leq(lhs, rhs)
                | Expr::Gt(lhs, rhs)
                | Expr::Geq(lhs, rhs)
                | Expr::Add(lhs, rhs)
                | Expr::Sub(lhs, rhs)
                | Expr::Mul(lhs, rhs)
                | Expr::Div(lhs, rhs)
                | Expr::IntDiv(lhs, rhs)
                | Expr::Mod(lhs, rhs)
                | Expr::Pow(lhs, rhs)
                | Expr::Range(lhs, rhs) => {
                    self.expr_is_scalar(&lhs.node) && self.expr_is_scalar(&rhs.node)
                }
                Expr::Not(inner) | Expr::Neg(inner) => self.expr_is_scalar(&inner.node),
                Expr::SetEnum(elements) => {
                    elements.iter().all(|elem| self.expr_is_scalar(&elem.node))
                }
                Expr::If(cond, then_branch, else_branch) => {
                    self.expr_is_scalar(&cond.node)
                        && self.expr_is_scalar(&then_branch.node)
                        && self.expr_is_scalar(&else_branch.node)
                }
                Expr::Forall(bounds, body) | Expr::Exists(bounds, body) => {
                    if !self.bounds_are_scalar(bounds) || !self.expr_is_scalar(&body.node) {
                        return false;
                    }
                    self.saw_quantifier = true;
                    true
                }
                // Everything else (CHOOSE, set builders/filters, SUBSET,
                // function definitions/sets/application, records, tuples,
                // LET, CASE, EXCEPT, temporal forms, module refs, ...):
                // conservative — stay on the AST path.
                _ => false,
            }
        }
    }

    let mut scan = ScalarScan {
        ctx,
        visited_ops: std::collections::HashSet::new(),
        depth: 0,
        saw_quantifier: false,
    };
    scan.expr_is_scalar(expr) && scan.saw_quantifier
}

/// Reject bytecode functions that touch state buffers (directly or through
/// reachable callees). ASSUME evaluation runs before any state exists, so the
/// VM is bound to an empty state environment; this guard keeps every
/// state-touching shape on the AST path instead.
fn bytecode_function_touches_state(
    chunk: &tla_tir::bytecode::BytecodeChunk,
    func_idx: u16,
    visited: &mut std::collections::HashSet<u16>,
) -> bool {
    use tla_tir::bytecode::Opcode;

    if !visited.insert(func_idx) {
        return false;
    }
    let Some(func) = chunk.functions.get(usize::from(func_idx)) else {
        return true;
    };
    func.instructions.iter().any(|op| match op {
        Opcode::LoadVar { .. }
        | Opcode::StoreVar { .. }
        | Opcode::LoadPrime { .. }
        | Opcode::SetPrimeMode { .. }
        | Opcode::Unchanged { .. } => true,
        Opcode::Call { op_idx, .. } => bytecode_function_touches_state(chunk, *op_idx, visited),
        _ => false,
    })
}

/// Compile iteration-bearing ASSUME expressions to TIR bytecode (as synthetic
/// zero-arg operators appended to a clone of the root module) and execute
/// them in the bytecode VM. Returns one `Option<Value>` per ASSUME: `Some`
/// only when the VM produced a clean value; `None` keeps that ASSUME on the
/// canonical AST path.
fn try_eval_assumes_via_bytecode_vm(
    ctx: &EvalCtx,
    assumes: &[(String, Spanned<Expr>)],
    root: &tla_core::ast::Module,
    deps: &[tla_core::ast::Module],
) -> Vec<Option<crate::value::Value>> {
    use tla_core::ast::Unit;

    let mut results: Vec<Option<crate::value::Value>> = vec![None; assumes.len()];

    let candidates: Vec<usize> = assumes
        .iter()
        .enumerate()
        .filter(|(_, (_, expr))| assume_expr_is_vm_candidate(ctx, &expr.node))
        .map(|(idx, _)| idx)
        .collect();
    if candidates.is_empty() {
        return results;
    }

    // Wrap each candidate in a synthetic zero-arg operator so the existing
    // operator-oriented bytecode pipeline can compile it.
    let mut root = root.clone();
    let mut names: Vec<(usize, String)> = Vec::with_capacity(candidates.len());
    for idx in candidates {
        let name = format!("__ty_assume_vm_{idx}");
        // `ASSUME T1` where `THEOREM T1 == ...`: theorem names resolve as
        // zero-arg operators in `ctx` but are not exported into the TIR
        // bytecode namespace. Unfold one level so the synthetic operator
        // wraps the theorem body directly.
        let mut body = assumes[idx].1.clone();
        if let Expr::Ident(ident, _) = &body.node {
            if let Some(def) = ctx
                .get_op(ctx.resolve_op_name(ident))
                .filter(|def| def.params.is_empty())
            {
                body = def.body.clone();
            }
        }
        let def = OperatorDef {
            name: Spanned::dummy(name.clone()),
            params: vec![],
            body,
            local: false,
            contains_prime: false,
            guards_depend_on_prime: false,
            has_primed_param: false,
            is_recursive: false,
            self_call_count: 0,
        };
        root.units.push(Spanned::dummy(Unit::Operator(def)));
        names.push((idx, name));
    }

    // ASSUME checking can run before `promote_env_constants_to_precomputed`
    // (the assume-only early path binds config constants straight into the
    // env), so augment the precomputed map with env-bound constants the same
    // way the promotion pass does. Precomputed entries win on conflict.
    let mut resolved_constants = ctx.precomputed_constants().clone();
    let var_registry = ctx.var_registry();
    for (name, value) in ctx.env().iter() {
        let name_id = tla_core::name_intern::intern_name(name.as_ref());
        if var_registry.get_by_name_id(name_id).is_some() {
            continue;
        }
        if resolved_constants.contains_key(&name_id) {
            continue;
        }
        resolved_constants.insert(name_id, value.clone());
    }

    let dep_refs: Vec<&tla_core::ast::Module> = deps.iter().collect();
    let tir_callees = tla_eval::bytecode_vm::collect_bytecode_namespace_callees(&root, &dep_refs);
    let op_names: Vec<String> = names.iter().map(|(_, name)| name.clone()).collect();
    let compiled = tla_eval::bytecode_vm::compile_operators_to_bytecode_full_with_state_vars(
        &root,
        &dep_refs,
        &op_names,
        &resolved_constants,
        Some(ctx.op_replacements()),
        Some(&tir_callees),
        None,
    );

    let reason_logs = crate::check::debug::bytecode_vm_stats_enabled();
    if reason_logs {
        for (name, error) in &compiled.failed {
            eprintln!("[bytecode] assume compile skip {name}: {error}");
        }
    }

    let chunk = &compiled.chunk;
    for (idx, name) in names {
        let Some(&func_idx) = compiled.op_indices.get(&name) else {
            continue;
        };
        if bytecode_function_touches_state(chunk, func_idx, &mut std::collections::HashSet::new()) {
            if reason_logs {
                eprintln!("[bytecode] assume vm skip {name}: touches state buffers");
            }
            continue;
        }
        let mut vm = tla_eval::bytecode_vm::BytecodeVm::new(chunk, &[], None).with_eval_ctx(ctx);
        match vm.execute_function(func_idx) {
            Ok(value) => results[idx] = Some(value),
            // Any VM error: leave the ASSUME on the AST path, which produces
            // the canonical result or error.
            Err(error) => {
                if reason_logs {
                    eprintln!("[bytecode] assume vm fallback {name}: {error}");
                }
            }
        }
    }
    results
}
