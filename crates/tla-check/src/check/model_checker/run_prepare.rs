// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Pre-BFS preparation: constant binding, symmetry computation, VIEW validation,
//! invariant compilation, operator expansion, and action compilation.
//!
//! Extracted from `run.rs` to separate setup concerns from runtime dispatch
//! (Part of #2385). TLC keeps construction/init concerns in `ModelChecker` init
//! path; this module mirrors that boundary.

use super::super::api::{check_error_to_result, CheckResult, ResolvedSpec, INLINE_NEXT_NAME};
use tla_value::Rp;
use super::super::check_error::CheckError;
use super::debug::debug_bytecode_vm;
#[cfg(debug_assertions)]
use super::debug::ty_debug;
use super::fingerprint::BfsFingerprintDomain;
use super::mc_struct::ModelChecker;
use super::trace_detect::compute_uses_trace;
use crate::constants::bind_constants_from_config;
use crate::{ConfigCheckError, EvalCheckError};
use num_traits::ToPrimitive;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::sync::Arc;
use tla_core::ast::{BoundPattern, BoundVar, Expr, Module, OperatorDef, RecordFieldName};
use tla_core::name_intern::{intern_name, NameId};
use tla_core::span::Spanned;
use tla_core::{
    free_vars, substitution_would_capture, BoundNameStack, ExprFold, SpanPolicy, SubstituteExpr,
};
// Part of #4398: consume fail-closed compiled-backend types through tla-check's local shim.
use crate::compiled_backend_unavailable::JitInvariantCache as JitInvariantCacheImpl;

/// AUTO engine-selection scanner: detect a native-hostile quantifier domain.
///
/// Returns `true` if `expr` (a resolved per-action body) contains a
/// quantifier (`\E`/`\A`) or `CHOOSE` whose bound domain enumerates a
/// FUNCTION-SET (`[D -> R]`, `Expr::FuncSet`) or POWERSET (`SUBSET S`,
/// `Expr::Powerset`), possibly reached through a set-filter / set-builder
/// generator (`{f \in [D -> R] : P}` / `{g : f \in SUBSET S}`). These domains
/// are not enumerable proof-backed compact-set domains: native successor
/// generation fails closed on them, so attempting the native compile is pure
/// overhead. The traversal also recurses into bodies, so nested hostile
/// quantifiers (the common case — the hostile `\E` sits inside an outer action
/// conjunction) are detected. Structural only; no names are inspected.
fn expr_has_native_hostile_quantifier_domain(expr: &Expr) -> bool {
    use tla_core::{walk_expr, ExprVisitor};

    /// `true` if a quantifier/CHOOSE bound *domain* expression is a function-set
    /// or powerset, including when it is the generator inside a set-filter or
    /// set-builder (the `S` in `{x \in S : P}` / `{e : x \in S}`).
    fn domain_is_native_hostile(domain: &Expr) -> bool {
        match domain {
            Expr::FuncSet(_, _) | Expr::Powerset(_) => true,
            // `{f \in [D -> R] : P}` — the filter generator domain is hostile.
            Expr::SetFilter(bv, _) => bv
                .domain
                .as_ref()
                .is_some_and(|d| domain_is_native_hostile(&d.node)),
            // `{e : f \in SUBSET S}` — any generator domain hostile.
            Expr::SetBuilder(_, bounds) => bounds.iter().any(|bv| {
                bv.domain
                    .as_ref()
                    .is_some_and(|d| domain_is_native_hostile(&d.node))
            }),
            _ => false,
        }
    }

    struct HostileDomainScan;
    impl ExprVisitor for HostileDomainScan {
        type Output = bool;
        fn visit_node(&mut self, expr: &Expr) -> Option<bool> {
            let bounds_hostile: bool = match expr {
                Expr::Forall(bounds, _) | Expr::Exists(bounds, _) => bounds.iter().any(|bv| {
                    bv.domain
                        .as_ref()
                        .is_some_and(|d| domain_is_native_hostile(&d.node))
                }),
                Expr::Choose(bv, _) => bv
                    .domain
                    .as_ref()
                    .is_some_and(|d| domain_is_native_hostile(&d.node)),
                _ => false,
            };
            // `Some(true)` short-circuits; `None` falls through to the default
            // traversal so domains AND bodies (with nested quantifiers) are still
            // scanned.
            if bounds_hostile {
                Some(true)
            } else {
                None
            }
        }
    }

    walk_expr(&mut HostileDomainScan, expr)
}

// Pure boolean reducer: each parameter is a one-bit signal computed by the
// caller from a single field. Wrapping each in a two-variant enum would add
// boilerplate without clarity gain.
#[allow(clippy::fn_params_excessive_bools)]
fn flat_state_primary_storage_admitted(
    flat_state_disabled: bool,
    has_view: bool,
    has_symmetry: bool,
    full_state_storage: bool,
    adapter_ready: bool,
    flat_primary_safe: bool,
) -> bool {
    !flat_state_disabled
        && !has_view
        && !has_symmetry
        && !full_state_storage
        && adapter_ready
        && flat_primary_safe
}

#[allow(clippy::too_many_arguments)]
fn collect_sequence_capacity_proofs(
    expr: &Expr,
    invariant: &str,
    registry: &crate::var_index::VarRegistry,
    constants: &tla_core::kani_types::HashMap<NameId, crate::Value>,
    op_defs: &tla_core::OpEnv,
    op_replacements: &tla_core::kani_types::HashMap<String, String>,
    proof_domains: &BTreeMap<String, Arc<[crate::Value]>>,
    bound_eval: Option<&dyn Fn(&Expr) -> Option<usize>>,
    scope: &mut ProofScope,
    visiting: &mut BTreeSet<String>,
    out: &mut Vec<crate::state::SequenceCapacityProof>,
) {
    match expr {
        Expr::And(left, right) => {
            collect_sequence_capacity_proofs(
                &left.node,
                invariant,
                registry,
                constants,
                op_defs,
                op_replacements,
                proof_domains,
                bound_eval,
                scope,
                visiting,
                out,
            );
            collect_sequence_capacity_proofs(
                &right.node,
                invariant,
                registry,
                constants,
                op_defs,
                op_replacements,
                proof_domains,
                bound_eval,
                scope,
                visiting,
                out,
            );
        }
        Expr::Forall(vars, body) => {
            if let Some(added) = push_bounded_quantifier_names(
                vars,
                constants,
                op_defs,
                op_replacements,
                proof_domains,
                scope,
            ) {
                collect_sequence_capacity_proofs(
                    &body.node,
                    invariant,
                    registry,
                    constants,
                    op_defs,
                    op_replacements,
                    proof_domains,
                    bound_eval,
                    scope,
                    visiting,
                    out,
                );
                for name in added {
                    scope.pop(&name);
                }
            }
        }
        Expr::Leq(left, right) => {
            let sequence_path = extract_bounded_sequence_path(
                &left.node,
                registry,
                scope,
                op_defs,
                op_replacements,
                visiting,
            );
            if let (Some((var_idx, path)), Some(max_len)) = (
                sequence_path,
                expr_usize_bound(&right.node, constants, op_replacements, bound_eval),
            ) {
                push_sequence_capacity_proof(
                    out,
                    crate::state::SequenceCapacityProof {
                        var_idx,
                        path,
                        max_len,
                        invariant: Arc::from(invariant),
                        heuristic: false,
                    },
                );
            }
        }
        Expr::Geq(left, right) => {
            let sequence_path = extract_bounded_sequence_path(
                &right.node,
                registry,
                scope,
                op_defs,
                op_replacements,
                visiting,
            );
            if let (Some(max_len), Some((var_idx, path))) = (
                expr_usize_bound(&left.node, constants, op_replacements, bound_eval),
                sequence_path,
            ) {
                push_sequence_capacity_proof(
                    out,
                    crate::state::SequenceCapacityProof {
                        var_idx,
                        path,
                        max_len,
                        invariant: Arc::from(invariant),
                        heuristic: false,
                    },
                );
            }
        }
        Expr::Ident(name, _) if !scope.is_bound(name) => {
            collect_sequence_capacity_proofs_from_zero_arg_op(
                name,
                invariant,
                registry,
                constants,
                op_defs,
                op_replacements,
                proof_domains,
                bound_eval,
                scope,
                visiting,
                out,
            );
        }
        Expr::OpRef(name) => {
            collect_sequence_capacity_proofs_from_zero_arg_op(
                name,
                invariant,
                registry,
                constants,
                op_defs,
                op_replacements,
                proof_domains,
                bound_eval,
                scope,
                visiting,
                out,
            );
        }
        _ => {}
    }
}

#[allow(clippy::too_many_arguments)]
fn collect_sequence_capacity_proofs_from_zero_arg_op(
    name: &str,
    invariant: &str,
    registry: &crate::var_index::VarRegistry,
    constants: &tla_core::kani_types::HashMap<NameId, crate::Value>,
    op_defs: &tla_core::OpEnv,
    op_replacements: &tla_core::kani_types::HashMap<String, String>,
    proof_domains: &BTreeMap<String, Arc<[crate::Value]>>,
    bound_eval: Option<&dyn Fn(&Expr) -> Option<usize>>,
    scope: &mut ProofScope,
    visiting: &mut BTreeSet<String>,
    out: &mut Vec<crate::state::SequenceCapacityProof>,
) {
    let Some((resolved_name, def)) = proof_safe_zero_arg_op_def(name, op_defs, op_replacements)
    else {
        return;
    };
    let resolved_name = resolved_name.to_owned();
    if !visiting.insert(resolved_name.clone()) {
        return;
    }
    collect_sequence_capacity_proofs(
        &def.body.node,
        invariant,
        registry,
        constants,
        op_defs,
        op_replacements,
        proof_domains,
        bound_eval,
        scope,
        visiting,
        out,
    );
    visiting.remove(resolved_name.as_str());
}

fn push_sequence_capacity_proof(
    out: &mut Vec<crate::state::SequenceCapacityProof>,
    proof: crate::state::SequenceCapacityProof,
) {
    if !out.iter().any(|existing| existing == &proof) {
        out.push(proof);
    }
}

#[derive(Default)]
struct ProofScope {
    bindings: BTreeMap<String, Vec<Option<Arc<[crate::Value]>>>>,
}

impl ProofScope {
    fn push(&mut self, name: String, homogeneous_domain: Option<Arc<[crate::Value]>>) {
        self.bindings
            .entry(name)
            .or_default()
            .push(homogeneous_domain);
    }

    fn pop(&mut self, name: &str) {
        if let Some(stack) = self.bindings.get_mut(name) {
            stack.pop();
            if stack.is_empty() {
                self.bindings.remove(name);
            }
        }
    }

    fn is_bound(&self, name: &str) -> bool {
        self.bindings
            .get(name)
            .is_some_and(|stack| !stack.is_empty())
    }

    fn homogeneous_bound_domain(&self, name: &str) -> Option<Arc<[crate::Value]>> {
        self.bindings
            .get(name)
            .and_then(|stack| stack.last())
            .and_then(|domain| domain.as_ref().map(Arc::clone))
    }
}

/// WP-33: whether a `\A x \in D : Len(v[x]) <= K` capacity-proof binder may
/// resolve `D` directly from the CONFIGURED CONSTANTS (env
/// `TY_SEQ_CAPACITY_PROOF=1`, default OFF).
///
/// Default OFF keeps every existing corpus spec BYTE-IDENTICAL: with the flag
/// off, a binder domain that is a bare constant (rather than a zero-arg
/// operator) resolves to `None` exactly as before, so the `\A` subtree is
/// skipped and the nested sequence keeps its `Observed` bound.
fn seq_capacity_constant_domain_enabled() -> bool {
    std::env::var_os("TY_SEQ_CAPACITY_PROOF").is_some_and(|v| v == "1")
}

fn push_bounded_quantifier_names(
    vars: &[BoundVar],
    constants: &tla_core::kani_types::HashMap<NameId, crate::Value>,
    op_defs: &tla_core::OpEnv,
    op_replacements: &tla_core::kani_types::HashMap<String, String>,
    proof_domains: &BTreeMap<String, Arc<[crate::Value]>>,
    scope: &mut ProofScope,
) -> Option<Vec<String>> {
    let mut added = Vec::new();
    for var in vars {
        let homogeneous_domain = bound_var_full_homogeneous_domain(
            var,
            constants,
            op_defs,
            op_replacements,
            proof_domains,
            scope,
        );
        homogeneous_domain.as_ref()?;
        match &var.pattern {
            None | Some(BoundPattern::Var(_)) => {
                let name = var.name.node.clone();
                scope.push(name.clone(), homogeneous_domain);
                added.push(name);
            }
            Some(BoundPattern::Tuple(_)) => return None,
        }
    }
    Some(added)
}

fn bound_var_full_homogeneous_domain(
    var: &BoundVar,
    constants: &tla_core::kani_types::HashMap<NameId, crate::Value>,
    op_defs: &tla_core::OpEnv,
    op_replacements: &tla_core::kani_types::HashMap<String, String>,
    proof_domains: &BTreeMap<String, Arc<[crate::Value]>>,
    scope: &ProofScope,
) -> Option<Arc<[crate::Value]>> {
    var.domain.as_ref().and_then(|domain| {
        full_homogeneous_domain_values(
            &domain.node,
            constants,
            op_defs,
            op_replacements,
            proof_domains,
            scope,
        )
    })
}

/// Resolve a `\A x \in D` binder domain to a fully enumerated, homogeneous
/// scalar value list.
///
/// `proof_domains` (built by `named_homogeneous_proof_domains`) only carries
/// zero-arg OPERATOR definitions, so historically `\A r \in Readers : ...` with
/// `Readers` a CONSTANT bound in the `.cfg` resolved to `None` and the whole
/// quantifier subtree was skipped. That is precisely why Disruptor's
/// `consumed` — a `Sequence` nested in a `Function` range — could never earn a
/// capacity proof: `collect_sequence_capacity_proofs` already walks
/// `Len(v[r]) <= K` into a `HomogeneousRange` path step (WP-07), and the
/// identical clause written as `\A r \in RSet` with `RSet == Readers` DOES
/// produce the proof today. The gap was only the binder-domain lookup.
///
/// Under `TY_SEQ_CAPACITY_PROOF=1` a bare-identifier domain that misses in
/// `proof_domains` falls through to `expression_proof_domain_values` — the
/// SAME resolver the `RSet == Readers` alias path already reaches via the
/// operator body, so the constant and alias spellings now yield the
/// byte-identical domain vector (constant lookup, `normalize_proof_domain_values`
/// homogeneity/dedup check and the <= 63-element cap all unchanged). The
/// `!scope.is_bound(name)` guard is retained, so an enclosing binder still
/// shadows the name and no capture can occur; anything the resolver cannot
/// fully enumerate keeps failing closed to `None`.
fn full_homogeneous_domain_values(
    expr: &Expr,
    constants: &tla_core::kani_types::HashMap<NameId, crate::Value>,
    op_defs: &tla_core::OpEnv,
    op_replacements: &tla_core::kani_types::HashMap<String, String>,
    proof_domains: &BTreeMap<String, Arc<[crate::Value]>>,
    scope: &ProofScope,
) -> Option<Arc<[crate::Value]>> {
    match expr {
        Expr::Ident(name, _) if !scope.is_bound(name) => {
            if let Some(values) = proof_domains.get(name).cloned() {
                return Some(values);
            }
            if !seq_capacity_constant_domain_enabled() {
                return None;
            }
            let values = expression_proof_domain_values(
                expr,
                op_defs,
                constants,
                op_replacements,
                &mut BTreeSet::new(),
            )?;
            Some(Arc::from(values.into_boxed_slice()))
        }
        _ => None,
    }
}

fn extract_bounded_sequence_path(
    expr: &Expr,
    registry: &crate::var_index::VarRegistry,
    scope: &ProofScope,
    op_defs: &tla_core::OpEnv,
    op_replacements: &tla_core::kani_types::HashMap<String, String>,
    visiting: &mut BTreeSet<String>,
) -> Option<(usize, Vec<crate::state::SequenceCapacityPathStep>)> {
    extract_bounded_sequence_path_with_env(
        expr,
        registry,
        scope,
        op_defs,
        op_replacements,
        visiting,
        &ProofExprEnv::default(),
    )
}

fn extract_bounded_sequence_path_with_env(
    expr: &Expr,
    registry: &crate::var_index::VarRegistry,
    scope: &ProofScope,
    op_defs: &tla_core::OpEnv,
    op_replacements: &tla_core::kani_types::HashMap<String, String>,
    visiting: &mut BTreeSet<String>,
    env: &ProofExprEnv,
) -> Option<(usize, Vec<crate::state::SequenceCapacityPathStep>)> {
    let Expr::Apply(op, args) = expr else {
        return None;
    };
    if args.len() == 1 && is_len_operator(&op.node, op_replacements) {
        let mut used_bindings = BTreeSet::new();
        return extract_state_path(
            &args[0].node,
            registry,
            scope,
            op_defs,
            op_replacements,
            visiting,
            env,
            &mut used_bindings,
        );
    }

    extract_bounded_sequence_path_from_wrapper(
        &op.node,
        args,
        registry,
        scope,
        op_defs,
        op_replacements,
        visiting,
        env,
    )
}

fn is_len_operator(
    expr: &Expr,
    op_replacements: &tla_core::kani_types::HashMap<String, String>,
) -> bool {
    matches!(
        expr,
        Expr::Ident(name, _) | Expr::OpRef(name)
            if matches!(resolve_proof_op_name(name, op_replacements), Some("Len"))
    )
}

#[derive(Clone)]
struct ProofExprEnv {
    scopes: Vec<BTreeMap<String, ProofActualExpr>>,
    allow_scope_bindings: bool,
}

impl Default for ProofExprEnv {
    fn default() -> Self {
        Self {
            scopes: Vec::new(),
            allow_scope_bindings: true,
        }
    }
}

#[derive(Clone)]
struct ProofActualExpr {
    expr: Expr,
    env: ProofExprEnv,
}

impl ProofExprEnv {
    fn lookup(&self, name: &str) -> Option<&ProofActualExpr> {
        self.scopes.iter().rev().find_map(|scope| scope.get(name))
    }

    fn with_operator_args(
        &self,
        def: &OperatorDef,
        args: &[tla_core::span::Spanned<Expr>],
    ) -> Option<Self> {
        if def.params.len() != args.len() || def.params.iter().any(|param| param.arity != 0) {
            return None;
        }

        let mut scope = BTreeMap::new();
        for (param, arg) in def.params.iter().zip(args) {
            let name = param.name.node.clone();
            if scope
                .insert(
                    name,
                    ProofActualExpr {
                        expr: arg.node.clone(),
                        env: self.clone(),
                    },
                )
                .is_some()
            {
                return None;
            }
        }

        let mut next = self.clone();
        next.scopes.push(scope);
        next.allow_scope_bindings = false;
        Some(next)
    }
}

#[allow(clippy::too_many_arguments)]
fn extract_bounded_sequence_path_from_wrapper(
    op: &Expr,
    args: &[tla_core::span::Spanned<Expr>],
    registry: &crate::var_index::VarRegistry,
    scope: &ProofScope,
    op_defs: &tla_core::OpEnv,
    op_replacements: &tla_core::kani_types::HashMap<String, String>,
    visiting: &mut BTreeSet<String>,
    env: &ProofExprEnv,
) -> Option<(usize, Vec<crate::state::SequenceCapacityPathStep>)> {
    let name = applied_proof_op_name(op)?;
    let (resolved_name, def) = proof_safe_param_op_def(name, args.len(), op_defs, op_replacements)?;
    let resolved_name = resolved_name.to_owned();
    if !visiting.insert(resolved_name.clone()) {
        return None;
    }
    let result = env.with_operator_args(def, args).and_then(|wrapper_env| {
        extract_bounded_sequence_path_with_env(
            &def.body.node,
            registry,
            scope,
            op_defs,
            op_replacements,
            visiting,
            &wrapper_env,
        )
    });
    visiting.remove(resolved_name.as_str());
    result
}

#[allow(clippy::too_many_arguments)]
fn extract_state_path(
    expr: &Expr,
    registry: &crate::var_index::VarRegistry,
    scope: &ProofScope,
    op_defs: &tla_core::OpEnv,
    op_replacements: &tla_core::kani_types::HashMap<String, String>,
    visiting: &mut BTreeSet<String>,
    env: &ProofExprEnv,
    used_bindings: &mut BTreeSet<String>,
) -> Option<(usize, Vec<crate::state::SequenceCapacityPathStep>)> {
    match expr {
        Expr::StateVar(_, idx, _) => Some((*idx as usize, Vec::new())),
        Expr::Ident(name, _) if env.lookup(name).is_some() => {
            let actual = env.lookup(name)?;
            extract_state_path(
                &actual.expr,
                registry,
                scope,
                op_defs,
                op_replacements,
                visiting,
                &actual.env,
                used_bindings,
            )
        }
        Expr::Ident(name, _) if !scope.is_bound(name) => {
            registry.get(name).map(|idx| (idx.0 as usize, Vec::new()))
        }
        Expr::FuncApply(func, arg) => {
            let (binding, domain) = bound_subscript_arg(&arg.node, scope, env)?;
            if !used_bindings.insert(binding) {
                return None;
            }
            let (var_idx, mut path) = extract_state_path(
                &func.node,
                registry,
                scope,
                op_defs,
                op_replacements,
                visiting,
                env,
                used_bindings,
            )?;
            path.push(crate::state::SequenceCapacityPathStep::HomogeneousRange { domain });
            Some((var_idx, path))
        }
        Expr::RecordAccess(base, field) => {
            let (var_idx, mut path) = extract_state_path(
                &base.node,
                registry,
                scope,
                op_defs,
                op_replacements,
                visiting,
                env,
                used_bindings,
            )?;
            path.push(crate::state::SequenceCapacityPathStep::RecordField(
                record_field_name(field),
            ));
            Some((var_idx, path))
        }
        Expr::Apply(op, args) => extract_state_path_from_wrapper(
            &op.node,
            args,
            registry,
            scope,
            op_defs,
            op_replacements,
            visiting,
            env,
            used_bindings,
        ),
        _ => None,
    }
}

#[allow(clippy::too_many_arguments)]
fn extract_state_path_from_wrapper(
    op: &Expr,
    args: &[tla_core::span::Spanned<Expr>],
    registry: &crate::var_index::VarRegistry,
    scope: &ProofScope,
    op_defs: &tla_core::OpEnv,
    op_replacements: &tla_core::kani_types::HashMap<String, String>,
    visiting: &mut BTreeSet<String>,
    env: &ProofExprEnv,
    used_bindings: &mut BTreeSet<String>,
) -> Option<(usize, Vec<crate::state::SequenceCapacityPathStep>)> {
    let name = applied_proof_op_name(op)?;
    let (resolved_name, def) = proof_safe_param_op_def(name, args.len(), op_defs, op_replacements)?;
    let resolved_name = resolved_name.to_owned();
    if !visiting.insert(resolved_name.clone()) {
        return None;
    }
    let result = env.with_operator_args(def, args).and_then(|wrapper_env| {
        extract_state_path(
            &def.body.node,
            registry,
            scope,
            op_defs,
            op_replacements,
            visiting,
            &wrapper_env,
            used_bindings,
        )
    });
    visiting.remove(resolved_name.as_str());
    result
}

fn bound_subscript_arg(
    expr: &Expr,
    scope: &ProofScope,
    env: &ProofExprEnv,
) -> Option<(String, Arc<[crate::Value]>)> {
    match expr {
        Expr::Ident(name, _) if env.lookup(name).is_some() => {
            let actual = env.lookup(name)?;
            bound_subscript_arg(&actual.expr, scope, &actual.env)
        }
        Expr::Ident(name, _) if env.allow_scope_bindings => {
            Some((name.clone(), scope.homogeneous_bound_domain(name)?))
        }
        _ => None,
    }
}

fn record_field_name(field: &RecordFieldName) -> Arc<str> {
    Arc::from(field.name.node.as_str())
}

fn expr_usize_bound(
    expr: &Expr,
    constants: &tla_core::kani_types::HashMap<NameId, crate::Value>,
    op_replacements: &tla_core::kani_types::HashMap<String, String>,
    bound_eval: Option<&dyn Fn(&Expr) -> Option<usize>>,
) -> Option<usize> {
    // Primary path: a bare integer literal or a constant identifier that folds
    // to an integer. This is always available (flag-independent).
    let literal = match expr {
        Expr::Int(n) => n.to_usize(),
        Expr::Ident(name, name_id) => {
            proof_domain_scalar_constant_value(name, *name_id, constants, op_replacements)
                .as_ref()
                .and_then(value_usize_bound)
        }
        _ => None,
    };
    if literal.is_some() {
        return literal;
    }
    // `TY_SEQ_CAPACITY_PROOF` extension (the rule-(a) "<= <bounded-expr>" case):
    // fold a STATE-FREE right-hand side of a `Len(seq) <= RHS` bound (e.g.
    // `Cardinality(SomeConstSet)`, `MaxPublished - 1`, `K1 + K2`) to a compile-
    // time constant. Soundness: `bound_eval` uses the const-level evaluator,
    // whose dependency tracker rejects any current/next-state read, so the
    // folded value is a proven upper bound on `Len(seq)` in EVERY reachable
    // state where the checked invariant/constraint holds. A state-dependent RHS
    // never folds (returns `None`) and therefore fails closed to `Observed`.
    bound_eval.and_then(|eval| eval(expr))
}

fn value_usize_bound(value: &crate::Value) -> Option<usize> {
    match value {
        crate::Value::SmallInt(n) => usize::try_from(*n).ok(),
        crate::Value::Int(n) => n.to_usize(),
        _ => None,
    }
}

fn proof_domain_values_from_value(value: &crate::Value) -> Option<Vec<crate::Value>> {
    if value.set_len()?.to_usize()? > 63 {
        return None;
    }
    let set = value.to_sorted_set()?;
    normalize_proof_domain_values(set.iter().cloned().collect())
}

fn expression_proof_domain_values(
    expr: &Expr,
    op_defs: &tla_core::OpEnv,
    constants: &tla_core::kani_types::HashMap<NameId, crate::Value>,
    op_replacements: &tla_core::kani_types::HashMap<String, String>,
    visiting: &mut BTreeSet<String>,
) -> Option<Vec<crate::Value>> {
    match expr {
        Expr::SetEnum(elems) => {
            let values: Option<Vec<crate::Value>> = elems
                .iter()
                .map(|elem| proof_domain_scalar_value(&elem.node, constants, op_replacements))
                .collect();
            normalize_proof_domain_values(values?)
        }
        Expr::Range(left, right) => {
            let lo = proof_domain_int_value(&left.node, constants, op_replacements)?;
            let hi = proof_domain_int_value(&right.node, constants, op_replacements)?;
            if hi < lo || hi - lo >= 63 {
                return None;
            }
            normalize_proof_domain_values((lo..=hi).map(crate::Value::SmallInt).collect())
        }
        Expr::SetMinus(left, right) => {
            let mut values = expression_proof_domain_values(
                &left.node,
                op_defs,
                constants,
                op_replacements,
                visiting,
            )?;
            let remove = expression_proof_domain_values(
                &right.node,
                op_defs,
                constants,
                op_replacements,
                visiting,
            )?;
            values.retain(|value| !remove.contains(value));
            normalize_proof_domain_values(values)
        }
        Expr::Ident(name, name_id) => {
            if let Some(values) =
                proof_domain_constant_values(name, *name_id, constants, op_replacements)
            {
                return Some(values);
            }
            let resolved = resolve_proof_op_name(name, op_replacements)?;
            if !visiting.insert(resolved.to_string()) {
                return None;
            }
            let result = op_defs.get(resolved).and_then(|def| {
                let def = def.as_ref();
                (def.params.is_empty() && !def.contains_prime && !def.is_recursive)
                    .then(|| {
                        expression_proof_domain_values(
                            &def.body.node,
                            op_defs,
                            constants,
                            op_replacements,
                            visiting,
                        )
                    })
                    .flatten()
            });
            visiting.remove(resolved);
            result
        }
        _ => None,
    }
}

fn resolve_proof_op_name<'a>(
    name: &'a str,
    op_replacements: &'a tla_core::kani_types::HashMap<String, String>,
) -> Option<&'a str> {
    let mut current = name;
    let mut seen = BTreeSet::new();
    loop {
        if !seen.insert(current) {
            return None;
        }
        let Some(next) = op_replacements.get(current) else {
            return Some(current);
        };
        current = next.as_str();
    }
}

fn proof_domain_constant_values(
    name: &str,
    name_id: NameId,
    constants: &tla_core::kani_types::HashMap<NameId, crate::Value>,
    op_replacements: &tla_core::kani_types::HashMap<String, String>,
) -> Option<Vec<crate::Value>> {
    let resolved = resolve_proof_op_name(name, op_replacements)?;
    let resolved_id = intern_name(resolved);
    if let Some(values) = constants
        .get(&resolved_id)
        .and_then(proof_domain_values_from_value)
    {
        return Some(values);
    }

    let id = if name_id == NameId::INVALID {
        intern_name(name)
    } else {
        name_id
    };
    if id != resolved_id && !op_replacements.contains_key(name) {
        constants.get(&id).and_then(proof_domain_values_from_value)
    } else {
        None
    }
}

fn proof_safe_zero_arg_op_def<'a>(
    name: &'a str,
    op_defs: &'a tla_core::OpEnv,
    op_replacements: &'a tla_core::kani_types::HashMap<String, String>,
) -> Option<(&'a str, &'a OperatorDef)> {
    proof_safe_param_op_def(name, 0, op_defs, op_replacements)
}

fn proof_safe_param_op_def<'a>(
    name: &'a str,
    arity: usize,
    op_defs: &'a tla_core::OpEnv,
    op_replacements: &'a tla_core::kani_types::HashMap<String, String>,
) -> Option<(&'a str, &'a OperatorDef)> {
    let resolved = resolve_proof_op_name(name, op_replacements)?;
    let def = op_defs.get(resolved)?.as_ref();
    (def.params.len() == arity
        && !def.contains_prime
        && !def.has_primed_param
        && !def.is_recursive
        && def.params.iter().all(|param| param.arity == 0))
    .then_some((resolved, def))
}

fn applied_proof_op_name(expr: &Expr) -> Option<&str> {
    match expr {
        Expr::Ident(name, _) | Expr::OpRef(name) => Some(name.as_str()),
        _ => None,
    }
}

fn lower_proof_operator_wrappers(
    ctx: &crate::eval::EvalCtx,
    expr: &Spanned<Expr>,
    op_defs: &tla_core::OpEnv,
    op_replacements: &tla_core::kani_types::HashMap<String, String>,
    visiting: BTreeSet<String>,
) -> Spanned<Expr> {
    // Fully inline INSTANCE-namespaced `ModuleRef` conjuncts (e.g. `Buffer!TypeOk`)
    // into their substitution-applied, qualified bodies BEFORE proof-operator
    // wrapper lowering. Without this, an invariant such as RingBuffer's
    // `TypeOk == /\ Buffer!TypeOk /\ ...` leaves the `ringbuffer \in [...]` membership
    // hidden behind a `ModuleRef` that the static proof walker skips, so the
    // SetBitmask-range universe is never proven for the instanced state var.
    // `flatten_property_module_refs` is fail-closed (unresolvable refs are left
    // intact and simply contribute no proof).
    let expr = crate::checker_ops::flatten_property_module_refs(ctx, expr.clone());
    let mut lowerer = ProofOperatorLowerer {
        op_defs,
        op_replacements,
        visiting,
        bound_names: BoundNameStack::new(),
        preserve_precomputed_constants: None,
    };
    lowerer.fold_expr(expr)
}

/// Like [`lower_proof_operator_wrappers`], but leaves zero-argument operators that
/// resolve to a precomputed-constant value as bare `Ident`s instead of inlining
/// their definition body.
///
/// This is used by the SetBitmask-range universe collector: a model operator like
/// `Id == [i \in 1..N |-> i]` is already evaluated to a concrete function `Value`
/// by the constant-precompute pass, so the universe extractor can resolve `Id` by
/// name. Inlining it to its `FuncDef` body instead would hide that name and force
/// the extractor to re-evaluate the function expression (which it cannot do
/// soundly without a full evaluator), causing the universe proof to be missed.
fn lower_proof_operator_wrappers_preserving_precomputed_constants(
    ctx: &crate::eval::EvalCtx,
    expr: &Spanned<Expr>,
    op_defs: &tla_core::OpEnv,
    op_replacements: &tla_core::kani_types::HashMap<String, String>,
    visiting: BTreeSet<String>,
    precomputed_constants: &tla_core::kani_types::HashMap<NameId, crate::Value>,
) -> Spanned<Expr> {
    // See `lower_proof_operator_wrappers`: inline INSTANCE `ModuleRef` conjuncts
    // (e.g. `Buffer!TypeOk`) so the SetBitmask-range universe collector reaches
    // the instanced state var's membership type. Fail-closed.
    let expr = crate::checker_ops::flatten_property_module_refs(ctx, expr.clone());
    let mut lowerer = ProofOperatorLowerer {
        op_defs,
        op_replacements,
        visiting,
        bound_names: BoundNameStack::new(),
        preserve_precomputed_constants: Some(precomputed_constants),
    };
    lowerer.fold_expr(expr)
}

struct ProofOperatorLowerer<'a> {
    op_defs: &'a tla_core::OpEnv,
    op_replacements: &'a tla_core::kani_types::HashMap<String, String>,
    visiting: BTreeSet<String>,
    bound_names: BoundNameStack,
    preserve_precomputed_constants: Option<&'a tla_core::kani_types::HashMap<NameId, crate::Value>>,
}

enum ProofOperatorExpansion {
    Expanded(Spanned<Expr>),
    Unsafe,
    NotProofOperator,
}

enum ProofOperatorFoldMode {
    Expr,
    QuantifierDomain,
}

impl ProofOperatorLowerer<'_> {
    fn expand_proof_operator_call(
        &mut self,
        name: &str,
        args: &[Spanned<Expr>],
    ) -> ProofOperatorExpansion {
        self.expand_proof_operator_call_with_mode(name, args, ProofOperatorFoldMode::Expr)
    }

    fn expand_proof_operator_domain_call(
        &mut self,
        name: &str,
        args: &[Spanned<Expr>],
    ) -> ProofOperatorExpansion {
        self.expand_proof_operator_call_with_mode(
            name,
            args,
            ProofOperatorFoldMode::QuantifierDomain,
        )
    }

    fn expand_proof_operator_call_with_mode(
        &mut self,
        name: &str,
        args: &[Spanned<Expr>],
        mode: ProofOperatorFoldMode,
    ) -> ProofOperatorExpansion {
        let Some((resolved_name, def)) =
            proof_safe_param_op_def(name, args.len(), self.op_defs, self.op_replacements)
        else {
            return ProofOperatorExpansion::NotProofOperator;
        };
        // Leave zero-arg operators that are already precomputed constants as bare
        // `Ident`s so downstream constant resolution (e.g. the SetBitmask-range
        // universe extractor) can look them up by name instead of seeing an
        // inlined definition body it cannot re-evaluate soundly.
        if args.is_empty() {
            if let Some(precomputed) = self.preserve_precomputed_constants {
                if precomputed.contains_key(&intern_name(resolved_name)) {
                    return ProofOperatorExpansion::NotProofOperator;
                }
            }
        }
        let resolved_name = resolved_name.to_owned();
        if self.visiting.contains(resolved_name.as_str())
            || !proof_operator_call_capture_safe(def, args, &self.bound_names)
        {
            return ProofOperatorExpansion::Unsafe;
        }

        let mut subs = HashMap::new();
        for (param, arg) in def.params.iter().zip(args.iter()) {
            if subs.insert(param.name.node.as_str(), arg).is_some() {
                return ProofOperatorExpansion::Unsafe;
            }
        }

        self.visiting.insert(resolved_name.clone());
        let substituted = {
            let mut substituter = SubstituteExpr {
                subs,
                span_policy: SpanPolicy::Preserve,
            };
            substituter.fold_expr(def.body.clone())
        };
        let result = match mode {
            ProofOperatorFoldMode::Expr => self.fold_expr(substituted),
            ProofOperatorFoldMode::QuantifierDomain => {
                self.fold_quantifier_domain_expr(substituted)
            }
        };
        self.visiting.remove(resolved_name.as_str());
        ProofOperatorExpansion::Expanded(result)
    }

    fn fold_bound_var_domains_with_sequential_scope(
        &mut self,
        vars: Vec<BoundVar>,
    ) -> Vec<BoundVar> {
        let mark = self.bound_names.mark();
        let mut folded = Vec::with_capacity(vars.len());
        for var in vars {
            let var = self.fold_bound_var_domain(var);
            self.bound_names.push_names(proof_bound_var_names(&var));
            folded.push(var);
        }
        self.bound_names.pop_to(mark);
        folded
    }

    fn fold_bound_var_domain(&mut self, var: BoundVar) -> BoundVar {
        BoundVar {
            name: var.name,
            domain: var
                .domain
                .map(|domain| Box::new(self.fold_quantifier_domain_expr(*domain))),
            pattern: var.pattern,
        }
    }

    fn fold_quantifier_domain_expr(&mut self, expr: Spanned<Expr>) -> Spanned<Expr> {
        let span = expr.span;
        match expr.node {
            // Keep bounded domains as names so proof-domain lookup still recognizes aliases
            // like Idx after wrapper lowering rewrites Dom(Idx) back to Idx.
            Expr::Ident(name, name_id) => Spanned::new(Expr::Ident(name, name_id), span),
            Expr::OpRef(name) => Spanned::new(Expr::OpRef(name), span),
            Expr::Apply(op, args) => {
                let lowered_args: Vec<_> = args
                    .into_iter()
                    .map(|arg| self.fold_quantifier_domain_expr(arg))
                    .collect();
                if let Some(name) = applied_proof_op_name(&op.node) {
                    if !self.bound_names.contains(name) {
                        match self.expand_proof_operator_domain_call(name, &lowered_args) {
                            ProofOperatorExpansion::Expanded(expanded) => return expanded,
                            ProofOperatorExpansion::Unsafe => {
                                return Spanned::new(Expr::Bool(false), span);
                            }
                            ProofOperatorExpansion::NotProofOperator => {}
                        }
                    }
                }
                Spanned::new(
                    Expr::Apply(
                        Box::new(self.fold_quantifier_domain_expr(*op)),
                        lowered_args,
                    ),
                    span,
                )
            }
            other => Spanned::new(other, span),
        }
    }

    fn fold_quantified_body(
        &mut self,
        vars: &[BoundVar],
        body: Spanned<Expr>,
    ) -> Box<Spanned<Expr>> {
        let mark = self.bound_names.mark();
        self.bound_names
            .push_names(vars.iter().flat_map(proof_bound_var_names));
        let body = Box::new(self.fold_expr(body));
        self.bound_names.pop_to(mark);
        body
    }

    fn fold_single_bound_body(
        &mut self,
        var: &BoundVar,
        body: Spanned<Expr>,
    ) -> Box<Spanned<Expr>> {
        let mark = self.bound_names.mark();
        self.bound_names.push_names(proof_bound_var_names(var));
        let body = Box::new(self.fold_expr(body));
        self.bound_names.pop_to(mark);
        body
    }
}

impl ExprFold for ProofOperatorLowerer<'_> {
    fn fold_expr(&mut self, expr: Spanned<Expr>) -> Spanned<Expr> {
        let span = expr.span;
        match expr.node {
            Expr::Ident(name, name_id) if !self.bound_names.contains(&name) => {
                match self.expand_proof_operator_call(&name, &[]) {
                    ProofOperatorExpansion::Expanded(expanded) => expanded,
                    ProofOperatorExpansion::Unsafe => Spanned::new(Expr::Bool(false), span),
                    ProofOperatorExpansion::NotProofOperator => {
                        Spanned::new(Expr::Ident(name, name_id), span)
                    }
                }
            }
            Expr::OpRef(name) if !self.bound_names.contains(&name) => {
                match self.expand_proof_operator_call(&name, &[]) {
                    ProofOperatorExpansion::Expanded(expanded) => expanded,
                    ProofOperatorExpansion::Unsafe => Spanned::new(Expr::Bool(false), span),
                    ProofOperatorExpansion::NotProofOperator => {
                        Spanned::new(Expr::OpRef(name), span)
                    }
                }
            }
            Expr::Apply(op, args) => {
                let lowered_args: Vec<_> =
                    args.into_iter().map(|arg| self.fold_expr(arg)).collect();
                if let Some(name) = applied_proof_op_name(&op.node) {
                    if !self.bound_names.contains(name) {
                        match self.expand_proof_operator_call(name, &lowered_args) {
                            ProofOperatorExpansion::Expanded(expanded) => return expanded,
                            ProofOperatorExpansion::Unsafe => {
                                return Spanned::new(Expr::Bool(false), span);
                            }
                            ProofOperatorExpansion::NotProofOperator => {}
                        }
                    }
                }
                Spanned::new(Expr::Apply(self.fold_box(op), lowered_args), span)
            }
            Expr::Forall(vars, body) => {
                let vars = self.fold_bound_var_domains_with_sequential_scope(vars);
                let body = self.fold_quantified_body(&vars, *body);
                Spanned::new(Expr::Forall(vars, body), span)
            }
            Expr::Exists(vars, body) => {
                let vars = self.fold_bound_var_domains_with_sequential_scope(vars);
                let body = self.fold_quantified_body(&vars, *body);
                Spanned::new(Expr::Exists(vars, body), span)
            }
            Expr::FuncDef(vars, body) => {
                let vars = self.fold_bound_var_domains_with_sequential_scope(vars);
                let body = self.fold_quantified_body(&vars, *body);
                Spanned::new(Expr::FuncDef(vars, body), span)
            }
            Expr::SetBuilder(body, vars) => {
                let vars = self.fold_bound_var_domains_with_sequential_scope(vars);
                let body = self.fold_quantified_body(&vars, *body);
                Spanned::new(Expr::SetBuilder(body, vars), span)
            }
            Expr::Choose(var, body) => {
                let var = self.fold_bound_var_domain(var);
                let body = self.fold_single_bound_body(&var, *body);
                Spanned::new(Expr::Choose(var, body), span)
            }
            Expr::SetFilter(var, body) => {
                let var = self.fold_bound_var_domain(var);
                let body = self.fold_single_bound_body(&var, *body);
                Spanned::new(Expr::SetFilter(var, body), span)
            }
            Expr::Lambda(params, body) => {
                let mark = self.bound_names.mark();
                self.bound_names
                    .push_names(params.iter().map(|param| param.node.clone()));
                let body = Box::new(self.fold_expr(*body));
                self.bound_names.pop_to(mark);
                Spanned::new(Expr::Lambda(params, body), span)
            }
            Expr::Let(defs, body) => {
                let mark = self.bound_names.mark();
                self.bound_names
                    .push_names(defs.iter().map(|def| def.name.node.clone()));
                let defs = defs
                    .into_iter()
                    .map(|def| {
                        let param_mark = self.bound_names.mark();
                        self.bound_names
                            .push_names(def.params.iter().map(|param| param.name.node.clone()));
                        let body = self.fold_expr(def.body);
                        self.bound_names.pop_to(param_mark);
                        OperatorDef {
                            name: def.name,
                            params: def.params,
                            body,
                            local: def.local,
                            contains_prime: def.contains_prime,
                            guards_depend_on_prime: def.guards_depend_on_prime,
                            has_primed_param: def.has_primed_param,
                            is_recursive: def.is_recursive,
                            self_call_count: def.self_call_count,
                        }
                    })
                    .collect();
                let body = Box::new(self.fold_expr(*body));
                self.bound_names.pop_to(mark);
                Spanned::new(Expr::Let(defs, body), span)
            }
            other => Spanned::new(self.fold_expr_inner(other), span),
        }
    }
}

fn proof_operator_call_capture_safe(
    def: &OperatorDef,
    args: &[Spanned<Expr>],
    call_site_bound_names: &BoundNameStack,
) -> bool {
    proof_operator_body_free_vars_capture_safe(def, call_site_bound_names)
        && proof_operator_args_capture_safe(def, args)
}

fn proof_operator_body_free_vars_capture_safe(
    def: &OperatorDef,
    call_site_bound_names: &BoundNameStack,
) -> bool {
    let mut free = free_vars(&def.body.node);
    for param in &def.params {
        free.remove(param.name.node.as_str());
    }
    free.iter()
        .all(|name| !call_site_bound_names.contains(name.as_str()))
}

fn proof_operator_args_capture_safe(def: &OperatorDef, args: &[Spanned<Expr>]) -> bool {
    def.params.iter().zip(args.iter()).all(|(param, arg)| {
        let free = free_vars(&arg.node);
        !substitution_would_capture(
            &def.body.node,
            &param.name.node,
            &free,
            &mut BoundNameStack::new(),
        )
    })
}

fn proof_bound_var_names(var: &BoundVar) -> Vec<String> {
    let mut names = vec![var.name.node.clone()];
    match &var.pattern {
        None => {}
        Some(BoundPattern::Var(name)) => names.push(name.node.clone()),
        Some(BoundPattern::Tuple(parts)) => {
            names.extend(parts.iter().map(|part| part.node.clone()));
        }
    }
    names
}

fn proof_domain_scalar_value(
    expr: &Expr,
    constants: &tla_core::kani_types::HashMap<NameId, crate::Value>,
    op_replacements: &tla_core::kani_types::HashMap<String, String>,
) -> Option<crate::Value> {
    match expr {
        Expr::Bool(value) => Some(crate::Value::Bool(*value)),
        Expr::Int(value) => value
            .to_i64()
            .map(crate::Value::SmallInt)
            .or_else(|| Some(crate::Value::Int(Rp::new(value.clone())))),
        Expr::String(value) => Some(crate::Value::String(Rp::from(value.as_str()))),
        Expr::Ident(name, name_id) => {
            proof_domain_scalar_constant_value(name, *name_id, constants, op_replacements)
        }
        _ => None,
    }
}

fn proof_domain_int_value(
    expr: &Expr,
    constants: &tla_core::kani_types::HashMap<NameId, crate::Value>,
    op_replacements: &tla_core::kani_types::HashMap<String, String>,
) -> Option<i64> {
    // Fold simple constant integer arithmetic so range-domain bounds written in
    // terms of model constants resolve (e.g. `Node == 0 .. N-1`, `1 .. N`,
    // `0 .. 2*N`). These are exact integer operations over already-resolved
    // constants, so the fold is sound; non-integer / overflowing forms return
    // `None` and fall back to the interpreter rather than guessing a domain.
    match expr {
        Expr::Add(left, right) => {
            let lo = proof_domain_int_value(&left.node, constants, op_replacements)?;
            let ro = proof_domain_int_value(&right.node, constants, op_replacements)?;
            return lo.checked_add(ro);
        }
        Expr::Sub(left, right) => {
            let lo = proof_domain_int_value(&left.node, constants, op_replacements)?;
            let ro = proof_domain_int_value(&right.node, constants, op_replacements)?;
            return lo.checked_sub(ro);
        }
        Expr::Mul(left, right) => {
            let lo = proof_domain_int_value(&left.node, constants, op_replacements)?;
            let ro = proof_domain_int_value(&right.node, constants, op_replacements)?;
            return lo.checked_mul(ro);
        }
        Expr::Neg(inner) => {
            let v = proof_domain_int_value(&inner.node, constants, op_replacements)?;
            return v.checked_neg();
        }
        _ => {}
    }
    let value = proof_domain_scalar_value(expr, constants, op_replacements)?;
    match &value {
        crate::Value::SmallInt(value) => Some(*value),
        crate::Value::Int(value) => value.to_i64(),
        _ => None,
    }
}

fn proof_domain_scalar_constant_value(
    name: &str,
    name_id: NameId,
    constants: &tla_core::kani_types::HashMap<NameId, crate::Value>,
    op_replacements: &tla_core::kani_types::HashMap<String, String>,
) -> Option<crate::Value> {
    let resolved = resolve_proof_op_name(name, op_replacements)?;
    let resolved_id = intern_name(resolved);
    if let Some(value) = constants
        .get(&resolved_id)
        .filter(|value| is_proof_scalar_value(value))
        .cloned()
    {
        return Some(value);
    }

    let id = if name_id == NameId::INVALID {
        intern_name(name)
    } else {
        name_id
    };
    if id != resolved_id && !op_replacements.contains_key(name) {
        constants
            .get(&id)
            .filter(|value| is_proof_scalar_value(value))
            .cloned()
    } else {
        None
    }
}

fn normalize_proof_domain_values(mut values: Vec<crate::Value>) -> Option<Vec<crate::Value>> {
    if values.is_empty() || values.len() > 63 || !values.iter().all(is_proof_scalar_value) {
        return None;
    }
    values.sort();
    values.dedup();
    Some(values)
}

fn is_proof_scalar_value(value: &crate::Value) -> bool {
    matches!(
        value,
        crate::Value::Bool(_)
            | crate::Value::SmallInt(_)
            | crate::Value::Int(_)
            | crate::Value::String(_)
            | crate::Value::ModelValue(_)
    )
}

#[derive(Clone)]
struct ActionTaggedRangeCandidate {
    var_idx: usize,
    domain: Arc<[crate::Value]>,
    scalar_type: crate::state::SlotType,
    set_universe: Vec<crate::state::FlatScalarValue>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum ActionTaggedRangeShape {
    Scalar,
    Set,
    TaggedScalarSetRead,
    CandidateFunction { set_update: bool },
}

#[derive(Default)]
struct ActionTaggedRangeScan {
    saw_store: bool,
    saw_set_write: bool,
}

#[derive(Clone, Copy)]
struct ActionTypedLoadImm {
    value: i64,
    scalar_type: crate::state::SlotType,
}

fn action_shape_is_scalar(shape: Option<&ActionTaggedRangeShape>) -> bool {
    shape == Some(&ActionTaggedRangeShape::Scalar)
}

fn action_shape_is_finite_set(shape: Option<&ActionTaggedRangeShape>) -> bool {
    matches!(
        shape,
        Some(ActionTaggedRangeShape::Set | ActionTaggedRangeShape::TaggedScalarSetRead)
    )
}

fn action_shape_func_except_replacement_set_update(
    shape: Option<&ActionTaggedRangeShape>,
) -> Option<bool> {
    match shape {
        Some(ActionTaggedRangeShape::Scalar | ActionTaggedRangeShape::TaggedScalarSetRead) => {
            Some(false)
        }
        Some(ActionTaggedRangeShape::Set) => Some(true),
        _ => None,
    }
}

fn action_tagged_range_candidates_from_seed_state(
    seed_state: &crate::state::ArrayState,
) -> Vec<ActionTaggedRangeCandidate> {
    seed_state
        .values()
        .iter()
        .enumerate()
        .filter_map(|(var_idx, compact)| {
            let value = crate::Value::from(compact);
            let crate::Value::Func(func) = &value else {
                return None;
            };
            if func.domain_is_empty() || func.domain_len() > 63 {
                return None;
            }

            let mut scalar_type = None;
            let mut domain = Vec::with_capacity(func.domain_len());
            let mut set_universe = Vec::with_capacity(func.domain_len());
            for (key, value) in func.iter() {
                domain.push(key.clone());
                set_universe.push(action_flat_scalar_from_value(key)?);
                let value_type = action_scalar_slot_type(value)?;
                if value_type == crate::state::SlotType::Int {
                    return None;
                }
                if scalar_type
                    .replace(value_type)
                    .is_some_and(|existing| existing != value_type)
                {
                    return None;
                }
            }

            Some(ActionTaggedRangeCandidate {
                var_idx,
                domain: Arc::from(domain.into_boxed_slice()),
                scalar_type: scalar_type?,
                set_universe,
            })
        })
        .collect()
}

fn action_seed_scalar_types(
    seed_state: &crate::state::ArrayState,
) -> Vec<Option<crate::state::SlotType>> {
    seed_state
        .values()
        .iter()
        .map(|compact| action_scalar_slot_type(&crate::Value::from(compact)))
        .collect()
}

fn action_scalar_slot_type(value: &crate::Value) -> Option<crate::state::SlotType> {
    match value {
        crate::Value::Bool(_) => Some(crate::state::SlotType::Bool),
        crate::Value::SmallInt(_) | crate::Value::Int(_) => Some(crate::state::SlotType::Int),
        crate::Value::String(_) => Some(crate::state::SlotType::String),
        crate::Value::ModelValue(_) => Some(crate::state::SlotType::ModelValue),
        _ => None,
    }
}

fn action_flat_scalar_from_value(value: &crate::Value) -> Option<crate::state::FlatScalarValue> {
    match value {
        crate::Value::Bool(value) => Some(crate::state::FlatScalarValue::Bool(*value)),
        crate::Value::SmallInt(value) => Some(crate::state::FlatScalarValue::Int(*value)),
        crate::Value::Int(value) => value.to_i64().map(crate::state::FlatScalarValue::Int),
        crate::Value::String(value) => {
            Some(crate::state::FlatScalarValue::String(value.clone().into()))
        }
        crate::Value::ModelValue(value) => {
            Some(crate::state::FlatScalarValue::ModelValue(value.clone().into()))
        }
        _ => None,
    }
}

fn action_value_fits_candidate_set(
    value: &crate::Value,
    candidate: &ActionTaggedRangeCandidate,
) -> bool {
    let Some(set) = value.to_sorted_set() else {
        return false;
    };
    set.iter().all(|element| {
        action_flat_scalar_from_value(element)
            .is_some_and(|flat| candidate.set_universe.contains(&flat))
    })
}

fn action_value_fits_candidate_scalar(
    value: &crate::Value,
    candidate: &ActionTaggedRangeCandidate,
) -> bool {
    action_scalar_slot_type(value) == Some(candidate.scalar_type)
}

fn action_shape_for_const_value(
    value: &crate::Value,
    candidate: &ActionTaggedRangeCandidate,
) -> Option<ActionTaggedRangeShape> {
    if action_value_fits_candidate_scalar(value, candidate) {
        Some(ActionTaggedRangeShape::Scalar)
    } else if action_value_fits_candidate_set(value, candidate) {
        Some(ActionTaggedRangeShape::Set)
    } else {
        None
    }
}

fn action_shape_for_load_imm(
    value: i64,
    candidate: &ActionTaggedRangeCandidate,
    typed_load_imms: &[ActionTypedLoadImm],
) -> Option<ActionTaggedRangeShape> {
    if candidate
        .set_universe
        .iter()
        .any(|element| matches!(element, crate::state::FlatScalarValue::Int(n) if *n == value))
    {
        return Some(ActionTaggedRangeShape::Scalar);
    }

    typed_load_imms
        .iter()
        .any(|typed| {
            typed.value == value
                && typed.scalar_type == candidate.scalar_type
                && candidate.set_universe.iter().any(|element| {
                    action_typed_load_imm_matches_universe_member(value, candidate, element)
                })
        })
        .then_some(ActionTaggedRangeShape::Scalar)
}

fn action_typed_load_imm_matches_universe_member(
    value: i64,
    candidate: &ActionTaggedRangeCandidate,
    element: &crate::state::FlatScalarValue,
) -> bool {
    let Ok(name_id) = u32::try_from(value) else {
        return false;
    };
    match (candidate.scalar_type, element) {
        (crate::state::SlotType::String, crate::state::FlatScalarValue::String(name))
        | (crate::state::SlotType::ModelValue, crate::state::FlatScalarValue::ModelValue(name)) => {
            intern_name(name).0 == name_id
        }
        _ => false,
    }
}

fn action_constant_function_shape(
    func: &tla_tir::bytecode::BytecodeFunction,
    chunk: &tla_tir::bytecode::BytecodeChunk,
    candidate: &ActionTaggedRangeCandidate,
    typed_load_imms: &[ActionTypedLoadImm],
    depth: usize,
) -> Option<ActionTaggedRangeShape> {
    use tla_tir::bytecode::Opcode;

    if func.arity != 0 || depth > 8 {
        return None;
    }

    let mut shapes = BTreeMap::<u8, ActionTaggedRangeShape>::new();
    for op in &func.instructions {
        match *op {
            Opcode::LoadImm { rd, value } => {
                if let Some(shape) = action_shape_for_load_imm(value, candidate, typed_load_imms) {
                    shapes.insert(rd, shape);
                } else {
                    shapes.remove(&rd);
                }
            }
            Opcode::LoadBool { rd, value } => {
                if candidate.scalar_type == crate::state::SlotType::Bool {
                    shapes.insert(rd, ActionTaggedRangeShape::Scalar);
                } else {
                    let bool_value = crate::Value::Bool(value);
                    if action_value_fits_candidate_set(&bool_value, candidate) {
                        shapes.insert(rd, ActionTaggedRangeShape::Set);
                    } else {
                        shapes.remove(&rd);
                    }
                }
            }
            Opcode::LoadConst { rd, idx } => {
                if let Some(shape) =
                    action_shape_for_const_value(chunk.constants.get_value(idx), candidate)
                {
                    shapes.insert(rd, shape);
                } else {
                    shapes.remove(&rd);
                }
            }
            Opcode::Move { rd, rs } => {
                if let Some(shape) = shapes.get(&rs).cloned() {
                    shapes.insert(rd, shape);
                } else {
                    shapes.remove(&rd);
                }
            }
            Opcode::SetEnum { rd, start, count } => {
                let is_set = (0..count).all(|offset| {
                    action_shape_is_scalar(
                        start.checked_add(offset).and_then(|reg| shapes.get(&reg)),
                    )
                });
                if is_set {
                    shapes.insert(rd, ActionTaggedRangeShape::Set);
                } else {
                    shapes.remove(&rd);
                }
            }
            Opcode::SetDiff { rd, r1, r2 }
            | Opcode::SetIntersect { rd, r1, r2 }
            | Opcode::SetUnion { rd, r1, r2 } => {
                if action_shape_is_finite_set(shapes.get(&r1))
                    && action_shape_is_finite_set(shapes.get(&r2))
                {
                    shapes.insert(rd, ActionTaggedRangeShape::Set);
                } else {
                    shapes.remove(&rd);
                }
            }
            Opcode::Call {
                rd,
                op_idx,
                argc: 0,
                ..
            } => {
                if let Some(shape) = action_constant_function_shape(
                    chunk.get_function(op_idx),
                    chunk,
                    candidate,
                    typed_load_imms,
                    depth + 1,
                ) {
                    shapes.insert(rd, shape);
                } else {
                    shapes.remove(&rd);
                }
            }
            Opcode::Ret { rs } => return shapes.get(&rs).cloned(),
            _ => {
                if let Some(rd) = op.dest_register() {
                    shapes.remove(&rd);
                }
            }
        }
    }

    None
}

fn action_function_supports_tagged_range_candidate(
    func: &tla_tir::bytecode::BytecodeFunction,
    chunk: &tla_tir::bytecode::BytecodeChunk,
    candidate: &ActionTaggedRangeCandidate,
    seed_scalar_types: &[Option<crate::state::SlotType>],
    typed_load_imms: &[ActionTypedLoadImm],
) -> Option<ActionTaggedRangeScan> {
    use tla_tir::bytecode::Opcode;

    let mut shapes = BTreeMap::<u8, ActionTaggedRangeShape>::new();
    for reg in 0..func.arity {
        shapes.insert(reg, ActionTaggedRangeShape::Scalar);
    }
    let mut scan = ActionTaggedRangeScan::default();

    for op in &func.instructions {
        match *op {
            Opcode::LoadImm { rd, value } => {
                if let Some(shape) = action_shape_for_load_imm(value, candidate, typed_load_imms) {
                    shapes.insert(rd, shape);
                } else {
                    shapes.remove(&rd);
                }
            }
            Opcode::LoadBool { rd, value } => {
                if candidate.scalar_type == crate::state::SlotType::Bool {
                    shapes.insert(rd, ActionTaggedRangeShape::Scalar);
                } else {
                    let bool_value = crate::Value::Bool(value);
                    if action_value_fits_candidate_set(&bool_value, candidate) {
                        shapes.insert(rd, ActionTaggedRangeShape::Set);
                    } else {
                        shapes.remove(&rd);
                    }
                }
            }
            Opcode::LoadConst { rd, idx } => {
                if let Some(shape) =
                    action_shape_for_const_value(chunk.constants.get_value(idx), candidate)
                {
                    shapes.insert(rd, shape);
                } else {
                    shapes.remove(&rd);
                }
            }
            Opcode::LoadVar { rd, var_idx } => {
                if usize::from(var_idx) == candidate.var_idx {
                    shapes.insert(
                        rd,
                        ActionTaggedRangeShape::CandidateFunction { set_update: false },
                    );
                } else if seed_scalar_types
                    .get(usize::from(var_idx))
                    .copied()
                    .flatten()
                    == Some(candidate.scalar_type)
                {
                    shapes.insert(rd, ActionTaggedRangeShape::Scalar);
                } else {
                    shapes.remove(&rd);
                }
            }
            Opcode::Move { rd, rs } => {
                if let Some(shape) = shapes.get(&rs).cloned() {
                    shapes.insert(rd, shape);
                } else {
                    shapes.remove(&rd);
                }
            }
            Opcode::SetEnum { rd, start, count } => {
                let is_set = (0..count).all(|offset| {
                    action_shape_is_scalar(
                        start.checked_add(offset).and_then(|reg| shapes.get(&reg)),
                    )
                });
                if is_set {
                    shapes.insert(rd, ActionTaggedRangeShape::Set);
                } else {
                    shapes.remove(&rd);
                }
            }
            Opcode::SetDiff { rd, r1, r2 }
            | Opcode::SetIntersect { rd, r1, r2 }
            | Opcode::SetUnion { rd, r1, r2 } => {
                if action_shape_is_finite_set(shapes.get(&r1))
                    && action_shape_is_finite_set(shapes.get(&r2))
                {
                    shapes.insert(rd, ActionTaggedRangeShape::Set);
                } else {
                    shapes.remove(&rd);
                }
            }
            Opcode::FuncApply { rd, func, arg } => {
                if matches!(
                    shapes.get(&func),
                    Some(ActionTaggedRangeShape::CandidateFunction { .. })
                ) && action_shape_is_scalar(shapes.get(&arg))
                {
                    shapes.insert(rd, ActionTaggedRangeShape::TaggedScalarSetRead);
                } else {
                    shapes.remove(&rd);
                }
            }
            Opcode::FuncExcept {
                rd,
                func,
                path,
                val,
            } => {
                let set_update = match (shapes.get(&func), shapes.get(&path)) {
                    (
                        Some(ActionTaggedRangeShape::CandidateFunction { set_update }),
                        Some(ActionTaggedRangeShape::Scalar),
                    ) => action_shape_func_except_replacement_set_update(shapes.get(&val))
                        .map(|replacement_set_update| *set_update || replacement_set_update),
                    _ => None,
                };
                if let Some(set_update) = set_update {
                    shapes.insert(rd, ActionTaggedRangeShape::CandidateFunction { set_update });
                } else {
                    shapes.remove(&rd);
                }
            }
            Opcode::Call {
                rd,
                op_idx,
                argc: 0,
                ..
            } => {
                if let Some(shape) = action_constant_function_shape(
                    chunk.get_function(op_idx),
                    chunk,
                    candidate,
                    typed_load_imms,
                    0,
                ) {
                    shapes.insert(rd, shape);
                } else {
                    shapes.remove(&rd);
                }
            }
            Opcode::ExistsBegin {
                rd,
                r_binding,
                r_domain,
                ..
            }
            | Opcode::ForallBegin {
                rd,
                r_binding,
                r_domain,
                ..
            } => {
                if action_shape_is_finite_set(shapes.get(&r_domain)) {
                    shapes.insert(r_binding, ActionTaggedRangeShape::Scalar);
                } else {
                    shapes.remove(&r_binding);
                }
                shapes.remove(&rd);
            }
            Opcode::StoreVar { var_idx, rs } if usize::from(var_idx) == candidate.var_idx => {
                let Some(ActionTaggedRangeShape::CandidateFunction { set_update }) =
                    shapes.get(&rs)
                else {
                    return None;
                };
                scan.saw_store = true;
                scan.saw_set_write |= *set_update;
            }
            _ => {
                if let Some(rd) = op.dest_register() {
                    shapes.remove(&rd);
                }
            }
        }
    }

    Some(scan)
}

fn action_bytecode_supports_tagged_range_candidate(
    bytecode: &tla_eval::bytecode_vm::CompiledBytecode,
    candidate: &ActionTaggedRangeCandidate,
    seed_scalar_types: &[Option<crate::state::SlotType>],
    typed_load_imms_by_action: &BTreeMap<String, Vec<ActionTypedLoadImm>>,
) -> bool {
    if !bytecode.failed.is_empty() {
        return false;
    }

    let mut saw_store = false;
    let mut saw_set_write = false;
    for (action_name, &func_idx) in &bytecode.op_indices {
        let func = bytecode.chunk.get_function(func_idx);
        let typed_load_imms = typed_load_imms_by_action
            .get(action_name)
            .map(Vec::as_slice)
            .unwrap_or(&[]);
        let Some(scan) = action_function_supports_tagged_range_candidate(
            func,
            &bytecode.chunk,
            candidate,
            seed_scalar_types,
            typed_load_imms,
        ) else {
            return false;
        };
        saw_store |= scan.saw_store;
        saw_set_write |= scan.saw_set_write;
    }
    saw_store && saw_set_write
}

fn action_typed_load_imms_by_split_action(
    meta: Option<&[super::mc_struct::ActionInstanceMeta]>,
) -> BTreeMap<String, Vec<ActionTypedLoadImm>> {
    let mut by_action = BTreeMap::new();
    let Some(meta) = meta else {
        return by_action;
    };

    for action in meta {
        if action.bindings.is_empty() {
            continue;
        }
        let Some(base_name) = action.name.as_deref() else {
            continue;
        };
        let Some(key) = tla_jit_abi::binding_key_for_bindings(base_name, &action.bindings) else {
            continue;
        };
        let mut typed = Vec::new();
        action_collect_typed_load_imms(&action.bindings, &mut typed);
        action_collect_typed_load_imms(&action.formal_bindings, &mut typed);
        if !typed.is_empty() {
            by_action.entry(key).or_default().extend(typed);
        }
    }

    by_action
}

fn action_collect_typed_load_imms(
    bindings: &[(Arc<str>, crate::Value)],
    typed: &mut Vec<ActionTypedLoadImm>,
) {
    for (_, value) in bindings {
        match value {
            crate::Value::String(name) => typed.push(ActionTypedLoadImm {
                value: i64::from(intern_name(name).0),
                scalar_type: crate::state::SlotType::String,
            }),
            crate::Value::ModelValue(name) => typed.push(ActionTypedLoadImm {
                value: i64::from(intern_name(name).0),
                scalar_type: crate::state::SlotType::ModelValue,
            }),
            _ => {}
        }
    }
}

/// Build a state-variable name→index map from the model checker's `VarRegistry`.
///
/// Used to resolve INSTANCE-imported operator bodies whose variable references
/// (e.g. the instance variable `ringbuffer`, mapped to the parent's same-named
/// state var via the instance's implicit substitution) are lowered by TIR
/// seeding as bare `Ident`s. The bytecode compiler's `compile_name_expr`
/// fallback uses this map to emit `LoadVar`/`StoreVar` for those names — the
/// same outer slot the interpreter resolves them to.
fn state_var_index_map(
    registry: &crate::var_index::VarRegistry,
) -> std::collections::HashMap<String, u16> {
    registry
        .names()
        .iter()
        .enumerate()
        .map(|(idx, name)| (name.to_string(), idx as u16))
        .collect()
}

/// Override TIR callee bodies for INSTANCE-imported action operators using the
/// resolved split-action expressions (Part of the INSTANCE name-resolution fix).
///
/// `collect_bytecode_namespace_callees` treats TIR-seeded callee bodies as the
/// authoritative root namespace because they normally preserve INSTANCE
/// substitution context. For action operators imported through an `INSTANCE`
/// layer, however, that seeded body can still reference the *inner* module's
/// (unsubstituted, unresolved) state variables, which the bytecode compiler
/// rejects as `unresolved identifier`.
///
/// The action splitter already produced the fully substituted
/// (`apply_substitutions` + `qualify_instance_ops`) and state-var-resolved body
/// for each such action in `ActionInstanceMeta.expr` — the identical AST the
/// interpreter evaluates. We lower that expression and use it to replace the
/// stale callee body so the native action-bytecode compile path sees the same
/// resolved action.
///
/// Soundness/blast-radius: we only override operators that are NOT defined in the
/// root module (i.e., genuinely imported through an INSTANCE / dependency). For
/// monolithic, root-defined actions the existing seeded body is already correct
/// and is left untouched, so this is a no-op for non-instance specs.
fn override_instance_action_callees_from_split(
    tir_callees: &mut std::collections::HashMap<String, tla_tir::bytecode::CalleeInfo>,
    root: &Module,
    deps: &[&Module],
    meta: Option<&[super::mc_struct::ActionInstanceMeta]>,
) {
    use tla_core::ast::Unit;
    let Some(meta) = meta else {
        return;
    };

    // Names defined directly in the root module are NOT instance-imported.
    let root_op_names: std::collections::HashSet<&str> = root
        .units
        .iter()
        .filter_map(|unit| match &unit.node {
            Unit::Operator(def) => Some(def.name.node.as_str()),
            _ => None,
        })
        .collect();

    let mut env = tla_tir::TirLoweringEnv::new(root);
    for dep in deps {
        env.add_module(dep);
    }

    let mut overridden = std::collections::HashSet::new();
    for action in meta {
        let Some(name) = action.name.as_deref() else {
            continue;
        };
        // Only override genuinely instance-imported (non-root) action operators,
        // and only with arity-0 bodies (split actions are zero-arg disjuncts).
        if root_op_names.contains(name) || overridden.contains(name) {
            continue;
        }
        let Some(expr) = action.expr.as_ref() else {
            continue;
        };
        // The split expression must be self-contained over the outer state vars.
        // Lower it against the root namespace; if lowering fails, leave the
        // existing callee body in place (fail-closed — never substitute a body
        // we cannot lower).
        let Ok(body) = tla_tir::lower_expr_with_env(&env, root, expr) else {
            continue;
        };
        tir_callees.insert(
            name.to_string(),
            tla_tir::bytecode::CalleeInfo {
                params: Vec::new(),
                body: std::sync::Arc::new(body),
                ast_body: Some(tla_tir::nodes::PreservedAstBody(std::sync::Arc::new(
                    expr.clone(),
                ))),
            },
        );
        overridden.insert(name.to_string());
    }
}

fn add_bound_split_action_synthetic_ops(
    root: &mut Module,
    meta: &[super::mc_struct::ActionInstanceMeta],
    action_names: &mut std::collections::HashSet<String>,
    op_defs: &tla_core::OpEnv,
    op_replacements: &tla_core::kani_types::HashMap<String, String>,
) {
    // Group split-action leaves by their native binding key, then DISJOIN every
    // arm sharing a key into the synthetic op body. A DISJUNCTIVE action operator
    // (e.g. TCommit's `Decide(rm) == commitArm \/ abortArm`) is split by the
    // action-instance splitter into ONE leaf per arm, ALL sharing the same
    // (name, bindings) and therefore the same key (`binding_key_for_bindings`
    // depends only on name+bindings). The previous dedup-by-key kept only the
    // FIRST arm and silently dropped the rest — deleting their successors from
    // the native BFS (observed: TCommit native 7 vs interpreter/TLC-with-symmetry
    // 13). Disjoining restores `arm1 \/ arm2 \/ ...`; the downstream next-state
    // action transform re-detects the top-level disjunction and splits it
    // union-exactly into per-arm sub-actions, so the BFS engine unions their
    // successors — bit-identical to the interpreter. A non-disjunctive action is
    // a single-member group and yields the identical body as before.
    let mut key_order: Vec<String> = Vec::new();
    let mut groups: std::collections::HashMap<String, Vec<&super::mc_struct::ActionInstanceMeta>> =
        std::collections::HashMap::new();
    for action in meta {
        if action.bindings.is_empty() {
            continue;
        }
        let Some(base_name) = action.name.as_deref() else {
            continue;
        };
        if action.expr.is_none() {
            continue;
        }
        let Some(key) = tla_jit_abi::binding_key_for_bindings(base_name, &action.bindings) else {
            continue;
        };
        groups
            .entry(key.clone())
            .or_insert_with(|| {
                key_order.push(key.clone());
                Vec::new()
            })
            .push(action);
    }
    for key in key_order {
        let members = &groups[&key];
        let first = members[0];
        // Substitute BOTH the EXISTS quantifier bindings AND the operator's
        // formal-parameter bindings as literals. The split-action leaf `expr`
        // is the body reached after the splitter recursed through the action
        // operator (and any INSTANCE op), so it references the operator's
        // formal parameter names (e.g. `writer`/`reader` for `BeginWrite(w)` /
        // `BeginRead(r)`). Without the formal bindings the synthetic raw split
        // op fails to compile with `unresolved identifier 'writer'`. All members
        // of a group share identical bindings (the key is a pure function of
        // name+bindings) and, being arms of one specialization, identical
        // formal_bindings — so the first member's bindings are exact for the
        // whole group. Both sets are the exact interpreter-bound literal values,
        // so the result is bit-identical to interpreter evaluation.
        let mut all_bindings = first.bindings.clone();
        for (name, value) in &first.formal_bindings {
            if all_bindings.iter().any(|(existing, _)| existing == name) {
                continue;
            }
            all_bindings.push((name.clone(), value.clone()));
        }
        // Fold every arm's expr into one top-level disjunction in recorded
        // (spec) order.
        let mut combined: Option<Spanned<Expr>> = None;
        for m in members {
            let arm = m.expr.as_ref().expect("filtered to Some above").clone();
            combined = Some(match combined {
                None => arm,
                Some(prev) => {
                    let span = prev.span;
                    Spanned::new(Expr::Or(Box::new(prev), Box::new(arm)), span)
                }
            });
        }
        let combined = combined.expect("group has at least one member");
        let span = combined.span;
        let Some(body) = substitute_action_bindings_as_literals(&combined, &all_bindings) else {
            continue;
        };
        let body = inline_action_replacement_ops(body, op_defs, op_replacements);
        root.units.push(Spanned::new(
            tla_core::ast::Unit::Operator(OperatorDef {
                name: Spanned::new(key.clone(), span),
                params: Vec::new(),
                body,
                local: false,
                contains_prime: true,
                guards_depend_on_prime: true,
                has_primed_param: false,
                is_recursive: false,
                self_call_count: 0,
            }),
            span,
        ));
        action_names.insert(key);
    }
}

fn inline_action_replacement_ops(
    expr: Spanned<Expr>,
    op_defs: &tla_core::OpEnv,
    op_replacements: &tla_core::kani_types::HashMap<String, String>,
) -> Spanned<Expr> {
    let mut inliner = ActionReplacementInliner {
        op_defs,
        op_replacements,
        visiting: BTreeSet::new(),
        bound_names: BoundNameStack::new(),
    };
    inliner.fold_expr(expr)
}

struct ActionReplacementInliner<'a> {
    op_defs: &'a tla_core::OpEnv,
    op_replacements: &'a tla_core::kani_types::HashMap<String, String>,
    visiting: BTreeSet<String>,
    bound_names: BoundNameStack,
}

impl ActionReplacementInliner<'_> {
    fn expand_action_replacement_call(
        &mut self,
        name: &str,
        args: &[Spanned<Expr>],
    ) -> Option<Spanned<Expr>> {
        let resolved_name = resolve_proof_op_name(name, self.op_replacements)?;
        let def = self.op_defs.get(resolved_name)?.as_ref();
        if def.params.len() != args.len()
            || def.is_recursive
            || !def.params.iter().all(|param| param.arity == 0)
        {
            return None;
        }

        let is_replacement = self.op_replacements.contains_key(name) || resolved_name != name;
        if !is_replacement && !def.contains_prime && !def.has_primed_param {
            return None;
        }
        if self.visiting.contains(resolved_name)
            || !proof_operator_call_capture_safe(def, args, &self.bound_names)
        {
            return None;
        }

        let mut subs = HashMap::new();
        for (param, arg) in def.params.iter().zip(args.iter()) {
            if subs.insert(param.name.node.as_str(), arg).is_some() {
                return None;
            }
        }

        self.visiting.insert(resolved_name.to_owned());
        let substituted = {
            let mut substituter = SubstituteExpr {
                subs,
                span_policy: SpanPolicy::Preserve,
            };
            substituter.fold_expr(def.body.clone())
        };
        let result = self.fold_expr(substituted);
        self.visiting.remove(resolved_name);
        Some(result)
    }

    fn fold_bound_var_domains_with_sequential_scope(
        &mut self,
        vars: Vec<BoundVar>,
    ) -> Vec<BoundVar> {
        let mark = self.bound_names.mark();
        let mut folded = Vec::with_capacity(vars.len());
        for var in vars {
            let var = self.fold_bound_var_domain(var);
            self.bound_names.push_names(proof_bound_var_names(&var));
            folded.push(var);
        }
        self.bound_names.pop_to(mark);
        folded
    }

    fn fold_bound_var_domain(&mut self, var: BoundVar) -> BoundVar {
        BoundVar {
            name: var.name,
            domain: var.domain.map(|domain| Box::new(self.fold_expr(*domain))),
            pattern: var.pattern,
        }
    }

    fn fold_quantified_body(
        &mut self,
        vars: &[BoundVar],
        body: Spanned<Expr>,
    ) -> Box<Spanned<Expr>> {
        let mark = self.bound_names.mark();
        self.bound_names
            .push_names(vars.iter().flat_map(proof_bound_var_names));
        let body = Box::new(self.fold_expr(body));
        self.bound_names.pop_to(mark);
        body
    }

    fn fold_single_bound_body(
        &mut self,
        var: &BoundVar,
        body: Spanned<Expr>,
    ) -> Box<Spanned<Expr>> {
        let mark = self.bound_names.mark();
        self.bound_names.push_names(proof_bound_var_names(var));
        let body = Box::new(self.fold_expr(body));
        self.bound_names.pop_to(mark);
        body
    }
}

impl ExprFold for ActionReplacementInliner<'_> {
    fn fold_expr(&mut self, expr: Spanned<Expr>) -> Spanned<Expr> {
        let span = expr.span;
        match expr.node {
            Expr::Ident(name, name_id) if !self.bound_names.contains(&name) => self
                .expand_action_replacement_call(&name, &[])
                .unwrap_or_else(|| Spanned::new(Expr::Ident(name, name_id), span)),
            Expr::OpRef(name) if !self.bound_names.contains(&name) => self
                .expand_action_replacement_call(&name, &[])
                .unwrap_or_else(|| Spanned::new(Expr::OpRef(name), span)),
            Expr::Apply(op, args) => {
                let folded_args: Vec<_> = args.into_iter().map(|arg| self.fold_expr(arg)).collect();
                if let Some(name) = applied_proof_op_name(&op.node) {
                    if !self.bound_names.contains(name) {
                        if let Some(expanded) =
                            self.expand_action_replacement_call(name, &folded_args)
                        {
                            return expanded;
                        }
                    }
                }
                Spanned::new(Expr::Apply(self.fold_box(op), folded_args), span)
            }
            Expr::Forall(vars, body) => {
                let vars = self.fold_bound_var_domains_with_sequential_scope(vars);
                let body = self.fold_quantified_body(&vars, *body);
                Spanned::new(Expr::Forall(vars, body), span)
            }
            Expr::Exists(vars, body) => {
                let vars = self.fold_bound_var_domains_with_sequential_scope(vars);
                let body = self.fold_quantified_body(&vars, *body);
                Spanned::new(Expr::Exists(vars, body), span)
            }
            Expr::FuncDef(vars, body) => {
                let vars = self.fold_bound_var_domains_with_sequential_scope(vars);
                let body = self.fold_quantified_body(&vars, *body);
                Spanned::new(Expr::FuncDef(vars, body), span)
            }
            Expr::SetBuilder(body, vars) => {
                let vars = self.fold_bound_var_domains_with_sequential_scope(vars);
                let body = self.fold_quantified_body(&vars, *body);
                Spanned::new(Expr::SetBuilder(body, vars), span)
            }
            Expr::Choose(var, body) => {
                let var = self.fold_bound_var_domain(var);
                let body = self.fold_single_bound_body(&var, *body);
                Spanned::new(Expr::Choose(var, body), span)
            }
            Expr::SetFilter(var, body) => {
                let var = self.fold_bound_var_domain(var);
                let body = self.fold_single_bound_body(&var, *body);
                Spanned::new(Expr::SetFilter(var, body), span)
            }
            Expr::Lambda(params, body) => {
                let mark = self.bound_names.mark();
                self.bound_names
                    .push_names(params.iter().map(|param| param.node.clone()));
                let body = Box::new(self.fold_expr(*body));
                self.bound_names.pop_to(mark);
                Spanned::new(Expr::Lambda(params, body), span)
            }
            Expr::Let(defs, body) => {
                let mark = self.bound_names.mark();
                self.bound_names
                    .push_names(defs.iter().map(|def| def.name.node.clone()));
                let defs = defs
                    .into_iter()
                    .map(|def| {
                        let param_mark = self.bound_names.mark();
                        self.bound_names
                            .push_names(def.params.iter().map(|param| param.name.node.clone()));
                        let body = self.fold_expr(def.body);
                        self.bound_names.pop_to(param_mark);
                        OperatorDef {
                            name: def.name,
                            params: def.params,
                            body,
                            local: def.local,
                            contains_prime: def.contains_prime,
                            guards_depend_on_prime: def.guards_depend_on_prime,
                            has_primed_param: def.has_primed_param,
                            is_recursive: def.is_recursive,
                            self_call_count: def.self_call_count,
                        }
                    })
                    .collect();
                let body = Box::new(self.fold_expr(*body));
                self.bound_names.pop_to(mark);
                Spanned::new(Expr::Let(defs, body), span)
            }
            other => Spanned::new(self.fold_expr_inner(other), span),
        }
    }
}

/// Print the bytecode listing of action `name` (function `func_idx` in
/// `chunk`) plus every transitively-reachable callee function, one opcode per
/// line with its PC. Debug aid behind `TY_BYTECODE_DUMP_ACTION`; purely
/// observational.
fn dump_action_bytecode(name: &str, func_idx: u16, chunk: &tla_tir::bytecode::BytecodeChunk) {
    let mut visited: std::collections::BTreeSet<u16> = std::collections::BTreeSet::new();
    let mut queue: Vec<u16> = vec![func_idx];
    while let Some(idx) = queue.pop() {
        if !visited.insert(idx) {
            continue;
        }
        let Some(func) = chunk.functions.get(idx as usize) else {
            continue;
        };
        eprintln!(
            "[bytecode-dump] action '{name}': function #{idx} '{}' arity={} ({} ops)",
            func.name,
            func.arity,
            func.instructions.len()
        );
        for (pc, op) in func.instructions.iter().enumerate() {
            eprintln!("[bytecode-dump]   pc {pc}: {op:?}");
            if let tla_tir::bytecode::Opcode::Call { op_idx, .. } = op {
                queue.push(*op_idx);
            }
        }
    }
}

fn substitute_action_bindings_as_literals(
    expr: &Spanned<Expr>,
    bindings: &[(Arc<str>, crate::Value)],
) -> Option<Spanned<Expr>> {
    let substitutions = bindings
        .iter()
        .map(|(name, value)| {
            let literal = action_binding_literal_expr(value)?;
            Some(tla_core::ast::Substitution {
                from: Spanned::dummy(name.to_string()),
                to: Spanned::dummy(literal),
            })
        })
        .collect::<Option<Vec<_>>>()?;
    Some(tla_core::apply_substitutions_v(expr, &substitutions))
}

fn action_binding_literal_expr(value: &crate::Value) -> Option<Expr> {
    match value {
        crate::Value::Bool(value) => Some(Expr::Bool(*value)),
        crate::Value::SmallInt(value) => Some(Expr::Int(num_bigint::BigInt::from(*value))),
        crate::Value::Int(value) => Some(Expr::Int((**value).clone())),
        crate::Value::String(value) => Some(Expr::String(value.to_string())),
        crate::Value::ModelValue(value) => {
            Some(Expr::Ident(value.to_string(), tla_core::NameId::INVALID))
        }
        crate::Value::Interval(interval) => Some(Expr::Range(
            Box::new(Spanned::dummy(Expr::Int(interval.low().clone()))),
            Box::new(Spanned::dummy(Expr::Int(interval.high().clone()))),
        )),
        crate::Value::Tuple(values) => values
            .iter()
            .map(|value| action_binding_literal_expr(value).map(Spanned::dummy))
            .collect::<Option<Vec<_>>>()
            .map(Expr::Tuple),
        crate::Value::Seq(values) => values
            .iter()
            .map(|value| action_binding_literal_expr(value).map(Spanned::dummy))
            .collect::<Option<Vec<_>>>()
            .map(Expr::Tuple),
        crate::Value::Set(values) => values
            .iter()
            .map(|value| action_binding_literal_expr(value).map(Spanned::dummy))
            .collect::<Option<Vec<_>>>()
            .map(Expr::SetEnum),
        crate::Value::Record(fields) => fields
            .iter_str()
            .map(|(name, value)| {
                Some((
                    Spanned::dummy(name.to_string()),
                    Spanned::dummy(action_binding_literal_expr(value)?),
                ))
            })
            .collect::<Option<Vec<_>>>()
            .map(Expr::Record),
        _ => None,
    }
}

impl ModelChecker<'_> {
    fn layout_inference_constants(&self) -> tla_core::kani_types::HashMap<NameId, crate::Value> {
        let mut constants = self.ctx.precomputed_constants().clone();
        for (name, value) in self.ctx.env() {
            let name_id = intern_name(name.as_ref());
            if self.ctx.var_registry().get_by_name_id(name_id).is_some() {
                continue;
            }
            if !constants.contains_key(&name_id) {
                constants.insert(name_id, value.clone());
            }
        }
        constants
    }

    fn log_or_panic_flat_roundtrip_verification(
        &self,
        adapter: &crate::state::FlatBfsAdapter,
        sample_state: &crate::state::ArrayState,
        registry: &crate::var_index::VarRegistry,
        roundtrip_ok: bool,
        stats_enabled: bool,
        label: &str,
        sample: &str,
        action: &str,
    ) {
        if roundtrip_ok {
            if stats_enabled {
                eprintln!(
                    "[flat_state] {label} roundtrip verification: PASS spec={} action={action} sample={sample}",
                    self.module.root_name
                );
            }
            return;
        }

        let detail = adapter
            .diagnose_roundtrip(sample_state, registry)
            .unwrap_or_else(|| "no detail available".to_string());
        let message = format!(
            "[flat_state] {label} roundtrip verification: FAIL spec={} action={action} sample={sample} ({detail}) - flat BFS will fall back to Owned entries",
            self.module.root_name
        );
        if super::debug::trust_cg_diag_enabled() {
            panic!("{message}; TY_TRUST_CG_DIAG=1");
        }
        eprintln!("{message}");
    }

    /// Register an inline NEXT expression from a ResolvedSpec.
    ///
    /// Delegates CST lowering and OperatorDef construction to the shared
    /// `checker_ops::lower_inline_next`, then registers the result in both
    /// the module's op_defs and the evaluation context.
    pub fn register_inline_next(&mut self, resolved: &ResolvedSpec) -> Result<(), CheckError> {
        let op_def = match crate::checker_ops::lower_inline_next(
            resolved.next_node.as_ref(),
            self.ctx.var_registry(),
        ) {
            None => return Ok(()),
            Some(result) => result?,
        };

        // Register the operator in our definitions and evaluation context.
        self.module
            .op_defs
            .insert(INLINE_NEXT_NAME.to_string(), op_def.clone());
        self.ctx.define_op(INLINE_NEXT_NAME.to_string(), op_def);

        Ok(())
    }

    /// Validate and cache the VIEW operator name from the configuration.
    ///
    /// Delegates to `checker_ops::validate_view_operator` — the single shared
    /// implementation used by both sequential and parallel checkers (Part of #810).
    pub(super) fn validate_view(&mut self) {
        self.compiled.cached_view_name =
            crate::checker_ops::validate_view_operator(&self.ctx, self.config);
        self.refresh_liveness_mode();
    }

    /// Source-level state predicates that statically prove reachable-state
    /// facts: the configured INVARIANTs *and* the configured CONSTRAINTs.
    ///
    /// Both are sound proof sources for sequence capacity/element layout:
    ///
    /// * An INVARIANT `P` is asserted to hold on every reachable state. If TLC
    ///   finds a reachable state violating `P` it reports a violation, so the
    ///   model is only meaningful when `P` holds everywhere — a `Len(s) <= k`
    ///   or `s \in Seq(T)` invariant therefore certifies the bound/shape.
    /// * A CONSTRAINT `C` is enforced as successor *pruning*: TLC never explores
    ///   successors of a state that violates `C`. The explored state space is
    ///   exactly the `C`-satisfying sub-graph. A `Len(s) <= k` constraint (the
    ///   canonical `qConstraint == Len(q) \leq qLen` idiom) therefore bounds the
    ///   length of `s` across *every explored state* by construction.
    ///
    /// The fixed-capacity flat layout reserves exactly `k` element slots, and
    /// `value_fits_flat_value_layout` rejects any value with `Len > k`. So the
    /// set of states a capacity-`k` layout can represent is identical to the set
    /// the `Len(s) <= k` constraint keeps. Even if a successor with `Len = k+1`
    /// is generated, it is pruned (interpreter/per-parent path evaluates the
    /// constraint before flat-encoding) or rejected by `array_state_fits_layout`
    /// (native fast path) — never silently truncated — so flat dedup stays exact
    /// and the explored state count is byte-identical to the unbounded layout.
    fn sequence_proof_source_predicates(&self) -> impl Iterator<Item = &String> {
        self.config
            .invariants
            .iter()
            .chain(self.config.constraints.iter())
    }

    fn configured_sequence_capacity_proofs(&self) -> Vec<crate::state::SequenceCapacityProof> {
        let mut proofs = Vec::new();
        let proof_domains = self.named_homogeneous_proof_domains();
        let op_defs = &self.ctx.shared().ops;
        let op_replacements = self.ctx.op_replacements();
        // `TY_SEQ_CAPACITY_PROOF` (default OFF): enable folding a STATE-FREE
        // right-hand side of a `Len(seq) <= RHS` checked bound to a compile-time
        // constant, so a growing sequence bounded by e.g. `Cardinality(ConstSet)`
        // or `K - 1` carries a proven capacity for the trust-cg growing-sequence
        // lowering. The const-level evaluator's dependency tracker rejects any
        // current/next-state read, so a state-dependent RHS never folds and the
        // sequence fails closed (Observed). Constant/literal RHS is handled
        // flag-independently as before.
        let seq_capacity_proof =
            matches!(std::env::var("TY_SEQ_CAPACITY_PROOF").as_deref(), Ok("1"));
        let bound_eval = |expr: &Expr| -> Option<usize> {
            let spanned = Spanned::dummy(expr.clone());
            let value = crate::eval::try_eval_const_level(&self.ctx, &spanned)?;
            value_usize_bound(&value)
        };
        let bound_eval_hook: Option<&dyn Fn(&Expr) -> Option<usize>> = if seq_capacity_proof {
            Some(&bound_eval)
        } else {
            None
        };
        for invariant in self.sequence_proof_source_predicates() {
            let Some((resolved_name, def)) =
                proof_safe_zero_arg_op_def(invariant, op_defs, op_replacements)
            else {
                continue;
            };
            let mut scope = ProofScope::default();
            let mut visiting = BTreeSet::from([resolved_name.to_owned()]);
            let lowered = lower_proof_operator_wrappers(
                &self.ctx,
                &def.body,
                op_defs,
                op_replacements,
                BTreeSet::from([resolved_name.to_owned()]),
            );
            collect_sequence_capacity_proofs(
                &lowered.node,
                invariant,
                self.ctx.var_registry(),
                self.ctx.precomputed_constants(),
                op_defs,
                op_replacements,
                &proof_domains,
                bound_eval_hook,
                &mut scope,
                &mut visiting,
                &mut proofs,
            );
        }
        proofs
    }

    /// Universe proofs from CHECKED invariants only (`v \in Seq(D)` with a
    /// finite constant `D`). Constraints are deliberately excluded: a state
    /// CONSTRAINT prunes exploration but violating states are still stored, so
    /// it does not bound stored-state shapes.
    fn configured_sequence_universe_proofs(&self) -> Vec<crate::state::SequenceUniverseProof> {
        let mut proofs = Vec::new();
        let op_defs = &self.ctx.shared().ops;
        let op_replacements = self.ctx.op_replacements();
        let eval_const_set = self.duplicate_free_eval_const_set();
        for invariant in &self.config.invariants {
            let Some((resolved_name, def)) =
                proof_safe_zero_arg_op_def(invariant, op_defs, op_replacements)
            else {
                continue;
            };
            let lowered = lower_proof_operator_wrappers(
                &self.ctx,
                &def.body,
                op_defs,
                op_replacements,
                BTreeSet::from([resolved_name.to_owned()]),
            );
            crate::state::collect_sequence_universe_proofs(
                &lowered.node,
                invariant,
                self.ctx.var_registry(),
                op_replacements,
                &eval_const_set,
                &mut proofs,
            );
        }
        proofs
    }

    /// Constant-level "evaluate to a finite set" hook shared by the
    /// duplicate-free sequence analysis. `try_eval_const_level`'s dependency
    /// tracking rejects any current/next-state read, so a state-dependent
    /// expression can never leak a wrong universe.
    fn duplicate_free_eval_const_set(&self) -> impl Fn(&Expr) -> Option<Vec<crate::Value>> + '_ {
        move |expr: &Expr| -> Option<Vec<crate::Value>> {
            let spanned = Spanned::dummy(expr.clone());
            let value = crate::eval::try_eval_const_level(&self.ctx, &spanned)?;
            if !value.is_set() || !value.is_finite_set() {
                return None;
            }
            let iter = crate::eval::eval_iter_set(&self.ctx, &value, None).ok()?;
            let mut elems: Vec<crate::Value> = iter.collect();
            elems.sort();
            elems.dedup();
            Some(elems)
        }
    }

    /// Collect duplicate-free bounded-universe sequence capacity proofs
    /// (`Len(v) <= |U|` for `v \in Seq(U)` sequences whose Init/Next writers
    /// provably keep them duplicate-free — see
    /// `layout_inference::collect_duplicate_free_sequence_capacity_proofs`)
    /// and merge them into `sequence_proofs`.
    fn collect_duplicate_free_sequence_capacity_proofs_into(
        &self,
        layout_constants: &tla_core::kani_types::HashMap<NameId, crate::Value>,
        sequence_proofs: &mut Vec<crate::state::SequenceCapacityProof>,
    ) {
        let (Some(init_name), Some(next_name)) = (&self.config.init, &self.config.next) else {
            return;
        };
        let op_defs = &self.ctx.shared().ops;
        let init_name = self.ctx.resolve_op_name(init_name).to_string();
        let next_name = self.ctx.resolve_op_name(next_name).to_string();
        let (Some(init_def), Some(next_def)) = (op_defs.get(&init_name), op_defs.get(&next_name))
        else {
            return;
        };
        let universe_proofs = self.configured_sequence_universe_proofs();
        if universe_proofs.is_empty() {
            return;
        }

        // Flatten INSTANCE `ModuleRef` references in the Init/Next bodies so
        // instanced writes (e.g. `Sched!Schedule`) become structurally
        // visible. Fail-closed: residual `ModuleRef`s poison the analysis via
        // the `flatten_module_ref` hook below.
        let flat_init =
            crate::checker_ops::flatten_property_module_refs(&self.ctx, init_def.body.clone());
        let flat_next =
            crate::checker_ops::flatten_property_module_refs(&self.ctx, next_def.body.clone());

        // Memoized per node address; the cache keeps every flattened body
        // alive (and unique) for the analysis duration so both analysis
        // passes see identical `Prime`-node addresses.
        #[allow(clippy::type_complexity)]
        let flatten_cache: std::cell::RefCell<
            HashMap<usize, Option<std::rc::Rc<Spanned<Expr>>>>,
        > = std::cell::RefCell::new(HashMap::new());
        let flatten_module_ref = |expr: &Expr| -> Option<std::rc::Rc<Spanned<Expr>>> {
            let key = std::ptr::from_ref(expr) as usize;
            if let Some(hit) = flatten_cache.borrow().get(&key) {
                return hit.clone();
            }
            let flattened = crate::checker_ops::flatten_property_module_refs(
                &self.ctx,
                Spanned::dummy(expr.clone()),
            );
            let entry = if crate::state::expr_contains_module_ref(&flattened.node) {
                None
            } else {
                Some(std::rc::Rc::new(flattened))
            };
            flatten_cache.borrow_mut().insert(key, entry.clone());
            entry
        };
        let eval_const_set = self.duplicate_free_eval_const_set();
        let df_debug = matches!(std::env::var("TY_DF_SEQ_DEBUG").as_deref(), Ok("1"));
        // Certificate evaluation runs against a STATE-CLEARED context: the
        // analysis statically verifies the expression is state-free, and any
        // state read this scan somehow missed errors out here (TLCState.Empty
        // semantics) instead of silently sampling a concrete state. Plain
        // `eval_entry` (not `try_eval_const_level`) because the dependency
        // tracker conservatively flags the recursive-function evaluations the
        // permutation-set certificate relies on (e.g. `PermSeqs`' recursive
        // LET function) as inconsistent even when they are state-free.
        let eval_domain_with_set_arg =
            |expr: &Expr, name: &str, arg: &crate::Value| -> Option<Vec<crate::Value>> {
                let bound_ctx = self.ctx.bind(name, arg.clone()).without_state_and_next();
                let spanned = Spanned::dummy(expr.clone());
                let value = match crate::eval::eval_entry(&bound_ctx, &spanned) {
                    Ok(value) => value,
                    Err(err) => {
                        if df_debug {
                            eprintln!("[df-seq] driver: certificate eval error: {err}");
                        }
                        return None;
                    }
                };
                if !value.is_set() || !value.is_finite_set() {
                    return None;
                }
                let iter = crate::eval::eval_iter_set(&bound_ctx, &value, None).ok()?;
                Some(iter.collect())
            };
        let hooks = crate::state::DuplicateFreeSeqProofHooks {
            flatten_module_ref: &flatten_module_ref,
            eval_const_set: &eval_const_set,
            eval_domain_with_set_arg: &eval_domain_with_set_arg,
        };
        crate::state::collect_duplicate_free_sequence_capacity_proofs(
            &flat_init.node,
            &flat_next.node,
            self.ctx.var_registry(),
            layout_constants,
            op_defs,
            self.ctx.op_replacements(),
            &universe_proofs,
            &hooks,
            sequence_proofs,
        );
    }

    /// HEURISTIC (unproven) capacity for a growing `v \in Seq(U)` sequence whose
    /// duplicate-free `Len(v) <= |U|` certificate did NOT fire (e.g. btree's
    /// `toSplit' = <<parent>> \o toSplit` — a prepend, not a DF-preserving write).
    ///
    /// Gated on `TY_SEQ_HEURISTIC_CAPACITY` (default OFF): with the flag off this
    /// is a no-op and every var stays `Observed` (byte-identical). When on, for
    /// each checked `v \in Seq(U)` universe proof whose var did NOT already earn a
    /// certified capacity proof, push a heuristic capacity `= |U|` (the element
    /// universe cardinality — btree `|Nodes| = 8`). This is deliberately a GUESS:
    /// its ONLY soundness guarantee is the fail-closed
    /// `SequenceLengthExceedsCapacity` overflow backstop in the flat write path
    /// (a reachable state longer than `|U|` fails flat serialization → the CLI
    /// re-runs WITHOUT flat storage → interpreter authoritative → never a wrong
    /// count). Runs AFTER the proven-capacity passes so it never shadows a real
    /// proof (and `unique_sequence_proof`'s uniqueness check would otherwise fail
    /// closed on a proven/heuristic conflict). Reuses the same `|U| <=
    /// DF_MAX_UNIVERSE`-style slot-width sanity cap the DF proof applies.
    fn collect_heuristic_sequence_capacity_proofs_into(
        &self,
        sequence_proofs: &mut Vec<crate::state::SequenceCapacityProof>,
    ) {
        if !crate::state::seq_heuristic_capacity_enabled() {
            return;
        }
        // Slot-width sanity cap: a heuristic capacity reserves `1 + |U| *
        // element_slots` i64 slots per state. Bound it exactly like the proven DF
        // analysis (`DF_MAX_UNIVERSE`) so a pathological universe cannot inflate
        // the flat buffer.
        const MAX_HEURISTIC_CAPACITY: usize = 32;
        let universe_proofs = self.configured_sequence_universe_proofs();
        for universe_proof in universe_proofs {
            let var_idx = universe_proof.var_idx;
            let max_len = universe_proof.universe.len();
            if max_len == 0 || max_len > MAX_HEURISTIC_CAPACITY {
                continue;
            }
            // Never shadow (or conflict with) an existing whole-var capacity
            // proof — proven proofs always win, and a proven+heuristic pair would
            // trip `unique_sequence_proof`'s fail-closed uniqueness check.
            if sequence_proofs
                .iter()
                .any(|proof| proof.var_idx == var_idx && proof.path.is_empty())
            {
                continue;
            }
            push_sequence_capacity_proof(
                sequence_proofs,
                crate::state::SequenceCapacityProof {
                    var_idx,
                    path: Vec::new(),
                    max_len,
                    invariant: Arc::clone(&universe_proof.invariant),
                    heuristic: true,
                },
            );
        }
    }

    fn configured_sequence_element_layout_proofs(
        &self,
    ) -> Vec<crate::state::SequenceElementLayoutProof> {
        let mut proofs = Vec::new();
        let proof_domains = self.named_homogeneous_proof_domains();
        let op_defs = &self.ctx.shared().ops;
        let op_replacements = self.ctx.op_replacements();
        for invariant in self.sequence_proof_source_predicates() {
            let Some((resolved_name, def)) =
                proof_safe_zero_arg_op_def(invariant, op_defs, op_replacements)
            else {
                continue;
            };
            let lowered = lower_proof_operator_wrappers(
                &self.ctx,
                &def.body,
                op_defs,
                op_replacements,
                BTreeSet::from([resolved_name.to_owned()]),
            );
            crate::state::collect_sequence_element_layout_proofs_with_ops(
                &lowered.node,
                invariant,
                self.ctx.var_registry(),
                self.ctx.precomputed_constants(),
                &proof_domains,
                op_defs,
                op_replacements,
                &mut proofs,
            );
        }
        proofs
    }

    fn configured_sequence_fixed_domain_type_proofs(
        &self,
    ) -> Vec<crate::state::SequenceFixedDomainTypeProof> {
        let mut proofs = Vec::new();
        let proof_domains = self.named_homogeneous_proof_domains();
        let op_defs = &self.ctx.shared().ops;
        let op_replacements = self.ctx.op_replacements();
        for invariant in self.sequence_proof_source_predicates() {
            let Some((resolved_name, def)) =
                proof_safe_zero_arg_op_def(invariant, op_defs, op_replacements)
            else {
                continue;
            };
            // Preserve config-overridden zero-arg constants (e.g. btree's
            // `NIL == CHOOSE x : x \notin Nodes` overridden by cfg `NIL = nil`)
            // as bare `Ident`s rather than inlining their (overridden-away) CHOOSE
            // body. Inlining loses the name, so a heterogeneous scalar-union range
            // like `[Nodes -> Nodes \cup {NIL}]` degrades to an unresolvable
            // `CHOOSE` element and the TaggedScalarUnion universe cannot be
            // assembled. Preserving the `Ident` lets `const_expr_to_value` resolve
            // it to the SAME precomputed value the interpreter uses (the model
            // value `nil`), which is exactly the sentinel stored in reachable
            // states — so the encoded universe matches the interpreter's values.
            let lowered = lower_proof_operator_wrappers_preserving_precomputed_constants(
                &self.ctx,
                &def.body,
                op_defs,
                op_replacements,
                BTreeSet::from([resolved_name.to_owned()]),
                self.ctx.precomputed_constants(),
            );
            crate::state::collect_sequence_fixed_domain_type_proofs_with_ops(
                &lowered.node,
                invariant,
                self.ctx.var_registry(),
                self.ctx.precomputed_constants(),
                &proof_domains,
                op_defs,
                op_replacements,
                &mut proofs,
            );
        }
        proofs
    }

    fn configured_tagged_scalar_set_range_type_proofs(
        &self,
    ) -> Vec<crate::state::TaggedScalarSetRangeTypeProof> {
        let mut proofs = Vec::new();
        let proof_domains = self.named_homogeneous_proof_domains();
        let op_defs = &self.ctx.shared().ops;
        let op_replacements = self.ctx.op_replacements();
        for invariant in &self.config.invariants {
            let Some((resolved_name, def)) =
                proof_safe_zero_arg_op_def(invariant, op_defs, op_replacements)
            else {
                continue;
            };
            let lowered = lower_proof_operator_wrappers(
                &self.ctx,
                &def.body,
                op_defs,
                op_replacements,
                BTreeSet::from([resolved_name.to_owned()]),
            );
            crate::state::collect_tagged_scalar_set_range_type_proofs_with_ops(
                &lowered.node,
                invariant,
                self.ctx.var_registry(),
                self.ctx.precomputed_constants(),
                &proof_domains,
                op_defs,
                op_replacements,
                &mut proofs,
            );
        }
        if let (Some(init_name), Some(next_name)) = (&self.config.init, &self.config.next) {
            let init_name = self.ctx.resolve_op_name(init_name).to_string();
            let next_name = self.ctx.resolve_op_name(next_name).to_string();
            if let (Some(init_def), Some(next_def)) =
                (op_defs.get(&init_name), op_defs.get(&next_name))
            {
                crate::state::collect_tagged_scalar_set_range_writer_proofs_with_ops(
                    &init_def.as_ref().body.node,
                    &next_def.as_ref().body.node,
                    "Init/Next writer proof",
                    self.ctx.var_registry(),
                    self.ctx.precomputed_constants(),
                    &proof_domains,
                    op_defs,
                    op_replacements,
                    &mut proofs,
                );
            }
        }
        proofs
    }

    fn configured_fixed_scalar_range_type_proofs(
        &self,
    ) -> Vec<crate::state::FixedScalarRangeTypeProof> {
        let mut proofs = Vec::new();
        let proof_domains = self.named_homogeneous_proof_domains();
        let op_defs = &self.ctx.shared().ops;
        let op_replacements = self.ctx.op_replacements();
        for invariant in &self.config.invariants {
            let Some((resolved_name, def)) =
                proof_safe_zero_arg_op_def(invariant, op_defs, op_replacements)
            else {
                continue;
            };
            let lowered = lower_proof_operator_wrappers(
                &self.ctx,
                &def.body,
                op_defs,
                op_replacements,
                BTreeSet::from([resolved_name.to_owned()]),
            );
            crate::state::collect_fixed_scalar_range_type_proofs_with_ops(
                &lowered.node,
                invariant,
                self.ctx.var_registry(),
                self.ctx.precomputed_constants(),
                &proof_domains,
                op_defs,
                op_replacements,
                &mut proofs,
            );
        }
        if let (Some(init_name), Some(next_name)) = (&self.config.init, &self.config.next) {
            let init_name = self.ctx.resolve_op_name(init_name).to_string();
            let next_name = self.ctx.resolve_op_name(next_name).to_string();
            if let (Some(init_def), Some(next_def)) =
                (op_defs.get(&init_name), op_defs.get(&next_name))
            {
                crate::state::collect_fixed_scalar_range_writer_proofs_with_ops(
                    &init_def.as_ref().body.node,
                    &next_def.as_ref().body.node,
                    "Init/Next writer proof",
                    self.ctx.var_registry(),
                    self.ctx.precomputed_constants(),
                    &proof_domains,
                    op_defs,
                    op_replacements,
                    &mut proofs,
                );
                // #43 fail-closed gate: a `TypeOK`-derived FixedScalar range
                // proof only asserts a type membership; it does NOT verify that
                // every writer keeps the function range scalar. If any writer can
                // store a SET (or other non-scalar) into the range, the var must
                // never be a flat-primary scalar slot — otherwise distinct
                // set-valued states alias in the flat fingerprint and the BFS
                // silently undercounts. Drop such proofs here (writer-derived
                // proofs are unaffected: their vars have no non-scalar writer).
                crate::state::retain_writer_corroborated_fixed_scalar_range_proofs(
                    &mut proofs,
                    &init_def.as_ref().body.node,
                    &next_def.as_ref().body.node,
                    self.ctx.var_registry(),
                    self.ctx.precomputed_constants(),
                    &proof_domains,
                    op_defs,
                    op_replacements,
                );
            }
        }
        proofs
    }

    /// Collect whole-variable scalar-union proofs (`focus \in Nodes \cup {NIL}`,
    /// `op \in {"get", "insert", NIL}`) from the checked type invariants, using
    /// the constant-preserving lowering so config-overridden sentinels resolve to
    /// the model value the interpreter uses (see gap (a)). Empty unless the
    /// `TY_TAGGED_SCALAR_UNION` gate is on.
    fn configured_tagged_scalar_union_var_type_proofs(
        &self,
    ) -> Vec<crate::state::TaggedScalarUnionVarTypeProof> {
        let mut proofs = Vec::new();
        let op_defs = &self.ctx.shared().ops;
        let op_replacements = self.ctx.op_replacements();
        for invariant in &self.config.invariants {
            let Some((resolved_name, def)) =
                proof_safe_zero_arg_op_def(invariant, op_defs, op_replacements)
            else {
                continue;
            };
            let lowered = lower_proof_operator_wrappers_preserving_precomputed_constants(
                &self.ctx,
                &def.body,
                op_defs,
                op_replacements,
                BTreeSet::from([resolved_name.to_owned()]),
                self.ctx.precomputed_constants(),
            );
            crate::state::collect_tagged_scalar_union_var_type_proofs_with_ops(
                &lowered.node,
                invariant,
                self.ctx.var_registry(),
                self.ctx.precomputed_constants(),
                op_defs,
                op_replacements,
                &mut proofs,
            );
        }
        proofs
    }

    /// WP-09/Part A: collect tuple-keyed function RANGE scalar-union proofs
    /// (btree `childOf \in [Nodes \X Keys -> Nodes \cup {NIL}]`,
    /// `valOf \in [Nodes \X Keys -> Vals \cup {NIL}]`) from the checked type
    /// invariants, using the constant-preserving lowering so config-overridden
    /// sentinels (`NIL`) resolve to the model value the interpreter uses.
    /// Empty unless the `TY_TAGGED_SCALAR_UNION` gate is on.
    ///
    /// Deliberately NOT #43-writer-corroborated, mirroring the WP-05
    /// whole-variable union override (`configured_tagged_scalar_union_var_type_proofs`),
    /// and unlike the `FixedScalar` range proofs. The #43 veto exists because
    /// the `FixedScalar` slot encode (`value_to_scalar_i64`) is INFALLIBLE and
    /// lossy — a set-valued writer would silently mis-encode and alias. The
    /// union-index encode is fail-closed by construction: a non-scalar,
    /// out-of-universe, missing-key, or domain-drifted value is a hard
    /// serialization/fit error (`write_tuple_keyed_array_slots` union arm /
    /// `value_fits_tuple_keyed_range_slot`), so such a state simply stays on
    /// the compound/interpreter path — never a wrong slot, never an aliased
    /// fingerprint. (Empirically the `NotProvablyScalar` veto rejects EVERY
    /// btree function var because their funcdef writers read the var's own
    /// range (`childOf[n, k]`), which the scanner cannot prove scalar; the
    /// fail-closed encode makes that conservatism unnecessary here.)
    fn configured_tagged_scalar_union_range_type_proofs(
        &self,
    ) -> Vec<crate::state::TaggedScalarUnionRangeTypeProof> {
        let mut proofs = Vec::new();
        let proof_domains = self.named_homogeneous_proof_domains();
        let op_defs = &self.ctx.shared().ops;
        let op_replacements = self.ctx.op_replacements();
        for invariant in &self.config.invariants {
            let Some((resolved_name, def)) =
                proof_safe_zero_arg_op_def(invariant, op_defs, op_replacements)
            else {
                continue;
            };
            let lowered = lower_proof_operator_wrappers_preserving_precomputed_constants(
                &self.ctx,
                &def.body,
                op_defs,
                op_replacements,
                BTreeSet::from([resolved_name.to_owned()]),
                self.ctx.precomputed_constants(),
            );
            crate::state::collect_tagged_scalar_union_range_type_proofs_with_ops(
                &lowered.node,
                invariant,
                self.ctx.var_registry(),
                self.ctx.precomputed_constants(),
                &proof_domains,
                op_defs,
                op_replacements,
                &mut proofs,
            );
        }
        if std::env::var_os("TY_LAYOUT_PROOF_DEBUG").is_some_and(|v| v == "1") {
            eprintln!(
                "[layout-proof] tuple-keyed union-range proofs: collected={}",
                proofs.len()
            );
            for proof in &proofs {
                eprintln!(
                    "[layout-proof] union-range var={} |domain|={} |universe|={} invariant={}",
                    proof.var_idx,
                    proof.domain.len(),
                    proof.proof.universe().len(),
                    proof.invariant
                );
            }
        }
        proofs
    }

    /// WP-ARGS: collect scalar-or-tuple union proofs (btree `args`: `NIL` /
    /// `<<key>>` / `<<key, val>>`) from `Init`/`Next` writer coverage.
    ///
    /// Unlike the scalar-union sibling above this reads no invariant — btree's
    /// `TypeOk` does not constrain `args` at all, so the union universe can only
    /// come from the writers. Empty unless `TY_SCALAR_TUPLE_UNION` is on.
    fn configured_scalar_tuple_union_var_proofs(
        &self,
    ) -> Vec<crate::state::ScalarTupleUnionVarWriterProof> {
        let mut proofs = Vec::new();
        let (Some(init_name), Some(next_name)) = (&self.config.init, &self.config.next) else {
            return proofs;
        };
        let op_defs = &self.ctx.shared().ops;
        let op_replacements = self.ctx.op_replacements();
        let init_name = self.ctx.resolve_op_name(init_name).to_string();
        let next_name = self.ctx.resolve_op_name(next_name).to_string();
        let (Some(init_def), Some(next_def)) = (op_defs.get(&init_name), op_defs.get(&next_name))
        else {
            return proofs;
        };
        crate::state::collect_scalar_tuple_union_var_writer_proofs(
            &init_def.as_ref().body.node,
            &next_def.as_ref().body.node,
            self.ctx.var_registry(),
            self.ctx.precomputed_constants(),
            &self.named_homogeneous_proof_domains(),
            op_defs,
            op_replacements,
            &mut proofs,
        );
        proofs
    }

    /// Collect sum-type (distinct-shape) proofs for polymorphic top-level vars
    /// (`v \in {NIL} \cup {<<k>>:...} \cup {<<k,v>>:...}`). Gated on
    /// `TY_TAGGED_UNION`; empty when off, so the default surface is byte-identical.
    fn configured_tagged_union_var_type_proofs(
        &self,
    ) -> Vec<crate::state::TaggedUnionVarTypeProof> {
        let mut proofs = Vec::new();
        if std::env::var_os("TY_TAGGED_UNION").is_none() {
            return proofs;
        }
        let op_defs = &self.ctx.shared().ops;
        let op_replacements = self.ctx.op_replacements();
        for invariant in &self.config.invariants {
            let Some((resolved_name, def)) =
                proof_safe_zero_arg_op_def(invariant, op_defs, op_replacements)
            else {
                continue;
            };
            let lowered = lower_proof_operator_wrappers_preserving_precomputed_constants(
                &self.ctx,
                &def.body,
                op_defs,
                op_replacements,
                BTreeSet::from([resolved_name.to_owned()]),
                self.ctx.precomputed_constants(),
            );
            crate::state::collect_tagged_union_var_type_proofs_with_ops(
                &lowered.node,
                invariant,
                self.ctx.var_registry(),
                self.ctx.precomputed_constants(),
                op_defs,
                op_replacements,
                &mut proofs,
            );
        }
        // WRITER-ANALYSIS fallback (btree `args`): a polymorphic sum-type var with
        // NO TypeOK conjunct is invisible to the invariant scan above, so its
        // variant set is derived from the Init/Next writes instead
        // (`args = NIL`, `args' = <<key>>`, `args' = <<key, val>>`). Runs after the
        // invariant scan and skips any var it already covered. Fail-closed: an
        // unclassifiable write leaves the var un-promoted, and a promoted var whose
        // reachable universe the census under-approximated is caught by the
        // retry-without-flat backstop (never a wrong count).
        if let (Some(init_name), Some(next_name)) = (&self.config.init, &self.config.next) {
            let init_name = self.ctx.resolve_op_name(init_name).to_string();
            let next_name = self.ctx.resolve_op_name(next_name).to_string();
            if let (Some(init_def), Some(next_def)) =
                (op_defs.get(&init_name), op_defs.get(&next_name))
            {
                crate::state::collect_tagged_union_var_writer_proofs_with_ops(
                    &init_def.as_ref().body.node,
                    &next_def.as_ref().body.node,
                    "Init/Next writer proof (tagged union)",
                    self.ctx.var_registry(),
                    self.ctx.precomputed_constants(),
                    op_defs,
                    op_replacements,
                    &mut proofs,
                );
            }
        }
        proofs
    }

    fn configured_tagged_scalar_union_function_var_type_proofs(
        &self,
    ) -> Vec<crate::state::TaggedScalarUnionFunctionVarTypeProof> {
        let mut proofs = Vec::new();
        let op_defs = &self.ctx.shared().ops;
        let op_replacements = self.ctx.op_replacements();
        for invariant in &self.config.invariants {
            let Some((resolved_name, def)) =
                proof_safe_zero_arg_op_def(invariant, op_defs, op_replacements)
            else {
                continue;
            };
            let lowered = lower_proof_operator_wrappers_preserving_precomputed_constants(
                &self.ctx,
                &def.body,
                op_defs,
                op_replacements,
                BTreeSet::from([resolved_name.to_owned()]),
                self.ctx.precomputed_constants(),
            );
            crate::state::collect_tagged_scalar_union_function_var_type_proofs_with_ops(
                &lowered.node,
                invariant,
                self.ctx.var_registry(),
                self.ctx.precomputed_constants(),
                op_defs,
                op_replacements,
                &mut proofs,
            );
        }
        proofs
    }

    /// Collect range-union proofs for tuple/cross-product-keyed function vars
    /// (`f \in [D1 \X D2 -> union]`), the tuple complement of
    /// [`Self::configured_tagged_scalar_union_function_var_type_proofs`]. Gated
    /// on BOTH `TY_TUPLE_KEY_CARRIER` (the native tuple key carrier the promoted
    /// layout needs) and — implicitly, inside the collect — `TY_TAGGED_SCALAR_UNION`
    /// (the union universe construction). Empty when either flag is off, so the
    /// default surface is byte-identical.
    fn configured_tagged_scalar_union_tuple_function_var_type_proofs(
        &self,
    ) -> Vec<crate::state::TaggedScalarUnionTupleFunctionVarTypeProof> {
        let mut proofs = Vec::new();
        if std::env::var_os("TY_TUPLE_KEY_CARRIER").is_none() {
            return proofs;
        }
        let op_defs = &self.ctx.shared().ops;
        let op_replacements = self.ctx.op_replacements();
        for invariant in &self.config.invariants {
            let Some((resolved_name, def)) =
                proof_safe_zero_arg_op_def(invariant, op_defs, op_replacements)
            else {
                continue;
            };
            let lowered = lower_proof_operator_wrappers_preserving_precomputed_constants(
                &self.ctx,
                &def.body,
                op_defs,
                op_replacements,
                BTreeSet::from([resolved_name.to_owned()]),
                self.ctx.precomputed_constants(),
            );
            crate::state::collect_tagged_scalar_union_tuple_function_var_type_proofs_with_ops(
                &lowered.node,
                invariant,
                self.ctx.var_registry(),
                self.ctx.precomputed_constants(),
                op_defs,
                op_replacements,
                &mut proofs,
            );
        }
        proofs
    }

    /// Collect range-FixedScalar proofs for tuple/cross-product-keyed function
    /// vars whose range is a HOMOGENEOUS finite model-value/string set
    /// (`valOf \in [Nodes \X Keys -> Vals \cup {NIL}]`), the homogeneous
    /// complement of the tuple union override. Gated on `TY_TUPLE_KEY_CARRIER`;
    /// empty when off, so the default surface is byte-identical.
    fn configured_fixed_scalar_range_tuple_function_var_type_proofs(
        &self,
    ) -> Vec<crate::state::FixedScalarRangeTupleFunctionVarTypeProof> {
        let mut proofs = Vec::new();
        if std::env::var_os("TY_TUPLE_KEY_CARRIER").is_none() {
            return proofs;
        }
        let op_defs = &self.ctx.shared().ops;
        let op_replacements = self.ctx.op_replacements();
        for invariant in &self.config.invariants {
            let Some((resolved_name, def)) =
                proof_safe_zero_arg_op_def(invariant, op_defs, op_replacements)
            else {
                continue;
            };
            let lowered = lower_proof_operator_wrappers_preserving_precomputed_constants(
                &self.ctx,
                &def.body,
                op_defs,
                op_replacements,
                BTreeSet::from([resolved_name.to_owned()]),
                self.ctx.precomputed_constants(),
            );
            crate::state::collect_fixed_scalar_range_tuple_function_var_type_proofs_with_ops(
                &lowered.node,
                invariant,
                self.ctx.var_registry(),
                self.ctx.precomputed_constants(),
                op_defs,
                op_replacements,
                &mut proofs,
            );
        }
        proofs
    }

    fn configured_fixed_scalar_var_type_proofs(
        &self,
    ) -> Vec<crate::state::FixedScalarVarTypeProof> {
        let mut proofs = Vec::new();
        let proof_domains = self.named_homogeneous_proof_domains();
        let op_defs = &self.ctx.shared().ops;
        let op_replacements = self.ctx.op_replacements();
        for invariant in &self.config.invariants {
            let Some((resolved_name, def)) =
                proof_safe_zero_arg_op_def(invariant, op_defs, op_replacements)
            else {
                continue;
            };
            let lowered = lower_proof_operator_wrappers(
                &self.ctx,
                &def.body,
                op_defs,
                op_replacements,
                BTreeSet::from([resolved_name.to_owned()]),
            );
            crate::state::collect_fixed_scalar_var_type_proofs_with_ops(
                &lowered.node,
                invariant,
                self.ctx.var_registry(),
                self.ctx.precomputed_constants(),
                &proof_domains,
                op_defs,
                op_replacements,
                &mut proofs,
            );
        }
        // #43 fail-closed gate: a top-level scalar `FixedScalarVarTypeProof` is
        // derived purely from a `TypeOK` `\in` clause and carries no writer
        // verification. If any Init/Next writer can assign a SET (or other
        // non-scalar) to the variable, promoting it to a flat-primary scalar slot
        // would alias distinct states and undercount. Drop such proofs.
        //
        // G2: a spec can also constrain a top-level scalar var only in `Init`
        // (e.g. DijkstraMutex `k \in Proc`, no `TypeOK` invariant). After the #43
        // veto prunes the TypeOK-derived proofs, collect a primary-flat
        // `FixedScalar` proof directly from `Init`/`Next`, gated by a
        // writer-coverage proof that the model-value domain is closed under every
        // `Next` writer (so the universe is total, exactly the obligation a checked
        // `TypeOK` would discharge). The G2 proofs are self-corroborated by that
        // closure proof, so adding them after the veto is sound.
        if let (Some(init_name), Some(next_name)) = (&self.config.init, &self.config.next) {
            let init_name = self.ctx.resolve_op_name(init_name).to_string();
            let next_name = self.ctx.resolve_op_name(next_name).to_string();
            if let (Some(init_def), Some(next_def)) =
                (op_defs.get(&init_name), op_defs.get(&next_name))
            {
                crate::state::retain_writer_corroborated_fixed_scalar_var_proofs(
                    &mut proofs,
                    &init_def.as_ref().body.node,
                    &next_def.as_ref().body.node,
                    self.ctx.var_registry(),
                    self.ctx.precomputed_constants(),
                    &proof_domains,
                    op_defs,
                    op_replacements,
                );
                crate::state::collect_fixed_scalar_var_writer_proofs_with_ops(
                    &init_def.as_ref().body.node,
                    &next_def.as_ref().body.node,
                    "Init/Next writer proof",
                    self.ctx.var_registry(),
                    self.ctx.precomputed_constants(),
                    &proof_domains,
                    op_defs,
                    op_replacements,
                    &mut proofs,
                );
            }
        }
        proofs
    }

    /// #43 fail-closed veto set: state-var indices whose Init/Next writers can
    /// assign a SET (or any other non-scalar) value.
    ///
    /// The #43 fix consulted this set only to drop TypeOK-derived `FixedScalar`
    /// proofs. It is consulted here a second time, by
    /// [`crate::state::StateLayout::veto_flat_primary_scalar_slot_vars`], to also
    /// veto a scalar-slot flat-primary layout that was admitted by *init-sampling*
    /// alone (no type-proof at all): e.g. `x = 0` in Init makes `x` a plain
    /// `Scalar`, yet a successor `x' = {1, 2}` would alias the set into the same
    /// i64 slot and silently undercount the BFS (missed violation). Computed once
    /// and applied to both flat-state inference entry points.
    fn nonscalar_writer_vetoed_vars(&self) -> std::collections::BTreeSet<usize> {
        let (Some(init_name), Some(next_name)) = (&self.config.init, &self.config.next) else {
            return std::collections::BTreeSet::new();
        };
        let op_defs = &self.ctx.shared().ops;
        let op_replacements = self.ctx.op_replacements();
        let proof_domains = self.named_homogeneous_proof_domains();
        let init_name = self.ctx.resolve_op_name(init_name).to_string();
        let next_name = self.ctx.resolve_op_name(next_name).to_string();
        let (Some(init_def), Some(next_def)) = (op_defs.get(&init_name), op_defs.get(&next_name))
        else {
            return std::collections::BTreeSet::new();
        };
        crate::state::nonscalar_writer_vetoed_vars(
            &init_def.as_ref().body.node,
            &next_def.as_ref().body.node,
            self.ctx.var_registry(),
            self.ctx.precomputed_constants(),
            &proof_domains,
            op_defs,
            op_replacements,
        )
    }

    fn configured_set_bitmask_range_type_proofs(
        &self,
    ) -> Vec<crate::state::SetBitmaskRangeTypeProof> {
        let mut proofs = Vec::new();
        let proof_domains = self.named_homogeneous_proof_domains();
        let op_defs = &self.ctx.shared().ops;
        let op_replacements = self.ctx.op_replacements();
        for invariant in &self.config.invariants {
            let Some((resolved_name, def)) =
                proof_safe_zero_arg_op_def(invariant, op_defs, op_replacements)
            else {
                continue;
            };
            let lowered = lower_proof_operator_wrappers_preserving_precomputed_constants(
                &self.ctx,
                &def.body,
                op_defs,
                op_replacements,
                BTreeSet::from([resolved_name.to_owned()]),
                self.ctx.precomputed_constants(),
            );
            crate::state::collect_set_bitmask_range_type_proofs_with_ops(
                &lowered.node,
                invariant,
                self.ctx.var_registry(),
                self.ctx.precomputed_constants(),
                &proof_domains,
                op_defs,
                op_replacements,
                &mut proofs,
            );
        }
        proofs
    }

    fn configured_set_bitmask_type_proofs(&self) -> Vec<crate::state::SetBitmaskTypeProof> {
        let mut proofs = Vec::new();
        let proof_domains = self.named_homogeneous_proof_domains();
        let op_defs = &self.ctx.shared().ops;
        let op_replacements = self.ctx.op_replacements();
        for invariant in &self.config.invariants {
            let Some((resolved_name, def)) =
                proof_safe_zero_arg_op_def(invariant, op_defs, op_replacements)
            else {
                continue;
            };
            let lowered = lower_proof_operator_wrappers(
                &self.ctx,
                &def.body,
                op_defs,
                op_replacements,
                BTreeSet::from([resolved_name.to_owned()]),
            );
            crate::state::collect_set_bitmask_type_proofs_with_ops(
                &lowered.node,
                invariant,
                self.ctx.var_registry(),
                self.ctx.precomputed_constants(),
                &proof_domains,
                op_defs,
                op_replacements,
                &mut proofs,
            );
        }
        proofs
    }

    /// Collect record-set bitmask type proofs from the configured invariants.
    ///
    /// Targets a top-level state variable constrained to a finite, statically
    /// enumerable *record* set: `v \in SUBSET RecSet` or `v \subseteq RecSet`,
    /// where `RecSet` (e.g. PaxosCommit's `Message`, Allocator's `Messages`) is a
    /// record-set / union-of-record-sets type expression.
    ///
    /// Soundness: the record universe is produced by the *real evaluator*
    /// (`eval_entry` + `eval_iter_set`), so cross-product, union, and nested-set
    /// record schemas enumerate exactly as the model checker would — there is no
    /// hand-rolled enumeration that could under-approximate. The proof is only
    /// emitted when every enumerated element is a `Value::Record`, the universe is
    /// non-empty and fits the bitmask width (`<= 63`), and the universe is
    /// sorted + deduped into canonical order. Any failure (non-record element,
    /// eval error, oversized universe, non-set value) fails closed and the
    /// variable stays `Dynamic`. The bounding invariant proves the universe is
    /// closed under every successor write, so the resulting bitmask universe is
    /// `ProvenClosed`.
    fn configured_record_set_bitmask_type_proofs(
        &self,
    ) -> Vec<crate::state::RecordSetBitmaskTypeProof> {
        let mut proofs = Vec::new();
        let op_defs = &self.ctx.shared().ops;
        let op_replacements = self.ctx.op_replacements();
        for invariant in &self.config.invariants {
            let Some((resolved_name, def)) =
                proof_safe_zero_arg_op_def(invariant, op_defs, op_replacements)
            else {
                continue;
            };
            let lowered = lower_proof_operator_wrappers(
                &self.ctx,
                &def.body,
                op_defs,
                op_replacements,
                BTreeSet::from([resolved_name.to_owned()]),
            );
            self.collect_record_set_bitmask_proofs_from_expr(&lowered.node, invariant, &mut proofs);
        }
        proofs
    }

    /// Recursive AST walk for [`Self::configured_record_set_bitmask_type_proofs`].
    /// Descends top-level conjunctions and looks for a bare state-variable
    /// membership in a record powerset / subset of a record set.
    fn collect_record_set_bitmask_proofs_from_expr(
        &self,
        expr: &Expr,
        invariant: &str,
        out: &mut Vec<crate::state::RecordSetBitmaskTypeProof>,
    ) {
        match expr {
            Expr::And(left, right) => {
                self.collect_record_set_bitmask_proofs_from_expr(&left.node, invariant, out);
                self.collect_record_set_bitmask_proofs_from_expr(&right.node, invariant, out);
            }
            // `v \in SUBSET RecSet` — the RHS is a powerset whose base is the
            // record universe.
            Expr::In(lhs, rhs) => {
                if let Expr::Powerset(base) = &rhs.node {
                    self.try_push_record_set_bitmask_proof(
                        &lhs.node, &base.node, rhs.span, invariant, out,
                    );
                }
            }
            // `v \subseteq RecSet` — the RHS itself is the record universe.
            Expr::Subseteq(lhs, rhs) => {
                self.try_push_record_set_bitmask_proof(
                    &lhs.node, &rhs.node, rhs.span, invariant, out,
                );
            }
            _ => {}
        }
    }

    /// Resolve `lhs` to a bare state var, evaluate `rec_set_expr` to a finite
    /// record set, and (if every soundness gate passes) push a proof.
    fn try_push_record_set_bitmask_proof(
        &self,
        lhs: &Expr,
        rec_set_expr: &Expr,
        span: tla_core::span::Span,
        invariant: &str,
        out: &mut Vec<crate::state::RecordSetBitmaskTypeProof>,
    ) {
        // `lhs` must be a bare state variable (empty access path). Function /
        // record sub-paths are handled by other collectors, not here. The proof
        // lowering rewrites state-var references to `Expr::StateVar`; an
        // un-lowered bare `Ident` that resolves to a state var is also accepted.
        let var_idx = match lhs {
            Expr::StateVar(_, idx, _) => *idx as usize,
            Expr::Ident(name, _) => match self.ctx.var_registry().get(name) {
                Some(idx) => idx.as_usize(),
                None => return,
            },
            _ => return,
        };

        // Evaluate the record-set type expression with the real evaluator. The
        // expression references only constants, so this is a constant
        // enumeration. Fail closed on any eval error.
        let spanned = Spanned::new(rec_set_expr.clone(), span);
        let Ok(value) = crate::eval::eval_entry(&self.ctx, &spanned) else {
            return;
        };
        if !value.is_set() || !value.is_finite_set() {
            return;
        }
        let Ok(iter) = crate::eval::eval_iter_set(&self.ctx, &value, Some(span)) else {
            return;
        };

        let mut universe: Vec<crate::Value> = Vec::new();
        for elem in iter {
            // Every element must be a record. A non-record element means this is
            // not a record-set type, so fail closed.
            if !matches!(elem, crate::Value::Record(_)) {
                return;
            }
            universe.push(elem);
            // Multi-slot bitmask: `ceil(|universe| / 64)` i64 slots, capped at
            // `MAX_RECORD_SET_BITMASK_UNIVERSE` records. Bail early on an
            // oversized universe (PaxosCommit's 144-record `Message` is 3 slots,
            // well under the cap).
            if universe.len() > crate::state::MAX_RECORD_SET_BITMASK_UNIVERSE {
                return;
            }
        }
        universe.sort();
        universe.dedup();
        if universe.is_empty() || universe.len() > crate::state::MAX_RECORD_SET_BITMASK_UNIVERSE {
            return;
        }

        out.push(crate::state::RecordSetBitmaskTypeProof {
            var_idx,
            path: Vec::new(),
            record_universe: universe,
            invariant: Arc::from(invariant),
        });
    }

    fn action_producer_tagged_scalar_set_range_type_proofs(
        &self,
        seed_state: &crate::state::ArrayState,
    ) -> Vec<crate::state::TaggedScalarSetRangeTypeProof> {
        let Some(bytecode) = self.action_bytecode.as_ref() else {
            return Vec::new();
        };
        let seed_scalar_types = action_seed_scalar_types(seed_state);
        let typed_load_imms =
            action_typed_load_imms_by_split_action(self.compiled.split_action_meta.as_deref());
        action_tagged_range_candidates_from_seed_state(seed_state)
            .into_iter()
            .filter(|candidate| {
                action_bytecode_supports_tagged_range_candidate(
                    bytecode,
                    candidate,
                    &seed_scalar_types,
                    &typed_load_imms,
                )
            })
            .map(|candidate| crate::state::TaggedScalarSetRangeTypeProof {
                var_idx: candidate.var_idx,
                path: Vec::new(),
                domain: candidate.domain,
                scalar_type: candidate.scalar_type,
                set_universe: candidate.set_universe,
                invariant: Arc::from(format!("action-producer:var{}", candidate.var_idx)),
            })
            .collect()
    }

    fn named_homogeneous_proof_domains(&self) -> BTreeMap<String, Arc<[crate::Value]>> {
        let ops = &self.ctx.shared().ops;
        let op_replacements = self.ctx.op_replacements();
        let mut domains: BTreeMap<String, Arc<[crate::Value]>> = ops
            .iter()
            .filter_map(|(name, def)| {
                if op_replacements.contains_key(name.as_str()) {
                    return None;
                }
                let def = def.as_ref();
                if !(def.params.is_empty()
                    && !def.contains_prime
                    && !def.has_primed_param
                    && !def.is_recursive)
                {
                    return None;
                }
                let values = expression_proof_domain_values(
                    &def.body.node,
                    ops,
                    self.ctx.precomputed_constants(),
                    op_replacements,
                    &mut BTreeSet::new(),
                )?;
                Some((name.clone(), Arc::from(values.into_boxed_slice())))
            })
            .collect();

        for from in op_replacements.keys() {
            if let Some(resolved) = resolve_proof_op_name(from, op_replacements) {
                if let Some(values) = domains.get(resolved).cloned() {
                    domains.insert(from.clone(), values);
                }
            }
        }
        domains
    }

    /// Shared setup for BFS model checking: constant binding, symmetry, VIEW validation,
    /// config validation, invariant compilation, operator expansion, and action compilation.
    /// Returns the resolved `Next` operator name on success.
    ///
    /// Both `check_impl` and `check_with_resume` call this to avoid duplicating setup logic.
    /// Part of #1230: extracted from check_impl/check_with_resume to eliminate copy-paste.
    #[allow(clippy::result_large_err)]
    pub(super) fn prepare_bfs_common(&mut self) -> Result<String, CheckResult> {
        // Install ENABLED hook (adaptive/parallel checkers install in their entry points).
        crate::eval::set_enabled_hook(crate::enabled::eval_enabled_cp);

        // Bind constants from config before checking
        // Part of #2356/#2777: Route through check_error_to_result so
        // ExitRequested maps to LimitReached(Exit).
        if let Err(e) = bind_constants_from_config(&mut self.ctx, self.config) {
            return Err(check_error_to_result(
                EvalCheckError::Eval(e).into(),
                &self.stats,
            ));
        }

        // Pre-evaluate zero-arity constant-level operators (Part of #2364).
        // Mirrors TLC's SpecProcessor.processConstantDefns(): evaluate all zero-arg
        // operators that don't reference state variables ONCE, store the result in
        // SharedCtx for O(1) lookup during model checking. This eliminates per-reference
        // overhead (dep tracking, cache key hashing, context stripping) for constant ops
        // like RingOfNodes, Initiator, N in EWD998ChanID.
        super::precompute::precompute_constant_operators(&mut self.ctx);

        // Part of #2895: Promote env constants (CONSTANT declarations from model config)
        // to precomputed_constants for NameId-keyed O(1) lookup in eval_ident.
        // Constants in env are looked up via string-key HashMap::get; moving them to
        // precomputed_constants (NameId key) eliminates string hashing on the fast path.
        // Only promotes non-state-variable entries (state vars stay in state_env).
        super::precompute::promote_env_constants_to_precomputed(&mut self.ctx);
        // Part of #3961: Build ident resolution hints for eval_ident fast-path dispatch.
        super::precompute::build_ident_hints(&mut self.ctx);
        // Projection certificates must observe the final config environment:
        // in particular, bare operator replacements and closure-valued config
        // constants passed to higher-order builtins must fail closed.
        self.invariant_verdict_cache
            .rebuild(&self.ctx, &self.config.invariants);
        self.state_constraint_verdict_cache
            .rebuild(&self.ctx, &self.config.constraints);
        // Audit finding #12: resolve the per-state successor cap from this
        // checker's Config (honoring TY_PER_STATE_SUCCESSOR_CAP) onto the shared
        // context — per-checker, never process-global.
        crate::enumerate::apply_per_state_successor_cap(&mut self.ctx, self.config);

        // Part of #4251 Stream 5: populate the TIR partial-evaluation
        // ConstantEnv from the authoritative precomputed_constants map now
        // that all CONSTANT bindings (from .cfg and --constant overrides)
        // have been resolved. The env is attached to every TirProgram built
        // during this run; partial-eval substitution runs at TIR preprocess
        // time only when `TY_PARTIAL_EVAL=1` / `--partial-eval` is set.
        if let Some(tir_parity) = self.tir_parity.as_mut() {
            let mut env = tla_tir::analysis::ConstantEnv::new();
            for (name_id, value) in self.ctx.precomputed_constants().iter() {
                env.bind(*name_id, value.clone());
            }
            tir_parity.set_partial_eval_env(env);
        }

        // Compute symmetry permutations now that constants are bound.
        // Two paths: explicit SYMMETRY config, or auto-detection from model value sets.
        //
        // SOUNDNESS GATE — decided ONCE, before any permutations are installed,
        // and shared by the declared and auto paths below: symmetry reduction
        // is UNSOUND for genuine temporal (liveness) properties. TY does not
        // implement the Emerson-Sistla annotated quotient, and the plain orbit
        // quotient can hide or synthesize fair cycles. Verified on this corpus:
        // AllocatorImplementation under declared SYMMETRY reports a FALSE
        // violation (threaded quotient cycle) where unreduced TY and TLC prove
        // HOLD. See the "Guard (c)" section in symmetry_detect.rs for the
        // formal soundness conditions (atom-invariance bisimulation /
        // Emerson-Sistla annotated quotient) and the measured no-go math for
        // relaxing this. Pure safety properties (`[]P`, Part of #2227) are
        // handled by the safety-temporal fast path and keep symmetry.
        let declared_symmetry_pending =
            self.symmetry.perms.is_empty() && self.config.symmetry.is_some();
        let auto_symmetry_pending =
            self.symmetry.perms.is_empty() && self.config.symmetry.is_none();
        // Auto-detection is ON by default in the production configuration
        // (like auto-POR). Gated by the per-checker override when set
        // (tests/embedders), otherwise by the TY_AUTO_SYMMETRY environment
        // variable (kill switch: =0).
        // Engagement gates (all structural). Beyond detection-level guards
        // (see symmetry_detect.rs), the auto path hard-disables when:
        //   - POR is explicitly requested: the POR+symmetry combination is
        //     not validated, and the user's explicit choice wins. (When
        //     auto-symmetry engages, auto-POR is released in run.rs —
        //     releasing auto-detected POR is always sound.)
        //   - a VIEW is configured: symmetry canonicalization composing
        //     with VIEW fingerprinting is not validated; fail closed.
        //   - trace invariants or a POSTCONDITION are configured: both can
        //     observe orbit-reduced traces/statistics, which would change
        //     their meaning under reduction; fail closed.
        //   - guard (c): any PROPERTY requires the genuine liveness
        //     checker (`has_genuine_temporal` below; shared with the
        //     declared path since the fix for the declared-SYMMETRY
        //     wrong-verdict path).
        let auto_eligible = auto_symmetry_pending
            && self
                .symmetry
                .auto_symmetry_override
                .unwrap_or_else(super::symmetry_detect::auto_symmetry_enabled)
            && !self.config.por_enabled
            && self.config.view.is_none()
            && self.config.trace_invariants.is_empty()
            && self.config.postcondition.is_none()
            && !super::symmetry_detect::detect_symmetric_model_value_sets(self.config).is_empty();
        // Classify properties only when symmetry could actually engage, so
        // symmetry-free runs skip the classification pass here entirely.
        let has_genuine_temporal = (declared_symmetry_pending || auto_eligible)
            && !self.config.properties.is_empty()
            && crate::checker_ops::any_property_requires_liveness_checker(
                &self.ctx,
                &self.module.op_defs,
                &self.config.properties,
            );

        // Benchmark-parity opt-in: TY_MATCH_DECLARED_SYMMETRY=1 makes ty apply
        // the declared SYMMETRY even under liveness checking, computing exactly
        // what TLC computes (same orbit-reduced space + orbit-quotient
        // liveness) so single-thread speed is compared apples-to-apples on
        // declared-SYMMETRY liveness specs. Default OFF keeps the sound refusal.
        let match_declared_symmetry_liveness =
            super::symmetry_detect::match_declared_symmetry_for_liveness();
        if declared_symmetry_pending && has_genuine_temporal && !match_declared_symmetry_liveness {
            // Part of #1963/#2227, declared-SYMMETRY wrong-verdict fix: this
            // path used to warn ("dangerous") and CONTINUE with the orbit
            // quotient, which can return a wrong verdict. Mirror the auto
            // path instead: leave `self.symmetry` empty (no perms, no
            // mvperms, no group names) so every downstream consumer — BFS
            // fingerprint canonicalization, LivenessMode, the liveness
            // graph, storage selection — sees no symmetry. This also makes
            // the former full-state storage auto-upgrade for symmetry
            // witness reconstruction (Part of #2200/#3222) unnecessary:
            // permutations are never installed alongside genuine temporal
            // properties.
            eprintln!(
                "Warning: declared SYMMETRY is ignored during liveness checking (the orbit \
                 quotient is unsound for temporal properties; TLC continues with symmetry \
                 and can report wrong verdicts). Checking without symmetry — expect a \
                 larger state space. Set TY_MATCH_DECLARED_SYMMETRY=1 to apply it anyway \
                 (matches TLC for timing parity; verdict then inherits TLC's orbit-quotient \
                 unsoundness)."
            );
        } else if declared_symmetry_pending {
            if has_genuine_temporal {
                eprintln!(
                    "Warning: TY_MATCH_DECLARED_SYMMETRY=1 — applying declared SYMMETRY under \
                     liveness checking to match TLC exactly (orbit quotient may be unsound for \
                     temporal properties; use this run for TIMING PARITY, not to trust the \
                     liveness verdict)."
                );
            }
            self.symmetry.perms =
                super::symmetry_perms::compute_symmetry_perms(&self.ctx, self.config)
                    .map_err(|e| check_error_to_result(e, &self.stats))?;
            // Extract group names from config for statistics.
            self.symmetry.group_names =
                super::symmetry_detect::detect_symmetric_model_value_sets(self.config)
                    .into_iter()
                    .map(|(name, _)| name)
                    .collect();
            self.symmetry.auto_detected = false;
            self.symmetry.mvperms = self // #358: MVPerm for O(1) model value lookup
                .symmetry
                .perms
                .iter()
                .map(crate::value::MVPerm::from_func_value)
                .collect::<Result<Vec<_>, _>>()
                // Part of #2356/#2777: Route through check_error_to_result so
                // ExitRequested maps to LimitReached(Exit).
                .map_err(|e| check_error_to_result(EvalCheckError::Eval(e).into(), &self.stats))?;
            self.refresh_liveness_mode();
        } else if auto_symmetry_pending {
            // Guard (c): genuine temporal properties hard-disable auto-symmetry
            // (decided above, before any permutations exist).
            let (auto_perms, group_names) = if auto_eligible && !has_genuine_temporal {
                super::symmetry_detect::auto_detect_symmetry_perms(
                    &self.ctx,
                    self.config,
                    &self.module.op_defs,
                )
            } else {
                (Vec::new(), Vec::new())
            };
            if !auto_perms.is_empty() {
                eprintln!(
                    "Symmetry: auto-detected {} permutation(s) from model value set(s) {:?} — \
                     distinct-state counts are orbit-reduced, verdicts unchanged \
                     (disable with TY_AUTO_SYMMETRY=0)",
                    auto_perms.len(),
                    group_names,
                );
                self.symmetry.perms = auto_perms;
                self.symmetry.group_names = group_names;
                self.symmetry.auto_detected = true;
                self.symmetry.mvperms = self
                    .symmetry
                    .perms
                    .iter()
                    .map(crate::value::MVPerm::from_func_value)
                    .collect::<Result<Vec<_>, _>>()
                    .map_err(|e| {
                        check_error_to_result(EvalCheckError::Eval(e).into(), &self.stats)
                    })?;
                self.refresh_liveness_mode();
            }
        }

        // The gate above guarantees permutations are never installed alongside
        // genuine temporal properties, so the former SYMMETRY+liveness
        // "dangerous" warn-and-continue (Part of #1963) and the SYMMETRY+
        // liveness full-state storage auto-upgrade (Part of #2200/#3222:
        // inline liveness recording is disabled under symmetry) are
        // unreachable and have been removed.

        // Validate and cache VIEW operator name now that constants are bound
        if self.compiled.cached_view_name.is_none() && self.config.view.is_some() {
            self.validate_view();
        }

        // Validate next_name
        let raw_next_name = match &self.config.next {
            Some(name) => name.clone(),
            None => {
                return Err(CheckResult::from_error(
                    ConfigCheckError::MissingNext.into(),
                    self.stats.clone(),
                ));
            }
        };

        // Cache the raw config alias for trace reconstruction and user-facing labels,
        // but resolve replacements for the actual operator body we execute/compile.
        self.trace.cached_next_name = Some(raw_next_name.clone());
        let resolved = self.ctx.resolve_op_name(&raw_next_name).to_string();
        self.trace.cached_resolved_next_name = Some(resolved);

        // Part of #254: Set initial TLC level for TLCGet("level") - TLC uses 1-based indexing.
        // Set level=1 before any expression evaluation (including constraint extraction)
        // to avoid side effects (like PrintT) seeing level=0. Later phases update this
        // to the correct depth during BFS exploration.
        self.ctx.set_tlc_level(1);

        // Validate config operators: existence, arity, level, and variables.
        // Part of #2573: TLC validates all config elements at setup time
        // (SpecProcessor.java:processConfigInvariants/Properties/Constraints).
        self.validate_config_ops()?;

        let next_name = self.ctx.resolve_op_name(&raw_next_name).to_string();

        // Classify PROPERTY entries into BFS-phase checking buckets (#2332, #2670, #2740).
        let classification = crate::checker_ops::classify_property_safety_parts(
            &self.ctx,
            &self.config.properties,
            &self.module.op_defs,
        );
        self.compiled.promoted_property_invariants = classification.promoted_invariant_properties;
        self.compiled.state_property_violation_names = classification.state_violation_properties;
        self.compiled.eval_implied_actions = classification.eval_implied_actions;
        // Bytecode-VM fast path for eval implied-action terms: compile the
        // term structure once, leaving unsupported leaf operators as
        // interpreter callbacks. Kill switch: TY_NO_IMPLIED_ACTION_BYTECODE=1.
        if !self.compiled.eval_implied_actions.is_empty() {
            let modules = self.tir_parity.as_ref().map(|tp| tp.clone_modules());
            if let Some((root, deps)) = modules {
                crate::checker_ops::attach_eval_implied_action_bytecode(
                    &self.ctx,
                    &root,
                    &deps,
                    &mut self.compiled.eval_implied_actions,
                );
            }
        }
        self.compiled.native_implied_actions = classification.native_implied_actions;
        self.compiled.eval_state_invariants = classification.eval_state_invariants;
        self.compiled.promoted_implied_action_properties =
            classification.promoted_action_properties;
        self.compiled.property_init_predicates = classification.init_predicates;
        // Fully promoted PROPERTY entries are handled by init/state/action BFS checks.
        // Disable the residual post-BFS liveness cache/mode when none remain.
        self.refresh_liveness_cache_requirement();

        // Part of #1121: Shared alias-aware trace detection (invariants + constraints + action_constraints).
        self.compiled.uses_trace = compute_uses_trace(self.config, &self.module.op_defs)
            .map_err(|e| CheckResult::from_error(e, self.stats.clone()))?;

        // Pre-expand operator references in the Next action body (Part of #207).
        // Delegates to checker_ops::expand_operator_body (Part of #810).
        if let Some(next_def) = self.module.op_defs.get(&next_name).cloned() {
            let expanded_def = crate::checker_ops::expand_operator_body(&self.ctx, &next_def);
            self.module.op_defs.insert(next_name.clone(), expanded_def);
        }

        // Part of #3100: Discover split-action metadata for liveness provenance.
        // Successor generation no longer uses compiled split actions, but inline
        // liveness still needs the split action names/bindings to match action
        // predicates against BFS actions.
        let res = {
            static ACTION_SPLIT_ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
            *ACTION_SPLIT_ENABLED.get_or_init(|| std::env::var("TY_NO_ACTION_SPLIT").is_err())
        };
        if res {
            if let Some(next_def) = self.module.op_defs.get(&next_name) {
                match crate::action_instance::split_action_instances(&self.ctx, &next_def.body) {
                    Ok(instances) if !instances.is_empty() => {
                        #[cfg(debug_assertions)]
                        if ty_debug() {
                            eprintln!(
                                "[#3100] Action split: {} instances recorded for liveness provenance",
                                instances.len()
                            );
                        }
                        // Stable metadata name for every split instance. Most
                        // instances carry a resolved `inst.name` (a named action
                        // such as `Receive` or a scoped `M!Op`). The remaining
                        // gap is the structural disjunct that resolves to no
                        // identifier — most commonly a bare `UNCHANGED vars`
                        // (full-stutter) disjunct inside `Next`, e.g.
                        // cf1s_folklore's `\E self : (... \/ UNCHANGED vars)`.
                        // Such instances previously got `name: None` whenever
                        // there was more than one instance, which then poisoned
                        // executable-action coverage accounting ("action
                        // instance N has no metadata name") and disqualified the
                        // whole run from the per-action / compiled-BFS native
                        // paths even for instances that DID lower. Give every
                        // unnamed instance a deterministic synthetic name
                        // derived from the resolved Next operator + instance
                        // index so the metadata is always complete and stable
                        // across runs. This is purely a labeling fix: the name
                        // feeds liveness provenance matching and coverage labels;
                        // the native action KEY for a synthetic-named instance
                        // still resolves through
                        // `trust_cg_action_executable_base_key_for_cache`, which
                        // only counts the instance as native when a matching
                        // compiled key exists, so soundness of the native gate is
                        // unchanged.
                        let meta: Vec<_> = instances
                            .iter()
                            .enumerate()
                            .map(|(idx, inst)| super::mc_struct::ActionInstanceMeta {
                                name: inst.name.clone().or_else(|| {
                                    if instances.len() == 1 {
                                        Some(next_name.clone())
                                    } else {
                                        Some(format!("{next_name}__instance_{idx}"))
                                    }
                                }),
                                bindings: inst.bindings.clone(),
                                formal_bindings: inst.formal_bindings.clone(),
                                expr: Some(inst.expr.clone()),
                            })
                            .collect();
                        self.compiled.split_action_meta = Some(meta);
                        // Explicit VM requests retain their historical ability
                        // to force action bytecode.  AUTO may only borrow the
                        // bindings when the structural native pre-check admits
                        // the trust-cg compile that was already going to happen.
                        let retain_complete_bindings = self.value_action_vm.requested()
                            || (self.value_action_vm.auto_candidate()
                                && self.trust_cg_native_hostile_action_structure().is_none());
                        self.compiled.split_action_complete_bindings = retain_complete_bindings
                            .then(|| {
                                instances
                                    .iter()
                                    .map(|inst| inst.complete_bindings.clone())
                                    .collect::<Vec<_>>()
                            });
                    }
                    Ok(_) =>
                    {
                        #[cfg(debug_assertions)]
                        if ty_debug() {
                            eprintln!("[#1150] Action split produced no instances");
                        }
                    }
                    Err(_e) =>
                    {
                        #[cfg(debug_assertions)]
                        if ty_debug() {
                            eprintln!("[#1150] Action split failed, using monolithic: {_e:?}");
                        }
                    }
                }
            }
        }

        // TLC_LIVE_FORMULA_TAUTOLOGY pre-check (EC 2253, #2215). Skipped
        // when TY_SKIP_LIVENESS=1, since the entire liveness pipeline
        // is bypassed in that benchmark mode.
        if !crate::check::debug::skip_liveness() {
            if let Some(result) = self.check_properties_for_tautologies() {
                return Err(result);
            }
        }

        // Certify context-free state CONSTRAINTs with the same fail-closed
        // dependency analysis used for ACTION_CONSTRAINTs. An exact duplicate
        // payload can reuse a prior successful state-constraint verdict only
        // when every configured predicate is pure and unprimed.
        self.compiled.state_constraints_reusable_on_exact_duplicate = if self
            .config
            .constraints
            .is_empty()
        {
            false
        } else {
            crate::checker_ops::ActionConstraintAnalysis::build(&self.ctx, &self.config.constraints)
                .supports_exact_seen_state_reuse()
        };

        // Pre-analyze ACTION_CONSTRAINTs for variable dependencies.
        // This enables skipping constraint evaluation when referenced variables
        // are unchanged between current and successor states.
        if !self.config.action_constraints.is_empty() {
            self.compiled.action_constraint_analysis =
                Some(crate::checker_ops::ActionConstraintAnalysis::build(
                    &self.ctx,
                    &self.config.action_constraints,
                ));
        }

        // Detect PlusCal pc-dispatch pattern for action guard hoisting.
        // When all disjuncts of Next are guarded by `pc = "label"`, the BFS
        // loop can skip evaluating actions whose pc guard doesn't match the
        // current state, avoiding wasted work in PlusCal-generated specs.
        if let Some(next_def) = self.module.op_defs.get(&next_name) {
            let registry = self.ctx.var_registry().clone();
            if let Some(table) = crate::checker_ops::pc_dispatch::detect_pc_dispatch(
                next_def,
                &self.module.vars,
                &registry,
                &self.ctx,
            ) {
                #[cfg(debug_assertions)]
                if ty_debug() {
                    eprintln!(
                        "[pc-dispatch] Detected PlusCal pattern: {} actions, {} distinct pc values",
                        table.total_actions,
                        table.dispatch.len(),
                    );
                }
                self.compiled.pc_var_idx = Some(table.pc_var_idx);
                self.compiled.pc_dispatch = Some(table);
            } else {
                // Part of #3805: Multi-process PlusCal guard hoisting.
                // When the full dispatch table can't be built (multi-process specs
                // use `pc[self] = "label"` instead of `pc = "label"`), we can still
                // enable guard hoisting when detection proves `self` is already in
                // scope at the guarded Or. The enumerator handles multi-process pc
                // values (Value::Func) by looking up that binding.
                if let Some(pc_var_idx) = registry.get("pc") {
                    let strict_detected =
                        crate::checker_ops::pc_dispatch::spec_has_pc_guards(next_def, &self.ctx);
                    // Temporary same-binary A/B switch for measuring the cost of
                    // the former loose detector. The legacy path is deliberately
                    // opt-in because it cannot prove the runtime `self` lookup.
                    let force_legacy =
                        crate::checker_ops::pc_dispatch::force_legacy_pc_guard_hoist();
                    let legacy_detected = force_legacy
                        && crate::checker_ops::pc_dispatch::spec_has_pc_guards_legacy(
                            next_def, &self.ctx,
                        );
                    if strict_detected || legacy_detected {
                        #[cfg(debug_assertions)]
                        if ty_debug() {
                            eprintln!(
                                "[pc-dispatch] Detected multi-process PlusCal pc guards (pc_var_idx={:?}, forced_legacy={legacy_detected})",
                                pc_var_idx,
                            );
                        }
                        self.compiled.pc_var_idx = Some(pc_var_idx);
                    }
                }
            }
        }

        // Part of #3578: Compile invariant operators to bytecode for VM fast path.
        // NOTE: Profiling shows the bytecode VM is currently ~2.6x slower than
        // tree-walking with TIR cache for invariant evaluation (EWD998Small:
        // 27.1s bytecode vs 10.3s tree-walk). The per-state VM setup overhead
        // (BytecodeVm::from_state_env) exceeds the benefit of avoiding AST
        // traversal. Skip bytecode compilation when TIR eval owns invariants.
        //
        // Previously JIT forced bytecode compilation here, but the compiled
        // bytecode activates the slow bytecode VM path for ALL invariant evals.
        // Since JIT invariant native code currently has 0% hit rate (all
        // FallbackNeeded), this caused a 33% regression. JIT bytecode
        // compilation is now deferred to the JIT compilation phase.
        let tir_blocks_bytecode_vm = self
            .tir_parity
            .as_ref()
            .is_some_and(super::tir_parity::TirParityState::is_eval_mode);
        if tla_eval::tir::bytecode_vm_enabled() && !tir_blocks_bytecode_vm {
            self.compile_invariant_bytecode();
        }

        // Lever 1 (#EWD998PCal): compile CONSTRAINT operators to bytecode
        // regardless of TIR mode (the TIR constraint path rebuilds a
        // `TirProgram` per check — exactly the waste this fast path
        // eliminates). Kill switch: TY_NO_CONSTRAINT_BYTECODE=1.
        if tla_eval::tir::bytecode_vm_enabled()
            && !self.config.constraints.is_empty()
            && !super::invariants::no_constraint_bytecode()
        {
            self.compile_constraint_bytecode();
        }

        // AUTO engine-selection: pre-compile structural veto.
        //
        // When the production default (`ty check` with no `--backend`) is in
        // effect, the trust-cg native path is the default engine — but the
        // per-action native bytecode compile below plus the native-fused setup
        // are pure overhead for specs whose actions cannot lower to native
        // code. The dominant such pattern is a quantifier/CHOOSE that
        // enumerates a FUNCTION-SET (`[D -> R]`) or POWERSET (`SUBSET S`)
        // domain: these are not enumerable proof-backed compact-set domains, so
        // native lowering fails closed AND the bytecode compile of the split
        // actions can blow up super-linearly. Detecting this cheaply on the
        // already-resolved split-action ASTs (no bytecode required) lets us skip
        // the entire trust-cg setup and run the plain interpreter at full speed,
        // guaranteeing the default never regresses below the interpreter on
        // these specs. Structural only (AST shape) — never spec-name based.
        //
        // Applies ONLY in AUTO mode; an explicit `--backend trust-cg` keeps the
        // forced-native behavior the supremacy harnesses depend on.
        if crate::check::debug::trust_cg_auto_select_enabled()
            && super::trust_cg_dispatch::should_use_trust_cg(self.trust_cg_structurally_vetoed())
        {
            if let Some(reason) = self.trust_cg_native_hostile_action_structure() {
                self.value_action_vm.discard_auto_candidate();
                if !self.value_action_vm.requested() {
                    self.compiled.split_action_complete_bindings = None;
                }
                self.set_trust_cg_structural_veto();
                eprintln!("engine: interpreter (native not beneficial: {reason})");
            }
        }

        // Part of #3910: Compile action operators to bytecode for JIT next-state dispatch.
        // This is separate from invariant bytecode because actions use StoreVar opcodes
        // for primed variables, and the JitNextStateCache requires action-specific bytecode.
        // Also needed when trust-codegen is enabled (#4190), since the trust-codegen pipeline takes
        // BytecodeFunction as input.
        {
            let need_action_bytecode = crate::check::debug::jit_enabled();
            let need_action_bytecode = need_action_bytecode
                || super::trust_cg_dispatch::should_use_trust_cg(
                    self.trust_cg_structurally_vetoed(),
                );
            let need_action_bytecode = need_action_bytecode || self.value_action_vm.requested();
            if need_action_bytecode {
                self.compile_action_bytecode();
            }
        }

        // Phase 1 of Algebraic Geometry: Static Topology Analysis
        if !self.symmetry.perms.is_empty() {
            #[cfg(debug_assertions)]
            eprintln!(
                "[topology] Symmetry group names: {:?}",
                self.symmetry.group_names
            );
            let analyzer = super::bfs::topology::analyzer::TopologyAnalyzer::new();
            if let Some(next_def) = self.module.op_defs.get(&next_name) {
                let evidence = analyzer.analyze_stability(next_def, &self.symmetry.group_names);
                if evidence.is_stable {
                    #[cfg(debug_assertions)]
                    eprintln!(
                        "[topology] Detected statically stable action topology: {}",
                        evidence.action_name
                    );
                    self.homotopic_canonicalizer = Some(
                        super::bfs::topology::canonicalize::HomotopicCanonicalizer::new(&evidence),
                    );
                }
            }
        }

        // Part of #3850: Initialize tiered JIT manager after action splitting
        // so we know the action count. The tier manager tracks per-action
        // compilation tiers and makes promotion decisions based on evaluation
        // frequency.
        if crate::check::debug::jit_enabled() {
            self.initialize_tier_manager();
        }

        // Part of #4118: Initialize trust-codegen native compilation cache.
        // Must be called after compile_action_bytecode() so bytecode is available.
        // Active by default (the interpreter is the fallback); skipped when
        // TY_TRUST_CG=0 forces the interpreter or the backend can't admit the spec.
        //
        // Layout-sensitive aggregate actions are deferred until init-state
        // layout is known. Specs such as EWD998Small and MCLamportMutex would
        // otherwise do a layout-blind first pass, emit stale diagnostics, then
        // rebuild immediately after authoritative flat/native ABI layout promotion.
        if super::trust_cg_dispatch::should_use_trust_cg(self.trust_cg_structurally_vetoed())
            && super::trust_cg_dispatch::should_defer_pre_layout_trust_cg_cache_build(
                self.action_bytecode.as_ref(),
            )
        {
            eprintln!(
                "[trust-cg] deferring pre-layout native compilation until init-state layout is available: action bytecode uses layout-sensitive state aggregate access",
            );
        } else {
            self.maybe_initialize_trust_cg_cache_eager_or_defer();
        }

        Ok(next_name)
    }

    /// Compile invariant operators to bytecode for VM-accelerated evaluation.
    ///
    /// Part of #3578: Attempts bytecode compilation for all configured invariant
    /// names. The result is stored in `self.bytecode` and consulted during
    /// `check_invariants_array` to bypass tree-walking evaluation.
    // pub(in crate::check): also driven standalone by the GPU admission path
    // (model_checker/gpu_admit.rs), which prepares bytecode without running
    // the CPU BFS.
    pub(in crate::check) fn compile_invariant_bytecode(&mut self) {
        if self.config.invariants.is_empty() && self.config.constraints.is_empty() {
            return;
        }

        // Compile the union of invariant + state-constraint operators into the
        // same bytecode chunk. The CPU invariant checker only ever evaluates
        // ops named in `config.invariants` (see `check_invariants_via_bytecode`,
        // which iterates the passed name list), so the extra constraint ops sit
        // in the chunk unused by safety checking — but the GPU admission gather
        // (`gpu_admit.rs`) needs the constraint bytecode present to lower state
        // constraints for on-device pruning. Deduplicate in case a name appears
        // in both lists.
        let mut operator_names: Vec<String> = self.config.invariants.clone();
        for c in &self.config.constraints {
            if !operator_names.contains(c) {
                operator_names.push(c.clone());
            }
        }

        // Get module references from tir_parity if available, otherwise use
        // the root module from the context.
        let (mut root, mut deps) = if let Some(tir) = &self.tir_parity {
            let (root, deps) = tir.clone_modules();
            (root, deps)
        } else {
            return;
        };

        // The cloned module's operator bodies contain Expr::Ident for state
        // variables (state var resolution in checker_setup.rs only applies to
        // the op_defs HashMap, not the module AST). The TIR lowerer needs
        // Expr::StateVar nodes to emit LoadVar opcodes; without this, variable
        // references lower to TirNameKind::Ident and the bytecode compiler
        // emits LoadConst with a string name instead of LoadVar with a
        // VarRegistry index — causing the VM to evaluate against wrong values.
        use tla_core::ast::Unit;
        let registry = self.ctx.var_registry().clone();
        for unit in &mut root.units {
            if let Unit::Operator(def) = &mut unit.node {
                tla_eval::state_var::resolve_state_vars_in_op_def(def, &registry);
            }
        }
        // Also resolve state vars in dependency modules — invariants defined
        // in EXTENDS'd base specs reference the same state variables.
        for dep in &mut deps {
            for unit in &mut dep.units {
                if let Unit::Operator(def) = &mut unit.node {
                    tla_eval::state_var::resolve_state_vars_in_op_def(def, &registry);
                }
            }
        }

        // Diagnostic: show what modules and operators are available for compilation.
        if super::debug::bytecode_vm_stats_enabled() {
            let root_ops: Vec<_> = root
                .units
                .iter()
                .filter_map(|u| {
                    if let Unit::Operator(def) = &u.node {
                        Some(def.name.node.as_str())
                    } else {
                        None
                    }
                })
                .collect();
            eprintln!(
                "[bytecode] root module '{}': {} operators: {:?}",
                root.name.node,
                root_ops.len(),
                &root_ops[..root_ops.len().min(10)]
            );
            for (i, dep) in deps.iter().enumerate() {
                let dep_ops: Vec<_> = dep
                    .units
                    .iter()
                    .filter_map(|u| {
                        if let Unit::Operator(def) = &u.node {
                            Some(def.name.node.as_str())
                        } else {
                            None
                        }
                    })
                    .collect();
                eprintln!(
                    "[bytecode] dep[{i}] module '{}': {} operators: {:?}",
                    dep.name.node,
                    dep_ops.len(),
                    &dep_ops[..dep_ops.len().min(10)]
                );
            }
        }

        let dep_refs: Vec<&Module> = deps.iter().collect();
        let tir_callees =
            tla_eval::bytecode_vm::collect_bytecode_namespace_callees(&root, &dep_refs);
        // INSTANCE coverage: pass the state-var name→index map so invariant
        // bodies that reach into INSTANCE-imported operators (e.g.
        // `Buffer!TypeOk` referencing the instance variable `ringbuffer`,
        // mapped to the parent's same-named state var via the instance's
        // implicit substitution) resolve to `LoadVar` slots instead of failing
        // with `unresolved identifier`.
        let state_var_map = state_var_index_map(&registry);
        let compiled = tla_eval::bytecode_vm::compile_operators_to_bytecode_full_with_state_vars(
            &root,
            &dep_refs,
            &operator_names,
            self.ctx.precomputed_constants(),
            Some(self.ctx.op_replacements()),
            Some(&tir_callees),
            Some(&state_var_map),
        );

        // Keep the rollout measurements behind the stats flag, but allow
        // release-mode reason logging via TY_DEBUG_BYTECODE_VM=1.
        let stats_enabled = super::debug::bytecode_vm_stats_enabled();
        let reason_logs_enabled = stats_enabled || debug_bytecode_vm();
        if stats_enabled {
            eprintln!(
                "[bytecode] compiled {}/{} invariants ({} failed)",
                compiled.op_indices.len(),
                self.config.invariants.len(),
                compiled.failed.len(),
            );
        }
        if reason_logs_enabled {
            for (name, err) in &compiled.failed {
                eprintln!("[bytecode]   skip {name}: {err}");
            }
        }
        #[cfg(debug_assertions)]
        if super::debug::ty_debug() {
            eprintln!(
                "[#3578] Bytecode VM: compiled {}/{} invariants ({} failed)",
                compiled.op_indices.len(),
                self.config.invariants.len(),
                compiled.failed.len(),
            );
            for (name, err) in &compiled.failed {
                eprintln!("[#3578]   skip {name}: {err}");
            }
        }
        if !compiled.op_indices.is_empty() {
            // Part of #3582: JIT-compile eligible bytecode invariants to native code.
            if crate::check::debug::jit_enabled() {
                match JitInvariantCacheImpl::build(&compiled.chunk, &compiled.op_indices) {
                    Ok(cache) => {
                        let jit_count = cache.len();
                        if jit_count > 0 {
                            if stats_enabled {
                                eprintln!(
                                    "[jit] compiled {}/{} bytecode invariants to native code",
                                    jit_count,
                                    compiled.op_indices.len(),
                                );
                            }
                            let all = cache.covers_all(&self.config.invariants);
                            self.jit_all_compiled = all;
                            self.jit_resolved_fns = if all {
                                cache.resolve_ordered(&self.config.invariants)
                            } else {
                                None
                            };
                            self.jit_cache = Some(cache);
                        }
                    }
                    Err(e) => {
                        if reason_logs_enabled {
                            eprintln!("[jit] JIT compilation failed: {e}");
                        }
                    }
                }
            }

            self.bytecode = Some(compiled);
        }
    }

    /// Lever 1 (#EWD998PCal): compile CONSTRAINT operators to a dedicated
    /// bytecode chunk for the per-candidate-state fast path in
    /// `check_state_constraints_array`.
    ///
    /// Mirrors `compile_invariant_bytecode` (state-var resolution, namespace
    /// callees, INSTANCE state-var map) but compiles ONLY the constraint ops,
    /// so it cannot flip invariant checking onto the bytecode VM path (which
    /// `tir_blocks_bytecode_vm` intentionally prevents in TIR eval mode).
    ///
    /// All-or-nothing: the fast path is only armed when EVERY configured
    /// constraint compiled — a partial set would force per-state wholesale
    /// fallback anyway, paying VM setup for nothing.
    pub(in crate::check) fn compile_constraint_bytecode(&mut self) {
        if self.config.constraints.is_empty() {
            return;
        }
        let (mut root, mut deps) = if let Some(tir) = &self.tir_parity {
            tir.clone_modules()
        } else {
            return;
        };

        // Resolve state vars so constraint bodies emit LoadVar (VarRegistry
        // slots) instead of unresolved identifiers — same requirement as
        // compile_invariant_bytecode.
        use tla_core::ast::Unit;
        let registry = self.ctx.var_registry().clone();
        for unit in &mut root.units {
            if let Unit::Operator(def) = &mut unit.node {
                tla_eval::state_var::resolve_state_vars_in_op_def(def, &registry);
            }
        }
        for dep in &mut deps {
            for unit in &mut dep.units {
                if let Unit::Operator(def) = &mut unit.node {
                    tla_eval::state_var::resolve_state_vars_in_op_def(def, &registry);
                }
            }
        }

        let dep_refs: Vec<&Module> = deps.iter().collect();
        let tir_callees =
            tla_eval::bytecode_vm::collect_bytecode_namespace_callees(&root, &dep_refs);
        let state_var_map = state_var_index_map(&registry);
        let compiled = tla_eval::bytecode_vm::compile_operators_to_bytecode_full_with_state_vars(
            &root,
            &dep_refs,
            &self.config.constraints,
            self.ctx.precomputed_constants(),
            Some(self.ctx.op_replacements()),
            Some(&tir_callees),
            Some(&state_var_map),
        );

        let stats_enabled = super::debug::bytecode_vm_stats_enabled();
        if stats_enabled || debug_bytecode_vm() {
            eprintln!(
                "[bytecode-constraint] compiled {}/{} constraints ({} failed)",
                compiled.op_indices.len(),
                self.config.constraints.len(),
                compiled.failed.len(),
            );
            for (name, err) in &compiled.failed {
                eprintln!("[bytecode-constraint]   skip {name}: {err}");
            }
        }

        let all_compiled = self
            .config
            .constraints
            .iter()
            .all(|c| compiled.op_indices.contains_key(c));
        if all_compiled {
            self.constraint_bytecode = Some(compiled);
        }
    }

    /// AUTO engine-selection structural pre-check.
    ///
    /// Returns `Some(reason)` when the next-state action structure makes the
    /// trust-cg native path non-beneficial, so the AUTO selector should route to
    /// the interpreter BEFORE paying any native compile cost. Returns `None`
    /// when native should be attempted (the post-compile coverage gate provides
    /// the second line of defense).
    ///
    /// Signal: a quantifier (`\E`/`\A`) or `CHOOSE` whose bound domain
    /// enumerates a FUNCTION-SET (`[D -> R]`) or POWERSET (`SUBSET S`) — directly
    /// or through a set-filter/set-builder generator (`{f \in [D -> R] : P}`).
    /// Native successor generation cannot lower these (they are not enumerable
    /// proof-backed compact-set domains; lowering fails closed), and the
    /// split-action bytecode compile of such bodies is super-linear, so the
    /// native attempt is strictly wasted work versus the interpreter. The scan
    /// is over the already-resolved per-action ASTs (`split_action_meta[*].expr`)
    /// and needs no bytecode — it is cheap relative to compilation.
    ///
    /// Purely structural (AST shape). It never inspects spec/action names.
    fn trust_cg_native_hostile_action_structure(&self) -> Option<&'static str> {
        let meta = self.compiled.split_action_meta.as_deref()?;
        for inst in meta {
            if let Some(expr) = inst.expr.as_ref() {
                if expr_has_native_hostile_quantifier_domain(&expr.node) {
                    return Some(
                        "action quantifies over a function-set or powerset domain \
                         (not natively enumerable); interpreter is faster",
                    );
                }
            }
        }
        None
    }

    /// Compile action operators to bytecode for JIT next-state dispatch.
    ///
    /// Part of #3910: The JitNextStateCache needs bytecode for split-action operators
    /// (like "Send", "Receive"), not invariant operators. This method compiles the
    /// action operators discovered by split_action_instances into bytecode that
    /// native backends can lower to machine code.
    ///
    /// No-op when:
    /// - No split_action_meta (monolithic single-action specs)
    /// - tir_parity modules unavailable (no AST to compile from)
    // pub(in crate::check): also driven standalone by the GPU admission path
    // (model_checker/gpu_admit.rs).
    pub(in crate::check) fn compile_action_bytecode(&mut self) {
        if self
            .compiled
            .split_action_meta
            .as_ref()
            .map_or(true, |m| m.is_empty())
        {
            self.compiled.split_action_complete_bindings = None;
            return;
        }

        // Collect unique action operator names from BOTH sources:
        // 1. split_action_meta (leaf actions: "RecvMsg", "PassToken", etc.)
        // 2. coverage.actions (detected actions: "System", "Environment", etc.)
        //
        // We need both because the JIT dispatch uses detected action names
        // (from run_gen.rs per-action loop) while deeper split actions may
        // also be referenced. Having both ensures cache hits regardless of
        // which naming level the dispatch uses.
        //
        // Part of: JIT name mismatch fix — detected vs split action names.
        let mut name_set = std::collections::HashSet::new();
        if let Some(meta) = self.compiled.split_action_meta.as_ref() {
            for m in meta {
                if let Some(name) = &m.name {
                    name_set.insert(name.clone());
                }
            }
        }
        for action in self.coverage.actions.iter() {
            name_set.insert(action.name.clone());
        }
        // Get module references (same source as invariant bytecode compilation).
        let (mut root, mut deps) = if let Some(tir) = &self.tir_parity {
            let (root, deps) = tir.clone_modules();
            (root, deps)
        } else {
            self.compiled.split_action_complete_bindings = None;
            return;
        };

        if let Some(meta) = self.compiled.split_action_meta.as_ref() {
            add_bound_split_action_synthetic_ops(
                &mut root,
                meta,
                &mut name_set,
                &self.ctx.shared().ops,
                self.ctx.op_replacements(),
            );
        }

        let action_names: Vec<String> = name_set.into_iter().collect();
        if action_names.is_empty() {
            self.compiled.split_action_complete_bindings = None;
            return;
        }

        // Resolve state vars in the AST (required for LoadVar/StoreVar opcodes).
        use tla_core::ast::Unit;
        let registry = self.ctx.var_registry().clone();
        for unit in &mut root.units {
            if let Unit::Operator(def) = &mut unit.node {
                tla_eval::state_var::resolve_state_vars_in_op_def(def, &registry);
                // Rewrite static `v' \in S` into `\E x \in S : v' = x` so the
                // already-native existential generator path can compile it,
                // instead of bailing on residual SetPrimeMode. Must run after
                // resolution so the primed LHS is an `Expr::StateVar`.
                tla_core::rewrite_static_prime_in_set_in_op_def(def);
            }
        }
        for dep in &mut deps {
            for unit in &mut dep.units {
                if let Unit::Operator(def) = &mut unit.node {
                    tla_eval::state_var::resolve_state_vars_in_op_def(def, &registry);
                    tla_core::rewrite_static_prime_in_set_in_op_def(def);
                }
            }
        }

        let dep_refs: Vec<&tla_core::ast::Module> = deps.iter().collect();
        let mut tir_callees =
            tla_eval::bytecode_vm::collect_bytecode_namespace_callees(&root, &dep_refs);

        // INSTANCE name-resolution fix: when an action operator is imported through
        // an `INSTANCE` layer (i.e., it is NOT defined in the root module), the
        // authoritative TIR-callee body collected above carries the *inner* module's
        // unresolved variable references (e.g. `clean!heat` referencing the inner
        // `tee`), which the bytecode compiler rejects as `unresolved identifier`.
        //
        // The action splitter has already produced the correct, substitution-applied
        // (`apply_substitutions` + `qualify_instance_ops`) and state-var-resolved
        // expression for each such action in `split_action_meta[*].expr` — this is the
        // SAME resolved AST the interpreter evaluates. Lower that expression and use it
        // to override the stale instance callee body, so the native action-bytecode
        // compile sees the interpreter's resolved action. Root-defined operators are
        // left untouched (the existing TIR-callee body is already correct for them),
        // bounding the blast radius to genuinely instance-imported actions.
        override_instance_action_callees_from_split(
            &mut tir_callees,
            &root,
            &dep_refs,
            self.compiled.split_action_meta.as_deref(),
        );

        // INSTANCE coverage: pass the state-var name→index map so action bodies
        // that reach into INSTANCE-imported operators (e.g. `Buffer!Write`
        // referencing the instance variable `ringbuffer`, mapped to the
        // parent's same-named state var via the instance's implicit
        // substitution) resolve to `LoadVar`/`StoreVar` slots instead of
        // failing with `unresolved identifier`.
        let state_var_map = state_var_index_map(&registry);
        // Action bytecode compiles with And/Or chain FLATTENING by default (no
        // register recycling — registers stay monotone-unique so the next-state
        // transform is unaffected): the flattener turns multi-arm And/Or chains into the ONE
        // canonical single-accumulator shape (`Move OR<-r ; JumpTrue OR->end`)
        // that the top-level-disjunction splitter detects — a 3+-arm PlusCal
        // action (`node == Arm1 \/ Arm2 \/ ...`) otherwise compiles as a
        // right-nested or-cascade the splitter fails closed on, blocking the
        // whole next-state/native pipeline (EWD998PCal). Flattening preserves
        // the exact left-to-right evaluation and short-circuit order (see
        // compile_bool_chain_recycled); TY_NO_ACTION_FLAT_CHAINS=1 restores
        // the historical pairwise-nested compile.
        let action_flat_chains = std::env::var_os("TY_NO_ACTION_FLAT_CHAINS").is_none();
        let compiled = if action_flat_chains {
            tla_eval::bytecode_vm::compile_operators_to_bytecode_with_flat_bool_chains(
                &root,
                &dep_refs,
                &action_names,
                self.ctx.precomputed_constants(),
                Some(self.ctx.op_replacements()),
                Some(&tir_callees),
                Some(&state_var_map),
            )
        } else {
            tla_eval::bytecode_vm::compile_operators_to_bytecode_full_with_state_vars(
                &root,
                &dep_refs,
                &action_names,
                self.ctx.precomputed_constants(),
                Some(self.ctx.op_replacements()),
                Some(&tir_callees),
                Some(&state_var_map),
            )
        };

        let stats_enabled = super::debug::bytecode_vm_stats_enabled();
        let reason_logs = stats_enabled || super::debug::debug_bytecode_vm();
        if reason_logs {
            eprintln!(
                "[bytecode] action compilation: {}/{} actions compiled ({} failed)",
                compiled.op_indices.len(),
                action_names.len(),
                compiled.failed.len(),
            );
            for (name, err) in &compiled.failed {
                eprintln!("[bytecode]   action skip {name}: {err}");
            }
        }

        if compiled.op_indices.is_empty() {
            self.compiled.split_action_complete_bindings = None;
            if self.value_action_vm.plan_requested() {
                self.value_action_vm
                    .reject_plan("no split action operator compiled to bytecode".to_string());
            }
            return;
        }

        // Part of #3910: Transform action predicate bytecode into next-state
        // function bytecode. Rewrites `LoadPrime(x) + Eq` → `StoreVar(x, expr)`
        // so the JIT next-state cache can produce successor states.
        let mut transformed_count = 0usize;
        let mut transformed = compiled;
        let action_entries: Vec<(String, u16)> = transformed
            .op_indices
            .iter()
            .map(|(name, &func_idx)| (name.clone(), func_idx))
            .collect();
        let action_entry_count = action_entries.len();
        transformed.op_indices.clear();
        for (name, func_idx) in action_entries {
            let Some(original_instructions) = transformed
                .chunk
                .functions
                .get(func_idx as usize)
                .map(|func| func.instructions.clone())
            else {
                continue;
            };

            // Diagnostics: `TY_BYTECODE_DUMP_ACTION=NameA,NameB` prints the
            // pre-transform bytecode listing of the named actions (plus their
            // transitive callees) so fail-closed skips are diagnosable without
            // a debugger. Purely observational.
            if std::env::var("TY_BYTECODE_DUMP_ACTION")
                .map(|v| v.split(',').any(|n| n == name))
                .unwrap_or(false)
            {
                dump_action_bytecode(&name, func_idx, &transformed.chunk);
            }

            // Fail-closed: a DISJUNCTIVE action (`A \/ B` at the action level,
            // visible as `Or` / short-circuit `JumpTrue` in the predicate
            // bytecode) is a relational fork — one state can satisfy BOTH arms
            // and yield two successors. The next-state transform produces a
            // single-successor generator and silently keeps only one arm
            // (observed: TCommit's Decide lost its "aborted" arm → 15 of 34
            // states). Until the transform can split arms, such actions get no
            // next-state bytecode: the interpreter keeps handling them and
            // every native/GPU consumer of this map declines them. The scan is
            // transitive over chunk callees (operator bodies hide the `\/`).
            // A DISJUNCTIVE action (`A \/ B` at the action level, visible as
            // `Or` / short-circuit `JumpTrue`) is a relational fork — one state
            // can satisfy several arms and yield several successors, which a
            // single-successor next-state generator cannot represent (observed:
            // TCommit's Decide lost its "aborted" arm → 15 of 34 states).
            //
            // Split the top-level disjunction into per-disjunct sub-actions
            // (`split_top_level_disjunction_general` is union-exact: the BFS
            // engine unions the successors of distinct action functions), then
            // transform EACH arm to next-state. On full success register the
            // arms as separate `{name}#d{k}` next-state generators; if the split
            // fails to match OR any arm fails to transform, fall back to the
            // sound fail-closed behavior (no next-state bytecode → the
            // interpreter keeps handling the monolithic action). Purely
            // additive: this can only convert prior declines into admits.
            if super::gpu_admit::bytecode_reaches_disjunction(
                &original_instructions,
                &transformed.chunk,
            ) {
                let base_func = transformed.chunk.functions.get(func_idx as usize).cloned();
                // Only split fully-specialized (arity-0) actions: the
                // next-state transform and trust-ir lowering require arity-0
                // entrypoints, and unspecialized `A(r)` actions are handled by
                // the separate binding-specialization path.
                let split = base_func
                    .as_ref()
                    .filter(|f| f.arity == 0)
                    .and_then(tla_tir::bytecode::split_top_level_disjunction_general);
                // Per arm: try the plain next-state transform first (keeps
                // previously-working splits byte-identical). Only when that
                // fails, retry after fail-closed bytecode normalization
                // (const-closure ValueApply -> Call, effectful-callee
                // inlining, boolean-position constant-domain quantifier
                // unrolling) — this is what lets guard-quantifier arms like
                // PaxosCommit's `Decide` become straight-line generators.
                if reason_logs {
                    match &split {
                        Some(subs) => eprintln!(
                            "[bytecode]   action '{name}': disjunction split -> {} arms",
                            subs.len()
                        ),
                        None => eprintln!(
                            "[bytecode]   action '{name}': disjunction split matched NO canonical chain"
                        ),
                    }
                }
                let transformed_arms: Option<Vec<tla_tir::bytecode::BytecodeFunction>> = split
                    .and_then(|subs| {
                        let mut arms = Vec::with_capacity(subs.len());
                        for (arm_idx, mut sub) in subs.into_iter().enumerate() {
                            let plain = tla_tir::bytecode::action_transform::transform_action_to_next_state_with_constants(
                                &sub.instructions,
                                &transformed.chunk.constants,
                            );
                            if let tla_tir::bytecode::action_transform::ActionTransformOutcome::Transformed(ni) = plain {
                                // Validate against the same chunk the arm
                                // will live in (constants/callees are shared
                                // with the parent action).
                                if super::validate_next_state_action_chunk(
                                    func_idx,
                                    &ni,
                                    &transformed.chunk,
                                    self.module.vars.len(),
                                )
                                .is_ok()
                                {
                                    sub.instructions = ni;
                                    arms.push(sub);
                                    continue;
                                }
                            }
                            // Retry with normalization (fail-closed; pool
                            // additions are append-only and never disturb
                            // existing constant indices).
                            let chunk = &mut transformed.chunk;
                            let Some(normalized) =
                                tla_tir::bytecode::normalize_action_function(
                                    &sub,
                                    &chunk.functions,
                                    &mut chunk.constants,
                                )
                            else {
                                if reason_logs {
                                    eprintln!(
                                        "[bytecode]   action '{name}' arm {arm_idx}: normalization failed"
                                    );
                                }
                                return None;
                            };
                            let retried = tla_tir::bytecode::action_transform::transform_action_to_next_state_with_constants(
                                &normalized.instructions,
                                &transformed.chunk.constants,
                            );
                            let tla_tir::bytecode::action_transform::ActionTransformOutcome::Transformed(ni) = retried
                            else {
                                if reason_logs {
                                    eprintln!(
                                        "[bytecode]   action '{name}' arm {arm_idx}: next-state transform failed ({retried:?})"
                                    );
                                }
                                return None;
                            };
                            if let Err(e) = super::validate_next_state_action_chunk(
                                func_idx,
                                &ni,
                                &transformed.chunk,
                                self.module.vars.len(),
                            ) {
                                if reason_logs {
                                    eprintln!(
                                        "[bytecode]   action '{name}' arm {arm_idx}: validation failed ({e})"
                                    );
                                }
                                return None;
                            }
                            let mut arm = normalized;
                            arm.instructions = ni;
                            arms.push(arm);
                        }
                        Some(arms)
                    });
                if let Some(arms) = transformed_arms.filter(|a| !a.is_empty()) {
                    for (k, mut arm) in arms.into_iter().enumerate() {
                        let arm_name = format!("{name}#d{k}");
                        arm.name = arm_name.clone();
                        let new_idx = u16::try_from(transformed.chunk.functions.len())
                            .expect("action function table exceeds u16");
                        transformed.chunk.functions.push(arm);
                        transformed.op_indices.insert(arm_name, new_idx);
                        transformed_count += 1;
                    }
                    if reason_logs {
                        eprintln!(
                            "[bytecode]   action '{name}': split disjunction into per-arm \
                             next-state sub-actions"
                        );
                    }
                } else {
                    transformed.failed.push((
                        name.clone(),
                        tla_tir::bytecode::CompileError::Unsupported(
                            "disjunctive action: split-and-transform of co-enabled arms failed"
                                .to_string(),
                        ),
                    ));
                    if reason_logs {
                        eprintln!(
                            "[bytecode]   action '{name}': skipped (disjunctive; arm split/transform failed)"
                        );
                    }
                }
                continue;
            };

            match tla_tir::bytecode::action_transform::transform_action_to_next_state_with_constants(
                &original_instructions,
                &transformed.chunk.constants,
            ) {
                tla_tir::bytecode::action_transform::ActionTransformOutcome::Transformed(
                    new_instructions,
                ) => match super::validate_next_state_action_chunk(
                    func_idx,
                    &new_instructions,
                    &transformed.chunk,
                    self.module.vars.len(),
                ) {
                    Ok(()) => {
                        if let Some(func) = transformed.chunk.functions.get_mut(func_idx as usize) {
                            func.instructions = new_instructions;
                        }
                        transformed.op_indices.insert(name.clone(), func_idx);
                        transformed_count += 1;
                        if reason_logs {
                            eprintln!("[bytecode]   action '{name}': transformed to next-state");
                        }
                    }
                    Err(reason) => {
                        transformed.failed.push((
                            name.clone(),
                            tla_tir::bytecode::CompileError::Unsupported(format!(
                                "unsafe next-state transform: {reason}"
                            )),
                        ));
                        if reason_logs {
                            eprintln!("[bytecode]   action '{name}': skipped ({reason})");
                        }
                    }
                },
                tla_tir::bytecode::action_transform::ActionTransformOutcome::NoRewrite => {
                    transformed.failed.push((
                        name.clone(),
                        tla_tir::bytecode::CompileError::Unsupported(
                            "no safe next-state rewrite found".to_string(),
                        ),
                    ));
                    if reason_logs {
                        eprintln!(
                            "[bytecode]   action '{name}': skipped (no prime assignment pattern found)"
                        );
                    }
                }
                tla_tir::bytecode::action_transform::ActionTransformOutcome::Unsafe(reason) => {
                    transformed.failed.push((
                        name.clone(),
                        tla_tir::bytecode::CompileError::Unsupported(format!(
                            "unsafe next-state transform: {reason}"
                        )),
                    ));
                    if reason_logs {
                        eprintln!("[bytecode]   action '{name}': skipped ({reason})");
                    }
                }
            }
        }
        if reason_logs {
            eprintln!(
                "[bytecode] action transform: {transformed_count}/{} actions → next-state",
                action_entry_count,
            );
        }
        if !transformed.op_indices.is_empty() {
            let value_action_vm_plan = self.value_action_vm.plan_requested().then(|| {
                super::value_action_vm::ValueActionVmPlan::build_with_first_guards(
                    self.compiled
                        .split_action_meta
                        .as_deref()
                        .unwrap_or_default(),
                    &transformed,
                    self.module.vars.len(),
                    self.compiled.split_action_complete_bindings.as_deref(),
                    &self.ctx,
                    &self.module.vars,
                )
            });
            self.compiled.split_action_complete_bindings = None;
            self.action_bytecode = Some(transformed);
            if let Some(plan) = value_action_vm_plan {
                match plan {
                    Ok(plan) => self.value_action_vm.install_plan(plan),
                    Err(reason) => self.value_action_vm.reject_plan(reason),
                }
            }
        } else if self.value_action_vm.plan_requested() {
            self.compiled.split_action_complete_bindings = None;
            self.value_action_vm
                .reject_plan("no action survived next-state transformation".to_string());
        }
    }

    /// Infer and store a `StateLayout` for flat i64 state representation.
    ///
    /// Called after init state solving when we have a concrete initial state
    /// to infer variable types from. The inferred layout maps each state
    /// variable to a contiguous region of i64 slots, enabling `FlatState`
    /// conversions for JIT-compiled transition functions and invariant checks.
    ///
    /// No-op when no initial states are available.
    ///
    /// Part of #3986: Wire FlatState into BFS path.
    pub(in crate::check) fn infer_flat_state_layout(
        &mut self,
        first_init_state: &crate::state::ArrayState,
    ) {
        let registry = self.ctx.var_registry().clone();
        let layout_constants = self.layout_inference_constants();
        let mut sequence_proofs = self.configured_sequence_capacity_proofs();
        let mut sequence_element_proofs = self.configured_sequence_element_layout_proofs();
        let set_bitmask_range_proofs = self.configured_set_bitmask_range_type_proofs();
        let fixed_scalar_range_proofs = self.configured_fixed_scalar_range_type_proofs();
        if let (Some(init_name), Some(next_name)) = (&self.config.init, &self.config.next) {
            let op_defs = &self.ctx.shared().ops;
            let init_name = self.ctx.resolve_op_name(init_name).to_string();
            let next_name = self.ctx.resolve_op_name(next_name).to_string();
            if let (Some(init_def), Some(next_def)) =
                (op_defs.get(&init_name), op_defs.get(&next_name))
            {
                let seed_values: Vec<crate::Value> = first_init_state
                    .values()
                    .iter()
                    .map(crate::Value::from)
                    .collect();
                crate::state::collect_sequence_capacity_writer_proofs_with_ops(
                    &init_def.as_ref().body.node,
                    &next_def.as_ref().body.node,
                    "Init/Next sequence writer proof",
                    &registry,
                    &seed_values,
                    &layout_constants,
                    op_defs,
                    self.ctx.op_replacements(),
                    &mut sequence_proofs,
                );
                let seeded_sequence_element_proofs = sequence_element_proofs.clone();
                crate::state::collect_sequence_element_layout_writer_proofs_with_ops(
                    &init_def.as_ref().body.node,
                    &next_def.as_ref().body.node,
                    "Init/Next sequence writer proof",
                    &registry,
                    &seed_values,
                    &layout_constants,
                    op_defs,
                    self.ctx.op_replacements(),
                    &seeded_sequence_element_proofs,
                    &mut sequence_element_proofs,
                );
            }
        }
        // Duplicate-free bounded-universe capacity proofs: prove `Len(v) <=
        // |U|` for `v \in Seq(U)` sequences whose writers provably keep them
        // duplicate-free (e.g. a scheduler queue that only appends
        // not-yet-scheduled clients). Runs after the plain writer fixpoint so
        // it can supersede a degenerate zero-capacity claim for the same slot.
        self.collect_duplicate_free_sequence_capacity_proofs_into(
            &layout_constants,
            &mut sequence_proofs,
        );
        // HEURISTIC (unproven, backstop-guarded) element-universe capacity for a
        // growing `v \in Seq(U)` sequence whose DF certificate did not fire
        // (`TY_SEQ_HEURISTIC_CAPACITY`, default OFF => no-op). Runs LAST so it
        // never shadows a proven capacity proof.
        self.collect_heuristic_sequence_capacity_proofs_into(&mut sequence_proofs);
        // Derive sequence-element layout proofs for function ranges proven by a
        // checked type invariant (e.g. `unchecked \in [1..N -> SUBSET Procs]`,
        // `nxt \in [1..N -> 1..N]`). Runs *after* the writer-relation element
        // proof so the latter still produces its own proofs for growable
        // writer-bounded sequences; structurally-identical duplicates are
        // de-conflicted by the element-proof uniqueness check. The range type is
        // enforced on every reachable state (incl. empty-at-INIT), so it is
        // sound to use directly as the sequence-element layout.
        crate::state::derive_set_valued_sequence_element_proofs(
            &set_bitmask_range_proofs,
            &fixed_scalar_range_proofs,
            &mut sequence_element_proofs,
        );
        let sequence_fixed_domain_type_proofs = self.configured_sequence_fixed_domain_type_proofs();
        let mut tagged_scalar_set_range_proofs =
            self.configured_tagged_scalar_set_range_type_proofs();
        let fixed_scalar_var_proofs = self.configured_fixed_scalar_var_type_proofs();
        let set_bitmask_type_proofs = self.configured_set_bitmask_type_proofs();
        let record_set_bitmask_type_proofs = self.configured_record_set_bitmask_type_proofs();
        tagged_scalar_set_range_proofs
            .extend(self.action_producer_tagged_scalar_set_range_type_proofs(first_init_state));
        let mut layout =
            crate::state::infer_layout_with_sequence_layout_tagged_set_type_and_range_proofs(
                first_init_state,
                &registry,
                &sequence_proofs,
                &sequence_element_proofs,
                &sequence_fixed_domain_type_proofs,
                &tagged_scalar_set_range_proofs,
                &fixed_scalar_range_proofs,
                &set_bitmask_type_proofs,
                &set_bitmask_range_proofs,
                &fixed_scalar_var_proofs,
                &record_set_bitmask_type_proofs,
            );

        // #43-extension fail-closed gate: an init-sampled scalar slot (e.g.
        // `x = 0` in Init → `Scalar`) carries no writer verification. If any
        // Init/Next writer can assign a SET (or other non-scalar) to such a var,
        // keeping it as a flat-primary scalar slot would alias distinct
        // set-valued states against scalar states in the flat fingerprint and
        // silently undercount the BFS (missed-violation soundness bug). Demote
        // those scalar-slot vars to `Dynamic` so they stay on the sound
        // interpreter successor path. Genuinely-scalar vars (no non-scalar
        // writer) are not in the veto set and are unaffected.
        let nonscalar_writer_veto = self.nonscalar_writer_vetoed_vars();
        let demoted = layout.veto_flat_primary_scalar_slot_vars(&nonscalar_writer_veto);
        if !demoted.is_empty() {
            eprintln!(
                "[flat_state] #43-veto: demoted {} init-sampled scalar-slot var(s) with non-scalar writers to Dynamic (flat-primary unsafe): {:?}",
                demoted.len(),
                demoted,
            );
        }

        // Promote proven heterogeneous scalar-union vars (`focus \in Nodes \cup
        // {NIL}`) from their fail-closed one-slot scalar kind to a
        // `TaggedScalarUnion` slot. Runs AFTER the #43 veto so a var with a
        // non-scalar writer is already `Dynamic` and is skipped — the union
        // override only fires on scalar kinds that survived the writer veto.
        let union_var_proofs = self.configured_tagged_scalar_union_var_type_proofs();
        let init_sample_rows = vec![first_init_state
            .values()
            .iter()
            .map(crate::Value::from)
            .collect::<Vec<crate::Value>>()];
        let union_function_var_proofs =
            self.configured_tagged_scalar_union_function_var_type_proofs();
        let union_tuple_function_var_proofs =
            self.configured_tagged_scalar_union_tuple_function_var_type_proofs();
        let fixed_scalar_tuple_function_var_proofs =
            self.configured_fixed_scalar_range_tuple_function_var_type_proofs();
        let tagged_union_var_proofs = self.configured_tagged_union_var_type_proofs();
        if !union_var_proofs.is_empty() {
            crate::state::apply_tagged_scalar_union_var_overrides(
                &mut layout,
                &union_var_proofs,
                &init_sample_rows,
            );
        }

        // WP-09/Part A: promote proven tuple-keyed function RANGE scalar
        // unions (btree `childOf`/`valOf`) from the fail-closed `ScalarSlots`
        // range encoding to the injective universe-index `TaggedScalarUnion`
        // encoding. Runs after the #43 veto (a var with a non-scalar writer is
        // already dropped by the collector's writer corroboration) and only
        // fires when every sampled range value fits the proven universe.
        let union_range_proofs = self.configured_tagged_scalar_union_range_type_proofs();
        if !union_range_proofs.is_empty() {
            crate::state::apply_tuple_keyed_tagged_scalar_union_range_overrides(
                &mut layout,
                &union_range_proofs,
                &init_sample_rows,
            );
        }

        // WP-ARGS: promote scalar-or-tuple union vars (btree `args`) from the
        // one-slot scalar kind Init-sampling inferred to a tag-dispatched
        // `TaggedUnion` slot. Runs after the scalar-union override so a var that
        // is a pure scalar union has already been claimed by the narrower
        // carrier.
        let scalar_tuple_proofs = self.configured_scalar_tuple_union_var_proofs();
        if !scalar_tuple_proofs.is_empty() {
            crate::state::apply_scalar_tuple_union_var_overrides(
                &mut layout,
                &scalar_tuple_proofs,
                &init_sample_rows,
            );
        }
        {
            // Function-range unions (`lastOf \in [Nodes -> Nodes \cup {NIL}]`):
            // promote the observed fail-closed function kind to a
            // `Recursive { [D -> union] }` slot so each range slot stores the
            // injective universe index and the native FuncApply/FuncExcept union
            // lowering engages. Runs after the top-level override; a var already
            // promoted there (a scalar union) is not a function candidate.
            crate::state::apply_tagged_scalar_union_function_var_overrides(
                &mut layout,
                &union_function_var_proofs,
                &init_sample_rows,
            );
            // Tuple/cross-product-domain function ranges
            // (`childOf \in [Nodes \X Keys -> Nodes \cup {NIL}]`): promote the
            // observed scalar-slot `TupleKeyedArray` to a `TaggedScalarUnion`
            // range encoding so the composed tuple-key-carrier + union-index
            // native lowering engages. Runs after the scalar function override;
            // a var it already promoted is no longer a scalar-slot tuple array.
            crate::state::apply_tagged_scalar_union_tuple_function_var_overrides(
                &mut layout,
                &union_tuple_function_var_proofs,
                &init_sample_rows,
            );
            // Homogeneous model-value/string tuple ranges
            // (`valOf \in [Nodes \X Keys -> Vals \cup {NIL}]`): promote the
            // observed scalar-slot `TupleKeyedArray` to a `FixedScalar` range so
            // the injective `NameId` slot becomes flat-primary safe and the tuple
            // carrier's native FuncApply/FuncExcept engage. Runs after the union
            // override; a var it already promoted is no longer scalar-slot.
            crate::state::apply_fixed_scalar_range_tuple_function_var_overrides(
                &mut layout,
                &fixed_scalar_tuple_function_var_proofs,
                &init_sample_rows,
            );
            // Polymorphic top-level SUM-TYPE vars (`args \in {NIL} \cup
            // {<<k>>:...} \cup {<<k,v>>:...}`): promote the observed fail-closed
            // scalar/dynamic kind to a `Recursive { TaggedUnion }` tag+payload
            // slot so the var has a fixed, round-trip-correct flat encoding.
            // (Native tag-dispatch that would make it flat-primary is future
            // work — currently the var maps to `Dynamic` on the ABI and every
            // action touching it fails closed to the interpreter.)
            crate::state::apply_tagged_union_var_overrides(
                &mut layout,
                &tagged_union_var_proofs,
                &init_sample_rows,
            );
        }

        let flat_bytes = crate::state::flat_state_bytes(&layout);
        let stats_enabled = super::debug::bytecode_vm_stats_enabled();
        if stats_enabled {
            eprintln!(
                "[flat_state] inferred layout: {} vars, {} total slots, {} bytes/state, \
                 all_scalar={}, trivial={}, fully_flat={}, has_dynamic={}",
                layout.var_count(),
                layout.total_slots(),
                flat_bytes,
                layout.is_all_scalar(),
                layout.is_trivial(),
                layout.is_fully_flat(),
                layout.has_dynamic_vars(),
            );
        }
        let layout_arc = std::sync::Arc::new(layout);
        // Part of #3986: Create the FlatBfsBridge alongside the layout.
        let bridge = crate::state::FlatBfsBridge::new(std::sync::Arc::clone(&layout_arc));

        if stats_enabled {
            eprintln!(
                "[flat_state] bridge: fully_flat={}, num_slots={}, bytes_per_state={}",
                bridge.is_fully_flat(),
                bridge.num_slots(),
                bridge.bytes_per_state(),
            );
        }

        self.flat_state_layout = Some(layout_arc);
        // Part of #4126: Create adapter for Tier 0 interpreter sandwich.
        // Verify the first initial state roundtrips correctly through the flat
        // representation. This catches specs with string/model-value variables
        // that layout inference classifies as Scalar but the i64 roundtrip
        // would corrupt (e.g., "black" → 0 → SmallInt(0)).
        let mut adapter = crate::state::FlatBfsAdapter::new(bridge.clone());
        let mut verify_state = first_init_state.clone();
        let roundtrip_ok = adapter.verify_roundtrip(&mut verify_state, &registry);
        // Log roundtrip result: always log on failure (auto-detect may have
        // wanted to activate), or when stats are enabled. Under
        // TY_TRUST_CG_DIAG=1 the failure becomes a hard guard instead of a
        // silent fallback (issue #4275).
        self.log_or_panic_flat_roundtrip_verification(
            &adapter,
            first_init_state,
            &registry,
            roundtrip_ok,
            stats_enabled,
            "adapter",
            "first_init",
            "Init",
        );
        self.flat_bfs_adapter = Some(adapter);
        self.flat_bfs_bridge = Some(bridge);

        // Part of #3986/#4356: Detect if flat i64 state can be the primary
        // BFS representation.
        //
        // Conditions: fully-flat fixed layout that is primary-safe, roundtrip
        // verified, no VIEW, no SYMMETRY, flat state not explicitly disabled,
        // and no full-state storage. Fully-flat includes scalar vars plus
        // fixed-size records/arrays whose complete value is represented in the
        // flat slot buffer; Dynamic layouts stay on the adapter/interpreter-
        // sandwich path.
        //
        // Part of #4298: Gate activation on `store_full_states == false`. Same
        // rationale as the #4281 fix for `jit_compiled_fp_active`: in full-state
        // mode the `seen` HashMap and `seen_fps` set are already populated with
        // FP64 fingerprints by `init_states_full_state()` before layout inference
        // runs here. If we now flip to flat-primary, successors would be
        // fingerprinted via `FlatState::fingerprint_compiled()` (xxh3 on raw i64
        // buffer) while init states remain under FP64 — the same state value
        // (e.g., stuttering `x=0`) gets two different fingerprints, inflating
        // the distinct-state count and breaking parity with TLC (e.g.,
        // `system_loop_no_fair_2w`).
        {
            let fully_flat = self
                .flat_state_layout
                .as_ref()
                .is_some_and(|l| l.is_fully_flat());
            let flat_primary_safe = self
                .flat_state_layout
                .as_ref()
                .is_some_and(|l| l.supports_flat_primary());
            let flat_bfs = self.should_use_flat_bfs();
            self.flat_state_primary = self.should_use_flat_state_primary_storage();
            telemetry_eprintln!(
                "[flat_state] flat_state_primary={}: roundtrip_ok={}, fully_flat={}, flat_primary_safe={}, view={}, symmetry={}, flat_bfs={}, full_state_storage={}",
                self.flat_state_primary,
                roundtrip_ok,
                fully_flat,
                flat_primary_safe,
                self.compiled.cached_view_name.is_some(),
                !self.symmetry.perms.is_empty(),
                flat_bfs,
                self.state_storage.store_full_states,
            );
            if stats_enabled && !flat_primary_safe {
                if let Some(layout) = self.flat_state_layout.as_ref() {
                    for reason in layout.flat_primary_blockers() {
                        eprintln!("[flat_state] flat-primary blocker: {reason}");
                    }
                }
            }
            if stats_enabled && self.flat_state_primary {
                eprintln!(
                    "[flat_state] flat_state_primary=true: fully-flat fixed layout, \
                     no VIEW, no SYMMETRY — flat i64 is primary BFS representation",
                );
            }
        }
        self.invalidate_compiled_bfs_after_flat_primary_promotion();
    }

    /// Infer and store a `StateLayout` from a wavefront of initial states.
    ///
    /// Like `infer_flat_state_layout` but examines multiple states for
    /// robustness. If variable shapes disagree across states, the
    /// conflicting variable falls back to `Dynamic`.
    ///
    /// Part of #3986: Layout inference from first wavefront (~1000 states).
    pub(in crate::check) fn infer_flat_state_layout_from_wavefront(
        &mut self,
        states: &[crate::state::ArrayState],
    ) {
        if states.is_empty() {
            return;
        }

        let registry = self.ctx.var_registry().clone();
        let layout_constants = self.layout_inference_constants();
        let mut sequence_proofs = self.configured_sequence_capacity_proofs();
        let mut sequence_element_proofs = self.configured_sequence_element_layout_proofs();
        let set_bitmask_range_proofs = self.configured_set_bitmask_range_type_proofs();
        let fixed_scalar_range_proofs = self.configured_fixed_scalar_range_type_proofs();
        if let (Some(init_name), Some(next_name)) = (&self.config.init, &self.config.next) {
            let op_defs = &self.ctx.shared().ops;
            let init_name = self.ctx.resolve_op_name(init_name).to_string();
            let next_name = self.ctx.resolve_op_name(next_name).to_string();
            if let (Some(init_def), Some(next_def)) =
                (op_defs.get(&init_name), op_defs.get(&next_name))
            {
                let seed_values: Vec<crate::Value> =
                    states[0].values().iter().map(crate::Value::from).collect();
                crate::state::collect_sequence_capacity_writer_proofs_with_ops(
                    &init_def.as_ref().body.node,
                    &next_def.as_ref().body.node,
                    "Init/Next sequence writer proof",
                    &registry,
                    &seed_values,
                    &layout_constants,
                    op_defs,
                    self.ctx.op_replacements(),
                    &mut sequence_proofs,
                );
                let seeded_sequence_element_proofs = sequence_element_proofs.clone();
                crate::state::collect_sequence_element_layout_writer_proofs_with_ops(
                    &init_def.as_ref().body.node,
                    &next_def.as_ref().body.node,
                    "Init/Next sequence writer proof",
                    &registry,
                    &seed_values,
                    &layout_constants,
                    op_defs,
                    self.ctx.op_replacements(),
                    &seeded_sequence_element_proofs,
                    &mut sequence_element_proofs,
                );
            }
        }
        // Duplicate-free bounded-universe capacity proofs (see
        // `infer_flat_state_layout` for rationale).
        self.collect_duplicate_free_sequence_capacity_proofs_into(
            &layout_constants,
            &mut sequence_proofs,
        );
        // HEURISTIC (unproven, backstop-guarded) element-universe capacity for a
        // growing `v \in Seq(U)` sequence whose DF certificate did not fire
        // (`TY_SEQ_HEURISTIC_CAPACITY`, default OFF => no-op). Runs LAST so it
        // never shadows a proven capacity proof.
        self.collect_heuristic_sequence_capacity_proofs_into(&mut sequence_proofs);
        // Derive sequence-element layout proofs for function ranges proven by a
        // checked type invariant (e.g. `unchecked \in [1..N -> SUBSET Procs]`,
        // `nxt \in [1..N -> 1..N]`). Runs *after* the writer-relation element
        // proof so the latter still produces its own proofs for growable
        // writer-bounded sequences; structurally-identical duplicates are
        // de-conflicted by the element-proof uniqueness check. The range type is
        // enforced on every reachable state (incl. empty-at-INIT), so it is
        // sound to use directly as the sequence-element layout.
        crate::state::derive_set_valued_sequence_element_proofs(
            &set_bitmask_range_proofs,
            &fixed_scalar_range_proofs,
            &mut sequence_element_proofs,
        );
        let sequence_fixed_domain_type_proofs = self.configured_sequence_fixed_domain_type_proofs();
        let mut tagged_scalar_set_range_proofs =
            self.configured_tagged_scalar_set_range_type_proofs();
        let fixed_scalar_var_proofs = self.configured_fixed_scalar_var_type_proofs();
        let set_bitmask_type_proofs = self.configured_set_bitmask_type_proofs();
        let record_set_bitmask_type_proofs = self.configured_record_set_bitmask_type_proofs();
        tagged_scalar_set_range_proofs
            .extend(self.action_producer_tagged_scalar_set_range_type_proofs(&states[0]));
        let layout =
            crate::state::infer_layout_from_wavefront_with_sequence_layout_tagged_set_type_and_range_proofs(
                states,
                &registry,
                &sequence_proofs,
                &sequence_element_proofs,
                &sequence_fixed_domain_type_proofs,
                &tagged_scalar_set_range_proofs,
                &fixed_scalar_range_proofs,
                &set_bitmask_type_proofs,
                &set_bitmask_range_proofs,
                &fixed_scalar_var_proofs,
                &record_set_bitmask_type_proofs,
            );

        // #43-extension fail-closed gate (see `infer_flat_state_layout`): demote
        // any init-sampled scalar-slot var with a non-scalar (set) writer to
        // `Dynamic` so it never becomes a flat-primary scalar slot that aliases
        // distinct set-valued states.
        let mut layout = layout;
        let nonscalar_writer_veto = self.nonscalar_writer_vetoed_vars();
        let demoted = layout.veto_flat_primary_scalar_slot_vars(&nonscalar_writer_veto);
        if !demoted.is_empty() {
            eprintln!(
                "[flat_state] #43-veto (wavefront): demoted {} init-sampled scalar-slot var(s) with non-scalar writers to Dynamic (flat-primary unsafe): {:?}",
                demoted.len(),
                demoted,
            );
        }

        // Promote proven heterogeneous scalar-union vars (see the init-state
        // path). Every wavefront state's value for the var must fit the proven
        // universe, else the override is skipped (fail closed).
        let union_var_proofs = self.configured_tagged_scalar_union_var_type_proofs();
        let wavefront_sample_rows: Vec<Vec<crate::Value>> = states
            .iter()
            .map(|s| s.values().iter().map(crate::Value::from).collect())
            .collect();
        let tagged_union_var_proofs = self.configured_tagged_union_var_type_proofs();
        if !union_var_proofs.is_empty() {
            crate::state::apply_tagged_scalar_union_var_overrides(
                &mut layout,
                &union_var_proofs,
                &wavefront_sample_rows,
            );
        }

        // WP-09/Part A: same tuple-keyed range-union promotion on the
        // wavefront path. Every sampled state's range values must fit the
        // proven universe, else the override is skipped (fail closed).
        let union_range_proofs = self.configured_tagged_scalar_union_range_type_proofs();
        if !union_range_proofs.is_empty() {
            crate::state::apply_tuple_keyed_tagged_scalar_union_range_overrides(
                &mut layout,
                &union_range_proofs,
                &wavefront_sample_rows,
            );
        }

        // WP-ARGS: same promotion on the wavefront path. Every sampled state's
        // value must fit the union, else the override is skipped (fail closed).
        let scalar_tuple_proofs = self.configured_scalar_tuple_union_var_proofs();
        if !scalar_tuple_proofs.is_empty() {
            crate::state::apply_scalar_tuple_union_var_overrides(
                &mut layout,
                &scalar_tuple_proofs,
                &wavefront_sample_rows,
            );
        }
        // Polymorphic top-level sum-type vars (see the init-state path).
        crate::state::apply_tagged_union_var_overrides(
            &mut layout,
            &tagged_union_var_proofs,
            &wavefront_sample_rows,
        );

        let flat_bytes = crate::state::flat_state_bytes(&layout);
        let stats_enabled = super::debug::bytecode_vm_stats_enabled();
        if stats_enabled {
            eprintln!(
                "[flat_state] wavefront layout ({} states): {} vars, {} total slots, {} bytes/state, \
                 all_scalar={}, trivial={}, fully_flat={}, has_dynamic={}",
                states.len(),
                layout.var_count(),
                layout.total_slots(),
                flat_bytes,
                layout.is_all_scalar(),
                layout.is_trivial(),
                layout.is_fully_flat(),
                layout.has_dynamic_vars(),
            );
        }

        let layout_arc = std::sync::Arc::new(layout);
        let bridge = crate::state::FlatBfsBridge::new(std::sync::Arc::clone(&layout_arc));

        if stats_enabled {
            eprintln!(
                "[flat_state] wavefront bridge: fully_flat={}, num_slots={}, bytes_per_state={}",
                bridge.is_fully_flat(),
                bridge.num_slots(),
                bridge.bytes_per_state(),
            );
        }

        self.flat_state_layout = Some(layout_arc.clone());
        // Part of #4126: Create adapter for Tier 0 interpreter sandwich.
        let mut adapter = crate::state::FlatBfsAdapter::new(bridge.clone());
        let mut verify_state = states[0].clone();
        let roundtrip_ok = adapter.verify_roundtrip(&mut verify_state, &registry);
        self.log_or_panic_flat_roundtrip_verification(
            &adapter,
            &states[0],
            &registry,
            roundtrip_ok,
            stats_enabled,
            "wavefront adapter",
            "wavefront[0]",
            "Init",
        );
        self.flat_bfs_adapter = Some(adapter);
        self.flat_bfs_bridge = Some(bridge);

        // Part of #3986/#4356: Detect if flat i64 state can be the primary
        // BFS representation. See the single-state setter above for the
        // full domain and full-state-storage rationale.
        // Part of #4298: Gate on `!store_full_states` — see the single-state
        // `infer_flat_state_layout` setter above for the full rationale. Flipping
        // the fingerprint algorithm after init states are already stored in the
        // `seen`/`seen_fps` domain inflates distinct-state counts (same state
        // value ends up with both an FP64 init fingerprint and an xxh3 successor
        // fingerprint).
        {
            let flat_primary_safe = layout_arc.supports_flat_primary();
            let flat_bfs = self.should_use_flat_bfs();
            self.flat_state_primary = self.should_use_flat_state_primary_storage();
            telemetry_eprintln!(
                "[flat_state] flat_state_primary={}: roundtrip_ok={}, fully_flat={}, flat_primary_safe={}, view={}, symmetry={}, flat_bfs={}, full_state_storage={}",
                self.flat_state_primary,
                roundtrip_ok,
                layout_arc.is_fully_flat(),
                flat_primary_safe,
                self.compiled.cached_view_name.is_some(),
                !self.symmetry.perms.is_empty(),
                flat_bfs,
                self.state_storage.store_full_states,
            );
            if stats_enabled && !flat_primary_safe {
                for reason in layout_arc.flat_primary_blockers() {
                    eprintln!("[flat_state] flat-primary blocker: {reason}");
                }
            }
            if stats_enabled && self.flat_state_primary {
                eprintln!(
                    "[flat_state] flat_state_primary=true (wavefront): fully-flat fixed layout, \
                     no VIEW, no SYMMETRY — flat i64 is primary BFS representation",
                );
            }
        }
        self.invalidate_compiled_bfs_after_flat_primary_promotion();
    }

    fn invalidate_compiled_bfs_after_flat_primary_promotion(&mut self) {
        if !self.flat_state_primary {
            return;
        }
        let promoted_non_scalar_layout = self
            .flat_state_layout
            .as_ref()
            .is_some_and(|layout| !layout.is_all_scalar());
        if !promoted_non_scalar_layout {
            return;
        }
        let flat_slots = self
            .flat_bfs_adapter
            .as_ref()
            .filter(|adapter| adapter.is_fully_flat())
            .map(|adapter| adapter.num_slots())
            .or_else(|| {
                self.flat_state_layout
                    .as_ref()
                    .map(|layout| layout.total_slots())
            });
        self.clear_layout_sensitive_compiled_bfs_artifacts(
            "flat_state_primary layout promotion",
            flat_slots,
        );
    }

    fn clear_layout_sensitive_compiled_bfs_artifacts(
        &mut self,
        reason: &str,
        flat_slots: Option<usize>,
    ) {
        let had_step = self.compiled_bfs_step.is_some();
        let had_level = self.compiled_bfs_level.is_some();
        let had_trust_cg = self.trust_cg_cache.is_some() || self.trust_cg_build_stats.is_some();

        if !had_step && !had_level && !had_trust_cg {
            return;
        }

        let width_detail = flat_slots
            .map(|slots| {
                format!(
                    ", flat_slots={slots}, logical_vars={}",
                    self.module.vars.len()
                )
            })
            .unwrap_or_default();

        eprintln!(
            "[compiled-bfs] clearing layout-sensitive compiled artifacts before rebuild: \
             reason={reason}{width_detail}, compiled_bfs_step={had_step}, \
             compiled_bfs_level={had_level}, trust_cg_cache={}, trust_cg_build_stats={}",
            self.trust_cg_cache.is_some(),
            self.trust_cg_build_stats.is_some(),
        );

        self.compiled_bfs_step = None;
        self.compiled_bfs_level = None;
        // The deferral marker is layout-coupled state: a rebuild re-runs
        // `initialize_trust_cg_cache`, which re-decides deferral for the new
        // layout. A stale marker without a cache would otherwise promote a
        // level for a layout that no longer exists.
        self.deferred_fused_level_build = false;
        {
            self.trust_cg_cache = None;
            self.trust_cg_build_stats = None;
        }
    }

    /// Get the flat state layout, if inferred.
    ///
    /// Returns `None` before `infer_flat_state_layout` has been called.
    ///
    /// Part of #3986: Wire FlatState into BFS path.
    #[must_use]
    #[allow(dead_code)]
    pub(in crate::check) fn flat_state_layout(
        &self,
    ) -> Option<&std::sync::Arc<crate::state::StateLayout>> {
        self.flat_state_layout.as_ref()
    }

    /// Get the FlatBfsBridge, if created.
    ///
    /// Returns `None` before `infer_flat_state_layout` has been called.
    /// The bridge provides cheap `ArrayState <-> FlatState` conversions
    /// and fingerprint bridging at the BFS boundary.
    ///
    /// Part of #3986: Wire FlatState into BFS engine.
    #[must_use]
    #[allow(dead_code)]
    pub(in crate::check) fn flat_bfs_bridge(&self) -> Option<&crate::state::FlatBfsBridge> {
        self.flat_bfs_bridge.as_ref()
    }

    /// Get a clone of the FlatBfsAdapter, if created.
    ///
    /// Returns `None` before `infer_flat_state_layout` has been called.
    /// The adapter wraps the bridge with BFS-specific convenience methods
    /// for the interpreter sandwich (FlatState -> ArrayState -> eval ->
    /// ArrayState -> FlatState).
    ///
    /// Returns a clone because adapters are per-worker (mutable stats).
    ///
    /// Part of #4126: FlatState as native BFS representation (Phase E).
    #[must_use]
    #[allow(dead_code)]
    pub(in crate::check) fn flat_bfs_adapter(&self) -> Option<crate::state::FlatBfsAdapter> {
        self.flat_bfs_adapter.clone()
    }

    /// Determine whether flat state may be the primary storage/fingerprint domain.
    ///
    /// This is separate from [`Self::should_use_flat_bfs`]. Flat BFS auto-
    /// admission is deliberately narrower because it controls the adapter
    /// sandwich queue policy. Flat-primary storage can admit any fully-flat,
    /// roundtrip-verified layout that `StateLayout::supports_flat_primary()`
    /// proves safe, while still honoring explicit flat-state disables and the
    /// feature gates that must stay fail-closed.
    #[must_use]
    fn should_use_flat_state_primary_storage(&self) -> bool {
        if !self.config.trace_invariants.is_empty() {
            return false;
        }

        flat_state_primary_storage_admitted(
            self.config.use_flat_state == Some(false) || crate::check::debug::flat_bfs_disabled(),
            self.compiled.cached_view_name.is_some(),
            !self.symmetry.perms.is_empty(),
            self.state_storage.store_full_states,
            self.flat_bfs_adapter
                .as_ref()
                .is_some_and(|adapter| adapter.roundtrip_verified() && adapter.is_fully_flat()),
            self.flat_state_layout
                .as_ref()
                .is_some_and(|layout| layout.supports_flat_primary()),
        )
    }

    /// Determine whether flat BFS should be used for this run.
    ///
    /// The decision hierarchy is:
    /// 1. `Config::use_flat_state = Some(false)` → disabled (programmatic override)
    /// 2. `TY_NO_FLAT_BFS=1` → disabled (env var override)
    /// 3. Trace invariants configured → disabled (history reconstruction needs
    ///    full parent-chain provenance)
    /// 4. `Config::use_flat_state = Some(true)` → enabled if adapter ready
    /// 5. Auto-detect: enabled when adapter is present, roundtrip verified,
    ///    and the layout passes the conservative flat-BFS auto-admission
    ///    predicate.
    ///
    /// The auto-detect path (5) is the default for specs where the inferred
    /// flat layout is complete and stable enough for default admission. It
    /// covers typical scalar specs while keeping init-only fixed function
    /// shapes out of the default path.
    ///
    /// Part of #4126: Auto-detect flat BFS for scalar specs.
    #[must_use]
    pub(in crate::check) fn should_use_flat_bfs(&self) -> bool {
        // 1. Programmatic force-disable
        if self.config.use_flat_state == Some(false) {
            return false;
        }
        // 2. Env var force-disable
        if crate::check::debug::flat_bfs_disabled() {
            return false;
        }
        // Trace-history invariants need complete parent-chain reconstruction.
        // Keep compact flat BFS modes fail-closed until their trace provenance
        // is proven complete.
        if !self.config.trace_invariants.is_empty() {
            return false;
        }

        // Programmatic force-enable still requires roundtrip verification AND a
        // growth-safety floor: a sampled-capacity (`Observed`) sequence is lossy
        // once a successor grows past the inferred capacity, which silently
        // corrupts flat dedup and inflates the state count.
        // `supports_forced_flat_bfs()` refuses those layouts while still
        // admitting the sequence-free model-value `StringKeyedArray` sandbox.
        let force_ready = self
            .flat_bfs_adapter
            .as_ref()
            .is_some_and(|a| a.roundtrip_verified() && a.layout().supports_forced_flat_bfs());

        // 3. Programmatic force-enable
        if self.config.use_flat_state == Some(true) {
            return force_ready;
        }

        let adapter_ready = self
            .flat_bfs_adapter
            .as_ref()
            .is_some_and(|a| a.roundtrip_verified());

        // 4. Auto-detect: enable only when the verified flat layout is also
        // safe for default admission. Fully-flat alone only proves sampled
        // states fit the layout; it does not prove future successors keep the
        // same shape.
        if !adapter_ready {
            return false;
        }
        self.flat_bfs_adapter
            .as_ref()
            .is_some_and(|a| a.layout().supports_flat_bfs_auto_admission())
    }

    /// Whether flat i64 state is the primary BFS representation for this run.
    ///
    /// True when ALL of:
    /// - The inferred layout is fully flat — `layout.is_fully_flat()`
    /// - Roundtrip verification passed
    /// - No VIEW expression configured
    /// - No SYMMETRY reduction active
    /// - Flat BFS is enabled
    /// - Full-state storage is disabled
    ///
    /// When true, the fingerprint-only BFS hot path can store states as
    /// contiguous `[i64]` buffers and use the flat compiled fingerprint
    /// domain. Scalar layouts remain the simple case; fixed-size records and
    /// arrays are also eligible when their slots are a complete state
    /// representation. Dynamic layouts are not eligible.
    ///
    /// Part of #3986: Flat i64 state as primary BFS representation.
    #[must_use]
    #[allow(dead_code)]
    pub(in crate::check) fn is_flat_state_primary(&self) -> bool {
        self.flat_state_primary
    }

    /// Upgrade the JIT invariant cache with compound layout information.
    ///
    /// Called after init state solving when layout information is available.
    /// If the check-side flat layout is fully fixed, that authoritative layout
    /// is converted for JIT/trust_cg use; otherwise this falls back to inferring
    /// from the concrete initial state. The initial `JitInvariantCache::build()`
    /// has no layout info, so compound-type variable accesses (records,
    /// functions, tuples) fall back to the interpreter. Rebuilding with
    /// `build_with_layout()` enables native compound access in JIT-compiled
    /// invariants.
    ///
    /// No-op when:
    /// - Legacy JIT compatibility path is disabled
    /// - No JIT cache exists (no invariants were JIT-compiled)
    /// - No bytecode exists (no invariants were bytecode-compiled)
    /// - The initial state has no compound variables (layout upgrade is unnecessary)
    ///
    /// Part of #3910: JIT invariant layout upgrade for native compound access.
    pub(in crate::check) fn upgrade_jit_cache_with_layout(
        &mut self,
        first_init_state: &crate::state::ArrayState,
    ) {
        // Derive layout for BOTH invariant JIT and next-state JIT, so compute
        // it even when jit_cache (invariant) is None.
        let compact_values = first_init_state.values();
        let has_compound = compact_values
            .iter()
            .any(|cv| !cv.is_int() && !cv.is_bool());
        if !has_compound {
            let has_action_bytecode = self
                .action_bytecode
                .as_ref()
                .is_some_and(|bytecode| !bytecode.op_indices.is_empty());
            if super::trust_cg_dispatch::should_use_trust_cg(self.trust_cg_structurally_vetoed())
                && has_action_bytecode
                && self.trust_cg_cache.is_none()
                && self.trust_cg_build_stats.is_none()
            {
                self.maybe_initialize_trust_cg_cache_eager_or_defer();
            } else if super::trust_cg_dispatch::should_use_trust_cg(
                self.trust_cg_structurally_vetoed(),
            ) && has_action_bytecode
                && self.flat_state_primary
                && !self.compiled.eval_implied_actions.is_empty()
                && !self.implied_actions_require_interpreter_eval()
                && self
                    .compiled_bfs_level
                    .as_ref()
                    .map_or(true, |level| !level.has_native_fused_level())
            {
                // All-scalar specs build the trust-codegen cache pre-layout in
                // `prepare_bfs_common` (before `flat_state_primary` is known).
                // At that point `native_fused_state_len` is absent, so any
                // native-capable action-property (`PROPERTY [][A]_vars`) term is
                // rejected by the fused parent loop and the whole
                // `CompiledBfsLevel` build returns `None` (leaving
                // `compiled_bfs_level` unset) — forcing the heavy interpreter
                // `eval_implied_actions` per-transition hook. Now that
                // `flat_state_primary` is confirmed, rebuild the cache so the fused
                // level is constructed with the verified flat slot width and the
                // implied action is checked inside the native parent loop. The
                // `map_or(true, …)` covers both the rejected-to-`None` case above
                // and any partial level that lacks the native fused loop. Gated
                // purely on structural predicates (scalar layout already flat-primary,
                // every implied-action term native-capable); never on spec name.
                // Mirrors the compound-spec post-layout rebuild below.
                let flat_slots = self
                    .flat_bfs_adapter
                    .as_ref()
                    .filter(|adapter| adapter.is_fully_flat())
                    .map(|adapter| adapter.num_slots());
                self.clear_layout_sensitive_compiled_bfs_artifacts(
                    "rebuild trust-codegen cache so scalar native-capable implied actions use the native fused parent loop",
                    flat_slots,
                );
                self.maybe_initialize_trust_cg_cache_eager_or_defer();
            }
            // All variables are scalar — layout offers no benefit.
            return;
        }

        let stats_enabled = super::debug::bytecode_vm_stats_enabled();
        let state_layout = if let Some(flat_layout) = self
            .flat_state_layout
            .as_deref()
            .filter(|layout| layout.is_fully_flat())
        {
            let state_layout = crate::state::check_layout_to_jit_layout(flat_layout);
            if stats_enabled {
                eprintln!(
                    "[jit] using authoritative flat layout for JIT: {} vars, {} compact slots",
                    state_layout.var_count(),
                    crate::state::layout_bridge::jit_layout_compact_slot_count(&state_layout),
                );
            }
            state_layout
        } else {
            let var_layouts: Vec<tla_jit_abi::VarLayout> = compact_values
                .iter()
                .map(|cv| {
                    let value = tla_value::Value::from(cv);
                    tla_jit_abi::infer_var_layout(&value)
                })
                .collect();
            let mut inferred = tla_jit_abi::StateLayout::new(var_layouts);
            // The trust-cg next-state buffer is encoded by the flat
            // `FlatBfsBridge` (see `prepare_trust_cg_next_state`) even when the
            // layout is not fully flat. A function range whose sampled init
            // value is empty infers a universe-less `Set` slot, but the bridge
            // writes that slot as a bitmask ordered by the proven `SUBSET
            // <const>` universe. Project those proven universes onto the JIT
            // layout so trust-ir read/write bit positions match the buffer.
            if let Some(flat_layout) = self.flat_state_layout.as_deref() {
                crate::state::layout_bridge::overlay_proven_set_bitmask_universes_from_flat(
                    &mut inferred,
                    flat_layout,
                );
            }
            inferred
        };

        // Store layout for next-state JIT compilation (Part of #3958).
        self.jit_state_layout = Some(state_layout.clone());

        {
            // trust-codegen action compilation currently runs during prepare(), before
            // the first init states exist. Compound specs only acquire
            // record/function layout information here, so the earlier
            // layout-blind build falls back to generic aggregate lowering.
            // Rebuild the trust-codegen cache once layout is known so action lowering
            // can take the fixed-layout fast paths.
            if super::trust_cg_dispatch::should_use_trust_cg(self.trust_cg_structurally_vetoed())
                && self.action_bytecode.is_some()
            {
                let flat_slots = self
                    .flat_bfs_adapter
                    .as_ref()
                    .filter(|adapter| adapter.is_fully_flat())
                    .map(|adapter| adapter.num_slots());
                self.clear_layout_sensitive_compiled_bfs_artifacts(
                    "rebuild trust-codegen cache with promoted flat layout",
                    flat_slots,
                );
                self.maybe_initialize_trust_cg_cache_eager_or_defer();
                // Item 4 M0-G1 (TY_HYBRID_NATIVE=1 + TY_HYBRID_FLAT_VIEW=1
                // only): additionally compile the per-action cache against the
                // HYBRID flat-view layout so compound specs with a partial
                // flat-admissible subset can dispatch flat-footprint actions
                // natively. Entirely separate cache + artifact identity from
                // the whole-state build above; a no-op unless the env switches
                // are set and the spec is compound with >=1 admissible var.
                self.maybe_initialize_trust_cg_hybrid_action_cache();
            }
        }

        // Upgrade the invariant JIT cache if it exists.
        let Some(ref bytecode) = self.bytecode else {
            return;
        };
        if self.jit_cache.is_none() {
            return;
        }

        match JitInvariantCacheImpl::build_with_layout(
            &bytecode.chunk,
            &bytecode.op_indices,
            &state_layout,
        ) {
            Ok(cache) => {
                let jit_count = cache.len();
                if jit_count > 0 {
                    if stats_enabled {
                        eprintln!(
                            "[jit] upgraded {jit_count} invariants with compound layout info",
                        );
                    }
                    let all = cache.covers_all(&self.config.invariants);
                    self.jit_all_compiled = all;
                    self.jit_resolved_fns = if all {
                        cache.resolve_ordered(&self.config.invariants)
                    } else {
                        None
                    };
                    self.jit_cache = Some(cache);
                }
            }
            Err(e) => {
                if stats_enabled {
                    eprintln!("[jit] layout upgrade failed (keeping basic cache): {e}");
                }
            }
        }
    }

    /// Attempt to activate compiled xxh3 fingerprinting for the BFS run.
    ///
    /// Checks all prerequisites:
    /// 1. All init state variables are scalar (Int/Bool) — compound values
    ///    cannot be represented as a single i64 for xxh3 hashing.
    /// 2. No VIEW expression configured — VIEW fingerprinting uses its own
    ///    custom computation path.
    /// 3. No SYMMETRY reduction — symmetry canonical fingerprinting is
    ///    incompatible with xxh3 raw-buffer hashing.
    /// 4. Native ABI state layout is available (needed for variable count).
    ///
    /// When all conditions are met, sets `jit_compiled_fp_active = true`.
    /// This flag MUST be set before any init states are fingerprinted.
    ///
    /// Part of #3987: Compiled xxh3 fingerprinting for the BFS hot path.
    pub(in crate::check) fn try_activate_compiled_fingerprinting(&mut self) {
        // Part of #4215: Structural guarantee that fingerprint algorithm cannot change
        // after BFS processing has started. If this fires, it means a code path is
        // attempting to switch fingerprint algorithms mid-run, which would cause
        // domain separation violations in the seen set.
        #[cfg(debug_assertions)]
        debug_assert!(
            !self.fp_algorithm_sealed,
            "BUG: try_activate_compiled_fingerprinting called after BFS loop started \
             (fingerprint algorithm sealed). This would mix xxh3 and FP64 fingerprints \
             in the same seen set, causing silent state loss. Part of #4215."
        );

        self.try_activate_compiled_fingerprinting_inner();

        // Part of #4319 Phase 0: fingerprint mixed-mode guard.
        //
        // Once activation decisions have been made, assert the invariant that
        // every fingerprint seen by the BFS comes from a single hash domain.
        //
        // When any trust_cg-compiled action is present, the BFS can emit two
        // classes of successors in the same run:
        //   (a) compiled successors produced by `try_trust_cg_action`
        //       (unflattened from an i64 buffer), and
        //   (b) interpreter successors produced by the fallback path.
        // Both classes ultimately flow through `array_state_fingerprint`,
        // which either (i) routes everything through
        // `fingerprint_flat_compiled` when `jit_compiled_fp_active` is true
        // (xxh3 + FLAT_COMPILED_DOMAIN_SEED — single domain), or
        // (ii) routes everything through FP64/FNV when
        // `jit_compiled_fp_active` is false (single FP64 domain). Either
        // configuration is sound *as long as the flag is consistent for the
        // whole run* — which `fp_algorithm_sealed` enforces at the BFS
        // entry and the runtime assertion in `state_fingerprint` checks for
        // the OrdMap path.
        //
        // This guard runs after both `initialize_trust_cg_cache` and
        // `try_activate_compiled_fingerprinting_inner`, so it observes the
        // final configuration.
        self.enforce_trust_cg_fingerprint_guard();
    }

    /// Actual activation logic, separated so the post-decision guard in
    /// `try_activate_compiled_fingerprinting` can inspect the final state.
    ///
    /// Part of #4319 Phase 0 refactor — body is unchanged from the previous
    /// implementation.
    fn try_activate_compiled_fingerprinting_inner(&mut self) {
        // Condition 1: No VIEW
        if self.compiled.cached_view_name.is_some() {
            return;
        }

        // Condition 2: No SYMMETRY
        if !self.symmetry.perms.is_empty() {
            return;
        }

        // Condition 3: native ABI state layout available OR flat_state_primary confirmed.
        // The jit_state_layout is only set for compound specs (upgrade path).
        // For all-scalar specs, flat_state_primary is the reliable indicator.
        // Part of #3986: Enable compiled fingerprinting for all-scalar specs
        // even without the native ABI layout (which is only needed for compound access).
        if self.jit_state_layout.is_none() && !self.flat_state_primary {
            return;
        }

        // Condition 4: Check if all init state variables are scalar.
        // We inspect the first init state for the all-Int/Bool fast-path shape.
        let all_scalar = if let Some(first_init) = self.get_first_init_state_for_layout() {
            first_init
                .values()
                .iter()
                .all(|cv| cv.is_int() || cv.is_bool())
        } else {
            // If no init state available for inspection, fall back to
            // flat_state_primary which was already verified via roundtrip.
            self.flat_state_primary
        };

        if !all_scalar {
            return;
        }

        // Condition 4b (SOUNDNESS — heterogeneously-typed scalar variable):
        //
        // The all-Int/Bool check above only inspects the FIRST init state. The
        // long-standing comment claimed "TLA+ is unityped per variable", but that
        // is FALSE: a variable can hold values of different scalar types across
        // states (e.g. `Init == x \in {1, "a"}` yields `x = 1` in one initial
        // state and `x = "a"` in another). The compiled-flat fingerprint domain
        // hashes `FlatState::from_array_state(state, compiled_fingerprint_layout)`,
        // and the wavefront layout inference (`infer_layout_from_wavefront`)
        // correctly demotes such a variable to `VarLayoutKind::Dynamic` because
        // its scalar shape conflicts across the sampled init states.
        //
        // A `Dynamic` slot is written as a constant `0` placeholder in the flat
        // buffer (the real value lives only in the ArrayState side table, which
        // the flat fingerprint never reads). Two states that differ ONLY in a
        // Dynamic var therefore produce identical flat buffers and collide under
        // the compiled-flat (xxh3) domain. Distinct logical states then either
        // (a) silently dedup to one (under-enumeration), or (b) trip the
        // canonical-payload-equality fail-closed guard (`FingerprintOverflow` /
        // `canonical_payload_mismatch`) when the differing ArrayStates are
        // compared for the same fingerprint — exactly the #523 heterogeneous
        // `SetEnum` failure.
        //
        // The authoritative homogeneity signal is the wavefront layout used for
        // compiled fingerprinting: if it carries ANY Dynamic var, the flat
        // fingerprint is lossy and the compiled-flat domain is unsound. Refuse
        // activation and keep this run on the lossless FP64/ArrayState domain,
        // which distinguishes `Int(1)` from `String("a")` faithfully.
        let compiled_fp_layout_fully_flat = self
            .flat_bfs_adapter
            .as_ref()
            .map(|adapter| adapter.layout().clone())
            .or_else(|| self.flat_state_layout.clone())
            .is_none_or(|layout| layout.is_fully_flat());
        if !compiled_fp_layout_fully_flat {
            return;
        }

        // Part of #4281: Gate activation on absence of batch-path-triggering features.
        //
        // The successor-generation dispatcher in `full_state_successors.rs` routes
        // specs with implied actions, constraints, POR, or coverage collection
        // through the batch path. The batch path calls `array_state_fingerprint`
        // with successors produced by `ArrayState::from_successor_state`, which
        // unconditionally pre-caches an FP64 (`finalize_fingerprint_xor`)
        // fingerprint on the successor `ArrayState`. The `array_state_fingerprint`
        // fast path then short-circuits on that cached FP64 value and never
        // executes the xxh3 branch — even though `jit_compiled_fp_active` is
        // true. This mixes FP64 (successors) with xxh3 (init re-fingerprint) in
        // the same `seen_fps` set, causing successors to never match init
        // fingerprints → spurious duplicate inflation.
        //
        // Refusing activation here when any batch-path trigger is configured
        // keeps `seen_fps` in a single FP64 domain for these specs. The
        // performance cost is negligible (xxh3 provides ~5% speedup on
        // all-scalar specs; specs using these features already take the slower
        // batch path).
        let batch_path_triggered = {
            {
                self.implied_actions_require_interpreter_eval()
                    || !self.config.constraints.is_empty()
                    || !self.config.action_constraints.is_empty()
                    || self.por.independence.is_some()
                    || (self.coverage.collect && !self.coverage.actions.is_empty())
            }
        };
        if batch_path_triggered {
            return;
        }

        // Part of #4281: Gate activation on `store_full_states == false`.
        //
        // When full-state storage is enabled (liveness, trace reconstruction,
        // fairness), the `seen` HashMap is already populated with FP64 keys by
        // `init_states_full_state()` before this function runs. Activating xxh3
        // here would cause successors to be looked up in `seen_fps` with xxh3
        // fingerprints while `seen` still holds FP64 keys, breaking downstream
        // consumers (`seen.get(fp)` in liveness/safety reconstruction, trace
        // replay). Re-keying the populated `seen` HashMap mid-run is invasive
        // and introduces its own correctness risks. Keep the full-state path on
        // FP64 for single-domain consistency across `seen` and `seen_fps`.
        if self.state_storage.store_full_states {
            return;
        }

        // All conditions met — activate compiled xxh3 fingerprinting.
        self.jit_compiled_fp_active = true;

        // Pre-allocate the flat fingerprint scratch buffer to avoid resize on first use.
        // Part of #3986: Eliminate per-state Vec<i64> allocation.
        let var_count = self.ctx.var_registry().len();
        self.flat_fp_scratch.resize(var_count, 0);

        // Log activation for diagnostics.
        if super::debug::bytecode_vm_stats_enabled() {
            eprintln!(
                "[jit-fp] Compiled xxh3 fingerprinting ACTIVATED (all-scalar, no VIEW/SYMMETRY)"
            );
        }
    }

    /// Enforce the fingerprint mixed-mode invariant for trust-codegen runs.
    ///
    /// When at least one action was compiled by the trust-codegen pipeline, the BFS
    /// dispatcher (`run_gen.rs`) can emit a per-state mixture of
    /// compiled-generated and interpreter-generated successors within the
    /// same run. The crucial question is not "was an action compiled?" but
    /// "which fingerprint domain actually owns BFS dedup for this run?".
    ///
    /// For the BFS seen-set to be sound, every fingerprint entered into
    /// `seen_fps` must come from a single hash domain. `bfs_fingerprint_domain`
    /// is that classifier:
    ///   * `CompiledFlat` — xxh3 over the flat i64 buffer.
    ///   * `View` / `SymmetryCanonical` — specialized canonical domains.
    ///   * `FullStateFp64` / `ArrayFp64` — plain FP64 domains.
    ///
    /// The important subtlety is `ArrayFp64`: constrained/per-action trust-codegen
    /// runs can still be perfectly sound even when `jit_compiled_fp_active`
    /// is false, because compiled successors are unflattened back into
    /// `ArrayState` and fingerprinted through the same FP64 path as
    /// interpreter successors.
    ///
    /// Part of #4319 Phase 0 (Option D).
    fn enforce_trust_cg_fingerprint_guard(&mut self) {
        let state = TrustCgFingerprintGuardState {
            trust_cg_has_compiled_action: self.trust_cg_has_compiled_action(),
            bfs_fingerprint_domain: self.bfs_fingerprint_domain(),
        };

        match state.evaluate() {
            TrustCgFingerprintGuardOutcome::NotApplicable => {}
            TrustCgFingerprintGuardOutcome::SingleDomain { domain } => {
                if super::debug::bytecode_vm_stats_enabled() {
                    eprintln!(
                        "[trust-cg] fingerprint mixed-mode guard OK: domain={domain} \
                         (compiled actions routed through single fingerprint domain)"
                    );
                }
            }
        }
    }

    /// Verify layout compatibility between the flat BFS bridge and the JIT
    /// state layout.
    ///
    /// When both `flat_bfs_bridge` and `jit_state_layout` have been created
    /// (after init state solving), this checks that their slot counts agree.
    /// This is a safety net: if the two independent inference paths produce
    /// incompatible layouts, we log a warning and disable the JIT BFS path.
    ///
    /// No-op when either layout is missing (native backend disabled or no compound vars).
    ///
    /// Part of #3986: Phase 3 layout bridge verification.
    pub(in crate::check) fn verify_layout_compatibility(&self) {
        let Some(ref bridge) = self.flat_bfs_bridge else {
            return;
        };
        let Some(ref jit_layout) = self.jit_state_layout else {
            return;
        };

        let compatible = bridge.is_compatible_with_jit(jit_layout);
        let stats_enabled = super::debug::bytecode_vm_stats_enabled();

        if compatible {
            if stats_enabled {
                eprintln!(
                    "[flat_state] layout bridge verified: check layout ({} slots) \
                     compatible with native ABI layout ({} vars)",
                    bridge.num_slots(),
                    jit_layout.var_count(),
                );
            }
        } else {
            // Log a warning. The JIT BFS path should not use the check layout's
            // buffer format if they disagree. The interpreter path is always safe.
            let jit_slots = crate::state::layout_bridge::jit_layout_compact_slot_count(jit_layout);
            let mismatch = crate::state::layout_bridge::first_layout_slot_mismatch(
                bridge.layout(),
                jit_layout,
            )
            .map(|(idx, check_slots, jit_slots)| {
                format!("; first slot mismatch var#{idx}: check={check_slots}, jit={jit_slots}")
            })
            .unwrap_or_default();
            eprintln!(
                "[flat_state] WARNING: layout mismatch between check ({} vars, {} compact slots) \
                 and JIT ({} vars, {} compact slots){}. JIT BFS will use its own layout.",
                bridge.layout().var_count(),
                bridge.num_slots(),
                jit_layout.var_count(),
                jit_slots,
                mismatch,
            );
        }
    }
}

/// Pure-data view of the inputs to the fingerprint mixed-mode guard.
///
/// Extracted from `ModelChecker` state so the decision logic in
/// [`TrustCgFingerprintGuardState::evaluate`] can be unit-tested without
/// constructing a full `ModelChecker`. See the trust_cg fingerprint-unification
/// design, Phase 0 / Option D.
///
/// Part of #4319.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::check) struct TrustCgFingerprintGuardState {
    /// True iff `TrustCgNativeCache::has_any_compiled_action()` reports at least
    /// one successfully compiled next-state action.
    pub trust_cg_has_compiled_action: bool,
    /// The actual BFS fingerprint domain selected for the current run.
    pub bfs_fingerprint_domain: BfsFingerprintDomain,
}

/// Outcome of the fingerprint mixed-mode guard.
///
/// Part of #4319 Phase 0 (Option D).
#[derive(Debug, Clone, PartialEq, Eq)]
pub(in crate::check) enum TrustCgFingerprintGuardOutcome {
    /// No trust-codegen compiled action is present; the guard has nothing to check.
    /// (The pure BFS interpreter path uses a single FP64 domain throughout.)
    NotApplicable,
    /// At least one trust-codegen compiled action is present AND the configuration
    /// pins all successors to a single fingerprint domain. `domain` is a short
    /// human-readable tag identifying which branch took effect, used for
    /// diagnostic logging.
    SingleDomain { domain: &'static str },
}

impl TrustCgFingerprintGuardState {
    /// Classify the current fingerprint configuration.
    ///
    /// Part of #4319 Phase 0 (Option D).
    #[must_use]
    pub(in crate::check) fn evaluate(&self) -> TrustCgFingerprintGuardOutcome {
        if !self.trust_cg_has_compiled_action {
            return TrustCgFingerprintGuardOutcome::NotApplicable;
        }
        TrustCgFingerprintGuardOutcome::SingleDomain {
            domain: self.bfs_fingerprint_domain.diagnostic_name(),
        }
    }
}

#[cfg(test)]
mod auto_select_hostile_domain_tests {
    //! Unit tests for the AUTO engine-selection pre-compile structural scanner
    //! [`super::expr_has_native_hostile_quantifier_domain`].

    use super::expr_has_native_hostile_quantifier_domain;
    use num_bigint::BigInt;
    use tla_core::ast::{BoundVar, Expr};
    use tla_core::name_intern::NameId;
    use tla_core::span::Spanned;

    fn ident(name: &str) -> Spanned<Expr> {
        Spanned::dummy(Expr::Ident(name.to_string(), NameId(0)))
    }

    fn bound(name: &str, domain: Expr) -> BoundVar {
        BoundVar {
            name: Spanned::dummy(name.to_string()),
            domain: Some(Box::new(Spanned::dummy(domain))),
            pattern: None,
        }
    }

    fn func_set() -> Expr {
        // [D -> R]
        Expr::FuncSet(Box::new(ident("D")), Box::new(ident("R")))
    }

    #[test]
    fn exists_over_function_set_is_hostile() {
        // \E f \in [D -> R] : TRUE
        let e = Expr::Exists(
            vec![bound("f", func_set())],
            Box::new(Spanned::dummy(Expr::Bool(true))),
        );
        assert!(expr_has_native_hostile_quantifier_domain(&e));
    }

    #[test]
    fn exists_over_powerset_is_hostile() {
        // \E s \in SUBSET S : TRUE
        let e = Expr::Exists(
            vec![bound("s", Expr::Powerset(Box::new(ident("S"))))],
            Box::new(Spanned::dummy(Expr::Bool(true))),
        );
        assert!(expr_has_native_hostile_quantifier_domain(&e));
    }

    #[test]
    fn exists_over_set_filter_of_function_set_is_hostile() {
        // \E f \in {g \in [D -> R] : P} : TRUE  — the YoYo `UpOther` shape.
        let filter = Expr::SetFilter(
            bound("g", func_set()),
            Box::new(Spanned::dummy(Expr::Bool(true))),
        );
        let e = Expr::Exists(
            vec![bound("f", filter)],
            Box::new(Spanned::dummy(Expr::Bool(true))),
        );
        assert!(expr_has_native_hostile_quantifier_domain(&e));
    }

    #[test]
    fn choose_over_powerset_is_hostile() {
        // CHOOSE s \in SUBSET S : TRUE
        let e = Expr::Choose(
            bound("s", Expr::Powerset(Box::new(ident("S")))),
            Box::new(Spanned::dummy(Expr::Bool(true))),
        );
        assert!(expr_has_native_hostile_quantifier_domain(&e));
    }

    #[test]
    fn nested_hostile_quantifier_inside_conjunction_is_detected() {
        // x = 1 /\ (\E f \in [D -> R] : TRUE) — hostile quantifier is nested.
        let inner = Expr::Exists(
            vec![bound("f", func_set())],
            Box::new(Spanned::dummy(Expr::Bool(true))),
        );
        let e = Expr::And(
            Box::new(Spanned::dummy(Expr::Eq(
                Box::new(ident("x")),
                Box::new(Spanned::dummy(Expr::Int(BigInt::from(1)))),
            ))),
            Box::new(Spanned::dummy(inner)),
        );
        assert!(expr_has_native_hostile_quantifier_domain(&e));
    }

    #[test]
    fn exists_over_plain_finite_set_is_not_hostile() {
        // \E x \in S : TRUE — a plain set domain is natively enumerable.
        let e = Expr::Exists(
            vec![bound("x", Expr::Ident("S".to_string(), NameId(0)))],
            Box::new(Spanned::dummy(Expr::Bool(true))),
        );
        assert!(!expr_has_native_hostile_quantifier_domain(&e));
    }

    #[test]
    fn exists_over_integer_range_is_not_hostile() {
        // \E x \in 1..10 : TRUE
        let range = Expr::Range(
            Box::new(Spanned::dummy(Expr::Int(BigInt::from(1)))),
            Box::new(Spanned::dummy(Expr::Int(BigInt::from(10)))),
        );
        let e = Expr::Exists(
            vec![bound("x", range)],
            Box::new(Spanned::dummy(Expr::Bool(true))),
        );
        assert!(!expr_has_native_hostile_quantifier_domain(&e));
    }

    #[test]
    fn function_set_outside_quantifier_domain_is_not_hostile() {
        // f = [D -> R] used as a value (not a quantifier domain) is fine.
        let e = Expr::Eq(Box::new(ident("f")), Box::new(Spanned::dummy(func_set())));
        assert!(!expr_has_native_hostile_quantifier_domain(&e));
    }
}

#[cfg(test)]
mod trust_cg_fingerprint_guard_tests {
    //! Unit tests for the Phase 0 fingerprint mixed-mode guard.
    //!
    //! Part of #4319. See the trust_cg fingerprint-unification design,
    //! Phase 0 / Option D.

    use super::{
        BfsFingerprintDomain, TrustCgFingerprintGuardOutcome, TrustCgFingerprintGuardState,
    };

    /// Baseline: no compiled action, no single-domain flags — guard is inert.
    #[test]
    fn no_compiled_action_is_not_applicable() {
        let state = TrustCgFingerprintGuardState {
            trust_cg_has_compiled_action: false,
            bfs_fingerprint_domain: BfsFingerprintDomain::ArrayFp64,
        };
        assert_eq!(
            state.evaluate(),
            TrustCgFingerprintGuardOutcome::NotApplicable
        );
    }

    /// Even if single-domain flags are set, a run without compiled actions
    /// never enters the guarded code path and must classify as NotApplicable.
    #[test]
    fn no_compiled_action_ignores_other_flags() {
        let state = TrustCgFingerprintGuardState {
            trust_cg_has_compiled_action: false,
            bfs_fingerprint_domain: BfsFingerprintDomain::CompiledFlat,
        };
        assert_eq!(
            state.evaluate(),
            TrustCgFingerprintGuardOutcome::NotApplicable
        );
    }

    /// Compiled action + jit_compiled_fp_active = xxh3 single domain.
    #[test]
    fn compiled_with_jit_fp_active_is_xxh3_single_domain() {
        let state = TrustCgFingerprintGuardState {
            trust_cg_has_compiled_action: true,
            bfs_fingerprint_domain: BfsFingerprintDomain::CompiledFlat,
        };
        assert_eq!(
            state.evaluate(),
            TrustCgFingerprintGuardOutcome::SingleDomain {
                domain: "xxh3_flat_compiled",
            }
        );
    }

    /// Compiled action + store_full_states = FP64 single domain.
    #[test]
    fn compiled_with_store_full_states_is_fp64_single_domain() {
        let state = TrustCgFingerprintGuardState {
            trust_cg_has_compiled_action: true,
            bfs_fingerprint_domain: BfsFingerprintDomain::FullStateFp64,
        };
        assert_eq!(
            state.evaluate(),
            TrustCgFingerprintGuardOutcome::SingleDomain {
                domain: "fp64_full_states",
            }
        );
    }

    /// Compiled action + VIEW = VIEW single domain (all fps via VIEW output).
    #[test]
    fn compiled_with_view_is_view_single_domain() {
        let state = TrustCgFingerprintGuardState {
            trust_cg_has_compiled_action: true,
            bfs_fingerprint_domain: BfsFingerprintDomain::View,
        };
        assert_eq!(
            state.evaluate(),
            TrustCgFingerprintGuardOutcome::SingleDomain { domain: "view" }
        );
    }

    /// Compiled action + symmetry = symmetry-canonical single domain.
    #[test]
    fn compiled_with_symmetry_is_symmetry_single_domain() {
        let state = TrustCgFingerprintGuardState {
            trust_cg_has_compiled_action: true,
            bfs_fingerprint_domain: BfsFingerprintDomain::SymmetryCanonical,
        };
        assert_eq!(
            state.evaluate(),
            TrustCgFingerprintGuardOutcome::SingleDomain {
                domain: "symmetry_canonical",
            }
        );
    }

    /// Regression for the constrained trust-codegen lane: compiled actions can still
    /// be sound on the plain ArrayState FP64 domain when constraints/implied
    /// actions force the per-action/full-state successor path.
    #[test]
    fn compiled_with_array_fp64_domain_is_still_single_domain() {
        let state = TrustCgFingerprintGuardState {
            trust_cg_has_compiled_action: true,
            bfs_fingerprint_domain: BfsFingerprintDomain::ArrayFp64,
        };
        assert_eq!(
            state.evaluate(),
            TrustCgFingerprintGuardOutcome::SingleDomain {
                domain: "fp64_array_state",
            }
        );
    }

    /// Each BFS fingerprint domain should map to a stable diagnostic tag.
    #[test]
    fn domain_tags_are_stable() {
        let domains = [
            (BfsFingerprintDomain::CompiledFlat, "xxh3_flat_compiled"),
            (BfsFingerprintDomain::FullStateFp64, "fp64_full_states"),
            (BfsFingerprintDomain::View, "view"),
            (
                BfsFingerprintDomain::SymmetryCanonical,
                "symmetry_canonical",
            ),
            (
                BfsFingerprintDomain::FlatSymmetryCanonical,
                "flat_symmetry_canonical",
            ),
            (BfsFingerprintDomain::ArrayFp64, "fp64_array_state"),
        ];

        for (domain, expected) in domains {
            let state = TrustCgFingerprintGuardState {
                trust_cg_has_compiled_action: true,
                bfs_fingerprint_domain: domain,
            };
            assert_eq!(
                state.evaluate(),
                TrustCgFingerprintGuardOutcome::SingleDomain { domain: expected }
            );
        }
    }
}

#[cfg(test)]
mod expr_usize_bound_tests {
    //! Unit tests for the `TY_SEQ_CAPACITY_PROOF` `<= <bounded-expr>` extension
    //! of [`super::expr_usize_bound`]: a state-free (const-foldable) right-hand
    //! side of a `Len(seq) <= RHS` bound folds to a proven capacity, while a
    //! state-dependent RHS (or a disabled flag) fails closed.

    use super::expr_usize_bound;
    use tla_core::ast::Expr;
    use tla_core::kani_types::HashMap;
    use tla_core::name_intern::NameId;

    fn empty_constants() -> HashMap<NameId, crate::Value> {
        HashMap::default()
    }

    fn empty_replacements() -> HashMap<String, String> {
        HashMap::default()
    }

    #[test]
    fn integer_literal_is_flag_independent() {
        // A bare `Len(seq) <= 7` folds without any evaluator hook.
        let expr = Expr::Int(7.into());
        assert_eq!(
            expr_usize_bound(&expr, &empty_constants(), &empty_replacements(), None),
            Some(7)
        );
    }

    #[test]
    fn non_constant_rhs_without_hook_fails_closed() {
        // Default (flag OFF): a non-literal/non-constant RHS such as
        // `Cardinality(S)` (modeled here as an opaque apply) is NOT bounded, so
        // the sequence stays `Observed` (fail closed).
        let expr = Expr::Apply(
            Box::new(tla_core::span::Spanned::dummy(Expr::Ident(
                "Cardinality".to_string(),
                NameId(0),
            ))),
            vec![tla_core::span::Spanned::dummy(Expr::Ident(
                "S".to_string(),
                NameId(0),
            ))],
        );
        assert_eq!(
            expr_usize_bound(&expr, &empty_constants(), &empty_replacements(), None),
            None,
            "flag-off must fail closed on a non-constant RHS"
        );
    }

    #[test]
    fn const_foldable_rhs_with_hook_yields_proven_bound() {
        // Flag ON: the state-free evaluator hook folds the RHS to a constant
        // (e.g. `Cardinality(Items) = 3`), upgrading the sequence to a proven
        // capacity.
        let expr = Expr::Apply(
            Box::new(tla_core::span::Spanned::dummy(Expr::Ident(
                "Cardinality".to_string(),
                NameId(0),
            ))),
            vec![tla_core::span::Spanned::dummy(Expr::Ident(
                "Items".to_string(),
                NameId(0),
            ))],
        );
        let hook = |_: &Expr| -> Option<usize> { Some(3) };
        assert_eq!(
            expr_usize_bound(
                &expr,
                &empty_constants(),
                &empty_replacements(),
                Some(&hook)
            ),
            Some(3)
        );
    }

    #[test]
    fn state_dependent_rhs_with_hook_fails_closed() {
        // Flag ON but the RHS reads state (e.g. `Len(seq) <= someStateVar`): the
        // const-level evaluator rejects the state read, the hook returns `None`,
        // and the sequence fails closed. An over-tight/unsound bound is never
        // emitted for a state-dependent RHS.
        let expr = Expr::Ident("someStateVar".to_string(), NameId(0));
        let hook = |_: &Expr| -> Option<usize> { None };
        assert_eq!(
            expr_usize_bound(
                &expr,
                &empty_constants(),
                &empty_replacements(),
                Some(&hook)
            ),
            None,
            "a state-dependent RHS must fail closed even with the flag on"
        );
    }
}

#[cfg(test)]
#[path = "run_prepare_tests.rs"]
mod run_prepare_tests;

#[cfg(test)]
mod wp33_constant_binder_domain_tests {
    //! WP-33 unit tests for [`super::full_homogeneous_domain_values`]: the
    //! binder-domain lookup that decides whether a
    //! `\A x \in D : Len(v[x]) <= K` clause can contribute a sequence-capacity
    //! proof for a Sequence nested in a Function range.
    //!
    //! Historically `D` resolved only through `proof_domains`, which carries
    //! zero-arg OPERATOR definitions — so `\A r \in Readers` with `Readers` a
    //! CONSTANT resolved to `None` and the whole quantifier subtree was
    //! skipped. This is the Disruptor item-1 blocker: `collect_sequence_capacity_proofs`
    //! already walks into function ranges, and the identical clause written as
    //! `\A r \in RSet` with `RSet == Readers` produced the proof.

    use tla_value::Rp;
    use super::{full_homogeneous_domain_values, ProofScope};
    use std::collections::BTreeMap;
    use std::sync::Arc;
    use tla_core::ast::Expr;
    use tla_core::kani_types::HashMap;
    use tla_core::name_intern::{intern_name, NameId};

    struct EnvVarGuard {
        name: &'static str,
        previous: Option<std::ffi::OsString>,
        _lock: std::sync::MutexGuard<'static, ()>,
    }

    impl EnvVarGuard {
        fn set(name: &'static str, value: Option<&str>) -> Self {
            let lock = crate::process_env_lock();
            let previous = std::env::var_os(name);
            match value {
                Some(value) => crate::env_guard::set_var(name, value),
                None => crate::env_guard::remove_var(name),
            }
            Self {
                name,
                previous,
                _lock: lock,
            }
        }
    }

    impl Drop for EnvVarGuard {
        fn drop(&mut self) {
            match self.previous.take() {
                Some(value) => crate::env_guard::set_var(self.name, value),
                None => crate::env_guard::remove_var(self.name),
            }
        }
    }

    /// `Readers = {r1, r2, r3}` bound as a CONSTANT (the Disruptor `.cfg` shape).
    fn readers_constants() -> HashMap<NameId, crate::Value> {
        let set = crate::Value::Set(Rp::new(tla_value::value::SortedSet::from_sorted_vec(
            vec![
                crate::Value::ModelValue(Rp::from("r1")),
                crate::Value::ModelValue(Rp::from("r2")),
                crate::Value::ModelValue(Rp::from("r3")),
            ],
        )));
        let mut constants = HashMap::default();
        constants.insert(intern_name("Readers"), set);
        constants
    }

    fn readers_ident() -> Expr {
        Expr::Ident("Readers".to_string(), intern_name("Readers"))
    }

    #[test]
    fn constant_binder_domain_fails_closed_with_gate_off() {
        let _guard = EnvVarGuard::set("TY_SEQ_CAPACITY_PROOF", None);
        let resolved = full_homogeneous_domain_values(
            &readers_ident(),
            &readers_constants(),
            &tla_core::OpEnv::default(),
            &HashMap::default(),
            &BTreeMap::new(),
            &ProofScope::default(),
        );
        assert!(
            resolved.is_none(),
            "gate OFF must keep the historical fail-closed behaviour for a \
             CONSTANT binder domain"
        );
    }

    #[test]
    fn constant_binder_domain_resolves_with_gate_on() {
        let _guard = EnvVarGuard::set("TY_SEQ_CAPACITY_PROOF", Some("1"));
        let resolved = full_homogeneous_domain_values(
            &readers_ident(),
            &readers_constants(),
            &tla_core::OpEnv::default(),
            &HashMap::default(),
            &BTreeMap::new(),
            &ProofScope::default(),
        );
        let values = resolved.expect("gate ON must resolve a CONSTANT binder domain");
        assert_eq!(
            &*values,
            &[
                crate::Value::ModelValue(Rp::from("r1")),
                crate::Value::ModelValue(Rp::from("r2")),
                crate::Value::ModelValue(Rp::from("r3")),
            ],
            "the constant spelling must yield the same domain vector the \
             `RSet == Readers` alias spelling already produced"
        );
    }

    #[test]
    fn enclosing_binder_still_shadows_the_constant_name() {
        // Capture safety: an enclosing `\A Readers \in ...` shadows the constant,
        // so the name must NOT resolve to the constant's value even gate-ON.
        let _guard = EnvVarGuard::set("TY_SEQ_CAPACITY_PROOF", Some("1"));
        let mut scope = ProofScope::default();
        scope.push("Readers".to_string(), None);
        let resolved = full_homogeneous_domain_values(
            &readers_ident(),
            &readers_constants(),
            &tla_core::OpEnv::default(),
            &HashMap::default(),
            &BTreeMap::new(),
            &scope,
        );
        assert!(
            resolved.is_none(),
            "a shadowing binder must veto constant resolution (no capture)"
        );
    }

    #[test]
    fn proof_domains_operator_entry_still_wins() {
        // The pre-existing operator path is unchanged and takes precedence.
        let _guard = EnvVarGuard::set("TY_SEQ_CAPACITY_PROOF", Some("1"));
        let mut proof_domains: BTreeMap<String, Arc<[crate::Value]>> = BTreeMap::new();
        proof_domains.insert(
            "Readers".to_string(),
            Arc::from(vec![crate::Value::SmallInt(1)].into_boxed_slice()),
        );
        let resolved = full_homogeneous_domain_values(
            &readers_ident(),
            &readers_constants(),
            &tla_core::OpEnv::default(),
            &HashMap::default(),
            &proof_domains,
            &ProofScope::default(),
        );
        assert_eq!(
            resolved.as_deref(),
            Some(&[crate::Value::SmallInt(1)][..]),
            "an existing proof_domains entry must still take precedence"
        );
    }
}
