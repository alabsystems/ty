// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Shared utilities for ay-based symbolic checking (PDR and BMC).
//!
//! Extracted from duplicated code in `ay_pdr.rs` and `ay_bmc.rs` to provide
//! a single sort inference implementation. The PDR version is the superset,
//! supporting function sorts, set enums, FuncSet, SetFilter, FuncDef, Except,
//! and IF/THEN/ELSE.

use std::cell::RefCell;
use std::collections::HashSet;
use std::sync::Arc;

use tla_ay::TlaSort;
use tla_ay::{AYFrontendFamily, AYSharedEngineLane};
use tla_core::ast::{BoundVar, Expr, Module, Unit};
use tla_core::Spanned;
use tla_mc_core::{
    BackendKind, PreparedAnalyticalSolveKind, PreparedBackendFamilyDescriptor,
    PreparedCandidateLaneDescriptor, PreparedCheckerProgram, PreparedFingerprintDescriptor,
    PreparedFingerprintScheme, PreparedProgramPayloadKind, PreparedStorageKind,
    PreparedSymbolicProofKind, PreparedValidationKind, PreparedValidationPlanDescriptor,
    ProblemKind, SetupTraceLaneKind, SharedEngineAdoptionEvidence,
    SharedEngineAdoptionFamilyBlocker, SharedEngineFrontendFamily, SolverFacet,
};

use crate::check::ResolvedSpec;
use crate::config::Config;
use crate::eval::EvalCtx;

const AY_SHARED_ENGINE_ROUTING_SCHEMA: &str = "tla-check.ay-shared-engine-routing/v1";
const AY_ANALYTICAL_PROOF_ADOPTION_OWNER: &str = "shared_ay_engine";
const AY_ANALYTICAL_PROOF_ADOPTION_ACCEPTANCE_TEST: &str =
    "cargo_test_-p_tla-check_--lib_ay_lane_admission_policy_analytical_proof_adoption_evidence_is_release_critical_and_frontend_neutral";
const AY_ANALYTICAL_PROOF_FUTURE_IMPORTER_BLOCKER: &str =
    "awaiting_registered_importer_payload_identity_layout_fingerprints_validation_receipts";
const AY_SHARED_ENGINE_PORTFOLIO_ADMISSION_POLICY: &str =
    "admit_only_when_candidate_fingerprint_validation_and_lane_prerequisites_pass";
const AY_SHARED_ENGINE_PORTFOLIO_WINNER_POLICY: &str =
    "lowest_rank_admitted_ay_solve_lane_can_win_after_validation_else_explicit_search_remains";
const AY_SHARED_ENGINE_EXPLICIT_SEARCH_REPLACEMENT_POLICY: &str =
    "replace_explicit_search_only_for_admitted_receipt_backed_ay_solve_lanes";
const AY_CACHE_FINGERPRINT_COMPATIBILITY_FRONTEND_REUSABLE: &str = "frontend_reusable";
const AY_CACHE_FINGERPRINT_COMPATIBILITY_NOT_DECLARED: &str = "not_declared";
const AY_CACHE_FINGERPRINT_CAPABILITY_PREFIX: &str = "cache_fingerprint_compatibility:";

/// Clone an EvalCtx and bind TLC config constants/operator replacements into it.
///
/// Symbolic entrypoints must do this themselves so direct `check_bmc` /
/// `check_pdr` calls honor config-provided values and operator replacements
/// without requiring the caller to pre-mutate the shared EvalCtx.
pub(crate) fn symbolic_ctx_with_config(ctx: &EvalCtx, config: &Config) -> Result<EvalCtx, String> {
    let mut ctx = ctx.clone();
    crate::bind_constants_from_config(&mut ctx, config)
        .map_err(|error| format!("failed to bind config constants: {error}"))?;
    Ok(ctx)
}

/// Collect state variable names for the symbolic engines.
///
/// SOURCE OF TRUTH: the `EvalCtx` variable registry, when it is populated. The
/// checker registers the FULL state-variable set — i.e. the EXTENDS-inherited
/// closure, not just the top module's own `VARIABLES` — into `ctx.var_registry()`
/// at setup time (`checker_setup::setup_checker_modules` and the symbolic-lane
/// construction sites, which now call `register_state_vars_from_modules` over
/// the module + `checker_modules`). Reading the registry is therefore how the
/// checker resolves inherited declarations, and it is the only way the symbolic
/// engines see the real `VARIABLES` of an MC-wrapper spec (`MCFoo.tla` does
/// `EXTENDS Foo` and the declarations live in `Foo.tla`).
///
/// FALLBACK: a bare top-module scan, used only when the registry is empty —
/// the case for direct test callers that build a `Module` via `lower()` and a
/// fresh `EvalCtx` without going through checker setup. Single-module specs are
/// unaffected: the registry (when populated) holds exactly the module's own
/// `VARIABLES` in the same order, and the scan is the identical list when it
/// isn't. Declaration/registration order is preserved; the registry is already
/// deduped, and the scan dedups against what it has seen.
pub(crate) fn collect_state_vars(module: &Module, ctx: &EvalCtx) -> Vec<Arc<str>> {
    let registry = ctx.var_registry();
    if !registry.is_empty() {
        return registry.names().to_vec();
    }

    // Registry not populated (bare ctx / direct test callers): fall back to a
    // top-module-only scan. This is the historical behavior and is correct for
    // single-module specs; MC-wrapper specs always go through a construction
    // site that registers the EXTENDS closure, so they take the branch above.
    let mut vars = Vec::new();
    for unit in &module.units {
        if let Unit::Variable(var_names) = &unit.node {
            for var in var_names {
                if !vars
                    .iter()
                    .any(|v: &Arc<str>| v.as_ref() == var.node.as_str())
                {
                    vars.push(Arc::from(var.node.as_str()));
                }
            }
        }
    }
    vars
}

/// Scan `VARIABLES` declarations from the main module and its EXTENDS-inherited
/// `checker_modules`, in declaration order with dedup (extended modules first,
/// matching TLC's declaration order and `validate_spec::collect_state_vars`).
///
/// This is the bridge that makes MC-wrapper specs symbolically checkable: the
/// symbolic-lane construction sites build an ad-hoc `EvalCtx` with only the top
/// module's units, so the real `VARIABLES` (declared in the EXTENDS-inherited
/// base module) are invisible. Feeding this set into `EvalCtx::register_vars`
/// populates `ctx.var_registry()`, which `collect_state_vars` then reads as the
/// source of truth.
///
/// ONLY modules in the transitive EXTENDS closure contribute variables —
/// mirroring the BFS checker's registry (`setup_imports::compute_import_sets`'s
/// `variable_contributing_modules`). `checker_modules` also carries
/// named-INSTANCE modules (loaded for `I!Op` resolution); their `VARIABLES`
/// are substituted at instantiation and must NOT enter the symbolic state
/// space. Registering them froze phantom variables in symbolic traces (the
/// alias-INSTANCE regression: `x`/`derived` from a named-INSTANCE `Inner`
/// shadowing the outer module's operators of the same names).
pub(crate) fn state_vars_in_module_closure(
    module: &Module,
    checker_modules: &[&Module],
) -> Vec<Arc<str>> {
    let mut vars: Vec<Arc<str>> = Vec::new();
    let push_unique = |name: &str, vars: &mut Vec<Arc<str>>| {
        if !vars.iter().any(|v: &Arc<str>| v.as_ref() == name) {
            vars.push(Arc::from(name));
        }
    };

    // Transitive EXTENDS-only closure over the loaded modules (INSTANCE edges
    // are deliberately not followed — BUG FIX #295 parity with the BFS setup).
    let by_name: std::collections::HashMap<&str, &Module> = checker_modules
        .iter()
        .map(|m| (m.name.node.as_str(), *m))
        .collect();
    let mut contributing: std::collections::HashSet<&str> = std::collections::HashSet::new();
    let mut stack: Vec<&str> = module.extends.iter().map(|s| s.node.as_str()).collect();
    while let Some(name) = stack.pop() {
        if !contributing.insert(name) {
            continue;
        }
        if let Some(m) = by_name.get(name) {
            stack.extend(m.extends.iter().map(|s| s.node.as_str()));
        }
    }

    // Extended (EXTENDS-inherited) modules first — TLC declaration order.
    for ext_mod in checker_modules {
        if !contributing.contains(ext_mod.name.node.as_str()) {
            continue;
        }
        for unit in &ext_mod.units {
            if let Unit::Variable(var_names) = &unit.node {
                for var in var_names {
                    push_unique(var.node.as_str(), &mut vars);
                }
            }
        }
    }
    // Then the top (MC-wrapper) module's own VARIABLES, if any.
    for unit in &module.units {
        if let Unit::Variable(var_names) = &unit.node {
            for var in var_names {
                push_unique(var.node.as_str(), &mut vars);
            }
        }
    }
    vars
}

/// Register the full EXTENDS-closure state-variable set into `ctx`'s registry so
/// the symbolic engines (which only see the top module) discover inherited
/// `VARIABLES`. Idempotent against an already-populated registry: variables that
/// are already registered are skipped by `VarRegistry::register`.
pub(crate) fn register_state_vars_from_modules(
    ctx: &mut EvalCtx,
    module: &Module,
    checker_modules: &[&Module],
) {
    let vars = state_vars_in_module_closure(module, checker_modules);
    ctx.register_vars(vars);
}

/// Resolve Init/Next operator names from config after constant/operator
/// replacements have been bound into the evaluation context.
///
/// Returns a `ResolvedSpec` or an error string for the missing part.
pub(crate) fn resolve_init_next(config: &Config, ctx: &EvalCtx) -> Result<ResolvedSpec, String> {
    match (&config.init, &config.next) {
        (Some(init), Some(next)) => Ok(ResolvedSpec {
            init: ctx.resolve_op_name(init).to_string(),
            next: ctx.resolve_op_name(next).to_string(),
            next_node: None,
            fairness: Vec::new(),
            stuttering_allowed: true,
        }),
        (None, _) => Err("INIT not specified".to_string()),
        (_, None) => Err("NEXT not specified".to_string()),
    }
}

/// Get operator body expression from EvalCtx.
///
/// Returns the operator body or an error string.
/// Byte offset of the start of the FIRST module's terminator line (`====…`) at
/// comment-depth 0, for injecting a certification sentinel operator just before
/// it (inside exactly the module `tla_core::lower` binds — the first).
///
/// Robust to both E4 edges: it skips terminator-looking text inside `(* … *)`
/// block comments and `\* …` line comments (the naive `find("\n====")` would
/// mis-anchor on a `====` line inside a header block comment), and it targets the
/// FIRST real terminator (the naive `rfind` picked the LAST module's, which lower
/// does not bind). Returns `None` if no comment-free terminator line exists
/// (the caller then declines fail-closed). The op is placed just before this
/// offset, so a mis-anchor only ever yields an honest `NotRederivable` decline,
/// never a wrong-J certificate.
pub(crate) fn first_module_terminator_pos(src: &str) -> Option<usize> {
    crate::check::eval_oracle::first_module_terminator_pos(src)
}

pub(crate) fn get_operator_body(ctx: &EvalCtx, name: &str) -> Result<Spanned<Expr>, String> {
    if let Some(def) = ctx.get_op(name) {
        if !def.params.is_empty() {
            return Err(format!(
                "Operator '{}' has parameters; symbolic checking requires parameterless Init/Next",
                name
            ));
        }
        Ok(def.body.clone())
    } else {
        Err(format!("Operator '{}' not found", name))
    }
}

/// Whether a symbolic SAFETY proof covers ALL of this run's proof obligations.
///
/// The symbolic safety lanes (PDR/IC3 and k-induction) prove ONLY the invariant
/// conjunction (`config.invariants`). They do not encode deadlock-freedom, and they
/// do not check liveness / temporal properties or trace invariants. So a symbolic
/// `Verdict::Satisfied` is authoritative for the WHOLE run only when the safety
/// invariants are the run's sole obligation.
///
/// SOUNDNESS (the symmetric dual of `reconcile_masked_violation`): when other
/// obligations exist (deadlock-checking on — the default — or PROPERTIES / trace
/// invariants / an unresolved temporal SPECIFICATION), a symbolic `Satisfied` must NOT
/// resolve the cooperative verdict, because resolving it makes the BFS lane exit early
/// and return an indistinguishable `CheckResult::Success` — silently masking a reachable
/// deadlock or a liveness violation that the (now-truncated) BFS lane would have caught.
/// In that case the symbolic proof is still recorded (lemmas + `set_invariants_proved`
/// for cross-validation), but BFS is left to run to completion and stays authoritative
/// for the obligations the symbolic lane did not verify.
///
/// Uses [`Config::has_liveness_properties`] — TY's canonical liveness predicate — so the
/// gate tracks the SAME signal the rest of the checker uses to decide whether to run a
/// liveness check. `normalize_resolved_specification` moves a resolved SPECIFICATION's
/// safety part into init/next and clears `specification`, leaving any temporal goal in
/// `properties`; an UNRESOLVED SPECIFICATION is gated by the explicit `specification`
/// check below.
///
/// CONSTRAINT / action constraints are deliberately NOT gated: they only RESTRICT the
/// reachable set, and a safety proof over the unconstrained transition relation (a
/// superset) implies safety over the constrained subset — so a symbolic `Satisfied`
/// proven without the constraint is never WEAKER than a constrained BFS would conclude
/// (at worst over-conservative, never a false PASS).
pub(crate) fn symbolic_safety_proof_covers_all_obligations(config: &Config) -> bool {
    !config.check_deadlock
        && !config.has_liveness_properties()
        && config.trace_invariants.is_empty()
        && config.specification.is_none()
}

/// Build conjunction of invariant expressions.
pub(crate) fn build_safety_conjunction(
    ctx: &EvalCtx,
    invariant_names: &[String],
) -> Result<Spanned<Expr>, String> {
    let mut exprs: Vec<Spanned<Expr>> = Vec::new();

    for name in invariant_names {
        let resolved_name = ctx.resolve_op_name(name);
        let body = get_operator_body(ctx, resolved_name)?;
        exprs.push(body);
    }

    if exprs.is_empty() {
        return Err("No invariants configured".to_string());
    }

    let mut result = exprs.pop().expect("exprs non-empty after empty check");
    while let Some(expr) = exprs.pop() {
        result = Spanned::dummy(Expr::And(Box::new(expr), Box::new(result)));
    }

    Ok(result)
}

thread_local! {
    /// The set of unbound (symbolic) `CONSTANT` names in scope for the CURRENT
    /// sort-inference call. Read ONLY by the `FuncSet` arm of
    /// [`infer_sort_from_set_expr`] to recognise a symbolic contiguous domain
    /// `lo..(N+offset)` as a [`TlaSort::FunctionSym`]. Defaults to empty (so
    /// every ordinary caller of [`infer_var_sorts`] is unaffected); populated
    /// only for the duration of an [`infer_var_sorts_with_symbolic`] call.
    static SYMBOLIC_CONSTANTS: RefCell<HashSet<String>> = RefCell::new(HashSet::new());
}

/// Like [`infer_var_sorts`], but with a set of SYMBOLIC (unbound) `CONSTANT`
/// names in scope. A function-set membership `f \in [lo..N -> T]` (or
/// `[lo..N-1 -> T]`) whose upper bound references such a constant is inferred as
/// [`TlaSort::FunctionSym`] (a map-only symbolic-domain function) rather than a
/// finite [`TlaSort::Function`]. Used by the all-`N` certification lane.
///
/// SOUNDNESS: over-recognising a domain as symbolic is completeness-only — a
/// symbolic-domain function forces the pointwise-∀ discipline, and if that
/// discipline cannot discharge the obligation the query stays SAT and the lane
/// declines. It can never turn a SAT query into a false accept.
pub(crate) fn infer_var_sorts_with_symbolic(
    vars: &[Arc<str>],
    init_expr: &Spanned<Expr>,
    invariant_names: &[String],
    ctx: &EvalCtx,
    symbolic: &HashSet<String>,
) -> Vec<(String, TlaSort)> {
    SYMBOLIC_CONSTANTS.with(|s| *s.borrow_mut() = symbolic.clone());
    let out = infer_var_sorts(vars, init_expr, invariant_names, ctx);
    SYMBOLIC_CONSTANTS.with(|s| s.borrow_mut().clear());
    out
}

/// Parse a symbolic upper bound expression into `(constant_name, affine_offset)`
/// where the bound denotes `constant_name + offset`. Recognises `N` (offset 0),
/// `N - k` (offset `-k`), and `N + k` (offset `+k`) for a symbolic `N`. Returns
/// `None` for any other shape or a non-symbolic base.
fn parse_symbolic_upper_bound(hi: &Spanned<Expr>, symbolic: &HashSet<String>) -> Option<(String, i64)> {
    let ident_if_symbolic = |e: &Spanned<Expr>| -> Option<String> {
        match &e.node {
            Expr::Ident(name, _) | Expr::StateVar(name, ..) if symbolic.contains(name) => {
                Some(name.clone())
            }
            _ => None,
        }
    };
    let int_lit = |e: &Spanned<Expr>| -> Option<i64> {
        match &e.node {
            Expr::Int(n) => n.try_into().ok(),
            Expr::Neg(inner) => match &inner.node {
                Expr::Int(n) => i64::try_from(n).ok().map(|v| -v),
                _ => None,
            },
            _ => None,
        }
    };
    match &hi.node {
        // bare `N`
        _ if ident_if_symbolic(hi).is_some() => ident_if_symbolic(hi).map(|n| (n, 0)),
        // `N - k`
        Expr::Sub(a, b) => {
            let name = ident_if_symbolic(a)?;
            let k = int_lit(b)?;
            Some((name, -k))
        }
        // `N + k`
        Expr::Add(a, b) => {
            if let Some(name) = ident_if_symbolic(a) {
                let k = int_lit(b)?;
                Some((name, k))
            } else if let Some(name) = ident_if_symbolic(b) {
                let k = int_lit(a)?;
                Some((name, k))
            } else {
                None
            }
        }
        _ => None,
    }
}

/// Recognise `[lo..hi -> R]` with a LITERAL `lo`, a SYMBOLIC-constant `hi`
/// (`N` / `N-1` / `N+1`), and a SCALAR range sort as a [`TlaSort::FunctionSym`].
/// Returns `None` when the domain is not a symbolic contiguous range or the
/// codomain is non-scalar (both cases fall back to the finite path / decline).
fn recognize_symbolic_func_domain(
    domain: &Spanned<Expr>,
    range: &Spanned<Expr>,
    symbolic: &HashSet<String>,
) -> Option<TlaSort> {
    let Expr::Range(lo, hi) = &domain.node else {
        return None;
    };
    let lo_val: i64 = match &lo.node {
        Expr::Int(n) => n.try_into().ok()?,
        _ => return None,
    };
    let (hi_const, hi_offset) = parse_symbolic_upper_bound(hi, symbolic)?;
    // Codomain must be scalar: this SINGLE guard excludes set / sequence /
    // (nested) function codomains, which would need array extensionality
    // (fail-closed at strict re-check). Enum literals intern to Int (scalar).
    let range_sort = infer_sort_from_set_expr(range)?;
    if !range_sort.is_scalar() {
        return None;
    }
    Some(TlaSort::FunctionSym {
        domain_lo: lo_val,
        domain_hi_const: hi_const,
        domain_hi_offset: hi_offset,
        range: Box::new(range_sort),
    })
}

/// Infer variable sorts from TypeOK invariant or Init constraints.
///
/// Inference strategy:
/// 1. Look for TypeOK in invariants and check for `x \in BOOLEAN` or `x \in Int`
/// 2. Fall back to Init constraints of the form `x = TRUE/FALSE` (Bool) or `x = n` (Int)
/// 3. Default to Int if ambiguous (most TLA+ specs use integers)
pub(crate) fn infer_var_sorts(
    vars: &[Arc<str>],
    init_expr: &Spanned<Expr>,
    invariant_names: &[String],
    ctx: &EvalCtx,
) -> Vec<(String, TlaSort)> {
    let mut sorts: Vec<(String, TlaSort)> = Vec::new();

    let type_ok_body = invariant_names.iter().find_map(|name| {
        let resolved_name = ctx.resolve_op_name(name);
        let lower = resolved_name.to_lowercase();
        if !(lower == "typeok" || lower == "typeinvariant" || lower == "type_ok") {
            return None;
        }
        ctx.get_op(resolved_name).map(|def| &def.body)
    });

    for var in vars {
        let sort = infer_single_var_sort(var, init_expr, type_ok_body);
        sorts.push((var.to_string(), sort));
    }

    sorts
}

fn infer_single_var_sort(
    var: &str,
    init_expr: &Spanned<Expr>,
    type_ok_body: Option<&Spanned<Expr>>,
) -> TlaSort {
    if let Some(body) = type_ok_body {
        if let Some(sort) = check_membership_constraint(var, body) {
            return sort;
        }
    }

    if let Some(sort) = check_init_assignment(var, init_expr) {
        return sort;
    }

    TlaSort::Int
}

/// Check if expression contains `var \in S` constraint that reveals the variable sort.
pub(crate) fn check_membership_constraint(var: &str, expr: &Spanned<Expr>) -> Option<TlaSort> {
    match &expr.node {
        Expr::And(a, b) | Expr::Or(a, b) | Expr::Implies(a, b) | Expr::Equiv(a, b) => {
            check_membership_constraint(var, a).or_else(|| check_membership_constraint(var, b))
        }
        Expr::Let(_, body) => check_membership_constraint(var, body),
        Expr::Label(label) => check_membership_constraint(var, &label.body),
        Expr::In(elem, set) => {
            let is_var = matches!(&elem.node, Expr::Ident(name, _) | Expr::StateVar(name, _, _) if name == var);
            if !is_var {
                return None;
            }
            infer_sort_from_set_expr(set)
        }
        _ => None,
    }
}

/// Check Init for assignment or membership patterns that reveal the variable sort.
pub(crate) fn check_init_assignment(var: &str, expr: &Spanned<Expr>) -> Option<TlaSort> {
    match &expr.node {
        Expr::And(a, b) | Expr::Or(a, b) => {
            check_init_assignment(var, a).or_else(|| check_init_assignment(var, b))
        }
        Expr::Let(_, body) => check_init_assignment(var, body),
        Expr::Label(label) => check_init_assignment(var, &label.body),
        Expr::Eq(lhs, rhs) => {
            let is_var = matches!(&lhs.node, Expr::Ident(name, _) | Expr::StateVar(name, _, _) if name == var);
            if !is_var {
                return None;
            }
            infer_sort_from_value_expr(rhs).or_else(|| infer_sort_from_set_expr(rhs))
        }
        Expr::In(elem, set) => {
            let is_var = matches!(&elem.node, Expr::Ident(name, _) | Expr::StateVar(name, _, _) if name == var);
            if !is_var {
                return None;
            }
            infer_sort_from_set_expr(set)
        }
        _ => None,
    }
}

fn infer_sort_from_value_expr(expr: &Spanned<Expr>) -> Option<TlaSort> {
    match &expr.node {
        Expr::Label(label) => infer_sort_from_value_expr(&label.body),
        Expr::Bool(_) => Some(TlaSort::Bool),
        Expr::Int(_) => Some(TlaSort::Int),
        Expr::String(_) => Some(TlaSort::String),
        // Part of #3749: set literal as a value — infer Set sort from elements.
        Expr::SetEnum(elements) => {
            let element_sort = infer_sort_from_set_enum(elements)?;
            Some(TlaSort::Set {
                element_sort: Box::new(element_sort),
            })
        }
        // Part of #3749: tuple literal as a sequence value candidate.
        Expr::Tuple(elements) if !elements.is_empty() => {
            // Try to infer a uniform element sort (for sequences).
            // If elements have mixed sorts, fall back to tuple sort.
            let first_sort = infer_sort_from_value_expr(&elements[0])?;
            let all_same = elements
                .iter()
                .skip(1)
                .all(|e| infer_sort_from_value_expr(e).as_ref() == Some(&first_sort));
            if all_same {
                // Could be either a tuple or a sequence — conservatively return Tuple.
                let element_sorts = elements
                    .iter()
                    .map(|e| infer_sort_from_value_expr(e))
                    .collect::<Option<Vec<_>>>()?;
                Some(TlaSort::Tuple { element_sorts })
            } else {
                let element_sorts = elements
                    .iter()
                    .map(|e| infer_sort_from_value_expr(e))
                    .collect::<Option<Vec<_>>>()?;
                Some(TlaSort::Tuple { element_sorts })
            }
        }
        Expr::Record(fields) => {
            let mut field_sorts = Vec::with_capacity(fields.len());
            for (field_name, field_expr) in fields {
                field_sorts.push((
                    field_name.node.clone(),
                    infer_sort_from_value_expr(field_expr)?,
                ));
            }
            Some((TlaSort::Record { field_sorts }).canonicalized())
        }
        Expr::RecordAccess(base, field) => {
            let TlaSort::Record { field_sorts } = infer_sort_from_value_expr(base)? else {
                return None;
            };
            field_sorts
                .into_iter()
                .find(|(name, _)| name == &field.name.node)
                .map(|(_, sort)| sort)
        }
        Expr::FuncDef(bounds, body) => {
            if bounds.len() != 1 {
                return None;
            }
            let domain = bounds[0].domain.as_ref()?;
            let domain_keys = extract_finite_domain_keys(domain)?;
            let range_sort = infer_sort_from_value_expr(body)?;
            Some(
                (TlaSort::Function {
                    domain_keys,
                    range: Box::new(range_sort),
                })
                .canonicalized(),
            )
        }
        Expr::Except(base, _) => infer_sort_from_value_expr(base),
        // Unary minus: `-x` has the same (numeric) sort as `x`. Without this arm
        // a function range body like `-1` (`Expr::Neg(Int(1))`) returns None, so
        // the whole function variable mis-sorts to scalar Int and symbolic
        // translation later fails with "expected Bool, got Int" — breaking many
        // function-heavy specs (e.g. Paxos's `maxBal = [a \in Acceptor |-> -1]`).
        Expr::Neg(inner) => infer_sort_from_value_expr(inner),
        Expr::If(_, then_branch, else_branch) => {
            let then_sort = infer_sort_from_value_expr(then_branch)?;
            let else_sort = infer_sort_from_value_expr(else_branch)?;
            (then_sort == else_sort).then_some(then_sort)
        }
        _ => None,
    }
}

fn infer_sort_from_set_expr(expr: &Spanned<Expr>) -> Option<TlaSort> {
    match &expr.node {
        Expr::Label(label) => infer_sort_from_set_expr(&label.body),
        Expr::Ident(name, _) if name == "BOOLEAN" => Some(TlaSort::Bool),
        Expr::Ident(name, _) if name == "Int" || name == "Nat" => Some(TlaSort::Int),
        Expr::Range(_, _) => Some(TlaSort::Int),
        Expr::SetEnum(elements) => infer_sort_from_set_enum(elements),
        // A filtered set `{c \in S : P(c)}` has the SAME element sort as its
        // domain `S` (the predicate only removes elements). This lets a state var
        // whose only structural constraint is `v \in {c \in RecordSet : …}` (e.g.
        // CoffeeCan's `Init == can \in {c \in Can : c.black + c.white \in 1..M}`)
        // still be typed as the record — without it `v` falls back to `Int` and a
        // `v.field` access is untranslatable.
        Expr::SetFilter(bv, _pred) => {
            bv.domain.as_ref().and_then(|d| infer_sort_from_set_expr(d))
        }
        Expr::RecordSet(fields) => {
            let mut field_sorts = Vec::with_capacity(fields.len());
            for (field_name, domain) in fields {
                field_sorts.push((field_name.node.clone(), infer_sort_from_set_expr(domain)?));
            }
            Some((TlaSort::Record { field_sorts }).canonicalized())
        }
        Expr::FuncSet(domain, range) => {
            // Symbolic contiguous domain `lo..(N+offset)` (all-N lane): recognise
            // it as a map-only FunctionSym before trying the finite path.
            if let Some(fsym) = SYMBOLIC_CONSTANTS.with(|s| {
                let sym = s.borrow();
                if sym.is_empty() {
                    None
                } else {
                    recognize_symbolic_func_domain(domain, range, &sym)
                }
            }) {
                return Some(fsym.canonicalized());
            }
            Some(
                (TlaSort::Function {
                    domain_keys: extract_finite_domain_keys(domain)?,
                    range: Box::new(infer_sort_from_set_expr(range)?),
                })
                .canonicalized(),
            )
        }
        Expr::SetFilter(bound, _) => infer_sort_from_set_expr(bound.domain.as_ref()?),
        // FuncSet-as-sequence builder: `{FunAsSeq(p, n, m) : p \in [1..n -> R]}`
        // (the Permutation idiom). A `FunAsSeq` over a function with a
        // contiguous `1..n` domain is a length-n sequence whose elements come
        // from R, so the variable bound to a member is a `Tuple` of n copies of
        // R's element sort. Recognising this lets specs like Einstein declare
        // `drinks`/`colors`/... as tuples-of-strings rather than mis-sorting to
        // scalar Int. Soundness: this only fixes the *declared sort*; the
        // membership constraint is still translated exactly downstream.
        Expr::SetBuilder(body, bounds) => infer_funasseq_builder_sort(body, bounds),
        // Part of #3749: SUBSET S — variable is a set of element_sort
        Expr::Powerset(inner) => {
            let element_sort = infer_sort_from_set_expr(inner)?;
            Some(TlaSort::Set {
                element_sort: Box::new(element_sort),
            })
        }
        // Part of #3749: Seq(S) — variable is a sequence of element_sort.
        // Seq(S) is lowered to Apply(OpRef("Seq"), [S]).
        Expr::Apply(op, args) if args.len() == 1 => {
            let is_seq = matches!(
                &op.node,
                Expr::Ident(name, _) | Expr::OpRef(name) if name == "Seq"
            );
            if is_seq {
                let element_sort = infer_sort_from_set_expr(&args[0])?;
                // Default max_len of 10 for finite model checking;
                // the caller can override if known.
                Some(TlaSort::Sequence {
                    element_sort: Box::new(element_sort),
                    max_len: 10,
                })
            } else {
                None
            }
        }
        _ => None,
    }
}

/// Peel `LET ... IN body` and `Label` wrappers, returning the innermost
/// expression. Operator expansion can wrap an inlined body in a `LET` that
/// binds (now-substituted) locals; for structural recognition we look through
/// them.
fn peel_wrappers(expr: &Spanned<Expr>) -> &Spanned<Expr> {
    let mut cur = expr;
    loop {
        match &cur.node {
            Expr::Let(_, body) => cur = body,
            Expr::Label(label) => cur = &label.body,
            _ => return cur,
        }
    }
}

/// Infer the sort of a member of `{FunAsSeq(p, n, m) : p \in D}` where `D`
/// (possibly under a `SetFilter`) is a function set `[1..n -> R]`.
///
/// Returns a `Tuple` sort with `n` elements, each the element sort of `R`.
/// Returns `None` when the builder is not the recognised `FunAsSeq` idiom.
fn infer_funasseq_builder_sort(body: &Spanned<Expr>, bounds: &[BoundVar]) -> Option<TlaSort> {
    // Peel any LET / Label wrappers the operator-expansion pass introduced
    // around the `FunAsSeq` application.
    let body = peel_wrappers(body);
    // Body must be a sequence-construction over the bound function: either the
    // raw `FunAsSeq(p, _, _)` or its lowered form `SubSeq(MkSeq(...), 1, n)`
    // (the Apalache module lowers `FunAsSeq` to `SubSeq`). Either way the
    // *member sort* is fully determined by the bound's function set below, so we
    // only need to confirm the body is one of these sequence builders.
    if !is_sequence_builder_body(body) {
        return None;
    }

    // Single bound `p \in D`.
    let bound = bounds.first()?;
    if bounds.len() != 1 {
        return None;
    }
    let mut domain = bound.domain.as_ref()?;
    // Peel any SetFilter wrapping the function set (the distinctness predicate).
    if let Expr::SetFilter(inner_bound, _) = &domain.node {
        domain = inner_bound.domain.as_ref()?;
    }

    let Expr::FuncSet(func_domain, func_range) = &domain.node else {
        return None;
    };

    // Function domain must be a finite, contiguous `1..n` range so the function
    // is a genuine sequence (FunAsSeq semantics).
    let n = match &func_domain.node {
        Expr::Range(lo, hi) => {
            let lo: i64 = match &lo.node {
                Expr::Int(v) => v.try_into().ok()?,
                _ => return None,
            };
            let hi: i64 = match &hi.node {
                Expr::Int(v) => v.try_into().ok()?,
                _ => return None,
            };
            if lo != 1 || hi < 1 {
                return None;
            }
            usize::try_from(hi).ok()?
        }
        _ => return None,
    };

    let element_sort = infer_sort_from_set_expr(func_range)?;
    let element_sorts = vec![element_sort; n];
    Some((TlaSort::Tuple { element_sorts }).canonicalized())
}

/// Is `body` a sequence-construction expression — `FunAsSeq(...)` or its
/// lowered `SubSeq(MkSeq(...), ...)` / `SubSeq([i \in 1..n |-> ...], ...)`?
fn is_sequence_builder_body(body: &Spanned<Expr>) -> bool {
    if let Expr::Apply(op, _) = &body.node {
        if let Expr::Ident(name, _) | Expr::OpRef(name) = &op.node {
            return name == "FunAsSeq" || name == "SubSeq" || name == "MkSeq";
        }
    }
    false
}

fn infer_sort_from_set_enum(elements: &[Spanned<Expr>]) -> Option<TlaSort> {
    let first = elements.first()?;
    let first_sort = infer_sort_from_value_expr(first)?;
    if elements
        .iter()
        .skip(1)
        .all(|expr| infer_sort_from_value_expr(expr).as_ref() == Some(&first_sort))
    {
        Some(first_sort)
    } else {
        None
    }
}

fn encode_domain_key_expr(expr: &Spanned<Expr>) -> Option<String> {
    match &expr.node {
        Expr::Int(n) => Some(format!("int:{n}")),
        Expr::Bool(b) => Some(format!("bool:{b}")),
        Expr::String(s) => Some(format!("str:{s}")),
        Expr::Ident(name, _) => Some(format!("id:{name}")),
        _ => None,
    }
}

fn extract_finite_domain_keys(domain: &Spanned<Expr>) -> Option<Vec<String>> {
    match &domain.node {
        Expr::SetEnum(elements) => elements.iter().map(encode_domain_key_expr).collect(),
        Expr::Range(lo, hi) => {
            let lo: i64 = match &lo.node {
                Expr::Int(n) => n.try_into().ok()?,
                _ => return None,
            };
            let hi: i64 = match &hi.node {
                Expr::Int(n) => n.try_into().ok()?,
                _ => return None,
            };
            if hi < lo {
                return Some(Vec::new());
            }
            Some((lo..=hi).map(|n| format!("int:{n}")).collect())
        }
        Expr::Ident(name, _) if name == "BOOLEAN" => {
            Some(vec!["bool:false".to_string(), "bool:true".to_string()])
        }
        _ => None,
    }
}

/// Check if any variable sort requires array-based SMT encoding.
///
/// Sets, Functions, and Sequences are encoded as SMT arrays, which
/// requires `QF_AUFLIA` logic instead of `QF_LIA`.
///
/// Extracted from `ay_bmc.rs` for reuse in k-induction, symbolic sim,
/// and inductive check entrypoints.
pub(crate) fn needs_array_logic(var_sorts: &[(String, TlaSort)]) -> bool {
    var_sorts.iter().any(|(_, sort)| {
        matches!(
            sort,
            TlaSort::Set { .. }
                | TlaSort::Function { .. }
                | TlaSort::FunctionSym { .. }
                | TlaSort::Sequence { .. }
        )
    })
}

/// Create a BMC translator with the appropriate logic for the given variable sorts.
///
/// Uses `QF_AUFLIA` if any variable requires array encoding (sets, functions,
/// sequences), otherwise `QF_LIA`.
///
/// Extracted from `ay_bmc.rs` for reuse in k-induction, symbolic sim,
/// and inductive check entrypoints.
///
/// native-seq decision (sequence vars): sequence-typed vars stay on the bounded
/// `(Array Int Elem) + len` encoding (`new_with_arrays`, `len <= max_len`), NOT
/// the native unbounded `Sort::Seq` path. `BmcTranslator` has no native seq path,
/// and — per `docs/ay-native-seq-feasibility-2026-06-06.md` — flipping BMC to
/// native unbounded sequences is gated on a deep-unroll solver-robustness
/// measurement (the `Unknown`/timeout rate across k≈10–30 with co-present Set/Func
/// vars) that determines whether native BMC preserves the "no CEX up to k"
/// guarantee. That gate is not run here, so bounded BMC remains the default: its
/// failure mode (miss a CEX needing a sequence longer than `max_len`) is the SAFE,
/// sound-under-approximation direction, whereas a native-BMC `Unknown` would
/// silently break the up-to-k guarantee. Native unbounded sequences are available
/// today only on the opt-in `AYTranslator::new_with_seq` symbolic path.
pub(crate) fn make_translator(
    var_sorts: &[(String, TlaSort)],
    depth: usize,
) -> Result<tla_ay::BmcTranslator, tla_ay::AYError> {
    if needs_array_logic(var_sorts) {
        tla_ay::BmcTranslator::new_with_arrays(depth)
    } else {
        tla_ay::BmcTranslator::new(depth)
    }
}

/// Stable candidate key for a frontend-neutral AY shared-engine lane.
pub(crate) fn ay_shared_engine_candidate_key(lane: AYSharedEngineLane) -> &'static str {
    match lane {
        AYSharedEngineLane::AllSatEnumeration => "ay_all_sat_enumeration",
        AYSharedEngineLane::Bmc => "ay_bmc",
        AYSharedEngineLane::Chc => "ay_chc",
        AYSharedEngineLane::Pdr => "ay_pdr",
        AYSharedEngineLane::KInduction => "ay_k_induction",
    }
}

/// Map a shared AY lane to the prepared analytical solve kind it can discharge.
pub(crate) fn ay_shared_engine_prepared_solve_kind(
    lane: AYSharedEngineLane,
) -> Option<PreparedAnalyticalSolveKind> {
    match lane {
        AYSharedEngineLane::AllSatEnumeration => Some(PreparedAnalyticalSolveKind::SmtQuery),
        AYSharedEngineLane::Bmc => Some(PreparedAnalyticalSolveKind::BoundedModelCheck),
        AYSharedEngineLane::Chc => None,
        AYSharedEngineLane::Pdr => Some(PreparedAnalyticalSolveKind::PdrSafety),
        AYSharedEngineLane::KInduction => Some(PreparedAnalyticalSolveKind::KInduction),
    }
}

/// Map a prepared analytical solve kind back to its AY shared-engine lane.
pub(crate) fn ay_shared_engine_lane_for_prepared_solve_kind(
    kind: PreparedAnalyticalSolveKind,
) -> Option<AYSharedEngineLane> {
    match kind {
        PreparedAnalyticalSolveKind::SmtQuery | PreparedAnalyticalSolveKind::SatQuery => {
            Some(AYSharedEngineLane::AllSatEnumeration)
        }
        PreparedAnalyticalSolveKind::BoundedModelCheck => Some(AYSharedEngineLane::Bmc),
        PreparedAnalyticalSolveKind::PdrSafety => Some(AYSharedEngineLane::Pdr),
        PreparedAnalyticalSolveKind::KInduction => Some(AYSharedEngineLane::KInduction),
        PreparedAnalyticalSolveKind::StateSpaceCardinality
        | PreparedAnalyticalSolveKind::DeadlockFreedom
        | PreparedAnalyticalSolveKind::Reachability
        | PreparedAnalyticalSolveKind::StableMarking
        | PreparedAnalyticalSolveKind::UpperBounds
        | PreparedAnalyticalSolveKind::LinearInvariant => None,
    }
}

/// Prepared problem class for one AY shared-engine lane.
pub(crate) fn ay_shared_engine_problem(lane: AYSharedEngineLane) -> ProblemKind {
    match lane {
        AYSharedEngineLane::AllSatEnumeration => ProblemKind::SymbolicExecution,
        AYSharedEngineLane::Bmc => ProblemKind::Bmc,
        AYSharedEngineLane::Chc | AYSharedEngineLane::Pdr => ProblemKind::Chc,
        AYSharedEngineLane::KInduction => ProblemKind::KInduction,
    }
}

/// Prepared symbolic proof/witness obligation for one AY shared-engine lane.
pub(crate) fn ay_shared_engine_symbolic_proof_kind(
    lane: AYSharedEngineLane,
) -> PreparedSymbolicProofKind {
    match lane {
        AYSharedEngineLane::AllSatEnumeration => PreparedSymbolicProofKind::ModelExtraction,
        AYSharedEngineLane::Bmc => PreparedSymbolicProofKind::BoundedModelCheck,
        AYSharedEngineLane::Chc => PreparedSymbolicProofKind::ChcQuery,
        AYSharedEngineLane::Pdr => PreparedSymbolicProofKind::PdrSafetyProof,
        AYSharedEngineLane::KInduction => PreparedSymbolicProofKind::KInduction,
    }
}

/// Shared backend family for one AY lane.
pub(crate) fn ay_shared_engine_backend(lane: AYSharedEngineLane) -> BackendKind {
    match lane {
        AYSharedEngineLane::Chc | AYSharedEngineLane::Pdr => BackendKind::AYChc,
        AYSharedEngineLane::AllSatEnumeration
        | AYSharedEngineLane::Bmc
        | AYSharedEngineLane::KInduction => BackendKind::AYSmt,
    }
}

/// Admission status for a prepared AY shared-engine lane candidate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum AYSharedEngineAdmissionStatus {
    /// All required descriptors and lane prerequisites are present.
    Admitted,
    /// The lane is compatible, but frontend lowering has not provided every
    /// semantic prerequisite needed to start the candidate.
    Delayed,
    /// A required shared-engine capability is absent or the frontend is unsupported.
    Blocked,
}

impl AYSharedEngineAdmissionStatus {
    #[must_use]
    pub(crate) const fn code(self) -> &'static str {
        match self {
            Self::Admitted => "admitted",
            Self::Delayed => "delayed",
            Self::Blocked => "blocked",
        }
    }
}

/// Frontend-neutral route/admission decision for one AY shared-engine lane.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct AYSharedEngineLaneAdmission {
    pub(crate) lane: AYSharedEngineLane,
    pub(crate) candidate_key: &'static str,
    pub(crate) frontend_family: Option<AYFrontendFamily>,
    pub(crate) status: AYSharedEngineAdmissionStatus,
    pub(crate) reason_code: &'static str,
    pub(crate) portfolio_rank: u32,
    pub(crate) required_prerequisites: Vec<&'static str>,
    pub(crate) satisfied_prerequisites: Vec<&'static str>,
    pub(crate) missing_prerequisites: Vec<&'static str>,
    pub(crate) required_capabilities: Vec<String>,
    pub(crate) satisfied_capabilities: Vec<String>,
    pub(crate) missing_capabilities: Vec<String>,
}

impl AYSharedEngineLaneAdmission {
    #[must_use]
    pub(crate) const fn status_code(&self) -> &'static str {
        self.status.code()
    }

    #[must_use]
    pub(crate) const fn reason_code(&self) -> &'static str {
        self.reason_code
    }

    #[must_use]
    pub(crate) const fn candidate_key(&self) -> &'static str {
        self.candidate_key
    }

    #[must_use]
    pub(crate) const fn portfolio_rank(&self) -> u32 {
        self.portfolio_rank
    }

    #[must_use]
    pub(crate) fn portfolio_candidate_id(&self) -> String {
        ay_shared_engine_lane_portfolio_candidate_id(self.lane)
    }

    #[must_use]
    pub(crate) fn portfolio_winner_eligible(&self) -> bool {
        self.status == AYSharedEngineAdmissionStatus::Admitted
            && ay_shared_engine_prepared_solve_kind(self.lane).is_some()
    }

    #[must_use]
    pub(crate) fn portfolio_lane_role_code(&self) -> &'static str {
        if ay_shared_engine_prepared_solve_kind(self.lane).is_some() {
            "analytical_solve_candidate"
        } else {
            "proof_obligation_support"
        }
    }

    #[must_use]
    pub(crate) fn explicit_search_replacement_status_code(&self) -> &'static str {
        match (
            self.status,
            ay_shared_engine_prepared_solve_kind(self.lane).is_some(),
        ) {
            (AYSharedEngineAdmissionStatus::Admitted, true) => "admitted_replaces_explicit_search",
            (AYSharedEngineAdmissionStatus::Admitted, false) => {
                "admitted_proof_obligation_no_direct_search_replacement"
            }
            (AYSharedEngineAdmissionStatus::Delayed, _) => "fail_closed_explicit_search_retained",
            (AYSharedEngineAdmissionStatus::Blocked, _) => "blocked_explicit_search_retained",
        }
    }

    #[must_use]
    pub(crate) fn publication_blocker_code(&self) -> &'static str {
        if self.status == AYSharedEngineAdmissionStatus::Admitted {
            "none"
        } else {
            self.reason_code
        }
    }

    #[must_use]
    pub(crate) fn prepared_candidate_lane<'a>(
        &self,
        program: &'a PreparedCheckerProgram,
    ) -> Option<&'a PreparedCandidateLaneDescriptor> {
        program
            .candidate_lanes
            .iter()
            .find(|lane| lane.candidate_key.as_deref() == Some(self.candidate_key))
    }

    #[must_use]
    pub(crate) fn reusable_cache_fingerprint_ready(&self) -> bool {
        self.satisfied_capabilities
            .iter()
            .any(|capability| ay_frontend_reusable_cache_fingerprint_capability(capability))
    }

    #[must_use]
    pub(crate) fn cache_fingerprint_compatibility_code(&self) -> &'static str {
        if self.reusable_cache_fingerprint_ready() {
            AY_CACHE_FINGERPRINT_COMPATIBILITY_FRONTEND_REUSABLE
        } else {
            AY_CACHE_FINGERPRINT_COMPATIBILITY_NOT_DECLARED
        }
    }

    #[must_use]
    pub(crate) fn render_evidence_row(
        &self,
        scope: &str,
        program: &PreparedCheckerProgram,
    ) -> String {
        let frontend_family = self
            .frontend_family
            .map(ay_adoption_family_for_frontend)
            .map(SharedEngineFrontendFamily::code)
            .unwrap_or("unsupported");
        let active_frontend_families = ay_current_operational_frontend_family_codes();
        let blocked_frontend_families = ay_blocked_frontend_family_codes();
        let frontend_family_blockers = ay_frontend_family_blocker_evidence();
        let portfolio_candidate_id = self.portfolio_candidate_id();
        format!(
            "{} ay_shared_engine_lane_admission schema={} lane={} candidate_key={} source_kind={} payload_kind={} frontend_family={} active_frontend_families={} default_frontend_families={} blocked_frontend_families={} frontend_family_blockers={} storage_kind={} admission_status={} reason_code={} admission_fail_closed=true portfolio_admission_policy={} portfolio_admission_status={} portfolio_lane_role={} portfolio_rank={} portfolio_candidate_id={} portfolio_winner_policy={} portfolio_winner_eligible={} explicit_search_replacement_policy={} explicit_search_replacement_status={} publication_blocker={} cache_fingerprint_compatibility={} reusable_cache_fingerprint_ready={} required_prerequisites={} satisfied_prerequisites={} missing_prerequisites={} required_capabilities={} satisfied_capabilities={} missing_capabilities={}",
            scope,
            AY_SHARED_ENGINE_ROUTING_SCHEMA,
            self.lane.code(),
            self.candidate_key,
            program.source_kind.code(),
            program.payload_kind.code(),
            frontend_family,
            active_frontend_families,
            active_frontend_families,
            blocked_frontend_families,
            frontend_family_blockers,
            program.storage_kind.code(),
            self.status_code(),
            self.reason_code,
            AY_SHARED_ENGINE_PORTFOLIO_ADMISSION_POLICY,
            self.status_code(),
            self.portfolio_lane_role_code(),
            self.portfolio_rank,
            portfolio_candidate_id,
            AY_SHARED_ENGINE_PORTFOLIO_WINNER_POLICY,
            self.portfolio_winner_eligible(),
            AY_SHARED_ENGINE_EXPLICIT_SEARCH_REPLACEMENT_POLICY,
            self.explicit_search_replacement_status_code(),
            self.publication_blocker_code(),
            self.cache_fingerprint_compatibility_code(),
            self.reusable_cache_fingerprint_ready(),
            join_static_evidence_values(&self.required_prerequisites),
            join_static_evidence_values(&self.satisfied_prerequisites),
            join_static_evidence_values(&self.missing_prerequisites),
            join_string_evidence_values(&self.required_capabilities),
            join_string_evidence_values(&self.satisfied_capabilities),
            join_string_evidence_values(&self.missing_capabilities),
        )
    }
}

/// Evaluate a prepared program against one frontend-neutral AY lane contract.
#[must_use]
pub(crate) fn ay_shared_engine_lane_admission(
    program: &PreparedCheckerProgram,
    lane: AYSharedEngineLane,
) -> AYSharedEngineLaneAdmission {
    let metadata = tla_ay::ay_shared_engine_lane_metadata(lane);
    let candidate_key = ay_shared_engine_candidate_key(lane);
    let problem = ay_shared_engine_problem(lane);
    let backend = ay_shared_engine_backend(lane);
    let proof_kind = ay_shared_engine_symbolic_proof_kind(lane);
    let solve_kind = ay_shared_engine_prepared_solve_kind(lane);
    let required_backend = ay_shared_engine_backend_family(lane.code(), backend, problem);
    let frontend_family = ay_frontend_family_for_payload(program.payload_kind);

    let mut required_capabilities = Vec::new();
    let mut satisfied_capabilities = Vec::new();
    let mut missing_capabilities = Vec::new();

    let frontend_capability = frontend_family
        .map(ay_adoption_family_for_frontend)
        .map(|family| format!("frontend_family:{}", family.code()))
        .unwrap_or_else(|| {
            format!(
                "frontend_family:unsupported_payload:{}",
                program.payload_kind.code()
            )
        });
    let candidate_lane = program
        .candidate_lanes
        .iter()
        .find(|candidate| candidate.candidate_key.as_deref() == Some(candidate_key));
    record_capability(
        &mut required_capabilities,
        &mut satisfied_capabilities,
        &mut missing_capabilities,
        frontend_capability,
        frontend_family.is_some_and(|family| metadata.supports_frontend(family)),
    );

    record_capability(
        &mut required_capabilities,
        &mut satisfied_capabilities,
        &mut missing_capabilities,
        format!("candidate_lane:{candidate_key}"),
        candidate_lane.is_some(),
    );

    if let Some(candidate) = candidate_lane {
        record_capability(
            &mut required_capabilities,
            &mut satisfied_capabilities,
            &mut missing_capabilities,
            format!("cache_fingerprint_compatibility:{candidate_key}:frontend_reusable"),
            candidate.identities.cache_key.is_some()
                && candidate.identities.fingerprint_identity.is_some(),
        );
    }

    record_capability(
        &mut required_capabilities,
        &mut satisfied_capabilities,
        &mut missing_capabilities,
        ay_backend_capability_label(&required_backend),
        program.backend_families.iter().any(|family| {
            family.backend == required_backend.backend
                && family.problem == required_backend.problem
                && required_backend
                    .facets
                    .iter()
                    .all(|facet| family.facets.contains(facet))
        }),
    );

    record_capability(
        &mut required_capabilities,
        &mut satisfied_capabilities,
        &mut missing_capabilities,
        format!("symbolic_proof:{}:{}", proof_kind.code(), problem.code()),
        program
            .symbolic_proofs
            .iter()
            .any(|proof| proof.kind == proof_kind && proof.problem == problem),
    );

    record_capability(
        &mut required_capabilities,
        &mut satisfied_capabilities,
        &mut missing_capabilities,
        format!(
            "validation_plan:{}:{}:required_fail_closed",
            PreparedValidationKind::AYProof.code(),
            problem.code()
        ),
        program.validation_plans.iter().any(|plan| {
            plan.kind == PreparedValidationKind::AYProof
                && plan.problem == problem
                && plan.required
                && plan.fail_closed
        }),
    );

    if let Some(solve_kind) = solve_kind {
        record_capability(
            &mut required_capabilities,
            &mut satisfied_capabilities,
            &mut missing_capabilities,
            format!("analytical_solve:{}:{}", solve_kind.code(), problem.code()),
            program
                .analytical_solves
                .iter()
                .any(|solve| solve.kind == solve_kind && solve.problem == problem),
        );
    }

    let missing_prerequisites = ay_shared_engine_missing_prerequisites(program, lane);
    let satisfied_prerequisites = metadata
        .generic_prerequisites
        .iter()
        .copied()
        .filter(|prerequisite| !missing_prerequisites.contains(prerequisite))
        .collect::<Vec<_>>();

    for &prerequisite in metadata.generic_prerequisites {
        record_capability(
            &mut required_capabilities,
            &mut satisfied_capabilities,
            &mut missing_capabilities,
            ay_lane_prerequisite_capability_label(lane, prerequisite),
            !missing_prerequisites.contains(&prerequisite),
        );
    }

    let (status, reason_code) = ay_shared_engine_admission_status_and_reason(
        &missing_capabilities,
        &missing_prerequisites,
        solve_kind,
    );

    AYSharedEngineLaneAdmission {
        lane,
        candidate_key,
        frontend_family,
        status,
        reason_code,
        portfolio_rank: ay_shared_engine_lane_portfolio_rank(lane),
        required_prerequisites: metadata.generic_prerequisites.to_vec(),
        satisfied_prerequisites,
        missing_prerequisites,
        required_capabilities,
        satisfied_capabilities,
        missing_capabilities,
    }
}

/// Evaluate all AY shared-engine lane candidates in stable lane order.
#[must_use]
pub(crate) fn ay_shared_engine_lane_admissions(
    program: &PreparedCheckerProgram,
) -> Vec<AYSharedEngineLaneAdmission> {
    tla_ay::AY_SHARED_ENGINE_LANES
        .iter()
        .copied()
        .map(|lane| ay_shared_engine_lane_admission(program, lane))
        .collect()
}

/// Render route/admission evidence rows for every AY shared-engine lane.
#[must_use]
pub(crate) fn ay_shared_engine_admission_evidence_rows(
    scope: &str,
    program: &PreparedCheckerProgram,
) -> Vec<String> {
    ay_shared_engine_lane_admissions(program)
        .iter()
        .map(|admission| admission.render_evidence_row(scope, program))
        .collect()
}

fn ay_frontend_family_for_payload(
    payload_kind: PreparedProgramPayloadKind,
) -> Option<AYFrontendFamily> {
    match payload_kind {
        PreparedProgramPayloadKind::Tla => Some(AYFrontendFamily::Tla),
        PreparedProgramPayloadKind::Quint => Some(AYFrontendFamily::Quint),
        PreparedProgramPayloadKind::MccPetri => Some(AYFrontendFamily::MccPetri),
        PreparedProgramPayloadKind::Aiger => Some(AYFrontendFamily::Aiger),
        PreparedProgramPayloadKind::Btor2 => Some(AYFrontendFamily::Btor2),
        PreparedProgramPayloadKind::AYOnly => Some(AYFrontendFamily::AYOnly),
        PreparedProgramPayloadKind::VmtInterchange => Some(AYFrontendFamily::VmtReplay),
        PreparedProgramPayloadKind::WitnessReplay => Some(AYFrontendFamily::WitnessReplay),
    }
}

fn ay_adoption_family_for_payload(
    payload_kind: PreparedProgramPayloadKind,
) -> SharedEngineFrontendFamily {
    match payload_kind {
        PreparedProgramPayloadKind::Tla => SharedEngineFrontendFamily::TlaPlus,
        PreparedProgramPayloadKind::Quint => SharedEngineFrontendFamily::Quint,
        PreparedProgramPayloadKind::MccPetri => SharedEngineFrontendFamily::MccPetri,
        PreparedProgramPayloadKind::Aiger => SharedEngineFrontendFamily::Aiger,
        PreparedProgramPayloadKind::Btor2 => SharedEngineFrontendFamily::Btor2,
        PreparedProgramPayloadKind::VmtInterchange => {
            SharedEngineFrontendFamily::VmtTransitionSystem
        }
        PreparedProgramPayloadKind::AYOnly => SharedEngineFrontendFamily::AYAnalytical,
        PreparedProgramPayloadKind::WitnessReplay => SharedEngineFrontendFamily::WitnessReplay,
    }
}

fn ay_adoption_family_for_frontend(
    frontend_family: AYFrontendFamily,
) -> SharedEngineFrontendFamily {
    match frontend_family {
        AYFrontendFamily::Tla => SharedEngineFrontendFamily::TlaPlus,
        AYFrontendFamily::Quint => SharedEngineFrontendFamily::Quint,
        AYFrontendFamily::MccPetri => SharedEngineFrontendFamily::MccPetri,
        AYFrontendFamily::Aiger => SharedEngineFrontendFamily::Aiger,
        AYFrontendFamily::Btor2 => SharedEngineFrontendFamily::Btor2,
        AYFrontendFamily::VmtReplay => SharedEngineFrontendFamily::VmtTransitionSystem,
        AYFrontendFamily::AYOnly => SharedEngineFrontendFamily::AYAnalytical,
        AYFrontendFamily::WitnessReplay => SharedEngineFrontendFamily::WitnessReplay,
        AYFrontendFamily::FutureImporter => SharedEngineFrontendFamily::FutureImporter,
    }
}

fn ay_analytical_proof_frontend_family_contract() -> Vec<SharedEngineFrontendFamily> {
    let mut families = Vec::new();
    for frontend in AYFrontendFamily::all() {
        match frontend {
            AYFrontendFamily::Tla => {
                push_unique_adoption_family(&mut families, SharedEngineFrontendFamily::TlaPlus);
            }
            AYFrontendFamily::Quint => {
                push_unique_adoption_family(&mut families, SharedEngineFrontendFamily::Quint);
            }
            AYFrontendFamily::MccPetri => {
                push_unique_adoption_family(&mut families, SharedEngineFrontendFamily::MccPetri);
            }
            AYFrontendFamily::Aiger => {
                push_unique_adoption_family(&mut families, SharedEngineFrontendFamily::Aiger);
            }
            AYFrontendFamily::Btor2 => {
                push_unique_adoption_family(&mut families, SharedEngineFrontendFamily::Btor2);
            }
            AYFrontendFamily::AYOnly => {
                push_unique_adoption_family(
                    &mut families,
                    SharedEngineFrontendFamily::AYAnalytical,
                );
            }
            AYFrontendFamily::VmtReplay => {
                push_unique_adoption_family(
                    &mut families,
                    SharedEngineFrontendFamily::VmtTransitionSystem,
                );
            }
            AYFrontendFamily::WitnessReplay => {
                push_unique_adoption_family(
                    &mut families,
                    SharedEngineFrontendFamily::WitnessReplay,
                );
            }
            AYFrontendFamily::FutureImporter => {}
        }
    }
    families
}

fn ay_analytical_proof_frontend_family_blockers() -> Vec<SharedEngineAdoptionFamilyBlocker> {
    vec![SharedEngineAdoptionFamilyBlocker::new(
        SharedEngineFrontendFamily::FutureImporter,
        AY_ANALYTICAL_PROOF_FUTURE_IMPORTER_BLOCKER,
    )]
}

fn ay_current_operational_frontend_family_codes() -> String {
    ay_adoption_family_codes(&ay_analytical_proof_frontend_family_contract())
}

fn ay_blocked_frontend_family_codes() -> String {
    ay_adoption_family_codes(
        &ay_analytical_proof_frontend_family_blockers()
            .iter()
            .map(|blocker| blocker.frontend_family)
            .collect::<Vec<_>>(),
    )
}

fn ay_frontend_family_blocker_evidence() -> String {
    ay_analytical_proof_frontend_family_blockers()
        .iter()
        .map(|blocker| format!("{}:{}", blocker.frontend_family.code(), blocker.blocker))
        .collect::<Vec<_>>()
        .join(",")
}

fn push_unique_adoption_family(
    families: &mut Vec<SharedEngineFrontendFamily>,
    family: SharedEngineFrontendFamily,
) {
    if !families.contains(&family) {
        families.push(family);
    }
}

fn ay_adoption_family_codes(families: &[SharedEngineFrontendFamily]) -> String {
    let codes = SharedEngineFrontendFamily::all()
        .iter()
        .copied()
        .filter(|family| families.contains(family))
        .map(|family| family.code())
        .collect::<Vec<_>>();
    if codes.is_empty() {
        "none".to_string()
    } else {
        codes.join(",")
    }
}

fn ay_analytical_proof_default_frontend_families(
    _first_beneficiary: SharedEngineFrontendFamily,
    _second_beneficiary: SharedEngineFrontendFamily,
    compatible_frontend_families: &[SharedEngineFrontendFamily],
) -> Vec<SharedEngineFrontendFamily> {
    SharedEngineFrontendFamily::all()
        .iter()
        .copied()
        .filter(|family| compatible_frontend_families.contains(family))
        .filter(|family| *family != SharedEngineFrontendFamily::FutureImporter)
        .collect()
}

fn ay_analytical_proof_remaining_frontend_families(
    default_frontend_families: &[SharedEngineFrontendFamily],
    compatible_frontend_families: &[SharedEngineFrontendFamily],
) -> Vec<SharedEngineFrontendFamily> {
    SharedEngineFrontendFamily::all()
        .iter()
        .copied()
        .filter(|family| compatible_frontend_families.contains(family))
        .filter(|family| !default_frontend_families.contains(family))
        .collect()
}

fn ay_analytical_proof_second_beneficiary(
    first: SharedEngineFrontendFamily,
    families: &[SharedEngineFrontendFamily],
) -> SharedEngineFrontendFamily {
    families
        .iter()
        .copied()
        .find(|family| *family != SharedEngineFrontendFamily::AYAnalytical && *family != first)
        .unwrap_or(SharedEngineFrontendFamily::TlaPlus)
}

fn ay_analytical_proof_contract_covers_all_payloads(
    families: &[SharedEngineFrontendFamily],
) -> bool {
    PreparedProgramPayloadKind::shared_engine_payloads()
        .iter()
        .all(|payload| families.contains(&ay_adoption_family_for_payload(*payload)))
}

fn ay_metadata_contract_is_frontend_neutral() -> bool {
    tla_ay::ay_shared_engine_all_lane_metadata()
        .iter()
        .all(|metadata| {
            metadata.frontend_neutral
                && AYFrontendFamily::all()
                    .iter()
                    .filter(|frontend| **frontend != AYFrontendFamily::FutureImporter)
                    .all(|frontend| metadata.supports_frontend(*frontend))
        })
}

fn ay_analytical_proof_release_ready(program: &PreparedCheckerProgram) -> bool {
    let families = ay_analytical_proof_frontend_family_contract();
    ay_metadata_contract_is_frontend_neutral()
        && ay_analytical_proof_contract_covers_all_payloads(&families)
        && ay_shared_engine_lane_admissions(program)
            .iter()
            .all(|admission| admission.status == AYSharedEngineAdmissionStatus::Admitted)
}

/// Render release-critical analytical/AY proof adoption evidence once all
/// prepared AY descriptors and lane prerequisites are admitted.
#[must_use]
pub(crate) fn ay_analytical_proof_shared_engine_adoption_evidence_row(
    scope: &str,
    program: &PreparedCheckerProgram,
) -> Option<String> {
    if !ay_analytical_proof_release_ready(program) {
        return None;
    }

    let families = ay_analytical_proof_frontend_family_contract();
    let blockers = ay_analytical_proof_frontend_family_blockers();
    let blocked_family_codes = blockers
        .iter()
        .map(|blocker| blocker.frontend_family.code())
        .collect::<Vec<_>>()
        .join(",");
    let first_beneficiary = ay_adoption_family_for_payload(program.payload_kind);
    let second_beneficiary = ay_analytical_proof_second_beneficiary(first_beneficiary, &families);
    let evidence = SharedEngineAdoptionEvidence::new(
        SharedEngineFrontendFamily::AYAnalytical.code(),
        tla_ay::AY_ANALYTICAL_PROOF_SHARED_ENGINE_COMPONENT,
        first_beneficiary.code(),
        second_beneficiary.code(),
        "shared-core-ready",
        AY_ANALYTICAL_PROOF_ADOPTION_OWNER,
        AY_ANALYTICAL_PROOF_ADOPTION_ACCEPTANCE_TEST,
    )
    .with_level_four_frontend_family_contract(families.iter().copied(), [], [], blockers)
    .with_generic_prerequisite("prepared_transition_descriptor")
    .with_generic_prerequisite("prepared_property_descriptor")
    .with_generic_prerequisite("ay_shared_engine_candidate_lane_descriptors")
    .with_generic_prerequisite("ay_symbolic_proof_descriptors")
    .with_generic_prerequisite("ay_proof_validation_plan")
    .with_generic_prerequisite("ay_proof_fingerprint_identity");

    let family_codes = ay_adoption_family_codes(&evidence.compatible_frontend_families);
    let default_families = ay_analytical_proof_default_frontend_families(
        first_beneficiary,
        second_beneficiary,
        &evidence.compatible_frontend_families,
    );
    let default_family_codes = ay_adoption_family_codes(&default_families);
    let downstream_family_codes =
        ay_adoption_family_codes(&evidence.downstream_beneficiary_families);
    let remaining_families = ay_analytical_proof_remaining_frontend_families(
        &default_families,
        &evidence.compatible_frontend_families,
    );
    let remaining_family_codes = ay_adoption_family_codes(&remaining_families);
    let base = evidence.render_evidence_row(scope);
    Some(format!(
        "{base} architecture_level=4 active_frontend_families={family_codes} \
         default_frontend_families={default_family_codes} \
         downstream_frontend_families={downstream_family_codes} \
         remaining_frontend_families={remaining_family_codes} \
         blocked_frontend_families={blocked_family_codes} \
         engine_phase.analytical_ay_proof.seconds=unknown \
         descriptor_basis=prepared_candidate_lane_descriptors,ay_lane_admission,proof_validation_plan \
         release_critical_component=true"
    ))
}

fn record_capability(
    required: &mut Vec<String>,
    satisfied: &mut Vec<String>,
    missing: &mut Vec<String>,
    label: String,
    is_satisfied: bool,
) {
    required.push(label.clone());
    if is_satisfied {
        satisfied.push(label);
    } else {
        missing.push(label);
    }
}

fn ay_frontend_reusable_cache_fingerprint_capability(capability: &str) -> bool {
    capability.starts_with(AY_CACHE_FINGERPRINT_CAPABILITY_PREFIX)
        && capability.ends_with(":frontend_reusable")
}

fn ay_backend_capability_label(family: &PreparedBackendFamilyDescriptor) -> String {
    let facets = family
        .facets
        .iter()
        .map(|facet| facet.code())
        .collect::<Vec<_>>()
        .join("+");
    format!(
        "backend_family:{}:{}:facets={facets}",
        family.backend.code(),
        family.problem.code()
    )
}

fn ay_lane_prerequisite_capability_label(
    lane: AYSharedEngineLane,
    prerequisite: &'static str,
) -> String {
    format!("lane_prerequisite:{}:{prerequisite}", lane.code())
}

fn ay_shared_engine_missing_prerequisites(
    program: &PreparedCheckerProgram,
    lane: AYSharedEngineLane,
) -> Vec<&'static str> {
    let mut missing = Vec::new();
    match lane {
        AYSharedEngineLane::AllSatEnumeration => {
            if program.storage_kind == PreparedStorageKind::Unknown {
                missing.push("typed_symbolic_variable_vector");
            }
            if program.properties.is_empty() && program.transitions.is_empty() {
                missing.push("state_or_property_query_predicate");
            }
        }
        AYSharedEngineLane::Bmc => {
            if program.storage_kind == PreparedStorageKind::Unknown {
                missing.push("typed_state_vector");
            }
            if program.transitions.is_empty() {
                missing.push("step_indexed_transition_relation");
            }
            if program.properties.is_empty() {
                missing.push("safety_property_at_each_bound");
            }
        }
        AYSharedEngineLane::Chc => {
            if program.storage_kind == PreparedStorageKind::Unknown {
                missing.push("typed_state_vector");
            }
            if program.transitions.is_empty() {
                missing.push("current_next_transition_relation");
            }
            if program.properties.is_empty() {
                missing.push("safety_property");
            }
        }
        AYSharedEngineLane::Pdr => {
            if program.transitions.is_empty() {
                missing.push("transition_system_encoded_as_chc");
            }
            if program.properties.is_empty() {
                missing.push("safety_query_clause");
            }
        }
        AYSharedEngineLane::KInduction => {
            if program.storage_kind == PreparedStorageKind::Unknown {
                missing.push("typed_state_vector");
            }
            if program.transitions.is_empty() {
                missing.push("current_next_transition_relation");
            }
            if program.properties.is_empty() {
                missing.push("safety_property");
            }
        }
    }
    missing
}

fn ay_shared_engine_admission_status_and_reason(
    missing_capabilities: &[String],
    missing_prerequisites: &[&'static str],
    solve_kind: Option<PreparedAnalyticalSolveKind>,
) -> (AYSharedEngineAdmissionStatus, &'static str) {
    if let Some(reason) = missing_capabilities
        .iter()
        .filter_map(|capability| ay_blocked_reason_for_missing_capability(capability))
        .next()
    {
        return (AYSharedEngineAdmissionStatus::Blocked, reason);
    }

    if missing_capabilities
        .iter()
        .any(|capability| ay_frontend_reusable_cache_fingerprint_capability(capability))
    {
        return (
            AYSharedEngineAdmissionStatus::Delayed,
            "delayed_missing_cache_fingerprint_compatibility",
        );
    }

    if let Some(first_missing) = missing_prerequisites.first().copied() {
        return (
            AYSharedEngineAdmissionStatus::Delayed,
            ay_delayed_reason_for_missing_prerequisite(first_missing),
        );
    }

    if solve_kind.is_some()
        && missing_capabilities
            .iter()
            .any(|capability| capability.starts_with("analytical_solve:"))
    {
        return (
            AYSharedEngineAdmissionStatus::Delayed,
            "delayed_missing_prepared_solve",
        );
    }

    (
        AYSharedEngineAdmissionStatus::Admitted,
        "admitted_prerequisites_satisfied",
    )
}

fn ay_blocked_reason_for_missing_capability(capability: &str) -> Option<&'static str> {
    if capability.starts_with("frontend_family:") {
        Some("blocked_unsupported_frontend")
    } else if capability.starts_with("candidate_lane:") {
        Some("blocked_missing_candidate_lane")
    } else if capability.starts_with("backend_family:") {
        Some("blocked_missing_backend_family")
    } else if capability.starts_with("symbolic_proof:") {
        Some("blocked_missing_symbolic_proof")
    } else if capability.starts_with("validation_plan:") {
        Some("blocked_missing_validation_plan")
    } else {
        None
    }
}

fn ay_delayed_reason_for_missing_prerequisite(prerequisite: &str) -> &'static str {
    match prerequisite {
        "typed_symbolic_variable_vector" | "typed_state_vector" => {
            "delayed_missing_typed_state_vector"
        }
        "state_or_property_query_predicate" => "delayed_missing_query_predicate",
        "step_indexed_transition_relation"
        | "current_next_transition_relation"
        | "transition_system_encoded_as_chc" => "delayed_missing_transition_relation",
        "safety_property_at_each_bound" | "safety_property" | "safety_query_clause" => {
            "delayed_missing_safety_property"
        }
        _ => "delayed_missing_lane_prerequisite",
    }
}

fn ay_shared_engine_lane_portfolio_rank(lane: AYSharedEngineLane) -> u32 {
    match lane {
        AYSharedEngineLane::Bmc => 20,
        AYSharedEngineLane::Pdr => 30,
        AYSharedEngineLane::KInduction => 40,
        AYSharedEngineLane::Chc => 50,
        AYSharedEngineLane::AllSatEnumeration => 60,
    }
}

fn ay_shared_engine_lane_portfolio_candidate_id(lane: AYSharedEngineLane) -> String {
    if ay_shared_engine_prepared_solve_kind(lane).is_some() {
        format!("ay.shared_engine.{}.solve", lane.code())
    } else {
        format!("ay.shared_engine.{}.proof_obligation", lane.code())
    }
}

/// Attach all frontend-neutral AY shared-engine lane descriptors to a prepared program.
pub(crate) fn add_ay_shared_engine_prepared_descriptors(
    mut program: PreparedCheckerProgram,
    root_identity: &str,
    root_digest: &str,
) -> PreparedCheckerProgram {
    let prepared_identity = |prefix: &str, suffix: &str| -> String {
        if suffix.is_empty() {
            format!("{prefix}:{root_identity}")
        } else {
            format!("{prefix}:{root_identity}:{suffix}")
        }
    };

    for metadata in tla_ay::ay_shared_engine_all_lane_metadata() {
        let lane = metadata.lane;
        let lane_code = lane.code();
        let candidate_key = ay_shared_engine_candidate_key(lane);
        let problem = ay_shared_engine_problem(lane);
        let backend = ay_shared_engine_backend(lane);
        let proof_kind = ay_shared_engine_symbolic_proof_kind(lane);
        let lane_identity = format!("tla-ay.shared_engine.{lane_code}");

        if let Some(solve_kind) = ay_shared_engine_prepared_solve_kind(lane) {
            program = program.add_analytical_solve(
                format!("ay.shared_engine.{lane_code}.solve"),
                solve_kind,
                problem,
            );
        }

        program = program
            .add_symbolic_proof(
                format!("ay.shared_engine.{lane_code}.proof_obligation"),
                proof_kind,
                problem,
            )
            .add_backend_family(ay_shared_engine_backend_family(lane_code, backend, problem))
            .add_candidate_lane(
                PreparedCandidateLaneDescriptor::new(
                    format!("ay.shared_engine.{lane_code}.candidate"),
                    SetupTraceLaneKind::AY,
                )
                .with_candidate_key(candidate_key)
                .with_cache_key(prepared_identity("ay.shared_engine.cache", lane_code))
                .with_candidate_identity(prepared_identity("ay.shared_engine.candidate", lane_code))
                .with_lane_identity(lane_identity)
                .with_frontend_payload_identity(prepared_identity(
                    "ay.shared_engine.frontend_payload",
                    lane_code,
                ))
                .with_artifact_identity(prepared_identity("ay.shared_engine.artifact", lane_code))
                .with_fingerprint_policy_identity("ay_shared_engine_proof_fingerprint_v1")
                .with_fingerprint_identity(prepared_identity("ay.shared_engine.proof", lane_code)),
            )
            .add_validation_plan(
                PreparedValidationPlanDescriptor::new(
                    format!("ay.shared_engine.{lane_code}.validation"),
                    PreparedValidationKind::AYProof,
                    problem,
                )
                .with_fingerprint(
                    PreparedFingerprintDescriptor::new(
                        format!("ay.shared_engine.{lane_code}.proof"),
                        PreparedFingerprintScheme::CanonicalBytesSha256,
                        "ay-shared-engine-proof-v1",
                    )
                    .with_fingerprint_policy_identity("ay_shared_engine_proof_fingerprint_v1")
                    .with_fingerprint_identity(prepared_identity(
                        "ay.shared_engine.proof",
                        lane_code,
                    )),
                )
                .with_artifact_identity(prepared_identity(
                    "ay.shared_engine.validation_artifact",
                    lane_code,
                )),
            );
    }

    program.add_canonical_identity(
        tla_mc_core::PreparedCanonicalIdentityDescriptor::new(
            "ay.shared_engine.metadata",
            tla_mc_core::PreparedCanonicalIdentityKind::LaneArtifact,
            tla_ay::AY_SHARED_ENGINE_METADATA_SCHEMA,
        )
        .with_digest("fnv1a64", root_digest),
    )
}

fn ay_shared_engine_backend_family(
    lane_code: &str,
    backend: BackendKind,
    problem: ProblemKind,
) -> PreparedBackendFamilyDescriptor {
    let mut family = PreparedBackendFamilyDescriptor::new(
        format!("ay.shared_engine.{lane_code}"),
        backend,
        problem,
    )
    .with_facet(SolverFacet::InProcess)
    .with_facet(SolverFacet::Proof)
    .with_facet(SolverFacet::Witness);

    match problem {
        ProblemKind::Bmc => {
            family = family
                .with_facet(SolverFacet::Bmc)
                .with_facet(SolverFacet::Smt)
                .with_facet(SolverFacet::ModelValues);
        }
        ProblemKind::Chc => {
            family = family
                .with_facet(SolverFacet::Chc)
                .with_facet(SolverFacet::Pdr)
                .with_facet(SolverFacet::LinearIntegerArithmetic);
        }
        ProblemKind::KInduction => {
            family = family
                .with_facet(SolverFacet::KInduction)
                .with_facet(SolverFacet::Smt)
                .with_facet(SolverFacet::LinearIntegerArithmetic);
        }
        ProblemKind::SymbolicExecution => {
            family = family
                .with_facet(SolverFacet::SymbolicExecution)
                .with_facet(SolverFacet::Smt)
                .with_facet(SolverFacet::AllSat)
                .with_facet(SolverFacet::ModelValues);
        }
        _ => {}
    }

    family
}

/// Render shared AY engine metadata rows consumed by TLA-check result surfaces.
pub(crate) fn ay_shared_engine_metadata_evidence_rows(scope: &str) -> Vec<String> {
    let mut rows = vec![tla_ay::render_ay_shared_engine_evidence(scope)];
    rows.extend(
        tla_ay::AY_SHARED_ENGINE_LANES
            .iter()
            .copied()
            .map(|lane| tla_ay::render_ay_shared_engine_lane_evidence(scope, lane)),
    );
    rows
}

/// Render metadata plus route/admission evidence for a prepared program.
pub(crate) fn ay_shared_engine_metadata_and_admission_evidence_rows(
    scope: &str,
    program: &PreparedCheckerProgram,
) -> Vec<String> {
    let mut rows = ay_shared_engine_metadata_evidence_rows(scope);
    rows.extend(ay_shared_engine_admission_evidence_rows(scope, program));
    if let Some(row) = ay_analytical_proof_shared_engine_adoption_evidence_row(scope, program) {
        rows.push(row);
    }
    rows
}

fn join_static_evidence_values(values: &[&'static str]) -> String {
    if values.is_empty() {
        "none".to_string()
    } else {
        values.join(",")
    }
}

fn join_string_evidence_values(values: &[String]) -> String {
    if values.is_empty() {
        "none".to_string()
    } else {
        values.join(",")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tla_mc_core::{
        validate_shared_engine_adoption_evidence_row, PreparedPropertyKind, PreparedTransitionKind,
    };

    // SOUNDNESS GATE (symmetric dual of reconcile_masked_violation): a symbolic
    // safety proof (PDR / k-induction) may resolve the cooperative verdict — and so
    // truncate the BFS lane into an indistinguishable Success — ONLY when the safety
    // invariants are the run's sole obligation. If deadlock-checking is on (the
    // default) or there are liveness/temporal/trace obligations the symbolic lane did
    // not verify, it must NOT resolve, or a reachable deadlock / liveness violation
    // BFS would catch is silently masked into a false "no error" exit 0.
    #[test]
    fn symbolic_safety_proof_resolves_only_when_safety_is_sole_obligation() {
        let with_inv = || Config {
            init: Some("Init".to_string()),
            next: Some("Next".to_string()),
            invariants: vec!["Inv".to_string()],
            ..Default::default()
        };

        // Pure safety, deadlock-checking off: the proof covers everything -> may resolve.
        let mut c = with_inv();
        c.check_deadlock = false;
        assert!(
            symbolic_safety_proof_covers_all_obligations(&c),
            "pure-safety (no deadlock/liveness/trace) must allow a symbolic Satisfied to resolve"
        );

        // Deadlock-checking ON (the default): a safety proof did NOT verify
        // deadlock-freedom -> must NOT resolve (BFS stays authoritative).
        let mut c = with_inv();
        c.check_deadlock = true;
        assert!(
            !symbolic_safety_proof_covers_all_obligations(&c),
            "deadlock-checking on: symbolic safety proof must NOT resolve/mask a deadlock"
        );

        // Liveness/temporal property present -> not covered by a safety proof.
        let mut c = with_inv();
        c.check_deadlock = false;
        c.properties = vec!["EventuallyDone".to_string()];
        assert!(
            !symbolic_safety_proof_covers_all_obligations(&c),
            "liveness property present: symbolic safety proof must NOT resolve"
        );

        // Trace invariant present -> not covered by a safety proof.
        let mut c = with_inv();
        c.check_deadlock = false;
        c.trace_invariants = vec!["TraceInv".to_string()];
        assert!(
            !symbolic_safety_proof_covers_all_obligations(&c),
            "trace invariant present: symbolic safety proof must NOT resolve"
        );

        // An UNRESOLVED temporal SPECIFICATION may carry a liveness goal a safety proof
        // does not discharge -> must NOT resolve.
        let mut c = with_inv();
        c.check_deadlock = false;
        c.specification = Some("Spec".to_string());
        assert!(
            !symbolic_safety_proof_covers_all_obligations(&c),
            "unresolved SPECIFICATION present: symbolic safety proof must NOT resolve"
        );

        // CONSTRAINT / action constraints only RESTRICT the reachable set; a safety proof
        // over the unconstrained superset implies safety over the subset, so they do NOT
        // gate a symbolic Satisfied (superset reasoning) -> MAY resolve.
        let mut c = with_inv();
        c.check_deadlock = false;
        c.constraints = vec!["StateConstraint".to_string()];
        c.action_constraints = vec!["ActionConstraint".to_string()];
        assert!(
            symbolic_safety_proof_covers_all_obligations(&c),
            "constraints only restrict the reachable set (superset reasoning) — must still resolve"
        );
    }

    fn storage_for_payload(payload: PreparedProgramPayloadKind) -> PreparedStorageKind {
        match payload {
            PreparedProgramPayloadKind::Tla | PreparedProgramPayloadKind::Quint => {
                PreparedStorageKind::TlaStateSlots
            }
            PreparedProgramPayloadKind::MccPetri => PreparedStorageKind::PetriMarking,
            PreparedProgramPayloadKind::Aiger | PreparedProgramPayloadKind::Btor2 => {
                PreparedStorageKind::HardwareRegisters
            }
            PreparedProgramPayloadKind::VmtInterchange | PreparedProgramPayloadKind::AYOnly => {
                PreparedStorageKind::SmtVariables
            }
            PreparedProgramPayloadKind::WitnessReplay => PreparedStorageKind::WitnessSteps,
        }
    }

    fn minimal_program(payload: PreparedProgramPayloadKind) -> PreparedCheckerProgram {
        PreparedCheckerProgram::new(
            format!("prepared:{}", payload.code()),
            payload,
            storage_for_payload(payload),
        )
        .add_transition("next", PreparedTransitionKind::SymbolicTransitionRelation)
        .add_property("safety", PreparedPropertyKind::Invariant)
    }

    fn minimal_program_with_ay(payload: PreparedProgramPayloadKind) -> PreparedCheckerProgram {
        add_ay_shared_engine_prepared_descriptors(
            minimal_program(payload),
            &format!("root:{}", payload.code()),
            "digest",
        )
    }

    fn payload_frontend_family_cases() -> [(PreparedProgramPayloadKind, AYFrontendFamily); 8] {
        [
            (PreparedProgramPayloadKind::Tla, AYFrontendFamily::Tla),
            (PreparedProgramPayloadKind::Quint, AYFrontendFamily::Quint),
            (
                PreparedProgramPayloadKind::MccPetri,
                AYFrontendFamily::MccPetri,
            ),
            (PreparedProgramPayloadKind::Aiger, AYFrontendFamily::Aiger),
            (PreparedProgramPayloadKind::Btor2, AYFrontendFamily::Btor2),
            (PreparedProgramPayloadKind::AYOnly, AYFrontendFamily::AYOnly),
            (
                PreparedProgramPayloadKind::VmtInterchange,
                AYFrontendFamily::VmtReplay,
            ),
            (
                PreparedProgramPayloadKind::WitnessReplay,
                AYFrontendFamily::WitnessReplay,
            ),
        ]
    }

    fn evidence_field<'a>(row: &'a str, key: &str) -> Option<&'a str> {
        let prefix = format!("{key}=");
        row.split_whitespace()
            .find_map(|field| field.strip_prefix(&prefix))
    }

    const EXPECTED_SHARED_ADOPTION_FAMILY_CODES: [&str; 9] = [
        "tla_plus",
        "quint",
        "mcc_petri",
        "aiger",
        "btor2",
        "vmt_transition_system",
        "ay_analytical",
        "witness_replay",
        "future_importer",
    ];
    const EXPECTED_COMPATIBLE_ADOPTION_FAMILY_CODES: [&str; 8] = [
        "tla_plus",
        "quint",
        "mcc_petri",
        "aiger",
        "btor2",
        "vmt_transition_system",
        "ay_analytical",
        "witness_replay",
    ];
    const LEGACY_SHARED_ADOPTION_FAMILY_CODES: [&str; 3] = ["ay", "replay", "vmt"];
    const FUTURE_IMPORTER_BLOCKER_EVIDENCE: &str =
        "future_importer:awaiting_registered_importer_payload_identity_layout_fingerprints_validation_receipts";

    fn family_tokens<'a>(row: &'a str, key: &str) -> Vec<&'a str> {
        match evidence_field(row, key) {
            Some("none") | None => Vec::new(),
            Some(value) => value.split(',').collect(),
        }
    }

    fn blocker_family_tokens<'a>(row: &'a str, key: &str) -> Vec<&'a str> {
        family_tokens(row, key)
            .into_iter()
            .filter_map(|blocker| blocker.split_once(':').map(|(family, _)| family))
            .collect()
    }

    fn assert_canonical_family_token(token: &str) {
        assert!(
            EXPECTED_SHARED_ADOPTION_FAMILY_CODES.contains(&token),
            "{token} must be a registered shared adoption family"
        );
        assert!(
            !LEGACY_SHARED_ADOPTION_FAMILY_CODES.contains(&token),
            "{token} must not appear as a legacy shared adoption family"
        );
    }

    fn assert_adoption_row_uses_only_canonical_family_tokens(row: &str) {
        for key in ["origin_frontend", "first_beneficiary", "second_beneficiary"] {
            let token = evidence_field(row, key).expect("adoption row should publish family roles");
            assert_canonical_family_token(token);
        }

        for key in [
            "compatible_frontend_families",
            "default_compatible_frontend_families",
            "downstream_beneficiary_families",
            "remaining_compatible_frontend_families",
            "active_frontend_families",
            "default_frontend_families",
            "downstream_frontend_families",
            "remaining_frontend_families",
            "blocked_frontend_families",
        ] {
            for token in family_tokens(row, key) {
                assert_canonical_family_token(token);
            }
        }

        for token in blocker_family_tokens(row, "frontend_family_blockers") {
            assert_canonical_family_token(token);
        }
    }

    fn assert_future_importer_reserved_in_adoption_row(row: &str) {
        for key in [
            "compatible_frontend_families",
            "default_compatible_frontend_families",
            "remaining_compatible_frontend_families",
            "active_frontend_families",
            "default_frontend_families",
            "remaining_frontend_families",
        ] {
            assert!(
                !family_tokens(row, key)
                    .contains(&SharedEngineFrontendFamily::FutureImporter.code()),
                "future_importer must stay out of {key} until importer payload identity, layout mapping, fingerprints, and validation receipts exist"
            );
        }
        assert_eq!(
            evidence_field(row, "blocked_frontend_families"),
            Some(SharedEngineFrontendFamily::FutureImporter.code())
        );
        assert_eq!(
            evidence_field(row, "frontend_family_blockers"),
            Some(FUTURE_IMPORTER_BLOCKER_EVIDENCE)
        );
        assert_eq!(
            evidence_field(row, "blocker_status"),
            Some("tracked-blockers")
        );
    }

    fn assert_analytical_beneficiary_defaults(
        row: &str,
        expected_first: SharedEngineFrontendFamily,
    ) {
        let first = evidence_field(row, "first_beneficiary")
            .expect("adoption row should publish a first beneficiary");
        let second = evidence_field(row, "second_beneficiary")
            .expect("adoption row should publish a second beneficiary");
        assert_eq!(first, expected_first.code());
        assert_ne!(
            first, second,
            "analytical adoption evidence must not collapse first and second beneficiaries"
        );

        let defaults = family_tokens(row, "default_frontend_families");
        assert_eq!(
            defaults,
            family_tokens(row, "compatible_frontend_families"),
            "release-critical analytical/AY evidence should default every active compatible family"
        );
        assert!(defaults.contains(&SharedEngineFrontendFamily::AYAnalytical.code()));
        assert!(
            defaults.contains(&first),
            "first beneficiary should be enabled in the default analytical adoption lane"
        );
        assert!(
            defaults.contains(&second),
            "second beneficiary should be enabled in the default analytical adoption lane"
        );
        for family in &defaults {
            assert!(
                family_tokens(row, "compatible_frontend_families").contains(family),
                "default family {family} must also be compatible"
            );
            assert!(
                !family_tokens(row, "remaining_frontend_families").contains(family),
                "default family {family} must not also be listed as remaining"
            );
        }
    }

    fn expected_payload_adoption_family_cases(
    ) -> [(PreparedProgramPayloadKind, SharedEngineFrontendFamily); 8] {
        [
            (
                PreparedProgramPayloadKind::Tla,
                SharedEngineFrontendFamily::TlaPlus,
            ),
            (
                PreparedProgramPayloadKind::Quint,
                SharedEngineFrontendFamily::Quint,
            ),
            (
                PreparedProgramPayloadKind::MccPetri,
                SharedEngineFrontendFamily::MccPetri,
            ),
            (
                PreparedProgramPayloadKind::Aiger,
                SharedEngineFrontendFamily::Aiger,
            ),
            (
                PreparedProgramPayloadKind::Btor2,
                SharedEngineFrontendFamily::Btor2,
            ),
            (
                PreparedProgramPayloadKind::VmtInterchange,
                SharedEngineFrontendFamily::VmtTransitionSystem,
            ),
            (
                PreparedProgramPayloadKind::AYOnly,
                SharedEngineFrontendFamily::AYAnalytical,
            ),
            (
                PreparedProgramPayloadKind::WitnessReplay,
                SharedEngineFrontendFamily::WitnessReplay,
            ),
        ]
    }

    #[test]
    fn ay_lane_admission_policy_admits_all_shared_frontend_families() {
        let frontends = payload_frontend_family_cases();
        assert_eq!(
            frontends.len(),
            PreparedProgramPayloadKind::shared_engine_payloads().len()
        );
        for payload in PreparedProgramPayloadKind::shared_engine_payloads() {
            assert!(
                frontends.iter().any(|(candidate, _)| candidate == payload),
                "{payload:?} should have explicit AY shared admission coverage"
            );
        }
        for family in AYFrontendFamily::all() {
            if *family == AYFrontendFamily::FutureImporter {
                continue;
            }
            assert!(
                frontends
                    .iter()
                    .any(|(_, represented)| represented == family),
                "{} should be represented by at least one prepared payload",
                family.name()
            );
        }
        assert!(
            !frontends
                .iter()
                .any(|(_, represented)| *represented == AYFrontendFamily::FutureImporter),
            "future importer is a reserved adoption family until a payload exists"
        );
        assert!(
            !ay_analytical_proof_frontend_family_contract()
                .contains(&SharedEngineFrontendFamily::FutureImporter),
            "future importer must not be a default AY analytical/proof consumer"
        );
        assert!(
            ay_analytical_proof_frontend_family_blockers()
                .iter()
                .any(|blocker| {
                    blocker.frontend_family == SharedEngineFrontendFamily::FutureImporter
                        && blocker.blocker == AY_ANALYTICAL_PROOF_FUTURE_IMPORTER_BLOCKER
                }),
            "future importer should remain in the AY adoption evidence as a tracked blocker"
        );

        for (payload, expected_family) in frontends {
            let expected_adoption_family = ay_adoption_family_for_frontend(expected_family);
            let program = minimal_program_with_ay(payload);
            let rows = ay_shared_engine_metadata_and_admission_evidence_rows(
                expected_family.name(),
                &program,
            );

            for lane in tla_ay::AY_SHARED_ENGINE_LANES {
                let admission = ay_shared_engine_lane_admission(&program, lane);
                assert_eq!(
                    admission.status,
                    AYSharedEngineAdmissionStatus::Admitted,
                    "{payload:?} should admit {} through the shared AY policy",
                    lane.name()
                );
                assert_eq!(admission.reason_code(), "admitted_prerequisites_satisfied");
                assert_eq!(admission.frontend_family, Some(expected_family));
                assert_eq!(
                    admission.cache_fingerprint_compatibility_code(),
                    "frontend_reusable"
                );
                assert!(admission.reusable_cache_fingerprint_ready());
                assert!(admission.missing_capabilities.is_empty());
                assert!(admission.missing_prerequisites.is_empty());

                for prerequisite in &admission.required_prerequisites {
                    let label = ay_lane_prerequisite_capability_label(lane, *prerequisite);
                    assert!(
                        admission.satisfied_capabilities.contains(&label),
                        "{payload:?} {} should expose satisfied prerequisite capability {label}",
                        lane.name()
                    );
                }

                let lane_row = rows
                    .iter()
                    .find(|row| {
                        row.contains("ay_shared_engine_lane_admission")
                            && row.contains(&format!("lane={}", lane.code()))
                    })
                    .expect("shared AY admission rows should include every lane");
                assert!(lane_row.contains(&format!("payload_kind={}", payload.code())));
                assert!(lane_row.contains(&format!(
                    "frontend_family={}",
                    expected_adoption_family.code()
                )));
                assert!(lane_row.contains(&format!(
                    "active_frontend_families={}",
                    EXPECTED_COMPATIBLE_ADOPTION_FAMILY_CODES.join(",")
                )));
                assert!(lane_row.contains(&format!(
                    "default_frontend_families={}",
                    EXPECTED_COMPATIBLE_ADOPTION_FAMILY_CODES.join(",")
                )));
                assert!(lane_row.contains("blocked_frontend_families=future_importer"));
                assert!(lane_row.contains(&format!(
                    "frontend_family_blockers={FUTURE_IMPORTER_BLOCKER_EVIDENCE}"
                )));
                assert!(lane_row.contains("admission_status=admitted"));
                assert!(lane_row.contains("admission_fail_closed=true"));
                assert!(lane_row.contains(
                    "portfolio_admission_policy=admit_only_when_candidate_fingerprint_validation_and_lane_prerequisites_pass"
                ));
                assert!(lane_row.contains("portfolio_admission_status=admitted"));
                assert!(lane_row.contains(&format!(
                    "portfolio_candidate_id={}",
                    admission.portfolio_candidate_id()
                )));
                assert!(lane_row.contains(
                    "portfolio_winner_policy=lowest_rank_admitted_ay_solve_lane_can_win_after_validation_else_explicit_search_remains"
                ));
                assert!(lane_row.contains(&format!(
                    "portfolio_winner_eligible={}",
                    admission.portfolio_winner_eligible()
                )));
                assert!(lane_row.contains(
                    "explicit_search_replacement_policy=replace_explicit_search_only_for_admitted_receipt_backed_ay_solve_lanes"
                ));
                if ay_shared_engine_prepared_solve_kind(lane).is_some() {
                    assert!(lane_row.contains("portfolio_lane_role=analytical_solve_candidate"));
                    assert!(lane_row.contains(
                        "explicit_search_replacement_status=admitted_replaces_explicit_search"
                    ));
                    assert!(admission.portfolio_winner_eligible());
                } else {
                    assert!(lane_row.contains("portfolio_lane_role=proof_obligation_support"));
                    assert!(lane_row.contains(
                        "explicit_search_replacement_status=admitted_proof_obligation_no_direct_search_replacement"
                    ));
                    assert!(!admission.portfolio_winner_eligible());
                }
                assert!(lane_row.contains("publication_blocker=none"));
                assert!(lane_row.contains("cache_fingerprint_compatibility=frontend_reusable"));
                assert!(lane_row.contains("reusable_cache_fingerprint_ready=true"));
                assert!(lane_row.contains("missing_capabilities=none"));
            }
        }
    }

    #[test]
    fn ay_lane_admission_policy_delays_when_frontend_lowering_is_incomplete() {
        let program = add_ay_shared_engine_prepared_descriptors(
            PreparedCheckerProgram::new(
                "delayed",
                PreparedProgramPayloadKind::Quint,
                PreparedStorageKind::TlaStateSlots,
            ),
            "root:delayed",
            "digest",
        );

        let admission = ay_shared_engine_lane_admission(&program, AYSharedEngineLane::Bmc);

        assert_eq!(admission.status, AYSharedEngineAdmissionStatus::Delayed);
        assert_eq!(
            admission.reason_code(),
            "delayed_missing_transition_relation"
        );
        assert!(admission
            .missing_prerequisites
            .contains(&"step_indexed_transition_relation"));
        assert!(admission
            .missing_prerequisites
            .contains(&"safety_property_at_each_bound"));
        assert!(admission
            .missing_capabilities
            .contains(&"lane_prerequisite:bmc:step_indexed_transition_relation".to_string()));
        assert!(admission
            .missing_capabilities
            .contains(&"lane_prerequisite:bmc:safety_property_at_each_bound".to_string()));

        let row = admission.render_evidence_row("Quint", &program);
        assert!(row.contains("ay_shared_engine_lane_admission"));
        assert!(row.contains("admission_status=delayed"));
        assert!(row.contains("reason_code=delayed_missing_transition_relation"));
        assert!(row.contains("admission_fail_closed=true"));
        assert!(row.contains("portfolio_admission_status=delayed"));
        assert!(row.contains("portfolio_winner_eligible=false"));
        assert!(row.contains("publication_blocker=delayed_missing_transition_relation"));
        assert!(
            row.contains("explicit_search_replacement_status=fail_closed_explicit_search_retained")
        );
        assert!(row.contains("missing_capabilities=lane_prerequisite:bmc:step_indexed_transition_relation,lane_prerequisite:bmc:safety_property_at_each_bound"));
    }

    #[test]
    fn ay_lane_admission_policy_blocks_when_shared_capabilities_are_absent() {
        let program = minimal_program(PreparedProgramPayloadKind::MccPetri);
        let admission = ay_shared_engine_lane_admission(&program, AYSharedEngineLane::Bmc);

        assert_eq!(admission.status, AYSharedEngineAdmissionStatus::Blocked);
        assert_eq!(admission.reason_code(), "blocked_missing_candidate_lane");
        assert!(admission
            .missing_capabilities
            .contains(&"candidate_lane:ay_bmc".to_string()));
        assert!(admission
            .satisfied_capabilities
            .contains(&"frontend_family:mcc_petri".to_string()));

        let rows = ay_shared_engine_metadata_and_admission_evidence_rows("MCC/Petri", &program);
        assert!(rows.iter().any(|row| {
            row.contains("ay_shared_engine_lane_admission")
                && row.contains("lane=bmc")
                && row.contains("admission_status=blocked")
                && row.contains("reason_code=blocked_missing_candidate_lane")
                && row.contains("portfolio_winner_eligible=false")
                && row.contains("publication_blocker=blocked_missing_candidate_lane")
                && row
                    .contains("explicit_search_replacement_status=blocked_explicit_search_retained")
        }));
        assert!(
            !rows
                .iter()
                .any(|row| row.contains("shared_engine_adoption")),
            "blocked shared AY lanes must not emit release adoption evidence"
        );
    }

    #[test]
    fn ay_lane_admission_policy_delays_without_cache_fingerprint_compatibility() {
        let mut program = minimal_program_with_ay(PreparedProgramPayloadKind::Tla);
        let bmc_lane = program
            .candidate_lanes
            .iter_mut()
            .find(|lane| lane.candidate_key.as_deref() == Some("ay_bmc"))
            .expect("minimal AY descriptors include the BMC lane");
        bmc_lane.identities.cache_key = None;

        let admission = ay_shared_engine_lane_admission(&program, AYSharedEngineLane::Bmc);

        assert_eq!(admission.status, AYSharedEngineAdmissionStatus::Delayed);
        assert_eq!(
            admission.reason_code(),
            "delayed_missing_cache_fingerprint_compatibility"
        );
        assert!(admission
            .missing_capabilities
            .contains(&"cache_fingerprint_compatibility:ay_bmc:frontend_reusable".to_string()));
        assert!(admission
            .satisfied_capabilities
            .contains(&"candidate_lane:ay_bmc".to_string()));

        let row = admission.render_evidence_row("TY", &program);
        assert!(row.contains("ay_shared_engine_lane_admission"));
        assert!(row.contains("admission_status=delayed"));
        assert!(row.contains("reason_code=delayed_missing_cache_fingerprint_compatibility"));
        assert!(row.contains("portfolio_admission_status=delayed"));
        assert!(row.contains("portfolio_winner_eligible=false"));
        assert!(row.contains("publication_blocker=delayed_missing_cache_fingerprint_compatibility"));
        assert!(row.contains("cache_fingerprint_compatibility=not_declared"));
        assert!(row.contains("reusable_cache_fingerprint_ready=false"));
        assert!(row.contains(
            "missing_capabilities=cache_fingerprint_compatibility:ay_bmc:frontend_reusable"
        ));
    }

    #[test]
    fn ay_lane_admission_policy_fails_closed_cross_frontend_without_fingerprint_evidence() {
        for payload in [
            PreparedProgramPayloadKind::Tla,
            PreparedProgramPayloadKind::Quint,
            PreparedProgramPayloadKind::MccPetri,
            PreparedProgramPayloadKind::Aiger,
            PreparedProgramPayloadKind::Btor2,
            PreparedProgramPayloadKind::AYOnly,
            PreparedProgramPayloadKind::VmtInterchange,
            PreparedProgramPayloadKind::WitnessReplay,
        ] {
            let mut program = minimal_program_with_ay(payload);
            let bmc_lane = program
                .candidate_lanes
                .iter_mut()
                .find(|lane| lane.candidate_key.as_deref() == Some("ay_bmc"))
                .expect("minimal AY descriptors include the BMC lane");
            bmc_lane.identities.fingerprint_identity = None;

            let admission = ay_shared_engine_lane_admission(&program, AYSharedEngineLane::Bmc);

            assert_eq!(
                admission.status,
                AYSharedEngineAdmissionStatus::Delayed,
                "{payload:?} must fail closed without reusable fingerprint evidence"
            );
            assert_eq!(
                admission.reason_code(),
                "delayed_missing_cache_fingerprint_compatibility"
            );
            assert_eq!(
                admission.cache_fingerprint_compatibility_code(),
                "not_declared"
            );
            assert!(!admission.reusable_cache_fingerprint_ready());
            assert!(admission
                .missing_capabilities
                .contains(&"cache_fingerprint_compatibility:ay_bmc:frontend_reusable".to_string()));

            let rows =
                ay_shared_engine_metadata_and_admission_evidence_rows("CrossFrontend", &program);
            assert!(rows.iter().any(|row| {
                row.contains("ay_shared_engine_lane_admission")
                    && row.contains("lane=bmc")
                    && row.contains(&format!("payload_kind={}", payload.code()))
                    && row.contains("admission_status=delayed")
                    && row.contains("portfolio_winner_eligible=false")
                    && row.contains(
                        "explicit_search_replacement_status=fail_closed_explicit_search_retained",
                    )
                    && row.contains("cache_fingerprint_compatibility=not_declared")
                    && row.contains("reusable_cache_fingerprint_ready=false")
            }));
            assert!(
                !rows.iter().any(|row| row.contains("shared_engine_adoption")),
                "{payload:?} must not emit adoption evidence while a reusable AY cache/fingerprint proof is missing"
            );
        }
    }

    #[test]
    fn ay_lane_admission_policy_blocks_hard_missing_capability_before_cache_delay() {
        let mut program = minimal_program_with_ay(PreparedProgramPayloadKind::Tla);
        let bmc_lane = program
            .candidate_lanes
            .iter_mut()
            .find(|lane| lane.candidate_key.as_deref() == Some("ay_bmc"))
            .expect("minimal AY descriptors include the BMC lane");
        bmc_lane.identities.cache_key = None;
        program
            .backend_families
            .retain(|family| family.problem != ProblemKind::Bmc);

        let admission = ay_shared_engine_lane_admission(&program, AYSharedEngineLane::Bmc);

        assert_eq!(admission.status, AYSharedEngineAdmissionStatus::Blocked);
        assert_eq!(admission.reason_code(), "blocked_missing_backend_family");
        assert!(admission
            .missing_capabilities
            .iter()
            .any(|capability| { capability.starts_with("backend_family:") }));
        assert!(admission
            .missing_capabilities
            .contains(&"cache_fingerprint_compatibility:ay_bmc:frontend_reusable".to_string()));

        let row = admission.render_evidence_row("TY", &program);
        assert!(row.contains("admission_status=blocked"));
        assert!(row.contains("reason_code=blocked_missing_backend_family"));
        assert!(row.contains("portfolio_admission_status=blocked"));
        assert!(row.contains("portfolio_winner_eligible=false"));
        assert!(row.contains("publication_blocker=blocked_missing_backend_family"));
        assert!(row.contains("cache_fingerprint_compatibility=not_declared"));
    }

    #[test]
    fn ay_lane_admission_policy_normalizes_legacy_frontend_aliases_before_evidence() {
        for (payload, expected_family, legacy_code) in [
            (
                PreparedProgramPayloadKind::Tla,
                SharedEngineFrontendFamily::TlaPlus,
                "tla",
            ),
            (
                PreparedProgramPayloadKind::AYOnly,
                SharedEngineFrontendFamily::AYAnalytical,
                "ay_only",
            ),
            (
                PreparedProgramPayloadKind::VmtInterchange,
                SharedEngineFrontendFamily::VmtTransitionSystem,
                "vmt_replay",
            ),
        ] {
            let program = minimal_program_with_ay(payload);
            let admission = ay_shared_engine_lane_admission(&program, AYSharedEngineLane::Bmc);
            let row = admission.render_evidence_row("CanonicalFrontend", &program);
            let canonical_frontend = expected_family.code();

            assert!(row.contains(&format!("frontend_family={canonical_frontend}")));
            assert!(row.contains(&format!("frontend_family:{canonical_frontend}")));
            assert!(
                !row.contains(&format!("frontend_family={legacy_code} ")),
                "{payload:?} should not publish legacy frontend_family={legacy_code}"
            );
            assert!(
                !row.contains(&format!("frontend_family:{legacy_code},")),
                "{payload:?} should not publish legacy frontend_family capability {legacy_code}"
            );
        }
    }

    #[test]
    fn ay_lane_admission_policy_real_payloads_publish_canonical_adoption_families() {
        let registered_family_codes = SharedEngineFrontendFamily::all()
            .iter()
            .map(|family| family.code())
            .collect::<Vec<_>>();
        assert_eq!(
            registered_family_codes,
            EXPECTED_SHARED_ADOPTION_FAMILY_CODES.to_vec()
        );
        assert_eq!(
            ay_adoption_family_codes(&ay_analytical_proof_frontend_family_contract()),
            EXPECTED_COMPATIBLE_ADOPTION_FAMILY_CODES.join(",")
        );
        assert!(
            !ay_analytical_proof_frontend_family_contract()
                .contains(&SharedEngineFrontendFamily::FutureImporter),
            "future_importer is reserved until a real importer payload exists"
        );

        let expected_compatible_families = EXPECTED_COMPATIBLE_ADOPTION_FAMILY_CODES.join(",");
        for (payload, expected_family) in expected_payload_adoption_family_cases() {
            assert_eq!(ay_adoption_family_for_payload(payload), expected_family);
            let frontend_family = ay_frontend_family_for_payload(payload)
                .expect("real shared payloads should have a AY frontend family");
            assert_eq!(
                ay_adoption_family_for_frontend(frontend_family),
                expected_family
            );
            assert_eq!(frontend_family.adoption_code(), expected_family.code());

            let program = minimal_program_with_ay(payload);
            let row = ay_analytical_proof_shared_engine_adoption_evidence_row(
                expected_family.code(),
                &program,
            )
            .expect("admitted shared AY descriptors should emit adoption evidence");

            assert_eq!(
                evidence_field(&row, "compatible_frontend_families"),
                Some(expected_compatible_families.as_str())
            );
            assert_adoption_row_uses_only_canonical_family_tokens(&row);
            assert_analytical_beneficiary_defaults(&row, expected_family);
            assert_future_importer_reserved_in_adoption_row(&row);
            validate_shared_engine_adoption_evidence_row(&row).unwrap();
        }
    }

    #[test]
    fn ay_lane_admission_policy_analytical_proof_adoption_evidence_is_release_critical_and_frontend_neutral(
    ) {
        let program = minimal_program_with_ay(PreparedProgramPayloadKind::MccPetri);
        let row = ay_analytical_proof_shared_engine_adoption_evidence_row("MCC/Petri", &program)
            .expect("admitted shared AY descriptors should emit adoption evidence");

        assert!(row.contains("shared_engine_adoption"));
        assert!(row.contains("shared_engine_component=analytical_ay_proof"));
        assert!(row.contains("origin_frontend=ay_analytical"));
        assert!(row.contains("first_beneficiary=mcc_petri"));
        assert!(row.contains("second_beneficiary=tla_plus"));
        assert!(row.contains("adoption_level=level-4"));
        assert!(row.contains("architecture_level=4"));
        assert!(row.contains("compatible_frontend_families=tla_plus,quint,mcc_petri,aiger,btor2,vmt_transition_system,ay_analytical,witness_replay"));
        assert!(
            row.contains("default_compatible_frontend_families=tla_plus,quint,mcc_petri,aiger,btor2,vmt_transition_system,ay_analytical,witness_replay")
        );
        assert!(row.contains("downstream_beneficiary_families=none"));
        assert!(row.contains("remaining_compatible_frontend_families=none"));
        assert!(row.contains("active_frontend_families=tla_plus,quint,mcc_petri,aiger,btor2,vmt_transition_system,ay_analytical,witness_replay"));
        assert!(row.contains("default_frontend_families=tla_plus,quint,mcc_petri,aiger,btor2,vmt_transition_system,ay_analytical,witness_replay"));
        assert!(row.contains("downstream_frontend_families=none"));
        assert!(row.contains("remaining_frontend_families=none"));
        assert!(row.contains("blocked_frontend_families=future_importer"));
        assert!(row.contains("frontend_family_blockers=future_importer:awaiting_registered_importer_payload_identity_layout_fingerprints_validation_receipts"));
        assert!(row.contains("blocker_status=tracked-blockers"));
        assert!(row.contains("engine_phase.analytical_ay_proof.seconds=unknown"));
        assert!(row.contains("release_critical_component=true"));
        assert_adoption_row_uses_only_canonical_family_tokens(&row);
        assert_future_importer_reserved_in_adoption_row(&row);
        assert_analytical_beneficiary_defaults(&row, SharedEngineFrontendFamily::MccPetri);
        validate_shared_engine_adoption_evidence_row(&row).unwrap();
    }

    #[test]
    fn ay_lane_admission_policy_analytical_proof_adoption_evidence_covers_every_payload_family() {
        let compatible_families =
            "tla_plus,quint,mcc_petri,aiger,btor2,vmt_transition_system,ay_analytical,witness_replay";
        for payload in PreparedProgramPayloadKind::shared_engine_payloads() {
            let expected_family = ay_adoption_family_for_payload(*payload);
            let program = minimal_program_with_ay(*payload);
            let row = ay_analytical_proof_shared_engine_adoption_evidence_row(
                expected_family.code(),
                &program,
            )
            .expect("admitted shared AY descriptors should emit adoption evidence");

            assert_eq!(
                evidence_field(&row, "origin_frontend"),
                Some(SharedEngineFrontendFamily::AYAnalytical.code())
            );
            assert_eq!(
                evidence_field(&row, "first_beneficiary"),
                Some(expected_family.code()),
                "{payload:?} must be named as the direct frontend-family beneficiary"
            );
            assert_eq!(
                evidence_field(&row, "compatible_frontend_families"),
                Some(compatible_families)
            );
            assert_eq!(
                evidence_field(&row, "active_frontend_families"),
                Some(compatible_families)
            );
            assert_eq!(
                evidence_field(&row, "default_frontend_families"),
                Some(compatible_families)
            );
            assert_eq!(
                evidence_field(&row, "remaining_frontend_families"),
                Some("none")
            );
            assert!(
                !evidence_field(&row, "compatible_frontend_families")
                    .expect("compatible families should be present")
                    .split(',')
                    .any(|family| family == SharedEngineFrontendFamily::FutureImporter.code()),
                "future importer must not be promoted as compatible before an importer payload exists"
            );
            assert_eq!(
                evidence_field(&row, "blocked_frontend_families"),
                Some(SharedEngineFrontendFamily::FutureImporter.code())
            );
            assert_eq!(
                evidence_field(&row, "frontend_family_blockers"),
                Some(FUTURE_IMPORTER_BLOCKER_EVIDENCE)
            );
            assert_adoption_row_uses_only_canonical_family_tokens(&row);
            assert_analytical_beneficiary_defaults(&row, expected_family);
            assert_future_importer_reserved_in_adoption_row(&row);
            validate_shared_engine_adoption_evidence_row(&row).unwrap();
        }
    }

    #[test]
    fn first_module_terminator_skips_comment_and_picks_first() {
        // Naive find("\n====") mis-anchors on a `====` inside a header block
        // comment; naive rfind picks the LAST module's terminator. The helper
        // must return the FIRST REAL terminator (comment-depth 0).
        let commented = "---- MODULE M ----\n(* box\n====\nend *)\nX == 1\n====\n";
        let p = super::first_module_terminator_pos(commented).expect("real terminator");
        // the byte at p starts the real `====` line, not the commented one
        assert!(commented[p..].starts_with("===="));
        assert!(
            p > commented.find("end *)").unwrap(),
            "must be AFTER the comment close"
        );

        let multi = "---- MODULE A ----\nX==1\n====\n---- MODULE B ----\nY==2\n====\n";
        let q = super::first_module_terminator_pos(multi).expect("first terminator");
        assert_eq!(
            q,
            multi.find("\n====").unwrap() + 1,
            "must be the FIRST module's"
        );

        assert!(super::first_module_terminator_pos("no terminator here").is_none());
    }
}
