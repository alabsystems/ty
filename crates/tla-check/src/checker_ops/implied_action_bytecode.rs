// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Bytecode-VM fast path for eval implied-action terms (`[][A]_v` PROPERTY
//! terms checked per transition through the interpreter, e.g. refinement
//! properties like EWD998PCal's `[][EWD998!Next]_EWD998!vars`).
//!
//! `check_eval_implied_actions_for_transition` tree-walks the term expression
//! once per transition; for refinement specs that walk is a wide disjunction
//! of abstract actions and dominates the per-transition cost — the leaf
//! refinement operators (`token`, `pending`, ...) are already served by the
//! fingerprint-keyed transition memo, so the residual is the boolean-skeleton
//! walk itself (`eval_or`/`eval_and`/`eval_exists` dispatch per node).
//!
//! This module compiles the term once at prepare time so the skeleton
//! executes in the bytecode VM instead:
//!
//! * Root-module state-dependent zero-arg operators (the refinement mapping
//!   operators) are PINNED to `CallExternal` interpreter callbacks
//!   (`force_external`) instead of being compiled inline — their values stay
//!   interpreter-produced (transition-memo served; CHOOSE order preserved by
//!   construction, no CHOOSE re-implementation runs for them) and a
//!   per-execution memo in the VM reuses each value across the sites that
//!   reference it within one transition.
//! * The compile uses the register-recycling allocator
//!   (`compile_operators_to_bytecode_with_reg_recycling`): a fully inlined
//!   abstract-Next disjunction overflows the default 256-register bump
//!   allocator.
//! * Primed references to pinned operators (`token'`) evaluate through
//!   `eval_prime`'s swapped-array discipline via the VM's prime-aware
//!   `CallExternal` (see `eval_zero_arg_external`).
//!
//! SOUNDNESS CONTRACT (enforced in `check_eval_implied_actions_for_transition`):
//! only a VM verdict of `Bool(true)` is consumed directly — the same trust
//! boundary as the production invariant bytecode path
//! (`check_invariants_via_bytecode`). `Bool(false)`, non-boolean, and VM
//! errors all fall back to a full interpreter evaluation of the same term, so
//! every violation / error the user sees is produced by the tree-walker,
//! byte-identically to a run with this fast path disabled. Disjunct order is
//! not changed: the VM compiles `\/` and `/\` as left-to-right short-circuit
//! jumps and iterates quantifier domains in canonical set order, exactly like
//! the interpreter.
//!
//! Kill switch: `TY_NO_IMPLIED_ACTION_BYTECODE=1` (nothing is compiled or
//! attached). Cross-check harness: `TY_IMPLIED_BC_XCHECK=1` re-evaluates every
//! VM boolean with the interpreter and reports any divergence.

use super::property_classify::{EvalImpliedActionTerm, EvalImpliedActionVm};
use crate::eval::EvalCtx;
use tla_core::ast::Module;

feature_flag!(
    pub(crate) no_implied_action_bytecode,
    "TY_NO_IMPLIED_ACTION_BYTECODE"
);

feature_flag!(pub(crate) implied_bytecode_xcheck, "TY_IMPLIED_BC_XCHECK");

feature_flag!(pub(crate) debug_implied_bytecode, "TY_IMPLIED_BC_DEBUG");

/// Compile eval implied-action term expressions to bytecode and attach the
/// per-term VM handles. Failures (whole-term or per-callee) are non-fatal:
/// un-attached terms simply keep the interpreter-only path.
///
/// Mirrors `compile_trust_cg_implied_action_bytecode_for_cache` (run_helpers):
/// synthesizes one zero-arg operator per term in a cloned module namespace,
/// resolves state variables against the model checker's `VarRegistry`, and
/// compiles through the TIR callee bridge so INSTANCE-imported operators keep
/// their substitution scope.
pub(crate) fn attach_eval_implied_action_bytecode(
    ctx: &EvalCtx,
    root_src: &Module,
    deps_src: &[Module],
    terms: &mut [EvalImpliedActionTerm],
) {
    if terms.is_empty() || no_implied_action_bytecode() || !tla_eval::tir::bytecode_vm_enabled() {
        return;
    }

    let mut root = root_src.clone();
    let mut deps = deps_src.to_vec();

    // Root-module state-dependent zero-arg operators are pinned to
    // CallExternal interpreter callbacks instead of being compiled inline:
    // (a) the fingerprint-keyed transition memo serves them once per state,
    //     while inline VM code would recompute them per transition (the exact
    //     cost the memo eliminated — EWD998PCal's CHOOSE-over-bag `token`);
    // (b) their values (incl. CHOOSE picks) stay interpreter-produced by
    //     construction. Root-module only: plain-name `EvalCtx::eval_op`
    //     resolution is unambiguous there, while a dep-module name could be
    //     shadowed by a same-named root operator.
    // Computed BEFORE the synthesized `__ty_eval_implied_action_*` ops are
    // appended (they reference state and must NOT externalize themselves) and
    // BEFORE state-var resolution (the mention check matches raw `Ident`s).
    let registry_names: Vec<String> = ctx
        .var_registry()
        .names()
        .iter()
        .map(std::string::ToString::to_string)
        .collect();
    let mut force_external: std::collections::HashSet<String> = std::collections::HashSet::new();
    {
        use tla_core::ast::Unit;
        for unit in &root.units {
            if let Unit::Operator(def) = &unit.node {
                if def.params.is_empty()
                    && !def.is_recursive
                    && registry_names
                        .iter()
                        .any(|v| tla_core::expr_mentions_name_spanned_v(&def.body, v))
                {
                    force_external.insert(def.name.node.clone());
                }
            }
        }
    }

    // These synthesized predicates execute as transition predicates over an
    // already materialized (state, next-state) pair; do not re-wrap them into
    // TIR ActionSubscript nodes (that form is for temporal classification and
    // is intentionally unsupported by the VM value compiler).
    root.action_subscript_spans.clear();
    for dep in &mut deps {
        dep.action_subscript_spans.clear();
    }

    let names: Vec<String> = terms
        .iter()
        .enumerate()
        .map(|(idx, _)| format!("__ty_eval_implied_action_{idx}"))
        .collect();

    use tla_core::ast::{OperatorDef, Unit};
    for (idx, term) in terms.iter().enumerate() {
        let name = names[idx].clone();
        root.units.push(tla_core::Spanned::new(
            Unit::Operator(OperatorDef {
                name: tla_core::Spanned::new(name, term.expr.span),
                params: Vec::new(),
                body: term.expr.clone(),
                local: true,
                contains_prime: true,
                guards_depend_on_prime: true,
                has_primed_param: false,
                is_recursive: false,
                self_call_count: 0,
            }),
            term.expr.span,
        ));
    }

    let registry = ctx.var_registry().clone();
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
    let tir_callees = tla_eval::bytecode_vm::collect_bytecode_namespace_callees(&root, &dep_refs);
    let state_var_map: std::collections::HashMap<String, u16> = registry
        .names()
        .iter()
        .enumerate()
        .map(|(idx, name)| (name.to_string(), idx as u16))
        .collect();
    // Register recycling: the fully inlined abstract-Next disjunction
    // overflows the default bump allocator's 256-register file; recycling
    // keeps pressure at max-per-operand. Gated to this entry point only.
    let compiled = tla_eval::bytecode_vm::compile_operators_to_bytecode_with_reg_recycling(
        &root,
        &dep_refs,
        &names,
        ctx.precomputed_constants(),
        Some(ctx.op_replacements()),
        Some(&tir_callees),
        Some(&state_var_map),
        Some(&force_external),
    );

    let debug = debug_implied_bytecode()
        || crate::check::debug::bytecode_vm_stats_enabled()
        || crate::check::debug::debug_bytecode_vm();
    if debug {
        let mut forced: Vec<&String> = force_external.iter().collect();
        forced.sort();
        eprintln!("[implied-bc] forced-external root ops: {forced:?}");
        eprintln!(
            "[implied-bc] eval implied-action bytecode: {}/{} terms compiled ({} failed)",
            compiled.op_indices.len(),
            names.len(),
            compiled.failed.len(),
        );
        for (name, err) in &compiled.failed {
            eprintln!("[implied-bc]   skip {name}: {err}");
        }
    }
    if compiled.op_indices.is_empty() {
        return;
    }

    let compiled = std::sync::Arc::new(compiled);
    for (idx, term) in terms.iter_mut().enumerate() {
        let Some(&func_idx) = compiled.op_indices.get(&names[idx]) else {
            continue;
        };
        if debug {
            eprintln!(
                "[implied-bc]   attached term {idx} ({}) -> func {func_idx}",
                term.name
            );
        }
        // TRUE-verdict cache plan: derive the term's exact observed-input
        // footprint from its compiled bytecode (fail closed — `None` keeps
        // the term on the plain VM + interpreter path with zero change).
        let verdict_cache = if super::implied_verdict_cache::no_implied_verdict_cache() {
            None
        } else {
            tla_tir::bytecode::analyze_predicate_state_footprint(&compiled.chunk, func_idx).map(
                |footprint| {
                    std::sync::Arc::new(super::implied_verdict_cache::ImpliedVerdictCacheSpec::new(
                        super::implied_verdict_cache::next_term_id(),
                        footprint.direct_slots.into(),
                        footprint.zero_arg_externals.into(),
                    ))
                },
            )
        };
        if debug || super::implied_verdict_cache::debug_implied_verdict_cache() {
            match &verdict_cache {
                Some(spec) => eprintln!(
                    "[implied-vc]   term {idx} ({}) verdict cache ARMED: direct_slots={:?} externals={:?}",
                    term.name, spec.direct_slots, spec.zero_arg_externals
                ),
                None => eprintln!(
                    "[implied-vc]   term {idx} ({}) verdict cache not armed (footprint rejected or disabled)",
                    term.name
                ),
            }
        }
        term.vm = Some(EvalImpliedActionVm {
            compiled: std::sync::Arc::clone(&compiled),
            func_idx,
            disabled: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
            verdict_cache,
        });
    }
}
