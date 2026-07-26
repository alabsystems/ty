// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! PlusCal `pc`-based action dispatch optimization.
//!
//! In PlusCal-generated specs, `Next` is a disjunction of actions, each guarded
//! by `pc = "label"` (single-process) or `pc[self] = "label"` (multi-process).
//! TLC optimizes this by dispatching on the `pc` value directly, avoiding the
//! cost of evaluating every action's guard for every state.
//!
//! This module detects the PlusCal pattern at setup time and builds a dispatch
//! table mapping `pc` values to split action indices. During BFS, the model checker
//! reads `pc` from the current state and only evaluates actions whose `pc` guard
//! matches, skipping all others.
//!
//! ## Soundness
//!
//! This optimization is sound because:
//! - Detection requires ALL disjuncts have a `pc = "literal"` guard as the first
//!   conjunct. If any disjunct lacks this pattern, the optimization is not applied.
//! - For a given state, the `pc` value uniquely selects the applicable subset of
//!   actions. Actions with a different `pc` guard would evaluate their guard to
//!   FALSE and produce zero successors — so skipping them changes nothing.
//! - The fallback path (evaluating all actions) is used when the `pc` value is
//!   not found in the dispatch table, ensuring no states are missed.
//! - Multi-process specs (`pc[self]`) are supported via Or-branch guard hoisting:
//!   when `self` is bound by EXISTS, the guard check evaluates `pc[self]` to
//!   determine the effective pc value for the current process.

use std::cell::RefCell;
#[cfg(test)]
use tla_value::Rp;

use rustc_hash::FxHashMap;

use crate::action_instance::{split_action_instances, ActionInstance};
#[cfg(test)]
use crate::coverage::detect_actions;
use crate::eval::EvalCtx;
use crate::value::Value;
use tla_core::ast::{Expr, OperatorDef};
use tla_core::VarIndex;

type LabelDispatch = FxHashMap<Value, Vec<usize>>;
type ProcessPcDispatch = FxHashMap<Value, LabelDispatch>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StrictPcGuardTarget {
    Scalar,
    SelfIndexed,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct StrictPcGuard {
    label: String,
    target: StrictPcGuardTarget,
}

#[derive(Debug, Default)]
pub(crate) struct PcGuardCacheEntry {
    label: Option<String>,
    /// Every source and config-resolved operator name followed to obtain
    /// `label`. A cache hit is usable only while all remain module-scoped.
    shared_op_names: Vec<String>,
    strict_target: Option<StrictPcGuardTarget>,
}

pub(crate) type PcGuardLabelCache = RefCell<FxHashMap<usize, PcGuardCacheEntry>>;

pub(crate) fn new_pc_guard_label_cache() -> PcGuardLabelCache {
    RefCell::new(FxHashMap::default())
}

/// Pre-computed dispatch table for PlusCal `pc`-dispatched specs.
///
/// Built once at setup time. Maps `pc` string values to the indices of
/// actions in the split action list that are guarded by that value.
/// Also stores the split action expressions so they can be enumerated
/// individually during BFS.
#[derive(Debug, Clone)]
pub(crate) struct PcDispatchTable {
    /// The variable index of `pc` in the state array.
    pub(crate) pc_var_idx: VarIndex,
    /// Map from pc value (as a TLA+ string Value) to the list of split action
    /// indices that are guarded by that pc value.
    pub(crate) dispatch: LabelDispatch,
    /// For bounded multi-process PlusCal, map `self -> pc label -> split action
    /// indices`. This avoids scanning every pc-guarded action when the current
    /// state stores `pc` as a function.
    process_dispatch: Option<ProcessPcDispatch>,
    /// The split action expressions from the Next disjunction, in order.
    /// Used during BFS to enumerate only matching actions.
    pub(crate) actions: Vec<ActionInstance>,
    /// Total number of actions in the spec (for debug logging and fallback path).
    #[allow(dead_code)] // Used only in debug_assertions builds
    pub(crate) total_actions: usize,
}

impl PcDispatchTable {
    /// Look up the action indices for a given `pc` value.
    ///
    /// Returns `Some(&[usize])` if the value is in the table, `None` if it
    /// is not (caller should fall back to evaluating all actions).
    #[inline]
    pub(crate) fn actions_for_pc(&self, pc_value: &Value) -> Option<&[usize]> {
        self.dispatch.get(pc_value).map(|v| v.as_slice())
    }

    /// Return matching action indices for the current state's `pc` value.
    ///
    /// For single-process PlusCal this is a direct label lookup. For bounded
    /// multi-process fanout, `pc` is a function and split actions carry the
    /// concrete `self` binding; match each split action against `pc[self]`.
    pub(crate) fn action_indices_for_current_pc(&self, pc_value: &Value) -> Option<Vec<usize>> {
        if let Some(indices) = self.actions_for_pc(pc_value) {
            return Some(indices.to_vec());
        }

        match pc_value {
            Value::Func(f) => self.action_indices_for_pc_func(|key| f.apply(key)),
            Value::IntFunc(f) => self.action_indices_for_pc_func(|key| f.apply(key)),
            _ => None,
        }
    }

    // PC label `Value`s only use immutable variants (String/ModelValue/SmallInt).
    #[allow(clippy::mutable_key_type)]
    fn action_indices_for_pc_func<'a>(
        &self,
        apply_pc: impl Fn(&Value) -> Option<&'a Value>,
    ) -> Option<Vec<usize>> {
        let process_dispatch = self.process_dispatch.as_ref()?;
        let mut out = Vec::new();

        for (self_value, label_dispatch) in process_dispatch {
            let effective_pc = apply_pc(self_value)?;
            if let Some(indices) = label_dispatch.get(effective_pc) {
                out.extend(indices.iter().copied());
            }
        }

        out.sort_unstable();
        Some(out)
    }
}

/// Attempt to detect PlusCal-style `pc` dispatch and build a dispatch table.
///
/// Returns `Some(PcDispatchTable)` if the spec follows the PlusCal pattern
/// where ALL split actions in the Next relation are guarded by `pc = "label"`
/// or `pc[self] = "label"`.
/// Returns `None` if the pattern is not detected (non-PlusCal spec or mixed
/// guard patterns).
///
/// # Arguments
/// * `next_def` - The expanded Next operator definition
/// * `vars` - State variable names
/// * `var_registry` - Variable index registry
/// * `ctx` - Evaluation context for resolving operator references
// `Value` carries interior mutability (Rp<SortedSet>) for fingerprint memoization,
// but the variants used as PC labels (Value::String, Value::ModelValue) are
// effectively immutable for hashing purposes.
#[allow(clippy::mutable_key_type)]
pub(crate) fn detect_pc_dispatch(
    next_def: &OperatorDef,
    vars: &[std::sync::Arc<str>],
    var_registry: &tla_core::VarRegistry,
    ctx: &EvalCtx,
) -> Option<PcDispatchTable> {
    // Step 1: Find the `pc` variable index.
    let pc_var_idx = var_registry.get("pc")?;

    // Step 2: Split top-level action disjuncts. This preserves the previous
    // single-process behavior while also fanning out bounded PlusCal process
    // actions such as `\E self \in 1..N : proc(self)`.
    let actions = split_action_instances(ctx, &next_def.body).ok()?;
    if actions.len() < 2 {
        // Single action or no actions — dispatch table would not help.
        return None;
    }

    // Step 3: For each action, check if its first conjunct is `pc = "literal"`.
    // If ALL actions follow this pattern, build the dispatch table.
    // When an action is an operator reference (Ident), resolve it to its body
    // to inspect the pc guard pattern.
    let mut dispatch: LabelDispatch = FxHashMap::default();
    let mut process_dispatch: ProcessPcDispatch = FxHashMap::default();
    let mut all_actions_have_self = true;
    let mut common_target = None;

    for (idx, action) in actions.iter().enumerate() {
        let self_value = action_self_binding(action);
        let mut shadowed: Vec<String> = action
            .bindings
            .iter()
            .map(|(name, _)| name.to_string())
            .collect();
        let mut shared_op_names = Vec::new();
        let guard = extract_strict_pc_guard(
            &action.expr.node,
            "pc",
            ctx,
            self_value.is_some().then_some("self"),
            &mut shadowed,
            &mut shared_op_names,
            0,
        );
        match guard {
            Some(guard) => {
                if common_target.is_some_and(|target| target != guard.target) {
                    return None;
                }
                common_target = Some(guard.target);

                let pc_value = Value::string(guard.label);
                dispatch.entry(pc_value.clone()).or_default().push(idx);
                match guard.target {
                    StrictPcGuardTarget::Scalar => all_actions_have_self = false,
                    StrictPcGuardTarget::SelfIndexed => {
                        let self_value = self_value?;
                        process_dispatch
                            .entry(self_value.clone())
                            .or_default()
                            .entry(pc_value)
                            .or_default()
                            .push(idx);
                    }
                }
            }
            None => {
                // This action doesn't follow the pc = "label" pattern.
                // Bail out — we can't use pc dispatch for this spec.
                return None;
            }
        }
    }

    // Step 4: Verify we have a non-trivial dispatch (at least 2 distinct pc values).
    if dispatch.len() < 2 {
        return None;
    }

    let _ = vars; // vars used for future multi-process extension
    let total_actions = actions.len();
    let process_dispatch =
        (all_actions_have_self && !process_dispatch.is_empty()).then_some(process_dispatch);

    Some(PcDispatchTable {
        pc_var_idx,
        dispatch,
        process_dispatch,
        actions,
        total_actions,
    })
}

fn action_self_binding(action: &ActionInstance) -> Option<&Value> {
    action
        .bindings
        .iter()
        .rev()
        .find_map(|(name, value)| (name.as_ref() == "self").then_some(value))
}

/// Extract the `pc` guard value from an action expression, resolving operator
/// references through the `EvalCtx`.
///
/// When the expression is an `Ident` (operator reference like `A`), looks up
/// the operator's definition and inspects its body for the pc guard pattern.
/// When the expression is `Apply(Ident("A"), [self])` (multi-process PlusCal),
/// resolves operator `A` and inspects its body.
///
/// This function handles recursive structural wrappers (EXISTS, LET, Label, And)
/// before trying operator resolution, so patterns like
/// `\E self \in S : p(self)` are correctly followed through to find the
/// pc guard inside p's body.
fn extract_pc_guard_resolved(expr: &Expr, pc_name: &str, ctx: &EvalCtx) -> Option<String> {
    match expr {
        Expr::Ident(name, _) => {
            // Resolve operator reference to its body and check the body.
            if let Some(op_def) = ctx.get_op(name) {
                // Recurse with resolution so nested Ident/Apply are also resolved.
                extract_pc_guard_resolved(&op_def.body.node, pc_name, ctx)
            } else {
                None
            }
        }
        // Multi-process PlusCal: Apply(Ident("ncs"), [self]) where ncs(self)
        // is an operator whose body starts with `pc[self] = "label"`.
        Expr::Apply(op_expr, _args) => {
            if let Expr::Ident(name, _) = &op_expr.node {
                if let Some(op_def) = ctx.get_op(name) {
                    return extract_pc_guard_resolved(&op_def.body.node, pc_name, ctx);
                }
            }
            extract_pc_guard(expr, pc_name)
        }
        // Structural wrappers: recurse with resolution so we can resolve
        // operator references inside EXISTS, LET, etc.
        Expr::Exists(_bounds, body) => extract_pc_guard_resolved(&body.node, pc_name, ctx),
        Expr::Let(_defs, body) => extract_pc_guard_resolved(&body.node, pc_name, ctx),
        Expr::Label(label) => extract_pc_guard_resolved(&label.body.node, pc_name, ctx),
        // Conjunction: check leftmost conjunct (may need resolution)
        Expr::And(lhs, _rhs) => extract_pc_guard_resolved(&lhs.node, pc_name, ctx),
        // Direct equality
        Expr::Eq(_, _) => extract_pc_eq_literal(expr, pc_name),
        // Or: not a guard pattern at this level
        _ => None,
    }
}

/// Extract the `pc` guard value from an action expression.
///
/// Looks for patterns:
/// - `pc = "label" /\ ...` (guard is first conjunct)
/// - `pc = "label"` (guard is the entire expression)
/// - Through EXISTS wrappers: `\E self \in S : pc = "label" /\ ...`
/// - Through LET wrappers: `LET x == ... IN pc = "label" /\ ...`
///
/// Returns `Some(label)` if found, `None` otherwise.
fn extract_pc_guard(expr: &Expr, pc_name: &str) -> Option<String> {
    match expr {
        // Conjunction: find the leftmost conjunct in a left-associative chain.
        // In `A /\ B /\ C`, the AST is `And(And(A, B), C)`, so the first
        // conjunct is found by recursively following the left child.
        Expr::And(lhs, _rhs) => extract_pc_guard(&lhs.node, pc_name),

        // Direct equality: `pc = "label"` (action is just a guard, rare but valid)
        Expr::Eq(_, _) => extract_pc_eq_literal(expr, pc_name),

        // EXISTS wrapper: look inside the body
        Expr::Exists(_bounds, body) => extract_pc_guard(&body.node, pc_name),

        // LET wrapper: look inside the body
        Expr::Let(_defs, body) => extract_pc_guard(&body.node, pc_name),

        // Label wrapper: look inside
        Expr::Label(label) => extract_pc_guard(&label.body.node, pc_name),

        // Operator application: the action is applied via an operator reference.
        // We cannot inspect the body without resolving it, so bail out.
        _ => None,
    }
}

/// Check if an expression is `pc = "literal"` or `pc[_] = "literal"` and
/// extract the literal value.
///
/// Handles:
/// - `Expr::Eq(Ident("pc"), String("label"))` — parser output, single-process
/// - `Expr::Eq(StateVar("pc", idx, nid), String("label"))` — after AST transform, single-process
/// - `Expr::Eq(FuncApply(Ident("pc"), _), String("label"))` — multi-process `pc[self] = "label"`
/// - `Expr::Eq(FuncApply(StateVar("pc", ..), _), String("label"))` — multi-process after AST transform
fn extract_pc_eq_literal(expr: &Expr, pc_name: &str) -> Option<String> {
    match expr {
        Expr::Eq(lhs, rhs) => {
            // Check lhs is `pc` (ident, state var, or function application of pc)
            let is_pc_lhs = is_pc_reference(&lhs.node, pc_name);
            if !is_pc_lhs {
                // Also check reversed: "label" = pc or "label" = pc[self]
                let is_pc_rhs = is_pc_reference(&rhs.node, pc_name);
                if is_pc_rhs {
                    // Extract literal from lhs
                    if let Expr::String(label) = &lhs.node {
                        return Some(label.clone());
                    }
                }
                return None;
            }
            // Extract literal from rhs
            if let Expr::String(label) = &rhs.node {
                return Some(label.clone());
            }
            None
        }
        _ => None,
    }
}

/// Check if an expression refers to `pc` — either directly (`pc`), as a state
/// variable (`StateVar("pc", ..)`), or as a function application (`pc[self]`).
fn is_pc_reference(expr: &Expr, pc_name: &str) -> bool {
    match expr {
        Expr::Ident(name, _) => name == pc_name,
        Expr::StateVar(name, _, _) => name == pc_name,
        // Multi-process PlusCal: pc[self] is FuncApply(pc, self)
        Expr::FuncApply(func_expr, _arg) => match &func_expr.node {
            Expr::Ident(name, _) => name == pc_name,
            Expr::StateVar(name, _, _) => name == pc_name,
            _ => false,
        },
        _ => false,
    }
}

/// Check if an Or branch has a `pc` guard that does NOT match the given current
/// value, meaning the branch is guaranteed to produce zero successors.
///
/// This is the hot-path function used by the unified enumerator to skip Or branches
/// in PlusCal-style specs. It resolves Ident references through the EvalCtx and
/// compares the guard literal against the current pc value without allocating.
///
/// Handles both single-process (`pc = "label"`) and multi-process (`pc[self] = "label"`)
/// PlusCal patterns. For multi-process, `current_pc` is the entire function value
/// and the function argument (`self`) is resolved from the ctx binding stack.
///
/// Returns `true` if the branch's pc guard is known to NOT match (i.e., the branch
/// should be skipped). Returns `false` if the guard matches or cannot be determined.
///
/// Part of #3923: guard hoisting for PlusCal dispatch patterns.
#[cfg_attr(not(test), allow(dead_code))]
#[inline]
pub(crate) fn or_branch_pc_guard_mismatches(
    expr: &Expr,
    current_pc: &Value,
    ctx: &EvalCtx,
) -> bool {
    let guard_label = extract_runtime_pc_guard(expr, current_pc, ctx);
    pc_guard_label_mismatches(guard_label.as_deref(), current_pc, ctx)
}

/// Cached variant for the unified enumerator hot path.
///
/// In bounded multi-process PlusCal specs, the same operator-body `Or` branches
/// are visited once per process binding for every parent state. Extracting their
/// `pc[self] = "label"` guards repeatedly walks through operator references and
/// allocates a new string. Cache by stable AST node address for the duration of
/// one successor-enumeration call; the current `self` binding is still read at
/// comparison time, so the result remains process-sensitive.
#[inline]
pub(crate) fn or_branch_pc_guard_mismatches_cached(
    expr: &Expr,
    current_pc: &Value,
    ctx: &EvalCtx,
    cache: &PcGuardLabelCache,
) -> bool {
    let force_legacy = force_legacy_pc_guard_hoist();
    if !force_legacy && !runtime_pc_lookup_is_plain(ctx, "pc") {
        return false;
    }

    let key = expr as *const Expr as usize;
    {
        let borrowed = cache.borrow();
        if let Some(entry) = borrowed.get(&key) {
            if !force_legacy && !strict_cache_entry_is_valid(entry, current_pc, ctx) {
                return false;
            }
            return pc_guard_label_mismatches(entry.label.as_deref(), current_pc, ctx);
        }
    }

    let entry = extract_runtime_pc_guard_cached(expr, current_pc, ctx, force_legacy);
    if !force_legacy && !strict_cache_entry_is_valid(&entry, current_pc, ctx) {
        return false;
    }
    let mismatches = pc_guard_label_mismatches(entry.label.as_deref(), current_pc, ctx);
    cache.borrow_mut().insert(key, entry);
    mismatches
}

fn strict_cache_entry_is_valid(
    entry: &PcGuardCacheEntry,
    current_pc: &Value,
    ctx: &EvalCtx,
) -> bool {
    if entry
        .shared_op_names
        .iter()
        .any(|name| ctx.name_in_local_scope(name))
    {
        return false;
    }
    let expected_target = match current_pc {
        Value::Func(_) | Value::IntFunc(_) => StrictPcGuardTarget::SelfIndexed,
        _ => StrictPcGuardTarget::Scalar,
    };
    entry.label.is_none() || entry.strict_target == Some(expected_target)
}

/// Function-valued `pc` can only be hoisted through the runtime's `self`
/// lookup. Keep scalar pc handling unchanged, but fail closed on `pc[t]` and
/// other function arguments even after another safe Or admitted the global
/// optimization. The explicit legacy force restores the pre-fix extractor for
/// same-binary performance A/B runs.
fn extract_runtime_pc_guard(expr: &Expr, current_pc: &Value, ctx: &EvalCtx) -> Option<String> {
    let force_legacy = force_legacy_pc_guard_hoist();
    if force_legacy {
        return extract_pc_guard_resolved(expr, "pc", ctx);
    }
    if !runtime_pc_lookup_is_plain(ctx, "pc") {
        return None;
    }

    let mut shadowed = Vec::new();
    let mut shared_op_names = Vec::new();
    let guard = extract_strict_pc_guard(
        expr,
        "pc",
        ctx,
        ctx.lookup_binding("self").is_some().then_some("self"),
        &mut shadowed,
        &mut shared_op_names,
        0,
    )?;
    let expected_target = match current_pc {
        Value::Func(_) | Value::IntFunc(_) => StrictPcGuardTarget::SelfIndexed,
        _ => StrictPcGuardTarget::Scalar,
    };
    (guard.target == expected_target).then_some(guard.label)
}

fn extract_runtime_pc_guard_cached(
    expr: &Expr,
    current_pc: &Value,
    ctx: &EvalCtx,
    force_legacy: bool,
) -> PcGuardCacheEntry {
    if force_legacy {
        return PcGuardCacheEntry {
            label: extract_pc_guard_resolved(expr, "pc", ctx),
            ..PcGuardCacheEntry::default()
        };
    }

    let mut shadowed = Vec::new();
    let mut shared_op_names = Vec::new();
    let guard = extract_strict_pc_guard(
        expr,
        "pc",
        ctx,
        ctx.lookup_binding("self").is_some().then_some("self"),
        &mut shadowed,
        &mut shared_op_names,
        0,
    );
    let expected_target = match current_pc {
        Value::Func(_) | Value::IntFunc(_) => StrictPcGuardTarget::SelfIndexed,
        _ => StrictPcGuardTarget::Scalar,
    };
    let guard = guard.filter(|guard| guard.target == expected_target);
    let (label, strict_target) = match guard {
        Some(guard) => (Some(guard.label), Some(guard.target)),
        None => (None, None),
    };
    PcGuardCacheEntry {
        label,
        shared_op_names,
        strict_target,
    }
}

/// The hoist reads the current state's registered `pc` slot directly. If normal
/// expression evaluation could instead observe a binding or substitution, a
/// syntactic guard is not comparable to that cached slot value.
fn runtime_pc_lookup_is_plain(ctx: &EvalCtx, pc_name: &str) -> bool {
    !ctx.name_in_local_scope(pc_name)
        && ctx.instance_substitutions().is_none()
        && ctx.call_by_name_subs().is_none()
}

#[inline]
fn pc_guard_label_mismatches(guard_label: Option<&str>, current_pc: &Value, ctx: &EvalCtx) -> bool {
    let Some(label) = guard_label else {
        return false;
    };

    match current_pc {
        // Single-process: current_pc is a string value, direct comparison.
        Value::String(current) => current.as_ref() != label,
        // Multi-process: current_pc is a function (pc : [Procs -> String]).
        // Look up the effective pc value for the current process by applying
        // the function to the `self` binding from the EXISTS scope.
        Value::Func(f) => match resolve_pc_func_apply(f, ctx) {
            Some(effective_pc) => pc_label_value_mismatches(label, effective_pc),
            None => false, // Can't resolve — don't skip (conservative)
        },
        Value::IntFunc(f) => match resolve_pc_int_func_apply(f, ctx) {
            Some(effective_pc) => pc_label_value_mismatches(label, effective_pc),
            None => false,
        },
        _ => false, // Unknown pc type — don't skip
    }
}

#[inline]
fn pc_label_value_mismatches(label: &str, value: &Value) -> bool {
    match value {
        Value::String(current) => current.as_ref() != label,
        _ => Value::string(label) != *value,
    }
}

/// Look up the `self` binding in the ctx and apply the pc function to get
/// the effective pc value for the current process.
///
/// Multi-process PlusCal specs use `pc[self] = "label"` guards where `pc`
/// is a function `[Procs -> String]` and `self` is bound by an EXISTS.
/// This function resolves the `self` binding and performs the function lookup
/// without constructing or evaluating AST nodes.
///
/// Overhead: one binding chain lookup (O(d) where d = binding depth, typically
/// 1-3 for EXISTS-bound variables) + one sorted-array binary search in FuncValue.
/// This is far cheaper than evaluating the entire action body (~100-1000 eval calls).
fn resolve_pc_func_apply<'a>(
    pc_func: &'a crate::value::FuncValue,
    ctx: &EvalCtx,
) -> Option<&'a Value> {
    // Look up "self" in the binding chain. PlusCal multi-process specs always
    // use "self" as the EXISTS-bound variable.
    let self_value = ctx.lookup_binding("self")?;
    pc_func.apply(&self_value)
}

/// Same as `resolve_pc_func_apply` but for IntIntervalFunc (integer-indexed pc).
fn resolve_pc_int_func_apply<'a>(
    pc_func: &'a crate::value::IntIntervalFunc,
    ctx: &EvalCtx,
) -> Option<&'a Value> {
    let self_value = ctx.lookup_binding("self")?;
    pc_func.apply(&self_value)
}

/// Check whether `Next` contains an `Or` whose `pc[self]` branch guards are
/// resolvable by the runtime hoist.
///
/// The runtime comparison reads `self` from the current [`EvalCtx`]. Detection
/// must therefore prove that `self` is already bound when the guarded `Or` is
/// dispatched. In particular, this rejects the superficially similar shape
/// `(∃ w : A(w)) \/ (∃ w : B(w))`: the outer `Or` runs before either
/// branch binds `w`, so enabling the hoist there only adds per-state work.
///
/// Returns `true` if at least two branches of a runtime-resolvable `Or` have
/// first-conjunct `pc[self] = "label"` guards.
///
/// Used by `run_prepare.rs` to enable multi-process PlusCal guard hoisting when
/// the full dispatch table can't be built (because pc is a function, not a string).
///
/// Part of #3805: multi-process PlusCal guard hoisting.
pub(crate) fn spec_has_pc_guards(next_def: &OperatorDef, ctx: &EvalCtx) -> bool {
    let mut shadowed: Vec<String> = next_def
        .params
        .iter()
        .map(|param| param.name.node.clone())
        .collect();
    has_resolvable_self_pc_guarded_or(&next_def.body.node, "pc", ctx, false, &mut shadowed, 0)
}

/// Legacy loose detector retained only for an explicit same-binary performance
/// A/B switch. It must never authorize the optimization by default because it
/// does not prove that the runtime's hard-coded `self` lookup can succeed.
pub(crate) fn spec_has_pc_guards_legacy(next_def: &OperatorDef, ctx: &EvalCtx) -> bool {
    let mut guard_count = 0;
    count_pc_guarded_or_branches_legacy(&next_def.body.node, "pc", ctx, &mut guard_count, 0);
    guard_count >= 2
}

/// Explicit diagnostic switch that restores both legacy admission and legacy
/// runtime extraction. It is intentionally process-stable and opt-in because
/// the old path cannot prove that `self` denotes the guard's function key.
pub(crate) fn force_legacy_pc_guard_hoist() -> bool {
    static FORCE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *FORCE.get_or_init(|| {
        std::env::var("TY_FORCE_LEGACY_PC_GUARD_HOIST")
            .is_ok_and(|value| matches!(value.trim().to_ascii_lowercase().as_str(), "1" | "true"))
    })
}

fn forwards_bound_self(op_def: &OperatorDef, args: &[tla_core::Spanned<Expr>]) -> bool {
    matches!(
        (op_def.params.as_slice(), args),
        ([param], [arg])
            if param.name.node == "self"
                && param.arity == 0
                && matches!(&arg.node, Expr::Ident(name, _) if name == "self")
    )
}

/// Preserve the canonical generated PlusCal rule requiring the literal
/// spelling `self` at both call boundaries.
fn forwarded_process_alias<'a>(
    op_def: &'a OperatorDef,
    args: &[tla_core::Spanned<Expr>],
    current_alias: Option<&str>,
) -> Option<&'a str> {
    let current_alias = current_alias?;
    let ([param], [arg]) = (op_def.params.as_slice(), args) else {
        return None;
    };
    if param.arity != 0 {
        return None;
    }
    let Expr::Ident(actual, _) = &arg.node else {
        return None;
    };
    if actual != current_alias || current_alias != "self" || param.name.node != "self" {
        return None;
    }
    Some(param.name.node.as_str())
}

fn name_is_shadowed(name: &str, shadowed: &[String]) -> bool {
    shadowed.iter().rev().any(|bound| bound == name)
}

fn remember_shared_op_name(names: &mut Vec<String>, name: &str) {
    if !names.iter().any(|existing| existing == name) {
        names.push(name.to_string());
    }
}

/// Resolve only a module-level operator. Both the source name and its config
/// replacement must be outside all lexical/runtime local scopes; otherwise
/// `EvalCtx::get_op` could select a LET or higher-order formal with the same
/// spelling and make an AST-pointer-only guard cache scope-dependent.
fn resolve_strict_shared_op<'a>(
    name: &str,
    ctx: &'a EvalCtx,
    shadowed: &[String],
    shared_op_names: &mut Vec<String>,
) -> Option<&'a OperatorDef> {
    if name_is_shadowed(name, shadowed) || ctx.name_in_local_scope(name) {
        return None;
    }
    let resolved = ctx.resolve_op_name(name).to_string();
    if name_is_shadowed(&resolved, shadowed) || ctx.name_in_local_scope(&resolved) {
        return None;
    }
    let op_def = ctx.get_op(&resolved)?.as_ref();
    remember_shared_op_name(shared_op_names, name);
    remember_shared_op_name(shared_op_names, &resolved);
    Some(op_def)
}

fn push_bound_shadow_names(bounds: &[tla_core::ast::BoundVar], shadowed: &mut Vec<String>) {
    for bound in bounds {
        shadowed.push(bound.name.node.clone());
        if let Some(tla_core::ast::BoundPattern::Tuple(components)) = &bound.pattern {
            shadowed.extend(components.iter().map(|component| component.node.clone()));
        }
    }
}

/// Find one concrete `Or` evaluation site with two directly certifiable
/// branches. Counts are deliberately not accumulated across unrelated Ors:
/// authorizing a global optimization from two one-guard sites is not useful and
/// makes false-positive reasoning needlessly broad.
fn has_resolvable_self_pc_guarded_or(
    expr: &Expr,
    pc_name: &str,
    ctx: &EvalCtx,
    self_bound: bool,
    shadowed: &mut Vec<String>,
    depth: usize,
) -> bool {
    if depth > 20 {
        return false;
    }

    match expr {
        Expr::Or(a, b) => {
            if self_bound {
                let mut shared_op_names = Vec::new();
                let left = extract_strict_pc_guard(
                    &a.node,
                    pc_name,
                    ctx,
                    Some("self"),
                    shadowed,
                    &mut shared_op_names,
                    depth + 1,
                );
                shared_op_names.clear();
                let right = extract_strict_pc_guard(
                    &b.node,
                    pc_name,
                    ctx,
                    Some("self"),
                    shadowed,
                    &mut shared_op_names,
                    depth + 1,
                );
                if matches!(
                    left,
                    Some(StrictPcGuard {
                        target: StrictPcGuardTarget::SelfIndexed,
                        ..
                    })
                ) && matches!(
                    right,
                    Some(StrictPcGuard {
                        target: StrictPcGuardTarget::SelfIndexed,
                        ..
                    })
                ) {
                    return true;
                }
            }
            has_resolvable_self_pc_guarded_or(
                &a.node,
                pc_name,
                ctx,
                self_bound,
                shadowed,
                depth + 1,
            ) || has_resolvable_self_pc_guarded_or(
                &b.node,
                pc_name,
                ctx,
                self_bound,
                shadowed,
                depth + 1,
            )
        }
        Expr::Exists(bounds, body) => {
            let binds_self = bounds.iter().any(|bound| bound.name.node == "self");
            let mark = shadowed.len();
            push_bound_shadow_names(bounds, shadowed);
            let found = has_resolvable_self_pc_guarded_or(
                &body.node,
                pc_name,
                ctx,
                self_bound || binds_self,
                shadowed,
                depth + 1,
            );
            shadowed.truncate(mark);
            found
        }
        // LET definitions are local operators. Strict mode intentionally does
        // not try to prove their capture/substitution environment.
        Expr::Let(_, _) => false,
        Expr::Label(label) => has_resolvable_self_pc_guarded_or(
            &label.body.node,
            pc_name,
            ctx,
            self_bound,
            shadowed,
            depth + 1,
        ),
        // A zero-argument top-level operator cannot lexically capture the
        // caller's bound `self`; following it would rely on dynamic scoping.
        Expr::Ident(_, _) => false,
        Expr::Apply(op_expr, args) => {
            let Expr::Ident(name, _) = &op_expr.node else {
                return false;
            };
            let mut shared_op_names = Vec::new();
            let Some(op_def) = resolve_strict_shared_op(name, ctx, shadowed, &mut shared_op_names)
            else {
                return false;
            };
            if op_def.params.len() != args.len() {
                return false;
            }

            // A callee formal named `self` shadows the outer binding. Retain
            // the proof only for the ordinary PlusCal forwarding form
            // `callee(self)`, where both bindings denote the same value.
            if self_bound && forwards_bound_self(op_def, args) {
                let mark = shadowed.len();
                shadowed.extend(op_def.params.iter().map(|param| param.name.node.clone()));
                let found = has_resolvable_self_pc_guarded_or(
                    &op_def.body.node,
                    pc_name,
                    ctx,
                    true,
                    shadowed,
                    depth + 1,
                );
                shadowed.truncate(mark);
                found
            } else {
                false
            }
        }
        _ => false,
    }
}

/// Extract a branch label only when its effective first guard reads the
/// process key that the runtime can resolve: `pc[self]`.
fn extract_strict_pc_guard<'a>(
    expr: &Expr,
    pc_name: &str,
    ctx: &'a EvalCtx,
    process_alias: Option<&'a str>,
    shadowed: &mut Vec<String>,
    shared_op_names: &mut Vec<String>,
    depth: usize,
) -> Option<StrictPcGuard> {
    if depth > 20 || name_is_shadowed(pc_name, shadowed) {
        return None;
    }

    match expr {
        Expr::Ident(name, _) => {
            let op_def = resolve_strict_shared_op(name, ctx, shadowed, shared_op_names)?;
            if !op_def.params.is_empty() {
                return None;
            }
            let guard = extract_strict_pc_guard(
                &op_def.body.node,
                pc_name,
                ctx,
                None,
                shadowed,
                shared_op_names,
                depth + 1,
            )?;
            (guard.target == StrictPcGuardTarget::Scalar).then_some(guard)
        }
        Expr::Apply(op_expr, args) => {
            let Expr::Ident(name, _) = &op_expr.node else {
                return None;
            };
            let op_def = resolve_strict_shared_op(name, ctx, shadowed, shared_op_names)?;
            if op_def.params.len() != args.len() {
                return None;
            }
            let forwarded_alias = forwarded_process_alias(op_def, args, process_alias);
            let mark = shadowed.len();
            shadowed.extend(op_def.params.iter().map(|param| param.name.node.clone()));
            let guard = extract_strict_pc_guard(
                &op_def.body.node,
                pc_name,
                ctx,
                forwarded_alias,
                shadowed,
                shared_op_names,
                depth + 1,
            );
            shadowed.truncate(mark);
            guard.and_then(|guard| match guard.target {
                StrictPcGuardTarget::Scalar if op_def.params.is_empty() => Some(guard),
                StrictPcGuardTarget::Scalar => None,
                StrictPcGuardTarget::SelfIndexed if forwarded_alias.is_some() => Some(guard),
                StrictPcGuardTarget::SelfIndexed => None,
            })
        }
        Expr::Let(_, _) => None,
        Expr::Label(label) => extract_strict_pc_guard(
            &label.body.node,
            pc_name,
            ctx,
            process_alias,
            shadowed,
            shared_op_names,
            depth + 1,
        ),
        Expr::And(lhs, _rhs) => extract_strict_pc_guard(
            &lhs.node,
            pc_name,
            ctx,
            process_alias,
            shadowed,
            shared_op_names,
            depth + 1,
        ),
        Expr::Eq(_, _) => {
            extract_exact_pc_eq_literal(expr, pc_name, ctx, process_alias).and_then(|guard| {
                match guard.target {
                    StrictPcGuardTarget::Scalar => Some(guard),
                    StrictPcGuardTarget::SelfIndexed if process_alias.is_some() => Some(guard),
                    StrictPcGuardTarget::SelfIndexed => None,
                }
            })
        }
        // An EXISTS inside a branch binds only after the containing Or has
        // selected that branch, so its binding cannot authorize Or dispatch.
        Expr::Exists(_, _) => None,
        _ => None,
    }
}

fn extract_exact_pc_eq_literal(
    expr: &Expr,
    pc_name: &str,
    ctx: &EvalCtx,
    process_alias: Option<&str>,
) -> Option<StrictPcGuard> {
    let Expr::Eq(lhs, rhs) = expr else {
        return None;
    };
    extract_exact_pc_reference(&lhs.node, pc_name, ctx, process_alias)
        .and_then(|target| match &rhs.node {
            Expr::String(label) => Some(StrictPcGuard {
                label: label.clone(),
                target,
            }),
            _ => None,
        })
        .or_else(|| {
            extract_exact_pc_reference(&rhs.node, pc_name, ctx, process_alias).and_then(|target| {
                match &lhs.node {
                    Expr::String(label) => Some(StrictPcGuard {
                        label: label.clone(),
                        target,
                    }),
                    _ => None,
                }
            })
        })
}

fn extract_exact_pc_reference(
    expr: &Expr,
    pc_name: &str,
    ctx: &EvalCtx,
    process_alias: Option<&str>,
) -> Option<StrictPcGuardTarget> {
    if is_exact_pc_state_var(expr, pc_name, ctx) {
        return Some(StrictPcGuardTarget::Scalar);
    }
    let Expr::FuncApply(func_expr, arg) = expr else {
        return None;
    };
    (is_exact_pc_state_var(&func_expr.node, pc_name, ctx)
        && matches!(
            (&arg.node, process_alias),
            (Expr::Ident(name, _), Some(alias)) if name == alias
        ))
    .then_some(StrictPcGuardTarget::SelfIndexed)
}

fn is_exact_pc_state_var(expr: &Expr, pc_name: &str, ctx: &EvalCtx) -> bool {
    let Some(expected_idx) = ctx.var_registry().get(pc_name) else {
        return false;
    };
    matches!(
        expr,
        Expr::StateVar(name, raw_idx, name_id)
            if name == pc_name
                && *raw_idx == expected_idx.0
                && (*name_id == tla_core::NameId::INVALID
                    || *name_id == ctx.var_registry().name_id_at(expected_idx))
    )
}

/// Recursively walk the expression tree using the pre-fix loose rules.
fn count_pc_guarded_or_branches_legacy(
    expr: &Expr,
    pc_name: &str,
    ctx: &EvalCtx,
    count: &mut usize,
    depth: usize,
) {
    // Prevent unbounded recursion through mutually-recursive operators.
    if depth > 20 {
        return;
    }
    match expr {
        Expr::Or(a, b) => {
            // Check if each branch has a pc guard
            if extract_pc_guard_resolved(&a.node, pc_name, ctx).is_some() {
                *count += 1;
            }
            if extract_pc_guard_resolved(&b.node, pc_name, ctx).is_some() {
                *count += 1;
            }
            // Also recurse into branches in case they contain nested Or nodes
            count_pc_guarded_or_branches_legacy(&a.node, pc_name, ctx, count, depth + 1);
            count_pc_guarded_or_branches_legacy(&b.node, pc_name, ctx, count, depth + 1);
        }
        Expr::Exists(_bounds, body) => {
            count_pc_guarded_or_branches_legacy(&body.node, pc_name, ctx, count, depth + 1);
        }
        Expr::Let(_defs, body) => {
            count_pc_guarded_or_branches_legacy(&body.node, pc_name, ctx, count, depth + 1);
        }
        Expr::Label(label) => {
            count_pc_guarded_or_branches_legacy(&label.body.node, pc_name, ctx, count, depth + 1);
        }
        Expr::Ident(name, _) => {
            if let Some(op_def) = ctx.get_op(name) {
                count_pc_guarded_or_branches_legacy(
                    &op_def.body.node,
                    pc_name,
                    ctx,
                    count,
                    depth + 1,
                );
            }
        }
        Expr::Apply(op_expr, _args) => {
            if let Expr::Ident(name, _) = &op_expr.node {
                if let Some(op_def) = ctx.get_op(name) {
                    count_pc_guarded_or_branches_legacy(
                        &op_def.body.node,
                        pc_name,
                        ctx,
                        count,
                        depth + 1,
                    );
                }
            }
        }
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::checker_setup::{setup_checker_modules, SetupOptions};
    use crate::config::{Config, ConstantValue};
    use crate::test_support::parse_module;

    fn make_dispatch_table(src: &str) -> Option<PcDispatchTable> {
        let module = parse_module(src);
        let config = Config {
            init: Some("Init".to_string()),
            next: Some("Next".to_string()),
            ..Default::default()
        };
        let mut setup = setup_checker_modules(
            &module,
            &[],
            &config,
            &SetupOptions {
                load_instances: true,
            },
        );
        setup.ctx.resolve_state_vars_in_loaded_ops();
        let next_def = setup.ctx.get_op("Next")?.as_ref().clone();
        let registry = setup.ctx.var_registry().clone();
        detect_pc_dispatch(&next_def, &setup.vars, &registry, &setup.ctx)
    }

    #[test]
    fn test_debug_detection() {
        let src = r#"
---- MODULE PcTestDbg ----
EXTENDS Integers
VARIABLE pc, x

Init == pc = "start" /\ x = 0

A == pc = "start" /\ x' = x + 1 /\ pc' = "middle"
B == pc = "middle" /\ x' = x * 2 /\ pc' = "done"
C == pc = "done" /\ UNCHANGED <<x, pc>>

Next == A \/ B \/ C
====
"#;
        let module = parse_module(src);
        let config = Config {
            init: Some("Init".to_string()),
            next: Some("Next".to_string()),
            ..Default::default()
        };
        let setup = setup_checker_modules(
            &module,
            &[],
            &config,
            &SetupOptions {
                load_instances: true,
            },
        );
        let next_def_arc = setup.ctx.get_op("Next");
        assert!(
            next_def_arc.is_some(),
            "Next operator should be found in ctx"
        );

        let next_def = next_def_arc.unwrap();
        let actions = detect_actions(next_def);
        assert_eq!(
            actions.len(),
            3,
            "should detect 3 disjunct actions (A, B, C)"
        );

        // Verify each action is resolvable in the context
        for action in &actions {
            assert!(
                setup.ctx.get_op(&action.name).is_some(),
                "action '{}' should be resolvable in ctx",
                action.name
            );
        }

        let registry = setup.ctx.var_registry().clone();
        assert!(
            registry.get("pc").is_some(),
            "pc variable should be registered"
        );
    }

    #[test]
    fn test_detect_pluscal_single_process() {
        let src = r#"
---- MODULE PcTest ----
EXTENDS Integers
VARIABLE pc, x

Init == pc = "start" /\ x = 0

A == pc = "start" /\ x' = x + 1 /\ pc' = "middle"
B == pc = "middle" /\ x' = x * 2 /\ pc' = "done"
C == pc = "done" /\ UNCHANGED <<x, pc>>

Next == A \/ B \/ C
====
"#;
        let table = make_dispatch_table(src);
        assert!(table.is_some(), "Should detect PlusCal pattern");
        let table = table.unwrap();
        assert_eq!(table.total_actions, 3);

        // Check dispatch entries
        let start = table.actions_for_pc(&Value::string("start"));
        assert!(start.is_some());
        assert_eq!(start.unwrap().len(), 1);

        let middle = table.actions_for_pc(&Value::string("middle"));
        assert!(middle.is_some());
        assert_eq!(middle.unwrap().len(), 1);

        let done = table.actions_for_pc(&Value::string("done"));
        assert!(done.is_some());
        assert_eq!(done.unwrap().len(), 1);

        // Unknown pc value
        let unknown = table.actions_for_pc(&Value::string("unknown"));
        assert!(unknown.is_none());
    }

    #[test]
    fn test_no_dispatch_for_non_pluscal() {
        let src = r#"
---- MODULE NoPcTest ----
EXTENDS Integers
VARIABLE x, y

Init == x = 0 /\ y = 0

A == x' = x + 1 /\ y' = y
B == x' = x /\ y' = y + 1

Next == A \/ B
====
"#;
        // No `pc` variable at all
        let table = make_dispatch_table(src);
        assert!(
            table.is_none(),
            "Should not detect PlusCal pattern without pc"
        );
    }

    #[test]
    fn test_no_dispatch_when_mixed_guards() {
        let src = r#"
---- MODULE MixedTest ----
EXTENDS Integers
VARIABLE pc, x

Init == pc = "start" /\ x = 0

A == pc = "start" /\ x' = x + 1 /\ pc' = "done"
B == x > 5 /\ x' = 0 /\ pc' = "start"

Next == A \/ B
====
"#;
        // B doesn't have a pc guard
        let table = make_dispatch_table(src);
        assert!(
            table.is_none(),
            "Should not detect when not all actions have pc guards"
        );
    }

    #[test]
    fn test_dispatch_with_shared_pc_value() {
        let src = r#"
---- MODULE SharedPcTest ----
EXTENDS Integers
VARIABLE pc, x

Init == pc = "start" /\ x = 0

A == pc = "start" /\ x < 10 /\ x' = x + 1 /\ pc' = "start"
B == pc = "start" /\ x >= 10 /\ x' = 0 /\ pc' = "done"
C == pc = "done" /\ UNCHANGED <<x, pc>>

Next == A \/ B \/ C
====
"#;
        let table = make_dispatch_table(src);
        assert!(table.is_some());
        let table = table.unwrap();

        // "start" should map to both A (idx 0) and B (idx 1)
        let start = table.actions_for_pc(&Value::string("start"));
        assert!(start.is_some());
        assert_eq!(start.unwrap().len(), 2);

        let done = table.actions_for_pc(&Value::string("done"));
        assert!(done.is_some());
        assert_eq!(done.unwrap().len(), 1);
    }

    #[test]
    fn test_bounded_multiprocess_barriers_style_fanout_dispatch() {
        let src = r#"
---- MODULE BarriersFanout ----
EXTENDS Integers
CONSTANT N
VARIABLES pc, x

ProcSet == 1..N

Init == pc = [self \in ProcSet |-> "a0"] /\ x = 0

a0(self) == /\ pc[self] = "a0"
            /\ pc' = [pc EXCEPT ![self] = "a1"]
            /\ x' = x + 1

a1(self) == /\ pc[self] = "a1"
            /\ pc' = [pc EXCEPT ![self] = "a2"]
            /\ x' = x + 1

a2(self) == /\ pc[self] = "a2"
            /\ pc' = [pc EXCEPT ![self] = "a0"]
            /\ x' = x

proc(self) == a0(self) \/ a1(self) \/ a2(self)

Next == \E self \in 1..N : proc(self)
====
"#;
        let module = parse_module(src);
        let mut config = Config {
            init: Some("Init".to_string()),
            next: Some("Next".to_string()),
            ..Default::default()
        };
        config
            .constants
            .insert("N".to_string(), ConstantValue::Value("2".to_string()));
        let mut setup = setup_checker_modules(
            &module,
            &[],
            &config,
            &SetupOptions {
                load_instances: true,
            },
        );
        crate::constants::bind_constants_from_config(&mut setup.ctx, &config)
            .expect("test constants should bind");
        setup.ctx.resolve_state_vars_in_loaded_ops();
        let next_def = setup.ctx.get_op("Next").unwrap().as_ref().clone();
        let registry = setup.ctx.var_registry().clone();

        let table = detect_pc_dispatch(&next_def, &setup.vars, &registry, &setup.ctx)
            .expect("bounded PlusCal process actions should fan out");
        assert_eq!(table.total_actions, 6);
        assert_eq!(table.dispatch.len(), 3);
        assert!(
            table.process_dispatch.is_some(),
            "bounded self-specialized actions should build direct process buckets"
        );

        for label in ["a0", "a1", "a2"] {
            let indices = table
                .actions_for_pc(&Value::string(label))
                .unwrap_or_else(|| panic!("missing dispatch bucket {label}"));
            assert_eq!(indices.len(), 2, "one action per process for {label}");
        }

        let mut pc_builder = crate::value::FuncBuilder::new();
        pc_builder.insert(Value::int(1), Value::string("a0"));
        pc_builder.insert(Value::int(2), Value::string("a2"));
        let current_pc = Value::Func(Rp::new(pc_builder.build()));
        let matching = table
            .action_indices_for_current_pc(&current_pc)
            .expect("pc function should select matching process actions");
        assert_eq!(matching.len(), 2);

        let selected_names: Vec<_> = matching
            .iter()
            .map(|idx| table.actions[*idx].name.as_deref())
            .collect();
        assert!(selected_names.contains(&Some("a0")));
        assert!(selected_names.contains(&Some("a2")));

        let mut partial_pc_builder = crate::value::FuncBuilder::new();
        partial_pc_builder.insert(Value::int(1), Value::string("a0"));
        let partial_pc = Value::Func(Rp::new(partial_pc_builder.build()));
        assert!(
            table.action_indices_for_current_pc(&partial_pc).is_none(),
            "missing pc[self] domain entries must fall back to full enumeration"
        );

        let mut unknown_label_pc_builder = crate::value::FuncBuilder::new();
        unknown_label_pc_builder.insert(Value::int(1), Value::string("blocked"));
        unknown_label_pc_builder.insert(Value::int(2), Value::string("a2"));
        let unknown_label_pc = Value::Func(Rp::new(unknown_label_pc_builder.build()));
        let matching = table
            .action_indices_for_current_pc(&unknown_label_pc)
            .expect("complete pc function should use direct process buckets");
        assert_eq!(matching.len(), 1);
        assert_eq!(table.actions[matching[0]].name.as_deref(), Some("a2"));
    }

    #[test]
    fn test_full_dispatch_uses_innermost_shadowed_self_binding() {
        let src = r#"
---- MODULE NestedSelfDispatch ----
VARIABLE pc

Init == pc = [i \in {1, 2} |-> "a"]
A(self) == pc[self] = "a" /\ UNCHANGED pc
B(self) == pc[self] = "b" /\ UNCHANGED pc
p(self) == A(self) \/ B(self)
Next == \E self \in {1} : \E self \in {2} : p(self)
====
"#;
        let module = parse_module(src);
        let config = Config {
            init: Some("Init".to_string()),
            next: Some("Next".to_string()),
            ..Default::default()
        };
        let mut setup = setup_checker_modules(
            &module,
            &[],
            &config,
            &SetupOptions {
                load_instances: true,
            },
        );
        setup.ctx.resolve_state_vars_in_loaded_ops();
        let next_def = setup.ctx.get_op("Next").unwrap().as_ref().clone();
        let registry = setup.ctx.var_registry().clone();
        let table = detect_pc_dispatch(&next_def, &setup.vars, &registry, &setup.ctx)
            .expect("nested finite process binders should build a full table");

        assert!(table.actions.iter().all(|action| {
            action_self_binding(action).is_some_and(|self_value| self_value == &Value::int(2))
        }));
        let mut pc_builder = crate::value::FuncBuilder::new();
        pc_builder.insert(Value::int(2), Value::string("a"));
        let selected = table
            .action_indices_for_current_pc(&Value::Func(Rp::new(pc_builder.build())))
            .expect("the process table must use the innermost self=2 binding");
        assert_eq!(selected.len(), 1);
        assert_eq!(table.actions[selected[0]].name.as_deref(), Some("A"));
    }

    #[test]
    fn test_full_dispatch_rejects_locally_shadowed_pc_binding() {
        let src = r#"
---- MODULE LocalPcBindingDispatch ----
VARIABLE pc

Init == pc = "a"
A == pc = "a" /\ UNCHANGED pc
B == pc = "b" /\ UNCHANGED pc
Next == \E pc \in {"a", "b"} : A \/ B
====
"#;
        let module = parse_module(src);
        let config = Config {
            init: Some("Init".to_string()),
            next: Some("Next".to_string()),
            ..Default::default()
        };
        let mut setup = setup_checker_modules(
            &module,
            &[],
            &config,
            &SetupOptions {
                load_instances: true,
            },
        );
        setup.ctx.resolve_state_vars_in_loaded_ops();
        let next_def = setup.ctx.get_op("Next").unwrap().as_ref().clone();
        let registry = setup.ctx.var_registry().clone();

        assert!(
            detect_pc_dispatch(&next_def, &setup.vars, &registry, &setup.ctx).is_none(),
            "a split-action witness named pc overrides StateVar pc during evaluation"
        );
    }

    /// Test that `or_branch_pc_guard_mismatches` correctly identifies branches
    /// with non-matching pc guards.
    #[test]
    fn test_or_branch_guard_mismatch_detection() {
        let src = r#"
---- MODULE PcMismatch ----
EXTENDS Integers
VARIABLE pc, x

Init == pc = "start" /\ x = 0

A == pc = "start" /\ x' = x + 1 /\ pc' = "middle"
B == pc = "middle" /\ x' = x * 2 /\ pc' = "done"
C == pc = "done" /\ UNCHANGED <<x, pc>>

Next == A \/ B \/ C
====
"#;
        let module = parse_module(src);
        let config = Config {
            init: Some("Init".to_string()),
            next: Some("Next".to_string()),
            ..Default::default()
        };
        let mut setup = setup_checker_modules(
            &module,
            &[],
            &config,
            &SetupOptions {
                load_instances: true,
            },
        );
        setup.ctx.resolve_state_vars_in_loaded_ops();

        // Test that or_branch_pc_guard_mismatches works with the ctx
        let current_pc_start = Value::string("start");
        let current_pc_middle = Value::string("middle");

        // Get the A operator body — its first conjunct is `pc = "start"`
        let a_def = setup.ctx.get_op("A").unwrap();
        // A's body has pc = "start" guard — should NOT mismatch when pc = "start"
        assert!(
            !or_branch_pc_guard_mismatches(
                &tla_core::ast::Expr::Ident("A".to_string(), tla_core::NameId::INVALID),
                &current_pc_start,
                &setup.ctx,
            ),
            "A should match when pc = start"
        );
        // A's body has pc = "start" guard — should mismatch when pc = "middle"
        assert!(
            or_branch_pc_guard_mismatches(
                &tla_core::ast::Expr::Ident("A".to_string(), tla_core::NameId::INVALID),
                &current_pc_middle,
                &setup.ctx,
            ),
            "A should NOT match when pc = middle"
        );

        // B's body has pc = "middle" guard
        assert!(
            or_branch_pc_guard_mismatches(
                &tla_core::ast::Expr::Ident("B".to_string(), tla_core::NameId::INVALID),
                &current_pc_start,
                &setup.ctx,
            ),
            "B should NOT match when pc = start"
        );
        assert!(
            !or_branch_pc_guard_mismatches(
                &tla_core::ast::Expr::Ident("B".to_string(), tla_core::NameId::INVALID),
                &current_pc_middle,
                &setup.ctx,
            ),
            "B should match when pc = middle"
        );

        let _ = a_def; // suppress unused warning
    }

    /// Test that guard hoisting produces the same successors as the non-optimized path.
    ///
    /// Part of #3923: correctness verification for pc-guard hoisting.
    /// Enumerates successors with and without guard hoisting and verifies
    /// the same set of successor fingerprints is produced.
    #[test]
    fn test_guard_hoisting_same_successors_as_unoptimized() {
        let src = r#"
---- MODULE PcHoistParity ----
EXTENDS Integers
VARIABLE pc, x

Init == pc = "start" /\ x = 0

A == pc = "start" /\ x' = x + 1 /\ pc' = "middle"
B == pc = "middle" /\ x' = x * 2 /\ pc' = "done"
C == pc = "done" /\ UNCHANGED <<x, pc>>

Next == A \/ B \/ C
====
"#;
        let module = parse_module(src);
        let config = Config {
            init: Some("Init".to_string()),
            next: Some("Next".to_string()),
            ..Default::default()
        };
        let mut setup = setup_checker_modules(
            &module,
            &[],
            &config,
            &SetupOptions {
                load_instances: true,
            },
        );
        setup.ctx.resolve_state_vars_in_loaded_ops();

        let registry = setup.ctx.var_registry().clone();
        let pc_idx = registry.get("pc").expect("pc variable should exist");

        // Detect the pc dispatch table
        let next_def = setup.ctx.get_op("Next").unwrap().as_ref().clone();
        let table = detect_pc_dispatch(&next_def, &setup.vars, &registry, &setup.ctx);
        assert!(table.is_some(), "Should detect PlusCal pattern");

        // Test with pc = "start": should get successor from A only
        {
            let state = crate::state::ArrayState::from_values(vec![
                Value::string("start"), // pc
                Value::int(0),          // x
            ]);

            // Without hoisting — bind state env so evaluator can read state vars
            let mut ctx1 = setup.ctx.clone();
            let _guard1 = ctx1.bind_state_env_guard(state.env_ref());
            let succs_no_hoist = crate::enumerate::enumerate_successors_array_with_tir(
                &mut ctx1,
                &next_def,
                &state,
                &setup.vars,
                None,
            )
            .unwrap();

            // With hoisting — bind state env for the same reason
            let mut ctx2 = setup.ctx.clone();
            let _guard2 = ctx2.bind_state_env_guard(state.env_ref());
            let diffs_with_hoist =
                crate::enumerate::successor_engine_test_helpers::run_with_pc_hoist(
                    &mut ctx2,
                    &next_def.body,
                    &state,
                    &setup.vars,
                    &registry,
                    pc_idx,
                )
                .unwrap();

            // Compare: same number of successors
            assert_eq!(
                succs_no_hoist.len(),
                diffs_with_hoist.len(),
                "pc=start: hoist and non-hoist should produce same successor count"
            );
        }

        // Test with pc = "middle": should get successor from B only
        {
            let state = crate::state::ArrayState::from_values(vec![
                Value::string("middle"), // pc
                Value::int(5),           // x
            ]);

            let mut ctx1 = setup.ctx.clone();
            let _guard1 = ctx1.bind_state_env_guard(state.env_ref());
            let succs_no_hoist = crate::enumerate::enumerate_successors_array_with_tir(
                &mut ctx1,
                &next_def,
                &state,
                &setup.vars,
                None,
            )
            .unwrap();

            let mut ctx2 = setup.ctx.clone();
            let _guard2 = ctx2.bind_state_env_guard(state.env_ref());
            let diffs_with_hoist =
                crate::enumerate::successor_engine_test_helpers::run_with_pc_hoist(
                    &mut ctx2,
                    &next_def.body,
                    &state,
                    &setup.vars,
                    &registry,
                    pc_idx,
                )
                .unwrap();

            assert_eq!(
                succs_no_hoist.len(),
                diffs_with_hoist.len(),
                "pc=middle: hoist and non-hoist should produce same successor count"
            );
        }

        // Test with pc = "done": should get successor from C (UNCHANGED)
        {
            let state = crate::state::ArrayState::from_values(vec![
                Value::string("done"), // pc
                Value::int(10),        // x
            ]);

            let mut ctx1 = setup.ctx.clone();
            let _guard1 = ctx1.bind_state_env_guard(state.env_ref());
            let succs_no_hoist = crate::enumerate::enumerate_successors_array_with_tir(
                &mut ctx1,
                &next_def,
                &state,
                &setup.vars,
                None,
            )
            .unwrap();

            let mut ctx2 = setup.ctx.clone();
            let _guard2 = ctx2.bind_state_env_guard(state.env_ref());
            let diffs_with_hoist =
                crate::enumerate::successor_engine_test_helpers::run_with_pc_hoist(
                    &mut ctx2,
                    &next_def.body,
                    &state,
                    &setup.vars,
                    &registry,
                    pc_idx,
                )
                .unwrap();

            assert_eq!(
                succs_no_hoist.len(),
                diffs_with_hoist.len(),
                "pc=done: hoist and non-hoist should produce same successor count"
            );
        }
    }

    /// Test that `extract_pc_guard_resolved` detects multi-process PlusCal patterns
    /// where the guard is `pc[self] = "label"` inside operator bodies.
    ///
    /// Part of #3805: multi-process PlusCal pc-dispatch.
    #[test]
    fn test_multi_process_guard_extraction() {
        let src = r#"
---- MODULE MultiProc ----
EXTENDS Integers
CONSTANT Procs
VARIABLE pc, x

Init == pc = [p \in Procs |-> "start"] /\ x = [p \in Procs |-> 0]

A(self) == pc[self] = "start" /\ x' = [x EXCEPT ![self] = x[self] + 1] /\ pc' = [pc EXCEPT ![self] = "done"]
B(self) == pc[self] = "done" /\ UNCHANGED <<x, pc>>

p(self) == A(self) \/ B(self)

Next == \E self \in Procs : p(self)
====
"#;
        let module = parse_module(src);
        let config = Config {
            init: Some("Init".to_string()),
            next: Some("Next".to_string()),
            ..Default::default()
        };
        let setup = setup_checker_modules(
            &module,
            &[],
            &config,
            &SetupOptions {
                load_instances: true,
            },
        );

        // Verify that extract_pc_guard_resolved can extract guard labels from
        // multi-process operator references like A(self) whose body starts with
        // pc[self] = "start".
        let a_ident = tla_core::ast::Expr::Apply(
            Box::new(tla_core::Spanned {
                node: tla_core::ast::Expr::Ident("A".to_string(), tla_core::NameId::INVALID),
                span: tla_core::Span::dummy(),
            }),
            vec![tla_core::Spanned {
                node: tla_core::ast::Expr::Ident("self".to_string(), tla_core::NameId::INVALID),
                span: tla_core::Span::dummy(),
            }],
        );
        let guard_a = extract_pc_guard_resolved(&a_ident, "pc", &setup.ctx);
        assert_eq!(
            guard_a,
            Some("start".to_string()),
            "A(self) should have pc guard 'start'"
        );

        let b_ident = tla_core::ast::Expr::Apply(
            Box::new(tla_core::Spanned {
                node: tla_core::ast::Expr::Ident("B".to_string(), tla_core::NameId::INVALID),
                span: tla_core::Span::dummy(),
            }),
            vec![tla_core::Spanned {
                node: tla_core::ast::Expr::Ident("self".to_string(), tla_core::NameId::INVALID),
                span: tla_core::Span::dummy(),
            }],
        );
        let guard_b = extract_pc_guard_resolved(&b_ident, "pc", &setup.ctx);
        assert_eq!(
            guard_b,
            Some("done".to_string()),
            "B(self) should have pc guard 'done'"
        );
    }

    /// Test that `or_branch_pc_guard_mismatches` correctly skips multi-process
    /// PlusCal branches when `pc` is a function and `self` is bound.
    ///
    /// Part of #3805: multi-process PlusCal pc-dispatch.
    #[test]
    fn test_multi_process_guard_mismatch() {
        use std::sync::Arc;

        let src = r#"
---- MODULE MultiProcGuard ----
EXTENDS Integers
CONSTANT Procs
VARIABLE pc, x

Init == pc = [p \in Procs |-> "start"] /\ x = [p \in Procs |-> 0]

A(self) == pc[self] = "start" /\ x' = [x EXCEPT ![self] = x[self] + 1] /\ pc' = [pc EXCEPT ![self] = "done"]
B(self) == pc[self] = "done" /\ UNCHANGED <<x, pc>>

p(self) == A(self) \/ B(self)

Next == \E self \in Procs : p(self)
====
"#;
        let module = parse_module(src);
        let config = Config {
            init: Some("Init".to_string()),
            next: Some("Next".to_string()),
            ..Default::default()
        };
        let mut setup = setup_checker_modules(
            &module,
            &[],
            &config,
            &SetupOptions {
                load_instances: true,
            },
        );
        setup.ctx.resolve_state_vars_in_loaded_ops();

        // Build a pc function: [p1 |-> "start", p2 |-> "done"]
        let mut fb = crate::value::FuncBuilder::new();
        fb.insert(Value::string("p1"), Value::string("start"));
        fb.insert(Value::string("p2"), Value::string("done"));
        let pc_func = Value::Func(Rp::new(fb.build()));

        let a_expr = tla_core::ast::Expr::Apply(
            Box::new(tla_core::Spanned {
                node: tla_core::ast::Expr::Ident("A".to_string(), tla_core::NameId::INVALID),
                span: tla_core::Span::dummy(),
            }),
            vec![tla_core::Spanned {
                node: tla_core::ast::Expr::Ident("self".to_string(), tla_core::NameId::INVALID),
                span: tla_core::Span::dummy(),
            }],
        );
        let cache = new_pc_guard_label_cache();

        // Bind self = "p1" (pc["p1"] = "start") — A should match, B should not
        {
            let mut ctx = setup.ctx.clone();
            ctx.push_binding(Arc::from("self"), Value::string("p1"));

            // A(self) has guard "start", pc["p1"] = "start" → should NOT mismatch
            assert!(
                !or_branch_pc_guard_mismatches(&a_expr, &pc_func, &ctx),
                "A(self) should match when pc[p1] = start"
            );
            assert!(
                !or_branch_pc_guard_mismatches_cached(&a_expr, &pc_func, &ctx, &cache),
                "cached A(self) should match when pc[p1] = start"
            );
        }
        assert_eq!(cache.borrow().len(), 1);

        // Bind self = "p2" (pc["p2"] = "done") — A should NOT match, B should
        {
            let mut ctx = setup.ctx.clone();
            ctx.push_binding(Arc::from("self"), Value::string("p2"));

            // A(self) has guard "start", pc["p2"] = "done" → should mismatch
            assert!(
                or_branch_pc_guard_mismatches(&a_expr, &pc_func, &ctx),
                "A(self) should NOT match when pc[p2] = done"
            );
            assert!(
                or_branch_pc_guard_mismatches_cached(&a_expr, &pc_func, &ctx, &cache),
                "cached A(self) should still honor the current self binding"
            );
        }
        assert_eq!(cache.borrow().len(), 1);
    }

    /// Once one safe Or admits function-valued pc hoisting, unrelated Ors in
    /// the same Next expression must not be compared through the wrong key.
    #[test]
    fn test_multi_process_runtime_rejects_mixed_pc_key() {
        use std::sync::Arc;

        let src = r#"
---- MODULE MixedProcGuardKeys ----
CONSTANT Procs
VARIABLE pc, x

Init == pc = [p \in Procs |-> "start"] /\ x = 0

A(self) == pc[self] = "start" /\ UNCHANGED <<pc, x>>
B(self) == pc[self] = "done"  /\ UNCHANGED <<pc, x>>
C(t)    == pc[t]    = "done"  /\ UNCHANGED <<pc, x>>
D(t)    == pc[t]    = "start" /\ UNCHANGED <<pc, x>>

p(self) == A(self) \/ B(self)
q(t) == C(t) \/ D(t)

Next == \E self \in Procs : p(self) \/ (\E t \in Procs : q(t))
====
"#;
        let module = parse_module(src);
        let config = Config {
            init: Some("Init".to_string()),
            next: Some("Next".to_string()),
            ..Default::default()
        };
        let mut setup = setup_checker_modules(
            &module,
            &[],
            &config,
            &SetupOptions {
                load_instances: true,
            },
        );
        setup.ctx.resolve_state_vars_in_loaded_ops();
        let next_def = setup.ctx.get_op("Next").unwrap().as_ref().clone();
        assert!(
            spec_has_pc_guards(&next_def, &setup.ctx),
            "the nested p(self) Or is a genuinely resolvable admission site"
        );

        let c_of_t = Expr::Apply(
            Box::new(tla_core::Spanned::dummy(Expr::Ident(
                "C".to_string(),
                tla_core::NameId::INVALID,
            ))),
            vec![tla_core::Spanned::dummy(Expr::Ident(
                "t".to_string(),
                tla_core::NameId::INVALID,
            ))],
        );
        let mut fb = crate::value::FuncBuilder::new();
        fb.insert(Value::string("p1"), Value::string("start"));
        fb.insert(Value::string("p2"), Value::string("done"));
        let pc_func = Value::Func(Rp::new(fb.build()));

        let mut ctx = setup.ctx.clone();
        ctx.push_binding(Arc::from("self"), Value::string("p1"));
        ctx.push_binding(Arc::from("t"), Value::string("p2"));
        assert!(
            !or_branch_pc_guard_mismatches(&c_of_t, &pc_func, &ctx),
            "pc[t] must fall through instead of being compared against pc[self]"
        );
        assert!(
            !or_branch_pc_guard_mismatches_cached(
                &c_of_t,
                &pc_func,
                &ctx,
                &new_pc_guard_label_cache(),
            ),
            "the cached runtime matcher must use the same strict key rule"
        );
    }

    /// Test that `spec_has_pc_guards` detects multi-process PlusCal specs.
    ///
    /// Part of #3805: multi-process PlusCal guard hoisting.
    #[test]
    fn test_spec_has_pc_guards_multiprocess() {
        let src = r#"
---- MODULE MultiProcDetect ----
EXTENDS Integers
CONSTANT Procs
VARIABLE pc, x

Init == pc = [p \in Procs |-> "start"] /\ x = [p \in Procs |-> 0]

A(self) == pc[self] = "start" /\ x' = [x EXCEPT ![self] = x[self] + 1] /\ pc' = [pc EXCEPT ![self] = "done"]
B(self) == pc[self] = "done" /\ UNCHANGED <<x, pc>>

p(self) == A(self) \/ B(self)

Next == \E self \in Procs : p(self)
====
"#;
        let module = parse_module(src);
        let config = Config {
            init: Some("Init".to_string()),
            next: Some("Next".to_string()),
            ..Default::default()
        };
        let mut setup = setup_checker_modules(
            &module,
            &[],
            &config,
            &SetupOptions {
                load_instances: true,
            },
        );
        setup.ctx.resolve_state_vars_in_loaded_ops();
        let next_def = setup.ctx.get_op("Next").unwrap().as_ref().clone();

        // Without a concrete process-set constant, action splitting cannot
        // build a complete fanout table and must leave runtime enumeration intact.
        let registry = setup.ctx.var_registry().clone();
        let table = detect_pc_dispatch(&next_def, &setup.vars, &registry, &setup.ctx);
        assert!(
            table.is_none(),
            "Full dispatch table should not be built for an unconfigured process set"
        );

        // But spec_has_pc_guards should detect the pattern
        assert!(
            spec_has_pc_guards(&next_def, &setup.ctx),
            "spec_has_pc_guards should detect multi-process PlusCal pattern"
        );
    }

    /// A function-valued `pc` is not enough to enable the runtime hoist: its
    /// current implementation can resolve only a binding named `self`, and the
    /// outer Or below runs before either `w` or `r` is bound.
    #[test]
    fn test_spec_has_pc_guards_rejects_unresolvable_process_binders() {
        let src = r#"
---- MODULE MultiProcUnresolvableBinders ----
CONSTANT Writers, Readers
VARIABLE pc, x

Init == pc = [a \in Writers \union Readers |-> "Advance"] /\ x = 0

BeginWrite(w) == pc[w] = "Advance" /\ UNCHANGED <<pc, x>>
EndWrite(w)   == pc[w] = "Access"  /\ UNCHANGED <<pc, x>>
BeginRead(r)  == pc[r] = "Advance" /\ UNCHANGED <<pc, x>>
EndRead(r)    == pc[r] = "Access"  /\ UNCHANGED <<pc, x>>

Next ==
  \/ \E w \in Writers : BeginWrite(w)
  \/ \E w \in Writers : EndWrite(w)
  \/ \E r \in Readers : BeginRead(r)
  \/ \E r \in Readers : EndRead(r)
====
"#;
        let module = parse_module(src);
        let config = Config {
            init: Some("Init".to_string()),
            next: Some("Next".to_string()),
            ..Default::default()
        };
        let mut setup = setup_checker_modules(
            &module,
            &[],
            &config,
            &SetupOptions {
                load_instances: true,
            },
        );
        setup.ctx.resolve_state_vars_in_loaded_ops();
        let next_def = setup.ctx.get_op("Next").unwrap().as_ref().clone();

        assert!(
            spec_has_pc_guards_legacy(&next_def, &setup.ctx),
            "the legacy loose detector should reproduce the false positive"
        );
        assert!(
            !spec_has_pc_guards(&next_def, &setup.ctx),
            "w/r are not resolvable through the runtime's hard-coded self lookup"
        );
    }

    /// Strict admission and the runtime matcher must inspect the final config
    /// replacement target, not the raw operator body named at the call site.
    #[test]
    fn test_spec_has_pc_guards_resolves_config_replacement() {
        use std::sync::Arc;

        let src = r#"
---- MODULE MultiProcReplacement ----
CONSTANT Procs
VARIABLE pc, x

Init == pc = [p \in Procs |-> "start"] /\ x = 0

A(self)  == pc[self] = "start" /\ UNCHANGED <<pc, x>>
B(self)  == pc[self] = "done"  /\ UNCHANGED <<pc, x>>
RA(self) == pc[self] = "done"  /\ UNCHANGED <<pc, x>>
RB(self) == pc[self] = "start" /\ UNCHANGED <<pc, x>>
p(self) == A(self) \/ B(self)

Next == \E self \in Procs : p(self)
====
"#;
        let module = parse_module(src);
        let mut config = Config {
            init: Some("Init".to_string()),
            next: Some("Next".to_string()),
            ..Default::default()
        };
        config.add_constant(
            "A".to_string(),
            ConstantValue::Replacement("RA".to_string()),
        );
        config.add_constant(
            "B".to_string(),
            ConstantValue::Replacement("RB".to_string()),
        );
        let mut setup = setup_checker_modules(
            &module,
            &[],
            &config,
            &SetupOptions {
                load_instances: true,
            },
        );
        crate::constants::bind_constants_from_config(&mut setup.ctx, &config)
            .expect("operator replacements should bind");
        setup.ctx.resolve_state_vars_in_loaded_ops();
        let next_def = setup.ctx.get_op("Next").unwrap().as_ref().clone();
        assert!(spec_has_pc_guards(&next_def, &setup.ctx));

        let a_of_self = Expr::Apply(
            Box::new(tla_core::Spanned::dummy(Expr::Ident(
                "A".to_string(),
                tla_core::NameId::INVALID,
            ))),
            vec![tla_core::Spanned::dummy(Expr::Ident(
                "self".to_string(),
                tla_core::NameId::INVALID,
            ))],
        );
        let mut fb = crate::value::FuncBuilder::new();
        fb.insert(Value::string("p1"), Value::string("start"));
        let pc_func = Value::Func(Rp::new(fb.build()));
        let mut ctx = setup.ctx.clone();
        ctx.push_binding(Arc::from("self"), Value::string("p1"));

        assert!(
            or_branch_pc_guard_mismatches(&a_of_self, &pc_func, &ctx),
            "A <- RA must compare RA's 'done' guard, not raw A's 'start' guard"
        );
    }

    #[test]
    fn test_full_dispatch_resolves_config_replacement_for_bounded_processes() {
        let src = r#"
---- MODULE BoundedMultiProcReplacement ----
CONSTANT Procs
VARIABLE pc

Init == pc = [p \in Procs |-> "raw-a"]

A(self)  == pc[self] = "raw-a"         /\ UNCHANGED pc
B(self)  == pc[self] = "raw-b"         /\ UNCHANGED pc
RA(self) == pc[self] = "replacement-a" /\ UNCHANGED pc
RB(self) == pc[self] = "replacement-b" /\ UNCHANGED pc
p(self) == A(self) \/ B(self)

Next == \E self \in Procs : p(self)
====
"#;
        let module = parse_module(src);
        let mut config = Config {
            init: Some("Init".to_string()),
            next: Some("Next".to_string()),
            ..Default::default()
        };
        config.add_constant(
            "Procs".to_string(),
            ConstantValue::Value("{1, 2}".to_string()),
        );
        config.add_constant(
            "A".to_string(),
            ConstantValue::Replacement("RA".to_string()),
        );
        config.add_constant(
            "B".to_string(),
            ConstantValue::Replacement("RB".to_string()),
        );
        let mut setup = setup_checker_modules(
            &module,
            &[],
            &config,
            &SetupOptions {
                load_instances: true,
            },
        );
        crate::constants::bind_constants_from_config(&mut setup.ctx, &config)
            .expect("bounded process set and replacements should bind");
        setup.ctx.resolve_state_vars_in_loaded_ops();
        let next_def = setup.ctx.get_op("Next").unwrap().as_ref().clone();
        let registry = setup.ctx.var_registry().clone();

        let table = detect_pc_dispatch(&next_def, &setup.vars, &registry, &setup.ctx)
            .expect("finite Procs should exercise the full dispatch-table path");
        assert_eq!(table.total_actions, 4);
        assert_eq!(
            table
                .actions_for_pc(&Value::string("replacement-a"))
                .map(|indices| indices.len()),
            Some(2)
        );
        assert_eq!(
            table
                .actions_for_pc(&Value::string("replacement-b"))
                .map(|indices| indices.len()),
            Some(2)
        );
        assert!(table.actions_for_pc(&Value::string("raw-a")).is_none());
        assert!(table.actions_for_pc(&Value::string("raw-b")).is_none());
    }

    /// Test that `spec_has_pc_guards` returns false for non-PlusCal specs.
    #[test]
    fn test_spec_has_pc_guards_non_pluscal() {
        let src = r#"
---- MODULE NoPcGuards ----
EXTENDS Integers
VARIABLE x, y

Init == x = 0 /\ y = 0
A == x' = x + 1 /\ y' = y
B == x' = x /\ y' = y + 1

Next == A \/ B
====
"#;
        let module = parse_module(src);
        let config = Config {
            init: Some("Init".to_string()),
            next: Some("Next".to_string()),
            ..Default::default()
        };
        let setup = setup_checker_modules(
            &module,
            &[],
            &config,
            &SetupOptions {
                load_instances: true,
            },
        );
        let next_def = setup.ctx.get_op("Next").unwrap().as_ref().clone();
        assert!(
            !spec_has_pc_guards(&next_def, &setup.ctx),
            "Non-PlusCal spec should not have pc guards"
        );
    }

    #[test]
    fn test_strict_runtime_rejects_raw_or_wrong_slot_pc() {
        use std::sync::Arc;

        let src = r#"
---- MODULE StrictPcIdentity ----
VARIABLES pc, x

Init == TRUE
Next == TRUE
====
"#;
        let module = parse_module(src);
        let config = Config {
            init: Some("Init".to_string()),
            next: Some("Next".to_string()),
            ..Default::default()
        };
        let setup = setup_checker_modules(
            &module,
            &[],
            &config,
            &SetupOptions {
                load_instances: true,
            },
        );

        let guard_for = |pc_expr| {
            Expr::Eq(
                Box::new(tla_core::Spanned::dummy(Expr::FuncApply(
                    Box::new(tla_core::Spanned::dummy(pc_expr)),
                    Box::new(tla_core::Spanned::dummy(Expr::Ident(
                        "self".to_string(),
                        tla_core::NameId::INVALID,
                    ))),
                ))),
                Box::new(tla_core::Spanned::dummy(Expr::String("done".to_string()))),
            )
        };

        let raw_pc_guard = guard_for(Expr::Ident("pc".to_string(), tla_core::NameId::INVALID));
        let registry = setup.ctx.var_registry();
        let pc_slot = registry.get("pc").expect("pc should have a state slot");
        let wrong_slot = registry.get("x").expect("x should have a state slot");
        let wrong_slot_guard = guard_for(Expr::StateVar(
            "pc".to_string(),
            wrong_slot.0,
            tla_core::NameId::INVALID,
        ));
        let wrong_name_id_guard = guard_for(Expr::StateVar(
            "pc".to_string(),
            pc_slot.0,
            registry.name_id_at(wrong_slot),
        ));
        let exact_guard = guard_for(Expr::StateVar(
            "pc".to_string(),
            pc_slot.0,
            registry.name_id_at(pc_slot),
        ));

        let mut fb = crate::value::FuncBuilder::new();
        fb.insert(Value::string("p1"), Value::string("start"));
        let current_pc = Value::Func(Rp::new(fb.build()));
        let mut ctx = setup.ctx.clone();
        ctx.push_binding(Arc::from("self"), Value::string("p1"));

        assert!(
            !or_branch_pc_guard_mismatches(&raw_pc_guard, &current_pc, &ctx),
            "an unresolved Ident named pc may be shadowed and must fail closed"
        );
        assert!(
            !or_branch_pc_guard_mismatches(&wrong_slot_guard, &current_pc, &ctx),
            "a StateVar named pc with another variable's slot must fail closed"
        );
        assert!(
            !or_branch_pc_guard_mismatches(&wrong_name_id_guard, &current_pc, &ctx),
            "a StateVar named pc with another variable's valid NameId must fail closed"
        );

        let pc_substitution = tla_core::ast::Substitution {
            from: tla_core::Spanned::dummy("pc".to_string()),
            to: tla_core::Spanned::dummy(Expr::String("done".to_string())),
        };
        let instance_ctx = ctx.with_instance_substitutions(vec![pc_substitution.clone()]);
        assert!(
            !or_branch_pc_guard_mismatches(&exact_guard, &current_pc, &instance_ctx),
            "an INSTANCE overlay can change StateVar pc evaluation and must disable hoisting"
        );
        let call_by_name_ctx = ctx.with_call_by_name_subs(vec![pc_substitution]);
        assert!(
            !or_branch_pc_guard_mismatches(&exact_guard, &current_pc, &call_by_name_ctx),
            "a call-by-name overlay can change StateVar pc evaluation and must disable hoisting"
        );
    }

    #[test]
    fn test_spec_has_pc_guards_rejects_formal_pc_shadow() {
        let src = r#"
---- MODULE FormalPcShadow ----
CONSTANT Procs
VARIABLE pc

Init == TRUE
A(self, pc) == pc[self] = "a"
B(self, pc) == pc[self] = "b"
p(self) == A(self, pc) \/ B(self, pc)
Next == \E self \in Procs : p(self)
====
"#;
        let module = parse_module(src);
        let config = Config {
            init: Some("Init".to_string()),
            next: Some("Next".to_string()),
            ..Default::default()
        };
        let mut setup = setup_checker_modules(
            &module,
            &[],
            &config,
            &SetupOptions {
                load_instances: true,
            },
        );
        setup.ctx.resolve_state_vars_in_loaded_ops();
        let next_def = setup.ctx.get_op("Next").unwrap().as_ref().clone();

        assert!(
            !spec_has_pc_guards(&next_def, &setup.ctx),
            "a formal named pc stays an Ident and must not certify the shared state variable"
        );
    }

    #[test]
    fn test_spec_has_pc_guards_rejects_let_local_operator_collision() {
        let src = r#"
---- MODULE LetPcGuardCollision ----
CONSTANT Procs
VARIABLE pc

Init == TRUE
A(self) == pc[self] = "a"
B(self) == pc[self] = "b"
p(self) ==
    LET A(self) == TRUE
        B(self) == TRUE
    IN A(self) \/ B(self)
Next == \E self \in Procs : p(self)
====
"#;
        let module = parse_module(src);
        let config = Config {
            init: Some("Init".to_string()),
            next: Some("Next".to_string()),
            ..Default::default()
        };
        let mut setup = setup_checker_modules(
            &module,
            &[],
            &config,
            &SetupOptions {
                load_instances: true,
            },
        );
        setup.ctx.resolve_state_vars_in_loaded_ops();
        let next_def = setup.ctx.get_op("Next").unwrap().as_ref().clone();

        assert!(
            spec_has_pc_guards_legacy(&next_def, &setup.ctx),
            "the loose resolver should demonstrate the global/local name collision"
        );
        assert!(
            !spec_has_pc_guards(&next_def, &setup.ctx),
            "strict mode must not resolve LET-local A/B through same-named globals"
        );
    }

    #[test]
    fn test_spec_has_pc_guards_rejects_operator_formal_shadow() {
        let src = r#"
---- MODULE OperatorFormalPcGuardShadow ----
CONSTANT Procs
VARIABLE pc

Init == TRUE
A(self) == pc[self] = "a"
p(self, A(_)) == A(self) \/ A(self)
Next == \E self \in Procs : p(self, A)
====
"#;
        let module = parse_module(src);
        let config = Config {
            init: Some("Init".to_string()),
            next: Some("Next".to_string()),
            ..Default::default()
        };
        let mut setup = setup_checker_modules(
            &module,
            &[],
            &config,
            &SetupOptions {
                load_instances: true,
            },
        );
        setup.ctx.resolve_state_vars_in_loaded_ops();
        let next_def = setup.ctx.get_op("Next").unwrap().as_ref().clone();

        assert!(
            !spec_has_pc_guards(&next_def, &setup.ctx),
            "operator-formal A must not resolve through the same-named global A"
        );
    }

    #[test]
    fn test_spec_has_pc_guards_does_not_aggregate_unrelated_or_sites() {
        let src = r#"
---- MODULE SeparatePcGuardOrs ----
CONSTANT Procs
VARIABLE pc

Init == TRUE
A(self) == pc[self] = "a"
B(self) == pc[self] = "b"
Left(self) == A(self) \/ TRUE
Right(self) == TRUE \/ B(self)
Both(self) == Left(self) \/ Right(self)
Next == \E self \in Procs : Both(self)
====
"#;
        let module = parse_module(src);
        let config = Config {
            init: Some("Init".to_string()),
            next: Some("Next".to_string()),
            ..Default::default()
        };
        let mut setup = setup_checker_modules(
            &module,
            &[],
            &config,
            &SetupOptions {
                load_instances: true,
            },
        );
        setup.ctx.resolve_state_vars_in_loaded_ops();
        let next_def = setup.ctx.get_op("Next").unwrap().as_ref().clone();

        assert!(
            spec_has_pc_guards_legacy(&next_def, &setup.ctx),
            "the loose global counter should demonstrate the aggregate false positive"
        );
        assert!(
            !spec_has_pc_guards(&next_def, &setup.ctx),
            "no individual Or site has two certifiable pc guards"
        );
    }

    #[test]
    fn test_strict_pc_guards_reject_eager_nonself_arguments() {
        use std::sync::Arc;

        let src = r#"
---- MODULE EagerPcGuardArguments ----
EXTENDS Integers
CONSTANT Procs
VARIABLE pc

Init == TRUE
A(self, x) == pc[self] = "a"
B(self, x) == pc[self] = "b"
p(self) == A(self, 1 / 0) \/ B(self, 1 / 0)
Next == \E self \in Procs : p(self)
====
"#;
        let module = parse_module(src);
        let config = Config {
            init: Some("Init".to_string()),
            next: Some("Next".to_string()),
            ..Default::default()
        };
        let mut setup = setup_checker_modules(
            &module,
            &[],
            &config,
            &SetupOptions {
                load_instances: true,
            },
        );
        setup.ctx.resolve_state_vars_in_loaded_ops();
        let next_def = setup.ctx.get_op("Next").unwrap().as_ref().clone();

        assert!(
            !spec_has_pc_guards(&next_def, &setup.ctx),
            "hoisting must not suppress eager evaluation of non-self arguments"
        );

        let a_call = {
            let p_def = setup.ctx.get_op("p").expect("p should be loaded");
            let Expr::Or(left, _) = &p_def.body.node else {
                panic!("p should contain an Or")
            };
            left.node.clone()
        };
        let mut fb = crate::value::FuncBuilder::new();
        fb.insert(Value::string("p1"), Value::string("b"));
        let current_pc = Value::Func(Rp::new(fb.build()));
        let mut ctx = setup.ctx.clone();
        ctx.push_binding(Arc::from("self"), Value::string("p1"));

        assert!(
            !or_branch_pc_guard_mismatches(&a_call, &current_pc, &ctx),
            "runtime matching must preserve eager evaluation of A's 1 / 0 argument"
        );
    }

    #[test]
    fn test_spec_has_pc_guards_rejects_higher_order_self_formal() {
        let src = r#"
---- MODULE HigherOrderSelfPcGuard ----
CONSTANT Procs
VARIABLE pc

Init == TRUE
p(self(_)) == (pc[self] = "a") \/ (pc[self] = "b")
Next == \E self \in Procs : p(self)
====
"#;
        let module = parse_module(src);
        let config = Config {
            init: Some("Init".to_string()),
            next: Some("Next".to_string()),
            ..Default::default()
        };
        let mut setup = setup_checker_modules(
            &module,
            &[],
            &config,
            &SetupOptions {
                load_instances: true,
            },
        );
        setup.ctx.resolve_state_vars_in_loaded_ops();
        let next_def = setup.ctx.get_op("Next").unwrap().as_ref().clone();

        assert!(
            !spec_has_pc_guards(&next_def, &setup.ctx),
            "an operator-valued self(_) formal cannot forward the scalar process binding"
        );
    }

    #[test]
    fn test_strict_runtime_rejects_locally_shadowed_operator() {
        use std::sync::Arc;

        let src = r#"
---- MODULE RuntimeLocalOperatorShadow ----
VARIABLE pc

Init == TRUE
A(self) == pc[self] = "done"
Next == TRUE
====
"#;
        let module = parse_module(src);
        let config = Config {
            init: Some("Init".to_string()),
            next: Some("Next".to_string()),
            ..Default::default()
        };
        let mut setup = setup_checker_modules(
            &module,
            &[],
            &config,
            &SetupOptions {
                load_instances: true,
            },
        );
        setup.ctx.resolve_state_vars_in_loaded_ops();

        let a_of_self = Expr::Apply(
            Box::new(tla_core::Spanned::dummy(Expr::Ident(
                "A".to_string(),
                tla_core::NameId::INVALID,
            ))),
            vec![tla_core::Spanned::dummy(Expr::Ident(
                "self".to_string(),
                tla_core::NameId::INVALID,
            ))],
        );
        let mut fb = crate::value::FuncBuilder::new();
        fb.insert(Value::string("p1"), Value::string("start"));
        let current_pc = Value::Func(Rp::new(fb.build()));
        let mut ctx = setup.ctx.clone();
        ctx.push_binding(Arc::from("self"), Value::string("p1"));
        ctx.push_binding(Arc::from("A"), Value::Bool(true));

        assert!(
            !or_branch_pc_guard_mismatches(&a_of_self, &current_pc, &ctx),
            "a locally bound A must not be resolved through global operator A"
        );
        assert!(
            !or_branch_pc_guard_mismatches_cached(
                &a_of_self,
                &current_pc,
                &ctx,
                &new_pc_guard_label_cache(),
            ),
            "the cached matcher must also fail closed in local operator scope"
        );
    }
}
