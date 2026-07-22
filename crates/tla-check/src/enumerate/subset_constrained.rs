// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Part of #3432: shared constrained-SUBSET matcher and runtime generator.

#[cfg(test)]
use std::cell::Cell;
use std::cell::RefCell;
use std::sync::{Arc, OnceLock};

use rustc_hash::FxHashMap;
use smallvec::SmallVec;
use tla_core::ast::{Expr, OperatorDef};
use tla_core::{expr_mentions_name_v, NameId, Span, Spanned};
use tla_value::{SortedSet, Value};

use crate::eval::EvalCtx;

use super::subset_profile;

pub(crate) struct ConstrainedSubsetPattern<'a> {
    pub(crate) base_set_expr: &'a Spanned<Expr>,
    pub(crate) superset_expr: &'a Spanned<Expr>,
    pub(crate) subset_expr: &'a Spanned<Expr>,
    pub(crate) remaining_body: Option<Spanned<Expr>>,
}

/// A deliberately narrow certificate for the Sailfish quorum guard:
///
/// ```text
/// \E delivered \in SUBSET S:
///   /\ IsQuorum({Node(v) : v \in delivered})
///   /\ ...
/// ```
///
/// Certification also requires the selected shared definitions to be exactly
/// `Node(v) == v[1]` and `IsQuorum(Q) == Q \in <literal set of sets>`.  The
/// owned fields let the caller evaluate the quorum family once and continue
/// with the exact residual body without retaining operator-definition borrows.
pub(crate) struct QuorumSubsetPattern {
    pub(crate) bound_name_id: NameId,
    pub(crate) quorum_family_expr: Arc<Spanned<Expr>>,
    pub(crate) quorum_family_value: Arc<OnceLock<Value>>,
    pub(crate) remaining_body: Option<Arc<Spanned<Expr>>>,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct QuorumSubsetCacheKey {
    shared_id: u64,
    shared_ptr: usize,
    precomputed_constants_version: u64,
    domain_ptr: usize,
    domain_span: Span,
    body_ptr: usize,
    body_span: Span,
}

#[derive(Clone)]
struct CachedQuorumSubsetCertificate {
    /// Pin the exact semantic snapshot. `SharedCtx::id()` survives COW clones,
    /// so pointer equality is the exact guard against same-id mutations.
    shared: Arc<tla_eval::SharedCtx>,
    bound_name: Arc<str>,
    bound_name_id: NameId,
    quorum_family_expr: Arc<Spanned<Expr>>,
    /// Initialized only after the powerset domain has been evaluated. Errors
    /// are never stored, so an unexpected evaluator failure retains its normal
    /// per-activation behavior.
    quorum_family_value: Arc<OnceLock<Value>>,
    remaining_body: Option<Arc<Spanned<Expr>>>,
    /// Identifier atoms in the literal quorum family. Shared data is pinned;
    /// only a dynamic lexical shadow can change their runtime resolution.
    quorum_atom_names: Arc<[Arc<str>]>,
}

impl CachedQuorumSubsetCertificate {
    fn is_still_valid(&self, ctx: &EvalCtx, bound_name: &str) -> bool {
        Arc::ptr_eq(&self.shared, ctx.shared())
            && self.bound_name.as_ref() == bound_name
            && ctx.instance_substitutions().is_none()
            && ctx.call_by_name_subs().is_none()
            && original_shared_op_still_selected(ctx, "Node")
            && original_shared_op_still_selected(ctx, "IsQuorum")
            && self
                .quorum_atom_names
                .iter()
                .all(|name| !ctx.name_in_local_scope(name))
    }

    fn pattern(&self) -> QuorumSubsetPattern {
        QuorumSubsetPattern {
            bound_name_id: self.bound_name_id,
            quorum_family_expr: Arc::clone(&self.quorum_family_expr),
            quorum_family_value: Arc::clone(&self.quorum_family_value),
            remaining_body: self.remaining_body.clone(),
        }
    }
}

thread_local! {
    /// Positive shared/syntax certificates. The cached Arc pins every shared
    /// resolution input; small lexical-shadow guards still run per activation.
    static QUORUM_SUBSET_SYNTAX_CACHE:
        RefCell<FxHashMap<QuorumSubsetCacheKey, CachedQuorumSubsetCertificate>> =
            RefCell::new(FxHashMap::default());
}

/// Clear raw-AST-pointer quorum syntax entries at the normal run boundary.
pub(crate) fn clear_quorum_subset_syntax_cache() {
    QUORUM_SUBSET_SYNTAX_CACHE.with(|cache| cache.borrow_mut().clear());
}

#[cfg(test)]
thread_local! {
    static QUORUM_SUBSET_PRUNE_TEST_OVERRIDE: Cell<Option<bool>> = const { Cell::new(None) };
    static QUORUM_SUBSET_PRUNE_TEST_ACTIVATIONS: Cell<u64> = const { Cell::new(0) };
}

/// Default-on operational gate.  Any presence of the kill switch restores the
/// ordinary powerset path in release builds as well as debug builds.
pub(crate) fn quorum_subset_prune_enabled() -> bool {
    #[cfg(test)]
    if let Some(enabled) = QUORUM_SUBSET_PRUNE_TEST_OVERRIDE.with(Cell::get) {
        return enabled;
    }

    static DISABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    !crate::debug_env::env_flag_is_set(&DISABLED, "TY_NO_QUORUM_SUBSET_PRUNE")
}

#[cfg(test)]
pub(crate) fn with_quorum_subset_prune_test_override<R>(enabled: bool, f: impl FnOnce() -> R) -> R {
    struct Reset(Option<bool>);

    impl Drop for Reset {
        fn drop(&mut self) {
            QUORUM_SUBSET_PRUNE_TEST_OVERRIDE.with(|slot| slot.set(self.0));
        }
    }

    let previous = QUORUM_SUBSET_PRUNE_TEST_OVERRIDE.with(|slot| slot.replace(Some(enabled)));
    let _reset = Reset(previous);
    f()
}

#[cfg(test)]
pub(crate) fn reset_quorum_subset_prune_test_activations() {
    QUORUM_SUBSET_PRUNE_TEST_ACTIVATIONS.with(|count| count.set(0));
}

#[cfg(test)]
pub(crate) fn quorum_subset_prune_test_activations() -> u64 {
    QUORUM_SUBSET_PRUNE_TEST_ACTIVATIONS.with(Cell::get)
}

/// Certify only the exact first-guard shape used by TLCSailfish1.
///
/// Every ambiguity fails closed: labels/wrappers, multiple bounds, dynamic
/// operator shadows, config replacements, non-literal quorum families, and
/// alternate `Node` implementations all retain generic SUBSET enumeration.
pub(crate) fn match_quorum_subset_exists(
    ctx: &EvalCtx,
    var_name: &str,
    domain_expr: &Spanned<Expr>,
    body: &Spanned<Expr>,
) -> Option<QuorumSubsetPattern> {
    if !quorum_subset_prune_enabled() || !matches!(&domain_expr.node, Expr::Powerset(_)) {
        return None;
    }

    let key = QuorumSubsetCacheKey {
        shared_id: ctx.shared().id(),
        shared_ptr: Arc::as_ptr(ctx.shared()) as usize,
        precomputed_constants_version: ctx.shared().precomputed_constants_version(),
        domain_ptr: domain_expr as *const Spanned<Expr> as usize,
        domain_span: domain_expr.span,
        body_ptr: body as *const Spanned<Expr> as usize,
        body_span: body.span,
    };
    let cached = QUORUM_SUBSET_SYNTAX_CACHE.with(|cache| {
        let cache = cache.borrow();
        cache.get(&key).map(|cached| {
            cached
                .is_still_valid(ctx, var_name)
                .then(|| cached.pattern())
        })
    });
    if let Some(pattern) = cached {
        return pattern;
    }

    // Probe the only shape we accept without allocating a flattened
    // conjunction. Full reconstruction is deferred to the positive syntax
    // cache's vacant path, so misses and cache hits remain allocation-free.
    let first = leftmost_conjunct(body);
    let Expr::Apply(quorum_op, quorum_args) = &first.node else {
        return None;
    };
    let Expr::Ident(quorum_name, quorum_name_id) = &quorum_op.node else {
        return None;
    };
    if quorum_name != "IsQuorum" || !name_id_matches_spelling(quorum_name, *quorum_name_id) {
        return None;
    }
    let [quorum_arg] = quorum_args.as_slice() else {
        return None;
    };

    let Expr::SetBuilder(node_call, image_bounds) = &quorum_arg.node else {
        return None;
    };
    let [image_bound] = image_bounds.as_slice() else {
        return None;
    };
    if image_bound.pattern.is_some() {
        return None;
    }
    let image_name = image_bound.name.node.as_str();
    let image_domain = image_bound.domain.as_ref()?;
    // The optimized loop binds by NameId directly. Require the guard's exact
    // reference to the outer EXISTS name to carry valid lowering metadata;
    // ambiguous/unlowered identifiers retain the generic binder path.
    let bound_name_id = exact_valid_ident_name_id(image_domain, var_name)?;

    let Expr::Apply(node_op, node_args) = &node_call.node else {
        return None;
    };
    let Expr::Ident(node_name, node_name_id) = &node_op.node else {
        return None;
    };
    if node_name != "Node" || !name_id_matches_spelling(node_name, *node_name_id) {
        return None;
    }
    // These binders are installed only after this matcher returns, so the
    // current EvalCtx cannot reveal a future collision. Keep the certificate's
    // operator and value namespaces demonstrably distinct instead of relying
    // on runtime Apply fall-through behavior for a same-spelled binding.
    if var_name == image_name
        || [var_name, image_name]
            .iter()
            .any(|bound| *bound == quorum_name || *bound == node_name)
    {
        return None;
    }
    let [node_arg] = node_args.as_slice() else {
        return None;
    };
    if !is_exact_ident(node_arg, image_name) {
        return None;
    }

    certify_node_projection(ctx, node_name)?;
    let (quorum_family_expr, quorum_atom_names) =
        certify_literal_quorum_operator(ctx, quorum_name, &[var_name, image_name])?;
    let conjuncts = flatten_conjunction(body);
    let cached = CachedQuorumSubsetCertificate {
        shared: Arc::clone(ctx.shared()),
        bound_name: Arc::from(var_name),
        bound_name_id,
        quorum_family_expr: Arc::new(quorum_family_expr.clone()),
        quorum_family_value: Arc::new(OnceLock::new()),
        remaining_body: reconstruct_remaining_body(body, &conjuncts, &[0]).map(Arc::new),
        quorum_atom_names: quorum_atom_names.into_vec().into(),
    };
    let pattern = cached.pattern();
    QUORUM_SUBSET_SYNTAX_CACHE.with(|cache| {
        cache.borrow_mut().insert(key, cached);
    });
    Some(pattern)
}

/// Generate only the powerset members whose exact `Node` image is present in
/// the certified quorum family.
///
/// `base_elements` must be in the same TLC-normalized order used by
/// `SubsetIterator::from_elements`.  Candidate plans are therefore ordered by
/// cardinality and then by their raw base indices, exactly matching ordinary
/// powerset enumeration without constructing any rejected candidates.
///
/// `None` means the direct tuple projection is ambiguous or not representable
/// and the caller must reuse the already-evaluated domain on the generic path.
pub(crate) fn generate_quorum_subsets(
    base_elements: &[Value],
    quorum_family: &Value,
) -> Option<Vec<Value>> {
    // Keep the optimization deliberately narrow and its retained output
    // bounded. Sailfish has at most three base vertices and three quorums.
    const MAX_DIRECT_BASE_ELEMENTS: usize = 64;
    const MAX_DIRECT_QUORUMS: usize = 64;

    let Value::Set(family) = quorum_family else {
        return None;
    };
    if base_elements.len() > MAX_DIRECT_BASE_ELEMENTS || family.len() > MAX_DIRECT_QUORUMS {
        return None;
    }

    // Direct generation is one projected node -> one base index. If two
    // distinct vertices project to the same node, several different vertex
    // subsets can have the same node image; falling back is the minimal exact
    // handling for that case.
    let mut projected_nodes: SmallVec<[&Value; 8]> = SmallVec::with_capacity(base_elements.len());
    for vertex in base_elements {
        let Value::Tuple(fields) = vertex else {
            return None;
        };
        let node = fields.first()?;
        if projected_nodes.iter().any(|existing| (*existing).eq(node)) {
            return None;
        }
        projected_nodes.push(node);
    }

    // One bit per normalized base index avoids a heap allocation for every
    // retained quorum. The certified direct lane is capped at 64 elements.
    let mut plans: SmallVec<[u64; 8]> = SmallVec::with_capacity(family.len());
    for quorum in family.iter() {
        let Value::Set(nodes) = quorum else {
            return None;
        };
        if nodes.len() > projected_nodes.len() {
            // A unique projection cannot realize a quorum larger than the
            // available base image.
            continue;
        }

        let mut plan = 0u64;
        let mut representable = true;
        for node in nodes.iter() {
            let Some(index) = projected_nodes
                .iter()
                .position(|projected| (*projected).eq(node))
            else {
                representable = false;
                break;
            };
            plan |= 1u64 << index;
        }
        if representable {
            plans.push(plan);
        }
    }

    // SortedSet iteration order for the family is unrelated to TLC powerset
    // order. Re-establish cardinality-first, then lexicographic base-index
    // order explicitly.
    plans.sort_unstable_by(|left, right| {
        left.count_ones()
            .cmp(&right.count_ones())
            .then_with(|| compare_plan_indices(*left, *right))
    });
    plans.dedup();

    let retained = plans
        .into_iter()
        .map(|mut plan| {
            let mut vertices = Vec::with_capacity(plan.count_ones() as usize);
            while plan != 0 {
                let index = plan.trailing_zeros() as usize;
                vertices.push(base_elements[index].clone());
                plan &= plan - 1;
            }
            // Base indices are TLC-normalized, while materialized sets use
            // Value::cmp order. This is the same final sort performed by
            // SubsetIterator::from_elements.
            vertices.sort();
            Value::from_sorted_set(SortedSet::from_sorted_vec(vertices))
        })
        .collect();

    #[cfg(test)]
    QUORUM_SUBSET_PRUNE_TEST_ACTIVATIONS.with(|count| count.set(count.get().saturating_add(1)));
    Some(retained)
}

fn compare_plan_indices(mut left: u64, mut right: u64) -> std::cmp::Ordering {
    loop {
        match (left, right) {
            (0, 0) => return std::cmp::Ordering::Equal,
            (0, _) => return std::cmp::Ordering::Less,
            (_, 0) => return std::cmp::Ordering::Greater,
            _ => {}
        }
        match left.trailing_zeros().cmp(&right.trailing_zeros()) {
            std::cmp::Ordering::Equal => {
                left &= left - 1;
                right &= right - 1;
            }
            ordering => return ordering,
        }
    }
}

fn certify_node_projection(ctx: &EvalCtx, name: &str) -> Option<()> {
    let def = resolve_original_shared_op(ctx, name)?;
    let [param] = def.params.as_slice() else {
        return None;
    };
    if param.arity != 0 || def.is_recursive || def.has_primed_param {
        return None;
    }
    let Expr::FuncApply(base, index) = &def.body.node else {
        return None;
    };
    if !is_exact_ident(base, param.name.node.as_str()) {
        return None;
    }
    matches!(&index.node, Expr::Int(value) if value == &1.into()).then_some(())
}

fn certify_literal_quorum_operator<'a>(
    ctx: &'a EvalCtx,
    name: &str,
    outer_bound_names: &[&str],
) -> Option<(&'a Spanned<Expr>, SmallVec<[Arc<str>; 8]>)> {
    let def = resolve_original_shared_op(ctx, name)?;
    let [param] = def.params.as_slice() else {
        return None;
    };
    if param.arity != 0 || def.is_recursive || def.has_primed_param {
        return None;
    }
    let Expr::In(candidate, family) = &def.body.node else {
        return None;
    };
    let mut forbidden = Vec::with_capacity(outer_bound_names.len() + 1);
    forbidden.extend_from_slice(outer_bound_names);
    forbidden.push(param.name.node.as_str());
    let mut atom_names = SmallVec::new();
    if !is_exact_ident(candidate, param.name.node.as_str())
        || !is_literal_set_of_sets(ctx, &family.node, &forbidden, &mut atom_names)
    {
        return None;
    }
    Some((family.as_ref(), atom_names))
}

fn is_literal_set_of_sets(
    ctx: &EvalCtx,
    expr: &Expr,
    forbidden: &[&str],
    atom_names: &mut SmallVec<[Arc<str>; 8]>,
) -> bool {
    let Expr::SetEnum(quorums) = expr else {
        return false;
    };
    quorums.iter().all(|quorum| {
        matches!(&quorum.node, Expr::SetEnum(nodes) if nodes.iter().all(|node| {
            match &node.node {
                Expr::String(_) | Expr::Int(_) | Expr::Bool(_) => true,
                Expr::Ident(name, name_id) => {
                    *name_id != NameId::INVALID
                        && tla_core::lookup_name_id(name) == Some(*name_id)
                        && !forbidden.iter().any(|bound| *bound == name)
                        && !ctx.name_in_local_scope(name)
                        && !ctx.op_replacements().contains_key(name)
                        && ctx
                            .precomputed_constants()
                            .get(name_id)
                            .is_some_and(Value::is_concrete_data)
                        && {
                            if !atom_names.iter().any(|existing| existing.as_ref() == name) {
                                atom_names.push(Arc::from(name.as_str()));
                            }
                            true
                        }
                }
                _ => false,
            }
        }))
    })
}

/// Resolve only the source-spelled shared operator.  This deliberately refuses
/// local operator scopes and config replacements even if their current bodies
/// happen to look equivalent.
fn resolve_original_shared_op<'a>(ctx: &'a EvalCtx, name: &str) -> Option<&'a OperatorDef> {
    if ctx.name_in_local_scope(name)
        || ctx.op_replacements().contains_key(name)
        || ctx.is_config_constant(name)
        || ctx.instance_substitutions().is_some()
        || ctx.call_by_name_subs().is_some()
    {
        return None;
    }
    let shared = ctx.ops().get(name)?;
    let selected = ctx.get_op(name)?;
    (Arc::ptr_eq(shared, selected)
        && !crate::eval::should_prefer_builtin_override(name, selected, 1, ctx))
    .then_some(shared.as_ref())
}

fn original_shared_op_still_selected(ctx: &EvalCtx, name: &str) -> bool {
    if ctx.name_in_local_scope(name) {
        return false;
    }
    matches!(
        (ctx.ops().get(name), ctx.get_op(name)),
        (Some(shared), Some(selected)) if Arc::ptr_eq(shared, selected)
    )
}

fn name_id_matches_spelling(name: &str, name_id: NameId) -> bool {
    name_id == NameId::INVALID || tla_core::lookup_name_id(name) == Some(name_id)
}

fn is_exact_ident(expr: &Spanned<Expr>, expected: &str) -> bool {
    matches!(
        &expr.node,
        Expr::Ident(name, name_id)
            if name == expected && name_id_matches_spelling(name, *name_id)
    )
}

fn exact_valid_ident_name_id(expr: &Spanned<Expr>, expected: &str) -> Option<NameId> {
    match &expr.node {
        Expr::Ident(name, name_id)
            if name == expected
                && *name_id != NameId::INVALID
                && tla_core::lookup_name_id(name) == Some(*name_id) =>
        {
            Some(*name_id)
        }
        _ => None,
    }
}

pub(crate) fn match_constrained_subset_exists<'a>(
    var_name: &str,
    domain_expr: &'a Spanned<Expr>,
    body: &'a Spanned<Expr>,
) -> Option<ConstrainedSubsetPattern<'a>> {
    let base_set_expr = match &domain_expr.node {
        Expr::Powerset(inner) => inner.as_ref(),
        _ => return None,
    };

    let conjuncts = flatten_conjunction(body);
    if conjuncts.len() < 2 {
        return None;
    }

    let mut superset_idx = None;
    let mut subset_idx = None;

    for (idx, conjunct) in conjuncts.iter().enumerate() {
        let Expr::Subseteq(left, right) = &conjunct.node else {
            continue;
        };

        if is_bare_ident(&left.node, var_name) {
            if superset_idx.replace(idx).is_some() {
                return None;
            }
            continue;
        }
        if is_bare_ident(&right.node, var_name) && subset_idx.replace(idx).is_some() {
            return None;
        }
    }

    let superset_idx = superset_idx?;
    let subset_idx = subset_idx?;

    // The optimized route evaluates the two independent bounds once, before
    // enumerating candidates. Preserve ordinary left-to-right conjunction
    // order exactly: any prefix, gap, reversal, or duplicate must fall back.
    if superset_idx != 0 || subset_idx != 1 {
        return None;
    }

    let superset_expr = match &conjuncts[superset_idx].node {
        Expr::Subseteq(_, right) => right.as_ref(),
        _ => unreachable!(),
    };
    let subset_expr = match &conjuncts[subset_idx].node {
        Expr::Subseteq(left, _) => left.as_ref(),
        _ => unreachable!(),
    };

    // Both bounds are evaluated before the quantified variable is installed.
    // If either one mentions that variable, the constrained route would read
    // an outer same-name binding or raise an error instead of evaluating each
    // candidate in the ordinary EXISTS scope. Refuse and use generic SUBSET
    // enumeration in every such case.
    if expr_mentions_name_v(&superset_expr.node, var_name)
        || expr_mentions_name_v(&subset_expr.node, var_name)
    {
        return None;
    }

    let remaining_body = reconstruct_remaining_body(body, &conjuncts, &[superset_idx, subset_idx]);

    Some(ConstrainedSubsetPattern {
        base_set_expr,
        superset_expr,
        subset_expr,
        remaining_body,
    })
}

/// Generate all subsets `r` of `base_elements` where `subset_bound ⊆ r ⊆ superset_bound`.
///
/// Returns `None` when membership checks are indeterminate, signaling the caller
/// to fall back to the generic SUBSET enumeration path.
pub(crate) fn generate_constrained_subsets(
    base_elements: &[Value],
    superset_bound: &Value,
    subset_bound: &Value,
) -> Option<Vec<Value>> {
    if !superset_bound.is_set() {
        subset_profile::record_fallback();
        return None;
    }

    let x_elements: Vec<&Value> = match subset_bound {
        Value::Set(set) => set.iter().collect(),
        _ => {
            // Valid lazy sets and invalid scalar values both need the generic
            // evaluator: the former may require context-aware membership,
            // while the latter must retain its ordinary TypeError.
            subset_profile::record_fallback();
            return None;
        }
    };

    for x_elem in &x_elements {
        match superset_bound.set_contains(x_elem) {
            Some(true) => {}
            Some(false) => {
                subset_profile::record_success(0, 0);
                return Some(Vec::new());
            }
            None => {
                subset_profile::record_fallback();
                return None;
            }
        }
        if !base_elements.contains(x_elem) {
            subset_profile::record_success(0, 0);
            return Some(Vec::new());
        }
    }

    let mut free_elements: Vec<&Value> = Vec::new();
    for base_elem in base_elements {
        match superset_bound.set_contains(base_elem) {
            Some(true) => {
                if x_elements.iter().all(|x| *x != base_elem) {
                    free_elements.push(base_elem);
                }
            }
            Some(false) => {}
            None => {
                subset_profile::record_fallback();
                return None;
            }
        }
    }

    let num_free = free_elements.len();
    if num_free > 20 {
        subset_profile::record_fallback();
        return None;
    }
    debug_assert!(
        num_free <= 31,
        "count_ones via u32 truncation requires num_free <= 31"
    );

    // Part of #3364: Pre-sort free_elements so we can merge-insert instead of
    // sorting each generated subset. Combined with k-subset combinatorial
    // iteration this reduces from O(n * 2^n) bitmask scanning to O(2^n) direct
    // enumeration, and from O(k log k) per-subset sort to O(k) merge.
    free_elements.sort();
    let x_sorted: Vec<Value> = x_elements.iter().map(|v| (*v).clone()).collect();

    let count = 1usize << num_free;
    let mut result = Vec::with_capacity(count);

    for k in 0..=num_free {
        if k == 0 {
            // No free elements selected — just the mandatory subset bound.
            result.push(Value::from_sorted_set(SortedSet::from_sorted_vec(
                x_sorted.clone(),
            )));
            continue;
        }
        // Enumerate all k-subsets of free_elements using index iteration.
        let mut indices: Vec<usize> = (0..k).collect();
        loop {
            let selected_free: Vec<Value> =
                indices.iter().map(|&i| free_elements[i].clone()).collect();
            let merged = merge_sorted(&x_sorted, &selected_free);
            result.push(Value::from_sorted_set(SortedSet::from_sorted_vec(merged)));
            if !advance_k_subset_indices(&mut indices, num_free) {
                break;
            }
        }
    }

    subset_profile::record_success(free_elements.len(), result.len());
    Some(result)
}

/// Merge two sorted `Value` slices into a single sorted `Vec<Value>`.
/// Both inputs must be sorted by `Value::cmp`. O(a.len() + b.len()).
fn merge_sorted(a: &[Value], b: &[Value]) -> Vec<Value> {
    let mut result = Vec::with_capacity(a.len() + b.len());
    let mut ai = 0;
    let mut bi = 0;
    while ai < a.len() && bi < b.len() {
        if a[ai] <= b[bi] {
            result.push(a[ai].clone());
            ai += 1;
        } else {
            result.push(b[bi].clone());
            bi += 1;
        }
    }
    while ai < a.len() {
        result.push(a[ai].clone());
        ai += 1;
    }
    while bi < b.len() {
        result.push(b[bi].clone());
        bi += 1;
    }
    result
}

/// Advance a k-subset index array to the next combination in lexicographic order.
/// Returns `false` when all combinations have been exhausted.
/// `indices` must be a sorted array of `k` distinct indices in `0..n`.
fn advance_k_subset_indices(indices: &mut [usize], n: usize) -> bool {
    let k = indices.len();
    let mut i = k;
    while i > 0 {
        i -= 1;
        if indices[i] < n - k + i {
            indices[i] += 1;
            for j in (i + 1)..k {
                indices[j] = indices[j - 1] + 1;
            }
            return true;
        }
    }
    false
}

fn reconstruct_remaining_body(
    original_body: &Spanned<Expr>,
    conjuncts: &[&Spanned<Expr>],
    removed_indices: &[usize],
) -> Option<Spanned<Expr>> {
    let mut remaining = conjuncts
        .iter()
        .enumerate()
        .filter(|(idx, _)| !removed_indices.contains(idx))
        .map(|(_, conjunct)| (*conjunct).clone());

    let first = remaining.next()?;
    Some(remaining.fold(first, |acc, next| {
        Spanned::new(Expr::And(Box::new(acc), Box::new(next)), original_body.span)
    }))
}

fn flatten_conjunction<'a>(expr: &'a Spanned<Expr>) -> Vec<&'a Spanned<Expr>> {
    let mut result = Vec::new();
    flatten_conjunction_inner(expr, &mut result);
    result
}

fn leftmost_conjunct(mut expr: &Spanned<Expr>) -> &Spanned<Expr> {
    while let Expr::And(left, _) = &expr.node {
        expr = left;
    }
    expr
}

fn flatten_conjunction_inner<'a>(expr: &'a Spanned<Expr>, out: &mut Vec<&'a Spanned<Expr>>) {
    match &expr.node {
        Expr::And(left, right) => {
            flatten_conjunction_inner(left, out);
            flatten_conjunction_inner(right, out);
        }
        _ => out.push(expr),
    }
}

fn is_bare_ident(expr: &Expr, name: &str) -> bool {
    matches!(expr, Expr::Ident(ident, _) if ident.as_str() == name)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tla_core::ast::Unit;
    use tla_core::{intern_name, lower, parse_to_syntax_tree, FileId};

    fn expr(node: Expr) -> Spanned<Expr> {
        Spanned::dummy(node)
    }

    fn ident(name: &str) -> Spanned<Expr> {
        expr(Expr::Ident(name.to_string(), tla_core::NameId::INVALID))
    }

    fn setup_quorum_matcher(src: &str) -> (EvalCtx, Arc<OperatorDef>) {
        let tree = parse_to_syntax_tree(src);
        let lowered = lower(FileId(0), &tree);
        let module = lowered.module.expect("test module should lower");
        let mut ctx = EvalCtx::new();
        ctx.load_module(&module);
        let vars: Vec<Arc<str>> = module
            .units
            .iter()
            .filter_map(|unit| match &unit.node {
                Unit::Variable(names) => Some(names.as_slice()),
                _ => None,
            })
            .flatten()
            .map(|name| Arc::from(name.node.as_str()))
            .collect();
        ctx.register_vars(vars);
        ctx.resolve_state_vars_in_loaded_ops();
        let constants = Arc::make_mut(ctx.shared_arc_mut()).precomputed_constants_mut();
        constants.insert(intern_name("n1"), Value::string("n1"));
        constants.insert(intern_name("n2"), Value::string("n2"));
        constants.insert(intern_name("n3"), Value::string("n3"));
        let next = Arc::clone(ctx.get_op("Next").expect("Next should exist"));
        (ctx, next)
    }

    fn quorum_parts(next: &OperatorDef) -> (&tla_core::ast::BoundVar, &Spanned<Expr>) {
        let Expr::Exists(bounds, body) = &next.body.node else {
            panic!("Next should be a top-level EXISTS")
        };
        let [bound] = bounds.as_slice() else {
            panic!("Next should have exactly one bound")
        };
        (bound, body)
    }

    #[test]
    fn quorum_matcher_certifies_only_exact_first_guard_and_definitions() {
        let exact = r#"
---- MODULE QuorumSubsetExact ----
CONSTANTS n1, n2, n3
VARIABLE y
Node(v) == v[1]
IsQuorum(Q) == Q \in {{n1, n3}, {n2, n3}, {n1, n2, n3}}
Next == \E delivered \in SUBSET {<<n1, 1>>, <<n2, 1>>, <<n3, 1>>}:
          /\ IsQuorum({Node(v) : v \in delivered})
          /\ y' = 1
====
"#;
        let (mut ctx, next) = setup_quorum_matcher(exact);
        let (bound, body) = quorum_parts(&next);
        let pattern = match_quorum_subset_exists(
            &ctx,
            bound.name.node.as_str(),
            bound.domain.as_ref().unwrap(),
            body,
        )
        .expect("the exact Sailfish guard should certify");
        assert!(matches!(
            pattern.remaining_body.as_deref().map(|expr| &expr.node),
            Some(Expr::Eq(_, _))
        ));
        let cached_pattern = match_quorum_subset_exists(
            &ctx,
            bound.name.node.as_str(),
            bound.domain.as_ref().unwrap(),
            body,
        )
        .expect("the repeated exact guard should remain certified");
        assert!(Arc::ptr_eq(
            &pattern.quorum_family_expr,
            &cached_pattern.quorum_family_expr,
        ));
        assert!(Arc::ptr_eq(
            &pattern.quorum_family_value,
            &cached_pattern.quorum_family_value,
        ));
        assert_eq!(pattern.bound_name_id, cached_pattern.bound_name_id);
        assert!(Arc::ptr_eq(
            pattern.remaining_body.as_ref().unwrap(),
            cached_pattern.remaining_body.as_ref().unwrap(),
        ));

        let stack_mark = ctx.mark_stack();
        ctx.push_binding(Arc::from("Node"), Value::SmallInt(0));
        assert!(match_quorum_subset_exists(
            &ctx,
            bound.name.node.as_str(),
            bound.domain.as_ref().unwrap(),
            body,
        )
        .is_none());
        ctx.pop_to_mark(&stack_mark);
        let unshadowed_pattern = match_quorum_subset_exists(
            &ctx,
            bound.name.node.as_str(),
            bound.domain.as_ref().unwrap(),
            body,
        )
        .expect("a transient dynamic shadow must not poison the positive cache");
        assert!(Arc::ptr_eq(
            &pattern.quorum_family_value,
            &unshadowed_pattern.quorum_family_value,
        ));

        let mut cow_ctx = ctx.clone();
        assert_eq!(cow_ctx.shared().id(), ctx.shared().id());
        Arc::make_mut(cow_ctx.shared_arc_mut())
            .precomputed_constants_mut()
            .insert(intern_name("n1"), Value::string("changed-n1"));
        let cow_pattern = match_quorum_subset_exists(
            &cow_ctx,
            bound.name.node.as_str(),
            bound.domain.as_ref().unwrap(),
            body,
        )
        .expect("a same-id COW snapshot should receive its own certificate");
        assert!(!Arc::ptr_eq(
            &pattern.quorum_family_value,
            &cow_pattern.quorum_family_value,
        ));

        with_quorum_subset_prune_test_override(false, || {
            assert!(match_quorum_subset_exists(
                &ctx,
                bound.name.node.as_str(),
                bound.domain.as_ref().unwrap(),
                body,
            )
            .is_none());
        });

        let prefixed = exact.replace(
            "/\\ IsQuorum({Node(v) : v \\in delivered})",
            "/\\ FALSE\n          /\\ IsQuorum({Node(v) : v \\in delivered})",
        );
        let (prefixed_ctx, prefixed_next) = setup_quorum_matcher(&prefixed);
        let (prefixed_bound, prefixed_body) = quorum_parts(&prefixed_next);
        assert!(match_quorum_subset_exists(
            &prefixed_ctx,
            prefixed_bound.name.node.as_str(),
            prefixed_bound.domain.as_ref().unwrap(),
            prefixed_body,
        )
        .is_none());

        let wrong_node = exact.replace("Node(v) == v[1]", "Node(v) == v[2]");
        let (wrong_ctx, wrong_next) = setup_quorum_matcher(&wrong_node);
        let (wrong_bound, wrong_body) = quorum_parts(&wrong_next);
        assert!(match_quorum_subset_exists(
            &wrong_ctx,
            wrong_bound.name.node.as_str(),
            wrong_bound.domain.as_ref().unwrap(),
            wrong_body,
        )
        .is_none());

        let future_node_shadow = exact.replace("delivered", "Node");
        let (shadow_ctx, shadow_next) = setup_quorum_matcher(&future_node_shadow);
        let (shadow_bound, shadow_body) = quorum_parts(&shadow_next);
        assert!(match_quorum_subset_exists(
            &shadow_ctx,
            shadow_bound.name.node.as_str(),
            shadow_bound.domain.as_ref().unwrap(),
            shadow_body,
        )
        .is_none());
    }

    #[test]
    fn quorum_generator_uses_exact_tlc_order_and_skips_missing_nodes() {
        let vertex =
            |node: &str, round: i64| Value::tuple([Value::string(node), Value::SmallInt(round)]);
        let v1 = vertex("n1", 1);
        let v2 = vertex("n2", 1);
        let v3 = vertex("n3", 1);
        let base_elements = vec![v1.clone(), v2.clone(), v3.clone()];
        let family = Value::set([
            Value::set([Value::string("n3")]),
            Value::set([Value::string("n1"), Value::string("n2")]),
            Value::set([Value::string("n1"), Value::string("n3")]),
            Value::set([Value::string("n2"), Value::string("n3")]),
            // No base vertex projects to n4, so this whitelist entry cannot
            // produce a powerset member.
            Value::set([Value::string("n1"), Value::string("n4")]),
        ]);
        let retained = generate_quorum_subsets(&base_elements, &family).unwrap();
        assert_eq!(
            retained,
            vec![
                Value::set([v3.clone()]),
                Value::set([v1.clone(), v2.clone()]),
                Value::set([v1.clone(), v3.clone()]),
                Value::set([v2, v3]),
            ]
        );
    }

    #[test]
    fn quorum_plan_masks_match_cardinality_then_index_lexicographic_order() {
        for width in 0..=6 {
            let mut actual: Vec<u64> = (0..(1u64 << width)).collect();
            actual.sort_unstable_by(|left, right| {
                left.count_ones()
                    .cmp(&right.count_ones())
                    .then_with(|| compare_plan_indices(*left, *right))
            });

            let mut expected: Vec<u64> = (0..(1u64 << width)).collect();
            expected.sort_unstable_by_key(|mask| {
                (
                    mask.count_ones(),
                    (0..width)
                        .filter(|index| mask & (1u64 << index) != 0)
                        .collect::<Vec<_>>(),
                )
            });
            assert_eq!(actual, expected, "mask ordering differed at width {width}");
        }

        // Numeric mask order gets this pair wrong: [0, 3] must precede [1, 2].
        assert_eq!(
            compare_plan_indices(0b1001, 0b0110),
            std::cmp::Ordering::Less
        );
    }

    #[test]
    fn quorum_generator_falls_back_on_duplicate_projection() {
        let vertex =
            |node: &str, round: i64| Value::tuple([Value::string(node), Value::SmallInt(round)]);
        let n1a = vertex("n1", 1);
        let n1b = vertex("n1", 2);
        let collision_base = vec![n1a.clone(), n1b.clone(), vertex("n2", 1)];
        let collision_family = Value::set([Value::set([Value::string("n1")])]);
        assert!(generate_quorum_subsets(&collision_base, &collision_family).is_none());
    }

    #[test]
    fn quorum_generator_falls_back_on_malformed_projection_values() {
        let family = Value::set([Value::set([Value::SmallInt(1)])]);
        assert!(generate_quorum_subsets(&[Value::SmallInt(1)], &family).is_none());
        assert!(generate_quorum_subsets(&[Value::tuple([])], &family).is_none());
    }

    #[test]
    fn matcher_rejects_bounds_that_depend_on_the_quantified_name() {
        let singleton = || expr(Expr::SetEnum(vec![expr(Expr::Int(1.into()))]));
        let empty = || expr(Expr::SetEnum(Vec::new()));
        let domain = expr(Expr::Powerset(Box::new(singleton())));

        let dependent_superset = expr(Expr::And(
            Box::new(expr(Expr::Subseteq(
                Box::new(ident("r")),
                Box::new(expr(Expr::Union(
                    Box::new(ident("r")),
                    Box::new(singleton()),
                ))),
            ))),
            Box::new(expr(Expr::Subseteq(
                Box::new(empty()),
                Box::new(ident("r")),
            ))),
        ));
        assert!(match_constrained_subset_exists("r", &domain, &dependent_superset).is_none());

        let dependent_subset = expr(Expr::And(
            Box::new(expr(Expr::Subseteq(
                Box::new(ident("r")),
                Box::new(singleton()),
            ))),
            Box::new(expr(Expr::Subseteq(
                Box::new(expr(Expr::Union(Box::new(ident("r")), Box::new(empty())))),
                Box::new(ident("r")),
            ))),
        ));
        assert!(match_constrained_subset_exists("r", &domain, &dependent_subset).is_none());

        let independent = expr(Expr::And(
            Box::new(expr(Expr::Subseteq(
                Box::new(ident("r")),
                Box::new(singleton()),
            ))),
            Box::new(expr(Expr::Subseteq(
                Box::new(empty()),
                Box::new(ident("r")),
            ))),
        ));
        assert!(match_constrained_subset_exists("r", &domain, &independent).is_some());

        let reversed = expr(Expr::And(
            Box::new(expr(Expr::Subseteq(
                Box::new(empty()),
                Box::new(ident("r")),
            ))),
            Box::new(expr(Expr::Subseteq(
                Box::new(ident("r")),
                Box::new(singleton()),
            ))),
        ));
        assert!(match_constrained_subset_exists("r", &domain, &reversed).is_none());

        let prefixed = expr(Expr::And(
            Box::new(expr(Expr::Bool(false))),
            Box::new(independent),
        ));
        assert!(match_constrained_subset_exists("r", &domain, &prefixed).is_none());
    }

    #[test]
    fn test_empty_subset_bound() {
        let base = vec![Value::SmallInt(1), Value::SmallInt(2), Value::SmallInt(3)];
        let superset = Value::set(base.clone());
        let subset = Value::empty_set();

        let results = generate_constrained_subsets(&base, &superset, &subset).unwrap();
        assert_eq!(results.len(), 8);
        let sizes: Vec<usize> = results
            .iter()
            .map(|value| match value {
                Value::Set(set) => set.len(),
                _ => panic!("expected set"),
            })
            .collect();
        assert_eq!(sizes, vec![0, 1, 1, 1, 2, 2, 2, 3]);
    }

    #[test]
    fn test_tight_constraints() {
        let base = vec![Value::SmallInt(1), Value::SmallInt(2), Value::SmallInt(3)];
        let superset = Value::set(vec![Value::SmallInt(1), Value::SmallInt(2)]);
        let subset = Value::set(vec![Value::SmallInt(1)]);

        let results = generate_constrained_subsets(&base, &superset, &subset).unwrap();
        assert_eq!(results.len(), 2);
        assert!(results.contains(&Value::set(vec![Value::SmallInt(1)])));
        assert!(results.contains(&Value::set(vec![Value::SmallInt(1), Value::SmallInt(2)])));
    }

    #[test]
    fn nonmaterialized_or_nonset_bounds_fall_back() {
        let base = vec![Value::SmallInt(1)];
        let eager = Value::set(base.clone());
        let empty = Value::empty_set();
        let interval =
            tla_value::range_set(&num_bigint::BigInt::from(1), &num_bigint::BigInt::from(1));

        assert!(generate_constrained_subsets(&base, &eager, &Value::SmallInt(1)).is_none());
        assert!(generate_constrained_subsets(&base, &Value::SmallInt(1), &empty).is_none());
        assert!(generate_constrained_subsets(&base, &eager, &interval).is_none());
    }

    #[test]
    fn test_unsatisfiable_x_not_in_t() {
        let base = vec![Value::SmallInt(1), Value::SmallInt(2)];
        let superset = Value::set(vec![Value::SmallInt(1)]);
        let subset = Value::set(vec![Value::SmallInt(2)]);

        let results = generate_constrained_subsets(&base, &superset, &subset).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn test_all_free() {
        let base = vec![Value::SmallInt(1), Value::SmallInt(2)];
        let superset = Value::set(base.clone());
        let subset = Value::empty_set();

        let results = generate_constrained_subsets(&base, &superset, &subset).unwrap();
        assert_eq!(results.len(), 4);
        let sizes: Vec<usize> = results
            .iter()
            .map(|value| match value {
                Value::Set(set) => set.len(),
                _ => panic!("expected set"),
            })
            .collect();
        assert_eq!(sizes, vec![0, 1, 1, 2]);
    }
}
