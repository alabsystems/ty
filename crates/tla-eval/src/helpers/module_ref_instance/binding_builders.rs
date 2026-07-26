// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Binding chain construction for INSTANCE substitutions.
//!
//! Contains lazy substitution binding builders and AST utility wrappers
//! (`expr_has_any_prime`, `expr_has_primed_param`).
//!
//! Part of #1643 (module_ref.rs decomposition).

use super::super::super::OpEvalDeps;
use crate::binding_chain::{BindingChain, BindingValue, LazyBinding};
use std::sync::Arc;
use tla_core::ast::{Expr, Substitution};
use tla_core::name_intern::intern_name;
use tla_core::OpEnv;

/// Focused same-binary A/B switch for shared INSTANCE lazy allocations.
#[inline]
fn shared_subst_lazy_disabled() -> bool {
    use std::sync::OnceLock;
    static FLAG: OnceLock<bool> = OnceLock::new();
    *FLAG.get_or_init(|| std::env::var_os("TY_NO_SHARED_SUBST_LAZY").is_some())
}

/// Part of #2991 Step 2: Build lazy substitution bindings — TLC Context.cons parity.
/// Each sub becomes one shared LazyBinding (deferred eval, per-state cache)
/// represented in two chain nodes. Cost: M × O(1) cons.
///
/// SAFETY: `expr_ptr` points into `subs` — caller must keep owning Arc alive
/// (guaranteed by `EvalCtxStable.instance_substitutions`).
/// Raw per-call wrapper without local_ops. Production paths now route through
/// [`build_lazy_subst_bindings_memoized`]; this remains for tests.
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn build_lazy_subst_bindings(
    def_site_chain: &BindingChain,
    subs: &[Substitution],
) -> BindingChain {
    crate::cache::subst_chain_memo::count_raw_build(subs.len());
    build_lazy_subst_bindings_with_local_ops(def_site_chain, None, subs)
}

/// Memoized variant of [`build_lazy_subst_bindings_with_local_ops`] for call
/// sites whose substitution list lives behind a run-stable
/// `Arc<Vec<Substitution>>` (module-ref scope cache entries, `SubstIn` scope
/// cache entries). Shares one canonical chain across all calls whose def-site
/// chain is provably inert for the key; see `cache::subst_chain_memo` for the
/// soundness argument. Falls back to the raw per-call builder for non-inert
/// chains and under `TY_NO_SUBST_MEMO=1`.
pub(crate) fn build_lazy_subst_bindings_memoized(
    ctx: &crate::core::EvalCtx,
    def_site_chain: &BindingChain,
    def_site_local_ops: Option<Arc<OpEnv>>,
    subs_arc: &Arc<Vec<Substitution>>,
    site_kind: u8,
    site_id: u64,
) -> BindingChain {
    crate::cache::subst_chain_memo::subst_chain_memoized(
        ctx,
        def_site_chain,
        def_site_local_ops,
        subs_arc,
        site_kind,
        site_id,
    )
}

pub(crate) fn build_lazy_subst_bindings_with_local_ops(
    def_site_chain: &BindingChain,
    def_site_local_ops: Option<Arc<OpEnv>>,
    subs: &[Substitution],
) -> BindingChain {
    let mut chain = BindingChain::empty();
    let mut enclosing_chain = def_site_chain.promote_all_to_heap();
    let shared_lazy_disabled = shared_subst_lazy_disabled();
    // Reverse iteration: first sub consed last → ends up at head → found first.
    for sub in subs.iter().rev() {
        let name_id = intern_name(sub.from.node.as_str());
        if shared_lazy_disabled {
            let lazy = LazyBinding::new_with_local_ops(
                std::ptr::addr_of!(sub.to),
                &enclosing_chain,
                def_site_local_ops.clone(),
            );
            chain = chain.cons_with_deps(
                name_id,
                BindingValue::Lazy(Box::new(lazy)),
                OpEvalDeps::default(),
            );
            let enclosing_lazy = LazyBinding::new_with_local_ops(
                std::ptr::addr_of!(sub.to),
                &enclosing_chain,
                def_site_local_ops.clone(),
            );
            enclosing_chain = enclosing_chain.cons_with_deps(
                name_id,
                BindingValue::Lazy(Box::new(enclosing_lazy)),
                OpEvalDeps::default(),
            );
        } else {
            // The visible binding and its duplicate in the enclosing chain
            // have the same expression, scope, and local operators. Sharing
            // also gives both positions the same sound per-state cache key.
            let lazy = Arc::new(LazyBinding::new_with_local_ops(
                std::ptr::addr_of!(sub.to),
                &enclosing_chain,
                def_site_local_ops.clone(),
            ));
            chain = chain.cons_with_deps(
                name_id,
                BindingValue::SharedLazy(Arc::clone(&lazy)),
                OpEvalDeps::default(),
            );
            enclosing_chain = enclosing_chain.cons_with_deps(
                name_id,
                BindingValue::SharedLazy(lazy),
                OpEvalDeps::default(),
            );
        }
    }
    chain
}

/// Check if an expression contains ANY Prime expressions (primed variables).
/// Used to determine whether operator result caching needs next_state context.
pub fn expr_has_any_prime(expr: &Expr) -> bool {
    tla_core::expr_has_any_prime_legacy_v(expr)
}

/// Check if an expression contains a primed reference to a specific parameter name.
/// This is used to detect when call-by-name semantics are needed for operator evaluation.
///
/// For example, in `Action1(c,d) == c' = [c EXCEPT ![1] = d']`:
/// - `expr_has_primed_param(body, "d")` returns true because `d'` appears
/// - `expr_has_primed_param(body, "c")` returns false (c appears but not primed as c')
pub fn expr_has_primed_param(expr: &Expr, param_name: &str) -> bool {
    tla_core::expr_contains_primed_param_v(expr, param_name)
}
