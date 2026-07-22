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

/// Part of #2991 Step 2: Build lazy substitution bindings — TLC Context.cons parity.
/// Each sub becomes a LazyBinding (deferred eval, OnceLock cache). Cost: M × O(1) cons.
///
/// SAFETY: `expr_ptr` points into `subs` — caller must keep owning Arc alive
/// (guaranteed by `EvalCtxStable.instance_substitutions`).
pub(crate) fn build_lazy_subst_bindings(
    def_site_chain: &BindingChain,
    subs: &[Substitution],
) -> BindingChain {
    build_lazy_subst_bindings_with_local_ops(def_site_chain, None, subs)
}

pub(crate) fn build_lazy_subst_bindings_with_local_ops(
    def_site_chain: &BindingChain,
    def_site_local_ops: Option<Arc<OpEnv>>,
    subs: &[Substitution],
) -> BindingChain {
    let mut chain = BindingChain::empty();
    let mut enclosing_chain = def_site_chain.promote_all_to_heap();
    // Reverse iteration: first sub consed last → ends up at head → found first.
    for sub in subs.iter().rev() {
        let name_id = intern_name(sub.from.node.as_str());
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
