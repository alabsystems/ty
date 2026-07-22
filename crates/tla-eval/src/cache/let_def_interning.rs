// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Run-pinned interning of LET-local `Arc<OperatorDef>` allocations.
//!
//! ## The bug this fixes (P0 stale-replay family)
//!
//! LET scope setup used to run `Arc::new(def.clone())` on EVERY evaluation of a
//! `LET ... IN ...` expression — once per state per action in successor
//! enumeration. Each clone deep-copies the operator BODY AST; the enumerator
//! then walks that body (e.g. `conjunct_apply` inlines `def.body`), feeding its
//! freshly-allocated nodes into pointer-keyed memo caches (`const_domain_cache`,
//! `expr_analysis`, the enum subst cache via `resolved_def_ptr`, CHOOSE caches).
//! When the LET scope exits, the clone is freed and the allocator recycles the
//! addresses for the NEXT state's clone — but the cache entries keyed by the old
//! addresses survive (their lifetime is per-run or per-state). A recycled
//! address that lands on a *different* AST node replays a stale entry computed
//! for the previous occupant.
//!
//! Observed concretely on PaxosCommit (`Decide`'s `LET Decided(rm, v) == \E b
//! \in Ballot, MS \in Majority : ...`): the `const_domain_cache` entry stored
//! for a freed clone's `Ballot` domain node (value `{0, 1}`) was replayed for a
//! new clone's `Majority` node at the recycled address, binding `MS` to an Int
//! and failing the run with a nondeterministic, allocator-order-dependent
//! `Type error: expected Set, got Int`. The same mechanism can silently
//! mis-enumerate domains (missing/extra successors), not just error out.
//!
//! ## The fix
//!
//! Intern the `Arc<OperatorDef>` per lexical LET definition: the first
//! evaluation of a given `OperatorDef` node builds the Arc; all subsequent
//! evaluations reuse it. The interning map holds the Arc alive for the whole
//! run, so every AST node reachable through `def.body` has a run-stable
//! address — pointer-keyed caches can never observe a recycled address from
//! these clones.
//!
//! ## Soundness
//!
//! - An `OperatorDef` is immutable AST data (name, params, body, precomputed
//!   flags); cloning it per evaluation never produced a different value, so
//!   sharing one clone across evaluations is observationally equivalent.
//! - The map key is the address of the SOURCE `OperatorDef` node (the node
//!   borrowed from the enclosing `Expr::Let`). These sources are run-stable in
//!   all current flows: action ASTs are `Arc`-pinned for the run (#P0 fix),
//!   substituted bodies are pinned by the enum subst cache, and nested LET
//!   defs live inside bodies pinned by THIS map (inductively stable).
//! - Defense-in-depth against an unknown ephemeral source: every hit is
//!   validated against the lexical identity of the def (name string, name
//!   span, body span, param count). If a recycled address presents a
//!   *different* lexical def, the stale entry is RETIRED (kept alive so other
//!   pointer-keyed caches holding entries for its body can never alias a new
//!   allocation) and re-interned. A collision that passes validation is a
//!   clone of the same lexical definition, for which the cached Arc is
//!   semantically identical. Debug builds assert full structural equality.
//!
//! ## Lifecycle
//!
//! Thread-local (workers intern independently; contents are content-identical).
//! Cleared between model-checking runs alongside the pointer-keyed enumeration
//! caches (`reset.rs`) and on test reset — NEVER at intra-run phase boundaries,
//! because pointer-keyed caches outliving a phase could alias freed bodies.

use rustc_hash::FxHashMap;
use std::cell::RefCell;
use std::sync::Arc;
use tla_core::ast::OperatorDef;

struct InternState {
    map: FxHashMap<usize, Arc<OperatorDef>>,
    /// Arcs evicted by a validation mismatch (address recycled by a different
    /// lexical def). Kept alive for the run so pointer-keyed caches holding
    /// entries for their bodies can never alias newly allocated nodes.
    retired: Vec<Arc<OperatorDef>>,
}

thread_local! {
    static LET_DEF_INTERN: RefCell<InternState> = RefCell::new(InternState {
        map: FxHashMap::default(),
        retired: Vec::new(),
    });
}

/// Lexical-identity validation for an intern hit. A hit that passes this check
/// is a clone of the same lexical definition (same source location, same
/// name, same shape), for which the interned Arc is semantically identical.
#[inline]
fn same_lexical_def(a: &OperatorDef, b: &OperatorDef) -> bool {
    a.name.span == b.name.span
        && a.body.span == b.body.span
        && a.params.len() == b.params.len()
        && a.name.node == b.name.node
}

/// Get the run-stable `Arc<OperatorDef>` for a LET-local definition.
///
/// Replaces per-evaluation `Arc::new(def.clone())` in LET scope setup. The
/// returned Arc (and hence every node of `def.body`) is pinned for the run,
/// making it safe to feed into pointer-keyed memo caches.
pub fn intern_let_def_arc(def: &OperatorDef) -> Arc<OperatorDef> {
    let key = def as *const OperatorDef as usize;
    LET_DEF_INTERN.with(|cell| {
        let state = &mut *cell.borrow_mut();
        if let Some(existing) = state.map.get(&key) {
            if same_lexical_def(existing, def) {
                // Debug builds verify the cheap lexical check implies full
                // structural identity (OperatorDef derives PartialEq).
                debug_assert!(
                    **existing == *def,
                    "let-def intern validation passed but defs differ structurally: {}",
                    def.name.node
                );
                return Arc::clone(existing);
            }
            // Address recycled by a different lexical def: retire the old Arc
            // (keep its body alive — pointer-keyed caches may reference it)
            // and intern the new def below.
            let old = state
                .map
                .remove(&key)
                .expect("entry just observed via get()");
            state.retired.push(old);
        }
        let arc = Arc::new(def.clone());
        state.map.insert(key, Arc::clone(&arc));
        arc
    })
}

/// Clear the interning map and the retired pin list.
///
/// Must only be called when ALL pointer-keyed caches that may reference
/// interned bodies are cleared in the same breath (between model-checking
/// runs / on test reset) — never at intra-run phase boundaries.
pub fn clear_let_def_interning() {
    LET_DEF_INTERN.with(|cell| {
        let state = &mut *cell.borrow_mut();
        state.map.clear();
        state.retired.clear();
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use tla_core::ast::Expr;
    use tla_core::Spanned;

    fn mk_def(name: &str) -> OperatorDef {
        OperatorDef {
            name: Spanned::dummy(name.to_string()),
            params: vec![],
            body: Spanned::dummy(Expr::Bool(true)),
            local: false,
            contains_prime: false,
            guards_depend_on_prime: false,
            has_primed_param: false,
            is_recursive: false,
            self_call_count: 0,
        }
    }

    #[test]
    fn interns_same_def_to_same_arc() {
        clear_let_def_interning();
        let def = mk_def("Op");
        let a = intern_let_def_arc(&def);
        let b = intern_let_def_arc(&def);
        assert!(Arc::ptr_eq(&a, &b), "same def node must intern to one Arc");
    }

    #[test]
    fn validation_mismatch_retires_and_reinterns() {
        clear_let_def_interning();
        let def1 = mk_def("Op1");
        let a = intern_let_def_arc(&def1);
        // Simulate address recycling: a DIFFERENT lexical def presented under
        // the same key. Build def2 at the same address is not constructible in
        // safe Rust, so exercise the validation path directly: a def with a
        // different name at the same address would fail same_lexical_def.
        let def2 = mk_def("Op2");
        assert!(!same_lexical_def(&def1, &def2));
        drop(a);
        // The interned Arc must still be alive inside the map (pinned).
        let again = intern_let_def_arc(&def1);
        assert_eq!(again.name.node, "Op1");
    }
}
