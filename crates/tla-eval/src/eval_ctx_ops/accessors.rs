// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! EvalCtx field accessors, shared-ctx convenience delegates, and scope guards.
//! Part of #2764 / #1643.

use crate::cache::bump_hoist_state_generation_ctx;
use crate::var_index::VarRegistry;
use crate::{Env, EvalCtx, HashMap, InstanceInfo, OpEnv, ScopeGuard, StateIdentityGuard, Value};
use std::sync::Arc;
use tla_core::name_intern::NameId;
use tla_core::span::Span;

impl EvalCtx {
    // ---- Field accessor methods (Part of #2739) ----
    // These provide controlled access to EvalCtx fields from downstream crates
    // (tla-check) after the fields are changed from `pub` to `pub(crate)`.

    /// Get a reference to the shared context.
    #[inline]
    pub fn shared(&self) -> &Arc<super::super::SharedCtx> {
        &self.shared
    }

    /// Get a mutable reference to the shared context Arc.
    ///
    /// Used with `Arc::make_mut()` for copy-on-write mutation of SharedCtx.
    #[inline]
    pub fn shared_arc_mut(&mut self) -> &mut Arc<super::super::SharedCtx> {
        &mut self.stable_mut().shared
    }

    /// Get a reference to the variable environment.
    #[inline]
    pub fn env(&self) -> &Env {
        &self.stable.env
    }

    /// Get a mutable reference to the variable environment (copy-on-write via Arc).
    /// Note: Arc::make_mut internally checks refcount — when refcount == 1, it
    /// returns a direct &mut with no clone (just an atomic CAS). The if-let-else
    /// Arc::get_mut pattern can't be used here due to borrow checker limitations
    /// with returning mutable references across branches.
    #[inline]
    pub fn env_mut(&mut self) -> &mut Env {
        Arc::make_mut(&mut self.stable_mut().env)
    }

    /// Get a reference to the next-state environment.
    #[inline]
    pub fn next_state(&self) -> &Option<Arc<Env>> {
        &self.stable.next_state
    }

    /// Get a mutable reference to the next-state environment.
    #[inline]
    pub fn next_state_mut(&mut self) -> &mut Option<Arc<Env>> {
        &mut self.stable_mut().next_state
    }

    /// Get a reference to the local operator definitions.
    #[inline]
    pub fn local_ops(&self) -> &Option<Arc<OpEnv>> {
        &self.stable.local_ops
    }

    /// Get a mutable reference to the local operator definitions.
    ///
    /// Part of #3099: Invalidates `scope_ids.local_ops` so that cache key
    /// builders lazily recompute the scope id from the mutated content.
    #[inline]
    pub fn local_ops_mut(&mut self) -> &mut Option<Arc<OpEnv>> {
        let s = self.stable_mut();
        s.scope_ids.local_ops = crate::cache::scope_ids::INVALIDATED;
        // The content is about to change; clear the cached recursive flag so the
        // INVALIDATED resolve path recomputes it from the mutated content.
        s.scope_ids.local_ops_recursive = false;
        &mut s.local_ops
    }

    /// Get the currently-stored `local_ops` scope id (may be the `INVALIDATED`
    /// sentinel). Used to save/restore the enclosing scope id across LET blocks.
    #[inline]
    pub fn local_ops_scope_id(&self) -> u64 {
        self.stable.scope_ids.local_ops
    }

    /// Set `local_ops` together with an already-computed scope id, avoiding the
    /// `INVALIDATED` sentinel that forces every subsequent cache-key build to
    /// re-walk the (immutable) `local_ops` HAMT.
    ///
    /// The scope id MUST be value-identical to
    /// `compute_local_ops_scope_id(&local_ops)` for the supplied map, otherwise
    /// cache keys would alias distinct scopes. Callers obtain it from
    /// `let_scope_id_memoized` / `compute_local_ops_scope_id`.
    ///
    /// This is the eager-id counterpart to `*ctx.local_ops_mut() = Some(arc)`:
    /// the scope id is computed once at scope construction instead of being
    /// recomputed from scratch on every cache lookup inside the scope.
    #[inline]
    pub fn set_local_ops_with_id(&mut self, local_ops: Arc<OpEnv>, scope_id: u64) {
        // Pointer-memoized recursive check (pure function of the pinned Arc).
        let recursive = crate::cache::openv_memo::scope_id_and_recursive_memoized(
            &local_ops,
            crate::cache::openv_memo::ScopeIdSite::SetWithId,
        )
        .1;
        let s = self.stable_mut();
        s.local_ops = Some(local_ops);
        s.scope_ids.local_ops = scope_id;
        s.scope_ids.local_ops_recursive = recursive;
    }

    /// Restore a previously-saved `local_ops` value together with its scope id.
    ///
    /// Used to restore the enclosing scope after a LET/INSTANCE block. The id
    /// passed here is the id that was valid for `local_ops` before the block was
    /// entered (saved alongside it via `local_ops_scope_id`), so no
    /// recomputation is required.
    #[inline]
    pub fn restore_local_ops_with_id(&mut self, local_ops: Option<Arc<OpEnv>>, scope_id: u64) {
        // Pointer-memoized recursive check: this runs on every LET/INSTANCE
        // scope exit; the previous `local_ops_recursive_flag` re-walked the
        // (unchanged) outer HAMT each time.
        let recursive = crate::cache::openv_memo::recursive_flag_memoized(
            &local_ops,
            crate::cache::openv_memo::ScopeIdSite::Restore,
        );
        let s = self.stable_mut();
        s.local_ops = local_ops;
        s.scope_ids.local_ops = scope_id;
        s.scope_ids.local_ops_recursive = recursive;
    }

    /// Set `local_ops` and compute its scope id eagerly (a single HAMT walk),
    /// instead of leaving the scope id `INVALIDATED` (which would force a walk on
    /// *every* cache lookup inside the scope).
    ///
    /// Used by INSTANCE/operator-merge scope entries where the merged map is not
    /// a static LET block (so the LET-block memo does not apply). The stored id
    /// is value-identical to `compute_local_ops_scope_id(&local_ops)`.
    #[inline]
    pub fn set_local_ops_eager(&mut self, local_ops: Arc<OpEnv>) {
        let (id, recursive) =
            crate::cache::scope_ids::compute_local_ops_scope_id_and_recursive(&local_ops);
        let s = self.stable_mut();
        s.local_ops = Some(local_ops);
        s.scope_ids.local_ops = id;
        s.scope_ids.local_ops_recursive = recursive;
    }

    /// Check if prime validation should be skipped.
    #[inline]
    pub fn skip_prime_validation(&self) -> bool {
        self.stable.skip_prime_validation
    }

    /// Set whether prime validation should be skipped.
    #[inline]
    pub fn set_skip_prime_validation(&mut self, skip: bool) {
        self.stable_mut().skip_prime_validation = skip;
    }

    /// Get a reference to the variable registry
    #[inline]
    pub fn var_registry(&self) -> &VarRegistry {
        &self.shared.var_registry
    }

    /// Resolve operator name through replacements (for config `CONSTANT Op <- Replacement`)
    ///
    /// This is critical for compiled_guard.rs to properly handle operator replacements
    /// when extracting next-state assignments from actions.
    pub fn resolve_op_name<'a>(&'a self, name: &'a str) -> &'a str {
        self.shared
            .op_replacements
            .get(name)
            .map_or(name, std::string::String::as_str)
    }

    // ---- Scope guards ----

    /// RAII guard that saves `env` and restores on drop. Part of #2738.
    /// Part of #3407: Bumps hoist state generation for scope-level protection.
    pub fn scope_guard(&mut self) -> ScopeGuard {
        let hoist_guard = bump_hoist_state_generation_ctx(self);
        let saved_env = Arc::clone(&self.env);
        ScopeGuard {
            ctx: self as *mut EvalCtx,
            saved_env,
            saved_next_state: None,
            _hoist_guard: hoist_guard,
            _state_identity_guard: None,
        }
    }

    /// Create an RAII guard that saves both `env` and `next_state`, restoring
    /// both on drop.
    ///
    /// Use this variant when the code between save/restore also modifies
    /// `next_state` (common in liveness property checking).
    ///
    /// Part of #2738, #3407.
    pub fn scope_guard_with_next_state(&mut self) -> ScopeGuard {
        let hoist_guard = bump_hoist_state_generation_ctx(self);
        let saved_env = Arc::clone(&self.env);
        let saved_next = self.next_state.clone();
        ScopeGuard {
            ctx: self as *mut EvalCtx,
            saved_env,
            saved_next_state: Some(saved_next),
            _hoist_guard: hoist_guard,
            _state_identity_guard: Some(StateIdentityGuard::restore_only()),
        }
    }

    // ---- Convenience shared-ctx accessors ----

    /// The module's operator definitions (delegates to [`SharedCtx::ops`](crate::SharedCtx)).
    #[inline]
    pub fn ops(&self) -> &OpEnv {
        &self.shared.ops
    }
    /// Named INSTANCE bindings, keyed by instance name (e.g. `InChan` for
    /// `InChan == INSTANCE Channel WITH ...`).
    #[inline]
    pub fn instances(&self) -> &HashMap<String, InstanceInfo> {
        &self.shared.instances
    }
    /// Operators imported from INSTANCE'd modules, keyed by module name (not yet
    /// substituted with the instance's `WITH` substitutions).
    #[inline]
    pub fn instance_ops(&self) -> &HashMap<String, OpEnv> {
        &self.shared.instance_ops
    }
    /// Config-driven operator replacements (`old_name -> new_name`) from
    /// `CONSTANT Op <- Replacement` directives.
    #[inline]
    pub fn op_replacements(&self) -> &HashMap<String, String> {
        &self.shared.op_replacements
    }
    /// Pre-evaluated values of zero-arity constant-level operators, keyed by
    /// interned operator [`NameId`], for O(1) identifier lookup.
    #[inline]
    pub fn precomputed_constants(&self) -> &HashMap<NameId, Value> {
        self.shared.precomputed_constants()
    }
    /// Whether `span` is the subscript of a lowered `[A]_v` / `<<A>>_v` action,
    /// preserving the syntactic distinction after lowering expands the form.
    #[inline]
    pub fn is_action_subscript_span(&self, span: Span) -> bool {
        self.shared.action_subscript_spans.contains(&span)
    }
}
