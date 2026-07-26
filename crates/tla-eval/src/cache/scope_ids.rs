// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Stable scope identity for evaluator caches.
//!
//! Part of #3099: Computes content-based u64 fingerprints for
//! `instance_substitutions` and non-recursive `local_ops` scopes. Recursive
//! `LET` operator environments fall back to `Arc` identity so cache keys
//! do not alias distinct captured bindings across recursion levels.
//!
//! The fingerprints are computed once at scope construction time. Cache
//! lookups read the stored u64 from `EvalCtxStable.scope_ids`.

use crate::core::OpEnv;
use std::hash::{Hash, Hasher};
use std::sync::Arc;
use tla_core::ast::{Expr, Substitution};
use tla_core::name_intern::{intern_name, NameId};

/// Sentinel value indicating the scope id was invalidated by a direct
/// mutation (e.g., `local_ops_mut()`). Cache key builders detect this
/// and lazily recompute from the current scope content.
pub(crate) const INVALIDATED: u64 = u64::MAX;

/// Stable scope identity for cache keys.
///
/// `None` / empty scope produces id `0`. Logically identical non-empty
/// scopes produce the same id. Different scope content produces different
/// ids (with astronomically low collision probability via SipHash).
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct EvalScopeIds {
    pub(crate) local_ops: u64,
    pub(crate) instance_substitutions: u64,
    /// Cached `local_ops_requires_arc_identity(local_ops)` for the current
    /// `local_ops` scope. Recomputing this on every cache-key build is a full
    /// `OpEnv` HAMT walk (`ops.values().any(..)`); profiling btree showed it at
    /// ~24% of single-thread CPU. The flag is a pure function of the immutable
    /// scope content, so it is computed once when the scope id is set and read
    /// O(1) by [`resolve_local_ops_id_with_recursive`].
    ///
    /// `false` when `local_ops` is `None`/empty/non-recursive. When the id is
    /// `INVALIDATED` (direct mutation) this flag is ignored and the recursive
    /// check is recomputed from the live content.
    pub(crate) local_ops_recursive: bool,
}

/// Compute a stable u64 fingerprint for a `local_ops` OpEnv.
///
/// Key shape: sorted (NameId, body.span.start, body.span.end, params.len())
/// per operator definition. Sorting by NameId ensures deterministic hashing
/// regardless of HashMap iteration order.
pub(crate) fn compute_local_ops_id(ops: &OpEnv) -> u64 {
    if ops.is_empty() {
        return 0;
    }
    // Collect fingerprint tuples and sort by NameId for determinism.
    let mut entries: Vec<(NameId, u32, u32, usize)> = ops
        .iter()
        .map(|(name, def)| {
            let name_id = intern_name(name);
            (
                name_id,
                def.body.span.start,
                def.body.span.end,
                def.params.len(),
            )
        })
        .collect();
    entries.sort_unstable();

    let mut hasher = rustc_hash::FxHasher::default();
    entries.len().hash(&mut hasher);
    for (nid, start, end, arity) in &entries {
        nid.hash(&mut hasher);
        start.hash(&mut hasher);
        end.hash(&mut hasher);
        arity.hash(&mut hasher);
    }
    hasher.finish()
}

/// Recursive `LET` operators change captured variable bindings at each recursion
/// level. Content-based fingerprinting would equate two levels with identical
/// operator bodies but different bound variables, causing incorrect cache hits.
/// Fall back to `Arc` pointer identity so each recursion frame gets a unique key.
#[inline]
pub(crate) fn local_ops_requires_arc_identity(ops: &OpEnv) -> bool {
    ops.values().any(|def| def.is_recursive)
}

/// Compute the effective cache scope id for a shared `local_ops` environment.
///
/// Recursive `LET` operators can capture different variable bindings at each
/// recursion level even when the operator bodies are identical. Those scopes
/// keep `Arc` pointer identity so different recursion frames cannot alias each
/// other in `SUBST_CACHE` / `NARY_OP_CACHE`.
#[inline]
pub(crate) fn compute_local_ops_scope_id(local_ops: &Arc<OpEnv>) -> u64 {
    if local_ops.is_empty() {
        0
    } else if local_ops_requires_arc_identity(local_ops) {
        Arc::as_ptr(local_ops) as usize as u64
    } else {
        compute_local_ops_id(local_ops)
    }
}

/// Whether a `local_ops` option requires Arc-identity scope keying.
///
/// `true` only for a non-empty scope containing at least one recursive operator
/// definition. This is the value cached in
/// [`EvalScopeIds::local_ops_recursive`] so the (single) `OpEnv` HAMT walk runs
/// once per scope construction instead of once per cache-key build.
#[inline]
#[cfg(test)]
pub(crate) fn local_ops_recursive_flag(local_ops: &Option<Arc<OpEnv>>) -> bool {
    local_ops
        .as_ref()
        .is_some_and(|ops| local_ops_requires_arc_identity(ops))
}

/// Compute the scope id and the recursive flag for `local_ops` in a single HAMT
/// walk. Construction sites use this to populate both
/// [`EvalScopeIds::local_ops`] and [`EvalScopeIds::local_ops_recursive`]
/// consistently.
#[inline]
pub(crate) fn compute_local_ops_scope_id_and_recursive(local_ops: &Arc<OpEnv>) -> (u64, bool) {
    if local_ops.is_empty() {
        (0, false)
    } else if local_ops_requires_arc_identity(local_ops) {
        (Arc::as_ptr(local_ops) as usize as u64, true)
    } else {
        (compute_local_ops_id(local_ops), false)
    }
}

/// Memoization key for the merged scope id produced by entering a `LET` block.
///
/// A LET block merges the (static) enclosing `local_ops` scope with the (static)
/// set of LET definitions. The resulting scope content is therefore a pure
/// function of:
///   - `outer_id`: the content fingerprint of the enclosing `local_ops` scope,
///   - the static identity of the LET block (its first definition's source span,
///     which uniquely identifies one `LET ... IN` construct in the source, plus
///     the definition count as a cheap robustness discriminator).
///
/// The enumeration hot path rebuilds an `Arc<OpEnv>` with identical content on
/// every state (clone outer ops + insert the same static defs), so the merged
/// scope id is value-identical every state. Memoizing on this static key turns
/// an O(merged_ops) HAMT walk per state into one walk per distinct LET block /
/// outer scope.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct LetScopeMemoKey {
    outer_id: u64,
    first_def_file: u32,
    first_def_start: u32,
    first_def_end: u32,
    defs_len: u32,
}

std::thread_local! {
    /// Thread-local memo: LET-block static identity -> merged scope id.
    /// Thread-local so it is correct under `--workers N` (each worker thread has
    /// its own evaluator caches and clears them at run/test reset). The stored
    /// value is a pure function of static AST + outer scope id, so cross-thread
    /// sharing would also be sound, but TLS keeps it lock-free on the hot path.
    static LET_SCOPE_ID_MEMO: std::cell::RefCell<rustc_hash::FxHashMap<LetScopeMemoKey, u64>> =
        std::cell::RefCell::new(rustc_hash::FxHashMap::default());
}

/// Cap on memo entries to bound memory; LET-block identities are O(spec size),
/// so this is effectively never hit, but it prevents pathological growth.
const LET_SCOPE_ID_MEMO_CAP: usize = 65_536;

/// Compute (and memoize) the merged scope id for a `LET` block.
///
/// `merged` is the freshly-built `Arc<OpEnv>` (outer ops + LET defs). `outer_id`
/// is the already-resolved scope id of the enclosing `local_ops`. `defs` are the
/// static LET definitions whose source span identifies the block.
///
/// Soundness: the returned id is computed by [`compute_local_ops_scope_id`] on
/// first encounter and reused thereafter only for LET blocks with the *same*
/// static identity AND the *same* outer scope id — i.e., logically identical
/// merged content. Recursive LET blocks (which require per-allocation `Arc`
/// identity, see [`local_ops_requires_arc_identity`]) are NOT memoized: they
/// must yield a fresh id per allocation, so we always delegate.
pub(crate) fn let_scope_id_memoized<D>(
    merged: &Arc<OpEnv>,
    outer_id: u64,
    defs: &[D],
    first_def_span: Option<(u32, u32, u32)>,
) -> u64 {
    // Recursive LET scopes must keep per-allocation Arc identity — never memoize.
    if local_ops_requires_arc_identity(merged) {
        return compute_local_ops_scope_id(merged);
    }
    // Without a stable source span to key on, fall back to a direct (correct,
    // unmemoized) computation rather than risk an unsound key.
    let Some((first_def_file, first_def_start, first_def_end)) = first_def_span else {
        return compute_local_ops_scope_id(merged);
    };
    let key = LetScopeMemoKey {
        outer_id,
        first_def_file,
        first_def_start,
        first_def_end,
        defs_len: defs.len() as u32,
    };
    LET_SCOPE_ID_MEMO.with(|memo| {
        if let Some(&id) = memo.borrow().get(&key) {
            return id;
        }
        let id = compute_local_ops_scope_id(merged);
        let mut m = memo.borrow_mut();
        if m.len() >= LET_SCOPE_ID_MEMO_CAP {
            m.clear();
        }
        m.insert(key, id);
        id
    })
}

/// Clear the LET-scope-id memo (run/test reset). The memo only holds pure
/// functions of static AST + scope ids, so clearing is purely a memory reset.
pub(crate) fn clear_let_scope_id_memo() {
    LET_SCOPE_ID_MEMO.with(|memo| memo.borrow_mut().clear());
}

/// Hash immediate scalar content of an expression without recursing into children.
///
/// Provides O(1) discrimination between expressions that share the same
/// discriminant and span. Only leaf variants carry hashable content; compound
/// variants rely on span + discriminant (which is correct when source spans differ).
///
/// Part of #3406: fixes aliasing of `Spanned::dummy` INSTANCE substitutions.
fn hash_expr_shallow(hasher: &mut impl Hasher, expr: &Expr) {
    std::mem::discriminant(expr).hash(hasher);
    match expr {
        Expr::Bool(b) => b.hash(hasher),
        Expr::Int(n) => n.hash(hasher),
        Expr::String(s) => s.hash(hasher),
        Expr::Ident(_, name_id) => name_id.hash(hasher),
        Expr::StateVar(_, idx, name_id) => {
            idx.hash(hasher);
            name_id.hash(hasher);
        }
        Expr::OpRef(s) => s.hash(hasher),
        // Compound variants: discriminant already hashed above.
        // In real parsed specs, these have distinct source spans.
        // The Spanned::dummy collision vector only affects leaf values.
        _ => {}
    }
}

/// Compute a stable u64 fingerprint for instance substitutions.
///
/// Key shape: ordered (from_name_id, to.span.start, to.span.end, shallow_expr_hash)
/// per substitution entry. Order preserved from the `Vec<Substitution>`.
///
/// Fix #3406: span + discriminant aliased expressions of same type at same location
/// (e.g., `Int(1)` vs `Int(2)` both with `Spanned::dummy`). Shallow content hash
/// includes immediate scalar content for leaf variants, providing O(1) discrimination
/// without recursing into children (avoiding the #3123 timeout regression).
pub(crate) fn compute_instance_subs_id(subs: &[Substitution]) -> u64 {
    if subs.is_empty() {
        return 0;
    }
    let mut hasher = rustc_hash::FxHasher::default();
    subs.len().hash(&mut hasher);
    for sub in subs {
        let name_id = intern_name(&sub.from.node);
        name_id.hash(&mut hasher);
        sub.to.span.start.hash(&mut hasher);
        sub.to.span.end.hash(&mut hasher);
        hash_expr_shallow(&mut hasher, &sub.to.node);
    }
    hasher.finish()
}

/// Resolve the effective local_ops scope id from context fields.
///
/// Returns the stored scope id if valid, or lazily recomputes from the
/// current `local_ops` content when the id is INVALIDATED (set by
/// `local_ops_mut()` direct mutations).
///
/// Recursive `local_ops` scopes always resolve from the current `Arc` so
/// callers that swap in a different captured scope do not need to keep the
/// stored id in sync manually.
///
/// # Performance
///
/// The recursive-vs-content decision (`local_ops_requires_arc_identity`) is an
/// `ops.values().any(..)` walk over the persistent `OpEnv` HAMT. Profiling btree
/// (which has a recursive LET operator, `FindLeafNode`) showed this walk consumed
/// ~24% of single-thread CPU because it ran on *every* cache-key build. The
/// decision is a pure function of the immutable scope content, so callers that
/// already know it pass it via [`resolve_local_ops_id_with_recursive`] to skip
/// the walk entirely; this wrapper performs the walk only as a fallback.
#[inline]
#[cfg(test)]
pub(crate) fn resolve_local_ops_id(scope_id: u64, local_ops: &Option<Arc<OpEnv>>) -> u64 {
    let requires_arc_identity = local_ops
        .as_ref()
        .is_some_and(|ops| local_ops_requires_arc_identity(ops));
    resolve_local_ops_id_with_recursive(scope_id, local_ops, requires_arc_identity)
}

/// `resolve_local_ops_id` with a precomputed `requires_arc_identity` flag.
///
/// `requires_arc_identity` must equal
/// `local_ops.as_ref().is_some_and(local_ops_requires_arc_identity)` for the
/// supplied scope. Callers that cache this flag at scope-construction time (it is
/// a pure function of the immutable `OpEnv` content) use this entry point to avoid
/// re-walking the `OpEnv` HAMT on the hot cache-key path.
///
/// # Soundness
///
/// - Recursive scopes (`requires_arc_identity == true`) resolve from the current
///   `Arc` exactly as before — preserving per-recursion-level identity so distinct
///   captured-binding frames never alias in the caches.
/// - Non-recursive scopes return the stored `scope_id` when valid, or recompute
///   the content fingerprint when `INVALIDATED`.
#[inline]
pub(crate) fn resolve_local_ops_id_with_recursive(
    scope_id: u64,
    local_ops: &Option<Arc<OpEnv>>,
    requires_arc_identity: bool,
) -> u64 {
    if requires_arc_identity {
        // Recursive scope: Arc identity (O(1), no HAMT walk). `local_ops` is
        // Some here because `requires_arc_identity` can only be true for a
        // non-empty scope, but fall back defensively just in case.
        return local_ops
            .as_ref()
            .map_or(0, |ops| Arc::as_ptr(ops) as usize as u64);
    }
    if scope_id != INVALIDATED {
        return scope_id;
    }
    // Invalidated, non-recursive: recompute the content fingerprint.
    local_ops.as_ref().map_or(0, compute_local_ops_scope_id)
}

/// Resolve the effective instance_substitutions scope id from context fields.
///
/// Returns the stored scope id if valid, or lazily recomputes from the
/// current `instance_substitutions` content when the id is INVALIDATED.
#[inline]
pub(crate) fn resolve_instance_subs_id(
    scope_id: u64,
    instance_subs: &Option<std::sync::Arc<Vec<Substitution>>>,
) -> u64 {
    if scope_id != INVALIDATED {
        scope_id
    } else {
        instance_subs
            .as_ref()
            .map_or(0, |s| compute_instance_subs_id(s))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tla_core::ast::OperatorDef;
    use tla_core::Spanned;

    fn make_def(name: &str, body_start: u32, body_end: u32, local: bool) -> Arc<OperatorDef> {
        let fid = tla_core::span::FileId(0);
        Arc::new(OperatorDef {
            name: Spanned::new(name.into(), tla_core::Span::new(fid, 0, 3)),
            params: vec![],
            body: Spanned::new(
                tla_core::ast::Expr::Bool(true),
                tla_core::Span::new(fid, body_start, body_end),
            ),
            local,
            contains_prime: false,
            guards_depend_on_prime: false,
            is_recursive: false,
            has_primed_param: false,
            self_call_count: 0,
        })
    }

    /// Create a recursive operator def (is_recursive=true, local=false).
    /// Models a `LET RECURSIVE SA[bb \in Ballot] == ...` pattern.
    fn make_recursive_def(name: &str, body_start: u32, body_end: u32) -> Arc<OperatorDef> {
        let fid = tla_core::span::FileId(0);
        Arc::new(OperatorDef {
            name: Spanned::new(name.into(), tla_core::Span::new(fid, 0, 3)),
            params: vec![],
            body: Spanned::new(
                tla_core::ast::Expr::Bool(true),
                tla_core::Span::new(fid, body_start, body_end),
            ),
            local: false,
            contains_prime: false,
            guards_depend_on_prime: false,
            is_recursive: true,
            has_primed_param: false,
            self_call_count: 1,
        })
    }

    #[test]
    fn test_empty_ops_returns_zero() {
        let ops = OpEnv::new();
        assert_eq!(compute_local_ops_id(&ops), 0);
    }

    #[test]
    fn test_empty_subs_returns_zero() {
        let subs: Vec<Substitution> = vec![];
        assert_eq!(compute_instance_subs_id(&subs), 0);
    }

    #[test]
    fn test_same_ops_same_id() {
        let mut ops1 = OpEnv::new();
        let mut ops2 = OpEnv::new();
        let def = make_def("foo", 10, 14, false);
        ops1.insert("foo".into(), Arc::clone(&def));
        ops2.insert("foo".into(), Arc::clone(&def));
        // Different Arc wrappers, same content → same id
        assert_eq!(compute_local_ops_id(&ops1), compute_local_ops_id(&ops2));
    }

    #[test]
    fn test_different_ops_different_id() {
        let mut ops1 = OpEnv::new();
        let mut ops2 = OpEnv::new();
        let def1 = make_def("foo", 10, 14, false);
        let def2 = make_def("bar", 20, 24, false);
        ops1.insert("foo".into(), def1);
        ops2.insert("bar".into(), def2);
        assert_ne!(compute_local_ops_id(&ops1), compute_local_ops_id(&ops2));
    }

    #[test]
    fn test_nonlocal_scope_uses_content_id_across_reconstructed_arcs() {
        let mut ops1 = OpEnv::new();
        let mut ops2 = OpEnv::new();
        ops1.insert("foo".into(), make_def("foo", 10, 14, false));
        ops2.insert("foo".into(), make_def("foo", 10, 14, false));
        let arc1 = Arc::new(ops1);
        let arc2 = Arc::new(ops2);

        assert_eq!(
            compute_local_ops_scope_id(&arc1),
            compute_local_ops_scope_id(&arc2)
        );
    }

    #[test]
    fn test_recursive_scope_uses_arc_identity_across_reconstructed_arcs() {
        let mut ops1 = OpEnv::new();
        let mut ops2 = OpEnv::new();
        ops1.insert("foo".into(), make_recursive_def("foo", 10, 14));
        ops2.insert("foo".into(), make_recursive_def("foo", 10, 14));
        let arc1 = Arc::new(ops1);
        let arc2 = Arc::new(ops2);

        // Identical recursive operator bodies but different Arc allocations →
        // different scope ids (Arc identity prevents cross-level aliasing).
        assert_ne!(
            compute_local_ops_scope_id(&arc1),
            compute_local_ops_scope_id(&arc2)
        );
    }

    /// Part of #3156: local-only (non-recursive) operators now use content hash,
    /// since the LOCAL keyword alone does not create scope aliasing risk.
    #[test]
    fn test_local_only_nonrecursive_uses_content_hash() {
        let mut ops1 = OpEnv::new();
        let mut ops2 = OpEnv::new();
        ops1.insert("foo".into(), make_def("foo", 10, 14, true));
        ops2.insert("foo".into(), make_def("foo", 10, 14, true));
        let arc1 = Arc::new(ops1);
        let arc2 = Arc::new(ops2);

        // local=true but is_recursive=false → content hash (same content = same id).
        assert_eq!(
            compute_local_ops_scope_id(&arc1),
            compute_local_ops_scope_id(&arc2)
        );
    }

    #[test]
    fn test_resolve_valid_id() {
        assert_eq!(resolve_local_ops_id(42, &None), 42);
    }

    #[test]
    fn test_resolve_invalidated_id() {
        assert_eq!(resolve_local_ops_id(INVALIDATED, &None), 0);
    }

    #[test]
    fn test_resolve_valid_recursive_scope_ignores_stale_stored_id() {
        let mut ops = OpEnv::new();
        ops.insert("foo".into(), make_recursive_def("foo", 10, 14));
        let ops = Arc::new(ops);
        let expected = compute_local_ops_scope_id(&ops);

        assert_eq!(
            resolve_local_ops_id(42, &Some(ops)),
            expected,
            "recursive local_ops must resolve from the current Arc even when a stored id is present"
        );
    }

    /// The cached-flag fast path must match the (walking) `resolve_local_ops_id`
    /// for the values produced by scope construction — both recursive and
    /// non-recursive — so the perf optimization is value-identical to the oracle.
    #[test]
    fn test_with_recursive_matches_walking_resolver() {
        // Non-recursive scope: stored content id, flag=false.
        let mut nr = OpEnv::new();
        nr.insert("foo".into(), make_def("foo", 10, 14, false));
        let nr = Arc::new(nr);
        let (nr_id, nr_rec) = compute_local_ops_scope_id_and_recursive(&nr);
        assert!(!nr_rec);
        let nr_opt = Some(nr);
        assert_eq!(
            resolve_local_ops_id_with_recursive(nr_id, &nr_opt, nr_rec),
            resolve_local_ops_id(nr_id, &nr_opt),
            "non-recursive cached-flag path must match the walking resolver",
        );

        // Recursive scope: stored Arc-ptr id, flag=true.
        let mut rec = OpEnv::new();
        rec.insert("bar".into(), make_recursive_def("bar", 20, 24));
        let rec = Arc::new(rec);
        let (rec_id, rec_rec) = compute_local_ops_scope_id_and_recursive(&rec);
        assert!(rec_rec);
        let rec_opt = Some(rec);
        assert_eq!(
            resolve_local_ops_id_with_recursive(rec_id, &rec_opt, rec_rec),
            resolve_local_ops_id(rec_id, &rec_opt),
            "recursive cached-flag path must match the walking resolver",
        );
    }

    /// Distinct recursion frames (different Arc allocations, identical content)
    /// must never alias in the cache keys, regardless of the cached flag value.
    /// This is the soundness property: a stale flag may cost a cache miss but can
    /// never produce an incorrect cache hit.
    #[test]
    fn test_with_recursive_no_aliasing_across_frames_any_flag() {
        let mut a = OpEnv::new();
        a.insert("rec".into(), make_recursive_def("rec", 10, 14));
        let a = Arc::new(a);
        let mut b = OpEnv::new();
        b.insert("rec".into(), make_recursive_def("rec", 10, 14));
        let b = Arc::new(b);

        let (a_id, _) = compute_local_ops_scope_id_and_recursive(&a);
        let (b_id, _) = compute_local_ops_scope_id_and_recursive(&b);
        let a_opt = Some(a);
        let b_opt = Some(b);

        // Even if the cached flag were stale in either direction, distinct frames
        // resolve to distinct ids (no incorrect sharing).
        for &flag in &[true, false] {
            let ra = resolve_local_ops_id_with_recursive(a_id, &a_opt, flag);
            let rb = resolve_local_ops_id_with_recursive(b_id, &b_opt, flag);
            assert_ne!(
                ra, rb,
                "distinct recursion frames must get distinct scope ids (flag={flag})",
            );
        }
    }

    #[test]
    fn test_resolve_invalidated_recursive_scope_preserves_arc_identity() {
        let mut ops1 = OpEnv::new();
        let mut ops2 = OpEnv::new();
        ops1.insert("foo".into(), make_recursive_def("foo", 10, 14));
        ops2.insert("foo".into(), make_recursive_def("foo", 10, 14));

        assert_ne!(
            resolve_local_ops_id(INVALIDATED, &Some(Arc::new(ops1))),
            resolve_local_ops_id(INVALIDATED, &Some(Arc::new(ops2)))
        );
    }

    /// Part of #3156: Regression test proving that recursive LET scopes
    /// without module-level LOCAL still use Arc identity. Previously,
    /// the predicate checked `def.local` (LOCAL keyword) instead of
    /// `def.is_recursive`, so non-LOCAL recursive LET operators would
    /// incorrectly take the content-hash path.
    #[test]
    fn test_3156_recursive_let_without_local_uses_arc_identity() {
        let mut ops1 = OpEnv::new();
        let mut ops2 = OpEnv::new();
        // is_recursive=true, local=false — exactly the case #3156 describes.
        let def1 = make_recursive_def("SA", 100, 200);
        let def2 = make_recursive_def("SA", 100, 200);
        assert!(
            !def1.local,
            "regression: recursive LET ops must not need LOCAL"
        );
        assert!(def1.is_recursive, "regression: must be recursive");
        ops1.insert("SA".into(), def1);
        ops2.insert("SA".into(), def2);
        let arc1 = Arc::new(ops1);
        let arc2 = Arc::new(ops2);

        // These two scopes have identical content but represent different
        // recursion levels with different captured variable bindings.
        // They MUST get different scope ids (Arc identity).
        assert_ne!(
            compute_local_ops_scope_id(&arc1),
            compute_local_ops_scope_id(&arc2),
            "recursive LET scopes without LOCAL must use Arc identity"
        );
        assert!(local_ops_requires_arc_identity(&arc1));
    }

    /// Part of #3156: Non-recursive, non-local operators still use content hashing.
    #[test]
    fn test_3156_nonrecursive_nonlocal_uses_content_hash() {
        let mut ops1 = OpEnv::new();
        let mut ops2 = OpEnv::new();
        ops1.insert("bar".into(), make_def("bar", 10, 14, false));
        ops2.insert("bar".into(), make_def("bar", 10, 14, false));
        let arc1 = Arc::new(ops1);
        let arc2 = Arc::new(ops2);

        // Same content → same scope id (content hash, not Arc identity).
        assert_eq!(
            compute_local_ops_scope_id(&arc1),
            compute_local_ops_scope_id(&arc2),
            "non-recursive non-local ops must use content hash"
        );
        assert!(!local_ops_requires_arc_identity(&arc1));
    }

    /// Part of #3099 Step 5: SUBST_CACHE lookup succeeds when the entry was stored
    /// with an eager scope id and the lookup key uses the INVALIDATED -> resolve path.
    #[test]
    fn test_3099_subst_cache_hit_across_eager_and_invalidated_local_ops_id() {
        use crate::cache::dep_tracking::OpEvalDeps;
        use crate::cache::subst_cache::{SubstCacheEntry, SubstCacheKey, SUBST_STATE};
        use tla_core::name_intern::intern_name;
        use tla_value::Value;

        crate::cache::lifecycle::clear_for_test_reset();

        let def = make_def("foo", 10, 14, false);
        let mut ops_a = OpEnv::new();
        ops_a.insert("foo".into(), Arc::clone(&def));
        let eager_id = compute_local_ops_scope_id(&Arc::new(ops_a));

        let name_id = intern_name("test_subst");
        let key_insert = SubstCacheKey {
            is_next_state: false,
            name_id,
            shared_id: 1,
            local_ops_id: eager_id,
            instance_subs_id: 0,
            chained_ref_eval: false,
        };
        SUBST_STATE.with(|s| {
            s.borrow_mut().cache.insert(
                key_insert,
                SubstCacheEntry {
                    value: Value::int(42),
                    deps: OpEvalDeps::default(),
                },
            );
        });

        // INVALIDATED -> resolve path with identical content, different Arc.
        let mut ops_b = OpEnv::new();
        ops_b.insert("foo".into(), Arc::clone(&def));
        let resolved_id = resolve_local_ops_id(INVALIDATED, &Some(Arc::new(ops_b)));
        assert_eq!(eager_id, resolved_id, "same content must produce same id");

        let key_lookup = SubstCacheKey {
            is_next_state: false,
            name_id,
            shared_id: 1,
            local_ops_id: resolved_id,
            instance_subs_id: 0,
            chained_ref_eval: false,
        };
        let found =
            SUBST_STATE.with(|s| s.borrow().cache.get(&key_lookup).map(|e| e.value.clone()));
        assert_eq!(found, Some(Value::int(42)), "SUBST_CACHE must hit");

        crate::cache::lifecycle::clear_for_test_reset();
    }

    /// Part of #3406: Int(1) vs Int(2) with dummy spans must produce different scope ids.
    #[test]
    fn test_3406_distinct_int_subs_with_dummy_span_get_distinct_ids() {
        use tla_core::ast::Expr;

        let sub1 = Substitution {
            from: Spanned::dummy("outer".into()),
            to: Spanned::dummy(Expr::Int(1.into())),
        };
        let sub2 = Substitution {
            from: Spanned::dummy("outer".into()),
            to: Spanned::dummy(Expr::Int(2.into())),
        };
        assert_ne!(
            compute_instance_subs_id(&[sub1]),
            compute_instance_subs_id(&[sub2]),
            "Int(1) and Int(2) with dummy spans must produce different scope ids"
        );
    }

    /// Part of #3406: Ident(channelA) vs Ident(channelB) with dummy spans must differ.
    #[test]
    fn test_3406_distinct_ident_subs_with_dummy_span_get_distinct_ids() {
        use tla_core::ast::Expr;
        use tla_core::name_intern::intern_name;

        let id_a = intern_name("channelA");
        let id_b = intern_name("channelB");
        let sub1 = Substitution {
            from: Spanned::dummy("chan".into()),
            to: Spanned::dummy(Expr::Ident("channelA".into(), id_a)),
        };
        let sub2 = Substitution {
            from: Spanned::dummy("chan".into()),
            to: Spanned::dummy(Expr::Ident("channelB".into(), id_b)),
        };
        assert_ne!(
            compute_instance_subs_id(&[sub1]),
            compute_instance_subs_id(&[sub2]),
            "Ident(channelA) and Ident(channelB) with dummy spans must differ"
        );
    }
}
