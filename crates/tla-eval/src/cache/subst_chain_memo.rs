// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Memoization of INSTANCE-substitution binding chains.
//!
//! Profiling MCNanoMedium (post lazy-union fix) attributed the dominant
//! residual to `build_lazy_subst_bindings_with_local_ops`: every `M!Op`
//! evaluation (and every `SubstIn` entry) rebuilds the lazy-substitution
//! `BindingChain` — one shared heap `LazyBinding` allocation plus 2 Arc-node
//! conses per substitution — even though the substitution list and definition-site
//! `local_ops` are identical on virtually every call (MCNanoSmall: 26,569
//! builds of 10 bindings each over ONE distinct substitution input). This is
//! the same redundancy class as the merged-LET memo in [`super::openv_memo`].
//!
//! # The def-site chain problem, and inertness
//!
//! The third input, the definition-site binding chain (`ctx.bindings` at the
//! `M!Op` call), is NOT identity-stable: successor generation conses fresh
//! arena quantifier bindings per iteration, so raw identity keying never hits
//! (measured: 26,560/26,569 calls carried arena nodes, all distinct). The
//! chain is captured as every `LazyBinding`'s `enclosing` tail, so sharing
//! requires proving the tail unobservable.
//!
//! A def-site chain is **inert** for a given `(subs, local_ops)` when no name
//! it binds can ever be *looked up* while (a) forcing any substitution RHS in
//! its captured enclosing scope, or (b) evaluating instanced operator bodies
//! whose name lookups fall through the substitution nodes into the captured
//! tail. Every such lookup is by a name that is:
//!
//! - a scope-aware free identifier of some substitution RHS (`extra`), or
//! - a scope-aware free identifier of some operator body reachable at
//!   evaluation time — all module-level and instanced-module operator tables
//!   plus the key's own `local_ops` (LET operators) bodies, or
//! - an `OpRef` operator name (collected unscoped — over-approximation), or
//! - a state variable / implicit-substitution target name, or
//! - a `local_ops` key or substitution from-name (shadowing parity: a
//!   chain-bound name could otherwise shadow-resolve differently against an
//!   empty tail).
//!
//! The union of the run-stable parts is the **readable-name universe**
//! (cached per `SharedCtx` id); the `(subs, local_ops)`-specific parts are
//! cached per memo entry (`extra`). If NO chain-bound name is in
//! `universe ∪ extra`, then no lookup during forcing or fall-through can hit
//! the chain: every lookup either resolves in the substitution nodes / local
//! conses above the tail, or misses the tail entirely (falling through to
//! state-var/op/constant resolution) — identically for the real tail and for
//! an EMPTY tail. Chains built over an empty tail are therefore semantically
//! interchangeable across all inert call sites, so ONE canonical chain per
//! `(subs, local_ops)` is shared by all of them. Non-inert calls (e.g.
//! nested-INSTANCE scopes whose ambient chain binds substituted names) fall
//! back to the raw per-call build — byte-for-byte the previous behavior.
//!
//! Names that appear free nowhere are never looked up, so the universe
//! over-approximates all lookup-reachable names. Every step of the
//! approximation only ADDS names (unscoped `OpRef`s, whole-table walks), i.e.
//! only causes needless raw builds, never false sharing.
//!
//! # Why sharing the `LazyBinding` allocations across states is sound
//!
//! A memoized chain returns the *same* `LazyBinding` allocations on every
//! hit, across states. This is sound because a `LazyBinding` no longer holds
//! any state-dependent mutable payload on the INSTANCE path:
//!
//! - Forced **values** live in the per-state `instance_lazy_cache` (keyed by
//!   `LazyBinding` address), cleared at every state/scope boundary (#3465,
//!   #4170). Pinned, memo-stable addresses make that pointer key *stronger*
//!   than today's (no allocator ABA between sibling scopes).
//! - Forced **deps** (which embed read *values* — state-dependent!) used to
//!   live in a per-`LazyBinding` `OnceLock` that would have leaked
//!   first-state deps into later states once bindings are shared. They now
//!   ride in the same `instance_lazy_cache` entry as the value, with the
//!   same epoch lifecycle. The `OnceLock`s (`cached`/`cached_primed`/
//!   `forced_deps*`) are never read or written on the INSTANCE path anymore,
//!   so shared bindings are immutable in effect.
//! - The chain nodes' `BindingSource::Instance(OpEvalDeps::default())`
//!   metadata is constant, and the canonical entry is built over an empty
//!   tail, so its content is a pure function of the memo key — independent
//!   of which call populates it (deterministic).
//!
//! # Pinning
//!
//! Entries pin the keyed `Arc<Vec<Substitution>>` (also keeping every
//! `LazyBinding::expr_ptr` target alive) and the keyed `Arc<OpEnv>`. Pinned
//! refcounts >= 2 foreclose both allocator reuse of the keyed addresses (no
//! ABA) and any `Arc::get_mut`-style in-place mutation (copy-on-write goes to
//! a new allocation), exactly as in `openv_memo`.
//!
//! # Lifecycle
//!
//! Thread-local, per-run: cleared in `clear_run_reset_impl` (alongside
//! `clear_openv_memos`). Clearing at ANY point is sound — later calls simply
//! rebuild equivalent chains. Bounded + clear-on-overflow.
//!
//! Set `TY_NO_SUBST_MEMO=1` to disable (bypass to the original build path).
//! Set `TY_SUBST_STATS=1` to print call/hit/build counters at end of run.

use crate::binding_chain::BindingChain;
use crate::core::{EvalCtx, OpEnv};
use rustc_hash::{FxHashMap, FxHashSet};
use std::cell::RefCell;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use tla_core::ast::{Expr, OperatorDef, Substitution};
use tla_core::name_intern::{intern_name, NameId};
use tla_core::visit::ExprVisitor;

// ===========================================================================
// Env-gated switches
// ===========================================================================

/// `TY_NO_SUBST_MEMO=1` bypasses the memo (original per-call build behavior).
#[inline]
pub(crate) fn subst_memo_disabled() -> bool {
    use std::sync::OnceLock;
    static FLAG: OnceLock<bool> = OnceLock::new();
    *FLAG.get_or_init(|| std::env::var_os("TY_NO_SUBST_MEMO").is_some())
}

/// `TY_SUBST_STATS=1` enables counters + end-of-run stats print.
#[inline]
pub(crate) fn subst_stats_enabled() -> bool {
    use std::sync::OnceLock;
    static FLAG: OnceLock<bool> = OnceLock::new();
    *FLAG.get_or_init(|| std::env::var_os("TY_SUBST_STATS").is_some())
}

// ===========================================================================
// Counters (process-global; summed across worker threads)
// ===========================================================================

/// Memoized-entry-point calls (the redundancy denominator).
static CHAIN_CALLS: AtomicU64 = AtomicU64::new(0);
/// Canonical-entry hits (chain reused).
static CHAIN_HITS: AtomicU64 = AtomicU64::new(0);
/// Canonical-entry first builds (one per distinct key).
static CHAIN_CANON_BUILDS: AtomicU64 = AtomicU64::new(0);
/// Raw builds because the def-site chain was not inert.
static CHAIN_NONINERT_BUILDS: AtomicU64 = AtomicU64::new(0);
/// Raw builds because the def-site chain exceeded the inertness walk cap.
static CHAIN_TOOLONG_BUILDS: AtomicU64 = AtomicU64::new(0);
/// Builds under `TY_NO_SUBST_MEMO=1`.
static CHAIN_DISABLED_BUILDS: AtomicU64 = AtomicU64::new(0);
/// Memo entry inserts (each computes the `extra` readable-name set).
static ENTRY_INSERTS: AtomicU64 = AtomicU64::new(0);
/// Substitution bindings materialized by all builds (one shared LazyBinding
/// allocation and two chain nodes each).
static CHAIN_BINDINGS_BUILT: AtomicU64 = AtomicU64::new(0);
/// Raw (unmemoized-entry-point) builds, e.g. test-only call sites.
static RAW_BUILDS: AtomicU64 = AtomicU64::new(0);

/// Total nanoseconds spent inside `subst_chain_memoized` (stats only).
static CHAIN_NANOS: AtomicU64 = AtomicU64::new(0);
/// Total nanoseconds spent inside raw builds routed through the memo (subset
/// of `CHAIN_NANOS`; stats only).
static CHAIN_BUILD_NANOS: AtomicU64 = AtomicU64::new(0);
/// Phase nanos: key derivation (ptr-cache miss → scope id + content hash).
static PHASE_KEY_NANOS: AtomicU64 = AtomicU64::new(0);
/// Phase nanos: entry ensure + verification (+ ptr-cache certify).
static PHASE_ENTRY_NANOS: AtomicU64 = AtomicU64::new(0);
/// Phase nanos: inertness walk.
static PHASE_INERT_NANOS: AtomicU64 = AtomicU64::new(0);

/// `compute_effective_instance_substitutions` calls.
static CES_CALLS: AtomicU64 = AtomicU64::new(0);
/// Distinct (module, explicit-subs shape) keys seen by CES.
static CES_DISTINCT: AtomicU64 = AtomicU64::new(0);

/// Run-lifetime scope-entry memo hits (per-state cache missed, run memo hit —
/// CES + instance-ops merge skipped). See `helpers::module_ref_cache`.
static SCOPE_MEMO_HITS: AtomicU64 = AtomicU64::new(0);
/// Run-lifetime scope-entry memo builds (both caches missed).
static SCOPE_MEMO_BUILDS: AtomicU64 = AtomicU64::new(0);
/// Bails to the unmemoized build (pointer-keyed recursive ambient scope, or
/// `TY_LEGACY_EPOCH_CLEAR=1`).
static SCOPE_MEMO_BAILS: AtomicU64 = AtomicU64::new(0);

/// Count a run scope-entry memo hit. Stats only.
#[inline]
pub(crate) fn count_scope_memo_hit() {
    if subst_stats_enabled() {
        SCOPE_MEMO_HITS.fetch_add(1, Ordering::Relaxed);
    }
}

/// Count a run scope-entry memo build (miss). Stats only.
#[inline]
pub(crate) fn count_scope_memo_build() {
    if subst_stats_enabled() {
        SCOPE_MEMO_BUILDS.fetch_add(1, Ordering::Relaxed);
    }
}

/// Count a run scope-entry memo bail. Stats only.
#[inline]
pub(crate) fn count_scope_memo_bail() {
    if subst_stats_enabled() {
        SCOPE_MEMO_BAILS.fetch_add(1, Ordering::Relaxed);
    }
}

/// INSTANCE lazy-binding forcings (substitution RHS evaluations). The #4170
/// per-prepare `instance_lazy_cache` clear makes every `M!Op` call re-force
/// all RHS reads within a state; this counter is the re-force denominator.
static INSTANCE_FORCES: AtomicU64 = AtomicU64::new(0);
/// INSTANCE lazy-binding cache hits (fused value+deps probe).
static INSTANCE_FORCE_HITS: AtomicU64 = AtomicU64::new(0);

/// Count an INSTANCE lazy-binding forcing. Stats only.
#[inline]
pub(crate) fn count_instance_force() {
    if subst_stats_enabled() {
        INSTANCE_FORCES.fetch_add(1, Ordering::Relaxed);
    }
}

/// Count an INSTANCE lazy-binding cache hit. Stats only.
#[inline]
pub(crate) fn count_instance_force_hit() {
    if subst_stats_enabled() {
        INSTANCE_FORCE_HITS.fetch_add(1, Ordering::Relaxed);
    }
}

/// Count a raw (non-memoized entry point) chain build. Stats only.
/// (The raw wrapper is test-only in production builds — see helpers/mod.rs.)
#[inline]
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn count_raw_build(n_subs: usize) {
    if subst_stats_enabled() {
        RAW_BUILDS.fetch_add(1, Ordering::Relaxed);
        CHAIN_BINDINGS_BUILT.fetch_add(n_subs as u64, Ordering::Relaxed);
    }
}

/// Count a `compute_effective_instance_substitutions` call. Stats only.
/// `key_hash` fingerprints (module name, explicit subs shape) so the printed
/// stats expose the redundancy factor (calls per distinct input).
pub(crate) fn count_ces_call(key_hash: u64) {
    if !subst_stats_enabled() {
        return;
    }
    CES_CALLS.fetch_add(1, Ordering::Relaxed);
    thread_local! {
        static CES_KEYS: RefCell<FxHashSet<u64>> = RefCell::new(FxHashSet::default());
    }
    CES_KEYS.with(|keys| {
        if keys.borrow_mut().insert(key_hash) {
            CES_DISTINCT.fetch_add(1, Ordering::Relaxed);
        }
    });
}

/// Print `TY_SUBST_STATS=1` counters (no-op otherwise). Called from
/// `print_eval_profile_stats` at end of model checking.
pub fn print_subst_memo_stats() {
    if !subst_stats_enabled() {
        return;
    }
    eprintln!("\n=== INSTANCE-subst memo stats (TY_SUBST_STATS) ===");
    if subst_memo_disabled() {
        eprintln!("  [memo DISABLED via TY_NO_SUBST_MEMO — every call is a build]");
    }
    let calls = CHAIN_CALLS.load(Ordering::Relaxed);
    let hits = CHAIN_HITS.load(Ordering::Relaxed);
    eprintln!(
        "  subst-chain builds: calls {calls}  hits {hits}  canon-builds {}  noninert-builds {}  toolong-builds {}  disabled-builds {}",
        CHAIN_CANON_BUILDS.load(Ordering::Relaxed),
        CHAIN_NONINERT_BUILDS.load(Ordering::Relaxed),
        CHAIN_TOOLONG_BUILDS.load(Ordering::Relaxed),
        CHAIN_DISABLED_BUILDS.load(Ordering::Relaxed),
    );
    eprintln!(
        "    bindings materialized {}  raw-path builds {}",
        CHAIN_BINDINGS_BUILT.load(Ordering::Relaxed),
        RAW_BUILDS.load(Ordering::Relaxed),
    );
    eprintln!(
        "    device time {:.3}ms (of which raw/canon builds {:.3}ms)",
        CHAIN_NANOS.load(Ordering::Relaxed) as f64 / 1e6,
        CHAIN_BUILD_NANOS.load(Ordering::Relaxed) as f64 / 1e6,
    );
    eprintln!(
        "    phases: key {:.3}ms  entry+verify {:.3}ms  inert-walk {:.3}ms  entry-inserts {}  live-entries {}",
        PHASE_KEY_NANOS.load(Ordering::Relaxed) as f64 / 1e6,
        PHASE_ENTRY_NANOS.load(Ordering::Relaxed) as f64 / 1e6,
        PHASE_INERT_NANOS.load(Ordering::Relaxed) as f64 / 1e6,
        ENTRY_INSERTS.load(Ordering::Relaxed),
        SUBST_CHAIN_MEMO.with(|m| m.borrow().len()),
    );
    eprintln!(
        "  compute_effective_instance_substitutions: calls {}  distinct-inputs {}",
        CES_CALLS.load(Ordering::Relaxed),
        CES_DISTINCT.load(Ordering::Relaxed),
    );
    eprintln!(
        "  run scope-entry memo: hits {}  builds {}  bails {}",
        SCOPE_MEMO_HITS.load(Ordering::Relaxed),
        SCOPE_MEMO_BUILDS.load(Ordering::Relaxed),
        SCOPE_MEMO_BAILS.load(Ordering::Relaxed),
    );
    eprintln!(
        "  instance lazy bindings: forces {}  cache-hits {}",
        INSTANCE_FORCES.load(Ordering::Relaxed),
        INSTANCE_FORCE_HITS.load(Ordering::Relaxed),
    );
}

// ===========================================================================
// Readable-name collection
// ===========================================================================

/// Collect the lookup-reachable names of one expression into `out`:
/// scope-aware free identifiers (via `tla_core::free_vars`, which covers
/// `Ident`, `StateVar`, and descends into `InstanceExpr`/`SubstIn`
/// substitution RHS and `ModuleRef` arguments) plus `OpRef` operator names
/// (collected unscoped — a sound over-approximation; TLA+ cannot bind
/// operator names as values, and op params are subtracted by callers).
fn collect_readable_names(expr: &Expr, out: &mut FxHashSet<NameId>) {
    for name in tla_core::free_vars(expr) {
        out.insert(intern_name(&name));
    }
    struct OpRefCollector<'a> {
        out: &'a mut FxHashSet<NameId>,
    }
    impl ExprVisitor for OpRefCollector<'_> {
        type Output = ();
        fn visit_node(&mut self, expr: &Expr) -> Option<()> {
            if let Expr::OpRef(name) = expr {
                self.out.insert(intern_name(name));
            }
            None
        }
    }
    OpRefCollector { out }.walk_expr(expr);
}

/// Collect an operator body's lookup-reachable names minus its parameters.
fn collect_def_body_names(def: &OperatorDef, out: &mut FxHashSet<NameId>) {
    let mut body_names = FxHashSet::default();
    collect_readable_names(&def.body.node, &mut body_names);
    for param in &def.params {
        body_names.remove(&intern_name(param.name.node.as_str()));
    }
    out.extend(body_names);
}

thread_local! {
    /// Readable-name universe per `SharedCtx` id. The universe is a pure
    /// function of the run-stable shared tables, so entries never go stale;
    /// keying by shared id isolates concurrent independent runs in tests.
    static READABLE_UNIVERSE: RefCell<FxHashMap<u64, Arc<FxHashSet<NameId>>>> =
        RefCell::new(FxHashMap::default());
}

/// The run-stable readable-name universe for this evaluation context: free
/// names of every global and instanced operator body (minus params), all
/// implicit-substitution target names, and all state variable names.
fn readable_universe(ctx: &EvalCtx) -> Arc<FxHashSet<NameId>> {
    let shared_id = ctx.shared.id;
    if let Some(u) = READABLE_UNIVERSE.with(|m| m.borrow().get(&shared_id).cloned()) {
        return u;
    }
    let mut set = FxHashSet::default();
    for (_, def) in ctx.shared.ops.iter() {
        collect_def_body_names(def, &mut set);
    }
    for env in ctx.shared.instance_ops.values() {
        for (_, def) in env.iter() {
            collect_def_body_names(def, &mut set);
        }
    }
    for targets in ctx.shared.instance_implicit_targets.values() {
        for target in targets {
            set.insert(intern_name(target));
        }
    }
    for name in ctx.shared.var_registry.names() {
        set.insert(intern_name(name));
    }
    let set = Arc::new(set);
    READABLE_UNIVERSE.with(|m| {
        m.borrow_mut().insert(shared_id, Arc::clone(&set));
    });
    set
}

// ===========================================================================
// The canonical-chain memo
// ===========================================================================

/// Producer-identity memo key: (substitution site, ambient scope content).
///
/// Pointer identity is useless here: the module-ref scope caches that own the
/// `Arc<Vec<Substitution>>` / `Arc<OpEnv>` are cleared at every state (and
/// even per successor candidate — the #3447 defense), so a fresh
/// content-identical `Arc` arrives on almost every call. Content hashing +
/// structural verification was tried and measured: exact verification of the
/// substitution RHS trees once per state per site cost as much as the builds
/// it saved. The key instead identifies the PRODUCER of the inputs:
///
/// - `site_kind`/`site_id`: which INSTANCE machinery produced the
///   substitution list — a named-instance module-ref scope (interned instance
///   name), a chained module-ref (compound chain key hash), or a `SubstIn`
///   node (AST address, run-stable). Given the site, the effective
///   substitution list and merged def-site `local_ops` are DETERMINISTIC
///   functions of the run-stable shared tables/config and the ambient scope
///   content (`compute_effective_instance_substitutions` and the scope-entry
///   builders take no other inputs).
/// - `inst_subs_id`/`local_ops_id`: the ambient scope content fingerprints
///   (#3099 `EvalScopeIds`, via `resolve_*` — the EXACT ids SUBST_CACHE
///   already keys evaluated substitution values by, so this introduces no new
///   trust assumption). Recursive `local_ops` scopes resolve to the Arc
///   pointer; entries pin every Arc they key, so pointer components can never
///   be recycled while mapped (no ABA — same argument as `openv_memo`).
///
/// Debug builds additionally assert full structural equality of the incoming
/// substitution list against the pinned one on every hit, so any violation of
/// the producer-determinism contract fails loudly across the test suite;
/// release hits keep an O(1) length sanity check that falls back to the raw
/// build.
#[derive(PartialEq, Eq, Hash, Clone, Copy)]
struct CanonKey {
    shared_id: u64,
    site_kind: u8,
    site_id: u64,
    inst_subs_id: u64,
    local_ops_id: u64,
}

/// Site families for [`CanonKey::site_kind`].
pub(crate) const SITE_NAMED_MODULE_REF: u8 = 0;
pub(crate) const SITE_RESOLVED_MODULE_REF: u8 = 1;
pub(crate) const SITE_CHAINED_MODULE_REF: u8 = 2;
pub(crate) const SITE_SUBST_IN: u8 = 3;

struct CanonEntry {
    /// Key-specific readable names: free names of this key's substitution
    /// RHS, the substitution from-names, the `local_ops` operator names, and
    /// the `local_ops` bodies' free names (minus params).
    extra: FxHashSet<NameId>,
    /// The canonical chain (built over an EMPTY tail), shared by every inert
    /// call with this key. `None` until the first inert call builds it.
    chain: Option<BindingChain>,
    /// The substitution vector the chain's `expr_ptr`s point into. Pin AND
    /// hit-verification oracle (full structural equality on every hit).
    subs: Arc<Vec<Substitution>>,
    /// The keyed `local_ops` env. Pin AND hit-verification (length check;
    /// content equality is carried by `local_ops_id`, the same content
    /// fingerprint the rest of the cache stack keys scopes by).
    local_ops: Option<Arc<OpEnv>>,
}

thread_local! {
    static SUBST_CHAIN_MEMO: RefCell<FxHashMap<CanonKey, CanonEntry>> =
        RefCell::new(FxHashMap::default());
}

/// Cap on memo entries. Keys are O(INSTANCE sites x local-op scopes) in
/// practice (both Arcs come from run-stable scope caches), so this is
/// effectively never hit; it bounds memory for pathological workloads.
/// Overflow clears the map — sound at any point (entries rebuild fresh).
const SUBST_CHAIN_MEMO_CAP: usize = 8_192;

/// Cap on the number of def-site chain nodes walked by the inertness test.
/// Longer chains are conservatively treated as non-inert (raw build), keeping
/// the per-call overhead bounded.
const INERT_WALK_CAP: usize = 64;

/// Build (or reuse) the lazy INSTANCE-substitution chain for
/// `(def_site_chain, def_site_local_ops, subs_arc)`.
///
/// If the def-site chain is inert for this key (see module docs), returns the
/// shared canonical chain (same `LazyBinding` allocations every call).
/// Otherwise (or with `TY_NO_SUBST_MEMO=1`) defers to
/// `build_lazy_subst_bindings_with_local_ops`, byte-for-byte identical to the
/// previous behavior.
pub(crate) fn subst_chain_memoized(
    ctx: &EvalCtx,
    def_site_chain: &BindingChain,
    def_site_local_ops: Option<Arc<OpEnv>>,
    subs_arc: &Arc<Vec<Substitution>>,
    site_kind: u8,
    site_id: u64,
) -> BindingChain {
    let stats = subst_stats_enabled();
    let t0 = if stats {
        Some(std::time::Instant::now())
    } else {
        None
    };
    let finish = |t0: Option<std::time::Instant>, built: bool, chain: BindingChain| {
        if let Some(t0) = t0 {
            let ns = t0.elapsed().as_nanos() as u64;
            CHAIN_NANOS.fetch_add(ns, Ordering::Relaxed);
            if built {
                CHAIN_BUILD_NANOS.fetch_add(ns, Ordering::Relaxed);
            }
        }
        chain
    };
    if stats {
        CHAIN_CALLS.fetch_add(1, Ordering::Relaxed);
        CHAIN_BINDINGS_BUILT.fetch_add(subs_arc.len() as u64, Ordering::Relaxed);
    }

    if subst_memo_disabled() {
        if stats {
            CHAIN_DISABLED_BUILDS.fetch_add(1, Ordering::Relaxed);
        }
        let chain = crate::helpers::build_lazy_subst_bindings_with_local_ops(
            def_site_chain,
            def_site_local_ops,
            subs_arc,
        );
        return finish(t0, true, chain);
    }

    // Producer-identity key: (site, ambient scope content ids). Both ids are
    // O(1) reads of the maintained #3099 EvalScopeIds (with the same
    // INVALIDATED-recompute fallback the SUBST_CACHE key path uses).
    let tk = t0.map(|_| std::time::Instant::now());
    let local_ops_id = super::scope_ids::resolve_local_ops_id_with_recursive(
        ctx.scope_ids.local_ops,
        &ctx.local_ops,
        ctx.scope_ids.local_ops_recursive,
    );
    // Recursive ambient scopes resolve to the Arc POINTER (per-recursion-frame
    // identity, #3156) — historically a key component that churned every state
    // (the module-ref scope caches were rebuilt per state), so memoizing
    // inserted a fresh entry (with its expensive readable-name set) per state
    // for zero reuse (measured: ~1 insert/state, ~200ms on MCNanoSmall).
    //
    // #3447/#4170 epoch policy: the run-lifetime scope memo
    // (`helpers::module_ref_cache`) now PINS the merged local_ops Arcs it
    // hands out, making those addresses run-stable (no allocator ABA while
    // pinned, immutable content, one address per scope for the run). A pinned
    // ambient pointer is therefore a sound, reusable key component — accept
    // it. Unpinned pointers keep the bail (fresh-entry churn + ABA risk).
    let ambient_ptr_keyed = ctx
        .local_ops
        .as_ref()
        .is_some_and(|o| local_ops_id == Arc::as_ptr(o) as usize as u64)
        && !crate::helpers::module_ref_cache::is_pinned_local_ops(local_ops_id as usize);
    if ambient_ptr_keyed {
        if stats {
            CHAIN_NONINERT_BUILDS.fetch_add(1, Ordering::Relaxed);
            if let Some(tk) = tk {
                PHASE_KEY_NANOS.fetch_add(tk.elapsed().as_nanos() as u64, Ordering::Relaxed);
            }
        }
        let chain = crate::helpers::build_lazy_subst_bindings_with_local_ops(
            def_site_chain,
            def_site_local_ops,
            subs_arc,
        );
        return finish(t0, true, chain);
    }
    let key = CanonKey {
        shared_id: ctx.shared.id,
        site_kind,
        site_id,
        inst_subs_id: super::scope_ids::resolve_instance_subs_id(
            ctx.scope_ids.instance_substitutions,
            &ctx.instance_substitutions,
        ),
        local_ops_id,
    };
    if let Some(tk) = tk {
        PHASE_KEY_NANOS.fetch_add(tk.elapsed().as_nanos() as u64, Ordering::Relaxed);
    }

    let (chain, built) = SUBST_CHAIN_MEMO.with(|memo| {
        let te = t0.map(|_| std::time::Instant::now());
        // Ensure the entry (with its `extra` name set) exists. On a hit, an
        // O(1) sanity check guards the producer-determinism contract (full
        // structural equality is asserted in debug builds); a mismatch falls
        // back to the raw build.
        {
            let mut m = memo.borrow_mut();
            match m.get(&key) {
                Some(entry) => {
                    debug_assert!(
                        *entry.subs == **subs_arc,
                        "subst_chain_memo: producer-determinism contract violated \
                         (same (site, ambient) key, different substitution content)"
                    );
                    let sane = entry.subs.len() == subs_arc.len()
                        && match (&entry.local_ops, &def_site_local_ops) {
                            (None, None) => true,
                            (Some(a), Some(b)) => Arc::ptr_eq(a, b) || a.len() == b.len(),
                            _ => false,
                        };
                    if !sane {
                        drop(m);
                        if stats {
                            CHAIN_NONINERT_BUILDS.fetch_add(1, Ordering::Relaxed);
                        }
                        return (
                            crate::helpers::build_lazy_subst_bindings_with_local_ops(
                                def_site_chain,
                                def_site_local_ops,
                                subs_arc,
                            ),
                            true,
                        );
                    }
                }
                None => {
                    if stats {
                        ENTRY_INSERTS.fetch_add(1, Ordering::Relaxed);
                    }
                    if m.len() >= SUBST_CHAIN_MEMO_CAP {
                        m.clear();
                    }
                    let mut extra = FxHashSet::default();
                    for sub in subs_arc.iter() {
                        extra.insert(intern_name(sub.from.node.as_str()));
                        collect_readable_names(&sub.to.node, &mut extra);
                    }
                    if let Some(local_ops) = def_site_local_ops.as_ref() {
                        for (name, def) in local_ops.iter() {
                            extra.insert(intern_name(name.as_str()));
                            collect_def_body_names(def, &mut extra);
                        }
                    }
                    m.insert(
                        key,
                        CanonEntry {
                            extra,
                            chain: None,
                            subs: Arc::clone(subs_arc),
                            local_ops: def_site_local_ops.clone(),
                        },
                    );
                }
            }
        }

        if let Some(te) = te {
            PHASE_ENTRY_NANOS.fetch_add(te.elapsed().as_nanos() as u64, Ordering::Relaxed);
        }
        let ti = t0.map(|_| std::time::Instant::now());
        // Inertness test: no chain-bound name may be lookup-reachable.
        let universe = readable_universe(ctx);
        let inert = {
            let m = memo.borrow();
            let entry = m.get(&key).expect("entry inserted above");
            let mut walked = 0usize;
            let mut inert = true;
            for (name_id, _, _) in def_site_chain.iter() {
                walked += 1;
                if walked > INERT_WALK_CAP {
                    if stats {
                        CHAIN_TOOLONG_BUILDS.fetch_add(1, Ordering::Relaxed);
                    }
                    inert = false;
                    break;
                }
                if universe.contains(&name_id) || entry.extra.contains(&name_id) {
                    if stats {
                        CHAIN_NONINERT_BUILDS.fetch_add(1, Ordering::Relaxed);
                    }
                    inert = false;
                    break;
                }
            }
            inert
        };

        if let Some(ti) = ti {
            PHASE_INERT_NANOS.fetch_add(ti.elapsed().as_nanos() as u64, Ordering::Relaxed);
        }
        if !inert {
            return (
                crate::helpers::build_lazy_subst_bindings_with_local_ops(
                    def_site_chain,
                    def_site_local_ops,
                    subs_arc,
                ),
                true,
            );
        }

        if let Some(chain) = memo
            .borrow()
            .get(&key)
            .and_then(|entry| entry.chain.clone())
        {
            if stats {
                CHAIN_HITS.fetch_add(1, Ordering::Relaxed);
            }
            return (chain, false);
        }

        // First inert call for this key: build the canonical chain over an
        // EMPTY tail (semantically interchangeable with any inert def-site
        // chain — see module docs) and cache it. The chain's expr_ptrs point
        // into the ENTRY's pinned substitution vector (structurally equal to
        // every later caller's), so it stays valid for the memo's lifetime.
        if stats {
            CHAIN_CANON_BUILDS.fetch_add(1, Ordering::Relaxed);
        }
        let (pinned_subs, pinned_local_ops) = {
            let m = memo.borrow();
            let entry = m.get(&key).expect("entry inserted above");
            (Arc::clone(&entry.subs), entry.local_ops.clone())
        };
        let chain = crate::helpers::build_lazy_subst_bindings_with_local_ops(
            &BindingChain::empty(),
            pinned_local_ops,
            &pinned_subs,
        );
        if let Some(entry) = memo.borrow_mut().get_mut(&key) {
            entry.chain = Some(chain.clone());
        }
        (chain, true)
    });
    finish(t0, built, chain)
}

/// Clear the memo and universe (run/test reset). Dropping entries drops their
/// pins in the same breath, so no entry can ever outlive the allocations it
/// keys on. Clearing is sound at any point: later calls rebuild fresh.
pub fn clear_subst_chain_memo() {
    SUBST_CHAIN_MEMO.with(|memo| memo.borrow_mut().clear());
    READABLE_UNIVERSE.with(|u| u.borrow_mut().clear());
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::binding_chain::BindingValue;
    use tla_core::Spanned;

    fn mk_subs(names: &[&str]) -> Arc<Vec<Substitution>> {
        Arc::new(
            names
                .iter()
                .map(|n| Substitution {
                    from: Spanned::dummy((*n).to_string()),
                    to: Spanned::dummy(Expr::Ident((*n).to_string(), intern_name(n))),
                })
                .collect(),
        )
    }

    fn chain_names(chain: &BindingChain) -> Vec<NameId> {
        chain.iter().map(|(name, _, _)| name).collect()
    }

    fn test_ctx() -> EvalCtx {
        EvalCtx::new()
    }

    #[test]
    fn inert_calls_share_the_canonical_chain() {
        clear_subst_chain_memo();
        let ctx = test_ctx();
        let subs = mk_subs(&["a", "b"]);
        let base = BindingChain::empty();
        // A chain binding a name unrelated to the subs/universe is inert.
        let unrelated = base.cons_local(
            intern_name("zz_unrelated_local"),
            BindingValue::eager(crate::value::Value::Bool(true)),
            0,
        );
        let c1 = subst_chain_memoized(&ctx, &base, None, &subs, SITE_NAMED_MODULE_REF, 1);
        let c2 = subst_chain_memoized(&ctx, &unrelated, None, &subs, SITE_NAMED_MODULE_REF, 1);
        // Structural equality with the raw builder over an empty tail.
        let raw = crate::helpers::build_lazy_subst_bindings_with_local_ops(&base, None, &subs);
        assert_eq!(chain_names(&c1), chain_names(&raw));
        assert_eq!(chain_names(&c1), chain_names(&c2));
        // Both calls share the exact same allocations (canonical entry).
        let (a1, h1) = c1.identity_parts();
        let (a2, h2) = c2.identity_parts();
        assert_eq!(a1, 0, "canonical chain must be all-heap");
        assert_eq!((a1, h1), (a2, h2), "inert calls must share the chain");
        for ((_, v1, _), (_, v2, _)) in c1.iter().zip(c2.iter()) {
            let l1 = v1.as_lazy().expect("lazy binding");
            let l2 = v2.as_lazy().expect("lazy binding");
            assert!(std::ptr::eq(l1, l2), "hit must share LazyBinding allocs");
        }
        clear_subst_chain_memo();
    }

    #[test]
    fn raw_builder_shares_duplicate_instance_lazy_allocation() {
        let subs = mk_subs(&["a", "b"]);
        let chain = crate::helpers::build_lazy_subst_bindings_with_local_ops(
            &BindingChain::empty(),
            None,
            &subs,
        );
        let mut visible = chain.iter();
        let a_lazy = visible.next().unwrap().1.as_lazy().unwrap();
        let b_visible = visible.next().unwrap().1.as_lazy().unwrap();
        let b_enclosing = a_lazy.enclosing.iter().next().unwrap().1.as_lazy().unwrap();
        assert!(
            std::ptr::eq(b_visible, b_enclosing),
            "visible and enclosing positions must share one LazyBinding"
        );
    }

    #[test]
    fn rhs_free_name_bound_in_chain_is_not_inert() {
        clear_subst_chain_memo();
        let ctx = test_ctx();
        // Substitution RHS reads free `x`; def-site chain binds `x`.
        let subs = Arc::new(vec![Substitution {
            from: Spanned::dummy("a".to_string()),
            to: Spanned::dummy(Expr::Ident("x".to_string(), intern_name("x"))),
        }]);
        let base = BindingChain::empty();
        let with_x = base.cons_local(
            intern_name("x"),
            BindingValue::eager(crate::value::Value::Bool(true)),
            0,
        );
        let c1 = subst_chain_memoized(&ctx, &with_x, None, &subs, SITE_NAMED_MODULE_REF, 2);
        let c2 = subst_chain_memoized(&ctx, &with_x, None, &subs, SITE_NAMED_MODULE_REF, 2);
        // Non-inert: fresh raw builds, chains NOT shared.
        let (_, h1) = c1.identity_parts();
        let (_, h2) = c2.identity_parts();
        assert_ne!(h1, h2, "non-inert calls must not share chains");
        // The raw build preserves the def-site tail in the built chain's
        // lazy enclosing (captured), unlike the canonical empty-tail build.
        let l1 = c1.iter().next().unwrap().1.as_lazy().unwrap();
        assert_eq!(
            l1.enclosing.iter().count(),
            1,
            "raw build must capture the def-site tail"
        );
        clear_subst_chain_memo();
    }

    #[test]
    fn from_name_bound_in_chain_is_not_inert() {
        clear_subst_chain_memo();
        let ctx = test_ctx();
        let subs = mk_subs(&["a"]);
        // Chain binds the FROM name `a` (nested-INSTANCE shadowing shape).
        let with_a = BindingChain::empty().cons_local(
            intern_name("a"),
            BindingValue::eager(crate::value::Value::Bool(true)),
            0,
        );
        let c1 = subst_chain_memoized(&ctx, &with_a, None, &subs, SITE_NAMED_MODULE_REF, 3);
        let c2 = subst_chain_memoized(&ctx, &with_a, None, &subs, SITE_NAMED_MODULE_REF, 3);
        let (_, h1) = c1.identity_parts();
        let (_, h2) = c2.identity_parts();
        assert_ne!(h1, h2, "from-name shadowing must fall back to raw builds");
        clear_subst_chain_memo();
    }

    #[test]
    fn same_site_content_identical_subs_share_across_arcs() {
        // The point of producer-identity keying: the module-ref scope caches
        // are cleared per state, so content-identical substitution vectors
        // arrive behind fresh Arcs every state and must still share.
        clear_subst_chain_memo();
        let ctx = test_ctx();
        let base = BindingChain::empty();
        let subs_a = mk_subs(&["a"]);
        let subs_b = mk_subs(&["a"]); // same content, different Arc
        let c1 = subst_chain_memoized(&ctx, &base, None, &subs_a, SITE_NAMED_MODULE_REF, 4);
        let c2 = subst_chain_memoized(&ctx, &base, None, &subs_b, SITE_NAMED_MODULE_REF, 4);
        let (_, h1) = c1.identity_parts();
        let (_, h2) = c2.identity_parts();
        assert_eq!(h1, h2, "content-identical subs at one site must share");
        clear_subst_chain_memo();
    }

    #[test]
    fn distinct_sites_do_not_share() {
        clear_subst_chain_memo();
        let ctx = test_ctx();
        let base = BindingChain::empty();
        let subs_a = mk_subs(&["a"]);
        // A different site with different substitution content must get its
        // own canonical chain (sites are the producer identity).
        let subs_b = Arc::new(vec![Substitution {
            from: Spanned::dummy("a".to_string()),
            to: Spanned::dummy(Expr::Ident("other".to_string(), intern_name("other"))),
        }]);
        let c1 = subst_chain_memoized(&ctx, &base, None, &subs_a, SITE_NAMED_MODULE_REF, 5);
        let c2 = subst_chain_memoized(&ctx, &base, None, &subs_b, SITE_SUBST_IN, 6);
        let (_, h1) = c1.identity_parts();
        let (_, h2) = c2.identity_parts();
        assert_ne!(h1, h2, "distinct sites must not share chains");
        // And the second chain's lazy must point at ITS OWN pinned expr.
        let l2 = c2.iter().next().unwrap().1.as_lazy().unwrap();
        match &l2.expr().node {
            Expr::Ident(name, _) => assert_eq!(name, "other"),
            other => panic!("unexpected RHS: {other:?}"),
        }
        clear_subst_chain_memo();
    }

    #[test]
    fn release_sanity_guard_falls_back_on_len_mismatch() {
        // Violating the producer-determinism contract (same key, different
        // substitution LENGTH) must fall back to a raw build in release; in
        // debug builds the contract is a debug_assert, so exercise the guard
        // only under cfg(not(debug_assertions)).
        if cfg!(debug_assertions) {
            return;
        }
        clear_subst_chain_memo();
        let ctx = test_ctx();
        let base = BindingChain::empty();
        let subs_a = mk_subs(&["a"]);
        let subs_b = mk_subs(&["a", "b"]);
        let c1 = subst_chain_memoized(&ctx, &base, None, &subs_a, SITE_NAMED_MODULE_REF, 7);
        let c2 = subst_chain_memoized(&ctx, &base, None, &subs_b, SITE_NAMED_MODULE_REF, 7);
        assert_eq!(c1.iter().count(), 1);
        assert_eq!(c2.iter().count(), 2, "guard must build the true subs");
        clear_subst_chain_memo();
    }

    #[test]
    fn clear_drops_entries_and_rebuilds_fresh() {
        clear_subst_chain_memo();
        let ctx = test_ctx();
        let base = BindingChain::empty();
        let subs = mk_subs(&["a", "b", "c"]);
        let c1 = subst_chain_memoized(&ctx, &base, None, &subs, SITE_NAMED_MODULE_REF, 8);
        clear_subst_chain_memo();
        let c2 = subst_chain_memoized(&ctx, &base, None, &subs, SITE_NAMED_MODULE_REF, 8);
        assert_eq!(chain_names(&c1), chain_names(&c2));
        let (_, h1) = c1.identity_parts();
        let (_, h2) = c2.identity_parts();
        assert_ne!(h1, h2, "clear must drop entries (fresh build)");
        clear_subst_chain_memo();
    }
}
