// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! TRUE-only ENABLED provenance from BFS successor generation.
//!
//! The sound redo of the disabled #3100 ENABLED-provenance bypass (see the
//! "Bug C" notes in `inline_record.rs`). The BFS `Next` enumeration already
//! visits every (action, quantifier-binding) pair reachable from the current
//! state; when the descent through an operator application `Op(v1, .., vn)`
//! emits at least one successor, that emission *witnesses* the satisfiability
//! of the action relation `Op(v1, .., vn)` from the current state — the exact
//! question `ENABLED` asks. This module records those witnesses during the
//! BFS generation pass and lets the inline fairness recorder skip the
//! from-scratch per-leaf action enumeration for the witnessed leaves.
//!
//! # Soundness (the TRUE-only asymmetry)
//!
//! Only `ENABLED = true` is ever derived from provenance:
//!
//! * **Presence ⇒ true is unconditionally sound.** An emitted successor `t`
//!   satisfies, by construction of the unified enumerator, EVERY constraint on
//!   its descent path — including the full body of each operator frame the
//!   path passed through, with the frame's formal parameters bound to the same
//!   argument VALUES the frame key records. Hence `A(s, t)` holds for each
//!   framed action `A`, so `∃ t': A(s, t')` (ENABLED without a subscript) is
//!   witnessed. For a fairness leaf `ENABLED <<A>>_e` (`require_state_change`),
//!   the witness additionally requires the emission to CHANGE `e`; a leaf is
//!   only registered here when its subscript statically covers ALL state
//!   variables (`subscript_covers_all_vars`), so "some emitted value genuinely
//!   differs from the base state" — an exact per-value comparison, the same
//!   test `diff_is_witness` / `emit_successor`'s fast path use — decides the
//!   subscript change exactly.
//! * **Absence is NEVER interpreted.** A leaf with no recorded witness falls
//!   back to the full evaluator (`eval_enabled_uncached`). Deduplicated
//!   successors are irrelevant (witnesses are recorded at RAW emission time,
//!   before any dedup), constraint-pruned successors are irrelevant (pruning
//!   happens after enumeration; and ty's ENABLED semantics, like TLC's,
//!   ignores constraints anyway), and enumeration caps/short-circuits can only
//!   LOSE witnesses — losing a witness merely costs a fallback evaluation,
//!   never a wrong verdict. This is what makes the design unconditionally
//!   sound: a bug anywhere in the *coverage* of the hooks degrades performance,
//!   not correctness; only a false *positive* witness could be unsound, and
//!   the recording sites are exactly "a successor was emitted through this
//!   frame" plus the exact value-diff test.
//!
//! # Fail-closed identity matching
//!
//! A frame key is `(operator-definition pointer, argument values)`. The
//! registration side (built in `prepare_inline_fairness_cache` from the
//! fairness `ActionPredHint`s) resolves the hint's operator name in the model
//! checker's root context and records `Arc::as_ptr` of that definition; the
//! enumerator side records the pointer of the definition IT resolved. Any
//! divergence — a LET-local operator shadowing the name, an INSTANCE-scoped
//! operator of the same name, a re-defined operator — yields a different
//! pointer and silently records nothing (fallback). Hints whose resolved body
//! crosses an INSTANCE boundary (`split_action_fast_path_safe == false`),
//! quantified/compound fairness actions that produce no `(name, args)` hint,
//! and leaves whose subscript does not provably cover all state variables are
//! never registered — the exact failure modes of the original #3100 bypass
//! fail CLOSED here instead of corrupting results.
//!
//! # Attribution scoping
//!
//! Witnesses are only valid for the state the BFS is currently expanding.
//! Recording is therefore active only when ALL of:
//! * the scratch is ARMED (`arm_state_guard`, wrapped tightly around the BFS
//!   successor-generation call for one parent state; the RAII guard disarms
//!   on scope exit, including error paths);
//! * the enumeration depth is exactly 1 (`enum_scope` guards at every
//!   top-level unified-enumeration entry): a NESTED enumeration — e.g. an
//!   `ENABLED` evaluation triggered from a streaming sink callback — runs at
//!   depth ≥ 2 and never records;
//! * no suppression scope is active (`suppress_scope` around the
//!   `might_need_prime_binding` post-validation path, whose emissions are
//!   provisional until re-validated — validated survivors are re-noted after
//!   the filter).
//!
//! Consumption (`witnessed_true`) verifies the queried fingerprint matches the
//! armed state's fingerprint; anything else answers "no witness" (fallback).
//!
//! Kill switch: `TY_DISABLE_ENABLED_PROVENANCE=1` disables registration, so
//! arming, frame matching and consumption all become no-ops and every leaf
//! takes the full evaluation path. This covers the absence-side guard plans
//! too — including the #guard-moduleref instance-scoped guards (Step A) and
//! the #guard-in-memo refinements (Step B), which are additionally covered by
//! the finer `TY_DISABLE_ENABLED_GUARD_MEMO=1` (Step B only).

use crate::eval::EvalCtx;
use crate::state::Fingerprint;
use crate::Value;
use rustc_hash::{FxHashMap, FxHashSet};
use smallvec::SmallVec;
use std::cell::{Cell, RefCell};
use std::sync::Arc;
use tla_core::ast::{Expr, ModuleTarget, Substitution};
use tla_core::{OpEnv, Spanned};

/// Tags (with their `require_state_change` flag) that one registered
/// `(operator, args)` action identity decides.
type FrameTags = SmallVec<[(u32, bool); 2]>;

/// Registered state-level guard prefix of one ENABLED leaf's action
/// (the SOUND absence side; see [`guard_prefix_refutes`]).
///
/// `guards` are owned clones of the LEADING run of prime-free, operator-safe
/// conjuncts of the leaf's RESOLVED action expression (cloned once at
/// registration and kept alive for the whole run — run-stable for the
/// pointer-keyed eval caches); `bindings` are the leaf's own quantifier
/// bindings (the same chain `eval_enabled_uncached` binds before enumerating).
pub(crate) struct GuardPlan {
    pub(crate) enabled_tag: u32,
    pub(crate) guards: Vec<GuardEntry>,
    /// The leaf's eager quantifier bindings, with their interned `NameId`s
    /// (resolved once at plan build — the consumption walk pushes them per
    /// state per leaf, and re-interning there measurably showed up in
    /// profiles).
    pub(crate) bindings: Vec<(Arc<str>, tla_core::name_intern::NameId, Value)>,
}

/// One registered guard conjunct of a [`GuardPlan`].
pub(crate) struct GuardEntry {
    /// The prime-free, operator-safe conjunct itself (owned clone, run-stable
    /// for the pointer-keyed eval caches).
    pub(crate) expr: Arc<Spanned<Expr>>,
    /// #guard-moduleref (Step A): when `Some`, the conjunct was lifted from
    /// the resolved body of an `Instance!Op` conjunct and must be evaluated
    /// under the replayed INSTANCE module scope (see [`InstanceGuardScope`]).
    /// `None` = evaluate directly in the consumption context (the original
    /// behavior).
    pub(crate) scope: Option<Arc<InstanceGuardScope>>,
    /// #guard-in-memo (Step B): when `Some`, the conjunct is
    /// `<binding-const> \in <bare-state-variable>` and is decided by a
    /// per-state memoized RHS value + `Value::set_contains` (see
    /// [`MemoInGuard`]). Only ever `Some` on root-scope (`scope == None`)
    /// entries.
    pub(crate) memo: Option<MemoInGuard>,
    /// #guard-in-memo (Step B): a guard whose value is a pure function of the
    /// plan's binding CONSTANTS (no state variables, no operators — see
    /// `expr_is_binding_const`), pre-evaluated once at plan build. Only
    /// `Some(false)` survives registration (an always-FALSE necessary
    /// condition: reaching it in evaluation order refutes the action);
    /// always-TRUE guards are dropped at build (a proven conjunct never
    /// refutes and never errors).
    pub(crate) const_val: Option<bool>,
    /// #guard-share: per-state outcome sharing group, assigned at
    /// registration (`u32::MAX` = unassigned). Two entries share a group only
    /// when their guard EVALUATION is provably the identical deterministic
    /// function of the state: structurally identical expression (Debug key),
    /// identical plan-binding values (Debug key), and — for instance-scoped
    /// guards — an identical replay recipe (same instance + operator, whose
    /// resolution from the pure root context is deterministic, plus
    /// identical formal argument expressions). Fairness/property-plan twin
    /// leaves (e.g. the Allocator's `Sched!Allocate` guards, derived once
    /// for the fairness leaf and once for the plan leaf with the same `c`)
    /// then evaluate once per state instead of once per twin. Only CLEAN
    /// boolean outcomes are shared; errors are never memoized (each walk
    /// re-encounters them, preserving no-claim behavior).
    pub(crate) share_group: u32,
}

/// #guard-moduleref (Step A): the replay recipe for evaluating guards lifted
/// out of a resolved `Instance!Op` body in the instanced module's scope.
///
/// `local_ops` and `subs` are captured VERBATIM from the `EvalCtx` that
/// `resolve_named_module_ref_body_ast_with_params` (the same fail-closed
/// AST resolver the liveness pinning proofs trust) produced at plan-build
/// time. Both are pure functions of run-static inputs — the instanced module
/// operator environments and the composed INSTANCE substitutions — PROVIDED
/// neither the build context nor the consumption context carries an active
/// local-op scope or INSTANCE substitutions of its own (both gates are
/// enforced fail-closed, at build in `module_ref_guard_entries` and at
/// consumption in `eval_instance_scoped_guard`). Under those gates,
/// `consumption_ctx.with_module_scope_arced_subs(local_ops, formals, subs)`
/// is bit-identical to re-running the resolver against the consumption
/// context — i.e. the guard evaluates in exactly the scope the full ENABLED
/// evaluation would enter when descending the same `Instance!Op` conjunct —
/// while the current state stays visible through the consumption context the
/// scope is layered onto.
///
/// `formals` pairs each formal parameter name of the resolved operator with
/// the caller-side argument EXPRESSION (state-level, verified at build);
/// the args are evaluated per state in the consumption context (with the
/// plan bindings pushed) and bound onto the scoped context — the same
/// binding the evaluating resolver (`resolve_module_ref_body`) performs.
pub(crate) struct InstanceGuardScope {
    pub(crate) local_ops: Arc<OpEnv>,
    pub(crate) subs: Arc<Vec<Substitution>>,
    pub(crate) formals: Vec<(Arc<str>, Arc<Spanned<Expr>>)>,
    /// The `(instance name, operator name)` the recipe was resolved from.
    /// Resolution from the pure root context is deterministic, so this pair
    /// identifies the `local_ops`/`subs` CONTENT across independently
    /// resolved copies (#guard-share group keying).
    pub(crate) resolved_from: (String, String),
}

/// #guard-in-memo (Step B): decision recipe for a `x \in v` guard where `x`
/// is one of the leaf's binding CONSTANTS and `v` is a bare STATE VARIABLE
/// (never an operator, never shadowed by the plan bindings — verified
/// fail-closed at build by `detect_memo_in_guard`).
///
/// The RHS value is a pure function of the current state, shared across every
/// leaf whose guard has a structurally identical RHS (`group`, assigned at
/// registration): it is evaluated ONCE per state (memoized keyed by the state
/// fingerprint — the same fp64 identity the `(fp, tag)` ENABLED cache already
/// trusts) and each leaf is decided by `rhs.set_contains(&lhs)`, the EXACT
/// membership routine `eval_in` bottoms out in for a concrete RHS value. An
/// indeterminate `set_contains` (`None`) falls back to the canonical full
/// evaluation of the guard expression.
pub(crate) struct MemoInGuard {
    pub(crate) lhs: Value,
    pub(crate) rhs: Arc<Spanned<Expr>>,
    /// Memo sharing group, assigned in [`extend_guard_plans`]. `u32::MAX`
    /// until registration.
    pub(crate) group: u32,
}

/// One argument position of a registered frame identity.
#[derive(Clone, PartialEq)]
pub(crate) enum ArgPattern {
    /// The leaf action fixes this formal to a const value; a frame matches
    /// only with the identical value.
    Exact(Value),
    /// The leaf action existentially quantifies this formal over the given
    /// (const-enumerated) domain (#3208 wildcard frames); a frame matches
    /// when its value at this position is a MEMBER of the domain — so an
    /// emission witnesses the existential even if `Next` quantifies the same
    /// operator over a wider domain.
    AnyOf(Vec<Value>),
}

impl ArgPattern {
    #[inline]
    fn matches(&self, value: &Value) -> bool {
        match self {
            ArgPattern::Exact(v) => v == value,
            ArgPattern::AnyOf(domain) => domain.contains(value),
        }
    }
}

/// One TRUE-side (frame witness) eligible ENABLED leaf, registered at
/// inline-fairness plan build time. Identity gates: unique plain `Op(args)`
/// (or `\E x \in D: Op(.., x, ..)`) hint, INSTANCE-free resolved body,
/// root-context resolution with const-level argument values/domains, and —
/// for `require_state_change` leaves — the subscript-coverage proof.
pub(crate) struct RegisteredEnabledLeaf {
    /// `Arc::as_ptr` of the resolved root-context operator definition.
    pub(crate) def_ptr: usize,
    /// Per-argument match patterns, in formal-parameter order (empty for
    /// zero-argument operators).
    pub(crate) args: Vec<ArgPattern>,
    /// The `LiveExpr::Enabled` leaf tag this identity decides.
    pub(crate) enabled_tag: u32,
    /// The leaf's `require_state_change`: when true, only a state-changing
    /// emission is a witness (the leaf's subscript was proven to cover all
    /// state variables at registration).
    pub(crate) needs_change: bool,
}

/// Per-operator frame candidates: the pattern list plus whether any pattern
/// contains a wildcard (`AnyOf`) position. Exact-only candidate lists are
/// mutually exclusive (distinct values), so the matcher may stop at the first
/// hit; with wildcards, several candidates can match one frame and their tags
/// are merged.
#[derive(Default)]
struct DefCandidates {
    cands: Vec<(Vec<ArgPattern>, FrameTags)>,
    has_wildcard: bool,
}

/// #frame-fp-pop: per-arm recorded emitted-successor fingerprints of ONE
/// FALSE-eligible ENABLED tag (see [`populate_pair_from_frame_fps`]).
///
/// An entry exists exactly when a frame carrying the tag was PUSHED during the
/// current arm AND the tag passed the static FALSE-eligibility certificate at
/// registration (`frame_false_tags`). `fps` then accumulates the fingerprint
/// of every RAW emission noted while such a frame was active. `incomplete`
/// poisons the entry fail-closed (fp unavailable, per-tag cap overflow, or an
/// unattributed replay emission under an active frame) — a poisoned entry is
/// never consumed.
#[derive(Default)]
struct FrameFpEntry {
    fps: Vec<Fingerprint>,
    incomplete: bool,
}

/// #frame-fp-pop: fail-closed cap on recorded fingerprints per tag per arm
/// (memory guard; overflow poisons the entry rather than truncating it).
const FRAME_FP_CAP: usize = 4096;

#[derive(Default)]
struct ProvState {
    /// Registered identities: def pointer → candidates. Small linear per-def
    /// candidate lists keep `Value` out of hash keys.
    reg: FxHashMap<usize, DefCandidates>,
    /// #frame-fp-pop: ENABLED tags whose paired ActionPred may be populated
    /// FALSE from recorded frame fingerprints — statically certified at
    /// registration (all-Exact frame args + Next-shape frame-completeness
    /// certificate + all-vars pinning proof on the pair; see
    /// [`next_shape_frame_complete`]).
    frame_false_tags: FxHashSet<u32>,
    /// #frame-fp-pop: per-arm per-tag recorded emission fingerprints
    /// (entries created at frame push, cleared at every arm).
    frame_fps: FxHashMap<u32, FrameFpEntry>,
    /// #frame-fp-pop: sticky per-arm poison — some armed depth-1 enumeration
    /// ended WITHOUT an explicit completion mark (sink stopped early, error
    /// unwind, or an enumeration entry point that does not participate in the
    /// completion protocol). FALSE population is then refused for the whole
    /// arm (recorded sets may be truncated).
    arm_enum_incomplete: bool,
    /// Absence-side plans: enabled tag → guard prefix (see [`GuardPlan`]).
    guard_plans: FxHashMap<u32, Arc<GuardPlan>>,
    /// #guard-in-memo: structural-RHS-key → memo sharing group id
    /// (registration-time only; see [`extend_guard_plans`]).
    memo_groups: FxHashMap<String, u32>,
    /// #guard-in-memo: per-group last evaluated RHS value, keyed by the state
    /// fingerprint it belongs to (a mismatching fingerprint is a miss and the
    /// slot is overwritten — one live state per group at a time).
    memo_vals: FxHashMap<u32, (Fingerprint, Value)>,
    /// #guard-share: identity-key → outcome sharing group id
    /// (registration-time only; see [`extend_guard_plans`]).
    share_groups: FxHashMap<String, u32>,
    /// #guard-share: per-group last CLEAN guard outcome, keyed by the state
    /// fingerprint it belongs to (same one-live-state discipline as
    /// `memo_vals`).
    guard_outcomes: FxHashMap<u32, (Fingerprint, bool)>,
    /// Scratch armed for the current BFS parent state.
    armed: bool,
    /// Top-level unified-enumeration nesting depth.
    enum_depth: u32,
    /// Provisional-emission suppression depth (prime-binding validation).
    suppress: u32,
    /// The BFS parent state the witnesses belong to.
    scratch_fp: Option<Fingerprint>,
    /// ENABLED leaf tags proven TRUE for `scratch_fp`.
    witnessed: FxHashSet<u32>,
    /// Active matched frames on the current descent path.
    frames: Vec<FrameTags>,
    /// Diagnostics (liveness_profile): consumption hits / registered leaves.
    hits: u64,
    registered: usize,
    /// Diagnostics: #frame-fp-pop consumption hits / fallbacks.
    frame_pop_hits: u64,
    frame_pop_fallbacks: u64,
    /// Diagnostics: full ENABLED evaluations by outcome (provenance misses).
    full_true: u64,
    full_false: u64,
    /// Diagnostics: absence-side guard-prefix refutations (total, and the
    /// memo-decided / const-decided subsets).
    guard_refutes: u64,
    guard_memo_refutes: u64,
    guard_const_refutes: u64,
    /// Diagnostics: #guard-share outcome-memo hits.
    guard_share_hits: u64,
}

thread_local! {
    static PROV: RefCell<ProvState> = RefCell::new(ProvState::default());
    /// Hot-path gate: true only while hooks may record
    /// (armed ∧ depth == 1 ∧ suppress == 0 ∧ registration non-empty).
    static PROV_HOT: Cell<bool> = const { Cell::new(false) };
}

/// Kill switch: `TY_DISABLE_ENABLED_PROVENANCE=1` forces full evaluation of
/// every ENABLED leaf (used by the verdict-identity differential). Disables
/// ALL of this module: frame witnesses, guard plans (including the
/// #guard-moduleref instance-scoped guards and #guard-in-memo memoization).
pub(crate) fn provenance_disabled() -> bool {
    static FLAG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *FLAG.get_or_init(|| std::env::var("TY_DISABLE_ENABLED_PROVENANCE").is_ok_and(|v| v == "1"))
}

/// Fine-grained kill switch: `TY_DISABLE_ENABLED_GUARD_MEMO=1` disables the
/// #guard-in-memo refinements (per-state RHS memoization and binding-const
/// guard pre-evaluation) while keeping the plain per-leaf guard evaluation —
/// plans then behave exactly as before Step B. Subsumed by
/// `TY_DISABLE_ENABLED_PROVENANCE=1` (which registers no plans at all).
pub(crate) fn guard_memo_disabled() -> bool {
    static FLAG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *FLAG.get_or_init(|| std::env::var("TY_DISABLE_ENABLED_GUARD_MEMO").is_ok_and(|v| v == "1"))
}

/// Fine-grained kill switch: `TY_DISABLE_ENABLED_QUANT_GUARDS=1` disables the
/// #guard-exists derivations (existential-body descent, `x # {} ∧ x ⊆ E ⇒
/// E # {}` pair inference, and inner-existential wrap lifts) while keeping
/// the plain and #guard-moduleref extraction — plans then behave exactly as
/// before the quantified extension. Subsumed by
/// `TY_DISABLE_ENABLED_PROVENANCE=1`.
pub(crate) fn quant_guards_disabled() -> bool {
    static FLAG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *FLAG.get_or_init(|| std::env::var("TY_DISABLE_ENABLED_QUANT_GUARDS").is_ok_and(|v| v == "1"))
}

/// Kill switch for the #frame-fp-pop mechanism (default ON).
///
/// `TY_DISABLE_FRAME_FP_POPULATION=1` disables frame-fingerprint recording and
/// consumption entirely: no `frame_false_tags` registration, no per-emission
/// fingerprint computation, no per-arm bookkeeping — and
/// [`populate_pair_from_frame_fps`] answers `false`, so every witnessed leaf
/// falls back to the LANDED re-enumeration population
/// (`populate_witnessed_pair`), which itself falls back further via
/// `TY_DISABLE_WITNESS_PAIR_POPULATION=1`. Used by the verdict-identity
/// differential.
pub(crate) fn frame_fp_population_enabled() -> bool {
    static FLAG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *FLAG.get_or_init(|| !std::env::var("TY_DISABLE_FRAME_FP_POPULATION").is_ok_and(|v| v == "1"))
}

fn hot(st: &ProvState) -> bool {
    st.armed && st.enum_depth == 1 && st.suppress == 0 && !st.reg.is_empty()
}

/// Replace the registered identities for this run. Clears all per-state
/// scratch. Called from `prepare_inline_fairness_cache` (with an empty vec at
/// the top of preparation so early returns leave nothing stale).
pub(crate) fn register(leaves: Vec<RegisteredEnabledLeaf>) {
    PROV.with(|p| {
        let mut st = p.borrow_mut();
        st.reg.clear();
        st.guard_plans.clear();
        st.memo_groups.clear();
        st.memo_vals.clear();
        st.share_groups.clear();
        st.guard_outcomes.clear();
        st.frame_false_tags.clear();
        st.frame_fps.clear();
        st.arm_enum_incomplete = false;
        st.armed = false;
        st.scratch_fp = None;
        st.witnessed.clear();
        st.frames.clear();
        st.hits = 0;
        st.registered = 0;
        st.frame_pop_hits = 0;
        st.frame_pop_fallbacks = 0;
        st.full_true = 0;
        st.full_false = 0;
        st.guard_refutes = 0;
        st.guard_memo_refutes = 0;
        st.guard_const_refutes = 0;
        st.guard_share_hits = 0;
        if !provenance_disabled() {
            for leaf in leaves {
                st.registered += 1;
                let entry = st.reg.entry(leaf.def_ptr).or_default();
                entry.has_wildcard |= leaf.args.iter().any(|a| matches!(a, ArgPattern::AnyOf(_)));
                if let Some((_, tags)) = entry.cands.iter_mut().find(|(a, _)| *a == leaf.args) {
                    tags.push((leaf.enabled_tag, leaf.needs_change));
                } else {
                    let mut tags = FrameTags::new();
                    tags.push((leaf.enabled_tag, leaf.needs_change));
                    entry.cands.push((leaf.args, tags));
                }
            }
        }
        PROV_HOT.set(false);
    });
}

/// Add absence-side guard plans (see [`GuardPlan`] / [`guard_prefix_refutes`]).
/// A tag that already has a plan keeps the existing one (first registration
/// wins — the fairness registration runs before any property-plan extension,
/// and identical tags denote identical leaves in the shared tag space).
/// No-op under the kill switch.
///
/// #guard-in-memo: memo sharing groups are assigned here, spanning EVERY
/// registration of the run (fairness plans + property-plan extension). The
/// group key is the structural identity of the memoized RHS — the Debug
/// serialization of its AST. Distinct leaves resolve their own clones of the
/// same source conjunct (pointer identity never dedups), and equal Debug keys
/// mean the same (root-scope) expression evaluated in the same per-state
/// context, so sharing one RHS value per group per state is exact.
/// Over-splitting (e.g. differing spans) merely costs sharing, never
/// correctness.
pub(crate) fn extend_guard_plans(plans: Vec<GuardPlan>) {
    if provenance_disabled() {
        return;
    }
    PROV.with(|p| {
        let mut st = p.borrow_mut();
        for mut plan in plans {
            let tag = plan.enabled_tag;
            if st.guard_plans.contains_key(&tag) {
                continue;
            }
            let bindings_key = format!(
                "{:?}",
                plan.bindings
                    .iter()
                    .map(|(n, _, v)| (n, v))
                    .collect::<Vec<_>>()
            );
            for entry in &mut plan.guards {
                if let Some(memo) = &mut entry.memo {
                    let key = normalize_identity_key(&format!("{:?}", memo.rhs));
                    let next = st.memo_groups.len() as u32;
                    memo.group = *st.memo_groups.entry(key).or_insert(next);
                }
                // #guard-share group assignment (see GuardEntry::share_group
                // for the identity argument).
                let scope_key = match &entry.scope {
                    None => String::new(),
                    Some(s) => format!(
                        "{}!{}|{:?}",
                        s.resolved_from.0, s.resolved_from.1, s.formals
                    ),
                };
                let key =
                    normalize_identity_key(&format!("{:?}|{bindings_key}|{scope_key}", entry.expr));
                let next = st.share_groups.len() as u32;
                entry.share_group = *st.share_groups.entry(key).or_insert(next);
            }
            st.guard_plans.insert(tag, Arc::new(plan));
        }
    });
}

/// Clear all registration and scratch (equivalent to registering nothing).
pub(crate) fn clear() {
    register(Vec::new());
}

/// Drop all backing allocations owned by ENABLED provenance on this thread.
///
/// [`clear`] intentionally keeps registration and scratch capacity warm across
/// ordinary property boundaries. Once a mid-BFS regeneration trip disables
/// inline recording for the run, none of that capacity will be reused, so the
/// trip replaces the complete state with a fresh default value.
pub(crate) fn release_enabled_provenance_storage() {
    PROV.with(|state| *state.borrow_mut() = ProvState::default());
    PROV_HOT.with(|hot| hot.set(false));
}

/// Normalize a Debug-rendered AST identity key by stripping the two
/// semantically inert annotations that vary between independently resolved
/// copies of the same source expression:
///
///   - `NameId(<n>)` → `NameId()`: a lowered `Ident`/`StateVar` carries either
///     `NameId::INVALID` or the global interner's id FOR ITS OWN NAME STRING
///     (ids are only ever embedded by interning that exact name), so the id
///     adds no semantic content beyond the name — evaluation resolves both
///     forms to the same binding (id-keyed chain hit vs name-keyed waterfall).
///   - `@ <start>..<end>` spans → `@`: source locations never enter
///     evaluation semantics (span-keyed caches key VALUES of the same
///     expression, not different values).
///
/// Equal normalized keys therefore denote expressions whose evaluation is the
/// same function of (scope recipe, bindings, state). Over-splitting from any
/// remaining Debug noise merely costs sharing, never correctness.
fn normalize_identity_key(debug: &str) -> String {
    let mut out = String::with_capacity(debug.len());
    let mut rest = debug;
    while !rest.is_empty() {
        if let Some(tail) = rest.strip_prefix("NameId(") {
            out.push_str("NameId(");
            rest = tail.trim_start_matches(|c: char| c.is_ascii_digit());
            continue;
        }
        if let Some(tail) = rest.strip_prefix("@ ") {
            out.push('@');
            rest = tail.trim_start_matches(|c: char| c.is_ascii_digit() || c == '.');
            continue;
        }
        let ch = rest.chars().next().expect("nonempty");
        out.push(ch);
        rest = &rest[ch.len_utf8()..];
    }
    out
}

/// RAII guard arming the per-state witness scratch; disarms on drop.
pub(crate) struct ArmGuard {
    _priv: (),
}

/// Arm witness collection for the BFS parent state `fp`. Must wrap ONLY the
/// successor-generation call for that state (see module docs). No-op (stays
/// disarmed) when nothing is registered.
pub(crate) fn arm_state_guard(fp: Fingerprint) -> ArmGuard {
    PROV.with(|p| {
        let mut st = p.borrow_mut();
        st.witnessed.clear();
        st.frames.clear();
        // #frame-fp-pop: recorded fingerprints belong to exactly one armed
        // state; clear them (and the per-arm truncation poison) at every arm.
        st.frame_fps.clear();
        st.arm_enum_incomplete = false;
        if st.reg.is_empty() {
            st.armed = false;
            st.scratch_fp = None;
        } else {
            st.armed = true;
            st.scratch_fp = Some(fp);
        }
        PROV_HOT.set(hot(&st));
    });
    ArmGuard { _priv: () }
}

/// #frame-fp-pop: register the ENABLED tags whose paired ActionPred is
/// FALSE-populatable from frame fingerprints (statically certified at plan
/// build — see the registration site in `inline_fairness.rs`). Called AFTER
/// [`register`] (which clears the set). No-op under either kill switch.
pub(crate) fn register_frame_false_tags(tags: Vec<u32>) {
    if provenance_disabled() || !frame_fp_population_enabled() {
        return;
    }
    PROV.with(|p| {
        let mut st = p.borrow_mut();
        st.frame_false_tags.extend(tags);
    });
}

impl Drop for ArmGuard {
    fn drop(&mut self) {
        PROV.with(|p| {
            let mut st = p.borrow_mut();
            st.armed = false;
            st.frames.clear();
            PROV_HOT.set(false);
        });
    }
}

/// RAII depth guard for one top-level unified enumeration. Placed at EVERY
/// entry that runs the unified recursion (`run_unified_with_options`,
/// `run_unified_into_with_options`, `enumerate_action_successors_*`), so a
/// nested enumeration — an ENABLED evaluation fired from a streaming sink
/// callback, for example — runs at depth ≥ 2 and never records.
///
/// #frame-fp-pop completion protocol: an ARMED depth-1 scope must be told the
/// enumeration ran to completion ([`EnumScope::mark_complete`] — called by the
/// entry point when the enumeration returned `Ok` AND its sink was never
/// stopped). A scope dropped WITHOUT the mark — sink Break, error unwind, or
/// an entry point that does not participate in the protocol (e.g. the capped
/// ENABLED enumeration, which never runs armed in production) — poisons the
/// whole arm fail-closed (`arm_enum_incomplete`): recorded frame-fingerprint
/// sets may then be truncated prefixes, so FALSE population is refused for
/// this state. TRUE-side witnesses are unaffected (a lost or truncated set
/// only loses witnesses).
pub(crate) struct EnumScope {
    /// Scope was opened while armed at depth 1 with #frame-fp-pop enabled —
    /// the only case the completion protocol applies to.
    armed_depth1: bool,
    completed: bool,
}

pub(crate) fn enum_scope() -> EnumScope {
    let armed_depth1 = PROV.with(|p| {
        let mut st = p.borrow_mut();
        st.enum_depth += 1;
        PROV_HOT.set(hot(&st));
        st.armed && st.enum_depth == 1
    });
    EnumScope {
        armed_depth1: armed_depth1 && frame_fp_population_enabled(),
        completed: false,
    }
}

impl EnumScope {
    /// #frame-fp-pop: the enumeration this scope brackets ran to completion
    /// (Ok result, sink never stopped). See the completion protocol above.
    pub(crate) fn mark_complete(&mut self) {
        self.completed = true;
    }
}

impl Drop for EnumScope {
    fn drop(&mut self) {
        PROV.with(|p| {
            let mut st = p.borrow_mut();
            st.enum_depth = st.enum_depth.saturating_sub(1);
            if self.armed_depth1 && !self.completed {
                st.arm_enum_incomplete = true;
            }
            PROV_HOT.set(hot(&st));
        });
    }
}

/// RAII suppression guard for enumeration regions whose emissions are
/// PROVISIONAL (dropped by a later validation pass). Recording a provisional
/// emission as a witness would be unsound; validated survivors are re-noted
/// by the caller after the filter.
pub(crate) struct SuppressScope {
    _priv: (),
}

pub(crate) fn suppress_scope() -> SuppressScope {
    PROV.with(|p| {
        let mut st = p.borrow_mut();
        st.suppress += 1;
        PROV_HOT.set(hot(&st));
    });
    SuppressScope { _priv: () }
}

impl Drop for SuppressScope {
    fn drop(&mut self) {
        PROV.with(|p| {
            let mut st = p.borrow_mut();
            st.suppress = st.suppress.saturating_sub(1);
            PROV_HOT.set(hot(&st));
        });
    }
}

/// Cheap prefilter for the enumerator's operator-application sites: should
/// argument values be collected for a frame on this operator definition?
#[inline]
pub(crate) fn wants_frame(def_ptr: usize) -> bool {
    if !PROV_HOT.with(Cell::get) {
        return false;
    }
    PROV.with(|p| p.borrow().reg.contains_key(&def_ptr))
}

/// RAII frame guard: pops the frame on drop (error paths included).
pub(crate) struct FrameGuard {
    pushed: bool,
}

/// Push a frame for the operator application `def_ptr(args)` if it matches a
/// registered identity. Returns a guard that pops the frame on drop; a
/// non-matching key pushes nothing (and the guard is inert).
pub(crate) fn push_frame(def_ptr: usize, args: &[Value]) -> FrameGuard {
    if !PROV_HOT.with(Cell::get) {
        return FrameGuard { pushed: false };
    }
    PROV.with(|p| {
        let mut st = p.borrow_mut();
        // A frame may match SEVERAL registered identities (e.g. the exact
        // per-binding leaf AND a wildcard existential leaf of the same
        // operator): merge every matching candidate's tags. Exact-only
        // candidate lists are mutually exclusive — stop at the first hit.
        let mut tags = FrameTags::new();
        if let Some(entry) = st.reg.get(&def_ptr) {
            for (pattern, cand_tags) in &entry.cands {
                if pattern.len() == args.len()
                    && pattern.iter().zip(args.iter()).all(|(p, v)| p.matches(v))
                {
                    tags.extend(cand_tags.iter().copied());
                    if !entry.has_wildcard {
                        break;
                    }
                }
            }
        }
        if tags.is_empty() {
            FrameGuard { pushed: false }
        } else {
            // #frame-fp-pop: a matching frame push for a FALSE-eligible tag
            // opens its per-arm fingerprint record. The mere existence of the
            // entry encodes "frame pushed at least once this arm" — under the
            // Next-shape certificate, a matching push at ANY application site
            // enumerates the action relation completely, so the recorded set
            // is exact once the arm completes untruncated.
            let st = &mut *st;
            for &(tag, _) in &tags {
                if st.frame_false_tags.contains(&tag) {
                    st.frame_fps.entry(tag).or_default();
                }
            }
            st.frames.push(tags);
            FrameGuard { pushed: true }
        }
    })
}

impl Drop for FrameGuard {
    fn drop(&mut self) {
        if self.pushed {
            PROV.with(|p| {
                let _ = p.borrow_mut().frames.pop();
            });
        }
    }
}

/// Record one RAW successor emission for every active frame.
///
/// `base_fp` is the (cached) fingerprint of the enumeration's base state, used
/// only to debug-assert the arm bracketing; `changed` decides — by exact value
/// comparison against the base state, computed lazily and at most once —
/// whether the emission changes the state (the witness condition for
/// `require_state_change` leaves whose subscript covers all variables).
///
/// #frame-fp-pop: `succ_fp` computes — lazily, at most once, and ONLY when an
/// active frame carries a FALSE-eligible tag with an open fingerprint record —
/// the emitted successor's fingerprint (the same incremental XOR fingerprint
/// the BFS worker later computes for the materialized successor, so recorded
/// fps compare bit-identically against the cached-successor fps at
/// consumption). `None` = the fingerprint is unavailable at this site: the
/// affected records are poisoned fail-closed (FALSE population refused, the
/// witnessed-TRUE side untouched).
#[inline]
pub(crate) fn note_emission(
    base_fp: Option<Fingerprint>,
    changed: impl FnOnce() -> bool,
    succ_fp: impl FnOnce() -> Option<Fingerprint>,
) {
    if !PROV_HOT.with(Cell::get) {
        return;
    }
    PROV.with(|p| {
        let st = &mut *p.borrow_mut();
        if st.frames.is_empty() {
            return;
        }
        debug_assert!(
            base_fp.is_none() || base_fp == st.scratch_fp,
            "ENABLED provenance: emission base state does not match the armed state"
        );
        let ch = changed();
        // Lazy one-shot successor fingerprint (shared across all frames of
        // this emission).
        let mut succ_fp = Some(succ_fp);
        let mut fp_memo: Option<Option<Fingerprint>> = None;
        for frame in &st.frames {
            for &(tag, needs_change) in frame {
                if !needs_change || ch {
                    st.witnessed.insert(tag);
                }
                // #frame-fp-pop: record the emission fp for open records.
                // Entries exist only for FALSE-eligible tags whose frame was
                // pushed this arm (see push_frame), so this is a no-op on
                // every other spec/tag.
                if let Some(entry) = st.frame_fps.get_mut(&tag) {
                    if !entry.incomplete {
                        let fp = *fp_memo.get_or_insert_with(|| {
                            succ_fp.take().expect("succ_fp evaluated once")()
                        });
                        match fp {
                            Some(f) if entry.fps.len() < FRAME_FP_CAP => entry.fps.push(f),
                            // Cap overflow or unavailable fp: poison the
                            // record (fail closed — the set can no longer be
                            // proven exact).
                            _ => entry.incomplete = true,
                        }
                    }
                }
            }
        }
    });
}

/// #frame-fp-pop: an emission was delivered to the sink WITHOUT flowing
/// through [`note_emission`] (the state-independent Or-branch replay cache
/// hit, which re-pushes successors recorded on an earlier state). If any
/// frame is active, its open fingerprint records can no longer be proven
/// complete — poison them fail-closed. Witnesses are NOT recorded (identical
/// to the landed TRUE-side behavior on this path: a replayed emission simply
/// loses its witness).
#[inline]
pub(crate) fn note_unattributed_emission() {
    if !PROV_HOT.with(Cell::get) {
        return;
    }
    PROV.with(|p| {
        let st = &mut *p.borrow_mut();
        if st.frames.is_empty() {
            return;
        }
        for frame in &st.frames {
            for &(tag, _) in frame {
                if let Some(entry) = st.frame_fps.get_mut(&tag) {
                    entry.incomplete = true;
                }
            }
        }
    });
}

/// #frame-fp-pop: populate the paired ActionPred scratchpad for ONE witnessed
/// ENABLED leaf purely from the frame fingerprints recorded during this
/// state's armed BFS generation — ZERO re-enumeration.
///
/// Returns `true` only when the population actually happened (TRUE membership
/// AND FALSE non-membership entries written for every cached successor);
/// `false` is "no claim" and the caller must fall back to the landed
/// re-enumeration population (`populate_witnessed_pair`).
///
/// # Soundness
///
/// TRUE side (fp present ⇒ the transition satisfies the action): every
/// recorded fp is the fingerprint of a RAW emission noted while a frame
/// matching this leaf's `(operator, argument-values)` identity was on the
/// descent path — the emission satisfies, by construction of the unified
/// enumerator, every constraint on that path INCLUDING the full frame body
/// under exactly those argument values. Same fp64-identity trust as the
/// enumeration-based population and the BFS dedup itself.
///
/// FALSE side (fp absent ⇒ the transition does NOT satisfy the action)
/// requires the recorded set to be EXACTLY the action relation from this
/// state. All gates fail closed:
///   * the caller verified the all-vars pinning proof
///     (`full_population_tag(pair_tag)`) — the action's enumeration emits
///     exactly its relation;
///   * `frame_false_tags` — the static Next-shape certificate
///     ([`next_shape_frame_complete`]): every application of the leaf's
///     operator the enumerator can descend to inside the BFS Next expression
///     sits in a purely disjunctive context (Or / prime-free-domain Exists /
///     Label ancestors only) and the leaf's frame identity is all-Exact, so
///     ANY matching frame push enumerates the full body relation with no
///     outer constraint pruning it and no wider quantification diluting the
///     match;
///   * a `frame_fps` entry exists — a matching frame WAS pushed this arm
///     (guard-refuted zero-emission descents included: an existing empty
///     record proves the relation empty from this state);
///   * the entry is not poisoned (`incomplete`: fp-unavailable, cap
///     overflow, or an unattributed replay emission under the frame);
///   * the arm is not poisoned (`arm_enum_incomplete`: every armed depth-1
///     enumeration explicitly completed — no sink Break, no error unwind, no
///     non-participating entry point).
///
/// Kill switch `TY_DISABLE_FRAME_FP_POPULATION=1` makes this answer `false`
/// unconditionally (the caller then reproduces the landed behavior exactly).
pub(crate) fn populate_pair_from_frame_fps(
    current_fp: Fingerprint,
    enabled_tag: u32,
    pair_tag: u32,
    succ_fps: impl Iterator<Item = Fingerprint>,
) -> bool {
    if !frame_fp_population_enabled() {
        return false;
    }
    PROV.with(|p| {
        let st = &mut *p.borrow_mut();
        if st.scratch_fp != Some(current_fp)
            || st.arm_enum_incomplete
            || !st.frame_false_tags.contains(&enabled_tag)
        {
            st.frame_pop_fallbacks += 1;
            return false;
        }
        let Some(entry) = st.frame_fps.get(&enabled_tag) else {
            st.frame_pop_fallbacks += 1;
            return false;
        };
        if entry.incomplete {
            st.frame_pop_fallbacks += 1;
            return false;
        }
        for succ_fp in succ_fps {
            let member = entry.fps.contains(&succ_fp);
            super::checker::insert_scan_pred_result(current_fp, succ_fp, pair_tag, member);
        }
        st.frame_pop_hits += 1;
        true
    })
}

/// TRUE-only consumption: `true` iff a genuine witness for `tag` was recorded
/// while generating the successors of the state with fingerprint `fp`. `false`
/// means "no claim" — the caller must fall back to full evaluation. Never
/// asserts ENABLED = false.
#[inline]
pub(crate) fn witnessed_true(fp: Fingerprint, tag: u32) -> bool {
    PROV.with(|p| {
        let mut st = p.borrow_mut();
        let hit = st.scratch_fp == Some(fp) && st.witnessed.contains(&tag);
        if hit {
            st.hits += 1;
        }
        hit
    })
}

// ── #frame-fp-pop: Next-shape frame-completeness certificate ─────────────
//
// Static proof that the ENUMERATOR, descending the BFS Next expression, can
// only ever push a frame for `target` from a PURELY DISJUNCTIVE position —
// every ancestor between the Next root and any reachable application of the
// target operator is an `Or`, a prime-free-domain `Exists`, a transparent
// `Label`, or a pass-through inlined operator body that itself sits in such a
// position. A matching frame push in such a position enumerates the frame
// body under NO outer constraints: no sibling conjunct can prune emissions,
// no outer binding can pin a primed variable, so — together with the all-vars
// pinning proof on the action — the emissions noted under the frame are
// EXACTLY the action relation from the armed state.
//
// Key structural fact the certificate leans on: `push_frame` is called ONLY
// from the enumerator's operator-descent sites (`unified_dispatch`
// Apply/Ident arms, `unified_scope::conjunct_apply`/`conjunct_ident`) — the
// expression EVALUATOR (guards, quantifier domains, assignment right-hand
// sides) never pushes frames and never emits successors. A subtree the
// enumerator cannot descend-reach the target through therefore contributes
// nothing to the target's record no matter what it computes, and the
// certificate only has to (over-)approximate DESCENT reachability
// ([`ReachScan`]): any `Ident`/`Apply`-head/`OpRef`/`ModuleRef` whose name or
// resolved definition pointer matches the target — looking through resolvable
// operator bodies and fail-closed through INSTANCE references — counts as
// reachable. Everything unresolvable or unrecognized counts as reachable too
// (certificate refused), never the other way around.

/// Budget on operator-body descents during one reachability scan (fail
/// closed: exhaustion counts as "reaches").
const REACH_SCAN_BUDGET: u32 = 4096;

/// Conservative transitive scan: can the enumerator's descent of `expr` reach
/// a reference to the target operator (by resolved definition pointer OR by
/// name — the name over-approximation covers INSTANCE substitutions, LET
/// aliases and higher-order passing)?
struct ReachScan<'t> {
    target_ptr: usize,
    target_name: &'t str,
    /// Memoized per-definition results (`Arc::as_ptr` keyed).
    memo: FxHashMap<usize, bool>,
    /// Definitions currently on the descent stack (cycle guard — a cycle
    /// re-entry contributes `false`; any genuine reach has an acyclic path).
    visiting: FxHashSet<usize>,
    budget: u32,
}

impl ReachScan<'_> {
    fn name_matches(&self, ctx: &EvalCtx, name: &str) -> bool {
        name == self.target_name || ctx.resolve_op_name(name) == self.target_name
    }

    /// Does a reference to operator `name` (in `ctx`) reach the target?
    fn op_ref_reaches(&mut self, ctx: &EvalCtx, name: &str) -> bool {
        if self.name_matches(ctx, name) {
            return true;
        }
        let resolved = ctx.resolve_op_name(name);
        let Some(def) = ctx.get_op(resolved) else {
            // Not an operator in this scope (state variable, bound value,
            // constant): cannot resolve to the target definition, and the
            // name check above already covered shadow-name hazards.
            return false;
        };
        let ptr = Arc::as_ptr(def) as usize;
        if ptr == self.target_ptr {
            return true;
        }
        if let Some(&cached) = self.memo.get(&ptr) {
            return cached;
        }
        if self.visiting.contains(&ptr) {
            return false; // cycle re-entry (see field docs)
        }
        if self.budget == 0 {
            return true; // budget exhausted: fail closed
        }
        self.budget -= 1;
        self.visiting.insert(ptr);
        let def = Arc::clone(def);
        let reached = self.reaches(ctx, &def.body.node);
        self.visiting.remove(&ptr);
        self.memo.insert(ptr, reached);
        reached
    }

    fn reaches(&mut self, ctx: &EvalCtx, expr: &Expr) -> bool {
        struct V<'a, 't> {
            scan: &'a mut ReachScan<'t>,
            ctx: &'a EvalCtx,
        }
        impl tla_core::visit::ExprVisitor for V<'_, '_> {
            type Output = bool;
            fn visit_node(&mut self, expr: &Expr) -> Option<bool> {
                match expr {
                    Expr::Ident(name, _) | Expr::OpRef(name) => {
                        Some(self.scan.op_ref_reaches(self.ctx, name))
                    }
                    Expr::StateVar(name, _, _) => {
                        // A lowered StateVar names a state variable, but the
                        // name check keeps shadow hazards fail-closed.
                        Some(self.scan.name_matches(self.ctx, name))
                    }
                    Expr::ModuleRef(target, op_name, args) => {
                        if self.scan.name_matches(self.ctx, op_name) {
                            return Some(true);
                        }
                        // Fail-closed INSTANCE descent: resolve the
                        // substitution-applied body (same resolver the
                        // pinning proofs trust) and scan it in the instanced
                        // scope; an unresolvable reference counts as a reach.
                        if self.scan.budget == 0 {
                            return Some(true);
                        }
                        self.scan.budget -= 1;
                        match crate::enabled::resolve_module_ref_body_ast(
                            self.ctx, target, op_name, args,
                        ) {
                            Some((inner_ctx, body)) => {
                                let mut hit = self.scan.reaches(&inner_ctx, &body.node);
                                // Substituted argument expressions are walked
                                // in the OUTER scope by default traversal of
                                // the args below only when we return None —
                                // we short-circuit here, so scan them
                                // explicitly.
                                for arg in args {
                                    if hit {
                                        break;
                                    }
                                    hit |= self.scan.reaches(self.ctx, &arg.node);
                                }
                                Some(hit)
                            }
                            None => Some(true),
                        }
                    }
                    _ => None,
                }
            }
        }
        tla_core::visit::ExprVisitor::walk_expr(&mut V { scan: self, ctx }, expr)
    }
}

/// Depth cap for the certified-context walk (nesting of Or/Exists/inlined
/// operator bodies; fail closed).
const MAX_FRAME_COMPLETE_DEPTH: u32 = 64;

/// #frame-fp-pop: the Next-shape frame-completeness certificate for one
/// registered target operator definition (see the section comment above).
///
/// `true` iff every enumerator-descent path from `next_body` to a reference
/// of `target` passes exclusively through disjunctive constructs. Fail
/// closed: any unrecognized construct, unresolvable reference, primed or
/// target-reaching quantifier domain, target-reaching argument expression, or
/// depth/budget exhaustion refuses the certificate.
pub(crate) fn next_shape_frame_complete(
    ctx: &EvalCtx,
    next_body: &Spanned<Expr>,
    target_ptr: usize,
    target_name: &str,
) -> bool {
    let mut scan = ReachScan {
        target_ptr,
        target_name,
        memo: FxHashMap::default(),
        visiting: FxHashSet::default(),
        budget: REACH_SCAN_BUDGET,
    };

    fn walk(
        scan: &mut ReachScan<'_>,
        ctx: &EvalCtx,
        expr: &Spanned<Expr>,
        target_ptr: usize,
        depth: u32,
    ) -> bool {
        if depth >= MAX_FRAME_COMPLETE_DEPTH {
            return false;
        }
        match &expr.node {
            // Disjunction: both branches stay disjunctive.
            Expr::Or(a, b) => {
                walk(scan, ctx, a, target_ptr, depth + 1)
                    && walk(scan, ctx, b, target_ptr, depth + 1)
            }
            // Label: the dispatcher unwraps it transparently.
            Expr::Label(label) => walk(scan, ctx, &label.body, target_ptr, depth + 1),
            // Existential: domains must be prime-free, evaluated-only (no
            // target reach), and every binding is enumerated — the body stays
            // disjunctive.
            Expr::Exists(bounds, body) => {
                bounds.iter().all(|b| {
                    b.domain.as_ref().is_some_and(|d| {
                        !crate::enumerate::expr_contains_any_prime(&d.node)
                            && !scan.reaches(ctx, &d.node)
                    })
                }) && walk(scan, ctx, body, target_ptr, depth + 1)
            }
            // Operator application.
            Expr::Apply(op_expr, args) => {
                let Expr::Ident(name, _) = &op_expr.node else {
                    // Non-Ident head: certified only if the subtree cannot
                    // descend-reach the target at all.
                    return !scan.reaches(ctx, &expr.node);
                };
                let resolved = ctx.resolve_op_name(name);
                let Some(def) = ctx.get_op(resolved) else {
                    return !scan.reaches(ctx, &expr.node);
                };
                let ptr = Arc::as_ptr(def) as usize;
                if ptr == target_ptr {
                    // A target application site in disjunctive context. It
                    // must provably take the CALL-BY-VALUE path (the exact
                    // condition the dispatcher's Apply arm negates for its
                    // substitution path): the frame is then always pushed
                    // around the whole body enumeration, so any nested
                    // self-application deeper inside the body only re-notes
                    // fingerprints the live outer frame already captures. A
                    // substituted site pushes NO outer frame, and a nested
                    // self-application under an inner conjunct could then
                    // record a pruned set — fail closed. Argument
                    // expressions are evaluated — they must not smuggle
                    // further target references either.
                    !def.has_primed_param
                        && args.iter().all(|a| {
                            !crate::expr_visitor::expr_is_action_level_v(ctx, &a.node)
                                && !scan.reaches(ctx, &a.node)
                        })
                } else {
                    // Pass-through inlined body: enumerated in this same
                    // disjunctive context (call-by-value or substituted).
                    let def = Arc::clone(def);
                    args.iter().all(|a| !scan.reaches(ctx, &a.node))
                        && (!scan.reaches(ctx, &def.body.node)
                            || walk(scan, ctx, &def.body, target_ptr, depth + 1))
                }
            }
            // Zero-argument operator reference.
            Expr::Ident(name, _) => {
                let resolved = ctx.resolve_op_name(name);
                let Some(def) = ctx.get_op(resolved) else {
                    // Not an operator here: no frame can be pushed, but keep
                    // the shadow-name hazard fail-closed.
                    return !scan.name_matches(ctx, name);
                };
                let ptr = Arc::as_ptr(def) as usize;
                if ptr == target_ptr {
                    true // a zero-arg target site in disjunctive context
                } else {
                    let def = Arc::clone(def);
                    !scan.reaches(ctx, &def.body.node)
                        || walk(scan, ctx, &def.body, target_ptr, depth + 1)
                }
            }
            // Anything else (And, IF, LET, ModuleRef, CASE, ...): certified
            // only if the enumerator cannot descend-reach the target through
            // it — then whatever it emits is never attributed to the target's
            // frame and completeness is unaffected.
            _ => !scan.reaches(ctx, &expr.node),
        }
    }

    walk(&mut scan, ctx, next_body, target_ptr, 0)
}

/// Flatten an `And` spine into its conjuncts, in source order.
fn flatten_and_spine<'a>(expr: &'a Spanned<Expr>, out: &mut Vec<&'a Spanned<Expr>>) {
    if let Expr::And(a, b) = &expr.node {
        flatten_and_spine(a, out);
        flatten_and_spine(b, out);
    } else {
        out.push(expr);
    }
}

/// Is `conjunct` provably state-level in `ctx`: not action-level
/// (`expr_is_action_level_v` — no primes or `UNCHANGED`, looking THROUGH
/// operator references) and not an operator-reference guard that could hide
/// action-level content (`is_operator_reference_guard_unsafe_v`)?
fn conjunct_is_state_level(ctx: &EvalCtx, conjunct: &Spanned<Expr>) -> bool {
    !crate::expr_visitor::expr_is_action_level_v(ctx, &conjunct.node)
        && !crate::expr_visitor::is_operator_reference_guard_unsafe_v(ctx, &conjunct.node)
}

/// Depth cap on the guard-extraction descent (nested existentials /
/// INSTANCE bodies). Fail-closed: hitting the cap stops emitting, never
/// asserts anything.
const MAX_GUARD_EXTRACT_DEPTH: u32 = 4;

/// Emission scope for extracted guards: the classification context (root or
/// the resolved INSTANCE scope) plus the replay recipe consumption evaluates
/// under (`None` = the consumption context directly).
struct ExtractScope<'a> {
    ctx: &'a EvalCtx,
    replay: Option<Arc<InstanceGuardScope>>,
}

impl ExtractScope<'_> {
    fn entry(&self, expr: Spanned<Expr>) -> GuardEntry {
        GuardEntry {
            expr: Arc::new(expr),
            scope: self.replay.clone(),
            memo: None,
            const_val: None,
            share_group: u32::MAX,
        }
    }
}

/// Conservative "mentions any of these names" test: true when the name occurs
/// as an `Ident`/`StateVar` reference OR as ANY binding site (nested
/// quantifier/LET/lambda) inside `expr` (`expr_contains_ident_v` — the
/// over-approximation is fail-closed: a shadowing binder makes the expression
/// ineligible rather than mis-scoped).
fn mentions_any(expr: &Expr, names: &FxHashSet<String>) -> bool {
    !names.is_empty()
        && names
            .iter()
            .any(|n| tla_core::expr_contains_ident_v(expr, n))
}

/// #guard-exists: match the total comparison `x # {}` (either operand order)
/// for a forbidden binder name `x`.
fn match_binder_nonempty<'e>(
    conjunct: &'e Spanned<Expr>,
    forbidden: &FxHashSet<String>,
) -> Option<&'e str> {
    let Expr::Neq(a, b) = &conjunct.node else {
        return None;
    };
    let (x, other) = match (&a.node, &b.node) {
        (Expr::Ident(x, _), other) => (x, other),
        (other, Expr::Ident(x, _)) => (x, other),
        _ => return None,
    };
    if !forbidden.contains(x.as_str()) {
        return None;
    }
    match other {
        Expr::SetEnum(elems) if elems.is_empty() => Some(x.as_str()),
        _ => None,
    }
}

/// Extract the LEADING run of state-level guard conjuncts of an action
/// operator body (the sound absence-side plan).
///
/// A conjunct qualifies when it is provably state-level
/// ([`conjunct_is_state_level`]). Extraction walks the flattened `And` spine
/// in source order and STOPS at the first non-qualifying conjunct (later
/// guards may be evaluation-ordered after assignments), with three recognized
/// descents at the stopping conjunct — see [`walk_guard_spine`]:
///
///   - #guard-moduleref (Step A): an `Instance!Op` conjunct (including a body
///     that IS a bare `ModuleRef`) is resolved fail-closed through the same
///     AST resolver the liveness pinning proofs trust, and the RESOLVED
///     body's own derived guards are appended, each carrying the replayed
///     instance scope (see [`InstanceGuardScope`]).
///   - #guard-exists (case 1): a `\E x ∈ D : body` conjunct (including a
///     bare-`Exists` action, e.g. Allocator's `\E S ∈ SUBSET Resources :
///     Allocate(c, S)`) is descended with `x` FORBIDDEN — only `x`-free
///     derived guards escape (a necessary condition of the body under any
///     witness binding is a necessary condition of the existential).
///   - #guard-exists (case 2): when case 1 derives nothing, the leading
///     `x`-usable state-level run `W` of the body is re-wrapped as the
///     synthetic guard `\E x ∈ D : ∧W` (monotone weakening of the body).
///
/// Every emitted guard is a state-level NECESSARY CONDITION of the action
/// relation, so one clean FALSE proves the relation empty. Any other body
/// shape yields an empty set — fail closed, the leaf keeps full evaluation.
pub(crate) fn extract_state_guard_entries(ctx: &EvalCtx, body: &Spanned<Expr>) -> Vec<GuardEntry> {
    let mut conjuncts = Vec::new();
    flatten_and_spine(body, &mut conjuncts);
    if conjuncts.len() == 1 && !matches!(conjuncts[0].node, Expr::ModuleRef(..) | Expr::Exists(..))
    {
        // A bare (non-And) body with no recognized descent is either a single
        // guard — in which case the action can produce no successor at all
        // and full evaluation is already trivial — or a single action
        // conjunct. Nothing to gain. (A bare ModuleRef body — e.g. a property
        // plan's `Sched!Schedule` leaf — and a bare Exists — the Allocator's
        // `\E S : ...` fairness actions — ARE worth descending into.)
        return Vec::new();
    }
    let scope = ExtractScope { ctx, replay: None };
    let mut entries = Vec::new();
    walk_guard_spine(&scope, &conjuncts, &FxHashSet::default(), 0, &mut entries);
    entries
}

/// Core spine walk shared by the top-level extraction, the #guard-moduleref
/// resolved-body descent, and the #guard-exists body descent.
///
/// Walks the conjuncts in canonical (source) order, emitting guards into
/// `out` in that same order. `forbidden` holds names bound by ENCLOSING
/// existential binders (or instance formals standing in for them) — a guard
/// may only be emitted if it provably cannot reference them
/// ([`mentions_any`], binding-site-inclusive over-approximation).
///
/// # Error-exactness invariant
///
/// The consumption walk (`guard_prefix_refutes`) evaluates emitted guards in
/// order and treats ANY evaluation error as "no claim" (full evaluation then
/// surfaces the canonical error). To keep refutation decisions bit-consistent
/// with the canonical enumeration on erroring states too, every conjunct this
/// walk SKIPS (rather than stops at) must have its potential evaluation
/// errors covered: either the conjunct is total given bound values (`x # {}`
/// — a value comparison against a literal), or an earlier-emitted guard
/// evaluates the same failure-relevant subexpression (the pair-inference
/// guard `E # {}` evaluates exactly the `E` of the skipped `x ⊆ E`). Any
/// forbidden-referencing conjunct outside these recognized shapes stops the
/// walk — nothing after it is emitted.
fn walk_guard_spine(
    scope: &ExtractScope<'_>,
    conjuncts: &[&Spanned<Expr>],
    forbidden: &FxHashSet<String>,
    depth: u32,
    out: &mut Vec<GuardEntry>,
) {
    if depth >= MAX_GUARD_EXTRACT_DEPTH {
        return;
    }
    let quant = !quant_guards_disabled();
    // #guard-exists pair inference: forbidden names x whose `x # {}`
    // conjunct has been seen on this spine.
    let mut nonempty_pending: FxHashSet<String> = FxHashSet::default();
    for conjunct in conjuncts {
        if conjunct_is_state_level(scope.ctx, conjunct) {
            if !mentions_any(&conjunct.node, forbidden) {
                out.push(scope.entry((*conjunct).clone()));
                continue;
            }
            if quant {
                // `x # {}` for a forbidden x: total (a pure value comparison
                // against a literal — the canonical enumeration cannot error
                // here), recorded for the pair inference below.
                if let Some(x) = match_binder_nonempty(conjunct, forbidden) {
                    nonempty_pending.insert(x.to_string());
                    continue;
                }
                // `x \subseteq E` with `x # {}` already seen and E
                // forbidden-free: ∃x: x # {} ∧ x ⊆ E ⇒ E # {}, so `E # {}`
                // is a state-level necessary condition — emit it. A clean
                // FALSE (E = {}) kills every canonical x # {} survivor AT
                // THIS conjunct; an error in E is exactly the error the
                // canonical evaluation of this conjunct would report (and
                // the consumption walk turns it into "no claim").
                if let Expr::Subseteq(l, r) = &conjunct.node {
                    if let Expr::Ident(x, _) = &l.node {
                        if nonempty_pending.contains(x.as_str())
                            && !mentions_any(&r.node, forbidden)
                        {
                            out.push(scope.entry(Spanned::dummy(Expr::Neq(
                                Box::new(r.as_ref().clone()),
                                Box::new(Spanned::dummy(Expr::SetEnum(Vec::new()))),
                            ))));
                            continue;
                        }
                    }
                }
            }
            // Unrecognized forbidden-referencing guard: skipping it could
            // hide a canonical error source, so the derived walk stops here.
            return;
        }
        // First non-state-level conjunct: one recognized descent, then stop.
        match &conjunct.node {
            Expr::ModuleRef(..) => {
                module_ref_guard_entries(scope, conjunct, forbidden, depth, out);
            }
            Expr::Exists(bounds, body) if quant => {
                exists_guard_entries(scope, bounds, body, forbidden, depth, out);
            }
            _ => {}
        }
        return;
    }
}

/// #guard-exists: derived guards for a `\E x1 ∈ D1, ... : body` conjunct.
///
/// Case 1 (descent): recurse into `body`'s spine with the binder names added
/// to `forbidden`. Every guard emitted there is binder-free, and a necessary
/// condition of `body` under ANY binding is a necessary condition of the
/// existential — sound for any domain (a witness binding must satisfy it).
/// Error-surface note: the derived guards do not evaluate the DOMAIN
/// expressions, so on a state where the canonical enumeration's domain
/// evaluation itself would error, a refutation here masks that error into
/// `ENABLED = false`. This can only occur for an action whose quantifier
/// domain errors at a reachable state — a spec that already fails BFS
/// expansion whenever the action (or its Next twin) is reachable; on every
/// error-free spec the decision is exact.
///
/// Case 2 (wrap), only when case 1 derived nothing: take the leading run `W`
/// of `body`'s spine truncated at the first conjunct that is action-level or
/// references `forbidden` (outer binders — THIS existential's binders are
/// allowed, they are re-bound by the wrap), and emit the synthetic guard
/// `\E x ∈ D : ∧W`. Monotone weakening: the full body implies `∧W` pointwise,
/// so the wrapped existential is a necessary condition; canonical branches
/// die at exactly the conjuncts `W` re-evaluates (order preserved), so a
/// clean FALSE is exact and any error surfaces in the guard itself (no
/// claim). Wrap gates (fail closed): simple named binders (no patterns),
/// every domain present, state-level, and forbidden-free.
fn exists_guard_entries(
    scope: &ExtractScope<'_>,
    bounds: &[tla_core::ast::BoundVar],
    body: &Spanned<Expr>,
    forbidden: &FxHashSet<String>,
    depth: u32,
    out: &mut Vec<GuardEntry>,
) {
    // Binder bookkeeping (both cases): simple named binders only, no
    // duplicates, no collision with the enclosing forbidden set (a collision
    // would make the name tracking ambiguous — fail closed).
    let mut binder_names: Vec<&str> = Vec::with_capacity(bounds.len());
    for bv in bounds {
        if bv.pattern.is_some() || bv.domain.is_none() {
            return;
        }
        let name = bv.name.node.as_str();
        if forbidden.contains(name) || binder_names.contains(&name) {
            return;
        }
        binder_names.push(name);
    }
    let mut body_spine = Vec::new();
    flatten_and_spine(body, &mut body_spine);

    // Case 1: descend with the binders forbidden.
    let mut inner_forbidden = forbidden.clone();
    for name in &binder_names {
        inner_forbidden.insert((*name).to_string());
    }
    let before = out.len();
    walk_guard_spine(scope, &body_spine, &inner_forbidden, depth + 1, out);
    if out.len() > before {
        return;
    }

    // Case 2: wrap the leading binder-usable state-level run.
    for bv in bounds {
        let domain = bv.domain.as_ref().expect("checked above");
        if !conjunct_is_state_level(scope.ctx, domain) || mentions_any(&domain.node, forbidden) {
            return;
        }
    }
    let mut wrap: Vec<&Spanned<Expr>> = Vec::new();
    for c in &body_spine {
        if !conjunct_is_state_level(scope.ctx, c) || mentions_any(&c.node, forbidden) {
            break;
        }
        wrap.push(c);
    }
    if wrap.is_empty() {
        return;
    }
    let mut wrapped: Spanned<Expr> = (*wrap[wrap.len() - 1]).clone();
    for c in wrap.iter().rev().skip(1) {
        wrapped = Spanned::dummy(Expr::And(Box::new((*c).clone()), Box::new(wrapped)));
    }
    out.push(scope.entry(Spanned::dummy(Expr::Exists(
        bounds.to_vec(),
        Box::new(wrapped),
    ))));
}

/// #guard-moduleref (Step A): lift derived guards out of a resolved
/// `Instance!Op` conjunct, each paired with the replayed instance scope it
/// must be evaluated under (see [`InstanceGuardScope`] for the
/// replay-identity argument).
///
/// Formal parameters are classified against `forbidden` (#guard-exists
/// integration): a formal whose argument expression mentions a forbidden
/// binder becomes FORBIDDEN inside the resolved body (its value varies with
/// the existential witness); every other argument must be provably
/// state-level (it is re-evaluated per state at consumption) and its formal
/// becomes evaluable through the scope's `formals`.
///
/// Fail-closed gates (each derives NO guards, keeping full evaluation):
///   - the surrounding scope is already a replayed instance scope (one
///     replay level only) or the build context carries an active local-op
///     scope / INSTANCE substitutions (the replay-identity argument needs
///     both pure);
///   - the target is not a plain `Named` instance;
///   - a non-forbidden argument expression is not provably state-level;
///   - the reference does not fully resolve
///     (`resolve_named_module_ref_body_ast_with_params` — unknown instance,
///     op-not-found, arity mismatch, substitution-only fallback);
///   - duplicate formal parameter names (binding-order ambiguity).
fn module_ref_guard_entries(
    scope: &ExtractScope<'_>,
    conjunct: &Spanned<Expr>,
    forbidden: &FxHashSet<String>,
    depth: u32,
    out: &mut Vec<GuardEntry>,
) {
    if depth >= MAX_GUARD_EXTRACT_DEPTH {
        return;
    }
    let ctx = scope.ctx;
    let Expr::ModuleRef(target, op_name, args) = &conjunct.node else {
        return;
    };
    if scope.replay.is_some() || ctx.local_ops().is_some() || ctx.instance_substitutions().is_some()
    {
        return;
    }
    let ModuleTarget::Named(instance_name) = target else {
        return;
    };
    let Some((instance_ctx, body, params)) =
        crate::enabled::resolve_named_module_ref_body_ast_with_params(
            ctx,
            instance_name,
            op_name,
            args,
        )
    else {
        return;
    };
    // Duplicate formal names would make binding order load-bearing; fail closed.
    {
        let mut names: Vec<&str> = params.iter().map(|p| p.as_ref()).collect();
        names.sort_unstable();
        if names.windows(2).any(|w| w[0] == w[1]) {
            return;
        }
    }
    // Classify formals: forbidden-referencing args make the FORMAL forbidden
    // inside the resolved body; the rest must be state-level and become
    // consumption-time bindings.
    let mut formals: Vec<(Arc<str>, Arc<Spanned<Expr>>)> = Vec::new();
    let mut inner_forbidden: FxHashSet<String> = FxHashSet::default();
    for (param, arg) in params.iter().zip(args.iter()) {
        if mentions_any(&arg.node, forbidden) {
            inner_forbidden.insert(param.to_string());
        } else if conjunct_is_state_level(ctx, arg) {
            formals.push((Arc::clone(param), Arc::new(arg.clone())));
        } else {
            return; // action-level argument: fail closed
        }
    }
    // The resolver builds the instance scope via `with_module_scope_arced_subs`,
    // which always installs a local-op map; a missing map means the resolution
    // took an unexpected shape — fail closed.
    let Some(local_ops) = instance_ctx.local_ops().clone() else {
        return;
    };
    let subs: Arc<Vec<Substitution>> = Arc::new(
        instance_ctx
            .instance_substitutions()
            .map(<[Substitution]>::to_vec)
            .unwrap_or_default(),
    );
    let replay = Arc::new(InstanceGuardScope {
        local_ops,
        subs,
        formals,
        resolved_from: (instance_name.clone(), op_name.clone()),
    });
    // Derived guards of the RESOLVED (substitution-applied) body, classified
    // in the INSTANCE scope so module-local operators (e.g.
    // SchedulingAllocator's `toSchedule`) resolve for the safety visitors.
    let mut body_spine = Vec::new();
    flatten_and_spine(&body, &mut body_spine);
    let inner_scope = ExtractScope {
        ctx: &instance_ctx,
        replay: Some(replay),
    };
    walk_guard_spine(&inner_scope, &body_spine, &inner_forbidden, depth + 1, out);
}

/// #guard-in-memo (Step B): is `expr`'s value a pure function of the plan's
/// binding CONSTANTS? Strict whitelist walk — every identifier leaf must be a
/// binding name, and only structural/comparison/access forms that cannot
/// reach state variables, operators, primes, or module references are
/// admitted. Anything else fails closed (the guard keeps its canonical
/// per-state evaluation).
fn expr_is_binding_const(expr: &Expr, binding_names: &FxHashSet<&str>) -> bool {
    match expr {
        Expr::Bool(_) | Expr::Int(_) | Expr::String(_) => true,
        Expr::Ident(name, _) => binding_names.contains(name.as_str()),
        Expr::RecordAccess(base, _) => expr_is_binding_const(&base.node, binding_names),
        Expr::Eq(a, b)
        | Expr::Neq(a, b)
        | Expr::In(a, b)
        | Expr::NotIn(a, b)
        | Expr::FuncApply(a, b) => {
            expr_is_binding_const(&a.node, binding_names)
                && expr_is_binding_const(&b.node, binding_names)
        }
        Expr::Tuple(elems) | Expr::SetEnum(elems) => elems
            .iter()
            .all(|e| expr_is_binding_const(&e.node, binding_names)),
        _ => false,
    }
}

/// #guard-in-memo (Step B): evaluate a binding-const guard ONCE at plan-build
/// time, under exactly the plan bindings the per-state evaluation would push.
/// By `expr_is_binding_const` the expression can reach nothing but those
/// constants, so the build-time value equals its value in EVERY state.
/// `None` (eval error / non-boolean) = no claim, keep the canonical per-state
/// evaluation.
fn eval_binding_const_guard(
    ctx: &EvalCtx,
    bindings: &[(Arc<str>, tla_core::name_intern::NameId, Value)],
    expr: &Spanned<Expr>,
) -> Option<bool> {
    let mut build_ctx = ctx.clone();
    for (name, id, value) in bindings {
        build_ctx.push_binding_preinterned(Arc::clone(name), value.clone(), *id);
    }
    match super::eval_live_entry(&build_ctx, expr) {
        Ok(v) => v.as_bool(),
        Err(_) => None,
    }
}

/// #guard-in-memo (Step B): detect the `<binding-const> \in <bare-state-var>`
/// shape (see [`MemoInGuard`]). Fail-closed gates: the LHS must be a bare
/// identifier naming one of the plan's binding constants; the RHS must be a
/// bare identifier/state-var reference that is registered as a STATE VARIABLE,
/// is not shadowed by any plan binding, and does not resolve to an operator
/// (an operator RHS could route `eval_in` through the lazy-membership paths).
fn detect_memo_in_guard(
    ctx: &EvalCtx,
    bindings: &[(Arc<str>, tla_core::name_intern::NameId, Value)],
    expr: &Spanned<Expr>,
) -> Option<MemoInGuard> {
    let Expr::In(lhs, rhs) = &expr.node else {
        return None;
    };
    let Expr::Ident(lhs_name, _) = &lhs.node else {
        return None;
    };
    let lhs_val = bindings
        .iter()
        .find(|(n, _, _)| n.as_ref() == lhs_name.as_str())?
        .2
        .clone();
    let rhs_name = match &rhs.node {
        Expr::Ident(n, _) | Expr::StateVar(n, _, _) => n.as_str(),
        _ => return None,
    };
    if bindings.iter().any(|(n, _, _)| n.as_ref() == rhs_name) {
        return None;
    }
    if ctx.var_registry().get(rhs_name).is_none() {
        return None;
    }
    if ctx.get_op(ctx.resolve_op_name(rhs_name)).is_some() {
        return None;
    }
    Some(MemoInGuard {
        lhs: lhs_val,
        rhs: Arc::new(rhs.as_ref().clone()),
        group: u32::MAX,
    })
}

/// Build absence-side guard plans for a set of liveness leaves: for every
/// `LiveExpr::Enabled` leaf, extract the state-level guard prefix of its
/// RESOLVED action expression and pair it with the leaf's own (fully
/// eagerly-observable) quantifier bindings — the same chain
/// `eval_enabled_uncached` binds before enumerating, so a guard evaluated
/// under these bindings is a necessary condition of exactly the relation the
/// full evaluator would decide. Fail-closed: leaves with no provable guard
/// prefix, a lazy binding (`all_bindings_eager` = None), or duplicate binding
/// names (shadowing-order ambiguity) get no plan.
pub(crate) fn build_enabled_guard_plans<'a>(
    ctx: &EvalCtx,
    leaves: impl Iterator<Item = &'a super::LiveExpr>,
) -> Vec<GuardPlan> {
    // #guard-in-memo (Step B) refinements are kill-switched separately
    // (TY_DISABLE_ENABLED_GUARD_MEMO=1): when disabled, every guard keeps the
    // plain per-leaf evaluation of the pre-Step-B plans.
    let refine = !guard_memo_disabled();
    let mut plans = Vec::new();
    for leaf in leaves {
        let super::LiveExpr::Enabled {
            action,
            bindings,
            tag,
            ..
        } = leaf
        else {
            continue;
        };
        let entries = extract_state_guard_entries(ctx, action);
        if entries.is_empty() {
            continue;
        }
        let chain: Vec<(Arc<str>, tla_core::name_intern::NameId, Value)> = match bindings {
            Some(chain) => match chain.all_bindings_eager() {
                Some(pairs) => pairs
                    .into_iter()
                    .map(|(n, v)| {
                        let id = tla_core::name_intern::intern_name(n.as_ref());
                        (n, id, v)
                    })
                    .collect(),
                None => continue, // lazy binding: fail closed
            },
            None => Vec::new(),
        };
        // Duplicate binding names would make push-order shadowing semantics
        // load-bearing; fail closed instead.
        let mut names: Vec<&str> = chain.iter().map(|(n, _, _)| n.as_ref()).collect();
        names.sort_unstable();
        if names.windows(2).any(|w| w[0] == w[1]) {
            continue;
        }
        let binding_names: FxHashSet<&str> = chain.iter().map(|(n, _, _)| n.as_ref()).collect();
        let mut guards = Vec::with_capacity(entries.len());
        for entry in entries {
            if refine && entry.scope.is_none() {
                // Binding-const guard: pre-evaluate once. TRUE = a proven
                // conjunct — dropped (it never refutes and, being const,
                // re-evaluating it per state could neither error nor change).
                // FALSE = an always-false necessary condition — kept as a
                // const entry so that REACHING it in evaluation order refutes
                // without an eval (order is preserved: an earlier guard that
                // errors still stops the walk first, exactly like the
                // canonical enumeration would surface that error).
                if expr_is_binding_const(&entry.expr.node, &binding_names) {
                    match eval_binding_const_guard(ctx, &chain, &entry.expr) {
                        Some(true) => continue,
                        Some(false) => {
                            guards.push(GuardEntry {
                                const_val: Some(false),
                                ..entry
                            });
                            continue;
                        }
                        None => {}
                    }
                }
                if let Some(memo) = detect_memo_in_guard(ctx, &chain, &entry.expr) {
                    guards.push(GuardEntry {
                        memo: Some(memo),
                        ..entry
                    });
                    continue;
                }
            }
            guards.push(entry);
        }
        if guards.is_empty() {
            continue;
        }
        plans.push(GuardPlan {
            enabled_tag: *tag,
            guards,
            bindings: chain,
        });
    }
    plans
}

/// #guard-in-memo (Step B): fetch (or evaluate-and-store) the per-state value
/// of a memoized RHS. One slot per sharing group, keyed by the state
/// fingerprint — a mismatching fingerprint is a miss (the previous state's
/// value can never leak). `None` = RHS evaluation failed; the caller makes no
/// claim (and the canonical evaluation of the full guard would surface the
/// same error).
fn memo_rhs_value(ctx: &EvalCtx, fp: Fingerprint, memo: &MemoInGuard) -> Option<Value> {
    let cached = PROV.with(|p| {
        p.borrow()
            .memo_vals
            .get(&memo.group)
            .and_then(|(f, v)| (*f == fp).then(|| v.clone()))
    });
    if cached.is_some() {
        return cached;
    }
    let v = super::eval_live_entry(ctx, &memo.rhs).ok()?;
    PROV.with(|p| {
        p.borrow_mut().memo_vals.insert(memo.group, (fp, v.clone()));
    });
    Some(v)
}

/// #guard-moduleref (Step A): build the scoped evaluation context for the
/// instance-scoped guards of one [`InstanceGuardScope`] by replaying the
/// stored module scope onto the consumption context (see
/// [`InstanceGuardScope`] for the replay-identity argument). `None` = no
/// claim (impure consumption context, or an argument failed to evaluate);
/// the caller stops the guard walk and full evaluation decides.
///
/// The result is reusable across every guard of the SAME scope within one
/// `guard_prefix_refutes` walk (same state, same plan bindings — the formal
/// evaluations are deterministic and `eval_live_entry` takes the context
/// immutably), so consecutive same-scope guards build it once.
fn build_scoped_guard_ctx(ctx: &EvalCtx, scope: &InstanceGuardScope) -> Option<EvalCtx> {
    // Consumption-side purity gate (the build side was gated identically in
    // `module_ref_guard_entries`): the replayed scope is only provably the
    // scope a fresh resolution would build when the context it is layered
    // onto carries no local-op scope / INSTANCE substitutions of its own.
    if ctx.local_ops().is_some() || ctx.instance_substitutions().is_some() {
        return None;
    }
    // Bind the resolved operator's formals to the caller-side argument
    // values, evaluated in the consumption context (plan bindings pushed) —
    // the same binding the evaluating resolver performs. Args were verified
    // state-level at build; an eval error here is no claim.
    let mut formal_binds: Vec<(Arc<str>, Value)> = Vec::with_capacity(scope.formals.len());
    for (name, arg) in &scope.formals {
        match super::eval_live_entry(ctx, arg) {
            Ok(v) => formal_binds.push((Arc::clone(name), v)),
            Err(_) => return None,
        }
    }
    Some(ctx.with_module_scope_arced_subs(
        Arc::clone(&scope.local_ops),
        formal_binds,
        Arc::clone(&scope.subs),
    ))
}

/// SOUND absence side: `true` iff the leaf's registered state-level guard
/// prefix REFUTES the action in `ctx` (bound to the current state, whose
/// fingerprint is `fp` — the same `(fp, ctx)` pairing the surrounding
/// `(fp, tag)` ENABLED cache already relies on).
///
/// Soundness: each registered guard is a prime-free, operator-safe CONJUNCT of
/// the leaf's action operator body, evaluated with the operator's formal
/// parameters bound to the leaf's const-level argument values — i.e. a
/// state-level necessary condition of the action relation. A guard evaluating
/// to `FALSE` therefore proves `¬∃t: A(s, t)`, hence `ENABLED <<A>>_e = false`
/// for ANY subscript `e`. Everything else — a guard evaluating to `TRUE`, a
/// non-boolean value, or an evaluation error — makes NO claim: the caller
/// falls through to the full evaluation, which computes the canonical result
/// (or reports the canonical error). The evaluation itself uses the same
/// evaluator (`eval_live_entry`) the full path uses, so a `FALSE` here is
/// bit-consistent with the enumeration's own guard check.
///
/// Guard forms (walked strictly in registration order, preserving the
/// canonical error surface — a guard that would error stops the walk before
/// any later guard is consulted):
///   - plain: evaluated directly in `ctx` (the original path);
///   - `const_val == Some(false)` (#guard-in-memo): a binding-const conjunct
///     pre-proven FALSE at build — reaching it refutes with no evaluation;
///   - memoized `x \in v` (#guard-in-memo): decided by the per-state shared
///     RHS value + `set_contains` (the exact routine `eval_in` bottoms out
///     in); an indeterminate membership falls through to the plain path;
///   - instance-scoped (#guard-moduleref): evaluated under the replayed
///     INSTANCE scope (see [`eval_instance_scoped_guard`]).
pub(crate) fn guard_prefix_refutes(ctx: &mut EvalCtx, fp: Fingerprint, tag: u32) -> bool {
    let Some(plan) = PROV.with(|p| p.borrow().guard_plans.get(&tag).cloned()) else {
        return false;
    };
    let mark = ctx.mark_stack();
    for (name, id, value) in &plan.bindings {
        ctx.push_binding_preinterned(Arc::clone(name), value.clone(), *id);
    }
    let mut refuted = false;
    let mut by_memo = false;
    let mut by_const = false;
    // Per-walk cache of the scoped context for consecutive guards sharing one
    // InstanceGuardScope (keyed by Arc pointer identity).
    let mut scoped_cache: Option<(usize, EvalCtx)> = None;
    'guards: for entry in &plan.guards {
        // Binding-const guard pre-proven FALSE at build: refute on reach.
        if entry.const_val == Some(false) {
            refuted = true;
            by_const = true;
            break;
        }
        // #guard-share: a twin leaf already evaluated this exact guard
        // (same expression, bindings, and scope recipe) for this state.
        if entry.share_group != u32::MAX {
            let shared = PROV.with(|p| {
                p.borrow()
                    .guard_outcomes
                    .get(&entry.share_group)
                    .and_then(|(f, v)| (*f == fp).then_some(*v))
            });
            if let Some(v) = shared {
                PROV.with(|p| p.borrow_mut().guard_share_hits += 1);
                if v {
                    continue;
                }
                refuted = true;
                break;
            }
        }
        // Memoized `x \in v` guard: shared per-state RHS + set_contains.
        if let Some(memo) = &entry.memo {
            match memo_rhs_value(&*ctx, fp, memo) {
                Some(rhs_val) => match rhs_val.set_contains(&memo.lhs) {
                    Some(false) => {
                        refuted = true;
                        by_memo = true;
                        break;
                    }
                    Some(true) => continue,
                    // Indeterminate membership (e.g. a lazy set shape):
                    // the canonical evaluation below decides this guard.
                    None => {}
                },
                None => break, // RHS eval error: no claim, full eval decides
            }
        }
        let result = match &entry.scope {
            None => super::eval_live_entry(&*ctx, &entry.expr),
            Some(scope) => {
                let ptr = Arc::as_ptr(scope) as usize;
                if !scoped_cache.as_ref().is_some_and(|(p, _)| *p == ptr) {
                    match build_scoped_guard_ctx(&*ctx, scope) {
                        Some(scoped) => scoped_cache = Some((ptr, scoped)),
                        None => break 'guards, // no claim, full eval decides
                    }
                }
                let (_, scoped) = scoped_cache.as_ref().expect("just set");
                super::eval_live_entry(scoped, &entry.expr)
            }
        };
        match result {
            Ok(v) => match v.as_bool() {
                Some(value) => {
                    if entry.share_group != u32::MAX {
                        PROV.with(|p| {
                            p.borrow_mut()
                                .guard_outcomes
                                .insert(entry.share_group, (fp, value));
                        });
                    }
                    if !value {
                        refuted = true;
                        break;
                    }
                }
                None => break, // non-boolean: no claim, full eval decides
            },
            Err(_) => break, // eval error: no claim, full eval decides
        }
    }
    ctx.pop_to_mark(&mark);
    if refuted {
        PROV.with(|p| {
            let mut st = p.borrow_mut();
            st.guard_refutes += 1;
            if by_memo {
                st.guard_memo_refutes += 1;
            }
            if by_const {
                st.guard_const_refutes += 1;
            }
        });
    }
    refuted
}

/// Diagnostics: count one full (provenance-miss) inline ENABLED evaluation by
/// outcome. Purely informational — printed by the liveness-profile stats.
#[inline]
pub(crate) fn note_full_eval(outcome: bool) {
    PROV.with(|p| {
        let mut st = p.borrow_mut();
        if outcome {
            st.full_true += 1;
        } else {
            st.full_false += 1;
        }
    });
}

/// Diagnostics for the liveness-profile stats printer:
/// `(registered_leaves, consumption_hits, guard_refutes, guard_memo_refutes,
/// guard_const_refutes, guard_share_hits, full_evals_true, full_evals_false)`.
/// The memo/const counts are subsets of `guard_refutes`.
pub(crate) fn stats() -> (usize, u64, u64, u64, u64, u64, u64, u64) {
    PROV.with(|p| {
        let st = p.borrow();
        (
            st.registered,
            st.hits,
            st.guard_refutes,
            st.guard_memo_refutes,
            st.guard_const_refutes,
            st.guard_share_hits,
            st.full_true,
            st.full_false,
        )
    })
}

/// Diagnostics for the liveness-profile stats printer: #frame-fp-pop
/// `(consumption hits, fallbacks-to-re-enumeration)`.
pub(crate) fn frame_pop_stats() -> (u64, u64) {
    PROV.with(|p| {
        let st = p.borrow();
        (st.frame_pop_hits, st.frame_pop_fallbacks)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fp(n: u64) -> Fingerprint {
        Fingerprint(n)
    }

    fn leaf(
        def_ptr: usize,
        args: Vec<Value>,
        tag: u32,
        needs_change: bool,
    ) -> RegisteredEnabledLeaf {
        RegisteredEnabledLeaf {
            def_ptr,
            args: args.into_iter().map(ArgPattern::Exact).collect(),
            enabled_tag: tag,
            needs_change,
        }
    }

    fn retained_capacity() -> usize {
        PROV.with(|state| {
            let state = state.borrow();
            state.reg.capacity()
                + state.frame_false_tags.capacity()
                + state.frame_fps.capacity()
                + state.guard_plans.capacity()
                + state.memo_groups.capacity()
                + state.memo_vals.capacity()
                + state.share_groups.capacity()
                + state.guard_outcomes.capacity()
                + state.witnessed.capacity()
                + state.frames.capacity()
        })
    }

    #[test]
    fn release_enabled_provenance_storage_drops_all_capacity() {
        release_enabled_provenance_storage();
        PROV.with(|state| {
            let mut state = state.borrow_mut();
            state.reg.reserve(64);
            state.frame_false_tags.reserve(64);
            state.frame_fps.reserve(64);
            state.guard_plans.reserve(64);
            state.memo_groups.reserve(64);
            state.memo_vals.reserve(64);
            state.share_groups.reserve(64);
            state.guard_outcomes.reserve(64);
            state.witnessed.reserve(64);
            state.frames.reserve(64);
            state.arm_enum_incomplete = true;
            state.armed = true;
            state.enum_depth = 2;
            state.suppress = 1;
            state.scratch_fp = Some(fp(9));
            state.hits = 1;
            state.registered = 1;
        });
        PROV_HOT.with(|hot| hot.set(true));
        assert!(retained_capacity() > 0);

        release_enabled_provenance_storage();

        assert_eq!(retained_capacity(), 0);
        PROV.with(|state| {
            let state = state.borrow();
            assert!(!state.arm_enum_incomplete);
            assert!(!state.armed);
            assert_eq!(state.enum_depth, 0);
            assert_eq!(state.suppress, 0);
            assert_eq!(state.scratch_fp, None);
            assert_eq!(state.hits, 0);
            assert_eq!(state.registered, 0);
        });
        assert!(!PROV_HOT.with(Cell::get));
    }

    /// TRUE-provenance positive: a state-changing emission through a matching
    /// frame witnesses the registered tag for the armed state.
    #[test]
    fn witness_recorded_for_matching_frame() {
        register(vec![leaf(0x1000, vec![Value::int(1)], 7, true)]);
        {
            let _arm = arm_state_guard(fp(42));
            let _scope = enum_scope();
            let _frame = push_frame(0x1000, &[Value::int(1)]);
            note_emission(Some(fp(42)), || true, || None);
        }
        assert!(witnessed_true(fp(42), 7));
        // Wrong fingerprint: no claim.
        assert!(!witnessed_true(fp(43), 7));
        clear();
    }

    /// Provenance-silent: an unmatched binding value records nothing — the
    /// leaf falls back to full evaluation.
    #[test]
    fn silent_for_unmatched_binding() {
        register(vec![leaf(0x1000, vec![Value::int(1)], 7, true)]);
        {
            let _arm = arm_state_guard(fp(42));
            let _scope = enum_scope();
            let _frame = push_frame(0x1000, &[Value::int(2)]); // different arg value
            note_emission(Some(fp(42)), || true, || None);
        }
        assert!(!witnessed_true(fp(42), 7));
        clear();
    }

    /// A non-state-changing emission is NOT a witness for a
    /// `require_state_change` leaf, but IS one for a plain-ENABLED leaf.
    #[test]
    fn state_change_gate() {
        register(vec![
            leaf(0x1000, vec![], 7, true),
            leaf(0x1000, vec![], 8, false),
        ]);
        {
            let _arm = arm_state_guard(fp(42));
            let _scope = enum_scope();
            let _frame = push_frame(0x1000, &[]);
            note_emission(Some(fp(42)), || false, || None); // stuttering emission
        }
        assert!(!witnessed_true(fp(42), 7));
        assert!(witnessed_true(fp(42), 8));
        clear();
    }

    // ── #frame-fp-pop unit tests ─────────────────────────────────────────

    /// Shared scaffold: register one FALSE-eligible zero-arg leaf (tag 7,
    /// pair 99), arm fp(42), and run `body` inside a marked-complete armed
    /// depth-1 enumeration scope with the frame pushed.
    fn frame_fp_run(complete: bool, body: impl FnOnce()) {
        register(vec![leaf(0x1000, vec![], 7, false)]);
        register_frame_false_tags(vec![7]);
        let _arm = arm_state_guard(fp(42));
        {
            let mut scope = enum_scope();
            {
                let _frame = push_frame(0x1000, &[]);
                body();
            }
            if complete {
                scope.mark_complete();
            }
        }
    }

    /// Positive TRUE + FALSE population: recorded fps decide membership for
    /// every cached successor, absence is a sound FALSE under the full gate
    /// stack (certified tag + pushed frame + complete arm + clean record).
    #[test]
    fn frame_fp_population_true_and_false() {
        super::super::checker::clear_scan_pred_results();
        frame_fp_run(true, || {
            note_emission(Some(fp(42)), || true, || Some(fp(100)));
            note_emission(Some(fp(42)), || true, || Some(fp(101)));
        });
        assert!(populate_pair_from_frame_fps(
            fp(42),
            7,
            99,
            [fp(100), fp(102)].into_iter()
        ));
        assert_eq!(
            super::super::checker::get_scan_pred_result(fp(42), fp(100), 99),
            Some(true)
        );
        assert_eq!(
            super::super::checker::get_scan_pred_result(fp(42), fp(102), 99),
            Some(false)
        );
        // Wrong state fingerprint: no claim.
        assert!(!populate_pair_from_frame_fps(
            fp(43),
            7,
            99,
            [fp(100)].into_iter()
        ));
        super::super::checker::clear_scan_pred_results();
        clear();
    }

    /// A pushed frame with ZERO emissions and a clean, complete arm proves
    /// the action relation empty from this state — every pair gets FALSE.
    #[test]
    fn frame_fp_empty_record_is_all_false() {
        super::super::checker::clear_scan_pred_results();
        frame_fp_run(true, || {});
        assert!(populate_pair_from_frame_fps(
            fp(42),
            7,
            99,
            [fp(100)].into_iter()
        ));
        assert_eq!(
            super::super::checker::get_scan_pred_result(fp(42), fp(100), 99),
            Some(false)
        );
        super::super::checker::clear_scan_pred_results();
        clear();
    }

    /// FALSE-only-under-proof: a tag NOT registered FALSE-eligible never
    /// populates from frame fps (no record is even opened) — the caller falls
    /// back to the landed re-enumeration.
    #[test]
    fn frame_fp_requires_false_eligibility() {
        super::super::checker::clear_scan_pred_results();
        register(vec![leaf(0x1000, vec![], 7, false)]);
        // No register_frame_false_tags.
        {
            let _arm = arm_state_guard(fp(42));
            let mut scope = enum_scope();
            {
                let _frame = push_frame(0x1000, &[]);
                note_emission(Some(fp(42)), || true, || Some(fp(100)));
            }
            scope.mark_complete();
        }
        assert!(!populate_pair_from_frame_fps(
            fp(42),
            7,
            99,
            [fp(100)].into_iter()
        ));
        assert_eq!(
            super::super::checker::get_scan_pred_result(fp(42), fp(100), 99),
            None
        );
        clear();
    }

    /// An armed depth-1 enumeration that ends WITHOUT the completion mark
    /// (sink Break / error unwind / non-participating entry point) poisons
    /// the whole arm: no FALSE population, fall back.
    #[test]
    fn frame_fp_incomplete_enumeration_blocks_population() {
        super::super::checker::clear_scan_pred_results();
        frame_fp_run(false, || {
            note_emission(Some(fp(42)), || true, || Some(fp(100)));
        });
        assert!(!populate_pair_from_frame_fps(
            fp(42),
            7,
            99,
            [fp(100)].into_iter()
        ));
        clear();
    }

    /// An emission whose fingerprint is unavailable poisons the record
    /// fail-closed (the set can no longer be proven exact).
    #[test]
    fn frame_fp_unavailable_fp_blocks_population() {
        super::super::checker::clear_scan_pred_results();
        frame_fp_run(true, || {
            note_emission(Some(fp(42)), || true, || Some(fp(100)));
            note_emission(Some(fp(42)), || true, || None); // fp unavailable
        });
        assert!(!populate_pair_from_frame_fps(
            fp(42),
            7,
            99,
            [fp(100)].into_iter()
        ));
        clear();
    }

    /// Unaudited-path emissions (the state-independent Or-branch replay
    /// cache) poison active frame records — no FALSE from a set that missed
    /// them.
    #[test]
    fn frame_fp_unattributed_emission_blocks_population() {
        super::super::checker::clear_scan_pred_results();
        frame_fp_run(true, || {
            note_emission(Some(fp(42)), || true, || Some(fp(100)));
            note_unattributed_emission();
        });
        assert!(!populate_pair_from_frame_fps(
            fp(42),
            7,
            99,
            [fp(100)].into_iter()
        ));
        clear();
    }

    /// Per-tag cap overflow poisons the record (capped sets are never treated
    /// as complete; the caller's fallback handles TRUE population).
    #[test]
    fn frame_fp_cap_overflow_blocks_population() {
        super::super::checker::clear_scan_pred_results();
        frame_fp_run(true, || {
            for i in 0..(FRAME_FP_CAP as u64 + 1) {
                note_emission(Some(fp(42)), || true, || Some(fp(1000 + i)));
            }
        });
        assert!(!populate_pair_from_frame_fps(
            fp(42),
            7,
            99,
            [fp(1000)].into_iter()
        ));
        clear();
    }

    /// A fresh arm clears the previous state's records — fps can never leak
    /// across states.
    #[test]
    fn frame_fp_cleared_on_rearm() {
        super::super::checker::clear_scan_pred_results();
        frame_fp_run(true, || {
            note_emission(Some(fp(42)), || true, || Some(fp(100)));
        });
        {
            // Re-arm for a different state; no frames pushed.
            let _arm = arm_state_guard(fp(43));
        }
        assert!(!populate_pair_from_frame_fps(
            fp(43),
            7,
            99,
            [fp(100)].into_iter()
        ));
        clear();
    }

    /// Nested enumerations (depth ≥ 2, e.g. an ENABLED evaluation running
    /// inside a streaming sink callback) never record.
    #[test]
    fn nested_enumeration_never_records() {
        register(vec![leaf(0x1000, vec![], 7, false)]);
        {
            let _arm = arm_state_guard(fp(42));
            let _outer = enum_scope();
            {
                let _inner = enum_scope(); // nested enumeration
                assert!(!wants_frame(0x1000));
                let _frame = push_frame(0x1000, &[]);
                note_emission(None, || true, || None);
            }
        }
        assert!(!witnessed_true(fp(42), 7));
        clear();
    }

    /// Suppressed regions (provisional emissions pending post-validation)
    /// never record.
    #[test]
    fn suppressed_region_never_records() {
        register(vec![leaf(0x1000, vec![], 7, false)]);
        {
            let _arm = arm_state_guard(fp(42));
            let _scope = enum_scope();
            let _frame = push_frame(0x1000, &[]);
            {
                let _sup = suppress_scope();
                note_emission(None, || true, || None);
            }
        }
        assert!(!witnessed_true(fp(42), 7));
        clear();
    }

    /// Unarmed (or disarmed) windows never record; witnesses survive disarm
    /// until the next arm so the inline recorder can consume them.
    #[test]
    fn arm_bracketing() {
        register(vec![leaf(0x1000, vec![], 7, false)]);
        // Not armed: nothing records.
        {
            let _scope = enum_scope();
            let _frame = push_frame(0x1000, &[]);
            note_emission(None, || true, || None);
        }
        assert!(!witnessed_true(fp(42), 7));
        // Armed: records; witness survives the guard drop (disarm).
        {
            let _arm = arm_state_guard(fp(42));
            let _scope = enum_scope();
            let _frame = push_frame(0x1000, &[]);
            note_emission(Some(fp(42)), || true, || None);
        }
        assert!(witnessed_true(fp(42), 7));
        // Post-disarm enumerations (e.g. post-BFS phases) never record.
        {
            let _scope = enum_scope();
            let _frame = push_frame(0x1000, &[]);
            note_emission(None, || true, || None);
        }
        // Re-arming for a new state clears the old scratch.
        {
            let _arm = arm_state_guard(fp(43));
        }
        assert!(!witnessed_true(fp(42), 7));
        clear();
    }

    /// Wildcard frames (#3208): an `AnyOf` position witnesses only when the
    /// emission's value lies IN the registered domain — a wider `Next`-side
    /// binding value records nothing.
    #[test]
    fn wildcard_domain_membership() {
        register(vec![RegisteredEnabledLeaf {
            def_ptr: 0x1000,
            args: vec![
                ArgPattern::Exact(Value::int(1)),
                ArgPattern::AnyOf(vec![Value::int(10), Value::int(11)]),
            ],
            enabled_tag: 7,
            needs_change: true,
        }]);
        {
            let _arm = arm_state_guard(fp(42));
            let _scope = enum_scope();
            {
                // In-domain wildcard value: witness.
                let _frame = push_frame(0x1000, &[Value::int(1), Value::int(11)]);
                note_emission(Some(fp(42)), || true, || None);
            }
            {
                // Out-of-domain wildcard value (Next quantified wider): no claim.
                let _frame = push_frame(0x1000, &[Value::int(1), Value::int(12)]);
                note_emission(Some(fp(42)), || true, || None);
            }
            {
                // Exact-position mismatch: no claim.
                let _frame = push_frame(0x1000, &[Value::int(2), Value::int(10)]);
                note_emission(Some(fp(42)), || true, || None);
            }
        }
        assert!(witnessed_true(fp(42), 7));
        // Re-arm and verify each non-matching case alone records nothing.
        {
            let _arm = arm_state_guard(fp(43));
            let _scope = enum_scope();
            let _frame = push_frame(0x1000, &[Value::int(1), Value::int(12)]);
            note_emission(Some(fp(43)), || true, || None);
        }
        assert!(!witnessed_true(fp(43), 7));
        clear();
    }

    /// The ambiguous-mapping fail-closed contract: registration is the ONLY
    /// source of frame identities, so an unregistered def pointer (the
    /// stand-in for LET-shadowed / INSTANCE-scoped / unhinted actions) records
    /// nothing.
    #[test]
    fn unregistered_def_fails_closed() {
        register(vec![leaf(0x1000, vec![], 7, false)]);
        {
            let _arm = arm_state_guard(fp(42));
            let _scope = enum_scope();
            assert!(!wants_frame(0x2000));
            let _frame = push_frame(0x2000, &[]);
            note_emission(None, || true, || None);
        }
        assert!(!witnessed_true(fp(42), 7));
        clear();
    }

    // ── #guard-moduleref (Step A) / #guard-in-memo (Step B) tests ──────────
    //
    // NOTE: the guard-plan registry is thread-local and each test runs in its
    // own registration window (`register(vec![])` … `clear()`), and every test
    // here uses tags unique across this module, so parallel test threads and
    // in-thread ordering can never alias plans.

    use crate::liveness::live_expr::LiveExpr;
    use tla_core::name_intern::{intern_name, NameId};
    use tla_core::{lower, parse_to_syntax_tree, FileId};

    fn sp(node: Expr) -> Spanned<Expr> {
        Spanned::dummy(node)
    }

    fn ident(name: &str) -> Spanned<Expr> {
        sp(Expr::Ident(name.to_string(), NameId::INVALID))
    }

    fn chain_of(name: &str, value: Value) -> crate::eval::BindingChain {
        crate::eval::BindingChain::empty()
            .cons(intern_name(name), crate::eval::BindingValue::eager(value))
    }

    /// Outer module instantiating `ProvGuardInner` (guard ops behind an
    /// INSTANCE boundary), mirroring the AllocatorImplementation /
    /// SchedulingAllocator shape the extraction targets.
    fn moduleref_test_ctx() -> EvalCtx {
        let inner_src = r#"
---- MODULE ProvGuardInner ----
VARIABLE x
gate == x = 0
Op == gate /\ x' = 1
Op2(v) == v = x /\ x' = 1
Op3(c, S) == S # {} /\ S \subseteq c /\ x' = x
====
"#;
        let outer_src = r#"
---- MODULE ProvGuardOuter ----
VARIABLE x
I == INSTANCE ProvGuardInner
====
"#;
        let inner = lower(FileId(1), &parse_to_syntax_tree(inner_src));
        assert!(inner.errors.is_empty(), "inner errors: {:?}", inner.errors);
        let inner = inner.module.expect("inner module");
        let outer = lower(FileId(0), &parse_to_syntax_tree(outer_src));
        assert!(outer.errors.is_empty(), "outer errors: {:?}", outer.errors);
        let outer = outer.module.expect("outer module");
        let mut ctx = EvalCtx::new();
        ctx.load_module(&outer);
        ctx.load_instance_module(inner.name.node.clone(), &inner);
        ctx.register_var("x");
        ctx
    }

    /// Step A positive: a bare `I!Op` action (the property-plan
    /// `Sched!Schedule` shape) yields the resolved body's leading guard
    /// (`gate`, a module-local operator), evaluated under the replayed
    /// INSTANCE scope: FALSE in the current state refutes, TRUE makes no
    /// claim.
    #[test]
    fn moduleref_guard_resolves_and_refutes() {
        let ctx = moduleref_test_ctx();
        let action = sp(Expr::ModuleRef(
            ModuleTarget::Named("I".to_string()),
            "Op".to_string(),
            vec![],
        ));
        let leaf = LiveExpr::enabled(Arc::new(action), 901);
        let plans = build_enabled_guard_plans(&ctx, std::iter::once(&leaf));
        assert_eq!(plans.len(), 1, "bare ModuleRef action should yield a plan");
        assert_eq!(plans[0].guards.len(), 1);
        assert!(
            plans[0].guards[0].scope.is_some(),
            "the lifted guard must carry the instance scope"
        );
        register(Vec::new());
        extend_guard_plans(plans);
        // gate == x = 0: x = 5 refutes (Op disabled), x = 0 makes no claim.
        let mut refute_ctx = ctx.clone();
        refute_ctx.bind_mut("x", Value::int(5));
        assert!(guard_prefix_refutes(&mut refute_ctx, fp(1), 901));
        let mut open_ctx = ctx.clone();
        open_ctx.bind_mut("x", Value::int(0));
        assert!(!guard_prefix_refutes(&mut open_ctx, fp(2), 901));
        clear();
    }

    /// Step A positive with arguments: `I!Op2(3)` binds the resolved formal
    /// `v` to the caller-side argument value, so the lifted guard `v = x`
    /// refutes exactly when `x # 3`.
    #[test]
    fn moduleref_guard_binds_formals() {
        let ctx = moduleref_test_ctx();
        let action = sp(Expr::ModuleRef(
            ModuleTarget::Named("I".to_string()),
            "Op2".to_string(),
            vec![sp(Expr::Int(3.into()))],
        ));
        let leaf = LiveExpr::enabled(Arc::new(action), 902);
        let plans = build_enabled_guard_plans(&ctx, std::iter::once(&leaf));
        assert_eq!(plans.len(), 1);
        register(Vec::new());
        extend_guard_plans(plans);
        let mut refute_ctx = ctx.clone();
        refute_ctx.bind_mut("x", Value::int(5));
        assert!(guard_prefix_refutes(&mut refute_ctx, fp(3), 902));
        let mut open_ctx = ctx.clone();
        open_ctx.bind_mut("x", Value::int(3));
        assert!(!guard_prefix_refutes(&mut open_ctx, fp(4), 902));
        clear();
    }

    /// Step A fail-closed: an unresolvable instance and an action-level
    /// argument each yield NO plan (full evaluation preserved).
    #[test]
    fn moduleref_guard_fails_closed() {
        let ctx = moduleref_test_ctx();
        // Unknown instance name.
        let unknown = LiveExpr::enabled(
            Arc::new(sp(Expr::ModuleRef(
                ModuleTarget::Named("NoSuchInstance".to_string()),
                "Op".to_string(),
                vec![],
            ))),
            903,
        );
        assert!(
            build_enabled_guard_plans(&ctx, std::iter::once(&unknown)).is_empty(),
            "unresolvable instance must fail closed"
        );
        // Op-not-found in a known instance.
        let no_op = LiveExpr::enabled(
            Arc::new(sp(Expr::ModuleRef(
                ModuleTarget::Named("I".to_string()),
                "NoSuchOp".to_string(),
                vec![],
            ))),
            904,
        );
        assert!(
            build_enabled_guard_plans(&ctx, std::iter::once(&no_op)).is_empty(),
            "op-not-found must fail closed"
        );
        // Action-level (primed) argument.
        let primed_arg = LiveExpr::enabled(
            Arc::new(sp(Expr::ModuleRef(
                ModuleTarget::Named("I".to_string()),
                "Op2".to_string(),
                vec![sp(Expr::Prime(Box::new(ident("x"))))],
            ))),
            905,
        );
        assert!(
            build_enabled_guard_plans(&ctx, std::iter::once(&primed_arg)).is_empty(),
            "primed argument must fail closed"
        );
        clear();
    }

    /// A `m \in net /\ <action conjunct>` action with a state-variable RHS
    /// (the Allocator message-receive shape) for binding `m = lhs`.
    fn memo_leaf(tag: u32, lhs: i64) -> LiveExpr {
        let guard = sp(Expr::In(Box::new(ident("m")), Box::new(ident("net"))));
        let assign = sp(Expr::Eq(
            Box::new(sp(Expr::Prime(Box::new(ident("net"))))),
            Box::new(ident("net")),
        ));
        let action = sp(Expr::And(Box::new(guard), Box::new(assign)));
        LiveExpr::enabled_with_bindings(
            Arc::new(action),
            false,
            None,
            tag,
            Some(chain_of("m", Value::int(lhs))),
        )
    }

    /// Step B positive: the `m \in net` guard is memo-decided (refutes on
    /// non-membership, no claim on membership), the per-state RHS value is
    /// SHARED across leaves of one state (structural grouping), and a new
    /// state fingerprint invalidates the memo.
    #[test]
    fn memo_in_guard_decides_and_invalidates_across_states() {
        let mut ctx = EvalCtx::new();
        ctx.register_var("net");
        let leaf_a = memo_leaf(906, 1);
        let leaf_b = memo_leaf(907, 2);
        let plans = build_enabled_guard_plans(&ctx, [leaf_a, leaf_b].iter());
        assert_eq!(plans.len(), 2);
        for plan in &plans {
            assert_eq!(plan.guards.len(), 1);
            assert!(plan.guards[0].memo.is_some(), "m \\in net must be memoized");
        }
        register(Vec::new());
        extend_guard_plans(plans);
        // Structurally identical RHS ⇒ one sharing group.
        PROV.with(|p| assert_eq!(p.borrow().memo_groups.len(), 1));

        // State fp(10): net = {2, 3} — m=1 refuted, m=2 not.
        let mut s1 = ctx.clone();
        s1.bind_mut("net", Value::set([Value::int(2), Value::int(3)]));
        assert!(guard_prefix_refutes(&mut s1, fp(10), 906));
        assert!(!guard_prefix_refutes(&mut s1, fp(10), 907));

        // Memo HIT within the same state: mutating the bound value while
        // keeping the fingerprint must still serve the memoized {2, 3}
        // (production never rebinds within one fingerprint — this pins the
        // fp-keyed sharing).
        s1.bind_mut("net", Value::set([Value::int(1)]));
        assert!(
            guard_prefix_refutes(&mut s1, fp(10), 906),
            "same-fingerprint queries must be served from the memo"
        );

        // New state fp(11): net = {1} — memo invalidated, m=1 now a member
        // (no claim), m=2 refuted.
        let mut s2 = ctx.clone();
        s2.bind_mut("net", Value::set([Value::int(1)]));
        assert!(!guard_prefix_refutes(&mut s2, fp(11), 906));
        assert!(guard_prefix_refutes(&mut s2, fp(11), 907));
        clear();
    }

    /// Step B fail-closed: an RHS that is not a bare state variable — a plan
    /// binding, an operator reference, or an unregistered name — is never
    /// memoized (the guard keeps its canonical per-state evaluation).
    #[test]
    fn memo_in_guard_fails_closed_for_non_state_rhs() {
        let ctx = moduleref_test_ctx(); // has op `gate` (via instance), var `x`
        let mk = |rhs: Spanned<Expr>, tag: u32| {
            let guard = sp(Expr::In(Box::new(ident("m")), Box::new(rhs)));
            let assign = sp(Expr::Eq(
                Box::new(sp(Expr::Prime(Box::new(ident("x"))))),
                Box::new(ident("x")),
            ));
            LiveExpr::enabled_with_bindings(
                Arc::new(sp(Expr::And(Box::new(guard), Box::new(assign)))),
                false,
                None,
                tag,
                Some(chain_of("m", Value::int(1))),
            )
        };
        // RHS = the binding itself (shadowed).
        let shadowed = mk(ident("m"), 908);
        // RHS = a name that is not a registered state variable.
        let unregistered = mk(ident("nosuchvar"), 909);
        let plans = build_enabled_guard_plans(&ctx, [shadowed, unregistered].iter());
        for plan in &plans {
            for guard in &plan.guards {
                assert!(
                    guard.memo.is_none(),
                    "non-state-variable RHS must not be memoized (tag {})",
                    plan.enabled_tag
                );
            }
        }
        clear();
    }

    /// Step B binding-const guards: a const conjunct proven FALSE at build
    /// refutes with zero per-state evaluation (in registration order), and a
    /// const conjunct proven TRUE is dropped from the plan.
    #[test]
    fn binding_const_guard_refutes_and_true_guard_drops() {
        let mut ctx = EvalCtx::new();
        ctx.register_var("net");
        let assign = sp(Expr::Eq(
            Box::new(sp(Expr::Prime(Box::new(ident("net"))))),
            Box::new(ident("net")),
        ));
        // m \in {2, 3} with m = 1: const FALSE.
        let const_false = sp(Expr::In(
            Box::new(ident("m")),
            Box::new(sp(Expr::SetEnum(vec![
                sp(Expr::Int(2.into())),
                sp(Expr::Int(3.into())),
            ]))),
        ));
        let leaf_false = LiveExpr::enabled_with_bindings(
            Arc::new(sp(Expr::And(
                Box::new(const_false),
                Box::new(assign.clone()),
            ))),
            false,
            None,
            910,
            Some(chain_of("m", Value::int(1))),
        );
        // m = 1 (const TRUE) followed by m \in net (memoized).
        let const_true = sp(Expr::Eq(
            Box::new(ident("m")),
            Box::new(sp(Expr::Int(1.into()))),
        ));
        let memo_guard = sp(Expr::In(Box::new(ident("m")), Box::new(ident("net"))));
        let both = sp(Expr::And(
            Box::new(const_true),
            Box::new(sp(Expr::And(Box::new(memo_guard), Box::new(assign)))),
        ));
        let leaf_true = LiveExpr::enabled_with_bindings(
            Arc::new(both),
            false,
            None,
            911,
            Some(chain_of("m", Value::int(1))),
        );
        let plans = build_enabled_guard_plans(&ctx, [leaf_false, leaf_true].iter());
        assert_eq!(plans.len(), 2);
        let p_false = plans.iter().find(|p| p.enabled_tag == 910).unwrap();
        assert_eq!(p_false.guards.len(), 1);
        assert_eq!(p_false.guards[0].const_val, Some(false));
        let p_true = plans.iter().find(|p| p.enabled_tag == 911).unwrap();
        assert_eq!(
            p_true.guards.len(),
            1,
            "the proven-TRUE const conjunct must be dropped"
        );
        assert!(p_true.guards[0].memo.is_some());
        register(Vec::new());
        extend_guard_plans(plans);
        // Const-FALSE plan refutes with NO state bound at all.
        let mut bare = ctx.clone();
        assert!(guard_prefix_refutes(&mut bare, fp(20), 910));
        clear();
    }

    // ── #guard-exists (quantified derivations) tests ────────────────────────

    fn exists_bounds(name: &str) -> Vec<tla_core::ast::BoundVar> {
        vec![tla_core::ast::BoundVar {
            name: Spanned::dummy(name.to_string()),
            domain: Some(Box::new(sp(Expr::SetEnum(vec![
                sp(Expr::SetEnum(vec![])),
                sp(Expr::SetEnum(vec![sp(Expr::Int(1.into()))])),
            ])))),
            pattern: None,
        }]
    }

    /// #guard-exists pair inference: `\E S : S # {} /\ S \subseteq net /\
    /// net' = ...` derives the guard `net # {}` — refutes exactly when the
    /// state's `net` is empty.
    #[test]
    fn exists_pair_inference_derives_and_refutes() {
        let mut ctx = EvalCtx::new();
        ctx.register_var("net");
        let body = sp(Expr::And(
            Box::new(sp(Expr::And(
                Box::new(sp(Expr::Neq(
                    Box::new(ident("S")),
                    Box::new(sp(Expr::SetEnum(vec![]))),
                ))),
                Box::new(sp(Expr::Subseteq(
                    Box::new(ident("S")),
                    Box::new(ident("net")),
                ))),
            ))),
            Box::new(sp(Expr::Eq(
                Box::new(sp(Expr::Prime(Box::new(ident("net"))))),
                Box::new(ident("net")),
            ))),
        ));
        let action = sp(Expr::Exists(exists_bounds("S"), Box::new(body)));
        let leaf = LiveExpr::enabled(Arc::new(action), 920);
        let plans = build_enabled_guard_plans(&ctx, std::iter::once(&leaf));
        assert_eq!(plans.len(), 1, "pair inference should derive a plan");
        assert_eq!(plans[0].guards.len(), 1);
        assert!(
            matches!(plans[0].guards[0].expr.node, Expr::Neq(..)),
            "derived guard should be `net # {{}}`"
        );
        register(Vec::new());
        extend_guard_plans(plans);
        let mut empty_net = ctx.clone();
        empty_net.bind_mut("net", Value::set(Vec::<Value>::new()));
        assert!(guard_prefix_refutes(&mut empty_net, fp(30), 920));
        let mut full_net = ctx.clone();
        full_net.bind_mut("net", Value::set([Value::int(1)]));
        assert!(!guard_prefix_refutes(&mut full_net, fp(31), 920));
        clear();
    }

    /// #guard-exists wrap lift (case 2): `\E i \in nums : i = target /\
    /// x' = 1` re-wraps the leading binder-usable run as
    /// `\E i \in nums : i = target` — refutes exactly when no member of
    /// `nums` equals `target`.
    #[test]
    fn exists_wrap_lift_derives_and_refutes() {
        let mut ctx = EvalCtx::new();
        ctx.register_var("nums");
        ctx.register_var("target");
        let inner = sp(Expr::And(
            Box::new(sp(Expr::Eq(
                Box::new(ident("i")),
                Box::new(ident("target")),
            ))),
            Box::new(sp(Expr::Eq(
                Box::new(sp(Expr::Prime(Box::new(ident("target"))))),
                Box::new(sp(Expr::Int(1.into()))),
            ))),
        ));
        let bounds = vec![tla_core::ast::BoundVar {
            name: Spanned::dummy("i".to_string()),
            domain: Some(Box::new(ident("nums"))),
            pattern: None,
        }];
        let quant = sp(Expr::Exists(bounds, Box::new(inner)));
        // Two-conjunct action so the spine walk (not the bare-body gate)
        // exercises the Exists arm.
        let action = sp(Expr::And(
            Box::new(quant),
            Box::new(sp(Expr::Eq(
                Box::new(sp(Expr::Prime(Box::new(ident("nums"))))),
                Box::new(ident("nums")),
            ))),
        ));
        let leaf = LiveExpr::enabled(Arc::new(action), 921);
        let plans = build_enabled_guard_plans(&ctx, std::iter::once(&leaf));
        assert_eq!(plans.len(), 1, "wrap lift should derive a plan");
        assert_eq!(plans[0].guards.len(), 1);
        assert!(
            matches!(plans[0].guards[0].expr.node, Expr::Exists(..)),
            "derived guard should be the re-wrapped existential"
        );
        register(Vec::new());
        extend_guard_plans(plans);
        let mut miss = ctx.clone();
        miss.bind_mut("nums", Value::set([Value::int(1), Value::int(2)]));
        miss.bind_mut("target", Value::int(5));
        assert!(guard_prefix_refutes(&mut miss, fp(32), 921));
        let mut hit = ctx.clone();
        hit.bind_mut("nums", Value::set([Value::int(1), Value::int(2)]));
        hit.bind_mut("target", Value::int(2));
        assert!(!guard_prefix_refutes(&mut hit, fp(33), 921));
        clear();
    }

    /// #guard-exists fail-closed: pattern binders, missing domains, and
    /// bodies whose leading conjunct is already action-level derive nothing.
    #[test]
    fn exists_fails_closed() {
        let mut ctx = EvalCtx::new();
        ctx.register_var("net");
        // Missing domain.
        let no_domain = sp(Expr::Exists(
            vec![tla_core::ast::BoundVar {
                name: Spanned::dummy("S".to_string()),
                domain: None,
                pattern: None,
            }],
            Box::new(sp(Expr::Eq(
                Box::new(sp(Expr::Prime(Box::new(ident("net"))))),
                Box::new(ident("S")),
            ))),
        ));
        let leaf = LiveExpr::enabled(Arc::new(no_domain), 922);
        assert!(build_enabled_guard_plans(&ctx, std::iter::once(&leaf)).is_empty());
        // Leading conjunct action-level: nothing to derive.
        let primed_body = sp(Expr::Exists(
            exists_bounds("S"),
            Box::new(sp(Expr::Eq(
                Box::new(sp(Expr::Prime(Box::new(ident("net"))))),
                Box::new(ident("S")),
            ))),
        ));
        let leaf2 = LiveExpr::enabled(Arc::new(primed_body), 923);
        assert!(build_enabled_guard_plans(&ctx, std::iter::once(&leaf2)).is_empty());
        clear();
    }

    /// #guard-exists + #guard-moduleref integration (the Allocator
    /// `\E S \in SUBSET Resources : Sched!Allocate(c, S)` shape): the binder
    /// flows into the instance formal, which becomes forbidden inside the
    /// resolved body, and the pair inference lifts `E # {}` against the
    /// EVALUABLE formal (`c ↦ net`).
    #[test]
    fn exists_moduleref_pair_inference() {
        let mut ctx = moduleref_test_ctx();
        ctx.register_var("net");
        let action = sp(Expr::Exists(
            exists_bounds("S"),
            Box::new(sp(Expr::ModuleRef(
                ModuleTarget::Named("I".to_string()),
                "Op3".to_string(),
                vec![ident("net"), ident("S")],
            ))),
        ));
        let leaf = LiveExpr::enabled(Arc::new(action), 924);
        let plans = build_enabled_guard_plans(&ctx, std::iter::once(&leaf));
        assert_eq!(
            plans.len(),
            1,
            "moduleref-under-exists should derive a plan"
        );
        assert_eq!(plans[0].guards.len(), 1);
        let guard = &plans[0].guards[0];
        assert!(guard.scope.is_some(), "guard must carry the instance scope");
        assert!(matches!(guard.expr.node, Expr::Neq(..)));
        register(Vec::new());
        extend_guard_plans(plans);
        // net = {}: Op3's `S \subseteq c` (c ↦ net) is unsatisfiable for any
        // S # {} — the derived `c # {}` refutes.
        let mut empty_net = ctx.clone();
        empty_net.bind_mut("x", Value::int(0));
        empty_net.bind_mut("net", Value::set(Vec::<Value>::new()));
        assert!(guard_prefix_refutes(&mut empty_net, fp(34), 924));
        // net = {1}: no claim.
        let mut full_net = ctx.clone();
        full_net.bind_mut("x", Value::int(0));
        full_net.bind_mut("net", Value::set([Value::int(1)]));
        assert!(!guard_prefix_refutes(&mut full_net, fp(35), 924));
        clear();
    }
}
