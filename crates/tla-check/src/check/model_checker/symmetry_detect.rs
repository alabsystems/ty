// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Automatic symmetry detection from model value sets in TLC configuration.
//!
//! When a spec declares a constant as a set of fresh model values — either the
//! replacement form `Procs <- {p1, p2, p3}` (a `ModelValueSet`) or the
//! assignment form `Procs = {p1, p2, p3}` (a `Value` whose members are all
//! fresh identifiers) — and no explicit `SYMMETRY` directive is configured,
//! the model values in that set are candidates for symmetry reduction via
//! `Permutations(Procs)`.
//!
//! This module detects such candidate sets, applies soundness guards, and
//! generates the corresponding permutation groups, enabling TLC-style symmetry
//! reduction without manual configuration.
//!
//! # Default and kill switch
//!
//! Auto-symmetry is part of the production default (like auto-POR).
//! Controlled by the `TY_AUTO_SYMMETRY` environment variable:
//! - unset, `1`, or `true`: enabled (default)
//! - `0` or `false`: disabled (kill switch)
//!
//! Tests and embedders must use `ModelChecker::set_auto_symmetry(..)` instead
//! of mutating the environment (see `auto_symmetry_enabled`).
//!
//! # Soundness guards (all structural, never name-based)
//!
//! A permutation of the members of a candidate set is a state-graph
//! automorphism only if nothing else in the model distinguishes the members.
//! Each candidate group must pass ALL of:
//!
//! - **Freshness**: every member is a fresh identifier — not itself a config
//!   constant (Toolbox-style `m = m` makes `m` referenceable from the spec)
//!   and not colliding with an operator definition. This proves the spec text
//!   cannot name a member directly.
//! - **Guard (a) — constant-environment invariance**: every other constant
//!   binding (config constants, precomputed zero-arity constant operators,
//!   module-scoped assignments) must be fixed by the group's generators.
//!   This excludes pinned members (`Root = n1`) and asymmetric derived
//!   constants (e.g. a randomly chosen edge set over the members). Values that
//!   cannot be verified (lazy/opaque, or too large to enumerate) fail closed.
//! - **Guard (b) — no order-sensitive use**: the candidate must not appear
//!   (transitively, through operator definitions) as the domain of a bounded
//!   `CHOOSE`, nor as an argument to order/randomness-sensitive builtins
//!   (`RandomElement`, `SortSeq`, `SetToSeq`, `ToString`, ...). This is the
//!   restriction TLC documents for declared symmetry sets; for the auto path
//!   we enforce it statically. Unbounded `CHOOSE x : x \notin S` is permitted:
//!   its witness lies outside the set and is fixed by every group element.
//! - **Guard (c) — no genuine temporal properties** (enforced by the caller in
//!   `run_prepare`): symmetry reduction is unsound for liveness checking, so
//!   auto-symmetry hard-disables when any configured PROPERTY requires the
//!   liveness checker. (Declared SYMMETRY keeps TLC's warn-and-continue.)
//!
//! # Guard (c): why symmetry + liveness stays disabled (2026-06 analysis)
//!
//! Naive orbit reduction (canonicalize every state, run the liveness check on
//! the quotient behavior graph) is unsound in BOTH directions whenever any
//! atom of the temporal formula is not orbit-invariant — which is the common
//! case: per-process fairness `\A p \in Procs : WF_vars(A(p))` and per-process
//! properties expand into atoms (`ENABLED <<A(p1)>>_v`, `<<A(p1)>>_v`,
//! state predicates mentioning `p1`) that distinguish orbit members.
//!
//! The failure mode is *threading*: a cycle in the quotient corresponds to a
//! longer cycle in the full graph that only closes after composing the
//! per-edge aligning permutations around the loop (`k = ord(pi)` laps), and
//! the atom values along that real cycle are the rep's values re-indexed by
//! the accumulated permutation — NOT the rep's values. Evaluating fairness/AE
//! constraints on representatives therefore both (a) accepts quotient cycles
//! whose threaded real cycle is unfair (false VIOLATED) and (b) refutes
//! candidates whose threaded cycle is genuinely fair (false HOLD).
//! Demonstrated live on this corpus: `AllocatorImplementation` (PROPERTY
//! `SchedAllocator`, group S3xS2) under declared SYMMETRY reports a violation
//! with a counterexample, while both unreduced TY and TLC prove HOLD.
//!
//! ## When it CAN be sound (conditions, not implemented)
//!
//! 1. *Fully symmetric atoms* (bisimulation case): if every state/action
//!    atom of `fairness /\ ~property` — after expansion through operator
//!    definitions and temporal-level quantifiers — is invariant under the
//!    group (true when no temporal-level quantifier ranges over a domain
//!    containing symmetric model values; member freshness + guards (a)/(b)
//!    already exclude direct naming), then the orbit map is a bisimulation
//!    w.r.t. the atom alphabet and the quotient preserves the verdict
//!    exactly. Even this case is NOT enabled today because the shipped
//!    symmetry-liveness machinery has rep-pair evaluation gaps that must be
//!    fixed first (see `witness_cycle_satisfies_pem`: the gate evaluates
//!    action checks directly on consecutive REP pairs, which need not be
//!    real spec steps — unlike `eval_check_on_transition`, it does not fold
//!    existentially over the concrete successor witnesses; its
//!    successor-reseeding fallback seeds rep states rather than witnesses;
//!    and inline leaf recording is disabled under symmetry entirely, so the
//!    quotient path also runs without the leaf-memo caches).
//! 2. *Symmetric formula, asymmetric atoms* (the Disruptor/Allocator class):
//!    soundness requires the permutation-annotated quotient (Emerson-Sistla):
//!    each quotient edge carries the canonicalizing permutation (the
//!    `best_perm` that `fingerprint_with_symmetry` computes and discards),
//!    the liveness check runs on the lift `(orbit-rep, g)` with check masks
//!    re-indexed by the group action on the check family. By
//!    orbit-stabilizer, the reachable lift has `sum_s |Stab(rep(s))| >= |S|`
//!    nodes — the liveness-phase graph CANNOT shrink below full-graph scale.
//!
//! ## Why the sound construction is a measured no-go on this corpus
//!
//! The blocked census losses are not liveness-graph-bound; their cost is BFS
//! plus inline fairness-leaf evaluation (measured, single worker, db903eac):
//!
//! | spec                      | TY full | naive-quotient TY | TLC full |
//! |---------------------------|---------|-------------------|----------|
//! | Disruptor_SPMC (S3)       |  1.78s  |  0.73s            |  0.77s   |
//! | Disruptor_MPMC_liv (S2xS2)|  3.26s  |  1.82s (UNSOUND)  |  1.31s   |
//! | AllocatorImpl (S3xS2)     | 24.2s   |  4.20s (FALSE CEX)|  4.15s   |
//!
//! The naive quotient is a strict WORK FLOOR for the sound annotated lift
//! (the lift performs the same per-orbit-edge leaf evaluations plus the
//! lifted-graph mask/SCC work), and that floor already loses to TLC on
//! Disruptor_MPMC_liveliness and only ties on the others. The dominant cost
//! (e.g. ~22.5s of 24.2s on AllocatorImplementation) is fairness-leaf
//! evaluation throughput, which symmetry does not address — the symmetry
//! path additionally bypasses the inline leaf-memo caches today. Conclusion:
//! keep guard (c); the leverage on this class is leaf-eval throughput, not
//! state-count reduction.
//!
//! Residual (documented, same contract as TLC's declared SYMMETRY): a bounded
//! CHOOSE (or order-sensitive builtin) over a *state-derived* set whose
//! elements include candidate model values cannot be excluded statically
//! without type inference. Such CHOOSEs are almost always uniquely-satisfied
//! selections (min/max patterns), which are orbit-equivariant.

use crate::config::{Config, ConstantValue};
use tla_value::Rp;
use crate::eval::EvalCtx;
use crate::value::{FuncValue, Value};
use rustc_hash::{FxHashMap, FxHashSet};
use std::sync::Arc;
use tla_core::ast::{Expr, OperatorDef};
use tla_core::span::Spanned;
use tla_core::visit::ExprVisitor;

/// Check if auto-symmetry detection is enabled via environment variable.
///
/// This is only consulted during checker setup (and only when no per-checker
/// override is set via `ModelChecker::set_auto_symmetry`), so keep it dynamic
/// rather than caching the first process-wide value.
///
/// NOTE: tests must NOT toggle this env var — `std::env::set_var` is
/// process-global and races with concurrently-running checkers. Use
/// `ModelChecker::set_auto_symmetry(..)` instead; the parsing logic is
/// unit-testable via `auto_symmetry_enabled_from_value`.
#[must_use]
pub(crate) fn auto_symmetry_enabled() -> bool {
    auto_symmetry_enabled_from_value(std::env::var("TY_AUTO_SYMMETRY").ok().as_deref())
}

/// Pure parser for the `TY_AUTO_SYMMETRY` value: enabled by default (unset),
/// `0`/`false` disable (kill switch). Extracted so tests can cover the parsing
/// without mutating the process environment.
#[must_use]
pub(crate) fn auto_symmetry_enabled_from_value(value: Option<&str>) -> bool {
    !matches!(value, Some("0") | Some("false"))
}

/// Whether to APPLY a declared `SYMMETRY` even for specs with genuine temporal
/// (liveness) properties, matching TLC's behavior exactly.
///
/// DEFAULT OFF (sound): ty normally IGNORES declared symmetry under liveness
/// checking because the orbit quotient is unsound for temporal properties (TLC
/// applies it anyway and can report wrong verdicts). Safety specs already apply
/// declared symmetry unconditionally — this gate only affects the liveness
/// path.
///
/// Set `TY_MATCH_DECLARED_SYMMETRY=1` to make ty compute EXACTLY what TLC
/// computes (same orbit-reduced state space + orbit-quotient liveness) — an
/// apples-to-apples BENCHMARK-PARITY tool for comparing single-thread speed on
/// declared-SYMMETRY liveness specs. ⚠️ The resulting verdict inherits TLC's
/// orbit-quotient (un)soundness for temporal properties; use it for TIMING
/// PARITY, not to trust a liveness verdict.
#[must_use]
pub(crate) fn match_declared_symmetry_for_liveness() -> bool {
    match_declared_symmetry_for_liveness_from_value(
        std::env::var("TY_MATCH_DECLARED_SYMMETRY").ok().as_deref(),
    )
}

/// Pure parser for `TY_MATCH_DECLARED_SYMMETRY`: disabled by default (unset),
/// `1`/`true` enable.
#[must_use]
pub(crate) fn match_declared_symmetry_for_liveness_from_value(value: Option<&str>) -> bool {
    matches!(value, Some("1") | Some("true"))
}

/// Detect model value set constants that are candidates for automatic symmetry.
///
/// Returns a list of (constant_name, model_value_names) pairs for:
/// - each `ModelValueSet` constant (`Name <- {a, b}` form), and
/// - each `Value` constant of the form `Name = {a, b}` where every member is a
///   plain identifier (not an integer/string/boolean literal) that is not
///   itself assigned elsewhere in the config (freshness — Toolbox-style
///   `a = a` lines make `a` spec-referenceable, which pins it).
///
/// Sets with fewer than 2 elements are excluded (trivial symmetry group).
#[must_use]
pub(crate) fn detect_symmetric_model_value_sets(config: &Config) -> Vec<(String, Vec<String>)> {
    let mut candidates = Vec::new();
    for (name, value) in &config.constants {
        match value {
            ConstantValue::ModelValueSet(values) => {
                if values.len() >= 2 {
                    candidates.push((name.clone(), values.clone()));
                }
            }
            ConstantValue::Value(raw) => {
                if let Some(values) = parse_eq_form_model_value_set(raw) {
                    if values.len() >= 2 && values.iter().all(|v| !config.constants.contains_key(v))
                    {
                        candidates.push((name.clone(), values));
                    }
                }
            }
            _ => {}
        }
    }
    // Sort by name for deterministic ordering.
    candidates.sort_by(|a, b| a.0.cmp(&b.0));
    candidates
}

/// Parse an `=`-form constant value as a set of fresh model-value identifiers.
///
/// Returns `Some(members)` only when the value is a flat `{id, id, ...}` set
/// literal whose members are all distinct plain identifiers. Booleans
/// (`TRUE`/`FALSE`), numbers, strings, and nested literals all return `None` —
/// those evaluate to non-model values and must never seed permutations.
fn parse_eq_form_model_value_set(raw: &str) -> Option<Vec<String>> {
    let trimmed = raw.trim();
    let inner = trimmed.strip_prefix('{')?.strip_suffix('}')?;
    let mut out: Vec<String> = Vec::new();
    for piece in inner.split(',') {
        let p = piece.trim();
        if !is_plain_identifier(p) || p == "TRUE" || p == "FALSE" {
            return None;
        }
        if out.iter().any(|e| e == p) {
            // Duplicate members: not a clean model value set.
            return None;
        }
        out.push(p.to_string());
    }
    if out.is_empty() {
        None
    } else {
        Some(out)
    }
}

/// A plain TLA+ identifier: alphabetic/underscore start, alphanumeric/underscore rest.
fn is_plain_identifier(s: &str) -> bool {
    let mut chars = s.chars();
    let Some(first) = chars.next() else {
        return false;
    };
    if !first.is_alphabetic() && first != '_' {
        return false;
    }
    chars.all(|c| c.is_alphanumeric() || c == '_')
}

/// A structurally validated symmetry candidate set (freshness and eval-level
/// validation passed; guard classification recorded for the two-phase group
/// construction in [`auto_detect_symmetry_perms`]).
struct ValidatedCandidate {
    /// Constant name of the candidate set.
    name: String,
    /// The set's member values, in set iteration (sorted) order.
    elements: Vec<Value>,
    /// Member names (the "moved" footprint of this set's permutations).
    moved: FxHashSet<Arc<str>>,
    /// Adjacent-transposition generators of the full symmetric group.
    // Symmetry-soundness struct: set at construction but not read; kept conservatively.
    #[allow(dead_code)]
    generators: Vec<FuncValue>,
    /// Guard (a): every other constant binding is fixed by this set's
    /// generators alone (per-set all-or-nothing admission).
    // Symmetry-soundness struct: set at construction but not read; kept conservatively.
    #[allow(dead_code)]
    guard_a: bool,
}

/// Generate symmetry permutations for auto-detected model value sets.
///
/// Phase 1 (per-set, all-or-nothing): for each candidate set that passes the
/// soundness guards (see module docs), generates all permutations (the full
/// symmetric group S_n) via adjacent transposition generators. Then computes
/// the group closure of the union of all surviving generators, matching
/// `Permutations(A) \cup Permutations(B)`.
///
/// Phase 2 (correlated-constant stabilizer subgroup): when some candidate set
/// is rejected ONLY because another constant binding correlates the candidate
/// sets (e.g. SlushProtocol's `HostMapping` pairing `Node`/`SlushLoopProcess`/
/// `SlushQueryProcess` rows), the per-set groups are unsound but a *subgroup*
/// of the product group can still be: exactly those product permutations that
/// fix every other constant binding element-wise. That surviving set is the
/// stabilizer of the constant environment inside the product group — closed
/// under composition by construction (verified defensively) — and each of its
/// elements is an automorphism of the state graph by the same argument as
/// phase 1 (the spec can only observe model values through the constant
/// environment, which every surviving element fixes). See
/// [`correlated_stabilizer_subgroup`].
///
/// The enabled/disabled decision (per-checker override or `TY_AUTO_SYMMETRY`)
/// and guard (c) (genuine temporal properties) are handled by the caller
/// (`run_prepare`); this function performs detection and guards (a)/(b).
///
/// Returns `(permutations, group_names)` where `group_names` lists the
/// constant names that contributed symmetry groups.
pub(crate) fn auto_detect_symmetry_perms(
    ctx: &EvalCtx,
    config: &Config,
    op_defs: &FxHashMap<String, OperatorDef>,
) -> (Vec<FuncValue>, Vec<String>) {
    // Don't auto-detect if explicit SYMMETRY is configured.
    if config.symmetry.is_some() {
        return (Vec::new(), Vec::new());
    }

    let candidates = detect_symmetric_model_value_sets(config);
    if candidates.is_empty() {
        return (Vec::new(), Vec::new());
    }

    let mut validated: Vec<ValidatedCandidate> = Vec::new();
    let mut all_generators = Vec::new();
    let mut group_names = Vec::new();

    'candidate: for (const_name, mv_names) in &candidates {
        // Look up the constant's value in the eval context to get the actual
        // model values (which have been registered with indices).
        let set_value = match ctx.lookup(const_name) {
            Some(v) => v,
            None => continue, // Constant not yet bound; skip.
        };

        let elements: Vec<Value> = match set_value.iter_set() {
            Some(iter) => iter.collect(),
            None => continue,
        };

        if elements.len() < 2 {
            continue;
        }

        // Eval-level validation (defense in depth): every member must be a
        // model value, and the member names must be exactly the identifiers
        // from the config. An `=`-form constant that evaluated to anything
        // else is not a model value set and must never seed permutations.
        let mut moved: FxHashSet<Arc<str>> = FxHashSet::default();
        for e in &elements {
            match e {
                Value::ModelValue(n) => {
                    moved.insert(n.clone().into());
                }
                _ => continue 'candidate,
            }
        }
        if moved.len() != mv_names.len() || !mv_names.iter().all(|n| moved.contains(n.as_str())) {
            continue;
        }

        // Freshness vs the spec text: if a member name collides with an
        // operator definition, the spec can reference that member directly
        // (and asymmetrically). Never permute such members.
        if mv_names.iter().any(|n| op_defs.contains_key(n)) {
            continue;
        }

        // Generate adjacent transpositions: (e[0] e[1]), (e[1] e[2]), ...
        // These generate the full symmetric group S_n. Invariance of a value
        // under all generators implies invariance under the generated group.
        let mut generators = Vec::new();
        for window in elements.windows(2) {
            let a = &window[0];
            let b = &window[1];
            // Build a function that swaps a <-> b and fixes everything else.
            let mut entries: Vec<(Value, Value)> = elements
                .iter()
                .map(|e| {
                    if e == a {
                        (e.clone(), b.clone())
                    } else if e == b {
                        (e.clone(), a.clone())
                    } else {
                        (e.clone(), e.clone())
                    }
                })
                .collect();
            entries.sort_by(|x, y| x.0.cmp(&y.0));
            generators.push(FuncValue::from_sorted_entries(entries));
        }

        // Guard (a): every other constant binding must be fixed by the group.
        let guard_a = constant_environment_invariant(ctx, config, const_name, &moved, &generators);

        // Guard (b): the candidate must not feed a bounded CHOOSE domain or an
        // order/randomness-sensitive builtin anywhere in the spec.
        // (Evaluated lazily after guard (a), preserving the original phase-1
        // control flow; phase 2 re-evaluates a relaxed variant per candidate.)
        if guard_a && !spec_uses_candidate_asymmetrically(op_defs, const_name) {
            all_generators.extend(generators.clone());
            group_names.push(const_name.clone());
        }

        validated.push(ValidatedCandidate {
            name: const_name.clone(),
            elements,
            moved,
            generators,
            guard_a,
        });
    }

    // Phase 1: closure of the union of per-set groups (shipped behavior).
    // Normalized to one entry per semantic group element (identity dropped).
    let phase1_perms = if all_generators.is_empty() {
        Vec::new()
    } else {
        super::symmetry_perms::normalize_perm_group(perm_group_closure(&all_generators))
    };

    // Phase 2: correlated-constant stabilizer subgroup. Only attempted when at
    // least one validated candidate was NOT admitted by phase 1 — otherwise the
    // stabilizer provably equals the phase-1 product group and the (byte
    // identical) phase-1 result is preferred.
    if validated.len() > group_names.len() {
        if let Some((stab_perms, stab_names)) =
            correlated_stabilizer_subgroup(ctx, config, op_defs, &validated)
        {
            // Compare semantic group orders (both lists are normalized: one
            // entry per non-identity group element). The stabilizer always
            // contains the phase-1 product, so strictly-greater means strictly
            // more reduction potential.
            let stab_perms = super::symmetry_perms::normalize_perm_group(stab_perms);
            if stab_perms.len() > phase1_perms.len() {
                eprintln!(
                    "Symmetry: correlated-constant stabilizer subgroup of order {} over \
                     model value set(s) {:?} (per-set admission found order {})",
                    stab_perms.len() + 1, // +identity
                    stab_names,
                    phase1_perms.len() + 1,
                );
                return (stab_perms, stab_names);
            }
        }
    }

    (phase1_perms, group_names)
}

/// Compute the group closure of a set of permutation generators using the same
/// frontier algorithm as `symmetry_perms.rs`. The result includes the identity
/// (generators are involutions, so `g ∘ g = id` enters the closure).
fn perm_group_closure(all_generators: &[FuncValue]) -> Vec<FuncValue> {
    use std::collections::BTreeSet;
    #[allow(clippy::mutable_key_type)]
    let mut seen_set: BTreeSet<FuncValue> = all_generators.iter().cloned().collect();
    let mut seen_vec: Vec<FuncValue> = seen_set.iter().cloned().collect();
    let mut frontier_start = 0;

    loop {
        let frontier_end = seen_vec.len();
        if frontier_start == frontier_end {
            break;
        }
        for idx in frontier_start..frontier_end {
            let elem = seen_vec[idx].clone();
            for gen in all_generators {
                let composed = gen.compose_perm(&elem);
                if seen_set.insert(composed.clone()) {
                    seen_vec.push(composed);
                }
            }
        }
        frontier_start = frontier_end;
    }

    seen_vec
}

// =============================================================================
// Phase 2: correlated-constant stabilizer subgroup
// =============================================================================

/// Maximum product-group order Phase 2 will enumerate. The stabilizer is found
/// by filtering every product element, so enumeration must stay cheap. Covers
/// e.g. S3×S3×S3 (216, SlushProtocol) and S5×S4 (2880) but not S8 (40320).
const STABILIZER_PRODUCT_BUDGET: usize = 20_160;

/// Structural cost cap on the stabilizer computation: product order × the
/// total footprint (node count) of the moved-member-bearing constant bindings
/// that must be permute-checked per element.
const STABILIZER_WORK_BUDGET: usize = 5_000_000;

/// Benefit heuristic: maximum surviving subgroup order to ENGAGE. Per-state
/// canonicalization cost scales linearly with group order while the state-fold
/// benefit is bounded by the orbit sizes. Measured bracket on this corpus:
/// order-24 groups (MCKVSSafetySmall) are a clear net win, while canonicalizing
/// over the (unsound) 216-element naive product on SlushSmall measured net
/// SLOWER than no reduction (16.5s vs 13.5s). 64 sits between those points.
const STABILIZER_GROUP_CAP: usize = 64;

/// Compute the correlated-constant stabilizer subgroup of the product group
/// over the validated candidate sets.
///
/// When per-set admission (phase 1) fails because a constant binding
/// *correlates* several candidate sets (e.g. SlushProtocol's
/// `HostMapping = {{n1,l1,q1}, {n2,l2,q2}, {n3,l3,q3}}`), the per-set
/// symmetric groups are individually unsound — a transposition of `n1`/`n2`
/// alone maps `HostMapping` to a different value. But product permutations
/// that act *consistently* across the correlated sets (here the diagonal
/// σ ∈ S3 applied to all three sets at once) map each row of `HostMapping`
/// onto another row, fixing the binding setwise. The set of ALL product
/// elements that fix every other constant binding is the stabilizer of the
/// constant environment: it contains the identity, and is closed under
/// composition (if `g` and `h` fix every binding, so does `g∘h`) — a subgroup
/// by finiteness. Every element is then a state-graph automorphism by exactly
/// the phase-1 argument: model values are uninterpreted atoms, so the spec's
/// semantics can distinguish them only through the constant environment, which
/// every surviving element fixes (member freshness and guard (b) exclude
/// direct naming and order-sensitive observation).
///
/// Guard (b) is applied in a RELAXED form here: a bounded `CHOOSE` over the
/// candidate is permitted when it occurs inside the body of a *precomputed
/// constant-level zero-arity operator* (e.g. SlushProtocol's
/// `HostOf[pid \in ...] == CHOOSE n \in Node : ...`). Such an operator is
/// evaluated once against the constant environment and its concrete VALUE is
/// verified element-wise below — re-evaluation is deterministic over the same
/// constant inputs, so order sensitivity cannot distinguish orbit members.
/// Order/randomness-sensitive builtins remain rejected everywhere (their
/// re-evaluation is NOT deterministic).
///
/// Fail-closed throughout: any unverifiable binding, overlapping candidate
/// supports, or busted budget returns `None` (callers fall back to phase 1).
fn correlated_stabilizer_subgroup(
    ctx: &EvalCtx,
    config: &Config,
    op_defs: &FxHashMap<String, OperatorDef>,
    validated: &[ValidatedCandidate],
) -> Option<(Vec<FuncValue>, Vec<String>)> {
    // Names of precomputed constant-level zero-arity operators (plus promoted
    // config constants — harmless to include: config constants have no body).
    let precomputed_op_names: FxHashSet<Arc<str>> = ctx
        .precomputed_constants()
        .keys()
        .map(|id| tla_core::name_intern::resolve_name_id(*id))
        .collect();

    // Participants: validated candidates passing the relaxed guard (b).
    let participants: Vec<&ValidatedCandidate> = validated
        .iter()
        .filter(|v| {
            !spec_uses_candidate_asymmetrically_relaxed(op_defs, &v.name, &precomputed_op_names)
        })
        .collect();
    if participants.is_empty() {
        return None;
    }

    // The direct-product construction requires pairwise-disjoint supports.
    let mut moved_all: FxHashSet<Arc<str>> = FxHashSet::default();
    for p in &participants {
        for name in &p.moved {
            if !moved_all.insert(name.clone()) {
                return None;
            }
        }
    }

    // Structural budget on the product order.
    let mut product_order: usize = 1;
    for p in &participants {
        product_order = product_order.checked_mul(factorial(p.elements.len())?)?;
        if product_order > STABILIZER_PRODUCT_BUDGET {
            return None;
        }
    }
    if product_order < 2 {
        return None;
    }

    // Pre-classify the constant environment ONCE (perm-independent):
    //   - bindings whose model-value footprint is disjoint from every moved
    //     member are fixed by all product elements — skipped;
    //   - bindings mentioning moved members are concretized (lazy enumerable
    //     variants materialized so permute/compare is faithful) and checked
    //     per product element below;
    //   - anything unverifiable fails closed (no stabilizer).
    let participant_names: FxHashSet<&str> = participants.iter().map(|p| p.name.as_str()).collect();
    let mut to_check: Vec<Value> = Vec::new();
    let mut total_footprint_nodes: usize = 0;
    for (name_id, value) in ctx.precomputed_constants() {
        let name = tla_core::name_intern::resolve_name_id(*name_id);
        // Each participating set itself is mapped onto itself by construction.
        if participant_names.contains(name.as_ref()) {
            continue;
        }
        // Auto-registered member self-binding (`m -> @m`), unless pinned as a
        // real config constant (same rule as guard (a)).
        if moved_all.contains(name.as_ref())
            && matches!(value, Value::ModelValue(n) if n.as_ref() == name.as_ref())
            && !config.constants.contains_key(name.as_ref())
        {
            continue;
        }
        classify_binding_for_stabilizer(
            value,
            &moved_all,
            &mut to_check,
            &mut total_footprint_nodes,
        )?;
    }
    for assigns in config.module_assignments.values() {
        for value_str in assigns.values() {
            let v = crate::constants::parse_constant_value(value_str).ok()?;
            classify_binding_for_stabilizer(
                &v,
                &moved_all,
                &mut to_check,
                &mut total_footprint_nodes,
            )?;
        }
    }

    // Structural cost cap on stabilizer computation itself.
    if product_order.saturating_mul(total_footprint_nodes.max(1)) > STABILIZER_WORK_BUDGET {
        return None;
    }

    // Enumerate the product group; keep the stabilizer of the environment.
    let per_set_perms: Vec<Vec<Vec<usize>>> = participants
        .iter()
        .map(|p| index_permutations(p.elements.len()))
        .collect();
    let mut surviving: Vec<FuncValue> = Vec::new();
    let mut combo = vec![0usize; participants.len()];
    'product: loop {
        // Build the product element for this combo (disjoint supports merge).
        let mut entries: Vec<(Value, Value)> = Vec::new();
        for (pi, p) in participants.iter().enumerate() {
            let perm = &per_set_perms[pi][combo[pi]];
            for (i, e) in p.elements.iter().enumerate() {
                entries.push((e.clone(), p.elements[perm[i]].clone()));
            }
        }
        entries.sort_by(|x, y| x.0.cmp(&y.0));
        let g = FuncValue::from_sorted_entries(entries);
        if to_check.iter().all(|v| v.permute(&g) == *v) {
            surviving.push(g);
        }
        // Odometer increment over the per-set permutation indices.
        for (pi, slot) in combo.iter_mut().enumerate() {
            *slot += 1;
            if *slot < per_set_perms[pi].len() {
                continue 'product;
            }
            *slot = 0;
        }
        break;
    }

    // Non-trivial? (The identity always survives.)
    if surviving.len() < 2 {
        return None;
    }
    // Benefit heuristic (structural): engage only below the group-order cap.
    if surviving.len() > STABILIZER_GROUP_CAP {
        return None;
    }
    // Defensive closure verification: the survivors are mathematically closed
    // under composition (a stabilizer), but verify against latent permute or
    // compose bugs — fail closed rather than reduce with a non-group.
    {
        use std::collections::BTreeSet;
        #[allow(clippy::mutable_key_type)]
        let set: BTreeSet<FuncValue> = surviving.iter().cloned().collect();
        for g in &surviving {
            for h in &surviving {
                if !set.contains(&g.compose_perm(h)) {
                    return None;
                }
            }
        }
    }

    // Report the sets the subgroup actually moves.
    let names: Vec<String> = participants
        .iter()
        .filter(|p| {
            surviving.iter().any(|g| {
                p.elements
                    .iter()
                    .any(|e| g.apply(e).is_some_and(|m| m != e))
            })
        })
        .map(|p| p.name.clone())
        .collect();
    if names.is_empty() {
        return None;
    }

    Some((surviving, names))
}

/// Classify a constant binding for the per-element stabilizer check.
///
/// Returns `None` (fail closed) when the binding cannot be verified; otherwise
/// pushes a concretized copy onto `to_check` when the binding mentions moved
/// members (bindings with disjoint footprints are trivially fixed and skipped).
fn classify_binding_for_stabilizer(
    v: &Value,
    moved: &FxHashSet<Arc<str>>,
    to_check: &mut Vec<Value>,
    total_footprint_nodes: &mut usize,
) -> Option<()> {
    let mut names: FxHashSet<Arc<str>> = FxHashSet::default();
    let mut budget = FOOTPRINT_BUDGET;
    if !model_value_footprint(v, &mut names, &mut budget) {
        return None; // Unverifiable → fail closed.
    }
    if names.iter().all(|n| !moved.contains(n)) {
        return Some(()); // Disjoint footprint: fixed by every product element.
    }
    let mut concretize_budget = FOOTPRINT_BUDGET;
    let concrete = concretize_for_invariance_check(v, &mut concretize_budget)?;
    *total_footprint_nodes += FOOTPRINT_BUDGET - budget;
    to_check.push(concrete);
    Some(())
}

/// Recursively materialize a constant value into fully concrete variants so
/// `Value::permute` transforms it faithfully and post-permutation comparison
/// is semantic. Lazy enumerable set variants (`RecordSet`, `SetCup`, ...) are
/// expanded via `iter_set()`; values that cannot be enumerated (closures, lazy
/// functions, infinite sets) return `None` (fail closed). Budget-bounded.
fn concretize_for_invariance_check(v: &Value, budget: &mut usize) -> Option<Value> {
    if *budget == 0 {
        return None;
    }
    *budget -= 1;
    match v {
        Value::Bool(_)
        | Value::SmallInt(_)
        | Value::Int(_)
        | Value::String(_)
        | Value::Interval(_)
        | Value::ModelValue(_) => Some(v.clone()),
        Value::Set(s) => {
            let elems: Option<Vec<Value>> = s
                .as_ref()
                .into_iter()
                .map(|e| concretize_for_invariance_check(e, budget))
                .collect();
            Some(Value::set(elems?))
        }
        Value::Seq(s) => {
            let elems: Option<Vec<Value>> = s
                .iter()
                .map(|e| concretize_for_invariance_check(e, budget))
                .collect();
            Some(Value::seq(elems?))
        }
        Value::Tuple(t) => {
            let elems: Option<Vec<Value>> = t
                .iter()
                .map(|e| concretize_for_invariance_check(e, budget))
                .collect();
            Some(Value::tuple(elems?))
        }
        Value::Record(r) => {
            let fields: Option<crate::value::RecordValue> = r
                .iter()
                .map(|(k, e)| concretize_for_invariance_check(e, budget).map(|c| (k, c)))
                .collect();
            Some(Value::Record(fields?))
        }
        Value::Func(f) => {
            let entries: Option<Vec<(Value, Value)>> = f
                .mapping_iter()
                .map(|(k, val)| {
                    let ck = concretize_for_invariance_check(k, budget)?;
                    let cv = concretize_for_invariance_check(val, budget)?;
                    Some((ck, cv))
                })
                .collect();
            let mut entries = entries?;
            // Concretizing a key can change its variant and therefore its sort
            // position; re-sort to keep the FuncValue invariant.
            entries.sort_by(|a, b| a.0.cmp(&b.0));
            Some(Value::Func(Rp::new(FuncValue::from_sorted_entries(
                entries,
            ))))
        }
        Value::IntFunc(f) => {
            if f.values().iter().all(fully_concrete) {
                Some(v.clone())
            } else {
                None // Rebuilding IntFunc internals is not exposed; fail closed.
            }
        }
        other => {
            // Lazy enumerable set variants: materialize within budget.
            let iter = other.iter_set()?;
            let mut elems = Vec::new();
            for e in iter {
                if *budget == 0 {
                    return None;
                }
                elems.push(concretize_for_invariance_check(&e, budget)?);
            }
            Some(Value::set(elems))
        }
    }
}

/// `n!` with overflow checking.
fn factorial(n: usize) -> Option<usize> {
    let mut acc: usize = 1;
    for k in 2..=n {
        acc = acc.checked_mul(k)?;
    }
    Some(acc)
}

/// All permutations of `0..n` in lexicographic order (deterministic).
fn index_permutations(n: usize) -> Vec<Vec<usize>> {
    let mut idx: Vec<usize> = (0..n).collect();
    let mut out = vec![idx.clone()];
    while next_index_permutation(&mut idx) {
        out.push(idx.clone());
    }
    out
}

/// Advance `idx` to the next lexicographic permutation; false when exhausted.
fn next_index_permutation(idx: &mut [usize]) -> bool {
    if idx.len() < 2 {
        return false;
    }
    // Find the longest non-increasing suffix.
    let mut i = idx.len() - 1;
    while i > 0 && idx[i - 1] >= idx[i] {
        i -= 1;
    }
    if i == 0 {
        return false;
    }
    // Pivot is idx[i-1]; find rightmost element greater than the pivot.
    let mut j = idx.len() - 1;
    while idx[j] <= idx[i - 1] {
        j -= 1;
    }
    idx.swap(i - 1, j);
    idx[i..].reverse();
    true
}

// =============================================================================
// Guard (a): constant-environment invariance
// =============================================================================

/// Budget for enumerating values while computing model-value footprints.
/// Exceeding it makes the value unverifiable, which fails closed.
const FOOTPRINT_BUDGET: usize = 100_000;

/// Check that every constant binding other than the candidate set itself is
/// fixed (element-wise) by all of the group's generators.
///
/// Coverage:
/// - `ctx.precomputed_constants()` — config constants AND precomputed
///   zero-arity constant operators (so asymmetric *derived* constants, e.g. a
///   randomly chosen structure over the members, are caught at value level).
/// - `config.module_assignments` — module-scoped `X = [M] v` bindings.
///
/// The auto-registered self-bindings of the set's own members (`m -> @m`,
/// created so the *config* can reference them) are skipped — unless the member
/// is also a real config constant, in which case the spec can name it and the
/// binding correctly pins the member (dropping the group).
fn constant_environment_invariant(
    ctx: &EvalCtx,
    config: &Config,
    const_name: &str,
    moved: &FxHashSet<Arc<str>>,
    generators: &[FuncValue],
) -> bool {
    for (name_id, value) in ctx.precomputed_constants() {
        let name = tla_core::name_intern::resolve_name_id(*name_id);
        // The candidate set itself is invariant under its own permutations.
        if name.as_ref() == const_name {
            continue;
        }
        // Skip the automatic member self-binding (`m -> @m`) unless the member
        // is itself a config constant (Toolbox `m = m`), which pins it.
        if moved.contains(name.as_ref())
            && matches!(value, Value::ModelValue(n) if n.as_ref() == name.as_ref())
            && !config.constants.contains_key(name.as_ref())
        {
            continue;
        }
        if !binding_invariant_under(value, generators, moved) {
            return false;
        }
    }

    // Module-scoped assignments can bind model values too (`X = [M] mv`).
    for assigns in config.module_assignments.values() {
        for value_str in assigns.values() {
            match crate::constants::parse_constant_value(value_str) {
                Ok(v) => {
                    if !binding_invariant_under(&v, generators, moved) {
                        return false;
                    }
                }
                // Unparseable here would have failed constant binding already;
                // fail closed regardless.
                Err(_) => return false,
            }
        }
    }

    true
}

/// Decide whether `v` is provably fixed by every generator.
///
/// Three-step, fail-closed:
/// 1. Compute the model-value footprint of `v` (enumerating lazy sets within
///    a budget). If the footprint cannot be completed, the value is
///    unverifiable → `false`.
/// 2. If the footprint is disjoint from the moved members, every generator
///    fixes `v` trivially → `true`.
/// 3. Otherwise `v` mentions moved members: verify `permute(g)(v) == v` for
///    each generator. This requires `v` to be fully concrete (the variants
///    `Value::permute` handles faithfully); otherwise → `false`.
fn binding_invariant_under(
    v: &Value,
    generators: &[FuncValue],
    moved: &FxHashSet<Arc<str>>,
) -> bool {
    let mut names: FxHashSet<Arc<str>> = FxHashSet::default();
    let mut budget = FOOTPRINT_BUDGET;
    if !model_value_footprint(v, &mut names, &mut budget) {
        return false; // Unverifiable → fail closed.
    }
    if names.iter().all(|n| !moved.contains(n)) {
        return true; // No moved member occurs in the value.
    }
    if !fully_concrete(v) {
        return false; // permute() can't faithfully transform it → fail closed.
    }
    generators.iter().all(|g| v.permute(g) == *v)
}

/// Collect all model-value names occurring in `v`.
///
/// Returns `true` if the walk is complete (every model value in the semantic
/// value was seen). Lazy set variants are enumerated via `iter_set()` within
/// the budget; values that cannot be enumerated (closures, lazy functions,
/// infinite sets) return `false`.
fn model_value_footprint(v: &Value, out: &mut FxHashSet<Arc<str>>, budget: &mut usize) -> bool {
    if *budget == 0 {
        return false;
    }
    *budget -= 1;
    match v {
        Value::Bool(_)
        | Value::SmallInt(_)
        | Value::Int(_)
        | Value::String(_)
        | Value::Interval(_) => true,
        Value::ModelValue(n) => {
            out.insert(n.clone().into());
            true
        }
        Value::Set(s) => {
            for e in s.as_ref() {
                if !model_value_footprint(e, out, budget) {
                    return false;
                }
            }
            true
        }
        Value::Seq(s) => {
            for e in s.iter() {
                if !model_value_footprint(e, out, budget) {
                    return false;
                }
            }
            true
        }
        Value::Tuple(t) => {
            for e in t.iter() {
                if !model_value_footprint(e, out, budget) {
                    return false;
                }
            }
            true
        }
        Value::Record(r) => {
            for (_, e) in r.iter() {
                if !model_value_footprint(e, out, budget) {
                    return false;
                }
            }
            true
        }
        Value::Func(f) => {
            for (k, val) in f.mapping_iter() {
                if !model_value_footprint(k, out, budget)
                    || !model_value_footprint(val, out, budget)
                {
                    return false;
                }
            }
            true
        }
        Value::IntFunc(f) => {
            for e in f.values() {
                if !model_value_footprint(e, out, budget) {
                    return false;
                }
            }
            true
        }
        other => {
            // Lazy set variants: enumerate within budget.
            match other.iter_set() {
                Some(iter) => {
                    for e in iter {
                        if *budget == 0 {
                            return false;
                        }
                        if !model_value_footprint(&e, out, budget) {
                            return false;
                        }
                    }
                    true
                }
                None => false,
            }
        }
    }
}

/// Whether `v` consists entirely of variants that `Value::permute` transforms
/// faithfully AND that compare structurally after permutation. Lazy variants
/// are excluded: `permute` passes them through unchanged, and cross-variant
/// equality (lazy vs materialized) is not semantic.
fn fully_concrete(v: &Value) -> bool {
    match v {
        Value::Bool(_)
        | Value::SmallInt(_)
        | Value::Int(_)
        | Value::String(_)
        | Value::Interval(_)
        | Value::ModelValue(_) => true,
        Value::Set(s) => s.as_ref().into_iter().all(fully_concrete),
        Value::Seq(s) => s.iter().all(fully_concrete),
        Value::Tuple(t) => t.iter().all(fully_concrete),
        Value::Record(r) => r.iter().all(|(_, e)| fully_concrete(e)),
        Value::Func(f) => f
            .mapping_iter()
            .all(|(k, val)| fully_concrete(k) && fully_concrete(val)),
        Value::IntFunc(f) => f.values().iter().all(fully_concrete),
        _ => false,
    }
}

// =============================================================================
// Guard (b): order-sensitive use of the candidate in the spec
// =============================================================================

/// Builtins whose results depend on enumeration order or randomness. Feeding a
/// symmetry candidate into one of these breaks symmetry exactly as a bounded
/// CHOOSE does (TLC documents the same restriction for declared symmetry sets).
const ORDER_SENSITIVE_BUILTINS: &[&str] = &[
    "RandomElement",
    "RandomSubset",
    "RandomSetOfSubsets",
    "SortSeq",
    "SetToSeq",
    "SetToSortSeq",
    "ToString",
];

/// True if any operator definition contains a bounded `CHOOSE` whose domain
/// (transitively, through operator definitions) references `candidate`, or
/// passes a `candidate`-referencing argument to an order-sensitive builtin.
///
/// Scope-conservative: identifier references are resolved by name without
/// tracking binder shadowing, so a parameter that happens to share the
/// candidate's name produces a false positive (rejecting the group), never a
/// false negative. Unbounded `CHOOSE x : P` is intentionally NOT flagged: the
/// dominant pattern `CHOOSE x : x \notin S` yields a witness outside `S`,
/// which every group element fixes.
pub(crate) fn spec_uses_candidate_asymmetrically(
    op_defs: &FxHashMap<String, OperatorDef>,
    candidate: &str,
) -> bool {
    let mut scan = AsymmetricUseScan {
        op_defs,
        candidate,
        check_choose: true,
    };
    op_defs.values().any(|def| scan.walk_expr(&def.body.node))
}

/// Relaxed variant of [`spec_uses_candidate_asymmetrically`] for the phase-2
/// stabilizer path: bounded `CHOOSE` over the candidate is permitted inside
/// the body of a precomputed constant-level zero-arity operator, because that
/// operator's concrete VALUE is separately verified element-wise by the
/// stabilizer construction (and re-evaluating a constant-level `CHOOSE` over
/// the unchanged constant environment is deterministic, yielding that same
/// verified value). Order/randomness-sensitive builtins remain rejected in
/// EVERY operator body: their re-evaluation is not deterministic, so a
/// value-level check cannot cover them.
fn spec_uses_candidate_asymmetrically_relaxed(
    op_defs: &FxHashMap<String, OperatorDef>,
    candidate: &str,
    precomputed_constant_ops: &FxHashSet<Arc<str>>,
) -> bool {
    op_defs.iter().any(|(name, def)| {
        let mut scan = AsymmetricUseScan {
            op_defs,
            candidate,
            // Skip the bounded-CHOOSE rejection only for bodies whose values
            // are precomputed (and thus covered by the element-wise check).
            check_choose: !precomputed_constant_ops.contains(name.as_str()),
        };
        scan.walk_expr(&def.body.node)
    })
}

struct AsymmetricUseScan<'a> {
    op_defs: &'a FxHashMap<String, OperatorDef>,
    candidate: &'a str,
    /// Whether bounded-CHOOSE domains are flagged (strict mode). The relaxed
    /// phase-2 scan disables this for precomputed constant-operator bodies;
    /// order-sensitive builtins are flagged regardless.
    check_choose: bool,
}

impl ExprVisitor for AsymmetricUseScan<'_> {
    type Output = bool;

    fn visit_node(&mut self, expr: &Expr) -> Option<bool> {
        if self.check_choose {
            if let Expr::Choose(bv, _) = expr {
                if let Some(dom) = &bv.domain {
                    if expr_references_transitively(self.op_defs, &dom.node, self.candidate) {
                        return Some(true);
                    }
                }
            }
        }
        None // Continue default traversal (nested CHOOSEs, builtin uses).
    }

    fn visit_apply(&mut self, op_expr: &Spanned<Expr>, args: &[Spanned<Expr>]) -> Option<bool> {
        let op_name = match &op_expr.node {
            Expr::Ident(n, _) => Some(n.as_str()),
            Expr::OpRef(n) => Some(n.as_str()),
            _ => None,
        };
        if let Some(name) = op_name {
            if ORDER_SENSITIVE_BUILTINS.contains(&name)
                && args
                    .iter()
                    .any(|a| expr_references_transitively(self.op_defs, &a.node, self.candidate))
            {
                return Some(true);
            }
        }
        None // Continue default traversal.
    }
}

/// True if `expr` references `target`, following operator definitions
/// transitively (with a cycle guard).
fn expr_references_transitively(
    op_defs: &FxHashMap<String, OperatorDef>,
    expr: &Expr,
    target: &str,
) -> bool {
    let mut scan = RefScan {
        op_defs,
        target,
        visiting: FxHashSet::default(),
    };
    scan.walk_expr(expr)
}

struct RefScan<'a> {
    op_defs: &'a FxHashMap<String, OperatorDef>,
    target: &'a str,
    visiting: FxHashSet<String>,
}

impl ExprVisitor for RefScan<'_> {
    type Output = bool;

    fn visit_node(&mut self, expr: &Expr) -> Option<bool> {
        let name = match expr {
            Expr::Ident(n, _) => n.as_str(),
            Expr::OpRef(n) => n.as_str(),
            _ => return None,
        };
        if name == self.target {
            return Some(true);
        }
        if let Some(def) = self.op_defs.get(name) {
            if self.visiting.insert(name.to_string()) {
                let body = def.body.node.clone();
                return Some(self.walk_expr(&body));
            }
        }
        Some(false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ConstantValue;

    /// Env-value parsing is covered as a pure function so tests never have to
    /// mutate the process-global environment (which races with concurrent tests).
    ///
    /// Auto-symmetry is ON by default; `0`/`false` are the kill switch.
    #[test]
    fn test_auto_symmetry_enabled_from_value() {
        assert!(auto_symmetry_enabled_from_value(Some("1")));
        assert!(auto_symmetry_enabled_from_value(Some("true")));
        assert!(!auto_symmetry_enabled_from_value(Some("0")));
        assert!(!auto_symmetry_enabled_from_value(Some("false")));
        // Default ON: unset and unrecognized values enable.
        assert!(auto_symmetry_enabled_from_value(None));
        assert!(auto_symmetry_enabled_from_value(Some("")));
        assert!(auto_symmetry_enabled_from_value(Some("yes")));
    }

    /// `TY_MATCH_DECLARED_SYMMETRY` is OFF by default (sound); `1`/`true` enable
    /// the benchmark-parity liveness-symmetry match.
    #[test]
    fn test_match_declared_symmetry_for_liveness_from_value() {
        assert!(match_declared_symmetry_for_liveness_from_value(Some("1")));
        assert!(match_declared_symmetry_for_liveness_from_value(Some(
            "true"
        )));
        // Default OFF: unset, kill-switch values, and unrecognized values stay off.
        assert!(!match_declared_symmetry_for_liveness_from_value(None));
        assert!(!match_declared_symmetry_for_liveness_from_value(Some("0")));
        assert!(!match_declared_symmetry_for_liveness_from_value(Some(
            "false"
        )));
        assert!(!match_declared_symmetry_for_liveness_from_value(Some("")));
        assert!(!match_declared_symmetry_for_liveness_from_value(Some(
            "yes"
        )));
    }

    #[test]
    fn test_detect_symmetric_model_value_sets_basic() {
        let mut config = Config::default();
        config.constants.insert(
            "Procs".to_string(),
            ConstantValue::ModelValueSet(vec![
                "p1".to_string(),
                "p2".to_string(),
                "p3".to_string(),
            ]),
        );
        config
            .constants
            .insert("N".to_string(), ConstantValue::Value("3".to_string()));

        let candidates = detect_symmetric_model_value_sets(&config);
        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].0, "Procs");
        assert_eq!(candidates[0].1.len(), 3);
    }

    #[test]
    fn test_detect_symmetric_excludes_small_sets() {
        let mut config = Config::default();
        config.constants.insert(
            "Single".to_string(),
            ConstantValue::ModelValueSet(vec!["s1".to_string()]),
        );

        let candidates = detect_symmetric_model_value_sets(&config);
        assert!(
            candidates.is_empty(),
            "single-element model value sets should be excluded"
        );
    }

    #[test]
    fn test_detect_symmetric_multiple_groups() {
        let mut config = Config::default();
        config.constants.insert(
            "Acceptors".to_string(),
            ConstantValue::ModelValueSet(vec![
                "a1".to_string(),
                "a2".to_string(),
                "a3".to_string(),
            ]),
        );
        config.constants.insert(
            "Values".to_string(),
            ConstantValue::ModelValueSet(vec!["v1".to_string(), "v2".to_string()]),
        );

        let candidates = detect_symmetric_model_value_sets(&config);
        assert_eq!(candidates.len(), 2);
        // Should be sorted by name.
        assert_eq!(candidates[0].0, "Acceptors");
        assert_eq!(candidates[1].0, "Values");
    }

    #[test]
    fn test_detect_symmetric_skips_when_explicit_symmetry() {
        let mut config = Config {
            symmetry: Some("Sym".to_string()),
            ..Default::default()
        };
        config.constants.insert(
            "Procs".to_string(),
            ConstantValue::ModelValueSet(vec!["p1".to_string(), "p2".to_string()]),
        );

        // When SYMMETRY is explicitly set, auto-detection should still find
        // candidates (the filtering happens at a higher level in the checker).
        // Here we verify the detection function itself returns the model value sets.
        let candidates = detect_symmetric_model_value_sets(&config);
        assert_eq!(
            candidates.len(),
            1,
            "should detect Procs as a symmetric model value set even with explicit SYMMETRY"
        );
        assert_eq!(candidates[0].0, "Procs");
    }

    // === `=`-form classification (Name = {m1, m2}) ===

    #[test]
    fn test_detect_eq_form_model_value_set() {
        let mut config = Config::default();
        config.constants.insert(
            "Key".to_string(),
            ConstantValue::Value("{k1, k2}".to_string()),
        );
        config.constants.insert(
            "TxId".to_string(),
            ConstantValue::Value("{t1, t2, t3}".to_string()),
        );
        config.constants.insert(
            "NoVal".to_string(),
            ConstantValue::Value("NoVal".to_string()),
        );
        config
            .constants
            .insert("MaxN".to_string(), ConstantValue::Value("10".to_string()));

        let candidates = detect_symmetric_model_value_sets(&config);
        assert_eq!(candidates.len(), 2, "Key and TxId should be candidates");
        assert_eq!(candidates[0].0, "Key");
        assert_eq!(candidates[0].1, vec!["k1", "k2"]);
        assert_eq!(candidates[1].0, "TxId");
        assert_eq!(candidates[1].1.len(), 3);
    }

    #[test]
    fn test_detect_eq_form_rejects_non_identifier_members() {
        let mut config = Config::default();
        // Integers: not model values.
        config.constants.insert(
            "Ints".to_string(),
            ConstantValue::Value("{1, 2}".to_string()),
        );
        // Booleans: not model values.
        config.constants.insert(
            "Bools".to_string(),
            ConstantValue::Value("{TRUE, FALSE}".to_string()),
        );
        // Strings: not model values.
        config.constants.insert(
            "Strs".to_string(),
            ConstantValue::Value("{\"a\", \"b\"}".to_string()),
        );
        // Nested sets: not a flat model value set (SlushSmall's HostMapping).
        config.constants.insert(
            "HostMapping".to_string(),
            ConstantValue::Value("{{n1, l1}, {n2, l2}}".to_string()),
        );
        // Duplicates: not a clean model value set.
        config.constants.insert(
            "Dup".to_string(),
            ConstantValue::Value("{a, a}".to_string()),
        );

        let candidates = detect_symmetric_model_value_sets(&config);
        assert!(
            candidates.is_empty(),
            "no =-form candidates expected, got {candidates:?}"
        );
    }

    #[test]
    fn test_detect_eq_form_rejects_members_bound_elsewhere() {
        let mut config = Config::default();
        // Toolbox form: `k1 = k1` makes k1 a config constant the spec can name.
        config.constants.insert(
            "Key".to_string(),
            ConstantValue::Value("{k1, k2}".to_string()),
        );
        config
            .constants
            .insert("k1".to_string(), ConstantValue::Value("k1".to_string()));

        let candidates = detect_symmetric_model_value_sets(&config);
        assert!(
            candidates.is_empty(),
            "members that are config constants themselves must disqualify the set"
        );
    }

    // === Guard (b): order-sensitive use scanning ===

    fn op_defs_from_module(src: &str) -> FxHashMap<String, OperatorDef> {
        use tla_core::ast::Unit;
        use tla_core::{lower, parse_to_syntax_tree, FileId};
        let tree = parse_to_syntax_tree(src);
        let lowered = lower(FileId(0), &tree);
        let module = lowered.module.expect("module should lower");
        let mut out = FxHashMap::default();
        for unit in &module.units {
            if let Unit::Operator(op) = &unit.node {
                out.insert(op.name.node.clone(), op.clone());
            }
        }
        out
    }

    #[test]
    fn test_guard_b_bounded_choose_over_candidate() {
        let op_defs = op_defs_from_module(
            r#"
---- MODULE GuardB1 ----
CONSTANT Hash
Pick == CHOOSE h \in Hash : TRUE
====
"#,
        );
        assert!(
            spec_uses_candidate_asymmetrically(&op_defs, "Hash"),
            "bounded CHOOSE directly over the candidate must be flagged"
        );
    }

    #[test]
    fn test_guard_b_bounded_choose_transitive() {
        let op_defs = op_defs_from_module(
            r#"
---- MODULE GuardB2 ----
CONSTANT Vals
AllVals == Vals \cup {99}
Pick == CHOOSE v \in AllVals : TRUE
====
"#,
        );
        assert!(
            spec_uses_candidate_asymmetrically(&op_defs, "Vals"),
            "bounded CHOOSE over a derived set referencing the candidate must be flagged"
        );
    }

    #[test]
    fn test_guard_b_unbounded_notin_choose_allowed() {
        // The classic fresh-witness pattern (`NoVal == CHOOSE v : v \notin Val`)
        // is sound: the witness is outside the set and fixed by every perm.
        let op_defs = op_defs_from_module(
            r#"
---- MODULE GuardB3 ----
CONSTANT Val
NoVal == CHOOSE v : v \notin Val
====
"#,
        );
        assert!(
            !spec_uses_candidate_asymmetrically(&op_defs, "Val"),
            "unbounded CHOOSE must not be flagged"
        );
    }

    #[test]
    fn test_guard_b_choose_over_unrelated_set_allowed() {
        let op_defs = op_defs_from_module(
            r#"
---- MODULE GuardB4 ----
CONSTANT Readers
VARIABLE read
Range(f) == {f[x] : x \in DOMAIN f}
MinRead == CHOOSE min \in Range(read) : \A r \in Readers : min <= read[r]
====
"#,
        );
        assert!(
            !spec_uses_candidate_asymmetrically(&op_defs, "Readers"),
            "bounded CHOOSE over a state-derived integer set must not be flagged \
             (the Disruptor pattern — the predicate references the candidate, \
              but the domain does not)"
        );
    }

    #[test]
    fn test_guard_b_random_element_over_candidate() {
        let op_defs = op_defs_from_module(
            r#"
---- MODULE GuardB5 ----
CONSTANT Nodes
Edges == UNION { {{n, m} : m \in RandomElement(SUBSET (Nodes \ {n}))} : n \in Nodes }
====
"#,
        );
        assert!(
            spec_uses_candidate_asymmetrically(&op_defs, "Nodes"),
            "RandomElement over a candidate-derived set must be flagged \
             (the SpanTreeRandom pattern)"
        );
    }

    // === Guard (a): constant-environment invariance unit coverage ===

    fn arc(s: &str) -> Arc<str> {
        Arc::from(s)
    }

    fn transposition(elements: &[Value], a: &Value, b: &Value) -> FuncValue {
        let mut entries: Vec<(Value, Value)> = elements
            .iter()
            .map(|e| {
                if e == a {
                    (e.clone(), b.clone())
                } else if e == b {
                    (e.clone(), a.clone())
                } else {
                    (e.clone(), e.clone())
                }
            })
            .collect();
        entries.sort_by(|x, y| x.0.cmp(&y.0));
        FuncValue::from_sorted_entries(entries)
    }

    // === Phase 2 helpers: permutation enumeration and concretization ===

    #[test]
    fn test_factorial_and_index_permutations() {
        assert_eq!(factorial(0), Some(1));
        assert_eq!(factorial(1), Some(1));
        assert_eq!(factorial(3), Some(6));
        assert_eq!(factorial(7), Some(5040));

        let perms = index_permutations(3);
        assert_eq!(perms.len(), 6, "S3 has 6 elements");
        // Lexicographic order, starting from identity.
        assert_eq!(perms[0], vec![0, 1, 2]);
        assert_eq!(perms[5], vec![2, 1, 0]);
        // All distinct, all bijections.
        let distinct: std::collections::BTreeSet<_> = perms.iter().cloned().collect();
        assert_eq!(distinct.len(), 6);
        for p in &perms {
            let mut sorted = p.clone();
            sorted.sort_unstable();
            assert_eq!(sorted, vec![0, 1, 2]);
        }

        assert_eq!(index_permutations(1), vec![vec![0]]);
    }

    #[test]
    fn test_concretize_for_invariance_check_concrete_passthrough() {
        let v = Value::set(vec![Value::int(1), Value::try_model_value("cm1").unwrap()]);
        let mut budget = FOOTPRINT_BUDGET;
        let c = concretize_for_invariance_check(&v, &mut budget).expect("concrete set");
        assert_eq!(c, v);
        assert!(fully_concrete(&c));
    }

    /// The relaxed guard (b) permits a bounded CHOOSE inside a PRECOMPUTED
    /// constant operator (its value is verified element-wise), but the strict
    /// guard must keep flagging it; order-sensitive builtins stay flagged in
    /// both modes.
    #[test]
    fn test_relaxed_guard_b_precomputed_choose() {
        let op_defs = op_defs_from_module(
            r#"
---- MODULE RelaxedGuardB ----
CONSTANTS Hash, Mapping
PickOf == CHOOSE h \in Hash : h \in Mapping
====
"#,
        );
        assert!(
            spec_uses_candidate_asymmetrically(&op_defs, "Hash"),
            "strict guard must flag the bounded CHOOSE"
        );
        let mut precomputed: FxHashSet<Arc<str>> = FxHashSet::default();
        assert!(
            spec_uses_candidate_asymmetrically_relaxed(&op_defs, "Hash", &precomputed),
            "relaxed guard must still flag when the op is NOT precomputed"
        );
        precomputed.insert(Arc::from("PickOf"));
        assert!(
            !spec_uses_candidate_asymmetrically_relaxed(&op_defs, "Hash", &precomputed),
            "relaxed guard must exempt a CHOOSE inside a precomputed constant op"
        );

        // Order-sensitive builtins are never exempted.
        let op_defs2 = op_defs_from_module(
            r#"
---- MODULE RelaxedGuardB2 ----
CONSTANT Nodes
Picked == RandomElement(Nodes)
====
"#,
        );
        let mut pre2: FxHashSet<Arc<str>> = FxHashSet::default();
        pre2.insert(Arc::from("Picked"));
        assert!(
            spec_uses_candidate_asymmetrically_relaxed(&op_defs2, "Nodes", &pre2),
            "RandomElement must stay flagged even in a precomputed op body"
        );
    }

    #[test]
    fn test_guard_a_binding_invariance() {
        let m1 = Value::try_model_value("gm1").unwrap();
        let m2 = Value::try_model_value("gm2").unwrap();
        let other = Value::try_model_value("gother").unwrap();
        let elements = vec![m1.clone(), m2.clone()];
        let gens = vec![transposition(&elements, &m1, &m2)];
        let moved: FxHashSet<Arc<str>> = [arc("gm1"), arc("gm2")].into_iter().collect();

        // A scalar binding outside the group is invariant.
        assert!(binding_invariant_under(&other, &gens, &moved));
        // A scalar binding that pins a member is NOT invariant (SpanTreeRandom's Root=n1).
        assert!(!binding_invariant_under(&m1, &gens, &moved));
        // The full set itself is invariant (swap maps the set onto itself).
        let full = Value::set(vec![m1.clone(), m2.clone()]);
        assert!(binding_invariant_under(&full, &gens, &moved));
        // A proper subset containing one member is NOT invariant.
        let subset = Value::set(vec![m1.clone(), other.clone()]);
        assert!(!binding_invariant_under(&subset, &gens, &moved));
        // An int is invariant.
        assert!(binding_invariant_under(&Value::int(7), &gens, &moved));
        // An asymmetric structure over members (HostMapping-style pairing) is
        // NOT invariant: {{gm1, ga}, {gm2, gb}} maps to {{gm2, ga}, {gm1, gb}}.
        let ga = Value::try_model_value("gga").unwrap();
        let gb = Value::try_model_value("ggb").unwrap();
        let pairing = Value::set(vec![
            Value::set(vec![m1.clone(), ga]),
            Value::set(vec![m2.clone(), gb]),
        ]);
        assert!(!binding_invariant_under(&pairing, &gens, &moved));
    }
}
