// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! KERNEL-CERTIFIED REFINEMENT MAPPINGS — `Spec_impl ⇒ Spec_abs` with a tiny-kernel trust base.
//!
//! TLA+'s central idea is that implementation is implication: a lower-level spec refines a
//! higher-level one when every implementation behavior, viewed through a REFINEMENT MAPPING
//! `f`, is a behavior of the abstract spec. For the (finite, enumerable) safety part this
//! module certifies exactly that, with the CLEAN KERNEL re-evaluating the abstract predicates:
//!
//!   * **Init refinement** — for every enumerated implementation initial state `s`:
//!     `Init_abs(f(s))` reduces to `Bool.true` IN THE KERNEL.
//!   * **Step-or-stutter refinement** — for every enumerated implementation TRANSITION
//!     `(s, s')`: `Next_abs(f(s), f(s')) ∨ f(s') = f(s)` reduces to `Bool.true` in the kernel
//!     (the stutter disjunct is what `⇒ □[Next_abs]_{vars}` requires — TLA+ specs are
//!     stuttering-insensitive, so implementation steps that do not move the abstract state
//!     are permitted).
//!
//! Together (by induction over any implementation behavior, whose states all lie in the
//! enumerated reachable set): `Init_impl ∧ □[Next_impl] ⇒ (Init_abs ∧ □[Next_abs]_vars) ∘ f`
//! — the SAFETY part of refinement. Fairness/liveness conclusions are OUT of scope and the
//! labels say so.
//!
//! ## Honest trust story
//!
//! Tier: **enumerated implementation graph** (the analog of the explicit-fixpoint
//! ENUMERATOR-ASSISTED tier): completeness of the transition set `T` rests on ty's
//! enumerator; each MAPPED obligation is then decided by the kernel (and, since every
//! obligation is a `bool_true_eq` over Nat/Bool terms, independently corroborated by the
//! ck0 second checker through the standard `kernel_accepts` gate). The abstract predicates
//! are recognized into the same serializable `PredIR` the fixpoint lane uses and re-derived
//! at verify time from the ABSTRACT spec source — a cert can never claim a different
//! abstract `Init`/`Next` than the abstract spec's.
//!
//! ## Refinement mappings: projection AND derived (data refinement)
//!
//! The mapping is per-abstract-variable, and each entry is an EXACT-AFFINE expression over the
//! IMPLEMENTATION columns (current state): `abs_var ↦ Σ c_j·impl_col_j + k`.
//!
//!   * A bare `impl_col` is the **projection** special case (the v1 mapping) — the abstract
//!     variable renames one implementation column, keeping its sort and value encoding exactly.
//!   * A compound affine combination (`sum ↦ a + b`) is a **DERIVED** mapping — the canonical
//!     DATA-REFINEMENT pattern where an abstract variable is an AGGREGATE of concrete state.
//!     The derived value is computed EXACTLY (an affine sum of nonneg-`Int` columns never
//!     truncates), the result sort is `Int`, and the mapped abstract tuple then feeds the SAME
//!     kernel obligations as a projection — the kernel re-evaluates the abstract `Init`/`Next`
//!     at the derived state. Fail-closed outside the exact-affine fragment (a `-` that could go
//!     negative, `\div`/`%`, a non-`Int` column under arithmetic, a primed reference): a
//!     refinement mapping is a STATE function of the current implementation state, computed
//!     exactly, never a weaker cert.
//!
//! `INSTANCE … WITH` substitution parsing (resolving a corpus mapping from the implementation
//! module's composition) is a FOLLOW-UP (R6); this lane's surface is the `--map` affine
//! expression, and the derived-mapping ENGINE beneath it.
//!
//! Fail-closed everywhere: out-of-fragment abstract predicates, unmappable variables, sort
//! mismatches, an out-of-fragment / overflowing mapping expression, enumeration overflow, or a
//! kernel rejection each yield `None` — never a false refinement claim.

#[cfg(feature = "clean-cic")]
use std::collections::BTreeMap;
#[cfg(feature = "clean-cic")]
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

#[cfg(feature = "clean-cic")]
use crate::config::Config;
#[cfg(feature = "clean-cic")]
use crate::explicit_fixpoint_cert::value_cell_encode;
use crate::explicit_fixpoint_cert::{ColSort, PredIR, ValIR};

/// Cap on the enumerated implementation reachable set (mirrors the fixpoint lane's cap).
pub const REFINEMENT_STATE_CAP: usize = 4096;
/// Cap on the enumerated transition set (each transition contributes one kernel conjunct).
pub const REFINEMENT_TRANSITION_CAP: usize = 16384;

// Only referenced by the clean-cic-gated certify_refinement / verify_refinement_cert.
#[cfg(feature = "clean-cic")]
const SCHEMA_V1: &str = "ty.refine-cert/v1";

/// A serialized, independently re-checkable refinement certificate.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RefinementCert {
    /// Schema tag (`ty.refine-cert/v1`).
    pub schema: String,
    /// Full implementation spec module text (self-contained re-check).
    pub impl_spec_src: String,
    /// Full abstract spec module text.
    pub abs_spec_src: String,
    /// Implementation `Init` operator name.
    pub impl_init: String,
    /// Implementation `Next` operator name.
    pub impl_next: String,
    /// Abstract `Init` operator name.
    pub abs_init: String,
    /// Abstract `Next` operator name.
    pub abs_next: String,
    /// The PROJECTION part of the refinement mapping: `(abstract variable, implementation
    /// variable)` pairs, in the ABSTRACT spec's variable declaration order (a projection/
    /// renaming; values map identically, the abstract variable inherits the implementation
    /// column's sort). Abstract variables with a DERIVED mapping live in [`Self::derived_map`]
    /// instead and DO NOT appear here.
    pub var_map: Vec<(String, String)>,
    /// The DERIVED part of the refinement mapping: `(abstract variable, affine ValIR over
    /// implementation columns)` pairs — the aggregate mappings (`sum ↦ a + b`) a projection
    /// cannot express. Each `ValIR` is in the EXACT-AFFINE fragment (`Lit`/`Var`-over-`Int`/
    /// `Add`/`Mul`, no primes), so the derived value is computed exactly and its result sort is
    /// `Int`. EMPTY (and omitted from the JSON) for a pure-projection certificate, so such a
    /// certificate serializes BYTE-IDENTICALLY to the v1 projection schema.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub derived_map: Vec<(String, ValIR)>,
    /// The enumerated implementation reachable TUPLES (impl variable declaration order per
    /// tuple), CANONICALLY SORTED — the graph is independent of traversal order.
    pub impl_reachable: Vec<Vec<u64>>,
    /// Indices (into `impl_reachable`) of the implementation initial states.
    pub impl_inits: Vec<usize>,
    /// The enumerated implementation TRANSITIONS as `(from, to)` indices into
    /// `impl_reachable` — the full successor relation over the reachable set.
    pub transitions: Vec<(usize, usize)>,
    /// Implementation per-column sorts (declaration order).
    pub impl_sorts: Vec<ColSort>,
    /// The recognized ABSTRACT `Init` predicate IR (over abstract columns).
    pub abs_init_ir: PredIR,
    /// The recognized ABSTRACT `Next` predicate IR.
    pub abs_next_ir: PredIR,
    /// IMPLEMENTATION `CONSTANT` bindings (name → value), SORTED by name — needed so verify
    /// re-enumerates the implementation graph under IDENTICAL constant bindings. EMPTY (and
    /// omitted from the JSON) for a constant-free implementation, so such a certificate
    /// serializes BYTE-IDENTICALLY to the v1 schema.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub impl_constants: Vec<(String, crate::config::ConstantValue)>,
    /// ABSTRACT `CONSTANT` bindings, SORTED by name — the model-value sets threaded into the
    /// abstract-predicate recognizer (so a `\A rm \in RM`/`[RM -> …]` over a model-value domain
    /// expands EXACTLY, re-derived identically at verify time). EMPTY (and omitted) for a
    /// constant-free abstract spec — BYTE-IDENTICAL to the v1 schema.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub abs_constants: Vec<(String, crate::config::ConstantValue)>,
    /// Kernel-checked INIT-refinement leg (`Eq.refl : Eq Bool C_init Bool.true`).
    pub init_leg: Vec<u8>,
    /// Kernel-checked STEP-OR-STUTTER refinement leg.
    pub step_leg: Vec<u8>,
    /// `sha256` over the canonical body (blank during hashing).
    pub digest: String,
}

impl RefinementCert {
    fn canonical_bytes(&self) -> Vec<u8> {
        let mut c = self.clone();
        c.digest = String::new();
        serde_json::to_vec(&c).unwrap_or_default()
    }
    /// Recompute the tamper-evidence digest.
    #[must_use]
    pub fn compute_digest(&self) -> String {
        let mut h = Sha256::new();
        h.update(self.canonical_bytes());
        h.finalize().iter().map(|b| format!("{b:02x}")).collect()
    }
    /// Serialize to pretty JSON.
    #[must_use]
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).unwrap_or_default()
    }
    /// Parse from JSON.
    pub fn from_json(s: &str) -> Result<Self, String> {
        serde_json::from_str(s).map_err(|e| format!("refinement certificate parse error: {e}"))
    }
}

/// The enumerated implementation state graph: canonical tuples, initial indices, and the
/// full successor relation — everything the refinement obligations quantify over.
#[cfg(feature = "clean-cic")]
struct ImplGraph {
    /// The column names of the STORED tuples. For a fully cell-encodable implementation this is
    /// every implementation variable (the v1 full-state graph); for an implementation with a
    /// non-cell-encodable variable (a message soup) it is only the MAPPED columns (see
    /// [`ImplGraph::projected`]).
    var_names: Vec<Arc<str>>,
    sorts: Vec<ColSort>,
    states: Vec<Vec<u64>>,
    inits: Vec<usize>,
    transitions: Vec<(usize, usize)>,
    /// `true` iff this graph was built by the PROJECTED path (some implementation variable is not
    /// cell-encodable, so states are identified by their full canonical value and only the mapped
    /// columns are stored/encoded). `false` for the v1 full-state path (every column stored).
    projected: bool,
}

/// Why the FULL (all-columns) enumeration declined — distinguishes a genuinely non-cell-encodable
/// column (which the PROJECTED path may still handle) from a hard decline (caps, parse, sort
/// disagreement, missing operators) for which projection cannot help.
#[cfg(feature = "clean-cic")]
enum FullEnumFail {
    /// Some implementation variable's value is not cell-encodable (e.g. `msgs ⊆ [type,rm,…]`, a
    /// set of records). The projected path (mapped-columns-only) is attempted next.
    NonEncodable,
    /// A hard decline (cap exceeded, parse failure, per-column sort disagreement, missing
    /// Init/Next). Projection cannot recover it.
    Decline,
}

/// Enumerate the implementation reachable graph. Tries the v1 FULL-STATE path first (every column
/// cell-encoded — BYTE-IDENTICAL to v1 for every already-supported spec); if and ONLY if that
/// declines because some implementation variable is not cell-encodable (a message soup), falls
/// back to the PROJECTED path, which identifies states by their FULL canonical value (collision-
/// free) but cell-encodes ONLY the `mapped_names` columns (the mapping's implementation columns).
/// `mapped_names` is consulted ONLY on the projected path. Fail-closed everywhere.
#[cfg(feature = "clean-cic")]
fn enumerate_impl_graph(
    spec_src: &str,
    config: &Config,
    mapped_names: &std::collections::BTreeSet<Arc<str>>,
) -> Option<ImplGraph> {
    match enumerate_impl_graph_full(spec_src, config) {
        Ok(graph) => Some(graph),
        // The full path declined ONLY because a column is not cell-encodable — try projection.
        Err(FullEnumFail::NonEncodable) => {
            enumerate_impl_graph_projected(spec_src, config, mapped_names)
        }
        // A hard decline (cap/parse/sort/missing-op): projection cannot recover it, and trying it
        // would risk a DIFFERENT (smaller) graph for a spec the full path owns — decline.
        Err(FullEnumFail::Decline) => None,
    }
}

/// The v1 FULL-STATE enumeration: every implementation variable is cell-encoded and the state
/// tuple IS the identity. Logic is UNCHANGED from v1 (so every already-supported refinement cert
/// is byte-identical); the only difference is the richer [`FullEnumFail`] return that lets the
/// caller distinguish a non-cell-encodable column (recoverable by projection) from a hard decline.
#[cfg(feature = "clean-cic")]
fn enumerate_impl_graph_full(spec_src: &str, config: &Config) -> Result<ImplGraph, FullEnumFail> {
    use crate::enumerate::{
        enumerate_states_from_constraint_branches, enumerate_successors, extract_init_constraints,
    };
    use crate::eval::EvalCtx;
    use crate::state::State;
    use tla_core::ast::{Module, OperatorDef, Unit};

    let d = || FullEnumFail::Decline;
    let init_name = config.init.as_deref().ok_or_else(d)?;
    let next_name = config.next.as_deref().ok_or_else(d)?;
    let tree = tla_core::parse_to_syntax_tree(spec_src);
    let lowered = tla_core::lower(tla_core::FileId(0), &tree);
    let module = lowered.module.ok_or_else(d)?;
    let var_names: Vec<Arc<str>> = module
        .units
        .iter()
        .flat_map(|u| match &u.node {
            Unit::Variable(decls) => decls
                .iter()
                .map(|d| Arc::<str>::from(d.node.as_str()))
                .collect::<Vec<_>>(),
            _ => Vec::new(),
        })
        .collect();
    if var_names.is_empty() {
        return Err(FullEnumFail::Decline);
    }
    let find_op = |name: &str| -> Option<&OperatorDef> {
        module.units.iter().find_map(|unit| match &unit.node {
            Unit::Operator(op) if op.name.node == name => Some(op),
            _ => None,
        })
    };
    let init_def = find_op(init_name).ok_or_else(d)?;
    let next_def = find_op(next_name).ok_or_else(d)?.clone();

    let build_ctx = |module: &Module| -> Option<EvalCtx> {
        let mut ctx = EvalCtx::new();
        ctx.load_module(module);
        for v in &var_names {
            ctx.register_var(Arc::clone(v));
        }
        crate::constants::bind_constants_from_config(&mut ctx, config).ok()?;
        Some(ctx)
    };

    let mut col_sorts: Option<Vec<ColSort>> = None;
    // A non-cell-encodable column ⇒ `NonEncodable` (projection may recover); a missing variable or
    // a per-column sort DISAGREEMENT ⇒ `Decline` (a hard, non-recoverable inconsistency).
    let mut state_tuple = |s: &State| -> Result<Vec<u64>, FullEnumFail> {
        let mut tup = Vec::with_capacity(var_names.len());
        let mut sorts = Vec::with_capacity(var_names.len());
        for v in &var_names {
            let val = s.get(v).ok_or(FullEnumFail::Decline)?;
            let (sort, cell) = value_cell_encode(val).ok_or(FullEnumFail::NonEncodable)?;
            sorts.push(sort);
            tup.push(cell);
        }
        match &col_sorts {
            Some(prev) if *prev != sorts => return Err(FullEnumFail::Decline),
            None => col_sorts = Some(sorts),
            _ => {}
        }
        Ok(tup)
    };

    let ctx = build_ctx(&module).ok_or_else(d)?;
    let branches =
        extract_init_constraints(&ctx, &init_def.body, &var_names, None).ok_or_else(d)?;
    let init_states = enumerate_states_from_constraint_branches(Some(&ctx), &var_names, &branches)
        .map_err(|_| FullEnumFail::Decline)?
        .filter(|v| !v.is_empty())
        .ok_or_else(d)?;

    // Graph traversal (DFS worklist) retaining the FULL transition relation; the result is
    // CANONICALIZED below, so traversal order never matters. States are interned to indices.
    let mut index: BTreeMap<Vec<u64>, usize> = BTreeMap::new();
    let mut states: Vec<Vec<u64>> = Vec::new();
    fn intern(
        index: &mut BTreeMap<Vec<u64>, usize>,
        states: &mut Vec<Vec<u64>>,
        tup: Vec<u64>,
    ) -> usize {
        if let Some(&i) = index.get(&tup) {
            return i;
        }
        let i = states.len();
        index.insert(tup.clone(), i);
        states.push(tup);
        i
    }
    let mut inits = Vec::new();
    let mut frontier: Vec<(usize, State)> = Vec::new();
    for s in &init_states {
        let tup = state_tuple(s)?;
        let i = intern(&mut index, &mut states, tup);
        if states.len() > REFINEMENT_STATE_CAP {
            return Err(FullEnumFail::Decline); // the cap applies to initial states too
        }
        if !inits.contains(&i) {
            inits.push(i);
            frontier.push((i, s.clone()));
        }
    }
    let mut transitions: Vec<(usize, usize)> = Vec::new();
    let mut next_ctx = build_ctx(&module).ok_or_else(d)?;
    while let Some((from, cur)) = frontier.pop() {
        let succs = enumerate_successors(&mut next_ctx, &next_def, &cur, &var_names)
            .map_err(|_| FullEnumFail::Decline)?;
        for succ in &succs {
            let tup = state_tuple(succ)?;
            let known = index.contains_key(&tup);
            let to = intern(&mut index, &mut states, tup);
            if states.len() > REFINEMENT_STATE_CAP {
                return Err(FullEnumFail::Decline); // reachable set exceeds the cap
            }
            transitions.push((from, to));
            if transitions.len() > REFINEMENT_TRANSITION_CAP {
                return Err(FullEnumFail::Decline);
            }
            if !known {
                frontier.push((to, succ.clone()));
            }
        }
    }
    let (states, inits, transitions) = canonicalize_graph(states, inits, transitions);
    Ok(ImplGraph {
        var_names,
        sorts: col_sorts.ok_or_else(d)?,
        states,
        inits,
        transitions,
        projected: false,
    })
}

/// CANONICALIZE a `(states, inits, transitions)` graph: sort the state tuples and remap indices,
/// so the graph (and therefore the certificate and the verify-time equality binding) depends only
/// on the reachable set and relation — never on traversal or successor-enumeration order. Shared
/// by the full and projected paths, so both canonicalize identically.
#[cfg(feature = "clean-cic")]
fn canonicalize_graph(
    states: Vec<Vec<u64>>,
    inits: Vec<usize>,
    transitions: Vec<(usize, usize)>,
) -> (Vec<Vec<u64>>, Vec<usize>, Vec<(usize, usize)>) {
    let mut order: Vec<usize> = (0..states.len()).collect();
    order.sort_unstable_by(|&a, &b| states[a].cmp(&states[b]));
    let mut new_ix = vec![0usize; states.len()];
    for (new_i, &old_i) in order.iter().enumerate() {
        new_ix[old_i] = new_i;
    }
    let states: Vec<Vec<u64>> = order.iter().map(|&i| states[i].clone()).collect();
    let mut inits: Vec<usize> = inits.into_iter().map(|i| new_ix[i]).collect();
    let mut transitions: Vec<(usize, usize)> = transitions
        .into_iter()
        .map(|(a, b)| (new_ix[a], new_ix[b]))
        .collect();
    transitions.sort_unstable();
    transitions.dedup();
    inits.sort_unstable();
    inits.dedup();
    (states, inits, transitions)
}

/// The PROJECTED (impl-decoupled) enumeration — the unblock for message-passing implementations
/// (`TwoPhase`'s `msgs`) whose state has a non-cell-encodable column. It enumerates the SAME live
/// reachable graph, but:
///   * identifies implementation states by their FULL CANONICAL VALUE (`State::to_record()`,
///     compared by `Value`'s extensional total order) — collision-free, so NO reachable state and
///     NO transition is ever dropped (a lossy 64-bit fingerprint could collide and silently drop a
///     violating transition — that is exactly the soundness hole this path avoids), and
///   * cell-encodes ONLY the MAPPED columns (`mapped_names`) into the stored tuples, PROJECTING the
///     graph onto those columns. Unmapped columns (`tmState`/`tmPrepared`/`msgs`) exist only to
///     distinguish full states — they are NEVER cell-encoded.
/// The kernel then re-evaluates the abstract `Init`/`Next` on these projected tuples exactly as on
/// the full path (with the stutter disjunct `proj(s)=proj(s')` carrying msgs/tm steps).
///
/// Requires an ACTUAL non-cell-encodable column (else it returns `None` so the full path stays the
/// sole owner of fully-encodable specs — preserving v1 byte-compat), and requires every MAPPED
/// column to be cell-encodable (else fail-closed). Fail-closed on caps/parse/missing-ops.
#[cfg(feature = "clean-cic")]
fn enumerate_impl_graph_projected(
    spec_src: &str,
    config: &Config,
    mapped_names: &std::collections::BTreeSet<Arc<str>>,
) -> Option<ImplGraph> {
    use crate::enumerate::{
        enumerate_states_from_constraint_branches, enumerate_successors, extract_init_constraints,
    };
    use crate::eval::EvalCtx;
    use crate::state::State;
    use crate::value::Value;
    use std::collections::BTreeSet;
    use tla_core::ast::{Module, OperatorDef, Unit};

    let init_name = config.init.as_deref()?;
    let next_name = config.next.as_deref()?;
    let tree = tla_core::parse_to_syntax_tree(spec_src);
    let lowered = tla_core::lower(tla_core::FileId(0), &tree);
    let module = lowered.module?;
    let var_names: Vec<Arc<str>> = module
        .units
        .iter()
        .flat_map(|u| match &u.node {
            Unit::Variable(decls) => decls
                .iter()
                .map(|d| Arc::<str>::from(d.node.as_str()))
                .collect::<Vec<_>>(),
            _ => Vec::new(),
        })
        .collect();
    if var_names.is_empty() {
        return None;
    }
    // The KEPT columns: the mapped implementation columns, in declaration order. A mapped name that
    // is not an implementation variable is ignored here (the mapping resolution declines later).
    let kept_idx: Vec<usize> = var_names
        .iter()
        .enumerate()
        .filter(|(_, v)| mapped_names.contains(*v))
        .map(|(i, _)| i)
        .collect();
    if kept_idx.is_empty() {
        return None; // no mapped columns to build the abstract obligation over
    }
    let kept_names: Vec<Arc<str>> = kept_idx
        .iter()
        .map(|&i| Arc::clone(&var_names[i]))
        .collect();

    let find_op = |name: &str| -> Option<&OperatorDef> {
        module.units.iter().find_map(|unit| match &unit.node {
            Unit::Operator(op) if op.name.node == name => Some(op),
            _ => None,
        })
    };
    let init_def = find_op(init_name)?;
    let next_def = find_op(next_name)?.clone();

    let build_ctx = |module: &Module| -> Option<EvalCtx> {
        let mut ctx = EvalCtx::new();
        ctx.load_module(module);
        for v in &var_names {
            ctx.register_var(Arc::clone(v));
        }
        crate::constants::bind_constants_from_config(&mut ctx, config).ok()?;
        Some(ctx)
    };

    let ctx = build_ctx(&module)?;
    let branches = extract_init_constraints(&ctx, &init_def.body, &var_names, None)?;
    let init_states = enumerate_states_from_constraint_branches(Some(&ctx), &var_names, &branches)
        .ok()?
        .filter(|v| !v.is_empty())?;

    // PASS 1 — enumerate the live reachable graph, identifying full states by their canonical value
    // (`State::to_record()`, `Value`'s extensional total order: `SortedSet`-canonical, so
    // `{a,b} == {b,a}`). This is EXACTLY TLA state identity — COLLISION-FREE — so no reachable full
    // state is skipped and NO transition is dropped (a lossy 64-bit fingerprint could collide and
    // silently drop a violating transition; this is the soundness hole the full-value identity
    // avoids). We record, per interned full state, ONLY its MAPPED columns' values — the unmapped
    // columns (msgs/tm…) serve solely to distinguish full states, never encoded.
    let mut ids: BTreeMap<Value, usize> = BTreeMap::new();
    let mut mapped_vals: Vec<Vec<Value>> = Vec::new(); // per full-state id: the kept columns' values
    let mut id_inits: BTreeSet<usize> = BTreeSet::new();
    let mut id_trans: BTreeSet<(usize, usize)> = BTreeSet::new();
    let mut frontier: Vec<State> = Vec::new();
    let intern = |s: &State,
                  ids: &mut BTreeMap<Value, usize>,
                  mapped_vals: &mut Vec<Vec<Value>>|
     -> Option<(usize, bool)> {
        let key = s.to_record();
        if let Some(&i) = ids.get(&key) {
            return Some((i, false));
        }
        let i = mapped_vals.len();
        let row: Vec<Value> = kept_idx
            .iter()
            .map(|&c| s.get(&var_names[c]).cloned())
            .collect::<Option<_>>()?;
        ids.insert(key, i);
        mapped_vals.push(row);
        Some((i, true))
    };

    for s in &init_states {
        let (i, fresh) = intern(s, &mut ids, &mut mapped_vals)?;
        id_inits.insert(i);
        if fresh {
            frontier.push(s.clone());
            if ids.len() > REFINEMENT_STATE_CAP {
                return None;
            }
        }
    }
    let mut next_ctx = build_ctx(&module)?;
    while let Some(cur) = frontier.pop() {
        let (from, _) = intern(&cur, &mut ids, &mut mapped_vals)?;
        let succs = enumerate_successors(&mut next_ctx, &next_def, &cur, &var_names).ok()?;
        for succ in &succs {
            let (to, fresh) = intern(succ, &mut ids, &mut mapped_vals)?;
            id_trans.insert((from, to));
            if id_trans.len() > REFINEMENT_TRANSITION_CAP {
                return None;
            }
            if fresh {
                frontier.push(succ.clone());
                if ids.len() > REFINEMENT_STATE_CAP {
                    return None;
                }
            }
        }
    }

    // PASS 2 — enum-inference-encode each MAPPED column over ITS values across ALL full states (the
    // per-column label union is sorted+deduped, so verify re-derives the identical sort + cells).
    let n_states = mapped_vals.len();
    let mut kept_sorts: Vec<ColSort> = Vec::with_capacity(kept_idx.len());
    let mut col_cells: Vec<Vec<u64>> = Vec::with_capacity(kept_idx.len()); // [col][state]
    for c in 0..kept_idx.len() {
        let vals: Vec<&Value> = (0..n_states).map(|s| &mapped_vals[s][c]).collect();
        let (sort, cells) = encode_mapped_column(&vals)?;
        kept_sorts.push(sort);
        col_cells.push(cells);
    }
    // Assemble each full state's PROJECTED tuple (mapped columns only), then PROJECT the graph onto
    // those tuples — merging full states that share a projection (sound: the obligation reads only
    // the mapped columns, so distinct full states with equal projection contribute the SAME pair).
    let full_tup: Vec<Vec<u64>> = (0..n_states)
        .map(|s| (0..kept_idx.len()).map(|c| col_cells[c][s]).collect())
        .collect();
    let proj_states: BTreeSet<Vec<u64>> = full_tup.iter().cloned().collect();
    let states: Vec<Vec<u64>> = proj_states.iter().cloned().collect();
    let index: BTreeMap<&Vec<u64>, usize> =
        states.iter().enumerate().map(|(i, t)| (t, i)).collect();
    let inits: Vec<usize> = id_inits.iter().map(|&id| index[&full_tup[id]]).collect();
    let transitions: Vec<(usize, usize)> = id_trans
        .iter()
        .map(|&(a, b)| (index[&full_tup[a]], index[&full_tup[b]]))
        .collect();
    let (states, inits, transitions) = canonicalize_graph(states, inits, transitions);
    Some(ImplGraph {
        var_names: kept_names,
        sorts: kept_sorts,
        states,
        inits,
        transitions,
        projected: true,
    })
}

/// Two-pass ENUM-INFERENCE encoding of ONE mapped implementation column across all its reachable
/// values — the piece the bare per-value [`value_cell_encode`] cannot do (a function to enum
/// strings like `rmState` needs the label union). Supports exactly the sorts the abstract
/// obligation reads at a mapped column:
///   * `Int` (nonneg) / `Bool` — the scalar arithmetic leaves (identical cells to `value_cell_encode`);
///   * scalar `Enum` — a `String`/model-value cell ⇒ the INDEX in the column's sorted label union;
///   * `FuncEnum` — a function to enum labels (`func_enum_view`) ⇒ the positional pack
///     `Σ_d idx(e_d)·|labels|^d` over the sorted label union.
/// The label union is SORTED+deduped (a deterministic per-column property), so verify re-derives the
/// identical `(ColSort, cells)`. Fail-closed (`None`) on an empty column, a heterogeneous column
/// (mixed kinds/shapes/arities/domains), an out-of-range digit, a `u64` overflow, or any sort
/// outside this fragment. Returns `(sort, cells)` with `cells[i]` the encoding of `values[i]`.
#[cfg(feature = "clean-cic")]
fn encode_mapped_column(values: &[&crate::value::Value]) -> Option<(ColSort, Vec<u64>)> {
    use crate::explicit_fixpoint_cert::{func_enum_view, EnumKind};
    use crate::value::Value;
    if values.is_empty() {
        return None;
    }
    let nonneg_int = |v: &Value| -> Option<u64> {
        match v {
            Value::SmallInt(n) if *n >= 0 => Some(*n as u64),
            Value::Int(n) => {
                use num_traits::Signed;
                if n.is_negative() {
                    None
                } else {
                    u64::try_from(n.as_ref().clone()).ok()
                }
            }
            _ => None,
        }
    };
    // (1) All nonneg Int.
    if values.iter().all(|v| nonneg_int(v).is_some()) {
        let cells: Vec<u64> = values.iter().map(|v| nonneg_int(v).unwrap()).collect();
        return Some((ColSort::Int, cells));
    }
    // (2) All Bool.
    if values.iter().all(|v| matches!(v, Value::Bool(_))) {
        let cells: Vec<u64> = values
            .iter()
            .map(|v| matches!(v, Value::Bool(true)) as u64)
            .collect();
        return Some((ColSort::Bool, cells));
    }
    // (3) Scalar Enum — every value a `String` (Str) or `ModelValue` (Model), homogeneous kind.
    let scalar_label = |v: &Value| -> Option<(EnumKind, String)> {
        match v {
            Value::String(s) => Some((EnumKind::Str, s.as_ref().to_string())),
            Value::ModelValue(s) => Some((EnumKind::Model, s.as_ref().to_string())),
            _ => None,
        }
    };
    if values.iter().all(|v| scalar_label(v).is_some()) {
        let mut kind: Option<EnumKind> = None;
        let mut labels: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
        for v in values {
            let (k, l) = scalar_label(v)?;
            match kind {
                Some(prev) if prev != k => return None, // mixed Str/model kinds
                _ => kind = Some(k),
            }
            labels.insert(l);
        }
        let labels: Vec<String> = labels.into_iter().collect(); // sorted (BTreeSet)
        let cells: Vec<u64> = values
            .iter()
            .map(|v| {
                let (_, l) = scalar_label(v)?;
                labels.iter().position(|x| *x == l).map(|i| i as u64)
            })
            .collect::<Option<_>>()?;
        return Some((
            ColSort::Enum {
                labels,
                kind: kind?,
            },
            cells,
        ));
    }
    // (4) FuncEnum — every value a function to enum labels (`func_enum_view`), with a CONSISTENT
    // arity, domain, domain-kind, and value-label kind across all states.
    let views: Vec<(u32, bool, Vec<String>, Vec<String>, EnumKind)> = values
        .iter()
        .map(|v| func_enum_view(v))
        .collect::<Option<_>>()?;
    let (arity0, is_model0, _, dom0, dom_kind0) = &views[0];
    if !views
        .iter()
        .all(|(a, m, _, d, dk)| a == arity0 && m == is_model0 && d == dom0 && dk == dom_kind0)
    {
        return None; // heterogeneous function shape across states ⇒ out of fragment
    }
    let mut labels: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
    for (_, _, value_labels, _, _) in &views {
        for l in value_labels {
            labels.insert(l.clone());
        }
    }
    let labels: Vec<String> = labels.into_iter().collect(); // sorted
    let base = labels.len() as u64;
    base.checked_pow(*arity0)?; // pack universe must fit u64
    let mut cells: Vec<u64> = Vec::with_capacity(values.len());
    for (_, _, value_labels, _, _) in &views {
        let mut pack: u64 = 0;
        for (d, lbl) in value_labels.iter().enumerate() {
            let idx = labels.iter().position(|x| x == lbl)? as u64;
            let place = base.checked_pow(d as u32)?;
            pack = pack.checked_add(idx.checked_mul(place)?)?;
        }
        cells.push(pack);
    }
    Some((
        ColSort::FuncEnum {
            arity: *arity0,
            labels,
            dom: dom0.clone(),
            dom_kind: *dom_kind0,
        },
        cells,
    ))
}

/// TRUTH-DIRECTION exactness filter (the load-bearing soundness gate of this lane). The
/// fixpoint lane's recognizer/embedder admit forms whose kernel evaluation OVER-APPROXIMATES
/// TLA+ truth (documented there as "harmless — at worst ADDS spurious pairs", correct for the
/// SUPERSET direction). Refinement needs the OPPOSITE guarantee: kernel-TRUE ⇒ TLA-TRUE — and so
/// does the explicit-fixpoint lane's GENERAL `R⊆Safety` leg (`safety_pred`/`safety_general`),
/// which shares this gate. This filter admits ONLY constructs whose embedding is EXACT:
///   * `Nat.sub` (`ValIR::Sub`) — TRUNCATES negatives (kernel says `0 = 0−1` is true) — REJECT;
///   * `ValIR::Div`/`Mod` — TLA `/` is REAL division and lowers to the same recognized shape
///     as `\div` from this side — REJECT both (conservative);
///   * Seq digit ops — absent-element digits truncate — REJECT;
///   * bare `Var`/`Prime` refs in arithmetic/comparisons — the recognizer is SORT-BLIND here,
///     so a Bool/Set CELL would pun as a Nat (kernel evaluates `FALSE + 1`) — require
///     `ColSort::Int`;
///   * Set nodes — canonical-bitmask ops (`SetEq`/`Mem`/`Subseteq`/`Cup`/`Cap` over Set-sorted
///     columns and literal masks) are exact — ADMIT; comprehension/quantifier folds — REJECT
///     (v1 conservative);
///   * `Unchanged`/`SetUnchanged` — canonical-cell equality is exact for every sort — ADMIT.
#[cfg(feature = "clean-cic")]
fn val_exact(v: &crate::explicit_fixpoint_cert::ValIR, sorts: &[ColSort]) -> bool {
    use crate::explicit_fixpoint_cert::ValIR as V;
    match v {
        V::Lit(_) => true,
        V::Var(i) | V::Prime(i) => matches!(sorts.get(*i), Some(ColSort::Int)),
        V::Add(a, b) | V::Mul(a, b) => val_exact(a, sorts) && val_exact(b, sorts),
        // The ONE admitted div/mod shape: POSITIONAL-PACK digit extraction over a `Record` or `Func`
        // column — see `compound_digit_exact`. Everything else keeps the conservative rejection below.
        V::Mod(..) if compound_digit_exact(v, sorts) => true,
        V::Div(..) | V::Mod(..) | V::Sub(..) => false,
        V::SeqLen { .. } | V::SeqTail { .. } | V::SeqAppend { .. } => false,
        // Cardinality (popcount) is EXACT nonneg-Nat arithmetic on the faithful bitmask (bit count = |S|),
        // truth-exact in both directions, admitted when its underlying set is exact.
        V::SetCard { set, .. } => set_exact(set, sorts),
        // Cardinality of a comprehension `Σ_{d∈D} boolToNat(P(d))` is EXACT iff EVERY per-element `P(d)` is
        // truth-exact — then each `boolToNat` is exactly 0/1 and the sum is `|{d∈D:P(d)}|`. An inexact term
        // could make a `boolToNat` mis-count, so REJECT unless all terms pass the same `pred_exact` gate.
        V::CountFold { terms } => terms.iter().all(|t| pred_exact(t, sorts)),
    }
}

/// The POSITIONAL-PACK digit form `(pack / base^idx) mod base` over a `Record{base, fields}` OR a
/// `Func{base, arity}` column — the ONE div/mod shape [`pred_exact`] admits (roadmap R3, extended to
/// `Func`). EXACT in both truth directions: a Record/Func column's cell is BY CONSTRUCTION the canonical
/// pack `Σ v_i·base^i` with every digit `< base` (`value_cell_encode` fails closed otherwise), so
/// `(pack / base^idx) mod base = v_idx` is exact nonneg integer arithmetic — the kernel's
/// `Nat.div`/`Nat.mod` on these concrete literals compute precisely the TLA field value `r.f_idx` /
/// function value `f[idx]`. The shape is matched STRUCTURALLY: `Mod(Div(Var|Prime of a Record/Func
/// column, Lit(base^idx)), Lit(base))` with `idx < arity`. The only recognizers that emit it are
/// `recognize_val_sorts`' RecordAccess/FuncApply arms (plus the record-set / function-set membership and
/// Bool-position forms built from them) — the surface `\div`/`%` arms REFUSE bare compound-column
/// operands precisely so no surface arithmetic can forge this shape (TLA arithmetic on a record/function
/// is undefined and must stay rejected).
#[cfg(feature = "clean-cic")]
fn compound_digit_exact(v: &crate::explicit_fixpoint_cert::ValIR, sorts: &[ColSort]) -> bool {
    use crate::explicit_fixpoint_cert::ValIR as V;
    let V::Mod(div, modulus) = v else {
        return false;
    };
    let (V::Div(pack, place), V::Lit(m)) = (div.as_ref(), modulus.as_ref()) else {
        return false;
    };
    let (col, V::Lit(place)) = (pack.as_ref(), place.as_ref()) else {
        return false;
    };
    let (V::Var(i) | V::Prime(i)) = col else {
        return false;
    };
    // Kind-AGNOSTIC on the position (`cells` ignored): the digit `(pack/base^idx) mod base` is exact nonneg
    // arithmetic on the canonical pack whether position `idx` is Int, Bool, or Enum (a Bool code is `{0,1}`
    // and an enum index is `< labels.len() ≤ base`). Admitted over BOTH `Record` and (Int-domain) `Func`
    // columns — their packs are identical `Σ code_j·base^j` positional numerals. The recognizer feeds a
    // Bool/Enum-position digit ONLY into a kind-checked EQUALITY (`= TRUE` / `∈ BOOLEAN` / `= "lbl"` /
    // `∈ {…}`/`∈ Data`), never an ordering, so admitting the digit shape here stays truth-exact.
    let (base, arity) = match sorts.get(*i) {
        Some(ColSort::Record { base, fields, .. }) => (*base, fields.len()),
        Some(ColSort::Func { base, arity, .. }) => (*base, *arity as usize),
        _ => return false,
    };
    if *m != base {
        return false;
    }
    // `place` must be `base^idx` for some position `idx < arity`.
    let mut p: u64 = 1;
    for _ in 0..arity {
        if p == *place {
            return true;
        }
        let Some(next) = p.checked_mul(base) else {
            return false;
        };
        p = next;
    }
    false
}

#[cfg(feature = "clean-cic")]
fn set_exact(s: &crate::explicit_fixpoint_cert::SetIR, sorts: &[ColSort]) -> bool {
    use crate::explicit_fixpoint_cert::SetIR as S;
    match s {
        S::Lit(_) => true,
        S::Var(i) | S::Prime(i) => {
            matches!(
                sorts.get(*i),
                Some(ColSort::Set { .. } | ColSort::SetMask { .. } | ColSort::SetMaskRec { .. })
            )
        }
        S::Cup(a, b) | S::Cap(a, b) => set_exact(a, sorts) && set_exact(b, sorts),
        // A FUNC-to-SET application `f[k]` digit is EXACT iff its `pack` is a `FuncSetMask` column ref: the
        // digit `(pack / base^slot) mod base` is a genuine `|E|`-bit subset mask by construction of the
        // canonical pack (a bijection with subsets of `E`), so set predicates over it are truth-exact — the
        // set analogue of `compound_digit_exact`. The recognizer only emits it from a `FuncSetMask` column
        // application, and the bare value has no arithmetic/ordering path, so it is never punned as a Nat.
        S::Digit { pack, .. } => matches!(
            pack.as_ref(),
            crate::explicit_fixpoint_cert::ValIR::Var(i)
                | crate::explicit_fixpoint_cert::ValIR::Prime(i)
                if matches!(sorts.get(*i), Some(ColSort::FuncSetMask { .. }))
        ),
        S::Filter { .. } => false,
    }
}

/// A well-formed FINITE-ENUM EQUALITY `Eq`/`Neq` over an [`ColSort::Enum`] column — the ONE extra shape
/// [`pred_exact`] admits on an enum-index operand (ORDERING `< / ≤ / > / ≥` on enum indices stays
/// REJECTED, as it is meaningless: an index carries no order). BIJECTION-exact (`label = label ⟺ index =
/// index`), so truth-exact in BOTH directions. Admitted iff at least one operand is an enum-column
/// `Var`/`Prime` — defining the label space `L` — and EVERY operand is either
///   * a `Lit(v)` with `v < L.len()` (a valid label index — the recognizer only emits observed indices), or
///   * an enum-column `Var`/`Prime` whose own labels EQUAL `L` (same label→index map ⇒ index eq tracks
///     label eq; a DIFFERENT label space would be unsound and is rejected).
/// Anything else (an Int/Bool operand against an enum column, an out-of-range index, a bare enum column
/// with no partner) ⇒ `false` (fail-closed).
#[cfg(feature = "clean-cic")]
fn enum_eq_exact(
    a: &crate::explicit_fixpoint_cert::ValIR,
    b: &crate::explicit_fixpoint_cert::ValIR,
    sorts: &[ColSort],
) -> bool {
    use crate::explicit_fixpoint_cert::ValIR as V;
    let enum_labels = |v: &V| -> Option<&[String]> {
        match v {
            V::Var(i) | V::Prime(i) => match sorts.get(*i) {
                // Exactness is decided by the label INDEX range only; the column KIND is irrelevant here
                // (kind is enforced at recognition time, in `enum_in_form`/`enum_eq_form`).
                Some(ColSort::Enum { labels, .. }) => Some(labels.as_slice()),
                _ => None,
            },
            _ => None,
        }
    };
    // The label space: at least one operand must be an enum column.
    let Some(space) = enum_labels(a).or_else(|| enum_labels(b)) else {
        return false;
    };
    let operand_ok = |v: &V| -> bool {
        match v {
            V::Lit(k) => (*k as usize) < space.len(), // a valid, observed label index
            _ => enum_labels(v) == Some(space),       // same-label-space enum column
        }
    };
    operand_ok(a) && operand_ok(b)
}

/// A BOOL-COLUMN EQUALITY `Eq`/`Neq` — the ONE extra shape [`pred_exact`] admits on a BARE Bool-column
/// (`ColSort::Bool`) operand (the `flag = TRUE`/`flag = FALSE`/`b1 = b2` scalar forms, and a bare Bool
/// predicate `flag` ≡ `Eq(Var, 1)`). EXACT in BOTH truth directions: a Bool cell is `0`/`1` by
/// construction (`value_cell_encode` maps `TRUE→1`,`FALSE→0`), so `cell = k` is truth-exact for ANY
/// literal `k` (a `k ∉ {0,1}` is faithfully never equal to the cell), and two Bool cells are equal iff
/// their codes agree iff the booleans agree. Admitted iff at least one operand is a Bool-column
/// `Var`/`Prime` and EVERY operand is a `Lit` or a Bool-column `Var`/`Prime`. ORDERING (`< ≤ > ≥`) on a
/// Bool operand stays REJECTED (the `val_exact` gate is unchanged) so `flag < 3` — a Bool punned as an
/// Int, UNDEFINED in TLA — never rides the exact leg. A mixed Bool-vs-Int-column operand also fails the
/// `operand_ok` gate (an Int `Var` is neither a `Lit` nor a Bool column) ⇒ closed. The only recognizers
/// that emit a Bool-column `Var`/`Prime` into an `Eq`/`Neq` are the bare-Bool-ident predicate arm and
/// `bool_eq_form`/`compound`-free scalar forms, all EQUALITY-ONLY.
#[cfg(feature = "clean-cic")]
fn bool_eq_exact(
    a: &crate::explicit_fixpoint_cert::ValIR,
    b: &crate::explicit_fixpoint_cert::ValIR,
    sorts: &[ColSort],
) -> bool {
    use crate::explicit_fixpoint_cert::ValIR as V;
    let is_bool_col = |v: &V| {
        matches!(
            v,
            V::Var(i) | V::Prime(i) if matches!(sorts.get(*i), Some(ColSort::Bool))
        )
    };
    let operand_ok = |v: &V| matches!(v, V::Lit(_)) || is_bool_col(v);
    (is_bool_col(a) || is_bool_col(b)) && operand_ok(a) && operand_ok(b)
}

/// The FUNC-of-ENUM digit form `(pack / |labels|^i) mod |labels|` over a [`ColSort::FuncEnum`] column
/// (`f[i]`, `i` a literal `< arity`), returning the column's label space — the ONE extra div/mod shape
/// [`pred_exact`] admits on a FuncEnum operand (ORDERING on a func-enum digit stays REJECTED, as it is
/// meaningless — an enum index carries no order). EXACT in both truth directions: a FuncEnum cell is BY
/// CONSTRUCTION the canonical pack `Σ_d idx(e_d)·|labels|^d` with every digit `< |labels|`
/// (`value_cell` fails closed otherwise), so `(pack / |labels|^i) mod |labels| = idx(e_i)` is exact nonneg
/// integer arithmetic on the concrete pack — the kernel's `Nat.div`/`Nat.mod` compute precisely the label
/// index `idx(pc[i])` (a bijection to the label). Matched STRUCTURALLY, exactly the shape
/// `cleancic::func_enum_app` emits: `Mod(Div(Var|Prime of a FuncEnum column, Lit(base^i)), Lit(base))`
/// with `base = |labels|` and `base^i` a valid place value `i < arity`. The only recognizers that emit it
/// are the FuncEnum equality/membership forms; the surface `\div`/`%` arms REFUSE bare compound-column
/// operands (FuncEnum `is_compound`), so no surface arithmetic can forge this shape.
#[cfg(feature = "clean-cic")]
fn func_enum_digit_space<'a>(
    v: &crate::explicit_fixpoint_cert::ValIR,
    sorts: &'a [ColSort],
) -> Option<&'a [String]> {
    use crate::explicit_fixpoint_cert::ValIR as V;
    let V::Mod(div, modulus) = v else { return None };
    let (V::Div(pack, place), V::Lit(m)) = (div.as_ref(), modulus.as_ref()) else {
        return None;
    };
    let (col, V::Lit(place)) = (pack.as_ref(), place.as_ref()) else {
        return None;
    };
    let (V::Var(i) | V::Prime(i)) = col else {
        return None;
    };
    let Some(ColSort::FuncEnum { arity, labels, .. }) = sorts.get(*i) else {
        return None;
    };
    let base = labels.len() as u64;
    if *m != base {
        return None;
    }
    // `place` must be `base^idx` for some domain position `idx < arity`.
    let mut p: u64 = 1;
    for _ in 0..*arity {
        if p == *place {
            return Some(labels.as_slice());
        }
        let Some(next) = p.checked_mul(base) else {
            return None;
        };
        p = next;
    }
    None
}

/// A well-formed FUNC-of-ENUM EQUALITY `Eq`/`Neq` over a [`ColSort::FuncEnum`] digit — the ONE extra shape
/// [`pred_exact`] admits on a func-enum digit operand (ORDERING stays REJECTED). BIJECTION-exact
/// (`label = label ⟺ digit-index = digit-index`), truth-exact in BOTH directions. Admitted iff at least
/// one operand is a FuncEnum digit (see [`func_enum_digit_space`]) — defining the label space `L` — and
/// EVERY operand is either a `Lit(v)` with `v < L.len()` (a valid label index — the recognizer only emits
/// observed indices) or another FuncEnum digit whose own labels EQUAL `L` (same label→index map ⇒ digit eq
/// tracks label eq; a DIFFERENT label space is unsound and rejected).
#[cfg(feature = "clean-cic")]
fn func_enum_eq_exact(
    a: &crate::explicit_fixpoint_cert::ValIR,
    b: &crate::explicit_fixpoint_cert::ValIR,
    sorts: &[ColSort],
) -> bool {
    use crate::explicit_fixpoint_cert::ValIR as V;
    let Some(space) = func_enum_digit_space(a, sorts).or_else(|| func_enum_digit_space(b, sorts))
    else {
        return false;
    };
    let operand_ok = |v: &V| -> bool {
        match v {
            V::Lit(k) => (*k as usize) < space.len(), // a valid, observed label index
            _ => func_enum_digit_space(v, sorts) == Some(space), // same-label-space func-enum digit
        }
    };
    operand_ok(a) && operand_ok(b)
}

/// A WHOLE-`FuncEnum`-COLUMN value equality `Eq`/`Neq`: one side a bare column ref (`Var(i)`/`Prime(i)`)
/// of sort [`ColSort::FuncEnum`], the OTHER side that column's CANONICAL POSITIONAL PACK — the shape the
/// recognizer emits for `f = [d ∈ D |-> lbl]` (an `Init` function literal, all-`Lit` digits) and
/// `f' = [f EXCEPT ![k] = lbl]` (a `Next` step: the changed slot a `Lit`, every other slot a COPIED
/// func-enum digit of `f`). TRUTH-EXACT in BOTH directions (`kernel-TRUE ⟺ TLA-TRUE`): the pack is
/// `Σ_{d<arity} A_d · base^d` (`base = |labels|`) with EVERY place `base^d` present EXACTLY once and EVERY
/// digit `A_d ∈ [0, base)` — a `Lit(k<base)` OR a func-enum digit `Mod(_, Lit(base))` (always `< base`).
/// Hence `eval(pack) ∈ [0, base^arity)` is ALWAYS a VALID cell, and the cell encoding is CANONICAL/
/// INJECTIVE, so `cell = eval(pack) ⟺ value = the-function-the-pack-denotes`. A BARE `Var`/`Prime` on the
/// pack side is REJECTED (a raw non-`Int` cell punning as a Nat would be unsound); only literals and
/// base-bounded digits are admitted, which is what keeps `eval(pack)` a valid, canonically-decodable cell.
#[cfg(feature = "clean-cic")]
fn funcenum_pack_exact(
    a: &crate::explicit_fixpoint_cert::ValIR,
    b: &crate::explicit_fixpoint_cert::ValIR,
    sorts: &[ColSort],
) -> bool {
    use crate::explicit_fixpoint_cert::ValIR as V;
    // A digit is valid iff it is `Lit(k < base)` (`base = |labels|`) or a func-enum digit
    // `Mod(_, Lit(base))` over a column whose label space EQUALS this column's labels. Requiring the
    // SAME label space (not merely the same length) makes the gate SELF-SUFFICIENT: a cross-column
    // digit over a DIFFERENT (equal-length) label space would carry a value under a different
    // label→index map and is REJECTED here — so soundness never rests on the recognizer only ever
    // emitting same-column digits (it does today, but this gate no longer relies on that).
    let digit_ok = |ad: &V, labels: &[String]| -> bool {
        match ad {
            V::Lit(k) => (*k as usize) < labels.len(),
            _ => func_enum_digit_space(ad, sorts).is_some_and(|sp| sp == labels),
        }
    };
    let try_one = |colref: &V, pack: &V| -> bool {
        let (V::Var(i) | V::Prime(i)) = colref else {
            return false;
        };
        let Some(ColSort::FuncEnum { arity, labels, .. }) = sorts.get(*i) else {
            return false;
        };
        let base = labels.len() as u64;
        let arity = *arity as usize;
        if base == 0 || arity == 0 {
            return false;
        }
        // Flatten the (left-nested) `Add` tree into its positional terms.
        fn flatten<'a>(e: &'a V, out: &mut Vec<&'a V>) {
            if let V::Add(x, y) = e {
                flatten(x, out);
                flatten(y, out);
            } else {
                out.push(e);
            }
        }
        let mut terms: Vec<&V> = Vec::new();
        flatten(pack, &mut terms);
        if terms.len() != arity {
            return false;
        }
        // The place `base^d` for a `Lit(place)` operand, `Some(d)` iff `d < arity` — else `None`.
        let place_index = |place_val: u64| -> Option<usize> {
            let mut p: u64 = 1;
            for d in 0..arity {
                if p == place_val {
                    return Some(d);
                }
                p = p.checked_mul(base)?;
            }
            None
        };
        // Interpret a term operand pair as `(digit · Lit(base^d))` in EITHER operand order.
        let interp = |digit: &V, place: &V| -> Option<usize> {
            let V::Lit(place_val) = place else {
                return None;
            };
            let d = place_index(*place_val)?;
            digit_ok(digit, labels).then_some(d)
        };
        let mut seen: std::collections::BTreeSet<usize> = std::collections::BTreeSet::new();
        for term in terms {
            let V::Mul(x, y) = term else { return false };
            let Some(d) = interp(x, y).or_else(|| interp(y, x)) else {
                return false;
            };
            if !seen.insert(d) {
                return false; // a place appeared twice ⇒ not a canonical positional pack
            }
        }
        seen.len() == arity // every place base^0..base^{arity-1} present exactly once
    };
    try_one(a, b) || try_one(b, a)
}

#[cfg(feature = "clean-cic")]
pub(crate) fn pred_exact(p: &PredIR, sorts: &[ColSort]) -> bool {
    use PredIR as P;
    match p {
        P::And(a, b) | P::Or(a, b) | P::Implies(a, b) | P::Equiv(a, b) => {
            pred_exact(a, sorts) && pred_exact(b, sorts)
        }
        P::Not(a) => pred_exact(a, sorts),
        P::BoolLit(_) => true,
        // EQUALITY admits the numeric-exact form (Int/compound operands), the SCALAR finite-enum equality
        // form (`enum_eq_exact`), AND the FUNC-of-ENUM digit equality form (`func_enum_eq_exact`) — the two
        // enum forms are the ONLY places an enum-index operand is allowed; the ordering arms below keep the
        // strict Int/compound `val_exact` gate (no enum, scalar or func).
        P::Eq(a, b) | P::Neq(a, b) => {
            (val_exact(a, sorts) && val_exact(b, sorts))
                || enum_eq_exact(a, b, sorts)
                || func_enum_eq_exact(a, b, sorts)
                || bool_eq_exact(a, b, sorts)
                || funcenum_pack_exact(a, b, sorts)
        }
        P::Lt(a, b) | P::Leq(a, b) | P::Gt(a, b) | P::Geq(a, b) => {
            val_exact(a, sorts) && val_exact(b, sorts)
        }
        P::Unchanged(_) | P::SetUnchanged(_) => true,
        P::SetEq(a, b) | P::SetNeq(a, b) | P::SetSubseteq(a, b) => {
            set_exact(a, sorts) && set_exact(b, sorts)
        }
        P::SetMem(_, s) | P::SetNotMem(_, s) => set_exact(s, sorts),
        P::SetForall { .. }
        | P::SetExists { .. }
        | P::SubsetForall { .. }
        | P::SubsetExists { .. } => false,
    }
}

/// Whether the IR mentions any PRIMED state (an abstract `Init` — and the explicit-fixpoint lane's
/// general `Safety` invariant — must be a STATE predicate).
#[cfg(feature = "clean-cic")]
pub(crate) fn pred_mentions_prime(p: &PredIR) -> bool {
    use crate::explicit_fixpoint_cert::{SetIR as S, ValIR as V};
    fn v(x: &V) -> bool {
        match x {
            V::Prime(_) => true,
            V::Lit(_) | V::Var(_) => false,
            V::Add(a, b) | V::Mul(a, b) | V::Div(a, b) | V::Mod(a, b) | V::Sub(a, b) => {
                v(a) || v(b)
            }
            V::SeqLen { pack, .. } | V::SeqTail { pack, .. } => v(pack),
            V::SeqAppend { pack, elem, .. } => v(pack) || v(elem),
            V::SetCard { set, .. } => s(set),
            V::CountFold { terms } => terms.iter().any(pred_mentions_prime),
        }
    }
    fn s(x: &S) -> bool {
        match x {
            S::Prime(_) => true,
            S::Lit(_) | S::Var(_) => false,
            // A FUNC-to-SET digit mentions a prime iff its `pack` (a `Var`/`Prime` column ref) does.
            S::Digit { pack, .. } => v(pack),
            S::Cup(a, b) | S::Cap(a, b) => s(a) || s(b),
            S::Filter { source, pred, .. } => s(source) || pred_mentions_prime(pred),
        }
    }
    use PredIR as P;
    match p {
        P::And(a, b) | P::Or(a, b) | P::Implies(a, b) | P::Equiv(a, b) => {
            pred_mentions_prime(a) || pred_mentions_prime(b)
        }
        P::Not(a) => pred_mentions_prime(a),
        P::BoolLit(_) => false,
        P::Eq(a, b) | P::Neq(a, b) | P::Lt(a, b) | P::Leq(a, b) | P::Gt(a, b) | P::Geq(a, b) => {
            v(a) || v(b)
        }
        P::Unchanged(_) | P::SetUnchanged(_) => true,
        P::SetEq(a, b) | P::SetNeq(a, b) | P::SetSubseteq(a, b) => s(a) || s(b),
        P::SetMem(_, x) | P::SetNotMem(_, x) => s(x),
        P::SetForall { source, body, .. } | P::SetExists { source, body, .. } => {
            s(source) || pred_mentions_prime(body)
        }
        P::SubsetForall { source, body, .. } | P::SubsetExists { source, body, .. } => {
            s(source) || pred_mentions_prime(body)
        }
    }
}

/// The `CONSTANT` model-value SETS of a config (`name → sorted+deduped member names`), threaded
/// into the abstract-predicate recognizer so a `\A rm \in RM` / `[RM -> …]` over a model-value
/// domain expands EXACTLY. DETERMINISTIC (a `BTreeMap`, each value sorted+deduped), so verify
/// re-derives the identical map ⇒ the identical IR. Mirrors the explicit-fixpoint safety lane's
/// `mvsets` construction (`ModelValueSet` gives member names directly; a `Value` whose raw string
/// is a brace-delimited identifier list `{d1,d2}` is TLC's alternate spelling, parsed the same).
#[cfg(feature = "clean-cic")]
fn build_mvsets(config: &Config) -> BTreeMap<String, Vec<String>> {
    let mut m = BTreeMap::new();
    for (name, cv) in &config.constants {
        let names: Option<Vec<String>> = match cv {
            crate::config::ConstantValue::ModelValueSet(ns) => Some(ns.clone()),
            crate::config::ConstantValue::Value(s) => {
                crate::explicit_fixpoint_cert::parse_brace_model_value_set(s)
            }
            _ => None,
        };
        if let Some(mut ns) = names {
            ns.sort();
            ns.dedup();
            m.insert(name.clone(), ns);
        }
    }
    m
}

/// Parse the ABSTRACT spec and recognize its `Init`/`Next` into `PredIR` over the abstract
/// variable list with the MAPPED sorts. The bodies are INLINED (`CertInlineEnv` — resolving
/// operator definitions like TCommit's `canCommit`/`Decide` and numeric/replacement CONSTANTs)
/// and recognized WITH the config's model-value sets (`\A rm \in RM` / `[RM -> …]` expansion) —
/// exactly the front-end the explicit-fixpoint safety lane uses, so an abstract spec that
/// safety-certifies is recognizable here too. Returns `(abs_vars, init_ir, next_ir)`.
#[cfg(feature = "clean-cic")]
fn recognize_abstract(
    abs_src: &str,
    abs_init: &str,
    abs_next: &str,
    abs_sorts: &[ColSort],
    abs_config: &Config,
) -> Option<(Vec<String>, PredIR, PredIR)> {
    use tla_core::ast::{OperatorDef, Unit};
    let tree = tla_core::parse_to_syntax_tree(abs_src);
    let lowered = tla_core::lower(tla_core::FileId(0), &tree);
    let module = lowered.module?;
    let abs_var_arcs: Vec<Arc<str>> = module
        .units
        .iter()
        .flat_map(|u| match &u.node {
            Unit::Variable(decls) => decls
                .iter()
                .map(|d| Arc::<str>::from(d.node.as_str()))
                .collect::<Vec<_>>(),
            _ => Vec::new(),
        })
        .collect();
    let abs_vars: Vec<String> = abs_var_arcs.iter().map(|v| v.to_string()).collect();
    if abs_vars.is_empty() || abs_vars.len() != abs_sorts.len() {
        return None;
    }
    let find_op = |name: &str| -> Option<&OperatorDef> {
        module.units.iter().find_map(|unit| match &unit.node {
            Unit::Operator(op) if op.name.node == name => Some(op),
            _ => None,
        })
    };
    let init_def = find_op(abs_init)?;
    let next_def = find_op(abs_next)?;
    // INLINE operator definitions / numeric+replacement constants, and build the model-value sets —
    // the same front-end the safety lane runs before recognition.
    let inline_env = crate::cert_inline::CertInlineEnv::new(&module, abs_config, &abs_var_arcs);
    let init_body = inline_env.inline(&init_def.body);
    let next_body = inline_env.inline(&next_def.body);
    let mvsets = build_mvsets(abs_config);
    let var_refs: Vec<&str> = abs_vars.iter().map(|v| v.as_str()).collect();
    let init_ir = crate::cleancic::recognize_pred_sorts_with_mvsets(
        &init_body.node,
        &var_refs,
        abs_sorts,
        &mvsets,
    )?;
    let next_ir = crate::cleancic::recognize_pred_sorts_with_mvsets(
        &next_body.node,
        &var_refs,
        abs_sorts,
        &mvsets,
    )?;
    // TRUTH-DIRECTION gates (see `pred_exact`): the recognizer's superset-safe forms are
    // NOT sound here; and an abstract `Init` must be a STATE predicate (no primes).
    if !pred_exact(&init_ir, abs_sorts) || !pred_exact(&next_ir, abs_sorts) {
        return None;
    }
    if pred_mentions_prime(&init_ir) {
        return None;
    }
    Some((abs_vars, init_ir, next_ir))
}

/// The EXACT-AFFINE mapping fragment: the ONLY `ValIR` shapes a DERIVED refinement mapping may
/// take. A refinement mapping `f` is a STATE function of the CURRENT implementation state whose
/// derived value must be computed EXACTLY (kernel-TRUE ⇒ TLA-TRUE, like [`pred_exact`]): an
/// affine sum/product of nonneg-`Int` columns never truncates. Admits `Lit`, `Var` over an `Int`
/// column, `Add`, and `Mul`; REJECTS `Prime` (`f` reads the current state only — a primed
/// reference would make `f` a transition function, not a state map), `Sub`/`Div`/`Mod` (can
/// truncate / be undefined), the sequence ops, and any `Var` over a non-`Int` column (a
/// Bool/Set/compound cell would pun as a Nat under `+`/`*`). A bare projection is NOT routed
/// through this gate — it inherits the column's sort verbatim; this gate governs the DERIVED
/// (compound) fragment only.
#[cfg(feature = "clean-cic")]
fn map_ir_exact(ir: &ValIR, impl_sorts: &[ColSort]) -> bool {
    use ValIR as V;
    match ir {
        V::Lit(_) => true,
        V::Var(i) => matches!(impl_sorts.get(*i), Some(ColSort::Int)),
        V::Add(a, b) | V::Mul(a, b) => map_ir_exact(a, impl_sorts) && map_ir_exact(b, impl_sorts),
        V::Prime(_)
        | V::Div(..)
        | V::Mod(..)
        | V::Sub(..)
        | V::SeqLen { .. }
        | V::SeqTail { .. }
        | V::SeqAppend { .. }
        | V::SetCard { .. }
        | V::CountFold { .. } => false,
    }
}

/// The abstract variable's SORT under a resolved mapping expression. A bare `Var(i)` is a
/// PROJECTION — the abstract variable inherits implementation column `i`'s sort (ANY sort:
/// Bool/Set/compound project verbatim, exactly as the v1 lane did). Any other (compound-affine,
/// or a `Lit` constant) mapping derives an `Int` value, so its abstract sort is `Int`. `None`
/// only for an out-of-range projection column.
#[cfg(feature = "clean-cic")]
fn map_result_sort(ir: &ValIR, impl_sorts: &[ColSort]) -> Option<ColSort> {
    match ir {
        ValIR::Var(i) => impl_sorts.get(*i).cloned(),
        _ => Some(ColSort::Int),
    }
}

/// EXACTLY evaluate a mapping `ValIR` at a concrete implementation tuple `s` (`Var` = current
/// column, no primes). Uses CHECKED arithmetic — an affine sum of nonneg columns is exact in ℤ,
/// so the only failure is a `u64` overflow (fail-closed rather than silently wrap/saturate) or
/// an out-of-fragment node (which the [`map_ir_exact`] gate already excludes, but re-checked
/// here for a tampered derived entry). `None` fail-closed.
#[cfg(feature = "clean-cic")]
fn eval_map_exact(ir: &ValIR, s: &[u64]) -> Option<u64> {
    use ValIR as V;
    match ir {
        V::Lit(v) => Some(*v),
        V::Var(i) => s.get(*i).copied(),
        V::Add(a, b) => eval_map_exact(a, s)?.checked_add(eval_map_exact(b, s)?),
        V::Mul(a, b) => eval_map_exact(a, s)?.checked_mul(eval_map_exact(b, s)?),
        V::Prime(_)
        | V::Div(..)
        | V::Mod(..)
        | V::Sub(..)
        | V::SeqLen { .. }
        | V::SeqTail { .. }
        | V::SeqAppend { .. }
        | V::SetCard { .. }
        | V::CountFold { .. } => None,
    }
}

/// Resolve the refinement mapping to a per-abstract-variable `ValIR` over IMPLEMENTATION columns
/// (declaration order). A DERIVED entry (in `derived_map`) is used verbatim after RE-VALIDATION
/// (exact-affine, columns in range) — so a tampered/re-sealed cert carrying an out-of-fragment
/// derived expression is REJECTED here, never trusted. Otherwise the abstract variable is a
/// PROJECTION: its implementation column name comes from `var_map` (or defaults to the same
/// name), resolved to `Var(col)` — identical to the v1 projection resolution.
#[cfg(feature = "clean-cic")]
fn resolve_mapping_ir(
    abs_vars: &[String],
    impl_vars: &[Arc<str>],
    impl_sorts: &[ColSort],
    var_map: &[(String, String)],
    derived_map: &[(String, ValIR)],
) -> Option<Vec<ValIR>> {
    abs_vars
        .iter()
        .map(|av| {
            if let Some((_, ir)) = derived_map.iter().find(|(a, _)| a == av) {
                // RE-VALIDATE a derived entry — never trust the cert's stored expression.
                return map_ir_exact(ir, impl_sorts).then(|| ir.clone());
            }
            let impl_name: &str = var_map
                .iter()
                .find(|(a, _)| a == av)
                .map_or(av.as_str(), |(_, i)| i.as_str());
            impl_vars
                .iter()
                .position(|v| v.as_ref() == impl_name)
                .map(ValIR::Var)
        })
        .collect()
}

/// `f(s)`: the MAPPED ABSTRACT TUPLE — evaluate each abstract variable's mapping expression at
/// the concrete implementation tuple `s`. A projection `Var(i)` yields `s[i]` (any sort,
/// verbatim); a derived affine expression yields its exact aggregate value. `None` fail-closed
/// on overflow / an out-of-fragment node.
#[cfg(feature = "clean-cic")]
fn map_tuple(s: &[u64], mapping: &[ValIR]) -> Option<Vec<u64>> {
    mapping.iter().map(|ir| eval_map_exact(ir, s)).collect()
}

/// Parse a `--map` RHS expression string into a `ValIR` over the IMPLEMENTATION columns, then
/// gate it to the EXACT-AFFINE derived fragment. The RHS is recognized against the same
/// sort-directed recognizer the abstract predicates use, then [`map_ir_exact`]-checked. `None`
/// fail-closed on a parse failure, an unrecognizable expression, or an out-of-fragment shape.
#[cfg(feature = "clean-cic")]
fn recognize_map_rhs(rhs: &str, impl_vars: &[Arc<str>], impl_sorts: &[ColSort]) -> Option<ValIR> {
    use tla_core::ast::Unit;
    // Synthetic module: the implementation VARIABLES + one operator whose body is the RHS. The
    // arithmetic operators lower structurally (`E::Add`/`E::Mul`) independent of `EXTENDS`.
    let decls = impl_vars
        .iter()
        .map(|v| v.as_ref())
        .collect::<Vec<_>>()
        .join(", ");
    let src = format!(
        "---- MODULE __TyMapRhs__ ----\nEXTENDS Integers\nVARIABLES {decls}\n__ty_map_rhs__ == {rhs}\n====\n"
    );
    let tree = tla_core::parse_to_syntax_tree(&src);
    let module = tla_core::lower(tla_core::FileId(0), &tree).module?;
    let body = module.units.iter().find_map(|u| match &u.node {
        Unit::Operator(op) if op.name.node == "__ty_map_rhs__" => Some(&op.body.node),
        _ => None,
    })?;
    let impl_refs: Vec<&str> = impl_vars.iter().map(|v| v.as_ref()).collect();
    let ir = crate::cleancic::recognize_val_sorts(body, &impl_refs, impl_sorts)?;
    map_ir_exact(&ir, impl_sorts).then_some(ir)
}

/// Whether a `--map` RHS is a BARE implementation-variable reference (the PROJECTION case,
/// stored as a name in `var_map` and resolved by column position — preserving the v1
/// projection semantics AND the byte-identical projection schema). A compound expression
/// (`a + b`) is a DERIVED mapping instead.
#[cfg(feature = "clean-cic")]
fn is_bare_ident(s: &str) -> bool {
    let s = s.trim();
    let mut chars = s.chars();
    matches!(chars.next(), Some(c) if c.is_ascii_alphabetic() || c == '_')
        && chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
}

/// The INIT-refinement obligation: `⋀_{s ∈ inits} Init_abs(f(s))` as a kernel Bool term. `None`
/// fail-closed if a mapping expression overflows / is out-of-fragment at any initial state.
#[cfg(feature = "clean-cic")]
fn init_refinement_bool(
    graph_states: &[Vec<u64>],
    inits: &[usize],
    mapping: &[ValIR],
    init_ir: &PredIR,
) -> Option<clean_kernel::Expr> {
    use clean_kernel::Expr;
    let mut acc: Option<Expr> = None;
    for &i in inits {
        let fs = map_tuple(&graph_states[i], mapping)?;
        // Init mentions no primed columns; pass `fs` twice (fixpoint-lane convention).
        let leg = crate::cleancic::embed_pred_ir(init_ir, &fs, &fs);
        acc = Some(match acc {
            None => leg,
            Some(a) => Expr::apps(Expr::const_str("Bool.and"), [a, leg]),
        });
    }
    Some(acc.unwrap_or_else(|| Expr::const_str("Bool.true")))
}

/// The STEP-OR-STUTTER obligation: `⋀_{(s,t) ∈ T} ( Next_abs(f(s), f(t)) ∨ f(t) = f(s) )`.
/// The stutter disjunct is the per-column `Nat.beq` conjunction of the MAPPED tuples — the
/// `[Next_abs]_vars` box-subscript that stuttering-insensitive refinement requires.
#[cfg(feature = "clean-cic")]
fn step_refinement_bool(
    graph_states: &[Vec<u64>],
    transitions: &[(usize, usize)],
    mapping: &[ValIR],
    next_ir: &PredIR,
) -> Option<clean_kernel::Expr> {
    use clean_kernel::Expr;
    let beq = |a: Expr, b: Expr| Expr::apps(Expr::const_str("Nat.beq"), [a, b]);
    let mut acc: Option<Expr> = None;
    for &(from, to) in transitions {
        let fs = map_tuple(&graph_states[from], mapping)?;
        let ft = map_tuple(&graph_states[to], mapping)?;
        let next = crate::cleancic::embed_pred_ir(next_ir, &fs, &ft);
        let mut stutter: Option<Expr> = None;
        for j in 0..fs.len() {
            let eq_j = beq(Expr::nat_lit(ft[j]), Expr::nat_lit(fs[j]));
            stutter = Some(match stutter {
                None => eq_j,
                Some(a) => Expr::apps(Expr::const_str("Bool.and"), [a, eq_j]),
            });
        }
        let stutter = stutter.unwrap_or_else(|| Expr::const_str("Bool.true"));
        let leg = Expr::apps(Expr::const_str("Bool.or"), [next, stutter]);
        acc = Some(match acc {
            None => leg,
            Some(a) => Expr::apps(Expr::const_str("Bool.and"), [a, leg]),
        });
    }
    Some(acc.unwrap_or_else(|| Expr::const_str("Bool.true")))
}

/// Parse a module's VARIABLE declarations into their names (declaration order). Cheap — used to
/// resolve the abstract variable list and to compute the mapped implementation columns before the
/// graph is enumerated.
#[cfg(feature = "clean-cic")]
fn parse_var_names(src: &str) -> Vec<String> {
    use tla_core::ast::Unit;
    let tree = tla_core::parse_to_syntax_tree(src);
    let Some(module) = tla_core::lower(tla_core::FileId(0), &tree).module else {
        return Vec::new();
    };
    module
        .units
        .iter()
        .flat_map(|u| match &u.node {
            Unit::Variable(decls) => decls.iter().map(|d| d.node.clone()).collect::<Vec<_>>(),
            _ => Vec::new(),
        })
        .collect()
}

/// The set of IMPLEMENTATION variable NAMES the refinement mapping references — the columns the
/// projected enumeration must keep (and cell-encode). For each abstract variable: the identifier
/// tokens of its `--map` RHS if present (a bare `impl_var`, or every identifier in a compound
/// `a + b`), else the abstract variable's own name (the default same-name projection). Computed
/// from names only (no sorts/graph), so certify and verify derive it identically BEFORE the graph
/// exists. Over-inclusion is harmless (a non-variable token never matches an implementation column);
/// under-inclusion fail-closes later at mapping resolution — never unsound.
#[cfg(feature = "clean-cic")]
fn mapped_impl_names(
    abs_src: &str,
    var_map: &[(String, String)],
) -> std::collections::BTreeSet<Arc<str>> {
    let mut out: std::collections::BTreeSet<Arc<str>> = std::collections::BTreeSet::new();
    let add_idents = |s: &str, out: &mut std::collections::BTreeSet<Arc<str>>| {
        let mut cur = String::new();
        let flush = |cur: &mut String, out: &mut std::collections::BTreeSet<Arc<str>>| {
            if !cur.is_empty() {
                if cur
                    .chars()
                    .next()
                    .is_some_and(|c| c.is_ascii_alphabetic() || c == '_')
                {
                    out.insert(Arc::<str>::from(cur.as_str()));
                }
                cur.clear();
            }
        };
        for c in s.chars() {
            if c.is_ascii_alphanumeric() || c == '_' {
                cur.push(c);
            } else {
                flush(&mut cur, out);
            }
        }
        flush(&mut cur, out);
    };
    for av in parse_var_names(abs_src) {
        match var_map.iter().find(|(a, _)| *a == av) {
            Some((_, rhs)) => add_idents(rhs, &mut out),
            None => {
                out.insert(Arc::<str>::from(av.as_str()));
            }
        }
    }
    out
}

/// A config's `CONSTANT` bindings as an ORDERED `Vec`, in the config's DECLARATION ORDER
/// (`constants_order`, then any remaining constants sorted by name for determinism). ORDER IS
/// LOAD-BEARING: `bind_constants_from_config` evaluates a `Value(expr)` constant against the ctx
/// built from EARLIER constants, so an interdependent constant (`Cap == M - 1`) must re-bind in
/// the same order at verify — hence the order is carried in the certificate (and covered by the
/// digest), and [`verify_refinement_cert`] restores it into the reconstructed config's
/// `constants_order`. Deterministic, so verify re-derives the identical bindings and graph.
#[cfg(feature = "clean-cic")]
fn constants_ordered(config: &Config) -> Vec<(String, crate::config::ConstantValue)> {
    let mut out: Vec<(String, crate::config::ConstantValue)> = Vec::new();
    let mut seen = std::collections::BTreeSet::new();
    // Declaration order first (so `Value`-expression constants re-bind against the same context).
    for name in &config.constants_order {
        if let Some(val) = config.constants.get(name) {
            if seen.insert(name.clone()) {
                out.push((name.clone(), val.clone()));
            }
        }
    }
    // Any constant not listed in `constants_order` (e.g. programmatically inserted): sorted by name.
    let mut rest: Vec<(&String, &crate::config::ConstantValue)> = config
        .constants
        .iter()
        .filter(|(k, _)| !seen.contains(*k))
        .collect();
    rest.sort_by(|a, b| a.0.cmp(b.0));
    for (k, val) in rest {
        out.push((k.clone(), val.clone()));
    }
    out
}

/// CERTIFY a refinement: enumerate the implementation graph, recognize the abstract spec,
/// build + KERNEL-CHECK both refinement legs, and seal the certificate. `None` (fail-closed)
/// on any out-of-fragment condition or kernel rejection.
#[cfg(feature = "clean-cic")]
pub fn certify_refinement(
    impl_src: &str,
    impl_config: &Config,
    abs_src: &str,
    abs_config: &Config,
    var_map: &[(String, String)],
) -> Option<RefinementCert> {
    // Duplicate abstract keys in the mapping would silently first-match — reject.
    {
        let mut seen = std::collections::BTreeSet::new();
        if !var_map.iter().all(|(a, _)| seen.insert(a.clone())) {
            return None;
        }
    }
    // The mapped implementation columns (needed only if the graph must be PROJECTED because an
    // implementation variable is not cell-encodable). Computed from names, so verify recomputes it
    // identically from the sealed `var_map`.
    let mapped_names = mapped_impl_names(abs_src, var_map);
    let graph = enumerate_impl_graph(impl_src, impl_config, &mapped_names)?;
    let abs_init = abs_config.init.as_deref()?;
    let abs_next = abs_config.next.as_deref()?;

    // Split the `--map` entries into the PROJECTION part (a bare implementation-variable name,
    // stored by name to preserve the v1 projection schema byte-for-byte) and the DERIVED part
    // (a compound affine expression, recognized to a `ValIR` over implementation columns and
    // gated to the exact-affine fragment — fail-closed outside it).
    let mut proj_map: Vec<(String, String)> = Vec::new();
    let mut derived_map: Vec<(String, ValIR)> = Vec::new();
    for (abs, rhs) in var_map {
        if is_bare_ident(rhs) {
            proj_map.push((abs.clone(), rhs.trim().to_string()));
        } else {
            let ir = recognize_map_rhs(rhs, &graph.var_names, &graph.sorts)?;
            derived_map.push((abs.clone(), ir));
        }
    }
    // v1 scope for the PROJECTED (impl-decoupled) path: only PROJECTION mappings. A DERIVED mapping
    // combined with a non-cell-encodable implementation column is out of fragment (verify recomputes
    // the mapped columns from names, which a derived expression's column indices cannot re-anchor) —
    // fail-closed rather than mint a cert verify might not reproduce.
    if graph.projected && !derived_map.is_empty() {
        return None;
    }

    // Two-pass mapping resolution: first the abstract VARIABLE LIST (needs the abstract
    // module), then sorts by mapping, then recognition against those sorts.
    let (abs_vars_probe, _, _) = recognize_abstract(
        abs_src,
        abs_init,
        abs_next,
        // Provisional sorts: recognition is sort-directed, so resolve the mapping first with a
        // cheap parse of the variable list only.
        &probe_abs_sorts(abs_src, &graph, &proj_map, &derived_map)?,
        abs_config,
    )?;
    let mapping = resolve_mapping_ir(
        &abs_vars_probe,
        &graph.var_names,
        &graph.sorts,
        &proj_map,
        &derived_map,
    )?;
    let abs_sorts: Vec<ColSort> = mapping
        .iter()
        .map(|ir| map_result_sort(ir, &graph.sorts))
        .collect::<Option<_>>()?;
    let (abs_vars, init_ir, next_ir) =
        recognize_abstract(abs_src, abs_init, abs_next, &abs_sorts, abs_config)?;

    let init_leg = crate::cleancic::certify_bool_true_obligation(init_refinement_bool(
        &graph.states,
        &graph.inits,
        &mapping,
        &init_ir,
    )?)?;
    let step_leg = crate::cleancic::certify_bool_true_obligation(step_refinement_bool(
        &graph.states,
        &graph.transitions,
        &mapping,
        &next_ir,
    )?)?;

    // Store the mapping SPLIT so a pure-projection cert is byte-identical to v1: every abstract
    // variable with a DERIVED mapping goes to `derived_map` (in abstract declaration order),
    // and every other abstract variable is a projection in `var_map` (default same-name).
    let cert_derived: Vec<(String, ValIR)> = abs_vars
        .iter()
        .filter_map(|av| {
            derived_map
                .iter()
                .find(|(a, _)| a == av)
                .map(|(_, ir)| (av.clone(), ir.clone()))
        })
        .collect();
    let cert_var_map: Vec<(String, String)> = abs_vars
        .iter()
        .filter(|av| !cert_derived.iter().any(|(a, _)| a == *av))
        .map(|av| {
            let impl_name = proj_map
                .iter()
                .find(|(a, _)| a == av)
                .map_or(av.clone(), |(_, i)| i.clone());
            (av.clone(), impl_name)
        })
        .collect();

    let mut cert = RefinementCert {
        schema: SCHEMA_V1.to_string(),
        impl_spec_src: impl_src.to_string(),
        abs_spec_src: abs_src.to_string(),
        impl_init: impl_config.init.clone()?,
        impl_next: impl_config.next.clone()?,
        abs_init: abs_init.to_string(),
        abs_next: abs_next.to_string(),
        var_map: cert_var_map,
        derived_map: cert_derived,
        impl_reachable: graph.states,
        impl_inits: graph.inits,
        transitions: graph.transitions,
        impl_sorts: graph.sorts,
        abs_init_ir: init_ir,
        abs_next_ir: next_ir,
        impl_constants: constants_ordered(impl_config),
        abs_constants: constants_ordered(abs_config),
        init_leg,
        step_leg,
        digest: String::new(),
    };
    cert.digest = cert.compute_digest();
    Some(cert)
}

/// Resolve the abstract sorts for the mapping without a chicken-and-egg on recognition: parse
/// only the abstract VARIABLE list, then derive each abstract variable's sort from its resolved
/// mapping expression (a projection inherits the implementation column's sort; a derived affine
/// expression is `Int`).
#[cfg(feature = "clean-cic")]
fn probe_abs_sorts(
    abs_src: &str,
    graph: &ImplGraph,
    var_map: &[(String, String)],
    derived_map: &[(String, ValIR)],
) -> Option<Vec<ColSort>> {
    use tla_core::ast::Unit;
    let tree = tla_core::parse_to_syntax_tree(abs_src);
    let module = tla_core::lower(tla_core::FileId(0), &tree).module?;
    let abs_vars: Vec<String> = module
        .units
        .iter()
        .flat_map(|u| match &u.node {
            Unit::Variable(decls) => decls.iter().map(|d| d.node.clone()).collect::<Vec<_>>(),
            _ => Vec::new(),
        })
        .collect();
    let mapping = resolve_mapping_ir(
        &abs_vars,
        &graph.var_names,
        &graph.sorts,
        var_map,
        derived_map,
    )?;
    mapping
        .iter()
        .map(|ir| map_result_sort(ir, &graph.sorts))
        .collect()
}

/// Independently RE-CHECK a refinement certificate: digest, re-enumerated implementation
/// graph equality (binds the graph to the IMPLEMENTATION spec), re-recognized abstract IRs
/// (binds the predicates to the ABSTRACT spec), and the kernel re-check of both legs at
/// obligation types REBUILT from the re-derived data. Fail-closed.
#[cfg(feature = "clean-cic")]
#[must_use]
pub fn verify_refinement_cert(cert: &RefinementCert) -> bool {
    if cert.schema != SCHEMA_V1 || cert.compute_digest() != cert.digest {
        return false;
    }
    // Reconstruct BOTH configs from the cert (init/next + the sealed CONSTANT bindings), so the
    // implementation graph re-enumerates under identical constants and the abstract recognizer
    // rebuilds the identical model-value sets. The digest already binds these fields.
    // Restore the DECLARATION ORDER (`constants_order`) as well as the map, so an interdependent
    // `Value`-expression constant re-binds against the identical earlier-constant context.
    let impl_config = Config {
        init: Some(cert.impl_init.clone()),
        next: Some(cert.impl_next.clone()),
        constants: cert.impl_constants.iter().cloned().collect(),
        constants_order: cert.impl_constants.iter().map(|(k, _)| k.clone()).collect(),
        ..Default::default()
    };
    let abs_config = Config {
        init: Some(cert.abs_init.clone()),
        next: Some(cert.abs_next.clone()),
        constants: cert.abs_constants.iter().cloned().collect(),
        constants_order: cert.abs_constants.iter().map(|(k, _)| k.clone()).collect(),
        ..Default::default()
    };
    // Recompute the mapped implementation columns identically to certify (from the sealed
    // `var_map`), so the PROJECTED path keeps the SAME columns and re-enumerates the SAME graph.
    let mapped_names = mapped_impl_names(&cert.abs_spec_src, &cert.var_map);
    let Some(graph) = enumerate_impl_graph(&cert.impl_spec_src, &impl_config, &mapped_names) else {
        return false;
    };
    if graph.states != cert.impl_reachable
        || graph.inits != cert.impl_inits
        || graph.transitions != cert.transitions
        || graph.sorts != cert.impl_sorts
    {
        return false;
    }
    let Some(abs_sorts) =
        probe_abs_sorts(&cert.abs_spec_src, &graph, &cert.var_map, &cert.derived_map)
    else {
        return false;
    };
    let Some((abs_vars, init_ir, next_ir)) = recognize_abstract(
        &cert.abs_spec_src,
        &cert.abs_init,
        &cert.abs_next,
        &abs_sorts,
        &abs_config,
    ) else {
        return false;
    };
    if init_ir != cert.abs_init_ir || next_ir != cert.abs_next_ir {
        return false;
    }
    // Re-derive the mapping from the cert's STORED projection + derived expressions (never
    // trusting cert-supplied mapped tuples): a tampered/re-sealed out-of-fragment derived
    // expression is rejected inside `resolve_mapping_ir`, and `map_tuple` recomputes `f(s)`.
    let Some(mapping) = resolve_mapping_ir(
        &abs_vars,
        &graph.var_names,
        &graph.sorts,
        &cert.var_map,
        &cert.derived_map,
    ) else {
        return false;
    };
    let (Some(init_obl), Some(step_obl)) = (
        init_refinement_bool(&graph.states, &graph.inits, &mapping, &init_ir),
        step_refinement_bool(&graph.states, &graph.transitions, &mapping, &next_ir),
    ) else {
        return false;
    };
    let init_ok = crate::cleancic::verify_bool_true_obligation(init_obl, &cert.init_leg);
    let step_ok = crate::cleancic::verify_bool_true_obligation(step_obl, &cert.step_leg);
    init_ok && step_ok
}

#[cfg(all(test, feature = "clean-cic"))]
mod tests {
    use super::*;

    fn cfg(init: &str, next: &str) -> Config {
        Config {
            init: Some(init.into()),
            next: Some(next.into()),
            ..Default::default()
        }
    }

    /// FINITE-ENUM truth-direction filter: `pred_exact` admits enum EQUALITY on an in-range label
    /// index but REJECTS ordering (`< / ≥`) and out-of-range indices — an enum index carries no
    /// order, and admitting one would be a false certificate.
    #[test]
    fn enum_equality_admitted_ordering_rejected() {
        use crate::explicit_fixpoint_cert::{ColSort, EnumKind, PredIR, ValIR};
        let sorts = [ColSort::Enum {
            labels: vec!["a".into(), "b".into(), "c".into()],
            kind: EnumKind::Str,
        }];
        assert!(
            pred_exact(&PredIR::Eq(ValIR::Var(0), ValIR::Lit(1)), &sorts),
            "in-range enum = admitted"
        );
        assert!(
            pred_exact(&PredIR::Neq(ValIR::Var(0), ValIR::Lit(2)), &sorts),
            "enum ≠ admitted"
        );
        assert!(
            !pred_exact(&PredIR::Lt(ValIR::Var(0), ValIR::Lit(1)), &sorts),
            "enum < rejected"
        );
        assert!(
            !pred_exact(&PredIR::Geq(ValIR::Var(0), ValIR::Lit(1)), &sorts),
            "enum ≥ rejected"
        );
        assert!(
            !pred_exact(&PredIR::Eq(ValIR::Var(0), ValIR::Lit(3)), &sorts),
            "an index past the observed label set is rejected"
        );
        // A bare enum column with no enum/lit partner (e.g. against an Int-literal-less nonsense) —
        // `Prime` in-range is fine, but pairing with a non-enum column must fail.
        let mixed = [
            ColSort::Enum {
                labels: vec!["a".into(), "b".into()],
                kind: EnumKind::Str,
            },
            ColSort::Int,
        ];
        assert!(
            !pred_exact(&PredIR::Eq(ValIR::Var(0), ValIR::Var(1)), &mixed),
            "enum column vs Int column is not an enum equality"
        );
    }

    /// Cross-column enum equality is EXACT only when the two columns share the SAME label space
    /// (identical label→index maps); different spaces are REJECTED (index eq would not track label eq).
    #[test]
    fn cross_column_enum_equality_same_space_only() {
        use crate::explicit_fixpoint_cert::{ColSort, EnumKind, PredIR, ValIR};
        let same = [
            ColSort::Enum {
                labels: vec!["a".into(), "b".into()],
                kind: EnumKind::Str,
            },
            ColSort::Enum {
                labels: vec!["a".into(), "b".into()],
                kind: EnumKind::Str,
            },
        ];
        assert!(
            pred_exact(&PredIR::Eq(ValIR::Var(0), ValIR::Var(1)), &same),
            "same space admitted"
        );
        let diff = [
            ColSort::Enum {
                labels: vec!["a".into(), "b".into()],
                kind: EnumKind::Str,
            },
            ColSort::Enum {
                labels: vec!["b".into(), "c".into()],
                kind: EnumKind::Str,
            },
        ];
        assert!(
            !pred_exact(&PredIR::Eq(ValIR::Var(0), ValIR::Var(1)), &diff),
            "different label spaces rejected (unsound otherwise)"
        );
    }

    /// The func-of-enum DIGIT `(pack / base^i) mod base` (`base = |labels|`, `i < arity`) — the shape
    /// `cleancic::func_enum_app` emits.
    #[cfg(feature = "clean-cic")]
    fn fe_digit(col: usize, place: u64, base: u64) -> crate::explicit_fixpoint_cert::ValIR {
        use crate::explicit_fixpoint_cert::ValIR as V;
        V::Mod(
            Box::new(V::Div(Box::new(V::Var(col)), Box::new(V::Lit(place)))),
            Box::new(V::Lit(base)),
        )
    }

    /// FUNC-of-ENUM truth-direction filter: `pred_exact` admits func-enum digit EQUALITY on an in-range
    /// label index but REJECTS ordering (`< / ≥`), an out-of-range index, a wrong place value, and a bare
    /// non-digit operand — a func-enum digit carries no order, and admitting one would be a false cert.
    #[test]
    fn func_enum_equality_admitted_ordering_rejected() {
        use crate::explicit_fixpoint_cert::{ColSort, PredIR, ValIR};
        // pc: FuncEnum{arity:3, labels:["a","b"]} ⇒ base 2; place at slot 1 is 2^1 = 2, slot 0 is 2^0 = 1.
        let sorts = [ColSort::FuncEnum {
            arity: 3,
            labels: vec!["a".into(), "b".into()],
            dom: vec![],
            dom_kind: crate::explicit_fixpoint_cert::EnumKind::Model,
        }];
        assert!(
            pred_exact(&PredIR::Eq(fe_digit(0, 2, 2), ValIR::Lit(0)), &sorts),
            "in-range func-enum digit = admitted (pc[1] = \"a\")"
        );
        assert!(
            pred_exact(&PredIR::Neq(fe_digit(0, 1, 2), ValIR::Lit(1)), &sorts),
            "func-enum digit ≠ admitted (pc[0] # \"b\")"
        );
        assert!(
            !pred_exact(&PredIR::Lt(fe_digit(0, 2, 2), ValIR::Lit(1)), &sorts),
            "func-enum digit < rejected (ordering is meaningless on an enum index)"
        );
        assert!(
            !pred_exact(&PredIR::Geq(fe_digit(0, 2, 2), ValIR::Lit(1)), &sorts),
            "func-enum digit ≥ rejected"
        );
        assert!(
            !pred_exact(&PredIR::Eq(fe_digit(0, 2, 2), ValIR::Lit(2)), &sorts),
            "an index past the observed label set (2 ≥ |labels|) is rejected"
        );
        assert!(
            !pred_exact(&PredIR::Eq(fe_digit(0, 8, 2), ValIR::Lit(0)), &sorts),
            "a place value that is not base^i for any i < arity is rejected (not a real digit)"
        );
        // A wrong modulus (not |labels|) is not a func-enum digit ⇒ falls to the strict Int gate ⇒ rejected.
        assert!(
            !pred_exact(&PredIR::Eq(fe_digit(0, 2, 3), ValIR::Lit(0)), &sorts),
            "a modulus ≠ |labels| is not a func-enum digit"
        );
    }

    /// Cross-column func-enum digit equality is EXACT only when the two columns share the SAME label space;
    /// different spaces are REJECTED (digit-index eq would not track label eq).
    #[test]
    fn cross_column_func_enum_equality_same_space_only() {
        use crate::explicit_fixpoint_cert::{ColSort, PredIR};
        let same = [
            ColSort::FuncEnum {
                arity: 3,
                labels: vec!["a".into(), "b".into()],
                dom: vec![],
                dom_kind: crate::explicit_fixpoint_cert::EnumKind::Model,
            },
            ColSort::FuncEnum {
                arity: 3,
                labels: vec!["a".into(), "b".into()],
                dom: vec![],
                dom_kind: crate::explicit_fixpoint_cert::EnumKind::Model,
            },
        ];
        assert!(
            pred_exact(&PredIR::Eq(fe_digit(0, 2, 2), fe_digit(1, 2, 2)), &same),
            "same label space admitted (pc1[1] = pc2[1])"
        );
        let diff = [
            ColSort::FuncEnum {
                arity: 3,
                labels: vec!["a".into(), "b".into()],
                dom: vec![],
                dom_kind: crate::explicit_fixpoint_cert::EnumKind::Model,
            },
            ColSort::FuncEnum {
                arity: 3,
                labels: vec!["a".into(), "c".into()],
                dom: vec![],
                dom_kind: crate::explicit_fixpoint_cert::EnumKind::Model,
            },
        ];
        assert!(
            !pred_exact(&PredIR::Eq(fe_digit(0, 2, 2), fe_digit(1, 2, 2)), &diff),
            "different label spaces rejected (unsound otherwise)"
        );
    }

    const ABS_COUNTER: &str = "---- MODULE Abs ----\n\
                               EXTENDS Integers\n\
                               VARIABLE x\n\
                               AInit == x = 0\n\
                               ANext == x' = x + 1 /\\ x < 10\n\
                               ====\n";

    /// A tight counter refines a loose one: every impl transition satisfies the abstract
    /// Next; the kernel re-evaluates the abstract predicates at every mapped pair.
    #[test]
    fn counter_refines_looser_counter() {
        let impl_spec = "---- MODULE Impl ----\n\
                         EXTENDS Integers\n\
                         VARIABLE x\n\
                         Init == x = 0\n\
                         Next == x' = x + 1 /\\ x < 3\n\
                         ====\n";
        let cert = certify_refinement(
            impl_spec,
            &cfg("Init", "Next"),
            ABS_COUNTER,
            &cfg("AInit", "ANext"),
            &[],
        )
        .expect("the tight counter must refine the loose one");
        assert_eq!(cert.impl_reachable.len(), 4, "R = {{0,1,2,3}}");
        assert_eq!(cert.transitions.len(), 3);
        assert!(
            verify_refinement_cert(&cert),
            "the refinement cert must re-verify"
        );
    }

    /// STUTTERING: an implementation with a private counter `y` whose `y`-ticks leave the
    /// abstract state unchanged refines the abstract counter — the stutter disjunct of
    /// `[Next_abs]_vars` carries those steps. THE load-bearing TLA+ semantics.
    #[test]
    fn stuttering_impl_refines_abstract_counter() {
        let impl_spec = "---- MODULE ImplStutter ----\n\
                         EXTENDS Integers\n\
                         VARIABLES x, y\n\
                         Init == x = 0 /\\ y = 0\n\
                         Next == (x' = x /\\ y' = y + 1 /\\ y < 2)\n\
                              \\/ (x' = x + 1 /\\ y' = 0 /\\ y = 2 /\\ x < 2)\n\
                         ====\n";
        let cert = certify_refinement(
            impl_spec,
            &cfg("Init", "Next"),
            ABS_COUNTER,
            &cfg("AInit", "ANext"),
            &[],
        )
        .expect("the stuttering implementation must refine the abstract counter");
        // The mapping projects (x, y) to x: y-ticks are STUTTERS (f(s) == f(t)), x-steps
        // are real abstract steps — both classes must be present and carried.
        let x_col = 0usize;
        let stutters = cert
            .transitions
            .iter()
            .filter(|(a, b)| cert.impl_reachable[*a][x_col] == cert.impl_reachable[*b][x_col])
            .count();
        let steps = cert.transitions.len() - stutters;
        assert!(stutters > 0, "y-ticks must appear as abstract stutters");
        assert!(steps > 0, "x-increments must appear as abstract steps");
        assert!(verify_refinement_cert(&cert));
    }

    /// FAIL-CLOSED: an implementation that steps by 2 does NOT refine the +1 counter — the
    /// kernel reduces the step obligation to Bool.false and certification declines.
    #[test]
    fn non_refinement_fails_closed() {
        let impl_spec = "---- MODULE ImplBad ----\n\
                         EXTENDS Integers\n\
                         VARIABLE x\n\
                         Init == x = 0\n\
                         Next == x' = x + 2 /\\ x < 4\n\
                         ====\n";
        assert!(
            certify_refinement(
                impl_spec,
                &cfg("Init", "Next"),
                ABS_COUNTER,
                &cfg("AInit", "ANext"),
                &[],
            )
            .is_none(),
            "x'=x+2 must NOT be certified as refining x'=x+1 (kernel refutes the step leg)"
        );
    }

    /// FAIL-CLOSED: a wrong initial state (Init x=5 vs abstract Init x=0) declines at the
    /// init-refinement leg.
    #[test]
    fn wrong_init_fails_closed() {
        let impl_spec = "---- MODULE ImplBadInit ----\n\
                         EXTENDS Integers\n\
                         VARIABLE x\n\
                         Init == x = 5\n\
                         Next == x' = x + 1 /\\ x < 7\n\
                         ====\n";
        assert!(certify_refinement(
            impl_spec,
            &cfg("Init", "Next"),
            ABS_COUNTER,
            &cfg("AInit", "ANext"),
            &[],
        )
        .is_none());
    }

    /// TRUTH-DIRECTION SOUNDNESS (review-confirmed PoC, now a permanent regression): the
    /// fixpoint recognizer's `Nat.sub`-truncating `v = a - b` form would let the kernel
    /// evaluate `0 = 0 - 1` as TRUE — certifying `x' = x` as refining `x' = x - 1`. The
    /// exactness filter must DECLINE any abstract spec using subtraction.
    #[test]
    fn sub_truncation_cannot_mint_false_refinement() {
        let impl_spec = "---- MODULE ImplStay ----\n\
                         EXTENDS Integers\n\
                         VARIABLES x, y\n\
                         Init == x = 0 /\\ y = 0\n\
                         Next == x' = x /\\ y' = y + 1 /\\ y < 2\n\
                         ====\n";
        let abs_spec = "---- MODULE AbsDec ----\n\
                        EXTENDS Integers\n\
                        VARIABLES x, y\n\
                        AInit == x = 0 /\\ y = 0\n\
                        ANext == x' = x - 1 /\\ y' = y + 1 /\\ y < 2\n\
                        ====\n";
        assert!(
            certify_refinement(
                impl_spec,
                &cfg("Init", "Next"),
                abs_spec,
                &cfg("AInit", "ANext"),
                &[],
            )
            .is_none(),
            "an abstract Next using `-` (Nat-truncating in the embedding) must DECLINE — \
             TLA says 0 = 0-1 is FALSE; the kernel's Nat.sub would say true"
        );
    }

    /// TRUTH-DIRECTION SOUNDNESS: a Bool implementation cell must not pun as a Nat in the
    /// abstract predicates (`FALSE + 1` is not TLA arithmetic). The sort-exactness rule
    /// requires Int columns under every arithmetic/comparison reference.
    #[test]
    fn bool_cell_cannot_pun_as_nat() {
        let impl_spec = "---- MODULE ImplBool ----\n\
                         VARIABLE b\n\
                         Init == b = FALSE\n\
                         Next == b' = TRUE /\\ b = FALSE\n\
                         ====\n";
        let abs_spec = "---- MODULE AbsArith ----\n\
                        EXTENDS Integers\n\
                        VARIABLE y\n\
                        AInit == y = 0\n\
                        ANext == y' = y + 1\n\
                        ====\n";
        assert!(
            certify_refinement(
                impl_spec,
                &cfg("Init", "Next"),
                abs_spec,
                &cfg("AInit", "ANext"),
                &[("y".to_string(), "b".to_string())],
            )
            .is_none(),
            "a Bool column mapped into abstract Nat arithmetic must DECLINE"
        );
    }

    /// TRUTH-DIRECTION SOUNDNESS: a Set implementation cell (bitmask) must not compare
    /// against an abstract Int literal (`{0,1} = 3` is not TLA-true; the raw cell is 3).
    #[test]
    fn set_cell_cannot_pun_as_nat_literal() {
        let impl_spec = "---- MODULE ImplSet ----\n\
                         EXTENDS Integers\n\
                         VARIABLE s\n\
                         Init == s = {0, 1}\n\
                         Next == s' = s\n\
                         ====\n";
        let abs_spec = "---- MODULE AbsLit ----\n\
                        EXTENDS Integers\n\
                        VARIABLE s\n\
                        AInit == s = 3\n\
                        ANext == s' = s\n\
                        ====\n";
        assert!(
            certify_refinement(
                impl_spec,
                &cfg("Init", "Next"),
                abs_spec,
                &cfg("AInit", "ANext"),
                &[],
            )
            .is_none(),
            "a Set bitmask cell compared to an abstract Int literal must DECLINE"
        );
    }

    /// Configured CONSTANTs are now CARRIED by the certificate (sorted, digest-bound) and re-bound
    /// at verify time, so a spec with a bound constant certifies and re-verifies — the graph
    /// re-enumerates under the identical binding. (A tampered constant breaks the digest; a
    /// constant that changed the graph would fail the re-enumeration binding.)
    #[test]
    fn configured_constants_carried_and_reverify() {
        let impl_spec = "---- MODULE ImplC ----\n\
                         EXTENDS Integers\n\
                         VARIABLE x\n\
                         Init == x = 0\n\
                         Next == x' = x + 1 /\\ x < 3\n\
                         ====\n";
        let mut c = cfg("Init", "Next");
        c.constants.insert(
            "N".to_string(),
            crate::config::ConstantValue::Value("3".to_string()),
        );
        let cert = certify_refinement(impl_spec, &c, ABS_COUNTER, &cfg("AInit", "ANext"), &[])
            .expect("a bound (unused) constant is carried, not a decline");
        assert_eq!(
            cert.impl_constants,
            vec![(
                "N".to_string(),
                crate::config::ConstantValue::Value("3".to_string())
            )]
        );
        assert!(
            verify_refinement_cert(&cert),
            "the constant re-binds and the cert re-verifies"
        );
        // Tampering the carried constant breaks the digest.
        let mut t = cert;
        t.impl_constants = vec![(
            "N".to_string(),
            crate::config::ConstantValue::Value("4".to_string()),
        )];
        assert!(
            !verify_refinement_cert(&t),
            "a tampered constant fails the digest"
        );
    }

    /// CONSTANT declaration ORDER is carried and RESTORED at verify: an interdependent constant
    /// `Cap == M - 1` (a `Value` expression evaluated against the earlier-bound `M`) must re-bind in
    /// the SAME order at verify. Carrying `constants_order` in the cert is what makes this re-verify;
    /// without it, verify would bind lexically (`Cap` before `M`), fail to evaluate `M - 1`, and
    /// reject a genuinely-refining certificate (fail-closed, but a false NEGATIVE).
    #[test]
    fn interdependent_constants_order_carried() {
        use crate::config::ConstantValue;
        let impl_spec = "---- MODULE ImplDep ----\n\
                         EXTENDS Integers\n\
                         CONSTANTS M, Cap\n\
                         VARIABLE x\n\
                         Init == x = 0\n\
                         Next == x' = x + 1 /\\ x < Cap\n\
                         ====\n";
        let mut c = cfg("Init", "Next");
        c.constants
            .insert("M".to_string(), ConstantValue::Value("3".to_string()));
        c.constants
            .insert("Cap".to_string(), ConstantValue::Value("M - 1".to_string()));
        // `Cap` DEPENDS on `M`, so `M` must bind first — the load-bearing declaration order.
        c.constants_order = vec!["M".to_string(), "Cap".to_string()];
        let cert = certify_refinement(impl_spec, &c, ABS_COUNTER, &cfg("AInit", "ANext"), &[])
            .expect("an interdependent-constant implementation certifies");
        assert_eq!(
            cert.impl_constants,
            vec![
                ("M".to_string(), ConstantValue::Value("3".to_string())),
                ("Cap".to_string(), ConstantValue::Value("M - 1".to_string())),
            ],
            "constants are carried in DECLARATION order (M before Cap)"
        );
        assert!(
            verify_refinement_cert(&cert),
            "verify must restore constants_order and re-bind M before Cap, then re-verify"
        );
    }

    /// Duplicate abstract keys in the mapping are rejected (silent first-match is a footgun).
    #[test]
    fn duplicate_map_keys_decline() {
        let impl_spec = "---- MODULE ImplD ----\n\
                         EXTENDS Integers\n\
                         VARIABLE x\n\
                         Init == x = 0\n\
                         Next == x' = x + 1 /\\ x < 3\n\
                         ====\n";
        let m = vec![
            ("x".to_string(), "x".to_string()),
            ("x".to_string(), "x".to_string()),
        ];
        assert!(certify_refinement(
            impl_spec,
            &cfg("Init", "Next"),
            ABS_COUNTER,
            &cfg("AInit", "ANext"),
            &m
        )
        .is_none());
    }

    /// TAMPER: mutating the transition set (or any bound field) breaks the digest; resealing
    /// the digest still fails the re-enumeration binding. Never a silently-accepted edit.
    #[test]
    fn tampered_cert_rejected() {
        let impl_spec = "---- MODULE Impl ----\n\
                         EXTENDS Integers\n\
                         VARIABLE x\n\
                         Init == x = 0\n\
                         Next == x' = x + 1 /\\ x < 3\n\
                         ====\n";
        let cert = certify_refinement(
            impl_spec,
            &cfg("Init", "Next"),
            ABS_COUNTER,
            &cfg("AInit", "ANext"),
            &[],
        )
        .expect("certifies");
        let mut tampered = cert.clone();
        tampered.transitions.pop();
        assert!(
            !verify_refinement_cert(&tampered),
            "digest catches the edit"
        );
        tampered.digest = tampered.compute_digest();
        assert!(
            !verify_refinement_cert(&tampered),
            "re-enumeration binding catches a resealed transition-set edit"
        );
        // An explicit variable-map to a non-existent impl variable fails closed.
        let mut remapped = cert;
        remapped.var_map = vec![("x".to_string(), "nope".to_string())];
        remapped.digest = remapped.compute_digest();
        assert!(!verify_refinement_cert(&remapped));
    }

    // ── PROJECTED (impl-decoupled) refinement: an UNMAPPED non-cell-encodable variable. ──

    /// The impl-decoupling unblock (the `TwoPhase.msgs` shape, in miniature): an implementation with
    /// an UNMAPPED, non-cell-encodable variable (`soup`, a set of RECORDS) refines a cell-encodable
    /// abstract counter via a PROJECTION mapping (`x <- x`). The graph is identified by the FULL
    /// canonical state (collision-free — no reachable state or transition dropped), but ONLY the
    /// mapped column `x` is cell-encoded; the soup is dropped (it exists solely to distinguish full
    /// states). Certifies + re-verifies, and the cert stores ONLY the mapped column. A broken variant
    /// (`x` steps by 2) DECLINES at the kernel step leg — the same soundness oracle as the full path.
    #[test]
    fn projected_unmapped_noncell_var_refines_counter() {
        // `soup` starts `{}` (a cell-encodable empty set) and becomes a set of RECORDS
        // (non-cell-encodable) — so the FULL path proceeds at init then declines mid-DFS with
        // `NonEncodable`, exercising the fallback to the projected path robustly.
        let impl_ok = "---- MODULE ImplSoup ----\n\
                       EXTENDS Integers\n\
                       VARIABLES x, soup\n\
                       Init == x = 0 /\\ soup = {}\n\
                       Next == x' = x + 1 /\\ x < 3 /\\ soup' = soup \\cup {[a |-> x]}\n\
                       ====\n";
        let cert = certify_refinement(
            impl_ok,
            &cfg("Init", "Next"),
            ABS_COUNTER,
            &cfg("AInit", "ANext"),
            &[],
        )
        .expect("the projected (soup-decoupled) refinement must certify");
        assert_eq!(
            cert.impl_sorts,
            vec![ColSort::Int],
            "ONLY the mapped column x is stored/encoded — the non-encodable soup is dropped"
        );
        assert_eq!(
            cert.impl_reachable.len(),
            4,
            "projected R over x = {{0,1,2,3}}"
        );
        assert_eq!(cert.var_map, vec![("x".to_string(), "x".to_string())]);
        assert!(
            verify_refinement_cert(&cert),
            "the projected cert must re-verify"
        );

        // Broken variant: `x` steps by 2 — NOT a refinement of `x' = x + 1`; the kernel refutes the
        // step leg even though the SAME non-encodable soup forces the projected path.
        let impl_bad = "---- MODULE ImplSoupBad ----\n\
                        EXTENDS Integers\n\
                        VARIABLES x, soup\n\
                        Init == x = 0 /\\ soup = {}\n\
                        Next == x' = x + 2 /\\ x < 4 /\\ soup' = soup \\cup {[a |-> x]}\n\
                        ====\n";
        assert!(
            certify_refinement(impl_bad, &cfg("Init", "Next"), ABS_COUNTER, &cfg("AInit", "ANext"), &[])
                .is_none(),
            "x'=x+2 must NOT certify as refining x'=x+1 on the projected path (kernel refutes the step)"
        );
    }

    // ── DERIVED (affine) refinement mappings — the DATA-REFINEMENT lane (`sum <- a + b`). ──

    /// The two-counter → sum DATA REFINEMENT: impl `VARIABLES a, b` step both by 1; abstract
    /// `VARIABLE sum` steps by 2 under the DERIVED mapping `sum <- a + b`. Every impl transition
    /// maps to an abstract step (the kernel re-evaluates `SNext` at the derived aggregate), and
    /// the cert re-verifies. THE deliverable: ty kernel-certifies an aggregate mapping.
    fn two_counter_impl() -> &'static str {
        "---- MODULE Impl ----\n\
         EXTENDS Integers\n\
         VARIABLES a, b\n\
         Init == a = 0 /\\ b = 0\n\
         Next == a' = a + 1 /\\ b' = b + 1 /\\ a < 3\n\
         ====\n"
    }
    const ABS_SUM: &str = "---- MODULE AbsSum ----\n\
                           EXTENDS Integers\n\
                           VARIABLE sum\n\
                           SInit == sum = 0\n\
                           SNext == sum' = sum + 2 /\\ sum < 6\n\
                           ====\n";

    #[test]
    fn two_counter_sum_data_refinement_certifies_and_verifies() {
        let cert = certify_refinement(
            two_counter_impl(),
            &cfg("Init", "Next"),
            ABS_SUM,
            &cfg("SInit", "SNext"),
            &[("sum".to_string(), "a + b".to_string())],
        )
        .expect("the two-counter impl must refine the +2 abstract counter under sum <- a + b");
        assert_eq!(
            cert.impl_reachable.len(),
            4,
            "R = {{(0,0),(1,1),(2,2),(3,3)}}"
        );
        assert_eq!(cert.transitions.len(), 3);
        // The mapping is DERIVED (`sum`), so it lives in `derived_map`, NOT `var_map`.
        assert!(
            cert.var_map.is_empty(),
            "no projection variables — sum is derived"
        );
        assert_eq!(cert.derived_map.len(), 1);
        assert_eq!(cert.derived_map[0].0, "sum");
        assert_eq!(
            cert.derived_map[0].1,
            ValIR::Add(Box::new(ValIR::Var(0)), Box::new(ValIR::Var(1))),
            "sum <- a + b, with a=col0, b=col1 in impl declaration order"
        );
        assert!(
            verify_refinement_cert(&cert),
            "the data-refinement cert must re-verify"
        );
    }

    /// FAIL-CLOSED: an implementation where `a + b` does NOT track the abstract `sum' = sum + 2`
    /// (only `a` moves, so the aggregate steps by 1) declines — the kernel refutes the step leg.
    #[test]
    fn non_refinement_under_derived_map_fails_closed() {
        let impl_spec = "---- MODULE ImplHalf ----\n\
                         EXTENDS Integers\n\
                         VARIABLES a, b\n\
                         Init == a = 0 /\\ b = 0\n\
                         Next == a' = a + 1 /\\ b' = b /\\ a < 3\n\
                         ====\n";
        assert!(
            certify_refinement(
                impl_spec,
                &cfg("Init", "Next"),
                ABS_SUM,
                &cfg("SInit", "SNext"),
                &[("sum".to_string(), "a + b".to_string())],
            )
            .is_none(),
            "a+b stepping by 1 must NOT certify as refining sum'=sum+2 (kernel refutes the step)"
        );
    }

    /// TAMPER on a DERIVED cert: (1) mutating the mapping expression to one that is NOT a
    /// refinement and re-sealing the digest still fails (the kernel re-checks the rebuilt
    /// obligation); (2) mutating it to an OUT-OF-FRAGMENT expression (`Sub`) fails at the
    /// exactness re-gate; (3) mutating the abstract IR fails the re-recognition binding.
    #[test]
    fn tampered_derived_cert_rejected() {
        let cert = certify_refinement(
            two_counter_impl(),
            &cfg("Init", "Next"),
            ABS_SUM,
            &cfg("SInit", "SNext"),
            &[("sum".to_string(), "a + b".to_string())],
        )
        .expect("certifies");
        // (1) `sum <- a` (bare projection, in-fragment) is not a +2 refinement ⇒ rejected.
        let mut t1 = cert.clone();
        t1.derived_map = vec![("sum".to_string(), ValIR::Var(0))];
        t1.digest = t1.compute_digest();
        assert!(
            !verify_refinement_cert(&t1),
            "a resealed non-refining mapping is refuted"
        );
        // (2) `sum <- a - b` is OUTSIDE the exact-affine fragment ⇒ rejected at the re-gate.
        let mut t2 = cert.clone();
        t2.derived_map = vec![(
            "sum".to_string(),
            ValIR::Sub(Box::new(ValIR::Var(0)), Box::new(ValIR::Var(1))),
        )];
        t2.digest = t2.compute_digest();
        assert!(
            !verify_refinement_cert(&t2),
            "an out-of-fragment mapping is rejected"
        );
        // (3) mutating the abstract Next IR breaks the re-recognition binding.
        let mut t3 = cert;
        t3.abs_next_ir = PredIR::BoolLit(true);
        t3.digest = t3.compute_digest();
        assert!(
            !verify_refinement_cert(&t3),
            "a mutated abstract IR is rejected"
        );
    }

    /// FAIL-CLOSED: a DERIVED expression outside the EXACT-AFFINE fragment declines at certify.
    /// `sum <- a - b` uses subtraction (Nat-truncating, could go negative); `sum <- a + flag`
    /// adds a NON-Int (Bool) column. Both must be refused — never a weaker cert.
    #[test]
    fn out_of_fragment_derived_map_declines() {
        // (a) subtraction.
        assert!(
            certify_refinement(
                two_counter_impl(),
                &cfg("Init", "Next"),
                ABS_SUM,
                &cfg("SInit", "SNext"),
                &[("sum".to_string(), "a - b".to_string())],
            )
            .is_none(),
            "a derived mapping using `-` must decline (Nat-truncating, unsound)"
        );
        // (b) a non-Int (Bool) column under `+`.
        let impl_bool = "---- MODULE ImplFlag ----\n\
                         EXTENDS Integers\n\
                         VARIABLES a, flag\n\
                         Init == a = 0 /\\ flag = FALSE\n\
                         Next == a' = a + 1 /\\ flag' = flag /\\ a < 2\n\
                         ====\n";
        assert!(
            certify_refinement(
                impl_bool,
                &cfg("Init", "Next"),
                ABS_SUM,
                &cfg("SInit", "SNext"),
                &[("sum".to_string(), "a + flag".to_string())],
            )
            .is_none(),
            "a Bool column under derived arithmetic must decline (would pun as a Nat)"
        );
    }

    /// A DERIVED mapping and a bare PROJECTION can COEXIST in one certificate: abstract `sum` is
    /// derived (`a + b`) while abstract `k` projects impl `c` — the split storage (`derived_map`
    /// + `var_map`) round-trips and re-verifies.
    #[test]
    fn mixed_projection_and_derived_mapping() {
        let impl_spec = "---- MODULE ImplMix ----\n\
                         EXTENDS Integers\n\
                         VARIABLES a, b, c\n\
                         Init == a = 0 /\\ b = 0 /\\ c = 0\n\
                         Next == a' = a + 1 /\\ b' = b + 1 /\\ c' = c + 1 /\\ a < 2\n\
                         ====\n";
        let abs_spec = "---- MODULE AbsMix ----\n\
                        EXTENDS Integers\n\
                        VARIABLES sum, k\n\
                        MInit == sum = 0 /\\ k = 0\n\
                        MNext == sum' = sum + 2 /\\ k' = k + 1 /\\ k < 2\n\
                        ====\n";
        let cert = certify_refinement(
            impl_spec,
            &cfg("Init", "Next"),
            abs_spec,
            &cfg("MInit", "MNext"),
            &[
                ("sum".to_string(), "a + b".to_string()),
                ("k".to_string(), "c".to_string()),
            ],
        )
        .expect("mixed derived+projection refinement certifies");
        assert_eq!(cert.derived_map.len(), 1, "sum is derived");
        assert_eq!(cert.derived_map[0].0, "sum");
        assert_eq!(
            cert.var_map,
            vec![("k".to_string(), "c".to_string())],
            "k projects c"
        );
        assert!(verify_refinement_cert(&cert));
    }

    /// BYTE-COMPAT: a PROJECTION-only certificate serializes IDENTICALLY to the v1 schema — the
    /// `derived_map` field is OMITTED (skip-if-empty), the top-level key set is exactly the v1
    /// set, and an old-format JSON (no `derived_map`) still parses and re-verifies. So the digest
    /// (sha256 over the same canonical bytes) is unchanged for every pre-existing projection cert.
    #[test]
    fn projection_cert_digest_byte_identical() {
        let impl_spec = "---- MODULE Impl ----\n\
                         EXTENDS Integers\n\
                         VARIABLE x\n\
                         Init == x = 0\n\
                         Next == x' = x + 1 /\\ x < 3\n\
                         ====\n";
        // Both the default-projection and the explicit same-name rename must be v1-shaped.
        for map in [Vec::new(), vec![("x".to_string(), "x".to_string())]] {
            let cert = certify_refinement(
                impl_spec,
                &cfg("Init", "Next"),
                ABS_COUNTER,
                &cfg("AInit", "ANext"),
                &map,
            )
            .expect("projection cert certifies");
            assert!(cert.derived_map.is_empty());
            let json = cert.to_json();
            assert!(
                !json.contains("derived_map"),
                "a projection cert must OMIT the derived_map field (v1 byte-compat)"
            );
            // Top-level key set is EXACTLY the v1 field set (no new key leaked in).
            let value: serde_json::Value = serde_json::from_str(&json).unwrap();
            let mut keys: Vec<&str> = value
                .as_object()
                .unwrap()
                .keys()
                .map(String::as_str)
                .collect();
            keys.sort_unstable();
            let mut expected = vec![
                "schema",
                "impl_spec_src",
                "abs_spec_src",
                "impl_init",
                "impl_next",
                "abs_init",
                "abs_next",
                "var_map",
                "impl_reachable",
                "impl_inits",
                "transitions",
                "impl_sorts",
                "abs_init_ir",
                "abs_next_ir",
                "init_leg",
                "step_leg",
                "digest",
            ];
            expected.sort_unstable();
            assert_eq!(
                keys, expected,
                "projection cert has exactly the v1 top-level keys"
            );
            // An old-format JSON (which never had a derived_map field) round-trips and re-verifies.
            let reparsed = RefinementCert::from_json(&json).expect("v1 JSON parses");
            assert!(
                reparsed.derived_map.is_empty(),
                "serde default fills an empty derived_map"
            );
            assert!(
                verify_refinement_cert(&reparsed),
                "a v1 projection cert still re-verifies"
            );
        }
    }

    /// The configured CONSTANT is USED IN THE ABSTRACT recognized predicate (`x < N`), so it must be
    /// INLINED by the shared `CertInlineEnv` (the same front-end the explicit-state SAFETY cert lane
    /// uses) when `recognize_abstract` builds the abstract `Next` IR — not merely carried for the
    /// implementation enumeration. This is the abstract-side companion to
    /// `configured_constants_carried_and_reverify` / `interdependent_constants_order_carried`, which
    /// bind a constant only in the IMPL enumeration (the evaluator's `bind_constants_from_config`);
    /// here the constant reaches the abstract-predicate RECOGNIZER instead.
    ///
    /// The refinement VERDICT must track the constant's VALUE (ground-truth agreement, never a wrong
    /// claim): with `N = 10` a tight `x<3` impl refines (`3 ≤ 10`), but a wide `x<20` impl does NOT —
    /// it reaches `x = 19 ≥ 10`, where the inlined `x < 10` is FALSE and the kernel refutes the step
    /// leg (the VIOLATED twin). Raising the SAME configured constant to `N = 25` makes the wide impl
    /// refine again (`all x < 25`). If `N` were ignored, the abstract would read as `x < N` (a free
    /// ident) and DECLINE regardless — so the `N = 25` case CERTIFYING is the load-bearing evidence
    /// that the constant is faithfully inlined, not dropped.
    #[test]
    fn abstract_predicate_constant_is_inlined_and_verdict_tracks_value() {
        use crate::config::ConstantValue;
        let abs_capped = "---- MODULE AbsCapped ----\n\
                          EXTENDS Integers\n\
                          CONSTANT N\n\
                          VARIABLE x\n\
                          AInit == x = 0\n\
                          ANext == x' = x + 1 /\\ x < N\n\
                          ====\n";
        let abs_cfg = |n: &str| {
            let mut c = cfg("AInit", "ANext");
            c.constants
                .insert("N".to_string(), ConstantValue::Value(n.to_string()));
            c.constants_order = vec!["N".to_string()];
            c
        };
        let impl_tight = "---- MODULE ImplTight ----\n\
                          EXTENDS Integers\n\
                          VARIABLE x\n\
                          Init == x = 0\n\
                          Next == x' = x + 1 /\\ x < 3\n\
                          ====\n";
        let impl_wide = "---- MODULE ImplWide ----\n\
                         EXTENDS Integers\n\
                         VARIABLE x\n\
                         Init == x = 0\n\
                         Next == x' = x + 1 /\\ x < 20\n\
                         ====\n";

        // (1) N = 10, tight impl (x<3): every step satisfies the inlined `x < 10` ⇒ certifies, and
        // the abstract constant N is carried (sorted, digest-bound) and re-binds at verify.
        let cert = certify_refinement(
            impl_tight,
            &cfg("Init", "Next"),
            abs_capped,
            &abs_cfg("10"),
            &[],
        )
        .expect("a tight impl refines the N=10-capped abstract counter");
        assert_eq!(
            cert.abs_constants,
            vec![("N".to_string(), ConstantValue::Value("10".to_string()))],
            "the abstract-predicate constant N is carried in the certificate"
        );
        assert!(
            verify_refinement_cert(&cert),
            "the abstract-constant cert re-verifies"
        );

        // (2) N = 10, WIDE impl (x<20) — the VIOLATED twin: it reaches x=19 ≥ 10, where the inlined
        // `x < 10` is FALSE, so the kernel refutes the step leg ⇒ DECLINE (never a false refinement).
        assert!(
            certify_refinement(impl_wide, &cfg("Init", "Next"), abs_capped, &abs_cfg("10"), &[])
                .is_none(),
            "a wide impl (reaching x=19) does NOT refine the N=10 abstract — the constant bound is \
             inlined and the kernel refutes the violating step (ground-truth: not a refinement)"
        );

        // (3) Raising the SAME configured constant to N = 25 makes the wide impl refine again — the
        // verdict TRACKS the constant's value, proving it is faithfully inlined rather than ignored.
        let cert3 = certify_refinement(
            impl_wide,
            &cfg("Init", "Next"),
            abs_capped,
            &abs_cfg("25"),
            &[],
        )
        .expect("the wide impl refines the N=25-capped abstract counter");
        assert!(verify_refinement_cert(&cert3), "the N=25 cert re-verifies");
    }
}
