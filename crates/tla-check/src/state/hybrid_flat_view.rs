// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Hybrid per-action flat-view projection (ty-side M0 of wishlist item 4).
//!
//! # Why
//!
//! The two flagship compound-state targets (Disruptor, btree) each carry ONE
//! genuinely un-flattenable variable (Disruptor's unbounded `slots`/`consumed`,
//! btree's unprovable-capacity `toSplit`). Today one such variable vetoes native
//! per-action dispatch for the WHOLE spec
//! (`StateLayout::supports_flat_primary` is a whole-state conjunction), so every
//! other variable — all of which flatten fine — is stranded behind the veto and
//! every per-var flattening improvement yields zero win on the flagship specs.
//!
//! Item 4 decouples this: dispatch the actions that CAN run natively on a
//! **hybrid** state — a partial flat `[i64]` view over just the flat-admissible
//! variable subset — while the un-flattenable variables stay compound.
//!
//! # What this module provides (the foundation)
//!
//! [`HybridFlatView`] is a projection built once per run from the inferred
//! [`StateLayout`]:
//!
//! - **flat-admissible subset**: the variables whose *individual* kind passes
//!   [`VarLayoutKind::supports_flat_primary`] (scalars, proven `FixedScalar`,
//!   `IntArray`, proven records/keyed functions, capacity-proven recursive
//!   sequences). The rest (`Dynamic`, `Bitmask`, un-proven string/model-value
//!   scalars, and — after demotion — the un-flattenable flagship vars) stay
//!   compound.
//!
//! - [`HybridFlatView::project`]: encode an `ArrayState`'s admissible variables
//!   into a partial flat buffer (a [`FlatState`] over a *hybrid layout* whose
//!   non-admissible variables are demoted to [`VarLayoutKind::Dynamic`], so they
//!   occupy a single inert placeholder slot and are never read back from the
//!   buffer).
//!
//! - [`HybridFlatView::reconstruct`]: rebuild a successor `ArrayState` from a
//!   compound **parent** plus an updated flat view — admissible variables are
//!   decoded from the view; non-admissible variables are taken from the parent
//!   (`Arc`-shared, no copy of the compound payload). This is exactly the
//!   inverse of `project` for the covered variables.
//!
//! # Soundness
//!
//! The canonical dedup/fingerprint representation in this checker is the
//! `ArrayState`. The hybrid state's canonical form IS the reconstructed
//! `ArrayState`, so **fingerprint parity is automatic**: a state reached through
//! the hybrid projection fingerprints identically to the same state built
//! directly, exactly when `reconstruct(project(s)) == s` for the admissible
//! variables. That round-trip is guaranteed for every admissible kind by
//! `supports_flat_primary` (the same contract that backs the whole-state flat
//! path), and is exhaustively asserted in this module's tests (G2).
//!
//! The reconstruction leaves the non-admissible variables byte-identical to the
//! parent, so a hybrid-eligible action (whose entire read/write footprint is
//! flat-admissible — enforced by the caller) reproduces the interpreter
//! successor exactly. Any residual doubt is caught by the caller's fail-closed
//! differential against the interpreter successor.

use std::sync::Arc;
#[cfg(test)]
use tla_value::Rp;

use super::flat_state::FlatState;
use super::state_layout::{
    FlatValueLayout, SequenceBoundEvidence, SetBitmaskUniverseClosure, StateLayout, VarLayoutKind,
};
use super::ArrayState;
use crate::var_index::VarRegistry;

/// Compact, human-readable summary of a [`SetBitmaskUniverseClosure`] for the
/// eligibility-debug layout dump.
fn describe_closure(closure: &SetBitmaskUniverseClosure) -> &'static str {
    match closure {
        SetBitmaskUniverseClosure::Sampled => "Sampled",
        SetBitmaskUniverseClosure::ProvenClosed { .. } => "ProvenClosed",
        SetBitmaskUniverseClosure::DynamicallyDiscovered { .. } => "DynamicallyDiscovered",
    }
}

/// Compact recursive summary of a [`FlatValueLayout`] for the eligibility-debug
/// layout dump: shape + the admission-relevant provenance (universe closure,
/// sequence bound evidence), never full universes (which can be huge).
fn describe_value_layout(layout: &FlatValueLayout) -> String {
    match layout {
        FlatValueLayout::Scalar(ty) => format!("Scalar({ty:?})"),
        FlatValueLayout::IntFunction {
            lo,
            len,
            value_layout,
        } => format!(
            "IntFunction{{lo={lo},len={len},value={}}}",
            describe_value_layout(value_layout)
        ),
        FlatValueLayout::Function {
            domain,
            value_layout,
        } => format!(
            "Function{{|dom|={},value={}}}",
            domain.len(),
            describe_value_layout(value_layout)
        ),
        FlatValueLayout::Record { field_layouts, .. } => format!(
            "Record{{{}}}",
            field_layouts
                .iter()
                .map(describe_value_layout)
                .collect::<Vec<_>>()
                .join(",")
        ),
        FlatValueLayout::SetBitmask {
            universe,
            universe_closure,
        } => format!(
            "SetBitmask{{|u|={},closure={}}}",
            universe.len(),
            describe_closure(universe_closure)
        ),
        FlatValueLayout::RecordSetBitmask {
            universe,
            universe_closure,
        } => format!(
            "RecordSetBitmask{{|u|={},closure={},representable={},gate={}}}",
            universe.len(),
            describe_closure(universe_closure),
            super::state_layout::record_set_bitmask_universe_native_representable(universe),
            super::state_layout::record_set_native_flat_primary_enabled()
        ),
        FlatValueLayout::NestedSetBitmask {
            outer_universe,
            inner_universe,
            outer_closure,
            inner_closure,
        } => format!(
            "NestedSetBitmask{{|outer|={},|inner|={},outer={},inner={}}}",
            outer_universe.len(),
            inner_universe.len(),
            describe_closure(outer_closure),
            describe_closure(inner_closure)
        ),
        FlatValueLayout::TaggedScalarUnion { proof } => {
            format!("TaggedScalarUnion{{|u|={}}}", proof.universe().len())
        }
        FlatValueLayout::TaggedUnion { proof } => format!(
            "TaggedUnion{{{}}}",
            proof
                .variants()
                .iter()
                .map(describe_value_layout)
                .collect::<Vec<_>>()
                .join(",")
        ),
        FlatValueLayout::HeterogeneousTuple { element_layouts } => format!(
            "Tuple{{{}}}",
            element_layouts
                .iter()
                .map(describe_value_layout)
                .collect::<Vec<_>>()
                .join(",")
        ),
        FlatValueLayout::Sequence {
            bound,
            max_len,
            element_layout,
        } => {
            let bound_tag = match bound {
                SequenceBoundEvidence::Observed => "Observed",
                SequenceBoundEvidence::FixedDomainTypeLayout { .. } => "FixedDomainTypeLayout",
                SequenceBoundEvidence::ProvenInvariant { .. } => "ProvenInvariant",
                SequenceBoundEvidence::ProvenInvariantWithElementLayout { .. } => {
                    "ProvenInvariantWithElementLayout"
                }
                SequenceBoundEvidence::HeuristicUniverseCapacity { .. } => {
                    "HeuristicUniverseCapacity"
                }
            };
            format!(
                "Sequence{{max_len={max_len},bound={bound_tag},elem={}}}",
                describe_value_layout(element_layout)
            )
        }
    }
}

/// Compact summary of a [`VarLayoutKind`] for the eligibility-debug layout
/// dump (WP-15 diagnosis surface: which kind, and which admission-relevant
/// evidence it carries).
fn describe_kind(kind: &VarLayoutKind) -> String {
    match kind {
        VarLayoutKind::Scalar => "Scalar".to_string(),
        VarLayoutKind::ScalarBool => "ScalarBool".to_string(),
        VarLayoutKind::ScalarString => "ScalarString".to_string(),
        VarLayoutKind::ScalarModelValue => "ScalarModelValue".to_string(),
        VarLayoutKind::FixedScalar { base, .. } => format!(
            "FixedScalar{{base={base:?},proof_valid={}}}",
            kind.fixed_scalar_var_proof().is_some()
        ),
        VarLayoutKind::IntArray {
            len,
            elements_are_bool,
            element_types,
            element_range_proof,
            ..
        } => format!(
            "IntArray{{len={len},bool={elements_are_bool},typed={},range_proof={}}}",
            element_types.is_some(),
            element_range_proof.is_some()
        ),
        VarLayoutKind::Record {
            field_types,
            field_range_proofs,
            ..
        } => format!(
            "Record{{types={field_types:?},range_proofs={}}}",
            field_range_proofs.is_some()
        ),
        VarLayoutKind::StringKeyedArray {
            domain_keys,
            value_types,
            range_encoding,
            ..
        } => format!(
            "StringKeyedArray{{|dom|={},value_types={value_types:?},encoding={}}}",
            domain_keys.len(),
            match range_encoding {
                super::state_layout::StringKeyedArrayRangeEncoding::ScalarSlots => "ScalarSlots",
                super::state_layout::StringKeyedArrayRangeEncoding::FixedScalar(_) => "FixedScalar",
                super::state_layout::StringKeyedArrayRangeEncoding::TaggedScalarOrSet(_) =>
                    "TaggedScalarOrSet",
            }
        ),
        VarLayoutKind::TupleKeyedArray {
            domain_keys,
            value_types,
            range_encoding,
        } => format!(
            "TupleKeyedArray{{|dom|={},value_types={value_types:?},encoding={}}}",
            domain_keys.len(),
            match range_encoding {
                super::state_layout::TupleKeyedArrayRangeEncoding::ScalarSlots => "ScalarSlots",
                super::state_layout::TupleKeyedArrayRangeEncoding::FixedScalar(_) => "FixedScalar",
                super::state_layout::TupleKeyedArrayRangeEncoding::TaggedScalarUnion(_) =>
                    "TaggedScalarUnion",
            }
        ),
        VarLayoutKind::Recursive { layout } => {
            format!("Recursive{{{}}}", describe_value_layout(layout))
        }
        VarLayoutKind::Bitmask { universe_size } => format!("Bitmask{{|u|={universe_size}}}"),
        VarLayoutKind::Dynamic => "Dynamic".to_string(),
    }
}

/// Precomputed hybrid flat-view projection for one model-checking run.
///
/// Built once from the run's inferred [`StateLayout`]; cheap to hold (an `Arc`
/// to the hybrid layout plus a per-variable admissibility mask). Immutable and
/// shareable.
#[derive(Debug, Clone)]
pub(crate) struct HybridFlatView {
    /// Number of state variables (== `layout.var_count()`).
    var_count: usize,
    /// Per-variable flat-admissibility, indexed by `VarIndex`. `true` when the
    /// variable's own layout kind passes `supports_flat_primary`.
    admissible_mask: Vec<bool>,
    /// The flat-admissible variable indices (a compact, sorted view of the
    /// `true` entries in `admissible_mask`).
    admissible_indices: Vec<usize>,
    /// Hybrid layout: the full layout with every non-admissible variable demoted
    /// to [`VarLayoutKind::Dynamic`] (one inert placeholder slot). Admissible
    /// variables keep their original kind and are encoded/decoded exactly as on
    /// the whole-state flat path.
    hybrid_layout: Arc<StateLayout>,
}

impl HybridFlatView {
    /// Build a hybrid flat view from the run's state layout.
    ///
    /// Returns `None` when NO variable is flat-admissible — there is nothing to
    /// project, so the caller stays entirely on the interpreter. Otherwise the
    /// view covers the admissible subset and leaves the rest compound.
    #[must_use]
    pub(crate) fn from_layout(layout: &StateLayout, registry: &VarRegistry) -> Option<Self> {
        let var_count = layout.var_count();
        if var_count != registry.len() {
            // Registry / layout disagreement: fail closed, no hybrid view.
            return None;
        }

        let mut admissible_mask = Vec::with_capacity(var_count);
        let mut admissible_indices = Vec::new();
        let mut hybrid_kinds: Vec<VarLayoutKind> = Vec::with_capacity(var_count);

        let layout_debug =
            std::env::var_os("TY_HYBRID_ELIGIBILITY_DEBUG").is_some_and(|v| v == "1");
        for (var_idx, var) in layout.iter().enumerate() {
            let admissible = var.kind.supports_flat_primary();
            // WP-15 diagnosis surface: with the per-action reason dump on, also
            // say WHAT each variable's inferred kind is, so a blocking write is
            // attributable to a concrete layout shape + missing proof rather
            // than a bare index. Debug-only (same env gate as the per-action
            // dump); the default surface is unchanged.
            if layout_debug {
                eprintln!(
                    "[hybrid-layout] var={var_idx} name={} admissible={admissible} kind={}",
                    var.name,
                    describe_kind(&var.kind),
                );
            }
            admissible_mask.push(admissible);
            if admissible {
                admissible_indices.push(var_idx);
                hybrid_kinds.push(var.kind.clone());
            } else {
                // Non-admissible variables live only in the compound parent; a
                // single inert Dynamic placeholder slot keeps the flat buffer
                // dense without ever being decoded from.
                hybrid_kinds.push(VarLayoutKind::Dynamic);
            }
        }

        if admissible_indices.is_empty() {
            return None;
        }

        let hybrid_layout = Arc::new(StateLayout::new(registry, hybrid_kinds));

        Some(Self {
            var_count,
            admissible_mask,
            admissible_indices,
            hybrid_layout,
        })
    }

    /// Number of flat-admissible variables (the "native share" denominator).
    #[must_use]
    pub(crate) fn flat_admissible_count(&self) -> usize {
        self.admissible_indices.len()
    }

    /// The hybrid [`StateLayout`] backing this view: admissible variables keep
    /// their original kind, non-admissible variables are demoted to
    /// [`VarLayoutKind::Dynamic`] placeholders. This is the layout the hybrid
    /// native compilation path converts (via `try_check_layout_to_jit_layout`) so
    /// compiled slot offsets match [`Self::project`]'s buffer exactly (item 4
    /// M0-G1).
    #[must_use]
    pub(crate) fn hybrid_layout(&self) -> &Arc<StateLayout> {
        &self.hybrid_layout
    }

    /// Total number of state variables.
    #[must_use]
    pub(crate) fn var_count(&self) -> usize {
        self.var_count
    }

    /// Whether variable `var_idx` is in the flat-admissible subset.
    #[must_use]
    pub(crate) fn is_var_flat_admissible(&self, var_idx: usize) -> bool {
        self.admissible_mask.get(var_idx).copied().unwrap_or(false)
    }

    /// Whether every variable in `vars` is flat-admissible.
    ///
    /// This is the per-action coverage predicate: an action's read/write
    /// footprint (variable indices) is hybrid-eligible iff it is entirely inside
    /// the flat-admissible subset. Fail-closed: an out-of-range index (which can
    /// never be admissible) makes the footprint ineligible.
    #[must_use]
    pub(crate) fn footprint_all_admissible(&self, vars: impl IntoIterator<Item = usize>) -> bool {
        vars.into_iter().all(|v| self.is_var_flat_admissible(v))
    }

    /// Project an `ArrayState`'s flat-admissible variables into a partial flat
    /// view.
    ///
    /// The result is a [`FlatState`] over the hybrid layout: admissible
    /// variables hold their real encoded slots; non-admissible variables hold an
    /// inert placeholder that `reconstruct` never reads. Returns `None`
    /// (fail-closed) when the fixed layout cannot encode a concrete admissible
    /// value (e.g. a sequence exceeding its proven capacity, a scalar exceeding
    /// `i64`); the caller then falls back to the interpreter.
    #[must_use]
    pub(crate) fn project(&self, state: &ArrayState) -> Option<FlatState> {
        FlatState::try_from_array_state(state, Arc::clone(&self.hybrid_layout)).ok()
    }

    /// Reconstruct a successor `ArrayState` from a compound parent plus an
    /// updated flat view.
    ///
    /// Admissible variables are decoded from `view`; non-admissible variables
    /// are taken from `parent` (the compound payload is `Arc`-shared, not
    /// copied). Returns `None` (fail-closed) when the view cannot be decoded.
    ///
    /// SOUNDNESS: for a hybrid-eligible action the caller guarantees the action
    /// does not write any non-admissible variable, so `parent`'s value for those
    /// variables IS the successor's value. Combined with lossless round-trip of
    /// the admissible variables, the result equals the interpreter successor.
    #[must_use]
    pub(crate) fn reconstruct(
        &self,
        parent: &ArrayState,
        view: &FlatState,
        registry: &VarRegistry,
    ) -> Option<ArrayState> {
        view.try_to_array_state_with_fallback(registry, parent).ok()
    }

    /// Convenience round-trip used by the M0 interpreter-through-projection stub
    /// and the parity tests: project `successor`'s admissible variables, then
    /// reconstruct with `parent` supplying the compound variables.
    ///
    /// When `parent` and `successor` agree on every non-admissible variable (the
    /// hybrid-eligibility invariant) and every admissible kind round-trips
    /// losslessly, the result equals `successor`.
    #[must_use]
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn project_then_reconstruct(
        &self,
        parent: &ArrayState,
        successor: &ArrayState,
        registry: &VarRegistry,
    ) -> Option<ArrayState> {
        let view = self.project(successor)?;
        self.reconstruct(parent, &view, registry)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::layout_inference::infer_layout;
    use crate::var_index::VarIndex;
    use crate::Value;
    use tla_value::value::{IntIntervalFunc, SortedSet};

    fn reg(names: &[&str]) -> VarRegistry {
        VarRegistry::from_names(names.iter().copied())
    }

    /// Assert two states are value-equal on every variable and fingerprint
    /// identically — the exact soundness bar (G2).
    fn assert_state_parity(a: &ArrayState, b: &ArrayState, registry: &VarRegistry) {
        let n = registry.len();
        for i in 0..n {
            let idx = VarIndex::new(i);
            assert_eq!(
                a.get(idx),
                b.get(idx),
                "value mismatch at var {i} ({})",
                registry.name(idx)
            );
        }
        let mut a_fp = a.clone();
        let mut b_fp = b.clone();
        assert_eq!(
            a_fp.fingerprint(registry),
            b_fp.fingerprint(registry),
            "fingerprint mismatch between hybrid-reconstructed and compound state"
        );
    }

    fn small_set(vals: Vec<Value>) -> Value {
        Value::Set(Rp::new(SortedSet::from_sorted_vec(vals)))
    }

    #[test]
    fn scalar_only_roundtrip_is_lossless_and_fp_exact() {
        let registry = reg(&["x", "y", "z"]);
        let state = ArrayState::from_values(vec![
            Value::SmallInt(42),
            Value::Bool(true),
            Value::SmallInt(-7),
        ]);
        let layout = infer_layout(&state, &registry);
        let view = HybridFlatView::from_layout(&layout, &registry)
            .expect("scalar layout must yield a hybrid view");

        // Every variable is flat-admissible for a pure-scalar spec.
        assert_eq!(view.flat_admissible_count(), 3);
        assert!(view.footprint_all_admissible([0usize, 1, 2]));

        let rebuilt = view
            .project_then_reconstruct(&state, &state, &registry)
            .expect("scalar round-trip must succeed");
        assert_state_parity(&rebuilt, &state, &registry);
    }

    #[test]
    fn hybrid_mixed_admissible_and_compound_roundtrip() {
        // var0: scalar (admissible), var1: a Set (compound / non-admissible),
        // var2: an int array (admissible).
        let registry = reg(&["pc", "data", "arr"]);
        let func = IntIntervalFunc::new(
            0,
            2,
            vec![
                Value::SmallInt(10),
                Value::SmallInt(20),
                Value::SmallInt(30),
            ],
        );
        let state = ArrayState::from_values(vec![
            Value::SmallInt(1),
            small_set(vec![Value::SmallInt(5), Value::SmallInt(9)]),
            Value::IntFunc(Rp::new(func)),
        ]);
        let layout = infer_layout(&state, &registry);
        let view = HybridFlatView::from_layout(&layout, &registry)
            .expect("mixed layout must yield a hybrid view");

        // The Set variable must be excluded from the flat-admissible subset.
        assert!(view.is_var_flat_admissible(0), "scalar var admissible");
        assert!(
            !view.is_var_flat_admissible(1),
            "compound Set var must NOT be flat-admissible"
        );
        assert!(view.is_var_flat_admissible(2), "int-array var admissible");
        assert_eq!(view.flat_admissible_count(), 2);

        // An action touching only {pc, arr} is hybrid-eligible; one touching
        // {data} is not.
        assert!(view.footprint_all_admissible([0usize, 2]));
        assert!(!view.footprint_all_admissible([1usize]));
        assert!(!view.footprint_all_admissible([0usize, 1, 2]));

        // Hybrid round-trip with parent == successor (the compound var is
        // supplied by the parent) reproduces the state exactly, including
        // fingerprint.
        let rebuilt = view
            .project_then_reconstruct(&state, &state, &registry)
            .expect("mixed round-trip must succeed");
        assert_state_parity(&rebuilt, &state, &registry);
    }

    #[test]
    fn compound_var_is_taken_from_parent_not_successor() {
        // Demonstrates the hybrid-eligibility invariant: reconstruct pulls the
        // compound (non-admissible) variable from the PARENT. If an action were
        // (wrongly) allowed to write a compound var, reconstruction would keep
        // the parent's value — which is precisely why the caller restricts
        // hybrid dispatch to actions whose footprint excludes compound vars, and
        // why the runtime differential fails closed on any divergence.
        let registry = reg(&["pc", "data"]);
        let parent = ArrayState::from_values(vec![
            Value::SmallInt(1),
            small_set(vec![Value::SmallInt(1)]),
        ]);
        // Successor "changed" the admissible scalar (pc: 1 -> 2) AND the compound
        // Set — but only pc is in the flat-admissible footprint.
        let successor = ArrayState::from_values(vec![
            Value::SmallInt(2),
            small_set(vec![Value::SmallInt(1), Value::SmallInt(2)]),
        ]);
        let layout = infer_layout(&parent, &registry);
        let view = HybridFlatView::from_layout(&layout, &registry).expect("view");

        let rebuilt = view
            .project_then_reconstruct(&parent, &successor, &registry)
            .expect("round-trip");

        // Admissible var reflects the successor.
        assert_eq!(rebuilt.get(VarIndex::new(0)), Value::SmallInt(2));
        // Compound var reflects the PARENT (the flat view carried no update for
        // it) — so it does NOT equal the successor's Set. A caller that admitted
        // this action would see the differential mismatch and fail closed.
        assert_eq!(rebuilt.get(VarIndex::new(1)), parent.get(VarIndex::new(1)));
        assert_ne!(
            rebuilt.get(VarIndex::new(1)),
            successor.get(VarIndex::new(1))
        );
    }

    #[test]
    fn hybrid_and_compound_states_dedup_identically() {
        // Build the SAME logical state two ways: (a) directly as a compound
        // ArrayState, (b) via the hybrid project/reconstruct path. Their
        // fingerprints must collide into a single dedup slot.
        let registry = reg(&["pc", "data", "arr"]);
        let func = IntIntervalFunc::new(0, 1, vec![Value::SmallInt(7), Value::SmallInt(8)]);
        let compound = ArrayState::from_values(vec![
            Value::SmallInt(3),
            small_set(vec![Value::SmallInt(4)]),
            Value::IntFunc(Rp::new(func)),
        ]);
        let layout = infer_layout(&compound, &registry);
        let view = HybridFlatView::from_layout(&layout, &registry).expect("view");

        let hybrid = view
            .project_then_reconstruct(&compound, &compound, &registry)
            .expect("round-trip");

        let mut seen: std::collections::HashSet<crate::state::Fingerprint> =
            std::collections::HashSet::new();
        let mut c = compound.clone();
        let mut h = hybrid.clone();
        seen.insert(c.fingerprint(&registry));
        seen.insert(h.fingerprint(&registry));
        assert_eq!(
            seen.len(),
            1,
            "hybrid-represented and compound-represented state must dedup to one fingerprint"
        );
    }

    #[test]
    fn record_kind_roundtrips_through_hybrid_view() {
        // A record with scalar fields is flat-admissible; verify it survives the
        // hybrid projection alongside a compound var.
        let registry = reg(&["rec", "data"]);
        let record = Value::record([("a", Value::SmallInt(11)), ("b", Value::Bool(false))]);
        let state = ArrayState::from_values(vec![record, small_set(vec![Value::SmallInt(1)])]);
        let layout = infer_layout(&state, &registry);
        let view = HybridFlatView::from_layout(&layout, &registry).expect("view");

        // If the record inferred as admissible, the round-trip must be exact.
        if view.is_var_flat_admissible(0) {
            let rebuilt = view
                .project_then_reconstruct(&state, &state, &registry)
                .expect("record round-trip");
            assert_state_parity(&rebuilt, &state, &registry);
        }
    }

    /// Generic env guard for admission-gate tests: set/unset a process env var
    /// while holding the process-wide env lock (`crate::process_env_lock`),
    /// restoring the previous value on drop. Mirrors the `ForceBatchEnvGuard`
    /// idiom in `parallel::tests::consistency`.
    struct EnvVarGuard {
        name: &'static str,
        previous: Option<std::ffi::OsString>,
        _lock: std::sync::MutexGuard<'static, ()>,
    }

    impl EnvVarGuard {
        fn set(name: &'static str, value: Option<&str>) -> Self {
            let lock = crate::process_env_lock();
            let previous = std::env::var_os(name);
            match value {
                Some(value) => crate::env_guard::set_var(name, value),
                None => crate::env_guard::remove_var(name),
            }
            Self {
                name,
                previous,
                _lock: lock,
            }
        }
    }

    impl Drop for EnvVarGuard {
        fn drop(&mut self) {
            match self.previous.take() {
                Some(value) => crate::env_guard::set_var(self.name, value),
                None => crate::env_guard::remove_var(self.name),
            }
        }
    }

    /// WP-15 (write-side flat admissibility, MCBakery `nxt` class): a
    /// fixed-domain sequence var whose type fact is proven by TWO checked
    /// invariants (label-only duplicate proofs) becomes flat-admissible under
    /// `TY_FLAT_WRITE_ADMIT=1`, and its hybrid projection round-trips
    /// losslessly with exact fingerprint parity. With the gate OFF the var
    /// stays non-admissible (historical fail-closed veto), pinned here so the
    /// default surface cannot drift.
    #[test]
    fn wp15_duplicate_proof_fixed_domain_sequence_roundtrips_under_gate() {
        use crate::state::layout_inference::{
            infer_layout_with_sequence_layout_tagged_set_type_and_range_proofs,
            SequenceFixedDomainTypeProof, SequenceTypeLayoutProof,
        };
        use crate::state::state_layout::{FlatValueLayout, SlotType};

        let registry = reg(&["nxt", "data"]);
        // nxt = <<1, 1>> (a [1..2 -> Procs] function stored as a sequence);
        // data = a compound Set that must stay on the parent.
        let state = ArrayState::from_values(vec![
            Value::Seq(Rp::new(tla_value::value::SeqValue::from_vec(vec![
                Value::SmallInt(1),
                Value::SmallInt(1),
            ]))),
            small_set(vec![Value::SmallInt(9)]),
        ]);
        let proof = |invariant: &str| SequenceFixedDomainTypeProof {
            var_idx: 0,
            path: Vec::new(),
            domain: Arc::from(vec![Value::SmallInt(1), Value::SmallInt(2)].into_boxed_slice()),
            element_layout: SequenceTypeLayoutProof::Flat(FlatValueLayout::Scalar(SlotType::Int)),
            invariant: Arc::from(invariant),
        };
        let proofs = [proof("TypeOK"), proof("Inv")];
        let infer = |state: &ArrayState| {
            infer_layout_with_sequence_layout_tagged_set_type_and_range_proofs(
                state,
                &registry,
                &[],
                &[],
                &proofs,
                &[],
                &[],
                &[],
                &[],
                &[],
                &[],
            )
        };

        {
            let _guard = EnvVarGuard::set("TY_FLAT_WRITE_ADMIT", None);
            let layout = infer(&state);
            assert!(
                !layout.var_layout(0).unwrap().kind.supports_flat_primary(),
                "gate OFF: the duplicate-proof var must stay non-admissible"
            );
        }

        let _guard = EnvVarGuard::set("TY_FLAT_WRITE_ADMIT", Some("1"));
        let layout = infer(&state);
        let view = HybridFlatView::from_layout(&layout, &registry).expect("view");
        assert!(
            view.is_var_flat_admissible(0),
            "gate ON: the duplicate-proof fixed-domain sequence var must be admissible"
        );
        assert!(
            !view.is_var_flat_admissible(1),
            "the compound Set var must stay non-admissible"
        );

        let rebuilt = view
            .project_then_reconstruct(&state, &state, &registry)
            .expect("round-trip");
        assert_state_parity(&rebuilt, &state, &registry);

        // A successor that changes the admissible sequence round-trips to the
        // successor's value (not the parent's) — the write side is live.
        let successor = ArrayState::from_values(vec![
            Value::Seq(Rp::new(tla_value::value::SeqValue::from_vec(vec![
                Value::SmallInt(2),
                Value::SmallInt(1),
            ]))),
            small_set(vec![Value::SmallInt(9)]),
        ]);
        let rebuilt = view
            .project_then_reconstruct(&state, &successor, &registry)
            .expect("successor round-trip");
        assert_state_parity(&rebuilt, &successor, &registry);
    }

    /// WP-15 (write-side flat admissibility, Paxos `msgs` class): a
    /// proven-closed, natively-representable record-set-bitmask var is
    /// flat-admissible under the existing `TY_RECORD_SET_NATIVE=1` opt-in, and
    /// its hybrid projection round-trips losslessly with exact fingerprint
    /// parity. Gate OFF keeps it non-admissible (byte-identical default),
    /// pinned in the same test.
    #[test]
    fn wp15_proven_record_set_bitmask_roundtrips_under_record_set_gate() {
        use crate::state::state_layout::{
            FlatValueLayout, SetBitmaskUniverseClosure, VarLayoutKind,
        };

        let rec = |ty: &str, ins: i64| {
            Value::Record(tla_value::value::RecordValue::from_sorted_str_entries(
                vec![
                    (Arc::from("ins"), Value::SmallInt(ins)),
                    (Arc::from("type"), Value::String(Rp::from(ty))),
                ],
            ))
        };
        let mut universe = vec![rec("phase1a", 1), rec("phase1a", 2), rec("phase1b", 1)];
        universe.sort();
        universe.dedup();

        let registry = reg(&["msgs", "pc"]);
        let kinds = vec![
            VarLayoutKind::Recursive {
                layout: FlatValueLayout::RecordSetBitmask {
                    universe: universe.clone(),
                    universe_closure: SetBitmaskUniverseClosure::ProvenClosed {
                        invariant: Arc::from("TypeOK"),
                    },
                },
            },
            VarLayoutKind::Scalar,
        ];
        let layout = StateLayout::new(&registry, kinds);

        // A non-empty strict subset of the universe plus a scalar.
        let state = ArrayState::from_values(vec![
            small_set(vec![rec("phase1a", 1), rec("phase1b", 1)]),
            Value::SmallInt(3),
        ]);

        {
            let _guard = EnvVarGuard::set("TY_RECORD_SET_NATIVE", None);
            let view = HybridFlatView::from_layout(&layout, &registry).expect("view");
            assert!(
                !view.is_var_flat_admissible(0),
                "gate OFF: even a proven record-set-bitmask var stays non-admissible \
                 (byte-identical default)"
            );
        }

        let _guard = EnvVarGuard::set("TY_RECORD_SET_NATIVE", Some("1"));
        let view = HybridFlatView::from_layout(&layout, &registry).expect("view");
        assert!(
            view.is_var_flat_admissible(0),
            "gate ON: the proven-closed record-set-bitmask var must be admissible"
        );

        let rebuilt = view
            .project_then_reconstruct(&state, &state, &registry)
            .expect("record-set round-trip");
        assert_state_parity(&rebuilt, &state, &registry);

        // The empty set and the full universe are both canonical mask values;
        // pin their round-trips too (the empty mask is the Paxos Init shape).
        for msgs in [small_set(vec![]), small_set(universe.clone())] {
            let state = ArrayState::from_values(vec![msgs, Value::SmallInt(3)]);
            let rebuilt = view
                .project_then_reconstruct(&state, &state, &registry)
                .expect("record-set round-trip");
            assert_state_parity(&rebuilt, &state, &registry);
        }
    }

    /// WP-09/Part A (btree `childOf` class): a tuple-keyed function var whose
    /// range is a proven MIXED Int ∪ model-value union becomes flat-admissible
    /// under `TY_TAGGED_SCALAR_UNION=1` via the universe-index range encoding,
    /// round-trips losslessly through the hybrid projection (parent AND
    /// changed-successor), and its hybrid-projected representation dedups to
    /// the identical fingerprint as the compound representation. Gate OFF
    /// keeps the var non-admissible (historical fail-closed default), pinned
    /// in the same test.
    #[test]
    fn wp09_tuple_keyed_union_range_roundtrips_with_fingerprint_parity_under_gate() {
        use crate::state::state_layout::{
            FlatScalarValue, SlotType, TaggedScalarUnionProof, TupleKeyedArrayRangeEncoding,
            VarLayoutKind,
        };
        use tla_value::value::FuncValue;

        let registry = reg(&["childOf", "data"]);
        let key = |n: i64, k: i64| Value::tuple(vec![Value::SmallInt(n), Value::SmallInt(k)]);
        let domain_keys = vec![key(1, 1), key(1, 2), key(2, 1), key(2, 2)];
        let proof = TaggedScalarUnionProof::new(
            vec![
                FlatScalarValue::Int(1),
                FlatScalarValue::Int(2),
                FlatScalarValue::ModelValue(Arc::from("nil")),
            ],
            Arc::from("TypeOk"),
        )
        .unwrap();
        let kinds = vec![
            VarLayoutKind::TupleKeyedArray {
                domain_keys: domain_keys.clone(),
                value_types: vec![SlotType::ModelValue; 4],
                range_encoding: TupleKeyedArrayRangeEncoding::TaggedScalarUnion(proof),
            },
            VarLayoutKind::Scalar,
        ];
        let layout = StateLayout::new(&registry, kinds);

        let nil = Value::ModelValue(Rp::from("nil"));
        let func = |values: [Value; 4]| {
            Value::Func(Rp::new(FuncValue::from_sorted_entries(
                domain_keys
                    .iter()
                    .cloned()
                    .zip(values.into_iter())
                    .collect(),
            )))
        };
        let state = ArrayState::from_values(vec![
            func([Value::SmallInt(1), nil.clone(), nil.clone(), nil.clone()]),
            Value::SmallInt(7),
        ]);

        {
            let _guard = EnvVarGuard::set("TY_TAGGED_SCALAR_UNION", None);
            let view = HybridFlatView::from_layout(&layout, &registry).expect("view");
            assert!(
                !view.is_var_flat_admissible(0),
                "gate OFF: the union-range tuple-keyed var must stay non-admissible \
                 (byte-identical default)"
            );
        }

        let _guard = EnvVarGuard::set("TY_TAGGED_SCALAR_UNION", Some("1"));
        let view = HybridFlatView::from_layout(&layout, &registry).expect("view");
        assert!(
            view.is_var_flat_admissible(0),
            "gate ON: the proven union-range tuple-keyed var must be admissible"
        );

        let rebuilt = view
            .project_then_reconstruct(&state, &state, &registry)
            .expect("round-trip");
        assert_state_parity(&rebuilt, &state, &registry);

        // A successor that flips one slot between the Int arm and the
        // model-value arm round-trips to the successor's value — the write
        // side is live across the sorts.
        let successor = ArrayState::from_values(vec![
            func([nil.clone(), nil.clone(), Value::SmallInt(2), nil]),
            Value::SmallInt(7),
        ]);
        let rebuilt = view
            .project_then_reconstruct(&state, &successor, &registry)
            .expect("successor round-trip");
        assert_state_parity(&rebuilt, &successor, &registry);

        // FINGERPRINT PARITY: the hybrid-projected representation and the
        // directly-built compound representation of the SAME logical state
        // must dedup into one fingerprint slot.
        let hybrid = view
            .project_then_reconstruct(&state, &state, &registry)
            .expect("round-trip");
        let mut seen: std::collections::HashSet<crate::state::Fingerprint> =
            std::collections::HashSet::new();
        let mut c = state.clone();
        let mut h = hybrid.clone();
        seen.insert(c.fingerprint(&registry));
        seen.insert(h.fingerprint(&registry));
        assert_eq!(
            seen.len(),
            1,
            "hybrid-represented and compound-represented union-range state must \
             dedup to one fingerprint"
        );
    }

    /// WP-33 shared fixture: a model-value-keyed function `ctr` whose range is
    /// the proven-finite Int universe `0..3` (checked `TypeOk` clause
    /// `ctr \in [Procs -> 0..3]`), plus a compound `Set` var that must stay on
    /// the parent. Returns `(registry, infer)`.
    fn wp33_int_range_fixture() -> (
        VarRegistry,
        impl Fn(&ArrayState) -> crate::state::state_layout::StateLayout,
    ) {
        use crate::state::layout_inference::{
            infer_layout_with_sequence_layout_tagged_set_type_and_range_proofs,
            FixedScalarRangeTypeProof,
        };
        use crate::state::state_layout::{FlatScalarValue, SlotType};

        let registry = reg(&["ctr", "data"]);
        let proof = FixedScalarRangeTypeProof {
            var_idx: 0,
            path: Vec::new(),
            domain: Arc::from(
                vec![
                    Value::ModelValue(Rp::from("p1")),
                    Value::ModelValue(Rp::from("p2")),
                ]
                .into_boxed_slice(),
            ),
            scalar_type: SlotType::Int,
            scalar_universe: vec![
                FlatScalarValue::Int(0),
                FlatScalarValue::Int(1),
                FlatScalarValue::Int(2),
                FlatScalarValue::Int(3),
            ],
            invariant: Arc::from("TypeOk"),
        };
        let registry_for_infer = reg(&["ctr", "data"]);
        let infer = move |state: &ArrayState| {
            infer_layout_with_sequence_layout_tagged_set_type_and_range_proofs(
                state,
                &registry_for_infer,
                &[],
                &[],
                &[],
                &[],
                std::slice::from_ref(&proof),
                &[],
                &[],
                &[],
                &[],
            )
        };
        (registry, infer)
    }

    /// Build the `ctr` state `[p1 |-> a, p2 |-> b]` with model-value keys.
    fn wp33_ctr_state(a: Value, b: Value) -> ArrayState {
        use tla_value::value::FuncValue;
        ArrayState::from_values(vec![
            Value::Func(Rp::new(FuncValue::from_sorted_entries(vec![
                (Value::ModelValue(Rp::from("p1")), a),
                (Value::ModelValue(Rp::from("p2")), b),
            ]))),
            small_set(vec![Value::SmallInt(9)]),
        ])
    }

    /// WP-33 (Disruptor item-1 blocker 2, the general admission gap WP-30 found):
    /// an INT-valued model-value-keyed array whose range is a PROVEN FINITE
    /// universe (`ctr \in [Procs -> 0..3]`) becomes flat-primary-admissible under
    /// `TY_FLAT_INT_RANGE_ADMIT=1`, round-trips losslessly through the hybrid
    /// projection, and fingerprints identically to the compound state. With the
    /// gate OFF it stays non-admissible (historical fail-closed veto), pinned
    /// here so the default surface cannot drift.
    #[test]
    fn wp33_finite_int_range_keyed_array_roundtrips_under_gate() {
        let (registry, infer) = wp33_int_range_fixture();
        let state = wp33_ctr_state(Value::SmallInt(0), Value::SmallInt(3));

        {
            let _guard = EnvVarGuard::set("TY_FLAT_INT_RANGE_ADMIT", None);
            let layout = infer(&state);
            assert!(
                !layout.var_layout(0).unwrap().kind.supports_flat_primary(),
                "gate OFF: a proven-finite Int range must stay non-admissible"
            );
        }

        let _guard = EnvVarGuard::set("TY_FLAT_INT_RANGE_ADMIT", Some("1"));
        let layout = infer(&state);
        let view = HybridFlatView::from_layout(&layout, &registry).expect("view");
        assert!(
            view.is_var_flat_admissible(0),
            "gate ON: the proven-finite Int-ranged keyed array must be admissible"
        );
        assert!(
            !view.is_var_flat_admissible(1),
            "the compound Set var must stay non-admissible"
        );
        assert_eq!(view.flat_admissible_count(), 1);

        // Lossless round-trip + fingerprint parity, over EVERY point of the
        // proven universe (including the boundary values 0 and 3).
        for a in 0..=3i64 {
            for b in 0..=3i64 {
                let s = wp33_ctr_state(Value::SmallInt(a), Value::SmallInt(b));
                let rebuilt = view
                    .project_then_reconstruct(&s, &s, &registry)
                    .unwrap_or_else(|| panic!("round-trip must succeed for ctr = [{a}, {b}]"));
                assert_state_parity(&rebuilt, &s, &registry);
            }
        }

        // Distinct universe points must NOT collide in the flat fingerprint —
        // the exact undercount hazard the general `StringKeyedArray` fail-closed
        // comment guards against.
        let mut seen: std::collections::HashSet<crate::state::Fingerprint> =
            std::collections::HashSet::new();
        for a in 0..=3i64 {
            for b in 0..=3i64 {
                let mut s = wp33_ctr_state(Value::SmallInt(a), Value::SmallInt(b));
                let mut hybrid = view
                    .project_then_reconstruct(&s, &s, &registry)
                    .expect("round-trip");
                let fp = s.fingerprint(&registry);
                assert_eq!(
                    fp,
                    hybrid.fingerprint(&registry),
                    "hybrid and compound representations of ctr = [{a}, {b}] must \
                     dedup to one fingerprint"
                );
                seen.insert(fp);
            }
        }
        assert_eq!(
            seen.len(),
            16,
            "all 16 distinct [Procs -> 0..3] states must have distinct fingerprints"
        );
    }

    /// WP-33 fail-closed: with the gate ON, a range value OUTSIDE the proven
    /// universe (an out-of-range Int, or a SET — the DijkstraMutex `temp` hazard
    /// the `StringKeyedArray` comment names) must not be encoded. Both the
    /// layout fit check and the projection decline, routing the state back to
    /// the interpreter rather than writing a slot the proof does not cover.
    #[test]
    fn wp33_finite_int_range_keyed_array_fails_closed_outside_universe() {
        let (registry, infer) = wp33_int_range_fixture();
        let admitted = wp33_ctr_state(Value::SmallInt(1), Value::SmallInt(2));

        let _guard = EnvVarGuard::set("TY_FLAT_INT_RANGE_ADMIT", Some("1"));
        let layout = infer(&admitted);
        let view = HybridFlatView::from_layout(&layout, &registry).expect("view");
        assert!(view.is_var_flat_admissible(0));

        let layout_arc = Arc::new(layout);
        for (label, bad) in [
            ("out-of-range Int", Value::SmallInt(7)),
            ("negative out-of-range Int", Value::SmallInt(-1)),
            (
                "finite set (DijkstraMutex `temp` hazard)",
                small_set(vec![Value::SmallInt(1)]),
            ),
            ("model value", Value::ModelValue(Rp::from("p1"))),
            ("string", Value::String(Rp::from("p1"))),
            ("bool", Value::Bool(true)),
        ] {
            let state = wp33_ctr_state(Value::SmallInt(1), bad);
            assert!(
                !FlatState::array_state_fits_layout(&state, &layout_arc),
                "{label}: must NOT fit the proven-universe layout"
            );
            assert!(
                view.project(&state).is_none(),
                "{label}: hybrid projection must fail closed"
            );
        }
    }

    #[test]
    fn from_layout_none_when_no_admissible_var() {
        // A single compound-only variable yields no flat-admissible subset.
        let registry = reg(&["data"]);
        let state = ArrayState::from_values(vec![small_set(vec![Value::SmallInt(1)])]);
        let layout = infer_layout(&state, &registry);
        // Either the Set var is non-admissible (expected) -> no view, or (if a
        // future inference admits it) the view exists; both are internally
        // consistent, but for a bare Set we expect None.
        if layout.iter().all(|v| !v.kind.supports_flat_primary()) {
            assert!(HybridFlatView::from_layout(&layout, &registry).is_none());
        }
    }
}
