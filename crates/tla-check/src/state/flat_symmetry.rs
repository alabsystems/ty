// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Flat-space symmetry canonicalization (wishlist item 9 — WP-11 slice 1).
//!
//! Declared-SYMMETRY specs are today vetoed from every native/flat path
//! because the fingerprint domain becomes `SymmetryCanonical`, an
//! interpreter-only min-over-permutations on the `Value` tree
//! (`state/symmetry.rs`). This module compiles the declared permutation group
//! into **slot-level actions on the flat i64 buffer** so that (in slice 2) the
//! canonical representative can be computed without materializing `Value`
//! trees at all.
//!
//! # What an action is
//!
//! For one permutation `p` of the declared group, [`FlatSymmetryAction`] is a
//! total function on encoded buffers such that for every state `s` that FITS
//! the layout (see the contract below):
//!
//! ```text
//! action_p(encode(s)) == encode(permute_value_tree(p, s))
//! ```
//!
//! It is represented as, per output slot `j`, a source slot `src[j]` plus a
//! [`SlotTransform`] applied to the source's raw i64:
//!
//! * slot-range permutation tables for model-value-KEYED containers (the
//!   window of key `k` moves to the window of `p(k)`),
//! * value remap tables for model-value payloads: interned `NameId -> NameId`
//!   remaps for raw model-value slots, universe-index remaps for
//!   `TaggedScalarUnion` slots, bit-position permutations for `SetBitmask`
//!   slots. `Int`/`Bool`/`String` payloads are identity (a TLA+ symmetry
//!   permutation only moves model values; see `Value::permute_impl`).
//!
//! # Soundness contract (READ THIS BEFORE WIRING — slice 2)
//!
//! 1. **Fits-contract.** Actions are proven equivariant only for buffers
//!    encoded from states that *fit* the layout
//!    (`FlatState::array_state_fits_layout` /
//!    `FlatState::try_from_array_state` success). The fit check is what makes
//!    slot typing unambiguous: `String` and `ModelValue` share one interned
//!    `NameId` space, so a model value sitting in a `String`-typed slot would
//!    be remap-invisible (and vice versa would be remap-corrupted). Fitting
//!    states cannot do that (`value_fits_slot_type` distinguishes the sorts).
//! 2. **Fail-closed admission.** [`FlatSymmetryCanonicalizer::compile`]
//!    returns `None` — declining the WHOLE layout — unless *every* variable's
//!    kind is provably equivariant-representable for *every* group element:
//!    - `Dynamic` and placeholder `Bitmask` vars: always decline.
//!    - Model-value-keyed containers whose domain is not closed under the
//!      group: decline.
//!    - Proof universes (`FixedScalar`, `TaggedScalarUnion`, `SetBitmask`
//!      masks, `TaggedScalarOrSet` set arms) not closed under the group:
//!      decline.
//!    - `SetBitmask` universes that are merely *sampled* (not
//!      `ProvenClosed`): decline unless the universe is provably fixed by the
//!      group (identity action).
//!    - Per-slot heterogeneously-typed ranges whose types differ across a key
//!      orbit: decline (the moved value could not legally re-encode at its
//!      destination slot).
//!    - Capacity-proven sequences whose element content the group can touch:
//!      decline (live-window remapping would be data-dependent — the trailing
//!      zero-padded windows must stay zero). A sequence whose content is
//!      provably fixed by the group is admitted as identity.
//!    - `TaggedUnion` / `RecordSetBitmask` / `NestedSetBitmask`: admitted only
//!      when provably fixed by the group (identity), declined otherwise.
//! 3. **Group closure.** `compute_symmetry_perms`
//!    (`model_checker/symmetry_perms.rs`) already enumerates the FULL closure
//!    (TLC's frontier algorithm) and `normalize_perm_group` strips identity
//!    entries. This module does NOT trust that: it re-encloses the input under
//!    composition with a hard cap ([`MAX_GROUP_ORDER`]) and declines beyond,
//!    so generator-only inputs are also correct and a non-closed input can
//!    never produce a non-invariant "canonical".
//! 4. **Streaming lexmin.** `canonical(x) = lexicographic min over g in G of
//!    g·x` computed per-slot with early abort — never materializing |G|
//!    buffers. Because the compiled set is a closed group, `canonical` is
//!    constant on orbits, and because every action is `encode ∘ permute ∘
//!    decode` on fitting states, two fitting states have equal canonicals IFF
//!    the interpreter's orbit relation identifies them (the representative
//!    itself may differ from the interpreter's — only the partition matters).
//!
//! # Fence
//!
//! The inert Geometric-Supremacy scaffold (`bfs/topology/` +
//! `HomotopicCanonicalizer`) sorts slot groups, which is NOT
//! lexmin-over-the-group and must never run on the same buffer path as this
//! machinery. See `FlatBufferCanonicalizationAuthority` in
//! `model_checker/fingerprint.rs` — slice 2 must route through it.
//!
//! Slice 1 (this change) deliberately wires NOTHING into production: no BFS
//! veto site, no `BfsFingerprintDomain` variant, no `run_helpers` change. The
//! module is exercised by its oracle tests only.

use tla_value::Rp;

use std::cmp::Ordering;
use std::collections::BTreeSet;
use std::sync::Arc;

use super::state_layout::{
    FlatScalarValue, FlatValueLayout, SlotType, StateLayout, StringKeyedArrayRangeEncoding,
    TupleKeyedArrayRangeEncoding, VarLayoutKind,
};
use crate::value::FuncValue;
use crate::Value;

/// Hard cap on the closed group order. TLC has no cap (group order is bounded
/// by n! for n model values), but a compiled per-element table set is
/// per-state-canonicalization O(|G| * slots) work — beyond this, flat
/// canonicalization stops being a win and the interpreter path should keep
/// the spec. 10_080 = 2 * 7! covers every corpus symmetry group in use.
pub(crate) const MAX_GROUP_ORDER: usize = 10_080;

/// Hard cap on `|G| * total_slots` — the total compiled table footprint.
pub(crate) const MAX_TOTAL_TABLE_SLOTS: usize = 4_000_000;

/// A symmetry permutation in model-value *name* space: a sorted, identity-free
/// map `name -> name` that is a bijection on its support.
///
/// This is deliberately independent of `FuncValue`/`MVPerm` so that group
/// composition and closure cannot be confused by representational differences
/// (explicit `x |-> x` entries, duplicate structural encodings — see
/// `normalize_perm_group`).
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct NamePerm {
    /// Sorted by source name; every entry has `from != to`.
    entries: Vec<(Arc<str>, Arc<str>)>,
}

impl NamePerm {
    /// Extract a name-space permutation from a declared symmetry `FuncValue`.
    ///
    /// Fails closed (`None`) when any entry is not `ModelValue -> ModelValue`
    /// or the moved entries are not a bijection on their support.
    pub(crate) fn from_func_value(perm: &FuncValue) -> Option<Self> {
        let mut entries: Vec<(Arc<str>, Arc<str>)> = Vec::with_capacity(perm.domain_len());
        for (key, val) in perm.mapping_iter() {
            let (Value::ModelValue(from), Value::ModelValue(to)) = (key, val) else {
                return None;
            };
            if from == to {
                continue; // identity entry — representational only
            }
            entries.push((from.into(), to.into()));
        }
        entries.sort_by(|a, b| a.0.cmp(&b.0));
        Self::from_moved_entries(entries)
    }

    /// Build from sorted moved entries, validating bijectivity on the support.
    fn from_moved_entries(entries: Vec<(Arc<str>, Arc<str>)>) -> Option<Self> {
        if entries.windows(2).any(|w| w[0].0 >= w[1].0) {
            return None; // duplicate / unsorted source names
        }
        let keys: BTreeSet<&str> = entries.iter().map(|(k, _)| &**k).collect();
        let mut image: BTreeSet<&str> = BTreeSet::new();
        for (_, v) in &entries {
            if !image.insert(&**v) {
                return None; // not injective
            }
        }
        if keys != image {
            // Not a permutation of any finite set: some name is mapped onto
            // but not mapped from (or vice versa), so two elements collide.
            return None;
        }
        Some(NamePerm { entries })
    }

    /// Image of `name`, or `None` when fixed.
    pub(crate) fn apply(&self, name: &str) -> Option<&Arc<str>> {
        self.entries
            .binary_search_by(|(k, _)| (**k).cmp(name))
            .ok()
            .map(|i| &self.entries[i].1)
    }

    /// Total image of `name` (identity on fixed names).
    pub(crate) fn image_of(&self, name: &Arc<str>) -> Arc<str> {
        self.apply(name)
            .cloned()
            .unwrap_or_else(|| Arc::clone(name))
    }

    /// `(self ∘ other)(x) = self(other(x))`.
    pub(crate) fn compose(&self, other: &NamePerm) -> Option<NamePerm> {
        let mut names: BTreeSet<Arc<str>> = BTreeSet::new();
        for (k, _) in &self.entries {
            names.insert(Arc::clone(k));
        }
        for (k, _) in &other.entries {
            names.insert(Arc::clone(k));
        }
        let mut entries = Vec::new();
        for name in names {
            let mid = other.image_of(&name);
            let out = self.image_of(&mid);
            if *out != *name {
                entries.push((name, out));
            }
        }
        // `names` iterates ascending, so entries are sorted by source.
        Self::from_moved_entries(entries)
    }

    pub(crate) fn is_identity(&self) -> bool {
        self.entries.is_empty()
    }

    pub(crate) fn entries(&self) -> &[(Arc<str>, Arc<str>)] {
        &self.entries
    }
}

/// Enumerate the group generated by `generators` under composition, using the
/// same incremental frontier algorithm as `compute_symmetry_perms`
/// (TLC `MVPerms.permutationSubgroup`), with a hard cap.
///
/// The identity element is intentionally excluded from the result: the lexmin
/// seeds with the unpermuted buffer, which is exactly the identity's
/// contribution. Returns `None` when the closure exceeds `cap` (fail closed).
pub(crate) fn close_group(generators: Vec<NamePerm>, cap: usize) -> Option<Vec<NamePerm>> {
    let mut seen_set: BTreeSet<NamePerm> = generators.iter().cloned().collect();
    let mut seen_vec: Vec<NamePerm> = seen_set.iter().cloned().collect();
    if seen_vec.len() > cap {
        return None;
    }
    let gens = generators;
    let mut frontier_start = 0;
    loop {
        let frontier_end = seen_vec.len();
        if frontier_start == frontier_end {
            break;
        }
        for idx in frontier_start..frontier_end {
            let elem = seen_vec[idx].clone();
            for gen in &gens {
                let composed = gen.compose(&elem)?;
                if composed.is_identity() {
                    continue; // g ∘ e = id contributes nothing new
                }
                if seen_set.insert(composed.clone()) {
                    seen_vec.push(composed);
                    if seen_vec.len() > cap {
                        return None;
                    }
                }
            }
        }
        frontier_start = frontier_end;
    }
    Some(seen_vec)
}

/// Interned-`NameId` remap for one group element: `raw NameId -> raw NameId`,
/// identity outside the (finite) support.
///
/// Total over ALL interned names, which is what makes an unrestricted
/// `ScalarModelValue` slot admissible: `Value::permute_impl` fixes every model
/// value outside the permutation domain, and so does this table.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct MvRemap {
    /// Sorted by source id.
    pairs: Vec<(i64, i64)>,
}

impl MvRemap {
    fn for_perm(perm: &NamePerm) -> Self {
        let mut pairs: Vec<(i64, i64)> = perm
            .entries()
            .iter()
            .map(|(from, to)| {
                (
                    i64::from(tla_core::intern_name(from).0),
                    i64::from(tla_core::intern_name(to).0),
                )
            })
            .collect();
        pairs.sort_unstable_by_key(|p| p.0);
        MvRemap { pairs }
    }

    #[inline]
    fn remap(&self, raw: i64) -> i64 {
        match self.pairs.binary_search_by(|p| p.0.cmp(&raw)) {
            Ok(i) => self.pairs[i].1,
            Err(_) => raw,
        }
    }
}

/// Per-slot value transform of a [`FlatSymmetryAction`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum SlotTransform {
    /// Raw value unchanged (`Int`/`Bool`/`String` payloads, fixed content).
    Identity,
    /// Raw is an interned model-value `NameId`: remap by the group element.
    MvRemap(Arc<MvRemap>),
    /// Raw is an index into a fixed proof universe (`TaggedScalarUnion`):
    /// `table[i]` = index of the permuted `universe[i]`.
    IndexRemap(Arc<Vec<i64>>),
    /// Raw is a one-slot set bitmask: bit `i` moves to bit `table[i]`.
    MaskPermute(Arc<Vec<u32>>),
    /// `TaggedScalarOrSet` slot: `raw >= 0` is the scalar arm (NameId/`Bool`
    /// payload), `raw < 0` is the `-1 - mask` set arm over the proof's set
    /// universe.
    TaggedScalarOrMask {
        scalar: Arc<MvRemap>,
        mask: Arc<Vec<u32>>,
    },
}

impl SlotTransform {
    #[inline]
    fn apply(&self, raw: i64) -> i64 {
        match self {
            SlotTransform::Identity => raw,
            SlotTransform::MvRemap(remap) => remap.remap(raw),
            SlotTransform::IndexRemap(table) => {
                if let Ok(index) = usize::try_from(raw) {
                    if index < table.len() {
                        return table[index];
                    }
                }
                // Cannot occur for buffers under the fits-contract (encode
                // only emits valid universe indices). Identity fallback keeps
                // release builds fail-safe rather than aliasing.
                debug_assert!(
                    false,
                    "IndexRemap raw {raw} outside universe (len {})",
                    table.len()
                );
                raw
            }
            SlotTransform::MaskPermute(dest) => permute_mask_bits(raw, dest),
            SlotTransform::TaggedScalarOrMask { scalar, mask } => {
                if raw >= 0 {
                    scalar.remap(raw)
                } else {
                    let Some(bits) = raw.checked_add(1).and_then(i64::checked_neg) else {
                        debug_assert!(false, "non-canonical tagged set slot {raw}");
                        return raw;
                    };
                    -1 - permute_mask_bits(bits, mask)
                }
            }
        }
    }

    fn is_identity(&self) -> bool {
        matches!(self, SlotTransform::Identity)
    }
}

/// Permute a one-word bitmask: source bit `i` lands on bit `dest[i]`.
fn permute_mask_bits(mask: i64, dest: &[u32]) -> i64 {
    let bits = mask as u64;
    debug_assert!(
        dest.len() >= 64 || bits >> dest.len() == 0,
        "bitmask has bits outside its {}-element universe",
        dest.len()
    );
    let mut out = 0u64;
    for (i, &d) in dest.iter().enumerate() {
        if (bits >> i) & 1 == 1 {
            out |= 1u64 << d;
        }
    }
    out as i64
}

/// One compiled group element: a total slot-level action on the flat buffer.
#[derive(Debug, Clone)]
pub(crate) struct FlatSymmetryAction {
    /// For output slot `j`, the input slot whose (transformed) value lands in
    /// `j`. Always a permutation of `0..len`.
    src: Box<[u32]>,
    /// Per-output-slot value transform, applied to `input[src[j]]`.
    transforms: Box<[SlotTransform]>,
    /// True when the action provably maps every fitting buffer to itself.
    identity: bool,
}

impl FlatSymmetryAction {
    /// Apply the action out-of-place: `output = action(input)`.
    pub(crate) fn apply(&self, input: &[i64], output: &mut [i64]) {
        debug_assert_eq!(input.len(), self.src.len());
        debug_assert_eq!(output.len(), self.src.len());
        for j in 0..self.src.len() {
            output[j] = self.transforms[j].apply(input[self.src[j] as usize]);
        }
    }

    /// The permuted buffer's slot `j`, computed on demand (streaming lexmin).
    #[inline]
    fn apply_slot(&self, input: &[i64], j: usize) -> i64 {
        self.transforms[j].apply(input[self.src[j] as usize])
    }

    pub(crate) fn is_identity(&self) -> bool {
        self.identity
    }
}

/// One slot of a per-variable permutation table, relative to the var's window.
struct SlotEntry {
    src_rel: usize,
    transform: SlotTransform,
}

type PermTable = Vec<SlotEntry>;

fn identity_table(slot_count: usize) -> PermTable {
    (0..slot_count)
        .map(|i| SlotEntry {
            src_rel: i,
            transform: SlotTransform::Identity,
        })
        .collect()
}

/// Transform for a scalar payload of the given slot type.
///
/// `Int`/`Bool` are untouched by TLA+ symmetry permutations; `String` too
/// (`Value::permute_impl` fixes strings — only *model values* move). Under the
/// fits-contract a `String`-typed slot always holds a genuine string, so
/// identity is exact and there is no NameId ambiguity with model values.
fn transform_for_slot_type(slot_type: SlotType, remap: &Arc<MvRemap>) -> SlotTransform {
    match slot_type {
        SlotType::Int | SlotType::Bool | SlotType::String => SlotTransform::Identity,
        SlotType::ModelValue => SlotTransform::MvRemap(Arc::clone(remap)),
    }
}

fn permute_flat_scalar(value: &FlatScalarValue, perm: &NamePerm) -> FlatScalarValue {
    match value {
        FlatScalarValue::ModelValue(name) => FlatScalarValue::ModelValue(perm.image_of(name)),
        other => other.clone(),
    }
}

/// True when the perm provably fixes every element of the scalar universe.
fn scalar_universe_fixed(universe: &[FlatScalarValue], perm: &NamePerm) -> bool {
    universe.iter().all(|value| match value {
        FlatScalarValue::ModelValue(name) => perm.apply(name).is_none(),
        _ => true,
    })
}

/// Destination table for a permuted scalar universe: `table[i]` = position of
/// the permuted `universe[i]`. `None` when the universe is not closed under
/// the perm or the induced map is not a bijection (fail closed).
fn universe_dest_positions(universe: &[FlatScalarValue], perm: &NamePerm) -> Option<Vec<usize>> {
    let mut dest = Vec::with_capacity(universe.len());
    let mut seen = vec![false; universe.len()];
    for value in universe {
        let permuted = permute_flat_scalar(value, perm);
        let d = universe.iter().position(|candidate| *candidate == permuted)?;
        if seen[d] {
            return None;
        }
        seen[d] = true;
        dest.push(d);
    }
    Some(dest)
}

fn universe_index_remap(universe: &[FlatScalarValue], perm: &NamePerm) -> Option<Vec<i64>> {
    universe_dest_positions(universe, perm).map(|t| t.into_iter().map(|d| d as i64).collect())
}

fn universe_bit_dest(universe: &[FlatScalarValue], perm: &NamePerm) -> Option<Vec<u32>> {
    universe_dest_positions(universe, perm).map(|t| t.into_iter().map(|d| d as u32).collect())
}

/// Conservative "the perm provably cannot change this value" check for
/// concrete universe members (e.g. `RecordSetBitmask` records). Anything not
/// recognized is reported as NOT fixed (decline).
fn value_definitely_fixed(value: &Value, perm: &NamePerm) -> bool {
    match value {
        Value::Bool(_) | Value::SmallInt(_) | Value::Int(_) | Value::String(_) => true,
        Value::ModelValue(name) => perm.apply(name).is_none(),
        Value::Tuple(items) => items.iter().all(|v| value_definitely_fixed(v, perm)),
        Value::Seq(items) => items.iter().all(|v| value_definitely_fixed(v, perm)),
        Value::Set(set) => set.iter().all(|v| value_definitely_fixed(v, perm)),
        Value::Record(record) => record.iter().all(|(_, v)| value_definitely_fixed(v, perm)),
        Value::Func(func) => func
            .mapping_iter()
            .all(|(k, v)| value_definitely_fixed(k, perm) && value_definitely_fixed(v, perm)),
        _ => false,
    }
}

/// True when the perm provably cannot affect ANY value encodable by `layout`.
/// This is the identity-admission escape used for shapes whose non-trivial
/// slot movement this module does not (yet) express.
fn flat_value_layout_fixed(layout: &FlatValueLayout, perm: &NamePerm) -> bool {
    if perm.is_identity() {
        return true;
    }
    match layout {
        FlatValueLayout::Scalar(SlotType::Int | SlotType::Bool | SlotType::String) => true,
        // An unrestricted model-value slot can hold any support element.
        FlatValueLayout::Scalar(SlotType::ModelValue) => false,
        FlatValueLayout::IntFunction { value_layout, .. } => {
            flat_value_layout_fixed(value_layout, perm)
        }
        FlatValueLayout::Function {
            domain,
            value_layout,
        } => scalar_universe_fixed(domain, perm) && flat_value_layout_fixed(value_layout, perm),
        FlatValueLayout::Record { field_layouts, .. } => field_layouts
            .iter()
            .all(|field| flat_value_layout_fixed(field, perm)),
        FlatValueLayout::SetBitmask { universe, .. } => scalar_universe_fixed(universe, perm),
        FlatValueLayout::RecordSetBitmask { universe, .. } => universe
            .iter()
            .all(|record| value_definitely_fixed(record, perm)),
        FlatValueLayout::NestedSetBitmask { inner_universe, .. } => {
            scalar_universe_fixed(inner_universe, perm)
        }
        FlatValueLayout::TaggedScalarUnion { proof } => {
            scalar_universe_fixed(proof.universe(), perm)
        }
        FlatValueLayout::TaggedUnion { proof } => proof
            .variants()
            .iter()
            .all(|variant| flat_value_layout_fixed(variant, perm)),
        FlatValueLayout::HeterogeneousTuple { element_layouts } => element_layouts
            .iter()
            .all(|element| flat_value_layout_fixed(element, perm)),
        FlatValueLayout::Sequence { element_layout, .. } => {
            flat_value_layout_fixed(element_layout, perm)
        }
    }
}

/// Compile one recursive value layout against one group element.
fn compile_flat_value_layout(
    layout: &FlatValueLayout,
    perm: &NamePerm,
    remap: &Arc<MvRemap>,
) -> Option<PermTable> {
    // Provably-fixed content is admitted as identity regardless of shape.
    // This is also the ONLY admission for Sequence / TaggedUnion /
    // RecordSetBitmask / NestedSetBitmask (see below).
    if flat_value_layout_fixed(layout, perm) {
        return Some(identity_table(layout.slot_count()));
    }
    match layout {
        FlatValueLayout::Scalar(slot_type) => Some(vec![SlotEntry {
            src_rel: 0,
            transform: transform_for_slot_type(*slot_type, remap),
        }]),
        FlatValueLayout::IntFunction {
            len, value_layout, ..
        } => {
            // Integer keys are fixed by the perm; only the child windows
            // transform in place.
            let child = compile_flat_value_layout(value_layout, perm, remap)?;
            let width = value_layout.slot_count();
            let mut table = Vec::with_capacity(len * width);
            for i in 0..*len {
                for entry in &child {
                    table.push(SlotEntry {
                        src_rel: i * width + entry.src_rel,
                        transform: entry.transform.clone(),
                    });
                }
            }
            Some(table)
        }
        FlatValueLayout::Function {
            domain,
            value_layout,
        } => {
            // Model-value keys move: the whole child window of key `k` lands
            // on the window of `p(k)`. Domain must be closed under the perm.
            let dest = universe_dest_positions(domain, perm)?;
            let child = compile_flat_value_layout(value_layout, perm, remap)?;
            let width = value_layout.slot_count();
            let mut table = identity_table(domain.len() * width);
            for (i, &d) in dest.iter().enumerate() {
                for (r, entry) in child.iter().enumerate() {
                    table[d * width + r] = SlotEntry {
                        src_rel: i * width + entry.src_rel,
                        transform: entry.transform.clone(),
                    };
                }
            }
            Some(table)
        }
        FlatValueLayout::Record { field_layouts, .. } => {
            // Record field names are NameId keys the perm never touches.
            let mut table = Vec::new();
            for field_layout in field_layouts {
                let child = compile_flat_value_layout(field_layout, perm, remap)?;
                let base = table.len();
                for entry in child {
                    table.push(SlotEntry {
                        src_rel: base + entry.src_rel,
                        transform: entry.transform,
                    });
                }
            }
            Some(table)
        }
        FlatValueLayout::SetBitmask {
            universe,
            universe_closure,
        } => {
            // A merely-sampled universe is declined for non-identity
            // transforms: the sample carries no guarantee that it is the
            // authoritative element ordering for every reachable write, and a
            // wrong canonicalization is an unsound dedup. (The fixed case was
            // already admitted as identity above.)
            if !universe_closure.is_proven_closed() {
                return None;
            }
            let dest = universe_bit_dest(universe, perm)?;
            Some(vec![SlotEntry {
                src_rel: 0,
                transform: SlotTransform::MaskPermute(Arc::new(dest)),
            }])
        }
        FlatValueLayout::TaggedScalarUnion { proof } => {
            let table = universe_index_remap(proof.universe(), perm)?;
            Some(vec![SlotEntry {
                src_rel: 0,
                transform: SlotTransform::IndexRemap(Arc::new(table)),
            }])
        }
        FlatValueLayout::HeterogeneousTuple { element_layouts } => {
            let mut table = Vec::new();
            for element_layout in element_layouts {
                let child = compile_flat_value_layout(element_layout, perm, remap)?;
                let base = table.len();
                for entry in child {
                    table.push(SlotEntry {
                        src_rel: base + entry.src_rel,
                        transform: entry.transform,
                    });
                }
            }
            Some(table)
        }
        // Non-fixed content in shapes whose slot movement is data-dependent
        // (Sequence live windows, TaggedUnion active variant) or crosses slot
        // words (RecordSetBitmask, NestedSetBitmask multi-word masks): decline.
        FlatValueLayout::Sequence { .. }
        | FlatValueLayout::TaggedUnion { .. }
        | FlatValueLayout::RecordSetBitmask { .. }
        | FlatValueLayout::NestedSetBitmask { .. } => None,
    }
}

/// Verify closure of a `FixedScalarRangeProof`-style universe when the proof
/// types model values; `String`-typed universes are trivially fixed.
fn fixed_scalar_universe_admissible(
    scalar_type: SlotType,
    scalar_universe: &[FlatScalarValue],
    perm: &NamePerm,
) -> bool {
    match scalar_type {
        SlotType::ModelValue => {
            scalar_universe_fixed(scalar_universe, perm)
                || universe_dest_positions(scalar_universe, perm).is_some()
        }
        _ => true,
    }
}

/// Compile one variable's layout kind against one group element.
fn compile_var_kind(
    kind: &VarLayoutKind,
    perm: &NamePerm,
    remap: &Arc<MvRemap>,
) -> Option<PermTable> {
    match kind {
        // Int / Bool / String scalars are fixed by every symmetry perm.
        VarLayoutKind::Scalar | VarLayoutKind::ScalarBool | VarLayoutKind::ScalarString => {
            Some(identity_table(1))
        }
        // Unrestricted model-value slot: total NameId remap (identity outside
        // the perm support, exactly like `Value::permute_impl`).
        VarLayoutKind::ScalarModelValue => Some(vec![SlotEntry {
            src_rel: 0,
            transform: SlotTransform::MvRemap(Arc::clone(remap)),
        }]),
        VarLayoutKind::FixedScalar { base, proof } => match base {
            SlotType::String => Some(identity_table(1)),
            SlotType::ModelValue => {
                // The proven universe must be closed under the perm, or the
                // permuted state would no longer satisfy the layout's proof.
                if !fixed_scalar_universe_admissible(
                    SlotType::ModelValue,
                    proof.scalar_universe(),
                    perm,
                ) {
                    return None;
                }
                Some(vec![SlotEntry {
                    src_rel: 0,
                    transform: SlotTransform::MvRemap(Arc::clone(remap)),
                }])
            }
            // FixedScalar is only constructed for String/ModelValue bases.
            SlotType::Int | SlotType::Bool => None,
        },
        VarLayoutKind::IntArray {
            len,
            element_types,
            element_range_proof,
            ..
        } => {
            if let Some(proof) = element_range_proof {
                if !fixed_scalar_universe_admissible(
                    proof.scalar_type(),
                    proof.scalar_universe(),
                    perm,
                ) {
                    return None;
                }
            }
            match element_types {
                // No element types => all Int/Bool => fixed.
                None => Some(identity_table(*len)),
                Some(types) => {
                    if types.len() != *len {
                        return None;
                    }
                    Some(
                        types
                            .iter()
                            .enumerate()
                            .map(|(i, slot_type)| SlotEntry {
                                src_rel: i,
                                transform: transform_for_slot_type(*slot_type, remap),
                            })
                            .collect(),
                    )
                }
            }
        }
        VarLayoutKind::Record {
            field_names,
            field_types,
            field_range_proofs,
            ..
        } => {
            if field_types.len() != field_names.len() {
                return None;
            }
            if let Some(proofs) = field_range_proofs {
                if proofs.len() != field_names.len() {
                    return None;
                }
                for proof in proofs.iter().flatten() {
                    if !fixed_scalar_universe_admissible(
                        proof.scalar_type(),
                        proof.scalar_universe(),
                        perm,
                    ) {
                        return None;
                    }
                }
            }
            // Field names are interned keys the perm never touches; only the
            // per-field payloads transform in place.
            Some(
                field_types
                    .iter()
                    .enumerate()
                    .map(|(i, slot_type)| SlotEntry {
                        src_rel: i,
                        transform: transform_for_slot_type(*slot_type, remap),
                    })
                    .collect(),
            )
        }
        VarLayoutKind::StringKeyedArray {
            domain_keys,
            domain_types,
            value_types,
            range_encoding,
        } => {
            let n = domain_keys.len();
            if domain_types.len() != n || value_types.len() != n {
                return None;
            }
            // dest[i] = slot of the permuted key i. String keys are fixed;
            // model-value keys must land on a model-value key of the domain
            // (closure), injectively.
            let mut dest = Vec::with_capacity(n);
            let mut seen = vec![false; n];
            for (i, (key, key_type)) in domain_keys.iter().zip(domain_types.iter()).enumerate() {
                let d = match key_type {
                    SlotType::String => i,
                    SlotType::ModelValue => {
                        let permuted = perm.image_of(key);
                        domain_keys
                            .iter()
                            .zip(domain_types.iter())
                            .position(|(candidate, candidate_type)| {
                                *candidate_type == SlotType::ModelValue && **candidate == *permuted
                            })?
                    }
                    // String-keyed layouts only carry String/ModelValue keys.
                    SlotType::Int | SlotType::Bool => return None,
                };
                if seen[d] {
                    return None;
                }
                seen[d] = true;
                dest.push(d);
            }
            let per_source: Vec<SlotTransform> = match range_encoding {
                StringKeyedArrayRangeEncoding::ScalarSlots => value_types
                    .iter()
                    .map(|slot_type| transform_for_slot_type(*slot_type, remap))
                    .collect(),
                StringKeyedArrayRangeEncoding::FixedScalar(proof) => {
                    if !fixed_scalar_universe_admissible(
                        proof.scalar_type(),
                        proof.scalar_universe(),
                        perm,
                    ) {
                        return None;
                    }
                    vec![transform_for_slot_type(proof.scalar_type(), remap); n]
                }
                StringKeyedArrayRangeEncoding::TaggedScalarOrSet(proof) => {
                    let mask = universe_bit_dest(proof.set_universe(), perm)?;
                    let scalar = match proof.scalar_type() {
                        SlotType::ModelValue => Arc::clone(remap),
                        // Int/Bool/String scalar arms are fixed: empty remap.
                        _ => Arc::new(MvRemap { pairs: Vec::new() }),
                    };
                    let transform = SlotTransform::TaggedScalarOrMask {
                        scalar,
                        mask: Arc::new(mask),
                    };
                    vec![transform; n]
                }
            };
            // The value moving from slot i must be legally encodable at its
            // destination slot: per-slot types must agree across the orbit.
            // (Uniform-range encodings satisfy this trivially.)
            if matches!(range_encoding, StringKeyedArrayRangeEncoding::ScalarSlots) {
                for (i, &d) in dest.iter().enumerate() {
                    if value_types[i] != value_types[d] {
                        return None;
                    }
                }
            }
            let mut table = identity_table(n);
            for (i, &d) in dest.iter().enumerate() {
                table[d] = SlotEntry {
                    src_rel: i,
                    transform: per_source[i].clone(),
                };
            }
            Some(table)
        }
        VarLayoutKind::TupleKeyedArray {
            domain_keys,
            value_types,
            range_encoding,
        } => {
            let n = domain_keys.len();
            if value_types.len() != n {
                return None;
            }
            let mut dest = Vec::with_capacity(n);
            let mut seen = vec![false; n];
            for key in domain_keys {
                let permuted = permute_scalar_tree(key, perm)?;
                // The canonical key table is sorted ascending; a miss means
                // the domain is not closed under the perm (or unsorted —
                // either way: decline).
                let d = domain_keys
                    .binary_search_by(|candidate| candidate.cmp(&permuted))
                    .ok()?;
                if seen[d] {
                    return None;
                }
                seen[d] = true;
                dest.push(d);
            }
            let per_source: Vec<SlotTransform> = match range_encoding {
                TupleKeyedArrayRangeEncoding::ScalarSlots => value_types
                    .iter()
                    .map(|slot_type| transform_for_slot_type(*slot_type, remap))
                    .collect(),
                TupleKeyedArrayRangeEncoding::FixedScalar(proof) => {
                    if !fixed_scalar_universe_admissible(
                        proof.scalar_type(),
                        proof.scalar_universe(),
                        perm,
                    ) {
                        return None;
                    }
                    vec![transform_for_slot_type(proof.scalar_type(), remap); n]
                }
                TupleKeyedArrayRangeEncoding::TaggedScalarUnion(proof) => {
                    let table = Arc::new(universe_index_remap(proof.universe(), perm)?);
                    vec![SlotTransform::IndexRemap(table); n]
                }
            };
            if matches!(
                range_encoding,
                TupleKeyedArrayRangeEncoding::ScalarSlots
                    | TupleKeyedArrayRangeEncoding::FixedScalar(_)
            ) {
                for (i, &d) in dest.iter().enumerate() {
                    if value_types[i] != value_types[d] {
                        return None;
                    }
                }
            }
            let mut table = identity_table(n);
            for (i, &d) in dest.iter().enumerate() {
                table[d] = SlotEntry {
                    src_rel: i,
                    transform: per_source[i].clone(),
                };
            }
            Some(table)
        }
        VarLayoutKind::Recursive { layout } => compile_flat_value_layout(layout, perm, remap),
        // Placeholder scaffold kind (stores 0 / raw int): no equivariance
        // proof exists — decline.
        VarLayoutKind::Bitmask { .. } => None,
        // Dynamic slots don't encode their value at all: decline.
        VarLayoutKind::Dynamic => None,
    }
}

/// Permute a scalar-or-tuple-of-scalars key `Value` (`TupleKeyedArray`
/// domains). `None` for any shape that is not a scalar tree (decline).
fn permute_scalar_tree(value: &Value, perm: &NamePerm) -> Option<Value> {
    match value {
        Value::Bool(_) | Value::SmallInt(_) | Value::Int(_) | Value::String(_) => {
            Some(value.clone())
        }
        Value::ModelValue(name) => Some(match perm.apply(name) {
            Some(to) => Value::ModelValue(to.into()),
            None => value.clone(),
        }),
        Value::Tuple(items) => {
            let mut out = Vec::with_capacity(items.len());
            for item in items.iter() {
                out.push(permute_scalar_tree(item, perm)?);
            }
            Some(Value::Tuple(out.into()))
        }
        _ => None,
    }
}

/// Compile one group element into a whole-buffer action.
fn compile_action(layout: &StateLayout, perm: &NamePerm) -> Option<FlatSymmetryAction> {
    let total = layout.total_slots();
    let mut src: Vec<u32> = Vec::with_capacity(total);
    for i in 0..total {
        src.push(u32::try_from(i).ok()?);
    }
    let mut transforms = vec![SlotTransform::Identity; total];
    let remap = Arc::new(MvRemap::for_perm(perm));
    for var_layout in layout.iter() {
        let table = compile_var_kind(&var_layout.kind, perm, &remap)?;
        if table.len() != var_layout.slot_count {
            debug_assert!(
                false,
                "compiled table width {} != slot_count {} for var {}",
                table.len(),
                var_layout.slot_count,
                var_layout.name
            );
            return None;
        }
        for (rel, entry) in table.into_iter().enumerate() {
            let abs = var_layout.offset + rel;
            src[abs] = u32::try_from(var_layout.offset + entry.src_rel).ok()?;
            transforms[abs] = entry.transform;
        }
    }
    let identity = src.iter().enumerate().all(|(j, &s)| s as usize == j)
        && transforms.iter().all(SlotTransform::is_identity);
    Some(FlatSymmetryAction {
        src: src.into_boxed_slice(),
        transforms: transforms.into_boxed_slice(),
        identity,
    })
}

/// The compiled flat-space symmetry group for one `StateLayout`: one action
/// per non-identity group element plus the streaming lexmin canonicalizer.
#[derive(Debug, Clone)]
pub(crate) struct FlatSymmetryCanonicalizer {
    num_slots: usize,
    /// Group elements (identity excluded), parallel to `actions`.
    elements: Vec<NamePerm>,
    actions: Vec<FlatSymmetryAction>,
    /// False when every action is the identity (all orbits are singletons for
    /// fitting states): `canonicalize_in_place` is a no-op then.
    has_effective_action: bool,
}

impl FlatSymmetryCanonicalizer {
    /// Compile the declared symmetry permutations against a layout.
    ///
    /// Fail-closed on ALL of: non-model-value/non-bijective perms, closure
    /// blow-up past [`MAX_GROUP_ORDER`], table footprint past
    /// [`MAX_TOTAL_TABLE_SLOTS`], and any variable kind that is not provably
    /// equivariant-representable for every group element (see module docs).
    /// Also returns `None` when the group is trivial (no symmetry to exploit).
    pub(crate) fn compile(
        layout: &StateLayout,
        declared_perms: &[FuncValue],
    ) -> Option<FlatSymmetryCanonicalizer> {
        let mut generators = Vec::new();
        for perm in declared_perms {
            let name_perm = NamePerm::from_func_value(perm)?;
            if !name_perm.is_identity() {
                generators.push(name_perm);
            }
        }
        if generators.is_empty() {
            return None;
        }
        let group = close_group(generators, MAX_GROUP_ORDER)?;
        let table_footprint = group.len().checked_mul(layout.total_slots())?;
        if table_footprint > MAX_TOTAL_TABLE_SLOTS {
            return None;
        }
        let mut elements = Vec::with_capacity(group.len());
        let mut actions = Vec::with_capacity(group.len());
        for perm in group {
            let action = compile_action(layout, &perm)?;
            elements.push(perm);
            actions.push(action);
        }
        let has_effective_action = actions.iter().any(|action| !action.is_identity());
        Some(FlatSymmetryCanonicalizer {
            num_slots: layout.total_slots(),
            elements,
            actions,
            has_effective_action,
        })
    }

    pub(crate) fn num_slots(&self) -> usize {
        self.num_slots
    }

    /// Group order including the (elided) identity element.
    pub(crate) fn group_order(&self) -> usize {
        self.actions.len() + 1
    }

    pub(crate) fn elements(&self) -> &[NamePerm] {
        &self.elements
    }

    pub(crate) fn actions(&self) -> &[FlatSymmetryAction] {
        &self.actions
    }

    /// Replace `buf` with the lexicographic minimum (per-slot signed i64
    /// order) of its orbit under the compiled group, streaming: candidates are
    /// compared slot-by-slot against the incumbent and abandoned at the first
    /// losing slot; a winning candidate finishes writing from its first
    /// strictly-smaller slot (the prefix is already equal). No |G|-way
    /// materialization ever happens. `scratch` holds one copy of the input.
    pub(crate) fn canonicalize_in_place(&self, buf: &mut [i64], scratch: &mut Vec<i64>) {
        debug_assert_eq!(
            buf.len(),
            self.num_slots,
            "canonicalize_in_place: buffer/layout slot-count mismatch"
        );
        if buf.len() != self.num_slots || !self.has_effective_action {
            // Fail-safe: never rewrite a buffer this group was not compiled
            // for (release builds keep the raw buffer, which is sound — just
            // uncanonicalized; slice 2 must never mix domains anyway).
            return;
        }
        scratch.clear();
        scratch.extend_from_slice(buf);
        let original: &[i64] = scratch;
        let n = buf.len();
        'actions: for action in &self.actions {
            if action.identity {
                continue;
            }
            for j in 0..n {
                let candidate = action.apply_slot(original, j);
                match candidate.cmp(&buf[j]) {
                    Ordering::Greater => continue 'actions,
                    Ordering::Less => {
                        // New minimum: prefix [0..j) already equals the
                        // incumbent's, finish this candidate directly.
                        buf[j] = candidate;
                        for r in (j + 1)..n {
                            buf[r] = action.apply_slot(original, r);
                        }
                        continue 'actions;
                    }
                    Ordering::Equal => {}
                }
            }
            // Candidate ties the incumbent exactly: keep the incumbent.
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::state_layout::{
        FixedScalarRangeProof, SequenceBoundEvidence, SetBitmaskUniverseClosure,
        TaggedScalarSetRangeProof, TaggedScalarUnionProof, TaggedUnionProof,
    };
    use super::*;
    use crate::state::{ArrayState, FlatState};
    use crate::var_index::VarRegistry;

    // ---------- oracle helpers ----------
    //
    // THE ORACLE: the interpreter's Value-tree permutation
    // (`Value::permute`, state/symmetry.rs machinery) applied per variable,
    // then encoded through the production `FlatState` writer. Every compiled
    // action must byte-match `encode ∘ permute` on every fitting state.

    fn mv(name: &str) -> Value {
        Value::ModelValue(Rp::from(name))
    }

    fn string_value(s: &str) -> Value {
        Value::String(Rp::from(s))
    }

    fn perm_func(pairs: &[(&str, &str)]) -> FuncValue {
        let mut entries: Vec<(Value, Value)> =
            pairs.iter().map(|(from, to)| (mv(from), mv(to))).collect();
        entries.sort_by(|a, b| a.0.cmp(&b.0));
        FuncValue::from_sorted_entries(entries)
    }

    fn func_value_of(perm: &NamePerm) -> FuncValue {
        let mut entries: Vec<(Value, Value)> = perm
            .entries()
            .iter()
            .map(|(from, to)| {
                (
                    Value::ModelValue(from.into()),
                    Value::ModelValue(to.into()),
                )
            })
            .collect();
        entries.sort_by(|a, b| a.0.cmp(&b.0));
        FuncValue::from_sorted_entries(entries)
    }

    fn func_of(entries: Vec<(Value, Value)>) -> Value {
        let mut entries = entries;
        entries.sort_by(|a, b| a.0.cmp(&b.0));
        Value::Func(Rp::new(FuncValue::from_sorted_entries(entries)))
    }

    fn permute_state(state: &ArrayState, perm: &FuncValue) -> ArrayState {
        let values: Vec<Value> = state
            .values()
            .iter()
            .map(|cv| Value::from(cv).permute(perm))
            .collect();
        ArrayState::from_values(values)
    }

    fn encode(state: &ArrayState, layout: &Arc<StateLayout>) -> Vec<i64> {
        FlatState::try_from_array_state(state, Arc::clone(layout))
            .expect("test state must encode under the layout")
            .buffer()
            .to_vec()
    }

    fn canonical_of(canon: &FlatSymmetryCanonicalizer, buf: &[i64]) -> Vec<i64> {
        let mut out = buf.to_vec();
        let mut scratch = Vec::new();
        canon.canonicalize_in_place(&mut out, &mut scratch);
        out
    }

    fn states_value_equal(a: &ArrayState, b: &ArrayState) -> bool {
        a.values().len() == b.values().len()
            && a.values()
                .iter()
                .zip(b.values().iter())
                .all(|(x, y)| Value::from(x) == Value::from(y))
    }

    /// Equivariance oracle: for every compiled group element g and state s,
    /// `action_g(encode(s)) == encode(permute_value_tree(g, s))`.
    fn assert_equivariant(
        layout: &Arc<StateLayout>,
        canon: &FlatSymmetryCanonicalizer,
        states: &[ArrayState],
    ) {
        assert_eq!(canon.elements().len(), canon.actions().len());
        for state in states {
            let base = encode(state, layout);
            for (perm, action) in canon.elements().iter().zip(canon.actions().iter()) {
                let oracle = encode(&permute_state(state, &func_value_of(perm)), layout);
                let mut got = vec![0i64; base.len()];
                action.apply(&base, &mut got);
                assert_eq!(
                    oracle, got,
                    "flat action != encode∘permute for perm {perm:?}"
                );
            }
        }
    }

    /// Group invariance + brute-force lexmin cross-check:
    /// `canonical(g·x) == canonical(x)` for every g, and the streaming lexmin
    /// equals the min over the materialized orbit.
    fn assert_canonical_sound(
        layout: &Arc<StateLayout>,
        canon: &FlatSymmetryCanonicalizer,
        states: &[ArrayState],
    ) {
        for state in states {
            let base = encode(state, layout);
            let canonical = canonical_of(canon, &base);
            let mut orbit = vec![base.clone()];
            for action in canon.actions() {
                let mut out = vec![0i64; base.len()];
                action.apply(&base, &mut out);
                orbit.push(out);
            }
            let brute_min = orbit.iter().min().expect("orbit is non-empty").clone();
            assert_eq!(canonical, brute_min, "streaming lexmin != brute-force min");
            for member in &orbit {
                assert_eq!(
                    canonical_of(canon, member),
                    canonical,
                    "canonical must be invariant across the orbit"
                );
            }
        }
    }

    /// ORBIT-PARTITION parity vs the interpreter canonicalizer: two states are
    /// in the same interpreter orbit IFF their flat canonicals agree. The
    /// representatives may differ (interpreter minimizes in Value order, this
    /// module in slot order) — only the partition is compared.
    fn assert_orbit_partition_matches_interpreter(
        layout: &Arc<StateLayout>,
        canon: &FlatSymmetryCanonicalizer,
        states: &[ArrayState],
    ) {
        let perms: Vec<FuncValue> = canon.elements().iter().map(func_value_of).collect();
        for a in states {
            let canonical_a = canonical_of(canon, &encode(a, layout));
            for b in states {
                let interp_same = states_value_equal(a, b)
                    || perms
                        .iter()
                        .any(|perm| states_value_equal(&permute_state(a, perm), b));
                let flat_same = canonical_a == canonical_of(canon, &encode(b, layout));
                assert_eq!(
                    interp_same, flat_same,
                    "orbit partition diverged from interpreter for a={a:?} b={b:?}"
                );
            }
        }
    }

    fn assert_all(
        layout: &Arc<StateLayout>,
        canon: &FlatSymmetryCanonicalizer,
        states: &[ArrayState],
    ) {
        assert_equivariant(layout, canon, states);
        assert_canonical_sound(layout, canon, states);
        assert_orbit_partition_matches_interpreter(layout, canon, states);
    }

    // ---------- scalar kinds ----------

    #[cfg_attr(test, ntest::timeout(60000))]
    #[test]
    fn scalar_kinds_equivariance_and_orbit_parity() {
        let registry = VarRegistry::from_names(["i", "b", "m", "s"]);
        let layout = Arc::new(StateLayout::new(
            &registry,
            vec![
                VarLayoutKind::Scalar,
                VarLayoutKind::ScalarBool,
                VarLayoutKind::ScalarModelValue,
                VarLayoutKind::ScalarString,
            ],
        ));
        let canon = FlatSymmetryCanonicalizer::compile(
            &layout,
            &[perm_func(&[("wp11a", "wp11b"), ("wp11b", "wp11a")])],
        )
        .expect("all-scalar layout must admit a model-value swap");
        assert_eq!(canon.group_order(), 2);

        let mut states = Vec::new();
        for mv_name in ["wp11a", "wp11b", "wp11c"] {
            for i in [0i64, 5] {
                for flag in [false, true] {
                    states.push(ArrayState::from_values(vec![
                        Value::SmallInt(i),
                        Value::Bool(flag),
                        mv(mv_name),
                        string_value("wp11str"),
                    ]));
                }
            }
        }
        assert_all(&layout, &canon, &states);
    }

    // ---------- IntArray + Record payload remaps ----------

    #[cfg_attr(test, ntest::timeout(60000))]
    #[test]
    fn int_array_and_record_payload_remap() {
        let registry = VarRegistry::from_names(["arr", "rec"]);
        let layout = Arc::new(StateLayout::new(
            &registry,
            vec![
                VarLayoutKind::IntArray {
                    lo: 1,
                    len: 3,
                    elements_are_bool: false,
                    element_types: Some(vec![
                        SlotType::Int,
                        SlotType::ModelValue,
                        SlotType::ModelValue,
                    ]),
                    element_range_proof: None,
                },
                VarLayoutKind::Record {
                    field_names: vec![Arc::from("f"), Arc::from("g")],
                    field_is_bool: vec![false, false],
                    field_types: vec![SlotType::ModelValue, SlotType::Int],
                    field_range_proofs: None,
                },
            ],
        ));
        let canon = FlatSymmetryCanonicalizer::compile(
            &layout,
            &[perm_func(&[("wp11a", "wp11b"), ("wp11b", "wp11a")])],
        )
        .expect("payload-only remap layout must admit");

        let record = |field_f: Value, field_g: Value| {
            let mut builder = crate::value::RecordBuilder::new();
            builder.insert(tla_core::intern_name("f"), field_f);
            builder.insert(tla_core::intern_name("g"), field_g);
            Value::Record(builder.build())
        };
        let arr = |e2: &str, e3: &str| {
            func_of(vec![
                (Value::SmallInt(1), Value::SmallInt(9)),
                (Value::SmallInt(2), mv(e2)),
                (Value::SmallInt(3), mv(e3)),
            ])
        };

        let mut states = Vec::new();
        for e2 in ["wp11a", "wp11b"] {
            for e3 in ["wp11a", "wp11c"] {
                for rf in ["wp11a", "wp11b"] {
                    states.push(ArrayState::from_values(vec![
                        arr(e2, e3),
                        record(mv(rf), Value::SmallInt(4)),
                    ]));
                }
            }
        }
        assert_all(&layout, &canon, &states);
    }

    // ---------- StringKeyedArray: key movement + value remap ----------

    fn mv_keyed_layout(value_types: Vec<SlotType>) -> (VarRegistry, Arc<StateLayout>) {
        let registry = VarRegistry::from_names(["f"]);
        let layout = Arc::new(StateLayout::new(
            &registry,
            vec![VarLayoutKind::StringKeyedArray {
                domain_keys: vec![Arc::from("wp11p1"), Arc::from("wp11p2"), Arc::from("wp11p3")],
                domain_types: vec![SlotType::ModelValue; 3],
                value_types,
                range_encoding: StringKeyedArrayRangeEncoding::ScalarSlots,
            }],
        ));
        (registry, layout)
    }

    #[cfg_attr(test, ntest::timeout(60000))]
    #[test]
    fn string_keyed_array_key_movement_int_range() {
        let (_registry, layout) = mv_keyed_layout(vec![SlotType::Int; 3]);
        let canon = FlatSymmetryCanonicalizer::compile(
            &layout,
            &[perm_func(&[("wp11p1", "wp11p2"), ("wp11p2", "wp11p1")])],
        )
        .expect("closed model-value domain must admit slot movement");

        let make = |v1: i64, v2: i64, v3: i64| {
            ArrayState::from_values(vec![func_of(vec![
                (mv("wp11p1"), Value::SmallInt(v1)),
                (mv("wp11p2"), Value::SmallInt(v2)),
                (mv("wp11p3"), Value::SmallInt(v3)),
            ])])
        };
        let states: Vec<ArrayState> = [(0, 0, 0), (1, 2, 3), (2, 1, 3), (5, 5, 1), (3, 2, 1)]
            .iter()
            .map(|&(a, b, c)| make(a, b, c))
            .collect();
        assert_all(&layout, &canon, &states);
    }

    #[cfg_attr(test, ntest::timeout(60000))]
    #[test]
    fn string_keyed_array_key_and_value_movement_product_group() {
        let (_registry, layout) = mv_keyed_layout(vec![SlotType::ModelValue; 3]);
        // Product group: keys (p1 p2) x values (v1 v2). Closure has 3
        // non-identity elements.
        let canon = FlatSymmetryCanonicalizer::compile(
            &layout,
            &[
                perm_func(&[("wp11p1", "wp11p2"), ("wp11p2", "wp11p1")]),
                perm_func(&[("wp11v1", "wp11v2"), ("wp11v2", "wp11v1")]),
            ],
        )
        .expect("closed domain + model-value range must admit");
        assert_eq!(canon.group_order(), 4, "S2 x S2 product group");

        let make = |v1: &str, v2: &str, v3: &str| {
            ArrayState::from_values(vec![func_of(vec![
                (mv("wp11p1"), mv(v1)),
                (mv("wp11p2"), mv(v2)),
                (mv("wp11p3"), mv(v3)),
            ])])
        };
        let names = ["wp11v1", "wp11v2"];
        let mut states = Vec::new();
        for a in names {
            for b in names {
                for c in names {
                    states.push(make(a, b, c));
                }
            }
        }
        assert_all(&layout, &canon, &states);
    }

    // ---------- TupleKeyedArray + TaggedScalarUnion range ----------

    #[cfg_attr(test, ntest::timeout(60000))]
    #[test]
    fn tuple_keyed_array_tagged_union_range() {
        let mut keys: Vec<Value> = Vec::new();
        for node in ["wp11n1", "wp11n2"] {
            for k in [1i64, 2] {
                keys.push(Value::Tuple(vec![mv(node), Value::SmallInt(k)].into()));
            }
        }
        keys.sort();
        let proof = TaggedScalarUnionProof::new(
            vec![
                FlatScalarValue::Int(0),
                FlatScalarValue::ModelValue(Arc::from("wp11n1")),
                FlatScalarValue::ModelValue(Arc::from("wp11n2")),
            ],
            Arc::from("wp11-test"),
        )
        .expect("valid union universe");
        let registry = VarRegistry::from_names(["childOf"]);
        let layout = Arc::new(StateLayout::new(
            &registry,
            vec![VarLayoutKind::TupleKeyedArray {
                domain_keys: keys.clone(),
                value_types: vec![SlotType::Int; 4],
                range_encoding: TupleKeyedArrayRangeEncoding::TaggedScalarUnion(proof),
            }],
        ));
        let canon = FlatSymmetryCanonicalizer::compile(
            &layout,
            &[perm_func(&[("wp11n1", "wp11n2"), ("wp11n2", "wp11n1")])],
        )
        .expect("closed tuple domain + closed union universe must admit");

        let range = [Value::SmallInt(0), mv("wp11n1"), mv("wp11n2")];
        let mut states = Vec::new();
        for a in 0..range.len() {
            for b in 0..range.len() {
                let entries: Vec<(Value, Value)> = keys
                    .iter()
                    .enumerate()
                    .map(|(i, key)| {
                        let val = if i % 2 == 0 { &range[a] } else { &range[b] };
                        (key.clone(), val.clone())
                    })
                    .collect();
                states.push(ArrayState::from_values(vec![func_of(entries)]));
            }
        }
        assert_all(&layout, &canon, &states);
    }

    // ---------- Recursive: MV-keyed Function over SetBitmask range ----------

    #[cfg_attr(test, ntest::timeout(60000))]
    #[test]
    fn recursive_function_over_proven_set_bitmask() {
        let universe = vec![
            FlatScalarValue::ModelValue(Arc::from("wp11n1")),
            FlatScalarValue::ModelValue(Arc::from("wp11n2")),
            FlatScalarValue::Int(1),
        ];
        let registry = VarRegistry::from_names(["holds"]);
        let layout = Arc::new(StateLayout::new(
            &registry,
            vec![VarLayoutKind::Recursive {
                layout: FlatValueLayout::Function {
                    domain: vec![
                        FlatScalarValue::ModelValue(Arc::from("wp11n1")),
                        FlatScalarValue::ModelValue(Arc::from("wp11n2")),
                    ],
                    value_layout: Box::new(FlatValueLayout::SetBitmask {
                        universe,
                        universe_closure: SetBitmaskUniverseClosure::ProvenClosed {
                            invariant: Arc::from("TypeOK"),
                        },
                    }),
                },
            }],
        ));
        let canon = FlatSymmetryCanonicalizer::compile(
            &layout,
            &[perm_func(&[("wp11n1", "wp11n2"), ("wp11n2", "wp11n1")])],
        )
        .expect("window movement + proven bitmask permutation must admit");

        let subsets: Vec<Value> = vec![
            Value::set(Vec::<Value>::new()),
            Value::set(vec![mv("wp11n1")]),
            Value::set(vec![mv("wp11n2"), Value::SmallInt(1)]),
            Value::set(vec![mv("wp11n1"), mv("wp11n2")]),
        ];
        let mut states = Vec::new();
        for a in &subsets {
            for b in &subsets {
                states.push(ArrayState::from_values(vec![func_of(vec![
                    (mv("wp11n1"), a.clone()),
                    (mv("wp11n2"), b.clone()),
                ])]));
            }
        }
        assert_all(&layout, &canon, &states);
    }

    // ---------- sequences: identity-only admission ----------

    #[cfg_attr(test, ntest::timeout(60000))]
    #[test]
    fn sequence_of_ints_admits_as_identity() {
        let registry = VarRegistry::from_names(["q"]);
        let layout = Arc::new(StateLayout::new(
            &registry,
            vec![VarLayoutKind::Recursive {
                layout: FlatValueLayout::Sequence {
                    bound: SequenceBoundEvidence::Observed,
                    max_len: 3,
                    element_layout: Box::new(FlatValueLayout::Scalar(SlotType::Int)),
                },
            }],
        ));
        let canon = FlatSymmetryCanonicalizer::compile(
            &layout,
            &[perm_func(&[("wp11a", "wp11b"), ("wp11b", "wp11a")])],
        )
        .expect("group-fixed sequence content must admit as identity");
        assert!(
            canon.actions().iter().all(FlatSymmetryAction::is_identity),
            "int sequences are fixed by model-value perms"
        );

        let seq = Value::Seq(Rp::new(vec![Value::SmallInt(3), Value::SmallInt(1)].into()));
        let state = ArrayState::from_values(vec![seq]);
        let buf = encode(&state, &layout);
        assert_eq!(canonical_of(&canon, &buf), buf, "identity group: no-op");
        assert_all(&layout, &canon, &[state]);
    }

    #[cfg_attr(test, ntest::timeout(60000))]
    #[test]
    fn sequence_of_model_values_declines() {
        let registry = VarRegistry::from_names(["q"]);
        let layout = Arc::new(StateLayout::new(
            &registry,
            vec![VarLayoutKind::Recursive {
                layout: FlatValueLayout::Sequence {
                    bound: SequenceBoundEvidence::ProvenInvariant {
                        invariant: Arc::from("LenBound"),
                    },
                    max_len: 3,
                    element_layout: Box::new(FlatValueLayout::Scalar(SlotType::ModelValue)),
                },
            }],
        ));
        assert!(
            FlatSymmetryCanonicalizer::compile(
                &layout,
                &[perm_func(&[("wp11a", "wp11b"), ("wp11b", "wp11a")])],
            )
            .is_none(),
            "permutable sequence content is data-dependent: must decline even with a proven bound"
        );
    }

    // ---------- decline matrix ----------

    fn single_var_layout(kind: VarLayoutKind) -> Arc<StateLayout> {
        let registry = VarRegistry::from_names(["x"]);
        Arc::new(StateLayout::new(&registry, vec![kind]))
    }

    fn swap_ab() -> Vec<FuncValue> {
        vec![perm_func(&[("wp11a", "wp11b"), ("wp11b", "wp11a")])]
    }

    #[cfg_attr(test, ntest::timeout(60000))]
    #[test]
    fn decline_dynamic_and_placeholder_bitmask() {
        assert!(
            FlatSymmetryCanonicalizer::compile(&single_var_layout(VarLayoutKind::Dynamic), &swap_ab())
                .is_none(),
            "Dynamic vars do not encode their value: decline"
        );
        assert!(
            FlatSymmetryCanonicalizer::compile(
                &single_var_layout(VarLayoutKind::Bitmask { universe_size: 4 }),
                &swap_ab()
            )
            .is_none(),
            "placeholder Bitmask kind has no equivariance proof: decline"
        );
    }

    #[cfg_attr(test, ntest::timeout(60000))]
    #[test]
    fn decline_unclosed_domains_and_universes() {
        // Domain {a, c} not closed under (a b).
        let unclosed_domain = single_var_layout(VarLayoutKind::StringKeyedArray {
            domain_keys: vec![Arc::from("wp11a"), Arc::from("wp11c")],
            domain_types: vec![SlotType::ModelValue; 2],
            value_types: vec![SlotType::Int; 2],
            range_encoding: StringKeyedArrayRangeEncoding::ScalarSlots,
        });
        assert!(
            FlatSymmetryCanonicalizer::compile(&unclosed_domain, &swap_ab()).is_none(),
            "model-value function domain not closed under the group: decline"
        );

        // FixedScalar universe {a, c} not closed under (a b).
        let proof = FixedScalarRangeProof::new(
            SlotType::ModelValue,
            vec![
                FlatScalarValue::ModelValue(Arc::from("wp11a")),
                FlatScalarValue::ModelValue(Arc::from("wp11c")),
            ],
            Arc::from("wp11-test"),
        )
        .expect("valid proof universe");
        let unclosed_universe = single_var_layout(VarLayoutKind::FixedScalar {
            base: SlotType::ModelValue,
            proof,
        });
        assert!(
            FlatSymmetryCanonicalizer::compile(&unclosed_universe, &swap_ab()).is_none(),
            "FixedScalar proof universe not closed under the group: decline"
        );

        // TaggedScalarUnion universe {Int 0, a} not closed under (a b).
        let union_proof = TaggedScalarUnionProof::new(
            vec![
                FlatScalarValue::Int(0),
                FlatScalarValue::ModelValue(Arc::from("wp11a")),
            ],
            Arc::from("wp11-test"),
        )
        .expect("valid union universe");
        let unclosed_union = single_var_layout(VarLayoutKind::TupleKeyedArray {
            domain_keys: vec![Value::Tuple(vec![Value::SmallInt(1)].into())],
            value_types: vec![SlotType::Int],
            range_encoding: TupleKeyedArrayRangeEncoding::TaggedScalarUnion(union_proof),
        });
        assert!(
            FlatSymmetryCanonicalizer::compile(&unclosed_union, &swap_ab()).is_none(),
            "TaggedScalarUnion universe not closed under the group: decline"
        );
    }

    #[cfg_attr(test, ntest::timeout(60000))]
    #[test]
    fn decline_sampled_bitmask_unless_fixed() {
        let sampled = |names: &[&str]| {
            single_var_layout(VarLayoutKind::Recursive {
                layout: FlatValueLayout::SetBitmask {
                    universe: names
                        .iter()
                        .map(|n| FlatScalarValue::ModelValue(Arc::from(*n)))
                        .collect(),
                    universe_closure: SetBitmaskUniverseClosure::Sampled,
                },
            })
        };
        assert!(
            FlatSymmetryCanonicalizer::compile(&sampled(&["wp11a", "wp11b"]), &swap_ab()).is_none(),
            "sampled bitmask universe touched by the group: decline"
        );
        let fixed = FlatSymmetryCanonicalizer::compile(&sampled(&["wp11x", "wp11y"]), &swap_ab())
            .expect("sampled universe disjoint from the group support: identity admit");
        assert!(fixed.actions().iter().all(FlatSymmetryAction::is_identity));
    }

    #[cfg_attr(test, ntest::timeout(60000))]
    #[test]
    fn decline_heterogeneous_value_types_across_orbit() {
        let layout = single_var_layout(VarLayoutKind::StringKeyedArray {
            domain_keys: vec![Arc::from("wp11a"), Arc::from("wp11b")],
            domain_types: vec![SlotType::ModelValue; 2],
            value_types: vec![SlotType::Int, SlotType::ModelValue],
            range_encoding: StringKeyedArrayRangeEncoding::ScalarSlots,
        });
        assert!(
            FlatSymmetryCanonicalizer::compile(&layout, &swap_ab()).is_none(),
            "a value moving between differently-typed slots cannot re-encode: decline"
        );
    }

    #[cfg_attr(test, ntest::timeout(60000))]
    #[test]
    fn tagged_union_admits_only_when_fixed() {
        let fixed_proof = TaggedUnionProof::new(
            vec![
                FlatValueLayout::Scalar(SlotType::Int),
                FlatValueLayout::Scalar(SlotType::Bool),
            ],
            Arc::from("wp11-test"),
        )
        .expect("valid tagged union");
        let fixed = single_var_layout(VarLayoutKind::Recursive {
            layout: FlatValueLayout::TaggedUnion { proof: fixed_proof },
        });
        let canon = FlatSymmetryCanonicalizer::compile(&fixed, &swap_ab())
            .expect("group-fixed tagged union admits as identity");
        assert!(canon.actions().iter().all(FlatSymmetryAction::is_identity));

        let touched_proof = TaggedUnionProof::new(
            vec![
                FlatValueLayout::Scalar(SlotType::Int),
                FlatValueLayout::Scalar(SlotType::ModelValue),
            ],
            Arc::from("wp11-test"),
        )
        .expect("valid tagged union");
        let touched = single_var_layout(VarLayoutKind::Recursive {
            layout: FlatValueLayout::TaggedUnion {
                proof: touched_proof,
            },
        });
        assert!(
            FlatSymmetryCanonicalizer::compile(&touched, &swap_ab()).is_none(),
            "tag/payload movement is data-dependent: decline when the group can touch a variant"
        );
    }

    #[cfg_attr(test, ntest::timeout(60000))]
    #[test]
    fn decline_invalid_declared_perms() {
        let layout = single_var_layout(VarLayoutKind::Scalar);
        // Not a bijection: a -> b with b unmapped (b would also stay fixed).
        let non_bijection = FuncValue::from_sorted_entries(vec![(mv("wp11a"), mv("wp11b"))]);
        assert!(FlatSymmetryCanonicalizer::compile(&layout, &[non_bijection]).is_none());
        // Non-model-value entries.
        let ints =
            FuncValue::from_sorted_entries(vec![(Value::SmallInt(1), Value::SmallInt(2))]);
        assert!(FlatSymmetryCanonicalizer::compile(&layout, &[ints]).is_none());
        // Empty / identity-only symmetry: nothing to exploit.
        assert!(FlatSymmetryCanonicalizer::compile(&layout, &[]).is_none());
        let identity = FuncValue::from_sorted_entries(vec![(mv("wp11a"), mv("wp11a"))]);
        assert!(FlatSymmetryCanonicalizer::compile(&layout, &[identity]).is_none());
    }

    // ---------- group closure ----------

    #[cfg_attr(test, ntest::timeout(60000))]
    #[test]
    fn generators_only_input_is_closed_internally() {
        let registry = VarRegistry::from_names(["m"]);
        let layout = Arc::new(StateLayout::new(
            &registry,
            vec![VarLayoutKind::ScalarModelValue],
        ));
        // A single 3-cycle generator: the closure must contain its square.
        let canon = FlatSymmetryCanonicalizer::compile(
            &layout,
            &[perm_func(&[
                ("wp11a", "wp11b"),
                ("wp11b", "wp11c"),
                ("wp11c", "wp11a"),
            ])],
        )
        .expect("cyclic group admits");
        assert_eq!(canon.group_order(), 3, "Z3 from a single generator");

        let states: Vec<ArrayState> = ["wp11a", "wp11b", "wp11c", "wp11d"]
            .iter()
            .map(|n| ArrayState::from_values(vec![mv(n)]))
            .collect();
        assert_all(&layout, &canon, &states);
    }

    #[cfg_attr(test, ntest::timeout(60000))]
    #[test]
    fn closure_cap_declines() {
        // S3 has 5 non-identity elements; a cap of 3 must decline.
        let gens = vec![
            NamePerm::from_func_value(&perm_func(&[("wp11a", "wp11b"), ("wp11b", "wp11a")]))
                .unwrap(),
            NamePerm::from_func_value(&perm_func(&[("wp11b", "wp11c"), ("wp11c", "wp11b")]))
                .unwrap(),
        ];
        assert!(close_group(gens.clone(), 3).is_none());
        let full = close_group(gens, 5).expect("S3 closure fits a cap of 5");
        assert_eq!(full.len(), 5);
    }

    // ---------- mixed layout, S3 group, exhaustive orbit parity ----------

    #[cfg_attr(test, ntest::timeout(120000))]
    #[test]
    fn mixed_layout_s3_orbit_partition_parity() {
        let registry = VarRegistry::from_names(["m1", "m2", "arr"]);
        let layout = Arc::new(StateLayout::new(
            &registry,
            vec![
                VarLayoutKind::ScalarModelValue,
                VarLayoutKind::ScalarModelValue,
                VarLayoutKind::IntArray {
                    lo: 0,
                    len: 2,
                    elements_are_bool: false,
                    element_types: Some(vec![SlotType::ModelValue, SlotType::Int]),
                    element_range_proof: None,
                },
            ],
        ));
        let canon = FlatSymmetryCanonicalizer::compile(
            &layout,
            &[
                perm_func(&[("wp11a", "wp11b"), ("wp11b", "wp11a")]),
                perm_func(&[("wp11b", "wp11c"), ("wp11c", "wp11b")]),
            ],
        )
        .expect("payload-only S3 layout admits");
        assert_eq!(canon.group_order(), 6, "S3");

        let names = ["wp11a", "wp11b", "wp11c"];
        let mut states = Vec::new();
        for m1 in names {
            for m2 in names {
                for e0 in names {
                    for i in [0i64, 1] {
                        states.push(ArrayState::from_values(vec![
                            mv(m1),
                            mv(m2),
                            func_of(vec![
                                (Value::SmallInt(0), mv(e0)),
                                (Value::SmallInt(1), Value::SmallInt(i)),
                            ]),
                        ]));
                    }
                }
            }
        }
        assert_equivariant(&layout, &canon, &states);
        assert_canonical_sound(&layout, &canon, &states[..12]);
        assert_orbit_partition_matches_interpreter(&layout, &canon, &states);
    }

    // ---------- TaggedScalarOrSet range ----------

    #[cfg_attr(test, ntest::timeout(60000))]
    #[test]
    fn tagged_scalar_or_set_range_equivariance() {
        let proof = TaggedScalarSetRangeProof::new(
            SlotType::ModelValue,
            vec![
                FlatScalarValue::ModelValue(Arc::from("wp11a")),
                FlatScalarValue::ModelValue(Arc::from("wp11b")),
            ],
            Arc::from("wp11-test"),
        )
        .expect("valid tagged scalar/set proof");
        let layout = single_var_layout(VarLayoutKind::StringKeyedArray {
            domain_keys: vec![Arc::from("wp11a"), Arc::from("wp11b")],
            domain_types: vec![SlotType::ModelValue; 2],
            value_types: vec![SlotType::ModelValue; 2],
            range_encoding: StringKeyedArrayRangeEncoding::TaggedScalarOrSet(proof),
        });
        let canon = FlatSymmetryCanonicalizer::compile(&layout, &swap_ab())
            .expect("closed scalar/set range must admit");

        let range = [
            mv("wp11a"),
            mv("wp11b"),
            Value::set(Vec::<Value>::new()),
            Value::set(vec![mv("wp11a")]),
            Value::set(vec![mv("wp11a"), mv("wp11b")]),
        ];
        let mut states = Vec::new();
        for a in &range {
            for b in &range {
                states.push(ArrayState::from_values(vec![func_of(vec![
                    (mv("wp11a"), a.clone()),
                    (mv("wp11b"), b.clone()),
                ])]));
            }
        }
        assert_all(&layout, &canon, &states);
    }

    // ---------- canonicalize misc ----------

    #[cfg_attr(test, ntest::timeout(60000))]
    #[test]
    fn canonicalize_never_touches_mismatched_buffer() {
        let registry = VarRegistry::from_names(["m"]);
        let layout = Arc::new(StateLayout::new(
            &registry,
            vec![VarLayoutKind::ScalarModelValue],
        ));
        let canon = FlatSymmetryCanonicalizer::compile(&layout, &swap_ab()).expect("admits");
        let mut wrong_width = vec![1i64, 2, 3];
        let before = wrong_width.clone();
        let mut scratch = Vec::new();
        // Release-mode fail-safe (debug builds assert): buffer stays intact.
        if !cfg!(debug_assertions) {
            canon.canonicalize_in_place(&mut wrong_width, &mut scratch);
            assert_eq!(wrong_width, before);
        }
    }
}
