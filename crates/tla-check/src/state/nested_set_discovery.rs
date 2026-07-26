// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Dynamic-universe DISCOVERY for set-of-sets (nested-set) state variables —
//! SHADOW / LOG-ONLY (nested-set discovery A4).
//!
//! # What this does
//!
//! A set-of-sets state variable (the canonical example is the `SlidingPuzzles`
//! `board`, a `SUBSET (SUBSET Pos)`) cannot be flat-encoded from INIT alone:
//! the reachable piece-shapes only appear as the board slides. This module
//! *samples reachable inner element-shapes across successors* and derives a
//! two-level [`FlatValueLayout::NestedSetBitmask`] universe:
//!
//! * `inner_universe` — the distinct inner scalar element-ids (≤ 64) shared by
//!   every inner set, canonical (sorted + deduped); and
//! * `outer_universe` — the distinct inner-set bitmasks / *piece shapes* (≤ the
//!   multi-slot cap), canonical (sorted + deduped).
//!
//! The discovered universe carries the
//! [`SetBitmaskUniverseClosure::DynamicallyDiscovered`] closure with
//! `monitor_enforced: false` — it is a *sampled* universe, NOT proven closed,
//! so it stays fail-closed for every flat-primary admission exactly like
//! `Sampled` (see `SetBitmaskUniverseClosure::is_proven_closed`).
//!
//! # SHADOW invariant (A4 — behavior-neutral)
//!
//! This module is pure and side-effect-free: it COMPUTES the would-be universe
//! and VALIDATES that sampled boards round-trip through the A3 codec, but it
//! NEVER substitutes a layout. The caller (the BFS prefix in
//! `run_bfs_full.rs`) wires it behind the `TY_NESTED_SET_DISCOVERY=1` env gate
//! and only LOGS the result — the variable keeps its real (`Dynamic`) layout,
//! so the run is byte-identical with and without the discovery. Promotion +
//! the per-successor out-of-universe monitor are A5.
//!
//! # The scalar-canonicalization that makes the A3 codec apply
//!
//! The A3 codec's `inner_universe` is `Vec<FlatScalarValue>` and its inner-set
//! fold (`inner_set_value_to_mask`) compares each inner element against
//! `flat_scalar_to_value(candidate)`. `SlidingPuzzles` inner elements are
//! *positions* (`Value::Tuple<<x, y>>`), which are NOT `FlatScalarValue`. So
//! the discovery assigns each distinct inner-element value a canonical scalar
//! id (`FlatScalarValue::Int(idx)` over the sorted distinct inner elements) and
//! validates the round-trip against the *id-canonicalized* board (positions →
//! scalar ids). This proves the combinatorial structure (|inner|, |outer|, no
//! escape) on real data; the A5 monitor would carry the same position↔id
//! bijection at its encode hook.

use std::collections::BTreeMap;
use tla_value::Rp;

use tla_value::value::SortedSet;

use super::flat_state::{
    record_set_bitmask_slot_count, try_reconstruct_flat_value, try_write_flat_value_slots,
    value_fits_flat_value_layout, MAX_NESTED_SET_INNER_UNIVERSE, MAX_RECORD_SET_BITMASK_UNIVERSE,
};
use super::state_layout::{FlatScalarValue, FlatValueLayout, SetBitmaskUniverseClosure};
use super::value_hash::value_fingerprint;
use super::value_hash_additive::{splitmix64, ADDITIVE_SET_SEED};
use crate::Value;

/// A discovered nested-set universe + the id↔value bijection used to
/// canonicalize tuple/record inner elements into the scalar `inner_universe`.
///
/// SHADOW: this is a *candidate* layout. It is never assigned as a variable's
/// real layout in A4 — it is only logged and round-trip-validated.
#[derive(Debug, Clone)]
pub(crate) struct DiscoveredNestedSet {
    /// The would-be flat layout (`NestedSetBitmask`) — never promoted in A4.
    pub(crate) layout: FlatValueLayout,
    /// Distinct inner-element values, in the canonical order that defines each
    /// element's `inner_universe` scalar id. `inner_id_of[i]` ↔
    /// `inner_universe[i]` ↔ `FlatScalarValue::Int(i)`.
    pub(crate) inner_elements: Vec<Value>,
    /// Number of distinct piece-shapes (= `outer_universe` len).
    pub(crate) outer_len: usize,
    /// Number of distinct inner element-ids (= `inner_universe` len).
    pub(crate) inner_len: usize,
}

/// Round-trip validation report for the discovered universe over sampled
/// boards. The escape count is the feasibility signal: a converged discovery
/// has zero escapes (every reachable board encodes).
#[derive(Debug, Clone, Default)]
pub(crate) struct NestedSetValidationReport {
    /// Boards validated (encoded + decoded byte-exact).
    pub(crate) sampled_boards: usize,
    /// Boards that round-tripped byte-exact through the A3 codec.
    pub(crate) roundtrip_ok: usize,
    /// Boards that did NOT fit the discovered universe (a shape outside the
    /// sampled inner/outer universe) — an *escape*.
    pub(crate) escapes: usize,
}

/// True when this variable value is a *set-of-sets* (nested set): a
/// `Value::Set` whose every element is itself a `Value::Set`. An empty outer
/// set is not enough signal to discover a universe, so it is rejected.
#[must_use]
pub(crate) fn is_nested_set_value(value: &Value) -> bool {
    let Value::Set(outer) = value else {
        return false;
    };
    !outer.is_empty() && outer.iter().all(|piece| matches!(piece, Value::Set(_)))
}

/// Derive a [`DiscoveredNestedSet`] from a sample of board values (one
/// set-of-sets value per sampled successor for a single variable).
///
/// Two-level derivation mirroring the A3 `nested_layout` test helper but driven
/// by *sampled data* rather than a static type:
///
/// 1. Collect the distinct inner-element values across every piece of every
///    sampled board; sort + dedup them into `inner_elements`. Each inner
///    element's index in this canonical order is its scalar id, and
///    `inner_universe = [Int(0), .., Int(n-1)]`.
/// 2. Fold each piece into a `u64` inner-mask over those ids; collect the
///    distinct masks; sort + dedup into `outer_universe`.
///
/// Returns `None` (fail-closed) when there is no nested-set sample, when the
/// inner universe exceeds the single-word cap (`> 64`), when the outer universe
/// exceeds the multi-slot cap, or when any inner element is non-canonical (the
/// caps mirror the A3 codec's so a discovered universe is always encodable).
#[must_use]
pub(crate) fn derive_nested_set_universe(samples: &[&Value]) -> Option<DiscoveredNestedSet> {
    // (1) Collect distinct inner-element values across all pieces of all boards.
    // BTreeMap keeps a single canonical insertion-independent order via `Value`
    // ordering, matching the SortedSet canonical order the codec reconstructs.
    let mut inner_set: std::collections::BTreeSet<Value> = std::collections::BTreeSet::new();
    let mut saw_nested = false;
    for &board in samples {
        let Value::Set(outer) = board else { continue };
        if outer.is_empty() || !outer.iter().all(|p| matches!(p, Value::Set(_))) {
            continue;
        }
        saw_nested = true;
        for piece in outer.iter() {
            let Value::Set(inner) = piece else { continue };
            for elem in inner.iter() {
                inner_set.insert(elem.clone());
            }
        }
    }
    if !saw_nested {
        return None;
    }

    let inner_elements: Vec<Value> = inner_set.into_iter().collect();
    if inner_elements.len() > MAX_NESTED_SET_INNER_UNIVERSE {
        // Inner universe too large to fold into a single u64 — fail closed.
        return None;
    }
    // `inner_universe[i] == Int(i)`: the scalar id space the codec encodes/decodes.
    let inner_universe: Vec<FlatScalarValue> = (0..inner_elements.len())
        .map(|i| FlatScalarValue::Int(i as i64))
        .collect();
    // Value -> scalar id, for canonicalizing boards on the validation path.
    let id_of: BTreeMap<Value, usize> = inner_elements
        .iter()
        .enumerate()
        .map(|(i, v)| (v.clone(), i))
        .collect();

    // (2) Fold every piece into its u64 inner-mask over the id space; collect
    // the distinct masks into the outer universe.
    let mut outer_set: std::collections::BTreeSet<u64> = std::collections::BTreeSet::new();
    for &board in samples {
        let Value::Set(outer) = board else { continue };
        for piece in outer.iter() {
            let Value::Set(inner) = piece else { continue };
            let mut mask = 0u64;
            for elem in inner.iter() {
                let &id = id_of.get(elem)?;
                mask |= 1u64 << id;
            }
            outer_set.insert(mask);
        }
    }
    let outer_universe: Vec<u64> = outer_set.into_iter().collect();
    if outer_universe.is_empty() || outer_universe.len() > MAX_RECORD_SET_BITMASK_UNIVERSE {
        return None;
    }
    // The multi-slot bitmask must resolve a finite slot count.
    record_set_bitmask_slot_count(outer_universe.len())?;

    let inner_len = inner_universe.len();
    let outer_len = outer_universe.len();
    let layout = FlatValueLayout::NestedSetBitmask {
        outer_universe,
        inner_universe,
        // SHADOW: a sampled universe, NOT proven closed. `monitor_enforced:
        // false` records that the A5 per-successor out-of-universe monitor is
        // not installed yet; `is_proven_closed()` stays false either way, so
        // this layout could never be admitted as a flat-primary slot even if it
        // were (wrongly) substituted.
        outer_closure: SetBitmaskUniverseClosure::DynamicallyDiscovered {
            monitor_enforced: false,
        },
        inner_closure: SetBitmaskUniverseClosure::DynamicallyDiscovered {
            monitor_enforced: false,
        },
    };

    Some(DiscoveredNestedSet {
        layout,
        inner_elements,
        outer_len,
        inner_len,
    })
}

/// STATIC (exploration-free) universe derivation for the rigid-slide nested-set
/// class — the "cheap discovery" that replaces the full sampling BFS for
/// `SlidingPuzzles`.
///
/// # When this is sound + complete
///
/// The caller must have a `slide_recognize` PROOF that the spec's `Next` is the
/// rigid-unit-slide relation over the position grid `positions` (the same proof
/// the DEFAULT slide-kernel arm requires). Under that proof every reachable
/// board's pieces are rigid TRANSLATES of the INIT pieces: the slide relation
/// only translates a piece by a unit vector (keeping it iff it stays in the grid
/// and disjoint), so a piece is never reshaped, only moved. Therefore the
/// COMPLETE set of piece-shapes that can EVER appear in any reachable board is
/// exactly
///
/// ```text
///   { all grid-fitting translates of every INIT piece }
/// ```
///
/// This function enumerates that set directly (no BFS, no state retention) and
/// reuses [`derive_nested_set_universe`] on a synthetic board that carries every
/// translate as one piece, so the resulting `NestedSetBitmask` universe +
/// bijection + caps are byte-identical in shape to the sampled path — only the
/// COST differs (O(#pieces × grid) vs a full reachable-space re-exploration).
///
/// # Why 0 escapes (completeness)
///
/// The enumerated universe is a proven SUPERSET of every reachable piece-shape,
/// so no reachable board can carry a piece outside it — the per-successor
/// monitor (which still guards every board) sees 0 escapes. The extra shapes a
/// static superset may include over the exact reachable set never appear as a
/// present piece, so the monitored dedup fingerprint (a sum over PRESENT pieces
/// only) is byte-identical either way — the verdict is unchanged. The monitor
/// remains the fail-closed backstop: if the rigid-slide assumption were ever
/// violated for some board, that board escapes and the var bails to raw
/// `value_fingerprint` (never a wrong dedup).
///
/// Returns `None` (caller falls back to the sampling path) when there is no
/// piece to enumerate, the grid is empty, or the derived universe exceeds the
/// A3 codec caps.
#[must_use]
pub(crate) fn derive_nested_set_universe_static(
    positions: &[(i64, i64)],
    init_boards: &[&Value],
) -> Option<DiscoveredNestedSet> {
    use std::collections::BTreeSet;

    // The exact grid membership set (handles non-rectangular grids with holes:
    // a translate is kept ONLY when every cell is a real grid position).
    let pos_set: BTreeSet<(i64, i64)> = positions.iter().copied().collect();
    let min_x = pos_set.iter().map(|&(x, _)| x).min()?;
    let max_x = pos_set.iter().map(|&(x, _)| x).max()?;
    let min_y = pos_set.iter().map(|&(_, y)| y).min()?;
    let max_y = pos_set.iter().map(|&(_, y)| y).max()?;

    // Every distinct grid-fitting translate of every INIT piece, canonicalized
    // as a sorted cell list so equal shapes-at-a-position collapse.
    let mut translates: BTreeSet<Vec<(i64, i64)>> = BTreeSet::new();
    let mut saw_piece = false;
    for &board in init_boards {
        let Value::Set(outer) = board else { continue };
        for piece in outer.iter() {
            let Value::Set(inner) = piece else {
                return None;
            };
            let mut cells: Vec<(i64, i64)> = Vec::with_capacity(inner.len());
            for elem in inner.iter() {
                cells.push(super::nested_set_slide::value_to_pos(elem)?);
            }
            if cells.is_empty() {
                continue;
            }
            saw_piece = true;
            let pmin_x = cells.iter().map(|&(x, _)| x).min()?;
            let pmax_x = cells.iter().map(|&(x, _)| x).max()?;
            let pmin_y = cells.iter().map(|&(_, y)| y).min()?;
            let pmax_y = cells.iter().map(|&(_, y)| y).max()?;
            // Translation offsets that keep the piece's bounding box within the
            // grid's bounding box; the per-cell membership test below then
            // rejects any translate that lands on a grid hole.
            for dx in (min_x - pmin_x)..=(max_x - pmax_x) {
                for dy in (min_y - pmin_y)..=(max_y - pmax_y) {
                    if cells
                        .iter()
                        .all(|&(x, y)| pos_set.contains(&(x + dx, y + dy)))
                    {
                        let mut t: Vec<(i64, i64)> =
                            cells.iter().map(|&(x, y)| (x + dx, y + dy)).collect();
                        t.sort_unstable();
                        translates.insert(t);
                    }
                }
            }
        }
    }
    if !saw_piece || translates.is_empty() {
        return None;
    }

    // One synthetic board carrying every translate as a piece, then reuse the
    // sampled derivation (which builds the id bijection + applies the codec
    // caps). A Value::Set holds each distinct translate as a distinct element.
    let pieces = translates.into_iter().map(|cells| {
        Value::Set(Rp::new(SortedSet::from_iter(cells.into_iter().map(
            |(x, y)| Value::Tuple(Rp::from(vec![Value::SmallInt(x), Value::SmallInt(y)])),
        ))))
    });
    let synthetic = Value::Set(Rp::new(SortedSet::from_iter(pieces)));
    derive_nested_set_universe(&[&synthetic])
}

/// Canonicalize a board value (set-of-sets of arbitrary inner elements) into
/// the scalar-id form the A3 codec expects: each inner element is replaced by
/// `Value::SmallInt(id)` where `id` is its index in `inner_elements`. Returns
/// `None` if a board is not nested or an inner element is outside the
/// discovered inner universe (an *escape*).
fn canonicalize_board(board: &Value, id_of: &BTreeMap<Value, usize>) -> Option<Value> {
    let Value::Set(outer) = board else {
        return None;
    };
    let mut pieces: Vec<Value> = Vec::with_capacity(outer.len());
    for piece in outer.iter() {
        let Value::Set(inner) = piece else {
            return None;
        };
        let mut ids: Vec<Value> = Vec::with_capacity(inner.len());
        for elem in inner.iter() {
            let &id = id_of.get(elem)?;
            ids.push(Value::SmallInt(id as i64));
        }
        pieces.push(Value::Set(Rp::new(SortedSet::from_iter(ids))));
    }
    Some(Value::Set(Rp::new(SortedSet::from_iter(pieces))))
}

/// Validate that every sampled board round-trips byte-exact through the A3
/// codec against the discovered universe, counting escapes.
///
/// For each board: canonicalize positions → scalar ids, check it `value_fits`
/// the layout (the A3 escape gate), then ENCODE via `try_write_flat_value_slots`
/// and DECODE via `try_reconstruct_flat_value`, asserting the decoded value is
/// identical to the canonicalized board. An escape (a board with a shape
/// outside the sampled universe) is counted, not panicked — this is the
/// feasibility signal A5 will turn into a re-derive / freeze policy.
#[must_use]
pub(crate) fn validate_roundtrip(
    discovered: &DiscoveredNestedSet,
    samples: &[&Value],
) -> NestedSetValidationReport {
    let id_of: BTreeMap<Value, usize> = discovered
        .inner_elements
        .iter()
        .enumerate()
        .map(|(i, v)| (v.clone(), i))
        .collect();
    let slot_count = discovered.layout.slot_count();
    let mut report = NestedSetValidationReport::default();

    for &board in samples {
        if !is_nested_set_value(board) {
            continue;
        }
        report.sampled_boards += 1;
        let Some(canon) = canonicalize_board(board, &id_of) else {
            // An inner element outside the discovered inner universe.
            report.escapes += 1;
            continue;
        };
        // A3 escape gate: a piece-shape outside the outer universe (or an inner
        // element outside the inner universe) does not fit.
        if !value_fits_flat_value_layout(&canon, &discovered.layout) {
            report.escapes += 1;
            continue;
        }
        let mut slots = vec![0i64; slot_count];
        if try_write_flat_value_slots(&canon, &discovered.layout, &mut slots).is_err() {
            report.escapes += 1;
            continue;
        }
        match try_reconstruct_flat_value(&discovered.layout, &slots) {
            Ok(restored) if restored == canon => report.roundtrip_ok += 1,
            // A canonical value that fits MUST round-trip; a mismatch/err here
            // is a codec divergence, surfaced as an escape (never silent).
            _ => report.escapes += 1,
        }
    }
    report
}

/// A FROZEN per-variable nested-set layout plus the per-successor escape
/// monitor (nested-set discovery A5) — the SOUNDNESS GATE.
///
/// Built once, after the discovery prefix converges, by [`freeze_nested_set_var`].
/// It carries the frozen universe (`layout`), the position↔id bijection
/// (`inner_elements` / `id_of`), and the precomputed per-id element fingerprints
/// (`inner_elem_fps`) so the hot dedup path never re-walks the inner element
/// values.
///
/// # The soundness mechanism (fail-closed)
///
/// On EVERY successor board (no exceptions), [`Self::encode_board`] is called on
/// the hot dedup path. It uses ONLY the Option-returning lossless encode
/// (`value_fits_flat_value_layout` + `try_write_flat_value_slots` — never the
/// panicking `write_*` / `from_array_state`). On any escape — a piece-shape
/// outside `outer_universe`, an inner element outside `inner_universe`, or any
/// codec error — it returns [`NestedSetEncodeOutcome::Escaped`] and the caller
/// flips [`Self::bailed`] (the whole variable falls back to the interpreter's
/// raw `value_fingerprint` for the rest of the run). NO board is ever silently
/// mis-encoded: two distinct boards can never alias to one mask because (a) the
/// codec is a bijection on the frozen universe and (b) any board that does NOT
/// fit the universe escapes rather than truncating into some other board's mask.
///
/// # Why the verdict is byte-identical
///
/// The monitored dedup fingerprint ([`Self::board_dedup_fp`]) is computed to
/// byte-exactly equal `value_fingerprint(original_board)` — the SAME fingerprint
/// the un-promoted interpreter path produces. The frozen `inner_elements`
/// bijection is applied INSIDE the fingerprint fold (each present inner bit
/// contributes `value_fingerprint(inner_elements[bit])`, the real position
/// tuple, NOT the scalar id), so an in-universe board fingerprints identically
/// whether routed through the monitored mask path or the raw value path. This
/// means the dedup domain is single and consistent even when the monitor bails
/// mid-run: an escaped board falls back to `value_fingerprint(original_board)`,
/// which is exactly what an in-universe board would have produced too. No domain
/// split, no aliasing — the run is identical to the baseline plus a fail-closed
/// gate.
#[derive(Debug, Clone)]
pub(crate) struct NestedSetVarMonitor {
    /// State-variable index this monitor guards.
    pub(crate) var_idx: usize,
    /// Frozen `NestedSetBitmask` layout (`monitor_enforced: true`).
    pub(crate) layout: FlatValueLayout,
    /// Frozen inner-element bijection: `inner_elements[id]` is the real inner
    /// `Value` (e.g. a position tuple) for scalar id `id`.
    #[allow(dead_code)]
    // retained for the id→Value decode direction (inverse of `id_of`); not yet wired
    pub(crate) inner_elements: Vec<Value>,
    /// Inverse bijection: real inner `Value` → scalar id. Used to canonicalize a
    /// board (positions → ids) before the codec, and to detect escapes.
    id_of: BTreeMap<Value, usize>,
    /// Precomputed `value_fingerprint(inner_elements[id])` per id, so the hot
    /// dedup fold never re-fingerprints inner element values.
    inner_elem_fps: Vec<u64>,
    /// Number of slots the frozen layout occupies (always 1 for SlidingPuzzles).
    slot_count: usize,
    /// FAIL-CLOSED latch: set on the first escape. Once latched, the variable
    /// permanently falls back to the interpreter's raw `value_fingerprint`. The
    /// fallback is sound (same fingerprint domain) — see the struct docs.
    pub(crate) bailed: bool,
    /// Diagnostics: boards encoded via the monitored mask path.
    pub(crate) encoded_count: u64,
    /// Diagnostics: boards that escaped the frozen universe (≥1 ⇒ `bailed`).
    pub(crate) escape_count: u64,
}

/// Outcome of monitoring + encoding a single board.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum NestedSetEncodeOutcome {
    /// Board fit the frozen universe; carries the dedup fingerprint
    /// (byte-identical to `value_fingerprint(board)`) and the compact mask slots.
    Encoded { dedup_fp: u64, slots: Vec<i64> },
    /// Board escaped the frozen universe (fail-closed). The caller bails the var.
    Escaped,
}

impl NestedSetVarMonitor {
    /// Number of frozen slots (compact storage width for one board).
    #[must_use]
    pub(crate) fn slot_count(&self) -> usize {
        self.slot_count
    }

    /// Canonicalize a board (inner positions → scalar ids) for the codec, OR
    /// return `None` (an escape) when any inner element is outside the frozen
    /// inner universe.
    fn canonicalize(&self, board: &Value) -> Option<Value> {
        canonicalize_board(board, &self.id_of)
    }

    /// THE PER-SUCCESSOR MONITOR + ENCODE (soundness-critical hot path).
    ///
    /// Returns [`NestedSetEncodeOutcome::Encoded`] iff the board fits the frozen
    /// universe and losslessly encodes; otherwise [`NestedSetEncodeOutcome::Escaped`]
    /// (fail-closed). Uses ONLY the Option/`try_*` lossless encode — never panics.
    ///
    /// The returned `dedup_fp` byte-exactly equals `value_fingerprint(board)`
    /// (the bijection is applied in [`Self::board_dedup_fp`]), so an Encoded
    /// board and the same board hashed via the raw value path are interchangeable.
    #[must_use]
    pub(crate) fn encode_board(&self, board: &Value) -> NestedSetEncodeOutcome {
        // Escape gate 1: must be a nested set at all.
        if !is_nested_set_value(board) {
            return NestedSetEncodeOutcome::Escaped;
        }
        // Escape gate 2: canonicalize positions → ids; any inner element outside
        // the frozen inner universe escapes.
        let Some(canon) = self.canonicalize(board) else {
            return NestedSetEncodeOutcome::Escaped;
        };
        // Escape gate 3 (A3 fit): a piece-shape outside the outer universe (or an
        // inner element outside the inner universe) does not fit — fail closed.
        if !value_fits_flat_value_layout(&canon, &self.layout) {
            return NestedSetEncodeOutcome::Escaped;
        }
        // Lossless encode via the Option/Result path ONLY (never write_*/panic).
        let mut slots = vec![0i64; self.slot_count];
        if try_write_flat_value_slots(&canon, &self.layout, &mut slots).is_err() {
            return NestedSetEncodeOutcome::Escaped;
        }
        // Dedup fingerprint computed DIRECTLY from the mask + frozen bijection,
        // byte-matching value_fingerprint(original board).
        let Some(dedup_fp) = self.board_dedup_fp(&slots) else {
            return NestedSetEncodeOutcome::Escaped;
        };
        NestedSetEncodeOutcome::Encoded { dedup_fp, slots }
    }

    /// THE PER-SUCCESSOR MONITOR (escape-only fast path — nested-set A6).
    ///
    /// Runs ONLY the three escape gates of [`Self::encode_board`] (nested-set,
    /// inner-element-in-universe, A3-fit + lossless-write) WITHOUT computing the
    /// dedup fingerprint. Used on the DIFF / streaming fingerprint path, where the
    /// successor's fingerprint is already computed losslessly by
    /// `compute_diff_fingerprint_with_xor` (the board's contribution is
    /// `value_fingerprint(board)`, which byte-matches the monitored `dedup_fp` —
    /// so the fingerprint NEVER depends on the monitor). The monitor's remaining
    /// job on that path is purely to OBSERVE every successor board and FAIL CLOSED
    /// on escape, exactly as the full-state batch path does.
    ///
    /// Returns `true` iff the board fit the frozen universe (encoded), `false` on
    /// escape. Side effects mirror the batch hook: increments `encoded_count` on
    /// fit, increments `escape_count` and latches `bailed` on escape. Idempotent
    /// once `bailed` (a bailed var has permanently fallen back to raw
    /// `value_fingerprint`, the same fp the diff path already produces).
    pub(crate) fn observe_board_escape_only(&mut self, board: &Value) -> bool {
        if self.bailed {
            // Already failed closed: the diff path's raw value_fingerprint is the
            // var's fingerprint for the rest of the run (same domain). Nothing to
            // re-check — re-observing cannot un-bail.
            return false;
        }
        // Escape gate 1: must be a nested set at all.
        // Escape gate 2: canonicalize positions → ids (inner-universe membership).
        // Escape gate 3 (A3 fit): piece-shape in the outer universe + lossless write.
        let encoded = is_nested_set_value(board)
            && self.canonicalize(board).is_some_and(|canon| {
                if !value_fits_flat_value_layout(&canon, &self.layout) {
                    return false;
                }
                let mut slots = vec![0i64; self.slot_count];
                try_write_flat_value_slots(&canon, &self.layout, &mut slots).is_ok()
            });
        if encoded {
            self.encoded_count += 1;
        } else {
            self.escape_count += 1;
            self.bailed = true;
        }
        encoded
    }

    /// Compute the board dedup fingerprint DIRECTLY from the compact mask slots
    /// using the frozen `inner_elements` bijection, byte-matching
    /// `value_fingerprint(original_board)`.
    ///
    /// The original board is `Value::Set` of `Value::Set` (pieces of positions),
    /// so the canonical additive scheme nests:
    ///   board_fp = ADDITIVE_SET_SEED + splitmix64(#present_pieces)
    ///              + Σ_{present outer i} splitmix64(piece_fp_i)
    ///   piece_fp_i = ADDITIVE_SET_SEED + splitmix64(#inner bits)
    ///              + Σ_{inner bit j set} splitmix64(value_fingerprint(inner_elements[j]))
    /// This is exactly `compute_set_additive_fp` at both tiers, with the inner
    /// element fp being the REAL position tuple's fingerprint (via the frozen
    /// bijection), NOT the scalar id — so it equals `value_fingerprint(board)`.
    ///
    /// Returns `None` only on a non-canonical mask (a bit outside the valid
    /// range), which is itself an escape.
    fn board_dedup_fp(&self, slots: &[i64]) -> Option<u64> {
        let FlatValueLayout::NestedSetBitmask {
            outer_universe,
            inner_universe,
            ..
        } = &self.layout
        else {
            return None;
        };
        let slot_count = record_set_bitmask_slot_count(outer_universe.len())?;
        if slots.len() != slot_count {
            return None;
        }
        // inner_elem_fps is indexed by inner-universe bit (== id); guard length.
        if self.inner_elem_fps.len() != inner_universe.len() {
            return None;
        }
        // Per-slot canonical validation + outer popcount.
        let mut outer_count = 0u64;
        for (slot_index, &raw) in slots.iter().enumerate().take(slot_count) {
            let valid = super::flat_state::record_set_bitmask_slot_valid_mask(
                outer_universe.len(),
                slot_index,
            )?;
            let word = raw as u64;
            if (word & !valid) != 0 {
                return None;
            }
            outer_count += (word & valid).count_ones() as u64;
        }
        let inner_set_fp = |inner_mask: u64| -> u64 {
            let mut ifp = ADDITIVE_SET_SEED;
            ifp = ifp.wrapping_add(splitmix64(inner_mask.count_ones() as u64));
            for bit in 0..inner_universe.len() {
                if (inner_mask & (1u64 << bit)) != 0 {
                    // Bijection-aware: fingerprint the REAL inner element value
                    // (position tuple), not the scalar id.
                    ifp = ifp.wrapping_add(splitmix64(self.inner_elem_fps[bit]));
                }
            }
            ifp
        };
        let mut fp = ADDITIVE_SET_SEED;
        fp = fp.wrapping_add(splitmix64(outer_count));
        for (index, inner_mask) in outer_universe.iter().enumerate() {
            let slot = slots[index / 64] as u64;
            if (slot & (1u64 << (index % 64))) != 0 {
                fp = fp.wrapping_add(splitmix64(inner_set_fp(*inner_mask)));
            }
        }
        Some(fp)
    }
}

/// FREEZE (nested-set discovery A5): build a [`NestedSetVarMonitor`] for a state
/// variable from a DISCOVERED universe.
///
/// Promotes the discovered `NestedSetBitmask` layout to `monitor_enforced: true`
/// (both tiers) — the closure that records the per-successor monitor is now
/// installed — and snapshots the bijection (`inner_elements` / `id_of`) alongside
/// the universe so decode (id → position) and the dedup fingerprint can recover
/// the real board. Precomputes the per-id inner-element fingerprints.
///
/// Returns `None` (fail-closed) when the discovered layout is not a
/// `NestedSetBitmask` (it always is here) or the inner universe length is
/// inconsistent — neither happens for a converged discovery, but the var simply
/// stays Dynamic if so.
#[must_use]
pub(crate) fn freeze_nested_set_var(
    var_idx: usize,
    discovered: &DiscoveredNestedSet,
) -> Option<NestedSetVarMonitor> {
    let FlatValueLayout::NestedSetBitmask {
        outer_universe,
        inner_universe,
        ..
    } = &discovered.layout
    else {
        return None;
    };
    if inner_universe.len() != discovered.inner_elements.len() {
        return None;
    }

    // FORCED-ESCAPE TEST HOOK (soundness gate proof): when
    // `TY_NESTED_SET_FORCE_SHRINK=N` is set, the frozen outer universe is
    // truncated to its first `N` piece-shapes. Every reachable board carrying a
    // dropped piece-shape then ESCAPES the (artificially incomplete) frozen
    // universe, exercising the per-successor monitor's fail-closed path on the
    // REAL run. The run must still produce the correct state count (the monitor
    // bails the var to the interpreter's raw `value_fingerprint`, same domain),
    // proving the monitor never undercounts. No-op (full universe) in normal
    // runs and when the var is unset.
    let outer_universe: Vec<u64> = match std::env::var("TY_NESTED_SET_FORCE_SHRINK")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
    {
        Some(n) if n < outer_universe.len() => {
            eprintln!(
                "[nested-set] A5 FORCED-ESCAPE TEST: shrinking frozen outer universe {} -> {} \
                 (boards using dropped piece-shapes will escape + fail closed)",
                outer_universe.len(),
                n,
            );
            outer_universe.iter().copied().take(n).collect()
        }
        _ => outer_universe.clone(),
    };
    let outer_universe = &outer_universe;

    let slot_count = record_set_bitmask_slot_count(outer_universe.len())?;
    // Promote the closure to monitor_enforced: true. The universe + caps are
    // unchanged from discovery; only the provenance flag flips.
    let layout = FlatValueLayout::NestedSetBitmask {
        outer_universe: outer_universe.clone(),
        inner_universe: inner_universe.clone(),
        outer_closure: SetBitmaskUniverseClosure::DynamicallyDiscovered {
            monitor_enforced: true,
        },
        inner_closure: SetBitmaskUniverseClosure::DynamicallyDiscovered {
            monitor_enforced: true,
        },
    };
    let id_of: BTreeMap<Value, usize> = discovered
        .inner_elements
        .iter()
        .enumerate()
        .map(|(i, v)| (v.clone(), i))
        .collect();
    let inner_elem_fps: Vec<u64> = discovered
        .inner_elements
        .iter()
        .map(value_fingerprint)
        .collect();
    Some(NestedSetVarMonitor {
        var_idx,
        layout,
        inner_elements: discovered.inner_elements.clone(),
        id_of,
        inner_elem_fps,
        slot_count,
        bailed: false,
        encoded_count: 0,
        escape_count: 0,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pos(x: i64, y: i64) -> Value {
        Value::Tuple(Rp::from(vec![Value::SmallInt(x), Value::SmallInt(y)]))
    }

    fn piece(positions: &[(i64, i64)]) -> Value {
        Value::Set(Rp::new(SortedSet::from_iter(
            positions.iter().map(|&(x, y)| pos(x, y)),
        )))
    }

    fn board(pieces: &[Vec<(i64, i64)>]) -> Value {
        Value::Set(Rp::new(SortedSet::from_iter(
            pieces.iter().map(|p| piece(p)),
        )))
    }

    #[test]
    fn discovers_tuple_position_universe_and_roundtrips() {
        // Two boards over a 3-position world with 3 distinct piece-shapes.
        let b0 = board(&[vec![(0, 0)], vec![(1, 0), (1, 1)]]);
        let b1 = board(&[vec![(1, 0), (1, 1)], vec![(0, 1)]]);
        let samples: Vec<&Value> = vec![&b0, &b1];

        let discovered =
            derive_nested_set_universe(&samples).expect("nested-set universe must be discovered");
        // Distinct inner positions: (0,0),(0,1),(1,0),(1,1) = 4.
        assert_eq!(discovered.inner_len, 4);
        // Distinct piece-shapes: {(0,0)}, {(1,0),(1,1)}, {(0,1)} = 3.
        assert_eq!(discovered.outer_len, 3);

        let report = validate_roundtrip(&discovered, &samples);
        assert_eq!(report.sampled_boards, 2);
        assert_eq!(report.roundtrip_ok, 2);
        assert_eq!(report.escapes, 0, "every sampled board must encode");
    }

    #[test]
    fn escape_when_board_has_unseen_piece_shape() {
        let b0 = board(&[vec![(0, 0)]]);
        let samples: Vec<&Value> = vec![&b0];
        let discovered = derive_nested_set_universe(&samples).expect("universe");
        // A board with a position never sampled escapes (inner element outside
        // the discovered inner universe).
        let unseen = board(&[vec![(9, 9)]]);
        let report = validate_roundtrip(&discovered, &[&unseen]);
        assert_eq!(report.sampled_boards, 1);
        assert_eq!(report.roundtrip_ok, 0);
        assert_eq!(report.escapes, 1);
    }

    /// The Klotski INIT board (the real `SlidingPuzzles` init), over the 4×5
    /// grid, as a nested-set `Value`.
    fn klotski_init_board() -> Value {
        board(&[
            vec![(0, 0), (0, 1)],
            vec![(1, 0), (2, 0), (1, 1), (2, 1)],
            vec![(3, 0), (3, 1)],
            vec![(0, 2), (0, 3)],
            vec![(1, 2), (2, 2)],
            vec![(3, 2), (3, 3)],
            vec![(1, 3)],
            vec![(2, 3)],
            vec![(0, 4)],
            vec![(3, 4)],
        ])
    }

    fn grid_4x5() -> Vec<(i64, i64)> {
        let mut g = Vec::new();
        for x in 0..4 {
            for y in 0..5 {
                g.push((x, y));
            }
        }
        g
    }

    /// STATIC (exploration-free) discovery over the Klotski grid derives the
    /// complete piece-shape universe: |inner|=20 (the 20 grid cells), |outer|=63
    /// (all grid-fitting translates of the 4 distinct INIT shapes: vertical
    /// domino 16 + horizontal domino 15 + 2×2 square 12 + singleton 20), a
    /// SUPERSET of the 60 reachable shapes. It fits a single 64-bit slot.
    #[test]
    fn static_discovery_derives_complete_klotski_universe() {
        let init = klotski_init_board();
        let positions = grid_4x5();
        let discovered = derive_nested_set_universe_static(&positions, &[&init])
            .expect("static Klotski universe must derive");
        assert_eq!(discovered.inner_len, 20, "20 grid cells");
        assert_eq!(
            discovered.outer_len, 63,
            "all grid-fitting translates: 16+15+12+20"
        );
        assert_eq!(
            discovered.layout.slot_count(),
            1,
            "63 shapes fit one u64 slot"
        );
    }

    /// The static universe is a SUPERSET of the sampled one AND freezes into a
    /// monitor that byte-matches `value_fingerprint` on the INIT board — the
    /// property that keeps the promotion verdict-identical.
    #[test]
    fn static_universe_freezes_and_fp_matches_on_init() {
        let init = klotski_init_board();
        let positions = grid_4x5();
        let discovered =
            derive_nested_set_universe_static(&positions, &[&init]).expect("static universe");
        let monitor = freeze_nested_set_var(0, &discovered).expect("freeze");
        match monitor.encode_board(&init) {
            NestedSetEncodeOutcome::Encoded { dedup_fp, slots } => {
                assert_eq!(slots.len(), 1, "1-slot mask");
                assert_eq!(
                    dedup_fp,
                    value_fingerprint(&init),
                    "monitored dedup fp must byte-match value_fingerprint(init board)"
                );
            }
            NestedSetEncodeOutcome::Escaped => panic!("INIT board must be in its own universe"),
        }
    }

    #[test]
    fn non_nested_value_is_not_discovered() {
        let scalar_set = Value::Set(Rp::new(SortedSet::from_iter([
            Value::SmallInt(1),
            Value::SmallInt(2),
        ])));
        assert!(!is_nested_set_value(&scalar_set));
        assert!(derive_nested_set_universe(&[&scalar_set]).is_none());
    }

    // ---- A5: frozen monitor + bijection-aware dedup fingerprint -------------

    /// The monitored dedup fingerprint of an IN-UNIVERSE board MUST byte-exactly
    /// equal `value_fingerprint(original_board)` (the same key the un-promoted
    /// interpreter path produces) — over tuple-position inner elements. This is
    /// the property that makes the promotion verdict-identical and the bail
    /// fail-closed (same fingerprint domain).
    #[test]
    fn monitored_fp_byte_matches_value_fingerprint_over_positions() {
        let b0 = board(&[vec![(0, 0)], vec![(1, 0), (1, 1)]]);
        let b1 = board(&[vec![(1, 0), (1, 1)], vec![(0, 1)]]);
        // A third distinct board over the same universe.
        let b2 = board(&[vec![(0, 0)], vec![(0, 1)]]);
        let samples: Vec<&Value> = vec![&b0, &b1];
        let discovered = derive_nested_set_universe(&samples).expect("universe");
        let monitor = freeze_nested_set_var(7, &discovered).expect("freeze");

        for board_val in [&b0, &b1] {
            match monitor.encode_board(board_val) {
                NestedSetEncodeOutcome::Encoded { dedup_fp, slots } => {
                    assert_eq!(slots.len(), monitor.slot_count());
                    assert_eq!(
                        dedup_fp,
                        value_fingerprint(board_val),
                        "monitored dedup fp must byte-match value_fingerprint(original board)"
                    );
                }
                NestedSetEncodeOutcome::Escaped => {
                    panic!("sampled board must not escape its own frozen universe")
                }
            }
        }

        // b2's pieces ({(0,0)} and {(0,1)}) were both sampled (b0 had {(0,0)},
        // b1 had {(0,1)}), so b2 is in-universe and must encode + byte-match too,
        // and dedup as DISTINCT from b0/b1 (no aliasing).
        let fp0 = match monitor.encode_board(&b0) {
            NestedSetEncodeOutcome::Encoded { dedup_fp, .. } => dedup_fp,
            NestedSetEncodeOutcome::Escaped => unreachable!(),
        };
        let fp1 = match monitor.encode_board(&b1) {
            NestedSetEncodeOutcome::Encoded { dedup_fp, .. } => dedup_fp,
            NestedSetEncodeOutcome::Escaped => unreachable!(),
        };
        match monitor.encode_board(&b2) {
            NestedSetEncodeOutcome::Encoded { dedup_fp, .. } => {
                assert_eq!(dedup_fp, value_fingerprint(&b2));
                assert_ne!(dedup_fp, fp0, "distinct boards must not alias");
                assert_ne!(dedup_fp, fp1, "distinct boards must not alias");
            }
            NestedSetEncodeOutcome::Escaped => {
                panic!("b2 is composed of sampled piece-shapes; must be in-universe")
            }
        }
        assert_ne!(fp0, fp1, "distinct boards must not alias");
    }

    /// FORCED-ESCAPE: a board carrying a position outside the frozen inner
    /// universe MUST be caught (fail-closed `Escaped`), never silently encoded.
    #[test]
    fn monitor_catches_inner_element_escape() {
        let b0 = board(&[vec![(0, 0)], vec![(1, 0)]]);
        let discovered = derive_nested_set_universe(&[&b0]).expect("universe");
        let monitor = freeze_nested_set_var(0, &discovered).expect("freeze");
        // (9, 9) was never sampled → outside the frozen inner universe.
        let escapee = board(&[vec![(9, 9)]]);
        assert_eq!(
            monitor.encode_board(&escapee),
            NestedSetEncodeOutcome::Escaped,
            "an unseen inner element MUST fail closed"
        );
        // The in-universe board still encodes + byte-matches.
        assert!(matches!(
            monitor.encode_board(&b0),
            NestedSetEncodeOutcome::Encoded { .. }
        ));
    }

    /// FORCED-ESCAPE: a board whose piece-SHAPE was never sampled (its inner
    /// elements ARE all in the inner universe, but the combination is a new
    /// piece-shape outside the frozen outer universe) MUST be caught.
    #[test]
    fn monitor_catches_piece_shape_escape() {
        // Sample only the singleton pieces {(0,0)} and {(1,1)}.
        let b0 = board(&[vec![(0, 0)], vec![(1, 1)]]);
        let discovered = derive_nested_set_universe(&[&b0]).expect("universe");
        let monitor = freeze_nested_set_var(0, &discovered).expect("freeze");
        // {(0,0),(1,1)} uses only in-universe inner elements but is a NEW
        // piece-shape (never sampled as a single piece) → outer-universe escape.
        let escapee = board(&[vec![(0, 0), (1, 1)]]);
        assert_eq!(
            monitor.encode_board(&escapee),
            NestedSetEncodeOutcome::Escaped,
            "an unseen piece-shape MUST fail closed"
        );
    }

    /// A6 diff-path hook: `observe_board_escape_only` is the escape-only twin of
    /// `encode_board` — it must accept exactly the in-universe boards (returning
    /// `true`, incrementing `encoded_count`) and FAIL CLOSED on the same escapes
    /// (returning `false`, latching `bailed`), without computing a fingerprint.
    #[test]
    fn observe_escape_only_matches_encode_board_membership() {
        let b0 = board(&[vec![(0, 0)], vec![(1, 0), (1, 1)]]);
        let b1 = board(&[vec![(1, 0), (1, 1)], vec![(0, 1)]]);
        let samples: Vec<&Value> = vec![&b0, &b1];
        let discovered = derive_nested_set_universe(&samples).expect("universe");
        let mut monitor = freeze_nested_set_var(7, &discovered).expect("freeze");

        // In-universe board: observed, encoded_count bumped, not bailed.
        assert!(monitor.observe_board_escape_only(&b0));
        assert_eq!(monitor.encoded_count, 1);
        assert!(!monitor.bailed);
        assert!(monitor.observe_board_escape_only(&b1));
        assert_eq!(monitor.encoded_count, 2);
        assert!(!monitor.bailed);

        // Escape (inner element outside the frozen inner universe): fail closed.
        let escapee = board(&[vec![(9, 9)]]);
        assert!(!monitor.observe_board_escape_only(&escapee));
        assert_eq!(monitor.escape_count, 1);
        assert!(monitor.bailed, "escape must latch bailed");

        // Once bailed, every observation short-circuits to false (cannot un-bail).
        assert!(!monitor.observe_board_escape_only(&b0));
        assert_eq!(monitor.encoded_count, 2, "no encode after bail");
    }

    /// The frozen layout carries `monitor_enforced: true` on both closures.
    #[test]
    fn frozen_layout_records_monitor_enforced() {
        let b0 = board(&[vec![(0, 0)]]);
        let discovered = derive_nested_set_universe(&[&b0]).expect("universe");
        let monitor = freeze_nested_set_var(3, &discovered).expect("freeze");
        match &monitor.layout {
            FlatValueLayout::NestedSetBitmask {
                outer_closure,
                inner_closure,
                ..
            } => {
                assert_eq!(
                    *outer_closure,
                    SetBitmaskUniverseClosure::DynamicallyDiscovered {
                        monitor_enforced: true
                    }
                );
                assert_eq!(
                    *inner_closure,
                    SetBitmaskUniverseClosure::DynamicallyDiscovered {
                        monitor_enforced: true
                    }
                );
                // Even with the monitor installed the universe is NOT statically
                // proven closed — flat-primary native dispatch stays fail-closed.
                assert!(!outer_closure.is_proven_closed());
            }
            _ => panic!("frozen layout must be NestedSetBitmask"),
        }
    }
}
