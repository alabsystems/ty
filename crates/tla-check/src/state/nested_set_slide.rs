// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Step B — native word-op SUCCESSOR KERNEL for the set-of-sets sliding-piece
//! class (the canonical example is `SlidingPuzzles`: `board \in SUBSET (SUBSET
//! Pos)` sliding rigid pieces on a `Pos == (0..W-1) \X (0..H-1)` grid).
//!
//! # What this replaces
//!
//! `SlidingPuzzles`'s `Next == \E e \in empty : board' \in update(e, empty)` is
//! a set-valued non-deterministic action over a nested set. The interpreter
//! evaluates it by materializing `empty = Pos \ UNION board`, then for every
//! empty cell a `dir`/`move`/`update` cascade of `Value::Set` allocations,
//! set-differences and set-builders — ~18× slower per successor than TLC.
//! This module lowers the same successor RELATION to word-ops over bitmasks.
//!
//! # Representation (self-describing — no outer universe needed here)
//!
//! A board is a **sorted `Vec<u64>` of piece masks**. Each piece mask is a
//! `u64` over the position grid: bit `b` set ⇔ grid cell `positions[b]` is
//! occupied by that piece. Because reachable pieces are pairwise DISJOINT (they
//! tile a subset of `Pos`), the `Vec<u64>` is a faithful, directly-comparable
//! encoding of the `SUBSET (SUBSET Pos)` board — dedup is a plain `Vec<u64>`
//! set membership. The compact single-slot NestedSetBitmask storage layout (the
//! frozen outer universe of shapes) is a *separate* concern (A3/A5); successor
//! generation needs only the position grid.
//!
//! # The word-op slide (value-preservation per op)
//!
//! Given a board `pieces` and its `occupied = OR pieces`:
//!
//! * `UNION board`                         → `occupied`  (OR-fold).
//! * `Pos \ UNION board` (empty cells)     → `full_pos & !occupied`.
//! * slide piece `pm` by unit vector `d`   → [`SlideGeometry::shift_piece`]:
//!   each set bit's grid cell `(x,y)` maps to `(x+dx, y+dy)`; a cell that leaves
//!   the grid makes the whole slide illegal (`None`) — this is exactly the
//!   original's `\A p \in m : p \in Pos` boundary filter.
//! * collision check `m \cap UNION(board \ {pc}) = {}` → `shifted & others == 0`
//!   where `others = occupied & !pm` (the OR of the OTHER pieces, since pieces
//!   are disjoint).
//! * successor board                       → `pieces` with `pm` replaced by
//!   `shifted`, re-sorted.
//!
//! # Equivalence to the spec's `update` (soundness)
//!
//! The original considers a move only for an empty cell `e` with an occupied
//! neighbour `s = e+d`: it slides the piece containing `s` by `-d` (toward `e`),
//! keeping it iff the translate stays in `Pos` and disjoint from the others.
//! This kernel instead enumerates `(piece, unit direction)` directly. The two
//! produce the IDENTICAL successor SET:
//!
//! * (kernel ⊆ spec) a kernel move slides `pm` by `d` with `shifted ⊆ Pos` and
//!   `shifted ∩ others = ∅`. The cells `shifted \ pm` are non-empty (nonzero
//!   translation of a finite piece), not in `pm`, and disjoint from `others`, so
//!   each is a currently-empty cell; pick one such cell `e = c+d` (empty, with
//!   `e-d = c ∈ pm` occupied) — then the spec's `update(e, empty)` with this `d`
//!   translates the same piece by `-(-d)=d` and passes the same filters,
//!   yielding the same board.
//! * (spec ⊆ kernel) a spec move for `(e, d)` slides the piece containing `e+d`
//!   by `-d`, kept iff the translate `m ⊆ Pos` and `m ∩ others = ∅` — exactly a
//!   kernel `(piece, -d)` legal move.
//!
//! [`reference_successors`] implements the spec's empty-cell-driven `update`
//! literally over sets, and the exhaustive-BFS test
//! `kernel_matches_reference_over_full_reachable_space` asserts the two agree at
//! EVERY reachable state (0 divergence) over the whole 25955-state Klotski
//! space — a self-contained crosscheck of the equivalence above.
//!
//! # Fail-closed
//!
//! Encoding a board with a cell outside the position grid returns `None`
//! (an escape): the caller must fall back to the interpreter, never silently
//! drop the state. A slide whose target leaves the grid is simply not a
//! successor (the spec agrees). The kernel never invents or misencodes a state.

use std::collections::HashMap;
use tla_value::Rp;
use std::sync::Arc;

use num_traits::ToPrimitive;
use tla_value::value::SortedSet;

use crate::Value;

/// The four von Neumann unit directions, in a fixed order.
pub(crate) const DIRS: [(i64, i64); 4] = [(1, 0), (0, 1), (-1, 0), (0, -1)];

/// A board as a sorted list of piece masks (each a `u64` over the position
/// grid). Sorted + deduped so equal boards compare equal.
pub(crate) type BoardMasks = Vec<u64>;

/// Position-grid geometry + shift tables for the slide kernel.
///
/// `positions` is the *exact* `Pos` universe (bit `b` ⇔ cell `positions[b]`).
/// Soundness of the boundary check depends on `positions` being exactly `Pos`:
/// a cell that the grid omits would let the kernel wrongly reject an in-`Pos`
/// slide (a missing successor). Construction therefore takes the position set
/// explicitly (from the frozen nested-set inner universe, or a rectangular
/// `W×H` grid), never a guess.
#[derive(Debug, Clone)]
pub(crate) struct SlideGeometry {
    /// Grid cell `(x, y)` for each position bit, in bit order.
    positions: Vec<(i64, i64)>,
    /// Inverse: grid cell → position bit.
    pos_index: HashMap<(i64, i64), usize>,
    /// `shift_tables[d][b]` = the position bit that cell `positions[b]` maps to
    /// under direction `DIRS[d]`, or `None` when that neighbour leaves the grid.
    shift_tables: [Vec<Option<usize>>; 4],
    /// All grid bits set — the `Pos` mask (`UNION` upper bound / empty-cell mask).
    full_pos_mask: u64,
}

impl SlideGeometry {
    /// Build geometry from the exact `Pos` universe. Positions are canonicalized
    /// (sorted, deduped) so the bit assignment is deterministic. Returns `None`
    /// when the grid exceeds a single `u64` (> 64 cells) — the single-word inner
    /// universe cap — failing closed.
    #[must_use]
    pub(crate) fn new(mut positions: Vec<(i64, i64)>) -> Option<Self> {
        positions.sort_unstable();
        positions.dedup();
        if positions.is_empty() || positions.len() > 64 {
            return None;
        }
        let pos_index: HashMap<(i64, i64), usize> = positions
            .iter()
            .enumerate()
            .map(|(b, &xy)| (xy, b))
            .collect();
        let shift_tables: [Vec<Option<usize>>; 4] = std::array::from_fn(|d| {
            let (dx, dy) = DIRS[d];
            positions
                .iter()
                .map(|&(x, y)| pos_index.get(&(x + dx, y + dy)).copied())
                .collect()
        });
        let full_pos_mask = if positions.len() == 64 {
            u64::MAX
        } else {
            (1u64 << positions.len()) - 1
        };
        Some(SlideGeometry {
            positions,
            pos_index,
            shift_tables,
            full_pos_mask,
        })
    }

    /// Build geometry for a rectangular `Pos == (0..w-1) \X (0..h-1)` grid.
    ///
    /// Sound ONLY when the spec's `Pos` really is this full rectangle (as in
    /// `SlidingPuzzles`, `W==4 H==5`). Used by the self-contained tests and any
    /// caller that has *proven* `Pos` rectangular.
    #[must_use]
    pub(crate) fn rectangular(w: i64, h: i64) -> Option<Self> {
        let mut positions = Vec::new();
        for x in 0..w {
            for y in 0..h {
                positions.push((x, y));
            }
        }
        Self::new(positions)
    }

    /// Number of grid cells (inner universe size).
    #[must_use]
    pub(crate) fn num_positions(&self) -> usize {
        self.positions.len()
    }

    /// `UNION board` — the OR-fold of the piece masks (occupied cells).
    #[must_use]
    #[inline]
    pub(crate) fn union_occupied(pieces: &[u64]) -> u64 {
        pieces.iter().fold(0u64, |acc, &m| acc | m)
    }

    /// `Pos \ UNION board` — the empty cells.
    #[must_use]
    #[inline]
    pub(crate) fn empty_cells(&self, occupied: u64) -> u64 {
        self.full_pos_mask & !occupied
    }

    /// Slide a piece mask by direction `DIRS[d]`. Returns the translated mask, or
    /// `None` when any cell would leave the grid (`\A p \in m : p \in Pos` fails).
    ///
    /// The result has the same popcount as the input (a bijective on-grid
    /// translation), so a piece never gains or loses cells.
    #[must_use]
    #[inline]
    pub(crate) fn shift_piece(&self, mask: u64, d: usize) -> Option<u64> {
        let table = &self.shift_tables[d];
        let mut m = mask;
        let mut result = 0u64;
        while m != 0 {
            let b = m.trailing_zeros() as usize;
            m &= m - 1;
            result |= 1u64 << table[b]?;
        }
        Some(result)
    }

    /// The successor boards of `pieces` under the word-op slide (the native
    /// analogue of `update`). Each successor is sorted+deduped `BoardMasks`; the
    /// returned list is itself sorted+deduped so it is a canonical SET.
    ///
    /// Precondition: `pieces` is a valid reachable board (pairwise-disjoint
    /// masks, each ⊆ grid). Callers obtain it from [`Self::board_to_masks`],
    /// which fails closed on any out-of-grid cell.
    #[must_use]
    pub(crate) fn slide_successors(&self, pieces: &[u64]) -> Vec<BoardMasks> {
        let occupied = Self::union_occupied(pieces);
        let mut out: Vec<BoardMasks> = Vec::new();
        for (i, &pm) in pieces.iter().enumerate() {
            let others = occupied & !pm;
            for d in 0..4 {
                let Some(shifted) = self.shift_piece(pm, d) else {
                    continue; // a cell left the grid → not a legal move
                };
                if shifted & others != 0 {
                    continue; // collides with another piece
                }
                // Replace piece i with the shifted mask, re-canonicalize.
                let mut next: BoardMasks = Vec::with_capacity(pieces.len());
                for (j, &q) in pieces.iter().enumerate() {
                    next.push(if j == i { shifted } else { q });
                }
                next.sort_unstable();
                out.push(next);
            }
        }
        out.sort_unstable();
        out.dedup();
        out
    }

    /// Successor boards GROUPED BY EMPTY CELL, in the interpreter's enumeration
    /// structure for `\E e \in empty : board' \in update(e, empty)`:
    /// `result[k]` is the set of successor boards produced by `update(e, empty)`
    /// for the `k`-th empty cell (empty cells in position bit order == canonical
    /// `Value` order of `<<x,y>>` tuples). Within a group the boards are the
    /// SET `update(e, empty)`; the caller emits them in `Value` order with
    /// first-occurrence dedup, reproducing the interpreter's successor sequence
    /// exactly (so the level-order BFS halts at the invariant violation after the
    /// identical states-explored count).
    #[must_use]
    pub(crate) fn slide_successors_by_empty_cell(&self, pieces: &[u64]) -> Vec<Vec<BoardMasks>> {
        let occupied = Self::union_occupied(pieces);
        let empty = self.empty_cells(occupied);
        let mut groups: Vec<Vec<BoardMasks>> = Vec::new();
        let mut em = empty;
        while em != 0 {
            let eb = em.trailing_zeros() as usize;
            em &= em - 1;
            let (ex, ey) = self.positions[eb];
            let mut group: Vec<BoardMasks> = Vec::new();
            for (dx, dy) in DIRS {
                let Some(&sb) = self.pos_index.get(&(ex + dx, ey + dy)) else {
                    continue;
                };
                let sbit = 1u64 << sb;
                if occupied & sbit == 0 {
                    continue; // e+d not occupied
                }
                let Some((pi, &pm)) = pieces.iter().enumerate().find(|(_, &m)| m & sbit != 0)
                else {
                    continue;
                };
                // Slide pc toward e: translate by -d.
                let Some(moved) = self.shift_piece(pm, neg_dir_index(dx, dy)) else {
                    continue;
                };
                let others = occupied & !pm;
                if moved & others != 0 {
                    continue;
                }
                let mut next: BoardMasks = Vec::with_capacity(pieces.len());
                for (j, &q) in pieces.iter().enumerate() {
                    next.push(if j == pi { moved } else { q });
                }
                next.sort_unstable();
                group.push(next);
            }
            group.sort_unstable();
            group.dedup();
            groups.push(group);
        }
        groups
    }

    /// REFERENCE successor generator — the spec's empty-cell-driven `update`,
    /// implemented literally over sets, for the equivalence crosscheck. Slower
    /// (this is the oracle, not the fast path). Produces the canonical successor
    /// SET (sorted+deduped `BoardMasks`).
    #[must_use]
    pub(crate) fn reference_successors(&self, pieces: &[u64]) -> Vec<BoardMasks> {
        let occupied = Self::union_occupied(pieces);
        let empty = self.empty_cells(occupied);
        let mut out: Vec<BoardMasks> = Vec::new();
        // For each empty cell e, each direction d: s = e+d; if s occupied, slide
        // the piece containing s by -d.
        let mut em = empty;
        while em != 0 {
            let eb = em.trailing_zeros() as usize;
            em &= em - 1;
            let (ex, ey) = self.positions[eb];
            for (dx, dy) in DIRS {
                // s = e + d.
                let Some(&sb) = self.pos_index.get(&(ex + dx, ey + dy)) else {
                    continue; // e+d off grid
                };
                let sbit = 1u64 << sb;
                if occupied & sbit == 0 {
                    continue; // e+d not occupied (empty) → not a dir(e) direction
                }
                // pc = the piece containing s.
                let Some((pi, &pm)) = pieces.iter().enumerate().find(|(_, &m)| m & sbit != 0)
                else {
                    continue;
                };
                // m = pc translated by -d.
                let Some(moved) = self.shift_piece(pm, neg_dir_index(dx, dy)) else {
                    continue; // a cell leaves Pos
                };
                let others = occupied & !pm;
                if moved & others != 0 {
                    continue; // overlaps another piece
                }
                let mut next: BoardMasks = Vec::with_capacity(pieces.len());
                for (j, &q) in pieces.iter().enumerate() {
                    next.push(if j == pi { moved } else { q });
                }
                next.sort_unstable();
                out.push(next);
            }
        }
        out.sort_unstable();
        out.dedup();
        out
    }

    /// Encode a board `Value` (`Set` of `Set` of position `Tuple`s) into sorted
    /// `BoardMasks`. Returns `None` (an ESCAPE — fail closed) when the value is
    /// not a set-of-sets, or any inner element is not a grid position `(x, y)`.
    #[must_use]
    pub(crate) fn board_to_masks(&self, board: &Value) -> Option<BoardMasks> {
        let Value::Set(outer) = board else {
            return None;
        };
        let mut masks: BoardMasks = Vec::with_capacity(outer.len());
        for piece in outer.iter() {
            let Value::Set(inner) = piece else {
                return None;
            };
            let mut mask = 0u64;
            for elem in inner.iter() {
                let xy = value_to_pos(elem)?;
                let &b = self.pos_index.get(&xy)?; // position outside Pos → escape
                mask |= 1u64 << b;
            }
            masks.push(mask);
        }
        masks.sort_unstable();
        // Reachable pieces are disjoint; a duplicate mask would collapse in the
        // TLA set. Preserve set semantics by deduping (never silently merges two
        // DISTINCT pieces — equal masks are the same set element).
        masks.dedup();
        Some(masks)
    }

    /// Decode `BoardMasks` back into the board `Value` (`Set` of `Set` of
    /// position `Tuple`s), byte-identical to the interpreter's board value.
    #[must_use]
    pub(crate) fn masks_to_board(&self, pieces: &[u64]) -> Value {
        let outer = pieces.iter().map(|&mask| self.mask_to_piece(mask));
        Value::Set(Rp::new(SortedSet::from_iter(outer)))
    }

    /// Decode a single piece mask into a `Set` of position `Tuple`s.
    fn mask_to_piece(&self, mask: u64) -> Value {
        let mut m = mask;
        let mut cells: Vec<Value> = Vec::with_capacity(mask.count_ones() as usize);
        while m != 0 {
            let b = m.trailing_zeros() as usize;
            m &= m - 1;
            let (x, y) = self.positions[b];
            cells.push(Value::Tuple(Rp::from(vec![
                Value::SmallInt(x),
                Value::SmallInt(y),
            ])));
        }
        Value::Set(Rp::new(SortedSet::from_iter(cells)))
    }

    /// The mask of a fixed cell set, or `None` if any cell is off-grid. Handy for
    /// invariants like `KlotskiGoal`'s goal square.
    #[must_use]
    pub(crate) fn cells_to_mask(&self, cells: &[(i64, i64)]) -> Option<u64> {
        let mut mask = 0u64;
        for &xy in cells {
            let &b = self.pos_index.get(&xy)?;
            mask |= 1u64 << b;
        }
        Some(mask)
    }
}

/// An ARMED slide kernel bound to one nested-set state variable — the native
/// successor fast-path.
///
/// The kernel is SOUND for the sliding-piece class only, so arming happens on
/// two paths with different justification:
///
/// * **DEFAULT (recognizer-proven)** — [`Self::try_arm_recognized`]: the static
///   recognizer (`check::model_checker::slide_recognize`) has PROVEN the
///   spec's `Next` is the rigid-unit-slide relation and evaluated the exact
///   `Pos` grid; this constructor adds the INIT value preconditions
///   (in-grid cells, pairwise-disjoint pieces). Kill switch:
///   `TY_NO_NESTED_SET_SLIDE=1`. A bounded first-N-states tripwire compares
///   kernel vs interpreter and disarms on any divergence.
/// * **FORCED (`TY_NESTED_SET_SLIDE=1`)** — [`Self::try_arm`]: the original
///   opt-in override that arms blindly from the INIT bounding box; validated
///   by the per-state crosscheck (`TY_NESTED_SET_SLIDE_CROSSCHECK=1`, which
///   compares against the interpreter and panics on any divergence).
#[derive(Debug, Clone)]
pub(crate) struct SlideKernelArm {
    /// The state-variable index this arm generates successors for.
    pub(crate) board_var_idx: usize,
    /// Position-grid geometry (`Pos`) + shift tables.
    pub(crate) geometry: SlideGeometry,
}

impl SlideKernelArm {
    /// Try to arm the slide kernel for a nested-set board variable.
    ///
    /// `init_boards` are the variable's INIT values (set-of-sets of position
    /// tuples). The position grid `Pos` is derived as the BOUNDING BOX of every
    /// position appearing in the init boards. This is exact when `Pos` is the
    /// full rectangle the pieces tile (as in `SlidingPuzzles`, `(0..3) \X
    /// (0..4)` — the Klotski init spans all four corners). A non-rectangular
    /// `Pos` (a grid with holes) would make the box a superset and is NOT
    /// supported here — the per-state crosscheck certifies the grid for the
    /// run. The DEFAULT path ([`Self::try_arm_recognized`]) instead receives
    /// the exact `Pos` evaluated by the static recognizer and has no such
    /// caveat.
    ///
    /// Returns `None` (do not arm — fall back to the interpreter) when the
    /// values are not set-of-sets of 2-int tuples, or the grid exceeds 64 cells.
    #[must_use]
    pub(crate) fn try_arm(board_var_idx: usize, init_boards: &[&Value]) -> Option<Self> {
        let mut positions: Vec<(i64, i64)> = Vec::new();
        let mut saw_board = false;
        for board in init_boards {
            let Value::Set(outer) = board else { continue };
            if outer.is_empty() || !outer.iter().all(|p| matches!(p, Value::Set(_))) {
                continue;
            }
            saw_board = true;
            for piece in outer.iter() {
                let Value::Set(inner) = piece else {
                    return None;
                };
                for elem in inner.iter() {
                    positions.push(value_to_pos(elem)?);
                }
            }
        }
        if !saw_board || positions.is_empty() {
            return None;
        }
        let min_x = positions.iter().map(|&(x, _)| x).min()?;
        let max_x = positions.iter().map(|&(x, _)| x).max()?;
        let min_y = positions.iter().map(|&(_, y)| y).min()?;
        let max_y = positions.iter().map(|&(_, y)| y).max()?;
        let mut grid: Vec<(i64, i64)> = Vec::new();
        for x in min_x..=max_x {
            for y in min_y..=max_y {
                grid.push((x, y));
            }
        }
        let geometry = SlideGeometry::new(grid)?;
        Some(SlideKernelArm {
            board_var_idx,
            geometry,
        })
    }

    /// AUTO-ARM (recognizer-proven) entry: arm the slide kernel over the EXACT
    /// `Pos` grid that the static recognizer
    /// ([`crate::check::model_checker::slide_recognize`]) evaluated from the
    /// spec — NOT the INIT bounding box. The recognizer has already PROVEN that
    /// the spec's `Next` is the rigid-unit-slide relation over `positions`;
    /// this constructor adds the remaining *value-level* preconditions the
    /// kernel⇔spec equivalence proof (module docs) assumes about the INIT
    /// boards:
    ///
    /// * every board is a set-of-sets of 2-int tuples, every cell inside the
    ///   grid (`board_to_masks` succeeds — otherwise the arm would escape on
    ///   state one anyway);
    /// * the pieces of each board are PAIRWISE DISJOINT. Disjointness is what
    ///   makes `UNION (board \ {pc})` equal `occupied & !pm` and what makes the
    ///   spec's unique-`CHOOSE` (`ChooseOne`) total; the kernel preserves it
    ///   inductively (a slide keeps `moved ∩ others = ∅` and leaves the other
    ///   pieces untouched), so checking the INIT boards proves it for every
    ///   reachable board.
    ///
    /// Any failure returns `None` — do not arm, fall back to the interpreter.
    #[must_use]
    pub(crate) fn try_arm_recognized(
        board_var_idx: usize,
        positions: Vec<(i64, i64)>,
        init_boards: &[&Value],
    ) -> Option<Self> {
        let geometry = SlideGeometry::new(positions)?;
        if init_boards.is_empty() {
            return None;
        }
        for board in init_boards {
            let masks = geometry.board_to_masks(board)?;
            // Pairwise-disjointness (see doc comment above).
            let mut acc = 0u64;
            for &m in &masks {
                if acc & m != 0 {
                    return None;
                }
                acc |= m;
            }
        }
        Some(SlideKernelArm {
            board_var_idx,
            geometry,
        })
    }

    /// Native successors of `board_value` as board `Value`s (fresh set-of-sets),
    /// or `None` (an ESCAPE — fail closed) when the board does not encode into
    /// the grid. On `None` the caller MUST fall back to the interpreter.
    #[must_use]
    pub(crate) fn successors(&self, board_value: &Value) -> Option<Vec<Value>> {
        let masks = self.geometry.board_to_masks(board_value)?;
        // Reproduce the interpreter's `\E e \in empty : board' \in update(e,
        // empty)` enumeration order: empty cells in position order (outer), the
        // `update(e, empty)` board SET in canonical `Value` order (inner),
        // first-occurrence dedup across empty cells. This keeps the level-order
        // BFS sibling sequence identical to the un-armed run, so it halts at the
        // invariant violation after the SAME states-explored count.
        let groups = self.geometry.slide_successors_by_empty_cell(&masks);
        let mut seen: std::collections::HashSet<BoardMasks> = std::collections::HashSet::new();
        let mut out: Vec<Value> = Vec::new();
        for group in groups {
            // `Value`-order the `update(e, empty)` board set for this empty cell.
            let mut group_boards: Vec<(Value, BoardMasks)> = group
                .into_iter()
                .map(|m| (self.geometry.masks_to_board(&m), m))
                .collect();
            group_boards.sort_by(|a, b| a.0.cmp(&b.0));
            for (board, key) in group_boards {
                if seen.insert(key) {
                    out.push(board);
                }
            }
        }
        Some(out)
    }
}

/// Index into [`DIRS`] of the negation of `(dx, dy)`.
fn neg_dir_index(dx: i64, dy: i64) -> usize {
    DIRS.iter()
        .position(|&(a, b)| a == -dx && b == -dy)
        .expect("DIRS is closed under negation")
}

/// Interpret a `Value` as a grid position `(x, y)` — a 2-tuple of integers.
/// Returns `None` for any other shape (fail closed).
pub(crate) fn value_to_pos(v: &Value) -> Option<(i64, i64)> {
    let Value::Tuple(t) = v else { return None };
    if t.len() != 2 {
        return None;
    }
    Some((value_to_int(&t[0])?, value_to_int(&t[1])?))
}

fn value_to_int(v: &Value) -> Option<i64> {
    match v {
        Value::SmallInt(i) => Some(*i),
        Value::Int(i) => i.to_i64(),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    // ---- Klotski fixture (the real SlidingPuzzles board) --------------------

    /// The initial Klotski board as `BoardMasks`, over a 4×5 rectangular grid.
    /// Mirrors `SlidingPuzzles.Klotski`.
    fn klotski_init(geo: &SlideGeometry) -> BoardMasks {
        let pieces: &[&[(i64, i64)]] = &[
            &[(0, 0), (0, 1)],
            &[(1, 0), (2, 0), (1, 1), (2, 1)],
            &[(3, 0), (3, 1)],
            &[(0, 2), (0, 3)],
            &[(1, 2), (2, 2)],
            &[(3, 2), (3, 3)],
            &[(1, 3)],
            &[(2, 3)],
            &[(0, 4)],
            &[(3, 4)],
        ];
        let mut masks: BoardMasks = pieces
            .iter()
            .map(|cells| geo.cells_to_mask(cells).unwrap())
            .collect();
        masks.sort_unstable();
        masks
    }

    fn goal_mask(geo: &SlideGeometry) -> u64 {
        geo.cells_to_mask(&[(1, 3), (1, 4), (2, 3), (2, 4)])
            .unwrap()
    }

    // ---- geometry + word-op unit tests --------------------------------------

    #[test]
    fn geometry_rectangular_grid_has_20_cells() {
        let geo = SlideGeometry::rectangular(4, 5).unwrap();
        assert_eq!(geo.num_positions(), 20);
        assert_eq!(geo.full_pos_mask.count_ones(), 20);
    }

    #[test]
    fn shift_piece_moves_in_bounds_and_rejects_off_grid() {
        let geo = SlideGeometry::rectangular(4, 5).unwrap();
        // A vertical domino {(0,0),(0,1)}.
        let dom = geo.cells_to_mask(&[(0, 0), (0, 1)]).unwrap();
        // Right (1,0) → {(1,0),(1,1)}.
        let right = geo.shift_piece(dom, 0).unwrap();
        assert_eq!(right, geo.cells_to_mask(&[(1, 0), (1, 1)]).unwrap());
        // Down (0,1) → {(0,1),(0,2)}.
        let down = geo.shift_piece(dom, 1).unwrap();
        assert_eq!(down, geo.cells_to_mask(&[(0, 1), (0, 2)]).unwrap());
        // Left (-1,0) leaves the grid (x=-1) → None.
        assert_eq!(geo.shift_piece(dom, 2), None);
        // Up (0,-1) leaves the grid (y=-1) → None.
        assert_eq!(geo.shift_piece(dom, 3), None);
        // popcount preserved on legal moves.
        assert_eq!(right.count_ones(), dom.count_ones());
    }

    #[test]
    fn union_and_empty_cells_are_complementary() {
        let geo = SlideGeometry::rectangular(4, 5).unwrap();
        let board = klotski_init(&geo);
        let occ = SlideGeometry::union_occupied(&board);
        // Klotski covers 18 of 20 cells; 2 empty.
        assert_eq!(occ.count_ones(), 18);
        let empty = geo.empty_cells(occ);
        assert_eq!(empty.count_ones(), 2);
        assert_eq!(occ & empty, 0);
        assert_eq!(occ | empty, geo.full_pos_mask);
        // The two empty cells are (1,4) and (2,4).
        assert_eq!(empty, geo.cells_to_mask(&[(1, 4), (2, 4)]).unwrap());
    }

    #[test]
    fn overlap_check_rejects_collision() {
        let geo = SlideGeometry::rectangular(4, 5).unwrap();
        let a = geo.cells_to_mask(&[(1, 0), (1, 1)]).unwrap();
        let b = geo.cells_to_mask(&[(1, 1), (1, 2)]).unwrap();
        assert_ne!(a & b, 0, "sharing (1,1) must overlap");
        let c = geo.cells_to_mask(&[(2, 0), (2, 1)]).unwrap();
        assert_eq!(a & c, 0, "disjoint columns must not overlap");
    }

    // ---- encode / decode round-trip -----------------------------------------

    #[test]
    fn board_masks_roundtrip_through_value() {
        let geo = SlideGeometry::rectangular(4, 5).unwrap();
        let board = klotski_init(&geo);
        let value = geo.masks_to_board(&board);
        let back = geo
            .board_to_masks(&value)
            .expect("in-grid board must encode");
        assert_eq!(back, board, "mask → Value → mask must round-trip");
    }

    #[test]
    fn board_to_masks_escapes_on_off_grid_position() {
        let geo = SlideGeometry::rectangular(4, 5).unwrap();
        // A board with a cell (9,9) outside the grid must fail closed (None).
        let piece = Value::Set(Rp::new(SortedSet::from_iter([Value::Tuple(Rp::from(
            vec![Value::SmallInt(9), Value::SmallInt(9)],
        ))])));
        let board = Value::Set(Rp::new(SortedSet::from_iter([piece])));
        assert_eq!(
            geo.board_to_masks(&board),
            None,
            "an out-of-grid position MUST escape (fail closed)"
        );
    }

    #[test]
    fn board_to_masks_escapes_on_non_nested_value() {
        let geo = SlideGeometry::rectangular(4, 5).unwrap();
        // A flat scalar set is not a set-of-sets board.
        let flat = Value::Set(Rp::new(SortedSet::from_iter([Value::SmallInt(1)])));
        assert_eq!(geo.board_to_masks(&flat), None);
    }

    // ---- kernel ↔ reference (spec `update`) equivalence ----------------------

    #[test]
    fn kernel_matches_reference_on_init() {
        let geo = SlideGeometry::rectangular(4, 5).unwrap();
        let board = klotski_init(&geo);
        let kernel = geo.slide_successors(&board);
        let reference = geo.reference_successors(&board);
        assert_eq!(
            kernel, reference,
            "kernel successors must equal spec update"
        );
        // Klotski's initial board has exactly the moves TLC finds from init.
        assert!(!kernel.is_empty());
        // Every successor is a valid board: 10 disjoint pieces covering 18 cells.
        for succ in &kernel {
            assert_eq!(succ.len(), board.len(), "piece count preserved");
            assert_eq!(
                SlideGeometry::union_occupied(succ).count_ones(),
                18,
                "occupied-cell count preserved"
            );
            // pairwise disjoint
            let mut acc = 0u64;
            for &m in succ {
                assert_eq!(acc & m, 0, "pieces must stay disjoint");
                acc |= m;
            }
        }
    }

    /// THE crosscheck: exhaustive BFS over the full Klotski reachable space,
    /// asserting the word-op kernel's successors EQUAL the spec's `update`
    /// (reference) at EVERY reachable state — 0 divergence — and that the space
    /// has the independently-known size (25955 states).
    #[test]
    fn kernel_matches_reference_over_full_reachable_space() {
        let geo = SlideGeometry::rectangular(4, 5).unwrap();
        let init = klotski_init(&geo);

        let mut seen: HashSet<BoardMasks> = HashSet::new();
        let mut frontier: Vec<BoardMasks> = vec![init.clone()];
        seen.insert(init);
        let mut divergences = 0usize;

        while let Some(board) = frontier.pop() {
            let kernel = geo.slide_successors(&board);
            let reference = geo.reference_successors(&board);
            if kernel != reference {
                divergences += 1;
            }
            for succ in kernel {
                if seen.insert(succ.clone()) {
                    frontier.push(succ);
                }
            }
        }

        assert_eq!(divergences, 0, "kernel diverged from spec `update`");
        assert_eq!(
            seen.len(),
            25955,
            "full Klotski reachable space size (independently verified)"
        );
    }

    /// The empty-cell-grouped enumeration (interpreter order) covers EXACTLY the
    /// same successor SET as the piece-driven kernel — reordering never changes
    /// the reachable set — over the full Klotski space.
    #[test]
    fn by_empty_cell_grouping_matches_kernel_set_over_full_space() {
        let geo = SlideGeometry::rectangular(4, 5).unwrap();
        let init = klotski_init(&geo);
        let mut seen: HashSet<BoardMasks> = HashSet::new();
        let mut frontier: Vec<BoardMasks> = vec![init.clone()];
        seen.insert(init);
        while let Some(board) = frontier.pop() {
            let kernel: HashSet<BoardMasks> = geo.slide_successors(&board).into_iter().collect();
            let grouped: HashSet<BoardMasks> = geo
                .slide_successors_by_empty_cell(&board)
                .into_iter()
                .flatten()
                .collect();
            assert_eq!(kernel, grouped, "grouped set must equal kernel set");
            for succ in kernel {
                if seen.insert(succ.clone()) {
                    frontier.push(succ);
                }
            }
        }
        assert_eq!(seen.len(), 25955);
    }

    /// KlotskiGoal reachability: the goal 2×2 square IS reachable (the spec's
    /// invariant is violated), and BFS finds a violating board.
    #[test]
    fn klotski_goal_is_reachable() {
        let geo = SlideGeometry::rectangular(4, 5).unwrap();
        let goal = goal_mask(&geo);
        let init = klotski_init(&geo);

        let mut seen: HashSet<BoardMasks> = HashSet::new();
        let mut frontier: Vec<BoardMasks> = vec![init.clone()];
        seen.insert(init);
        let mut found = false;
        while let Some(board) = frontier.pop() {
            if board.iter().any(|&m| m == goal) {
                found = true;
                break;
            }
            for succ in geo.slide_successors(&board) {
                if seen.insert(succ.clone()) {
                    frontier.push(succ);
                }
            }
        }
        assert!(
            found,
            "the Klotski goal square must be reachable (violation)"
        );
    }

    /// Non-injective / collapsed-piece fail-safe: `board_to_masks` dedups equal
    /// piece masks (they are the same TLA set element), never fabricating an
    /// extra piece.
    #[test]
    fn board_to_masks_dedups_equal_pieces() {
        let geo = SlideGeometry::rectangular(4, 5).unwrap();
        let p = Value::Set(Rp::new(SortedSet::from_iter([Value::Tuple(Rp::from(
            vec![Value::SmallInt(0), Value::SmallInt(0)],
        ))])));
        // A Value::Set cannot actually hold two equal elements, so this exercises
        // the single-element path; the dedup guard is defensive.
        let board = Value::Set(Rp::new(SortedSet::from_iter([p])));
        let masks = geo.board_to_masks(&board).unwrap();
        assert_eq!(masks.len(), 1);
        assert_eq!(masks[0], geo.cells_to_mask(&[(0, 0)]).unwrap());
    }

    // ---- arm (promotion) positive + fail-closed ------------------------------

    #[test]
    fn arm_derives_grid_from_klotski_init_and_matches_reference() {
        let geo = SlideGeometry::rectangular(4, 5).unwrap();
        let init_value = geo.masks_to_board(&klotski_init(&geo));
        let arm = SlideKernelArm::try_arm(0, &[&init_value]).expect("Klotski init must arm");
        // Bounding box of Klotski positions is exactly the 4×5 grid = Pos.
        assert_eq!(arm.geometry.num_positions(), 20);
        // The armed kernel's successors (as decoded boards) equal the reference's.
        let kernel_boards = arm.successors(&init_value).unwrap();
        let kernel_masks: HashSet<BoardMasks> = kernel_boards
            .iter()
            .map(|b| arm.geometry.board_to_masks(b).unwrap())
            .collect();
        let ref_masks: HashSet<BoardMasks> = geo
            .reference_successors(&klotski_init(&geo))
            .into_iter()
            .collect();
        assert_eq!(kernel_masks, ref_masks);
    }

    #[test]
    fn arm_fails_closed_on_non_nested_or_bad_shape() {
        // Flat scalar set → not a set-of-sets board → do not arm.
        let flat = Value::Set(Rp::new(SortedSet::from_iter([Value::SmallInt(1)])));
        assert!(SlideKernelArm::try_arm(0, &[&flat]).is_none());
        // A set-of-sets whose inner elements are NOT 2-int tuples → do not arm.
        let bad_inner = Value::Set(Rp::new(SortedSet::from_iter([Value::Set(Rp::new(
            SortedSet::from_iter([Value::SmallInt(3)]),
        ))])));
        assert!(SlideKernelArm::try_arm(0, &[&bad_inner]).is_none());
        // Empty input → do not arm.
        assert!(SlideKernelArm::try_arm(0, &[]).is_none());
    }

    #[test]
    fn arm_escapes_on_out_of_grid_successor_input() {
        let geo = SlideGeometry::rectangular(4, 5).unwrap();
        let init_value = geo.masks_to_board(&klotski_init(&geo));
        let arm = SlideKernelArm::try_arm(0, &[&init_value]).unwrap();
        // A board carrying a cell outside the derived grid must ESCAPE (None),
        // so the caller falls back to the interpreter.
        let off_grid = Value::Set(Rp::new(SortedSet::from_iter([Value::Set(Rp::new(
            SortedSet::from_iter([Value::Tuple(Rp::from(vec![
                Value::SmallInt(99),
                Value::SmallInt(99),
            ]))]),
        ))])));
        assert!(arm.successors(&off_grid).is_none());
    }

    #[test]
    fn geometry_rejects_oversized_grid() {
        // > 64 cells cannot fit a single-word inner universe → fail closed.
        assert!(SlideGeometry::rectangular(9, 9).is_none());
        assert!(SlideGeometry::new((0..65).map(|i| (i, 0)).collect()).is_none());
    }
}
