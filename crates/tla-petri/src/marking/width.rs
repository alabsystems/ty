// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

use crate::invariant::{structural_place_bound, PInvariant};
use crate::petri_net::PetriNet;

/// Token storage width for memory-efficient BFS exploration.
///
/// Determined by structural analysis of the Petri net. For token-conserving
/// nets (every transition preserves total token count), the maximum tokens
/// in any single place is bounded by the initial total token count, enabling
/// up to 8x memory savings per stored state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum TokenWidth {
    /// All reachable token counts fit in u8 (<= 255). 8x savings.
    U8,
    /// All reachable token counts fit in u16 (<= 65535). 4x savings.
    U16,
    /// Full u64 width. No savings.
    U64,
}

impl TokenWidth {
    /// Bytes per token value.
    #[must_use]
    pub fn bytes(self) -> usize {
        match self {
            Self::U8 => 1,
            Self::U16 => 2,
            Self::U64 => 8,
        }
    }
}

/// Determine the compact token width for a Petri net.
///
/// For token-conserving nets (every transition has equal total input and
/// output weights), the total token count is invariant across all reachable
/// markings. This bounds max(any place) <= total(initial marking), enabling
/// compact storage when the total is small.
///
/// Non-conserving nets fall back to full u64 width.
pub(crate) fn determine_width(net: &PetriNet) -> TokenWidth {
    let conserving = net.transitions.iter().all(|transition| {
        let in_weight: u64 = transition.inputs.iter().map(|arc| arc.weight).sum();
        let out_weight: u64 = transition.outputs.iter().map(|arc| arc.weight).sum();
        in_weight == out_weight
    });

    if !conserving {
        return TokenWidth::U64;
    }

    let total: u64 = net.initial_marking.iter().sum();
    if u8::try_from(total).is_ok() {
        TokenWidth::U8
    } else if u16::try_from(total).is_ok() {
        TokenWidth::U16
    } else {
        TokenWidth::U64
    }
}

/// Determine compact token width using P-invariant structural bounds.
///
/// Strictly tighter than [`determine_width`] for nets where P-invariants
/// provide per-place bounds even when the net is not globally conserving,
/// or where invariant bounds are tighter than the initial token total.
///
/// Falls back to [`determine_width`] if `invariants` is empty or any
/// place is uncovered by invariants.
///
/// Convenience wrapper that collapses [`determine_width_per_place`] into a
/// single worst-case width across all places. New callers should prefer
/// [`determine_width_per_place`] for memory-density gains on mixed-bound
/// nets.
pub(crate) fn determine_width_with_invariants(
    net: &PetriNet,
    invariants: &[PInvariant],
) -> TokenWidth {
    let per_place = determine_width_per_place(net, invariants);
    max_width(&per_place).unwrap_or_else(|| determine_width(net))
}

/// Smallest [`TokenWidth`] whose `max_value()` >= `bound`.
fn width_for_bound(bound: u64) -> TokenWidth {
    if u8::try_from(bound).is_ok() {
        TokenWidth::U8
    } else if u16::try_from(bound).is_ok() {
        TokenWidth::U16
    } else {
        TokenWidth::U64
    }
}

/// Maximum (worst-case) width across a per-place width vector.
///
/// Returns `None` if the vector is empty.
fn max_width(widths: &[TokenWidth]) -> Option<TokenWidth> {
    widths.iter().copied().max_by_key(|w| w.bytes())
}

/// Determine a per-place [`TokenWidth`] vector for compact marking storage.
///
/// For each place, picks the smallest width whose `max_value()` >= the
/// place's structural bound from `invariants`. Uncovered places fall back
/// to the net-wide [`determine_width`] result (which itself is U64 for
/// non-conserving nets without bounds, or token-total-derived for
/// conserving nets).
///
/// Returns a vector of length `net.num_places()` indexed by place index.
pub(crate) fn determine_width_per_place(
    net: &PetriNet,
    invariants: &[PInvariant],
) -> Vec<TokenWidth> {
    let n = net.num_places();
    if n == 0 {
        return Vec::new();
    }
    let fallback = determine_width(net);
    if invariants.is_empty() {
        return vec![fallback; n];
    }

    let mut out = Vec::with_capacity(n);
    for place in 0..n {
        let chosen = match structural_place_bound(invariants, place) {
            Some(bound) => {
                let invariant_w = width_for_bound(bound);
                // Never widen beyond the net-wide fallback: a tighter
                // invariant-derived width supersedes only when it saves bytes.
                if invariant_w.bytes() < fallback.bytes() {
                    invariant_w
                } else {
                    fallback
                }
            }
            None => fallback,
        };
        out.push(chosen);
    }
    out
}

// ---------------------------------------------------------------------------
// Layer 2: sub-byte bit-packing
// ---------------------------------------------------------------------------

/// Sub-byte token storage width, composed on top of [`TokenWidth`].
///
/// Layer 2 of the per-place packing pipeline. Each variant stores token
/// counts in fewer than 8 bits for places whose structural bound permits:
///
/// * `B1` — 1 bit, for 1-safe places (bound <= 1).
/// * `B3` — 3 bits, for places with bound <= 7.
/// * `B6` — 6 bits, for places with bound <= 63.
/// * `Byte(w)` — fall through to the Layer 1 byte-aligned width `w`.
///
/// When packing a marking, bit-widths are concatenated into a packed byte
/// stream with no per-place alignment. The encoder/decoder ([`codec`]) use
/// a shift register to span byte boundaries.
///
/// **Hot-path note:** the variant is computed offline (in the Farkas /
/// P-invariant phase) and stored in `MarkingConfig::bit_widths_per_place`,
/// so the BFS loop only matches on the precomputed enum value.
///
/// [`codec`]: super::codec
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BitWidth {
    /// Exact bit-packing width (1 to 63 bits). Denser than U8/U16/U64 fallbacks.
    Packed(u8),
    /// Byte-aligned fallback to a Layer 1 [`TokenWidth`].
    Byte(TokenWidth),
}

impl BitWidth {
    /// Number of bits consumed by one token slot at this width.
    #[must_use]
    #[inline]
    pub fn bits(self) -> u32 {
        match self {
            Self::Packed(bits) => bits as u32,
            Self::Byte(w) => (w.bytes() as u32) * 8,
        }
    }

    /// True if this width is byte-aligned (Layer 1 fallback).
    #[must_use]
    #[inline]
    pub fn is_byte_aligned(self) -> bool {
        matches!(self, Self::Byte(_))
    }
}

/// Smallest [`BitWidth`] whose representable range >= `bound`, given the
/// Layer 1 byte-width fallback for higher bounds.
#[inline]
pub(crate) fn bit_width_for_bound(bound: u64, layer1: TokenWidth) -> BitWidth {
    if bound == 0 {
        return BitWidth::Packed(1);
    }
    let required_bits = 64 - bound.leading_zeros();
    let layer1_bits = layer1.bytes() as u32 * 8;

    if required_bits < layer1_bits {
        BitWidth::Packed(required_bits as u8)
    } else {
        BitWidth::Byte(layer1)
    }
}

/// Determine a per-place [`BitWidth`] vector composed on top of Layer 1
/// `width_per_place`.
///
/// For each place with a structural bound from `invariants`, picks the
/// smallest [`BitWidth`] whose representable range covers it. Uncovered
/// places fall through to `BitWidth::Byte(width_per_place[p])`.
///
/// Invariant: `bit_widths_per_place[p].bits() <=
/// width_per_place[p].bytes() * 8` for every place. Layer 2 never widens
/// Layer 1.
pub(crate) fn determine_bit_widths_per_place(
    net: &PetriNet,
    invariants: &[PInvariant],
    width_per_place: &[TokenWidth],
) -> Vec<BitWidth> {
    let n = net.num_places();
    debug_assert_eq!(width_per_place.len(), n);
    if n == 0 {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(n);
    for place in 0..n {
        let layer1 = width_per_place[place];
        let chosen = match structural_place_bound(invariants, place) {
            Some(bound) => {
                let bw = bit_width_for_bound(bound, layer1);
                // Never widen beyond Layer 1: only shrink to a sub-byte
                // width when the bound permits AND the result is narrower
                // in bits than Layer 1.
                if bw.bits() < layer1.bytes() as u32 * 8 {
                    bw
                } else {
                    BitWidth::Byte(layer1)
                }
            }
            None => BitWidth::Byte(layer1),
        };
        out.push(chosen);
    }
    out
}
