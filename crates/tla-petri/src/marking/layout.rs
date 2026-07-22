// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

use super::width::{
    determine_bit_widths_per_place, determine_width_per_place, determine_width_with_invariants,
    BitWidth, TokenWidth,
};
use crate::invariant::{compute_p_invariants, find_implied_places, ImpliedPlace};
use crate::petri_net::PetriNet;

/// Configuration for packing markings with optional implied place exclusion.
///
/// When implied places are detected via P-invariants, the packed representation
/// omits them from the hash key, reducing per-state memory. The full marking
/// (including implied places) is reconstructed on unpack.
#[derive(Debug, Clone)]
pub(crate) struct MarkingConfig {
    /// Token storage width.
    pub(crate) width: TokenWidth,
    /// Total number of places in the original net.
    pub(crate) num_places: usize,
    /// Number of places in the packed form (num_places - excluded count).
    pub(crate) packed_len: usize,
    /// Number of bytes in the packed representation.
    pub(crate) packed_bytes: usize,
    /// Per-place byte-width fallback. Entries for excluded places are retained
    /// so place indices remain stable.
    pub(crate) width_per_place: Vec<TokenWidth>,
    /// Per-place bit width. Entries for excluded places are retained so place
    /// indices remain stable.
    pub(crate) bit_widths_per_place: Vec<BitWidth>,
    /// Bitset: excluded[i] = true if place i is an implied place.
    pub(super) excluded: Vec<bool>,
    /// Implied places with reconstruction data, sorted by place index.
    pub(super) implied: Vec<ImpliedPlace>,
}

impl MarkingConfig {
    /// Create a config with no excluded places.
    pub(crate) fn standard(num_places: usize, width: TokenWidth) -> Self {
        Self {
            width,
            num_places,
            packed_len: num_places,
            packed_bytes: num_places.saturating_mul(width.bytes()),
            width_per_place: vec![width; num_places],
            bit_widths_per_place: vec![BitWidth::Byte(width); num_places],
            excluded: vec![false; num_places],
            implied: vec![],
        }
    }

    /// Create a config that excludes implied places from the packed form.
    pub(crate) fn with_implied(
        num_places: usize,
        width: TokenWidth,
        implied: Vec<ImpliedPlace>,
    ) -> Self {
        let width_per_place = vec![width; num_places];
        Self::from_parts(num_places, width_per_place, implied)
    }

    /// Create a config with explicit per-place widths and (optional) implied
    /// places. The scalar `width` field is derived as the max across
    /// `width_per_place`.
    ///
    /// The bit-width vector defaults to `BitWidth::Byte(width_per_place[p])`
    /// for every place (Layer-1-only fallback). Use
    /// [`from_parts_with_bit_widths`](Self::from_parts_with_bit_widths) to
    /// enable Layer 2 sub-byte packing.
    pub(crate) fn from_parts(
        num_places: usize,
        width_per_place: Vec<TokenWidth>,
        implied: Vec<ImpliedPlace>,
    ) -> Self {
        let bit_widths_per_place = width_per_place
            .iter()
            .copied()
            .map(BitWidth::Byte)
            .collect();
        Self::from_parts_with_bit_widths(num_places, width_per_place, bit_widths_per_place, implied)
    }

    /// Create a config with explicit per-place byte widths AND sub-byte
    /// bit widths (Layer 2). `packed_bytes` is computed as
    /// `ceil(sum(bit_widths_per_place[p].bits()) / 8)` over stored places.
    pub(crate) fn from_parts_with_bit_widths(
        num_places: usize,
        width_per_place: Vec<TokenWidth>,
        bit_widths_per_place: Vec<BitWidth>,
        implied: Vec<ImpliedPlace>,
    ) -> Self {
        debug_assert_eq!(
            width_per_place.len(),
            num_places,
            "width_per_place length must equal num_places",
        );
        debug_assert_eq!(
            bit_widths_per_place.len(),
            num_places,
            "bit_widths_per_place length must equal num_places",
        );
        let mut excluded = vec![false; num_places];
        for implied_place in &implied {
            excluded[implied_place.place] = true;
        }
        let packed_len = excluded.iter().filter(|&&is_excluded| !is_excluded).count();
        let packed_bits: u64 = bit_widths_per_place
            .iter()
            .enumerate()
            .filter(|(place, _)| !excluded[*place])
            .map(|(_, width)| u64::from(width.bits()))
            .sum();
        let packed_bytes = packed_bits.div_ceil(8) as usize;
        let width = width_per_place
            .iter()
            .copied()
            .max()
            .unwrap_or(TokenWidth::U8);
        Self {
            width,
            num_places,
            packed_len,
            packed_bytes,
            width_per_place,
            bit_widths_per_place,
            excluded,
            implied,
        }
    }

    /// True when every stored place uses byte-aligned Layer-1 encoding.
    #[must_use]
    pub(crate) fn all_byte_aligned(&self) -> bool {
        self.stored_places()
            .all(|place| self.bit_widths_per_place[place].is_byte_aligned())
    }

    /// True if any places are excluded.
    #[cfg_attr(not(test), allow(dead_code))]
    #[must_use]
    pub(crate) fn has_exclusions(&self) -> bool {
        !self.implied.is_empty()
    }

    /// Number of excluded (implied) places.
    #[cfg_attr(not(test), allow(dead_code))]
    #[must_use]
    pub(crate) fn num_excluded(&self) -> usize {
        self.implied.len()
    }

    pub(super) fn stored_places(&self) -> impl Iterator<Item = usize> + '_ {
        (0..self.num_places).filter(|&place| !self.excluded[place])
    }

    pub(super) fn implied_places(&self) -> &[ImpliedPlace] {
        &self.implied
    }

    pub(crate) fn excluded_places(&self) -> &[bool] {
        &self.excluded
    }
}

/// Fully analyzed packed-marking layout for an explored Petri net.
#[derive(Debug, Clone)]
pub(crate) struct PreparedMarking {
    pub(crate) config: MarkingConfig,
    pub(crate) width: TokenWidth,
}

impl PreparedMarking {
    /// Derive the packed-marking layout used by exploration backends.
    #[must_use]
    pub(crate) fn analyze(net: &PetriNet) -> Self {
        let invariants = compute_p_invariants(net);
        let width_per_place = determine_width_per_place(net, &invariants);
        let bit_widths_per_place =
            determine_bit_widths_per_place(net, &invariants, &width_per_place);
        let width = determine_width_with_invariants(net, &invariants);
        let implied = find_implied_places(&invariants);
        let config = MarkingConfig::from_parts_with_bit_widths(
            net.num_places(),
            width_per_place,
            bit_widths_per_place,
            implied,
        );
        Self { config, width }
    }

    #[must_use]
    pub(crate) fn packed_capacity(&self) -> usize {
        self.config.packed_bytes
    }

    #[must_use]
    pub(crate) fn packed_places(&self) -> usize {
        self.config.packed_len
    }
}
