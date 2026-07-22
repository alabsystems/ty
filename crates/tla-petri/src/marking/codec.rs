// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

use super::layout::MarkingConfig;
use super::width::{BitWidth, TokenWidth};
use crate::invariant::ImpliedPlace;

fn reserve_packed_bytes(buf: &mut Vec<u8>, width: TokenWidth, values: usize) {
    buf.reserve(values * width.bytes());
}

fn encode_selected_tokens<'a>(
    values: impl Iterator<Item = &'a u64>,
    width: TokenWidth,
    count: usize,
    buf: &mut Vec<u8>,
) {
    buf.clear();
    reserve_packed_bytes(buf, width, count);

    let mut values = values;
    match width {
        TokenWidth::U8 => {
            for &value in values.by_ref() {
                buf.push(value as u8);
            }
        }
        TokenWidth::U16 => {
            for &value in values.by_ref() {
                buf.extend_from_slice(&(value as u16).to_le_bytes());
            }
        }
        TokenWidth::U64 => {
            for &value in values.by_ref() {
                buf.extend_from_slice(&value.to_le_bytes());
            }
        }
    }
}

fn decode_selected_slots(
    packed: &[u8],
    width: TokenWidth,
    target_indices: impl Iterator<Item = usize>,
    out: &mut [u64],
) {
    let mut target_indices = target_indices;
    match width {
        TokenWidth::U8 => {
            for (byte_idx, place) in target_indices.by_ref().enumerate() {
                out[place] = packed[byte_idx] as u64;
            }
        }
        TokenWidth::U16 => {
            let mut byte_idx = 0;
            for place in target_indices.by_ref() {
                out[place] = u16::from_le_bytes([packed[byte_idx], packed[byte_idx + 1]]) as u64;
                byte_idx += 2;
            }
        }
        TokenWidth::U64 => {
            let mut byte_idx = 0;
            for place in target_indices.by_ref() {
                let chunk = &packed[byte_idx..byte_idx + 8];
                out[place] = u64::from_le_bytes(chunk.try_into().expect("8-byte chunk"));
                byte_idx += 8;
            }
        }
    }
}

fn encode_scalar(value: u64, width: TokenWidth, buf: &mut Vec<u8>) {
    match width {
        TokenWidth::U8 => buf.push(value as u8),
        TokenWidth::U16 => buf.extend_from_slice(&(value as u16).to_le_bytes()),
        TokenWidth::U64 => buf.extend_from_slice(&value.to_le_bytes()),
    }
}

fn decode_scalar(packed: &[u8], byte_idx: &mut usize, width: TokenWidth) -> u64 {
    match width {
        TokenWidth::U8 => {
            let value = packed[*byte_idx] as u64;
            *byte_idx += 1;
            value
        }
        TokenWidth::U16 => {
            let value = u16::from_le_bytes([packed[*byte_idx], packed[*byte_idx + 1]]) as u64;
            *byte_idx += 2;
            value
        }
        TokenWidth::U64 => {
            let chunk = &packed[*byte_idx..*byte_idx + 8];
            *byte_idx += 8;
            u64::from_le_bytes(chunk.try_into().expect("8-byte chunk"))
        }
    }
}

#[inline]
fn encode_bit_width(value: u64, width: BitWidth, bit_pos: &mut usize, buf: &mut [u8]) {
    let bits = width.bits() as usize;
    let mut byte_idx = *bit_pos / 8;
    let mut bit_offset = *bit_pos % 8;

    let mask = if bits == 64 { !0 } else { (1u64 << bits) - 1 };
    let mut v = value & mask;
    let mut bits_left = bits;

    while bits_left > 0 {
        let bits_this_byte = (8 - bit_offset).min(bits_left);
        let byte_mask = ((1u16 << bits_this_byte) - 1) as u8;
        buf[byte_idx] |= ((v as u8) & byte_mask) << bit_offset;

        v >>= bits_this_byte;
        bits_left -= bits_this_byte;
        byte_idx += 1;
        bit_offset = 0;
    }
    *bit_pos += bits;
}

#[inline]
fn decode_bit_width(packed: &[u8], width: BitWidth, bit_pos: &mut usize) -> u64 {
    let bits = width.bits() as usize;
    let mut byte_idx = *bit_pos / 8;
    let mut bit_offset = *bit_pos % 8;

    let mut value = 0u64;
    let mut bits_left = bits;
    let mut out_shift = 0;

    while bits_left > 0 {
        let bits_this_byte = (8 - bit_offset).min(bits_left);
        let byte_mask = ((1u16 << bits_this_byte) - 1) as u8;

        let chunk = (packed[byte_idx] >> bit_offset) & byte_mask;
        value |= (chunk as u64) << out_shift;

        out_shift += bits_this_byte;
        bits_left -= bits_this_byte;
        byte_idx += 1;
        bit_offset = 0;
    }
    *bit_pos += bits;
    value
}

fn encode_selected_tokens_per_place(marking: &[u64], config: &MarkingConfig, buf: &mut Vec<u8>) {
    buf.clear();
    buf.reserve(config.packed_bytes);
    if config.all_byte_aligned() {
        for place in config.stored_places() {
            encode_scalar(marking[place], config.width_per_place[place], buf);
        }
        return;
    }

    buf.resize(config.packed_bytes, 0);
    let mut bit_pos = 0usize;
    for place in config.stored_places() {
        encode_bit_width(
            marking[place],
            config.bit_widths_per_place[place],
            &mut bit_pos,
            buf.as_mut_slice(),
        );
    }
}

fn decode_selected_tokens_per_place(packed: &[u8], config: &MarkingConfig, out: &mut [u64]) {
    if config.all_byte_aligned() {
        let mut byte_idx = 0usize;
        for place in config.stored_places() {
            out[place] = decode_scalar(packed, &mut byte_idx, config.width_per_place[place]);
        }
        return;
    }

    let mut bit_pos = 0usize;
    for place in config.stored_places() {
        out[place] = decode_bit_width(packed, config.bit_widths_per_place[place], &mut bit_pos);
    }
}

/// Pack a u64 marking into a compact byte representation.
///
/// Reuses `buf` to avoid allocation in the hot BFS loop.
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn pack_marking(marking: &[u64], width: TokenWidth, buf: &mut Vec<u8>) {
    encode_selected_tokens(marking.iter(), width, marking.len(), buf);
}

/// Unpack a compact byte representation back to u64 token values.
///
/// Reuses `out` to avoid allocation in the hot BFS loop.
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn unpack_marking(
    packed: &[u8],
    width: TokenWidth,
    num_places: usize,
    out: &mut Vec<u64>,
) {
    out.clear();
    out.resize(num_places, 0);
    decode_selected_slots(packed, width, 0..num_places, out.as_mut_slice());
}

/// Pack a full marking into compact bytes, excluding implied places.
pub(crate) fn pack_marking_config(marking: &[u64], config: &MarkingConfig, buf: &mut Vec<u8>) {
    encode_selected_tokens_per_place(marking, config, buf);
}

/// Unpack compact bytes and reconstruct implied places to produce a full marking.
pub(crate) fn unpack_marking_config(packed: &[u8], config: &MarkingConfig, out: &mut Vec<u64>) {
    out.clear();
    out.resize(config.num_places, 0);
    decode_selected_tokens_per_place(packed, config, out.as_mut_slice());
    reconstruct_implied_places(out, config.implied_places());
}

/// Reconstruct implied places in a full marking vector.
///
/// For each implied place p:
///   `m(p) = (C - sum(w_q * m(q))) / d`
///
/// Division is exact for reachable markings by the P-invariant property.
pub(crate) fn reconstruct_implied_places(marking: &mut [u64], implied: &[ImpliedPlace]) {
    for implied_place in implied {
        let reconstruction = &implied_place.reconstruction;
        let sum = reconstruction
            .terms
            .iter()
            .fold(0u64, |acc, &(place, weight)| {
                let token = *marking.get(place).unwrap_or_else(|| {
                    panic!("P-invariant reconstruction: term references out-of-range place {place}")
                });
                let term = weight.checked_mul(token).unwrap_or_else(|| {
                    panic!(
                        "P-invariant reconstruction overflow for place {:?}: \
                         weight {weight} * marking[{place}] {token} exceeded u64::MAX",
                        implied_place.place,
                    )
                });
                acc.checked_add(term).unwrap_or_else(|| {
                    panic!(
                        "P-invariant reconstruction overflow for place {:?}: \
                         weighted sum exceeded u64::MAX",
                        implied_place.place,
                    )
                })
            });
        let numerator = reconstruction.constant.checked_sub(sum).unwrap_or_else(|| {
            panic!(
                "P-invariant reconstruction underflow for place {:?}: constant {} < weighted_sum {sum}",
                implied_place.place,
                reconstruction.constant,
            )
        });
        let divisor = reconstruction.divisor;
        assert!(
            divisor != 0,
            "P-invariant reconstruction: zero divisor for place {:?}",
            implied_place.place,
        );
        assert!(
            numerator % divisor == 0,
            "P-invariant reconstruction: non-exact division for place {:?} \
             (numerator={numerator}, divisor={divisor})",
            implied_place.place,
        );
        let slot = marking.get_mut(implied_place.place).unwrap_or_else(|| {
            panic!(
                "P-invariant reconstruction: implied place {:?} is out of range",
                implied_place.place,
            )
        });
        *slot = numerator / divisor;
    }
}
