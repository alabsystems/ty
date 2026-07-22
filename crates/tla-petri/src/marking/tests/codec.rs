// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

use crate::marking::{
    bit_width_for_bound, determine_bit_widths_per_place, pack_marking, pack_marking_config,
    unpack_marking, unpack_marking_config, BitWidth, MarkingConfig, TokenWidth,
};

#[test]
fn test_pack_unpack_u8_roundtrip() {
    let marking = vec![0, 1, 255, 42, 0];
    let mut buf = Vec::new();
    pack_marking(&marking, TokenWidth::U8, &mut buf);
    assert_eq!(buf.len(), 5);
    assert_eq!(buf, vec![0, 1, 255, 42, 0]);

    let mut out = Vec::new();
    unpack_marking(&buf, TokenWidth::U8, 5, &mut out);
    assert_eq!(out, marking);
}

#[test]
fn test_pack_unpack_u16_roundtrip() {
    let marking = vec![0, 300, 65_535, 1000];
    let mut buf = Vec::new();
    pack_marking(&marking, TokenWidth::U16, &mut buf);
    assert_eq!(buf.len(), 8);

    let mut out = Vec::new();
    unpack_marking(&buf, TokenWidth::U16, 4, &mut out);
    assert_eq!(out, marking);
}

#[test]
fn test_pack_unpack_u64_roundtrip() {
    let marking = vec![0, 1, u64::MAX, 1_000_000];
    let mut buf = Vec::new();
    pack_marking(&marking, TokenWidth::U64, &mut buf);
    assert_eq!(buf.len(), 32);

    let mut out = Vec::new();
    unpack_marking(&buf, TokenWidth::U64, 4, &mut out);
    assert_eq!(out, marking);
}

#[test]
fn test_pack_empty_marking() {
    let marking: Vec<u64> = vec![];
    let mut buf = Vec::new();
    pack_marking(&marking, TokenWidth::U8, &mut buf);
    assert!(buf.is_empty());

    let mut out = Vec::new();
    unpack_marking(&buf, TokenWidth::U8, 0, &mut out);
    assert!(out.is_empty());
}

#[test]
fn test_pack_reuses_buffer() {
    let m1 = vec![1, 2, 3];
    let m2 = vec![4, 5, 6];
    let mut buf = Vec::new();

    pack_marking(&m1, TokenWidth::U8, &mut buf);
    let ptr1 = buf.as_ptr();
    assert_eq!(buf, vec![1, 2, 3]);

    pack_marking(&m2, TokenWidth::U8, &mut buf);
    assert_eq!(buf, vec![4, 5, 6]);
    assert_eq!(buf.as_ptr(), ptr1);
}

#[test]
fn test_distinct_markings_distinct_packed_u8() {
    let m1 = vec![1, 0];
    let m2 = vec![0, 1];
    let mut buf1 = Vec::new();
    let mut buf2 = Vec::new();
    pack_marking(&m1, TokenWidth::U8, &mut buf1);
    pack_marking(&m2, TokenWidth::U8, &mut buf2);
    assert_ne!(buf1, buf2);
}

#[test]
fn test_identical_markings_identical_packed_u16() {
    let marking = vec![300, 100];
    let mut buf1 = Vec::new();
    let mut buf2 = Vec::new();
    pack_marking(&marking, TokenWidth::U16, &mut buf1);
    pack_marking(&marking, TokenWidth::U16, &mut buf2);
    assert_eq!(buf1, buf2);
}

#[test]
fn test_marking_config_standard_matches_original() {
    let marking = vec![1, 2, 3, 4, 5];
    let config = MarkingConfig::standard(5, TokenWidth::U8);
    let mut buf_config = Vec::new();
    let mut buf_plain = Vec::new();

    pack_marking_config(&marking, &config, &mut buf_config);
    pack_marking(&marking, TokenWidth::U8, &mut buf_plain);
    assert_eq!(buf_config, buf_plain);

    let mut out_config = Vec::new();
    let mut out_plain = Vec::new();
    unpack_marking_config(&buf_config, &config, &mut out_config);
    unpack_marking(&buf_plain, TokenWidth::U8, 5, &mut out_plain);
    assert_eq!(out_config, out_plain);
}

#[test]
fn test_pack_unpack_per_place_width_roundtrip() {
    // Mixed widths: U8 (1B), U16 (2B), U64 (8B), U16 (2B), U8 (1B) = 14 bytes.
    let widths = vec![
        TokenWidth::U8,
        TokenWidth::U16,
        TokenWidth::U64,
        TokenWidth::U16,
        TokenWidth::U8,
    ];
    let marking = vec![7u64, 300, u64::MAX, 65_000, 250];
    let config = MarkingConfig::from_parts(5, widths, vec![]);
    assert_eq!(config.packed_bytes, 1 + 2 + 8 + 2 + 1);
    assert_eq!(
        config.width,
        TokenWidth::U64,
        "scalar width = max across places"
    );

    let mut buf = Vec::new();
    pack_marking_config(&marking, &config, &mut buf);
    assert_eq!(buf.len(), config.packed_bytes);

    let mut out = Vec::new();
    unpack_marking_config(&buf, &config, &mut out);
    assert_eq!(out, marking);
}

#[test]
fn test_pack_unpack_implied_place_with_per_place_width() {
    use crate::invariant::{ImpliedPlace, ImpliedPlaceReconstruction};
    // 4 places, mixed widths, place 2 is implied (excluded from storage).
    // Storage order: p0 (U8), p1 (U16), p3 (U64) = 1 + 2 + 8 = 11 bytes.
    let widths = vec![
        TokenWidth::U8,
        TokenWidth::U16,
        TokenWidth::U8, // implied place's width is unused for storage
        TokenWidth::U64,
    ];
    let implied = vec![ImpliedPlace {
        place: 2,
        reconstruction: ImpliedPlaceReconstruction {
            constant: 500,
            divisor: 1,
            terms: vec![(0, 1), (1, 1), (3, 1)],
        },
    }];
    let config = MarkingConfig::from_parts(4, widths, implied);
    assert_eq!(config.packed_len, 3);
    assert_eq!(config.packed_bytes, 1 + 2 + 8);

    let marking = vec![100u64, 300, 50, 50];
    // Sanity: invariant holds (100 + 300 + 50 + 50 = 500).

    let mut buf = Vec::new();
    pack_marking_config(&marking, &config, &mut buf);
    assert_eq!(buf.len(), 11);

    let mut out = Vec::new();
    unpack_marking_config(&buf, &config, &mut out);
    assert_eq!(out, marking, "implied place reconstructed correctly");
}

#[test]
fn test_per_place_width_uses_smallest_fitting() {
    use crate::invariant::PInvariant;
    use crate::marking::{
        determine_width, determine_width_per_place, determine_width_with_invariants,
    };
    use crate::petri_net::{Arc, PetriNet, PlaceIdx, PlaceInfo, TransitionInfo};

    // 1-safe place (bound 1) -> U8; mid-range (bound 1000) -> U16; large
    // (bound 100_000) -> U64. We use a non-conserving net so the global
    // fallback would be U64, and supply hand-rolled P-invariants that
    // bound each place independently. The per-place choice must shrink
    // each below the fallback when the invariant allows it.
    let net = PetriNet {
        name: None,
        places: (0..3)
            .map(|i| PlaceInfo {
                id: format!("p{i}"),
                name: None,
            })
            .collect(),
        // Non-conserving sink transition forces determine_width -> U64.
        transitions: vec![TransitionInfo {
            id: "t0".into(),
            name: None,
            inputs: vec![Arc {
                place: PlaceIdx(0),
                weight: 1,
            }],
            outputs: vec![],
        }],
        initial_marking: vec![1, 0, 0],
    };
    assert_eq!(determine_width(&net), TokenWidth::U64);

    // Hand-rolled per-place invariants: y_p = 1, token_count = bound.
    let invariants = vec![
        PInvariant {
            weights: vec![(0u32, 1u64)],
            token_count: 1,
        },
        PInvariant {
            weights: vec![(1u32, 1u64)],
            token_count: 1000,
        },
        PInvariant {
            weights: vec![(2u32, 1u64)],
            token_count: 100_000,
        },
    ];

    let per_place = determine_width_per_place(&net, &invariants);
    assert_eq!(
        per_place,
        vec![TokenWidth::U8, TokenWidth::U16, TokenWidth::U64]
    );

    // Scalar fallback is the max across places (still U64 here).
    assert_eq!(
        determine_width_with_invariants(&net, &invariants),
        TokenWidth::U64
    );
}

// ---------------------------------------------------------------------------
// Layer 2 (sub-byte bit-packing) tests
// ---------------------------------------------------------------------------

/// Build a `MarkingConfig` with explicit Layer-1 widths AND Layer-2 bit
/// widths, no implied places.
fn config_with_bit_widths(
    num_places: usize,
    width_per_place: Vec<TokenWidth>,
    bit_widths: Vec<BitWidth>,
) -> MarkingConfig {
    MarkingConfig::from_parts_with_bit_widths(num_places, width_per_place, bit_widths, vec![])
}

#[test]
fn test_bit_width_selection_matches_structural_bound() {
    // Boundaries: <=1 -> B1, <=7 -> B3, <=63 -> B6, otherwise -> Byte(layer1).
    assert_eq!(bit_width_for_bound(0, TokenWidth::U8), BitWidth::Packed(1));
    assert_eq!(bit_width_for_bound(1, TokenWidth::U8), BitWidth::Packed(1));
    assert_eq!(bit_width_for_bound(2, TokenWidth::U8), BitWidth::Packed(2));
    assert_eq!(bit_width_for_bound(7, TokenWidth::U8), BitWidth::Packed(3));
    assert_eq!(bit_width_for_bound(8, TokenWidth::U8), BitWidth::Packed(4));
    assert_eq!(bit_width_for_bound(63, TokenWidth::U8), BitWidth::Packed(6));
    assert_eq!(bit_width_for_bound(64, TokenWidth::U8), BitWidth::Packed(7));
    assert_eq!(
        bit_width_for_bound(255, TokenWidth::U8),
        BitWidth::Byte(TokenWidth::U8)
    );
    // Layer 1 width carried through for higher bounds:
    assert_eq!(
        bit_width_for_bound(1000, TokenWidth::U16),
        BitWidth::Packed(10)
    );
    assert_eq!(
        bit_width_for_bound(1_000_000, TokenWidth::U64),
        BitWidth::Packed(20)
    );

    // Bit counts.
    assert_eq!(BitWidth::Packed(1).bits(), 1);
    assert_eq!(BitWidth::Packed(3).bits(), 3);
    assert_eq!(BitWidth::Packed(6).bits(), 6);
    assert_eq!(BitWidth::Byte(TokenWidth::U8).bits(), 8);
    assert_eq!(BitWidth::Byte(TokenWidth::U16).bits(), 16);
    assert_eq!(BitWidth::Byte(TokenWidth::U64).bits(), 64);
}

#[test]
fn test_pack_unpack_bit_widths_roundtrip_all_b1() {
    // 10 one-safe places: 10 bits -> ceil(10/8) = 2 bytes.
    let bit_widths = vec![BitWidth::Packed(1); 10];
    let widths = vec![TokenWidth::U8; 10];
    let config = config_with_bit_widths(10, widths, bit_widths);
    assert_eq!(config.packed_bytes, 2);

    let marking = vec![1u64, 0, 1, 1, 0, 0, 1, 0, 1, 1];
    let mut buf = Vec::new();
    pack_marking_config(&marking, &config, &mut buf);
    assert_eq!(buf.len(), 2);

    let mut out = Vec::new();
    unpack_marking_config(&buf, &config, &mut out);
    assert_eq!(out, marking);
}

#[test]
fn test_pack_unpack_bit_widths_roundtrip_mixed() {
    // Mixed sub-byte and byte widths spanning byte boundaries:
    //   B1 (1) + B3 (3) + B6 (6) + B1 (1) + U8 (8) + B3 (3) + U16 (16)
    //   = 38 bits -> ceil(38/8) = 5 bytes.
    let bit_widths = vec![
        BitWidth::Packed(1),
        BitWidth::Packed(3),
        BitWidth::Packed(6),
        BitWidth::Packed(1),
        BitWidth::Byte(TokenWidth::U8),
        BitWidth::Packed(3),
        BitWidth::Byte(TokenWidth::U16),
    ];
    let widths = vec![
        TokenWidth::U8,
        TokenWidth::U8,
        TokenWidth::U8,
        TokenWidth::U8,
        TokenWidth::U8,
        TokenWidth::U8,
        TokenWidth::U16,
    ];
    let config = config_with_bit_widths(7, widths, bit_widths);
    assert_eq!(config.packed_bytes, 5);
    assert!(!config.all_byte_aligned());

    let marking = vec![1u64, 5, 42, 0, 200, 7, 60_000];
    let mut buf = Vec::new();
    pack_marking_config(&marking, &config, &mut buf);
    assert_eq!(buf.len(), config.packed_bytes);

    let mut out = Vec::new();
    unpack_marking_config(&buf, &config, &mut out);
    assert_eq!(out, marking);
}

#[test]
fn test_pack_unpack_bit_widths_max_values_per_width() {
    // Each width at its representable max:
    //   B1: 1, B3: 7, B6: 63, U8: 255, U16: 65535
    let bit_widths = vec![
        BitWidth::Packed(1),
        BitWidth::Packed(3),
        BitWidth::Packed(6),
        BitWidth::Byte(TokenWidth::U8),
        BitWidth::Byte(TokenWidth::U16),
    ];
    let widths = vec![
        TokenWidth::U8,
        TokenWidth::U8,
        TokenWidth::U8,
        TokenWidth::U8,
        TokenWidth::U16,
    ];
    let config = config_with_bit_widths(5, widths, bit_widths);
    let marking = vec![1u64, 7, 63, 255, 65_535];

    let mut buf = Vec::new();
    pack_marking_config(&marking, &config, &mut buf);
    let mut out = Vec::new();
    unpack_marking_config(&buf, &config, &mut out);
    assert_eq!(out, marking);
}

#[test]
fn test_pack_unpack_bit_widths_with_implied_places() {
    use crate::invariant::{ImpliedPlace, ImpliedPlaceReconstruction};
    // 4 places, place 2 is implied. Stored places (0, 1, 3) use bit-packed
    // widths: B1 + B3 + B6 = 10 bits -> ceil(10/8) = 2 bytes.
    let widths = vec![
        TokenWidth::U8,
        TokenWidth::U8,
        TokenWidth::U8, // implied: width unused for storage
        TokenWidth::U8,
    ];
    let bit_widths = vec![
        BitWidth::Packed(1),
        BitWidth::Packed(3),
        BitWidth::Packed(6), // implied: bit width unused for storage
        BitWidth::Packed(6),
    ];
    let implied = vec![ImpliedPlace {
        place: 2,
        reconstruction: ImpliedPlaceReconstruction {
            constant: 50,
            divisor: 1,
            terms: vec![(0, 1), (1, 1), (3, 1)],
        },
    }];
    let config = MarkingConfig::from_parts_with_bit_widths(4, widths, bit_widths, implied);
    assert_eq!(config.packed_len, 3);
    assert_eq!(config.packed_bytes, 2);

    // Invariant: 1 + 5 + (44 implied) + 0 = 50.
    let marking = vec![1u64, 5, 44, 0];
    let mut buf = Vec::new();
    pack_marking_config(&marking, &config, &mut buf);
    assert_eq!(buf.len(), 2);

    let mut out = Vec::new();
    unpack_marking_config(&buf, &config, &mut out);
    assert_eq!(out, marking, "implied place reconstructed under Layer 2");
}

#[test]
fn test_layer1_only_path_still_works() {
    // Regression guard: when every stored place is byte-aligned, the codec
    // takes the Layer-1 fast path (no bit-seek), and the output matches the
    // pre-Layer-2 byte-aligned encoding exactly.
    let widths = vec![
        TokenWidth::U8,
        TokenWidth::U16,
        TokenWidth::U64,
        TokenWidth::U16,
    ];
    let marking = vec![7u64, 300, 1_000_000_000_000, 65_000];

    // Layer-2 config but all slots byte-aligned (fallback to layer1).
    let bit_widths = widths.iter().copied().map(BitWidth::Byte).collect();
    let config = config_with_bit_widths(4, widths.clone(), bit_widths);
    assert!(config.all_byte_aligned());
    assert_eq!(config.packed_bytes, 1 + 2 + 8 + 2);

    let mut buf_layer2 = Vec::new();
    pack_marking_config(&marking, &config, &mut buf_layer2);
    assert_eq!(buf_layer2.len(), 13);

    // Layer-1-only config (no bit_widths argument -> defaults to all-byte).
    let config_layer1 = MarkingConfig::from_parts(4, widths, vec![]);
    let mut buf_layer1 = Vec::new();
    pack_marking_config(&marking, &config_layer1, &mut buf_layer1);

    assert_eq!(
        buf_layer1, buf_layer2,
        "byte-aligned fast path output is bit-exact regardless of Layer 2 enabled"
    );

    let mut out = Vec::new();
    unpack_marking_config(&buf_layer2, &config, &mut out);
    assert_eq!(out, marking);
}

#[test]
fn test_determine_bit_widths_per_place_uses_smallest() {
    use crate::invariant::PInvariant;
    use crate::petri_net::{Arc, PetriNet, PlaceIdx, PlaceInfo, TransitionInfo};

    // 4 places with bounds 1, 5, 50, 1000 — should map to B1, B3, B6, Byte(layer1).
    let net = PetriNet {
        name: None,
        places: (0..4)
            .map(|i| PlaceInfo {
                id: format!("p{i}"),
                name: None,
            })
            .collect(),
        transitions: vec![TransitionInfo {
            id: "t0".into(),
            name: None,
            inputs: vec![Arc {
                place: PlaceIdx(0),
                weight: 1,
            }],
            outputs: vec![],
        }],
        initial_marking: vec![1, 5, 50, 1000],
    };
    let invariants = vec![
        PInvariant {
            weights: vec![(0u32, 1u64)],
            token_count: 1,
        },
        PInvariant {
            weights: vec![(1u32, 1u64)],
            token_count: 5,
        },
        PInvariant {
            weights: vec![(2u32, 1u64)],
            token_count: 50,
        },
        PInvariant {
            weights: vec![(3u32, 1u64)],
            token_count: 1000,
        },
    ];
    let widths = vec![
        TokenWidth::U8,
        TokenWidth::U8,
        TokenWidth::U8,
        TokenWidth::U16,
    ];
    let bit_widths = determine_bit_widths_per_place(&net, &invariants, &widths);
    assert_eq!(
        bit_widths,
        vec![
            BitWidth::Packed(1),
            BitWidth::Packed(3),
            BitWidth::Packed(6),
            BitWidth::Packed(10),
        ]
    );
}

#[test]
fn test_bit_widths_per_place_uncovered_falls_back_to_layer1() {
    use crate::petri_net::{Arc, PetriNet, PlaceIdx, PlaceInfo, TransitionInfo};

    let net = PetriNet {
        name: None,
        places: (0..3)
            .map(|i| PlaceInfo {
                id: format!("p{i}"),
                name: None,
            })
            .collect(),
        transitions: vec![TransitionInfo {
            id: "t0".into(),
            name: None,
            inputs: vec![],
            outputs: vec![Arc {
                place: PlaceIdx(0),
                weight: 1,
            }],
        }],
        initial_marking: vec![0, 0, 0],
    };
    // No invariants -> every place uncovered -> Byte(layer1).
    let widths = vec![TokenWidth::U8, TokenWidth::U16, TokenWidth::U64];
    let bit_widths = determine_bit_widths_per_place(&net, &[], &widths);
    assert_eq!(
        bit_widths,
        vec![
            BitWidth::Byte(TokenWidth::U8),
            BitWidth::Byte(TokenWidth::U16),
            BitWidth::Byte(TokenWidth::U64),
        ]
    );
}

#[test]
fn test_pack_unpack_bit_widths_large_synthetic_net() {
    // 50 places: 30 one-safe (B1), 10 small (B3), 10 mid (B6).
    // Total: 30 + 30 + 60 = 120 bits -> 15 bytes (vs Layer-1 50 bytes).
    let mut bit_widths = Vec::with_capacity(50);
    let mut widths = Vec::with_capacity(50);
    let mut marking = Vec::with_capacity(50);
    for i in 0..30 {
        bit_widths.push(BitWidth::Packed(1));
        widths.push(TokenWidth::U8);
        marking.push((i % 2) as u64);
    }
    for i in 0..10 {
        bit_widths.push(BitWidth::Packed(3));
        widths.push(TokenWidth::U8);
        marking.push((i % 8) as u64);
    }
    for i in 0..10 {
        bit_widths.push(BitWidth::Packed(6));
        widths.push(TokenWidth::U8);
        marking.push((i * 6) as u64);
    }
    let config = config_with_bit_widths(50, widths, bit_widths);
    assert_eq!(config.packed_bytes, 15);

    let mut buf = Vec::new();
    pack_marking_config(&marking, &config, &mut buf);
    assert_eq!(buf.len(), 15);

    let mut out = Vec::new();
    unpack_marking_config(&buf, &config, &mut out);
    assert_eq!(out, marking);
}

#[test]
fn test_bit_widths_never_widen_layer1() {
    use crate::invariant::PInvariant;
    use crate::petri_net::{Arc, PetriNet, PlaceIdx, PlaceInfo, TransitionInfo};

    // Tight invariant on a U64-fallback place must still pick a sub-byte
    // width (B6) — and a loose invariant on a U8 place must NOT widen
    // beyond the Layer 1 byte.
    let net = PetriNet {
        name: None,
        places: (0..2)
            .map(|i| PlaceInfo {
                id: format!("p{i}"),
                name: None,
            })
            .collect(),
        transitions: vec![TransitionInfo {
            id: "t0".into(),
            name: None,
            inputs: vec![Arc {
                place: PlaceIdx(0),
                weight: 1,
            }],
            outputs: vec![],
        }],
        initial_marking: vec![0, 0],
    };
    // Place 0: bound 50, Layer 1 says U64 (e.g. mixed inputs); Layer 2 -> B6.
    // Place 1: bound 200, Layer 1 says U8; Layer 2 must NOT shrink past U8.
    let invariants = vec![
        PInvariant {
            weights: vec![(0u32, 1u64)],
            token_count: 50,
        },
        PInvariant {
            weights: vec![(1u32, 1u64)],
            token_count: 200,
        },
    ];
    let widths = vec![TokenWidth::U64, TokenWidth::U8];
    let bit_widths = determine_bit_widths_per_place(&net, &invariants, &widths);
    // Place 0: B6 (bound 50 <= 63), narrower than U64 — picked.
    assert_eq!(bit_widths[0], BitWidth::Packed(6));
    // Place 1: bound 200, falls to Byte(U8). Never widens.
    assert_eq!(bit_widths[1], BitWidth::Byte(TokenWidth::U8));
}

#[test]
fn test_hot_path_bit_packed_within_budget_vs_byte_aligned() {
    use std::time::Instant;

    // 50-place synthetic net: 40 one-safe + 10 byte-aligned.
    // The bit-packed version compresses 40*1 + 10*8 = 120 bits = 15 bytes
    // (vs byte-aligned 40*8 + 10*8 = 50 bytes).
    let mut bit_widths_packed = Vec::with_capacity(50);
    let mut bit_widths_byte = Vec::with_capacity(50);
    let mut widths = Vec::with_capacity(50);
    let mut marking = Vec::with_capacity(50);
    for i in 0..40 {
        bit_widths_packed.push(BitWidth::Packed(1));
        bit_widths_byte.push(BitWidth::Byte(TokenWidth::U8));
        widths.push(TokenWidth::U8);
        marking.push((i % 2) as u64);
    }
    for i in 0..10 {
        bit_widths_packed.push(BitWidth::Byte(TokenWidth::U8));
        bit_widths_byte.push(BitWidth::Byte(TokenWidth::U8));
        widths.push(TokenWidth::U8);
        marking.push((i * 13 % 256) as u64);
    }
    let cfg_packed =
        MarkingConfig::from_parts_with_bit_widths(50, widths.clone(), bit_widths_packed, vec![]);
    let cfg_byte = MarkingConfig::from_parts_with_bit_widths(50, widths, bit_widths_byte, vec![]);
    assert!(!cfg_packed.all_byte_aligned());
    assert!(cfg_byte.all_byte_aligned());

    const N: usize = 10_000;

    // Warm the buffer.
    let mut buf = Vec::new();
    pack_marking_config(&marking, &cfg_packed, &mut buf);
    pack_marking_config(&marking, &cfg_byte, &mut buf);

    let mut buf_a = Vec::new();
    let t0 = Instant::now();
    for _ in 0..N {
        pack_marking_config(&marking, &cfg_byte, &mut buf_a);
    }
    let byte_elapsed = t0.elapsed();

    let mut buf_b = Vec::new();
    let t1 = Instant::now();
    for _ in 0..N {
        pack_marking_config(&marking, &cfg_packed, &mut buf_b);
    }
    let packed_elapsed = t1.elapsed();

    // Sanity: packed result is shorter (compression actually happened).
    assert!(buf_b.len() < buf_a.len());

    // Hot-path budget: bit-packed encode must stay within ~3x of byte-aligned
    // wall-clock on 10000 markings. We use 3x (generous) for CI stability —
    // the real-world hot path also amortizes against fingerprint hashing,
    // delta computation, and observer dispatch, so per-call codec time is
    // a small fraction of BFS work.
    let ratio = packed_elapsed.as_nanos() as f64 / byte_elapsed.as_nanos().max(1) as f64;
    assert!(
        ratio < 3.0,
        "bit-packed encode {:?} > 3x byte-aligned {:?} (ratio={:.2})",
        packed_elapsed,
        byte_elapsed,
        ratio,
    );

    // Decode roundtrip budget (same ratio).
    let mut out = vec![0u64; 50];
    let t2 = Instant::now();
    for _ in 0..N {
        unpack_marking_config(&buf_a, &cfg_byte, &mut out);
    }
    let byte_dec = t2.elapsed();
    let t3 = Instant::now();
    for _ in 0..N {
        unpack_marking_config(&buf_b, &cfg_packed, &mut out);
    }
    let packed_dec = t3.elapsed();
    let dec_ratio = packed_dec.as_nanos() as f64 / byte_dec.as_nanos().max(1) as f64;
    assert!(
        dec_ratio < 3.0,
        "bit-packed decode {:?} > 3x byte-aligned {:?} (ratio={:.2})",
        packed_dec,
        byte_dec,
        dec_ratio,
    );

    // Diagnostic (visible with `cargo test -- --nocapture`).
    eprintln!(
        "Layer 2 codec timing: byte enc={:?} packed enc={:?} (ratio={:.2}); \
         byte dec={:?} packed dec={:?} (ratio={:.2}); \
         packed_bytes={} vs byte={}",
        byte_elapsed,
        packed_elapsed,
        ratio,
        byte_dec,
        packed_dec,
        dec_ratio,
        buf_b.len(),
        buf_a.len(),
    );
}
