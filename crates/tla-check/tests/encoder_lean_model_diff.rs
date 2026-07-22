// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! RUST↔LEAN DIFFERENTIAL bridge for the explicit-state certificate encoder — the FIRST
//! concrete step of the "Rust↔Lean bridge" residual named in the kernel-cert proof roadmap.
//!
//! # What the Lean proofs give us, and the gap this file closes
//!
//! The Aristotle Lean proofs in `proofs/lean/aristotle/` prove three ENCODING FORMULAS are
//! INJECTIVE:
//!   * S1 `positional_encoding_injective` — the base-`b` positional pack `∑ dᵢ·bⁱ`
//!     (`d i < b`) is injective in the digit vector `d`;
//!   * S2 `sum_two_pow_injective` — the bitmask `∑_{e∈S} 2^e` is injective in the set `S`;
//!   * S3 `pack_injective` — the shifted self-delimiting pack `∑ (aᵢ+1)·Dⁱ`
//!     (`RequestProject.pack D`, `pack D (a::rest) = (a+1) + D·pack D rest`) is injective in
//!     the list `a`.
//!
//! TY's trusted-Rust [`value_cell_encode_at`] is the IMPLEMENTATION that is SUPPOSED to compute
//! exactly those three formulas (per encoding family), turning one TLA `Value` into one `u64`
//! cell. The sibling audit `tests/encoder_injectivity_cert.rs` exhaustively checks that the Rust
//! encoder is INJECTIVE — the *property* S1–S3 prove — but it does NOT check that the Rust encoder
//! computes the SAME FUNCTION as the Lean formulas. So today the Lean⇄Rust link is only
//! property-level: "both are injective", not "they are the same map".
//!
//! # What this file adds (function-level differential agreement)
//!
//! For each encoding family, this test:
//!   1. constructs many concrete TLA `Value`s over the bounded ranges the encoder admits;
//!   2. calls the REAL Rust [`value_cell_encode_at`] to get the `(ColSort, u64)` cell;
//!   3. INDEPENDENTLY recomputes the Lean-model closed form (`∑ dᵢ·bⁱ` / `∑ 2^e` /
//!      `∑ (aᵢ+1)·Dⁱ`) directly from the value's components, using the SAME base/universe the
//!      encoder uses — the formula is hand-written MATH here, it is NOT read back from the encoder;
//!   4. asserts the Rust `u64` cell EQUALS the independently-computed formula, value-for-value.
//!
//! This upgrades the bridge from "Rust is injective (property)" to "Rust computes the exact
//! proved formula (function)" on the bounded input universe.
//!
//! # HONEST residual — what this is and is NOT
//!
//! A differential test over bounded inputs is STRONG EVIDENCE that the Rust implements the proved
//! formula, but it is NOT the for-all-inputs formal proof. The remaining bridge step — a machine
//! proof that `value_cell_encode_at` computes `∑ dᵢ·bⁱ` (etc.) for EVERY input (e.g. Creusot
//! contracts on the Rust, or extraction of the Rust into Lean) — is still open. This file narrows
//! the residual from "the Lean covers *a* formula; does the Rust compute *that* formula?" to "the
//! Rust computes that formula on all bounded inputs we can enumerate; the unbounded proof remains".
//!
//! # Formula-map finding (see the module report)
//!
//! Every family's Rust pack is LITERALLY its Lean formula — same digit order (index 0 is least
//! significant / `b^0`), same `+1` seq shift, no extra offset. The ONE convention note: the seq
//! arm ([`pack_seq`]) admits elements `a < D-1` (digit `a+1 < D`), which is a *stricter* domain
//! than S3's `a < D`; the pack formula is identical, so S3's injectivity still covers every input
//! the Rust actually encodes. That is a domain restriction, not a formula mismatch.

#![cfg(feature = "clean-cic")]
// The S1/S2/S3 in names deliberately echo the Lean theorem identifiers
// (`positional_encoding_injective` = S1, etc.) each leg differential-checks against.
#![allow(non_snake_case)]

use std::sync::Arc;

use tla_check::explicit_fixpoint_cert::{
    value_cell_encode_at, ColSort, RECORD_FUNC_BASE, SEQ_MAX_LEN, SET_UNIVERSE_BITS,
};
use tla_check::value::{FuncBuilder, IntIntervalFunc, RecordBuilder, SeqValue, SortedSet, Value};

// ─────────────────────────── value builders ───────────────────────────

fn vint(n: i64) -> Value {
    Value::int(n)
}
fn vstr(s: &str) -> Value {
    Value::String(Arc::from(s))
}
fn vmodel(s: &str) -> Value {
    Value::ModelValue(Arc::from(s))
}
fn vset(elems: &[Value]) -> Value {
    Value::Set(Arc::new(SortedSet::from_vec(elems.to_vec())))
}
fn vtuple(elems: &[Value]) -> Value {
    Value::Tuple(Arc::from(elems.to_vec()))
}
fn vseq(elems: &[Value]) -> Value {
    Value::Seq(Arc::new(SeqValue::from_vec(elems.to_vec())))
}
fn vfunc(pairs: &[(Value, Value)]) -> Value {
    let mut b = FuncBuilder::new();
    for (k, v) in pairs {
        b.insert(k.clone(), v.clone());
    }
    Value::Func(Arc::new(b.build()))
}
/// `[0..len-1 |-> values]` — an `IntFunc` over the encodable 0-based prefix.
fn vintfunc(values: &[Value]) -> Value {
    let max = values.len() as i64 - 1;
    Value::IntFunc(Arc::new(IntIntervalFunc::new(0, max, values.to_vec())))
}

// ─────────────── INDEPENDENT Lean-model closed forms (hand-written MATH) ───────────────
//
// These recompute the three proved formulas DIRECTLY from a value's components. They deliberately
// do NOT touch `value_cell_encode_at` — the differential test compares the encoder's output
// against THESE, so mirroring the encoder here would defeat the purpose. Each mirrors the Lean
// model, not the Rust encoder.

/// S1 — the base-`b` positional pack `∑ dᵢ·bⁱ` from `proofs/lean/aristotle/S1_positional_pack.lean`
/// (`positional_encoding_injective`: `Finset.univ.sum (fun i => d i * b ^ (i : Nat))`). Digit 0 is
/// the least-significant (multiplied by `b^0`), matching the Lean `Fin n` indexing.
fn lean_S1_positional_pack(digits: &[u64], b: u64) -> u64 {
    let mut acc: u64 = 0;
    for (i, &d) in digits.iter().enumerate() {
        let place = b.checked_pow(i as u32).expect("test place fits u64");
        acc = acc
            .checked_add(d.checked_mul(place).expect("test term fits u64"))
            .expect("sum fits");
    }
    acc
}

/// S2 — the bitmask `∑_{e∈S} 2^e` from `proofs/lean/aristotle/S2_bitmask.lean`
/// (`sum_two_pow_injective`: `S.sum (fun e => 2 ^ e)`). The exponent IS the set element's Int value.
fn lean_S2_bitmask(elems: &[u64]) -> u64 {
    let mut acc: u64 = 0;
    for &e in elems {
        acc = acc
            .checked_add(1u64.checked_shl(e as u32).expect("test 2^e fits u64"))
            .expect("sum");
    }
    acc
}

/// S3 — the shifted self-delimiting pack `∑ (aᵢ+1)·Dⁱ` from
/// `proofs/lean/aristotle/S3_seqpack.lean` (`RequestProject.pack D`:
/// `pack D (a::rest) = (a+1) + D·pack D rest`). Element 0 is the least-significant; the `+1` shift
/// makes digit 0 self-delimit the length. Written here in the direct `∑` form; the `foldr` below
/// cross-checks that this equals the Lean recursive `pack`.
fn lean_S3_seq_pack(elems: &[u64], d: u64) -> u64 {
    let mut acc: u64 = 0;
    for (i, &a) in elems.iter().enumerate() {
        let place = d.checked_pow(i as u32).expect("test place fits u64");
        acc = acc
            .checked_add((a + 1).checked_mul(place).expect("test term fits u64"))
            .expect("sum fits");
    }
    acc
}

/// The Lean `RequestProject.pack` verbatim as a right fold (`pack D [] = 0`,
/// `pack D (a::rest) = (a+1) + D·pack D rest`) — used to confirm the `∑` form in
/// [`lean_S3_seq_pack`] agrees with the recursive Lean definition it is claimed to equal.
fn lean_S3_pack_recursive(elems: &[u64], d: u64) -> u64 {
    match elems.split_first() {
        None => 0,
        Some((&a, rest)) => (a + 1) + d * lean_S3_pack_recursive(rest, d),
    }
}

// ─────────────────────────── enumeration helper ───────────────────────────

/// All `base^arity` little-endian digit tuples — the full cartesian product `{0..base}^arity`.
fn digit_tuples(arity: usize, base: u64) -> Vec<Vec<u64>> {
    let mut out: Vec<Vec<u64>> = vec![Vec::new()];
    for _ in 0..arity {
        let mut next = Vec::with_capacity(out.len() * base as usize);
        for prefix in &out {
            for d in 0..base {
                let mut p = prefix.clone();
                p.push(d);
                next.push(p);
            }
        }
        out = next;
    }
    out
}

// ═══════════════════════════ S1 — POSITIONAL PACK ═══════════════════════════

/// Rust `value_cell_encode_at` (`Value::Func`, Int-prefix domain) ≡ S1
/// `positional_encoding_injective`'s `∑ dᵢ·bⁱ` — checked value-for-value.
///
/// For every arity 1..=4 and every digit tuple in `{0..base}^arity`, build the finite function
/// `[0..n-1 |-> digits]`, encode it, and assert the Rust cell equals the independently-computed
/// `∑ digitsᵢ·baseⁱ`. Exhaustive over the bounded ranges (like `encoder_injectivity_cert.rs`).
#[test]
fn func_int_prefix_matches_S1_positional_pack() {
    let mut checked = 0usize;
    // Floor base (10, arities 1..=4) plus a WIDER derived base (12, arities 1..=3) so the pack's
    // base-dependence — not just the digits — is differential-checked against the same `base`.
    for &(base, max_arity) in &[(RECORD_FUNC_BASE, 4usize), (12u64, 3usize)] {
        for arity in 1..=max_arity {
            for digits in digit_tuples(arity, base) {
                let pairs: Vec<(Value, Value)> = digits
                    .iter()
                    .enumerate()
                    .map(|(i, &d)| (vint(i as i64), vint(d as i64)))
                    .collect();
                let f = vfunc(&pairs);
                let (sort, cell) =
                    value_cell_encode_at(&f, base).expect("in-fragment Int-prefix func encodes");
                assert!(
                    matches!(sort, ColSort::Func { .. }),
                    "expected Func sort, got {sort:?}"
                );
                let expected = lean_S1_positional_pack(&digits, base);
                assert_eq!(
                    cell, expected,
                    "S1 differential MISMATCH: Func {digits:?} at base {base} → Rust cell {cell} \
                     but Lean ∑dᵢ·bⁱ = {expected}"
                );
                checked += 1;
            }
        }
    }
    // 10+100+1000+10000 + 12+144+1728 = 12994 (value, cell) pairs.
    assert_eq!(checked, 12_994, "expected the full bounded Func universe");
}

/// Rust `value_cell_encode_at` (`Value::IntFunc`, the 0-based interval fast path) ≡ S1
/// `positional_encoding_injective`'s `∑ dᵢ·bⁱ` — checked value-for-value. Same pack as `Value::Func`
/// via a different code path, so it gets its own differential leg.
#[test]
fn intfunc_matches_S1_positional_pack() {
    let mut checked = 0usize;
    let base = RECORD_FUNC_BASE;
    for arity in 1..=4usize {
        for digits in digit_tuples(arity, base) {
            let values: Vec<Value> = digits.iter().map(|&d| vint(d as i64)).collect();
            let f = vintfunc(&values);
            let (sort, cell) = value_cell_encode_at(&f, base).expect("0-based IntFunc encodes");
            assert!(
                matches!(sort, ColSort::Func { .. }),
                "expected Func sort, got {sort:?}"
            );
            let expected = lean_S1_positional_pack(&digits, base);
            assert_eq!(
                cell, expected,
                "S1 differential MISMATCH: IntFunc {digits:?} at base {base} → Rust {cell} \
                 but Lean ∑dᵢ·bⁱ = {expected}"
            );
            checked += 1;
        }
    }
    assert_eq!(checked, 11_110); // 10+100+1000+10000
}

/// Rust `value_cell_encode_at` (`Value::Record`) ≡ S1 `positional_encoding_injective`'s `∑ dᵢ·bⁱ`
/// — checked value-for-value. The record's positional order is its field names' CANONICAL sorted
/// order; to prove the test is independent of the encoder's ordering, records are BUILT with fields
/// inserted in REVERSE name order and the expected pack is computed from the names sorted ASCENDING.
#[test]
fn record_matches_S1_positional_pack() {
    let mut checked = 0usize;
    let names = ["a", "b", "c"]; // sorted ascending ⇒ position 0 = "a" = least significant
    for &(base, max_arity) in &[(RECORD_FUNC_BASE, 3usize), (12u64, 3usize)] {
        for arity in 1..=max_arity {
            for digits in digit_tuples(arity, base) {
                // Build inserting fields in REVERSE order — the encoder must still pack by sorted
                // name, so the expected pack uses `digits` aligned to the ASCENDING names.
                let mut b = RecordBuilder::new();
                for i in (0..arity).rev() {
                    b.insert_str(names[i], vint(digits[i] as i64));
                }
                let r = Value::Record(b.build());
                let (sort, cell) = value_cell_encode_at(&r, base).expect("Int record encodes");
                assert!(
                    matches!(sort, ColSort::Record { .. }),
                    "expected Record, got {sort:?}"
                );
                // digits[i] is already aligned to names[i] (ascending) ⇒ position i.
                let expected = lean_S1_positional_pack(&digits, base);
                assert_eq!(
                    cell,
                    expected,
                    "S1 differential MISMATCH: Record over {:?} = {digits:?} at base {base} → \
                     Rust {cell} but Lean ∑dᵢ·bⁱ = {expected}",
                    &names[..arity]
                );
                checked += 1;
            }
        }
    }
    // (10+100+1000) + (12+144+1728) = 2994
    assert_eq!(checked, 2_994);
}

/// Rust `value_cell_encode_at` (`Value::Func`, ATOM domain) ≡ S1 `positional_encoding_injective`'s
/// `∑ dᵢ·bⁱ` — checked value-for-value, where position `i` is the `i`-th domain key in the domain's
/// canonical `Value::cmp` order. The expected pack sorts the (key,value) pairs by `Value::cmp`
/// INDEPENDENTLY (the same total order the encoder's `func_enum_domain_keys` reads off the sorted
/// domain), then applies `∑ dᵢ·bⁱ`. Covers both String-atom and model-value key kinds.
#[test]
fn atom_domain_func_matches_S1_positional_pack() {
    let mut checked = 0usize;
    let base = RECORD_FUNC_BASE;
    // Two kinds of atom key sets; keys deliberately given out of Value::cmp order at build time.
    let key_sets: [Vec<Value>; 2] = [
        vec![vstr("y"), vstr("x"), vstr("z")],
        vec![vmodel("q"), vmodel("p"), vmodel("r")],
    ];
    for keys in &key_sets {
        for arity in 1..=keys.len() {
            let ks = &keys[..arity];
            for digits in digit_tuples(arity, base) {
                let pairs: Vec<(Value, Value)> = ks
                    .iter()
                    .cloned()
                    .zip(digits.iter().map(|&d| vint(d as i64)))
                    .collect();
                let f = vfunc(&pairs);
                let (sort, cell) =
                    value_cell_encode_at(&f, base).expect("atom-domain func encodes");
                assert!(
                    matches!(sort, ColSort::Func { .. }),
                    "expected Func, got {sort:?}"
                );
                // INDEPENDENTLY reorder the digits into Value::cmp key order, then ∑ dᵢ·bⁱ. Carry
                // the raw u64 digit alongside its key so the reorder never re-reads a Value cell.
                let mut keyed: Vec<(Value, u64)> =
                    ks.iter().cloned().zip(digits.iter().copied()).collect();
                keyed.sort_by(|(ka, _), (kb, _)| ka.cmp(kb));
                let sorted_digits: Vec<u64> = keyed.iter().map(|(_, d)| *d).collect();
                let expected = lean_S1_positional_pack(&sorted_digits, base);
                assert_eq!(
                    cell, expected,
                    "S1 atom-domain MISMATCH: keys {ks:?} digits {digits:?} → Rust {cell} but \
                     Lean ∑dᵢ·bⁱ (Value::cmp key order) = {expected}"
                );
                checked += 1;
            }
        }
    }
    // 2 key kinds × (10 + 100 + 1000) = 2220
    assert_eq!(checked, 2_220);
}

// ═══════════════════════════ S2 — BITMASK ═══════════════════════════

/// Rust `value_cell_encode_at` (`Value::Set`) ≡ S2 `sum_two_pow_injective`'s `∑_{e∈S} 2^e` —
/// checked value-for-value. Exhaustive over EVERY subset of `{0,1,…,K-1}` (all `2^K` masks), so
/// this is an exhaustive differential over the bounded universe. Plus high-bit singletons that
/// exercise exponents up to `SET_UNIVERSE_BITS-1`.
#[test]
fn set_matches_S2_bitmask() {
    let mut checked = 0usize;
    let k = 12u32; // 2^12 = 4096 subsets, elements 0..11
    for bits in 0u64..(1u64 << k) {
        let elems: Vec<u64> = (0..k as u64).filter(|i| bits & (1 << i) != 0).collect();
        let vals: Vec<Value> = elems.iter().map(|&e| vint(e as i64)).collect();
        let (sort, mask) =
            value_cell_encode_at(&vset(&vals), RECORD_FUNC_BASE).expect("set encodes");
        assert!(
            matches!(sort, ColSort::Set { .. }),
            "expected Set sort, got {sort:?}"
        );
        let expected = lean_S2_bitmask(&elems);
        assert_eq!(
            mask, expected,
            "S2 differential MISMATCH: subset {elems:?} → Rust {mask} but ∑2^e = {expected}"
        );
        // (the mask also literally equals the enumeration index `bits` — a base-2 numeral)
        assert_eq!(mask, bits, "Set mask must equal the raw bit pattern");
        checked += 1;
    }
    // High-bit exponents near the universe ceiling (0..K above never reaches them).
    for e in [12u64, 13, 14, (SET_UNIVERSE_BITS - 1) as u64] {
        let (_, mask) = value_cell_encode_at(&vset(&[vint(e as i64)]), RECORD_FUNC_BASE).unwrap();
        assert_eq!(mask, lean_S2_bitmask(&[e]), "S2 high-bit MISMATCH at e={e}");
        checked += 1;
    }
    // A few multi-element high sets spanning low+high bits.
    for &(a, b) in &[(0u64, 15u64), (3, 14), (7, 12)] {
        let (_, mask) =
            value_cell_encode_at(&vset(&[vint(a as i64), vint(b as i64)]), RECORD_FUNC_BASE)
                .unwrap();
        assert_eq!(
            mask,
            lean_S2_bitmask(&[a, b]),
            "S2 span MISMATCH at {{{a},{b}}}"
        );
        checked += 1;
    }
    assert_eq!(checked, 4096 + 4 + 3);
}

// ═══════════════════════════ S3 — SHIFTED SEQ PACK ═══════════════════════════

/// Rust `value_cell_encode_at` (`Value::Tuple` and `Value::Seq`) ≡ S3 `pack_injective`'s
/// `∑ (aᵢ+1)·Dⁱ` (`RequestProject.pack D`) — checked value-for-value. Exhaustive over every length
/// `0..=SEQ_MAX_LEN` and every element tuple in `{0..M}^len`, at both the floor radix and a wider
/// derived radix. The independent `∑` form is additionally cross-checked against the Lean recursive
/// `pack` definition. Both the `Tuple` and `Seq` arms are exercised (TLA-equal, distinct code).
#[test]
fn seq_tuple_matches_S3_shifted_pack() {
    let mut checked = 0usize;
    // The encoder's self-delimiting radix is `D = rf_base`; elements must be `< D-1` (digit `< D`).
    // Floor: D=10 (elements 0..8). Wider: D=12 (elements 0..10). `SEQ_MAX_LEN = 4`.
    for &(d, max_elem) in &[(RECORD_FUNC_BASE, 8u64), (12u64, 10u64)] {
        let m = max_elem + 1; // element digits 0..=max_elem
        for len in 0..=(SEQ_MAX_LEN as usize) {
            for digits in digit_tuples(len, m) {
                // sanity: the ∑ form and the Lean recursive `pack` agree on these inputs.
                let expected = lean_S3_seq_pack(&digits, d);
                assert_eq!(
                    expected,
                    lean_S3_pack_recursive(&digits, d),
                    "the ∑ form and the Lean recursive pack disagree on {digits:?} D={d}"
                );
                let vals: Vec<Value> = digits.iter().map(|&a| vint(a as i64)).collect();
                // Tuple arm.
                let (ts, tcell) =
                    value_cell_encode_at(&vtuple(&vals), d).expect("short tuple encodes");
                assert!(
                    matches!(ts, ColSort::Seq { .. }),
                    "expected Seq sort for tuple, got {ts:?}"
                );
                assert_eq!(
                    tcell, expected,
                    "S3 Tuple MISMATCH: {digits:?} at D={d} → Rust {tcell} but ∑(aᵢ+1)·Dⁱ = {expected}"
                );
                // Seq arm — same value, distinct encoder path.
                let (ss, scell) = value_cell_encode_at(&vseq(&vals), d).expect("short seq encodes");
                assert!(
                    matches!(ss, ColSort::Seq { .. }),
                    "expected Seq sort, got {ss:?}"
                );
                assert_eq!(
                    scell, expected,
                    "S3 Seq MISMATCH: {digits:?} at D={d} → Rust {scell} but ∑(aᵢ+1)·Dⁱ = {expected}"
                );
                checked += 2; // one Tuple + one Seq pair
            }
        }
    }
    // per radix: 9^0+9^1+9^2+9^3+9^4 = 7381 (D=10); 11^0+..+11^4 = 16105 (D=12).
    // (7381 + 16105) tuples × 2 arms = 46972
    assert_eq!(checked, (7_381 + 16_105) * 2);
}
