// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! ENCODER-INJECTIVITY audit + lock-down for the explicit-state certificate lane
//! (proof-roadmap §2 milestone B1 — the certificate encoder's faithfulness).
//!
//! # Why this file exists (the soundness stake)
//!
//! The explicit-fixpoint certificate reduces a spec's reachable set to `R: Vec<Vec<u64>>`
//! (a sorted+deduped set of `u64` TUPLES; BFS `visited` is a `BTreeSet<Vec<u64>>`). Each column
//! of a state is turned into ONE `u64` cell by the trusted-Rust encoder
//! [`value_cell_encode_at`] (plus the union-dependent per-column encoders in the certify loop).
//! The whole soundness story rests on ONE property:
//!
//!   > two states that differ as TLA value-assignments must NEVER encode to the same `Vec<u64>`.
//!
//! If two DISTINCT states collapsed to one element of `R`, a reachable VIOLATING state could be
//! silently dropped from the deduplicated set ⇒ a FALSE certificate. Because the per-column
//! [`ColSort`] vector is REQUIRED to agree across every enumerated state (the `col_sorts`
//! agreement check in `enumerate_at` fails closed on any disagreement), an accepted certificate
//! has ONE fixed `ColSort` per column, and the collapse question reduces to the per-column claim:
//!
//!   > for a FIXED `ColSort`, the map `Value -> u64` is INJECTIVE (modulo TLA value equality).
//!
//! Equivalently — the statement this file mechanizes — `value_cell_encode_at` viewed as
//! `Value -> (ColSort, u64)` never sends two distinct TLA values to the same `(ColSort, u64)`
//! pair. (A differing `ColSort` is harmless: it forces `col_sorts` disagreement ⇒ the whole
//! column is DECLINED, never a silent merge.)
//!
//! # What is EXHAUSTIVELY guaranteed here (proof-strength over bounded universes)
//!
//! An exhaustive check over a bounded value universe IS a proof of injectivity for that instance.
//! This file:
//!   * enumerates DISTINCT values across every `value_cell_encode_at` arm
//!     (`Int`/`Bool`/`Set`/`Record`/`Func`/`IntFunc`/`Seq`/`Tuple`), encodes each at a FIXED
//!     column base, and asserts all `(ColSort, u64)` cells are pairwise distinct;
//!   * proves per-fiber BIJECTIONS (enumerate ALL values of one fixed `ColSort` and assert the
//!     packs hit exactly `[0, base^arity)` / `[0, 2^K)` — injective AND surjective);
//!   * pins the width/cardinality caps as EXACT (a compound just under the `base^arity ≤ u64::MAX`
//!     cap encodes; just over DECLINES — so an overflow-wrap collapse is impossible);
//!   * covers the CROSS-KIND matrix (`Int 1` vs `TRUE` vs `String "1"` vs model `1`): the
//!     `CellSort`/`EnumKind`/`dom_kind` tag lives in the `ColSort`, so cross-kind look-alikes that
//!     share a `u64` carry DISTINCT `ColSort`s (⇒ `col_sorts` decline, never a merge);
//!   * pins [`record_value_key`] (the `SetMaskRec` record→key bijection) as injective, incl. the
//!     cross-kind matrix.
//!
//! # What stays ARGUED-not-proved (the honest residual)
//!
//! Injectivity for UNBOUNDED universes (all arities/bases/label-unions) is argued structurally in
//! the module docs, not mechanized — that is the target of the later Rocq/Creusot mechanization.
//! The union-dependent sorts (`Enum`/`FuncEnum`/`SetMask`/`SetMaskRec`/`FuncSetMask`) reuse the
//! SAME three primitives proved here — the positional base-`b` pack (`Record`/`Func`), the
//! self-delimiting seq pack (`Seq`), and the `Σ 2^idx` bitmask (`Set`) — composed with a
//! sorted-label index bijection; their injectivity is therefore covered at the primitive level,
//! with the composition itself argued in the `ColSort` docs.

#![cfg(feature = "clean-cic")]

use std::collections::{HashMap, HashSet};
use tla_value::Rp;

use tla_check::explicit_fixpoint_cert::{
    record_value_key, value_cell_encode, value_cell_encode_at, ColSort, RECORD_FUNC_BASE,
    SEQ_MAX_LEN, SET_UNIVERSE_BITS,
};
use tla_check::value::{FuncBuilder, IntIntervalFunc, RecordBuilder, SeqValue, SortedSet, Value};

// ─────────────────────────── value builders ───────────────────────────

fn vint(n: i64) -> Value {
    Value::int(n)
}
fn vbool(b: bool) -> Value {
    Value::Bool(b)
}
fn vstr(s: &str) -> Value {
    Value::String(Rp::from(s))
}
fn vmodel(s: &str) -> Value {
    Value::ModelValue(Rp::from(s))
}
fn vset(elems: &[Value]) -> Value {
    Value::Set(Rp::new(SortedSet::from_vec(elems.to_vec())))
}
fn vtuple(elems: &[Value]) -> Value {
    Value::Tuple(Rp::from(elems.to_vec()))
}
fn vseq(elems: &[Value]) -> Value {
    Value::Seq(Rp::new(SeqValue::from_vec(elems.to_vec())))
}
fn vrecord(fields: &[(&str, Value)]) -> Value {
    let mut b = RecordBuilder::new();
    for (n, v) in fields {
        b.insert_str(n, v.clone());
    }
    Value::Record(b.build())
}
fn vfunc(pairs: &[(Value, Value)]) -> Value {
    let mut b = FuncBuilder::new();
    for (k, v) in pairs {
        b.insert(k.clone(), v.clone());
    }
    Value::Func(Rp::new(b.build()))
}
/// `[min..min+len-1 |-> values]` — an `IntFunc` (min=0 is the encodable 0-based prefix).
fn vintfunc(min: i64, values: &[Value]) -> Value {
    let max = min + values.len() as i64 - 1;
    Value::IntFunc(Rp::new(IntIntervalFunc::new(min, max, values.to_vec())))
}

// ─────────────────────────── injectivity engine ───────────────────────────

/// The heart of the audit. Given labelled candidate values:
///   1. DEDUP the inputs by TLA value equality (equal values legitimately share a cell — that is
///      determinism, not a collapse — e.g. an `IntFunc` and the equal `Func`, or a `Seq` and the
///      equal `Tuple`); a real collapse needs DISTINCT values, which survive dedup;
///   2. encode every survivor at the FIXED column `base` and key the result by `(ColSort, u64)`;
///   3. PANIC if two distinct (post-dedup) values share a cell — that is exactly the false-cert
///      collapse (two states would merge in `R`).
/// Returns `(distinct_values, encoded)` for coverage reporting.
fn assert_injective_at(base: u64, values: &[(&str, Value)]) -> (usize, usize) {
    // (1) dedup by TLA value equality.
    let mut distinct: Vec<(&str, Value)> = Vec::new();
    for (n, v) in values {
        if !distinct.iter().any(|(_, u)| u == v) {
            distinct.push((*n, v.clone()));
        }
    }
    // (2)+(3) encode + collision-detect.
    let mut seen: HashMap<String, (String, ColSort, u64)> = HashMap::new();
    let mut encoded = 0usize;
    for (name, v) in &distinct {
        if let Some((sort, cell)) = value_cell_encode_at(v, base) {
            encoded += 1;
            let key = format!("{sort:?}\u{1}{cell}");
            if let Some((prev_name, prev_sort, prev_cell)) = seen.get(&key) {
                panic!(
                    "INJECTIVITY COLLAPSE at base {base}: distinct TLA values `{prev_name}` and \
                     `{name}` both encode to the SAME cell ({prev_sort:?}, {prev_cell}). Two \
                     distinct states would dedup into ONE element of R ⇒ a violating state could \
                     be silently dropped ⇒ FALSE CERTIFICATE."
                );
            }
            seen.insert(key, (name.to_string(), sort, cell));
        }
    }
    (distinct.len(), encoded)
}

// The mixed cross-sort universe, deliberately including the adversarial SAME-u64 clusters
// (values across DIFFERENT sorts that pack to the same `u64` — separated only by `ColSort`).
fn mixed_universe() -> Vec<(&'static str, Value)> {
    let mut u: Vec<(&'static str, Value)> = Vec::new();
    // ── scalars ──
    for k in 0..=6i64 {
        u.push((leak(format!("int_{k}")), vint(k)));
    }
    u.push(("bool_true", vbool(true)));
    u.push(("bool_false", vbool(false)));
    // ── sets (bitmask) ── incl. empty, singletons, pairs, triple, high bit
    u.push(("set_empty", vset(&[])));
    u.push(("set_0", vset(&[vint(0)])));
    u.push(("set_1", vset(&[vint(1)])));
    u.push(("set_2", vset(&[vint(2)])));
    u.push(("set_01", vset(&[vint(0), vint(1)])));
    u.push(("set_02", vset(&[vint(0), vint(2)])));
    u.push(("set_12", vset(&[vint(1), vint(2)])));
    u.push(("set_012", vset(&[vint(0), vint(1), vint(2)])));
    u.push(("set_3", vset(&[vint(3)])));
    u.push(("set_15", vset(&[vint(15)])));
    // ── records ── incl. field-name ALIASING and Bool cells
    u.push(("rec_empty", vrecord(&[])));
    u.push(("rec_a0", vrecord(&[("a", vint(0))])));
    u.push(("rec_a1", vrecord(&[("a", vint(1))])));
    u.push(("rec_a2", vrecord(&[("a", vint(2))])));
    u.push(("rec_b0", vrecord(&[("b", vint(0))])));
    u.push(("rec_b1", vrecord(&[("b", vint(1))])));
    u.push(("rec_ab_00", vrecord(&[("a", vint(0)), ("b", vint(0))])));
    u.push(("rec_ab_10", vrecord(&[("a", vint(1)), ("b", vint(0))])));
    u.push(("rec_ab_01", vrecord(&[("a", vint(0)), ("b", vint(1))])));
    u.push(("rec_ab_12", vrecord(&[("a", vint(1)), ("b", vint(2))])));
    u.push(("rec_cd_12", vrecord(&[("c", vint(1)), ("d", vint(2))]))); // aliases rec_ab_12's pack (21)
    u.push(("rec_a_true", vrecord(&[("a", vbool(true))]))); // cells [Bool], pack 1 — aliases rec_a1
    u.push(("rec_a_false", vrecord(&[("a", vbool(false))]))); // cells [Bool], pack 0 — aliases rec_a0
    u.push(("rec_ab_t1", vrecord(&[("a", vbool(true)), ("b", vint(1))])));
    u.push(("rec_xy_00", vrecord(&[("x", vint(0)), ("y", vint(0))])));
    // ── funcs (Int-prefix domain) ── incl. arity-0 and Bool value
    u.push(("func_empty", vfunc(&[])));
    u.push(("func_0__0", vfunc(&[(vint(0), vint(0))])));
    u.push(("func_0__1", vfunc(&[(vint(0), vint(1))])));
    u.push(("func_0__2", vfunc(&[(vint(0), vint(2))])));
    u.push((
        "func_01__00",
        vfunc(&[(vint(0), vint(0)), (vint(1), vint(0))]),
    ));
    u.push((
        "func_01__10",
        vfunc(&[(vint(0), vint(1)), (vint(1), vint(0))]),
    ));
    u.push((
        "func_01__01",
        vfunc(&[(vint(0), vint(0)), (vint(1), vint(1))]),
    ));
    u.push((
        "func_01__12",
        vfunc(&[(vint(0), vint(1)), (vint(1), vint(2))]),
    ));
    u.push(("func_0__true", vfunc(&[(vint(0), vbool(true))]))); // cells [Bool], pack 1
                                                                // ── funcs (atom domain) ── model vs String key of the SAME name (dom_kind separation)
    u.push(("func_pmodel__0", vfunc(&[(vmodel("p"), vint(0))])));
    u.push(("func_pstr__0", vfunc(&[(vstr("p"), vint(0))])));
    u.push((
        "func_pq_model__01",
        vfunc(&[(vmodel("p"), vint(0)), (vmodel("q"), vint(1))]),
    ));
    // ── tuples / sequences (self-delimiting pack) ──
    u.push(("tup_empty", vtuple(&[])));
    u.push(("tup_0", vtuple(&[vint(0)])));
    u.push(("tup_1", vtuple(&[vint(1)])));
    u.push(("tup_2", vtuple(&[vint(2)])));
    u.push(("tup_00", vtuple(&[vint(0), vint(0)])));
    u.push(("tup_01", vtuple(&[vint(0), vint(1)])));
    u.push(("tup_10", vtuple(&[vint(1), vint(0)])));
    u.push(("tup_012", vtuple(&[vint(0), vint(1), vint(2)])));
    u.push(("seq_5", vseq(&[vint(5)])));
    u.push(("seq_34", vseq(&[vint(3), vint(4)])));
    u
}

// Leak a String into a &'static str for test-label ergonomics (bounded, test-only).
fn leak(s: String) -> &'static str {
    Box::leak(s.into_boxed_str())
}

// ═══════════════════════════ TESTS ═══════════════════════════

/// The primary cross-sort injectivity guarantee at the FLOOR base (`RECORD_FUNC_BASE`), the
/// byte-compatible pre-widening radix. Includes the adversarial same-`u64` clusters
/// (`Int 1` / `Bool TRUE` / `Record [a|->1]` / `Func [0|->1]` / `Tuple <<0>>` / `Set {0}` all
/// touch `u64 = 1`; the empties all touch `u64 = 0`; `[a|->1,b|->2]` and `[c|->1,d|->2]` both
/// pack to `21`). NONE may collide once the `ColSort` is taken into account.
#[test]
fn injective_mixed_universe_floor_base() {
    let (distinct, encoded) = assert_injective_at(RECORD_FUNC_BASE, &mixed_universe());
    // sanity: the universe is large + mostly encodable at the floor base.
    assert!(
        distinct >= 45,
        "universe collapsed too much under value-eq dedup: {distinct}"
    );
    assert!(
        encoded >= 40,
        "expected most of the floor-base universe to encode: {encoded}"
    );
}

/// The SAME universe re-encoded at WIDER column bases (the per-column derived radix a real column
/// uses once a field value forces widening). A wider base only spreads packs further apart, so
/// injectivity must continue to hold; this exercises the widened fibers.
#[test]
fn injective_mixed_universe_wide_bases() {
    for base in [11u64, 12, 16, 37, 256] {
        let (_, encoded) = assert_injective_at(base, &mixed_universe());
        assert!(
            encoded >= 40,
            "expected the universe to encode at base {base}: {encoded}"
        );
    }
}

/// EXHAUSTIVE per-fiber BIJECTION — Int-prefix `Func` of fixed arity at the floor base: enumerate
/// ALL `base^arity` functions, assert (a) every one carries the IDENTICAL `ColSort` (one fiber),
/// and (b) the packs are EXACTLY `{0, 1, …, base^arity − 1}` — an injective AND surjective
/// base-`base` numeral. This is a *proof* of injectivity for the fiber, not a sample.
#[test]
fn exhaustive_func_fiber_bijection() {
    let base = RECORD_FUNC_BASE; // 10
    for arity in 1usize..=3 {
        let count = base.pow(arity as u32);
        let mut packs: HashSet<u64> = HashSet::new();
        let mut the_sort: Option<ColSort> = None;
        for digits in digit_tuples(arity, base) {
            let pairs: Vec<(Value, Value)> = digits
                .iter()
                .enumerate()
                .map(|(i, d)| (vint(i as i64), vint(*d as i64)))
                .collect();
            let f = vfunc(&pairs);
            let (sort, cell) = value_cell_encode(&f).expect("in-fragment func encodes");
            match &the_sort {
                Some(s) => assert_eq!(*s, sort, "fiber sort drifted at arity {arity}"),
                None => the_sort = Some(sort),
            }
            assert!(
                packs.insert(cell),
                "DUPLICATE pack {cell} at arity {arity} (non-injective!)"
            );
        }
        assert_eq!(
            packs.len() as u64,
            count,
            "fiber not surjective onto [0,{count})"
        );
        assert_eq!(
            *packs.iter().max().unwrap(),
            count - 1,
            "max pack != base^arity - 1"
        );
    }
}

/// EXHAUSTIVE per-fiber BIJECTION — ATOM-DOMAIN `Func` over a fixed model-value key set `{p,q}`.
/// This pins the one convention the atom-domain pack rests on: the key→position map is the
/// `SortedSet`-normalized (sorted-by-`Value::cmp`, position-aligned) domain, so `{p|->0,q|->1}` and
/// `{p|->1,q|->0}` MUST pack differently. All `base^2` value assignments map to distinct packs.
#[test]
fn exhaustive_atom_domain_func_fiber_bijection() {
    let base = RECORD_FUNC_BASE;
    let mut packs: HashSet<u64> = HashSet::new();
    let mut the_sort: Option<ColSort> = None;
    for dp in 0..base {
        for dq in 0..base {
            let f = vfunc(&[
                (vmodel("p"), vint(dp as i64)),
                (vmodel("q"), vint(dq as i64)),
            ]);
            let (sort, cell) = value_cell_encode(&f).expect("atom-domain func encodes");
            match &the_sort {
                Some(s) => assert_eq!(*s, sort, "atom-domain fiber sort drifted"),
                None => the_sort = Some(sort),
            }
            assert!(
                packs.insert(cell),
                "DUPLICATE atom-domain func pack {cell} (non-injective!)"
            );
        }
    }
    assert_eq!(packs.len() as u64, base * base);
    // The ColSort must carry the sorted key names + kind (the identity that keeps two key sets apart).
    match the_sort.unwrap() {
        ColSort::Func {
            dom,
            dom_kind,
            arity,
            ..
        } => {
            assert_eq!(dom, vec!["p".to_string(), "q".to_string()]);
            assert_eq!(arity, 2);
            assert!(matches!(
                dom_kind,
                tla_check::explicit_fixpoint_cert::EnumKind::Model
            ));
        }
        other => panic!("expected atom-domain Func sort, got {other:?}"),
    }
}

/// EXHAUSTIVE per-fiber BIJECTION — 2-field `Record` at the floor base (`fields = [a,b]`, sorted):
/// all `base^2` records pack to a distinct numeral in `[0, base^2)`.
#[test]
fn exhaustive_record_fiber_bijection() {
    let base = RECORD_FUNC_BASE;
    let count = base * base;
    let mut packs: HashSet<u64> = HashSet::new();
    for da in 0..base {
        for db in 0..base {
            let r = vrecord(&[("a", vint(da as i64)), ("b", vint(db as i64))]);
            let (_, cell) = value_cell_encode(&r).expect("record encodes");
            assert!(
                packs.insert(cell),
                "DUPLICATE record pack {cell} (non-injective!)"
            );
        }
    }
    assert_eq!(packs.len() as u64, count);
}

/// EXHAUSTIVE per-fiber BIJECTION — `Set` bitmask over the element universe `{0..K}`: all `2^K`
/// subsets map to distinct masks `[0, 2^K)`. This is the `Σ 2^idx` bitmask primitive that
/// `SetMask`/`SetMaskRec`/`FuncSetMask` all reuse.
#[test]
fn exhaustive_set_bitmask_bijection() {
    let k = 6u32; // 64 subsets
    let mut masks: HashSet<u64> = HashSet::new();
    for bits in 0u64..(1u64 << k) {
        let elems: Vec<Value> = (0..k)
            .filter(|i| bits & (1 << i) != 0)
            .map(|i| vint(i as i64))
            .collect();
        let (_, mask) = value_cell_encode(&vset(&elems)).expect("small set encodes");
        assert_eq!(mask, bits, "bitmask != Σ 2^e");
        assert!(
            masks.insert(mask),
            "DUPLICATE set mask {mask} (non-injective!)"
        );
    }
    assert_eq!(masks.len(), 1usize << k);
}

/// EXHAUSTIVE — self-delimiting `Tuple`/`Seq` pack over all lengths `0..=SEQ_MAX_LEN` with element
/// digits `0..3`: distinct sequences (varying length AND content) map to distinct packs. The `+1`
/// digit shift makes length self-delimiting, so `<<0>>` ≠ `<<>>` ≠ `<<0,0>>`.
#[test]
fn exhaustive_seq_selfdelimiting_injective() {
    let elem_digits = 3u64; // elements 0..3, well under SEQ_BASE
    let mut seen: HashMap<u64, String> = HashMap::new();
    for len in 0..=(SEQ_MAX_LEN as usize) {
        for digits in digit_tuples(len, elem_digits) {
            let elems: Vec<Value> = digits.iter().map(|d| vint(*d as i64)).collect();
            let label = format!(
                "<<{}>>",
                digits
                    .iter()
                    .map(|d| d.to_string())
                    .collect::<Vec<_>>()
                    .join(",")
            );
            let (_, pack) = value_cell_encode(&vtuple(&elems)).expect("short tuple encodes");
            if let Some(prev) = seen.insert(pack, label.clone()) {
                panic!("SEQ COLLAPSE: distinct {prev} and {label} share pack {pack}");
            }
        }
    }
}

/// The width/cardinality CAPS are EXACT — a value just UNDER the cap encodes, just OVER DECLINES.
/// A declined value is `None` (never admitted), so an overflow-WRAP collapse (two distinct compound
/// values → same `u64` mod 2^64) is structurally impossible.
#[test]
fn width_caps_are_exact_no_overflow_collapse() {
    // Func pack cap at the floor base: 10^19 < u64::MAX < 10^20.
    let f19 = vintfunc(0, &vec![vint(0); 19]); // 10^19 fits u64
    let f20 = vintfunc(0, &vec![vint(0); 20]); // 10^20 overflows u64
    assert!(
        value_cell_encode(&f19).is_some(),
        "arity-19 base-10 pack (10^19) fits u64 ⇒ encodes"
    );
    assert!(
        value_cell_encode(&f20).is_none(),
        "arity-20 base-10 pack (10^20) overflows ⇒ DECLINES"
    );

    // Func pack cap at base 16: 16^15 = 2^60 fits, 16^16 = 2^64 overflows.
    let g15 = vintfunc(0, &vec![vint(0); 15]);
    let g16 = vintfunc(0, &vec![vint(0); 16]);
    assert!(
        value_cell_encode_at(&g15, 16).is_some(),
        "arity-15 base-16 pack (2^60) fits u64"
    );
    assert!(
        value_cell_encode_at(&g16, 16).is_none(),
        "arity-16 base-16 pack (2^64) overflows ⇒ DECLINES"
    );

    // Seq length cap.
    let t_ok = vtuple(&vec![vint(0); SEQ_MAX_LEN as usize]);
    let t_over = vtuple(&vec![vint(0); SEQ_MAX_LEN as usize + 1]);
    assert!(
        value_cell_encode(&t_ok).is_some(),
        "len==SEQ_MAX_LEN encodes"
    );
    assert!(
        value_cell_encode(&t_over).is_none(),
        "len>SEQ_MAX_LEN DECLINES"
    );

    // Set element-universe bit cap.
    let s_ok = vset(&[vint((SET_UNIVERSE_BITS - 1) as i64)]);
    let s_over = vset(&[vint(SET_UNIVERSE_BITS as i64)]);
    assert!(
        value_cell_encode(&s_ok).is_some(),
        "element < SET_UNIVERSE_BITS encodes"
    );
    assert!(
        value_cell_encode(&s_over).is_none(),
        "element >= SET_UNIVERSE_BITS DECLINES"
    );

    // Record field-value digit must be < base (else Widen, not a wrap): value 10 at base 10 declines.
    let r_over = vrecord(&[("a", vint(10))]);
    assert!(
        value_cell_encode(&r_over).is_none(),
        "field value == base DECLINES (caller widens)"
    );
    assert!(
        value_cell_encode_at(&r_over, 11).is_some(),
        "…and encodes once the base admits it"
    );
}

/// CROSS-KIND separation at the `value_cell` level — look-alike values that SHARE a `u64` carry
/// DISTINCT `ColSort`s (the `CellSort`/`dom_kind` tag). In a real column a `ColSort` mismatch forces
/// `col_sorts` disagreement ⇒ the column is DECLINED (never a silent merge).
#[test]
fn cross_kind_separated_by_colsort() {
    // Int n vs Bool: same u64, distinct sort (Int vs Bool).
    let (si0, ci0) = value_cell_encode(&vint(0)).unwrap();
    let (sbf, cbf) = value_cell_encode(&vbool(false)).unwrap();
    assert_eq!(ci0, cbf, "Int 0 and FALSE share u64 0");
    assert_ne!(si0, sbf, "…but must carry distinct ColSorts");
    let (si1, ci1) = value_cell_encode(&vint(1)).unwrap();
    let (sbt, cbt) = value_cell_encode(&vbool(true)).unwrap();
    assert_eq!(ci1, cbt);
    assert_ne!(si1, sbt);

    // Scalar atoms (String / model value) and negatives are OUT of value_cell's fragment — the
    // scalar cross-kind separation is enforced by the certify loop's Enum path + col_sorts, not here.
    assert!(
        value_cell_encode(&vstr("1")).is_none(),
        "scalar String not in value_cell fragment"
    );
    assert!(
        value_cell_encode(&vmodel("1")).is_none(),
        "scalar model value not in value_cell fragment"
    );
    assert!(
        value_cell_encode(&vint(-1)).is_none(),
        "negative Int declines"
    );

    // Record: Bool cell vs Int cell at the same position — same pack, distinct sort via `cells`.
    let (sri, cri) = value_cell_encode(&vrecord(&[("a", vint(1))])).unwrap();
    let (srb, crb) = value_cell_encode(&vrecord(&[("a", vbool(true))])).unwrap();
    assert_eq!(cri, crb, "[a|->1] and [a|->TRUE] both pack to 1");
    assert_ne!(
        sri, srb,
        "…but the CellSort tag (Int vs Bool) makes the ColSorts distinct"
    );

    // Func atom domain: model key vs String key of the SAME name — same pack, distinct dom_kind.
    let (sfm, cfm) = value_cell_encode(&vfunc(&[(vmodel("p"), vint(0))])).unwrap();
    let (sfs, cfs) = value_cell_encode(&vfunc(&[(vstr("p"), vint(0))])).unwrap();
    assert_eq!(cfm, cfs, "[p(model)|->0] and [p(str)|->0] both pack to 0");
    assert_ne!(
        sfm, sfs,
        "…but dom_kind (Model vs Str) makes the ColSorts distinct"
    );
}

/// The equivalent-representation probe: a `Seq` and a `Tuple` (resp. a `Func` and an `IntFunc`)
/// of identical contents are TLA-EQUAL values, so mapping them to the same cell is DETERMINISM,
/// not a collapse. If they were ever DISTINCT-yet-same-cell, that would be a real collapse — so
/// this test asserts `same value ⟺ same cell` for those representation pairs.
#[test]
fn equivalent_representations_same_cell_iff_equal() {
    let cases: [(Value, Value, &str); 3] = [
        (
            vseq(&[vint(0), vint(1)]),
            vtuple(&[vint(0), vint(1)]),
            "Seq vs Tuple",
        ),
        (
            vfunc(&[(vint(0), vint(0)), (vint(1), vint(1))]),
            vintfunc(0, &[vint(0), vint(1)]),
            "Func vs IntFunc",
        ),
        (vseq(&[]), vtuple(&[]), "empty Seq vs empty Tuple"),
    ];
    for (a, b, name) in cases {
        let ca = value_cell_encode(&a);
        let cb = value_cell_encode(&b);
        if a == b {
            assert_eq!(
                ca, cb,
                "{name}: EQUAL TLA values must share a cell (determinism)"
            );
        } else {
            assert_ne!(
                ca, cb,
                "{name}: DISTINCT TLA values map to the same cell ⇒ INJECTIVITY COLLAPSE"
            );
        }
    }
}

/// EXHAUSTIVE — [`record_value_key`] (the `SetMaskRec` record→key bijection) over a bounded universe
/// of records whose fields hold the CROSS-KIND look-alikes (`Int 1` / `TRUE` / `String "1"` / model
/// `1`). Distinct records must get distinct keys; equal records equal keys; a non-leaf field ⇒ `None`.
#[test]
fn record_value_key_exhaustive_cross_kind_injective() {
    // Cross-kind leaf values that "look alike" as text — the tag must keep them apart.
    let leaves: Vec<(&str, Value)> = vec![
        ("i0", vint(0)),
        ("i1", vint(1)),
        ("bt", vbool(true)),
        ("bf", vbool(false)),
        ("s1", vstr("1")),
        ("s0", vstr("0")),
        ("sa", vstr("a")),
        ("m1", vmodel("1")),
        ("ma", vmodel("a")),
    ];
    // Records over field-sets {f}, {g}, {f,g}. Enumerate all leaf assignments; dedup by TLA value.
    let mut records: Vec<(String, Value)> = Vec::new();
    for (ln, lv) in &leaves {
        records.push((format!("[f|->{ln}]"), vrecord(&[("f", lv.clone())])));
        records.push((format!("[g|->{ln}]"), vrecord(&[("g", lv.clone())])));
    }
    for (ln1, lv1) in &leaves {
        for (ln2, lv2) in &leaves {
            records.push((
                format!("[f|->{ln1},g|->{ln2}]"),
                vrecord(&[("f", lv1.clone()), ("g", lv2.clone())]),
            ));
        }
    }
    // Dedup by value, then assert key injectivity.
    let mut distinct: Vec<(String, Value)> = Vec::new();
    for (n, v) in records {
        if !distinct.iter().any(|(_, u)| *u == v) {
            distinct.push((n, v));
        }
    }
    let mut seen: HashMap<String, String> = HashMap::new();
    for (name, rec) in &distinct {
        let key = record_value_key(rec).expect("all-leaf record is keyable");
        if let Some(prev) = seen.insert(key.clone(), name.clone()) {
            panic!(
                "record_value_key COLLAPSE: distinct records {prev} and {name} share key {key:?} \
                 ⇒ two records would share a SetMaskRec bit ⇒ FALSE CERTIFICATE"
            );
        }
    }
    // The cross-kind single-field records specifically must NOT conflate (9 leaves × field f).
    let single_f_keys: HashSet<String> = leaves
        .iter()
        .map(|(_, lv)| record_value_key(&vrecord(&[("f", lv.clone())])).unwrap())
        .collect();
    assert_eq!(
        single_f_keys.len(),
        leaves.len(),
        "cross-kind leaves conflated in [f|->·]"
    );

    // A non-leaf field value (nested record) is UNKEYABLE ⇒ None (the column fails closed).
    let nested = vrecord(&[("f", vrecord(&[("g", vint(0))]))]);
    assert!(
        record_value_key(&nested).is_none(),
        "nested-record field ⇒ unkeyable ⇒ None"
    );
    let set_field = vrecord(&[("f", vset(&[vint(0)]))]);
    assert!(
        record_value_key(&set_field).is_none(),
        "set field ⇒ unkeyable ⇒ None"
    );
}

// ─────────────────────────── helpers ───────────────────────────

/// All `base^arity` digit tuples (little-endian), i.e. the full cartesian product `{0..base}^arity`.
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
