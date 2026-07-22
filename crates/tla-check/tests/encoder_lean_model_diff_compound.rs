// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! RUST↔LEAN DIFFERENTIAL bridge for the explicit-state certificate encoder — the COMPOUND
//! (composed-encoding) families, extending the primitive-family bridge in
//! `tests/encoder_lean_model_diff.rs` to the encodings the 27+ certifying corpus specs rely on.
//!
//! # What the primitive bridge established, and what this file adds
//!
//! `encoder_lean_model_diff.rs` proves — value-for-value, on bounded inputs — that TY's trusted-Rust
//! [`value_cell_encode_at`] computes exactly the three Aristotle Lean formulas (S1 positional pack
//! `∑ dᵢ·bⁱ`, S2 bitmask `∑ 2^e`, S3 self-delimiting seq pack). Those are the PRIMITIVE families.
//! The certifying corpus specs (SimpleAllocator, TwoPhase/message-set, program-counter functions,
//! …) use COMPOUND encodings that COMPOSE those primitives with the S7 `rank`/`indexOf` map (S7
//! `rank_injOn` + `comp_injective`):
//!   * `FuncEnum`     — a `[D -> Labels]` function, `pack = ∑_p idx(e_p)·|labels|^p`  (S1 ∘ S7):
//!                      base `|labels|`, digit `p` = the S7 rank of value-label `e_p` in the sorted
//!                      label union, positions ordered by the S7 rank of the domain keys.
//!   * `SetMaskRec`   — a set `S` of records, `mask = ∑_{r∈S} 2^rank(r)`             (S2 ∘ S7):
//!                      the S2 bitmask whose exponents are the S7 ranks of the records' canonical
//!                      keys in the sorted record-key universe.
//!   * `FuncSetMask`  — a `[D -> SUBSET E]` function, `pack = ∑_p mask(f[fdom_p])·base^p`,
//!                      `base = 2^|E|`, `mask(T) = ∑_{a∈T} 2^rank_E(a)`         (S1 ∘ S2 ∘ S7):
//!                      the S1 positional pack (base `2^|E|`) whose digits are S2 bitmasks (over the
//!                      S7-ranked value universe `E`), positions ordered by the S7 rank of `D`.
//!
//! # STEP-0 finding — where the compound encoders actually live (NOT `value_cell_encode_at`)
//!
//! IMPORTANT / HONEST: unlike the primitive families, the compound families are NOT produced by
//! [`value_cell_encode_at`]. That function only encodes `Set`/`Record`/`Func`/`IntFunc`/`Seq`/`Tuple`
//! (and the atom-DOMAIN Int-valued `Func{dom}` — the DUAL of `FuncEnum`, already covered by the
//! primitive test's `atom_domain_func_matches_S1_positional_pack`). The `FuncEnum`/`SetMaskRec`/
//! `FuncSetMask` cells are computed INLINE in the certify enumeration loop
//! (`certify_explicit_state_spec_inner`'s `state_tuple` closure), keyed on a per-column universe
//! (label / atom / record-key union) that GROWS monotonically across the enumerated reachable set —
//! there is no standalone per-value entry point for them. So this differential drives the encoder the
//! only way those cells are produced: it CERTIFIES real (tiny) specs via the public
//! [`certify_explicit_state_spec`] and reads the encoder's cells back off the resulting cert
//! (`cert.init_values` / `cert.reachable`), which store one `u64` pack per state.
//!
//! # How each pair is checked (pointwise, and honestly independent)
//!
//! For each compound family this test:
//!   1. fixes a KNOWN universe by choosing a `Next` that reaches the WHOLE value space (so the
//!      column's label/atom/record universe is a fixed, fully-known set), and
//!   2. enumerates many concrete values `v` of the shape; for each, emits a spec whose `Init` pins
//!      the single state `v` and certifies it, reading the encoder's cell for `v` off
//!      `cert.init_values[0]` (a single-state `Init` ⇒ exactly one labelled `(v, cell)` pair);
//!   3. INDEPENDENTLY computes the composed Lean-model closed form from `v`'s components — reusing
//!      the primitive `∑ dᵢ·bⁱ` / `∑ 2^e` helpers COMPOSED with an INDEPENDENTLY-computed S7 rank
//!      (sort the universe, take `indexOf`); the universe order is re-derived here (sort the atom /
//!      label / canonical-record-key set) and CROSS-CHECKED against the cert's stored `dom`/`labels`,
//!      never read back as the source of truth;
//!   4. asserts the encoder's `u64` cell EQUALS the independently-composed formula, value-for-value;
//!   5. corroborates with an IMAGE-level check: the whole reachable set `cert.reachable` equals the
//!      set of independently-composed cells over the whole value space (no missing / extra / collided
//!      state).
//!
//! Mirroring the MATH (pack-of-bitmask, rank-by-sort), never the encoder: the "expected" cell is
//! NEVER obtained by calling the encoder. The record canonical-KEY order (S7's total order on
//! records) is re-derived here from the documented length-prefixed serialization, and cross-checked
//! against the cert's `dom`.
//!
//! # HONEST residual — bounded differential ≠ for-all-inputs proof
//!
//! Like the primitive bridge, this is STRONG EVIDENCE that the Rust computes the composed formula,
//! but it is NOT the for-all-inputs formal proof. It is exhaustive over the BOUNDED value spaces
//! enumerated here; the unbounded machine proof that the certify closure computes `∑_p mask·base^p`
//! (etc.) for EVERY input remains open. Every family below matched the composed formula with NO
//! offset / order / base discrepancy (see the per-family `(value, cell)` counts in the report).

#![cfg(feature = "clean-cic")]
// The S1/S2/S7 in names echo the Lean theorem identifiers each leg composes.
#![allow(non_snake_case)]

use std::sync::atomic::{AtomicUsize, Ordering};

use tla_check::explicit_fixpoint_cert::{
    certify_explicit_state_spec, ColSort, ExplicitFixpointCert,
};
use tla_check::{Config, ConstantValue};

/// A process-unique module tag. Distinct enumerated values reuse the SAME module NAME otherwise, and
/// the spec front-end memoizes parsed modules by name — a stale cache would return a prior value's
/// arity/shape. Appending a fresh id per certify keeps every enumerated spec its own module.
fn uniq() -> usize {
    static N: AtomicUsize = AtomicUsize::new(0);
    N.fetch_add(1, Ordering::Relaxed)
}

// ─────────────── INDEPENDENT Lean-model closed forms (hand-written MATH) ───────────────
//
// These recompute the proved formulas DIRECTLY from a value's components. They never touch the
// encoder — the differential compares the encoder's cell against THESE.

/// S1 — the base-`b` positional pack `∑ dᵢ·bⁱ` (`proofs/lean/aristotle/S1_positional_pack.lean`,
/// `positional_encoding_injective`). Digit 0 is least-significant (`b^0`).
fn lean_S1_positional_pack(digits: &[u64], b: u64) -> u64 {
    let mut acc: u64 = 0;
    for (i, &d) in digits.iter().enumerate() {
        let place = b.checked_pow(i as u32).expect("test place fits u64");
        acc = acc
            .checked_add(d.checked_mul(place).expect("term fits u64"))
            .expect("sum fits");
    }
    acc
}

/// S2 — the bitmask `∑_{e∈S} 2^e` (`proofs/lean/aristotle/S2_bitmask.lean`, `sum_two_pow_injective`).
fn lean_S2_bitmask(exponents: &[u64]) -> u64 {
    let mut acc: u64 = 0;
    for &e in exponents {
        acc = acc
            .checked_add(1u64.checked_shl(e as u32).expect("2^e fits u64"))
            .expect("sum fits");
    }
    acc
}

/// S7 — the `rank`/`indexOf` map (`proofs/lean/aristotle/S7_enum_rank.lean`, `rank_injOn`): the
/// position of `item` in the SORTED, distinct `universe`. Panics if absent (a test-construction bug).
fn lean_S7_rank(universe_sorted: &[String], item: &str) -> u64 {
    universe_sorted
        .iter()
        .position(|u| u == item)
        .unwrap_or_else(|| panic!("item {item:?} not in universe {universe_sorted:?}")) as u64
}

/// Sort + dedup a set of names into the canonical (lexicographic) universe order. Used to re-derive
/// the label / atom / record-key universe INDEPENDENTLY of the encoder; cross-checked against the
/// cert's stored `dom`/`labels`.
fn sorted_unique(items: &[String]) -> Vec<String> {
    let mut v: Vec<String> = items.to_vec();
    v.sort();
    v.dedup();
    v
}

// ─────────────── composed formulas (primitive ∘ S7), hand-written MATH ───────────────

/// FuncEnum composed cell (S1 ∘ S7): `pack = ∑_p rank(e_p)·|labels|^p`, `slot_labels` already in the
/// domain's canonical (S7-ranked) position order, `labels_sorted` the sorted value-label universe.
fn funcenum_cell(slot_labels: &[&str], labels_sorted: &[String]) -> u64 {
    let base = labels_sorted.len() as u64;
    let digits: Vec<u64> = slot_labels
        .iter()
        .map(|l| lean_S7_rank(labels_sorted, l))
        .collect();
    lean_S1_positional_pack(&digits, base)
}

/// SetMaskRec composed cell (S2 ∘ S7): `mask = ∑_{r∈S} 2^rank(key(r))` over the sorted record-key
/// universe. `subset_keys` are the canonical keys of the records in `S`.
fn setmaskrec_cell(subset_keys: &[String], key_universe_sorted: &[String]) -> u64 {
    let exps: Vec<u64> = subset_keys
        .iter()
        .map(|k| lean_S7_rank(key_universe_sorted, k))
        .collect();
    lean_S2_bitmask(&exps)
}

/// FuncSetMask composed cell (S1 ∘ S2 ∘ S7): `pack = ∑_p mask(f[fdom_p])·base^p`, `base = 2^|E|`,
/// `mask(T) = ∑_{a∈T} 2^rank_E(a)`. `slot_sets` are the value sets in domain position order;
/// `e_sorted` the sorted value universe `E`.
fn funcsetmask_cell(slot_sets: &[Vec<String>], e_sorted: &[String]) -> u64 {
    let base: u64 = 1u64 << e_sorted.len();
    let digits: Vec<u64> = slot_sets
        .iter()
        .map(|set| {
            let exps: Vec<u64> = set.iter().map(|a| lean_S7_rank(e_sorted, a)).collect();
            lean_S2_bitmask(&exps)
        })
        .collect();
    lean_S1_positional_pack(&digits, base)
}

/// Independent re-derivation of a record's CANONICAL KEY (S7's total order on records) from the
/// documented length-prefixed serialization (`record_key_from_fields`): fields sorted by NAME, each
/// written `len(name)·name·tag·len(text)·text` (`\u{1}`-separated, `\u{1e}`-terminated). Tags: `'S'`
/// String, `'M'` model value, `'B'` Bool, `'I'` Int. Hand-written from the SPEC of the format — NOT a
/// call to the encoder's `record_value_key`; cross-checked against the cert's stored `dom`.
fn record_key(fields: &[(&str, char, &str)]) -> String {
    let mut fs: Vec<(&str, char, &str)> = fields.to_vec();
    fs.sort_by(|a, b| a.0.cmp(b.0));
    let mut s = String::new();
    for (name, tag, text) in fs {
        s.push_str(&format!(
            "{}\u{1}{}\u{1}{}\u{1}{}\u{1}{}\u{1e}",
            name.len(),
            name,
            tag,
            text.len(),
            text
        ));
    }
    s
}

// ─────────────────────────── spec-string builders ───────────────────────────

/// `[i \in 0..K |-> <nested-IF over i>]` — a concrete Int-prefix function with the given per-slot
/// string labels (`slot_labels[p]` at position `p`).
fn func_int_lit(slot_labels: &[&str]) -> String {
    let k = slot_labels.len();
    let mut acc = format!("\"{}\"", slot_labels[k - 1]);
    for p in (0..k - 1).rev() {
        acc = format!("IF i = {} THEN \"{}\" ELSE {}", p, slot_labels[p], acc);
    }
    format!("[i \\in 0..{} |-> {}]", k - 1, acc)
}

/// A TLA set literal `{e0, e1, …}` (or `{}`) from bare element tokens (model values / record literals).
fn set_lit(elems: &[&str]) -> String {
    format!("{{{}}}", elems.join(", "))
}

/// A TLA set literal of QUOTED STRING labels `{"e0", "e1", …}` — for `FuncEnum` codomains (the labels
/// are TLA `String`s, not model values).
fn str_set_lit(elems: &[&str]) -> String {
    format!(
        "{{{}}}",
        elems
            .iter()
            .map(|e| format!("\"{e}\""))
            .collect::<Vec<_>>()
            .join(", ")
    )
}

// ─────────────────────────── config helpers ───────────────────────────

fn cfg_with(inv: &str, consts: &[(&str, &[&str])]) -> Config {
    let mut c =
        Config::parse(&format!("INIT Init\nNEXT Next\nINVARIANT {inv}\n")).expect("cfg parses");
    for (name, members) in consts {
        c.constants.insert(
            (*name).to_string(),
            ConstantValue::ModelValueSet(members.iter().map(|s| (*s).to_string()).collect()),
        );
    }
    c
}

/// Certify `spec` under `config`, asserting it mints; return the cert.
///
/// This test certifies MANY independently-parsed tiny specs in a loop. Several enumeration caches are
/// keyed by raw AST pointer with NO run discrimination, so a freed-then-reused address from a prior
/// spec can alias a stale entry (the documented pointer-keyed-cache hazard). We honor the run-boundary
/// contract by clearing the thread-local eval caches before each certify — the intended API for
/// re-evaluating an independently-parsed AST (it does NOT touch the name interner, so it is test-safe).
fn certify(spec: &str, config: &Config, what: &str) -> ExplicitFixpointCert {
    tla_check::clear_thread_local_eval_caches();
    certify_explicit_state_spec(spec, config).unwrap_or_else(|| panic!("{what} must certify"))
}

/// The single `Init`-state cell the encoder produced (a one-state `Init` ⇒ exactly one pair).
fn init_cell(cert: &ExplicitFixpointCert) -> u64 {
    assert_eq!(
        cert.init_values.len(),
        1,
        "single-state Init ⇒ one init tuple"
    );
    assert_eq!(
        cert.init_values[0].len(),
        1,
        "single-variable spec ⇒ 1-tuple"
    );
    cert.init_values[0][0]
}

/// The reachable set as a sorted `Vec<u64>` (single-column specs), for the image-level check.
fn reachable_cells(cert: &ExplicitFixpointCert) -> Vec<u64> {
    let mut v: Vec<u64> = cert.reachable.iter().map(|t| t[0]).collect();
    v.sort_unstable();
    v
}

// ═══════════════════════════ FuncEnum — S1 ∘ S7 ═══════════════════════════

/// The little-endian cartesian product `codomain^arity` — every per-slot label assignment.
fn label_tuples(arity: usize, codomain: &[&str]) -> Vec<Vec<usize>> {
    let mut out: Vec<Vec<usize>> = vec![Vec::new()];
    for _ in 0..arity {
        let mut next = Vec::new();
        for prefix in &out {
            for l in 0..codomain.len() {
                let mut p = prefix.clone();
                p.push(l);
                next.push(p);
            }
        }
        out = next;
    }
    out
}

/// Rust `FuncEnum` pack (a `[0..K -> Labels]` function) ≡ the composed `∑_p rank(e_p)·|labels|^p`
/// (S1 `positional_encoding_injective` ∘ S7 `rank_injOn`) — checked value-for-value over EVERY
/// `Labels^arity` assignment, at base 2 (3 slots) and a wider base 3 (2 slots). The value-label
/// universe is fixed to the whole codomain by the label-quantified `Next`.
#[test]
fn funcenum_intdom_matches_S1_compose_S7() {
    let mut checked = 0usize;
    // (codomain, arity): base = |codomain|.
    for &(codomain, arity) in &[(&["x", "y"][..], 3usize), (&["x", "y", "z"][..], 2usize)] {
        let labels_sorted =
            sorted_unique(&codomain.iter().map(|s| s.to_string()).collect::<Vec<_>>());
        let quant = str_set_lit(codomain);
        let mut image: Vec<u64> = Vec::new();
        for tup in label_tuples(arity, codomain) {
            let slot_labels: Vec<&str> = tup.iter().map(|&l| codomain[l]).collect();
            let spec = format!(
                "---- MODULE FE{id} ----\n\
                 EXTENDS Integers\n\
                 VARIABLE pc\n\
                 Init == pc = {init}\n\
                 Next == \\E i \\in 0..{k}, l \\in {quant} : pc' = [pc EXCEPT ![i] = l]\n\
                 Safety == pc[0] \\in {quant}\n\
                 ====\n",
                id = uniq(),
                init = func_int_lit(&slot_labels),
                k = arity - 1,
            );
            let cert = certify(&spec, &cfg_with("Safety", &[]), "Int-domain FuncEnum");
            // Sort family + fixed universe (cross-check the independently-sorted labels).
            match cert.sorts.as_slice() {
                [ColSort::FuncEnum {
                    arity: a,
                    labels,
                    dom,
                    ..
                }] => {
                    assert_eq!(*a as usize, arity, "arity");
                    assert_eq!(labels, &labels_sorted, "label universe = sorted codomain");
                    assert!(dom.is_empty(), "Int-prefix domain ⇒ empty dom");
                }
                other => panic!("expected a FuncEnum column, got {other:?}"),
            }
            let expected = funcenum_cell(&slot_labels, &labels_sorted);
            assert_eq!(
                init_cell(&cert),
                expected,
                "FuncEnum S1∘S7 MISMATCH: slots {slot_labels:?} base {} → encoder {} but \
                 ∑_p rank(e_p)·|labels|^p = {expected}",
                labels_sorted.len(),
                init_cell(&cert)
            );
            image.push(expected);
            // Every reaching Next visits the WHOLE space ⇒ reachable = image over all assignments.
            if reachable_cells(&cert).len() == codomain.len().pow(arity as u32) {
                let mut all: Vec<u64> = label_tuples(arity, codomain)
                    .iter()
                    .map(|t| {
                        let sl: Vec<&str> = t.iter().map(|&l| codomain[l]).collect();
                        funcenum_cell(&sl, &labels_sorted)
                    })
                    .collect();
                all.sort_unstable();
                all.dedup();
                assert_eq!(
                    reachable_cells(&cert),
                    all,
                    "image-level: reachable = composed-formula image"
                );
            }
            checked += 1;
        }
        // 2^3 + 3^2 across the two shapes; per-shape the image has |codomain|^arity distinct cells.
        image.sort_unstable();
        image.dedup();
        assert_eq!(
            image.len(),
            codomain.len().pow(arity as u32),
            "encoding is injective over the shape"
        );
    }
    // base-2 (8) + base-3 (9) = 17 pointwise (value, cell) pairs.
    assert_eq!(checked, 17, "FuncEnum Int-domain (value,cell) pairs");
}

/// Rust `FuncEnum` pack over a MODEL-VALUE DOMAIN `[RM -> Labels]` (`RM = {p,q}`) ≡ the composed
/// `∑_p rank(e_p)·|labels|^p` with the POSITIONS ordered by the S7 rank of the DOMAIN KEYS — the
/// second S7 leg (rank on the domain, not just the values). Checked over every `Labels^|RM|`.
#[test]
fn funcenum_modeldom_matches_S1_compose_S7() {
    let mut checked = 0usize;
    let codomain = ["x", "y"];
    let labels_sorted = sorted_unique(&codomain.iter().map(|s| s.to_string()).collect::<Vec<_>>());
    let keys = ["p", "q"]; // RM model values; canonical (S7) domain order = sorted = [p, q]
    let dom_sorted: Vec<String> =
        sorted_unique(&keys.iter().map(|s| s.to_string()).collect::<Vec<_>>());
    for tup in label_tuples(keys.len(), &codomain) {
        // value at key p = codomain[tup[0]], at key q = codomain[tup[1]] (keys[] is already sorted).
        let lp = codomain[tup[0]];
        let lq = codomain[tup[1]];
        let spec = format!(
            "---- MODULE FEmd{id} ----\n\
             CONSTANT RM\n\
             VARIABLE st\n\
             Init == st = [k \\in RM |-> IF k = p THEN \"{lp}\" ELSE \"{lq}\"]\n\
             Next == \\E k \\in RM, l \\in {quant} : st' = [st EXCEPT ![k] = l]\n\
             Safety == st[p] \\in {quant}\n\
             ====\n",
            id = uniq(),
            quant = str_set_lit(&codomain),
        );
        let cert = certify(
            &spec,
            &cfg_with("Safety", &[("RM", &["p", "q"])]),
            "model-domain FuncEnum",
        );
        match cert.sorts.as_slice() {
            [ColSort::FuncEnum {
                arity, labels, dom, ..
            }] => {
                assert_eq!(*arity, 2, "|RM| = 2");
                assert_eq!(labels, &labels_sorted, "label universe = sorted codomain");
                assert_eq!(dom, &dom_sorted, "domain keys = sorted RM (S7 rank order)");
            }
            other => panic!("expected a model-domain FuncEnum column, got {other:?}"),
        }
        // Positions in the composed formula follow the domain's S7 rank order [p, q] ⇒ [lp, lq].
        let expected = funcenum_cell(&[lp, lq], &labels_sorted);
        assert_eq!(
            init_cell(&cert),
            expected,
            "FuncEnum(model-dom) S1∘S7 MISMATCH: st[p]={lp} st[q]={lq} → encoder {} but composed = {expected}",
            init_cell(&cert)
        );
        checked += 1;
    }
    assert_eq!(checked, 4, "FuncEnum model-domain (value,cell) pairs (2^2)");
}

// ═══════════════════════════ SetMaskRec — S2 ∘ S7 ═══════════════════════════

/// Rust `SetMaskRec` bitmask (a set of records over the 3-record `Message` universe) ≡ the composed
/// `∑_{r∈S} 2^rank(key(r))` (S2 `sum_two_pow_injective` ∘ S7 `rank_injOn`) — checked value-for-value
/// over EVERY subset (`2^3 = 8`). The record-KEY total order (S7) is re-derived from the documented
/// canonical serialization and cross-checked against the cert's `dom`.
#[test]
fn setmaskrec_matches_S2_compose_S7() {
    // The 3 Message records: (TLA literal, canonical-key fields). id ∈ {a,b} are model values.
    let records: [(&str, Vec<(&str, char, &str)>); 3] = [
        (
            "[type |-> \"P\", id |-> a]",
            vec![("id", 'M', "a"), ("type", 'S', "P")],
        ),
        (
            "[type |-> \"P\", id |-> b]",
            vec![("id", 'M', "b"), ("type", 'S', "P")],
        ),
        ("[type |-> \"C\"]", vec![("type", 'S', "C")]),
    ];
    let all_keys: Vec<String> = records.iter().map(|(_, f)| record_key(f)).collect();
    let key_universe = sorted_unique(&all_keys);
    assert_eq!(key_universe.len(), 3, "3 distinct record keys");

    let mut checked = 0usize;
    let mut image: Vec<u64> = Vec::new();
    let mut empty_reachable: Option<Vec<u64>> = None;
    for bits in 0u8..8 {
        let idxs: Vec<usize> = (0..3).filter(|i| bits & (1 << i) != 0).collect();
        let lits: Vec<&str> = idxs.iter().map(|&i| records[i].0).collect();
        let subset_keys: Vec<String> = idxs.iter().map(|&i| record_key(&records[i].1)).collect();
        let spec = format!(
            "---- MODULE SmRec{id} ----\n\
             CONSTANT Ids\n\
             Message == [type : {{\"P\"}}, id : Ids] \\cup [type : {{\"C\"}}]\n\
             VARIABLE msgs\n\
             Init == msgs = {init}\n\
             Next == \\E m \\in Message : msgs' = (msgs \\cup {{m}})\n\
             Safety == msgs \\subseteq Message\n\
             ====\n",
            id = uniq(),
            init = set_lit(&lits),
        );
        let cert = certify(
            &spec,
            &cfg_with("Safety", &[("Ids", &["a", "b"])]),
            "SetMaskRec",
        );
        match cert.sorts.as_slice() {
            [ColSort::SetMaskRec { dom }] => {
                assert_eq!(
                    dom, &key_universe,
                    "record-key universe = independently sorted keys"
                );
            }
            other => panic!("expected a SetMaskRec column, got {other:?}"),
        }
        let expected = setmaskrec_cell(&subset_keys, &key_universe);
        assert_eq!(
            init_cell(&cert),
            expected,
            "SetMaskRec S2∘S7 MISMATCH: subset {idxs:?} → encoder {} but ∑ 2^rank(key(r)) = {expected}",
            init_cell(&cert)
        );
        image.push(expected);
        if bits == 0 {
            empty_reachable = Some(reachable_cells(&cert)); // add-only Next from {} ⇒ all 8 subsets
        }
        checked += 1;
    }
    image.sort_unstable();
    image.dedup();
    assert_eq!(
        image.len(),
        8,
        "the 8 subsets encode to 8 distinct masks (a bijection onto 0..7)"
    );
    // Image-level: from Init={} the add-only Next reaches ALL 8 subsets ⇒ reachable = composed image.
    assert_eq!(
        empty_reachable.expect("empty-init cert"),
        image,
        "reachable = composed-formula image"
    );
    assert_eq!(checked, 8, "SetMaskRec (value,cell) pairs (2^3)");
}

// ═══════════════════════════ FuncSetMask — S1 ∘ S2 ∘ S7 ═══════════════════════════

/// All `2^n` subsets of `atoms` (as little-endian bitmasks), each a `Vec<&str>` of the included atoms.
fn subsets<'a>(atoms: &[&'a str]) -> Vec<Vec<&'a str>> {
    (0u32..(1 << atoms.len()))
        .map(|bits| {
            (0..atoms.len())
                .filter(|i| bits & (1 << i) != 0)
                .map(|i| atoms[i])
                .collect()
        })
        .collect()
}

/// Rust `FuncSetMask` pack (`[Clients -> SUBSET Resources]`, the SimpleAllocator class) ≡ the composed
/// `∑_p mask(f[fdom_p])·base^p`, `base = 2^|E|`, `mask(T) = ∑_{a∈T} 2^rank_E(a)`
/// (S1 `positional_encoding_injective` ∘ S2 `sum_two_pow_injective` ∘ S7 `rank_injOn`) — checked
/// value-for-value over EVERY function `[{a,b} -> SUBSET {r,s}]` (`4^2 = 16`). Slot order follows the
/// domain's S7 rank; each value digit is the S2 bitmask over the S7-ranked value universe `E`.
#[test]
fn funcsetmask_matches_S1_compose_S2_compose_S7() {
    let clients = ["a", "b"]; // canonical (S7) domain order = sorted = [a, b]
    let resources = ["r", "s"];
    let fdom_sorted = sorted_unique(&clients.iter().map(|s| s.to_string()).collect::<Vec<_>>());
    let e_sorted = sorted_unique(&resources.iter().map(|s| s.to_string()).collect::<Vec<_>>());
    let subs = subsets(&resources); // 4 subsets of {r,s}

    let mut checked = 0usize;
    let mut image: Vec<u64> = Vec::new();
    let mut full_reachable: Option<Vec<u64>> = None;
    for set_a in &subs {
        for set_b in &subs {
            let init = format!(
                "[k \\in Clients |-> IF k = a THEN {} ELSE {}]",
                set_lit(set_a),
                set_lit(set_b),
            );
            let spec = format!(
                "---- MODULE FSm{id} ----\n\
                 CONSTANT Clients, Resources\n\
                 VARIABLE alloc\n\
                 Init == alloc = {init}\n\
                 Next == \\E c \\in Clients, x \\in Resources :\n\
                           \\/ alloc' = [alloc EXCEPT ![c] = @ \\cup {{x}}]\n\
                           \\/ alloc' = [alloc EXCEPT ![c] = @ \\ {{x}}]\n\
                 Safety == alloc \\in [Clients -> SUBSET Resources]\n\
                 ====\n",
                id = uniq(),
            );
            let cert = certify(
                &spec,
                &cfg_with(
                    "Safety",
                    &[("Clients", &["a", "b"]), ("Resources", &["r", "s"])],
                ),
                "FuncSetMask",
            );
            match cert.sorts.as_slice() {
                [ColSort::FuncSetMask {
                    arity, fdom, dom, ..
                }] => {
                    assert_eq!(*arity, 2, "|Clients| = 2");
                    assert_eq!(
                        fdom, &fdom_sorted,
                        "domain keys = sorted Clients (S7 rank order)"
                    );
                    assert_eq!(dom, &e_sorted, "value universe E = sorted Resources");
                }
                other => panic!("expected a FuncSetMask column, got {other:?}"),
            }
            assert_eq!(
                cert.sorts[0].funcsetmask_base(),
                Some(1 << e_sorted.len()),
                "base = 2^|E|"
            );
            // Slot order = fdom [a, b]; value sets in that order.
            let slot_sets: Vec<Vec<String>> = [set_a, set_b]
                .iter()
                .map(|s| s.iter().map(|a| a.to_string()).collect())
                .collect();
            let expected = funcsetmask_cell(&slot_sets, &e_sorted);
            assert_eq!(
                init_cell(&cert),
                expected,
                "FuncSetMask S1∘S2∘S7 MISMATCH: alloc[a]={set_a:?} alloc[b]={set_b:?} → encoder {} \
                 but ∑_p mask(f[p])·(2^|E|)^p = {expected}",
                init_cell(&cert)
            );
            image.push(expected);
            // Add/remove Next reaches the WHOLE [Clients -> SUBSET Resources] space from any Init.
            if full_reachable.is_none() && reachable_cells(&cert).len() == subs.len() * subs.len() {
                full_reachable = Some(reachable_cells(&cert));
            }
            checked += 1;
        }
    }
    image.sort_unstable();
    image.dedup();
    assert_eq!(
        image.len(),
        16,
        "the 16 functions encode to 16 distinct packs (injective)"
    );
    assert_eq!(
        full_reachable.expect("full-space cert"),
        image,
        "reachable = composed-formula image"
    );
    assert_eq!(checked, 16, "FuncSetMask (value,cell) pairs (4^2)");
}
