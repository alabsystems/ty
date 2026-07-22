// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! ATOM-DOMAIN (model-value / `String`) Int/Bool/enum-VALUED FUNCTION cell encoding for the explicit-state
//! certificate lane — the DieHarder `contents \in [Jug -> Nat]` class (`Jug` a `String`-atom set). This is
//! the DUAL of `func_enum_cell_cert.rs`: `FuncEnum` is an atom/Int-domain function with ENUM values;
//! `ColSort::Func{dom, dom_kind, ..}` is an atom/Int-domain function with Int/Bool/enum VALUES. The
//! `dom`/`dom_kind` mechanism (a sorted key list + kind) is REUSED verbatim from `FuncEnum` (via
//! `func_enum_domain_keys`), so a domain key `f[k]` resolves to the SAME positional slot the encoder packed
//! it into, kind-checked. Covered:
//!
//!   * a `String`-domain Int-valued function certifies via BOTH the function-set membership form
//!     `f \in [D -> Nat]` (the `DOMAIN f = D` conjunct discharged by the stored `dom`) AND the bounded
//!     quantifier `\A k \in D : f[k] <= N` (the atom fold resolves each `String` key `k` to its slot);
//!   * a model-value-domain Int-valued function certifies the same way;
//!   * a VIOLATED twin DECLINES (never a false certificate) — the certify≡check gate;
//!   * the CROSS-KIND guard: a `String`-literal key against a MODEL-value-declared domain DECLINES (a
//!     `String` "m1" and a model value named `m1` are DISTINCT TLA values, never one slot) — fail-closed;
//!   * BYTE-COMPAT: an Int-prefix `Func` (empty `dom`, `Model` kind) serializes with NEITHER `dom` NOR
//!     `dom_kind` keys, byte-identical to a pre-domain-shape cert.

#![cfg(feature = "clean-cic")]

use tla_check::explicit_fixpoint_cert::{
    certify_explicit_state_spec, verify_explicit_state_cert, ColSort, EnumKind,
    ExplicitFixpointCert,
};
use tla_check::{Config, ConstantValue};

fn cfg_inv(inv: &str) -> Config {
    Config::parse(&format!("INIT Init\nNEXT Next\nINVARIANT {inv}\n")).expect("config parses")
}

/// A `String`-domain Int-valued function `f: [{"a","b"} -> Nat]`, each key climbing `0 -> 1 -> 2` — the
/// DieHarder `contents` class. Certifies via BOTH invariants: the function-set membership `f \in
/// [{"a","b"} -> Nat]` (domain discharged by the stored `dom = [a, b]`, each Int digit `>= 0`) AND the
/// bounded quantifier `\A k \in {"a","b"} : f[k] <= 2` (the fold resolves `f["a"]`, `f["b"]` to slots 0/1).
/// R is the 9 packs (`f["a"] + f["b"]*10`), the cert kernel-re-verifies + round-trips, and the sort carries
/// `dom = [a, b]`, `dom_kind = Str`.
#[test]
fn func_string_domain_int_values_certifies_membership_and_quantifier() {
    let spec = "---- MODULE StrDomFunc ----\n\
                EXTENDS Naturals\n\
                VARIABLE f\n\
                Init == f = [k \\in {\"a\",\"b\"} |-> 0]\n\
                Next == \\E k \\in {\"a\",\"b\"} : \
                          f' = [f EXCEPT ![k] = IF f[k] < 2 THEN f[k] + 1 ELSE f[k]]\n\
                Safety == /\\ f \\in [{\"a\",\"b\"} -> Nat]\n\
                          /\\ \\A k \\in {\"a\",\"b\"} : f[k] <= 2\n\
                ====\n";
    let cert = certify_explicit_state_spec(spec, &cfg_inv("Safety"))
        .expect("String-domain Int-valued Func certifies (membership + quantifier)");
    assert_eq!(
        cert.sorts,
        vec![ColSort::Func {
            base: 10,
            arity: 2,
            cells: vec![],
            dom: vec!["a".to_string(), "b".to_string()],
            dom_kind: EnumKind::Str,
        }],
        "the column is a positional Int-value Func pack over a sorted String-atom domain"
    );
    // R = the 9 packs f["a"] + f["b"]*10 for f["a"],f["b"] in {0,1,2}.
    let expected: Vec<Vec<u64>> = (0u64..3)
        .flat_map(|b| (0u64..3).map(move |a| vec![a + b * 10]))
        .collect();
    let mut got = cert.reachable.clone();
    got.sort();
    let mut want = expected;
    want.sort();
    assert_eq!(
        got, want,
        "R is the 9 positional packs over the String domain"
    );
    assert!(
        cert.safety_general.is_some(),
        "kernel-checked ⋀_{{s∈R}} Safety(s) leg"
    );
    assert!(verify_explicit_state_cert(&cert), "kernel re-check passes");

    // Serde round-trip.
    let bytes = serde_json::to_vec(&cert).expect("serialize");
    let back: ExplicitFixpointCert = serde_json::from_slice(&bytes).expect("deserialize");
    assert_eq!(cert, back, "String-domain Func cert round-trips");
    assert!(
        verify_explicit_state_cert(&back),
        "re-loaded cert still verifies"
    );
}

/// The VIOLATED twin: the SAME `String`-domain function but an invariant `f[k] <= 1` that the reachable
/// value `2` breaks. certify MUST DECLINE (the general safety leg cannot reduce `⋀_{s∈R} Safety(s)` to
/// `Bool.true` when some state violates it) — the certify≡check soundness gate (never a false certificate).
#[test]
fn func_string_domain_violated_invariant_declines() {
    let spec = "---- MODULE StrDomFuncBad ----\n\
                EXTENDS Naturals\n\
                VARIABLE f\n\
                Init == f = [k \\in {\"a\",\"b\"} |-> 0]\n\
                Next == \\E k \\in {\"a\",\"b\"} : \
                          f' = [f EXCEPT ![k] = IF f[k] < 2 THEN f[k] + 1 ELSE f[k]]\n\
                Safety == \\A k \\in {\"a\",\"b\"} : f[k] <= 1\n\
                ====\n";
    assert!(
        certify_explicit_state_spec(spec, &cfg_inv("Safety")).is_none(),
        "a violated invariant over a String-domain function DECLINES (never a false certificate)"
    );
}

/// A MODEL-value-domain Int-valued function `f: [{m1,m2} -> Nat]` (config CONSTANT `Dom = {m1, m2}`). The
/// atom fold resolves each model-value key `Ident` to its slot. Certifies with `dom = [m1, m2]`, `dom_kind
/// = Model` (the skip-serialized default — byte-omitted, but `PartialEq`-equal).
#[test]
fn func_model_value_domain_int_values_certifies() {
    let spec = "---- MODULE MvDomFunc ----\n\
                EXTENDS Naturals\n\
                CONSTANT Dom\n\
                VARIABLE f\n\
                Init == f = [k \\in Dom |-> 0]\n\
                Next == \\E k \\in Dom : f' = [f EXCEPT ![k] = IF f[k] < 2 THEN f[k] + 1 ELSE f[k]]\n\
                Safety == \\A k \\in Dom : f[k] <= 2\n\
                ====\n";
    let mut config = cfg_inv("Safety");
    config.constants.insert(
        "Dom".to_string(),
        ConstantValue::ModelValueSet(vec!["m1".to_string(), "m2".to_string()]),
    );
    let cert = certify_explicit_state_spec(spec, &config)
        .expect("model-value-domain Int-valued Func certifies");
    assert_eq!(
        cert.sorts,
        vec![ColSort::Func {
            base: 10,
            arity: 2,
            cells: vec![],
            dom: vec!["m1".to_string(), "m2".to_string()],
            dom_kind: EnumKind::Model,
        }],
        "the column is a positional Int-value Func pack over a model-value domain"
    );
    assert!(verify_explicit_state_cert(&cert), "kernel re-check passes");

    // Round-trip AND byte-compat: a `Model`-kind dom_kind is skip-serialized (only the `dom` key emits).
    let bytes = serde_json::to_vec(&cert).expect("serialize");
    let text = String::from_utf8(bytes.clone()).unwrap();
    assert!(
        !text.contains("dom_kind"),
        "a Model dom_kind is skip-serialized: {text}"
    );
    let back: ExplicitFixpointCert = serde_json::from_slice(&bytes).expect("deserialize");
    assert_eq!(cert, back, "model-value-domain Func cert round-trips");
}

/// CROSS-KIND GUARD: the SAME model-value-domain function, but an invariant quantifying over `String`
/// literals `{"m1","m2"}` (NOT the model-value CONSTANT). The atom fold substitutes each bound var with a
/// `String` literal, so `f["m1"]` must resolve a `String` key against the column's `Model`-kind `dom` — and
/// the kind guard REFUSES it (a `String` "m1" and a model value named `m1` are DISTINCT TLA values). The
/// invariant recognizer declines ⇒ certify DECLINES (fail-closed). WITHOUT the kind tag this would wrongly
/// certify by conflating the two atom kinds at identical names.
#[test]
fn func_cross_kind_string_key_on_model_domain_declines() {
    let spec = "---- MODULE MvDomFuncStrKey ----\n\
                EXTENDS Naturals\n\
                CONSTANT Dom\n\
                VARIABLE f\n\
                Init == f = [k \\in Dom |-> 0]\n\
                Next == \\E k \\in Dom : f' = [f EXCEPT ![k] = IF f[k] < 2 THEN f[k] + 1 ELSE f[k]]\n\
                Safety == \\A k \\in {\"m1\",\"m2\"} : f[k] <= 2\n\
                ====\n";
    let mut config = cfg_inv("Safety");
    config.constants.insert(
        "Dom".to_string(),
        ConstantValue::ModelValueSet(vec!["m1".to_string(), "m2".to_string()]),
    );
    assert!(
        certify_explicit_state_spec(spec, &config).is_none(),
        "a String key against a Model-declared domain DECLINES (the cross-kind guard, fail-closed)"
    );
}

/// BYTE-COMPAT: an INT-PREFIX Int-valued function (the historical `[0..n-1 -> Int]` shape) still certifies
/// with an EMPTY `dom` and `Model` dom_kind, and serializes with NEITHER `dom` NOR `dom_kind` — byte-
/// identical to a pre-domain-shape cert. The domain descriptor is purely additive for atom domains.
#[test]
fn func_int_prefix_domain_stays_byte_identical() {
    let spec = "---- MODULE IntPrefixFunc ----\n\
                EXTENDS Integers\n\
                VARIABLE f\n\
                Init == f = [i \\in 0..1 |-> 0]\n\
                Next == f' = [f EXCEPT ![0] = IF f[0] < 2 THEN f[0] + 1 ELSE f[0]]\n\
                Safety == f[0] <= 2\n\
                ====\n";
    let cert = certify_explicit_state_spec(spec, &cfg_inv("Safety"))
        .expect("Int-prefix Int-valued Func certifies");
    assert_eq!(
        cert.sorts,
        vec![ColSort::Func {
            base: 10,
            arity: 2,
            cells: vec![],
            dom: vec![],
            dom_kind: EnumKind::Model,
        }],
        "an Int-prefix Func has an empty dom / Model kind"
    );
    let text = String::from_utf8(serde_json::to_vec(&cert).unwrap()).unwrap();
    assert!(
        !text.contains("dom"),
        "an Int-prefix Func omits dom/dom_kind (byte-compat): {text}"
    );
    assert!(verify_explicit_state_cert(&cert), "kernel re-check passes");
}
