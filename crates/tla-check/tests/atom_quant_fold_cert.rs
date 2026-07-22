// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! BOUNDED-DOMAIN QUANTIFIER-FOLD extensions for the explicit-state certificate lane — the R4
//! "quantifier fold over a general set" cluster, of which SingleLaneBridge's `\A a,b \in Cars : …`
//! invariant is the headline. Three additive, reuse-first, fail-closed fold capabilities are exercised
//! here, each SOUND only over the COMPLETE fixed domain with a truth-exact body:
//!
//!   * ATOM-LITERAL EQUALITY `a = b` (`atom_lit_eq_form`): the residue of an atom-domain fold whose body
//!     compares two bound vars (`\A a,b∈D : … ⇒ a = b`). After the fold substitutes each bound var with
//!     its atom literal, `a = b` is a COMPILE-TIME constant (two atoms are equal iff same name/kind) ⇒ a
//!     `BoolLit`. Tested via the Leibniz TAUTOLOGY `\A a,b∈D : a = b ⇒ f[a] = f[b]` (TRUE, must certify)
//!     and its VIOLATED f-INJECTIVITY twin `\A a,b∈D : f[a] = f[b] ⇒ a = b` (FALSE at a non-injective
//!     reachable state — must DECLINE; folding `a = b` to TRUE instead of FALSE would falsely certify it,
//!     so this pins the constant's polarity).
//!   * UNION-DOMAIN atom fold (`materialize_atom_set` union arm): `\A x ∈ A ∪ B : P` over two SAME-kind
//!     config atom sets (SingleLaneBridge's `Cars == CarsRight ∪ CarsLeft`) folds over the sorted-deduped
//!     merge. A cross-KIND union (`{"a","b"} ∪ {1,2}`) DECLINES (fail-closed).
//!   * TUPLE-PATTERN over a CARTESIAN PRODUCT (`recognize_bounded_quant` pre-transform + the `cert_inline`
//!     `Times` arm): `\A <<a,b>> ∈ A × B : P` is the multi-var `\A a∈A, b∈B : P`, folded by the existing
//!     odometer.
//!
//! FAIL-CLOSED: a fold over a MAP comprehension `{f[k] : k∈D}` (a general set, not a fixed atom domain)
//! DECLINES. Every certificate here is kernel-re-checked (`verify_explicit_state_cert`) and round-trips.

#![cfg(feature = "clean-cic")]

use tla_check::explicit_fixpoint_cert::{
    certify_explicit_state_spec, verify_explicit_state_cert, ExplicitFixpointCert,
};
use tla_check::{Config, ConstantValue};

fn cfg_inv(inv: &str) -> Config {
    Config::parse(&format!("INIT Init\nNEXT Next\nINVARIANT {inv}\n")).expect("config parses")
}

/// A model-value-domain config (`D = {m1, m2}`) for the atom-literal-equality tests.
fn cfg_inv_mv(inv: &str) -> Config {
    let mut c = cfg_inv(inv);
    c.constants.insert(
        "D".to_string(),
        ConstantValue::ModelValueSet(vec!["m1".to_string(), "m2".to_string()]),
    );
    c
}

/// A model-value-domain Int-valued function `f: [{m1,m2} -> Nat]`, each key climbing `0->1->2`. The
/// Leibniz TAUTOLOGY `\A a,b ∈ D : a = b => f[a] = f[b]` holds on EVERY state (if the keys are equal the
/// values are equal), and it exercises the atom-literal-equality fold in the ANTECEDENT: after the
/// odometer substitutes `a := mᵢ, b := mⱼ`, the antecedent `mᵢ = mⱼ` folds to a `BoolLit` (TRUE on the
/// diagonal, FALSE off it), and the whole `⋀` rides the general R⊆Safety leg. Certifies + kernel-re-checks
/// + round-trips.
#[test]
fn atom_literal_equality_leibniz_tautology_certifies() {
    let spec = "---- MODULE AtomEqTrue ----\n\
                EXTENDS Naturals\n\
                CONSTANT D\n\
                VARIABLE f\n\
                Init == f = [k \\in D |-> 0]\n\
                Next == \\E k \\in D : f' = [f EXCEPT ![k] = IF f[k] < 2 THEN f[k] + 1 ELSE f[k]]\n\
                Safety == \\A a,b \\in D : a = b => f[a] = f[b]\n\
                ====\n";
    let cert = certify_explicit_state_spec(spec, &cfg_inv_mv("Safety")).expect(
        "Leibniz tautology over a model-value domain certifies (atom-literal equality fold)",
    );
    assert!(
        cert.safety_general.is_some(),
        "kernel-checked ⋀_{{s∈R}} Safety(s) leg"
    );
    assert!(verify_explicit_state_cert(&cert), "kernel re-check passes");
    let bytes = serde_json::to_vec(&cert).expect("serialize");
    let back: ExplicitFixpointCert = serde_json::from_slice(&bytes).expect("deserialize");
    assert_eq!(cert, back, "atom-eq fold cert round-trips");
    assert!(
        verify_explicit_state_cert(&back),
        "re-loaded cert still verifies"
    );
}

/// The VIOLATED twin — f-INJECTIVITY `\A a,b ∈ D : f[a] = f[b] => a = b`. At `Init` both keys map to `0`,
/// so `f[m1] = f[m2]` while `m1 # m2` — the invariant is FALSE. certify MUST DECLINE (the general safety
/// leg cannot reduce `⋀_{s∈R} Safety(s)` to `Bool.true`). This is the SOUNDNESS GATE for the atom-literal
/// equality fold: had `m1 = m2` folded to TRUE (wrong), the instance `f[m1]=f[m2] => m1=m2` would collapse
/// to TRUE and the violated invariant would be FALSELY certified. Because it folds to FALSE, the instance
/// is `~(f[m1]=f[m2])`, which the non-injective Init breaks ⇒ correct DECLINE (certify≡check).
#[test]
fn atom_literal_injectivity_violated_twin_declines() {
    let spec = "---- MODULE AtomEqBad ----\n\
                EXTENDS Naturals\n\
                CONSTANT D\n\
                VARIABLE f\n\
                Init == f = [k \\in D |-> 0]\n\
                Next == \\E k \\in D : f' = [f EXCEPT ![k] = IF f[k] < 2 THEN f[k] + 1 ELSE f[k]]\n\
                Safety == \\A a,b \\in D : f[a] = f[b] => a = b\n\
                ====\n";
    assert!(
        certify_explicit_state_spec(spec, &cfg_inv_mv("Safety")).is_none(),
        "a violated f-injectivity invariant DECLINES — the atom-eq fold must produce the FALSE constant"
    );
}

/// UNION-DOMAIN atom fold: `f: [({"a","b"} ∪ {"c","d"}) -> Nat]` and the invariant
/// `\A k ∈ {"a","b"} ∪ {"c","d"} : f[k] <= 2` over the union of two `String`-atom sets — the SingleLaneBridge
/// `Cars == CarsRight ∪ CarsLeft` shape (after the constants inline to two `String` `SetEnum`s). The union
/// materializes to the sorted-deduped `[a,b,c,d]` and each `f[k]` resolves to its slot. Certifies +
/// kernel-re-checks.
#[test]
fn union_domain_atom_fold_certifies() {
    let spec = "---- MODULE UnionDom ----\n\
                EXTENDS Naturals\n\
                VARIABLE f\n\
                Init == f = [k \\in ({\"a\",\"b\"} \\union {\"c\",\"d\"}) |-> 0]\n\
                Next == \\E k \\in ({\"a\",\"b\"} \\union {\"c\",\"d\"}) : \
                          f' = [f EXCEPT ![k] = IF f[k] < 2 THEN f[k] + 1 ELSE f[k]]\n\
                Safety == \\A k \\in ({\"a\",\"b\"} \\union {\"c\",\"d\"}) : f[k] <= 2\n\
                ====\n";
    let cert = certify_explicit_state_spec(spec, &cfg_inv("Safety"))
        .expect("union-of-String-atom-sets quantifier fold certifies");
    assert!(
        cert.safety_general.is_some(),
        "kernel-checked general safety leg"
    );
    assert!(verify_explicit_state_cert(&cert), "kernel re-check passes");
}

/// TUPLE-PATTERN over a CARTESIAN PRODUCT: `\A <<a,b>> ∈ {"a","b"} × {"c","d"} : f[a] <= 2 /\ f[b] <= 2`
/// — SingleLaneBridge's `\A <<r,l>> ∈ CarsRight × CarsLeft : …` shape. The pre-transform rewrites it to the
/// multi-var `\A a ∈ {"a","b"}, b ∈ {"c","d"}` which the atom odometer folds over the 2×2 product.
/// Certifies + kernel-re-checks.
#[test]
fn tuple_product_atom_fold_certifies() {
    let spec = "---- MODULE TupleProd ----\n\
                EXTENDS Naturals\n\
                VARIABLE f\n\
                Init == f = [k \\in ({\"a\",\"b\"} \\union {\"c\",\"d\"}) |-> 0]\n\
                Next == \\E k \\in ({\"a\",\"b\"} \\union {\"c\",\"d\"}) : \
                          f' = [f EXCEPT ![k] = IF f[k] < 2 THEN f[k] + 1 ELSE f[k]]\n\
                Safety == \\A <<a,b>> \\in ({\"a\",\"b\"} \\X {\"c\",\"d\"}) : f[a] <= 2 /\\ f[b] <= 2\n\
                ====\n";
    let cert = certify_explicit_state_spec(spec, &cfg_inv("Safety"))
        .expect("tuple-pattern over a cartesian product certifies");
    assert!(
        cert.safety_general.is_some(),
        "kernel-checked general safety leg"
    );
    assert!(verify_explicit_state_cert(&cert), "kernel re-check passes");
}

/// FAIL-CLOSED: a fold over a MAP comprehension `{f[k] : k ∈ {"a","b"}}` — a GENERAL set (a set-builder
/// image, NOT a fixed atom domain nor a bitmask-encodable column) — must DECLINE. The domain does not
/// materialize to a fixed element list and is not a recognizable set, so `recognize_bounded_quant` returns
/// `None` and no certificate is emitted (no over-approximation of the domain).
#[test]
fn map_comprehension_domain_declines() {
    let spec = "---- MODULE MapCompDom ----\n\
                EXTENDS Naturals\n\
                VARIABLE f\n\
                Init == f = [k \\in ({\"a\",\"b\"} \\union {\"c\",\"d\"}) |-> 0]\n\
                Next == \\E k \\in ({\"a\",\"b\"} \\union {\"c\",\"d\"}) : \
                          f' = [f EXCEPT ![k] = IF f[k] < 2 THEN f[k] + 1 ELSE f[k]]\n\
                Safety == \\A x \\in { f[k] : k \\in ({\"a\",\"b\"}) } : x <= 2\n\
                ====\n";
    assert!(
        certify_explicit_state_spec(spec, &cfg_inv("Safety")).is_none(),
        "a fold over a map-comprehension (general) set DECLINES (fail-closed)"
    );
}

/// FAIL-CLOSED: a CROSS-KIND union domain `{"a","b"} ∪ {1,2}` (a `String`-atom set unioned with an Int set)
/// has no single column kind, so the union arm of `materialize_atom_set` DECLINES rather than fold. The
/// whole quantifier then declines — no certificate.
#[test]
fn cross_kind_union_domain_declines() {
    let spec = "---- MODULE CrossKindUnion ----\n\
                EXTENDS Naturals\n\
                VARIABLE f\n\
                Init == f = [k \\in ({\"a\",\"b\"} \\union {\"c\",\"d\"}) |-> 0]\n\
                Next == \\E k \\in ({\"a\",\"b\"} \\union {\"c\",\"d\"}) : \
                          f' = [f EXCEPT ![k] = IF f[k] < 2 THEN f[k] + 1 ELSE f[k]]\n\
                Safety == \\A k \\in ({\"a\",\"b\"} \\union {1,2}) : f[k] <= 2\n\
                ====\n";
    assert!(
        certify_explicit_state_spec(spec, &cfg_inv("Safety")).is_none(),
        "a cross-kind union domain DECLINES (fail-closed)"
    );
}
