// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! SET-of-ATOMS (`ColSort::SetMask`) cell encoding for the explicit-state certificate lane — a state
//! variable holding a set `S ⊆ D` over a SMALL FIXED FINITE ATOM UNIVERSE `D` (config CONSTANT model
//! values, or `String` atoms), encoded as the `|D|`-bit BITMASK `Σ_{a∈S} 2^idx(a)` (bit `idx(a)` ⟺
//! `a ∈ S`). This is the ATOM-DOMAIN analogue of the Int-valued `ColSort::Set` bitmask: the universe `D`
//! is the SORTED cross-state atom union (the same `Enum`/`FuncEnum` grow mechanism), so the encoding is a
//! BIJECTION between subsets of `D` and `|D|`-bit values.
//!
//! SOUNDNESS proof (the certify≡check backstop, in-process):
//!   * a set variable `S ⊆ {a,b,c}` with `Init S={}`, `Next` adding elements, and a real safety invariant
//!     over `S` (`S ⊆ Vals`, `a ∈ S`, `Cardinality(S) ≤ 1`) CERTIFIES + kernel-re-checks + full-Leg-E
//!     accepts; the reachable set is exactly the reachable bitmasks;
//!   * a VIOLATED twin (an invariant the set-ops falsify — `a ∉ S` while `Next` adds `a`, or a violated
//!     `S ⊆ T`) is NOT CERTIFIED — the kernel `⋀_{s∈R} Safety(s) ⇒ Bool.true` leg rejects the offending
//!     state, so the bitmask + membership/⊆/Cardinality folds do NOT hide a violation (the recognizer DOES
//!     build the obligation — proven by the TRUE sibling with the SAME invariant shape certifying);
//!   * a cross-`dom` guard: `S ⊆ T` / `S = T` between two `SetMask` columns needs matching universes;
//!   * a `String`-atom universe certifies the same way (kind-guarded against model values);
//!   * tamper: a mutated bitmask cell / mutated `dom` fails the kernel re-check / Leg-E binding.
//!
//! The corresponding `ty check` cross-check (each TRUE spec `No error`, each VIOLATED spec
//! `Invariant … is violated`, same state count) is exercised end-to-end at the CLI; here the certify side
//! is the arbiter (certify must never certify what check flags).

#![cfg(feature = "clean-cic")]

use tla_check::explicit_fixpoint_cert::{
    certify_explicit_state_spec, verify_explicit_state_cert, ColSort, EnumKind,
    ExplicitFixpointCert,
};
use tla_check::{Config, ConstantValue};

/// A config with three scalar model-value constants `a`, `b`, `c` (each `x = x`) and the given invariant.
/// The spec defines `Vals == {a,b,c}` (inlined to a model-value set literal), so `S ⊆ Vals` / `d ∈ Vals`
/// resolve against the observed universe.
fn cfg_abc(inv: &str) -> Config {
    let mut config =
        Config::parse(&format!("INIT Init\nNEXT Next\nINVARIANT {inv}\n")).expect("config parses");
    for name in ["a", "b", "c"] {
        config
            .constants
            .insert(name.to_string(), ConstantValue::Value(name.to_string()));
    }
    config
}

/// A config with a model-value SET constant `Vals = {a,b,c}` (the MCConsensus `Value <- {a,b,c}` shape)
/// and the given invariant.
fn cfg_valset(inv: &str) -> Config {
    let mut config =
        Config::parse(&format!("INIT Init\nNEXT Next\nINVARIANT {inv}\n")).expect("config parses");
    config.constants.insert(
        "Vals".to_string(),
        ConstantValue::ModelValueSet(vec!["a".to_string(), "b".to_string(), "c".to_string()]),
    );
    config
}

fn setmask(dom: &[&str], kind: EnumKind) -> ColSort {
    ColSort::SetMask {
        dom: dom.iter().map(|s| s.to_string()).collect(),
        dom_kind: kind,
    }
}

/// TRUE — a monotone-growing model-value set `S ⊆ {a,b,c}`. `Init S={}`, `Next` adds any element, so R is
/// ALL 8 subsets (masks 0..7). The invariant `S ⊆ Vals` is a tautology over the universe. Certifies with
/// the column sort `SetMask{dom=[a,b,c], Model}`, kernel-re-verifies, and full Leg-E accepts.
#[test]
fn model_value_setmask_subset_certifies() {
    let spec = "---- MODULE SmSubset ----\n\
                CONSTANTS a, b, c\n\
                Vals == {a, b, c}\n\
                VARIABLE S\n\
                Init == S = {}\n\
                Next == \\E d \\in Vals : S' = (S \\union {d})\n\
                Safety == S \\subseteq Vals\n\
                ====\n";
    let config = cfg_abc("Safety");
    let cert = certify_explicit_state_spec(spec, &config).expect("model-value SetMask certifies");
    assert_eq!(
        cert.sorts,
        vec![setmask(&["a", "b", "c"], EnumKind::Model)],
        "the column is a SetMask over the sorted model-value universe {{a,b,c}}"
    );
    // R = all 8 subsets of {a,b,c}, as bitmasks 0..7 (each a 1-tuple).
    assert_eq!(cert.reachable.len(), 8, "R is all 8 subsets");
    assert_eq!(
        cert.reachable,
        (0u64..8).map(|m| vec![m]).collect::<Vec<_>>(),
        "R is exactly the bitmasks 0..7"
    );
    assert!(
        cert.safety_pred.is_some(),
        "the ⊆ invariant rides the general safety leg"
    );
    assert!(
        cert.safety_general.is_some(),
        "kernel-checked ⋀_{{s∈R}} Safety(s) leg"
    );
    assert!(verify_explicit_state_cert(&cert), "kernel re-check passes");

    // Serde round-trip.
    let bytes = serde_json::to_vec(&cert).expect("serialize");
    let back: ExplicitFixpointCert = serde_json::from_slice(&bytes).expect("deserialize");
    assert_eq!(cert, back, "SetMask cert round-trips");
    assert!(
        verify_explicit_state_cert(&back),
        "the re-loaded SetMask cert verifies"
    );

    // Full Leg-E: build a SafetyCertificate and re-verify end-to-end (re-enumerates + re-derives the sort).
    let sc = tla_check::cert::build_explicit_fixpoint_certificate(spec, &config, cert);
    let report = tla_check::cert::verify_safety_certificate(&sc);
    assert!(
        report.accepted,
        "full Leg-E re-check accepts: {}",
        report.detail
    );
}

/// TRUE — the MCConsensus shape via a model-value SET CONSTANT `Vals = {a,b,c}` and the `S ⊆ Vals` invariant
/// (the constant resolved through the config `mvsets`, not an inlined literal). Certifies with the same
/// `SetMask` sort.
#[test]
fn model_value_setmask_via_constant_set_certifies() {
    let spec = "---- MODULE SmConst ----\n\
                CONSTANT Vals\n\
                VARIABLE S\n\
                Init == S = {}\n\
                Next == \\E d \\in Vals : S' = (S \\union {d})\n\
                Safety == S \\subseteq Vals\n\
                ====\n";
    let config = cfg_valset("Safety");
    let cert = certify_explicit_state_spec(spec, &config)
        .expect("model-value SetMask via a CONSTANT set certifies");
    assert_eq!(cert.sorts, vec![setmask(&["a", "b", "c"], EnumKind::Model)]);
    assert_eq!(cert.reachable.len(), 8);
    assert!(verify_explicit_state_cert(&cert), "kernel re-check passes");
}

/// TRUE — a POSITIVE membership invariant `a \in S` that holds on every reachable state (`Init S={a}`,
/// `Next` only ever adds `b`/`c`, never removes `a`). Proves the `SetMem` bit-test fold is faithful in the
/// TRUE direction — the same recognizer path its VIOLATED twin below declines on.
#[test]
fn setmask_membership_true_certifies() {
    let spec = "---- MODULE SmMemTrue ----\n\
                CONSTANTS a, b, c\n\
                Vals == {a, b, c}\n\
                VARIABLE S\n\
                Init == S = {a}\n\
                Next == \\E d \\in Vals : S' = (S \\union {d})\n\
                Safety == a \\in S\n\
                ====\n";
    let config = cfg_abc("Safety");
    let cert =
        certify_explicit_state_spec(spec, &config).expect("a∈S holds on all reachable states");
    assert!(matches!(cert.sorts.as_slice(), [ColSort::SetMask { dom, .. }] if dom.len() == 3));
    assert!(
        cert.safety_pred.is_some(),
        "the membership invariant is recognized (SetMem)"
    );
    assert!(verify_explicit_state_cert(&cert), "kernel re-check passes");
}

/// VIOLATED twin — the SAME `SetMask` column and the SAME membership recognizer path, but the invariant
/// `a \notin S` is FALSE on the reachable state `{a}` (`Next` adds `a`). The recognizer BUILDS the `SetNotMem`
/// obligation (its TRUE sibling above certifies the shape); the kernel `⋀_{s∈R} Safety(s)` leg then reduces
/// to `Bool.false` on `{a}` and REJECTS ⇒ NOT CERTIFIED. Proves the bitmask + membership fold do not hide a
/// violation (the certify≡check gate: `ty check` reports `Invariant Safety is violated` on this spec).
#[test]
fn setmask_membership_violated_declines() {
    let spec = "---- MODULE SmMemViol ----\n\
                CONSTANTS a, b, c\n\
                Vals == {a, b, c}\n\
                VARIABLE S\n\
                Init == S = {}\n\
                Next == \\E d \\in Vals : S' = (S \\union {d})\n\
                Safety == a \\notin S\n\
                ====\n";
    let config = cfg_abc("Safety");
    assert!(
        certify_explicit_state_spec(spec, &config).is_none(),
        "a violated membership invariant (a∈S reachable) must NOT certify"
    );
}

/// TRUE — a cross-column subset `S ⊆ T` maintained as an invariant: `S` grows only in lock-step with `T`
/// (the second disjunct adds `d` to BOTH), so `S ⊆ T` always. Both columns observe the whole universe
/// `{a,b,c}` (identical `dom`), so the `SetSubseteq(Var(S), Var(T))` recognizer fires. Certifies.
#[test]
fn setmask_cross_column_subset_true_certifies() {
    let spec = "---- MODULE SmSubTrue ----\n\
                CONSTANTS a, b, c\n\
                Vals == {a, b, c}\n\
                VARIABLES S, T\n\
                Init == S = {} /\\ T = {}\n\
                Next == \\E d \\in Vals :\n\
                          \\/ (T' = (T \\union {d}) /\\ S' = S)\n\
                          \\/ (S' = (S \\union {d}) /\\ T' = (T \\union {d}))\n\
                Safety == S \\subseteq T\n\
                ====\n";
    let config = cfg_abc("Safety");
    let cert = certify_explicit_state_spec(spec, &config).expect("S⊆T is maintained ⇒ certifies");
    assert!(
        cert.sorts
            .iter()
            .all(|s| matches!(s, ColSort::SetMask { dom, .. } if dom.len() == 3)),
        "both columns are SetMask over the same 3-atom universe"
    );
    assert!(
        cert.safety_pred.is_some(),
        "the cross-column ⊆ invariant is recognized"
    );
    assert!(verify_explicit_state_cert(&cert), "kernel re-check passes");
}

/// VIOLATED twin — `S` and `T` grow INDEPENDENTLY over `{a,b,c}`, so `S ⊆ T` is FALSE on reachable states
/// (e.g. `S={a}, T={}`). Same recognizer path (`SetSubseteq` over two same-universe columns — both observe
/// all atoms), but the kernel leg rejects the offending state ⇒ NOT CERTIFIED. (`ty check` reports the
/// violation.)
#[test]
fn setmask_cross_column_subset_violated_declines() {
    let spec = "---- MODULE SmSubViol ----\n\
                CONSTANTS a, b, c\n\
                Vals == {a, b, c}\n\
                VARIABLES S, T\n\
                Init == S = {} /\\ T = {}\n\
                Next == \\E d \\in Vals :\n\
                          \\/ (T' = (T \\union {d}) /\\ S' = S)\n\
                          \\/ (S' = (S \\union {d}) /\\ T' = T)\n\
                Safety == S \\subseteq T\n\
                ====\n";
    let config = cfg_abc("Safety");
    assert!(
        certify_explicit_state_spec(spec, &config).is_none(),
        "a violated S⊆T (independent growth) must NOT certify"
    );
}

/// TRUE — the MCConsensus safety property `Cardinality(S) <= 1` over a model-value set, with a `Next` that
/// only ever makes `S` a singleton or empty. Exercises the `SetCard` POPCOUNT fold + `IsFiniteSet`
/// tautology. Certifies; R is exactly `{ {}, {a}, {b}, {c} }` (masks 0,1,2,4).
#[test]
fn setmask_cardinality_certifies() {
    let spec = "---- MODULE SmCard ----\n\
                EXTENDS Naturals, FiniteSets\n\
                CONSTANTS a, b, c\n\
                Vals == {a, b, c}\n\
                VARIABLE S\n\
                Init == S = {}\n\
                Next == (S = {}) /\\ (\\E d \\in Vals : S' = {d})\n\
                Safety == /\\ IsFiniteSet(S)\n\
                          /\\ Cardinality(S) <= 1\n\
                ====\n";
    let config = cfg_abc("Safety");
    let cert =
        certify_explicit_state_spec(spec, &config).expect("Cardinality(S)<=1 holds ⇒ certifies");
    assert!(matches!(cert.sorts.as_slice(), [ColSort::SetMask { dom, .. }] if dom.len() == 3));
    // R = {∅, {a}, {b}, {c}} = masks {0, 1, 2, 4}.
    let masks: Vec<u64> = cert.reachable.iter().map(|t| t[0]).collect();
    assert_eq!(
        masks,
        vec![0, 1, 2, 4],
        "R is the empty set and the three singletons"
    );
    assert!(verify_explicit_state_cert(&cert), "kernel re-check passes");
}

/// VIOLATED twin — the SAME singleton-reaching `Next`, but the invariant `Cardinality(S) = 0` is FALSE on
/// every singleton state. The `SetCard` popcount fold reduces to `1 ≠ 0` on `{a}` ⇒ the kernel rejects ⇒
/// NOT CERTIFIED. Proves the popcount fold does not hide a violation.
#[test]
fn setmask_cardinality_violated_declines() {
    let spec = "---- MODULE SmCardViol ----\n\
                EXTENDS Naturals, FiniteSets\n\
                CONSTANTS a, b, c\n\
                Vals == {a, b, c}\n\
                VARIABLE S\n\
                Init == S = {}\n\
                Next == (S = {}) /\\ (\\E d \\in Vals : S' = {d})\n\
                Safety == Cardinality(S) = 0\n\
                ====\n";
    let config = cfg_abc("Safety");
    assert!(
        certify_explicit_state_spec(spec, &config).is_none(),
        "Cardinality(S)=0 is false on the singleton states ⇒ must NOT certify"
    );
}

/// TRUE — a `String`-atom universe `S ⊆ {"x","y"}` (the `Str` kind, distinct from model values). `Init
/// S={}`, `Next` adds string atoms; the invariant `S \subseteq {"x","y"}` is a tautology. Certifies with
/// `SetMask{dom=["x","y"], Str}`.
#[test]
fn string_setmask_certifies() {
    let spec = "---- MODULE SmStr ----\n\
                VARIABLE S\n\
                Init == S = {}\n\
                Next == \\E d \\in {\"x\", \"y\"} : S' = (S \\union {d})\n\
                Safety == S \\subseteq {\"x\", \"y\"}\n\
                ====\n";
    let config = Config::parse("INIT Init\nNEXT Next\nINVARIANT Safety\n").expect("config parses");
    let cert = certify_explicit_state_spec(spec, &config).expect("String-atom SetMask certifies");
    assert_eq!(cert.sorts, vec![setmask(&["x", "y"], EnumKind::Str)]);
    assert_eq!(cert.reachable.len(), 4, "R is all 4 subsets of {{x,y}}");
    assert!(verify_explicit_state_cert(&cert), "kernel re-check passes");
}

/// TAMPER — (a) a mutated bitmask cell fails the kernel re-check; (b) a mutated `dom` fails the Leg-E spec
/// re-derivation binding (verify re-enumerates and requires `re.sorts == fp.sorts`).
#[test]
fn tampered_setmask_cert_rejects() {
    let spec = "---- MODULE SmTamper ----\n\
                CONSTANTS a, b, c\n\
                Vals == {a, b, c}\n\
                VARIABLE S\n\
                Init == S = {}\n\
                Next == \\E d \\in Vals : S' = (S \\union {d})\n\
                Safety == S \\subseteq Vals\n\
                ====\n";
    let config = cfg_abc("Safety");
    let cert = certify_explicit_state_spec(spec, &config).expect("SetMask cert mints");

    // (a) kernel-level tamper: push a reachable mask out of the universe (bit 3 exceeds |dom|=3).
    let mut kernel_tampered = cert.clone();
    kernel_tampered.reachable[0][0] = 1u64 << 5;
    assert!(
        !verify_explicit_state_cert(&kernel_tampered),
        "a tampered bitmask cell must fail the kernel re-check"
    );

    // (b) dom tamper: rename an atom in the stored sort. The kernel legs are index-blind Nat equalities, so
    // the binding gate is Leg-E: re-derivation re-collects the real atoms and the sorts no longer match.
    let mut dom_tampered = cert.clone();
    if let ColSort::SetMask { dom, .. } = &mut dom_tampered.sorts[0] {
        dom[0] = "zzz".to_string();
    } else {
        panic!("column 0 must be a SetMask sort");
    }
    let sc = tla_check::cert::build_explicit_fixpoint_certificate(spec, &config, dom_tampered);
    let report = tla_check::cert::verify_safety_certificate(&sc);
    assert!(
        !report.accepted,
        "a cert with a mutated SetMask atom must be rejected: {}",
        report.detail
    );

    // Control: the genuine cert is accepted through the same full Leg-E path.
    let sc_ok = tla_check::cert::build_explicit_fixpoint_certificate(spec, &config, cert);
    let ok_report = tla_check::cert::verify_safety_certificate(&sc_ok);
    assert!(
        ok_report.accepted,
        "the genuine SetMask cert must be accepted: {}",
        ok_report.detail
    );
}
