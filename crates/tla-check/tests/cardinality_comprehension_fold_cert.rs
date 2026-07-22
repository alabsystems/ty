// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! CARDINALITY-OF-SET-COMPREHENSION counting fold for the explicit-state certificate lane — the R4
//! capability that lets SingleLaneBridge's bridge-capacity invariant
//! `Cardinality(CarsInBridge) < Cardinality(Bridge) + 1`, with
//! `CarsInBridge == {c \in Cars : Location[c] \in Bridge}`, certify. The fold encodes
//! `Cardinality({d \in D : P(d)}) = Σ_{d∈D} boolToNat(P(d))` over a COMPLETE fixed finite domain `D`
//! with each per-element `P(d)` truth-EXACTLY recognized (so `boolToNat(P(d))` is exactly `0`/`1`), and
//! folds `Cardinality(<constant finite set>)` to its literal size.
//!
//! SOUNDNESS is pinned by a TRUE/VIOLATED pair over the SAME encoded state:
//!   * TRUE  — `Cardinality({d∈D:loc[d]∈Bridge}) <= |D|` holds on every reachable state ⇒ certifies +
//!     kernel-re-checks + round-trips.
//!   * VIOLATED (over-count twin) — the SAME comprehension with a TIGHTER bound `<= 1` that a reachable
//!     state (two/three keys in the bridge) EXCEEDS ⇒ must DECLINE. Had the fold UNDER-counted (capped the
//!     sum), the kernel would see `1 <= 1` and FALSELY certify the violated invariant; because the sum is
//!     the exact count `2`/`3`, the kernel sees `3 <= 1 = false` and declines (certify ≡ check).
//!
//! FAIL-CLOSED: a comprehension over a GENERAL (non-fixed) set — a map-comprehension image or a
//! state-dependent interval — DECLINES (the domain does not materialize to a complete fixed element list,
//! so no over-approximation of the count is ever emitted). Every certificate is kernel-re-checked
//! (`verify_explicit_state_cert`) and round-trips.

#![cfg(feature = "clean-cic")]

use tla_check::explicit_fixpoint_cert::{
    certify_explicit_state_spec, verify_explicit_state_cert, ExplicitFixpointCert,
};
use tla_check::{Config, ConstantValue};

/// A config with INIT/NEXT/INVARIANT and a model-value domain `D = {m1, m2, m3}` (three atom keys) — the
/// `Cars` analogue. The counting fold materializes `D` to the sorted-deduped `[m1, m2, m3]`.
fn cfg_inv_d3(inv: &str) -> Config {
    let mut c =
        Config::parse(&format!("INIT Init\nNEXT Next\nINVARIANT {inv}\n")).expect("config parses");
    c.constants.insert(
        "D".to_string(),
        ConstantValue::ModelValueSet(vec!["m1".to_string(), "m2".to_string(), "m3".to_string()]),
    );
    c
}

/// A 0/1 function `loc : [D -> {0,1}]` whose `Next` TOGGLES one key each step, so every one of the
/// `2^3 = 8` assignments is reachable and the count `|{d∈D : loc[d] = 1}|` ranges over the full `0..3`.
/// `Bridge == {1}`, so `loc[d] \in Bridge` means key `d` is "in the bridge". The comprehension-count
/// invariant `Cardinality(InBridge) <= 3` is TRUE on every state (at most `|D| = 3` keys) — certifies,
/// kernel-re-checks, round-trips. This is the counting fold's headline shape (SingleLaneBridge's
/// `CarsInBridge`), in isolation.
#[test]
fn comprehension_count_le_domain_size_certifies() {
    let spec = "---- MODULE CardCompTrue ----\n\
                EXTENDS Naturals, FiniteSets\n\
                CONSTANT D\n\
                VARIABLE loc\n\
                Bridge == {1}\n\
                InBridge == { d \\in D : loc[d] \\in Bridge }\n\
                Init == loc = [d \\in D |-> 0]\n\
                Next == \\E d \\in D : loc' = [loc EXCEPT ![d] = 1 - loc[d]]\n\
                Safety == Cardinality(InBridge) <= 3\n\
                ====\n";
    let cert = certify_explicit_state_spec(spec, &cfg_inv_d3("Safety")).expect(
        "Cardinality of a comprehension over a fixed atom domain certifies (counting fold)",
    );
    assert!(
        cert.safety_general.is_some(),
        "kernel-checked ⋀_{{s∈R}} Safety(s) leg present"
    );
    assert!(verify_explicit_state_cert(&cert), "kernel re-check passes");
    let bytes = serde_json::to_vec(&cert).expect("serialize");
    let back: ExplicitFixpointCert = serde_json::from_slice(&bytes).expect("deserialize");
    assert_eq!(cert, back, "CountFold cert round-trips byte-identically");
    assert!(
        verify_explicit_state_cert(&back),
        "re-loaded cert still verifies"
    );
}

/// The VIOLATED over-count twin — the SAME comprehension with the TIGHTER bound `Cardinality(InBridge) <=
/// 1`. Toggling reaches states with two or three keys set (`loc[d] = 1`), so the count is `2`/`3 > 1` and
/// the invariant is FALSE. certify MUST DECLINE (the general safety leg cannot reduce `⋀_{s∈R} Safety(s)`
/// to `Bool.true`). This is the DECISIVE SOUNDNESS GATE: an off-by-one / under-counting fold would make the
/// kernel see `1 <= 1` (true) and FALSELY certify the violated invariant. Because the fold emits the EXACT
/// `Σ boolToNat(P(d))`, the over-count state reduces `3 <= 1` to `false` ⇒ correct decline (certify ≡ the
/// `ty check` verdict, which reports this very violation).
#[test]
fn comprehension_overcount_violation_declines() {
    let spec = "---- MODULE CardCompBad ----\n\
                EXTENDS Naturals, FiniteSets\n\
                CONSTANT D\n\
                VARIABLE loc\n\
                Bridge == {1}\n\
                InBridge == { d \\in D : loc[d] \\in Bridge }\n\
                Init == loc = [d \\in D |-> 0]\n\
                Next == \\E d \\in D : loc' = [loc EXCEPT ![d] = 1 - loc[d]]\n\
                Safety == Cardinality(InBridge) <= 1\n\
                ====\n";
    assert!(
        certify_explicit_state_spec(spec, &cfg_inv_d3("Safety")).is_none(),
        "an over-count comprehension invariant DECLINES — the CountFold must emit the EXACT count, \
         never an under-count that would falsely certify"
    );
}

/// The SingleLaneBridge RHS shape `Cardinality(InBridge) < Cardinality(FullBridge) + 1` — the right side
/// is `Cardinality(<constant finite set>)`, which folds to the LITERAL distinct-element count. `FullBridge
/// == {1,2,3}` ⇒ `Cardinality(FullBridge) = 3` ⇒ the bound is `< 4` (i.e. `<= 3`), TRUE on every state.
/// Exercises BOTH the comprehension counting fold (LHS) and the constant-set cardinality fold (RHS) in one
/// comparison — the exact structure of SingleLaneBridge's `Cardinality(CarsInBridge) < Cardinality(Bridge)
/// + 1`. Certifies + kernel-re-checks.
#[test]
fn comprehension_count_lt_constant_set_cardinality_plus_one_certifies() {
    let spec = "---- MODULE CardCompConstRhs ----\n\
                EXTENDS Naturals, FiniteSets\n\
                CONSTANT D\n\
                VARIABLE loc\n\
                Bridge == {1}\n\
                FullBridge == {1,2,3}\n\
                InBridge == { d \\in D : loc[d] \\in Bridge }\n\
                Init == loc = [d \\in D |-> 0]\n\
                Next == \\E d \\in D : loc' = [loc EXCEPT ![d] = 1 - loc[d]]\n\
                Safety == Cardinality(InBridge) < Cardinality(FullBridge) + 1\n\
                ====\n";
    let cert = certify_explicit_state_spec(spec, &cfg_inv_d3("Safety")).expect(
        "comprehension count < |constant set| + 1 certifies (LHS CountFold + RHS const-card)",
    );
    assert!(
        cert.safety_general.is_some(),
        "kernel-checked general safety leg present"
    );
    assert!(verify_explicit_state_cert(&cert), "kernel re-check passes");
}

/// FAIL-CLOSED: a comprehension over a GENERAL set — a map-comprehension image `{ loc[k] : k \in D }`,
/// NOT a fixed atom/Int domain nor a bitmask-encodable column. The domain does not materialize to a
/// complete fixed element list, so the counting fold returns `None` and no certificate is emitted (no
/// over-approximation of the count). Mirrors the R4 `map_comprehension_domain_declines` guard.
#[test]
fn comprehension_over_general_domain_declines() {
    let spec = "---- MODULE CardCompGenDom ----\n\
                EXTENDS Naturals, FiniteSets\n\
                CONSTANT D\n\
                VARIABLE loc\n\
                Bridge == {1}\n\
                Init == loc = [d \\in D |-> 0]\n\
                Next == \\E d \\in D : loc' = [loc EXCEPT ![d] = 1 - loc[d]]\n\
                Safety == Cardinality({ v \\in { loc[k] : k \\in D } : v \\in Bridge }) <= 3\n\
                ====\n";
    assert!(
        certify_explicit_state_spec(spec, &cfg_inv_d3("Safety")).is_none(),
        "a count over a map-comprehension (general) domain DECLINES (fail-closed)"
    );
}

/// FAIL-CLOSED: a comprehension over a STATE-DEPENDENT interval `0 .. loc[m1]` (the upper bound is a
/// function value, not a ground literal) is NOT a fixed finite domain known at cert time, so the counting
/// fold declines — a count can only be exact over the COMPLETE fixed domain the checker enumerates.
#[test]
fn comprehension_over_state_dependent_interval_declines() {
    let spec = "---- MODULE CardCompStateDom ----\n\
                EXTENDS Naturals, FiniteSets\n\
                CONSTANT D\n\
                VARIABLE loc\n\
                Init == loc = [d \\in D |-> 0]\n\
                Next == \\E d \\in D : loc' = [loc EXCEPT ![d] = 1 - loc[d]]\n\
                Safety == Cardinality({ j \\in 0 .. loc[m1] : j > 0 }) <= 3\n\
                ====\n";
    assert!(
        certify_explicit_state_spec(spec, &cfg_inv_d3("Safety")).is_none(),
        "a count over a state-dependent interval domain DECLINES (fail-closed)"
    );
}
