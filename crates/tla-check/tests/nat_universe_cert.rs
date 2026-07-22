// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! TYPE-UNIVERSE membership `x ∈ Nat` / `x ∈ Int` in the explicit-state certificate lane — the TypeOK
//! "the state variable is a natural / an integer" conjunct (the PCR / `glowingRaccoon/clean` corpus class:
//! `TypeOK == … ∧ primer ∈ Nat ∧ dna ∈ Nat ∧ …`). An `Int` cell embeds `Int.ofNat v`, so its value is `≥0`
//! on every state BY CONSTRUCTION (the column carries the `x≥0` Safety conjunct); the recognizer therefore
//! reads the membership EXACTLY as the column's own numeric invariant:
//!   * `x ∈ Nat  ⟺  0 ≤ x`  (`PredIR::Leq(0, x)` — always true here, faithful to the Nat bound);
//!   * `x ∈ Int  ⟺  TRUE`   (`PredIR::BoolLit(true)` — a tautology on `R` for an integer-valued `x`).
//! Gated to a BARE (possibly primed) Int column: an arithmetic `x` (`a − b` truncates in `Nat.sub`) or a
//! Bool/Enum/Set/compound column fails closed (see `cleancic::int_universe_in_form`).
//!
//! SOUNDNESS proof (the certify≡check backstop, in-process):
//!   * TRUE — a two-counter conservation spec with `TypeOK == x ∈ Nat ∧ y ∈ Int` and `consv == x+y=3`
//!     CERTIFIES with two `ColSort::Int` columns, kernel-re-checks Init⊆R / closure / R⊆Safety, serde
//!     round-trips, and full Leg-E accepts; R is exactly the 4 conservation states — the SAME set (and
//!     count 4) `ty check` explores (`No error`);
//!   * VIOLATED twin — the SAME transition system but the invariant `x ∈ Nat ∧ x ≤ 2` holds only until `x`
//!     reaches 3: the `x ≤ 2` conjunct falsifies at the reachable state `x=3`, the kernel
//!     `⋀_{s∈R} Safety(s) ⇒ Bool.true` leg rejects it ⇒ NOT CERTIFIED. The `x ∈ Nat` reading does NOT hide
//!     the violation `ty check` reports (`Invariant … violated`);
//!   * FAIL-CLOSED — `b ∈ Nat` over a BOOL column is FALSE in TLA (`TRUE ∉ Nat`); the recognizer's Int-column
//!     gate DECLINES it (never emits the always-true `0 ≤ b` pun), so the spec is NOT CERTIFIED — matching
//!     `ty check`, which reports the genuine `Invariant … violated`.
//!
//! The corresponding `ty check` cross-check (TRUE `No error` / VIOLATED + FAIL-CLOSED `Invariant … violated`,
//! all at the SAME state count) is exercised end-to-end at the CLI; here the certify side is the arbiter
//! (certify must never certify what check flags).

#![cfg(feature = "clean-cic")]

use tla_check::explicit_fixpoint_cert::{
    certify_explicit_state_spec, verify_explicit_state_cert, ColSort, ExplicitFixpointCert,
};
use tla_check::Config;

fn cfg(inv: &str) -> Config {
    Config::parse(&format!("INIT Init\nNEXT Next\nINVARIANT {inv}\n")).expect("config parses")
}

/// A two-counter conservation system: `x` steps up while `y` steps down (and vice-versa), preserving
/// `x + y = 3`. Both columns stay nonneg on `R = {(0,3),(1,2),(2,1),(3,0)}`. `Int` provides `Nat` AND `Int`.
const SPEC: &str = "---- MODULE NatUniv ----\n\
     EXTENDS Integers\n\
     VARIABLES x, y\n\
     vars == <<x, y>>\n\
     Init == x = 0 /\\ y = 3\n\
     Next == \\/ (x < 3 /\\ x' = x + 1 /\\ y' = y - 1)\n\
             \\/ (y < 3 /\\ y' = y + 1 /\\ x' = x - 1)\n\
     TypeOK == x \\in Nat /\\ y \\in Int\n\
     consv == x + y = 3\n\
     Safety == TypeOK /\\ consv\n\
     BadSafety == x \\in Nat /\\ x <= 2\n\
     ====\n";

/// TRUE — `TypeOK ∧ consv` (`x ∈ Nat`, `y ∈ Int`, conservation). Certifies with two `Int` columns,
/// kernel-re-verifies, serde round-trips, and full Leg-E accepts. R is exactly the 4 conservation
/// states — the SAME set (and count) `ty check` explores.
#[test]
fn nat_int_universe_certifies() {
    let config = cfg("Safety");
    let cert =
        certify_explicit_state_spec(SPEC, &config).expect("x∈Nat ∧ y∈Int type-universe certifies");
    assert_eq!(
        cert.sorts,
        vec![ColSort::Int, ColSort::Int],
        "both counters are nonneg Int columns"
    );
    assert_eq!(
        cert.reachable,
        vec![vec![0, 3], vec![1, 2], vec![2, 1], vec![3, 0]],
        "R is exactly the 4 conservation states"
    );
    assert!(
        cert.safety_pred.is_some(),
        "the x∈Nat / y∈Int / consv invariant rides the general safety leg"
    );
    assert!(
        cert.safety_general.is_some(),
        "kernel-checked ⋀_{{s∈R}} Safety(s) leg present"
    );
    assert!(verify_explicit_state_cert(&cert), "kernel re-check passes");

    // Serde round-trip: no new sort/IR variant, so the cert round-trips verbatim and still verifies.
    let bytes = serde_json::to_vec(&cert).expect("serialize");
    let back: ExplicitFixpointCert = serde_json::from_slice(&bytes).expect("deserialize");
    assert_eq!(cert, back, "type-universe cert round-trips");
    assert!(
        verify_explicit_state_cert(&back),
        "the re-loaded cert verifies"
    );

    // Full Leg-E: build a SafetyCertificate and re-verify end-to-end (re-enumerates + re-derives the IR).
    let sc = tla_check::cert::build_explicit_fixpoint_certificate(SPEC, &config, cert);
    let report = tla_check::cert::verify_safety_certificate(&sc);
    assert!(
        report.accepted,
        "full Leg-E re-check accepts: {}",
        report.detail
    );
}

/// VIOLATED twin — `x ∈ Nat ∧ x ≤ 2`. The `x ∈ Nat` reading is exact (always true), but the `x ≤ 2`
/// conjunct FALSIFIES at the reachable state `x=3`; the kernel `⋀_{s∈R} Safety(s)` leg reduces to
/// `Bool.false` ⇒ NOT CERTIFIED. The `∈ Nat` reading never hides the violation `ty check` reports.
#[test]
fn violated_bound_under_nat_is_not_certified() {
    let config = cfg("BadSafety");
    assert!(
        certify_explicit_state_spec(SPEC, &config).is_none(),
        "a state with x=3 violates x≤2 — the x∈Nat conjunct must NOT mask it; certify DECLINES"
    );
}

/// FAIL-CLOSED — `b ∈ Nat` over a BOOL column. In TLA `TRUE ∉ Nat`, so this invariant is FALSE; the
/// recognizer's Int-column gate DECLINES (never emits the always-true `0 ≤ b` Bool pun), so the spec is
/// NOT CERTIFIED — faithful to `ty check`, which reports the genuine violation.
#[test]
fn bool_column_in_nat_fails_closed() {
    const BOOL_SPEC: &str = "---- MODULE BoolNat ----\n\
         EXTENDS Naturals\n\
         VARIABLE b\n\
         Init == b = TRUE\n\
         Next == b' = ~b\n\
         BoolNat == b \\in Nat\n\
         ====\n";
    let config = cfg("BoolNat");
    assert!(
        certify_explicit_state_spec(BOOL_SPEC, &config).is_none(),
        "b∈Nat over a Bool column is FALSE in TLA — the Int-column gate fails closed (no 0≤b pun)"
    );
}
