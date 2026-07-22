// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! EXPLICIT-Int-SET FUNCTION CODOMAIN — the controlled SOUNDNESS PROOF for the `f ∈ [D -> {v₁,…,vₙ}]`
//! function-set-membership recognizer extension (the TeachingConcurrency/Simple `x ∈ [0..N-1 -> {0,1}]`
//! class). A [`ColSort::Func`] column with Int cells and a codomain that is an EXPLICIT finite Int set
//! `{0,1}` now discharges into the per-position disjunction `⋀_i ⋁_j (f[i] = vⱼ)` — the exact analogue
//! of the value-membership `x ∈ {…}` fold, applied to the digit that faithfully carries `f[i]`. The
//! quantifier DOMAIN is the DEFINED nullary operator `Foo == 0..2`, inlined to its interval body before
//! the interval recognizer fires — so this file jointly exercises the defined-operator-set domain and the
//! explicit-Int-set codomain the change targets.
//!
//! THE SOUNDNESS REQUIREMENT under test: the `{0,1}` codomain fold must not HIDE an out-of-set value, and
//! the defined-op-set quantifier fold must not HIDE a bound violation. Four cases, the fail-closed proof:
//!   (a) SOUND POSITIVE (codomain): `g ∈ [Foo -> {0,1}]` over a reachable set where every cell is 0/1
//!       ⇒ KERNEL-CERTIFIED, and the full Leg-E re-check (`verify_safety_certificate`) ACCEPTS.
//!   (b) SOUND POSITIVE (domain): `\A i ∈ Foo : g[i] <= 1` — the defined-op-set (`Foo == 0..2`) quantifier
//!       fold over the same reachable set ⇒ KERNEL-CERTIFIED + Leg-E ACCEPTS.
//!   (c) CODOMAIN VIOLATION load-bearing: a twin whose `Next` drives a cell to `2` ⇒ `g ∈ [Foo -> {0,1}]`
//!       is VIOLATED on a reachable state ⇒ NOT CERTIFIED (the `{0,1}` fold catches the value `2` — the
//!       out-of-set value is not hidden). `ty check` independently reports the violation (CLI cross-check).
//!   (d) BOUND VIOLATION twin: `\A i ∈ Foo : g[i] <= 0` on the positive spec (cells reach 1) ⇒ NOT
//!       CERTIFIED (the defined-op-set fold does not hide the `g[i] = 1 > 0` violation).
//!
//! These call `certify_explicit_state_spec` DIRECTLY (the explicit-state kernel-fixpoint lane the change
//! lives in), so no other prover lane can mask the behavior.

#![cfg(feature = "clean-cic")]

use tla_check::explicit_fixpoint_cert::{
    certify_explicit_state_spec, verify_explicit_state_cert, CellSort, ColSort,
};
use tla_check::Config;

fn cfg_inv(inv: &str) -> Config {
    Config::parse(&format!("INIT Init\nNEXT Next\nINVARIANT {inv}\n")).expect("config parses")
}

// The positive spec: `g` is a function `Foo -> {0,1}` (`Foo == 0..2` a DEFINED nullary operator). From
// the all-0 function, each step sets one cell to 1, so the reachable set is exactly {0,1}^3 = 8 states —
// every cell always 0 or 1 (`TypeOK` holds) and ≤ 1 (`Bound` holds).
const POSITIVE_SPEC: &str = "---- MODULE FooIntSet ----\n\
    EXTENDS Integers\n\
    VARIABLE g\n\
    Foo == 0..2\n\
    Init == g = [i \\in Foo |-> 0]\n\
    Next == \\E i \\in Foo : g' = [g EXCEPT ![i] = 1]\n\
    TypeOK == g \\in [Foo -> {0,1}]\n\
    Bound == \\A i \\in Foo : g[i] <= 1\n\
    BoundBad == \\A i \\in Foo : g[i] <= 0\n\
    ====\n";

// ─────────────────────────────────────────────────────────────────────────────────────────────────────
// (a) SOUND POSITIVE (codomain) ⇒ KERNEL-CERTIFIED + Leg-E ACCEPTS. `g ∈ [Foo -> {0,1}]` — the explicit
// finite Int-set codomain over an Int-cell `Func` column, folded to `⋀_i (g[i]=0 ∨ g[i]=1)`.
// ─────────────────────────────────────────────────────────────────────────────────────────────────────
#[test]
fn func_int_set_codomain_certifies() {
    let cert = certify_explicit_state_spec(POSITIVE_SPEC, &cfg_inv("TypeOK"))
        .expect("g \\in [Foo -> {0,1}] certifies (explicit Int-set function codomain)");
    // The `g` column is an Int-domain Func of arity 3 (Foo = 0..2) with all-Int cells.
    match &cert.sorts[..] {
        [ColSort::Func {
            arity, cells, dom, ..
        }] => {
            assert_eq!(*arity, 3, "Foo = 0..2 ⇒ a 3-position function");
            assert!(
                dom.is_empty(),
                "an Int-prefix (0..arity-1) domain carries no stored key set"
            );
            assert!(
                cells.iter().all(|c| matches!(c, CellSort::Int)),
                "the codomain {{0,1}} is Int — every cell is an Int leaf, never a Bool code"
            );
        }
        other => panic!("expected a single Func column, got {other:?}"),
    }
    assert_eq!(
        cert.reachable.len(),
        8,
        "the reachable set is exactly {{0,1}}^3 = 8 states"
    );
    assert!(
        cert.safety_pred.is_some(),
        "the Int-set codomain membership rides the general safety leg"
    );
    assert!(
        cert.safety_general.is_some(),
        "kernel-checked ⋀_{{s∈R}} TypeOK(s) leg present"
    );
    assert!(
        verify_explicit_state_cert(&cert),
        "the Int-set-codomain cert kernel-re-checks"
    );

    // Full Leg-E: build the self-contained certificate and re-check it end to end.
    let config = cfg_inv("TypeOK");
    let sc = tla_check::cert::build_explicit_fixpoint_certificate(POSITIVE_SPEC, &config, cert);
    let report = tla_check::cert::verify_safety_certificate(&sc);
    assert!(
        report.accepted,
        "ty cert-check (Leg-E) ACCEPTS the Int-set-codomain cert: {}",
        report.detail
    );
}

// ─────────────────────────────────────────────────────────────────────────────────────────────────────
// (b) SOUND POSITIVE (domain) ⇒ KERNEL-CERTIFIED + Leg-E ACCEPTS. `\A i ∈ Foo : g[i] <= 1` — the DEFINED
// nullary-operator set `Foo == 0..2` is inlined to its interval body, then the interval-domain quantifier
// fold expands the body per literal index.
// ─────────────────────────────────────────────────────────────────────────────────────────────────────
#[test]
fn defined_op_set_quantifier_bound_certifies() {
    let cert = certify_explicit_state_spec(POSITIVE_SPEC, &cfg_inv("Bound"))
        .expect("\\A i \\in Foo : g[i] <= 1 certifies (defined-op-set interval quantifier fold)");
    assert!(
        cert.safety_pred.is_some(),
        "the defined-op-set quantifier fold rides the general safety leg"
    );
    assert!(
        verify_explicit_state_cert(&cert),
        "the quantifier-fold cert kernel-re-checks"
    );
    let config = cfg_inv("Bound");
    let sc = tla_check::cert::build_explicit_fixpoint_certificate(POSITIVE_SPEC, &config, cert);
    let report = tla_check::cert::verify_safety_certificate(&sc);
    assert!(
        report.accepted,
        "Leg-E ACCEPTS the quantifier-fold cert: {}",
        report.detail
    );
}

// ─────────────────────────────────────────────────────────────────────────────────────────────────────
// (c) CODOMAIN VIOLATION load-bearing ⇒ NOT CERTIFIED. A twin whose `Next` drives a cell to `2`, so
// `g ∈ [Foo -> {0,1}]` is FALSE on a reachable state. The `{0,1}` fold `g[i]=0 ∨ g[i]=1` is FALSE at `2`,
// so the kernel cannot prove the `R ⊆ Safety` leg — the out-of-set value is NOT hidden. (`ty check`
// independently reports `Invariant TypeOK is violated`.)
// ─────────────────────────────────────────────────────────────────────────────────────────────────────
#[test]
fn int_set_codomain_violation_not_certified() {
    let spec = "---- MODULE FooBadCodom ----\n\
        EXTENDS Integers\n\
        VARIABLE g\n\
        Foo == 0..2\n\
        Init == g = [i \\in Foo |-> 0]\n\
        Next == \\E i \\in Foo : g' = [g EXCEPT ![i] = 2]\n\
        TypeOK == g \\in [Foo -> {0,1}]\n\
        ====\n";
    assert!(
        certify_explicit_state_spec(spec, &cfg_inv("TypeOK")).is_none(),
        "a reachable cell = 2 violates `g \\in [Foo -> {{0,1}}]` ⇒ NOT CERTIFIED (the {{0,1}} \
         codomain fold catches the out-of-set value — it is not hidden)"
    );
}

// ─────────────────────────────────────────────────────────────────────────────────────────────────────
// (d) BOUND VIOLATION twin ⇒ NOT CERTIFIED. `\A i ∈ Foo : g[i] <= 0` on the positive spec (cells reach 1)
// is FALSE on any state with a `1` cell. The defined-op-set quantifier fold does not hide the violation —
// the kernel cannot prove the false `R ⊆ Safety` leg. (`ty check` reports `Invariant BoundBad is violated`.)
// ─────────────────────────────────────────────────────────────────────────────────────────────────────
#[test]
fn defined_op_set_quantifier_bound_violation_not_certified() {
    assert!(
        certify_explicit_state_spec(POSITIVE_SPEC, &cfg_inv("BoundBad")).is_none(),
        "a reachable cell = 1 violates `\\A i \\in Foo : g[i] <= 0` ⇒ NOT CERTIFIED (the defined-op-set \
         quantifier fold does not hide the bound violation)"
    );
}
