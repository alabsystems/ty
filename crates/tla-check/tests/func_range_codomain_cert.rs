// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! INT-RANGE FUNCTION CODOMAIN — the controlled SOUNDNESS PROOF for the `f ∈ [D -> lo..hi]`
//! function-set-membership recognizer extension (the EWD426/TokenRing `c ∈ [Node -> 0..M-1]` class). A
//! [`ColSort::Func`] column with Int cells and a codomain that is a bounded Int interval `lo..hi` whose
//! endpoints fold to GROUND nonneg-Int constants discharges into the per-position bound
//! `⋀_i (lo ≤ f[i] ∧ f[i] ≤ hi)` — the exact analogue of the scalar interval-membership `x ∈ lo..hi`
//! fold, applied to the digit that faithfully carries `f[i]`.
//!
//! The load-bearing change: the endpoints resolve through `const_eval_nonneg_int` (the same evaluator
//! the range-DOMAIN path uses), so a CONSTANT bound `M-1` — `Sub(6,1)` after CONSTANT inlining — folds
//! to the literal `5`. The prior form accepted only a BARE `Int` literal (`nonneg_int_lit`), so
//! `[Node -> 0..M-1]` declined; a plain literal endpoint (`[D -> a..b]`, the CoffeeCan class) yields the
//! IDENTICAL literal, so its certificate stays byte-identical.
//!
//! THE SOUNDNESS REQUIREMENT under test: the interval bound must not HIDE an out-of-range value, and a
//! NON-ground bound (outside the exact fragment) must fail CLOSED, never be silently mis-recognized.
//! Four cases, each verdict cross-checked against `ty check` (`check_module`):
//!   (a) SOUND POSITIVE: `f ∈ [{a,b} -> 0..(Cap-1)]`, `Cap == 3` ⇒ KERNEL-CERTIFIED, Leg-E ACCEPTS, and
//!       `ty check` finds NO error with the SAME reachable-state count. (The `Cap-1` endpoint is the
//!       `Sub` that the old `nonneg_int_lit` form rejected.)
//!   (b) VIOLATION twin: a cell driven to `Cap = 3` (outside `0..2`) ⇒ NOT CERTIFIED (the bound catches
//!       the value `3`); `ty check` independently reports the violation.
//!   (c) FAIL-CLOSED (state-dependent bound): `f ∈ [{a,b} -> 0..(j+k)]` with `j,k` VARIABLES — the
//!       endpoint is not GROUND, so it DECLINES even though the invariant is genuinely TRUE (`ty check`
//!       finds no error). Conservative fail-close, never a false certificate.
//!   (d) FAIL-CLOSED (variable `Sub`): `f ∈ [{a,b} -> 0..(k-1)]` with `k` a VARIABLE — likewise
//!       out-of-fragment (a non-ground `Sub`) ⇒ DECLINES on a TRUE invariant.
//!
//! These call `certify_explicit_state_spec` DIRECTLY (the explicit-state kernel-fixpoint lane the change
//! lives in), so no other prover lane can mask the behaviour.

#![cfg(feature = "clean-cic")]

mod common;

use tla_check::explicit_fixpoint_cert::{
    certify_explicit_state_spec, verify_explicit_state_cert, CellSort, ColSort,
};
use tla_check::{check_module, CheckResult, Config};

fn cfg_inv(inv: &str) -> Config {
    Config::parse(&format!("INIT Init\nNEXT Next\nINVARIANT {inv}\n")).expect("config parses")
}

/// Run the explicit-state model checker in-process (the `ty check` cross-check).
fn ty_check(spec: &str, inv: &str) -> CheckResult {
    let module = common::parse_module(spec);
    check_module(&module, &cfg_inv(inv))
}

fn assert_check_ok(spec: &str, inv: &str, want_states: usize) {
    match ty_check(spec, inv) {
        CheckResult::Success(stats) => assert_eq!(
            stats.states_found, want_states,
            "ty check reachable-state count must match the certified R"
        ),
        other => panic!("ty check expected Success ({want_states} states), got {other:?}"),
    }
}

fn assert_check_violated(spec: &str, inv: &str) {
    match ty_check(spec, inv) {
        CheckResult::InvariantViolation { invariant, .. } => assert_eq!(invariant, inv),
        other => panic!("ty check expected InvariantViolation of {inv}, got {other:?}"),
    }
}

// ─────────────────────────────────────────────────────────────────────────────────────────────────────
// (a) SOUND POSITIVE — CONST-FOLDED endpoint (the TokenRing `c ∈ [Node -> 0..M-1]` class). The codomain
// `0..(Cap-1)` has `hi = Cap-1 = Sub(3,1)`; `const_eval_nonneg_int` folds it to the literal `2`, so the
// per-cell bound is `0 ≤ f[k] ∧ f[k] ≤ 2`. Cells climb `0 -> 1 -> 2`, R = {0,1,2}^2 = 9 states.
// ─────────────────────────────────────────────────────────────────────────────────────────────────────
const CONST_FOLD_SPEC: &str = "---- MODULE FuncRangeConstFold ----\n\
    EXTENDS Integers\n\
    VARIABLE f\n\
    Cap == 3\n\
    Dom == {\"a\", \"b\"}\n\
    Init == f = [k \\in Dom |-> 0]\n\
    Next == \\E k \\in Dom : f' = [f EXCEPT ![k] = IF f[k] < Cap - 1 THEN f[k] + 1 ELSE f[k]]\n\
    TypeOK == f \\in [Dom -> 0 .. (Cap - 1)]\n\
    ====\n";

#[test]
fn func_range_const_fold_codomain_certifies() {
    let cert = certify_explicit_state_spec(CONST_FOLD_SPEC, &cfg_inv("TypeOK"))
        .expect("f \\in [Dom -> 0..(Cap-1)] certifies (const-folded interval codomain)");
    match &cert.sorts[..] {
        [ColSort::Func {
            arity, cells, dom, ..
        }] => {
            assert_eq!(*arity, 2, "Dom = {{a,b}} ⇒ a 2-position function");
            assert_eq!(dom.len(), 2, "the String-atom domain stores its two keys");
            assert!(
                cells.iter().all(|c| matches!(c, CellSort::Int)),
                "the codomain 0..2 is Int — every cell is an Int leaf"
            );
        }
        other => panic!("expected a single Func column, got {other:?}"),
    }
    assert_eq!(cert.reachable.len(), 9, "R = {{0,1,2}}^2 = 9 states");
    assert!(
        cert.safety_pred.is_some(),
        "interval-codomain membership rides the general safety leg"
    );
    assert!(
        cert.safety_general.is_some(),
        "kernel-checked ⋀_{{s∈R}} TypeOK(s) leg present"
    );
    assert!(
        verify_explicit_state_cert(&cert),
        "the interval-codomain cert kernel-re-checks"
    );

    // Full Leg-E: build the self-contained certificate and re-check it end to end.
    let config = cfg_inv("TypeOK");
    let sc = tla_check::cert::build_explicit_fixpoint_certificate(CONST_FOLD_SPEC, &config, cert);
    let report = tla_check::cert::verify_safety_certificate(&sc);
    assert!(
        report.accepted,
        "Leg-E (ty cert-check) ACCEPTS: {}",
        report.detail
    );

    // certify ≡ check: the model checker agrees on No error and the SAME state count.
    assert_check_ok(CONST_FOLD_SPEC, "TypeOK", 9);
}

// ─────────────────────────────────────────────────────────────────────────────────────────────────────
// (b) VIOLATION twin — `Next` drives a cell to `Cap = 3`, outside `0..(Cap-1) = 0..2`. The bound
// `f[k] ≤ 2` is FALSE at `3`, so `R ⊆ Safety` cannot be proven ⇒ NOT CERTIFIED; `ty check` reports it.
// ─────────────────────────────────────────────────────────────────────────────────────────────────────
const CONST_FOLD_BAD_SPEC: &str = "---- MODULE FuncRangeConstFoldBad ----\n\
    EXTENDS Integers\n\
    VARIABLE f\n\
    Cap == 3\n\
    Dom == {\"a\", \"b\"}\n\
    Init == f = [k \\in Dom |-> 0]\n\
    Next == \\E k \\in Dom : f' = [f EXCEPT ![k] = IF f[k] < Cap THEN f[k] + 1 ELSE f[k]]\n\
    TypeOK == f \\in [Dom -> 0 .. (Cap - 1)]\n\
    ====\n";

#[test]
fn func_range_const_fold_violation_not_certified() {
    assert!(
        certify_explicit_state_spec(CONST_FOLD_BAD_SPEC, &cfg_inv("TypeOK")).is_none(),
        "a reachable cell = 3 violates `f \\in [Dom -> 0..(Cap-1)]` ⇒ NOT CERTIFIED \
         (the interval bound catches the out-of-range value)"
    );
    assert_check_violated(CONST_FOLD_BAD_SPEC, "TypeOK");
}

// ─────────────────────────────────────────────────────────────────────────────────────────────────────
// (c) FAIL-CLOSED — STATE-DEPENDENT bound `0..(j+k)` with `j,k` VARIABLES. The endpoint is not a GROUND
// constant (`const_eval_nonneg_int` needs ground operands and `j`,`k` are state columns), so it resolves
// to NONE and the membership DECLINES — even though the invariant is genuinely TRUE (j=k=1, cells ≤ 2 =
// j+k, so `ty check` finds NO error). Conservative fail-close, never a false certificate.
// ─────────────────────────────────────────────────────────────────────────────────────────────────────
const STATE_DEP_SPEC: &str = "---- MODULE FuncRangeStateDep ----\n\
    EXTENDS Integers\n\
    VARIABLE f, j, k\n\
    Dom == {\"a\", \"b\"}\n\
    Init == /\\ f = [x \\in Dom |-> 0]\n\
            /\\ j = 1\n\
            /\\ k = 1\n\
    Next == \\E x \\in Dom : /\\ f' = [f EXCEPT ![x] = IF f[x] < j + k THEN f[x] + 1 ELSE f[x]]\n\
                            /\\ UNCHANGED <<j, k>>\n\
    TypeOK == f \\in [Dom -> 0 .. (j + k)]\n\
    ====\n";

#[test]
fn func_range_state_dependent_endpoint_fails_closed() {
    assert_check_ok(STATE_DEP_SPEC, "TypeOK", 9); // the invariant is TRUE
    assert!(
        certify_explicit_state_spec(STATE_DEP_SPEC, &cfg_inv("TypeOK")).is_none(),
        "`0..(j+k)` with j,k VARIABLES is a non-ground bound, outside the exact codomain fragment ⇒ \
         NOT CERTIFIED (conservative fail-close, even though the invariant holds)"
    );
}

// ─────────────────────────────────────────────────────────────────────────────────────────────────────
// (d) FAIL-CLOSED — variable `Sub` bound `0..(k-1)` with `k` a VARIABLE. Likewise not GROUND ⇒ the
// endpoint resolves to NONE and the membership DECLINES on a TRUE invariant (k=3 fixed, cells ≤ 2 = k-1).
// ─────────────────────────────────────────────────────────────────────────────────────────────────────
const SUB_VAR_SPEC: &str = "---- MODULE FuncRangeSubVar ----\n\
    EXTENDS Integers\n\
    VARIABLE f, k\n\
    Dom == {\"a\", \"b\"}\n\
    Init == /\\ f = [x \\in Dom |-> 0]\n\
            /\\ k = 3\n\
    Next == \\E x \\in Dom : /\\ f' = [f EXCEPT ![x] = IF f[x] < k - 1 THEN f[x] + 1 ELSE f[x]]\n\
                            /\\ UNCHANGED k\n\
    TypeOK == f \\in [Dom -> 0 .. (k - 1)]\n\
    ====\n";

#[test]
fn func_range_sub_of_variable_endpoint_fails_closed() {
    assert_check_ok(SUB_VAR_SPEC, "TypeOK", 9); // the invariant is TRUE
    assert!(
        certify_explicit_state_spec(SUB_VAR_SPEC, &cfg_inv("TypeOK")).is_none(),
        "`0..(k-1)` with k a VARIABLE is a non-ground bound ⇒ NOT CERTIFIED (conservative fail-close)"
    );
}
