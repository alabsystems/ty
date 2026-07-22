// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! STATE-DEPENDENT quantifier DOMAINS for the explicit-state certificate lane — the last wall for
//! EWD840 (`Inv == … \/ P1:: \E j \in 0 .. tpos : color[j] = "black" \/ …`) where the quantifier's
//! upper bound `tpos` is a STATE VARIABLE, so `0..tpos` is NOT a fixed finite set.
//!
//! THE SOUND CONSTRUCTION (see `cleancic::recognize_bounded_quant`): let `M` be the per-column MAX of
//! the interval's upper bound over the ENUMERATED reachable set `R`. Then, EXACTLY per reachable state,
//!     `∃j∈lo..hi:P(j)`  ≡  ⋁_{k=lo..M} ( Leq(k, hi)  ∧  P(k) )
//!     `∀j∈lo..hi:P(j)`  ≡  ⋀_{k=lo..M} ( Leq(k, hi)  ⇒  P(k) )
//! The guard `k ≤ hi(s)` makes every `k > hi(s)` term VACUOUS, so the fold equals the real `lo..hi(s)`
//! disjunction/conjunction at each `s`; SOUND because `M ≥ hi(s)` for every reachable `s`.
//!
//! A WRONG expansion is a FALSE-SAFE certificate, so these tests are non-negotiable:
//!   (a) a HOLDS-everywhere state-dependent `∃` certifies + kernel-re-checks + full-Leg-E-accepts;
//!   (b) a VIOLATED one DECLINES (the kernel rejects the expanded disjunction);
//!   (c) THE GUARD IS REAL — a spec where `P(j)` is TRUE only for some `j > tpos` (never for `j ≤ tpos`)
//!       must DECLINE; a buggy expansion that dropped the `k ≤ hi` guard would (wrongly) certify it;
//!   (d) an upper bound with NO cheap sound static bound (`0..(k % 2)`) FAILS CLOSED even when the
//!       invariant actually holds.

#![cfg(feature = "clean-cic")]

use tla_check::explicit_fixpoint_cert::{certify_explicit_state_spec, verify_explicit_state_cert};
use tla_check::Config;

fn cfg() -> Config {
    Config::parse("INIT Init\nNEXT Next\nINVARIANT Safety\n").expect("config parses")
}

/// (a) A state-dependent `∃j∈0..k : f[j] = TRUE` that HOLDS on every reachable state (slot 0 is always
/// TRUE, and `0..k` always contains 0). `k` ranges `0,1,2` (state-dependent — NOT a fixed set), so the
/// only recognizer arm that can admit it is the state-dependent-interval expansion. Certifies, kernel-
/// re-checks, AND passes the full Leg-E re-derivation (which recomputes `M` from the re-enumerated `R`).
#[test]
fn state_dep_exists_holds_certifies_and_verifies() {
    let spec = "---- MODULE SDExists ----\n\
                EXTENDS Integers\n\
                VARIABLES f, k\n\
                Init == f = [i \\in 0..2 |-> TRUE] /\\ k = 0\n\
                Next == k' = (IF k < 2 THEN k + 1 ELSE k) /\\ f' = f\n\
                Safety == \\E j \\in 0 .. k : f[j] = TRUE\n\
                ====\n";
    let config = cfg();
    let cert = certify_explicit_state_spec(spec, &config).expect(
        "a state-dependent ∃ that holds everywhere must certify via the interval expansion",
    );
    assert!(
        cert.safety_pred.is_some(),
        "the state-dependent ∃ rides the general safety leg (an expanded ⋁ of guarded instances)"
    );
    assert!(
        cert.safety_general.is_some(),
        "kernel-checked ⋀_{{s∈R}} Safety(s) leg present"
    );
    assert!(verify_explicit_state_cert(&cert), "kernel re-check passes");

    // Full Leg-E: re-enumerate the spec, recompute `M` from `R`, re-recognize the invariant, and require
    // the SAME expanded IR — then re-run the kernel end-to-end.
    let sc = tla_check::cert::build_explicit_fixpoint_certificate(spec, &config, cert);
    let report = tla_check::cert::verify_safety_certificate(&sc);
    assert!(
        report.accepted,
        "full Leg-E re-check accepts: {}",
        report.detail
    );
}

/// (b) A VIOLATED state-dependent `∃`: `f` is all FALSE, so `∃j∈0..k : f[j]=TRUE` is FALSE at every
/// reachable state. The expanded disjunction reduces to `Bool.false` in the kernel ⇒ NO certificate.
#[test]
fn state_dep_exists_violated_declines() {
    let spec = "---- MODULE SDExistsBad ----\n\
                EXTENDS Integers\n\
                VARIABLES f, k\n\
                Init == f = [i \\in 0..2 |-> FALSE] /\\ k = 0\n\
                Next == k' = (IF k < 2 THEN k + 1 ELSE k) /\\ f' = f\n\
                Safety == \\E j \\in 0 .. k : f[j] = TRUE\n\
                ====\n";
    assert!(
        certify_explicit_state_spec(spec, &cfg()).is_none(),
        "a violated state-dependent ∃ must fail closed (the kernel rejects the expanded disjunction)"
    );
}

/// (c) THE GUARD IS REAL. At the reachable state `k = 0`, `f = [FALSE, TRUE, TRUE]`: the real domain
/// `0..0 = {0}` has `f[0] = FALSE`, so `∃j∈0..k : f[j]=TRUE` is FALSE there — the invariant is VIOLATED
/// and MUST decline. But `f[1] = f[2] = TRUE` with `j = 1,2 > k`, and another reachable state has
/// `k = 2` so `M = 2`; a buggy expansion that DROPPED the `k ≤ hi` guard would compute
/// `f[0] ∨ f[1] ∨ f[2] = TRUE` and (wrongly) CERTIFY. A correct guard excludes the `j > k` witnesses ⇒
/// decline. This is the test that a false-safe (guard-ignoring) expansion would pass.
#[test]
fn state_dep_exists_guard_excludes_out_of_range_witness() {
    let spec = "---- MODULE SDGuard ----\n\
                EXTENDS Integers\n\
                VARIABLES f, k\n\
                Init == f = [i \\in 0..2 |-> IF i = 0 THEN FALSE ELSE TRUE] /\\ k = 0\n\
                Next == k' = 2 /\\ f' = f\n\
                Safety == \\E j \\in 0 .. k : f[j] = TRUE\n\
                ====\n";
    assert!(
        certify_explicit_state_spec(spec, &cfg()).is_none(),
        "the k≤hi guard must exclude the j>k witnesses — a violated ∃ with a TRUE slot ABOVE the token \
         position must decline (a guard-dropping expansion would falsely certify)"
    );
}

/// (d) FAIL-CLOSED on an upper bound with no cheap sound static bound: `0..(k % 2)` has `hi = k % 2`,
/// a `Nat.mod` term for which `val_ir_static_upper_bound` declines (returns `None`). Even though the
/// invariant actually HOLDS on every reachable state (`f[0] = TRUE` and `0..(k%2)` always contains 0),
/// the state-dependent-interval arm CANNOT derive `M`, so it declines — no certificate is minted.
#[test]
fn state_dep_exists_non_affine_bound_fails_closed() {
    let spec = "---- MODULE SDMod ----\n\
                EXTENDS Integers\n\
                VARIABLES f, k\n\
                Init == f = [i \\in 0..2 |-> TRUE] /\\ k = 0\n\
                Next == k' = (IF k < 2 THEN k + 1 ELSE k) /\\ f' = f\n\
                Safety == \\E j \\in 0 .. (k % 2) : f[j] = TRUE\n\
                ====\n";
    assert!(
        certify_explicit_state_spec(spec, &cfg()).is_none(),
        "a non-affine (Mod) interval upper bound has no static bound ⇒ fail closed even when the \
         invariant holds"
    );
}

/// A state-dependent `∀j∈0..k : f[j] = TRUE` (the conjunctive dual) that HOLDS everywhere (`f` is all
/// TRUE) certifies via `⋀_{k=lo..M} (k ≤ hi ⇒ f[k]=TRUE)`, and its VIOLATED sibling (a FALSE slot inside
/// the token range) declines — exercising the `Implies`-guarded ∀ fold, not just the ∃ path.
#[test]
fn state_dep_forall_holds_and_violated() {
    let good = "---- MODULE SDForall ----\n\
                EXTENDS Integers\n\
                VARIABLES f, k\n\
                Init == f = [i \\in 0..2 |-> TRUE] /\\ k = 0\n\
                Next == k' = (IF k < 2 THEN k + 1 ELSE k) /\\ f' = f\n\
                Safety == \\A j \\in 0 .. k : f[j] = TRUE\n\
                ====\n";
    let cert = certify_explicit_state_spec(good, &cfg())
        .expect("a state-dependent ∀ that holds everywhere must certify");
    assert!(
        verify_explicit_state_cert(&cert),
        "kernel re-check passes for the ∀ expansion"
    );

    // VIOLATED: slot 2 is FALSE and `k` reaches 2, so at `k = 2` the real `∀j∈0..2` is FALSE.
    let bad = "---- MODULE SDForallBad ----\n\
               EXTENDS Integers\n\
               VARIABLES f, k\n\
               Init == f = [i \\in 0..2 |-> IF i = 2 THEN FALSE ELSE TRUE] /\\ k = 0\n\
               Next == k' = (IF k < 2 THEN k + 1 ELSE k) /\\ f' = f\n\
               Safety == \\A j \\in 0 .. k : f[j] = TRUE\n\
               ====\n";
    assert!(
        certify_explicit_state_spec(bad, &cfg()).is_none(),
        "a violated state-dependent ∀ must fail closed (the kernel rejects the expanded conjunction)"
    );
}
