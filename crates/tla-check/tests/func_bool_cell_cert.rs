// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! FUNCTION-of-BOOL cell encoding for the explicit-state certificate lane (the Bool-valued analogue of
//! `func_enum_cell_cert.rs`, unblocking EWD840's `active \in [Node -> BOOLEAN]`):
//!
//!   * a state variable holding a FUNCTION `[d \in 0..arity-1 |-> BOOLEAN]` (the EWD840 `active` node
//!     status) encodes POSITIONALLY into ONE `Nat` cell — the existing `ColSort::Func{base, arity,
//!     cells}` positional pack with per-position `CellSort::Bool` leaves (`TRUE`->1, `FALSE`->0), so a
//!     Bool-valued function needs NO new cell sort and leaves every String/model-value FuncEnum cert
//!     BYTE-IDENTICAL;
//!   * an EQUALITY invariant over a func-of-Bool APPLICATION at a LITERAL index (`active[1] = TRUE`,
//!     `active[i] = FALSE`, `active[i] \in BOOLEAN`) certifies + kernel-re-checks + full-Leg-E-accepts —
//!     the digit-extraction `(pack / base^i) mod base` is now admitted by `pred_exact` over a `Func`
//!     column (not just `Record`), EQUALITY-ONLY;
//!   * a bare Bool-position access used DIRECTLY as a predicate (`active[0]`) recognizes as `= TRUE`;
//!   * FUNCTION-SET membership `active \in [Domain -> BOOLEAN]` (the TypeOK class) recognizes as the
//!     per-index `digit \in {0,1}` tautology and certifies;
//!   * the encoding is truth-EXACT for equality only — an ORDERING / non-literal / out-of-range `[i]`
//!     access fails closed (never a false certificate);
//!   * tamper: a mutated pack cell fails verify.

#![cfg(feature = "clean-cic")]

use tla_check::explicit_fixpoint_cert::{
    certify_explicit_state_spec, verify_explicit_state_cert, CellSort, ColSort, EnumKind,
};
use tla_check::Config;

fn cfg() -> Config {
    Config::parse("INIT Init\nNEXT Next\nINVARIANT Safety\n").expect("config parses")
}

/// A three-slot Bool function `active: [0..2 -> BOOLEAN]` whose slot 0 flips `TRUE -> FALSE`, with an
/// `active[1] = TRUE` invariant. The column's sort is the positional `Func{base:10, arity:3, cells:[Bool;3]}`
/// pack (`TRUE`->1 / `FALSE`->0), R is the two packs, and the cert kernel-re-verifies + round-trips.
#[test]
fn func_bool_active_certifies_and_verifies() {
    let spec = "---- MODULE ActFunc ----\n\
                EXTENDS Integers\n\
                VARIABLE active\n\
                Init == active = [i \\in 0..2 |-> TRUE]\n\
                Next == active' = [active EXCEPT ![0] = FALSE]\n\
                Safety == active[1] = TRUE\n\
                ====\n";
    let cert = certify_explicit_state_spec(spec, &cfg()).expect("func-of-Bool cert mints");
    assert_eq!(
        cert.sorts,
        vec![ColSort::Func {
            base: 10,
            arity: 3,
            cells: vec![CellSort::Bool; 3],
            dom: vec![],
            dom_kind: EnumKind::Model,
        }],
        "the column's sort is the positional Func pack with per-position Bool leaves"
    );
    // R = the two positional packs (base 10): [T,T,T] = 1+10+100 = 111, [F,T,T] = 0+10+100 = 110.
    assert_eq!(
        cert.reachable,
        vec![vec![110], vec![111]],
        "R is the two positional packs"
    );
    // The invariant rides the general safety leg (no Int column ⇒ no nonneg tuple leg).
    assert!(
        cert.safety_pred.is_some(),
        "func-of-Bool equality invariant rides the general safety leg"
    );
    assert!(
        cert.safety_general.is_some(),
        "kernel-checked ⋀_{{s∈R}} Safety(s) leg"
    );
    assert!(verify_explicit_state_cert(&cert), "kernel re-check passes");

    // Serde round-trip: the sort survives and still verifies.
    let bytes = serde_json::to_vec(&cert).expect("serialize");
    let back: tla_check::explicit_fixpoint_cert::ExplicitFixpointCert =
        serde_json::from_slice(&bytes).expect("deserialize");
    assert_eq!(cert, back, "func-of-Bool cert round-trips");
    assert!(
        verify_explicit_state_cert(&back),
        "the re-loaded func-of-Bool cert verifies"
    );
}

/// A bare Bool-position access `active[0]` used DIRECTLY as a predicate ≡ `active[0] = TRUE` — the form
/// EWD840's `~active[i]` negates. Certifies + kernel-re-checks.
#[test]
fn func_bool_bare_predicate_certifies() {
    let spec = "---- MODULE ActBare ----\n\
                EXTENDS Integers\n\
                VARIABLE active\n\
                Init == active = [i \\in 0..2 |-> TRUE]\n\
                Next == active' = active\n\
                Safety == active[0]\n\
                ====\n";
    let cert = certify_explicit_state_spec(spec, &cfg()).expect("bare Bool-access cert mints");
    assert!(cert.safety_general.is_some());
    assert!(verify_explicit_state_cert(&cert), "kernel re-check passes");
}

/// FUNCTION-SET membership `active \in [Node -> BOOLEAN]` as a (TypeOK-style) invariant — the EWD840 gap.
/// Recognizes as the per-index `digit \in {0,1}` tautology and certifies via the full Leg-E re-check path
/// (re-enumerates the spec, re-derives the Func sort + safety IR, binds them to the cert). Uses a
/// zero-arity operator `Node` for the domain to exercise the `cert_inline` FuncSet traversal.
#[test]
fn func_bool_function_set_membership_full_lege() {
    let spec = "---- MODULE ActMem ----\n\
                EXTENDS Integers\n\
                VARIABLE active\n\
                Node == 0..2\n\
                Init == active = [i \\in Node |-> TRUE]\n\
                Next == active' = [active EXCEPT ![0] = FALSE]\n\
                Safety == active \\in [Node -> BOOLEAN]\n\
                ====\n";
    let config = cfg();
    let cert = certify_explicit_state_spec(spec, &config).expect("func-set membership cert mints");
    assert_eq!(
        cert.sorts,
        vec![ColSort::Func {
            base: 10,
            arity: 3,
            cells: vec![CellSort::Bool; 3],
            dom: vec![],
            dom_kind: EnumKind::Model,
        }],
    );
    assert!(cert.safety_pred.is_some());
    assert!(verify_explicit_state_cert(&cert), "kernel re-check passes");
    // Full Leg-E: build a SafetyCertificate and re-verify end-to-end.
    let sc = tla_check::cert::build_explicit_fixpoint_certificate(spec, &config, cert);
    let report = tla_check::cert::verify_safety_certificate(&sc);
    assert!(
        report.accepted,
        "full Leg-E re-check accepts: {}",
        report.detail
    );
}

/// TAMPER: a mutated pack cell that violates the `active[1] = TRUE` invariant fails the kernel re-check
/// (the `safety_general` leg reduces to `Bool.false`).
#[test]
fn tampered_func_bool_cert_rejects() {
    let spec = "---- MODULE ActTamper ----\n\
                EXTENDS Integers\n\
                VARIABLE active\n\
                Init == active = [i \\in 0..2 |-> TRUE]\n\
                Next == active' = [active EXCEPT ![0] = FALSE]\n\
                Safety == active[1] = TRUE\n\
                ====\n";
    let config = cfg();
    let cert = certify_explicit_state_spec(spec, &config).expect("func-of-Bool cert mints");

    // Push a reachable pack whose slot-1 digit is no longer TRUE: pack 100 ⇒ active[1] = (100/10) mod 10 = 0.
    let mut kernel_tampered = cert.clone();
    kernel_tampered.reachable[0][0] = 100;
    assert!(
        !verify_explicit_state_cert(&kernel_tampered),
        "a tampered func-of-Bool pack must fail the kernel re-check"
    );

    // Control: the genuine cert is accepted through the full Leg-E path.
    let sc_ok = tla_check::cert::build_explicit_fixpoint_certificate(spec, &config, cert);
    let ok_report = tla_check::cert::verify_safety_certificate(&sc_ok);
    assert!(
        ok_report.accepted,
        "the genuine func-of-Bool cert must be accepted: {}",
        ok_report.detail
    );
}

/// FAIL-CLOSED at the surface `[i]`-access recognizer: an ORDERING over a Bool-func application (an enum
/// index carries no order), an OUT-OF-RANGE literal index, or a NON-LITERAL index is not in the recognized
/// fragment, so no certificate is minted.
#[test]
fn func_bool_out_of_fragment_declines() {
    let base = "---- MODULE ActDecline ----\n\
                EXTENDS Integers\n\
                VARIABLE active\n\
                Init == active = [i \\in 0..2 |-> TRUE]\n\
                Next == active' = [active EXCEPT ![0] = FALSE]\n";
    // ORDERING over a Bool-func application (meaningless on a Bool code) ⇒ decline.
    let ord = format!("{base}Safety == active[1] < active[2]\n====\n");
    assert!(
        certify_explicit_state_spec(&ord, &cfg()).is_none(),
        "an ordering over func-of-Bool applications must fail closed (no cert)"
    );
    // OUT-OF-RANGE index (arity 3, index 9) ⇒ decline.
    let oor = format!("{base}Safety == active[9] = TRUE\n====\n");
    assert!(
        certify_explicit_state_spec(&oor, &cfg()).is_none(),
        "an out-of-range func-of-Bool index must fail closed (no cert)"
    );
    // NON-LITERAL index (`1 + 0` is an Add, not a literal) ⇒ decline.
    let nonlit = format!("{base}Safety == active[1 + 0] = TRUE\n====\n");
    assert!(
        certify_explicit_state_spec(&nonlit, &cfg()).is_none(),
        "a non-literal func-of-Bool index must fail closed (no cert)"
    );
}

/// SOUNDNESS (2026-07-04 quantified-process increment): a bounded `\A i ∈ 0..n : f[i] = …` whose
/// body applies a FUNCTION at the BOUND-VAR index must expand by AST literal-substitution (`f[i]` →
/// `f[0] ∧ f[1] ∧ …`), certify when TRUE on every reachable state, and FAIL CLOSED when violated.
#[test]
fn forall_over_func_at_bound_var_index() {
    let base = "---- MODULE FA ----\n\
                EXTENDS Integers\n\
                VARIABLE active\n\
                Init == active = [i \\in 0..2 |-> TRUE]\n\
                Next == \\E k \\in 0..2 : active' = [active EXCEPT ![k] = TRUE]\n";
    // TRUE on every reachable state (every slot starts TRUE and stays TRUE) ⇒ certifies.
    let good = format!("{base}Safety == \\A i \\in 0..2 : active[i] = TRUE\n====\n");
    let cert = certify_explicit_state_spec(&good, &cfg())
        .expect("a forall over func-at-bound-var that holds everywhere must certify");
    assert!(
        verify_explicit_state_cert(&cert),
        "the quantified-func cert must re-check"
    );

    // VIOLATED: a spec where some reachable state has a FALSE slot ⇒ the expanded conjunction is
    // false in the kernel ⇒ fail-closed (no cert). Init has slot 0 = FALSE.
    let bad = "---- MODULE FAbad ----\n\
               EXTENDS Integers\n\
               VARIABLE active\n\
               Init == active = [i \\in 0..2 |-> IF i = 0 THEN FALSE ELSE TRUE]\n\
               Next == active' = active\n\
               Safety == \\A i \\in 0..2 : active[i] = TRUE\n\
               ====\n";
    assert!(
        certify_explicit_state_spec(bad, &cfg()).is_none(),
        "a violated forall-over-func invariant must fail closed (kernel rejects the conjunction)"
    );
}
