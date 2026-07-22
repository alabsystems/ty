// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! DEFINED-OPERATOR SET domains + LABELED predicates in the explicit-state certificate lane.
//!
//! A bounded quantifier / membership whose domain is a user-defined operator with a SET/interval body
//! — `Dom == 0..N-1`, `Color == {"white","black"}` — is inlined to its literal form by
//! [`tla_check::cert_inline`] BEFORE the kernel recognizer runs (`\A i ∈ Dom : …` ⇒ `\A i ∈ 0..N-1`,
//! then the configured `N` literalizes ⇒ `\A i ∈ 0..1`), so the R4 finite-Int expansion folds it. A
//! LABELED predicate `P0:: e` (Dijkstra's `Inv == \/ P0:: … \/ …`, EWD840) is stripped by the same pass:
//! a label denotes exactly its body, and the recognizer reads no `Label` node, so a labeled defined-
//! operator-set forall recognizes identically to its unlabeled twin.
//!
//! This is the CONTROLLED SOUNDNESS PROOF for that front-end inlining:
//!   * a TRUE `\A i ∈ Dom : f[i] = 0` (both plain and LABELED) over `Dom == 0..N-1`, N=2, certifies,
//!     kernel-re-checks, and full-Leg-E-accepts;
//!   * its VIOLATED twin (some reachable `f[i] = 1`) is NOT CERTIFIED — the fold recognizes the SAME
//!     shape (proven by the certifying twin) but the kernel's `R⊆Safety` leg reduces to `Bool.false`
//!     on the violating reachable state, so the inlined fold does NOT hide the violation.

#![cfg(feature = "clean-cic")]

use tla_check::explicit_fixpoint_cert::{certify_explicit_state_spec, verify_explicit_state_cert};
use tla_check::Config;

/// A config binding `N = 2` (so `Dom == 0..N-1` = `{0,1}`) with the given invariant.
fn cfg_n2(inv: &str) -> Config {
    Config::parse(&format!(
        "CONSTANTS N = 2\nINIT Init\nNEXT Next\nINVARIANT {inv}\n"
    ))
    .expect("config parses")
}

/// TRUE, PLAIN: `\A i ∈ Dom : f[i] = 0` over the DEFINED-OPERATOR set `Dom == 0..N-1` (N=2). The pass
/// inlines `Dom` ⇒ `0..1` and the R4 fold expands to `f[0]=0 ∧ f[1]=0`; the cert kernel-re-checks and
/// full-Leg-E-accepts. (Isolates the pre-existing operator-set inlining from the label transparency below.)
#[test]
fn dom_operator_forall_true_certifies() {
    let spec = "---- MODULE DomForallTrue ----\n\
                EXTENDS Integers\n\
                CONSTANT N\n\
                VARIABLE f\n\
                Dom == 0..N-1\n\
                Init == f = [i \\in Dom |-> 0]\n\
                Next == f' = f\n\
                Inv == \\A i \\in Dom : f[i] = 0\n\
                ====\n";
    let config = cfg_n2("Inv");
    let cert = certify_explicit_state_spec(spec, &config)
        .expect("a TRUE ∀-over-defined-operator-set invariant certifies");
    assert!(
        cert.safety_general.is_some(),
        "the fold rides the general safety leg"
    );
    assert!(verify_explicit_state_cert(&cert), "kernel re-check passes");
    let sc = tla_check::cert::build_explicit_fixpoint_certificate(spec, &config, cert);
    let report = tla_check::cert::verify_safety_certificate(&sc);
    assert!(
        report.accepted,
        "full Leg-E re-derives the inlined fold and ACCEPTS: {}",
        report.detail
    );
}

/// TRUE, LABELED (the EWD840-shaped construct): `Inv == P0:: \A i ∈ Dom : f[i] = 0`. The pass STRIPS the
/// `P0::` label (value-preserving) AND inlines `Dom`, so the labeled predicate certifies identically to
/// its unlabeled twin. Directly exercises the label-transparency arm of `cert_inline` — without it the
/// recognizer (which has no `Label` case) would decline.
#[test]
fn dom_operator_labeled_forall_true_certifies() {
    let spec = "---- MODULE DomForallLbl ----\n\
                EXTENDS Integers\n\
                CONSTANT N\n\
                VARIABLE f\n\
                Dom == 0..N-1\n\
                Init == f = [i \\in Dom |-> 0]\n\
                Next == f' = f\n\
                Inv == P0:: \\A i \\in Dom : f[i] = 0\n\
                ====\n";
    let config = cfg_n2("Inv");
    let cert = certify_explicit_state_spec(spec, &config)
        .expect("a TRUE LABELED ∀-over-defined-operator-set invariant certifies (label stripped)");
    assert!(verify_explicit_state_cert(&cert), "kernel re-check passes");
    let sc = tla_check::cert::build_explicit_fixpoint_certificate(spec, &config, cert);
    let report = tla_check::cert::verify_safety_certificate(&sc);
    assert!(
        report.accepted,
        "full Leg-E accepts the labeled fold: {}",
        report.detail
    );
}

/// VIOLATED, LABELED: same `Inv == P0:: \A i ∈ Dom : f[i] = 0`, but `Next` can flip a slot to 1, so a
/// reachable state has `f[i] = 1`. NOT CERTIFIED: the recognizer accepts the SAME shape as the certifying
/// twin above (so this is the KERNEL rejecting, not a recognizer refusal), but the `R⊆Safety` leg is
/// `Bool.false` on the violating state ⇒ no cert. The inlined fold does NOT hide the violation. (`ty check`
/// independently reports the counterexample `f = [0|->1, 1|->0]` — the CLI cross-check.)
#[test]
fn dom_operator_labeled_forall_violation_declines() {
    let spec = "---- MODULE DomForallBad ----\n\
                EXTENDS Integers\n\
                CONSTANT N\n\
                VARIABLE f\n\
                Dom == 0..N-1\n\
                Init == f = [i \\in Dom |-> 0]\n\
                Next == \\E i \\in Dom : f' = [f EXCEPT ![i] = 1]\n\
                Inv == P0:: \\A i \\in Dom : f[i] = 0\n\
                ====\n";
    assert!(
        certify_explicit_state_spec(spec, &cfg_n2("Inv")).is_none(),
        "a reachable f[i]=1 violates the fold ⇒ NOT CERTIFIED (the labeled fold does not hide it)"
    );
}
