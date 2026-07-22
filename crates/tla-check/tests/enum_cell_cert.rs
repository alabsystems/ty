// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! FINITE-ENUM cell encoding for the explicit-state certificate lane:
//!
//!   * a scalar state variable holding a `String` drawn from a small fixed label set encodes to the
//!     INDEX of its label in the column's SORTED observed-label union, baked into
//!     `ColSort::Enum{labels}` — the analogue of the record wide-base column property;
//!   * an EQUALITY invariant over the enum column (`s = "a" \/ …`, `s \in {"a","b"}`, `s1 = s2`)
//!     certifies + kernel-re-checks + full-Leg-E-accepts;
//!   * the enum sort is truth-EXACT for equality only — an ORDERING form over an enum column is
//!     REJECTED by `pred_exact` (never a false certificate);
//!   * digest back-compat: adding the `ColSort::Enum` serde variant leaves every non-enum fixture
//!     cert BYTE-IDENTICAL (asserted by the record fixture round-trip);
//!   * tamper: a mutated enum index / mutated label set fails verify.

#![cfg(feature = "clean-cic")]

use tla_check::explicit_fixpoint_cert::{
    certify_explicit_state_spec, verify_explicit_state_cert, ColSort, EnumKind,
};
use tla_check::Config;

fn cfg() -> Config {
    Config::parse("INIT Init\nNEXT Next\nINVARIANT Safety\n").expect("config parses")
}

/// A three-label PC cycle `"a" -> "b" -> "c" -> "a"` with a disjunctive equality invariant. The
/// column's sort is `Enum{["a","b","c"]}` (SORTED union), the reachable set is the three label
/// indices, and the cert kernel-re-verifies + round-trips.
#[test]
fn scalar_enum_cycle_certifies_and_verifies() {
    let spec = "---- MODULE PcCycle ----\n\
                VARIABLE s\n\
                Init == s = \"a\"\n\
                Next == s' = IF s = \"a\" THEN \"b\" ELSE IF s = \"b\" THEN \"c\" ELSE \"a\"\n\
                Safety == s = \"a\" \\/ s = \"b\" \\/ s = \"c\"\n\
                ====\n";
    let cert = certify_explicit_state_spec(spec, &cfg()).expect("scalar enum cert mints");
    assert_eq!(
        cert.sorts,
        vec![ColSort::Enum {
            labels: vec!["a".to_string(), "b".to_string(), "c".to_string()],
            kind: EnumKind::Str,
        }],
        "the column's sort is the SORTED observed-label union"
    );
    // R = the three label indices (sorted tuples): a=0, b=1, c=2.
    assert_eq!(
        cert.reachable,
        vec![vec![0], vec![1], vec![2]],
        "R is the three label indices"
    );
    // The invariant rides the general safety leg (no Int column ⇒ no nonneg tuple leg).
    assert!(
        cert.safety_pred.is_some(),
        "enum equality invariant rides the general safety leg"
    );
    assert!(
        cert.safety_general.is_some(),
        "kernel-checked ⋀_{{s∈R}} Safety(s) leg"
    );
    assert!(verify_explicit_state_cert(&cert), "kernel re-check passes");

    // Serde round-trip: the new Enum sort survives and still verifies.
    let bytes = serde_json::to_vec(&cert).expect("serialize");
    let back: tla_check::explicit_fixpoint_cert::ExplicitFixpointCert =
        serde_json::from_slice(&bytes).expect("deserialize");
    assert_eq!(cert, back, "enum cert round-trips");
    assert!(
        verify_explicit_state_cert(&back),
        "the re-loaded enum cert verifies"
    );
}

/// The `\in {…}` membership form and the full Leg-E re-check path (spec re-derivation binding the
/// stored Enum sort + safety IR to the spec).
#[test]
fn scalar_enum_set_membership_full_lege() {
    let spec = "---- MODULE PcMem ----\n\
                VARIABLE s\n\
                Init == s = \"read\"\n\
                Next == s' = IF s = \"read\" THEN \"write\" ELSE \"read\"\n\
                Safety == s \\in {\"read\", \"write\"}\n\
                ====\n";
    let config = cfg();
    let cert = certify_explicit_state_spec(spec, &config).expect("enum membership cert mints");
    assert_eq!(
        cert.sorts,
        vec![ColSort::Enum {
            labels: vec!["read".to_string(), "write".to_string()],
            kind: EnumKind::Str,
        }],
    );
    assert!(cert.safety_pred.is_some());
    assert!(verify_explicit_state_cert(&cert), "kernel re-check passes");
    // Full Leg-E: build a SafetyCertificate and re-verify end-to-end (re-enumerates the spec,
    // re-derives the Enum sort + safety IR, and binds them to the cert).
    let sc = tla_check::cert::build_explicit_fixpoint_certificate(spec, &config, cert);
    let report = tla_check::cert::verify_safety_certificate(&sc);
    assert!(
        report.accepted,
        "full Leg-E re-check accepts: {}",
        report.detail
    );
}

/// TAMPER: (a) a mutated enum index cell fails the kernel re-check; (b) a mutated label set fails
/// the Leg-E spec re-derivation binding (verify re-enumerates and requires `re.sorts == fp.sorts`).
#[test]
fn tampered_enum_cert_rejects() {
    let spec = "---- MODULE PcTamper ----\n\
                VARIABLE s\n\
                Init == s = \"a\"\n\
                Next == s' = IF s = \"a\" THEN \"b\" ELSE \"a\"\n\
                Safety == s = \"a\" \\/ s = \"b\"\n\
                ====\n";
    let config = cfg();
    let cert = certify_explicit_state_spec(spec, &config).expect("enum cert mints");

    // (a) kernel-level tamper: push a reachable index out of the label range.
    let mut kernel_tampered = cert.clone();
    kernel_tampered.reachable[0][0] += 10;
    assert!(
        !verify_explicit_state_cert(&kernel_tampered),
        "a tampered enum index must fail the kernel re-check"
    );

    // (b) label-set tamper: rename a label in the stored sort. The kernel legs are index-blind Nat
    // equalities, so the binding gate is Leg-E: re-derivation re-collects the real labels and the
    // sorts no longer match ⇒ REJECTED.
    let mut label_tampered = cert.clone();
    if let ColSort::Enum { labels, .. } = &mut label_tampered.sorts[0] {
        labels[0] = "zzz".to_string();
    } else {
        panic!("column 0 must be an Enum sort");
    }
    let sc = tla_check::cert::build_explicit_fixpoint_certificate(spec, &config, label_tampered);
    let report = tla_check::cert::verify_safety_certificate(&sc);
    assert!(
        !report.accepted,
        "a cert with a mutated enum label must be rejected: {}",
        report.detail
    );

    // Control: the genuine cert is accepted through the same full Leg-E path.
    let sc_ok = tla_check::cert::build_explicit_fixpoint_certificate(spec, &config, cert);
    let ok_report = tla_check::cert::verify_safety_certificate(&sc_ok);
    assert!(
        ok_report.accepted,
        "the genuine enum cert must be accepted: {}",
        ok_report.detail
    );
}

/// A two-variable spec with an enum column AND an Int column certifies (the enum column rides the
/// membership legs on its index cells; the Int column carries its nonneg conjunct). Exercises the
/// mixed-sort tuple path.
#[test]
fn enum_plus_int_tuple_certifies() {
    let spec = "---- MODULE PcInt ----\n\
                EXTENDS Integers\n\
                VARIABLES s, n\n\
                Init == s = \"a\" /\\ n = 0\n\
                Next == /\\ s' = IF s = \"a\" THEN \"b\" ELSE \"a\"\n\
                        /\\ n' = n\n\
                Safety == (s = \"a\" \\/ s = \"b\") /\\ n >= 0\n\
                ====\n";
    let cert = certify_explicit_state_spec(spec, &cfg()).expect("enum+int tuple cert mints");
    assert_eq!(cert.sorts.len(), 2);
    assert!(matches!(cert.sorts[0], ColSort::Enum { .. }));
    assert_eq!(cert.sorts[1], ColSort::Int);
    assert!(verify_explicit_state_cert(&cert), "kernel re-check passes");
}
