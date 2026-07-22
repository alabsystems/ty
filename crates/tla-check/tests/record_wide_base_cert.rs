// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Roadmap R3 — RECORD/FUNC cell encoding at DERIVED per-column base/arity (no fixed `10/1024`,`4/6`):
//!
//!   * a record column whose observed field values exceed the FLOOR radix is re-encoded at the
//!     SMALLEST admitting per-column base `max(10, maxValue+1)`, chosen by the adaptive derivation and
//!     BAKED into the serialized `ColSort::Record{base,..}` (no fixed wide rung);
//!   * the arity bound is DERIVED (`base^arity < 2^64`), not a magic cap: a base-10 record of arity 5
//!     (previously over the cap) now certifies at base 10; a pack that overflows the u64 cell fails closed;
//!   * a base-10 spec's certificate is BYTE-IDENTICAL to one minted before the widening landed
//!     (fixtures under `tests/fixtures/` were dumped from the pre-widening code) — the digest
//!     back-compat hard rule;
//!   * a tampered derived-base cert REJECTS (kernel re-check and the Leg-E spec re-derivation binding);
//!   * the CoffeeCan class (`TypeInvariant == can \in Can`, `Can == [black: 0..M, white: 0..M]`
//!     over a configured CONSTANT) certifies end-to-end through the record-set membership
//!     recognition + the zero-arity operator/CONSTANT inliner.

#![cfg(feature = "clean-cic")]

use tla_check::explicit_fixpoint_cert::{
    certify_explicit_state_spec, verify_explicit_state_cert, ColSort,
};
use tla_check::Config;

fn cfg() -> Config {
    Config::parse("INIT Init\nNEXT Next\nINVARIANT Safety\n").expect("config parses")
}

/// A two-field record with values beyond base 10 certifies at the DERIVED tight base
/// `max(10, 99+1) = 100` — NOT a fixed wide rung; the pack is the base-100 numeral over the CANONICAL
/// (sorted) field order, and the cert round-trips + verifies.
#[test]
fn wide_record_field_values_certify_at_the_derived_base() {
    let spec = "---- MODULE WideRec ----\n\
                EXTENDS Integers\n\
                VARIABLES x, r\n\
                Init == x = 0 /\\ r = [black |-> 99, white |-> 3]\n\
                Next == x' = x /\\ r' = r\n\
                Safety == x >= 0\n\
                ====\n";
    let cert = certify_explicit_state_spec(spec, &cfg()).expect("wide record cert mints");
    assert_eq!(
        cert.sorts,
        vec![
            ColSort::Int,
            ColSort::Record {
                base: 100, // max(10, maxValue 99 + 1) — the smallest-admitting derived base
                fields: vec!["black".to_string(), "white".to_string()],
                cells: vec![], // all-Int record (canonical empty per-position kinds)
            }
        ],
        "the derived column's sort SAYS base 100 (99+1), not a fixed 1024"
    );
    // pack = black·100^0 + white·100^1 = 99 + 3·100 = 399 (canonical sorted field order).
    assert_eq!(
        cert.reachable,
        vec![vec![0, 99 + 3 * 100]],
        "the base-100 positional pack"
    );
    assert!(
        verify_explicit_state_cert(&cert),
        "the derived-base cert kernel-re-verifies"
    );
    // Serde round-trip: the derived sort (base in the serialized sort) survives and still verifies.
    let bytes = serde_json::to_vec(&cert).expect("serialize");
    let back: tla_check::explicit_fixpoint_cert::ExplicitFixpointCert =
        serde_json::from_slice(&bytes).expect("deserialize");
    assert_eq!(cert, back, "derived-base cert round-trips byte-identically");
    assert!(
        verify_explicit_state_cert(&back),
        "the re-loaded derived-base cert verifies"
    );
}

/// Arity 5 (values all `< 10`) exceeds the old DEFAULT arity cap (4) but its base-10 pack `10^5` fits a
/// `u64` — with the DERIVED arity bound the column now certifies at base 10 (no widening: values fit the
/// floor, so the smallest-admitting base IS 10).
#[test]
fn record_arity_five_certifies_at_base_ten() {
    let spec = "---- MODULE Arity5 ----\n\
                EXTENDS Integers\n\
                VARIABLES x, r\n\
                Init == x = 0 /\\ r = [a |-> 1, b |-> 2, c |-> 3, d |-> 4, e |-> 5]\n\
                Next == x' = x /\\ r' = r\n\
                Safety == x >= 0\n\
                ====\n";
    let cert = certify_explicit_state_spec(spec, &cfg()).expect("arity-5 record cert mints");
    let ColSort::Record { base, fields, .. } = &cert.sorts[1] else {
        panic!("column 1 must be a Record sort: {:?}", cert.sorts)
    };
    assert_eq!(
        *base, 10,
        "values all < 10 ⇒ the smallest-admitting base is the floor 10"
    );
    assert_eq!(fields.len(), 5);
    // pack = 1 + 2·10 + 3·100 + 4·1000 + 5·10000 = 54321.
    assert_eq!(
        cert.reachable,
        vec![vec![0, 54321]],
        "the base-10 positional pack"
    );
    assert!(verify_explicit_state_cert(&cert));
}

/// PACK OVERFLOW fails closed: the derived base has no fixed ceiling, but `base^arity` must fit a `u64`.
/// A base-10 record of arity 20 has pack universe `10^20 > 2^64` ⇒ `checked_pow` declines ⇒ no cert.
#[test]
fn record_pack_overflow_fails_closed() {
    let spec = "---- MODULE Arity20 ----\n\
                EXTENDS Integers\n\
                VARIABLES x, r\n\
                Init == x = 0 /\\ r = [a0|->1,a1|->1,a2|->1,a3|->1,a4|->1,a5|->1,a6|->1,a7|->1,a8|->1,a9|->1,b0|->1,b1|->1,b2|->1,b3|->1,b4|->1,b5|->1,b6|->1,b7|->1,b8|->1,b9|->1]\n\
                Next == x' = x /\\ r' = r\n\
                Safety == x >= 0\n\
                ====\n";
    assert!(
        certify_explicit_state_spec(spec, &cfg()).is_none(),
        "arity 20 at base 10 overflows the u64 pack (10^20 > 2^64) — fail closed"
    );
}

/// The base TRACKS the value with no fixed ceiling: a single-field record with value `V` derives base
/// `V+1` and packs to `V`. `1023` derives base `1024`; `1024` derives base `1025` — BOTH certify (the old
/// fixed `<1024` wall is gone; the only wall is the u64 pack ceiling, unreachable at arity 1).
#[test]
fn derived_base_tracks_the_value() {
    for (v, want_base) in [(1023u64, 1024u64), (1024, 1025), (50_000, 50_001)] {
        let spec = format!(
            "---- MODULE B ----\nEXTENDS Integers\nVARIABLES x, r\n\
             Init == x = 0 /\\ r = [a |-> {v}]\nNext == x' = x /\\ r' = r\n\
             Safety == x >= 0\n====\n"
        );
        let cert = certify_explicit_state_spec(&spec, &cfg())
            .unwrap_or_else(|| panic!("value {v} certifies at the derived base"));
        let ColSort::Record { base, .. } = &cert.sorts[1] else {
            panic!("column 1 must be a Record sort")
        };
        assert_eq!(*base, want_base, "value {v} ⇒ derived base {want_base}");
        assert_eq!(
            cert.reachable,
            vec![vec![0, v]],
            "single-field pack = the value"
        );
        assert!(verify_explicit_state_cert(&cert));
    }
}

/// DIGEST BACK-COMPAT (hard rule): a base-10 spec's cert JSON is BYTE-IDENTICAL to the cert the
/// PRE-WIDENING code minted (fixtures dumped from the pre-change tree). The floor base, packs,
/// sorts, kernel terms, and serialization are all untouched by R3's derivation for values that fit base 10.
#[test]
fn base10_cert_json_byte_identical_to_pre_widening_fixture() {
    let cases: &[(&str, &str, &str)] = &[
        (
            "base10_record",
            "---- MODULE RecSerde ----\nEXTENDS Integers\nVARIABLES x, r\nInit == x = 0 /\\ r = [a |-> 1, b |-> 2]\nNext == x' = x /\\ r' = r\nSafety == x >= 0\n====\n",
            include_str!("fixtures/fixture_base10_record.json"),
        ),
        (
            "interval_safety",
            "---- MODULE IntervalSafety ----\nEXTENDS Naturals\nVARIABLE x\nInit == x = 0\nNext == x' = x\nSafety == x \\in 0..2\n====\n",
            include_str!("fixtures/fixture_interval_safety.json"),
        ),
    ];
    for (name, spec, fixture) in cases {
        let config = Config::parse("INIT Init\nNEXT Next\nINVARIANT Safety\n").unwrap();
        let cert = certify_explicit_state_spec(spec, &config)
            .unwrap_or_else(|| panic!("{name} must still certify"));

        // Commit 31dae3c6 added the default-on, ORTHOGONAL `deadlock_free` corroboration leg
        // AFTER these fixtures were dumped (at 3c459b29). Assert the leg is present and that the
        // full cert kernel-re-verifies, then STRIP the post-fixture leg before the digest
        // back-compat byte-compare: `deadlock_free` carries
        // `#[serde(skip_serializing_if = "Option::is_none")]`, so a `None` leg is OMITTED from the
        // JSON — restoring byte-identity with the pre-leg fixture while the safety-part encoding
        // the fixture protects stays fully exercised.
        assert!(
            cert.deadlock_free.is_some(),
            "{name}: the default-on deadlock-free corroboration leg must be present"
        );
        assert!(
            verify_explicit_state_cert(&cert),
            "{name}: the full cert (safety + orthogonal deadlock-free legs) must kernel-re-verify"
        );

        let mut stripped = cert.clone();
        stripped.deadlock_free = None;
        let json = String::from_utf8(serde_json::to_vec(&stripped).unwrap()).unwrap();
        assert_eq!(
            json, *fixture,
            "{name}: the base-10 cert JSON (deadlock-free leg stripped) must be byte-identical to the pre-widening fixture"
        );
    }
}

/// TAMPER: (a) a flipped reachable cell fails the kernel re-check; (b) a derived-base cert whose sort
/// base is flipped back to 10 fails the Leg-E spec re-derivation binding (`verify` re-enumerates
/// the spec, re-applies the smallest-base derivation, and requires sort equality).
#[test]
fn tampered_wide_cert_rejects() {
    let spec = "---- MODULE WideTamper ----\n\
                EXTENDS Integers\n\
                VARIABLES x, r\n\
                Init == x = 0 /\\ r = [black |-> 99, white |-> 3]\n\
                Next == x' = x /\\ r' = r\n\
                Safety == x >= 0\n\
                ====\n";
    let config = cfg();
    let cert = certify_explicit_state_spec(spec, &config).expect("derived-base cert mints");

    // (a) kernel-level tamper: nudge the packed record cell of the one reachable state.
    let mut kernel_tampered = cert.clone();
    kernel_tampered.reachable[0][1] += 1;
    assert!(
        !verify_explicit_state_cert(&kernel_tampered),
        "a tampered reachable pack must fail the kernel re-check"
    );

    // (b) sort-base tamper: claim the column was packed at base 10. The kernel legs are literal
    // Nat equalities (base-blind), so the binding gate is Leg-E: re-derivation from the embedded
    // spec re-applies the smallest-base derivation (base 100) and the sorts no longer match ⇒ REJECTED.
    let mut base_tampered = cert.clone();
    if let ColSort::Record { base, .. } = &mut base_tampered.sorts[1] {
        *base = 10;
    } else {
        panic!("column 1 must be a Record sort");
    }
    let sc = tla_check::cert::build_explicit_fixpoint_certificate(spec, &config, base_tampered);
    let report = tla_check::cert::verify_safety_certificate(&sc);
    assert!(
        !report.accepted,
        "a derived-base cert with the sort base flipped to 10 must be rejected: {}",
        report.detail
    );

    // Control: the untampered cert is accepted through the same full Leg-E path.
    let sc_ok = tla_check::cert::build_explicit_fixpoint_certificate(spec, &config, cert);
    let ok_report = tla_check::cert::verify_safety_certificate(&sc_ok);
    assert!(
        ok_report.accepted,
        "the genuine derived-base cert must be accepted: {}",
        ok_report.detail
    );
}

/// The CoffeeCan CLASS end-to-end (scaled instance): ONE record variable, `Init` a record-set
/// comprehension, `Next` a disjunction of EXCEPT actions, and the invariant
/// `TypeInvariant == can \in Can` where `Can == [black: 0..MaxBeanCount, white: 0..MaxBeanCount]`
/// with `MaxBeanCount` a configured CONSTANT — certifies via the inliner + the record-set
/// membership recognition, and the cert re-verifies through the full Leg-E path.
#[test]
fn coffeecan_class_record_set_invariant_certifies_end_to_end() {
    let spec = "---- MODULE MiniCoffeeCan ----\n\
                EXTENDS Naturals\n\
                CONSTANT MaxBeanCount\n\
                VARIABLE can\n\
                Can == [black : 0..MaxBeanCount, white : 0..MaxBeanCount]\n\
                TypeInvariant == can \\in Can\n\
                Init == can \\in {c \\in Can : c.black + c.white \\in 1..MaxBeanCount}\n\
                BeanCount == can.black + can.white\n\
                PickSameColorBlack ==\n\
                    /\\ BeanCount > 1\n\
                    /\\ can.black >= 2\n\
                    /\\ can' = [can EXCEPT !.black = @ - 1]\n\
                PickSameColorWhite ==\n\
                    /\\ BeanCount > 1\n\
                    /\\ can.white >= 2\n\
                    /\\ can' = [can EXCEPT !.black = @ + 1, !.white = @ - 2]\n\
                PickDifferentColor ==\n\
                    /\\ BeanCount > 1\n\
                    /\\ can.black >= 1\n\
                    /\\ can.white >= 1\n\
                    /\\ can' = [can EXCEPT !.black = @ - 1]\n\
                Termination == BeanCount = 1 /\\ UNCHANGED can\n\
                Next == PickSameColorWhite \\/ PickSameColorBlack \\/ PickDifferentColor \\/ Termination\n\
                ====\n";
    let config = Config::parse(
        "CONSTANTS\n    MaxBeanCount = 4\nINIT Init\nNEXT Next\nINVARIANT TypeInvariant\n",
    )
    .expect("config parses");
    let cert = certify_explicit_state_spec(spec, &config)
        .expect("the CoffeeCan class must certify (record-set membership + inliner)");
    // R = every can with 1 ≤ black+white ≤ 4: Σ_{s=1..4}(s+1) = 14 states.
    assert_eq!(cert.reachable.len(), 14, "R is the 14-state bean lattice");
    assert_eq!(
        cert.sorts,
        vec![ColSort::Record {
            base: 10,
            fields: vec!["black".to_string(), "white".to_string()],
            cells: vec![]
        }],
        "values ≤ 4 fit the FLOOR radix — no widening (smallest-base rule)"
    );
    assert!(
        cert.safety_pred.is_some(),
        "TypeInvariant rides the general safety leg"
    );
    assert!(
        cert.safety_general.is_some(),
        "kernel-checked ⋀_{{s∈R}} TypeInvariant(s) leg"
    );
    assert!(verify_explicit_state_cert(&cert), "kernel re-check passes");
    let sc = tla_check::cert::build_explicit_fixpoint_certificate(spec, &config, cert);
    let report = tla_check::cert::verify_safety_certificate(&sc);
    assert!(
        report.accepted,
        "full Leg-E re-check accepts: {}",
        report.detail
    );
}

/// The SAME CoffeeCan class at a bean count that pushes field values past the FLOOR radix
/// (`MaxBeanCount = 12` — values in [0,12] ⇒ the column DERIVES base `max(10, 12+1) = 13`) —
/// recognition, derivation, and the kernel legs compose.
#[test]
fn coffeecan_class_derives_base_when_bean_count_exceeds_floor_radix() {
    let spec = "---- MODULE WideCoffee ----\n\
                EXTENDS Naturals\n\
                CONSTANT MaxBeanCount\n\
                VARIABLE can\n\
                Can == [black : 0..MaxBeanCount, white : 0..MaxBeanCount]\n\
                TypeInvariant == can \\in Can\n\
                Init == can = [black |-> MaxBeanCount, white |-> MaxBeanCount]\n\
                Next == UNCHANGED can\n\
                ====\n";
    let config = Config::parse(
        "CONSTANTS\n    MaxBeanCount = 12\nINIT Init\nNEXT Next\nINVARIANT TypeInvariant\n",
    )
    .expect("config parses");
    let cert =
        certify_explicit_state_spec(spec, &config).expect("derived-base CoffeeCan-class cert");
    assert_eq!(
        cert.sorts,
        vec![ColSort::Record {
            base: 13, // max(10, maxValue 12 + 1)
            fields: vec!["black".to_string(), "white".to_string()],
            cells: vec![], // all-Int record
        }],
        "field values 12 ≥ 10 ⇒ the column derives base 13"
    );
    // pack = black·13^0 + white·13^1 = 12 + 12·13 = 168.
    assert_eq!(cert.reachable, vec![vec![12 + 12 * 13]], "the base-13 pack");
    assert!(
        cert.safety_pred.is_some(),
        "record-set membership recognized at the DERIVED base"
    );
    assert!(verify_explicit_state_cert(&cert), "kernel re-check passes");
}
