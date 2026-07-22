// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! VALUE-TYPE LEAF — the controlled SOUNDNESS PROOF for per-POSITION value sorts inside a compound
//! (`Record` / Int-domain `Func`) cell. Each positional-pack digit now carries a `CellSort` (Int / Bool):
//! the pack stays the uniform base-`b` numeral `Σ code_i·b^i`, but a position's digit MEANS an Int value
//! OR a Bool code (`TRUE`→1, `FALSE`→0), fixed by the position's kind.
//!
//! THE SOUNDNESS REQUIREMENT under test: a Bool `TRUE` and an Int `1` are DISTINCT TLA values; they must
//! never share a digit code across states, and a Bool/enum position must be EQUALITY-ONLY (no ordering).
//! The four cases below are the fail-closed proof:
//!   (a) MIXED-KIND: a field Int `1` then Bool `TRUE` ⇒ the column's per-position KIND DISCRIMINANT makes
//!       the two states DIFFERENT sorts ⇒ the cross-state `col_sorts` agreement DECLINES (never collapses).
//!   (b) COLLAPSE GUARD load-bearing: a Bool field with an invariant `r.b = TRUE` VIOLATED on a reachable
//!       successor (`r.b = FALSE`) ⇒ NOT CERTIFIED (the kernel cannot prove the false `R ⊆ Safety` leg).
//!   (c) SOUND POSITIVE: a record composing an Int field AND a Bool field with a TRUE invariant
//!       (`r.active ∈ BOOLEAN ∧ r.count ≤ 2`, and the record-set form `r ∈ [active: BOOLEAN, count: 0..2]`)
//!       ⇒ KERNEL-CERTIFIED, and the full Leg-E re-check (`verify_safety_certificate`) ACCEPTS.
//!   (d) ADVERSARIAL: ordering on a Bool field (`r.on >= 0`, Bool punned as Int) ⇒ NOT CERTIFIED (the bare
//!       `r.on` value access declines at recognition — a Bool position is equality-only).
//!
//! These call `certify_explicit_state_spec` DIRECTLY (the explicit-state kernel-fixpoint lane the
//! value-type-leaf change lives in), so no other prover lane can mask the behavior.

#![cfg(feature = "clean-cic")]

use tla_check::explicit_fixpoint_cert::{
    certify_explicit_state_spec, verify_explicit_state_cert, CellSort, ColSort,
};
use tla_check::Config;

fn cfg_inv(inv: &str) -> Config {
    Config::parse(&format!("INIT Init\nNEXT Next\nINVARIANT {inv}\n")).expect("config parses")
}

// ─────────────────────────────────────────────────────────────────────────────────────────────────────
// (a) MIXED-KIND ⇒ DECLINE. A record field is Int `1` in the initial state and Bool `TRUE` in a
// successor. The two states encode to DIFFERENT `ColSort`s (all-Int record `cells: []` vs `cells: [Bool]`),
// so the cross-state per-column sort-agreement check fails closed. The distinct TLA values `1` and `TRUE`
// are NEVER collapsed to one digit — the whole column (and the certificate) declines.
// ─────────────────────────────────────────────────────────────────────────────────────────────────────
#[test]
fn mixed_kind_int_then_bool_field_declines() {
    let spec = "---- MODULE MixedKind ----\n\
                EXTENDS Integers\n\
                VARIABLE r\n\
                Init == r = [f |-> 1]\n\
                Next == r' = [f |-> TRUE]\n\
                Safety == r = r\n\
                ====\n";
    assert!(
        certify_explicit_state_spec(spec, &cfg_inv("Safety")).is_none(),
        "a field that is Int 1 then Bool TRUE MIXES kinds ⇒ the column DECLINES (never collapses \
         two distinct TLA values to one digit)"
    );
}

// ─────────────────────────────────────────────────────────────────────────────────────────────────────
// (b) COLLAPSE GUARD load-bearing ⇒ NOT CERTIFIED. A Bool field `b` with the invariant `r.b = TRUE` that
// is VIOLATED on a reachable successor (`r.b = FALSE`). The safety leg `R ⊆ (r.b = TRUE)` is FALSE on that
// state, so the kernel cannot mint the certificate — the Bool encoding does NOT hide the violation.
// ─────────────────────────────────────────────────────────────────────────────────────────────────────
#[test]
fn bool_field_invariant_violated_on_reachable_state_not_certified() {
    let spec = "---- MODULE Collapse ----\n\
                EXTENDS Integers\n\
                VARIABLE r\n\
                Init == r = [b |-> TRUE]\n\
                Next == r' = [b |-> ~r.b]\n\
                Safety == r.b = TRUE\n\
                ====\n";
    // The reachable set is { [b|->TRUE], [b|->FALSE] }; `r.b = TRUE` is false on [b|->FALSE].
    assert!(
        certify_explicit_state_spec(spec, &cfg_inv("Safety")).is_none(),
        "a reachable state violates `r.b = TRUE` ⇒ NOT CERTIFIED (the Bool field encoding is \
         collapse-guarded — the violation is not hidden)"
    );
}

// ─────────────────────────────────────────────────────────────────────────────────────────────────────
// (c) SOUND POSITIVE ⇒ KERNEL-CERTIFIED + Leg-E ACCEPTS. A record composing an Int field (`count`) and a
// Bool field (`active`), with a TRUE invariant that reads the Bool field kind-checked (`active ∈ BOOLEAN`)
// and the Int field as an ordinary digit (`count ≤ 2`).
// ─────────────────────────────────────────────────────────────────────────────────────────────────────
const POSITIVE_SPEC: &str = "---- MODULE BoolIntRec ----\n\
    EXTENDS Integers\n\
    VARIABLE r\n\
    Init == r = [active |-> TRUE, count |-> 0]\n\
    Next == r' = [active |-> ~r.active, count |-> IF r.count = 2 THEN 0 ELSE r.count + 1]\n\
    Inv == r.active \\in BOOLEAN /\\ r.count <= 2\n\
    TypeOK == r \\in [active : BOOLEAN, count : 0..2]\n\
    ====\n";

#[test]
fn bool_and_int_fields_compose_and_certify() {
    let cert = certify_explicit_state_spec(POSITIVE_SPEC, &cfg_inv("Inv"))
        .expect("a record with a Bool field + an Int field certifies (value-type leaf)");
    // The record column's per-position kinds: position 0 = `active` (Bool), position 1 = `count` (Int),
    // in CANONICAL sorted-by-name field order ("active" < "count").
    assert_eq!(
        cert.sorts,
        vec![ColSort::Record {
            base: 10,
            fields: vec!["active".to_string(), "count".to_string()],
            cells: vec![CellSort::Bool, CellSort::Int],
        }],
        "the record carries a per-position KIND vector [Bool, Int] (active is Bool, count is Int)"
    );
    // Reachable packs: active is the units digit (Bool 0|1), count the tens digit (Int 0..2).
    let mut r = cert.reachable.clone();
    r.sort_unstable();
    assert_eq!(
        r,
        vec![vec![0], vec![1], vec![10], vec![11], vec![20], vec![21]],
        "6 packed states"
    );
    assert!(
        cert.safety_pred.is_some(),
        "the mixed Bool/Int invariant rides the general safety leg"
    );
    assert!(
        cert.safety_general.is_some(),
        "kernel-checked ⋀_{{s∈R}} Inv(s) leg present"
    );
    assert!(
        verify_explicit_state_cert(&cert),
        "the value-type-leaf cert kernel-re-checks"
    );

    // Full Leg-E: build the self-contained SafetyCertificate and re-check it end to end (re-enumerate R,
    // re-derive the per-position kinds byte-identically, kernel-re-check every leg).
    let config = cfg_inv("Inv");
    let sc = tla_check::cert::build_explicit_fixpoint_certificate(POSITIVE_SPEC, &config, cert);
    let report = tla_check::cert::verify_safety_certificate(&sc);
    assert!(
        report.accepted,
        "ty cert-check (Leg-E) ACCEPTS the value-type-leaf cert: {}",
        report.detail
    );
}

#[test]
fn record_set_membership_with_bool_domain_certifies() {
    // `TypeOK == r \in [active : BOOLEAN, count : 0..2]` — the record-set membership form, KIND-CHECKED
    // per field: `active` (Bool) takes the `BOOLEAN` domain, `count` (Int) a literal `0..2` range.
    let cert = certify_explicit_state_spec(POSITIVE_SPEC, &cfg_inv("TypeOK"))
        .expect("record-set membership with a BOOLEAN field domain certifies");
    assert!(
        cert.safety_pred.is_some(),
        "record-set membership rides the general safety leg"
    );
    assert!(
        verify_explicit_state_cert(&cert),
        "the record-set-membership cert kernel-re-checks"
    );
    let config = cfg_inv("TypeOK");
    let sc = tla_check::cert::build_explicit_fixpoint_certificate(POSITIVE_SPEC, &config, cert);
    let report = tla_check::cert::verify_safety_certificate(&sc);
    assert!(
        report.accepted,
        "Leg-E ACCEPTS the record-set-membership cert: {}",
        report.detail
    );
}

// ─────────────────────────────────────────────────────────────────────────────────────────────────────
// (d) ADVERSARIAL ⇒ NOT CERTIFIED. Ordering on a Bool field (`r.on >= 0`, treating the Bool code as an
// Int) must FAIL CLOSED: a bare `r.on` value access over a Bool position DECLINES at recognition (a Bool
// position is equality-only), so the invariant does not recognize and no safety leg is built.
// ─────────────────────────────────────────────────────────────────────────────────────────────────────
#[test]
fn ordering_on_bool_field_not_certified() {
    let spec = "---- MODULE Adversarial ----\n\
                EXTENDS Integers\n\
                VARIABLE r\n\
                Init == r = [on |-> TRUE]\n\
                Next == r' = [on |-> ~r.on]\n\
                Safety == r.on >= 0\n\
                ====\n";
    assert!(
        certify_explicit_state_spec(spec, &cfg_inv("Safety")).is_none(),
        "`r.on >= 0` orders a Bool field (Bool punned as Int) ⇒ the bare access declines ⇒ NOT CERTIFIED"
    );
}

// A companion positive: an ENUM-domain field (`r.mode \in {\"a\",\"b\"}`, String values) is now the LANDED
// value-type-leaf Enum position — the record's String field encodes to the per-position label index, and the
// membership invariant certifies (see compound_enum_cell_cert.rs for the full controlled soundness proof).
#[test]
fn string_field_position_now_certifies() {
    let spec = "---- MODULE StrField ----\n\
                EXTENDS Integers\n\
                VARIABLE r\n\
                Init == r = [mode |-> \"a\"]\n\
                Next == r' = [mode |-> IF r.mode = \"a\" THEN \"b\" ELSE \"a\"]\n\
                Safety == r.mode \\in {\"a\", \"b\"}\n\
                ====\n";
    let cert = certify_explicit_state_spec(spec, &cfg_inv("Safety"))
        .expect("a String-valued record field now certifies (value-type-leaf Enum position)");
    assert_eq!(
        cert.sorts,
        vec![ColSort::Record {
            base: 10,
            fields: vec!["mode".to_string()],
            cells: vec![CellSort::Enum {
                labels: vec!["a".to_string(), "b".to_string()],
                kind: tla_check::explicit_fixpoint_cert::EnumKind::Str,
            }],
        }],
        "the record's `mode` position is a per-position Str Enum over the sorted label union"
    );
    assert!(
        verify_explicit_state_cert(&cert),
        "the compound-enum cert kernel-re-checks"
    );
}
