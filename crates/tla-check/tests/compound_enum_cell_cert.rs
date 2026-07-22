// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! COMPOUND-ENUM value-type leaf — the controlled SOUNDNESS PROOF for a `String` / model-value ENUM
//! position inside a compound (`Record` field / Int-domain `Func` value). Each positional-pack digit may
//! now carry `CellSort::Enum { labels, kind }`: the pack stays the uniform base-`b` numeral `Σ code_i·b^i`,
//! but an enum position's digit is the INDEX of its `String`/model-value LABEL in that POSITION's sorted
//! cross-state label UNION — the per-position analogue of the scalar `ColSort::Enum`, with a fail-closed
//! kind discriminant.
//!
//! THE SOUNDNESS REQUIREMENTS under test (no two distinct TLA values ever share a (position, code)):
//!   (a) MIXED-KIND: a field Int then String, AND a field String "x" then model value `x` (text-equal but
//!       DISTINCT TLA values) ⇒ DECLINES (the per-position KIND DISCRIMINANT makes the states different
//!       sorts / fails the kind guard — never collapses).
//!   (b) COLLAPSE GUARD load-bearing: a String field with an invariant `r.mode = "a"` VIOLATED on a
//!       reachable successor (`r.mode = "b"`) ⇒ NOT CERTIFIED (distinct labels ⇒ distinct codes ⇒ the
//!       violation is visible; the kernel cannot prove the false `R ⊆ Safety` leg).
//!   (c) SOUND POSITIVE: a record composing a String-Enum field AND an Int field, TRUE invariant
//!       (`r.mode \in {"a","b"} /\ r.n <= 2`) ⇒ KERNEL-CERTIFIED, `cells` carries the Enum labels/kind, and
//!       the full Leg-E re-check (`verify_safety_certificate`) ACCEPTS (re-derives the per-position labels).
//!   (d) ADVERSARIAL: ordering / arithmetic on an enum field (`r.mode >= "a"`, `r.mode + 0 = 0`) ⇒ NOT
//!       CERTIFIED (a bare enum-position value access declines — equality-only).
//!   (e) LABEL-UNION: a field "a" in state 1 and "b" in state 2 (union {"a","b"}) with a TRUE membership
//!       invariant ⇒ CERTIFIES (both states index the SHARED union), and Leg-E ACCEPTS (rebuilds the
//!       identical union).
//!
//! These call `certify_explicit_state_spec` DIRECTLY (the explicit-state kernel-fixpoint lane the leaf
//! lives in), so no other prover lane can mask the behavior.

#![cfg(feature = "clean-cic")]

use tla_check::explicit_fixpoint_cert::{
    certify_explicit_state_spec, verify_explicit_state_cert, CellSort, ColSort, EnumKind,
};
use tla_check::{Config, ConstantValue};

fn cfg_inv(inv: &str) -> Config {
    Config::parse(&format!("INIT Init\nNEXT Next\nINVARIANT {inv}\n")).expect("config parses")
}

fn str_enum(labels: &[&str]) -> CellSort {
    CellSort::Enum {
        labels: labels.iter().map(|s| s.to_string()).collect(),
        kind: EnumKind::Str,
    }
}

// ─────────────────────────────────────────────────────────────────────────────────────────────────────
// (a) MIXED-KIND ⇒ DECLINE.
// ─────────────────────────────────────────────────────────────────────────────────────────────────────

// (a.1) Int then String: the two states encode the field to DIFFERENT `CellSort`s (Int cell vs Enum cell),
// so the cross-state per-column sort-agreement check fails closed. Int `1` and String "x" are never
// collapsed to one digit.
#[test]
fn mixed_kind_int_then_string_field_declines() {
    let spec = "---- MODULE MixedIS ----\n\
                EXTENDS Integers\n\
                VARIABLE r\n\
                Init == r = [f |-> 1]\n\
                Next == r' = [f |-> \"x\"]\n\
                Safety == r = r\n\
                ====\n";
    assert!(
        certify_explicit_state_spec(spec, &cfg_inv("Safety")).is_none(),
        "a field that is Int 1 then String \"x\" MIXES kinds ⇒ DECLINES (never collapses)"
    );
}

// (a.2) String "x" then MODEL VALUE `x` (SAME text, DISTINCT TLA values): the per-position kind guard
// rejects the mix (a `Str` position observing a `Model` label fails closed), so the text-equal String and
// model value NEVER share an index. This is the sharp kind-discriminant test.
#[test]
fn mixed_kind_string_then_textequal_model_value_declines() {
    let spec = "---- MODULE MixedSM ----\n\
                EXTENDS Integers\n\
                CONSTANT x\n\
                VARIABLE r\n\
                Init == r = [f |-> \"x\"]\n\
                Next == r' = [f |-> x]\n\
                Safety == r = r\n\
                ====\n";
    let mut config = cfg_inv("Safety");
    config
        .constants
        .insert("x".to_string(), ConstantValue::ModelValue);
    assert!(
        certify_explicit_state_spec(spec, &config).is_none(),
        "a String \"x\" then a text-equal model value x are DISTINCT TLA values ⇒ the kind discriminant \
         DECLINES (they never share a (position, code))"
    );
}

// ─────────────────────────────────────────────────────────────────────────────────────────────────────
// (b) COLLAPSE GUARD load-bearing ⇒ NOT CERTIFIED. A String field `mode` with the invariant `r.mode = "a"`
// that is VIOLATED on a reachable successor (`r.mode = "b"`). Distinct labels ⇒ distinct codes ⇒ the "b"
// state is present in R with a distinct pack; the safety leg `R ⊆ (r.mode = "a")` is FALSE on it, so the
// kernel cannot mint the certificate — the enum encoding does NOT hide the violation.
// ─────────────────────────────────────────────────────────────────────────────────────────────────────
#[test]
fn string_field_invariant_violated_on_reachable_state_not_certified() {
    let spec = "---- MODULE CollapseEnum ----\n\
                EXTENDS Integers\n\
                VARIABLE r\n\
                Init == r = [mode |-> \"a\"]\n\
                Next == r' = [mode |-> IF r.mode = \"a\" THEN \"b\" ELSE \"a\"]\n\
                Safety == r.mode = \"a\"\n\
                ====\n";
    // Reachable = { [mode|->"a"], [mode|->"b"] }; `r.mode = "a"` is false on [mode|->"b"].
    assert!(
        certify_explicit_state_spec(spec, &cfg_inv("Safety")).is_none(),
        "a reachable state violates `r.mode = \"a\"` ⇒ NOT CERTIFIED (distinct labels ⇒ distinct codes ⇒ \
         the violation is visible, not hidden)"
    );
}

// ─────────────────────────────────────────────────────────────────────────────────────────────────────
// (c) SOUND POSITIVE ⇒ KERNEL-CERTIFIED + Leg-E ACCEPTS. A record composing a String-Enum field (`mode`)
// and an Int field (`n`), with a TRUE invariant that reads the enum field kind-checked (`mode ∈ {"a","b"}`)
// and the Int field as an ordinary digit (`n ≤ 2`).
// ─────────────────────────────────────────────────────────────────────────────────────────────────────
const POSITIVE_SPEC: &str = "---- MODULE ModeNRec ----\n\
    EXTENDS Integers\n\
    VARIABLE r\n\
    Init == r = [mode |-> \"a\", n |-> 0]\n\
    Next == r' = [mode |-> IF r.mode = \"a\" THEN \"b\" ELSE \"a\", n |-> IF r.n = 2 THEN 0 ELSE r.n + 1]\n\
    Inv == r.mode \\in {\"a\", \"b\"} /\\ r.n <= 2\n\
    ModeEq == r.mode = \"a\" \\/ r.mode = \"b\"\n\
    ====\n";

#[test]
fn string_enum_and_int_fields_compose_and_certify() {
    let cert = certify_explicit_state_spec(POSITIVE_SPEC, &cfg_inv("Inv"))
        .expect("a record with a String-Enum field + an Int field certifies (value-type leaf)");
    // Fields sorted by name: position 0 = `mode` (Str Enum over {"a","b"}), position 1 = `n` (Int).
    assert_eq!(
        cert.sorts,
        vec![ColSort::Record {
            base: 10,
            fields: vec!["mode".to_string(), "n".to_string()],
            cells: vec![str_enum(&["a", "b"]), CellSort::Int],
        }],
        "the record carries a per-position KIND vector [Enum(Str, a|b), Int] — the serialized `cells` \
         carries the Enum labels/kind"
    );
    // Reachable packs: mode is the units digit (label idx 0|1), n the tens digit (Int 0..2).
    let mut r = cert.reachable.clone();
    r.sort_unstable();
    assert_eq!(
        r,
        vec![vec![0], vec![1], vec![10], vec![11], vec![20], vec![21]],
        "6 packed states"
    );
    assert!(
        cert.safety_pred.is_some(),
        "the mixed Enum/Int invariant rides the general safety leg"
    );
    assert!(
        cert.safety_general.is_some(),
        "kernel-checked ⋀_{{s∈R}} Inv(s) leg present"
    );
    assert!(
        verify_explicit_state_cert(&cert),
        "the compound-enum cert kernel-re-checks"
    );

    // Full Leg-E: build the self-contained SafetyCertificate and re-check it end to end (re-enumerate R,
    // re-derive the per-position labels+kind byte-identically, kernel-re-check every leg).
    let config = cfg_inv("Inv");
    let sc = tla_check::cert::build_explicit_fixpoint_certificate(POSITIVE_SPEC, &config, cert);
    let report = tla_check::cert::verify_safety_certificate(&sc);
    assert!(
        report.accepted,
        "ty cert-check (Leg-E) ACCEPTS the compound-enum cert: {}",
        report.detail
    );
}

#[test]
fn string_enum_equality_invariant_certifies_and_lege_accepts() {
    // `ModeEq == r.mode = "a" \/ r.mode = "b"` — the enum-position EQUALITY form (disjunction of
    // label-index equalities), kind-checked (String literal ⇒ Str position).
    let cert = certify_explicit_state_spec(POSITIVE_SPEC, &cfg_inv("ModeEq"))
        .expect("enum-position equality invariant certifies");
    assert!(
        cert.safety_pred.is_some(),
        "enum-position equality rides the general safety leg"
    );
    assert!(verify_explicit_state_cert(&cert), "kernel re-check passes");
    let config = cfg_inv("ModeEq");
    let sc = tla_check::cert::build_explicit_fixpoint_certificate(POSITIVE_SPEC, &config, cert);
    let report = tla_check::cert::verify_safety_certificate(&sc);
    assert!(
        report.accepted,
        "Leg-E ACCEPTS the enum-position equality cert: {}",
        report.detail
    );
}

// Record-set membership `r \in [mode: {"a","b"}, n: 0..2]` — KIND-CHECKED per field: `mode` (Str Enum)
// takes a String set, `n` (Int) a literal `0..2` range.
#[test]
fn record_set_membership_with_string_enum_domain_certifies() {
    let spec = "---- MODULE ModeNSet ----\n\
        EXTENDS Integers\n\
        VARIABLE r\n\
        Init == r = [mode |-> \"a\", n |-> 0]\n\
        Next == r' = [mode |-> IF r.mode = \"a\" THEN \"b\" ELSE \"a\", n |-> IF r.n = 2 THEN 0 ELSE r.n + 1]\n\
        TypeOK == r \\in [mode : {\"a\", \"b\"}, n : 0..2]\n\
        ====\n";
    let cert = certify_explicit_state_spec(spec, &cfg_inv("TypeOK"))
        .expect("record-set membership with a String-Enum field domain certifies");
    assert!(
        cert.safety_pred.is_some(),
        "record-set membership rides the general safety leg"
    );
    assert!(
        verify_explicit_state_cert(&cert),
        "the record-set-membership cert kernel-re-checks"
    );
    let config = cfg_inv("TypeOK");
    let sc = tla_check::cert::build_explicit_fixpoint_certificate(spec, &config, cert);
    let report = tla_check::cert::verify_safety_certificate(&sc);
    assert!(
        report.accepted,
        "Leg-E ACCEPTS the record-set-membership cert: {}",
        report.detail
    );
}

// ─────────────────────────────────────────────────────────────────────────────────────────────────────
// (d) ADVERSARIAL ⇒ NOT CERTIFIED. Ordering / arithmetic on an enum field must FAIL CLOSED: a bare enum
// -position value access DECLINES at recognition (equality-only), so the invariant does not recognize.
// ─────────────────────────────────────────────────────────────────────────────────────────────────────
#[test]
fn ordering_on_enum_field_not_certified() {
    let spec = "---- MODULE OrdEnum ----\n\
                EXTENDS Integers\n\
                VARIABLE r\n\
                Init == r = [mode |-> \"a\"]\n\
                Next == r' = [mode |-> IF r.mode = \"a\" THEN \"b\" ELSE \"a\"]\n\
                Safety == r.mode >= \"a\"\n\
                ====\n";
    assert!(
        certify_explicit_state_spec(spec, &cfg_inv("Safety")).is_none(),
        "`r.mode >= \"a\"` orders an enum field (index has no order) ⇒ the bare access declines ⇒ NOT CERTIFIED"
    );
}

#[test]
fn arithmetic_on_enum_field_not_certified() {
    let spec = "---- MODULE ArithEnum ----\n\
                EXTENDS Integers\n\
                VARIABLE r\n\
                Init == r = [mode |-> \"a\"]\n\
                Next == r' = [mode |-> IF r.mode = \"a\" THEN \"b\" ELSE \"a\"]\n\
                Safety == r.mode + 0 = 0\n\
                ====\n";
    assert!(
        certify_explicit_state_spec(spec, &cfg_inv("Safety")).is_none(),
        "arithmetic on an enum field (index punned as Nat) ⇒ the bare access declines ⇒ NOT CERTIFIED"
    );
}

// ─────────────────────────────────────────────────────────────────────────────────────────────────────
// (e) LABEL-UNION ⇒ CERTIFIES + Leg-E ACCEPTS. A field "a" in state 1 and "b" in state 2: the position's
// label union is {"a","b"}, both states index the SAME sorted union, and the TRUE membership invariant
// `r.mode \in {"a","b"}` certifies. Leg-E re-enumerates and rebuilds the IDENTICAL union.
// ─────────────────────────────────────────────────────────────────────────────────────────────────────
#[test]
fn label_union_across_states_certifies_and_lege_accepts() {
    let spec = "---- MODULE UnionEnum ----\n\
                EXTENDS Integers\n\
                VARIABLE r\n\
                Init == r = [mode |-> \"a\"]\n\
                Next == r' = [mode |-> IF r.mode = \"a\" THEN \"b\" ELSE \"a\"]\n\
                Safety == r.mode \\in {\"a\", \"b\"}\n\
                ====\n";
    let config = cfg_inv("Safety");
    let cert =
        certify_explicit_state_spec(spec, &config).expect("label-union membership certifies");
    assert_eq!(
        cert.sorts,
        vec![ColSort::Record {
            base: 10,
            fields: vec!["mode".to_string()],
            cells: vec![str_enum(&["a", "b"])],
        }],
        "the `mode` position's Enum labels are the SORTED cross-state UNION of a and b"
    );
    // Both states index the shared union: "a"→0, "b"→1.
    let mut r = cert.reachable.clone();
    r.sort_unstable();
    assert_eq!(
        r,
        vec![vec![0], vec![1]],
        "both states index the shared union"
    );
    assert!(verify_explicit_state_cert(&cert), "kernel re-check passes");
    // Full Leg-E: re-enumerate + rebuild the identical union + kernel-re-check.
    let sc = tla_check::cert::build_explicit_fixpoint_certificate(spec, &config, cert);
    let report = tla_check::cert::verify_safety_certificate(&sc);
    assert!(
        report.accepted,
        "Leg-E ACCEPTS the label-union cert (rebuilds the identical union): {}",
        report.detail
    );
}

// MODEL-VALUE positive: a record field holding a config CONSTANT model value, with a `r.f \in Data`
// membership invariant (Data a model-value CONSTANT set) ⇒ the Model-kind compound-enum position, resolved
// through `mvsets` exactly as the scalar lane. Exercises the Model kind + the mvsets threading in the
// compound recognizer.
#[test]
fn model_value_field_membership_certifies() {
    let spec = "---- MODULE MVField ----\n\
                EXTENDS Integers\n\
                CONSTANTS MV, Data\n\
                VARIABLE r\n\
                Init == r = [f |-> MV]\n\
                Next == r' = r\n\
                Safety == r.f \\in Data\n\
                ====\n";
    let mut config = cfg_inv("Safety");
    config
        .constants
        .insert("MV".to_string(), ConstantValue::ModelValue);
    config.constants.insert(
        "Data".to_string(),
        ConstantValue::ModelValueSet(vec!["MV".to_string()]),
    );
    let cert = certify_explicit_state_spec(spec, &config)
        .expect("a model-value record field with a `\\in Data` invariant certifies");
    assert_eq!(
        cert.sorts,
        vec![ColSort::Record {
            base: 10,
            fields: vec!["f".to_string()],
            cells: vec![CellSort::Enum {
                labels: vec!["MV".to_string()],
                kind: EnumKind::Model
            }],
        }],
        "the `f` position is a per-position MODEL Enum over the observed model-value union"
    );
    assert!(
        verify_explicit_state_cert(&cert),
        "the model-value compound-enum cert kernel-re-checks"
    );
}

// FUNC enum position: an Int-domain FUNCTION with a MIXED enum/Int value list (`[0 |-> "a", 1 |-> 7]`) —
// NOT a uniform-label FuncEnum, so it rides the per-position compound-Func-enum path. Position 0 is a Str
// Enum cell, position 1 an Int cell. `f[0] = "a"` (enum-position digit) and `f[1] = 7` (Int-position digit)
// both certify + Leg-E accepts (record_digit_exact now admits the Func digit shape symmetrically to Record).
#[test]
fn func_mixed_enum_and_int_position_certifies() {
    let spec = "---- MODULE FuncMix ----\n\
                EXTENDS Integers\n\
                VARIABLE f\n\
                Init == f = [i \\in 0..1 |-> IF i = 0 THEN \"a\" ELSE 7]\n\
                Next == f' = f\n\
                Safety == f[0] = \"a\" /\\ f[1] = 7\n\
                ====\n";
    let config = cfg_inv("Safety");
    let cert = certify_explicit_state_spec(spec, &config)
        .expect("an Int-domain Func with a mixed enum/Int value list certifies");
    assert_eq!(
        cert.sorts,
        vec![ColSort::Func {
            base: 10,
            arity: 2,
            cells: vec![str_enum(&["a"]), CellSort::Int],
            dom: vec![],
            dom_kind: EnumKind::Model,
        }],
        "the Func carries per-position cells [Enum(Str, a), Int]"
    );
    assert!(
        verify_explicit_state_cert(&cert),
        "the mixed-Func-enum cert kernel-re-checks"
    );
    let sc = tla_check::cert::build_explicit_fixpoint_certificate(spec, &config, cert);
    let report = tla_check::cert::verify_safety_certificate(&sc);
    assert!(
        report.accepted,
        "Leg-E ACCEPTS the mixed-Func-enum cert: {}",
        report.detail
    );
}

// TAMPER: a mutated per-position enum label in the stored sort fails the Leg-E spec re-derivation binding
// (verify re-enumerates and requires `re.sorts == fp.sorts`); the genuine cert is accepted through the same
// path (control).
#[test]
fn tampered_compound_enum_label_rejects() {
    let spec = "---- MODULE TamperEnum ----\n\
                EXTENDS Integers\n\
                VARIABLE r\n\
                Init == r = [mode |-> \"a\"]\n\
                Next == r' = [mode |-> IF r.mode = \"a\" THEN \"b\" ELSE \"a\"]\n\
                Safety == r.mode \\in {\"a\", \"b\"}\n\
                ====\n";
    let config = cfg_inv("Safety");
    let cert = certify_explicit_state_spec(spec, &config).expect("compound-enum cert mints");

    // Kernel-level tamper: push a reachable index out of the label range ⇒ kernel re-check fails.
    let mut kernel_tampered = cert.clone();
    kernel_tampered.reachable[0][0] += 10;
    assert!(
        !verify_explicit_state_cert(&kernel_tampered),
        "a tampered enum index must fail the kernel re-check"
    );

    // Label-set tamper: rename a per-position label. Kernel legs are index-blind Nat equalities, so the
    // binding gate is Leg-E: re-derivation re-collects the real labels ⇒ `re.sorts != fp.sorts` ⇒ REJECTED.
    let mut label_tampered = cert.clone();
    if let ColSort::Record { cells, .. } = &mut label_tampered.sorts[0] {
        if let CellSort::Enum { labels, .. } = &mut cells[0] {
            labels[0] = "zzz".to_string();
        } else {
            panic!("position 0 must be an Enum cell");
        }
    } else {
        panic!("column 0 must be a Record sort");
    }
    let sc = tla_check::cert::build_explicit_fixpoint_certificate(spec, &config, label_tampered);
    let report = tla_check::cert::verify_safety_certificate(&sc);
    assert!(
        !report.accepted,
        "a cert with a mutated per-position enum label must be rejected: {}",
        report.detail
    );

    // Control: the genuine cert is accepted through the same full Leg-E path.
    let sc_ok = tla_check::cert::build_explicit_fixpoint_certificate(spec, &config, cert);
    let ok_report = tla_check::cert::verify_safety_certificate(&sc_ok);
    assert!(
        ok_report.accepted,
        "the genuine compound-enum cert must be accepted: {}",
        ok_report.detail
    );
}
