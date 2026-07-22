// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! FUNCTION-of-ENUM cell encoding for the explicit-state certificate lane (the finite-function
//! analogue of `enum_cell_cert.rs`):
//!
//!   * a state variable holding a FUNCTION `[d ∈ 0..arity-1 |-> label]` (a per-process program
//!     counter `pc: [0..K -> {"a","b","Done"}]`) encodes POSITIONALLY into ONE `Nat` cell —
//!     `pack = Σ_d idx(e_d)·|labels|^d` — baked into `ColSort::FuncEnum{arity, labels}` (the SORTED
//!     observed-value union, exactly like a scalar `ColSort::Enum`);
//!   * an EQUALITY invariant over a func-enum APPLICATION at a LITERAL index (`pc[1] = "a"`,
//!     `pc[i] \in {…}`, `pc[i] = pc[j]`) certifies + kernel-re-checks + full-Leg-E-accepts;
//!   * the encoding is truth-EXACT for equality only — an ordering / non-literal / out-of-range
//!     `[i]` access fails closed (never a false certificate);
//!   * digest back-compat: adding the `ColSort::FuncEnum` serde variant leaves every non-func-enum
//!     fixture cert BYTE-IDENTICAL (the scalar-enum / record / interval suites stay green);
//!   * tamper: a mutated pack cell / label set / arity fails verify.

#![cfg(feature = "clean-cic")]

use tla_check::explicit_fixpoint_cert::{
    certify_explicit_state_spec, verify_explicit_state_cert, ColSort, EnumKind,
};
use tla_check::{Config, ConstantValue};

fn cfg() -> Config {
    Config::parse("INIT Init\nNEXT Next\nINVARIANT Safety\n").expect("config parses")
}

/// A config with `INVARIANT <inv>` and a two-member model-value domain `D = {d1, d2}` — the shape a
/// `[D -> …]` function-domain / quantifier-over-function spec needs.
fn cfg_dom(inv: &str) -> Config {
    let mut c = Config::parse(&format!("INIT Init\nNEXT Next\nINVARIANT {inv}\n")).expect("parses");
    c.constants.insert(
        "D".to_string(),
        ConstantValue::ModelValueSet(vec!["d1".to_string(), "d2".to_string()]),
    );
    c
}

/// A three-slot program counter `pc: [0..2 -> {"a","b"}]` whose slot 0 advances `"a" -> "b"`, with a
/// `pc[1] = "a"` invariant. The column's sort is `FuncEnum{arity:3, labels:["a","b"]}` (base `|labels|=2`),
/// the reachable set is the two positional packs, and the cert kernel-re-verifies + round-trips.
#[test]
fn func_enum_pc_certifies_and_verifies() {
    let spec = "---- MODULE PcFunc ----\n\
                EXTENDS Integers\n\
                VARIABLE pc\n\
                Init == pc = [i \\in 0..2 |-> \"a\"]\n\
                Next == pc' = [pc EXCEPT ![0] = \"b\"]\n\
                Safety == pc[1] = \"a\"\n\
                ====\n";
    let cert = certify_explicit_state_spec(spec, &cfg()).expect("func-enum cert mints");
    assert_eq!(
        cert.sorts,
        vec![ColSort::FuncEnum {
            arity: 3,
            labels: vec!["a".to_string(), "b".to_string()],
            dom: vec![],
            dom_kind: EnumKind::Model,
        }],
        "the column's sort is the func-enum encoding (arity + sorted observed-label union)"
    );
    // R = the two positional packs: [a,a,a]=0, [b,a,a]=idx(b)·2^0=1.
    assert_eq!(
        cert.reachable,
        vec![vec![0], vec![1]],
        "R is the two positional packs"
    );
    // The invariant rides the general safety leg (no Int column ⇒ no nonneg tuple leg).
    assert!(
        cert.safety_pred.is_some(),
        "func-enum equality invariant rides the general safety leg"
    );
    assert!(
        cert.safety_general.is_some(),
        "kernel-checked ⋀_{{s∈R}} Safety(s) leg"
    );
    assert!(verify_explicit_state_cert(&cert), "kernel re-check passes");

    // Serde round-trip: the new FuncEnum sort survives and still verifies.
    let bytes = serde_json::to_vec(&cert).expect("serialize");
    let back: tla_check::explicit_fixpoint_cert::ExplicitFixpointCert =
        serde_json::from_slice(&bytes).expect("deserialize");
    assert_eq!(cert, back, "func-enum cert round-trips");
    assert!(
        verify_explicit_state_cert(&back),
        "the re-loaded func-enum cert verifies"
    );
}

/// The `pc[i] \in {…}` membership form and the full Leg-E re-check path (spec re-derivation binding the
/// stored FuncEnum sort + safety IR to the spec).
#[test]
fn func_enum_set_membership_full_lege() {
    let spec = "---- MODULE PcMem ----\n\
                EXTENDS Integers\n\
                VARIABLE pc\n\
                Init == pc = [i \\in 0..1 |-> \"read\"]\n\
                Next == pc' = [pc EXCEPT ![0] = \"write\"]\n\
                Safety == pc[1] \\in {\"read\", \"write\"}\n\
                ====\n";
    let config = cfg();
    let cert = certify_explicit_state_spec(spec, &config).expect("func-enum membership cert mints");
    assert_eq!(
        cert.sorts,
        vec![ColSort::FuncEnum {
            arity: 2,
            labels: vec!["read".to_string(), "write".to_string()],
            dom: vec![],
            dom_kind: EnumKind::Model,
        }],
    );
    assert!(cert.safety_pred.is_some());
    assert!(verify_explicit_state_cert(&cert), "kernel re-check passes");
    // Full Leg-E: build a SafetyCertificate and re-verify end-to-end (re-enumerates the spec,
    // re-derives the FuncEnum sort + safety IR, and binds them to the cert).
    let sc = tla_check::cert::build_explicit_fixpoint_certificate(spec, &config, cert);
    let report = tla_check::cert::verify_safety_certificate(&sc);
    assert!(
        report.accepted,
        "full Leg-E re-check accepts: {}",
        report.detail
    );
}

/// TAMPER: (a) a mutated pack cell that violates the invariant fails the kernel re-check; (b) a mutated
/// label set and (c) a mutated arity fail the Leg-E spec re-derivation binding (verify re-enumerates and
/// requires `re.sorts == fp.sorts`).
#[test]
fn tampered_func_enum_cert_rejects() {
    let spec = "---- MODULE PcTamper ----\n\
                EXTENDS Integers\n\
                VARIABLE pc\n\
                Init == pc = [i \\in 0..2 |-> \"a\"]\n\
                Next == pc' = [pc EXCEPT ![0] = \"b\"]\n\
                Safety == pc[1] = \"a\"\n\
                ====\n";
    let config = cfg();
    let cert = certify_explicit_state_spec(spec, &config).expect("func-enum cert mints");

    // (a) kernel-level tamper: push a reachable pack so its slot-1 digit is no longer "a" — the
    // safety_general leg reduces to Bool.false and the kernel rejects.
    let mut kernel_tampered = cert.clone();
    kernel_tampered.reachable[0][0] += 10; // pack 10 ⇒ pc[1] = (10/2) mod 2 = 1 ≠ idx("a")=0
    assert!(
        !verify_explicit_state_cert(&kernel_tampered),
        "a tampered func-enum pack must fail the kernel re-check"
    );

    // (b) label-set tamper: rename a label in the stored sort. The kernel legs are index-blind Nat
    // equalities, so the binding gate is Leg-E: re-derivation re-collects the real labels and the
    // sorts no longer match ⇒ REJECTED.
    let mut label_tampered = cert.clone();
    if let ColSort::FuncEnum { labels, .. } = &mut label_tampered.sorts[0] {
        labels[0] = "zzz".to_string();
    } else {
        panic!("column 0 must be a FuncEnum sort");
    }
    let sc = tla_check::cert::build_explicit_fixpoint_certificate(spec, &config, label_tampered);
    let report = tla_check::cert::verify_safety_certificate(&sc);
    assert!(
        !report.accepted,
        "a cert with a mutated func-enum label must be rejected: {}",
        report.detail
    );

    // (c) arity tamper: shrink the stored arity. Leg-E re-derivation gets the real arity ⇒ mismatch.
    let mut arity_tampered = cert.clone();
    if let ColSort::FuncEnum { arity, .. } = &mut arity_tampered.sorts[0] {
        *arity = 2;
    } else {
        panic!("column 0 must be a FuncEnum sort");
    }
    let sc_a = tla_check::cert::build_explicit_fixpoint_certificate(spec, &config, arity_tampered);
    let report_a = tla_check::cert::verify_safety_certificate(&sc_a);
    assert!(
        !report_a.accepted,
        "a cert with a mutated func-enum arity must be rejected: {}",
        report_a.detail
    );

    // Control: the genuine cert is accepted through the same full Leg-E path.
    let sc_ok = tla_check::cert::build_explicit_fixpoint_certificate(spec, &config, cert);
    let ok_report = tla_check::cert::verify_safety_certificate(&sc_ok);
    assert!(
        ok_report.accepted,
        "the genuine func-enum cert must be accepted: {}",
        ok_report.detail
    );
}

/// FAIL-CLOSED at the surface `[i]`-access recognizer: an OUT-OF-RANGE literal index, a NON-LITERAL
/// index, or an ORDERING over a func-enum application is not in the recognized fragment, so the invariant
/// does not recognize and NO certificate is minted (the honest tiers stay authoritative).
#[test]
fn func_enum_out_of_fragment_index_declines() {
    let base = "---- MODULE PcDecline ----\n\
                EXTENDS Integers\n\
                VARIABLE pc\n\
                Init == pc = [i \\in 0..2 |-> \"a\"]\n\
                Next == pc' = [pc EXCEPT ![0] = \"b\"]\n";
    // OUT-OF-RANGE index (arity 3, index 9) ⇒ decline.
    let oor = format!("{base}Safety == pc[9] = \"a\"\n====\n");
    assert!(
        certify_explicit_state_spec(&oor, &cfg()).is_none(),
        "an out-of-range func-enum index must fail closed (no cert)"
    );
    // NON-LITERAL index (`1 + 0` is an Add, not a literal) ⇒ decline.
    let nonlit = format!("{base}Safety == pc[1 + 0] = \"a\"\n====\n");
    assert!(
        certify_explicit_state_spec(&nonlit, &cfg()).is_none(),
        "a non-literal func-enum index must fail closed (no cert)"
    );
    // ORDERING over a func-enum application (meaningless on an enum index) ⇒ decline.
    let ord = format!("{base}Safety == pc[1] < pc[2]\n====\n");
    assert!(
        certify_explicit_state_spec(&ord, &cfg()).is_none(),
        "an ordering over func-enum applications must fail closed (no cert)"
    );
}

/// QUANTIFIER-OVER-FUNCTION-VALUES, TRUE: `\A d \in D : st[d] = "a"` over a MODEL-VALUE function domain
/// `D = {d1,d2}` folds to `⋀_{d∈D} digit_d = idx("a")`. The column is a model-value-domain `FuncEnum`
/// (`dom` carries the sorted model-value names), the fold rides the kernel safety leg, and Leg-E re-derives.
#[test]
fn func_enum_model_domain_forall_true_certifies() {
    let spec = "---- MODULE QFoldGood ----\n\
                EXTENDS Integers\n\
                CONSTANT D\n\
                VARIABLE st\n\
                Init == st = [d \\in D |-> \"a\"]\n\
                Next == st' = st\n\
                Inv == \\A d \\in D : st[d] = \"a\"\n\
                ====\n";
    let cert = certify_explicit_state_spec(spec, &cfg_dom("Inv"))
        .expect("a TRUE ∀-over-model-value-domain-function invariant certifies");
    assert!(
        matches!(&cert.sorts[..], [ColSort::FuncEnum { dom, labels, .. }]
            if !dom.is_empty() && labels.contains(&"a".to_string())),
        "the column is a model-value-domain FuncEnum (non-empty dom): {:?}",
        cert.sorts
    );
    assert!(verify_explicit_state_cert(&cert), "kernel re-check passes");
    let sc = tla_check::cert::build_explicit_fixpoint_certificate(spec, &cfg_dom("Inv"), cert);
    assert!(
        tla_check::cert::verify_safety_certificate(&sc).accepted,
        "full Leg-E re-derives the model-value domain + fold and ACCEPTS"
    );
}

/// COLLAPSE GUARD for the ∀-fold: `\A d \in D : st[d] = "a"` is VIOLATED once some `st[d]` flips to "b".
/// The fold `⋀_{d∈D} digit_d = idx("a")` is FALSE on that reachable state, so the kernel's R⊆Safety leg
/// cannot mint the cert — the quantifier fold does NOT hide the violation (cf. the certifying twin above,
/// proving it is the kernel rejecting, not a recognizer refusal). An adversarial ORDERING twin
/// `\A d \in D : st[d] >= "a"` also declines (an enum index carries no order ⇒ fails closed).
#[test]
fn func_enum_model_domain_forall_violation_declines() {
    let spec = "---- MODULE QFoldBad ----\n\
                EXTENDS Integers\n\
                CONSTANT D\n\
                VARIABLE st\n\
                Init == st = [d \\in D |-> \"a\"]\n\
                Next == \\E d \\in D : st' = [st EXCEPT ![d] = \"b\"]\n\
                Inv == \\A d \\in D : st[d] = \"a\"\n\
                ====\n";
    assert!(
        certify_explicit_state_spec(spec, &cfg_dom("Inv")).is_none(),
        "a reachable state violates the ∀-over-function fold ⇒ NOT CERTIFIED (the fold is not hidden)"
    );

    // Adversarial ORDERING: `st[d] >= "a"` over an enum-valued function must fail closed (no ordering on
    // an enum index), so even though every value equals "a" the spec does NOT certify.
    let ord = "---- MODULE QFoldOrd ----\n\
               EXTENDS Integers\n\
               CONSTANT D\n\
               VARIABLE st\n\
               Init == st = [d \\in D |-> \"a\"]\n\
               Next == st' = st\n\
               Inv == \\A d \\in D : st[d] >= \"a\"\n\
               ====\n";
    assert!(
        certify_explicit_state_spec(ord, &cfg_dom("Inv")).is_none(),
        "an ordering over a model-value-domain func-enum application must fail closed (no cert)"
    );
}

/// A config with `INVARIANT <inv>` and NO CONSTANT — the shape a `String`-atom-domain function needs
/// (its domain is the inline set literal `{"a","b"}`, not a config model-value set).
fn cfg_inv(inv: &str) -> Config {
    Config::parse(&format!("INIT Init\nNEXT Next\nINVARIANT {inv}\n")).expect("config parses")
}

/// STRING-ATOM FUNCTION DOMAIN, TRUE (the APTCommit shape distilled): a function `f: [{"a","b"} -> …]`
/// whose DOMAIN is a set of `String` atoms (NOT a model-value set, NOT the `0..n-1` Int prefix). Both the
/// TypeOK membership form `f \in [{"a","b"} -> {"lo","hi"}]` and the quantified form
/// `\A k \in {"a","b"} : f[k] \in {"lo","hi"}` certify: the column is a `Str`-kind `FuncEnum` (`dom` = the
/// sorted string keys, `dom_kind = Str`), the `String`-set fold resolves each `f["a"]`/`f["b"]` to its
/// packed slot, the fold rides the kernel safety leg, and Leg-E re-derives the sort + fold byte-identically.
#[test]
fn func_enum_string_domain_true_certifies() {
    // (1) TypeOK membership `f \in [{"a","b"} -> {"lo","hi"}]` — exercises func_set_membership_form with a
    // Str-kind domain (`DOMAIN f = {"a","b"}` discharged against the stored string keys).
    let mem = "---- MODULE StrDomMem ----\n\
               EXTENDS Integers\n\
               VARIABLE f\n\
               Init == f = [k \\in {\"a\", \"b\"} |-> \"lo\"]\n\
               Next == f' = [f EXCEPT ![\"a\"] = \"hi\"]\n\
               Inv == f \\in [{\"a\", \"b\"} -> {\"lo\", \"hi\"}]\n\
               ====\n";
    let cert = certify_explicit_state_spec(mem, &cfg_inv("Inv"))
        .expect("a TRUE String-domain TypeOK membership invariant certifies");
    assert!(
        matches!(&cert.sorts[..], [ColSort::FuncEnum { dom, dom_kind, .. }]
            if dom == &vec!["a".to_string(), "b".to_string()] && *dom_kind == EnumKind::Str),
        "the column is a String-atom-domain FuncEnum (dom=[a,b], kind=Str): {:?}",
        cert.sorts
    );
    assert!(
        verify_explicit_state_cert(&cert),
        "kernel re-check passes (membership)"
    );
    let sc = tla_check::cert::build_explicit_fixpoint_certificate(mem, &cfg_inv("Inv"), cert);
    assert!(
        tla_check::cert::verify_safety_certificate(&sc).accepted,
        "full Leg-E re-derives the String domain + membership fold and ACCEPTS"
    );

    // (2) Quantified form `\A k \in {"a","b"} : f[k] \in {"lo","hi"}` — exercises the String-set quantifier
    // fold + func_enum_app string-key resolution.
    let quant = "---- MODULE StrDomQuant ----\n\
                 EXTENDS Integers\n\
                 VARIABLE f\n\
                 Init == f = [k \\in {\"a\", \"b\"} |-> \"lo\"]\n\
                 Next == f' = [f EXCEPT ![\"a\"] = \"hi\"]\n\
                 Inv == \\A k \\in {\"a\", \"b\"} : f[k] \\in {\"lo\", \"hi\"}\n\
                 ====\n";
    let cert2 = certify_explicit_state_spec(quant, &cfg_inv("Inv"))
        .expect("a TRUE ∀-over-String-domain-function invariant certifies");
    assert!(
        verify_explicit_state_cert(&cert2),
        "kernel re-check passes (quantified)"
    );
    let sc2 = tla_check::cert::build_explicit_fixpoint_certificate(quant, &cfg_inv("Inv"), cert2);
    assert!(
        tla_check::cert::verify_safety_certificate(&sc2).accepted,
        "full Leg-E re-derives the String domain + ∀-fold and ACCEPTS"
    );
}

/// STRING-ATOM FUNCTION DOMAIN, VIOLATED twin: `\A k \in {"a","b"} : f[k] = "lo"` is FALSE on the
/// reachable state where `f["a"]` has flipped to `"hi"`. The fold `⋀_{k} digit_k = idx("lo")` is FALSE
/// there, so the kernel's R⊆Safety leg cannot mint the cert — the String-domain fold does NOT hide the
/// violation (cf. the certifying twin above, proving it is the kernel rejecting, not a recognizer refusal).
#[test]
fn func_enum_string_domain_violation_declines() {
    let spec = "---- MODULE StrDomBad ----\n\
                EXTENDS Integers\n\
                VARIABLE f\n\
                Init == f = [k \\in {\"a\", \"b\"} |-> \"lo\"]\n\
                Next == f' = [f EXCEPT ![\"a\"] = \"hi\"]\n\
                Inv == \\A k \\in {\"a\", \"b\"} : f[k] = \"lo\"\n\
                ====\n";
    assert!(
        certify_explicit_state_spec(spec, &cfg_inv("Inv")).is_none(),
        "a reachable state violates the String-domain ∀-fold ⇒ NOT CERTIFIED (the fold is not hidden)"
    );
}

/// FAIL-CLOSED / KIND-SAFETY for the String-atom domain — three adversarial forms that must DECLINE:
///
///   (a) ORDERING on a String-domain func-enum value (`f[k] >= "lo"`) — an enum index carries no order.
///   (b) MIXED-kind DOMAIN (`[k \in (M \union {"s1"}) |-> …]`, `M` a model-value CONSTANT) — a domain
///       mixing a model value and a `String` atom is out of the fragment (`func_enum_domain_keys` returns
///       `None` on a kind mix), so the cell does not encode and NO cert is minted.
///   (c) CROSS-KIND key resolution — a function with a `String`-atom domain `{"d1","d2"}` but the invariant
///       quantifies over the MODEL-VALUE set `D = {d1,d2}` (same NAMES). The fold substitutes model-value
///       `Ident`s, and `func_enum_app` REFUSES a model-value `Ident` against a `Str`-kind dom (a `String`
///       `"d1"` and a model value `d1` are DISTINCT TLA values ⇒ never one slot), so it DECLINES rather
///       than punning the two. This is the decisive soundness guard: without the kind check the model-value
///       `d1` would resolve to the string key `"d1"`'s slot — an UNSOUND conflation.
#[test]
fn func_enum_string_domain_fail_closed() {
    // (a) ordering on a String-domain func-enum value.
    let ord = "---- MODULE StrDomOrd ----\n\
               EXTENDS Integers\n\
               VARIABLE f\n\
               Init == f = [k \\in {\"a\", \"b\"} |-> \"lo\"]\n\
               Next == f' = f\n\
               Inv == \\A k \\in {\"a\", \"b\"} : f[k] >= \"lo\"\n\
               ====\n";
    assert!(
        certify_explicit_state_spec(ord, &cfg_inv("Inv")).is_none(),
        "ordering over a String-domain func-enum value must fail closed (no cert)"
    );

    // (b) mixed model-value + String domain — out of the fragment.
    let mixed = "---- MODULE StrDomMixed ----\n\
                 EXTENDS Integers\n\
                 CONSTANT M\n\
                 VARIABLE f\n\
                 Init == f = [k \\in (M \\union {\"s1\"}) |-> \"lo\"]\n\
                 Next == f' = f\n\
                 Inv == \\A k \\in (M \\union {\"s1\"}) : f[k] = \"lo\"\n\
                 ====\n";
    let mut cfg_m = cfg_inv("Inv");
    cfg_m.constants.insert(
        "M".to_string(),
        ConstantValue::ModelValueSet(vec!["m1".to_string()]),
    );
    assert!(
        certify_explicit_state_spec(mixed, &cfg_m).is_none(),
        "a domain mixing a model value and a String atom must fail closed (no cert)"
    );

    // (c) cross-kind: a String-atom-domain function keyed "d1","d2" quantified over the model-value set
    // D = {d1,d2} (same names) — the model-value key must NOT resolve to the String key's slot.
    let cross = "---- MODULE StrDomCross ----\n\
                 EXTENDS Integers\n\
                 CONSTANT D\n\
                 VARIABLE f\n\
                 Init == f = [k \\in {\"d1\", \"d2\"} |-> \"a\"]\n\
                 Next == f' = f\n\
                 Inv == \\A d \\in D : f[d] = \"a\"\n\
                 ====\n";
    assert!(
        certify_explicit_state_spec(cross, &cfg_dom("Inv")).is_none(),
        "a model-value key against a String-kind domain must fail closed (no slot conflation)"
    );
}
