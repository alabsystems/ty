// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! INT-KEYED (1-based / general `lo..hi`) FUNCTION DOMAIN for the explicit-state certificate lane — the
//! F3 fragment that certifies PlusCal process-counters `pc ∈ [1..N -> labels]` (Lock / Peterson / Barrier).
//!
//! A TLA function `[i ∈ 1..N |-> …]` IS a 1-based sequence, so ty represents it as `Value::Seq`; the F3
//! extension recognizes a homogeneous-label sequence as a `FuncEnum` (and a Bool sequence as a `Func`) over
//! the Int domain `1..N` — `dom = ["1",…,"N"]`, `dom_kind = EnumKind::Int`, `f[k]` at slot `k − 1`. This
//! suite is the controlled soundness proof:
//!
//!   * a 1-based `pc ∈ [1..3 -> {"a","b"}]` with a TRUE invariant CERTIFIES + kernel-re-checks + full-Leg-E
//!     accepts, and the column's sort is the Int-keyed `FuncEnum` (`dom = ["1","2","3"]`, `dom_kind = Int`);
//!   * a `[1..N -> BOOLEAN]` function (Peterson's `c`) certifies as an Int-keyed `Func` with `Bool` cells;
//!   * a multi-var quantifier `\A i, j ∈ 1..N` + a GROUND-expression index (`pc[3-i]`, `pc[Other(i)]`)
//!     recognize (the Lock/Peterson mutual-exclusion clause);
//!   * a VIOLATED twin does NOT certify (the kernel rejects, faithful to `ty check`);
//!   * fail-closed guards: an OUT-OF-INTERVAL key, an Int key against a MODEL-value domain (kind cross),
//!     and a NON-GROUND (state-dependent) index all decline;
//!   * digest back-compat: a 0-based Int-prefix `FuncEnum` cert serializes with NO `dom`/`dom_kind` bytes
//!     (BYTE-IDENTICAL to a pre-F3 cert — the additive `Int` kind never leaks into existing shapes).

#![cfg(feature = "clean-cic")]

use tla_check::explicit_fixpoint_cert::{
    certify_explicit_state_spec, recognize_spec_fixpoint_irs, verify_explicit_state_cert, CellSort,
    ColSort, EnumKind,
};
use tla_check::{Config, ConstantValue};

fn cfg_inv(inv: &str) -> Config {
    Config::parse(&format!("INIT Init\nNEXT Next\nINVARIANT {inv}\n")).expect("config parses")
}

/// A config with `INVARIANT <inv>` and a two-member MODEL-VALUE domain `D = {d1, d2}` — for the cross-kind
/// guard (an Int key against a model-value-domain function).
fn cfg_dom(inv: &str) -> Config {
    let mut c = cfg_inv(inv);
    c.constants.insert(
        "D".to_string(),
        ConstantValue::ModelValueSet(vec!["d1".to_string(), "d2".to_string()]),
    );
    c
}

/// 1-BASED INT DOMAIN, TRUE. `pc ∈ [1..3 -> {"a","b"}]` (a `Value::Seq` of labels) encodes as an Int-keyed
/// `FuncEnum` (`dom = ["1","2","3"]`, `dom_kind = Int`, base `|labels| = 2`). The membership invariant
/// `pc ∈ [1..3 -> {"a","b"}]` discharges `DOMAIN pc = 1..3` numerically, and both the literal-index form
/// `pc[2] = "a"` and the `∀ i ∈ 1..3` fold resolve each `pc[i]` to slot `i − 1`. Certifies + kernel-re-checks
/// + full Leg-E re-derives the SAME sort.
#[test]
fn func_int_domain_true_certifies() {
    let spec = "---- MODULE PcInt1 ----\n\
                EXTENDS Integers\n\
                VARIABLE pc\n\
                ProcSet == 1..3\n\
                Init == pc = [i \\in ProcSet |-> \"a\"]\n\
                Next == pc' = [pc EXCEPT ![1] = \"b\"]\n\
                TypeOK == pc \\in [ProcSet -> {\"a\", \"b\"}]\n\
                ====\n";
    let cert = certify_explicit_state_spec(spec, &cfg_inv("TypeOK"))
        .expect("a 1-based Int-domain FuncEnum TypeOK certifies");
    assert!(
        matches!(&cert.sorts[..], [ColSort::FuncEnum { arity: 3, dom, dom_kind, .. }]
            if dom == &vec!["1".to_string(), "2".to_string(), "3".to_string()]
                && *dom_kind == EnumKind::Int),
        "the column is an Int-keyed FuncEnum (arity 3, dom=[1,2,3], kind=Int): {:?}",
        cert.sorts
    );
    assert!(
        verify_explicit_state_cert(&cert),
        "kernel re-check passes (membership)"
    );
    let sc = tla_check::cert::build_explicit_fixpoint_certificate(spec, &cfg_inv("TypeOK"), cert);
    assert!(
        tla_check::cert::verify_safety_certificate(&sc).accepted,
        "full Leg-E re-derives the Int domain + membership fold and ACCEPTS"
    );

    // The literal-index equality `pc[2] = "a"` (slot of key "2" is 1) is TRUE on every reachable state.
    let eq = "---- MODULE PcInt1Eq ----\n\
              EXTENDS Integers\n\
              VARIABLE pc\n\
              Init == pc = [i \\in 1..3 |-> \"a\"]\n\
              Next == pc' = [pc EXCEPT ![1] = \"b\"]\n\
              Safety == pc[2] = \"a\"\n\
              ====\n";
    let cert2 = certify_explicit_state_spec(eq, &cfg_inv("Safety"))
        .expect("a TRUE literal-index equality over an Int-domain FuncEnum certifies");
    assert!(
        verify_explicit_state_cert(&cert2),
        "kernel re-check passes (literal index)"
    );
}

/// 1-BASED INT DOMAIN, BOOL codomain (Peterson's `c`). `c ∈ [1..2 -> BOOLEAN]` (a `Value::Seq` of Bools)
/// encodes as an Int-keyed `Func` whose cells are `Bool`. The membership `c ∈ [1..2 -> BOOLEAN]` discharges
/// per-position `digit ∈ {0,1}`; a pure-Int sequence would instead stay a `ColSort::Seq` (byte-compat).
#[test]
fn func_int_domain_bool_certifies() {
    let spec = "---- MODULE CInt1 ----\n\
                EXTENDS Integers\n\
                VARIABLE c\n\
                ProcSet == 1..2\n\
                Init == c = [i \\in ProcSet |-> FALSE]\n\
                Next == c' = [c EXCEPT ![1] = TRUE]\n\
                TypeOK == c \\in [ProcSet -> BOOLEAN]\n\
                ====\n";
    let cert = certify_explicit_state_spec(spec, &cfg_inv("TypeOK"))
        .expect("a 1-based Int-domain Bool Func certifies");
    assert!(
        matches!(&cert.sorts[..], [ColSort::Func { arity: 2, dom, dom_kind, cells, .. }]
            if dom == &vec!["1".to_string(), "2".to_string()]
                && *dom_kind == EnumKind::Int
                && cells == &vec![CellSort::Bool, CellSort::Bool]),
        "the column is an Int-keyed Bool Func (arity 2, dom=[1,2], kind=Int, Bool cells): {:?}",
        cert.sorts
    );
    assert!(
        verify_explicit_state_cert(&cert),
        "kernel re-check passes (Bool func)"
    );
    let sc = tla_check::cert::build_explicit_fixpoint_certificate(spec, &cfg_inv("TypeOK"), cert);
    assert!(
        tla_check::cert::verify_safety_certificate(&sc).accepted,
        "full Leg-E re-derives the Int-domain Bool Func and ACCEPTS"
    );
}

/// MULTI-VAR quantifier + GROUND-EXPRESSION index (the Lock/Peterson mutual-exclusion shape). Over
/// `pc ∈ [1..2 -> {"a","b"}]` (all values always in `{"a","b"}`), every clause is a TAUTOLOGY on `R`, so this
/// isolates RECOGNITION: (1) `\A i, j ∈ 1..2` decomposes the 2-var Int quantifier (peel-one-var), (2) the
/// GROUND `IF` index `pc[Other(i)]` and (3) the GROUND arithmetic index `pc[3 - i]` fold to a literal slot.
#[test]
fn func_int_domain_multivar_and_ground_index_certifies() {
    let spec = "---- MODULE PcIntMV ----\n\
                EXTENDS Integers\n\
                VARIABLE pc\n\
                ProcSet == 1..2\n\
                Other(p) == IF p = 1 THEN 2 ELSE 1\n\
                Init == pc = [i \\in ProcSet |-> \"a\"]\n\
                Next == pc' = [pc EXCEPT ![1] = \"b\"]\n\
                Inv == /\\ \\A i, j \\in ProcSet : (i # j) => (pc[i] \\in {\"a\", \"b\"} /\\ pc[j] \\in {\"a\", \"b\"})\n\
                       /\\ \\A i \\in ProcSet : pc[Other(i)] \\in {\"a\", \"b\"}\n\
                       /\\ \\A i \\in ProcSet : pc[3 - i] \\in {\"a\", \"b\"}\n\
                ====\n";
    let cert = certify_explicit_state_spec(spec, &cfg_inv("Inv"))
        .expect("multi-var Int quantifier + ground-expression indices recognize and certify");
    assert!(
        verify_explicit_state_cert(&cert),
        "kernel re-check passes (multi-var + ground index)"
    );
    let sc = tla_check::cert::build_explicit_fixpoint_certificate(spec, &cfg_inv("Inv"), cert);
    assert!(
        tla_check::cert::verify_safety_certificate(&sc).accepted,
        "full Leg-E re-derives the multi-var fold + ground-index resolution and ACCEPTS"
    );
}

/// VIOLATED TWIN. The SAME `[1..3 -> {"a","b"}]` state machine with `Safety == pc[1] = "a"` — FALSE on the
/// reachable state where `Next` has flipped `pc[1]` to `"b"`. The kernel's `R ⊆ Safety` leg cannot reduce
/// the offending state to `Bool.true`, so NO cert is minted — the Int-domain resolution does NOT hide the
/// violation (cf. the certifying `pc[2] = "a"` twin above: it is the kernel rejecting, not a recognizer
/// refusal). Cross-checked against `ty check`, which reports the same violation.
#[test]
fn func_int_domain_violation_declines() {
    let spec = "---- MODULE PcInt1Bad ----\n\
                EXTENDS Integers\n\
                VARIABLE pc\n\
                Init == pc = [i \\in 1..3 |-> \"a\"]\n\
                Next == pc' = [pc EXCEPT ![1] = \"b\"]\n\
                Safety == pc[1] = \"a\"\n\
                ====\n";
    assert!(
        certify_explicit_state_spec(spec, &cfg_inv("Safety")).is_none(),
        "a reachable state violates pc[1]=\"a\" ⇒ NOT CERTIFIED (the Int-domain fold is not hidden)"
    );
}

/// FAIL-CLOSED / KIND-SAFETY for the Int-keyed domain — three adversarial forms that must DECLINE:
///
///   (a) OUT-OF-INTERVAL key — `pc[4]` over a `1..3` domain: `4 ∉ dom` ⇒ no slot ⇒ the access declines and
///       no cert is minted (making `f[k]` decline for `k ∉ dom(f)` is the soundness guard against reading an
///       out-of-domain apply).
///   (b) CROSS-KIND key — an Int LITERAL index `f[1]` against a MODEL-VALUE domain `[D -> …]` (`D={d1,d2}`):
///       an Int key never routes to an atom (`Model`) domain (`dom_kind != Int`), so it DECLINES rather than
///       punning slot 0. (The dual — a model-value `Ident` key against an `Int` domain — cannot arise: a
///       1-based domain's keys are literals, never idents.)
///   (c) NON-GROUND index — `pc[turn]` where `turn` is a STATE VARIABLE: `const_eval_nonneg_int` yields
///       `None` for a column reference, so a state-dependent index fails closed (never punned to a slot).
#[test]
fn func_int_domain_fail_closed() {
    // (a) out-of-interval literal key.
    let oor = "---- MODULE PcIntOOR ----\n\
               EXTENDS Integers\n\
               VARIABLE pc\n\
               Init == pc = [i \\in 1..3 |-> \"a\"]\n\
               Next == pc' = [pc EXCEPT ![1] = \"b\"]\n\
               Safety == pc[4] = \"a\"\n\
               ====\n";
    assert!(
        certify_explicit_state_spec(oor, &cfg_inv("Safety")).is_none(),
        "an out-of-interval Int key (4 ∉ 1..3) must fail closed (no cert)"
    );

    // (b) cross-kind: an Int literal key against a MODEL-VALUE-domain function.
    let cross = "---- MODULE FModelIntKey ----\n\
                 EXTENDS Integers\n\
                 CONSTANT D\n\
                 VARIABLE f\n\
                 Init == f = [k \\in D |-> \"a\"]\n\
                 Next == f' = f\n\
                 Safety == f[1] = \"a\"\n\
                 ====\n";
    assert!(
        certify_explicit_state_spec(cross, &cfg_dom("Safety")).is_none(),
        "an Int literal key against a model-value domain must fail closed (no slot conflation)"
    );

    // (c) non-ground (state-dependent) index.
    let nonground = "---- MODULE PcIntVarIdx ----\n\
                     EXTENDS Integers\n\
                     VARIABLES pc, turn\n\
                     Init == pc = [i \\in 1..2 |-> \"a\"] /\\ turn = 1\n\
                     Next == pc' = [pc EXCEPT ![1] = \"b\"] /\\ turn' = turn\n\
                     Safety == pc[turn] = \"a\"\n\
                     ====\n";
    assert!(
        certify_explicit_state_spec(nonground, &cfg_inv("Safety")).is_none(),
        "a non-ground (state-variable) index must fail closed (no cert)"
    );
}

/// DIGEST BACK-COMPAT. A 0-based Int-prefix `FuncEnum` (`pc ∈ [0..2 -> {"a","b"}]`) still serializes with
/// its `dom` EMPTY and its `dom_kind` at the skip-serialized `Model` default — so the JSON carries NEITHER a
/// `dom` NOR a `dom_kind` key, BYTE-IDENTICAL to a pre-F3 cert (the additive `Int` kind never leaks into an
/// existing shape). This is the digest-compatibility invariant for every pre-existing 0-based/atom cert.
#[test]
fn int_prefix_funcenum_cert_byte_identical() {
    let spec = "---- MODULE Pc0Based ----\n\
                EXTENDS Integers\n\
                VARIABLE pc\n\
                Init == pc = [i \\in 0..2 |-> \"a\"]\n\
                Next == pc' = [pc EXCEPT ![0] = \"b\"]\n\
                Safety == pc[1] = \"a\"\n\
                ====\n";
    let cert = certify_explicit_state_spec(spec, &cfg_inv("Safety"))
        .expect("a 0-based Int-prefix FuncEnum certifies");
    assert!(
        matches!(&cert.sorts[..], [ColSort::FuncEnum { dom, dom_kind, .. }]
            if dom.is_empty() && *dom_kind == EnumKind::Model),
        "a 0-based prefix FuncEnum keeps EMPTY dom + Model kind: {:?}",
        cert.sorts
    );
    let json = serde_json::to_string(&cert).expect("serialize");
    assert!(
        !json.contains("\"dom\""),
        "a 0-based FuncEnum cert must omit the `dom` key (byte-identical to pre-F3): {json}"
    );
    assert!(
        !json.contains("\"dom_kind\"") && !json.contains("\"Int\""),
        "a 0-based FuncEnum cert must omit `dom_kind`/`Int` (byte-identical to pre-F3): {json}"
    );
    assert!(
        verify_explicit_state_cert(&cert),
        "the 0-based cert still verifies"
    );
}

/// BARRIER-SHAPED, ENUMERATOR-FREE. The corpus `Barrier` shape: `Next` is a disjunction of a per-process
/// `∃ p ∈ ProcSet : pc' = [pc EXCEPT ![p] = "b1"]` step AND a `b1` reset whose update is a FULL
/// CONSTRUCTOR `pc' = [p ∈ ProcSet |-> "b0"]` over the OPERATOR-defined Int domain `ProcSet == 1..2`.
/// The reset's `FuncDef` domain (and the Init's) is a bare operator ident, so it MUST be inlined to the
/// literal interval `1..2` before `func_enum_update_eq_form`'s constructor arm can `int_domain_matches`
/// it. This test is the regression guard for the `cert_inline` `FuncDef`/`Except` descent: without it the
/// constructor arm saw a bare `Ident("ProcSet")` and declined ⇒ `next_pred = None` ⇒ ENUMERATOR-ASSISTED.
/// With the fix the whole `Next`/`Init` recognize ⇒ the cert carries `next_pred` + the two general
/// completeness legs (enumerator-FREE). It ALSO guards the Int-key→slot mapping: a WRONG slot would make a
/// recognized successor fall OUTSIDE the enumerated `R`, so the closure completeness leg would fail and
/// `next_pred` would stay `None` — this assertion only holds when every `![p]`/`[p |-> …]` slot is exact.
#[test]
fn barrier_shaped_funcdef_next_certifies_enumerator_free() {
    let spec = "---- MODULE BarShape ----\n\
                EXTENDS Integers\n\
                VARIABLE pc\n\
                ProcSet == 1..2\n\
                Init == pc = [p \\in ProcSet |-> \"b0\"]\n\
                b0(self) == pc[self] = \"b0\" /\\ pc' = [pc EXCEPT ![self] = \"b1\"]\n\
                b1 == (\\A p \\in ProcSet : pc[p] = \"b1\") /\\ pc' = [p \\in ProcSet |-> \"b0\"]\n\
                Next == (\\E p \\in ProcSet : b0(p)) \\/ b1\n\
                TypeOK == pc \\in [ProcSet -> {\"b0\", \"b1\"}]\n\
                ====\n";
    let cert = certify_explicit_state_spec(spec, &cfg_inv("TypeOK"))
        .expect("a Barrier-shaped Int-domain FuncEnum spec certifies");
    assert!(
        matches!(&cert.sorts[..], [ColSort::FuncEnum { arity: 2, dom, dom_kind, .. }]
            if dom == &vec!["1".to_string(), "2".to_string()] && *dom_kind == EnumKind::Int),
        "the column is the Int-keyed FuncEnum [1..2 -> {{b0,b1}}]: {:?}",
        cert.sorts
    );
    // The ENUMERATOR-FREE tier: the FuncDef-in-Next constructor + the ∃-EXCEPT step both recognized, so the
    // kernel re-evaluates Next/Init over the finite domain (not TY's enumerated image).
    assert!(
        cert.next_pred.is_some() && cert.next_general_completeness.is_some(),
        "Barrier-shaped Next recognizes ⇒ enumerator-FREE (next_pred + completeness present)"
    );
    assert!(
        cert.init_pred.is_some() && cert.init_general_completeness.is_some(),
        "the Init FuncDef constructor recognizes ⇒ enumerator-FREE Init leg present"
    );
    assert!(
        verify_explicit_state_cert(&cert),
        "kernel re-check passes (Barrier shape)"
    );
    let sc = tla_check::cert::build_explicit_fixpoint_certificate(spec, &cfg_inv("TypeOK"), cert);
    assert!(
        tla_check::cert::verify_safety_certificate(&sc).accepted,
        "full Leg-E re-derives the Barrier-shaped inlined FuncDef/EXCEPT and ACCEPTS"
    );
}

/// REFLECT-BIND (`reflect-check --full` spec-bind) for an Int-domain FuncEnum. The recognition-only
/// re-derivation `recognize_spec_fixpoint_irs` — which derives the column sort STRUCTURALLY from the type
/// invariant `pc ∈ [1..2 -> {"b0","b1"}]` (NO enumerator) — must produce the SAME Int-keyed FuncEnum sort
/// the certify enumerator mints, else the `--full` spec-bind is INCONCLUSIVE (the reflected legs can't bind
/// to the spec). Regression guard for `derive_funcenum_sort`'s Int-interval branch: it reproduces
/// `int_interval_domain_keys(1, 2)` = `dom = ["1","2"], dom_kind = Int`, matching the stored sort exactly.
#[test]
fn int_domain_funcenum_reflect_binds() {
    let spec = "---- MODULE BarBind ----\n\
                EXTENDS Integers\n\
                VARIABLE pc\n\
                ProcSet == 1..2\n\
                Init == pc = [p \\in ProcSet |-> \"b0\"]\n\
                Next == \\E p \\in ProcSet : pc[p] = \"b0\" /\\ pc' = [pc EXCEPT ![p] = \"b1\"]\n\
                TypeOK == pc \\in [ProcSet -> {\"b0\", \"b1\"}]\n\
                ====\n";
    let cfg = cfg_inv("TypeOK");
    // The enumerator-minted sort (ground truth the reflect-bind must match).
    let cert = certify_explicit_state_spec(spec, &cfg).expect("certifies");
    let re = recognize_spec_fixpoint_irs(spec, &cfg).expect(
        "the reflect-bind RE-DERIVES (sorts, Init, Next, Safety) for an Int-domain FuncEnum",
    );
    assert_eq!(
        re.sorts, cert.sorts,
        "the STRUCTURALLY-derived Int-domain FuncEnum sort equals the certify enumerator's sort"
    );
    assert!(
        matches!(&re.sorts[..], [ColSort::FuncEnum { arity: 2, dom, dom_kind, .. }]
            if dom == &vec!["1".to_string(), "2".to_string()] && *dom_kind == EnumKind::Int),
        "reflect-bind derived the Int-keyed FuncEnum [1..2] from the type invariant: {:?}",
        re.sorts
    );
}

/// INT-KEY → SLOT POSITION RESOLUTION (the soundness-critical mapping `key n ↦ position n − lo`). Over
/// `pc ∈ [1..2 -> {"a","b"}]` an `EXCEPT ![1]` must update SLOT 0 (key 1), leaving `pc[2]` (slot 1, key 2)
/// UNCHANGED — so `Safety == pc[2] = "a"` is INVARIANT and the spec CERTIFIES enumerator-free. The twin
/// that updates `![2]` (slot 1) makes `pc[2]` become `"b"`, VIOLATING the same `Safety`, so it does NOT
/// certify. A wrong mapping (off-by-one / n-vs-(n−1) / swapped) would flip BOTH verdicts — pinning the
/// wrong slot recognizes a different Next ⇒ a false closure. The paired certify/decline is the guard.
#[test]
fn int_key_resolves_to_correct_slot() {
    // ![1] updates slot 0 ⇒ pc[2] stays "a" ⇒ Safety pc[2]="a" holds ⇒ CERTIFIES enumerator-free.
    let key1 = "---- MODULE Key1 ----\n\
                EXTENDS Integers\n\
                VARIABLE pc\n\
                Init == pc = [i \\in 1..2 |-> \"a\"]\n\
                Next == pc[1] = \"a\" /\\ pc' = [pc EXCEPT ![1] = \"b\"]\n\
                Safety == pc[2] = \"a\"\n\
                ====\n";
    let cert = certify_explicit_state_spec(key1, &cfg_inv("Safety"))
        .expect("![1] updates slot 0 (key 1); pc[2] (slot 1) unchanged ⇒ Safety holds ⇒ certifies");
    assert!(
        cert.next_pred.is_some(),
        "enumerator-FREE: the recognized ![1]→slot-0 successor lands inside R (closure holds)"
    );
    // Reachable is exactly {[a,a]=0, [b,a]=1} — slot 1 (pc[2]) never flips (its digit stays 0="a"). A
    // wrong ![1]→slot-1 mapping would instead reach [a,b]=2.
    let mut reach: Vec<Vec<u64>> = cert.reachable.clone();
    reach.sort();
    assert_eq!(
        reach,
        vec![vec![0u64], vec![1u64]],
        "only slot 0 flips: R = {{0,1}}, never pack 2"
    );

    // The TWIN: ![2] updates slot 1 ⇒ pc[2] becomes "b" ⇒ Safety pc[2]="a" VIOLATED ⇒ does NOT certify.
    let key2 = "---- MODULE Key2 ----\n\
                EXTENDS Integers\n\
                VARIABLE pc\n\
                Init == pc = [i \\in 1..2 |-> \"a\"]\n\
                Next == pc[2] = \"a\" /\\ pc' = [pc EXCEPT ![2] = \"b\"]\n\
                Safety == pc[2] = \"a\"\n\
                ====\n";
    assert!(
        certify_explicit_state_spec(key2, &cfg_inv("Safety")).is_none(),
        "![2] updates slot 1 ⇒ pc[2] becomes b ⇒ Safety violated ⇒ NOT CERTIFIED (no wrong-slot false safe)"
    );
}
