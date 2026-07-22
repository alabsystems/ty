// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! GENERALIZED SEQUENCE-ELEMENT encoding for the explicit-state certificate lane (roadmap family F6):
//! `ColSort::Seq` is generalized from Int-only elements to an arbitrary value-type leaf
//! (`CellSort::{Int, Bool, Enum{labels,kind}}`), so a bounded QUEUE over ATOMS / model values / `Bool`s
//! (manipulated with `Append`/`Tail`/`Len`) packs SELF-DELIMITINGLY into ONE `Nat` cell and certifies:
//!
//!   * a VARYING-length queue over `String`/model-value ATOMS encodes as `ColSort::Seq{elem:Enum}` — the
//!     element `code = idx(label)` in the column's sorted element union — with the length machinery
//!     unchanged (`pack = Σ (code+1)·D^i`, `D = base+1`); distinct atom sequences get distinct packs (no
//!     value collapse), the kernel re-checks Init⊆R / image⊆R / R⊆Safety, and the cert round-trips;
//!   * a `Bool` queue encodes as `ColSort::Seq{elem:Bool}` (`code = 1|0`) — the `elem` sort discriminant
//!     keeps `<<FALSE>>` ≠ `<<0>>` (an Int-element and a Bool-element `Seq` are DIFFERENT sorts);
//!   * a PURE-Int queue stays `ColSort::Seq{elem:Int}` — the `elem` field is serde-default + skip-serialized
//!     when `Int`, so every pre-existing Int-`Seq` cert is BYTE-IDENTICAL;
//!   * a FIXED-arity `[1..n -> labels]` function (a program counter — always ONE length, never a queue)
//!     stays a `ColSort::FuncEnum` (queue detection promotes a column only when it is seen at ≥2 different
//!     sequence lengths) — the regression guard for `locks_auxiliary_vars/Lock`'s `pc`;
//!   * fail-closed: a VIOLATED invariant declines (never a false certificate), and a NESTED (tuple/record)
//!     or negative-Int sequence element is out of the value-type-leaf fragment and declines.

#![cfg(feature = "clean-cic")]

use tla_check::explicit_fixpoint_cert::{
    certify_explicit_state_spec, verify_explicit_state_cert, CellSort, ColSort, EnumKind,
};
use tla_check::Config;

fn cfg() -> Config {
    Config::parse("INIT Init\nNEXT Next\nINVARIANT Safety\n").expect("config parses")
}

/// A bounded QUEUE `q` over the ATOMS `{"a","b"}` (enqueue `"a"`/`"b"`, dequeue), with a TRUE invariant
/// `deq <= enq` (you can never dequeue more than you enqueued). The column encodes as the generalized
/// `ColSort::Seq{elem:Enum{["a","b"],Str}}`, distinct atom sequences get DISTINCT packs (`<<"a">>` = 1,
/// `<<"b">>` = 2 — no value collapse), and the kernel re-verifies + the cert round-trips.
#[test]
fn atom_element_queue_certifies_no_collapse() {
    let spec = "---- MODULE AtomQ ----\n\
                EXTENDS Integers, Sequences\n\
                VARIABLES q, enq, deq\n\
                Init == q = <<>> /\\ enq = 0 /\\ deq = 0\n\
                EnqA == enq < 2 /\\ q' = Append(q, \"a\") /\\ enq' = enq + 1 /\\ deq' = deq\n\
                EnqB == enq < 2 /\\ q' = Append(q, \"b\") /\\ enq' = enq + 1 /\\ deq' = deq\n\
                Deq  == q # <<>> /\\ q' = Tail(q) /\\ deq' = deq + 1 /\\ enq' = enq\n\
                Next == EnqA \\/ EnqB \\/ Deq\n\
                Safety == deq <= enq\n\
                ====\n";
    let cert = certify_explicit_state_spec(spec, &cfg()).expect("atom-element queue cert mints");
    assert_eq!(
        cert.sorts[0],
        ColSort::Seq {
            base: 9,
            max_len: 4,
            elem: CellSort::Enum {
                labels: vec!["a".to_string(), "b".to_string()],
                kind: EnumKind::Str
            },
        },
        "q is the generalized atom-element Seq (element = label index in the sorted union)"
    );
    // NO VALUE COLLAPSE: `<<"a">>` packs to the digit `idx("a")+1 = 1`, `<<"b">>` to `idx("b")+1 = 2` —
    // DISTINCT reachable rows (a collapse would equate them and hide a transition).
    let q_packs: std::collections::BTreeSet<u64> = cert.reachable.iter().map(|t| t[0]).collect();
    assert!(
        q_packs.contains(&1) && q_packs.contains(&2),
        "\"a\" and \"b\" get DISTINCT packs: {q_packs:?}"
    );
    assert!(
        verify_explicit_state_cert(&cert),
        "the Clean CIC kernel must re-check the atom-Seq cert (Init⊆R, image⊆R, R⊆Safety)"
    );
}

/// The VIOLATED twin of [`atom_element_queue_certifies_no_collapse`]: the SAME queue with a FALSE
/// invariant `deq <= 0` (a dequeue makes `deq = 1`) must DECLINE — the kernel R⊆Safety leg cannot reduce
/// the conjunction to `Bool.true` on the offending reachable state, so no certificate is minted.
#[test]
fn atom_element_queue_violated_twin_declines() {
    let spec = "---- MODULE AtomQBad ----\n\
                EXTENDS Integers, Sequences\n\
                VARIABLES q, enq, deq\n\
                Init == q = <<>> /\\ enq = 0 /\\ deq = 0\n\
                EnqA == enq < 2 /\\ q' = Append(q, \"a\") /\\ enq' = enq + 1 /\\ deq' = deq\n\
                Deq  == q # <<>> /\\ q' = Tail(q) /\\ deq' = deq + 1 /\\ enq' = enq\n\
                Next == EnqA \\/ Deq\n\
                Safety == deq <= 0\n\
                ====\n";
    assert!(
        certify_explicit_state_spec(spec, &cfg()).is_none(),
        "a FALSE invariant over the atom queue must fail closed (never a false certificate)"
    );
}

/// A `Bool` queue encodes as `ColSort::Seq{elem:Bool}` (`code = 1|0`) — the `elem` sort discriminant keeps
/// a `Bool`-element sequence DISTINCT from an Int-element one, so no `<<FALSE>>`≡`<<0>>` collapse.
#[test]
fn bool_element_queue_certifies() {
    let spec = "---- MODULE BoolQ ----\n\
                EXTENDS Integers, Sequences\n\
                VARIABLES q, enq, deq\n\
                Init == q = <<>> /\\ enq = 0 /\\ deq = 0\n\
                EnqT == enq < 2 /\\ q' = Append(q, TRUE) /\\ enq' = enq + 1 /\\ deq' = deq\n\
                EnqF == enq < 2 /\\ q' = Append(q, FALSE) /\\ enq' = enq + 1 /\\ deq' = deq\n\
                Deq  == q # <<>> /\\ q' = Tail(q) /\\ deq' = deq + 1 /\\ enq' = enq\n\
                Next == EnqT \\/ EnqF \\/ Deq\n\
                Safety == deq <= enq\n\
                ====\n";
    let cert = certify_explicit_state_spec(spec, &cfg()).expect("Bool-element queue cert mints");
    assert_eq!(
        cert.sorts[0],
        ColSort::Seq {
            base: 9,
            max_len: 4,
            elem: CellSort::Bool
        },
        "q is the generalized Bool-element Seq"
    );
    assert!(
        verify_explicit_state_cert(&cert),
        "the kernel must re-check the Bool-Seq cert"
    );
}

/// A PURE-Int queue stays `ColSort::Seq{elem:Int}`, and its serialized cell is BYTE-IDENTICAL to a
/// pre-generalization cert — the `elem` field is skip-serialized when `Int` (no `"elem"` key emitted).
#[test]
fn int_element_queue_byte_identical() {
    let spec = "---- MODULE IntQ ----\n\
                EXTENDS Integers, Sequences\n\
                VARIABLES q, enq, deq\n\
                Init == q = <<>> /\\ enq = 0 /\\ deq = 0\n\
                Enq == enq < 2 /\\ q' = Append(q, 0) /\\ enq' = enq + 1 /\\ deq' = deq\n\
                Deq == q # <<>> /\\ q' = Tail(q) /\\ deq' = deq + 1 /\\ enq' = enq\n\
                Next == Enq \\/ Deq\n\
                Safety == deq <= enq\n\
                ====\n";
    let cert = certify_explicit_state_spec(spec, &cfg()).expect("Int-element queue cert mints");
    assert_eq!(
        cert.sorts[0],
        ColSort::Seq {
            base: 9,
            max_len: 4,
            elem: CellSort::Int
        },
        "a pure-Int queue keeps the historical Int element leaf"
    );
    // BYTE-IDENTITY: the `elem: Int` default is skip-serialized, so an Int-`Seq` cell emits NO `"elem"`
    // key — a pre-existing Int-`Seq` cert re-serializes unchanged.
    let json = serde_json::to_string(&cert.sorts[0]).expect("sort serializes");
    assert!(
        !json.contains("elem"),
        "Int-Seq sort must skip-serialize the elem field: {json}"
    );
    assert!(
        verify_explicit_state_cert(&cert),
        "the kernel must re-check the Int-Seq cert"
    );
}

/// REGRESSION GUARD (`locks_auxiliary_vars/Lock`): a FIXED-arity `[1..n -> labels]` function (a program
/// counter — always ONE length, never a queue) must stay a `ColSort::FuncEnum`, NOT be stolen by the
/// generalized-Seq path. Queue detection promotes a `Value::Seq`/`Value::Tuple` column ONLY when it is
/// seen at ≥2 different sequence lengths; a single-length function never triggers it.
#[test]
fn fixed_arity_label_function_stays_funcenum() {
    let spec = "---- MODULE PcFunc ----\n\
                EXTENDS Integers\n\
                VARIABLE pc\n\
                Init == pc = [i \\in 1..2 |-> \"a\"]\n\
                Next == pc' = [pc EXCEPT ![1] = \"b\"]\n\
                Safety == pc[2] = \"a\"\n\
                ====\n";
    let cert =
        certify_explicit_state_spec(spec, &cfg()).expect("fixed-arity label function cert mints");
    assert!(
        matches!(cert.sorts[0], ColSort::FuncEnum { arity: 2, .. }),
        "a fixed-arity [1..2 -> labels] program counter stays a FuncEnum (not a generalized Seq): {:?}",
        cert.sorts[0]
    );
    assert!(
        verify_explicit_state_cert(&cert),
        "the kernel must re-check the FuncEnum cert"
    );
}

/// FAIL-CLOSED: a sequence element OUTSIDE the value-type-leaf fragment (a NESTED tuple element) declines
/// — the generalized `Seq{elem}` packs Int / `Bool` / atom LEAVES only, never a nested compound.
#[test]
fn nested_element_sequence_fails_closed() {
    let spec = "---- MODULE NestQ ----\n\
                EXTENDS Integers, Sequences\n\
                VARIABLES q, n\n\
                Init == q = <<>> /\\ n = 0\n\
                Enq == n < 2 /\\ q' = Append(q, <<1, 2>>) /\\ n' = n + 1\n\
                Deq == q # <<>> /\\ q' = Tail(q) /\\ n' = n\n\
                Next == Enq \\/ Deq\n\
                Safety == n >= 0\n\
                ====\n";
    assert!(
        certify_explicit_state_spec(spec, &cfg()).is_none(),
        "a sequence of TUPLES (a nested element) is out of the value-type-leaf fragment (fail-closed)"
    );
}
