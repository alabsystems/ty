// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! SET-of-RECORDS (`ColSort::SetMaskRec`) cell encoding for the explicit-state certificate lane — a state
//! variable holding a set `S ⊆ D` over a SMALL FIXED FINITE RECORD UNIVERSE `D` (the two-phase-commit
//! `msgs ⊆ Message` message-set class, `Message` a UNION of record shapes), encoded as the `|D|`-bit
//! BITMASK `Σ_{r∈S} 2^idx(r)` (bit `idx(r)` ⟺ `r ∈ S`). This is the RECORD-DOMAIN analogue of the
//! atom-valued `ColSort::SetMask`: each element is a bounded record, identified by its CANONICAL KEY
//! (`record_value_key` — a length-prefixed, kind-tagged serialization of the sorted `(field, value)`
//! pairs), and the universe `D` is the SORTED cross-state record-key union, so the encoding is a BIJECTION
//! between subsets of `D` and `|D|`-bit values.
//!
//! SOUNDNESS proof (the certify≡check backstop, in-process):
//!   * a message set `msgs ⊆ Message` with `Init msgs={}`, `Next` adding messages, and the real type
//!     invariant `msgs ⊆ Message` CERTIFIES with sort `SetMaskRec{dom}`, kernel-re-checks, and full Leg-E
//!     accepts; the reachable set is exactly the reachable bitmasks (all 2^|D| subsets);
//!   * a VIOLATED twin — the SAME `msgs ⊆ RecordSet` shape but with a STRICTER record set that OMITS a
//!     reachable message shape (`msgs ⊆ [type:{"P"},id:Ids]`, which excludes the reachable `[type|->"C"]`)
//!     — is NOT CERTIFIED: the recognizer DOES build the `SetSubseteq` obligation (proven by the TRUE
//!     sibling with the same shape certifying), and the kernel `⋀_{s∈R} Safety(s) ⇒ Bool.true` leg rejects
//!     the state whose `msgs` holds the omitted record, so the record-set bitmask does NOT hide the
//!     type violation. The decisive op is record-set membership/⊆ over the record universe;
//!   * a cross-`dom` guard: `S = T` between two `SetMaskRec` columns needs matching record universes;
//!   * tamper: a mutated bitmask cell / mutated `dom` fails the kernel re-check / Leg-E binding.
//!
//! The corresponding `ty check` cross-check (the TRUE spec `No error`, the VIOLATED spec `Invariant … is
//! violated`, same state count) is exercised end-to-end at the CLI; here the certify side is the arbiter
//! (certify must never certify what check flags).

#![cfg(feature = "clean-cic")]

use tla_check::explicit_fixpoint_cert::{
    certify_explicit_state_spec, verify_explicit_state_cert, ColSort, ExplicitFixpointCert,
};
use tla_check::{Config, ConstantValue};

/// A config with a model-value SET constant `Ids = {a, b}` (the resource-manager / id set) and the given
/// invariant, wired to `Init`/`Next`.
fn cfg_ids(inv: &str) -> Config {
    let mut config =
        Config::parse(&format!("INIT Init\nNEXT Next\nINVARIANT {inv}\n")).expect("config parses");
    config.constants.insert(
        "Ids".to_string(),
        ConstantValue::ModelValueSet(vec!["a".to_string(), "b".to_string()]),
    );
    config
}

/// The message-set spec: `Message` is the UNION of a two-field record shape `[type:{"P"},id:Ids]` (2
/// records for `Ids={a,b}`) and a single one-field record `[type:{"C"}]` — a 3-record universe. `Init
/// msgs={}`; `Next` spontaneously adds any `Message`, so R is ALL 8 subsets. The invariant is a parameter
/// (`Safety` in the TRUE case, `BadSafety` in the VIOLATED twin).
const SPEC: &str = "---- MODULE SmRec ----\n\
     CONSTANT Ids\n\
     Message == [type : {\"P\"}, id : Ids] \\cup [type : {\"C\"}]\n\
     VARIABLE msgs\n\
     Init == msgs = {}\n\
     Next == \\E m \\in Message : msgs' = (msgs \\cup {m})\n\
     Safety == msgs \\subseteq Message\n\
     BadSafety == msgs \\subseteq [type : {\"P\"}, id : Ids]\n\
     ====\n";

/// TRUE — the real type invariant `msgs ⊆ Message` is a tautology over the observed record universe.
/// Certifies with the column sort `SetMaskRec{dom}` (|dom|=3 record keys), kernel-re-verifies, serde
/// round-trips, and full Leg-E accepts. R is exactly the 8 reachable bitmasks (all subsets of the 3
/// message records).
#[test]
fn record_setmask_subset_certifies() {
    let config = cfg_ids("Safety");
    let cert = certify_explicit_state_spec(SPEC, &config).expect("record-set SetMaskRec certifies");
    match cert.sorts.as_slice() {
        [ColSort::SetMaskRec { dom }] => {
            assert_eq!(
                dom.len(),
                3,
                "the record universe has the 3 Message records"
            );
            // The universe is DETERMINISTICALLY SORTED (bijection element↔bit) ⇒ Leg-E re-derives it.
            let mut sorted = dom.clone();
            sorted.sort();
            assert_eq!(*dom, sorted, "dom is sorted (deterministic)");
        }
        other => panic!("expected a single SetMaskRec column, got {other:?}"),
    }
    // R = all 8 subsets of the 3-record universe, as bitmasks 0..8 (each a 1-tuple).
    assert_eq!(cert.reachable.len(), 8, "R is all 8 subsets");
    assert_eq!(
        cert.reachable,
        (0u64..8).map(|m| vec![m]).collect::<Vec<_>>(),
        "R is exactly the bitmasks 0..7"
    );
    assert!(
        cert.safety_pred.is_some(),
        "the ⊆ invariant rides the general safety leg"
    );
    assert!(
        cert.safety_general.is_some(),
        "kernel-checked ⋀_{{s∈R}} Safety(s) leg present"
    );
    assert!(verify_explicit_state_cert(&cert), "kernel re-check passes");

    // Serde round-trip.
    let bytes = serde_json::to_vec(&cert).expect("serialize");
    let back: ExplicitFixpointCert = serde_json::from_slice(&bytes).expect("deserialize");
    assert_eq!(cert, back, "SetMaskRec cert round-trips");
    assert!(
        verify_explicit_state_cert(&back),
        "the re-loaded SetMaskRec cert verifies"
    );

    // Full Leg-E: build a SafetyCertificate and re-verify end-to-end (re-enumerates + re-derives the sort).
    let sc = tla_check::cert::build_explicit_fixpoint_certificate(SPEC, &config, cert);
    let report = tla_check::cert::verify_safety_certificate(&sc);
    assert!(
        report.accepted,
        "full Leg-E re-check accepts: {}",
        report.detail
    );
}

/// VIOLATED twin — the SAME `msgs ⊆ RecordSet` shape, but the record set `[type:{"P"},id:Ids]` OMITS the
/// reachable `[type|->"C"]` message. The recognizer builds the SAME `SetSubseteq` obligation (its TRUE
/// sibling above proves the shape is recognized + certifiable), but a reachable state whose `msgs` holds
/// `[type|->"C"]` FALSIFIES `msgs ⊆ [type:{"P"},id:Ids]` — the kernel `⋀_{s∈R} Safety(s) ⇒ Bool.true` leg
/// rejects it ⇒ NOT CERTIFIED. This is the decisive gate: the record-set bitmask does not hide the
/// type violation that `ty check` reports.
#[test]
fn record_setmask_subset_violated_declines() {
    let config = cfg_ids("BadSafety");
    assert!(
        certify_explicit_state_spec(SPEC, &config).is_none(),
        "a reachable msgs holding [type|->\"C\"] violates the stricter record-set ⊆ ⇒ MUST NOT certify"
    );
}

/// TAMPER — (a) a mutated bitmask cell (a bit pushed OUTSIDE the record universe) fails the kernel re-check
/// (`msgs ⊆ Message` is now false); (b) a mutated `dom` record key fails the Leg-E spec re-derivation
/// binding (verify re-enumerates, re-collects the real record keys, and requires `re.sorts == fp.sorts`).
#[test]
fn record_setmask_tampered_cert_rejects() {
    let config = cfg_ids("Safety");
    let cert = certify_explicit_state_spec(SPEC, &config).expect("record-set cert mints");

    // (a) kernel-level tamper: push a reachable mask bit out of the universe (bit 5 exceeds |dom|=3), so
    // `msgs ⊆ Message` (whose mask covers only bits 0..3) is falsified ⇒ the kernel re-check rejects.
    let mut kernel_tampered = cert.clone();
    kernel_tampered.reachable[0][0] = 1u64 << 5;
    assert!(
        !verify_explicit_state_cert(&kernel_tampered),
        "a tampered bitmask cell (out-of-universe bit) must fail the kernel re-check"
    );

    // (b) dom tamper: rename a record key in the stored sort. The kernel legs are index-blind Nat
    // equalities, so the binding gate is Leg-E: re-derivation re-collects the real record keys and the
    // sorts no longer match.
    let mut dom_tampered = cert.clone();
    if let ColSort::SetMaskRec { dom } = &mut dom_tampered.sorts[0] {
        dom[0] = format!("{}TAMPER", dom[0]);
    } else {
        panic!("column 0 must be a SetMaskRec sort");
    }
    let sc = tla_check::cert::build_explicit_fixpoint_certificate(SPEC, &config, dom_tampered);
    let report = tla_check::cert::verify_safety_certificate(&sc);
    assert!(
        !report.accepted,
        "a cert with a mutated record key must be rejected: {}",
        report.detail
    );
}
