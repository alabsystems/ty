// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! FUNCTION-to-SET (`ColSort::FuncSetMask`) cell encoding for the explicit-state certificate lane — the
//! "F2" fragment: a state variable holding a bounded finite function `f ∈ [D -> SUBSET E]` whose VALUES are
//! SUBSETS of a small fixed atom universe `E` (the SimpleAllocator `alloc ∈ [Clients -> SUBSET Resources]`
//! class). It is the COMPOSITION of the `ColSort::Func` DOMAIN pack (domain `D`, the key→slot machinery)
//! with the `ColSort::SetMask` VALUE bijection (each value `S_d ⊆ E` encoded as the `|E|`-bit bitmask
//! `Σ_{e∈S_d} 2^idx(e)`): `pack = Σ_d mask(f[fdom_d])·base^d`, `base = 2^|E|`. `f[k]` extracts the value
//! mask digit `(pack / base^slot(k)) mod base`, and every set op on it (`x ∈ f[k]`, `f[k] ⊆ T`, `f[k] = C`,
//! `f[k1] ∩ f[k2] = {}`) is EXACTLY the SetMask bit op on that digit.
//!
//! SOUNDNESS proof (the certify≡check backstop, in-process):
//!   * a mutex-preserving allocator `alloc ∈ [{c1,c2} -> SUBSET {r1,r2}]` with `Init alloc=[c|->{}]`, a
//!     `Next` that grabs a resource only when free / releases it, and the REAL safety invariant
//!     `TypeOK ∧ Mutex` (the function-set type membership AND the disjointness `alloc[c1]∩alloc[c2]={}`
//!     quantifier fold over the value masks) CERTIFIES with sort `FuncSetMask{arity,fdom,dom}`, kernel-
//!     re-checks Init⊆R / image⊆R / R⊆Safety, serde round-trips, and full Leg-E accepts; R is exactly the
//!     9 disjoint resource-holder assignments;
//!   * a VIOLATED twin — the SAME spec but the FALSE invariant `\A c : alloc[c] = {}` (true only in the
//!     initial state) — is NOT CERTIFIED: a reachable state whose `alloc[c]` is non-empty falsifies the
//!     per-key `alloc[c] = {}` mask equality and the kernel `⋀_{s∈R} Safety(s) ⇒ Bool.true` leg rejects it,
//!     so the func-to-set bitmask does NOT hide the violation `ty check` reports (`Invariant … violated`);
//!   * the Int-VALUE-UNIVERSE twin `f ∈ [{a,b} -> SUBSET (0..2)]` with `TypeOK ∧ (∀k: f[k] ⊆ {0,1,2})`
//!     CERTIFIES with sort `FuncSetMask{dom_kind:Int, dom:["0","1","2"]}` over all 64 value assignments;
//!     a VIOLATED subset twin `∀k: f[k] ⊆ {0,1}` and a VIOLATED membership twin `∀k: 2 ∉ f[k]` (both false
//!     once the reachable `2 ∈ f[k]`, using the SAME recognized Int-bitmask shapes) are NOT CERTIFIED — the
//!     kernel `⋀_{s∈R} Safety(s) ⇒ Bool.true` leg rejects them;
//!   * FAIL-CLOSED boundaries: a NEGATIVE-Int value (`{-1}`, outside the nonneg bitmask fragment), an
//!     OVER-WIDTH Int universe (`0..64`, exceeding the one-cell 63-bit cap), and a NESTED value set
//!     (`{{r1}}`, a set of sets) all DECLINE — the encoder never puns a negative / oversize / nested value
//!     into a mask bit;
//!   * TAMPER: a reachable pack cell mutated to a mutex-violating value fails the kernel re-check; a mutated
//!     value-universe `dom` fails the Leg-E spec re-derivation binding.
//!
//! The corresponding `ty check` cross-check (the TRUE spec `No error`, the VIOLATED spec `Invariant …
//! violated`, SAME 9-state count) is exercised end-to-end at the CLI; here the certify side is the arbiter
//! (certify must never certify what check flags).

#![cfg(feature = "clean-cic")]

use tla_check::explicit_fixpoint_cert::{
    certify_explicit_state_spec, verify_explicit_state_cert, ColSort, ExplicitFixpointCert,
};
use tla_check::{Config, ConstantValue};

/// A config with model-value SET constants `Clients = {c1,c2}` and `Resources = {r1,r2}` and the given
/// invariant, wired to `Init`/`Next`.
fn cfg(inv: &str) -> Config {
    let mut config =
        Config::parse(&format!("INIT Init\nNEXT Next\nINVARIANT {inv}\n")).expect("config parses");
    config.constants.insert(
        "Clients".to_string(),
        ConstantValue::ModelValueSet(vec!["c1".to_string(), "c2".to_string()]),
    );
    config.constants.insert(
        "Resources".to_string(),
        ConstantValue::ModelValueSet(vec!["r1".to_string(), "r2".to_string()]),
    );
    config
}

/// The mutex-preserving allocator: `alloc ∈ [Clients -> SUBSET Resources]`; `Init` maps every client to the
/// empty set; `Next` GRABS a resource only when no client holds it (preserving disjointness) or RELEASES a
/// resource. `Safety`/`BadSafety` are the parametric invariants.
const SPEC: &str = "---- MODULE FSm ----\n\
     CONSTANT Clients, Resources\n\
     VARIABLE alloc\n\
     Init == alloc = [c \\in Clients |-> {}]\n\
     Next == \\E c \\in Clients, r \\in Resources :\n\
               \\/ (\\A d \\in Clients : r \\notin alloc[d]) /\\ alloc' = [alloc EXCEPT ![c] = @ \\cup {r}]\n\
               \\/ alloc' = [alloc EXCEPT ![c] = @ \\ {r}]\n\
     TypeOK == alloc \\in [Clients -> SUBSET Resources]\n\
     Mutex == \\A c1, c2 \\in Clients : c1 # c2 => alloc[c1] \\cap alloc[c2] = {}\n\
     Safety == TypeOK /\\ Mutex\n\
     BadSafety == \\A c \\in Clients : alloc[c] = {}\n\
     ====\n";

/// The 9 reachable `alloc` packs (a single 1-tuple column) — every disjoint resource→holder assignment.
/// `base = 2^|Resources| = 4`, `arity = |Clients| = 2`, `pack = mask(alloc[c1]) + 4·mask(alloc[c2])` with
/// `mask(c1) & mask(c2) = 0`: `{0,1,2,3,4,6,8,9,12}`.
fn expected_reachable() -> Vec<Vec<u64>> {
    [0u64, 1, 2, 3, 4, 6, 8, 9, 12]
        .iter()
        .map(|&m| vec![m])
        .collect()
}

/// TRUE — the real invariant `TypeOK ∧ Mutex`. Certifies with sort `FuncSetMask`, kernel-re-verifies, serde
/// round-trips, and full Leg-E accepts. R is exactly the 9 disjoint assignments — the SAME set (and count)
/// `ty check` explores.
#[test]
fn funcsetmask_certifies() {
    let config = cfg("Safety");
    let cert =
        certify_explicit_state_spec(SPEC, &config).expect("func-to-set FuncSetMask certifies");
    match cert.sorts.as_slice() {
        [ColSort::FuncSetMask {
            arity,
            fdom,
            fdom_kind,
            dom,
            dom_kind,
        }] => {
            assert_eq!(*arity, 2, "|D| = |Clients| = 2");
            assert_eq!(
                fdom,
                &["c1", "c2"],
                "domain keys are the sorted Clients model values"
            );
            assert_eq!(
                dom,
                &["r1", "r2"],
                "value universe is the sorted Resources model values"
            );
            use tla_check::explicit_fixpoint_cert::EnumKind;
            assert_eq!(*fdom_kind, EnumKind::Model, "domain keys are model values");
            assert_eq!(*dom_kind, EnumKind::Model, "value atoms are model values");
        }
        other => panic!("expected a single FuncSetMask column, got {other:?}"),
    }
    // base = 2^|dom| = 4, derived (never stored).
    assert_eq!(
        cert.sorts[0].funcsetmask_base(),
        Some(4),
        "base = 2^|E| = 4"
    );
    assert_eq!(
        cert.reachable,
        expected_reachable(),
        "R is exactly the 9 disjoint assignments"
    );
    assert!(
        cert.safety_pred.is_some(),
        "the type + mutex invariant rides the general safety leg"
    );
    assert!(
        cert.safety_general.is_some(),
        "kernel-checked ⋀_{{s∈R}} Safety(s) leg present"
    );
    assert!(verify_explicit_state_cert(&cert), "kernel re-check passes");

    // Serde round-trip (the FuncSetMask sort + SetIR::Digit obligations survive verbatim).
    let bytes = serde_json::to_vec(&cert).expect("serialize");
    let back: ExplicitFixpointCert = serde_json::from_slice(&bytes).expect("deserialize");
    assert_eq!(cert, back, "FuncSetMask cert round-trips");
    assert!(
        verify_explicit_state_cert(&back),
        "the re-loaded FuncSetMask cert verifies"
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

/// VIOLATED twin — the SAME spec, but the invariant `\A c : alloc[c] = {}` holds ONLY in the initial state.
/// A reachable state whose `alloc[c]` is non-empty FALSIFIES the per-key mask equality; the kernel
/// `⋀_{s∈R} Safety(s) ⇒ Bool.true` leg rejects it ⇒ NOT CERTIFIED. The decisive gate: the func-to-set
/// bitmask does not hide the violation `ty check` reports.
#[test]
fn funcsetmask_violated_declines() {
    let config = cfg("BadSafety");
    assert!(
        certify_explicit_state_spec(SPEC, &config).is_none(),
        "a reachable non-empty alloc[c] violates `\\A c : alloc[c] = {{}}` ⇒ MUST NOT certify"
    );
}

/// The Int VALUE-UNIVERSE func-to-set spec `f ∈ [Keys -> SUBSET (0..2)]` (the SimpleRegular
/// `x ∈ [0..N-1 -> SUBSET {0,1}]` class): `Init` empties every key; `Next` adds / removes any `v ∈ 0..2`.
/// `TypeOK` + parametric subset / membership invariants ride the value masks over the Int universe.
const INT_SPEC: &str = "---- MODULE FSmIntU ----\n\
     EXTENDS Integers\n\
     CONSTANT Keys\n\
     VARIABLE f\n\
     Init == f = [k \\in Keys |-> {}]\n\
     Next == \\E k \\in Keys, v \\in 0..2 :\n\
               \\/ f' = [f EXCEPT ![k] = @ \\cup {v}]\n\
               \\/ f' = [f EXCEPT ![k] = @ \\ {v}]\n\
     TypeOK == f \\in [Keys -> SUBSET (0..2)]\n\
     Good == TypeOK /\\ \\A k \\in Keys : f[k] \\subseteq {0, 1, 2}\n\
     BadSubset == TypeOK /\\ \\A k \\in Keys : f[k] \\subseteq {0, 1}\n\
     BadMember == \\A k \\in Keys : 2 \\notin f[k]\n\
     ====\n";

/// A config with model-value domain `Keys = {a,b}`, wired to `Init`/`Next` and the given invariant.
fn int_cfg(inv: &str) -> Config {
    let mut config =
        Config::parse(&format!("INIT Init\nNEXT Next\nINVARIANT {inv}\n")).expect("config parses");
    config.constants.insert(
        "Keys".to_string(),
        ConstantValue::ModelValueSet(vec!["a".to_string(), "b".to_string()]),
    );
    config
}

/// TRUE — the Int VALUE UNIVERSE `f ∈ [{a,b} -> SUBSET (0..2)]`. Certifies with sort
/// `FuncSetMask{dom_kind:Int, dom:["0","1","2"]}` (the value universe is the observed Int decimals), base
/// `2^3 = 8`, over all `8 × 8 = 64` value assignments; kernel-re-verifies, serde round-trips (`dom_kind:Int`
/// survives), and full Leg-E re-derives the SAME Int sort. The SAME faithful subset↔mask bijection as the
/// atom universe, over Int literals.
#[test]
fn funcsetmask_int_universe_certifies() {
    use tla_check::explicit_fixpoint_cert::EnumKind;
    let config = int_cfg("Good");
    let cert =
        certify_explicit_state_spec(INT_SPEC, &config).expect("Int-universe func-to-set certifies");
    match cert.sorts.as_slice() {
        [ColSort::FuncSetMask {
            arity,
            fdom,
            fdom_kind,
            dom,
            dom_kind,
        }] => {
            assert_eq!(*arity, 2, "|D| = |Keys| = 2");
            assert_eq!(
                fdom,
                &["a", "b"],
                "domain keys are the sorted Keys model values"
            );
            assert_eq!(*fdom_kind, EnumKind::Model, "domain keys are model values");
            assert_eq!(
                dom,
                &["0", "1", "2"],
                "value universe = observed Int decimals 0,1,2"
            );
            assert_eq!(*dom_kind, EnumKind::Int, "value universe kind is Int");
        }
        other => panic!("expected a single Int-universe FuncSetMask column, got {other:?}"),
    }
    assert_eq!(
        cert.sorts[0].funcsetmask_base(),
        Some(8),
        "base = 2^|E| = 2^3 = 8"
    );
    assert_eq!(
        cert.reachable.len(),
        64,
        "8 subsets per key × 2 keys = 64 value assignments"
    );
    assert!(
        cert.safety_general.is_some(),
        "the type + subset invariant rides the general safety leg"
    );
    assert!(verify_explicit_state_cert(&cert), "kernel re-check passes");

    // Serde round-trip — the `dom_kind:Int` value universe survives verbatim.
    let bytes = serde_json::to_vec(&cert).expect("serialize");
    let back: ExplicitFixpointCert = serde_json::from_slice(&bytes).expect("deserialize");
    assert_eq!(cert, back, "Int-universe FuncSetMask cert round-trips");
    assert!(
        verify_explicit_state_cert(&back),
        "the re-loaded cert verifies"
    );

    // Full Leg-E: re-enumerate + re-derive the Int sort from the spec and re-verify end-to-end.
    let sc = tla_check::cert::build_explicit_fixpoint_certificate(INT_SPEC, &config, cert);
    let report = tla_check::cert::verify_safety_certificate(&sc);
    assert!(
        report.accepted,
        "full Leg-E re-check accepts: {}",
        report.detail
    );
}

/// VIOLATED (subset) twin — `\A k : f[k] \subseteq {0,1}` holds only until some reachable `2 ∈ f[k]`. The
/// per-key subset is the SAME recognized Int-bitmask shape the TRUE `Good` invariant uses (`⊆ {0,1,2}`), so
/// this is a genuine kernel-leg decline, not a recognition bail: the kernel `⋀_{s∈R} Safety(s) ⇒ Bool.true`
/// leg reduces to FALSE on the reachable `2 ∈ f[k]` state ⇒ NOT CERTIFIED. The func-to-set Int bitmask does
/// not hide the violation `ty check` reports.
#[test]
fn funcsetmask_int_universe_violated_subset_declines() {
    assert!(
        certify_explicit_state_spec(INT_SPEC, &int_cfg("BadSubset")).is_none(),
        "a reachable 2∈f[k] violates `f[k] ⊆ {{0,1}}` ⇒ MUST NOT certify"
    );
}

/// VIOLATED (membership) twin — `\A k : 2 \notin f[k]` is false once `2` is added to some `f[k]`. Exercises
/// the Int-universe MEMBERSHIP recognizer (`2 ∈ f[k]` ⇒ a bit-test on the value digit) in the falsifying
/// direction: the kernel safety leg reduces to FALSE on the reachable `2 ∈ f[k]` state ⇒ NOT CERTIFIED.
#[test]
fn funcsetmask_int_universe_violated_membership_declines() {
    assert!(
        certify_explicit_state_spec(INT_SPEC, &int_cfg("BadMember")).is_none(),
        "a reachable 2∈f[k] violates `2 ∉ f[k]` ⇒ MUST NOT certify"
    );
}

/// FAIL-CLOSED — a NEGATIVE Int value (`f[k] = @ ∪ {-1}`) is outside the NONNEG bitmask fragment: the
/// encoder's `nonneg_small_int` rejects `-1`, so `funcsetmask_view` declines and the negative-valued
/// successor cannot be encoded as `FuncSetMask` (nor any packed sort) ⇒ certify DECLINES, never punning a
/// negative Int into a mask bit.
#[test]
fn funcsetmask_negative_int_fails_closed() {
    const NEG_SPEC: &str = "---- MODULE FSmNeg ----\n\
         EXTENDS Integers\n\
         CONSTANT Keys\n\
         VARIABLE f\n\
         Init == f = [k \\in Keys |-> {}]\n\
         Next == \\E k \\in Keys, v \\in -1..1 : f' = [f EXCEPT ![k] = @ \\cup {v}]\n\
         Safety == \\A k \\in Keys : f[k] \\subseteq {-1, 0, 1}\n\
         ====\n";
    assert!(
        certify_explicit_state_spec(NEG_SPEC, &int_cfg("Safety")).is_none(),
        "a negative Int value is outside the nonneg bitmask fragment ⇒ MUST fail closed"
    );
}

/// FAIL-CLOSED — an OVER-WIDTH Int value universe (`f[k] = 0..64`, a 65-value universe) exceeds the one-cell
/// bitmask width cap (`|E| ≤ 63`): the `GrowSetMask`/width guard declines, so certify DECLINES rather than
/// truncating the universe. Same one-cell cap as the atom `SetMask`, shared verbatim.
#[test]
fn funcsetmask_int_universe_overwidth_fails_closed() {
    const WIDE_SPEC: &str = "---- MODULE FSmWide ----\n\
         EXTENDS Integers\n\
         CONSTANT Keys\n\
         VARIABLE f\n\
         Init == f = [k \\in Keys |-> 0..64]\n\
         Next == UNCHANGED f\n\
         Safety == \\A k \\in Keys : f[k] \\subseteq 0..64\n\
         ====\n";
    assert!(
        certify_explicit_state_spec(WIDE_SPEC, &int_cfg("Safety")).is_none(),
        "a 65-value Int universe exceeds the one-cell 63-bit cap ⇒ MUST fail closed"
    );
}

/// FAIL-CLOSED — a NESTED value set (`alloc[c] = {{r1}}`, a set of SETS) has a non-atom element, so
/// `funcsetmask_view` declines it; the nested-set successor is unencodable ⇒ certify DECLINES (never
/// keying a set element as an atom).
#[test]
fn funcsetmask_nested_set_fails_closed() {
    const NESTED_SPEC: &str = "---- MODULE FSmNest ----\n\
         CONSTANT Clients, Resources\n\
         VARIABLE alloc\n\
         Init == alloc = [c \\in Clients |-> {}]\n\
         Next == \\E c \\in Clients, r \\in Resources : alloc' = [alloc EXCEPT ![c] = @ \\cup {{r}}]\n\
         Safety == \\A c \\in Clients : alloc[c] \\subseteq SUBSET Resources\n\
         ====\n";
    let config = cfg("Safety");
    assert!(
        certify_explicit_state_spec(NESTED_SPEC, &config).is_none(),
        "a nested (set-of-sets) value is outside the atom-only F2 scope ⇒ MUST fail closed"
    );
}

/// TAMPER — (a) a reachable pack mutated to a MUTEX-VIOLATING value (`pack = 5` ⇒ `alloc[c1]={r1}` AND
/// `alloc[c2]={r1}`, both holding `r1`) fails the kernel re-check (`Mutex` is now false on that state); (b) a
/// mutated value-universe `dom` fails the Leg-E spec re-derivation binding (verify re-enumerates, re-collects
/// the real value atoms, and requires `re.sorts == fp.sorts`).
#[test]
fn funcsetmask_tampered_cert_rejects() {
    let config = cfg("Safety");
    let cert = certify_explicit_state_spec(SPEC, &config).expect("func-to-set cert mints");
    // `pack = 5` = mask(c1)=1 (bit r1) + 4·mask(c2)=1 (bit r1) ⇒ both clients hold r1 ⇒ Mutex violated. It
    // is NOT a reachable pack (R = {0,1,2,3,4,6,8,9,12}), so injecting it introduces a mutex-violating state.
    assert!(
        !cert.reachable.iter().any(|t| t == &vec![5u64]),
        "pack 5 is not already reachable"
    );

    let mut kernel_tampered = cert.clone();
    kernel_tampered.reachable[0][0] = 5;
    assert!(
        !verify_explicit_state_cert(&kernel_tampered),
        "a tampered pack holding a mutex-violating assignment must fail the kernel re-check"
    );

    // dom tamper: rename a value-universe atom in the stored sort. The kernel legs are index-blind Nat
    // equalities, so the binding gate is Leg-E: re-derivation re-collects the real value atoms and the sorts
    // no longer match.
    let mut dom_tampered = cert.clone();
    if let ColSort::FuncSetMask { dom, .. } = &mut dom_tampered.sorts[0] {
        dom[0] = format!("{}TAMPER", dom[0]);
    } else {
        panic!("column 0 must be a FuncSetMask sort");
    }
    let sc = tla_check::cert::build_explicit_fixpoint_certificate(SPEC, &config, dom_tampered);
    let report = tla_check::cert::verify_safety_certificate(&sc);
    assert!(
        !report.accepted,
        "a cert with a mutated value atom must be rejected: {}",
        report.detail
    );
}
