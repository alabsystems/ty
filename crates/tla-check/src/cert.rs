// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Serialized, re-checkable inductive-safety + deadlock-freedom certificates
//! (`ty.cert/v1`).
//!
//! Certifying verification turns a model-checking VERDICT into a re-checkable PROOF
//! artifact: a serialized certificate that `ty cert-check` re-validates.
//!
//! ## What is certified
//!
//! A certificate certifies BOTH (a) inductive INVARIANT-safety — the configured
//! invariants hold on every reachable state — AND (b) DEADLOCK-FREEDOM — every
//! reachable state has a successor (`J => Enabled(Next)`). It does NOT (yet) certify
//! liveness. A spec that can deadlock is REFUSED by the producer and REJECTED by the
//! verifier (see `test_deadlocking_spec_rejected`).
//!
//! ## What "re-check" means, and its trust boundary (HONEST scope)
//!
//! `verify_safety_certificate` re-validates by legs that ALL must hold (fail-closed).
//! The three SMT obligations are re-checked from EMBEDDED proofs WITHOUT re-solving
//! (Leg D); deadlock-freedom and the eval oracle are unchanged:
//!  - **Leg A** — the explicit-state eval oracle: re-derives reachable states from
//!    the certificate's own `spec_src` and confirms `J`, the safety invariants, and
//!    deadlock-freedom by a DIFFERENT engine (the tla-eval tree-walk, no solver) than
//!    the symbolic AY/SMT path that produced the cert. It is a bounded REFUTER —
//!    `NoViolation` within its 4096-state bound never *carries* acceptance; it is a
//!    required cross-check that must not refute. It is engine-diverse at the
//!    EVALUATION stage only.
//!  - **Leg D (the ACCEPTANCE BASIS)** — the EXTERNAL proof re-check. For each of the
//!    THREE SMT obligations (initiation `Init /\ ~J`, consecution `J /\ Next /\ ~J'`,
//!    safety `J /\ ~Safety`) the certificate embeds a `SerializableProofBundle`. The
//!    verifier (1) deserializes it (fail-closed on schema/parse), (2) calls
//!    `ay_proof::re_check_bundle_strict`, which rebuilds a CHECKER-ONLY `TermStore`
//!    and runs `check_proof_strict` — proving the embedded assertion set UNSAT via
//!    AY's audited checker with NO solver search, (3) requires the proof's `Assume`
//!    set to equal the bundle's `obligation_assertions` (SET), (4) requires the
//!    bundle's canonical-rendered assertion multiset to EQUAL the multiset TY produces
//!    by re-translating the SAME obligation through its own `BmcTranslator` WITHOUT
//!    solving, and (5) for SCALAR (Int/Bool) specs, ENGINE-DIVERSELY confirms the
//!    embedded AY obligation denotes the spec: at a set of probe states it folds the
//!    embedded obligation to a boolean (AY `substitute`+`simplify`) and requires
//!    agreement with `tla-eval` evaluating the SAME TLA obligation — a DIFFERENT
//!    engine than `BmcTranslator`, so a translation/negation bug is CAUGHT. (3)+(4)+(5)
//!    bind the audited proof to the obligation TY recognizes.
//!  - **Leg C (DEFENSE-IN-DEPTH)** — AY re-discharges all four obligations in process
//!    (re-solve + strict-check). KEPT as an additional necessary condition, no longer
//!    the sole basis.
//!
//! TRUST BOUNDARY (honest, NARROWED scope): Leg D removes the producer's SOLVER from
//! the trust base for the three SMT obligations — acceptance no longer rests on
//! re-running AY's search, only on re-checking an embedded proof object with AY's
//! audited checker, so it is producer-SOLVER-INDEPENDENT. For SCALAR specs the
//! obligation-binding is also engine-DIVERSE: step (5) confirms the embedded obligation
//! denotes the spec via `tla-eval`, NOT `BmcTranslator`, so a TLA->AY *translation* (or
//! `negate_normalized`) bug is caught — it no longer hides behind the render equality
//! (step 4), which trusts `BmcTranslator` on both sides. And a THIRD comparand (step 5,
//! `cert_indep_frontend`) re-derives the obligation through a SECOND, fully INDEPENDENT
//! TLA front end (its own tokenizer + parser + evaluator, sharing NOTHING with
//! `tla_core` parse/lower OR `BmcTranslator` — enforced by an import self-test); when it
//! can parse the scalar fragment it must ALSO agree, so even a parse/lower bug is caught
//! over the probed states. What remains: the probe set is BOUNDED (a refuter, not a
//! completeness proof) and the independent front end covers only the scalar fragment
//! (compound-sort specs still rely on the shared front end). A fully independent check
//! over ALL inputs (not just probes) is the residual gap.
//! COMPOUND-sort obligations (functions/records/sequences/sets/strings) are out of
//! probe scope and remain bound only by the (translator-trusting) render equality;
//! step (5) returns `None` for them and never forces acceptance. The probe set is a
//! bounded REFUTATION aid (it can find a disagreement, not prove denotation equivalence)
//! so it AUGMENTS, never replaces, step (4). The external gate covers ONLY the three SMT
//! obligations; deadlock-freedom keeps its structural marker (cross-checked by Leg A
//! with `check_deadlock=true`). Without the `ay` feature Leg D cannot run ->
//! Inconclusive, never a false accept.
//!
//! The `sha256` digest is tamper-EVIDENCE for the file; it is NOT a source of
//! soundness (a hash of text proves nothing about the spec). Soundness comes from the
//! AY-confirmed inductive proof, cross-checked by the eval oracle re-deriving from
//! `spec_src` — so a tampered `J` is rejected even if the digest is recomputed.
//!
//! ## Certifiable fragment (completeness limits are explicit, not silent)
//!
//! A certificate is producible only when: (1) the safety conjunction is 1-inductive
//! directly or after interval strengthening; (2) `J` and the obligations lie in the
//! fragment AY strict-verifies — comparisons, equalities, and boolean connectives
//! (`negate_normalized` normalizes `Not(comparison)`/`Not(Bool)`/De Morgan; other
//! shapes fall back to a non-strict `Not(expr)` and are REJECTED, sound but
//! incomplete); and (3) `Next` is cleanly decomposable into guards + total
//! assignments so `Enabled(Next)` can be extracted (undecomposable/disjunctive Next
//! is REFUSED). Widening this fragment is tracked, not hidden behind opaque rejects.

use std::fmt::Write as _;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::check::eval_oracle::{eval_oracle_inductive_safe, InductiveOracleVerdict};
use crate::config::Config;

/// Default bound for the explicit-state eval oracle during certificate
/// verification. Large enough to be a meaningful refuter on typical divergent
/// counters while staying fast; the verdict records whether the bound was hit.
pub const DEFAULT_ORACLE_STATE_BOUND: usize = 4096;

/// The `ty.cert/v1` schema tag.
pub const SCHEMA_V1: &str = "ty.cert/v1";

/// The `serde_stacker` growth parameters for cert (de)serialization — shared by the cert
/// envelope here and the embedded kernel-term codec (`cleancic::expr_to_bytes` /
/// `expr_from_bytes`). The crate's defaults (64 KB red zone / 2 MB growth) are tuned for
/// release-build `serde_json::Value` frames; a DEBUG-build derived deserialize level of a
/// 20+-variant enum threaded through three adapter layers can burn through a 64 KB red zone
/// inside ONE level, overflowing before the next growth check (observed at 10K deep). A 1 MB
/// red zone with 16 MB segments leaves orders of magnitude of headroom per level in either
/// build profile.
pub(crate) const STACKER_RED_ZONE: usize = 1024 * 1024;
pub(crate) const STACKER_GROWTH: usize = 16 * 1024 * 1024;

/// A serialized, self-contained inductive-safety certificate.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq)]
pub struct SafetyCertificate {
    /// Schema tag (`ty.cert/v1`).
    pub schema: String,
    /// The verdict being certified.
    pub verdict: String,
    /// The FULL spec source — makes the certificate self-contained and lets the
    /// re-checker re-derive reachable states independently (Leg A).
    pub spec_src: String,
    /// `INIT` operator name.
    pub init: Option<String>,
    /// `NEXT` operator name.
    pub next: Option<String>,
    /// Configured safety invariants.
    pub invariants: Vec<String>,
    /// The proven inductive invariant `J`, as TLA+ text.
    pub invariant_j_tla: String,
    /// Inferred `(variable, sort)` signature.
    pub var_sorts: Vec<(String, String)>,
    /// Configured `CONSTANT` bindings (name-sorted) — carried so every re-check leg
    /// re-derives with the SAME bindings the certificate was minted under. Empty for
    /// constant-free specs AND absent from the serialized form, so pre-existing cert
    /// digests stay byte-identical (do not remove the `skip_serializing_if`).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub constants: Vec<(String, crate::config::ConstantValue)>,
    /// AY's own re-checkable proof for each obligation (the AY proof-artifact leg).
    /// Empty in a feature-light build; populated by `certify_spec` under `ay`.
    #[serde(default)]
    pub ay_proof_obligations: Vec<AyObligationProof>,
    /// EXPLICIT-STATE lane (alternative to the inductive obligations above): a kernel-checked
    /// fixpoint certificate over the live-enumerated reachable set `R` — `Init⊆R ∧ closed-under-Next ∧
    /// R⊆Safety`, every leg a `clean_kernel::Expr`. Present only when the explicit lane certified (the
    /// symbolic AY prover refused); the re-checker re-runs the Clean kernel on the legs bound to the
    /// embedded `R` (Leg E). Inside the digest body, so tampering breaks the `sha256`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub explicit_fixpoint: Option<crate::explicit_fixpoint_cert::ExplicitFixpointCert>,
    /// `sha256` hex over the canonical body (this field empty during hashing).
    pub digest: String,
}

/// A single obligation's AY proof, embedded in the certificate.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq)]
pub struct AyObligationProof {
    /// `"initiation"`, `"consecution"`, or `"safety"`.
    pub name: String,
    /// Whether AY strict-checked the proof (`check_proof_strict` accepted it).
    pub strict_verified: bool,
    /// Strict-verified AND every step in the clean-supported subset.
    pub clean_supported: bool,
    /// Whether a replayable LRAT SAT-backbone certificate was emitted.
    pub lrat_present: bool,
    /// Rendered Alethe proof text (problem-scoped), for offline audit.
    /// PRESENTATION-ONLY at verification time: the accepted basis is
    /// `bundle_json` (strict re-check + render-binding) plus any kernel leg —
    /// this transcript is NOT re-checked, so treat it as a human-readable aid,
    /// never as evidence (adversarial-verify note 2026-07-06).
    pub alethe: String,
    /// Leg D: the portable, checker-only `SerializableProofBundle` (serde_json)
    /// AY exported for this obligation's UNSAT query. Empty when absent (old
    /// certs / structural deadlock marker / non-ay producer). A covered SMT
    /// obligation whose bundle is empty cannot be externally re-checked, so the
    /// verifier reports Inconclusive (never a false accept). The bundle is inside
    /// the digest body, so tampering with it breaks the `sha256`.
    #[serde(default)]
    pub bundle_json: String,
    /// Leg K (kernel): the `CleanCic` CIC proof term (serde_json of a `clean_kernel::Expr`) this
    /// obligation was kernel-CERTIFIED with — present only for a reflexive obligation built under
    /// the `clean-cic` feature. Empty otherwise. The re-checker re-runs the Clean kernel on it
    /// (the strongest leg: trust base = the small CIC kernel, not the SMT solver). Inside the
    /// digest body, so tampering breaks the `sha256`.
    #[serde(default)]
    pub clean_cic_term: Vec<u8>,
}

impl SafetyCertificate {
    /// Canonical bytes for hashing: the JSON with `digest` blanked. On the (practically impossible)
    /// serialization failure, return a DISTINCT non-JSON sentinel — never an empty body — so a
    /// serialization error can NEVER alias a valid (or valid-empty) canonical body into the same digest;
    /// the re-computed digest of the real cert then won't match, and verification fails closed.
    ///
    /// STACK-SAFE (as `to_json`/`from_json` below): serialized through `serde_stacker` so a
    /// deeply nested embedded field (a big `PredIR`, a deep term) can never overflow the Rust
    /// stack mid-digest. The BYTES are identical to plain `serde_json` output — digests are
    /// untouched (pinned by the fixture back-compat test).
    fn canonical_bytes(&self) -> Vec<u8> {
        let mut clone = self.clone();
        clone.digest = String::new();
        let mut out = Vec::new();
        let mut ser = serde_json::Serializer::new(&mut out);
        let mut adapter = serde_stacker::Serializer::new(&mut ser);
        adapter.red_zone = STACKER_RED_ZONE;
        adapter.stack_size = STACKER_GROWTH;
        match Serialize::serialize(&clone, adapter) {
            Ok(()) => out,
            Err(_) => b"\0__ty canonical_bytes serialization error__\0".to_vec(),
        }
    }

    /// Recompute the `sha256` over the canonical body.
    pub fn compute_digest(&self) -> String {
        let mut hasher = Sha256::new();
        hasher.update(self.canonical_bytes());
        hex_lower(&hasher.finalize())
    }

    /// Serialize to pretty JSON (stack-safe; byte-identical to plain `to_string_pretty`).
    pub fn to_json(&self) -> String {
        let mut out = Vec::new();
        let mut ser = serde_json::Serializer::pretty(&mut out);
        let mut adapter = serde_stacker::Serializer::new(&mut ser);
        adapter.red_zone = STACKER_RED_ZONE;
        adapter.stack_size = STACKER_GROWTH;
        if Serialize::serialize(self, adapter).is_err() {
            return String::new();
        }
        String::from_utf8(out).unwrap_or_default()
    }

    /// Parse from JSON — WITHOUT `serde_json`'s 128-deep recursion cap (the pre-existing wall
    /// that made large embedded certs unparseable) and stack-safely (`serde_stacker` grows the
    /// stack as needed; depth stays bounded by the input length).
    pub fn from_json(s: &str) -> Result<Self, String> {
        let mut de = serde_json::Deserializer::from_str(s);
        de.disable_recursion_limit();
        let mut adapter = serde_stacker::Deserializer::new(&mut de);
        adapter.red_zone = STACKER_RED_ZONE;
        adapter.stack_size = STACKER_GROWTH;
        Self::deserialize(adapter).map_err(|e| format!("certificate parse error: {e}"))
    }

    /// The `Config` this certificate was produced under (init/next/invariants/constants).
    fn reconstructed_config(&self) -> Config {
        Config {
            init: self.init.clone(),
            next: self.next.clone(),
            invariants: self.invariants.clone(),
            constants: self.constants.iter().cloned().collect(),
            constants_order: self.constants.iter().map(|(n, _)| n.clone()).collect(),
            ..Default::default()
        }
    }
}

/// Build a certificate from a proven inductive invariant.
///
/// `invariant_j_tla` is the proven `J` as TLA+ text (e.g. from
/// `tla_core::pretty_expr` on `InductiveProof::invariant_j`); `var_sorts` is the
/// inferred signature. The digest is computed over the finished body.
pub fn build_safety_certificate(
    spec_src: &str,
    config: &Config,
    invariant_j_tla: String,
    var_sorts: Vec<(String, String)>,
) -> SafetyCertificate {
    let mut cert = SafetyCertificate {
        schema: SCHEMA_V1.to_string(),
        verdict: "inductive-safety-safe".to_string(),
        spec_src: spec_src.to_string(),
        init: config.init.clone(),
        next: config.next.clone(),
        invariants: config.invariants.clone(),
        invariant_j_tla,
        var_sorts,
        constants: {
            let mut cs: Vec<_> = config
                .constants
                .iter()
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect();
            cs.sort_by(|a, b| a.0.cmp(&b.0));
            cs
        },
        ay_proof_obligations: Vec::new(),
        explicit_fixpoint: None,
        digest: String::new(),
    };
    cert.digest = cert.compute_digest();
    cert
}

/// Build a serialized `SafetyCertificate` for the EXPLICIT-STATE lane from a kernel-checked
/// [`ExplicitFixpointCert`]. Verdict `explicit-state-fixpoint-safe`; the inductive obligations are
/// empty (this lane does not use an inductive `J`); the fixpoint legs ride in `explicit_fixpoint` and
/// are re-checked by [`verify_safety_certificate`] (Leg E). Self-contained + digest-sealed.
pub fn build_explicit_fixpoint_certificate(
    spec_src: &str,
    config: &Config,
    fixpoint: crate::explicit_fixpoint_cert::ExplicitFixpointCert,
) -> SafetyCertificate {
    let mut cert = SafetyCertificate {
        schema: SCHEMA_V1.to_string(),
        verdict: "explicit-state-fixpoint-safe".to_string(),
        spec_src: spec_src.to_string(),
        init: config.init.clone(),
        next: config.next.clone(),
        invariants: config.invariants.clone(),
        invariant_j_tla: String::new(),
        var_sorts: Vec::new(),
        constants: {
            let mut cs: Vec<_> = config
                .constants
                .iter()
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect();
            cs.sort_by(|a, b| a.0.cmp(&b.0));
            cs
        },
        ay_proof_obligations: Vec::new(),
        explicit_fixpoint: Some(fixpoint),
        digest: String::new(),
    };
    cert.digest = cert.compute_digest();
    cert
}

// `clean_cic_obligation` + `clean_cic_term_for` (the old idx-keyed build of the φ-INDEPENDENT
// polymorphic-identity reflexive term, bound only by lineage digest) were REMOVED with the faithful
// fix: the live build routes reflexive obligations through `clean_cic_initiation_term` /
// `clean_cic_safety_term` (→ `certify_reflexive_faithful`, kernel-bound to `embed(ante)→embed(cons)`)
// and Leg K re-checks via `verify_reflexive_leg` → `verify_reflexive_faithful`.

/// The kernel-CERTIFIED consecution term (`J∧Next⇒J'`): the EUF transitivity engine for equality
/// invariants, else the faithful Int congruence engine for `J(x) ∧ x'=x` invariants. Empty when not
/// strict-verified, outside both fragments, or the kernel rejects (fail-closed).
#[cfg(all(feature = "ay", feature = "clean-cic"))]
fn clean_cic_consecution_term(
    proof: &crate::ay_bmc::InductiveProof,
    strict_verified: bool,
) -> Vec<u8> {
    let gated = if strict_verified {
        crate::cleancic::certify_tla_equality_consecution(&proof.invariant_j.node, &proof.next.node)
            .or_else(|| {
                crate::cleancic::certify_int_congruence(&proof.invariant_j.node, &proof.next.node)
            })
            .or_else(|| {
                crate::cleancic::certify_lia_consecution(&proof.invariant_j.node, &proof.next.node)
            })
    } else {
        None
    };
    // Phase D/E: the COUPLED-AFFINE engine (`b' = b + a`) runs WITHOUT the ay-strict gate — the
    // kernel itself decides the obligation (fail-closed), and this fragment is exactly the one
    // ay's strict single-equality Farkas re-check cannot cover. The kernel term then becomes
    // the obligation's acceptance basis (per-obligation strong coverage in
    // `verify_safety_certificate`), not a strengthening of an ay-strict basis.
    gated
        .or_else(|| {
            crate::cleancic::certify_affine_sum_consecution(
                &proof.invariant_j.node,
                &proof.next.node,
            )
        })
        .unwrap_or_default()
}

#[cfg(all(feature = "ay", not(feature = "clean-cic")))]
fn clean_cic_consecution_term(
    _proof: &crate::ay_bmc::InductiveProof,
    _strict_verified: bool,
) -> Vec<u8> {
    Vec::new()
}

/// The kernel-CERTIFIED initiation term (`Init⇒J`): when `Init≡J` (reflexive), the FAITHFUL reflexive
/// cert kernel-bound to `Π(vars). embed(Init) → embed(J)` (the kernel rejects a non-reflexive
/// obligation — not the φ-independent identity); else the faithful GROUND engine for `x=c ⇒ x≥0`.
/// Empty otherwise (fail-closed).
#[cfg(all(feature = "ay", feature = "clean-cic"))]
fn clean_cic_initiation_term(
    proof: &crate::ay_bmc::InductiveProof,
    strict_verified: bool,
    reflexive: bool,
) -> Vec<u8> {
    // `unwrap_or_default()` → empty bytes when the obligation is outside the embeddable fragment
    // (`embed_prop`/`certify_*` return None) or the kernel rejects: the obligation simply stays at
    // the honest `Discharged` tier, NEVER silently `Certified` (fail-closed).
    if reflexive {
        return crate::cleancic::certify_reflexive_faithful(
            &proof.init.node,
            &proof.invariant_j.node,
        )
        .unwrap_or_default();
    }
    let gated = if strict_verified {
        crate::cleancic::certify_ground_init(&proof.init.node, &proof.invariant_j.node)
    } else {
        None
    };
    // Phase D/E: the CONJUNCTIVE ground engine (`⋀ v=c ⇒ ⋀ v≥0`) is kernel-gated only.
    gated
        .or_else(|| {
            crate::cleancic::certify_conj_ground_init(&proof.init.node, &proof.invariant_j.node)
        })
        .unwrap_or_default()
}

#[cfg(all(feature = "ay", not(feature = "clean-cic")))]
fn clean_cic_initiation_term(
    _proof: &crate::ay_bmc::InductiveProof,
    _strict_verified: bool,
    _reflexive: bool,
) -> Vec<u8> {
    Vec::new()
}

/// The kernel-CERTIFIED safety term (`J⇒Safety`): when `J≡Safety` (reflexive), the FAITHFUL reflexive
/// cert kernel-bound to `Π(vars). embed(J) → embed(Safety)`. Empty otherwise (fail-closed).
#[cfg(all(feature = "ay", feature = "clean-cic"))]
fn clean_cic_safety_term(proof: &crate::ay_bmc::InductiveProof, reflexive: bool) -> Vec<u8> {
    if reflexive {
        return crate::cleancic::certify_reflexive_faithful(
            &proof.invariant_j.node,
            &proof.safety.node,
        )
        .unwrap_or_default();
    }
    // Phase D/E: a STRENGTHENED conjunctive J (`J = Safety ∧ …`) proves Safety by And-projection.
    crate::cleancic::certify_conj_projection_safety(&proof.invariant_j.node, &proof.safety.node)
        .unwrap_or_default()
}

#[cfg(all(feature = "ay", not(feature = "clean-cic")))]
fn clean_cic_safety_term(_proof: &crate::ay_bmc::InductiveProof, _reflexive: bool) -> Vec<u8> {
    Vec::new()
}

/// Obligation-aware FAITHFUL reflexive re-check: re-derive `(ante, cons)` from the cert's spec
/// (`Init,J` for initiation; `J,Safety` for safety) and require the Clean kernel to accept the term
/// at `Π(vars). embed(ante) → embed(cons)`. A term for a different/non-reflexive obligation is
/// rejected BY THE KERNEL — closing the spec-blind / cross-spec-replay holes. Fail-closed.
#[cfg(all(feature = "ay", feature = "clean-cic"))]
fn verify_reflexive_leg(cert: &SafetyCertificate, name: &str, term: &[u8]) -> bool {
    let config = Config {
        init: cert.init.clone(),
        next: cert.next.clone(),
        invariants: cert.invariants.clone(),
        ..Default::default()
    };
    let Some(inputs) =
        crate::ay_bmc::rederive_obligation_inputs(&cert.spec_src, &config, &cert.invariant_j_tla)
    else {
        return false;
    };
    let (ante, cons) = match name {
        "initiation" => (&inputs.init.node, &inputs.j.node),
        "safety" => (&inputs.j.node, &inputs.safety.node),
        _ => return false,
    };
    crate::cleancic::verify_reflexive_faithful(ante, cons, term)
}

/// Leg K — re-run the Clean CIC kernel on every embedded `clean_cic_term`. Three-valued:
/// `Some(true)` = at least one kernel term present and ALL kernel-re-checked; `Some(false)` = a
/// present term failed the kernel (tampered/invalid) → forces Rejected; `None` = no kernel terms
/// (or the `clean-cic` feature is off) → this leg contributes nothing (never a false accept).
fn verify_leg_k(cert: &SafetyCertificate) -> Option<bool> {
    #[cfg(feature = "clean-cic")]
    {
        // (under `clean-cic` without `ay`, every `any = true` is in an `#[cfg(ay)]` block)
        #[allow(unused_mut)]
        let mut any = false;
        for o in cert.ay_proof_obligations.iter() {
            if o.clean_cic_term.is_empty() {
                continue;
            }
            // The consecution (`J∧Next⇒J'`) carries an EUF proof: re-DERIVE its obligation type
            // from (J, Next) and re-run the kernel against it (obligation-aware, needs ay).
            if o.name == "consecution" {
                #[cfg(feature = "ay")]
                {
                    any = true;
                    if !verify_consecution_term(cert, &o.clean_cic_term) {
                        return Some(false);
                    }
                }
                // Without ay we cannot re-derive J/Next — skip this term (not a refutation).
                continue;
            }
            // Initiation (`Init⇒J`): FAITHFUL reflexive (kernel re-derives embed(Init)→embed(J) and
            // rejects a non-reflexive obligation) OR faithful ground (`x=c⇒x≥0`). Needs ay to
            // re-derive; without ay we cannot re-derive — skip (not a refutation).
            if o.name == "initiation" {
                #[cfg(feature = "ay")]
                {
                    any = true;
                    let ok = verify_reflexive_leg(cert, "initiation", &o.clean_cic_term)
                        || verify_initiation_ground(cert, &o.clean_cic_term);
                    if !ok {
                        return Some(false);
                    }
                }
                continue;
            }
            // Safety (`J⇒Safety`): FAITHFUL reflexive (kernel re-derives embed(J)→embed(Safety)).
            if o.name == "safety" {
                #[cfg(feature = "ay")]
                {
                    any = true;
                    let ok = verify_reflexive_leg(cert, "safety", &o.clean_cic_term)
                        || verify_safety_conj(cert, &o.clean_cic_term);
                    if !ok {
                        return Some(false);
                    }
                }
                continue;
            }
            // Any other obligation carrying a term is unexpected; do not false-accept on it.
        }
        return if any { Some(true) } else { None };
    }
    #[cfg(not(feature = "clean-cic"))]
    {
        let _ = cert;
        None
    }
}

/// Obligation-aware re-check of a consecution EUF proof term: re-DERIVE `(J, Next)` from the cert
/// (`J` = the cert's claimed invariant, `Next` from the embedded spec) and require the Clean kernel
/// to accept the term at the freshly-built EUF obligation type. So a term proving some OTHER
/// proposition is rejected. Fail-closed (cannot re-derive / kernel rejects → false). No SMT.
#[cfg(all(feature = "ay", feature = "clean-cic"))]
fn verify_consecution_term(cert: &SafetyCertificate, term: &[u8]) -> bool {
    let config = Config {
        init: cert.init.clone(),
        next: cert.next.clone(),
        invariants: cert.invariants.clone(),
        ..Default::default()
    };
    match crate::ay_bmc::rederive_obligation_inputs(&cert.spec_src, &config, &cert.invariant_j_tla)
    {
        Some(inputs) => {
            crate::cleancic::verify_tla_equality_consecution(
                &inputs.j.node,
                &inputs.next.node,
                term,
            ) || crate::cleancic::verify_int_congruence(&inputs.j.node, &inputs.next.node, term)
                || crate::cleancic::verify_lia_consecution(&inputs.j.node, &inputs.next.node, term)
                || crate::cleancic::verify_affine_sum_consecution(
                    &inputs.j.node,
                    &inputs.next.node,
                    term,
                )
        }
        None => false,
    }
}

/// Obligation-aware re-check of a GROUND initiation term (`x=c ⇒ x≥0`): re-DERIVE `(Init, J)` from
/// the cert and require the kernel to accept the term at the freshly-built type. Fail-closed.
#[cfg(all(feature = "ay", feature = "clean-cic"))]
fn verify_initiation_ground(cert: &SafetyCertificate, term: &[u8]) -> bool {
    let config = Config {
        init: cert.init.clone(),
        next: cert.next.clone(),
        invariants: cert.invariants.clone(),
        ..Default::default()
    };
    match crate::ay_bmc::rederive_obligation_inputs(&cert.spec_src, &config, &cert.invariant_j_tla)
    {
        Some(inputs) => {
            crate::cleancic::verify_ground_init(&inputs.init.node, &inputs.j.node, term)
                || crate::cleancic::verify_conj_ground_init(&inputs.init.node, &inputs.j.node, term)
        }
        None => false,
    }
}

/// Obligation-aware re-check of a CONJUNCTIVE safety projection term (`J = Safety ∧ … ⇒ Safety`):
/// re-DERIVE `(J, Safety)` from the cert and require the kernel to accept the term at the
/// freshly-built projection type. Fail-closed.
#[cfg(all(feature = "ay", feature = "clean-cic"))]
fn verify_safety_conj(cert: &SafetyCertificate, term: &[u8]) -> bool {
    let config = Config {
        init: cert.init.clone(),
        next: cert.next.clone(),
        invariants: cert.invariants.clone(),
        ..Default::default()
    };
    match crate::ay_bmc::rederive_obligation_inputs(&cert.spec_src, &config, &cert.invariant_j_tla)
    {
        Some(inputs) => crate::cleancic::verify_conj_projection_safety(
            &inputs.j.node,
            &inputs.safety.node,
            term,
        ),
        None => false,
    }
}

/// The three-valued outcome of re-checking a certificate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CertVerdict {
    /// All required legs passed: the certificate is a genuine proof.
    Accepted,
    /// A leg definitively REFUTED the certificate (a reachable violation/deadlock,
    /// an obligation SAT, a digest mismatch, a forged/mis-bound proof bundle, or a
    /// tampered kernel term). An honest-but-non-strict bundle is NOT a refutation —
    /// its obligation must instead be covered by a kernel term or acceptance is
    /// withheld (Inconclusive).
    Rejected,
    /// The re-checker could NOT determine a verdict — the AY proof leg was
    /// unavailable (the `ay` feature is off) or the obligations were not
    /// re-derivable (e.g. an undecomposable Next). Fail-closed (never an accept) but
    /// distinct from a definitive Rejected.
    Inconclusive,
}

/// The outcome of independently re-checking a certificate.
#[derive(Debug, Clone)]
pub struct VerifyReport {
    /// Whether the stored `sha256` matches the recomputed digest (tamper-evidence).
    pub digest_ok: bool,
    /// Leg A — the explicit-state eval oracle verdict (the soundness spine).
    pub oracle: InductiveOracleVerdict,
    /// Leg C — the AY re-discharge of the three obligations (strengthening, not the
    /// spine: it shares the producer's translator). `Some(true)` confirmed,
    /// `Some(false)` refuted (forces rejection), `None` not run / not re-derivable.
    pub ay_redischarge: Option<bool>,
    /// Leg K — the Clean CIC kernel re-check of the embedded `CleanCic` terms (the strongest tier).
    /// `Some(true)` = kernel terms present and ALL re-accepted by the kernel; `Some(false)` = a
    /// present term failed the kernel (forces rejection); `None` = no kernel terms / `clean-cic`
    /// feature off. Since the Phase-D/E per-obligation acceptance, a kernel term re-accepted at
    /// the obligation type REBUILT from the spec is a valid ACCEPTANCE BASIS for its obligation
    /// (reflexive, ground, congruence/EUF/LIA, coupled-affine, and conjunctive engines).
    pub kernel_recheck: Option<bool>,
    /// Whether the certificate is ACCEPTED (all required legs hold). Equivalent to
    /// `verdict == CertVerdict::Accepted`.
    pub accepted: bool,
    /// Three-valued verdict (Accepted / Rejected / Inconclusive).
    pub verdict: CertVerdict,
    /// Human-readable summary.
    pub detail: String,
}

/// Independently re-validate a certificate, AY-only.
///
/// Accepts iff the schema is recognized, the digest matches, AND Leg A (the
/// explicit-state eval oracle) reports `NoViolation` over its bound. A `Refuted`
/// or `Inconclusive` oracle is never accepted. Note that the eval oracle re-derives
/// reachable states from `spec_src`, so a tampered `J` is rejected even when the
/// digest was recomputed to match.
///
/// Leg D — the genuinely-external proof re-check (the new acceptance BASIS for
/// the ay-proof-backed part of the gate). For each of the THREE SMT obligations
/// (initiation `Init /\ ~J`, consecution `J /\ Next /\ ~J'`, safety
/// `J /\ ~Safety`) the certificate embeds a `SerializableProofBundle`. The
/// verifier (1) deserializes it (fail-closed on schema/parse), (2) requires the
/// bundle's canonical-rendered assertion multiset to EQUAL the multiset TY
/// produces by re-translating the SAME obligation through its own `BmcTranslator`
/// WITHOUT solving (a loud MISMATCH is forgery — `Some(false)`), (3) calls
/// `ay_proof::re_check_bundle_strict`, which rebuilds a CHECKER-ONLY `TermStore`
/// and runs `check_proof_strict` — proving the embedded assertion set UNSAT via
/// AY's audited checker with NO solver search, (4) assume-coverage: the proof's
/// `Assume` axiom set must be a SUBSET of the bound `obligation_assertions` (each
/// assumed literal is verbatim an obligation assertion, so the obligation entails
/// it and UNSAT-of-the-assumes discharges the obligation). If the solver
/// PRE-PROCESSED the problem (e.g. the new ay-sat eliminating definitionally-fixed
/// primed vars into a mirrored literal), the assume set is no longer a syntactic
/// subset and leg-D cannot re-derive the transform without solving — it yields
/// per-obligation `None` (defer to the kernel term / withhold), never a false
/// accept — and (5) for SCALAR specs ENGINE-DIVERSELY confirms the embedded
/// obligation denotes the spec via `tla-eval` probe states (a different engine than
/// `BmcTranslator`, catching a translation/negation bug step 2 would share). (2)+(4)
/// +(5) bind the audited proof to the obligation TY recognizes. Returns `Some(true)`
/// iff all three pass, `Some(false)` on a definitive refutation, `None` when not
/// runnable (feature off / not re-derivable / a covered obligation lacks a bundle /
/// the proof's assumes were solver-preprocessed out of the obligation's literals).
///
/// HONEST SCOPE: Leg D removes the producer's SOLVER from the trust base for the
/// three SMT obligations (acceptance no longer rests on re-running AY's search,
/// only on re-checking an embedded proof with AY's audited checker), so it is
/// producer-SOLVER-INDEPENDENT. It is NOT yet fully third-party machine-checkable:
/// step (2) still re-translates via TY's OWN front end + TLA->AY translator. The
/// fourth obligation, deadlock-freedom, has no SMT proof for the unguarded
/// total-Next case (structural marker); Leg D does not cover it — it stays
/// cross-checked by the deadlock-aware eval oracle.
/// Per-obligation Leg-D outcomes: `Some(true)` externally re-checked + bound, `Some(false)`
/// definitively refuted, `None` not runnable for THAT obligation (missing bundle / obligation
/// not re-translatable). Outer `None` = the shared setup failed (inputs not re-derivable).
#[cfg(feature = "ay")]
fn verify_leg_d_detail(cert: &SafetyCertificate) -> Option<Vec<(&'static str, Option<bool>)>> {
    use crate::ay_bmc::SmtObligation;
    use tla_ay::{
        re_check_bundle_strict, render_term_canonical, SerializableProofBundle, TermStore,
        PROOF_BUNDLE_SCHEMA,
    };

    // Re-derive the SAME obligation inputs the producer/Leg-C path used, via TY's
    // front end + translator pipeline (NOT its solver). `None` => not re-derivable
    // => Inconclusive, never accept.
    let config = cert.reconstructed_config();
    let inputs =
        crate::ay_bmc::rederive_obligation_inputs(&cert.spec_src, &config, &cert.invariant_j_tla)?;

    // Engine-diverse binding (part-2): a tla-eval context (state VARIABLES
    // registered) for cross-checking the embedded obligation against the spec by
    // a DIFFERENT engine than BmcTranslator. `None` => probe check is skipped
    // (the render-only binding still applies); it never forces acceptance.
    let probe_ctx =
        crate::ay_bmc::build_probe_eval_ctx(&cert.spec_src, &config, &cert.invariant_j_tla);

    // THIRD, fully INDEPENDENT front end (parser+evaluator sharing nothing with
    // tla_core or BmcTranslator). `None` for non-scalar / out-of-fragment specs ->
    // the indep gate is simply absent (the other bindings still apply).
    let indep_spec = match (config.init.as_deref(), config.next.as_deref()) {
        (Some(init_name), Some(next_name)) => crate::cert_indep_frontend::IndepSpec::parse(
            &cert.spec_src,
            init_name,
            next_name,
            &config.invariants,
            &cert.invariant_j_tla,
        ),
        _ => None,
    };

    let mut out: Vec<(&'static str, Option<bool>)> = Vec::new();
    for ob in [
        SmtObligation::Initiation,
        SmtObligation::Consecution,
        SmtObligation::Safety,
    ] {
        let outcome: Option<bool> = 'ob: {
            // Locate this obligation's embedded proof. A covered obligation with no
            // embedded bundle is not externally re-checkable -> None for THIS obligation.
            let Some(emb) = cert
                .ay_proof_obligations
                .iter()
                .find(|o| o.name == ob.name())
            else {
                break 'ob None;
            };
            if emb.bundle_json.is_empty() {
                break 'ob None;
            }

            // (1) Deserialize; fail-CLOSED on parse / schema-version mismatch.
            let bundle: SerializableProofBundle = match serde_json::from_str(&emb.bundle_json) {
                Ok(b) => b,
                Err(_) => break 'ob Some(false),
            };
            if bundle.schema != PROOF_BUNDLE_SCHEMA {
                break 'ob Some(false);
            }

            // (2) OBLIGATION BINDING FIRST (LOAD-BEARING, runs regardless of the proof's
            // step quality): bind the asserted problem to TY's INTENDED obligation at the
            // AY-term level, via TY's translator (not its solver). Multiset A (embedded,
            // rendered store-independently) == multiset B (TY's no-solve re-translation of
            // the same obligation). A MISMATCH is forgery evidence — the bundle asserts a
            // DIFFERENT problem than this obligation — and hard-rejects even when the
            // grafted proof's steps are also corrupted (ordering this before the strict
            // re-check keeps that tamper evidence loud).
            let mut embedded_store = TermStore::from_entries(
                bundle.term_entries.clone(),
                bundle.true_term,
                bundle.false_term,
                bundle.var_counter,
            );
            let mut a: Vec<String> = bundle
                .obligation_assertions
                .iter()
                .map(|&id| render_term_canonical(&embedded_store, id))
                .collect();
            let Some(mut b) = crate::ay_bmc::retranslate_obligation_canonical(ob, &inputs) else {
                break 'ob None;
            };
            a.sort();
            b.sort();
            if a != b {
                break 'ob Some(false);
            }

            // (3) AY-checker re-check (terminal empty clause enforced inside). NO
            // solver search runs — only check_proof_strict over the embedded steps.
            // `re_check_bundle_strict` returns Ok ONLY when every step strict-verified;
            // the `is_complete` gate double-checks the proof is trust/hole-free.
            // A failure HERE — on a bundle whose ASSERTIONS were just bound to the right
            // obligation — means "this bundle is not strict evidence" (e.g. the producer's
            // proof legitimately contains trust-demoted steps for an obligation outside
            // ay's strict fragment): per-obligation None, NOT a refutation. The obligation
            // must then be carried by its kernel term (per-obligation strong coverage) or
            // acceptance is withheld.
            let recheck = match re_check_bundle_strict(&bundle) {
                Ok(r) => r,
                Err(_) => break 'ob None,
            };
            if !recheck.quality.is_complete() {
                break 'ob None;
            }

            // (4) Assume-coverage (SET semantics, sound under solver PRE-PROCESSING).
            // The strict proof derives the empty clause from its `Assume` axiom set
            // `S = proof_assumes`. For that to discharge the obligation `O = asserted`
            // (which step (2) already bound, at the AY-term level, to TY's own
            // re-translation), we need `O ⊨ s` for every `s ∈ S` — then `O ⊨ S ⊨ ⊥`,
            // so `O` is UNSAT. leg-D verifies this WITHOUT a solver via SYNTACTIC
            // CONTAINMENT `S ⊆ O`: every assumed literal is verbatim one of the bound
            // obligation's assertions, so `O ⊨ s` holds trivially. Exact equality is
            // the common case; a strict SUBSET (the proof ignored a redundant
            // assertion) is equally sound — UNSAT of a subset implies UNSAT of `O`.
            //
            // When `S ⊄ O` — the proof assumes a literal that is NOT verbatim in `O` —
            // leg-D CANNOT cheaply certify that the extra literal is a consequence of
            // `O`. This is exactly what the new ay-sat's incremental fixed-literal
            // mirroring produces: it eliminates the definitionally-fixed primed
            // variables, rewriting e.g. `(a'=a+1 ∧ b'=b+a) ∧ (b'<0 ∨ a'<0)` into the
            // single mirrored literal `(b+a<0 ∨ a+1<0)` — a SOUND transform, but one
            // leg-D would have to re-solve to reproduce. So leg-D must NOT vouch for
            // this obligation on the strength of this bundle: it yields a per-obligation
            // `None` (deferring to the obligation's kernel term or withholding
            // acceptance), NOT a refutation. This is FAIL-SAFE: an unbound leg-D proof
            // can never be an acceptance basis (per-obligation strong coverage requires
            // leg-D `Some(true)` OR a kernel-checked term), so no forged/mis-bound proof
            // is accepted here. Loud forgery — a bundle asserting a DIFFERENT obligation
            // — is still caught as `Some(false)` by the step-(2) render binding above.
            let mut proof_assumes = recheck.assume_terms.clone();
            let mut asserted = bundle.obligation_assertions.clone();
            proof_assumes.sort();
            proof_assumes.dedup();
            asserted.sort();
            asserted.dedup();
            if !proof_assumes.iter().all(|t| asserted.contains(t)) {
                break 'ob None;
            }

            // (5) ENGINE-DIVERSE binding (closes the translator trust): for scalar
            // specs, confirm the embedded AY obligation denotes the spec via tla-eval
            // probes — a DIFFERENT engine than BmcTranslator. A disagreement REJECTS
            // (a translation bug caught). `None` (non-scalar / unfoldable / eval
            // error) leaves the render-only binding above as the verdict — never an
            // accept on its own. Runs AFTER the render compare (which borrows the
            // store immutably); the probe folds constants into the store.
            if let Some(ref pctx) = probe_ctx {
                match crate::ay_bmc::probe_check_obligation_engine_diverse(
                    ob,
                    &mut embedded_store,
                    &bundle.obligation_assertions,
                    &inputs,
                    pctx,
                    indep_spec.as_ref(),
                ) {
                    Some(false) => break 'ob Some(false),
                    Some(true) | None => {}
                }
            }
            Some(true)
        };
        out.push((ob.name(), outcome));
    }
    Some(out)
}

#[cfg(not(feature = "ay"))]
fn verify_leg_d(_cert: &SafetyCertificate) -> Option<bool> {
    None
}

/// Independently re-checks a serialized safety certificate and reports the verdict.
///
/// Re-check an EXPLICIT-STATE fixpoint certificate (Leg E): independently RE-ENUMERATE `R` from the
/// spec (binding the embedded `R` to THIS spec) and RE-RUN the Clean kernel on all three legs. Accepts
/// iff `digest_ok` ∧ `R` re-derives identically ∧ every leg re-checks. Fail-closed: without `clean-cic`
/// it is Inconclusive; a mismatched `R` or a failing leg is Rejected.
#[cfg(feature = "clean-cic")]
fn verify_explicit_fixpoint_report(
    cert: &SafetyCertificate,
    fp: &crate::explicit_fixpoint_cert::ExplicitFixpointCert,
    digest_ok: bool,
) -> VerifyReport {
    let config = cert.reconstructed_config();
    // (1) Independently re-enumerate R from the spec and require an identical fixpoint (binds R to spec).
    let matches_rederivation = |re: &crate::explicit_fixpoint_cert::ExplicitFixpointCert| {
        re.reachable == fp.reachable
            && re.init_values == fp.init_values
            && re.image == fp.image
            // Bind the per-column SORT vector to the spec too: otherwise a cert could mislabel a Bool
            // column as Int (or vice-versa) and have the kernel prove a mis-typed `≥0` leg the spec
            // does not actually assert. The sorts are re-inferred from the spec here, so require equality.
            && re.sorts == fp.sorts
            // Bind the recognized affine `Next` shape + literal `Init` set: the kernel-re-evaluated
            // completeness legs are re-run by `verify_explicit_state_cert` below, but the SHAPES they
            // re-evaluate must be the ones re-derived from the spec (a cert cannot claim a different
            // counter or init set than the spec's).
            // LEG-SUBSET TOLERANCE: a cert may carry FEWER kernel legs than a fresh
            // re-derivation produces (certs minted before a bound-rule widening, or under
            // `--require-domain-complete`). A MISSING leg (`fp.X == None`) is a WEAKER claim,
            // never unsound — `verify_explicit_state_cert` requires nothing for an absent leg
            // and the tier labels are derived from the CERT's contents, so an absent leg never
            // yields an enumerator-free claim. A PRESENT leg must still EQUAL the re-derived
            // one — a cert can never claim a DIFFERENT shape/predicate/bound than the spec's.
            && (fp.next_shape.is_none() || re.next_shape == fp.next_shape)
            && (fp.init_shape.is_none() || re.init_shape == fp.init_shape)
            // Bind the GENERAL predicate IR + its derived domain bound `H` too: the kernel-re-evaluated
            // general-completeness legs (re-run by `verify_explicit_state_cert`) must re-evaluate the IR
            // re-recognized from the spec over the domain re-derived from the spec — a cert cannot claim
            // a different `Next`/`Init` predicate (or a slacker bound) than the spec's. `re.next_pred`
            // carries both the IR and the spec-derived `H`, so equality re-binds the domain derivation.
            && (fp.next_pred.is_none() || re.next_pred == fp.next_pred)
            && (fp.init_pred.is_none() || re.init_pred == fp.init_pred)
            // Bind the UNBOUNDED parametric inductive-invariant witness to the spec: re-recognizing the
            // unbounded affine shape `(c, δ)` from the spec must reproduce the SAME witness (a cert
            // cannot claim a different counter/init than the spec's; the legs are deterministic in
            // `(c, δ)`, so struct equality re-binds `(c, δ)` AND the kernel-checked terms).
            && re.unbounded_invariant == fp.unbounded_invariant
            // Bind the GENERAL `R⊆Safety` invariant IR to the spec — STRICT equality, deliberately
            // NOT the leg-subset tolerance above: unlike the optional completeness legs, the safety
            // leg is LOAD-BEARING. An absent `safety_pred` is only sound when the re-derivation also
            // has none (the spec's invariant IS the conjunctive-nonneg shape the tuple `safety_term`
            // proves); a present one must be EXACTLY the invariant re-recognized from the re-parsed
            // spec (`verify_explicit_state_cert` kernel-checks the stored IR over R, but a WIDER
            // invariant than the spec's would still reduce true — this equality closes that gap).
            && re.safety_pred == fp.safety_pred
            // Bind the deadlock-freedom leg's witnesses to the spec: a re-enumeration reproduces the
            // SAME first-successor witness per reachable state (deterministic), so a present leg must
            // EQUAL the re-derived one — a tampered cert cannot substitute spurious witnesses (they are
            // ALSO kernel-re-checked in `verify_explicit_state_cert`). LEG-SUBSET TOLERANCE: an ABSENT
            // leg (`fp.deadlock_free == None`: enumerator-assisted tier, `--no-deadlock` mint, or a
            // cert minted before this leg existed) is a weaker claim, never unsound — the tier label is
            // derived from the cert's contents, so an absent leg never yields a kernel deadlock claim.
            && (fp.deadlock_free.is_none() || re.deadlock_free == fp.deadlock_free)
    };
    let rederived =
        crate::explicit_fixpoint_cert::certify_explicit_state_spec(&cert.spec_src, &config);
    let mut r_matches = rederived.as_ref().map(|re| matches_rederivation(re));
    // A cert minted under `--require-domain-complete` (Phase A strict mode) carries FEWER
    // completeness legs than the default re-derivation. It is still spec-bound: re-derive in
    // strict mode too and accept if THAT reproduces the cert (the strict derivation is a
    // sound subset — declining legs never weakens the enumerated-leg acceptance basis).
    if r_matches == Some(false) {
        if let Some(strict) =
            crate::explicit_fixpoint_cert::certify_explicit_state_spec_strict_domain(
                &cert.spec_src,
                &config,
            )
        {
            if matches_rederivation(&strict) {
                r_matches = Some(true);
            }
        }
    }
    // (2) Re-run the kernel on the embedded legs (trust base = the small kernel).
    let kernel_ok = crate::explicit_fixpoint_cert::verify_explicit_state_cert(fp);
    let accepted = digest_ok && r_matches == Some(true) && kernel_ok;
    let (verdict, detail) = if accepted {
        // Transparency (2026-07-08): distinguish the safety-only certificate from one that ALSO
        // carries a kernel-corroborated deadlock-freedom leg. The `deadlock_free` leg (present on
        // the enumerator-free tier, absent on `--no-deadlock`/assisted mints — a strictly weaker
        // claim) IS re-derived and kernel-re-checked here (see the `deadlock_free` binding above),
        // so the reader must be able to tell which guarantee they hold — "all 3 legs" both
        // under-counts and hides the deadlock status.
        let deadlock_line = if fp.deadlock_free.is_some() {
            " + the deadlock-freedom witnesses (every reachable state has a kernel-verified successor)"
        } else {
            "; deadlock-freedom is NOT part of this certificate (safety only)"
        };
        (
            CertVerdict::Accepted,
            format!(
                "ACCEPTED (explicit-state fixpoint): R re-derived from spec ({} states), the 3 safety \
                 legs (Init⊆R, closure, R⊆Safety){deadlock_line} KERNEL-re-checked",
                fp.reachable.len()
            ),
        )
    } else if !digest_ok {
        (
            CertVerdict::Rejected,
            "REJECTED: digest mismatch (tampered)".to_string(),
        )
    } else if !kernel_ok {
        (
            CertVerdict::Rejected,
            "REJECTED: a fixpoint leg failed the Clean kernel re-check".to_string(),
        )
    } else if r_matches == Some(false) {
        (
            CertVerdict::Rejected,
            "REJECTED: embedded R is not the spec's enumerated reachable set".to_string(),
        )
    } else {
        (
            CertVerdict::Inconclusive,
            "INCONCLUSIVE: could not independently re-derive R from the spec".to_string(),
        )
    };
    VerifyReport {
        digest_ok,
        oracle: InductiveOracleVerdict::Inconclusive {
            reason: "explicit-state lane (kernel-based, no SMT oracle)".to_string(),
        },
        ay_redischarge: None,
        kernel_recheck: Some(kernel_ok),
        accepted,
        verdict,
        detail,
    }
}

#[cfg(not(feature = "clean-cic"))]
fn verify_explicit_fixpoint_report(
    _cert: &SafetyCertificate,
    _fp: &crate::explicit_fixpoint_cert::ExplicitFixpointCert,
    digest_ok: bool,
) -> VerifyReport {
    VerifyReport {
        digest_ok,
        oracle: InductiveOracleVerdict::Inconclusive {
            reason: "explicit-state lane needs the clean-cic kernel to re-check".to_string(),
        },
        ay_redischarge: None,
        kernel_recheck: None,
        accepted: false,
        verdict: CertVerdict::Inconclusive,
        detail: "INCONCLUSIVE: explicit-state fixpoint cert requires the `clean-cic` feature to re-check"
            .to_string(),
    }
}

/// Verify a MULTI-BRANCH consecution bundle (the blocker-2 McCarthy close): the
/// obligation's `bundle_json` is a JSON ARRAY of per-branch proof bundles, one
/// per equality-partition branch `{i*=p*, i*≠p*}`. Each branch bundle is
/// independently re-checked strict (trust/hole-free), assume-covered (its proof
/// axioms ⊆ its asserted obligation), and render-bound to the matching branch's
/// canonical re-translation `branch_assertions[k]` (the anti-forgery gate). Ok
/// iff EVERY branch verifies. SOUND: the branches partition the consecution
/// EXHAUSTIVELY (`⋀ branch UNSAT ⟹ consecution UNSAT`), so the conjunctive "all
/// branches" requirement is the correct discharge — and the render binding pins
/// each branch bundle to TY's own re-derivation, so a forged/mis-bound branch
/// (asserting a different problem) is rejected exactly as in the single-bundle path.
#[cfg(feature = "ay")]
pub(crate) fn verify_multibranch_consecution(
    bundle_json: &str,
    branch_assertions: &[Vec<String>],
) -> Result<(), String> {
    use tla_ay::{
        re_check_bundle_strict, render_term_canonical, SerializableProofBundle, TermStore,
        PROOF_BUNDLE_SCHEMA,
    };
    // `bundle_json` is a JSON array of per-branch bundle JSON STRINGS (each the
    // serde form of one `SerializableProofBundle`), matching the mint's
    // `serde_json::to_string(&Vec<String>)`.
    let branch_jsons: Vec<String> = serde_json::from_str(bundle_json)
        .map_err(|_| "consecution multi-branch bundle parse error".to_string())?;
    if branch_jsons.len() != branch_assertions.len() {
        return Err(format!(
            "consecution branch count mismatch ({} bundles vs {} re-translated)",
            branch_jsons.len(),
            branch_assertions.len()
        ));
    }
    for (bj, expected) in branch_jsons.iter().zip(branch_assertions) {
        let bundle: SerializableProofBundle = serde_json::from_str(bj)
            .map_err(|_| "consecution branch bundle parse error".to_string())?;
        if bundle.schema != PROOF_BUNDLE_SCHEMA {
            return Err("consecution branch bundle schema mismatch".to_string());
        }
        let recheck = re_check_bundle_strict(&bundle)
            .map_err(|_| "consecution branch failed strict re-check".to_string())?;
        if !recheck.quality.is_complete() {
            return Err("consecution branch proof not trust/hole-free".to_string());
        }
        let assume: std::collections::BTreeSet<u32> =
            recheck.assume_terms.iter().map(|t| t.0).collect();
        let oblig: std::collections::BTreeSet<u32> =
            bundle.obligation_assertions.iter().map(|t| t.0).collect();
        if !assume.is_subset(&oblig) {
            return Err("consecution branch uses an axiom outside it".to_string());
        }
        let store = TermStore::from_entries(
            bundle.term_entries.clone(),
            bundle.true_term,
            bundle.false_term,
            bundle.var_counter,
        );
        let mut a: Vec<String> = bundle
            .obligation_assertions
            .iter()
            .map(|&id| render_term_canonical(&store, id))
            .collect();
        let mut b = expected.clone();
        a.sort();
        b.sort();
        if a != b {
            return Err("consecution branch proof does not match its re-translation".to_string());
        }
    }
    Ok(())
}

/// Verify a MULTI-CASE proof bundle: `bundle_json` is a JSON array of per-case
/// bundle JSON strings (the mint's `serde_json::to_string(&Vec<String>)`), and
/// `case_assertions[k]` is the caller's canonical re-translation of case `k`. Each
/// case bundle must strict-re-check UNSAT, be trust/hole-free, use no axiom outside
/// its own asserted obligation, and render-bind to `case_assertions[k]`. Ok iff
/// EVERY case verifies AND the counts match (so a dropped case is rejected). This
/// is the shape-agnostic core of [`verify_multibranch_consecution`], reused for the
/// disjunctive-deadlock DNF coverage: the cases EXHAUSTIVELY cover the obligation
/// (`⋀ case UNSAT ⟹ obligation UNSAT`), so "all cases verify" is the correct
/// discharge and the render binding pins each case bundle to TY's own re-derivation.
#[cfg(feature = "ay")]
pub(crate) fn verify_multicase_bundle(
    what: &str,
    bundle_json: &str,
    case_assertions: &[Vec<String>],
) -> Result<(), String> {
    use tla_ay::{
        re_check_bundle_strict, render_term_canonical, SerializableProofBundle, TermStore,
        PROOF_BUNDLE_SCHEMA,
    };
    let case_jsons: Vec<String> = serde_json::from_str(bundle_json)
        .map_err(|_| format!("{what} multi-case bundle parse error"))?;
    if case_jsons.len() != case_assertions.len() {
        return Err(format!(
            "{what} case count mismatch ({} bundles vs {} re-translated)",
            case_jsons.len(),
            case_assertions.len()
        ));
    }
    for (bj, expected) in case_jsons.iter().zip(case_assertions) {
        let bundle: SerializableProofBundle =
            serde_json::from_str(bj).map_err(|_| format!("{what} case bundle parse error"))?;
        if bundle.schema != PROOF_BUNDLE_SCHEMA {
            return Err(format!("{what} case bundle schema mismatch"));
        }
        let recheck = re_check_bundle_strict(&bundle)
            .map_err(|_| format!("{what} case failed strict re-check"))?;
        if !recheck.quality.is_complete() {
            return Err(format!("{what} case proof not trust/hole-free"));
        }
        let assume: std::collections::BTreeSet<u32> =
            recheck.assume_terms.iter().map(|t| t.0).collect();
        let oblig: std::collections::BTreeSet<u32> =
            bundle.obligation_assertions.iter().map(|t| t.0).collect();
        if !assume.is_subset(&oblig) {
            return Err(format!("{what} case uses an axiom outside it"));
        }
        let store = TermStore::from_entries(
            bundle.term_entries.clone(),
            bundle.true_term,
            bundle.false_term,
            bundle.var_counter,
        );
        let mut a: Vec<String> = bundle
            .obligation_assertions
            .iter()
            .map(|&id| render_term_canonical(&store, id))
            .collect();
        let mut b = expected.clone();
        a.sort();
        b.sort();
        if a != b {
            return Err(format!(
                "{what} case proof does not match its re-translation"
            ));
        }
    }
    Ok(())
}

/// Re-check a serialized [`SafetyCertificate`] independently. Validates the digest, dispatches the
/// EXPLICIT-STATE lane to the kernel-based fixpoint re-check, otherwise re-runs the inductive oracle
/// and (with `ay`) the Leg-D SMT re-discharge. The returned [`VerifyReport`] distinguishes acceptance
/// from rejection with diagnostic detail; never panics on a well-formed certificate.
pub fn verify_safety_certificate(cert: &SafetyCertificate) -> VerifyReport {
    if cert.schema != SCHEMA_V1 {
        return VerifyReport {
            digest_ok: false,
            oracle: InductiveOracleVerdict::Inconclusive {
                reason: format!("unrecognized schema `{}`", cert.schema),
            },
            ay_redischarge: None,
            kernel_recheck: None,
            accepted: false,
            verdict: CertVerdict::Rejected,
            detail: format!("REJECTED: unrecognized schema `{}`", cert.schema),
        };
    }

    let digest_ok = cert.compute_digest() == cert.digest;

    // EXPLICIT-STATE lane (Leg E): a fixpoint cert is re-checked by RE-DERIVING the reachable set `R`
    // from the spec independently (binding the embedded `R` to THIS spec) and RE-RUNNING the Clean
    // kernel on its three legs. The inductive-safety legs below (oracle / ay / Leg D) do not apply.
    if let Some(fp) = &cert.explicit_fixpoint {
        return verify_explicit_fixpoint_report(cert, fp, digest_ok);
    }

    // Leg A: explicit-state eval oracle (engine-diverse, no solver) — the spine.
    let config = cert.reconstructed_config();
    let oracle = eval_oracle_inductive_safe(
        &cert.spec_src,
        &config,
        &cert.invariant_j_tla,
        DEFAULT_ORACLE_STATE_BOUND,
    );
    let oracle_ok = matches!(oracle, InductiveOracleVerdict::NoViolation { .. });

    // Leg C: AY re-discharge of the three obligations WITH AY's own proofs, in
    // process (re-asserting the re-derived obligation + re-solving + strict-
    // checking — so the proof is of THIS obligation by construction, closing the
    // trap where a serialized proof of a different formula strict-verifies).
    // `ay_redischarge` = all three UNSAT; `ay_strict` = how many of those carry an
    // AY strict-verified proof (today AY demotes LIA Farkas lemmas to trust, so
    // this is reported, not yet required — see the AY Farkas fix).
    #[cfg(feature = "ay")]
    let (ay_redischarge, ay_strict, ay_total, deadlock_strict): (
        Option<bool>,
        usize,
        usize,
        bool,
    ) = {
        match crate::ay_bmc::certificate_obligation_proofs(
            &cert.spec_src,
            &config,
            &cert.invariant_j_tla,
        ) {
            Some(obs) => {
                let total = obs.len();
                let all_unsat = obs.iter().all(|o| o.unsat);
                let strict = obs.iter().filter(|o| o.strict_verified).count();
                // DEADLOCK-FREEDOM is outside both per-obligation legs (no `SmtObligation`
                // member for Leg D, no kernel term for Leg K), so its ONLY proof-checked
                // evidence is the strict check of this fresh in-process re-solve. REQUIRE it
                // (the structural marker for an unguarded total Next is strict_verified=true,
                // so only guarded specs are gated — exactly the specs whose deadlock query is
                // a real solver obligation).
                let deadlock_strict = obs
                    .iter()
                    .all(|o| o.name != "deadlock_freedom" || o.strict_verified);
                (Some(all_unsat), strict, total, deadlock_strict)
            }
            None => (None, 0, 0, false),
        }
    };
    #[cfg(not(feature = "ay"))]
    let (ay_redischarge, ay_strict, ay_total, deadlock_strict): (
        Option<bool>,
        usize,
        usize,
        bool,
    ) = (None, 0, 0, false);
    let _ = deadlock_strict;

    // Leg D — the EXTERNAL proof re-check: re-check each SMT obligation's EMBEDDED proof
    // bundle with AY's audited checker (NO solver search), under the assume-coverage gates
    // that bind the proof to the obligation TY's translator recognizes. Per-obligation
    // detail (Phase D/E): each obligation individually Some(true)/Some(false)/None.
    #[cfg(feature = "ay")]
    let leg_d_detail = verify_leg_d_detail(cert);
    #[cfg(feature = "ay")]
    let leg_d: Option<bool> = match &leg_d_detail {
        None => None,
        Some(d) if d.iter().any(|(_, r)| *r == Some(false)) => Some(false),
        Some(d) if d.iter().all(|(_, r)| *r == Some(true)) => Some(true),
        Some(_) => None,
    };
    #[cfg(not(feature = "ay"))]
    let leg_d = verify_leg_d(cert);

    // Leg K (kernel): re-run the Clean CIC kernel on every embedded CleanCic term — the STRONGEST
    // tier (trust base = the small kernel, not the SMT solver). It covers only reflexive
    // obligations, so it STRENGTHENS rather than replaces the Leg-D acceptance basis; but a present
    // term that fails the kernel forces Rejected (tamper/invalid detection).
    let leg_k = verify_leg_k(cert);

    // Leg C (in-process re-solve) is KEPT as DEFENSE-IN-DEPTH: every obligation must still
    // re-solve UNSAT in-process, and DEADLOCK-FREEDOM must strict-verify here (it is the one
    // obligation the per-obligation strong coverage below cannot reach — no Leg-D bundle
    // membership, no kernel term). The old blanket `ay_strict == ay_total` for the THREE SMT
    // obligations is replaced by per-obligation strong coverage — the strictness evidence
    // lives in Leg D's bundle re-check or a kernel-checked term, per obligation.
    let in_process_ok = ay_total > 0 && ay_redischarge == Some(true) && deadlock_strict;

    // PER-OBLIGATION STRONG COVERAGE (Phase D/E): each of the three SMT obligations must be
    // discharged by at least one of the two independent proof-checking legs —
    //   • Leg D: its embedded bundle strict-re-checked by AY's audited checker AND bound to
    //     the obligation (assume-coverage + render + probe gates), or
    //   • Leg K: an embedded CIC term the CLEAN KERNEL re-accepts at the obligation type
    //     REBUILT from the re-derived spec (`verify_consecution_term` / reflexive / ground /
    //     conjunctive engines) — the STRONGER tier.
    // This is what lets a kernel-checked coupled consecution (`b' = b + a`, outside ay's
    // strict Farkas fragment) carry its obligation: kernel-checked > ay-checker-checked.
    #[cfg(all(feature = "ay", feature = "clean-cic"))]
    let (strong_all, obligation_basis): (bool, Vec<String>) = {
        let mut basis = Vec::new();
        let mut all = true;
        for name in ["initiation", "consecution", "safety"] {
            let d_ok = leg_d_detail
                .as_ref()
                .is_some_and(|d| d.iter().any(|(n, r)| *n == name && *r == Some(true)));
            let k_ok = cert.ay_proof_obligations.iter().any(|o| {
                o.name == name && !o.clean_cic_term.is_empty() && {
                    match name {
                        "consecution" => verify_consecution_term(cert, &o.clean_cic_term),
                        "initiation" => {
                            verify_reflexive_leg(cert, "initiation", &o.clean_cic_term)
                                || verify_initiation_ground(cert, &o.clean_cic_term)
                        }
                        "safety" => {
                            verify_reflexive_leg(cert, "safety", &o.clean_cic_term)
                                || verify_safety_conj(cert, &o.clean_cic_term)
                        }
                        _ => false,
                    }
                }
            });
            basis.push(format!(
                "{name}={}",
                match (d_ok, k_ok) {
                    (true, true) => "ay-checker+KERNEL",
                    (true, false) => "ay-checker",
                    (false, true) => "KERNEL",
                    (false, false) => "UNCOVERED",
                }
            ));
            all &= d_ok || k_ok;
        }
        (all, basis)
    };
    #[cfg(all(feature = "ay", not(feature = "clean-cic")))]
    let (strong_all, obligation_basis): (bool, Vec<String>) = (leg_d == Some(true), Vec::new());
    #[cfg(not(feature = "ay"))]
    let (strong_all, obligation_basis): (bool, Vec<String>) = (false, Vec::new());
    let _ = &obligation_basis;

    // SOUND acceptance. The safety claim for a (possibly infinite-state) spec rests on the
    // per-obligation STRONG coverage above (external audited-checker proof or kernel-checked
    // term, each bound to the re-derived obligation), with the in-process re-solve as
    // defense-in-depth. The engine-diverse eval oracle is a REQUIRED CROSS-CHECK (must not
    // refute), NEVER the basis: NoViolation within a 4096-state bound is not a proof for an
    // unbounded state space. Fail-closed — an uncovered obligation, any non-UNSAT re-solve,
    // or any definitive Leg-D refutation never accepts.
    let ay_proof_backed = strong_all && in_process_ok && leg_d != Some(false);
    // A tampered/invalid embedded kernel term (Leg K Some(false)) is TAMPER-EVIDENCE: the certificate
    // object cannot be trusted, so it never Accepts even if the SMT legs independently re-derive
    // safety. (Leg K None — no kernel terms / feature off — does not gate.)
    let accepted = digest_ok && oracle_ok && ay_proof_backed && leg_k != Some(false);

    let verdict = if accepted {
        CertVerdict::Accepted
    } else if matches!(oracle, InductiveOracleVerdict::Refuted { .. })
        || !digest_ok
        || ay_redischarge == Some(false)
        || leg_d == Some(false)
        || leg_k == Some(false)
    {
        CertVerdict::Rejected
    } else {
        // leg_d == None (feature off / undecomposable / bundle absent) and nothing
        // refuted -> fail-closed Inconclusive.
        CertVerdict::Inconclusive
    };

    let ay_leg = match ay_redischarge {
        Some(true) => format!(
            "ay in-process re-discharge UNSAT {ay_total}/{ay_total} ({ay_strict} strict at \
             re-solve; deadlock-freedom strictness enforced here; the 3 SMT obligations' \
             strictness is judged per-obligation by leg D / leg K)"
        ),
        Some(false) => "ay-redischarge REFUTED".to_string(),
        None => "ay-redischarge unavailable (needs the `ay` feature)".to_string(),
    };
    let leg_d_note = match leg_d {
        // Surface the per-obligation basis on the ACCEPTED path too, not just on
        // failures/aggregate-None: the basis is the load-bearing acceptance
        // evidence, and an auditor of an accepted certificate must see which
        // leg(s) carried each obligation (e.g. a coupled consecution reading
        // `ay-checker+KERNEL` vs `KERNEL` records exactly which independent
        // checkers covered it — drift in that string is a visible engine-
        // coverage change instead of a silent one).
        Some(true) if !obligation_basis.is_empty() => format!(
            "leg-D external proof re-check PASSED (3 SMT obligations, no re-solve; \
             per-obligation acceptance basis: {})",
            obligation_basis.join(", ")
        ),
        Some(true) => {
            "leg-D external proof re-check PASSED (3 SMT obligations, no re-solve)".to_string()
        }
        Some(false) => "leg-D external proof re-check REFUTED".to_string(),
        None if !obligation_basis.is_empty() => format!(
            "per-obligation acceptance basis: {} (leg-D aggregate not applicable)",
            obligation_basis.join(", ")
        ),
        None => "leg-D external proof re-check unavailable (no ay / not re-derivable / no bundle)"
            .to_string(),
    };
    // Leg K: how many obligations carry a Clean-kernel-CERTIFIED proof term (the strongest tier).
    let kernel_certified = cert
        .ay_proof_obligations
        .iter()
        .filter(|o| !o.clean_cic_term.is_empty())
        .count();
    let leg_k_note = match leg_k {
        Some(true) => format!(
            ", {kernel_certified} obligation(s) KERNEL-CERTIFIED (Clean CIC kernel re-checked — \
             trust base = the kernel checker (~33K LOC inside a ~722K-LOC crate, incl. its \
             structurally-admitted prelude + native reducers), not the SMT solver)"
        ),
        _ => String::new(),
    };

    let detail = if accepted {
        match &oracle {
            InductiveOracleVerdict::NoViolation {
                states_explored,
                complete,
            } => format!(
                "VERIFIED (external proof re-check): inductive-safety + \
                 deadlock-freedom for invariant `{}` — digest ok, {leg_d_note}, eval-oracle \
                 agrees ({} states{}), {ay_leg}{leg_k_note}",
                cert.invariant_j_tla,
                states_explored,
                if *complete {
                    ", complete"
                } else {
                    ", within bound"
                }
            ),
            _ => "VERIFIED".to_string(),
        }
    } else {
        let mut reasons = Vec::new();
        if !digest_ok {
            reasons.push("digest mismatch".to_string());
        }
        match &oracle {
            InductiveOracleVerdict::Refuted { detail, .. } => {
                reasons.push(format!("eval-oracle REFUTED ({detail})"))
            }
            InductiveOracleVerdict::Inconclusive { reason } => {
                reasons.push(format!("eval-oracle inconclusive ({reason})"))
            }
            InductiveOracleVerdict::NoViolation { .. } => {}
        }
        match ay_redischarge {
            Some(false) => reasons.push("ay-redischarge REFUTED (J not inductive or J=>Safety fails)".to_string()),
            None => reasons.push(
                if cfg!(feature = "ay") {
                    "AY inductive proof not re-derivable (Next is undecomposable for                      deadlock-freedom, or J/spec is outside the strict-verifiable fragment)"
                        .to_string()
                } else {
                    "AY inductive proof not re-checkable: this build lacks the `ay` feature"
                        .to_string()
                },
            ),
            Some(true) if !in_process_ok => reasons.push(
                "ay obligations UNSAT but deadlock-freedom did not strict-verify at the \
                 in-process re-solve (its only proof-checked gate)"
                    .to_string(),
            ),
            Some(true) => {}
        }
        // A tampered/invalid embedded kernel term is an independent Rejected trigger — say so.
        if leg_k == Some(false) {
            reasons.push(
                "leg-K kernel re-check REFUTED an embedded CleanCic term (tampered or not a \
                 proof of this spec's obligation)"
                    .to_string(),
            );
        }
        match leg_d {
            Some(false) => reasons.push(
                "leg-D external proof re-check REFUTED (embedded proof invalid, or does not \
                 match the obligation TY re-translates)"
                    .to_string(),
            ),
            // None under the ay feature: a covered obligation lacked a bundle or the
            // obligation was not re-derivable -> Inconclusive (never accept). The
            // non-ay case is already explained by the ay-redischarge None arm above.
            None if cfg!(feature = "ay") && ay_redischarge.is_some() => reasons.push(
                "leg-D external proof re-check unavailable (certificate carries no proof \
                 bundle for a covered obligation, or it is not re-derivable)"
                    .to_string(),
            ),
            None | Some(true) => {}
        }
        // Per-obligation acceptance basis (Phase D/E) — surfaced on FAILURES too, so an
        // auditor sees exactly which obligation is UNCOVERED rather than a generic note.
        if !obligation_basis.is_empty() && !strong_all {
            reasons.push(format!(
                "per-obligation acceptance basis: {}",
                obligation_basis.join(", ")
            ));
        }
        let label = if verdict == CertVerdict::Inconclusive {
            "INCONCLUSIVE"
        } else {
            "REJECTED"
        };
        format!("{label}: {}", reasons.join("; "))
    };

    VerifyReport {
        digest_ok,
        oracle,
        ay_redischarge,
        kernel_recheck: leg_k,
        accepted,
        verdict,
        detail,
    }
}

/// Prove inductive safety for a spec and, on success, build a certificate.
///
/// Runs the AY-backed inductive-safety prover ([`crate::ay_bmc::prove_inductive_safety_for_cert`])
/// and serializes the proven invariant `J` into a [`SafetyCertificate`]. Returns
/// `None` when the spec is not in the provable class (the caller falls back to
/// the normal verdict). The certificate it produces is re-checkable by
/// [`verify_safety_certificate`] with no access to this prover.
#[cfg(feature = "ay")]
pub fn certify_spec(spec_src: &str, config: &Config) -> Option<SafetyCertificate> {
    let tree = tla_core::parse_to_syntax_tree(spec_src);
    let lowered = tla_core::lower(tla_core::FileId(0), &tree);
    let module = lowered.module?;

    let mut ctx = crate::eval::EvalCtx::new();
    ctx.load_module(&module);

    let vars = crate::ay_shared::collect_state_vars(&module, &ctx);
    // Certify-mode prover: proves inductive safety for ANY 1-inductive (or
    // interval-strengthenable) spec, not only unguarded divergent counters (P2).
    let proof = crate::ay_bmc::prove_inductive_safety_for_cert(&ctx, config, &vars)?;

    let invariant_j_tla = tla_core::pretty_expr(&proof.invariant_j.node);
    let var_sorts = proof
        .var_sorts
        .iter()
        .map(|(name, sort)| (name.clone(), format!("{sort:?}")))
        .collect();

    let mut cert = build_safety_certificate(spec_src, config, invariant_j_tla.clone(), var_sorts);

    // Embed AY's own re-checkable proof for each obligation (the AY proof leg).
    if let Some(obligations) =
        crate::ay_bmc::certificate_obligation_proofs(spec_src, config, &invariant_j_tla)
    {
        cert.ay_proof_obligations = obligations
            .into_iter()
            .map(|o| {
                // Reflexivity HINT (`φ ⇒ φ`): initiation is `Init ⇒ J`, safety is `J ⇒ Safety`. We
                // compare the CONSTANT-concretized predicates' span-insensitive canonical rendering
                // (`pretty_expr`, consistent with the ay_bmc Module path) rather than `node ==`. It is
                // only a HINT — the Clean kernel is the arbiter: `clean_cic_{initiation,safety}_term`
                // certify via `certify_reflexive_faithful` (`Π(vars). embed(ante)→embed(cons)`), which
                // the kernel accepts ONLY when `embed(ante) ≡ embed(cons)`. So a hint false-positive
                // yields no cert (Discharged), never a false `Certified`.
                let denote_eq = |a: &tla_core::ast::Expr, b: &tla_core::ast::Expr| {
                    tla_core::pretty_expr(a) == tla_core::pretty_expr(b)
                };
                let reflexive = match o.name {
                    "initiation" => denote_eq(&proof.init.node, &proof.invariant_j.node),
                    "safety" => denote_eq(&proof.invariant_j.node, &proof.safety.node),
                    _ => false,
                };
                AyObligationProof {
                    name: o.name.to_string(),
                    strict_verified: o.strict_verified,
                    clean_supported: o.clean_supported,
                    lrat_present: o.lrat_present,
                    alethe: o.alethe,
                    // Embed the portable proof bundle for offline (Leg D) re-check.
                    bundle_json: o.bundle_json.unwrap_or_default(),
                    // Leg K: kernel-CERTIFIED CIC term. Reflexive Init≡J / J≡Safety → the FAITHFUL
                    // reflexive cert (kernel-bound to embed(ante)→embed(cons), rejects non-reflexive);
                    // consecution → EUF transitivity / faithful Int congruence / LIA; non-reflexive
                    // initiation → faithful ground (x=c ⇒ x≥0).
                    clean_cic_term: match o.name {
                        "consecution" => clean_cic_consecution_term(&proof, o.strict_verified),
                        "initiation" => {
                            clean_cic_initiation_term(&proof, o.strict_verified, reflexive)
                        }
                        "safety" => clean_cic_safety_term(&proof, reflexive),
                        _ => Vec::new(),
                    },
                }
            })
            .collect();
        cert.digest = cert.compute_digest();
    }

    Some(cert)
}

/// The CONSTANT names declared by a spec.
///
/// FIXED-INSTANCE SCOPE (symbolic-N is research): a `ty.cert` certifies the
/// CONCRETE instance defined by the config — every `CONSTANT` is concretized to
/// its configured value before the SMT translation (`fold_config_constant_ident`).
/// It is NOT a parametric (all-`N`) proof. Surfacing the declared constants lets
/// `ty certify` label a fixed-instance certificate honestly so it is never
/// misread as an all-`N` result. (A parametric certificate is in any case
/// fail-closed to INCONCLUSIVE at re-check, since the certificate carries no
/// constant values to re-derive the obligation with — so this is honest labeling,
/// not a soundness fix.) Proving a property for ALL `N` (keep-`N`-symbolic
/// plumbing + a native-`Forall` / `ay-chc` quantifier path + a v2 schema) is a
/// documented research track.
#[must_use]
pub fn declared_constants(spec_src: &str) -> Vec<String> {
    let tree = tla_core::parse_to_syntax_tree(spec_src);
    let Some(module) = tla_core::lower(tla_core::FileId(0), &tree).module else {
        return Vec::new();
    };
    module
        .units
        .iter()
        .filter_map(|u| match &u.node {
            tla_core::ast::Unit::Constant(decls) => Some(
                decls
                    .iter()
                    .map(|d| d.name.node.clone())
                    .collect::<Vec<_>>(),
            ),
            _ => None,
        })
        .flatten()
        .collect()
}

/// Lowercase hex of a byte slice.
fn hex_lower(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        let _ = write!(s, "{b:02x}");
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Explicit-state fixpoint lane SERIALIZED end-to-end: certify a real spec → build a
    /// SafetyCertificate → JSON round-trip → `verify_safety_certificate` (i.e. `ty cert-check`)
    /// re-derives R from the spec, re-runs the kernel on all 3 legs, and ACCEPTS; tampering the
    /// embedded R breaks the digest → Rejected.
    #[cfg(all(feature = "ay", feature = "clean-cic"))]
    #[test]
    fn explicit_fixpoint_cert_serializes_and_recheck_accepts() {
        let spec = "---- MODULE EF ----\n\
                    EXTENDS Integers\n\
                    VARIABLE x\n\
                    Init == x = 2 \\/ x = 5\n\
                    Next == x' = x\n\
                    Safety == x >= 0\n\
                    ====\n";
        let config = Config {
            init: Some("Init".into()),
            next: Some("Next".into()),
            invariants: vec!["Safety".into()],
            ..Default::default()
        };
        let fp = crate::explicit_fixpoint_cert::certify_explicit_state_spec(spec, &config)
            .expect("explicit-state spec certifies");
        let cert = build_explicit_fixpoint_certificate(spec, &config, fp);
        // JSON round-trip (the serialized, self-contained artifact `ty cert-check` consumes).
        let json = serde_json::to_string(&cert).expect("serialize");
        let back: SafetyCertificate = serde_json::from_str(&json).expect("deserialize");
        assert!(
            back.explicit_fixpoint.is_some(),
            "the fixpoint cert survives the round-trip"
        );
        assert_eq!(
            verify_safety_certificate(&back).verdict,
            CertVerdict::Accepted,
            "ty cert-check re-derives R + kernel-re-checks all 3 legs and ACCEPTS"
        );
        // Tamper the embedded R (drop a reachable state) without re-sealing → digest mismatch → Rejected.
        let mut tampered = back.clone();
        tampered.explicit_fixpoint.as_mut().unwrap().reachable.pop();
        assert_eq!(
            verify_safety_certificate(&tampered).verdict,
            CertVerdict::Rejected,
            "a tampered R must be rejected (digest)"
        );
    }

    /// The AFFINE counter exercises the KERNEL-RE-EVALUATED `Next`-completeness leg through the FULL
    /// `ty cert-check` path (`verify_safety_certificate` → Leg E): the kernel re-evaluates `Next` over
    /// the finite domain, and the recognized shape is bound to the spec (a cert claiming a different
    /// counter is rejected). The companion to the membership/digest round-trip above.
    #[cfg(all(feature = "ay", feature = "clean-cic"))]
    #[test]
    fn explicit_fixpoint_affine_completeness_rechecks_end_to_end() {
        let spec = "---- MODULE AC ----\n\
                    EXTENDS Integers\n\
                    VARIABLE x\n\
                    Init == x = 0\n\
                    Next == x' = x + 1 /\\ x < 3\n\
                    Safety == x >= 0\n\
                    ====\n";
        let config = Config {
            init: Some("Init".into()),
            next: Some("Next".into()),
            invariants: vec!["Safety".into()],
            ..Default::default()
        };
        let fp = crate::explicit_fixpoint_cert::certify_explicit_state_spec(spec, &config)
            .expect("affine counter certifies with both kernel-re-evaluated completeness legs");
        assert!(
            fp.next_shape.is_some() && fp.init_shape.is_some(),
            "both the Next- and Init-completeness legs are present for the affine fragment"
        );
        let cert = build_explicit_fixpoint_certificate(spec, &config, fp);
        let back: SafetyCertificate =
            serde_json::from_str(&serde_json::to_string(&cert).unwrap()).unwrap();
        assert!(
            back.explicit_fixpoint
                .as_ref()
                .unwrap()
                .next_shape
                .is_some(),
            "next_shape survives serde"
        );
        assert_eq!(
            verify_safety_certificate(&back).verdict,
            CertVerdict::Accepted,
            "ty cert-check re-evaluates Init+Next over the finite domain and ACCEPTS"
        );
        // Tamper the recognized `Next` shape and RE-SEAL the digest: Leg E re-derives the shape from the
        // spec (bound=3) and the kernel re-runs the completeness leg with the bogus bound=99 — BOTH the
        // shape-binding and the kernel re-check reject a cert claiming a different counter than the spec.
        let mut tampered = back.clone();
        if let Some(sh) = tampered
            .explicit_fixpoint
            .as_mut()
            .unwrap()
            .next_shape
            .as_mut()
        {
            sh.bound = 99;
        }
        tampered.digest = tampered.compute_digest();
        assert_eq!(
            verify_safety_certificate(&tampered).verdict,
            CertVerdict::Rejected,
            "a cert claiming a different Next counter than the spec must be rejected (Leg E + kernel)"
        );
    }

    /// GENERAL (non-affine) Next exercises the KERNEL-RE-EVALUATED general-completeness leg through the
    /// FULL `ty cert-check` path (`verify_safety_certificate` → Leg E): the kernel re-evaluates the
    /// ACTUAL `Next` predicate IR over the finite domain, and the IR + its derived bound `H` are bound
    /// to the spec (a cert claiming a different predicate/bound is rejected by Leg-E's spec re-derivation).
    #[cfg(all(feature = "ay", feature = "clean-cic"))]
    #[test]
    fn explicit_fixpoint_general_completeness_rechecks_end_to_end() {
        let spec = "---- MODULE GC ----\n\
                    EXTENDS Integers\n\
                    VARIABLE x\n\
                    Init == x = 0\n\
                    Next == x' = (x + 2) % 5 /\\ x' < 9\n\
                    Safety == x >= 0\n\
                    ====\n";
        let config = Config {
            init: Some("Init".into()),
            next: Some("Next".into()),
            invariants: vec!["Safety".into()],
            ..Default::default()
        };
        let fp = crate::explicit_fixpoint_cert::certify_explicit_state_spec(spec, &config)
            .expect("non-affine modular Next certifies via the general IR leg");
        assert!(
            fp.next_shape.is_none() && fp.next_pred.is_some(),
            "the general (not affine) Next leg is present"
        );
        let cert = build_explicit_fixpoint_certificate(spec, &config, fp);
        let back: SafetyCertificate =
            serde_json::from_str(&serde_json::to_string(&cert).unwrap()).unwrap();
        assert!(
            back.explicit_fixpoint.as_ref().unwrap().next_pred.is_some(),
            "next_pred (general IR) survives serde"
        );
        assert_eq!(
            verify_safety_certificate(&back).verdict,
            CertVerdict::Accepted,
            "ty cert-check re-evaluates the real Next predicate over the finite domain and ACCEPTS"
        );
        // Tamper the serialized domain bound `H` (8→3) and RE-SEAL the digest: Leg E re-derives the IR +
        // bound from the spec (H=8) and binds `re.next_pred == fp.next_pred`; the bogus H=3 mismatches →
        // Rejected (the spec re-derivation, not a trusted serialized bound, governs the domain).
        let mut tampered = back.clone();
        if let Some(np) = tampered
            .explicit_fixpoint
            .as_mut()
            .unwrap()
            .next_pred
            .as_mut()
        {
            np.hi = vec![3];
        }
        tampered.digest = tampered.compute_digest();
        assert_eq!(
            verify_safety_certificate(&tampered).verdict,
            CertVerdict::Rejected,
            "a cert claiming a different domain bound than the spec must be rejected (Leg E)"
        );
    }

    /// GENERAL `R⊆Safety` leg end-to-end through the FULL `ty cert-check` path
    /// (`verify_safety_certificate` → Leg E): an interval-membership invariant (`x ∈ 1..12`, the
    /// HourClock class — NOT the `⋀ x≥0` shape) certifies via the general embedded-safety lane,
    /// serializes, and ACCEPTS. Tampering the stored `safety_pred` to a WIDER invariant (12→13) is
    /// the case the kernel ALONE cannot catch (`⋀_{s∈R} 1≤s∧s≤13` still reduces true over R) —
    /// Leg-E's spec re-recognition equality (`re.safety_pred == fp.safety_pred`) rejects it even
    /// after re-sealing the digest.
    #[cfg(all(feature = "ay", feature = "clean-cic"))]
    #[test]
    fn explicit_fixpoint_general_safety_rechecks_end_to_end() {
        use crate::explicit_fixpoint_cert::{PredIR, ValIR};
        let spec = "---- MODULE HCShape ----\n\
                    EXTENDS Integers\n\
                    VARIABLE x\n\
                    Init == x \\in 2..5\n\
                    Next == x' = x\n\
                    Safety == x \\in 1..12\n\
                    ====\n";
        let config = Config {
            init: Some("Init".into()),
            next: Some("Next".into()),
            invariants: vec!["Safety".into()],
            ..Default::default()
        };
        let fp = crate::explicit_fixpoint_cert::certify_explicit_state_spec(spec, &config)
            .expect("the interval-membership invariant certifies via the general safety leg");
        assert!(
            fp.safety_pred.is_some() && fp.safety_general.is_some(),
            "the general R⊆Safety leg is present"
        );
        let cert = build_explicit_fixpoint_certificate(spec, &config, fp);
        let back: SafetyCertificate =
            serde_json::from_str(&serde_json::to_string(&cert).unwrap()).unwrap();
        assert!(
            back.explicit_fixpoint
                .as_ref()
                .unwrap()
                .safety_pred
                .is_some(),
            "safety_pred survives serde"
        );
        assert_eq!(
            verify_safety_certificate(&back).verdict,
            CertVerdict::Accepted,
            "ty cert-check re-recognizes the invariant, rebuilds the obligation over the \
             re-derived R, kernel-re-checks — ACCEPT"
        );
        // WIDEN the stored invariant (12→13) and RE-SEAL the digest: the kernel re-check still
        // passes (the conjunction is true over R at the wider bound), so ONLY Leg-E's equality
        // with the spec-re-recognized IR can reject — and it must.
        let mut tampered = back.clone();
        tampered.explicit_fixpoint.as_mut().unwrap().safety_pred = Some(PredIR::And(
            Box::new(PredIR::Leq(ValIR::Lit(1), ValIR::Var(0))),
            Box::new(PredIR::Leq(ValIR::Var(0), ValIR::Lit(13))),
        ));
        tampered.digest = tampered.compute_digest();
        assert_eq!(
            verify_safety_certificate(&tampered).verdict,
            CertVerdict::Rejected,
            "a cert claiming a WIDER invariant than the spec's must be rejected by Leg-E \
             re-recognition (the kernel alone cannot see the widening)"
        );
    }

    /// MULTI-VARIABLE UNBOUNDED end-to-end through the FULL `ty cert-check` path: the parametric
    /// conjoined inductive-invariant cert for `VARIABLES x,y / Init x=0∧y=3 / Next x'=x+1∧y'=y+2 /
    /// Safety x≥0∧y≥0` serializes, and `verify_safety_certificate` (Leg E) re-recognizes the tuple
    /// `[(0,1),(3,2)]` from the spec, requires `re.unbounded_invariant == fp.unbounded_invariant`, and
    /// the kernel re-checks the 3 conjoined legs — ACCEPT. Tampering the claimed `pairs` (and re-sealing
    /// the digest) is rejected: Leg E re-derives `[(0,1),(3,2)]` from the spec and the mismatch fails.
    #[cfg(all(feature = "ay", feature = "clean-cic"))]
    #[test]
    fn multivar_unbounded_invariant_rechecks_end_to_end() {
        let spec = "---- MODULE MVUB ----\n\
                    EXTENDS Integers\n\
                    VARIABLES x, y\n\
                    Init == x = 0 /\\ y = 3\n\
                    Next == x' = x + 1 /\\ y' = y + 2\n\
                    Safety == x >= 0 /\\ y >= 0\n\
                    ====\n";
        let config = Config {
            init: Some("Init".into()),
            next: Some("Next".into()),
            invariants: vec!["Safety".into()],
            ..Default::default()
        };
        let fp = crate::explicit_fixpoint_cert::certify_explicit_state_spec(spec, &config)
            .expect("multi-var unbounded counter certifies via the conjoined parametric tuple leg");
        let ub = fp
            .unbounded_invariant
            .as_ref()
            .expect("carries the parametric invariant");
        assert_eq!(ub.pairs, vec![(0, 1), (3, 2)]);
        assert!(
            fp.reachable.is_empty(),
            "NO enumeration for an unbounded cert"
        );
        let cert = build_explicit_fixpoint_certificate(spec, &config, fp);
        let back: SafetyCertificate =
            serde_json::from_str(&serde_json::to_string(&cert).unwrap()).unwrap();
        assert_eq!(
            back.explicit_fixpoint
                .as_ref()
                .unwrap()
                .unbounded_invariant
                .as_ref()
                .unwrap()
                .pairs,
            vec![(0, 1), (3, 2)],
            "the conjoined `pairs` vector survives serde"
        );
        assert_eq!(
            verify_safety_certificate(&back).verdict,
            CertVerdict::Accepted,
            "ty cert-check re-recognizes the tuple from the spec and kernel-re-checks the conjoined legs"
        );
        // Tamper the claimed per-variable increment (δ_y 2→9) and RE-SEAL the digest: Leg E re-derives
        // [(0,1),(3,2)] from the spec and requires `re.unbounded_invariant == fp.unbounded_invariant`,
        // so the bogus δ_y=9 mismatches the spec → Rejected (the spec, not the serialized vector, governs).
        let mut tampered = back.clone();
        if let Some(ub) = tampered
            .explicit_fixpoint
            .as_mut()
            .unwrap()
            .unbounded_invariant
            .as_mut()
        {
            ub.pairs[1].1 = 9;
        }
        tampered.digest = tampered.compute_digest();
        assert_eq!(
            verify_safety_certificate(&tampered).verdict,
            CertVerdict::Rejected,
            "a cert claiming a different δ than the spec must be rejected (Leg E spec re-derivation)"
        );
    }

    /// Leg K end-to-end: a certificate carrying a kernel-Certified term re-checks via the Clean
    /// kernel inside `verify_safety_certificate`; tampering the embedded term is detected (Rejected).
    #[cfg(all(feature = "ay", feature = "clean-cic"))]
    #[test]
    fn leg_k_re_runs_the_kernel_and_detects_tampering() {
        // A REAL reflexive spec (J ≡ Safety = x≥0): the safety obligation carries the FAITHFUL
        // reflexive kernel cert (Π(x). embed(J) → embed(Safety)); Leg K re-derives (J,Safety) from
        // the spec and re-runs the Clean kernel — no SMT, no model checker.
        let spec = "---- MODULE NN ----\n\
                    VARIABLE x\n\
                    Init == x = 3\n\
                    Next == x' = x\n\
                    Safety == x >= 0\n\
                    ====\n";
        let config = Config {
            init: Some("Init".to_string()),
            next: Some("Next".to_string()),
            invariants: vec!["Safety".to_string()],
            ..Default::default()
        };
        let mut cert = certify_spec(spec, &config).expect("x≥0 spec certifies");
        assert_eq!(
            verify_leg_k(&cert),
            Some(true),
            "the embedded kernel terms must independently re-check via the Clean kernel"
        );

        // Tamper a kernel term: Leg K must REFUTE (forces the overall verdict away from Accepted).
        let o = cert
            .ay_proof_obligations
            .iter_mut()
            .find(|o| !o.clean_cic_term.is_empty())
            .expect("a kernel-checked obligation is present");
        o.clean_cic_term = b"{\"BVar\":0}".to_vec();
        cert.digest = cert.compute_digest(); // re-seal so the rejection is via Leg K, not the digest
        assert_eq!(
            verify_leg_k(&cert),
            Some(false),
            "a tampered kernel term must be rejected by the re-check"
        );
        assert_ne!(
            verify_safety_certificate(&cert).verdict,
            CertVerdict::Accepted,
            "a certificate with a tampered kernel term is never Accepted"
        );
    }

    /// End-to-end: a real spec reaches FULL kernel-Certified (all three obligations) through
    /// certify_spec → verify_safety_certificate, with the consecution re-checked obligation-aware.
    #[cfg(all(feature = "ay", feature = "clean-cic"))]
    #[test]
    fn x_stays_zero_consecution_reaches_kernel_certified_end_to_end() {
        let spec = "---- MODULE XStaysZero ----\n\
                    VARIABLE x\n\
                    Init == x = 0\n\
                    Next == x' = x\n\
                    Safety == x = 0\n\
                    ====\n";
        let config = Config {
            init: Some("Init".to_string()),
            next: Some("Next".to_string()),
            invariants: vec!["Safety".to_string()],
            ..Default::default()
        };
        let cert = certify_spec(spec, &config).expect("x-stays-0 must certify");
        // The consecution (`J∧Next⇒J'`) carries a kernel-CERTIFIED EUF proof term.
        let consec = cert
            .ay_proof_obligations
            .iter()
            .find(|o| o.name == "consecution")
            .expect("consecution obligation present");
        assert!(
            !consec.clean_cic_term.is_empty(),
            "the consecution reaches the kernel-checked Certified tier via the EUF engine"
        );
        // verify_safety_certificate re-runs the kernel — including the OBLIGATION-AWARE consecution
        // re-check (re-derives J/Next, rejects a proof of a different proposition) — and accepts.
        let report = verify_safety_certificate(&cert);
        assert_eq!(
            report.kernel_recheck,
            Some(true),
            "Leg K (kernel re-check, incl. the consecution) must pass: {}",
            report.detail
        );
        assert!(
            report.accepted,
            "the certificate is accepted: {}",
            report.detail
        );
    }

    /// A faithful ARITHMETIC spec reaches FULL kernel-Certified across THREE distinct engines:
    /// init `x=3⇒x≥0` (ground), consecution `x≥0∧x'=x⇒x'≥0` (faithful Int congruence), safety
    /// `x≥0⇒x≥0` (reflexive identity) — all via the real Int.le embedding, re-checked by the kernel.
    #[cfg(all(feature = "ay", feature = "clean-cic"))]
    #[test]
    fn nonneg_int_spec_fully_kernel_certified_end_to_end() {
        let spec = "---- MODULE NN ----\n\
                    VARIABLE x\n\
                    Init == x = 3\n\
                    Next == x' = x\n\
                    Safety == x >= 0\n\
                    ====\n";
        let config = Config {
            init: Some("Init".to_string()),
            next: Some("Next".to_string()),
            invariants: vec!["Safety".to_string()],
            ..Default::default()
        };
        let cert = certify_spec(spec, &config).expect("the x≥0 spec must certify");
        for name in ["initiation", "consecution", "safety"] {
            let o = cert
                .ay_proof_obligations
                .iter()
                .find(|o| o.name == name)
                .unwrap_or_else(|| panic!("{name} obligation present"));
            assert!(
                !o.clean_cic_term.is_empty(),
                "{name} must be kernel-certified via the faithful Int embedding"
            );
        }
        // ty cert-check re-runs the kernel on all three engines (ground + congruence + reflexive).
        let report = verify_safety_certificate(&cert);
        assert_eq!(
            report.kernel_recheck,
            Some(true),
            "Leg K kernel re-check must pass for all three: {}",
            report.detail
        );
        assert!(report.accepted, "certificate accepted: {}", report.detail);
    }

    /// The LIA arithmetic consecution delivered END-TO-END on a real monotone-counter spec
    /// (ACCUMULATOR: Init x=0, Next x'=x+1, Safety x≥0). The consecution `x≥0 ∧ x'=x+1 ⇒ x'≥0`
    /// carries a kernel-CERTIFIED LIA term (NonNeg.rec+Eq.subst) and Leg K re-runs the kernel on it.
    /// (Leg A's explicit reachable-set enumeration is a separate model-checker concern from the
    /// kernel certificate — which is exactly the point: the kernel cert is out of that loop.)
    #[cfg(all(feature = "ay", feature = "clean-cic"))]
    #[test]
    fn monotone_counter_lia_consecution_kernel_certified_end_to_end() {
        let cert = certify_spec(ACCUMULATOR, &acc_config()).expect("the accumulator must certify");
        let consec = cert
            .ay_proof_obligations
            .iter()
            .find(|o| o.name == "consecution")
            .expect("consecution obligation present");
        assert!(
            !consec.clean_cic_term.is_empty(),
            "the LIA consecution x≥0 ∧ x'=x+1 ⇒ x'≥0 must be kernel-certified (NonNeg.rec)"
        );
        let report = verify_safety_certificate(&cert);
        assert_eq!(
            report.kernel_recheck,
            Some(true),
            "Leg K kernel re-check must pass for the LIA consecution: {}",
            report.detail
        );
    }

    const ACCUMULATOR: &str = "---- MODULE Accumulator ----\n\
                               EXTENDS Integers\n\
                               VARIABLE x\n\
                               Init == x = 0\n\
                               Next == x' = x + 1\n\
                               Safety == x >= 0\n\
                               ====\n";

    fn acc_config() -> Config {
        Config {
            init: Some("Init".to_string()),
            next: Some("Next".to_string()),
            invariants: vec!["Safety".to_string()],
            ..Default::default()
        }
    }

    /// Fixed-instance scope: a spec's declared CONSTANTs are surfaced so a cert is
    /// labeled honestly (concrete instance, not all-N). No constants -> empty.
    /// DIGEST BACK-COMPAT: a constant-free certificate must serialize with NO `constants`
    /// key at all (the `skip_serializing_if` contract) — otherwise every pre-existing cert
    /// digest breaks. Mirror of the `bound` field pattern; do not remove either attr.
    #[test]
    fn constant_free_cert_serialization_is_byte_stable() {
        let cert = build_safety_certificate(
            "---- MODULE M ----\nVARIABLE x\nInit == x = 0\nNext == x' = x\n====\n",
            &Config::default(),
            "TRUE".to_string(),
            Vec::new(),
        );
        assert!(cert.constants.is_empty());
        assert!(
            !cert.to_json().contains("\"constants\""),
            "constant-free certs must not serialize the constants key (digest back-compat)"
        );
        // And a cert WITH constants round-trips them through reconstructed_config.
        let mut cfg = Config::default();
        cfg.constants.insert(
            "N".to_string(),
            crate::config::ConstantValue::Value("3".to_string()),
        );
        cfg.constants_order.push("N".to_string());
        let cert2 = build_safety_certificate(
            "---- MODULE M ----\nVARIABLE x\nInit == x = 0\nNext == x' = x\n====\n",
            &cfg,
            "TRUE".to_string(),
            Vec::new(),
        );
        let rt = SafetyCertificate::from_json(&cert2.to_json()).unwrap();
        assert_eq!(
            rt.reconstructed_config().constants.get("N"),
            cfg.constants.get("N")
        );
    }

    #[test]
    fn test_declared_constants_for_honest_scope() {
        let parametric = "---- MODULE P ----\n\
                          CONSTANT N, M\n\
                          VARIABLE x\n\
                          Init == x = 0\n\
                          ====\n";
        let cs = declared_constants(parametric);
        assert!(
            cs.contains(&"N".to_string()) && cs.contains(&"M".to_string()),
            "got {cs:?}"
        );
        assert!(
            declared_constants(ACCUMULATOR).is_empty(),
            "Accumulator has no constants"
        );
    }

    #[cfg(feature = "ay")]
    #[test]
    fn test_certificate_roundtrip_and_verify() {
        // Produce a REAL certificate (via the prover) so it embeds the per-
        // obligation proof bundles Leg D re-checks. A hand-built cert (no bundles)
        // is now fail-closed to Inconclusive — see `test_bundleless_cert_inconclusive`.
        let cert = certify_spec(ACCUMULATOR, &acc_config())
            .expect("Accumulator (x'=x+1, Inv x>=0) must be certifiable");

        // The freshly-produced certificate verifies (digest ok + Leg A agrees +
        // Leg D external proof re-check passes for all three SMT obligations).
        let report = verify_safety_certificate(&cert);
        assert!(report.accepted, "fresh cert must verify: {}", report.detail);
        assert!(report.digest_ok);
        assert!(
            !cert.ay_proof_obligations.is_empty()
                && cert
                    .ay_proof_obligations
                    .iter()
                    .filter(|o| o.name != "deadlock_freedom")
                    .all(|o| !o.bundle_json.is_empty()),
            "every SMT obligation must carry an embedded proof bundle"
        );

        // JSON round-trip (including the embedded bundles) preserves verification.
        let json = cert.to_json();
        let reloaded = SafetyCertificate::from_json(&json).expect("reload");
        assert_eq!(reloaded, cert);
        assert!(verify_safety_certificate(&reloaded).accepted);
    }

    /// DELIVERABLE 3 — the SYNTHESIS lane vs. the parametric fast-path. A pipeline accumulator
    /// `VARIABLES a,b / Init a=0∧b=0 / Next a'=a+1 ∧ b'=b+a / Safety b≥0` is OUTSIDE the parametric
    /// unbounded fragment (the `b'=b+a` step has a VARIABLE increment, not a literal δ; and Safety is
    /// `b≥0` alone, not `⋀ x_j≥0`), so `certify_explicit_state_spec` DECLINES it — while `certify_spec`
    /// SYNTHESIZES the strengthening `a≥0` and proves `b≥0 ∧ a≥0` 1-inductive. This is exactly the lane
    /// split documented in `cmd_certify.rs`: the parametric path is a fast, enumeration-free special
    /// case; `certify_spec` is the general synthesis lane.
    #[cfg(all(feature = "ay", feature = "clean-cic"))]
    #[test]
    fn synthesis_lane_certifies_a_spec_the_parametric_path_declines() {
        let spec = "---- MODULE Pipeline ----\n\
                    EXTENDS Integers\n\
                    VARIABLES a, b\n\
                    Init == a = 0 /\\ b = 0\n\
                    Next == a' = a + 1 /\\ b' = b + a\n\
                    Safety == b >= 0\n\
                    ====\n";
        let config = Config {
            init: Some("Init".into()),
            next: Some("Next".into()),
            invariants: vec!["Safety".into()],
            ..Default::default()
        };
        // (1) The PARAMETRIC unbounded/explicit path DECLINES (variable increment `b+a`; Safety is a
        //     single `b≥0`, not the conjunctive-nonneg shape). Fail-closed, no false cert.
        assert!(
            crate::explicit_fixpoint_cert::certify_explicit_state_spec(spec, &config).is_none(),
            "the parametric fast-path must DECLINE this out-of-fragment spec"
        );
        // (2) The SYNTHESIS lane DERIVES the `a≥0` strengthening (`a≥0 ∧ b≥0` is 1-inductive) that the
        //     bare Safety `b≥0` is NOT. This is the load-bearing claim of DELIVERABLE 3: an invariant
        //     the spec did NOT state was SYNTHESIZED (Houdini-lite strengthening), not merely a given J
        //     checked — and it fires exactly where the enumeration-free parametric fast-path declines.
        let cert = certify_spec(spec, &config)
            .expect("certify_spec must SYNTHESIZE a≥0 and certify the pipeline accumulator");
        assert!(
            cert.invariant_j_tla.contains('a') && cert.invariant_j_tla.contains(">="),
            "the synthesized invariant J strengthens Safety with a nonneg conjunct on `a`: J = `{}`",
            cert.invariant_j_tla
        );
        // THE PHASE-D/E FLIP (`docs/kernel-checked-tla-plan.md`): the coupled consecution
        // (`b' = b + a`, a genuinely multi-variable dependence) was originally OUTSIDE ay's
        // strict Farkas re-check fragment — for years the honest outcome here was fail-closed
        // non-acceptance. The COUPLED-AFFINE KERNEL ENGINE synthesizes the consecution proof
        // (`Π(a b a' b':Int). NonNeg a → NonNeg b → Eq a' (a+1) → Eq b' (b+a) → And (NonNeg a')
        // (NonNeg b')` via Int.NonNeg.add folds + Eq.subst transport), the conjunctive
        // ground/projection engines carry initiation and safety, and the PER-OBLIGATION
        // acceptance basis admits the kernel-checked term — the STRONGER tier (trust base =
        // the Clean kernel, not the SMT checker). Since ay's multi-equality Farkas fix
        // (upstream bump ed035b88), the ay-strict bundle re-check ALSO covers this coupled
        // fragment, so the basis legitimately reads `ay-checker+KERNEL` — an engine
        // IMPROVEMENT (two independent proof-checking legs instead of one).
        //
        // The regression canary is the KERNEL side: whatever ay's fragment does, the
        // consecution's basis must KEEP its kernel coverage (`…+KERNEL` or bare `KERNEL`) —
        // a cert accepted with the kernel silently gone would rest acceptance on the SMT
        // checker alone, masking a coupled-affine kernel-engine regression.
        let report = verify_safety_certificate(&cert);
        assert!(
            report.digest_ok,
            "the freshly-synthesized cert's digest must be self-consistent"
        );
        assert_eq!(
            report.verdict,
            CertVerdict::Accepted,
            "the coupled consecution is kernel-certified (Phase D/E) — the cert must ACCEPT: {}",
            report.detail
        );
        assert_eq!(
            report.kernel_recheck,
            Some(true),
            "the kernel terms must be present and re-accepted: {}",
            report.detail
        );
        assert!(
            report.detail.contains("consecution=ay-checker+KERNEL")
                || report.detail.contains("consecution=KERNEL"),
            "the consecution's acceptance basis must include the KERNEL tier (a kernel-\
             coverage loss would rest acceptance on the SMT checker alone and mask a \
             coupled-affine kernel-engine regression): {}",
            report.detail
        );
        // The consecution obligation specifically must carry a kernel term (the coupled
        // engine) — and, string-matching aside, the term must ACTUALLY re-verify at the
        // obligation type rebuilt from the re-derived spec. This is the drift-proof form
        // of the canary above.
        let consec = cert
            .ay_proof_obligations
            .iter()
            .find(|o| o.name == "consecution")
            .expect("consecution present");
        assert!(
            !consec.clean_cic_term.is_empty(),
            "the coupled-affine kernel engine must have minted the consecution term"
        );
        assert!(
            verify_consecution_term(&cert, &consec.clean_cic_term),
            "the minted consecution term must re-verify in the Clean kernel at the \
             obligation type rebuilt from the re-derived spec"
        );
    }

    /// A certificate built WITHOUT proof bundles (e.g. a pre-Leg-D producer) is
    /// fail-closed to INCONCLUSIVE, never Accepted: the external proof re-check
    /// cannot run, so acceptance is withheld (never a false accept).
    #[cfg(feature = "ay")]
    #[test]
    fn test_bundleless_cert_inconclusive() {
        let cert = build_safety_certificate(
            ACCUMULATOR,
            &acc_config(),
            "x >= 0".to_string(),
            vec![("x".to_string(), "Int".to_string())],
        );
        let report = verify_safety_certificate(&cert);
        assert!(!report.accepted, "a bundle-less cert must not be Accepted");
        assert_eq!(
            report.verdict,
            CertVerdict::Inconclusive,
            "missing proof bundles -> Inconclusive (fail-closed), got: {}",
            report.detail
        );
    }

    /// SOUNDNESS (Leg D assume-coverage gate): a forged certificate that embeds a
    /// VALID proof under the WRONG obligation — the initiation proof relabeled as
    /// the safety obligation — is REJECTED, even though the grafted proof
    /// strict-verifies on its own. Part-2 of the gate re-translates the `safety`
    /// obligation independently and finds the embedded proof is about a DIFFERENT
    /// formula (`Init /\ ~J`, not `J /\ ~Safety`). This is what makes the offline
    /// re-check non-vacuous: `check_proof_strict` alone would accept the grafted
    /// proof (it carries its own self-consistent term table).
    #[cfg(feature = "ay")]
    #[test]
    fn test_legd_rejects_proof_of_wrong_obligation() {
        let cert =
            certify_spec(ACCUMULATOR, &acc_config()).expect("Accumulator must be certifiable");
        assert!(
            verify_safety_certificate(&cert).accepted,
            "baseline must verify"
        );

        let init_bundle = cert
            .ay_proof_obligations
            .iter()
            .find(|o| o.name == "initiation")
            .map(|o| o.bundle_json.clone())
            .expect("initiation bundle present");
        assert!(!init_bundle.is_empty());

        // Graft the (valid) initiation proof onto the safety obligation and
        // recompute the digest, as a forging attacker would.
        let mut forged = cert.clone();
        for o in forged.ay_proof_obligations.iter_mut() {
            if o.name == "safety" {
                o.bundle_json = init_bundle.clone();
            }
        }
        forged.digest = forged.compute_digest();

        let report = verify_safety_certificate(&forged);
        assert!(report.digest_ok, "attacker recomputed a consistent digest");
        assert_eq!(
            report.verdict,
            CertVerdict::Rejected,
            "a proof of the WRONG obligation must be rejected by the coverage gate: {}",
            report.detail
        );
    }

    /// ENGINE-DIVERSE binding (Leg D part-2): the probe cross-check (1) actually
    /// RUNS and AGREES on the honest obligation (Some(true), not a silent None),
    /// and (2) REJECTS when the embedded AY obligation denotes a DIFFERENT
    /// predicate than the spec — caught by tla-eval, a different engine than
    /// BmcTranslator. This is the property that closes the TLA->AY translator
    /// trust: a translation/negation bug that the render-equality (which trusts
    /// BmcTranslator on both sides) would miss is caught here.
    #[cfg(feature = "ay")]
    #[test]
    fn test_legd_probe_catches_translator_mismatch() {
        use crate::ay_bmc::{
            build_probe_eval_ctx, probe_check_obligation_engine_diverse,
            rederive_obligation_inputs, SmtObligation,
        };
        use tla_ay::TermStore;

        let cert =
            certify_spec(ACCUMULATOR, &acc_config()).expect("Accumulator must be certifiable");
        let config = cert.reconstructed_config();

        let emb = cert
            .ay_proof_obligations
            .iter()
            .find(|o| o.name == "initiation")
            .expect("initiation bundle present");
        let parsed: tla_ay::SerializableProofBundle =
            serde_json::from_str(&emb.bundle_json).expect("bundle deserializes");
        let rebuild = || {
            TermStore::from_entries(
                parsed.term_entries.clone(),
                parsed.true_term,
                parsed.false_term,
                parsed.var_counter,
            )
        };

        // (1) HONEST: correct J (= the cert's J; the embedded ~J encodes x<0). The
        // probe must RUN and AGREE on every probe -> Some(true). A None here would
        // mean the engine-diverse check silently did nothing.
        let inputs_ok =
            rederive_obligation_inputs(&cert.spec_src, &config, &cert.invariant_j_tla).unwrap();
        let ctx_ok = build_probe_eval_ctx(&cert.spec_src, &config, &cert.invariant_j_tla).unwrap();
        let mut store_ok = rebuild();
        assert_eq!(
            probe_check_obligation_engine_diverse(
                SmtObligation::Initiation,
                &mut store_ok,
                &parsed.obligation_assertions,
                &inputs_ok,
                &ctx_ok,
                None,
            ),
            Some(true),
            "engine-diverse probe must RUN and AGREE on the honest obligation",
        );

        // (2) MISTRANSLATION: a spec whose J is x>=1 makes the TLA obligation
        // Init/\~(x>=1) = (x=0)/\(x<1), which is TRUE at x=0; but the EMBEDDED
        // obligation encodes ~J = x<0, FALSE at x=0. tla-eval catches the
        // disagreement -> Some(false). (Render-equality, trusting BmcTranslator on
        // both sides, would not have caught a divergence of this kind.)
        let inputs_bad = rederive_obligation_inputs(&cert.spec_src, &config, "x >= 1").unwrap();
        let ctx_bad = build_probe_eval_ctx(&cert.spec_src, &config, "x >= 1").unwrap();
        let mut store_bad = rebuild();
        assert_eq!(
            probe_check_obligation_engine_diverse(
                SmtObligation::Initiation,
                &mut store_bad,
                &parsed.obligation_assertions,
                &inputs_bad,
                &ctx_bad,
                None,
            ),
            Some(false),
            "engine-diverse probe must REJECT when the embedded obligation denotes \
             a different predicate than the spec",
        );
    }

    /// SECOND INDEPENDENT FRONT END (Leg D part-2, third comparand): the
    /// independent parser+evaluator (sharing nothing with tla_core/BmcTranslator)
    /// (1) RUNS and AGREES on the honest obligation, and (2) REJECTS when the
    /// embedded obligation denotes a different predicate than the spec — caught
    /// WITHOUT any shared front end, so even a parse/lower bug would be visible.
    #[cfg(feature = "ay")]
    #[test]
    fn test_legd_indep_frontend_catches_mismatch() {
        use crate::ay_bmc::{
            build_probe_eval_ctx, probe_check_obligation_engine_diverse,
            rederive_obligation_inputs, SmtObligation,
        };
        use crate::cert_indep_frontend::IndepSpec;
        use tla_ay::TermStore;

        let cert =
            certify_spec(ACCUMULATOR, &acc_config()).expect("Accumulator must be certifiable");
        let config = cert.reconstructed_config();
        let emb = cert
            .ay_proof_obligations
            .iter()
            .find(|o| o.name == "initiation")
            .expect("initiation bundle present");
        let parsed: tla_ay::SerializableProofBundle =
            serde_json::from_str(&emb.bundle_json).expect("bundle deserializes");
        let rebuild = || {
            TermStore::from_entries(
                parsed.term_entries.clone(),
                parsed.true_term,
                parsed.false_term,
                parsed.var_counter,
            )
        };
        // tla-eval side correct (so any rejection comes from the INDEP gate only).
        let inputs =
            rederive_obligation_inputs(&cert.spec_src, &config, &cert.invariant_j_tla).unwrap();
        let ctx = build_probe_eval_ctx(&cert.spec_src, &config, &cert.invariant_j_tla).unwrap();

        // (1) HONEST indep front end (J = the cert's J): RUNS and AGREES.
        let indep_ok = IndepSpec::parse(
            &cert.spec_src,
            config.init.as_deref().unwrap(),
            config.next.as_deref().unwrap(),
            &config.invariants,
            &cert.invariant_j_tla,
        )
        .expect("scalar fragment must parse independently");
        let mut store_ok = rebuild();
        assert_eq!(
            probe_check_obligation_engine_diverse(
                SmtObligation::Initiation,
                &mut store_ok,
                &parsed.obligation_assertions,
                &inputs,
                &ctx,
                Some(&indep_ok),
            ),
            Some(true),
            "independent front end must RUN and AGREE on the honest obligation",
        );

        // (2) INDEP parsed with a WRONG J (x>=1): independently disagrees with the
        // embedded obligation (which encodes ~J = x<0). Rejection comes purely from
        // the front-end-independent path (tla-eval side is correct).
        let indep_bad = IndepSpec::parse(
            &cert.spec_src,
            config.init.as_deref().unwrap(),
            config.next.as_deref().unwrap(),
            &config.invariants,
            "x >= 1",
        )
        .expect("scalar fragment must parse");
        let mut store_bad = rebuild();
        assert_eq!(
            probe_check_obligation_engine_diverse(
                SmtObligation::Initiation,
                &mut store_bad,
                &parsed.obligation_assertions,
                &inputs,
                &ctx,
                Some(&indep_bad),
            ),
            Some(false),
            "independent front end must REJECT a mismatched obligation with NO shared front end",
        );
    }

    #[test]
    fn test_tampered_invariant_rejected_even_with_recomputed_digest() {
        let cert = build_safety_certificate(
            ACCUMULATOR,
            &acc_config(),
            "x >= 0".to_string(),
            vec![("x".to_string(), "Int".to_string())],
        );

        // Tamper J to a WRONG invariant and recompute the digest (a forging
        // attacker). The digest now matches, but Leg A re-derives reachable states
        // from spec_src and REFUTES x>=1 at the initial state x=0.
        let mut forged = cert.clone();
        forged.invariant_j_tla = "x >= 1".to_string();
        forged.digest = forged.compute_digest();

        let report = verify_safety_certificate(&forged);
        assert!(report.digest_ok, "attacker recomputed a consistent digest");
        assert!(
            !report.accepted,
            "a tampered invariant must be REJECTED by the eval oracle despite a valid digest"
        );
        assert!(
            matches!(report.oracle, InductiveOracleVerdict::Refuted { .. }),
            "expected oracle Refuted, got {:?}",
            report.oracle
        );
    }

    /// End-to-end: PROVE inductive safety via AY, build a certificate, then
    /// independently re-verify it (Leg A eval oracle) — the full certify/check
    /// loop on a spec explicit BFS can never finish (x diverges).
    #[cfg(feature = "ay")]
    #[test]
    fn test_certify_then_verify_end_to_end() {
        let cert = certify_spec(ACCUMULATOR, &acc_config())
            .expect("Accumulator (x'=x+1, Inv x>=0) must be certifiable");
        assert_eq!(cert.schema, SCHEMA_V1);
        // The proven J entails the safety property (here it is exactly x >= 0).
        assert!(
            cert.invariant_j_tla.contains(">="),
            "expected an inequality invariant, got `{}`",
            cert.invariant_j_tla
        );
        let report = verify_safety_certificate(&cert);
        assert!(
            report.accepted,
            "certified spec must independently verify: {}",
            report.detail
        );
    }

    /// P2 widening: a GUARDED but deadlock-free spec (`x >= 0 /\ x' = x+1`) does NOT fire the
    /// divergence trigger of the narrow BFS-skip prover, yet cert-mode proves its
    /// safety invariant `x >= 0` is 1-inductive and emits a verifiable certificate.
    #[cfg(feature = "ay")]
    #[test]
    fn test_certify_guarded_spec_widened_class() {
        let src = "---- MODULE BoundedCounter ----\n\
                   EXTENDS Integers\n\
                   VARIABLE x\n\
                   Init == x = 0\n\
                   Next == x >= 0 /\\ x' = x + 1\n\
                   Safety == x >= 0\n\
                   ====\n";
        let config = Config {
            init: Some("Init".to_string()),
            next: Some("Next".to_string()),
            invariants: vec!["Safety".to_string()],
            ..Default::default()
        };

        // Cert-mode proves it and the certificate independently verifies.
        let cert =
            certify_spec(src, &config).expect("cert-mode must certify the guarded bounded counter");
        assert!(
            verify_safety_certificate(&cert).accepted,
            "widened certificate must verify: {}",
            verify_safety_certificate(&cert).detail
        );
    }

    /// GAP-2 regression (deadlock-freedom): MODULE Dead (x<3 /\ x'=x+1) DEADLOCKS at
    /// x=3, yet J=x<=3 is 1-inductive and implies Safety (its 3 safety obligations
    /// strict-verify). A forged self-consistent certificate claiming it safe MUST be
    /// REJECTED — via the explicit deadlock-freedom check, not an incidental shortfall.
    #[cfg(feature = "ay")]
    #[test]
    fn test_deadlocking_spec_rejected() {
        let src = "---- MODULE Dead ----\n\
                   EXTENDS Integers\n\
                   VARIABLE x\n\
                   Init == x = 0\n\
                   Next == x < 3 /\\ x' = x + 1\n\
                   Safety == x <= 3\n\
                   ====\n";
        let config = Config {
            init: Some("Init".to_string()),
            next: Some("Next".to_string()),
            invariants: vec!["Safety".to_string()],
            ..Default::default()
        };
        // Adversary forges a self-consistent cert (recomputed digest) for the
        // deadlocking spec; the producer would refuse, but cert-check must too.
        let forged = build_safety_certificate(
            src,
            &config,
            "x <= 3".to_string(),
            vec![("x".to_string(), "Int".to_string())],
        );
        let report = verify_safety_certificate(&forged);
        assert!(
            !report.accepted,
            "a deadlocking spec must be REJECTED: {}",
            report.detail
        );
        assert!(
            report.detail.to_lowercase().contains("deadlock")
                || matches!(report.oracle, InductiveOracleVerdict::Refuted { .. })
                || report.ay_redischarge == Some(false),
            "rejection must trace to deadlock-freedom, got: {}",
            report.detail
        );
    }

    #[test]
    fn test_plain_digest_tamper_rejected() {
        let cert = build_safety_certificate(
            ACCUMULATOR,
            &acc_config(),
            "x >= 0".to_string(),
            vec![("x".to_string(), "Int".to_string())],
        );
        // Flip J without fixing the digest: caught by the digest check.
        let mut tampered = cert.clone();
        tampered.invariant_j_tla = "x >= 1".to_string();
        let report = verify_safety_certificate(&tampered);
        assert!(!report.digest_ok, "stale digest must be detected");
        assert!(!report.accepted);
    }
}
