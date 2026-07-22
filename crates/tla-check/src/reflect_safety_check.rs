// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! `reflect-check`: an ADDITIVE, embedder-free re-discharge of an explicit-state fixpoint
//! certificate's `R ⊆ Safety` obligation through the KERNEL-DEFINED reflected evaluator
//! ([`crate::reflect`]'s `TyReflectEvalP`) instead of the shallow Rust embedder
//! ([`crate::cleancic::embed_pred_ir`]).
//!
//! ## Why this exists (the trust delta)
//!
//! The primary `safety_general` leg proves `⋀_{s∈R} ⟦Safety⟧(s) = Bool.true` by building each
//! `⟦Safety⟧(s)` kernel term OP BY OP in trusted Rust (`Lt` ↦ `Nat.ble (a+1) b`, `And` ↦
//! `Bool.and`, …). A bug in that construction silently changes what the kernel is asked to prove.
//! This re-check discharges the SAME obligation with the embedder OUT of the loop: it quotes the
//! recognized `Safety_ir` with the line-auditable 1:1 quoter and lets the KERNEL reduce the
//! deep evaluator `TyReflectEvalP` (whose op choices are kernel-checked DEFINITION data, admitted
//! once via `add_inductive`/`add_decl`, not per-obligation Rust).
//!
//! **What this leg does NOT rely on:** `cleancic::embed_pred_ir`'s per-node kernel-term
//! construction. **What it still trusts:** the quoter ([`crate::reflect::quote_pred`] — one
//! constructor per match arm), the clean kernel, and the recognizer that produced `Safety_ir`
//! (bound to the spec by re-derivation below).
//!
//! ## Fail-closed discipline (SOUNDNESS-CRITICAL)
//!
//! A reachable state the deep evaluator reduces to `Bool.false` is a genuine unsafe state and is
//! reported [`ReflectCheckVerdict::NotSafe`] — NEVER `Accepted`. Any IR outside the reflect scalar
//! fragment (Set/quantifier/Seq node, out-of-bounds column) is [`ReflectCheckVerdict::Inconclusive`],
//! never a silent skip. The discharged `(Safety_ir, R)` are BOUND to the cert's own spec by an
//! independent re-derivation (Leg-E): a tampered `spec_src`/`reachable`/`safety_pred` yields
//! [`ReflectCheckVerdict::Rejected`]. This module modifies NO existing cert struct or encoding arm.

// `SafetyCertificate` is referenced only by the `clean-cic`-gated re-check entry point and its
// tests; gate the import so the feature-light build (enum-only) stays warning-clean.
#[cfg(feature = "clean-cic")]
use crate::cert::SafetyCertificate;

/// Verdict of the reflected `R ⊆ Safety` re-check.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReflectCheckVerdict {
    /// Every reachable state's reflected `⟦Safety⟧(s)` reduced to `Bool.true` under the deep
    /// evaluator, AND the discharged `(Safety_ir, R, sorts)` re-derive IDENTICALLY from the cert's
    /// own spec (spec-bound). Carries the reachable-set size.
    Accepted {
        /// Number of reachable states discharged through the deep evaluator.
        states: usize,
    },
    /// A reachable state FALSIFIES the invariant under the deep evaluator (the kernel reduced its
    /// reflected obligation to `Bool.false`) — a genuine unsafe state. NEVER a false accept.
    NotSafe {
        /// The reachable tuple whose reflected `⟦Safety⟧(state)` reduced to `Bool.false`.
        state: Vec<u64>,
    },
    /// The discharged obligation does not match the one re-derived from the cert's spec: a tampered
    /// `spec_src` / `reachable` / `safety_pred`, or a non-deterministic re-derivation.
    Rejected(String),
    /// The reflected lane declines (fail-closed): not an explicit-state cert, the primary
    /// nonneg-`⋀ x≥0` lane (no general `safety_pred`), an out-of-fragment IR, an OOB column, or a
    /// build/feature gap. NOT a verdict on the certificate.
    Inconclusive(String),
}

/// Re-discharge a certificate's `R ⊆ Safety` obligation through the reflected (deep-embedding)
/// evaluator — see the module docs for the trust story. `clean-cic`-only (the reflected kernel
/// evaluator + the spec re-derivation both need the linked kernel).
#[cfg(feature = "clean-cic")]
pub fn reflect_check_safety_cert(cert: &SafetyCertificate) -> ReflectCheckVerdict {
    use crate::reflect::{reflect_safety_over_reachable, ReflectSafetyOutcome};

    let Some(fp) = &cert.explicit_fixpoint else {
        return ReflectCheckVerdict::Inconclusive(
            "not an explicit-state fixpoint certificate (no `explicit_fixpoint` leg)".into(),
        );
    };
    let Some(safety_ir) = &fp.safety_pred else {
        return ReflectCheckVerdict::Inconclusive(
            "primary nonneg lane: the invariant is the `⋀ x≥0` tuple shape (no general \
             `safety_pred`); the reflected safety leg covers only the general scalar fragment"
                .into(),
        );
    };
    // The stored IR must be a truth-direction-EXACT STATE predicate — the SAME gate the shallow
    // `verify_explicit_state_cert` applies. A primed or over-approximating IR is out of the
    // reflected state-predicate discipline (and would make `s == sp` read the current state for a
    // prime): decline fail-closed rather than reduce it.
    if crate::refinement_cert::pred_mentions_prime(safety_ir)
        || !crate::refinement_cert::pred_exact(safety_ir, &fp.sorts)
    {
        return ReflectCheckVerdict::Inconclusive(
            "stored safety IR is primed or not truth-direction-exact — out of the reflected \
             state-predicate fragment"
                .into(),
        );
    }

    // (1) SOUNDNESS-CRITICAL discharge: reduce ⟦Safety⟧(s) through the DEEP evaluator over the
    // STORED reachable set. A reachable state the kernel reduces to Bool.false is a genuine unsafe
    // state — reported NotSafe, never Safe. Runs on the STORED (safety_pred, reachable) so a
    // tamper that INTRODUCES a violating state is caught HERE by the kernel, not only by binding.
    match reflect_safety_over_reachable(safety_ir, &fp.reachable) {
        ReflectSafetyOutcome::Safe => {}
        ReflectSafetyOutcome::NotSafe { state, .. } => {
            return ReflectCheckVerdict::NotSafe { state };
        }
        ReflectSafetyOutcome::Inconclusive(reason) => {
            return ReflectCheckVerdict::Inconclusive(reason);
        }
    }

    // (2) BIND the discharged (Safety_ir, R, sorts) to the cert's own spec (Leg-E): re-derive the
    // fixpoint from `spec_src` and require the reflected obligation was over EXACTLY the spec's
    // invariant and reachable set. Without this, a tamper that WEAKENS `safety_pred` to a predicate
    // R happens to satisfy would slip past step (1). The re-derivation is the shallow lane's, used
    // ONLY to bind — the reflected discharge above did NOT depend on it.
    let config = config_from_cert(cert);
    let Some(re) =
        crate::explicit_fixpoint_cert::certify_explicit_state_spec(&cert.spec_src, &config)
    else {
        return ReflectCheckVerdict::Inconclusive(
            "could not independently re-derive the fixpoint from the cert's spec (cannot bind the \
             reflected obligation to the spec)"
                .into(),
        );
    };
    if re.safety_pred.as_ref() != Some(safety_ir) {
        return ReflectCheckVerdict::Rejected(
            "the invariant re-recognized from the spec differs from the cert's stored `safety_pred` \
             (tampered invariant or spec)"
                .into(),
        );
    }
    if re.reachable != fp.reachable {
        return ReflectCheckVerdict::Rejected(
            "the reachable set re-enumerated from the spec differs from the cert's stored \
             `reachable` (tampered reachable set or spec)"
                .into(),
        );
    }
    if re.sorts != fp.sorts {
        return ReflectCheckVerdict::Rejected(
            "the column sorts re-inferred from the spec differ from the cert's stored `sorts`"
                .into(),
        );
    }
    ReflectCheckVerdict::Accepted {
        states: fp.reachable.len(),
    }
}

// ===========================================================================
// R2 MILESTONE: the reflected ALL-LEGS discharge. Extends the `R⊆Safety` re-check
// above to ALSO discharge the two COMPLETENESS legs — Init-completeness and
// Next-completeness (closure) — through the reflected evaluator, so a spec that
// passes all THREE reflected legs has its safety verdict backed with the shallow
// embedder OUT of every obligation.
// ===========================================================================

/// The domain-coverage BASIS of a reflected all-legs discharge: whether every completeness-leg
/// product-domain axis is its column's FULL sort universe (coverage by construction — no per-column
/// bound rule in the trust story) or ≥1 axis rests on a TRUSTED-RUST structural bound rule
/// (`crate::cleancic::{next,init}_domain_bounds_from_ir`). Classified STRUCTURALLY from the IR and
/// deliberately EMBEDDER-FREE: the `KernelProven` coverage upgrade uses the symbolic embedder
/// `embed_pred_ir_sym`, so it is NOT consulted on the reflected path (that would re-introduce the
/// embedder family into the coverage argument).
#[cfg(feature = "clean-cic")]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReflectCoverageBasis {
    /// Every completeness-leg axis is its column's full sort universe (`{0,1}` Bool / `2^K−1` Set):
    /// `D ⊇ Succ(R)` and `D ⊇ {s:Init(s)}` hold by construction — no per-column Rust bound rule is
    /// trusted (sort-faithfulness itself still rests on the enumerated values + recognizer).
    ConstructionComplete,
    /// ≥1 axis's domain bound is a trusted-Rust structural rule (Int primed-upper-bound / stutter /
    /// Eq-pin / Or-split). The reflected legs still REDUCE (closure/init-coverage proved RELATIVE to
    /// `D`), but `D ⊇ Succ(R)`/`D ⊇ Init` on these axes rests on TY's rule — surfaced, not hidden.
    RustDerived {
        /// Next-leg column indices whose axis bound is Rust-derived.
        next_rust: Vec<usize>,
        /// Init-leg column indices whose axis bound is Rust-derived.
        init_rust: Vec<usize>,
    },
}

/// Verdict of the reflected ALL-LEGS re-check ([`reflect_check_safety_cert_full`]).
#[cfg(feature = "clean-cic")]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReflectFullVerdict {
    /// All three reflected legs discharged AND every IR + sorts RE-RECOGNIZE identically from the spec
    /// (a RECOGNITION-ONLY spec-bind — no re-enumeration, no embedder): `R⊆Safety`, `Init⊆R`, and `R`
    /// closed under `Next`, each reduced by the kernel-defined evaluator with the shallow embedder OUT of
    /// the obligation. Trust base = `kernel + recognizer + quoter` (R is enumerator-PROVIDED but its
    /// soundness as an inductive invariant is kernel-verified, so the enumerator is NOT trusted).
    Certified {
        /// `|R|` — reachable states carried through the safety + closure legs.
        states: usize,
        /// `|D_init|` — the re-derived Init completeness domain size.
        init_domain: usize,
        /// `|D_next|` — the re-derived Next completeness domain size.
        next_domain: usize,
        /// `|R| × |D_next|` — the closure implication pairs the kernel discharged.
        next_pairs: usize,
        /// The domain-coverage basis (whether `D`'s axes are construction-complete or Rust-derived).
        coverage: ReflectCoverageBasis,
    },
    /// A reachable state falsifies the invariant under the deep evaluator (`R⊆Safety` reduced to
    /// `Bool.false`). NEVER a false accept.
    NotSafe {
        /// The reachable tuple whose reflected `⟦Safety⟧(state)` reduced to `Bool.false`.
        state: Vec<u64>,
    },
    /// `R` is NOT closed under `Next`: a successor `sp` of some `s∈R` (with `sp∈D`) satisfies
    /// `Next(s,sp)` but `sp ∉ R` — the reflected closure leg reduced to `Bool.false`. The decisive
    /// non-closed-R guard (a dropped/missing successor is caught HERE by the kernel).
    NotClosed {
        /// The source state whose successor escapes `R`.
        s: Vec<u64>,
        /// The escaping successor (`Next(s,sp)` holds, `sp ∉ R`).
        sp: Vec<u64>,
    },
    /// An `Init`-satisfying domain state is MISSING from `R` (Init-completeness reduced to
    /// `Bool.false`): `Init(s)` holds but `s ∉ R`.
    NotInitComplete {
        /// The `Init` state absent from `R`.
        s: Vec<u64>,
    },
    /// A discharged IR / `R` / sorts does not match the one re-derived from the cert's spec — a
    /// tampered `spec_src` / `reachable` / IR.
    Rejected(String),
    /// The reflected all-legs lane declines (fail-closed): not an explicit-state general-leg cert, an
    /// out-of-fragment IR, an OOB column, a domain over the cap, a build gap, or (under
    /// `--require-domain-complete`) a Rust-derived domain. NOT a verdict on the certificate.
    Inconclusive(String),
}

/// Re-discharge ALL THREE of a certificate's safety obligations — `R⊆Safety`, Init-completeness
/// (`Init⊆R`), and Next-completeness / closure (`R` closed under `Next`) — through the reflected
/// (deep-embedding) evaluator, with the shallow `embed_pred_ir` OUT of every leg.
///
/// The two completeness legs re-evaluate the cert's STORED general `Init`/`Next` IR over a product
/// domain `D` RE-DERIVED from that IR by the SAME structural bound rules the cert uses (never trusting
/// the serialized `hi`), composing the deep predicate evaluator `TyReflectEvalP` with the deep
/// membership evaluator `TyReflectMem`: the kernel reduces `Init(s) ⇒ s∈R` for every `s∈D` and
/// `Next(s,sp) ⇒ sp∈R` for every `s∈R, sp∈D`. Then every discharged IR + `R` + sorts is BOUND to the
/// spec by re-derivation.
///
/// `require_domain_complete`: when set, DECLINE ([`ReflectFullVerdict::Inconclusive`]) any cert whose
/// completeness domain rests on a trusted-Rust bound rule (an axis that is not its column's full
/// universe) — the reflected legs still reduce, but closure/init-coverage is only RELATIVE to a
/// Rust-bounded `D`. When unset, such a cert still CERTIFIES and the residual is surfaced in the
/// carried [`ReflectCoverageBasis`].
#[cfg(feature = "clean-cic")]
pub fn reflect_check_safety_cert_full(
    cert: &SafetyCertificate,
    require_domain_complete: bool,
) -> ReflectFullVerdict {
    use crate::reflect::{
        reflect_init_completeness_over_domain, reflect_next_completeness_over_domain,
        reflect_safety_over_reachable, ReflectClosureOutcome, ReflectInitOutcome,
        ReflectSafetyOutcome,
    };

    let Some(fp) = &cert.explicit_fixpoint else {
        return ReflectFullVerdict::Inconclusive(
            "not an explicit-state fixpoint certificate (no `explicit_fixpoint` leg)".into(),
        );
    };
    let Some(safety_ir) = &fp.safety_pred else {
        return ReflectFullVerdict::Inconclusive(
            "primary nonneg lane: no general `safety_pred` (the reflected all-legs path covers only \
             the general scalar fragment)"
                .into(),
        );
    };
    // The reflected completeness legs re-evaluate the STORED general Init/Next IR; require both. A
    // cert resting on the enumerated `image ⊆ R` or the single-Int affine/literal shortcut is out of
    // this path (fail-closed, never a silent pass on an unverified completeness obligation).
    let (Some(init_domir), Some(next_domir)) = (&fp.init_pred, &fp.next_pred) else {
        return ReflectFullVerdict::Inconclusive(
            "the reflected all-legs path requires the GENERAL kernel-re-evaluation legs (init_pred + \
             next_pred); this cert uses the affine/literal shortcut or rests on the enumerated image"
                .into(),
        );
    };
    if fp.init_shape.is_some() || fp.next_shape.is_some() {
        return ReflectFullVerdict::Inconclusive(
            "cert carries the single-Int affine/literal SHORTCUT legs (init_shape/next_shape) — out \
             of the reflected general-completeness fragment"
                .into(),
        );
    }
    // Safety IR must be a truth-direction-EXACT STATE predicate (same gate as the R⊆Safety leg).
    if crate::refinement_cert::pred_mentions_prime(safety_ir)
        || !crate::refinement_cert::pred_exact(safety_ir, &fp.sorts)
    {
        return ReflectFullVerdict::Inconclusive(
            "stored safety IR is primed or not truth-direction-exact — out of the reflected \
             state-predicate fragment"
                .into(),
        );
    }

    // (1) R ⊆ Safety — reflected over the STORED reachable set (a tamper INTRODUCING a violating
    // state is caught HERE by the kernel).
    match reflect_safety_over_reachable(safety_ir, &fp.reachable) {
        ReflectSafetyOutcome::Safe => {}
        ReflectSafetyOutcome::NotSafe { state, .. } => {
            return ReflectFullVerdict::NotSafe { state };
        }
        ReflectSafetyOutcome::Inconclusive(reason) => {
            return ReflectFullVerdict::Inconclusive(reason);
        }
    }

    let n_cols = fp.sorts.len();
    let cap = crate::explicit_fixpoint_cert::DEFAULT_FIXPOINT_STATE_CAP;

    // (2) Init-completeness — reflected over D_init RE-DERIVED from the stored Init IR (never the
    // serialized `hi`): `∀ s∈D: Init(s) ⇒ s∈R`.
    let Some(init_bounds) =
        crate::cleancic::init_domain_bounds_from_ir(&init_domir.pred, n_cols, &fp.sorts)
    else {
        return ReflectFullVerdict::Inconclusive(
            "could not re-derive the Init completeness domain bounds from the stored Init IR"
                .into(),
        );
    };
    let Some(init_domain) = crate::cleancic::product_domain(&init_bounds, cap) else {
        return ReflectFullVerdict::Inconclusive(
            "the Init product domain exceeds the state cap (or overflows)".into(),
        );
    };
    match reflect_init_completeness_over_domain(&init_domir.pred, &init_domain, &fp.reachable) {
        ReflectInitOutcome::Complete { .. } => {}
        ReflectInitOutcome::NotComplete { s, .. } => {
            return ReflectFullVerdict::NotInitComplete { s };
        }
        ReflectInitOutcome::Inconclusive(reason) => {
            return ReflectFullVerdict::Inconclusive(reason);
        }
    }

    // (3) Next-completeness / closure — reflected over D_next RE-DERIVED from the stored Next IR:
    // `∀ s∈R, sp∈D: Next(s,sp) ⇒ sp∈R`. THE load-bearing closure leg (a non-closed R reduces it to
    // Bool.false HERE).
    let Some(next_bounds) = crate::cleancic::next_domain_bounds_from_ir(
        &next_domir.pred,
        n_cols,
        &fp.reachable,
        &fp.sorts,
    ) else {
        return ReflectFullVerdict::Inconclusive(
            "could not re-derive the Next completeness domain bounds from the stored Next IR"
                .into(),
        );
    };
    let Some(next_domain) = crate::cleancic::product_domain(&next_bounds, cap) else {
        return ReflectFullVerdict::Inconclusive(
            "the Next product domain exceeds the state cap (or overflows)".into(),
        );
    };
    let next_pairs = match reflect_next_completeness_over_domain(
        &next_domir.pred,
        &fp.reachable,
        &next_domain,
    ) {
        ReflectClosureOutcome::Closed { pairs } => pairs,
        ReflectClosureOutcome::NotClosed { s, sp, .. } => {
            return ReflectFullVerdict::NotClosed { s, sp };
        }
        ReflectClosureOutcome::Inconclusive(reason) => {
            return ReflectFullVerdict::Inconclusive(reason);
        }
    };

    // (4) RECOGNITION-ONLY SPEC-BIND: re-derive `(sorts, Init IR, Next IR, Safety IR)` from `spec_src`
    // with the enumerator AND the shallow embedder ([`crate::cleancic::embed_pred_ir`]) OUT of the loop
    // ([`crate::explicit_fixpoint_cert::recognize_spec_fixpoint_irs`]: parse → inline → derive sorts from
    // the invariants' TYPE declarations → recognize each body with the 1:1 recognizer), and require
    // structural equality with the stored IRs + sorts. This ties the reflected obligations to the spec
    // WITHOUT re-running the shallow certifier — so the verdict's trust base is exactly
    // `kernel + recognizer + quoter` (NO enumerator, NO embedder).
    //
    // R is DELIBERATELY NOT re-enumerated (and NOT compared to any enumerated set): the reflected legs
    // (1)-(3) already kernel-verify that R is a SOUND INDUCTIVE INVARIANT (`Init⊆R ∧ R-closed ∧ R⊆Safety`
    // ⇒ `Reachable ⊆ R ⊆ Safety`), and ANY sound inductive invariant proves safety — R need not equal the
    // enumerated reachable set. A dropped successor / injected unsafe state is caught by legs (1)-(3), not
    // by an enumeration bind.
    let config = config_from_cert(cert);
    let Some(re) =
        crate::explicit_fixpoint_cert::recognize_spec_fixpoint_irs(&cert.spec_src, &config)
    else {
        return ReflectFullVerdict::Inconclusive(
            "could not RECOGNIZE (sorts, Init, Next, Safety) from the cert's spec without enumerating: a \
             column sort not spec-derivable from a type invariant, or a body outside the recognizer \
             fragment (cannot bind the reflected obligations to the spec)"
                .into(),
        );
    };
    // The sorts DERIVED STRUCTURALLY from the spec's type invariants must equal the stored sorts — a
    // mistyped/tampered column (a FuncEnum relabelled `Int`, a shrunk/grown label set) is REJECTED here.
    if re.sorts != fp.sorts {
        return ReflectFullVerdict::Rejected(
            "the column sorts DERIVED from the spec's type invariants differ from the stored `sorts` \
             (a mistyped or tampered column sort)"
                .into(),
        );
    }
    // Each IR RE-RECOGNIZED from the spec body (over the derived sorts) must equal the stored one. A
    // tampered stored IR — one that still passes the reflected legs over the stored R but does NOT match
    // the spec — is caught HERE (the recognizer re-derives the TRUE IR from the actual spec body).
    if &re.safety_ir != safety_ir {
        return ReflectFullVerdict::Rejected(
            "the invariant RE-RECOGNIZED from the spec differs from the stored `safety_pred` (tampered \
             invariant IR or spec)"
                .into(),
        );
    }
    if re.init_ir != init_domir.pred {
        return ReflectFullVerdict::Rejected(
            "the Init IR RE-RECOGNIZED from the spec differs from the stored `init_pred` (tampered Init \
             IR or spec)"
                .into(),
        );
    }
    if re.next_ir != next_domir.pred {
        return ReflectFullVerdict::Rejected(
            "the Next IR RE-RECOGNIZED from the spec differs from the stored `next_pred` (tampered Next \
             IR or spec)"
                .into(),
        );
    }

    // (5) Domain-coverage basis — classified STRUCTURALLY from the IR (embedder-free).
    let coverage = classify_reflect_coverage(fp, n_cols);
    if require_domain_complete {
        if let ReflectCoverageBasis::RustDerived {
            next_rust,
            init_rust,
        } = &coverage
        {
            return ReflectFullVerdict::Inconclusive(format!(
                "--require-domain-complete: the completeness domain rests on TRUSTED-RUST bound rules \
                 (Next-leg column(s) {next_rust:?}, Init-leg column(s) {init_rust:?}); the reflected \
                 legs reduced (closure/init-coverage RELATIVE to D), but `D ⊇ Succ(R)`/`D ⊇ Init` on \
                 these axes is not construction-complete"
            ));
        }
    }

    ReflectFullVerdict::Certified {
        states: fp.reachable.len(),
        init_domain: init_domain.len(),
        next_domain: next_domain.len(),
        next_pairs,
        coverage,
    }
}

/// Classify the domain-coverage basis of a cert's general completeness legs STRUCTURALLY from the IR
/// (no embedder, no `KernelProven` upgrade): each axis is either its column's full universe
/// ([`crate::cleancic::DomainCoverage::UniverseComplete`]) or a trusted-Rust bound
/// ([`crate::cleancic::DomainCoverage::RustDerived`]).
#[cfg(feature = "clean-cic")]
fn classify_reflect_coverage(
    fp: &crate::explicit_fixpoint_cert::ExplicitFixpointCert,
    n_cols: usize,
) -> ReflectCoverageBasis {
    use crate::cleancic::DomainCoverage;
    let rust_cols = |cov: Option<Vec<(u64, DomainCoverage)>>| -> Vec<usize> {
        cov.map(|v| {
            v.iter()
                .enumerate()
                .filter(|(_, (_, c))| *c == DomainCoverage::RustDerived)
                .map(|(i, _)| i)
                .collect()
        })
        .unwrap_or_default()
    };
    let next_rust = fp
        .next_pred
        .as_ref()
        .map(|d| {
            rust_cols(crate::cleancic::next_domain_bounds_cov_from_ir(
                &d.pred,
                n_cols,
                &fp.reachable,
                &fp.sorts,
            ))
        })
        .unwrap_or_default();
    let init_rust = fp
        .init_pred
        .as_ref()
        .map(|d| {
            rust_cols(crate::cleancic::init_domain_bounds_cov_from_ir(
                &d.pred, n_cols, &fp.sorts,
            ))
        })
        .unwrap_or_default();
    if next_rust.is_empty() && init_rust.is_empty() {
        ReflectCoverageBasis::ConstructionComplete
    } else {
        ReflectCoverageBasis::RustDerived {
            next_rust,
            init_rust,
        }
    }
}

/// Reconstruct the [`Config`](crate::Config) the cert was minted under from its PUBLIC fields
/// (mirrors `SafetyCertificate::reconstructed_config`, replicated here to keep this module additive
/// — it touches no existing file's internals). Shared with the AST-direct lane
/// ([`crate::reflect_ast_direct`]), which reads the SAME deterministic config.
#[cfg(feature = "clean-cic")]
pub(crate) fn config_from_cert(cert: &SafetyCertificate) -> crate::Config {
    crate::Config {
        init: cert.init.clone(),
        next: cert.next.clone(),
        invariants: cert.invariants.clone(),
        constants: cert.constants.iter().cloned().collect(),
        constants_order: cert.constants.iter().map(|(n, _)| n.clone()).collect(),
        ..Default::default()
    }
}

// ===========================================================================
// RECOGNIZER-EXACTNESS CROSS-CHECK (proof-roadmap §2, B2 — task #17 wiring).
//
// The recognizer-exactness claim `embed_pred_ir(recognize_pred(P), s) ⇒ Bool.true
// ⟺ TLA_eval(P, s) = TRUE` rests today on the SYNTACTIC filter `pred_exact` +
// violated-twin tests + the AST-rooted `cross_check_pred_embedders` (which cross-checks
// the RECOGNITION step, AST→IR, against a second SHALLOW embedder that shares the same
// Rust op conventions). This adds an INDEPENDENT SEMANTIC leg for the EMBEDDING step:
// for every reachable state `s`, the reflect-v2 DEEP evaluator (`TyReflectEvalP`, whose
// per-op realization is kernel-checked DEFINITION data, not per-obligation Rust) evaluates
// the recognized `safety_ir` and its verdict is compared to the shallow leg's.
//
// SOUNDNESS: this can only DECLINE or corroborate — never accept more. At the mint wiring
// point the shallow safety leg has ALREADY reduced `⟦Safety⟧(s)` to `Bool.true` for every
// `s ∈ R` (else no cert). So a reflect DISAGREE (deep evaluator reduces the SAME IR to
// `Bool.false` at some `s ∈ R`) means the two op-realizations of the recognized invariant
// disagree on truth — a genuine recognizer/embedder EXACTNESS bug. FAIL CLOSED.
//
// HONEST COVERAGE: reflect v2 covers only the SCALAR fragment (`quote_pred`: Bool combinators
// over `{Eq,Neq,Lt,Leq,Gt,Geq,Unchanged}` on scalar-`Nat` cells — Int comparisons, enum-index
// equality, Bool-cell equality). Set/quantifier/Seq/`SetCard`/`CountFold` nodes are OUT of scope
// (`Uncovered`), and there the existing `pred_exact`/twin guarantee still governs. So this is an
// ADDITIVE semantic corroboration WHERE COVERED, not a replacement of `pred_exact`.
// ===========================================================================

/// Outcome of the additive reflect-v2 exactness cross-check over a certifying spec's reachable set.
#[cfg(feature = "clean-cic")]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReflectExactnessOutcome {
    /// The reflect-v2 DEEP evaluator reduced the recognized `safety_ir` to `Bool.true` on every
    /// COVERED reachable state (`covered` of `total`), corroborating the shallow safety leg's
    /// acceptance with an INDEPENDENT op-realization. `covered < total` means the remaining states'
    /// IR is outside reflect v2's scalar fragment (still governed by `pred_exact`).
    Corroborated {
        /// Reachable states the deep evaluator reduced to `Bool.true` (in the scalar fragment).
        covered: usize,
        /// `|R|`.
        total: usize,
    },
    /// A RECOGNIZER-EXACTNESS SOUNDNESS BUG: the deep evaluator reduced the recognized `safety_ir`
    /// to `Bool.false` at a reachable state the shallow safety leg accepted — the two op-realizations
    /// of the same IR disagree on truth. FAIL CLOSED (the caller must decline the certificate).
    Mismatch {
        /// The reachable tuple whose reflected `⟦Safety⟧(s)` reduced to `Bool.false`.
        state: Vec<u64>,
        /// The corroborator's disagreement detail.
        detail: String,
    },
    /// Reflect v2 covers NONE of the reachable states for this `safety_ir` (out of the scalar
    /// fragment on every state, or the reflect env failed to build). The exactness guarantee for
    /// this spec stays SYNTACTIC (`pred_exact` + twins). NOT a verdict on the certificate.
    Uncovered,
}

/// Additive reflect-v2 exactness cross-check: for each reachable state `s`, evaluate the recognized
/// `safety_ir` through the INDEPENDENT deep kernel-defined evaluator (`TyReflectEvalP`, reflect v2)
/// as a STATE predicate (`sp = s`) and compare to the shallow safety leg (whose mint invariant is
/// `⟦Safety⟧(s) = Bool.true` for every `s ∈ R`). Continues past out-of-fragment states to record
/// coverage, fail-closes on the FIRST reflect disagreement. Never re-runs the shallow embedder — the
/// deep evaluator's op choices are kernel-checked definition data, so this is a genuinely independent
/// semantic reference for the covered fragment.
#[cfg(feature = "clean-cic")]
pub fn reflect_exactness_over_reachable(
    safety_ir: &crate::explicit_fixpoint_cert::PredIR,
    reachable: &[Vec<u64>],
) -> ReflectExactnessOutcome {
    use crate::reflect::{reflect_corroborate, ReflectOutcome};
    let mut covered = 0usize;
    for s in reachable {
        // A safety invariant is a STATE predicate ⇒ sp = s. `expect_true`: the shallow leg already
        // accepted this state, so the deep evaluator MUST also reduce it to Bool.true.
        match reflect_corroborate(safety_ir, s, s, true) {
            ReflectOutcome::Corroborated => covered += 1,
            ReflectOutcome::Disagree(detail) => {
                return ReflectExactnessOutcome::Mismatch {
                    state: s.clone(),
                    detail,
                };
            }
            // Out of reflect v2's scalar fragment / OOB column on this state — uncheckable here,
            // but NOT a decline: keep scanning so the coverage count is honest.
            ReflectOutcome::Unavailable(_) => {}
        }
    }
    if covered > 0 {
        ReflectExactnessOutcome::Corroborated {
            covered,
            total: reachable.len(),
        }
    } else {
        ReflectExactnessOutcome::Uncovered
    }
}

#[cfg(all(test, feature = "clean-cic"))]
mod tests {
    use super::*;
    use crate::explicit_fixpoint_cert::{ColSort, ExplicitFixpointCert, PredIR, ValIR};

    /// A minimal HourClock-SHAPED explicit-state cert: single Int column `hr`, R = {1..12},
    /// safety `1 ≤ hr ∧ hr ≤ 12`. The kernel-term byte fields are irrelevant to the reflected
    /// leg (it re-quotes `safety_pred` itself), so they stay empty. `spec_src`/config carry the
    /// REAL HourClock so the spec-binding re-derivation reproduces the same (safety_pred, R).
    fn hourclock_cert(safety: PredIR, reachable: Vec<Vec<u64>>) -> SafetyCertificate {
        const SRC: &str = "\
---------------------- MODULE HourClock ----------------------
EXTENDS Naturals
VARIABLE hr
HCini  ==  hr \\in (1 .. 12)
HCnxt  ==  hr' = IF hr # 12 THEN hr + 1 ELSE 1
HC  ==  HCini /\\ [][HCnxt]_hr
--------------------------------------------------------------
THEOREM  HC => []HCini
==============================================================
";
        let fp = ExplicitFixpointCert {
            reachable,
            init_values: (1..=12u64).map(|h| vec![h]).collect(),
            image: (1..=12u64).map(|h| vec![h]).collect(),
            sorts: vec![ColSort::Int],
            safety_term: Vec::new(),
            init_member_terms: Vec::new(),
            closed_member_terms: Vec::new(),
            next_shape: None,
            next_completeness: None,
            init_shape: None,
            init_completeness: None,
            next_pred: None,
            next_general_completeness: None,
            init_pred: None,
            init_general_completeness: None,
            unbounded_invariant: None,
            safety_pred: Some(safety),
            safety_general: Some(vec![0]), // non-empty marker; the reflected leg never reads it
            init_member_reflected: None,
            closed_member_reflected: None,
            deadlock_free: None,
            deadlock_scan: Default::default(),
        };
        let mut cert = crate::cert::build_explicit_fixpoint_certificate(
            SRC,
            &crate::Config {
                init: Some("HCini".into()),
                next: Some("HCnxt".into()),
                invariants: vec!["HCini".into()],
                ..Default::default()
            },
            fp,
        );
        cert.digest = cert.compute_digest();
        cert
    }

    fn and(a: PredIR, b: PredIR) -> PredIR {
        PredIR::And(Box::new(a), Box::new(b))
    }

    /// POSITIVE: the genuine HourClock cert shape → the reflected leg ACCEPTS (safe over R AND the
    /// obligation binds to the re-derived spec).
    #[test]
    fn reflect_check_accepts_hourclock_and_binds_to_spec() {
        let safety = and(
            PredIR::Leq(ValIR::Lit(1), ValIR::Var(0)),
            PredIR::Leq(ValIR::Var(0), ValIR::Lit(12)),
        );
        let r: Vec<Vec<u64>> = (1..=12u64).map(|h| vec![h]).collect();
        let cert = hourclock_cert(safety, r);
        assert_eq!(
            reflect_check_safety_cert(&cert),
            ReflectCheckVerdict::Accepted { states: 12 }
        );
    }

    /// DECISIVE SOUNDNESS: a cert whose stored invariant is `hr < 12` over the reachable set that
    /// includes `hr = 12` must be NotSafe at [12] — the reflected leg catches the violation the
    /// kernel reduces to Bool.false. (Such a cert can only arise by TAMPER, since `ty certify`
    /// fail-closes minting a violated invariant; the reflected leg is the second, independent guard.)
    #[test]
    fn reflect_check_notsafe_on_reachable_violation() {
        let bad = PredIR::Lt(ValIR::Var(0), ValIR::Lit(12));
        let r: Vec<Vec<u64>> = (1..=12u64).map(|h| vec![h]).collect();
        let cert = hourclock_cert(bad, r);
        assert_eq!(
            reflect_check_safety_cert(&cert),
            ReflectCheckVerdict::NotSafe { state: vec![12] }
        );
    }

    /// TAMPER (spec-binding): a stored invariant that is SAFE over R but WIDER than the spec's
    /// (`0 ≤ hr ∧ hr ≤ 100`) passes step (1) but is Rejected by the spec re-derivation — the
    /// re-recognized invariant is HourClock's `1 ≤ hr ∧ hr ≤ 12`, not the widened one.
    #[test]
    fn reflect_check_rejects_widened_invariant_via_spec_binding() {
        let wide = and(
            PredIR::Leq(ValIR::Lit(0), ValIR::Var(0)),
            PredIR::Leq(ValIR::Var(0), ValIR::Lit(100)),
        );
        let r: Vec<Vec<u64>> = (1..=12u64).map(|h| vec![h]).collect();
        let cert = hourclock_cert(wide, r);
        assert!(matches!(
            reflect_check_safety_cert(&cert),
            ReflectCheckVerdict::Rejected(_)
        ));
    }

    /// TAMPER (reachable set): the genuine invariant, but a reachable entry mutated to an
    /// out-of-range `hr = 13` → the reflected leg reports NotSafe at [13] (the deep evaluator
    /// reduces `13 ≤ 12` to Bool.false) — caught BEFORE spec-binding.
    #[test]
    fn reflect_check_notsafe_on_tampered_reachable_entry() {
        let safety = and(
            PredIR::Leq(ValIR::Lit(1), ValIR::Var(0)),
            PredIR::Leq(ValIR::Var(0), ValIR::Lit(12)),
        );
        let mut r: Vec<Vec<u64>> = (1..=12u64).map(|h| vec![h]).collect();
        r.push(vec![13]);
        let cert = hourclock_cert(safety, r);
        assert_eq!(
            reflect_check_safety_cert(&cert),
            ReflectCheckVerdict::NotSafe { state: vec![13] }
        );
    }

    /// FRAGMENT BOUNDARY: an out-of-fragment stored `safety_pred` (a bounded set QUANTIFIER — the
    /// bitmask `Set*` predicates are now covered, so the quantifier folds are the honest residual) →
    /// Inconclusive (fail-closed), never Accepted. (`pred_exact` also declines it first — either way,
    /// not a false accept.)
    #[test]
    fn reflect_check_inconclusive_out_of_fragment() {
        use crate::explicit_fixpoint_cert::SetIR;
        let set_ir = PredIR::SetForall {
            source: SetIR::Lit(1),
            universe: 1,
            bound_col: 1,
            body: Box::new(PredIR::BoolLit(true)),
        };
        let r: Vec<Vec<u64>> = vec![vec![1]];
        let cert = hourclock_cert(set_ir, r);
        assert!(matches!(
            reflect_check_safety_cert(&cert),
            ReflectCheckVerdict::Inconclusive(_)
        ));
    }

    // ── reflect-v2 exactness cross-check (proof-roadmap §2 B2, task #17) ────────────────────────
    // (reuses the module's `and` helper defined above.)

    /// POSITIVE: a recognizer IR that MATCHES the reflect-v2 deep evaluator on every reachable state
    /// ⇒ `Corroborated { covered = total }`. HourClock-shaped `1 ≤ hr ∧ hr ≤ 12` over R = {1..12} —
    /// the independent deep evaluator reduces it to Bool.true at every reachable `hr`, corroborating
    /// the shallow safety leg's acceptance with kernel-checked op-realization data.
    #[test]
    fn reflect_exactness_corroborates_matching_recognizer() {
        let safety = and(
            PredIR::Leq(ValIR::Lit(1), ValIR::Var(0)),
            PredIR::Leq(ValIR::Var(0), ValIR::Lit(12)),
        );
        let r: Vec<Vec<u64>> = (1..=12u64).map(|h| vec![h]).collect();
        assert_eq!(
            reflect_exactness_over_reachable(&safety, &r),
            ReflectExactnessOutcome::Corroborated {
                covered: 12,
                total: 12
            }
        );
    }

    /// DECISIVE NEGATIVE: a DELIBERATELY-INEXACT predicate — `hr < 12` over a reachable set that
    /// INCLUDES `hr = 12` — models a recognizer/embedder whose accepted invariant is actually FALSE
    /// at a reachable state (the shallow safety leg's mint invariant being `⟦Safety⟧(s)=true ∀s∈R`).
    /// The INDEPENDENT reflect-v2 deep evaluator reduces `12 < 12` to Bool.false and the cross-check
    /// FAILS CLOSED with `Mismatch { state = [12] }` — never a silent corroboration. This is the
    /// exactness bug the wiring declines on.
    #[test]
    fn reflect_exactness_mismatch_catches_inexact_predicate() {
        let bad = PredIR::Lt(ValIR::Var(0), ValIR::Lit(12));
        let r: Vec<Vec<u64>> = (1..=12u64).map(|h| vec![h]).collect();
        match reflect_exactness_over_reachable(&bad, &r) {
            ReflectExactnessOutcome::Mismatch { state, .. } => assert_eq!(state, vec![12]),
            other => {
                panic!("reflect v2 must catch the inexact predicate as Mismatch, got {other:?}")
            }
        }
        // Restricting R to states that DO satisfy `hr < 12` flips it back to Corroborated — the
        // outcome tracks the actual reachable states, not the predicate alone.
        let r_ok: Vec<Vec<u64>> = (1..=11u64).map(|h| vec![h]).collect();
        assert_eq!(
            reflect_exactness_over_reachable(&bad, &r_ok),
            ReflectExactnessOutcome::Corroborated {
                covered: 11,
                total: 11
            }
        );
    }

    /// COVERAGE HONESTY: an IR outside reflect v2's covered fragment (a bounded set QUANTIFIER — the
    /// honest residual now that the bitmask `Set*` predicates + `SetCard`/`CountFold` are covered) on
    /// every reachable state ⇒ `Uncovered` (the exactness guarantee stays syntactic), NOT a false
    /// corroboration and NOT a decline. Pins that the cross-check is additive where covered, silent where not.
    #[test]
    fn reflect_exactness_uncovered_out_of_scalar_fragment() {
        use crate::explicit_fixpoint_cert::SetIR;
        let set_ir = PredIR::SetForall {
            source: SetIR::Lit(1),
            universe: 1,
            bound_col: 1,
            body: Box::new(PredIR::BoolLit(true)),
        };
        let r: Vec<Vec<u64>> = vec![vec![0], vec![1]];
        assert_eq!(
            reflect_exactness_over_reachable(&set_ir, &r),
            ReflectExactnessOutcome::Uncovered
        );
    }

    /// PARTIAL COVERAGE: an out-of-BOUNDS column on a subset of states is skipped (uncovered) while
    /// the in-fragment states are corroborated — `covered < total`, honestly reported, never a
    /// decline for the merely-uncheckable states.
    #[test]
    fn reflect_exactness_partial_coverage_counts_honestly() {
        // `hr ≤ 12` over 1-column states; a stray 2-column state is still in-fragment (col 0), so
        // use an OOB-column IR to force partial coverage instead.
        let safety = PredIR::Leq(ValIR::Var(0), ValIR::Lit(12));
        // Mixed arity: the 0-column tuple makes `Var(0)` out of range ⇒ that state is uncovered.
        let r: Vec<Vec<u64>> = vec![vec![5], vec![], vec![9]];
        match reflect_exactness_over_reachable(&safety, &r) {
            ReflectExactnessOutcome::Corroborated { covered, total } => {
                assert_eq!(
                    (covered, total),
                    (2, 3),
                    "2 in-bounds states covered, the empty tuple skipped"
                );
            }
            other => panic!("expected partial Corroborated, got {other:?}"),
        }
    }

    // ── SET-fragment exactness cross-check (the SetMask/FuncSetMask/Cardinality coverage lift) ───────

    /// POSITIVE (Set fragment): a bitmask SET-predicate `safety_ir` that MATCHES the reflect-v2 deep
    /// evaluator on every reachable state ⇒ `Corroborated { covered = total }`. Models a `SetMask`
    /// invariant `2 ∈ S ∧ S ⊆ {0,1,2}` over a Set column whose reachable masks all satisfy it. These
    /// nodes WERE `Uncovered` (out of the scalar fragment) and are now semantically checked.
    #[test]
    fn reflect_exactness_corroborates_matching_set_recognizer() {
        use crate::explicit_fixpoint_cert::SetIR;
        // 2 ∈ S ∧ S ⊆ {0,1,2}(=mask 7). Reachable masks: {2}=4, {0,2}=5, {1,2}=6, {0,1,2}=7 — all hold.
        let safety = and(
            PredIR::SetMem(2, SetIR::Var(0)),
            PredIR::SetSubseteq(SetIR::Var(0), SetIR::Lit(7)),
        );
        let r: Vec<Vec<u64>> = vec![vec![4], vec![5], vec![6], vec![7]];
        assert_eq!(
            reflect_exactness_over_reachable(&safety, &r),
            ReflectExactnessOutcome::Corroborated {
                covered: 4,
                total: 4
            }
        );
        // A `Cardinality(S) ≤ 3` invariant over the same masks (each ≤ 3 bits) also corroborates n/n.
        let card = PredIR::Leq(
            ValIR::SetCard {
                set: SetIR::Var(0),
                universe: 4,
            },
            ValIR::Lit(3),
        );
        assert_eq!(
            reflect_exactness_over_reachable(&card, &r),
            ReflectExactnessOutcome::Corroborated {
                covered: 4,
                total: 4
            }
        );
    }

    /// DECISIVE NEGATIVE (Set fragment): a DELIBERATELY-INEXACT set predicate — `1 ∈ S` over a reachable
    /// set that INCLUDES a mask WITHOUT bit 1 (`{0,2}=5`) — models a recognizer/embedder whose accepted
    /// invariant is actually FALSE at a reachable state. The INDEPENDENT reflect-v2 deep evaluator reduces
    /// the bit test to `Bool.false` and the cross-check FAILS CLOSED with `Mismatch` — proving the new set
    /// quote arms can CATCH inexactness, never rubber-stamp it. (Also pins `Cardinality`: `|S| = 1` over a
    /// 2-bit mask is caught.)
    #[test]
    fn reflect_exactness_mismatch_catches_inexact_set_predicate() {
        use crate::explicit_fixpoint_cert::SetIR;
        let bad = PredIR::SetMem(1, SetIR::Var(0)); // 1 ∈ S — FALSE at mask 5 = {0,2}
        let r: Vec<Vec<u64>> = vec![vec![6], vec![5]]; // {1,2} then {0,2}
        match reflect_exactness_over_reachable(&bad, &r) {
            ReflectExactnessOutcome::Mismatch { state, .. } => assert_eq!(state, vec![5]),
            other => panic!("an inexact set predicate must be caught as Mismatch, got {other:?}"),
        }
        // Restricting R to masks that DO contain bit 1 flips it back to Corroborated (tracks the states).
        let r_ok: Vec<Vec<u64>> = vec![vec![6], vec![2], vec![3]];
        assert_eq!(
            reflect_exactness_over_reachable(&bad, &r_ok),
            ReflectExactnessOutcome::Corroborated {
                covered: 3,
                total: 3
            }
        );
        // A wrong-count Cardinality claim is caught too: `|S| = 1` is FALSE at the 2-bit mask 5.
        let bad_card = PredIR::Eq(
            ValIR::SetCard {
                set: SetIR::Var(0),
                universe: 4,
            },
            ValIR::Lit(1),
        );
        match reflect_exactness_over_reachable(&bad_card, &vec![vec![1], vec![5]]) {
            ReflectExactnessOutcome::Mismatch { state, .. } => assert_eq!(state, vec![5]),
            other => panic!("a wrong Cardinality must be caught as Mismatch, got {other:?}"),
        }
    }
}

// ── R2 milestone: the reflected ALL-LEGS discharge (safety + both completeness legs) ────────────
#[cfg(all(test, feature = "clean-cic"))]
mod full_tests {
    use super::*;
    use crate::explicit_fixpoint_cert::{certify_explicit_state_spec, ColSort, PredIR, SetIR};

    const HC_SRC: &str = "\
---------------------- MODULE HourClock ----------------------
EXTENDS Naturals
VARIABLE hr
HCini  ==  hr \\in (1 .. 12)
HCnxt  ==  hr' = IF hr # 12 THEN hr + 1 ELSE 1
HC  ==  HCini /\\ [][HCnxt]_hr
--------------------------------------------------------------
THEOREM  HC => []HCini
==============================================================
";

    fn hc_config() -> crate::Config {
        crate::Config {
            init: Some("HCini".into()),
            next: Some("HCnxt".into()),
            invariants: vec!["HCini".into()],
            ..Default::default()
        }
    }

    /// Build a REAL HourClock cert from the certifier (so `init_pred`/`next_pred`/`reachable` are
    /// genuine and the spec-binding re-derivation reproduces them exactly).
    fn hourclock_full_cert() -> SafetyCertificate {
        let fp = certify_explicit_state_spec(HC_SRC, &hc_config())
            .expect("HourClock must explicit-state certify with the general legs");
        assert!(
            fp.init_pred.is_some() && fp.next_pred.is_some(),
            "general legs present"
        );
        let mut cert = crate::cert::build_explicit_fixpoint_certificate(HC_SRC, &hc_config(), fp);
        cert.digest = cert.compute_digest();
        cert
    }

    /// POSITIVE: HourClock reflected-certifies ALL THREE legs; the domain-coverage basis is
    /// (honestly) RustDerived (the single Int axis) — the residual is surfaced, not hidden.
    #[test]
    fn full_certifies_hourclock_all_three_legs() {
        match reflect_check_safety_cert_full(&hourclock_full_cert(), false) {
            ReflectFullVerdict::Certified {
                states, coverage, ..
            } => {
                assert_eq!(states, 12);
                assert!(
                    matches!(coverage, ReflectCoverageBasis::RustDerived { .. }),
                    "HourClock's Int axis is a trusted-Rust bound (not construction-complete)"
                );
            }
            other => panic!("expected Certified, got {other:?}"),
        }
    }

    /// ATTACK C: `--require-domain-complete` DECLINES HourClock — its completeness domain rests on a
    /// trusted-Rust bound rule (Int axis), so closure is only RELATIVE to a Rust-bounded D.
    #[test]
    fn full_require_domain_complete_declines_rustderived_hourclock() {
        assert!(matches!(
            reflect_check_safety_cert_full(&hourclock_full_cert(), true),
            ReflectFullVerdict::Inconclusive(_)
        ));
    }

    /// ATTACK B: a NON-CLOSED R (drop the reachable state [1]) must DECLINE — never a false safe. For
    /// HourClock [1] is BOTH an Init state and a successor of [12], so the Init leg (run first) fires;
    /// either way the verdict is a definitive decline, and the isolated Next-completeness reduction is
    /// pinned in `reflect::reflect_next_completeness_hourclock_closed_and_nonclosed`.
    #[test]
    fn full_non_closed_r_declines() {
        let mut cert = hourclock_full_cert();
        cert.explicit_fixpoint
            .as_mut()
            .unwrap()
            .reachable
            .retain(|t| t != &vec![1]);
        let v = reflect_check_safety_cert_full(&cert, false);
        assert!(
            matches!(
                v,
                ReflectFullVerdict::NotInitComplete { .. } | ReflectFullVerdict::NotClosed { .. }
            ),
            "a non-closed R must decline (never Certified), got {v:?}"
        );
    }

    /// ATTACK D: an out-of-fragment Next IR (a bounded set QUANTIFIER — the honest residual now that the
    /// bitmask `Set*` predicates are covered) makes the reflected Next-completeness leg fail-closed
    /// INCONCLUSIVE — never a certify on an un-reduced obligation.
    #[test]
    fn full_out_of_fragment_next_declines() {
        let mut cert = hourclock_full_cert();
        cert.explicit_fixpoint
            .as_mut()
            .unwrap()
            .next_pred
            .as_mut()
            .unwrap()
            .pred = PredIR::SetForall {
            source: SetIR::Lit(1),
            universe: 1,
            bound_col: 1,
            body: Box::new(PredIR::BoolLit(true)),
        };
        assert!(matches!(
            reflect_check_safety_cert_full(&cert, false),
            ReflectFullVerdict::Inconclusive(_)
        ));
    }

    // ── RECOGNITION-ONLY spec-bind regressions (the --full trust base = kernel + recognizer + quoter) ──
    // A small FuncEnum spec (the TCommit shape) with a `rmState \in [RM -> {labels}]` TYPE invariant, so
    // the reflect-check `--full` spec-bind derives the column sort STRUCTURALLY from the type (no
    // enumeration, no embedder) and re-recognizes Init/Next/Safety.

    const TC2_SRC: &str = "\
---------------------------- MODULE TC2 ----------------------------
VARIABLE rmState
TCTypeOK == rmState \\in [RM -> {\"aborted\", \"committed\", \"working\"}]
Init == rmState = [rm \\in RM |-> \"working\"]
Commit(rm) == rmState[rm] = \"working\" /\\ rmState' = [rmState EXCEPT ![rm] = \"committed\"]
Abort(rm)  == rmState[rm] = \"working\" /\\ rmState' = [rmState EXCEPT ![rm] = \"aborted\"]
Next == \\E rm \\in RM : Commit(rm) \\/ Abort(rm)
===================================================================
";

    fn tc2_config() -> crate::Config {
        use crate::config::ConstantValue;
        let mut config = crate::Config {
            init: Some("Init".into()),
            next: Some("Next".into()),
            invariants: vec!["TCTypeOK".into()],
            ..Default::default()
        };
        config.constants.insert(
            "RM".to_string(),
            ConstantValue::ModelValueSet(vec!["r1".into(), "r2".into()]),
        );
        config
    }

    /// A genuine FuncEnum cert from the certifier (so `sorts`/`init_pred`/`next_pred`/`safety_pred` are
    /// real and the recognition-only bind re-derives them exactly).
    fn tc2_full_cert() -> SafetyCertificate {
        let fp = certify_explicit_state_spec(TC2_SRC, &tc2_config())
            .expect("TC2 FuncEnum spec must explicit-state certify with the general legs");
        assert!(
            fp.init_pred.is_some() && fp.next_pred.is_some(),
            "general legs present"
        );
        assert!(
            matches!(fp.sorts.as_slice(), [ColSort::FuncEnum { arity: 2, .. }]),
            "rmState is a FuncEnum column, got {:?}",
            fp.sorts
        );
        let mut cert = crate::cert::build_explicit_fixpoint_certificate(TC2_SRC, &tc2_config(), fp);
        cert.digest = cert.compute_digest();
        cert
    }

    /// POSITIVE: the recognition-only bind CERTIFIES the FuncEnum spec — the column sort is DERIVED from
    /// the `rmState \in [RM -> {labels}]` type invariant (not enumerated), and every re-recognized IR
    /// matches. The FuncEnum axis is its full label universe ⇒ ConstructionComplete (no Rust bound rule).
    #[test]
    fn full_recognition_bind_certifies_funcenum_construction_complete() {
        match reflect_check_safety_cert_full(&tc2_full_cert(), false) {
            ReflectFullVerdict::Certified { coverage, .. } => {
                assert_eq!(coverage, ReflectCoverageBasis::ConstructionComplete);
            }
            other => panic!("expected Certified (construction-complete), got {other:?}"),
        }
    }

    /// ATTACK (wrong stored IR): weaken the stored invariant to the trivially-true predicate. It still
    /// passes the reflected R⊆Safety leg over R, but the recognizer re-derives the REAL FuncEnum type
    /// invariant from the spec ⇒ the recognition-only bind REJECTS (re-recognized ≠ stored).
    #[test]
    fn full_recognition_bind_rejects_tampered_safety_ir() {
        let mut cert = tc2_full_cert();
        cert.explicit_fixpoint.as_mut().unwrap().safety_pred = Some(PredIR::BoolLit(true));
        assert!(
            matches!(
                reflect_check_safety_cert_full(&cert, false),
                ReflectFullVerdict::Rejected(_)
            ),
            "a widened (trivially-true) stored invariant must REJECT via re-recognition"
        );
    }

    /// ATTACK (wrong stored sort): relabel a FuncEnum label (same arity/base/dom, so the reflected legs
    /// still reduce over R), but the sort DERIVED from the spec's `[RM -> {labels}]` type differs ⇒ REJECT.
    #[test]
    fn full_recognition_bind_rejects_tampered_sort() {
        let mut cert = tc2_full_cert();
        if let Some(ColSort::FuncEnum { labels, .. }) =
            cert.explicit_fixpoint.as_mut().unwrap().sorts.get_mut(0)
        {
            for l in labels.iter_mut() {
                if l == "working" {
                    *l = "zzz_working".to_string(); // still sorts LAST ⇒ index unchanged ⇒ legs still reduce
                }
            }
        }
        assert!(
            matches!(
                reflect_check_safety_cert_full(&cert, false),
                ReflectFullVerdict::Rejected(_)
            ),
            "a relabelled FuncEnum sort must REJECT (spec-derived labels differ from the stored sort)"
        );
    }

    /// ATTACK (underivable sort): a FuncEnum spec whose invariant is a bounded-∀ (NOT a `rmState \in
    /// [RM -> S]` TYPE membership), so no column sort is spec-derivable ⇒ the recognition-only bind
    /// declines INCONCLUSIVE (fail-closed — it NEVER falls back to trusting the stored sort).
    #[test]
    fn full_recognition_bind_inconclusive_on_underivable_sort() {
        const TC_FORALL_SRC: &str = "\
---------------------------- MODULE TCf ----------------------------
VARIABLE rmState
TCInv == \\A rm \\in RM : rmState[rm] \\in {\"aborted\", \"committed\", \"working\"}
Init == rmState = [rm \\in RM |-> \"working\"]
Commit(rm) == rmState[rm] = \"working\" /\\ rmState' = [rmState EXCEPT ![rm] = \"committed\"]
Abort(rm)  == rmState[rm] = \"working\" /\\ rmState' = [rmState EXCEPT ![rm] = \"aborted\"]
Next == \\E rm \\in RM : Commit(rm) \\/ Abort(rm)
===================================================================
";
        use crate::config::ConstantValue;
        let mut config = crate::Config {
            init: Some("Init".into()),
            next: Some("Next".into()),
            invariants: vec!["TCInv".into()],
            ..Default::default()
        };
        config.constants.insert(
            "RM".to_string(),
            ConstantValue::ModelValueSet(vec!["r1".into(), "r2".into()]),
        );
        let fp = certify_explicit_state_spec(TC_FORALL_SRC, &config)
            .expect("the bounded-∀ FuncEnum spec still certifies with the general legs");
        assert!(fp.safety_pred.is_some() && fp.init_pred.is_some() && fp.next_pred.is_some());
        let mut cert = crate::cert::build_explicit_fixpoint_certificate(TC_FORALL_SRC, &config, fp);
        cert.digest = cert.compute_digest();
        assert!(
            matches!(
                reflect_check_safety_cert_full(&cert, false),
                ReflectFullVerdict::Inconclusive(_)
            ),
            "a spec with no type-membership invariant is not sort-derivable ⇒ INCONCLUSIVE (never trust \
             the stored sort)"
        );
    }
}
