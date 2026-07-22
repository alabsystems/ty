// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! `ty reflect-check` subcommand: re-discharge an explicit-state fixpoint certificate's
//! `R ⊆ Safety` obligation through the KERNEL-DEFINED reflected evaluator
//! (`tla_check::reflect`'s `TyReflectEvalP`) instead of the shallow Rust embedder
//! (`cleancic::embed_pred_ir`).
//!
//! This is an ADDITIVE second discharge of the safety obligation: it quotes the recognized
//! invariant with the line-auditable 1:1 quoter and lets the clean kernel reduce the deep
//! evaluator over every reachable state, then binds the discharged obligation to the cert's own
//! spec by re-derivation. Exits 0 on REFLECTED-SAFE, 1 on NOT-SAFE / REJECTED, 2 on INCONCLUSIVE.

use std::path::Path;

use anyhow::{Context, Result};
use tla_check::cert::SafetyCertificate;

/// Run `ty reflect-check`.
pub(crate) fn cmd_reflectcheck(
    cert_file: &Path,
    full: bool,
    require_domain_complete: bool,
    ast_direct: bool,
) -> Result<()> {
    let json = std::fs::read_to_string(cert_file)
        .with_context(|| format!("read certificate {}", cert_file.display()))?;
    let cert = SafetyCertificate::from_json(&json).map_err(anyhow::Error::msg)?;

    if ast_direct {
        return cmd_reflectcheck_ast_direct(&cert);
    }
    if full {
        return cmd_reflectcheck_full(&cert, require_domain_complete);
    }

    #[cfg(feature = "clean-cic")]
    {
        use tla_check::reflect_safety_check::{reflect_check_safety_cert, ReflectCheckVerdict};
        match reflect_check_safety_cert(&cert) {
            ReflectCheckVerdict::Accepted { states } => {
                println!(
                    "REFLECTED-SAFE: the KERNEL-DEFINED reflected evaluator (TyReflectEvalP) reduced \
                     Safety(s) => Bool.true for all {states} reachable state(s), and the discharged \
                     (Safety_ir, R, sorts) re-derive identically from the cert's own spec.\n\
                     trust delta: this R⊆Safety obligation does NOT rely on the shallow Rust embedder \
                     (cleancic::embed_pred_ir); it rests on the 1:1 quoter + the clean kernel + the \
                     recognizer producing Safety_ir (bound to the spec by re-derivation)."
                );
                Ok(())
            }
            ReflectCheckVerdict::NotSafe { state } => {
                eprintln!(
                    "NOT-SAFE: the reflected evaluator reduced Safety to Bool.false at reachable \
                     state {state:?} — this reachable state VIOLATES the invariant (fail-closed, \
                     never a false accept)."
                );
                std::process::exit(1);
            }
            ReflectCheckVerdict::Rejected(why) => {
                eprintln!("REJECTED: {why}");
                std::process::exit(1);
            }
            ReflectCheckVerdict::Inconclusive(why) => {
                eprintln!("INCONCLUSIVE: {why} (NOT a verdict on the certificate)");
                std::process::exit(2);
            }
        }
    }
    #[cfg(not(feature = "clean-cic"))]
    {
        let _ = cert;
        eprintln!(
            "INCONCLUSIVE: `ty reflect-check` requires a `clean-cic` build (the reflected kernel \
             evaluator is not linked). (This is NOT a verdict on the certificate.)"
        );
        std::process::exit(2);
    }
}

/// Run `ty reflect-check --ast-direct`: the RECOGNIZER-FREE all-legs discharge (design pivot
/// increment 1, docs/cert/design-pivot-reflect.md) — Init/Next/Safety quoted DIRECTLY from the
/// spec's own (re-parsed, operator-inlined) TLA+ AST into the kernel's deep embedding, plus a
/// MANDATORY cross-check against the recognized-IR reflected lane (hard failure on divergence).
fn cmd_reflectcheck_ast_direct(cert: &SafetyCertificate) -> Result<()> {
    #[cfg(feature = "clean-cic")]
    {
        use tla_check::reflect_ast_direct::{
            reflect_check_ast_direct_with_crosscheck, AstCrossCheck, AstDirectVerdict,
        };
        let (verdict, cc) = reflect_check_ast_direct_with_crosscheck(cert);
        // A conclusive DIVERGENCE between the lanes overrides EVERYTHING (incl. an apparent
        // certify): it is evidence of a quoter/recognizer/embedding bug — fail closed, loudly.
        if let AstCrossCheck::Divergence { ast, recognized } = &cc {
            eprintln!(
                "CROSS-CHECK-DIVERGENCE (fail-closed — trust NEITHER verdict): the AST-direct \
                 reflected lane and the recognized-IR reflected lane reached CONTRADICTORY \
                 conclusive verdicts on the same certificate:\n  AST-direct : {ast}\n  \
                 recognized : {recognized}\nThis indicates a quoter / recognizer / embedding \
                 bug. The certificate is NOT certified by this run."
            );
            std::process::exit(1);
        }
        match verdict {
            AstDirectVerdict::Certified {
                states,
                init_domain,
                next_domain,
                next_pairs,
            } => {
                let inv_names = if cert.invariants.is_empty() {
                    "(configured invariants)".to_string()
                } else {
                    cert.invariants.join(" ∧ ")
                };
                println!(
                    "REFLECTED-CERTIFIED (AST-DIRECT, recognizer-free): safety of the spec — the \
                     configured invariant(s) [{inv_names}] hold on every reachable state \
                     (R⊆Safety), Init⊆R, and R closed under Next — each predicate quoted \
                     DIRECTLY from the spec's own TLA+ AST (re-parsed from the cert's embedded \
                     spec_src, operator-inlined by cert_inline) into the kernel-admitted deep \
                     embedding, and reduced by the kernel-defined evaluator (TyReflectEvalP \
                     composed with TyReflectMem). The RECOGNIZER (cleancic recognition arms) and \
                     the shallow embedder are OUT of every obligation and out of the spec-bind.\n\
                     legs: R⊆Safety over {states} reachable state(s); Init⊆R over \
                     D_init={init_domain} domain state(s); R closed under Next over {next_pairs} \
                     (s∈R, sp∈D_next={next_domain}) pair(s).\n\
                     trust base = kernel + AST-quoter + parser/inliner + AST domain-bound rule \
                     (+ enumerator-provided R, kernel-verified: the completeness legs check \
                     Init⊆R and closure, and ANY sound inductive invariant proves safety — so \
                     the enumerator is NOT trusted).\n\
                     honesty: \"kernel + AST-quoter\" alone would OVERCLAIM — the quoter consumes \
                     the AST produced by the tla-core parser and the cert_inline operator \
                     inliner, so BOTH are in this verdict's trust base (and are common-mode with \
                     the recognized-IR lane, so the cross-check does not discharge them); the \
                     completeness domain D is derived by an AST-level Int bound rule (trusted \
                     Rust — the same residual class as the recognized lane's RustDerived axes, \
                     surfaced not hidden; its coverage theorem `D ⊇ Succ(R)` is EXTERNALLY \
                     PROVEN sorry-free in Lean (Aristotle proof \
                     L1_ast_direct_domain_covers.lean, internal cert archive) — so the rule is \
                     corroborated, though it \
                     remains Rust code in TY's own TCB until the kernel re-checks the bound)."
                );
                // Honest disclosure (increment 3): a spec with an ENUM / model-value column adds
                // the cert's stored label→code map (`sorts[i].labels`) to this verdict's trust
                // base — the quoter resolves a label atom to the code the enumerator packed it as.
                // Common-mode with the recognized lane + the enumerated R (same map), and
                // fail-closed on a non-injective map, but surfaced here, never hidden.
                if let Some(fp) = &cert.explicit_fixpoint {
                    let enum_cols: Vec<String> = fp
                        .sorts
                        .iter()
                        .enumerate()
                        .filter_map(|(i, s)| {
                            matches!(s, tla_check::explicit_fixpoint_cert::ColSort::Enum { .. })
                                .then(|| i.to_string())
                        })
                        .collect();
                    if !enum_cols.is_empty() {
                        println!(
                            "enum trust: column(s) [{}] are enum/model-value sorts, so the cert's \
                             stored label→code map (`sorts[i].labels`) is ALSO in this verdict's \
                             trust base — the quoter resolves each label atom to its code. It is a \
                             BIJECTION (distinct, non-empty labels — fail-closed otherwise) and \
                             common-mode with the recognized lane and the enumerated R (same map).",
                            enum_cols.join(", ")
                        );
                    }
                }
                match cc {
                    AstCrossCheck::Agree { recognized } => println!(
                        "cross-check: the recognized-IR reflected lane (trust base kernel + \
                         recognizer + quoter) independently reached the SAME verdict class: \
                         {recognized}"
                    ),
                    AstCrossCheck::Unavailable { reason } => println!(
                        "cross-check: UNAVAILABLE — the recognized-IR lane declined \
                         ({reason}); the AST-direct verdict stands on its own stated trust base.\n\
                         NOTE: the independent mis-translation cross-check (which would fail LOUD on a \
                         quoter/embedding bug by disagreeing with the recognized lane) is INACTIVE for \
                         this certificate — there is no second lane to corroborate the quote. This \
                         verdict therefore rests SOLELY on the AST-quoter audit + the Lean domain-\
                         coverage proof (L1_ast_direct_domain_covers), with no automated second opinion."
                    ),
                    AstCrossCheck::Divergence { .. } => unreachable!("handled above"),
                }
                Ok(())
            }
            AstDirectVerdict::NotSafe { state } => {
                eprintln!(
                    "NOT-SAFE (AST-direct): the kernel reduced the spec's own AST-quoted \
                     invariant to Bool.false at reachable state {state:?} — this reachable \
                     state VIOLATES the invariant (fail-closed, never a false accept)."
                );
                std::process::exit(1);
            }
            AstDirectVerdict::NotClosed { s, sp } => {
                eprintln!(
                    "NOT-CLOSED (AST-direct): the kernel reduced Next(s,sp) => sp∈R to \
                     Bool.false at s={s:?}, sp={sp:?} — Next(s,sp) holds but sp ∉ R, so R is \
                     NOT closed under the spec's own AST-quoted Next (a missing/dropped \
                     successor). Fail-closed; never a false safe."
                );
                std::process::exit(1);
            }
            AstDirectVerdict::NotInitComplete { s } => {
                eprintln!(
                    "NOT-INIT-COMPLETE (AST-direct): the kernel reduced Init(s) => s∈R to \
                     Bool.false at s={s:?} — Init(s) holds but s ∉ R (a missing Init state)."
                );
                std::process::exit(1);
            }
            AstDirectVerdict::Inconclusive(why) => {
                eprintln!(
                    "INCONCLUSIVE (AST-direct): {why} (NOT a verdict on the certificate). The \
                     AST-direct v1 fragment is deliberately narrow and fail-closed; the \
                     recognized-IR reflected lane (`ty reflect-check --full`, trust base kernel \
                     + recognizer + quoter) is the fallback tier for this certificate."
                );
                if let AstCrossCheck::Unavailable { reason } = cc {
                    eprintln!("(recognized-IR lane, for reference: {reason})");
                }
                std::process::exit(2);
            }
        }
    }
    #[cfg(not(feature = "clean-cic"))]
    {
        let _ = cert;
        eprintln!(
            "INCONCLUSIVE: `ty reflect-check --ast-direct` requires a `clean-cic` build (the \
             reflected kernel evaluator is not linked). (This is NOT a verdict on the \
             certificate.)"
        );
        std::process::exit(2);
    }
}

/// Run `ty reflect-check --full`: re-discharge ALL THREE safety obligations (R⊆Safety,
/// Init-completeness, Next-completeness/closure) through the reflected evaluator.
fn cmd_reflectcheck_full(cert: &SafetyCertificate, require_domain_complete: bool) -> Result<()> {
    #[cfg(feature = "clean-cic")]
    {
        use tla_check::reflect_safety_check::{
            reflect_check_safety_cert_full, ReflectCoverageBasis, ReflectFullVerdict,
        };
        match reflect_check_safety_cert_full(cert, require_domain_complete) {
            ReflectFullVerdict::Certified {
                states,
                init_domain,
                next_domain,
                next_pairs,
                coverage,
            } => {
                let inv_names = if cert.invariants.is_empty() {
                    "(configured invariants)".to_string()
                } else {
                    cert.invariants.join(" ∧ ")
                };
                println!(
                    "REFLECTED-CERTIFIED (embedder-free): safety of the spec — the configured invariant(s) \
                     [{inv_names}] hold on every reachable state (R⊆Safety), Init⊆R, and \
                     R closed under Next, each reduced by the kernel-defined evaluator (TyReflectEvalP \
                     composed with TyReflectMem); the shallow embedder (cleancic::embed_pred_ir) is \
                     OUT of the obligation.\n\
                     legs: R⊆Safety over {states} reachable state(s); Init⊆R over D_init={init_domain} \
                     domain state(s); R closed under Next over {next_pairs} (s∈R, sp∈D_next={next_domain}) \
                     pair(s) — the enumerator PROVIDES R but the reflected completeness legs kernel-VERIFY \
                     it (any sound inductive R proves safety), so R needn't be trusted.\n\
                     spec-bind: the discharged (sorts, Init, Next, Safety) were RE-DERIVED from the spec by \
                     RECOGNITION ONLY (structural sort derivation + the 1:1 recognizer) — NO re-enumeration, \
                     NO embedder.\n\
                     trust base = kernel + recognizer + quoter."
                );
                match coverage {
                    ReflectCoverageBasis::ConstructionComplete => {
                        println!(
                            "domain coverage: every completeness-leg axis is its column's full SORT \
                             universe — no per-column bound rule is trusted (D ⊇ Succ(R)/Init by \
                             construction)."
                        );
                    }
                    ReflectCoverageBasis::RustDerived {
                        next_rust,
                        init_rust,
                    } => {
                        println!(
                            "domain-coverage RESIDUAL (surfaced, NOT hidden): the completeness domain \
                             D=⨉_i{{0..=H_i}} covers Succ(R)/Init by a TRUSTED-RUST structural bound \
                             rule (cleancic::{{next,init}}_domain_bounds_from_ir) on Next-leg \
                             column(s) {next_rust:?} and Init-leg column(s) {init_rust:?} — this is \
                             EMBEDDER-FREE (no embed_pred_ir, no embed_pred_ir_sym) but NOT kernel-only: \
                             D's coverage rests on TY's bound rule, not the kernel. Pass \
                             --require-domain-complete to DECLINE such a domain."
                        );
                    }
                }
                Ok(())
            }
            ReflectFullVerdict::NotSafe { state } => {
                eprintln!(
                    "NOT-SAFE: the reflected evaluator reduced Safety to Bool.false at reachable \
                     state {state:?} — this reachable state VIOLATES the invariant (fail-closed)."
                );
                std::process::exit(1);
            }
            ReflectFullVerdict::NotClosed { s, sp } => {
                eprintln!(
                    "NOT-CLOSED: the reflected Next-completeness leg reduced Next(s,sp) => sp∈R to \
                     Bool.false at s={s:?}, sp={sp:?} — Next(s,sp) holds but sp ∉ R, so R is NOT \
                     closed under Next (a missing/dropped successor). Fail-closed; never a false safe."
                );
                std::process::exit(1);
            }
            ReflectFullVerdict::NotInitComplete { s } => {
                eprintln!(
                    "NOT-INIT-COMPLETE: the reflected Init-completeness leg reduced Init(s) => s∈R \
                     to Bool.false at s={s:?} — Init(s) holds but s ∉ R (a missing Init state)."
                );
                std::process::exit(1);
            }
            ReflectFullVerdict::Rejected(why) => {
                eprintln!("REJECTED: {why}");
                std::process::exit(1);
            }
            ReflectFullVerdict::Inconclusive(why) => {
                eprintln!("INCONCLUSIVE: {why} (NOT a verdict on the certificate)");
                std::process::exit(2);
            }
        }
    }
    #[cfg(not(feature = "clean-cic"))]
    {
        let _ = (cert, require_domain_complete);
        eprintln!(
            "INCONCLUSIVE: `ty reflect-check --full` requires a `clean-cic` build (the reflected \
             kernel evaluator is not linked). (This is NOT a verdict on the certificate.)"
        );
        std::process::exit(2);
    }
}
