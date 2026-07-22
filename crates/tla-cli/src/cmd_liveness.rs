// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! `ty certify-liveness` / `ty live-check`: produce and re-check a re-checkable
//! liveness certificate (`ty.live-cert/v1`) — a well-founded-descent proof of
//! `<>P` under `WF(Next)`.

use std::path::Path;

use anyhow::{bail, Context, Result};
use tla_check::Config;

use crate::helpers::read_source;

/// Resolve the config path (explicit `--config`, else `<spec>.cfg`).
fn resolve_config(file: &Path, config_path: Option<&Path>) -> Result<std::path::PathBuf> {
    match config_path {
        Some(p) => Ok(p.to_path_buf()),
        None => {
            let mut cfg = file.to_path_buf();
            cfg.set_extension("cfg");
            if !cfg.exists() {
                bail!(
                    "No config file specified and {} does not exist.",
                    cfg.display()
                );
            }
            Ok(cfg)
        }
    }
}

/// Load a spec for a certification lane: flatten the EXTENDS closure into one
/// self-contained module (exactly as `ty certify` does, so corpus specs reach
/// the lane on equal footing), parse the config, and decompose a
/// SPECIFICATION-form config (the common corpus shape, e.g. CoffeeCan's
/// `SPECIFICATION Spec`) into INIT/NEXT against the flattened module.
fn load_flattened_spec_and_config(
    file: &Path,
    config_path: Option<&Path>,
    lane: &str,
) -> Result<(String, Config)> {
    let _ = read_source(file)?; // existence/readability check with the standard error shape
    let flat = crate::flatten::flatten_extends_closure(file).map_err(|e| {
        anyhow::anyhow!("NOT CERTIFIED ({lane}): cannot flatten to a self-contained module — {e}")
    })?;
    let source = flat.source;

    let config_path = resolve_config(file, config_path)?;
    let config_source = std::fs::read_to_string(&config_path)
        .with_context(|| format!("read config {}", config_path.display()))?;
    let mut config = match Config::parse(&config_source) {
        Ok(c) => c,
        Err(errors) => {
            for err in &errors {
                eprintln!("{}:{}: {}", config_path.display(), err.line(), err);
            }
            bail!("config parse failed with {} error(s)", errors.len());
        }
    };
    if (config.init.is_none() || config.next.is_none()) && config.specification.is_some() {
        let tree = tla_core::parse_to_syntax_tree(&source);
        if let Ok(resolved) = tla_check::resolve_spec_from_config_with_extends(&config, &tree, &[])
        {
            if config.init.is_none() {
                config.init = Some(resolved.init);
            }
            if config.next.is_none() {
                config.next = Some(resolved.next);
            }
        }
    }
    // Fail-closed well-formedness gate (certify/check parity) — shared with `ty certify`. Runs
    // here so BOTH `certify-liveness` and `certify-all-n` (which share these inputs) decline an
    // ill-formed spec/config rather than minting a certificate for a mis-read spec.
    crate::cert_gate::certify_wellformedness_gate(&source, &config, file)?;
    Ok((source, config))
}

/// Run `ty certify-liveness`.
pub(crate) fn cmd_certify_liveness(
    file: &Path,
    config_path: Option<&Path>,
    property: &str,
    measure: &str,
    out: &Path,
) -> Result<()> {
    let (source, config) = load_flattened_spec_and_config(file, config_path, "liveness")?;

    // LANE 0 — the ENUMERATOR-FREE COUNTDOWN KERNEL lane (`clean-cic`): the first `<>P` verdict whose
    // trust base is the Clean CIC kernel ALONE — NO enumerator, NO solver. Restricted to the
    // deterministic-counter fragment (`Init = x = c`, `Next = x' = x-1`, measure `x`, property
    // `<>(x = 0)`): the ground countdown chain `c > c-1 > … > 0` is kernel-checked and, under the
    // deterministic decrement, the single trace is every trace. Tried FIRST; on decline we fall
    // through to the enumerator-ASSISTED explicit lane (below).
    #[cfg(feature = "clean-cic")]
    {
        if let Some(cert) =
            tla_check::live_cert::certify_liveness_free(&source, &config, property, measure)
        {
            std::fs::write(out, cert.to_json())
                .with_context(|| format!("write certificate {}", out.display()))?;
            println!(
                "CERTIFIED (enumerator-free liveness): {}\ncertificate -> {}\nre-check with: ty live-check {}",
                cert.verdict,
                out.display(),
                out.display()
            );
            return Ok(());
        }
    }

    // LANE 1 — the SOLVER-FREE EXPLICIT-STATE KERNEL lane (`clean-cic`): the first full, kernel-backed
    // `<>P` verdict with NO solver. Descent + bounded-below + enabledness(every terminal state
    // satisfies P) are kernel-checked over the enumerated reachable set. Tried after the enumerator-
    // free lane; on decline we fall through to the AY well-founded-descent lane (below).
    #[cfg(feature = "clean-cic")]
    {
        if let Some(cert) =
            tla_check::live_cert::certify_liveness_explicit(&source, &config, property, measure)
        {
            std::fs::write(out, cert.to_json())
                .with_context(|| format!("write certificate {}", out.display()))?;
            println!(
                "CERTIFIED (explicit-state liveness): {}\ncertificate -> {}\nre-check with: ty live-check {}",
                cert.verdict,
                out.display(),
                out.display()
            );
            return Ok(());
        }
    }

    // LANE 2 — the AY well-founded-descent lane (5 SMT obligations + descent kernel leg).
    #[cfg(feature = "ay")]
    {
        if let Some(cert) =
            tla_check::live_cert::certify_liveness_spec(&source, &config, property, measure)
        {
            std::fs::write(out, cert.to_json())
                .with_context(|| format!("write certificate {}", out.display()))?;
            println!(
                "CERTIFIED (liveness): {}\ncertificate -> {}\nre-check with: ty live-check {}",
                cert.verdict,
                out.display(),
                out.display()
            );
            return Ok(());
        }
    }

    // Neither lane certified — honest decline, surfacing any partial (descent) result.
    #[cfg(any(feature = "ay", feature = "clean-cic"))]
    {
        eprintln!(
            "NOT CERTIFIED (liveness): no lane could produce a re-checkable `<>P` certificate \
             (property not `<>P`, undecomposable/non-affine Next, non-Int measure, a reachable \
             TERMINAL state violating P, a non-finite reachable set, a state outside the \
             scalar/Bool encodable fragment — e.g. a record variable — or, in the AY lane, an \
             SMT obligation that is not strict-re-checkable)."
        );
        // HONEST partial: even when the full liveness cert declines, the TERMINATION DESCENT itself
        // may kernel-certify (solver-free). Surface it (what IS proven) — e.g. for CoffeeCan, whose
        // record variable the enabledness embedder does not yet encode.
        #[cfg(feature = "clean-cic")]
        if let Some(sz) =
            tla_check::live_cert::affine_descent_kernel_status_explicit(&source, &config, measure)
        {
            eprintln!(
                "  However, the termination DESCENT is KERNEL-CERTIFIED (Clean CIC, NO solver): \
                 every state-changing action strictly decreases the affine measure `{measure}` by 1 \
                 (a {sz}-byte kernel-checked term). Under WF(Next) + bounded-below that is \
                 termination; the missing part is the enabledness-at-terminals leg (blocked here by \
                 an unrecognizable P or an un-encodable state shape)."
            );
        }
        std::process::exit(2);
    }
    #[cfg(not(any(feature = "ay", feature = "clean-cic")))]
    {
        let _ = (out, &config, property, measure, &source);
        bail!("`ty certify-liveness` requires the `ay` or `clean-cic` feature");
    }
}

/// Per-conjunct certificate output path: `cert.json` -> `cert.c<i>.json`.
#[cfg(feature = "ay")]
fn conjunct_out_path(out: &Path, i: usize) -> std::path::PathBuf {
    let stem = out
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("all-n-cert");
    match out.extension().and_then(|s| s.to_str()) {
        Some(ext) => out.with_file_name(format!("{stem}.c{i}.{ext}")),
        None => out.with_file_name(format!("{stem}.c{i}")),
    }
}

/// Equality-half certificate output path: `cert.json` -> `cert.c<i>.<le|ge>.json`.
#[cfg(feature = "ay")]
fn eq_half_out_path(out: &Path, i: usize, tag: &str) -> std::path::PathBuf {
    let stem = out
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("all-n-cert");
    match out.extension().and_then(|s| s.to_str()) {
        Some(ext) => out.with_file_name(format!("{stem}.c{i}.{tag}.{ext}")),
        None => out.with_file_name(format!("{stem}.c{i}.{tag}")),
    }
}

/// Run `ty certify-all-n`.
pub(crate) fn cmd_certify_all_n(
    file: &Path,
    config_path: Option<&Path>,
    constant: &str,
    invariant_j: Option<&str>,
    out: &Path,
) -> Result<()> {
    let (source, config) = load_flattened_spec_and_config(file, config_path, "all-N")?;

    #[cfg(feature = "ay")]
    {
        use tla_check::cert_all_n::{
            certify_all_n_auto, certify_all_n_with_reason, AllNAutoOutcome, ConjunctCoverage,
        };

        // Explicit --invariant-j: single attempt, reasoned decline.
        if let Some(j) = invariant_j {
            match certify_all_n_with_reason(&source, &config, constant, j) {
                Ok(cert) => {
                    std::fs::write(out, cert.to_json())
                        .with_context(|| format!("write certificate {}", out.display()))?;
                    println!(
                        "CERTIFIED (all-N): {}\ncertificate -> {}\nre-check with: ty all-n-check {}",
                        cert.verdict,
                        out.display(),
                        out.display()
                    );
                    return Ok(());
                }
                Err(reason) => {
                    eprintln!(
                        "NOT CERTIFIED (all-N): `{j}` is not certified for all {constant} — \
                         {reason}"
                    );
                    std::process::exit(2);
                }
            }
        }

        // AUTO-J: default J from the configured INVARIANT(s) — whole conjunction
        // first, per-conjunct coverage as the sound fallback.
        if config.invariants.is_empty() {
            eprintln!(
                "NOT CERTIFIED (all-N): no INVARIANT in the config to default J from; \
                 pass --invariant-j."
            );
            std::process::exit(2);
        }
        println!(
            "auto-J: defaulting J to the configured invariant conjunction `{}`",
            config.invariants.join(" /\\ ")
        );
        match certify_all_n_auto(&source, &config, constant) {
            Ok(AllNAutoOutcome::Whole(cert)) => {
                std::fs::write(out, cert.to_json())
                    .with_context(|| format!("write certificate {}", out.display()))?;
                println!(
                    "CERTIFIED (all-N): {}\ncertificate -> {}\nre-check with: ty all-n-check {}",
                    cert.verdict,
                    out.display(),
                    out.display()
                );
                Ok(())
            }
            Ok(AllNAutoOutcome::PerConjunct {
                whole_decline,
                legs,
            }) => {
                let total = legs.len();
                println!(
                    "auto-J: whole-invariant J declined ({whole_decline}); trying \
                     per-conjunct coverage ({total} conjuncts)"
                );
                let mut certified = 0usize;
                for (i, leg) in legs.iter().enumerate() {
                    match leg {
                        ConjunctCoverage::Cert(cert) => {
                            let path = conjunct_out_path(out, i);
                            std::fs::write(&path, cert.to_json())
                                .with_context(|| format!("write certificate {}", path.display()))?;
                            certified += 1;
                            println!(
                                "  conjunct {}/{total}: CERTIFIED -> {}",
                                i + 1,
                                path.display()
                            );
                        }
                        ConjunctCoverage::EqSplit { le, ge } => {
                            let le_path = eq_half_out_path(out, i, "le");
                            let ge_path = eq_half_out_path(out, i, "ge");
                            std::fs::write(&le_path, le.to_json()).with_context(|| {
                                format!("write certificate {}", le_path.display())
                            })?;
                            std::fs::write(&ge_path, ge.to_json()).with_context(|| {
                                format!("write certificate {}", ge_path.display())
                            })?;
                            certified += 1;
                            println!(
                                "  conjunct {}/{total}: CERTIFIED as an EQUALITY via its <= and \
                                 >= halves (jointly <=> the equality) -> {} + {}",
                                i + 1,
                                le_path.display(),
                                ge_path.display()
                            );
                        }
                        ConjunctCoverage::JointCovered(cert) => {
                            let path = conjunct_out_path(out, i);
                            std::fs::write(&path, cert.to_json())
                                .with_context(|| format!("write certificate {}", path.display()))?;
                            certified += 1;
                            let members = cert
                                .joint_members
                                .as_deref()
                                .unwrap_or_default()
                                .iter()
                                .map(|m| (m + 1).to_string())
                                .collect::<Vec<_>>()
                                .join(",");
                            println!(
                                "  conjunct {}/{total}: CERTIFIED via an inductive JOINT \
                                 strengthening over conjuncts {{{members}}} (the joint is the \
                                 inductive WITNESS; the claim stays this conjunct only) -> {}",
                                i + 1,
                                path.display()
                            );
                        }
                        ConjunctCoverage::Declined(reason) => {
                            println!("  conjunct {}/{total}: declined — {reason}", i + 1);
                        }
                    }
                }
                if certified == total {
                    println!(
                        "CERTIFIED (all-N, per-conjunct): {certified}/{total} conjuncts of the \
                         configured invariant hold for EVERY {constant} — FULL coverage.\n\
                         re-check each with: ty all-n-check <cert>"
                    );
                    Ok(())
                } else {
                    eprintln!(
                        "NOT CERTIFIED (all-N): PARTIAL per-conjunct coverage \
                         ({certified}/{total} conjuncts) — the certificates written above are \
                         individually sound but do NOT cover the full configured invariant."
                    );
                    std::process::exit(2);
                }
            }
            Err(reason) => {
                eprintln!(
                    "NOT CERTIFIED (all-N): auto-J (configured invariant) is not certified \
                     for all {constant} — {reason}"
                );
                std::process::exit(2);
            }
        }
    }
    #[cfg(not(feature = "ay"))]
    {
        let _ = (out, &config, constant, invariant_j, &source);
        bail!("`ty certify-all-n` requires the `ay` feature");
    }
}

/// Run `ty alln-check`.
pub(crate) fn cmd_alln_check(cert_file: &Path) -> Result<()> {
    let json = std::fs::read_to_string(cert_file)
        .with_context(|| format!("read certificate {}", cert_file.display()))?;
    #[cfg(feature = "ay")]
    {
        use tla_check::cert_all_n::{verify_all_n_certificate, AllNCertificate, AllNVerdict};
        let cert = AllNCertificate::from_json(&json).map_err(anyhow::Error::msg)?;
        let report = verify_all_n_certificate(&cert);
        println!("{}", report.detail);
        match report.verdict {
            AllNVerdict::Accepted => Ok(()),
            AllNVerdict::Rejected => std::process::exit(1),
            AllNVerdict::Inconclusive => std::process::exit(2),
        }
    }
    #[cfg(not(feature = "ay"))]
    {
        let _ = json;
        bail!("`ty alln-check` requires the `ay` feature");
    }
}

/// Run `ty live-check`.
pub(crate) fn cmd_livecheck(cert_file: &Path) -> Result<()> {
    let json = std::fs::read_to_string(cert_file)
        .with_context(|| format!("read certificate {}", cert_file.display()))?;

    // The ENUMERATOR-FREE COUNTDOWN cert (`ty.live-free-cert/v1`) is kernel-only (no enumerator, no
    // solver) — dispatch it FIRST by schema tag, and re-check it WITHOUT enumerating.
    #[cfg(feature = "clean-cic")]
    if json.contains("ty.live-free-cert/v1") {
        use tla_check::live_cert::{verify_liveness_free, LiveExplicitVerdict, LivenessFreeCert};
        let cert = LivenessFreeCert::from_json(&json).map_err(anyhow::Error::msg)?;
        let report = verify_liveness_free(&cert);
        println!("{}", report.detail);
        match report.verdict {
            LiveExplicitVerdict::Accepted => return Ok(()),
            LiveExplicitVerdict::Rejected => std::process::exit(1),
            LiveExplicitVerdict::Inconclusive => std::process::exit(2),
        }
    }

    // The EXPLICIT-STATE KERNEL cert (`ty.live-explicit-cert/v1`) is solver-free — dispatch it next
    // by schema tag, so `ty live-check` re-checks it even in a `clean-cic`-only (no `ay`) build.
    #[cfg(feature = "clean-cic")]
    if json.contains("ty.live-explicit-cert/v1") {
        use tla_check::live_cert::{
            verify_liveness_explicit, LiveExplicitVerdict, LivenessExplicitCert,
        };
        let cert = LivenessExplicitCert::from_json(&json).map_err(anyhow::Error::msg)?;
        let report = verify_liveness_explicit(&cert);
        println!("{}", report.detail);
        match report.verdict {
            LiveExplicitVerdict::Accepted => return Ok(()),
            LiveExplicitVerdict::Rejected => std::process::exit(1),
            LiveExplicitVerdict::Inconclusive => std::process::exit(2),
        }
    }

    #[cfg(feature = "ay")]
    {
        use tla_check::live_cert::{verify_liveness_certificate, LiveVerdict, LivenessCertificate};
        let cert = LivenessCertificate::from_json(&json).map_err(anyhow::Error::msg)?;
        let report = verify_liveness_certificate(&cert);
        println!("{}", report.detail);
        match report.verdict {
            LiveVerdict::Accepted => Ok(()),
            LiveVerdict::Rejected => std::process::exit(1),
            LiveVerdict::Inconclusive => std::process::exit(2),
        }
    }
    #[cfg(not(feature = "ay"))]
    {
        let _ = json;
        #[cfg(not(feature = "clean-cic"))]
        bail!("`ty live-check` requires the `ay` or `clean-cic` feature");
        #[cfg(feature = "clean-cic")]
        bail!(
            "not a recognized liveness certificate schema (expected `ty.live-explicit-cert/v1`; \
             the AY `ty.live-cert/v1` lane requires the `ay` feature)"
        );
    }
}
