// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! `ty certify` subcommand: emit a re-checkable inductive-safety certificate.
//!
//! Runs the AY-backed inductive-safety prover and serializes the proven invariant
//! into a self-contained `ty.cert/v1` JSON certificate that `ty cert-check`
//! re-validates independently (AY-only, no external solver). Certifying
//! verification: a proof you can check yourself, not just a verdict.

use std::path::Path;

use anyhow::{bail, Context, Result};
use tla_check::Config;

use crate::helpers::read_source;

/// Resolve the config path (explicit `--config`, else `<spec>.cfg`).
fn resolve_config(file: &Path, config_path: Option<&Path>) -> Result<std::path::PathBuf> {
    match config_path {
        Some(p) => Ok(p.to_path_buf()),
        None => {
            let mut cfg_path = file.to_path_buf();
            cfg_path.set_extension("cfg");
            if !cfg_path.exists() {
                bail!(
                    "No config file specified and {} does not exist.\n\
                     Use --config to specify a configuration file.",
                    cfg_path.display()
                );
            }
            Ok(cfg_path)
        }
    }
}

/// Run `ty certify`.
pub(crate) fn cmd_certify(
    file: &Path,
    config_path: Option<&Path>,
    out: &Path,
    require_domain_complete: bool,
    no_deadlock: bool,
) -> Result<()> {
    let _ = read_source(file)?; // existence/readability check with the standard error shape

    // R6-lite (docs/north-star-roadmap.md): flatten the EXTENDS closure into ONE
    // self-contained module source — corpus `MC.tla` wrappers become certifiable, and
    // the certificate stays self-contained (it embeds the FLATTENED source; the
    // re-checker needs nothing else). Fail-closed on INSTANCE/LOCAL/name clashes.
    let flat = crate::flatten::flatten_extends_closure(file).map_err(|e| {
        anyhow::anyhow!("NOT CERTIFIED: cannot flatten to a self-contained module — {e}")
    })?;
    if !flat.inlined.is_empty() {
        println!(
            "note: module composition (EXTENDS/full-module INSTANCE) resolved into the \
             certificate source (inlined: {})",
            flat.inlined.join(", ")
        );
    }
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
    // SPECIFICATION-form configs: decompose `Spec == Init /\ [][Next]_vars ...` into
    // INIT/NEXT against the (flattened, self-contained) module — same resolution the
    // model-checking lanes use.
    let mut source = source;
    if (config.init.is_none() || config.next.is_none()) && config.specification.is_some() {
        let tree = tla_core::parse_to_syntax_tree(&source);
        match tla_check::resolve_spec_from_config_with_extends(&config, &tree, &[]) {
            Ok(resolved) => {
                let mut next_name = resolved.next.clone();
                // INLINE next (`[][\E n \in N: A(n)]_v`): the checker synthesizes an
                // internal operator, which a self-contained certificate cannot reference.
                // Inject a NAMED operator with the same body into the embedded source, so
                // the cert re-checks from its own text alone. Fail-closed on clash or if
                // the injected source no longer lowers.
                if let Some(node) = &resolved.next_node {
                    let name = "TyInlineNext";
                    if source.contains(name) {
                        bail!(
                            "NOT CERTIFIED: cannot synthesize `{name}` for the inline \
                             next-state relation — the name is already taken"
                        );
                    }
                    let body = node.text().to_string();
                    // Anchor on the terminator LINE start (`\n====`), NOT `rfind("====")`
                    // — the latter matches the rightmost 4-char window INSIDE a long
                    // `====…====` terminator (corpus specs use 70+ chars), splitting it and
                    // stranding the injected operator after a truncated terminator.
                    let Some(end) = source
                        .rfind("\n====")
                        .map(|p| p + 1)
                        .or_else(|| source.find("===="))
                    else {
                        bail!("NOT CERTIFIED: flattened module has no terminating ====");
                    };
                    source.insert_str(end, &format!("{name} == {body}\n"));
                    let t2 = tla_core::parse_to_syntax_tree(&source);
                    let l2 = tla_core::lower(tla_core::FileId(0), &t2);
                    if l2.module.is_none() || !l2.errors.is_empty() {
                        bail!(
                            "NOT CERTIFIED: the inline next-state relation does not lower \
                             as a standalone operator (fail-closed)"
                        );
                    }
                    next_name = name.to_string();
                    println!("note: inline next-state relation synthesized as operator `{name}`");
                }
                if config.init.is_none() {
                    config.init = Some(resolved.init.clone());
                }
                if config.next.is_none() {
                    config.next = Some(next_name.clone());
                }
                println!(
                    "note: SPECIFICATION resolved to INIT `{}` / NEXT `{next_name}`",
                    resolved.init
                );
            }
            Err(e) => bail!("NOT CERTIFIED: cannot resolve SPECIFICATION to INIT/NEXT — {e}"),
        }
    }
    let config = config;

    // Deadlock policy — MIRRORS `ty check` (`cmd_check::helpers::select_check_deadlock`): ON by
    // default, disabled by `--no-deadlock`, and ALSO honoring a `CHECK_DEADLOCK FALSE` line in the
    // `.cfg` (`config.check_deadlock`). With deadlock checking ON, `ty certify` verifies that every
    // reachable state has a successor under Next and DECLINES a spec with a reachable deadlock —
    // matching `ty check`, which reports such a spec as a Deadlock. (The AY symbolic lane below always
    // includes deadlock-freedom in its proof — it refuses deadlocking specs regardless of this flag.)
    let check_deadlock = !no_deadlock && config.check_deadlock;

    // Fail-closed well-formedness gate (certify/check parity) — declines an ill-formed spec or
    // config `ty check` would refuse, so neither certify lane mints a certificate for a mis-read
    // spec. Shared with every other certificate entry point (see `cert_gate`).
    crate::cert_gate::certify_wellformedness_gate(&source, &config, file)?;

    #[cfg(feature = "ay")]
    {
        // MANDATORY MINT-SIDE SELF-VERIFICATION (the d1a0f1d3 discipline, fixed-N lane):
        // the symbolic synthesis lane's certificate must pass the SAME offline re-check
        // `ty cert-check` runs — with FULL acceptance. A cert whose own re-check is
        // Inconclusive/Rejected (e.g. an uncovered consecution obligation after a fragment
        // widening let this lane claim a spec it previously declined) is treated as a lane
        // DECLINE so the explicit-state / parametric KERNEL lane below gets its chance —
        // never shipping a weaker verdict for a spec the kernel lane fully certifies.
        let symbolic_cert = tla_check::cert::certify_spec(&source, &config).filter(|cert| {
            let report = tla_check::cert::verify_safety_certificate(cert);
            let ok = matches!(report.verdict, tla_check::cert::CertVerdict::Accepted);
            if !ok {
                println!(
                    "note: symbolic lane minted a certificate but its mandatory offline                      self-verification was NOT fully accepted ({:?}) — treating as a lane                      decline and trying the kernel lane",
                    report.verdict
                );
            }
            ok
        });
        match symbolic_cert {
            Some(cert) => {
                std::fs::write(out, cert.to_json())
                    .with_context(|| format!("write certificate {}", out.display()))?;
                println!(
                    "CERTIFIED: inductive-safety + deadlock-freedom — invariant `{}` holds on \
                     all reachable states and Next never deadlocks\n\
                     certificate -> {}\n\
                     re-check (external proof re-check, no re-solve) with: ty cert-check {}",
                    cert.invariant_j_tla,
                    out.display(),
                    out.display()
                );
                // Honest FIXED-INSTANCE scope: a certificate proves the CONCRETE
                // configured instance (constants concretized), not an all-N family.
                let constants = tla_check::cert::declared_constants(&source);
                if !constants.is_empty() {
                    println!(
                        "note: fixed-instance — this certificate covers the configured values of \
                         CONSTANT {} only, NOT all N (parametric/all-N proving is future work)",
                        constants.join(", ")
                    );
                }
                Ok(())
            }
            None => {
                // FALLBACK: the SYMBOLIC synthesis lane declined. Try the explicit-state /
                // parametric-unbounded kernel path.
                //
                // The two lanes are complementary, and `certify_spec` runs FIRST here on purpose:
                //   * `certify_spec` (cert.rs) is the GENERAL invariant-SYNTHESIS lane. It does not
                //     merely check a given J — it SYNTHESIZES one: it first tries `J = Safety` directly
                //     (1-inductive via the ay SMT solver), then STRENGTHENS with derived candidates
                //     (per-variable lower bounds, then intervals — a Houdini-lite search) until an
                //     inductive J is found (`ay_bmc::prove_inductive_safety_for_cert` /
                //     `derive_strengthening_candidates`). It handles UNBOUNDED specs (it dropped the old
                //     divergence trigger) and multi-variable specs, and emits a re-checkable
                //     `SafetyCertificate` with per-obligation proof legs.
                //   * The explicit-state path below is either (a) a finite kernel fixpoint (enumerate R,
                //     kernel-check Init⊆R ∧ R closed-under-Next ∧ R⊆Safety), or (b) — for an INFINITE
                //     reachable set — the PARAMETRIC unbounded lane: a fast, enumeration-free special
                //     case that kernel-proves the affine-nonneg fragment (`⋀_j x_j≥0` under
                //     `x_j'=x_j+δ_j`) and the relational `x=y` fragment (lock-step `+δ`) via
                //     universally-quantified CIC legs (`Int.NonNeg.add` / `And.intro` / `Eq.subst`).
                //
                // So: a spec the parametric path DECLINES (non-affine, guarded-but-infinite, a
                // relational invariant beyond `x=y`, or one needing genuine interval strengthening) is
                // ALREADY attempted by `certify_spec` first; and a spec `certify_spec` declines but that
                // is in the parametric fragment is caught here. General invariant SYNTHESIS beyond the
                // Houdini-lite candidate set (arbitrary relational/PDR-style invariants) remains open —
                // the parametric path is exact/fail-closed over its fragment, not a general synthesizer.
                match try_explicit_state_certify(
                    &source,
                    &config,
                    out,
                    require_domain_complete,
                    check_deadlock,
                ) {
                    ExplicitCertOutcome::Certified => return Ok(()),
                    ExplicitCertOutcome::DeadlockDeclined => {
                        // A reachable DEADLOCK was found with deadlock checking ON (default) — the
                        // decline message (naming the state) was already printed. Exit non-zero,
                        // matching `ty check`, which reports this spec as a Deadlock. Re-run with
                        // `--no-deadlock` to certify inductive SAFETY only.
                        std::process::exit(2);
                    }
                    ExplicitCertOutcome::NotApplicable => {}
                }
                eprintln!(
                    "NOT CERTIFIED: this spec is not in the inductive-safety provable class \
                     (no certificate emitted). The normal `ty check` verdict still applies."
                );
                // North star: "wherever ty cannot deliver, it says so exactly." Re-run the
                // pipeline's stages one by one and print every stage-level decline reason.
                // These are DIAGNOSTIC reasons, not promises — an empty stage report means
                // the decline is in a deeper kernel/prover leg.
                let stage_reasons =
                    tla_check::certify_explain::explain_certify_decline(&source, &config);
                if stage_reasons.is_empty() {
                    eprintln!(
                        "  reason: all probed pipeline stages pass — the decline is in a \
                         deeper kernel/prover leg (no stage-level reason identified)"
                    );
                } else {
                    for r in &stage_reasons {
                        eprintln!("  reason: {r}");
                    }
                }
                std::process::exit(2);
            }
        }
    }

    #[cfg(not(feature = "ay"))]
    {
        let _ = (
            out,
            &config,
            &source,
            require_domain_complete,
            check_deadlock,
        );
        bail!("`ty certify` requires the `ay` feature (the AY-backed inductive prover)");
    }
}

/// Outcome of the explicit-state fixpoint certify attempt.
#[cfg(all(feature = "clean-cic", feature = "ay"))]
enum ExplicitCertOutcome {
    /// A certificate was produced AND re-verified via the Clean kernel (the `KERNEL-CERTIFIED` lines
    /// and deadlock-freedom verdict were printed; the cert was written).
    Certified,
    /// Deadlock checking was ON (default) and a reachable DEADLOCK was found — the decline message was
    /// printed and NO cert was written. The caller declines with a non-zero exit (matching `ty check`).
    DeadlockDeclined,
    /// This spec is not in the explicit-state fixpoint fragment (or the kernel re-check failed). The
    /// caller falls through to the generic NOT CERTIFIED path.
    NotApplicable,
}

/// LIVE explicit-state fixpoint Certified attempt: enumerate the model checker's reachable set `R`
/// and kernel-CHECK the fixpoint witness. Prints the `KERNEL-CERTIFIED` lines + deadlock-freedom
/// verdict and returns [`ExplicitCertOutcome::Certified`] iff a cert was produced AND re-verifies via
/// the Clean kernel; [`ExplicitCertOutcome::DeadlockDeclined`] when deadlock checking is ON (default)
/// and a reachable deadlock is found; else [`ExplicitCertOutcome::NotApplicable`] (fail-closed:
/// out-of-fragment or kernel rejection). Only active under `clean-cic`. (`ay`-gated: the only caller is
/// the `ay` certify path; this avoids an unused-fn warning otherwise.)
#[cfg(all(feature = "clean-cic", feature = "ay"))]
fn try_explicit_state_certify(
    source: &str,
    config: &Config,
    out: &std::path::Path,
    require_domain_complete: bool,
    check_deadlock: bool,
) -> ExplicitCertOutcome {
    use tla_check::explicit_fixpoint_cert::{
        certify_explicit_state_spec, certify_explicit_state_spec_strict_domain,
        verify_explicit_state_cert,
    };
    // Phase A: the fail-closed domain-coverage mode declines Rust-derived domain bounds.
    let certify = if require_domain_complete {
        certify_explicit_state_spec_strict_domain
    } else {
        certify_explicit_state_spec
    };
    let Some(cert) = certify(source, config) else {
        return ExplicitCertOutcome::NotApplicable;
    };
    // Tally the tiny second checker (clean-ck0; Phase 1, docs/kernel-checked-tla-plan.md) and
    // the gate's axiom accounting (B#3) across the VERIFY pass ONLY — it re-runs
    // kernel_accepts on each leg exactly once, so the printed counts are per-OBLIGATION
    // (tallying the mint pass too would double-count every leg).
    tla_check::ck0_bridge::begin_tally();
    tla_check::kernel_census::begin_axiom_tally();
    // Re-run the Clean kernel on every leg (the arbiter) before claiming Certified.
    if !verify_explicit_state_cert(&cert) {
        tla_check::ck0_bridge::take_tally();
        tla_check::kernel_census::take_axiom_tally();
        return ExplicitCertOutcome::NotApplicable;
    }
    let ck0 = tla_check::ck0_bridge::take_tally().unwrap_or_default();
    let axioms = tla_check::kernel_census::take_axiom_tally().unwrap_or_default();

    // ── DEADLOCK-FREEDOM decision (certify/check parity; policy = `check_deadlock`) ─────────────────
    // The mint BFS recorded the enumerator's deadlock verdict on `cert.deadlock_scan` — the SAME
    // successor enumeration `ty check` uses to decide deadlock (a self-loop `x'=x` enumerates a
    // successor, so is NOT a deadlock). With deadlock checking ON (default) and NO `TERMINAL` predicate
    // configured, a reachable state with no successor is a DEADLOCK: DECLINE (naming it), exactly as
    // `ty check` reports it. This runs BEFORE the cert is written, so a declined spec leaves no file.
    if check_deadlock && config.terminal.is_none() {
        if let Some(state) = cert.deadlock_scan.0.as_ref() {
            let tup = state
                .iter()
                .map(|n| n.to_string())
                .collect::<Vec<_>>()
                .join(",");
            eprintln!(
                "NOT CERTIFIED: reachable DEADLOCK — state ({tup}) has NO successor under Next, so \
                 deadlock-freedom fails (`ty check` reports this spec as Deadlock). The inductive \
                 invariant itself holds over the reachable set; re-run with `--no-deadlock` to certify \
                 inductive SAFETY only (with a deadlock-freedom-unverified disclosure)."
            );
            return ExplicitCertOutcome::DeadlockDeclined;
        }
    }
    // Whether a kernel COMPLETENESS/parametric leg fired (the ENUMERATOR-FREE tier — the kernel itself
    // re-evaluated Next over the domain / proved the parametric invariant), vs. closure resting ONLY on
    // the enumerated `image ⊆ R` leg (which trusts TY's enumerator to have found every successor).
    let enumerator_free = cert.unbounded_invariant.is_some()
        || cert.next_shape.is_some()
        || cert.next_completeness.is_some()
        || cert.next_general_completeness.is_some();
    // Whether R ⊇ {Init states} is ALSO kernel-proven over the domain (vs. resting on the enumerated
    // initial states). The `enumerator_free` flag above is strictly about CLOSURE (the Next RELATION);
    // a spec can have enumerator-free closure while its Init stays outside the general Init fragment
    // (e.g. CoffeeCan's filtered record-set comprehension `{c ∈ Can : c.black+c.white ∈ 1..N}`). The
    // header must not claim to have re-evaluated Init when only the closure legs fired.
    let init_enumerator_free = cert.unbounded_invariant.is_some()
        || cert.init_shape.is_some()
        || cert.init_completeness.is_some()
        || cert.init_general_completeness.is_some();
    // SERIALIZE into the SafetyCertificate schema + WRITE it so `ty cert-check` can independently re-check.
    let sc = tla_check::cert::build_explicit_fixpoint_certificate(source, config, cert.clone());
    if let Err(e) = std::fs::write(out, sc.to_json()) {
        eprintln!(
            "warning: could not write certificate to {}: {e}",
            out.display()
        );
    }
    let r = cert
        .reachable
        .iter()
        .map(|t| {
            format!(
                "({})",
                t.iter()
                    .map(|n| n.to_string())
                    .collect::<Vec<_>>()
                    .join(",")
            )
        })
        .collect::<Vec<_>>()
        .join(", ");
    if cert.unbounded_invariant.is_some() {
        println!(
            "KERNEL-CERTIFIED (unbounded explicit-state): the Clean CIC kernel proved a PARAMETRIC \
             inductive invariant (Init⇒J, J∧Next⇒J', J⇒Safety) over the INFINITE state space — no \
             enumeration."
        );
    } else if enumerator_free && init_enumerator_free {
        println!(
            "KERNEL-CERTIFIED (explicit-state fixpoint): R = {{{r}}} ({} state(s)); the Clean CIC kernel \
             RE-EVALUATED Init AND Next over the finite domain and re-checked all legs — Init⊆R, closure over \
             the Next RELATION (not TY's enumerated sample), and R⊆Safety. Trust base = the kernel, NOT \
             the model checker.",
            cert.reachable.len()
        );
    } else if enumerator_free {
        // CLOSURE is enumerator-free (the kernel re-evaluated the Next RELATION over the domain), but the
        // Init predicate stayed outside the general Init fragment, so R ⊇ {Init states} rests on the
        // enumerated initial states. HONEST: claim enumerator-free closure ONLY, not for Init.
        println!(
            "KERNEL-CERTIFIED (explicit-state fixpoint, ENUMERATOR-FREE CLOSURE): R = {{{r}}} ({} state(s)); \
             the Clean CIC kernel RE-EVALUATED the Next RELATION over the finite domain and re-checked \
             closure (image over the Next relation ⊆ R, NOT TY's enumerated sample) and R⊆Safety. Trust \
             base for CLOSURE = the kernel, NOT the model checker. NOTE: Init⊆R rests on the enumerated \
             initial states (this Init is outside the general Init-completeness fragment), so R ⊇ {{Init \
             states}} still trusts TY's enumerator — closure is enumerator-free, initial coverage is not.",
            cert.reachable.len()
        );
    } else {
        // Honest reason for the missing completeness legs: strict mode DECLINED them by
        // policy (Rust-derived domain bounds), vs. the spec genuinely being outside the
        // kernel-re-evaluation fragment.
        let why = if require_domain_complete {
            "the kernel-re-evaluation legs were DECLINED by --require-domain-complete (their \
             domain bounds are Rust-derived, not universe-complete)"
        } else {
            "this Next is outside the kernel-re-evaluation fragment"
        };
        println!(
            "KERNEL-CERTIFIED (explicit-state fixpoint, ENUMERATOR-ASSISTED): R = {{{r}}} ({} state(s)); \
             the Clean CIC kernel re-checked Init⊆R, image⊆R, and R⊆Safety over the ENUMERATED reachable \
             set. NOTE: {why}, so closure (image⊆R) rests \
             on TY's enumerator having found every successor — this tier is NOT enumerator-free (trust \
             base = the kernel + TY's enumerator for successor-completeness).",
            cert.reachable.len()
        );
    }
    // Deadlock-freedom VERDICT (certify/check parity). With deadlock checking ON (the default), the
    // decline branch above already returned for a deadlocking spec, so reaching here means every
    // reachable state has a successor — state HOW that was established (kernel-corroborated vs
    // enumerator), matching `ty check`'s successor-enumeration semantics. With `--no-deadlock` (or a
    // `CHECK_DEADLOCK FALSE` cfg), fall back to the honest safety-only scope disclosure. (The AY
    // symbolic lane, by contrast, always REFUSES deadlocking specs and claims deadlock-freedom.)
    if !check_deadlock {
        println!(
            "scope: deadlock checking DISABLED (--no-deadlock / CHECK_DEADLOCK FALSE) — this \
             certificate proves inductive SAFETY only (Init⊆R, R closed under Next, R⊆Safety). It does \
             NOT verify deadlock-freedom; `ty check` checks for deadlock by default, so a spec that \
             deadlocks can still be safety-CERTIFIED here."
        );
    } else if config.terminal.is_some() {
        // A TERMINAL predicate exempts configured end-states from deadlock; this lane does not model
        // that exemption, so it does not claim deadlock-freedom (honest — never a false label).
        println!(
            "deadlock-freedom: NOT VERIFIED — a TERMINAL predicate is configured and this lane does \
             not model terminal-state exemption. Inductive SAFETY is certified above; run `ty check` \
             for the deadlock verdict."
        );
    } else if cert.unbounded_invariant.is_some() {
        println!(
            "deadlock-freedom: VERIFIED — the recognized unbounded Next (x'=x+δ) is unconditionally \
             enabled, so every state has the successor x'=x+δ (no reachable deadlock)."
        );
    } else if cert.deadlock_free.is_some() {
        println!(
            "deadlock-freedom: VERIFIED — every one of the {} reachable state(s) has a successor. The \
             Clean CIC kernel re-checked ⟦Next⟧(s, wₛ) = Bool.true at an enumerated witness successor \
             wₛ of each reachable state s (the same recognized Next relation the closure leg \
             re-evaluates). The deadlock DECISION matches `ty check`'s successor enumeration; because \
             the recognized Next over-approximates, this kernel leg CORROBORATES the enumerator's \
             witnesses (trust base = the kernel-re-checked Next-embedding + TY's witness successors).",
            cert.reachable.len()
        );
    } else {
        println!(
            "deadlock-freedom: VERIFIED (enumerator-assisted) — every one of the {} reachable state(s) \
             has an enumerated successor under Next. This tier's Next is outside the kernel-re-evaluation \
             fragment, so deadlock-freedom rests on TY's enumerator (exactly as the enumerator-assisted \
             closure leg), not on a kernel Next-reduction.",
            cert.reachable.len()
        );
    }
    // The general R⊆Safety lane (invariants beyond the `⋀ x≥0` shape): say WHICH leg carries the
    // spec's Safety claim. This changes nothing about the Next-completeness story above — the
    // enumerator-free/assisted tier is decided by the closure legs alone.
    if cert.safety_pred.is_some() {
        println!(
            "safety leg: the configured invariant is NOT the `⋀ x≥0` shortcut shape — the kernel \
             reduced the GENERAL embedded invariant over every reachable state \
             (`⋀_{{s∈R}} Safety(s)` ⇒ Bool.true, the `safety_general` leg)."
        );
    }
    // The HONEST trust base (Phase 0 corrected the old "~2 kLOC" claim): the primary checker is
    // clean-kernel's ~33K-LOC checker proper — plus its structurally-admitted prelude construction
    // and native def-eq reducers — inside a ~722K-LOC linked crate. The tiny-TCB claim attaches
    // ONLY to the obligations the independent second checker re-checked.
    println!(
        "trust base: clean-kernel CIC checker (~33K-LOC checker inside a ~722K-LOC crate, incl. \
         structurally-admitted prelude + native reducers)."
    );
    if ck0.corroborated > 0 {
        println!(
            "second checker: clean-ck0 (~8K-LOC src, #![forbid(unsafe_code)], zero shared production \
             code, axiom-free env) INDEPENDENTLY re-checked {} kernel obligation(s) — decided by \
             ck0's own checker, reached via ty's fail-closed translator; {} obligation(s) not \
             corroborated (Int legs, or ck0's depth/budget caps) remain clean-kernel-tier.",
            ck0.corroborated, ck0.unavailable
        );
    } else {
        println!(
            "second checker: clean-ck0 corroborated no obligation on this run ({} not in its \
             Nat/Bool fragment or past its depth/budget caps) — the verdict rests on the \
             clean-kernel tier alone.",
            ck0.unavailable
        );
    }
    // Phase A: the honest D ⊇ Succ(R) coverage line — WHY the kernel's product domain covers
    // every successor, per completeness leg. RECOMPUTED from the cert (never trusted).
    let coverage = tla_check::explicit_fixpoint_cert::domain_coverage_of_cert(&cert);
    if coverage.any_completeness_leg {
        // HONESTY: `Succ(R)` phrasing only applies when a NEXT-completeness leg is present. A cert
        // whose only kernel-re-evaluated leg is the INIT one (a spec whose Next stays outside the
        // kernel-re-evaluation fragment; HourClock GRADUATED from this class when the IF-desugared
        // Or shape reached the general leg) must not read as successor coverage — say which leg
        // the coverage classifies.
        let has_next_leg =
            cert.next_completeness.is_some() || cert.next_general_completeness.is_some();
        let scope = if has_next_leg {
            ""
        } else {
            " (Init leg ONLY — closure is enumerator-assisted, see above)"
        };
        let kernel_proven = coverage.next_kernel_columns.len() + coverage.init_kernel_columns.len();
        if coverage.fully_construction_covered() {
            if kernel_proven > 0 {
                println!(
                    "domain coverage{scope}: the kernel-re-evaluation domain(s) cover WITHOUT \
                     trusting any Rust bound rule — {kernel_proven} axis/axes KERNEL-PROVEN \
                     (a Π-quantified domain-bound lemma — `∀successor. Next ⇒ bound` for a Next \
                     axis, `∀state. Init ⇒ bound` for an Init axis — was synthesized and \
                     accepted by the kernel); the rest are their columns' full SORT universes."
                );
            } else {
                println!(
                    "domain coverage: every completeness-leg axis is its column's full SORT \
                     universe — no per-column bound rule is trusted (sort-faithfulness itself is \
                     attested by the enumerated values, as everywhere in the explicit-state lane)."
                );
            }
        } else {
            let mut rust_bits = Vec::new();
            if coverage.shortcut_legs {
                rust_bits.push("the single-Int shortcut leg(s)".to_string());
            }
            if !coverage.next_rust_columns.is_empty() {
                rust_bits.push(format!(
                    "Next-leg column(s) {:?}",
                    coverage.next_rust_columns
                ));
            }
            if !coverage.init_rust_columns.is_empty() {
                rust_bits.push(format!(
                    "Init-leg column(s) {:?}",
                    coverage.init_rust_columns
                ));
            }
            println!(
                "domain coverage: {} rely on TRUSTED-RUST bound rules (primed-bound/stutter) — \
                 surfaced; the kernel proves closure RELATIVE to D, and D ⊇ Succ(R) for those \
                 axes rests on TY's rules. Pass --require-domain-complete to decline such legs.",
                rust_bits.join(" and ")
            );
        }
    }
    // B#3: axiom accounting of every accepted proof term. Domain-axiom-backed legs are
    // admitted (Phase 0 blocks only trust markers) but must be VISIBLE in the verdict.
    if axioms.axiom_dependent > 0 {
        println!(
            "axiom closure: {} obligation(s) constructive; {} obligation(s) rest on admitted \
             domain axioms ({}) — honestly AxiomDependent, not foundational-clean (see `ty \
             tcb-census`).",
            axioms.constructive,
            axioms.axiom_dependent,
            axioms
                .domain_axioms_seen
                .iter()
                .cloned()
                .collect::<Vec<_>>()
                .join(", ")
        );
    } else if axioms.constructive > 0 {
        println!(
            "axiom closure: all {} accepted obligation(s) constructive — no proof term rests \
             on any axiom beyond the foundational base.",
            axioms.constructive
        );
    }
    println!(
        "certificate -> {}\nre-check: ty cert-check {}",
        out.display(),
        out.display()
    );
    ExplicitCertOutcome::Certified
}

#[cfg(all(feature = "ay", not(feature = "clean-cic")))]
#[allow(dead_code)] // `Certified`/`DeadlockDeclined` are matched by the caller but only produced by the
                    // `clean-cic` path; without the kernel this lane always returns `NotApplicable`.
enum ExplicitCertOutcome {
    Certified,
    DeadlockDeclined,
    NotApplicable,
}

#[cfg(all(feature = "ay", not(feature = "clean-cic")))]
fn try_explicit_state_certify(
    _source: &str,
    _config: &Config,
    _out: &std::path::Path,
    _require_domain_complete: bool,
    _check_deadlock: bool,
) -> ExplicitCertOutcome {
    ExplicitCertOutcome::NotApplicable
}

/// `ty tcb-census` (plan item B#3, `docs/kernel-checked-tla-plan.md`): print the honest
/// trust census of the REAL kernel env `Certified` verdicts type-check against — the
/// number an auditor must weigh, as opposed to the curated "3-axiom" certificate env.
#[cfg(feature = "clean-cic")]
pub(crate) fn cmd_tcb_census(full: bool) -> Result<()> {
    let census = tla_check::kernel_census::kernel_trust_census();
    println!(
        "TRUST CENSUS of the REAL kernel env (`Environment::with_prelude()` — what \
         `Certified` verdicts type-check against):"
    );
    println!(
        "  declarations: {}  theorems: {} ({} constructive, {} axiom-dependent)",
        census.total_declarations,
        census.theorems,
        census.constructive_theorems,
        census.axiom_dependent_theorems
    );
    println!(
        "  axioms: {} total; {} non-foundational (domain) axioms",
        census.axioms, census.total_domain_axioms
    );
    println!(
        "  TRUST MARKERS present in the env: {} — each types `{{α : Sort u}} → α` (a \
         closed proof of ANY proposition). The Phase-0 gate REJECTS any Certified term \
         whose transitive closure reaches one; their presence in the env is why \
         `check_type` alone is not sufficient.",
        census.trust_markers.join(", ")
    );
    println!(
        "  NOTE: Clean's green \"3-axiom\" soundness certificate describes a CURATED \
         env (soundness_certificate_env), NOT this one. This census is the honest \
         operating number."
    );
    println!(
        "  AUDIT SURFACE (the CODE you trust, not the axiom data above): the proof checker \
         is ~23K LOC (clean-kernel `tc/`) or ~9K LOC via the independent ck0 second checker \
         that corroborates every reachable obligation. The internal trust-base notes keep the \
         measured, reproducible breakdown (checker / second checker / translation glue / \
         enumerator-for-assisted-certs-only) and the honest gap to \"a few thousand lines\"."
    );
    if full {
        println!("  domain axioms ({}):", census.domain_axioms.len());
        for ax in &census.domain_axioms {
            println!("    {ax}");
        }
    } else {
        println!(
            "  (run with --full to list all {} domain-axiom names)",
            census.domain_axioms.len()
        );
    }
    Ok(())
}

#[cfg(not(feature = "clean-cic"))]
pub(crate) fn cmd_tcb_census(_full: bool) -> Result<()> {
    bail!("`ty tcb-census` requires a `clean-cic` build (the kernel env is not linked)");
}
