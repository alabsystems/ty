// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! `ty refine-certify` / `ty refine-check`: KERNEL-CERTIFIED refinement mappings.
//!
//! `ty refine-certify Impl.tla --config Impl.cfg --abstract Abs.tla --abstract-config Abs.cfg`
//! certifies that the implementation refines the abstract spec (safety part, enumerated
//! implementation graph): the Clean CIC kernel re-evaluates the ABSTRACT `Init`/`Next` at
//! every mapped implementation initial state and transition (with the stuttering disjunct
//! `[Next_abs]_vars` requires), and the certificate is independently re-checkable with
//! `ty refine-check` (re-enumerates the graph, re-recognizes the abstract predicates,
//! re-runs the kernel).

use std::path::Path;

use anyhow::{bail, Result};
// The helpers below (and the anyhow::Context / tla_check::Config they use) exist
// only to support the clean-cic-gated refine/refine-check bodies; the default
// build's not(clean-cic) arms just bail, so gate them to stay warning-clean.
#[cfg(feature = "clean-cic")]
use anyhow::Context;
#[cfg(feature = "clean-cic")]
use tla_check::Config;

#[cfg(feature = "clean-cic")]
use crate::helpers::read_source;

#[cfg(feature = "clean-cic")]
fn load_config(path: &Path) -> Result<Config> {
    let src =
        std::fs::read_to_string(path).with_context(|| format!("read config {}", path.display()))?;
    match Config::parse(&src) {
        Ok(c) => Ok(c),
        Err(errors) => {
            for err in &errors {
                eprintln!("{}:{}: {}", path.display(), err.line(), err);
            }
            bail!("config parse failed with {} error(s)", errors.len());
        }
    }
}

/// Decompose a `SPECIFICATION Spec` config (`Spec == Init /\ [][Next]_vars ...`) into `INIT`/`NEXT`
/// against `source`, mutating `config`. Same resolution the model-checking and safety-cert lanes
/// use. An INLINE next relation (`[][\E n \in N: A(n)]_v`) is injected into the (returned) source as
/// a NAMED operator so the certificate re-checks from its own embedded text alone. Fail-closed on a
/// name clash, a non-lowering injection, or an unresolvable SPECIFICATION. `role` names the spec
/// (impl/abstract) in messages. Returns the (possibly modified) source.
#[cfg(feature = "clean-cic")]
fn resolve_spec_into_init_next(source: String, config: &mut Config, role: &str) -> Result<String> {
    if !(config.init.is_none() || config.next.is_none()) || config.specification.is_none() {
        return Ok(source);
    }
    let mut source = source;
    let tree = tla_core::parse_to_syntax_tree(&source);
    let resolved = tla_check::resolve_spec_from_config_with_extends(config, &tree, &[])
        .map_err(|e| anyhow::anyhow!("cannot resolve {role} SPECIFICATION to INIT/NEXT — {e}"))?;
    let mut next_name = resolved.next.clone();
    if let Some(node) = &resolved.next_node {
        // INLINE next: the resolver synthesized an internal relation with no operator name a
        // self-contained certificate can reference — inject a NAMED operator with the same body.
        let name = "TyInlineNext";
        if source.contains(name) {
            bail!(
                "cannot synthesize `{name}` for the inline {role} next-state relation — name taken"
            );
        }
        let body = node.text().to_string();
        let Some(end) = source
            .rfind("\n====")
            .map(|p| p + 1)
            .or_else(|| source.find("===="))
        else {
            bail!("{role} module has no terminating ====");
        };
        source.insert_str(end, &format!("{name} == {body}\n"));
        let t2 = tla_core::parse_to_syntax_tree(&source);
        let l2 = tla_core::lower(tla_core::FileId(0), &t2);
        if l2.module.is_none() || !l2.errors.is_empty() {
            bail!("the inline {role} next-state relation does not lower as a standalone operator");
        }
        next_name = name.to_string();
        println!("note: inline {role} next-state relation synthesized as operator `{name}`");
    }
    if config.init.is_none() {
        config.init = Some(resolved.init.clone());
    }
    if config.next.is_none() {
        config.next = Some(next_name.clone());
    }
    println!(
        "note: {role} SPECIFICATION resolved to INIT `{}` / NEXT `{next_name}`",
        resolved.init
    );
    Ok(source)
}

/// Parse `--map "abs1=<expr1>,abs2=<expr2>"` into `(abstract, RHS-expression)` pairs. Each RHS
/// is an affine expression over the IMPLEMENTATION variables: a bare variable (`flag=c`) is a
/// PROJECTION, a compound affine combination (`sum=a+b`) is a DERIVED (aggregate) mapping —
/// certify recognizes it and fails closed outside the exact-affine fragment.
#[cfg(feature = "clean-cic")]
fn parse_map(map: Option<&str>) -> Result<Vec<(String, String)>> {
    let Some(map) = map else {
        return Ok(Vec::new());
    };
    map.split(',')
        .filter(|s| !s.trim().is_empty())
        .map(|pair| {
            let (a, i) = pair
                .split_once('=')
                .with_context(|| format!("--map entry `{pair}` is not `abs=<expr>`"))?;
            Ok((a.trim().to_string(), i.trim().to_string()))
        })
        .collect()
}

/// Run `ty refine`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn cmd_refine_certify(
    impl_file: &Path,
    impl_config: &Path,
    abs_file: &Path,
    abs_config: &Path,
    map: Option<&str>,
    out: &Path,
) -> Result<()> {
    #[cfg(feature = "clean-cic")]
    {
        let impl_src = read_source(impl_file)?;
        let abs_src = read_source(abs_file)?;
        let mut impl_cfg = load_config(impl_config)?;
        let mut abs_cfg = load_config(abs_config)?;
        // Decompose SPECIFICATION-form configs (`Spec == Init /\ [][Next]_vars`) into INIT/NEXT — the
        // corpus's transaction_commit configs (TwoPhase's TPSpec, TCommit's TCSpec) declare a
        // SPECIFICATION, not explicit INIT/NEXT.
        let impl_src = resolve_spec_into_init_next(impl_src, &mut impl_cfg, "implementation")?;
        let abs_src = resolve_spec_into_init_next(abs_src, &mut abs_cfg, "abstract")?;
        // Fail-closed well-formedness gate (certify/check parity) on BOTH the implementation and
        // abstract specs — shared with `ty certify`. A refinement certificate re-evaluates the
        // abstract predicates over the implementation graph, so an ill-formed spec on either side
        // must decline rather than certifying a mis-read module.
        crate::cert_gate::certify_wellformedness_gate(&impl_src, &impl_cfg, impl_file)?;
        crate::cert_gate::certify_wellformedness_gate(&abs_src, &abs_cfg, abs_file)?;
        let var_map = parse_map(map)?;

        // Tally the second checker across the run (every refinement obligation is in the
        // Nat/Bool bool_true_eq fragment, so ck0 corroboration is expected, not incidental).
        tla_check::ck0_bridge::begin_tally();
        let Some(cert) = tla_check::refinement_cert::certify_refinement(
            &impl_src, &impl_cfg, &abs_src, &abs_cfg, &var_map,
        ) else {
            tla_check::ck0_bridge::take_tally();
            eprintln!(
                "NOT CERTIFIED (refinement): either the implementation has a transition (or \
                 initial state) whose image violates the abstract spec — i.e. it genuinely \
                 does not refine — or the specs are outside this lane's enumerable / \
                 exactly-recognizable fragment (configured CONSTANTs, subtraction/division, \
                 non-Int columns in arithmetic, quantifier folds). Fail-closed either way: \
                 no claim is made."
            );
            std::process::exit(2);
        };
        // Immediately re-verify before claiming anything (the CLI never emits an unchecked cert).
        if !tla_check::refinement_cert::verify_refinement_cert(&cert) {
            tla_check::ck0_bridge::take_tally();
            bail!("internal error: freshly minted refinement cert failed its own re-check");
        }
        let ck0 = tla_check::ck0_bridge::take_tally().unwrap_or_default();
        std::fs::write(out, cert.to_json())
            .with_context(|| format!("write certificate {}", out.display()))?;
        println!(
            "KERNEL-CERTIFIED REFINEMENT (safety part, enumerated implementation graph): every \
             one of the {} implementation transition(s) over {} reachable state(s) maps to an \
             abstract `{}` step or stutter, and every initial state maps into abstract `{}` — \
             each obligation RE-EVALUATED by the Clean CIC kernel.",
            cert.transitions.len(),
            cert.impl_reachable.len(),
            cert.abs_next,
            cert.abs_init,
        );
        println!(
            "trust base: the kernel + ty's enumerator (completeness of the transition set) + \
             ty's abstract-predicate recognizer (exactness-filtered to the truth-direction \
             fragment); fairness/liveness NOT covered. second checker: clean-ck0 corroborated \
             {} of the kernel obligations ({} not corroborated).",
            ck0.corroborated, ck0.unavailable
        );
        println!(
            "certificate -> {}\nre-check: ty refine-check {}",
            out.display(),
            out.display()
        );
        Ok(())
    }
    #[cfg(not(feature = "clean-cic"))]
    {
        let _ = (impl_file, impl_config, abs_file, abs_config, map, out);
        bail!("`ty refine` requires a `clean-cic` build (the kernel is not linked)");
    }
}

/// Run `ty refine-check`.
pub(crate) fn cmd_refine_check(cert_path: &Path) -> Result<()> {
    #[cfg(feature = "clean-cic")]
    {
        let src = std::fs::read_to_string(cert_path)
            .with_context(|| format!("read certificate {}", cert_path.display()))?;
        let cert = tla_check::refinement_cert::RefinementCert::from_json(&src)
            .map_err(|e| anyhow::anyhow!(e))?;
        if tla_check::refinement_cert::verify_refinement_cert(&cert) {
            println!(
                "VERIFIED (refinement, against the certificate's EMBEDDED spec sources — \
                 confirm they are the modules you mean): implementation graph re-enumerated \
                 ({} states, {} transitions), abstract `{}`/`{}` re-recognized, and both \
                 kernel legs re-accepted.",
                cert.impl_reachable.len(),
                cert.transitions.len(),
                cert.abs_init,
                cert.abs_next,
            );
            Ok(())
        } else {
            eprintln!("REJECTED: the refinement certificate failed independent re-checking.");
            std::process::exit(1);
        }
    }
    #[cfg(not(feature = "clean-cic"))]
    {
        let _ = cert_path;
        bail!("`ty refine-check` requires a `clean-cic` build");
    }
}
