// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Shared fail-closed well-formedness gate for EVERY certificate-minting entry point.
//!
//! `ty check` (the trusted TLC-parity model checker) runs full name resolution and config
//! validation before it will run a spec. The certificate lanes historically skipped those
//! stages, so a spec `ty check` REFUSES as ill-formed (a duplicate definition, an undefined
//! operator, a `.cfg` naming a non-existent SYMMETRY/CONSTRAINT/ACTION_CONSTRAINT operator)
//! could still be CERTIFIED — a false safe, because the certifier silently read a DIFFERENT
//! spec than `ty check` does (e.g. `find_op` keeps the first of two duplicate definitions).
//!
//! This module runs the SAME gates `ty check` enforces, BEFORE any lane mints. A decline
//! never weakens an obligation, so it is always sound. It is a SHARED helper (not per-lane)
//! specifically so no minting entry point is missed: an earlier version gated only plain
//! `ty certify` and left `certify-liveness` / `certify-all-n` / `refine-certify` ungated —
//! the exact "incomplete fix" pattern this false-safe family repeatedly shows. Every entry
//! point that writes a certificate MUST call [`certify_wellformedness_gate`] on each spec
//! source (and its config) it certifies.
//!
//! Parity is enforced by REUSING `check`'s machinery, not re-implementing it: the semantic
//! gate roots a `ModuleLoader` at the originating spec's directory and resolves against the
//! very same loaded EXTENDS/INSTANCE closure `check` resolves against, so "certify declines
//! IFF check declines" holds by construction. See [`semantic_gate`].

use std::path::Path;

use anyhow::{bail, Result};
use tla_check::Config;

/// Run BOTH well-formedness gates on one certified spec source (+ its config). Bails
/// (declines to mint) on any `ty check`-parity ill-formedness. Call this from EVERY
/// certificate entry point before it writes a certificate.
///
/// `spec_file` is the ORIGINATING spec path (not the flattened text): the semantic gate
/// roots a `ModuleLoader` at its directory so it can load the SAME EXTENDS/INSTANCE closure
/// `ty check` loads, and thus reproduce `check`'s exact accept/reject decision (see
/// [`semantic_gate`]).
pub(crate) fn certify_wellformedness_gate(
    source: &str,
    config: &Config,
    spec_file: &Path,
) -> Result<()> {
    semantic_gate(source, spec_file)?;
    config_operator_gate(source, config)?;
    Ok(())
}

/// Fail-closed name-resolution gate (certify/check parity).
///
/// INVARIANT: certify DECLINES a spec IFF `ty check` refuses to run it for a
/// name-resolution reason — no stricter, no looser. This is enforced STRUCTURALLY: the
/// gate calls the SAME resolver `ty check` calls (`run_semantic_analysis` →
/// `resolve_with_extends_and_instances_with_options`) on the SAME inputs — the module PLUS
/// its loaded EXTENDS/INSTANCE closure. So every NAME-RESOLUTION ill-formedness `check`
/// catches still fires (a duplicate definition; an undefined operator reachable from
/// INIT/NEXT/INVARIANTS or appearing in a THEOREM body), and NOTHING is dropped or invented.
/// SCOPE, precisely: this resolver surfaces UNDEFINED-identifier and DUPLICATE-definition
/// errors; it does NOT construct arity/kind-mismatch errors, so — like `check`'s own
/// resolver — the gate is not the layer that rejects those. An arity/kind error is caught
/// fail-closed DOWNSTREAM (the prover's successor enumeration / kernel PredIR recognizer
/// declines rather than mints), so certify never certifies against one; the gate simply is
/// not that layer. Do not read "arity/kind" into this gate's guarantee.
///
/// The earlier version resolved the flattened `source` STANDALONE (`&[], &[]`), on the
/// premise that the certify lanes always inline EXTENDS/INSTANCE. That premise is FALSE for
/// a spec carrying a *surviving* standalone `INSTANCE M` the flattener declines to inline
/// (e.g. VoucherIssue's `INSTANCE VoucherLifeCycle` under its non-identity wrapper): a
/// standalone resolve then reports `M`'s operators (referenced e.g. in a `THEOREM`) as
/// undefined and certify DECLINED a spec `ty check` ACCEPTS — a false decline. Rooting a
/// `ModuleLoader` at `spec_file`'s directory and loading that same closure fixes it: for a
/// genuinely self-contained flattened source the closure is empty and this degrades to the
/// old standalone resolve (byte-identical behavior); only a surviving INSTANCE pulls in the
/// sibling that DEFINES the names — exactly the symbols `check` sees. A genuine typo stays
/// undefined either way (its provider does not exist), so the guard is retained in full.
///
/// If lowering fails to produce a module at all, this is a no-op: the downstream prover
/// lanes already decline such input via the normal `NOT CERTIFIED` path, and this gate must
/// not change that error shape.
fn semantic_gate(source: &str, spec_file: &Path) -> Result<()> {
    let tree = tla_core::parse_to_syntax_tree(source);
    let Some(module) = tla_core::lower(tla_core::FileId(0), &tree).module else {
        return Ok(());
    };
    // Reproduce `ty check`'s EXACT resolution context (cmd_check/mod.rs + setup.rs): seed the
    // main file's inline modules, load the EXTENDS/INSTANCE closure rooted at the spec dir,
    // then resolve against the SAME `(extended, instanced)` module lists `check` feeds
    // `run_semantic_analysis`. A load failure is a fail-closed decline (`check` also bails
    // when a referenced module cannot be loaded).
    let mut loader = tla_core::ModuleLoader::new(spec_file);
    loader.seed_from_syntax_tree(&tree, spec_file);
    loader.load_extends(&module).map_err(|e| {
        anyhow::anyhow!(
            "NOT CERTIFIED: cannot load an EXTENDS-referenced module `ty check` would load — {e}"
        )
    })?;
    loader.load_instances(&module).map_err(|e| {
        anyhow::anyhow!(
            "NOT CERTIFIED: cannot load an INSTANCE-referenced module `ty check` would load — {e}"
        )
    })?;
    let (extended_modules, instanced_modules) = loader.modules_for_semantic_resolution(&module);
    let result = tla_core::resolve_with_extends_and_instances_with_options(
        &module,
        &extended_modules,
        &instanced_modules,
        tla_core::ResolveOptions::model_checking(),
    );
    if !result.errors.is_empty() {
        for err in &result.errors {
            eprintln!("  semantic error: {err}");
        }
        bail!(
            "NOT CERTIFIED: spec is ill-formed — {} semantic error(s) (the same name-resolution \
             gate `ty check` enforces). certify declines rather than certifying a mis-read spec.",
            result.errors.len()
        );
    }
    Ok(())
}

/// Fail-closed gate for config-referenced operators (certify/check parity).
///
/// `ty check` rejects a run whose `.cfg` names a SYMMETRY / CONSTRAINT / ACTION_CONSTRAINT
/// operator absent from the module (a hard `CONFIG`/`Setup` error). The certify lanes
/// ignored these directives outright, so a config `check` refuses to run was still certified.
/// This re-checks that each such name is a defined operator in the flattened certificate
/// module and declines otherwise. A decline never weakens an obligation, so it is always
/// sound.
///
/// Scope: SYMMETRY, CONSTRAINT, ACTION_CONSTRAINT — the directives `check` treats as HARD
/// failures when unresolved. VIEW is intentionally excluded: `check` only WARNS on a missing
/// VIEW operator (and proceeds with full-state fingerprints), so declining on it would be a
/// stricter policy than `check`, not parity.
fn config_operator_gate(source: &str, config: &Config) -> Result<()> {
    let tree = tla_core::parse_to_syntax_tree(source);
    let Some(module) = tla_core::lower(tla_core::FileId(0), &tree).module else {
        return Ok(());
    };
    let defined: std::collections::HashSet<&str> = module
        .units
        .iter()
        .filter_map(|u| match &u.node {
            tla_core::ast::Unit::Operator(op) => Some(op.name.node.as_str()),
            _ => None,
        })
        .collect();

    let mut missing: Vec<(&str, &str)> = Vec::new();
    if let Some(sym) = &config.symmetry {
        if !defined.contains(sym.as_str()) {
            missing.push(("SYMMETRY", sym.as_str()));
        }
    }
    for c in &config.constraints {
        if !defined.contains(c.as_str()) {
            missing.push(("CONSTRAINT", c.as_str()));
        }
    }
    for ac in &config.action_constraints {
        if !defined.contains(ac.as_str()) {
            missing.push(("ACTION_CONSTRAINT", ac.as_str()));
        }
    }
    if !missing.is_empty() {
        for (kind, name) in &missing {
            eprintln!("  config error: {kind} operator `{name}` is not defined in the module");
        }
        bail!(
            "NOT CERTIFIED: configuration is ill-formed — {} config-operator reference(s) name \
             an undefined operator (the same references `ty check` hard-fails on). certify \
             declines rather than certifying against a mis-read configuration.",
            missing.len()
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tla_check::Config;

    fn write(dir: &Path, name: &str, body: &str) -> std::path::PathBuf {
        let p = dir.join(name);
        std::fs::write(&p, body).unwrap();
        p
    }

    fn init_next_cfg() -> Config {
        Config {
            init: Some("Init".to_string()),
            next: Some("Next".to_string()),
            ..Default::default()
        }
    }

    /// (a) The VoucherIssue shape: a name that is undefined in the bare module but PROVIDED
    /// by a loaded standalone `INSTANCE` sibling, referenced ONLY in a `THEOREM`, PASSES the
    /// gate. Standalone resolution (the pre-fix behavior) flagged such a name undefined and
    /// declined a spec `ty check` accepts; loading the closure (what `check` does) defines it.
    #[test]
    fn instance_provided_name_only_in_theorem_passes() {
        let dir = tempfile::tempdir().unwrap();
        write(
            dir.path(),
            "Sib.tla",
            "---- MODULE Sib ----\nEXTENDS Naturals\nVARIABLE x\nSibProp == x >= 0\n====\n",
        );
        let main = write(
            dir.path(),
            "Main.tla",
            "---- MODULE Main ----\nEXTENDS Naturals\nVARIABLE x\n\
             Init == x = 0\nNext == x' = x + 1\n\
             INSTANCE Sib\nTHEOREM Init => SibProp\n====\n",
        );
        let src = std::fs::read_to_string(&main).unwrap();
        assert!(
            certify_wellformedness_gate(&src, &init_next_cfg(), &main).is_ok(),
            "a THEOREM-only name provided by a loaded INSTANCE must pass (parity with check)"
        );
    }

    /// (a') GUARD against the candidate-(b) false safe: a GENUINE undefined name (no
    /// INSTANCE/EXTENDS provides it), even appearing ONLY in a `THEOREM`, STILL DECLINES.
    /// `ty check` walks THEOREM bodies and refuses, so the gate must too — the fix drops
    /// nothing, so a typo stays undefined and fails closed.
    #[test]
    fn genuine_typo_only_in_theorem_still_declines() {
        let dir = tempfile::tempdir().unwrap();
        let main = write(
            dir.path(),
            "Solo.tla",
            "---- MODULE Solo ----\nEXTENDS Naturals\nVARIABLE x\n\
             Init == x = 0\nNext == x' = x + 1\n\
             THEOREM Init => UndefinedTypo\n====\n",
        );
        let src = std::fs::read_to_string(&main).unwrap();
        assert!(
            certify_wellformedness_gate(&src, &init_next_cfg(), &main).is_err(),
            "a genuine undefined name in a THEOREM must decline (parity with check)"
        );
    }

    /// (b) An undefined operator REACHABLE from INIT/NEXT still FAILS the gate.
    #[test]
    fn undefined_reachable_from_init_declines() {
        let dir = tempfile::tempdir().unwrap();
        let main = write(
            dir.path(),
            "B.tla",
            "---- MODULE B ----\nEXTENDS Naturals\nVARIABLE x\n\
             Init == x = MissingOp\nNext == x' = x\n====\n",
        );
        let src = std::fs::read_to_string(&main).unwrap();
        assert!(
            certify_wellformedness_gate(&src, &init_next_cfg(), &main).is_err(),
            "an undefined operator reachable from INIT must decline"
        );
    }

    /// (c) A duplicate INIT-reachable definition still FAILS the gate.
    #[test]
    fn duplicate_definition_declines() {
        let dir = tempfile::tempdir().unwrap();
        let main = write(
            dir.path(),
            "C.tla",
            "---- MODULE C ----\nEXTENDS Naturals\nVARIABLE x\n\
             Init == x = 0\nInit == x = 1\nNext == x' = x\n====\n",
        );
        let src = std::fs::read_to_string(&main).unwrap();
        assert!(
            certify_wellformedness_gate(&src, &init_next_cfg(), &main).is_err(),
            "a duplicate definition must decline"
        );
    }

    /// (d) The config-operator gate is unchanged: a `.cfg` naming a SYMMETRY operator absent
    /// from the module still DECLINES.
    #[test]
    fn missing_symmetry_operator_declines() {
        let dir = tempfile::tempdir().unwrap();
        let main = write(
            dir.path(),
            "D.tla",
            "---- MODULE D ----\nEXTENDS Naturals\nVARIABLE x\n\
             Init == x = 0\nNext == x' = x\n====\n",
        );
        let src = std::fs::read_to_string(&main).unwrap();
        let cfg = Config {
            symmetry: Some("NoSuchSym".to_string()),
            ..init_next_cfg()
        };
        assert!(
            certify_wellformedness_gate(&src, &cfg, &main).is_err(),
            "a missing SYMMETRY operator must decline (config gate unchanged)"
        );
    }

    /// A genuinely self-contained flattened source (no EXTENDS/INSTANCE closure) resolves
    /// standalone exactly as before the fix — the loaded closure is empty, so a well-formed
    /// spec still PASSES.
    #[test]
    fn self_contained_wellformed_passes() {
        let dir = tempfile::tempdir().unwrap();
        let main = write(
            dir.path(),
            "OK.tla",
            "---- MODULE OK ----\nEXTENDS Naturals\nVARIABLE x\n\
             Init == x = 0\nNext == x' = x + 1\n====\n",
        );
        let src = std::fs::read_to_string(&main).unwrap();
        assert!(
            certify_wellformedness_gate(&src, &init_next_cfg(), &main).is_ok(),
            "a well-formed self-contained spec must pass"
        );
    }
}
