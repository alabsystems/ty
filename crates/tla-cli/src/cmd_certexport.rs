// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! `ty cert-export` + the `--carcara` third-party re-check.
//!
//! A TY certificate embeds, per obligation, AY's own Alethe proof. **carcara**
//! (`github.com/ufmg-smite/carcara`) is a SEPARATE, independently-implemented
//! Alethe checker. Re-checking the embedded proofs with carcara is genuine
//! N-version redundancy: a false *PROVE* would need the same bug in BOTH AY's
//! strict checker AND carcara.
//!
//! carcara needs a separate `problem.smt2` (its proof parser rejects `declare-fun`
//! lines, and AY's Alethe is not self-contained — no `set-logic`/`assert`/
//! `check-sat`). So we SPLIT the embedded `alethe`: `declare-fun` → the problem
//! preamble; `(assume id term)` / `(step ...)` → the proof; each `(assume _ term)`
//! also becomes `(assert term)` in the problem; step-indexed state vars are
//! declared; `set-logic` + `check-sat` bracket it.
//!
//! HONEST SCOPE: carcara re-checks the UNSAT *proof* of each SMT obligation — NOT
//! the spec→SMT translation (that is bound by Leg D's render + engine-diverse +
//! independent-front-end checks). The structural deadlock-freedom obligation has
//! no Alethe proof and is excluded.

use std::collections::BTreeSet;
use std::path::Path;
use std::process::Command;

use anyhow::{bail, Context, Result};
use serde::Deserialize;

use tla_check::cert::AyObligationProof;
use tla_check::{BmcScalarSymbol, BmcTranslator};

/// The common shape across all three certificate kinds (safety / liveness /
/// all-N) — they all carry `schema`, `var_sorts`, and `ay_proof_obligations`.
#[derive(Deserialize)]
struct AnyCert {
    #[serde(default)]
    schema: String,
    #[serde(default)]
    var_sorts: Vec<(String, String)>,
    #[serde(default)]
    ay_proof_obligations: Vec<AyObligationProof>,
}

/// True for an obligation with no carcara-checkable Alethe proof (empty, or the
/// structural deadlock-freedom marker).
fn is_structural(o: &AyObligationProof) -> bool {
    o.alethe.trim().is_empty() || o.alethe.starts_with("structural:")
}

/// Extract the asserted term from an Alethe `(assume <id> <term>)` line.
fn extract_assume_term(line: &str) -> Option<String> {
    let inner = line.trim().strip_prefix("(assume")?.trim();
    let inner = inner.strip_suffix(')')?; // drop the assume's closing paren
    let (_id, term) = inner.trim().split_once(char::is_whitespace)?;
    let term = term.trim();
    if term.is_empty() {
        None
    } else {
        Some(term.to_string())
    }
}

/// SMT-LIB / Alethe word-operators that appear in terms and must NOT be declared
/// as free symbols (the non-alphabetic operators like `<=` / `+` are not even
/// scanned, since symbols start with a letter or `_`).
const TERM_KEYWORDS: &[&str] = &[
    "and", "or", "not", "ite", "true", "false", "div", "mod", "distinct", "abs", "let", "exists",
    "forall", "select", "store", "to_int", "to_real", "is_int",
];

/// Collect the free variables and rigid constants appearing in an Alethe assume
/// term so the reconstructed problem can declare their canonical BMC symbols.
fn collect_free_symbols(text: &str, out: &mut BTreeSet<String>) {
    let bytes = text.as_bytes();
    let is_sym = |c: u8| c.is_ascii_alphanumeric() || c == b'_';
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i].is_ascii_alphabetic() || bytes[i] == b'_' {
            let start = i;
            while i < bytes.len() && is_sym(bytes[i]) {
                i += 1;
            }
            let tok = &text[start..i];
            if !TERM_KEYWORDS.contains(&tok) {
                out.insert(tok.to_string());
            }
        } else {
            i += 1;
        }
    }
}

/// Split an AY Alethe proof into a carcara-checkable `(problem.smt2,
/// proof.alethe)` pair (see module docs). `var_sorts` maps a state-var base name
/// to its SMT sort string (`"Int"` / `"Bool"`); unknown bases default to `Int`.
pub(crate) fn split_alethe_for_carcara(
    alethe: &str,
    var_sorts: &[(String, String)],
) -> (String, String) {
    let mut declares: Vec<String> = Vec::new();
    let mut declared: BTreeSet<String> = BTreeSet::new();
    let mut proof_lines: Vec<String> = Vec::new();
    let mut asserts: Vec<String> = Vec::new();
    // Free symbols are collected from the ASSUME terms only (the problem premises),
    // not the step lines (which carry rule names / `cl` that are not symbols).
    let mut free_syms: BTreeSet<String> = BTreeSet::new();

    for raw in alethe.lines() {
        let line = raw.trim();
        if line.is_empty() {
            continue;
        }
        if line.starts_with("(declare-fun") {
            if let Some(sym) = line.split_whitespace().nth(1) {
                declared.insert(sym.to_string());
            }
            declares.push(line.to_string());
        } else if line.starts_with("(assume") {
            proof_lines.push(line.to_string());
            if let Some(term) = extract_assume_term(line) {
                collect_free_symbols(&term, &mut free_syms);
                asserts.push(format!("(assert {term})"));
            }
        } else {
            proof_lines.push(line.to_string());
        }
    }

    let sort_of = |base: &str| -> &str {
        match var_sorts.iter().find(|(n, _)| n == base) {
            Some((_, s)) if s == "Bool" => "Bool",
            _ => "Int",
        }
    };
    let mut var_decls: Vec<String> = Vec::new();
    for sym in &free_syms {
        if declared.contains(sym) {
            continue;
        }
        // Decode the collision-free source name for sort lookup. Preserve the
        // old `base__step` parser so already-issued certificates remain
        // exportable after the naming migration.
        let base = match BmcTranslator::parse_scalar_symbol(sym) {
            Some(BmcScalarSymbol::State { name, .. }) | Some(BmcScalarSymbol::Rigid { name }) => {
                name
            }
            None => match sym.rsplit_once("__") {
                Some((base, step))
                    if !base.is_empty() && step.bytes().all(|byte| byte.is_ascii_digit()) =>
                {
                    base.to_string()
                }
                _ => sym.clone(),
            },
        };
        var_decls.push(format!("(declare-fun {sym} () {})", sort_of(&base)));
    }

    let mut problem = String::from("(set-logic QF_UFLIA)\n");
    for d in &declares {
        problem.push_str(d);
        problem.push('\n');
    }
    for d in &var_decls {
        problem.push_str(d);
        problem.push('\n');
    }
    for a in &asserts {
        problem.push_str(a);
        problem.push('\n');
    }
    problem.push_str("(check-sat)\n");

    let mut proof = proof_lines.join("\n");
    proof.push('\n');
    (problem, proof)
}

/// Resolve the carcara binary: `CARCARA_PATH`, then `~/.cargo/bin/carcara`, then
/// `PATH`.
fn find_carcara() -> Option<std::path::PathBuf> {
    if let Ok(p) = std::env::var("CARCARA_PATH") {
        let p = std::path::PathBuf::from(p);
        if p.exists() {
            return Some(p);
        }
    }
    if let Ok(home) = std::env::var("HOME") {
        let p = std::path::PathBuf::from(home).join(".cargo/bin/carcara");
        if p.exists() {
            return Some(p);
        }
    }
    // PATH lookup via `command -v`.
    let out = Command::new("carcara").arg("--version").output().ok()?;
    if out.status.success() {
        Some(std::path::PathBuf::from("carcara"))
    } else {
        None
    }
}

/// carcara's verdict for one obligation.
pub(crate) enum CarcaraVerdict {
    /// `valid` — trust-free, fully re-checked by carcara.
    Valid,
    /// `holey` — verified but with a hole/trust step (REJECT for N-version).
    Holey,
    /// `invalid` — carcara rejected the proof.
    Invalid(String),
    /// carcara is not installed (Inconclusive — never an accept).
    Missing,
}

/// Run carcara on one obligation's embedded Alethe proof (trust-free; requires
/// stdout `valid`, since `holey` also exits 0).
pub(crate) fn carcara_check_obligation(
    carcara: Option<&Path>,
    alethe: &str,
    var_sorts: &[(String, String)],
) -> CarcaraVerdict {
    let Some(carcara) = carcara else {
        return CarcaraVerdict::Missing;
    };
    let (problem, proof) = split_alethe_for_carcara(alethe, var_sorts);
    let dir = match tempdir() {
        Some(d) => d,
        None => return CarcaraVerdict::Invalid("could not create temp dir".into()),
    };
    let problem_path = dir.join("problem.smt2");
    let proof_path = dir.join("proof.alethe");
    if std::fs::write(&problem_path, problem).is_err()
        || std::fs::write(&proof_path, proof).is_err()
    {
        return CarcaraVerdict::Invalid("could not write temp files".into());
    }
    // `carcara check -- <proof> <problem>` WITHOUT --allowed-rules (trust-free).
    let out = Command::new(carcara)
        .arg("check")
        .arg("--")
        .arg(&proof_path)
        .arg(&problem_path)
        .output();
    let _ = std::fs::remove_dir_all(&dir);
    match out {
        Ok(o) => {
            let stdout = String::from_utf8_lossy(&o.stdout);
            let last = stdout.trim().lines().last().unwrap_or("").trim();
            match last {
                "valid" => CarcaraVerdict::Valid,
                "holey" => CarcaraVerdict::Holey,
                _ => {
                    let stderr = String::from_utf8_lossy(&o.stderr);
                    CarcaraVerdict::Invalid(
                        stderr.lines().next().unwrap_or(last).trim().to_string(),
                    )
                }
            }
        }
        Err(e) => CarcaraVerdict::Invalid(format!("could not run carcara: {e}")),
    }
}

/// Re-check every (non-structural) obligation of a parsed certificate with
/// carcara, printing a third-party verdict line per obligation. Returns
/// `Some(false)` if any obligation is `holey`/`invalid` (an additional REJECT),
/// `Some(true)` if all checked obligations are `valid`, `None` if carcara is
/// absent (Inconclusive note — never flips an accept).
pub(crate) fn carcara_recheck_certificate(json: &str) -> Result<Option<bool>> {
    let cert: AnyCert =
        serde_json::from_str(json).context("parse certificate for carcara re-check")?;
    let carcara = find_carcara();
    if carcara.is_none() {
        println!(
            "  carcara: NOT FOUND (set CARCARA_PATH or install carcara) — third-party \
             re-check skipped (this is NOT a verdict on the certificate)"
        );
        return Ok(None);
    }
    let mut all_valid = true;
    let mut checked = 0usize;
    for o in &cert.ay_proof_obligations {
        if is_structural(o) {
            continue;
        }
        checked += 1;
        match carcara_check_obligation(carcara.as_deref(), &o.alethe, &cert.var_sorts) {
            CarcaraVerdict::Valid => {
                println!(
                    "  carcara [{}]: valid (trust-free, N-version re-checked)",
                    o.name
                )
            }
            CarcaraVerdict::Holey => {
                all_valid = false;
                println!(
                    "  carcara [{}]: HOLEY — proof has a trust/hole step (rejected)",
                    o.name
                );
            }
            CarcaraVerdict::Invalid(why) => {
                all_valid = false;
                println!("  carcara [{}]: INVALID — {why}", o.name);
            }
            CarcaraVerdict::Missing => return Ok(None),
        }
    }
    println!(
        "  carcara: independently re-checked the UNSAT PROOF of {checked} SMT obligation(s); \
         this does NOT validate the spec->SMT translation (bound separately by Leg D)."
    );
    Ok(Some(all_valid))
}

/// A unique temp directory under the system temp dir (no external crate).
fn tempdir() -> Option<std::path::PathBuf> {
    let base = std::env::temp_dir();
    let pid = std::process::id();
    for n in 0..1000u32 {
        let p = base.join(format!("ty-carcara-{pid}-{n}"));
        if std::fs::create_dir(&p).is_ok() {
            return Some(p);
        }
    }
    None
}

/// Run `ty cert-export`: write a carcara-checkable `<name>.problem.smt2` +
/// `<name>.proof.alethe` per (non-structural) obligation into `out_dir`.
pub(crate) fn cmd_cert_export(cert_file: &Path, out_dir: &Path) -> Result<()> {
    let json = std::fs::read_to_string(cert_file)
        .with_context(|| format!("read certificate {}", cert_file.display()))?;
    let cert: AnyCert = serde_json::from_str(&json).context("parse certificate")?;
    std::fs::create_dir_all(out_dir)
        .with_context(|| format!("create out-dir {}", out_dir.display()))?;

    let mut written = 0usize;
    for o in &cert.ay_proof_obligations {
        if is_structural(o) {
            continue;
        }
        let (problem, proof) = split_alethe_for_carcara(&o.alethe, &cert.var_sorts);
        let pp = out_dir.join(format!("{}.problem.smt2", o.name));
        let pf = out_dir.join(format!("{}.proof.alethe", o.name));
        std::fs::write(&pp, problem).with_context(|| format!("write {}", pp.display()))?;
        std::fs::write(&pf, proof).with_context(|| format!("write {}", pf.display()))?;
        written += 1;
    }
    println!(
        "exported {written} obligation proof(s) from `{}` to {}\n\
         re-check each with a third-party Alethe checker, e.g.:\n  \
         carcara check -- <name>.proof.alethe <name>.problem.smt2   # expect: valid",
        cert.schema,
        out_dir.display()
    );
    if written == 0 {
        bail!("no carcara-checkable obligations (all structural or empty Alethe)");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_alethe_recovers_bool_sorts_from_canonical_state_and_rigid_symbols() {
        let state = BmcTranslator::state_step_symbol("x__0_step_7", 12);
        let rigid = BmcTranslator::rigid_const_symbol("N__0");
        let alethe = format!("(assume a0 (and {state} {rigid}))\n");
        let (problem, _) = split_alethe_for_carcara(
            &alethe,
            &[
                ("x__0_step_7".to_string(), "Bool".to_string()),
                ("N__0".to_string(), "Bool".to_string()),
            ],
        );

        assert!(problem.contains(&format!("(declare-fun {state} () Bool)")));
        assert!(problem.contains(&format!("(declare-fun {rigid} () Bool)")));
    }

    #[test]
    fn split_alethe_retains_legacy_step_symbol_sort_lookup() {
        let (problem, _) = split_alethe_for_carcara(
            "(assume a0 legacy__name__3)\n",
            &[("legacy__name".to_string(), "Bool".to_string())],
        );
        assert!(problem.contains("(declare-fun legacy__name__3 () Bool)"));
    }
}
