// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! `ty corpus doctor`: the TY-vs-TLC **comparability preflight**.
//!
//! Answers one question, per eligible row of
//! `tests/tlc_comparison/strict_corpus_manifest.json`, before anyone spends
//! days of machine time collecting evidence:
//!
//! > *Can this row be compared against TLC at all, and do we already know where
//! > it stands?*
//!
//! # Why it exists
//!
//! The strict claim's definition of done requires that non-comparable rows be
//! "classified explicitly rather than silently omitted". Three ways a row can be
//! non-comparable were being discovered by hand, one session at a time:
//!
//! 1. **TLC cannot parse it.** 25 of 141 eligible rows `EXTENDS` a TLAPS proof
//!    module. Without the proof library TLC dies in the parser; with the repo's
//!    first-party stub it runs but the comparison is mediated by a TY-authored
//!    artifact. See [`crate::cmd_tlc::proof_library`].
//! 2. **Exact parity is unreachable on the upstream cfg.** TY refuses TLC's
//!    declared `SYMMETRY` orbit quotient when a property needs the genuine
//!    liveness checker, because that quotient is unsound for liveness. TY then
//!    explores the full space and TLC explores a quotient, so the manifest's
//!    exact distinct-state parity rule cannot hold. That is a **soundness
//!    premium**, and scoring it as a performance loss is the single most
//!    misleading thing the retained evidence does (`MCKVsnap` reads as a 0.13x
//!    runtime / 5.63x memory catastrophe while TY correctly explores 189,664
//!    states against TLC's 32,293).
//!
//!    The fix is *not* to exclude those rows — that would shrink the claim.
//!    The manifest declares a frozen symmetry-free **parity variant** for each,
//!    run by BOTH tools, which restores exact parity (verified: MCKVsnap
//!    189,664/1/365,596 and BufferedRandomAccessFile 6,376/1/248,698, identical
//!    from each tool). This preflight applies a declared variant only after its
//!    sha256 matches the manifest; a missing or altered variant is reported as
//!    `INVALID` and the row falls back to the upstream cfg rather than silently
//!    comparing against an unpinned file.
//! 3. **Nobody has measured it.** Neither condition applies, but no TY runtime
//!    exists in the baseline or anywhere else, so its standing is unknown.
//!
//! # How each check decides
//!
//! Both engine-side questions are answered by **running the real tools**, never
//! by pattern-matching source:
//!
//! * TLC parseability shells out to `tla2sany.SANY` with the exact classpath and
//!   `-DTLA-Library` the comparison harness uses. A parse that needs a module
//!   nobody supplied is reported with the module's name.
//! * The symmetry disposition re-invokes **this binary** as
//!   `ty check … --max-states 1`, which runs model preparation (where the
//!   decision is made) and stops before exploration — tens of milliseconds per
//!   row. Subprocess isolation is deliberate: a row that panics degrades to one
//!   `unknown` cell instead of killing the sweep.
//!
//! Following `cmd_corpus`/`cmd_tlc`, external tools are invoked as processes
//! rather than linked.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

use anyhow::{bail, Context, Result};
use serde::Serialize;
use serde_json::Value;

use crate::cli_schema::{CorpusDoctorFormat, SupremacyMode};
use crate::cmd_tlc::proof_library::{self, LibraryProvenance};

/// Schema tag for `--format json`.
const DOCTOR_SCHEMA: &str = "ty.corpus-doctor/v1";

/// How a row's TLC parse resolved.
#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
#[serde(tag = "status", rename_all = "snake_case")]
pub(crate) enum ParseStatus {
    /// SANY parsed the module cleanly.
    Ok,
    /// SANY could not resolve a module — the name is the actionable part.
    MissingModule { module: String },
    /// SANY reported errors that are not a missing module.
    ParseError { detail: String },
    /// SANY could not be run (no JDK, no jar) or timed out.
    Unknown { reason: String },
    /// The check was not requested.
    Skipped,
}

impl ParseStatus {
    fn is_blocking(&self) -> bool {
        matches!(self, Self::MissingModule { .. } | Self::ParseError { .. })
    }

    fn label(&self) -> String {
        match self {
            Self::Ok => "ok".into(),
            Self::MissingModule { module } => format!("missing:{module}"),
            Self::ParseError { .. } => "parse-error".into(),
            Self::Unknown { .. } => "unknown".into(),
            Self::Skipped => "skipped".into(),
        }
    }
}

/// What TY does with a declared `SYMMETRY` on this row.
#[derive(Clone, Copy, Debug, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum SymmetryDisposition {
    /// No `SYMMETRY` declared — nothing to decide.
    NotDeclared,
    /// Declared and applied: properties are pure safety, so the quotient is
    /// sound and TLC-exact parity remains reachable.
    RetainedSafety,
    /// Declared and **refused**: a property needs the genuine liveness checker.
    /// TY explores the full space, TLC explores the quotient, and exact
    /// distinct-state parity is impossible. Not a performance defect.
    DroppedForLiveness,
    /// Could not be determined (the probe failed or timed out).
    Unknown,
}

impl SymmetryDisposition {
    fn label(self) -> &'static str {
        match self {
            Self::NotDeclared => "-",
            Self::RetainedSafety => "retained",
            Self::DroppedForLiveness => "DROPPED(liveness)",
            Self::Unknown => "unknown",
        }
    }
}

/// Whether this row ran a frozen symmetry-free parity variant.
#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
#[serde(tag = "status", rename_all = "snake_case")]
pub(crate) enum VariantStatus {
    /// No variant declared for this row.
    None,
    /// A declared variant was found and its sha256 matched the manifest.
    Applied,
    /// A variant is declared but is missing or its digest does not match. The
    /// row falls back to the upstream cfg and is reported honestly rather than
    /// silently comparing against an unpinned file.
    Invalid(String),
}

impl VariantStatus {
    fn label(&self) -> &str {
        match self {
            Self::None => "-",
            Self::Applied => "applied",
            Self::Invalid(_) => "INVALID",
        }
    }
}

/// Verify a declared parity variant exists and matches its manifest digest.
fn verify_variant_digest(path: &Path, expected: &str) -> Result<(), String> {
    if !path.is_file() {
        return Err(format!("missing {}", path.display()));
    }
    match crate::cmd_tlc::sha256_hex(path) {
        Ok(got) if got.eq_ignore_ascii_case(expected) => Ok(()),
        Ok(got) => Err(format!("sha256 {got} != manifest {expected}")),
        Err(e) => Err(format!("hash failed: {e}")),
    }
}

/// Whether a current TY performance number exists for this row.
#[derive(Clone, Copy, Debug, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum MeasurementStatus {
    /// The baseline carries a TY runtime for this row.
    Present,
    /// The row exists in the baseline but has no TY runtime.
    Absent,
    /// The row is not in the baseline at all.
    NotInBaseline,
}

/// The single verdict for a row: what, if anything, blocks it from `PASS_BOTH`.
#[derive(Clone, Copy, Debug, Serialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub(crate) enum RowVerdict {
    /// Comparable and measured — nothing outstanding from the preflight.
    Ready,
    /// Comparable, but no TY performance number exists yet.
    Unmeasured,
    /// TLC cannot parse it with the resolved toolchain.
    TlcUnparseable,
    /// Exact parity is impossible without adopting TLC's unsound quotient.
    ParityImpossible,
    /// A probe failed; standing genuinely unknown.
    Indeterminate,
}

impl RowVerdict {
    fn label(self) -> &'static str {
        match self {
            Self::Ready => "ready",
            Self::Unmeasured => "unmeasured",
            Self::TlcUnparseable => "tlc-unparseable",
            Self::ParityImpossible => "parity-impossible",
            Self::Indeterminate => "indeterminate",
        }
    }

    /// Whether this verdict blocks a complete strict claim.
    fn blocks_claim(self) -> bool {
        !matches!(self, Self::Ready)
    }
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct RowReport {
    name: String,
    cfg_path: String,
    tla_path: String,
    verdict: RowVerdict,
    parse: ParseStatus,
    symmetry: SymmetryDisposition,
    variant: VariantStatus,
    measurement: MeasurementStatus,
}

#[derive(Debug, Serialize)]
struct ToolchainReport {
    corpus_dir: String,
    corpus_cfg_count: usize,
    manifest_path: String,
    manifest_sha256: Option<String>,
    baseline_path: String,
    tlc_jar: Option<String>,
    tlc_jar_sha256: Option<String>,
    community_modules: Option<String>,
    community_modules_sha256: Option<String>,
    java: Option<String>,
    tla_library: Option<String>,
    tla_library_provenance: Option<String>,
    tla_library_strict_qualified: bool,
    tlapm_pin: &'static str,
}

#[derive(Debug, Serialize)]
struct ReconciliationReport {
    eligible_rows: usize,
    excluded_rows: usize,
    baseline_entries: usize,
    eligible_missing_from_baseline: Vec<String>,
    baseline_entries_not_in_corpus: usize,
    baseline_entries_not_in_corpus_sample: Vec<String>,
    eligible_rows_with_ty_runtime: usize,
}

#[derive(Debug, Serialize)]
struct DoctorReport {
    schema: &'static str,
    toolchain: ToolchainReport,
    reconciliation: ReconciliationReport,
    summary: BTreeMap<String, usize>,
    rows: Vec<RowReport>,
    blocking_rows: usize,
}

// ---------------------------------------------------------------------------
// Manifest / baseline models (read-only views; the authoritative parsers live
// in cmd_supremacy::matrix, which validates far more than a preflight needs).
// ---------------------------------------------------------------------------

#[derive(Debug, serde::Deserialize)]
struct Manifest {
    schema_version: u64,
    claim: String,
    eligibility: ManifestEligibility,
    rows: Vec<ManifestRow>,
}

#[derive(Debug, serde::Deserialize)]
struct ManifestEligibility {
    exclusions: BTreeMap<String, Value>,
    /// Frozen symmetry-free cfg variants, keyed by the upstream cfg path.
    ///
    /// A row lands here when its upstream cfg declares `SYMMETRY` alongside a
    /// property that needs the genuine liveness checker: TLC applies the orbit
    /// quotient (unsound for liveness), TY soundly refuses it, and the two
    /// tools therefore do different work. Both tools run the variant instead,
    /// which restores exact generated-work parity.
    #[serde(default)]
    parity_variants: BTreeMap<String, ManifestParityVariant>,
}

#[derive(Debug, Clone, serde::Deserialize)]
struct ManifestParityVariant {
    variant_path: String,
    sha256: String,
}

#[derive(Debug, Clone, serde::Deserialize)]
struct ManifestRow {
    name: String,
    cfg_path: String,
    tla_path: String,
}

/// Entry point for `ty corpus doctor`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn cmd_doctor(
    corpus_dest: PathBuf,
    manifest: Option<PathBuf>,
    baseline: Option<PathBuf>,
    tlc_jar: Option<PathBuf>,
    community_modules: Option<PathBuf>,
    tla_library: Option<PathBuf>,
    format: CorpusDoctorFormat,
    mode: SupremacyMode,
    jobs: usize,
    filter: Option<String>,
    skip_parse: bool,
    out: Option<&Path>,
) -> Result<()> {
    let repo_root = repo_root();
    let manifest_path = manifest
        .unwrap_or_else(|| repo_root.join("tests/tlc_comparison/strict_corpus_manifest.json"));
    let baseline_path =
        baseline.unwrap_or_else(|| repo_root.join("tests/tlc_comparison/spec_baseline.json"));

    let manifest_text = std::fs::read_to_string(&manifest_path)
        .with_context(|| format!("read strict corpus manifest {}", manifest_path.display()))?;
    let manifest: Manifest = serde_json::from_str(&manifest_text)
        .with_context(|| format!("parse strict corpus manifest {}", manifest_path.display()))?;
    if manifest.schema_version != 1 || manifest.claim != "ty_vs_tlc_strict_superiority" {
        bail!(
            "unsupported manifest identity (schema={}, claim={})",
            manifest.schema_version,
            manifest.claim
        );
    }

    let specs_dir = corpus_dest.join("specifications");
    if !specs_dir.is_dir() {
        bail!(
            "corpus NOT found: {} does not exist. Run `ty corpus fetch`.",
            specs_dir.display()
        );
    }

    // ---- Resolve the toolchain, and say exactly what resolved. -------------
    let tlc_jar = tlc_jar.or_else(|| default_if_file(home().join("tlaplus/tytools.jar")));
    let community_modules =
        community_modules.or_else(|| default_if_file(home().join("tlaplus/CommunityModules.jar")));
    let (library, library_provenance) = resolve_tla_library(tla_library, &repo_root);

    let excluded: BTreeSet<&str> = manifest
        .eligibility
        .exclusions
        .keys()
        .map(|s| s.as_str())
        .collect();
    let eligible: Vec<ManifestRow> = manifest
        .rows
        .iter()
        .filter(|r| !excluded.contains(r.cfg_path.as_str()))
        .filter(|r| {
            filter
                .as_deref()
                .is_none_or(|f| r.name.contains(f) || r.cfg_path.contains(f))
        })
        .cloned()
        .collect();

    // ---- Baseline measurement coverage. ------------------------------------
    let baseline_specs = load_baseline_specs(&baseline_path);
    let baseline_entry_count = baseline_specs.as_ref().map(|m| m.len()).unwrap_or(0);

    // ---- Per-row probes, parallelised over `jobs`. -------------------------
    let ty_bin = std::env::current_exe().ok();
    let total = eligible.len();
    let next = AtomicUsize::new(0);
    let results: Mutex<Vec<Option<RowReport>>> = Mutex::new(vec![None; total]);

    let parity_variants = &manifest.eligibility.parity_variants;
    let probe = |idx: usize| -> RowReport {
        let row = &eligible[idx];
        let tla = specs_dir.join(&row.tla_path);
        // A frozen parity variant, when declared, is what BOTH tools run — so
        // it is also what the symmetry probe must ask about. Probing the
        // upstream cfg here would keep reporting `parity-impossible` for a row
        // the variant already made comparable.
        let (cfg, variant) = match parity_variants.get(&row.cfg_path) {
            Some(v) => {
                let path = repo_root.join(&v.variant_path);
                match verify_variant_digest(&path, &v.sha256) {
                    Ok(()) => (path, VariantStatus::Applied),
                    Err(reason) => (
                        specs_dir.join(&row.cfg_path),
                        VariantStatus::Invalid(reason),
                    ),
                }
            }
            None => (specs_dir.join(&row.cfg_path), VariantStatus::None),
        };

        let parse = if skip_parse {
            ParseStatus::Skipped
        } else {
            probe_tlc_parse(
                &tla,
                tlc_jar.as_deref(),
                community_modules.as_deref(),
                library.as_deref(),
            )
        };
        let symmetry = probe_symmetry(&tla, &cfg, ty_bin.as_deref());
        let measurement = match &baseline_specs {
            Some(specs) => match specs.get(&row.name) {
                Some(has_ty) => {
                    if *has_ty {
                        MeasurementStatus::Present
                    } else {
                        MeasurementStatus::Absent
                    }
                }
                None => MeasurementStatus::NotInBaseline,
            },
            None => MeasurementStatus::NotInBaseline,
        };

        // Precedence is deliberate: a row TLC cannot parse is not "unmeasured",
        // it is uncomparable, and reporting the softer verdict would understate
        // the work.
        let verdict = if parse.is_blocking() {
            RowVerdict::TlcUnparseable
        } else if symmetry == SymmetryDisposition::DroppedForLiveness {
            RowVerdict::ParityImpossible
        } else if matches!(parse, ParseStatus::Unknown { .. })
            || symmetry == SymmetryDisposition::Unknown
        {
            RowVerdict::Indeterminate
        } else if measurement == MeasurementStatus::Present {
            RowVerdict::Ready
        } else {
            RowVerdict::Unmeasured
        };

        RowReport {
            name: row.name.clone(),
            cfg_path: row.cfg_path.clone(),
            tla_path: row.tla_path.clone(),
            verdict,
            parse,
            symmetry,
            variant,
            measurement,
        }
    };

    let jobs = jobs.max(1).min(total.max(1));
    std::thread::scope(|scope| {
        for _ in 0..jobs {
            scope.spawn(|| loop {
                let idx = next.fetch_add(1, Ordering::Relaxed);
                if idx >= total {
                    break;
                }
                let report = probe(idx);
                results.lock().expect("doctor results mutex")[idx] = Some(report);
            });
        }
    });

    let rows: Vec<RowReport> = results
        .into_inner()
        .expect("doctor results mutex")
        .into_iter()
        .flatten()
        .collect();

    // ---- Manifest <-> baseline reconciliation. -----------------------------
    let manifest_names: BTreeSet<&str> = manifest.rows.iter().map(|r| r.name.as_str()).collect();
    let eligible_names: BTreeSet<&str> = eligible.iter().map(|r| r.name.as_str()).collect();
    let (missing_from_baseline, not_in_corpus, with_ty_runtime) = match &baseline_specs {
        Some(specs) => {
            let baseline_names: BTreeSet<&str> = specs.keys().map(|s| s.as_str()).collect();
            let missing: Vec<String> = eligible_names
                .difference(&baseline_names)
                .map(|s| (*s).to_string())
                .collect();
            let extra: Vec<String> = baseline_names
                .difference(&manifest_names)
                .map(|s| (*s).to_string())
                .collect();
            let measured = eligible_names
                .iter()
                .filter(|n| specs.get(**n).copied().unwrap_or(false))
                .count();
            (missing, extra, measured)
        }
        None => (Vec::new(), Vec::new(), 0),
    };

    let mut summary: BTreeMap<String, usize> = BTreeMap::new();
    for row in &rows {
        *summary.entry(row.verdict.label().to_string()).or_insert(0) += 1;
    }
    let blocking_rows = rows.iter().filter(|r| r.verdict.blocks_claim()).count();

    let report = DoctorReport {
        schema: DOCTOR_SCHEMA,
        toolchain: ToolchainReport {
            corpus_dir: corpus_dest.display().to_string(),
            corpus_cfg_count: super::count_cfgs(&specs_dir),
            manifest_path: manifest_path.display().to_string(),
            manifest_sha256: sha256_of(&manifest_path),
            baseline_path: baseline_path.display().to_string(),
            tlc_jar: tlc_jar.as_ref().map(|p| p.display().to_string()),
            tlc_jar_sha256: tlc_jar.as_deref().and_then(sha256_of),
            community_modules: community_modules.as_ref().map(|p| p.display().to_string()),
            community_modules_sha256: community_modules.as_deref().and_then(sha256_of),
            java: java_version(),
            tla_library: library.as_ref().map(|p| p.display().to_string()),
            tla_library_provenance: library_provenance.map(|p| p.as_str().to_string()),
            tla_library_strict_qualified: library_provenance
                .is_some_and(LibraryProvenance::is_strict_qualified),
            tlapm_pin: proof_library::TLAPM_PIN,
        },
        reconciliation: ReconciliationReport {
            eligible_rows: eligible.len(),
            excluded_rows: excluded.len(),
            baseline_entries: baseline_entry_count,
            eligible_missing_from_baseline: missing_from_baseline,
            baseline_entries_not_in_corpus: not_in_corpus.len(),
            baseline_entries_not_in_corpus_sample: not_in_corpus.into_iter().take(8).collect(),
            eligible_rows_with_ty_runtime: with_ty_runtime,
        },
        summary,
        rows,
        blocking_rows,
    };

    let rendered = match format {
        CorpusDoctorFormat::Table => render_table(&report),
        CorpusDoctorFormat::Json => serde_json::to_string_pretty(&report)? + "\n",
        CorpusDoctorFormat::Markdown => render_markdown(&report),
    };

    match out {
        Some(path) => {
            std::fs::write(path, &rendered)
                .with_context(|| format!("writing doctor report to {}", path.display()))?;
            println!("wrote {} ({} rows)", path.display(), report.rows.len());
        }
        None => print!("{rendered}"),
    }

    if mode == SupremacyMode::Enforce && report.blocking_rows > 0 {
        bail!(
            "corpus doctor: {} of {} eligible rows are not ready for strict collection \
             (see the report above; re-run with --mode warn to report without failing)",
            report.blocking_rows,
            report.rows.len()
        );
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Probes
// ---------------------------------------------------------------------------

/// Run SANY over one module with the harness's exact classpath and library.
fn probe_tlc_parse(
    tla: &Path,
    tlc_jar: Option<&Path>,
    community: Option<&Path>,
    library: Option<&Path>,
) -> ParseStatus {
    let Some(jar) = tlc_jar else {
        return ParseStatus::Unknown {
            reason: "no TLC jar (run `ty install-tlc install`)".into(),
        };
    };
    if !tla.is_file() {
        return ParseStatus::Unknown {
            reason: format!("module not found: {}", tla.display()),
        };
    }
    let mut classpath = jar.display().to_string();
    if let Some(cm) = community {
        classpath.push(':');
        classpath.push_str(&cm.display().to_string());
    }

    let mut cmd = Command::new("java");
    if let Some(lib) = library {
        cmd.arg(format!("-DTLA-Library={}", lib.display()));
    }
    cmd.arg("-cp").arg(&classpath).arg("tla2sany.SANY");
    // SANY resolves sibling modules relative to the working directory.
    if let Some(dir) = tla.parent() {
        cmd.current_dir(dir);
    }
    cmd.arg(tla.file_name().unwrap_or_default());

    let out = match cmd.output() {
        Ok(o) => o,
        Err(e) => {
            return ParseStatus::Unknown {
                reason: format!("could not run java: {e}"),
            }
        }
    };
    let combined = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    classify_sany_output(&combined)
}

/// Pure classifier for SANY output — unit-testable without a JDK.
fn classify_sany_output(combined: &str) -> ParseStatus {
    const MISSING: &str = "Cannot find source file for module ";
    if let Some(pos) = combined.find(MISSING) {
        let rest = &combined[pos + MISSING.len()..];
        let module: String = rest
            .chars()
            .take_while(|c| c.is_alphanumeric() || *c == '_')
            .collect();
        return ParseStatus::MissingModule { module };
    }
    if combined.contains("Fatal errors")
        || combined.contains("*** Errors")
        || combined.contains("Semantic errors")
    {
        let detail = combined
            .lines()
            .find(|l| {
                l.contains("Fatal errors") || l.contains("*** Errors") || l.contains("Semantic")
            })
            .unwrap_or("parse failed")
            .trim()
            .to_string();
        return ParseStatus::ParseError { detail };
    }
    ParseStatus::Ok
}

/// Ask the real checker what it does with this row's declared `SYMMETRY`.
///
/// Re-invokes this binary with `--max-states 1`: model preparation (where the
/// decision is made and announced) runs, exploration does not. Milliseconds.
fn probe_symmetry(tla: &Path, cfg: &Path, ty_bin: Option<&Path>) -> SymmetryDisposition {
    if !declares_symmetry(cfg) {
        return SymmetryDisposition::NotDeclared;
    }
    let Some(bin) = ty_bin else {
        return SymmetryDisposition::Unknown;
    };
    let out = Command::new(bin)
        .arg("check")
        .arg(tla)
        .arg("-c")
        .arg(cfg)
        .args(["--max-states", "1", "--force"])
        .output();
    let Ok(out) = out else {
        return SymmetryDisposition::Unknown;
    };
    let combined = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    classify_symmetry_output(&combined)
}

/// Pure classifier for the checker's symmetry announcement.
///
/// Matches the decisive runtime warning, NOT the informational soundness
/// deviation — the latter is emitted for every SYMMETRY+PROPERTY spec whether
/// or not symmetry was actually dropped, and confusing the two inverts the
/// answer.
fn classify_symmetry_output(combined: &str) -> SymmetryDisposition {
    if combined.contains("declared SYMMETRY is ignored during liveness checking") {
        SymmetryDisposition::DroppedForLiveness
    } else {
        SymmetryDisposition::RetainedSafety
    }
}

/// Whether a `.cfg` declares `SYMMETRY` (comments stripped).
fn declares_symmetry(cfg: &Path) -> bool {
    let Ok(text) = std::fs::read_to_string(cfg) else {
        return false;
    };
    text.lines()
        .map(|l| l.split("\\*").next().unwrap_or(""))
        .any(|l| l.trim_start().starts_with("SYMMETRY"))
}

// ---------------------------------------------------------------------------
// Toolchain resolution
// ---------------------------------------------------------------------------

/// Resolve the TLA library exactly as the comparison harness does, and classify
/// what resolved.
///
/// Order: explicit flag, `TLA_LIBRARY`, `TLA_PLUS_LIBRARY`, the installed
/// upstream proof library, then the repo's first-party stub set. The installed
/// upstream library is placed **ahead of** the repo stub so that installing it
/// is sufficient to make strict runs use upstream sources.
fn resolve_tla_library(
    explicit: Option<PathBuf>,
    repo_root: &Path,
) -> (Option<PathBuf>, Option<LibraryProvenance>) {
    let candidate = explicit
        .or_else(|| non_empty_env("TLA_LIBRARY"))
        .or_else(|| non_empty_env("TLA_PLUS_LIBRARY"))
        .or_else(|| default_if_dir(crate::cmd_tlc::default_proof_library()))
        .or_else(|| default_if_dir(repo_root.join("test_specs/tla_library")));
    match candidate {
        Some(dir) => {
            let provenance = proof_library::classify_library(&dir);
            (Some(dir), Some(provenance))
        }
        None => (None, None),
    }
}

fn non_empty_env(key: &str) -> Option<PathBuf> {
    std::env::var_os(key)
        .filter(|v| !v.is_empty())
        .map(PathBuf::from)
}

fn default_if_dir(p: PathBuf) -> Option<PathBuf> {
    p.is_dir().then_some(p)
}

fn default_if_file(p: PathBuf) -> Option<PathBuf> {
    p.is_file().then_some(p)
}

fn home() -> PathBuf {
    PathBuf::from(std::env::var("HOME").unwrap_or_else(|_| ".".to_string()))
}

fn repo_root() -> PathBuf {
    std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."))
}

fn java_version() -> Option<String> {
    let out = Command::new("java").arg("-version").output().ok()?;
    let text = String::from_utf8_lossy(&out.stderr);
    text.lines().next().map(|l| l.trim().to_string())
}

fn sha256_of(path: &Path) -> Option<String> {
    crate::cmd_tlc::sha256_hex(path).ok()
}

/// Map baseline spec name -> whether it carries a TY runtime.
///
/// Returns `None` when the baseline is unreadable, so "no baseline" and "no TY
/// runtime" stay distinguishable in the report.
fn load_baseline_specs(path: &Path) -> Option<BTreeMap<String, bool>> {
    let text = std::fs::read_to_string(path).ok()?;
    let value: Value = serde_json::from_str(&text).ok()?;
    let specs = value.get("specs")?.as_object()?;
    Some(
        specs
            .iter()
            .map(|(name, entry)| {
                let has_ty = entry
                    .get("ty")
                    .and_then(Value::as_object)
                    .is_some_and(|ty| {
                        ty.iter().any(|(k, v)| {
                            (k.contains("time") || k.contains("second")) && !v.is_null()
                        })
                    });
                (name.clone(), has_ty)
            })
            .collect(),
    )
}

// ---------------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------------

fn render_table(report: &DoctorReport) -> String {
    let mut s = String::new();
    let t = &report.toolchain;
    s.push_str("TY-vs-TLC comparability preflight\n\n");
    s.push_str(&format!(
        "  corpus            {} ({} .cfg)\n",
        t.corpus_dir, t.corpus_cfg_count
    ));
    s.push_str(&format!(
        "  manifest          {}{}\n",
        t.manifest_path,
        t.manifest_sha256
            .as_deref()
            .map(|h| format!("  sha256 {}", &h[..12]))
            .unwrap_or_default()
    ));
    s.push_str(&format!("  baseline          {}\n", t.baseline_path));
    s.push_str(&format!(
        "  TLC jar           {}\n",
        t.tlc_jar
            .as_deref()
            .unwrap_or("MISSING (ty install-tlc install)")
    ));
    s.push_str(&format!(
        "  CommunityModules  {}\n",
        t.community_modules.as_deref().unwrap_or("MISSING")
    ));
    s.push_str(&format!(
        "  java              {}\n",
        t.java.as_deref().unwrap_or("MISSING (TLC needs a JDK)")
    ));
    s.push_str(&format!(
        "  TLA library       {}\n",
        t.tla_library.as_deref().unwrap_or("none resolved")
    ));
    s.push_str(&format!(
        "  library origin    {}{}\n",
        t.tla_library_provenance.as_deref().unwrap_or("-"),
        if t.tla_library_strict_qualified {
            "  (strict-qualified: upstream, pinned)"
        } else {
            "  (NOT strict-qualified — run `ty install-tlc proof-library`)"
        }
    ));

    let r = &report.reconciliation;
    s.push_str("\nCorpus reconciliation\n\n");
    s.push_str(&format!(
        "  eligible rows                     {}\n  excluded rows                     {}\n",
        r.eligible_rows, r.excluded_rows
    ));
    s.push_str(&format!(
        "  baseline entries                  {}\n",
        r.baseline_entries
    ));
    s.push_str(&format!(
        "  eligible rows with a TY runtime   {} of {}\n",
        r.eligible_rows_with_ty_runtime, r.eligible_rows
    ));
    if !r.eligible_missing_from_baseline.is_empty() {
        s.push_str(&format!(
            "  eligible rows ABSENT from baseline {:?}\n",
            r.eligible_missing_from_baseline
        ));
    }
    if r.baseline_entries_not_in_corpus > 0 {
        s.push_str(&format!(
            "  baseline entries not in corpus    {} (e.g. {:?})\n",
            r.baseline_entries_not_in_corpus, r.baseline_entries_not_in_corpus_sample
        ));
    }

    s.push_str("\nPer-row verdicts\n\n");
    s.push_str(&format!(
        "  {:<30} {:<18} {:<22} {:<18} {:<9} {}\n",
        "row", "verdict", "tlc parse", "symmetry", "variant", "measurement"
    ));
    let mut rows: Vec<&RowReport> = report.rows.iter().collect();
    rows.sort_by(|a, b| a.verdict.cmp(&b.verdict).then(a.name.cmp(&b.name)));
    for row in rows {
        if row.verdict == RowVerdict::Ready {
            continue;
        }
        s.push_str(&format!(
            "  {:<30} {:<18} {:<22} {:<18} {:<9} {:?}\n",
            row.name,
            row.verdict.label(),
            row.parse.label(),
            row.symmetry.label(),
            row.variant.label(),
            row.measurement
        ));
    }
    let ready = report
        .rows
        .iter()
        .filter(|r| r.verdict == RowVerdict::Ready)
        .count();
    s.push_str(&format!("  ({ready} ready rows omitted)\n"));

    s.push_str("\nSummary\n\n");
    for (k, v) in &report.summary {
        s.push_str(&format!("  {k:<20} {v}\n"));
    }
    s.push_str(&format!(
        "\n  {} of {} eligible rows are not ready for strict collection.\n",
        report.blocking_rows,
        report.rows.len()
    ));
    s
}

fn render_markdown(report: &DoctorReport) -> String {
    let mut s = String::from("# TY-vs-TLC comparability preflight\n\n");
    let t = &report.toolchain;
    s.push_str("## Toolchain\n\n| input | value |\n|---|---|\n");
    s.push_str(&format!("| corpus | `{}` |\n", t.corpus_dir));
    s.push_str(&format!("| manifest | `{}` |\n", t.manifest_path));
    s.push_str(&format!(
        "| TLC jar | `{}` |\n",
        t.tlc_jar.as_deref().unwrap_or("MISSING")
    ));
    s.push_str(&format!(
        "| CommunityModules | `{}` |\n",
        t.community_modules.as_deref().unwrap_or("MISSING")
    ));
    s.push_str(&format!(
        "| TLA library | `{}` ({}) |\n",
        t.tla_library.as_deref().unwrap_or("none"),
        t.tla_library_provenance.as_deref().unwrap_or("-")
    ));
    s.push_str(&format!("| tlapm pin | `{}` |\n", t.tlapm_pin));

    s.push_str("\n## Summary\n\n| verdict | rows |\n|---|---:|\n");
    for (k, v) in &report.summary {
        s.push_str(&format!("| `{k}` | {v} |\n"));
    }

    s.push_str("\n## Rows needing work\n\n| row | verdict | tlc parse | symmetry | variant | measurement |\n|---|---|---|---|---|---|\n");
    let mut rows: Vec<&RowReport> = report
        .rows
        .iter()
        .filter(|r| r.verdict.blocks_claim())
        .collect();
    rows.sort_by(|a, b| a.verdict.cmp(&b.verdict).then(a.name.cmp(&b.name)));
    for row in rows {
        s.push_str(&format!(
            "| `{}` | {} | {} | {} | {} | {:?} |\n",
            row.name,
            row.verdict.label(),
            row.parse.label(),
            row.symmetry.label(),
            row.variant.label(),
            row.measurement
        ));
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_module_is_named_not_just_flagged() {
        let out =
            "Cannot find source file for module TLAPS imported in module Voting.\n*** Errors: 1";
        assert_eq!(
            classify_sany_output(out),
            ParseStatus::MissingModule {
                module: "TLAPS".into()
            }
        );
    }

    #[test]
    fn missing_module_wins_over_generic_error_text() {
        // The real SANY output contains BOTH; the actionable classification is
        // the module name, not "there were errors".
        let out = "Fatal errors while parsing TLA+ spec in file MCBakery.tla\n\
                   Cannot find source file for module TLAPS imported in module Bakery.";
        assert!(matches!(
            classify_sany_output(out),
            ParseStatus::MissingModule { .. }
        ));
    }

    #[test]
    fn clean_parse_is_ok() {
        assert_eq!(
            classify_sany_output("Parsing file Foo.tla\nSemantic processing of module Foo\n"),
            ParseStatus::Ok
        );
    }

    #[test]
    fn semantic_errors_are_parse_errors_not_missing_modules() {
        assert!(matches!(
            classify_sany_output("Semantic errors:\n*** Errors: 2"),
            ParseStatus::ParseError { .. }
        ));
    }

    #[test]
    fn symmetry_classifier_keys_on_the_decision_not_the_informational_note() {
        // The soundness provenance note is emitted for EVERY SYMMETRY+PROPERTY
        // spec, including ones where symmetry was retained. Keying on it would
        // report every such row as parity-impossible.
        let informational = "declared SYMMETRY is ignored when any PROPERTY requires genuine \
             liveness/temporal checking (the run is checked without symmetry — sound verdicts, \
             larger state space); pure-safety PROPERTY and INVARIANT checking still uses symmetry.";
        assert_eq!(
            classify_symmetry_output(informational),
            SymmetryDisposition::RetainedSafety
        );

        let decisive = "Warning: declared SYMMETRY is ignored during liveness checking (the orbit \
             quotient is unsound for temporal properties; ...)";
        assert_eq!(
            classify_symmetry_output(decisive),
            SymmetryDisposition::DroppedForLiveness
        );
    }

    #[test]
    fn parse_blocking_only_for_real_failures() {
        assert!(!ParseStatus::Ok.is_blocking());
        assert!(!ParseStatus::Skipped.is_blocking());
        assert!(!ParseStatus::Unknown { reason: "x".into() }.is_blocking());
        assert!(ParseStatus::MissingModule {
            module: "TLAPS".into()
        }
        .is_blocking());
    }

    #[test]
    fn only_ready_rows_do_not_block_the_claim() {
        assert!(!RowVerdict::Ready.blocks_claim());
        for v in [
            RowVerdict::Unmeasured,
            RowVerdict::TlcUnparseable,
            RowVerdict::ParityImpossible,
            RowVerdict::Indeterminate,
        ] {
            assert!(v.blocks_claim(), "{v:?} must block");
        }
    }

    #[test]
    fn symmetry_probe_short_circuits_when_no_symmetry_declared() {
        let dir = std::env::temp_dir().join("ty-doctor-symmetry-test");
        std::fs::create_dir_all(&dir).unwrap();
        let cfg = dir.join("NoSym.cfg");
        std::fs::write(&cfg, "SPECIFICATION Spec\nINVARIANT TypeOK\n").unwrap();
        assert!(!declares_symmetry(&cfg));
        // No binary needed: the declaration check runs first.
        assert_eq!(
            probe_symmetry(&dir.join("NoSym.tla"), &cfg, None),
            SymmetryDisposition::NotDeclared
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn commented_out_symmetry_does_not_count() {
        let dir = std::env::temp_dir().join("ty-doctor-symmetry-comment-test");
        std::fs::create_dir_all(&dir).unwrap();
        let cfg = dir.join("Commented.cfg");
        std::fs::write(&cfg, "\\* SYMMETRY Perms\nSPECIFICATION Spec\n").unwrap();
        assert!(!declares_symmetry(&cfg));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn variant_digest_must_match_the_manifest() {
        let dir = std::env::temp_dir().join("ty-doctor-variant-digest-test");
        std::fs::create_dir_all(&dir).unwrap();
        let v = dir.join("Row.nosym.cfg");
        std::fs::write(&v, "SPECIFICATION Spec\n").unwrap();
        let good = crate::cmd_tlc::sha256_hex(&v).unwrap();

        assert!(verify_variant_digest(&v, &good).is_ok());
        // An altered variant must be rejected, not silently used: it would
        // change what BOTH tools run and invalidate the parity claim.
        let err = verify_variant_digest(&v, &"0".repeat(64)).unwrap_err();
        assert!(err.contains("sha256"), "{err}");
        // A declared-but-missing variant is reported, never treated as absent.
        let missing = verify_variant_digest(&dir.join("Nope.cfg"), &good).unwrap_err();
        assert!(missing.contains("missing"), "{missing}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn checked_in_parity_variants_match_the_manifest_digests() {
        // Guards the real artifacts: if a variant is edited without updating
        // the manifest (or vice versa), the parity proof silently rots.
        let repo = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .and_then(Path::parent)
            .expect("repo root");
        let manifest_path = repo.join("tests/tlc_comparison/strict_corpus_manifest.json");
        let Ok(text) = std::fs::read_to_string(&manifest_path) else {
            return; // manifest not present in this checkout shape
        };
        let manifest: Manifest = serde_json::from_str(&text).expect("parse manifest");
        assert!(
            !manifest.eligibility.parity_variants.is_empty(),
            "expected declared parity variants"
        );
        for (cfg, v) in &manifest.eligibility.parity_variants {
            let path = repo.join(&v.variant_path);
            verify_variant_digest(&path, &v.sha256)
                .unwrap_or_else(|e| panic!("parity variant for {cfg}: {e}"));
            let body = std::fs::read_to_string(&path).unwrap();
            assert!(
                !body.lines().any(|l| l.trim_start().starts_with("SYMMETRY")),
                "parity variant for {cfg} must not declare SYMMETRY"
            );
        }
    }

    #[test]
    fn declared_symmetry_is_detected() {
        let dir = std::env::temp_dir().join("ty-doctor-symmetry-real-test");
        std::fs::create_dir_all(&dir).unwrap();
        let cfg = dir.join("Sym.cfg");
        std::fs::write(
            &cfg,
            "SPECIFICATION Spec\nSYMMETRY Perms\nPROPERTIES Term\n",
        )
        .unwrap();
        assert!(declares_symmetry(&cfg));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
