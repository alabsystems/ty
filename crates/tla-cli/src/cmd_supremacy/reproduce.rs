// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! `ty supremacy reproduce`: one-command, single-thread TY-vs-TLC(+Apalache)
//! runtime+memory reproduction with an HONEST scorecard.
//!
//! This module is CLI ORCHESTRATION ONLY. It does not implement model-checking,
//! comparison parsing, or verdict policy: it composes the pieces already shipped
//! in this binary.
//!
//!   * prerequisites — `ty corpus fetch` / `ty install-tlc install` / `ty install-apalache install`
//!     (each skipped when already present, or all skipped with `--no-install`);
//!   * TLC differential — the existing `super::compare::run` (single-thread
//!     TLC-vs-TY, wall-time + peak RSS, gated on verdict/state parity), whose
//!     `compare.json` is re-read for the scorecard;
//!   * Apalache differential — the cross-platform
//!     `scripts/ty_vs_apalache_memtime.sh` (bounded, symbolic, verdict-parity
//!     only), whose CSV is re-read for the scorecard.
//!
//! The scorecard reports the REAL numbers: TY does NOT win every spec, so wins
//! AND structural losses are both printed, with the comparability caveats
//! surfaced rather than hidden.

use std::collections::BTreeMap;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{bail, Context, Result};
use serde::Deserialize;

use crate::cli_schema::{
    ApalacheAction, CorpusAction, SupremacyCompareArgs, SupremacyCompareBackend,
    SupremacyComparePolicy, SupremacyCompareSpecSource, SupremacyMode, SupremacyOutputFormat,
    SupremacyReproduceArgs, SupremacyReproduceVs, TlcAction,
};

/// Small, fast, in-corpus demo set: all three are tiny, verified-match,
/// check-mode baseline rows (DiningPhilosophers 67 states, EWD840 302 states,
/// Prisoners 214 states), chosen so the default run finishes quickly.
const DEMO_SPECS: &[&str] = &["DiningPhilosophers", "EWD840", "Prisoners"];

const DEFAULT_BASELINE: &str = "tests/tlc_comparison/spec_baseline.json";
const APALACHE_SCRIPT: &str = "scripts/ty_vs_apalache_memtime.sh";

pub(super) fn run(args: SupremacyReproduceArgs) -> Result<()> {
    if args.workers == 0 {
        bail!("--workers must be >= 1");
    }
    if args.timeout == 0 {
        bail!("--timeout must be >= 1");
    }

    let specs: Vec<String> = if args.specs.is_empty() {
        DEMO_SPECS.iter().map(|s| (*s).to_string()).collect()
    } else {
        args.specs.clone()
    };

    let do_tlc = matches!(
        args.vs,
        SupremacyReproduceVs::Tlc | SupremacyReproduceVs::Both
    );
    let do_apalache = matches!(
        args.vs,
        SupremacyReproduceVs::Apalache | SupremacyReproduceVs::Both
    );

    let repo_root = env::current_dir().context("resolve current working directory")?;
    let output_dir = args
        .output_dir
        .clone()
        .unwrap_or_else(|| default_output_dir());
    fs::create_dir_all(&output_dir)
        .with_context(|| format!("create output dir {}", output_dir.display()))?;

    eprintln!("== ty supremacy reproduce ==");
    eprintln!(
        "specs: {}  |  vs: {}  |  workers: {}  |  output: {}",
        specs.join(", "),
        vs_label(args.vs),
        args.workers,
        output_dir.display()
    );

    ensure_prerequisites(args.no_install, do_tlc, do_apalache)?;

    let tlc_rows = if do_tlc {
        Some(run_tlc_compare(&args, &specs, &output_dir)?)
    } else {
        None
    };

    let apa_rows = if do_apalache {
        Some(run_apalache(&args, &specs, &repo_root, &output_dir)?)
    } else {
        None
    };

    print_scorecard(args.vs, args.len, tlc_rows.as_deref(), apa_rows.as_deref());
    Ok(())
}

fn vs_label(vs: SupremacyReproduceVs) -> &'static str {
    match vs {
        SupremacyReproduceVs::Tlc => "tlc",
        SupremacyReproduceVs::Apalache => "apalache",
        SupremacyReproduceVs::Both => "both (tlc + apalache)",
    }
}

fn default_output_dir() -> PathBuf {
    Path::new("reports").join("perf").join(format!(
        "{}-supremacy-reproduce",
        chrono::Utc::now().format("%Y%m%dT%H%M%SZ")
    ))
}

// --- prerequisites -----------------------------------------------------------

fn ensure_prerequisites(no_install: bool, do_tlc: bool, do_apalache: bool) -> Result<()> {
    eprintln!("-- prerequisites --");

    // Corpus is needed for every backend (specs are resolved from it).
    let (corpus_dir, cfg_count) = crate::cmd_corpus::probe_default();
    if cfg_count > 0 {
        eprintln!(
            "corpus: present ({} .cfg specs at {}) — skipping fetch",
            cfg_count,
            corpus_dir.display()
        );
    } else if no_install {
        bail!(
            "corpus not found at {} and --no-install was set; run `ty corpus fetch` first",
            corpus_dir.display()
        );
    } else {
        eprintln!("corpus: missing at {} — fetching", corpus_dir.display());
        crate::cmd_corpus::cmd_corpus(CorpusAction::Fetch {
            dest: None,
            from_upstream: false,
            force: false,
        })
        .context("auto-fetch corpus failed")?;
    }

    if do_tlc {
        let (tlc_jar, present) = crate::cmd_tlc::probe_default();
        if present {
            eprintln!("tlc: present ({}) — skipping install", tlc_jar.display());
        } else if no_install {
            bail!(
                "TLC jar not found at {} and --no-install was set; run `ty install-tlc install` first",
                tlc_jar.display()
            );
        } else {
            eprintln!("tlc: missing at {} — installing", tlc_jar.display());
            // Take the proof library too: 25 eligible corpus rows EXTEND TLAPS /
            // FiniteSetTheorems / NaturalsInduction and TLC cannot parse them
            // without it. A reproduce run that silently skipped those rows would
            // understate the corpus.
            crate::cmd_tlc::cmd_tlc(TlcAction::Install {
                dest: None,
                force: false,
                with_proof_library: true,
            })
            .context("auto-install TLC failed")?;
        }
    }

    if do_apalache {
        let (launcher, present) = crate::cmd_apalache::probe_default();
        if present {
            eprintln!(
                "apalache: present ({}) — skipping install",
                launcher.display()
            );
        } else if no_install {
            bail!(
                "Apalache not found at {} and --no-install was set; run `ty install-apalache install` first",
                launcher.display()
            );
        } else {
            eprintln!("apalache: missing at {} — installing", launcher.display());
            crate::cmd_apalache::cmd_apalache(ApalacheAction::Install {
                dest: None,
                force: false,
            })
            .context("auto-install Apalache failed")?;
        }
    }

    Ok(())
}

// --- TLC differential (reuse super::compare) ---------------------------------

/// A single comparable spec result, normalized for the honest scorecard. Both
/// the TLC and Apalache paths produce these so the scorecard renders uniformly.
#[derive(Clone, Debug)]
struct ScoreRow {
    spec: String,
    /// "ok" / verdict-parity status as reported by the underlying tool.
    comparable: bool,
    parity: String,
    ty_time_s: Option<f64>,
    tool_time_s: Option<f64>,
    ty_rss_b: Option<u64>,
    tool_rss_b: Option<u64>,
    note: String,
}

impl ScoreRow {
    /// TY time ratio (TY / tool). < 1.0 means TY is faster.
    fn time_ratio(&self) -> Option<f64> {
        ratio(self.ty_time_s, self.tool_time_s)
    }
    /// TY memory ratio (TY / tool). < 1.0 means TY uses less memory.
    fn mem_ratio(&self) -> Option<f64> {
        ratio(
            self.ty_rss_b.map(|b| b as f64),
            self.tool_rss_b.map(|b| b as f64),
        )
    }
    fn time_win(&self) -> Option<bool> {
        self.time_ratio().map(|r| r < 1.0)
    }
    fn mem_win(&self) -> Option<bool> {
        self.mem_ratio().map(|r| r < 1.0)
    }
}

fn ratio(num: Option<f64>, den: Option<f64>) -> Option<f64> {
    match (num, den) {
        (Some(n), Some(d)) if d > 0.0 && n.is_finite() && d.is_finite() => Some(n / d),
        _ => None,
    }
}

fn run_tlc_compare(
    args: &SupremacyReproduceArgs,
    specs: &[String],
    output_dir: &Path,
) -> Result<Vec<ScoreRow>> {
    eprintln!("-- TY vs TLC (single-thread, exhaustive; state-count + verdict parity) --");
    let compare_out = output_dir.join("tlc-compare");
    fs::create_dir_all(&compare_out)
        .with_context(|| format!("create {}", compare_out.display()))?;
    // The committed baseline pins `inputs.examples_dir` to the author's machine
    // (a /Users/... path). Materialize a machine-local copy pointing at the
    // resolved corpus so `compare::run` finds the specs anywhere.
    let baseline = localize_baseline(&compare_out)?;
    let compare_args = SupremacyCompareArgs {
        spec_source: SupremacyCompareSpecSource::Baseline,
        baseline,
        specs: specs.to_vec(),
        tla: None,
        config: None,
        // Interpreter is compare's documented default and reproduces TLC's exact
        // distinct-state counts on arbitrary corpus specs. (The strict
        // native-fused trust-cg launch env is tuned for the pinned launch corpus
        // and is not a fair general-purpose comparison engine here.)
        backend: SupremacyCompareBackend::Interpreter,
        workers: vec![args.workers],
        runs: 1,
        // warn: collect the numbers without failing the reproduce run on a
        // structural loss — the scorecard reports losses honestly instead.
        mode: SupremacyMode::Warn,
        policy: SupremacyComparePolicy::Parity,
        min_speedup: 1.05,
        max_memory_ratio: 0.95,
        output_dir: Some(compare_out.clone()),
        ty_bin: None,
        tlc_jar: None,
        tlc_bin: None,
        community_modules: None,
        tla_library: None,
        timeout: args.timeout,
        ty_flag: vec![],
        cases: vec![],
        ty_env: vec![],
        case_env: vec![],
        format: SupremacyOutputFormat::Human,
    };

    super::compare::run(compare_args).context("ty supremacy compare (vs TLC) failed")?;

    let report_path = compare_out.join("compare.json");
    let report = read_compare_report(&report_path)
        .with_context(|| format!("read {}", report_path.display()))?;

    Ok(report.rows.into_iter().map(score_tlc_row).collect())
}

/// Map a compare.json row to a scorecard row with an HONEST comparability call.
///
/// A row counts as comparable (its time/memory ratios feed the win tally) only
/// when compare's parity gate passes. That gate requires matching verdict,
/// distinct states, and exact raw initial/successor/total generated work.
/// Diagnostic/internal transition counters never promote a failed row.
fn score_tlc_row(row: CompareRowView) -> ScoreRow {
    let (comparable, parity, note) = if row.passed {
        (true, "ok".to_string(), row.reason)
    } else {
        (false, format!("FAIL ({})", row.class), row.reason)
    };

    ScoreRow {
        spec: row.spec,
        comparable,
        parity,
        ty_time_s: Some(row.backend_run.elapsed_seconds),
        tool_time_s: Some(row.tlc.elapsed_seconds),
        ty_rss_b: row.backend_run.peak_rss_bytes,
        tool_rss_b: row.tlc.peak_rss_bytes,
        note,
    }
}

/// Write a machine-local copy of the committed baseline JSON whose
/// `inputs.examples_dir` points at the resolved corpus `specifications/` dir,
/// and return its path. The committed baseline pins this to the author's macOS
/// path, which does not exist on other machines; rewriting only this one field
/// (leaving every recorded count/verdict untouched) lets `compare::run` resolve
/// the spec paths anywhere without changing its resolution logic.
fn localize_baseline(out_dir: &Path) -> Result<PathBuf> {
    let text = fs::read_to_string(DEFAULT_BASELINE)
        .with_context(|| format!("read baseline {DEFAULT_BASELINE}"))?;
    let mut value: serde_json::Value = serde_json::from_str(&text)
        .with_context(|| format!("parse baseline {DEFAULT_BASELINE}"))?;

    let (corpus_dir, _) = crate::cmd_corpus::probe_default();
    let corpus_specs = corpus_dir.join("specifications");
    let inputs = value
        .as_object_mut()
        .context("baseline JSON is not an object")?
        .entry("inputs")
        .or_insert_with(|| serde_json::json!({}));
    inputs
        .as_object_mut()
        .context("baseline inputs is not an object")?
        .insert(
            "examples_dir".to_string(),
            serde_json::Value::String(corpus_specs.display().to_string()),
        );

    let dest = out_dir.join("baseline.local.json");
    fs::write(&dest, serde_json::to_string_pretty(&value)? + "\n")
        .with_context(|| format!("write {}", dest.display()))?;
    Ok(dest)
}

/// Minimal re-read of the compare.json schema written by `super::compare`.
/// Later schemas retain the aggregate `tlc` and `backend_run` aliases, so the
/// one-pair reproduce scorecard remains compatible while the full paired
/// observations stay authoritative in compare.json.
#[derive(Debug, Deserialize)]
struct CompareReportView {
    schema: Option<String>,
    rows: Vec<CompareRowView>,
}

#[derive(Debug, Deserialize)]
struct CompareRowView {
    spec: String,
    passed: bool,
    #[serde(default)]
    class: String,
    #[serde(default)]
    reason: String,
    tlc: RunObservationView,
    backend_run: RunObservationView,
}

#[derive(Debug, Deserialize)]
struct RunObservationView {
    elapsed_seconds: f64,
    #[serde(default)]
    peak_rss_bytes: Option<u64>,
}

fn read_compare_report(path: &Path) -> Result<CompareReportView> {
    let text = fs::read_to_string(path)?;
    let report: CompareReportView = serde_json::from_str(&text)?;
    if !matches!(
        report.schema.as_deref(),
        Some(
            "ty.supremacy.compare.v1"
                | "ty.supremacy.compare.v2"
                | "ty.supremacy.compare.v3"
                | "ty.supremacy.compare.v4"
        )
    ) {
        bail!("unsupported supremacy compare schema {:?}", report.schema);
    }
    Ok(report)
}

// --- Apalache differential (shell out to the cross-platform script) ----------

fn run_apalache(
    args: &SupremacyReproduceArgs,
    specs: &[String],
    repo_root: &Path,
    output_dir: &Path,
) -> Result<Vec<ScoreRow>> {
    eprintln!(
        "-- TY vs Apalache (single-thread, bounded len={}, symbolic; VERDICT parity only) --",
        args.len
    );
    let script = repo_root.join(APALACHE_SCRIPT);
    if !script.is_file() {
        bail!(
            "Apalache differential script not found at {} (run from the ty repo root)",
            script.display()
        );
    }

    let (corpus_dir, _) = crate::cmd_corpus::probe_default();
    let corpus_specs = corpus_dir.join("specifications");
    let cfgs = resolve_corpus_cfgs(specs, &corpus_specs)?;

    let ty_exe = env::current_exe().context("resolve ty binary")?;
    let (apalache_launcher, _) = crate::cmd_apalache::probe_default();
    let out_csv = output_dir.join("apalache.csv");

    let timeout_bin = which_first(&["gtimeout", "timeout"]);
    let time_bin =
        first_existing(&["/usr/bin/time"]).unwrap_or_else(|| PathBuf::from("/usr/bin/time"));

    let mut cmd = Command::new("bash");
    cmd.arg(&script);
    for cfg in &cfgs {
        cmd.arg(cfg);
    }
    cmd.env("REPO", repo_root);
    cmd.env("TY", &ty_exe);
    cmd.env("APALACHE", &apalache_launcher);
    cmd.env("CORPUS", &corpus_specs);
    cmd.env("LEN", args.len.to_string());
    cmd.env("TIMEOUT", args.timeout.to_string());
    cmd.env("OUT_CSV", &out_csv);
    cmd.env("TIME_BIN", &time_bin);
    if let Some(tb) = &timeout_bin {
        cmd.env("TIMEOUT_BIN", tb);
    }

    let status = cmd
        .status()
        .with_context(|| format!("spawn {}", script.display()))?;
    if !status.success() {
        // The script's per-spec failures are recorded in the CSV (non-comparable
        // rows), so a non-zero exit is surfaced as a warning, not a hard error:
        // the scorecard still reports whatever rows were collected.
        eprintln!(
            "[reproduce] WARNING: Apalache differential script exited {status}; reporting collected rows"
        );
    }

    parse_apalache_csv(&out_csv)
        .with_context(|| format!("parse Apalache CSV {}", out_csv.display()))
}

/// Resolve `--spec` names to their corpus `.cfg` paths via the baseline JSON's
/// recorded source paths. This is spec selection (path resolution to feed the
/// script positionally), not verdict policy.
fn resolve_corpus_cfgs(specs: &[String], corpus_specs: &Path) -> Result<Vec<PathBuf>> {
    let text = fs::read_to_string(DEFAULT_BASELINE)
        .with_context(|| format!("read baseline {DEFAULT_BASELINE}"))?;
    let baseline: BaselineView = serde_json::from_str(&text)
        .with_context(|| format!("parse baseline {DEFAULT_BASELINE}"))?;

    let mut out = Vec::new();
    for name in specs {
        let entry = baseline
            .specs
            .get(name)
            .with_context(|| format!("baseline spec {name:?} not found in {DEFAULT_BASELINE}"))?;
        let source = entry
            .source
            .as_ref()
            .with_context(|| format!("baseline spec {name:?} has no source paths"))?;
        let cfg = corpus_specs.join(&source.cfg_path);
        if !cfg.is_file() {
            bail!("corpus cfg not found for spec {name:?}: {}", cfg.display());
        }
        out.push(cfg);
    }
    Ok(out)
}

#[derive(Debug, Deserialize)]
struct BaselineView {
    specs: BTreeMap<String, BaselineEntryView>,
}

#[derive(Debug, Deserialize)]
struct BaselineEntryView {
    source: Option<BaselineSourceView>,
}

#[derive(Debug, Deserialize)]
struct BaselineSourceView {
    cfg_path: PathBuf,
}

/// Parse the Apalache differential CSV. Columns (header row):
/// `spec,cfg,len,ty_verdict,apalache_verdict,verdict_match,ty_time_s,apa_time_s,
///  time_ratio,time_win,ty_rss_b,apa_rss_b,mem_ratio,mem_win,status,note`.
fn parse_apalache_csv(path: &Path) -> Result<Vec<ScoreRow>> {
    let text = fs::read_to_string(path)?;
    let mut rows = Vec::new();
    for (i, line) in text.lines().enumerate() {
        if i == 0 || line.trim().is_empty() {
            continue; // header / blank
        }
        let f: Vec<&str> = line.split(',').collect();
        if f.len() < 16 {
            continue;
        }
        let spec = f[0].to_string();
        let ty_verdict = f[3];
        let apa_verdict = f[4];
        let verdict_match = f[5];
        let status = f[14];
        let note = f[15].to_string();
        let comparable = status == "OK";
        let parity = if comparable {
            "ok".to_string()
        } else {
            format!("non-comparable: {status} (ty={ty_verdict} apa={apa_verdict})")
        };
        let _ = verdict_match;
        rows.push(ScoreRow {
            spec,
            comparable,
            parity,
            ty_time_s: parse_f64(f[6]),
            tool_time_s: parse_f64(f[7]),
            ty_rss_b: parse_u64(f[10]),
            tool_rss_b: parse_u64(f[11]),
            note,
        });
    }
    Ok(rows)
}

fn parse_f64(s: &str) -> Option<f64> {
    let s = s.trim();
    if s.is_empty() || s == "-" {
        None
    } else {
        s.parse().ok()
    }
}

fn parse_u64(s: &str) -> Option<u64> {
    let s = s.trim();
    if s.is_empty() || s == "-" {
        None
    } else {
        s.parse().ok()
    }
}

fn which_first(names: &[&str]) -> Option<PathBuf> {
    for name in names {
        if let Ok(out) = Command::new("sh")
            .arg("-c")
            .arg(format!("command -v {name}"))
            .output()
        {
            if out.status.success() {
                let p = String::from_utf8_lossy(&out.stdout).trim().to_string();
                if !p.is_empty() {
                    return Some(PathBuf::from(p));
                }
            }
        }
    }
    None
}

fn first_existing(paths: &[&str]) -> Option<PathBuf> {
    paths.iter().map(PathBuf::from).find(|p| p.exists())
}

// --- honest scorecard --------------------------------------------------------

fn print_scorecard(
    vs: SupremacyReproduceVs,
    len: usize,
    tlc_rows: Option<&[ScoreRow]>,
    apa_rows: Option<&[ScoreRow]>,
) {
    println!();
    println!("================ HONEST SCORECARD ================");
    println!("(time ratio = TY/tool, memory ratio = TY/tool; < 1.0 means TY wins)");

    if let Some(rows) = tlc_rows {
        print_section(
            "TY vs TLC (explicit-state, exhaustive)",
            rows,
            "Comparable rows require verdict + state-count parity. Non-comparable rows are excluded from win counts.",
        );
    }
    if let Some(rows) = apa_rows {
        print_section(
            &format!("TY vs Apalache (symbolic, bounded len={len})"),
            rows,
            "CAVEAT: Apalache is BOUNDED (len) and SYMBOLIC: verdict-parity only (no state count); an Apalache \"ok\" corroborates a TY \"ok\" only up to len, and many corpus specs are non-comparable because Apalache needs @type annotations / INIT-NEXT they lack.",
        );
    }

    let _ = vs;
    println!("==================================================");
}

fn print_section(title: &str, rows: &[ScoreRow], caveat: &str) {
    println!();
    println!("--- {title} ---");
    println!(
        "{:<26} {:<10} {:>10} {:>10} {:>9} {:>9}",
        "spec", "parity", "ty_time", "tool_time", "t_ratio", "m_ratio"
    );
    for row in rows {
        println!(
            "{:<26} {:<10} {:>10} {:>10} {:>9} {:>9}",
            truncate(&row.spec, 26),
            short_parity(&row.parity),
            fmt_secs(row.ty_time_s),
            fmt_secs(row.tool_time_s),
            fmt_ratio(row.time_ratio()),
            fmt_ratio(row.mem_ratio()),
        );
        if !row.comparable && !row.note.trim().is_empty() {
            println!("    note: {}", truncate(row.note.trim(), 120));
        }
    }

    let comparable: Vec<&ScoreRow> = rows.iter().filter(|r| r.comparable).collect();
    let total = rows.len();
    let comp = comparable.len();
    let time_wins = comparable
        .iter()
        .filter(|r| r.time_win() == Some(true))
        .count();
    let mem_wins = comparable
        .iter()
        .filter(|r| r.mem_win() == Some(true))
        .count();
    let time_losses: Vec<&str> = comparable
        .iter()
        .filter(|r| r.time_win() == Some(false))
        .map(|r| r.spec.as_str())
        .collect();
    let mem_losses: Vec<&str> = comparable
        .iter()
        .filter(|r| r.mem_win() == Some(false))
        .map(|r| r.spec.as_str())
        .collect();
    let non_comparable: Vec<&str> = rows
        .iter()
        .filter(|r| !r.comparable)
        .map(|r| r.spec.as_str())
        .collect();

    println!();
    println!(
        "summary: TY faster on {time_wins}/{comp}, less memory on {mem_wins}/{comp} (comparable specs only; {comp}/{total} comparable)."
    );
    if time_losses.is_empty() && mem_losses.is_empty() {
        println!("losses: none on the comparable set.");
    } else {
        if !time_losses.is_empty() {
            println!("time losses (TY slower): {}", time_losses.join(", "));
        }
        if !mem_losses.is_empty() {
            println!("memory losses (TY heavier): {}", mem_losses.join(", "));
        }
    }
    if !non_comparable.is_empty() {
        println!(
            "non-comparable (excluded from win counts): {}",
            non_comparable.join(", ")
        );
    }
    println!("{caveat}");
}

fn short_parity(parity: &str) -> String {
    truncate(parity, 10)
}

fn truncate(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        s.to_string()
    } else {
        let mut out: String = s.chars().take(max.saturating_sub(1)).collect();
        out.push('…');
        out
    }
}

fn fmt_secs(v: Option<f64>) -> String {
    v.map(|s| format!("{s:.3}s"))
        .unwrap_or_else(|| "n/a".to_string())
}

fn fmt_ratio(v: Option<f64>) -> String {
    v.map(|r| format!("{r:.3}x"))
        .unwrap_or_else(|| "n/a".to_string())
}
