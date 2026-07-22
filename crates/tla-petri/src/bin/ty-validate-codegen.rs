// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Rust port of `scripts/validate_codegen.py`.
//!
//! Validate TY codegen coverage against the full spec baseline suite.
//! Reads specs from `tests/tlc_comparison/spec_baseline.json`, runs
//! `ty codegen` (both AST and TIR paths) on each spec, and reports
//! coverage metrics including:
//!
//! * `codegen_ok` -- code generation succeeded (Rust source emitted)
//! * `codegen_error` -- code generation failed (parse/lower/emit error)
//! * `compile_ok` -- generated Rust compiles with `rustc --edition 2021`
//! * `compile_error` -- generated Rust fails to compile
//! * Error categorization by root cause
//!
//! Replaces the Python script with a compiler-enforced Rust CLI so the
//! ty codegen-coverage harness has a single binary entry point.

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use anyhow::{anyhow, Context, Result};
use clap::{Parser, ValueEnum};
use serde::{Deserialize, Serialize};
use serde_json::Value;

const CODEGEN_TIMEOUT: Duration = Duration::from_secs(30);
const COMPILE_TIMEOUT: Duration = Duration::from_secs(30);

#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
enum Category {
    Small,
    Medium,
    Large,
    Xlarge,
}

impl Category {
    fn as_baseline_str(self) -> &'static str {
        match self {
            Category::Small => "small",
            Category::Medium => "medium",
            Category::Large => "large",
            Category::Xlarge => "xlarge",
        }
    }
}

#[derive(Parser, Debug)]
#[command(
    name = "ty-validate-codegen",
    about = "Validate TY codegen against spec baseline suite.",
    long_about = "Rust port of scripts/validate_codegen.py. Runs `ty codegen` \
                  (AST path, and optionally TIR path) on every spec in \
                  tests/tlc_comparison/spec_baseline.json whose source files \
                  resolve under ~/tlaplus-examples/specifications, captures \
                  per-spec results and an optional `rustc` compile check, \
                  classifies errors, and writes a Markdown coverage report \
                  (defaults to reports/codegen-coverage.md). Exit 0 always; \
                  non-zero only on hard usage errors (binary missing, etc)."
)]
struct Cli {
    /// Path to the `ty` binary used for codegen.
    #[arg(
        long,
        value_name = "PATH",
        default_value = "/tmp/ty-codegen-r4/release/ty"
    )]
    binary: PathBuf,

    /// Filter specs by baseline category.
    #[arg(long, value_enum, value_name = "CAT")]
    category: Option<Category>,

    /// Maximum number of specs to test.
    #[arg(long, value_name = "N")]
    limit: Option<usize>,

    /// Test specific spec(s) by name (repeatable).
    #[arg(long = "spec", value_name = "NAME")]
    specs: Vec<String>,

    /// Also check if generated Rust compiles with `rustc`.
    #[arg(long)]
    compile: bool,

    /// Also test TIR-based codegen path.
    #[arg(long)]
    tir: bool,

    /// Output report path (Markdown).
    #[arg(long, value_name = "PATH")]
    output: Option<PathBuf>,

    /// Also write JSON results alongside the Markdown report.
    #[arg(long)]
    json: bool,

    /// Optional override for the repository root. Defaults to the parent of
    /// this binary's `CARGO_MANIFEST_DIR/../..` so the binary works whether
    /// it was launched from the workspace root or via Cargo.
    #[arg(long = "repo-root", value_name = "PATH")]
    repo_root: Option<PathBuf>,

    /// Override the directory holding the resolved spec sources.
    /// Defaults to `~/tlaplus-examples/specifications`.
    #[arg(long = "examples-dir", value_name = "PATH")]
    examples_dir: Option<PathBuf>,
}

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
struct SpecResult {
    name: String,
    category: String,
    tla_path: String,
    cfg_path: String,
    tlc_states: Option<i64>,
    tlc_status: String,
    codegen_ok: bool,
    codegen_error: Option<String>,
    codegen_elapsed: f64,
    generated_lines: usize,
    compile_ok: Option<bool>,
    compile_error: Option<String>,
    tir_codegen_ok: Option<bool>,
    tir_codegen_error: Option<String>,
    tir_generated_lines: usize,
    meaningful: bool,
    tir_meaningful: bool,
    error_category: Option<String>,
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    match run(&cli) {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("ERROR: {err:#}");
            ExitCode::from(1)
        }
    }
}

fn run(cli: &Cli) -> Result<()> {
    let repo_root = resolve_repo_root(cli.repo_root.as_deref())?;
    let examples_dir = cli
        .examples_dir
        .clone()
        .unwrap_or_else(default_examples_dir);
    let baseline_path = repo_root
        .join("tests")
        .join("tlc_comparison")
        .join("spec_baseline.json");
    let output_path = cli
        .output
        .clone()
        .unwrap_or_else(|| repo_root.join("reports").join("codegen-coverage.md"));

    if !cli.binary.exists() {
        eprintln!("ERROR: binary not found at {}", cli.binary.display());
        eprintln!(
            "Build with: RUSTUP_TOOLCHAIN=stable-aarch64-apple-darwin \
             $(rustup which cargo --toolchain stable) build --release \
             --bin ty --target-dir /tmp/ty-codegen-r4"
        );
        return Err(anyhow!("binary missing"));
    }

    let baseline_text = fs::read_to_string(&baseline_path)
        .with_context(|| format!("reading {}", baseline_path.display()))?;
    let baseline: Value = serde_json::from_str(&baseline_text)
        .with_context(|| format!("parsing {}", baseline_path.display()))?;

    let specs_map = baseline
        .get("specs")
        .and_then(|s| s.as_object())
        .ok_or_else(|| anyhow!("baseline missing .specs object"))?;

    let selected = select_specs(specs_map, &examples_dir, cli)?;
    if selected.is_empty() {
        return Err(anyhow!("no specs selected for testing"));
    }

    println!(
        "Testing {} specs with binary: {}",
        selected.len(),
        cli.binary.display()
    );
    println!("Options: compile={}, tir={}", cli.compile, cli.tir);
    println!();

    let total = selected.len();
    let mut results: Vec<SpecResult> = Vec::with_capacity(total);
    for (idx, (name, spec)) in selected.iter().enumerate() {
        let category = spec
            .get("category")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown")
            .to_string();
        let source = spec.get("source").and_then(|v| v.as_object());
        let tla_rel = source
            .and_then(|s| s.get("tla_path"))
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let cfg_rel = source
            .and_then(|s| s.get("cfg_path"))
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let tla_path = examples_dir.join(&tla_rel);
        let cfg_path: Option<PathBuf> = if cfg_rel.is_empty() {
            None
        } else {
            Some(examples_dir.join(&cfg_rel))
        };

        let tlc_states = spec
            .get("tlc")
            .and_then(|v| v.get("states"))
            .and_then(|v| v.as_i64());
        let tlc_status = spec
            .get("tlc")
            .and_then(|v| v.get("status"))
            .and_then(|v| v.as_str())
            .unwrap_or("unknown")
            .to_string();

        let mut sr = SpecResult {
            name: name.clone(),
            category: category.clone(),
            tla_path: tla_rel.clone(),
            cfg_path: cfg_rel.clone(),
            tlc_states,
            tlc_status,
            ..Default::default()
        };

        print!("[{}/{}] {} ({})... ", idx + 1, total, name, category);
        // flush
        use std::io::Write as _;
        let _ = std::io::stdout().flush();

        let (ok, output, elapsed, lines) =
            run_codegen(&cli.binary, &tla_path, cfg_path.as_deref(), false);
        sr.codegen_ok = ok;
        sr.codegen_elapsed = elapsed;
        sr.generated_lines = lines;

        if !ok {
            sr.codegen_error = Some(output.clone());
            sr.error_category = Some(classify_error(&output));
            println!(
                "CODEGEN_ERROR ({}) {:.1}s",
                sr.error_category.as_deref().unwrap_or("?"),
                elapsed
            );
        } else {
            sr.meaningful = is_meaningful_codegen(&output);
            let tag = if sr.meaningful { "meaningful" } else { "stub" };
            print!("OK ({} lines, {}, {:.1}s)", lines, tag, elapsed);

            if cli.compile {
                let (compile_ok, compile_err) = run_compile_check(&output);
                sr.compile_ok = Some(compile_ok);
                if !compile_ok {
                    sr.compile_error = Some(compile_err);
                    print!(" COMPILE_ERROR");
                } else {
                    print!(" COMPILES");
                }
            }
            println!();
        }

        if cli.tir {
            let (tir_ok, tir_output, _, tir_lines) =
                run_codegen(&cli.binary, &tla_path, cfg_path.as_deref(), true);
            sr.tir_codegen_ok = Some(tir_ok);
            sr.tir_generated_lines = tir_lines;
            if tir_ok {
                sr.tir_meaningful = is_meaningful_codegen(&tir_output);
            } else {
                sr.tir_codegen_error = Some(tir_output);
            }
        }

        results.push(sr);
    }

    print_summary(&results, cli);

    let report = generate_report(&results, &cli.binary, cli.compile, cli.tir);
    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent).with_context(|| format!("creating {}", parent.display()))?;
    }
    fs::write(&output_path, report)
        .with_context(|| format!("writing {}", output_path.display()))?;
    println!("\nReport written to: {}", output_path.display());

    if cli.json {
        let json_path = output_path.with_extension("json");
        let payload = build_json_report(&results, &cli.binary);
        fs::write(
            &json_path,
            serde_json::to_string_pretty(&payload).context("json serialize")?,
        )
        .with_context(|| format!("writing {}", json_path.display()))?;
        println!("JSON written to: {}", json_path.display());
    }

    Ok(())
}

fn select_specs<'a>(
    specs: &'a serde_json::Map<String, Value>,
    examples_dir: &Path,
    cli: &Cli,
) -> Result<Vec<(String, &'a Value)>> {
    let want_names: Option<Vec<&str>> = if cli.specs.is_empty() {
        None
    } else {
        Some(cli.specs.iter().map(String::as_str).collect())
    };
    let want_category = cli.category.map(Category::as_baseline_str);

    let mut keys: Vec<&String> = specs.keys().collect();
    keys.sort();

    let mut out: Vec<(String, &Value)> = Vec::new();
    for name in keys {
        if let Some(wanted) = want_names.as_ref() {
            if !wanted.contains(&name.as_str()) {
                continue;
            }
        }
        let spec = &specs[name];
        if let Some(cat) = want_category {
            let spec_cat = spec.get("category").and_then(|v| v.as_str()).unwrap_or("");
            if spec_cat != cat {
                continue;
            }
        }
        let tla_rel = spec
            .get("source")
            .and_then(|s| s.get("tla_path"))
            .and_then(|v| v.as_str())
            .unwrap_or("");
        if tla_rel.is_empty() {
            continue;
        }
        let tla_path = examples_dir.join(tla_rel);
        if !tla_path.exists() {
            continue;
        }
        out.push((name.clone(), spec));
    }
    if let Some(limit) = cli.limit {
        out.truncate(limit);
    }
    Ok(out)
}

fn run_codegen(
    binary: &Path,
    tla_path: &Path,
    cfg_path: Option<&Path>,
    tir: bool,
) -> (bool, String, f64, usize) {
    let mut cmd = Command::new(binary);
    cmd.arg("codegen").arg(tla_path);
    if tir {
        cmd.arg("--tir");
    }
    if let Some(cfg) = cfg_path {
        if cfg.exists() {
            cmd.arg("--config").arg(cfg);
        }
    }
    if let Some(parent) = tla_path.parent() {
        cmd.current_dir(parent);
    }
    let start = Instant::now();
    match run_with_timeout(cmd, CODEGEN_TIMEOUT) {
        Ok(output) => {
            let elapsed = start.elapsed().as_secs_f64();
            if output.status.success() {
                let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
                let lines = stdout.matches('\n').count();
                (true, stdout, elapsed, lines)
            } else {
                let stderr = String::from_utf8_lossy(&output.stderr);
                let stdout = String::from_utf8_lossy(&output.stdout);
                let mut error = stderr.trim().to_string();
                if error.is_empty() {
                    error = stdout.trim().to_string();
                }
                if error.len() > 500 {
                    error.truncate(500);
                    error.push_str("...");
                }
                (false, error, elapsed, 0)
            }
        }
        Err(err) => {
            let elapsed = start.elapsed().as_secs_f64();
            (
                false,
                if err.is_timeout() {
                    format!("timeout after {}s", CODEGEN_TIMEOUT.as_secs())
                } else {
                    err.message
                },
                elapsed,
                0,
            )
        }
    }
}

fn run_compile_check(rust_code: &str) -> (bool, String) {
    let mut tmp = match tempfile::Builder::new().suffix(".rs").tempfile() {
        Ok(t) => t,
        Err(err) => return (false, format!("tempfile: {err}")),
    };
    use std::io::Write as _;
    if let Err(err) = tmp.write_all(rust_code.as_bytes()) {
        return (false, format!("write: {err}"));
    }
    if let Err(err) = tmp.flush() {
        return (false, format!("flush: {err}"));
    }
    let tmp_path = tmp.path().to_path_buf();
    let mut cmd = Command::new("rustc");
    cmd.args([
        "--edition",
        "2021",
        "--crate-type",
        "lib",
        "-o",
        "/dev/null",
    ])
    .arg(&tmp_path);
    match run_with_timeout(cmd, COMPILE_TIMEOUT) {
        Ok(output) => {
            if output.status.success() {
                (true, String::new())
            } else {
                let mut error = String::from_utf8_lossy(&output.stderr).trim().to_string();
                if error.len() > 500 {
                    error.truncate(500);
                    error.push_str("...");
                }
                (false, error)
            }
        }
        Err(err) => {
            if err.is_timeout() {
                (false, "compile timeout".to_string())
            } else {
                (false, err.message)
            }
        }
    }
}

struct RunErr {
    message: String,
    timed_out: bool,
}

impl RunErr {
    fn is_timeout(&self) -> bool {
        self.timed_out
    }
}

/// Run a command with a wall-clock timeout. Kills the child on timeout.
fn run_with_timeout(mut cmd: Command, timeout: Duration) -> Result<std::process::Output, RunErr> {
    use std::io::Read as _;
    use std::process::Stdio;

    cmd.stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    let mut child = cmd.spawn().map_err(|err| RunErr {
        message: err.to_string(),
        timed_out: false,
    })?;

    let start = Instant::now();
    loop {
        match child.try_wait() {
            Ok(Some(status)) => {
                let mut stdout = Vec::new();
                let mut stderr = Vec::new();
                if let Some(mut s) = child.stdout.take() {
                    let _ = s.read_to_end(&mut stdout);
                }
                if let Some(mut s) = child.stderr.take() {
                    let _ = s.read_to_end(&mut stderr);
                }
                return Ok(std::process::Output {
                    status,
                    stdout,
                    stderr,
                });
            }
            Ok(None) => {}
            Err(err) => {
                return Err(RunErr {
                    message: err.to_string(),
                    timed_out: false,
                });
            }
        }
        if start.elapsed() >= timeout {
            let _ = child.kill();
            let _ = child.wait();
            return Err(RunErr {
                message: format!("timeout after {}s", timeout.as_secs()),
                timed_out: true,
            });
        }
        std::thread::sleep(Duration::from_millis(50));
    }
}

fn classify_error(error_text: &str) -> String {
    let lower = error_text.to_ascii_lowercase();
    let contains = |needle: &str| lower.contains(needle);

    if contains("parse") || contains("syntax") || contains("unexpected token") {
        return "parse_error".to_string();
    }
    if contains("lower failed") || contains("lower produced no module") {
        return "lower_error".to_string();
    }
    if contains("tir lowering failed") {
        return "tir_lower_error".to_string();
    }
    if contains("not yet implemented") || contains("todo") || contains("unimplemented") {
        return "unimplemented".to_string();
    }
    if contains("extends") || contains("module not found") || contains("load extends") {
        return "module_not_found".to_string();
    }
    if contains("instance") || contains("load instance") {
        return "instance_error".to_string();
    }
    if contains("unknown identifier") {
        return "unknown_identifier".to_string();
    }
    if contains("unsupported for codegen") {
        return "unsupported_feature".to_string();
    }
    if contains("unsupported") {
        return "unsupported_feature".to_string();
    }
    if contains("timeout") || contains("timed out") {
        return "timeout".to_string();
    }
    if contains("panic") || contains("stack overflow") || contains("thread") {
        return "crash".to_string();
    }
    if contains("no such file") || contains("os error 2") {
        return "file_not_found".to_string();
    }
    if contains("type inference error") {
        return "type_inference".to_string();
    }
    if contains("type") && (contains("mismatch") || contains("error")) {
        return "type_error".to_string();
    }
    "other".to_string()
}

fn is_meaningful_codegen(rust_code: &str) -> bool {
    let mut has_init = false;
    let mut has_next = false;
    for line in rust_code.lines() {
        let stripped = line.trim();
        if stripped.contains("fn init(") && !rust_code.contains("No Init operator found") {
            has_init = true;
        }
        if stripped.contains("fn next(") && !rust_code.contains("No Next operator found") {
            has_next = true;
        }
    }
    has_init && has_next
}

fn print_summary(results: &[SpecResult], cli: &Cli) {
    let total = results.len();
    let codegen_ok = results.iter().filter(|r| r.codegen_ok).count();
    let meaningful = results.iter().filter(|r| r.meaningful).count();
    println!();
    println!("=== CODEGEN COVERAGE SUMMARY (AST path) ===");
    println!("Total:        {}", total);
    println!("Codegen OK:   {} ({}%)", codegen_ok, pct(codegen_ok, total));
    println!("  Meaningful: {} ({}%)", meaningful, pct(meaningful, total));
    let stubs = codegen_ok.saturating_sub(meaningful);
    println!("  Stub only:  {} ({}%)", stubs, pct(stubs, total));
    let err = total - codegen_ok;
    println!("Codegen ERR:  {} ({}%)", err, pct(err, total));

    if cli.compile {
        let tested: Vec<&SpecResult> = results.iter().filter(|r| r.compile_ok.is_some()).collect();
        if !tested.is_empty() {
            let ok = tested.iter().filter(|r| r.compile_ok == Some(true)).count();
            println!(
                "Compiles OK:  {}/{} ({}%)",
                ok,
                tested.len(),
                pct(ok, tested.len())
            );
        }
    }

    if cli.tir {
        let tested: Vec<&SpecResult> = results
            .iter()
            .filter(|r| r.tir_codegen_ok.is_some())
            .collect();
        if !tested.is_empty() {
            let ok = tested
                .iter()
                .filter(|r| r.tir_codegen_ok == Some(true))
                .count();
            let meaningful = tested.iter().filter(|r| r.tir_meaningful).count();
            println!();
            println!("=== TIR PATH ===");
            println!(
                "TIR OK:       {}/{} ({}%)",
                ok,
                tested.len(),
                pct(ok, tested.len())
            );
            println!(
                "  Meaningful: {} ({}%)",
                meaningful,
                pct(meaningful, tested.len())
            );
            let stubs = ok.saturating_sub(meaningful);
            println!("  Stub only:  {} ({}%)", stubs, pct(stubs, tested.len()));
        }
    }

    let mut counts: BTreeMap<String, usize> = BTreeMap::new();
    for r in results.iter().filter(|r| !r.codegen_ok) {
        let cat = r.error_category.clone().unwrap_or_else(|| "?".to_string());
        *counts.entry(cat).or_insert(0) += 1;
    }
    if !counts.is_empty() {
        println!();
        println!("Error breakdown:");
        let mut sorted: Vec<(String, usize)> = counts.into_iter().collect();
        sorted.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
        for (cat, count) in sorted {
            println!("  {}: {}", cat, count);
        }
    }
}

fn pct(num: usize, den: usize) -> u64 {
    if den == 0 {
        0
    } else {
        (num as f64 * 100.0 / den as f64).round() as u64
    }
}

fn generate_report(
    results: &[SpecResult],
    binary: &Path,
    do_compile: bool,
    do_tir: bool,
) -> String {
    let mut out = String::new();
    out.push_str("# Codegen Coverage Report\n\n");
    let _ = writeln!(out, "**Date:** {}", local_timestamp());
    let _ = writeln!(out, "**Binary:** `{}`", binary.display());
    let _ = writeln!(out, "**Specs tested:** {}", results.len());
    let _ = writeln!(
        out,
        "**Compile check:** {}",
        if do_compile { "yes" } else { "no" }
    );
    let _ = writeln!(out, "**TIR path:** {}\n", if do_tir { "yes" } else { "no" });

    let total = results.len();
    let codegen_ok = results.iter().filter(|r| r.codegen_ok).count();
    let codegen_err = total - codegen_ok;
    let meaningful = results.iter().filter(|r| r.meaningful).count();

    out.push_str("## Summary (AST Path)\n\n");
    out.push_str("| Metric | Count | Pct |\n");
    out.push_str("|--------|------:|----:|\n");
    let p = |n| pct(n, total);
    let _ = writeln!(out, "| Codegen OK | {} | {}% |", codegen_ok, p(codegen_ok));
    let _ = writeln!(
        out,
        "| Meaningful (Init+Next) | {} | {}% |",
        meaningful,
        p(meaningful)
    );
    let stubs = codegen_ok.saturating_sub(meaningful);
    let _ = writeln!(out, "| Stub Only | {} | {}% |", stubs, p(stubs));
    let _ = writeln!(
        out,
        "| Codegen Error | {} | {}% |",
        codegen_err,
        p(codegen_err)
    );

    if do_compile {
        let tested: Vec<&SpecResult> = results.iter().filter(|r| r.compile_ok.is_some()).collect();
        if !tested.is_empty() {
            let ok = tested.iter().filter(|r| r.compile_ok == Some(true)).count();
            let er = tested.len() - ok;
            let _ = writeln!(out, "| Compiles OK | {} | {}% |", ok, pct(ok, tested.len()));
            let _ = writeln!(
                out,
                "| Compile Error | {} | {}% |",
                er,
                pct(er, tested.len())
            );
        }
    }

    if do_tir {
        let tested: Vec<&SpecResult> = results
            .iter()
            .filter(|r| r.tir_codegen_ok.is_some())
            .collect();
        if !tested.is_empty() {
            let ok = tested
                .iter()
                .filter(|r| r.tir_codegen_ok == Some(true))
                .count();
            let er = tested.len() - ok;
            let mf = tested.iter().filter(|r| r.tir_meaningful).count();
            out.push_str("\n**TIR Path:**\n\n");
            out.push_str("| Metric | Count | Pct |\n");
            out.push_str("|--------|------:|----:|\n");
            let _ = writeln!(
                out,
                "| TIR Codegen OK | {} | {}% |",
                ok,
                pct(ok, tested.len())
            );
            let _ = writeln!(
                out,
                "| TIR Meaningful (Init+Next) | {} | {}% |",
                mf,
                pct(mf, tested.len())
            );
            let stub = ok.saturating_sub(mf);
            let _ = writeln!(
                out,
                "| TIR Stub Only | {} | {}% |",
                stub,
                pct(stub, tested.len())
            );
            let _ = writeln!(
                out,
                "| TIR Codegen Error | {} | {}% |",
                er,
                pct(er, tested.len())
            );
        }
    }
    out.push('\n');

    // By category
    out.push_str("## Coverage by Category\n\n");
    out.push_str("| Category | Total | Codegen OK | Pct |\n");
    out.push_str("|----------|------:|-----------:|----:|\n");
    let mut by_cat: BTreeMap<&str, Vec<&SpecResult>> = BTreeMap::new();
    for r in results {
        by_cat.entry(r.category.as_str()).or_default().push(r);
    }
    for cat in ["small", "medium", "large", "xlarge", "unknown"] {
        if let Some(cat_results) = by_cat.get(cat) {
            let cat_ok = cat_results.iter().filter(|r| r.codegen_ok).count();
            let _ = writeln!(
                out,
                "| {} | {} | {} | {}% |",
                cat,
                cat_results.len(),
                cat_ok,
                pct(cat_ok, cat_results.len())
            );
        }
    }
    out.push('\n');

    // Error categories
    let errors: Vec<&SpecResult> = results.iter().filter(|r| !r.codegen_ok).collect();
    if !errors.is_empty() {
        out.push_str("## Error Categories\n\n");
        let mut counts: BTreeMap<String, usize> = BTreeMap::new();
        for r in &errors {
            let cat = r.error_category.clone().unwrap_or_else(|| "?".to_string());
            *counts.entry(cat).or_insert(0) += 1;
        }
        out.push_str("| Error Category | Count |\n");
        out.push_str("|----------------|------:|\n");
        let mut sorted: Vec<(String, usize)> = counts.into_iter().collect();
        sorted.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
        for (cat, count) in sorted {
            let _ = writeln!(out, "| {} | {} |", cat, count);
        }
        out.push('\n');
    }

    // Detailed results
    out.push_str("## Detailed Results\n\n");
    let mut header = "| Spec | Category | TLC States | Codegen | Lines |".to_string();
    let mut sep = "|------|----------|-----------|---------|------:|".to_string();
    if do_compile {
        header.push_str(" Compile |");
        sep.push_str("---------|");
    }
    if do_tir {
        header.push_str(" TIR |");
        sep.push_str("-----|");
    }
    header.push_str(" Error |");
    sep.push_str("-------|");
    out.push_str(&header);
    out.push('\n');
    out.push_str(&sep);
    out.push('\n');

    let mut sorted: Vec<&SpecResult> = results.iter().collect();
    sorted.sort_by(|a, b| {
        (!a.codegen_ok)
            .cmp(&!b.codegen_ok)
            .then(a.name.cmp(&b.name))
    });
    for r in sorted {
        let states_str = r
            .tlc_states
            .map(|s| s.to_string())
            .unwrap_or_else(|| "-".to_string());
        let codegen_str = if r.codegen_ok { "OK" } else { "FAIL" };
        let lines_str = if r.codegen_ok {
            r.generated_lines.to_string()
        } else {
            "-".to_string()
        };
        let error_str = r.error_category.clone().unwrap_or_default();
        let mut row = format!(
            "| {} | {} | {} | {} | {} |",
            r.name, r.category, states_str, codegen_str, lines_str
        );
        if do_compile {
            row.push_str(match r.compile_ok {
                Some(true) => " OK |",
                Some(false) => " FAIL |",
                None => " - |",
            });
        }
        if do_tir {
            row.push_str(match r.tir_codegen_ok {
                Some(true) => " OK |",
                Some(false) => " FAIL |",
                None => " - |",
            });
        }
        let _ = write!(row, " {} |", error_str);
        out.push_str(&row);
        out.push('\n');
    }
    out.push('\n');

    // Error details
    let errors: Vec<&SpecResult> = results.iter().filter(|r| !r.codegen_ok).collect();
    if !errors.is_empty() {
        out.push_str("## Error Details\n\n");
        let mut sorted_err: Vec<&&SpecResult> = errors.iter().collect();
        sorted_err.sort_by(|a, b| a.name.cmp(&b.name));
        for r in sorted_err {
            let _ = writeln!(out, "### {}", r.name);
            let _ = writeln!(
                out,
                "- **Category:** {}",
                r.error_category.as_deref().unwrap_or("?")
            );
            let _ = writeln!(out, "- **File:** `{}`", r.tla_path);
            let error_text = r.codegen_error.clone().unwrap_or_default();
            let trimmed = error_text.trim();
            if !trimmed.is_empty() {
                out.push_str("```\n");
                for line in trimmed.lines().take(3) {
                    out.push_str(line);
                    out.push('\n');
                }
                out.push_str("```\n");
            }
            out.push('\n');
        }
    }

    out
}

fn build_json_report(results: &[SpecResult], binary: &Path) -> Value {
    let total = results.len();
    let codegen_ok = results.iter().filter(|r| r.codegen_ok).count();
    let meaningful = results.iter().filter(|r| r.meaningful).count();
    let specs: Vec<Value> = results
        .iter()
        .map(|r| {
            serde_json::json!({
                "name": r.name,
                "category": r.category,
                "codegen_ok": r.codegen_ok,
                "meaningful": r.meaningful,
                "codegen_error": r.codegen_error,
                "error_category": r.error_category,
                "generated_lines": r.generated_lines,
                "codegen_elapsed": round2(r.codegen_elapsed),
                "compile_ok": r.compile_ok,
                "tir_codegen_ok": r.tir_codegen_ok,
                "tir_meaningful": r.tir_meaningful,
                "tlc_states": r.tlc_states,
            })
        })
        .collect();
    serde_json::json!({
        "date": iso_timestamp(),
        "binary": binary.display().to_string(),
        "total": total,
        "codegen_ok": codegen_ok,
        "codegen_meaningful": meaningful,
        "codegen_error": total - codegen_ok,
        "specs": specs,
    })
}

fn round2(x: f64) -> f64 {
    (x * 100.0).round() / 100.0
}

fn iso_timestamp() -> String {
    // Best-effort RFC 3339-like timestamp without bringing chrono into the
    // production deps. Mirrors `time.strftime("%Y-%m-%dT%H:%M:%S")`.
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    let secs = now.as_secs() as i64;
    format_unix_seconds(secs)
}

fn local_timestamp() -> String {
    // Matches `time.strftime('%Y-%m-%d %H:%M:%S')` in UTC. Tests assert format only.
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    let secs = now.as_secs() as i64;
    let formatted = format_unix_seconds(secs);
    formatted.replace('T', " ")
}

/// Format a Unix epoch second as ISO-8601 (UTC) without a timezone suffix.
/// Avoids pulling chrono into production deps for a one-shot timestamp.
fn format_unix_seconds(secs: i64) -> String {
    // Algorithm: convert to civil date via days-since-epoch arithmetic.
    let days = secs.div_euclid(86_400);
    let time_of_day = secs.rem_euclid(86_400);
    let (y, m, d) = civil_from_days(days);
    let h = time_of_day / 3600;
    let min = (time_of_day % 3600) / 60;
    let s = time_of_day % 60;
    format!("{y:04}-{m:02}-{d:02}T{h:02}:{min:02}:{s:02}")
}

/// Howard Hinnant's days-from-civil/days-to-civil algorithm. Public domain.
fn civil_from_days(days: i64) -> (i64, u32, u32) {
    // Shift epoch from 1970-01-01 to 0000-03-01.
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = (z - era * 146_097) as u32;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y_shifted = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5) + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y_shifted + 1 } else { y_shifted };
    (y, m, d)
}

fn resolve_repo_root(explicit: Option<&Path>) -> Result<PathBuf> {
    if let Some(p) = explicit {
        return Ok(p.to_path_buf());
    }
    if let Ok(p) = std::env::current_dir() {
        // Walk up looking for Cargo.toml + tests/tlc_comparison/spec_baseline.json
        let mut cur: Option<&Path> = Some(p.as_path());
        while let Some(dir) = cur {
            let cargo = dir.join("Cargo.toml");
            let baseline = dir
                .join("tests")
                .join("tlc_comparison")
                .join("spec_baseline.json");
            if cargo.exists() && baseline.exists() {
                return Ok(dir.to_path_buf());
            }
            cur = dir.parent();
        }
        // Fall back to current dir if nothing matched.
        return Ok(p);
    }
    Err(anyhow!("could not resolve repo root"))
}

fn default_examples_dir() -> PathBuf {
    let home = std::env::var_os("HOME")
        .map(PathBuf::from)
        .unwrap_or_default();
    home.join("tlaplus-examples").join("specifications")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classify_error_parse() {
        assert_eq!(
            classify_error("Parse error: unexpected token"),
            "parse_error"
        );
        assert_eq!(classify_error("SYNTAX error"), "parse_error");
    }

    #[test]
    fn classify_error_lower() {
        assert_eq!(classify_error("Lower failed: foo"), "lower_error");
        assert_eq!(classify_error("Lower produced no module"), "lower_error");
        assert_eq!(classify_error("TIR lowering failed: x"), "tir_lower_error");
    }

    #[test]
    fn classify_error_unimplemented() {
        assert_eq!(classify_error("not yet implemented"), "unimplemented");
        assert_eq!(classify_error("TODO: implement"), "unimplemented");
        assert_eq!(classify_error("unimplemented!()"), "unimplemented");
    }

    #[test]
    fn classify_error_module_and_instance() {
        assert_eq!(classify_error("extends Foo missing"), "module_not_found");
        assert_eq!(classify_error("Load extends Bar"), "module_not_found");
        assert_eq!(classify_error("instance error"), "instance_error");
    }

    #[test]
    fn classify_error_unsupported_and_other() {
        assert_eq!(
            classify_error("Unsupported for codegen: x"),
            "unsupported_feature"
        );
        assert_eq!(classify_error("Unsupported foo"), "unsupported_feature");
        assert_eq!(classify_error("totally fine"), "other");
    }

    #[test]
    fn classify_error_type() {
        assert_eq!(classify_error("Type inference error"), "type_inference");
        assert_eq!(classify_error("Type mismatch: x"), "type_error");
    }

    #[test]
    fn is_meaningful_codegen_detects_init_next() {
        let code = "pub fn init() {}\npub fn next() {}";
        assert!(is_meaningful_codegen(code));
    }

    #[test]
    fn is_meaningful_codegen_rejects_empty_stubs() {
        let code = "// No Init operator found\nfn init() {}\nfn next() {}";
        assert!(!is_meaningful_codegen(code));
    }

    #[test]
    fn pct_round_trip() {
        assert_eq!(pct(0, 0), 0);
        assert_eq!(pct(1, 4), 25);
        assert_eq!(pct(3, 4), 75);
        assert_eq!(pct(2, 3), 67);
    }

    #[test]
    fn civil_from_days_known_dates() {
        // 1970-01-01 -> days 0
        assert_eq!(civil_from_days(0), (1970, 1, 1));
        // 2000-03-01 -> days 11017
        assert_eq!(civil_from_days(11017), (2000, 3, 1));
        // 2026-05-18 -> verify a 5-digit day field roundtrips correctly.
        // (Sanity check; the function is the public-domain Hinnant algorithm.)
        let s = format_unix_seconds(0);
        assert_eq!(s, "1970-01-01T00:00:00");
    }

    #[test]
    fn select_specs_respects_filters() {
        let specs = serde_json::json!({
            "specA": {"category": "small", "source": {"tla_path": "a.tla", "cfg_path": "a.cfg"}},
            "specB": {"category": "medium", "source": {"tla_path": "b.tla", "cfg_path": "b.cfg"}},
            "specC": {"category": "small", "source": {"tla_path": "c.tla", "cfg_path": ""}},
        });
        let map = specs.as_object().unwrap();
        let tmpdir = tempfile::tempdir().unwrap();
        let dir = tmpdir.path();
        for f in ["a.tla", "b.tla", "c.tla"] {
            fs::write(dir.join(f), "").unwrap();
        }
        let cli = Cli {
            binary: PathBuf::from("/dev/null"),
            category: Some(Category::Small),
            limit: None,
            specs: vec![],
            compile: false,
            tir: false,
            output: None,
            json: false,
            repo_root: None,
            examples_dir: Some(dir.to_path_buf()),
        };
        let out = select_specs(map, dir, &cli).unwrap();
        let names: Vec<&str> = out.iter().map(|(n, _)| n.as_str()).collect();
        assert_eq!(names, vec!["specA", "specC"]);
    }

    #[test]
    fn select_specs_skips_missing_files() {
        let specs = serde_json::json!({
            "exists": {"category": "small", "source": {"tla_path": "a.tla", "cfg_path": ""}},
            "missing": {"category": "small", "source": {"tla_path": "missing.tla", "cfg_path": ""}},
        });
        let map = specs.as_object().unwrap();
        let tmpdir = tempfile::tempdir().unwrap();
        let dir = tmpdir.path();
        fs::write(dir.join("a.tla"), "").unwrap();
        let cli = Cli {
            binary: PathBuf::from("/dev/null"),
            category: None,
            limit: None,
            specs: vec![],
            compile: false,
            tir: false,
            output: None,
            json: false,
            repo_root: None,
            examples_dir: Some(dir.to_path_buf()),
        };
        let out = select_specs(map, dir, &cli).unwrap();
        let names: Vec<&str> = out.iter().map(|(n, _)| n.as_str()).collect();
        assert_eq!(names, vec!["exists"]);
    }
}
