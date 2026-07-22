// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Rust port of `scripts/validate_codegen_state_counts.py`.
//!
//! End-to-end validation pipeline:
//!
//! 1. Run `ty codegen --tir` on a spec to generate Rust code
//! 2. Create a standalone Cargo project with the generated code
//! 3. Build and run the generated model checker
//! 4. Compare state counts against `spec_baseline.json`
//!
//! This validates the codegen pipeline produces *semantically correct*
//! code, not just code that compiles. State count mismatches indicate
//! bugs in the code generation (incorrect Init/Next/invariant translation).

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode, Stdio};
use std::time::{Duration, Instant};

use anyhow::{anyhow, Context, Result};
use clap::{Parser, ValueEnum};
use serde_json::Value;

const CODEGEN_TIMEOUT: Duration = Duration::from_secs(30);
const BUILD_TIMEOUT: Duration = Duration::from_secs(180);
const RUN_TIMEOUT: Duration = Duration::from_secs(120);

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
    name = "ty-validate-codegen-state-counts",
    about = "Validate codegen state counts against TLC baselines.",
    long_about = "Rust port of scripts/validate_codegen_state_counts.py. \
                  Runs `ty codegen --tir` on each baseline spec, creates a \
                  standalone Cargo project with the generated code + the \
                  workspace `tla-runtime` crate, builds and runs the generated \
                  model checker, and compares the distinct-state count with \
                  the TLC baseline. Exit 0 on completion; non-zero on hard \
                  usage errors (binary missing, runtime crate missing, etc)."
)]
struct Cli {
    /// Path to the `ty` binary used for codegen.
    #[arg(long, value_name = "PATH", required = true)]
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

    /// Write a JSON report to this path.
    #[arg(long, value_name = "PATH")]
    json: Option<PathBuf>,

    /// Don't delete the temporary work directory.
    #[arg(long = "keep-work-dir")]
    keep_work_dir: bool,

    /// Shared Cargo target dir (avoids recompiling `tla-runtime` per spec).
    #[arg(long = "shared-target", value_name = "PATH")]
    shared_target: Option<PathBuf>,

    /// Override the repository root.
    #[arg(long = "repo-root", value_name = "PATH")]
    repo_root: Option<PathBuf>,

    /// Override the examples directory.
    /// Defaults to `~/tlaplus-examples/specifications`.
    #[arg(long = "examples-dir", value_name = "PATH")]
    examples_dir: Option<PathBuf>,
}

#[derive(Debug, Default, Clone)]
struct ValidationResult {
    name: String,
    category: String,
    tlc_states: Option<i64>,
    tlc_status: String,
    codegen_ok: bool,
    codegen_error: Option<String>,
    build_ok: bool,
    build_error: Option<String>,
    run_ok: bool,
    run_error: Option<String>,
    codegen_distinct_states: Option<i64>,
    codegen_states_explored: Option<i64>,
    states_match: Option<bool>,
    has_deadlock: bool,
    has_violation: bool,
    codegen_elapsed: f64,
    build_elapsed: f64,
    run_elapsed: f64,
}

impl ValidationResult {
    fn overall_status(&self) -> &'static str {
        if !self.codegen_ok {
            return "codegen_error";
        }
        if !self.build_ok {
            return "build_error";
        }
        if !self.run_ok {
            return "run_error";
        }
        if self.has_violation {
            return "violation";
        }
        match self.states_match {
            Some(true) => "PASS",
            Some(false) => "STATE_MISMATCH",
            None => "unknown",
        }
    }
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
    let runtime_path = repo_root.join("crates").join("tla-runtime");
    let vendored_stacker = repo_root.join("vendored").join("stacker");
    let examples_dir = cli
        .examples_dir
        .clone()
        .unwrap_or_else(default_examples_dir);
    let baseline_path = repo_root
        .join("tests")
        .join("tlc_comparison")
        .join("spec_baseline.json");

    if !cli.binary.exists() {
        return Err(anyhow!("binary not found at {}", cli.binary.display()));
    }
    if !runtime_path.exists() {
        return Err(anyhow!(
            "tla-runtime not found at {}",
            runtime_path.display()
        ));
    }

    let baseline_text = fs::read_to_string(&baseline_path)
        .with_context(|| format!("reading {}", baseline_path.display()))?;
    let baseline: Value = serde_json::from_str(&baseline_text)
        .with_context(|| format!("parsing {}", baseline_path.display()))?;

    let specs_map = baseline
        .get("specs")
        .and_then(|s| s.as_object())
        .ok_or_else(|| anyhow!("baseline missing .specs object"))?;

    let selected = select_specs(specs_map, &examples_dir, cli);
    if selected.is_empty() {
        return Err(anyhow!("no specs selected for testing"));
    }

    let raw_cargo = find_raw_cargo();
    let shared_target = cli.shared_target.clone();
    if let Some(p) = &shared_target {
        fs::create_dir_all(p).with_context(|| format!("creating {}", p.display()))?;
    }

    println!(
        "Testing {} specs with binary: {}",
        selected.len(),
        cli.binary.display()
    );
    println!("Runtime: {}", runtime_path.display());
    println!();

    let work_dir = tempfile::Builder::new()
        .prefix("codegen_validate_")
        .tempdir()
        .context("creating work dir")?;
    println!("Work directory: {}", work_dir.path().display());
    println!();

    let mut results: Vec<ValidationResult> = Vec::with_capacity(selected.len());
    for (idx, (name, spec)) in selected.iter().enumerate() {
        print!(
            "[{}/{}] {} ({})... ",
            idx + 1,
            selected.len(),
            name,
            spec.get("category").and_then(|v| v.as_str()).unwrap_or("?")
        );
        use std::io::Write as _;
        let _ = std::io::stdout().flush();

        let vr = validate_spec(
            &cli.binary,
            name,
            spec,
            work_dir.path(),
            &runtime_path,
            &vendored_stacker,
            &examples_dir,
            &raw_cargo,
            shared_target.as_deref(),
        );

        let status = vr.overall_status();
        match status {
            "PASS" => println!(
                "PASS (states={}, codegen={:.1}s, build={:.1}s, run={:.1}s)",
                vr.codegen_distinct_states.unwrap_or(-1),
                vr.codegen_elapsed,
                vr.build_elapsed,
                vr.run_elapsed
            ),
            "STATE_MISMATCH" => println!(
                "MISMATCH (tlc={}, codegen={})",
                vr.tlc_states.map(|n| n.to_string()).unwrap_or("?".into()),
                vr.codegen_distinct_states
                    .map(|n| n.to_string())
                    .unwrap_or("?".into())
            ),
            "violation" => println!(
                "VIOLATION (states={})",
                vr.codegen_distinct_states.unwrap_or(-1)
            ),
            other => {
                let err = vr
                    .codegen_error
                    .clone()
                    .or_else(|| vr.build_error.clone())
                    .or_else(|| vr.run_error.clone())
                    .unwrap_or_else(|| "?".to_string());
                let first_line = err.lines().next().unwrap_or("?");
                let trimmed = first_line.chars().take(60).collect::<String>();
                println!("{}: {}", other, trimmed);
            }
        }

        results.push(vr);
    }

    print_summary(&results);

    if let Some(path) = &cli.json {
        write_json_report(&results, path)?;
    }

    if cli.keep_work_dir {
        // Persist the work dir by leaking the TempDir handle.
        let kept = work_dir.keep();
        println!("Work directory kept: {}", kept.display());
    } else {
        let path = work_dir.path().to_path_buf();
        drop(work_dir);
        println!("Cleaned up work directory: {}", path.display());
    }

    Ok(())
}

fn select_specs<'a>(
    specs: &'a serde_json::Map<String, Value>,
    examples_dir: &Path,
    cli: &Cli,
) -> Vec<(String, &'a Value)> {
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
        let tlc = spec.get("tlc");
        let tlc_status = tlc
            .and_then(|t| t.get("status"))
            .and_then(|v| v.as_str())
            .unwrap_or("");
        if tlc_status != "pass" {
            continue;
        }
        let tlc_states = tlc.and_then(|t| t.get("states")).and_then(|v| v.as_i64());
        if tlc_states.is_none() {
            continue;
        }
        let tla_rel = spec
            .get("source")
            .and_then(|s| s.get("tla_path"))
            .and_then(|v| v.as_str())
            .unwrap_or("");
        if tla_rel.is_empty() {
            continue;
        }
        if !resolve_tla_path(tla_rel, examples_dir)
            .map(|p| p.exists())
            .unwrap_or(false)
        {
            continue;
        }
        out.push((name.clone(), spec));
    }
    if let Some(limit) = cli.limit {
        out.truncate(limit);
    }
    out
}

fn resolve_tla_path(rel: &str, examples_dir: &Path) -> Option<PathBuf> {
    if rel.starts_with('/') {
        let p = PathBuf::from(rel);
        if p.exists() {
            return Some(p);
        }
        return None;
    }
    let p = examples_dir.join(rel);
    if p.exists() {
        Some(p)
    } else {
        None
    }
}

#[allow(clippy::too_many_arguments)]
fn validate_spec(
    binary: &Path,
    name: &str,
    spec: &Value,
    work_dir: &Path,
    runtime_path: &Path,
    vendored_stacker: &Path,
    examples_dir: &Path,
    raw_cargo: &str,
    shared_target: Option<&Path>,
) -> ValidationResult {
    let tla_rel = spec
        .get("source")
        .and_then(|s| s.get("tla_path"))
        .and_then(|v| v.as_str())
        .unwrap_or("");
    let cfg_rel = spec
        .get("source")
        .and_then(|s| s.get("cfg_path"))
        .and_then(|v| v.as_str())
        .unwrap_or("");
    let tla_path = match resolve_tla_path(tla_rel, examples_dir) {
        Some(p) => p,
        None => {
            let vr = ValidationResult {
                name: name.to_string(),
                codegen_error: Some(format!("tla not found: {tla_rel}")),
                ..Default::default()
            };
            return vr;
        }
    };
    let cfg_path = if cfg_rel.is_empty() {
        None
    } else if cfg_rel.starts_with('/') {
        let p = PathBuf::from(cfg_rel);
        if p.exists() {
            Some(p)
        } else {
            None
        }
    } else {
        let p = examples_dir.join(cfg_rel);
        if p.exists() {
            Some(p)
        } else {
            None
        }
    };

    let tlc = spec.get("tlc");
    let mut vr = ValidationResult {
        name: name.to_string(),
        category: spec
            .get("category")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown")
            .to_string(),
        tlc_states: tlc.and_then(|t| t.get("states")).and_then(|v| v.as_i64()),
        tlc_status: tlc
            .and_then(|t| t.get("status"))
            .and_then(|v| v.as_str())
            .unwrap_or("unknown")
            .to_string(),
        ..Default::default()
    };

    // Step 1: codegen.
    let (ok, code_or_error, elapsed) = run_codegen(binary, &tla_path, cfg_path.as_deref());
    vr.codegen_elapsed = elapsed;
    if !ok {
        vr.codegen_error = Some(code_or_error);
        return vr;
    }
    let rust_code = code_or_error;
    vr.codegen_ok = true;

    if !rust_code.contains("fn init(") || !rust_code.contains("fn next(") {
        vr.codegen_error = Some("generated code missing init/next".to_string());
        vr.codegen_ok = false;
        return vr;
    }

    // Step 2: create project and build.
    let project_dir =
        match create_project(&rust_code, name, work_dir, runtime_path, vendored_stacker) {
            Ok(p) => p,
            Err(err) => {
                vr.build_error = Some(format!("project setup: {err}"));
                return vr;
            }
        };
    let (build_ok, build_err, build_elapsed) =
        build_project(&project_dir, raw_cargo, shared_target);
    vr.build_elapsed = build_elapsed;
    if !build_ok {
        vr.build_ok = false;
        vr.build_error = Some(build_err);
        return vr;
    }
    vr.build_ok = true;

    // Step 3: run model checker.
    let (run_ok, parsed, run_elapsed) = run_model_check(&project_dir, shared_target);
    vr.run_elapsed = run_elapsed;
    vr.codegen_distinct_states = parsed.get("distinct_states").and_then(|v| v.as_i64());
    vr.codegen_states_explored = parsed.get("states_explored").and_then(|v| v.as_i64());
    vr.has_violation = parsed
        .get("violation")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    vr.has_deadlock = parsed
        .get("deadlock")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    if !run_ok && !vr.has_violation {
        vr.run_error = Some(
            parsed
                .get("error")
                .or_else(|| parsed.get("stderr"))
                .and_then(|v| v.as_str())
                .unwrap_or("unknown runtime error")
                .to_string(),
        );
        return vr;
    }
    vr.run_ok = true;

    // Step 4: compare state counts.
    if let (Some(tlc), Some(codegen)) = (vr.tlc_states, vr.codegen_distinct_states) {
        if vr.tlc_status == "pass" {
            if vr.has_violation {
                vr.states_match = None;
            } else {
                vr.states_match = Some(codegen == tlc);
            }
        }
    }

    vr
}

fn run_codegen(binary: &Path, tla_path: &Path, cfg_path: Option<&Path>) -> (bool, String, f64) {
    let mut cmd = Command::new(binary);
    cmd.arg("codegen").arg("--tir").arg(tla_path);
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
                (
                    true,
                    String::from_utf8_lossy(&output.stdout).into_owned(),
                    elapsed,
                )
            } else {
                let stderr = String::from_utf8_lossy(&output.stderr);
                let stdout = String::from_utf8_lossy(&output.stdout);
                let mut error = stderr.trim().to_string();
                if error.is_empty() {
                    error = stdout.trim().to_string();
                }
                if error.len() > 500 {
                    error.truncate(500);
                }
                (false, error, elapsed)
            }
        }
        Err(err) => {
            let elapsed = start.elapsed().as_secs_f64();
            (false, err.message, elapsed)
        }
    }
}

fn sanitize_package_name(name: &str) -> String {
    let snake = to_snake_case(name);
    let mut result = if snake
        .chars()
        .next()
        .map(|c| !c.is_ascii_alphabetic())
        .unwrap_or(true)
    {
        format!("spec_{snake}")
    } else {
        snake
    };
    result = result
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' || c == '-' {
                c
            } else {
                '_'
            }
        })
        .collect();
    result
}

fn to_snake_case(s: &str) -> String {
    let mut out = String::new();
    for (i, c) in s.chars().enumerate() {
        if c.is_ascii_uppercase() && i > 0 {
            out.push('_');
        }
        for lc in c.to_lowercase() {
            out.push(lc);
        }
    }
    out
}

fn extract_machine_name(rust_code: &str) -> String {
    // pub struct <Name>State
    for line in rust_code.lines() {
        let line = line.trim();
        if let Some(rest) = line.strip_prefix("pub struct ") {
            if let Some(name_end) = rest.find("State") {
                let name = &rest[..name_end];
                // Reject identifiers that contain whitespace etc.
                if !name.is_empty() && name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') {
                    return name.to_string();
                }
            }
        }
    }
    // impl StateMachine for X
    for line in rust_code.lines() {
        let line = line.trim();
        if let Some(rest) = line.strip_prefix("impl StateMachine for ") {
            let mut name = String::new();
            for c in rest.chars() {
                if c.is_ascii_alphanumeric() || c == '_' {
                    name.push(c);
                } else {
                    break;
                }
            }
            if !name.is_empty() {
                return name;
            }
        }
    }
    "Spec".to_string()
}

fn create_project(
    rust_code: &str,
    spec_name: &str,
    work_dir: &Path,
    runtime_path: &Path,
    vendored_stacker: &Path,
) -> Result<PathBuf> {
    let project_dir = work_dir.join(format!("{spec_name}_codegen"));
    let src_dir = project_dir.join("src");
    fs::create_dir_all(&src_dir)?;

    let mod_name = sanitize_package_name(spec_name);
    let pkg_name = sanitize_package_name(spec_name);
    let machine_type = extract_machine_name(rust_code);

    let mut stacker_patch = String::new();
    if vendored_stacker.exists() {
        stacker_patch = format!(
            "\n[patch.crates-io]\nstacker = {{ path = \"{}\" }}\n",
            vendored_stacker.display()
        );
    }

    let cargo_toml = format!(
        "[package]\n\
         name = \"{pkg}-codegen-test\"\n\
         version = \"0.1.0\"\n\
         edition = \"2021\"\n\
         \n\
         [workspace]\n\
         \n\
         [[bin]]\n\
         name = \"check\"\n\
         path = \"src/main.rs\"\n\
         \n\
         [dependencies]\n\
         tla-runtime = {{ path = \"{runtime}\" }}\n\
         {patch}",
        pkg = pkg_name,
        runtime = runtime_path.display(),
        patch = stacker_patch,
    );
    fs::write(project_dir.join("Cargo.toml"), cargo_toml)?;

    fs::write(src_dir.join(format!("{mod_name}.rs")), rust_code)?;

    let main_rs = format!(
        "#![allow(unused)]\n\
         mod {mod_name};\n\
         \n\
         use {mod_name}::{machine};\n\
         use tla_runtime::prelude::*;\n\
         \n\
         fn main() {{\n\
         \x20   let machine = {machine};\n\
         \x20   let max_states = 10_000_000;\n\
         \n\
         \x20   let result = model_check(&machine, max_states);\n\
         \n\
         \x20   println!(\"CODEGEN_DISTINCT_STATES={{}}\", result.distinct_states);\n\
         \x20   println!(\"CODEGEN_STATES_EXPLORED={{}}\", result.states_explored);\n\
         \n\
         \x20   if let Some(ref violation) = result.violation {{\n\
         \x20       println!(\"CODEGEN_VIOLATION=true\");\n\
         \x20       eprintln!(\"INVARIANT VIOLATION: {{:?}}\", violation.state);\n\
         \x20   }} else {{\n\
         \x20       println!(\"CODEGEN_VIOLATION=false\");\n\
         \x20   }}\n\
         \n\
         \x20   if let Some(ref _deadlock) = result.deadlock {{\n\
         \x20       println!(\"CODEGEN_DEADLOCK=true\");\n\
         \x20   }} else {{\n\
         \x20       println!(\"CODEGEN_DEADLOCK=false\");\n\
         \x20   }}\n\
         \n\
         \x20   if result.is_ok() || result.deadlock.is_some() {{\n\
         \x20       println!(\"CODEGEN_STATUS=ok\");\n\
         \x20   }} else {{\n\
         \x20       println!(\"CODEGEN_STATUS=error\");\n\
         \x20       std::process::exit(1);\n\
         \x20   }}\n\
         }}\n",
        mod_name = mod_name,
        machine = machine_type,
    );
    fs::write(src_dir.join("main.rs"), main_rs)?;

    Ok(project_dir)
}

fn find_raw_cargo() -> String {
    let mut cmd = Command::new("rustup");
    cmd.args(["which", "cargo", "--toolchain", "stable"]);
    if let Ok(out) = cmd.output() {
        if out.status.success() {
            let s = String::from_utf8_lossy(&out.stdout).trim().to_string();
            if !s.is_empty() {
                return s;
            }
        }
    }
    "cargo".to_string()
}

fn build_project(
    project_dir: &Path,
    raw_cargo: &str,
    shared_target: Option<&Path>,
) -> (bool, String, f64) {
    let target_dir = shared_target
        .map(Path::to_path_buf)
        .unwrap_or_else(|| project_dir.join("target"));
    let mut cmd = Command::new(raw_cargo);
    cmd.args(["build", "--release", "--target-dir"])
        .arg(&target_dir)
        .current_dir(project_dir)
        .env("RUSTUP_TOOLCHAIN", "stable-aarch64-apple-darwin");
    let start = Instant::now();
    match run_with_timeout(cmd, BUILD_TIMEOUT) {
        Ok(output) => {
            let elapsed = start.elapsed().as_secs_f64();
            if output.status.success() {
                (true, String::new(), elapsed)
            } else {
                let stderr = String::from_utf8_lossy(&output.stderr);
                let mut error_lines: Vec<String> = Vec::new();
                for line in stderr.lines() {
                    let stripped = line.trim();
                    if stripped.starts_with("error")
                        || stripped.starts_with("-->")
                        || stripped.contains("mismatched")
                    {
                        error_lines.push(stripped.to_string());
                    }
                }
                let error = if !error_lines.is_empty() {
                    error_lines
                        .into_iter()
                        .take(10)
                        .collect::<Vec<_>>()
                        .join("\n")
                } else {
                    let mut s = stderr.trim().to_string();
                    s.truncate(s.len().min(1000));
                    s
                };
                (false, error, elapsed)
            }
        }
        Err(err) => {
            let elapsed = start.elapsed().as_secs_f64();
            (false, err.message, elapsed)
        }
    }
}

fn run_model_check(
    project_dir: &Path,
    shared_target: Option<&Path>,
) -> (bool, HashMap<String, Value>, f64) {
    let target_dir = shared_target
        .map(Path::to_path_buf)
        .unwrap_or_else(|| project_dir.join("target"));
    let binary = target_dir.join("release").join("check");
    if !binary.exists() {
        let mut parsed: HashMap<String, Value> = HashMap::new();
        parsed.insert(
            "error".to_string(),
            Value::String(format!("binary not found at {}", binary.display())),
        );
        return (false, parsed, 0.0);
    }

    let mut cmd = Command::new(&binary);
    cmd.current_dir(project_dir);
    let start = Instant::now();
    match run_with_timeout(cmd, RUN_TIMEOUT) {
        Ok(output) => {
            let elapsed = start.elapsed().as_secs_f64();
            let stdout = String::from_utf8_lossy(&output.stdout);
            let mut parsed = parse_model_check_output(&stdout);
            if output.status.success() {
                (true, parsed, elapsed)
            } else {
                parsed.insert(
                    "exit_code".to_string(),
                    Value::from(output.status.code().unwrap_or(-1) as i64),
                );
                let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
                let truncated: String = stderr.chars().take(500).collect();
                parsed.insert("stderr".to_string(), Value::String(truncated));
                (false, parsed, elapsed)
            }
        }
        Err(err) => {
            let elapsed = start.elapsed().as_secs_f64();
            let mut parsed: HashMap<String, Value> = HashMap::new();
            if err.timed_out {
                parsed.insert(
                    "error".to_string(),
                    Value::String(format!("runtime timeout ({}s)", RUN_TIMEOUT.as_secs())),
                );
            } else {
                parsed.insert("error".to_string(), Value::String(err.message));
            }
            (false, parsed, elapsed)
        }
    }
}

fn parse_model_check_output(output: &str) -> HashMap<String, Value> {
    let mut result: HashMap<String, Value> = HashMap::new();
    for line in output.lines() {
        if let Some((key, value)) = line.split_once('=') {
            let key = key.trim();
            let value = value.trim();
            match key {
                "CODEGEN_DISTINCT_STATES" => {
                    if let Ok(n) = value.parse::<i64>() {
                        result.insert("distinct_states".to_string(), Value::from(n));
                    }
                }
                "CODEGEN_STATES_EXPLORED" => {
                    if let Ok(n) = value.parse::<i64>() {
                        result.insert("states_explored".to_string(), Value::from(n));
                    }
                }
                "CODEGEN_VIOLATION" => {
                    result.insert("violation".to_string(), Value::from(value == "true"));
                }
                "CODEGEN_DEADLOCK" => {
                    result.insert("deadlock".to_string(), Value::from(value == "true"));
                }
                "CODEGEN_STATUS" => {
                    result.insert("status".to_string(), Value::String(value.to_string()));
                }
                _ => {}
            }
        }
    }
    result
}

fn print_summary(results: &[ValidationResult]) {
    let total = results.len();
    if total == 0 {
        println!("No specs tested.");
        return;
    }
    let codegen_ok = results.iter().filter(|r| r.codegen_ok).count();
    let build_ok = results.iter().filter(|r| r.build_ok).count();
    let run_ok = results.iter().filter(|r| r.run_ok).count();
    let pass = results
        .iter()
        .filter(|r| r.overall_status() == "PASS")
        .count();
    let mismatch = results
        .iter()
        .filter(|r| r.overall_status() == "STATE_MISMATCH")
        .count();
    let violations = results
        .iter()
        .filter(|r| r.overall_status() == "violation")
        .count();

    println!();
    println!("{}", "=".repeat(70));
    println!("CODEGEN STATE COUNT VALIDATION SUMMARY");
    println!("{}", "=".repeat(70));
    println!("Total specs tested:      {}", total);
    println!("Codegen OK:              {}/{}", codegen_ok, total);
    println!("Build OK:                {}/{}", build_ok, total);
    println!("Run OK:                  {}/{}", run_ok, total);
    println!("State count PASS:        {}/{}", pass, total);
    println!("State count MISMATCH:    {}/{}", mismatch, total);
    println!("Invariant violation:     {}/{}", violations, total);
    println!();

    println!(
        "{:<30} {:<16} {:>8} {:>8} {:>6}",
        "Spec", "Status", "TLC", "Codegen", "Match"
    );
    println!("{}", "-".repeat(70));
    let mut sorted: Vec<&ValidationResult> = results.iter().collect();
    sorted.sort_by(|a, b| {
        (a.overall_status() != "PASS")
            .cmp(&(b.overall_status() != "PASS"))
            .then(a.name.cmp(&b.name))
    });
    for r in sorted {
        let tlc_str = r
            .tlc_states
            .map(|n| n.to_string())
            .unwrap_or_else(|| "-".to_string());
        let cg_str = r
            .codegen_distinct_states
            .map(|n| n.to_string())
            .unwrap_or_else(|| "-".to_string());
        let match_str = match r.states_match {
            Some(true) => "YES",
            Some(false) => "NO",
            None => "-",
        };
        println!(
            "{:<30} {:<16} {:>8} {:>8} {:>6}",
            r.name,
            r.overall_status(),
            tlc_str,
            cg_str,
            match_str,
        );
    }

    let errors: Vec<&ValidationResult> = results
        .iter()
        .filter(|r| !matches!(r.overall_status(), "PASS" | "STATE_MISMATCH"))
        .collect();
    if !errors.is_empty() {
        println!();
        println!("ERRORS:");
        for r in errors {
            let msg = r
                .codegen_error
                .clone()
                .or_else(|| r.build_error.clone())
                .or_else(|| r.run_error.clone())
                .unwrap_or_else(|| "unknown".to_string());
            let first_line = msg.lines().next().unwrap_or("");
            let first_line: String = first_line.chars().take(80).collect();
            println!("  {}: {} - {}", r.name, r.overall_status(), first_line);
        }
    }
}

fn write_json_report(results: &[ValidationResult], path: &Path) -> Result<()> {
    let pass = results
        .iter()
        .filter(|r| r.overall_status() == "PASS")
        .count();
    let mismatch = results
        .iter()
        .filter(|r| r.overall_status() == "STATE_MISMATCH")
        .count();
    let specs: Vec<Value> = results
        .iter()
        .map(|r| {
            serde_json::json!({
                "name": r.name,
                "category": r.category,
                "status": r.overall_status(),
                "tlc_states": r.tlc_states,
                "codegen_distinct_states": r.codegen_distinct_states,
                "codegen_states_explored": r.codegen_states_explored,
                "states_match": r.states_match,
                "has_violation": r.has_violation,
                "has_deadlock": r.has_deadlock,
                "codegen_error": r.codegen_error,
                "build_error": r.build_error,
                "run_error": r.run_error,
                "timings": {
                    "codegen_s": round2(r.codegen_elapsed),
                    "build_s": round2(r.build_elapsed),
                    "run_s": round2(r.run_elapsed),
                }
            })
        })
        .collect();
    let payload = serde_json::json!({
        "date": iso_timestamp(),
        "total": results.len(),
        "pass": pass,
        "mismatch": mismatch,
        "specs": specs,
    });
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).with_context(|| format!("creating {}", parent.display()))?;
    }
    fs::write(
        path,
        serde_json::to_string_pretty(&payload).context("json serialize")?,
    )
    .with_context(|| format!("writing {}", path.display()))?;
    println!("\nJSON report: {}", path.display());
    Ok(())
}

fn round2(x: f64) -> f64 {
    (x * 100.0).round() / 100.0
}

fn iso_timestamp() -> String {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    let secs = now.as_secs() as i64;
    let days = secs.div_euclid(86_400);
    let tod = secs.rem_euclid(86_400);
    let (y, m, d) = civil_from_days(days);
    let h = tod / 3600;
    let min = (tod % 3600) / 60;
    let s = tod % 60;
    format!("{y:04}-{m:02}-{d:02}T{h:02}:{min:02}:{s:02}")
}

fn civil_from_days(days: i64) -> (i64, u32, u32) {
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

struct RunErr {
    message: String,
    timed_out: bool,
}

fn run_with_timeout(mut cmd: Command, timeout: Duration) -> Result<std::process::Output, RunErr> {
    use std::io::Read as _;
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

fn resolve_repo_root(explicit: Option<&Path>) -> Result<PathBuf> {
    if let Some(p) = explicit {
        return Ok(p.to_path_buf());
    }
    if let Ok(p) = std::env::current_dir() {
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
    fn snake_case_conversions() {
        assert_eq!(to_snake_case("HourClock"), "hour_clock");
        assert_eq!(to_snake_case("DieHard"), "die_hard");
        assert_eq!(to_snake_case("hour_clock"), "hour_clock");
    }

    #[test]
    fn sanitize_package_name_handles_leading_digit() {
        // After snake-case "2PCwithBTM" stays "2_p_cwith_b_t_m" (digit head).
        let out = sanitize_package_name("2PCwithBTM");
        assert!(out.starts_with("spec_"));
        // ensure no invalid chars
        assert!(out
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-'));
    }

    #[test]
    fn sanitize_package_name_letter_first() {
        let out = sanitize_package_name("DieHard");
        assert_eq!(out, "die_hard");
    }

    #[test]
    fn extract_machine_name_state_struct() {
        let code = "pub struct DieHardState { /* ... */ }\npub struct DieHard;";
        assert_eq!(extract_machine_name(code), "DieHard");
    }

    #[test]
    fn extract_machine_name_state_machine_impl() {
        let code = "impl StateMachine for HourClock {\n    fn init(&self) {}\n}";
        assert_eq!(extract_machine_name(code), "HourClock");
    }

    #[test]
    fn extract_machine_name_fallback() {
        assert_eq!(extract_machine_name(""), "Spec");
    }

    #[test]
    fn parse_model_check_output_full() {
        let stdout = "CODEGEN_DISTINCT_STATES=42\n\
                      CODEGEN_STATES_EXPLORED=100\n\
                      CODEGEN_VIOLATION=false\n\
                      CODEGEN_DEADLOCK=false\n\
                      CODEGEN_STATUS=ok\n";
        let parsed = parse_model_check_output(stdout);
        assert_eq!(
            parsed.get("distinct_states").and_then(|v| v.as_i64()),
            Some(42)
        );
        assert_eq!(
            parsed.get("states_explored").and_then(|v| v.as_i64()),
            Some(100)
        );
        assert_eq!(
            parsed.get("violation").and_then(|v| v.as_bool()),
            Some(false)
        );
        assert_eq!(
            parsed.get("deadlock").and_then(|v| v.as_bool()),
            Some(false)
        );
        assert_eq!(parsed.get("status").and_then(|v| v.as_str()), Some("ok"));
    }

    #[test]
    fn parse_model_check_output_handles_violation() {
        let stdout = "CODEGEN_DISTINCT_STATES=16\n\
                      CODEGEN_VIOLATION=true\n";
        let parsed = parse_model_check_output(stdout);
        assert_eq!(
            parsed.get("violation").and_then(|v| v.as_bool()),
            Some(true)
        );
        assert_eq!(
            parsed.get("distinct_states").and_then(|v| v.as_i64()),
            Some(16)
        );
    }

    #[test]
    fn validation_status_codegen_error() {
        let vr = ValidationResult {
            name: "x".into(),
            ..Default::default()
        };
        assert_eq!(vr.overall_status(), "codegen_error");
    }

    #[test]
    fn validation_status_pass() {
        let vr = ValidationResult {
            codegen_ok: true,
            build_ok: true,
            run_ok: true,
            states_match: Some(true),
            ..Default::default()
        };
        assert_eq!(vr.overall_status(), "PASS");
    }

    #[test]
    fn validation_status_state_mismatch() {
        let vr = ValidationResult {
            codegen_ok: true,
            build_ok: true,
            run_ok: true,
            states_match: Some(false),
            ..Default::default()
        };
        assert_eq!(vr.overall_status(), "STATE_MISMATCH");
    }

    #[test]
    fn validation_status_violation_overrides_match() {
        let vr = ValidationResult {
            codegen_ok: true,
            build_ok: true,
            run_ok: true,
            has_violation: true,
            states_match: Some(true),
            ..Default::default()
        };
        assert_eq!(vr.overall_status(), "violation");
    }

    #[test]
    fn select_specs_filters_non_pass() {
        let specs = serde_json::json!({
            "ok_spec": {
                "category": "small",
                "source": {"tla_path": "ok.tla", "cfg_path": ""},
                "tlc": {"status": "pass", "states": 4}
            },
            "fail_spec": {
                "category": "small",
                "source": {"tla_path": "fail.tla", "cfg_path": ""},
                "tlc": {"status": "fail", "states": 4}
            },
            "no_states": {
                "category": "small",
                "source": {"tla_path": "no.tla", "cfg_path": ""},
                "tlc": {"status": "pass"}
            },
        });
        let map = specs.as_object().unwrap();
        let tmpdir = tempfile::tempdir().unwrap();
        let dir = tmpdir.path();
        for f in ["ok.tla", "fail.tla", "no.tla"] {
            fs::write(dir.join(f), "").unwrap();
        }
        let cli = Cli {
            binary: PathBuf::from("/dev/null"),
            category: None,
            limit: None,
            specs: vec![],
            json: None,
            keep_work_dir: false,
            shared_target: None,
            repo_root: None,
            examples_dir: Some(dir.to_path_buf()),
        };
        let out = select_specs(map, dir, &cli);
        let names: Vec<&str> = out.iter().map(|(n, _)| n.as_str()).collect();
        assert_eq!(names, vec!["ok_spec"]);
    }
}
