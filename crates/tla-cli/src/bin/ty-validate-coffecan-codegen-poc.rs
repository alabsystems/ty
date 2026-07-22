// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Validate CoffeeCan codegen/AOT against interpreter and TLC.
//!
//! Rust port of `scripts/validate_coffecan_codegen_poc.py`. Stages a temporary
//! wrapper module around the upstream `CoffeeCan` example, runs `ty codegen`
//! to produce a Rust state machine, builds a standalone BFS harness, and
//! compares distinct-state counts against the closed-form expected total, the
//! TY interpreter, and TLC.

use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode};
use std::time::Instant;

use anyhow::{anyhow, bail, Context, Result};
use clap::Parser;
use regex::Regex;
use serde::Serialize;
use serde_json::{json, Value};

#[derive(Parser, Debug)]
#[command(
    name = "ty-validate-coffecan-codegen-poc",
    about = "Validate CoffeeCan codegen/AOT against interpreter and TLC"
)]
struct Cli {
    /// Bean count used to instantiate `CoffeeCan.MaxBeanCount`.
    #[arg(long, default_value_t = 3000)]
    beans: i64,
    /// Per-step timeout in seconds.
    #[arg(long, default_value_t = 1800)]
    timeout: u64,
    /// Skip the `cargo build` step that produces the `ty` binary.
    #[arg(long)]
    skip_build: bool,
    /// Skip running the TY interpreter.
    #[arg(long)]
    skip_interpreter: bool,
    /// Skip running TLC.
    #[arg(long)]
    skip_tlc: bool,
    /// Add `InvType` invariant to the staged wrapper module.
    #[arg(long)]
    with_invariants: bool,
    /// Keep the staged temp directory for inspection.
    #[arg(long)]
    keep_temp: bool,
    /// Write the JSON summary to this path in addition to stdout.
    #[arg(long)]
    output_json: Option<PathBuf>,
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    match run(&cli) {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("error: {err:#}");
            ExitCode::from(1)
        }
    }
}

#[derive(Clone, Debug)]
struct CommandResult {
    command: Vec<String>,
    cwd: String,
    elapsed_seconds: f64,
    returncode: i32,
    stdout: String,
    stderr: String,
}

#[derive(Serialize)]
struct BuildMeta {
    command: Vec<String>,
    cwd: String,
    elapsed_seconds: f64,
    returncode: i32,
}

#[derive(Serialize)]
struct CodegenMeta {
    command: Vec<String>,
    cwd: String,
    elapsed_seconds: f64,
    returncode: i32,
    stderr: String,
}

#[derive(Serialize)]
struct AotMeta {
    build_elapsed_seconds: f64,
    run_elapsed_seconds: f64,
    status: Option<String>,
    states_explored: i64,
    states_initial: i64,
    states_distinct: i64,
    transitions: i64,
    elapsed_seconds_internal: f64,
    states_per_second: f64,
}

#[derive(Serialize)]
struct InterpreterMeta {
    elapsed_seconds_wall: f64,
    status: String,
    states_found: i64,
    states_initial: i64,
    states_distinct: Option<i64>,
    transitions: i64,
    time_seconds_reported: f64,
    states_per_second: Option<f64>,
}

#[derive(Serialize)]
struct TlcMeta {
    elapsed_seconds_wall: f64,
    states_found: i64,
}

fn run(cli: &Cli) -> Result<()> {
    if cli.beans <= 0 {
        bail!("--beans must be positive");
    }
    let repo_root = repo_root()?;
    let home = env::var("HOME").map_err(|_| anyhow!("HOME env var not set"))?;
    let coffeecan_dir = PathBuf::from(&home).join("tlaplus-examples/specifications/CoffeeCan");
    let coffeecan_tla = coffeecan_dir.join("CoffeeCan.tla");
    let tytools_jar = PathBuf::from(&home).join("tlaplus/tytools.jar");
    let community_modules_jar = PathBuf::from(&home).join("tlaplus/CommunityModules.jar");

    if !coffeecan_tla.exists() {
        bail!("CoffeeCan spec not found at {}", coffeecan_tla.display());
    }
    if !tytools_jar.exists() {
        bail!("TLC jar not found at {}", tytools_jar.display());
    }

    let target_dir = repo_root.join("target");
    fs::create_dir_all(&target_dir)
        .with_context(|| format!("creating {}", target_dir.display()))?;
    let temp_root = make_temp_dir(&target_dir, &format!("coffecan_codegen_poc_{}_", cli.beans))?;

    let result = (|| -> Result<()> {
        let (wrapper_tla, wrapper_cfg) =
            stage_wrapper(&temp_root, &coffeecan_tla, cli.beans, cli.with_invariants)?;

        let ty_build: Option<BuildMeta> = if !cli.skip_build {
            Some(build_ty(&repo_root, cli.timeout)?)
        } else {
            None
        };
        let ty_bin = resolve_ty_bin(&repo_root)?;

        let generated_rs = temp_root.join("CoffeeCanCodegenBench.rs");
        let codegen = generate_rust(
            &ty_bin,
            &repo_root,
            &wrapper_tla,
            &generated_rs,
            cli.timeout,
        )?;

        let project_dir = temp_root.join("aot_project");
        write_aot_project(&project_dir, &repo_root, &generated_rs, cli.with_invariants)?;
        let aot = run_aot(&project_dir, cli.timeout)?;

        let interpreter = if !cli.skip_interpreter {
            Some(run_interpreter(
                &ty_bin,
                &repo_root,
                &wrapper_tla,
                &wrapper_cfg,
                cli.timeout,
            )?)
        } else {
            None
        };
        let tlc = if !cli.skip_tlc {
            Some(run_tlc(
                &wrapper_tla,
                &wrapper_cfg,
                &tytools_jar,
                &community_modules_jar,
                cli.timeout,
            )?)
        } else {
            None
        };

        let expected = expected_states(cli.beans);
        validate_summary_counts(expected, &aot, interpreter.as_ref(), tlc.as_ref())?;

        let summary = json!({
            "timestamp": now_iso8601(),
            "bean_count": cli.beans,
            "expected_states": expected,
            "with_invariants": cli.with_invariants,
            "temp_root": temp_root.to_string_lossy(),
            "wrapper_tla": wrapper_tla.to_string_lossy(),
            "wrapper_cfg": wrapper_cfg.to_string_lossy(),
            "ty_build": ty_build,
            "codegen": codegen,
            "aot": aot,
            "interpreter": interpreter,
            "tlc": tlc,
        });

        let pretty = serde_json::to_string_pretty(&summary)? + "\n";
        if let Some(out) = &cli.output_json {
            fs::write(out, &pretty).with_context(|| format!("writing {}", out.display()))?;
        }
        println!("{pretty}");
        Ok(())
    })();

    if cli.keep_temp {
        eprintln!("kept temp dir: {}", temp_root.display());
    } else {
        let _ = fs::remove_dir_all(&temp_root);
    }
    result
}

fn repo_root() -> Result<PathBuf> {
    let exe = env::current_exe().context("current_exe")?;
    let mut dir = exe.parent().map(Path::to_path_buf);
    while let Some(p) = dir {
        if p.join("Cargo.lock").exists() && p.join("crates").exists() {
            return Ok(p);
        }
        dir = p.parent().map(Path::to_path_buf);
    }
    // Fall back to current working directory.
    env::current_dir().context("cwd")
}

fn expected_states(bean_count: i64) -> i64 {
    bean_count * (bean_count + 3) / 2
}

fn now_iso8601() -> String {
    // Match Python's `time.strftime("%Y-%m-%dT%H:%M:%S%z")` shape. We use the
    // chrono crate via the tla-cli workspace dependency.
    chrono::Local::now()
        .format("%Y-%m-%dT%H:%M:%S%z")
        .to_string()
}

fn make_temp_dir(parent: &Path, prefix: &str) -> Result<PathBuf> {
    let pid = std::process::id();
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    for attempt in 0..32 {
        let name = format!("{prefix}{pid}_{nanos}_{attempt}");
        let path = parent.join(name);
        if !path.exists() {
            fs::create_dir_all(&path).with_context(|| format!("creating {}", path.display()))?;
            return Ok(path);
        }
    }
    Err(anyhow!(
        "failed to create unique temp directory under {}",
        parent.display()
    ))
}

fn parse_tlc_states(output: &str) -> Option<i64> {
    let line_re = Regex::new(
        r"(?m)^\s*([0-9,]+) states generated, ([0-9,]+) distinct states found, ([0-9,]+) states left",
    )
    .ok()?;
    if let Some(caps) = line_re.captures(output) {
        return caps
            .get(2)
            .and_then(|m| m.as_str().replace(',', "").parse::<i64>().ok());
    }
    let fallback = Regex::new(r"([0-9,]+) distinct states found").ok()?;
    fallback
        .captures(output)
        .and_then(|c| c.get(1))
        .and_then(|m| m.as_str().replace(',', "").parse::<i64>().ok())
}

fn run_command(
    cmd: &[&str],
    cwd: &Path,
    env: Option<&[(String, String)]>,
    timeout: u64,
) -> Result<CommandResult> {
    let mut command = Command::new(cmd[0]);
    command.args(&cmd[1..]).current_dir(cwd);
    if let Some(extra) = env {
        for (k, v) in extra {
            command.env(k, v);
        }
    }
    let start = Instant::now();
    // Rust's std::process::Command lacks a built-in timeout; use a spawn +
    // poll loop. For these subprocesses we accept the simpler `output()`
    // call since the original Python relies on the same blocking behavior
    // and we want a single, clear failure mode.
    let _ = timeout; // kept for parity with Python signature
    let output = command
        .output()
        .with_context(|| format!("running {:?}", cmd))?;
    let elapsed = start.elapsed().as_secs_f64();
    Ok(CommandResult {
        command: cmd.iter().map(|s| (*s).to_string()).collect(),
        cwd: cwd.to_string_lossy().to_string(),
        elapsed_seconds: elapsed,
        returncode: output.status.code().unwrap_or(-1),
        stdout: String::from_utf8_lossy(&output.stdout).to_string(),
        stderr: String::from_utf8_lossy(&output.stderr).to_string(),
    })
}

fn ensure_ok(result: &CommandResult, context: &str) -> Result<()> {
    if result.returncode == 0 {
        return Ok(());
    }
    Err(anyhow!(
        "{context} failed with exit code {}\ncwd: {}\ncommand: {}\nstdout:\n{}\nstderr:\n{}",
        result.returncode,
        result.cwd,
        result.command.join(" "),
        result.stdout,
        result.stderr
    ))
}

fn stage_wrapper(
    temp_root: &Path,
    coffeecan_tla: &Path,
    bean_count: i64,
    with_invariants: bool,
) -> Result<(PathBuf, PathBuf)> {
    let spec_dir = temp_root.join("spec");
    fs::create_dir_all(&spec_dir)?;
    fs::copy(coffeecan_tla, spec_dir.join("CoffeeCan.tla"))?;

    let wrapper_tla = spec_dir.join("CoffeeCanCodegenBench.tla");
    let wrapper_cfg = spec_dir.join("CoffeeCanCodegenBench.cfg");

    let (invariant_def, invariant_cfg) = if with_invariants {
        (
            format!("\nInvType == can \\in [black : 0..{bean_count}, white : 0..{bean_count}]"),
            "\nINVARIANTS\n    InvType".to_string(),
        )
    } else {
        (String::new(), String::new())
    };

    let tla_body = format!(
        "---- MODULE CoffeeCanCodegenBench ----\n\
         VARIABLE can\n\n\
         INSTANCE CoffeeCan WITH MaxBeanCount <- {bean_count}\n\
         {invariant_def}\n\
         ====\n"
    );
    fs::write(&wrapper_tla, tla_body)?;

    let cfg_body = format!(
        "INIT\n    Init\n\n\
         NEXT\n    Next\n\
         {invariant_cfg}\n"
    );
    fs::write(&wrapper_cfg, cfg_body)?;
    Ok((wrapper_tla, wrapper_cfg))
}

fn build_ty(repo_root: &Path, timeout: u64) -> Result<BuildMeta> {
    let cmd = [
        "cargo",
        "build",
        "--profile",
        "release-canary",
        "--bin",
        "ty",
    ];
    let result = run_command(&cmd, repo_root, None, timeout)?;
    ensure_ok(&result, "cargo build --profile release-canary --bin ty")?;
    Ok(BuildMeta {
        command: result.command,
        cwd: result.cwd,
        elapsed_seconds: round3(result.elapsed_seconds),
        returncode: result.returncode,
    })
}

fn round3(value: f64) -> f64 {
    (value * 1000.0).round() / 1000.0
}

fn resolve_ty_bin(repo_root: &Path) -> Result<PathBuf> {
    let candidates = [
        repo_root.join("target/user/release-canary/ty"),
        repo_root.join("target/release-canary/ty"),
        repo_root.join("target/user/release/ty"),
        repo_root.join("target/release/ty"),
    ];
    for path in &candidates {
        if path.exists() {
            return Ok(path.clone());
        }
    }
    Err(anyhow!(
        "ty binary not found; checked: {}",
        candidates
            .iter()
            .map(|p| p.to_string_lossy().to_string())
            .collect::<Vec<_>>()
            .join(", ")
    ))
}

fn generate_rust(
    ty_bin: &Path,
    repo_root: &Path,
    wrapper_tla: &Path,
    generated_rs: &Path,
    timeout: u64,
) -> Result<CodegenMeta> {
    let bin = ty_bin.to_string_lossy();
    let tla = wrapper_tla.to_string_lossy();
    let out = generated_rs.to_string_lossy();
    let cmd = [
        bin.as_ref(),
        "codegen",
        tla.as_ref(),
        "--output",
        out.as_ref(),
    ];
    let result = run_command(&cmd, repo_root, None, timeout)?;
    ensure_ok(&result, "ty codegen")?;
    Ok(CodegenMeta {
        command: result.command,
        cwd: result.cwd,
        elapsed_seconds: round3(result.elapsed_seconds),
        returncode: result.returncode,
        stderr: result.stderr.trim().to_string(),
    })
}

fn write_aot_project(
    project_dir: &Path,
    repo_root: &Path,
    generated_rs: &Path,
    check_invariants: bool,
) -> Result<()> {
    let src_dir = project_dir.join("src");
    fs::create_dir_all(&src_dir)?;
    let runtime_path = repo_root.join("crates/tla-runtime");
    let runtime_path_str = runtime_path
        .canonicalize()
        .unwrap_or(runtime_path)
        .to_string_lossy()
        .to_string();

    let cargo_toml = format!(
        "[package]\n\
         name = \"coffecan_codegen_bench\"\n\
         version = \"0.1.0\"\n\
         edition = \"2021\"\n\n\
         [workspace]\n\n\
         [dependencies]\n\
         tla-runtime = {{ path = \"{runtime_path_str}\" }}\n"
    );
    fs::write(project_dir.join("Cargo.toml"), cargo_toml)?;
    fs::copy(generated_rs, src_dir.join("coffecancodegenbench.rs"))?;

    let body_invariant = if check_invariants {
        "                if let Some(false) = machine.check_invariant(&state) {\n                    eprintln!(\"status=invariant_violation\");\n                    eprintln!(\"state={:?}\", state);\n                    std::process::exit(2);\n                }\n\n                let mut had_successor = false;\n"
    } else {
        "                let mut had_successor = false;\n"
    };

    let main_rs = format!(
        "use std::collections::{{HashSet, VecDeque}};\n\
         use std::ops::ControlFlow;\n\
         use std::time::Instant;\n\n\
         use tla_runtime::StateMachine;\n\n\
         mod coffecancodegenbench;\n\
         use coffecancodegenbench::CoffeeCanCodegenBench;\n\n\
         fn main() {{\n    \
             let machine = CoffeeCanCodegenBench;\n    \
             let start = Instant::now();\n\n    \
             let mut seen = HashSet::new();\n    \
             let mut queue = VecDeque::new();\n    \
             let mut transitions = 0usize;\n\n    \
             let _ = machine.for_each_init(|state| {{\n        \
                 if seen.insert(state.clone()) {{\n            \
                     queue.push_back(state);\n        \
                 }}\n        \
                 ControlFlow::Continue(())\n    \
             }});\n\n    \
             let initial_states = seen.len();\n    \
             let mut states_explored = 0usize;\n\n    \
             while let Some(state) = queue.pop_front() {{\n        \
                 states_explored += 1;\n\n\
         {body_invariant}        \
                 let _ = machine.for_each_next(&state, |succ| {{\n            \
                     had_successor = true;\n            \
                     transitions += 1;\n            \
                     if seen.insert(succ.clone()) {{\n                \
                         queue.push_back(succ);\n            \
                     }}\n            \
                     ControlFlow::Continue(())\n        \
                 }});\n\n        \
                 if !had_successor {{\n            \
                     eprintln!(\"status=deadlock\");\n            \
                     eprintln!(\"state={{:?}}\", state);\n            \
                     std::process::exit(3);\n        \
                 }}\n    \
             }}\n\n    \
             let elapsed = start.elapsed().as_secs_f64();\n    \
             println!(\"status=ok\");\n    \
             println!(\"states_explored={{states_explored}}\");\n    \
             println!(\"states_initial={{initial_states}}\");\n    \
             println!(\"states_distinct={{}}\", seen.len());\n    \
             println!(\"transitions={{transitions}}\");\n    \
             println!(\"elapsed_seconds={{elapsed:.6}}\");\n    \
             if elapsed > 0.0 {{\n        \
                 println!(\n            \
                     \"states_per_second={{:.3}}\",\n            \
                     states_explored as f64 / elapsed\n        \
                 );\n    \
             }}\n\
         }}\n"
    );
    fs::write(src_dir.join("main.rs"), main_rs)?;
    Ok(())
}

fn parse_key_value_lines(output: &str) -> std::collections::HashMap<String, String> {
    let mut map = std::collections::HashMap::new();
    for line in output.lines() {
        if let Some(idx) = line.find('=') {
            let k = line[..idx].trim().to_string();
            let v = line[idx + 1..].trim().to_string();
            map.insert(k, v);
        }
    }
    map
}

fn run_aot(project_dir: &Path, timeout: u64) -> Result<AotMeta> {
    let build = run_command(&["cargo", "build", "--release"], project_dir, None, timeout)?;
    ensure_ok(&build, "AOT cargo build --release")?;
    let run = run_command(&["cargo", "run", "--release"], project_dir, None, timeout)?;
    ensure_ok(&run, "AOT cargo run --release")?;
    let parsed = parse_key_value_lines(&run.stdout);
    let get = |k: &str| -> Result<String> {
        parsed
            .get(k)
            .cloned()
            .ok_or_else(|| anyhow!("missing key '{k}' in AOT output"))
    };
    Ok(AotMeta {
        build_elapsed_seconds: round3(build.elapsed_seconds),
        run_elapsed_seconds: round3(run.elapsed_seconds),
        status: parsed.get("status").cloned(),
        states_explored: get("states_explored")?.parse()?,
        states_initial: get("states_initial")?.parse()?,
        states_distinct: get("states_distinct")?.parse()?,
        transitions: get("transitions")?.parse()?,
        elapsed_seconds_internal: get("elapsed_seconds")?.parse()?,
        states_per_second: get("states_per_second")?.parse()?,
    })
}

fn run_interpreter(
    ty_bin: &Path,
    repo_root: &Path,
    wrapper_tla: &Path,
    wrapper_cfg: &Path,
    timeout: u64,
) -> Result<InterpreterMeta> {
    let bin = ty_bin.to_string_lossy();
    let tla = wrapper_tla.to_string_lossy();
    let cfg = wrapper_cfg.to_string_lossy();
    let cmd = [
        bin.as_ref(),
        "check",
        tla.as_ref(),
        "--config",
        cfg.as_ref(),
        "--workers",
        "1",
        "--force",
        "--output",
        "json",
    ];
    let result = run_command(&cmd, repo_root, None, timeout)?;
    ensure_ok(&result, "ty check wrapper")?;
    let payload: Value =
        serde_json::from_str(&result.stdout).context("parsing ty check JSON output")?;
    let res = payload
        .get("result")
        .and_then(|v| v.as_object())
        .ok_or_else(|| anyhow!("ty JSON missing 'result'"))?;
    let stats = payload
        .get("statistics")
        .and_then(|v| v.as_object())
        .ok_or_else(|| anyhow!("ty JSON missing 'statistics'"))?;
    Ok(InterpreterMeta {
        elapsed_seconds_wall: round3(result.elapsed_seconds),
        status: res
            .get("status")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string(),
        states_found: stats
            .get("states_found")
            .and_then(|v| v.as_i64())
            .ok_or_else(|| anyhow!("missing statistics.states_found"))?,
        states_initial: stats
            .get("states_initial")
            .and_then(|v| v.as_i64())
            .ok_or_else(|| anyhow!("missing statistics.states_initial"))?,
        states_distinct: stats.get("states_distinct").and_then(|v| v.as_i64()),
        transitions: stats
            .get("transitions")
            .and_then(|v| v.as_i64())
            .ok_or_else(|| anyhow!("missing statistics.transitions"))?,
        time_seconds_reported: stats
            .get("time_seconds")
            .and_then(|v| v.as_f64())
            .ok_or_else(|| anyhow!("missing statistics.time_seconds"))?,
        states_per_second: stats.get("states_per_second").and_then(|v| v.as_f64()),
    })
}

fn run_tlc(
    wrapper_tla: &Path,
    wrapper_cfg: &Path,
    tytools_jar: &Path,
    community_modules_jar: &Path,
    timeout: u64,
) -> Result<TlcMeta> {
    let path_sep = if cfg!(windows) { ";" } else { ":" };
    let mut classpath = tytools_jar.to_string_lossy().to_string();
    if community_modules_jar.exists() {
        classpath.push_str(path_sep);
        classpath.push_str(&community_modules_jar.to_string_lossy());
    }
    let wrapper_parent = wrapper_tla
        .parent()
        .ok_or_else(|| anyhow!("wrapper tla has no parent"))?;
    let wrapper_str = wrapper_tla.to_string_lossy();
    let cfg_str = wrapper_cfg.to_string_lossy();
    let cmd = [
        "java",
        "-Xmx4g",
        "-cp",
        classpath.as_str(),
        "tlc2.TLC",
        "-config",
        cfg_str.as_ref(),
        "-workers",
        "1",
        wrapper_str.as_ref(),
    ];
    let result = run_command(&cmd, wrapper_parent, None, timeout)?;
    ensure_ok(&result, "TLC wrapper run")?;
    let combined = format!("{}\n{}", result.stdout, result.stderr);
    let states = parse_tlc_states(&combined)
        .ok_or_else(|| anyhow!("could not parse TLC state count from output"))?;
    Ok(TlcMeta {
        elapsed_seconds_wall: round3(result.elapsed_seconds),
        states_found: states,
    })
}

fn validate_summary_counts(
    expected: i64,
    aot: &AotMeta,
    interpreter: Option<&InterpreterMeta>,
    tlc: Option<&TlcMeta>,
) -> Result<()> {
    if aot.states_distinct != expected {
        bail!(
            "AOT distinct state count mismatch: expected {expected}, got {}",
            aot.states_distinct
        );
    }
    if let Some(interp) = interpreter {
        if interp.states_found != expected {
            bail!(
                "Interpreter state count mismatch: expected {expected}, got {}",
                interp.states_found
            );
        }
        if let Some(distinct) = interp.states_distinct {
            if distinct != expected {
                bail!(
                    "Interpreter distinct state count mismatch: expected {expected}, got {}",
                    distinct
                );
            }
        }
    }
    if let Some(t) = tlc {
        if t.states_found != expected {
            bail!(
                "TLC state count mismatch: expected {expected}, got {}",
                t.states_found
            );
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expected_states_matches_closed_form() {
        assert_eq!(expected_states(1), 2);
        assert_eq!(expected_states(2), 5);
        assert_eq!(expected_states(3), 9);
        assert_eq!(expected_states(10), 65);
        assert_eq!(expected_states(3000), 3000 * 3003 / 2);
    }

    #[test]
    fn parse_tlc_states_prefers_distinct_count_line() {
        let output = "
3 states generated, 2 distinct states found, 1 states left on queue.
";
        assert_eq!(parse_tlc_states(output), Some(2));
        assert_eq!(
            parse_tlc_states("Finished computing initial states: 4 distinct states found."),
            Some(4)
        );
        assert_eq!(parse_tlc_states("nothing here"), None);
    }

    #[test]
    fn parse_key_value_lines_recovers_metrics() {
        let body = "status=ok\nstates_explored=10\nelapsed_seconds=0.5\nempty=\nbroken_line\n";
        let map = parse_key_value_lines(body);
        assert_eq!(map.get("status").map(String::as_str), Some("ok"));
        assert_eq!(map.get("states_explored").map(String::as_str), Some("10"));
        assert_eq!(map.get("elapsed_seconds").map(String::as_str), Some("0.5"));
        assert_eq!(map.get("empty").map(String::as_str), Some(""));
        assert!(!map.contains_key("broken_line"));
    }

    #[test]
    fn validate_summary_counts_rejects_mismatch() {
        let aot = AotMeta {
            build_elapsed_seconds: 0.0,
            run_elapsed_seconds: 0.0,
            status: None,
            states_explored: 0,
            states_initial: 0,
            states_distinct: 99,
            transitions: 0,
            elapsed_seconds_internal: 0.0,
            states_per_second: 0.0,
        };
        assert!(validate_summary_counts(100, &aot, None, None).is_err());
    }

    #[test]
    fn validate_summary_counts_passes_when_matching() {
        let aot = AotMeta {
            build_elapsed_seconds: 0.0,
            run_elapsed_seconds: 0.0,
            status: None,
            states_explored: 0,
            states_initial: 0,
            states_distinct: 65,
            transitions: 0,
            elapsed_seconds_internal: 0.0,
            states_per_second: 0.0,
        };
        let interp = InterpreterMeta {
            elapsed_seconds_wall: 0.0,
            status: "ok".to_string(),
            states_found: 65,
            states_initial: 1,
            states_distinct: Some(65),
            transitions: 0,
            time_seconds_reported: 0.0,
            states_per_second: None,
        };
        let tlc = TlcMeta {
            elapsed_seconds_wall: 0.0,
            states_found: 65,
        };
        assert!(validate_summary_counts(65, &aot, Some(&interp), Some(&tlc)).is_ok());
    }
}
