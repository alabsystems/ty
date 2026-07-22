// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! `ty trace mine` — spec mining v1.
//!
//! Mines a CANDIDATE TLA+ module (+ config) from observed trace corpora and
//! closes the loop: the emitted spec is model checked and the input traces
//! are re-validated against it. Candidates refuted by a counterexample are
//! dropped and the check re-runs (counterexample-guided pruning).
//!
//! Everything emitted here is a hypothesis generalized from finitely many
//! observations, for HUMAN REVIEW — not ground truth. See docs/trace-mining.md.

use std::fs;
use std::io::Cursor;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{bail, Context, Result};
use tla_check::{
    bind_constants_from_config, mine_spec, read_trace_events, render_config, render_module,
    resolve_trace_input_format, ActionLabelMode, CheckResult, Config, EvalCtx, MineOptions,
    MinedSpec, MiningTrace, ModelChecker, TraceActionLabel, TraceEventSink, TraceHeader,
    TraceInputFormat, TraceInputFormatSelection, TraceSourceHint, TraceStep, TraceValidationEngine,
};
use tla_core::ast::Unit;
use tla_core::{lower_main_module, FileId, ModuleLoader};

use crate::{parse_or_report, read_source};

use super::TraceInputFormatArg;

/// Arguments for `ty trace mine`.
pub(crate) struct MineArgs {
    /// Trace input files.
    pub files: Vec<PathBuf>,
    /// Input format selection.
    pub input_format: TraceInputFormatArg,
    /// Mined module name (and output file stem).
    pub module_name: String,
    /// Output directory.
    pub out: PathBuf,
    /// Integer-domain enumeration threshold.
    pub max_domain_enum: usize,
    /// Maximum refinement rounds.
    pub max_rounds: usize,
    /// State bound for the verification check.
    pub max_states: usize,
    /// Skip the check/validate loop.
    pub skip_verify: bool,
}

/// A pruned candidate, for the report.
struct DroppedCandidate {
    name: String,
    round: usize,
    reason: String,
}

/// Outcome of one model-checking round on the mined spec.
enum CheckRound {
    /// Reachable state space exhausted; all remaining candidates hold.
    Clean { states: usize },
    /// State bound hit without a violation: candidates hold within the bound.
    Bounded { states: usize },
    /// A named invariant/property candidate was refuted.
    Violated {
        candidate: String,
        kind: &'static str,
        detail: String,
    },
    /// The check could not complete (setup error, vacuous basis, ...).
    Failed { message: String },
}

/// Final check status for the report.
enum CheckStatus {
    Skipped,
    Clean { states: usize, rounds: usize },
    Bounded { states: usize, rounds: usize },
}

/// Per-trace validation outcome for the report.
struct TraceOutcome {
    name: String,
    steps: usize,
    result: Result<usize, String>, // Ok(warning count) | Err(message)
}

/// Entry point for `ty trace mine`.
pub(crate) fn cmd_trace_mine(args: &MineArgs) -> Result<()> {
    if !is_identifier(&args.module_name) {
        bail!(
            "--module-name {:?} is not a valid TLA+ module name",
            args.module_name
        );
    }
    if args.max_rounds == 0 {
        bail!("--max-rounds must be at least 1");
    }
    // The report is the product here; suppress the checker's progress telemetry.
    tla_check::set_telemetry_quiet(true);

    // --- 1. Load the corpus ---
    let mut traces: Vec<MiningTrace> = Vec::new();
    for file in &args.files {
        let loaded = load_traces(file, args.input_format)
            .with_context(|| format!("load traces from {}", file.display()))?;
        traces.extend(loaded);
    }
    if traces.is_empty() {
        bail!("no traces found in the given file(s)");
    }

    // --- 2. Mine the candidate spec ---
    let options = MineOptions {
        module_name: args.module_name.clone(),
        max_domain_enum: args.max_domain_enum,
        ..MineOptions::default()
    };
    let mut spec = mine_spec(&traces, &options).context("spec mining failed")?;

    // --- 3. Emit module + config ---
    fs::create_dir_all(&args.out)
        .with_context(|| format!("create output directory {}", args.out.display()))?;
    let spec_path = args.out.join(format!("{}.tla", args.module_name));
    let cfg_path = args.out.join(format!("{}.cfg", args.module_name));
    write_spec(&spec, &spec_path, &cfg_path)?;

    // --- 4. The closed loop: check, prune refuted candidates, re-check ---
    let mut dropped: Vec<DroppedCandidate> = Vec::new();
    let check_status = if args.skip_verify {
        CheckStatus::Skipped
    } else {
        run_refinement_loop(
            &mut spec,
            &spec_path,
            &cfg_path,
            args.max_rounds,
            args.max_states,
            &mut dropped,
        )?
    };

    // --- 5. Re-validate the input corpus against the mined spec ---
    let outcomes = if args.skip_verify {
        Vec::new()
    } else {
        validate_corpus(&spec_path, &cfg_path, &traces)?
    };

    // --- 6. Report ---
    print_report(
        &spec,
        &traces,
        &dropped,
        &check_status,
        &outcomes,
        &spec_path,
        &cfg_path,
    );

    let failed: Vec<&TraceOutcome> = outcomes.iter().filter(|o| o.result.is_err()).collect();
    if !failed.is_empty() {
        bail!(
            "mined spec did not validate {} of {} input trace(s) — the mined \
             candidates do not reproduce their own corpus",
            failed.len(),
            outcomes.len()
        );
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Corpus loading
// ---------------------------------------------------------------------------

/// Sink collecting a whole trace into memory.
#[derive(Default)]
struct CollectSink {
    header: Option<TraceHeader>,
    steps: Vec<TraceStep>,
}

impl TraceEventSink for CollectSink {
    fn on_header(&mut self, header: TraceHeader) {
        self.header = Some(header);
    }
    fn on_step(&mut self, step: TraceStep) {
        self.steps.push(step);
    }
}

/// Load all traces from one file: `ty` trace format (JSON/JSONL) or
/// `ty trace-gen --format json` output (auto-detected).
fn load_traces(path: &Path, input_format: TraceInputFormatArg) -> Result<Vec<MiningTrace>> {
    let selection = TraceInputFormatSelection::from(input_format);
    let format = resolve_trace_input_format(selection, TraceSourceHint::Path(path));
    let content = fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    let file_name = path
        .file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| path.display().to_string());

    // trace-gen envelopes are JSON objects tagged with `"tool": "ty trace-gen"`.
    if format == TraceInputFormat::Json {
        if let Ok(json) = serde_json::from_str::<serde_json::Value>(&content) {
            if json.get("tool").and_then(|t| t.as_str()) == Some("ty trace-gen") {
                return convert_tracegen_envelope(&file_name, &json);
            }
        }
    }

    let mut sink = CollectSink::default();
    read_trace_events(Cursor::new(content.as_bytes()), format, &mut sink)
        .with_context(|| format!("parse trace {} as {:?}", path.display(), format))?;
    let header = sink
        .header
        .context("trace parser did not deliver a header")?;
    Ok(vec![MiningTrace {
        name: file_name,
        variables: header.variables,
        steps: sink.steps,
    }])
}

/// Convert a `ty trace-gen --format json` envelope into mining traces.
///
/// Placeholder action names emitted by trace-gen ("Initial predicate",
/// "Action") are treated as absent labels.
fn convert_tracegen_envelope(
    file_name: &str,
    json: &serde_json::Value,
) -> Result<Vec<MiningTrace>> {
    let traces_json = json
        .get("traces")
        .and_then(|t| t.as_array())
        .context("trace-gen envelope has no `traces` array")?;

    let mut traces = Vec::new();
    for (ordinal, trace_json) in traces_json.iter().enumerate() {
        let states = trace_json
            .get("states")
            .and_then(|s| s.as_array())
            .with_context(|| format!("trace-gen trace {ordinal} has no `states` array"))?;
        if states.is_empty() {
            continue;
        }

        let mut variables: Vec<String> = Vec::new();
        let mut steps = Vec::with_capacity(states.len());
        for (idx, state_json) in states.iter().enumerate() {
            let vars_json = state_json
                .get("variables")
                .and_then(|v| v.as_object())
                .with_context(|| {
                    format!("trace-gen trace {ordinal} state {idx} has no `variables` object")
                })?;
            let mut state = std::collections::HashMap::with_capacity(vars_json.len());
            for (name, value_json) in vars_json {
                if !variables.contains(name) {
                    variables.push(name.clone());
                }
                let value = serde_json::from_value(value_json.clone()).with_context(|| {
                    format!(
                        "trace-gen trace {ordinal} state {idx}: variable {name:?} \
                         is not a typed trace value"
                    )
                })?;
                state.insert(name.clone(), value);
            }
            let action = if idx == 0 {
                None
            } else {
                state_json
                    .get("action")
                    .and_then(|a| a.as_str())
                    .filter(|name| *name != "Initial predicate" && *name != "Action")
                    .map(|name| TraceActionLabel {
                        name: name.to_string(),
                        params: None,
                    })
            };
            steps.push(TraceStep {
                index: Some(idx),
                state,
                action,
            });
        }

        let trace_id = trace_json
            .get("trace_id")
            .and_then(serde_json::Value::as_u64)
            .map_or_else(|| (ordinal + 1).to_string(), |id| id.to_string());
        traces.push(MiningTrace {
            name: format!("{file_name}#{trace_id}"),
            variables,
            steps,
        });
    }
    if traces.is_empty() {
        bail!("trace-gen envelope contains no non-empty traces");
    }
    Ok(traces)
}

// ---------------------------------------------------------------------------
// The closed loop
// ---------------------------------------------------------------------------

/// Write the current candidate spec to disk.
fn write_spec(spec: &MinedSpec, spec_path: &Path, cfg_path: &Path) -> Result<()> {
    fs::write(spec_path, render_module(spec))
        .with_context(|| format!("write {}", spec_path.display()))?;
    fs::write(cfg_path, render_config(spec))
        .with_context(|| format!("write {}", cfg_path.display()))?;
    Ok(())
}

/// Check the mined spec; drop refuted candidates and re-check, up to
/// `max_rounds` rounds.
fn run_refinement_loop(
    spec: &mut MinedSpec,
    spec_path: &Path,
    cfg_path: &Path,
    max_rounds: usize,
    max_states: usize,
    dropped: &mut Vec<DroppedCandidate>,
) -> Result<CheckStatus> {
    for round in 1..=max_rounds {
        let outcome = run_check_round(spec_path, cfg_path, max_states)
            .with_context(|| format!("model check of the mined spec (round {round})"))?;
        match outcome {
            CheckRound::Clean { states } => {
                return Ok(CheckStatus::Clean {
                    states,
                    rounds: round,
                });
            }
            CheckRound::Bounded { states } => {
                eprintln!(
                    "Note: state bound ({max_states}) reached; surviving candidates \
                     hold within the explored bound only"
                );
                return Ok(CheckStatus::Bounded {
                    states,
                    rounds: round,
                });
            }
            CheckRound::Violated {
                candidate,
                kind,
                detail,
            } => {
                if !spec.drop_candidate(&candidate) {
                    bail!(
                        "model check refuted {kind} {candidate:?}, which is not a mined \
                         candidate — the mined actions themselves are inconsistent ({detail})"
                    );
                }
                eprintln!("round {round}: dropped {kind} {candidate} ({detail})");
                dropped.push(DroppedCandidate {
                    name: candidate,
                    round,
                    reason: format!("{kind} refuted: {detail}"),
                });
                write_spec(spec, spec_path, cfg_path)?;
            }
            CheckRound::Failed { message } => {
                bail!("model check of the mined spec failed: {message}");
            }
        }
    }
    bail!(
        "counterexample-guided pruning did not converge within {max_rounds} round(s); \
         re-run with a larger --max-rounds"
    );
}

/// Parse + lower a mined module and load its (stdlib) imports.
struct LoadedModule {
    module: tla_core::ast::Module,
    loader: ModuleLoader,
    config: Config,
}

/// Load the mined spec and its config from disk.
fn load_mined_spec(spec_path: &Path, cfg_path: &Path) -> Result<LoadedModule> {
    let source = read_source(spec_path)?;
    let tree = parse_or_report(spec_path, &source)?;

    let hint_name = spec_path
        .file_stem()
        .and_then(|s| s.to_str())
        .filter(|s| !s.is_empty());
    let lower_result = lower_main_module(FileId(0), &tree, hint_name);
    if !lower_result.errors.is_empty() {
        let diags: Vec<String> = lower_result
            .errors
            .iter()
            .map(|err| format!("{}..{}: {}", err.span.start, err.span.end, err.message))
            .collect();
        bail!(
            "mined module {} failed to lower (mining bug):\n{}",
            spec_path.display(),
            diags.join("\n")
        );
    }
    let module = lower_result.module.context("lower produced no module")?;

    let mut loader = ModuleLoader::new(spec_path);
    loader.seed_from_syntax_tree(&tree, spec_path);
    loader
        .load_extends(&module)
        .context("failed to load extended modules")?;
    loader
        .load_instances(&module)
        .context("failed to load instanced modules")?;

    let config_source = fs::read_to_string(cfg_path)
        .with_context(|| format!("read config {}", cfg_path.display()))?;
    let config = Config::parse(&config_source).map_err(|errors| {
        let diags: Vec<String> = errors
            .iter()
            .map(|err| format!("{}:{}: {}", cfg_path.display(), err.line(), err))
            .collect();
        anyhow::anyhow!(
            "mined config failed to parse (mining bug):\n{}",
            diags.join("\n")
        )
    })?;

    Ok(LoadedModule {
        module,
        loader,
        config,
    })
}

/// One `ty check` run over the mined spec.
fn run_check_round(spec_path: &Path, cfg_path: &Path, max_states: usize) -> Result<CheckRound> {
    let loaded = load_mined_spec(spec_path, cfg_path)?;
    let checker_modules = loaded.loader.modules_for_model_checking(&loaded.module);
    let runtime_config = loaded.config.runtime_model_config();

    let mut checker =
        ModelChecker::new_with_extends(&loaded.module, &checker_modules, &runtime_config);
    checker.set_max_states(max_states);
    checker.register_file_path(FileId(0), spec_path.to_path_buf());

    Ok(match checker.check() {
        CheckResult::Success(stats) => CheckRound::Clean {
            states: stats.states_found,
        },
        CheckResult::LimitReached { stats, .. } => CheckRound::Bounded {
            states: stats.states_found,
        },
        CheckResult::InvariantViolation {
            invariant, trace, ..
        } => CheckRound::Violated {
            candidate: invariant,
            kind: "invariant",
            detail: format!("counterexample of {} state(s)", trace.len()),
        },
        CheckResult::PropertyViolation {
            property, trace, ..
        } => CheckRound::Violated {
            candidate: property,
            kind: "property",
            detail: format!("counterexample of {} state(s)", trace.len()),
        },
        CheckResult::LivenessViolation {
            property,
            prefix,
            cycle,
            ..
        } => CheckRound::Violated {
            candidate: property,
            kind: "property",
            detail: format!(
                "lasso counterexample ({} prefix + {} cycle states)",
                prefix.len(),
                cycle.len()
            ),
        },
        CheckResult::Deadlock { trace, .. } => CheckRound::Failed {
            message: format!(
                "unexpected deadlock after {} state(s) despite CHECK_DEADLOCK FALSE",
                trace.len()
            ),
        },
        CheckResult::Vacuous { reason, .. } => CheckRound::Failed {
            message: format!("vacuous check: {reason:?}"),
        },
        CheckResult::Error { error, .. } => CheckRound::Failed {
            message: error.to_string(),
        },
        // CheckResult is #[non_exhaustive]; fail closed on future variants.
        other => CheckRound::Failed {
            message: format!("unrecognized check result: {other:?}"),
        },
    })
}

// ---------------------------------------------------------------------------
// Corpus re-validation
// ---------------------------------------------------------------------------

/// Validate every input trace against the mined spec
/// (partial observations allowed, action-label mismatches as warnings).
fn validate_corpus(
    spec_path: &Path,
    cfg_path: &Path,
    traces: &[MiningTrace],
) -> Result<Vec<TraceOutcome>> {
    let loaded = load_mined_spec(spec_path, cfg_path)?;
    let checker_modules = loaded.loader.modules_for_model_checking(&loaded.module);

    let mut ctx = EvalCtx::new();
    ctx.load_module(&loaded.module);
    for module in &checker_modules {
        ctx.load_module(module);
    }
    bind_constants_from_config(&mut ctx, &loaded.config)
        .context("failed to bind constants from the mined config")?;

    let init_name = loaded.config.init.as_deref().unwrap_or("Init");
    let next_name = loaded.config.next.as_deref().unwrap_or("Next");
    let init_def = ctx
        .get_op(init_name)
        .with_context(|| format!("Init operator {init_name:?} not found in mined spec"))?
        .clone();
    let next_def = ctx
        .get_op(next_name)
        .with_context(|| format!("Next operator {next_name:?} not found in mined spec"))?
        .clone();

    let vars = collect_state_vars(&loaded.module, &checker_modules);
    if vars.is_empty() {
        bail!("mined spec declares no state variables (mining bug)");
    }

    let mut outcomes = Vec::with_capacity(traces.len());
    for trace in traces {
        let mut engine = TraceValidationEngine::new(&mut ctx, &init_def, &next_def, vars.clone())
            .with_action_label_mode(ActionLabelMode::Warn)
            .with_allow_partial_observations(true);
        let result = match engine.validate_trace(trace.steps.clone()) {
            Ok(success) => Ok(success.warnings.len()),
            Err(err) => Err(err.to_string()),
        };
        outcomes.push(TraceOutcome {
            name: trace.name.clone(),
            steps: trace.steps.len(),
            result,
        });
    }
    Ok(outcomes)
}

/// Collect state variables from the mined module (and any loaded modules).
fn collect_state_vars(
    module: &tla_core::ast::Module,
    checker_modules: &[&tla_core::ast::Module],
) -> Vec<Arc<str>> {
    let mut vars: Vec<Arc<str>> = Vec::new();
    let mut push_from = |m: &tla_core::ast::Module| {
        for unit in &m.units {
            if let Unit::Variable(var_names) = &unit.node {
                for var in var_names {
                    if !vars.iter().any(|v| v.as_ref() == var.node.as_str()) {
                        vars.push(Arc::from(var.node.as_str()));
                    }
                }
            }
        }
    };
    for ext in checker_modules {
        push_from(ext);
    }
    push_from(module);
    vars
}

// ---------------------------------------------------------------------------
// Report
// ---------------------------------------------------------------------------

/// Print the exit report: mined artifacts, surviving/dropped candidates,
/// check status, and per-trace validation results.
fn print_report(
    spec: &MinedSpec,
    traces: &[MiningTrace],
    dropped: &[DroppedCandidate],
    check_status: &CheckStatus,
    outcomes: &[TraceOutcome],
    spec_path: &Path,
    cfg_path: &Path,
) {
    println!("ty trace mine — candidate spec report");
    println!(
        "corpus: {} trace(s), {} steps",
        traces.len(),
        spec.total_steps
    );

    println!("\nvariables ({}):", spec.variables.len());
    for dom in &spec.domains {
        println!("  {} \\in {} — {}", dom.var, dom.expr, dom.description);
    }

    println!("\nactions ({}):", spec.actions.len());
    for action in &spec.actions {
        let origin = match &action.cluster {
            tla_check::ClusterKind::Label(label) => format!("label {label:?}"),
            tla_check::ClusterKind::ChangedVars(vars) if vars.is_empty() => {
                "no observed change".to_string()
            }
            tla_check::ClusterKind::ChangedVars(vars) => {
                format!("changed {{{}}}", vars.join(", "))
            }
        };
        println!(
            "  {} ({origin}, {} instance(s))",
            action.name, action.instances
        );
        for guard in &action.guards {
            println!("    guard: {guard}");
        }
        for update in &action.updates {
            for conjunct in &update.conjuncts {
                println!("    {conjunct}");
            }
        }
        if !action.unchanged.is_empty() {
            println!("    UNCHANGED <<{}>>", action.unchanged.join(", "));
        }
    }

    println!(
        "\ninvariant candidates ({} survived):",
        spec.invariants.len()
    );
    for inv in &spec.invariants {
        println!(
            "  survived {} == {} — {}",
            inv.name, inv.def, inv.description
        );
    }
    println!(
        "\nproperty candidates ({} survived):",
        spec.properties.len()
    );
    for prop in &spec.properties {
        println!(
            "  survived {} == {} — {}",
            prop.name, prop.def, prop.description
        );
    }
    if !dropped.is_empty() {
        println!("\ndropped candidates ({}):", dropped.len());
        for drop in dropped {
            println!(
                "  dropped {} (round {}: {})",
                drop.name, drop.round, drop.reason
            );
        }
    }

    if !spec.notes.is_empty() {
        println!("\nmining notes:");
        for note in &spec.notes {
            println!("  - {note}");
        }
    }

    match check_status {
        CheckStatus::Skipped => println!("\ncheck: SKIPPED (--skip-verify)"),
        CheckStatus::Clean { states, rounds } => {
            println!("\ncheck: PASS — {states} state(s) explored, {rounds} round(s)")
        }
        CheckStatus::Bounded { states, rounds } => println!(
            "\ncheck: PASS WITHIN BOUND — state limit reached at {states} state(s), \
             {rounds} round(s); candidates hold within the explored bound only"
        ),
    }

    if !outcomes.is_empty() {
        println!("\nvalidation (input traces vs mined spec):");
        for outcome in outcomes {
            match &outcome.result {
                Ok(0) => println!("  {} — OK ({} steps)", outcome.name, outcome.steps),
                Ok(warnings) => println!(
                    "  {} — OK ({} steps, {warnings} action-label warning(s))",
                    outcome.name, outcome.steps
                ),
                Err(message) => {
                    println!("  {} — FAILED: {message}", outcome.name);
                }
            }
        }
    }

    println!("\nwrote: {}", spec_path.display());
    println!("       {}", cfg_path.display());
    println!(
        "\nThe mined module is a CANDIDATE for human review: it generalizes \
         finitely many observations and is not ground truth."
    );
}

/// Minimal TLA+ identifier check for --module-name.
fn is_identifier(s: &str) -> bool {
    let mut chars = s.chars();
    chars
        .next()
        .is_some_and(|c| c.is_ascii_alphabetic() || c == '_')
        && chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
        && s.chars().any(|c| c.is_ascii_alphabetic())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn is_identifier_accepts_module_names() {
        assert!(is_identifier("Mined"));
        assert!(is_identifier("Mined_2"));
        assert!(is_identifier("_x1"));
        assert!(!is_identifier("2Mined"));
        assert!(!is_identifier(""));
        assert!(!is_identifier("has space"));
        assert!(!is_identifier("___"));
    }

    #[test]
    fn tracegen_envelope_converts_states_and_labels() {
        let envelope = serde_json::json!({
            "tool": "ty trace-gen",
            "traces": [{
                "trace_id": 1,
                "states": [
                    {"index": 1, "action": "Initial predicate",
                     "variables": {"x": {"type": "int", "value": 0}}},
                    {"index": 2, "action": "Inc",
                     "variables": {"x": {"type": "int", "value": 1}}},
                    {"index": 3, "action": "Action",
                     "variables": {"x": {"type": "int", "value": 2}}}
                ]
            }]
        });
        let traces = convert_tracegen_envelope("gen.json", &envelope).expect("convert");
        assert_eq!(traces.len(), 1);
        let trace = &traces[0];
        assert_eq!(trace.name, "gen.json#1");
        assert_eq!(trace.variables, vec!["x".to_string()]);
        assert_eq!(trace.steps.len(), 3);
        assert!(trace.steps[0].action.is_none(), "step 0 never has a label");
        assert_eq!(
            trace.steps[1].action.as_ref().map(|a| a.name.as_str()),
            Some("Inc")
        );
        assert!(
            trace.steps[2].action.is_none(),
            "placeholder 'Action' label is dropped"
        );
    }

    #[test]
    fn tracegen_envelope_rejects_empty() {
        let envelope = serde_json::json!({"tool": "ty trace-gen", "traces": []});
        assert!(convert_tracegen_envelope("gen.json", &envelope).is_err());
    }
}
