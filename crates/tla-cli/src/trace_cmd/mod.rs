// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Trace-related tooling (parsing, validation, visualization).

mod mine;
mod validate_format;
mod validate_spec;
pub(crate) mod view;

use std::path::PathBuf;

use anyhow::Result;
use clap::{Subcommand, ValueEnum};
use tla_check::{ActionLabelMode, TraceInputFormatSelection};

use crate::cli_schema::TraceViewOutputFormat;

/// Trace-related tooling (parsing, validation, visualization).
///
/// Spec-based validation (#1082) uses TraceValidationEngine from tla-check.
#[derive(Debug, Subcommand)]
pub enum TraceCommand {
    /// View a counterexample trace with variable diffs between states.
    View {
        /// Path to the JSON output file from `ty check --output json`.
        trace_file: PathBuf,
        /// Output format: human (colored with change markers), json, or table.
        #[arg(long, value_enum, default_value = "human")]
        format: TraceViewOutputFormat,
        /// Show only specified variables (can be repeated).
        #[arg(long = "var", value_name = "VARIABLE")]
        filter: Vec<String>,
        /// Show only a specific step in detail.
        #[arg(long)]
        step: Option<usize>,
        /// Show variable diffs between consecutive steps (default: true).
        #[arg(long, default_value = "true", action = clap::ArgAction::Set)]
        diff: bool,
        /// Show unchanged variables alongside diffs.
        #[arg(long)]
        show_unchanged: bool,
    },
    /// Validate trace input parsing and invariants (header, indices, variable keys).
    ///
    /// Without --spec: validates trace format only (JSON structure, indices, variable keys).
    /// With --spec: validates trace against TLA+ specification (states match Init/Next).
    Validate {
        /// Trace input file (`.json` or `.jsonl`). Use `-` for stdin.
        file: PathBuf,
        /// Input format selection (default: `auto`).
        ///
        /// `auto` prefers JSONL by extension (`.jsonl`), otherwise falls back to JSON.
        /// When reading from stdin (`-`), `auto` defaults to JSON; use `--input-format jsonl` for JSONL.
        #[arg(long, value_enum, default_value = "auto")]
        input_format: TraceInputFormatArg,
        /// TLA+ specification file for spec-based validation.
        ///
        /// When provided, validates that each trace step matches a valid spec state
        /// reachable via Init/Next transitions.
        #[arg(long)]
        spec: Option<PathBuf>,
        /// Configuration file for the specification (default: <code>&lt;spec&gt;.cfg</code>).
        ///
        /// Specifies INIT, NEXT, CONSTANTS, etc. Required when --spec is provided.
        #[arg(long)]
        config: Option<PathBuf>,
        /// Action label enforcement mode (default: `error`).
        ///
        /// `error`: action label mismatches fail validation.
        /// `warn`: action label mismatches produce warnings but validation continues
        /// using observation-matched candidates. Useful for traces where the runtime
        /// combines multiple spec actions into a single step.
        #[arg(long, value_enum, default_value = "error")]
        action_label_mode: ActionLabelModeArg,
        /// Allow trace steps that observe only a subset of spec variables.
        ///
        /// Requires --spec. Candidate spec states are filtered on the observed
        /// variables only; unobserved variables are unconstrained. Steps with an
        /// empty state map still filter via the Next relation and their action
        /// label. Off by default: every step must then observe every spec
        /// variable. Requires an enumerable Init predicate, and can be expensive
        /// when little is observed (candidate sets are enumerated explicitly).
        #[arg(long, requires = "spec")]
        allow_partial_observations: bool,
    },
    /// Mine a CANDIDATE TLA+ spec from observed traces (spec mining v1).
    ///
    /// Synthesizes variable domains, actions (clustered by action label when
    /// present, else by the set of changed variables), guards, and candidate
    /// invariants/monotonicity properties from one or more trace files, then
    /// closes the loop: the emitted module + config are model checked
    /// (`ty check` semantics) and the input traces are re-validated against
    /// the mined spec. Invariant/property candidates refuted by a
    /// counterexample are dropped and the check is re-run
    /// (counterexample-guided pruning, up to --max-rounds).
    ///
    /// The output is a HYPOTHESIS generalized from finitely many
    /// observations — candidates for human review, NOT ground truth.
    /// See docs/trace-mining.md.
    Mine {
        /// Trace input files: the `ty` trace format (`.json` / `.jsonl`, as
        /// accepted by `ty trace validate`) or `ty trace-gen --format json`
        /// output (auto-detected).
        #[arg(required = true)]
        files: Vec<PathBuf>,
        /// Input format selection (default: `auto`).
        ///
        /// `auto` prefers JSONL by extension (`.jsonl`), otherwise JSON.
        #[arg(long, value_enum, default_value = "auto")]
        input_format: TraceInputFormatArg,
        /// Module name for the mined spec (also the output file stem).
        #[arg(long, default_value = "Mined")]
        module_name: String,
        /// Directory to write `<MODULE>.tla` and `<MODULE>.cfg` into.
        #[arg(long, default_value = ".")]
        out: PathBuf,
        /// Integer domains with more distinct observed values than this are
        /// over-approximated as `lo..hi` ranges instead of enumerated sets.
        #[arg(long, default_value = "8")]
        max_domain_enum: usize,
        /// Maximum counterexample-guided refinement rounds.
        #[arg(long, default_value = "4")]
        max_rounds: usize,
        /// State bound for the verification check of the mined spec.
        #[arg(long, default_value = "100000")]
        max_states: usize,
        /// Emit the candidate spec only; skip the check + trace-validation loop.
        #[arg(long)]
        skip_verify: bool,
    },
}

#[derive(Clone, Copy, Debug, Default, ValueEnum)]
pub enum ActionLabelModeArg {
    /// Action label mismatches are hard errors (default).
    #[default]
    Error,
    /// Action label mismatches produce warnings but do not fail validation.
    Warn,
}

impl From<ActionLabelModeArg> for ActionLabelMode {
    fn from(arg: ActionLabelModeArg) -> Self {
        match arg {
            ActionLabelModeArg::Error => ActionLabelMode::Error,
            ActionLabelModeArg::Warn => ActionLabelMode::Warn,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, ValueEnum)]
pub enum TraceInputFormatArg {
    /// Auto-detect by extension (`.jsonl` => JSONL), else JSON.
    #[default]
    Auto,
    /// JSON object form: `{version,module,variables,steps:[...]}`.
    Json,
    /// JSON Lines form: `header` then `step` events, one JSON object per line.
    #[value(name = "jsonl")]
    Jsonl,
}

impl From<TraceInputFormatArg> for TraceInputFormatSelection {
    fn from(arg: TraceInputFormatArg) -> Self {
        match arg {
            TraceInputFormatArg::Auto => TraceInputFormatSelection::Auto,
            TraceInputFormatArg::Json => TraceInputFormatSelection::Json,
            TraceInputFormatArg::Jsonl => TraceInputFormatSelection::Jsonl,
        }
    }
}

/// Entry point for the `ty trace` subcommand: dispatch to the [`TraceCommand`]
/// handler (counterexample viewing or trace validation).
///
/// For `Validate`, spec-based validation runs only when `--spec` is supplied;
/// otherwise only the trace's structural format is checked.
///
/// # Errors
///
/// Propagates any error from the underlying handler — for example a missing or
/// malformed trace file, a config/spec parse failure, or a validation failure
/// (an unreachable trace step or an action-label mismatch under
/// [`ActionLabelMode::Error`]).
pub fn cmd_trace(command: TraceCommand) -> Result<()> {
    match command {
        TraceCommand::View {
            trace_file,
            format,
            filter,
            step,
            diff,
            show_unchanged,
        } => view::cmd_trace_view(&trace_file, format, &filter, step, diff, show_unchanged),
        TraceCommand::Validate {
            file,
            input_format,
            spec,
            config,
            action_label_mode,
            allow_partial_observations,
        } => {
            if let Some(spec_path) = spec {
                validate_spec::cmd_trace_validate_with_spec(
                    &file,
                    input_format,
                    &spec_path,
                    config.as_deref(),
                    action_label_mode.into(),
                    allow_partial_observations,
                )
            } else {
                validate_format::cmd_trace_validate_format(&file, input_format)
            }
        }
        TraceCommand::Mine {
            files,
            input_format,
            module_name,
            out,
            max_domain_enum,
            max_rounds,
            max_states,
            skip_verify,
        } => mine::cmd_trace_mine(&mine::MineArgs {
            files,
            input_format,
            module_name,
            out,
            max_domain_enum,
            max_rounds,
            max_states,
            skip_verify,
        }),
    }
}
