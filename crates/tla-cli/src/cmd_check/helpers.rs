// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

use std::sync::OnceLock;

use tla_check::{bind_constants_from_config, CheckResult, Config, EvalCtx, Progress};

use crate::cli_schema::OutputFormat;

// `should_report_store_states_auto_enable` was removed: the checker no longer
// auto-upgrades to full-state storage for SYMMETRY + genuine-liveness runs.
// It disables declared SYMMETRY for those runs instead (the plain orbit
// quotient is unsound for temporal properties), and fp-only liveness needs no
// full-state storage, so there is no auto-enable left to report.

pub(crate) fn select_check_deadlock(
    no_deadlock: bool,
    _resolved_spec: Option<&tla_check::ResolvedSpec>,
    config: &Config,
) -> bool {
    if no_deadlock {
        return false;
    }

    // Match TLC: deadlock checking is controlled by CHECK_DEADLOCK (and CLI flags), not implicitly
    // disabled just because the spec uses `[A]_v` stuttering form.
    config.check_deadlock
}

pub(super) fn skip_liveness_for_benchmark() -> bool {
    static FLAG: OnceLock<bool> = OnceLock::new();
    *FLAG.get_or_init(|| {
        matches!(
            std::env::var("TY_SKIP_LIVENESS").ok().as_deref(),
            Some("1") | Some("true") | Some("TRUE")
        )
    })
}

pub(super) fn register_module_vars(ctx: &mut EvalCtx, module: &tla_core::ast::Module) {
    for unit in &module.units {
        if let tla_core::ast::Unit::Variable(var_names) = &unit.node {
            ctx.register_vars(var_names.iter().map(|v| v.node.clone()));
        }
    }
}

pub(super) fn apply_allow_incomplete_overflow(
    result: CheckResult,
    allow_incomplete: bool,
    output_format: OutputFormat,
) -> (CheckResult, Option<String>) {
    if !allow_incomplete {
        return (result, None);
    }

    match result {
        CheckResult::Error {
            error:
                tla_check::CheckError::Infra(tla_check::InfraCheckError::FingerprintOverflow {
                    dropped,
                    ..
                }),
            stats,
            ..
        } => {
            let note = format!(
                "Fingerprint storage overflow: {dropped} states were dropped. \
                 Continuing because --allow-incomplete is set; results are incomplete."
            );
            if matches!(output_format, OutputFormat::Human) {
                eprintln!("Warning: {note}");
            }
            (CheckResult::Success(stats), Some(note))
        }
        other => (other, None),
    }
}

pub(super) fn merge_strategy_info(base: Option<String>, extra: Option<String>) -> Option<String> {
    match (base, extra) {
        (Some(base), Some(extra)) => Some(format!("{base}\n{extra}")),
        (Some(base), None) => Some(base),
        (None, Some(extra)) => Some(extra),
        (None, None) => None,
    }
}

/// Create a progress callback that optionally includes state space estimation.
///
/// When `show_estimate` is true, the callback tracks per-level state counts
/// and displays a running estimate of total reachable states.
pub(super) fn make_progress_callback_with_estimate(
    show_estimate: bool,
) -> Box<dyn Fn(&Progress) + Send + Sync> {
    // Part of #3247: progress output is always-on for Human mode.
    // Uses a 10-second time guard to avoid flooding output.
    let last_progress_secs = std::sync::Mutex::new(0.0_f64);
    let last_depth = std::sync::Mutex::new(0_usize);
    let estimator = std::sync::Mutex::new(tla_check::StateSpaceEstimator::new());
    let estimator_initialized = std::sync::Mutex::new(false);

    Box::new(move |progress: &Progress| {
        // Update estimator on depth transitions
        if show_estimate {
            if let (Ok(mut prev_depth), Ok(mut est), Ok(mut initialized)) = (
                last_depth.lock(),
                estimator.lock(),
                estimator_initialized.lock(),
            ) {
                if !*initialized && progress.states_found > 0 {
                    // Record initial states at first progress report
                    est.record_initial_states(progress.states_found, progress.elapsed_secs);
                    *initialized = true;
                    *prev_depth = progress.current_depth;
                } else if progress.current_depth > *prev_depth {
                    // Depth transition detected — record level completion
                    est.record_level(
                        progress.current_depth,
                        progress.states_found,
                        progress.elapsed_secs,
                    );
                    *prev_depth = progress.current_depth;
                }
            }
        }

        let should_print = match last_progress_secs.lock() {
            Ok(mut last) => {
                if progress.elapsed_secs - *last >= 10.0 {
                    *last = progress.elapsed_secs;
                    true
                } else {
                    false
                }
            }
            Err(_) => true,
        };
        if should_print {
            let mem_str = match progress.memory_usage_bytes {
                Some(bytes) => format!(", {:.0} MB", bytes as f64 / (1024.0 * 1024.0)),
                None => String::new(),
            };

            // Build estimate string if enabled
            let est_str = if show_estimate {
                estimator
                    .lock()
                    .ok()
                    .and_then(|est| {
                        est.estimate().and_then(|result| {
                            if result.confidence > 0.1 {
                                Some(format!(". {}", result.format_human()))
                            } else {
                                None
                            }
                        })
                    })
                    .unwrap_or_default()
            } else {
                String::new()
            };

            eprintln!(
                "Progress({}) at {:.1}s: {} states generated, {} distinct states found, {:.0} states/sec{}{}",
                progress.current_depth,
                progress.elapsed_secs,
                progress.transitions,
                progress.states_found,
                progress.states_per_sec,
                mem_str,
                est_str,
            );
        }
    })
}

pub(super) fn apply_alias_transform(
    result: CheckResult,
    config: &Config,
    module: &tla_core::ast::Module,
    checker_modules: &[&tla_core::ast::Module],
    extended_modules: &[&tla_core::ast::Module],
) -> CheckResult {
    let Some(ref alias_name) = config.alias else {
        return result;
    };
    let mut alias_ctx = EvalCtx::new();
    // Load main module first (ALIAS operator is typically defined here),
    // then EXTENDS/INSTANCE modules for transitive dependencies.
    alias_ctx.load_module(module);
    for m in checker_modules {
        alias_ctx.load_module(m);
    }
    // Match the checker state boundary: only the main module plus the
    // transitive EXTENDS closure contribute state variables.
    register_module_vars(&mut alias_ctx, module);
    for m in extended_modules {
        register_module_vars(&mut alias_ctx, m);
    }
    if let Err(e) = bind_constants_from_config(&mut alias_ctx, config) {
        eprintln!(
            "Warning: ALIAS evaluation skipped (constant binding error: {})",
            e
        );
        result
    } else {
        result.apply_alias(&mut alias_ctx, alias_name)
    }
}
