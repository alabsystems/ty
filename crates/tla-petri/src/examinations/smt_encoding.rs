// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Shared SMT encoding and solver helpers for reachability analyses.

use std::collections::HashMap;
use std::io::{ErrorKind, Read, Write};
#[cfg(unix)]
use std::os::fd::AsRawFd;
#[cfg(unix)]
use std::os::raw::c_int;
#[cfg(unix)]
use std::os::unix::process::CommandExt;
use std::process::{Child, ChildStdout, ExitStatus};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use tla_mc_core::{
    BackendCapability, BackendDomain, BackendKind, CapabilityReport, CapabilityRole, ProblemKind,
    SolverFacet, SolverLimits, UnsupportedReason,
};

use crate::petri_net::PetriNet;
use crate::resolved_predicate::{ResolvedIntExpr, ResolvedPredicate};

/// Maximum per-depth timeout for ay solver invocation.
pub(super) const PER_DEPTH_TIMEOUT: Duration = Duration::from_secs(3);

/// Shared geometric depth ladder for SMT-based reachability analyses.
pub(super) const DEPTH_LADDER: &[usize] = &[1, 2, 4, 8, 16];

const AY_STDOUT_LIMIT_BYTES: usize = 8 * 1024 * 1024;
const AY_STDOUT_DRAIN_TIMEOUT: Duration = Duration::from_secs(1);
const RAW_SMT_PROCESS_OUTPUT_UNAVAILABLE: &str = "none";

#[cfg(unix)]
const F_GETFL: c_int = 3;
#[cfg(unix)]
const F_SETFL: c_int = 4;
#[cfg(all(
    unix,
    any(
        target_os = "macos",
        target_os = "ios",
        target_os = "freebsd",
        target_os = "openbsd",
        target_os = "netbsd"
    )
))]
const O_NONBLOCK: c_int = 0x0004;
#[cfg(all(unix, target_os = "linux"))]
const O_NONBLOCK: c_int = 0o0004000;
#[cfg(all(
    unix,
    not(any(
        target_os = "macos",
        target_os = "ios",
        target_os = "freebsd",
        target_os = "openbsd",
        target_os = "netbsd",
        target_os = "linux"
    ))
))]
const O_NONBLOCK: c_int = 0o0004000;

#[cfg(unix)]
unsafe extern "C" {
    fn fcntl(fd: c_int, cmd: c_int, ...) -> c_int;
}

/// Outcome of a single property check at a given SMT depth.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum SolverOutcome {
    /// Solver returned `sat` - witness/counterexample exists.
    Sat,
    /// Solver returned `unsat` - the query is disproved at this depth.
    Unsat,
    /// Solver returned `unknown`, timed out, or produced unparseable output.
    Unknown,
}

/// AY-owned solve/profile evidence for one solver invocation.
///
/// This wrapper deliberately stores the rendered row produced by `tla-ay`
/// instead of rebuilding MCC sidecar fields locally in the Petri backend.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct AYSolveProfileEvidence {
    row: String,
}

impl AYSolveProfileEvidence {
    /// Render MCC solve/profile evidence from a typed AY solve envelope.
    pub(super) fn mcc_from_solve_details(details: &tla_ay::SolveDetails) -> Self {
        Self {
            row: tla_ay::solve_details_decision_profile_summary_evidence("MCC", Some(details)),
        }
    }

    /// Render MCC solve/profile evidence from a typed AY decision/profile summary.
    pub(super) fn mcc_from_decision_profile_summary(
        summary: &tla_ay::SolveDecisionProfileSummary,
    ) -> Self {
        Self {
            row: tla_ay::solve_decision_profile_summary_evidence("MCC", Some(summary)),
        }
    }

    /// Render MCC-compatible solve/profile evidence from AY-owned raw SMT metadata.
    pub(super) fn mcc_from_raw_smt_summary(summary: &ay_dpll::RawSmtSolveProfileSummary) -> Self {
        Self {
            row: render_mcc_raw_smt_solve_profile_summary(summary),
        }
    }

    pub(super) fn as_row(&self) -> &str {
        &self.row
    }

    pub(super) fn into_row(self) -> String {
        self.row
    }
}

/// Solver outcomes plus optional typed AY solve/profile evidence.
///
/// When `fail_closed` is `true`, the underlying solver process did not produce
/// real per-property outcomes (timeout, non-zero exit, or short stdout). The
/// `outcomes` vector is then a fail-closed `[Unknown; num_properties]` shell
/// kept only so callers that want raw-SMT process evidence can still consume
/// `solve_profile`. Callers that need real solver answers (the BMC retry
/// pipeline, the thin `run_ay` wrapper) must check `is_fail_closed()` and
/// treat fail-closed reports the same as `None`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct SolverRunReport {
    outcomes: Vec<SolverOutcome>,
    solve_profile: Option<AYSolveProfileEvidence>,
    fail_closed: bool,
}

impl SolverRunReport {
    pub(super) fn from_outcomes(outcomes: Vec<SolverOutcome>) -> Self {
        Self {
            outcomes,
            solve_profile: None,
            fail_closed: false,
        }
    }

    pub(super) fn from_outcomes_and_decision_profile_summary(
        outcomes: Vec<SolverOutcome>,
        summary: &tla_ay::SolveDecisionProfileSummary,
    ) -> Self {
        Self::from_outcomes(outcomes).with_solve_profile(
            AYSolveProfileEvidence::mcc_from_decision_profile_summary(summary),
        )
    }

    pub(super) fn with_solve_profile(mut self, solve_profile: AYSolveProfileEvidence) -> Self {
        self.solve_profile = Some(solve_profile);
        self
    }

    pub(super) fn with_fail_closed(mut self, fail_closed: bool) -> Self {
        self.fail_closed = fail_closed;
        self
    }

    pub(super) fn outcomes(&self) -> &[SolverOutcome] {
        &self.outcomes
    }

    pub(super) fn into_outcomes(self) -> Vec<SolverOutcome> {
        self.outcomes
    }

    pub(super) fn is_fail_closed(&self) -> bool {
        self.fail_closed
    }

    pub(super) fn solve_profile(&self) -> Option<&AYSolveProfileEvidence> {
        self.solve_profile.as_ref()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct RawSmtProcessRun {
    stdout: String,
    stderr: String,
    exit_code: Option<i32>,
    wall_time_ms: u128,
    timed_out: bool,
    deadline_exceeded: bool,
}

impl RawSmtProcessRun {
    pub(super) fn new(
        stdout: String,
        stderr: String,
        exit_code: Option<i32>,
        wall_time_ms: u128,
        timed_out: bool,
        deadline_exceeded: bool,
    ) -> Self {
        Self {
            stdout,
            stderr,
            exit_code,
            wall_time_ms,
            timed_out,
            deadline_exceeded,
        }
    }

    pub(super) fn from_incremental_stdout(
        stdout: String,
        wall_time_ms: u128,
        deadline_exceeded: bool,
    ) -> Self {
        Self::new(
            stdout,
            String::new(),
            Some(0),
            wall_time_ms,
            false,
            deadline_exceeded,
        )
    }

    pub(super) fn failed(
        stderr: impl Into<String>,
        wall_time_ms: u128,
        timed_out: bool,
        deadline_exceeded: bool,
    ) -> Self {
        Self::new(
            String::new(),
            stderr.into(),
            None,
            wall_time_ms,
            timed_out,
            deadline_exceeded,
        )
    }

    pub(super) fn timed_out(&self) -> bool {
        self.timed_out
    }

    pub(super) fn exit_code(&self) -> Option<i32> {
        self.exit_code
    }

    fn to_ay_raw_summary(
        &self,
        solver_path: &std::path::Path,
        logic: Option<&str>,
    ) -> ay_dpll::RawSmtSolveProfileSummary {
        ay_dpll::raw_smt_solve_profile_summary_from_process(
            ay_dpll::RawSmtProcessSolveProfileInput::new(
                &solver_path.display().to_string(),
                logic,
                &self.stdout,
                &self.stderr,
                self.exit_code,
            )
            .with_wall_time_ms(self.wall_time_ms)
            .with_timed_out(self.timed_out)
            .with_deadline_exceeded(self.deadline_exceeded),
        )
    }
}

pub(super) fn raw_smt_process_run_report(
    solver_path: &std::path::Path,
    logic: Option<&str>,
    process: &RawSmtProcessRun,
    outcomes: Vec<SolverOutcome>,
) -> SolverRunReport {
    let summary = process.to_ay_raw_summary(solver_path, logic);
    SolverRunReport::from_outcomes(outcomes)
        .with_solve_profile(AYSolveProfileEvidence::mcc_from_raw_smt_summary(&summary))
}

fn render_mcc_raw_smt_solve_profile_summary(
    summary: &ay_dpll::RawSmtSolveProfileSummary,
) -> String {
    let validation = ay_dpll::validate_raw_smt_solve_profile_summary(summary);
    let status = if summary.accepted_for_consumer {
        "Available"
    } else {
        "Unavailable"
    };
    let decision_name = summary
        .decision
        .map_or(RAW_SMT_PROCESS_OUTPUT_UNAVAILABLE, |decision| {
            decision.name()
        });
    let consumer_rejection_code = summary
        .consumer_rejection_code
        .unwrap_or(RAW_SMT_PROCESS_OUTPUT_UNAVAILABLE);
    let unknown_reason_code = summary
        .unknown_reason_code
        .unwrap_or(RAW_SMT_PROCESS_OUTPUT_UNAVAILABLE);
    let unknown_limit_code = summary
        .unknown_limit_code
        .unwrap_or(RAW_SMT_PROCESS_OUTPUT_UNAVAILABLE);
    let verification_level_code = summary
        .verification_level_code
        .unwrap_or(RAW_SMT_PROCESS_OUTPUT_UNAVAILABLE);
    let process_exit_code = summary.process_exit_code.map_or_else(
        || RAW_SMT_PROCESS_OUTPUT_UNAVAILABLE.to_string(),
        |code| code.to_string(),
    );

    format!(
        "MCC ay_solver_decision_profile_summary status={status} \
         status_code={status_code} schema={schema} schema_version={schema_version} \
         producer_revision={producer_revision} source={source} reason_code={reason_code} \
         decision={decision_name} decision_code={decision_code} \
         accepted_for_consumer={accepted_for_consumer} \
         consumer_rejection_code={consumer_rejection_code} model_validated={model_validated} \
         verification_level_code={verification_level_code} \
         unknown_reason_code={unknown_reason_code} unknown_limit_code={unknown_limit_code} \
         wall_time_ms={wall_time_ms} conflicts={conflicts} decisions={decisions} \
         propagations={propagations} restarts={restarts} \
         learned_clause_count={learned_clause_count} \
         profile_wall_time_ms={wall_time_ms} profile_conflicts={conflicts} \
         profile_decisions={decisions} profile_propagations={propagations} \
         profile_restarts={restarts} profile_learned_clause_count={learned_clause_count} \
         profile_num_assertions={profile_num_assertions} profile_term_count={profile_term_count} \
         typed_consumer={typed_consumer} timed_out={timed_out} \
         deadline_exceeded={deadline_exceeded} process_exit_code={process_exit_code} \
         raw_smt_status_code={status_code} raw_smt_reason_code={reason_code} \
         raw_smt_validation_status_code={validation_status_code} \
         raw_smt_validation_reason_code={validation_reason_code} \
         production_selected=false fail_closed={fail_closed}",
        status_code = summary.status_code,
        schema = summary.schema,
        schema_version = summary.schema_version,
        producer_revision = summary.producer_revision,
        source = summary.source_code,
        reason_code = summary.reason_code,
        decision_code = summary.decision_code,
        accepted_for_consumer = summary.accepted_for_consumer,
        model_validated = summary.model_validated,
        wall_time_ms = summary.profile.wall_time_ms,
        conflicts = summary.profile.conflicts,
        decisions = summary.profile.decisions,
        propagations = summary.profile.propagations,
        restarts = summary.profile.restarts,
        learned_clause_count = summary.profile.learned_clause_count,
        profile_num_assertions = summary.profile.num_assertions,
        profile_term_count = summary.profile.term_count,
        typed_consumer = summary.typed_consumer,
        timed_out = summary.timed_out,
        deadline_exceeded = summary.deadline_exceeded,
        validation_status_code = validation.status_code,
        validation_reason_code = validation.reason_code,
        fail_closed = summary.fail_closed,
    )
}

/// Boolean assignments extracted from a SAT model/get-value response.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct SolverBoolModel {
    values: HashMap<String, bool>,
}

impl SolverBoolModel {
    pub(super) fn bool_value(&self, name: &str) -> Option<bool> {
        self.values.get(name).copied()
    }

    /// Construct a model directly from `(name, value)` pairs.
    ///
    /// Test-only: lets sibling SAT->verdict lanes exercise their
    /// decode/replay/confirm soundness path without spawning a real solver.
    #[cfg(test)]
    pub(in crate::examinations) fn from_pairs<I, S>(pairs: I) -> Self
    where
        I: IntoIterator<Item = (S, bool)>,
        S: Into<String>,
    {
        Self {
            values: pairs
                .into_iter()
                .map(|(name, value)| (name.into(), value))
                .collect(),
        }
    }
}

#[cfg(unix)]
mod unix_process_group {
    unsafe extern "C" {
        fn kill(pid: i32, sig: i32) -> i32;
    }

    const SIGKILL: i32 = 9;

    pub(super) fn kill_group(pid: u32) {
        if i32::try_from(pid).is_ok() {
            // Negative pid targets the solver process group. This also clears
            // batch solvers that leave helper processes holding stdout open.
            let _ = unsafe { kill(-(pid as i32), SIGKILL) };
        }
    }
}

fn terminate_solver(child: &mut Child) {
    #[cfg(unix)]
    unix_process_group::kill_group(child.id());
    let _ = child.kill();
    let _ = child.wait();
}

fn kill_solver_group(child: &Child) {
    #[cfg(unix)]
    unix_process_group::kill_group(child.id());
}

/// Find ay binary using the shared TY solver-discovery policy.
pub(super) fn find_ay() -> Option<std::path::PathBuf> {
    find_ay_with_report(ProblemKind::Smt, SolverLimits::default()).0
}

/// Find ay and produce shared backend evidence for the caller's routing decision.
pub(crate) fn find_ay_with_report(
    problem: ProblemKind,
    limits: SolverLimits,
) -> (Option<std::path::PathBuf>, CapabilityReport) {
    let found = tla_mc_core::find_ay_binary();
    let mut report = CapabilityReport::new(problem).with_limits(limits);

    match &found {
        Some(found) => {
            report.select(
                found
                    .capability(BackendDomain::PetriMcc)
                    .for_problem(problem)
                    .with_facets([
                        SolverFacet::ExternalProcess,
                        SolverFacet::Smt,
                        SolverFacet::ModelValues,
                    ])
                    .with_role(CapabilityRole::Production),
            );
            report.add_evidence(format!("external ay selected at {}", found.path.display()));
        }
        None => {
            report.reject(
                BackendCapability::unavailable(
                    BackendDomain::PetriMcc,
                    BackendKind::ExternalAYBinary,
                    UnsupportedReason::MissingBinary("ay"),
                )
                .for_problem(problem)
                .with_facets([SolverFacet::ExternalProcess, SolverFacet::Smt]),
            );
        }
    }

    (found.map(|found| found.path), report)
}

#[cfg(test)]
pub(crate) fn ay_env_lock() -> std::sync::MutexGuard<'static, ()> {
    // Delegate to the single crate-wide env lock so every env-touching test —
    // including those in other modules that mutate AY_PATH/HOME/PATH or the BMC
    // feature flags — serializes against each other. A per-module lock here would
    // only serialize smt_encoding's own tests, which is what previously let
    // readers race concurrent mutators (intermittent BMC/k-induction flakes).
    crate::env_test_lock()
}

/// Encode a resolved predicate as an SMT-LIB expression at a given step.
pub(super) fn encode_predicate(pred: &ResolvedPredicate, step: usize, net: &PetriNet) -> String {
    match pred {
        ResolvedPredicate::True => "true".to_string(),
        ResolvedPredicate::False => "false".to_string(),
        ResolvedPredicate::And(children) => {
            if children.is_empty() {
                return "true".to_string();
            }
            if children.len() == 1 {
                return encode_predicate(&children[0], step, net);
            }
            let parts: Vec<String> = children
                .iter()
                .map(|child| encode_predicate(child, step, net))
                .collect();
            format!("(and {})", parts.join(" "))
        }
        ResolvedPredicate::Or(children) => {
            if children.is_empty() {
                return "false".to_string();
            }
            if children.len() == 1 {
                return encode_predicate(&children[0], step, net);
            }
            let parts: Vec<String> = children
                .iter()
                .map(|child| encode_predicate(child, step, net))
                .collect();
            format!("(or {})", parts.join(" "))
        }
        ResolvedPredicate::Not(inner) => {
            format!("(not {})", encode_predicate(inner, step, net))
        }
        ResolvedPredicate::IntLe(left, right) => {
            format!(
                "(<= {} {})",
                encode_int_expr(left, step),
                encode_int_expr(right, step)
            )
        }
        ResolvedPredicate::IsFireable(transitions) => {
            if transitions.is_empty() {
                return "false".to_string();
            }
            let parts: Vec<String> = transitions
                .iter()
                .map(|transition_idx| {
                    let transition = &net.transitions[transition_idx.0 as usize];
                    if transition.inputs.is_empty() {
                        return "true".to_string();
                    }
                    let guards: Vec<String> = transition
                        .inputs
                        .iter()
                        .map(|arc| {
                            format!("(>= m_{}_{} {})", step, arc.place.0 as usize, arc.weight)
                        })
                        .collect();
                    if guards.len() == 1 {
                        guards[0].clone()
                    } else {
                        format!("(and {})", guards.join(" "))
                    }
                })
                .collect();
            if parts.len() == 1 {
                parts[0].clone()
            } else {
                format!("(or {})", parts.join(" "))
            }
        }
    }
}

/// Encode a resolved integer expression as SMT-LIB at a given step.
pub(super) fn encode_int_expr(expr: &ResolvedIntExpr, step: usize) -> String {
    match expr {
        ResolvedIntExpr::Constant(value) => format!("{value}"),
        ResolvedIntExpr::TokensCount(places) => {
            if places.is_empty() {
                "0".to_string()
            } else if places.len() == 1 {
                format!("m_{}_{}", step, places[0].0)
            } else {
                let parts: Vec<String> = places
                    .iter()
                    .map(|place| format!("m_{}_{}", step, place.0))
                    .collect();
                format!("(+ {})", parts.join(" "))
            }
        }
    }
}

/// Run ay on an SMT script and parse outcomes for each property.
///
/// Returns `None` when no process metadata can be collected.
/// Returns `Some(outcomes)` with one outcome per property.
pub(super) fn run_ay(
    ay_path: &std::path::Path,
    script: &str,
    num_properties: usize,
    timeout: Duration,
) -> Option<Vec<SolverOutcome>> {
    let report = run_ay_with_report(ay_path, script, num_properties, timeout)?;
    if report.is_fail_closed() {
        return None;
    }
    Some(report.into_outcomes())
}

/// Run ay and keep a AY-owned raw SMT solve/profile handoff with the parsed outcomes.
///
/// The external-process SMT-LIB adapter does not expose typed in-process
/// `SolveDetails`, so it feeds captured process metadata through the AY raw-SMT
/// summary facade and forwards a MCC-compatible row derived from that summary.
pub(super) fn run_ay_with_report(
    ay_path: &std::path::Path,
    script: &str,
    num_properties: usize,
    timeout: Duration,
) -> Option<SolverRunReport> {
    run_ay_with_report_deadline(ay_path, script, num_properties, timeout, false)
}

pub(super) fn run_ay_with_report_deadline(
    ay_path: &std::path::Path,
    script: &str,
    num_properties: usize,
    timeout: Duration,
    deadline_exceeded_on_timeout: bool,
) -> Option<SolverRunReport> {
    let process = run_ay_process(ay_path, script, timeout, deadline_exceeded_on_timeout)?;
    let mut outcomes = parse_solver_outcomes(&process.stdout);

    // The solver "really answered" only when the process exited cleanly without
    // a timeout AND produced at least `num_properties` outcomes. Any other case
    // is fail-closed evidence: we still attach a raw-SMT profile row, but the
    // outcomes vector is a `[Unknown; num_properties]` shell, not real solver
    // answers. Callers that act on outcomes (BMC retry, run_ay) check
    // `is_fail_closed()`.
    let process_failed = process.timed_out || process.exit_code != Some(0);
    let outcomes_short = outcomes.len() < num_properties;
    let fail_closed = process_failed || outcomes_short;

    if outcomes_short {
        if outcomes.is_empty() || process_failed {
            outcomes.resize(num_properties, SolverOutcome::Unknown);
        } else {
            return None;
        }
    }

    Some(
        raw_smt_process_run_report(
            ay_path,
            detect_smt_logic(script),
            &process,
            outcomes[outcomes.len() - num_properties..].to_vec(),
        )
        .with_fail_closed(fail_closed),
    )
}

/// Run ay on a single SAT query that asks for Boolean model values.
///
/// Returns `None` unless the first definitive solver status is `sat` and the
/// model output is parseable without conflicting assignments.
pub(super) fn run_ay_bool_model(
    ay_path: &std::path::Path,
    script: &str,
    timeout: Duration,
) -> Option<SolverBoolModel> {
    let stdout = run_ay_stdout(ay_path, script, timeout)?;
    if parse_solver_outcomes(&stdout).first().copied() != Some(SolverOutcome::Sat) {
        return None;
    }
    parse_solver_bool_model(&stdout)
}

fn run_ay_stdout(ay_path: &std::path::Path, script: &str, timeout: Duration) -> Option<String> {
    run_ay_process(ay_path, script, timeout, false).and_then(|process| {
        (!process.timed_out && process.exit_code == Some(0)).then_some(process.stdout)
    })
}

fn run_ay_process(
    ay_path: &std::path::Path,
    script: &str,
    timeout: Duration,
    deadline_exceeded_on_timeout: bool,
) -> Option<RawSmtProcessRun> {
    #[cfg(unix)]
    {
        run_ay_process_unix(ay_path, script, timeout, deadline_exceeded_on_timeout)
    }
    #[cfg(not(unix))]
    {
        run_ay_process_blocking(ay_path, script, timeout, deadline_exceeded_on_timeout)
    }
}

#[cfg(unix)]
fn run_ay_process_unix(
    ay_path: &std::path::Path,
    script: &str,
    timeout: Duration,
    deadline_exceeded_on_timeout: bool,
) -> Option<RawSmtProcessRun> {
    let mut command = Command::new(ay_path);
    command
        .arg("-smt2")
        .arg("-in")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null());
    command.process_group(0);

    let start = Instant::now();
    let mut child = command.spawn().ok()?;
    let Some(mut stdout) = child.stdout.take() else {
        terminate_solver(&mut child);
        return None;
    };
    if set_nonblocking(stdout.as_raw_fd()).is_none() {
        terminate_solver(&mut child);
        return None;
    }

    let Some(mut stdin) = child.stdin.take() else {
        terminate_solver(&mut child);
        return Some(RawSmtProcessRun::failed(
            "missing solver stdin",
            start.elapsed().as_millis(),
            false,
            false,
        ));
    };
    if stdin.write_all(script.as_bytes()).is_err() {
        terminate_solver(&mut child);
        return Some(RawSmtProcessRun::failed(
            "solver stdin write failed",
            start.elapsed().as_millis(),
            false,
            false,
        ));
    }
    drop(stdin);

    let mut output = Vec::new();
    let exit_status: ExitStatus;
    loop {
        if drain_stdout_nonblocking(&mut stdout, &mut output).is_none() {
            terminate_solver(&mut child);
            return None;
        }
        match child.try_wait() {
            Ok(Some(status)) => {
                exit_status = status;
                break;
            }
            Ok(None) => {
                if start.elapsed() >= timeout {
                    terminate_solver(&mut child);
                    return Some(RawSmtProcessRun::new(
                        String::from_utf8_lossy(&output).into_owned(),
                        String::new(),
                        None,
                        start.elapsed().as_millis(),
                        true,
                        deadline_exceeded_on_timeout,
                    ));
                }
                std::thread::sleep(Duration::from_millis(10));
            }
            Err(_) => {
                terminate_solver(&mut child);
                return None;
            }
        }
    }

    kill_solver_group(&child);
    let _ = child.wait();
    drain_stdout_after_exit(&mut stdout, &mut output)?;
    Some(RawSmtProcessRun::new(
        String::from_utf8_lossy(&output).into_owned(),
        String::new(),
        exit_status.code(),
        start.elapsed().as_millis(),
        false,
        false,
    ))
}

#[cfg(not(unix))]
fn run_ay_process_blocking(
    ay_path: &std::path::Path,
    script: &str,
    timeout: Duration,
    deadline_exceeded_on_timeout: bool,
) -> Option<RawSmtProcessRun> {
    let mut command = Command::new(ay_path);
    command
        .arg("-smt2")
        .arg("-in")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null());
    let start = Instant::now();
    let mut child = command.spawn().ok()?;

    let Some(mut stdin) = child.stdin.take() else {
        terminate_solver(&mut child);
        return Some(RawSmtProcessRun::failed(
            "missing solver stdin",
            start.elapsed().as_millis(),
            false,
            false,
        ));
    };
    if stdin.write_all(script.as_bytes()).is_err() {
        terminate_solver(&mut child);
        return Some(RawSmtProcessRun::failed(
            "solver stdin write failed",
            start.elapsed().as_millis(),
            false,
            false,
        ));
    }
    drop(stdin);

    loop {
        match child.try_wait() {
            Ok(Some(_status)) => break,
            Ok(None) => {
                if start.elapsed() >= timeout {
                    terminate_solver(&mut child);
                    return Some(RawSmtProcessRun::new(
                        String::new(),
                        String::new(),
                        None,
                        start.elapsed().as_millis(),
                        true,
                        deadline_exceeded_on_timeout,
                    ));
                }
                std::thread::sleep(Duration::from_millis(10));
            }
            Err(_) => {
                terminate_solver(&mut child);
                return None;
            }
        }
    }

    let output = child.wait_with_output().ok()?;
    if output.stdout.len() > AY_STDOUT_LIMIT_BYTES {
        return None;
    }
    Some(RawSmtProcessRun::new(
        String::from_utf8_lossy(&output.stdout).into_owned(),
        String::new(),
        output.status.code(),
        start.elapsed().as_millis(),
        false,
        false,
    ))
}

fn detect_smt_logic(script: &str) -> Option<&str> {
    script.lines().find_map(|line| {
        let trimmed = line.trim();
        trimmed
            .strip_prefix("(set-logic ")
            .and_then(|rest| rest.strip_suffix(')'))
            .map(str::trim)
            .filter(|logic| !logic.is_empty())
    })
}

#[cfg(unix)]
fn set_nonblocking(fd: std::os::fd::RawFd) -> Option<()> {
    let flags = unsafe { fcntl(fd, F_GETFL) };
    if flags < 0 {
        return None;
    }
    let result = unsafe { fcntl(fd, F_SETFL, flags | O_NONBLOCK) };
    (result >= 0).then_some(())
}

#[cfg(unix)]
fn drain_stdout_after_exit(stdout: &mut ChildStdout, output: &mut Vec<u8>) -> Option<()> {
    let deadline = Instant::now() + AY_STDOUT_DRAIN_TIMEOUT;
    loop {
        match drain_stdout_nonblocking(stdout, output)? {
            DrainResult::Eof => return Some(()),
            DrainResult::WouldBlock if Instant::now() >= deadline => return Some(()),
            DrainResult::WouldBlock => std::thread::sleep(Duration::from_millis(10)),
        }
    }
}

#[cfg(unix)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DrainResult {
    Eof,
    WouldBlock,
}

#[cfg(unix)]
fn drain_stdout_nonblocking(stdout: &mut ChildStdout, output: &mut Vec<u8>) -> Option<DrainResult> {
    let mut buf = [0_u8; 8192];
    loop {
        match stdout.read(&mut buf) {
            Ok(0) => return Some(DrainResult::Eof),
            Ok(n) => {
                if output.len().saturating_add(n) > AY_STDOUT_LIMIT_BYTES {
                    return None;
                }
                output.extend_from_slice(&buf[..n]);
            }
            Err(error) if error.kind() == ErrorKind::WouldBlock => {
                return Some(DrainResult::WouldBlock);
            }
            Err(error) if error.kind() == ErrorKind::Interrupted => {}
            Err(_) => return None,
        }
    }
}

fn parse_solver_outcomes(stdout: &str) -> Vec<SolverOutcome> {
    // Ignore diagnostic chatter and keep only definitive SMT solver status
    // tokens. The local ay binary may emit `[DIAG-SAT] ...` lines on stdout
    // before the actual `sat`/`unsat` result.
    stdout
        .lines()
        .map(str::trim)
        .filter_map(|line| match line {
            "sat" => Some(SolverOutcome::Sat),
            "unsat" => Some(SolverOutcome::Unsat),
            "unknown" => Some(SolverOutcome::Unknown),
            _ => None,
        })
        .collect()
}

fn parse_solver_bool_model(stdout: &str) -> Option<SolverBoolModel> {
    let tokens = tokenize_smt_output(stdout);
    let mut values = HashMap::new();

    for index in 0..tokens.len() {
        if index + 3 < tokens.len()
            && tokens[index] == "("
            && tokens[index + 3] == ")"
            && is_bmc_model_bool_name(&tokens[index + 1])
        {
            if let Some(value) = bool_token_value(&tokens[index + 2]) {
                insert_bool_assignment(&mut values, &tokens[index + 1], value)?;
            }
        }

        if index + 7 < tokens.len()
            && tokens[index] == "("
            && tokens[index + 1] == "define-fun"
            && tokens[index + 3] == "("
            && tokens[index + 4] == ")"
            && tokens[index + 5] == "Bool"
            && tokens[index + 7] == ")"
            && is_bmc_model_bool_name(&tokens[index + 2])
        {
            if let Some(value) = bool_token_value(&tokens[index + 6]) {
                insert_bool_assignment(&mut values, &tokens[index + 2], value)?;
            }
        }
    }

    Some(SolverBoolModel { values })
}

fn tokenize_smt_output(stdout: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut current = String::new();

    let flush_current = |tokens: &mut Vec<String>, current: &mut String| {
        if !current.is_empty() {
            tokens.push(std::mem::take(current));
        }
    };

    for ch in stdout.chars() {
        match ch {
            '(' | ')' => {
                flush_current(&mut tokens, &mut current);
                tokens.push(ch.to_string());
            }
            ch if ch.is_whitespace() => flush_current(&mut tokens, &mut current),
            _ => current.push(ch),
        }
    }
    flush_current(&mut tokens, &mut current);

    tokens
}

fn is_bmc_model_bool_name(name: &str) -> bool {
    name.starts_with("stay_") || name.starts_with("fire_")
}

fn bool_token_value(token: &str) -> Option<bool> {
    match token {
        "true" => Some(true),
        "false" => Some(false),
        _ => None,
    }
}

fn insert_bool_assignment(
    values: &mut HashMap<String, bool>,
    name: &str,
    value: bool,
) -> Option<()> {
    match values.insert(name.to_string(), value) {
        Some(previous) if previous != value => None,
        _ => Some(()),
    }
}

/// Encode the transition relation for steps 0..depth into the SMT script.
pub(super) fn encode_transition_relation(
    script: &mut String,
    net: &PetriNet,
    num_places: usize,
    num_transitions: usize,
    depth: usize,
) {
    encode_transition_relation_steps(script, net, num_places, num_transitions, 0..depth);
}

/// Encode the transition relation for steps 0..depth, constraining stutter to
/// deadlock markings. Used exclusively by the LTL lasso BMC encoding; see
/// [`encode_transition_relation_steps_opts`] for the semantics.
pub(super) fn encode_transition_relation_deadlock_stutter(
    script: &mut String,
    net: &PetriNet,
    num_places: usize,
    num_transitions: usize,
    depth: usize,
) {
    encode_transition_relation_steps_opts(script, net, num_places, num_transitions, 0..depth, true);
}

/// Encode the transition relation for a specific step range.
///
/// Each step encodes: exactly-one-fires-or-stutters, guard+effect assertions,
/// and frame conditions for all places. The stutter (`stay`) action is allowed
/// freely (i.e. at any marking, deadlocked or not), which is the semantics
/// reachability BMC and k-induction rely on.
pub(super) fn encode_transition_relation_steps(
    script: &mut String,
    net: &PetriNet,
    num_places: usize,
    num_transitions: usize,
    step_range: std::ops::Range<usize>,
) {
    encode_transition_relation_steps_opts(
        script,
        net,
        num_places,
        num_transitions,
        step_range,
        false,
    );
}

/// Encode the transition relation for a specific step range, with an opt-in flag
/// constraining the stutter action to genuine deadlock markings.
///
/// Each step encodes: exactly-one-fires-or-stutters, guard+effect assertions,
/// and frame conditions for all places.
///
/// When `deadlock_stutter_only` is `false` (the default for reachability BMC and
/// k-induction) the stutter is free: `stay` may be selected at any marking. When
/// it is `true`, an additional constraint is emitted so that `stay_step` may only
/// hold when NO transition is enabled at `step` (a genuine deadlock), matching the
/// on-the-fly Büchi self-loop semantics required for sound LTL lasso search.
pub(super) fn encode_transition_relation_steps_opts(
    script: &mut String,
    net: &PetriNet,
    num_places: usize,
    num_transitions: usize,
    step_range: std::ops::Range<usize>,
    deadlock_stutter_only: bool,
) {
    for step in step_range {
        script.push_str(&format!("(assert (or stay_{step}"));
        for transition in 0..num_transitions {
            script.push_str(&format!(" fire_{}_{}", step, transition));
        }
        script.push_str("))\n");

        let mut all_options = vec![format!("stay_{step}")];
        for transition in 0..num_transitions {
            all_options.push(format!("fire_{}_{}", step, transition));
        }
        for left in 0..all_options.len() {
            for right in (left + 1)..all_options.len() {
                script.push_str(&format!(
                    "(assert (not (and {} {})))\n",
                    all_options[left], all_options[right]
                ));
            }
        }

        for place in 0..num_places {
            script.push_str(&format!(
                "(assert (=> stay_{} (= m_{}_{} m_{}_{})))\n",
                step,
                step + 1,
                place,
                step,
                place
            ));
        }

        // Lasso-only: a stutter is legal only at a deadlock marking (no enabled
        // transition). This forbids the solver from manufacturing a spurious
        // self-loop at a live marking, which would otherwise close a fake
        // accepting lasso and emit a wrong LTL FALSE. The per-transition
        // enabledness conjunction reuses the SAME guard built below for `fire`.
        if deadlock_stutter_only {
            let mut not_enabled_terms = Vec::with_capacity(num_transitions);
            for transition_idx in 0..num_transitions {
                let transition = &net.transitions[transition_idx];
                let mut enabled_terms = Vec::with_capacity(transition.inputs.len());
                for arc in &transition.inputs {
                    let place = arc.place.0 as usize;
                    enabled_terms.push(format!("(>= m_{}_{} {})", step, place, arc.weight));
                }
                let enabled_expr = match enabled_terms.len() {
                    0 => "true".to_string(),
                    1 => enabled_terms.into_iter().next().expect("one term"),
                    _ => format!("(and {})", enabled_terms.join(" ")),
                };
                not_enabled_terms.push(format!("(not {enabled_expr})"));
            }
            let body = match not_enabled_terms.len() {
                0 => "true".to_string(),
                1 => not_enabled_terms.into_iter().next().expect("one term"),
                _ => format!("(and {})", not_enabled_terms.join(" ")),
            };
            script.push_str(&format!("(assert (=> stay_{step} {body}))\n"));
        }

        for transition_idx in 0..num_transitions {
            let transition = &net.transitions[transition_idx];
            let fire_var = format!("fire_{}_{}", step, transition_idx);

            for arc in &transition.inputs {
                let place = arc.place.0 as usize;
                script.push_str(&format!(
                    "(assert (=> {} (>= m_{}_{} {})))\n",
                    fire_var, step, place, arc.weight
                ));
            }

            let mut deltas: Vec<(usize, i64)> = Vec::new();
            for arc in &transition.inputs {
                let place = arc.place.0 as usize;
                match deltas
                    .iter_mut()
                    .find(|(existing_place, _)| *existing_place == place)
                {
                    Some((_, delta)) => *delta -= arc.weight as i64,
                    None => deltas.push((place, -(arc.weight as i64))),
                }
            }
            for arc in &transition.outputs {
                let place = arc.place.0 as usize;
                match deltas
                    .iter_mut()
                    .find(|(existing_place, _)| *existing_place == place)
                {
                    Some((_, delta)) => *delta += arc.weight as i64,
                    None => deltas.push((place, arc.weight as i64)),
                }
            }

            for &(place, delta) in &deltas {
                if delta == 0 {
                    script.push_str(&format!(
                        "(assert (=> {} (= m_{}_{} m_{}_{})))\n",
                        fire_var,
                        step + 1,
                        place,
                        step,
                        place
                    ));
                } else if delta > 0 {
                    script.push_str(&format!(
                        "(assert (=> {} (= m_{}_{} (+ m_{}_{} {}))))\n",
                        fire_var,
                        step + 1,
                        place,
                        step,
                        place,
                        delta
                    ));
                } else {
                    script.push_str(&format!(
                        "(assert (=> {} (= m_{}_{} (- m_{}_{} {}))))\n",
                        fire_var,
                        step + 1,
                        place,
                        step,
                        place,
                        -delta
                    ));
                }
            }

            let affected: Vec<usize> = deltas.iter().map(|(place, _)| *place).collect();
            for place in 0..num_places {
                if !affected.contains(&place) {
                    script.push_str(&format!(
                        "(assert (=> {} (= m_{}_{} m_{}_{})))\n",
                        fire_var,
                        step + 1,
                        place,
                        step,
                        place
                    ));
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    #[cfg(unix)]
    use std::os::unix::fs::PermissionsExt;
    use std::path::{Path, PathBuf};
    use std::time::Duration;

    use tempfile::TempDir;

    /// Solver budget for tests that spawn the fake shell solver and expect it to
    /// run to completion (a real `Sat`/`Unsat`, or a deterministic exit-code
    /// failure). The fake solver answers in milliseconds when scheduled; this is
    /// only a safety bound. A tight 1s value let a CPU-starved subprocess (under
    /// full-parallel test load) hit the timeout and return the fail-closed
    /// `Unknown` shell, flaking the assertion. Generous budget removes the
    /// spurious timeout without changing the scheduled-case result.
    const FAKE_SOLVER_ANSWER_BUDGET: Duration = Duration::from_secs(30);

    struct EnvVarGuard {
        key: &'static str,
        prev: Option<String>,
    }

    impl EnvVarGuard {
        fn set(key: &'static str, value: &str) -> Self {
            let prev = std::env::var(key).ok();
            crate::env_guard::set_var(key, value);
            Self { key, prev }
        }

        fn remove(key: &'static str) -> Self {
            let prev = std::env::var(key).ok();
            crate::env_guard::remove_var(key);
            Self { key, prev }
        }
    }

    impl Drop for EnvVarGuard {
        fn drop(&mut self) {
            if let Some(prev) = &self.prev {
                crate::env_guard::set_var(self.key, prev);
            } else {
                crate::env_guard::remove_var(self.key);
            }
        }
    }

    fn write_fake_solver_script(dir: &Path, name: &str, body: &str) -> PathBuf {
        let path = dir.join(name);
        let script = format!("#!/bin/sh\nset -eu\n{body}\n");
        fs::write(&path, script).expect("failed to write fake solver script");
        #[cfg(unix)]
        {
            let mut permissions = fs::metadata(&path)
                .expect("script metadata should exist")
                .permissions();
            permissions.set_mode(0o755);
            fs::set_permissions(&path, permissions).expect("failed to mark fake solver executable");
        }
        path
    }

    #[test]
    fn find_ay_with_report_selects_external_ay() {
        let _guard = ay_env_lock();
        let temp = TempDir::new().expect("tempdir should create");
        let ay_path = temp.path().join("ay");
        fs::write(&ay_path, b"fake ay").expect("fake ay should write");
        let _ay_path = EnvVarGuard::set("AY_PATH", ay_path.to_str().expect("utf8 temp path"));

        let limits = SolverLimits {
            time_budget: Some(Duration::from_secs(2)),
            max_depth: Some(16),
            max_states: None,
            max_memory_bytes: None,
        };
        let (path, report) = find_ay_with_report(ProblemKind::Bmc, limits);

        assert_eq!(path.as_deref(), Some(ay_path.as_path()));
        assert_eq!(find_ay().as_deref(), Some(ay_path.as_path()));
        assert_eq!(report.problem, Some(ProblemKind::Bmc));
        assert_eq!(report.limits, limits);
        assert!(report.has_selected(BackendKind::ExternalAYBinary));
        assert!(report.rejected.is_empty());

        let selected = report
            .selected
            .iter()
            .find(|cap| cap.backend == BackendKind::ExternalAYBinary)
            .expect("external ay should be selected");
        assert_eq!(selected.domain, BackendDomain::PetriMcc);
        assert_eq!(selected.role, CapabilityRole::Production);
        assert_eq!(selected.status, tla_mc_core::CapabilityStatus::Available);
        assert_eq!(selected.problem, Some(ProblemKind::Bmc));
        assert_eq!(selected.reason_code(), None);
        assert!(selected.facets.contains(&SolverFacet::ExternalProcess));
        assert!(selected.facets.contains(&SolverFacet::Smt));
        assert!(selected.facets.contains(&SolverFacet::ModelValues));
    }

    #[test]
    fn find_ay_with_report_rejects_missing_ay() {
        let _guard = ay_env_lock();
        let temp = TempDir::new().expect("tempdir should create");
        let bin_dir = temp.path().join("bin");
        fs::create_dir(&bin_dir).expect("bin dir should create");
        let _ay_path = EnvVarGuard::remove("AY_PATH");
        let _home = EnvVarGuard::set("HOME", temp.path().to_str().expect("utf8 temp path"));
        let _path = EnvVarGuard::set("PATH", bin_dir.to_str().expect("utf8 temp path"));

        let (path, report) = find_ay_with_report(ProblemKind::Smt, SolverLimits::default());

        assert!(path.is_none());
        assert!(report.selected.is_empty());
        assert_eq!(
            report.rejection_reason(BackendKind::ExternalAYBinary),
            Some(&UnsupportedReason::MissingBinary("ay"))
        );
        assert_eq!(
            report.rejection_reason_code(BackendKind::ExternalAYBinary),
            Some("missing_binary")
        );
        let rejected = report
            .rejected
            .iter()
            .find(|cap| cap.backend == BackendKind::ExternalAYBinary)
            .expect("missing external ay should be rejected");
        assert_eq!(rejected.problem, Some(ProblemKind::Smt));
        assert_eq!(rejected.reason_code(), Some("missing_binary"));
        assert!(rejected.facets.contains(&SolverFacet::ExternalProcess));
        assert!(rejected.facets.contains(&SolverFacet::Smt));
    }

    #[test]
    fn test_parse_solver_bool_model_get_value_pairs() {
        let model =
            parse_solver_bool_model("sat\n((stay_0 false)\n (fire_0_0 true)\n (fire_0_1 false))\n")
                .expect("get-value model should parse");

        assert_eq!(model.bool_value("stay_0"), Some(false));
        assert_eq!(model.bool_value("fire_0_0"), Some(true));
        assert_eq!(model.bool_value("fire_0_1"), Some(false));
    }

    #[test]
    fn test_parse_solver_bool_model_define_fun_assignments() {
        let model = parse_solver_bool_model(
            "sat\n(model\n  (define-fun stay_0 () Bool false)\n  (define-fun fire_0_0 () Bool true)\n)\n",
        )
        .expect("define-fun model should parse");

        assert_eq!(model.bool_value("stay_0"), Some(false));
        assert_eq!(model.bool_value("fire_0_0"), Some(true));
    }

    #[test]
    fn test_parse_solver_bool_model_rejects_conflicting_assignments() {
        assert!(parse_solver_bool_model(
            "sat\n((stay_0 false)\n (stay_0 true)\n (fire_0_0 true))\n",
        )
        .is_none());
    }

    #[test]
    fn run_ay_with_report_consumes_ay_raw_smt_process_summary() {
        let tempdir = TempDir::new().expect("tempdir should create");
        let solver = write_fake_solver_script(
            tempdir.path(),
            "fake-ay-raw-smt-sat",
            "cat >/dev/null\nprintf 'sat\\n'",
        );

        let report = run_ay_with_report(
            &solver,
            "(set-logic QF_LIA)\n(check-sat)\n",
            1,
            FAKE_SOLVER_ANSWER_BUDGET,
        )
        .expect("raw SMT report should be available");

        assert_eq!(report.outcomes(), &[SolverOutcome::Sat]);
        let profile = report
            .solve_profile()
            .expect("raw SMT profile should be attached")
            .as_row();
        assert!(profile.contains("MCC ay_solver_decision_profile_summary"));
        assert!(profile.contains("schema=ay.raw-smt-solve-profile-summary.v1"));
        assert!(profile.contains("source=raw_process_execution"));
        assert!(profile.contains("reason_code=raw_process_status"));
        assert!(profile.contains("decision=SAT"));
        assert!(profile.contains("decision_code=sat"));
        assert!(profile.contains("accepted_for_consumer=true"));
        assert!(profile.contains("typed_consumer=false"));
        assert!(profile.contains("process_exit_code=0"));
        assert!(profile.contains("raw_smt_validation_status_code=accepted"));
        assert!(profile.contains("production_selected=false"));
        assert!(profile.contains("fail_closed=false"));
    }

    #[test]
    fn run_ay_with_report_records_raw_smt_timeout_deadline() {
        let tempdir = TempDir::new().expect("tempdir should create");
        let solver = write_fake_solver_script(
            tempdir.path(),
            "fake-ay-raw-smt-timeout",
            "cat >/dev/null\nsleep 5",
        );

        let report = run_ay_with_report_deadline(
            &solver,
            "(set-logic QF_LIA)\n(check-sat)\n",
            1,
            Duration::from_millis(50),
            true,
        )
        .expect("timeout should still produce fail-closed raw SMT evidence");

        assert_eq!(report.outcomes(), &[SolverOutcome::Unknown]);
        let profile = report
            .solve_profile()
            .expect("timeout raw SMT profile should be attached")
            .as_row();
        assert!(profile.contains("status=Unavailable"));
        assert!(profile.contains("reason_code=raw_process_timeout"));
        assert!(profile.contains("decision=none"));
        assert!(profile.contains("decision_code=none"));
        assert!(profile.contains("accepted_for_consumer=false"));
        assert!(profile.contains("timed_out=true"));
        assert!(profile.contains("deadline_exceeded=true"));
        assert!(profile.contains("process_exit_code=none"));
        assert!(profile.contains("fail_closed=true"));
        assert!(profile.contains("raw_smt_validation_status_code=accepted"));
    }

    #[test]
    fn run_ay_with_report_records_raw_smt_exit_code_failure() {
        let tempdir = TempDir::new().expect("tempdir should create");
        let solver = write_fake_solver_script(
            tempdir.path(),
            "fake-ay-raw-smt-exit-code",
            "cat >/dev/null\nexit 17",
        );

        let report = run_ay_with_report(
            &solver,
            "(set-logic QF_LIA)\n(check-sat)\n",
            1,
            FAKE_SOLVER_ANSWER_BUDGET,
        )
        .expect("process failure should still produce fail-closed raw SMT evidence");

        assert_eq!(report.outcomes(), &[SolverOutcome::Unknown]);
        let profile = report
            .solve_profile()
            .expect("exit-code raw SMT profile should be attached")
            .as_row();
        assert!(profile.contains("status=Unavailable"));
        assert!(profile.contains("reason_code=raw_process_error"));
        assert!(profile.contains("process_exit_code=17"));
        assert!(profile.contains("accepted_for_consumer=false"));
        assert!(profile.contains("fail_closed=true"));
        assert!(profile.contains("raw_smt_validation_status_code=accepted"));
    }

    #[test]
    fn ay_solve_profile_evidence_consumes_tla_ay_solve_details() {
        let mut solver = tla_ay::Solver::try_new(tla_ay::Logic::QfLia).expect("solver");
        let x = solver.declare_const("x", tla_ay::Sort::Int);
        let five = solver.int_const(5);
        let eq = solver.try_eq(x, five).expect("eq");
        solver.try_assert_term(eq).expect("assert");

        let details = solver.try_check_sat_with_details().expect("solve details");
        let profile = AYSolveProfileEvidence::mcc_from_solve_details(&details);

        assert!(profile
            .as_row()
            .contains("MCC ay_solver_decision_profile_summary"));
        assert!(profile
            .as_row()
            .contains("status_code=typed_summary_available"));
        assert!(profile.as_row().contains("typed_consumer=true"));

        let report = SolverRunReport::from_outcomes(vec![SolverOutcome::Sat])
            .with_solve_profile(profile.clone());
        assert_eq!(report.solve_profile(), Some(&profile));
    }

    #[test]
    fn ay_solve_profile_evidence_consumes_tla_ay_decision_profile_summary() {
        let mut solver = tla_ay::Solver::try_new(tla_ay::Logic::QfLia).expect("solver");
        let x = solver.declare_const("x", tla_ay::Sort::Int);
        let five = solver.int_const(5);
        let eq = solver.try_eq(x, five).expect("eq");
        solver.try_assert_term(eq).expect("assert");

        let details = solver.try_check_sat_with_details().expect("solve details");
        let summary = details.decision_profile_summary();
        let profile = AYSolveProfileEvidence::mcc_from_decision_profile_summary(&summary);

        assert!(profile
            .as_row()
            .contains("MCC ay_solver_decision_profile_summary"));
        assert!(profile
            .as_row()
            .contains("status_code=typed_summary_available"));
        assert!(profile.as_row().contains("typed_consumer=true"));

        let report = SolverRunReport::from_outcomes_and_decision_profile_summary(
            vec![SolverOutcome::Sat],
            &summary,
        );
        assert_eq!(report.solve_profile(), Some(&profile));
    }
}
