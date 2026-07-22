// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

use std::fs;
#[cfg(unix)]
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use tempfile::TempDir;

use super::{IncrementalSolver, SolverOutcome};

/// Upper bound proving an operation returned/killed its child WITHOUT waiting for
/// the fake solver's 5-second `sleep`. The meaningful claim is "did not block on
/// the 5s sleep", so any bound comfortably below 5s proves it. A 1s bound was the
/// load-fragile part: under full-parallel test load, subprocess spawn + probe +
/// kill scheduling latency alone can exceed 1s even though the child was killed
/// promptly. 4s keeps a full second of margin below the 5s sleep.
const KILLED_BEFORE_SLEEP_CEILING: Duration = Duration::from_secs(4);

fn write_fake_solver_script(dir: &Path, name: &str, body: &str) -> PathBuf {
    let path = dir.join(name);
    let script = format!("#!/bin/sh\nset -eu\n{body}\n");
    fs::write(&path, script).expect("failed to write fake solver script");
    #[cfg(unix)]
    {
        let mut perms = fs::metadata(&path)
            .expect("script metadata should exist")
            .permissions();
        perms.set_mode(0o755);
        fs::set_permissions(&path, perms).expect("failed to mark fake solver executable");
    }
    path
}

#[test]
fn test_incremental_solver_round_trips_sat_then_unsat() {
    let tempdir = TempDir::new().expect("tempdir should create");
    let solver = write_fake_solver_script(
        tempdir.path(),
        "fake-ay-incremental",
        r#"probe_done=0
count=0
while IFS= read -r line; do
  case "$line" in
    "(check-sat)")
      if [ "$probe_done" -eq 0 ]; then
        probe_done=1
        printf 'sat\n'
      else
        count=$((count + 1))
        if [ "$count" -eq 1 ]; then
          printf 'sat\n'
        else
          printf 'unsat\n'
        fi
      fi
      ;;
    "(exit)")
      exit 0
      ;;
  esac
done"#,
    );

    let mut incremental = IncrementalSolver::new(&solver).expect("probe should succeed");
    assert!(incremental.send("(set-logic QF_LIA)\n"));
    assert!(incremental.push(), "push should succeed");
    assert_eq!(
        incremental.check_sat(Duration::from_secs(1)),
        SolverOutcome::Sat
    );
    assert!(incremental.pop(), "pop should succeed");
    assert_eq!(
        incremental.check_sat(Duration::from_secs(1)),
        SolverOutcome::Unsat
    );
}

#[test]
fn test_incremental_solver_times_out_without_blocking() {
    let tempdir = TempDir::new().expect("tempdir should create");
    let solver = write_fake_solver_script(
        tempdir.path(),
        "fake-ay-incremental-timeout",
        r#"probe_done=0
while IFS= read -r line; do
  case "$line" in
    "(check-sat)")
      if [ "$probe_done" -eq 0 ]; then
        probe_done=1
        printf 'sat\n'
      else
        sleep 5
      fi
      ;;
    "(exit)")
      exit 0
      ;;
  esac
done"#,
    );

    let mut incremental = IncrementalSolver::new(&solver).expect("probe should succeed");
    let start = Instant::now();
    assert_eq!(
        incremental.check_sat(Duration::from_millis(50)),
        SolverOutcome::Unknown
    );
    assert!(
        start.elapsed() < KILLED_BEFORE_SLEEP_CEILING,
        "timeout-safe reads should fail closed quickly"
    );
}

#[test]
fn test_incremental_solver_drop_kills_exit_ignoring_child() {
    let tempdir = TempDir::new().expect("tempdir should create");
    let solver = write_fake_solver_script(
        tempdir.path(),
        "fake-ay-incremental-ignores-exit",
        r#"probe_done=0
while IFS= read -r line; do
  case "$line" in
    "(check-sat)")
      if [ "$probe_done" -eq 0 ]; then
        probe_done=1
        printf 'sat\n'
      else
        printf 'unknown\n'
      fi
      ;;
    "(exit)")
      sleep 5
      ;;
  esac
done"#,
    );

    let start = Instant::now();
    let incremental = IncrementalSolver::new(&solver).expect("probe should succeed");
    drop(incremental);
    assert!(
        start.elapsed() < KILLED_BEFORE_SLEEP_CEILING,
        "dropping a solver that ignores (exit) should kill it quickly"
    );
}

#[test]
fn test_incremental_solver_check_sat_report_uses_raw_smt_summary() {
    let tempdir = TempDir::new().expect("tempdir should create");
    let solver = write_fake_solver_script(
        tempdir.path(),
        "fake-ay-incremental-raw-summary",
        r#"probe_done=0
while IFS= read -r line; do
  case "$line" in
    "(check-sat)")
      if [ "$probe_done" -eq 0 ]; then
        probe_done=1
        printf 'sat\n'
      else
        printf 'unknown\n'
      fi
      ;;
    "(exit)")
      exit 0
      ;;
  esac
done"#,
    );

    let mut incremental = IncrementalSolver::new(&solver).expect("probe should succeed");
    let report = incremental.check_sat_with_report(Duration::from_secs(1), false);

    assert_eq!(report.outcomes(), &[SolverOutcome::Unknown]);
    let profile = report
        .solve_profile()
        .expect("incremental check should attach raw SMT profile")
        .as_row();
    assert!(profile.contains("MCC ay_solver_decision_profile_summary"));
    assert!(profile.contains("schema=ay.raw-smt-solve-profile-summary.v1"));
    assert!(profile.contains("source=raw_process_execution"));
    assert!(profile.contains("reason_code=raw_process_unknown"));
    assert!(profile.contains("decision_code=unknown"));
    assert!(profile.contains("accepted_for_consumer=true"));
    assert!(profile.contains("process_exit_code=0"));
    assert!(profile.contains("typed_consumer=false"));
    assert!(profile.contains("fail_closed=false"));
}

#[test]
fn test_incremental_solver_timeout_report_carries_deadline_flag() {
    let tempdir = TempDir::new().expect("tempdir should create");
    let solver = write_fake_solver_script(
        tempdir.path(),
        "fake-ay-incremental-raw-timeout",
        r#"probe_done=0
while IFS= read -r line; do
  case "$line" in
    "(check-sat)")
      if [ "$probe_done" -eq 0 ]; then
        probe_done=1
        printf 'sat\n'
      else
        sleep 5
      fi
      ;;
    "(exit)")
      exit 0
      ;;
  esac
done"#,
    );

    let mut incremental = IncrementalSolver::new(&solver).expect("probe should succeed");
    let report = incremental.check_sat_with_report(Duration::from_millis(50), true);

    assert_eq!(report.outcomes(), &[SolverOutcome::Unknown]);
    let profile = report
        .solve_profile()
        .expect("timeout check should attach raw SMT profile")
        .as_row();
    assert!(profile.contains("status=Unavailable"));
    assert!(profile.contains("reason_code=raw_process_timeout"));
    assert!(profile.contains("timed_out=true"));
    assert!(profile.contains("deadline_exceeded=true"));
    assert!(profile.contains("process_exit_code=none"));
    assert!(profile.contains("fail_closed=true"));
}

#[test]
fn test_incremental_solver_batch_only_orphan_pipe_fails_closed_quickly() {
    let tempdir = TempDir::new().expect("tempdir should create");
    let solver = write_fake_solver_script(
        tempdir.path(),
        "fake-ay-batch-only-orphan-pipe",
        r#"(while IFS= read -r _line; do :; done) &
exit 0"#,
    );

    let start = Instant::now();
    let incremental = IncrementalSolver::new(&solver);
    assert!(
        start.elapsed() < Duration::from_secs(3),
        "batch-only solvers that leave a pipe holder behind should fail closed quickly"
    );
    assert!(
        incremental.is_none(),
        "non-interactive batch solver should not be accepted as incremental"
    );
}

#[test]
fn test_incremental_solver_real_ay_startup_fails_closed_quickly() {
    let _guard = super::super::smt_encoding::ay_env_lock();

    let Some(ay_path) = super::super::smt_encoding::find_ay() else {
        eprintln!("SKIP: ay not available");
        return;
    };

    let start = Instant::now();
    let solver = IncrementalSolver::new(&ay_path);
    assert!(
        start.elapsed() < Duration::from_secs(3),
        "real-ay startup probing should either succeed or fail closed quickly"
    );
    drop(solver);
}
