// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

use std::fs;
#[cfg(unix)]
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::time::Duration;

use tempfile::TempDir;

use crate::petri_net::PetriNet;

use super::{
    emit_bmc_incremental_preamble, run_depth_ladder, run_depth_ladder_incremental,
    run_depth_ladder_with_report, DepthAction, DepthQuery, IncrementalPropertyQuery, SolverOutcome,
};

/// Per-depth solver budget for tests that expect the fake solver to actually
/// answer. The fake shell solver replies in milliseconds when scheduled; this
/// budget is only a safety bound. A tight 1s value misclassified a CPU-starved
/// subprocess (under full-parallel test load) as a timeout, flipping the real
/// `Sat`/`Unsat` outcome to the fail-closed `Unknown` shell and flaking the
/// assertion. Generous budget = same result when scheduled, no spurious timeout.
const FAKE_SOLVER_ANSWER_BUDGET: Duration = Duration::from_secs(30);

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

fn empty_net() -> PetriNet {
    PetriNet {
        name: Some("empty".to_string()),
        places: vec![],
        transitions: vec![],
        initial_marking: vec![],
    }
}

#[test]
fn test_run_depth_ladder_returns_last_explored_depth_before_unknown() {
    // These tests spawn a fake-solver subprocess that inherits the process-global
    // environment (PATH/HOME/AY_PATH, etc.). Other tests mutate those vars under
    // the same crate-wide env lock; hold it here so the inherited environment of
    // our spawned shell script is stable for the whole run. (Previously this raced
    // and flaked under full parallelism.)
    let _env = crate::env_test_lock();
    let tempdir = TempDir::new().expect("tempdir should create");
    let calls_path = tempdir.path().join("calls.log");
    let state_path = tempdir.path().join("state");
    let solver = write_fake_solver_script(
        tempdir.path(),
        "fake-ay-depth-runner-unknown",
        &format!(
            "printf 'call\\n' >> \"{}\"\ncat >/dev/null\nif [ ! -f \"{}\" ]; then\n  : > \"{}\"\n  printf 'unsat\\n'\nelse\n  printf 'unknown\\n'\nfi",
            calls_path.display(),
            state_path.display(),
            state_path.display()
        ),
    );

    let mut built_depths = Vec::new();
    let max_depth = run_depth_ladder(
        &solver,
        &[1, 2, 4],
        None,
        FAKE_SOLVER_ANSWER_BUDGET,
        &mut built_depths,
        |built_depths, depth| {
            built_depths.push(depth);
            Some(DepthQuery::new("(check-sat)\n".to_string(), 1))
        },
        |_, _, results| match results {
            Some([SolverOutcome::Unsat]) => DepthAction::Explored,
            Some([SolverOutcome::Unknown]) | None => DepthAction::StopDeepening,
            other => panic!("unexpected solver result: {other:?}"),
        },
    );

    if calls_path.exists() {
        assert_eq!(max_depth, Some(1));
        assert_eq!(built_depths, vec![1, 2]);
        assert_eq!(
            fs::read_to_string(&calls_path)
                .expect("call log should exist")
                .lines()
                .count(),
            2,
            "the helper should stop before trying the third ladder depth"
        );
    } else {
        assert_eq!(max_depth, None);
        assert_eq!(
            built_depths,
            vec![1],
            "solver startup failure should stop deepening after the first attempted depth"
        );
    }
}

#[test]
fn test_run_depth_ladder_stops_when_builder_has_no_more_work() {
    let _env = crate::env_test_lock();
    let tempdir = TempDir::new().expect("tempdir should create");
    let calls_path = tempdir.path().join("calls.log");
    let solver = write_fake_solver_script(
        tempdir.path(),
        "fake-ay-depth-runner-no-more-work",
        &format!(
            "printf 'call\\n' >> \"{}\"\ncat >/dev/null\nprintf 'unsat\\n'",
            calls_path.display()
        ),
    );

    let mut built_depths = Vec::new();
    let max_depth = run_depth_ladder(
        &solver,
        &[1, 2, 4],
        None,
        FAKE_SOLVER_ANSWER_BUDGET,
        &mut built_depths,
        |built_depths, depth| {
            built_depths.push(depth);
            if depth == 1 {
                Some(DepthQuery::new("(check-sat)\n".to_string(), 1))
            } else {
                None
            }
        },
        |_, _, results| match results {
            Some([SolverOutcome::Unsat]) => DepthAction::Explored,
            None => DepthAction::StopDeepening,
            other => panic!("unexpected solver result: {other:?}"),
        },
    );

    if calls_path.exists() {
        assert_eq!(max_depth, Some(1));
        assert_eq!(built_depths, vec![1, 2]);
        assert_eq!(
            fs::read_to_string(&calls_path)
                .expect("call log should exist")
                .lines()
                .count(),
            1,
            "the helper should not invoke the solver after build_query returns None"
        );
    } else {
        assert_eq!(max_depth, None);
        assert_eq!(
            built_depths,
            vec![1],
            "solver startup failure should stop before the builder reaches the second depth"
        );
    }
}

#[test]
fn test_run_depth_ladder_with_report_exposes_raw_smt_process_profile() {
    let _env = crate::env_test_lock();
    let tempdir = TempDir::new().expect("tempdir should create");
    let solver = write_fake_solver_script(
        tempdir.path(),
        "fake-ay-depth-runner-report",
        "cat >/dev/null\nprintf 'unsat\\n'",
    );

    let mut saw_profile = None;
    let max_depth = run_depth_ladder_with_report(
        &solver,
        &[1],
        None,
        FAKE_SOLVER_ANSWER_BUDGET,
        &mut saw_profile,
        |_, _| Some(DepthQuery::new("(check-sat)\n".to_string(), 1)),
        |saw_profile, _, report| match report {
            Some(report) => {
                assert_eq!(report.outcomes(), &[SolverOutcome::Unsat]);
                let profile = report
                    .solve_profile()
                    .expect("batch solver report should include raw SMT profile")
                    .as_row();
                assert!(profile.contains("MCC ay_solver_decision_profile_summary"));
                assert!(profile.contains("schema=ay.raw-smt-solve-profile-summary.v1"));
                assert!(profile.contains("reason_code=raw_process_status"));
                assert!(profile.contains("decision_code=unsat"));
                assert!(profile.contains("process_exit_code=0"));
                assert!(profile.contains("typed_consumer=false"));
                *saw_profile = Some(true);
                DepthAction::Explored
            }
            None => DepthAction::StopDeepening,
        },
    );

    if saw_profile.is_some() {
        assert_eq!(max_depth, Some(1));
        assert_eq!(
            saw_profile,
            Some(true),
            "the external-process parser should expose AY-owned raw SMT profile evidence"
        );
    } else {
        assert_eq!(max_depth, None);
    }
}

#[test]
fn test_incremental_depth_ladder_skips_next_depth_when_builder_has_no_more_work() {
    let _env = crate::env_test_lock();
    let tempdir = TempDir::new().expect("tempdir should create");
    let stdin_log_path = tempdir.path().join("stdin.log");
    let solver = write_fake_solver_script(
        tempdir.path(),
        "fake-ay-incremental-no-more-work",
        &format!(
            "probe_done=0\nwhile IFS= read -r line; do\n  printf '%s\\n' \"$line\" >> \"{}\"\n  case \"$line\" in\n    \"(check-sat)\")\n      if [ \"$probe_done\" -eq 0 ]; then\n        probe_done=1\n        printf 'sat\\n'\n      else\n        printf 'unsat\\n'\n      fi\n      ;;\n    \"(exit)\")\n      exit 0\n      ;;\n  esac\ndone",
            stdin_log_path.display()
        ),
    );
    let net = empty_net();
    let mut built_depths = Vec::new();

    let max_depth = run_depth_ladder_incremental(
        &solver,
        &[1, 2],
        None,
        FAKE_SOLVER_ANSWER_BUDGET,
        &net,
        &mut built_depths,
        emit_bmc_incremental_preamble,
        |built_depths, depth| {
            built_depths.push(depth);
            (depth == 1).then(|| IncrementalPropertyQuery {
                assertions: vec!["(assert true)\n".to_string()],
            })
        },
        |_, _, results| match results {
            Some([SolverOutcome::Unsat]) => DepthAction::Explored,
            Some([SolverOutcome::Unknown]) | None => DepthAction::StopDeepening,
            other => panic!("unexpected solver result: {other:?}"),
        },
        |_, _| panic!("incremental solver should not fall back to batch mode"),
        |_, _, _| panic!("incremental solver should not fall back to batch mode"),
    );

    assert_eq!(max_depth, Some(1));
    assert_eq!(built_depths, vec![1, 2]);

    let stdin_log = fs::read_to_string(&stdin_log_path).expect("stdin log should exist");
    assert_eq!(
        stdin_log.matches("(check-sat)").count(),
        2,
        "only the startup probe plus one property check should run after the builder returns None"
    );
    assert!(
        !stdin_log.contains("(assert (or stay_1))"),
        "the runner should not encode transition constraints for depth 2 when there is no work left: {stdin_log}"
    );
}

#[test]
fn test_incremental_depth_ladder_shares_one_timeout_budget_across_properties() {
    let _env = crate::env_test_lock();
    let tempdir = TempDir::new().expect("tempdir should create");
    let stdin_log_path = tempdir.path().join("stdin.log");
    let solver = write_fake_solver_script(
        tempdir.path(),
        "fake-ay-incremental-shared-timeout",
        &format!(
            "probe_done=0\nprop_checks=0\nwhile IFS= read -r line; do\n  printf '%s\\n' \"$line\" >> \"{}\"\n  case \"$line\" in\n    \"(check-sat)\")\n      if [ \"$probe_done\" -eq 0 ]; then\n        probe_done=1\n        printf 'sat\\n'\n      else\n        prop_checks=$((prop_checks + 1))\n        if [ \"$prop_checks\" -eq 1 ]; then\n          printf 'unsat\\n'\n        else\n          sleep 1\n          printf 'unsat\\n'\n        fi\n      fi\n      ;;\n    \"(exit)\")\n      exit 0\n      ;;\n  esac\ndone",
            stdin_log_path.display()
        ),
    );
    let net = empty_net();
    let mut seen_results = Vec::new();

    let max_depth = run_depth_ladder_incremental(
        &solver,
        &[1],
        None,
        Duration::from_millis(200),
        &net,
        &mut seen_results,
        emit_bmc_incremental_preamble,
        |_, _| {
            Some(IncrementalPropertyQuery {
                assertions: vec!["(assert true)\n".to_string(), "(assert true)\n".to_string()],
            })
        },
        |seen_results, _, results| {
            let outcomes = results.expect("incremental run should still report outcomes");
            seen_results.push(outcomes.to_vec());
            if outcomes
                .iter()
                .all(|outcome| *outcome == SolverOutcome::Unsat)
            {
                DepthAction::Explored
            } else {
                DepthAction::StopDeepening
            }
        },
        |_, _| panic!("incremental solver should not fall back to batch mode"),
        |_, _, _| panic!("incremental solver should not fall back to batch mode"),
    );

    assert_eq!(max_depth, None);
    assert_eq!(
        seen_results,
        vec![vec![SolverOutcome::Unsat, SolverOutcome::Unknown]],
        "the second property should inherit only the remaining depth budget"
    );

    let stdin_log = fs::read_to_string(&stdin_log_path).expect("stdin log should exist");
    assert_eq!(
        stdin_log.matches("(check-sat)").count(),
        3,
        "the runner should begin the second query after the startup probe and before the shared depth budget expires"
    );
}
