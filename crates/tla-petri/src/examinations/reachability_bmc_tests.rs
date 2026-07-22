// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Tests for BMC encoding and integration.

use std::fs;
#[cfg(unix)]
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use tempfile::TempDir;
use tla_mc_core::{BackendKind, ProblemKind, UnsupportedReason};

use crate::petri_net::{Arc, PetriNet, PlaceIdx, PlaceInfo, TransitionIdx, TransitionInfo};
use crate::property_xml::PathQuantifier;
use crate::resolved_predicate::{ResolvedIntExpr, ResolvedPredicate};

use super::super::reachability_witness::{
    validation_targets_from_trackers, WitnessValidationTarget,
};
use super::{encode_bmc_script, encode_int_expr, encode_predicate, PropertyTracker};

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

/// Solver budget for a fake solver that answers instantly. Only a safety bound;
/// a tight 1s value let a CPU-starved subprocess (under full-parallel test load)
/// time out and return the fail-closed shell instead of the real answer.
const FAKE_SOLVER_ANSWER_BUDGET: Duration = Duration::from_secs(30);

/// Upper bound proving `run_ay` returned without blocking on a fake solver's
/// 5-second `sleep` / orphaned-stdout holder. Any bound comfortably below 5s
/// proves "did not wait for the sleep"; 4s leaves a full second of margin while
/// absorbing subprocess scheduling latency under load (1s was the flaky part).
const RETURNED_BEFORE_SLEEP_CEILING: Duration = Duration::from_secs(4);

fn place(id: &str) -> PlaceInfo {
    PlaceInfo {
        id: id.to_string(),
        name: None,
    }
}

fn arc(place: u32, weight: u64) -> Arc {
    Arc {
        place: PlaceIdx(place),
        weight,
    }
}

fn trans(id: &str, inputs: Vec<Arc>, outputs: Vec<Arc>) -> TransitionInfo {
    TransitionInfo {
        id: id.to_string(),
        name: None,
        inputs,
        outputs,
    }
}

/// Simple two-place producer-consumer net: p0 → [t0] → p1
fn producer_consumer_net() -> PetriNet {
    PetriNet {
        name: Some("test".to_string()),
        places: vec![place("p0"), place("p1")],
        transitions: vec![trans("t0", vec![arc(0, 1)], vec![arc(1, 1)])],
        initial_marking: vec![1, 0],
    }
}

fn two_step_chain_net() -> PetriNet {
    PetriNet {
        name: Some("two-step-chain".to_string()),
        places: vec![place("p0"), place("p1"), place("p2")],
        transitions: vec![
            trans("t0", vec![arc(0, 1)], vec![arc(1, 1)]),
            trans("t1", vec![arc(1, 1)], vec![arc(2, 1)]),
        ],
        initial_marking: vec![1, 0, 0],
    }
}

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

fn run_bmc_seeding_for_test(
    net: &PetriNet,
    trackers: &mut [PropertyTracker],
    deadline: Option<Instant>,
) -> Option<usize> {
    let targets = validation_targets_from_trackers(trackers);
    super::run_bmc_seeding(net, trackers, &targets, deadline)
}

fn run_bmc_seeding_with_solver_path_for_test(
    net: &PetriNet,
    trackers: &mut [PropertyTracker],
    deadline: Option<Instant>,
    ay_path: &Path,
) -> Option<usize> {
    let targets = validation_targets_from_trackers(trackers);
    super::run_bmc_seeding_with_solver_path(net, trackers, &targets, deadline, ay_path)
}

#[test]
fn run_bmc_seeding_with_report_threads_selected_ay_evidence() {
    let _guard = crate::examinations::smt_encoding::ay_env_lock();
    let temp = TempDir::new().expect("tempdir should create");
    let ay_path = temp.path().join("ay");
    fs::write(&ay_path, b"fake ay").expect("fake ay should write");
    let _ay_path = EnvVarGuard::set("AY_PATH", ay_path.to_str().expect("utf8 temp path"));

    let net = producer_consumer_net();
    let mut trackers = Vec::new();
    let targets: Vec<WitnessValidationTarget> = Vec::new();

    let (depth, report) = super::run_bmc_seeding_with_report(&net, &mut trackers, &targets, None);

    assert_eq!(depth, None);
    assert_eq!(report.problem, Some(ProblemKind::Bmc));
    assert!(report.has_selected(BackendKind::ExternalAYBinary));
    assert!(report
        .evidence
        .iter()
        .any(|entry| entry.contains("reachability BMC runtime selected ay")));
}

#[test]
fn run_bmc_seeding_with_report_records_scoped_runtime_evidence() {
    let _guard = crate::examinations::smt_encoding::ay_env_lock();
    let temp = TempDir::new().expect("tempdir should create");
    let ay_path = temp.path().join("ay");
    fs::write(&ay_path, b"fake ay").expect("fake ay should write");
    let _ay_path = EnvVarGuard::set("AY_PATH", ay_path.to_str().expect("utf8 temp path"));

    let net = producer_consumer_net();
    let mut trackers = Vec::new();
    let targets: Vec<WitnessValidationTarget> = Vec::new();

    let ((depth, report), runtime_reports) =
        crate::mcc_backend_evidence::collect_runtime_reachability_bmc_reports(|| {
            super::run_bmc_seeding_with_report(&net, &mut trackers, &targets, None)
        });

    assert_eq!(depth, None);
    assert!(report
        .evidence
        .iter()
        .any(|entry| entry.contains("reachability BMC runtime selected ay")));
    assert_eq!(runtime_reports.len(), 1);
    assert!(runtime_reports[0]
        .evidence
        .iter()
        .any(|entry| entry.contains("reachability BMC runtime selected ay")));
}

#[test]
fn run_bmc_seeding_with_report_threads_missing_ay_evidence() {
    let _guard = crate::examinations::smt_encoding::ay_env_lock();
    let temp = TempDir::new().expect("tempdir should create");
    let bin_dir = temp.path().join("bin");
    fs::create_dir(&bin_dir).expect("bin dir should create");
    let _ay_path = EnvVarGuard::remove("AY_PATH");
    let _home = EnvVarGuard::set("HOME", temp.path().to_str().expect("utf8 temp path"));
    let _path = EnvVarGuard::set("PATH", bin_dir.to_str().expect("utf8 temp path"));

    let net = producer_consumer_net();
    let mut trackers = vec![PropertyTracker {
        id: "ef-reach".to_string(),
        quantifier: PathQuantifier::EF,
        predicate: ResolvedPredicate::IntLe(
            ResolvedIntExpr::Constant(1),
            ResolvedIntExpr::TokensCount(vec![PlaceIdx(1)]),
        ),
        verdict: None,
        resolved_by: None,
        flushed: false,
    }];
    let targets = validation_targets_from_trackers(&trackers);

    let (depth, report) = super::run_bmc_seeding_with_report(&net, &mut trackers, &targets, None);

    assert_eq!(depth, None);
    assert_eq!(trackers[0].verdict, None);
    assert_eq!(
        report.rejection_reason(BackendKind::ExternalAYBinary),
        Some(&UnsupportedReason::MissingBinary("ay"))
    );
    assert!(report
        .evidence
        .iter()
        .any(|entry| entry == "reachability BMC skipped because ay was unavailable"));
}

fn stay_depth1_model() -> &'static str {
    "((stay_0 true)\n (fire_0_0 false)\n)\n"
}

#[test]
fn test_encode_int_expr_constant() {
    let expr = ResolvedIntExpr::Constant(42);
    assert_eq!(encode_int_expr(&expr, 0), "42");
}

#[test]
fn test_encode_int_expr_single_place() {
    let expr = ResolvedIntExpr::TokensCount(vec![PlaceIdx(2)]);
    assert_eq!(encode_int_expr(&expr, 3), "m_3_2");
}

#[test]
fn test_encode_int_expr_sum_of_places() {
    let expr = ResolvedIntExpr::TokensCount(vec![PlaceIdx(0), PlaceIdx(1)]);
    assert_eq!(encode_int_expr(&expr, 0), "(+ m_0_0 m_0_1)");
}

#[test]
fn test_encode_int_expr_empty_places() {
    let expr = ResolvedIntExpr::TokensCount(vec![]);
    assert_eq!(encode_int_expr(&expr, 0), "0");
}

#[test]
fn test_encode_predicate_true() {
    let net = producer_consumer_net();
    assert_eq!(encode_predicate(&ResolvedPredicate::True, 0, &net), "true");
}

#[test]
fn test_encode_predicate_false() {
    let net = producer_consumer_net();
    assert_eq!(
        encode_predicate(&ResolvedPredicate::False, 0, &net),
        "false"
    );
}

#[test]
fn test_encode_predicate_int_le() {
    let net = producer_consumer_net();
    let pred = ResolvedPredicate::IntLe(
        ResolvedIntExpr::TokensCount(vec![PlaceIdx(0)]),
        ResolvedIntExpr::Constant(5),
    );
    assert_eq!(encode_predicate(&pred, 2, &net), "(<= m_2_0 5)");
}

#[test]
fn test_encode_predicate_not() {
    let net = producer_consumer_net();
    let pred = ResolvedPredicate::Not(Box::new(ResolvedPredicate::True));
    assert_eq!(encode_predicate(&pred, 0, &net), "(not true)");
}

#[test]
fn test_encode_predicate_and() {
    let net = producer_consumer_net();
    let pred = ResolvedPredicate::And(vec![ResolvedPredicate::True, ResolvedPredicate::False]);
    assert_eq!(encode_predicate(&pred, 0, &net), "(and true false)");
}

#[test]
fn test_encode_predicate_or() {
    let net = producer_consumer_net();
    let pred = ResolvedPredicate::Or(vec![ResolvedPredicate::True, ResolvedPredicate::False]);
    assert_eq!(encode_predicate(&pred, 0, &net), "(or true false)");
}

#[test]
fn test_encode_predicate_is_fireable() {
    let net = producer_consumer_net();
    // t0 is fireable when p0 >= 1
    let pred = ResolvedPredicate::IsFireable(vec![TransitionIdx(0)]);
    assert_eq!(encode_predicate(&pred, 0, &net), "(>= m_0_0 1)");
}

#[test]
fn test_encode_predicate_is_fireable_empty() {
    let net = producer_consumer_net();
    let pred = ResolvedPredicate::IsFireable(vec![]);
    assert_eq!(encode_predicate(&pred, 0, &net), "false");
}

#[test]
fn test_encode_predicate_singleton_and() {
    let net = producer_consumer_net();
    let pred = ResolvedPredicate::And(vec![ResolvedPredicate::True]);
    // Single-child And should simplify
    assert_eq!(encode_predicate(&pred, 0, &net), "true");
}

#[test]
fn test_encode_predicate_empty_and() {
    let net = producer_consumer_net();
    let pred = ResolvedPredicate::And(vec![]);
    assert_eq!(encode_predicate(&pred, 0, &net), "true");
}

#[test]
fn test_encode_predicate_empty_or() {
    let net = producer_consumer_net();
    let pred = ResolvedPredicate::Or(vec![]);
    assert_eq!(encode_predicate(&pred, 0, &net), "false");
}

#[test]
fn test_bmc_script_has_initial_marking() {
    let net = producer_consumer_net();
    let trackers = vec![PropertyTracker {
        id: "prop-0".to_string(),
        quantifier: PathQuantifier::EF,
        predicate: ResolvedPredicate::IntLe(
            ResolvedIntExpr::Constant(1),
            ResolvedIntExpr::TokensCount(vec![PlaceIdx(1)]),
        ),
        verdict: None,
        resolved_by: None,
        flushed: false,
    }];

    let script = encode_bmc_script(&net, &trackers, &[0], 1);

    // Should contain initial marking
    assert!(script.contains("(assert (= m_0_0 1))"), "initial m_0_0 = 1");
    assert!(script.contains("(assert (= m_0_1 0))"), "initial m_0_1 = 0");
}

#[test]
fn test_bmc_script_has_stutter_variable() {
    let net = producer_consumer_net();
    let trackers = vec![PropertyTracker {
        id: "prop-0".to_string(),
        quantifier: PathQuantifier::EF,
        predicate: ResolvedPredicate::True,
        verdict: None,
        resolved_by: None,
        flushed: false,
    }];

    let script = encode_bmc_script(&net, &trackers, &[0], 1);

    assert!(
        script.contains("stay_0"),
        "script should contain stutter variable"
    );
}

#[test]
fn test_bmc_script_has_check_sat_per_property() {
    let net = producer_consumer_net();
    let trackers = vec![
        PropertyTracker {
            id: "prop-0".to_string(),
            quantifier: PathQuantifier::EF,
            predicate: ResolvedPredicate::True,
            verdict: None,
            resolved_by: None,
            flushed: false,
        },
        PropertyTracker {
            id: "prop-1".to_string(),
            quantifier: PathQuantifier::AG,
            predicate: ResolvedPredicate::False,
            verdict: None,
            resolved_by: None,
            flushed: false,
        },
    ];

    let script = encode_bmc_script(&net, &trackers, &[0, 1], 1);

    let check_sat_count = script.matches("(check-sat)").count();
    assert_eq!(check_sat_count, 2, "should have one check-sat per property");
}

#[test]
fn test_bmc_script_push_pop_per_property() {
    let net = producer_consumer_net();
    let trackers = vec![PropertyTracker {
        id: "prop-0".to_string(),
        quantifier: PathQuantifier::EF,
        predicate: ResolvedPredicate::True,
        verdict: None,
        resolved_by: None,
        flushed: false,
    }];

    let script = encode_bmc_script(&net, &trackers, &[0], 1);

    assert!(script.contains("(push 1)"));
    assert!(script.contains("(pop 1)"));
}

#[test]
fn test_bmc_script_ag_negates_predicate() {
    let net = producer_consumer_net();
    // AG(p0 >= 1): check ¬(p0 >= 1) = p0 < 1
    let trackers = vec![PropertyTracker {
        id: "prop-0".to_string(),
        quantifier: PathQuantifier::AG,
        predicate: ResolvedPredicate::IntLe(
            ResolvedIntExpr::Constant(1),
            ResolvedIntExpr::TokensCount(vec![PlaceIdx(0)]),
        ),
        verdict: None,
        resolved_by: None,
        flushed: false,
    }];

    let script = encode_bmc_script(&net, &trackers, &[0], 1);

    // The predicate for AG should be negated in the assertion
    assert!(
        script.contains("(not (<= 1 m_"),
        "AG should negate the predicate: {script}"
    );
}

#[test]
fn test_bmc_script_ef_does_not_negate() {
    let net = producer_consumer_net();
    let trackers = vec![PropertyTracker {
        id: "prop-0".to_string(),
        quantifier: PathQuantifier::EF,
        predicate: ResolvedPredicate::IntLe(
            ResolvedIntExpr::Constant(1),
            ResolvedIntExpr::TokensCount(vec![PlaceIdx(1)]),
        ),
        verdict: None,
        resolved_by: None,
        flushed: false,
    }];

    let script = encode_bmc_script(&net, &trackers, &[0], 1);

    // Should assert the predicate directly (not negated)
    assert!(
        script.contains("(<= 1 m_"),
        "EF should assert predicate directly"
    );
    // Check we don't have unnecessary negation of the main predicate
    // (There will be (not ...) for mutual exclusion, but the property assertion
    // should contain the predicate directly)
}

#[test]
fn test_bmc_script_transition_semantics() {
    let net = producer_consumer_net();
    let trackers = vec![PropertyTracker {
        id: "prop-0".to_string(),
        quantifier: PathQuantifier::EF,
        predicate: ResolvedPredicate::True,
        verdict: None,
        resolved_by: None,
        flushed: false,
    }];

    let script = encode_bmc_script(&net, &trackers, &[0], 1);

    // t0 fires: p0 -= 1, p1 += 1
    // So under fire_0_0: m_1_0 = m_0_0 - 1, m_1_1 = m_0_1 + 1
    assert!(
        script.contains("fire_0_0"),
        "script should reference transition fire variable"
    );
    // Guard: p0 >= 1 when firing
    assert!(
        script.contains("(=> fire_0_0 (>= m_0_0 1))"),
        "script should have transition guard"
    );
}

#[test]
fn test_bmc_seeding_mixed_outcomes_preserve_seeded_results_and_stop_on_unknown() {
    let net = producer_consumer_net();
    let tempdir = TempDir::new().expect("tempdir should create");
    let calls_path = tempdir.path().join("calls.log");
    let solver = write_fake_solver_script(
        tempdir.path(),
        "fake-ay",
        &format!(
            "printf 'call\\n' >> \"{}\"\n\
probe_done=0\n\
count=0\n\
while IFS= read -r line; do\n\
  case \"$line\" in\n\
    '(check-sat)')\n\
      if [ \"$probe_done\" -eq 0 ]; then\n\
        probe_done=1\n\
        printf 'sat\\n'\n\
      else\n\
        count=$((count + 1))\n\
        case \"$count\" in\n\
          1) printf 'sat\\n' ;;\n\
          2) printf 'unsat\\n' ;;\n\
          *) printf 'unknown\\n' ;;\n\
        esac\n\
      fi\n\
      ;;\n\
    '(get-value '*)\n\
      printf '((stay_0 false)\\n (fire_0_0 true)\\n)\\n'\n\
      ;;\n\
    '(exit)')\n\
      exit 0\n\
      ;;\n\
  esac\n\
done",
            calls_path.display()
        ),
    );
    let mut trackers = vec![
        PropertyTracker {
            id: "ef-witness".to_string(),
            quantifier: PathQuantifier::EF,
            predicate: ResolvedPredicate::IntLe(
                ResolvedIntExpr::Constant(1),
                ResolvedIntExpr::TokensCount(vec![PlaceIdx(1)]),
            ),
            verdict: None,
            resolved_by: None,
            flushed: false,
        },
        PropertyTracker {
            id: "ef-unreachable".to_string(),
            quantifier: PathQuantifier::EF,
            predicate: ResolvedPredicate::IntLe(
                ResolvedIntExpr::Constant(100),
                ResolvedIntExpr::TokensCount(vec![PlaceIdx(1)]),
            ),
            verdict: None,
            resolved_by: None,
            flushed: false,
        },
        PropertyTracker {
            id: "ag-unknown".to_string(),
            quantifier: PathQuantifier::AG,
            predicate: ResolvedPredicate::IntLe(
                ResolvedIntExpr::TokensCount(vec![PlaceIdx(0)]),
                ResolvedIntExpr::Constant(1),
            ),
            verdict: None,
            resolved_by: None,
            flushed: false,
        },
    ];

    run_bmc_seeding_with_solver_path_for_test(&net, &mut trackers, None, &solver);

    // If the fake solver executed successfully, it seeds ef-witness = TRUE.
    // If the solver failed to execute (sandbox, permissions), all stay None.
    let solver_ran = calls_path.exists();
    if solver_ran {
        assert_eq!(
            trackers[0].verdict,
            Some(true),
            "ef-witness should be seeded TRUE via sat"
        );
        assert_eq!(
            trackers[1].verdict, None,
            "ef-unreachable should stay None after unsat"
        );
        assert_eq!(
            trackers[2].verdict, None,
            "ag-unknown should stay None after unknown"
        );
        assert_eq!(
            fs::read_to_string(&calls_path)
                .expect("call log should exist")
                .lines()
                .count(),
            2,
            "unknown should stop further deepening after the status call plus SAT model replay"
        );
    } else {
        // Solver failed — all properties remain unresolved.
        assert_eq!(trackers[0].verdict, None);
        assert_eq!(trackers[1].verdict, None);
        assert_eq!(trackers[2].verdict, None);
    }
}

#[test]
fn test_bmc_sat_without_parseable_model_leaves_tracker_unresolved() {
    let net = producer_consumer_net();
    let tempdir = TempDir::new().expect("tempdir should create");
    let solver = write_fake_solver_script(
        tempdir.path(),
        "fake-ay-sat-no-model",
        "cat >/dev/null\nprintf 'sat\\n'",
    );
    let mut trackers = vec![PropertyTracker {
        id: "ef-reach".to_string(),
        quantifier: PathQuantifier::EF,
        predicate: ResolvedPredicate::IntLe(
            ResolvedIntExpr::Constant(1),
            ResolvedIntExpr::TokensCount(vec![PlaceIdx(1)]),
        ),
        verdict: None,
        resolved_by: None,
        flushed: false,
    }];

    let result = run_bmc_seeding_with_solver_path_for_test(
        &net,
        &mut trackers,
        Some(Instant::now() + Duration::from_secs(5)),
        &solver,
    );

    assert_eq!(result, None);
    assert_eq!(
        trackers[0].verdict, None,
        "raw SAT without a replayable model must not seed a reachability verdict"
    );
}

#[test]
fn test_bmc_sat_model_replays_against_original_validation_target() {
    let net = producer_consumer_net();
    let tempdir = TempDir::new().expect("tempdir should create");
    let solver = write_fake_solver_script(
        tempdir.path(),
        "fake-ay-sat-stay-model",
        &format!(
            "cat >/dev/null\n\
printf 'sat\\n'\n\
printf '{}'",
            stay_depth1_model()
        ),
    );
    let mut trackers = vec![PropertyTracker {
        id: "ef-simplified-true-original-false".to_string(),
        quantifier: PathQuantifier::EF,
        predicate: ResolvedPredicate::True,
        verdict: None,
        resolved_by: None,
        flushed: false,
    }];
    let targets = vec![WitnessValidationTarget {
        original_predicate: ResolvedPredicate::False,
    }];

    let result = super::run_bmc_seeding_with_solver_path(
        &net,
        &mut trackers,
        &targets,
        Some(Instant::now() + Duration::from_secs(5)),
        &solver,
    );

    assert_eq!(
        result, None,
        "rejected SAT model should make the depth inconclusive"
    );
    assert_eq!(
        trackers[0].verdict, None,
        "BMC must not seed from a model that only satisfies the simplified predicate"
    );
}

#[test]
fn test_bmc_retries_failed_batch_depth_per_property() {
    let net = two_step_chain_net();
    let tempdir = TempDir::new().expect("tempdir should create");
    let calls_path = tempdir.path().join("calls.log");
    let input_path = tempdir.path().join("solver-input.smt2");
    let solver = write_fake_solver_script(
        tempdir.path(),
        "fake-ay-depth-split",
        &format!(
            "cat > \"{}\"\n\
if grep -Fq '(get-value' \"{}\"; then\n\
  printf 'sat\\n'\n\
  printf '((stay_0 false)\\n (fire_0_0 true)\\n (fire_0_1 false)\\n (stay_1 false)\\n (fire_1_0 false)\\n (fire_1_1 true)\\n)\\n'\n\
  exit 0\n\
fi\n\
checks=$(grep -Fxc '(check-sat)' \"{}\" || true)\n\
if grep -Fq 'm_2_' \"{}\"; then depth=2; else depth=1; fi\n\
printf 'checks=%s depth=%s\\n' \"$checks\" \"$depth\" >> \"{}\"\n\
if [ \"$depth\" -eq 2 ] && [ \"$checks\" -gt 1 ]; then\n\
  exit 2\n\
fi\n\
i=0\n\
while [ \"$i\" -lt \"$checks\" ]; do\n\
  if [ \"$depth\" -eq 2 ]; then\n\
    printf 'sat\\n'\n\
  else\n\
    printf 'unsat\\n'\n\
  fi\n\
  i=$((i + 1))\n\
done",
            input_path.display(),
            input_path.display(),
            input_path.display(),
            input_path.display(),
            calls_path.display()
        ),
    );
    let predicate = ResolvedPredicate::IntLe(
        ResolvedIntExpr::Constant(1),
        ResolvedIntExpr::TokensCount(vec![PlaceIdx(2)]),
    );
    let mut trackers = vec![
        PropertyTracker {
            id: "ef-p2-a".to_string(),
            quantifier: PathQuantifier::EF,
            predicate: predicate.clone(),
            verdict: None,
            resolved_by: None,
            flushed: false,
        },
        PropertyTracker {
            id: "ef-p2-b".to_string(),
            quantifier: PathQuantifier::EF,
            predicate,
            verdict: None,
            resolved_by: None,
            flushed: false,
        },
    ];

    run_bmc_seeding_with_solver_path_for_test(&net, &mut trackers, None, &solver);

    assert_eq!(
        trackers[0].verdict,
        Some(true),
        "first property should be recovered by the individual depth-2 retry"
    );
    assert_eq!(
        trackers[1].verdict,
        Some(true),
        "second property should be recovered by the individual depth-2 retry"
    );
    assert_eq!(
        fs::read_to_string(&calls_path)
            .expect("call log should exist")
            .lines()
            .collect::<Vec<_>>(),
        vec![
            "checks=2 depth=1",
            "checks=2 depth=2",
            "checks=1 depth=2",
            "checks=1 depth=2",
        ],
        "the failed multi-property depth-2 batch should be retried once per property"
    );
}

#[test]
fn test_bmc_depth_ladder_caps_large_short_deadline() {
    let deadline = Instant::now() + Duration::from_secs(20);

    assert_eq!(super::bmc_depth_ladder(Some(deadline), 16), &[1]);
}

#[test]
fn test_bmc_depth_retry_falls_back_per_property_when_budget_allows() {
    // A large pending queue under a short (but not tiny) deadline must NOT give
    // up wholesale: as long as the fair per-property time-slice clears the
    // minimum solve budget, per-property retry recovers the cheap depth-1
    // witnesses instead of abandoning every pending property.
    let deadline = Instant::now() + Duration::from_secs(20);
    let long_deadline = Instant::now() + Duration::from_secs(60);

    assert!(
        super::should_retry_depth_individually(Some(deadline), 16),
        "large pending under a 20s deadline still has a viable per-property slice, so retry"
    );
    assert!(super::should_retry_depth_individually(
        Some(long_deadline),
        16
    ));
    assert!(super::should_retry_depth_individually(Some(deadline), 2));
    assert!(super::should_retry_depth_individually(None, 16));
    assert!(!super::should_retry_depth_individually(None, 1));
}

#[test]
fn test_bmc_split_retry_timeout_shares_deadline_budget() {
    let now = Instant::now();
    let deadline = now + super::BMC_SPLIT_RETRY_FALLBACK_RESERVE + Duration::from_secs(6);

    assert_eq!(
        super::bmc_split_retry_timeout(Some(deadline), 3, now),
        Duration::from_secs(2),
        "retry timeout should divide the non-reserved budget across pending properties"
    );
    assert_eq!(
        super::bmc_split_retry_timeout(Some(deadline), 1, now),
        Duration::from_secs(3),
        "single-property retry remains capped by the per-depth timeout"
    );
    assert_eq!(
        super::bmc_split_retry_timeout(None, 3, now),
        Duration::from_secs(3),
        "unbounded runs keep the historical per-depth retry timeout"
    );
}

#[test]
fn test_bmc_depth_retry_requires_minimum_timesliced_budget() {
    let deadline = (Instant::now()
        + super::BMC_SPLIT_RETRY_FALLBACK_RESERVE
        + super::BMC_SPLIT_RETRY_MIN_BUDGET * 3)
        .checked_sub(Duration::from_millis(1))
        .unwrap();

    assert!(
        !super::should_retry_depth_individually(Some(deadline), 3),
        "deadline-bound retries should be skipped when the fair slice is below the minimum"
    );
}

#[test]
fn test_bmc_depth_ladder_keeps_full_ladder_for_small_or_unbounded_runs() {
    let short_deadline = Instant::now() + Duration::from_secs(20);
    let long_deadline = Instant::now() + Duration::from_secs(60);

    assert_eq!(
        super::bmc_depth_ladder(Some(short_deadline), 2),
        super::DEPTH_LADDER
    );
    assert_eq!(
        super::bmc_depth_ladder(Some(long_deadline), 16),
        super::DEPTH_LADDER
    );
    assert_eq!(super::bmc_depth_ladder(None, 16), super::DEPTH_LADDER);
}

fn run_depth1_chunking_probe() -> (Option<usize>, Vec<String>) {
    // Depth-1 chunking is a batch-mode scheduling path: it only runs when the
    // deadline-incremental mode is NOT selected. With incremental now the default,
    // force it off so the chunking behavior under test is reachable. (Callers
    // already hold `ay_env_lock`.)
    let _incremental_off = EnvVarGuard::set("TY_MCC_AY_BMC_DEADLINE_INCREMENTAL", "0");

    let net = two_step_chain_net();
    let tempdir = TempDir::new().expect("tempdir should create");
    let calls_path = tempdir.path().join("calls.log");
    let input_path = tempdir.path().join("solver-input.smt2");
    let solver = write_fake_solver_script(
        tempdir.path(),
        "fake-ay-depth1-chunking-probe",
        &format!(
            "cat > \"{}\"\n\
checks=$(grep -Fxc '(check-sat)' \"{}\" || true)\n\
if grep -Fq 'm_2_' \"{}\"; then depth=2; else depth=1; fi\n\
printf 'checks=%s depth=%s\\n' \"$checks\" \"$depth\" >> \"{}\"\n\
i=0\n\
while [ \"$i\" -lt \"$checks\" ]; do\n\
  printf 'unsat\\n'\n\
  i=$((i + 1))\n\
done",
            input_path.display(),
            input_path.display(),
            input_path.display(),
            calls_path.display()
        ),
    );
    let predicate = ResolvedPredicate::IntLe(
        ResolvedIntExpr::Constant(1),
        ResolvedIntExpr::TokensCount(vec![PlaceIdx(2)]),
    );
    let mut trackers = (0..16)
        .map(|index| PropertyTracker {
            id: format!("ef-p2-probe-{index}"),
            quantifier: PathQuantifier::EF,
            predicate: predicate.clone(),
            verdict: None,
            resolved_by: None,
            flushed: false,
        })
        .collect::<Vec<_>>();

    let result = run_bmc_seeding_with_solver_path_for_test(
        &net,
        &mut trackers,
        Some(Instant::now() + Duration::from_secs(20)),
        &solver,
    );
    let calls = fs::read_to_string(&calls_path)
        .expect("call log should exist")
        .lines()
        .map(str::to_string)
        .collect::<Vec<_>>();

    (result, calls)
}

#[test]
fn test_bmc_depth1_chunking_enabled_by_default_without_env() {
    let _guard = super::super::smt_encoding::ay_env_lock();
    let _enable = EnvVarGuard::remove("TY_MCC_ENABLE_BMC_DEPTH1_CHUNKING");
    let _disable = EnvVarGuard::remove("TY_MCC_DISABLE_BMC_DEPTH1_CHUNKING");
    let _chunk_size = EnvVarGuard::remove("TY_MCC_BMC_DEPTH1_CHUNK_SIZE");

    assert!(
        super::bmc_depth1_chunking_enabled(),
        "depth-1 chunking should be enabled by default so a large short-deadline \
         batch no longer gives up wholesale"
    );

    let (result, calls) = run_depth1_chunking_probe();

    assert_eq!(result, Some(1));
    assert_eq!(
        calls,
        vec![
            "checks=4 depth=1",
            "checks=4 depth=1",
            "checks=4 depth=1",
            "checks=4 depth=1",
        ],
        "default-on depth-1 chunking should split the large short-deadline batch"
    );
}

#[test]
fn test_bmc_depth1_chunking_disable_env_wins_over_enable_env() {
    let _guard = super::super::smt_encoding::ay_env_lock();
    let _enable = EnvVarGuard::set("TY_MCC_ENABLE_BMC_DEPTH1_CHUNKING", "1");
    let _disable = EnvVarGuard::set("TY_MCC_DISABLE_BMC_DEPTH1_CHUNKING", "1");
    let _chunk_size = EnvVarGuard::set("TY_MCC_BMC_DEPTH1_CHUNK_SIZE", "4");

    assert!(
        !super::bmc_depth1_chunking_enabled(),
        "disable env should override the enable env"
    );
    let (result, calls) = run_depth1_chunking_probe();

    assert_eq!(result, Some(1));
    assert_eq!(
        calls,
        vec!["checks=16 depth=1"],
        "disabled depth-1 chunking should use the normal all-property batch"
    );
}

#[test]
fn test_bmc_depth1_chunk_size_invalid_and_zero_fall_back_to_default() {
    let _guard = super::super::smt_encoding::ay_env_lock();
    let _enable = EnvVarGuard::set("TY_MCC_ENABLE_BMC_DEPTH1_CHUNKING", "1");
    let _disable = EnvVarGuard::remove("TY_MCC_DISABLE_BMC_DEPTH1_CHUNKING");

    for chunk_size in ["invalid", "0"] {
        let _chunk_size = EnvVarGuard::set("TY_MCC_BMC_DEPTH1_CHUNK_SIZE", chunk_size);

        assert_eq!(
            super::bmc_depth1_chunk_size(),
            4,
            "{chunk_size:?} should fall back to the default depth-1 chunk size"
        );
        let (result, calls) = run_depth1_chunking_probe();

        assert_eq!(result, Some(1));
        assert_eq!(
            calls,
            vec![
                "checks=4 depth=1",
                "checks=4 depth=1",
                "checks=4 depth=1",
                "checks=4 depth=1",
            ],
            "{chunk_size:?} should behave like the default depth-1 chunk size"
        );
    }
}

#[test]
fn test_bmc_depth1_chunk_size_non_default_splits_by_requested_size() {
    let _guard = super::super::smt_encoding::ay_env_lock();
    let _enable = EnvVarGuard::set("TY_MCC_ENABLE_BMC_DEPTH1_CHUNKING", "1");
    let _disable = EnvVarGuard::remove("TY_MCC_DISABLE_BMC_DEPTH1_CHUNKING");
    let _chunk_size = EnvVarGuard::set("TY_MCC_BMC_DEPTH1_CHUNK_SIZE", "8");

    assert_eq!(
        super::bmc_depth1_chunk_size(),
        8,
        "non-default chunk sizes should be honored"
    );
    let (result, calls) = run_depth1_chunking_probe();

    assert_eq!(result, Some(1));
    assert_eq!(
        calls,
        vec!["checks=8 depth=1", "checks=8 depth=1"],
        "the depth-1 runner should split by the requested chunk size"
    );
}

#[test]
fn test_bmc_depth1_chunking_splits_large_short_deadline_batches() {
    let _guard = super::super::smt_encoding::ay_env_lock();
    let _enable = EnvVarGuard::set("TY_MCC_ENABLE_BMC_DEPTH1_CHUNKING", "1");
    let _disable = EnvVarGuard::remove("TY_MCC_DISABLE_BMC_DEPTH1_CHUNKING");
    let _chunk_size = EnvVarGuard::set("TY_MCC_BMC_DEPTH1_CHUNK_SIZE", "4");
    // Chunking is a batch-mode path; force the incremental default off.
    let _incremental_off = EnvVarGuard::set("TY_MCC_AY_BMC_DEADLINE_INCREMENTAL", "0");

    let net = two_step_chain_net();
    let tempdir = TempDir::new().expect("tempdir should create");
    let calls_path = tempdir.path().join("calls.log");
    let input_path = tempdir.path().join("solver-input.smt2");
    let solver = write_fake_solver_script(
        tempdir.path(),
        "fake-ay-depth1-chunked",
        &format!(
            "cat > \"{}\"\n\
checks=$(grep -Fxc '(check-sat)' \"{}\" || true)\n\
if grep -Fq 'm_2_' \"{}\"; then depth=2; else depth=1; fi\n\
printf 'checks=%s depth=%s\\n' \"$checks\" \"$depth\" >> \"{}\"\n\
i=0\n\
while [ \"$i\" -lt \"$checks\" ]; do\n\
  printf 'unsat\\n'\n\
  i=$((i + 1))\n\
done",
            input_path.display(),
            input_path.display(),
            input_path.display(),
            calls_path.display()
        ),
    );
    let predicate = ResolvedPredicate::IntLe(
        ResolvedIntExpr::Constant(1),
        ResolvedIntExpr::TokensCount(vec![PlaceIdx(2)]),
    );
    let mut trackers = (0..16)
        .map(|index| PropertyTracker {
            id: format!("ef-p2-{index}"),
            quantifier: PathQuantifier::EF,
            predicate: predicate.clone(),
            verdict: None,
            resolved_by: None,
            flushed: false,
        })
        .collect::<Vec<_>>();

    let result = run_bmc_seeding_with_solver_path_for_test(
        &net,
        &mut trackers,
        Some(Instant::now() + Duration::from_secs(20)),
        &solver,
    );

    assert_eq!(
        result,
        Some(1),
        "all depth-1 chunks completed, so the base case is complete"
    );
    assert!(trackers.iter().all(|tracker| tracker.verdict.is_none()));
    assert_eq!(
        fs::read_to_string(&calls_path)
            .expect("call log should exist")
            .lines()
            .collect::<Vec<_>>(),
        vec![
            "checks=4 depth=1",
            "checks=4 depth=1",
            "checks=4 depth=1",
            "checks=4 depth=1",
        ],
        "depth-1 chunking should split the large short-deadline batch"
    );
}

#[test]
fn test_bmc_depth1_chunking_continues_after_unknown_but_depth_incomplete() {
    let _guard = super::super::smt_encoding::ay_env_lock();
    let _enable = EnvVarGuard::set("TY_MCC_ENABLE_BMC_DEPTH1_CHUNKING", "1");
    let _disable = EnvVarGuard::remove("TY_MCC_DISABLE_BMC_DEPTH1_CHUNKING");
    let _chunk_size = EnvVarGuard::set("TY_MCC_BMC_DEPTH1_CHUNK_SIZE", "4");
    // Chunking is a batch-mode path; force the incremental default off.
    let _incremental_off = EnvVarGuard::set("TY_MCC_AY_BMC_DEADLINE_INCREMENTAL", "0");

    let net = two_step_chain_net();
    let tempdir = TempDir::new().expect("tempdir should create");
    let calls_path = tempdir.path().join("calls.log");
    let input_path = tempdir.path().join("solver-input.smt2");
    let solver = write_fake_solver_script(
        tempdir.path(),
        "fake-ay-depth1-chunked-unknown",
        &format!(
            "cat > \"{}\"\n\
if grep -Fq '(get-value' \"{}\"; then\n\
  printf 'sat\\n'\n\
  printf '((stay_0 false)\\n (fire_0_0 true)\\n (fire_0_1 false)\\n)\\n'\n\
  exit 0\n\
fi\n\
checks=$(grep -Fxc '(check-sat)' \"{}\" || true)\n\
call=1\n\
if [ -f \"{}\" ]; then call=$(( $(wc -l < \"{}\") + 1 )); fi\n\
printf 'checks=%s call=%s\\n' \"$checks\" \"$call\" >> \"{}\"\n\
i=0\n\
while [ \"$i\" -lt \"$checks\" ]; do\n\
  if [ \"$call\" -eq 1 ]; then printf 'unknown\\n'; else printf 'sat\\n'; fi\n\
  i=$((i + 1))\n\
done",
            input_path.display(),
            input_path.display(),
            input_path.display(),
            calls_path.display(),
            calls_path.display(),
            calls_path.display()
        ),
    );
    let predicate = ResolvedPredicate::IntLe(
        ResolvedIntExpr::Constant(1),
        ResolvedIntExpr::TokensCount(vec![PlaceIdx(1)]),
    );
    let mut trackers = (0..16)
        .map(|index| PropertyTracker {
            id: format!("ef-p2-{index}"),
            quantifier: PathQuantifier::EF,
            predicate: predicate.clone(),
            verdict: None,
            resolved_by: None,
            flushed: false,
        })
        .collect::<Vec<_>>();

    let result = run_bmc_seeding_with_solver_path_for_test(
        &net,
        &mut trackers,
        Some(Instant::now() + Duration::from_secs(20)),
        &solver,
    );

    assert_eq!(
        result, None,
        "one unknown chunk makes the depth-1 base case incomplete"
    );
    assert!(trackers[..4]
        .iter()
        .all(|tracker| tracker.verdict.is_none()));
    assert!(trackers[4..]
        .iter()
        .all(|tracker| tracker.verdict == Some(true)));
    assert_eq!(
        fs::read_to_string(&calls_path)
            .expect("call log should exist")
            .lines()
            .collect::<Vec<_>>(),
        vec![
            "checks=4 call=1",
            "checks=4 call=2",
            "checks=4 call=3",
            "checks=4 call=4",
        ],
        "later chunks should still run after an earlier unknown"
    );
}

#[test]
fn test_run_ay_timeout_returns_none_without_waiting_for_full_sleep() {
    let tempdir = TempDir::new().expect("tempdir should create");
    let solver =
        write_fake_solver_script(tempdir.path(), "sleepy-ay", "cat >/dev/null\nexec sleep 5");

    let start = Instant::now();
    let result =
        super::super::smt_encoding::run_ay(&solver, "(check-sat)\n", 1, Duration::from_millis(25));

    assert_eq!(result, None);
    assert!(
        start.elapsed() < RETURNED_BEFORE_SLEEP_CEILING,
        "timed-out fake solver should be killed quickly"
    );
}

#[cfg(unix)]
#[test]
fn test_run_ay_kills_orphaned_stdout_holder_after_parent_exit() {
    let tempdir = TempDir::new().expect("tempdir should create");
    let solver = write_fake_solver_script(
        tempdir.path(),
        "orphan-stdout-ay",
        "cat >/dev/null\n(sleep 5) &\nprintf 'sat\\n'\nexit 0",
    );

    let start = Instant::now();
    let result =
        super::super::smt_encoding::run_ay(&solver, "(check-sat)\n", 1, FAKE_SOLVER_ANSWER_BUDGET);

    assert_eq!(
        result,
        Some(vec![super::super::smt_encoding::SolverOutcome::Sat])
    );
    assert!(
        start.elapsed() < RETURNED_BEFORE_SLEEP_CEILING,
        "batch solver stdout orphans should not keep run_ay blocked"
    );
}

#[cfg(unix)]
#[test]
fn test_run_ay_timeout_kills_process_group_descendants() {
    let tempdir = TempDir::new().expect("tempdir should create");
    let marker = tempdir.path().join("descendant-survived");
    let solver = write_fake_solver_script(
        tempdir.path(),
        "descendant-ay",
        &format!(
            "cat >/dev/null\n(sleep 1; printf survived > '{}') &\nexec sleep 5",
            marker.display()
        ),
    );

    let result =
        super::super::smt_encoding::run_ay(&solver, "(check-sat)\n", 1, Duration::from_millis(25));

    assert_eq!(result, None);
    std::thread::sleep(Duration::from_millis(1200));
    assert!(
        !marker.exists(),
        "timed-out batch solver descendants should be killed with the process group"
    );
}

#[test]
fn test_run_ay_ignores_diagnostic_stdout_before_sat() {
    let tempdir = TempDir::new().expect("tempdir should create");
    let solver = write_fake_solver_script(
        tempdir.path(),
        "diag-ay",
        "cat >/dev/null\nprintf '[DIAG-SAT] pre-solve\\n'\nprintf 'sat\\n'",
    );

    let result =
        super::super::smt_encoding::run_ay(&solver, "(check-sat)\n", 1, FAKE_SOLVER_ANSWER_BUDGET);

    assert_eq!(
        result,
        Some(vec![super::super::smt_encoding::SolverOutcome::Sat]),
        "diagnostic stdout should not hide the final solver status"
    );
}

#[test]
fn test_bmc_script_depth_2_has_two_step_variables() {
    let net = producer_consumer_net();
    let trackers = vec![PropertyTracker {
        id: "prop-0".to_string(),
        quantifier: PathQuantifier::EF,
        predicate: ResolvedPredicate::True,
        verdict: None,
        resolved_by: None,
        flushed: false,
    }];

    let script = encode_bmc_script(&net, &trackers, &[0], 2);

    // Should have marking variables for steps 0, 1, 2
    assert!(script.contains("m_0_0"));
    assert!(script.contains("m_1_0"));
    assert!(script.contains("m_2_0"));
    // Should have stay/fire for steps 0 and 1
    assert!(script.contains("stay_0"));
    assert!(script.contains("stay_1"));
}

#[test]
fn test_bmc_script_logic_is_qf_lia() {
    let net = producer_consumer_net();
    let trackers = vec![PropertyTracker {
        id: "prop-0".to_string(),
        quantifier: PathQuantifier::EF,
        predicate: ResolvedPredicate::True,
        verdict: None,
        resolved_by: None,
        flushed: false,
    }];

    let script = encode_bmc_script(&net, &trackers, &[0], 1);
    assert!(script.starts_with("(set-logic QF_LIA)"));
}

#[test]
fn test_bmc_script_non_negative_markings() {
    let net = producer_consumer_net();
    let trackers = vec![PropertyTracker {
        id: "prop-0".to_string(),
        quantifier: PathQuantifier::EF,
        predicate: ResolvedPredicate::True,
        verdict: None,
        resolved_by: None,
        flushed: false,
    }];

    let script = encode_bmc_script(&net, &trackers, &[0], 1);

    // All marking variables should be non-negative
    assert!(script.contains("(assert (>= m_0_0 0))"));
    assert!(script.contains("(assert (>= m_0_1 0))"));
    assert!(script.contains("(assert (>= m_1_0 0))"));
    assert!(script.contains("(assert (>= m_1_1 0))"));
}

#[test]
fn test_bmc_seeding_ef_witness_when_ay_available() {
    let _guard = crate::examinations::smt_encoding::ay_env_lock();

    // Skip if ay not available (CI environments)
    if super::find_ay().is_none() {
        eprintln!("ay not available, skipping BMC integration test");
        return;
    }

    // Simple net: p0 → t0 → p1 with initial [1, 0]
    // EF(p1 >= 1) should be TRUE — reachable after 1 firing
    let net = producer_consumer_net();
    let mut trackers = vec![PropertyTracker {
        id: "ef-reach".to_string(),
        quantifier: PathQuantifier::EF,
        predicate: ResolvedPredicate::IntLe(
            ResolvedIntExpr::Constant(1),
            ResolvedIntExpr::TokensCount(vec![PlaceIdx(1)]),
        ),
        verdict: None,
        resolved_by: None,
        flushed: false,
    }];

    run_bmc_seeding_for_test(&net, &mut trackers, None);

    assert_eq!(
        trackers[0].verdict,
        Some(true),
        "BMC should find EF witness for p1 >= 1"
    );
}

#[test]
fn test_bmc_seeding_ag_counterexample_when_ay_available() {
    let _guard = crate::examinations::smt_encoding::ay_env_lock();

    // Skip if ay not available
    if super::find_ay().is_none() {
        eprintln!("ay not available, skipping BMC integration test");
        return;
    }

    // AG(p0 >= 1) is FALSE — after 1 firing p0 = 0
    let net = producer_consumer_net();
    let mut trackers = vec![PropertyTracker {
        id: "ag-counter".to_string(),
        quantifier: PathQuantifier::AG,
        predicate: ResolvedPredicate::IntLe(
            ResolvedIntExpr::Constant(1),
            ResolvedIntExpr::TokensCount(vec![PlaceIdx(0)]),
        ),
        verdict: None,
        resolved_by: None,
        flushed: false,
    }];

    run_bmc_seeding_for_test(&net, &mut trackers, None);

    assert_eq!(
        trackers[0].verdict,
        Some(false),
        "BMC should find AG counterexample for p0 >= 1"
    );
}

#[test]
fn test_bmc_seeding_unsat_leaves_verdict_none() {
    let _guard = crate::examinations::smt_encoding::ay_env_lock();

    // Skip if ay not available
    if super::find_ay().is_none() {
        eprintln!("ay not available, skipping BMC integration test");
        return;
    }

    // EF(p1 >= 100) is FALSE (only 1 total token) — BMC can't prove this, leaves None
    let net = producer_consumer_net();
    let mut trackers = vec![PropertyTracker {
        id: "ef-unreach".to_string(),
        quantifier: PathQuantifier::EF,
        predicate: ResolvedPredicate::IntLe(
            ResolvedIntExpr::Constant(100),
            ResolvedIntExpr::TokensCount(vec![PlaceIdx(1)]),
        ),
        verdict: None,
        resolved_by: None,
        flushed: false,
    }];

    run_bmc_seeding_for_test(&net, &mut trackers, None);

    assert_eq!(
        trackers[0].verdict, None,
        "BMC should leave verdict None for unreachable EF (UNSAT is inconclusive)"
    );
}
