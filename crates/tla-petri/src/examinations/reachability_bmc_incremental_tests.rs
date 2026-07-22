// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

use std::fs;
#[cfg(unix)]
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use tempfile::TempDir;

use crate::petri_net::{Arc, PetriNet, PlaceIdx, PlaceInfo, TransitionInfo};
use crate::property_xml::PathQuantifier;
use crate::resolved_predicate::{ResolvedIntExpr, ResolvedPredicate};

use super::super::reachability::PropertyTracker;
use super::super::reachability_witness::validation_targets_from_trackers;

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

fn producer_consumer_net() -> PetriNet {
    PetriNet {
        name: Some("test".to_string()),
        places: vec![place("p0"), place("p1")],
        transitions: vec![trans("t0", vec![arc(0, 1)], vec![arc(1, 1)])],
        initial_marking: vec![1, 0],
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

fn sample_trackers() -> Vec<PropertyTracker> {
    vec![
        PropertyTracker {
            id: "ef-reach".to_string(),
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
            id: "ag-counterexample".to_string(),
            quantifier: PathQuantifier::AG,
            predicate: ResolvedPredicate::IntLe(
                ResolvedIntExpr::TokensCount(vec![PlaceIdx(1)]),
                ResolvedIntExpr::Constant(0),
            ),
            verdict: None,
            resolved_by: None,
            flushed: false,
        },
    ]
}

fn tracker_summary(trackers: &[PropertyTracker]) -> Vec<(String, Option<bool>)> {
    trackers
        .iter()
        .map(|tracker| (tracker.id.clone(), tracker.verdict))
        .collect()
}

fn run_bmc_seeding_with_solver_path_for_test(
    net: &PetriNet,
    trackers: &mut [PropertyTracker],
    ay_path: &Path,
) -> Option<usize> {
    run_bmc_seeding_with_solver_path_and_deadline_for_test(net, trackers, None, ay_path)
}

fn run_bmc_seeding_with_solver_path_and_deadline_for_test(
    net: &PetriNet,
    trackers: &mut [PropertyTracker],
    deadline: Option<Instant>,
    ay_path: &Path,
) -> Option<usize> {
    run_bmc_seeding_with_solver_path_mode_for_test(net, trackers, deadline, ay_path, false)
}

/// RAII guard that sets an env var for the duration of a test and restores the
/// previous value (or unsets it) on drop. Used to exercise the deadline-mode
/// gate (`bmc_deadline_incremental_enabled`) through the real selection path.
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
        match &self.prev {
            Some(prev) => crate::env_guard::set_var(self.key, prev),
            None => crate::env_guard::remove_var(self.key),
        }
    }
}

/// Run BMC seeding through the real deadline-mode gate (reads
/// `TY_MCC_AY_BMC_DEADLINE_INCREMENTAL`), so tests can verify the default
/// selection and the explicit override.
fn run_bmc_seeding_with_solver_path_via_gate_for_test(
    net: &PetriNet,
    trackers: &mut [PropertyTracker],
    deadline: Option<Instant>,
    ay_path: &Path,
) -> Option<usize> {
    let targets = validation_targets_from_trackers(trackers);
    super::run_bmc_seeding_with_solver_path(net, trackers, &targets, deadline, ay_path)
}

fn run_bmc_seeding_with_solver_path_mode_for_test(
    net: &PetriNet,
    trackers: &mut [PropertyTracker],
    deadline: Option<Instant>,
    ay_path: &Path,
    use_deadline_incremental: bool,
) -> Option<usize> {
    let targets = validation_targets_from_trackers(trackers);
    super::run_bmc_seeding_with_solver_path_mode(
        net,
        trackers,
        &targets,
        deadline,
        ay_path,
        use_deadline_incremental,
    )
}

#[test]
fn test_bmc_seeding_falls_back_to_batch_when_incremental_probe_fails() {
    let net = producer_consumer_net();
    let tempdir = TempDir::new().expect("tempdir should create");
    let calls_path = tempdir.path().join("calls.log");
    let solver = write_fake_solver_script(
        tempdir.path(),
        "fake-ay-batch-only",
        &format!(
            "printf 'call\\n' >> \"{}\"\n\
input=\"{}\"\n\
cat > \"$input\"\n\
if grep -Fq '(get-value' \"$input\"; then\n\
  printf 'sat\\n'\n\
  printf '((stay_0 false)\\n (fire_0_0 true)\\n)\\n'\n\
else\n\
  printf 'sat\\n'\n\
fi",
            calls_path.display(),
            tempdir.path().join("batch-only-input.smt2").display()
        ),
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

    run_bmc_seeding_with_solver_path_for_test(&net, &mut trackers, &solver);

    assert_eq!(
        trackers[0].verdict,
        Some(true),
        "batch fallback should preserve the original witness semantics"
    );
    assert_eq!(
        fs::read_to_string(&calls_path)
            .expect("call log should exist")
            .lines()
            .count(),
        3,
        "incremental probe should fail first, then batch fallback and SAT model validation should run"
    );
}

#[test]
fn test_bmc_deadline_uses_batch_mode_when_incremental_disabled() {
    let _guard = crate::examinations::smt_encoding::ay_env_lock();
    // Default is now incremental; the explicit `=0` override forces the legacy
    // batch path. This pins the fallback escape hatch through the real gate.
    let _env = EnvVarGuard::set("TY_MCC_AY_BMC_DEADLINE_INCREMENTAL", "0");

    let net = producer_consumer_net();
    let tempdir = TempDir::new().expect("tempdir should create");
    let calls_path = tempdir.path().join("calls.log");
    let input_path = tempdir.path().join("batch-input.smt2");
    let solver = write_fake_solver_script(
        tempdir.path(),
        "fake-ay-deadline-default-batch",
        &format!(
            "cat > \"{}\"\n\
if grep -Fq '(set-option :timeout' \"{}\"; then\n\
  printf 'mode=incremental\\n' >> \"{}\"\n\
  exit 2\n\
fi\n\
checks=$(grep -Fxc '(check-sat)' \"{}\" || true)\n\
printf 'mode=batch checks=%s\\n' \"$checks\" >> \"{}\"\n\
i=0\n\
while [ \"$i\" -lt \"$checks\" ]; do\n\
  printf 'unsat\\n'\n\
  i=$((i + 1))\n\
done",
            input_path.display(),
            input_path.display(),
            calls_path.display(),
            input_path.display(),
            calls_path.display()
        ),
    );
    let mut trackers = sample_trackers();

    let depth = run_bmc_seeding_with_solver_path_via_gate_for_test(
        &net,
        &mut trackers,
        Some(Instant::now() + Duration::from_secs(20)),
        &solver,
    );

    assert_eq!(depth, Some(16));
    assert_eq!(
        fs::read_to_string(&calls_path)
            .expect("call log should exist")
            .lines()
            .collect::<Vec<_>>(),
        vec![
            "mode=batch checks=3",
            "mode=batch checks=3",
            "mode=batch checks=3",
            "mode=batch checks=3",
            "mode=batch checks=3",
        ],
        "deadline BMC should use the batch runner when the incremental env is explicitly disabled"
    );
}

#[test]
fn test_bmc_deadline_incremental_mode_uses_incremental_solver() {
    let _guard = crate::examinations::smt_encoding::ay_env_lock();

    let net = producer_consumer_net();
    let tempdir = TempDir::new().expect("tempdir should create");
    let calls_path = tempdir.path().join("calls.log");
    let stdin_log_path = tempdir.path().join("stdin.log");
    let solver = write_fake_solver_script(
        tempdir.path(),
        "fake-ay-deadline-incremental",
        &format!(
            "printf 'process\\n' >> \"{}\"\n\
probe_done=0\n\
while IFS= read -r line; do\n\
  printf '%s\\n' \"$line\" >> \"{}\"\n\
  case \"$line\" in\n\
    \"(check-sat)\")\n\
      if [ \"$probe_done\" -eq 0 ]; then\n\
        probe_done=1\n\
        printf 'sat\\n'\n\
      else\n\
        printf 'unsat\\n'\n\
      fi\n\
      ;;\n\
    \"(exit)\")\n\
      exit 0\n\
      ;;\n\
  esac\n\
done",
            calls_path.display(),
            stdin_log_path.display()
        ),
    );
    let mut trackers = sample_trackers();

    let depth = run_bmc_seeding_with_solver_path_mode_for_test(
        &net,
        &mut trackers,
        Some(Instant::now() + Duration::from_secs(20)),
        &solver,
        true,
    );

    assert_eq!(depth, Some(16));
    assert_eq!(
        fs::read_to_string(&calls_path)
            .expect("call log should exist")
            .lines()
            .count(),
        1,
        "deadline incremental mode should keep one ay process across the depth ladder"
    );

    let stdin_log = fs::read_to_string(&stdin_log_path).expect("stdin log should exist");
    assert_eq!(
        stdin_log.matches("(set-option :timeout").count(),
        super::DEPTH_LADDER.len() * trackers.len(),
        "incremental deadline mode should issue one timed check per property per depth"
    );
}

#[test]
fn test_bmc_deadline_uses_incremental_mode_by_default() {
    let _guard = crate::examinations::smt_encoding::ay_env_lock();
    // No env override set: the deadline path must default to incremental, keeping
    // a single ay process across the depth ladder.
    let _env = EnvVarGuard::remove("TY_MCC_AY_BMC_DEADLINE_INCREMENTAL");

    let net = producer_consumer_net();
    let tempdir = TempDir::new().expect("tempdir should create");
    let calls_path = tempdir.path().join("calls.log");
    let stdin_log_path = tempdir.path().join("stdin.log");
    let solver = write_fake_solver_script(
        tempdir.path(),
        "fake-ay-deadline-default-incremental",
        &format!(
            "printf 'process\\n' >> \"{}\"\n\
probe_done=0\n\
while IFS= read -r line; do\n\
  printf '%s\\n' \"$line\" >> \"{}\"\n\
  case \"$line\" in\n\
    \"(check-sat)\")\n\
      if [ \"$probe_done\" -eq 0 ]; then\n\
        probe_done=1\n\
        printf 'sat\\n'\n\
      else\n\
        printf 'unsat\\n'\n\
      fi\n\
      ;;\n\
    \"(exit)\")\n\
      exit 0\n\
      ;;\n\
  esac\n\
done",
            calls_path.display(),
            stdin_log_path.display()
        ),
    );
    let mut trackers = sample_trackers();

    let depth = run_bmc_seeding_with_solver_path_via_gate_for_test(
        &net,
        &mut trackers,
        Some(Instant::now() + Duration::from_secs(20)),
        &solver,
    );

    assert_eq!(depth, Some(16));
    assert_eq!(
        fs::read_to_string(&calls_path)
            .expect("call log should exist")
            .lines()
            .count(),
        1,
        "default deadline mode should keep one ay process across the depth ladder (incremental)"
    );

    let stdin_log = fs::read_to_string(&stdin_log_path).expect("stdin log should exist");
    assert_eq!(
        stdin_log.matches("(set-option :timeout").count(),
        super::DEPTH_LADDER.len() * trackers.len(),
        "default deadline mode should issue one timed check per property per depth"
    );
}

#[test]
fn test_incremental_bmc_matches_batch_verdicts_and_depth() {
    let _guard = crate::examinations::smt_encoding::ay_env_lock();

    let Some(ay_path) = super::find_ay() else {
        eprintln!("SKIP: ay not available for incremental-vs-batch parity test");
        return;
    };

    let net = producer_consumer_net();
    let tempdir = TempDir::new().expect("tempdir should create");
    let batch_calls_path = tempdir.path().join("batch-calls.log");
    let batch_solver = write_fake_solver_script(
        tempdir.path(),
        "ay-batch-proxy",
        &format!(
            "printf 'call\\n' >> \"{}\"\n\
stdin_copy=\"{}\"\n\
cat > \"$stdin_copy\"\n\
line_count=$(wc -l < \"$stdin_copy\")\n\
check_sat_count=$(grep -Fxc '(check-sat)' \"$stdin_copy\" || true)\n\
if [ \"$line_count\" -eq 3 ] && [ \"$check_sat_count\" -eq 1 ] && grep -Fqx '(push 1)' \"$stdin_copy\" && grep -Fqx '(pop 1)' \"$stdin_copy\"; then\n  exit 1\nfi\n\
exec \"{}\" -smt2 -in < \"$stdin_copy\"",
            batch_calls_path.display(),
            tempdir.path().join("batch-input.smt2").display(),
            ay_path.display()
        ),
    );

    let mut incremental_trackers = sample_trackers();
    let mut batch_trackers = sample_trackers();

    let incremental_depth =
        run_bmc_seeding_with_solver_path_for_test(&net, &mut incremental_trackers, &ay_path);
    let batch_depth =
        run_bmc_seeding_with_solver_path_for_test(&net, &mut batch_trackers, &batch_solver);

    assert_eq!(
        tracker_summary(&incremental_trackers),
        tracker_summary(&batch_trackers),
        "incremental BMC should preserve the same seeded verdicts as batch fallback"
    );
    assert_eq!(
        incremental_depth, batch_depth,
        "incremental BMC should stop deepening at the same depth as batch fallback"
    );
    assert_eq!(
        tracker_summary(&incremental_trackers),
        vec![
            ("ef-reach".to_string(), Some(true)),
            ("ef-unreachable".to_string(), None),
            ("ag-counterexample".to_string(), Some(false)),
        ],
        "the parity fixture should seed the EF witness and AG counterexample identically"
    );
    assert!(
        incremental_depth.is_some(),
        "both parity paths should complete at least one ladder depth"
    );
    let batch_call_count = fs::read_to_string(&batch_calls_path)
        .expect("batch proxy call log should exist")
        .lines()
        .count();
    assert!(
        batch_call_count >= 2,
        "batch parity path should include the incremental probe and at least one fallback depth"
    );
}
