// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

mod common;

fn run_parallel_check(report_tier: bool) -> (i32, String, String) {
    let dir = common::TempDir::new("parallel-engine-provenance");
    let (spec, cfg) = common::write_spec_and_config(
        &dir,
        "ParallelEngineProvenance",
        r#"---- MODULE ParallelEngineProvenance ----
VARIABLE x
Init == x = 0
Next == x < 2 /\ x' = x + 1
TypeOK == x \in 0..2
====
"#,
        "INIT Init\nNEXT Next\nINVARIANT TypeOK\nCHECK_DEADLOCK FALSE\n",
    );
    let args = [
        "check",
        spec.to_str().expect("UTF-8 spec path"),
        "--config",
        cfg.to_str().expect("UTF-8 config path"),
        "--workers",
        "2",
        "--bfs-only",
        "--no-gpu",
        "--output",
        "json",
    ];
    let env = report_tier.then_some(("TY_ENGINE_TIER", "1"));
    common::run_tla_parsed_with_env(
        &args,
        env.as_slice(),
        if report_tier {
            &[]
        } else {
            &["TY_ENGINE_TIER"]
        },
    )
}

fn run_adaptive_parallel_check() -> (i32, String, String) {
    let dir = common::TempDir::new("adaptive-parallel-engine-provenance");
    let (spec, cfg) = common::write_spec_and_config(
        &dir,
        "AdaptiveParallelEngineProvenance",
        r#"---- MODULE AdaptiveParallelEngineProvenance ----
VARIABLE x
\* Exceed the exact tiny-spec pilot cap (5,000 states) so adaptive mode
\* exercises ParallelChecker instead of correctly proving this run tiny.
Init == x \in 0..5000
Next == UNCHANGED x
====
"#,
        "INIT Init\nNEXT Next\nCHECK_DEADLOCK FALSE\n",
    );
    common::run_tla_parsed_with_env(
        &[
            "check",
            spec.to_str().expect("UTF-8 spec path"),
            "--config",
            cfg.to_str().expect("UTF-8 config path"),
            "--workers",
            "0",
            "--bfs-only",
            "--no-gpu",
            "--output",
            "json",
        ],
        &[("TY_ENGINE_TIER", "1")],
        &["TY_TIR_EVAL", "TY_TIR_PARITY"],
    )
}

fn assert_parallel_provenance(stdout: &str) {
    let output: serde_json::Value =
        serde_json::from_str(stdout).expect("parallel check stdout must be JSON");
    assert_eq!(output["result"]["status"], "ok");
    assert_eq!(output["engine_provenance"]["tier"], "parallel BFS");
    assert_eq!(output["engine_provenance"]["workers"], 2);
}

#[cfg_attr(test, ntest::timeout(30000))]
#[test]
fn adaptive_workers_zero_parallel_route_records_and_emits_provenance() {
    let (code, stdout, stderr) = run_adaptive_parallel_check();
    assert_eq!(
        code, 0,
        "adaptive parallel check failed\nstdout:\n{stdout}\nstderr:\n{stderr}"
    );
    let output: serde_json::Value =
        serde_json::from_str(&stdout).expect("adaptive check stdout must be JSON");
    assert_eq!(output["result"]["status"], "ok");
    assert_eq!(output["engine_provenance"]["tier"], "parallel BFS");
    assert!(
        output["engine_provenance"]["workers"]
            .as_u64()
            .is_some_and(|workers| workers >= 1),
        "adaptive parallel provenance must name its effective worker count: {output}"
    );
    assert_eq!(
        stderr
            .lines()
            .rev()
            .find(|line| line.starts_with("[engine] execution tier:")),
        Some("[engine] execution tier: parallel BFS"),
        "adaptive route's final parseable tier line must name parallel BFS"
    );
}

#[cfg_attr(test, ntest::timeout(30000))]
#[test]
fn parallel_bfs_always_records_structured_provenance_but_gates_stderr() {
    let (quiet_code, quiet_stdout, quiet_stderr) = run_parallel_check(false);
    assert_eq!(
        quiet_code, 0,
        "quiet parallel check failed\nstdout:\n{quiet_stdout}\nstderr:\n{quiet_stderr}"
    );
    assert_parallel_provenance(&quiet_stdout);
    assert!(
        !quiet_stderr.contains("[engine] execution tier:"),
        "engine tier leaked without TY_ENGINE_TIER\nstderr:\n{quiet_stderr}"
    );

    let (reported_code, reported_stdout, reported_stderr) = run_parallel_check(true);
    assert_eq!(
        reported_code, 0,
        "reported parallel check failed\nstdout:\n{reported_stdout}\nstderr:\n{reported_stderr}"
    );
    assert_parallel_provenance(&reported_stdout);
    assert_eq!(
        reported_stderr
            .lines()
            .filter(|line| line.starts_with("[engine] execution tier:"))
            .collect::<Vec<_>>(),
        ["[engine] execution tier: parallel BFS"],
        "parallel BFS must emit exactly one parseable tier line"
    );
}
