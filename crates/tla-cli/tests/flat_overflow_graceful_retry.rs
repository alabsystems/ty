// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Graceful flat-storage value-overflow handling (e2e).
//!
//! A spec whose all-scalar state variable crosses i64 mid-run is admitted to
//! flat-primary storage (the first init state roundtrips fine), then produces
//! a value the fixed flat i64 layout cannot represent. The checker must NOT
//! panic/abort: it surfaces the typed `FlatLayoutUnsupportedValue` error, the
//! CLI prints a note and transparently re-runs once with flat state storage
//! disabled, and the final verdict is correct.

mod common;
use common::TempDir;
use std::time::Duration;

fn run_tla(args: &[&str], env: &[(&str, &str)]) -> (i32, String, String) {
    // Sanitize flat-related env so auto-detection is in its default state.
    common::run_tla_parsed_with_env_timeout(
        args,
        env,
        &["TY_NO_FLAT_BFS", "TY_NO_COMPILED_BFS"],
        Duration::from_secs(60),
    )
}

/// x: 2 -> 4 -> 16 -> 256 -> 65536 -> 2^32 -> 2^64 -> 2^128 -> (self-loop).
/// 2^64 > i64::MAX is the first state the flat layout cannot encode.
/// 8 distinct states; Inv (x # 0) holds throughout.
const OVERFLOW_SPEC: &[u8] = br#"
---- MODULE FlatOverflow ----
EXTENDS Integers
VARIABLE x
Init == x = 2
Next == x' = IF x < 100000000000000000000 THEN x * x ELSE x
Inv == x # 0
====
"#;

fn write_overflow_spec(dir: &TempDir) -> (String, String) {
    let spec = dir.path.join("FlatOverflow.tla");
    let cfg = dir.path.join("FlatOverflow.cfg");
    common::write_file(&spec, OVERFLOW_SPEC);
    common::write_file(&cfg, b"INIT Init\nNEXT Next\nINVARIANT Inv\n");
    (
        spec.to_str().unwrap().to_string(),
        cfg.to_str().unwrap().to_string(),
    )
}

#[cfg_attr(test, ntest::timeout(120_000))]
#[test]
fn scalar_crossing_i64_completes_gracefully_with_correct_verdict() {
    let dir = TempDir::new("ty-flat-overflow-retry");
    let (spec, cfg) = write_overflow_spec(&dir);

    // Exercise both the default (fused) route and the classic BFS route.
    for extra in [&[][..], &["--bfs-only"][..]] {
        let mut args = vec!["check", spec.as_str(), "--config", cfg.as_str()];
        args.extend_from_slice(extra);
        let (code, stdout, stderr) = run_tla(&args, &[]);

        // Correct verdict, no panic/abort ("invalid flat state serialization"
        // / "encoding collapsed" were the former fail-stop modes).
        assert_eq!(
            code, 0,
            "expected success (args {args:?})\nstderr:\n{stderr}\nstdout:\n{stdout}"
        );
        assert!(
            !stderr.contains("invalid flat state serialization")
                && !stderr.contains("encoding collapsed")
                && !stderr.contains("panicked"),
            "no flat serialization panic allowed\nstderr:\n{stderr}"
        );
        assert!(
            stdout.contains("No errors found") || stdout.contains("No error has been found"),
            "expected passing verdict (args {args:?})\nstdout:\n{stdout}"
        );
        // Exact state count: the squaring chain has 8 distinct states.
        assert!(
            stdout.contains("States found: 8"),
            "expected 8 distinct states (args {args:?})\nstdout:\n{stdout}"
        );
        // When a flat-primary path has no per-state fallback it surfaces the
        // typed error and the CLI transparently retries ONCE (note printed
        // once); when a sound fallback absorbs the state, no retry happens.
        // Either way the note appears at most once and results exactly once.
        let note = "re-running with flat state storage disabled";
        assert!(
            stderr.matches(note).count() <= 1,
            "retry happens at most once (args {args:?})\nstderr:\n{stderr}"
        );
        assert_eq!(
            stdout.matches("Model checking complete").count(),
            1,
            "results reported exactly once (args {args:?})\nstdout:\n{stdout}"
        );
    }
}

/// The --portfolio route must also handle the flat overflow gracefully: its
/// BFS lane can hit `FlatLayoutUnsupportedValue`, and the CLI re-runs the
/// portfolio once with flat state storage disabled instead of surfacing the
/// raw error ("Portfolio mode produced unexpected result").
#[cfg_attr(test, ntest::timeout(120_000))]
#[test]
fn portfolio_route_retries_flat_overflow_and_passes() {
    let dir = TempDir::new("ty-flat-overflow-portfolio");
    let (spec, cfg) = write_overflow_spec(&dir);

    let (code, stdout, stderr) = run_tla(
        &[
            "check",
            &spec,
            "--config",
            &cfg,
            "--portfolio",
            "--portfolio-strategies",
            "bfs",
        ],
        &[],
    );
    assert_eq!(
        code, 0,
        "portfolio route must complete with the correct verdict\nstderr:\n{stderr}\nstdout:\n{stdout}"
    );
    assert!(
        stdout.contains("No error has been found"),
        "expected passing portfolio verdict\nstdout:\n{stdout}"
    );
    // Exact state count: the squaring chain has 8 distinct states.
    assert!(
        stdout.contains("8 states found"),
        "expected 8 distinct states\nstdout:\n{stdout}"
    );
    // The raw flat-layout error must never surface as the verdict on this
    // route (that was the former failure mode: the Error result fell into the
    // portfolio result handler's catch-all bail).
    assert!(
        !stderr.contains("Portfolio mode produced unexpected result"),
        "raw flat-layout error must not surface\nstderr:\n{stderr}"
    );
    // Retry is single-shot: the note appears at most once (zero times when a
    // sound per-state fallback absorbs the value without the typed error).
    assert!(
        stderr
            .matches("re-running portfolio with flat state storage disabled")
            .count()
            <= 1,
        "portfolio retry happens at most once\nstderr:\n{stderr}"
    );
}

/// TLC tool-protocol stream marker (diagnostics): when the sequential path's
/// flat-overflow retry re-runs under `--output tlc-tool`, the second
/// init-phase lifecycle must be announced IN the @!@!@ stream (EC 1000
/// GENERAL note), not only on stderr — one stdout marker per stderr note.
#[cfg_attr(test, ntest::timeout(120_000))]
#[test]
fn tlc_tool_stream_marks_flat_overflow_retry() {
    let dir = TempDir::new("ty-flat-overflow-toolstream");
    let (spec, cfg) = write_overflow_spec(&dir);

    let (code, stdout, stderr) = run_tla(
        &[
            "check",
            &spec,
            "--config",
            &cfg,
            "--bfs-only",
            "--output",
            "tlc-tool",
        ],
        &[],
    );
    assert_eq!(
        code, 0,
        "expected success\nstderr:\n{stderr}\nstdout:\n{stdout}"
    );
    let note = "re-running with flat state storage disabled";
    let stderr_notes = stderr.matches(note).count();
    let stdout_notes = stdout.matches(note).count();
    assert!(
        stderr_notes <= 1,
        "retry happens at most once\nstderr:\n{stderr}"
    );
    assert_eq!(
        stdout_notes, stderr_notes,
        "each stderr retry note must have a matching tool-protocol marker in stdout\nstdout:\n{stdout}\nstderr:\n{stderr}"
    );
    if stdout_notes == 1 {
        // The marker must be a tagged GENERAL (EC 1000) tool message.
        let idx = stdout.find(note).unwrap();
        let preceding = &stdout[..idx];
        let tag = "@!@!@STARTMSG 1000:0 @!@!@";
        assert!(
            preceding.rfind(tag).is_some_and(|t| {
                // No other STARTMSG between the GENERAL tag and the note body.
                !preceding[t + tag.len()..].contains("@!@!@STARTMSG")
            }),
            "retry note must be wrapped in a GENERAL (1000) tool message\nstdout:\n{stdout}"
        );
    }
}

#[cfg_attr(test, ntest::timeout(120_000))]
#[test]
fn scalar_crossing_i64_with_flat_disabled_needs_no_retry() {
    let dir = TempDir::new("ty-flat-overflow-noflat");
    let (spec, cfg) = write_overflow_spec(&dir);

    // With flat storage force-disabled up front, the spec passes directly and
    // no retry note is printed.
    let (code, stdout, stderr) = common::run_tla_parsed_with_env_timeout(
        &["check", &spec, "--config", &cfg],
        &[("TY_NO_FLAT_BFS", "1")],
        &[],
        Duration::from_secs(60),
    );
    assert_eq!(
        code, 0,
        "expected direct success\nstderr:\n{stderr}\nstdout:\n{stdout}"
    );
    assert!(
        stdout.contains("States found: 8"),
        "expected 8 distinct states\nstdout:\n{stdout}"
    );
    assert!(
        !stderr.contains("re-running with flat state storage disabled"),
        "no retry note expected when flat is already disabled\nstderr:\n{stderr}"
    );
}

#[cfg_attr(test, ntest::timeout(120_000))]
#[test]
fn normal_scalar_spec_is_unaffected_no_retry_note() {
    let dir = TempDir::new("ty-flat-overflow-normal");
    let spec = dir.path.join("SmallCounter.tla");
    let cfg = dir.path.join("SmallCounter.cfg");
    common::write_file(
        &spec,
        br#"
---- MODULE SmallCounter ----
EXTENDS Integers
VARIABLE x
Init == x = 0
Next == x' = IF x < 5 THEN x + 1 ELSE 0
Inv == x <= 5
====
"#,
    );
    common::write_file(&cfg, b"INIT Init\nNEXT Next\nINVARIANT Inv\n");

    let (code, stdout, stderr) = run_tla(
        &[
            "check",
            spec.to_str().unwrap(),
            "--config",
            cfg.to_str().unwrap(),
        ],
        &[],
    );
    assert_eq!(
        code, 0,
        "expected success\nstderr:\n{stderr}\nstdout:\n{stdout}"
    );
    assert!(
        stdout.contains("States found: 6"),
        "expected 6 distinct states\nstdout:\n{stdout}"
    );
    assert!(
        !stderr.contains("re-running with flat state storage disabled"),
        "in-range spec must not trigger the flat retry\nstderr:\n{stderr}"
    );
}
