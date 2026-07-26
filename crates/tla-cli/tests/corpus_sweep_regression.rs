// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Continuous corpus-sweep regression guard — the north-star "measured
//! continuously, in public against the full tlaplus/Examples corpus" clause,
//! turned into a test that RE-MEASURES every run instead of trusting a single
//! committed artifact.
//!
//! The committed baseline is `crates/tla-cli/tests/corpus-sweep-baseline.json`
//! (`ty.corpus-sweep/v1`, 181 cfgs). This test re-runs `ty corpus sweep
//! --format json` (shelling out to the built `ty` binary — the CLI has no
//! in-process sweep API; its library target is intentionally empty) and guards
//! the FRESH per-spec certify tiers against that baseline so a change that
//! silently un-certifies a spec fails CI.
//!
//! # What it guards (the capability-rank invariant)
//!
//! Each `outcome` string maps to a capability RANK:
//!
//! | outcome                                                     | rank |
//! |-------------------------------------------------------------|------|
//! | `kernel-certified (unbounded parametric)`                   | 4    |
//! | `kernel-certified (enumerator-free fixpoint)`               | 3    |
//! | `kernel-certified (enumerator-free closure; Init enumerated)` | 2  |
//! | `kernel-certified (enumerator-assisted fixpoint)`           | 1    |
//! | `smt-certified`                                             | 1    |
//! | `declined` / `error`                                        | 0    |
//! | `timeout` / `unpaired` / `not-attempted`                    | none (inconclusive) |
//!
//! For every spec present in BOTH baseline and fresh: a DROP in known rank with
//! baseline rank >= 1 is a REGRESSION (was certified, now weaker/declined) ->
//! FAIL. A RISE is an improvement (noted; refresh the baseline). Equal is fine.
//!
//! # Why timeout is TOLERATED, not failed
//!
//! Timing is machine- and load-dependent. Several genuinely enumerator-free
//! specs sit near the per-spec budget (Barrier, Peterson, VLC, CatOdd/EvenBoxes,
//! SimpleRegular ...) and this test build is a DEBUG `ty` (the kernel type
//! checker is much slower unoptimized than the release build that produced the
//! baseline). A spec that was certified in the baseline but `timeout`s this run
//! is therefore reported as a WARNING, never a regression. Symmetrically a
//! baseline `timeout` is treated as inconclusive: a fresh certification of it is
//! an improvement, a fresh decline is not judged.
//!
//! # Fast vs slow split
//!
//! * [`corpus_sweep_core_free_set_no_regression`] (per-PR):
//!   sweeps a small curated allowlist of known enumerator-free specs
//!   (HourClock, AsynchInterface, ABCorrectness, Lock, TCommit — all < 15 s even
//!   in this debug build) with a generous timeout and the SAME regression guard.
//!   Catches a capability regression on the core free set quickly.
//! * [`corpus_sweep_full_no_regression`] (nightly/on-demand): the whole 181-cfg
//!   sweep — the actual "continuous measurement" run. It is ~15 min in a debug
//!   build, so it requires explicit resource authorization:
//!   `TY_RUN_FULL_CORPUS_SWEEP=1 cargo test -p tla-cli \
//!        --features "ay clean-cic" --test corpus_sweep_regression \
//!        corpus_sweep_full_no_regression -- --nocapture`
//!   Override params with `CORPUS_SWEEP_TIMEOUT` (secs, default 45) and
//!   `CORPUS_SWEEP_JOBS` (default = available parallelism).
//!
//! Both skip gracefully (like `spec_regression`) when the corpus is absent.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::process::Command;

use serde::Deserialize;

// ---------------------------------------------------------------------------
// `ty.corpus-sweep/v1` JSON schema (the subset this guard reads)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Deserialize)]
struct SweepFile {
    #[serde(default)]
    build: BuildInfo,
    summary: Summary,
    rows: Vec<Row>,
}

#[derive(Debug, Clone, Default, Deserialize)]
struct BuildInfo {
    #[serde(default)]
    ay: bool,
    #[serde(default)]
    clean_cic: bool,
}

#[derive(Debug, Clone, Deserialize)]
struct Summary {
    total_cfgs: u64,
    kernel_certified_any: u64,
    kernel_unbounded_parametric: u64,
    kernel_enumerator_free: u64,
    kernel_enumerator_free_closure: u64,
    kernel_enumerator_assisted: u64,
    smt_certified: u64,
    declined: u64,
    not_attempted: u64,
    timeout: u64,
    error: u64,
    unpaired: u64,
    buckets_sum_to_total: bool,
}

impl Summary {
    /// Recompute the kernel-any total from the four kernel tiers (do NOT trust
    /// the serialized `kernel_certified_any`).
    fn recomputed_kernel_any(&self) -> u64 {
        self.kernel_unbounded_parametric
            + self.kernel_enumerator_free
            + self.kernel_enumerator_free_closure
            + self.kernel_enumerator_assisted
    }

    fn recomputed_bucket_sum(&self) -> u64 {
        self.recomputed_kernel_any()
            + self.smt_certified
            + self.declined
            + self.not_attempted
            + self.timeout
            + self.error
            + self.unpaired
    }
}

#[derive(Debug, Clone, Deserialize)]
struct Row {
    spec: String,
    outcome: String,
    #[serde(default)]
    elapsed_ms: u64,
}

// ---------------------------------------------------------------------------
// Capability rank + per-pair classification (pure — unit-tested below)
// ---------------------------------------------------------------------------

/// Capability rank of an outcome, or `None` for INCONCLUSIVE outcomes whose
/// value carries no capability signal (timing- or environment-dependent).
fn rank(outcome: &str) -> Option<u8> {
    match outcome {
        "kernel-certified (unbounded parametric)" => Some(4),
        "kernel-certified (enumerator-free fixpoint)" => Some(3),
        "kernel-certified (enumerator-free closure; Init enumerated)" => Some(2),
        "kernel-certified (enumerator-assisted fixpoint)" => Some(1),
        "smt-certified" => Some(1),
        "declined" | "error" => Some(0),
        // Inconclusive: timing (`timeout`) or infra/build (`unpaired`,
        // `not-attempted`) — never a capability verdict, so never a regression.
        "timeout" | "unpaired" | "not-attempted" => None,
        // Unknown label = schema drift; treat as inconclusive but surface it.
        _ => None,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PairClass {
    /// Rank dropped from a certified baseline — the guard FAILS on these.
    Regression,
    /// Rank rose (assisted->free, declined->certified, timeout->certified): the
    /// baseline is stale-in-a-good-way and should be refreshed. Never fails.
    Improvement,
    /// Fresh `timeout`: tolerated (timing-dependent). Warning, never a failure.
    TimeoutTolerated,
    /// Either side inconclusive in a way we cannot rule on (baseline timeout ->
    /// fresh decline, fresh unpaired, unknown label ...). Noted, never fails.
    Inconclusive,
    /// Same known rank. Fine.
    Unchanged,
}

/// Classify a (baseline, fresh) outcome pair. THIS is the guard core; the
/// synthetic-fixture unit tests pin its three load-bearing cases.
fn classify(base: &str, fresh: &str) -> PairClass {
    // A fresh timeout is ALWAYS tolerated: timing is machine/load/build-profile
    // dependent and this test build is an unoptimized `ty`.
    if fresh == "timeout" {
        return PairClass::TimeoutTolerated;
    }
    match (rank(base), rank(fresh)) {
        (Some(b), Some(f)) => {
            if b >= 1 && f < b {
                PairClass::Regression
            } else if f > b {
                PairClass::Improvement
            } else {
                PairClass::Unchanged
            }
        }
        // Baseline inconclusive (e.g. baseline timeout), fresh is a real verdict.
        (None, Some(f)) => {
            if f >= 1 {
                PairClass::Improvement
            } else {
                PairClass::Inconclusive
            }
        }
        // Fresh inconclusive (non-timeout: unpaired / not-attempted / unknown).
        (_, None) => PairClass::Inconclusive,
    }
}

// ---------------------------------------------------------------------------
// Guard aggregation
// ---------------------------------------------------------------------------

#[derive(Debug, Default)]
struct GuardReport {
    /// (spec, baseline outcome, fresh outcome) — a FAIL for each.
    regressions: Vec<(String, String, String)>,
    improvements: Vec<(String, String, String)>,
    timeout_warnings: Vec<(String, String)>,
    inconclusive: Vec<(String, String, String)>,
    unchanged: usize,
    /// Fresh specs absent from the baseline (new specs — refresh the baseline).
    new_specs: Vec<(String, String)>,
    /// Specs compared (present in both).
    compared: usize,
}

/// Run the capability guard: for every FRESH row whose spec exists in the
/// baseline, classify the pair. Fresh specs not in the baseline are recorded as
/// `new_specs` (never a failure).
fn run_guard(baseline: &BTreeMap<String, String>, fresh: &[Row]) -> GuardReport {
    let mut r = GuardReport::default();
    for row in fresh {
        let Some(base_outcome) = baseline.get(&row.spec) else {
            r.new_specs.push((row.spec.clone(), row.outcome.clone()));
            continue;
        };
        r.compared += 1;
        match classify(base_outcome, &row.outcome) {
            PairClass::Regression => {
                r.regressions
                    .push((row.spec.clone(), base_outcome.clone(), row.outcome.clone()))
            }
            PairClass::Improvement => {
                r.improvements
                    .push((row.spec.clone(), base_outcome.clone(), row.outcome.clone()))
            }
            PairClass::TimeoutTolerated => r
                .timeout_warnings
                .push((row.spec.clone(), base_outcome.clone())),
            PairClass::Inconclusive => {
                r.inconclusive
                    .push((row.spec.clone(), base_outcome.clone(), row.outcome.clone()))
            }
            PairClass::Unchanged => r.unchanged += 1,
        }
    }
    r
}

// ---------------------------------------------------------------------------
// Honesty invariant (mirrors the sweep's own non-negotiable check)
// ---------------------------------------------------------------------------

/// Assert the fresh sweep's totals add up — the project's honesty invariant.
fn assert_honesty_invariant(s: &Summary, context: &str) {
    assert!(
        s.buckets_sum_to_total,
        "{context}: fresh sweep reports buckets_sum_to_total=false — the sweep's own \
         honesty invariant is broken",
    );
    assert_eq!(
        s.recomputed_kernel_any(),
        s.kernel_certified_any,
        "{context}: kernel_certified_any ({}) != sum of the four kernel tiers ({})",
        s.kernel_certified_any,
        s.recomputed_kernel_any(),
    );
    assert_eq!(
        s.recomputed_bucket_sum(),
        s.total_cfgs,
        "{context}: buckets sum to {} but total_cfgs={} — totals do not add up",
        s.recomputed_bucket_sum(),
        s.total_cfgs,
    );
}

// ---------------------------------------------------------------------------
// Baseline / binary / corpus plumbing
// ---------------------------------------------------------------------------

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("crates/ parent")
        .parent()
        .expect("workspace root")
        .to_path_buf()
}

fn load_baseline() -> SweepFile {
    let path = workspace_root().join("crates/tla-cli/tests/corpus-sweep-baseline.json");
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("read baseline {}: {e}", path.display()));
    serde_json::from_str(&text).unwrap_or_else(|e| panic!("parse baseline {}: {e}", path.display()))
}

fn baseline_outcome_map(b: &SweepFile) -> BTreeMap<String, String> {
    b.rows
        .iter()
        .map(|r| (r.spec.clone(), r.outcome.clone()))
        .collect()
}

fn ty_binary() -> String {
    env!("CARGO_BIN_EXE_ty").to_string()
}

/// Resolve the corpus `specifications/` dir the way `ty corpus` does; return it
/// only if it exists. Absent -> `None` so the test can skip gracefully.
fn resolve_corpus_specs() -> Option<PathBuf> {
    // 1. $TLAPLUS_EXAMPLES (normalize: may point AT `specifications` or its parent).
    if let Ok(env) = std::env::var("TLAPLUS_EXAMPLES") {
        if !env.is_empty() {
            let p = PathBuf::from(&env);
            let specs = if p.file_name().and_then(|s| s.to_str()) == Some("specifications") {
                p
            } else {
                p.join("specifications")
            };
            if specs.is_dir() {
                return Some(specs);
            }
        }
    }
    // 2. $HOME/tlaplus-examples/specifications
    let home = std::env::var("HOME").unwrap_or_default();
    let specs = PathBuf::from(home).join("tlaplus-examples/specifications");
    specs.is_dir().then_some(specs)
}

/// Skip-guard shared by both variants. Prints a skip note and returns `None`
/// when the corpus is absent (never fails on absence — mirrors spec_regression).
fn corpus_or_skip(test: &str) -> Option<PathBuf> {
    match resolve_corpus_specs() {
        Some(p) => Some(p),
        None => {
            eprintln!(
                "skipping {test}: corpus not installed (set TLAPLUS_EXAMPLES or run \
                 `ty corpus fetch`) — north-star continuous measurement not exercised"
            );
            None
        }
    }
}

/// Shell out to `ty corpus sweep --format json` and parse the report. `filter`
/// restricts to specs whose corpus-relative path contains the substring.
fn run_sweep(filter: Option<&str>, timeout_secs: u64, jobs: usize) -> SweepFile {
    let out = std::env::temp_dir().join(format!(
        "ty-corpus-sweep-{}-{}.json",
        std::process::id(),
        filter
            .map(|f| f.replace(['/', '.'], "_"))
            .unwrap_or_else(|| "full".to_string()),
    ));
    let mut cmd = Command::new(ty_binary());
    cmd.arg("corpus")
        .arg("sweep")
        .arg("--format")
        .arg("json")
        .arg("--timeout")
        .arg(timeout_secs.to_string())
        .arg("--jobs")
        .arg(jobs.to_string())
        .arg("--out")
        .arg(&out);
    if let Some(f) = filter {
        cmd.arg("--filter").arg(f);
    }
    // The sweep prints a headline to stdout and per-spec progress to stderr; let
    // both through (so `--nocapture` shows progress).
    let status = cmd
        .status()
        .unwrap_or_else(|e| panic!("spawn `ty corpus sweep`: {e}"));
    assert!(
        status.success(),
        "`ty corpus sweep{}` exited with {status}",
        filter.map(|f| format!(" --filter {f}")).unwrap_or_default(),
    );
    let text = std::fs::read_to_string(&out)
        .unwrap_or_else(|e| panic!("read sweep output {}: {e}", out.display()));
    let _ = std::fs::remove_file(&out);
    serde_json::from_str(&text).unwrap_or_else(|e| {
        panic!(
            "parse sweep output: {e}\n--- json head ---\n{}",
            &text[..text.len().min(400)]
        )
    })
}

/// Refuse to "measure" with a build that has no certify lane — otherwise every
/// spec is `not-attempted` and the guard passes vacuously. Skip loudly instead.
fn require_certify_lanes(build: &BuildInfo, test: &str) -> bool {
    if build.ay && build.clean_cic {
        return true;
    }
    eprintln!(
        "skipping {test}: the swept `ty` binary was built WITHOUT ay+clean-cic \
         (ay={}, clean_cic={}) — no certify lane, cannot measure capability. \
         Rebuild: cargo test -p tla-cli --features \"ay clean-cic\" --test \
         corpus_sweep_regression",
        build.ay, build.clean_cic
    );
    false
}

// ---------------------------------------------------------------------------
// Reporting
// ---------------------------------------------------------------------------

/// Print the fresh summary next to the baseline summary — the test IS a
/// measurement, not just a guard (the "measured continuously" surface).
fn print_tally(baseline: &Summary, fresh: &Summary) {
    let row = |label: &str, b: u64, f: u64| {
        let delta = f as i64 - b as i64;
        let mark = if delta > 0 {
            format!("+{delta}")
        } else if delta < 0 {
            format!("{delta}")
        } else {
            "·".to_string()
        };
        eprintln!("  {label:<38} {b:>6}  {f:>6}   {mark}");
    };
    eprintln!();
    eprintln!("=== north-star tally: baseline vs THIS run ===");
    eprintln!(
        "  {:<38} {:>6}  {:>6}   {}",
        "metric", "base", "fresh", "delta"
    );
    row(
        "kernel-certified (any tier)",
        baseline.kernel_certified_any,
        fresh.kernel_certified_any,
    );
    row(
        "  unbounded parametric",
        baseline.kernel_unbounded_parametric,
        fresh.kernel_unbounded_parametric,
    );
    row(
        "  enumerator-free fixpoint",
        baseline.kernel_enumerator_free,
        fresh.kernel_enumerator_free,
    );
    row(
        "  enumerator-free closure",
        baseline.kernel_enumerator_free_closure,
        fresh.kernel_enumerator_free_closure,
    );
    row(
        "  enumerator-assisted fixpoint",
        baseline.kernel_enumerator_assisted,
        fresh.kernel_enumerator_assisted,
    );
    row("smt-certified", baseline.smt_certified, fresh.smt_certified);
    row("declined", baseline.declined, fresh.declined);
    row("not-attempted", baseline.not_attempted, fresh.not_attempted);
    row("timeout", baseline.timeout, fresh.timeout);
    row("error", baseline.error, fresh.error);
    row("unpaired", baseline.unpaired, fresh.unpaired);
    row("total .cfg", baseline.total_cfgs, fresh.total_cfgs);
    eprintln!();
}

/// Print the guard's per-spec findings and PANIC iff there were regressions.
fn report_and_assert(g: &GuardReport, context: &str) {
    eprintln!("=== {context}: capability guard ===");
    eprintln!(
        "  compared {}  unchanged {}  improvements {}  timeout-tolerated {}  \
         inconclusive {}  new-specs {}",
        g.compared,
        g.unchanged,
        g.improvements.len(),
        g.timeout_warnings.len(),
        g.inconclusive.len(),
        g.new_specs.len(),
    );
    for (spec, base) in &g.timeout_warnings {
        eprintln!("  WARN  {spec}: baseline {base:?} -> timeout this run (TOLERATED; timing)");
    }
    for (spec, base, fresh) in &g.improvements {
        eprintln!("  IMPROVED  {spec}: {base:?} -> {fresh:?} (refresh the baseline)");
    }
    for (spec, base, fresh) in &g.inconclusive {
        eprintln!("  note  {spec}: {base:?} -> {fresh:?} (inconclusive; not judged)");
    }
    for (spec, fresh) in &g.new_specs {
        eprintln!("  NEW  {spec}: {fresh:?} not in baseline (refresh the baseline)");
    }
    if !g.regressions.is_empty() {
        eprintln!();
        eprintln!("  === CAPABILITY REGRESSIONS ({}) ===", g.regressions.len());
        for (spec, base, fresh) in &g.regressions {
            eprintln!("  REGRESSION  {spec}: baseline {base:?} -> {fresh:?}");
        }
        panic!(
            "{context}: {} capability regression(s) vs crates/tla-cli/tests/corpus-sweep-baseline.json: {}",
            g.regressions.len(),
            g.regressions
                .iter()
                .map(|(s, b, f)| format!("{s} ({b} -> {f})"))
                .collect::<Vec<_>>()
                .join("; "),
        );
    }
}

// ---------------------------------------------------------------------------
// FAST variant — curated core enumerator-free set, runs per-PR
// ---------------------------------------------------------------------------

/// The core enumerator-free specs, by exact corpus-relative `.cfg` path (each
/// path is also a unique `--filter` substring). All were
/// `kernel-certified (enumerator-free fixpoint)` in the baseline and all
/// certify in < 15 s even in this debug build.
const CORE_FREE_SPECS: [&str; 5] = [
    "SpecifyingSystems/AsynchronousInterface/AsynchInterface.cfg",
    "SpecifyingSystems/HourClock/HourClock.cfg",
    "SpecifyingSystems/TLC/ABCorrectness.cfg",
    "locks_auxiliary_vars/Lock.cfg",
    "transaction_commit/TCommit.cfg",
];

/// Per-PR guard: sweep only the curated core free set and assert no capability
/// regression on it. Fast (~30 s), so every PR catches a
/// regression on the specs the north star most cares about.
#[test]
fn corpus_sweep_core_free_set_no_regression() {
    if corpus_or_skip("corpus_sweep_core_free_set_no_regression").is_none() {
        return;
    }
    let baseline = load_baseline();
    let base_map = baseline_outcome_map(&baseline);

    // Generous per-spec timeout: these are known-fast, but a loaded CI box on a
    // debug build must not spuriously time out and mask a real verdict.
    let timeout = 90;
    let mut fresh_rows: Vec<Row> = Vec::new();
    for filter in CORE_FREE_SPECS {
        let report = run_sweep(Some(filter), timeout, 1);
        if !require_certify_lanes(&report.build, "corpus_sweep_core_free_set_no_regression") {
            return;
        }
        assert_honesty_invariant(&report.summary, &format!("core/{filter}"));
        assert!(
            report.rows.iter().any(|r| r.spec == filter),
            "filter {filter:?} matched no row — corpus layout drifted; got: {:?}",
            report.rows.iter().map(|r| &r.spec).collect::<Vec<_>>(),
        );
        fresh_rows.extend(report.rows);
    }

    // The test must not pass vacuously.
    assert_eq!(
        fresh_rows
            .iter()
            .filter(|r| CORE_FREE_SPECS.contains(&r.spec.as_str()))
            .count(),
        CORE_FREE_SPECS.len(),
        "did not sweep every core free spec",
    );
    for r in &fresh_rows {
        if CORE_FREE_SPECS.contains(&r.spec.as_str()) {
            eprintln!("  core: {} -> {} ({} ms)", r.spec, r.outcome, r.elapsed_ms);
        }
    }

    let g = run_guard(&base_map, &fresh_rows);
    report_and_assert(&g, "core free set");
}

// ---------------------------------------------------------------------------
// SLOW variant — the whole corpus. It IS the continuous-measurement run
// (nightly / on-demand), ~15 min in a debug build, so the test explicitly
// qualifies the resource-intensive campaign through an environment variable.
// ---------------------------------------------------------------------------

/// The full continuous-measurement run. Re-sweeps all 181 cfgs, prints the
/// baseline-vs-fresh tally, and guards every spec against a capability
/// regression (tolerating timeouts). Run it as the nightly/on-demand
/// north-star measurement:
///
/// ```text
/// TY_RUN_FULL_CORPUS_SWEEP=1 cargo test -p tla-cli \
///     --features "ay clean-cic" --test corpus_sweep_regression \
///     corpus_sweep_full_no_regression -- --nocapture
/// ```
#[test]
fn corpus_sweep_full_no_regression() {
    if !std::env::var_os("TY_RUN_FULL_CORPUS_SWEEP").is_some_and(|value| value == "1") {
        eprintln!(
            "SKIP corpus_sweep_full_no_regression: set TY_RUN_FULL_CORPUS_SWEEP=1 \
             to authorize the ~15-minute, 181-config campaign"
        );
        return;
    }
    if corpus_or_skip("corpus_sweep_full_no_regression").is_none() {
        return;
    }
    let baseline = load_baseline();
    let base_map = baseline_outcome_map(&baseline);

    let timeout: u64 = std::env::var("CORPUS_SWEEP_TIMEOUT")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(45);
    let jobs: usize = std::env::var("CORPUS_SWEEP_JOBS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or_else(|| {
            std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(4)
        });

    eprintln!(
        "=== full corpus sweep: {} spec(s) expected, {timeout}s/spec, {jobs} job(s) ===",
        baseline.summary.total_cfgs
    );
    let fresh = run_sweep(None, timeout, jobs);
    if !require_certify_lanes(&fresh.build, "corpus_sweep_full_no_regression") {
        return;
    }

    assert_honesty_invariant(&fresh.summary, "full");
    print_tally(&baseline.summary, &fresh.summary);

    let g = run_guard(&base_map, &fresh.rows);
    report_and_assert(&g, "full corpus");
}

// ---------------------------------------------------------------------------
// Unit tests — the guard core on synthetic baseline-vs-fresh pairs. NO corpus,
// NO subprocess: these pin the load-bearing semantics and always run.
// ---------------------------------------------------------------------------

#[cfg(test)]
mod guard_unit_tests {
    use super::*;

    const FREE: &str = "kernel-certified (enumerator-free fixpoint)";
    const CLOSURE: &str = "kernel-certified (enumerator-free closure; Init enumerated)";
    const ASSISTED: &str = "kernel-certified (enumerator-assisted fixpoint)";
    const UNBOUNDED: &str = "kernel-certified (unbounded parametric)";

    #[test]
    fn rank_ordering_is_monotone_by_capability() {
        assert!(rank(UNBOUNDED) > rank(FREE));
        assert!(rank(FREE) > rank(CLOSURE));
        assert!(rank(CLOSURE) > rank(ASSISTED));
        assert!(rank(ASSISTED) > rank("declined"));
        assert_eq!(rank("smt-certified"), rank(ASSISTED)); // both rank 1
        assert_eq!(rank("declined"), rank("error"));
        assert_eq!(rank("timeout"), None);
        assert_eq!(rank("unpaired"), None);
        assert_eq!(rank("not-attempted"), None);
    }

    // The three load-bearing fixtures the task calls out.

    #[test]
    fn fixture_free_to_declined_is_a_regression() {
        assert_eq!(classify(FREE, "declined"), PairClass::Regression);
        // ...and the aggregate guard FAILS on it.
        let base: BTreeMap<String, String> = [("d/S.cfg".to_string(), FREE.to_string())]
            .into_iter()
            .collect();
        let fresh = vec![Row {
            spec: "d/S.cfg".into(),
            outcome: "declined".into(),
            elapsed_ms: 1,
        }];
        let g = run_guard(&base, &fresh);
        assert_eq!(
            g.regressions.len(),
            1,
            "free->declined must be a regression"
        );
        assert_eq!(g.regressions[0].0, "d/S.cfg");
    }

    #[test]
    fn fixture_free_to_timeout_is_tolerated() {
        assert_eq!(classify(FREE, "timeout"), PairClass::TimeoutTolerated);
        let base: BTreeMap<String, String> = [("d/S.cfg".to_string(), FREE.to_string())]
            .into_iter()
            .collect();
        let fresh = vec![Row {
            spec: "d/S.cfg".into(),
            outcome: "timeout".into(),
            elapsed_ms: 45000,
        }];
        let g = run_guard(&base, &fresh);
        assert!(
            g.regressions.is_empty(),
            "free->timeout must NOT be a regression"
        );
        assert_eq!(
            g.timeout_warnings.len(),
            1,
            "free->timeout is a tolerated warning"
        );
    }

    #[test]
    fn fixture_assisted_to_free_is_an_improvement() {
        assert_eq!(classify(ASSISTED, FREE), PairClass::Improvement);
        let base: BTreeMap<String, String> = [("d/S.cfg".to_string(), ASSISTED.to_string())]
            .into_iter()
            .collect();
        let fresh = vec![Row {
            spec: "d/S.cfg".into(),
            outcome: FREE.into(),
            elapsed_ms: 1,
        }];
        let g = run_guard(&base, &fresh);
        assert!(
            g.regressions.is_empty(),
            "assisted->free must NOT be a regression"
        );
        assert_eq!(g.improvements.len(), 1);
    }

    #[test]
    fn free_to_closure_is_a_capability_regression() {
        // Init re-entered the trust base: full-free -> closure-only is a drop.
        assert_eq!(classify(FREE, CLOSURE), PairClass::Regression);
    }

    #[test]
    fn declined_to_certified_is_an_improvement_not_a_regression() {
        assert_eq!(classify("declined", FREE), PairClass::Improvement);
        assert_eq!(classify("declined", "declined"), PairClass::Unchanged);
    }

    #[test]
    fn baseline_timeout_is_inconclusive_or_improvement() {
        // baseline timeout -> fresh certified: an improvement (baseline stale).
        assert_eq!(classify("timeout", FREE), PairClass::Improvement);
        // baseline timeout -> fresh declined: cannot judge, inconclusive.
        assert_eq!(classify("timeout", "declined"), PairClass::Inconclusive);
    }

    #[test]
    fn fresh_unpaired_is_inconclusive_not_regression() {
        // Corpus/pairing drift is not a certification regression.
        assert_eq!(classify(FREE, "unpaired"), PairClass::Inconclusive);
    }

    #[test]
    fn new_fresh_spec_absent_from_baseline_is_not_a_regression() {
        let base: BTreeMap<String, String> = BTreeMap::new();
        let fresh = vec![Row {
            spec: "brand/New.cfg".into(),
            outcome: "declined".into(),
            elapsed_ms: 1,
        }];
        let g = run_guard(&base, &fresh);
        assert!(g.regressions.is_empty());
        assert_eq!(g.new_specs.len(), 1);
        assert_eq!(g.compared, 0);
    }

    #[test]
    fn honesty_invariant_recomputes_from_fields() {
        let s = Summary {
            total_cfgs: 10,
            kernel_certified_any: 3,
            kernel_unbounded_parametric: 0,
            kernel_enumerator_free: 2,
            kernel_enumerator_free_closure: 0,
            kernel_enumerator_assisted: 1,
            smt_certified: 0,
            declined: 5,
            not_attempted: 0,
            timeout: 1,
            error: 0,
            unpaired: 1,
            buckets_sum_to_total: true,
        };
        assert_eq!(s.recomputed_kernel_any(), 3);
        assert_eq!(s.recomputed_bucket_sum(), 10);
        assert_honesty_invariant(&s, "unit");
    }

    #[test]
    #[should_panic(expected = "totals do not add up")]
    fn honesty_invariant_catches_bad_totals() {
        let s = Summary {
            total_cfgs: 10,
            kernel_certified_any: 3,
            kernel_unbounded_parametric: 0,
            kernel_enumerator_free: 2,
            kernel_enumerator_free_closure: 0,
            kernel_enumerator_assisted: 1,
            smt_certified: 0,
            declined: 4, // one short -> sum = 9 != 10
            not_attempted: 0,
            timeout: 1,
            error: 0,
            unpaired: 1,
            buckets_sum_to_total: true,
        };
        assert_honesty_invariant(&s, "unit");
    }

    /// The real committed baseline must parse and satisfy its own honesty
    /// invariant (this runs with no corpus — pure file read).
    #[test]
    fn committed_baseline_is_self_consistent() {
        let b = load_baseline();
        assert_honesty_invariant(&b.summary, "committed baseline");
        assert_eq!(
            b.rows.len() as u64,
            b.summary.total_cfgs,
            "row count == total_cfgs"
        );
    }
}
