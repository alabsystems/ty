// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0
//
// mcc-keyword-guard: allow-spaced-mention
// (verdict-normalization helpers below recognize the spaced 2025-archive
// form for back-compat parity with the sweep harness.)

//! Fast verdict regression gate for `ty-mcc`.
//!
//! Loads MCC's `raw-result-analysis.csv` (consensus column), runs the
//! local `ty-mcc` binary on a subset of (model, examination) rows, and
//! emits a divergence diff plus per-row TSV. Designed as a CI fast-gate
//! parallel to the full `mcc_benchmarks` sweep — orders-of-magnitude
//! faster signal (CSV load + subset run vs hours per model).
//!
//! Exit code semantics:
//!   * 0 — every selected row has `wrong_units == 0`.
//!   * 1 — at least one row diverged from consensus (wrong unit).
//!   * 2 — harness error (CSV parse failure, binary not found, etc.).
//!
//! FIXME: factor CSV parsing / ExpectedUnit normalization / parse_mcc_output /
//! compare_output / run_with_timeout into a shared `mccctl_cmd/csv_shared`
//! module after MCC 2026 ships. For now these helpers are copied from
//! `mccctl_cmd/sweep.rs` (~150 LOC) per a 2026-05-23 verified-scope audit:
//! lifting them would balloon
//! beyond the 500-LOC refactor budget because several depend on the sweep
//! `Cli` struct and internal helper sets (`SINGLE_FORMULA_EXAMS`,
//! `STATE_SPACE_METRICS`, `normalize_state_space_metric`, …).

use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File};
use std::io::{BufRead, BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode, Stdio};
use std::time::{Duration, Instant};

use anyhow::{anyhow, bail, Context, Result};
use clap::Parser;
use serde::Serialize;
use serde_json::json;

use tla_petri::examination::Examination;
use tla_petri::mcc_keywords::{
    CANNOT_COMPUTE, FORMULA, MAX_TOKEN_IN_PLACE, MAX_TOKEN_PER_MARKING, STATES, STATE_SPACE,
    TECHNIQUES, TRANSITIONS,
};
use tla_petri::mcc_unit_compare::numeric_units_equal;

// ---------- CLI ----------

#[derive(Parser, Debug)]
#[command(
    name = "ty-mcc-csv-compare",
    about = "Fast verdict regression gate: run ty-mcc on a CSV-driven subset and diff vs consensus.",
    long_about = "Loads MCC's raw-result-analysis.csv (or .csv.zip), \
                  filters by --subset and --exams, invokes the target \
                  ty-mcc binary per (model, examination), parses stdout \
                  per the MCC manual (page 7), and writes per-row TSV. \
                  Exit 0 iff every row has wrong_units == 0 (CI gate)."
)]
struct Cli {
    /// MCC `raw-result-analysis.csv` (or `.csv.zip`).
    #[arg(long, value_name = "PATH")]
    csv_path: PathBuf,

    /// Comma-separated model-name whitelist (default: every Input in the CSV).
    #[arg(long, value_name = "NAMES")]
    subset: Option<String>,

    /// Comma-separated examinations (e.g. `StateSpace,ReachabilityDeadlock`).
    #[arg(
        long,
        value_name = "NAMES",
        default_value = "StateSpace,ReachabilityDeadlock"
    )]
    exams: String,

    /// Directory of model dirs (each containing `model.pnml` + optional property XMLs).
    #[arg(long, value_name = "DIR")]
    models_root: PathBuf,

    /// Candidate `ty-mcc` binary path.
    #[arg(long, value_name = "PATH")]
    binary: PathBuf,

    /// Worker threads passed to the candidate.
    #[arg(long, default_value_t = 1)]
    threads: u32,

    /// Outer harness timeout per case, in seconds (kills the child).
    #[arg(long, default_value_t = 60)]
    timeout: u64,

    /// Per-row TSV output path.
    #[arg(long, value_name = "PATH", default_value = "results.tsv")]
    results_tsv: PathBuf,

    /// Optional aggregate JSON summary output path.
    #[arg(long, value_name = "PATH")]
    summary_json: Option<PathBuf>,

    /// Hard-fail (exit 2) on any detected corpus-version mismatch instead of
    /// just categorizing affected rows as `harness_corpus_mismatch`. Use in
    /// CI to enforce that the `--models-root` corpus vintage agrees with the
    /// consensus CSV vintage. Default (off) preserves the existing exit-code
    /// semantics for backward compatibility with measurement runs that
    /// intentionally mix vintages.
    #[arg(long, default_value_t = false)]
    strict_corpus: bool,

    /// Compute and print the MCC points-ranking score (per the MCC rules
    /// III.4 / E-4.x formula) over the comparison rows, in addition to the
    /// existing unit-count diff. See `docs/mcc-2026/scoring-formula.md`.
    /// Off by default: the existing exit-code / TSV behavior is unchanged.
    #[arg(long, default_value_t = false)]
    score: bool,

    /// Field bar to compare TY's grand-total score against, in MCC points.
    /// Typically the score of the tool/field you want to beat (e.g. last
    /// year's gold-medal total for the selected categories). The score report
    /// prints `ty_total vs field_bar` and the margin. Only meaningful with
    /// `--score`.
    #[arg(long, value_name = "POINTS")]
    field_bar: Option<f64>,

    /// Comma-separated allowlist of model names to treat as "surprise" models
    /// (MCC score multiplier x10, per E-4.4). Every other model defaults to
    /// "known" (x1). Only meaningful with `--score`.
    #[arg(long, value_name = "NAMES")]
    surprise_models: Option<String>,

    /// Optional path to write the score report as JSON (only with `--score`).
    #[arg(long, value_name = "PATH")]
    score_json: Option<PathBuf>,
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    match run(cli) {
        Ok(code) => code,
        Err(err) => {
            eprintln!("{err:#}");
            ExitCode::from(2)
        }
    }
}

fn run(cli: Cli) -> Result<ExitCode> {
    let exams: Vec<Examination> = cli
        .exams
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|s| Examination::from_name(s).map_err(|e| anyhow!("unknown examination {s:?}: {e}")))
        .collect::<Result<_>>()?;
    if exams.is_empty() {
        bail!("--exams produced an empty examination list");
    }
    let strict_corpus = cli.strict_corpus;

    let subset: Option<BTreeSet<String>> = cli.subset.as_deref().map(|raw| {
        raw.split(',')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect()
    });

    let exam_filter: BTreeSet<String> = exams.iter().map(|e| e.as_str().to_string()).collect();

    let rows_raw = open_csv_rows(&cli.csv_path)?;
    // Corpus-vintage signal extracted once from the CSV's `run id` column.
    // `run id` looks like `r###-host-XXXXXXXXXX...` where the leading 10
    // digits of the tail are a Unix epoch (e.g. `1748537266` -> 2025).
    // We poll the first few non-empty rows and majority-vote; the cohort is
    // stable per-CSV. None signals the CSV had no parsable vintage hint
    // (toy CSVs in tests, CSVs from older MCC dumps without `run id`).
    let csv_vintage = extract_csv_vintage_year(&rows_raw);
    let mut selected: Vec<(String, Examination, Vec<ExpectedUnit>)> = Vec::new();
    for row in rows_raw {
        let row = clean_row(&row);
        let input = row.get("Input").map(String::as_str).unwrap_or("").trim();
        let exam = row
            .get("Examination")
            .map(String::as_str)
            .unwrap_or("")
            .trim();
        let estimate = row
            .get("estimated result")
            .map(String::as_str)
            .unwrap_or("")
            .trim();
        if input.is_empty() || exam.is_empty() {
            continue;
        }
        if !exam_filter.contains(exam) {
            continue;
        }
        if let Some(ref set) = subset {
            if !set.contains(input) {
                continue;
            }
        }
        let exam_enum = match Examination::from_name(exam) {
            Ok(e) => e,
            Err(_) => continue, // unknown examination column — skip silently.
        };
        let units = expected_units_from_raw(exam, estimate);
        selected.push((input.to_string(), exam_enum, units));
    }
    if selected.is_empty() {
        bail!("no CSV rows matched --subset and --exams filters");
    }

    // Dedup: `raw-result-analysis.csv` has one row per (tool, model, exam) —
    // e.g. a model evaluated by 7 tools yields 7 rows for each examination,
    // all sharing identical (Input, Examination) and `estimated result`. The
    // expensive `ty-mcc` invocation only depends on (Input, Examination), so
    // run each unique pair once and reuse its cached verdict when emitting the
    // per-original-row TSV. On the 2026-05-23 10-model 280-row measurement
    // this saved ~5 min (~7x reduction in candidate invocations); scales
    // linearly with corpus size + tool-row replication.
    //
    // We dedup by (Input, Examination) and use the *first* occurrence's
    // `ExpectedUnit` vector as the canonical case shape. Per-row TSV
    // granularity is preserved by emitting one row per original CSV row in
    // the loop below, each with a fresh `compare_output` against its own
    // expected units (the parsed output is reused from cache).
    let mut unique_cases: Vec<UniqueCase> = Vec::new();
    let mut seen_keys: BTreeSet<(String, String)> = BTreeSet::new();
    for (input, exam, units) in &selected {
        let key = (input.clone(), exam.as_str().to_string());
        if !seen_keys.insert(key) {
            continue;
        }
        unique_cases.push(UniqueCase {
            input: input.clone(),
            exam: *exam,
            units: units.clone(),
        });
    }

    let timeout = Duration::from_secs(cli.timeout);
    // Cache: (input, exam.as_str()) -> CachedRun. None signals missing model.
    let mut cache: BTreeMap<(String, String), Option<CachedRun>> = BTreeMap::new();
    for (idx, case) in unique_cases.iter().enumerate() {
        let key = (case.input.clone(), case.exam.as_str().to_string());
        let model_dir = cli.models_root.join(&case.input);
        if !model_dir.join("model.pnml").is_file() {
            eprintln!(
                "[{n}/{total}] {input} {exam}: missing_model_dir (cached, will fan out to original rows)",
                n = idx + 1,
                total = unique_cases.len(),
                input = case.input,
                exam = case.exam.as_str(),
            );
            cache.insert(key, None);
            continue;
        }
        let cmd = build_command(&cli.binary, &model_dir, case.exam, cli.threads, cli.timeout);
        let started = Instant::now();
        let (rc, timed_out, stdout, _stderr) = run_with_timeout(cmd, timeout)?;
        let elapsed_ms = started.elapsed().as_millis();
        let parsed = parse_mcc_output(case.exam, &stdout);
        eprintln!(
            "[{n}/{total}] {input} {exam}: ran (rc={rc} timeout={timed_out} elapsed_ms={elapsed_ms})",
            n = idx + 1,
            total = unique_cases.len(),
            input = case.input,
            exam = case.exam.as_str(),
        );
        cache.insert(
            key,
            Some(CachedRun {
                rc,
                timed_out,
                elapsed_ms,
                parsed,
            }),
        );
    }

    // Track which keys we've already emitted at least one row for, so the
    // per-row log can flag every subsequent emission as a cache hit (the
    // original row's `ty-mcc` invocation was elided).
    let mut emitted_keys: BTreeSet<(String, String)> = BTreeSet::new();
    // Per-(model, exam) corpus-vintage verdict cache. Computed lazily on the
    // first per-formula row we see for the key, and reused for every fan-out
    // row. The bool is "mismatch detected" — true means flag the row as
    // `harness_corpus_mismatch` instead of running the verdict comparator.
    let mut corpus_verdict: BTreeMap<(String, String), CorpusVerdict> = BTreeMap::new();
    let mut corpus_mismatch_seen = false;
    let mut results: Vec<RowResult> = Vec::with_capacity(selected.len());
    for (row_idx, (input, exam, units)) in selected.iter().enumerate() {
        let key = (input.clone(), exam.as_str().to_string());
        let cached_hit = !emitted_keys.insert(key.clone());
        let cached = cache.get(&key).expect("cache populated above");
        let Some(run) = cached else {
            results.push(RowResult::missing_model(input, *exam, units));
            continue;
        };
        let model_dir = cli.models_root.join(input);
        let ids = match expected_ids_for_exam(&model_dir, *exam, units.len()) {
            Ok(ids) => ids,
            Err(err) => {
                // No property XML / unreadable / wrong shape: surface as a
                // harness shape error with a descriptive note so the per-row
                // TSV still emits a row and the sweep proceeds.
                eprintln!(
                    "[{idx}/{total}] {input} {exam}: harness_expected_shape_error ({err:#})",
                    idx = row_idx + 1,
                    total = selected.len(),
                    exam = exam.as_str(),
                );
                results.push(RowResult {
                    input: input.clone(),
                    exam: exam.as_str().to_string(),
                    category: "harness_expected_shape_error".to_string(),
                    rc: run.rc,
                    timeout: run.timed_out,
                    format_ok: false,
                    elapsed_ms: run.elapsed_ms,
                    expected: units
                        .iter()
                        .map(|u| u.value.as_str())
                        .collect::<Vec<_>>()
                        .join(" "),
                    note: format!("expected-id lookup: {err:#}"),
                    ..RowResult::default()
                });
                continue;
            }
        };
        // Corpus-version mismatch guard (Option B+C from the 2026-05-25
        // incident remediation): for per-formula examinations, compare the
        // harness's XML-derived id vintage against the CSV's `run id`
        // vintage and against TY's emitted FORMULA id set. On mismatch,
        // categorize the row as `harness_corpus_mismatch` (NEW category,
        // distinct from `wrong`) so a stale-corpus run no longer inflates
        // the wrong-count. Each (model, exam) emits the scary warning
        // exactly once; subsequent fan-out rows inherit the cached verdict.
        let verdict = corpus_verdict.entry(key.clone()).or_insert_with(|| {
            validate_corpus_vintage(
                &model_dir,
                *exam,
                &ids,
                &run.parsed,
                csv_vintage.as_deref(),
                &cli.csv_path,
            )
        });
        if let CorpusVerdict::Mismatch { note } = verdict {
            if !corpus_mismatch_seen {
                corpus_mismatch_seen = true;
            }
            eprintln!(
                "[{idx}/{total}] {input} {exam}: harness_corpus_mismatch — {note} \
                 (cached={cached_hit}) (suppressing wrong-count inflation)",
                idx = row_idx + 1,
                total = selected.len(),
                exam = exam.as_str(),
            );
            results.push(RowResult {
                input: input.clone(),
                exam: exam.as_str().to_string(),
                category: "harness_corpus_mismatch".to_string(),
                rc: run.rc,
                timeout: run.timed_out,
                format_ok: run.parsed.format_ok,
                elapsed_ms: run.elapsed_ms,
                expected: units
                    .iter()
                    .map(|u| u.value.as_str())
                    .collect::<Vec<_>>()
                    .join(" "),
                actual: canonical_actual(
                    &ExpectedCase {
                        ids: ids.clone(),
                        units: units.clone(),
                        source: "raw-result-analysis.csv".to_string(),
                    },
                    &run.parsed,
                ),
                note: note.clone(),
                ..RowResult::default()
            });
            continue;
        }
        let expected = ExpectedCase {
            ids,
            units: units.clone(),
            source: "raw-result-analysis.csv".to_string(),
        };
        let result = compare_output(
            input,
            *exam,
            &expected,
            &run.parsed,
            run.rc,
            run.timed_out,
            run.elapsed_ms,
        );
        eprintln!(
            "[{idx}/{total}] {input} {exam}: {category} rc={rc} wrong={wrong} elapsed_ms={elapsed_ms} (cached={cached_hit})",
            idx = row_idx + 1,
            total = selected.len(),
            exam = exam.as_str(),
            category = result.category,
            rc = result.rc,
            wrong = result.wrong_units,
            elapsed_ms = result.elapsed_ms,
        );
        results.push(result);
    }

    write_results_tsv(&cli.results_tsv, &results)?;
    if let Some(path) = &cli.summary_json {
        let summary = aggregate(&results, unique_cases.len());
        fs::write(path, serde_json::to_string_pretty(&summary)? + "\n")
            .with_context(|| format!("write {}", path.display()))?;
    }

    if cli.score {
        let surprise: BTreeSet<String> = cli
            .surprise_models
            .as_deref()
            .map(|raw| {
                raw.split(',')
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty())
                    .collect()
            })
            .unwrap_or_default();
        let report = score_results(&results, &surprise);
        print_score_report(&report, cli.field_bar);
        if let Some(path) = &cli.score_json {
            fs::write(
                path,
                serde_json::to_string_pretty(&report.to_json(cli.field_bar))? + "\n",
            )
            .with_context(|| format!("write {}", path.display()))?;
        }
    }

    if corpus_mismatch_seen {
        let n_rows = results
            .iter()
            .filter(|r| r.category == "harness_corpus_mismatch")
            .count();
        eprintln!(
            "WARNING: detected corpus-version mismatch on {n_rows} row(s). \
             Re-point --models-root at the corpus vintage matching the consensus CSV \
             (csv_path={csv}). Affected rows are categorized as `harness_corpus_mismatch` \
             and are NOT counted as `wrong`.{strict_hint}",
            csv = cli.csv_path.display(),
            strict_hint = if strict_corpus {
                " --strict-corpus is set: exiting with code 2."
            } else {
                " Pass --strict-corpus to convert this warning into a hard fail (exit 2)."
            }
        );
        if strict_corpus {
            return Ok(ExitCode::from(2));
        }
    }
    let any_wrong = results.iter().any(|r| r.wrong_units > 0);
    Ok(if any_wrong {
        ExitCode::from(1)
    } else {
        ExitCode::SUCCESS
    })
}

/// One (Input, Examination) pair to evaluate. Multiple CSV rows may map to
/// the same `UniqueCase` (one per evaluating tool); we run `ty-mcc` once per
/// `UniqueCase` and fan the cached verdict back out per original row.
#[derive(Debug, Clone)]
struct UniqueCase {
    input: String,
    exam: Examination,
    // Retained for parity with the source CSV row grouping; the run loop keys
    // only on (input, exam) and reuses the cached verdict per original row.
    #[allow(dead_code)]
    units: Vec<ExpectedUnit>,
}

/// Result of running `ty-mcc` once on a `UniqueCase`. Cached and reused for
/// every original CSV row that shares the same (Input, Examination) key.
#[derive(Debug, Clone)]
struct CachedRun {
    rc: i32,
    timed_out: bool,
    elapsed_ms: u128,
    parsed: ParsedOutput,
}

// ---------- Domain types (copied from mccctl_cmd/sweep.rs — see FIXME above) ----------

const STATE_SPACE_METRICS: [&str; 4] = [
    STATES,
    TRANSITIONS,
    MAX_TOKEN_IN_PLACE,
    MAX_TOKEN_PER_MARKING,
];

#[derive(Debug, Clone)]
struct ExpectedUnit {
    value: String,
    known: bool,
    soft: bool,
}

#[derive(Debug, Clone)]
struct ExpectedCase {
    ids: Vec<String>,
    units: Vec<ExpectedUnit>,
    #[allow(dead_code)]
    source: String,
}

#[derive(Debug, Default, Clone)]
struct ParsedOutput {
    format_ok: bool,
    note: String,
    values: BTreeMap<String, String>,
}

#[derive(Debug, Default, Clone, Serialize)]
struct RowResult {
    input: String,
    exam: String,
    category: String,
    rc: i32,
    timeout: bool,
    format_ok: bool,
    elapsed_ms: u128,
    expected: String,
    actual: String,
    exact_units: u64,
    wrong_units: u64,
    cannot_compute_units: u64,
    missing_units: u64,
    note: String,
}

impl RowResult {
    fn missing_model(input: &str, exam: Examination, units: &[ExpectedUnit]) -> Self {
        let expected = units
            .iter()
            .map(|u| u.value.as_str())
            .collect::<Vec<_>>()
            .join(" ");
        Self {
            input: input.to_string(),
            exam: exam.as_str().to_string(),
            category: "missing_model_dir".to_string(),
            expected,
            note: "no model.pnml under --models-root".to_string(),
            ..Self::default()
        }
    }
}

// ---------- Expected-id construction (per-examination dispatch) ----------
//
// The MCC `raw-result-analysis.csv` `estimated result` column is a
// space-separated positional vector whose meaning depends on the
// examination shape:
//
// * `StateSpace`              -> 4 metrics in `STATE_SPACE_METRICS` order:
//                                STATES TRANSITIONS MAX_TOKEN_IN_PLACE
//                                MAX_TOKEN_PER_MARKING.
// * Single-formula booleans   -> one T/F/? token. The "id" is the
//                                examination name itself (the
//                                `ty-mcc` stdout emits
//                                `FORMULA <exam-name> <T|F|?> TECHNIQUES …`).
// * Per-formula examinations  -> one token per formula in the property
//                                XML, in the order returned by
//                                `sorted_property_ids` (alphabetic by id,
//                                which matches the zero-padded numeric
//                                suffix order in MCC's `<exam>.xml` files).
//                                Covers CTL/LTL/Reachability
//                                {Cardinality,Fireability} + UpperBounds —
//                                all 7 newer examinations introduced in
//                                MCC 2017+ that broke the original
//                                4-examination comparator.
//
// On error (missing or unparsable property XML) the caller emits a
// `harness_expected_shape_error` row with the propagated context so the
// failure is debuggable from the per-row TSV alone.
fn expected_ids_for_exam(
    model_dir: &Path,
    exam: Examination,
    unit_count: usize,
) -> Result<Vec<String>> {
    if exam == Examination::StateSpace {
        return Ok(STATE_SPACE_METRICS
            .iter()
            .take(unit_count)
            .map(|s| (*s).to_string())
            .collect());
    }
    if is_single_formula(exam) {
        return Ok(vec![exam.as_str().to_string()]);
    }
    // Per-formula examinations: ids come from the property XML.
    sorted_property_ids(model_dir, exam)
}

/// Port of `sweep::sorted_property_ids` (sweep.rs lines 1041-1077). Returns
/// the property ids declared in `<model_dir>/<exam>.xml` in sorted order.
/// The MCC consensus CSV's `estimated result` vector lists per-formula
/// verdicts in this same sorted order, so callers can pair `ids[i]` with
/// the i-th expected unit.
fn sorted_property_ids(model_dir: &Path, exam: Examination) -> Result<Vec<String>> {
    let xml_path = model_dir.join(format!("{}.xml", exam.as_str()));
    if !xml_path.exists() {
        bail!("missing property XML: {}", xml_path.display());
    }
    let content =
        fs::read_to_string(&xml_path).with_context(|| format!("read {}", xml_path.display()))?;
    let doc = roxmltree::Document::parse(&content)
        .map_err(|e| anyhow!("invalid property XML: {}: {e}", xml_path.display()))?;
    let mut ids: Vec<String> = Vec::new();
    for node in doc.descendants() {
        if node.tag_name().name() != "property" {
            continue;
        }
        let prop_ids: Vec<String> = node
            .children()
            .filter(|c| c.is_element() && c.tag_name().name() == "id")
            .map(|c| c.text().unwrap_or("").trim().to_string())
            .filter(|s| !s.is_empty())
            .collect();
        if prop_ids.len() != 1 {
            bail!("property without exactly one id in {}", xml_path.display());
        }
        ids.push(prop_ids[0].clone());
    }
    if ids.is_empty() {
        bail!("no property ids in {}", xml_path.display());
    }
    let mut seen: BTreeSet<&String> = BTreeSet::new();
    for id in &ids {
        if !seen.insert(id) {
            bail!("duplicate property ids in {}", xml_path.display());
        }
    }
    ids.sort();
    Ok(ids)
}

// ---------- Corpus-vintage mismatch guard ----------
//
// Defends against the 2026-05-25 incident where `ty-mcc-csv-compare` produced
// 1071 "wrong" rows that were actually a corpus hygiene issue: the harness
// read April-2024 property XMLs while the consensus CSV referenced May-2025
// XML formula IDs. Same formula NAMES, different formulas at each position
// in the per-formula vector. Result: 1071 silent false `wrong` rows that
// looked like a TY soundness regression but were stale-data noise.
//
// Three independent signals are combined; ANY of them firing categorizes the
// row as `harness_corpus_mismatch` (NEW category, distinct from `wrong`):
//
//   1. **XML id vintage tag vs CSV `run id` year** — CTL/Reachability XMLs
//      have explicit year tags (`*-CTLCardinality-2025-00`). The CSV's
//      `run id` column has the form `r###-host-XXXXXXXXXX...` where the
//      first 10 digits of the tail are a Unix epoch. Years must match.
//   2. **TY-emitted FORMULA id set vs XML-derived expected id set** —
//      Option C sentinel. Both should be identical (both come from the same
//      XML), so a divergence here means TY read a DIFFERENT XML than the
//      harness — e.g. PNG/PNML symlinks that bypass the per-exam XML.
//      Belt-and-suspenders.
//   3. **Per-row id-list shape** — already enforced by `compare_output`'s
//      `expected.ids.len() != expected.units.len()` gate; left in place
//      upstream for back-compat with existing test contracts.
//
// LTL/UpperBounds XMLs lack explicit year tags, so signal #1 misses them.
// Signal #2 still catches LTL/UpperBounds if TY's and the harness's XML
// reads disagree. A fingerprint manifest (Option A) is the long-term fix
// for vintage-tag-less exams; tracked in `docs/mcc-2026/`.

#[derive(Debug, Clone)]
enum CorpusVerdict {
    Ok,
    Mismatch { note: String },
}

fn validate_corpus_vintage(
    model_dir: &Path,
    exam: Examination,
    harness_ids: &[String],
    parsed: &ParsedOutput,
    csv_vintage: Option<&str>,
    csv_path: &Path,
) -> CorpusVerdict {
    // Skip non-per-formula examinations: StateSpace verdicts are positional
    // metric vectors that can't drift across vintages, and single-formula
    // examinations have one canonical id == examination name.
    if exam == Examination::StateSpace || is_single_formula(exam) {
        return CorpusVerdict::Ok;
    }

    // Signal #2 (Option C sentinel): TY's emitted FORMULA id set vs the
    // harness's expected (XML-derived) id set. These come from the same
    // XML and MUST agree; if they don't, the run is incoherent and any
    // verdict comparison would be meaningless. Skip when TY's output is
    // malformed — `compare_output` will categorize as `malformed_output`.
    if parsed.format_ok && !parsed.values.is_empty() {
        let harness_set: BTreeSet<&str> = harness_ids.iter().map(String::as_str).collect();
        let ty_set: BTreeSet<&str> = parsed.values.keys().map(String::as_str).collect();
        if harness_set != ty_set {
            let only_harness: Vec<&&str> = harness_set.difference(&ty_set).collect();
            let only_ty: Vec<&&str> = ty_set.difference(&harness_set).collect();
            eprintln!(
                "FATAL CORPUS MISMATCH for {}/{}:\n  harness expected {} formula id(s), TY emitted {}.\n  only in harness (XML at {}): {:?}\n  only in TY output: {:?}\n  CSV: {}\n  Re-point --models-root at the corpus vintage matching the consensus CSV.",
                model_dir.display(),
                exam.as_str(),
                harness_ids.len(),
                parsed.values.len(),
                model_dir.join(format!("{}.xml", exam.as_str())).display(),
                only_harness,
                only_ty,
                csv_path.display(),
            );
            return CorpusVerdict::Mismatch {
                note: format!(
                    "TY emitted FORMULA ids disagree with XML at {}; harness_only={} ty_only={}",
                    model_dir.join(format!("{}.xml", exam.as_str())).display(),
                    only_harness.len(),
                    only_ty.len(),
                ),
            };
        }
    }

    // Signal #1 (Option B): XML id vintage tag vs CSV `run id` year.
    if let (Some(csv_year), Some(xml_year)) = (csv_vintage, extract_xml_vintage_year(harness_ids)) {
        if csv_year != xml_year {
            eprintln!(
                "FATAL CORPUS MISMATCH for {}/{}:\n  XML at {} has vintage year {} (from id tag), but CSV {} has run-id year {}.\n  These vintages produce DIFFERENT formula sets at the same positional indices, so per-formula verdict comparison is GARBAGE.\n  Re-point --models-root at a {} corpus.",
                model_dir.display(),
                exam.as_str(),
                model_dir.join(format!("{}.xml", exam.as_str())).display(),
                xml_year,
                csv_path.display(),
                csv_year,
                csv_year,
            );
            return CorpusVerdict::Mismatch {
                note: format!(
                    "XML vintage {xml_year} != CSV vintage {csv_year}; re-point --models-root at a {csv_year} corpus"
                ),
            };
        }
    }

    CorpusVerdict::Ok
}

/// Extract the corpus EDITION year from a list of formula IDs: the MAXIMUM
/// year (4-digit string) embedded in any id matching the MCC year-tag pattern
/// `<...>-<YYYY>-<digits>` where YYYY is in `2000..=2099`. CTL/Reachability
/// XMLs use this pattern; LTL/UpperBounds typically don't, in which case this
/// returns `None` and signal #1 is silently skipped.
///
/// We take the MAX, not the first, because an MCC examination file carries
/// FORWARD formulas from prior editions: a 2025-edition `CTLCardinality.xml`
/// legitimately contains both `-2025-NN` and reused `-2023-NN` ids. The
/// edition is the newest year present — a 2025 file can hold 2023 formulas,
/// but a 2023 file can never hold 2025 ones. (Taking the first id's year was
/// a false-mismatch bug: `harness_ids` are sorted, so the carried-forward
/// `2023-12` block sorts ahead of `2025-00` and was misread as the edition.)
fn extract_xml_vintage_year(ids: &[String]) -> Option<String> {
    ids.iter().filter_map(|id| vintage_year_from_id(id)).max()
}

fn vintage_year_from_id(id: &str) -> Option<String> {
    // Match a 4-digit `20YY` segment between dashes, followed by another
    // dash and digits (the per-formula index). E.g. `Foo-CTLCardinality-2025-00`.
    let bytes = id.as_bytes();
    let mut i = 0;
    while i + 6 < bytes.len() {
        if bytes[i] == b'-'
            && bytes[i + 1] == b'2'
            && bytes[i + 2] == b'0'
            && bytes[i + 3].is_ascii_digit()
            && bytes[i + 4].is_ascii_digit()
            && bytes[i + 5] == b'-'
            && bytes[i + 6].is_ascii_digit()
        {
            return Some(format!(
                "20{}{}",
                bytes[i + 3] as char,
                bytes[i + 4] as char
            ));
        }
        i += 1;
    }
    None
}

/// Extract the corpus vintage year from the CSV's `run id` column. Polls up
/// to 32 rows and majority-votes; the cohort is stable per-CSV in practice
/// (one MCC competition cycle => one epoch range). Returns `None` if the
/// CSV has no `run id` column or no parseable epoch tail (toy CSVs in
/// tests, older MCC dumps without run-id).
fn extract_csv_vintage_year(rows: &[BTreeMap<String, String>]) -> Option<String> {
    let mut counts: BTreeMap<String, u32> = BTreeMap::new();
    for row in rows.iter().take(32) {
        let cleaned = clean_row(row);
        let run_id = cleaned.get("run id")?.trim().to_string();
        if let Some(year) = vintage_year_from_run_id(&run_id) {
            *counts.entry(year).or_default() += 1;
        }
    }
    counts.into_iter().max_by_key(|(_, n)| *n).map(|(y, _)| y)
}

fn vintage_year_from_run_id(run_id: &str) -> Option<String> {
    // Format: `r###-host-XXXXXXXXXX...` where the leading 10 digits of the
    // tail past the last `-` are a Unix epoch (in seconds, ~9-10 digit range
    // for the 2001-2286 era which covers any plausible MCC cycle).
    let tail = run_id.rsplit('-').next()?;
    let digits: String = tail.chars().take_while(char::is_ascii_digit).collect();
    if digits.len() < 10 {
        return None;
    }
    let epoch: u64 = digits[..10].parse().ok()?;
    // Convert Unix epoch -> calendar year. Avoid pulling chrono for one
    // function: use the standard 365.2425 days/year approximation seeded at
    // 1970-01-01 UTC. Accurate to the year within the entire 1970-2099
    // range that MCC could realistically span.
    let days = epoch / 86_400;
    // Use 1970-01-01 as day 0; integer year via Howard Hinnant's
    // civil_from_days algorithm (public-domain, see
    // https://howardhinnant.github.io/date_algorithms.html#civil_from_days).
    let z = days as i64 + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = (z - era * 146_097) as u64; // 0..146096
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365; // 0..399
    let y = (yoe as i64) + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // 0..365
    let mp = (5 * doy + 2) / 153; // 0..11
    let m = if mp < 10 { mp + 3 } else { mp - 9 }; // 1..12
    let year = if m <= 2 { y + 1 } else { y };
    Some(year.to_string())
}

// ---------- CSV loader (copied from sweep.rs lines 885-961) ----------

fn open_csv_rows(path: &Path) -> Result<Vec<BTreeMap<String, String>>> {
    let suffix = path
        .extension()
        .and_then(|s| s.to_str())
        .map(str::to_ascii_lowercase);
    if suffix.as_deref() == Some("zip") {
        let csv_bytes = extract_first_csv_from_zip(path)
            .with_context(|| format!("scan CSV member in {}", path.display()))?;
        return parse_csv_records(&csv_bytes[..]);
    }
    let file = File::open(path).with_context(|| format!("open {}", path.display()))?;
    parse_csv_records(file)
}

fn clean_row(row: &BTreeMap<String, String>) -> BTreeMap<String, String> {
    let mut out = BTreeMap::new();
    for (k, v) in row {
        let key = k.trim().trim_start_matches('#').trim().to_string();
        out.insert(key, v.clone());
    }
    out
}

fn parse_csv_records<R: Read>(reader: R) -> Result<Vec<BTreeMap<String, String>>> {
    let mut buf = BufReader::new(reader);
    let mut header_line = String::new();
    if buf.read_line(&mut header_line)? == 0 {
        return Ok(Vec::new());
    }
    let headers: Vec<String> = parse_csv_line(header_line.trim_end_matches(['\r', '\n']));
    let mut rows = Vec::new();
    for line in buf.lines() {
        let line = line?;
        let trimmed = line.trim_end_matches(['\r', '\n']);
        if trimmed.is_empty() {
            continue;
        }
        let fields = parse_csv_line(trimmed);
        let mut row = BTreeMap::new();
        for (i, value) in fields.into_iter().enumerate() {
            if let Some(key) = headers.get(i) {
                row.insert(key.clone(), value);
            }
        }
        rows.push(row);
    }
    Ok(rows)
}

fn parse_csv_line(line: &str) -> Vec<String> {
    let mut fields = Vec::new();
    let mut current = String::new();
    let mut in_quote = false;
    let mut chars = line.chars().peekable();
    while let Some(c) = chars.next() {
        if in_quote {
            if c == '"' {
                if matches!(chars.peek(), Some('"')) {
                    current.push('"');
                    chars.next();
                } else {
                    in_quote = false;
                }
            } else {
                current.push(c);
            }
        } else if c == '"' {
            in_quote = true;
        } else if c == ',' {
            fields.push(std::mem::take(&mut current));
        } else {
            current.push(c);
        }
    }
    fields.push(current);
    fields
}

fn extract_first_csv_from_zip(path: &Path) -> Result<Vec<u8>> {
    let list = Command::new("unzip")
        .arg("-Z1")
        .arg(path)
        .output()
        .with_context(|| format!("invoke unzip -Z1 {}", path.display()))?;
    if !list.status.success() {
        bail!(
            "unzip -Z1 failed for {}: {}",
            path.display(),
            String::from_utf8_lossy(&list.stderr)
        );
    }
    let listing = String::from_utf8_lossy(&list.stdout);
    let csv_member = listing
        .lines()
        .map(str::trim)
        .find(|line| line.to_ascii_lowercase().ends_with(".csv"))
        .ok_or_else(|| anyhow!("no CSV member in {}", path.display()))?;
    let extracted = Command::new("unzip")
        .arg("-p")
        .arg(path)
        .arg(csv_member)
        .output()
        .with_context(|| format!("invoke unzip -p {} {}", path.display(), csv_member))?;
    if !extracted.status.success() {
        bail!(
            "unzip -p failed for {} member {}: {}",
            path.display(),
            csv_member,
            String::from_utf8_lossy(&extracted.stderr)
        );
    }
    Ok(extracted.stdout)
}

// ---------- ExpectedUnit normalization (copied from sweep.rs lines 1159-1262) ----------

fn normalize_unit_from_str(raw: &str) -> ExpectedUnit {
    let raw = raw.trim();
    let (token_text, soft) = strip_soft_marker(raw);
    let upper = token_text.trim().to_ascii_uppercase().replace('_', " ");
    let value = if matches!(upper.as_str(), "TRUE" | "T") {
        "T".to_string()
    } else if matches!(upper.as_str(), "FALSE" | "F") {
        "F".to_string()
    // mcc-keyword-guard: allow-spaced-mention
    } else if matches!(
        upper.as_str(),
        "CANNOT COMPUTE" | "CANNOTCOMPUTE" | "DNC" | "D"
    ) {
        "D".to_string()
    } else if upper == "?" || upper.is_empty() {
        "?".to_string()
    } else {
        normalize_number(token_text)
    };
    let lower = value.to_ascii_lowercase();
    let known =
        !value.is_empty() && value != "?" && !matches!(lower.as_str(), "inf" | "+inf" | "-inf");
    ExpectedUnit { value, known, soft }
}

fn strip_soft_marker(token: &str) -> (&str, bool) {
    let t = token.trim();
    if t.len() >= 2 && t.starts_with('(') && t.ends_with(')') {
        (t[1..t.len() - 1].trim(), true)
    } else {
        (t, false)
    }
}

fn normalize_number(token: &str) -> String {
    let t = token.trim();
    if t.is_empty() {
        return String::new();
    }
    if let Ok(n) = t.parse::<i128>() {
        return n.to_string();
    }
    if let Ok(f) = t.parse::<f64>() {
        if !f.is_finite() {
            return t.to_string();
        }
        // Exact equality is intentional: `f == f.trunc()` is the canonical test
        // for "this float has no fractional part" so we can render it as an int.
        #[allow(clippy::float_cmp)]
        let is_whole = f == f.trunc();
        if is_whole && f.abs() < 1e18 {
            return format!("{}", f as i64);
        }
        let formatted = format!("{f:.18}");
        let mut s = formatted;
        while s.contains('.') && s.ends_with('0') {
            s.pop();
        }
        if s.ends_with('.') {
            s.pop();
        }
        return s;
    }
    t.to_string()
}

fn expected_units_from_raw(exam: &str, raw: &str) -> Vec<ExpectedUnit> {
    let raw = raw.trim();
    if raw.is_empty() {
        return Vec::new();
    }
    if exam == "StateSpace" || exam == "UpperBounds" {
        return raw
            .split_whitespace()
            .map(normalize_unit_from_str)
            .collect();
    }
    parse_bool_vector(raw)
        .into_iter()
        .map(|t| normalize_unit_from_str(&t))
        .collect()
}

fn parse_bool_vector(raw: &str) -> Vec<String> {
    let compact: String = raw.split_whitespace().collect();
    let bytes = compact.as_bytes();
    let mut tokens: Vec<String> = Vec::new();
    let mut i = 0usize;
    while i < bytes.len() {
        if bytes[i] == b'(' {
            if let Some(end_rel) = compact[i + 1..].find(')') {
                let end = i + 1 + end_rel;
                tokens.push(compact[i..=end].to_string());
                i = end + 1;
            } else {
                tokens.push(compact[i..].to_string());
                break;
            }
        } else {
            tokens.push(compact[i..=i].to_string());
            i += 1;
        }
    }
    tokens
}

// ---------- MCC output parsing (copied from sweep.rs lines 1266-1430) ----------

const SINGLE_FORMULA_EXAMS: [Examination; 5] = [
    Examination::ReachabilityDeadlock,
    Examination::OneSafe,
    Examination::QuasiLiveness,
    Examination::StableMarking,
    Examination::Liveness,
];

fn is_single_formula(exam: Examination) -> bool {
    SINGLE_FORMULA_EXAMS.contains(&exam)
}

fn normalize_state_space_metric(metric: &str) -> String {
    match metric.trim() {
        // mcc-keyword-guard: allow-spaced-mention
        "MAX TOKEN IN PLACE" => MAX_TOKEN_IN_PLACE.to_string(),
        // mcc-keyword-guard: allow-spaced-mention
        "MAX TOKEN PER MARKING" => MAX_TOKEN_PER_MARKING.to_string(),
        other => other.to_string(),
    }
}

fn parse_mcc_output(exam: Examination, stdout: &str) -> ParsedOutput {
    let lines: Vec<&str> = stdout
        .lines()
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .collect();
    if lines.is_empty() {
        return ParsedOutput {
            format_ok: false,
            note: "empty stdout".to_string(),
            values: BTreeMap::new(),
        };
    }
    if exam == Examination::StateSpace {
        return parse_state_space_lines(&lines);
    }
    parse_formula_lines(exam, &lines)
}

fn parse_state_space_lines(lines: &[&str]) -> ParsedOutput {
    if lines.len() == 1 {
        let l = lines[0].trim();
        if l == CANNOT_COMPUTE {
            let mut values = BTreeMap::new();
            values.insert(STATE_SPACE.to_string(), "D".to_string());
            return ParsedOutput {
                format_ok: true,
                note: "cannot-compute".to_string(),
                values,
            };
        }
        let canonical_prefix = format!("{STATE_SPACE} {CANNOT_COMPUTE} {TECHNIQUES} ");
        // mcc-keyword-guard: allow-spaced-mention
        let legacy_prefix = "STATE SPACE CANNOT COMPUTE TECHNIQUES ";
        if l.starts_with(&canonical_prefix) || l.starts_with(legacy_prefix) {
            let mut values = BTreeMap::new();
            values.insert(STATE_SPACE.to_string(), "D".to_string());
            return ParsedOutput {
                format_ok: true,
                note: "cannot-compute".to_string(),
                values,
            };
        }
    }
    let mut values: BTreeMap<String, String> = BTreeMap::new();
    for line in lines {
        let parts: Vec<&str> = line.split_whitespace().collect();
        let Some(tech_index) = parts.iter().position(|p| *p == TECHNIQUES) else {
            return ParsedOutput {
                format_ok: false,
                note: format!("invalid StateSpace line: {line}"),
                values: BTreeMap::new(),
            };
        };
        if parts.first().copied() == Some(STATE_SPACE) {
            if tech_index < 3 {
                return ParsedOutput {
                    format_ok: false,
                    note: format!("invalid StateSpace metric: {line}"),
                    values: BTreeMap::new(),
                };
            }
            let metric = normalize_state_space_metric(parts[1]);
            let value_token = parts[tech_index - 1];
            values.insert(metric, normalize_unit_from_str(value_token).value);
        } else if parts.len() >= 2
            && parts[0] == "STATE"
            // mcc-keyword-guard: allow-spaced-mention
            && parts[1] == "SPACE"
        {
            if tech_index < 4 {
                return ParsedOutput {
                    format_ok: false,
                    note: format!("invalid StateSpace metric: {line}"),
                    values: BTreeMap::new(),
                };
            }
            let joined = parts[2..tech_index - 1].join(" ");
            let metric = normalize_state_space_metric(&joined);
            let value_token = parts[tech_index - 1];
            values.insert(metric, normalize_unit_from_str(value_token).value);
        } else {
            return ParsedOutput {
                format_ok: false,
                note: format!("invalid StateSpace line: {line}"),
                values: BTreeMap::new(),
            };
        }
    }
    ParsedOutput {
        format_ok: true,
        note: "ok".to_string(),
        values,
    }
}

fn parse_formula_lines(exam: Examination, lines: &[&str]) -> ParsedOutput {
    let mut values: BTreeMap<String, String> = BTreeMap::new();
    for line in lines {
        let Some((id, raw_value)) = parse_formula_line(line) else {
            return ParsedOutput {
                format_ok: false,
                note: format!("invalid FORMULA line: {line}"),
                values: BTreeMap::new(),
            };
        };
        if values.contains_key(&id) {
            return ParsedOutput {
                format_ok: false,
                note: format!("duplicate FORMULA id: {id}"),
                values: BTreeMap::new(),
            };
        }
        values.insert(id, normalize_unit_from_str(&raw_value).value);
    }
    if is_single_formula(exam) {
        let expected_id = exam.as_str();
        let only_expected = values.len() == 1 && values.contains_key(expected_id);
        if !only_expected {
            let mut ids: Vec<&str> = values.keys().map(String::as_str).collect();
            ids.sort();
            return ParsedOutput {
                format_ok: false,
                note: format!(
                    "unexpected FORMULA ids for {}: {}",
                    exam.as_str(),
                    ids.join(",")
                ),
                values,
            };
        }
    }
    ParsedOutput {
        format_ok: true,
        note: "ok".to_string(),
        values,
    }
}

fn parse_formula_line(line: &str) -> Option<(String, String)> {
    let tokens: Vec<&str> = line.split_whitespace().collect();
    if tokens.len() < 3 || tokens[0] != FORMULA {
        return None;
    }
    if tokens.len() == 3 && tokens[2] == CANNOT_COMPUTE {
        return Some((tokens[1].to_string(), CANNOT_COMPUTE.to_string()));
    }
    if tokens.len() < 5 {
        return None;
    }
    let tech_index = tokens.iter().position(|t| *t == TECHNIQUES)?;
    if tech_index < 3 {
        return None;
    }
    let id = tokens[1].to_string();
    let verdict = tokens[2..tech_index].join(" ");
    Some((id, verdict))
}

// ---------- Comparison (copied from sweep.rs lines 1434-1596, RowResult-flavored) ----------

fn canonical_expected(expected: &ExpectedCase) -> String {
    expected
        .units
        .iter()
        .map(|u| u.value.as_str())
        .collect::<Vec<_>>()
        .join(" ")
}

fn canonical_actual(expected: &ExpectedCase, parsed: &ParsedOutput) -> String {
    expected
        .ids
        .iter()
        .filter_map(|id| parsed.values.get(id))
        .filter(|v| !v.is_empty())
        .cloned()
        .collect::<Vec<_>>()
        .join(" ")
}

fn compare_output(
    input: &str,
    exam: Examination,
    expected: &ExpectedCase,
    parsed: &ParsedOutput,
    rc: i32,
    timed_out: bool,
    elapsed_ms: u128,
) -> RowResult {
    let mut result = RowResult {
        input: input.to_string(),
        exam: exam.as_str().to_string(),
        category: "exact".to_string(),
        rc,
        timeout: timed_out,
        format_ok: parsed.format_ok,
        elapsed_ms,
        expected: canonical_expected(expected),
        actual: canonical_actual(expected, parsed),
        note: parsed.note.clone(),
        ..RowResult::default()
    };

    if timed_out {
        result.category = "timeout".to_string();
    } else if rc != 0 {
        result.category = "nonzero_exit".to_string();
    } else if !parsed.format_ok {
        result.category = "malformed_output".to_string();
    }

    if expected.ids.len() != expected.units.len() {
        result.format_ok = false;
        result.category = "harness_expected_shape_error".to_string();
        result.note = format!(
            "expected ids/units mismatch: ids={} units={}",
            expected.ids.len(),
            expected.units.len()
        );
        return result;
    }

    for (unit_id, expected_unit) in expected.ids.iter().zip(expected.units.iter()) {
        if !expected_unit.known {
            continue;
        }
        let mut actual = parsed.values.get(unit_id).cloned();
        if actual.is_none()
            && exam == Examination::StateSpace
            && parsed.values.get(STATE_SPACE).map(String::as_str) == Some("D")
        {
            actual = Some("D".to_string());
        }
        let actual = actual.unwrap_or_default();
        if actual.is_empty() {
            if !expected_unit.soft {
                result.missing_units += 1;
            }
            continue;
        }
        // Numeric-equivalence first: the MCC consensus CSV truncates
        // large state counts into 5-sig-digit scientific notation
        // (e.g. `1.3391E+6` for TY's `1339104`), so raw string equality
        // would flag these as `wrong` even though the values agree
        // within the truncation precision. Falls back to string
        // equality for non-numeric verdicts (`T`/`F`/`?`/`D`).
        if actual == expected_unit.value || numeric_units_equal(&actual, &expected_unit.value) {
            if !expected_unit.soft {
                result.exact_units += 1;
            }
        } else if actual == "D" {
            if !expected_unit.soft {
                result.cannot_compute_units += 1;
            }
        } else if !expected_unit.soft {
            result.wrong_units += 1;
        }
    }

    if matches!(
        result.category.as_str(),
        "timeout" | "nonzero_exit" | "malformed_output"
    ) {
        return result;
    }
    if result.wrong_units > 0 {
        result.category = "wrong".to_string();
    } else if result.missing_units > 0 || result.cannot_compute_units > 0 {
        result.category = "incomplete".to_string();
    } else {
        result.category = "exact".to_string();
    }
    result
}

// ---------- Subprocess orchestration (run_with_timeout copied from sweep.rs lines 1700-1736) ----------

fn build_command(
    binary: &Path,
    model_dir: &Path,
    exam: Examination,
    threads: u32,
    inner_timeout: u64,
) -> Command {
    let mut cmd = Command::new(binary);
    let name = binary
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or_default();
    if name == "ty" {
        cmd.arg("mcc");
    }
    cmd.arg(model_dir)
        .arg("--examination")
        .arg(exam.as_str())
        .arg("--threads")
        .arg(threads.to_string())
        .arg("--timeout")
        .arg(inner_timeout.to_string());
    cmd
}

fn run_with_timeout(mut cmd: Command, timeout: Duration) -> Result<(i32, bool, String, String)> {
    cmd.stdout(Stdio::piped());
    cmd.stderr(Stdio::piped());
    let mut child = cmd.spawn().context("spawn candidate")?;
    let start = Instant::now();
    loop {
        match child.try_wait()? {
            Some(status) => {
                let mut stdout = String::new();
                let mut stderr = String::new();
                if let Some(mut s) = child.stdout.take() {
                    let _ = s.read_to_string(&mut stdout);
                }
                if let Some(mut s) = child.stderr.take() {
                    let _ = s.read_to_string(&mut stderr);
                }
                return Ok((status.code().unwrap_or(-1), false, stdout, stderr));
            }
            None => {
                if start.elapsed() >= timeout {
                    let _ = child.kill();
                    let _ = child.wait();
                    let mut stdout = String::new();
                    let mut stderr = String::new();
                    if let Some(mut s) = child.stdout.take() {
                        let _ = s.read_to_string(&mut stdout);
                    }
                    if let Some(mut s) = child.stderr.take() {
                        let _ = s.read_to_string(&mut stderr);
                    }
                    return Ok((124, true, stdout, stderr));
                }
                std::thread::sleep(Duration::from_millis(50));
            }
        }
    }
}

// ---------- MCC points scoring (rules III.4 / E-4.x) ----------
//
// This is the *ranking* score MCC uses to award medals — points, not solve
// rate. The full formula and verbatim rule quotes live in
// `docs/mcc-2026/scoring-formula.md`. Summary:
//
//   * Each examination instance has N trusted values (StateSpace=4,
//     GlobalProperties sub-exams=1 each, all other formula exams=16).
//   * ScoreValue = 16 / N  (E-4.1: examination total is 16 points).
//   * Per value: correct -> +ScoreValue; wrong -> -2*ScoreValue (PenaltyValue,
//     E-4.1/E-4.2: "worth twice the points when correct"); not-computed /
//     unknown -> 0 (E-4.2). Oracle-unknown values are excluded entirely.
//   * Examination score = sum over its values, then x model multiplier
//     (E-4.4: known x1, surprise x10).
//   * Category score = sum over its examinations; grand total = sum over all.
//
// SOUNDNESS: we read TY's per-value verdict tallies straight from the
// `RowResult` the comparator already produced (`exact_units` = correct,
// `wrong_units` = wrong, `cannot_compute_units` + `missing_units` = not
// computed). A value TY did not compute earns 0 — never a guess, never a
// penalty. Only `wrong_units` (TY emitted a concrete value disagreeing with a
// KNOWN oracle value) is penalized. Rows that never reached value-level
// comparison (timeout / nonzero_exit / malformed_output / missing_model_dir /
// the two harness_* categories) contribute 0 of everything: their
// exact/wrong/cannot/missing tallies are all zero, so they score 0. This is
// exactly the conservative behavior the task requires.

/// MCC scoring category (the six podiums of E-4.5).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum ScoreCategory {
    StateSpace,
    GlobalProperties,
    UpperBounds,
    ReachabilityFormulas,
    CtlFormulas,
    LtlFormulas,
}

impl ScoreCategory {
    fn as_str(self) -> &'static str {
        match self {
            Self::StateSpace => "StateSpace",
            Self::GlobalProperties => "GlobalProperties",
            Self::UpperBounds => "UpperBounds",
            Self::ReachabilityFormulas => "ReachabilityFormulas",
            Self::CtlFormulas => "CTLFormulas",
            Self::LtlFormulas => "LTLFormulas",
        }
    }

    /// The six categories in canonical (E-4.5) order.
    const ALL: [ScoreCategory; 6] = [
        Self::StateSpace,
        Self::GlobalProperties,
        Self::UpperBounds,
        Self::ReachabilityFormulas,
        Self::CtlFormulas,
        Self::LtlFormulas,
    ];
}

/// Maps an examination to its MCC scoring category (E-4.5) and the number of
/// trusted values N in one instance of that examination (E-4.3). The
/// per-value ScoreValue is then `16 / N`.
///
/// `Examination` is `#[non_exhaustive]`, so any future variant we don't yet
/// classify returns `None` and is scored as 0 — sound by construction (we
/// never assign points to an examination whose value count we don't know).
fn exam_category_and_value_count(exam: Examination) -> Option<(ScoreCategory, u32)> {
    Some(match exam {
        Examination::StateSpace => (ScoreCategory::StateSpace, 4),
        // GlobalProperties: each sub-examination is a single-value instance
        // (E-1.3 lists five subcategories; each is run independently and is
        // worth the full 16 points for its one value — see the doc's note).
        Examination::ReachabilityDeadlock
        | Examination::QuasiLiveness
        | Examination::StableMarking
        | Examination::Liveness
        | Examination::OneSafe => (ScoreCategory::GlobalProperties, 1),
        Examination::UpperBounds => (ScoreCategory::UpperBounds, 16),
        Examination::ReachabilityCardinality | Examination::ReachabilityFireability => {
            (ScoreCategory::ReachabilityFormulas, 16)
        }
        Examination::CTLCardinality | Examination::CTLFireability => {
            (ScoreCategory::CtlFormulas, 16)
        }
        Examination::LTLCardinality | Examination::LTLFireability => {
            (ScoreCategory::LtlFormulas, 16)
        }
        // Non-exhaustive guard: a future examination we haven't classified.
        _ => return None,
    })
}

const MCC_EXAMINATION_TOTAL_POINTS: f64 = 16.0;

#[derive(Debug, Clone, Default)]
struct CategoryScore {
    /// Net points (correct credit minus wrong penalties), already multiplied
    /// by the per-model known/surprise multiplier.
    points: f64,
    /// Best achievable: ScoreValue * (count of oracle-known values), summed,
    /// also multiplied by the model multiplier. The ceiling TY could reach by
    /// answering every oracle-known value correctly.
    max_possible: f64,
    /// Tally of value-level outcomes (un-multiplied, for the report).
    correct_values: u64,
    wrong_values: u64,
    notcomputed_values: u64,
    /// Examination instances that contributed at least one scored row.
    instances: u64,
}

#[derive(Debug, Clone, Default)]
struct ScoreReport {
    per_category: BTreeMap<String, CategoryScore>,
    grand_total: f64,
    grand_max_possible: f64,
    total_correct: u64,
    total_wrong: u64,
    total_notcomputed: u64,
}

/// Score a slice of comparison rows per the MCC points formula.
///
/// `surprise` is the set of model names to score with the x10 surprise
/// multiplier (E-4.4); every other model uses x1.
fn score_results(rows: &[RowResult], surprise: &BTreeSet<String>) -> ScoreReport {
    let mut report = ScoreReport::default();
    for row in rows {
        // Resolve the examination; unknown names cannot be scored (0).
        let Ok(exam) = Examination::from_name(&row.exam) else {
            continue;
        };
        let Some((category, n_values)) = exam_category_and_value_count(exam) else {
            continue;
        };
        if n_values == 0 {
            continue;
        }
        let score_value = MCC_EXAMINATION_TOTAL_POINTS / f64::from(n_values);

        // Per-value outcome tallies for THIS row, taken verbatim from the
        // comparator. These already exclude oracle-unknown values (the
        // comparator's loop skips `!expected_unit.known`) and never count a
        // value TY didn't compute as wrong.
        let correct = row.exact_units;
        let wrong = row.wrong_units;
        // Not-computed = TY emitted CANNOT_COMPUTE (`D`) or nothing at all for
        // an oracle-known value. Scored 0 either way (E-4.2).
        let notcomputed = row.cannot_compute_units + row.missing_units;
        // Oracle-known values for this instance = everything that reached the
        // value-level tally. (Soft/oracle-unknown values are not in any of
        // these buckets.)
        let known_values = correct + wrong + notcomputed;
        if known_values == 0 {
            // No oracle-known value to score (timeout, malformed, missing
            // model, fully-unknown oracle, ...). Contributes nothing.
            continue;
        }

        let multiplier = if surprise.contains(&row.input) {
            10.0
        } else {
            1.0
        };

        let raw_points = score_value * (correct as f64) - 2.0 * score_value * (wrong as f64);
        let instance_points = raw_points * multiplier;
        let instance_max = score_value * (known_values as f64) * multiplier;

        let entry = report
            .per_category
            .entry(category.as_str().to_string())
            .or_default();
        entry.points += instance_points;
        entry.max_possible += instance_max;
        entry.correct_values += correct;
        entry.wrong_values += wrong;
        entry.notcomputed_values += notcomputed;
        entry.instances += 1;

        report.grand_total += instance_points;
        report.grand_max_possible += instance_max;
        report.total_correct += correct;
        report.total_wrong += wrong;
        report.total_notcomputed += notcomputed;
    }
    report
}

impl ScoreReport {
    fn to_json(&self, field_bar: Option<f64>) -> serde_json::Value {
        let mut cats = serde_json::Map::new();
        for cat in ScoreCategory::ALL {
            let key = cat.as_str();
            if let Some(s) = self.per_category.get(key) {
                cats.insert(
                    key.to_string(),
                    json!({
                        "points": s.points,
                        "max_possible": s.max_possible,
                        "correct_values": s.correct_values,
                        "wrong_values": s.wrong_values,
                        "notcomputed_values": s.notcomputed_values,
                        "instances": s.instances,
                    }),
                );
            }
        }
        let margin = field_bar.map(|bar| self.grand_total - bar);
        json!({
            "formula_source": "MCC 2023 rules III.4 (E-4.1..E-4.6)",
            "per_category": cats,
            "grand_total_points": self.grand_total,
            "grand_max_possible_points": self.grand_max_possible,
            "total_correct_values": self.total_correct,
            "total_wrong_values": self.total_wrong,
            "total_notcomputed_values": self.total_notcomputed,
            "field_bar": field_bar,
            "margin_vs_field_bar": margin,
            "would_win": field_bar.map(|bar| self.grand_total >= bar),
        })
    }
}

fn print_score_report(report: &ScoreReport, field_bar: Option<f64>) {
    eprintln!("==== MCC POINTS SCORE (rules III.4 / E-4.x) ====");
    eprintln!(
        "{:<22} {:>12} {:>12} {:>8} {:>7} {:>10} {:>6}",
        "category", "points", "max_poss", "correct", "wrong", "notcomp", "exams"
    );
    for cat in ScoreCategory::ALL {
        let key = cat.as_str();
        let Some(s) = report.per_category.get(key) else {
            continue;
        };
        eprintln!(
            "{:<22} {:>12.3} {:>12.3} {:>8} {:>7} {:>10} {:>6}",
            key,
            s.points,
            s.max_possible,
            s.correct_values,
            s.wrong_values,
            s.notcomputed_values,
            s.instances,
        );
    }
    eprintln!(
        "{:<22} {:>12.3} {:>12.3} {:>8} {:>7} {:>10}",
        "GRAND TOTAL",
        report.grand_total,
        report.grand_max_possible,
        report.total_correct,
        report.total_wrong,
        report.total_notcomputed,
    );
    if let Some(bar) = field_bar {
        let margin = report.grand_total - bar;
        let verdict = if report.grand_total >= bar {
            "WOULD WIN"
        } else {
            "would NOT win"
        };
        eprintln!(
            "vs field bar {bar:.3}: ty={ty:.3}  margin={margin:+.3}  -> {verdict}",
            ty = report.grand_total,
        );
    }
    if report.total_wrong > 0 {
        eprintln!(
            "NOTE: {} wrong value(s) cost 2x ScoreValue each (PenaltyValue). \
             A 0-wrong run maximizes points.",
            report.total_wrong
        );
    }
}

// ---------- Output writers ----------

fn write_results_tsv(path: &Path, rows: &[RowResult]) -> Result<()> {
    let mut file = File::create(path).with_context(|| format!("create {}", path.display()))?;
    writeln!(
        file,
        "Input\tExamination\tExpected\tActual\tCategory\tExact_Units\tWrong_Units\tCannot_Compute_Units\tMissing_Units\tTime_Ms"
    )?;
    for row in rows {
        writeln!(
            file,
            "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
            tsv_clean(&row.input),
            tsv_clean(&row.exam),
            tsv_clean(&row.expected),
            tsv_clean(&row.actual),
            tsv_clean(&row.category),
            row.exact_units,
            row.wrong_units,
            row.cannot_compute_units,
            row.missing_units,
            row.elapsed_ms,
        )?;
    }
    Ok(())
}

fn tsv_clean(s: &str) -> String {
    s.replace(['\t', '\r', '\n'], " ")
}

fn aggregate(rows: &[RowResult], unique_cases_evaluated: usize) -> serde_json::Value {
    let mut totals = BTreeMap::<String, u64>::new();
    let mut wrong = 0u64;
    let mut exact = 0u64;
    let mut cannot = 0u64;
    let mut missing = 0u64;
    for r in rows {
        *totals.entry(r.category.clone()).or_default() += 1;
        wrong += r.wrong_units;
        exact += r.exact_units;
        cannot += r.cannot_compute_units;
        missing += r.missing_units;
    }
    // Backward-compatible field set: every original key (`rows`,
    // `wrong_units`, `exact_units`, `cannot_compute_units`, `missing_units`,
    // `categories`) is preserved with identical semantics. The new keys
    // (`unique_cases_evaluated`, `row_count`, `dedup_factor`) document the
    // (Input, Examination) dedup performed before `ty-mcc` is invoked: with
    // 7-tool replication on `raw-result-analysis.csv` the dedup factor is
    // ~7x. `row_count` is intentionally a synonym of `rows` for callers that
    // prefer the explicit name.
    let row_count = rows.len();
    let dedup_factor = if unique_cases_evaluated == 0 {
        0.0
    } else {
        row_count as f64 / unique_cases_evaluated as f64
    };
    json!({
        "rows": row_count,
        "row_count": row_count,
        "unique_cases_evaluated": unique_cases_evaluated,
        "dedup_factor": dedup_factor,
        "wrong_units": wrong,
        "exact_units": exact,
        "cannot_compute_units": cannot,
        "missing_units": missing,
        "categories": totals,
    })
}

// ---------- Tests ----------

#[cfg(test)]
mod scoring_tests {
    use super::*;

    /// Build a minimal `RowResult` with the value-level tallies the scorer
    /// reads. Other fields are irrelevant to scoring.
    fn row(
        input: &str,
        exam: Examination,
        exact: u64,
        wrong: u64,
        cannot: u64,
        missing: u64,
    ) -> RowResult {
        RowResult {
            input: input.to_string(),
            exam: exam.as_str().to_string(),
            category: "test".to_string(),
            exact_units: exact,
            wrong_units: wrong,
            cannot_compute_units: cannot,
            missing_units: missing,
            ..RowResult::default()
        }
    }

    fn approx(a: f64, b: f64) {
        assert!((a - b).abs() < 1e-9, "expected {b}, got {a}");
    }

    #[test]
    fn xml_vintage_year_is_edition_max_not_first_sorted_id() {
        // An MCC examination file carries forward older formulas: a 2025-edition
        // CTLCardinality.xml holds both -2025-NN and reused -2023-NN ids. The
        // harness sorts ids, so -2023-12 sorts ahead of -2025-00; taking the
        // FIRST id's year misread the edition as 2023 and false-rejected the
        // (valid) 2025 corpus as a vintage mismatch. The edition is the MAX year.
        let mixed: Vec<String> = vec![
            "Kanban-PT-00200-CTLCardinality-2023-12".to_string(),
            "Kanban-PT-00200-CTLCardinality-2023-13".to_string(),
            "Kanban-PT-00200-CTLCardinality-2025-00".to_string(),
            "Kanban-PT-00200-CTLCardinality-2025-11".to_string(),
        ];
        assert_eq!(extract_xml_vintage_year(&mixed).as_deref(), Some("2025"));

        // A genuine 2023-only corpus still reads as 2023, so a real
        // cross-edition mismatch vs a 2025 CSV is still caught.
        let only_2023: Vec<String> = vec![
            "Foo-CTLCardinality-2023-00".to_string(),
            "Foo-CTLCardinality-2023-01".to_string(),
        ];
        assert_eq!(
            extract_xml_vintage_year(&only_2023).as_deref(),
            Some("2023")
        );

        // No year tag (LTL/UpperBounds shape) => None => signal #1 skipped.
        let no_year: Vec<String> = vec!["Foo-UpperBounds-p0".to_string()];
        assert_eq!(extract_xml_vintage_year(&no_year), None);
    }

    #[test]
    fn all_correct_16_formula_examination_earns_max_16_points() {
        // ReachabilityCardinality has N=16 values -> ScoreValue = 16/16 = 1.
        // All 16 correct -> 16 * 1 = 16 points = the full examination total,
        // and that equals max_possible.
        let rows = vec![row(
            "modelA",
            Examination::ReachabilityCardinality,
            16,
            0,
            0,
            0,
        )];
        let report = score_results(&rows, &BTreeSet::new());
        approx(report.grand_total, 16.0);
        approx(report.grand_max_possible, 16.0);
        assert_eq!(report.total_wrong, 0);
        let cat = &report.per_category["ReachabilityFormulas"];
        approx(cat.points, 16.0);
        assert_eq!(cat.correct_values, 16);
        assert_eq!(cat.instances, 1);
    }

    #[test]
    fn one_wrong_value_incurs_double_penalty() {
        // 16-formula exam, ScoreValue = 1. 15 correct (+15), 1 wrong (-2).
        // Net = 15 - 2 = 13. Forgoing+penalizing one value swings 3 points
        // below the 16 a clean run would score, and below the 15 max_possible
        // (15 correct + 1 wrong = 16 oracle-known, but the wrong one cost 2).
        let rows = vec![row("modelA", Examination::CTLCardinality, 15, 1, 0, 0)];
        let report = score_results(&rows, &BTreeSet::new());
        // 15 * 1 - 2 * 1 * 1 = 13.
        approx(report.grand_total, 13.0);
        // max_possible = ScoreValue * known_values = 1 * 16 = 16.
        approx(report.grand_max_possible, 16.0);
        assert_eq!(report.total_wrong, 1);
        let cat = &report.per_category["CTLFormulas"];
        approx(cat.points, 13.0);
        assert_eq!(cat.wrong_values, 1);
    }

    #[test]
    fn statespace_value_score_is_four() {
        // StateSpace has N=4 -> ScoreValue = 16/4 = 4. All 4 correct -> 16.
        let rows = vec![row("m", Examination::StateSpace, 4, 0, 0, 0)];
        let report = score_results(&rows, &BTreeSet::new());
        approx(report.grand_total, 16.0);
        // One wrong StateSpace metric: 3 correct (+12), 1 wrong (-8) = 4.
        let rows2 = vec![row("m", Examination::StateSpace, 3, 1, 0, 0)];
        let report2 = score_results(&rows2, &BTreeSet::new());
        approx(report2.grand_total, 4.0); // 3*4 - 2*4*1 = 12 - 8 = 4.
    }

    #[test]
    fn global_properties_single_value_worth_full_sixteen() {
        // ReachabilityDeadlock is a single-value examination -> ScoreValue=16.
        let rows = vec![row("m", Examination::ReachabilityDeadlock, 1, 0, 0, 0)];
        let report = score_results(&rows, &BTreeSet::new());
        approx(report.grand_total, 16.0);
        // A wrong single value: 0 correct, 1 wrong -> -2*16 = -32.
        let rows2 = vec![row("m", Examination::OneSafe, 0, 1, 0, 0)];
        let report2 = score_results(&rows2, &BTreeSet::new());
        approx(report2.grand_total, -32.0);
        assert_eq!(report2.per_category["GlobalProperties"].wrong_values, 1);
    }

    #[test]
    fn not_computed_and_unknown_earn_zero_never_guess() {
        // 10 correct (+10), 0 wrong, 3 cannot-compute, 3 missing -> the 6
        // not-computed values earn 0 (no penalty, no credit). Net = 10.
        // max_possible covers all 16 oracle-known values = 16.
        let rows = vec![row("m", Examination::LTLFireability, 10, 0, 3, 3)];
        let report = score_results(&rows, &BTreeSet::new());
        approx(report.grand_total, 10.0);
        approx(report.grand_max_possible, 16.0);
        assert_eq!(report.total_wrong, 0);
        assert_eq!(report.total_notcomputed, 6);
    }

    #[test]
    fn rows_without_value_comparison_score_zero() {
        // A timeout/malformed row has all-zero tallies -> contributes nothing
        // (not even an instance). Soundness: we never invent points.
        let rows = vec![row("m", Examination::ReachabilityCardinality, 0, 0, 0, 0)];
        let report = score_results(&rows, &BTreeSet::new());
        approx(report.grand_total, 0.0);
        assert!(report.per_category.is_empty());
    }

    #[test]
    fn surprise_models_get_ten_x_multiplier() {
        // Same all-correct 16-formula instance, but flagged surprise -> x10.
        let mut surprise = BTreeSet::new();
        surprise.insert("surpriseModel".to_string());
        let rows = vec![row(
            "surpriseModel",
            Examination::ReachabilityFireability,
            16,
            0,
            0,
            0,
        )];
        let report = score_results(&rows, &surprise);
        approx(report.grand_total, 160.0); // 16 * 10.
        approx(report.grand_max_possible, 160.0);
        // Penalty also scales: 15 correct + 1 wrong, x10 -> (15 - 2) * 10 = 130.
        let rows2 = vec![row(
            "surpriseModel",
            Examination::ReachabilityFireability,
            15,
            1,
            0,
            0,
        )];
        let report2 = score_results(&rows2, &surprise);
        approx(report2.grand_total, 130.0);
    }

    #[test]
    fn category_and_grand_totals_sum_across_examinations() {
        let rows = vec![
            row("m1", Examination::ReachabilityCardinality, 16, 0, 0, 0), // +16 Reach
            row("m2", Examination::ReachabilityFireability, 14, 0, 1, 1), // +14 Reach
            row("m1", Examination::CTLCardinality, 16, 0, 0, 0),          // +16 CTL
            row("m1", Examination::StateSpace, 4, 0, 0, 0),               // +16 StateSpace
        ];
        let report = score_results(&rows, &BTreeSet::new());
        approx(report.per_category["ReachabilityFormulas"].points, 30.0);
        approx(report.per_category["CTLFormulas"].points, 16.0);
        approx(report.per_category["StateSpace"].points, 16.0);
        approx(report.grand_total, 62.0);
        assert_eq!(report.per_category["ReachabilityFormulas"].instances, 2);
    }

    #[test]
    fn field_bar_margin_and_would_win() {
        let rows = vec![row("m", Examination::ReachabilityCardinality, 16, 0, 0, 0)];
        let report = score_results(&rows, &BTreeSet::new());
        // ty=16. Beat a bar of 10.
        let j = report.to_json(Some(10.0));
        assert_eq!(j["would_win"], serde_json::json!(true));
        approx(j["margin_vs_field_bar"].as_f64().unwrap(), 6.0);
        // Lose to a bar of 20.
        let j2 = report.to_json(Some(20.0));
        assert_eq!(j2["would_win"], serde_json::json!(false));
        approx(j2["margin_vs_field_bar"].as_f64().unwrap(), -4.0);
    }
}
