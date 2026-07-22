// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! EFMNOP reduction evidence harness.
//!
//! This is intentionally a test harness rather than production API. It can
//! inspect crate-private legacy and EFMNOP analysis paths without widening the
//! public Petri surface just to collect benchmark evidence.

use std::path::{Path, PathBuf};

use crate::petri_net::PetriNet;
use crate::reduction::analysis::{analyze, analyze_efmnop_fixpoint};
use crate::reduction::{reduce_iterative_structural_with_mode, ReductionMode, ReductionReport};

struct BenchCase {
    label: String,
    source: String,
    net: PetriNet,
}

#[derive(Debug)]
struct RemovalMetrics {
    places: usize,
    transitions: usize,
    added_transitions: usize,
}

impl RemovalMetrics {
    fn from_report(report: &ReductionReport) -> Self {
        Self {
            places: report.places_removed(),
            transitions: report.transitions_removed(),
            added_transitions: report.transitions_added(),
        }
    }

    fn total_removed(&self) -> usize {
        self.places + self.transitions
    }

    fn net_removed(&self) -> isize {
        self.total_removed() as isize - self.added_transitions as isize
    }
}

fn ratio(metrics: &RemovalMetrics, original_places: usize, original_transitions: usize) -> f64 {
    let total = original_places + original_transitions;
    if total == 0 {
        0.0
    } else {
        metrics.net_removed().max(0) as f64 / total as f64
    }
}

fn pct(delta: f64) -> f64 {
    delta * 100.0
}

fn mcc_fixture_cases() -> Vec<BenchCase> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join("tests")
        .join("mcc_benchmarks");
    load_model_dirs(&root)
}

fn load_model_dirs(root: &Path) -> Vec<BenchCase> {
    let Ok(entries) = std::fs::read_dir(root) else {
        return Vec::new();
    };
    let mut cases = Vec::new();
    for entry in entries.flatten() {
        let path = entry.path();
        let pnml = path.join("model.pnml");
        if !pnml.exists() {
            continue;
        }
        let Some(label) = path.file_name().and_then(|name| name.to_str()) else {
            continue;
        };
        let net = crate::parser::parse_pnml_file(&pnml)
            .unwrap_or_else(|error| panic!("failed to parse {}: {error:?}", pnml.display()));
        cases.push(BenchCase {
            label: label.to_owned(),
            source: pnml.display().to_string(),
            net,
        });
    }
    cases.sort_by(|a, b| a.label.cmp(&b.label));
    cases
}

#[test]
fn test_efmnop_reduction_benchmark_evidence_from_env_or_fixtures() {
    let mut cases = mcc_fixture_cases();
    let default_fixture_count = cases.len();

    if let Some(root) = std::env::var_os("TY_EFMNOP_MCC_INPUT_ROOT") {
        let root = PathBuf::from(root);
        assert!(
            root.exists(),
            "TY_EFMNOP_MCC_INPUT_ROOT must exist: {}",
            root.display()
        );
        cases.extend(load_model_dirs(&root));
    }

    assert!(
        !cases.is_empty(),
        "expected at least one MCC fixture or TY_EFMNOP_MCC_INPUT_ROOT model"
    );

    let mut improved_by_ten_pct = Vec::new();
    for case in &cases {
        let original_places = case.net.num_places();
        let original_transitions = case.net.num_transitions();
        let legacy = analyze(&case.net);
        let efmnop = analyze_efmnop_fixpoint(&case.net, &[], ReductionMode::Reachability);
        let structural = reduce_iterative_structural_with_mode(
            &case.net,
            &[],
            ReductionMode::Reachability,
            None,
        )
        .unwrap_or_else(|error| panic!("failed to reduce {}: {error:?}", case.label));

        let legacy_metrics = RemovalMetrics::from_report(&legacy);
        let efmnop_metrics = RemovalMetrics::from_report(&efmnop.report);
        let legacy_ratio = ratio(&legacy_metrics, original_places, original_transitions);
        let efmnop_ratio = ratio(&efmnop_metrics, original_places, original_transitions);
        let improvement = efmnop_ratio - legacy_ratio;
        if improvement >= 0.10 {
            improved_by_ten_pct.push(case.label.as_str());
        }

        println!(
            "EFMNOP_BENCH label={} source={} original_places={} original_transitions={} legacy_places_removed={} legacy_transitions_removed={} legacy_transitions_added={} legacy_net_removed={} legacy_ratio_pct={:.2} efmnop_places_removed={} efmnop_transitions_removed={} efmnop_transitions_added={} efmnop_net_removed={} efmnop_ratio_pct={:.2} improvement_pct={:.2} efmnop_iterations={} efmnop_cascade_dead={} efmnop_rule_e_workqueue_dead={} efmnop_rule_n_fixpoint_lower_bound={} structural_reduced_places={} structural_reduced_transitions={}",
            case.label,
            case.source,
            original_places,
            original_transitions,
            legacy_metrics.places,
            legacy_metrics.transitions,
            legacy_metrics.added_transitions,
            legacy_metrics.net_removed(),
            pct(legacy_ratio),
            efmnop_metrics.places,
            efmnop_metrics.transitions,
            efmnop_metrics.added_transitions,
            efmnop_metrics.net_removed(),
            pct(efmnop_ratio),
            pct(improvement),
            efmnop.iterations,
            efmnop.dead_removed_by_cascade,
            efmnop.per_rule_progress.rule_e_workqueue_dead,
            efmnop.per_rule_progress.rule_n_fixpoint_lower_bound,
            structural.net.num_places(),
            structural.net.num_transitions(),
        );
    }

    println!(
        "EFMNOP_BENCH_SUMMARY cases={} default_mcc_fixtures={} improved_by_10pct={} improved_labels={:?}",
        cases.len(),
        default_fixture_count,
        improved_by_ten_pct.len(),
        improved_by_ten_pct,
    );

    if std::env::var_os("TY_EFMNOP_REQUIRE_10PCT").is_some() {
        assert!(
            !improved_by_ten_pct.is_empty(),
            "expected at least one EFMNOP benchmark case with >=10 percentage-point reduction-ratio improvement"
        );
    }
}
