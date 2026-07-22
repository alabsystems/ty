// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

use super::fixtures::simple_net;
use super::*;
use crate::examination::ExaminationValue;
use crate::examinations::certified::CertifiedExaminationRecord;
use crate::explorer::{
    explore_checkpointable_observer, explore_observer, CheckpointConfig, ExplorationConfig,
};
use crate::reduction::ReducedNet;
use tempfile::tempdir;

use super::super::types::{assemble_results_certified, flush_certified};

fn eventual_tokens_in_p1_property() -> Vec<Property> {
    vec![Property {
        id: "Model-RC-00".to_string(),
        formula: Formula::Reachability(ReachabilityFormula {
            quantifier: PathQuantifier::EF,
            predicate: StatePredicate::IntLe(
                IntExpr::Constant(3),
                IntExpr::TokensCount(vec!["p1".to_string()]),
            ),
        }),
    }]
}

fn certified_parity_properties() -> Vec<Property> {
    vec![
        Property {
            id: "Model-RC-EF".to_string(),
            formula: Formula::Reachability(ReachabilityFormula {
                quantifier: PathQuantifier::EF,
                predicate: StatePredicate::IntLe(
                    IntExpr::Constant(1),
                    IntExpr::TokensCount(vec!["p1".to_string()]),
                ),
            }),
        },
        Property {
            id: "Model-RC-AG".to_string(),
            formula: Formula::Reachability(ReachabilityFormula {
                quantifier: PathQuantifier::AG,
                predicate: StatePredicate::IntLe(
                    IntExpr::Constant(1),
                    IntExpr::TokensCount(vec!["p1".to_string()]),
                ),
            }),
        },
    ]
}

fn certified_to_legacy(records: Vec<CertifiedExaminationRecord>) -> Vec<(String, Verdict)> {
    records
        .into_iter()
        .map(|record| {
            let legacy = record.to_legacy_record();
            let ExaminationValue::Verdict(verdict) = legacy.value else {
                panic!("reachability certified record must adapt to Verdict");
            };
            (legacy.formula_id, verdict)
        })
        .collect()
}

#[test]
fn reachability_observer_checkpoint_resume_matches_fresh_run() {
    let net = simple_net();
    let properties = eventual_tokens_in_p1_property();
    let tempdir = tempdir().expect("tempdir");
    let checkpoint_dir = tempdir.path().join("reachability-observer");

    let interrupted_config = ExplorationConfig::new(100).with_checkpoint(
        CheckpointConfig::new(checkpoint_dir.clone(), 1).with_stop_after_first_save(true),
    );
    let mut interrupted = ReachabilityObserver::new(&net, &properties);
    let interrupted_result =
        explore_checkpointable_observer(&net, &interrupted_config, &mut interrupted)
            .expect("checkpointed reachability run should save successfully");
    assert!(!interrupted_result.completed);
    assert_eq!(interrupted.results(), vec![("Model-RC-00", None)]);

    let resume_config = ExplorationConfig::new(100)
        .with_checkpoint(CheckpointConfig::new(checkpoint_dir, 1).with_resume(true));
    let mut resumed = ReachabilityObserver::new(&net, &properties);
    let resumed_result = explore_checkpointable_observer(&net, &resume_config, &mut resumed)
        .expect("resumed reachability run should load checkpoint successfully");
    assert!(resumed_result.stopped_by_observer);

    let mut fresh = ReachabilityObserver::new(&net, &properties);
    let fresh_result = explore_observer(&net, &ExplorationConfig::new(100), &mut fresh);
    assert!(fresh_result.stopped_by_observer);

    assert_eq!(resumed.results_completed(), fresh.results_completed());
}

#[test]
fn reduced_reachability_checkpoint_resume_matches_fresh_run() {
    let net = simple_net();
    let properties = eventual_tokens_in_p1_property();
    let (prepared, trackers) = prepare_trackers(&net, &properties);
    let reduced = ReducedNet::identity(&net);
    let tempdir = tempdir().expect("tempdir");
    let checkpoint_dir = tempdir.path().join("reachability-reduced");

    let interrupted_config = ExplorationConfig::new(100).with_checkpoint(
        CheckpointConfig::new(checkpoint_dir.clone(), 1).with_stop_after_first_save(true),
    );
    let (interrupted_result, interrupted_trackers) =
        explore_reachability_on_reduced_net(&net, &reduced, trackers.clone(), &interrupted_config)
            .expect("checkpointed reduced-net exploration should succeed");
    assert!(!interrupted_result.completed);
    assert_eq!(
        assemble_results(
            &prepared,
            &interrupted_trackers,
            interrupted_result.completed,
            false
        ),
        vec![("Model-RC-00".to_string(), Verdict::CannotCompute)]
    );

    let resume_config = ExplorationConfig::new(100)
        .with_checkpoint(CheckpointConfig::new(checkpoint_dir, 1).with_resume(true));
    let (resumed_result, resumed_trackers) =
        explore_reachability_on_reduced_net(&net, &reduced, trackers.clone(), &resume_config)
            .expect("resumed reduced-net exploration should load checkpoint successfully");
    assert!(resumed_result.stopped_by_observer);

    let (fresh_result, fresh_trackers) =
        explore_reachability_on_reduced_net(&net, &reduced, trackers, &ExplorationConfig::new(100))
            .expect("fresh reduced-net exploration should succeed");
    assert!(fresh_result.stopped_by_observer);

    assert_eq!(
        assemble_results(
            &prepared,
            &resumed_trackers,
            resumed_result.completed,
            false
        ),
        assemble_results(&prepared, &fresh_trackers, fresh_result.completed, false),
    );
}

#[test]
fn assemble_results_certified_matches_legacy_for_incomplete() {
    let net = simple_net();
    let properties = certified_parity_properties();
    let (prepared, trackers) = prepare_trackers(&net, &properties);

    let legacy = assemble_results(&prepared, &trackers, false, false);
    let certified = certified_to_legacy(assemble_results_certified(
        &prepared, &trackers, false, false,
    ));

    assert_eq!(certified, legacy);
    assert_eq!(
        legacy,
        vec![
            ("Model-RC-EF".to_string(), Verdict::CannotCompute),
            ("Model-RC-AG".to_string(), Verdict::CannotCompute),
        ]
    );
}

#[test]
fn assemble_results_certified_matches_legacy_for_completed() {
    let net = simple_net();
    let properties = certified_parity_properties();
    let (prepared, trackers) = prepare_trackers(&net, &properties);

    let legacy = assemble_results(&prepared, &trackers, true, false);
    let certified = certified_to_legacy(assemble_results_certified(
        &prepared, &trackers, true, false,
    ));

    assert_eq!(certified, legacy);
    assert_eq!(
        legacy,
        vec![
            ("Model-RC-EF".to_string(), Verdict::False),
            ("Model-RC-AG".to_string(), Verdict::True),
        ]
    );
}

#[test]
fn assemble_results_certified_skips_flushed_like_legacy() {
    let net = simple_net();
    let properties = certified_parity_properties();
    let (prepared, mut trackers) = prepare_trackers(&net, &properties);
    trackers[0].flushed = true;

    let legacy = assemble_results(&prepared, &trackers, false, true);
    let certified = certified_to_legacy(assemble_results_certified(
        &prepared, &trackers, false, true,
    ));

    assert_eq!(certified, legacy);
    assert_eq!(
        legacy,
        vec![("Model-RC-AG".to_string(), Verdict::CannotCompute)]
    );
}

#[test]
fn flush_certified_marks_and_counts_like_flush_resolved() {
    let net = simple_net();
    let properties = certified_parity_properties();
    let (_prepared, mut trackers) = prepare_trackers(&net, &properties);
    trackers[0].verdict = Some(true);

    let flushed = flush_certified(&mut trackers);

    assert_eq!(flushed, 1);
    assert!(trackers[0].flushed);
    assert!(!trackers[1].flushed);
}
