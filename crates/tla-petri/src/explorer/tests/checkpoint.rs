// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

use super::fixtures::counting_net;
use super::*;
use crate::examinations::deadlock::DeadlockObserver;
use crate::examinations::state_space::StateSpaceObserver;
use serde_json::Value;

#[test]
fn test_checkpoint_resume_state_space_matches_fresh_run() {
    let net = counting_net(5);
    let tempdir = tempfile::tempdir().expect("tempdir");
    let checkpoint_dir = tempdir.path().join("state-space");

    let interrupted_config = ExplorationConfig::new(100).with_checkpoint(
        CheckpointConfig::new(checkpoint_dir.clone(), 2).with_stop_after_first_save(true),
    );
    let mut interrupted = StateSpaceObserver::new(&net.initial_marking);
    let interrupted_result =
        explore_checkpointable_observer(&net, &interrupted_config, &mut interrupted)
            .expect("checkpointed run should save successfully");
    assert!(!interrupted_result.completed);

    let resume_config = ExplorationConfig::new(100)
        .with_checkpoint(CheckpointConfig::new(checkpoint_dir, 2).with_resume(true));
    let mut resumed = StateSpaceObserver::new(&net.initial_marking);
    let resumed_result = explore_checkpointable_observer(&net, &resume_config, &mut resumed)
        .expect("resumed run should load checkpoint successfully");
    assert!(resumed_result.completed);

    let mut fresh = StateSpaceObserver::new(&net.initial_marking);
    let fresh_result = explore_observer(&net, &ExplorationConfig::new(100), &mut fresh);
    assert!(fresh_result.completed);

    assert_eq!(resumed.stats().states, fresh.stats().states);
    assert_eq!(resumed.stats().edges, fresh.stats().edges);
    assert_eq!(
        resumed.stats().max_token_in_place,
        fresh.stats().max_token_in_place
    );
    assert_eq!(resumed.stats().max_token_sum, fresh.stats().max_token_sum);
}

#[test]
fn test_checkpoint_resume_rejects_wrong_observer_kind() {
    let net = counting_net(3);
    let tempdir = tempfile::tempdir().expect("tempdir");
    let checkpoint_dir = tempdir.path().join("observer-mismatch");

    let interrupted_config = ExplorationConfig::new(100).with_checkpoint(
        CheckpointConfig::new(checkpoint_dir.clone(), 1).with_stop_after_first_save(true),
    );
    let mut interrupted = StateSpaceObserver::new(&net.initial_marking);
    let _ = explore_checkpointable_observer(&net, &interrupted_config, &mut interrupted)
        .expect("checkpoint write should succeed");

    let resume_config = ExplorationConfig::new(100)
        .with_checkpoint(CheckpointConfig::new(checkpoint_dir, 1).with_resume(true));
    let mut wrong_observer = DeadlockObserver::new();
    let error = explore_checkpointable_observer(&net, &resume_config, &mut wrong_observer)
        .expect_err("loading a state-space checkpoint into deadlock observer must fail");
    assert_eq!(error.kind(), std::io::ErrorKind::InvalidData);
}

#[test]
fn test_checkpoint_manifest_uses_exact_subbyte_packed_bytes() {
    let places = (0..10)
        .map(|idx| PlaceInfo {
            id: format!("P{idx}"),
            name: None,
        })
        .collect();
    let net = PetriNet {
        name: Some("subbyte-checkpoint".into()),
        places,
        transitions: vec![],
        initial_marking: vec![1; 10],
    };
    let setup = ExplorationSetup::analyze(&net);
    assert_eq!(setup.marking_config.packed_len, 10);
    assert_eq!(setup.marking_config.width.bytes(), 1);
    assert_eq!(setup.marking_config.packed_bytes, 2);

    let tempdir = tempfile::tempdir().expect("tempdir");
    let checkpoint_dir = tempdir.path().join("subbyte");
    let config = ExplorationConfig::new(100)
        .with_checkpoint(CheckpointConfig::new(checkpoint_dir.clone(), 1));
    let mut observer = StateSpaceObserver::new(&net.initial_marking);
    let result = explore_checkpointable_observer(&net, &config, &mut observer)
        .expect("checkpoint write should succeed");

    assert!(result.completed);
    let manifest_path = checkpoint_dir.join("checkpoint.json");
    let manifest: Value = serde_json::from_reader(
        std::fs::File::open(manifest_path).expect("manifest should be written"),
    )
    .expect("manifest should parse");
    assert_eq!(manifest["packed_bytes"], 2);
}
