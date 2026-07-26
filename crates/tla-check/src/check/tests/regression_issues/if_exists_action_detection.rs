// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

use super::*;
use tla_core::{lower, parse_to_syntax_tree, FileId};

/// Regression for detected-action decomposition of an IF whose condition has
/// multiple existential witnesses. The condition is a Boolean selector, so its
/// three witnesses must enable `Advance` once, not emit three copies of the
/// same successor through a synthesized action-level conjunction.
#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn if_exists_condition_emits_one_successor_and_attributes_the_selected_branch() {
    let src = r#"
---- MODULE IfExistsActionDetection ----
VARIABLE x

Init == x = 0

HasWitness == \E i \in {1, 2, 3} : x = 0
Advance == x' = 1
Stop == FALSE

Next == IF HasWitness THEN Advance ELSE Stop
====
"#;

    let tree = parse_to_syntax_tree(src);
    let lower_result = lower(FileId(0), &tree);
    assert!(
        lower_result.errors.is_empty(),
        "lowering errors: {:?}",
        lower_result.errors
    );
    let module = lower_result.module.expect("module should lower");
    let config = Config {
        init: Some("Init".to_string()),
        next: Some("Next".to_string()),
        auto_por: Some(false),
        ..Default::default()
    };

    let mut checker = ModelChecker::new(&module, &config);
    checker.set_deadlock_check(false);
    checker.set_collect_coverage(true);
    let result = checker.check();

    let CheckResult::Success(stats) = result else {
        panic!("expected success, got {result:?}");
    };
    assert_eq!(stats.states_found, 2);
    assert_eq!(stats.raw_initial_states_generated, 1);
    assert_eq!(
        stats.raw_successors_generated, 1,
        "an existential IF condition is Boolean, not one action proof per witness"
    );

    let coverage = stats
        .coverage
        .expect("default dead-action tracking should collect branch coverage");
    let mut advance = None;
    let mut stop = None;
    for action in coverage.actions.values() {
        match action.name.as_str() {
            "Advance" => advance = Some((action.transitions, action.times_fired)),
            "Stop" => stop = Some((action.transitions, action.times_fired)),
            _ => {}
        }
    }
    assert_eq!(advance, Some((1, 1)));
    assert_eq!(stop, Some((0, 0)));
}
