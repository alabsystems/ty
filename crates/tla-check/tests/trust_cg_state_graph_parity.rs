// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Cross-backend state-graph parity for trust-codegen native dispatch.
//!
//! Part of #4374: trust-codegen #384 needs a committed TY-owned test that compares
//! the reachable graph, not only terminal state counts.

mod common;

use common::EnvVarGuard;
use std::path::{Path, PathBuf};
use tla_check::{
    resolve_spec_from_config_with_extends, CheckResult, Config, FairnessConstraint, ModelChecker,
    StateGraphSnapshot,
};
use tla_core::{lower, parse_to_syntax_tree, FileId, ModuleLoader};

struct LoadedFixture {
    module: tla_core::ast::Module,
    checker_modules: Vec<tla_core::ast::Module>,
    config: Config,
    fairness: Vec<FairnessConstraint>,
    stuttering_allowed: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct RunSummary {
    states_found: usize,
    initial_states: usize,
    transitions: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct GraphRun {
    summary: RunSummary,
    graph: StateGraphSnapshot,
    trust_cg_action_coverage: Option<(usize, usize)>,
    trust_cg_action_dispatch_stats: Option<(usize, usize, usize)>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct SummaryRun {
    summary: RunSummary,
    trust_cg_action_coverage: Option<(usize, usize)>,
    trust_cg_action_dispatch_stats: Option<(usize, usize, usize)>,
}

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .to_path_buf()
}

fn load_fixture(spec_name: &str, cfg_name: &str) -> LoadedFixture {
    let spec_path = repo_root().join("test_specs").join(spec_name);
    let cfg_path = repo_root().join("test_specs").join(cfg_name);

    tla_core::clear_global_name_interner();
    let spec_source = std::fs::read_to_string(&spec_path)
        .unwrap_or_else(|error| panic!("failed to read {}: {error}", spec_path.display()));
    let tree = parse_to_syntax_tree(&spec_source);
    let lower_result = lower(FileId(0), &tree);
    let mut module = lower_result
        .module
        .unwrap_or_else(|| panic!("failed to lower {}", spec_path.display()));
    tla_core::compute_is_recursive(&mut module);

    let mut loader = ModuleLoader::new(&spec_path);
    loader.seed_from_syntax_tree(&tree, &spec_path);
    loader
        .load_extends(&module)
        .unwrap_or_else(|error| panic!("failed to load EXTENDS for {}: {error}", spec_name));
    loader
        .load_instances(&module)
        .unwrap_or_else(|error| panic!("failed to load INSTANCEs for {}: {error}", spec_name));

    let (extended_modules_for_resolve, instanced_modules_for_resolve) =
        loader.modules_for_semantic_resolution(&module);
    let spec_scope_module_names: Vec<&str> = extended_modules_for_resolve
        .iter()
        .chain(instanced_modules_for_resolve.iter())
        .map(|loaded| loaded.name.node.as_str())
        .collect();
    let extended_syntax_trees: Vec<_> = spec_scope_module_names
        .iter()
        .filter_map(|name| loader.get(name).map(|loaded| &loaded.syntax_tree))
        .collect();
    let checker_modules = loader
        .modules_for_model_checking(&module)
        .into_iter()
        .cloned()
        .collect::<Vec<_>>();

    let cfg_source = std::fs::read_to_string(&cfg_path)
        .unwrap_or_else(|error| panic!("failed to read {}: {error}", cfg_path.display()));
    let mut config = Config::parse(&cfg_source)
        .unwrap_or_else(|errors| panic!("failed to parse {}: {errors:?}", cfg_path.display()));
    let resolved = resolve_spec_from_config_with_extends(&config, &tree, &extended_syntax_trees)
        .unwrap_or_else(|error| panic!("failed to resolve SPEC for {}: {error}", cfg_name));
    if config.init.is_none() {
        config.init = Some(resolved.init.clone());
    }
    if config.next.is_none() {
        config.next = Some(resolved.next.clone());
    }
    config.auto_por = Some(false);

    LoadedFixture {
        module,
        checker_modules,
        config,
        fairness: resolved.fairness,
        stuttering_allowed: resolved.stuttering_allowed,
    }
}

#[derive(Debug, Clone, Copy)]
struct FixtureOptions {
    require_trust_cg_actions: bool,
    require_trust_cg_dispatch: bool,
    disable_flat_bfs: bool,
    use_compiled_bfs: bool,
}

fn run_graph(fixture: &LoadedFixture, trust_cg: bool, options: FixtureOptions) -> GraphRun {
    let _trust_cg = EnvVarGuard::set("TY_trust_cg", trust_cg.then_some("1"));
    let _trust_cg_bfs = EnvVarGuard::remove("TY_TRUST_CG_BFS");
    let _no_compiled = EnvVarGuard::remove("TY_NO_COMPILED_BFS");
    let _entry_counter_gate = EnvVarGuard::set(
        "TY_TRUST_CG_ENTRY_COUNTER_GATE",
        (trust_cg && options.require_trust_cg_dispatch).then_some("1000000"),
    );
    let _no_flat = EnvVarGuard::set("TY_NO_FLAT_BFS", options.disable_flat_bfs.then_some("1"));
    let _auto_por = EnvVarGuard::set("TY_AUTO_POR", Some("0"));

    tla_eval::clear_for_test_reset();
    let mut config = fixture.config.clone();
    config.use_compiled_bfs = Some(options.use_compiled_bfs);
    let checker_modules = fixture.checker_modules.iter().collect::<Vec<_>>();
    let mut checker = ModelChecker::new_with_extends(&fixture.module, &checker_modules, &config);
    checker.set_store_states(false);
    checker.set_collect_coverage(options.require_trust_cg_dispatch);
    checker.enable_state_graph_capture_for_testing();
    checker.set_fairness(fixture.fairness.clone());
    checker.set_stuttering_allowed(fixture.stuttering_allowed);

    let stats = match checker.check() {
        CheckResult::Success(stats) => stats,
        other => panic!("expected successful model check, got {other:?}"),
    };
    let graph = checker.state_graph_snapshot_for_testing();
    assert_eq!(
        graph.nodes.len(),
        stats.states_found,
        "captured graph nodes must match the explored state count"
    );
    assert_eq!(
        graph.successor_count, stats.transitions,
        "captured graph edges must match the explored transition count"
    );

    GraphRun {
        summary: RunSummary {
            states_found: stats.states_found,
            initial_states: stats.initial_states,
            transitions: stats.transitions,
        },
        graph,
        trust_cg_action_coverage: checker.trust_cg_action_coverage_for_testing(),
        trust_cg_action_dispatch_stats: checker.trust_cg_action_dispatch_stats_for_testing(),
    }
}

fn run_summary(fixture: &LoadedFixture, trust_cg: bool, options: FixtureOptions) -> SummaryRun {
    let _trust_cg = EnvVarGuard::set("TY_TRUST_CG", trust_cg.then_some("1"));
    let _trust_cg_bfs = EnvVarGuard::remove("TY_TRUST_CG_BFS");
    let _no_compiled = EnvVarGuard::remove("TY_NO_COMPILED_BFS");
    let _entry_counter_gate = EnvVarGuard::set(
        "TY_TRUST_CG_ENTRY_COUNTER_GATE",
        (trust_cg && options.require_trust_cg_dispatch).then_some("1000000"),
    );
    let _no_flat = EnvVarGuard::set("TY_NO_FLAT_BFS", options.disable_flat_bfs.then_some("1"));
    let _auto_por = EnvVarGuard::set("TY_AUTO_POR", Some("0"));

    tla_eval::clear_for_test_reset();
    let mut config = fixture.config.clone();
    config.use_compiled_bfs = Some(options.use_compiled_bfs);
    let checker_modules = fixture.checker_modules.iter().collect::<Vec<_>>();
    let mut checker = ModelChecker::new_with_extends(&fixture.module, &checker_modules, &config);
    checker.set_store_states(false);
    checker.set_collect_coverage(options.require_trust_cg_dispatch);
    checker.set_fairness(fixture.fairness.clone());
    checker.set_stuttering_allowed(fixture.stuttering_allowed);

    let stats = match checker.check() {
        CheckResult::Success(stats) => stats,
        other => panic!("expected successful model check, got {other:?}"),
    };

    SummaryRun {
        summary: RunSummary {
            states_found: stats.states_found,
            initial_states: stats.initial_states,
            transitions: stats.transitions,
        },
        trust_cg_action_coverage: checker.trust_cg_action_coverage_for_testing(),
        trust_cg_action_dispatch_stats: checker.trust_cg_action_dispatch_stats_for_testing(),
    }
}

fn assert_fixture_parity(spec_name: &str, cfg_name: &str, options: FixtureOptions) -> RunSummary {
    let fixture = load_fixture(spec_name, cfg_name);
    let baseline = run_graph(&fixture, false, options);
    let trust_cg = run_graph(&fixture, true, options);

    assert_eq!(
        trust_cg.summary, baseline.summary,
        "{spec_name}: backend summaries must match"
    );
    assert_eq!(
        trust_cg.graph, baseline.graph,
        "{spec_name}: reachable state graph must match exactly"
    );
    if options.require_trust_cg_actions {
        let (compiled, total) = trust_cg
            .trust_cg_action_coverage
            .expect("trust-cg run should record action coverage");
        assert!(
            compiled > 0 && compiled == total,
            "{spec_name}: expected all trust-codegen action instances to compile, got {compiled}/{total}"
        );
    }
    if options.require_trust_cg_dispatch {
        let (enabled, disabled, runtime_errors) = trust_cg
            .trust_cg_action_dispatch_stats
            .expect("trust-cg run should record action dispatch stats");
        assert_eq!(
            runtime_errors, 0,
            "{spec_name}: trust-codegen per-action dispatch should not hit runtime errors"
        );
        assert!(
            enabled > 0,
            "{spec_name}: trust-codegen per-action dispatch should produce enabled successors"
        );
        assert!(
            enabled + disabled > 0,
            "{spec_name}: trust-codegen per-action dispatch should execute native actions"
        );
    }
    baseline.summary
}

fn assert_ignored_dynamic_range_regression(
    spec_name: &str,
    cfg_name: &str,
    expected_summary: RunSummary,
    expected_compiled_actions: (usize, usize),
) {
    let fixture = load_fixture(spec_name, cfg_name);
    let options = FixtureOptions {
        require_trust_cg_actions: true,
        require_trust_cg_dispatch: true,
        disable_flat_bfs: true,
        use_compiled_bfs: false,
    };
    let baseline = run_summary(&fixture, false, options);
    let trust_cg = run_summary(&fixture, true, options);

    assert_eq!(
        trust_cg.summary, baseline.summary,
        "{spec_name}: Trust-CG must match default backend summary"
    );
    assert_eq!(
        baseline.summary, expected_summary,
        "{spec_name}: fixture should stay small and deterministic"
    );

    let compiled_actions = trust_cg
        .trust_cg_action_coverage
        .expect("Trust-CG run should record action coverage");
    assert_eq!(
        compiled_actions, expected_compiled_actions,
        "{spec_name}: unexpected Trust-CG action coverage"
    );

    let (enabled, disabled, runtime_errors) = trust_cg
        .trust_cg_action_dispatch_stats
        .expect("Trust-CG run should record action dispatch stats");
    assert_eq!(
        runtime_errors, 0,
        "{spec_name}: Trust-CG native dispatch should not hit runtime errors"
    );
    assert!(
        enabled > 0,
        "{spec_name}: Trust-CG should produce enabled successors"
    );
    assert!(
        enabled + disabled > 0,
        "{spec_name}: Trust-CG native action should be dispatched"
    );
}

#[cfg_attr(test, ntest::timeout(120000))]
#[test]
fn trust_cg_state_graph_matches_default_backend_for_add_two_record_sequence_and_bcast_small() {
    let add_two = assert_fixture_parity(
        "AddTwoTest.tla",
        "AddTwoTest.cfg",
        FixtureOptions {
            require_trust_cg_actions: true,
            require_trust_cg_dispatch: false,
            disable_flat_bfs: true,
            use_compiled_bfs: true,
        },
    );
    let aggregate = assert_fixture_parity(
        "TrustCgRecordSequenceParity.tla",
        "TrustCgRecordSequenceParity.cfg",
        FixtureOptions {
            require_trust_cg_actions: true,
            require_trust_cg_dispatch: true,
            disable_flat_bfs: true,
            use_compiled_bfs: false,
        },
    );
    let bcast = assert_fixture_parity(
        "bcastFolklore.tla",
        "bcastFolklore_small.cfg",
        FixtureOptions {
            require_trust_cg_actions: false,
            require_trust_cg_dispatch: false,
            disable_flat_bfs: true,
            use_compiled_bfs: true,
        },
    );
    assert!(
        aggregate.states_found > add_two.states_found,
        "record/sequence parity fixture should be larger than AddTwoTest"
    );
    assert!(
        bcast.states_found > add_two.states_found,
        "bcastFolklore_small should be the larger parity fixture"
    );
}

#[cfg_attr(test, ntest::timeout(30000))]
#[test]
#[ignore = "known Trust-CG mismatch: InsertNewline-style range FuncDef with guarded sequence indexing violates TypeOK"]
fn trust_cg_matches_default_backend_for_insert_newline_range_function() {
    assert_ignored_dynamic_range_regression(
        "InsertNewlineRangeFunc.tla",
        "InsertNewlineRangeFunc.cfg",
        RunSummary {
            states_found: 6,
            initial_states: 3,
            transitions: 9,
        },
        (1, 2),
    );
}

#[cfg_attr(test, ntest::timeout(30000))]
#[test]
#[ignore = "known Trust-CG mismatch: dynamic range FuncDef assigned into compact sequence storage"]
fn trust_cg_matches_default_backend_for_dynamic_range_func_compact_sequence() {
    assert_ignored_dynamic_range_regression(
        "DynamicRangeFuncCompactSeq.tla",
        "DynamicRangeFuncCompactSeq.cfg",
        RunSummary {
            states_found: 4,
            initial_states: 2,
            transitions: 6,
        },
        (1, 2),
    );
}
