// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! WP-16: interpreter-vs-fused summary equality (transitions accounting).
//!
//! The whole-state native fused BFS loop (`trust_cg_native` fused loop in
//! `bfs/compiled_bfs_loop.rs`) counts one transition per kernel-emitted
//! successor record. Runtime-guarded inner-EXISTS witness expansion compiles
//! one kernel per witness of the static domain universe, and each kernel used
//! to contain the FULL action body — so an enabling path that bypasses the
//! exists region entirely (a witness-independent branch, e.g. Bakery
//! `e2`/`w1`: `IF unchecked[self] # {} THEN \E i \in unchecked[self]: ..
//! ELSE ..`) was re-emitted by EVERY witness kernel, while the interpreter's
//! enumeration emits it exactly once. States and verdict stayed exact (the
//! duplicates are byte-identical and collapse at admission), but the
//! `transitions` summary drifted: MCBakery 11150 (fused) vs 10658
//! (interpreter). The fix gates every NON-canonical witness kernel on a
//! participation flag (set only when its runtime witness-membership guard
//! actually passes), so witness-independent successors are emitted exactly
//! once — matching interpreter accounting without touching witness-scoped
//! emissions.
//!
//! These tests pin:
//! 1. MCBakery A/B exact: full gates + `TY_FLAT_WRITE_ADMIT=1` (the fused
//!    loop) vs no gates (interpreter) — identical
//!    states/initial/transitions, at the interpreter-reference values
//!    2303/1/10658.
//! 2. Witness-SCOPED duplicate successors still count once PER WITNESS in
//!    both engines (`\E i \in {1,2}: x' = 1` counts 2 per parent — the
//!    interpreter enumerates each witness), guarding against overshooting
//!    the fix into payload-level dedup.

mod common;

use std::path::{Path, PathBuf};
use tla_check::{CheckResult, Config, ModelChecker};
use tla_core::{lower, parse_to_syntax_tree, FileId, ModuleLoader};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Summary {
    states_found: usize,
    initial_states: usize,
    transitions: usize,
}

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .to_path_buf()
}

struct LoadedFixture {
    module: tla_core::ast::Module,
    checker_modules: Vec<tla_core::ast::Module>,
    config: Config,
}

/// Load a spec + cfg from `test_specs/`, resolving EXTENDS (MCBakery EXTENDS
/// Bakery). Mirrors the `trust_cg_state_graph_parity.rs` fixture loader.
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
        .unwrap_or_else(|error| panic!("failed to load EXTENDS for {spec_name}: {error}"));
    loader
        .load_instances(&module)
        .unwrap_or_else(|error| panic!("failed to load INSTANCEs for {spec_name}: {error}"));

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
        .unwrap_or_else(|error| panic!("failed to parse {cfg_name}: {error:?}"));
    if config.init.is_none() || config.next.is_none() {
        let resolved = tla_check::resolve_spec_from_config_with_extends(
            &config,
            &tree,
            &extended_syntax_trees,
        )
        .unwrap_or_else(|error| panic!("failed to resolve SPEC for {cfg_name}: {error}"));
        if config.init.is_none() {
            config.init = Some(resolved.init.clone());
        }
        if config.next.is_none() {
            config.next = Some(resolved.next.clone());
        }
    }
    config.auto_por = Some(false);

    LoadedFixture {
        module,
        checker_modules,
        config,
    }
}

/// The full WP-15/WP-16 campaign gate stack that flips MCBakery onto the
/// whole-state native fused BFS loop (flat_primary_safe under
/// `TY_FLAT_WRITE_ADMIT=1`).
fn fused_gate_guards() -> Vec<common::EnvVarGuard> {
    vec![
        common::EnvVarGuard::set("TY_TRUST_CG", Some("1")),
        common::EnvVarGuard::set("TY_HYBRID_FLAT_VIEW", Some("1")),
        common::EnvVarGuard::set("TY_TAGGED_SCALAR_UNION", Some("1")),
        common::EnvVarGuard::set("TY_SEQ_CAPACITY_PROOF", Some("1")),
        common::EnvVarGuard::set("TY_SCALAR_TUPLE_UNION", Some("1")),
        common::EnvVarGuard::set("TY_HYBRID_NATIVE", Some("1")),
        common::EnvVarGuard::set("TY_HYBRID_COMPOUND_READ", Some("1")),
        common::EnvVarGuard::set("TY_HYBRID_ENGINE_GAP", Some("1")),
        common::EnvVarGuard::set("TY_FLAT_WRITE_ADMIT", Some("1")),
        common::EnvVarGuard::set("TY_AUTO_POR", Some("0")),
        common::EnvVarGuard::remove("TY_NO_COMPILED_BFS"),
        common::EnvVarGuard::remove("TY_NO_FLAT_BFS"),
        common::EnvVarGuard::remove("TY_HYBRID_NATIVE_AUTHORITATIVE"),
        common::EnvVarGuard::remove("TY_COMPILED_BFS_INTERPRETER_CROSSCHECK"),
    ]
}

fn interpreter_guards() -> Vec<common::EnvVarGuard> {
    vec![
        common::EnvVarGuard::remove("TY_TRUST_CG"),
        common::EnvVarGuard::remove("TY_HYBRID_FLAT_VIEW"),
        common::EnvVarGuard::remove("TY_TAGGED_SCALAR_UNION"),
        common::EnvVarGuard::remove("TY_SEQ_CAPACITY_PROOF"),
        common::EnvVarGuard::remove("TY_SCALAR_TUPLE_UNION"),
        common::EnvVarGuard::remove("TY_HYBRID_NATIVE"),
        common::EnvVarGuard::remove("TY_HYBRID_COMPOUND_READ"),
        common::EnvVarGuard::remove("TY_HYBRID_ENGINE_GAP"),
        common::EnvVarGuard::remove("TY_FLAT_WRITE_ADMIT"),
        common::EnvVarGuard::set("TY_AUTO_POR", Some("0")),
    ]
}

fn run_fixture(fixture: &LoadedFixture) -> (Summary, Option<(usize, usize)>) {
    tla_eval::clear_for_test_reset();
    tla_trust_cg::compile::clear_jit_cache();
    let checker_modules = fixture.checker_modules.iter().collect::<Vec<_>>();
    let mut checker =
        ModelChecker::new_with_extends(&fixture.module, &checker_modules, &fixture.config);
    checker.set_store_states(false);
    let stats = match checker.check() {
        CheckResult::Success(stats) => stats,
        other => panic!("expected successful model check, got {other:?}"),
    };
    let coverage = checker.trust_cg_action_coverage_for_testing();
    (
        Summary {
            states_found: stats.states_found,
            initial_states: stats.initial_states,
            transitions: stats.transitions,
        },
        coverage,
    )
}

/// MCBakery A/B: the fused loop's summary must equal the interpreter's
/// exactly, at the interpreter-reference values (2303 states / 10658
/// transitions). Before the WP-16 witness-participation gate the fused arm
/// reported 11150 transitions (492 duplicate witness-independent ELSE-arm
/// emissions from `e2`/`w1` sibling witness kernels).
#[cfg_attr(test, ntest::timeout(300000))]
#[test]
fn fused_loop_transitions_match_interpreter_for_mcbakery() {
    let interpreter = {
        let _guards = interpreter_guards();
        let fixture = load_fixture("MCBakery.tla", "MCBakery.cfg");
        run_fixture(&fixture).0
    };
    assert_eq!(
        interpreter,
        Summary {
            states_found: 2303,
            initial_states: 1,
            transitions: 10658,
        },
        "interpreter reference summary drifted — investigate before touching the fused arm"
    );

    let (fused, coverage) = {
        let _guards = fused_gate_guards();
        let fixture = load_fixture("MCBakery.tla", "MCBakery.cfg");
        run_fixture(&fixture)
    };
    // The fused arm is only a meaningful differential when trust-codegen
    // compiled every action instance (the CompiledBfsLevel eligibility
    // precondition). If this stops holding the test would pass vacuously with
    // both arms on the interpreter — fail loudly instead.
    let (compiled, total) = coverage.expect("gated MCBakery run should record trust-cg coverage");
    assert_eq!(
        (compiled, total),
        (42, 42),
        "MCBakery no longer fully compiles under the campaign gates; the fused arm would not \
         exercise the native fused BFS loop"
    );
    assert_eq!(
        fused, interpreter,
        "fused-loop summary must match interpreter accounting exactly"
    );
}

const WITNESS_SCOPED_DUP_TLA: &str = r#"
---------------------------- MODULE WitnessScopedDup --------------------------
VARIABLE x
Init == x = 0
Next == \E i \in {1, 2} : x' = 1
=============================================================================
"#;

const WITNESS_SCOPED_DUP_CFG: &str = r#"
INIT Init
NEXT Next
CHECK_DEADLOCK FALSE
"#;

/// Witness-SCOPED byte-identical successors must keep counting once PER
/// WITNESS on the fused path: the interpreter enumerates `\E i \in {1,2}:
/// x' = 1` as two firings per parent (states 2, transitions 4), and the
/// per-witness kernels must do the same. This is the shape that forbids
/// fixing WP-16 by payload-level dedup.
#[cfg_attr(test, ntest::timeout(120000))]
#[test]
fn fused_loop_counts_witness_scoped_duplicates_like_interpreter() {
    let run = |guards: Vec<common::EnvVarGuard>| -> Summary {
        let _guards = guards;
        tla_eval::clear_for_test_reset();
        tla_trust_cg::compile::clear_jit_cache();
        tla_core::clear_global_name_interner();
        let module = common::parse_module(WITNESS_SCOPED_DUP_TLA);
        let config = Config::parse(WITNESS_SCOPED_DUP_CFG).expect("valid cfg");
        let mut checker = ModelChecker::new(&module, &config);
        checker.set_store_states(false);
        let stats = match checker.check() {
            CheckResult::Success(stats) => stats,
            other => panic!("expected successful model check, got {other:?}"),
        };
        Summary {
            states_found: stats.states_found,
            initial_states: stats.initial_states,
            transitions: stats.transitions,
        }
    };

    let interpreter = run(interpreter_guards());
    assert_eq!(
        interpreter,
        Summary {
            states_found: 2,
            initial_states: 1,
            transitions: 4,
        },
        "interpreter must count one transition per witness"
    );

    let fused = run(fused_gate_guards());
    assert_eq!(
        fused, interpreter,
        "fused path must keep per-witness transition accounting"
    );
}
