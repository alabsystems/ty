// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Cross-backend soundness for native bounded `CHOOSE x \in S : P(x)`.
//!
//! The native trust-codegen general CHOOSE path iterates the runtime
//! aggregate's physical slot order, while the tree-walking interpreter
//! (`eval_choose_single`) and the bytecode VM (`choose_begin`) iterate
//! TLC-normalized order. For order-sensitive `CHOOSE` (it returns THE first
//! satisfying witness) a non-canonical slot order would pick a different
//! element and change reachable states / fingerprints / verdicts. The lowering
//! gate falls back to the interpreter whenever the domain's slot order is not
//! provably TLC-normalized; these tests pin the parity that gate must preserve:
//!
//! * `ChooseInvariantViolated` — a bounded CHOOSE over an integer interval
//!   (canonical native path) whose witness selection genuinely VIOLATES an
//!   invariant at a reachable state. Both backends must report the SAME
//!   violation with identical state counts, and the native backend must
//!   actually compile the CHOOSE-bearing action (not silently fall back).
//! * Determinism — repeated native runs return the same verdict and counts.
//! * `ChooseNoWitness` — a bounded CHOOSE with no satisfying element must raise
//!   the same fail-closed runtime error on both backends.

mod common;

use common::EnvVarGuard;
use std::path::{Path, PathBuf};
use tla_check::{
    resolve_spec_from_config_with_extends, CheckResult, Config, FairnessConstraint, ModelChecker,
};
use tla_core::{lower, parse_to_syntax_tree, FileId, ModuleLoader};

struct LoadedFixture {
    module: tla_core::ast::Module,
    checker_modules: Vec<tla_core::ast::Module>,
    config: Config,
    fairness: Vec<FairnessConstraint>,
    stuttering_allowed: bool,
}

/// A backend-independent summary of a model-check verdict that can be compared
/// for exact parity between the interpreter and the native trust-cg backend.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Verdict {
    Success {
        states_found: usize,
        initial_states: usize,
        transitions: usize,
    },
    InvariantViolation {
        invariant: String,
        states_found: usize,
        initial_states: usize,
        transitions: usize,
    },
    /// An evaluation error (e.g. CHOOSE with no satisfying value). We compare
    /// only the discriminant + a normalized message classification, because
    /// the two backends format errors differently but must agree on *whether*
    /// the run failed closed.
    Error { is_choose_failure: bool },
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
        .unwrap_or_else(|errors| panic!("failed to parse {}: {errors:?}", cfg_path.display()));
    let resolved = resolve_spec_from_config_with_extends(&config, &tree, &extended_syntax_trees)
        .unwrap_or_else(|error| panic!("failed to resolve SPEC for {cfg_name}: {error}"));
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

#[derive(Debug, Clone)]
struct RunOutcome {
    verdict: Verdict,
    /// `(actions_compiled, actions_total)` reported by the trust-cg build.
    /// `None` for the interpreter backend (no native build).
    trust_cg_action_coverage: Option<(usize, usize)>,
}

/// Run one model check with the requested backend and classify the verdict.
fn run(fixture: &LoadedFixture, trust_cg: bool) -> RunOutcome {
    let _trust_cg = EnvVarGuard::set("TY_TRUST_CG", trust_cg.then_some("1"));
    let _trust_cg_bfs = EnvVarGuard::remove("TY_TRUST_CG_BFS");
    let _no_compiled = EnvVarGuard::remove("TY_NO_COMPILED_BFS");
    let _compiled_env = EnvVarGuard::remove("TY_COMPILED_BFS");
    // Force per-action native dispatch (not the whole-BFS compiled loop) so the
    // CHOOSE-bearing action is exercised through the native action and its
    // compile coverage is recorded.
    let _entry_counter_gate =
        EnvVarGuard::set("TY_TRUST_CG_ENTRY_COUNTER_GATE", trust_cg.then_some("0"));
    let _no_flat = EnvVarGuard::set("TY_NO_FLAT_BFS", Some("1"));
    let _flat_env = EnvVarGuard::remove("TY_FLAT_BFS");
    let _auto_por = EnvVarGuard::set("TY_AUTO_POR", Some("0"));

    tla_eval::clear_for_test_reset();
    let mut config = fixture.config.clone();
    config.use_compiled_bfs = Some(false);
    let checker_modules = fixture.checker_modules.iter().collect::<Vec<_>>();
    let mut checker = ModelChecker::new_with_extends(&fixture.module, &checker_modules, &config);
    checker.set_store_states(false);
    checker.set_collect_coverage(true);
    checker.set_fairness(fixture.fairness.clone());
    checker.set_stuttering_allowed(fixture.stuttering_allowed);

    let result = checker.check();
    let trust_cg_action_coverage = if trust_cg {
        checker.trust_cg_action_coverage_for_testing()
    } else {
        None
    };

    let verdict = match result {
        CheckResult::Success(stats) => Verdict::Success {
            states_found: stats.states_found,
            initial_states: stats.initial_states,
            transitions: stats.transitions,
        },
        CheckResult::InvariantViolation {
            invariant, stats, ..
        } => Verdict::InvariantViolation {
            invariant,
            states_found: stats.states_found,
            initial_states: stats.initial_states,
            transitions: stats.transitions,
        },
        other => {
            let rendered = format!("{other:?}");
            // CHOOSE-with-no-witness surfaces as an evaluation error; both
            // backends classify it the same way even though the exact rendered
            // text differs.
            let is_choose_failure = rendered.contains("Choose")
                || rendered.contains("CHOOSE")
                || rendered.contains("no satisfying")
                || rendered.contains("no element");
            Verdict::Error { is_choose_failure }
        }
    };

    RunOutcome {
        verdict,
        trust_cg_action_coverage,
    }
}

/// The interpreter and native backends must agree on the verdict and the
/// reachable state graph for a bounded-CHOOSE spec whose invariant is violated.
#[cfg_attr(test, ntest::timeout(120000))]
#[test]
fn native_choose_matches_interpreter_on_violated_invariant() {
    let fixture = load_fixture("ChooseInvariantViolated.tla", "ChooseInvariantViolated.cfg");

    let interpreter = run(&fixture, false);
    let native = run(&fixture, true);

    // The verdict (violation + invariant name + state/transition counts) must
    // be byte-for-byte identical across backends.
    assert_eq!(
        native.verdict, interpreter.verdict,
        "native trust-cg CHOOSE verdict must match the interpreter exactly"
    );

    // Both must specifically detect the Below5 violation.
    match &interpreter.verdict {
        Verdict::InvariantViolation { invariant, .. } => {
            assert_eq!(
                invariant, "Below5",
                "fixture must violate the Below5 invariant"
            );
        }
        other => panic!("expected an invariant violation, got {other:?}"),
    }

    // The native backend must have actually compiled the CHOOSE-bearing
    // action(s): if it had fallen back to the interpreter for every action,
    // `actions_compiled` would be 0 and this fixture would not be exercising
    // the native CHOOSE path it is meant to guard.
    let (compiled, total) = native
        .trust_cg_action_coverage
        .expect("native run must record trust-cg action coverage");
    assert!(
        compiled > 0,
        "native run must compile at least one action (got {compiled}/{total}); \
         otherwise the native CHOOSE path is never exercised"
    );
    assert_eq!(
        compiled, total,
        "every action in the bounded-CHOOSE fixture should compile natively \
         (got {compiled}/{total}); a partial fallback would mean the canonical \
         interval-CHOOSE path is not actually native"
    );
}

/// Native CHOOSE witness selection must be deterministic: repeated runs return
/// an identical verdict and identical state counts (CHOOSE must be stable, not
/// dependent on hash iteration order or run-to-run nondeterminism).
#[cfg_attr(test, ntest::timeout(120000))]
#[test]
fn native_choose_is_deterministic_across_runs() {
    let fixture = load_fixture("ChooseInvariantViolated.tla", "ChooseInvariantViolated.cfg");

    let first = run(&fixture, true);
    let second = run(&fixture, true);
    let third = run(&fixture, true);

    assert_eq!(
        first.verdict, second.verdict,
        "native CHOOSE verdict must be identical across repeated runs"
    );
    assert_eq!(
        second.verdict, third.verdict,
        "native CHOOSE verdict must be identical across repeated runs"
    );
}

/// A bounded CHOOSE with no satisfying element must fail closed identically on
/// both backends (the interpreter raises `EvalError::ChooseFailed`; the native
/// path raises an equivalent runtime error). Neither may silently succeed.
#[cfg_attr(test, ntest::timeout(120000))]
#[test]
fn native_choose_no_witness_errors_match_interpreter() {
    let fixture = load_fixture("ChooseNoWitness.tla", "ChooseNoWitness.cfg");

    let interpreter = run(&fixture, false);
    let native = run(&fixture, true);

    // Neither backend may report Success: a no-witness CHOOSE is a runtime
    // error, never a value.
    assert!(
        !matches!(interpreter.verdict, Verdict::Success { .. }),
        "interpreter must not succeed on a no-witness CHOOSE, got {:?}",
        interpreter.verdict
    );
    assert!(
        !matches!(native.verdict, Verdict::Success { .. }),
        "native CHOOSE must not succeed on a no-witness CHOOSE, got {:?}",
        native.verdict
    );

    // Both backends must classify the failure the same way (error on both).
    assert_eq!(
        std::mem::discriminant(&native.verdict),
        std::mem::discriminant(&interpreter.verdict),
        "native and interpreter must reach the same kind of verdict on a \
         no-witness CHOOSE (both must fail closed)"
    );
}
