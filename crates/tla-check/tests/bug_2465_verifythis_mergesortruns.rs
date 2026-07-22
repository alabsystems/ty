// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Regression test for #2465: VerifyThis `MergesortRuns` false positive.
//!
//! Root cause: zero-arg LET Tier 1.5 caching in `tla-eval` permanently keyed a
//! branch-sensitive LET body by only the first observed local dependency set.
//! In `MergeAcc`, helpers like `copy1` and `copy2` first execute on a branch
//! that reads only `t1` or `t2`, then later execute on branches that also read
//! `r1`, `r2`, `di1`, `di2`, `ri1`, and `ri2`. Reusing the under-keyed cache
//! entry produced a bogus merged output and violated `PermutationCorrect`.

use std::path::{Path, PathBuf};
use tla_check::{resolve_spec_from_config_with_extends, CheckResult, Config, ModelChecker};
use tla_core::{lower, parse_to_syntax_tree, FileId, ModuleLoader};
use tla_value::Value;

/// Directory holding the VerifyThis practice benchmarks (MergesortRuns.tla +
/// .cfg). Set `TY_VERIFYTHIS_PRACTICE_DIR` to point at a local checkout; when
/// unset the test skips.
fn verifythis_practice_dir() -> Option<PathBuf> {
    let path = std::env::var_os("TY_VERIFYTHIS_PRACTICE_DIR")?;
    let path = PathBuf::from(path);
    path.exists().then_some(path)
}

fn check_mergesort_runs_spec(spec_path: &Path, cfg_path: &Path) -> CheckResult {
    let spec_source = std::fs::read_to_string(spec_path)
        .unwrap_or_else(|e| panic!("Failed to read {}: {}", spec_path.display(), e));
    let tree = parse_to_syntax_tree(&spec_source);
    let mut module = lower(FileId(0), &tree)
        .module
        .expect("MergesortRuns should lower successfully");
    tla_core::compute_is_recursive(&mut module);

    let mut loader = ModuleLoader::new(spec_path);
    loader.seed_from_syntax_tree(&tree, spec_path);
    loader
        .load_extends(&module)
        .expect("MergesortRuns EXTENDS dependencies should load");
    loader
        .load_instances(&module)
        .expect("MergesortRuns INSTANCE dependencies should load");

    let (extended_modules_for_resolve, instanced_modules_for_resolve) =
        loader.modules_for_semantic_resolution(&module);
    let checker_modules = loader.modules_for_model_checking(&module);
    let spec_scope_module_names: Vec<&str> = extended_modules_for_resolve
        .iter()
        .chain(instanced_modules_for_resolve.iter())
        .map(|m| m.name.node.as_str())
        .collect();
    let extended_syntax_trees: Vec<_> = spec_scope_module_names
        .iter()
        .filter_map(|name| loader.get(name).map(|loaded| &loaded.syntax_tree))
        .collect();

    let config_source = std::fs::read_to_string(cfg_path)
        .unwrap_or_else(|e| panic!("Failed to read {}: {}", cfg_path.display(), e));
    let mut config = Config::parse(&config_source).unwrap_or_else(|errors| {
        panic!(
            "Failed to parse {}: {} error(s)",
            cfg_path.display(),
            errors.len()
        )
    });
    let resolved = resolve_spec_from_config_with_extends(&config, &tree, &extended_syntax_trees)
        .expect("MergesortRuns SPECIFICATION should resolve across extended modules");
    if config.init.is_none() {
        config.init = Some(resolved.init.clone());
    }
    if config.next.is_none() {
        config.next = Some(resolved.next.clone());
    }
    config.normalize_resolved_specification();

    let mut checker = ModelChecker::new_with_extends(&module, &checker_modules, &config);
    checker.set_store_states(true);
    checker.set_fairness(resolved.fairness);
    checker.set_stuttering_allowed(resolved.stuttering_allowed);
    checker.check()
}

fn eval_with_ops(defs: &str, expr: &str) -> Value {
    tla_eval::clear_for_test_reset();

    let module_src = format!(
        "---- MODULE Test ----\n\n{}\n\nOp == {}\n\n====",
        defs, expr
    );
    let tree = parse_to_syntax_tree(&module_src);
    let lower_result = lower(FileId(0), &tree);
    assert!(
        lower_result.errors.is_empty(),
        "inline #2465 fixture should lower without errors: {:?}",
        lower_result.errors
    );
    let module = lower_result
        .module
        .expect("inline #2465 fixture should produce a module");

    let mut ctx = tla_eval::EvalCtx::new();
    ctx.load_module(&module);
    ctx.eval_op("Op")
        .expect("inline #2465 fixture should evaluate")
}

#[cfg_attr(test, ntest::timeout(60000))]
#[test]
fn bug_2465_branch_sensitive_let_cache_has_repo_local_coverage() {
    let result = eval_with_ops(
        "F(flag, x) == LET y == IF flag THEN x ELSE 0 IN y",
        "<<F(FALSE, 1), F(TRUE, 2), F(TRUE, 3)>>",
    );
    let expected = Value::Tuple(vec![Value::int(0), Value::int(2), Value::int(3)].into());

    assert_eq!(
        result, expected,
        "Bug #2465 regression: branch-sensitive LET cache must distinguish TRUE-branch results by x"
    );
    tla_eval::clear_for_test_reset();
}

#[cfg_attr(test, ntest::timeout(60000))]
#[test]
fn bug_2465_mergesortruns_real_spec_passes_with_tlc_state_parity_when_corpus_available() {
    let Some(practice_dir) = verifythis_practice_dir() else {
        eprintln!(
            "Skipping external VerifyThis MergesortRuns regression; set \
             TY_VERIFYTHIS_PRACTICE_DIR to the VerifyThis practice corpus to run it."
        );
        return;
    };

    let spec_dir = practice_dir.join("2022-c2-mergesort-runs");
    let spec_path = spec_dir.join("MergesortRuns.tla");
    let cfg_path = spec_dir.join("MergesortRuns.cfg");

    match check_mergesort_runs_spec(&spec_path, &cfg_path) {
        CheckResult::Success(stats) => {
            assert_eq!(
                stats.states_found, 1_092,
                "Bug #2465 regression: MergesortRuns should match TLC's 1,092 states, got {}",
                stats.states_found
            );
        }
        other => panic!(
            "Bug #2465 regression: MergesortRuns should pass its VerifyThis safety invariants, got {other:?}"
        ),
    }
}
