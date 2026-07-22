// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Tests for `crate::checker_ops`.

use super::*;
use crate::eval::EvalCtx;
use crate::state::ArrayState;
use rustc_hash::FxHashMap;

mod implied_action_bytecode;
mod invariant_checks;
mod property_classify;
mod property_classify_instance;
mod property_tautology;
mod temporal_gaps;

/// Parse a TLA+ source and produce an EvalCtx + op_defs FxHashMap suitable
/// for `classify_property_safety_parts` and `check_property_tautologies`.
fn setup_for_classification(
    src: &str,
) -> (
    FxHashMap<String, tla_core::ast::OperatorDef>,
    EvalCtx,
    String,
) {
    let tree = tla_core::parse_to_syntax_tree(src);
    let lower_result = tla_core::lower(tla_core::FileId(0), &tree);
    let module = lower_result.module.unwrap();

    let mut ctx = EvalCtx::new();
    ctx.load_module(&module);

    let mut op_defs: FxHashMap<String, tla_core::ast::OperatorDef> = FxHashMap::default();
    for unit in &module.units {
        match &unit.node {
            tla_core::ast::Unit::Variable(var_names) => {
                for var in var_names {
                    ctx.register_var(Arc::from(var.node.as_str()));
                }
            }
            tla_core::ast::Unit::Operator(def) => {
                op_defs.insert(def.name.node.clone(), def.clone());
            }
            _ => {}
        }
    }

    let root_name = module.name.node.clone();
    (op_defs, ctx, root_name)
}

#[test]
fn fresh_semantic_context_clears_same_named_view_projection() {
    let first_source = r#"
---- MODULE FirstView ----
VARIABLE x, y
View == <<x>>
===="#;
    let second_source = r#"
---- MODULE SecondView ----
VARIABLE x, y
View == <<y>>
===="#;
    let state = ArrayState::from_values(vec![crate::Value::int(10), crate::Value::int(20)]);

    crate::clear_thread_local_eval_caches();
    let (_, mut first_ctx, _) = setup_for_classification(first_source);
    first_ctx.resolve_state_vars_in_loaded_ops();
    let first = compute_view_fingerprint_array(&mut first_ctx, &state, "View", 1)
        .expect("first VIEW must fingerprint");

    let (_, mut second_ctx, _) = setup_for_classification(second_source);
    second_ctx.resolve_state_vars_in_loaded_ops();
    crate::clear_thread_local_eval_caches();
    let second = compute_view_fingerprint_array(&mut second_ctx, &state, "View", 1)
        .expect("second VIEW must fingerprint");
    let expected = compute_view_fingerprint_from_projection(
        &state,
        &analyze_view_projection(&second_ctx, "View"),
    )
    .expect("second VIEW is a direct projection");

    assert_eq!(
        second, expected,
        "fresh input must use the second module's VIEW body"
    );
    assert_ne!(
        first, second,
        "x and y projections must remain distinguishable"
    );
}

#[test]
fn direct_model_checker_boundaries_clear_same_named_view_projection() {
    let first_source = r#"
---- MODULE FirstViewBoundary ----
VARIABLE x, y
View == <<x>>
===="#;
    let second_source = r#"
---- MODULE SecondViewBoundary ----
VARIABLE x, y
Init == /\ x = 10 /\ y = 20
Next == UNCHANGED <<x, y>>
View == <<y>>
===="#;
    let state = ArrayState::from_values(vec![crate::Value::int(10), crate::Value::int(20)]);

    crate::clear_thread_local_eval_caches();
    let (_, mut first_ctx, _) = setup_for_classification(first_source);
    first_ctx.resolve_state_vars_in_loaded_ops();
    let first = compute_view_fingerprint_array(&mut first_ctx, &state, "View", 1)
        .expect("first VIEW must fingerprint");

    let tree = tla_core::parse_to_syntax_tree(second_source);
    let second_module = tla_core::lower(tla_core::FileId(0), &tree)
        .module
        .expect("second module must lower");
    let mut config = crate::Config::default();
    config.init = Some("Init".to_string());
    config.next = Some("Next".to_string());
    config.view = Some("View".to_string());
    let mut checker = crate::ModelChecker::new(&second_module, &config);

    // The direct constructor is itself a fresh semantic-input boundary.
    let (_, mut second_ctx, _) = setup_for_classification(second_source);
    second_ctx.resolve_state_vars_in_loaded_ops();
    let second_after_construction =
        compute_view_fingerprint_array(&mut second_ctx, &state, "View", 1)
            .expect("second VIEW must fingerprint after construction");
    assert_ne!(first, second_after_construction);

    // Re-prime the stale projection after construction. A delayed public
    // execution must clear it again before evaluating the second module.
    crate::clear_thread_local_eval_caches();
    let _ = compute_view_fingerprint_array(&mut first_ctx, &state, "View", 1)
        .expect("first VIEW must re-prime the cache");
    let _ = checker.check();

    let (_, mut final_ctx, _) = setup_for_classification(second_source);
    final_ctx.resolve_state_vars_in_loaded_ops();
    let second_after_check = compute_view_fingerprint_array(&mut final_ctx, &state, "View", 1)
        .expect("second VIEW must fingerprint after check");
    assert_eq!(second_after_check, second_after_construction);
}
