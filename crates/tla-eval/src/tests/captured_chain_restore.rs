// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

use super::*;
use crate::eval_cache_lifecycle::force_lazy_thunk_if_needed;
use crate::helpers::apply_closure_with_values;
use crate::helpers::function_values::apply_resolved_func_value;
use crate::value::{
    CapturedChain, ClosureValue, LazyDomain, LazyFuncCaptures, LazyFuncValue, SetPredCaptures,
    SetPredValue,
};
use std::sync::Arc;
use tla_core::ast::{BoundVar, Expr};
use tla_core::kani_types::HashMap;
use tla_core::{Span, Spanned};
use tla_value::Rp;

#[derive(Clone)]
struct MockCapturedChain {
    locals: Vec<(Arc<str>, Value)>,
}

impl CapturedChain for MockCapturedChain {
    fn clone_box(&self) -> Box<dyn CapturedChain> {
        Box::new(self.clone())
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn materialize_locals(&self, env: &mut HashMap<Arc<str>, Value>) {
        for (name, value) in &self.locals {
            env.insert(Arc::clone(name), value.clone());
        }
    }
}

fn ident_expr(name: &str) -> Spanned<Expr> {
    Spanned::new(
        Expr::Ident(name.to_string(), tla_core::name_intern::NameId::INVALID),
        Span::dummy(),
    )
}

fn simple_bound(name: &str) -> BoundVar {
    BoundVar {
        name: Spanned::new(name.to_string(), Span::dummy()),
        domain: None,
        pattern: None,
    }
}

fn assert_binding_chain_mismatch(err: EvalError, site: &str) {
    match err {
        EvalError::Internal { message, .. } => {
            assert!(
                message.contains(site),
                "expected mismatch message to name {site}, got: {message}"
            );
            assert!(
                message.contains("expected BindingChain"),
                "expected BindingChain invariant message, got: {message}"
            );
        }
        other => panic!("expected internal captured-chain mismatch error, got: {other:?}"),
    }
}

fn assert_captured_binding(
    captured: Option<&dyn CapturedChain>,
    name: &str,
    expected: &Value,
    constructor: &str,
) {
    let captured = captured.unwrap_or_else(|| panic!("{constructor} must capture a binding chain"));
    let chain = captured
        .as_any()
        .downcast_ref::<BindingChain>()
        .unwrap_or_else(|| panic!("{constructor} capture must remain a BindingChain"));
    assert_eq!(
        chain.lookup_by_name(name).as_ref(),
        Some(expected),
        "{constructor} must preserve arena-backed locals after arena reuse"
    );
}

/// A durable capture must promote the arena segment at the initial trait-object
/// boundary. `Box::new(chain.clone()) as Box<dyn CapturedChain>` does not call
/// `CapturedChain::clone_box`; it leaves the original trait object pointing at
/// the arena node, which the next state deterministically overwrites at offset
/// zero. Exercise every production constructor that owns such a capture.
#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_production_captures_survive_eval_arena_reuse() {
    use crate::eval_arena::{init_thread_arena, ArenaStateGuard};

    init_thread_arena();
    let name = "test_durable_arena_capture_x";
    let name_id = tla_core::name_intern::intern_name(name);
    let expected = Value::Bool(true);

    let (closure, arc_closure, lazy_func, set_pred) = {
        let _guard = ArenaStateGuard::new();
        // This is the first arena allocation in the state, so the replacement
        // below is guaranteed to reuse the same address after reset.
        let ctx = EvalCtx::new().into_bind_by_id(name_id, expected.clone());

        let closure = ctx.create_closure(Vec::new(), ident_expr(name), None);
        let arc_closure = ctx.create_closure_arc(Vec::new(), Arc::new(ident_expr(name)), None);
        let lazy_func = build_lazy_func_from_ctx(
            &ctx,
            None,
            LazyDomain::General(Box::new(Value::set([Value::int(0)]))),
            &[simple_bound("arg")],
            ident_expr(name),
        );
        let set_pred = eval_str_with_ctx("{ y \\in SUBSET (1..9) : x }", &ctx)
            .expect("large SUBSET filter should construct a lazy SetPredValue");
        assert!(
            matches!(&set_pred, Value::SetPred(_)),
            "expected lazy SetPredValue, got {set_pred:?}"
        );

        (closure, arc_closure, lazy_func, set_pred)
    };

    {
        let _guard = ArenaStateGuard::new();
        let _replacement =
            BindingChain::empty().cons(name_id, BindingValue::eager(Value::Bool(false)));

        assert_captured_binding(closure.captured_chain(), name, &expected, "create_closure");
        assert_captured_binding(
            arc_closure.captured_chain(),
            name,
            &expected,
            "create_closure_arc",
        );
        assert_captured_binding(
            lazy_func.captured_chain(),
            name,
            &expected,
            "build_lazy_func_from_ctx",
        );
        let Value::SetPred(set_pred) = &set_pred else {
            unreachable!("SetPredValue variant checked at construction")
        };
        assert_captured_binding(
            set_pred.captured_chain(),
            name,
            &expected,
            "lazy SetPredValue",
        );
    }
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_closure_restore_rejects_non_binding_chain_capture() {
    let ctx = EvalCtx::new();
    let closure = ClosureValue::new(
        vec!["y".to_string()],
        ident_expr("x"),
        Arc::new(HashMap::new()),
        None,
    )
    .with_captured_chain(
        Box::new(MockCapturedChain {
            locals: vec![(Arc::from("x"), Value::int(42))],
        }),
        1,
    );

    let err = apply_closure_with_values(&ctx, &closure, &[Value::int(7)], None)
        .expect_err("closure restore should reject non-BindingChain captures");
    assert_binding_chain_mismatch(err, "build_closure_ctx");
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_lazy_func_restore_rejects_non_binding_chain_capture() {
    let ctx = EvalCtx::new();
    let lazy_func = tla_value::Rp::new(LazyFuncValue::new(
        None,
        LazyDomain::General(Box::new(Value::set([Value::int(1)]))),
        simple_bound("y"),
        ident_expr("x"),
        LazyFuncCaptures::new(Arc::new(HashMap::new()), None, None, None).with_captured_chain(
            Box::new(MockCapturedChain {
                locals: vec![(Arc::from("x"), Value::int(42))],
            }),
            1,
        ),
    ));

    let err = apply_resolved_func_value(
        &ctx,
        &Value::LazyFunc(Rp::clone(&lazy_func)),
        Value::int(1),
        None,
        None,
        None,
    )
    .expect_err("lazy-function restore should reject non-BindingChain captures");
    assert_binding_chain_mismatch(err, "build_lazy_func_ctx");
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_setpred_restore_rejects_non_binding_chain_capture() {
    let ctx = EvalCtx::new();
    let captures = SetPredCaptures::new(Arc::new(HashMap::new()), None, None).with_captured_chain(
        Box::new(MockCapturedChain {
            locals: vec![(Arc::from("x"), Value::Bool(true))],
        }),
        1,
    );
    let spv = SetPredValue::new_with_captures(
        Value::set([Value::int(1)]),
        simple_bound("y"),
        ident_expr("x"),
        captures,
    );

    let err = check_set_pred_membership(&ctx, &Value::int(1), &spv, None)
        .expect_err("set-predicate restore should reject non-BindingChain captures");
    assert_binding_chain_mismatch(err, "restore_setpred_ctx");
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_lazy_thunk_restore_rejects_non_binding_chain_capture() {
    let ctx = EvalCtx::new();
    let thunk = Value::Closure(Rp::new(
        ClosureValue::new(vec![], ident_expr("x"), Arc::new(HashMap::new()), None)
            .with_captured_chain(
                Box::new(MockCapturedChain {
                    locals: vec![(Arc::from("x"), Value::int(42))],
                }),
                1,
            ),
    ));

    let err = force_lazy_thunk_if_needed(&ctx, thunk)
        .expect_err("thunk forcing should reject non-BindingChain captures");
    assert_binding_chain_mismatch(err, "force_lazy_thunk_if_needed");
}
