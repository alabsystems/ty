// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Direct TIR tree-walker tests for two-element tuple membership.

use crate::error::EvalError;
use crate::tir::eval_tir;
use crate::{EvalCtx, Value};
use tla_core::{FileId, Span, Spanned};
use tla_tir::{TirExpr, TirType};

fn constant(value: Value) -> Spanned<TirExpr> {
    Spanned::dummy(TirExpr::Const {
        value,
        ty: TirType::Dyn,
    })
}

fn tuple2_membership(second: i64, set: Value, span: Span) -> Spanned<TirExpr> {
    Spanned::new(
        TirExpr::In {
            elem: Box::new(Spanned::dummy(TirExpr::Tuple(vec![
                constant(Value::SmallInt(1)),
                constant(Value::SmallInt(second)),
            ]))),
            set: Box::new(constant(set)),
        },
        span,
    )
}

#[test]
fn test_tir_tree_tuple2_membership_matches_sequence_representation() {
    let ctx = EvalCtx::new();
    let set = Value::set([Value::seq([Value::SmallInt(1), Value::SmallInt(2)])]);

    let hit = tuple2_membership(2, set.clone(), Span::dummy());
    assert_eq!(eval_tir(&ctx, &hit).unwrap(), Value::Bool(true));

    let miss = tuple2_membership(3, set, Span::dummy());
    assert_eq!(eval_tir(&ctx, &miss).unwrap(), Value::Bool(false));
}

#[test]
fn test_tir_tree_tuple2_membership_fallback_keeps_outer_span() {
    let ctx = EvalCtx::new();
    let membership_span = Span::new(FileId(7), 11, 29);
    let expr = tuple2_membership(2, Value::SmallInt(7), membership_span);

    let error = eval_tir(&ctx, &expr).unwrap_err();
    assert!(
        matches!(
            error,
            EvalError::TypeError {
                expected: "Set",
                span: Some(span),
                ..
            } if span == membership_span
        ),
        "tuple2 fallback must retain the outer membership span, got {error:?}"
    );
}
