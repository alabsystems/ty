// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

use super::*;
use crate::nodes::*;
use crate::types::TirType;
use tla_core::name_intern::intern_name;
use tla_core::Span;

mod const_fold;
mod control_and_calls;
mod cross_module;
mod except_at_free;
mod expressions;
mod fold_builtins;
mod lambda;
mod let_guard_hoist;
mod quantifiers;
mod round_apply;
mod round_step_eq;

fn spanned(expr: TirExpr) -> Spanned<TirExpr> {
    Spanned {
        node: expr,
        span: Span::default(),
    }
}

fn ident_name(name: &str) -> TirNameRef {
    TirNameRef {
        name: name.to_string(),
        name_id: tla_core::NameId(0),
        kind: TirNameKind::Ident,
        ty: TirType::Dyn,
    }
}

fn field_name(name: &str) -> TirFieldName {
    TirFieldName {
        name: name.to_string(),
        field_id: intern_name(name),
    }
}
