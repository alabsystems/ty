// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Bulk-output helpers for init constraint enumeration.

use rustc_hash::FxHashSet;
use std::sync::Arc;

use tla_core::ast::Expr;
use tla_core::Spanned;

use crate::error::EvalError;
use crate::eval::EvalCtx;
use crate::Value;

use super::{collect_state_var_refs, BulkConstraintEnumerationError};

/// A filter constraint with dependency info for early evaluation.
pub(super) struct FilterConstraint {
    /// Canonical AST filter expression evaluated once its dependencies are bound.
    pub(super) expr: Spanned<Expr>,
    /// Index at which this filter becomes applicable (max immediate dependency index).
    pub(super) trigger_idx: usize,
    /// True if this filter references any state var that is not an immediate variable.
    pub(super) requires_deferred: bool,
}

pub(super) fn plan_filter_constraints(
    ctx: &EvalCtx,
    vars: &[Arc<str>],
    immediate_vars: &[Arc<str>],
    filters: &[&Spanned<Expr>],
) -> Vec<FilterConstraint> {
    let immediate_positions: rustc_hash::FxHashMap<&str, usize> = immediate_vars
        .iter()
        .enumerate()
        .map(|(index, var)| (var.as_ref(), index))
        .collect();

    filters
        .iter()
        .map(|filter_expr| {
            let mut refs = FxHashSet::default();
            collect_state_var_refs(ctx, filter_expr, vars, &mut refs);

            // An empty syntactic ref set is not proof that the filter is constant:
            // opaque selectors/module refs such as Inv!P0 can hide state-var reads.
            let mut requires_deferred = immediate_vars.is_empty() || refs.is_empty();
            let mut trigger_idx = 0usize;
            for reference in refs {
                if let Some(&position) = immediate_positions.get(reference.as_ref()) {
                    trigger_idx = trigger_idx.max(position);
                } else {
                    requires_deferred = true;
                }
            }

            FilterConstraint {
                expr: (*filter_expr).clone(),
                trigger_idx,
                requires_deferred,
            }
        })
        .collect()
}

pub(crate) fn eval_filter_expr(ctx: &EvalCtx, expr: &Spanned<Expr>) -> Result<bool, EvalError> {
    match crate::eval::eval_entry(ctx, expr)? {
        Value::Bool(value) => Ok(value),
        other => Err(EvalError::type_error("BOOLEAN", &other, Some(expr.span))),
    }
}

/// Mutable output sink for bulk enumeration: storage, dedup set, filter, count, and index map.
pub(super) struct BulkEnumSink<'a, F, E> {
    pub(super) storage: &'a mut crate::arena::BulkStateStorage,
    pub(super) seen: &'a mut FxHashSet<u64>,
    pub(super) filter: &'a mut F,
    pub(super) generated_count: &'a mut usize,
    pub(super) added_count: &'a mut usize,
    pub(super) var_indices: rustc_hash::FxHashMap<&'a str, usize>,
    pub(super) _filter_error: std::marker::PhantomData<E>,
}

pub(super) fn emit_values_to_bulk<F, E>(
    ctx: &mut EvalCtx,
    values: &[Value],
    sink: &mut BulkEnumSink<'_, F, E>,
) -> Result<(), BulkConstraintEnumerationError<E>>
where
    F: FnMut(&[Value], &mut EvalCtx) -> Result<bool, E>,
{
    if !(sink.filter)(values, ctx).map_err(BulkConstraintEnumerationError::Filter)? {
        return Ok(());
    }

    *sink.generated_count += 1;
    let fingerprint = compute_values_fingerprint(values);
    if !sink.seen.insert(fingerprint) {
        return Ok(());
    }

    sink.storage.push_from_values(values);
    *sink.added_count += 1;
    Ok(())
}

/// Compute a fingerprint for a value slice (for deduplication).
pub(super) fn compute_values_fingerprint(values: &[Value]) -> u64 {
    use std::hash::{Hash, Hasher};

    let mut hasher = rustc_hash::FxHasher::default();
    for value in values {
        value.hash(&mut hasher);
    }
    hasher.finish()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use tla_core::ast::{Expr, ModuleTarget, Unit};
    use tla_core::{lower, parse_to_syntax_tree, FileId, Span};

    fn spanned(expr: Expr) -> Spanned<Expr> {
        Spanned::new(expr, Span::dummy())
    }

    #[test]
    fn plan_filter_constraints_defers_empty_ref_filters() {
        let mut ctx = EvalCtx::new();
        let vars: Vec<Arc<str>> = ["x", "y"].into_iter().map(Arc::from).collect();
        ctx.register_vars(vars.iter().cloned());

        let filter = spanned(Expr::ModuleRef(
            ModuleTarget::Named("Inv".to_string()),
            "P0".to_string(),
            Vec::new(),
        ));
        let filters = [&filter];

        let planned = plan_filter_constraints(&ctx, &vars, &vars, &filters);

        assert_eq!(planned.len(), 1);
        assert!(
            planned[0].requires_deferred,
            "empty dependency scans must be treated as unknown/deferred"
        );
    }

    #[test]
    fn opaque_module_ref_filter_waits_until_immediate_vars_are_bound() {
        let src = r#"
---- MODULE Test ----
VARIABLE x, y

Inv ==
  /\ P0:: y = 1
  /\ TRUE

Init ==
  /\ x \in {0, 1}
  /\ y \in {0, 1}
  /\ Inv!P0
====
"#;

        let tree = parse_to_syntax_tree(src);
        let lower_result = lower(FileId(0), &tree);
        assert!(
            lower_result.errors.is_empty(),
            "lower errors: {:?}",
            lower_result.errors
        );
        let module = lower_result.module.expect("module");

        let mut ctx = EvalCtx::new();
        ctx.load_module(&module);

        let mut vars = Vec::new();
        for unit in &module.units {
            if let Unit::Variable(var_names) = &unit.node {
                for var in var_names {
                    let name = Arc::from(var.node.as_str());
                    ctx.register_var(Arc::clone(&name));
                    vars.push(name);
                }
            }
        }

        let init_def = module
            .units
            .iter()
            .find_map(|unit| match &unit.node {
                Unit::Operator(def) if def.name.node == "Init" => Some(def),
                _ => None,
            })
            .expect("Init operator");
        let branches =
            crate::enumerate::extract_init_constraints(&ctx, &init_def.body, &vars, None)
                .expect("Init should extract");

        let mut storage = crate::arena::BulkStateStorage::new(ctx.var_registry().len(), 4);
        let count = super::super::enumerate_constraints_to_bulk(
            &mut ctx,
            &vars,
            &branches,
            &mut storage,
            |_values, _ctx| Ok(true),
        )
        .expect("bulk enumeration should not evaluate Inv!P0 before y is bound")
        .expect("bulk enumeration should handle the constraints");

        assert_eq!(count, 2);
        assert_eq!(storage.len(), 2);
        let mut states = (0..storage.len())
            .map(|index| storage.get_state(index as u32).to_vec())
            .collect::<Vec<_>>();
        states.sort();
        assert_eq!(
            states,
            vec![
                vec![Value::int(0), Value::int(1)],
                vec![Value::int(1), Value::int(1)],
            ]
        );
    }
}
