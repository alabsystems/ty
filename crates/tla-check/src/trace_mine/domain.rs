// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Per-variable domain mining: the set of values each variable was observed
//! to take, as a TLA+ set expression.

use std::collections::BTreeMap;

use crate::json_output::JsonValue;

use super::values::{as_int, render_value, value_key, RenderCtx};
use super::{MineError, MineOptions, MiningTrace, VarDomain};

/// Collect the distinct observed values of `var` across the whole corpus,
/// keyed deterministically.
pub(crate) fn observed_values<'a>(
    traces: &'a [MiningTrace],
    var: &str,
) -> BTreeMap<String, &'a JsonValue> {
    let mut values = BTreeMap::new();
    for trace in traces {
        for step in &trace.steps {
            if let Some(value) = step.state.get(var) {
                values.entry(value_key(value)).or_insert(value);
            }
        }
    }
    values
}

/// Mine a [`VarDomain`] per variable.
pub(crate) fn mine_domains(
    traces: &[MiningTrace],
    variables: &[String],
    options: &MineOptions,
    render: &mut RenderCtx,
    notes: &mut Vec<String>,
) -> Result<Vec<VarDomain>, MineError> {
    let mut domains = Vec::with_capacity(variables.len());
    for var in variables {
        let values = observed_values(traces, var);
        debug_assert!(!values.is_empty(), "mined variables are observed somewhere");

        let ints: Option<Vec<i64>> = values.values().map(|v| as_int(v)).collect();
        let (expr, description) = match &ints {
            Some(ints) => {
                render.needs_integers = true;
                let mut sorted = ints.clone();
                sorted.sort_unstable();
                let (lo, hi) = (sorted[0], sorted[sorted.len() - 1]);
                let contiguous = (hi - lo) as usize + 1 == sorted.len();
                if contiguous && sorted.len() > 1 {
                    (
                        format!("{lo}..{hi}"),
                        format!("{} contiguous int values {lo}..{hi}", sorted.len()),
                    )
                } else if sorted.len() <= options.max_domain_enum {
                    let rendered: Vec<String> = sorted
                        .iter()
                        .map(std::string::ToString::to_string)
                        .collect();
                    (
                        format!("{{{}}}", rendered.join(", ")),
                        format!("{} int values in {lo}..{hi}", sorted.len()),
                    )
                } else {
                    notes.push(format!(
                        "domain of {var:?} over-approximated to {lo}..{hi} \
                         ({} distinct observed values)",
                        sorted.len()
                    ));
                    (
                        format!("{lo}..{hi}"),
                        format!(
                            "{} int values, over-approximated to range {lo}..{hi}",
                            sorted.len()
                        ),
                    )
                }
            }
            None => {
                let mut rendered = values
                    .values()
                    .map(|v| render_value(v, var, render))
                    .collect::<Result<Vec<_>, _>>()?;
                rendered.sort();
                rendered.dedup();
                if rendered.len() > options.max_domain_enum {
                    notes.push(format!(
                        "domain of {var:?} enumerates {} distinct values",
                        rendered.len()
                    ));
                }
                (
                    format!("{{{}}}", rendered.join(", ")),
                    format!("{} distinct values (enumerated)", rendered.len()),
                )
            }
        };

        let (min_int, max_int) = match &ints {
            Some(ints) => (ints.iter().copied().min(), ints.iter().copied().max()),
            None => (None, None),
        };

        domains.push(VarDomain {
            var: var.clone(),
            op_name: domain_op_name(var, variables),
            expr,
            description,
            all_int: ints.is_some(),
            min_int,
            max_int,
            value_count: values.len(),
        });
    }
    Ok(domains)
}

/// Name for the domain operator of `var`, avoiding variable-name collisions.
fn domain_op_name(var: &str, variables: &[String]) -> String {
    let mut name = format!("{var}Domain");
    while variables.iter().any(|v| v == &name) {
        name.push('_');
    }
    name
}
