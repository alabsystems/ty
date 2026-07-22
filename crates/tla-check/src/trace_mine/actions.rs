// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Action inference: cluster observed step pairs (by action label when
//! present, else by the set of changed variables) and mine each cluster's
//! update patterns and guards.

use std::collections::{BTreeMap, BTreeSet, HashMap};

use crate::json_output::JsonValue;

use super::values::{as_int, render_value, sanitize_identifier, value_key, RenderCtx};
use super::{
    ActionUpdate, ClusterKind, MineError, MinedAction, MiningTrace, UpdatePattern, VarDomain,
};

/// One observed transition: a (pre, post) state pair.
struct Pair<'a> {
    pre: &'a HashMap<String, JsonValue>,
    post: &'a HashMap<String, JsonValue>,
}

/// Deterministic cluster key: labeled clusters sort before changed-set ones.
type ClusterKey = (u8, String);

/// Mine candidate actions from all consecutive step pairs of the corpus.
pub(crate) fn mine_actions(
    traces: &[MiningTrace],
    variables: &[String],
    domains: &[VarDomain],
    render: &mut RenderCtx,
    notes: &mut Vec<String>,
) -> Result<Vec<MinedAction>, MineError> {
    // --- Cluster the step pairs ---
    let mut clusters: BTreeMap<ClusterKey, (ClusterKind, Vec<Pair<'_>>)> = BTreeMap::new();
    for trace in traces {
        for window in trace.steps.windows(2) {
            let pair = Pair {
                pre: &window[0].state,
                post: &window[1].state,
            };
            let (key, kind) = match window[1].action.as_ref() {
                Some(label) => (
                    (0u8, label.name.clone()),
                    ClusterKind::Label(label.name.clone()),
                ),
                None => {
                    let changed = changed_vars(&pair, variables);
                    ((1u8, changed.join(",")), ClusterKind::ChangedVars(changed))
                }
            };
            clusters
                .entry(key)
                .or_insert_with(|| (kind, Vec::new()))
                .1
                .push(pair);
        }
    }

    if clusters.is_empty() {
        notes.push(
            "corpus has no step pairs (single-state traces only); Next falls back to stuttering"
                .to_string(),
        );
    }

    // Names already claimed by variables and generated operators.
    let mut used: BTreeSet<String> = variables.iter().cloned().collect();
    used.extend(domains.iter().map(|d| d.op_name.clone()));
    used.extend(["Init".to_string(), "Next".to_string(), "vars".to_string()]);
    for var in variables {
        used.insert(format!("TypeOK_{var}"));
        used.insert(format!("{var}_Monotone"));
        for other in variables {
            used.insert(format!("Rel_{var}_{other}"));
        }
    }

    // --- Mine each cluster ---
    let mut actions = Vec::with_capacity(clusters.len());
    for (kind, pairs) in clusters.into_values() {
        let base = match &kind {
            ClusterKind::Label(label) => sanitize_identifier(label),
            ClusterKind::ChangedVars(vars) if vars.is_empty() => "Stutter".to_string(),
            ClusterKind::ChangedVars(vars) => format!("Change_{}", vars.join("_")),
        };
        let name = fresh_name(base, &mut used);
        actions.push(mine_cluster(
            name, kind, &pairs, variables, domains, render, notes,
        )?);
    }
    Ok(actions)
}

/// Variables observed on both sides of the pair with differing values.
fn changed_vars(pair: &Pair<'_>, variables: &[String]) -> Vec<String> {
    let mut changed: Vec<String> = variables
        .iter()
        .filter(|var| match (pair.pre.get(*var), pair.post.get(*var)) {
            (Some(pre), Some(post)) => value_key(pre) != value_key(post),
            _ => false,
        })
        .cloned()
        .collect();
    changed.sort();
    changed
}

/// Mine one cluster into a candidate action.
fn mine_cluster(
    name: String,
    cluster: ClusterKind,
    pairs: &[Pair<'_>],
    variables: &[String],
    domains: &[VarDomain],
    render: &mut RenderCtx,
    notes: &mut Vec<String>,
) -> Result<MinedAction, MineError> {
    let mut guards = Vec::new();
    let mut updates = Vec::new();
    let mut unchanged = Vec::new();

    for var in variables {
        let domain = domains
            .iter()
            .find(|d| &d.var == var)
            .expect("every mined variable has a domain");

        // Guard evidence: pre-state values of this variable across the cluster.
        let pre_values: Vec<&JsonValue> = pairs.iter().filter_map(|p| p.pre.get(var)).collect();
        if let Some(guard) = mine_guard(var, &pre_values, domain, render)? {
            guards.push(guard);
        }

        // Update evidence: pairs observing the variable on both sides.
        let evidence: Vec<(&JsonValue, &JsonValue)> = pairs
            .iter()
            .filter_map(|p| Some((p.pre.get(var)?, p.post.get(var)?)))
            .collect();

        if evidence.is_empty() {
            // Never observed across a pair in this cluster: weakest enumerable
            // update — havoc within the variable's whole observed domain.
            notes.push(format!(
                "action {name}: no pair evidence for {var:?}; havocked within its domain"
            ));
            updates.push(ActionUpdate {
                var: var.clone(),
                conjuncts: vec![format!("{var}' \\in {}", domain.op_name)],
                pattern: UpdatePattern::Havoc,
            });
            continue;
        }

        if evidence
            .iter()
            .all(|(pre, post)| value_key(pre) == value_key(post))
        {
            unchanged.push(var.clone());
            continue;
        }

        updates.push(mine_update(var, &evidence, domain, render)?);
    }

    Ok(MinedAction {
        name,
        cluster,
        instances: pairs.len(),
        guards,
        updates,
        unchanged,
    })
}

/// Mine the update pattern for one changed variable, trying (in order):
/// constant delta, constant assignment, monotone increase, havoc within the
/// observed post-value set.
fn mine_update(
    var: &str,
    evidence: &[(&JsonValue, &JsonValue)],
    domain: &VarDomain,
    render: &mut RenderCtx,
) -> Result<ActionUpdate, MineError> {
    let ints: Option<Vec<(i64, i64)>> = evidence
        .iter()
        .map(|(pre, post)| Some((as_int(pre)?, as_int(post)?)))
        .collect();

    // 1. Constant delta: x' = x + k.
    if let Some(pairs) = &ints {
        let deltas: BTreeSet<i64> = pairs.iter().map(|(pre, post)| post - pre).collect();
        if deltas.len() == 1 {
            let k = *deltas.first().expect("non-empty");
            debug_assert_ne!(k, 0, "all-equal evidence is handled as UNCHANGED");
            render.needs_integers = true;
            let conjunct = if k >= 0 {
                format!("{var}' = {var} + {k}")
            } else {
                format!("{var}' = {var} - {}", -k)
            };
            return Ok(ActionUpdate {
                var: var.to_string(),
                conjuncts: vec![conjunct],
                pattern: UpdatePattern::ConstDelta(k),
            });
        }
    }

    // 2. Constant assignment: x' = c.
    let post_keys: BTreeSet<String> = evidence.iter().map(|(_, post)| value_key(post)).collect();
    if post_keys.len() == 1 {
        let lit = render_value(evidence[0].1, var, render)?;
        return Ok(ActionUpdate {
            var: var.to_string(),
            conjuncts: vec![format!("{var}' = {lit}")],
            pattern: UpdatePattern::ConstAssign(lit),
        });
    }

    // 3. Monotone increase (strict, then non-decreasing), bounded by the
    //    domain so the action stays enumerable.
    if let Some(pairs) = &ints {
        render.needs_integers = true;
        if pairs.iter().all(|(pre, post)| post > pre) {
            return Ok(ActionUpdate {
                var: var.to_string(),
                conjuncts: vec![
                    format!("{var}' \\in {}", domain.op_name),
                    format!("{var}' > {var}"),
                ],
                pattern: UpdatePattern::MonotoneStrict,
            });
        }
        if pairs.iter().all(|(pre, post)| post >= pre) {
            return Ok(ActionUpdate {
                var: var.to_string(),
                conjuncts: vec![
                    format!("{var}' \\in {}", domain.op_name),
                    format!("{var}' >= {var}"),
                ],
                pattern: UpdatePattern::MonotoneNonDec,
            });
        }
    }

    // 4. Havoc within the observed post-value set of this cluster.
    let post_values: Vec<&JsonValue> = evidence.iter().map(|(_, post)| *post).collect();
    let set_expr = render_set_expr(var, &post_values, render)?;
    Ok(ActionUpdate {
        var: var.to_string(),
        conjuncts: vec![format!("{var}' \\in {set_expr}")],
        pattern: UpdatePattern::Havoc,
    })
}

/// Mine a guard for one variable from the cluster's pre-state values:
/// equality with a constant, else informative int bounds relative to the
/// variable's whole observed domain.
fn mine_guard(
    var: &str,
    pre_values: &[&JsonValue],
    domain: &VarDomain,
    render: &mut RenderCtx,
) -> Result<Option<String>, MineError> {
    if pre_values.is_empty() {
        return Ok(None);
    }

    let keys: BTreeSet<String> = pre_values.iter().map(|v| value_key(v)).collect();
    if keys.len() == 1 {
        let lit = render_value(pre_values[0], var, render)?;
        return Ok(Some(format!("{var} = {lit}")));
    }

    let ints: Option<Vec<i64>> = pre_values.iter().map(|v| as_int(v)).collect();
    if let Some(ints) = ints {
        let cluster_min = ints.iter().copied().min().expect("non-empty");
        let cluster_max = ints.iter().copied().max().expect("non-empty");
        let mut bounds = Vec::new();
        if domain.min_int.is_some_and(|gmin| cluster_min > gmin) {
            bounds.push(format!("{var} >= {cluster_min}"));
        }
        if domain.max_int.is_some_and(|gmax| cluster_max < gmax) {
            bounds.push(format!("{var} <= {cluster_max}"));
        }
        if !bounds.is_empty() {
            render.needs_integers = true;
            return Ok(Some(bounds.join(" /\\ ")));
        }
    }
    Ok(None)
}

/// Render a set of observed values as a TLA+ set expression (contiguous int
/// sets become ranges).
fn render_set_expr(
    var: &str,
    values: &[&JsonValue],
    render: &mut RenderCtx,
) -> Result<String, MineError> {
    let ints: Option<BTreeSet<i64>> = values.iter().map(|v| as_int(v)).collect();
    if let Some(ints) = ints {
        render.needs_integers = true;
        let (lo, hi) = (
            *ints.first().expect("non-empty"),
            *ints.last().expect("non-empty"),
        );
        if ints.len() > 1 && (hi - lo) as usize + 1 == ints.len() {
            return Ok(format!("{lo}..{hi}"));
        }
        let rendered: Vec<String> = ints.iter().map(std::string::ToString::to_string).collect();
        return Ok(format!("{{{}}}", rendered.join(", ")));
    }
    let mut rendered = values
        .iter()
        .map(|v| render_value(v, var, render))
        .collect::<Result<Vec<_>, _>>()?;
    rendered.sort();
    rendered.dedup();
    Ok(format!("{{{}}}", rendered.join(", ")))
}

/// Return `base` if unused, else `base_2`, `base_3`, ... Claims the result.
fn fresh_name(base: String, used: &mut BTreeSet<String>) -> String {
    if used.insert(base.clone()) {
        return base;
    }
    let mut n = 2usize;
    loop {
        let candidate = format!("{base}_{n}");
        if used.insert(candidate.clone()) {
            return candidate;
        }
        n += 1;
    }
}
