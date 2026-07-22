// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Candidate invariants beyond domains: pairwise linear relations across all
//! observed states, and per-variable monotonicity properties.

use std::collections::BTreeSet;

use super::values::as_int;
use super::{InvariantKind, MineOptions, MinedInvariant, MinedProperty, MiningTrace};

/// Mine pairwise linear relation candidates (`x = y + k`, `x <= y`) that hold
/// in every observed state where both variables are integers.
pub(crate) fn mine_relations(
    traces: &[MiningTrace],
    variables: &[String],
    options: &MineOptions,
) -> Vec<MinedInvariant> {
    let mut relations = Vec::new();
    for (i, x) in variables.iter().enumerate() {
        for y in &variables[i + 1..] {
            // Evidence: states observing both variables as integers. A state
            // observing both where either is non-integer disqualifies the pair.
            let mut evidence: Vec<(i64, i64)> = Vec::new();
            let mut disqualified = false;
            'outer: for trace in traces {
                for step in &trace.steps {
                    match (step.state.get(x), step.state.get(y)) {
                        (Some(vx), Some(vy)) => match (as_int(vx), as_int(vy)) {
                            (Some(ix), Some(iy)) => evidence.push((ix, iy)),
                            _ => {
                                disqualified = true;
                                break 'outer;
                            }
                        },
                        _ => {}
                    }
                }
            }
            if disqualified || evidence.len() < options.min_relation_evidence {
                continue;
            }

            let name = format!("Rel_{x}_{y}");
            let deltas: BTreeSet<i64> = evidence.iter().map(|(ix, iy)| ix - iy).collect();
            if deltas.len() == 1 {
                let k = *deltas.first().expect("non-empty");
                let def = match k {
                    0 => format!("{x} = {y}"),
                    k if k > 0 => format!("{x} = {y} + {k}"),
                    k => format!("{x} = {y} - {}", -k),
                };
                relations.push(MinedInvariant {
                    name,
                    description: format!("linear relation over {} observed states", evidence.len()),
                    def,
                    kind: InvariantKind::Relation,
                });
            } else if evidence.iter().all(|(ix, iy)| ix <= iy) {
                relations.push(MinedInvariant {
                    name,
                    def: format!("{x} <= {y}"),
                    kind: InvariantKind::Relation,
                    description: format!("ordering over {} observed states", evidence.len()),
                });
            } else if evidence.iter().all(|(ix, iy)| iy <= ix) {
                relations.push(MinedInvariant {
                    name,
                    def: format!("{y} <= {x}"),
                    kind: InvariantKind::Relation,
                    description: format!("ordering over {} observed states", evidence.len()),
                });
            }
        }
    }
    relations
}

/// Mine per-variable monotonicity property candidates: variables that never
/// decreased across any observed step pair (with at least one strict
/// increase) become `[][x' >= x]_vars` candidates. Monotone *decrease* is
/// reported as a note only.
pub(crate) fn mine_monotone_properties(
    traces: &[MiningTrace],
    variables: &[String],
    notes: &mut Vec<String>,
) -> Vec<MinedProperty> {
    let mut properties = Vec::new();
    for var in variables {
        let mut pairs: Vec<(i64, i64)> = Vec::new();
        let mut all_int = true;
        for trace in traces {
            for window in trace.steps.windows(2) {
                if let (Some(pre), Some(post)) =
                    (window[0].state.get(var), window[1].state.get(var))
                {
                    match (as_int(pre), as_int(post)) {
                        (Some(ipre), Some(ipost)) => pairs.push((ipre, ipost)),
                        _ => all_int = false,
                    }
                }
            }
        }
        if !all_int || pairs.is_empty() {
            continue;
        }
        let has_strict_increase = pairs.iter().any(|(pre, post)| post > pre);
        let has_strict_decrease = pairs.iter().any(|(pre, post)| post < pre);
        if !has_strict_decrease && has_strict_increase {
            properties.push(MinedProperty {
                name: format!("{var}_Monotone"),
                def: format!("[][{var}' >= {var}]_vars"),
                description: format!("never decreased across {} observed step pairs", pairs.len()),
            });
        } else if has_strict_decrease && !has_strict_increase {
            notes.push(format!(
                "{var:?} is monotone non-increasing across {} observed step pairs \
                 (note only; no candidate emitted)",
                pairs.len()
            ));
        }
    }
    properties
}
