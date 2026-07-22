// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Rendering a [`MinedSpec`] as a TLA+ module and a `.cfg` file.

use std::fmt::Write as _;

use super::MinedSpec;

/// Render the mined candidate spec as TLA+ module source.
#[must_use]
pub fn render_module(spec: &MinedSpec) -> String {
    let mut out = String::new();
    let _ = writeln!(out, "---- MODULE {} ----", spec.module_name);
    out.push_str(
        "\\* CANDIDATE specification mined from observed traces by `ty trace mine`.\n\
         \\* It generalizes finitely many observations and is a hypothesis for\n\
         \\* HUMAN REVIEW — not ground truth about the system that produced them.\n",
    );

    let mut extends = Vec::new();
    if spec.needs_integers {
        extends.push("Integers");
    }
    if spec.needs_tlc {
        extends.push("TLC");
    }
    if !extends.is_empty() {
        let _ = writeln!(out, "EXTENDS {}", extends.join(", "));
    }
    if !spec.constants.is_empty() {
        let _ = writeln!(out, "CONSTANTS {}", spec.constants.join(", "));
    }

    let _ = writeln!(out, "\nVARIABLES {}\n", spec.variables.join(", "));
    let _ = writeln!(out, "vars == <<{}>>", spec.variables.join(", "));

    out.push_str("\n\\* Observed variable domains.\n");
    for dom in &spec.domains {
        let _ = writeln!(out, "{} == {}", dom.op_name, dom.expr);
    }

    if !spec.invariants.is_empty() {
        out.push_str("\n\\* Candidate invariants.\n");
        for inv in &spec.invariants {
            let _ = writeln!(out, "{} == {}", inv.name, inv.def);
        }
    }

    out.push_str("\n\\* Init: join of the observed initial states.\nInit ==\n");
    for conjuncts in &spec.init_disjuncts {
        let _ = writeln!(out, "  \\/ ({})", conjuncts.join(" /\\ "));
    }

    out.push_str("\n\\* Mined candidate actions.\n");
    for action in &spec.actions {
        let _ = writeln!(out, "{} ==", action.name);
        let mut conjuncts: Vec<String> = action.guards.clone();
        for update in &action.updates {
            conjuncts.extend(update.conjuncts.iter().cloned());
        }
        if action.unchanged.len() == spec.variables.len() {
            conjuncts.push("UNCHANGED vars".to_string());
        } else if !action.unchanged.is_empty() {
            conjuncts.push(format!("UNCHANGED <<{}>>", action.unchanged.join(", ")));
        }
        debug_assert!(!conjuncts.is_empty(), "actions always constrain something");
        for conjunct in &conjuncts {
            let _ = writeln!(out, "  /\\ {conjunct}");
        }
        out.push('\n');
    }

    out.push_str("Next ==\n");
    if spec.actions.is_empty() {
        // Degenerate corpus (no step pairs): stuttering keeps the spec checkable.
        out.push_str("  UNCHANGED vars\n");
    } else {
        for action in &spec.actions {
            let _ = writeln!(out, "  \\/ {}", action.name);
        }
    }

    if !spec.properties.is_empty() {
        out.push_str("\n\\* Candidate monotonicity properties.\n");
        for prop in &spec.properties {
            let _ = writeln!(out, "{} == {}", prop.name, prop.def);
        }
    }

    out.push_str("====\n");
    out
}

/// Render the model-checking config for the mined spec.
///
/// `CHECK_DEADLOCK FALSE`: a mined spec stops at the observation boundary by
/// construction, so terminal states are expected, not defects.
#[must_use]
pub fn render_config(spec: &MinedSpec) -> String {
    let mut out = String::new();
    out.push_str("INIT Init\nNEXT Next\nCHECK_DEADLOCK FALSE\n");
    for constant in &spec.constants {
        let _ = writeln!(out, "CONSTANT {constant} = {constant}");
    }
    for inv in &spec.invariants {
        let _ = writeln!(out, "INVARIANT {}", inv.name);
    }
    for prop in &spec.properties {
        let _ = writeln!(out, "PROPERTY {}", prop.name);
    }
    out
}
