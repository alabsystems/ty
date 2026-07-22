// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! TLA+ Module generation from ConcurrentModel IR.
//!
//! Translates the concurrent model into a programmatic TLA+ AST using the
//! same `Module`/`Expr`/`OperatorDef` types that the parser produces.

mod actions;
mod fairness;
mod helpers;
mod init;
mod next;
mod properties;
mod variables;

use std::collections::HashSet;

use tla_core::ast::{Module, Unit};
use tla_core::span::{FileId, Span, Spanned};

use crate::model::ConcurrentModel;

/// Generate a TLA+ Module from a ConcurrentModel.
///
/// The generated module contains:
/// - VARIABLES: pc, mutex_owner, locks_held, rw_readers, rw_writer, condvar_waiting,
///   channel_buf, senders_alive, receiver_alive, alive, shared vars, mutex_poisoned
/// - Init: initial state predicate
/// - Next: disjunction of all process actions
/// - Invariants: DeadlockFreedom, LockOrderConsistency, DataRaceFreedom
/// - Temporal properties: for opt-in liveness checks
pub fn generate_tla_module(model: &ConcurrentModel) -> Module {
    let mut units: Vec<Spanned<Unit>> = Vec::new();

    // 1. Variables
    let var_names = variables::collect_variable_names(model);
    units.push(spanned(Unit::Variable(
        var_names.iter().map(|n| spanned(n.clone())).collect(),
    )));

    // 2. Helper operators (process set, terminal states, etc.)
    for op in helpers::generate_helper_operators(model) {
        units.push(spanned(Unit::Operator(op)));
    }

    // 3. Init predicate
    units.push(spanned(Unit::Operator(init::generate_init(model))));

    // 4. Per-process action operators
    for process in &model.processes {
        for action_op in actions::generate_process_actions(model, process) {
            units.push(spanned(Unit::Operator(action_op)));
        }
    }

    // 5. Next relation (disjunction of all process actions)
    units.push(spanned(Unit::Operator(next::generate_next(model))));

    // 6. Property operators (invariants + temporal)
    for prop_op in properties::generate_property_operators(model) {
        units.push(spanned(Unit::Operator(prop_op)));
    }

    // 7. Fairness specification (if liveness properties present)
    if let Some(fair_op) = fairness::generate_fairness_spec(model) {
        units.push(spanned(Unit::Operator(fair_op)));
    }

    Module {
        name: spanned("ConcurrentSpec".to_string()),
        extends: vec![
            spanned("Naturals".to_string()),
            spanned("Sequences".to_string()),
            spanned("FiniteSets".to_string()),
        ],
        units,
        action_subscript_spans: HashSet::new(),
        span: make_span(),
    }
}

/// Build a Config for the generated module.
#[cfg(feature = "check")]
pub fn build_config(model: &ConcurrentModel) -> tla_check::Config {
    let mut config = tla_check::Config::new();
    config.init = Some("Init".to_string());
    config.next = Some("Next".to_string());
    config.check_deadlock = true; // Built-in deadlock detection; Next has stuttering for terminal states

    // Add property invariants (except DeadlockFreedom, handled by check_deadlock)
    for prop in &model.properties {
        if matches!(prop, crate::property::Property::DeadlockFreedom) {
            continue; // Handled by check_deadlock = true
        }
        // The temporal body for these liveness properties is not yet implemented
        // (Phase 2): `generate_property_body` emits a trivially-true operator, so
        // adding them to the checked set would report a false "verified" result.
        // Skip them with a warning rather than silently pass.
        // Every property whose `generate_property_body` returns a placeholder
        // `TRUE` (the explicit stubs + the `_ => bool_lit(true)` catch-all) must
        // be skipped: adding a trivially-true operator to the checked set would
        // report a false "verified" result for an unimplemented property. Only
        // the genuinely-implemented properties (DeadlockFreedom [via
        // check_deadlock], DataRaceFreedom, Termination, JoinCompleteness) are
        // checked. (audit3 N7)
        if matches!(
            prop,
            crate::property::Property::CondvarProgress { .. }
                | crate::property::Property::RwLockWriterProgress { .. }
                | crate::property::Property::LockOrderConsistency
                | crate::property::Property::PoisonHandled { .. }
                | crate::property::Property::ChannelFifo { .. }
                | crate::property::Property::ChannelDisconnect { .. }
                | crate::property::Property::BarrierCorrectness { .. }
                | crate::property::Property::ChannelNoLoss { .. }
                | crate::property::Property::Linearizable { .. }
                | crate::property::Property::Custom { .. }
        ) {
            eprintln!(
                "warning: property {} is not yet implemented; skipping it \
                 (checking it would assert a trivially-true operator and spuriously pass)",
                properties::property_operator_name(prop),
            );
            continue;
        }
        let name = properties::property_operator_name(prop);
        if prop.is_safety() {
            config.invariants.push(name);
        } else {
            config.properties.push(name);
        }
    }

    config
}

// ── Span helpers ─────────────────────────────────────────────────

fn make_span() -> Span {
    Span::new(FileId(0), 0, 0)
}

fn spanned<T>(node: T) -> Spanned<T> {
    Spanned::new(node, make_span())
}
