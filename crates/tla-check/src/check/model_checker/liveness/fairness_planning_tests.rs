// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Tests for fairness classification: PROPERTY-embedded fairness is planned as
//! EAAction (negated obligation), while SPECIFICATION-extracted fairness is
//! planned as AEAction (assumption). Verifies TLC's temporals/impliedTemporals split.

use super::*;
use crate::config::Config;
use crate::liveness::LiveExpr;
use crate::resolve_spec_from_config;
use crate::test_support::parse_module_with_id;
use tla_core::ast::{Expr, ModuleTarget};
use tla_core::{lower, parse_to_syntax_tree, ExprVisitor, FileId, Spanned};

const SPECIFICATION_PROPERTY_SPEC: &str = r#"
---- MODULE SpecPropMinRepro ----
EXTENDS Naturals
VARIABLE x
vars == <<x>>
Init == x = 0
Next == x' = (x + 1) % 3
Spec == Init /\ [][Next]_vars /\ WF_vars(Next)
Live == <>(x = 2)
SafeProp == [](x >= 0)
====
"#;

const PROPERTY_EMBEDDED_FAIRNESS_SPEC: &str = r#"
---- MODULE PropertyEmbeddedFairness ----
EXTENDS Naturals
VARIABLE x
vars == <<x>>
Init == x = 0
Next == x' = (x + 1) % 3
Fairness == WF_vars(Next)
Spec == Init /\ [][Next]_vars /\ Fairness
====
"#;

fn spec_property_config() -> Config {
    Config {
        specification: Some("Spec".to_string()),
        properties: vec!["Live".to_string()],
        ..Default::default()
    }
}

fn embedded_fairness_property_config() -> Config {
    Config {
        init: Some("Init".to_string()),
        next: Some("Next".to_string()),
        properties: vec!["Spec".to_string()],
        ..Default::default()
    }
}

/// TLC classifies `PROPERTY Spec` as livecheck, not livespec, even when `Spec`
/// itself contains `WF_vars(...)`. That means the property-side fairness term is
/// negated and planned as an EAAction obligation (`<>[]...`), not as an AEAction
/// fairness assumption (`[]<>...`).
#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn property_embedded_fairness_is_planned_as_ea_action() {
    let tree = parse_to_syntax_tree(PROPERTY_EMBEDDED_FAIRNESS_SPEC);
    let module = lower(FileId(0), &tree).module.expect("lowered module");
    let config = embedded_fairness_property_config();

    let mut checker = ModelChecker::new(&module, &config);
    let def = checker
        .module
        .op_defs
        .get("Spec")
        .expect("property operator should exist")
        .clone();
    let (_safety_parts, liveness_expr) = checker
        .separate_safety_liveness_parts("Spec", &def.body)
        .expect("property split should succeed");
    let liveness_expr = liveness_expr.expect("Spec should retain a liveness term");

    let (plans, max_fairness_tag) = checker
        .build_grouped_plans_for_property("Spec", &liveness_expr)
        .expect("grouped plans should build");

    assert_eq!(
        max_fairness_tag, 0,
        "embedded PROPERTY fairness should not be treated as extracted fairness"
    );
    assert_eq!(
        plans.len(),
        1,
        "WF-only property should yield one grouped plan"
    );
    assert!(
        matches!(plans[0].tf, LiveExpr::Bool(true)),
        "pure embedded fairness should use the no-tableau fast path after classification"
    );
    assert_eq!(
        plans[0].pems.len(),
        1,
        "WF-only property should yield one PEM"
    );
    assert_eq!(
        plans[0].check_action.len(),
        1,
        "negated WF should contribute one action-level check"
    );
    assert!(
        plans[0].check_state.is_empty(),
        "negated WF should not create top-level state-only checks"
    );
    assert_eq!(
        plans[0].pems[0].ea_action_idx.len(),
        1,
        "embedded PROPERTY fairness must become an EAAction obligation"
    );
    assert!(
        plans[0].pems[0].ae_action_idx.is_empty(),
        "embedded PROPERTY fairness must not be reclassified as extracted fairness"
    );
}

/// The same `WF_vars(Next)` term should become extracted fairness when it comes
/// from `SPECIFICATION`, matching TLC's `temporals`/`impliedTemporals` split.
#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn specification_fairness_is_planned_as_ae_action() {
    let tree = parse_to_syntax_tree(SPECIFICATION_PROPERTY_SPEC);
    let module = lower(FileId(0), &tree).module.expect("lowered module");
    let unresolved = spec_property_config();
    let resolved =
        resolve_spec_from_config(&unresolved, &tree).expect("SPECIFICATION should resolve");

    let config = Config {
        init: Some(resolved.init.clone()),
        next: Some(resolved.next.clone()),
        specification: unresolved.specification.clone(),
        properties: unresolved.properties.clone(),
        ..Default::default()
    };

    let mut checker = ModelChecker::new(&module, &config);
    checker.set_fairness(resolved.fairness);

    let def = checker
        .module
        .op_defs
        .get("Live")
        .expect("property operator should exist")
        .clone();
    let (_safety_parts, liveness_expr) = checker
        .separate_safety_liveness_parts("Live", &def.body)
        .expect("property split should succeed");
    let liveness_expr = liveness_expr.expect("Live should retain a liveness term");

    let (plans, max_fairness_tag) = checker
        .build_grouped_plans_for_property("Live", &liveness_expr)
        .expect("grouped plans should build");

    assert!(
        max_fairness_tag > 0,
        "SPECIFICATION fairness should be converted before property negation"
    );
    assert_eq!(
        plans.len(),
        1,
        "simple property should yield one grouped plan"
    );
    assert!(
        !matches!(plans[0].tf, LiveExpr::Bool(true)),
        "negated <> property should still require a tableau term"
    );
    assert_eq!(
        plans[0].pems.len(),
        1,
        "simple property should yield one PEM"
    );
    assert_eq!(
        plans[0].check_action.len(),
        1,
        "SPECIFICATION fairness should contribute one action-level fairness check"
    );
    assert_eq!(
        plans[0].pems[0].ae_action_idx.len(),
        1,
        "extracted SPECIFICATION fairness must become an AEAction assumption"
    );
    assert!(
        plans[0].pems[0].ea_action_idx.is_empty(),
        "SPECIFICATION fairness must not be negated into EAAction"
    );
}

const CHAINED_TD_MODULE: &str = r#"
---- MODULE ChainedTD ----
VARIABLE chan
CONSTANT N, inbox
Node == 0..N-1
Live == <>(\A n \in Node : chan = chan /\ inbox[n] = n)
====
"#;

const CHAINED_MID_MODULE: &str = r#"
---- MODULE ChainedMid ----
EXTENDS Integers
VARIABLE chan
N == 2
Node == 0..N-1
TD == INSTANCE ChainedTD WITH inbox <- [i \in Node |-> i]
====
"#;

const CHAINED_OUTER_MODULE: &str = r#"
---- MODULE ChainedOuter ----
VARIABLE chan
Node == {TLCModelValue("n1"), TLCModelValue("n2")}
GoodInbox == [n \in Node |-> 99]
Mid == INSTANCE ChainedMid
Prop == Mid!TD!Live
====
"#;

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn chained_instance_liveness_leaf_keeps_intermediate_scope() {
    let td = parse_module_with_id(CHAINED_TD_MODULE, FileId(20));
    let mid = parse_module_with_id(CHAINED_MID_MODULE, FileId(21));
    let outer = parse_module_with_id(CHAINED_OUTER_MODULE, FileId(22));
    let config = Config {
        properties: vec!["Prop".to_string()],
        ..Default::default()
    };

    let checker = ModelChecker::new_with_extends(&outer, &[&mid, &td], &config);
    let def = checker
        .module
        .op_defs
        .get("Prop")
        .expect("property operator should exist")
        .clone();
    let converter = crate::liveness::AstToLive::new()
        .with_location_module_name(checker.module.root_name.as_str());
    let live = converter
        .convert(&checker.ctx, &def.body)
        .expect("chained liveness property should convert");

    let mut state_predicates = Vec::new();
    collect_state_predicates(&live, &mut state_predicates);
    assert_eq!(
        state_predicates.len(),
        1,
        "simple <> predicate should produce exactly one state predicate leaf"
    );

    let leaf = state_predicates[0];
    assert!(
        contains_chained_module_ref(&leaf.node, "Node"),
        "TD liveness leaf must qualify the int-keyed Node operator"
    );
    assert!(
        !contains_unqualified_ident(&leaf.node, "Node"),
        "TD liveness leaf must not fall back to the outer model-value Node"
    );

    let mut eval_ctx = checker.ctx.clone();
    let state = vec![crate::Value::int(0)];
    eval_ctx.bind_state_array(&state);
    let value = crate::eval::eval_entry(&eval_ctx, leaf)
        .expect("qualified chained liveness leaf should evaluate");
    assert_eq!(
        value,
        crate::Value::Bool(true),
        "quantified leaf should use the intermediate int-keyed inbox"
    );
}

fn collect_state_predicates<'a>(expr: &'a LiveExpr, out: &mut Vec<&'a Spanned<Expr>>) {
    match expr {
        LiveExpr::StatePred { expr, .. } => out.push(expr.as_ref()),
        LiveExpr::And(parts) | LiveExpr::Or(parts) => {
            for part in parts {
                collect_state_predicates(part, out);
            }
        }
        LiveExpr::Not(inner)
        | LiveExpr::Always(inner)
        | LiveExpr::Eventually(inner)
        | LiveExpr::Next(inner) => collect_state_predicates(inner, out),
        LiveExpr::Bool(_)
        | LiveExpr::ActionPred { .. }
        | LiveExpr::Enabled { .. }
        | LiveExpr::StateChanged { .. } => {}
    }
}

fn contains_chained_module_ref(expr: &Expr, op_name: &str) -> bool {
    tla_core::walk_expr(&mut ChainedModuleRefFinder { op_name }, expr)
}

fn contains_unqualified_ident(expr: &Expr, needle: &str) -> bool {
    tla_core::walk_expr(&mut UnqualifiedIdentFinder { needle }, expr)
}

struct ChainedModuleRefFinder<'a> {
    op_name: &'a str,
}

impl ExprVisitor for ChainedModuleRefFinder<'_> {
    type Output = bool;

    fn visit_node(&mut self, expr: &Expr) -> Option<Self::Output> {
        match expr {
            Expr::ModuleRef(ModuleTarget::Chained(_), actual, _) if actual == self.op_name => {
                Some(true)
            }
            _ => None,
        }
    }
}

struct UnqualifiedIdentFinder<'a> {
    needle: &'a str,
}

impl ExprVisitor for UnqualifiedIdentFinder<'_> {
    type Output = bool;

    fn visit_node(&mut self, expr: &Expr) -> Option<Self::Output> {
        match expr {
            Expr::Ident(name, _) if name == self.needle => Some(true),
            _ => None,
        }
    }
}
