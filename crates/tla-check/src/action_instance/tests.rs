// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Tests for action instance splitting.

use super::*;
use crate::EvalCtx;
use tla_core::ast::{BoundVar, Expr, Module};
use tla_core::{lower, parse_to_syntax_tree, FileId};

fn load_and_find_op(src: &str, name: &str) -> (EvalCtx, OperatorDef) {
    let tree = parse_to_syntax_tree(src);
    let lowered = lower(FileId(0), &tree);
    assert!(
        lowered.errors.is_empty(),
        "unexpected lower errors: {:?}",
        lowered.errors
    );
    let module = lowered.module.expect("lowered module");

    let mut ctx = EvalCtx::new();
    ctx.load_module(&module);

    let def = module
        .units
        .iter()
        .find_map(|u| match &u.node {
            tla_core::ast::Unit::Operator(d) if d.name.node == name => Some(d.clone()),
            _ => None,
        })
        .unwrap_or_else(|| panic!("missing operator {name}"));
    (ctx, def)
}

fn lower_module(src: &str) -> Module {
    let tree = parse_to_syntax_tree(src);
    let lowered = lower(FileId(0), &tree);
    assert!(
        lowered.errors.is_empty(),
        "unexpected lower errors: {:?}",
        lowered.errors
    );
    lowered.module.expect("lowered module")
}

fn expect_if(expr: &Spanned<Expr>) -> (&Spanned<Expr>, &Spanned<Expr>, &Spanned<Expr>) {
    match &expr.node {
        Expr::If(cond, then_branch, else_branch) => {
            (cond.as_ref(), then_branch.as_ref(), else_branch.as_ref())
        }
        other => panic!("expected IF expression, got {other:?}"),
    }
}

fn expect_let_named<'a>(expr: &'a Spanned<Expr>, expected_name: &str) -> &'a Spanned<Expr> {
    match &expr.node {
        Expr::Let(defs, body) => {
            let actual_name = defs
                .first()
                .map(|def| def.name.node.as_str())
                .expect("LET should have at least one definition");
            assert_eq!(actual_name, expected_name);
            body.as_ref()
        }
        other => panic!("expected LET {expected_name}, got {other:?}"),
    }
}

fn assert_name_ref(expr: &Spanned<Expr>, expected_name: &str) {
    match &expr.node {
        Expr::Ident(name, _) | Expr::StateVar(name, _, _) => {
            assert_eq!(name.as_str(), expected_name)
        }
        other => panic!("expected name reference {expected_name}, got {other:?}"),
    }
}

fn assert_int(expr: &Spanned<Expr>, expected: i64) {
    match &expr.node {
        Expr::Int(actual) => assert_eq!(actual.to_string(), expected.to_string()),
        other => panic!("expected int literal {expected}, got {other:?}"),
    }
}

fn assert_eq_name_int(expr: &Spanned<Expr>, expected_name: &str, expected_value: i64) {
    match &expr.node {
        Expr::Eq(left, right) => {
            assert_name_ref(left, expected_name);
            assert_int(right, expected_value);
        }
        other => panic!("expected equality guard, got {other:?}"),
    }
}

fn successor_xs_by_action(
    ctx: &mut EvalCtx,
    next_def: &OperatorDef,
    x: i64,
) -> Vec<(Option<String>, Vec<i64>)> {
    let state = State::from_pairs([("x", Value::int(x))]);
    let vars = vec![Arc::from("x")];

    enumerate_successors_by_action_instance(ctx, next_def, &state, &vars)
        .expect("enumerate successors")
        .into_iter()
        .map(|per_action| {
            let xs = per_action
                .successors
                .iter()
                .map(|succ| {
                    succ.vars()
                        .find_map(|(k, v)| (k.as_ref() == "x").then_some(v))
                        .and_then(tla_value::Value::as_i64)
                        .expect("successor has x=int(..)")
                })
                .collect();
            (per_action.instance.name, xs)
        })
        .collect()
}

// Part of #3354 Slice 4: bosco_real_next, action_contains_subset_constrained,
// and test_split_bosco_real_next_preserves_constrained_variant removed —
// they tested CompiledAction pattern preservation which is no longer applicable.

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_split_if_branches_preserve_condition_and_are_enforced() {
    let src = r#"
---- MODULE Test ----
VARIABLE x

ThenAction == x' = 1
ElseAction == x' = 2

Next == IF x = 0 THEN ThenAction ELSE ElseAction
====
"#;

    let (mut ctx, next_def) = load_and_find_op(src, "Next");
    let actions = split_action_instances(&ctx, &next_def.body).unwrap();
    assert_eq!(actions.len(), 2);

    assert_eq!(actions[0].name.as_deref(), Some("ThenAction"));
    let (then_cond, then_branch, then_else) = expect_if(&actions[0].expr);
    assert_eq_name_int(then_cond, "x", 0);
    assert!(matches!(&then_branch.node, Expr::Eq(_, _)));
    assert!(matches!(&then_else.node, Expr::Bool(false)));

    assert_eq!(actions[1].name.as_deref(), Some("ElseAction"));
    let (else_cond, else_then, else_branch) = expect_if(&actions[1].expr);
    assert_eq_name_int(else_cond, "x", 0);
    assert!(matches!(&else_then.node, Expr::Bool(false)));
    assert!(matches!(&else_branch.node, Expr::Eq(_, _)));

    let at_zero = successor_xs_by_action(&mut ctx, &next_def, 0);
    assert_eq!(
        at_zero,
        vec![
            (Some("ThenAction".to_string()), vec![1]),
            (Some("ElseAction".to_string()), vec![]),
        ]
    );

    let at_one = successor_xs_by_action(&mut ctx, &next_def, 1);
    assert_eq!(
        at_one,
        vec![
            (Some("ThenAction".to_string()), vec![]),
            (Some("ElseAction".to_string()), vec![2]),
        ]
    );
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_split_if_preserves_lexical_let_order_around_branch_condition() {
    let src = r#"
---- MODULE Test ----
VARIABLE x

ThenAction ==
  LET Inner == x' = 1
  IN Inner

ElseAction == x' = 2

Next ==
  LET Outer == x = 0
  IN LET BranchGuard == Outer
     IN IF BranchGuard THEN
          ThenAction
        ELSE ElseAction
====
"#;

    let (ctx, next_def) = load_and_find_op(src, "Next");
    let actions = split_action_instances(&ctx, &next_def.body).unwrap();
    assert_eq!(actions.len(), 2);

    let outer_body = expect_let_named(&actions[0].expr, "Outer");
    let branch_if = expect_let_named(outer_body, "BranchGuard");
    let (cond, selected_body, unselected_body) = expect_if(branch_if);
    assert_name_ref(cond, "BranchGuard");
    assert!(matches!(&unselected_body.node, Expr::Bool(false)));

    let inner_body = expect_let_named(selected_body, "Inner");
    assert_name_ref(inner_body, "Inner");
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_split_if_does_not_force_unsplittable_exists() {
    let src = r#"
---- MODULE Test ----
EXTENDS Naturals
VARIABLE x

ThenAction == \E y \in Nat : x' = y
ElseAction == x' = 2

Next == IF x = 0 THEN ThenAction ELSE ElseAction
====
"#;

    let (ctx, next_def) = load_and_find_op(src, "Next");
    let actions = split_action_instances(&ctx, &next_def.body).unwrap();
    assert_eq!(actions.len(), 2);

    assert_eq!(actions[0].name.as_deref(), Some("ThenAction"));
    let (then_cond, then_body, then_else) = expect_if(&actions[0].expr);
    assert_eq_name_int(then_cond, "x", 0);
    assert!(matches!(&then_else.node, Expr::Bool(false)));
    assert!(
        matches!(&then_body.node, Expr::Exists(_, _)),
        "unsplittable EXISTS should remain a conditional leaf: {:?}",
        &then_body.node
    );

    let (else_cond, else_then, _) = expect_if(&actions[1].expr);
    assert_eq_name_int(else_cond, "x", 0);
    assert!(matches!(&else_then.node, Expr::Bool(false)));
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_split_if_existential_condition_is_boolean_not_witness_multiplicity() {
    let src = r#"
---- MODULE Test ----
VARIABLE x

HasWitness == \E i \in {1, 2, 3} : x = 0
ThenAction == x' = 1
ElseAction == x' = 2

Next == IF HasWitness THEN ThenAction ELSE ElseAction
====
"#;

    let (mut ctx, next_def) = load_and_find_op(src, "Next");
    let actions = split_action_instances(&ctx, &next_def.body).unwrap();
    assert_eq!(actions.len(), 2);

    let at_zero = successor_xs_by_action(&mut ctx, &next_def, 0);
    assert_eq!(
        at_zero,
        vec![
            (Some("ThenAction".to_string()), vec![1]),
            (Some("ElseAction".to_string()), vec![]),
        ]
    );
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_split_if_direct_assignment_branches_remain_monolithic() {
    let src = r#"
---- MODULE Test ----
VARIABLE x

Next == IF x = 0 THEN x' = 1 ELSE UNCHANGED x
====
"#;

    let (mut ctx, next_def) = load_and_find_op(src, "Next");
    let actions = split_action_instances(&ctx, &next_def.body).unwrap();
    assert_eq!(actions.len(), 1);
    assert!(actions[0].name.is_none());
    assert!(
        matches!(&actions[0].expr.node, Expr::If(_, _, _)),
        "direct IF-shaped action should stay as one leaf: {:?}",
        &actions[0].expr.node
    );

    let at_zero = successor_xs_by_action(&mut ctx, &next_def, 0);
    assert_eq!(at_zero, vec![(None, vec![1])]);

    let at_one = successor_xs_by_action(&mut ctx, &next_def, 1);
    assert_eq!(at_one, vec![(None, vec![1])]);
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_split_specializes_const_level_apply_but_not_state_level() {
    let src = r#"
---- MODULE Test ----
VARIABLE x

Op(a) ==
  \/ a = 1 /\ x' = 1
  \/ a = 1 /\ x' = 2

Next == Op(1) \/ Op(x)
====
"#;

    let (ctx, next_def) = load_and_find_op(src, "Next");
    let actions = split_action_instances(&ctx, &next_def.body).unwrap();

    // Op(1) specializes and splits into 2 actions; Op(x) does not specialize and remains 1 leaf.
    assert_eq!(actions.len(), 3);

    let a1 = Value::int(1);
    let specialized = actions
        .iter()
        .filter(|a| {
            a.bindings
                .iter()
                .any(|(k, v)| k.as_ref() == "a" && v == &a1)
        })
        .count();
    assert_eq!(specialized, 2);
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_split_allows_bounded_exists_bindings_for_const_level_actuals() {
    let src = r#"
---- MODULE Test ----
VARIABLE x

Op(a) ==
  \/ a = 1 /\ x' = 1
  \/ a = 2 /\ x' = 2

Next == \E y \in {1, 2} : Op(y)
====
"#;

    let (ctx, next_def) = load_and_find_op(src, "Next");
    let actions = split_action_instances(&ctx, &next_def.body).unwrap();

    // y is enumerated (2 values), and each Op(y) specializes + splits into 2 disjunct actions.
    assert_eq!(actions.len(), 4);

    let a_vals: Vec<i64> = actions
        .iter()
        .filter_map(|a| {
            let y = a.bindings.iter().find_map(|(k, v)| {
                if k.as_ref() != "y" {
                    return None;
                }
                v.as_i64()
            })?;
            let formal = a
                .formal_bindings
                .iter()
                .find_map(|(k, v)| (k.as_ref() == "a").then_some(v))
                .and_then(tla_value::Value::as_i64)?;
            assert_eq!(formal, y);
            Some(y)
        })
        .collect();
    assert_eq!(a_vals.iter().copied().filter(|v| *v == 1).count(), 2);
    assert_eq!(a_vals.iter().copied().filter(|v| *v == 2).count(), 2);
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_split_ewd_style_exists_action_keeps_witness_and_formal_bindings() {
    let src = r#"
---- MODULE Test ----
VARIABLE x

Node == {0, 1}

SendMsg(sender) == x' = sender

Next == \E i \in Node : SendMsg(i)
====
"#;

    let (ctx, next_def) = load_and_find_op(src, "Next");
    let actions = split_action_instances(&ctx, &next_def.body).unwrap();
    assert_eq!(actions.len(), 2);

    let mut seen = Vec::new();
    for action in actions {
        assert_eq!(action.name.as_deref(), Some("SendMsg"));
        assert_eq!(action.bindings.len(), 1);
        assert_eq!(action.formal_bindings.len(), 1);

        let outer = action
            .bindings
            .iter()
            .find_map(|(k, v)| (k.as_ref() == "i").then_some(v))
            .and_then(tla_value::Value::as_i64)
            .expect("instance has outer witness i=int(..)");
        let formal = action
            .formal_bindings
            .iter()
            .find_map(|(k, v)| (k.as_ref() == "sender").then_some(v))
            .and_then(tla_value::Value::as_i64)
            .expect("instance has formal sender=int(..)");
        assert_eq!(formal, outer);

        assert_eq!(action.formal_bindings[0].0.as_ref(), "sender");
        assert_eq!(action.formal_bindings[0].1.as_i64(), Some(outer));
        seen.push(outer);
    }

    seen.sort_unstable();
    assert_eq!(seen, vec![0, 1]);
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_split_supports_dependent_bounded_exists_domains() {
    let src = r#"
---- MODULE Test ----
VARIABLE x

Op(a) == x' = a

Next == \E y \in {1, 2} : \E z \in {y} : Op(z)
====
"#;

    let (ctx, next_def) = load_and_find_op(src, "Next");
    let actions = split_action_instances(&ctx, &next_def.body).unwrap();
    assert_eq!(actions.len(), 2);

    let mut a_vals: Vec<i64> = actions
        .iter()
        .map(|a| {
            a.bindings
                .iter()
                .find_map(|(k, v)| (k.as_ref() == "z").then_some(v))
                .and_then(tla_value::Value::as_i64)
                .expect("instance has z=int(..)")
        })
        .collect();
    a_vals.sort_unstable();
    assert_eq!(a_vals, vec![1, 2]);
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_split_supports_dependent_bounded_exists_domains_when_bounds_reversed() {
    let src = r#"
---- MODULE Test ----
VARIABLE x

Op(a) == x' = a
====
"#;

    let module = lower_module(src);

    let mut ctx = EvalCtx::new();
    ctx.load_module(&module);

    let y_bound = BoundVar {
        name: Spanned::dummy("y".to_string()),
        domain: Some(Box::new(Spanned::dummy(Expr::SetEnum(vec![
            Spanned::dummy(Expr::Int(1.into())),
            Spanned::dummy(Expr::Int(2.into())),
        ])))),
        pattern: None,
    };

    let z_bound = BoundVar {
        name: Spanned::dummy("z".to_string()),
        domain: Some(Box::new(Spanned::dummy(Expr::SetEnum(vec![
            Spanned::dummy(Expr::Ident(
                "y".to_string(),
                tla_core::name_intern::NameId::INVALID,
            )),
        ])))),
        pattern: None,
    };

    let body = Spanned::dummy(Expr::Apply(
        Box::new(Spanned::dummy(Expr::Ident(
            "Op".to_string(),
            tla_core::name_intern::NameId::INVALID,
        ))),
        vec![Spanned::dummy(Expr::Ident(
            "z".to_string(),
            tla_core::name_intern::NameId::INVALID,
        ))],
    ));

    // Deliberately reversed vs spec order: z depends on y, so the forward split fails and we
    // rely on the reverse-order fallback.
    let expr = Spanned::dummy(Expr::Exists(vec![z_bound, y_bound], Box::new(body)));

    let actions = split_action_instances(&ctx, &expr).unwrap();
    assert_eq!(actions.len(), 2);

    let mut a_vals: Vec<i64> = actions
        .iter()
        .map(|a| {
            a.bindings
                .iter()
                .find_map(|(k, v)| (k.as_ref() == "z").then_some(v))
                .and_then(tla_value::Value::as_i64)
                .expect("instance has z=int(..)")
        })
        .collect();
    a_vals.sort_unstable();
    assert_eq!(a_vals, vec![1, 2]);
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_split_binds_const_level_module_ref_actuals() {
    let inst_src = r#"
---- MODULE Inst ----
VARIABLE x

Op(a) == x' = a
====
"#;
    let main_src = r#"
---- MODULE Test ----
VARIABLE x

I == INSTANCE Inst

Next == I!Op(1) \/ I!Op(2)
====
"#;

    let inst_module = lower_module(inst_src);
    let test_module = lower_module(main_src);

    let mut ctx = EvalCtx::new();
    ctx.load_instance_module("Inst".to_string(), &inst_module);
    ctx.load_module(&test_module);

    let next_def = test_module
        .units
        .iter()
        .find_map(|u| match &u.node {
            tla_core::ast::Unit::Operator(d) if d.name.node == "Next" => Some(d.clone()),
            _ => None,
        })
        .expect("missing Next");

    let actions = split_action_instances(&ctx, &next_def.body).unwrap();
    assert_eq!(actions.len(), 2);

    let mut a_vals: Vec<i64> = actions
        .iter()
        .map(|a| {
            assert_eq!(a.name.as_deref(), Some("I!Op"));
            a.bindings
                .iter()
                .find_map(|(k, v)| (k.as_ref() == "a").then_some(v))
                .and_then(tla_value::Value::as_i64)
                .expect("instance has a=int(..)")
        })
        .collect();
    a_vals.sort_unstable();
    assert_eq!(a_vals, vec![1, 2]);
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_split_named_instance_next_expands_qualified_leaf_actions() {
    let inst_src = r#"
---- MODULE Inst ----
CONSTANT Proc
VARIABLE x

Step(p) == x' = p

Next == \E p \in Proc : Step(p)
====
"#;
    let main_src = r#"
---- MODULE Test ----
VARIABLE x

Nodes == {1, 2}

I == INSTANCE Inst WITH Proc <- Nodes

Next == I!Next
====
"#;

    let inst_module = lower_module(inst_src);
    let test_module = lower_module(main_src);

    let mut ctx = EvalCtx::new();
    ctx.load_instance_module("Inst".to_string(), &inst_module);
    ctx.load_module(&test_module);

    let next_def = test_module
        .units
        .iter()
        .find_map(|u| match &u.node {
            tla_core::ast::Unit::Operator(d) if d.name.node == "Next" => Some(d.clone()),
            _ => None,
        })
        .expect("missing Next");

    let actions = split_action_instances(&ctx, &next_def.body).unwrap();
    assert_eq!(actions.len(), 2);
    assert!(
        actions
            .iter()
            .all(|action| action.name.as_deref() == Some("I!Step")),
        "named INSTANCE split should expose qualified leaf action names: {actions:?}"
    );

    let mut p_vals: Vec<i64> = actions
        .iter()
        .map(|action| {
            action
                .bindings
                .iter()
                .find_map(|(k, v)| (k.as_ref() == "p").then_some(v))
                .and_then(tla_value::Value::as_i64)
                .expect("instance has p=int(..)")
        })
        .collect();
    p_vals.sort_unstable();
    assert_eq!(p_vals, vec![1, 2]);

    let mut successors = successor_xs_by_action(&mut ctx, &next_def, 0);
    successors.sort_by(|a, b| a.1.cmp(&b.1));
    assert_eq!(
        successors,
        vec![
            (Some("I!Step".to_string()), vec![1]),
            (Some("I!Step".to_string()), vec![2]),
        ],
        "expanded named INSTANCE actions must remain executable by action-instance enumeration",
    );
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_split_exists_call_uses_witness_once_for_dispatch_bindings() {
    let src = r#"
---- MODULE Test ----
VARIABLE x

Proc == {1, 2}

Act(i) == x' = i

Next == \E self \in Proc : Act(self)
====
"#;

    let (ctx, next_def) = load_and_find_op(src, "Next");
    let actions = split_action_instances(&ctx, &next_def.body).unwrap();
    assert_eq!(actions.len(), 2);

    let mut seen = actions
        .iter()
        .map(|action| {
            assert_eq!(action.name.as_deref(), Some("Act"));
            assert_eq!(
                action.bindings.len(),
                1,
                "outer EXISTS witness should be the dispatch binding without a duplicate formal"
            );
            assert_eq!(action.bindings[0].0.as_ref(), "self");
            assert_eq!(
                action.formal_bindings.len(),
                1,
                "operator formal still needs to be available for arity specialization"
            );
            assert_eq!(action.formal_bindings[0].0.as_ref(), "i");
            let binding_value = action.bindings[0]
                .1
                .as_i64()
                .expect("binding should be int");
            let formal_value = action.formal_bindings[0]
                .1
                .as_i64()
                .expect("formal should be int");
            assert_eq!(binding_value, formal_value);
            binding_value
        })
        .collect::<Vec<_>>();
    seen.sort_unstable();
    assert_eq!(seen, vec![1, 2]);
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_split_keeps_distinct_actual_witnesses_with_equal_values() {
    let src = r#"
---- MODULE Test ----
VARIABLE x

Proc == {1}

Pair(a, b) == x' = a + b

Next == \E p \in Proc : \E q \in Proc : Pair(p, q)
====
"#;

    let (ctx, next_def) = load_and_find_op(src, "Next");
    let actions = split_action_instances(&ctx, &next_def.body).unwrap();
    assert_eq!(actions.len(), 1);
    assert_eq!(actions[0].name.as_deref(), Some("Pair"));

    let binding_names = actions[0]
        .bindings
        .iter()
        .map(|(name, value)| (name.as_ref(), value.as_i64()))
        .collect::<Vec<_>>();
    assert_eq!(binding_names, vec![("p", Some(1)), ("q", Some(1))]);

    let formal_names = actions[0]
        .formal_bindings
        .iter()
        .map(|(name, value)| (name.as_ref(), value.as_i64()))
        .collect::<Vec<_>>();
    assert_eq!(formal_names, vec![("a", Some(1)), ("b", Some(1))]);
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_split_literal_call_keeps_formal_as_dispatch_binding() {
    let src = r#"
---- MODULE Test ----
VARIABLE x

Act(i) == x' = i

Next == Act(7)
====
"#;

    let (ctx, next_def) = load_and_find_op(src, "Next");
    let actions = split_action_instances(&ctx, &next_def.body).unwrap();
    assert_eq!(actions.len(), 1);
    assert_eq!(actions[0].name.as_deref(), Some("Act"));
    assert_eq!(
        actions[0].bindings.len(),
        1,
        "literal actuals still need a dispatch binding so arity-positive actions specialize"
    );
    assert_eq!(actions[0].bindings[0].0.as_ref(), "i");
    assert_eq!(actions[0].bindings[0].1.as_i64(), Some(7));
    assert_eq!(actions[0].formal_bindings, actions[0].bindings);
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_enumerate_successors_by_action_instance_attributes_bindings() {
    let src = r#"
---- MODULE Test ----
VARIABLE x

Op(a) == x' = a

Next == \E y \in {1, 2} : Op(y)
====
"#;

    let (mut ctx, next_def) = load_and_find_op(src, "Next");
    let state = State::from_pairs([("x", Value::int(0))]);
    let vars = vec![Arc::from("x")];

    let per = enumerate_successors_by_action_instance(&mut ctx, &next_def, &state, &vars)
        .expect("enumerate successors");
    assert_eq!(per.len(), 2);

    let mut xs = Vec::new();
    for inst in per {
        assert_eq!(inst.successors.len(), 1);
        let succ = &inst.successors[0];
        let x = succ
            .vars()
            .find_map(|(k, v)| (k.as_ref() == "x").then_some(v))
            .and_then(tla_value::Value::as_i64)
            .expect("successor has x=int(..)");

        let y = inst
            .instance
            .bindings
            .iter()
            .find_map(|(k, v)| (k.as_ref() == "y").then_some(v))
            .and_then(tla_value::Value::as_i64)
            .expect("instance has y=int(..)");
        let a = inst
            .instance
            .formal_bindings
            .iter()
            .find_map(|(k, v)| (k.as_ref() == "a").then_some(v))
            .and_then(tla_value::Value::as_i64)
            .expect("instance has formal a=int(..)");
        assert_eq!(y, x);
        assert_eq!(a, x);
        xs.push(x);
    }

    xs.sort_unstable();
    assert_eq!(xs, vec![1, 2]);
}

/// Regression test for #1886: split_action_instances correctly handles
/// const-level SetFilter domains that ARE enumerable.
///
/// When `try_eval_const_level` successfully evaluates a SetFilter domain
/// (e.g., `{n \in {1,2} : n > 0}` → `{1,2}`), the resulting concrete set
/// is iterable and produces one action per element. This verifies the
/// happy path still works after #1886 error discrimination changes.
#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_1886_split_action_instances_const_setfilter_splits() {
    // Valid SetFilter domain evaluates at const-level to a concrete set.
    let src = r#"
---- MODULE SplitSetFilter ----
EXTENDS Integers
VARIABLE x
Next == \E v \in {n \in {1, 2} : n > 0} : x' = v
====
"#;
    let (ctx, next_def) = load_and_find_op(src, "Next");
    let result = split_action_instances(&ctx, &next_def.body);
    assert!(
        result.is_ok(),
        "#1886: valid SetFilter domain should not error: {:?}",
        result
    );
    let actions = result.unwrap();
    // {n \in {1,2} : n > 0} evaluates to {1,2} → 2 action instances.
    assert_eq!(
        actions.len(),
        2,
        "#1886: const-level SetFilter domain should produce 2 action instances"
    );
}

/// Regression test for #1886: split_action_instances returns Ok(leaf) when
/// SetFilter evaluation fails in try_eval_const_level (error caught before
/// reaching eval_iter_set).
///
/// The broken SetFilter `{n \in {1,2} : 1}` (non-boolean predicate) fails
/// during try_eval_const_level's eager evaluation, which catches the error
/// and returns None → leaf action. The #1886 error discrimination in
/// `try_split_exists_all_bounds` is defensive code for the case where
/// try_eval_const_level succeeds but eval_iter_set fails.
#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_1886_split_action_instances_broken_setfilter_leaf() {
    let src = r#"
---- MODULE SplitBrokenFilter ----
EXTENDS Integers
VARIABLE x
Next == \E v \in {n \in {1, 2} : 1} : x' = v
====
"#;
    let (ctx, next_def) = load_and_find_op(src, "Next");
    let result = split_action_instances(&ctx, &next_def.body);
    assert!(
        result.is_ok(),
        "#1886: broken SetFilter caught by try_eval_const_level should produce \
         leaf action, not error: {:?}",
        result
    );
    let actions = result.unwrap();
    // try_eval_const_level returns None for the broken SetFilter,
    // so the EXISTS is not split → 1 leaf action.
    assert_eq!(
        actions.len(),
        1,
        "#1886: broken SetFilter domain should fall back to 1 leaf action"
    );
}

/// Regression test for #1886: split_action_instances correctly falls back
/// for non-enumerable domains (defer-class errors) without propagating.
#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_1886_split_action_instances_defers_non_enumerable() {
    // Nat is not const-level (try_eval_const_level returns None), so the
    // split should return false → leaf action. This tests the existing
    // behavior is preserved: non-enumerable domains don't error.
    let src = r#"
---- MODULE SplitDefer ----
EXTENDS Naturals
VARIABLE x
Next == \E v \in Nat : x' = v
====
"#;
    let (ctx, next_def) = load_and_find_op(src, "Next");
    let result = split_action_instances(&ctx, &next_def.body);
    assert!(
        result.is_ok(),
        "#1886: non-enumerable domain should fall back to leaf, not error: {:?}",
        result
    );
    let actions = result.unwrap();
    // Should produce 1 leaf action (the EXISTS is not split).
    assert_eq!(
        actions.len(),
        1,
        "#1886: non-splittable EXISTS should produce 1 leaf action"
    );
}

/// Regression test for #1920: Verify that split_action_instances PROPAGATES
/// fatal errors from eval_iter_set when try_eval_const_level succeeds.
///
/// This exercises the Defer-vs-Fatal discrimination in `try_split_exists_all_bounds`
/// (line 323-329). The domain `{n \in SUBSET {1,2} : 1}` uses SUBSET (a lazy set),
/// so `try_eval_const_level` returns `Some(Value::SetPred(...))` without evaluating
/// the predicate. Then `eval_iter_set` materializes the SetPred, evaluating predicate
/// `1` for each element → TypeError(expected: BOOLEAN) → Fatal → propagated as Err.
///
/// This is the ONLY action_instance test that exercises the Fatal return path at
/// line 328. The other test_1886_* tests all have try_eval_const_level catching
/// errors before eval_iter_set runs.
#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_1920_split_fatal_error_propagated_from_lazy_setpred() {
    // SUBSET {1,2} is lazy → SetFilter over it creates a SetPred value.
    // try_eval_const_level returns Some(SetPred) — construction succeeds.
    // eval_iter_set evaluates the predicate `1` (not a boolean) → Fatal TypeError.
    let src = r#"
---- MODULE SplitFatalSetPred ----
EXTENDS FiniteSets
VARIABLE x
Next == \E v \in {n \in SUBSET {1, 2} : 1} : x' = v
====
"#;
    let (ctx, next_def) = load_and_find_op(src, "Next");
    let result = split_action_instances(&ctx, &next_def.body);
    assert!(
        result.is_err(),
        "#1920: SetPred with non-boolean predicate should propagate \
         Fatal error, not silently defer to leaf action. Got Ok({:?})",
        result.as_ref().ok().map(std::vec::Vec::len)
    );
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_enumerate_successors_by_action_instance_restores_ctx_env() {
    let src = r#"
---- MODULE Test ----
VARIABLE x

Op(a) == x' = a

Next == \E y \in {1, 2} : Op(y)
====
"#;

    let (mut ctx, next_def) = load_and_find_op(src, "Next");
    let state = State::from_pairs([("x", Value::int(0))]);
    let vars = vec![Arc::from("x")];

    assert!(ctx.env().get("x").is_none(), "precondition: x unbound");
    let _ = enumerate_successors_by_action_instance(&mut ctx, &next_def, &state, &vars)
        .expect("enumerate successors");
    assert!(ctx.env().get("x").is_none(), "ctx.env should be restored");
}
