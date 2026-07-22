// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

use super::*;

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_split_named_instance_real_action_subscript_as_safety() {
    let inner = inner_named_instance_module();
    let outer = outer_named_instance_module();
    let config = named_instance_config("Refines");
    let checker = ModelChecker::new_with_extends(&outer, &[&inner], &config);

    let (safety_parts, liveness_expr) = checker
        .separate_safety_liveness_parts("Refines", operator_body(&outer, "Refines"))
        .expect("named-instance property should split successfully");

    assert_eq!(
        safety_parts.init_terms.len(),
        1,
        "named-instance Spec should keep one init predicate"
    );
    assert_eq!(
        safety_parts.always_terms.len(),
        1,
        "named-instance [][Next]_vars should stay on the safety action lane"
    );
    assert!(
        liveness_expr.is_none(),
        "named-instance [][Next]_vars must not leak to the liveness checker"
    );
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_split_named_instance_expanded_action_stays_liveness() {
    let inner = inner_named_instance_module();
    let outer = outer_named_instance_module();
    let config = named_instance_config("Expanded");
    let checker = ModelChecker::new_with_extends(&outer, &[&inner], &config);

    let (safety_parts, liveness_expr) = checker
        .separate_safety_liveness_parts("Expanded", operator_body(&outer, "Expanded"))
        .expect("expanded named-instance property should split successfully");

    assert!(
        safety_parts.init_terms.is_empty(),
        "expanded [](A \\/ UNCHANGED vars) should not extract init predicates"
    );
    assert!(
        safety_parts.always_terms.is_empty(),
        "expanded [](A \\/ UNCHANGED vars) must not be misclassified as safety"
    );
    assert!(
        liveness_expr.is_some(),
        "expanded [](A \\/ UNCHANGED vars) should remain on the liveness/rejection path"
    );
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_split_named_instance_raw_action_stays_liveness() {
    let inner = inner_named_instance_module();
    let outer = outer_named_instance_module();
    let config = named_instance_config("RawRefines");
    let checker = ModelChecker::new_with_extends(&outer, &[&inner], &config);

    let (safety_parts, liveness_expr) = checker
        .separate_safety_liveness_parts("RawRefines", operator_body(&outer, "RawRefines"))
        .expect("named-instance raw action property should split successfully");

    assert!(
        safety_parts.init_terms.is_empty(),
        "named-instance []Next should not extract init predicates"
    );
    assert!(
        safety_parts.always_terms.is_empty(),
        "named-instance []Next must not be misclassified as safety"
    );
    assert!(
        liveness_expr.is_some(),
        "named-instance []Next should remain on the liveness/rejection path"
    );
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_split_named_instance_init_substitution_keeps_structural_action_promotion() {
    let inner = inner_named_instance_module();
    let outer = outer_named_instance_init_split_module();
    let config = named_instance_config("Refines");
    let checker = ModelChecker::new_with_extends(&outer, &[&inner], &config);

    let (safety_parts, liveness_expr) = checker
        .separate_safety_liveness_parts("Refines", operator_body(&outer, "Refines"))
        .expect("named-instance property with substituted init split should still split");

    assert_eq!(
        safety_parts.init_terms.len(),
        2,
        "substituting Init <- (InitLeft /\\ InitRight) should preserve both init conjuncts"
    );
    assert_eq!(
        safety_parts.always_terms.len(),
        1,
        "substituting one source conjunct into two init conjuncts must not drop [][Next]_vars"
    );
    assert!(
        liveness_expr.is_none(),
        "named-instance structural [][Next]_vars should remain fully handled after init expansion"
    );
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_split_ewd998_chanid_property_action_stays_instance_qualified() {
    let inner = parse_module(
        r#"
---- MODULE EWD998ChanShape ----
EXTENDS Integers
CONSTANT N, inbox
VARIABLE chan
Node == 0..N-1
vars == <<chan>>
Init == chan = inbox[0]
Next == \E n \in Node : chan' = inbox[n]
Spec == Init /\ [][Next]_vars
===="#,
        FileId(3),
    );
    let outer = parse_module(
        r#"
---- MODULE EWD998ChanIDShape ----
EXTENDS Integers, Functions
VARIABLE chan

ModelNode(i) ==
  CASE i = 0 -> TLCModelValue("n1")
    [] i = 1 -> TLCModelValue("n2")
    [] i = 2 -> TLCModelValue("n3")
    [] i = 3 -> TLCModelValue("n4")
    [] i = 4 -> TLCModelValue("n5")
    [] OTHER -> TLCModelValue("bad")

Node == {ModelNode(0), ModelNode(1), ModelNode(2), ModelNode(3), ModelNode(4)}
NodeIndex(n) ==
  CASE n = ModelNode(0) -> 0
    [] n = ModelNode(1) -> 1
    [] n = ModelNode(2) -> 2
    [] n = ModelNode(3) -> 3
    [] n = ModelNode(4) -> 4
    [] OTHER -> -1

nat2node == [i \in 0..4 |-> CHOOSE n \in Node : NodeIndex(n) = i]
node2nat == AntiFunction(nat2node)
EWD998ChanInbox == [n \in Node |-> node2nat[n]]
Node2Nat(fcn) == [i \in 0..4 |-> fcn[nat2node[i]]]

Init == chan = Node2Nat(EWD998ChanInbox)[0]
Next == \E n \in Node : chan' = node2nat[n]
I == INSTANCE EWD998ChanShape WITH N <- 5, inbox <- Node2Nat(EWD998ChanInbox)
Refines == I!Spec
===="#,
        FileId(4),
    );
    let config = named_instance_config("Refines");
    let checker = ModelChecker::new_with_extends(&outer, &[&inner], &config);

    let (safety_parts, liveness_expr) = checker
        .separate_safety_liveness_parts("Refines", operator_body(&outer, "Refines"))
        .expect("EWD998ChanID-shaped named-instance property should split");

    assert_eq!(safety_parts.init_terms.len(), 1);
    assert_eq!(safety_parts.always_terms.len(), 1);
    assert!(
        liveness_expr.is_none(),
        "real [][Next]_vars instance PROPERTY term should stay on the safety action lane"
    );
    assert_qualified_action_subscript(&safety_parts.always_terms[0], "I");
    assert_no_unqualified_inner_action_names(&safety_parts.always_terms[0]);
}

fn assert_qualified_action_subscript(expr: &Spanned<Expr>, instance: &str) {
    match &expr.node {
        Expr::Or(left, right) => {
            assert_module_ref(left, instance, "Next");
            match &right.node {
                Expr::Unchanged(inner) => assert_module_ref(inner, instance, "vars"),
                other => panic!("right side should be UNCHANGED I!vars, got {other:?}"),
            }
        }
        other => panic!("PROPERTY action should remain I!Next \\/ UNCHANGED I!vars, got {other:?}"),
    }
}

fn assert_module_ref(expr: &Spanned<Expr>, instance: &str, op_name: &str) {
    match &expr.node {
        Expr::ModuleRef(ModuleTarget::Named(name), actual, args) => {
            assert_eq!(name, instance);
            assert_eq!(actual, op_name);
            assert!(
                args.is_empty(),
                "qualified zero-arg instance operator should not gain arguments"
            );
        }
        other => panic!("expected {instance}!{op_name}, got {other:?}"),
    }
}

fn assert_no_unqualified_inner_action_names(expr: &Spanned<Expr>) {
    match &expr.node {
        Expr::Ident(name, _) => {
            assert!(
                name != "Next" && name != "vars" && name != "Node" && name != "inbox",
                "promoted PROPERTY action term mixed in unqualified inner name {name}"
            );
        }
        Expr::And(left, right)
        | Expr::Or(left, right)
        | Expr::Implies(left, right)
        | Expr::Equiv(left, right) => {
            assert_no_unqualified_inner_action_names(left);
            assert_no_unqualified_inner_action_names(right);
        }
        Expr::Not(inner)
        | Expr::Prime(inner)
        | Expr::Unchanged(inner)
        | Expr::Enabled(inner)
        | Expr::Always(inner)
        | Expr::Eventually(inner) => {
            assert_no_unqualified_inner_action_names(inner);
        }
        Expr::Apply(op, args) => {
            assert_no_unqualified_inner_action_names(op);
            for arg in args {
                assert_no_unqualified_inner_action_names(arg);
            }
        }
        Expr::ModuleRef(_, _, args) => {
            for arg in args {
                assert_no_unqualified_inner_action_names(arg);
            }
        }
        Expr::WeakFair(subscript, action) | Expr::StrongFair(subscript, action) => {
            assert_no_unqualified_inner_action_names(subscript);
            assert_no_unqualified_inner_action_names(action);
        }
        _ => {}
    }
}
