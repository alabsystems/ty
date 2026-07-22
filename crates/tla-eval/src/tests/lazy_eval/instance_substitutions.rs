// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

use super::*;
use crate::binding_chain::{BindingChain, BindingValue, LazyBinding};
use std::sync::Arc;
use tla_core::ast::Substitution;
use tla_core::name_intern::{intern_name, NameId};
use tla_core::OpEnv;

fn zero_arg_operator(name: &str, body: Expr) -> Arc<tla_core::ast::OperatorDef> {
    Arc::new(tla_core::ast::OperatorDef {
        name: Spanned::dummy(name.to_string()),
        params: vec![],
        body: Spanned::dummy(body),
        local: false,
        contains_prime: false,
        guards_depend_on_prime: false,
        has_primed_param: false,
        is_recursive: false,
        self_call_count: 0,
    })
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_instance_lazy_binding_preserves_state_deps_after_untracked_force() {
    use crate::cache::eval_with_dep_tracking;

    let mut ctx = EvalCtx::new();
    let x_idx = ctx.register_var("x");
    let state = vec![Value::int(7)];
    ctx.bind_state_array(&state);

    let subs = std::sync::Arc::new(vec![Substitution {
        from: Spanned::dummy("y".to_string()),
        to: Spanned::dummy(Expr::StateVar("x".to_string(), x_idx.0, intern_name("x"))),
    }]);

    let mut sub_ctx = ctx.clone();
    sub_ctx.bindings = build_lazy_subst_bindings(&ctx.bindings, &subs);
    let stable = sub_ctx.stable_mut();
    stable.instance_substitutions = Some(subs);
    // Part of #3099: Invalidate scope_ids since we wrote instance_substitutions directly.
    stable.scope_ids.instance_substitutions = crate::cache::scope_ids::INVALIDATED;
    stable.eager_subst_bindings = Some(std::sync::Arc::new(vec![]));

    let y_expr = Spanned::dummy(Expr::Ident("y".to_string(), intern_name("y")));

    let first = eval(&sub_ctx, &y_expr).expect("initial lazy substitution force should succeed");
    assert_eq!(first, Value::int(7));

    let (second, deps) = eval_with_dep_tracking(&sub_ctx, &y_expr)
        .expect("cached lazy substitution should retain state deps for tracked reads");
    assert_eq!(second, Value::int(7));
    assert_eq!(
        deps.state.len(),
        1,
        "tracked read of cached lazy substitution must recover the underlying state dependency"
    );
    assert!(deps.next.is_empty());
    assert!(deps.local.is_empty());
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_lazy_instance_substitution_captures_definition_site_local_ops() {
    let root_ctx = EvalCtx::new();

    let mut def_site_ops = OpEnv::new();
    def_site_ops.insert(
        "Node".into(),
        zero_arg_operator("Node", Expr::Int(42.into())),
    );
    let def_site_ctx = root_ctx.with_instance_scope(def_site_ops, vec![]);

    let subs = Arc::new(vec![Substitution {
        from: Spanned::dummy("x".to_string()),
        to: Spanned::dummy(Expr::Ident("Node".to_string(), NameId::INVALID)),
    }]);

    let mut sub_ctx = def_site_ctx.clone();
    let lazy = LazyBinding::new_with_local_ops(
        std::ptr::addr_of!(subs[0].to),
        &def_site_ctx.bindings,
        def_site_ctx.local_ops.clone(),
    );
    sub_ctx.bindings = BindingChain::empty().cons_with_deps(
        intern_name("x"),
        BindingValue::Lazy(Box::new(lazy)),
        OpEvalDeps::default(),
    );

    let mut wrong_pre_scope_ops = OpEnv::new();
    wrong_pre_scope_ops.insert(
        "Node".into(),
        zero_arg_operator("Node", Expr::Int(7.into())),
    );
    let mut forcing_ops = OpEnv::new();
    forcing_ops.insert(
        "Node".into(),
        zero_arg_operator("Node", Expr::Int((-1).into())),
    );

    let stable = sub_ctx.stable_mut();
    stable.instance_substitutions = Some(subs);
    stable.scope_ids.instance_substitutions = crate::cache::scope_ids::INVALIDATED;
    stable.eager_subst_bindings = Some(Arc::new(vec![]));
    stable.local_ops = Some(Arc::new(forcing_ops));
    stable.pre_scope_local_ops = Some(Arc::new(wrong_pre_scope_ops));
    stable.scope_ids.local_ops = crate::cache::scope_ids::INVALIDATED;

    let x_expr = Spanned::dummy(Expr::Ident("x".to_string(), intern_name("x")));
    let value = eval(&sub_ctx, &x_expr)
        .expect("lazy substitution must resolve with captured definition-site local_ops");
    assert_eq!(
        value,
        Value::int(42),
        "lazy INSTANCE substitution RHS must use the local_ops captured when the lazy binding was built"
    );
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_eval_module_ref_resolves_unqualified_ops_within_instance_module() {
    // Regression: when evaluating an instance reference `Inst!Next(1)`, operator calls inside
    // `Next` must resolve within the instanced module, not against the current module.
    //
    // This matters when both modules define the same operator name with different arities.
    let mod_m = r#"
---- MODULE M ----
EXTENDS Integers
SendMsg(x) == x + 1
Next(i) == SendMsg(i)
===="#;
    let mod_main = r#"
---- MODULE Main ----
EXTENDS Integers
SendMsg(x, y) == x + y
Inst == INSTANCE M
Op == Inst!Next(1)
===="#;

    let tree_m = parse_to_syntax_tree(mod_m);
    let lower_m = lower(FileId(0), &tree_m);
    assert!(
        lower_m.errors.is_empty(),
        "lower M errors: {:?}",
        lower_m.errors
    );
    let module_m = lower_m.module.expect("lower produced no module M");

    let tree_main = parse_to_syntax_tree(mod_main);
    let lower_main = lower(FileId(0), &tree_main);
    assert!(
        lower_main.errors.is_empty(),
        "lower Main errors: {:?}",
        lower_main.errors
    );
    let module_main = lower_main.module.expect("lower produced no module Main");

    let mut ctx = EvalCtx::new();
    ctx.load_module(&module_main);
    ctx.load_instance_module("M".to_string(), &module_m);

    let op_def = ctx.get_op("Op").expect("Op not found").clone();
    let v = eval(&ctx, &op_def.body).unwrap();
    assert_eq!(v, Value::int(2));
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_chained_module_ref_cache_respects_outer_instance_substitutions() {
    let mod_b = r#"
---- MODULE B ----
Val == x
===="#;
    let mod_a = r#"
---- MODULE A ----
BInst == INSTANCE B WITH x <- y
===="#;
    let mod_main = r#"
---- MODULE Main ----
AInst == INSTANCE A WITH y <- z
z == 0
Op == AInst!BInst!Val
===="#;

    let tree_b = parse_to_syntax_tree(mod_b);
    let lower_b = lower(FileId(0), &tree_b);
    assert!(
        lower_b.errors.is_empty(),
        "lower B errors: {:?}",
        lower_b.errors
    );
    let module_b = lower_b.module.expect("lower produced no module B");

    let tree_a = parse_to_syntax_tree(mod_a);
    let lower_a = lower(FileId(0), &tree_a);
    assert!(
        lower_a.errors.is_empty(),
        "lower A errors: {:?}",
        lower_a.errors
    );
    let module_a = lower_a.module.expect("lower produced no module A");

    let tree_main = parse_to_syntax_tree(mod_main);
    let lower_main = lower(FileId(0), &tree_main);
    assert!(
        lower_main.errors.is_empty(),
        "lower Main errors: {:?}",
        lower_main.errors
    );
    let module_main = lower_main.module.expect("lower produced no module Main");

    let mut ctx = EvalCtx::new();
    ctx.load_module(&module_main);
    ctx.load_instance_module("A".to_string(), &module_a);
    ctx.load_instance_module("B".to_string(), &module_b);

    let op_def = ctx.get_op("Op").expect("Op not found").clone();

    // Evaluate once to populate the chained reference cache.
    let ctx_z1 = ctx.with_instance_substitutions(vec![Substitution {
        from: Spanned::dummy("z".to_string()),
        to: Spanned::dummy(Expr::Int(1.into())),
    }]);
    let v1 = eval(&ctx_z1, &op_def.body).expect("first chained module ref eval should succeed");
    assert_eq!(v1, Value::int(1));

    // Evaluate again with a different outer substitution. If cache entries are keyed
    // too coarsely, this would incorrectly return 1.
    let ctx_z2 = ctx.with_instance_substitutions(vec![Substitution {
        from: Spanned::dummy("z".to_string()),
        to: Spanned::dummy(Expr::Int(2.into())),
    }]);
    let v2 = eval(&ctx_z2, &op_def.body).expect("second chained module ref eval should succeed");
    assert_eq!(v2, Value::int(2));
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_lazy_instance_substitution_uses_outer_operator_scope() {
    // Regression for #3056 Phase 5: forcing a lazy INSTANCE substitution must
    // clear the current instance's local operator scope. Otherwise the inner
    // module's `Node` would shadow Main's `Node`, and `I!Val` would incorrectly
    // evaluate to 1 instead of 2.
    let mod_inner = r#"
---- MODULE Inner ----
CONSTANT x
Node == 1
Val == x
===="#;
    let mod_main = r#"
---- MODULE Main ----
Node == 2
I == INSTANCE Inner WITH x <- Node
Op == I!Val
===="#;

    let tree_inner = parse_to_syntax_tree(mod_inner);
    let lower_inner = lower(FileId(0), &tree_inner);
    assert!(
        lower_inner.errors.is_empty(),
        "lower Inner errors: {:?}",
        lower_inner.errors
    );
    let module_inner = lower_inner.module.expect("lower produced no module Inner");

    let tree_main = parse_to_syntax_tree(mod_main);
    let lower_main = lower(FileId(0), &tree_main);
    assert!(
        lower_main.errors.is_empty(),
        "lower Main errors: {:?}",
        lower_main.errors
    );
    let module_main = lower_main.module.expect("lower produced no module Main");

    let mut ctx = EvalCtx::new();
    ctx.load_module(&module_main);
    ctx.load_instance_module("Inner".to_string(), &module_inner);

    let op_def = ctx.get_op("Op").expect("Op not found").clone();
    let value = eval(&ctx, &op_def.body).expect("lazy instance substitution should evaluate");
    assert_eq!(
        value,
        Value::int(2),
        "lazy substitution RHS must resolve Node in the outer module scope"
    );
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_chained_lazy_instance_substitution_keeps_definition_site_ops_across_deeper_instance() {
    let mod_d = r#"
---- MODULE D ----
CONSTANT w
Val == w
===="#;
    let mod_c = r#"
---- MODULE C ----
CONSTANT z
Node == 30
DInst == INSTANCE D WITH w <- z
CVal == DInst!Val
===="#;
    let mod_b = r#"
---- MODULE B ----
CONSTANT x
Node == 20
CInst == INSTANCE C WITH z <- x
===="#;
    let mod_a = r#"
---- MODULE A ----
Node == 10
BInst == INSTANCE B WITH x <- Node
Run == BInst!CInst!CVal
===="#;
    let mod_main = r#"
---- MODULE Main ----
Node == 100
AInst == INSTANCE A
Op == AInst!Run
===="#;

    let tree_d = parse_to_syntax_tree(mod_d);
    let lower_d = lower(FileId(0), &tree_d);
    assert!(
        lower_d.errors.is_empty(),
        "lower D errors: {:?}",
        lower_d.errors
    );
    let module_d = lower_d.module.expect("lower produced no module D");

    let tree_c = parse_to_syntax_tree(mod_c);
    let lower_c = lower(FileId(0), &tree_c);
    assert!(
        lower_c.errors.is_empty(),
        "lower C errors: {:?}",
        lower_c.errors
    );
    let module_c = lower_c.module.expect("lower produced no module C");

    let tree_b = parse_to_syntax_tree(mod_b);
    let lower_b = lower(FileId(0), &tree_b);
    assert!(
        lower_b.errors.is_empty(),
        "lower B errors: {:?}",
        lower_b.errors
    );
    let module_b = lower_b.module.expect("lower produced no module B");

    let tree_a = parse_to_syntax_tree(mod_a);
    let lower_a = lower(FileId(0), &tree_a);
    assert!(
        lower_a.errors.is_empty(),
        "lower A errors: {:?}",
        lower_a.errors
    );
    let module_a = lower_a.module.expect("lower produced no module A");

    let tree_main = parse_to_syntax_tree(mod_main);
    let lower_main = lower(FileId(0), &tree_main);
    assert!(
        lower_main.errors.is_empty(),
        "lower Main errors: {:?}",
        lower_main.errors
    );
    let module_main = lower_main.module.expect("lower produced no module Main");

    let mut ctx = EvalCtx::new();
    ctx.load_module(&module_main);
    ctx.load_instance_module("A".to_string(), &module_a);
    ctx.load_instance_module("B".to_string(), &module_b);
    ctx.load_instance_module("C".to_string(), &module_c);
    ctx.load_instance_module("D".to_string(), &module_d);

    let op_def = ctx.get_op("Op").expect("Op not found").clone();
    let value = eval(&ctx, &op_def.body)
        .expect("chained lazy instance substitution should preserve definition-site ops");
    assert_eq!(
        value,
        Value::int(10),
        "substitution RHS must resolve Node in A's definition-site operator scope"
    );
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_chained_instance_implicit_operator_uses_parent_scope_not_root_collision() {
    let mod_leaf = r#"
---- MODULE Leaf ----
CONSTANT N
VARIABLE terminationDetected
Node == 0..N-1
Live == \A n \in Node : n \in Node /\ terminationDetected
===="#;
    let mod_parent = r#"
---- MODULE Parent ----
N == 2
terminationDetected == TRUE
TD == INSTANCE Leaf
===="#;
    let mod_main = r#"
---- MODULE Main ----
terminationDetected == FALSE
ParentInst == INSTANCE Parent
Op == ParentInst!TD!Live
===="#;

    let tree_leaf = parse_to_syntax_tree(mod_leaf);
    let lower_leaf = lower(FileId(0), &tree_leaf);
    assert!(
        lower_leaf.errors.is_empty(),
        "lower Leaf errors: {:?}",
        lower_leaf.errors
    );
    let module_leaf = lower_leaf.module.expect("lower produced no module Leaf");

    let tree_parent = parse_to_syntax_tree(mod_parent);
    let lower_parent = lower(FileId(0), &tree_parent);
    assert!(
        lower_parent.errors.is_empty(),
        "lower Parent errors: {:?}",
        lower_parent.errors
    );
    let module_parent = lower_parent
        .module
        .expect("lower produced no module Parent");

    let tree_main = parse_to_syntax_tree(mod_main);
    let lower_main = lower(FileId(0), &tree_main);
    assert!(
        lower_main.errors.is_empty(),
        "lower Main errors: {:?}",
        lower_main.errors
    );
    let module_main = lower_main.module.expect("lower produced no module Main");

    let mut ctx = EvalCtx::new();
    ctx.load_module(&module_main);
    ctx.load_instance_module("Parent".to_string(), &module_parent);
    ctx.load_instance_module("Leaf".to_string(), &module_leaf);

    let op_def = ctx.get_op("Op").expect("Op not found").clone();
    let value =
        eval(&ctx, &op_def.body).expect("chained implicit operator substitution should evaluate");
    assert_eq!(
        value,
        Value::Bool(true),
        "Leaf variable terminationDetected must bind to Parent!terminationDetected, not Main!terminationDetected"
    );
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_chained_instance_inherited_substitution_rhs_uses_parent_scope_not_root_collision() {
    let mod_leaf = r#"
---- MODULE Leaf ----
CONSTANT N
VARIABLE pending
Node == 0..N-1
Use == \A i \in Node : pending[i] = i
===="#;
    let mod_parent = r#"
---- MODULE Parent ----
N == 2
Node == 0..N-1
idx == [i \in Node |-> i]
TD == INSTANCE Leaf WITH pending <- [i \in Node |-> idx[i]]
===="#;
    let mod_main = r#"
---- MODULE Main ----
Node == {10, 20}
idx == [i \in 0..1 |-> i]
ParentInst == INSTANCE Parent
Op == ParentInst!TD!Use
===="#;

    let tree_leaf = parse_to_syntax_tree(mod_leaf);
    let lower_leaf = lower(FileId(0), &tree_leaf);
    assert!(
        lower_leaf.errors.is_empty(),
        "lower Leaf errors: {:?}",
        lower_leaf.errors
    );
    let module_leaf = lower_leaf.module.expect("lower produced no module Leaf");

    let tree_parent = parse_to_syntax_tree(mod_parent);
    let lower_parent = lower(FileId(0), &tree_parent);
    assert!(
        lower_parent.errors.is_empty(),
        "lower Parent errors: {:?}",
        lower_parent.errors
    );
    let module_parent = lower_parent
        .module
        .expect("lower produced no module Parent");

    let tree_main = parse_to_syntax_tree(mod_main);
    let lower_main = lower(FileId(0), &tree_main);
    assert!(
        lower_main.errors.is_empty(),
        "lower Main errors: {:?}",
        lower_main.errors
    );
    let module_main = lower_main.module.expect("lower produced no module Main");

    let mut ctx = EvalCtx::new();
    ctx.load_module(&module_main);
    ctx.load_instance_module("Parent".to_string(), &module_parent);
    ctx.load_instance_module("Leaf".to_string(), &module_leaf);

    let op_def = ctx.get_op("Op").expect("Op not found").clone();
    let value = eval(&ctx, &op_def.body)
        .expect("chained inherited substitution RHS should use parent operator scope");
    assert_eq!(
        value,
        Value::Bool(true),
        "pending RHS must bind Node/idx from Parent, not same-name operators from Main"
    );
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_chained_instance_implicit_inherited_substitution_rhs_keeps_definition_site_scope() {
    let mod_leaf = r#"
---- MODULE Leaf ----
CONSTANT N
VARIABLE pending
Node == 0..N-1
Safe == \A i \in Node : pending[i] = i
===="#;
    let mod_mid = r#"
---- MODULE Mid ----
CONSTANT N
VARIABLE pending
Node == 0..N-1
TD == INSTANCE Leaf
===="#;
    let mod_parent = r#"
---- MODULE Parent ----
N == 2
Node == 0..N-1
idx == [i \in Node |-> i]
MidInst == INSTANCE Mid WITH pending <- [i \in Node |-> idx[i]]
===="#;
    let mod_main = r#"
---- MODULE Main ----
Node == {10, 20}
idx == [i \in 0..1 |-> i]
ParentInst == INSTANCE Parent
Op == ParentInst!MidInst!TD!Safe
===="#;

    let tree_leaf = parse_to_syntax_tree(mod_leaf);
    let lower_leaf = lower(FileId(0), &tree_leaf);
    assert!(
        lower_leaf.errors.is_empty(),
        "lower Leaf errors: {:?}",
        lower_leaf.errors
    );
    let module_leaf = lower_leaf.module.expect("lower produced no module Leaf");

    let tree_mid = parse_to_syntax_tree(mod_mid);
    let lower_mid = lower(FileId(0), &tree_mid);
    assert!(
        lower_mid.errors.is_empty(),
        "lower Mid errors: {:?}",
        lower_mid.errors
    );
    let module_mid = lower_mid.module.expect("lower produced no module Mid");

    let tree_parent = parse_to_syntax_tree(mod_parent);
    let lower_parent = lower(FileId(0), &tree_parent);
    assert!(
        lower_parent.errors.is_empty(),
        "lower Parent errors: {:?}",
        lower_parent.errors
    );
    let module_parent = lower_parent
        .module
        .expect("lower produced no module Parent");

    let tree_main = parse_to_syntax_tree(mod_main);
    let lower_main = lower(FileId(0), &tree_main);
    assert!(
        lower_main.errors.is_empty(),
        "lower Main errors: {:?}",
        lower_main.errors
    );
    let module_main = lower_main.module.expect("lower produced no module Main");

    let mut ctx = EvalCtx::new();
    ctx.load_module(&module_main);
    ctx.load_instance_module("Parent".to_string(), &module_parent);
    ctx.load_instance_module("Mid".to_string(), &module_mid);
    ctx.load_instance_module("Leaf".to_string(), &module_leaf);

    let op_def = ctx.get_op("Op").expect("Op not found").clone();
    let value = eval(&ctx, &op_def.body).expect(
        "chained implicit inherited substitution RHS should keep definition-site operator scope",
    );
    assert_eq!(
        value,
        Value::Bool(true),
        "inherited pending RHS must keep Parent's Node/idx when Leaf sees it through Mid!TD"
    );
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_chained_instance_inherited_variable_substitution_rhs_keeps_definition_site_scope() {
    let mod_leaf = r#"
---- MODULE Leaf ----
CONSTANT N
VARIABLE pending
Node == 0..N-1
Safe == \A i \in Node : pending[i] = i
===="#;
    let mod_mid = r#"
---- MODULE Mid ----
CONSTANT N
VARIABLE pending
TD == INSTANCE Leaf
===="#;
    let mod_parent = r#"
---- MODULE Parent ----
CONSTANT N
VARIABLE inbox
Node == 0..N-1
MidInst == INSTANCE Mid WITH pending <- [i \in Node |-> inbox[i]]
===="#;
    let mod_main = r#"
---- MODULE Main ----
EXTENDS Integers
Node == {"a", "b"}
nat2node[i \in 0..1] == IF i = 0 THEN "a" ELSE "b"
rootInbox == [n \in Node |-> IF n = "a" THEN 0 ELSE 1]
Node2Nat(f) == [i \in 0..1 |-> f[nat2node[i]]]
ParentInst == INSTANCE Parent WITH N <- 2, inbox <- Node2Nat(rootInbox)
Op == ParentInst!MidInst!TD!Safe
===="#;

    let tree_leaf = parse_to_syntax_tree(mod_leaf);
    let lower_leaf = lower(FileId(0), &tree_leaf);
    assert!(
        lower_leaf.errors.is_empty(),
        "lower Leaf errors: {:?}",
        lower_leaf.errors
    );
    let module_leaf = lower_leaf.module.expect("lower produced no module Leaf");

    let tree_mid = parse_to_syntax_tree(mod_mid);
    let lower_mid = lower(FileId(0), &tree_mid);
    assert!(
        lower_mid.errors.is_empty(),
        "lower Mid errors: {:?}",
        lower_mid.errors
    );
    let module_mid = lower_mid.module.expect("lower produced no module Mid");

    let tree_parent = parse_to_syntax_tree(mod_parent);
    let lower_parent = lower(FileId(0), &tree_parent);
    assert!(
        lower_parent.errors.is_empty(),
        "lower Parent errors: {:?}",
        lower_parent.errors
    );
    let module_parent = lower_parent
        .module
        .expect("lower produced no module Parent");

    let tree_main = parse_to_syntax_tree(mod_main);
    let lower_main = lower(FileId(0), &tree_main);
    assert!(
        lower_main.errors.is_empty(),
        "lower Main errors: {:?}",
        lower_main.errors
    );
    let module_main = lower_main.module.expect("lower produced no module Main");

    let mut ctx = EvalCtx::new();
    ctx.load_module(&module_main);
    ctx.load_instance_module("Parent".to_string(), &module_parent);
    ctx.load_instance_module("Mid".to_string(), &module_mid);
    ctx.load_instance_module("Leaf".to_string(), &module_leaf);

    let op_def = ctx.get_op("Op").expect("Op not found").clone();
    let value = eval(&ctx, &op_def.body).expect(
        "inherited variable substitution RHS should use the intermediate module operator scope",
    );
    assert_eq!(
        value,
        Value::Bool(true),
        "pending RHS must iterate over Parent's numeric Node while using the inherited inbox substitution"
    );
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_module_ref_operator_body_prefers_instance_node_over_root_constant() {
    let mod_parent = r#"
---- MODULE Parent ----
CONSTANT N
VARIABLE inbox
Node == 0..N-1
tpos == CHOOSE i \in Node : inbox[i] = 1
===="#;
    let mod_main = r#"
---- MODULE Main ----
Node == {"a", "b"}
nat2node[i \in 0..1] == IF i = 0 THEN "a" ELSE "b"
rootInbox == [n \in Node |-> IF n = "a" THEN 0 ELSE 1]
Node2Nat(f) == [i \in 0..1 |-> f[nat2node[i]]]
ParentInst == INSTANCE Parent WITH N <- 2, inbox <- Node2Nat(rootInbox)
Op == ParentInst!tpos
===="#;

    let tree_parent = parse_to_syntax_tree(mod_parent);
    let lower_parent = lower(FileId(0), &tree_parent);
    assert!(
        lower_parent.errors.is_empty(),
        "lower Parent errors: {:?}",
        lower_parent.errors
    );
    let module_parent = lower_parent
        .module
        .expect("lower produced no module Parent");

    let tree_main = parse_to_syntax_tree(mod_main);
    let lower_main = lower(FileId(0), &tree_main);
    assert!(
        lower_main.errors.is_empty(),
        "lower Main errors: {:?}",
        lower_main.errors
    );
    let module_main = lower_main.module.expect("lower produced no module Main");

    let mut ctx = EvalCtx::new();
    ctx.load_module(&module_main);
    ctx.load_instance_module("Parent".to_string(), &module_parent);

    let op_def = ctx.get_op("Op").expect("Op not found").clone();
    let value = eval(&ctx, &op_def.body)
        .expect("module-ref operator body should use Parent's numeric Node");
    assert_eq!(
        value,
        Value::int(1),
        "ParentInst!tpos must iterate Parent's numeric Node, not Main's symbolic Node"
    );
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_nested_instance_substitution_resolves_outer_instance_operator() {
    // Regression: nested named INSTANCE evaluation must keep the outer instance
    // module's operators available for substitution RHS evaluation.
    //
    // MidInst!Spec evaluates InnerInst!Spec inside Mid, where InnerInst maps
    // `pending <- Node`. `Node` is defined in Mid (not in Main). If nested
    // evaluation drops Mid's local operator scope, this fails with:
    // "Undefined variable: Node".
    let mod_inner = r#"
---- MODULE Inner ----
VARIABLE pending
Spec == pending = pending
===="#;
    let mod_mid = r#"
---- MODULE Mid ----
Node == {0, 1}
InnerInst == INSTANCE Inner WITH pending <- Node
Spec == InnerInst!Spec
===="#;
    let mod_main = r#"
---- MODULE Main ----
MidInst == INSTANCE Mid
Op == MidInst!Spec
===="#;

    let tree_inner = parse_to_syntax_tree(mod_inner);
    let lower_inner = lower(FileId(0), &tree_inner);
    assert!(
        lower_inner.errors.is_empty(),
        "lower Inner errors: {:?}",
        lower_inner.errors
    );
    let module_inner = lower_inner.module.expect("lower produced no module Inner");

    let tree_mid = parse_to_syntax_tree(mod_mid);
    let lower_mid = lower(FileId(0), &tree_mid);
    assert!(
        lower_mid.errors.is_empty(),
        "lower Mid errors: {:?}",
        lower_mid.errors
    );
    let module_mid = lower_mid.module.expect("lower produced no module Mid");

    let tree_main = parse_to_syntax_tree(mod_main);
    let lower_main = lower(FileId(0), &tree_main);
    assert!(
        lower_main.errors.is_empty(),
        "lower Main errors: {:?}",
        lower_main.errors
    );
    let module_main = lower_main.module.expect("lower produced no module Main");

    let mut ctx = EvalCtx::new();
    ctx.load_module(&module_main);
    ctx.load_instance_module("Mid".to_string(), &module_mid);
    ctx.load_instance_module("Inner".to_string(), &module_inner);

    let op_def = ctx.get_op("Op").expect("Op not found").clone();
    let v = eval(&ctx, &op_def.body).expect("nested instance substitution should evaluate");
    assert_eq!(v, Value::Bool(true));
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_substin_ident_substitution_preserves_outer_binding_chain() {
    // Regression for #3056 Phase 5 reachability: Expr::SubstIn installs
    // instance_substitutions without going through with_instance_scope().
    // The explicit-substitution fallback still runs here with a non-empty
    // binding chain, so clearing that chain would lose the outer binding for y.
    let expr = Spanned::dummy(Expr::SubstIn(
        vec![Substitution {
            from: Spanned::dummy("x".to_string()),
            to: Spanned::dummy(Expr::Ident(
                "y".to_string(),
                tla_core::name_intern::NameId::INVALID,
            )),
        }],
        Box::new(Spanned::dummy(Expr::Ident(
            "x".to_string(),
            tla_core::name_intern::NameId::INVALID,
        ))),
    ));

    let ctx = EvalCtx::new().bind_local("y", Value::int(7));
    let value = eval(&ctx, &expr).expect("SubstIn ident substitution should evaluate");
    assert_eq!(
        value,
        Value::int(7),
        "SubstIn fallback must preserve the outer binding chain for substitution RHS evaluation"
    );
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_substin_statevar_substitution_preserves_outer_binding_chain() {
    // Same reachability gap as the ident case, but through eval_statevar's
    // explicit-substitution fallback. The substitution RHS depends on an outer
    // local binding, so dropping the chain would incorrectly make y undefined.
    let mut ctx = EvalCtx::new();
    let x_idx = ctx.register_var("x");
    let state = vec![Value::int(99)];
    ctx.bind_state_array(&state);
    let ctx = ctx.bind_local("y", Value::int(11));

    let expr = Spanned::dummy(Expr::SubstIn(
        vec![Substitution {
            from: Spanned::dummy("x".to_string()),
            to: Spanned::dummy(Expr::Ident(
                "y".to_string(),
                tla_core::name_intern::NameId::INVALID,
            )),
        }],
        Box::new(Spanned::dummy(Expr::StateVar(
            "x".to_string(),
            x_idx.0,
            tla_core::name_intern::intern_name("x"),
        ))),
    ));

    let value = eval(&ctx, &expr).expect("SubstIn state var substitution should evaluate");
    assert_eq!(
        value,
        Value::int(11),
        "SubstIn state-var fallback must preserve the outer binding chain for substitution RHS evaluation"
    );
}

// #3056 Phase 5 boundary rewind tests moved to instance_boundary.rs
