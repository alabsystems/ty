// Author: Andrew Yates <andrewyates.name@gmail.com>
// Copyright 2026 Andrew Yates Apache-2.0.
//
// Rewrites `v' \in S` (a primed state-variable nondeterministic assignment over
// a static set S) into the semantically-identical existential generator
// `\E <fresh> \in S : v' = <fresh>`.
//
// Why: the existential generator form is already natively compilable (multi /
// nested-exists native lowering), whereas a residual primed `\in` forces
// SetPrimeMode in the action bytecode and bails the whole spec to the
// interpreter. Rewriting the easy case unlocks the fully-native fast path for
// otherwise-native actions (e.g. AsynchInterface's
// `Send == ... /\ val' \in Data /\ ...`) with no new opcodes, transform logic,
// or native-fused admission exceptions.
//
// Soundness: when S is determined by the pre-state (contains no primed
// variable), `v' \in S` is equivalent to `\E x \in S : v' = x` — each distinct
// element of S yields exactly the successor `v' = x`, duplicates collapse
// (sets), and an empty S yields zero successors. This matches the interpreter's
// `SymbolicAssignment::InSet` reference semantics exactly. The rewrite only
// fires in positive (generator) polarity, so a primed membership used as a
// negated guard is left untouched (it bails to the interpreter as before).

use crate::ast::{BoundVar, Expr, OperatorDef};
use crate::expr_contains_any_prime_v;
use crate::name_intern::intern_name;
use crate::span::Spanned;

/// Polarity of a sub-formula relative to the action's top-level conjunction.
/// The rewrite only applies in `Positive` position, where `v' \in S` acts as a
/// successor generator rather than a boolean guard.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Polarity {
    Positive,
    NonPositive,
}

impl Polarity {
    fn flip(self) -> Self {
        match self {
            Polarity::Positive => Polarity::NonPositive,
            Polarity::NonPositive => Polarity::Positive,
        }
    }
}

/// Rewrite qualifying static `v' \in S` conjuncts in an operator body in place.
///
/// Must be called AFTER `resolve_state_vars_in_op_def` so the primed left-hand
/// side is already an `Expr::StateVar`.
pub fn rewrite_static_prime_in_set_in_op_def(def: &mut OperatorDef) {
    let mut next_id: u32 = 0;
    rewrite_expr(&mut def.body, Polarity::Positive, &mut next_id);
}

fn rewrite_expr(node: &mut Spanned<Expr>, pol: Polarity, next_id: &mut u32) {
    // Try to rewrite this node when it is a generator-position primed membership
    // over a static set. Check before descending: the rewritten existential's
    // children need no further rewriting.
    if pol == Polarity::Positive && qualifies(&node.node) {
        rewrite_in_node(node, next_id);
        return;
    }

    // Otherwise descend through logical structure only, carrying polarity.
    // Non-logical (value / atomic) positions cannot host a successor generator,
    // so we deliberately stop there — leaving any primed membership buried in a
    // value expression untouched (it falls back to the interpreter as before).
    match &mut node.node {
        Expr::And(a, b) | Expr::Or(a, b) => {
            rewrite_expr(a, pol, next_id);
            rewrite_expr(b, pol, next_id);
        }
        Expr::Not(a) => rewrite_expr(a, pol.flip(), next_id),
        Expr::Implies(a, b) => {
            rewrite_expr(a, pol.flip(), next_id);
            rewrite_expr(b, pol, next_id);
        }
        Expr::Exists(bounds, body) | Expr::Forall(bounds, body) => {
            for bnd in bounds.iter_mut() {
                if let Some(dom) = &mut bnd.domain {
                    rewrite_expr(dom, Polarity::NonPositive, next_id);
                }
            }
            rewrite_expr(body, pol, next_id);
        }
        Expr::If(cond, then_b, else_b) => {
            rewrite_expr(cond, Polarity::NonPositive, next_id);
            rewrite_expr(then_b, pol, next_id);
            rewrite_expr(else_b, pol, next_id);
        }
        Expr::Label(label) => rewrite_expr(&mut label.body, pol, next_id),
        _ => {}
    }
}

/// True iff `expr` is `v' \in S` where `v` is a state variable and `S` contains
/// no primed variable (static, pre-state-determined).
fn qualifies(expr: &Expr) -> bool {
    let Expr::In(lhs, rhs) = expr else {
        return false;
    };
    let Expr::Prime(inner) = &lhs.node else {
        return false;
    };
    // Post-resolution, a primed state variable is an `Expr::StateVar`. Anything
    // else (primed bound var, primed function application, etc.) is out of scope.
    if !matches!(inner.node, Expr::StateVar(_, _, _)) {
        return false;
    }
    // S must be static: no primed variable inside the right-hand set. The HARD
    // case (S references the post-state) is deliberately not rewritten.
    !expr_contains_any_prime_v(&rhs.node)
}

fn rewrite_in_node(node: &mut Spanned<Expr>, next_id: &mut u32) {
    let span = node.span;
    let id = *next_id;
    *next_id += 1;
    // '$' cannot appear in a TLA+ identifier, so this synthetic binder name
    // cannot collide with any state variable, constant, operator parameter, or
    // user-introduced bound variable.
    let fresh = format!("$inset_{id}");
    let fresh_nid = intern_name(&fresh);

    let placeholder = Expr::Bool(true);
    let in_expr = std::mem::replace(&mut node.node, placeholder);
    let Expr::In(lhs, rhs) = in_expr else {
        // `qualifies` already matched `Expr::In`; this branch is unreachable.
        return;
    };

    let fresh_ref = Box::new(Spanned::new(Expr::Ident(fresh.clone(), fresh_nid), span));
    let eq = Spanned::new(Expr::Eq(lhs, fresh_ref), span);
    let bound = BoundVar {
        name: Spanned::new(fresh, span),
        domain: Some(rhs),
        pattern: None,
    };
    node.node = Expr::Exists(vec![bound], Box::new(eq));
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::name_intern::NameId;

    fn statevar(name: &str) -> Spanned<Expr> {
        Spanned::dummy(Expr::StateVar(name.to_string(), 0, NameId::INVALID))
    }
    fn ident(name: &str) -> Spanned<Expr> {
        Spanned::dummy(Expr::Ident(name.to_string(), NameId::INVALID))
    }
    fn set_enum(elems: Vec<Spanned<Expr>>) -> Spanned<Expr> {
        Spanned::dummy(Expr::SetEnum(elems))
    }
    fn prime(inner: Spanned<Expr>) -> Spanned<Expr> {
        Spanned::dummy(Expr::Prime(Box::new(inner)))
    }
    fn in_expr(lhs: Spanned<Expr>, rhs: Spanned<Expr>) -> Spanned<Expr> {
        Spanned::dummy(Expr::In(Box::new(lhs), Box::new(rhs)))
    }
    fn rewrite(e: &mut Spanned<Expr>) {
        let mut n = 0u32;
        rewrite_expr(e, Polarity::Positive, &mut n);
    }

    #[test]
    fn rewrites_static_primed_membership() {
        // val' \in {1, 2, 3}
        let mut e = in_expr(
            prime(statevar("val")),
            set_enum(vec![
                Spanned::dummy(Expr::Int(1.into())),
                Spanned::dummy(Expr::Int(2.into())),
                Spanned::dummy(Expr::Int(3.into())),
            ]),
        );
        rewrite(&mut e);
        match &e.node {
            Expr::Exists(bounds, body) => {
                assert_eq!(bounds.len(), 1);
                assert!(bounds[0].name.node.starts_with("$inset_"));
                assert!(bounds[0].domain.is_some());
                match &body.node {
                    Expr::Eq(l, r) => {
                        assert!(matches!(&l.node, Expr::Prime(_)));
                        match &r.node {
                            Expr::Ident(n, _) => assert_eq!(n, &bounds[0].name.node),
                            other => panic!("rhs not fresh ident: {other:?}"),
                        }
                    }
                    other => panic!("body not Eq: {other:?}"),
                }
            }
            other => panic!("not rewritten to Exists: {other:?}"),
        }
    }

    #[test]
    fn leaves_dynamic_set_untouched() {
        // val' \in {w'} — S references a primed variable (HARD case)
        let mut e = in_expr(prime(statevar("val")), set_enum(vec![prime(statevar("w"))]));
        rewrite(&mut e);
        assert!(
            matches!(e.node, Expr::In(_, _)),
            "dynamic S must not rewrite"
        );
    }

    #[test]
    fn leaves_unprimed_membership_untouched() {
        // x \in S (no prime) — not a generator we target
        let mut e = in_expr(
            ident("x"),
            set_enum(vec![Spanned::dummy(Expr::Int(1.into()))]),
        );
        rewrite(&mut e);
        assert!(
            matches!(e.node, Expr::In(_, _)),
            "unprimed membership must not rewrite"
        );
    }

    #[test]
    fn leaves_negated_membership_untouched() {
        // ~(val' \in S) — guard polarity, not a generator
        let inner = in_expr(
            prime(statevar("val")),
            set_enum(vec![Spanned::dummy(Expr::Int(1.into()))]),
        );
        let mut e = Spanned::dummy(Expr::Not(Box::new(inner)));
        rewrite(&mut e);
        match &e.node {
            Expr::Not(inner) => assert!(
                matches!(inner.node, Expr::In(_, _)),
                "negated membership must not rewrite"
            ),
            other => panic!("structure changed: {other:?}"),
        }
    }

    #[test]
    fn rewrites_inside_conjunction() {
        // rdy = ack /\ val' \in Data  (AsynchInterface Send shape)
        let conj = Spanned::dummy(Expr::And(
            Box::new(Spanned::dummy(Expr::Eq(
                Box::new(statevar("rdy")),
                Box::new(statevar("ack")),
            ))),
            Box::new(in_expr(prime(statevar("val")), ident("Data"))),
        ));
        let mut e = conj;
        rewrite(&mut e);
        match &e.node {
            Expr::And(_, b) => assert!(
                matches!(b.node, Expr::Exists(_, _)),
                "membership conjunct should be rewritten"
            ),
            other => panic!("not a conjunction: {other:?}"),
        }
    }
}
