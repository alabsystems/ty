// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! V3 vacuity check — vacuously-true invariant detection (the SOUND subset).
//!
//! General "logically equivalent to TRUE over the reachable set" is undecidable,
//! so we ship ONLY the two statically-decidable, sound special cases
//! (TRUST_VACUITY_GATE §1.A V3):
//!
//! 1. **Constant-`TRUE`** — the invariant body folds to a `TRUE` constant
//!    independent of state (`Inv == TRUE`, or a tautology that reduces to it).
//!    A constant-`TRUE` invariant constrains nothing.
//! 2. **Antecedent-never-holds** — a top-level implication `P => Q` whose
//!    antecedent `P` is the constant `FALSE`. If `P` is statically false in
//!    every state, the implication is vacuously true and proved nothing.
//!
//! Both are detected by pure structural / constant-folding analysis on the
//! resolved invariant body, so there are **no false positives**: if `P` truly is
//! the constant `FALSE` (resp. the body is the constant `TRUE`), the invariant
//! genuinely constrains nothing. We do NOT attempt the general,
//! state-mentioning "semantically trivial" case (that needs equivalence
//! checking we explicitly do not claim).

use tla_core::ast::Expr;
use tla_core::Spanned;

use crate::vacuity::VacuityWarning;

/// Classify a single invariant body. Returns a [`VacuityWarning`] when the
/// invariant is one of the two sound vacuous-true special cases, else `None`.
///
/// `name` is the invariant's user-facing name (for the warning text).
#[must_use]
pub(crate) fn classify_invariant(name: &str, body: &Spanned<Expr>) -> Option<VacuityWarning> {
    // Strip outer wrappers that do not change vacuity: `[]Inv` (always) and
    // top-level universal quantifiers `\A x \in S : Inv`.
    let core = strip_vacuity_neutral_wrappers(&body.node);

    // (a) Top-level implication `P => Q` with `P` statically FALSE. Checked
    // first so an over-constrained `P` is reported as the more precise
    // "antecedent never holds" rather than the generic constant-TRUE case.
    if let Expr::Implies(p, _q) = core {
        if folds_to_false(&p.node) {
            return Some(VacuityWarning::AntecedentNeverHolds {
                invariant: name.to_string(),
            });
        }
    }

    // (b) Constant-TRUE: the whole invariant folds to `TRUE`.
    if folds_to_true(core) {
        return Some(VacuityWarning::ConstantTrueInvariant {
            invariant: name.to_string(),
        });
    }

    None
}

/// Peel wrappers that are neutral for vacuity classification: `[]E`, `\A .. : E`,
/// and labels. Returns the innermost relevant expression by reference.
fn strip_vacuity_neutral_wrappers(mut e: &Expr) -> &Expr {
    loop {
        match e {
            Expr::Always(inner) => e = &inner.node,
            Expr::Forall(_vars, inner) => e = &inner.node,
            Expr::Label(label) => e = &label.body.node,
            _ => return e,
        }
    }
}

/// Sound, conservative "constant-folds to TRUE". Only returns `true` when the
/// expression is provably the constant `TRUE` without reference to state — never
/// a false positive.
fn folds_to_true(e: &Expr) -> bool {
    match e {
        Expr::Bool(b) => *b,
        Expr::Not(inner) => folds_to_false(&inner.node),
        // A /\ B is TRUE iff both fold to TRUE.
        Expr::And(a, b) => folds_to_true(&a.node) && folds_to_true(&b.node),
        // A \/ B is TRUE if either folds to TRUE.
        Expr::Or(a, b) => folds_to_true(&a.node) || folds_to_true(&b.node),
        // P => Q is TRUE if P folds to FALSE or Q folds to TRUE.
        Expr::Implies(p, q) => folds_to_false(&p.node) || folds_to_true(&q.node),
        // A <=> B is TRUE if both sides fold to the same constant.
        Expr::Equiv(a, b) => {
            (folds_to_true(&a.node) && folds_to_true(&b.node))
                || (folds_to_false(&a.node) && folds_to_false(&b.node))
        }
        Expr::Label(label) => folds_to_true(&label.body.node),
        _ => false,
    }
}

/// Sound, conservative "constant-folds to FALSE". Only returns `true` when the
/// expression is provably the constant `FALSE` without reference to state.
fn folds_to_false(e: &Expr) -> bool {
    match e {
        Expr::Bool(b) => !*b,
        Expr::Not(inner) => folds_to_true(&inner.node),
        // A /\ B is FALSE if either folds to FALSE.
        Expr::And(a, b) => folds_to_false(&a.node) || folds_to_false(&b.node),
        // A \/ B is FALSE iff both fold to FALSE.
        Expr::Or(a, b) => folds_to_false(&a.node) && folds_to_false(&b.node),
        Expr::Label(label) => folds_to_false(&label.body.node),
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::parse_module;

    /// Resolve `name`'s body from a parsed single-module source and classify it.
    fn classify_from_src(src: &str, name: &str) -> Option<VacuityWarning> {
        use tla_core::ast::Unit;
        let module = parse_module(src);
        let body = module
            .units
            .iter()
            .find_map(|u| match &u.node {
                Unit::Operator(def) if def.name.node == name => Some(def.body.clone()),
                _ => None,
            })
            .unwrap_or_else(|| panic!("operator {name} not found"));
        classify_invariant(name, &body)
    }

    #[test]
    fn constant_true_is_flagged() {
        let src = r#"
---- MODULE M ----
VARIABLE x
Inv == TRUE
====
"#;
        assert!(matches!(
            classify_from_src(src, "Inv"),
            Some(VacuityWarning::ConstantTrueInvariant { .. })
        ));
    }

    #[test]
    fn antecedent_false_implication_is_flagged() {
        let src = r#"
---- MODULE M ----
VARIABLE x
Inv == FALSE => (x = 0)
====
"#;
        assert!(matches!(
            classify_from_src(src, "Inv"),
            Some(VacuityWarning::AntecedentNeverHolds { .. })
        ));
    }

    #[test]
    fn constraining_invariant_is_not_flagged() {
        let src = r#"
---- MODULE M ----
VARIABLE x
Inv == x \in {0, 1}
====
"#;
        assert!(classify_from_src(src, "Inv").is_none());
    }

    #[test]
    fn real_implication_is_not_flagged() {
        // P mentions state, so it is NOT statically FALSE — must not flag.
        let src = r#"
---- MODULE M ----
VARIABLE x
Inv == (x = 0) => (x >= 0)
====
"#;
        assert!(classify_from_src(src, "Inv").is_none());
    }
}
