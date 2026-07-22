// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Formula operations: substitution, alpha-equivalence, tableau decomposition, and unification.

use std::collections::HashSet;

use crate::types::{Formula, Term};

// ============================================================================
// Negation and tableau helpers
// ============================================================================

fn neg(formula: &Formula) -> Formula {
    Formula::Not(Box::new(formula.clone()))
}

fn is_negated_quantifier_instance(body: &Formula, var: &str, formula: &Formula) -> bool {
    if let Formula::Not(inner) = formula {
        is_existential_intro_instance(body, var, inner)
    } else {
        false
    }
}

/// Decide whether `conclusion` follows from `premise` by a *single-premise*
/// (linear) tableau rule that is **sound**: the premise, on its own, logically
/// entails the conclusion.
///
/// # Soundness contract
///
/// This function admits ONLY rules where `premise ⊨ conclusion` unconditionally.
/// Concretely, that is the α (non-branching) rules, the γ (universal
/// instantiation) rule, and equality symmetry:
///
/// - `A ∧ B ⊢ A` and `A ∧ B ⊢ B`
/// - `A ↔ B ⊢ (A → B)` and `A ↔ B ⊢ (B → A)`
/// - `∀x. P(x) ⊢ P(t)` for any term `t`
/// - `a = b ⊢ b = a`
/// - `¬(A ∨ B) ⊢ ¬A` and `¬(A ∨ B) ⊢ ¬B`
/// - `¬(A → B) ⊢ A` and `¬(A → B) ⊢ ¬B`
/// - `¬∃x. P(x) ⊢ ¬P(t)` for any term `t` (i.e. `∀x. ¬P(x)`)
/// - `¬FALSE ⊢ TRUE`
///
/// It deliberately does **not** admit the β (branching) rules or the δ
/// (existential-witness) rule, because none of them is a sound single-premise
/// consequence:
///
/// - `A ∨ B ⊬ A` (nor `B`); `A → B ⊬ ¬A` (nor `B`); `¬(A ∧ B) ⊬ ¬A` (nor `¬B`);
///   `¬(A ↔ B)` yields neither branch alone. Deriving one branch of a
///   disjunction is unsound — it requires a case split (see
///   [`Justification::CaseSplit`](crate::Justification::CaseSplit)), which the
///   checker validates with per-branch fact scoping.
/// - `∃x. P(x) ⊬ P(t)` for an arbitrary `t`, and `¬∀x. P(x) ⊬ ¬P(t)` for an
///   arbitrary `t`. Existential elimination is only sound for a *fresh* witness,
///   a side-condition this linear check cannot enforce, so it is rejected.
pub(crate) fn is_valid_tableau_decomposition(premise: &Formula, conclusion: &Formula) -> bool {
    if alpha_equiv(premise, conclusion) {
        return true;
    }

    match premise {
        // α: A ∧ B ⊢ A  and  A ∧ B ⊢ B  (both conjuncts are entailed)
        Formula::And(a, b) => {
            alpha_equiv(conclusion, a.as_ref()) || alpha_equiv(conclusion, b.as_ref())
        }
        // α: A ↔ B ⊢ (A → B)  and  A ↔ B ⊢ (B → A)  (both directions are entailed)
        Formula::Equiv(a, b) => {
            let ab = Formula::Implies(Box::new(a.as_ref().clone()), Box::new(b.as_ref().clone()));
            let ba = Formula::Implies(Box::new(b.as_ref().clone()), Box::new(a.as_ref().clone()));
            alpha_equiv(conclusion, &ab) || alpha_equiv(conclusion, &ba)
        }

        // γ: ∀x. P(x) ⊢ P(t) for any term t (sound universal instantiation).
        // NB: the existential (`∃x. P(x) ⊢ P(t)`) is NOT admitted here — that
        // would be existential elimination to an arbitrary term, which is
        // unsound without a freshness side-condition.
        Formula::Forall(var, body) => is_existential_intro_instance(body, var, conclusion),

        // equality symmetry: a = b ⊢ b = a
        Formula::Eq(a, b) => {
            let sym = Formula::Eq(b.clone(), a.clone());
            alpha_equiv(conclusion, &sym)
        }

        Formula::Not(inner) => match inner.as_ref() {
            // α: ¬(A ∨ B) ⊢ ¬A  and  ¬(A ∨ B) ⊢ ¬B  (both are entailed)
            Formula::Or(a, b) => {
                alpha_equiv(conclusion, &neg(a.as_ref()))
                    || alpha_equiv(conclusion, &neg(b.as_ref()))
            }
            // α: ¬(A → B) ⊢ A  and  ¬(A → B) ⊢ ¬B  (both are entailed)
            Formula::Implies(a, b) => {
                alpha_equiv(conclusion, a.as_ref()) || alpha_equiv(conclusion, &neg(b.as_ref()))
            }
            // γ': ¬∃x. P(x) ⊢ ¬P(t) for any term t (≡ ∀x. ¬P(x)).
            // NB: the ¬∀ case (`¬∀x. P(x) ⊢ ¬P(t)`) is NOT admitted — it is a
            // δ (existential-witness) rule requiring a fresh constant.
            Formula::Exists(var, body) => is_negated_quantifier_instance(body, var, conclusion),
            // ¬FALSE ⊢ TRUE
            Formula::Bool(false) => matches!(conclusion, Formula::Bool(true)),
            //
            // The β rules `¬(A ∧ B)` and `¬(A ↔ B)`, and the δ rule `¬∀x. P(x)`,
            // are intentionally absent: none entails a single conclusion.
            _ => false,
        },

        // A ∨ B, A → B (β rules) and ∃x. P(x) (δ rule) are intentionally absent:
        // none is a sound single-premise consequence. A ∨ B / A → B require a
        // case split; ∃ requires a fresh witness.
        _ => false,
    }
}

// ============================================================================
// Term and formula substitution
// ============================================================================

/// Collect the free variables of `term` into `out`.
///
/// Terms have no binders, so every variable occurring in a term is free.
fn term_free_vars(term: &Term, out: &mut HashSet<String>) {
    match term {
        Term::Var(v) => {
            out.insert(v.clone());
        }
        Term::Const(_) | Term::Int(_) => {}
        Term::App(_, args) => {
            for arg in args {
                term_free_vars(arg, out);
            }
        }
    }
}

fn term_free_vars_set(term: &Term) -> HashSet<String> {
    let mut out = HashSet::new();
    term_free_vars(term, &mut out);
    out
}

/// Does `term` contain `old` as a subterm (including `term == old` itself)?
fn term_contains(term: &Term, old: &Term) -> bool {
    if term == old {
        return true;
    }
    match term {
        Term::Var(_) | Term::Const(_) | Term::Int(_) => false,
        Term::App(_, args) => args.iter().any(|a| term_contains(a, old)),
    }
}

/// Does `formula` contain an occurrence of `old` that
/// [`substitute_term_in_formula`] would actually replace — i.e. one not blocked
/// by an enclosing binder that binds a free variable of `old`?
fn formula_has_replaceable_occurrence(formula: &Formula, old: &Term) -> bool {
    match formula {
        Formula::Bool(_) => false,
        Formula::Predicate(_, terms) => terms.iter().any(|t| term_contains(t, old)),
        Formula::Not(f) => formula_has_replaceable_occurrence(f, old),
        Formula::And(l, r) | Formula::Or(l, r) | Formula::Implies(l, r) | Formula::Equiv(l, r) => {
            formula_has_replaceable_occurrence(l, old) || formula_has_replaceable_occurrence(r, old)
        }
        Formula::Forall(var, body) | Formula::Exists(var, body) => {
            !term_free_vars_set(old).contains(var) && formula_has_replaceable_occurrence(body, old)
        }
        Formula::Eq(l, r) => term_contains(l, old) || term_contains(r, old),
    }
}

/// Substitute all occurrences of `old` term with `new` term in a formula,
/// **failing closed on variable capture**: returns `None` when the substitution
/// cannot be performed without a free variable of `new` being captured by a
/// binder of the same name. Callers must treat `None` as "this proof step does
/// not verify" — never as a successful substitution.
///
/// # Soundness
///
/// A naive (capturing) substitution is unsound in every rule that uses it: for
/// example, instantiating `∀x. ∃y. ¬(x=y)` (true in any ≥2-element domain) with
/// `x := y` must NOT produce the captured `∃y. ¬(y=y)` (false). This function
/// therefore enforces two side-conditions at each binder `Q v. body`:
///
/// - **Shadowing** (generalized): if `v` is a free variable of `old`, every
///   occurrence of `old` inside `body` refers to the *bound* `v`, not the free
///   term the substitution is about, so the body is left untouched. For
///   `old = Var(v)` this is exactly the classic shadowing rule; for compound
///   `old` (e.g. `f(x)` under `∀x`) it refuses to rewrite occurrences whose
///   variables are bound at the occurrence site.
/// - **Capture** (fail closed): otherwise, if `v` is a free variable of `new`
///   and `body` still contains an occurrence of `old` that would actually be
///   replaced, performing the substitution would capture `new`'s free `v` —
///   return `None`. No silent alpha-renaming is attempted; a certificate step
///   that needs a capture-avoiding rename is simply rejected.
pub(crate) fn substitute_term_in_formula(
    formula: &Formula,
    old: &Term,
    new: &Term,
) -> Option<Formula> {
    Some(match formula {
        Formula::Bool(b) => Formula::Bool(*b),
        Formula::Predicate(name, terms) => {
            let new_terms: Vec<Term> = terms.iter().map(|t| substitute_term(t, old, new)).collect();
            Formula::Predicate(name.clone(), new_terms)
        }
        Formula::Not(f) => Formula::Not(Box::new(substitute_term_in_formula(f, old, new)?)),
        Formula::And(l, r) => Formula::And(
            Box::new(substitute_term_in_formula(l, old, new)?),
            Box::new(substitute_term_in_formula(r, old, new)?),
        ),
        Formula::Or(l, r) => Formula::Or(
            Box::new(substitute_term_in_formula(l, old, new)?),
            Box::new(substitute_term_in_formula(r, old, new)?),
        ),
        Formula::Implies(l, r) => Formula::Implies(
            Box::new(substitute_term_in_formula(l, old, new)?),
            Box::new(substitute_term_in_formula(r, old, new)?),
        ),
        Formula::Equiv(l, r) => Formula::Equiv(
            Box::new(substitute_term_in_formula(l, old, new)?),
            Box::new(substitute_term_in_formula(r, old, new)?),
        ),
        Formula::Forall(var, body) => Formula::Forall(
            var.clone(),
            Box::new(substitute_in_binder_body(var, body, old, new)?),
        ),
        Formula::Exists(var, body) => Formula::Exists(
            var.clone(),
            Box::new(substitute_in_binder_body(var, body, old, new)?),
        ),
        Formula::Eq(l, r) => {
            Formula::Eq(substitute_term(l, old, new), substitute_term(r, old, new))
        }
    })
}

/// Substitute inside the body of a binder `Q var. body`, enforcing the two
/// binder side-conditions documented on [`substitute_term_in_formula`].
fn substitute_in_binder_body(var: &str, body: &Formula, old: &Term, new: &Term) -> Option<Formula> {
    // Shadowing (generalized): the binder binds a free variable of `old`, so
    // occurrences of `old` in `body` are not the free term being replaced.
    // Leave the body untouched.
    if term_free_vars_set(old).contains(var) {
        return Some(body.clone());
    }
    // Capture (fail closed): the binder binds a free variable of `new`, and
    // `body` has at least one occurrence that would actually be replaced.
    // Substituting would capture — refuse rather than build a bogus formula.
    if term_free_vars_set(new).contains(var) && formula_has_replaceable_occurrence(body, old) {
        return None;
    }
    substitute_term_in_formula(body, old, new)
}

/// Substitute all occurrences of `old` term with `new` term in a term
fn substitute_term(term: &Term, old: &Term, new: &Term) -> Term {
    if term == old {
        return new.clone();
    }
    match term {
        Term::Var(_) | Term::Const(_) | Term::Int(_) => term.clone(),
        Term::App(name, args) => {
            let new_args: Vec<Term> = args.iter().map(|a| substitute_term(a, old, new)).collect();
            Term::App(name.clone(), new_args)
        }
    }
}

/// Substitute all free occurrences of variable `var` with term `replacement`
/// in a formula. Returns `None` when the substitution would capture a free
/// variable of `replacement` (see [`substitute_term_in_formula`]); callers must
/// treat `None` as "this proof step does not verify".
pub(crate) fn substitute_var_in_formula(
    formula: &Formula,
    var: &str,
    replacement: &Term,
) -> Option<Formula> {
    let var_term = Term::Var(var.to_string());
    substitute_term_in_formula(formula, &var_term, replacement)
}

// ============================================================================
// Alpha-equivalence
// ============================================================================

/// Check whether two formulas are alpha-equivalent: equal up to consistent
/// renaming of bound variables.
///
/// For example, `∀x. P(x)` and `∀y. P(y)` are alpha-equivalent, whereas
/// `∀x. P(x)` and `∀y. P(z)` (which captures a different free variable) are
/// not. Free variables must match by name; only quantifier-bound names may
/// differ. This is the comparison the checker uses to match a certificate's
/// final step against its goal, so that a prover's choice of bound-variable
/// names does not affect validity.
pub fn alpha_equiv(f1: &Formula, f2: &Formula) -> bool {
    alpha_equiv_formula(f1, f2, &mut Vec::new(), &mut Vec::new())
}

/// Check alpha-equivalence of formulas with bound variable tracking.
/// `bindings1` and `bindings2` track corresponding bound variables from f1 and f2.
fn alpha_equiv_formula(
    f1: &Formula,
    f2: &Formula,
    bindings1: &mut Vec<String>,
    bindings2: &mut Vec<String>,
) -> bool {
    match (f1, f2) {
        (Formula::Bool(a), Formula::Bool(b)) => a == b,

        (Formula::Predicate(name1, args1), Formula::Predicate(name2, args2)) => {
            name1 == name2
                && args1.len() == args2.len()
                && args1
                    .iter()
                    .zip(args2)
                    .all(|(t1, t2)| alpha_equiv_term(t1, t2, bindings1, bindings2))
        }

        (Formula::Not(a), Formula::Not(b)) => alpha_equiv_formula(a, b, bindings1, bindings2),

        (Formula::And(a1, a2), Formula::And(b1, b2))
        | (Formula::Or(a1, a2), Formula::Or(b1, b2))
        | (Formula::Implies(a1, a2), Formula::Implies(b1, b2))
        | (Formula::Equiv(a1, a2), Formula::Equiv(b1, b2)) => {
            alpha_equiv_formula(a1, b1, bindings1, bindings2)
                && alpha_equiv_formula(a2, b2, bindings1, bindings2)
        }

        (Formula::Forall(v1, body1), Formula::Forall(v2, body2))
        | (Formula::Exists(v1, body1), Formula::Exists(v2, body2)) => {
            // Push corresponding bound variables
            bindings1.push(v1.clone());
            bindings2.push(v2.clone());
            let result = alpha_equiv_formula(body1, body2, bindings1, bindings2);
            bindings1.pop();
            bindings2.pop();
            result
        }

        (Formula::Eq(t1a, t1b), Formula::Eq(t2a, t2b)) => {
            alpha_equiv_term(t1a, t2a, bindings1, bindings2)
                && alpha_equiv_term(t1b, t2b, bindings1, bindings2)
        }

        _ => false,
    }
}

/// Check alpha-equivalence of terms with bound variable tracking.
fn alpha_equiv_term(t1: &Term, t2: &Term, bindings1: &[String], bindings2: &[String]) -> bool {
    match (t1, t2) {
        (Term::Var(v1), Term::Var(v2)) => {
            // Check if both are bound at corresponding positions
            let pos1 = bindings1.iter().rposition(|b| b == v1);
            let pos2 = bindings2.iter().rposition(|b| b == v2);

            match (pos1, pos2) {
                // Both bound at the same relative position
                (Some(p1), Some(p2)) => p1 == p2,
                // Both free - must have same name
                (None, None) => v1 == v2,
                // One bound, one free - not equivalent
                _ => false,
            }
        }

        (Term::Const(c1), Term::Const(c2)) => c1 == c2,
        (Term::Int(i1), Term::Int(i2)) => i1 == i2,

        (Term::App(name1, args1), Term::App(name2, args2)) => {
            name1 == name2
                && args1.len() == args2.len()
                && args1
                    .iter()
                    .zip(args2)
                    .all(|(a1, a2)| alpha_equiv_term(a1, a2, bindings1, bindings2))
        }

        _ => false,
    }
}

// ============================================================================
// Existential introduction and unification
// ============================================================================

/// Check if `witness` is an instance of `body` where free occurrences of `var`
/// are replaced by a single (consistent) witness term.
/// This function supports alpha-equivalence for inner quantifiers.
pub(crate) fn is_existential_intro_instance(body: &Formula, var: &str, witness: &Formula) -> bool {
    let mut inferred: Option<Term> = None;
    let mut bindings_body: Vec<String> = Vec::new();
    let mut bindings_witness: Vec<String> = Vec::new();

    if !unify_formula_with_witness_term(
        body,
        witness,
        var,
        &mut inferred,
        &mut bindings_body,
        &mut bindings_witness,
    ) {
        return false;
    }

    match inferred {
        Some(t) => {
            // Verify by substitution and alpha-equivalence check. A `None`
            // substitution means `body[var := t]` is not performable without
            // variable capture, so `witness` cannot be certified as a genuine
            // instance — fail closed and reject.
            match substitute_var_in_formula(body, var, &t) {
                Some(substituted) => alpha_equiv(&substituted, witness),
                None => false,
            }
        }
        None => alpha_equiv(body, witness),
    }
}

/// Unify `body` against `witness` to find what term `var` was replaced with.
/// Supports alpha-equivalence for inner quantifiers by tracking bound variables.
fn unify_formula_with_witness_term(
    body: &Formula,
    witness: &Formula,
    var: &str,
    inferred: &mut Option<Term>,
    bindings_body: &mut Vec<String>,
    bindings_witness: &mut Vec<String>,
) -> bool {
    match (body, witness) {
        (Formula::Bool(a), Formula::Bool(b)) => a == b,
        (Formula::Predicate(name_a, args_a), Formula::Predicate(name_b, args_b)) => {
            name_a == name_b
                && args_a.len() == args_b.len()
                && args_a.iter().zip(args_b).all(|(t1, t2)| {
                    unify_term_with_witness_term(
                        t1,
                        t2,
                        var,
                        inferred,
                        bindings_body,
                        bindings_witness,
                    )
                })
        }
        (Formula::Not(a), Formula::Not(b)) => {
            unify_formula_with_witness_term(a, b, var, inferred, bindings_body, bindings_witness)
        }
        (Formula::And(al, ar), Formula::And(bl, br))
        | (Formula::Or(al, ar), Formula::Or(bl, br))
        | (Formula::Implies(al, ar), Formula::Implies(bl, br))
        | (Formula::Equiv(al, ar), Formula::Equiv(bl, br)) => {
            unify_formula_with_witness_term(al, bl, var, inferred, bindings_body, bindings_witness)
                && unify_formula_with_witness_term(
                    ar,
                    br,
                    var,
                    inferred,
                    bindings_body,
                    bindings_witness,
                )
        }
        (Formula::Forall(v1, a), Formula::Forall(v2, b))
        | (Formula::Exists(v1, a), Formula::Exists(v2, b)) => {
            // Track bindings for alpha-equivalence
            bindings_body.push(v1.clone());
            bindings_witness.push(v2.clone());
            let result = unify_formula_with_witness_term(
                a,
                b,
                var,
                inferred,
                bindings_body,
                bindings_witness,
            );
            bindings_body.pop();
            bindings_witness.pop();
            result
        }
        (Formula::Eq(a1, a2), Formula::Eq(b1, b2)) => {
            unify_term_with_witness_term(a1, b1, var, inferred, bindings_body, bindings_witness)
                && unify_term_with_witness_term(
                    a2,
                    b2,
                    var,
                    inferred,
                    bindings_body,
                    bindings_witness,
                )
        }
        _ => false,
    }
}

/// Check if `var` is currently shadowed by a binding in the body.
fn is_var_shadowed(var: &str, bindings_body: &[String]) -> bool {
    bindings_body.iter().any(|b| b == var)
}

fn unify_term_with_witness_term(
    body: &Term,
    witness: &Term,
    var: &str,
    inferred: &mut Option<Term>,
    bindings_body: &[String],
    bindings_witness: &[String],
) -> bool {
    // Check if we're at the existential variable (not shadowed)
    if !is_var_shadowed(var, bindings_body) {
        if let Term::Var(v) = body {
            if v == var {
                return match inferred {
                    None => {
                        *inferred = Some(witness.clone());
                        true
                    }
                    Some(existing) => existing == witness,
                };
            }
        }
    }

    // For other terms, use alpha-equivalence logic
    match (body, witness) {
        (Term::Var(v1), Term::Var(v2)) => {
            // Check if both are bound at corresponding positions
            let pos1 = bindings_body.iter().rposition(|b| b == v1);
            let pos2 = bindings_witness.iter().rposition(|b| b == v2);

            match (pos1, pos2) {
                // Both bound at the same relative position
                (Some(p1), Some(p2)) => p1 == p2,
                // Both free - must have same name
                (None, None) => v1 == v2,
                // One bound, one free - not equivalent
                _ => false,
            }
        }
        (Term::Const(a), Term::Const(b)) => a == b,
        (Term::Int(a), Term::Int(b)) => a == b,
        (Term::App(name_a, args_a), Term::App(name_b, args_b)) => {
            name_a == name_b
                && args_a.len() == args_b.len()
                && args_a.iter().zip(args_b).all(|(t1, t2)| {
                    unify_term_with_witness_term(
                        t1,
                        t2,
                        var,
                        inferred,
                        bindings_body,
                        bindings_witness,
                    )
                })
        }
        _ => false,
    }
}
