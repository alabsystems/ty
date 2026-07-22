// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Soundness regression tests for the trusted certificate checker.
//!
//! The checker is the trusted core: a valid certificate must be trustworthy
//! without trusting the producer. These tests pin down that the checker REJECTS
//! unsound single-premise "tableau decompositions" — in particular the
//! excluded-middle → arbitrary-atom exploit — while still ACCEPTING genuinely
//! valid derivations, including sound per-branch-scoped case splits.

use tla_cert::{
    Axiom, Backend, CaseBranch, Certificate, CertificateChecker, CertificateStep, Formula,
    Justification, Term, VerificationError, VerificationResult,
};

fn pred(name: &str) -> Formula {
    Formula::Predicate(name.to_string(), vec![])
}

fn not(f: Formula) -> Formula {
    Formula::Not(Box::new(f))
}

fn verify(cert: &Certificate) -> VerificationResult {
    CertificateChecker::new().verify(cert)
}

// ===========================================================================
// THE EXPLOIT: excluded middle must not license an arbitrary contingent atom.
// ===========================================================================

/// The confirmed exploit: `A ∨ ¬A` (a tautology) must NOT let a single-premise
/// `TableauDecomposition` derive the contingent atom `A`. Deriving one disjunct
/// of a disjunction is unsound; the checker must reject it.
#[test]
fn excluded_middle_cannot_prove_arbitrary_atom() {
    let a = pred("A");
    let excluded_middle = Formula::Or(Box::new(a.clone()), Box::new(not(a.clone())));

    let cert = Certificate {
        id: "exploit-excluded-middle".to_string(),
        goal: a.clone(), // contingent atom "A"
        hypotheses: vec![],
        steps: vec![
            CertificateStep {
                id: 0,
                formula: excluded_middle,
                justification: Justification::Axiom(Axiom::ExcludedMiddle(a.clone())),
            },
            CertificateStep {
                id: 1,
                formula: a,
                justification: Justification::TableauDecomposition { premise: 0 },
            },
        ],
        backend: Backend::Zenon,
    };

    // MUST be rejected — the offending step is the bogus decomposition.
    assert!(matches!(
        verify(&cert),
        VerificationResult::Invalid(VerificationError::InvalidJustification { step: 1, .. })
    ));
}

/// Same family: `A → B` does not entail `¬A` (nor `B`) on its own.
#[test]
fn implication_cannot_yield_a_single_branch() {
    let a = pred("A");
    let b = pred("B");
    let imp = Formula::Implies(Box::new(a.clone()), Box::new(b));

    let cert = Certificate {
        id: "exploit-implies-branch".to_string(),
        goal: not(a.clone()),
        hypotheses: vec![imp.clone()],
        steps: vec![
            CertificateStep {
                id: 0,
                formula: imp,
                justification: Justification::Hypothesis(0),
            },
            CertificateStep {
                id: 1,
                formula: not(a),
                justification: Justification::TableauDecomposition { premise: 0 },
            },
        ],
        backend: Backend::Zenon,
    };

    assert!(matches!(
        verify(&cert),
        VerificationResult::Invalid(VerificationError::InvalidJustification { step: 1, .. })
    ));
}

/// `¬(A ∧ B)` does not entail `¬A` (nor `¬B`) on its own (β rule).
#[test]
fn negated_conjunction_cannot_yield_a_single_branch() {
    let a = pred("A");
    let b = pred("B");
    let nand = not(Formula::And(Box::new(a.clone()), Box::new(b)));

    let cert = Certificate {
        id: "exploit-not-and-branch".to_string(),
        goal: not(a.clone()),
        hypotheses: vec![nand.clone()],
        steps: vec![
            CertificateStep {
                id: 0,
                formula: nand,
                justification: Justification::Hypothesis(0),
            },
            CertificateStep {
                id: 1,
                formula: not(a),
                justification: Justification::TableauDecomposition { premise: 0 },
            },
        ],
        backend: Backend::Zenon,
    };

    assert!(matches!(
        verify(&cert),
        VerificationResult::Invalid(VerificationError::InvalidJustification { step: 1, .. })
    ));
}

/// `∃x. P(x)` does not entail `P(t)` for an arbitrary `t` (δ rule; needs a fresh
/// witness the linear check cannot enforce).
#[test]
fn existential_cannot_be_instantiated_at_arbitrary_term() {
    // ∃x. P(x)
    let body = Formula::Predicate("P".to_string(), vec![Term::Var("x".to_string())]);
    let exists = Formula::Exists("x".to_string(), Box::new(body));
    // Claimed instance P(c) for a specific constant c.
    let instance = Formula::Predicate("P".to_string(), vec![Term::Const("c".to_string())]);

    let cert = Certificate {
        id: "exploit-exists-instance".to_string(),
        goal: instance.clone(),
        hypotheses: vec![exists.clone()],
        steps: vec![
            CertificateStep {
                id: 0,
                formula: exists,
                justification: Justification::Hypothesis(0),
            },
            CertificateStep {
                id: 1,
                formula: instance,
                justification: Justification::TableauDecomposition { premise: 0 },
            },
        ],
        backend: Backend::Zenon,
    };

    assert!(matches!(
        verify(&cert),
        VerificationResult::Invalid(VerificationError::InvalidJustification { step: 1, .. })
    ));
}

// ===========================================================================
// Sanity: the SOUND single-premise consequences are still accepted, so the
// checker is not trivially rejecting every tableau decomposition.
// ===========================================================================

/// `A ∧ B ⊢ A` remains a valid single-premise decomposition.
#[test]
fn conjunction_still_decomposes_soundly() {
    let a = pred("A");
    let b = pred("B");
    let and = Formula::And(Box::new(a.clone()), Box::new(b));

    let cert = Certificate {
        id: "sound-and-decomp".to_string(),
        goal: a.clone(),
        hypotheses: vec![and.clone()],
        steps: vec![
            CertificateStep {
                id: 0,
                formula: and,
                justification: Justification::Hypothesis(0),
            },
            CertificateStep {
                id: 1,
                formula: a,
                justification: Justification::TableauDecomposition { premise: 0 },
            },
        ],
        backend: Backend::Zenon,
    };

    assert!(verify(&cert).is_valid());
}

// ===========================================================================
// Sound case splits ARE accepted; unsound ones are not.
// ===========================================================================

/// A genuinely valid proof by cases: from `P → Q` and `¬Q`, conclude `¬P`.
/// Left branch assumes `¬P` (establishes the conclusion directly); right branch
/// assumes `Q` and closes by contradiction with `¬Q`, deriving `¬P` ex falso.
#[test]
fn sound_case_split_is_accepted() {
    let p = pred("P");
    let q = pred("Q");
    let imp = Formula::Implies(Box::new(p.clone()), Box::new(q.clone()));
    let not_q = not(q.clone());
    let not_p = not(p.clone());

    let cert = Certificate {
        id: "sound-case-split".to_string(),
        goal: not_p.clone(),
        hypotheses: vec![imp.clone(), not_q.clone()],
        steps: vec![
            CertificateStep {
                id: 2,
                formula: imp,
                justification: Justification::Hypothesis(0),
            },
            CertificateStep {
                id: 3,
                formula: not_q,
                justification: Justification::Hypothesis(1),
            },
            CertificateStep {
                id: 4,
                formula: not_p.clone(),
                justification: Justification::CaseSplit {
                    premise: 2, // P → Q  splits into  ¬P | Q
                    branches: vec![
                        // Left branch: assume ¬P. Conclusion ¬P holds directly.
                        CaseBranch {
                            assumptions: vec![(5, not_p.clone())],
                            steps: vec![],
                        },
                        // Right branch: assume Q; with ¬Q this is contradictory,
                        // so ¬P follows ex falso.
                        CaseBranch {
                            assumptions: vec![(6, q.clone())],
                            steps: vec![CertificateStep {
                                id: 7,
                                formula: not_p.clone(),
                                justification: Justification::TableauDecomposition { premise: 2 },
                            }],
                        },
                    ],
                },
            },
        ],
        backend: Backend::Zenon,
    };

    assert!(
        verify(&cert).is_valid(),
        "sound case split should verify: {:?}",
        verify(&cert).error()
    );
}

/// A branch may assume ONLY its own disjunct. If a branch's assumption does not
/// match the premise's β-decomposition, the case split is rejected — otherwise a
/// forger could assume an arbitrary formula.
#[test]
fn case_split_rejects_mismatched_branch_assumption() {
    let p = pred("P");
    let q = pred("Q");
    let imp = Formula::Implies(Box::new(p.clone()), Box::new(q.clone()));

    let cert = Certificate {
        id: "bad-case-split-assumption".to_string(),
        goal: Formula::Bool(false),
        hypotheses: vec![imp.clone()],
        steps: vec![
            CertificateStep {
                id: 0,
                formula: imp,
                justification: Justification::Hypothesis(0),
            },
            CertificateStep {
                id: 1,
                formula: Formula::Bool(false),
                justification: Justification::CaseSplit {
                    premise: 0, // P → Q  requires branches  ¬P | Q
                    branches: vec![
                        // WRONG: left branch assumes P instead of ¬P.
                        CaseBranch {
                            assumptions: vec![(2, p.clone())],
                            steps: vec![],
                        },
                        CaseBranch {
                            assumptions: vec![(3, q.clone())],
                            steps: vec![],
                        },
                    ],
                },
            },
        ],
        backend: Backend::Zenon,
    };

    assert!(matches!(
        verify(&cert),
        VerificationResult::Invalid(VerificationError::InvalidJustification { step: 1, .. })
    ));
}

/// Per-branch scoping: a contradiction reached in ONE branch must not license
/// the conclusion in another branch. Here the right branch is contradictory
/// (`Q`, `¬Q`) but the left branch (`¬P` alone) cannot establish `FALSE`, so the
/// whole split is rejected — no cross-branch ex falso.
#[test]
fn case_split_rejects_when_a_branch_does_not_close() {
    let p = pred("P");
    let q = pred("Q");
    let imp = Formula::Implies(Box::new(p.clone()), Box::new(q.clone()));
    let not_q = not(q.clone());

    let cert = Certificate {
        id: "bad-case-split-open-branch".to_string(),
        goal: Formula::Bool(false),
        hypotheses: vec![imp.clone(), not_q.clone()],
        steps: vec![
            CertificateStep {
                id: 0,
                formula: imp,
                justification: Justification::Hypothesis(0),
            },
            CertificateStep {
                id: 1,
                formula: not_q,
                justification: Justification::Hypothesis(1),
            },
            CertificateStep {
                id: 2,
                formula: Formula::Bool(false),
                justification: Justification::CaseSplit {
                    premise: 0, // P → Q  splits into  ¬P | Q
                    branches: vec![
                        // Left branch assumes ¬P — NOT contradictory, cannot
                        // establish FALSE.
                        CaseBranch {
                            assumptions: vec![(3, not(p.clone()))],
                            steps: vec![CertificateStep {
                                id: 4,
                                formula: Formula::Bool(false),
                                justification: Justification::TableauDecomposition { premise: 0 },
                            }],
                        },
                        // Right branch assumes Q — contradictory with ¬Q.
                        CaseBranch {
                            assumptions: vec![(5, q.clone())],
                            steps: vec![CertificateStep {
                                id: 6,
                                formula: Formula::Bool(false),
                                justification: Justification::TableauDecomposition { premise: 0 },
                            }],
                        },
                    ],
                },
            },
        ],
        backend: Backend::Zenon,
    };

    // Rejected: the left branch (¬P alone) cannot establish FALSE. The failure
    // surfaces on the offending in-branch step, not step 2 — the point is that
    // the split as a whole does not verify (no cross-branch ex falso).
    assert!(matches!(
        verify(&cert),
        VerificationResult::Invalid(VerificationError::InvalidJustification { .. })
    ));
}

// ===========================================================================
// THE CAPTURE EXPLOIT: substitution must be capture-avoiding (fail-closed).
//
// A naive (capturing) substitution turns `∃y. ¬(x=y)` under x := y into the
// captured `∃y. ¬(y=y)` — a FALSE formula "derived" from a true hypothesis.
// Every rule that substitutes (UniversalInstantiation, Rewrite, the tableau
// γ rule, ExistentialIntro's instance check) must refuse the capturing
// substitution and reject the step.
// ===========================================================================

fn var(name: &str) -> Term {
    Term::Var(name.to_string())
}

fn eq(l: Term, r: Term) -> Formula {
    Formula::Eq(l, r)
}

fn exists(v: &str, body: Formula) -> Formula {
    Formula::Exists(v.to_string(), Box::new(body))
}

fn forall(v: &str, body: Formula) -> Formula {
    Formula::Forall(v.to_string(), Box::new(body))
}

/// The exploit, UniversalInstantiation form: from `∀x. ∃y. ¬(x=y)` (true in
/// any ≥2-element domain), instantiating x := y must NOT yield the captured
/// `∃y. ¬(y=y)` (false). The step must be rejected.
#[test]
fn capturing_universal_instantiation_is_rejected() {
    // ∀x. ∃y. ¬(x=y)
    let hyp = forall("x", exists("y", not(eq(var("x"), var("y")))));
    // ∃y. ¬(y=y)  — FALSE
    let goal = exists("y", not(eq(var("y"), var("y"))));

    let cert = Certificate {
        id: "exploit-capture-univ-inst".to_string(),
        goal: goal.clone(),
        hypotheses: vec![hyp],
        steps: vec![CertificateStep {
            id: 1,
            formula: goal,
            justification: Justification::UniversalInstantiation {
                forall: 0,      // the hypothesis fact
                term: var("y"), // captured by the inner ∃y
            },
        }],
        backend: Backend::Zenon,
    };

    assert!(matches!(
        verify(&cert),
        VerificationResult::Invalid(VerificationError::InvalidJustification { step: 1, .. })
    ));
}

/// The exploit, tableau γ form: `∀x. ∃y. ¬(x=y) ⊢ ∃y. ¬(y=y)` must not be
/// accepted as a γ decomposition (the inferred instantiation term `y` is
/// captured by the inner binder).
#[test]
fn capturing_tableau_gamma_is_rejected() {
    let hyp = forall("x", exists("y", not(eq(var("x"), var("y")))));
    let goal = exists("y", not(eq(var("y"), var("y"))));

    let cert = Certificate {
        id: "exploit-capture-gamma".to_string(),
        goal: goal.clone(),
        hypotheses: vec![hyp],
        steps: vec![CertificateStep {
            id: 1,
            formula: goal,
            justification: Justification::TableauDecomposition { premise: 0 },
        }],
        backend: Backend::Zenon,
    };

    assert!(matches!(
        verify(&cert),
        VerificationResult::Invalid(VerificationError::InvalidJustification { step: 1, .. })
    ));
}

/// The exploit, Rewrite form: from the satisfiable hypotheses `x = y` and
/// `∃y. ¬(x=y)`, rewriting x := y inside the target must NOT capture and
/// produce `∃y. ¬(y=y)` (false).
#[test]
fn capturing_rewrite_is_rejected() {
    let equality = eq(var("x"), var("y"));
    let target = exists("y", not(eq(var("x"), var("y"))));
    let goal = exists("y", not(eq(var("y"), var("y"))));

    let cert = Certificate {
        id: "exploit-capture-rewrite".to_string(),
        goal: goal.clone(),
        hypotheses: vec![equality, target],
        steps: vec![CertificateStep {
            id: 2,
            formula: goal,
            justification: Justification::Rewrite {
                equality: 0,
                target: 1,
            },
        }],
        backend: Backend::Zenon,
    };

    assert!(matches!(
        verify(&cert),
        VerificationResult::Invalid(VerificationError::InvalidJustification { step: 2, .. })
    ));
}

/// The exploit, Rewrite dual form (the replaced term contains a variable that
/// is BOUND at the occurrence site): from `f(x) = g(x)` (about the free `x`)
/// and `∀x. f(x) = h`, rewriting must NOT touch the bound occurrence `f(x)`
/// inside the quantifier and conclude `∀x. g(x) = h` — a countermodel exists
/// (f ≡ h everywhere, g agrees with f only at the free x's value).
#[test]
fn rewrite_under_binder_that_binds_lhs_variable_is_rejected() {
    let f_x = Term::App("f".to_string(), vec![var("x")]);
    let g_x = Term::App("g".to_string(), vec![var("x")]);
    let h = Term::Const("h".to_string());

    let equality = eq(f_x.clone(), g_x.clone());
    let target = forall("x", eq(f_x, h.clone()));
    let goal = forall("x", eq(g_x, h));

    let cert = Certificate {
        id: "exploit-rewrite-bound-lhs".to_string(),
        goal: goal.clone(),
        hypotheses: vec![equality, target],
        steps: vec![CertificateStep {
            id: 2,
            formula: goal,
            justification: Justification::Rewrite {
                equality: 0,
                target: 1,
            },
        }],
        backend: Backend::Zenon,
    };

    assert!(matches!(
        verify(&cert),
        VerificationResult::Invalid(VerificationError::InvalidJustification { step: 2, .. })
    ));
}

/// The exploit, ExistentialIntro form: from the valid `∀y. y=y`, introducing
/// `∃x. ∀y. x=y` (false in any ≥2-element domain) requires the capturing
/// instantiation x := y; the instance check must fail closed.
#[test]
fn capturing_existential_intro_is_rejected() {
    let witness = forall("y", eq(var("y"), var("y")));
    let goal = exists("x", forall("y", eq(var("x"), var("y"))));

    let cert = Certificate {
        id: "exploit-capture-exists-intro".to_string(),
        goal: goal.clone(),
        hypotheses: vec![witness],
        steps: vec![CertificateStep {
            id: 1,
            formula: goal,
            justification: Justification::ExistentialIntro {
                witness: 0,
                variable: "x".to_string(),
            },
        }],
        backend: Backend::Zenon,
    };

    assert!(matches!(
        verify(&cert),
        VerificationResult::Invalid(VerificationError::InvalidJustification { step: 1, .. })
    ));
}

// ===========================================================================
// Sanity: legitimate, NON-capturing substitutions still verify, so the
// capture check is not trivially rejecting every instantiation.
// ===========================================================================

/// `∀x. ∃y. ¬(x=y)` instantiated with a constant (no free variables, so no
/// capture is possible) still verifies: `∃y. ¬(c=y)`.
#[test]
fn noncapturing_universal_instantiation_still_verifies() {
    let hyp = forall("x", exists("y", not(eq(var("x"), var("y")))));
    let goal = exists("y", not(eq(Term::Const("c".to_string()), var("y"))));

    let cert = Certificate {
        id: "sound-univ-inst-const".to_string(),
        goal: goal.clone(),
        hypotheses: vec![hyp],
        steps: vec![CertificateStep {
            id: 1,
            formula: goal,
            justification: Justification::UniversalInstantiation {
                forall: 0,
                term: Term::Const("c".to_string()),
            },
        }],
        backend: Backend::Zenon,
    };

    assert!(verify(&cert).is_valid());
}

/// Instantiating with a free VARIABLE whose name does not collide with any
/// inner binder still verifies: x := z into `∃y. ¬(x=y)` gives `∃y. ¬(z=y)`.
#[test]
fn noncapturing_variable_instantiation_still_verifies() {
    let hyp = forall("x", exists("y", not(eq(var("x"), var("y")))));
    let goal = exists("y", not(eq(var("z"), var("y"))));

    let cert = Certificate {
        id: "sound-univ-inst-var".to_string(),
        goal: goal.clone(),
        hypotheses: vec![hyp],
        steps: vec![CertificateStep {
            id: 1,
            formula: goal,
            justification: Justification::UniversalInstantiation {
                forall: 0,
                term: var("z"),
            },
        }],
        backend: Backend::Zenon,
    };

    assert!(verify(&cert).is_valid());
}

/// A legitimate rewrite under a quantifier still verifies when nothing is
/// captured: from `c = d` and `∀x. P(x, c)`, conclude `∀x. P(x, d)`.
#[test]
fn noncapturing_rewrite_still_verifies() {
    let c = Term::Const("c".to_string());
    let d = Term::Const("d".to_string());
    let p = |t: Term| Formula::Predicate("P".to_string(), vec![var("x"), t]);
    let goal = forall("x", p(d.clone()));

    let cert = Certificate {
        id: "sound-rewrite-const".to_string(),
        goal: goal.clone(),
        hypotheses: vec![eq(c.clone(), d), forall("x", p(c))],
        steps: vec![CertificateStep {
            id: 2,
            formula: goal,
            justification: Justification::Rewrite {
                equality: 0,
                target: 1,
            },
        }],
        backend: Backend::Zenon,
    };

    assert!(verify(&cert).is_valid());
}

/// The tableau γ rule with a non-capturing inferred term still verifies:
/// `∀x. ∃y. ¬(x=y) ⊢ ∃y. ¬(c=y)`.
#[test]
fn noncapturing_tableau_gamma_still_verifies() {
    let hyp = forall("x", exists("y", not(eq(var("x"), var("y")))));
    let goal = exists("y", not(eq(Term::Const("c".to_string()), var("y"))));

    let cert = Certificate {
        id: "sound-gamma-const".to_string(),
        goal: goal.clone(),
        hypotheses: vec![hyp],
        steps: vec![CertificateStep {
            id: 1,
            formula: goal,
            justification: Justification::TableauDecomposition { premise: 0 },
        }],
        backend: Backend::Zenon,
    };

    assert!(verify(&cert).is_valid());
}
