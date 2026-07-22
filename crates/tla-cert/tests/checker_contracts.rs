// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Integration tests for `tla-cert` public contracts that the in-crate unit
//! tests do not yet exercise: the documented error variants of every
//! verification rule (`UnknownStep` / `InvalidJustification` / `GoalMismatch`),
//! the `Certificate` file-I/O API and its `CertificateIoError` paths, checker
//! state semantics (fact-table reset, definition overwrite), the
//! `VerificationResult::error` accessor, and the axiom schemas (set theory,
//! associativity, multiplication, excluded middle, identity, reflexivity) that
//! have no positive coverage elsewhere.

use tla_cert::{
    alpha_equiv, ArithmeticAxiom, Axiom, Backend, Certificate, CertificateChecker,
    CertificateIoError, CertificateStep, Formula, Justification, SetAxiom, Term, VerificationError,
    VerificationResult,
};

// ---------------------------------------------------------------------------
// Small builders to keep certificates readable.
// ---------------------------------------------------------------------------

fn pred(name: &str) -> Formula {
    Formula::Predicate(name.to_string(), vec![])
}

fn single_step_cert(goal: Formula, formula: Formula, just: Justification) -> Certificate {
    Certificate {
        id: "t".to_string(),
        goal,
        hypotheses: vec![],
        steps: vec![CertificateStep {
            id: 0,
            formula,
            justification: just,
        }],
        backend: Backend::Zenon,
    }
}

fn verify(cert: &Certificate) -> VerificationResult {
    CertificateChecker::new().verify(cert)
}

// ===========================================================================
// GoalMismatch: empty steps, and final step not alpha-equivalent to goal.
// ===========================================================================

#[test]
fn empty_certificate_is_goal_mismatch() {
    // Documented: GoalMismatch covers "the certificate has no steps".
    let cert = Certificate {
        id: "empty".to_string(),
        goal: pred("P"),
        hypotheses: vec![pred("P")],
        steps: vec![],
        backend: Backend::Zenon,
    };
    let result = verify(&cert);
    assert!(matches!(
        result,
        VerificationResult::Invalid(VerificationError::GoalMismatch)
    ));
    // VerificationResult::error must surface the same reason.
    assert!(matches!(
        result.error(),
        Some(VerificationError::GoalMismatch)
    ));
}

#[test]
fn final_step_must_match_goal_even_when_steps_check() {
    // The step is a perfectly valid axiom instance, but it is not the goal.
    let axiom_formula = Formula::Or(
        Box::new(pred("P")),
        Box::new(Formula::Not(Box::new(pred("P")))),
    );
    let cert = single_step_cert(
        pred("Q"), // goal differs from the proven step
        axiom_formula.clone(),
        Justification::Axiom(Axiom::ExcludedMiddle(pred("P"))),
    );
    assert!(matches!(
        verify(&cert),
        VerificationResult::Invalid(VerificationError::GoalMismatch)
    ));
}

// ===========================================================================
// VerificationResult accessors.
// ===========================================================================

#[test]
fn verification_result_accessors_are_consistent() {
    let valid = VerificationResult::Valid;
    assert!(valid.is_valid());
    assert!(valid.error().is_none());

    let invalid = VerificationResult::Invalid(VerificationError::UnknownStep(7));
    assert!(!invalid.is_valid());
    assert!(matches!(
        invalid.error(),
        Some(VerificationError::UnknownStep(7))
    ));
}

// ===========================================================================
// UnknownStep: every rule that references an earlier step must report the
// missing id (not silently fail or panic).
// ===========================================================================

#[test]
fn modus_ponens_unknown_step_reports_premise() {
    let cert = single_step_cert(
        pred("Q"),
        pred("Q"),
        Justification::ModusPonens {
            premise: 55,
            implication: 56,
        },
    );
    // premise is looked up first.
    assert!(matches!(
        verify(&cert),
        VerificationResult::Invalid(VerificationError::UnknownStep(55))
    ));
}

#[test]
fn and_elim_left_unknown_step() {
    let cert = single_step_cert(
        pred("P"),
        pred("P"),
        Justification::AndElimLeft { conjunction: 99 },
    );
    assert!(matches!(
        verify(&cert),
        VerificationResult::Invalid(VerificationError::UnknownStep(99))
    ));
}

#[test]
fn rewrite_unknown_equality_step() {
    let cert = single_step_cert(
        pred("P"),
        pred("P"),
        Justification::Rewrite {
            equality: 12,
            target: 13,
        },
    );
    assert!(matches!(
        verify(&cert),
        VerificationResult::Invalid(VerificationError::UnknownStep(12))
    ));
}

#[test]
fn universal_instantiation_unknown_step() {
    let cert = single_step_cert(
        pred("P"),
        pred("P"),
        Justification::UniversalInstantiation {
            forall: 31,
            term: Term::Const("a".to_string()),
        },
    );
    assert!(matches!(
        verify(&cert),
        VerificationResult::Invalid(VerificationError::UnknownStep(31))
    ));
}

// ===========================================================================
// And-elimination (left & right) — no positive coverage anywhere.
// ===========================================================================

fn and_pq() -> Formula {
    Formula::And(Box::new(pred("P")), Box::new(pred("Q")))
}

#[test]
fn and_elim_left_extracts_left_conjunct() {
    let cert = Certificate {
        id: "and-elim-l".to_string(),
        goal: pred("P"),
        hypotheses: vec![and_pq()],
        steps: vec![
            CertificateStep {
                id: 0,
                formula: and_pq(),
                justification: Justification::Hypothesis(0),
            },
            CertificateStep {
                id: 1,
                formula: pred("P"),
                justification: Justification::AndElimLeft { conjunction: 0 },
            },
        ],
        backend: Backend::Zenon,
    };
    assert!(verify(&cert).is_valid());
}

#[test]
fn and_elim_right_extracts_right_conjunct() {
    let cert = Certificate {
        id: "and-elim-r".to_string(),
        goal: pred("Q"),
        hypotheses: vec![and_pq()],
        steps: vec![
            CertificateStep {
                id: 0,
                formula: and_pq(),
                justification: Justification::Hypothesis(0),
            },
            CertificateStep {
                id: 1,
                formula: pred("Q"),
                justification: Justification::AndElimRight { conjunction: 0 },
            },
        ],
        backend: Backend::Zenon,
    };
    assert!(verify(&cert).is_valid());
}

#[test]
fn and_elim_left_rejects_wrong_conjunct() {
    // Extracting Q via and-elim-LEFT from (P ∧ Q) is unsound.
    let cert = Certificate {
        id: "and-elim-l-bad".to_string(),
        goal: pred("Q"),
        hypotheses: vec![and_pq()],
        steps: vec![
            CertificateStep {
                id: 0,
                formula: and_pq(),
                justification: Justification::Hypothesis(0),
            },
            CertificateStep {
                id: 1,
                formula: pred("Q"),
                justification: Justification::AndElimLeft { conjunction: 0 },
            },
        ],
        backend: Backend::Zenon,
    };
    assert!(matches!(
        verify(&cert),
        VerificationResult::Invalid(VerificationError::InvalidJustification { step: 1, .. })
    ));
}

#[test]
fn and_elim_on_non_conjunction_is_rejected() {
    // The referenced fact is not an And at all.
    let cert = Certificate {
        id: "and-elim-non-conj".to_string(),
        goal: pred("P"),
        hypotheses: vec![pred("P")],
        steps: vec![
            CertificateStep {
                id: 0,
                formula: pred("P"),
                justification: Justification::Hypothesis(0),
            },
            CertificateStep {
                id: 1,
                formula: pred("P"),
                justification: Justification::AndElimRight { conjunction: 0 },
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
// Negative cases for the propositional rules.
// ===========================================================================

#[test]
fn modus_ponens_rejects_mismatched_antecedent() {
    // Have R and P => Q. Modus ponens must not fire (R ≠ P).
    let p_implies_q = Formula::Implies(Box::new(pred("P")), Box::new(pred("Q")));
    let cert = Certificate {
        id: "mp-bad".to_string(),
        goal: pred("Q"),
        hypotheses: vec![pred("R"), p_implies_q.clone()],
        steps: vec![
            CertificateStep {
                id: 0,
                formula: pred("R"),
                justification: Justification::Hypothesis(0),
            },
            CertificateStep {
                id: 1,
                formula: p_implies_q,
                justification: Justification::Hypothesis(1),
            },
            CertificateStep {
                id: 2,
                formula: pred("Q"),
                justification: Justification::ModusPonens {
                    premise: 0,
                    implication: 1,
                },
            },
        ],
        backend: Backend::Zenon,
    };
    assert!(matches!(
        verify(&cert),
        VerificationResult::Invalid(VerificationError::InvalidJustification { step: 2, .. })
    ));
}

#[test]
fn and_intro_rejects_swapped_conjuncts() {
    // From P and Q, AndIntro must produce P ∧ Q (in that order), not Q ∧ P.
    let q_and_p = Formula::And(Box::new(pred("Q")), Box::new(pred("P")));
    let cert = Certificate {
        id: "and-intro-bad".to_string(),
        goal: q_and_p.clone(),
        hypotheses: vec![pred("P"), pred("Q")],
        steps: vec![
            CertificateStep {
                id: 0,
                formula: pred("P"),
                justification: Justification::Hypothesis(0),
            },
            CertificateStep {
                id: 1,
                formula: pred("Q"),
                justification: Justification::Hypothesis(1),
            },
            CertificateStep {
                id: 2,
                formula: q_and_p,
                justification: Justification::AndIntro { left: 0, right: 1 },
            },
        ],
        backend: Backend::Zenon,
    };
    assert!(matches!(
        verify(&cert),
        VerificationResult::Invalid(VerificationError::InvalidJustification { step: 2, .. })
    ));
}

#[test]
fn or_intro_left_rejects_mismatched_disjunct() {
    // From P, OrIntroLeft with right=Q must yield P ∨ Q; claiming P ∨ R fails.
    let p_or_r = Formula::Or(Box::new(pred("P")), Box::new(pred("R")));
    let cert = Certificate {
        id: "or-intro-bad".to_string(),
        goal: p_or_r.clone(),
        hypotheses: vec![pred("P")],
        steps: vec![
            CertificateStep {
                id: 0,
                formula: pred("P"),
                justification: Justification::Hypothesis(0),
            },
            CertificateStep {
                id: 1,
                formula: p_or_r,
                justification: Justification::OrIntroLeft {
                    premise: 0,
                    right: pred("Q"),
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

#[test]
fn double_neg_elim_rejects_single_negation() {
    // ¬P (only one negation) cannot justify P via double-negation elimination.
    let not_p = Formula::Not(Box::new(pred("P")));
    let cert = Certificate {
        id: "dne-bad".to_string(),
        goal: pred("P"),
        hypotheses: vec![not_p.clone()],
        steps: vec![
            CertificateStep {
                id: 0,
                formula: not_p,
                justification: Justification::Hypothesis(0),
            },
            CertificateStep {
                id: 1,
                formula: pred("P"),
                justification: Justification::DoubleNegElim { premise: 0 },
            },
        ],
        backend: Backend::Zenon,
    };
    assert!(matches!(
        verify(&cert),
        VerificationResult::Invalid(VerificationError::InvalidJustification { step: 1, .. })
    ));
}

#[test]
fn universal_instantiation_rejects_non_forall_fact() {
    // Referenced fact is P (not a ∀), so instantiation must fail.
    let cert = Certificate {
        id: "ui-bad".to_string(),
        goal: pred("P"),
        hypotheses: vec![pred("P")],
        steps: vec![
            CertificateStep {
                id: 0,
                formula: pred("P"),
                justification: Justification::Hypothesis(0),
            },
            CertificateStep {
                id: 1,
                formula: pred("P"),
                justification: Justification::UniversalInstantiation {
                    forall: 0,
                    term: Term::Const("a".to_string()),
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

// ===========================================================================
// Rewrite in both directions (a->b and b->a), per the documented contract.
// ===========================================================================

#[test]
fn rewrite_applies_in_reverse_direction() {
    // Given a = b and P(b), derive P(a) by rewriting b back to a.
    let a = Term::Const("a".to_string());
    let b = Term::Const("b".to_string());
    let eq_ab = Formula::Eq(a.clone(), b.clone());
    let pb = Formula::Predicate("P".to_string(), vec![b]);
    let pa = Formula::Predicate("P".to_string(), vec![a]);

    let cert = Certificate {
        id: "rewrite-rev".to_string(),
        goal: pa.clone(),
        hypotheses: vec![eq_ab.clone(), pb.clone()],
        steps: vec![
            CertificateStep {
                id: 0,
                formula: eq_ab,
                justification: Justification::Hypothesis(0),
            },
            CertificateStep {
                id: 1,
                formula: pb,
                justification: Justification::Hypothesis(1),
            },
            CertificateStep {
                id: 2,
                formula: pa,
                justification: Justification::Rewrite {
                    equality: 0,
                    target: 1,
                },
            },
        ],
        backend: Backend::Z3,
    };
    assert!(verify(&cert).is_valid());
}

#[test]
fn rewrite_rejects_when_equality_step_is_not_an_equality() {
    // The "equality" step is actually P, so the rule cannot fire.
    let cert = Certificate {
        id: "rewrite-bad".to_string(),
        goal: pred("Q"),
        hypotheses: vec![pred("P"), pred("Q")],
        steps: vec![
            CertificateStep {
                id: 0,
                formula: pred("P"),
                justification: Justification::Hypothesis(0),
            },
            CertificateStep {
                id: 1,
                formula: pred("Q"),
                justification: Justification::Hypothesis(1),
            },
            CertificateStep {
                id: 2,
                formula: pred("Q"),
                justification: Justification::Rewrite {
                    equality: 0,
                    target: 1,
                },
            },
        ],
        backend: Backend::Z3,
    };
    assert!(matches!(
        verify(&cert),
        VerificationResult::Invalid(VerificationError::InvalidJustification { step: 2, .. })
    ));
}

// ===========================================================================
// Hypothesis rule edge cases.
// ===========================================================================

#[test]
fn hypothesis_out_of_range_index_is_invalid() {
    let cert = single_step_cert(pred("P"), pred("P"), Justification::Hypothesis(5));
    assert!(matches!(
        verify(&cert),
        VerificationResult::Invalid(VerificationError::InvalidJustification { step: 0, .. })
    ));
}

#[test]
fn hypothesis_formula_must_match_the_referenced_hypothesis() {
    // Cite hypothesis 0 (which is P) but claim it establishes Q.
    let cert = Certificate {
        id: "hyp-mismatch".to_string(),
        goal: pred("Q"),
        hypotheses: vec![pred("P")],
        steps: vec![CertificateStep {
            id: 0,
            formula: pred("Q"),
            justification: Justification::Hypothesis(0),
        }],
        backend: Backend::Zenon,
    };
    assert!(matches!(
        verify(&cert),
        VerificationResult::Invalid(VerificationError::InvalidJustification { step: 0, .. })
    ));
}

// ===========================================================================
// Definition rule.
// ===========================================================================

#[test]
fn definition_unknown_name_is_invalid() {
    let cert = single_step_cert(
        pred("P"),
        pred("P"),
        Justification::Definition {
            name: "NoSuchDef".to_string(),
        },
    );
    assert!(matches!(
        verify(&cert),
        VerificationResult::Invalid(VerificationError::InvalidJustification { step: 0, .. })
    ));
}

#[test]
fn add_definition_reregistration_overwrites() {
    // Documented: re-registering a name overwrites it. The certificate's step
    // claims D == P; after overwriting D to Q it must no longer verify.
    let cert = single_step_cert(
        pred("P"),
        pred("P"),
        Justification::Definition {
            name: "D".to_string(),
        },
    );

    let mut checker = CertificateChecker::new();
    checker.add_definition("D".to_string(), pred("P"));
    assert!(checker.verify(&cert).is_valid());

    // Overwrite D with a different formula; the same step must now fail.
    checker.add_definition("D".to_string(), pred("Q"));
    assert!(matches!(
        checker.verify(&cert),
        VerificationResult::Invalid(VerificationError::InvalidJustification { step: 0, .. })
    ));
}

// ===========================================================================
// Checker state semantics: facts are reset between verify() calls, but
// definitions persist (documented).
// ===========================================================================

#[test]
fn facts_do_not_leak_across_verify_calls() {
    // First certificate establishes step id 7 = P.
    let first = Certificate {
        id: "first".to_string(),
        goal: pred("P"),
        hypotheses: vec![pred("P")],
        steps: vec![CertificateStep {
            id: 7,
            formula: pred("P"),
            justification: Justification::Hypothesis(0),
        }],
        backend: Backend::Zenon,
    };

    // Second certificate references id 7 with no hypotheses/steps to define it.
    let second = single_step_cert(
        pred("P"),
        pred("P"),
        Justification::AndElimLeft { conjunction: 7 },
    );

    let mut checker = CertificateChecker::new();
    assert!(checker.verify(&first).is_valid());
    // If facts leaked, id 7 would resolve; it must be UnknownStep instead.
    assert!(matches!(
        checker.verify(&second),
        VerificationResult::Invalid(VerificationError::UnknownStep(7))
    ));
}

#[test]
fn definitions_persist_across_verify_calls() {
    let cert = single_step_cert(
        pred("P"),
        pred("P"),
        Justification::Definition {
            name: "D".to_string(),
        },
    );
    let mut checker = CertificateChecker::new();
    checker.add_definition("D".to_string(), pred("P"));
    assert!(checker.verify(&cert).is_valid());
    // A second run reuses the same definition without re-registering.
    assert!(checker.verify(&cert).is_valid());
}

// ===========================================================================
// Axiom schemas with no positive coverage elsewhere.
// ===========================================================================

fn axiom_cert(formula: Formula, axiom: Axiom) -> Certificate {
    single_step_cert(formula.clone(), formula, Justification::Axiom(axiom))
}

#[test]
fn excluded_middle_axiom() {
    let p = pred("P");
    let f = Formula::Or(
        Box::new(p.clone()),
        Box::new(Formula::Not(Box::new(p.clone()))),
    );
    assert!(verify(&axiom_cert(f, Axiom::ExcludedMiddle(p))).is_valid());
}

#[test]
fn excluded_middle_rejects_wrong_proposition() {
    // P ∨ ¬P claimed, but the axiom is parameterised with Q.
    let f = Formula::Or(
        Box::new(pred("P")),
        Box::new(Formula::Not(Box::new(pred("P")))),
    );
    let cert = axiom_cert(f, Axiom::ExcludedMiddle(pred("Q")));
    assert!(matches!(
        verify(&cert),
        VerificationResult::Invalid(VerificationError::InvalidAxiom(_))
    ));
}

#[test]
fn identity_axiom() {
    let p = pred("P");
    let f = Formula::Implies(Box::new(p.clone()), Box::new(p.clone()));
    assert!(verify(&axiom_cert(f, Axiom::Identity(p))).is_valid());
}

#[test]
fn equality_refl_axiom_positive() {
    let a = Term::Const("a".to_string());
    let f = Formula::Eq(a.clone(), a);
    assert!(verify(&axiom_cert(f, Axiom::EqualityRefl)).is_valid());
}

#[test]
fn add_assoc_axiom() {
    // (a + b) + c = a + (b + c)
    let a = Term::Var("a".to_string());
    let b = Term::Var("b".to_string());
    let c = Term::Var("c".to_string());
    let a_plus_b = Term::App("+".to_string(), vec![a.clone(), b.clone()]);
    let b_plus_c = Term::App("+".to_string(), vec![b, c.clone()]);
    let lhs = Term::App("+".to_string(), vec![a_plus_b, c]);
    let rhs = Term::App("+".to_string(), vec![a, b_plus_c]);
    let f = Formula::Eq(lhs, rhs);
    assert!(verify(&axiom_cert(f, Axiom::Arithmetic(ArithmeticAxiom::AddAssoc))).is_valid());
}

#[test]
fn mul_one_axiom() {
    // a * 1 = a
    let a = Term::Var("a".to_string());
    let lhs = Term::App("*".to_string(), vec![a.clone(), Term::Int(1)]);
    let f = Formula::Eq(lhs, a);
    assert!(verify(&axiom_cert(f, Axiom::Arithmetic(ArithmeticAxiom::MulOne))).is_valid());
}

#[test]
fn mul_one_rejects_multiply_by_two() {
    let a = Term::Var("a".to_string());
    let lhs = Term::App("*".to_string(), vec![a.clone(), Term::Int(2)]);
    let f = Formula::Eq(lhs, a);
    let cert = axiom_cert(f, Axiom::Arithmetic(ArithmeticAxiom::MulOne));
    assert!(matches!(
        verify(&cert),
        VerificationResult::Invalid(VerificationError::InvalidAxiom(_))
    ));
}

#[test]
fn mul_zero_axiom() {
    // a * 0 = 0
    let a = Term::Var("a".to_string());
    let lhs = Term::App("*".to_string(), vec![a, Term::Int(0)]);
    let f = Formula::Eq(lhs, Term::Int(0));
    assert!(verify(&axiom_cert(f, Axiom::Arithmetic(ArithmeticAxiom::MulZero))).is_valid());
}

#[test]
fn mul_zero_rejects_nonzero_rhs() {
    // a * 0 = a is not the MulZero schema (rhs must be 0).
    let a = Term::Var("a".to_string());
    let lhs = Term::App("*".to_string(), vec![a.clone(), Term::Int(0)]);
    let f = Formula::Eq(lhs, a);
    let cert = axiom_cert(f, Axiom::Arithmetic(ArithmeticAxiom::MulZero));
    assert!(matches!(
        verify(&cert),
        VerificationResult::Invalid(VerificationError::InvalidAxiom(_))
    ));
}

// --- Set-theory axioms (no coverage anywhere) -----------------------------

fn member(x: Term, set: Term) -> Formula {
    Formula::Predicate("∈".to_string(), vec![x, set])
}

#[test]
fn empty_set_axiom() {
    // x ∈ {} ↔ FALSE
    let x = Term::Var("x".to_string());
    let f = Formula::Equiv(
        Box::new(member(x, Term::Const("{}".to_string()))),
        Box::new(Formula::Bool(false)),
    );
    assert!(verify(&axiom_cert(f, Axiom::SetTheory(SetAxiom::EmptySet))).is_valid());
}

#[test]
fn empty_set_rejects_true_rhs() {
    // x ∈ {} ↔ TRUE is unsound.
    let x = Term::Var("x".to_string());
    let f = Formula::Equiv(
        Box::new(member(x, Term::Const("{}".to_string()))),
        Box::new(Formula::Bool(true)),
    );
    let cert = axiom_cert(f, Axiom::SetTheory(SetAxiom::EmptySet));
    assert!(matches!(
        verify(&cert),
        VerificationResult::Invalid(VerificationError::InvalidAxiom(_))
    ));
}

#[test]
fn singleton_axiom() {
    // x ∈ {a} ↔ x = a
    let x = Term::Var("x".to_string());
    let a = Term::Const("a".to_string());
    let singleton = Term::App("singleton".to_string(), vec![a.clone()]);
    let f = Formula::Equiv(
        Box::new(member(x.clone(), singleton)),
        Box::new(Formula::Eq(x, a)),
    );
    assert!(verify(&axiom_cert(f, Axiom::SetTheory(SetAxiom::Singleton))).is_valid());
}

#[test]
fn union_axiom() {
    // x ∈ S ∪ T ↔ x ∈ S ∨ x ∈ T
    let x = Term::Var("x".to_string());
    let s = Term::Const("S".to_string());
    let t = Term::Const("T".to_string());
    let union = Term::App("∪".to_string(), vec![s.clone(), t.clone()]);
    let f = Formula::Equiv(
        Box::new(member(x.clone(), union)),
        Box::new(Formula::Or(
            Box::new(member(x.clone(), s)),
            Box::new(member(x, t)),
        )),
    );
    assert!(verify(&axiom_cert(f, Axiom::SetTheory(SetAxiom::Union))).is_valid());
}

#[test]
fn union_rejects_conjunction_rhs() {
    // Union must pair with ∨; using ∧ is the Intersection shape, not Union.
    let x = Term::Var("x".to_string());
    let s = Term::Const("S".to_string());
    let t = Term::Const("T".to_string());
    let union = Term::App("∪".to_string(), vec![s.clone(), t.clone()]);
    let f = Formula::Equiv(
        Box::new(member(x.clone(), union)),
        Box::new(Formula::And(
            Box::new(member(x.clone(), s)),
            Box::new(member(x, t)),
        )),
    );
    let cert = axiom_cert(f, Axiom::SetTheory(SetAxiom::Union));
    assert!(matches!(
        verify(&cert),
        VerificationResult::Invalid(VerificationError::InvalidAxiom(_))
    ));
}

#[test]
fn intersection_axiom() {
    // x ∈ S ∩ T ↔ x ∈ S ∧ x ∈ T
    let x = Term::Var("x".to_string());
    let s = Term::Const("S".to_string());
    let t = Term::Const("T".to_string());
    let inter = Term::App("∩".to_string(), vec![s.clone(), t.clone()]);
    let f = Formula::Equiv(
        Box::new(member(x.clone(), inter)),
        Box::new(Formula::And(
            Box::new(member(x.clone(), s)),
            Box::new(member(x, t)),
        )),
    );
    assert!(verify(&axiom_cert(f, Axiom::SetTheory(SetAxiom::Intersection))).is_valid());
}

// ===========================================================================
// File I/O: save_to_file / load_from_file roundtrip + documented error paths.
// ===========================================================================

fn sample_cert() -> Certificate {
    let a = Term::Var("a".to_string());
    let goal = Formula::Eq(Term::App("+".to_string(), vec![Term::Int(0), a.clone()]), a);
    single_step_cert(
        goal.clone(),
        goal,
        Justification::Axiom(Axiom::Arithmetic(ArithmeticAxiom::AddZero)),
    )
}

fn temp_path(name: &str) -> std::path::PathBuf {
    let mut p = std::env::temp_dir();
    p.push(format!(
        "tla_cert_test_{}_{}_{}.json",
        name,
        std::process::id(),
        // monotonic-ish nonce so parallel tests never collide
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
    ));
    p
}

#[test]
fn save_and_load_roundtrip_preserves_validity() {
    let cert = sample_cert();
    let path = temp_path("roundtrip");

    cert.save_to_file(&path).expect("save should succeed");
    let loaded = Certificate::load_from_file(&path).expect("load should succeed");
    let _ = std::fs::remove_file(&path);

    // Loaded certificate is structurally equal where it matters and still verifies.
    assert_eq!(cert.id, loaded.id);
    assert!(alpha_equiv(&cert.goal, &loaded.goal));
    assert_eq!(cert.backend, loaded.backend);
    assert!(CertificateChecker::new().verify(&loaded).is_valid());
}

#[test]
fn load_from_missing_file_is_io_error() {
    let path = temp_path("definitely_missing");
    // Ensure it does not exist.
    let _ = std::fs::remove_file(&path);
    let err = Certificate::load_from_file(&path).expect_err("loading a missing file must error");
    assert!(
        matches!(err, CertificateIoError::Io(_)),
        "expected Io error, got {:?}",
        err
    );
    // Display should mention I/O.
    assert!(err.to_string().contains("I/O"));
}

#[test]
fn load_from_malformed_json_is_json_error() {
    let path = temp_path("malformed");
    std::fs::write(&path, b"{ this is not valid certificate json ").expect("write temp");
    let err = Certificate::load_from_file(&path).expect_err("malformed JSON must error");
    let _ = std::fs::remove_file(&path);
    assert!(
        matches!(err, CertificateIoError::Json(_)),
        "expected Json error, got {:?}",
        err
    );
    assert!(err.to_string().contains("JSON"));
}

#[test]
fn from_json_rejects_invalid_input() {
    // Public from_json contract: invalid JSON / schema mismatch -> Err.
    assert!(Certificate::from_json("not json at all").is_err());
    // Valid JSON but wrong schema (missing required fields).
    assert!(Certificate::from_json(r#"{"id":"x"}"#).is_err());
}

#[test]
fn save_to_file_with_bad_parent_dir_is_io_error() {
    // Parent directory does not exist -> documented CertificateIoError::Io.
    let mut path = std::env::temp_dir();
    path.push("tla_cert_nonexistent_dir_xyz");
    path.push("nested");
    path.push("cert.json");
    let err = sample_cert()
        .save_to_file(&path)
        .expect_err("writing under a missing directory must error");
    assert!(
        matches!(err, CertificateIoError::Io(_)),
        "expected Io error, got {:?}",
        err
    );
}
