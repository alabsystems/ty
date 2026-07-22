// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Certificate verification implementation.

use std::collections::{HashMap, HashSet};

use crate::axiom_check;
use crate::formula_ops::{
    alpha_equiv, is_existential_intro_instance, is_valid_tableau_decomposition,
    substitute_term_in_formula, substitute_var_in_formula,
};
use crate::types::{
    CaseBranch, Certificate, CertificateStep, Formula, Justification, StepId, VerificationError,
    VerificationResult,
};

/// A fact table: step id → the formula that step established. Verification
/// threads one of these explicitly (rather than through checker state) so that
/// a [`Justification::CaseSplit`] branch can be checked against a *scoped* copy
/// — the outer facts plus the branch's assumptions — without leaking the
/// branch's derived facts to siblings or to the outer proof.
type Facts = HashMap<StepId, Formula>;

/// Look up a fact by step id, reporting [`VerificationError::UnknownStep`] if it
/// has not been established in the current scope.
fn get_fact<'a>(facts: &'a Facts, id: &StepId) -> Result<&'a Formula, VerificationError> {
    facts.get(id).ok_or(VerificationError::UnknownStep(*id))
}

/// Detect whether the given fact scope is contradictory: it directly contains
/// `FALSE`, or it contains some formula `F` together with its negation `¬F`.
///
/// # Soundness
///
/// Because every rule the checker admits derives only facts that are entailed by
/// the certificate's hypotheses (the α/γ single-premise rules, the intro/elim
/// rules, and per-branch-scoped case splits — no β/δ rule ever injects a
/// non-entailed disjunct into a scope), a contradiction found here is a *genuine*
/// one: it witnesses that the hypotheses of this scope are jointly unsatisfiable,
/// hence entail every formula. Letting a [`Justification::TableauDecomposition`]
/// close such a scope (ex falso quodlibet) is therefore sound. This relies
/// critically on the fact that the unsound single-premise β/δ "decompositions"
/// have been removed from [`is_valid_tableau_decomposition`]; were they present,
/// a scope could accumulate facts from mutually-exclusive branches and this
/// check would fire spuriously.
fn has_contradiction(facts: &Facts) -> bool {
    if facts
        .values()
        .any(|formula| matches!(formula, Formula::Bool(false)))
    {
        return true;
    }

    let formulas: HashSet<&Formula> = facts.values().collect();
    for formula in &formulas {
        if let Formula::Not(inner) = formula {
            if formulas.contains(inner.as_ref()) {
                return true;
            }
        }
    }

    false
}

/// The β-decomposition of a formula into the assumption group(s) of its
/// branches, or `None` if `premise` is not a branching (β) formula. Each inner
/// `Vec` is the set of formulas the corresponding branch assumes.
///
/// The disjunction of the branches is (classically) equivalent to `premise`, so
/// a conclusion proved under every branch's assumptions follows from `premise`:
///
/// - `A ∨ B`      → `[[A], [B]]`
/// - `A → B`      → `[[¬A], [B]]`
/// - `¬(A ∧ B)`   → `[[¬A], [¬B]]`
/// - `¬(A ↔ B)`   → `[[A, ¬B], [¬A, B]]`
fn beta_decompose(premise: &Formula) -> Option<Vec<Vec<Formula>>> {
    let not = |f: &Formula| Formula::Not(Box::new(f.clone()));
    match premise {
        Formula::Or(a, b) => Some(vec![vec![(**a).clone()], vec![(**b).clone()]]),
        Formula::Implies(a, b) => Some(vec![vec![not(a)], vec![(**b).clone()]]),
        Formula::Not(inner) => match inner.as_ref() {
            Formula::And(a, b) => Some(vec![vec![not(a)], vec![not(b)]]),
            Formula::Equiv(a, b) => Some(vec![
                vec![(**a).clone(), not(b)],
                vec![not(a), (**b).clone()],
            ]),
            _ => None,
        },
        _ => None,
    }
}

/// Independent verifier for proof [`Certificate`]s.
///
/// The checker is the trusted core of the crate: it re-derives a certificate
/// from scratch rather than trusting the backend that produced it. It holds two
/// pieces of state — the facts established so far during a run (step id →
/// formula) and the set of registered definitions — and exposes
/// [`verify`](Self::verify) to check a certificate end to end.
///
/// Register any definitions a certificate may expand *before* calling
/// [`verify`](Self::verify); the fact table is reset at the start of every
/// [`verify`](Self::verify) call, so a single checker can be reused across
/// certificates that share the same definitions.
///
/// # Examples
///
/// ```
/// use tla_cert::{
///     Axiom, ArithmeticAxiom, Backend, Certificate, CertificateChecker,
///     CertificateStep, Formula, Justification, Term,
/// };
///
/// // Goal: 0 + a = a, justified directly by the AddZero arithmetic axiom.
/// let goal = Formula::Eq(
///     Term::App("+".into(), vec![Term::Int(0), Term::Var("a".into())]),
///     Term::Var("a".into()),
/// );
/// let cert = Certificate {
///     id: "example".into(),
///     goal: goal.clone(),
///     hypotheses: vec![],
///     steps: vec![CertificateStep {
///         id: 0,
///         formula: goal,
///         justification: Justification::Axiom(Axiom::Arithmetic(ArithmeticAxiom::AddZero)),
///     }],
///     backend: Backend::Zenon,
/// };
///
/// let mut checker = CertificateChecker::new();
/// assert!(checker.verify(&cert).is_valid());
/// ```
pub struct CertificateChecker {
    /// Definitions available for [`Justification::Definition`] expansion, keyed
    /// by name. Populated via [`add_definition`](Self::add_definition).
    definitions: HashMap<String, Formula>,
}

impl CertificateChecker {
    /// Create a new checker with no definitions registered.
    pub fn new() -> Self {
        Self {
            definitions: HashMap::new(),
        }
    }

    /// Register a named definition that [`Justification::Definition`] steps may
    /// expand to `formula`. Re-registering an existing `name` overwrites it.
    pub fn add_definition(&mut self, name: String, formula: Formula) {
        self.definitions.insert(name, formula);
    }

    /// Verify `cert` from scratch, returning the verdict.
    ///
    /// The fact table is cleared, the hypotheses are seeded as facts at ids
    /// `0..n`, then each step is checked in order against the rule named by its
    /// justification; a verified step is added to the fact table for later
    /// steps to reference. The certificate is [`VerificationResult::Valid`] only
    /// if every step checks and the final step is alpha-equivalent to
    /// [`Certificate::goal`]. The first failure short-circuits to
    /// [`VerificationResult::Invalid`] with the offending
    /// [`VerificationError`](crate::VerificationError).
    pub fn verify(&mut self, cert: &Certificate) -> VerificationResult {
        let mut facts: Facts = HashMap::new();

        // Add hypotheses as facts
        for (i, hyp) in cert.hypotheses.iter().enumerate() {
            facts.insert(i as StepId, hyp.clone());
        }

        // Verify each step
        for step in &cert.steps {
            match self.verify_step(&facts, step, &cert.hypotheses) {
                Ok(()) => {
                    facts.insert(step.id, step.formula.clone());
                }
                Err(e) => return VerificationResult::Invalid(e),
            }
        }

        // Check that the goal was proven (using alpha-equivalence for quantified formulas)
        let last_step = cert.steps.last();
        match last_step {
            Some(step) if alpha_equiv(&step.formula, &cert.goal) => VerificationResult::Valid,
            _ => VerificationResult::Invalid(VerificationError::GoalMismatch),
        }
    }

    /// Verify a single step against the fact table `facts` (the facts visible in
    /// the current scope). Used both for top-level steps and, recursively, for
    /// the scoped steps inside a [`Justification::CaseSplit`] branch.
    fn verify_step(
        &self,
        facts: &Facts,
        step: &CertificateStep,
        hypotheses: &[Formula],
    ) -> Result<(), VerificationError> {
        match &step.justification {
            Justification::Hypothesis(idx) => self.verify_hypothesis(step, hypotheses, *idx),
            Justification::Axiom(axiom) => axiom_check::verify_axiom(axiom, &step.formula),
            Justification::ModusPonens {
                premise,
                implication,
            } => self.verify_modus_ponens(facts, step, premise, implication),
            Justification::AndIntro { left, right } => {
                self.verify_and_intro(facts, step, left, right)
            }
            Justification::AndElimLeft { conjunction } => {
                self.verify_and_elim_left(facts, step, conjunction)
            }
            Justification::AndElimRight { conjunction } => {
                self.verify_and_elim_right(facts, step, conjunction)
            }
            Justification::OrIntroLeft { premise, right } => {
                self.verify_or_intro_left(facts, step, premise, right)
            }
            Justification::OrIntroRight { left, premise } => {
                self.verify_or_intro_right(facts, step, left, premise)
            }
            Justification::DoubleNegElim { premise } => {
                self.verify_double_neg_elim(facts, step, premise)
            }
            Justification::Rewrite { equality, target } => {
                self.verify_rewrite(facts, step, equality, target)
            }
            Justification::Definition { name } => self.verify_definition(step, name),
            Justification::UniversalInstantiation { forall, term } => {
                self.verify_universal_inst(facts, step, forall, term)
            }
            Justification::TableauDecomposition { premise } => {
                self.verify_tableau_decomp(facts, step, premise)
            }
            Justification::CaseSplit { premise, branches } => {
                self.verify_case_split(facts, step, premise, branches, hypotheses)
            }
            Justification::ExistentialIntro { witness, variable } => {
                self.verify_existential_intro(facts, step, witness, variable)
            }
        }
    }

    fn verify_hypothesis(
        &self,
        step: &CertificateStep,
        hypotheses: &[Formula],
        idx: usize,
    ) -> Result<(), VerificationError> {
        if idx < hypotheses.len() && hypotheses[idx] == step.formula {
            Ok(())
        } else {
            Err(VerificationError::InvalidJustification {
                step: step.id,
                reason: "Invalid hypothesis reference".to_string(),
            })
        }
    }

    fn verify_modus_ponens(
        &self,
        facts: &Facts,
        step: &CertificateStep,
        premise: &StepId,
        implication: &StepId,
    ) -> Result<(), VerificationError> {
        let p = get_fact(facts, premise)?;
        let imp = get_fact(facts, implication)?;

        if let Formula::Implies(ante, cons) = imp {
            if ante.as_ref() == p && cons.as_ref() == &step.formula {
                return Ok(());
            }
        }

        Err(VerificationError::InvalidJustification {
            step: step.id,
            reason: "Modus ponens doesn't apply".to_string(),
        })
    }

    fn verify_and_intro(
        &self,
        facts: &Facts,
        step: &CertificateStep,
        left: &StepId,
        right: &StepId,
    ) -> Result<(), VerificationError> {
        let l = get_fact(facts, left)?;
        let r = get_fact(facts, right)?;

        if step.formula == Formula::And(Box::new(l.clone()), Box::new(r.clone())) {
            Ok(())
        } else {
            Err(VerificationError::InvalidJustification {
                step: step.id,
                reason: "And-intro doesn't match".to_string(),
            })
        }
    }

    fn verify_and_elim_left(
        &self,
        facts: &Facts,
        step: &CertificateStep,
        conjunction: &StepId,
    ) -> Result<(), VerificationError> {
        let conj = get_fact(facts, conjunction)?;
        if let Formula::And(left, _) = conj {
            if left.as_ref() == &step.formula {
                return Ok(());
            }
        }
        Err(VerificationError::InvalidJustification {
            step: step.id,
            reason: "And-elim-left doesn't apply".to_string(),
        })
    }

    fn verify_and_elim_right(
        &self,
        facts: &Facts,
        step: &CertificateStep,
        conjunction: &StepId,
    ) -> Result<(), VerificationError> {
        let conj = get_fact(facts, conjunction)?;
        if let Formula::And(_, right) = conj {
            if right.as_ref() == &step.formula {
                return Ok(());
            }
        }
        Err(VerificationError::InvalidJustification {
            step: step.id,
            reason: "And-elim-right doesn't apply".to_string(),
        })
    }

    fn verify_or_intro_left(
        &self,
        facts: &Facts,
        step: &CertificateStep,
        premise: &StepId,
        right: &Formula,
    ) -> Result<(), VerificationError> {
        let p = get_fact(facts, premise)?;
        let expected = Formula::Or(Box::new(p.clone()), Box::new(right.clone()));
        if step.formula == expected {
            Ok(())
        } else {
            Err(VerificationError::InvalidJustification {
                step: step.id,
                reason: "Or-intro-left doesn't match".to_string(),
            })
        }
    }

    fn verify_or_intro_right(
        &self,
        facts: &Facts,
        step: &CertificateStep,
        left: &Formula,
        premise: &StepId,
    ) -> Result<(), VerificationError> {
        let q = get_fact(facts, premise)?;
        let expected = Formula::Or(Box::new(left.clone()), Box::new(q.clone()));
        if step.formula == expected {
            Ok(())
        } else {
            Err(VerificationError::InvalidJustification {
                step: step.id,
                reason: "Or-intro-right doesn't match".to_string(),
            })
        }
    }

    fn verify_double_neg_elim(
        &self,
        facts: &Facts,
        step: &CertificateStep,
        premise: &StepId,
    ) -> Result<(), VerificationError> {
        let nnp = get_fact(facts, premise)?;
        if let Formula::Not(inner) = nnp {
            if let Formula::Not(p) = inner.as_ref() {
                if p.as_ref() == &step.formula {
                    return Ok(());
                }
            }
        }
        Err(VerificationError::InvalidJustification {
            step: step.id,
            reason: "Double negation elimination doesn't apply".to_string(),
        })
    }

    fn verify_rewrite(
        &self,
        facts: &Facts,
        step: &CertificateStep,
        equality: &StepId,
        target: &StepId,
    ) -> Result<(), VerificationError> {
        let eq_formula = get_fact(facts, equality)?;
        let target_formula = get_fact(facts, target)?;

        if let Formula::Eq(a, b) = eq_formula {
            // `None` means the rewrite would capture a variable under a binder
            // (see `substitute_term_in_formula`); fail closed — such a step
            // does not verify.
            let rewritten_ab = substitute_term_in_formula(target_formula, a, b);
            let rewritten_ba = substitute_term_in_formula(target_formula, b, a);
            if rewritten_ab.as_ref() == Some(&step.formula)
                || rewritten_ba.as_ref() == Some(&step.formula)
            {
                return Ok(());
            }
        }

        Err(VerificationError::InvalidJustification {
            step: step.id,
            reason: "Rewrite doesn't apply".to_string(),
        })
    }

    fn verify_definition(
        &self,
        step: &CertificateStep,
        name: &str,
    ) -> Result<(), VerificationError> {
        if let Some(def_formula) = self.definitions.get(name) {
            if &step.formula == def_formula {
                return Ok(());
            }
        }
        Err(VerificationError::InvalidJustification {
            step: step.id,
            reason: format!("Definition '{}' not found or doesn't match", name),
        })
    }

    fn verify_universal_inst(
        &self,
        facts: &Facts,
        step: &CertificateStep,
        forall: &StepId,
        term: &crate::types::Term,
    ) -> Result<(), VerificationError> {
        let forall_formula = get_fact(facts, forall)?;
        if let Formula::Forall(var, body) = forall_formula {
            // `None` means instantiating would capture a free variable of
            // `term` under an inner binder (e.g. x := y into `∃y. ¬(x=y)`);
            // fail closed — the step does not verify.
            let instantiated = substitute_var_in_formula(body, var, term);
            if instantiated.as_ref() == Some(&step.formula) {
                return Ok(());
            }
        }
        Err(VerificationError::InvalidJustification {
            step: step.id,
            reason: "Universal instantiation doesn't apply".to_string(),
        })
    }

    fn verify_tableau_decomp(
        &self,
        facts: &Facts,
        step: &CertificateStep,
        premise: &StepId,
    ) -> Result<(), VerificationError> {
        let premise_formula = get_fact(facts, premise)?;
        // Accept either a sound single-premise (α/γ) consequence, or ex falso
        // when the current scope is already contradictory. The latter is sound
        // precisely because every admitted rule only derives scope-entailed
        // facts (see `has_contradiction`), so a contradiction here means the
        // scope's hypotheses are unsatisfiable and entail anything.
        if is_valid_tableau_decomposition(premise_formula, &step.formula)
            || has_contradiction(facts)
        {
            Ok(())
        } else {
            Err(VerificationError::InvalidJustification {
                step: step.id,
                reason: "Tableau decomposition doesn't apply".to_string(),
            })
        }
    }

    /// Verify a [`Justification::CaseSplit`]: the `premise` must β-decompose into
    /// branch assumptions, and every branch must establish `step.formula` under
    /// the outer facts extended with *only* that branch's assumptions.
    fn verify_case_split(
        &self,
        facts: &Facts,
        step: &CertificateStep,
        premise: &StepId,
        branches: &[CaseBranch],
        hypotheses: &[Formula],
    ) -> Result<(), VerificationError> {
        let premise_formula = get_fact(facts, premise)?;
        let invalid = |reason: &str| VerificationError::InvalidJustification {
            step: step.id,
            reason: reason.to_string(),
        };

        let expected = beta_decompose(premise_formula)
            .ok_or_else(|| invalid("case-split premise is not a branching (beta) formula"))?;
        if branches.len() != expected.len() {
            return Err(invalid(
                "case-split branch count does not match the premise's decomposition",
            ));
        }

        for (branch, expected_assumptions) in branches.iter().zip(&expected) {
            // The branch's declared assumptions must be exactly the disjunct(s)
            // this branch of the β-rule introduces (order-sensitive, up to
            // alpha-equivalence). This is what makes ex-falso-per-branch sound:
            // a branch may only assume the case it is meant to discharge.
            if branch.assumptions.len() != expected_assumptions.len() {
                return Err(invalid(
                    "case-split branch assumptions do not match the premise",
                ));
            }
            for ((_, assumed), expected_formula) in
                branch.assumptions.iter().zip(expected_assumptions)
            {
                if !alpha_equiv(assumed, expected_formula) {
                    return Err(invalid(
                        "case-split branch assumption does not match the premise's decomposition",
                    ));
                }
            }

            // Scope: outer facts + this branch's assumptions only. Branch-derived
            // facts stay local and never leak to siblings or the outer proof.
            let mut scoped: Facts = facts.clone();
            for (id, assumed) in &branch.assumptions {
                scoped.insert(*id, assumed.clone());
            }
            for branch_step in &branch.steps {
                self.verify_step(&scoped, branch_step, hypotheses)?;
                scoped.insert(branch_step.id, branch_step.formula.clone());
            }

            // The branch must establish the case-split's shared conclusion.
            if !scoped.values().any(|f| alpha_equiv(f, &step.formula)) {
                return Err(invalid(
                    "case-split branch did not establish the shared conclusion",
                ));
            }
        }

        Ok(())
    }

    fn verify_existential_intro(
        &self,
        facts: &Facts,
        step: &CertificateStep,
        witness: &StepId,
        variable: &str,
    ) -> Result<(), VerificationError> {
        let witness_formula = get_fact(facts, witness)?;
        if let Formula::Exists(var, body) = &step.formula {
            if var == variable && is_existential_intro_instance(body, var, witness_formula) {
                return Ok(());
            }
        }
        Err(VerificationError::InvalidJustification {
            step: step.id,
            reason: "Existential introduction doesn't apply".to_string(),
        })
    }
}

impl Default for CertificateChecker {
    fn default() -> Self {
        Self::new()
    }
}
