// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Certificate builder state for generating linear step sequences.

use std::collections::HashMap;
use tla_cert::{CertificateStep, Formula as CertFormula, Justification, StepId};

/// State for certificate generation.
pub(super) struct CertificateBuilder {
    /// Next step ID to assign.
    next_id: StepId,
    /// Generated steps.
    pub(super) steps: Vec<CertificateStep>,
    /// Map from formulas to their step IDs (for referencing previously proven facts).
    formula_to_step: HashMap<CertFormula, StepId>,
}

impl CertificateBuilder {
    pub(super) fn new() -> Self {
        Self {
            next_id: 0,
            steps: Vec::new(),
            formula_to_step: HashMap::new(),
        }
    }

    /// Add a step and return its ID.
    pub(super) fn add_step(
        &mut self,
        formula: CertFormula,
        justification: Justification,
    ) -> StepId {
        let id = self.next_id;
        self.next_id += 1;

        // Record this formula's step ID for later reference
        self.formula_to_step.insert(formula.clone(), id);

        self.steps.push(CertificateStep {
            id,
            formula,
            justification,
        });

        id
    }

    /// Look up a formula's step ID.
    pub(super) fn lookup(&self, formula: &CertFormula) -> Option<StepId> {
        self.formula_to_step.get(formula).copied()
    }

    /// Fork a child builder for generating a scoped case-split branch.
    ///
    /// The child shares this builder's id space (so branch-step ids never
    /// collide with outer ids) and inherits a snapshot of the currently-visible
    /// facts (so branch steps may reference outer facts). Steps emitted into the
    /// child are collected separately and are NOT recorded in the parent's fact
    /// map; call [`absorb_counter`](Self::absorb_counter) afterward to advance
    /// the parent's id counter past everything the branch consumed.
    pub(super) fn child_scope(&self) -> CertificateBuilder {
        CertificateBuilder {
            next_id: self.next_id,
            steps: Vec::new(),
            formula_to_step: self.formula_to_step.clone(),
        }
    }

    /// Advance this builder's id counter past every id a child scope consumed,
    /// keeping ids globally unique across the whole certificate.
    pub(super) fn absorb_counter(&mut self, child: &CertificateBuilder) {
        self.next_id = child.next_id;
    }

    /// Record a branch assumption in this (branch) scope: it becomes visible to
    /// later branch steps under a fresh id, but is not itself a derivation step.
    /// Returns the assigned id.
    pub(super) fn add_assumption(&mut self, formula: CertFormula) -> StepId {
        let id = self.next_id;
        self.next_id += 1;
        self.formula_to_step.insert(formula, id);
        id
    }
}
