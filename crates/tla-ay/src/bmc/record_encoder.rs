// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Record and tuple encoding for BMC translator.
//!
//! Records are encoded as per-field SMT variables whose collision-free names
//! come from [`BmcTranslator::record_field_symbol`]. Field access and `EXCEPT`
//! operate directly on those carriers.
//!
//! Tuples are encoded as per-element SMT variables whose collision-free names
//! come from [`BmcTranslator::tuple_element_symbol`].
//!
//! Part of #3787: ay symbolic engine record and tuple encoding for BMC.

use ay_dpll::api::Term;
use tla_core::ast::{ExceptPathElement, ExceptSpec, Expr};
use tla_core::{dispatch_translate_bool, dispatch_translate_int, Spanned};

use crate::error::{AYError, AYResult};
use crate::TlaSort;

use super::{BmcCarrierKind, BmcTranslator};

/// Information about a record variable across all BMC steps.
///
/// Each record is encoded as a set of per-field SMT variables per step.
/// The field sorts are stored in declaration order. Operations that compare
/// record shapes canonicalize by field name first.
///
/// Part of #3787: Record encoding in BMC translator.
#[derive(Debug)]
pub(super) struct BmcRecordVarInfo {
    /// Field sorts in declaration order.
    pub(super) field_sorts: Vec<(String, TlaSort)>,
    /// Per-field terms per step: field_terms[field_idx][step] = Term.
    pub(super) field_terms: Vec<Vec<Term>>,
}

/// Information about a tuple variable across all BMC steps.
///
/// Each tuple is encoded as a set of per-element SMT variables per step.
/// Element indices are 1-based (matching TLA+ tuple indexing).
///
/// Part of #3787: Tuple encoding in BMC translator.
#[derive(Debug)]
pub(super) struct BmcTupleVarInfo {
    /// Element sorts (0-indexed internally, but 1-indexed in TLA+).
    pub(super) element_sorts: Vec<TlaSort>,
    /// Per-element terms per step: element_terms[elem_idx][step] = Term.
    pub(super) element_terms: Vec<Vec<Term>>,
}

/// Canonicalize a record shape by field name, rejecting duplicate fields.
fn canonical_record_shape(field_sorts: &[(String, TlaSort)]) -> Option<Vec<(String, TlaSort)>> {
    let mut shape: Vec<(String, TlaSort)> = field_sorts
        .iter()
        .map(|(name, sort)| (name.clone(), sort.clone().canonicalized()))
        .collect();
    shape.sort_by(|left, right| left.0.cmp(&right.0));
    (!shape.windows(2).any(|pair| pair[0].0 == pair[1].0)).then_some(shape)
}

/// Decide whether two record shapes are equal or definitely denote disjoint
/// TLA+ values. Set element-sort metadata is not enough to prove inequality:
/// for example, empty `Set(Int)` and empty `Set(String)` values are equal.
fn record_shapes_equal_or_false(
    left: &[(String, TlaSort)],
    right: &[(String, TlaSort)],
    context: &str,
) -> AYResult<bool> {
    let Some(left) = canonical_record_shape(left) else {
        return Ok(false);
    };
    let Some(right) = canonical_record_shape(right) else {
        return Ok(false);
    };
    if left.len() != right.len()
        || left
            .iter()
            .zip(&right)
            .any(|((left_name, _), (right_name, _))| left_name != right_name)
    {
        return Ok(false);
    }
    for ((field_name, left_sort), (_, right_sort)) in left.iter().zip(&right) {
        if left_sort == right_sort {
            continue;
        }
        if matches!(
            (left_sort, right_sort),
            (TlaSort::Set { .. }, TlaSort::Set { .. })
        ) {
            return Err(AYError::UnsupportedOp(format!(
                "BMC cannot decide {context} for field '{field_name}' with differing set \
                 element sorts {left_sort} and {right_sort}"
            )));
        }
        return Ok(false);
    }
    Ok(true)
}

impl BmcTranslator {
    // === Record variable declaration ===

    /// Declare a record state variable for all k+1 steps.
    ///
    /// Each record field becomes an independently and canonically named SMT
    /// variable per step.
    ///
    /// The field sorts must be scalar (Bool, Int, or String) or Set. Nested
    /// records are rejected until every record operation has a recursive
    /// encoding; accepting them with placeholder terms would leave equality,
    /// `UNCHANGED`, and `EXCEPT` unconstrained at the nested field.
    /// Re-declaration is idempotent only for a canonical-equivalent field
    /// shape; field declaration order is ignored.
    ///
    /// Part of #3787: Record encoding in BMC translator.
    pub fn declare_record_var(
        &mut self,
        name: &str,
        field_sorts: Vec<(String, TlaSort)>,
    ) -> AYResult<()> {
        // Validate the ENTIRE shape, including AY carrier conversion, before
        // declaring a single solver symbol. Otherwise a late bad field could
        // leave a partially registered SMT record behind after this method
        // returns `Err`.
        let mut field_names = std::collections::HashSet::with_capacity(field_sorts.len());
        let mut ay_sorts = Vec::with_capacity(field_sorts.len());
        for (field_name, sort) in &field_sorts {
            if !field_names.insert(field_name.as_str()) {
                return Err(AYError::UnsupportedOp(format!(
                    "BMC record '{name}' declares duplicate field '{field_name}'"
                )));
            }
            if !sort.is_scalar() && !matches!(sort, TlaSort::Set { .. }) {
                return Err(AYError::UnsupportedOp(format!(
                    "BMC record field must be scalar or Set (nested records require a \
                     recursive encoding), got {sort} for field \
                    '{field_name}' of record '{name}'"
                )));
            }
            if matches!(
                sort,
                TlaSort::Set { element_sort }
                    if (**element_sort).clone().canonicalized() == TlaSort::Bool
            ) {
                return Err(AYError::UnsupportedOp(format!(
                    "BMC record field '{field_name}' of '{name}' cannot use Set(Bool): no Bool-index encoding is defined"
                )));
            }
            ay_sorts.push(sort.to_ay()?);
        }
        self.ensure_declaration_carrier(name, BmcCarrierKind::Record)?;

        // Re-declaration is idempotent only for the same logical record shape.
        // Field order is not semantically significant in TLA+, so compare
        // canonicalized shapes while retaining the original declaration order
        // used to index `field_terms`.
        let requested_shape = TlaSort::Record {
            field_sorts: field_sorts.clone(),
        }
        .canonicalized();
        if let Some(existing) = self.record_vars.get(name) {
            let existing_shape = TlaSort::Record {
                field_sorts: existing.field_sorts.clone(),
            }
            .canonicalized();
            if existing_shape == requested_shape {
                return Ok(());
            }
            return Err(AYError::TypeMismatch {
                name: name.to_string(),
                expected: existing_shape.to_string(),
                actual: requested_shape.to_string(),
            });
        }

        let mut all_field_terms = Vec::with_capacity(field_sorts.len());

        for ((field_name, _), ay_sort) in field_sorts.iter().zip(ay_sorts) {
            // Scalar or Set: declare directly (Set maps to (Array Int Bool)).
            // The validation pass above rejects every representation that lacks
            // a complete per-step carrier before this loop mutates the solver.
            let mut step_terms = Vec::with_capacity(self.bound_k + 1);
            for step in 0..=self.bound_k {
                let var_name = Self::record_field_symbol(name, field_name, step);
                let term = self.solver.declare_const(&var_name, ay_sort.clone());
                step_terms.push(term);
            }
            all_field_terms.push(step_terms);
        }

        self.record_vars.insert(
            name.to_string(),
            BmcRecordVarInfo {
                field_sorts,
                field_terms: all_field_terms,
            },
        );
        Ok(())
    }

    // === Tuple variable declaration ===

    /// Declare a tuple state variable for all k+1 steps.
    ///
    /// Each tuple element becomes an independently and canonically named SMT
    /// variable per step (with a 1-indexed element number).
    ///
    /// The element sorts must be scalar (Bool, Int, or String).
    /// Re-declaration is idempotent only for the exact ordered element shape.
    ///
    /// Part of #3787: Tuple encoding in BMC translator.
    pub fn declare_tuple_var(&mut self, name: &str, element_sorts: Vec<TlaSort>) -> AYResult<()> {
        // Validate and convert the entire shape before mutating the solver.
        let mut ay_sorts = Vec::with_capacity(element_sorts.len());
        for (i, sort) in element_sorts.iter().enumerate() {
            if !sort.is_scalar() {
                return Err(AYError::UnsupportedOp(format!(
                    "BMC tuple element must be scalar, got {sort} for element {} \
                     of tuple '{name}'",
                    i + 1
                )));
            }
            ay_sorts.push(sort.to_ay()?);
        }
        self.ensure_declaration_carrier(name, BmcCarrierKind::Tuple)?;

        // Tuple order is semantically significant. Re-declaration is
        // idempotent only when every element sort at every index matches.
        let requested_shape = TlaSort::Tuple {
            element_sorts: element_sorts.clone(),
        }
        .canonicalized();
        if let Some(existing) = self.tuple_vars.get(name) {
            let existing_shape = TlaSort::Tuple {
                element_sorts: existing.element_sorts.clone(),
            }
            .canonicalized();
            if existing_shape == requested_shape {
                return Ok(());
            }
            return Err(AYError::TypeMismatch {
                name: name.to_string(),
                expected: existing_shape.to_string(),
                actual: requested_shape.to_string(),
            });
        }

        let mut all_element_terms = Vec::with_capacity(element_sorts.len());

        for (i, ay_sort) in ay_sorts.into_iter().enumerate() {
            let mut step_terms = Vec::with_capacity(self.bound_k + 1);
            for step in 0..=self.bound_k {
                let var_name = Self::tuple_element_symbol(name, i + 1, step);
                let term = self.solver.declare_const(&var_name, ay_sort.clone());
                step_terms.push(term);
            }
            all_element_terms.push(step_terms);
        }

        self.tuple_vars.insert(
            name.to_string(),
            BmcTupleVarInfo {
                element_sorts,
                element_terms: all_element_terms,
            },
        );
        Ok(())
    }

    // === Record field access ===

    /// Get the term for a specific record field at a given step.
    ///
    /// Part of #3787.
    pub(crate) fn get_record_field_at_step(
        &self,
        name: &str,
        field: &str,
        step: usize,
    ) -> AYResult<Term> {
        let info = self.record_vars.get(name).ok_or_else(|| {
            AYError::UnknownVariable(format!("record {name} (field {field}, step {step})"))
        })?;
        if step > self.bound_k {
            return Err(AYError::UntranslatableExpr(format!(
                "step {step} exceeds bound {}",
                self.bound_k
            )));
        }
        let field_idx = info
            .field_sorts
            .iter()
            .position(|(f, _)| f == field)
            .ok_or_else(|| {
                AYError::UntranslatableExpr(format!("record '{name}' has no field '{field}'"))
            })?;
        Ok(info.field_terms[field_idx][step])
    }

    // === Tuple element access ===

    /// Get the term for a specific tuple element at a given step.
    ///
    /// `index` is 1-based (TLA+ convention).
    ///
    /// Part of #3787.
    pub(crate) fn get_tuple_element_at_step(
        &self,
        name: &str,
        index: usize,
        step: usize,
    ) -> AYResult<Term> {
        let info = self.tuple_vars.get(name).ok_or_else(|| {
            AYError::UnknownVariable(format!("tuple {name} (element {index}, step {step})"))
        })?;
        if step > self.bound_k {
            return Err(AYError::UntranslatableExpr(format!(
                "step {step} exceeds bound {}",
                self.bound_k
            )));
        }
        if index == 0 || index > info.element_sorts.len() {
            return Err(AYError::UntranslatableExpr(format!(
                "tuple '{name}' index {index} out of bounds (1..={})",
                info.element_sorts.len()
            )));
        }
        Ok(info.element_terms[index - 1][step])
    }

    // === Record/tuple variable detection ===

    /// Check whether an expression refers to a declared record variable.
    pub(super) fn is_record_var_expr(&self, expr: &Spanned<Expr>) -> bool {
        match &expr.node {
            Expr::Ident(name, _) | Expr::StateVar(name, ..) => self.record_vars.contains_key(name),
            Expr::Prime(inner) => match &inner.node {
                Expr::Ident(name, _) | Expr::StateVar(name, ..) => {
                    self.record_vars.contains_key(name)
                }
                _ => false,
            },
            _ => false,
        }
    }

    /// Check whether an expression refers to a declared tuple variable.
    pub(super) fn is_tuple_var_expr(&self, expr: &Spanned<Expr>) -> bool {
        match &expr.node {
            Expr::Ident(name, _) | Expr::StateVar(name, ..) => self.tuple_vars.contains_key(name),
            Expr::Prime(inner) => match &inner.node {
                Expr::Ident(name, _) | Expr::StateVar(name, ..) => {
                    self.tuple_vars.contains_key(name)
                }
                _ => false,
            },
            _ => false,
        }
    }

    /// Resolve a record variable expression to `(variable_name, step)`.
    pub(super) fn resolve_record_var(&self, expr: &Spanned<Expr>) -> AYResult<(String, usize)> {
        match &expr.node {
            Expr::Ident(name, _) | Expr::StateVar(name, ..) => {
                if self.record_vars.contains_key(name) {
                    Ok((name.clone(), self.current_step))
                } else {
                    Err(AYError::UnknownVariable(format!("record {name}")))
                }
            }
            Expr::Prime(inner) => match &inner.node {
                Expr::Ident(name, _) | Expr::StateVar(name, ..) => {
                    if self.record_vars.contains_key(name) {
                        Ok((name.clone(), self.current_step + 1))
                    } else {
                        Err(AYError::UnknownVariable(format!("record {name}")))
                    }
                }
                _ => Err(AYError::UntranslatableExpr(
                    "BMC record operation requires variable reference".to_string(),
                )),
            },
            _ => Err(AYError::UntranslatableExpr(
                "BMC record operation requires variable reference".to_string(),
            )),
        }
    }

    /// Resolve a tuple variable expression to `(variable_name, step)`.
    pub(super) fn resolve_tuple_var(&self, expr: &Spanned<Expr>) -> AYResult<(String, usize)> {
        match &expr.node {
            Expr::Ident(name, _) | Expr::StateVar(name, ..) => {
                if self.tuple_vars.contains_key(name) {
                    Ok((name.clone(), self.current_step))
                } else {
                    Err(AYError::UnknownVariable(format!("tuple {name}")))
                }
            }
            Expr::Prime(inner) => match &inner.node {
                Expr::Ident(name, _) | Expr::StateVar(name, ..) => {
                    if self.tuple_vars.contains_key(name) {
                        Ok((name.clone(), self.current_step + 1))
                    } else {
                        Err(AYError::UnknownVariable(format!("tuple {name}")))
                    }
                }
                _ => Err(AYError::UntranslatableExpr(
                    "BMC tuple operation requires variable reference".to_string(),
                )),
            },
            _ => Err(AYError::UntranslatableExpr(
                "BMC tuple operation requires variable reference".to_string(),
            )),
        }
    }

    // === Record construction ===

    /// Translate a record construction `[a |-> e1, b |-> e2]` into per-field
    /// constraints on a fresh set of variables.
    ///
    /// Returns a conjunction of equalities: `fresh_a = e1 /\ fresh_b = e2`.
    /// The fresh variables are stored for later access.
    ///
    /// Part of #3787.
    pub(super) fn translate_record_construct(
        &mut self,
        fields: &[(Spanned<String>, Spanned<Expr>)],
    ) -> AYResult<Term> {
        if fields.is_empty() {
            return Ok(self.solver.bool_const(true));
        }

        let mut conjuncts = Vec::with_capacity(fields.len());

        for (field_name, value) in fields {
            let purpose = format!("record literal field {}", field_name.node);

            // Try translating as Int first, fall back to Bool
            if let Ok(val_term) = dispatch_translate_int(self, value) {
                let (_, fresh) = self.declare_internal_const(&purpose, ay_dpll::api::Sort::Int);
                let eq = self.solver.try_eq(fresh, val_term)?;
                conjuncts.push(eq);
            } else {
                let val_term = dispatch_translate_bool(self, value)?;
                let (_, fresh) = self.declare_internal_const(&purpose, ay_dpll::api::Sort::Bool);
                // Bool equality: (fresh => val) /\ (val => fresh)
                let fwd = self.solver.try_implies(fresh, val_term)?;
                let bwd = self.solver.try_implies(val_term, fresh)?;
                let eq_both = self.solver.try_and(fwd, bwd)?;
                conjuncts.push(eq_both);
            }
        }

        self.build_conjunction(conjuncts)
    }

    /// Translate record field access `r.field` into the appropriate field term.
    ///
    /// Resolves the record variable and step, then looks up the field term.
    ///
    /// Part of #3787.
    pub(super) fn translate_record_access(
        &mut self,
        record_expr: &Spanned<Expr>,
        field_name: &str,
    ) -> AYResult<Term> {
        let (name, step) = self.resolve_record_var(record_expr)?;
        self.get_record_field_at_step(&name, field_name, step)
    }

    /// Translate record EXCEPT `[r EXCEPT !.a = v]` as equality constraints.
    ///
    /// For a record with fields {a, b, c}:
    /// - `target.a = v`  (the overridden field)
    /// - `target.b = source.b`  (copied)
    /// - `target.c = source.c`  (copied)
    ///
    /// Returns a conjunction of per-field equalities between target and
    /// source (with overrides applied).
    ///
    /// Part of #3787.
    pub(super) fn translate_record_except_eq(
        &mut self,
        target: &(String, usize),
        source: &(String, usize),
        specs: &[ExceptSpec],
    ) -> AYResult<Term> {
        let source_field_sorts = self
            .record_vars
            .get(&source.0)
            .ok_or_else(|| AYError::UnknownVariable(format!("record {}", source.0)))?
            .field_sorts
            .clone();
        let target_field_sorts = self
            .record_vars
            .get(&target.0)
            .ok_or_else(|| AYError::UnknownVariable(format!("record {}", target.0)))?
            .field_sorts
            .clone();

        let field_names: Vec<String> = source_field_sorts
            .iter()
            .map(|(name, _)| name.clone())
            .collect();
        let num_fields = field_names.len();

        // Collect all field overrides from EXCEPT specs
        let mut overrides: std::collections::HashMap<String, &Spanned<Expr>> =
            std::collections::HashMap::new();
        for spec in specs {
            match spec.path.as_slice() {
                [ExceptPathElement::Field(field)] => {
                    overrides.insert(field.name.node.clone(), &spec.value);
                }
                _ => {
                    return Err(AYError::UnsupportedOp(
                        "BMC record EXCEPT requires exactly one direct .field path element; \
                         nested and dynamic paths require a recursive encoding"
                            .to_string(),
                    ));
                }
            }
        }

        // Derive the result shape before translating any override. EXCEPT
        // preserves the source DOMAIN but may change an overridden field's
        // value kind. Thus `target.a:String = [source.a:Int EXCEPT !.a = "x"]`
        // is supported, while a copied or overridden field that does not match
        // the target is FALSE. Unknown direct fields remain the TLA+ no-op.
        let mut result_field_sorts = Vec::with_capacity(source_field_sorts.len());
        for (field_name, source_sort) in &source_field_sorts {
            let result_sort = if let Some(value_expr) = overrides.get(field_name) {
                self.value_expr_sort(value_expr).ok_or_else(|| {
                    AYError::UnsupportedOp(format!(
                        "BMC cannot determine the sort of record EXCEPT field '{field_name}'"
                    ))
                })?
            } else {
                source_sort.clone()
            };
            result_field_sorts.push((field_name.clone(), result_sort));
        }
        if !record_shapes_equal_or_false(
            &target_field_sorts,
            &result_field_sorts,
            "record EXCEPT equality",
        )? {
            return Ok(self.solver.bool_const(false));
        }

        let mut conjuncts = Vec::with_capacity(num_fields);

        for field_name in &field_names {
            let target_term = self.get_record_field_at_step(&target.0, field_name, target.1)?;

            if let Some(value_expr) = overrides.get(field_name) {
                // Overridden field: target.field = value
                let field_sort = target_field_sorts
                    .iter()
                    .find(|(candidate, _)| candidate == field_name)
                    .map(|(_, sort)| sort)
                    .ok_or_else(|| {
                        AYError::UnsupportedOp(format!(
                            "record EXCEPT target is missing field '{field_name}'"
                        ))
                    })?;

                let val_term = match field_sort {
                    TlaSort::Bool => dispatch_translate_bool(self, value_expr)?,
                    TlaSort::Int => dispatch_translate_int(self, value_expr)?,
                    TlaSort::String if self.is_string_scalar(value_expr) => {
                        self.string_scalar_term(value_expr)?
                    }
                    TlaSort::String => dispatch_translate_int(self, value_expr)?,
                    compound => {
                        return Err(AYError::UnsupportedOp(format!(
                            "BMC record EXCEPT does not support field '{field_name}' with sort \
                             {compound}"
                        )))
                    }
                };

                let eq = self.solver.try_eq(target_term, val_term)?;
                conjuncts.push(eq);
            } else {
                // Copied field: target.field = source.field
                let source_term = self.get_record_field_at_step(&source.0, field_name, source.1)?;
                let eq = self.solver.try_eq(target_term, source_term)?;
                conjuncts.push(eq);
            }
        }

        self.build_conjunction(conjuncts)
    }

    // === Tuple construction ===

    /// Translate a tuple construction `<<e1, e2, e3>>` as per-element
    /// constraints on fresh variables.
    ///
    /// Returns a conjunction: `fresh_1 = e1 /\ fresh_2 = e2 /\ ...`
    ///
    /// Part of #3787.
    #[allow(dead_code)]
    pub(super) fn translate_tuple_construct(
        &mut self,
        elements: &[Spanned<Expr>],
    ) -> AYResult<Term> {
        if elements.is_empty() {
            return Ok(self.solver.bool_const(true));
        }

        let mut conjuncts = Vec::with_capacity(elements.len());

        for (i, elem) in elements.iter().enumerate() {
            let purpose = format!("tuple literal element {}", i + 1);

            // Try translating as Int first, fall back to Bool
            if let Ok(val_term) = dispatch_translate_int(self, elem) {
                let (_, fresh) = self.declare_internal_const(&purpose, ay_dpll::api::Sort::Int);
                let eq = self.solver.try_eq(fresh, val_term)?;
                conjuncts.push(eq);
            } else {
                let val_term = dispatch_translate_bool(self, elem)?;
                let (_, fresh) = self.declare_internal_const(&purpose, ay_dpll::api::Sort::Bool);
                // Bool equality: (fresh => val) /\ (val => fresh)
                let fwd = self.solver.try_implies(fresh, val_term)?;
                let bwd = self.solver.try_implies(val_term, fresh)?;
                let eq_both = self.solver.try_and(fwd, bwd)?;
                conjuncts.push(eq_both);
            }
        }

        self.build_conjunction(conjuncts)
    }

    /// Translate tuple indexing `t[i]` into the appropriate element term.
    ///
    /// Resolves the tuple variable and step, then looks up the element term.
    /// The index must be a constant integer literal.
    ///
    /// Part of #3787.
    pub(super) fn translate_tuple_index(
        &mut self,
        tuple_expr: &Spanned<Expr>,
        index_expr: &Spanned<Expr>,
    ) -> AYResult<Term> {
        let (name, step) = self.resolve_tuple_var(tuple_expr)?;

        // Index must be a constant integer. Constant-fold simple integer
        // arithmetic first (e.g. `colors[i + 1]` after a quantifier substitutes
        // `i := 3` becomes `colors[3 + 1]`, whose index node is `Add(3, 1)`, not
        // a bare `Int(4)`). Folding here keeps tuple indexing exact for the
        // common `t[i + k]` neighbour-access idiom.
        let index = match const_fold_int_index(index_expr) {
            Some(i) if i >= 1 => i as usize,
            Some(i) => {
                return Err(AYError::UntranslatableExpr(format!(
                    "BMC tuple index out of range (got {i}, must be >= 1)"
                )));
            }
            None => {
                return Err(AYError::UntranslatableExpr(
                    "BMC tuple indexing requires constant integer index".to_string(),
                ));
            }
        };

        self.get_tuple_element_at_step(&name, index, step)
    }

    // === UNCHANGED for record/tuple ===

    /// Translate UNCHANGED for a record variable.
    ///
    /// Produces: `r.a' = r.a /\ r.b' = r.b /\ ...` for all fields.
    ///
    /// Part of #3787.
    pub(super) fn translate_unchanged_record(&mut self, name: &str) -> AYResult<Term> {
        let info = self
            .record_vars
            .get(name)
            .ok_or_else(|| AYError::UnknownVariable(format!("record {name}")))?;

        let num_fields = info.field_sorts.len();
        let mut conjuncts = Vec::with_capacity(num_fields);

        for field_idx in 0..num_fields {
            let current = info.field_terms[field_idx][self.current_step];
            let next = info.field_terms[field_idx][self.current_step + 1];
            let eq = self.solver.try_eq(next, current)?;
            conjuncts.push(eq);
        }

        self.build_conjunction(conjuncts)
    }

    /// Translate UNCHANGED for a tuple variable.
    ///
    /// Produces: `t[1]' = t[1] /\ t[2]' = t[2] /\ ...` for all elements.
    ///
    /// Part of #3787.
    pub(super) fn translate_unchanged_tuple(&mut self, name: &str) -> AYResult<Term> {
        let info = self
            .tuple_vars
            .get(name)
            .ok_or_else(|| AYError::UnknownVariable(format!("tuple {name}")))?;

        let num_elements = info.element_sorts.len();
        let mut conjuncts = Vec::with_capacity(num_elements);

        for elem_idx in 0..num_elements {
            let current = info.element_terms[elem_idx][self.current_step];
            let next = info.element_terms[elem_idx][self.current_step + 1];
            let eq = self.solver.try_eq(next, current)?;
            conjuncts.push(eq);
        }

        self.build_conjunction(conjuncts)
    }

    // === Try-translate helpers for equality dispatch ===

    /// Try to translate record EXCEPT equality: `r' = [r EXCEPT !.a = v]`.
    ///
    /// Returns `None` if neither side involves a record EXCEPT.
    /// Returns `Some(result)` if record EXCEPT equality is detected.
    ///
    /// Part of #3787.
    pub(super) fn try_translate_record_except_eq(
        &mut self,
        left: &Spanned<Expr>,
        right: &Spanned<Expr>,
    ) -> Option<AYResult<Term>> {
        if let Some(result) = self.try_translate_record_except_eq_directed(left, right) {
            return Some(result);
        }
        self.try_translate_record_except_eq_directed(right, left)
    }

    /// Try record EXCEPT equality in one direction:
    /// lhs is a (possibly primed) record variable, rhs is an EXCEPT expression.
    fn try_translate_record_except_eq_directed(
        &mut self,
        lhs: &Spanned<Expr>,
        rhs: &Spanned<Expr>,
    ) -> Option<AYResult<Term>> {
        // rhs must be Except(base, specs) with a record variable base
        let (base, specs) = match &rhs.node {
            Expr::Except(base, specs) if self.is_record_var_expr(base) => {
                (base.as_ref(), specs.as_slice())
            }
            _ => return None,
        };

        // lhs must be a (possibly primed) record variable
        if !self.is_record_var_expr(lhs) {
            return None;
        }

        let target = match self.resolve_record_var(lhs) {
            Ok(r) => r,
            Err(e) => return Some(Err(e)),
        };
        let source = match self.resolve_record_var(base) {
            Ok(r) => r,
            Err(e) => return Some(Err(e)),
        };

        Some(self.translate_record_except_eq(&target, &source, specs))
    }

    /// Try to translate record field equality: `r' = [a |-> e1, b |-> e2]`
    /// or `r.field = expr` patterns.
    ///
    /// Returns `None` if neither side is record-related.
    ///
    /// Part of #3787.
    pub(super) fn try_translate_record_eq(
        &mut self,
        left: &Spanned<Expr>,
        right: &Spanned<Expr>,
    ) -> Option<AYResult<Term>> {
        // Try: record_var = [a |-> e1, b |-> e2]
        if let Some(result) = self.try_translate_record_construct_eq(left, right) {
            return Some(result);
        }
        if let Some(result) = self.try_translate_record_construct_eq(right, left) {
            return Some(result);
        }

        // Try: record_var1 = record_var2
        if self.is_record_var_expr(left) && self.is_record_var_expr(right) {
            let l = match self.resolve_record_var(left) {
                Ok(r) => r,
                Err(e) => return Some(Err(e)),
            };
            let r = match self.resolve_record_var(right) {
                Ok(r) => r,
                Err(e) => return Some(Err(e)),
            };
            return Some(self.translate_record_var_eq(&l, &r));
        }

        None
    }

    /// Try record construction equality: lhs is a record var, rhs is Record literal.
    fn try_translate_record_construct_eq(
        &mut self,
        lhs: &Spanned<Expr>,
        rhs: &Spanned<Expr>,
    ) -> Option<AYResult<Term>> {
        if !self.is_record_var_expr(lhs) {
            return None;
        }

        let fields = match &rhs.node {
            Expr::Record(fields) => fields,
            _ => return None,
        };

        let target = match self.resolve_record_var(lhs) {
            Ok(r) => r,
            Err(e) => return Some(Err(e)),
        };

        Some(self.translate_record_literal_eq(&target, fields))
    }

    /// Return the sort of a compound-literal component expression when it is
    /// known without translating (and therefore without mutating the solver).
    ///
    /// Record equality must compare shapes before it emits any pointwise
    /// constraints. In particular, `Int` and `String` share BMC's SMT `Int`
    /// representation, so allowing their terms to reach `try_eq` would erase a
    /// real TLA+ sort mismatch.
    fn value_expr_sort(&self, expr: &Spanned<Expr>) -> Option<TlaSort> {
        match &expr.node {
            Expr::Bool(_) => Some(TlaSort::Bool),
            Expr::Int(_) => Some(TlaSort::Int),
            Expr::String(_) => Some(TlaSort::String),
            Expr::Ident(name, _) | Expr::StateVar(name, ..) => {
                if let Some(info) = self.vars.get(name) {
                    return Some(info.sort.clone());
                }
                if let Some(info) = self.record_vars.get(name) {
                    return Some(
                        TlaSort::Record {
                            field_sorts: info.field_sorts.clone(),
                        }
                        .canonicalized(),
                    );
                }
                self.tuple_vars.get(name).map(|info| TlaSort::Tuple {
                    element_sorts: info.element_sorts.clone(),
                })
            }
            Expr::Prime(inner) => self.value_expr_sort(inner),
            Expr::Label(label) => self.value_expr_sort(&label.body),
            Expr::SubstIn(_, body) => self.value_expr_sort(body),
            Expr::If(_, then_expr, else_expr) => {
                let then_sort = self.value_expr_sort(then_expr)?.canonicalized();
                let else_sort = self.value_expr_sort(else_expr)?.canonicalized();
                (then_sort == else_sort).then_some(then_sort)
            }
            Expr::And(..)
            | Expr::Or(..)
            | Expr::Not(..)
            | Expr::Implies(..)
            | Expr::Equiv(..)
            | Expr::Forall(..)
            | Expr::Exists(..)
            | Expr::In(..)
            | Expr::NotIn(..)
            | Expr::Subseteq(..)
            | Expr::Eq(..)
            | Expr::Neq(..)
            | Expr::Lt(..)
            | Expr::Leq(..)
            | Expr::Gt(..)
            | Expr::Geq(..)
            | Expr::Always(..)
            | Expr::Eventually(..)
            | Expr::LeadsTo(..)
            | Expr::WeakFair(..)
            | Expr::StrongFair(..)
            | Expr::Enabled(..) => Some(TlaSort::Bool),
            Expr::Add(..)
            | Expr::Sub(..)
            | Expr::Mul(..)
            | Expr::Div(..)
            | Expr::IntDiv(..)
            | Expr::Mod(..)
            | Expr::Pow(..)
            | Expr::Neg(..) => Some(TlaSort::Int),
            Expr::Range(..) => Some(TlaSort::Set {
                element_sort: Box::new(TlaSort::Int),
            }),
            Expr::RecordAccess(record, field) => {
                let (name, _) = self.resolve_record_var(record).ok()?;
                self.record_vars
                    .get(&name)?
                    .field_sorts
                    .iter()
                    .find(|(candidate, _)| candidate == &field.name.node)
                    .map(|(_, sort)| sort.clone().canonicalized())
            }
            Expr::Record(fields) => {
                let mut field_sorts = Vec::with_capacity(fields.len());
                for (name, value) in fields {
                    field_sorts.push((
                        name.node.clone(),
                        self.value_expr_sort(value)?.canonicalized(),
                    ));
                }
                field_sorts.sort_by(|left, right| left.0.cmp(&right.0));
                if field_sorts.windows(2).any(|pair| pair[0].0 == pair[1].0) {
                    return None;
                }
                Some(TlaSort::Record { field_sorts })
            }
            _ => None,
        }
    }

    /// Translate `target = [a |-> e1, b |-> e2]` as per-field equalities.
    fn translate_record_literal_eq(
        &mut self,
        target: &(String, usize),
        fields: &[(Spanned<String>, Spanned<Expr>)],
    ) -> AYResult<Term> {
        let target_field_sorts = self
            .record_vars
            .get(&target.0)
            .ok_or_else(|| AYError::UnknownVariable(format!("record {}", target.0)))?
            .field_sorts
            .clone();

        // TLA+ record equality includes DOMAIN equality. Comparing only the
        // fields mentioned by the literal is unsound under negation: for a
        // target with fields {a, b}, translating `target = [a |-> 0]` as merely
        // `target.a = 0` turns the false equality into a satisfiable one, and
        // turns its negation into the stronger `target.a # 0`. Require the
        // complete field-name set before building any pointwise constraints.
        let mut target_names: Vec<&str> = target_field_sorts
            .iter()
            .map(|(name, _)| name.as_str())
            .collect();
        let mut literal_names: Vec<&str> =
            fields.iter().map(|(name, _)| name.node.as_str()).collect();
        target_names.sort_unstable();
        literal_names.sort_unstable();
        let target_has_duplicate = target_names.windows(2).any(|pair| pair[0] == pair[1]);
        let literal_has_duplicate = literal_names.windows(2).any(|pair| pair[0] == pair[1]);
        if target_has_duplicate || literal_has_duplicate || target_names != literal_names {
            return Ok(self.solver.bool_const(false));
        }

        // Compare every statically known literal field sort before translating
        // any value. This keeps unequal shapes equal to the Boolean constant
        // FALSE in both operand orders, including under negation, and avoids
        // leaking partial auxiliary constraints from an earlier field.
        for (field_name, field_sort) in &target_field_sorts {
            let value_expr = fields
                .iter()
                .find(|(literal_name, _)| literal_name.node == *field_name)
                .map(|(_, value)| value)
                .ok_or_else(|| {
                    AYError::UnsupportedOp(format!(
                        "record literal is missing field '{field_name}'"
                    ))
                })?;
            let literal_sort = self.value_expr_sort(value_expr).ok_or_else(|| {
                AYError::UnsupportedOp(format!(
                    "BMC cannot determine the sort of record literal field '{field_name}'"
                ))
            })?;
            let field_sort = field_sort.clone().canonicalized();
            let literal_sort = literal_sort.canonicalized();
            if field_sort != literal_sort {
                if matches!(
                    (&field_sort, &literal_sort),
                    (TlaSort::Set { .. }, TlaSort::Set { .. })
                ) {
                    return Err(AYError::UnsupportedOp(format!(
                        "BMC cannot decide record-literal equality for field '{field_name}' \
                         with differing set element sorts {field_sort} and {literal_sort}"
                    )));
                }
                return Ok(self.solver.bool_const(false));
            }
        }

        let mut conjuncts = Vec::with_capacity(target_field_sorts.len());

        for (field_name, field_sort) in &target_field_sorts {
            let value_expr = fields
                .iter()
                .find(|(literal_name, _)| literal_name.node == *field_name)
                .map(|(_, value)| value)
                // The exact-name-set check above makes this unreachable, but
                // keep the translator fail-closed if that invariant changes.
                .ok_or_else(|| {
                    AYError::UnsupportedOp(format!(
                        "record literal is missing field '{field_name}'"
                    ))
                })?;
            let target_term = self.get_record_field_at_step(&target.0, field_name, target.1)?;

            let val_term = match field_sort {
                TlaSort::Bool => dispatch_translate_bool(self, value_expr)?,
                TlaSort::Int => dispatch_translate_int(self, value_expr)?,
                TlaSort::String if self.is_string_scalar(value_expr) => {
                    self.string_scalar_term(value_expr)?
                }
                TlaSort::String => dispatch_translate_int(self, value_expr)?,
                compound => {
                    return Err(AYError::UnsupportedOp(format!(
                        "BMC record-literal equality does not support field '{field_name}' \
                         with sort {compound}"
                    )))
                }
            };

            let eq = self.solver.try_eq(target_term, val_term)?;
            conjuncts.push(eq);
        }

        self.build_conjunction(conjuncts)
    }

    /// Translate equality between two record variables: all fields must match.
    fn translate_record_var_eq(
        &mut self,
        lhs: &(String, usize),
        rhs: &(String, usize),
    ) -> AYResult<Term> {
        let lhs_field_sorts = self
            .record_vars
            .get(&lhs.0)
            .ok_or_else(|| AYError::UnknownVariable(format!("record {}", lhs.0)))?
            .field_sorts
            .clone();
        let rhs_field_sorts = self
            .record_vars
            .get(&rhs.0)
            .ok_or_else(|| AYError::UnknownVariable(format!("record {}", rhs.0)))?
            .field_sorts
            .clone();

        // Equality is symmetric and includes the complete record DOMAIN. Sort
        // each declaration by field name so declaration order is irrelevant,
        // reject malformed duplicate names, and return the TLA value FALSE for
        // any field-name or field-sort mismatch. Iterating only the lhs fields
        // made `small = large` weaker while `large = small` errored.
        if !record_shapes_equal_or_false(
            &lhs_field_sorts,
            &rhs_field_sorts,
            "record-variable equality",
        )? {
            return Ok(self.solver.bool_const(false));
        }

        let lhs_shape = canonical_record_shape(&lhs_field_sorts)
            .ok_or_else(|| AYError::UnsupportedOp("duplicate record field shape".to_string()))?;

        let mut conjuncts = Vec::with_capacity(lhs_shape.len());

        for (field_name, _) in &lhs_shape {
            let l_term = self.get_record_field_at_step(&lhs.0, field_name, lhs.1)?;
            let r_term = self.get_record_field_at_step(&rhs.0, field_name, rhs.1)?;
            let eq = self.solver.try_eq(l_term, r_term)?;
            conjuncts.push(eq);
        }

        self.build_conjunction(conjuncts)
    }

    // === Tuple equality dispatch ===

    /// Try to translate tuple equality: `t' = <<e1, e2>>` or `t = t'`.
    ///
    /// Part of #3787.
    pub(super) fn try_translate_tuple_eq(
        &mut self,
        left: &Spanned<Expr>,
        right: &Spanned<Expr>,
    ) -> Option<AYResult<Term>> {
        // Try: tuple_var = <<e1, e2>>
        if let Some(result) = self.try_translate_tuple_literal_eq(left, right) {
            return Some(result);
        }
        if let Some(result) = self.try_translate_tuple_literal_eq(right, left) {
            return Some(result);
        }

        // Try: tuple_var1 = tuple_var2
        if self.is_tuple_var_expr(left) && self.is_tuple_var_expr(right) {
            let l = match self.resolve_tuple_var(left) {
                Ok(r) => r,
                Err(e) => return Some(Err(e)),
            };
            let r = match self.resolve_tuple_var(right) {
                Ok(r) => r,
                Err(e) => return Some(Err(e)),
            };
            return Some(self.translate_tuple_var_eq(&l, &r));
        }

        None
    }

    /// Try tuple literal equality: lhs is tuple var, rhs is Tuple literal.
    fn try_translate_tuple_literal_eq(
        &mut self,
        lhs: &Spanned<Expr>,
        rhs: &Spanned<Expr>,
    ) -> Option<AYResult<Term>> {
        if !self.is_tuple_var_expr(lhs) {
            return None;
        }

        let elements = match &rhs.node {
            Expr::Tuple(elems) => elems,
            _ => return None,
        };

        let target = match self.resolve_tuple_var(lhs) {
            Ok(r) => r,
            Err(e) => return Some(Err(e)),
        };

        Some(self.translate_tuple_literal_eq(&target, elements))
    }

    /// Translate `target = <<e1, e2, e3>>` as per-element equalities.
    fn translate_tuple_literal_eq(
        &mut self,
        target: &(String, usize),
        elements: &[Spanned<Expr>],
    ) -> AYResult<Term> {
        let target_element_sorts = self
            .tuple_vars
            .get(&target.0)
            .ok_or_else(|| AYError::UnknownVariable(format!("tuple {}", target.0)))?
            .element_sorts
            .clone();

        // Tuple equality includes exact arity and element value kinds. Validate
        // the entire literal before translating an element so short literals
        // cannot leave suffix elements unconstrained and Int/String cannot
        // alias through their shared SMT Int carrier.
        if target_element_sorts.len() != elements.len() {
            return Ok(self.solver.bool_const(false));
        }
        for (index, (element_sort, element_expr)) in
            target_element_sorts.iter().zip(elements).enumerate()
        {
            let literal_sort = self.value_expr_sort(element_expr).ok_or_else(|| {
                AYError::UnsupportedOp(format!(
                    "BMC cannot determine the sort of tuple literal element {}",
                    index + 1
                ))
            })?;
            if element_sort.clone().canonicalized() != literal_sort.canonicalized() {
                return Ok(self.solver.bool_const(false));
            }
        }

        let mut conjuncts = Vec::with_capacity(elements.len());

        for (i, (elem, elem_sort)) in elements.iter().zip(&target_element_sorts).enumerate() {
            let index_1based = i + 1;
            let target_term = self.get_tuple_element_at_step(&target.0, index_1based, target.1)?;

            let val_term = match elem_sort {
                TlaSort::Bool => dispatch_translate_bool(self, elem)?,
                TlaSort::Int => dispatch_translate_int(self, elem)?,
                TlaSort::String if self.is_string_scalar(elem) => self.string_scalar_term(elem)?,
                TlaSort::String => dispatch_translate_int(self, elem)?,
                compound => {
                    return Err(AYError::UnsupportedOp(format!(
                        "BMC tuple-literal equality does not support element {} with sort \
                         {compound}",
                        i + 1
                    )))
                }
            };

            let eq = self.solver.try_eq(target_term, val_term)?;
            conjuncts.push(eq);
        }

        self.build_conjunction(conjuncts)
    }

    /// Translate equality between two tuple variables: all elements must match.
    fn translate_tuple_var_eq(
        &mut self,
        lhs: &(String, usize),
        rhs: &(String, usize),
    ) -> AYResult<Term> {
        let lhs_element_sorts = self
            .tuple_vars
            .get(&lhs.0)
            .ok_or_else(|| AYError::UnknownVariable(format!("tuple {}", lhs.0)))?
            .element_sorts
            .iter()
            .cloned()
            .map(TlaSort::canonicalized)
            .collect::<Vec<_>>();
        let rhs_element_sorts = self
            .tuple_vars
            .get(&rhs.0)
            .ok_or_else(|| AYError::UnknownVariable(format!("tuple {}", rhs.0)))?
            .element_sorts
            .iter()
            .cloned()
            .map(TlaSort::canonicalized)
            .collect::<Vec<_>>();

        // Tuple equality is symmetric and exact in arity and per-index sorts.
        // Comparing only the lhs prefix made short = long weaker while the
        // reverse orientation errored, and Int/String could alias in SMT.
        if lhs_element_sorts != rhs_element_sorts {
            return Ok(self.solver.bool_const(false));
        }

        let num_elems = lhs_element_sorts.len();
        let mut conjuncts = Vec::with_capacity(num_elems);

        for i in 0..num_elems {
            let l_term = self.get_tuple_element_at_step(&lhs.0, i + 1, lhs.1)?;
            let r_term = self.get_tuple_element_at_step(&rhs.0, i + 1, rhs.1)?;
            let eq = self.solver.try_eq(l_term, r_term)?;
            conjuncts.push(eq);
        }

        self.build_conjunction(conjuncts)
    }

    // === Shared helpers ===

    /// Build a conjunction from a list of terms.
    fn build_conjunction(&mut self, mut conjuncts: Vec<Term>) -> AYResult<Term> {
        if conjuncts.is_empty() {
            return Ok(self.solver.bool_const(true));
        }
        if conjuncts.len() == 1 {
            return Ok(conjuncts.pop().expect("invariant: len checked == 1"));
        }
        let mut result = conjuncts.pop().expect("invariant: len checked > 1");
        for c in conjuncts.into_iter().rev() {
            result = self.solver.try_and(c, result)?;
        }
        Ok(result)
    }
}

/// Constant-fold a tuple/sequence index expression to an `i64`.
///
/// Handles integer literals and the literal-only arithmetic that arises after a
/// bounded quantifier substitutes its variable (`t[i + 1]`, `t[i - 1]`, etc.).
/// Returns `None` if the index is not a constant integer expression, so the
/// caller falls back to its existing (non-constant) handling.
pub(super) fn const_fold_int_index(expr: &Spanned<Expr>) -> Option<i64> {
    match &expr.node {
        Expr::Int(n) => i64::try_from(n).ok(),
        Expr::Neg(a) => const_fold_int_index(a)?.checked_neg(),
        Expr::Add(a, b) => const_fold_int_index(a)?.checked_add(const_fold_int_index(b)?),
        Expr::Sub(a, b) => const_fold_int_index(a)?.checked_sub(const_fold_int_index(b)?),
        Expr::Mul(a, b) => const_fold_int_index(a)?.checked_mul(const_fold_int_index(b)?),
        _ => None,
    }
}
