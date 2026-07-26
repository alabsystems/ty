// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Compound type dispatch helpers for the BMC expression translator.
//!
//! Wires set operations (union, intersect, subset, cardinality), function
//! operations (EXCEPT, DOMAIN, construction), and universe extraction into
//! the BMC translator's expression evaluation path.
//!
//! Part of #3778: wire compound type encoders into BMC expression translator.

use ay_dpll::api::{Sort, Term};
use tla_core::ast::Expr;
use tla_core::Spanned;

use crate::error::{AYError, AYResult};

use super::BmcTranslator;

/// Set binary operation kind for dispatch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum SetBinOp {
    Union,
    Intersect,
    Minus,
}

/// Upper bound for finite supports materialized into pointwise BMC terms.
/// Larger supports would create thousands of array stores/selects, so decline
/// them before iteration or solver mutation.
pub(super) const MAX_BMC_SET_MATERIALIZATION: usize = 4096;

#[derive(Debug, Clone, PartialEq, Eq)]
enum SetElementEvidence {
    /// The syntactic empty set literal has no element kind and can compare
    /// across metadata boundaries.
    Empty,
    Known(crate::TlaSort),
    /// Nonempty-or-symbolic set expression whose element kind is unavailable.
    Unknown,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
enum ConcreteSetElement {
    Int(i64),
    String(String),
}

/// Canonical value for a closed set literal.  Nested sets are represented by
/// sorted, duplicate-free element vectors so equality is extensional rather
/// than dependent on source order or duplicate spelling.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
enum ClosedSetElement {
    Bool(bool),
    Int(num_bigint::BigInt),
    String(String),
    Set(Vec<ClosedSetElement>),
}

impl BmcTranslator {
    /// Return whether `expr` has a statically complete finite support whose
    /// terms can be collected before encoding. A set may be array-translatable
    /// without having such a support (for example a symbolic set variable).
    pub(super) fn set_has_complete_finite_support(&self, expr: &Spanned<Expr>) -> bool {
        match &expr.node {
            Expr::SetEnum(elements) => {
                elements.len() <= MAX_BMC_SET_MATERIALIZATION
                    && elements.iter().all(|element| {
                        matches!(element.node, Expr::Int(_) | Expr::String(_))
                            || matches!(&element.node, Expr::Neg(inner) if matches!(inner.node, Expr::Int(_)))
                    })
            }
            Expr::Range(lo, hi) => Self::checked_range_literals(lo, hi, "finite support").is_ok(),
            Expr::Union(left, right) => {
                self.set_has_complete_finite_support(left)
                    && self.set_has_complete_finite_support(right)
            }
            Expr::Intersect(left, right) => {
                self.set_has_complete_finite_support(left)
                    || self.set_has_complete_finite_support(right)
            }
            Expr::SetMinus(left, _) => self.set_has_complete_finite_support(left),
            Expr::Label(label) => self.set_has_complete_finite_support(&label.body),
            Expr::Prime(inner) => self.set_has_complete_finite_support(inner),
            _ => false,
        }
    }

    /// Validate that an expression can be represented as one exact set array.
    /// Compound arrays require a complete support because AY has no pointwise
    /// array-map operator; without it, unconstrained out-of-support cells would
    /// make equality/cardinality/subset reasoning unsound.
    pub(super) fn ensure_set_expr_array_exact(
        &self,
        expr: &Spanned<Expr>,
        context: &str,
    ) -> AYResult<()> {
        self.require_supported_set_element_kind(expr, context)?;
        match &expr.node {
            Expr::SetEnum(elements) => {
                Self::ensure_materialization_size(elements.len(), context)
            }
            Expr::Range(lo, hi) => {
                Self::checked_range_literals(lo, hi, context)?;
                Ok(())
            }
            Expr::Ident(name, _) | Expr::StateVar(name, ..) => match self.vars.get(name) {
                Some(info) if matches!(info.sort, crate::TlaSort::Set { .. }) => Ok(()),
                Some(info) => Err(AYError::TypeMismatch {
                    name: name.clone(),
                    expected: "Set".to_string(),
                    actual: info.sort.to_string(),
                }),
                None => Err(AYError::UnknownVariable(format!("{name} (set expression)"))),
            },
            Expr::Prime(inner) => self.ensure_set_expr_array_exact(inner, context),
            Expr::Label(label) => self.ensure_set_expr_array_exact(&label.body, context),
            Expr::SubstIn(..) => Err(AYError::UnsupportedOp(format!(
                "BMC {context} cannot materialize deferred INSTANCE substitution sets"
            ))),
            Expr::Union(left, right) => {
                self.ensure_compatible_set_element_kinds(left, right, context)?;
                self.ensure_set_expr_array_exact(left, context)?;
                self.ensure_set_expr_array_exact(right, context)?;
                if self.set_has_complete_finite_support(left)
                    && self.set_has_complete_finite_support(right)
                {
                    Ok(())
                } else {
                    Err(AYError::UnsupportedOp(format!(
                        "BMC {context} requires complete finite support for both union operands"
                    )))
                }
            }
            Expr::Intersect(left, right) => {
                self.ensure_compatible_set_element_kinds(left, right, context)?;
                self.ensure_set_expr_array_exact(left, context)?;
                self.ensure_set_expr_array_exact(right, context)?;
                if self.set_has_complete_finite_support(left)
                    || self.set_has_complete_finite_support(right)
                {
                    Ok(())
                } else {
                    Err(AYError::UnsupportedOp(format!(
                        "BMC {context} requires complete finite support for at least one intersection operand"
                    )))
                }
            }
            Expr::SetMinus(left, right) => {
                self.ensure_compatible_set_element_kinds(left, right, context)?;
                self.ensure_set_expr_array_exact(left, context)?;
                self.ensure_set_expr_array_exact(right, context)?;
                if self.set_has_complete_finite_support(left) {
                    Ok(())
                } else {
                    Err(AYError::UnsupportedOp(format!(
                        "BMC {context} requires complete finite support for the set-difference left operand"
                    )))
                }
            }
            Expr::SetFilter(..) | Expr::SetBuilder(..) => Err(AYError::UnsupportedOp(format!(
                "BMC {context} cannot materialize filter/builder sets as exact arrays; use membership directly"
            ))),
            _ => Err(AYError::UnsupportedOp(format!(
                "BMC {context} cannot materialize this set expression exactly"
            ))),
        }
    }

    fn require_supported_set_element_kind(
        &self,
        expr: &Spanned<Expr>,
        context: &str,
    ) -> AYResult<()> {
        match self.set_element_sort_evidence(expr)? {
            SetElementEvidence::Empty => Ok(()),
            SetElementEvidence::Known(crate::TlaSort::Int | crate::TlaSort::String) => Ok(()),
            SetElementEvidence::Known(crate::TlaSort::Bool) => Err(AYError::UnsupportedOp(
                format!(
                    "BMC {context} does not support Set(Bool): set arrays use Int indices and no Bool-index encoding is defined"
                ),
            )),
            SetElementEvidence::Known(sort) => Err(AYError::UnsupportedOp(format!(
                "BMC {context} does not support set element sort {sort}"
            ))),
            SetElementEvidence::Unknown => Err(AYError::UnsupportedOp(format!(
                "BMC {context} requires exact set element-kind evidence"
            ))),
        }
    }

    fn ensure_compatible_set_element_kinds(
        &self,
        left: &Spanned<Expr>,
        right: &Spanned<Expr>,
        context: &str,
    ) -> AYResult<()> {
        let left = self.set_element_sort_evidence(left)?;
        let right = self.set_element_sort_evidence(right)?;
        match (left, right) {
            (SetElementEvidence::Empty, _) | (_, SetElementEvidence::Empty) => Ok(()),
            (SetElementEvidence::Known(left), SetElementEvidence::Known(right))
                if left == right =>
            {
                Ok(())
            }
            (SetElementEvidence::Known(left), SetElementEvidence::Known(right)) => {
                Err(AYError::UnsupportedOp(format!(
                    "BMC {context} cannot combine set element sorts {left} and {right} on a shared index carrier"
                )))
            }
            _ => Err(AYError::UnsupportedOp(format!(
                "BMC {context} requires exact set element-kind evidence"
            ))),
        }
    }

    /// Evaluate a closed finite Int/String set expression exactly and return
    /// its logical element sort plus carrier terms. Used by powerset expansion,
    /// where enumerating a mere support superset would introduce subsets that
    /// are not actually subsets of the base value.
    pub(super) fn concrete_set_terms(
        &mut self,
        expr: &Spanned<Expr>,
        context: &str,
        max_size: usize,
    ) -> AYResult<(crate::TlaSort, Vec<Term>)> {
        self.require_supported_set_element_kind(expr, context)?;
        let evidence = self.set_element_sort_evidence(expr)?;
        let sort = match evidence {
            SetElementEvidence::Known(sort) => sort.canonicalized(),
            SetElementEvidence::Empty => crate::TlaSort::Int,
            SetElementEvidence::Unknown => {
                return Err(AYError::UnsupportedOp(format!(
                    "BMC {context} requires a closed concrete base set"
                )))
            }
        };
        let elements = Self::evaluate_concrete_set(expr)?;
        Self::ensure_materialization_size(elements.len(), context)?;
        if elements.len() > max_size {
            return Err(AYError::UnsupportedOp(format!(
                "BMC {context} has {} elements, exceeding its limit of {max_size}",
                elements.len()
            )));
        }
        let mut terms = Vec::with_capacity(elements.len());
        for element in elements {
            let term = match element {
                ConcreteSetElement::Int(value) => self.solver.int_const(value),
                ConcreteSetElement::String(value) => {
                    let id = self.bmc_intern_string(&value);
                    self.solver.int_const(id)
                }
            };
            terms.push(term);
        }
        Ok((sort, terms))
    }

    /// Evaluate a closed concrete Int set without creating solver terms. The
    /// nested-powerset encoder is Int-specific and must decline String/Bool or
    /// symbolic bases rather than treating their uncollected support as empty.
    pub(super) fn concrete_int_set_values(
        &self,
        expr: &Spanned<Expr>,
        context: &str,
    ) -> AYResult<Vec<i64>> {
        self.require_supported_set_element_kind(expr, context)?;
        match self.set_element_sort_evidence(expr)? {
            SetElementEvidence::Known(crate::TlaSort::Int) | SetElementEvidence::Empty => {}
            SetElementEvidence::Known(sort) => {
                return Err(AYError::UnsupportedOp(format!(
                    "BMC {context} is Int-specific and cannot encode element sort {sort}"
                )))
            }
            SetElementEvidence::Unknown => {
                return Err(AYError::UnsupportedOp(format!(
                    "BMC {context} requires a closed concrete Int set"
                )))
            }
        }

        let elements = Self::evaluate_concrete_set(expr)?;
        Self::ensure_materialization_size(elements.len(), context)?;
        elements
            .into_iter()
            .map(|element| match element {
                ConcreteSetElement::Int(value) => Ok(value),
                ConcreteSetElement::String(_) => Err(AYError::UnsupportedOp(format!(
                    "BMC {context} is Int-specific and cannot encode String elements"
                ))),
            })
            .collect()
    }

    fn evaluate_concrete_set(
        expr: &Spanned<Expr>,
    ) -> AYResult<std::collections::BTreeSet<ConcreteSetElement>> {
        match &expr.node {
            Expr::SetEnum(elements) => {
                Self::ensure_materialization_size(elements.len(), "concrete set")?;
                elements
                    .iter()
                    .map(|element| match &element.node {
                        Expr::String(value) => Ok(ConcreteSetElement::String(value.clone())),
                        Expr::Int(_) | Expr::Neg(_) => Ok(ConcreteSetElement::Int(
                            Self::extract_int_literal_static(element)?,
                        )),
                        _ => Err(AYError::UnsupportedOp(
                            "BMC concrete set requires literal Int/String elements".to_string(),
                        )),
                    })
                    .collect()
            }
            Expr::Range(lo, hi) => {
                let (lo, hi) = Self::checked_range_literals(lo, hi, "concrete set")?;
                Ok(if lo <= hi {
                    (lo..=hi).map(ConcreteSetElement::Int).collect()
                } else {
                    std::collections::BTreeSet::new()
                })
            }
            Expr::Union(left, right) => {
                let mut result = Self::evaluate_concrete_set(left)?;
                result.extend(Self::evaluate_concrete_set(right)?);
                Self::ensure_materialization_size(result.len(), "concrete set union")?;
                Ok(result)
            }
            Expr::Intersect(left, right) => {
                let left = Self::evaluate_concrete_set(left)?;
                let right = Self::evaluate_concrete_set(right)?;
                Ok(left.intersection(&right).cloned().collect())
            }
            Expr::SetMinus(left, right) => {
                let left = Self::evaluate_concrete_set(left)?;
                let right = Self::evaluate_concrete_set(right)?;
                Ok(left.difference(&right).cloned().collect())
            }
            Expr::Prime(inner) => Self::evaluate_concrete_set(inner),
            Expr::Label(label) => Self::evaluate_concrete_set(&label.body),
            _ => Err(AYError::UnsupportedOp(
                "BMC powerset enumeration requires a closed concrete set expression".to_string(),
            )),
        }
    }

    /// Validate the exact finite-universe preconditions for `S \subseteq T`.
    /// A complete left support bounds every possible witness. A complete right
    /// support is also sufficient, provided the encoder additionally proves
    /// that the symbolic left array is false outside that support.
    pub(super) fn ensure_subseteq_exact(
        &self,
        left: &Spanned<Expr>,
        right: &Spanned<Expr>,
        context: &str,
    ) -> AYResult<()> {
        self.ensure_compatible_set_element_kinds(left, right, context)?;
        self.ensure_set_expr_array_exact(left, context)?;
        self.ensure_set_expr_array_exact(right, context)?;
        if self.set_has_complete_finite_support(left) || self.set_has_complete_finite_support(right)
        {
            Ok(())
        } else {
            Err(AYError::UnsupportedOp(format!(
                "BMC {context} requires complete finite support for at least one subset operand"
            )))
        }
    }

    /// Translate `S \subseteq T` by extracting a finite universe from the
    /// operands and expanding pointwise.
    pub(super) fn translate_subseteq_dispatch(
        &mut self,
        left: &Spanned<Expr>,
        right: &Spanned<Expr>,
    ) -> AYResult<Term> {
        self.translate_subseteq_bmc(left, right)
    }

    /// Translate `{e1, ..., en}` as a Bool expression.
    ///
    /// Builds the SMT array term and returns TRUE to signal success. The set
    /// is stored in the solver context; the return value is used only because
    /// the shared dispatch requires a Bool-sorted result from
    /// `translate_bool_extended`.
    pub(super) fn translate_set_enum_bool(&mut self, elements: &[Spanned<Expr>]) -> AYResult<Term> {
        let set_expr = Spanned::dummy(Expr::SetEnum(elements.to_vec()));
        self.ensure_set_expr_array_exact(&set_expr, "set enumeration")?;
        // Build: (store (store (const false) e1 true) ... en true)
        let false_val = self.solver.bool_const(false);
        let true_val = self.solver.bool_const(true);
        let mut arr = self.solver.try_const_array(Sort::Int, false_val)?;
        for elem in elements {
            let elem_term = self.translate_int(elem)?;
            arr = self.solver.try_store(arr, elem_term, true_val)?;
        }
        // The array is built and available in the solver; return TRUE.
        let _ = arr;
        Ok(self.solver.bool_const(true))
    }

    /// Translate a set binary operation (`Union`, `Intersect`, `SetMinus`) as a
    /// Bool expression.
    ///
    /// Extracts a finite universe from both operands, builds the result array
    /// via the appropriate encoder, and returns TRUE.
    pub(super) fn translate_set_binop_bool(
        &mut self,
        left: &Spanned<Expr>,
        right: &Spanned<Expr>,
        op: SetBinOp,
    ) -> AYResult<Term> {
        self.ensure_compatible_set_element_kinds(left, right, "set operation")?;
        self.ensure_set_expr_array_exact(left, "set operation")?;
        self.ensure_set_expr_array_exact(right, "set operation")?;
        let has_required_support = match op {
            SetBinOp::Union => {
                self.set_has_complete_finite_support(left)
                    && self.set_has_complete_finite_support(right)
            }
            SetBinOp::Intersect => {
                self.set_has_complete_finite_support(left)
                    || self.set_has_complete_finite_support(right)
            }
            SetBinOp::Minus => self.set_has_complete_finite_support(left),
        };
        if !has_required_support {
            let requirement = match op {
                SetBinOp::Union => "both union operands",
                SetBinOp::Intersect => "at least one intersection operand",
                SetBinOp::Minus => "the set-difference left operand",
            };
            return Err(AYError::UnsupportedOp(format!(
                "BMC set operation requires complete finite support for {requirement}"
            )));
        }
        let universe = self.extract_universe_from_exprs(&[left, right])?;
        let set_s = self.translate_set_expr(left, &universe)?;
        let set_t = self.translate_set_expr(right, &universe)?;
        let _result = match op {
            SetBinOp::Union => self.encode_union(set_s, set_t, &universe)?,
            SetBinOp::Intersect => self.encode_intersect(set_s, set_t, &universe)?,
            SetBinOp::Minus => self.encode_set_minus(set_s, set_t, &universe)?,
        };
        Ok(self.solver.bool_const(true))
    }

    /// Translate `DOMAIN f` in Bool context.
    ///
    /// DOMAIN returns a set, not a Bool. In the dispatch we return an error
    /// with guidance to use `x \in DOMAIN f` for membership tests instead.
    pub(super) fn translate_domain_bool_dispatch(
        &mut self,
        func_expr: &Spanned<Expr>,
    ) -> AYResult<Term> {
        match &func_expr.node {
            Expr::Ident(name, _) | Expr::StateVar(name, ..) => {
                if self.func_vars.contains_key(name) {
                    Err(AYError::UntranslatableExpr(
                        "DOMAIN f is a set, not a Bool expression; \
                         use `x \\in DOMAIN f` for membership tests"
                            .to_string(),
                    ))
                } else {
                    Err(AYError::UnknownVariable(format!(
                        "DOMAIN {name}: not a declared function variable"
                    )))
                }
            }
            _ => Err(AYError::UntranslatableExpr(
                "BMC DOMAIN requires a variable reference".to_string(),
            )),
        }
    }

    /// Translate `[f EXCEPT ![a] = b]` as a Bool constraint.
    ///
    /// For function variables, builds a `(store mapping idx val)` term.
    /// Returns TRUE after the EXCEPT encoding is asserted.
    pub(super) fn translate_except_bool_dispatch(
        &mut self,
        base: &Spanned<Expr>,
        specs: &[tla_core::ast::ExceptSpec],
    ) -> AYResult<Term> {
        let _mapping = self.translate_func_except_bmc(base, specs)?;
        Ok(self.solver.bool_const(true))
    }

    /// Translate `[x \in S |-> expr]` as a Bool constraint.
    ///
    /// For finite set domains, builds per-element constraints on a fresh
    /// mapping array. Returns TRUE after the constraints are asserted.
    pub(super) fn translate_func_def_bool_dispatch(
        &mut self,
        bounds: &[tla_core::ast::BoundVar],
        body: &Spanned<Expr>,
    ) -> AYResult<Term> {
        let (_domain, _mapping) = self.translate_func_construct_bmc(bounds, body)?;
        Ok(self.solver.bool_const(true))
    }

    /// Translate `Cardinality(S)` as an Int term.
    ///
    /// Extracts a finite universe from the set expression, builds the SMT
    /// array term for S, and sums `ite((select S u), 1, 0)` over the universe.
    pub(super) fn translate_cardinality_int_dispatch(
        &mut self,
        set_expr: &Spanned<Expr>,
    ) -> AYResult<Term> {
        self.ensure_set_expr_array_exact(set_expr, "Cardinality")?;
        if !self.set_has_complete_finite_support(set_expr) {
            return Err(AYError::UnsupportedOp(
                "BMC Cardinality requires complete finite support".to_string(),
            ));
        }
        let universe = self.extract_universe_from_exprs(&[set_expr])?;
        if universe.is_empty() {
            return Ok(self.solver.int_const(0));
        }

        let set_term = self.translate_set_expr(set_expr, &universe)?;

        // Sum of ite((select S u), 1, 0) for each u in universe
        let one = self.solver.int_const(1);
        let zero = self.solver.int_const(0);
        let mut sum = self.solver.int_const(0);

        for &u in &universe {
            let in_set = self.solver.try_select(set_term, u)?;
            let contrib = self.solver.try_ite(in_set, one, zero)?;
            sum = self.solver.try_add(sum, contrib)?;
        }

        Ok(sum)
    }

    /// Extract a finite universe of Int-carrier terms from set expressions.
    ///
    /// Collects concrete Int values and interned String values from set
    /// enumerations, ranges, and compound subexpressions. Element-sort
    /// preflights ensure the two logical kinds never share one extraction.
    pub(super) fn extract_universe_from_exprs(
        &mut self,
        exprs: &[&Spanned<Expr>],
    ) -> AYResult<Vec<Term>> {
        let mut values = std::collections::BTreeSet::new();
        for expr in exprs {
            Self::collect_universe_elements(expr, &mut values)?;
        }
        Self::ensure_materialization_size(values.len(), "finite universe")?;

        let mut terms = Vec::with_capacity(values.len());
        for value in values {
            let carrier = match value {
                ConcreteSetElement::Int(value) => value,
                ConcreteSetElement::String(value) => self.bmc_intern_string(&value),
            };
            terms.push(self.solver.int_const(carrier));
        }
        Ok(terms)
    }

    /// Recursively collect all integer values that could appear in a set
    /// expression.
    ///
    /// Handles `SetEnum`, `Range`, compound set operations (`Union`,
    /// `Intersect`, `SetMinus`, `Subseteq`), and `Ident` (treated as
    /// potentially unresolvable — skipped rather than errored).
    fn collect_universe_elements(
        expr: &Spanned<Expr>,
        values: &mut std::collections::BTreeSet<ConcreteSetElement>,
    ) -> AYResult<()> {
        match &expr.node {
            Expr::SetEnum(elements) => {
                for e in elements {
                    match &e.node {
                        Expr::Int(_) | Expr::Neg(_) => {
                            Self::insert_universe_element(
                                values,
                                ConcreteSetElement::Int(Self::extract_int_literal_static(e)?),
                            )?;
                        }
                        Expr::String(value) => {
                            Self::insert_universe_element(
                                values,
                                ConcreteSetElement::String(value.clone()),
                            )?;
                        }
                        _ => {}
                    }
                }
            }
            Expr::Range(lo, hi) => {
                let (lo_val, hi_val) = Self::checked_range_literals(lo, hi, "finite universe")?;
                for v in lo_val..=hi_val {
                    Self::insert_universe_element(values, ConcreteSetElement::Int(v))?;
                }
            }
            Expr::Union(left, right)
            | Expr::Intersect(left, right)
            | Expr::SetMinus(left, right)
            | Expr::Subseteq(left, right) => {
                Self::collect_universe_elements(left, values)?;
                Self::collect_universe_elements(right, values)?;
            }
            Expr::Powerset(base) => {
                Self::collect_universe_elements(base, values)?;
            }
            Expr::BigUnion(inner) => {
                if let Expr::SetEnum(components) = &inner.node {
                    for comp in components {
                        Self::collect_universe_elements(comp, values)?;
                    }
                }
            }
            Expr::Prime(inner) => {
                Self::collect_universe_elements(inner, values)?;
            }
            Expr::Label(label) => Self::collect_universe_elements(&label.body, values)?,
            // Ident could refer to a constant def but BMC does not have
            // constant_defs — skip without error.
            _ => {}
        }
        Ok(())
    }

    fn insert_universe_element(
        values: &mut std::collections::BTreeSet<ConcreteSetElement>,
        value: ConcreteSetElement,
    ) -> AYResult<()> {
        values.insert(value);
        Self::ensure_materialization_size(values.len(), "finite universe")
    }

    /// Translate `UNION {S1, S2, ...}` (BigUnion over enumerated set-of-sets)
    /// as a Bool expression.
    ///
    /// Extracts the finite universe from all component sets, builds their
    /// array representations, and produces a fresh result array containing
    /// the pointwise OR of all components.  Returns TRUE after asserting
    /// the result constraints.
    ///
    /// Part of #3778: Apalache parity — BigUnion in BMC.
    pub(super) fn translate_big_union_bool(&mut self, inner: &Spanned<Expr>) -> AYResult<Term> {
        match &inner.node {
            Expr::SetEnum(component_exprs) => {
                if component_exprs.is_empty() {
                    return Ok(self.solver.bool_const(true));
                }

                // UNION of set-valued components is materializable only when
                // every component is exact and finitely supported. Validate
                // the entire input before interning Strings or creating terms.
                let mut representative = None;
                for component in component_exprs {
                    self.ensure_set_expr_array_exact(component, "UNION")?;
                    if !self.set_has_complete_finite_support(component) {
                        return Err(AYError::UnsupportedOp(
                            "BMC UNION requires complete finite support for every component"
                                .to_string(),
                        ));
                    }
                    if !matches!(
                        self.set_element_sort_evidence(component)?,
                        SetElementEvidence::Empty
                    ) {
                        if let Some(previous) = representative {
                            self.ensure_compatible_set_element_kinds(previous, component, "UNION")?;
                        } else {
                            representative = Some(component);
                        }
                    }
                }

                // Collect universe from all component sets
                let comp_refs: Vec<&Spanned<Expr>> = component_exprs.iter().collect();
                let universe = self.extract_universe_from_exprs(&comp_refs)?;

                // Translate each component set as an array
                let mut component_terms = Vec::with_capacity(component_exprs.len());
                for comp in component_exprs {
                    let term = self.translate_set_expr(comp, &universe)?;
                    component_terms.push(term);
                }

                // Build an exact false-default result array. The previous
                // fresh array constrained only on `universe`, leaving every
                // out-of-support membership unconstrained.
                let false_val = self.solver.bool_const(false);
                let mut result = self.solver.try_const_array(Sort::Int, false_val)?;
                for &u in &universe {
                    // OR of all component memberships
                    let mut membership = false_val;
                    for comp in &component_terms {
                        let in_comp = self.solver.try_select(*comp, u)?;
                        membership = self.solver.try_or(membership, in_comp)?;
                    }
                    result = self.solver.try_store(result, u, membership)?;
                }

                let _ = result;
                Ok(self.solver.bool_const(true))
            }
            _ => Err(AYError::UnsupportedOp(
                "BMC UNION over non-enumerated set-of-sets not supported; \
                 only UNION {S1, S2, ...} is supported"
                    .to_string(),
            )),
        }
    }

    /// Translate `SUBSET S` (powerset) as a Bool expression in BMC context.
    ///
    /// For a base set S with known finite universe of n elements (n <= 16),
    /// this encodes SUBSET S as the set of all 2^n subsets. Each subset is
    /// a concrete `(Array Int Bool)` term built via store chains.
    ///
    /// In Bool context, the powerset itself is not directly useful as a
    /// predicate — it's a set-of-sets. This method returns TRUE after
    /// constructing the array terms (they can be used by quantifiers that
    /// range over SUBSET S).
    ///
    /// For larger base sets (n > 16), returns an error since 2^n subsets
    /// would be impractical.
    pub(super) fn translate_powerset_bool(&mut self, base: &Spanned<Expr>) -> AYResult<Term> {
        // Enumerating a support superset is unsound for intersection/minus:
        // it introduces subsets containing values absent from the actual base.
        // Require and evaluate a closed concrete base instead.
        let _ = self.concrete_set_terms(
            base,
            "SUBSET",
            crate::translate::powerset_encoder::MAX_POWERSET_SIZE,
        )?;
        Ok(self.solver.bool_const(true))
    }

    /// Enumerate all subsets of a base set for powerset membership in BMC.
    ///
    /// Given a base set expression S with known universe elements, generates
    /// all 2^n subset array terms. Each subset is a concrete
    /// `(store ... (const false) ... true)` chain.
    ///
    /// Used by quantifier expansion: `\E T \in SUBSET S : P(T)` becomes
    /// `P({}) \/ P({e1}) \/ P({e2}) \/ P({e1,e2}) \/ ...`
    ///
    /// Returns an error if the universe exceeds `MAX_POWERSET_SIZE` (16).
    pub(super) fn enumerate_powerset_subsets(
        &mut self,
        base: &Spanned<Expr>,
    ) -> AYResult<Vec<Term>> {
        Ok(self.enumerate_powerset_subsets_typed(base)?.1)
    }

    /// Typed variant used by quantifier expansion so temporary set bindings
    /// retain the base set's actual logical element sort.
    pub(super) fn enumerate_powerset_subsets_typed(
        &mut self,
        base: &Spanned<Expr>,
    ) -> AYResult<(crate::TlaSort, Vec<Term>)> {
        let (element_sort, universe) = self.concrete_set_terms(
            base,
            "SUBSET enumeration",
            crate::translate::powerset_encoder::MAX_POWERSET_SIZE,
        )?;
        let n = universe.len();

        let num_subsets = 1usize << n;
        let mut subsets = Vec::with_capacity(num_subsets);

        let false_val = self.solver.bool_const(false);
        let true_val = self.solver.bool_const(true);

        for mask in 0..num_subsets {
            let mut arr = self.solver.try_const_array(Sort::Int, false_val)?;
            for (i, &elem) in universe.iter().enumerate() {
                if mask & (1 << i) != 0 {
                    arr = self.solver.try_store(arr, elem, true_val)?;
                }
            }
            subsets.push(arr);
        }

        Ok((element_sort, subsets))
    }

    /// Encode a symbolic subset constraint for BMC: creates a fresh set
    /// variable constrained to be a subset of the given base set.
    ///
    /// For each element u in the universe:
    ///   `(select sub u) => (select base u)`
    ///
    /// This is used when the base set is too large for full enumeration,
    /// or when a symbolic approach is preferred.
    ///
    /// Returns the fresh symbolic subset variable.
    pub(super) fn encode_symbolic_subset(&mut self, base: &Spanned<Expr>) -> AYResult<Term> {
        self.ensure_set_expr_array_exact(base, "symbolic subset")?;
        if !self.set_has_complete_finite_support(base) {
            return Err(AYError::UnsupportedOp(
                "BMC symbolic subset encoding requires complete finite support".to_string(),
            ));
        }
        let universe = self.extract_universe_from_exprs(&[base])?;
        let base_set = self.translate_set_expr(base, &universe)?;

        // Build the symbolic subset as a store chain over the universe,
        // starting from `(const false)`. This guarantees that any index
        // outside the universe is `false` by construction — the SMT solver
        // cannot place a witness there. (Previously the subset was a free
        // `(declare-const _ (Array Int Bool))`, which left values outside
        // the universe unconstrained and allowed stray elements through.)
        let false_val = self.solver.bool_const(false);
        let mut sub = self.solver.try_const_array(Sort::Int, false_val)?;
        for &u in &universe {
            // Fresh Bool flag per universe element, constrained to imply
            // membership in the base set.
            let (_, flag) = self.declare_internal_const("symbolic subset flag", Sort::Bool);
            let in_base = self.solver.try_select(base_set, u)?;
            let implication = self.solver.try_implies(flag, in_base)?;
            self.solver
                .try_assert_term(implication)
                .expect("invariant: implies is Bool-sorted");
            sub = self.solver.try_store(sub, u, flag)?;
        }

        Ok(sub)
    }

    /// Translate `lo..hi` as a set expression (Array Int Bool).
    ///
    /// Builds a store chain: `(store ... (store (const false) lo true) ... hi true)`.
    ///
    /// Part of #3778: Apalache parity — Range-as-set in BMC.
    pub(super) fn translate_range_set_term(
        &mut self,
        lo: &Spanned<Expr>,
        hi: &Spanned<Expr>,
    ) -> AYResult<Term> {
        let (lo_val, hi_val) = Self::checked_range_literals(lo, hi, "range set")?;
        let false_val = self.solver.bool_const(false);
        let true_val = self.solver.bool_const(true);
        let mut arr = self.solver.try_const_array(Sort::Int, false_val)?;
        for v in lo_val..=hi_val {
            let elem = self.solver.int_const(v);
            arr = self.solver.try_store(arr, elem, true_val)?;
        }
        Ok(arr)
    }

    /// Try to translate function equality: `f = g` where both are function
    /// variables.
    ///
    /// For two function variables with exact finite domain metadata, returns
    /// pointwise mapping equality over the live keys. Different domains are
    /// unequal; unknown domains fail closed.
    ///
    /// Returns `None` if neither side is a recognized function variable.
    ///
    /// Part of #3778: Apalache parity — function equality in BMC.
    pub(super) fn try_translate_func_equality(
        &mut self,
        left: &Spanned<Expr>,
        right: &Spanned<Expr>,
    ) -> Option<AYResult<Term>> {
        let (left_name, left_step) = self.resolve_func_name_step(left)?;
        let (right_name, right_step) = self.resolve_func_name_step(right)?;

        // Compare only live DOMAIN cells. Full mapping-array equality observes
        // representation ghosts and can make Init/Next artificially false.
        let l_map = match self.get_func_mapping_at_step(&left_name, left_step) {
            Ok(t) => t,
            Err(e) => return Some(Err(e)),
        };
        let r_map = match self.get_func_mapping_at_step(&right_name, right_step) {
            Ok(t) => t,
            Err(e) => return Some(Err(e)),
        };

        Some(self.translate_func_logical_mapping_eq(&left_name, l_map, &right_name, r_map))
    }

    /// Resolve a function expression to `(name, step)`, or `None` if not
    /// a function variable.
    fn resolve_func_name_step(&self, expr: &Spanned<Expr>) -> Option<(String, usize)> {
        match &expr.node {
            Expr::Ident(name, _) | Expr::StateVar(name, ..) => {
                if self.func_vars.contains_key(name) {
                    Some((name.clone(), self.current_step))
                } else {
                    None
                }
            }
            Expr::Prime(inner) => match &inner.node {
                Expr::Ident(name, _) | Expr::StateVar(name, ..) => {
                    if self.func_vars.contains_key(name) {
                        Some((name.clone(), self.current_step + 1))
                    } else {
                        None
                    }
                }
                _ => None,
            },
            _ => None,
        }
    }

    /// Try to translate set equality: `S = T` where both sides are
    /// set-typed expressions (SetEnum, set variable, Powerset base, etc.).
    ///
    /// Translates each side as an `(Array Int Bool)` term and returns exact
    /// array equality after validating the TLA+ element metadata.
    ///
    /// Returns `None` if neither side is a recognizable set expression.
    ///
    /// Part of #3826: set equality for nested powerset expansion.
    pub(super) fn try_translate_set_eq(
        &mut self,
        left: &Spanned<Expr>,
        right: &Spanned<Expr>,
    ) -> Option<AYResult<Term>> {
        // Only trigger if at least one side is clearly a set expression.
        if !self.is_set_expr(left) && !self.is_set_expr(right) {
            return None;
        }

        let result = (|| -> AYResult<Term> {
            // Closed literals, including sets of sets produced by nested
            // powerset expansion, can be compared exactly without choosing an
            // SMT array index carrier.  This also handles polymorphic empty
            // sets and avoids forcing compound set elements through the scalar
            // Int/String carrier.
            if let (Some(left_value), Some(right_value)) = (
                Self::closed_set_literal_value(left),
                Self::closed_set_literal_value(right),
            ) {
                return Ok(self.solver.bool_const(left_value == right_value));
            }

            let left_element_sort = self.set_element_sort_evidence(left)?;
            let right_element_sort = self.set_element_sort_evidence(right)?;
            match (&left_element_sort, &right_element_sort) {
                (SetElementEvidence::Known(left_sort), SetElementEvidence::Known(right_sort))
                    if left_sort != right_sort =>
                {
                    // A blanket FALSE would be wrong when either operand can
                    // be empty.  Decline instead of comparing the shared Int
                    // index carriers (where String ids can alias Int values).
                    return Err(AYError::UnsupportedOp(format!(
                        "BMC cannot compare sets with differing element sorts \
                         {left_sort} and {right_sort}; empty sets can cross this \
                         metadata boundary"
                    )));
                }
                (SetElementEvidence::Unknown, SetElementEvidence::Empty)
                | (SetElementEvidence::Empty, SetElementEvidence::Unknown) => {
                    // Exact empty-set equality is independent of the other
                    // operand's element metadata and array equality below is
                    // representation-exact.
                }
                (SetElementEvidence::Unknown, _) | (_, SetElementEvidence::Unknown) => {
                    return Err(AYError::UnsupportedOp(
                        "BMC cannot compare nonempty/symbolic sets without exact element-kind evidence"
                            .to_string(),
                    ));
                }
                _ => {}
            }

            // Validate exact array materialization before universe extraction;
            // in particular this rejects Set(Bool), filters/builders, and
            // compound arrays without a complete bounding support.
            self.ensure_set_expr_array_exact(left, "set equality")?;
            self.ensure_set_expr_array_exact(right, "set equality")?;

            let universe = self.extract_universe_from_exprs(&[left, right])?;
            let left_arr = self.translate_set_expr(left, &universe)?;
            let right_arr = self.translate_set_expr(right, &universe)?;
            // Set variables are full arrays.  Comparing only the finite
            // literal-derived universe makes `S = {}` a tautology when that
            // universe is empty and ignores arbitrary members outside a
            // nonempty literal universe.  Array equality is the exact set
            // equality for this representation.
            Ok(self.solver.try_eq(left_arr, right_arr)?)
        })();

        Some(result)
    }

    fn closed_set_literal_value(expr: &Spanned<Expr>) -> Option<Vec<ClosedSetElement>> {
        let elements = match &expr.node {
            Expr::SetEnum(elements) => elements,
            Expr::Prime(inner) | Expr::SubstIn(_, inner) => {
                return Self::closed_set_literal_value(inner);
            }
            Expr::Label(label) => return Self::closed_set_literal_value(&label.body),
            _ => return None,
        };

        let mut values = elements
            .iter()
            .map(Self::closed_set_element_value)
            .collect::<Option<Vec<_>>>()?;
        values.sort();
        values.dedup();
        Some(values)
    }

    fn closed_set_element_value(expr: &Spanned<Expr>) -> Option<ClosedSetElement> {
        match &expr.node {
            Expr::Bool(value) => Some(ClosedSetElement::Bool(*value)),
            Expr::Int(value) => Some(ClosedSetElement::Int(value.clone())),
            Expr::String(value) => Some(ClosedSetElement::String(value.clone())),
            Expr::Neg(inner) => match &inner.node {
                Expr::Int(value) => Some(ClosedSetElement::Int(-value.clone())),
                _ => None,
            },
            Expr::SetEnum(_) | Expr::Prime(_) | Expr::SubstIn(..) | Expr::Label(_) => {
                Self::closed_set_literal_value(expr).map(ClosedSetElement::Set)
            }
            _ => None,
        }
    }

    /// Check if an expression is a set-typed expression (heuristic).
    ///
    /// Returns true for SetEnum, Powerset, Union, Intersect, SetMinus, Range,
    /// and identifiers known to be set variables.
    fn is_set_expr(&self, expr: &Spanned<Expr>) -> bool {
        match &expr.node {
            Expr::SetEnum(_)
            | Expr::Powerset(_)
            | Expr::Union(_, _)
            | Expr::Intersect(_, _)
            | Expr::SetMinus(_, _)
            | Expr::Range(_, _)
            | Expr::SetFilter(_, _)
            | Expr::SetBuilder(_, _) => true,
            Expr::Ident(name, _) | Expr::StateVar(name, ..) => {
                self.vars.get(name).map_or(false, |info| {
                    matches!(info.sort, crate::TlaSort::Set { .. })
                })
            }
            Expr::Prime(inner) | Expr::SubstIn(_, inner) => self.is_set_expr(inner),
            Expr::Label(label) => self.is_set_expr(&label.body),
            _ => false,
        }
    }

    /// Recover homogeneous set-element metadata without translating an
    /// element into the shared SMT Int namespace. Polymorphic emptiness is kept
    /// distinct from missing evidence so only the former may safely cross an
    /// element-metadata boundary.
    fn set_element_sort_evidence(&self, expr: &Spanned<Expr>) -> AYResult<SetElementEvidence> {
        use crate::TlaSort;

        let merge = |left: SetElementEvidence, right: SetElementEvidence| match (left, right) {
            (SetElementEvidence::Empty, evidence) | (evidence, SetElementEvidence::Empty) => {
                Ok(evidence)
            }
            (SetElementEvidence::Known(left), SetElementEvidence::Known(right))
                if left == right =>
            {
                Ok(SetElementEvidence::Known(left))
            }
            (SetElementEvidence::Known(left), SetElementEvidence::Known(right)) => {
                Err(AYError::UnsupportedOp(format!(
                    "BMC set expression mixes element sorts {left} and {right}"
                )))
            }
            _ => Ok(SetElementEvidence::Unknown),
        };

        match &expr.node {
            Expr::SetEnum(elements) => {
                let mut result = SetElementEvidence::Empty;
                for element in elements {
                    let element_evidence = match self.scalar_expr_sort(element) {
                        Some(sort) => SetElementEvidence::Known(sort.canonicalized()),
                        None => SetElementEvidence::Unknown,
                    };
                    result = merge(result, element_evidence)?;
                }
                Ok(result)
            }
            Expr::Range(..) => Ok(SetElementEvidence::Known(TlaSort::Int)),
            Expr::Ident(name, _) | Expr::StateVar(name, ..) => match self.vars.get(name) {
                Some(info) => match &info.sort {
                    TlaSort::Set { element_sort } => Ok(SetElementEvidence::Known(
                        (**element_sort).clone().canonicalized(),
                    )),
                    _ => Ok(SetElementEvidence::Unknown),
                },
                None => Ok(SetElementEvidence::Unknown),
            },
            Expr::Prime(inner) | Expr::SubstIn(_, inner) => self.set_element_sort_evidence(inner),
            Expr::Label(label) => self.set_element_sort_evidence(&label.body),
            Expr::Union(left, right)
            | Expr::Intersect(left, right)
            | Expr::SetMinus(left, right) => merge(
                self.set_element_sort_evidence(left)?,
                self.set_element_sort_evidence(right)?,
            ),
            Expr::SetFilter(bound, _) => match &bound.domain {
                Some(domain) => self.set_element_sort_evidence(domain),
                None => Ok(SetElementEvidence::Unknown),
            },
            _ => Ok(SetElementEvidence::Unknown),
        }
    }

    fn ensure_materialization_size(size: usize, context: &str) -> AYResult<()> {
        if size > MAX_BMC_SET_MATERIALIZATION {
            Err(AYError::UnsupportedOp(format!(
                "BMC {context} finite support has {size} elements, exceeding the materialization limit of {MAX_BMC_SET_MATERIALIZATION}"
            )))
        } else {
            Ok(())
        }
    }

    /// Resolve and size-check a literal range without iterating it. The i128
    /// subtraction is intentional: `i64::MIN..i64::MAX` must be rejected, not
    /// overflow while computing its cardinality.
    pub(super) fn checked_range_literals(
        lo: &Spanned<Expr>,
        hi: &Spanned<Expr>,
        context: &str,
    ) -> AYResult<(i64, i64)> {
        let lo = Self::extract_int_literal_static(lo)?;
        let hi = Self::extract_int_literal_static(hi)?;
        let size = if lo <= hi {
            i128::from(hi) - i128::from(lo) + 1
        } else {
            0
        };
        if size > MAX_BMC_SET_MATERIALIZATION as i128 {
            return Err(AYError::UnsupportedOp(format!(
                "BMC {context} finite range has {size} elements, exceeding the materialization limit of {MAX_BMC_SET_MATERIALIZATION}"
            )));
        }
        Ok((lo, hi))
    }

    /// Extract an integer literal from an expression (static, no solver access).
    fn extract_int_literal_static(expr: &Spanned<Expr>) -> AYResult<i64> {
        fn big_int_literal(expr: &Spanned<Expr>) -> Option<num_bigint::BigInt> {
            match &expr.node {
                Expr::Int(value) => Some(value.clone()),
                Expr::Neg(inner) => Some(-big_int_literal(inner)?),
                _ => None,
            }
        }

        let value = big_int_literal(expr).ok_or_else(|| {
            AYError::UnsupportedOp(format!(
                "expected integer literal in universe extraction, got {:?}",
                std::mem::discriminant(&expr.node)
            ))
        })?;
        value
            .try_into()
            .map_err(|_| AYError::IntegerOverflow("integer literal too large for i64".to_string()))
    }
}
