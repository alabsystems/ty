// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! BMC-specific translation methods for [`BmcTranslator`].
//!
//! Extracts `translate_init`, `translate_next`, `translate_not_safety_all_steps`,
//! `extract_trace`, and internal translation helpers (div/mod linearization,
//! UNCHANGED, membership) from the parent module.

use std::collections::HashMap;

use ay_dpll::api::{Model, ModelValue, SolveResult, Sort, Term};
use num_bigint::BigInt;
use tla_core::ast::{BoundVar, Expr};
use tla_core::name_intern::NameId;
use tla_core::{
    dispatch_translate_bool, dispatch_translate_int, ExprFold, SpanPolicy, Spanned, SubstituteExpr,
};

use crate::error::{AYError, AYResult};
use crate::translate::nested_powerset::BaseElement;

use super::{BmcState, BmcTranslator, BmcValue, TlaSort};

impl BmcTranslator {
    /// Translate Init predicate for step 0
    ///
    /// Variables are interpreted as x__0.
    pub fn translate_init(&mut self, init: &Spanned<Expr>) -> AYResult<Term> {
        self.current_step = 0;
        self.translate_bool(init)
    }

    /// Translate Next predicate for step i -> i+1
    ///
    /// Unprimed variables are x__i, primed variables are x__i+1.
    pub fn translate_next(&mut self, next: &Spanned<Expr>, step: usize) -> AYResult<Term> {
        if step + 1 > self.bound_k {
            return Err(AYError::UntranslatableExpr(format!(
                "Next at step {} would exceed bound {}",
                step, self.bound_k
            )));
        }
        self.current_step = step;
        self.translate_bool(next)
    }

    /// Translate safety property negation for ALL steps 0..=k
    ///
    /// Returns: ¬Safety(s0) ∨ ¬Safety(s1) ∨ ... ∨ ¬Safety(sk)
    pub fn translate_not_safety_all_steps(
        &mut self,
        safety: &Spanned<Expr>,
        k: usize,
    ) -> AYResult<Term> {
        let actual_k = k.min(self.bound_k);
        let mut disjuncts = Vec::with_capacity(actual_k + 1);

        for step in 0..=actual_k {
            self.current_step = step;
            let safety_at_step = self.translate_bool(safety)?;
            let not_safety = self.solver.try_not(safety_at_step)?;
            disjuncts.push(not_safety);
        }

        // Build disjunction
        if disjuncts.is_empty() {
            Ok(self.solver.bool_const(false))
        } else if disjuncts.len() == 1 {
            Ok(disjuncts.into_iter().next().expect("len checked == 1"))
        } else {
            let mut result = disjuncts.pop().expect("len checked > 1");
            for term in disjuncts.into_iter().rev() {
                result = self.solver.try_or(term, result)?;
            }
            Ok(result)
        }
    }

    /// Translate safety property at a specific step.
    ///
    /// Returns: Safety(s_step)
    ///
    /// Used by k-induction to assert the induction hypothesis (safety holds
    /// at steps 0..k-1). Part of #3722.
    pub fn translate_safety_at_step(
        &mut self,
        safety: &Spanned<Expr>,
        step: usize,
    ) -> AYResult<Term> {
        self.current_step = step;
        self.translate_bool(safety)
    }

    /// Translate negated safety property at a specific step.
    ///
    /// Returns: ¬Safety(s_step)
    ///
    /// Used by k-induction to assert the induction check (safety violated
    /// at step k). Part of #3722.
    pub fn translate_not_safety_at_step(
        &mut self,
        safety: &Spanned<Expr>,
        step: usize,
    ) -> AYResult<Term> {
        self.current_step = step;
        let safety_term = self.translate_bool(safety)?;
        Ok(self.solver.try_not(safety_term)?)
    }

    /// Extract counterexample trace from SAT model
    ///
    /// Returns the states from step 0 to k (all steps in the bound).
    pub fn extract_trace(&self, model: &Model) -> Vec<BmcState> {
        let mut trace = Vec::with_capacity(self.bound_k + 1);

        for step in 0..=self.bound_k {
            let mut assignments = HashMap::new();

            for (name, info) in &self.vars {
                let step_name = format!("{name}__{step}");
                match &info.sort {
                    TlaSort::Bool => {
                        if let Some(val) = model.bool_val(&step_name) {
                            assignments.insert(name.clone(), BmcValue::Bool(val));
                        } else {
                            eprintln!(
                                "Warning: BMC extract_trace: Bool variable '{name}' \
                                 not found in model at step {step}"
                            );
                        }
                    }
                    TlaSort::Int | TlaSort::String => {
                        if let Some(val) = model.int_val(&step_name) {
                            use std::convert::TryFrom;
                            if let Ok(v) = i64::try_from(val) {
                                assignments.insert(name.clone(), BmcValue::Int(v));
                            } else {
                                // Part of #3888: preserve big integers instead of dropping them
                                assignments.insert(name.clone(), BmcValue::BigInt(val.clone()));
                            }
                        } else {
                            eprintln!(
                                "Warning: BMC extract_trace: Int variable '{name}' \
                                 not found in model at step {step}"
                            );
                        }
                    }
                    TlaSort::Set { .. } => {
                        // Sets are encoded as (Array Int Bool). Extract domain keys
                        // where membership is true.
                        let domain_keys = Self::extract_array_domain_keys(model, &step_name);
                        let members: Vec<BmcValue> =
                            domain_keys.into_iter().map(BmcValue::Int).collect();
                        assignments.insert(name.clone(), BmcValue::Set(members));
                    }
                    TlaSort::Sequence { .. } => {
                        // Sequences are delegated to seq_vars; handled below.
                        // If var is in seq_vars, it will be extracted there.
                        // Otherwise it might be tracked only through vars map.
                    }
                    TlaSort::Function { .. } | TlaSort::FunctionSym { .. } => {
                        // Function sort in vars is delegated to func_vars;
                        // handled below in the func_vars loop. Part of #3786.
                        // Symbolic-domain functions are map-only (no domain
                        // array) but still live in func_vars.
                    }
                    TlaSort::Tuple { .. } => {
                        // Tuples are delegated to tuple_vars; handled below.
                    }
                    TlaSort::Record { .. } => {
                        // Records are delegated to record_vars; handled below.
                    }
                }
            }

            // Extract function variable values from func_vars.
            // Part of #3786: Function encoding in BMC translator.
            for (name, info) in &self.func_vars {
                let dom_name = format!("{name}__dom__{step}");
                let map_name = format!("{name}__map__{step}");

                let domain_keys = Self::extract_array_domain_keys(model, &dom_name);

                if domain_keys.is_empty() {
                    let _ = &info.range_sort;
                    eprintln!(
                        "Warning: BMC extract_trace: function variable '{name}' \
                         domain could not be extracted from model at step {step}"
                    );
                } else {
                    let mut entries = Vec::new();
                    for key in &domain_keys {
                        let val = Self::extract_array_int_at(model, &map_name, *key);
                        let bmc_val = match val {
                            Some(n) => {
                                use std::convert::TryFrom;
                                if let Ok(v) = i64::try_from(&n) {
                                    BmcValue::Int(v)
                                } else {
                                    BmcValue::BigInt(n)
                                }
                            }
                            None => BmcValue::Int(0),
                        };
                        entries.push((*key, bmc_val));
                    }
                    entries.sort_by_key(|(k, _)| *k);
                    assignments.insert(name.clone(), BmcValue::Function(entries));
                }
            }

            // Extract sequence variable values from seq_vars.
            for (name, info) in &self.seq_vars {
                let arr_name = format!("{name}__arr__{step}");
                let len_name = format!("{name}__len__{step}");

                let len = model.int_val_i64(&len_name).unwrap_or(0);
                let mut elements = Vec::with_capacity(len.max(0) as usize);
                for i in 1..=len {
                    let val = Self::extract_array_int_at(model, &arr_name, i);
                    let bmc_val = match val {
                        Some(n) => {
                            use std::convert::TryFrom;
                            if let Ok(v) = i64::try_from(&n) {
                                BmcValue::Int(v)
                            } else {
                                BmcValue::BigInt(n)
                            }
                        }
                        None => BmcValue::Int(0),
                    };
                    elements.push(bmc_val);
                }
                let _ = &info.element_sort;
                assignments.insert(name.clone(), BmcValue::Sequence(elements));
            }

            // Extract record variable values from record_vars.
            // Part of #3787: Record encoding — per-field SMT variables.
            for (name, info) in &self.record_vars {
                let mut fields = Vec::with_capacity(info.field_sorts.len());
                for (field_name, _sort) in &info.field_sorts {
                    let field_var_name = format!("{name}__f_{field_name}__{step}");
                    let bmc_val = Self::extract_scalar_from_model(model, &field_var_name);
                    fields.push((field_name.clone(), bmc_val));
                }
                fields.sort_by(|(a, _), (b, _)| a.cmp(b));
                assignments.insert(name.clone(), BmcValue::Record(fields));
            }

            // Extract tuple variable values from tuple_vars.
            // Part of #3787: Tuple encoding — per-element SMT variables.
            for (name, info) in &self.tuple_vars {
                let mut elements = Vec::with_capacity(info.element_sorts.len());
                for (i, _sort) in info.element_sorts.iter().enumerate() {
                    let elem_var_name = format!("{name}__e_{}__{step}", i + 1);
                    let bmc_val = Self::extract_scalar_from_model(model, &elem_var_name);
                    elements.push(bmc_val);
                }
                assignments.insert(name.clone(), BmcValue::Tuple(elements));
            }

            trace.push(BmcState { step, assignments });
        }

        // Decode interned string ids back into the original string literals.
        // BMC stores `TlaSort::String` terms as interned (negative) integers
        // (see `bmc_intern_string`); the model assigns those raw ids. Replaying
        // a CEX through the interpreter requires the genuine string value, so we
        // map every `Int(id)` matching a known interned id back to its string.
        // Sound: interned ids are far-negative and disjoint from ordinary
        // integer data, so no real integer value is ever misdecoded.
        if !self.string_intern.is_empty() {
            let reverse: HashMap<i64, &str> = self
                .string_intern
                .iter()
                .map(|(s, &id)| (id, s.as_str()))
                .collect();
            for state in &mut trace {
                for value in state.assignments.values_mut() {
                    Self::decode_interned_strings(value, &reverse);
                }
            }
        }

        trace
    }

    /// Recursively rewrite `BmcValue::Int(id)` into `BmcValue::String(s)` for
    /// every interned-string id present in `reverse`.
    fn decode_interned_strings(value: &mut BmcValue, reverse: &HashMap<i64, &str>) {
        match value {
            BmcValue::Int(n) => {
                if let Some(s) = reverse.get(n) {
                    *value = BmcValue::String((*s).to_string());
                }
            }
            BmcValue::Set(elems) | BmcValue::Sequence(elems) | BmcValue::Tuple(elems) => {
                for e in elems {
                    Self::decode_interned_strings(e, reverse);
                }
            }
            BmcValue::Record(fields) => {
                for (_, v) in fields {
                    Self::decode_interned_strings(v, reverse);
                }
            }
            BmcValue::Function(entries) => {
                for (_, v) in entries {
                    Self::decode_interned_strings(v, reverse);
                }
            }
            BmcValue::Bool(_) | BmcValue::BigInt(_) | BmcValue::String(_) => {}
        }
    }

    /// Extract a scalar BmcValue from the model by variable name.
    ///
    /// Checks int first, then bool. Returns `BmcValue::Int(0)` as default.
    fn extract_scalar_from_model(model: &Model, name: &str) -> BmcValue {
        if let Some(val) = model.int_val(name) {
            use std::convert::TryFrom;
            if let Ok(v) = i64::try_from(val) {
                BmcValue::Int(v)
            } else {
                BmcValue::BigInt(val.clone())
            }
        } else if let Some(b) = model.bool_val(name) {
            BmcValue::Bool(b)
        } else {
            BmcValue::Int(0)
        }
    }

    /// Extract domain keys from an `(Array Int Bool)` model value.
    ///
    /// Reads the structured `ModelValue::Array { default, stores }` from the
    /// solver model. For a set encoded as `(store ... (const false) k1 true ...)`,
    /// the stores with `true` values are the domain keys.
    ///
    /// Part of #3786.
    fn extract_array_domain_keys(model: &Model, name: &str) -> Vec<i64> {
        let Some(array_val) = model.array_val(name) else {
            return Vec::new();
        };
        match array_val {
            ModelValue::Array { default: _, stores } => {
                let mut keys = Vec::new();
                for (index, value) in stores {
                    // Only include keys where the value is `true` (set membership)
                    let is_member = match value {
                        ModelValue::Bool(true) => true,
                        ModelValue::Int(n) => *n != BigInt::from(0),
                        _ => false,
                    };
                    if is_member {
                        if let ModelValue::Int(k) = index {
                            use num_traits::ToPrimitive;
                            if let Some(key) = k.to_i64() {
                                keys.push(key);
                            }
                        }
                    }
                }
                keys.sort();
                keys
            }
            _ => Vec::new(),
        }
    }

    /// Extract an integer value at a specific index from an `(Array Int Int)` model.
    ///
    /// Reads the structured `ModelValue::Array { default, stores }` and looks
    /// for a store at the given key index. If no explicit store exists, returns
    /// the default value.
    ///
    /// Part of #3786.
    fn extract_array_int_at(model: &Model, name: &str, key: i64) -> Option<num_bigint::BigInt> {
        let array_val = model.array_val(name)?;
        match array_val {
            ModelValue::Array { default, stores } => {
                // Check for an explicit store at the key
                let key_bi = BigInt::from(key);
                for (index, value) in stores {
                    if let ModelValue::Int(idx) = index {
                        if *idx == key_bi {
                            if let ModelValue::Int(v) = value {
                                return Some(v.clone());
                            }
                        }
                    }
                }
                // Fall back to default value
                if let ModelValue::Int(d) = default.as_ref() {
                    Some(d.clone())
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    // === Translation methods ===

    /// Translate a boolean expression at the current step via shared dispatch
    pub(super) fn translate_bool(&mut self, expr: &Spanned<Expr>) -> AYResult<Term> {
        dispatch_translate_bool(self, expr)
    }

    /// Translate an integer expression at the current step via shared dispatch
    pub(super) fn translate_int(&mut self, expr: &Spanned<Expr>) -> AYResult<Term> {
        dispatch_translate_int(self, expr)
    }

    /// BMC-specific IntDiv translation with QF_LIA linearization
    pub(super) fn translate_int_div_bmc(
        &mut self,
        left: &Spanned<Expr>,
        right: &Spanned<Expr>,
    ) -> AYResult<Term> {
        match &right.node {
            Expr::Int(r) => {
                let zero = num_bigint::BigInt::from(0);
                if *r == zero {
                    return Err(AYError::UnsupportedOp(
                        "BMC cannot translate division by zero".to_string(),
                    ));
                }
                if *r < zero {
                    return Err(AYError::UnsupportedOp(
                        "BMC cannot translate division by negative constant (use positive divisor)"
                            .to_string(),
                    ));
                }

                let k: i64 = i64::try_from(r)
                    .map_err(|_| AYError::IntegerOverflow(format!("divisor {r} too large")))?;

                // Constant-constant case: compute at translation time
                if let Expr::Int(l) = &left.node {
                    let mut q = l / r;
                    if ((l < &zero) != (r < &zero)) && (&q * r != *l) {
                        q -= 1;
                    }
                    let q_i64: i64 = i64::try_from(&q)
                        .map_err(|_| AYError::IntegerOverflow(format!("integer {q} too large")))?;
                    return Ok(self.solver.int_const(q_i64));
                }

                // Variable-constant case: use QF_LIA linearization
                let x_term = self.translate_int(left)?;
                let (q, _r) = self.linearize_div_mod(x_term, k)?;
                Ok(q)
            }
            _ => Err(AYError::UntranslatableExpr(
                "BMC cannot translate IntDiv with non-constant divisor".to_string(),
            )),
        }
    }

    /// BMC-specific Mod translation with QF_LIA linearization
    pub(super) fn translate_mod_bmc(
        &mut self,
        left: &Spanned<Expr>,
        right: &Spanned<Expr>,
    ) -> AYResult<Term> {
        match &right.node {
            Expr::Int(r) => {
                let zero = num_bigint::BigInt::from(0);
                if *r <= zero {
                    return Err(AYError::UnsupportedOp(
                        "BMC cannot translate modulo with non-positive divisor (TLC requires divisor > 0)".to_string(),
                    ));
                }

                let k: i64 = i64::try_from(r)
                    .map_err(|_| AYError::IntegerOverflow(format!("divisor {r} too large")))?;

                // Constant-constant case: compute at translation time
                if let Expr::Int(l) = &left.node {
                    let mut m = l % r;
                    if m < zero {
                        m += r;
                    }
                    let m_i64: i64 = i64::try_from(&m)
                        .map_err(|_| AYError::IntegerOverflow(format!("integer {m} too large")))?;
                    return Ok(self.solver.int_const(m_i64));
                }

                // Variable-constant case: use QF_LIA linearization
                let x_term = self.translate_int(left)?;
                let (_q, r) = self.linearize_div_mod(x_term, k)?;
                Ok(r)
            }
            _ => Err(AYError::UntranslatableExpr(
                "BMC cannot translate Mod with non-constant divisor".to_string(),
            )),
        }
    }

    /// Translate UNCHANGED expression
    /// UNCHANGED can take a single var or a tuple <<x, y, z>>
    pub(super) fn translate_unchanged_expr(&mut self, expr: &Spanned<Expr>) -> AYResult<Term> {
        match &expr.node {
            Expr::Ident(name, _) | Expr::StateVar(name, ..) => {
                // Check for sequence variable first
                if self.seq_vars.contains_key(name) {
                    return self.translate_unchanged_seq(name);
                }
                // Check for function variable
                if self.func_vars.contains_key(name) {
                    return self.translate_unchanged_func(name);
                }
                // Check for record variable (Part of #3787)
                if self.record_vars.contains_key(name) {
                    return self.translate_unchanged_record(name);
                }
                // Check for tuple variable (Part of #3787)
                if self.tuple_vars.contains_key(name) {
                    return self.translate_unchanged_tuple(name);
                }
                // Single scalar/set variable: x' = x
                let current = self.get_var_at_step(name, self.current_step)?;
                let next = self.get_var_at_step(name, self.current_step + 1)?;
                Ok(self.solver.try_eq(next, current)?)
            }
            Expr::Tuple(elems) => {
                // Tuple of variables: <<x, y, z>> -> x' = x /\ y' = y /\ z' = z
                if elems.is_empty() {
                    return Ok(self.solver.bool_const(true));
                }

                let mut constraints = Vec::with_capacity(elems.len());
                for elem in elems {
                    let c = self.translate_unchanged_expr(elem)?;
                    constraints.push(c);
                }

                if constraints.len() == 1 {
                    Ok(constraints.into_iter().next().expect("len checked == 1"))
                } else {
                    let mut result = constraints.pop().expect("len checked > 1");
                    for c in constraints.into_iter().rev() {
                        result = self.solver.try_and(c, result)?;
                    }
                    Ok(result)
                }
            }
            _ => Err(AYError::UntranslatableExpr(format!(
                "UNCHANGED requires variable or tuple, got: {:?}",
                std::mem::discriminant(&expr.node)
            ))),
        }
    }

    /// Translate set membership (x \in S)
    pub(super) fn translate_membership(
        &mut self,
        elem: &Spanned<Expr>,
        set: &Spanned<Expr>,
    ) -> AYResult<Term> {
        match &set.node {
            // x \in SUBSET S -> x \subseteq S
            // For set-typed element x and base set S, membership in SUBSET S
            // means x is a subset of S. We extract the universe from S,
            // translate both x and S as set expressions, then assert the
            // pointwise subset constraint.
            Expr::Powerset(base) => {
                let universe = self.extract_universe_from_exprs(&[elem, base])?;
                let elem_set = self.translate_set_expr(elem, &universe)?;
                let base_set = self.translate_set_expr(base, &universe)?;

                if universe.is_empty() {
                    return Ok(self.solver.bool_const(true));
                }

                // x \subseteq S: \A u \in universe : (select x u) => (select S u)
                let mut conjuncts = Vec::with_capacity(universe.len());
                for &u in &universe {
                    let in_x = self.solver.try_select(elem_set, u)?;
                    let in_s = self.solver.try_select(base_set, u)?;
                    let implication = self.solver.try_implies(in_x, in_s)?;
                    conjuncts.push(implication);
                }

                let mut result = conjuncts[0];
                for &c in &conjuncts[1..] {
                    result = self.solver.try_and(result, c)?;
                }
                Ok(result)
            }

            // x \in {a, b, c} -> x = a \/ x = b \/ x = c
            Expr::SetEnum(elements) => {
                if elements.is_empty() {
                    return Ok(self.solver.bool_const(false));
                }

                let mut disjuncts = Vec::with_capacity(elements.len());
                for e in elements {
                    let eq = self.translate_bool(&Spanned::dummy(Expr::Eq(
                        Box::new(elem.clone()),
                        Box::new(e.clone()),
                    )))?;
                    disjuncts.push(eq);
                }

                if disjuncts.len() == 1 {
                    Ok(disjuncts.into_iter().next().expect("len checked == 1"))
                } else {
                    let mut result = disjuncts.pop().expect("len checked > 1");
                    for d in disjuncts.into_iter().rev() {
                        result = self.solver.try_or(d, result)?;
                    }
                    Ok(result)
                }
            }

            // x \in lo..hi -> lo <= x /\ x <= hi
            Expr::Range(lo, hi) => {
                let x = self.translate_int(elem)?;
                let lo_val = self.translate_int(lo)?;
                let hi_val = self.translate_int(hi)?;
                let ge_lo = self.solver.try_ge(x, lo_val)?;
                let le_hi = self.solver.try_le(x, hi_val)?;
                Ok(self.solver.try_and(ge_lo, le_hi)?)
            }

            // x \in BOOLEAN -> x = TRUE \/ x = FALSE (trivially true for Bool sort)
            Expr::Ident(name, _) if name == "BOOLEAN" => {
                // For a boolean variable, this is always true
                Ok(self.solver.bool_const(true))
            }

            // x \in Int -> (= x x): a tautology that still references x so the
            // model assigns it. Mirrors the SAT/DPLL arm
            // (translate/membership/mod.rs) and the CHC arm — the BMC lane was
            // the only path missing it. POLARITY-EXACT: `x ∈ Int` is an
            // equivalence for an Int-sorted x, so the encoding is sound in both
            // the J and the negated (`¬J` / consecution `¬J'`) positions.
            // TypeMismatch ⇒ `false` is FAIL-CLOSED: a genuinely non-Int x is
            // indeed not in Int, and if a sort were mis-inferred, J collapsing
            // to `false` makes initiation (`Init ∧ ¬J`) SAT — an honest decline,
            // never a false certificate.
            Expr::Ident(name, _) if name == "Int" => match self.translate_int(elem) {
                Ok(i) => Ok(self.solver.try_eq(i, i)?),
                Err(AYError::TypeMismatch { .. }) => Ok(self.solver.bool_const(false)),
                Err(e) => Err(e),
            },

            // x \in Nat -> x >= 0. Same equivalence/polarity argument as the
            // Int arm: `¬(x ∈ Nat)` becomes exactly `x < 0`. The arm fires ONLY
            // when the element is Int-sorted (translate_int succeeds); anything
            // else declines through the same fail-closed TypeMismatch route.
            Expr::Ident(name, _) if name == "Nat" => match self.translate_int(elem) {
                Ok(i) => {
                    let zero = self.solver.int_const(0);
                    Ok(self.solver.try_ge(i, zero)?)
                }
                Err(AYError::TypeMismatch { .. }) => Ok(self.solver.bool_const(false)),
                Err(e) => Err(e),
            },

            // x \in S where S is a set-typed variable -> (select S x)
            Expr::Ident(name, _) | Expr::StateVar(name, ..)
                if self
                    .vars
                    .get(name)
                    .map_or(false, |info| matches!(info.sort, TlaSort::Set { .. })) =>
            {
                let set_term = self.get_var_at_step(name, self.current_step)?;
                let elem_term = self.translate_int(elem)?;
                Ok(self.solver.try_select(set_term, elem_term)?)
            }

            // x \in DOMAIN f -> (select f__dom x) (Part of #3786), OR the
            // arithmetic bound `lo <= x <= N+offset` for a symbolic-domain
            // (map-only) function (function-state all-N encoding).
            Expr::Domain(func) => {
                if let Some((lo, hi_const, hi_offset)) =
                    Self::func_expr_base_name(func).and_then(|n| self.func_symbolic_domain(&n))
                {
                    let elem_term = self.translate_int(elem)?;
                    return self.symbolic_func_domain_bound(lo, &hi_const, hi_offset, elem_term);
                }
                let dom_term = self.translate_func_domain_bmc(func)?;
                let elem_term = self.translate_int(elem)?;
                Ok(self.solver.try_select(dom_term, elem_term)?)
            }

            // x \in S' where S is a set variable -> (select S__next x) (Part of #3806)
            Expr::Prime(inner) => match &inner.node {
                Expr::Ident(name, _) | Expr::StateVar(name, ..)
                    if self
                        .vars
                        .get(name)
                        .map_or(false, |info| matches!(info.sort, TlaSort::Set { .. })) =>
                {
                    let set_term = self.get_var_at_step(name, self.current_step + 1)?;
                    let elem_term = self.translate_int(elem)?;
                    Ok(self.solver.try_select(set_term, elem_term)?)
                }
                _ => Err(AYError::UntranslatableExpr(format!(
                    "BMC cannot translate membership in primed: {:?}",
                    std::mem::discriminant(&inner.node)
                ))),
            },

            // x \in (S \cup T) -> (x \in S) \/ (x \in T). Part of #3806.
            Expr::Union(left, right) => {
                let in_left = self.translate_membership(elem, left)?;
                let in_right = self.translate_membership(elem, right)?;
                Ok(self.solver.try_or(in_left, in_right)?)
            }

            // x \in (S \cap T) -> (x \in S) /\ (x \in T). Part of #3806.
            Expr::Intersect(left, right) => {
                let in_left = self.translate_membership(elem, left)?;
                let in_right = self.translate_membership(elem, right)?;
                Ok(self.solver.try_and(in_left, in_right)?)
            }

            // x \in (S \ T) -> (x \in S) /\ ~(x \in T). Part of #3806.
            Expr::SetMinus(left, right) => {
                let in_left = self.translate_membership(elem, left)?;
                let in_right = self.translate_membership(elem, right)?;
                let not_in_right = self.solver.try_not(in_right)?;
                Ok(self.solver.try_and(in_left, not_in_right)?)
            }

            // x \in {y \in S : P(y)} -> x \in S /\ P(x). Part of #3806.
            Expr::SetFilter(bound, pred) => {
                // x must be in the base set S
                let base_set = bound.domain.as_ref().ok_or_else(|| {
                    AYError::UntranslatableExpr(
                        "SetFilter without domain in BMC membership".to_string(),
                    )
                })?;
                let in_base = self.translate_membership(elem, base_set)?;

                // Substitute the bound variable with the element and evaluate the predicate
                let var_name = &bound.name.node;
                let subs = std::collections::HashMap::from([(var_name.as_str(), elem)]);
                let mut folder = SubstituteExpr {
                    subs,
                    span_policy: SpanPolicy::Preserve,
                };
                let substituted_pred = folder.fold_expr((**pred).clone());
                let pred_holds = self.translate_bool(&substituted_pred)?;

                Ok(self.solver.try_and(in_base, pred_holds)?)
            }

            // x \in {f(y) : y \in S} -> \E y \in S : x = f(y).
            // Mirrors the PDR/CHC SetBuilder rewrite (translate/membership/mod.rs).
            // Combined with the FuncSet quantifier arm, this reaches specs whose
            // membership domain is a `{FunAsSeq(p, n, n) : p \in [1..n -> R]}`
            // permutation set (Einstein/Zebra). Part of the symbolic FuncSet lever.
            Expr::SetBuilder(builder_body, bounds) => {
                let eq_body = Expr::Eq(Box::new(elem.clone()), builder_body.clone());
                let exists =
                    Expr::Exists(bounds.clone(), Box::new(Spanned::new(eq_body, set.span)));
                self.translate_bool(&Spanned::new(exists, set.span))
            }

            // x \in [D -> R] -> \E f \in [D -> R] : x = f.
            // Reuses the FuncSet quantifier enumeration so direct function-set
            // membership (without an enclosing SetBuilder) is also handled.
            Expr::FuncSet(_, _) => {
                let fresh = self.fresh_funcset_member_name();
                let bound = BoundVar {
                    name: Spanned::new(fresh.clone(), set.span),
                    domain: Some(Box::new(set.clone())),
                    pattern: None,
                };
                let eq_body = Expr::Eq(
                    Box::new(elem.clone()),
                    Box::new(Spanned::new(Expr::Ident(fresh, NameId::INVALID), set.span)),
                );
                self.translate_bmc_quantifier(
                    std::slice::from_ref(&bound),
                    &Spanned::new(eq_body, set.span),
                    false,
                )
            }

            _ => Err(AYError::UntranslatableExpr(format!(
                "BMC cannot translate membership in: {:?}",
                std::mem::discriminant(&set.node)
            ))),
        }
    }

    /// Fresh name for a function-set membership witness variable.
    fn fresh_funcset_member_name(&mut self) -> String {
        let id = self.aux_var_counter;
        self.aux_var_counter += 1;
        format!("__fs_member_{id}")
    }

    // === Set operations (Part of #3778) ===

    /// Translate a set expression to an SMT array term `(Array Int Bool)`.
    ///
    /// Handles:
    /// - `SetEnum({e1, ..., en})` -> `(store ... (store (const false) e1 true) ... en true)`
    /// - `Ident/StateVar` for set-typed variables -> lookup at current step
    /// - `Prime(set_expr)` -> lookup at next step
    /// - `Union(S, T)` -> fresh array with pointwise OR constraints
    /// - `Intersect(S, T)` -> fresh array with pointwise AND constraints
    /// - `SetMinus(S, T)` -> fresh array with pointwise AND-NOT constraints
    pub(super) fn translate_set_expr(
        &mut self,
        expr: &Spanned<Expr>,
        universe: &[Term],
    ) -> AYResult<Term> {
        match &expr.node {
            Expr::SetEnum(elements) => {
                // Build: (store (store (const false) e1 true) ... en true)
                let false_val = self.solver.bool_const(false);
                let true_val = self.solver.bool_const(true);
                let mut arr = self.solver.try_const_array(Sort::Int, false_val)?;
                for elem in elements {
                    let elem_term = self.translate_int(elem)?;
                    arr = self.solver.try_store(arr, elem_term, true_val)?;
                }
                Ok(arr)
            }

            Expr::Ident(name, _) | Expr::StateVar(name, ..) => {
                let info = self
                    .vars
                    .get(name)
                    .ok_or_else(|| AYError::UnknownVariable(format!("{name} (set expression)")))?;
                if !matches!(info.sort, TlaSort::Set { .. }) {
                    return Err(AYError::TypeMismatch {
                        name: name.clone(),
                        expected: "Set".to_string(),
                        actual: format!("{}", info.sort),
                    });
                }
                self.get_var_at_step(name, self.current_step)
            }

            Expr::Prime(inner) => match &inner.node {
                Expr::Ident(name, _) | Expr::StateVar(name, ..) => {
                    self.get_var_at_step(name, self.current_step + 1)
                }
                _ => {
                    let old_step = self.current_step;
                    self.current_step += 1;
                    let result = self.translate_set_expr(inner, universe);
                    self.current_step = old_step;
                    result
                }
            },

            Expr::Union(left, right) => {
                let set_s = self.translate_set_expr(left, universe)?;
                let set_t = self.translate_set_expr(right, universe)?;
                self.encode_union(set_s, set_t, universe)
            }

            Expr::Intersect(left, right) => {
                let set_s = self.translate_set_expr(left, universe)?;
                let set_t = self.translate_set_expr(right, universe)?;
                self.encode_intersect(set_s, set_t, universe)
            }

            Expr::SetMinus(left, right) => {
                let set_s = self.translate_set_expr(left, universe)?;
                let set_t = self.translate_set_expr(right, universe)?;
                self.encode_set_minus(set_s, set_t, universe)
            }

            Expr::Range(lo, hi) => self.translate_range_set_term(lo, hi),

            _ => Err(AYError::UntranslatableExpr(format!(
                "BMC cannot translate set expression: {:?}",
                std::mem::discriminant(&expr.node)
            ))),
        }
    }

    /// Translate `S \subseteq T` over a known finite universe.
    ///
    /// `\A u \in universe : (select S u) => (select T u)`
    pub(super) fn translate_subseteq_bmc(
        &mut self,
        left: &Spanned<Expr>,
        right: &Spanned<Expr>,
        universe: &[Term],
    ) -> AYResult<Term> {
        let set_s = self.translate_set_expr(left, universe)?;
        let set_t = self.translate_set_expr(right, universe)?;

        if universe.is_empty() {
            return Ok(self.solver.bool_const(true));
        }

        let mut conjuncts = Vec::with_capacity(universe.len());
        for &u in universe {
            let in_s = self.solver.try_select(set_s, u)?;
            let in_t = self.solver.try_select(set_t, u)?;
            let implication = self.solver.try_implies(in_s, in_t)?;
            conjuncts.push(implication);
        }

        let mut result = conjuncts[0];
        for &c in &conjuncts[1..] {
            result = self.solver.try_and(result, c)?;
        }
        Ok(result)
    }

    /// Encode set union as a fresh array with pointwise OR constraints.
    pub(super) fn encode_union(
        &mut self,
        set_s: Term,
        set_t: Term,
        universe: &[Term],
    ) -> AYResult<Term> {
        let result = self.declare_fresh_set("bmc_union")?;

        for &u in universe {
            let in_s = self.solver.try_select(set_s, u)?;
            let in_t = self.solver.try_select(set_t, u)?;
            let in_result = self.solver.try_select(result, u)?;
            let s_or_t = self.solver.try_or(in_s, in_t)?;
            let eq = self.solver.try_eq(in_result, s_or_t)?;
            self.solver
                .try_assert_term(eq)
                .expect("invariant: eq is Bool-sorted");
        }

        Ok(result)
    }

    /// Encode set intersection as a fresh array with pointwise AND constraints.
    pub(super) fn encode_intersect(
        &mut self,
        set_s: Term,
        set_t: Term,
        universe: &[Term],
    ) -> AYResult<Term> {
        let result = self.declare_fresh_set("bmc_intersect")?;

        for &u in universe {
            let in_s = self.solver.try_select(set_s, u)?;
            let in_t = self.solver.try_select(set_t, u)?;
            let in_result = self.solver.try_select(result, u)?;
            let s_and_t = self.solver.try_and(in_s, in_t)?;
            let eq = self.solver.try_eq(in_result, s_and_t)?;
            self.solver
                .try_assert_term(eq)
                .expect("invariant: eq is Bool-sorted");
        }

        Ok(result)
    }

    /// Encode set difference as a fresh array with pointwise AND-NOT constraints.
    pub(super) fn encode_set_minus(
        &mut self,
        set_s: Term,
        set_t: Term,
        universe: &[Term],
    ) -> AYResult<Term> {
        let result = self.declare_fresh_set("bmc_setminus")?;

        for &u in universe {
            let in_s = self.solver.try_select(set_s, u)?;
            let in_t = self.solver.try_select(set_t, u)?;
            let not_in_t = self.solver.try_not(in_t)?;
            let in_result = self.solver.try_select(result, u)?;
            let s_and_not_t = self.solver.try_and(in_s, not_in_t)?;
            let eq = self.solver.try_eq(in_result, s_and_not_t)?;
            self.solver
                .try_assert_term(eq)
                .expect("invariant: eq is Bool-sorted");
        }

        Ok(result)
    }

    /// Declare a fresh set variable (Array Int Bool) with a unique name.
    pub(crate) fn declare_fresh_set(&mut self, prefix: &str) -> AYResult<Term> {
        let id = self.aux_var_counter;
        self.aux_var_counter += 1;
        let name = format!("__{prefix}_{id}");
        let set_sort = Sort::array(Sort::Int, Sort::Bool);
        Ok(self.solver.declare_const(&name, set_sort))
    }

    // === Function operations (Part of #3786) ===

    /// Translate function application `f[x]` to `(select mapping x)`.
    ///
    /// Looks up the function variable's mapping array at the current step
    /// and applies `select` with the argument.
    ///
    /// Part of #3786: Function encoding in BMC translator.
    pub(super) fn translate_func_apply_bmc(
        &mut self,
        func: &Spanned<Expr>,
        arg: &Spanned<Expr>,
    ) -> AYResult<Term> {
        match &func.node {
            Expr::Ident(name, _) | Expr::StateVar(name, ..) => {
                // Direct function variable: f[x] -> (select f__map x)
                let mapping = self.get_func_mapping_at_step(name, self.current_step)?;
                let arg_term = self.translate_int(arg)?;
                Ok(self.solver.try_select(mapping, arg_term)?)
            }
            Expr::Prime(inner) => match &inner.node {
                Expr::Ident(name, _) | Expr::StateVar(name, ..) => {
                    // f'[x] -> (select f__map__{step+1} x)
                    let mapping = self.get_func_mapping_at_step(name, self.current_step + 1)?;
                    let arg_term = self.translate_int(arg)?;
                    Ok(self.solver.try_select(mapping, arg_term)?)
                }
                _ => Err(AYError::UntranslatableExpr(
                    "BMC function apply: primed non-variable function".to_string(),
                )),
            },
            _ => Err(AYError::UntranslatableExpr(format!(
                "BMC function apply requires variable, got: {:?}",
                std::mem::discriminant(&func.node)
            ))),
        }
    }

    /// Resolve a function expression (variable or primed variable) to its base
    /// name, if it is a direct (possibly primed) function variable reference.
    pub(super) fn func_expr_base_name(func: &Spanned<Expr>) -> Option<String> {
        match &func.node {
            Expr::Ident(name, _) | Expr::StateVar(name, ..) => Some(name.clone()),
            Expr::Prime(inner) => match &inner.node {
                Expr::Ident(name, _) | Expr::StateVar(name, ..) => Some(name.clone()),
                _ => None,
            },
            _ => None,
        }
    }

    /// Build the arithmetic domain-membership term `lo <= elem <= (N+offset)`
    /// for a symbolic-domain (map-only) function. `elem_term` is the already
    /// translated Int index. Part of the function-state all-N encoding.
    pub(super) fn symbolic_func_domain_bound(
        &mut self,
        lo: i64,
        hi_const: &str,
        hi_offset: i64,
        elem_term: Term,
    ) -> AYResult<Term> {
        let lo_t = self.solver.int_const(lo);
        // The symbolic upper-bound constant `N` is declared as a rigid const:
        // the SAME term across steps, so any step resolves it.
        let n_t = self.get_var_at_step(hi_const, self.current_step)?;
        let hi_t = if hi_offset == 0 {
            n_t
        } else {
            let off = self.solver.int_const(hi_offset);
            self.solver.try_add(n_t, off)?
        };
        let ge_lo = self.solver.try_ge(elem_term, lo_t)?;
        let le_hi = self.solver.try_le(elem_term, hi_t)?;
        Ok(self.solver.try_and(ge_lo, le_hi)?)
    }

    /// Translate `DOMAIN f` to the domain set array.
    ///
    /// Returns the `(Array Int Bool)` domain term for the function at the
    /// current step.
    ///
    /// Part of #3786: Function encoding in BMC translator.
    pub(super) fn translate_func_domain_bmc(&mut self, func: &Spanned<Expr>) -> AYResult<Term> {
        match &func.node {
            Expr::Ident(name, _) | Expr::StateVar(name, ..) => {
                self.get_func_domain_at_step(name, self.current_step)
            }
            Expr::Prime(inner) => match &inner.node {
                Expr::Ident(name, _) | Expr::StateVar(name, ..) => {
                    self.get_func_domain_at_step(name, self.current_step + 1)
                }
                _ => Err(AYError::UntranslatableExpr(
                    "BMC DOMAIN: primed non-variable function".to_string(),
                )),
            },
            _ => Err(AYError::UntranslatableExpr(format!(
                "BMC DOMAIN requires variable, got: {:?}",
                std::mem::discriminant(&func.node)
            ))),
        }
    }

    /// Translate `[f EXCEPT ![x] = v]` to an updated mapping array.
    ///
    /// Produces a new mapping term via SMT `store` operations:
    /// - Single: `[f EXCEPT ![a] = b]` -> `(store f_map a b)`
    /// - Multi:  `[f EXCEPT ![a] = b, ![c] = d]` -> `(store (store f_map a b) c d)`
    /// - Nested: `[f EXCEPT ![a][b] = c]` -> `(store f_map a (store (select f_map a) b c))`
    ///
    /// Returns the resulting mapping term. The domain is unchanged (EXCEPT
    /// does not alter the domain of a function in TLA+).
    ///
    /// Part of #3786: Function encoding in BMC translator.
    pub(super) fn translate_func_except_bmc(
        &mut self,
        func: &Spanned<Expr>,
        specs: &[tla_core::ast::ExceptSpec],
    ) -> AYResult<Term> {
        if specs.is_empty() {
            // No-op EXCEPT: [f EXCEPT ] = f — return original mapping
            return self.resolve_func_mapping(func);
        }

        let mut mapping = self.resolve_func_mapping(func)?;

        // Apply each spec sequentially (left to right, as in TLA+ semantics)
        for spec in specs {
            if spec.path.is_empty() {
                return Err(AYError::UnsupportedOp(
                    "BMC EXCEPT with empty path".to_string(),
                ));
            }
            mapping = self.apply_except_spec(mapping, &spec.path, &spec.value)?;
        }

        Ok(mapping)
    }

    /// Resolve the mapping array for a function expression (variable or primed variable).
    pub(super) fn resolve_func_mapping(&mut self, func: &Spanned<Expr>) -> AYResult<Term> {
        match &func.node {
            Expr::Ident(name, _) | Expr::StateVar(name, ..) => {
                self.get_func_mapping_at_step(name, self.current_step)
            }
            Expr::Prime(inner) => match &inner.node {
                Expr::Ident(name, _) | Expr::StateVar(name, ..) => {
                    self.get_func_mapping_at_step(name, self.current_step + 1)
                }
                _ => Err(AYError::UntranslatableExpr(
                    "BMC EXCEPT requires function variable as base".to_string(),
                )),
            },
            Expr::Except(inner_base, inner_specs) => {
                // Chained EXCEPT: [[f EXCEPT ![a] = b] EXCEPT ![c] = d]
                self.translate_func_except_bmc(inner_base, inner_specs)
            }
            _ => Err(AYError::UntranslatableExpr(
                "BMC EXCEPT requires function variable as base".to_string(),
            )),
        }
    }

    /// Apply a single EXCEPT spec (path + value) to a mapping array.
    ///
    /// For a single-level path `![a]`, produces `(store mapping a val)`.
    /// For a nested path `![a][b]`, produces `(store mapping a (store (select mapping a) b val))`.
    fn apply_except_spec(
        &mut self,
        mapping: Term,
        path: &[tla_core::ast::ExceptPathElement],
        value: &Spanned<Expr>,
    ) -> AYResult<Term> {
        match path.len() {
            0 => Err(AYError::UnsupportedOp(
                "BMC EXCEPT with empty path".to_string(),
            )),
            1 => {
                // Single-level: (store mapping key val)
                let key = self.translate_except_path_key(&path[0])?;
                let val_term = self.translate_int(value)?;
                Ok(self.solver.try_store(mapping, key, val_term)?)
            }
            _ => {
                // Nested: [f EXCEPT ![a][b]...[z] = val]
                // Encode as: (store mapping a (store (select mapping a) b ... val))
                // Requires the range sort to be an array (i.e., function of functions).
                let key = self.translate_except_path_key(&path[0])?;
                let inner = self.solver.try_select(mapping, key).map_err(|_| {
                    AYError::UnsupportedOp(format!(
                        "BMC nested EXCEPT path depth {} requires array-of-array function sort",
                        path.len()
                    ))
                })?;
                let updated_inner = self.apply_except_spec(inner, &path[1..], value)?;
                self.solver
                    .try_store(mapping, key, updated_inner)
                    .map_err(|_| {
                        AYError::UnsupportedOp(format!(
                            "BMC nested EXCEPT: inner store failed (path depth {})",
                            path.len()
                        ))
                    })
            }
        }
    }

    /// Translate an EXCEPT path element to an SMT key term.
    fn translate_except_path_key(
        &mut self,
        elem: &tla_core::ast::ExceptPathElement,
    ) -> AYResult<Term> {
        match elem {
            tla_core::ast::ExceptPathElement::Index(idx) => self.translate_int(idx),
            tla_core::ast::ExceptPathElement::Field(field_name) => {
                Err(AYError::UnsupportedOp(format!(
                    "BMC EXCEPT on record field '.{}' not supported (use record encoding)",
                    field_name.name.node
                )))
            }
        }
    }

    /// Whether the (single) bound variable's domain is a set of string
    /// literals, in which case the function construction must use a native
    /// `String`-indexed encoding (#5). Returns `false` for any non-string or
    /// non-finite-enum domain (the existing integer path handles those).
    pub(super) fn func_construct_keys_are_strings(bound_vars: &[tla_core::ast::BoundVar]) -> bool {
        let Some(bv) = bound_vars.first() else {
            return false;
        };
        let Some(domain) = bv.domain.as_ref() else {
            return false;
        };
        match &domain.node {
            Expr::SetEnum(elems) => {
                !elems.is_empty() && elems.iter().all(|e| matches!(e.node, Expr::String(_)))
            }
            _ => false,
        }
    }

    /// Translate function construction `[x \in S |-> expr]`.
    ///
    /// Builds a (domain, mapping) pair:
    /// - domain: translate S as a set expression (Array Int Bool)
    /// - mapping: a fresh `(Array Int RangeSort)` with constraints from the body
    ///
    /// For finite set domains `{e1, ..., en}`, adds per-element constraints:
    /// `(select mapping ei) = body_with_x_replaced_by_ei` for each `ei`.
    ///
    /// Returns a `(domain_term, mapping_term)` pair.
    ///
    /// Part of #3786: Function encoding in BMC translator.
    pub(super) fn translate_func_construct_bmc(
        &mut self,
        bound_vars: &[tla_core::ast::BoundVar],
        body: &Spanned<Expr>,
    ) -> AYResult<(Term, Term)> {
        // Declare a fresh mapping array, then constrain it with the per-element
        // value constraints. Used when the construction is *not* directly bound
        // to a known target function variable (e.g. a standalone FuncDef value).
        //
        // The fresh mapping's index sort must match the construction's domain
        // key sort: string-literal domains use a native `(Array String _)` so a
        // string key cannot alias an integer-literal key (soundness, #5).
        let key_index_sort = if Self::func_construct_keys_are_strings(bound_vars) {
            Sort::String
        } else {
            Sort::Int
        };
        let id = self.aux_var_counter;
        self.aux_var_counter += 1;
        let map_name = format!("__func_map_{id}");
        let map_sort = Sort::array(key_index_sort, Sort::Int);
        let mapping = self.solver.declare_const(&map_name, map_sort);
        let domain_arr = self.translate_func_construct_bmc_into(bound_vars, body, mapping)?;
        Ok((domain_arr, mapping))
    }

    /// Constrain a *provided* `target_mapping` array so that
    /// `(select target_mapping e) = body[x := e]` for every element `e` of the
    /// (finite) construction domain, and return the construction's domain array.
    ///
    /// Unlike [`Self::translate_func_construct_bmc`], this does NOT declare a
    /// fresh mapping array. Callers that already own the target mapping —
    /// directed function-construction equality `f = [x \in S |-> e]` — pass
    /// `f`'s own `f__map__{step}` array so the value constraints land on it
    /// directly. That avoids emitting an array-to-array equality
    /// `f__map = __func_map` whose RHS carries several `select` constraints,
    /// which trips an AY QF_AUFLIA model-construction gap: AY cannot certify a
    /// complete model for `select` reads taken through such an alias and
    /// conservatively degrades `Sat -> Unknown` (the symptom fixed in
    /// `test_bmc_func_construct_eq_init`). Constraining the target in place
    /// matches the always-SAT `test_bmc_assert_concrete_func_state` encoding.
    pub(super) fn translate_func_construct_bmc_into(
        &mut self,
        bound_vars: &[tla_core::ast::BoundVar],
        body: &Spanned<Expr>,
        target_mapping: Term,
    ) -> AYResult<Term> {
        // Only support single bound variable for now
        if bound_vars.len() != 1 {
            return Err(AYError::UnsupportedOp(format!(
                "BMC function construction supports single bound variable, got {}",
                bound_vars.len()
            )));
        }
        let bv = &bound_vars[0];
        let var_name = &bv.name.node;

        // Domain must be a finite set enumeration for now
        let domain_expr = bv.domain.as_ref().ok_or_else(|| {
            AYError::UntranslatableExpr("BMC function construction requires domain".to_string())
        })?;

        // Extract domain elements for constraint generation
        let domain_elements: Vec<Spanned<Expr>> = match &domain_expr.node {
            Expr::SetEnum(elems) => elems.clone(),
            Expr::Range(lo, hi) => {
                // Expand lo..hi to individual elements (only for literal ranges)
                let lo_val = match &lo.node {
                    Expr::Int(n) => i64::try_from(n)
                        .map_err(|_| AYError::IntegerOverflow(format!("range low {n}")))?,
                    _ => {
                        return Err(AYError::UntranslatableExpr(
                            "BMC FuncDef: non-literal range bounds".to_string(),
                        ));
                    }
                };
                let hi_val = match &hi.node {
                    Expr::Int(n) => i64::try_from(n)
                        .map_err(|_| AYError::IntegerOverflow(format!("range high {n}")))?,
                    _ => {
                        return Err(AYError::UntranslatableExpr(
                            "BMC FuncDef: non-literal range bounds".to_string(),
                        ));
                    }
                };
                (lo_val..=hi_val)
                    .map(|i| Spanned::dummy(Expr::Int(num_bigint::BigInt::from(i))))
                    .collect()
            }
            _ => {
                return Err(AYError::UntranslatableExpr(
                    "BMC function construction requires finite set domain".to_string(),
                ));
            }
        };

        // Determine the domain *key* sort from the live domain AST. A domain
        // of string literals is encoded with native `Sort::String` keys so a
        // string key can never alias an integer-literal key (soundness, #5).
        // The domain is homogeneous (TLA+ function domains are single-typed):
        // if every element is a string literal the key sort is `String`,
        // otherwise `Int` (the existing path).
        let keys_are_strings = !domain_elements.is_empty()
            && domain_elements
                .iter()
                .all(|e| matches!(e.node, Expr::String(_)));
        if domain_elements
            .iter()
            .any(|e| matches!(e.node, Expr::String(_)))
            && !keys_are_strings
        {
            return Err(AYError::UnsupportedOp(
                "BMC function construction: mixed string/non-string domain keys".to_string(),
            ));
        }
        let key_tla_sort = if keys_are_strings {
            TlaSort::String
        } else {
            TlaSort::Int
        };
        let key_index_sort = if keys_are_strings {
            Sort::String
        } else {
            Sort::Int
        };

        // Translate a single domain key to its SMT term, using a *native* string
        // constant for string keys (NOT `translate_int`, which interns strings
        // into the integer namespace and would re-introduce the aliasing).
        let key_term = |this: &mut Self, elem: &Spanned<Expr>| -> AYResult<Term> {
            if keys_are_strings {
                match &elem.node {
                    Expr::String(s) => Ok(this.solver.string_const(s)),
                    _ => Err(AYError::UnsupportedOp(
                        "BMC function construction: non-string key in string-keyed domain"
                            .to_string(),
                    )),
                }
            } else {
                this.translate_int(elem)
            }
        };

        // Build domain set: (store ... (store (const false) e1 true) ... en true)
        let true_val = self.solver.bool_const(true);
        // Default ("not in domain") value for the const-array. For string-keyed
        // domains we use a *fresh* Bool constant pinned to `false` rather than
        // the shared interned `false`. AY interns `const-array` by its value
        // term only (not by index sort — AY bug #6734), so an `(Array String
        // Bool)` and an `(Array Int Bool)` const-array sharing the same `false`
        // value term collide: in release the second request silently returns the
        // first's *wrong-sorted* term. A distinct (but provably-false) default
        // term gives the string const-array its own identity. Semantically
        // identical to `false`, so the domain still defaults to "absent".
        let false_default = if keys_are_strings {
            let d = self.solver.declare_const(
                &format!("__strdom_false_{}", self.aux_var_counter),
                Sort::Bool,
            );
            self.aux_var_counter += 1;
            let shared_false = self.solver.bool_const(false);
            let d_is_false = self.solver.try_eq(d, shared_false)?;
            self.solver
                .try_assert_term(d_is_false)
                .expect("invariant: eq is Bool-sorted");
            d
        } else {
            self.solver.bool_const(false)
        };
        let mut domain_arr = self.solver.try_const_array(key_index_sort, false_default)?;
        for elem in &domain_elements {
            let elem_term = key_term(self, elem)?;
            domain_arr = self.solver.try_store(domain_arr, elem_term, true_val)?;
        }

        // Constrain the provided mapping array with per-element value
        // constraints. For each domain element e: (select mapping e) = body[x := e]
        let mapping = target_mapping;

        // For each domain element, substitute the bound variable and constrain
        for elem in &domain_elements {
            let elem_term = key_term(self, elem)?;

            // Translate body with a temporary variable binding.
            // We temporarily add a scalar variable for the bound var, pointing
            // to elem_term, then translate the body, then remove it. The bound
            // variable carries the domain key sort so a body that reads `x`
            // (e.g. as a string) resolves to the correctly-sorted term.
            let _bv_step_name = format!("{var_name}__{}", self.current_step);
            let existed_before = self.vars.contains_key(var_name);

            // Temporarily inject bound variable as a scalar
            if !existed_before {
                self.vars.insert(
                    var_name.clone(),
                    super::BmcVarInfo {
                        sort: key_tla_sort.clone(),
                        terms: vec![elem_term; self.bound_k + 1],
                    },
                );
            } else {
                // Save old value and replace with elem_term for current step
                let info = self.vars.get_mut(var_name).expect("checked above");
                info.terms[self.current_step] = elem_term;
            }

            let body_term = self.translate_int(body)?;

            // Clean up: remove temporary binding if we added it
            if !existed_before {
                self.vars.remove(var_name);
            }

            let selected = self.solver.try_select(mapping, elem_term)?;
            let eq = self.solver.try_eq(selected, body_term)?;
            self.solver
                .try_assert_term(eq)
                .expect("invariant: eq is Bool-sorted");
        }

        Ok(domain_arr)
    }

    /// Declare a fresh mapping array `(Array Int Sort)` with a unique name.
    ///
    /// Part of #3786.
    #[allow(dead_code)]
    fn declare_fresh_mapping(&mut self, prefix: &str, range_sort: Sort) -> AYResult<Term> {
        let id = self.aux_var_counter;
        self.aux_var_counter += 1;
        let name = format!("__{prefix}_{id}");
        let arr_sort = Sort::array(Sort::Int, range_sort);
        Ok(self.solver.declare_const(&name, arr_sort))
    }

    // === Incremental BMC methods (Part of #3724) ===

    /// Translate Next for a specific step transition (step -> step+1).
    ///
    /// Returns a term representing `Next(s_step, s_{step+1})`.
    /// This is the incremental counterpart to asserting all Next transitions
    /// at once: the caller asserts each transition permanently as the search
    /// deepens.
    ///
    /// Part of #3724: Incremental SMT solver for TLA+ BMC.
    pub fn translate_next_at_step(
        &mut self,
        next_expr: &Spanned<Expr>,
        step: usize,
    ) -> AYResult<Term> {
        self.translate_next(next_expr, step)
    }

    /// Translate negated safety at a specific step for incremental checking.
    ///
    /// Returns a term representing `!Safety(s_step)`. The caller pushes a
    /// scope, asserts this term, checks SAT, then pops to undo the negation
    /// before moving to the next depth.
    ///
    /// Part of #3724: Incremental SMT solver for TLA+ BMC.
    pub fn check_safety_at_step(
        &mut self,
        safety_expr: &Spanned<Expr>,
        step: usize,
    ) -> AYResult<Term> {
        self.translate_not_safety_at_step(safety_expr, step)
    }

    /// Run incremental BMC, checking depths 0 through `bound_k` one at a time.
    ///
    /// The algorithm reuses solver state across depths:
    /// 1. Assert `Init(s_0)` permanently.
    /// 2. For each depth `d` from 0 to `bound_k`:
    ///    a. Push scope.
    ///    b. Assert `!Safety(s_d)`.
    ///    c. Check SAT. If SAT, return `Some(d)` — counterexample found at depth `d`.
    ///    d. Pop scope (undo the negated safety assertion).
    ///    e. If `d < bound_k`, assert `Next(s_d, s_{d+1})` permanently.
    /// 3. If all depths are UNSAT, return `None`.
    ///
    /// This is more efficient than monolithic BMC because:
    /// - Shorter counterexamples are found first (no wasted work on deeper depths).
    /// - The solver incrementally builds on prior assertions rather than starting
    ///   from scratch at each depth.
    /// - Push/pop avoids re-asserting Init and prior Next transitions.
    ///
    /// Part of #3724: Incremental SMT solver for TLA+ BMC.
    pub fn run_incremental(
        &mut self,
        init: &Spanned<Expr>,
        next: &Spanned<Expr>,
        safety: &Spanned<Expr>,
    ) -> AYResult<Option<usize>> {
        // Step 1: Assert Init at step 0 permanently
        let init_term = self.translate_init(init)?;
        self.assert(init_term);

        // Step 2: Iterate over depths 0..=bound_k
        for d in 0..=self.bound_k {
            // 2a: Push scope for the negated safety check
            self.push_scope()?;

            // 2b: Assert !Safety at step d
            let not_safety = self.check_safety_at_step(safety, d)?;
            self.assert(not_safety);

            // 2c: Check SAT
            let result = self.try_check_sat()?;
            match result {
                SolveResult::Sat => {
                    // Counterexample found at depth d.
                    // Leave scope pushed so the model is accessible.
                    return Ok(Some(d));
                }
                SolveResult::Unsat(_) => {
                    // Safety holds at depth d. Pop and continue deeper.
                }
                _ => {
                    // Unknown or other result — pop and report error
                    self.pop_scope()?;
                    return Err(AYError::SolverUnknown);
                }
            }

            // 2d: Pop scope (undo !Safety)
            self.pop_scope()?;

            // 2e: Assert Next(d, d+1) permanently (if more depths remain)
            if d < self.bound_k {
                let next_term = self.translate_next_at_step(next, d)?;
                self.assert(next_term);
            }
        }

        // Step 3: All depths UNSAT — safety holds up to bound_k
        Ok(None)
    }

    // === Sequence operations (Part of #3793) ===

    /// Translate `Len(s)` — returns the length term for a sequence variable.
    ///
    /// Part of #3793.
    pub(super) fn translate_seq_len_bmc(&mut self, seq_expr: &Spanned<Expr>) -> AYResult<Term> {
        let (name, step) = self.resolve_seq_var(seq_expr)?;
        self.get_seq_length_at_step(&name, step)
    }

    /// Translate `Head(s)` — `(select arr 1)`.
    ///
    /// Part of #3793.
    pub(super) fn translate_seq_head_bmc(&mut self, seq_expr: &Spanned<Expr>) -> AYResult<Term> {
        let (name, step) = self.resolve_seq_var(seq_expr)?;
        let arr = self.get_seq_array_at_step(&name, step)?;
        let one = self.solver.int_const(1);
        Ok(self.solver.try_select(arr, one)?)
    }

    /// Translate `Tail(s)` — a new sequence with the first element removed.
    ///
    /// Returns `(array_term, length_term)`. The result array is a fresh
    /// array with shifted indices: `result[i] = s[i+1]` for `1 <= i < len`.
    /// The result length is `len - 1`.
    ///
    /// Part of #3793.
    pub(super) fn translate_seq_tail_bmc(
        &mut self,
        seq_expr: &Spanned<Expr>,
    ) -> AYResult<(Term, Term)> {
        let (name, step) = self.resolve_seq_var(seq_expr)?;
        let arr = self.get_seq_array_at_step(&name, step)?;
        let len = self.get_seq_length_at_step(&name, step)?;
        let max_len = self.get_seq_max_len(&name)?;

        // Build the tail as an explicit STORE CHAIN over a fresh base array
        // rather than a fresh FREE array pinned by sparse guarded select-
        // equalities. Newer AY (>= the array-model-finding changes around
        // ay@0794a38f) abstains (Unknown) when asked to MODEL-FIND a free array
        // whose contents are constrained only by `i < len => select(result,i)=..`
        // implications — it cannot synthesize a witness. An explicit store chain
        // gives it a concrete witness array it can always model (same shape AY's
        // store-chain-over-free-base handling expects). Positions i >= len are
        // beyond the result length (len' = len-1) and are don't-cares, so storing
        // seq[i+1] there unconditionally is semantics-preserving for the
        // length-bounded encoding (and keeps the UNSAT cases UNSAT: in-bounds
        // contradictions are still forced exactly).
        let mut result_arr = self.declare_fresh_seq_array("bmc_seq_tail")?;
        for i in 1..max_len {
            let i_term = self.solver.int_const(i as i64);
            let i_plus_1 = self.solver.int_const((i + 1) as i64);
            let src_val = self.solver.try_select(arr, i_plus_1)?;
            result_arr = self.solver.try_store(result_arr, i_term, src_val)?;
        }

        // len' = max(0, len - 1)
        // Guard against Tail of empty sequence producing negative length.
        // TLA+ leaves Tail(<<>>) undefined, but the SMT encoding must not
        // produce an unconstrained negative length term.
        let one = self.solver.int_const(1);
        let zero = self.solver.int_const(0);
        let raw_len = self.solver.try_sub(len, one)?;
        let len_ge_one = self.solver.try_ge(len, one)?;
        let new_len = self.solver.try_ite(len_ge_one, raw_len, zero)?;

        Ok((result_arr, new_len))
    }

    /// Translate `Append(s, e)` — append element `e` to sequence `s`.
    ///
    /// Returns `(array_term, length_term)`.
    /// Result: `(store arr (len+1) e)` with `len' = len + 1`.
    ///
    /// Part of #3793.
    pub(super) fn translate_seq_append_bmc(
        &mut self,
        seq_expr: &Spanned<Expr>,
        elem_expr: &Spanned<Expr>,
    ) -> AYResult<(Term, Term)> {
        let (name, step) = self.resolve_seq_var(seq_expr)?;
        let arr = self.get_seq_array_at_step(&name, step)?;
        let len = self.get_seq_length_at_step(&name, step)?;

        let elem_term = self.translate_int(elem_expr)?;

        let one = self.solver.int_const(1);
        let new_len = self.solver.try_add(len, one)?;
        let new_arr = self.solver.try_store(arr, new_len, elem_term)?;

        Ok((new_arr, new_len))
    }

    /// Translate `SubSeq(s, m, n)` — extract a subsequence from index m to n.
    ///
    /// Returns `(array_term, length_term)`. The result array is a fresh
    /// array with shifted indices: `result[i] = s[m + i - 1]` for `1 <= i <= n - m + 1`.
    /// The result length is `max(0, n - m + 1)`.
    ///
    /// Part of #3793.
    pub(super) fn translate_seq_subseq_bmc(
        &mut self,
        seq_expr: &Spanned<Expr>,
        m_expr: &Spanned<Expr>,
        n_expr: &Spanned<Expr>,
    ) -> AYResult<(Term, Term)> {
        let (name, step) = self.resolve_seq_var(seq_expr)?;
        let arr = self.get_seq_array_at_step(&name, step)?;
        let max_len = self.get_seq_max_len(&name)?;

        let m_term = self.translate_int(m_expr)?;
        let n_term = self.translate_int(n_expr)?;

        // STORE CHAIN (see translate_seq_tail_bmc): an explicit witness array
        // newer AY can model-find, vs a free array pinned only by sparse guarded
        // select-equalities (which it now abstains on). result[i] = s[m+i-1].
        // Positions beyond new_len = n-m+1 are don't-cares (sequence reads are
        // length-bounded), so a literal SubSeq range only needs `new_len` stores.
        // Symbolic ranges retain the conservative max_len chain. Besides reducing
        // terms, this avoids a sharp array-refinement cliff on the common
        // constant-bound TLA+ form without changing the encoded models.
        let copy_len = match (&m_expr.node, &n_expr.node) {
            (Expr::Int(m), Expr::Int(n)) => {
                let literal_len = n - m + BigInt::from(1);
                if literal_len <= BigInt::from(0) {
                    0
                } else {
                    usize::try_from(&literal_len)
                        .unwrap_or(max_len)
                        .min(max_len)
                }
            }
            _ => max_len,
        };
        let mut result_arr = self.declare_fresh_seq_array("bmc_seq_subseq")?;
        for i in 1..=copy_len {
            let i_term = self.solver.int_const(i as i64);
            let one = self.solver.int_const(1);
            // src_idx = m + i - 1
            let m_plus_i = self.solver.try_add(m_term, i_term)?;
            let src_idx = self.solver.try_sub(m_plus_i, one)?;
            let src_val = self.solver.try_select(arr, src_idx)?;
            result_arr = self.solver.try_store(result_arr, i_term, src_val)?;
        }

        // len' = max(0, n - m + 1)
        let one = self.solver.int_const(1);
        let zero = self.solver.int_const(0);
        let n_minus_m = self.solver.try_sub(n_term, m_term)?;
        let raw_len = self.solver.try_add(n_minus_m, one)?;
        let len_ge_zero = self.solver.try_ge(raw_len, zero)?;
        let new_len = self.solver.try_ite(len_ge_zero, raw_len, zero)?;

        Ok((result_arr, new_len))
    }

    /// Translate `s \o t` (concatenation) — concatenate two sequences.
    ///
    /// Returns `(array_term, length_term)`. The result array combines both
    /// sequences: `result[i] = s[i]` for `1 <= i <= len_s`, and
    /// `result[len_s + j] = t[j]` for `1 <= j <= len_t`.
    /// The result length is `len_s + len_t`.
    ///
    /// Part of #3793.
    pub(super) fn translate_seq_concat_bmc(
        &mut self,
        s_expr: &Spanned<Expr>,
        t_expr: &Spanned<Expr>,
    ) -> AYResult<(Term, Term)> {
        let (s_name, s_step) = self.resolve_seq_var(s_expr)?;
        let s_arr = self.get_seq_array_at_step(&s_name, s_step)?;
        let s_len = self.get_seq_length_at_step(&s_name, s_step)?;
        let s_max = self.get_seq_max_len(&s_name)?;

        let (t_name, t_step) = self.resolve_seq_var(t_expr)?;
        let t_arr = self.get_seq_array_at_step(&t_name, t_step)?;
        let t_len = self.get_seq_length_at_step(&t_name, t_step)?;
        let t_max = self.get_seq_max_len(&t_name)?;

        let result_arr = self.declare_fresh_seq_array("bmc_seq_concat")?;

        // Copy s elements: for i in 1..=s_max, result[i] = s[i] when i <= len_s
        for i in 1..=s_max {
            let i_term = self.solver.int_const(i as i64);
            let in_bounds = self.solver.try_le(i_term, s_len)?;
            let src_val = self.solver.try_select(s_arr, i_term)?;
            let dst_val = self.solver.try_select(result_arr, i_term)?;
            let vals_eq = self.solver.try_eq(dst_val, src_val)?;
            let guarded = self.solver.try_implies(in_bounds, vals_eq)?;
            self.solver
                .try_assert_term(guarded)
                .expect("invariant: implies is Bool-sorted");
        }

        // Copy t elements: for j in 1..=t_max, result[len_s + j] = t[j] when j <= len_t
        for j in 1..=t_max {
            let j_term = self.solver.int_const(j as i64);
            let in_bounds = self.solver.try_le(j_term, t_len)?;
            let dst_idx = self.solver.try_add(s_len, j_term)?;
            let src_val = self.solver.try_select(t_arr, j_term)?;
            let dst_val = self.solver.try_select(result_arr, dst_idx)?;
            let vals_eq = self.solver.try_eq(dst_val, src_val)?;
            let guarded = self.solver.try_implies(in_bounds, vals_eq)?;
            self.solver
                .try_assert_term(guarded)
                .expect("invariant: implies is Bool-sorted");
        }

        // len' = len_s + len_t
        let new_len = self.solver.try_add(s_len, t_len)?;

        Ok((result_arr, new_len))
    }

    /// Translate `UNCHANGED seq` for a sequence variable.
    ///
    /// Asserts: `arr' = arr /\ len' = len`.
    ///
    /// Part of #3793.
    fn translate_unchanged_seq(&mut self, name: &str) -> AYResult<Term> {
        let curr_arr = self.get_seq_array_at_step(name, self.current_step)?;
        let next_arr = self.get_seq_array_at_step(name, self.current_step + 1)?;
        let curr_len = self.get_seq_length_at_step(name, self.current_step)?;
        let next_len = self.get_seq_length_at_step(name, self.current_step + 1)?;

        let arr_eq = self.solver.try_eq(next_arr, curr_arr)?;
        let len_eq = self.solver.try_eq(next_len, curr_len)?;
        Ok(self.solver.try_and(arr_eq, len_eq)?)
    }

    /// Translate UNCHANGED for a function variable.
    ///
    /// Asserts: `dom' = dom /\ map' = map`.
    ///
    /// Part of #3786.
    fn translate_unchanged_func(&mut self, name: &str) -> AYResult<Term> {
        let curr_dom = self.get_func_domain_at_step(name, self.current_step)?;
        let next_dom = self.get_func_domain_at_step(name, self.current_step + 1)?;
        let curr_map = self.get_func_mapping_at_step(name, self.current_step)?;
        let next_map = self.get_func_mapping_at_step(name, self.current_step + 1)?;

        let dom_eq = self.solver.try_eq(next_dom, curr_dom)?;
        let map_eq = self.solver.try_eq(next_map, curr_map)?;
        Ok(self.solver.try_and(dom_eq, map_eq)?)
    }

    /// Check whether an expression refers to a declared sequence variable.
    ///
    /// Part of #3793.
    pub(super) fn is_seq_var_expr(&self, expr: &Spanned<Expr>) -> bool {
        match &expr.node {
            Expr::Ident(name, _) | Expr::StateVar(name, ..) => self.seq_vars.contains_key(name),
            Expr::Prime(inner) => match &inner.node {
                Expr::Ident(name, _) | Expr::StateVar(name, ..) => self.seq_vars.contains_key(name),
                _ => false,
            },
            _ => false,
        }
    }

    /// Resolve a sequence expression to `(variable_name, step)`.
    ///
    /// Part of #3793.
    pub(super) fn resolve_seq_var(&self, expr: &Spanned<Expr>) -> AYResult<(String, usize)> {
        match &expr.node {
            Expr::Ident(name, _) | Expr::StateVar(name, ..) => {
                if self.seq_vars.contains_key(name) {
                    Ok((name.clone(), self.current_step))
                } else {
                    Err(AYError::UnknownVariable(format!("sequence {name}")))
                }
            }
            Expr::Prime(inner) => match &inner.node {
                Expr::Ident(name, _) | Expr::StateVar(name, ..) => {
                    if self.seq_vars.contains_key(name) {
                        Ok((name.clone(), self.current_step + 1))
                    } else {
                        Err(AYError::UnknownVariable(format!("sequence {name}")))
                    }
                }
                _ => Err(AYError::UntranslatableExpr(
                    "BMC sequence operation requires variable reference".to_string(),
                )),
            },
            _ => Err(AYError::UntranslatableExpr(
                "BMC sequence operation requires variable reference".to_string(),
            )),
        }
    }

    /// Declare a fresh sequence array `(Array Int Int)` with a unique name.
    ///
    /// Part of #3793.
    fn declare_fresh_seq_array(&mut self, prefix: &str) -> AYResult<Term> {
        let id = self.aux_var_counter;
        self.aux_var_counter += 1;
        let name = format!("__{prefix}_{id}");
        let arr_sort = Sort::array(Sort::Int, Sort::Int);
        Ok(self.solver.declare_const(&name, arr_sort))
    }

    // === Quantifier translation ===

    /// Maximum domain size for direct expansion in BMC. Domains larger
    /// than this use Skolemization for `\E`. Mirrors the threshold in
    /// `translate::quantifier`.
    const BMC_SKOLEM_THRESHOLD: i64 = 16;

    /// Maximum range size for `\A` expansion. Beyond this the conjunction
    /// is too large for QF_LIA.
    const BMC_FORALL_RANGE_LIMIT: i64 = 100;

    /// Translate a quantified formula (`\A` or `\E`) in BMC context.
    ///
    /// For `\E`, uses Skolemization (fresh witness constant) when the
    /// domain exceeds `BMC_SKOLEM_THRESHOLD`, and direct disjunction
    /// expansion for small domains.
    ///
    /// For `\A`, uses direct conjunction expansion (no native `\forall`
    /// in QF_LIA).
    pub(super) fn translate_bmc_quantifier(
        &mut self,
        bounds: &[BoundVar],
        body: &Spanned<Expr>,
        is_forall: bool,
    ) -> AYResult<Term> {
        // Only handle single bound variable
        if bounds.len() != 1 {
            return Err(AYError::UnsupportedOp(
                "BMC quantifiers with multiple bound variables not supported".to_string(),
            ));
        }

        let bound = &bounds[0];
        let domain = match &bound.domain {
            Some(d) => d,
            None => {
                return Err(AYError::UnsupportedOp(
                    "BMC quantifiers require bounded domain".to_string(),
                ))
            }
        };

        // BOOLEAN domain — always expand (only 2 elements)
        if let Expr::Ident(name, _) = &domain.node {
            if name == "BOOLEAN" {
                return self.expand_bmc_boolean_quantifier(bound, body, is_forall);
            }
        }

        // Powerset domain: \Q T \in SUBSET S : P(T)
        // Enumerate all 2^n subsets and expand directly. Each subset
        // becomes an `(Array Int Bool)` term, and we temporarily bind the
        // quantified variable as a set-typed variable for body translation.
        if let Expr::Powerset(base) = &domain.node {
            return self.expand_bmc_powerset_quantifier(bound, base, body, is_forall);
        }

        // Finite set enumeration {e1, ..., en}
        if let Expr::SetEnum(elements) = &domain.node {
            if is_forall {
                return self.expand_bmc_set_enum_quantifier(bound, elements, body, true);
            }
            if elements.len() as i64 > Self::BMC_SKOLEM_THRESHOLD {
                return self.skolemize_bmc_exists_set_enum(bound, elements, body);
            }
            return self.expand_bmc_set_enum_quantifier(bound, elements, body, false);
        }

        // Integer range lo..hi
        if let Expr::Range(lo, hi) = &domain.node {
            if let (Expr::Int(lo_val), Expr::Int(hi_val)) = (&lo.node, &hi.node) {
                let range_size = hi_val - lo_val + BigInt::from(1);
                if let Ok(size) = i64::try_from(&range_size) {
                    if size <= 0 {
                        return Ok(self.solver.bool_const(is_forall));
                    }
                    if !is_forall {
                        if size > Self::BMC_SKOLEM_THRESHOLD {
                            return self.skolemize_bmc_exists_range(bound, lo_val, hi_val, body);
                        }
                        return self
                            .expand_bmc_range_quantifier(bound, lo_val, hi_val, body, false);
                    }
                    if size <= Self::BMC_FORALL_RANGE_LIMIT {
                        return self.expand_bmc_range_quantifier(bound, lo_val, hi_val, body, true);
                    }
                }
            }
        }

        // SetFilter: \Q y \in {z \in S : P(z)} : body
        if let Expr::SetFilter(filter_bound, pred) = &domain.node {
            if let Some(inner_domain) = &filter_bound.domain {
                let filter_var = &filter_bound.name.node;
                let bound_var = &bound.name.node;

                let replacement =
                    Spanned::new(Expr::Ident(bound_var.clone(), NameId::INVALID), pred.span);
                let mut sub = SubstituteExpr {
                    subs: HashMap::from([(filter_var.as_str(), &replacement)]),
                    span_policy: SpanPolicy::Preserve,
                };
                let pred_spanned = sub.fold_expr(*pred.clone());

                let new_body = if is_forall {
                    Expr::Implies(Box::new(pred_spanned), Box::new(body.clone()))
                } else {
                    Expr::And(Box::new(pred_spanned), Box::new(body.clone()))
                };
                let new_body_spanned = Spanned::new(new_body, body.span);

                let new_bound = BoundVar {
                    name: bound.name.clone(),
                    domain: Some(inner_domain.clone()),
                    pattern: bound.pattern.clone(),
                };

                return self.translate_bmc_quantifier(&[new_bound], &new_body_spanned, is_forall);
            }
        }

        // FuncSet domain: \Q f \in [D -> R] : body
        // Exhaustively enumerate the concrete function table. Sound (exact)
        // because every function is substituted as a concrete value.
        if let Expr::FuncSet(func_domain, func_range) = &domain.node {
            return self.expand_bmc_funcset_quantifier(
                bound,
                func_domain,
                func_range,
                body,
                is_forall,
            );
        }

        // Set-difference / union / intersection over enumerable integer
        // domains: \Q i \in (1..5 \ {3}) : P(i) (the Einstein "GreenOwner"
        // rule). When the whole domain reduces to a concrete finite integer set
        // we enumerate it directly. Sound (exact) — the element set is fully
        // computed, not abstracted.
        if matches!(
            &domain.node,
            Expr::SetMinus(_, _) | Expr::Union(_, _) | Expr::Intersect(_, _)
        ) {
            if let Some(elems) = enumerate_concrete_int_set(domain) {
                let element_exprs: Vec<Spanned<Expr>> = elems
                    .into_iter()
                    .map(|v| Spanned::new(Expr::Int(BigInt::from(v)), domain.span))
                    .collect();
                return self.expand_bmc_set_enum_quantifier(bound, &element_exprs, body, is_forall);
            }
        }

        Err(AYError::UnsupportedOp(
            "BMC quantifiers over this domain type not supported".to_string(),
        ))
    }

    // --- BMC powerset expansion ---

    /// Expand a BMC quantifier over a powerset domain: `\Q T \in SUBSET S : P(T)`.
    ///
    /// Enumerates all 2^n subsets of S (where n = |universe of S|) and
    /// evaluates the body for each subset. The bound variable T is
    /// temporarily injected as a set-typed variable at all BMC steps,
    /// pointing to the concrete subset term.
    ///
    /// For `\E`: returns the disjunction `P(sub_0) \/ P(sub_1) \/ ...`
    /// For `\A`: returns the conjunction `P(sub_0) /\ P(sub_1) /\ ...`
    ///
    /// Requires the base set universe to have at most 16 elements
    /// (MAX_POWERSET_SIZE). Beyond that, 2^n subsets is impractical.
    ///
    /// **Nested SUBSET detection (Part of #3826):** When `base` is itself
    /// `Expr::Powerset(inner)`, this is `\Q T \in SUBSET(SUBSET S) : P(T)`.
    /// Instead of enumerating 2^(2^n) outer subsets, routes to
    /// [`expand_bmc_nested_powerset_quantifier`] which uses
    /// `NestedPowersetEncoder` with cardinality-filtered base elements.
    fn expand_bmc_powerset_quantifier(
        &mut self,
        bound: &BoundVar,
        base: &Spanned<Expr>,
        body: &Spanned<Expr>,
        is_forall: bool,
    ) -> AYResult<Term> {
        // Detect nested SUBSET pattern: base is SUBSET(inner)
        // This means the original domain was SUBSET(SUBSET(inner))
        if let Expr::Powerset(inner) = &base.node {
            return self.expand_bmc_nested_powerset_quantifier(bound, inner, body, is_forall);
        }

        let var_name = &bound.name.node;

        // Enumerate all concrete subset terms
        let subsets = self.enumerate_powerset_subsets(base)?;

        if subsets.is_empty() {
            return Ok(self.solver.bool_const(is_forall));
        }

        let mut results = Vec::with_capacity(subsets.len());
        let existed_before = self.vars.contains_key(var_name);

        for subset_term in &subsets {
            // Temporarily inject the bound variable as a set-typed variable
            // pointing to this specific subset term at all steps.
            if !existed_before {
                self.vars.insert(
                    var_name.clone(),
                    super::BmcVarInfo {
                        sort: TlaSort::Set {
                            element_sort: Box::new(TlaSort::Int),
                        },
                        terms: vec![*subset_term; self.bound_k + 1],
                    },
                );
            } else {
                let info = self.vars.get_mut(var_name).expect("checked above");
                for t in info.terms.iter_mut() {
                    *t = *subset_term;
                }
            }

            let body_term = self.translate_bool(body)?;
            results.push(body_term);
        }

        // Clean up: remove temporary binding if we added it
        if !existed_before {
            self.vars.remove(var_name);
        }

        self.combine_bool_terms(&results, is_forall)
    }

    // --- Nested SUBSET(SUBSET) expansion (Part of #3826) ---

    /// Maximum inner set size for nested powerset expansion without
    /// cardinality filter. With |inner| = n, unfiltered base has 2^n
    /// elements and enumeration produces 2^(2^n) solutions. Only tractable
    /// for small n.
    const MAX_NESTED_INNER_UNFILTERED: usize = 4;

    /// Expand a BMC quantifier over a nested powerset domain:
    /// `\Q T \in SUBSET(SUBSET S) : P(T)`.
    ///
    /// Uses [`NestedPowersetEncoder`] to enumerate solutions efficiently:
    ///
    /// 1. Extract the inner universe S (e.g., Nodes = {1..5})
    /// 2. Try to detect a cardinality filter from the body:
    ///    `\A e \in T : Cardinality(e) = K` reduces base to C(|S|, K) elements
    /// 3. Compute filtered base elements (k-subsets) or all subsets
    /// 4. Use `NestedPowersetEncoder` to enumerate all valid outer sets
    /// 5. For each outer set, substitute a concrete `SetEnum(SetEnum(...))`
    ///    value into the body and translate
    ///
    /// For SpanTreeTest5Nodes: S = {1..5}, K = 2, base = C(5,2) = 10 edges,
    /// solutions = 2^10 = 1024 edge sets.
    fn expand_bmc_nested_powerset_quantifier(
        &mut self,
        bound: &BoundVar,
        inner_base: &Spanned<Expr>,
        body: &Spanned<Expr>,
        is_forall: bool,
    ) -> AYResult<Term> {
        use crate::translate::nested_powerset::{NestedPowersetConfig, NestedPowersetEncoder};

        let var_name = &bound.name.node;

        // Step 1: Extract inner universe as concrete integers
        let inner_universe = self.extract_universe_ints(inner_base)?;

        // Step 2: Try to detect cardinality filter from body
        // Pattern: ... /\ \A e \in Var : Cardinality(e) = K /\ ...
        let cardinality_k = Self::detect_cardinality_filter(body, var_name);

        // Step 3: Compute base elements
        let base_elements = if let Some(k) = cardinality_k {
            // Filtered: only k-element subsets of inner universe
            crate::translate::nested_powerset::k_subsets(&inner_universe, k)
        } else if inner_universe.len() <= Self::MAX_NESTED_INNER_UNFILTERED {
            // Small enough for all subsets (2^n base elements)
            Self::all_subsets_as_base_elements(&inner_universe)
        } else {
            return Err(AYError::UnsupportedOp(format!(
                "SUBSET(SUBSET S) with |S| = {} and no cardinality filter is too large; \
                 would produce 2^{} base elements. Add a Cardinality filter or reduce S.",
                inner_universe.len(),
                inner_universe.len()
            )));
        };

        if base_elements.is_empty() {
            // Only one outer set: the empty set
            return self.bmc_substitute_and_translate(body, var_name, &Expr::SetEnum(Vec::new()));
        }

        // Step 4: Enumerate all outer sets using NestedPowersetEncoder
        let mut encoder = NestedPowersetEncoder::new(base_elements.clone())?;
        let config = NestedPowersetConfig {
            max_solutions: 2_000_000,
            solve_timeout: Some(std::time::Duration::from_mins(1)),
        };
        let solutions = encoder.enumerate_all(&config)?;

        if solutions.solutions.is_empty() {
            return Ok(self.solver.bool_const(is_forall));
        }

        // Step 5: For each solution, build a concrete Expr and substitute
        let mut results = Vec::with_capacity(solutions.solutions.len());

        for solution in &solutions.solutions {
            // Convert solution (list of BaseElements) to Expr::SetEnum(...)
            let set_of_sets_expr = Self::base_elements_to_set_enum(solution);
            let body_term = self.bmc_substitute_and_translate(body, var_name, &set_of_sets_expr)?;
            results.push(body_term);
        }

        self.combine_bool_terms(&results, is_forall)
    }

    /// Extract the inner universe as concrete i64 integers.
    ///
    /// Similar to `extract_universe_from_exprs` but returns raw i64 values
    /// instead of solver terms.
    fn extract_universe_ints(&self, expr: &Spanned<Expr>) -> AYResult<Vec<i64>> {
        let mut values = std::collections::BTreeSet::new();
        Self::collect_universe_ints_static(expr, &mut values)?;
        Ok(values.into_iter().collect())
    }

    /// Recursively collect integer values from a set expression (static).
    fn collect_universe_ints_static(
        expr: &Spanned<Expr>,
        values: &mut std::collections::BTreeSet<i64>,
    ) -> AYResult<()> {
        match &expr.node {
            Expr::SetEnum(elements) => {
                for e in elements {
                    if let Expr::Int(n) = &e.node {
                        let v: i64 = n.try_into().map_err(|_| {
                            AYError::IntegerOverflow("set element too large for i64".to_string())
                        })?;
                        values.insert(v);
                    }
                }
            }
            Expr::Range(lo, hi) => {
                let lo_val = Self::extract_int_literal(lo)?;
                let hi_val = Self::extract_int_literal(hi)?;
                for v in lo_val..=hi_val {
                    values.insert(v);
                }
            }
            Expr::Powerset(base) => {
                Self::collect_universe_ints_static(base, values)?;
            }
            Expr::Union(left, right)
            | Expr::Intersect(left, right)
            | Expr::SetMinus(left, right) => {
                Self::collect_universe_ints_static(left, values)?;
                Self::collect_universe_ints_static(right, values)?;
            }
            // Ident: may reference a constant set, but we cannot resolve
            // without runtime context. Try to handle common patterns.
            Expr::Ident(name, _) | Expr::StateVar(name, ..) => {
                // If we had a way to look up constant definitions, we would
                // do so here. For now, this is unresolvable at BMC time.
                return Err(AYError::UnsupportedOp(format!(
                    "cannot extract universe from identifier '{}' in nested \
                     powerset context",
                    name
                )));
            }
            _ => {}
        }
        Ok(())
    }

    /// Extract an integer literal from an expression.
    fn extract_int_literal(expr: &Spanned<Expr>) -> AYResult<i64> {
        match &expr.node {
            Expr::Int(n) => n.try_into().map_err(|_| {
                AYError::IntegerOverflow("integer literal too large for i64".to_string())
            }),
            Expr::Neg(inner) => {
                let v = Self::extract_int_literal(inner)?;
                Ok(-v)
            }
            _ => Err(AYError::UnsupportedOp(format!(
                "expected integer literal, got {:?}",
                std::mem::discriminant(&expr.node)
            ))),
        }
    }

    /// Detect a cardinality filter pattern in the body of a nested powerset
    /// quantifier.
    ///
    /// Looks for the pattern:
    /// `... /\ \A e \in Var : Cardinality(e) = K /\ ...`
    ///
    /// where `Var` is the bound variable of the outer quantifier.
    /// Returns `Some(K)` if found, `None` otherwise.
    pub(crate) fn detect_cardinality_filter(body: &Spanned<Expr>, var_name: &str) -> Option<usize> {
        Self::detect_cardinality_in_expr(&body.node, var_name)
    }

    /// Recursively search for a cardinality filter in an expression tree.
    fn detect_cardinality_in_expr(expr: &Expr, var_name: &str) -> Option<usize> {
        match expr {
            // Conjunction: check both sides
            Expr::And(left, right) => Self::detect_cardinality_in_expr(&left.node, var_name)
                .or_else(|| Self::detect_cardinality_in_expr(&right.node, var_name)),
            // \A e \in Var : Cardinality(e) = K
            Expr::Forall(bounds, inner_body) => {
                if bounds.len() == 1 {
                    let bound = &bounds[0];
                    // Check if domain is our variable
                    if let Some(domain) = &bound.domain {
                        let domain_is_var = matches!(
                            &domain.node,
                            Expr::Ident(name, _) | Expr::StateVar(name, ..)
                            if name == var_name
                        );
                        if domain_is_var {
                            // Check if body is Cardinality(e) = K
                            return Self::extract_cardinality_eq(
                                &inner_body.node,
                                &bound.name.node,
                            );
                        }
                    }
                }
                None
            }
            // Also check inside Implies (from SetFilter desugaring of \A)
            Expr::Implies(_, consequent) => {
                Self::detect_cardinality_in_expr(&consequent.node, var_name)
            }
            _ => None,
        }
    }

    /// Extract K from `Cardinality(var) = K` pattern.
    fn extract_cardinality_eq(expr: &Expr, elem_var: &str) -> Option<usize> {
        if let Expr::Eq(left, right) = expr {
            // Try both orders: Cardinality(e) = K and K = Cardinality(e)
            if let Some(k) = Self::match_cardinality_eq_inner(&left.node, &right.node, elem_var) {
                return Some(k);
            }
            if let Some(k) = Self::match_cardinality_eq_inner(&right.node, &left.node, elem_var) {
                return Some(k);
            }
        }
        // Also handle conjunction: Cardinality(e) = K /\ more_stuff
        if let Expr::And(left, right) = expr {
            if let Some(k) = Self::extract_cardinality_eq(&left.node, elem_var) {
                return Some(k);
            }
            return Self::extract_cardinality_eq(&right.node, elem_var);
        }
        None
    }

    /// Match `Cardinality(var) = int_literal` pattern.
    fn match_cardinality_eq_inner(
        maybe_card: &Expr,
        maybe_k: &Expr,
        elem_var: &str,
    ) -> Option<usize> {
        // Check if left side is Cardinality(var)
        if let Expr::Apply(op, args) = maybe_card {
            if let Expr::Ident(name, _) = &op.node {
                if name == "Cardinality" && args.len() == 1 {
                    // Check if argument is our element variable
                    let arg_is_var = matches!(
                        &args[0].node,
                        Expr::Ident(n, _) | Expr::StateVar(n, ..)
                        if n == elem_var
                    );
                    if arg_is_var {
                        // Extract K from the other side
                        if let Expr::Int(k) = maybe_k {
                            if let Ok(k_val) = usize::try_from(k) {
                                return Some(k_val);
                            }
                        }
                    }
                }
            }
        }
        None
    }

    /// Compute all subsets of a set as BaseElements (for small inner sets).
    fn all_subsets_as_base_elements(elements: &[i64]) -> Vec<BaseElement> {
        let n = elements.len();
        let num_subsets = 1usize << n;
        let mut result = Vec::with_capacity(num_subsets);
        for mask in 0..num_subsets {
            let mut members = Vec::new();
            for (i, &elem) in elements.iter().enumerate() {
                if mask & (1 << i) != 0 {
                    members.push(elem);
                }
            }
            result.push(BaseElement { members });
        }
        result
    }

    /// Convert a list of BaseElements (a solution from NestedPowersetEncoder)
    /// to an `Expr::SetEnum` of `Expr::SetEnum`s.
    ///
    /// Each BaseElement becomes a `SetEnum` of its members as `Int` literals.
    /// The outer list becomes a `SetEnum` of those inner sets.
    fn base_elements_to_set_enum(elements: &[BaseElement]) -> Expr {
        let inner_sets: Vec<Spanned<Expr>> = elements
            .iter()
            .map(|elem| {
                let members: Vec<Spanned<Expr>> = elem
                    .members
                    .iter()
                    .map(|&m| Spanned::dummy(Expr::Int(BigInt::from(m))))
                    .collect();
                Spanned::dummy(Expr::SetEnum(members))
            })
            .collect();
        Expr::SetEnum(inner_sets)
    }

    // --- BMC expansion helpers ---

    /// Combine a list of Bool terms into a conjunction (is_forall=true) or
    /// disjunction (is_forall=false) using a balanced binary tree.
    ///
    /// Optimizations applied:
    /// - **Empty list**: returns the identity element (TRUE for AND, FALSE for OR).
    /// - **Singleton**: returns the single term directly.
    /// - **Constant folding**: drops identity elements; short-circuits on
    ///   absorbing element (FALSE in AND, TRUE in OR).
    /// - **Balanced tree**: splits remaining terms at the midpoint so the
    ///   solver sees a balanced tree instead of a left-skewed chain. This
    ///   helps DPLL unit propagation and clause learning.
    pub(super) fn combine_bool_terms(&mut self, terms: &[Term], is_forall: bool) -> AYResult<Term> {
        let identity = self.solver.bool_const(is_forall);
        let absorb = self.solver.bool_const(!is_forall);

        // Constant-fold: remove identity elements, short-circuit on absorb.
        let mut filtered = Vec::with_capacity(terms.len());
        for &t in terms {
            if t == absorb {
                return Ok(absorb);
            }
            if t != identity {
                filtered.push(t);
            }
        }

        match filtered.len() {
            0 => Ok(identity),
            1 => Ok(filtered[0]),
            _ => self.combine_balanced(&filtered, is_forall),
        }
    }

    /// Recursively build a balanced binary tree of AND/OR from a non-empty slice.
    fn combine_balanced(&mut self, terms: &[Term], is_forall: bool) -> AYResult<Term> {
        debug_assert!(!terms.is_empty());
        if terms.len() == 1 {
            return Ok(terms[0]);
        }
        if terms.len() == 2 {
            return if is_forall {
                Ok(self.solver.try_and(terms[0], terms[1])?)
            } else {
                Ok(self.solver.try_or(terms[0], terms[1])?)
            };
        }
        let mid = terms.len() / 2;
        let left = self.combine_balanced(&terms[..mid], is_forall)?;
        let right = self.combine_balanced(&terms[mid..], is_forall)?;
        if is_forall {
            Ok(self.solver.try_and(left, right)?)
        } else {
            Ok(self.solver.try_or(left, right)?)
        }
    }

    /// Expand a BMC quantifier over BOOLEAN domain.
    fn expand_bmc_boolean_quantifier(
        &mut self,
        bound: &BoundVar,
        body: &Spanned<Expr>,
        is_forall: bool,
    ) -> AYResult<Term> {
        let var_name = &bound.name.node;

        let body_true = self.bmc_substitute_and_translate(body, var_name, &Expr::Bool(true))?;
        let body_false = self.bmc_substitute_and_translate(body, var_name, &Expr::Bool(false))?;

        self.combine_bool_terms(&[body_true, body_false], is_forall)
    }

    /// Expand a BMC quantifier over a finite set enumeration.
    fn expand_bmc_set_enum_quantifier(
        &mut self,
        bound: &BoundVar,
        elements: &[Spanned<Expr>],
        body: &Spanned<Expr>,
        is_forall: bool,
    ) -> AYResult<Term> {
        if elements.is_empty() {
            return Ok(self.solver.bool_const(is_forall));
        }

        // Singleton fast path: skip combine machinery entirely.
        if elements.len() == 1 {
            return self.bmc_substitute_and_translate(body, &bound.name.node, &elements[0].node);
        }

        let var_name = &bound.name.node;
        let mut results = Vec::with_capacity(elements.len());

        for elem in elements {
            let substituted = self.bmc_substitute_and_translate(body, var_name, &elem.node)?;
            results.push(substituted);
        }

        self.combine_bool_terms(&results, is_forall)
    }

    /// Expand a BMC quantifier over an integer range.
    fn expand_bmc_range_quantifier(
        &mut self,
        bound: &BoundVar,
        lo: &BigInt,
        hi: &BigInt,
        body: &Spanned<Expr>,
        is_forall: bool,
    ) -> AYResult<Term> {
        if lo > hi {
            return Ok(self.solver.bool_const(is_forall));
        }

        // Singleton fast path.
        if lo == hi {
            return self.bmc_substitute_and_translate(
                body,
                &bound.name.node,
                &Expr::Int(lo.clone()),
            );
        }

        let var_name = &bound.name.node;
        let mut results = Vec::new();

        let mut i = lo.clone();
        while &i <= hi {
            let substituted =
                self.bmc_substitute_and_translate(body, var_name, &Expr::Int(i.clone()))?;
            results.push(substituted);
            i += 1;
        }

        self.combine_bool_terms(&results, is_forall)
    }

    // --- BMC Skolemization helpers ---

    /// Skolemize `\E x \in {e1, ..., en} : P(x)` in BMC context.
    ///
    /// Introduces a fresh Int constant and asserts membership + predicate.
    fn skolemize_bmc_exists_set_enum(
        &mut self,
        bound: &BoundVar,
        elements: &[Spanned<Expr>],
        body: &Spanned<Expr>,
    ) -> AYResult<Term> {
        let var_name = &bound.name.node;

        // Create fresh Skolem constant
        let sk_name = self.fresh_bmc_name(&format!("sk_{var_name}"));
        let sk_term = self.solver.declare_const(&sk_name, Sort::Int);

        // Assert membership: sk = e1 \/ sk = e2 \/ ...
        // Uses balanced tree for large membership disjunctions.
        let mut membership_terms = Vec::with_capacity(elements.len());
        for elem in elements {
            let elem_term = self.translate_int(elem)?;
            let eq = self.solver.try_eq(sk_term, elem_term)?;
            membership_terms.push(eq);
        }
        let membership = self.combine_bool_terms(&membership_terms, false)?;
        self.assert(membership);

        // Register the Skolem constant as a variable at all steps
        // (it is a constant, so same value at all steps)
        self.vars.insert(
            sk_name.clone(),
            super::BmcVarInfo {
                sort: TlaSort::Int,
                terms: vec![sk_term; self.bound_k + 1],
            },
        );

        // Assert P[sk/x]
        let sk_expr = Expr::Ident(sk_name, NameId::INVALID);
        let pred = self.bmc_substitute_and_translate(body, var_name, &sk_expr)?;
        self.assert(pred);

        Ok(self.solver.bool_const(true))
    }

    /// Skolemize `\E x \in lo..hi : P(x)` in BMC context.
    ///
    /// Introduces a fresh Int constant and asserts range bounds + predicate.
    fn skolemize_bmc_exists_range(
        &mut self,
        bound: &BoundVar,
        lo: &BigInt,
        hi: &BigInt,
        body: &Spanned<Expr>,
    ) -> AYResult<Term> {
        let var_name = &bound.name.node;

        let sk_name = self.fresh_bmc_name(&format!("sk_{var_name}"));
        let sk_term = self.solver.declare_const(&sk_name, Sort::Int);

        // Assert: lo <= sk /\ sk <= hi
        let lo_i64 = i64::try_from(lo)
            .map_err(|_| AYError::IntegerOverflow(format!("range lower bound too large: {lo}")))?;
        let hi_i64 = i64::try_from(hi)
            .map_err(|_| AYError::IntegerOverflow(format!("range upper bound too large: {hi}")))?;
        let lo_term = self.solver.int_const(lo_i64);
        let hi_term = self.solver.int_const(hi_i64);
        let ge_lo = self.solver.try_ge(sk_term, lo_term)?;
        let le_hi = self.solver.try_le(sk_term, hi_term)?;
        self.assert(ge_lo);
        self.assert(le_hi);

        // Register the Skolem constant
        self.vars.insert(
            sk_name.clone(),
            super::BmcVarInfo {
                sort: TlaSort::Int,
                terms: vec![sk_term; self.bound_k + 1],
            },
        );

        // Assert P[sk/x]
        let sk_expr = Expr::Ident(sk_name, NameId::INVALID);
        let pred = self.bmc_substitute_and_translate(body, var_name, &sk_expr)?;
        self.assert(pred);

        Ok(self.solver.bool_const(true))
    }

    // --- CHOOSE translation ---

    /// Translate `CHOOSE x \in S : P(x)` via Skolemization.
    ///
    /// CHOOSE picks an arbitrary element from `S` satisfying `P`. In SMT
    /// (QF_LIA), this becomes a fresh Skolem constant `c` with assertions:
    ///   - `c \in S`       (domain membership)
    ///   - `P[c/x]`        (predicate holds)
    ///
    /// The Skolem constant is returned as an Int term that can be used in
    /// both integer and boolean contexts.
    ///
    /// Domain types supported:
    ///   - `SetEnum({e1, ..., en})`: membership via disjunction `c = e1 \/ ... \/ c = en`
    ///   - `Range(lo..hi)`: membership via bounds `lo <= c /\ c <= hi`
    ///   - `BOOLEAN`: membership via `c = 0 \/ c = 1` (Bool encoded as Int)
    ///   - `SetFilter({x \in S : P(x)})`: rewrite to inner domain + conjoined predicate
    ///
    /// If the body is `TRUE` (unbounded CHOOSE), only the domain constraint
    /// is asserted.
    pub(super) fn translate_choose_bmc(
        &mut self,
        bound: &BoundVar,
        body: &Spanned<Expr>,
    ) -> AYResult<Term> {
        let domain = match &bound.domain {
            Some(d) => d,
            None => {
                return Err(AYError::UnsupportedOp(
                    "BMC CHOOSE requires bounded domain".to_string(),
                ))
            }
        };

        let var_name = &bound.name.node;

        // BOOLEAN domain: Skolem constant in {0, 1}
        if let Expr::Ident(name, _) = &domain.node {
            if name == "BOOLEAN" {
                return self.skolemize_choose_boolean(var_name, body);
            }
        }

        // Finite set enumeration {e1, ..., en}
        if let Expr::SetEnum(elements) = &domain.node {
            return self.skolemize_choose_set_enum(var_name, elements, body);
        }

        // Integer range lo..hi
        if let Expr::Range(lo, hi) = &domain.node {
            if let (Expr::Int(lo_val), Expr::Int(hi_val)) = (&lo.node, &hi.node) {
                return self.skolemize_choose_range(var_name, lo_val, hi_val, body);
            }
        }

        // SetFilter: CHOOSE y \in {z \in S : P(z)} : body
        // Rewrite to CHOOSE y \in S : P[y/z] /\ body
        if let Expr::SetFilter(filter_bound, pred) = &domain.node {
            if let Some(inner_domain) = &filter_bound.domain {
                let filter_var = &filter_bound.name.node;

                let replacement =
                    Spanned::new(Expr::Ident(var_name.clone(), NameId::INVALID), pred.span);
                let mut sub = SubstituteExpr {
                    subs: HashMap::from([(filter_var.as_str(), &replacement)]),
                    span_policy: SpanPolicy::Preserve,
                };
                let pred_spanned = sub.fold_expr(*pred.clone());

                // New body: filter_pred /\ original_body
                let new_body = Expr::And(Box::new(pred_spanned), Box::new(body.clone()));
                let new_body_spanned = Spanned::new(new_body, body.span);

                let new_bound = BoundVar {
                    name: bound.name.clone(),
                    domain: Some(inner_domain.clone()),
                    pattern: bound.pattern.clone(),
                };

                return self.translate_choose_bmc(&new_bound, &new_body_spanned);
            }
        }

        Err(AYError::UnsupportedOp(
            "BMC CHOOSE over this domain type not supported".to_string(),
        ))
    }

    /// Skolemize `CHOOSE x \in BOOLEAN : P(x)`.
    ///
    /// Creates a fresh Int constant constrained to {0, 1} (Bool-as-Int encoding).
    fn skolemize_choose_boolean(&mut self, var_name: &str, body: &Spanned<Expr>) -> AYResult<Term> {
        let sk_name = self.fresh_bmc_name(&format!("choose_{var_name}"));
        let sk_term = self.solver.declare_const(&sk_name, Sort::Int);

        // Assert: sk = 0 \/ sk = 1
        let zero = self.solver.int_const(0);
        let one = self.solver.int_const(1);
        let eq_zero = self.solver.try_eq(sk_term, zero)?;
        let eq_one = self.solver.try_eq(sk_term, one)?;
        let membership = self.solver.try_or(eq_zero, eq_one)?;
        self.assert(membership);

        // Register Skolem constant so variable lookup works during body translation
        self.vars.insert(
            sk_name.clone(),
            super::BmcVarInfo {
                sort: TlaSort::Int,
                terms: vec![sk_term; self.bound_k + 1],
            },
        );

        // Assert P[sk/x] (skip if body is TRUE — unbounded CHOOSE)
        if !matches!(body.node, Expr::Bool(true)) {
            let sk_expr = Expr::Ident(sk_name, NameId::INVALID);
            let pred = self.bmc_substitute_and_translate(body, var_name, &sk_expr)?;
            self.assert(pred);
        }

        Ok(sk_term)
    }

    /// Skolemize `CHOOSE x \in {e1, ..., en} : P(x)`.
    ///
    /// Creates a fresh Int constant, asserts membership via disjunction,
    /// and asserts the predicate body.
    fn skolemize_choose_set_enum(
        &mut self,
        var_name: &str,
        elements: &[Spanned<Expr>],
        body: &Spanned<Expr>,
    ) -> AYResult<Term> {
        if elements.is_empty() {
            return Err(AYError::UnsupportedOp(
                "CHOOSE over empty set is undefined".to_string(),
            ));
        }

        let sk_name = self.fresh_bmc_name(&format!("choose_{var_name}"));
        let sk_term = self.solver.declare_const(&sk_name, Sort::Int);

        // Assert membership: sk = e1 \/ sk = e2 \/ ...
        // Uses balanced tree for large membership disjunctions.
        let mut membership_terms = Vec::with_capacity(elements.len());
        for elem in elements {
            let elem_term = self.translate_int(elem)?;
            let eq = self.solver.try_eq(sk_term, elem_term)?;
            membership_terms.push(eq);
        }
        let membership = self.combine_bool_terms(&membership_terms, false)?;
        self.assert(membership);

        // Register Skolem constant
        self.vars.insert(
            sk_name.clone(),
            super::BmcVarInfo {
                sort: TlaSort::Int,
                terms: vec![sk_term; self.bound_k + 1],
            },
        );

        // Assert P[sk/x] (skip if body is TRUE — unbounded CHOOSE)
        if !matches!(body.node, Expr::Bool(true)) {
            let sk_expr = Expr::Ident(sk_name, NameId::INVALID);
            let pred = self.bmc_substitute_and_translate(body, var_name, &sk_expr)?;
            self.assert(pred);
        }

        Ok(sk_term)
    }

    /// Skolemize `CHOOSE x \in lo..hi : P(x)`.
    ///
    /// Creates a fresh Int constant, asserts range bounds, and asserts
    /// the predicate body.
    fn skolemize_choose_range(
        &mut self,
        var_name: &str,
        lo: &BigInt,
        hi: &BigInt,
        body: &Spanned<Expr>,
    ) -> AYResult<Term> {
        let sk_name = self.fresh_bmc_name(&format!("choose_{var_name}"));
        let sk_term = self.solver.declare_const(&sk_name, Sort::Int);

        // Assert: lo <= sk /\ sk <= hi
        let lo_i64 = i64::try_from(lo).map_err(|_| {
            AYError::IntegerOverflow(format!("CHOOSE range lower bound too large: {lo}"))
        })?;
        let hi_i64 = i64::try_from(hi).map_err(|_| {
            AYError::IntegerOverflow(format!("CHOOSE range upper bound too large: {hi}"))
        })?;
        let lo_term = self.solver.int_const(lo_i64);
        let hi_term = self.solver.int_const(hi_i64);
        let ge_lo = self.solver.try_ge(sk_term, lo_term)?;
        let le_hi = self.solver.try_le(sk_term, hi_term)?;
        self.assert(ge_lo);
        self.assert(le_hi);

        // Register Skolem constant
        self.vars.insert(
            sk_name.clone(),
            super::BmcVarInfo {
                sort: TlaSort::Int,
                terms: vec![sk_term; self.bound_k + 1],
            },
        );

        // Assert P[sk/x] (skip if body is TRUE — unbounded CHOOSE)
        if !matches!(body.node, Expr::Bool(true)) {
            let sk_expr = Expr::Ident(sk_name, NameId::INVALID);
            let pred = self.bmc_substitute_and_translate(body, var_name, &sk_expr)?;
            self.assert(pred);
        }

        Ok(sk_term)
    }

    // --- Shared helpers ---

    /// Generate a fresh variable name for BMC Skolem constants.
    fn fresh_bmc_name(&mut self, prefix: &str) -> String {
        let id = self.aux_var_counter;
        self.aux_var_counter += 1;
        format!("__bmc_{prefix}_{id}")
    }

    /// Substitute a value for a bound variable and translate to Bool.
    ///
    /// This is the BMC equivalent of `AYTranslator::substitute_and_translate`.
    /// It performs AST-level substitution then translates via BMC dispatch.
    fn bmc_substitute_and_translate(
        &mut self,
        expr: &Spanned<Expr>,
        var_name: &str,
        replacement: &Expr,
    ) -> AYResult<Term> {
        let replacement_spanned = Spanned::new(replacement.clone(), expr.span);
        let mut sub = SubstituteExpr {
            subs: HashMap::from([(var_name, &replacement_spanned)]),
            span_policy: SpanPolicy::Preserve,
        };
        let substituted = sub.fold_expr(expr.clone());
        // Also substitute StateVar references (same pattern as AYTranslator)
        let mut state_var_sub = BmcSubstituteStateVar {
            var_name,
            replacement,
        };
        let substituted = state_var_sub.fold_expr(substituted);
        self.translate_bool(&substituted)
    }
}

/// Helper: substitute state variables with the bound variable name.
///
/// Mirrors `SubstituteStateVar` in `translate::quantifier`.
struct BmcSubstituteStateVar<'a> {
    var_name: &'a str,
    replacement: &'a Expr,
}

impl ExprFold for BmcSubstituteStateVar<'_> {
    fn fold_state_var(
        &mut self,
        name: String,
        idx: u16,
        id: tla_core::name_intern::NameId,
    ) -> Expr {
        if name == self.var_name {
            self.replacement.clone()
        } else {
            Expr::StateVar(name, idx, id)
        }
    }
}

/// Maximum size of a concrete integer set enumerated for a quantifier domain.
const MAX_ENUM_INT_SET: usize = 4096;

/// Enumerate a set expression to its concrete ordered integer elements, when it
/// is built entirely from integer-literal ranges/enums combined with `\`, `\cup`
/// and `\cap`. Returns `None` for any non-literal or oversize component.
///
/// Used so a quantifier over `1..5 \ {3}` (and similar) can be expanded by
/// reusing the existing concrete set-enum expansion. Sound: the element set is
/// computed exactly.
fn enumerate_concrete_int_set(expr: &Spanned<Expr>) -> Option<Vec<i64>> {
    use std::collections::BTreeSet;

    fn collect(expr: &Spanned<Expr>, out: &mut BTreeSet<i64>) -> Option<()> {
        match &expr.node {
            Expr::Range(lo, hi) => {
                let lo = lit_i64(lo)?;
                let hi = lit_i64(hi)?;
                if hi >= lo && (hi - lo) as usize >= MAX_ENUM_INT_SET {
                    return None;
                }
                for v in lo..=hi {
                    out.insert(v);
                }
                Some(())
            }
            Expr::SetEnum(elems) => {
                for e in elems {
                    out.insert(lit_i64(e)?);
                }
                Some(())
            }
            Expr::Union(a, b) => {
                collect(a, out)?;
                collect(b, out)
            }
            Expr::Intersect(a, b) => {
                let mut sa = BTreeSet::new();
                let mut sb = BTreeSet::new();
                collect(a, &mut sa)?;
                collect(b, &mut sb)?;
                for v in sa.intersection(&sb) {
                    out.insert(*v);
                }
                Some(())
            }
            Expr::SetMinus(a, b) => {
                let mut sa = BTreeSet::new();
                let mut sb = BTreeSet::new();
                collect(a, &mut sa)?;
                collect(b, &mut sb)?;
                for v in sa.difference(&sb) {
                    out.insert(*v);
                }
                Some(())
            }
            _ => None,
        }
    }

    fn lit_i64(expr: &Spanned<Expr>) -> Option<i64> {
        match &expr.node {
            Expr::Int(n) => i64::try_from(n).ok(),
            Expr::Neg(a) => lit_i64(a)?.checked_neg(),
            _ => None,
        }
    }

    let mut set = BTreeSet::new();
    collect(expr, &mut set)?;
    if set.len() > MAX_ENUM_INT_SET {
        return None;
    }
    Some(set.into_iter().collect())
}
