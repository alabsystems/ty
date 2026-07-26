// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! [`TranslateExpr`] trait implementation for [`BmcTranslator`].
//!
//! This bridges the shared `dispatch_translate_{bool,int}` arms in `tla_core`
//! to the BMC-specific translation methods in the parent module.
//!
//! Compound type encoders (set operations, function EXCEPT/DOMAIN, function
//! construction, sequence ops, and Cardinality) are wired into the dispatch so
//! that the BmcTranslator automatically routes compound type operations to the
//! appropriate encoder. Part of #3778.

use ay_dpll::api::Term;
use tla_core::ast::Expr;
use tla_core::{dispatch_translate_bool, dispatch_translate_int, Spanned, TranslateExpr};

use crate::error::{AYError, AYResult};

use super::compound_dispatch::SetBinOp;
use super::BmcTranslator;
use crate::TlaSort;

/// Extract the base function-variable name from a (possibly primed) function
/// reference (`f`, `f'`). Returns `None` for non-variable bases. Used to
/// upgrade a function variable's key sort to `String` (#5).
fn func_var_base_name(expr: &Spanned<Expr>) -> Option<String> {
    match &expr.node {
        Expr::Ident(name, _) | Expr::StateVar(name, ..) => Some(name.clone()),
        Expr::Prime(inner) => match &inner.node {
            Expr::Ident(name, _) | Expr::StateVar(name, ..) => Some(name.clone()),
            _ => None,
        },
        _ => None,
    }
}

impl TranslateExpr for BmcTranslator {
    type Bool = Term;
    type Int = Term;
    type Error = AYError;

    fn bool_const(&mut self, val: bool) -> Term {
        self.solver.bool_const(val)
    }

    fn int_const(&mut self, val: i64) -> AYResult<Term> {
        Ok(self.solver.int_const(val))
    }

    fn lookup_bool_var(&mut self, name: &str) -> AYResult<Term> {
        let term = self.get_var_at_step(name, self.current_step)?;
        if let Some(info) = self.vars.get(name) {
            if info.sort != TlaSort::Bool {
                return Err(AYError::TypeMismatch {
                    name: name.to_string(),
                    expected: "Bool".to_string(),
                    actual: format!("{}", info.sort),
                });
            }
        }
        Ok(term)
    }

    fn lookup_int_var(&mut self, name: &str) -> AYResult<Term> {
        let term = self.get_var_at_step(name, self.current_step)?;
        if let Some(info) = self.vars.get(name) {
            if info.sort != TlaSort::Int {
                return Err(AYError::TypeMismatch {
                    name: name.to_string(),
                    expected: "Int".to_string(),
                    actual: format!("{}", info.sort),
                });
            }
        }
        Ok(term)
    }

    fn and(&mut self, lhs: Term, rhs: Term) -> Term {
        self.solver
            .try_and(lhs, rhs)
            .expect("invariant: and requires Bool-sorted terms")
    }

    fn or(&mut self, lhs: Term, rhs: Term) -> Term {
        self.solver
            .try_or(lhs, rhs)
            .expect("invariant: or requires Bool-sorted terms")
    }

    fn not(&mut self, expr: Term) -> Term {
        self.solver
            .try_not(expr)
            .expect("invariant: not requires Bool-sorted term")
    }

    fn implies(&mut self, lhs: Term, rhs: Term) -> Term {
        self.solver
            .try_implies(lhs, rhs)
            .expect("invariant: implies requires Bool-sorted terms")
    }

    // iff() uses default from TranslateExpr: (a => b) /\ (b => a)

    fn lt(&mut self, lhs: Term, rhs: Term) -> Term {
        self.solver
            .try_lt(lhs, rhs)
            .expect("invariant: lt requires Int-sorted terms")
    }

    fn le(&mut self, lhs: Term, rhs: Term) -> Term {
        self.solver
            .try_le(lhs, rhs)
            .expect("invariant: le requires Int-sorted terms")
    }

    fn gt(&mut self, lhs: Term, rhs: Term) -> Term {
        self.solver
            .try_gt(lhs, rhs)
            .expect("invariant: gt requires Int-sorted terms")
    }

    fn ge(&mut self, lhs: Term, rhs: Term) -> Term {
        self.solver
            .try_ge(lhs, rhs)
            .expect("invariant: ge requires Int-sorted terms")
    }

    fn add(&mut self, lhs: Term, rhs: Term) -> Term {
        self.solver
            .try_add(lhs, rhs)
            .expect("invariant: add requires Int-sorted terms")
    }

    fn sub(&mut self, lhs: Term, rhs: Term) -> Term {
        self.solver
            .try_sub(lhs, rhs)
            .expect("invariant: sub requires Int-sorted terms")
    }

    fn mul(&mut self, lhs: Term, rhs: Term) -> AYResult<Term> {
        Ok(self.solver.try_mul(lhs, rhs)?)
    }

    fn neg(&mut self, expr: Term) -> Term {
        // BMC uses 0 - x for negation (QF_LIA compatible)
        let zero = self.solver.int_const(0);
        self.solver
            .try_sub(zero, expr)
            .expect("invariant: sub requires Int-sorted terms")
    }

    fn div(&mut self, _lhs: Term, _rhs: Term) -> AYResult<Term> {
        // BMC handles div via translate_int_extended (needs AST-level access
        // to check for constant divisors and do QF_LIA linearization).
        // This path should not be reached -- the extended hook intercepts first.
        Err(AYError::UntranslatableExpr(
            "BMC div requires constant divisor (handled by extension hook)".to_string(),
        ))
    }

    fn modulo(&mut self, _lhs: Term, _rhs: Term) -> AYResult<Term> {
        // Same as div -- BMC needs AST-level access for linearization.
        Err(AYError::UntranslatableExpr(
            "BMC modulo requires constant divisor (handled by extension hook)".to_string(),
        ))
    }

    fn ite_bool(&mut self, cond: Term, then_b: Term, else_b: Term) -> Term {
        self.solver
            .try_ite(cond, then_b, else_b)
            .expect("invariant: ite requires Bool cond, matching then/else sorts")
    }

    fn ite_int(&mut self, cond: Term, then_i: Term, else_i: Term) -> Term {
        self.solver
            .try_ite(cond, then_i, else_i)
            .expect("invariant: ite requires Bool cond, matching then/else sorts")
    }

    fn translate_eq(&mut self, left: &Spanned<Expr>, right: &Spanned<Expr>) -> AYResult<Term> {
        // Check for function EXCEPT equality: f' = [f EXCEPT ![a] = b]
        if let Some(result) = self.try_translate_func_except_eq(left, right) {
            return result;
        }

        // Check for function construction equality: f = [x \in S |-> e(x)]
        // Part of #3786: Function encoding in BMC translator.
        if let Some(result) = self.try_translate_func_construct_eq(left, right) {
            return result;
        }

        // Check for function variable equality: f = g (both function variables)
        // Part of #3778: Apalache parity — function equality in BMC.
        if let Some(result) = self.try_translate_func_equality(left, right) {
            return result;
        }

        // Check for record equality (Part of #3787)
        // Pattern: r' = [r EXCEPT !.a = v], r' = [a |-> e1, ...], r = r'
        if let Some(result) = self.try_translate_record_except_eq(left, right) {
            return result;
        }
        if let Some(result) = self.try_translate_record_eq(left, right) {
            return result;
        }

        // Check for tuple equality (Part of #3787)
        // Pattern: t' = <<e1, e2>>, t = t'
        if let Some(result) = self.try_translate_tuple_eq(left, right) {
            return result;
        }

        // Check for sequence equality (Part of #3793)
        // Pattern: s' = Tail(s), s' = Append(s, e), s = s'
        if let Some(result) = self.try_translate_seq_eq(left, right) {
            return result;
        }

        // Check for set equality (Part of #3826): S = T where both sides
        // are set-typed expressions (SetEnum, set variable, etc.).
        // Pointwise: \A u \in universe : (select S u) = (select T u).
        if let Some(result) = self.try_translate_set_eq(left, right) {
            return result;
        }

        // TLA+ scalar kinds are disjoint even when their current SMT carriers
        // are not.  In particular, scalar Strings are injectively interned as
        // Int terms, so translating before checking kinds could make the first
        // string literal equal the ordinary integer -1_000_000_007.  Decide a
        // cross-kind equality before producing either carrier term.  `Neq`
        // goes through this method and negates the returned FALSE in the shared
        // dispatcher, so both operators inherit the same exact kind rule.
        match (self.scalar_expr_sort(left), self.scalar_expr_sort(right)) {
            (Some(left_sort), Some(right_sort)) => {
                if left_sort.canonicalized() != right_sort.canonicalized() {
                    return Ok(self.solver.bool_const(false));
                }
            }
            _ => {
                if let Some(name) = self
                    .unknown_direct_scalar_name(left)
                    .or_else(|| self.unknown_direct_scalar_name(right))
                {
                    return Err(AYError::UnknownVariable(name.to_string()));
                }
                return Err(AYError::UnsupportedOp(
                    "BMC cannot compare scalar expressions without exact kind evidence".to_string(),
                ));
            }
        }

        // STRING-SCALAR equality: `v = "lit"`, `v' = "lit"`, `v = w` where the
        // operands are string-sorted state variables and/or string literals.
        // `TlaSort::String` variables are declared as `Sort::Int` (the interned
        // representation) but `lookup_int_var`/`lookup_bool_var` reject the
        // `String` sort, so a string-VARIABLE equality has no path through the
        // int/bool fallback below. Map each string-scalar operand to its
        // interned-int term and assert integer equality. SOUND for a finite
        // closed literal universe: the string→int interning is INJECTIVE, so
        // equality and disequality are preserved bijectively (distinct literals
        // get distinct ids; a string can never alias an ordinary int). Fires
        // ONLY when BOTH operands are string scalars — never string-vs-int.
        if self.is_string_scalar(left) && self.is_string_scalar(right) {
            let l = self.string_scalar_term(left)?;
            let r = self.string_scalar_term(right)?;
            return Ok(self.solver.try_eq(l, r)?);
        }

        // Try integer equality first, then bool
        if let (Ok(l), Ok(r)) = (
            dispatch_translate_int(self, left),
            dispatch_translate_int(self, right),
        ) {
            Ok(self.solver.try_eq(l, r)?)
        } else {
            let l = dispatch_translate_bool(self, left)?;
            let r = dispatch_translate_bool(self, right)?;
            // Bool equality: (a /\ b) \/ (~a /\ ~b)
            Ok(crate::dispatch_shared::encode_bool_eq(
                &mut self.solver,
                l,
                r,
            )?)
        }
    }

    fn translate_bool_extended(&mut self, expr: &Spanned<Expr>) -> Option<AYResult<Term>> {
        match &expr.node {
            // Fix #3822: Rewrite negated quantifiers to avoid unsound Skolemization.
            // ~(\E x \in S : P(x)) == \A x \in S : ~P(x)
            // ~(\A x \in S : P(x)) == \E x \in S : ~P(x)
            Expr::Not(inner) => match &inner.node {
                Expr::Exists(bounds, body) => {
                    let negated_body = Spanned::new(Expr::Not(body.clone()), body.span);
                    Some(self.translate_bmc_quantifier(bounds, &negated_body, true))
                }
                Expr::Forall(bounds, body) => {
                    let negated_body = Spanned::new(Expr::Not(body.clone()), body.span);
                    Some(self.translate_bmc_quantifier(bounds, &negated_body, false))
                }
                _ => None,
            },
            // Quantifiers: \A x \in S : P(x), \E x \in S : P(x)
            Expr::Forall(bounds, body) => Some(self.translate_bmc_quantifier(bounds, body, true)),
            Expr::Exists(bounds, body) => Some(self.translate_bmc_quantifier(bounds, body, false)),
            Expr::Prime(inner) => {
                // Primed variable: use next step
                match &inner.node {
                    Expr::Ident(name, _) | Expr::StateVar(name, ..) => {
                        Some(self.get_var_at_step(name, self.current_step + 1))
                    }
                    _ => {
                        // Complex primed expression: temporarily shift step
                        let old_step = self.current_step;
                        self.current_step += 1;
                        let result = dispatch_translate_bool(self, inner);
                        self.current_step = old_step;
                        Some(result)
                    }
                }
            }
            Expr::In(elem, set) => Some(self.translate_membership(elem, set)),
            // x \notin S => ~(x \in S)
            Expr::NotIn(elem, set) => Some(
                self.translate_membership(elem, set)
                    .and_then(|t| self.solver.try_not(t).map_err(AYError::Solver)),
            ),
            // CHOOSE x \in S : P(x) — Skolemized, returns Int, wrap for Bool context
            Expr::Choose(bound, body) => {
                Some(self.translate_choose_bmc(bound, body).and_then(|int_term| {
                    // In Bool context, CHOOSE result is truthy if non-zero.
                    // This handles CHOOSE x \in BOOLEAN : P(x) where result
                    // is 0 or 1 (Bool-as-Int). Compare: ~(sk = 0).
                    let zero = self.solver.int_const(0);
                    let eq_zero = self.solver.try_eq(int_term, zero)?;
                    self.solver.try_not(eq_zero).map_err(AYError::Solver)
                }))
            }
            Expr::Unchanged(inner) => Some(self.translate_unchanged_expr(inner)),
            // --- Compound type dispatch (Part of #3778) ---

            // S \subseteq T: extract universe from both operands, then expand
            // pointwise \A u \in universe : (select S u) => (select T u).
            Expr::Subseteq(left, right) => Some(self.translate_subseteq_dispatch(left, right)),

            // Set enumeration {e1, ..., en}: build SMT array, return TRUE.
            Expr::SetEnum(elements) => Some(self.translate_set_enum_bool(elements)),

            // Set operations (Union, Intersect, SetMinus): build SMT array
            // terms with extracted universe, return TRUE.
            Expr::Union(left, right) => {
                Some(self.translate_set_binop_bool(left, right, SetBinOp::Union))
            }
            Expr::Intersect(left, right) => {
                Some(self.translate_set_binop_bool(left, right, SetBinOp::Intersect))
            }
            Expr::SetMinus(left, right) => {
                Some(self.translate_set_binop_bool(left, right, SetBinOp::Minus))
            }

            // SUBSET S: powerset (set of all subsets).
            // For small base sets, enumerates subsets; in Bool context,
            // returns TRUE after ensuring the base is translatable.
            Expr::Powerset(base) => Some(self.translate_powerset_bool(base)),

            // UNION S: big union of a set-of-sets (flattening).
            // Part of #3778: Apalache parity — BigUnion in BMC.
            Expr::BigUnion(inner) => Some(self.translate_big_union_bool(inner)),

            // a..b as a set expression in Bool context.
            // Part of #3778: Apalache parity — Range-as-set in BMC.
            Expr::Range(lo, hi) => {
                // translate_range_set_term builds the array; in Bool context
                // we just need to ensure it's constructed, then return TRUE.
                Some(
                    self.translate_range_set_term(lo, hi)
                        .map(|_arr| self.solver.bool_const(true)),
                )
            }

            // DOMAIN f: set-valued expression. In Bool context, return
            // an error with guidance to use membership tests instead.
            Expr::Domain(func_expr) => Some(self.translate_domain_bool_dispatch(func_expr)),

            // [f EXCEPT ![a] = b]: function or record update.
            Expr::Except(base, specs) => Some(self.translate_except_bool_dispatch(base, specs)),

            // [x \in S |-> expr]: function construction.
            Expr::FuncDef(bounds, body) => {
                Some(self.translate_func_def_bool_dispatch(bounds, body))
            }
            // Function application with Bool result via FunctionEncoder
            Expr::FuncApply(func, arg) => {
                // Try as function variable first
                if self.is_func_var_expr(func) {
                    Some(self.translate_func_apply_bmc(func, arg))
                } else if self.is_seq_var_expr(func) {
                    // Sequence indexing: s[i] returning Bool
                    // For Bool-valued sequences (uncommon), route through Int
                    // and let the solver sort-check.
                    Some(self.translate_seq_index_bmc(func, arg))
                } else if self.is_tuple_var_expr(func) {
                    // Tuple indexing: t[i] returning Bool (Part of #3787)
                    Some(self.translate_tuple_index(func, arg))
                } else {
                    None
                }
            }
            // Sequence operations returning Bool (e.g., Head of Bool seq)
            Expr::Apply(op, args) => self.translate_seq_bool_op(op, args),
            // Record construction: [a |-> e1, b |-> e2] (Part of #3787)
            Expr::Record(fields) => Some(self.translate_record_construct(fields)),
            // Record field access on a record variable (Part of #3787)
            Expr::RecordAccess(record_expr, field_name) if self.is_record_var_expr(record_expr) => {
                Some(self.translate_record_access(record_expr, &field_name.name.node))
            }
            _ => None,
        }
    }

    fn translate_int_extended(&mut self, expr: &Spanned<Expr>) -> Option<AYResult<Term>> {
        match &expr.node {
            // String literal -> its interned integer id. `TlaSort::String`
            // variables are declared as `Sort::Int`, so string literals must
            // translate to the same interned-int namespace for scalar string
            // equality to be sound. This is the scalar string path (e.g. the
            // FuncSet-enumerated permutation values in the Einstein riddle);
            // string-*keyed function domains* keep their separate native-String
            // encoding elsewhere.
            Expr::String(s) => {
                let id = self.bmc_intern_string(s);
                Some(Ok(self.solver.int_const(id)))
            }
            // CHOOSE x \in S : P(x) — Skolemized, returns Int term
            Expr::Choose(bound, body) => Some(self.translate_choose_bmc(bound, body)),
            Expr::Prime(inner) => match &inner.node {
                Expr::Ident(name, _) | Expr::StateVar(name, ..) => {
                    Some(self.get_var_at_step(name, self.current_step + 1))
                }
                _ => {
                    let old_step = self.current_step;
                    self.current_step += 1;
                    let result = dispatch_translate_int(self, inner);
                    self.current_step = old_step;
                    Some(result)
                }
            },
            Expr::Mul(left, right) => {
                // Part of #771: reject non-linear multiplication under QF_LIA
                if crate::translate::is_nonlinear_mul(left, right) {
                    return Some(Err(AYError::UnsupportedOp(
                        "BMC cannot translate non-linear integer multiplication (x * y) under QF_LIA"
                            .to_string(),
                    )));
                }
                // Linear multiplication: let shared dispatch handle via trait's mul()
                None
            }
            Expr::IntDiv(left, right) => Some(self.translate_int_div_bmc(left, right)),
            Expr::Mod(left, right) => Some(self.translate_mod_bmc(left, right)),
            // Function application with Int result via FunctionEncoder
            Expr::FuncApply(func, arg) => {
                if self.is_func_var_expr(func) {
                    Some(self.translate_func_apply_bmc(func, arg))
                } else if self.is_seq_var_expr(func) {
                    // Sequence indexing: s[i] -> (select arr i)
                    Some(self.translate_seq_index_bmc(func, arg))
                } else if self.is_tuple_var_expr(func) {
                    // Tuple indexing: t[i] -> element variable (Part of #3787)
                    Some(self.translate_tuple_index(func, arg))
                } else {
                    None
                }
            }
            // Operator applications returning Int: Cardinality, Len, Head
            Expr::Apply(op, args) => {
                // Cardinality(S) via set encoding (Part of #3778)
                if let Expr::Ident(name, _) = &op.node {
                    if name == "Cardinality" && args.len() == 1 {
                        return Some(self.translate_cardinality_int_dispatch(&args[0]));
                    }
                }
                // Sequence operations (Len, Head returning Int)
                self.translate_seq_int_op(op, args)
            }
            // Record field access on a record variable, in Int context.
            // The shared Int dispatch has no `RecordAccess` arm, and
            // `translate_eq` probes Int translation before Bool. Without
            // this extension, `r.a = 5` collapsed to the Bool fallback and
            // failed on the `Int(5)` operand. Part of #3787.
            Expr::RecordAccess(record_expr, field_name) if self.is_record_var_expr(record_expr) => {
                Some(self.translate_record_access(record_expr, &field_name.name.node))
            }
            _ => None,
        }
    }
}

impl BmcTranslator {
    /// Return a declared sequence's logical element sort.
    fn seq_element_sort(&self, name: &str) -> AYResult<TlaSort> {
        self.seq_vars
            .get(name)
            .map(|info| info.element_sort.clone().canonicalized())
            .ok_or_else(|| AYError::UnknownVariable(format!("sequence {name}")))
    }

    /// Determine a scalar expression's TLA+ value kind without translating it.
    /// Several BMC encodings use the same SMT `Int` carrier for TLA+ Int and
    /// interned String values, so kind checks must happen before terms are
    /// emitted. Unknown shapes fail closed at the caller when kind evidence is
    /// required.
    pub(super) fn scalar_expr_sort(&self, expr: &Spanned<Expr>) -> Option<TlaSort> {
        match &expr.node {
            Expr::Bool(_) => Some(TlaSort::Bool),
            Expr::Int(_) => Some(TlaSort::Int),
            Expr::String(_) => Some(TlaSort::String),
            Expr::Ident(name, _) | Expr::StateVar(name, ..) => {
                self.vars.get(name).and_then(|info| {
                    info.sort
                        .is_scalar()
                        .then(|| info.sort.clone().canonicalized())
                })
            }
            Expr::Prime(inner) | Expr::SubstIn(_, inner) => self.scalar_expr_sort(inner),
            Expr::Label(label) => self.scalar_expr_sort(&label.body),
            Expr::If(_, then_expr, else_expr) => {
                let then_sort = self.scalar_expr_sort(then_expr)?.canonicalized();
                let else_sort = self.scalar_expr_sort(else_expr)?.canonicalized();
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
            | Expr::Enabled(..)
            | Expr::Unchanged(..) => Some(TlaSort::Bool),
            Expr::Add(..)
            | Expr::Sub(..)
            | Expr::Mul(..)
            | Expr::Div(..)
            | Expr::IntDiv(..)
            | Expr::Mod(..)
            | Expr::Pow(..)
            | Expr::Neg(..) => Some(TlaSort::Int),
            Expr::RecordAccess(record, field) => {
                let name = match &record.node {
                    Expr::Ident(name, _) | Expr::StateVar(name, ..) => name,
                    Expr::Prime(inner) => match &inner.node {
                        Expr::Ident(name, _) | Expr::StateVar(name, ..) => name,
                        _ => return None,
                    },
                    _ => return None,
                };
                self.record_vars
                    .get(name)?
                    .field_sorts
                    .iter()
                    .find(|(candidate, _)| candidate == &field.name.node)
                    .and_then(|(_, sort)| sort.is_scalar().then(|| sort.clone().canonicalized()))
            }
            Expr::FuncApply(container, _) if self.is_seq_var_expr(container) => {
                let (name, _) = self.resolve_seq_var(container).ok()?;
                self.seq_element_sort(&name).ok()
            }
            Expr::FuncApply(func, _) if self.is_func_var_expr(func) => {
                let name = Self::func_expr_base_name(func)?;
                self.func_vars.get(&name).and_then(|info| {
                    info.range_sort
                        .is_scalar()
                        .then(|| info.range_sort.clone().canonicalized())
                })
            }
            Expr::FuncApply(tuple, index) if self.is_tuple_var_expr(tuple) => {
                let (name, _) = self.resolve_tuple_var(tuple).ok()?;
                let index = Self::scalar_tuple_index(index)?;
                self.tuple_vars
                    .get(&name)?
                    .element_sorts
                    .get(index.checked_sub(1)?)
                    .and_then(|sort| sort.is_scalar().then(|| sort.clone().canonicalized()))
            }
            // Every currently-supported CHOOSE encoding returns an Int-carrier
            // witness, including the BOOLEAN-domain encoding.
            Expr::Choose(..) => Some(TlaSort::Int),
            Expr::Apply(op, args)
                if args.len() == 1
                    && matches!(&op.node, Expr::Ident(name, _) if name == "Head")
                    && self.is_seq_var_expr(&args[0]) =>
            {
                let (name, _) = self.resolve_seq_var(&args[0]).ok()?;
                self.seq_element_sort(&name).ok()
            }
            Expr::Apply(op, args)
                if args.len() == 1
                    && matches!(&op.node, Expr::Ident(name, _) if name == "Len" || name == "Cardinality") =>
            {
                Some(TlaSort::Int)
            }
            _ => None,
        }
    }

    /// Recover a precise diagnostic when scalar-kind preflight encounters a
    /// direct reference to an undeclared carrier.  Kind preflight must still
    /// fail closed for genuinely ambiguous expressions, but it should not
    /// hide an ordinary unknown-variable error behind that ambiguity.
    fn unknown_direct_scalar_name<'a>(&self, expr: &'a Spanned<Expr>) -> Option<&'a str> {
        match &expr.node {
            Expr::Ident(name, _) | Expr::StateVar(name, ..)
                if !self.vars.contains_key(name)
                    && !self.func_vars.contains_key(name)
                    && !self.seq_vars.contains_key(name)
                    && !self.record_vars.contains_key(name)
                    && !self.tuple_vars.contains_key(name) =>
            {
                Some(name)
            }
            Expr::Prime(inner) | Expr::SubstIn(_, inner) => self.unknown_direct_scalar_name(inner),
            Expr::Label(label) => self.unknown_direct_scalar_name(&label.body),
            _ => None,
        }
    }

    /// Translate a scalar only after its TLA+ kind is proven to match the
    /// expected function/sequence carrier. Bool must never flow through the Int
    /// path, and interned String must never alias an ordinary Int.
    pub(super) fn translate_scalar_as_sort(
        &mut self,
        expected: &TlaSort,
        expr: &Spanned<Expr>,
        context: &str,
    ) -> AYResult<Term> {
        let expected = expected.clone().canonicalized();
        let actual = self.scalar_expr_sort(expr).ok_or_else(|| {
            AYError::UnsupportedOp(format!("BMC cannot determine {context} scalar kind"))
        })?;
        let actual = actual.canonicalized();
        if actual != expected {
            return Err(AYError::UnsupportedOp(format!(
                "BMC {context} scalar kind mismatch: expected {expected}, got {actual}"
            )));
        }
        match expected {
            TlaSort::Bool => dispatch_translate_bool(self, expr),
            TlaSort::Int => dispatch_translate_int(self, expr),
            TlaSort::String => self.string_scalar_term(expr),
            compound => Err(AYError::UnsupportedOp(format!(
                "BMC {context} has unsupported scalar kind {compound}"
            ))),
        }
    }

    fn scalar_tuple_index(expr: &Spanned<Expr>) -> Option<usize> {
        let value = super::record_encoder::const_fold_int_index(expr)?;
        usize::try_from(value).ok()
    }

    fn require_same_seq_element_sort(
        &self,
        context: &str,
        left: &TlaSort,
        right: &TlaSort,
    ) -> AYResult<()> {
        if left.clone().canonicalized() == right.clone().canonicalized() {
            return Ok(());
        }
        Err(AYError::UnsupportedOp(format!(
            "BMC cannot translate {context} with differing sequence element sorts {left} and \
             {right}; empty sequences can cross this metadata boundary, so the comparison must \
             fail closed"
        )))
    }

    /// Translate sequence indexing: `s[i]` -> `(select arr i)`.
    fn translate_seq_index_bmc(
        &mut self,
        seq_expr: &Spanned<Expr>,
        index_expr: &Spanned<Expr>,
    ) -> AYResult<Term> {
        let (name, step) = self.resolve_seq_var(seq_expr)?;
        let arr = self.get_seq_array_at_step(&name, step)?;
        let idx = dispatch_translate_int(self, index_expr)?;
        Ok(self.solver.try_select(arr, idx)?)
    }

    /// Try to translate a sequence operation that returns Int.
    ///
    /// Handles:
    /// - `Len(s)` -> length term
    /// - `Head(s)` -> `(select arr 1)`
    ///
    /// Returns `None` if the Apply is not a known sequence operation.
    fn translate_seq_int_op(
        &mut self,
        op: &Spanned<Expr>,
        args: &[Spanned<Expr>],
    ) -> Option<AYResult<Term>> {
        let op_name = match &op.node {
            Expr::Ident(name, _) => name.as_str(),
            _ => return None,
        };

        match op_name {
            "Len" if args.len() == 1 => {
                if self.is_seq_var_expr(&args[0]) {
                    Some(self.translate_seq_len_bmc(&args[0]))
                } else {
                    None
                }
            }
            "Head" if args.len() == 1 => {
                if self.is_seq_var_expr(&args[0]) {
                    Some(self.translate_seq_head_bmc(&args[0]))
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    /// Try to translate a sequence operation that returns Bool.
    ///
    /// Currently no sequence operations return Bool in BMC context,
    /// but this provides the extension point for future operations.
    ///
    /// Returns `None` if the Apply is not a known Bool sequence operation.
    fn translate_seq_bool_op(
        &mut self,
        op: &Spanned<Expr>,
        args: &[Spanned<Expr>],
    ) -> Option<AYResult<Term>> {
        let op_name = match &op.node {
            Expr::Ident(name, _) => name.as_str(),
            _ => return None,
        };

        // Head of a Bool-valued sequence would go here; for now
        // Bool sequence ops are uncommon in BMC context.
        match op_name {
            "Head" if args.len() == 1 && self.is_seq_var_expr(&args[0]) => {
                // Head returns the element at index 1; works for any element sort
                Some(self.translate_seq_head_bmc(&args[0]))
            }
            _ => None,
        }
    }

    /// Try to translate sequence equality: `s' = Tail(s)`, `s' = Append(s, e)`,
    /// or `s = s'` where at least one side is a sequence-valued expression.
    ///
    /// Returns `None` if neither side involves a sequence variable.
    /// Returns `Some(result)` if sequence equality is detected.
    fn try_translate_seq_eq(
        &mut self,
        left: &Spanned<Expr>,
        right: &Spanned<Expr>,
    ) -> Option<AYResult<Term>> {
        // Try both directions: left = right, right = left
        if let Some(result) = self.try_translate_seq_eq_directed(left, right) {
            return Some(result);
        }
        self.try_translate_seq_eq_directed(right, left)
    }

    /// Try sequence equality in one direction: lhs is a seq variable,
    /// rhs is a seq operation or seq variable.
    fn try_translate_seq_eq_directed(
        &mut self,
        lhs: &Spanned<Expr>,
        rhs: &Spanned<Expr>,
    ) -> Option<AYResult<Term>> {
        // lhs must be a (possibly primed) sequence variable
        if !self.is_seq_var_expr(lhs) {
            return None;
        }

        let lhs_resolved = match self.resolve_seq_var(lhs) {
            Ok(r) => r,
            Err(e) => return Some(Err(e)),
        };

        // rhs can be: another seq variable, Tail(s), Append(s, e), or <<e1,...,en>>
        if self.is_seq_var_expr(rhs) {
            // seq = seq
            let rhs_resolved = match self.resolve_seq_var(rhs) {
                Ok(r) => r,
                Err(e) => return Some(Err(e)),
            };
            return Some(self.assert_seq_eq_vars(&lhs_resolved, &rhs_resolved));
        }

        // Check for Apply(op, args) patterns: Tail, Append
        if let Expr::Apply(op, args) = &rhs.node {
            if let Expr::Ident(op_name, _) = &op.node {
                match op_name.as_str() {
                    "Tail" if args.len() == 1 && self.is_seq_var_expr(&args[0]) => {
                        return Some(self.translate_seq_eq_tail(&lhs_resolved, &args[0]));
                    }
                    "Append" if args.len() == 2 && self.is_seq_var_expr(&args[0]) => {
                        return Some(self.translate_seq_eq_append(
                            &lhs_resolved,
                            &args[0],
                            &args[1],
                        ));
                    }
                    "SubSeq" if args.len() == 3 && self.is_seq_var_expr(&args[0]) => {
                        return Some(self.translate_seq_eq_subseq(
                            &lhs_resolved,
                            &args[0],
                            &args[1],
                            &args[2],
                        ));
                    }
                    "\\o"
                        if args.len() == 2
                            && self.is_seq_var_expr(&args[0])
                            && self.is_seq_var_expr(&args[1]) =>
                    {
                        return Some(self.translate_seq_eq_concat(
                            &lhs_resolved,
                            &args[0],
                            &args[1],
                        ));
                    }
                    _ => {}
                }
            }
        }

        // Check for Tuple (sequence literal): s = <<e1, e2, ...>>
        if let Expr::Tuple(elems) = &rhs.node {
            return Some(self.translate_seq_eq_literal(&lhs_resolved, elems));
        }

        None
    }

    /// Assert equality between two sequence variables over their logical cells.
    fn assert_seq_eq_vars(
        &mut self,
        lhs: &(String, usize),
        rhs: &(String, usize),
    ) -> AYResult<Term> {
        let l_sort = self.seq_element_sort(&lhs.0)?;
        let r_sort = self.seq_element_sort(&rhs.0)?;
        self.require_same_seq_element_sort("sequence-variable equality", &l_sort, &r_sort)?;

        let l_arr = self.get_seq_array_at_step(&lhs.0, lhs.1)?;
        let l_len = self.get_seq_length_at_step(&lhs.0, lhs.1)?;
        let r_arr = self.get_seq_array_at_step(&rhs.0, rhs.1)?;
        let r_len = self.get_seq_length_at_step(&rhs.0, rhs.1)?;
        let l_max = self.get_seq_max_len(&lhs.0)?;

        self.translate_seq_logical_eq(l_arr, l_len, l_max, r_arr, r_len)
    }

    /// Translate `lhs = Tail(s)` over lhs's logical cells.
    fn translate_seq_eq_tail(
        &mut self,
        lhs: &(String, usize),
        seq_expr: &Spanned<Expr>,
    ) -> AYResult<Term> {
        let (source_name, _) = self.resolve_seq_var(seq_expr)?;
        let lhs_sort = self.seq_element_sort(&lhs.0)?;
        let source_sort = self.seq_element_sort(&source_name)?;
        self.require_same_seq_element_sort("Tail equality", &lhs_sort, &source_sort)?;

        let (tail_arr, tail_len) = self.translate_seq_tail_bmc(seq_expr)?;
        let l_arr = self.get_seq_array_at_step(&lhs.0, lhs.1)?;
        let l_len = self.get_seq_length_at_step(&lhs.0, lhs.1)?;
        let l_max = self.get_seq_max_len(&lhs.0)?;

        self.translate_seq_logical_eq(l_arr, l_len, l_max, tail_arr, tail_len)
    }

    /// Translate `lhs = Append(s, e)` over lhs's logical cells.
    fn translate_seq_eq_append(
        &mut self,
        lhs: &(String, usize),
        seq_expr: &Spanned<Expr>,
        elem_expr: &Spanned<Expr>,
    ) -> AYResult<Term> {
        let (source_name, _) = self.resolve_seq_var(seq_expr)?;
        let lhs_sort = self.seq_element_sort(&lhs.0)?;
        let source_sort = self.seq_element_sort(&source_name)?;
        self.require_same_seq_element_sort("Append equality", &lhs_sort, &source_sort)?;
        let appended_sort = self.scalar_expr_sort(elem_expr).ok_or_else(|| {
            AYError::UnsupportedOp(
                "BMC cannot determine the appended sequence element sort".to_string(),
            )
        })?;
        self.require_same_seq_element_sort("Append element", &lhs_sort, &appended_sort)?;

        let (append_arr, append_len) = self.translate_seq_append_bmc(seq_expr, elem_expr)?;
        let l_arr = self.get_seq_array_at_step(&lhs.0, lhs.1)?;
        let l_len = self.get_seq_length_at_step(&lhs.0, lhs.1)?;
        let l_max = self.get_seq_max_len(&lhs.0)?;

        self.translate_seq_logical_eq(l_arr, l_len, l_max, append_arr, append_len)
    }

    /// Translate `lhs = SubSeq(s, m, n)` over lhs's logical cells.
    fn translate_seq_eq_subseq(
        &mut self,
        lhs: &(String, usize),
        seq_expr: &Spanned<Expr>,
        m_expr: &Spanned<Expr>,
        n_expr: &Spanned<Expr>,
    ) -> AYResult<Term> {
        let (source_name, _) = self.resolve_seq_var(seq_expr)?;
        let lhs_sort = self.seq_element_sort(&lhs.0)?;
        let source_sort = self.seq_element_sort(&source_name)?;
        self.require_same_seq_element_sort("SubSeq equality", &lhs_sort, &source_sort)?;

        let (subseq_arr, subseq_len) = self.translate_seq_subseq_bmc(seq_expr, m_expr, n_expr)?;
        let l_arr = self.get_seq_array_at_step(&lhs.0, lhs.1)?;
        let l_len = self.get_seq_length_at_step(&lhs.0, lhs.1)?;
        let l_max = self.get_seq_max_len(&lhs.0)?;

        self.translate_seq_logical_eq(l_arr, l_len, l_max, subseq_arr, subseq_len)
    }

    /// Translate `lhs = s \o t` over the logical sequence domains.
    ///
    /// Array cells past a sequence's length are representation-only ghosts and
    /// must not participate in TLA+ sequence equality.  In particular, avoid a
    /// whole-array equality between `lhs` and a concat witness: that would make
    /// capacity slack observable and also asks the solver to reconstruct an
    /// extensional model for irrelevant cells.  The two guarded copy regions
    /// below are disjoint and cover exactly `1..=len_s + len_t`.
    fn translate_seq_eq_concat(
        &mut self,
        lhs: &(String, usize),
        s_expr: &Spanned<Expr>,
        t_expr: &Spanned<Expr>,
    ) -> AYResult<Term> {
        let lhs_sort = self.seq_element_sort(&lhs.0)?;
        let (s_name, s_step) = self.resolve_seq_var(s_expr)?;
        let s_sort = self.seq_element_sort(&s_name)?;
        self.require_same_seq_element_sort("concatenation left operand", &lhs_sort, &s_sort)?;
        let (t_name, t_step) = self.resolve_seq_var(t_expr)?;
        let t_sort = self.seq_element_sort(&t_name)?;
        self.require_same_seq_element_sort("concatenation right operand", &lhs_sort, &t_sort)?;

        let l_arr = self.get_seq_array_at_step(&lhs.0, lhs.1)?;
        let l_len = self.get_seq_length_at_step(&lhs.0, lhs.1)?;

        let s_arr = self.get_seq_array_at_step(&s_name, s_step)?;
        let s_len = self.get_seq_length_at_step(&s_name, s_step)?;
        let s_max = self.get_seq_max_len(&s_name)?;

        let t_arr = self.get_seq_array_at_step(&t_name, t_step)?;
        let t_len = self.get_seq_length_at_step(&t_name, t_step)?;
        let t_max = self.get_seq_max_len(&t_name)?;

        let concat_len = self.solver.try_add(s_len, t_len)?;
        let len_eq = self.solver.try_eq(l_len, concat_len)?;
        let mut result = len_eq;

        // Copy only s's logical cells to lhs[1..=len_s].
        for i in 1..=s_max {
            let i_term = self.solver.int_const(i as i64);
            let is_live = self.solver.try_le(i_term, s_len)?;
            let src = self.solver.try_select(s_arr, i_term)?;
            let dst = self.solver.try_select(l_arr, i_term)?;
            let values_eq = self.solver.try_eq(dst, src)?;
            let copy_live = self.solver.try_implies(is_live, values_eq)?;
            result = self.solver.try_and(result, copy_live)?;
        }

        // Copy only t's logical cells immediately after s's logical prefix.
        for j in 1..=t_max {
            let j_term = self.solver.int_const(j as i64);
            let is_live = self.solver.try_le(j_term, t_len)?;
            let dst_index = self.solver.try_add(s_len, j_term)?;
            let src = self.solver.try_select(t_arr, j_term)?;
            let dst = self.solver.try_select(l_arr, dst_index)?;
            let values_eq = self.solver.try_eq(dst, src)?;
            let copy_live = self.solver.try_implies(is_live, values_eq)?;
            result = self.solver.try_and(result, copy_live)?;
        }

        Ok(result)
    }

    /// Translate `lhs = <<e1, e2, ...>>`: set length and elements
    fn translate_seq_eq_literal(
        &mut self,
        lhs: &(String, usize),
        elements: &[Spanned<Expr>],
    ) -> AYResult<Term> {
        let element_sort = self.seq_element_sort(&lhs.0)?;
        // The empty tuple denotes the one empty sequence regardless of element
        // metadata. For a nonempty literal every live element must match the
        // declared homogeneous carrier; unknown/mixed values decline rather
        // than aliasing String and Int through SMT Int.
        for (index, element) in elements.iter().enumerate() {
            let literal_sort = self.scalar_expr_sort(element).ok_or_else(|| {
                AYError::UnsupportedOp(format!(
                    "BMC cannot determine sequence literal element {} sort",
                    index + 1
                ))
            })?;
            self.require_same_seq_element_sort(
                "nonempty sequence-literal equality",
                &element_sort,
                &literal_sort,
            )?;
        }

        let l_arr = self.get_seq_array_at_step(&lhs.0, lhs.1)?;
        let l_len = self.get_seq_length_at_step(&lhs.0, lhs.1)?;

        // Assert length
        let len_val = self.solver.int_const(elements.len() as i64);
        let len_eq = self.solver.try_eq(l_len, len_val)?;

        // Assert each element: (select arr i) = ei
        let mut conjuncts = vec![len_eq];
        for (i, elem) in elements.iter().enumerate() {
            let idx = self.solver.int_const((i + 1) as i64);
            let elem_term = match &element_sort {
                TlaSort::Bool => dispatch_translate_bool(self, elem)?,
                TlaSort::Int | TlaSort::String => dispatch_translate_int(self, elem)?,
                compound => {
                    return Err(AYError::UnsupportedOp(format!(
                        "BMC sequence literal has unsupported element sort {compound}"
                    )))
                }
            };
            let selected = self.solver.try_select(l_arr, idx)?;
            let eq = self.solver.try_eq(selected, elem_term)?;
            conjuncts.push(eq);
        }

        // Build conjunction
        let mut result = conjuncts[0];
        for &c in &conjuncts[1..] {
            result = self.solver.try_and(result, c)?;
        }
        Ok(result)
    }

    /// Check whether an expression refers to a declared function variable.
    ///
    /// Used by `translate_{bool,int}_extended` to decide whether
    /// to handle `FuncApply` or defer to the shared dispatch.
    fn is_func_var_expr(&self, expr: &Spanned<Expr>) -> bool {
        match &expr.node {
            Expr::Ident(name, _) | Expr::StateVar(name, ..) => self.func_vars.contains_key(name),
            Expr::Prime(inner) => match &inner.node {
                Expr::Ident(name, _) | Expr::StateVar(name, ..) => {
                    self.func_vars.contains_key(name)
                }
                _ => false,
            },
            _ => false,
        }
    }

    /// Try to translate function EXCEPT equality.
    ///
    /// Handles patterns like:
    /// - `f' = [f EXCEPT ![a] = b]`
    /// - `[f EXCEPT ![a] = b] = f'`
    /// - `f' = [f EXCEPT ![a] = b, ![c] = d]`
    /// - `f' = [f EXCEPT ![a][b] = c]`
    ///
    /// The EXCEPT produces a new mapping array via `(store ...)`. This
    /// method asserts that the target function variable's mapping equals
    /// the store result, and that the domain is preserved.
    ///
    /// Returns `None` if neither side involves a function EXCEPT.
    fn try_translate_func_except_eq(
        &mut self,
        left: &Spanned<Expr>,
        right: &Spanned<Expr>,
    ) -> Option<AYResult<Term>> {
        // Try left = EXCEPT, right = func var (or vice versa)
        if let Some(result) = self.try_translate_func_except_eq_directed(left, right) {
            return Some(result);
        }
        self.try_translate_func_except_eq_directed(right, left)
    }

    /// Try function EXCEPT equality in one direction:
    /// lhs is a (possibly primed) function variable, rhs is an EXCEPT expression.
    fn try_translate_func_except_eq_directed(
        &mut self,
        lhs: &Spanned<Expr>,
        rhs: &Spanned<Expr>,
    ) -> Option<AYResult<Term>> {
        // rhs must be Except(base, specs) with a function variable base
        let (base, specs) = match &rhs.node {
            Expr::Except(base, specs) if self.is_func_var_expr(base) => {
                (base.as_ref(), specs.as_slice())
            }
            _ => return None,
        };

        // lhs must be a (possibly primed) function variable
        if !self.is_func_var_expr(lhs) {
            return None;
        }

        // Resolve the target mapping (lhs)
        let target_mapping = match self.resolve_func_mapping(lhs) {
            Ok(t) => t,
            Err(e) => return Some(Err(e)),
        };

        // Build the EXCEPT result mapping (rhs)
        let except_mapping = match self.translate_func_except_bmc(base, specs) {
            Ok(t) => t,
            Err(e) => return Some(Err(e)),
        };

        let target_name = match Self::func_expr_base_name(lhs) {
            Some(name) => name,
            None => {
                return Some(Err(AYError::UntranslatableExpr(
                    "BMC EXCEPT equality target must be a function variable".to_string(),
                )))
            }
        };
        let source_name = match Self::func_expr_base_name(base) {
            Some(name) => name,
            None => {
                return Some(Err(AYError::UntranslatableExpr(
                    "BMC EXCEPT equality source must be a function variable".to_string(),
                )))
            }
        };

        // Compare only values at the exact source/target DOMAIN. The store term
        // may differ from the target at arbitrary out-of-domain ghost cells.
        let map_eq = match self.translate_func_logical_mapping_eq(
            &target_name,
            target_mapping,
            &source_name,
            except_mapping,
        ) {
            Ok(t) => t,
            Err(e) => return Some(Err(e)),
        };

        // Also assert domain preservation: target_dom = source_dom
        let dom_eq = match self.assert_domain_preserved(lhs, base) {
            Ok(t) => t,
            Err(e) => return Some(Err(e)),
        };

        Some(self.solver.try_and(map_eq, dom_eq).map_err(AYError::Solver))
    }

    /// Try to translate function construction equality.
    ///
    /// Handles patterns like:
    /// - `f = [x \in S |-> e(x)]`
    /// - `f' = [x \in S |-> e(x)]`
    /// - `[x \in S |-> e(x)] = f`
    ///
    /// The FuncDef produces a (domain, mapping) pair via
    /// `translate_func_construct_bmc`. This method asserts that the target
    /// function variable's domain and mapping equal the construction results.
    ///
    /// Returns `None` if neither side involves a function construction.
    ///
    /// Part of #3786: Function encoding in BMC translator.
    fn try_translate_func_construct_eq(
        &mut self,
        left: &Spanned<Expr>,
        right: &Spanned<Expr>,
    ) -> Option<AYResult<Term>> {
        if let Some(result) = self.try_translate_func_construct_eq_directed(left, right) {
            return Some(result);
        }
        self.try_translate_func_construct_eq_directed(right, left)
    }

    /// Try function construction equality in one direction:
    /// lhs is a (possibly primed) function variable, rhs is a FuncDef expression.
    ///
    /// Part of #3786.
    fn try_translate_func_construct_eq_directed(
        &mut self,
        lhs: &Spanned<Expr>,
        rhs: &Spanned<Expr>,
    ) -> Option<AYResult<Term>> {
        // rhs must be FuncDef(bounds, body)
        let (bounds, body) = match &rhs.node {
            Expr::FuncDef(bounds, body) => (bounds.as_slice(), body.as_ref()),
            _ => return None,
        };

        // lhs must be a (possibly primed) function variable
        if !self.is_func_var_expr(lhs) {
            return None;
        }

        // If the construction has string-literal keys, the target function
        // variable must be encoded with a native `String`-indexed domain so a
        // string key cannot alias an integer-literal key (#5). The
        // Exact `TlaSort::Function` declarations retain this kind directly.
        // Only legacy generic declarations begin Int-keyed and need a one-time
        // upgrade here, before their arrays are resolved or constrained.
        if Self::func_construct_keys_are_strings(bounds) {
            if let Some(fname) = func_var_base_name(lhs) {
                if matches!(self.func_key_sort(&fname), Some(TlaSort::Int)) {
                    if let Err(e) = self.upgrade_func_key_sort_to_string(&fname) {
                        return Some(Err(e));
                    }
                }
            }
        }

        // Resolve the target mapping and domain (lhs)
        let target_mapping = match self.resolve_func_mapping(lhs) {
            Ok(t) => t,
            Err(e) => return Some(Err(e)),
        };
        let target_domain = match self.resolve_func_domain(lhs) {
            Ok(t) => t,
            Err(e) => return Some(Err(e)),
        };
        let target_name = match func_var_base_name(lhs) {
            Some(name) => name,
            None => {
                return Some(Err(AYError::UntranslatableExpr(
                    "BMC function construction target must be a variable".to_string(),
                )))
            }
        };
        let target_range_sort = match self.func_vars.get(&target_name) {
            Some(info) => info.range_sort.clone(),
            None => {
                return Some(Err(AYError::UnknownVariable(format!(
                    "function {target_name}"
                ))))
            }
        };

        // Constrain the target's own mapping array *directly* with the
        // construction's per-element value constraints, and obtain the
        // construction domain. We deliberately do NOT build a fresh mapping
        // array and assert `target_map = __func_map`: that array-to-array
        // equality (whose RHS carries several `select` constraints) triggers
        // an AY QF_AUFLIA model-construction gap that degrades `Sat -> Unknown`
        // (see `test_bmc_func_construct_eq_init`). Constraining the target in
        // place matches the always-SAT `test_bmc_assert_concrete_func_state`
        // encoding and needs only the domain equality returned below.
        let construct_domain = match self.translate_func_construct_bmc_into(
            bounds,
            body,
            target_mapping,
            &target_range_sort,
        ) {
            Ok(d) => d,
            Err(e) => return Some(Err(e)),
        };

        // Assert domain equality: target_dom = construct_dom
        let dom_eq = match self.solver.try_eq(target_domain, construct_domain) {
            Ok(t) => t,
            Err(e) => return Some(Err(AYError::Solver(e))),
        };

        Some(Ok(dom_eq))
    }

    /// Assert that two function variables have the same domain.
    ///
    /// Used when translating EXCEPT equality: the domain of f' must
    /// equal the domain of f (EXCEPT does not change the domain).
    fn assert_domain_preserved(
        &mut self,
        target: &Spanned<Expr>,
        source: &Spanned<Expr>,
    ) -> AYResult<Term> {
        // Map-only symbolic-domain functions carry NO domain membership array —
        // their domain is the fixed arithmetic fact `lo <= x <= N+offset`, which
        // EXCEPT preserves trivially. There is nothing to assert (and no
        // `domain_terms` to read), so the preservation constraint is `TRUE`.
        if self.func_expr_is_symbolic_domain(target) || self.func_expr_is_symbolic_domain(source) {
            return Ok(self.solver.bool_const(true));
        }
        let target_dom = self.resolve_func_domain(target)?;
        let source_dom = self.resolve_func_domain(source)?;
        Ok(self.solver.try_eq(target_dom, source_dom)?)
    }

    /// Whether `expr` refers to a (possibly primed) symbolic-domain (map-only)
    /// function variable.
    fn func_expr_is_symbolic_domain(&self, expr: &Spanned<Expr>) -> bool {
        Self::func_expr_base_name(expr).is_some_and(|n| self.func_symbolic_domain(&n).is_some())
    }

    /// Resolve the domain array for a function expression.
    fn resolve_func_domain(&mut self, expr: &Spanned<Expr>) -> AYResult<Term> {
        match &expr.node {
            Expr::Ident(name, _) | Expr::StateVar(name, ..) => {
                self.get_func_domain_at_step(name, self.current_step)
            }
            Expr::Prime(inner) => match &inner.node {
                Expr::Ident(name, _) | Expr::StateVar(name, ..) => {
                    self.get_func_domain_at_step(name, self.current_step + 1)
                }
                _ => Err(AYError::UntranslatableExpr(
                    "BMC domain resolution requires function variable".to_string(),
                )),
            },
            _ => Err(AYError::UntranslatableExpr(
                "BMC domain resolution requires function variable".to_string(),
            )),
        }
    }

    // Compound type dispatch helpers (subseteq, set enum, set binop, domain,
    // except, func def, cardinality, universe extraction) are in
    // compound_dispatch.rs — Part of #3778.
}

impl BmcTranslator {
    /// Returns `true` iff `e` is a STRING-SCALAR operand: a string literal, a
    /// `TlaSort::String` state variable, or a primed string state variable.
    /// Pure (no mutation) so both operands of an equality can be checked before
    /// translating either. See the string-scalar arm of `translate_eq`.
    pub(super) fn is_string_scalar(&self, e: &Spanned<Expr>) -> bool {
        let is_string_var = |name: &str| {
            self.vars
                .get(name)
                .map_or(false, |info| info.sort == TlaSort::String)
        };
        match &e.node {
            Expr::String(_) => true,
            Expr::Ident(name, _) | Expr::StateVar(name, ..) => is_string_var(name),
            Expr::Prime(inner) => matches!(
                &inner.node,
                Expr::Ident(name, _) | Expr::StateVar(name, ..) if is_string_var(name)
            ),
            _ => false,
        }
    }

    /// Translate a STRING-SCALAR operand (see [`Self::is_string_scalar`]) to its
    /// interned-int term: a string literal interns to a distinct id, a string
    /// variable resolves to its `Sort::Int` term at the current step (or the
    /// next step when primed). Precondition: `is_string_scalar(e)` holds.
    pub(super) fn string_scalar_term(&mut self, e: &Spanned<Expr>) -> AYResult<Term> {
        match &e.node {
            Expr::String(s) => {
                let id = self.bmc_intern_string(s);
                Ok(self.solver.int_const(id))
            }
            Expr::Ident(name, _) | Expr::StateVar(name, ..) => {
                self.get_var_at_step(name, self.current_step)
            }
            Expr::Prime(inner) => match &inner.node {
                Expr::Ident(name, _) | Expr::StateVar(name, ..) => {
                    self.get_var_at_step(name, self.current_step + 1)
                }
                _ => Err(AYError::UntranslatableExpr(
                    "string_scalar_term: primed non-variable".to_string(),
                )),
            },
            _ => Err(AYError::UntranslatableExpr(
                "string_scalar_term: not a string scalar".to_string(),
            )),
        }
    }
}
