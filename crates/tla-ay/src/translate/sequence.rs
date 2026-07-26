// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Sequence operations for AY translation.
//!
//! Bridges the per-variable `SequenceVarInfo` representation (used for
//! `s[i]` indexing via ITE chains) with the array-based `SequenceEncoder`
//! (used for compositional operations like Tail, Append, SubSeq).
//!
//! # Dispatch
//!
//! - `s[i]` — dispatched from `membership/mod.rs` via `translate_seq_apply_{bool,int}`
//! - `Len(s)`, `Head(s)`, `Tail(s)`, `Append(s,e)`, `SubSeq(s,m,n)` — dispatched
//!   from `translate_expr_impl.rs` via `Expr::Apply` in `translate_{bool,int}_extended`
//!
//! Part of #3793: sequence encoding as bounded arrays.

use ay_dpll::api::{Sort, Term};
use tla_core::ast::Expr;
use tla_core::Spanned;

use super::sequence_encoder::{SeqTerm, SequenceEncoder};
use super::{AYTranslator, SequenceVarInfo, TlaSort};
use crate::error::{AYError, AYResult};

// =========================================================================
// Default max_len for intermediate sequences when no SequenceVarInfo exists
// =========================================================================
const DEFAULT_MAX_LEN: usize = 10;

impl AYTranslator {
    // =====================================================================
    // s[i] indexing (dispatched from membership/mod.rs)
    // =====================================================================

    /// Translate sequence application `s[i]` returning Bool.
    ///
    /// Uses an ITE chain over per-variable terms (same pattern as tuple indexing).
    pub(super) fn translate_seq_apply_bool(
        &mut self,
        var_name: &str,
        seq_info: &SequenceVarInfo,
        arg: &Spanned<Expr>,
    ) -> AYResult<Term> {
        if seq_info.element_sort != TlaSort::Bool {
            return Err(AYError::TypeMismatch {
                name: format!("{var_name}[i]"),
                expected: "Bool".to_string(),
                actual: format!("{}", seq_info.element_sort),
            });
        }

        // Constant index fast path
        if let Some(idx) = self.try_expr_to_int(arg) {
            if idx < 1 || idx as usize > seq_info.max_len {
                return Err(AYError::UnsupportedOp(format!(
                    "sequence index {idx} out of bounds (max_len={})",
                    seq_info.max_len
                )));
            }
            return seq_info
                .element_terms
                .get(&(idx as usize))
                .copied()
                .ok_or_else(|| AYError::UnsupportedOp(format!("{var_name}[{idx}] not found")));
        }

        // Dynamic index: build ITE chain
        self.build_seq_ite_chain_bool(var_name, seq_info, arg)
    }

    /// Translate sequence application `s[i]` returning Int.
    ///
    /// Uses an ITE chain over per-variable terms (same pattern as tuple indexing).
    pub(super) fn translate_seq_apply_int(
        &mut self,
        var_name: &str,
        seq_info: &SequenceVarInfo,
        arg: &Spanned<Expr>,
    ) -> AYResult<Term> {
        if seq_info.element_sort != TlaSort::Int {
            return Err(AYError::TypeMismatch {
                name: format!("{var_name}[i]"),
                expected: "Int".to_string(),
                actual: format!("{}", seq_info.element_sort),
            });
        }

        // Constant index fast path
        if let Some(idx) = self.try_expr_to_int(arg) {
            if idx < 1 || idx as usize > seq_info.max_len {
                return Err(AYError::UnsupportedOp(format!(
                    "sequence index {idx} out of bounds (max_len={})",
                    seq_info.max_len
                )));
            }
            return seq_info
                .element_terms
                .get(&(idx as usize))
                .copied()
                .ok_or_else(|| AYError::UnsupportedOp(format!("{var_name}[{idx}] not found")));
        }

        // Dynamic index: build ITE chain
        self.build_seq_ite_chain_int(var_name, seq_info, arg)
    }

    // =====================================================================
    // ITE chain builders (mirror tuple.rs pattern)
    // =====================================================================

    /// Build ITE chain: IF idx=1 THEN s__1 ELSE IF idx=2 THEN s__2 ELSE ... s__n
    fn build_seq_ite_chain_int(
        &mut self,
        var_name: &str,
        seq_info: &SequenceVarInfo,
        index: &Spanned<Expr>,
    ) -> AYResult<Term> {
        let index_term = self.translate_int(index)?;
        let max = seq_info.max_len;

        // Default to last element
        let mut result = seq_info
            .element_terms
            .get(&max)
            .copied()
            .ok_or_else(|| AYError::UnsupportedOp(format!("{var_name}[{max}] not found")))?;

        // Build from max-1 down to 1
        for idx in (1..max).rev() {
            let idx_const = self.solver_mut().int_const(idx as i64);
            let cond = self.solver_mut().try_eq(index_term, idx_const)?;
            let elem_term =
                seq_info.element_terms.get(&idx).copied().ok_or_else(|| {
                    AYError::UnsupportedOp(format!("{var_name}[{idx}] not found"))
                })?;
            result = self.solver_mut().try_ite(cond, elem_term, result)?;
        }

        Ok(result)
    }

    /// Build ITE chain for Bool-sorted sequence elements.
    fn build_seq_ite_chain_bool(
        &mut self,
        var_name: &str,
        seq_info: &SequenceVarInfo,
        index: &Spanned<Expr>,
    ) -> AYResult<Term> {
        let index_term = self.translate_int(index)?;
        let max = seq_info.max_len;

        let mut result = seq_info
            .element_terms
            .get(&max)
            .copied()
            .ok_or_else(|| AYError::UnsupportedOp(format!("{var_name}[{max}] not found")))?;

        for idx in (1..max).rev() {
            let idx_const = self.solver_mut().int_const(idx as i64);
            let cond = self.solver_mut().try_eq(index_term, idx_const)?;
            let elem_term =
                seq_info.element_terms.get(&idx).copied().ok_or_else(|| {
                    AYError::UnsupportedOp(format!("{var_name}[{idx}] not found"))
                })?;
            result = self.solver_mut().try_ite(cond, elem_term, result)?;
        }

        Ok(result)
    }

    // =====================================================================
    // Stdlib sequence operations (dispatched from translate_expr_impl.rs)
    // =====================================================================

    /// Try to translate a sequence operation returning Bool.
    ///
    /// Handles `Expr::Apply(Ident("Head"), [s])` when the sequence has Bool elements.
    /// Returns `None` if the Apply is not a recognized sequence operation.
    pub(super) fn try_translate_seq_op_bool(
        &mut self,
        func: &Spanned<Expr>,
        args: &[Spanned<Expr>],
    ) -> Option<AYResult<Term>> {
        let name = match &func.node {
            Expr::Ident(name, _) => name.as_str(),
            _ => return None,
        };

        match name {
            "Head" if args.len() == 1 => {
                // Native path first; FALL THROUGH to the array path when the
                // operand is not a native sequence (e.g. a seq var declared on
                // the bounded array path because of a non-scalar element).
                if self.native_seq {
                    if let Some(result) = self.native_head(&args[0]) {
                        return Some(result);
                    }
                }
                Some(self.translate_head_bool(&args[0]))
            }
            _ => None,
        }
    }

    /// Try to translate a sequence operation returning Int.
    ///
    /// Handles:
    /// - `Len(s)` — always returns Int
    /// - `Head(s)` — when the sequence has Int elements
    ///
    /// Returns `None` if the Apply is not a recognized sequence operation.
    pub(super) fn try_translate_seq_op_int(
        &mut self,
        func: &Spanned<Expr>,
        args: &[Spanned<Expr>],
    ) -> Option<AYResult<Term>> {
        let name = match &func.node {
            Expr::Ident(name, _) => name.as_str(),
            _ => return None,
        };

        match name {
            "Len" if args.len() == 1 => {
                // Native path first; FALL THROUGH to the array path when the
                // operand is not a native sequence (resolve-failure), mirroring
                // how `try_native_seq_index` returns None and falls through.
                if self.native_seq {
                    if let Some(result) = self.native_len(&args[0]) {
                        return Some(result);
                    }
                }
                Some(self.translate_len(&args[0]))
            }
            "Head" if args.len() == 1 => {
                if self.native_seq {
                    if let Some(result) = self.native_head(&args[0]) {
                        return Some(result);
                    }
                }
                Some(self.translate_head_int(&args[0]))
            }
            _ => None,
        }
    }

    // =====================================================================
    // Individual operation implementations
    // =====================================================================

    /// Translate `Len(s)` — returns the length term of the sequence.
    fn translate_len(&mut self, seq_expr: &Spanned<Expr>) -> AYResult<Term> {
        let seq = self.resolve_seq_term(seq_expr)?;
        let enc = SequenceEncoder::new(Sort::Int); // sort doesn't matter for Len
        Ok(enc.encode_len(&seq))
    }

    /// Translate `Head(s)` where the sequence has Int elements.
    fn translate_head_int(&mut self, seq_expr: &Spanned<Expr>) -> AYResult<Term> {
        let (seq, elem_sort) = self.resolve_seq_term_with_sort(seq_expr)?;
        let enc = SequenceEncoder::new(elem_sort);
        enc.encode_head(self, &seq)
    }

    /// Translate `Head(s)` where the sequence has Bool elements.
    fn translate_head_bool(&mut self, seq_expr: &Spanned<Expr>) -> AYResult<Term> {
        let (seq, elem_sort) = self.resolve_seq_term_with_sort(seq_expr)?;
        let enc = SequenceEncoder::new(elem_sort);
        enc.encode_head(self, &seq)
    }

    // =====================================================================
    // SeqTerm resolution: bridge SequenceVarInfo -> SeqTerm
    // =====================================================================

    /// Resolve a TLA+ expression to a `SeqTerm` (array + length).
    ///
    /// For `Ident` expressions that reference a declared sequence variable,
    /// converts the per-variable `SequenceVarInfo` into an array-based `SeqTerm`
    /// by building `(store ... (store arr 1 s__1) 2 s__2) ... n s__n)`.
    fn resolve_seq_term(&mut self, expr: &Spanned<Expr>) -> AYResult<SeqTerm> {
        let (seq, _sort) = self.resolve_seq_term_with_sort(expr)?;
        Ok(seq)
    }

    /// Like `resolve_seq_term` but also returns the element sort.
    ///
    /// Handles:
    /// - `Ident(name)` — lookup in `seq_vars` and bridge to `SeqTerm`
    /// - `Apply(Ident("Tail"), [s])` — resolve `s` then apply Tail encoding
    /// - `Apply(Ident("Append"), [s, e])` — resolve `s` then apply Append encoding
    /// - `Apply(Ident("SubSeq"), [s, m, n])` — resolve `s` then apply SubSeq encoding
    /// - `Apply(Ident("\\o"), [s, t])` — resolve both then apply Concat encoding
    fn resolve_seq_term_with_sort(&mut self, expr: &Spanned<Expr>) -> AYResult<(SeqTerm, Sort)> {
        match &expr.node {
            Expr::Ident(name, _) => {
                if let Some(info) = self.seq_vars.get(name).cloned() {
                    let ay_sort = info.element_sort.to_ay()?;
                    let seq = self.seq_var_to_seq_term(&info, &ay_sort)?;
                    Ok((seq, ay_sort))
                } else {
                    Err(AYError::UnsupportedOp(format!(
                        "{name} is not a declared sequence variable"
                    )))
                }
            }
            Expr::Apply(op, args) => {
                let name = match &op.node {
                    Expr::Ident(name, _) => name.as_str(),
                    _ => {
                        return Err(AYError::UnsupportedOp(
                            "sequence operation on non-Ident Apply not supported".to_string(),
                        ))
                    }
                };
                match name {
                    "Tail" if args.len() == 1 => self.resolve_tail_seq_term(&args[0]),
                    "Append" if args.len() == 2 => self.resolve_append_seq_term(&args[0], &args[1]),
                    "SubSeq" if args.len() == 3 => {
                        self.resolve_subseq_seq_term(&args[0], &args[1], &args[2])
                    }
                    "\\o" if args.len() == 2 => self.resolve_concat_seq_term(&args[0], &args[1]),
                    _ => Err(AYError::UnsupportedOp(format!(
                        "unrecognized sequence operation: {name}"
                    ))),
                }
            }
            _ => Err(AYError::UnsupportedOp(
                "sequence operations on non-variable expressions not yet supported".to_string(),
            )),
        }
    }

    /// Resolve `Tail(s)` to a `SeqTerm`.
    ///
    /// Recursively resolves `s`, then applies the Tail encoding from
    /// `SequenceEncoder`. Uses the inner sequence's max_len (from
    /// `SequenceVarInfo` if available, otherwise `DEFAULT_MAX_LEN`).
    fn resolve_tail_seq_term(&mut self, seq_expr: &Spanned<Expr>) -> AYResult<(SeqTerm, Sort)> {
        let max_len = self.infer_max_len(seq_expr);
        let (seq, elem_sort) = self.resolve_seq_term_with_sort(seq_expr)?;
        let enc = SequenceEncoder::new(elem_sort.clone());
        let result = enc.encode_tail(self, &seq, max_len)?;
        Ok((result, elem_sort))
    }

    /// Resolve `Append(s, e)` to a `SeqTerm`.
    ///
    /// Recursively resolves `s`, translates `e` as an Int or Bool term,
    /// then applies the Append encoding from `SequenceEncoder`.
    fn resolve_append_seq_term(
        &mut self,
        seq_expr: &Spanned<Expr>,
        elem_expr: &Spanned<Expr>,
    ) -> AYResult<(SeqTerm, Sort)> {
        let (seq, elem_sort) = self.resolve_seq_term_with_sort(seq_expr)?;
        let elem_term = match elem_sort {
            Sort::Int => self.translate_int(elem_expr)?,
            Sort::Bool => self.translate_bool(elem_expr)?,
            _ => {
                return Err(AYError::UnsupportedOp(format!(
                    "Append: unsupported element sort {elem_sort:?}"
                )))
            }
        };
        let enc = SequenceEncoder::new(elem_sort.clone());
        let result = enc.encode_append(self, &seq, elem_term)?;
        Ok((result, elem_sort))
    }

    /// Resolve `SubSeq(s, m, n)` to a `SeqTerm`.
    ///
    /// Recursively resolves `s`, translates `m` and `n` as Int terms,
    /// then applies the SubSeq encoding from `SequenceEncoder`.
    fn resolve_subseq_seq_term(
        &mut self,
        seq_expr: &Spanned<Expr>,
        m_expr: &Spanned<Expr>,
        n_expr: &Spanned<Expr>,
    ) -> AYResult<(SeqTerm, Sort)> {
        let max_len = self.infer_max_len(seq_expr);
        let (seq, elem_sort) = self.resolve_seq_term_with_sort(seq_expr)?;
        let m = self.translate_int(m_expr)?;
        let n = self.translate_int(n_expr)?;
        let enc = SequenceEncoder::new(elem_sort.clone());
        let result = enc.encode_subseq(self, &seq, m, n, max_len)?;
        Ok((result, elem_sort))
    }

    /// Resolve `s \o t` (concatenation) to a `SeqTerm`.
    ///
    /// Recursively resolves both `s` and `t`, then applies the Concat
    /// encoding from `SequenceEncoder`.
    fn resolve_concat_seq_term(
        &mut self,
        s_expr: &Spanned<Expr>,
        t_expr: &Spanned<Expr>,
    ) -> AYResult<(SeqTerm, Sort)> {
        let max_len_s = self.infer_max_len(s_expr);
        let max_len_t = self.infer_max_len(t_expr);
        let (seq_s, elem_sort_s) = self.resolve_seq_term_with_sort(s_expr)?;
        let (seq_t, elem_sort_t) = self.resolve_seq_term_with_sort(t_expr)?;
        // Both sequences must have the same element sort
        if elem_sort_s != elem_sort_t {
            return Err(AYError::TypeMismatch {
                name: "\\o".to_string(),
                expected: format!("{elem_sort_s:?}"),
                actual: format!("{elem_sort_t:?}"),
            });
        }
        let max_len = max_len_s + max_len_t;
        let enc = SequenceEncoder::new(elem_sort_s.clone());
        let result = enc.encode_concat(self, &seq_s, &seq_t, max_len)?;
        Ok((result, elem_sort_s))
    }

    /// Infer the max_len for a sequence expression.
    ///
    /// If the expression is an `Ident` referencing a declared sequence variable,
    /// returns that variable's `max_len`. Otherwise returns `DEFAULT_MAX_LEN`.
    fn infer_max_len(&self, expr: &Spanned<Expr>) -> usize {
        match &expr.node {
            Expr::Ident(name, _) => self
                .seq_vars
                .get(name)
                .map_or(DEFAULT_MAX_LEN, |info| info.max_len),
            _ => DEFAULT_MAX_LEN,
        }
    }

    /// Convert a per-variable `SequenceVarInfo` to an array-based `SeqTerm`.
    ///
    /// Builds the array by storing each per-variable term:
    /// `(store (store ... (store base_arr 1 s__1) 2 s__2) ... n s__n)`
    fn seq_var_to_seq_term(
        &mut self,
        info: &SequenceVarInfo,
        elem_sort: &Sort,
    ) -> AYResult<SeqTerm> {
        let arr_sort = Sort::array(Sort::Int, elem_sort.clone());
        let arr_name = self.fresh_name("seq_arr");
        let mut arr = self.solver_mut().declare_const(&arr_name, arr_sort);

        // Store each per-variable element into the array
        for idx in 1..=info.max_len {
            if let Some(&elem_term) = info.element_terms.get(&idx) {
                let idx_term = self.solver_mut().int_const(idx as i64);
                arr = self.solver_mut().try_store(arr, idx_term, elem_term)?;
            }
        }

        Ok(SeqTerm {
            array: arr,
            len: info.len_term,
        })
    }
}

// =========================================================================
// Option-A: native unbounded `Sort::Seq` operations.
//
// Enabled when `self.native_seq` is set (via `AYTranslator::new_with_seq`).
// Each TLA+ sequence op is lowered to an AY `seq.*` builder. TLA+ sequences
// are 1-INDEXED; native `seq.nth` / `seq.extract` are 0-INDEXED, so callers
// translate the 1-based TLA+ index `i` to `i - 1`.
//
// op -> builder mapping:
//   Len(s)        -> try_seq_len(s)
//   s[i]          -> try_seq_nth(s, i - 1)
//   Head(s)       -> try_seq_nth(s, 0)
//   Tail(s)       -> try_seq_extract(s, 1, seq_len(s) - 1)
//   <<>>          -> seq_empty(elem)
//   <<e1..en>>    -> fold try_seq_concat over try_seq_unit(ei)
//   Append(s, e)  -> try_seq_concat(s, try_seq_unit(e))
//   s \o t        -> try_seq_concat(s, t)
//   SubSeq(s,m,n) -> try_seq_extract(s, m - 1, n - m + 1)
//   s = t         -> try_eq(s, t)   (native term equality)
// =========================================================================
impl AYTranslator {
    /// Translate `Len(s)` natively: `seq.len(s)` (unbounded Int).
    ///
    /// Returns `None` when `seq_expr` does not resolve to a native sequence term
    /// (e.g. a seq var declared on the bounded array path), so the caller can
    /// FALL THROUGH to the array `translate_len`. `Some(Err(..))` is reserved for
    /// genuine solver/build failures on an established native sequence.
    pub(super) fn native_len(&mut self, seq_expr: &Spanned<Expr>) -> Option<AYResult<Term>> {
        let s = match self.resolve_native_seq_term(seq_expr) {
            Ok((s, _elem)) => s,
            Err(_) => return None,
        };
        Some((|| Ok(self.solver_mut().try_seq_len(s)?))())
    }

    /// Translate `Head(s)` natively: `seq.nth(s, 0)` (0-indexed first element).
    ///
    /// Returns the element-sorted term directly; the caller's int/bool context
    /// is satisfied because the native element sort matches the sequence's.
    ///
    /// Returns `None` when `seq_expr` does not resolve to a native sequence term,
    /// so the caller can FALL THROUGH to the array `translate_head_{int,bool}`.
    pub(super) fn native_head(&mut self, seq_expr: &Spanned<Expr>) -> Option<AYResult<Term>> {
        // Constant-fold Head over a fully-literal sequence: Head(<<c1,..>>) = c1.
        // ay's native Seq theory returns Unknown on seq.nth over a seq.++-built
        // literal, so discharge the definitional fold here instead.
        if let Some(elements) = self.fold_literal_seq_elements(seq_expr) {
            if let Some(first) = elements.first() {
                let first = first.clone();
                return Some(self.translate_native_elem(&first).map(|(term, _)| term));
            }
            // Head(<<>>) is undefined; fall through to the native encoding.
        }
        let s = match self.resolve_native_seq_term(seq_expr) {
            Ok((s, _elem)) => s,
            Err(_) => return None,
        };
        Some((|| {
            let zero = self.solver_mut().int_const(0);
            Ok(self.solver_mut().try_seq_nth(s, zero)?)
        })())
    }

    /// Translate native sequence indexing `s[i]` (TLA+ 1-indexed): `seq.nth(s, i-1)`.
    ///
    /// `base` is any expression that resolves to a native sequence term
    /// (an `Ident` seq var, a `<<..>>` literal, or a `Tail`/`Append`/`SubSeq`/`\o`
    /// application). Returns `None` when `base` is not a native sequence (so the
    /// caller falls through to the function/tuple application paths).
    pub(super) fn try_native_seq_index(
        &mut self,
        base: &Spanned<Expr>,
        index: &Spanned<Expr>,
    ) -> Option<AYResult<Term>> {
        if !self.native_seq {
            return None;
        }
        // Constant-fold literal-sequence indexing: `<<c1,..,cn>>[k] = c_k`
        // (1-indexed), also over `\o`/`Append`/`SubSeq`/`Tail` literal structure.
        // Discharges the definitional fold that ay's native seq.nth-over-literal
        // cannot (returns Unknown).
        if let Some(k) = self.try_expr_to_int(index) {
            if let Some(elements) = self.fold_literal_seq_elements(base) {
                if k >= 1 && (k as usize) <= elements.len() {
                    let elem = elements[(k - 1) as usize].clone();
                    return Some(self.translate_native_elem(&elem).map(|(term, _)| term));
                }
                // Out-of-bounds / empty: leave to the native encoding.
            }
        }
        // Probe whether `base` is a native sequence without committing to errors
        // that belong to the function/tuple paths.
        let (s, _elem) = match self.resolve_native_seq_term(base) {
            Ok(v) => v,
            Err(_) => return None,
        };
        Some((|| {
            let idx = self.translate_int(index)?;
            let one = self.solver_mut().int_const(1);
            let idx0 = self.solver_mut().try_sub(idx, one)?;
            Ok(self.solver_mut().try_seq_nth(s, idx0)?)
        })())
    }

    /// Probe native sequence equality `s = t` (Option-A).
    ///
    /// Returns `Ok(Some(term))` when both sides resolve to native sequence terms,
    /// `Ok(None)` when either side is not a native sequence (so the caller falls
    /// through to tuple / record / function / bool equality).
    pub(super) fn try_translate_native_seq_equality(
        &mut self,
        left: &Spanned<Expr>,
        right: &Spanned<Expr>,
    ) -> AYResult<Option<Term>> {
        if !self.native_seq {
            return Ok(None);
        }
        let (l, _) = match self.resolve_native_seq_term(left) {
            Ok(v) => v,
            Err(_) => return Ok(None),
        };
        let (r, _) = match self.resolve_native_seq_term(right) {
            Ok(v) => v,
            Err(_) => return Ok(None),
        };
        Ok(Some(self.solver_mut().try_eq(l, r)?))
    }

    /// Resolve a TLA+ expression to a native `(Seq Elem)` term + element sort.
    ///
    /// Handles:
    /// - `Ident(name)`            — native seq var lookup
    /// - `Tuple([..])`            — `<<e1, .., en>>` sequence literal
    /// - `Apply(Ident("Tail"),   [s])`
    /// - `Apply(Ident("Append"), [s, e])`
    /// - `Apply(Ident("SubSeq"), [s, m, n])`
    /// - `Apply(Ident("\\o"),    [s, t])`
    ///
    /// Errors for anything else (used as a probe by indexing/equality, which
    /// translate `Err` into "not a sequence -> fall through").
    pub(super) fn resolve_native_seq_term(
        &mut self,
        expr: &Spanned<Expr>,
    ) -> AYResult<(Term, TlaSort)> {
        match &expr.node {
            Expr::Ident(name, _) => {
                if let Some(info) = self.seq_native_vars.get(name) {
                    Ok((info.term, info.element_sort.clone()))
                } else {
                    Err(AYError::UnsupportedOp(format!(
                        "{name} is not a native sequence variable"
                    )))
                }
            }
            Expr::Tuple(elements) => self.build_native_seq_literal(elements),
            Expr::Apply(op, args) => {
                let name = match &op.node {
                    Expr::Ident(name, _) => name.as_str(),
                    _ => {
                        return Err(AYError::UnsupportedOp(
                            "native sequence op on non-Ident Apply".to_string(),
                        ))
                    }
                };
                match name {
                    "Tail" if args.len() == 1 => self.native_tail(&args[0]),
                    "Append" if args.len() == 2 => self.native_append(&args[0], &args[1]),
                    "SubSeq" if args.len() == 3 => self.native_subseq(&args[0], &args[1], &args[2]),
                    "\\o" if args.len() == 2 => self.native_concat(&args[0], &args[1]),
                    other => Err(AYError::UnsupportedOp(format!(
                        "unrecognized native sequence operation: {other}"
                    ))),
                }
            }
            _ => Err(AYError::UnsupportedOp(
                "expression is not a native sequence".to_string(),
            )),
        }
    }

    /// Constant-fold a native-sequence expression to its concrete element list
    /// when the sequence STRUCTURE is fully determined at translation time.
    ///
    /// Returns `Some(elements)` (the element sub-expressions, in order) for a
    /// `<<..>>` tuple and for `Tail`/`Append`/`\o`/`SubSeq` applied to already
    /// foldable operands (with integer-literal `SubSeq` bounds). Returns `None`
    /// otherwise (e.g. a seq VARIABLE, or a non-literal `SubSeq` bound) so
    /// callers fall back to the native `seq.nth`/`seq.extract` encoding.
    ///
    /// These folds are DEFINITIONAL — `Head(<<c1,..>>) = c1`,
    /// `<<c1,..,cn>>[k] = c_k`, `(s \o t)[k]`, `Append(s,e)[k]`,
    /// `SubSeq(s,m,n)[k]` all reduce closed sequence structure to the element it
    /// denotes — hence SOUND. They exist because ay's native Seq theory returns
    /// `Unknown` on `seq.nth` over a `seq.++`-built literal sequence.
    fn fold_literal_seq_elements(&self, expr: &Spanned<Expr>) -> Option<Vec<Spanned<Expr>>> {
        match &expr.node {
            Expr::Tuple(elements) => Some(elements.clone()),
            Expr::Apply(op, args) => {
                let name = match &op.node {
                    Expr::Ident(name, _) => name.as_str(),
                    _ => return None,
                };
                match name {
                    "\\o" if args.len() == 2 => {
                        let mut left = self.fold_literal_seq_elements(&args[0])?;
                        let right = self.fold_literal_seq_elements(&args[1])?;
                        left.extend(right);
                        Some(left)
                    }
                    "Append" if args.len() == 2 => {
                        let mut base = self.fold_literal_seq_elements(&args[0])?;
                        base.push(args[1].clone());
                        Some(base)
                    }
                    "Tail" if args.len() == 1 => {
                        let base = self.fold_literal_seq_elements(&args[0])?;
                        // Tail(<<>>) is undefined; refuse to fold.
                        if base.is_empty() {
                            return None;
                        }
                        Some(base[1..].to_vec())
                    }
                    "SubSeq" if args.len() == 3 => {
                        let base = self.fold_literal_seq_elements(&args[0])?;
                        let m = self.try_expr_to_int(&args[1])?;
                        let n = self.try_expr_to_int(&args[2])?;
                        // TLA+ SubSeq(s, m, n): 1-indexed inclusive; n < m => <<>>.
                        if n < m {
                            return Some(Vec::new());
                        }
                        if m < 1 {
                            return None;
                        }
                        let start = (m - 1) as usize;
                        let end = n as usize; // inclusive index n -> exclusive slice end n
                        if end > base.len() {
                            return None;
                        }
                        Some(base[start..end].to_vec())
                    }
                    _ => None,
                }
            }
            _ => None,
        }
    }

    /// Build a `<<e1, .., en>>` literal as a native sequence.
    ///
    /// The empty literal `<<>>` becomes `seq_empty(Int)` (Int is the default
    /// element sort when no element fixes it). A non-empty literal folds
    /// `seq.++` over `seq.unit(ei)` so the result length is exactly `n`.
    fn build_native_seq_literal(
        &mut self,
        elements: &[Spanned<Expr>],
    ) -> AYResult<(Term, TlaSort)> {
        if elements.is_empty() {
            let empty = self.solver_mut().seq_empty(Sort::Int);
            return Ok((empty, TlaSort::Int));
        }

        let (first_term, elem_sort) = self.translate_native_elem(&elements[0])?;
        let mut acc = self.solver_mut().try_seq_unit(first_term)?;
        for elem in &elements[1..] {
            let term = self.translate_native_elem_as(elem, &elem_sort)?;
            let unit = self.solver_mut().try_seq_unit(term)?;
            acc = self.solver_mut().try_seq_concat(acc, unit)?;
        }
        Ok((acc, elem_sort))
    }

    /// Translate a scalar sequence element, inferring its sort.
    ///
    /// Tries Int first, then String (interned as Int), then Bool. Strings are
    /// represented as interned integers, so a `String`-element native sequence
    /// is an `(Seq Int)` whose elements are interned IDs.
    fn translate_native_elem(&mut self, elem: &Spanned<Expr>) -> AYResult<(Term, TlaSort)> {
        if let Ok(t) = self.translate_int(elem) {
            Ok((t, TlaSort::Int))
        } else if let Ok(t) = self.translate_string(elem) {
            Ok((t, TlaSort::String))
        } else {
            let t = self.translate_bool(elem)?;
            Ok((t, TlaSort::Bool))
        }
    }

    /// Translate a scalar sequence element to a term of the given (already
    /// inferred) element sort. Keeps subsequent literal elements consistent
    /// with the sort fixed by the first element.
    fn translate_native_elem_as(
        &mut self,
        elem: &Spanned<Expr>,
        elem_sort: &TlaSort,
    ) -> AYResult<Term> {
        match elem_sort {
            TlaSort::Bool => self.translate_bool(elem),
            TlaSort::String => self.translate_string(elem),
            _ => self.translate_int(elem),
        }
    }

    /// `Append(s, e)` -> `seq.++(s, seq.unit(e))`.
    fn native_append(
        &mut self,
        seq_expr: &Spanned<Expr>,
        elem_expr: &Spanned<Expr>,
    ) -> AYResult<(Term, TlaSort)> {
        let (s, elem_sort) = self.resolve_native_seq_term(seq_expr)?;
        let elem = self.translate_native_elem_as(elem_expr, &elem_sort)?;
        let unit = self.solver_mut().try_seq_unit(elem)?;
        let result = self.solver_mut().try_seq_concat(s, unit)?;
        Ok((result, elem_sort))
    }

    /// `s \o t` -> `seq.++(s, t)`.
    fn native_concat(
        &mut self,
        s_expr: &Spanned<Expr>,
        t_expr: &Spanned<Expr>,
    ) -> AYResult<(Term, TlaSort)> {
        let (s, elem_sort_s) = self.resolve_native_seq_term(s_expr)?;
        let (t, _elem_sort_t) = self.resolve_native_seq_term(t_expr)?;
        let result = self.solver_mut().try_seq_concat(s, t)?;
        Ok((result, elem_sort_s))
    }

    /// `Tail(s)` -> `seq.extract(s, 1, seq.len(s) - 1)` (drop the 0-indexed head).
    fn native_tail(&mut self, seq_expr: &Spanned<Expr>) -> AYResult<(Term, TlaSort)> {
        let (s, elem_sort) = self.resolve_native_seq_term(seq_expr)?;
        let len = self.solver_mut().try_seq_len(s)?;
        let one = self.solver_mut().int_const(1);
        let len_minus_1 = self.solver_mut().try_sub(len, one)?;
        let offset = self.solver_mut().int_const(1);
        let result = self.solver_mut().try_seq_extract(s, offset, len_minus_1)?;
        Ok((result, elem_sort))
    }

    /// `SubSeq(s, m, n)` -> `seq.extract(s, m - 1, n - m + 1)` (1-indexed inclusive).
    fn native_subseq(
        &mut self,
        seq_expr: &Spanned<Expr>,
        m_expr: &Spanned<Expr>,
        n_expr: &Spanned<Expr>,
    ) -> AYResult<(Term, TlaSort)> {
        let (s, elem_sort) = self.resolve_native_seq_term(seq_expr)?;
        let m = self.translate_int(m_expr)?;
        let n = self.translate_int(n_expr)?;
        let one = self.solver_mut().int_const(1);
        // offset = m - 1
        let offset = self.solver_mut().try_sub(m, one)?;
        // len = n - m + 1
        let n_minus_m = self.solver_mut().try_sub(n, m)?;
        let sub_len = self.solver_mut().try_add(n_minus_m, one)?;
        let result = self.solver_mut().try_seq_extract(s, offset, sub_len)?;
        Ok((result, elem_sort))
    }
}

#[cfg(test)]
mod tests {
    use ay_dpll::api::SolveResult;

    use super::*;

    /// Helper: create a translator with array support and declare a sequence variable.
    fn setup_int_seq(name: &str, max_len: usize) -> AYTranslator {
        let mut trans = AYTranslator::new_with_arrays();
        trans
            .declare_seq_var(name, TlaSort::Int, max_len)
            .expect("declare_seq_var should succeed");
        trans
    }

    fn setup_bool_seq(name: &str, max_len: usize) -> AYTranslator {
        let mut trans = AYTranslator::new_with_arrays();
        trans
            .declare_seq_var(name, TlaSort::Bool, max_len)
            .expect("declare_seq_var should succeed");
        trans
    }

    fn make_int_expr(val: i64) -> Spanned<Expr> {
        Spanned::new(Expr::Int(num_bigint::BigInt::from(val)), Default::default())
    }

    fn make_ident_expr(name: &str) -> Spanned<Expr> {
        Spanned::new(
            Expr::Ident(name.to_string(), tla_core::name_intern::NameId::INVALID),
            Default::default(),
        )
    }

    // -----------------------------------------------------------------
    // s[i] indexing tests
    // -----------------------------------------------------------------

    #[test]
    fn test_seq_apply_int_constant_index() {
        let mut trans = setup_int_seq("s", 3);
        let seq_info = trans.get_seq_var("s").unwrap().clone();
        let arg = make_int_expr(2);
        let result = trans.translate_seq_apply_int("s", &seq_info, &arg);
        assert!(result.is_ok(), "constant index should succeed");
    }

    #[test]
    fn test_seq_apply_int_out_of_bounds() {
        let mut trans = setup_int_seq("s", 3);
        let seq_info = trans.get_seq_var("s").unwrap().clone();
        let arg = make_int_expr(5);
        let result = trans.translate_seq_apply_int("s", &seq_info, &arg);
        assert!(result.is_err(), "index 5 on max_len=3 should fail");
    }

    #[test]
    fn test_seq_apply_int_zero_index() {
        let mut trans = setup_int_seq("s", 3);
        let seq_info = trans.get_seq_var("s").unwrap().clone();
        let arg = make_int_expr(0);
        let result = trans.translate_seq_apply_int("s", &seq_info, &arg);
        assert!(result.is_err(), "index 0 should fail (1-indexed)");
    }

    #[test]
    fn test_seq_apply_bool_type_mismatch() {
        let mut trans = setup_int_seq("s", 3);
        let seq_info = trans.get_seq_var("s").unwrap().clone();
        let arg = make_int_expr(1);
        let result = trans.translate_seq_apply_bool("s", &seq_info, &arg);
        assert!(result.is_err(), "Bool access on Int seq should fail");
    }

    #[test]
    fn test_seq_apply_bool_constant_index() {
        let mut trans = setup_bool_seq("b", 3);
        let seq_info = trans.get_seq_var("b").unwrap().clone();
        let arg = make_int_expr(1);
        let result = trans.translate_seq_apply_bool("b", &seq_info, &arg);
        assert!(result.is_ok(), "Bool access on Bool seq should succeed");
    }

    #[test]
    fn test_seq_apply_int_dynamic_index() {
        let mut trans = setup_int_seq("s", 3);
        // Declare an index variable
        trans.declare_var("idx", TlaSort::Int).expect("declare idx");
        let seq_info = trans.get_seq_var("s").unwrap().clone();
        let arg = make_ident_expr("idx");
        let result = trans.translate_seq_apply_int("s", &seq_info, &arg);
        assert!(result.is_ok(), "dynamic index should build ITE chain");
    }

    // -----------------------------------------------------------------
    // Len tests
    // -----------------------------------------------------------------

    #[test]
    fn test_len_returns_length_term() {
        let mut trans = setup_int_seq("s", 5);
        let seq_expr = make_ident_expr("s");
        let len_term = trans.translate_len(&seq_expr);
        assert!(len_term.is_ok(), "Len(s) should succeed");
    }

    #[test]
    fn test_len_constrained_sat() {
        let mut trans = setup_int_seq("s", 5);
        let seq_expr = make_ident_expr("s");
        let len_term = trans.translate_len(&seq_expr).unwrap();

        // Assert Len(s) = 3
        let three = trans.solver_mut().int_const(3);
        let eq = trans.solver_mut().try_eq(len_term, three).unwrap();
        trans.assert(eq);

        assert_eq!(trans.check_sat(), SolveResult::Sat);
    }

    #[test]
    fn test_len_exceeds_max_unsat() {
        let mut trans = setup_int_seq("s", 5);
        let seq_expr = make_ident_expr("s");
        let len_term = trans.translate_len(&seq_expr).unwrap();

        // Assert Len(s) = 10 (max_len=5, so unsat)
        let ten = trans.solver_mut().int_const(10);
        let eq = trans.solver_mut().try_eq(len_term, ten).unwrap();
        trans.assert(eq);

        assert!(matches!(trans.check_sat(), SolveResult::Unsat(_)));
    }

    // -----------------------------------------------------------------
    // Head tests
    // -----------------------------------------------------------------

    #[test]
    fn test_head_int() {
        let mut trans = setup_int_seq("s", 3);
        let seq_expr = make_ident_expr("s");
        let head = trans.translate_head_int(&seq_expr);
        assert!(head.is_ok(), "Head(s) for Int seq should succeed");
    }

    #[test]
    fn test_head_bool() {
        let mut trans = setup_bool_seq("b", 3);
        let seq_expr = make_ident_expr("b");
        let head = trans.translate_head_bool(&seq_expr);
        assert!(head.is_ok(), "Head(b) for Bool seq should succeed");
    }

    #[test]
    fn test_head_equals_first_element_sat() {
        let mut trans = setup_int_seq("s", 3);

        // Set s__1 = 42
        let s1 = *trans
            .get_seq_var("s")
            .unwrap()
            .element_terms
            .get(&1)
            .unwrap();
        let forty_two = trans.solver_mut().int_const(42);
        let eq1 = trans.solver_mut().try_eq(s1, forty_two).unwrap();
        trans.assert(eq1);

        // Assert Len(s) >= 1
        let len = trans.get_seq_var("s").unwrap().len_term;
        let one = trans.solver_mut().int_const(1);
        let ge = trans.solver_mut().try_ge(len, one).unwrap();
        trans.assert(ge);

        // Head(s) should equal 42
        let seq_expr = make_ident_expr("s");
        let head = trans.translate_head_int(&seq_expr).unwrap();
        let eq_head = trans.solver_mut().try_eq(head, forty_two).unwrap();
        trans.assert(eq_head);

        assert_eq!(trans.check_sat(), SolveResult::Sat);
    }

    // -----------------------------------------------------------------
    // seq_var_to_seq_term bridge test
    // -----------------------------------------------------------------

    #[test]
    fn test_seq_var_to_seq_term_roundtrip() {
        let mut trans = setup_int_seq("s", 3);
        let info = trans.get_seq_var("s").unwrap().clone();
        let result = trans.seq_var_to_seq_term(&info, &Sort::Int);
        assert!(result.is_ok(), "bridge should succeed");

        let seq = result.unwrap();
        // The length should be the same term
        // The array should be a store chain
        assert_eq!(trans.check_sat(), SolveResult::Sat);

        // Use the encoder to get Head and assert it equals s__1
        let enc = SequenceEncoder::new(Sort::Int);
        let head = enc.encode_head(&mut trans, &seq).unwrap();
        let s1 = info.element_terms.get(&1).copied().unwrap();
        let eq = trans.solver_mut().try_eq(head, s1).unwrap();
        trans.assert(eq);
        assert_eq!(trans.check_sat(), SolveResult::Sat);
    }

    // -----------------------------------------------------------------
    // resolve_seq_term error cases
    // -----------------------------------------------------------------

    #[test]
    fn test_resolve_seq_term_unknown_var() {
        let mut trans = AYTranslator::new_with_arrays();
        let expr = make_ident_expr("nonexistent");
        let result = trans.resolve_seq_term(&expr);
        assert!(result.is_err(), "unknown seq var should fail");
    }

    #[test]
    fn test_resolve_seq_term_non_ident() {
        let mut trans = setup_int_seq("s", 3);
        let expr = make_int_expr(42);
        let result = trans.resolve_seq_term(&expr);
        assert!(result.is_err(), "non-Ident should fail");
    }
}

// =========================================================================
// Option-A native unbounded Seq tests.
//
// These build a translator via `new_with_seq()` (native_seq = true,
// Logic::All), construct TLA+ sequence expressions, and assert CORRECT
// (sound) TLA+ semantics — NOT array-encoding parity. In particular,
// `Len(s) > 100` is SAT here (unbounded), where the bounded array encoding
// would report UNSAT.
// =========================================================================
#[cfg(test)]
mod native_seq_tests {
    use ay_dpll::api::SolveResult;
    use num_bigint::BigInt;
    use tla_core::ast::Expr;
    use tla_core::name_intern::NameId;
    use tla_core::Spanned;

    use super::super::{AYTranslator, TlaSort};

    // ----- AST builders -----

    fn int(v: i64) -> Spanned<Expr> {
        Spanned::new(Expr::Int(BigInt::from(v)), Default::default())
    }

    fn ident(name: &str) -> Spanned<Expr> {
        Spanned::new(
            Expr::Ident(name.to_string(), NameId::INVALID),
            Default::default(),
        )
    }

    fn str_lit(s: &str) -> Spanned<Expr> {
        Spanned::new(Expr::String(s.to_string()), Default::default())
    }

    /// `<<s1, s2, ...>>` over string literals.
    fn str_seq(vals: &[&str]) -> Spanned<Expr> {
        tuple(vals.iter().copied().map(str_lit).collect())
    }

    fn tuple(elems: Vec<Spanned<Expr>>) -> Spanned<Expr> {
        Spanned::new(Expr::Tuple(elems), Default::default())
    }

    fn apply(op: &str, args: Vec<Spanned<Expr>>) -> Spanned<Expr> {
        Spanned::new(Expr::Apply(Box::new(ident(op)), args), Default::default())
    }

    fn func_apply(base: Spanned<Expr>, arg: Spanned<Expr>) -> Spanned<Expr> {
        Spanned::new(
            Expr::FuncApply(Box::new(base), Box::new(arg)),
            Default::default(),
        )
    }

    fn eq(l: Spanned<Expr>, r: Spanned<Expr>) -> Spanned<Expr> {
        Spanned::new(Expr::Eq(Box::new(l), Box::new(r)), Default::default())
    }

    fn gt(l: Spanned<Expr>, r: Spanned<Expr>) -> Spanned<Expr> {
        Spanned::new(Expr::Gt(Box::new(l), Box::new(r)), Default::default())
    }

    /// `<<v1, v2, ...>>` over integer literals.
    fn int_seq(vals: &[i64]) -> Spanned<Expr> {
        tuple(vals.iter().copied().map(int).collect())
    }

    /// Translate a Bool-valued expr, assert it, and return the sat result.
    fn check(trans: &mut AYTranslator, expr: &Spanned<Expr>) -> SolveResult {
        let term = trans.translate_bool(expr).expect("translate_bool");
        trans.assert(term);
        trans.check_sat()
    }

    fn assert_sat(trans: &mut AYTranslator, expr: &Spanned<Expr>) {
        assert_eq!(
            check(trans, expr),
            SolveResult::Sat,
            "expected SAT for {expr:?}"
        );
    }

    fn assert_unsat(trans: &mut AYTranslator, expr: &Spanned<Expr>) {
        assert!(
            matches!(check(trans, expr), SolveResult::Unsat(_)),
            "expected UNSAT for {expr:?}",
        );
    }

    // ----- Len -----

    #[cfg_attr(test, ntest::timeout(20000))]
    #[test]
    fn native_len_literal_correct() {
        // Len(<<10,20,30>>) = 3 is SAT.
        let mut trans = AYTranslator::new_with_seq();
        assert_sat(
            &mut trans,
            &eq(apply("Len", vec![int_seq(&[10, 20, 30])]), int(3)),
        );
    }

    #[cfg_attr(test, ntest::timeout(20000))]
    #[test]
    fn native_len_literal_wrong_unsat() {
        // Len(<<10,20,30>>) = 4 is UNSAT.
        let mut trans = AYTranslator::new_with_seq();
        assert_unsat(
            &mut trans,
            &eq(apply("Len", vec![int_seq(&[10, 20, 30])]), int(4)),
        );
    }

    #[cfg_attr(test, ntest::timeout(20000))]
    #[test]
    fn native_empty_len_zero() {
        // Len(<<>>) = 0 is SAT; Len(<<>>) = 1 is UNSAT.
        let mut sat = AYTranslator::new_with_seq();
        assert_sat(&mut sat, &eq(apply("Len", vec![tuple(vec![])]), int(0)));

        let mut unsat = AYTranslator::new_with_seq();
        assert_unsat(&mut unsat, &eq(apply("Len", vec![tuple(vec![])]), int(1)));
    }

    // ----- Unbounded length (the Option-A verdict change) -----

    #[cfg_attr(test, ntest::timeout(20000))]
    #[test]
    fn native_len_unbounded_sat() {
        // Give the native variable a concrete 128-element witness.  AY's model
        // finder cannot currently synthesize a free sequence from only
        // `Len(s) > 100` and returns Unknown, but it can validate this explicit
        // native `seq.++` witness.  Reusing each concat result keeps the term a
        // small DAG: seven doublings grow a singleton to length 128.  The nominal
        // max_len of 10 is deliberately exceeded: it is ignored on the native
        // path, whereas the bounded array encoding would make this UNSAT.
        let mut trans = AYTranslator::new_with_seq();
        trans
            .declare_seq_var("s", TlaSort::Int, 10)
            .expect("declare native seq");

        let zero = trans.solver_mut().int_const(0);
        let mut witness = trans
            .solver_mut()
            .try_seq_unit(zero)
            .expect("build singleton witness");
        for _ in 0..7 {
            witness = trans
                .solver_mut()
                .try_seq_concat(witness, witness)
                .expect("double native witness");
        }
        let s = trans.get_native_seq_var("s").expect("native seq info").term;
        let bind_witness = trans
            .solver_mut()
            .try_eq(s, witness)
            .expect("bind native witness");
        trans.assert(bind_witness);

        assert_sat(&mut trans, &gt(apply("Len", vec![ident("s")]), int(100)));
    }

    // ----- Indexing (1-indexed TLA+) -----

    #[cfg_attr(test, ntest::timeout(20000))]
    #[test]
    fn native_index_literal() {
        // <<10,20,30>>[2] = 20 (SAT); = 99 (UNSAT).
        let mut sat = AYTranslator::new_with_seq();
        assert_sat(
            &mut sat,
            &eq(func_apply(int_seq(&[10, 20, 30]), int(2)), int(20)),
        );

        let mut unsat = AYTranslator::new_with_seq();
        assert_unsat(
            &mut unsat,
            &eq(func_apply(int_seq(&[10, 20, 30]), int(2)), int(99)),
        );
    }

    #[cfg_attr(test, ntest::timeout(20000))]
    #[test]
    fn native_concat_index() {
        // (<<1,2>> \o <<3>>)[3] = 3 is SAT; = 2 is UNSAT.
        let cat = || apply("\\o", vec![int_seq(&[1, 2]), int_seq(&[3])]);

        let mut sat = AYTranslator::new_with_seq();
        assert_sat(&mut sat, &eq(func_apply(cat(), int(3)), int(3)));

        let mut unsat = AYTranslator::new_with_seq();
        assert_unsat(&mut unsat, &eq(func_apply(cat(), int(3)), int(2)));
    }

    #[cfg_attr(test, ntest::timeout(20000))]
    #[test]
    fn native_concat_len() {
        // Len(<<1,2>> \o <<3,4,5>>) = 5 is SAT.
        let mut trans = AYTranslator::new_with_seq();
        let cat = apply("\\o", vec![int_seq(&[1, 2]), int_seq(&[3, 4, 5])]);
        assert_sat(&mut trans, &eq(apply("Len", vec![cat]), int(5)));
    }

    // ----- Head -----

    #[cfg_attr(test, ntest::timeout(20000))]
    #[test]
    fn native_head() {
        // Head(<<7,8,9>>) = 7 (SAT); = 8 (UNSAT).
        let mut sat = AYTranslator::new_with_seq();
        assert_sat(
            &mut sat,
            &eq(apply("Head", vec![int_seq(&[7, 8, 9])]), int(7)),
        );

        let mut unsat = AYTranslator::new_with_seq();
        assert_unsat(
            &mut unsat,
            &eq(apply("Head", vec![int_seq(&[7, 8, 9])]), int(8)),
        );
    }

    // ----- Tail -----

    #[cfg_attr(test, ntest::timeout(20000))]
    #[test]
    fn native_tail_len_and_index() {
        // Tail(<<1,2,3>>): Len = 2 (SAT), Len = 3 (UNSAT), [1] = 2 (SAT).
        let tail = || apply("Tail", vec![int_seq(&[1, 2, 3])]);

        let mut len_sat = AYTranslator::new_with_seq();
        assert_sat(&mut len_sat, &eq(apply("Len", vec![tail()]), int(2)));

        let mut len_unsat = AYTranslator::new_with_seq();
        assert_unsat(&mut len_unsat, &eq(apply("Len", vec![tail()]), int(3)));

        let mut idx = AYTranslator::new_with_seq();
        assert_sat(&mut idx, &eq(func_apply(tail(), int(1)), int(2)));
    }

    #[cfg_attr(test, ntest::timeout(20000))]
    #[test]
    fn native_tail_equality() {
        // Tail(<<1,2,3>>) = <<2,3>> is SAT (native term equality).
        let mut trans = AYTranslator::new_with_seq();
        assert_sat(
            &mut trans,
            &eq(apply("Tail", vec![int_seq(&[1, 2, 3])]), int_seq(&[2, 3])),
        );
    }

    // ----- Append -----

    #[cfg_attr(test, ntest::timeout(20000))]
    #[test]
    fn native_append_len_and_value() {
        // Append(<<1,2>>, 3): Len = 3 (SAT), [3] = 3 (SAT), [1] = 1 (SAT).
        let app = || apply("Append", vec![int_seq(&[1, 2]), int(3)]);

        let mut len = AYTranslator::new_with_seq();
        assert_sat(&mut len, &eq(apply("Len", vec![app()]), int(3)));

        let mut last = AYTranslator::new_with_seq();
        assert_sat(&mut last, &eq(func_apply(app(), int(3)), int(3)));

        let mut first = AYTranslator::new_with_seq();
        assert_sat(&mut first, &eq(func_apply(app(), int(1)), int(1)));
    }

    // ----- SubSeq -----

    #[cfg_attr(test, ntest::timeout(20000))]
    #[test]
    fn native_subseq_len_and_elements() {
        // SubSeq(<<10,20,30,40>>, 2, 3) => <<20,30>>:
        //   Len = 2 (SAT), [1] = 20 (SAT), [2] = 30 (SAT).
        let sub = || apply("SubSeq", vec![int_seq(&[10, 20, 30, 40]), int(2), int(3)]);

        let mut len = AYTranslator::new_with_seq();
        assert_sat(&mut len, &eq(apply("Len", vec![sub()]), int(2)));

        let mut e1 = AYTranslator::new_with_seq();
        assert_sat(&mut e1, &eq(func_apply(sub(), int(1)), int(20)));

        let mut e2 = AYTranslator::new_with_seq();
        assert_sat(&mut e2, &eq(func_apply(sub(), int(2)), int(30)));
    }

    // ----- Equality -----

    #[cfg_attr(test, ntest::timeout(20000))]
    #[test]
    fn native_seq_equality() {
        // <<1,2,3>> = <<1,2,3>> is SAT.
        let mut same = AYTranslator::new_with_seq();
        assert_sat(&mut same, &eq(int_seq(&[1, 2, 3]), int_seq(&[1, 2, 3])));

        // <<1,2>> = <<1,2,3>> is UNSAT (different lengths).
        let mut diff_len = AYTranslator::new_with_seq();
        assert_unsat(&mut diff_len, &eq(int_seq(&[1, 2]), int_seq(&[1, 2, 3])));
    }

    // ----- String-element sequence literals (FOLLOW-UP 2) -----

    #[cfg_attr(test, ntest::timeout(20000))]
    #[test]
    fn native_string_seq_len() {
        // Len(<<"a","b","c">>) = 3 is SAT (string-element literal builds natively);
        // = 5 is UNSAT.
        let mut sat = AYTranslator::new_with_seq();
        assert_sat(
            &mut sat,
            &eq(apply("Len", vec![str_seq(&["a", "b", "c"])]), int(3)),
        );

        let mut unsat = AYTranslator::new_with_seq();
        assert_unsat(
            &mut unsat,
            &eq(apply("Len", vec![str_seq(&["a", "b", "c"])]), int(5)),
        );
    }

    #[cfg_attr(test, ntest::timeout(20000))]
    #[test]
    fn native_string_seq_equality_same() {
        // <<"a","b">> = <<"a","b">> is SAT (string-element literals build natively
        // and route through native seq equality).
        let mut same = AYTranslator::new_with_seq();
        assert_sat(&mut same, &eq(str_seq(&["a", "b"]), str_seq(&["a", "b"])));
    }

    /// Multi-element native sequence equality is SOUND.
    ///
    /// `<<1,2>> = <<1,3>>` is UNSAT. This previously returned SAT — ay's seq
    /// theory treated `seq.++` (concatenation) as opaque EUF and never decided
    /// concat equality element-wise. Fixed upstream by the seq concat-extensionality
    /// decision rule (ay `fix(seq): decide seq.++ equality via concat extensionality`,
    /// pulled in by the ay pin bump), which unblocked flipping native_seq on.
    #[cfg_attr(test, ntest::timeout(20000))]
    #[test]
    fn native_seq_equality_multi_element_sound() {
        let mut diff = AYTranslator::new_with_seq();
        // TLA+ semantics: distinct-element sequences of equal length are unequal.
        assert_unsat(&mut diff, &eq(int_seq(&[1, 2]), int_seq(&[1, 3])));
    }

    // ----- Fall-through: Len/Head on a non-native seq var (FOLLOW-UP 1) -----

    #[cfg_attr(test, ntest::timeout(20000))]
    #[test]
    fn native_len_falls_through_to_array_path() {
        // A translator with native_seq ON but a sequence var registered ONLY on
        // the bounded array path (simulating a non-native declaration). Len must
        // FALL THROUGH to the array translate_len rather than erroring.
        let mut trans = AYTranslator::new_with_seq();
        // Manually register an array-path seq var (bypasses the native path that
        // declare_seq_var would otherwise take for a scalar element).
        trans
            .declare_array_seq_var_for_test("s", TlaSort::Int, 5)
            .expect("declare array-path seq var");
        // Len(s) = 3 should be SAT via the array path fall-through.
        assert_sat(&mut trans, &eq(apply("Len", vec![ident("s")]), int(3)));
    }

    // ----- Declaration: native path uses no per-position vars -----

    #[cfg_attr(test, ntest::timeout(20000))]
    #[test]
    fn native_declare_registers_native_var_only() {
        let mut trans = AYTranslator::new_with_seq();
        trans
            .declare_seq_var("s", TlaSort::Int, 5)
            .expect("declare native seq");
        assert!(trans.is_native_seq());
        assert!(trans.get_native_seq_var("s").is_some());
        // No bounded SequenceVarInfo was created on the native path.
        assert!(trans.get_seq_var("s").is_err());
    }
}
