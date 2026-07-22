// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Finite-domain `FuncSet` (`[D -> R]`) enumeration for the BMC translator.
//!
//! Encodes a bounded existential `\E f \in [D -> R] : P(f)` (and the membership
//! `elem \in [D -> R]` rewritten through it) by **exhaustively enumerating the
//! concrete function table**. For each function `f : D -> R`, the bound variable
//! is substituted by its concrete value and the body is translated; the results
//! are OR-combined (`\E`) or AND-combined (`\A`).
//!
//! This is the class TY loses to Apalache on (e.g. the Einstein/Zebra riddle,
//! whose `Permutation(S)` idiom is `{FunAsSeq(p, n, n) : p \in [1..n -> S]}`).
//!
//! ## Soundness
//!
//! Exhaustive concrete enumeration is **exact** — every function in `[D -> R]`
//! is one of the substituted values, and each substituted body is translated by
//! the ordinary (already-validated) expression translator. No Skolem constant or
//! abstraction is introduced, so the encoding is faithful in both directions.
//! The BMC violation path is additionally cross-validated (a found CEX is
//! replayed through the interpreter), so this is the netted direction.
//!
//! ## Function-as-value substitution
//!
//! A function over a contiguous `1..n` domain is the sequence
//! `<<f[1], ..., f[n]>>`. The [`FuncTableSubstitute`] folder rewrites, for the
//! bound variable `p`:
//!   * `p[k]`            (literal key)  -> the concrete range value at `k`
//!   * `FunAsSeq(p,_,_)`               -> `<<v_1, ..., v_n>>`
//!   * bare `p`                        -> `<<v_1, ..., v_n>>`
//! so the body becomes fully concrete and is handled by the existing tuple /
//! string / membership machinery.

use std::collections::HashMap;

use ay_dpll::api::Term;
use num_bigint::BigInt;
use tla_core::ast::{BoundVar, Expr, OperatorDef};
use tla_core::{ExprFold, SpanPolicy, Spanned, SubstituteExpr};

use crate::error::{AYError, AYResult};

use super::BmcTranslator;

/// Maximum number of concrete functions enumerated for a single `[D -> R]`
/// existential/universal. `|R|^|D|` grows fast; the cap keeps the formula size
/// (and translation time) bounded. The Einstein riddle uses `5^5 = 3125` per
/// variable, comfortably under this limit.
const MAX_FUNCSET_ENUM: usize = 100_000;

/// A folder that substitutes a concrete function value for a bound function
/// variable. The function is described by `keys` (its domain, in order) and
/// `values` (the corresponding range value expression for each key).
struct FuncTableSubstitute<'a> {
    /// Name of the bound function variable being replaced.
    var: &'a str,
    /// Domain keys, in order (e.g. integers `1..n`).
    keys: &'a [i64],
    /// Range value expression for each key (parallel to `keys`).
    values: &'a [Spanned<Expr>],
}

impl<'a> FuncTableSubstitute<'a> {
    /// The function as a literal sequence `<<v_1, ..., v_n>>`. Only meaningful
    /// when the domain is the contiguous `1..n`; callers gate on that.
    fn as_tuple(&self) -> Expr {
        Expr::Tuple(self.values.to_vec())
    }

    /// Look up the value for a literal key, if the key is in the domain.
    fn value_for_key(&self, key: i64) -> Option<&Spanned<Expr>> {
        self.keys
            .iter()
            .position(|&k| k == key)
            .map(|i| &self.values[i])
    }

    /// Is `expr` a reference to the bound function variable `self.var`?
    fn is_var(&self, expr: &Expr) -> bool {
        matches!(expr, Expr::Ident(name, _) | Expr::StateVar(name, ..) if name == self.var)
    }
}

impl<'a> ExprFold for FuncTableSubstitute<'a> {
    fn fold_expr(&mut self, expr: Spanned<Expr>) -> Spanned<Expr> {
        let span = expr.span;
        match expr.node {
            // p[k] with a literal key -> concrete range value.
            Expr::FuncApply(func, arg) => {
                if self.is_var(&func.node) {
                    if let Expr::Int(n) = &arg.node {
                        if let Ok(k) = i64::try_from(n) {
                            if let Some(v) = self.value_for_key(k) {
                                return v.clone();
                            }
                        }
                    }
                }
                // Recurse structurally (the argument may itself reference p).
                Spanned::new(
                    Expr::FuncApply(self.fold_box(func), self.fold_box(arg)),
                    span,
                )
            }
            // FunAsSeq(p, _, _) -> <<v_1, ..., v_n>>.
            Expr::Apply(op, args) => {
                let is_funasseq = matches!(
                    &op.node,
                    Expr::Ident(name, _) | Expr::OpRef(name) if name == "FunAsSeq"
                );
                if is_funasseq {
                    if let Some(first) = args.first() {
                        if self.is_var(&first.node) {
                            return Spanned::new(self.as_tuple(), span);
                        }
                    }
                }
                Spanned::new(Expr::Apply(self.fold_box(op), self.fold_vec(args)), span)
            }
            // Bare `p` in a value position -> <<v_1, ..., v_n>>.
            Expr::Ident(ref name, _) if name == self.var => Spanned::new(self.as_tuple(), span),
            Expr::StateVar(ref name, ..) if name == self.var => Spanned::new(self.as_tuple(), span),
            other => Spanned::new(self.fold_expr_inner(other), span),
        }
    }
}

// ---------------------------------------------------------------------------
// Concrete sequence/function reduction
// ---------------------------------------------------------------------------
//
// After FuncSet enumeration substitutes a concrete function `p` (as a literal
// tuple `<<v_1, ..., v_n>>`), permutation specs leave a fully-concrete
// "sequence builder" chain around it — e.g. the Apalache `FunAsSeq(p, n, n)`
// lowers to
//
//     LET ctor(__i) == p[__i] IN SubSeq(MkSeq(n, ctor), 1, len)
//     MkSeq(__N, __F) == [__i \in 1..__N |-> __F(__i)]
//
// which (with `p` concrete) is the *value* `<<p[1], ..., p[len]>>`. This pass
// reduces such fully-concrete constructs to a literal `Expr::Tuple` so the
// surrounding equality (`drinks = <that tuple>`) becomes plain tuple-literal
// equality, which the existing translator already handles.
//
// SOUNDNESS: this is pure constant folding — it only rewrites a construct to an
// equivalent literal when *every* relevant sub-expression is already a concrete
// literal. Any non-literal leaves the construct unchanged (`reduce_*` returns
// `None`), so the normal translator path is used and no wrong value is ever
// synthesized.

/// Maximum length of a sequence/tuple produced by concrete reduction. Bounds
/// the work and the resulting formula size.
const MAX_REDUCED_SEQ_LEN: i64 = 4096;

/// Recursively reduce fully-concrete sequence/function-builder constructs in
/// `expr` to literal tuples. Leaves any non-reducible sub-expression untouched.
///
/// `ops` carries `LET`-bound operator definitions currently in scope so that
/// `FunAsSeq`'s `LET ctor(__i) == ... IN ...` element constructor can be
/// applied during reduction.
fn reduce_concrete_constructs(
    expr: &Spanned<Expr>,
    ops: &HashMap<String, OperatorDef>,
) -> Spanned<Expr> {
    let span = expr.span;
    match &expr.node {
        // LET defs IN body — add the (param-carrying) operator defs to scope and
        // reduce the body. The defs themselves are not emitted; if the body
        // reduces to a literal the LET is gone, otherwise we rebuild it below.
        Expr::Let(defs, body) => {
            let mut extended = ops.clone();
            for d in defs {
                extended.insert(d.name.node.clone(), d.clone());
            }
            let reduced_body = reduce_concrete_constructs(body, &extended);
            // If the body fully reduced to a value with no remaining reference
            // to the LET-bound ops, drop the LET. Conservatively, only drop it
            // when the result is a literal tuple / scalar (no free op refs).
            if is_concrete_value(&reduced_body) {
                reduced_body
            } else {
                Spanned::new(Expr::Let(defs.clone(), Box::new(reduced_body)), span)
            }
        }

        // SubSeq(seq, lo, hi) — reduce `seq` to a literal tuple, then slice.
        Expr::Apply(op, args) if is_op(op, "SubSeq") && args.len() == 3 => {
            if let Some(t) = reduce_subseq(&args[0], &args[1], &args[2], ops) {
                return t;
            }
            rebuild_apply(op, args, ops, span)
        }

        // MkSeq(n, F) — build the tuple <<F(1), ..., F(n)>> when n and F are
        // concrete. F is an operator name bound in `ops`.
        Expr::Apply(op, args) if is_op(op, "MkSeq") && args.len() == 2 => {
            if let Some(t) = reduce_mkseq(&args[0], &args[1], ops) {
                return t;
            }
            rebuild_apply(op, args, ops, span)
        }

        // Application of a LET-bound operator: ctor(arg) -> body[param := arg].
        Expr::Apply(op, args) => {
            if let Expr::Ident(name, _) | Expr::OpRef(name) = &op.node {
                if let Some(def) = ops.get(name) {
                    if def.params.len() == args.len() {
                        let reduced_args: Vec<Spanned<Expr>> = args
                            .iter()
                            .map(|a| reduce_concrete_constructs(a, ops))
                            .collect();
                        let substituted = substitute_op_params(def, &reduced_args);
                        return reduce_concrete_constructs(&substituted, ops);
                    }
                }
            }
            rebuild_apply(op, args, ops, span)
        }

        // FuncDef [v \in 1..n |-> body] -> <<body[v:=1], ..., body[v:=n]>> when
        // the domain is a concrete contiguous 1..n range.
        Expr::FuncDef(bounds, body) => {
            if let Some(t) = reduce_funcdef_to_tuple(bounds, body, ops) {
                return t;
            }
            Spanned::new(
                Expr::FuncDef(
                    bounds.clone(),
                    Box::new(reduce_concrete_constructs(body, ops)),
                ),
                span,
            )
        }

        // (<<v1, ..., vn>>)[k] -> v_k for a concrete integer index k.
        Expr::FuncApply(func, arg) => {
            let rf = reduce_concrete_constructs(func, ops);
            let ra = reduce_concrete_constructs(arg, ops);
            if let (Expr::Tuple(elems), Expr::Int(idx)) = (&rf.node, &ra.node) {
                if let Ok(i) = i64::try_from(idx) {
                    if i >= 1 && (i as usize) <= elems.len() {
                        return elems[(i - 1) as usize].clone();
                    }
                }
            }
            Spanned::new(Expr::FuncApply(Box::new(rf), Box::new(ra)), span)
        }

        // Tuple literal: reduce each element (e.g. nested SubSeq).
        Expr::Tuple(elems) => Spanned::new(
            Expr::Tuple(
                elems
                    .iter()
                    .map(|e| reduce_concrete_constructs(e, ops))
                    .collect(),
            ),
            span,
        ),

        // Everything else: structurally recurse into children so the value
        // reducers above are reached wherever a sequence-builder chain appears
        // (e.g. the right-hand side of `drinks = FunAsSeq(...)`, nested inside
        // the conjunction). This recursion only *rewrites* the recognised
        // constructs; all other nodes are rebuilt identically.
        _ => descend_reduce(expr, ops),
    }
}

/// Structurally recurse `reduce_concrete_constructs` into the children of
/// `expr`, rebuilding the same node. Covers the boolean / set / arithmetic /
/// comparison shapes that wrap a value-position sequence builder. Binders that
/// would need scope handling (`Let`, `FuncDef`, quantifiers) are handled by the
/// dedicated arms in `reduce_concrete_constructs`, so here we conservatively
/// leave any *other* binder untouched to avoid unsound capture.
fn descend_reduce(expr: &Spanned<Expr>, ops: &HashMap<String, OperatorDef>) -> Spanned<Expr> {
    let span = expr.span;
    let r = |e: &Spanned<Expr>| Box::new(reduce_concrete_constructs(e, ops));
    let node = match &expr.node {
        Expr::And(a, b) => Expr::And(r(a), r(b)),
        Expr::Or(a, b) => Expr::Or(r(a), r(b)),
        Expr::Not(a) => Expr::Not(r(a)),
        Expr::Implies(a, b) => Expr::Implies(r(a), r(b)),
        Expr::Equiv(a, b) => Expr::Equiv(r(a), r(b)),
        Expr::Eq(a, b) => Expr::Eq(r(a), r(b)),
        Expr::Neq(a, b) => Expr::Neq(r(a), r(b)),
        Expr::In(a, b) => Expr::In(r(a), r(b)),
        Expr::NotIn(a, b) => Expr::NotIn(r(a), r(b)),
        Expr::Lt(a, b) => Expr::Lt(r(a), r(b)),
        Expr::Leq(a, b) => Expr::Leq(r(a), r(b)),
        Expr::Gt(a, b) => Expr::Gt(r(a), r(b)),
        Expr::Geq(a, b) => Expr::Geq(r(a), r(b)),
        Expr::Add(a, b) => Expr::Add(r(a), r(b)),
        Expr::Sub(a, b) => Expr::Sub(r(a), r(b)),
        Expr::SetEnum(elems) => Expr::SetEnum(
            elems
                .iter()
                .map(|e| reduce_concrete_constructs(e, ops))
                .collect(),
        ),
        Expr::SetMinus(a, b) => Expr::SetMinus(r(a), r(b)),
        Expr::Union(a, b) => Expr::Union(r(a), r(b)),
        Expr::Intersect(a, b) => Expr::Intersect(r(a), r(b)),
        Expr::If(c, t, e) => Expr::If(r(c), r(t), r(e)),
        // Any other node (including binders we do not special-case): leave it
        // unchanged so we never risk variable capture or dropping scope.
        _ => return expr.clone(),
    };
    Spanned::new(node, span)
}

/// Is `op` the operator named `name` (`Ident` or `OpRef`)?
fn is_op(op: &Spanned<Expr>, name: &str) -> bool {
    matches!(&op.node, Expr::Ident(n, _) | Expr::OpRef(n) if n == name)
}

/// Rebuild an `Apply`, reducing its arguments. Used as the fall-through when a
/// recognised operator could not be fully reduced.
fn rebuild_apply(
    op: &Spanned<Expr>,
    args: &[Spanned<Expr>],
    ops: &HashMap<String, OperatorDef>,
    span: tla_core::span::Span,
) -> Spanned<Expr> {
    let reduced_args = args
        .iter()
        .map(|a| reduce_concrete_constructs(a, ops))
        .collect();
    Spanned::new(Expr::Apply(Box::new(op.clone()), reduced_args), span)
}

/// Substitute an operator's actual arguments for its formal parameters.
fn substitute_op_params(def: &OperatorDef, args: &[Spanned<Expr>]) -> Spanned<Expr> {
    let subs: HashMap<&str, &Spanned<Expr>> = def
        .params
        .iter()
        .zip(args.iter())
        .map(|(p, a)| (p.name.node.as_str(), a))
        .collect();
    let mut sub = SubstituteExpr {
        subs,
        span_policy: SpanPolicy::Preserve,
    };
    sub.fold_expr(def.body.clone())
}

/// Reduce `SubSeq(seq, lo, hi)` to the literal tuple of elements `lo..=hi`,
/// when `seq` reduces to a tuple and `lo`/`hi` are concrete integers in range.
fn reduce_subseq(
    seq: &Spanned<Expr>,
    lo: &Spanned<Expr>,
    hi: &Spanned<Expr>,
    ops: &HashMap<String, OperatorDef>,
) -> Option<Spanned<Expr>> {
    let seq_r = reduce_concrete_constructs(seq, ops);
    let Expr::Tuple(elems) = &seq_r.node else {
        return None;
    };
    let lo_i = concrete_int(lo, ops)?;
    let hi_i = concrete_int(hi, ops)?;
    if lo_i < 1 || hi_i > elems.len() as i64 || lo_i > hi_i + 1 {
        return None;
    }
    let slice: Vec<Spanned<Expr>> = ((lo_i)..=(hi_i))
        .map(|i| elems[(i - 1) as usize].clone())
        .collect();
    Some(Spanned::new(Expr::Tuple(slice), seq_r.span))
}

/// Reduce `MkSeq(n, F)` to `<<F(1), ..., F(n)>>` when `n` is a concrete integer
/// and `F` names a LET-bound unary operator.
fn reduce_mkseq(
    n: &Spanned<Expr>,
    f: &Spanned<Expr>,
    ops: &HashMap<String, OperatorDef>,
) -> Option<Spanned<Expr>> {
    let n_i = concrete_int(n, ops)?;
    if !(0..=MAX_REDUCED_SEQ_LEN).contains(&n_i) {
        return None;
    }
    let op_name = match &f.node {
        Expr::Ident(name, _) | Expr::OpRef(name) => name.clone(),
        _ => return None,
    };
    let def = ops.get(&op_name)?;
    if def.params.len() != 1 {
        return None;
    }
    let mut elems = Vec::with_capacity(n_i as usize);
    for i in 1..=n_i {
        let arg = Spanned::new(Expr::Int(BigInt::from(i)), f.span);
        let applied = substitute_op_params(def, std::slice::from_ref(&arg));
        elems.push(reduce_concrete_constructs(&applied, ops));
    }
    Some(Spanned::new(Expr::Tuple(elems), f.span))
}

/// Reduce a `FuncDef [v \in 1..n |-> body]` to a literal tuple
/// `<<body[v:=1], ..., body[v:=n]>>` for a concrete contiguous `1..n` domain.
fn reduce_funcdef_to_tuple(
    bounds: &[BoundVar],
    body: &Spanned<Expr>,
    ops: &HashMap<String, OperatorDef>,
) -> Option<Spanned<Expr>> {
    if bounds.len() != 1 {
        return None;
    }
    let bound = &bounds[0];
    let domain = bound.domain.as_ref()?;
    let (lo, hi) = match &domain.node {
        Expr::Range(lo, hi) => (concrete_int(lo, ops)?, concrete_int(hi, ops)?),
        _ => return None,
    };
    if lo != 1 || hi < 1 || hi > MAX_REDUCED_SEQ_LEN {
        return None;
    }
    let var_name = bound.name.node.clone();
    let mut elems = Vec::with_capacity(hi as usize);
    for i in lo..=hi {
        let idx = Spanned::new(Expr::Int(BigInt::from(i)), body.span);
        let subs = HashMap::from([(var_name.as_str(), &idx)]);
        let mut sub = SubstituteExpr {
            subs,
            span_policy: SpanPolicy::Preserve,
        };
        let substituted = sub.fold_expr(body.clone());
        elems.push(reduce_concrete_constructs(&substituted, ops));
    }
    Some(Spanned::new(Expr::Tuple(elems), body.span))
}

/// Extract a concrete `i64` from `expr`, reducing it first.
fn concrete_int(expr: &Spanned<Expr>, ops: &HashMap<String, OperatorDef>) -> Option<i64> {
    match &reduce_concrete_constructs(expr, ops).node {
        Expr::Int(n) => i64::try_from(n).ok(),
        _ => None,
    }
}

/// Is `expr` a fully-concrete value (a literal scalar or a tuple of concrete
/// values)? Used to decide whether a `LET` can be dropped after reduction.
fn is_concrete_value(expr: &Spanned<Expr>) -> bool {
    match &expr.node {
        Expr::Int(_) | Expr::Bool(_) | Expr::String(_) => true,
        Expr::Tuple(elems) => elems.iter().all(is_concrete_value),
        _ => false,
    }
}

impl BmcTranslator {
    /// Expand `\Q f \in [D -> R] : body` over a finite domain `D` and finite
    /// range `R` by exhaustive concrete enumeration of the function table.
    ///
    /// Returns the OR (for `\E`) / AND (for `\A`) over each concrete function's
    /// substituted body. The bound variable's domain must be a contiguous
    /// `1..n` range (so the function is a sequence) and the range must be a
    /// finite literal set.
    pub(super) fn expand_bmc_funcset_quantifier(
        &mut self,
        bound: &BoundVar,
        domain: &Spanned<Expr>,
        range: &Spanned<Expr>,
        body: &Spanned<Expr>,
        is_forall: bool,
    ) -> AYResult<Term> {
        let var_name = bound.name.node.clone();
        let keys = extract_contiguous_one_based_domain(domain)?;
        let range_values = extract_finite_range_values(range)?;

        let num_keys = keys.len();
        let num_values = range_values.len();

        // Empty domain: `[{} -> R]` has exactly one element (the empty
        // function); the body is evaluated with `p` bound to `<<>>`.
        if num_keys == 0 {
            return self.bmc_substitute_funcset(&var_name, &keys, &[], body);
        }
        if num_values == 0 {
            // `[D -> {}]` with non-empty D is the empty set: no functions.
            return Ok(self.solver.bool_const(is_forall));
        }

        // Guard the |R|^|D| blow-up.
        let total = checked_pow(num_values, num_keys).ok_or_else(|| {
            AYError::UnsupportedOp(format!(
                "FuncSet [D -> R] with |D|={num_keys}, |R|={num_values} \
                 exceeds enumeration limit"
            ))
        })?;
        if total > MAX_FUNCSET_ENUM {
            return Err(AYError::UnsupportedOp(format!(
                "FuncSet [D -> R] would enumerate {total} functions \
                 (|D|={num_keys}, |R|={num_values}); exceeds limit of {MAX_FUNCSET_ENUM}"
            )));
        }

        // Mixed-radix enumeration of all |R|^|D| functions: digit i selects the
        // range value for key i.
        let absorb = self.solver.bool_const(!is_forall);
        let mut results = Vec::with_capacity(total);
        let mut digits = vec![0usize; num_keys];
        'enumerate: loop {
            let chosen: Vec<Spanned<Expr>> =
                digits.iter().map(|&d| range_values[d].clone()).collect();
            let term = self.bmc_substitute_funcset(&var_name, &keys, &chosen, body)?;

            // Short-circuit: \E hits a true disjunct, \A hits a false conjunct.
            if term == absorb {
                return Ok(absorb);
            }
            results.push(term);

            // Increment the mixed-radix counter; carry out of the top digit
            // means every function has been enumerated.
            let mut i = 0;
            loop {
                if i == num_keys {
                    break 'enumerate;
                }
                digits[i] += 1;
                if digits[i] < num_values {
                    break;
                }
                digits[i] = 0;
                i += 1;
            }
        }

        self.combine_bool_terms(&results, is_forall)
    }

    /// Substitute the concrete function value for `var_name` in `body` and
    /// translate the result to a Bool term.
    fn bmc_substitute_funcset(
        &mut self,
        var_name: &str,
        keys: &[i64],
        values: &[Spanned<Expr>],
        body: &Spanned<Expr>,
    ) -> AYResult<Term> {
        let mut folder = FuncTableSubstitute {
            var: var_name,
            keys,
            values,
        };
        let substituted = folder.fold_expr(body.clone());
        // Constant-fold any concrete sequence/function-builder chain left around
        // the substituted function (e.g. `FunAsSeq(p, n, n)` lowered to
        // `SubSeq(MkSeq(n, \i. p[i]), 1, len)`) into a literal tuple, so the
        // surrounding equality becomes plain tuple-literal equality.
        let reduced = reduce_concrete_constructs(&substituted, &HashMap::new());
        self.translate_bool(&reduced)
    }
}

/// Extract a contiguous `1..n` (one-based) function domain as ordered integer
/// keys. Required so the function is a genuine sequence. Also accepts an empty
/// domain (`n = 0`).
fn extract_contiguous_one_based_domain(domain: &Spanned<Expr>) -> AYResult<Vec<i64>> {
    match &domain.node {
        Expr::Range(lo, hi) => {
            let lo = int_literal(lo)?;
            let hi = int_literal(hi)?;
            if lo != 1 {
                return Err(AYError::UnsupportedOp(format!(
                    "FuncSet domain must be 1..n for sequence encoding, got {lo}..{hi}"
                )));
            }
            if hi < 1 {
                return Ok(Vec::new());
            }
            Ok((1..=hi).collect())
        }
        Expr::SetEnum(elems) => {
            // Accept a literal set of integers iff it is exactly {1, ..., n}.
            let mut keys = Vec::with_capacity(elems.len());
            for e in elems {
                keys.push(int_literal(e)?);
            }
            keys.sort_unstable();
            keys.dedup();
            let is_one_based = keys.iter().enumerate().all(|(i, &k)| k == (i as i64) + 1);
            if !is_one_based {
                return Err(AYError::UnsupportedOp(
                    "FuncSet domain must be 1..n for sequence encoding".to_string(),
                ));
            }
            Ok(keys)
        }
        _ => Err(AYError::UnsupportedOp(format!(
            "FuncSet domain must be a 1..n range, got {:?}",
            std::mem::discriminant(&domain.node)
        ))),
    }
}

/// Extract the concrete value expressions of a finite range set `R`.
/// Supports `SetEnum` (of literals) and integer `Range`. The values are
/// returned as `Expr` so they can be substituted into the body and translated
/// by the ordinary scalar/string machinery.
fn extract_finite_range_values(range: &Spanned<Expr>) -> AYResult<Vec<Spanned<Expr>>> {
    match &range.node {
        Expr::SetEnum(elems) => Ok(elems.clone()),
        Expr::Range(lo, hi) => {
            let lo = int_literal(lo)?;
            let hi = int_literal(hi)?;
            if hi < lo {
                return Ok(Vec::new());
            }
            Ok((lo..=hi)
                .map(|v| Spanned::new(Expr::Int(BigInt::from(v)), range.span))
                .collect())
        }
        Expr::Ident(name, _) if name == "BOOLEAN" => Ok(vec![
            Spanned::new(Expr::Bool(false), range.span),
            Spanned::new(Expr::Bool(true), range.span),
        ]),
        _ => Err(AYError::UnsupportedOp(format!(
            "FuncSet range must be a finite literal set, got {:?}",
            std::mem::discriminant(&range.node)
        ))),
    }
}

fn int_literal(expr: &Spanned<Expr>) -> AYResult<i64> {
    match &expr.node {
        Expr::Int(n) => i64::try_from(n)
            .map_err(|_| AYError::IntegerOverflow("FuncSet bound too large for i64".to_string())),
        _ => Err(AYError::UnsupportedOp(
            "FuncSet bound must be an integer literal".to_string(),
        )),
    }
}

/// `base^exp` with overflow check (saturating to `None` past `usize`).
fn checked_pow(base: usize, exp: usize) -> Option<usize> {
    let mut acc: usize = 1;
    for _ in 0..exp {
        acc = acc.checked_mul(base)?;
        if acc > MAX_FUNCSET_ENUM {
            // Early out: any further multiplication only grows; report the
            // over-limit value so the caller's message is meaningful.
            return Some(acc);
        }
    }
    Some(acc)
}
