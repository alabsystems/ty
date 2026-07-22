// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! F1 (lever L2): compile-time folding of constant set-constructor subtrees.
//!
//! `compile_expr` calls [`FnCompileState::try_const_fold_set_expr`] at the
//! EAGER set-constructor arms (`SetEnum`, `SetBinOp`, `BigUnion`; see
//! [`is_const_fold_candidate`] for why the lazy `Powerset`/`Range`/`FuncSet`
//! constructors keep their opcodes at top level while still folding as inner
//! nodes). When the subtree's free names are all compile-time constants and
//! the estimated one-shot execution cost fits the work fuse, the subtree is
//! compiled into a self-contained scratch function and executed ONCE on the
//! REAL bytecode VM (injected by `tla-eval` via
//! `install_const_fold_executor`); the resulting `Value` is embedded as a
//! `LoadConst`. Any refusal — non-constant name, unsupported node kind,
//! budget overflow, scratch-compile error, or ANY VM error — silently falls
//! through to today's per-state compilation path (fail-open), preserving
//! runtime error sites exactly (e.g. `SUBSET 5` still fails at evaluation
//! time with the identical error).

use super::super::chunk::ConstantPool;
use super::super::const_fold::{
    const_fold_enabled, const_fold_executor, record_const_fold, CONST_FOLD_BUDGET,
};
use super::super::opcode::{Opcode, Register};
use super::{CompileError, FnCompileState};
use crate::nodes::{TirBoundPattern, TirBoundVar, TirExpr, TirNameKind, TirNameRef, TirSetOp};
use num_traits::ToPrimitive;
use tla_core::Spanned;
use tla_value::Value;

/// Whether this node kind is a fold-candidate set constructor.
///
/// Only constructors whose opcodes do EAGER per-state work fold at top
/// level: `SetEnum` (allocates + sorts a `SortedSet` per state), `SetBinOp`
/// (`to_sorted_set` on BOTH operands — including `powerset_eager` expansion
/// of lazy `Subset` operands — plus the union/intersection walk), and
/// `BigUnion` (materializes the outer set and every element).
///
/// The LAZY O(1) constructors (`Powerset`, `Range`, `FuncSet`) are
/// deliberately NOT folded at top level:
/// - their runtime opcodes only wrap already-computed operands in a lazy
///   value (an Arc bump), so folding buys ~nothing per state; and
/// - replacing their opcodes with `LoadConst` compound constants knocks
///   invariants out of the trust-cg native tier, which pattern-matches the
///   constructor opcodes (measured: trust-ir lowers `Range` symbolically but
///   refuses `LoadConst` intervals over its 64-element materialization
///   limit — folding `0..1000` broke CoffeeCan's native record-set path).
///
/// They still fold when nested INSIDE an eager constructor — e.g. the
/// measured MCTypeOK codomain `(SUBSET Proc) \cup (Proc \cup
/// {defaultInitValue})` folds as one unit, eliminating the per-state
/// powerset materialization + union.
///
/// Set COMPREHENSIONS (`SetFilter` `{x \in S : P(x)}`, `SetBuilder`
/// `{e(x) : x \in S, ...}`) fold too when their domain is a compile-time
/// constant and their predicate/body references only the comprehension's own
/// bound variables plus constants (no state var, no outer runtime binder, no
/// call). Their opcodes materialize an eagerly-collected `SortedSet` per state
/// (a full `SetFilterBegin`/`SetBuilderBegin` loop), so folding them to a
/// `LoadConst` is a strict per-state win and — unlike the lazy constructors —
/// carries no native-tier pattern-match to preserve (a constant `SetFilter`
/// never lowered natively in the first place; `record_set_scalarize` only
/// matches comprehensions over a *state* variable, whose domain is never a
/// compile-time constant, so those are left untouched). `GameOfLife`'s
/// `nbrs == {x \in {-1,0,1} \X {-1,0,1} : x /= <<0,0>>}` is the canonical case.
pub(super) fn is_const_fold_candidate(expr: &TirExpr) -> bool {
    matches!(
        expr,
        TirExpr::SetEnum(_)
            | TirExpr::SetBinOp { .. }
            | TirExpr::BigUnion(_)
            | TirExpr::SetFilter { .. }
            | TirExpr::SetBuilder { .. }
    )
}

/// The source names a bound variable introduces (one for a plain binder, one
/// per component for a `<<a, b>>` tuple-destructuring binder). Mirrors
/// `push_bound_var_bindings` so the estimator's "is this name bound by the
/// comprehension we are folding" test matches exactly which names the scratch
/// compile will bind.
fn bound_var_names(var: &TirBoundVar) -> Vec<String> {
    match &var.pattern {
        Some(TirBoundPattern::Tuple(components)) => {
            components.iter().map(|(name, _)| name.clone()).collect()
        }
        Some(TirBoundPattern::Var(name, _)) => vec![name.clone()],
        None => vec![var.name.clone()],
    }
}

impl<'a> FnCompileState<'a> {
    /// Try to fold a constant set-constructor subtree into a `LoadConst`.
    ///
    /// Returns `Ok(Some(register))` when the fold succeeded, `Ok(None)` to
    /// fall through to normal compilation. Errors only propagate from
    /// embedding the folded constant itself (pool/register overflow).
    pub(super) fn try_const_fold_set_expr(
        &mut self,
        expr: &Spanned<TirExpr>,
    ) -> Result<Option<Register>, CompileError> {
        // Inside a scratch fold compilation the whole subtree already
        // executes at once; nested re-folding would be redundant work.
        if self.const_folding || !const_fold_enabled() {
            return Ok(None);
        }
        let Some(executor) = const_fold_executor() else {
            return Ok(None);
        };
        // Constancy check + work-budget estimation in one recursive pass
        // (review H1: the fuse covers EVERY materialization the one-shot
        // execution could perform).
        let mut work: u64 = 0;
        if self.const_fold_estimate(expr, &[], &mut work).is_none() {
            return Ok(None);
        }

        // Compile the subtree into a self-contained scratch function with its
        // own constant pool, no bindings, and no state/callee context. Name
        // resolution inside the scratch compile follows compile_name_expr's
        // exact order, and the estimator already guaranteed every name
        // resolves through resolved_constants.
        let mut scratch_pool = ConstantPool::new();
        let func = {
            let mut scratch = FnCompileState::new("<const-fold>".to_string(), 0, &mut scratch_pool);
            scratch.resolved_constants = self.resolved_constants;
            scratch.const_folding = true;
            let Ok(result_reg) = scratch.compile_expr(expr) else {
                // Refusal: fall through so normal compilation surfaces its
                // own (identical) diagnostics.
                return Ok(None);
            };
            if result_reg != 0 {
                scratch.func.emit(Opcode::Move {
                    rd: 0,
                    rs: result_reg,
                });
            }
            scratch.func.emit(Opcode::Ret { rs: 0 });
            scratch.func
        };

        // Defense-in-depth: the estimator guarantees the subtree is
        // state-free and call-free, so the scratch function can run against
        // an empty chunk with no state arrays. Refuse if any opcode escaping
        // that contract ever slips through.
        if func.instructions.iter().any(|op| {
            matches!(
                op,
                Opcode::Call { .. }
                    | Opcode::CallExternal { .. }
                    | Opcode::CallBuiltin { .. }
                    | Opcode::ValueApply { .. }
                    | Opcode::MakeClosure { .. }
                    | Opcode::LoadVar { .. }
                    | Opcode::LoadPrime { .. }
                    | Opcode::StoreVar { .. }
                    | Opcode::SetPrimeMode { .. }
                    | Opcode::Unchanged { .. }
            )
        }) {
            return Ok(None);
        }

        // Execute the REAL VM once. ANY error refuses the fold so that the
        // identical runtime error still surfaces at the same evaluation
        // point (error-site fidelity, e.g. `SUBSET 5`).
        let Ok(value) = executor(&func, &scratch_pool) else {
            return Ok(None);
        };
        record_const_fold(&self.func.name, &value);
        self.compile_const(&value).map(Some)
    }

    /// Estimate the compile-time execution cost of a constant subtree.
    ///
    /// Returns `Some(card)` when the subtree is a compile-time constant whose
    /// one-shot VM execution fits [`CONST_FOLD_BUDGET`]; `card` is a
    /// conservative upper bound on the result's enumerable cardinality
    /// (`Some(None)` when the result is not a known-finite enumerable set,
    /// e.g. a scalar `SetEnum` element or a `FuncSet` over lazy shapes).
    /// Returns `None` (outer) to REFUSE the fold.
    ///
    /// The cost model mirrors the VM's materialization behavior in
    /// `tla-eval`'s `execute_compound.rs`:
    /// - `SetUnion`/`SetIntersect`/`SetDiff` call `to_sorted_set` on BOTH
    ///   operands — each operand charges its full cardinality (an Interval
    ///   operand charges its width, so `{0} \cup (1..10^8)` refuses; a lazy
    ///   `Subset` operand charges `2^|base|` because `to_sorted_set` expands
    ///   it via `powerset_eager`).
    /// - `Powerset`/`Range`/`FuncSet` construct lazy values in O(1); their
    ///   cardinality is only charged when a parent enumerates them.
    /// - `BigUnion` enumerates the outer set and every element.
    ///
    /// Any `len() == None` lazy shape refuses wherever a cardinality is
    /// needed. Refusal is fail-open: the runtime never pays for runtime-dead
    /// branches, and neither does compile time.
    ///
    /// `bound` holds the source names of the comprehension binders enclosing
    /// `expr` WITHIN this fold (empty at the fold root). A `Name` in `bound`
    /// is a constant-iteration scalar element — it contributes no enumerable
    /// cardinality and no work, and crucially it does NOT refuse the fold. Any
    /// OTHER free `Name` must resolve to a compile-time constant via
    /// [`Self::const_fold_leaf`] (which refuses outer *runtime* binders and
    /// state vars), so a comprehension that reads a runtime value — e.g.
    /// GameOfLife's `points == {<<p[1]+x, p[2]+y>> : <<x,y>> \in nbrs}`, whose
    /// body reads the outer runtime binder `p` — fails closed and is left as a
    /// per-state loop.
    fn const_fold_estimate(
        &self,
        expr: &Spanned<TirExpr>,
        bound: &[String],
        work: &mut u64,
    ) -> Option<Option<u64>> {
        match &expr.node {
            TirExpr::Const { value, .. } => Some(const_value_card(value)),
            TirExpr::Name(name_ref) => {
                if bound.iter().any(|b| b == &name_ref.name) {
                    // Bound by an enclosing folded comprehension: a
                    // constant-iteration scalar element of unknown enumerable
                    // cardinality. Does not refuse; charges no work.
                    return Some(None);
                }
                Some(const_value_card(self.const_fold_leaf(name_ref)?))
            }
            TirExpr::SetEnum(elements) => {
                for element in elements {
                    // Children contribute their own construction work only;
                    // their cardinality is irrelevant here (they are element
                    // VALUES, which may be scalars).
                    self.const_fold_estimate(element, bound, work)?;
                }
                charge(work, elements.len() as u64)?;
                Some(Some(elements.len() as u64))
            }
            TirExpr::SetBinOp { left, op, right } => {
                // The VM materializes BOTH operands via to_sorted_set; a
                // non-enumerable operand (scalar or unknown-length lazy
                // shape) refuses, preserving the runtime error site.
                let cl = self.const_fold_estimate(left, bound, work)??;
                let cr = self.const_fold_estimate(right, bound, work)??;
                charge(work, cl.saturating_add(cr))?;
                Some(Some(match op {
                    TirSetOp::Union => cl.saturating_add(cr),
                    TirSetOp::Intersect => cl.min(cr),
                    TirSetOp::Minus => cl,
                }))
            }
            TirExpr::Powerset(inner) => {
                // Opcode::Powerset builds a lazy SubsetValue in O(1).
                // Requiring a known base cardinality both rejects non-set
                // bases (`SUBSET 5` keeps its runtime error site) and lets
                // parents charge the 2^n expansion if they enumerate the
                // result (respects the powerset_eager n>=64 bound: 2^17 is
                // already over budget).
                let ci = self.const_fold_estimate(inner, bound, work)??;
                Some(Some(if ci >= 63 { u64::MAX } else { 1u64 << ci }))
            }
            TirExpr::Range { lo, hi } => {
                // Opcode::Range builds a lazy Interval in O(1); bounds must
                // be integer constants (non-int bounds keep their runtime
                // error site).
                let lo = self.const_fold_int(lo)?;
                let hi = self.const_fold_int(hi)?;
                let card = if hi < lo {
                    0
                } else {
                    u64::try_from(i128::from(hi) - i128::from(lo) + 1).unwrap_or(u64::MAX)
                };
                Some(Some(card))
            }
            TirExpr::BigUnion(inner) => {
                // The VM enumerates the outer set (to_sorted_set) and then
                // every element (to_sorted_set per element).
                let co = self.const_fold_estimate(inner, bound, work)??;
                charge(work, co)?;
                let elem_sum = self.const_fold_elem_card_sum(inner, bound)?;
                charge(work, elem_sum)?;
                Some(Some(elem_sum))
            }
            TirExpr::FuncSet { domain, range } => {
                // Opcode::FuncSet builds a lazy FuncSetValue in O(1) and
                // never errors, so unknown-cardinality sides do not refuse;
                // |range|^|domain| is charged only by enumerating parents.
                let cd = self.const_fold_estimate(domain, bound, work)?;
                let cr = self.const_fold_estimate(range, bound, work)?;
                Some(match (cd, cr) {
                    (Some(d), Some(r)) => Some(pow_card_saturating(r, d)),
                    _ => None,
                })
            }
            TirExpr::Times(components) => {
                // `S \X T \X ...`: the VM materializes the product when a
                // parent (a comprehension domain) enumerates it, so charge the
                // full product. Every component must be a constant finite set.
                let mut product: u64 = 1;
                for component in components {
                    let cc = self.const_fold_estimate(component, bound, work)??;
                    product = product.saturating_mul(cc);
                }
                charge(work, product)?;
                Some(Some(product))
            }
            TirExpr::SetFilter { var, body } => {
                // `{x \in S : P(x)}`: S must be a constant finite domain; P
                // may read only x (added to `bound`) and constants. Charge the
                // domain enumeration once and the predicate once per element.
                let domain = var.domain.as_ref()?;
                let d = self.const_fold_estimate(domain, bound, work)??;
                let mut inner = bound.to_vec();
                inner.extend(bound_var_names(var));
                let mut body_work: u64 = 0;
                self.const_fold_estimate(body, &inner, &mut body_work)?;
                charge(work, d)?;
                charge(work, d.saturating_mul(body_work.max(1)))?;
                // A filter keeps a subset of the domain.
                Some(Some(d))
            }
            TirExpr::SetBuilder { body, vars } => {
                // `{e(x, y, ...) : x \in S, y \in T, ...}`: the domains form a
                // cross product (each later domain may read earlier binders),
                // the body may read the binders and constants. The result has
                // at most `|S| * |T| * ...` elements (fewer if the map is not
                // injective; the runtime dedups — a smaller card is a safe
                // over-estimate here).
                let mut product: u64 = 1;
                let mut inner = bound.to_vec();
                for var in vars {
                    let domain = var.domain.as_ref()?;
                    let d = self.const_fold_estimate(domain, &inner, work)??;
                    product = product.saturating_mul(d);
                    inner.extend(bound_var_names(var));
                }
                let mut body_work: u64 = 0;
                self.const_fold_estimate(body, &inner, &mut body_work)?;
                charge(work, product)?;
                charge(work, product.saturating_mul(body_work.max(1)))?;
                Some(Some(product))
            }
            // Scalar / boolean / tuple sub-expressions: constant as long as
            // every child is constant (given `bound`). They produce a
            // scalar-ish value (unknown enumerable cardinality → `Some(None)`),
            // and only their children's own materialization work is charged.
            TirExpr::ArithBinOp { left, right, .. }
            | TirExpr::BoolBinOp { left, right, .. }
            | TirExpr::Cmp { left, right, .. }
            | TirExpr::Subseteq { left, right } => {
                self.const_fold_estimate(left, bound, work)?;
                self.const_fold_estimate(right, bound, work)?;
                Some(None)
            }
            TirExpr::In { elem, set } => {
                // `elem \in set` scans `set` (its estimate charges the card).
                self.const_fold_estimate(elem, bound, work)?;
                self.const_fold_estimate(set, bound, work)?;
                Some(None)
            }
            TirExpr::ArithNeg(inner) | TirExpr::BoolNot(inner) => {
                self.const_fold_estimate(inner, bound, work)?;
                Some(None)
            }
            TirExpr::Tuple(elements) => {
                for element in elements {
                    self.const_fold_estimate(element, bound, work)?;
                }
                Some(None)
            }
            TirExpr::If { cond, then_, else_ } => {
                self.const_fold_estimate(cond, bound, work)?;
                self.const_fold_estimate(then_, bound, work)?;
                self.const_fold_estimate(else_, bound, work)?;
                Some(None)
            }
            TirExpr::FuncApply { func, arg } => {
                // Applying a constant function to a constant/bound key. The
                // function operand must itself be constant (its estimate
                // refuses a state var / non-constant operator).
                self.const_fold_estimate(func, bound, work)?;
                self.const_fold_estimate(arg, bound, work)?;
                Some(None)
            }
            TirExpr::RecordAccess { record, .. } => {
                self.const_fold_estimate(record, bound, work)?;
                Some(None)
            }
            // Any other node kind (quantifiers, CHOOSE, EXCEPT, FuncDef,
            // KSubset, state access, calls, ...) is outside the folded grammar
            // → refuse the whole subtree. Nested constant constructors still
            // fold individually when normal compilation recurses into them.
            _ => None,
        }
    }

    /// Upper-bound the summed cardinalities of a `UNION` operand's elements.
    ///
    /// Conservative: only `SetEnum` constructors (children sized recursively)
    /// and leaf constants that are materialized `Value::Set`s (elements sized
    /// by inspection) are supported; anything else refuses. This refuses more
    /// than strictly necessary (e.g. `UNION` over a huge Interval singleton
    /// that the runtime's singleton branch returns in O(1)) — a beyond-spec
    /// refusal that only costs a missed fold, never correctness.
    fn const_fold_elem_card_sum(&self, inner: &Spanned<TirExpr>, bound: &[String]) -> Option<u64> {
        match &inner.node {
            TirExpr::SetEnum(children) => {
                let mut sum: u64 = 0;
                for child in children {
                    // Work was already charged by the caller's estimate of
                    // `inner`; this pass only queries cardinalities.
                    let mut scratch_work = 0u64;
                    sum = sum.saturating_add(self.const_fold_estimate(
                        child,
                        bound,
                        &mut scratch_work,
                    )??);
                }
                Some(sum)
            }
            TirExpr::Const { value, .. } => elem_card_sum_of_set_value(value),
            TirExpr::Name(name_ref) => elem_card_sum_of_set_value(self.const_fold_leaf(name_ref)?),
            _ => None,
        }
    }

    /// Resolve a `Name` leaf to its compile-time constant value.
    ///
    /// Mirrors `compile_name_expr`'s resolution order exactly: a name
    /// shadowed by ANY binder in scope (quantifier/LET binding,
    /// last-binding-wins via `lookup_binding`) is NOT a constant; only
    /// unshadowed `Ident`s that resolve in `resolved_constants` (same
    /// `name_id`/`lookup_name_id` logic) qualify. State vars, operator
    /// names, and replacement-chained names all refuse — the scratch
    /// compile would resolve them differently.
    fn const_fold_leaf(&self, name_ref: &TirNameRef) -> Option<&'a Value> {
        if self.lookup_binding(&name_ref.name).is_some() {
            return None;
        }
        if !matches!(name_ref.kind, TirNameKind::Ident) {
            return None;
        }
        self.resolved_constant_value(name_ref)
    }

    /// Resolve an integer `Range` bound: literal or resolved-constant int.
    fn const_fold_int(&self, expr: &Spanned<TirExpr>) -> Option<i64> {
        match &expr.node {
            TirExpr::Const { value, .. } => value.as_i64(),
            TirExpr::Name(name_ref) => self.const_fold_leaf(name_ref)?.as_i64(),
            _ => None,
        }
    }
}

/// Conservative enumerable cardinality of a leaf constant.
///
/// `None` covers scalars AND lazy shapes with unknown length (`set_len() ==
/// None`), which refuse wherever a cardinality is required.
fn const_value_card(value: &Value) -> Option<u64> {
    value.set_len()?.to_u64()
}

/// Sum of element cardinalities for a materialized `Value::Set`, refusing on
/// non-set or unknown-length elements (the runtime's `BigUnion` would raise a
/// type error on non-set elements — refusal keeps that error at runtime).
fn elem_card_sum_of_set_value(value: &Value) -> Option<u64> {
    let Value::Set(set) = value else { return None };
    let mut sum: u64 = 0;
    for elem in set.iter() {
        sum = sum.saturating_add(const_value_card(elem)?);
    }
    Some(sum)
}

/// Charge `amount` units of work against the fuse; `None` when over budget.
fn charge(work: &mut u64, amount: u64) -> Option<()> {
    *work = work.saturating_add(amount);
    (*work <= CONST_FOLD_BUDGET).then_some(())
}

/// `base^exp` with saturation (cardinality of `[S -> T]` is `|T|^|S|`).
fn pow_card_saturating(base: u64, exp: u64) -> u64 {
    match (base, exp) {
        (_, 0) => 1,
        (0, _) => 0,
        (1, _) => 1,
        _ => {
            if exp > 64 {
                return u64::MAX;
            }
            let mut acc: u64 = 1;
            for _ in 0..exp {
                acc = acc.saturating_mul(base);
                if acc == u64::MAX {
                    break;
                }
            }
            acc
        }
    }
}
