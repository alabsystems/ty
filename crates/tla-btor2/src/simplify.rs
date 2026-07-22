// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

// Expression simplification for BTOR2-to-CHC translation.
//
// Applied at the memoization point in the translator: each BTOR2 node
// is simplified exactly once before caching. This reduces the size of
// the CHC problem before it reaches the ay-chc solver.
//
// Simplifications performed:
// - Constant folding for BV arithmetic and bitwise ops
// - Identity elimination (x + 0 → x, x & all_ones → x)
// - Annihilation (x & 0 → 0)
// - Double negation (~~x → x)
// - ITE simplification (ite(true, a, b) → a; ite(c, x, x) → x)
// - Comparison simplification (x == x → true)
// - Boolean simplification

use std::sync::Arc;

use ay_chc::{ChcExpr, ChcOp};

/// Simplify a CHC expression bottom-up.
///
/// Recursively simplifies children first, then applies local rewrites.
pub(crate) fn simplify_expr(expr: &ChcExpr) -> ChcExpr {
    match expr {
        ChcExpr::Op(op, args) => {
            // Recursively simplify children.
            let simplified_args: Vec<Arc<ChcExpr>> =
                args.iter().map(|a| Arc::new(simplify_expr(a))).collect();
            // Try to simplify this node.
            simplify_op(op, &simplified_args)
        }
        // Leaves (Var, Bool, BitVec, Int) are already simple.
        other => other.clone(),
    }
}

/// Extract a BV constant value and width from a ChcExpr.
fn as_bv_const(expr: &ChcExpr) -> Option<(u128, u32)> {
    match expr {
        ChcExpr::BitVec(v, w) => Some((*v, *w)),
        _ => None,
    }
}

/// Create a BV constant expression.
fn make_bv(value: u128, width: u32) -> ChcExpr {
    let mask = bv_mask(width);
    ChcExpr::BitVec(value & mask, width)
}

/// Compute the mask for a bitvector of the given width.
fn bv_mask(width: u32) -> u128 {
    if width >= 128 {
        u128::MAX
    } else {
        (1u128 << width) - 1
    }
}

/// Check if an expression is the zero BV constant.
fn is_bv_zero(expr: &ChcExpr) -> bool {
    matches!(expr, ChcExpr::BitVec(0, _))
}

/// Check if an expression is the one BV constant.
fn is_bv_one(expr: &ChcExpr) -> bool {
    matches!(expr, ChcExpr::BitVec(1, _))
}

/// Check if an expression is all-ones for its width.
fn is_bv_all_ones(expr: &ChcExpr) -> bool {
    if let ChcExpr::BitVec(v, w) = expr {
        *v == bv_mask(*w)
    } else {
        false
    }
}

/// Wrap args into a new Op expression (default, no simplification).
fn make_op(op: &ChcOp, args: &[Arc<ChcExpr>]) -> ChcExpr {
    ChcExpr::Op(*op, args.to_vec())
}

/// Decide whether two array-index expressions are provably equal or distinct:
/// `Some(true)` when they must be equal (syntactically identical — including the
/// same variable — or equal constants), `Some(false)` when they are distinct
/// constants of the same width, and `None` when it cannot be decided. Used only
/// for SOUND read-over-write / write-over-write rewrites: since `ChcExpr` is a
/// pure term (no side effects, a variable denotes one value per state),
/// syntactic equality of indices implies value equality.
fn indices_equal(a: &ChcExpr, b: &ChcExpr) -> Option<bool> {
    if a == b {
        return Some(true);
    }
    if let (Some((va, wa)), Some((vb, wb))) = (as_bv_const(a), as_bv_const(b)) {
        if wa == wb {
            return Some(va == vb);
        }
    }
    None
}

/// Try to simplify an operation given its already-simplified arguments.
fn simplify_op(op: &ChcOp, args: &[Arc<ChcExpr>]) -> ChcExpr {
    match op {
        // -------------------------------------------------------------------
        // BV arithmetic: constant folding + identity
        // -------------------------------------------------------------------
        ChcOp::BvAdd => {
            if args.len() == 2 {
                if let (Some((a, wa)), Some((b, wb))) =
                    (as_bv_const(&args[0]), as_bv_const(&args[1]))
                {
                    if wa == wb {
                        return make_bv(a.wrapping_add(b), wa);
                    }
                }
                if is_bv_zero(&args[0]) {
                    return (*args[1]).clone();
                }
                if is_bv_zero(&args[1]) {
                    return (*args[0]).clone();
                }
            }
            make_op(op, args)
        }
        ChcOp::BvSub => {
            if args.len() == 2 {
                if let (Some((a, wa)), Some((b, wb))) =
                    (as_bv_const(&args[0]), as_bv_const(&args[1]))
                {
                    if wa == wb {
                        return make_bv(a.wrapping_sub(b), wa);
                    }
                }
                if is_bv_zero(&args[1]) {
                    return (*args[0]).clone();
                }
            }
            make_op(op, args)
        }
        ChcOp::BvMul => {
            if args.len() == 2 {
                if let (Some((a, wa)), Some((b, wb))) =
                    (as_bv_const(&args[0]), as_bv_const(&args[1]))
                {
                    if wa == wb {
                        return make_bv(a.wrapping_mul(b), wa);
                    }
                }
                if is_bv_zero(&args[0]) || is_bv_zero(&args[1]) {
                    if let Some((_, w)) = as_bv_const(&args[0]).or(as_bv_const(&args[1])) {
                        return make_bv(0, w);
                    }
                }
                if is_bv_one(&args[0]) {
                    return (*args[1]).clone();
                }
                if is_bv_one(&args[1]) {
                    return (*args[0]).clone();
                }
            }
            make_op(op, args)
        }

        // -------------------------------------------------------------------
        // Bitwise operations
        // -------------------------------------------------------------------
        ChcOp::BvAnd => {
            if args.len() == 2 {
                if let (Some((a, wa)), Some((b, wb))) =
                    (as_bv_const(&args[0]), as_bv_const(&args[1]))
                {
                    if wa == wb {
                        return make_bv(a & b, wa);
                    }
                }
                // x & 0 = 0
                if is_bv_zero(&args[0]) {
                    return (*args[0]).clone();
                }
                if is_bv_zero(&args[1]) {
                    return (*args[1]).clone();
                }
                // x & all_ones = x
                if is_bv_all_ones(&args[0]) {
                    return (*args[1]).clone();
                }
                if is_bv_all_ones(&args[1]) {
                    return (*args[0]).clone();
                }
            }
            make_op(op, args)
        }
        ChcOp::BvOr => {
            if args.len() == 2 {
                if let (Some((a, wa)), Some((b, wb))) =
                    (as_bv_const(&args[0]), as_bv_const(&args[1]))
                {
                    if wa == wb {
                        return make_bv(a | b, wa);
                    }
                }
                if is_bv_zero(&args[0]) {
                    return (*args[1]).clone();
                }
                if is_bv_zero(&args[1]) {
                    return (*args[0]).clone();
                }
                if is_bv_all_ones(&args[0]) {
                    return (*args[0]).clone();
                }
                if is_bv_all_ones(&args[1]) {
                    return (*args[1]).clone();
                }
            }
            make_op(op, args)
        }
        ChcOp::BvXor => {
            if args.len() == 2 {
                if let (Some((a, wa)), Some((b, wb))) =
                    (as_bv_const(&args[0]), as_bv_const(&args[1]))
                {
                    if wa == wb {
                        return make_bv(a ^ b, wa);
                    }
                }
                if is_bv_zero(&args[0]) {
                    return (*args[1]).clone();
                }
                if is_bv_zero(&args[1]) {
                    return (*args[0]).clone();
                }
                // x ^ x = 0
                if args[0] == args[1] {
                    if let Some((_, w)) = as_bv_const(&args[0]) {
                        return make_bv(0, w);
                    }
                }
            }
            make_op(op, args)
        }
        ChcOp::BvNot => {
            if args.len() == 1 {
                if let Some((v, w)) = as_bv_const(&args[0]) {
                    return make_bv(!v, w);
                }
                // Double negation: ~~x = x
                if let ChcExpr::Op(ChcOp::BvNot, inner) = args[0].as_ref() {
                    if inner.len() == 1 {
                        return (*inner[0]).clone();
                    }
                }
            }
            make_op(op, args)
        }
        ChcOp::BvNeg => {
            if args.len() == 1 {
                if let Some((v, w)) = as_bv_const(&args[0]) {
                    return make_bv((!v).wrapping_add(1), w);
                }
            }
            make_op(op, args)
        }

        // -------------------------------------------------------------------
        // Comparisons
        // -------------------------------------------------------------------
        ChcOp::Eq => {
            if args.len() == 2 {
                // const == const
                if let (Some((a, wa)), Some((b, wb))) =
                    (as_bv_const(&args[0]), as_bv_const(&args[1]))
                {
                    if wa == wb {
                        return ChcExpr::Bool(a == b);
                    }
                }
                // x == x → true
                if args[0] == args[1] {
                    return ChcExpr::Bool(true);
                }
            }
            make_op(op, args)
        }
        ChcOp::BvULt => {
            if args.len() == 2 {
                if let (Some((a, wa)), Some((b, wb))) =
                    (as_bv_const(&args[0]), as_bv_const(&args[1]))
                {
                    if wa == wb {
                        return ChcExpr::Bool(a < b);
                    }
                }
                // x < x → false
                if args[0] == args[1] {
                    return ChcExpr::Bool(false);
                }
            }
            make_op(op, args)
        }
        ChcOp::BvULe => {
            if args.len() == 2 {
                if let (Some((a, wa)), Some((b, wb))) =
                    (as_bv_const(&args[0]), as_bv_const(&args[1]))
                {
                    if wa == wb {
                        return ChcExpr::Bool(a <= b);
                    }
                }
                // x <= x → true
                if args[0] == args[1] {
                    return ChcExpr::Bool(true);
                }
            }
            make_op(op, args)
        }

        // -------------------------------------------------------------------
        // Shifts
        // -------------------------------------------------------------------
        ChcOp::BvShl | ChcOp::BvLShr => {
            if args.len() == 2 {
                if is_bv_zero(&args[0]) {
                    return (*args[0]).clone();
                }
                if is_bv_zero(&args[1]) {
                    return (*args[0]).clone();
                }
                if let (Some((a, wa)), Some((b, wb))) =
                    (as_bv_const(&args[0]), as_bv_const(&args[1]))
                {
                    if wa == wb {
                        let shift = b as u32;
                        if shift >= wa {
                            return make_bv(0, wa);
                        }
                        let result = if matches!(op, ChcOp::BvShl) {
                            a << shift
                        } else {
                            a >> shift
                        };
                        return make_bv(result, wa);
                    }
                }
            }
            make_op(op, args)
        }

        // -------------------------------------------------------------------
        // ITE (if-then-else)
        // -------------------------------------------------------------------
        ChcOp::Ite => {
            if args.len() == 3 {
                // ite(true, a, b) → a
                if *args[0] == ChcExpr::Bool(true) {
                    return (*args[1]).clone();
                }
                // ite(false, a, b) → b
                if *args[0] == ChcExpr::Bool(false) {
                    return (*args[2]).clone();
                }
                // ite(c, x, x) → x
                if args[1] == args[2] {
                    return (*args[1]).clone();
                }
            }
            make_op(op, args)
        }

        // -------------------------------------------------------------------
        // Extract
        // -------------------------------------------------------------------
        ChcOp::BvExtract(hi, lo) => {
            if args.len() == 1 {
                if let Some((v, _w)) = as_bv_const(&args[0]) {
                    let width = hi - lo + 1;
                    let mask = bv_mask(width);
                    return make_bv((v >> lo) & mask, width);
                }
            }
            make_op(op, args)
        }

        // -------------------------------------------------------------------
        // Extend
        // -------------------------------------------------------------------
        ChcOp::BvZeroExtend(extra) => {
            if args.len() == 1 && *extra == 0 {
                return (*args[0]).clone();
            }
            make_op(op, args)
        }
        ChcOp::BvSignExtend(extra) => {
            if args.len() == 1 && *extra == 0 {
                return (*args[0]).clone();
            }
            make_op(op, args)
        }

        // -------------------------------------------------------------------
        // Concat
        // -------------------------------------------------------------------
        ChcOp::BvConcat => {
            if args.len() == 2 {
                if let (Some((a, wa)), Some((b, wb))) =
                    (as_bv_const(&args[0]), as_bv_const(&args[1]))
                {
                    let result = ((a & bv_mask(wa)) << wb) | (b & bv_mask(wb));
                    return make_bv(result, wa + wb);
                }
            }
            make_op(op, args)
        }

        // -------------------------------------------------------------------
        // Boolean operations
        // -------------------------------------------------------------------
        ChcOp::And => {
            if args.len() == 2 {
                if *args[0] == ChcExpr::Bool(false) || *args[1] == ChcExpr::Bool(false) {
                    return ChcExpr::Bool(false);
                }
                if *args[0] == ChcExpr::Bool(true) {
                    return (*args[1]).clone();
                }
                if *args[1] == ChcExpr::Bool(true) {
                    return (*args[0]).clone();
                }
            }
            make_op(op, args)
        }
        ChcOp::Or => {
            if args.len() == 2 {
                if *args[0] == ChcExpr::Bool(true) || *args[1] == ChcExpr::Bool(true) {
                    return ChcExpr::Bool(true);
                }
                if *args[0] == ChcExpr::Bool(false) {
                    return (*args[1]).clone();
                }
                if *args[1] == ChcExpr::Bool(false) {
                    return (*args[0]).clone();
                }
            }
            make_op(op, args)
        }
        ChcOp::Not => {
            if args.len() == 1 {
                if *args[0] == ChcExpr::Bool(true) {
                    return ChcExpr::Bool(false);
                }
                if *args[0] == ChcExpr::Bool(false) {
                    return ChcExpr::Bool(true);
                }
                // Double negation
                if let ChcExpr::Op(ChcOp::Not, inner) = args[0].as_ref() {
                    if inner.len() == 1 {
                        return (*inner[0]).clone();
                    }
                }
            }
            make_op(op, args)
        }
        ChcOp::Implies => {
            if args.len() == 2 {
                if *args[0] == ChcExpr::Bool(false) || *args[1] == ChcExpr::Bool(true) {
                    return ChcExpr::Bool(true);
                }
                if *args[0] == ChcExpr::Bool(true) {
                    return (*args[1]).clone();
                }
            }
            make_op(op, args)
        }

        // -------------------------------------------------------------------
        // Arrays: read-over-write, write-over-write, const-array select.
        //
        // These shrink the array problem BEFORE it reaches the ay-chc solver,
        // which is exactly the large-array case that is declined by the
        // bit-blaster and routed to the CHC portfolio. Every rewrite is a
        // classic sound array-theory identity.
        // -------------------------------------------------------------------
        ChcOp::Select => {
            if args.len() == 2 {
                // select(const_array(_, v), _) = v — every element is `v`.
                if let ChcExpr::ConstArray(_, val) = args[0].as_ref() {
                    return val.as_ref().clone();
                }
                // Read-over-write: select(store(a, i, v), j).
                if let ChcExpr::Op(ChcOp::Store, st) = args[0].as_ref() {
                    if st.len() == 3 {
                        match indices_equal(&st[1], &args[1]) {
                            // j == i: the read returns the just-written value.
                            Some(true) => return st[2].as_ref().clone(),
                            // j != i: the store is irrelevant; read the base
                            // array (recurse to peel further constant stores).
                            Some(false) => {
                                return simplify_op(
                                    &ChcOp::Select,
                                    &[st[0].clone(), args[1].clone()],
                                );
                            }
                            None => {}
                        }
                    }
                }
            }
            make_op(op, args)
        }
        ChcOp::Store => {
            if args.len() == 3 {
                // Write-back is a no-op: store(a, i, select(a, i)) = a.
                if let ChcExpr::Op(ChcOp::Select, sel) = args[2].as_ref() {
                    if sel.len() == 2
                        && sel[0] == args[0]
                        && indices_equal(&sel[1], &args[1]) == Some(true)
                    {
                        return args[0].as_ref().clone();
                    }
                }
                // Write-over-write to a provably-equal index: the inner store is
                // dead. store(store(a, i, v1), i, v2) = store(a, i, v2).
                if let ChcExpr::Op(ChcOp::Store, inner) = args[0].as_ref() {
                    if inner.len() == 3 && indices_equal(&inner[1], &args[1]) == Some(true) {
                        return simplify_op(
                            &ChcOp::Store,
                            &[inner[0].clone(), args[1].clone(), args[2].clone()],
                        );
                    }
                }
            }
            make_op(op, args)
        }

        // Default: no simplification.
        _ => make_op(op, args),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bv_var(name: &str) -> ChcExpr {
        ChcExpr::Var(ay_chc::ChcVar {
            name: name.to_string(),
            sort: ay_chc::ChcSort::BitVec(8),
        })
    }

    #[test]
    fn test_constant_fold_add() {
        let expr = ChcExpr::Op(
            ChcOp::BvAdd,
            vec![
                Arc::new(ChcExpr::BitVec(3, 8)),
                Arc::new(ChcExpr::BitVec(5, 8)),
            ],
        );
        assert_eq!(simplify_expr(&expr), ChcExpr::BitVec(8, 8));
    }

    #[test]
    fn test_identity_add_zero() {
        let x = bv_var("x");
        let expr = ChcExpr::Op(
            ChcOp::BvAdd,
            vec![Arc::new(x.clone()), Arc::new(ChcExpr::BitVec(0, 8))],
        );
        assert_eq!(simplify_expr(&expr), x);
    }

    #[test]
    fn test_and_zero_annihilation() {
        let x = bv_var("x");
        let expr = ChcExpr::Op(
            ChcOp::BvAnd,
            vec![Arc::new(x), Arc::new(ChcExpr::BitVec(0, 8))],
        );
        assert_eq!(simplify_expr(&expr), ChcExpr::BitVec(0, 8));
    }

    #[test]
    fn test_or_zero_identity() {
        let x = bv_var("x");
        let expr = ChcExpr::Op(
            ChcOp::BvOr,
            vec![Arc::new(ChcExpr::BitVec(0, 8)), Arc::new(x.clone())],
        );
        assert_eq!(simplify_expr(&expr), x);
    }

    #[test]
    fn test_double_negation_bvnot() {
        let x = bv_var("x");
        let expr = ChcExpr::Op(
            ChcOp::BvNot,
            vec![Arc::new(ChcExpr::Op(
                ChcOp::BvNot,
                vec![Arc::new(x.clone())],
            ))],
        );
        assert_eq!(simplify_expr(&expr), x);
    }

    #[test]
    fn test_eq_same_operands() {
        let x = bv_var("x");
        let expr = ChcExpr::Op(ChcOp::Eq, vec![Arc::new(x.clone()), Arc::new(x)]);
        assert_eq!(simplify_expr(&expr), ChcExpr::Bool(true));
    }

    #[test]
    fn test_ite_true_branch() {
        let a = bv_var("a");
        let b = bv_var("b");
        let expr = ChcExpr::Op(
            ChcOp::Ite,
            vec![
                Arc::new(ChcExpr::Bool(true)),
                Arc::new(a.clone()),
                Arc::new(b),
            ],
        );
        assert_eq!(simplify_expr(&expr), a);
    }

    #[test]
    fn test_ite_false_branch() {
        let a = bv_var("a");
        let b = bv_var("b");
        let expr = ChcExpr::Op(
            ChcOp::Ite,
            vec![
                Arc::new(ChcExpr::Bool(false)),
                Arc::new(a),
                Arc::new(b.clone()),
            ],
        );
        assert_eq!(simplify_expr(&expr), b);
    }

    #[test]
    fn test_ite_same_branches() {
        let c = bv_var("c");
        let x = bv_var("x");
        let expr = ChcExpr::Op(
            ChcOp::Ite,
            vec![Arc::new(c), Arc::new(x.clone()), Arc::new(x.clone())],
        );
        assert_eq!(simplify_expr(&expr), x);
    }

    #[test]
    fn test_extract_constant() {
        // extract[3:0](0xFF) = 0x0F
        let expr = ChcExpr::Op(
            ChcOp::BvExtract(3, 0),
            vec![Arc::new(ChcExpr::BitVec(0xFF, 8))],
        );
        assert_eq!(simplify_expr(&expr), ChcExpr::BitVec(0x0F, 4));
    }

    #[test]
    fn test_boolean_and_simplification() {
        let x = bv_var("x");
        let expr = ChcExpr::Op(
            ChcOp::And,
            vec![
                Arc::new(ChcExpr::Bool(true)),
                Arc::new(ChcExpr::Bool(false)),
            ],
        );
        assert_eq!(simplify_expr(&expr), ChcExpr::Bool(false));

        let expr2 = ChcExpr::Op(
            ChcOp::And,
            vec![
                Arc::new(ChcExpr::Bool(true)),
                Arc::new(ChcExpr::Op(
                    ChcOp::Eq,
                    vec![Arc::new(x.clone()), Arc::new(x)],
                )),
            ],
        );
        // true AND (x == x) → (x == x) → true
        assert_eq!(simplify_expr(&expr2), ChcExpr::Bool(true));
    }

    // ---- Array simplification -------------------------------------------

    fn arr_var(name: &str) -> ChcExpr {
        ChcExpr::Var(ay_chc::ChcVar {
            name: name.to_string(),
            sort: ay_chc::ChcSort::Array(
                Box::new(ay_chc::ChcSort::BitVec(4)),
                Box::new(ay_chc::ChcSort::BitVec(8)),
            ),
        })
    }
    fn idx(v: u128) -> Arc<ChcExpr> {
        Arc::new(ChcExpr::BitVec(v, 4))
    }
    fn val(v: u128) -> Arc<ChcExpr> {
        Arc::new(ChcExpr::BitVec(v, 8))
    }
    fn store(a: ChcExpr, i: Arc<ChcExpr>, v: Arc<ChcExpr>) -> ChcExpr {
        ChcExpr::Op(ChcOp::Store, vec![Arc::new(a), i, v])
    }
    fn select(a: ChcExpr, i: Arc<ChcExpr>) -> ChcExpr {
        ChcExpr::Op(ChcOp::Select, vec![Arc::new(a), i])
    }

    #[test]
    fn test_select_over_const_array() {
        // select(const_array(_, 7), i) = 7
        let ca = ChcExpr::ConstArray(
            ay_chc::ChcSort::Array(
                Box::new(ay_chc::ChcSort::BitVec(4)),
                Box::new(ay_chc::ChcSort::BitVec(8)),
            ),
            val(7),
        );
        assert_eq!(simplify_expr(&select(ca, idx(3))), *val(7));
    }

    #[test]
    fn test_read_over_write_same_const_index() {
        // select(store(a, 3, 42), 3) = 42
        let e = select(store(arr_var("a"), idx(3), val(42)), idx(3));
        assert_eq!(simplify_expr(&e), *val(42));
    }

    #[test]
    fn test_read_over_write_distinct_const_index() {
        // select(store(a, 3, 42), 5) = select(a, 5) — store is irrelevant.
        let e = select(store(arr_var("a"), idx(3), val(42)), idx(5));
        assert_eq!(simplify_expr(&e), select(arr_var("a"), idx(5)));
    }

    #[test]
    fn test_read_over_write_same_symbolic_index() {
        // select(store(a, x, v), x) = v even when x is a variable.
        let x = Arc::new(ChcExpr::Var(ay_chc::ChcVar {
            name: "x".into(),
            sort: ay_chc::ChcSort::BitVec(4),
        }));
        let e = select(store(arr_var("a"), x.clone(), val(9)), x);
        assert_eq!(simplify_expr(&e), *val(9));
    }

    #[test]
    fn test_read_over_write_peels_store_chain() {
        // select(store(store(a, 1, 11), 2, 22), 1) = 11 — peel the distinct
        // store at index 2 down to the matching store at index 1.
        let inner = store(arr_var("a"), idx(1), val(11));
        let outer = store(inner, idx(2), val(22));
        assert_eq!(simplify_expr(&select(outer, idx(1))), *val(11));
    }

    #[test]
    fn test_write_over_write_same_index_eliminated() {
        // store(store(a, 3, 11), 3, 22) = store(a, 3, 22) — inner write is dead.
        let e = store(store(arr_var("a"), idx(3), val(11)), idx(3), val(22));
        assert_eq!(simplify_expr(&e), store(arr_var("a"), idx(3), val(22)));
    }

    #[test]
    fn test_store_writeback_is_noop() {
        // store(a, 3, select(a, 3)) = a — writing back the current value.
        let a = arr_var("a");
        let e = store(a.clone(), idx(3), Arc::new(select(a.clone(), idx(3))));
        assert_eq!(simplify_expr(&e), a);
        // But writing back a DIFFERENT index's value is NOT a no-op.
        let e2 = store(a.clone(), idx(3), Arc::new(select(a.clone(), idx(5))));
        assert_ne!(simplify_expr(&e2), a);
    }

    #[test]
    fn test_read_over_write_unknown_indices_not_simplified() {
        // select(store(a, x, v), y) with distinct variables x,y: MUST NOT fold
        // (they might be equal at runtime).
        let x = Arc::new(ChcExpr::Var(ay_chc::ChcVar {
            name: "x".into(),
            sort: ay_chc::ChcSort::BitVec(4),
        }));
        let y = Arc::new(ChcExpr::Var(ay_chc::ChcVar {
            name: "y".into(),
            sort: ay_chc::ChcSort::BitVec(4),
        }));
        let e = select(store(arr_var("a"), x, val(9)), y);
        // Unchanged: still a select over the store.
        assert_eq!(simplify_expr(&e), e);
    }

    #[test]
    fn test_nested_simplification() {
        // (x + 0) + (3 + 5) → x + 8
        let x = bv_var("x");
        let expr = ChcExpr::Op(
            ChcOp::BvAdd,
            vec![
                Arc::new(ChcExpr::Op(
                    ChcOp::BvAdd,
                    vec![Arc::new(x.clone()), Arc::new(ChcExpr::BitVec(0, 8))],
                )),
                Arc::new(ChcExpr::Op(
                    ChcOp::BvAdd,
                    vec![
                        Arc::new(ChcExpr::BitVec(3, 8)),
                        Arc::new(ChcExpr::BitVec(5, 8)),
                    ],
                )),
            ],
        );
        let result = simplify_expr(&expr);
        match &result {
            ChcExpr::Op(ChcOp::BvAdd, args) if args.len() == 2 => {
                assert_eq!(*args[0], x);
                assert_eq!(*args[1], ChcExpr::BitVec(8, 8));
            }
            _ => panic!("expected BvAdd(x, 8), got: {:?}", result),
        }
    }
}
