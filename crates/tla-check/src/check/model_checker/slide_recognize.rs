// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Static AUTO-ARM RECOGNIZER for the nested-set slide kernel
//! ([`crate::state::nested_set_slide`]) — the proof obligation that lets the
//! kernel arm BY DEFAULT.
//!
//! The kernel replaces the interpreter's successor generation for the
//! `SlidingPuzzles` idiom
//!
//! ```tla
//! Next == LET empty == Pos \ UNION board
//!         IN  \E e \in empty : board' \in update(e, empty)
//! ```
//!
//! where `update`/`move`/`dir` implement the rigid-unit slide (for the empty
//! cell `e`, each piece containing a cell adjacent to `e` may slide one unit
//! toward `e`, kept iff it stays inside `Pos` and disjoint from the other
//! pieces). The kernel is sound for EXACTLY that relation, so a default-on arm
//! must PROVE the spec's `Next` is that relation. This module is that proof:
//! a structural matcher over the lowered `tla_core` AST — the very
//! representation the interpreter evaluates — that mirrors the fail-closed
//! discipline of `trust_cg_dispatch::sum_fold_scalarize` (the fold
//! recognizer). Operator NAMES never matter (`update`/`move`/`dir` are
//! resolved from the call sites and matched by STRUCTURE); constants are
//! resolved by EVALUATION with the checker's own const-fold machinery.
//!
//! # What the recognizer proves (per the kernel's move-correspondence proof)
//!
//! Writing `B` for the (single) state variable and `?x` for arbitrary
//! spec-chosen names, the matcher accepts precisely this shape:
//!
//! ```text
//! Next   == LET ?empty == PosExpr \ UNION B
//!           IN \E ?e \in ?empty : B' \in ?update(?e, ?empty)
//! ?update(?e, ?es) ==
//!           LET ?dirs  == ?dir(?e, ?es)
//!               ?moved == {?move(?e, ?d) : ?d \in ?dirs}
//!               ?free  == {<<?pc, ?m>> \in ?moved :
//!                            /\ ?m \cap (UNION (B \ {?pc})) = {}
//!                            /\ \A ?p \in ?m : ?p \in PosExpr'}
//!           IN {(B \ {?pc}) \cup {?m} : <<?pc, ?m>> \in ?free}
//! ?dir(?p, ?es) ==
//!           LET ?dset == DirExpr
//!           IN {?d \in ?dset : /\ <<?p[1]+?d[1], ?p[2]+?d[2]>> \in PosExpr''
//!                              /\ <<?p[1]+?d[1], ?p[2]+?d[2]>> \notin ?es}
//! ?move(?p, ?d) ==
//!           LET ?s  == <<?p[1]+?d[1], ?p[2]+?d[2]>>
//!               ?pc == ?ch(B, LAMBDA ?c : ?s \in ?c)
//!           IN <<?pc, {<<?q[1]-?d[1], ?q[2]-?d[2]>> : ?q \in ?pc}>>
//! ?ch(?S, ?P(_)) ==
//!           CHOOSE ?x \in ?S : ?P(?x) /\ \A ?y \in ?S : ?P(?y) => ?y = ?x
//! ```
//!
//! with the semantic side conditions:
//!
//! * `PosExpr`, `PosExpr'`, `PosExpr''` are STATE-FREE (proved by a bounded
//!   transitive walk that resolves operator references and rejects any state
//!   variable / prime / action construct) and all EVALUATE to the same finite
//!   set of 2-int tuples, `1..=64` cells — the exact `Pos` grid handed to
//!   [`crate::state::SlideGeometry`]. This replaces the INIT bounding-box
//!   heuristic: a non-rectangular or larger-than-init `Pos` is handled
//!   correctly because the kernel gets the TRUE grid.
//! * `DirExpr` is state-free and evaluates to EXACTLY the four von Neumann
//!   unit vectors `{<<1,0>>, <<0,1>>, <<-1,0>>, <<0,-1>>}` — the kernel's
//!   `DIRS`. A diagonal (or missing) direction fails the match.
//! * The spec has exactly ONE state variable (the kernel regenerates the whole
//!   state from the board slot).
//! * Every name-reference is scope-checked: within each matched operator the
//!   bound names the pattern relies on are required pairwise-distinct and
//!   distinct from the state variable, so plain name equality coincides with
//!   lexical resolution (any shadowing fails closed).
//!
//! Under these facts the accepted `Next` IS the kernel's slide relation — the
//! bidirectional move-correspondence proof in the kernel's module docs applies
//! verbatim (`?dir` keeps direction `?d` iff `e+d` is on-grid and occupied;
//! `?move` translates the unique piece containing `e+d` by `-d`; `?free` is
//! the collision + boundary filter; the result builder is the single-piece
//! replacement `(B \ {pc}) ∪ {translate(pc)}`). The remaining value-level
//! precondition (INIT pieces pairwise disjoint, cells inside the grid) is
//! checked at arm time by [`crate::state::SlideKernelArm::try_arm_recognized`].
//!
//! # Fail-closed
//!
//! ANY deviation — an extra conjunct, a reordered filter, a different update,
//! a non-unit translation, a fifth direction, a state-dependent `Pos`, a
//! shadowed name, an operator replaced via config (`CONSTANT Op <- ...`), a
//! grid over 64 cells — returns `None` and the run stays byte-identical on the
//! interpreter path. The recognizer is a PROOF, not a heuristic: `None` is
//! always safe, `Some` must be exact.

use num_traits::ToPrimitive;
use tla_core::ast::{BoundPattern, BoundVar, Expr, OperatorDef};
use tla_core::Spanned;
use tla_eval::EvalCtx;

use crate::state::slide_value_to_pos as value_to_pos;

/// Successful recognition: the proven board variable and the evaluated `Pos`.
pub(in crate::check) struct RecognizedSlide {
    /// State-variable index of the board (always the single variable).
    pub(in crate::check) board_var_idx: usize,
    /// The EXACT evaluated `Pos` grid cells (sorted, deduped).
    pub(in crate::check) positions: Vec<(i64, i64)>,
}

/// Single-word inner-universe cap (mirrors `SlideGeometry::new`).
const MAX_GRID_CELLS: usize = 64;
/// Node budget for the transitive state-freedom walk (fail closed past it).
const STATE_FREE_NODE_BUDGET: usize = 20_000;
/// Operator-resolution depth cap for the state-freedom walk.
const STATE_FREE_MAX_DEPTH: usize = 16;

/// The four von Neumann unit vectors, sorted — what `DirExpr` must equal.
const VON_NEUMANN_DIRS: [(i64, i64); 4] = [(-1, 0), (0, -1), (0, 1), (1, 0)];

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

/// Prove that `next_name`'s definition is the rigid-unit-slide relation over a
/// single nested-set state variable, returning the board index and the
/// evaluated `Pos` grid. `None` = not proven; the caller MUST NOT arm.
pub(in crate::check) fn recognize_slide_next(
    ctx: &EvalCtx,
    next_name: &str,
) -> Option<RecognizedSlide> {
    // Exactly one state variable: the kernel rebuilds successor states by
    // replacing the board slot; any second variable would be unconstrained.
    let registry = ctx.var_registry();
    if registry.len() != 1 {
        return None;
    }
    let board = registry.name(tla_core::VarIndex::new(0)).to_string();
    let board = board.as_str();

    let next = resolve_user_op(ctx, next_name)?;
    if !next.params.is_empty() {
        return None;
    }

    // Next == LET ?empty == PosExpr \ UNION B IN
    //         \E ?e \in ?empty : B' \in ?update(?e, ?empty)
    let Expr::Let(defs, body) = &next.body.node else {
        return None;
    };
    let [empty_def] = defs.as_slice() else {
        return None;
    };
    if !empty_def.params.is_empty() {
        return None;
    }
    let empty_name = empty_def.name.node.as_str();
    let Expr::SetMinus(pos_expr, union_expr) = &empty_def.body.node else {
        return None;
    };
    let Expr::BigUnion(union_arg) = &union_expr.node else {
        return None;
    };
    if !is_board_ref(&union_arg.node, board) {
        return None;
    }
    let positions = eval_const_pair_set(ctx, pos_expr, board, MAX_GRID_CELLS)?;

    let Expr::Exists(bvs, ex_body) = &body.node else {
        return None;
    };
    let [bv] = bvs.as_slice() else {
        return None;
    };
    let (e_name, e_dom) = simple_bound_var(bv)?;
    if ident_name(e_dom) != Some(empty_name) {
        return None;
    }
    if !all_distinct(&[e_name, empty_name, board]) {
        return None;
    }
    let Expr::In(lhs, rhs) = &ex_body.node else {
        return None;
    };
    let Expr::Prime(primed) = &lhs.node else {
        return None;
    };
    if !is_board_ref(&primed.node, board) {
        return None;
    }
    let Expr::Apply(callee, args) = &rhs.node else {
        return None;
    };
    let update_name = ident_name(&callee.node)?;
    // The callee must not be shadowed by any name bound here.
    if !all_distinct(&[update_name, e_name, empty_name, board]) {
        return None;
    }
    let [a0, a1] = args.as_slice() else {
        return None;
    };
    if ident_name(&a0.node) != Some(e_name) || ident_name(&a1.node) != Some(empty_name) {
        return None;
    }

    let update = resolve_user_op(ctx, update_name)?;
    match_update(ctx, &update, board, &positions)?;

    Some(RecognizedSlide {
        board_var_idx: 0,
        positions,
    })
}

// ---------------------------------------------------------------------------
// ?update(?e, ?es)
// ---------------------------------------------------------------------------

/// Match the `update` operator (see module docs) and, transitively, the
/// `dir`/`move`/`ChooseOne` operators it calls. `positions` is the evaluated
/// `Pos` from `Next`; every other `Pos` occurrence must evaluate EQUAL to it.
fn match_update(
    ctx: &EvalCtx,
    op: &OperatorDef,
    board: &str,
    positions: &[(i64, i64)],
) -> Option<()> {
    let [pe, pes] = op.params.as_slice() else {
        return None;
    };
    if pe.arity != 0 || pes.arity != 0 {
        return None;
    }
    let (e, es) = (pe.name.node.as_str(), pes.name.node.as_str());

    let Expr::Let(defs, result) = &op.body.node else {
        return None;
    };
    let [dirs_def, moved_def, free_def] = defs.as_slice() else {
        return None;
    };
    if !dirs_def.params.is_empty() || !moved_def.params.is_empty() || !free_def.params.is_empty() {
        return None;
    }
    let dirs = dirs_def.name.node.as_str();
    let moved = moved_def.name.node.as_str();
    let free = free_def.name.node.as_str();
    if !all_distinct(&[e, es, dirs, moved, free, board]) {
        return None;
    }

    // ?dirs == ?dir(?e, ?es)
    let Expr::Apply(dir_callee, dir_args) = &dirs_def.body.node else {
        return None;
    };
    let dir_name = ident_name(&dir_callee.node)?;
    if !all_distinct(&[dir_name, e, es, board]) {
        return None;
    }
    let [da0, da1] = dir_args.as_slice() else {
        return None;
    };
    if ident_name(&da0.node) != Some(e) || ident_name(&da1.node) != Some(es) {
        return None;
    }

    // ?moved == {?move(?e, ?d) : ?d \in ?dirs}
    let Expr::SetBuilder(mv_body, mv_bvs) = &moved_def.body.node else {
        return None;
    };
    let [mv_bv] = mv_bvs.as_slice() else {
        return None;
    };
    let (d, d_dom) = simple_bound_var(mv_bv)?;
    if ident_name(d_dom) != Some(dirs) {
        return None;
    }
    if !all_distinct(&[d, e, es, dirs, board]) {
        return None;
    }
    let Expr::Apply(mv_callee, mv_args) = &mv_body.node else {
        return None;
    };
    let move_name = ident_name(&mv_callee.node)?;
    if !all_distinct(&[move_name, d, e, board]) {
        return None;
    }
    let [ma0, ma1] = mv_args.as_slice() else {
        return None;
    };
    if ident_name(&ma0.node) != Some(e) || ident_name(&ma1.node) != Some(d) {
        return None;
    }

    // ?free == {<<?pc, ?m>> \in ?moved : /\ ?m \cap (UNION (B \ {?pc})) = {}
    //                                    /\ \A ?p \in ?m : ?p \in Pos}
    let Expr::SetFilter(f_bv, f_pred) = &free_def.body.node else {
        return None;
    };
    let (pc, m, f_dom) = pair_bound_var(f_bv)?;
    if ident_name(f_dom) != Some(moved) {
        return None;
    }
    if !all_distinct(&[pc, m, board]) {
        return None;
    }
    let Expr::And(collision, boundary) = &f_pred.node else {
        return None;
    };
    // Collision filter: ?m \cap (UNION (B \ {?pc})) = {}
    {
        let Expr::Eq(cap, empty_lit) = &collision.node else {
            return None;
        };
        if !is_empty_set_literal(&empty_lit.node) {
            return None;
        }
        let Expr::Intersect(mm, others) = &cap.node else {
            return None;
        };
        if ident_name(&mm.node) != Some(m) {
            return None;
        }
        let Expr::BigUnion(rest) = &others.node else {
            return None;
        };
        let Expr::SetMinus(b, pc_set) = &rest.node else {
            return None;
        };
        if !is_board_ref(&b.node, board) {
            return None;
        }
        if ident_name(singleton_element(&pc_set.node)?) != Some(pc) {
            return None;
        }
    }
    // Boundary filter: \A ?p \in ?m : ?p \in Pos
    {
        let Expr::Forall(b_bvs, b_body) = &boundary.node else {
            return None;
        };
        let [b_bv] = b_bvs.as_slice() else {
            return None;
        };
        let (p, p_dom) = simple_bound_var(b_bv)?;
        if ident_name(p_dom) != Some(m) {
            return None;
        }
        if !all_distinct(&[p, pc, m, board]) {
            return None;
        }
        let Expr::In(pp, pos2) = &b_body.node else {
            return None;
        };
        if ident_name(&pp.node) != Some(p) {
            return None;
        }
        if eval_const_pair_set(ctx, pos2, board, MAX_GRID_CELLS)?.as_slice() != positions {
            return None;
        }
    }

    // Result: {(B \ {?pc}) \cup {?m} : <<?pc, ?m>> \in ?free}
    {
        let Expr::SetBuilder(r_body, r_bvs) = &result.node else {
            return None;
        };
        let [r_bv] = r_bvs.as_slice() else {
            return None;
        };
        let (pc2, m2, r_dom) = pair_bound_var(r_bv)?;
        if ident_name(r_dom) != Some(free) {
            return None;
        }
        if !all_distinct(&[pc2, m2, board]) {
            return None;
        }
        let Expr::Union(without_pc, m_set) = &r_body.node else {
            return None;
        };
        let Expr::SetMinus(b, pc_set) = &without_pc.node else {
            return None;
        };
        if !is_board_ref(&b.node, board) {
            return None;
        }
        if ident_name(singleton_element(&pc_set.node)?) != Some(pc2) {
            return None;
        }
        if ident_name(singleton_element(&m_set.node)?) != Some(m2) {
            return None;
        }
    }

    let dir_op = resolve_user_op(ctx, dir_name)?;
    match_dir(ctx, &dir_op, board, positions)?;
    let move_op = resolve_user_op(ctx, move_name)?;
    match_move(ctx, &move_op, board)?;
    Some(())
}

// ---------------------------------------------------------------------------
// ?dir(?p, ?es)
// ---------------------------------------------------------------------------

/// Match the `dir` operator: the constant direction set must evaluate to
/// EXACTLY the four von Neumann unit vectors, and the filter must keep `d`
/// iff `p+d \in Pos /\ p+d \notin es` (i.e. `p+d` is on-grid and occupied —
/// the kernel reference's "s occupied" test).
fn match_dir(ctx: &EvalCtx, op: &OperatorDef, board: &str, positions: &[(i64, i64)]) -> Option<()> {
    let [pp, pes] = op.params.as_slice() else {
        return None;
    };
    if pp.arity != 0 || pes.arity != 0 {
        return None;
    }
    let (p, es) = (pp.name.node.as_str(), pes.name.node.as_str());

    let Expr::Let(defs, body) = &op.body.node else {
        return None;
    };
    let [dset_def] = defs.as_slice() else {
        return None;
    };
    if !dset_def.params.is_empty() {
        return None;
    }
    let dset = dset_def.name.node.as_str();
    if !all_distinct(&[p, es, dset, board]) {
        return None;
    }
    // THE unit-vector proof: the direction set evaluates to exactly D4.
    let mut dirs = eval_const_pair_set(ctx, &dset_def.body, board, 8)?;
    dirs.sort_unstable();
    if dirs.as_slice() != VON_NEUMANN_DIRS {
        return None;
    }

    let Expr::SetFilter(bv, pred) = &body.node else {
        return None;
    };
    let (d, d_dom) = simple_bound_var(bv)?;
    if ident_name(d_dom) != Some(dset) {
        return None;
    }
    if !all_distinct(&[d, p, es, board]) {
        return None;
    }
    let Expr::And(on_grid, not_empty) = &pred.node else {
        return None;
    };
    let Expr::In(t1, pos3) = &on_grid.node else {
        return None;
    };
    if !is_step_tuple(&t1.node, p, d, StepOp::Add) {
        return None;
    }
    if eval_const_pair_set(ctx, pos3, board, MAX_GRID_CELLS)?.as_slice() != positions {
        return None;
    }
    let Expr::NotIn(t2, es_ref) = &not_empty.node else {
        return None;
    };
    if !is_step_tuple(&t2.node, p, d, StepOp::Add) {
        return None;
    }
    if ident_name(&es_ref.node) != Some(es) {
        return None;
    }
    Some(())
}

// ---------------------------------------------------------------------------
// ?move(?p, ?d)
// ---------------------------------------------------------------------------

/// Match the `move` operator: `s = p+d`, `pc` = the UNIQUE piece containing
/// `s` (via a unique-CHOOSE helper), result `<<pc, translate(pc, -d)>>` with a
/// componentwise UNIT translation.
fn match_move(ctx: &EvalCtx, op: &OperatorDef, board: &str) -> Option<()> {
    let [pp, pd] = op.params.as_slice() else {
        return None;
    };
    if pp.arity != 0 || pd.arity != 0 {
        return None;
    }
    let (p, d) = (pp.name.node.as_str(), pd.name.node.as_str());

    let Expr::Let(defs, body) = &op.body.node else {
        return None;
    };
    let [s_def, pc_def] = defs.as_slice() else {
        return None;
    };
    if !s_def.params.is_empty() || !pc_def.params.is_empty() {
        return None;
    }
    let s = s_def.name.node.as_str();
    let pc = pc_def.name.node.as_str();
    if !all_distinct(&[p, d, s, pc, board]) {
        return None;
    }

    // ?s == <<?p[1]+?d[1], ?p[2]+?d[2]>>
    if !is_step_tuple(&s_def.body.node, p, d, StepOp::Add) {
        return None;
    }

    // ?pc == ?ch(B, LAMBDA ?c : ?s \in ?c)
    {
        let Expr::Apply(ch_callee, ch_args) = &pc_def.body.node else {
            return None;
        };
        let ch_name = ident_name(&ch_callee.node)?;
        if !all_distinct(&[ch_name, p, d, s, board]) {
            return None;
        }
        let [b_arg, lam] = ch_args.as_slice() else {
            return None;
        };
        if !is_board_ref(&b_arg.node, board) {
            return None;
        }
        let Expr::Lambda(lam_params, lam_body) = &lam.node else {
            return None;
        };
        let [c] = lam_params.as_slice() else {
            return None;
        };
        let c = c.node.as_str();
        // `?s` inside the lambda must still denote the LET `s` (no shadowing).
        if !all_distinct(&[c, s, board]) {
            return None;
        }
        let Expr::In(sl, cl) = &lam_body.node else {
            return None;
        };
        if ident_name(&sl.node) != Some(s) || ident_name(&cl.node) != Some(c) {
            return None;
        }
        let ch_op = resolve_user_op(ctx, ch_name)?;
        match_unique_choose(&ch_op)?;
    }

    // <<?pc, {<<?q[1]-?d[1], ?q[2]-?d[2]>> : ?q \in ?pc}>>
    {
        let Expr::Tuple(items) = &body.node else {
            return None;
        };
        let [pc_ref, translated] = items.as_slice() else {
            return None;
        };
        if ident_name(&pc_ref.node) != Some(pc) {
            return None;
        }
        let Expr::SetBuilder(t_body, t_bvs) = &translated.node else {
            return None;
        };
        let [t_bv] = t_bvs.as_slice() else {
            return None;
        };
        let (q, q_dom) = simple_bound_var(t_bv)?;
        if ident_name(q_dom) != Some(pc) {
            return None;
        }
        if !all_distinct(&[q, p, d, s, pc, board]) {
            return None;
        }
        if !is_step_tuple(&t_body.node, q, d, StepOp::Sub) {
            return None;
        }
    }
    Some(())
}

// ---------------------------------------------------------------------------
// ?ch(?S, ?P(_)) — unique CHOOSE
// ---------------------------------------------------------------------------

/// Match `?ch(?S, ?P(_)) == CHOOSE ?x \in ?S : ?P(?x) /\ \A ?y \in ?S :
/// ?P(?y) => ?y = ?x` — "the unique element satisfying P". Uniqueness is what
/// lets the kernel find the piece by scan: on a disjoint board the piece
/// containing `s` is unique, so CHOOSE-of-the-unique-witness and first-match
/// agree. A plain `CHOOSE x \in S : P(x)` does NOT prove uniqueness and is
/// rejected.
fn match_unique_choose(op: &OperatorDef) -> Option<()> {
    let [ps, pf] = op.params.as_slice() else {
        return None;
    };
    if ps.arity != 0 || pf.arity != 1 {
        return None;
    }
    let (s, f) = (ps.name.node.as_str(), pf.name.node.as_str());

    let Expr::Choose(bv, pred) = &op.body.node else {
        return None;
    };
    let (x, x_dom) = simple_bound_var(bv)?;
    if ident_name(x_dom) != Some(s) {
        return None;
    }
    if !all_distinct(&[x, s, f]) {
        return None;
    }
    let Expr::And(px, uniq) = &pred.node else {
        return None;
    };
    if !is_unary_apply(&px.node, f, x) {
        return None;
    }
    let Expr::Forall(u_bvs, u_body) = &uniq.node else {
        return None;
    };
    let [u_bv] = u_bvs.as_slice() else {
        return None;
    };
    let (y, y_dom) = simple_bound_var(u_bv)?;
    if ident_name(y_dom) != Some(s) {
        return None;
    }
    if !all_distinct(&[y, x, s, f]) {
        return None;
    }
    let Expr::Implies(py, eq) = &u_body.node else {
        return None;
    };
    if !is_unary_apply(&py.node, f, y) {
        return None;
    }
    let Expr::Eq(l, r) = &eq.node else {
        return None;
    };
    if ident_name(&l.node) != Some(y) || ident_name(&r.node) != Some(x) {
        return None;
    }
    Some(())
}

// ---------------------------------------------------------------------------
// Shape helpers
// ---------------------------------------------------------------------------

/// `?f(?a)` with `f`/`a` given by name.
fn is_unary_apply(e: &Expr, f: &str, a: &str) -> bool {
    let Expr::Apply(callee, args) = e else {
        return false;
    };
    if ident_name(&callee.node) != Some(f) {
        return false;
    }
    let [arg] = args.as_slice() else {
        return false;
    };
    ident_name(&arg.node) == Some(a)
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum StepOp {
    Add,
    Sub,
}

/// `<<?a[1] ± ?b[1], ?a[2] ± ?b[2]>>` — the componentwise UNIT translation of
/// a 2-int tuple. Component `k` must apply BOTH `a` and `b` at index `k`.
fn is_step_tuple(e: &Expr, a: &str, b: &str, op: StepOp) -> bool {
    let Expr::Tuple(items) = e else {
        return false;
    };
    let [c1, c2] = items.as_slice() else {
        return false;
    };
    is_step_component(&c1.node, a, b, 1, op) && is_step_component(&c2.node, a, b, 2, op)
}

/// `?a[k] ± ?b[k]`.
fn is_step_component(e: &Expr, a: &str, b: &str, k: i64, op: StepOp) -> bool {
    let (l, r) = match (op, e) {
        (StepOp::Add, Expr::Add(l, r)) | (StepOp::Sub, Expr::Sub(l, r)) => (l, r),
        _ => return false,
    };
    is_indexed_apply(&l.node, a, k) && is_indexed_apply(&r.node, b, k)
}

/// `?name[k]` — `FuncApply(Ident(name), Int(k))`.
fn is_indexed_apply(e: &Expr, name: &str, k: i64) -> bool {
    let Expr::FuncApply(f, arg) = e else {
        return false;
    };
    if ident_name(&f.node) != Some(name) {
        return false;
    }
    let Expr::Int(i) = &arg.node else {
        return false;
    };
    i.to_i64() == Some(k)
}

/// The name of a PLAIN identifier (not a state-var node — the board is matched
/// separately via [`is_board_ref`], never through this).
fn ident_name(e: &Expr) -> Option<&str> {
    match e {
        Expr::Ident(name, _) => Some(name.as_str()),
        _ => None,
    }
}

/// Does `e` reference the (single) state variable? Both the pre-resolution
/// form (`Ident("board")`, as parsed) and the post-`resolve_state_vars` form
/// (`StateVar("board", 0, _)`, what the running checker's ops contain) are
/// accepted. Sound because every matched local name is required distinct from
/// the state-variable name, so an `Ident` bearing the board's name can only
/// denote the board.
fn is_board_ref(e: &Expr, board: &str) -> bool {
    match e {
        Expr::Ident(name, _) => name == board,
        Expr::StateVar(name, idx, _) => *idx == 0 && name == board,
        _ => false,
    }
}

/// A simple bound var `?x \in dom` (no tuple pattern) → `(name, domain)`.
fn simple_bound_var(bv: &BoundVar) -> Option<(&str, &Expr)> {
    if bv.pattern.is_some() {
        return None;
    }
    let dom = bv.domain.as_ref()?;
    Some((bv.name.node.as_str(), &dom.node))
}

/// A pair-destructuring bound var `<<?a, ?b>> \in dom` → `(a, b, domain)`.
fn pair_bound_var(bv: &BoundVar) -> Option<(&str, &str, &Expr)> {
    let Some(BoundPattern::Tuple(names)) = &bv.pattern else {
        return None;
    };
    let [a, b] = names.as_slice() else {
        return None;
    };
    let dom = bv.domain.as_ref()?;
    Some((a.node.as_str(), b.node.as_str(), &dom.node))
}

/// The literal empty set `{}`.
fn is_empty_set_literal(e: &Expr) -> bool {
    matches!(e, Expr::SetEnum(items) if items.is_empty())
}

/// `{x}` → `x`.
fn singleton_element(e: &Expr) -> Option<&Expr> {
    let Expr::SetEnum(items) = e else {
        return None;
    };
    let [x] = items.as_slice() else {
        return None;
    };
    Some(&x.node)
}

/// All names pairwise distinct (the scope-hygiene requirement that makes name
/// equality coincide with lexical resolution — see module docs).
fn all_distinct(names: &[&str]) -> bool {
    for (i, a) in names.iter().enumerate() {
        for b in &names[i + 1..] {
            if a == b {
                return false;
            }
        }
    }
    true
}

/// Resolve a user operator by name, refusing anything the interpreter would
/// resolve DIFFERENTLY than a plain `shared.ops` lookup: a config operator
/// replacement (`CONSTANT Op <- Other`) or a config-constant override would
/// make the structural match a proof about the WRONG definition.
fn resolve_user_op(ctx: &EvalCtx, name: &str) -> Option<std::sync::Arc<OperatorDef>> {
    if ctx.shared().op_replacements.contains_key(name)
        || ctx.shared().config_constants.contains(name)
    {
        return None;
    }
    ctx.get_op(name).cloned()
}

// ---------------------------------------------------------------------------
// Constant `Pos` / direction-set evaluation
// ---------------------------------------------------------------------------

/// Prove `expr` state-free, then EVALUATE it with the checker's own evaluator
/// (the const-fold machinery — config constants, zero-arg operator folding and
/// lazy set values all behave exactly as at run time) and decode it as a
/// finite set of 2-int tuples with `1..=cap` elements. Returns the sorted,
/// deduped cells. `None` = fail closed.
fn eval_const_pair_set(
    ctx: &EvalCtx,
    expr: &Spanned<Expr>,
    board: &str,
    cap: usize,
) -> Option<Vec<(i64, i64)>> {
    let mut budget = STATE_FREE_NODE_BUDGET;
    let mut visiting: Vec<String> = Vec::new();
    if !expr_state_free(ctx, &expr.node, board, &mut visiting, &mut budget, 0) {
        return None;
    }
    let value = tla_eval::eval(ctx, expr).ok()?;
    // Cap BEFORE iterating: a lazy set (e.g. `SUBSET S`) must not be
    // enumerated past the grid bound.
    let len = value.set_len()?.to_usize()?;
    if len == 0 || len > cap {
        return None;
    }
    let mut cells: Vec<(i64, i64)> = Vec::with_capacity(len);
    for elem in value.iter_set()? {
        cells.push(value_to_pos(&elem)?);
    }
    cells.sort_unstable();
    cells.dedup();
    if cells.is_empty() || cells.len() > cap {
        return None;
    }
    Some(cells)
}

/// Bounded TRANSITIVE state-freedom proof: no state variable, no prime, no
/// action/temporal construct, no INSTANCE indirection — following operator
/// references through `ctx` (cycle- and depth-capped, fail closed). A `true`
/// verdict means evaluating the expression can only ever read constants, so
/// its value at arm time equals its value at every state — the property that
/// makes the evaluated `Pos`/direction sets legitimate compile-time constants.
///
/// Conservative by construction: anything unrecognized is `false` (no arm).
fn expr_state_free(
    ctx: &EvalCtx,
    e: &Expr,
    board: &str,
    visiting: &mut Vec<String>,
    budget: &mut usize,
    depth: usize,
) -> bool {
    if *budget == 0 || depth > STATE_FREE_MAX_DEPTH {
        return false;
    }
    *budget -= 1;

    // Sub-expression convenience.
    macro_rules! free {
        ($sub:expr) => {
            expr_state_free(ctx, &$sub.node, board, visiting, budget, depth)
        };
    }

    match e {
        // -- state / action constructs: never constant --------------------
        Expr::StateVar(..)
        | Expr::Prime(_)
        | Expr::Enabled(_)
        | Expr::Unchanged(_)
        | Expr::Always(_)
        | Expr::Eventually(_)
        | Expr::LeadsTo(..)
        | Expr::WeakFair(..)
        | Expr::StrongFair(..)
        // -- indirection we do not chase: fail closed ----------------------
        | Expr::ModuleRef(..)
        | Expr::InstanceExpr(..)
        | Expr::SubstIn(..)
        | Expr::Label(_)
        | Expr::OpRef(_) => false,

        // -- literals -------------------------------------------------------
        Expr::Bool(_) | Expr::Int(_) | Expr::String(_) => true,

        // -- names ----------------------------------------------------------
        Expr::Ident(name, _) => {
            if name == board {
                return false;
            }
            // Config-constant override: the env binding is a constant VALUE.
            if ctx.shared().config_constants.contains(name.as_str()) {
                return true;
            }
            // A replaced operator resolves elsewhere — refuse to reason.
            if ctx.shared().op_replacements.contains_key(name.as_str()) {
                return false;
            }
            match ctx.get_op(name) {
                Some(op) => {
                    if !op.params.is_empty() || op.contains_prime {
                        return false;
                    }
                    if visiting.iter().any(|n| n == name) {
                        return false; // recursive constant: fail closed
                    }
                    let op = op.clone();
                    visiting.push(name.clone());
                    let ok = expr_state_free(
                        ctx,
                        &op.body.node,
                        board,
                        visiting,
                        budget,
                        depth + 1,
                    );
                    visiting.pop();
                    ok
                }
                // Not an operator: a declared CONSTANT (env value), a builtin
                // constant set, or a local binder — all state-free. (A state
                // variable is impossible: the single variable is `board`,
                // rejected above.)
                None => true,
            }
        }

        // -- operator application --------------------------------------------
        Expr::Apply(callee, args) => {
            let Expr::Ident(name, _) = &callee.node else {
                return false;
            };
            if name == board || ctx.shared().op_replacements.contains_key(name.as_str()) {
                return false;
            }
            let Some(op) = ctx.get_op(name) else {
                // Unresolvable callee (builtin operator or a higher-order
                // parameter): we cannot inspect its body — fail closed.
                return false;
            };
            if op.params.len() != args.len() || op.contains_prime {
                return false;
            }
            if visiting.iter().any(|n| n == name) {
                return false;
            }
            let op = op.clone();
            visiting.push(name.clone());
            let ok = expr_state_free(ctx, &op.body.node, board, visiting, budget, depth + 1);
            visiting.pop();
            ok && args.iter().all(|a| free!(a))
        }

        // -- pure compound expressions: constant iff all children are -------
        Expr::Lambda(_, body) | Expr::Not(body) | Expr::Neg(body) | Expr::Powerset(body)
        | Expr::BigUnion(body) | Expr::Domain(body) => free!(body),
        Expr::RecordAccess(body, _) => free!(body),
        Expr::And(a, b)
        | Expr::Or(a, b)
        | Expr::Implies(a, b)
        | Expr::Equiv(a, b)
        | Expr::In(a, b)
        | Expr::NotIn(a, b)
        | Expr::Subseteq(a, b)
        | Expr::Union(a, b)
        | Expr::Intersect(a, b)
        | Expr::SetMinus(a, b)
        | Expr::FuncApply(a, b)
        | Expr::FuncSet(a, b)
        | Expr::Eq(a, b)
        | Expr::Neq(a, b)
        | Expr::Lt(a, b)
        | Expr::Leq(a, b)
        | Expr::Gt(a, b)
        | Expr::Geq(a, b)
        | Expr::Add(a, b)
        | Expr::Sub(a, b)
        | Expr::Mul(a, b)
        | Expr::Div(a, b)
        | Expr::IntDiv(a, b)
        | Expr::Mod(a, b)
        | Expr::Pow(a, b)
        | Expr::Range(a, b) => free!(a) && free!(b),
        Expr::If(c, t, f) => free!(c) && free!(t) && free!(f),
        Expr::SetEnum(items) | Expr::Tuple(items) | Expr::Times(items) => {
            items.iter().all(|i| free!(i))
        }
        Expr::Record(fields) | Expr::RecordSet(fields) => {
            fields.iter().all(|(_, v)| free!(v))
        }
        Expr::SetBuilder(body, bvs) | Expr::FuncDef(bvs, body) => {
            bvs.iter().all(|bv| bound_var_state_free(ctx, bv, board, visiting, budget, depth))
                && free!(body)
        }
        Expr::SetFilter(bv, body) => {
            bound_var_state_free(ctx, bv, board, visiting, budget, depth) && free!(body)
        }
        Expr::Forall(bvs, body) | Expr::Exists(bvs, body) => {
            bvs.iter().all(|bv| bound_var_state_free(ctx, bv, board, visiting, budget, depth))
                && free!(body)
        }
        Expr::Choose(bv, body) => {
            bound_var_state_free(ctx, bv, board, visiting, budget, depth) && free!(body)
        }
        Expr::Except(base, specs) => {
            free!(base)
                && specs.iter().all(|s| {
                    free!(s.value)
                        && s.path.iter().all(|p| match p {
                            tla_core::ast::ExceptPathElement::Index(ix) => free!(ix),
                            tla_core::ast::ExceptPathElement::Field(_) => true,
                        })
                })
        }
        Expr::Case(arms, other) => {
            arms.iter().all(|arm| free!(arm.guard) && free!(arm.body))
                && other.as_ref().is_none_or(|o| free!(o))
        }
        Expr::Let(defs, body) => {
            defs.iter().all(|d| {
                expr_state_free(ctx, &d.body.node, board, visiting, budget, depth)
            }) && free!(body)
        }
    }
}

/// State-freedom of a bound var's domain (if any).
fn bound_var_state_free(
    ctx: &EvalCtx,
    bv: &BoundVar,
    board: &str,
    visiting: &mut Vec<String>,
    budget: &mut usize,
    depth: usize,
) -> bool {
    match &bv.domain {
        Some(dom) => expr_state_free(ctx, &dom.node, board, visiting, budget, depth),
        None => false, // unbounded CHOOSE/quantifier: not a finite constant
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::check::model_checker::ModelChecker;
    use crate::Config;
    use tla_core::{lower, parse_to_syntax_tree, FileId};

    /// The real SlidingPuzzles spec (tlaplus/Examples), verbatim.
    const SLIDING_PUZZLES: &str = r#"
--------------------------- MODULE SlidingPuzzles ---------------------------
EXTENDS Integers

VARIABLE board

W == 4 H == 5
Pos == (0 .. W - 1) \X (0 .. H - 1)
Piece == SUBSET Pos

Klotski == {{<<0, 0>>, <<0, 1>>},
            {<<1, 0>>, <<2, 0>>, <<1, 1>>, <<2, 1>>},
            {<<3, 0>>, <<3, 1>>},{<<0, 2>>, <<0, 3>>},
            {<<1, 2>>, <<2, 2>>},{<<3, 2>>, <<3, 3>>},
            {<<1, 3>>}, {<<2, 3>>}, {<<0, 4>>}, {<<3, 4>>}}

KlotskiGoal == {<<1, 3>>, <<1, 4>>, <<2, 3>>, <<2, 4>>} \notin board

ChooseOne(S, P(_)) == CHOOSE x \in S : P(x) /\ \A y \in S : P(y) => y = x

TypeOK == board \in SUBSET Piece

dir(p, es) == LET dir == {<<1, 0>>, <<0, 1>>, <<-1, 0>>, <<0, -1>>}
              IN {d \in dir : /\ <<p[1] + d[1], p[2] + d[2]>> \in Pos
                              /\ <<p[1] + d[1], p[2] + d[2]>> \notin es}

move(p, d) == LET s == <<p[1] + d[1], p[2] + d[2]>>
                  pc == ChooseOne(board, LAMBDA pc : s \in pc)
              IN <<pc, {<<q[1] - d[1], q[2] - d[2]>> : q \in pc}>>

update(e, es) == LET dirs  == dir(e, es)
                     moved == {move(e, d) : d \in dirs}
                     free  == {<<pc, m>> \in moved :
                                 /\ m \cap (UNION (board \ {pc})) = {}
                                 /\ \A p \in m : p \in Pos}
                 IN {(board \ {pc}) \cup {m} : <<pc, m>> \in free}

Init == board = Klotski

Next == LET empty == Pos \ UNION board
        IN  \E e \in empty : board' \in update(e, empty)

=============================================================================
"#;

    fn parse_module(src: &str) -> tla_core::ast::Module {
        let tree = parse_to_syntax_tree(src);
        let result = lower(FileId(0), &tree);
        let mut module = result.module.expect("parse failed");
        tla_core::compute_is_recursive(&mut module);
        module
    }

    fn config() -> Config {
        Config {
            init: Some("Init".to_string()),
            next: Some("Next".to_string()),
            invariants: vec!["KlotskiGoal".to_string()],
            ..Default::default()
        }
    }

    /// Run the recognizer against a spec source through the REAL checker
    /// setup (state-var resolution, constant precompute — the exact ops the
    /// BFS-time arm sees).
    fn recognize(src: &str) -> Option<RecognizedSlide> {
        let module = parse_module(src);
        let config = config();
        let checker = ModelChecker::new(&module, &config);
        recognize_slide_next(&checker.ctx, "Next")
    }

    /// A variant of the base spec with `old` replaced by `new` (asserting the
    /// pattern actually occurred, so probes can't silently no-op).
    fn variant(old: &str, new: &str) -> String {
        assert!(
            SLIDING_PUZZLES.contains(old),
            "probe pattern not found: {old}"
        );
        SLIDING_PUZZLES.replace(old, new)
    }

    // -- positive ----------------------------------------------------------

    #[test]
    fn recognizes_sliding_puzzles_and_evaluates_exact_pos() {
        let rec = recognize(SLIDING_PUZZLES).expect("SlidingPuzzles must be recognized");
        assert_eq!(rec.board_var_idx, 0);
        // The evaluated Pos is the exact 4×5 grid — 20 cells, sorted.
        let mut expect: Vec<(i64, i64)> = Vec::new();
        for x in 0..4 {
            for y in 0..5 {
                expect.push((x, y));
            }
        }
        expect.sort_unstable();
        assert_eq!(rec.positions, expect);
    }

    /// Pos EVALUATION vs the INIT bounding box: on a 4×6 grid the Klotski
    /// init spans only y ∈ 0..4, so the bounding-box heuristic under-covers
    /// (20 cells) while the recognizer proves the TRUE 24-cell grid — the
    /// case where a piece may slide OUTSIDE the init bounding box.
    #[test]
    fn pos_evaluation_beats_init_bounding_box_on_taller_grid() {
        let src = variant("W == 4 H == 5", "W == 4 H == 6");
        let rec = recognize(&src).expect("taller grid must still be recognized");
        assert_eq!(rec.positions.len(), 24, "true Pos is 4×6 = 24 cells");

        // The bounding-box arm (the opt-in shortcut) derives only 4×5 = 20
        // cells from the same init — the recognizer's grid is strictly larger.
        let geo = crate::state::SlideGeometry::rectangular(4, 5).unwrap();
        let init_pieces: &[&[(i64, i64)]] = &[
            &[(0, 0), (0, 1)],
            &[(1, 0), (2, 0), (1, 1), (2, 1)],
            &[(3, 0), (3, 1)],
            &[(0, 2), (0, 3)],
            &[(1, 2), (2, 2)],
            &[(3, 2), (3, 3)],
            &[(1, 3)],
            &[(2, 3)],
            &[(0, 4)],
            &[(3, 4)],
        ];
        let masks: Vec<u64> = init_pieces
            .iter()
            .map(|cells| geo.cells_to_mask(cells).unwrap())
            .collect();
        let init_value = geo.masks_to_board(&masks);
        let bbox_arm = crate::state::SlideKernelArm::try_arm(0, &[&init_value])
            .expect("bounding-box arm still arms");
        assert_eq!(
            bbox_arm.geometry.num_positions(),
            20,
            "bounding box under-covers the true 24-cell Pos"
        );

        // The recognized grid arms with the full 24 cells.
        let rec_arm =
            crate::state::SlideKernelArm::try_arm_recognized(0, rec.positions, &[&init_value])
                .expect("recognized arm must accept the init board");
        assert_eq!(rec_arm.geometry.num_positions(), 24);
    }

    // -- negative probes (every deviation must NOT arm) ---------------------

    #[test]
    fn rejects_extra_conjunct_in_next() {
        let src = variant(
            "\\E e \\in empty : board' \\in update(e, empty)",
            "\\E e \\in empty : board' \\in update(e, empty) /\\ e \\in Pos",
        );
        assert!(recognize(&src).is_none(), "extra conjunct must not arm");
    }

    #[test]
    fn rejects_diagonal_direction() {
        let src = variant(
            "{<<1, 0>>, <<0, 1>>, <<-1, 0>>, <<0, -1>>}",
            "{<<1, 0>>, <<0, 1>>, <<-1, 0>>, <<0, -1>>, <<1, 1>>}",
        );
        assert!(recognize(&src).is_none(), "diagonal move must not arm");
    }

    #[test]
    fn rejects_missing_direction() {
        let src = variant(
            "{<<1, 0>>, <<0, 1>>, <<-1, 0>>, <<0, -1>>}",
            "{<<1, 0>>, <<0, 1>>, <<-1, 0>>}",
        );
        assert!(recognize(&src).is_none(), "3-direction set must not arm");
    }

    #[test]
    fn rejects_piece_swap_update() {
        let src = variant(
            "{(board \\ {pc}) \\cup {m} : <<pc, m>> \\in free}",
            "{(board \\ {m}) \\cup {pc} : <<pc, m>> \\in free}",
        );
        assert!(recognize(&src).is_none(), "piece-swap update must not arm");
    }

    #[test]
    fn rejects_non_unit_translation() {
        let src = variant(
            "{<<q[1] - d[1], q[2] - d[2]>> : q \\in pc}",
            "{<<q[1] - 2 * d[1], q[2] - 2 * d[2]>> : q \\in pc}",
        );
        assert!(recognize(&src).is_none(), "2-unit translation must not arm");
    }

    #[test]
    fn rejects_mismatched_component_indices() {
        let src = variant(
            "s == <<p[1] + d[1], p[2] + d[2]>>",
            "s == <<p[1] + d[2], p[2] + d[1]>>",
        );
        assert!(recognize(&src).is_none(), "swapped indices must not arm");
    }

    #[test]
    fn rejects_plain_choose_without_uniqueness() {
        let src = variant(
            "ChooseOne(board, LAMBDA pc : s \\in pc)",
            "CHOOSE c \\in board : s \\in c",
        );
        assert!(
            recognize(&src).is_none(),
            "plain CHOOSE (no uniqueness proof) must not arm"
        );
    }

    #[test]
    fn rejects_dropped_boundary_filter() {
        let src = variant(
            "free  == {<<pc, m>> \\in moved :
                                 /\\ m \\cap (UNION (board \\ {pc})) = {}
                                 /\\ \\A p \\in m : p \\in Pos}",
            "free  == {<<pc, m>> \\in moved :
                                 m \\cap (UNION (board \\ {pc})) = {}}",
        );
        assert!(
            recognize(&src).is_none(),
            "missing boundary filter must not arm"
        );
    }

    #[test]
    fn rejects_swapped_filter_conjuncts() {
        let src = variant(
            "free  == {<<pc, m>> \\in moved :
                                 /\\ m \\cap (UNION (board \\ {pc})) = {}
                                 /\\ \\A p \\in m : p \\in Pos}",
            "free  == {<<pc, m>> \\in moved :
                                 /\\ \\A p \\in m : p \\in Pos
                                 /\\ m \\cap (UNION (board \\ {pc})) = {}}",
        );
        assert!(
            recognize(&src).is_none(),
            "reordered filter conjuncts deviate from the proven shape"
        );
    }

    #[test]
    fn rejects_state_dependent_pos() {
        let src = variant(
            "Pos == (0 .. W - 1) \\X (0 .. H - 1)",
            "Pos == ((0 .. W - 1) \\X (0 .. H - 1)) \\cup UNION board",
        );
        assert!(
            recognize(&src).is_none(),
            "state-dependent Pos must not arm"
        );
    }

    #[test]
    fn rejects_grid_over_64_cells() {
        let src = variant("W == 4 H == 5", "W == 9 H == 9");
        assert!(recognize(&src).is_none(), "81-cell grid must not arm");
    }

    #[test]
    fn rejects_multi_variable_spec() {
        let src = variant("VARIABLE board", "VARIABLE board, other")
            .replace(
                "Init == board = Klotski",
                "Init == board = Klotski /\\ other = 0",
            )
            .replace(
                "\\E e \\in empty : board' \\in update(e, empty)",
                "\\E e \\in empty : board' \\in update(e, empty) /\\ other' = other",
            );
        assert!(recognize(&src).is_none(), "second state var must not arm");
    }

    #[test]
    fn rejects_wrong_move_argument_order() {
        let src = variant(
            "moved == {move(e, d) : d \\in dirs}",
            "moved == {move(d, e) : d \\in dirs}",
        );
        assert!(recognize(&src).is_none(), "swapped move args must not arm");
    }

    #[test]
    fn rejects_translation_toward_d_instead_of_away() {
        // `+ d` instead of `- d`: the piece would move AWAY from the empty
        // cell — a different relation.
        let src = variant(
            "{<<q[1] - d[1], q[2] - d[2]>> : q \\in pc}",
            "{<<q[1] + d[1], q[2] + d[2]>> : q \\in pc}",
        );
        assert!(
            recognize(&src).is_none(),
            "inverted translation must not arm"
        );
    }

    // -- arm-time value validation ------------------------------------------

    // -- end-to-end DEFAULT arming through the real checker ------------------

    /// A tiny 3×3 slide spec (one domino + two singles) — small enough that
    /// the INTERPRETER full closure runs in milliseconds, so armed and
    /// un-armed runs can be compared exactly in a unit test.
    fn small_grid_spec() -> String {
        variant("W == 4 H == 5", "W == 3 H == 3").replace(
            r#"Klotski == {{<<0, 0>>, <<0, 1>>},
            {<<1, 0>>, <<2, 0>>, <<1, 1>>, <<2, 1>>},
            {<<3, 0>>, <<3, 1>>},{<<0, 2>>, <<0, 3>>},
            {<<1, 2>>, <<2, 2>>},{<<3, 2>>, <<3, 3>>},
            {<<1, 3>>}, {<<2, 3>>}, {<<0, 4>>}, {<<3, 4>>}}"#,
            "Klotski == {{<<0, 0>>, <<0, 1>>}, {<<1, 0>>}, {<<2, 0>>}}",
        )
    }

    /// Run a spec to completion with TypeOK-free closure semantics (no
    /// invariants), returning `(armed, states_found)`.
    fn run_closure(src: &str) -> (bool, usize) {
        let module = parse_module(src);
        let config = Config {
            init: Some("Init".to_string()),
            next: Some("Next".to_string()),
            ..Default::default()
        };
        let mut checker = ModelChecker::new(&module, &config);
        let result = checker.check();
        let stats = match &result {
            crate::CheckResult::Success(stats) => stats,
            other => panic!("expected closure success, got {other:?}"),
        };
        (
            checker.nested_set_slide_armed_for_testing(),
            stats.states_found,
        )
    }

    /// DEFAULT-ON end-to-end: the small slide spec ARMS with no env var and
    /// its full closure equals the interpreter's closure of the semantically
    /// IDENTICAL spec whose `Next` carries a redundant extra conjunct
    /// (`e \in Pos`, a no-op since `empty \subseteq Pos`) — which the
    /// recognizer must NOT arm. Same state space, two engines, equal counts.
    #[test]
    fn default_arm_matches_interpreter_closure_on_small_grid() {
        let armed_src = small_grid_spec();
        let (armed, armed_states) = run_closure(&armed_src);
        assert!(armed, "small slide spec must auto-arm by default");

        let interp_src = armed_src.replace(
            "\\E e \\in empty : board' \\in update(e, empty)",
            "\\E e \\in empty : board' \\in update(e, empty) /\\ e \\in Pos",
        );
        assert_ne!(armed_src, interp_src, "probe replace must apply");
        let (interp_armed, interp_states) = run_closure(&interp_src);
        assert!(
            !interp_armed,
            "extra-conjunct variant must stay on the interpreter"
        );
        assert_eq!(
            armed_states, interp_states,
            "kernel closure must equal interpreter closure exactly"
        );
        assert!(armed_states > 3, "closure must actually explore");
    }

    /// ALIAS is trace-OUTPUT-only in both TLC and ty: `Trace::apply_alias`
    /// runs post-check (from the CLI) on the reconstructed trace's concrete
    /// states, never through the successor machinery the kernel replaces. An
    /// `ALIAS` config therefore must NOT block the default arm (this is the
    /// `SlidingPuzzles_anim` shape: the inherited slide `Next` plus
    /// `ALIAS AnimAlias`). The aliased run must arm AND produce the exact
    /// alias-free closure.
    #[test]
    fn alias_config_still_arms_and_closure_is_exact() {
        // Ground truth: the alias-free closure (arms by default).
        let (armed_plain, plain_states) = run_closure(&small_grid_spec());
        assert!(armed_plain, "alias-free small grid must arm");

        let src = small_grid_spec().replace(
            "Init == board = Klotski",
            "BoardAlias == [board |-> board]\nInit == board = Klotski",
        );
        assert!(src.contains("BoardAlias"), "alias op insertion must apply");
        let module = parse_module(&src);
        let config = Config {
            init: Some("Init".to_string()),
            next: Some("Next".to_string()),
            alias: Some("BoardAlias".to_string()),
            ..Default::default()
        };
        let mut checker = ModelChecker::new(&module, &config);
        let result = checker.check();
        let stats = match &result {
            crate::CheckResult::Success(stats) => stats,
            other => panic!("expected closure success, got {other:?}"),
        };
        assert!(
            checker.nested_set_slide_armed_for_testing(),
            "ALIAS config must not block the default arm"
        );
        assert_eq!(
            stats.states_found, plain_states,
            "aliased armed closure must equal the alias-free closure exactly"
        );
    }

    /// Fail-closed retention probe for the config gate: VIEW (which DOES
    /// change fingerprint semantics inside the successor/dedup machinery)
    /// must still block the default arm, even on a spec whose `Next` proves.
    #[test]
    fn view_config_still_blocks_default_arm() {
        let (armed_plain, plain_states) = run_closure(&small_grid_spec());
        assert!(armed_plain);

        let src = small_grid_spec().replace(
            "Init == board = Klotski",
            "BoardView == board\nInit == board = Klotski",
        );
        assert!(src.contains("BoardView"), "view op insertion must apply");
        let module = parse_module(&src);
        let config = Config {
            init: Some("Init".to_string()),
            next: Some("Next".to_string()),
            view: Some("BoardView".to_string()),
            ..Default::default()
        };
        let mut checker = ModelChecker::new(&module, &config);
        let result = checker.check();
        let stats = match &result {
            crate::CheckResult::Success(stats) => stats,
            other => panic!("expected closure success, got {other:?}"),
        };
        assert!(
            !checker.nested_set_slide_armed_for_testing(),
            "VIEW config must keep the default arm CLOSED"
        );
        // Identity view: same closure, interpreter path.
        assert_eq!(stats.states_found, plain_states);
    }

    #[test]
    fn track_only_coverage_blocks_default_arm_and_keeps_action_boundaries() {
        let (armed_plain, plain_states) = run_closure(&small_grid_spec());
        assert!(armed_plain);

        let src = small_grid_spec();
        let module = parse_module(&src);
        let config = Config {
            init: Some("Init".to_string()),
            next: Some("Next".to_string()),
            ..Default::default()
        };
        let mut checker = ModelChecker::new(&module, &config);
        checker.set_track_coverage(true);
        let result = checker.check();
        let stats = match &result {
            crate::CheckResult::Success(stats) => stats,
            other => panic!("expected closure success, got {other:?}"),
        };
        assert!(
            !checker.nested_set_slide_armed_for_testing(),
            "track-only coverage must retain the per-action interpreter route"
        );
        assert_eq!(stats.states_found, plain_states);
    }

    /// The REAL SlidingPuzzles spec: default run (no env) must arm and find
    /// the KlotskiGoal violation after exactly 24005 states — the
    /// TLC-verified count the opt-in kernel landed with.
    #[test]
    fn sliding_puzzles_arms_by_default_and_flips() {
        let module = parse_module(SLIDING_PUZZLES);
        let config = Config {
            init: Some("Init".to_string()),
            next: Some("Next".to_string()),
            invariants: vec!["TypeOK".to_string(), "KlotskiGoal".to_string()],
            ..Default::default()
        };
        let mut checker = ModelChecker::new(&module, &config);
        let result = checker.check();
        assert!(
            checker.nested_set_slide_armed_for_testing(),
            "SlidingPuzzles must arm BY DEFAULT"
        );
        match &result {
            crate::CheckResult::InvariantViolation {
                invariant, stats, ..
            } => {
                assert_eq!(invariant, "KlotskiGoal");
                assert_eq!(
                    stats.states_found, 24005,
                    "states-at-violation must match the TLC-exact count"
                );
            }
            other => panic!("expected KlotskiGoal violation, got {other:?}"),
        }
    }

    /// Negative probe END-TO-END: the diagonal-move variant must not arm and
    /// must produce ITS OWN interpreter-exact closure (different relation,
    /// different space) — proving un-recognized specs are untouched.
    #[test]
    fn diagonal_variant_runs_uninstrumented_on_interpreter() {
        let src = small_grid_spec().replace(
            "{<<1, 0>>, <<0, 1>>, <<-1, 0>>, <<0, -1>>}",
            "{<<1, 0>>, <<0, 1>>, <<-1, 0>>, <<0, -1>>, <<1, 1>>}",
        );
        let (armed, states) = run_closure(&src);
        assert!(!armed, "diagonal variant must NOT arm");
        assert!(states > 0);
    }

    /// TRIPWIRE end-to-end: inject a deliberately WRONG arm (the standard
    /// unit-slide kernel) into a checker for the DIAGONAL-move spec — a
    /// relation the kernel does not implement. The first expanded state's
    /// kernel successor set misses the diagonal moves, the tripwire fires,
    /// the kernel DISARMS, and the run completes on the interpreter with the
    /// diagonal spec's own exact closure. This is the defense-in-depth layer
    /// a hypothetical recognizer false-accept would hit.
    #[test]
    fn tripwire_disarms_wrongly_armed_kernel_and_run_stays_exact() {
        let diag_src = small_grid_spec().replace(
            "{<<1, 0>>, <<0, 1>>, <<-1, 0>>, <<0, -1>>}",
            "{<<1, 0>>, <<0, 1>>, <<-1, 0>>, <<0, -1>>, <<1, 1>>}",
        );
        // Ground truth: the diagonal spec's interpreter closure (never arms).
        let (interp_armed, interp_states) = run_closure(&diag_src);
        assert!(!interp_armed);

        // Now run the same spec with a WRONG arm injected (bypassing the
        // recognizer) and the tripwire enabled, as a default arm would be.
        let module = parse_module(&diag_src);
        let config = Config {
            init: Some("Init".to_string()),
            next: Some("Next".to_string()),
            ..Default::default()
        };
        let mut checker = ModelChecker::new(&module, &config);
        let positions: Vec<(i64, i64)> = (0..3).flat_map(|x| (0..3).map(move |y| (x, y))).collect();
        let geo = crate::state::SlideGeometry::new(positions.clone()).unwrap();
        let init_masks: Vec<u64> = [
            geo.cells_to_mask(&[(0, 0), (0, 1)]).unwrap(),
            geo.cells_to_mask(&[(1, 0)]).unwrap(),
            geo.cells_to_mask(&[(2, 0)]).unwrap(),
        ]
        .to_vec();
        let init_board = geo.masks_to_board(&init_masks);
        checker.nested_set_slide_arm = Some(
            crate::state::SlideKernelArm::try_arm_recognized(0, positions, &[&init_board])
                .expect("wrong-arm fixture must construct"),
        );
        checker.nested_set_slide_tripwire = super::super::run_bfs_full::SLIDE_TRIPWIRE_STATES;

        let result = checker.check();
        let stats = match &result {
            crate::CheckResult::Success(stats) => stats,
            other => panic!("expected closure success, got {other:?}"),
        };
        assert!(
            !checker.nested_set_slide_armed_for_testing(),
            "tripwire must have DISARMED the wrong kernel"
        );
        assert_eq!(
            stats.states_found, interp_states,
            "post-disarm run must equal the interpreter closure exactly"
        );
    }

    #[test]
    fn try_arm_recognized_rejects_overlapping_init_pieces() {
        let geo = crate::state::SlideGeometry::rectangular(4, 5).unwrap();
        let a = geo.cells_to_mask(&[(0, 0), (0, 1)]).unwrap();
        let b = geo.cells_to_mask(&[(0, 1), (0, 2)]).unwrap(); // overlaps `a`
        let board = geo.masks_to_board(&[a, b]);
        let positions: Vec<(i64, i64)> = (0..4).flat_map(|x| (0..5).map(move |y| (x, y))).collect();
        assert!(
            crate::state::SlideKernelArm::try_arm_recognized(0, positions, &[&board]).is_none(),
            "overlapping pieces violate the disjointness precondition"
        );
    }
}
