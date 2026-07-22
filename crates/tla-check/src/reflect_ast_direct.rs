// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! `reflect-check --ast-direct`: the RECOGNIZER-FREE reflected discharge — increment 1 of the
//! design pivot (`docs/cert/design-pivot-reflect.md`).
//!
//! The existing reflect-check `--full` path discharges all three safety obligations through the
//! kernel-defined deep evaluator, but the predicates it quotes are the RECOGNIZED `PredIR`s —
//! produced by `cleancic.rs`'s 18K-line recognizer — so its honest trust base is "kernel +
//! recognizer + quoter". This module quotes `Init`/`Next`/`Safety` DIRECTLY from the spec's own
//! TLA+ AST (re-parsed from the cert's embedded `spec_src`, operator-inlined by the SAME
//! deterministic `cert_inline` pass every lane uses) into the kernel-admitted deep embedding
//! ([`crate::reflect`]'s `TyReflectPPred`), retiring the recognizer from this verdict's trust
//! base entirely.
//!
//! ## What an auditor must read for this verdict (the honest trust base)
//!
//! * the **clean kernel** (type checker + checked `add_inductive`/`add_decl` admission — it
//!   derives and checks the recursor the evaluator is defined by, and it REDUCES every
//!   obligation; no Rust computes a verdict);
//! * the **AST-direct quoter** below ([`quote_pred_ast`]/[`quote_val_ast`]) — deliberately
//!   syntax-directed: one match arm per admitted TLA+ AST constructor, each arm EXACTLY one
//!   deep-constructor application ([`crate::reflect::deep`]), fail-closed (`None`) on every
//!   node outside the fragment;
//! * the **parser + operator inliner** (`tla_core::parse_to_syntax_tree`/`lower` and
//!   [`crate::cert_inline::CertInlineEnv`]) — the quoter consumes THEIR output, so they are in
//!   the trust base. Claiming "kernel + quoter" without them would OVERCLAIM, and they are
//!   COMMON-MODE with the recognized-IR lane (its spec-bind parses + inlines identically), so
//!   the cross-check below does NOT discharge them;
//! * the **AST-level domain-bound rule** ([`ast_init_col_bound`]/[`ast_next_col_bound`]) — the
//!   completeness legs prove `Init⊆R` and closure RELATIVE to a product domain `D`; that
//!   `D ⊇ {s:Init(s)}` and `D ⊇ Succ(R)` rests on the trusted-Rust rule documented at those
//!   functions (the same residual class as the recognized lane's `RustDerived` axes — surfaced,
//!   never hidden);
//! * (increment 3) for a spec with an ENUM / model-value column, the cert's stored
//!   `sorts[i].labels` (the per-column label→code map): the quoter resolves a label atom
//!   `d1`/`"idle"` to the code the enumerator packed it as. This is enumerator-provided data, so
//!   admitting enum columns adds it to the trust base — but it is COMMON-MODE with the recognized
//!   lane (which reads the SAME `sorts`) and with the enumerated `R` (whose codes use the SAME
//!   map), and `prepare_ast_direct` fail-closes on a non-injective (duplicate/empty) label set,
//!   so the map is a BIJECTION and the code-space argument transfers to the label-space one. The
//!   verdict label states this: "kernel + quoter (+ the cert's enum label map for enum columns)".
//!
//! What is NOT trusted: the RECOGNIZER (`cleancic.rs` recognition arms — not consulted at all),
//! the shallow embedder (`embed_pred_ir` — not consulted), and the ENUMERATOR: `R` is
//! enumerator-provided, but the three kernel legs verify it is a sound inductive invariant
//! (`Init⊆R ∧ R closed ∧ R⊆Safety` ⇒ every reachable state is safe — ANY sound inductive
//! invariant proves safety, so R need not equal the enumerated reachable set).
//!
//! ## The fragment (HourClock through multi-var Int + enum/model-value; fail-closed beyond it)
//!
//! Predicates: `∧ ∨ ¬ ⇒ ⇔`, comparisons `= ≠ < ≤ > ≥` over value terms, interval membership
//! `x ∈ lo..hi`, the pinned conditional update `l = IF c THEN t ELSE f`, and (increment 2)
//! `UNCHANGED x` / `UNCHANGED <<x,y,…>>` over named state variables. Value terms: nonneg integer
//! literals, state variables, primed state variables (Next only), and `+`. Increment 2 also
//! admits multi-variable Int specs and a DISJUNCTIVE `Next` (top-level `A ∨ B ∨ …` of
//! And-conjunctions): the quoter's `∨` arm already composes disjuncts, and the completeness
//! domain rule bounds each column by the MAX over disjuncts of its per-disjunct pin, declining
//! (fail-closed) unless EVERY disjunct pins EVERY column (see the domain-rule block below).
//!
//! Increment 3 adds two atoms, with ZERO new kernel constructors:
//!   * **finite-set membership over a LITERAL set** `x ∈ {v1,…,vn}` — desugared to the left-nested
//!     `Or`-fold of equalities `(x=v1) ∨ … ∨ (x=vn)` (the Aristotle-proved `mem_iff_any_eq`,
//!     `R1_set_membership_or_fold.lean`), reusing the existing `Or`/`Eq` ctors. `x` must be a
//!     (possibly primed) COLUMN and every element an Int literal / enum label for that column's
//!     sort; a NON-literal domain (a variable, a set-builder) and an EMPTY set decline;
//!   * **enum / model-value equality** `x = "label"` / `x = modelValue` (and `≠`, and
//!     membership over an enum column) — quoted to `Eq(⟦x⟧, lit code)` where `code` is the label's
//!     index in the column's `sorts[i].labels`. KIND-guarded and column-LOCAL: an unknown label, a
//!     cross-column label, or a cross-kind atom DECLINES (never a wrong code). The enum column's
//!     completeness axis is `[0, max resolved code]` (`≤ n_labels−1`, all codes ≥ 0 by
//!     construction), derived from the pinning membership/equality exactly as the Int rules.
//!
//! Increment 5 admits ONE narrow, PROVED-Nat-safe use of subtraction (see [`quote_val_ast`]'s
//! `Sub` arm): a NONNEG-LITERAL minuend minus an UNPRIMED Int COLUMN whose reachable-column max
//! (`max_r[i]`) is `≤` that literal, and ONLY in an ACTION (`Next`) context. On every state where
//! such a term is evaluated (the closure leg reads the current state from `R`, whose column max is
//! `max_r[i]`) the subtrahend `x ≤ c`, so by the Aristotle-proved `bounded_sub_coincide`
//! (`x ≤ c ⇒ (c:ℤ)-(x:ℤ) = ((c ∸ x : ℕ):ℤ) ≥ 0`; job a0a42e6b, `RequestProject/BoundedSub.lean`)
//! the kernel's truncating `Nat.sub` COINCIDES with TLA+ integer `c - x` — no truncation, no
//! sign loss. This is the ONE constructor that would be UNSOUND admitted generally (a column
//! whose bound exceeds `c` could truncate `c - x` to 0 where TLA+ goes negative — a FALSE SAFE),
//! so it is fail-closed on every axis (non-literal minuend, `x - y` between columns, a primed or
//! compound subtrahend, a non-Int column, or a column bound `> c`).
//!
//! DELIBERATE EXCLUSIONS (soundness, not laziness): general `-` (a non-literal minuend, an `x - y`
//! between columns, or a subtrahend not proved `≤` the minuend), `÷`/`%`/unary minus are NOT
//! quoted even though the deep embedding has `Nat.sub`/`Nat.div`/`Nat.mod` constructors — over
//! TLA+ INTEGERS `3 - 5 = -2`, but the deep evaluator computes truncated-Nat `3 ∸ 5 = 0`, so
//! quoting them would silently change the predicate's meaning. Within the admitted fragment
//! (nonneg literals, Nat-valued state cells, `+`, PROVED-bounded `c - x`, branch selection, enum
//! codes) TLA+ integer/atom semantics and the kernel's Nat semantics COINCIDE, which is the
//! adequacy argument for the whole lane. Anything else — `CHOOSE`, sequences, records, functions,
//! quantifiers, sets beyond `lo..hi`/a literal enum, a non-literal set domain — declines (`None` ⇒
//! INCONCLUSIVE), never a wrong quote.
//!
//! Increment 6 adds NO new quoter constructor: a column-to-column equality `ack = rdy` in Init
//! ALREADY quotes ([`quote_eq_ast`]'s Int `(None, None)` arm → `eq(var i, var j)`). What it adds is
//! a TWO-PASS Init DOMAIN rule ([`ast_init_col_eq_pin`]): a column NOT literally pinned inherits a
//! ONE-HOP bound from a LITERALLY-pinned column it is equated to — `s_j ≤ H_j ∧ s_i = s_j ⇒
//! s_i ≤ H_j` (Aristotle `init_domain_pin_transfer`, job 8ed16b94, sorry-free). ONE HOP ONLY (no
//! chains/cycles — pass 2 reads only pass-1 literal bounds), FAIL-CLOSED on an equality to a
//! non-literally-pinned column, a cycle, or conflicting pins. This flips the corpus spec
//! AsynchInterface (`Init == val ∈ Data ∧ rdy ∈ {0,1} ∧ ack = rdy`) — `ack` is pinned to `rdy`'s
//! bound — the 2nd corpus spec on the recognizer-free route.
//!
//! ## Fail-closed discipline (SOUNDNESS-CRITICAL)
//!
//! [`AstDirectVerdict::NotSafe`]/[`AstDirectVerdict::NotClosed`]/
//! [`AstDirectVerdict::NotInitComplete`] are DEFINITIVE kernel verdicts (the kernel reduced the
//! obligation to `Bool.false`) and are never collapsed into a certify.
//! [`AstDirectVerdict::Inconclusive`] is a decline, not a verdict. Where the recognized-IR lane
//! is also conclusive, [`reflect_check_ast_direct_with_crosscheck`] REQUIRES class agreement: a
//! conclusive contradiction is [`AstCrossCheck::Divergence`] — a hard, loud error meaning a
//! quoter/recognizer/embedding bug; callers must trust NEITHER verdict.

use std::sync::Arc;

use clean_kernel::Expr;
use num_traits::ToPrimitive;
use tla_core::ast::{Expr as AstExpr, OperatorDef, Unit};
use tla_core::Spanned;

use crate::cert::SafetyCertificate;
use crate::explicit_fixpoint_cert::{ColSort, EnumKind};
use crate::reflect::{deep, kernel_eval_quoted_implies_mem, kernel_eval_quoted_pred};

// ===========================================================================
// THE AST-DIRECT QUOTER — the audited translation surface of this lane.
//
// Syntax-directed and 1:1: one match arm per admitted TLA+ AST constructor,
// each arm EXACTLY one deep-constructor application over recursively-quoted
// children. NO idiom recognition, NO arithmetic, NO op selection beyond the
// fixed arm↦constructor map. Fail-closed: EVERY node outside the fragment
// (including `-`/`÷`/`%` — see the module docs' Nat-truncation note) returns
// `None`, and any `None` poisons the whole quote.
//
// Column indices: an identifier is a state-column reference iff its name is in
// `vars` (the spec's VARIABLE declaration order — the SAME tuple-column order
// the certifier and every cert lane use); the quoted index is its position.
// Everything else an `Ident` could mean (CONSTANT, operator) was already
// resolved away by `cert_inline` — an unresolved one declines. A pre-resolved
// `StateVar` node is mapped BY NAME (its embedded index is never trusted).
// ===========================================================================

/// Column index of a state-variable name, by position in the declaration-order list.
fn col_of(name: &str, vars: &[&str]) -> Option<u64> {
    vars.iter().position(|v| *v == name).map(|i| i as u64)
}

/// The state-variable name under a `Prime`/value leaf: `Ident` or pre-resolved `StateVar`
/// (matched BY NAME; the `StateVar` index is deliberately ignored).
fn var_name(e: &AstExpr) -> Option<&str> {
    match e {
        AstExpr::Ident(name, _) => Some(name.as_str()),
        AstExpr::StateVar(name, _, _) => Some(name.as_str()),
        _ => None,
    }
}

/// The state columns an `UNCHANGED` operand freezes: a single state variable, or a
/// `<<v1, …, vk>>` tuple of state variables. Fail-closed (`None`) on any non-variable /
/// non-tuple-of-variables operand — and on an EMPTY tuple (degenerate) — so the quoter and the
/// domain rule both decline anything outside the `UNCHANGED <named vars>` shape. Shared by both
/// so the two can never disagree about which columns an `UNCHANGED` pins. Duplicate names
/// collapse harmlessly (an idempotent `x'=x`); `col_of` rejects any element that is not a
/// declared variable.
fn unchanged_cols(operand: &AstExpr, vars: &[&str]) -> Option<Vec<u64>> {
    match operand {
        AstExpr::Ident(..) | AstExpr::StateVar(..) => Some(vec![col_of(var_name(operand)?, vars)?]),
        AstExpr::Tuple(items) if !items.is_empty() => items
            .iter()
            .map(|it| col_of(var_name(&it.node)?, vars))
            .collect(),
        _ => None,
    }
}

// ── Increment 3: enum / model-value atoms + finite-set membership ───────────
//
// An enum / model-value column ([`ColSort::Enum`]) stores each cell as the label
// CODE — the position of the label in the column's sorted, DISTINCT `labels`
// (the cert's `sorts` carries them). The deep evaluator reads every column as a
// `Nat` ([`crate::reflect::quote_state`] quotes the tuple as `List Nat`), so the
// stored code IS the Nat the kernel compares: `mode = d1` quotes to
// `deep::eq(var i, lit code(d1))`, reducing to `Nat.beq (nth s i) code` — EXACTLY
// enum equality, reusing ZERO new kernel constructors (the same `eq`/`var`/`lit`
// this lane already exercises). Faithful ⟺ the label→code map is a BIJECTION,
// which `prepare_ast_direct` enforces (distinct, non-empty labels; fail-closed on
// a forged cert). SOUNDNESS: resolution is KIND-GUARDED and column-LOCAL — a model
// value resolves ONLY against a `Model` column, a `String` ONLY against a `Str`
// column, and ONLY against the labels of the column it is compared to; an unknown
// label, a cross-column label, or a cross-kind atom declines (never a wrong code).

/// A (possibly primed) state COLUMN reference: `(index, quoted-column-term)`. `Ident`/`StateVar`
/// ⇒ `var i`; `Prime(var)` ⇒ `prime i` (Next only). `None` on anything else — the SAME
/// var/prime gate as [`quote_val_ast`], factored so the enum/membership arms resolve a column
/// index and its sort identically. A primed leaf in a state predicate (`allow_prime=false`)
/// declines here too.
fn col_ref_ast(e: &AstExpr, vars: &[&str], allow_prime: bool) -> Option<(u64, Expr)> {
    match e {
        AstExpr::Ident(..) | AstExpr::StateVar(..) => {
            let i = col_of(var_name(e)?, vars)?;
            Some((i, deep::var(i)))
        }
        AstExpr::Prime(inner) if allow_prime => {
            let i = col_of(var_name(&inner.node)?, vars)?;
            Some((i, deep::prime(i)))
        }
        _ => None,
    }
}

/// If `e` is a (possibly primed) column whose sort is an ENUM, return its
/// `(index, quoted-column-term, labels, kind)`. `None` for a non-column, an out-of-range index,
/// or a non-enum column. The `labels` slice is borrowed from the cert's `sorts`.
fn enum_col_ref<'s>(
    e: &AstExpr,
    vars: &[&str],
    sorts: &'s [ColSort],
    allow_prime: bool,
) -> Option<(u64, Expr, &'s [String], EnumKind)> {
    let (i, col_expr) = col_ref_ast(e, vars, allow_prime)?;
    match sorts.get(i as usize)? {
        ColSort::Enum { labels, kind } => Some((i, col_expr, labels.as_slice(), *kind)),
        _ => None,
    }
}

/// True if EITHER operand of an order comparison is an enum / model-value column. Such a
/// comparison is a cross-kind category error (a model value is not an ordered integer), so the
/// caller DECLINES (fail-closed) — the enum code must never be treated as an ordered integer.
fn cmp_has_enum_operand(
    a: &Spanned<AstExpr>,
    b: &Spanned<AstExpr>,
    vars: &[&str],
    sorts: &[ColSort],
    allow_prime: bool,
) -> bool {
    enum_col_ref(&a.node, vars, sorts, allow_prime).is_some()
        || enum_col_ref(&b.node, vars, sorts, allow_prime).is_some()
}

/// The label CODE of an enum atom `elem` against a column's `labels`/`kind`: the atom's position
/// in `labels`. KIND-GUARDED and column-LOCAL (mirrors the recognizer's `seq_atom_elem_code`): a
/// `String` literal resolves ONLY against a `Str` column, a model-value `Ident` ONLY against a
/// `Model` column (and NEVER a state-variable ident — a column is not a model atom). `None`
/// (fail-closed) on a kind mismatch, a state variable, or a label NOT in this column's set — so
/// an UNKNOWN label (`x = "notALabel"`) and a WRONG-COLUMN label (this column's sort lacks it)
/// both DECLINE. Never a guessed code.
fn resolve_enum_code(
    elem: &AstExpr,
    labels: &[String],
    kind: EnumKind,
    vars: &[&str],
) -> Option<u64> {
    let name: &str = match (kind, elem) {
        (EnumKind::Str, AstExpr::String(s)) => s.as_str(),
        (EnumKind::Model, AstExpr::Ident(n, _)) => {
            if vars.iter().any(|v| *v == n.as_str()) {
                return None; // a state column is not a model-value atom
            }
            n.as_str()
        }
        // Kind mismatch (`String` vs a `Model` column, or vice versa), an `Int`-kind func-domain
        // marker used as a scalar cell, or a non-atom element ⇒ fail-closed.
        _ => return None,
    };
    labels.iter().position(|l| l == name).map(|p| p as u64)
}

/// The CODE a set-membership / equality element denotes against a COLUMN's sort: an `Int` column
/// takes a nonneg `Int` LITERAL (`code = value`); an `Enum` column takes a resolvable label
/// (`code = idx`). `None` (fail-closed) on any other sort, a negative/oversized Int literal, or
/// an unresolvable label — so `x ∈ {a, notALabel}` and `x ∈ {-1, 0}` both decline.
fn resolve_member_code(elem: &AstExpr, sort: &ColSort, vars: &[&str]) -> Option<u64> {
    match sort {
        ColSort::Int => match elem {
            AstExpr::Int(n) => n.to_u64(),
            _ => None,
        },
        ColSort::Enum { labels, kind } => resolve_enum_code(elem, labels, *kind, vars),
        _ => None,
    }
}

/// Quote an equality/inequality `a ⊕ b` (⊕ ∈ {`=`,`≠`}), enum-AWARE. Three sound shapes:
///   · ENUM column `⊕` label      → `⊕(⟦col⟧, lit code)` (the label resolved against THIS column);
///   · ENUM column `⊕` ENUM column → `⊕(⟦col_a⟧, ⟦col_b⟧)` ONLY when the two share the SAME sort
///       (identical `labels` + `kind`, hence the SAME code map — else the codes are incomparable);
///   · neither side an enum column → the Int-fragment `⊕(⟦a⟧, ⟦b⟧)` (unchanged from v1/2).
/// FAIL-CLOSED: an enum column compared to a NON-label (an Int literal, a cross-sort label, an
/// arithmetic term) declines — it never silently falls through to the Int path (which would
/// compare an enum code to an integer, a category error that can be a WRONG verdict).
fn quote_eq_ast(
    a: &AstExpr,
    b: &AstExpr,
    vars: &[&str],
    sorts: &[ColSort],
    max_r: &[u64],
    allow_prime: bool,
    is_neq: bool,
) -> Option<Expr> {
    let mk = |x: Expr, y: Expr| {
        if is_neq {
            deep::neq(x, y)
        } else {
            deep::eq(x, y)
        }
    };
    let ea = enum_col_ref(a, vars, sorts, allow_prime);
    let eb = enum_col_ref(b, vars, sorts, allow_prime);
    Some(match (ea, eb) {
        (Some((_, col_a, labels, kind)), None) => {
            mk(col_a, deep::lit(resolve_enum_code(b, labels, kind, vars)?))
        }
        (None, Some((_, col_b, labels, kind))) => {
            mk(deep::lit(resolve_enum_code(a, labels, kind, vars)?), col_b)
        }
        (Some((_, col_a, la, ka)), Some((_, col_b, lb, kb))) => {
            // Two enum columns compare index-exactly ONLY when their code maps coincide.
            if la == lb && ka == kb {
                mk(col_a, col_b)
            } else {
                return None;
            }
        }
        (None, None) => mk(
            quote_val_ast(a, vars, sorts, max_r, allow_prime)?,
            quote_val_ast(b, vars, sorts, max_r, allow_prime)?,
        ),
    })
}

/// Quote finite-set membership `x ∈ {v1,…,vn}` (a LITERAL set) as the left-nested `Or`-fold of
/// equalities `(x=v1) ∨ (x=v2) ∨ … ∨ (x=vn)` — the Aristotle-proved desugar (`mem_iff_any_eq`,
/// `R1_set_membership_or_fold.lean`), reusing ONLY the existing `Or`/`Eq`/`var`/`prime`/`lit`
/// ctors (ZERO new kernel constructors). `x` must be a (possibly primed) COLUMN; each element is
/// resolved against THAT column's sort ([`resolve_member_code`]). FAIL-CLOSED (`None`) on: an
/// empty set (`x ∈ {}` is always false — declining is sound; the embedding has no false-pred
/// ctor), a non-column `x`, or ANY element outside the column's value fragment (an unknown label,
/// a negative Int, a nested set). A non-literal domain (a variable, a set-builder) never reaches
/// here — the `In` arm only routes a `SetEnum` node in.
fn quote_setenum_mem_ast(
    x: &AstExpr,
    elems: &[Spanned<AstExpr>],
    vars: &[&str],
    sorts: &[ColSort],
    allow_prime: bool,
) -> Option<Expr> {
    if elems.is_empty() {
        return None; // `x ∈ {}` ≡ FALSE; decline (fail-closed) rather than fabricate a pred
    }
    let (i, col_expr) = col_ref_ast(x, vars, allow_prime)?;
    let sort = sorts.get(i as usize)?;
    let mut acc: Option<Expr> = None;
    for e in elems {
        let code = resolve_member_code(&e.node, sort, vars)?;
        let eq = deep::eq(col_expr.clone(), deep::lit(code));
        acc = Some(match acc {
            None => eq,
            Some(prev) => deep::or(prev, eq),
        });
    }
    acc // non-empty ⇒ Some
}

/// Quote a TLA+ VALUE expression into a deep `TyReflectPExp` term. `allow_prime` is `false`
/// for STATE predicates (`Init`/`Safety`) — a primed leaf then declines, which is the lane's
/// state-predicate gate. `sorts`/`max_r` are consumed ONLY by the bounded-subtraction arm
/// ([increment 5]) to prove the subtrahend cannot truncate. Fail-closed (`None`) outside the
/// fragment.
pub(crate) fn quote_val_ast(
    e: &AstExpr,
    vars: &[&str],
    sorts: &[ColSort],
    max_r: &[u64],
    allow_prime: bool,
) -> Option<Expr> {
    Some(match e {
        // Nonneg integer literal. A NEGATIVE (or > u64) literal declines: the deep evaluator
        // computes over Nat, and only nonneg literals keep TLA+ Int semantics coincident.
        AstExpr::Int(n) => deep::lit(n.to_u64()?),
        // State-variable reference (current state). A name that is NOT a declared variable
        // declines — `cert_inline` already substituted constants/operators, so an unresolved
        // identifier is out of fragment. SOUNDNESS (root-cause cross-kind guard): an ENUM /
        // model-value column reaching a VALUE position (`+`, an order-comparison operand, an `IF`
        // branch, a range bound `lo`/`hi`) is a category error — its Nat CODE would be treated as
        // an ordered integer (`mode + 0 < 2`, `y ∈ 0..mode`). DECLINE at the value leaf so EVERY
        // nested position is closed uniformly (the per-arm `Lt`/`Leq`/`Gt`/`Geq`/`In(Range)` enum
        // guards become redundant defense-in-depth). Legit enum VALUE uses — `col = label`,
        // `col = col` (same-sort copy), `col ∈ {labels}` — never reach here: `quote_eq_ast` and
        // `quote_setenum_mem_ast` resolve enum operands themselves via `enum_col_ref` BEFORE any
        // `quote_val_ast` fallthrough.
        AstExpr::Ident(..) | AstExpr::StateVar(..) => {
            let i = col_of(var_name(e)?, vars)?;
            if matches!(sorts.get(i as usize)?, ColSort::Enum { .. }) {
                return None;
            }
            deep::var(i)
        }
        // Primed state variable (next state) — Next only. Same enum-in-value-position guard.
        AstExpr::Prime(inner) if allow_prime => {
            let i = col_of(var_name(&inner.node)?, vars)?;
            if matches!(sorts.get(i as usize)?, ColSort::Enum { .. }) {
                return None;
            }
            deep::prime(i)
        }
        AstExpr::Add(a, b) => deep::add(
            quote_val_ast(&a.node, vars, sorts, max_r, allow_prime)?,
            quote_val_ast(&b.node, vars, sorts, max_r, allow_prime)?,
        ),
        // ── Increment 5: Nat-SAFE BOUNDED SUBTRACTION `c - x` ──────────────────────────────
        // The deliberate exclusion of `-` is RELAXED for exactly one PROVED-safe shape: a
        // NONNEG-LITERAL minuend `c` minus an UNPRIMED Int COLUMN `x` whose reachable-column
        // maximum (`max_r[i]`) is `≤ c`, and ONLY in an ACTION context (`allow_prime` — i.e.
        // `Next`). Quoted to the truncating `Nat.sub` ctor (`deep::sub(lit c, var i)`).
        //
        // SOUNDNESS (Aristotle `bounded_sub_coincide`, job a0a42e6b): the kernel evaluates
        // `sub` as `Nat.sub`, which over Nat TRUNCATES (`3 ∸ 5 = 0`) whereas TLA+ integer
        // `3 - 5 = -2`. They COINCIDE iff the subtrahend `x ≤ c` (then `(c:ℤ)-(x:ℤ) =
        // ((c ∸ x):ℤ) ≥ 0`). We admit ONLY when `x ≤ c` is PROVED for every state at which the
        // term is evaluated:
        //   · `allow_prime` restricts this term to `Next`; `Next` is kernel-evaluated ONLY in
        //     the closure leg, whose CURRENT state `s` ranges over the stored `R`. An unprimed
        //     column `x = var i` reads `nth s i = s_i` with `s ∈ R`, so `x ≤ max_r[i]` — a TRUE
        //     upper bound over EVERY evaluated state (not just an observed sample: it is the max
        //     over the very set `s` iterates). With `max_r[i] ≤ c`, `x ≤ c` on every leg-3 state.
        //   · A PRIMED subtrahend would read `sp ∈ D_next` (a Rust-bounded product domain, not
        //     `R`) — a different, larger bound — so it is DECLINED (fail-closed); only an
        //     UNPRIMED column subtrahend is admitted.
        //   · In a STATE predicate (`allow_prime = false`, `Init`/`Safety`) the current state is
        //     `R` (safety leg) or `D_init` (init-completeness leg, bound by `init_bounds`, not
        //     `max_r`) — a DIFFERENT bound — so subtraction is DECLINED there entirely.
        // DECLINE: a non-literal minuend, `x - y` (both columns), a primed/compound subtrahend,
        // a non-Int column subtrahend, or a column whose `max_r[i] > c`. Any of these could let
        // `Nat.sub` truncate where TLA+ goes negative — a FALSE SAFE — so all fail closed.
        AstExpr::Sub(a, b) if allow_prime => {
            let c = match &a.node {
                AstExpr::Int(n) => n.to_u64()?,
                _ => return None, // non-literal minuend: no proved bound on `c - x`
            };
            // Subtrahend: an UNPRIMED column (a primed/compound/literal subtrahend declines).
            let i = match &b.node {
                AstExpr::Ident(..) | AstExpr::StateVar(..) => col_of(var_name(&b.node)?, vars)?,
                _ => return None,
            };
            // The column must be an Int (a subtraction over an enum CODE is a category error),
            // and its reachable maximum must be `≤ c` (so `x ≤ c`, no truncation).
            if !matches!(sorts.get(i as usize)?, ColSort::Int) {
                return None;
            }
            if *max_r.get(i as usize)? > c {
                return None;
            }
            deep::sub(deep::lit(c), deep::var(i))
        }
        // EVERYTHING else declines — deliberately including general `Sub` (a non-literal
        // minuend, an `x - y`, or an unproved bound), `Div`/`IntDiv`/`Mod`/`Neg`
        // (truncated-Nat infidelity; see module docs) and value-position `If` outside the
        // `l = IF …` shape (the embedding has no free-standing value conditional yet).
        _ => return None,
    })
}

/// Quote a TLA+ PREDICATE expression into a deep `TyReflectPPred` term. `allow_prime` as in
/// [`quote_val_ast`]. `sorts` is the cert's per-column [`ColSort`] list (used to resolve
/// enum/model-value label atoms to their codes, and — increment 5 — to gate bounded subtraction
/// to Int columns). `max_r` is the per-column reachable maximum, consumed ONLY by the
/// bounded-subtraction arm of [`quote_val_ast`]. Fail-closed (`None`) outside the fragment.
pub(crate) fn quote_pred_ast(
    e: &AstExpr,
    vars: &[&str],
    sorts: &[ColSort],
    max_r: &[u64],
    allow_prime: bool,
) -> Option<Expr> {
    let qp = |x: &Spanned<AstExpr>| quote_pred_ast(&x.node, vars, sorts, max_r, allow_prime);
    let qv = |x: &Spanned<AstExpr>| quote_val_ast(&x.node, vars, sorts, max_r, allow_prime);
    Some(match e {
        AstExpr::And(a, b) => deep::and(qp(a)?, qp(b)?),
        AstExpr::Or(a, b) => deep::or(qp(a)?, qp(b)?),
        AstExpr::Not(a) => deep::not(qp(a)?),
        AstExpr::Implies(a, b) => deep::implies(qp(a)?, qp(b)?),
        AstExpr::Equiv(a, b) => deep::equiv(qp(a)?, qp(b)?),
        // `l = IF c THEN t ELSE f` — the ONE composite pattern in this quoter (two AST
        // constructors deep), honestly noted: the v1 embedding has no free-standing value
        // conditional (that needs a mutual PExp/PPred inductive — the next fragment step), so
        // the conditional is admitted exactly in its Eq-pinned form via the kernel-admitted
        // `eqIte` constructor. Still purely syntactic — no idiom recognition. (Int-branch only:
        // an enum-column `IF` rhs declines via `qv` on its label branches — fail-closed.)
        AstExpr::Eq(a, b) => match &b.node {
            // The `eq_ite` composite is an Int-VALUE-pinned conditional. If the lhs is an ENUM
            // column, `l = IF c THEN t ELSE f` would compare a label CODE to an Int conditional (a
            // category error — `modelValue = 0` is FALSE in TLA, but the code comparison could
            // reduce to true) ⇒ DECLINE (fail-closed). An enum-branch `IF` already declines via
            // `qv` on its non-value label branches; this guards the Int-branch case.
            AstExpr::If(c, t, f) => {
                if enum_col_ref(&a.node, vars, sorts, allow_prime).is_some() {
                    return None;
                }
                deep::eq_ite(qv(a)?, qp(c)?, qv(t)?, qv(f)?)
            }
            // `=` — enum-aware (increment 3): `col = label` / same-sort `col = col` / Int `=`.
            _ => quote_eq_ast(&a.node, &b.node, vars, sorts, max_r, allow_prime, false)?,
        },
        // `≠` — enum-aware (increment 3), same shapes as `=`.
        AstExpr::Neq(a, b) => {
            quote_eq_ast(&a.node, &b.node, vars, sorts, max_r, allow_prime, true)?
        }
        // ORDER comparisons `< ≤ > ≥` are INTEGER ops. An ENUM / model-value column quotes to its
        // Nat CODE (`deep::var i`), so `mode < 2` would compare a label code as an integer — a
        // CATEGORY ERROR (`ty check` rejects "first argument of < should be an integer"; a model
        // value is never `< n` in TLA). DECLINE (fail-closed) if EITHER operand is an enum column,
        // so the code is never treated as an ordered integer. (The `=`/`≠` arms are already
        // enum-aware via `quote_eq_ast`; this closes the same cross-kind gap for the order arms.)
        AstExpr::Lt(a, b) if cmp_has_enum_operand(a, b, vars, sorts, allow_prime) => return None,
        AstExpr::Leq(a, b) if cmp_has_enum_operand(a, b, vars, sorts, allow_prime) => return None,
        AstExpr::Gt(a, b) if cmp_has_enum_operand(a, b, vars, sorts, allow_prime) => return None,
        AstExpr::Geq(a, b) if cmp_has_enum_operand(a, b, vars, sorts, allow_prime) => return None,
        AstExpr::Lt(a, b) => deep::lt(qv(a)?, qv(b)?),
        AstExpr::Leq(a, b) => deep::leq(qv(a)?, qv(b)?),
        AstExpr::Gt(a, b) => deep::gt(qv(a)?, qv(b)?),
        AstExpr::Geq(a, b) => deep::geq(qv(a)?, qv(b)?),
        // Membership `x ∈ dom`:
        //   · `x ∈ lo..hi`   (an INTERVAL) — quoted 1:1 to the `inRange` constructor (v1);
        //   · `x ∈ {v1,…,vn}` (a LITERAL SET) — desugared (increment 3) to the `Or`-fold of
        //       equalities `(x=v1) ∨ … ∨ (x=vn)` via [`quote_setenum_mem_ast`] (ZERO new kernel
        //       ctors; the Aristotle-proved `mem_iff_any_eq`), each element resolved against `x`'s
        //       column sort (Int literal / enum label). A NON-literal domain (a variable, a
        //       set-builder, a set expression) is not in the embedding ⇒ declines.
        AstExpr::In(x, dom) => match &dom.node {
            // `x ∈ lo..hi` is an INTEGER interval. An enum column in an Int interval (`mode ∈ 0..1`)
            // is a category error (a model value is never in an integer range; `ty check` reports
            // the invariant VIOLATED) — quoting it via `inRange` on the label CODE would falsely
            // certify. DECLINE if `x` is an enum column. (A literal ENUM set `x ∈ {l1,l2}` goes
            // through the enum-aware `SetEnum` arm below, which resolves each element to a label
            // code and is sound.)
            AstExpr::Range(_, _) if enum_col_ref(&x.node, vars, sorts, allow_prime).is_some() => {
                return None
            }
            AstExpr::Range(lo, hi) => deep::in_range(qv(x)?, qv(lo)?, qv(hi)?),
            AstExpr::SetEnum(elems) => {
                quote_setenum_mem_ast(&x.node, elems, vars, sorts, allow_prime)?
            }
            _ => return None,
        },
        // `UNCHANGED x` ≡ `x' = x`; `UNCHANGED <<x,y,…>>` ≡ `x'=x ∧ y'=y ∧ …` (a next-state
        // ACTION leaf, so only when primes are allowed — a state predicate declines). Quoted to
        // the SAME deep term as that primed-equality conjunction: `deep::eq(prime i, var i)` per
        // frozen column, And-folded left-nested. This reuses ONLY the `eq`/`prime`/`var` ctors
        // this lane already exercises and tests (rather than wiring the embedding's separate
        // `unchanged` ctor into this lane's audited surface), and it is kernel-FAITHFUL because
        // the two reduce to the IDENTICAL normal form: `eq(prime i, var i)` reduces via the
        // `eq`/`EvalV` arms to `Nat.beq (EvalV (prime i) s sp) (EvalV (var i) s sp)` =
        // `Nat.beq (nth sp i) (nth s i)`, which is byte-for-byte the `unchanged i` evaluator arm
        // (`Nat.beq (nth sp i) (nth s i)`). Fail-closed via `unchanged_cols` on any
        // non-variable / non-tuple-of-variables operand (or an empty tuple).
        AstExpr::Unchanged(operand) if allow_prime => {
            let cols = unchanged_cols(&operand.node, vars)?;
            let mut it = cols.into_iter();
            // `unchanged_cols` returns a non-empty vec (empty tuple already declined).
            let first = it.next()?;
            let mut acc = deep::eq(deep::prime(first), deep::var(first));
            for i in it {
                acc = deep::and(acc, deep::eq(deep::prime(i), deep::var(i)));
            }
            acc
        }
        // EVERYTHING else declines: quantifiers, CHOOSE, sets, functions, records, sequences,
        // temporal operators, CASE, LET, …
        _ => return None,
    })
}

// ===========================================================================
// AST-LEVEL DOMAIN-BOUND RULE — trusted Rust, surfaced in the verdict label.
//
// The completeness legs reduce `Init(s) ⇒ s∈R` over D_init and
// `Next(s,sp) ⇒ sp∈R` over R × D_next. Those reductions are kernel work; what
// this rule owns is the COVERAGE meta-argument that D_init ⊇ {s : Init(s)} and
// D_next ⊇ {sp : ∃s∈R. Next(s,sp)} — including that every such state is a
// NONNEG integer tuple at all (TLA+ Ints can be negative; a Nat product domain
// cannot represent them, so admitting a shape that permits negative Init
// states or successors would make the legs silently vacuous there).
//
// v1 rules (deliberately STRICTER than the recognized lane's Int rules):
//
// INIT axis i: some conjunct must PIN the column into a finite nonneg range —
//   `x_i = lit`   (either orientation; lit a nonneg literal) ⇒ H_i = lit, or
//   `x_i ∈ lo..hi` (lo, hi nonneg literals)                  ⇒ H_i = hi.
// Then Init(s) ⇒ s_i ∈ Nat ∧ s_i ≤ H_i. Bare upper bounds (`x ≤ M`) are NOT
// admitted: they leave negative values satisfiable. Extra conjuncts only
// shrink the Init set, so scanning for one pinning conjunct is sound.
//
// INIT axis i — PASS 2 (increment 6, the ONE-HOP column-equality pin,
// `ast_init_col_eq_pin`): a column i NOT literally pinned above inherits
// H_i = H_j from an Init conjunct `x_i = x_j` / `x_j = x_i` where x_j IS
// literally pinned (pass 1). Sound by `s_j ≤ H_j ∧ s_i = s_j ⇒ s_i ≤ H_j`
// (Aristotle `init_domain_pin_transfer`, job 8ed16b94, sorry-free), so the axis
// [0, H_j] covers x_i's Init values. ONE HOP ONLY: pass 2 reads ONLY pass-1
// literal bounds, so a chain `x = y ∧ y = z ∧ z ∈ {0,1}` does NOT propagate to x
// (its partner y is not LITERALLY pinned). Fail-closed on an unpinned partner, a
// cycle `x = y ∧ y = x`, or conflicting equality pins ⇒ DECLINE (never a guess).
//
// NEXT axis i (per DISJUNCT): the Next body is split into top-level disjuncts
//   (`flatten_or`, mirroring `flatten_and` — a disjunctive Next `A ∨ B` is the
//   common multi-action shape); each disjunct is flattened into conjuncts, and
//   within a disjunct column i must be PINNED by one of:
//     · Eq-pin `x_i' = rhs`, `rhs` in the PRIME-FREE value fragment (nonneg
//       literals, current-state variables, `+`, `IF c THEN t ELSE f` over such
//       branches). For s ∈ R ⊆ Nat^k, ⟦rhs⟧(s) is a nonneg integer ≤ ub(rhs)
//       where ub replaces each variable by its max over R, sums `+`, and takes
//       max over IF branches (the condition only SELECTS a branch, so it cannot
//       raise the bound and needs no constraint itself — though v2 conservatively
//       requires the whole rhs prime-free); or
//     · an `UNCHANGED x_i` conjunct (`UNCHANGED x`/`UNCHANGED <<…,x_i,…>>`),
//       which forces the successor's column i to EQUAL its current value, so its
//       bound is max_r[i] (the current-value column max over R). This is exactly
//       the `x_i' = x_i` Eq-pin bound `ast_val_ub(x_i) = max_r[i]`.
//   Multiple pins for the SAME column within ONE disjunct min-fold (the
//   successor satisfies all of them, so the tightest is a sound upper bound).
//   A disjunct's non-pinning conjuncts (guards, an `Or` sub-term, cross-column
//   constraints) only SHRINK the transitions and cannot raise a pinned column,
//   so they are ignored.
//
//   CROSS-DISJUNCT COMBINE — column i's OVERALL Next bound is the MAX over
//   disjuncts of its per-disjunct bound. A successor produced by disjunct j is
//   bounded by disjunct j's per-column bound ≤ the MAX, so D_next (the product
//   over [0, MAX_j b_{j,i}]) covers EVERY disjunct's successors. A MIN — or a MAX
//   over only the disjuncts that happen to pin — would UNDER-cover and be
//   UNSOUND. THEREFORE: if ANY disjunct fails to pin column i (no Eq-pin and no
//   `UNCHANGED x_i`), that disjunct admits an unbounded/unknown successor for
//   column i ⇒ the whole Next domain is underivable ⇒ DECLINE (`None` ⇒
//   INCONCLUSIVE). So every disjunct's every column must be pinned.
//
//   v2 scope: TOP-LEVEL `Or` of `And`-conjunctions. A conjunct that is itself an
//   `Or`, or deeper nesting that does not cleanly split, leaves its columns
//   unpinned at the conjunct level ⇒ fail-closed decline. Any unpinned column ⇒
//   `None` ⇒ INCONCLUSIVE (fail-closed), never a guessed bound.
// ===========================================================================

/// Flatten nested `And` into a conjunct list.
fn flatten_and<'a>(e: &'a Spanned<AstExpr>, out: &mut Vec<&'a Spanned<AstExpr>>) {
    if let AstExpr::And(a, b) = &e.node {
        flatten_and(a, out);
        flatten_and(b, out);
    } else {
        out.push(e);
    }
}

/// Flatten nested top-level `Or` into a disjunct list (the mirror of [`flatten_and`]). A
/// non-`Or` node is a single disjunct, so a plain conjunctive Next yields exactly `[next]` —
/// byte-identical to the pre-disjunctive single-conjunct-list path.
fn flatten_or<'a>(e: &'a Spanned<AstExpr>, out: &mut Vec<&'a Spanned<AstExpr>>) {
    if let AstExpr::Or(a, b) = &e.node {
        flatten_or(a, out);
        flatten_or(b, out);
    } else {
        out.push(e);
    }
}

/// Increment 4: rewrite `In(x, Ident(C))` → `In(x, SetEnum({m1,…,mk}))` for exactly the `.cfg`'s
/// [`crate::ConstantValue::ModelValueSet`] constants (`CONSTANT Modes = {idle, busy}`), recursing
/// through the connectives this fragment admits (`∧`/`∨`/`¬`/`IF`/`In`). `cert_inline`
/// deliberately leaves a model-value-set constant as a bare `Ident` (the recognized lane's mvsets
/// path reads it directly), so without this the quoter's `In` arm — which routes only literal
/// `Range`/`SetEnum` domains — declines. FAIL-CLOSED by construction: only the DOMAIN slot of an
/// `In` is rewritten, only for a `ModelValueSet` constant (a `Value`/`Replacement`/unknown ident
/// stays a bare `Ident` and declines downstream), the substituted elements are the cfg's own
/// model-value atoms (each still label/kind-gated per column at `quote_setenum_mem_ast` — an
/// element outside the column's labels declines, never a wrong code), and every node this
/// function does not understand passes through UNTOUCHED (a wrong rewrite cannot be introduced
/// for shapes outside the admitted fragment — they decline exactly as before).
/// Parse a `.cfg` set-literal constant string `"{d1, d2, d3}"` into its element atoms. Returns
/// `None` (fail-closed) for anything that is NOT a simple brace-delimited list of bare, non-empty
/// atoms — a scalar `"3"`, a nested `"{{…}}"`, an empty `"{}"`, or a list with an empty element.
/// Deliberately conservative: only the exact model-value-set shape is admitted, so a rewrite is
/// never introduced for a domain the quoter would otherwise decline.
fn parse_set_literal(s: &str) -> Option<Vec<String>> {
    let t = s.trim();
    let inner = t.strip_prefix('{')?.strip_suffix('}')?.trim();
    if inner.is_empty() {
        return None; // `{}` — empty set; decline (the desugar itself declines an empty set too).
    }
    let mut out = Vec::new();
    for part in inner.split(',') {
        let atom = part.trim();
        // Bare identifier atoms only — no nested braces, no whitespace-separated tokens.
        if atom.is_empty() || atom.contains(['{', '}', ' ', '\t']) {
            return None;
        }
        out.push(atom.to_string());
    }
    Some(out)
}

fn resolve_mvset_domains(e: Spanned<AstExpr>, config: &crate::Config) -> Spanned<AstExpr> {
    let Spanned { node, span } = e;
    let rw = |b: Box<Spanned<AstExpr>>| Box::new(resolve_mvset_domains(*b, config));
    let node = match node {
        AstExpr::And(a, b) => AstExpr::And(rw(a), rw(b)),
        AstExpr::Or(a, b) => AstExpr::Or(rw(a), rw(b)),
        AstExpr::Not(a) => AstExpr::Not(rw(a)),
        AstExpr::If(c, t, f) => AstExpr::If(rw(c), rw(t), rw(f)),
        AstExpr::In(x, dom) => {
            let dom = if let AstExpr::Ident(name, _) = &dom.node {
                // The cfg's element list for a model-value-set constant, from EITHER storage form:
                // `ModelValueSet([d1,d2,d3])` (the `CONSTANT C <- {…}` syntax) OR `Value("{d1, d2,
                // d3}")` (an auto-instantiated / `CONSTANT C = {…}` set literal, stored as raw cfg
                // text — how an UNASSIGNED model-value constant like AsynchInterface's `Data` under
                // `SPECIFICATION Spec` is recorded). A scalar `Value("3")` is NOT a set literal and
                // is left untouched (declines downstream). This is the same `.cfg` constant `ty
                // check` reads, so no new trust surface.
                let labels: Option<Vec<String>> = match config.constants.get(name.as_str()) {
                    Some(crate::ConstantValue::ModelValueSet(ls)) => Some(ls.clone()),
                    Some(crate::ConstantValue::Value(s)) => parse_set_literal(s),
                    _ => None,
                };
                match labels {
                    Some(labels) if !labels.is_empty() => {
                        Box::new(Spanned::dummy(AstExpr::SetEnum(
                            labels
                                .iter()
                                .map(|m| {
                                    Spanned::dummy(AstExpr::Ident(
                                        m.clone(),
                                        tla_core::name_intern::NameId::INVALID,
                                    ))
                                })
                                .collect(),
                        )))
                    }
                    _ => dom,
                }
            } else {
                dom
            };
            AstExpr::In(x, dom)
        }
        other => other,
    };
    Spanned { node, span }
}

/// Structural nonneg upper bound of a PRIME-FREE in-fragment value expression, with each
/// state variable bounded by its column maximum over `R`. `sorts` is threaded only for the
/// `If`-condition fragment gate. `None` on any out-of-fragment node, any primed leaf, or
/// arithmetic overflow (fail-closed).
fn ast_val_ub(e: &AstExpr, vars: &[&str], sorts: &[ColSort], max_r: &[u64]) -> Option<u64> {
    match e {
        AstExpr::Int(n) => n.to_u64(),
        AstExpr::Ident(..) | AstExpr::StateVar(..) => max_r
            .get(usize::try_from(col_of(var_name(e)?, vars)?).ok()?)
            .copied(),
        AstExpr::Add(a, b) => ast_val_ub(&a.node, vars, sorts, max_r)?
            .checked_add(ast_val_ub(&b.node, vars, sorts, max_r)?),
        // Increment 5: `ub(a - b) = ub(a)`. The kernel evaluates `sub` as TRUNCATING `Nat.sub`,
        // whose result is `≤` the minuend for ANY subtrahend (`Nat.sub A B ≤ A`, attained at
        // `B = 0`), so the minuend's bound covers the successor regardless of the subtrahend
        // (truncation only LOWERS it). The subtrahend is still required in-fragment so the shape
        // matches an admitted quote — but its bound does not enter the result. (Coverage is a
        // SEPARATE concern from the quoter's Nat-fidelity gate: this bound is sound whether or not
        // the subtraction was truncation-safe; an unsafe one already DECLINED at the quote, before
        // any domain is derived.)
        AstExpr::Sub(a, b) => {
            let ua = ast_val_ub(&a.node, vars, sorts, max_r)?;
            let _sub_in_fragment = ast_val_ub(&b.node, vars, sorts, max_r)?;
            Some(ua)
        }
        // The condition only SELECTS a branch; the value is one of the branches. v1 still
        // requires the condition itself in the prime-free PREDICATE fragment (conservative —
        // it keeps "the pin's rhs is prime-free and in-fragment" a single uniform statement).
        AstExpr::If(c, t, f) => {
            // Gate only — the quoted condition term is unused here (the bound is branch-max).
            let _cond_in_fragment: Expr = quote_pred_ast(&c.node, vars, sorts, max_r, false)?;
            Some(
                ast_val_ub(&t.node, vars, sorts, max_r)?
                    .max(ast_val_ub(&f.node, vars, sorts, max_r)?),
            )
        }
        _ => None,
    }
}

/// The upper bound a LITERAL-SET membership `x ∈ {v1,…,vn}` (over column `i`'s sort) imposes on
/// column `i`: `max_j code(vj)` — every satisfying value is one of the codes, so `[0, max]`
/// covers them (Int literals ARE their value; enum labels their index). `None` (⇒ this conjunct
/// does not contribute a bound) on an empty set or ANY unresolvable element, so a column pinned
/// ONLY by an unresolvable set stays unpinned ⇒ the domain declines (fail-closed).
fn ast_setenum_col_bound(elems: &[Spanned<AstExpr>], sort: &ColSort, vars: &[&str]) -> Option<u64> {
    let mut mx: Option<u64> = None;
    for e in elems {
        let c = resolve_member_code(&e.node, sort, vars)?;
        mx = Some(mx.map_or(c, |m: u64| m.max(c)));
    }
    mx // empty set ⇒ None
}

/// INIT axis bound for column `i` (see the rule block above): a conjunct pins the column when it
/// is `x_i = v` (`v` a nonneg Int literal OR — increment 3 — an enum label for column `i`'s sort,
/// resolved to its code), `x_i ∈ lo..hi` (nonneg literals), or `x_i ∈ {v1,…,vn}` (a LITERAL SET,
/// bound `max_j code(vj)`). Min-folded over pinning conjuncts. `sorts` resolves enum labels.
fn ast_init_col_bound(
    conjs: &[&Spanned<AstExpr>],
    i: u64,
    vars: &[&str],
    sorts: &[ColSort],
) -> Option<u64> {
    let mut h: Option<u64> = None;
    let mut fold = |hi: u64| h = Some(h.map_or(hi, |p: u64| p.min(hi)));
    let sort_i = sorts.get(i as usize);
    for c in conjs {
        match &c.node {
            // `x_i = v` / `v = x_i` — pinned to `code(v)` against column `i`'s sort (a nonneg
            // Int literal ⇒ its value; an enum label ⇒ its index). `resolve_member_code`
            // unifies both and fail-closes on any non-atom (a variable, an expression) ⇒ no pin.
            AstExpr::Eq(a, b) => {
                let try_side = |var_side: &AstExpr, other: &AstExpr| -> Option<u64> {
                    (var_name(var_side).and_then(|n| col_of(n, vars)) == Some(i))
                        .then(|| resolve_member_code(other, sort_i?, vars))
                        .flatten()
                };
                if let Some(code) =
                    try_side(&a.node, &b.node).or_else(|| try_side(&b.node, &a.node))
                {
                    fold(code);
                }
            }
            AstExpr::In(x, dom) => {
                if var_name(&x.node).and_then(|n| col_of(n, vars)) != Some(i) {
                    continue;
                }
                match &dom.node {
                    // `x_i ∈ lo..hi` with literal nonneg bounds — pinned into [lo, hi] ⊆ Nat.
                    AstExpr::Range(lo, hi) => {
                        if let (AstExpr::Int(lo), AstExpr::Int(hi)) = (&lo.node, &hi.node) {
                            if let (Some(_), Some(hi)) = (lo.to_u64(), hi.to_u64()) {
                                fold(hi);
                            }
                        }
                    }
                    // `x_i ∈ {v1,…,vn}` — pinned to `max_j code(vj)` (increment 3).
                    AstExpr::SetEnum(elems) => {
                        if let Some(b) = sort_i.and_then(|s| ast_setenum_col_bound(elems, s, vars))
                        {
                            fold(b);
                        }
                    }
                    _ => {}
                }
            }
            _ => {}
        }
    }
    h
}

/// Pass 2 of the two-pass Init domain rule (increment 6): the ONE-HOP column-equality pin. For a
/// column `i` NOT literally pinned by pass 1 ([`ast_init_col_bound`]), an Init conjunct
/// `x_i = x_j` / `x_j = x_i` (a COLUMN-to-COLUMN equality, `j ≠ i`) where `x_j` IS LITERALLY
/// pinned transfers `x_j`'s bound to `x_i`: by the Aristotle-proved `init_domain_pin_transfer`
/// (job 8ed16b94, sorry-free), `s_j ≤ H_j ∧ s_i = s_j ⇒ s_i ≤ H_j`, so the product axis
/// `[0, H_j]` covers every Init value of `x_i` (`D_init ⊇ {s : Init(s)}` on axis i).
/// AsynchInterface's `Init == val ∈ Data ∧ rdy ∈ {0,1} ∧ ack = rdy` pins `ack` (not literally
/// pinned) to `rdy`'s literal bound 1 through this rule.
///
/// FAIL-CLOSED (returns `None` ⇒ column `i` stays unpinned ⇒ the whole lane declines):
///   · `lit_bounds` is the PASS-1 (literal-pin) result ONLY — this function NEVER consults a
///     pass-2-derived bound, which is EXACTLY what makes it ONE HOP. A CHAIN
///     `x = y ∧ y = z ∧ z ∈ {0,1}` pins `y` (one hop from the literally-pinned `z`) but NOT `x`:
///     `x`'s partner `y` is not LITERALLY pinned, so no bound transfers ⇒ decline `x` ⇒ the lane
///     declines (ATTACK 1). It never "chains" to a possibly-wrong bound;
///   · a partner column that is NOT literally pinned (an unpinned/unbounded RHS — ATTACK 2 — or a
///     CYCLE `x = y ∧ y = x` where neither is literally pinned) ⇒ `None`;
///   · MULTIPLE partners with CONFLICTING (distinct) literal bounds ⇒ `None` (never a guess;
///     agreeing pins are fine);
///   · a non-column other side (a literal / an expression) is SKIPPED — those are pass-1's job
///     (`x_i = lit`) or out of fragment (`x_i = x_j + 1`); if pass 1 did not pin `i`, it declines;
///   · a self-equality `x_i = x_i` is a tautology contributing no pin (skipped).
fn ast_init_col_eq_pin(
    conjs: &[&Spanned<AstExpr>],
    i: u64,
    vars: &[&str],
    lit_bounds: &[Option<u64>],
) -> Option<u64> {
    let mut pinned: Option<u64> = None;
    for c in conjs {
        let AstExpr::Eq(a, b) = &c.node else { continue };
        // A column=column equality touching column i; `j` is the OTHER (distinct) column.
        let ca = var_name(&a.node).and_then(|n| col_of(n, vars));
        let cb = var_name(&b.node).and_then(|n| col_of(n, vars));
        let j = match (ca, cb) {
            (Some(x), Some(y)) if x == i && y != i => y,
            (Some(x), Some(y)) if y == i && x != i => x,
            // Not a `x_i = <other column>` equality (a literal / expression / self-eq) ⇒ pass-1's
            // job or out of fragment; skip (leaves i unpinned unless another conjunct pins it).
            _ => continue,
        };
        // ONE HOP: the partner must be LITERALLY pinned (pass-1). An equality to an unpinned
        // column (ATTACK 2, and the non-terminal edge of a CHAIN — ATTACK 1) DECLINES fail-closed.
        let Some(hj) = lit_bounds.get(j as usize).copied().flatten() else {
            return None;
        };
        match pinned {
            None => pinned = Some(hj),
            Some(prev) if prev == hj => {}
            Some(_) => return None, // conflicting equality pins ⇒ decline (never a guessed bound)
        }
    }
    pinned
}

/// NEXT axis bound for column `i` WITHIN ONE DISJUNCT (see the rule block above): column `i`
/// is pinned by an Eq-pin `x_i' = rhs` (`rhs` a prime-free/in-fragment Int value — bound
/// [`ast_val_ub`]`(rhs)` — OR (increment 3) an enum label for column `i`'s sort — bound its
/// code), a set-membership `x_i' ∈ {v1,…,vn}` / `x_i' ∈ lo..hi` (increment 3 — a
/// non-deterministic assignment; bound `max_j code(vj)` / `hi`), or an `UNCHANGED x_i` conjunct
/// (bound `max_r[i]` — the successor equals the current value). Min-folded over pinning
/// conjuncts. `None` (⇒ DECLINE) when NO conjunct pins column `i`.
fn ast_next_col_bound(
    conjs: &[&Spanned<AstExpr>],
    i: u64,
    vars: &[&str],
    sorts: &[ColSort],
    max_r: &[u64],
) -> Option<u64> {
    let mut h: Option<u64> = None;
    let is_primed_i = |e: &AstExpr| matches!(e, AstExpr::Prime(inner) if var_name(&inner.node).and_then(|n| col_of(n, vars)) == Some(i));
    for c in conjs {
        // The upper bound this conjunct imposes on the successor's column `i`, if it pins it.
        let ub: Option<u64> = match &c.node {
            // Eq-pin `x_i' = rhs`: the Int-fragment `ast_val_ub(rhs)`, else (rhs an enum label
            // for column i's sort) its code. `mode' = mode` (an enum copy) already bounds via
            // `ast_val_ub` (a variable ⇒ `max_r[i]`); only a bare LABEL rhs needs the fallback.
            AstExpr::Eq(lhs, rhs) if is_primed_i(&lhs.node) => {
                ast_val_ub(&rhs.node, vars, sorts, max_r).or_else(|| {
                    sorts
                        .get(i as usize)
                        .and_then(|s| resolve_member_code(&rhs.node, s, vars))
                })
            }
            // Non-deterministic assignment `x_i' ∈ dom`: a literal set (bound `max code`) or an
            // interval (bound `hi`) — every successor value ≤ the bound, so `[0, bound]` covers it.
            AstExpr::In(x, dom) if is_primed_i(&x.node) => match &dom.node {
                AstExpr::SetEnum(elems) => sorts
                    .get(i as usize)
                    .and_then(|s| ast_setenum_col_bound(elems, s, vars)),
                AstExpr::Range(lo, hi) => match (&lo.node, &hi.node) {
                    (AstExpr::Int(lo), AstExpr::Int(hi)) => {
                        lo.to_u64().and(hi.to_u64()) // nonneg lower + return hi
                    }
                    _ => None,
                },
                _ => None,
            },
            // `UNCHANGED x_i` (or `UNCHANGED <<…,x_i,…>>`) — successor's column i EQUALS the
            // current value s_i, and s ∈ R ⇒ s_i ≤ max_r[i]. This is exactly the `x_i' = x_i`
            // Eq-pin bound (`ast_val_ub(x_i) = max_r[i]`); computed directly so the quoter's
            // eq-conjunction expansion and this bound stay in lockstep via `unchanged_cols`.
            AstExpr::Unchanged(operand) => match unchanged_cols(&operand.node, vars) {
                Some(cols) if cols.contains(&i) => max_r.get(usize::try_from(i).ok()?).copied(),
                _ => None,
            },
            _ => None,
        };
        if let Some(ub) = ub {
            h = Some(h.map_or(ub, |p: u64| p.min(ub)));
        }
    }
    h
}

// ===========================================================================
// The AST-direct verdict + the three-leg discharge.
// ===========================================================================

/// Verdict of the AST-direct reflected discharge.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AstDirectVerdict {
    /// All three legs kernel-discharged over the AST-quoted predicates: `R⊆Safety`,
    /// `Init⊆R` (over `D_init`), and `R` closed under `Next` (over `R × D_next`).
    Certified {
        /// `|R|` — reachable states carried through the safety + closure legs.
        states: usize,
        /// `|D_init|` — the AST-rule-derived Init completeness domain size.
        init_domain: usize,
        /// `|D_next|` — the AST-rule-derived Next completeness domain size.
        next_domain: usize,
        /// `|R| × |D_next|` — the closure implication pairs the kernel discharged.
        next_pairs: usize,
    },
    /// A reachable state falsifies the AST-quoted invariant (kernel reduced it to
    /// `Bool.false`). NEVER collapsed into a certify.
    NotSafe {
        /// The offending reachable tuple.
        state: Vec<u64>,
    },
    /// `R` is not closed under the AST-quoted `Next`: `Next(s,sp)` holds but `sp ∉ R`.
    NotClosed {
        /// The source state.
        s: Vec<u64>,
        /// The escaping successor.
        sp: Vec<u64>,
    },
    /// An `Init`-satisfying domain state is missing from `R`.
    NotInitComplete {
        /// The missing `Init` state.
        s: Vec<u64>,
    },
    /// The AST-direct lane DECLINES (fail-closed): out-of-fragment AST, an unboundable
    /// domain axis, a non-Int column, a parse/inline failure, a domain over the cap, or a
    /// kernel non-reduction. NOT a verdict on the certificate.
    Inconclusive(String),
}

/// The quoted obligations + domains an AST-direct run discharges. Factored out so the
/// mis-translation attack tests can substitute a DOCTORED quote and prove the failure is loud.
pub(crate) struct AstDirectPrepared {
    /// Deep-quoted `Safety` (state predicate — quoted with primes disallowed).
    pub(crate) q_safety: Expr,
    /// Deep-quoted `Init` (state predicate).
    pub(crate) q_init: Expr,
    /// Deep-quoted `Next` (action predicate — primes allowed).
    pub(crate) q_next: Expr,
    /// `D_init` — product domain from the AST Init bound rule.
    pub(crate) init_domain: Vec<Vec<u64>>,
    /// `D_next` — product domain from the AST Next bound rule.
    pub(crate) next_domain: Vec<Vec<u64>>,
}

/// Front-end + quote + domain derivation (everything before the kernel legs). `Err` carries
/// the Inconclusive reason. Deterministic: a pure function of the cert's embedded
/// `spec_src` + config fields + stored `reachable`/`sorts`.
fn prepare_ast_direct(cert: &SafetyCertificate) -> Result<AstDirectPrepared, String> {
    let fp = cert
        .explicit_fixpoint
        .as_ref()
        .ok_or("not an explicit-state fixpoint certificate (no `explicit_fixpoint` leg)")?;
    let config = crate::reflect_safety_check::config_from_cert(cert);
    let init_name = config
        .init
        .as_deref()
        .ok_or("no configured Init operator")?;
    let next_name = config
        .next
        .as_deref()
        .ok_or("no configured Next operator")?;
    if config.invariants.is_empty() {
        return Err("no configured invariants".into());
    }

    // Re-parse the cert's own embedded spec — the AST the quoter consumes IS the spec's.
    let tree = tla_core::parse_to_syntax_tree(&cert.spec_src);
    // Defense-in-depth module-binding parity: this lane binds the FIRST module (`lower`), matching
    // how the cert was minted. A genuine cert's `spec_src` is single-module (the certify lanes
    // decline a multi-module file whose stem module is not first — the module-binding false-safe
    // fix). A FORGED multi-module `spec_src` where the two module-selection rules disagree would let
    // this lane quote a DIFFERENT module than `ty check` reads; refuse it fail-closed rather than
    // certify an ambiguous binding.
    let first_bind = tla_core::lower(tla_core::FileId(0), &tree).module;
    let hint_bind = tla_core::lower_main_module(tla_core::FileId(0), &tree, None).module;
    if let (Some(fb), Some(hb)) = (&first_bind, &hint_bind) {
        if fb.name.node != hb.name.node {
            return Err(format!(
                "spec_src binds an ambiguous module (first `{}` vs main `{}`) — refusing to \
                 AST-quote a multi-module spec_src (fail-closed)",
                fb.name.node, hb.name.node
            ));
        }
    }
    let lowered = tla_core::lower(tla_core::FileId(0), &tree);
    let module = lowered.module.ok_or("spec_src did not parse to a module")?;

    // State variables in declaration order — the tuple-column order every lane uses.
    let var_names: Vec<Arc<str>> = module
        .units
        .iter()
        .flat_map(|u| match &u.node {
            Unit::Variable(decls) => decls
                .iter()
                .map(|d| Arc::<str>::from(d.node.as_str()))
                .collect::<Vec<_>>(),
            _ => Vec::new(),
        })
        .collect();
    if var_names.is_empty() {
        return Err("spec declares no VARIABLES".into());
    }
    let n_cols = var_names.len();

    // Consistency + soundness gates on the stored tuples. The arity gate is LOAD-BEARING:
    // quoted column indices are `0..n_cols`, and the deep `TyReflectNth` TOTALIZES an
    // out-of-range index to 0 — a short tuple would be mis-read, so it declines instead.
    if fp.sorts.len() != n_cols {
        return Err(format!(
            "cert stores {} column sort(s) but the spec declares {} variable(s)",
            fp.sorts.len(),
            n_cols
        ));
    }
    // AST-direct covers Int columns and (increment 3) ENUM / model-value columns. Any OTHER sort
    // (Bool, Set, Func, Record, Seq, …) is out of the current embedding ⇒ decline (fail-closed).
    if !fp
        .sorts
        .iter()
        .all(|s| matches!(s, ColSort::Int | ColSort::Enum { .. }))
    {
        return Err(
            "AST-direct covers Int and enum/model-value columns only (another column sort — Bool, \
             Set, Func, Record, Seq, … — is stored)"
                .into(),
        );
    }
    // SOUNDNESS GATE for enum columns: the label→code map is `labels.position(..)`, which is a
    // BIJECTION only when the labels are DISTINCT and non-empty. A genuine cert stores the sorted
    // DISTINCT union, but a FORGED cert with duplicate/empty labels would make the map
    // non-injective (two model values ⇒ one code) — refuse it rather than quote a wrong code.
    for (i, s) in fp.sorts.iter().enumerate() {
        if let ColSort::Enum { labels, .. } = s {
            if labels.is_empty() {
                return Err(format!("enum column {i} has an empty label set — refusing"));
            }
            let mut seen = labels.clone();
            seen.sort();
            seen.dedup();
            if seen.len() != labels.len() {
                return Err(format!(
                    "enum column {i} has duplicate labels (a non-injective label→code map) — \
                     refusing (fail-closed)"
                ));
            }
        }
    }
    if fp.reachable.iter().any(|t| t.len() != n_cols) {
        return Err("a stored reachable tuple's arity differs from the variable count".into());
    }

    let find_op = |name: &str| -> Option<&OperatorDef> {
        module.units.iter().find_map(|unit| match &unit.node {
            Unit::Operator(op) if op.name.node == name => Some(op),
            _ => None,
        })
    };
    let init_def =
        find_op(init_name).ok_or_else(|| format!("Init operator `{init_name}` not found"))?;
    let next_def =
        find_op(next_name).ok_or_else(|| format!("Next operator `{next_name}` not found"))?;

    // Inline zero-arity operators / constants — the SAME deterministic pass (and therefore the
    // same trust surface) the certifier and the recognized lane's spec-bind use.
    let inline_env = crate::cert_inline::CertInlineEnv::new(&module, &config, &var_names);
    let init_body = inline_env.inline(&init_def.body);
    let next_body = inline_env.inline(&next_def.body);
    // The safety predicate: the configured invariants conjoined left-nested in config order —
    // identical to every other lane's reading.
    let safety_body = {
        let mut it = config.invariants.iter();
        let first = it.next().expect("non-empty checked above");
        let first_def =
            find_op(first).ok_or_else(|| format!("invariant operator `{first}` not found"))?;
        let mut acc = inline_env.inline(&first_def.body);
        for name in it {
            let def =
                find_op(name).ok_or_else(|| format!("invariant operator `{name}` not found"))?;
            let leg = inline_env.inline(&def.body);
            acc = Spanned::dummy(AstExpr::And(Box::new(acc), Box::new(leg)));
        }
        acc
    };

    // Increment 4: resolve MODEL-VALUE-SET CONSTANT domains. `cert_inline` deliberately leaves a
    // `ModelValueSet` constant as a bare `Ident` (the recognized lane's mvsets path reads it), so a
    // membership `mode ∈ Modes` reaches the quoter with an `Ident` domain and declines. The `.cfg`
    // assignment (`CONSTANT Modes = {idle, busy}`) IS a literal element list, so rewrite
    // `In(x, Ident(C))` → `In(x, SetEnum({m1,…,mk}))` for exactly the cfg's `ModelValueSet`
    // constants — after which the PROVEN Or-fold desugar (`R1_set_membership_or_fold.lean`) and the
    // label→code resolution apply unchanged. Fail-closed: only the domain slot of `In`, only for a
    // `ModelValueSet` constant (a `Value`/`Replacement`/unknown Ident domain still declines), and
    // the rewritten elements still pass the column-local kind/label gates (an mvset element that is
    // not one of the column's labels declines at `quote_setenum_mem_ast`, never a wrong code). The
    // trust surface is the same `.cfg` constants map every lane (and `ty check`) already reads.
    let init_body = resolve_mvset_domains(init_body, &config);
    let next_body = resolve_mvset_domains(next_body, &config);
    let safety_body = resolve_mvset_domains(safety_body, &config);

    let vars: Vec<&str> = var_names.iter().map(|v| v.as_ref()).collect();

    // Per-column reachable maximum over the STORED `R`. Computed BEFORE the quote because the
    // bounded-subtraction arm (increment 5) needs `max_r[i]` to prove a subtrahend `≤` its
    // minuend. This IS a true upper bound on an unprimed column's value wherever `Next` is
    // kernel-evaluated: the closure leg reads the current state from exactly this `R`.
    //
    // LOAD-BEARING INVARIANT (do NOT break in a refactor): the bounded-subtraction fidelity of
    // increment 5 rests on `q_next` being kernel-evaluated (leg 3, `run_ast_direct_legs`) with the
    // CURRENT state `s` ranging over EXACTLY this `fp.reachable` — the same slice `max_r` is the
    // column-wise max of. The admitted subtrahend is UNPRIMED, so `Next` reads `s_i ≤ max_r[i] ≤ c`
    // and `Nat.sub c s_i` cannot truncate. If a future change evaluated `q_next` with a current
    // state drawn from a LARGER domain than this `R` (e.g. reusing `q_next` in a symmetry/liveness
    // leg over `D_next`), the `x ≤ c` premise would no longer hold and the truncation barrier would
    // silently break. Keep leg-3's current-state domain ⊆ this `max_r` basis. (Skeptic-flagged.)
    let max_r: Vec<u64> = (0..n_cols)
        .map(|i| fp.reachable.iter().map(|t| t[i]).max().unwrap_or(0))
        .collect();

    // QUOTE — the recognizer is never consulted. Init/Safety are STATE predicates (primes
    // disallowed — the quoter itself is the primed-leaf gate).
    let q_safety = quote_pred_ast(&safety_body.node, &vars, &fp.sorts, &max_r, false)
        .ok_or("safety body outside the AST-direct fragment (or mentions a primed variable)")?;
    let q_init = quote_pred_ast(&init_body.node, &vars, &fp.sorts, &max_r, false)
        .ok_or("Init body outside the AST-direct fragment (or mentions a primed variable)")?;
    let q_next = quote_pred_ast(&next_body.node, &vars, &fp.sorts, &max_r, true)
        .ok_or("Next body outside the AST-direct fragment")?;

    // Domain derivation (the trusted-Rust bound rule — see the rule block above).
    let mut init_conjs = Vec::new();
    flatten_and(&init_body, &mut init_conjs);
    // Split Next into top-level disjuncts, each flattened into its own conjunct list. A plain
    // conjunctive Next yields a SINGLE disjunct whose conjunct list equals the pre-disjunctive
    // `flatten_and(&next_body)` — so HourClock's derivation is byte-identical.
    let mut next_disjuncts = Vec::new();
    flatten_or(&next_body, &mut next_disjuncts);
    let disjunct_conjs: Vec<Vec<&Spanned<AstExpr>>> = next_disjuncts
        .iter()
        .map(|d| {
            let mut cs = Vec::new();
            flatten_and(d, &mut cs);
            cs
        })
        .collect();
    // TWO-PASS Init domain derivation (increment 6). PASS 1: each column's LITERAL pin (`= lit`/`=
    // label`, `∈ lo..hi`, `∈ {…}`) via `ast_init_col_bound`. PASS 2 (`ast_init_col_eq_pin`): a
    // column NOT literally pinned inherits a ONE-HOP column-equality bound `x_i = x_j` from a
    // LITERALLY-pinned `x_j` (Aristotle `init_domain_pin_transfer`). Pass 2 consults ONLY the
    // pass-1 `lit_init_bounds` — never a pass-2 result — so it is one hop: no chains, no cycles.
    let lit_init_bounds: Vec<Option<u64>> = (0..n_cols)
        .map(|i| ast_init_col_bound(&init_conjs, i as u64, &vars, &fp.sorts))
        .collect();
    let mut init_bounds = Vec::with_capacity(n_cols);
    let mut next_bounds = Vec::with_capacity(n_cols);
    for i in 0..n_cols {
        let init_bound = lit_init_bounds[i]
            .or_else(|| ast_init_col_eq_pin(&init_conjs, i as u64, &vars, &lit_init_bounds));
        init_bounds.push(init_bound.ok_or_else(|| {
            format!(
                "Init does not pin column {i} (`{}`) into a finite nonneg range (`= lit`/`= \
                 label`, `∈ lo..hi`, `∈ {{v1,…,vn}}` with literal/label elements, or `= \
                 <literally-pinned column>` — the one-hop column-equality pin) — the \
                 Init-completeness domain is underivable",
                vars[i]
            )
        })?);
        // Next axis i = MAX over disjuncts of each disjunct's per-column bound. EVERY disjunct
        // must pin column i (Eq-pin or `UNCHANGED x_i`); if ANY does not, its successors escape
        // an under-covering domain ⇒ DECLINE (soundness: D_next ⊇ Succ(R) requires covering the
        // MAX, and a single unpinned disjunct means an unbounded successor — never MAX over the
        // pinning subset). `disjunct_conjs` is non-empty (a non-`Or` Next is one disjunct).
        let mut nb: Option<u64> = None;
        for cs in &disjunct_conjs {
            let b =
                ast_next_col_bound(cs, i as u64, &vars, &fp.sorts, &max_r).ok_or_else(|| {
                    format!(
                        "a Next disjunct does not pin column {i} (`{}`) via `{}' = <prime-free \
                     in-fragment value / label>`, `{}' ∈ {{…}}`/`lo..hi`, or `UNCHANGED {}` — the \
                     Next-completeness domain is underivable (fail-closed: one unpinned disjunct \
                     admits an unbounded successor)",
                        vars[i], vars[i], vars[i], vars[i]
                    )
                })?;
            nb = Some(nb.map_or(b, |p: u64| p.max(b)));
        }
        next_bounds.push(nb.ok_or("Next body has no disjuncts (empty) — underivable domain")?);
    }
    let cap = crate::explicit_fixpoint_cert::DEFAULT_FIXPOINT_STATE_CAP;
    let init_domain = crate::cleancic::product_domain(&init_bounds, cap)
        .ok_or("the Init product domain exceeds the state cap (or overflows)")?;
    let next_domain = crate::cleancic::product_domain(&next_bounds, cap)
        .ok_or("the Next product domain exceeds the state cap (or overflows)")?;

    Ok(AstDirectPrepared {
        q_safety,
        q_init,
        q_next,
        init_domain,
        next_domain,
    })
}

/// Discharge the three kernel legs over prepared (quoted) obligations. Fail-closed on the
/// first non-reduction; definitive on the first `Bool.false`.
pub(crate) fn run_ast_direct_legs(
    prep: &AstDirectPrepared,
    reachable: &[Vec<u64>],
) -> AstDirectVerdict {
    // (1) R ⊆ Safety — over the STORED reachable set (a tamper that INTRODUCES a violating
    // state is caught HERE by the kernel).
    for s in reachable {
        match kernel_eval_quoted_pred(&prep.q_safety, s, s) {
            Some(true) => {}
            Some(false) => return AstDirectVerdict::NotSafe { state: s.clone() },
            None => {
                return AstDirectVerdict::Inconclusive(format!(
                    "safety obligation did not kernel-reduce at state {s:?}"
                ))
            }
        }
    }
    // (2) Init-completeness: ∀ s∈D_init: Init(s) ⇒ s∈R.
    for s in &prep.init_domain {
        match kernel_eval_quoted_implies_mem(&prep.q_init, s, s, s, reachable) {
            Some(true) => {}
            Some(false) => return AstDirectVerdict::NotInitComplete { s: s.clone() },
            None => {
                return AstDirectVerdict::Inconclusive(format!(
                    "Init-completeness obligation did not kernel-reduce at state {s:?}"
                ))
            }
        }
    }
    // (3) Next-completeness / closure: ∀ s∈R, sp∈D_next: Next(s,sp) ⇒ sp∈R. THE load-bearing
    // closure guard — a non-closed R reduces to Bool.false HERE. NOTE (increment-5 invariant): the
    // current state `s` here ranges over EXACTLY `reachable` — the same `R` `max_r` bounds in
    // `prepare_ast_direct`. This coupling is what makes the admitted bounded subtraction `c - x`
    // (x unprimed, so read from `s ∈ R`, x ≤ max_r ≤ c) truncation-free. Do not widen `s`'s domain.
    let mut pairs = 0usize;
    for s in reachable {
        for sp in &prep.next_domain {
            match kernel_eval_quoted_implies_mem(&prep.q_next, s, sp, sp, reachable) {
                Some(true) => pairs += 1,
                Some(false) => {
                    return AstDirectVerdict::NotClosed {
                        s: s.clone(),
                        sp: sp.clone(),
                    }
                }
                None => {
                    return AstDirectVerdict::Inconclusive(format!(
                        "closure obligation did not kernel-reduce at s={s:?}, sp={sp:?}"
                    ))
                }
            }
        }
    }
    AstDirectVerdict::Certified {
        states: reachable.len(),
        init_domain: prep.init_domain.len(),
        next_domain: prep.next_domain.len(),
        next_pairs: pairs,
    }
}

/// The AST-direct reflected discharge (no cross-check): quote `Init`/`Next`/`Safety` from the
/// cert's own re-parsed + inlined AST, derive the completeness domains by the AST bound rule,
/// and kernel-discharge the three legs. See the module docs for the honest trust base.
pub fn reflect_check_safety_cert_ast_direct(cert: &SafetyCertificate) -> AstDirectVerdict {
    let fp_reachable = match &cert.explicit_fixpoint {
        Some(fp) => fp.reachable.clone(),
        None => {
            return AstDirectVerdict::Inconclusive(
                "not an explicit-state fixpoint certificate (no `explicit_fixpoint` leg)".into(),
            )
        }
    };
    match prepare_ast_direct(cert) {
        Ok(prep) => run_ast_direct_legs(&prep, &fp_reachable),
        Err(why) => AstDirectVerdict::Inconclusive(why),
    }
}

// ===========================================================================
// CROSS-CHECK against the recognized-IR reflected lane (`--full`).
// ===========================================================================

/// Outcome of cross-checking the AST-direct verdict against the recognized-IR reflected lane
/// on the SAME certificate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AstCrossCheck {
    /// Both lanes conclusive and CLASS-CONSISTENT (both certify, or both refuse — the refusal
    /// classes may differ because the lanes derive different domains / run legs over different
    /// quotes; what matters for soundness is that no lane certifies what the other refutes).
    Agree {
        /// Human-readable summary of the recognized lane's verdict.
        recognized: String,
    },
    /// The recognized-IR lane is INCONCLUSIVE (out of ITS fragment) — no cross-check
    /// available. Surfaced, not an error: the AST-direct verdict stands on its own stated
    /// trust base.
    Unavailable {
        /// The recognized lane's decline reason.
        reason: String,
    },
    /// HARD FAIL-CLOSED ERROR: the lanes reached CONTRADICTORY conclusive verdicts (one
    /// certifies, the other refutes). Evidence of a quoter / recognizer / embedding bug —
    /// callers must trust NEITHER verdict and must NOT certify.
    Divergence {
        /// The AST-direct verdict, rendered.
        ast: String,
        /// The recognized-IR verdict, rendered.
        recognized: String,
    },
}

/// Classify the cross-check. Divergence ⟺ both lanes are conclusive and exactly one
/// certifies. An AST-direct `Inconclusive` never diverges (the lane declined — the recognized
/// outcome is reported alongside as fallback information).
pub(crate) fn classify_crosscheck(
    ast: &AstDirectVerdict,
    recognized: &crate::reflect_safety_check::ReflectFullVerdict,
) -> AstCrossCheck {
    use crate::reflect_safety_check::ReflectFullVerdict as R;
    let rec_str = format!("{recognized:?}");
    let rec_certifies = match recognized {
        R::Certified { .. } => Some(true),
        R::NotSafe { .. } | R::NotClosed { .. } | R::NotInitComplete { .. } | R::Rejected(_) => {
            Some(false)
        }
        R::Inconclusive(why) => {
            return AstCrossCheck::Unavailable {
                reason: why.clone(),
            };
        }
    };
    let ast_certifies = match ast {
        AstDirectVerdict::Certified { .. } => Some(true),
        AstDirectVerdict::NotSafe { .. }
        | AstDirectVerdict::NotClosed { .. }
        | AstDirectVerdict::NotInitComplete { .. } => Some(false),
        AstDirectVerdict::Inconclusive(_) => None,
    };
    match ast_certifies {
        None => AstCrossCheck::Unavailable {
            reason: format!(
                "the AST-direct lane declined; the recognized-IR lane's independent verdict: \
                 {rec_str}"
            ),
        },
        Some(a) if Some(a) == rec_certifies => AstCrossCheck::Agree {
            recognized: rec_str,
        },
        Some(_) => AstCrossCheck::Divergence {
            ast: format!("{ast:?}"),
            recognized: rec_str,
        },
    }
}

/// Run the AST-direct discharge AND the recognized-IR reflected lane on the same cert, and
/// REQUIRE verdict-class agreement wherever both are conclusive ([`AstCrossCheck::Divergence`]
/// otherwise — a hard error; trust neither). The recognized lane is run WITHOUT
/// `--require-domain-complete` (its domain residual is orthogonal to the agreement question).
pub fn reflect_check_ast_direct_with_crosscheck(
    cert: &SafetyCertificate,
) -> (AstDirectVerdict, AstCrossCheck) {
    let ast = reflect_check_safety_cert_ast_direct(cert);
    let recognized = crate::reflect_safety_check::reflect_check_safety_cert_full(cert, false);
    let cc = classify_crosscheck(&ast, &recognized);
    (ast, cc)
}

// ===========================================================================
// Tests. The kernel is the arbiter for every semantic assertion; the attack
// tests prove wrong quotes are LOUD (divergence or a failing leg), never
// silent.
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::explicit_fixpoint_cert::{
        certify_explicit_state_spec, ColSort, ExplicitFixpointCert, PredIR, ValIR,
    };
    use tla_core::name_intern::NameId;

    // ── AST shorthands (tests only) ─────────────────────────────────────────
    fn sp(e: AstExpr) -> Box<Spanned<AstExpr>> {
        Box::new(Spanned::dummy(e))
    }
    fn ident(name: &str) -> AstExpr {
        AstExpr::Ident(name.to_string(), NameId::INVALID)
    }
    fn int(n: i64) -> AstExpr {
        AstExpr::Int(num_bigint::BigInt::from(n))
    }
    fn prime(name: &str) -> AstExpr {
        AstExpr::Prime(sp(ident(name)))
    }

    const VARS: &[&str] = &["hr", "m"];
    /// Two Int columns — the sort context for the Int-fragment quoter tests.
    const SORTS: &[ColSort] = &[ColSort::Int, ColSort::Int];
    /// Per-column reachable maxima for the quoter tests (consumed only by the bounded-subtraction
    /// arm). Deliberately GENEROUS (12) so the general 1:1 / decline tests are unaffected; the
    /// subtraction tests pass their own tight `max_r` to exercise the `x ≤ c` gate.
    const MAX_R: &[u64] = &[12, 12];

    /// Per-constructor 1:1: each admitted AST node quotes to EXACTLY its deep constructor
    /// over recursively-quoted children — pinned arm by arm against the `deep` builders.
    #[test]
    fn quoter_is_1to1_per_constructor() {
        let qp = |e: &AstExpr| quote_pred_ast(e, VARS, SORTS, MAX_R, true);
        let qv = |e: &AstExpr| quote_val_ast(e, VARS, SORTS, MAX_R, true);
        // Values.
        assert_eq!(qv(&int(7)), Some(deep::lit(7)));
        assert_eq!(qv(&ident("hr")), Some(deep::var(0)));
        assert_eq!(qv(&ident("m")), Some(deep::var(1)));
        assert_eq!(qv(&prime("m")), Some(deep::prime(1)));
        assert_eq!(
            qv(&AstExpr::Add(sp(ident("hr")), sp(int(1)))),
            Some(deep::add(deep::var(0), deep::lit(1)))
        );
        // A pre-resolved StateVar maps BY NAME (its embedded index is ignored — here a WRONG
        // index 7 is stored and the quote still yields column 0).
        assert_eq!(
            qv(&AstExpr::StateVar("hr".into(), 7, NameId::INVALID)),
            Some(deep::var(0))
        );
        // Predicates: comparisons.
        let hr_lt_12 = AstExpr::Lt(sp(ident("hr")), sp(int(12)));
        assert_eq!(qp(&hr_lt_12), Some(deep::lt(deep::var(0), deep::lit(12))));
        assert_eq!(
            qp(&AstExpr::Leq(sp(ident("hr")), sp(int(12)))),
            Some(deep::leq(deep::var(0), deep::lit(12)))
        );
        assert_eq!(
            qp(&AstExpr::Gt(sp(int(3)), sp(ident("m")))),
            Some(deep::gt(deep::lit(3), deep::var(1)))
        );
        assert_eq!(
            qp(&AstExpr::Geq(sp(ident("m")), sp(int(0)))),
            Some(deep::geq(deep::var(1), deep::lit(0)))
        );
        assert_eq!(
            qp(&AstExpr::Eq(sp(ident("hr")), sp(int(5)))),
            Some(deep::eq(deep::var(0), deep::lit(5)))
        );
        assert_eq!(
            qp(&AstExpr::Neq(sp(ident("hr")), sp(int(12)))),
            Some(deep::neq(deep::var(0), deep::lit(12)))
        );
        // Boolean combinators.
        let t = || AstExpr::Eq(sp(ident("hr")), sp(int(1)));
        let u = || AstExpr::Eq(sp(ident("m")), sp(int(2)));
        let dt = || deep::eq(deep::var(0), deep::lit(1));
        let du = || deep::eq(deep::var(1), deep::lit(2));
        assert_eq!(
            qp(&AstExpr::And(sp(t()), sp(u()))),
            Some(deep::and(dt(), du()))
        );
        assert_eq!(
            qp(&AstExpr::Or(sp(t()), sp(u()))),
            Some(deep::or(dt(), du()))
        );
        assert_eq!(qp(&AstExpr::Not(sp(t()))), Some(deep::not(dt())));
        assert_eq!(
            qp(&AstExpr::Implies(sp(t()), sp(u()))),
            Some(deep::implies(dt(), du()))
        );
        assert_eq!(
            qp(&AstExpr::Equiv(sp(t()), sp(u()))),
            Some(deep::equiv(dt(), du()))
        );
        // Interval membership.
        assert_eq!(
            qp(&AstExpr::In(
                sp(ident("hr")),
                sp(AstExpr::Range(sp(int(1)), sp(int(12))))
            )),
            Some(deep::in_range(deep::var(0), deep::lit(1), deep::lit(12)))
        );
        // The pinned conditional update.
        let ite = AstExpr::Eq(
            sp(prime("hr")),
            sp(AstExpr::If(
                sp(AstExpr::Neq(sp(ident("hr")), sp(int(12)))),
                sp(AstExpr::Add(sp(ident("hr")), sp(int(1)))),
                sp(int(1)),
            )),
        );
        assert_eq!(
            qp(&ite),
            Some(deep::eq_ite(
                deep::prime(0),
                deep::neq(deep::var(0), deep::lit(12)),
                deep::add(deep::var(0), deep::lit(1)),
                deep::lit(1),
            ))
        );
    }

    /// FAIL-CLOSED: every out-of-fragment node declines (`None`) — never a wrong quote.
    /// Covers the deliberate general-`Sub`/`Div`/`Mod`/`Neg` exclusions (Nat-truncation
    /// infidelity), negative literals, unknown identifiers, primes in state predicates, primes of
    /// non-variables, `In` over non-`Range` domains, and structural out-of-fragment nodes. (The
    /// ONE admitted subtraction shape — a bounded `c - x` — is covered by its own increment-5
    /// tests; here every OTHER `Sub` shape must decline.)
    #[test]
    fn quoter_declines_out_of_fragment() {
        let qp = |e: &AstExpr| quote_pred_ast(e, VARS, SORTS, MAX_R, true);
        let qv = |e: &AstExpr| quote_val_ast(e, VARS, SORTS, MAX_R, true);
        // Nat-infidelity exclusions. `3 - 5` (TLA+ `-2`, Nat `0`) declines: the subtrahend `5` is
        // a LITERAL, not a bounded column, so it is outside the admitted `c - x` shape. `x - y`
        // (both columns) and `hr - 1` (non-literal minuend) likewise decline.
        assert_eq!(qv(&AstExpr::Sub(sp(int(3)), sp(int(5)))), None);
        assert_eq!(qv(&AstExpr::Sub(sp(ident("hr")), sp(ident("m")))), None); // x - y (both cols)
        assert_eq!(qv(&AstExpr::Sub(sp(ident("hr")), sp(int(1)))), None); // non-literal minuend
        assert_eq!(qv(&AstExpr::Div(sp(ident("hr")), sp(int(2)))), None);
        assert_eq!(qv(&AstExpr::IntDiv(sp(ident("hr")), sp(int(2)))), None);
        assert_eq!(qv(&AstExpr::Mod(sp(ident("hr")), sp(int(2)))), None);
        assert_eq!(qv(&AstExpr::Neg(sp(int(1)))), None);
        // Negative / oversized literals.
        assert_eq!(qv(&int(-1)), None);
        // An identifier that is NOT a declared state variable (an un-inlined constant).
        assert_eq!(qv(&ident("MaxHour")), None);
        // Primes disallowed in STATE predicates; primes of non-variables always decline.
        assert_eq!(quote_val_ast(&prime("hr"), VARS, SORTS, MAX_R, false), None);
        assert_eq!(
            qv(&AstExpr::Prime(sp(AstExpr::Add(
                sp(ident("hr")),
                sp(int(1))
            )))),
            None
        );
        // Membership over a NON-literal domain declines (increment 3 admits `{v1,…,vn}` and
        // `lo..hi`, but a variable / set-builder domain is still out of fragment).
        assert_eq!(
            qp(&AstExpr::In(sp(ident("hr")), sp(ident("m")))), // `hr ∈ m` (m a variable set)
            None
        );
        // An EMPTY literal set `hr ∈ {}` declines (always-false; no false-pred ctor to fabricate).
        assert_eq!(
            qp(&AstExpr::In(sp(ident("hr")), sp(AstExpr::SetEnum(vec![])))),
            None
        );
        // A literal set whose ELEMENT is out of the column's value fragment (`hr` is Int; a
        // String element is not an Int literal) declines — the poison rule.
        assert_eq!(
            qp(&AstExpr::In(
                sp(ident("hr")),
                sp(AstExpr::SetEnum(vec![
                    Spanned::dummy(int(1)),
                    Spanned::dummy(AstExpr::String("bad".into())),
                ]))
            )),
            None
        );
        // Structural out-of-fragment nodes: CHOOSE, quantifiers, functions, records, strings.
        assert_eq!(qp(&AstExpr::String("working".into())), None);
        assert_eq!(qv(&AstExpr::String("working".into())), None);
        assert_eq!(qp(&AstExpr::FuncApply(sp(ident("hr")), sp(int(1)))), None);
        assert_eq!(qp(&AstExpr::Bool(true)), None);
        // A bare value-position If OUTSIDE the `l = IF …` shape declines (only the Eq-pinned
        // conditional is in the v1 embedding).
        assert_eq!(
            qv(&AstExpr::If(
                sp(AstExpr::Bool(true)),
                sp(int(1)),
                sp(int(2))
            )),
            None
        );
        // An If on the LEFT of Eq also declines (the composite arm admits RHS only).
        assert_eq!(
            qp(&AstExpr::Eq(
                sp(AstExpr::If(sp(AstExpr::Bool(true)), sp(int(1)), sp(int(2)))),
                sp(prime("hr")),
            )),
            None
        );
        // Poisoning: an out-of-fragment CHILD declines the whole quote.
        assert_eq!(
            qp(&AstExpr::And(
                sp(AstExpr::Eq(sp(ident("hr")), sp(int(1)))),
                sp(AstExpr::String("bad".into()))
            )),
            None
        );
    }

    /// INCREMENT 5 — the bounded-subtraction ADMISSION (`c - x`, `x ≤ c`). `1 - x` for a 0/1
    /// column quotes to `deep::sub(lit 1, var i)` (the AsynchInterface shape), and ONLY that
    /// proved-safe shape: a wider bound, a non-literal minuend, `x - y`, a primed subtrahend, a
    /// state predicate, or a non-Int column all DECLINE. `sub_cols`/`sub_max_r`: `x∈{0,1}`,
    /// `y∈{0,5}` — so `y`'s column max is 5.
    #[test]
    fn quoter_admits_bounded_subtraction_1_minus_x() {
        let sub_vars: &[&str] = &["x", "y"];
        let sub_sorts: &[ColSort] = &[ColSort::Int, ColSort::Int];
        let sub_max_r: &[u64] = &[1, 5]; // x's reachable max is 1; y's is 5
                                         // POSITIVE: `1 - x` in an ACTION context (allow_prime = true), x ∈ {0,1} (max_r=1 ≤ 1) ⇒
                                         // quotes to `sub(lit 1, var 0)` — 1:1 with the kernel `TyReflectPExp.sub` ctor.
        assert_eq!(
            quote_val_ast(
                &AstExpr::Sub(sp(int(1)), sp(ident("x"))),
                sub_vars,
                sub_sorts,
                sub_max_r,
                true
            ),
            Some(deep::sub(deep::lit(1), deep::var(0)))
        );
        // A wider-but-still-covering literal: `5 - y` with y's max 5 (bound = c) ⇒ ADMIT (5 ≤ 5).
        assert_eq!(
            quote_val_ast(
                &AstExpr::Sub(sp(int(5)), sp(ident("y"))),
                sub_vars,
                sub_sorts,
                sub_max_r,
                true
            ),
            Some(deep::sub(deep::lit(5), deep::var(1)))
        );
        // ATTACK 1 (THE truncation test): `3 - y` where y can reach 5 (bound 5 > 3). TLA+
        // `3 - 5 = -2`; Nat `3 ∸ 5 = 0`. Admitting it would let the kernel compute 0 where TLA+
        // goes negative — a FALSE SAFE. The bound `max_r[y]=5 > 3` MUST force a DECLINE.
        assert_eq!(
            quote_val_ast(
                &AstExpr::Sub(sp(int(3)), sp(ident("y"))),
                sub_vars,
                sub_sorts,
                sub_max_r,
                true
            ),
            None
        );
        // `x - 1` where x can be 0: TLA+ `0 - 1 = -1`, Nat `0`. Non-literal minuend ⇒ DECLINE.
        assert_eq!(
            quote_val_ast(
                &AstExpr::Sub(sp(ident("x")), sp(int(1))),
                sub_vars,
                sub_sorts,
                sub_max_r,
                true
            ),
            None
        );
        // `x - y` (both columns) ⇒ DECLINE (no literal minuend to bound against).
        assert_eq!(
            quote_val_ast(
                &AstExpr::Sub(sp(ident("x")), sp(ident("y"))),
                sub_vars,
                sub_sorts,
                sub_max_r,
                true
            ),
            None
        );
        // A PRIMED subtrahend `1 - x'` reads the successor from D_next (a different, Rust-bounded
        // domain, not R) ⇒ DECLINE (only an unprimed column is proved ≤ c by max_r).
        assert_eq!(
            quote_val_ast(
                &AstExpr::Sub(sp(int(1)), sp(prime("x"))),
                sub_vars,
                sub_sorts,
                sub_max_r,
                true
            ),
            None
        );
        // STATE predicate (allow_prime = false): subtraction is DECLINED entirely (the current
        // state in a state leg is R or D_init — max_r is not the governing bound there).
        assert_eq!(
            quote_val_ast(
                &AstExpr::Sub(sp(int(1)), sp(ident("x"))),
                sub_vars,
                sub_sorts,
                sub_max_r,
                false
            ),
            None
        );
        // A non-Int (enum) column subtrahend ⇒ DECLINE (`c - modelvalue` is a category error).
        let enum_sorts: &[ColSort] = &[
            ColSort::Enum {
                labels: vec!["a".into(), "b".into()],
                kind: EnumKind::Model,
            },
            ColSort::Int,
        ];
        assert_eq!(
            quote_val_ast(
                &AstExpr::Sub(sp(int(1)), sp(ident("x"))),
                sub_vars,
                enum_sorts,
                sub_max_r,
                true
            ),
            None
        );
    }

    /// INCREMENT 5 — the domain-bound rule `ub(c - x) = c`. The kernel computes truncating
    /// `Nat.sub`, whose result is `≤` the minuend, so `c` covers the successor `col' = c - x`
    /// regardless of x. This holds even when the QUOTER would have DECLINED (`ub` is a coverage
    /// bound, not a fidelity gate) — e.g. `ub(3 - y) = 3` even though `3 - y` (y≤5) is unsafe.
    #[test]
    fn ast_val_ub_subtraction_bound_is_the_minuend() {
        let vars = &["x", "y"];
        let sorts: &[ColSort] = &[ColSort::Int, ColSort::Int];
        let max_r = &[1u64, 5u64];
        assert_eq!(
            ast_val_ub(
                &AstExpr::Sub(sp(int(1)), sp(ident("x"))),
                vars,
                sorts,
                max_r
            ),
            Some(1)
        );
        assert_eq!(
            ast_val_ub(
                &AstExpr::Sub(sp(int(3)), sp(ident("y"))),
                vars,
                sorts,
                max_r
            ),
            Some(3)
        );
        // A Next Eq-pin `x' = 1 - x` (over max_r=[1,5]) bounds the successor column to 1.
        let pin = Spanned::dummy(AstExpr::Eq(
            sp(prime("x")),
            sp(AstExpr::Sub(sp(int(1)), sp(ident("x")))),
        ));
        let mut cs = Vec::new();
        flatten_and(&pin, &mut cs);
        assert_eq!(ast_next_col_bound(&cs, 0, vars, sorts, max_r), Some(1));
    }

    /// The AST bound rules on HourClock's own (parsed) bodies, plus their fail-closed edges:
    /// bare upper bounds do NOT bound Init (negative-Int honesty), unpinned Next columns are
    /// unboundable, primed rhs declines.
    #[test]
    fn ast_bound_rules_hourclock_and_fail_closed_edges() {
        let vars = &["hr"];
        let sorts: &[ColSort] = &[ColSort::Int];
        // Init `hr ∈ 1..12` ⇒ H = 12.
        let init = Spanned::dummy(AstExpr::In(
            sp(ident("hr")),
            sp(AstExpr::Range(sp(int(1)), sp(int(12)))),
        ));
        let mut conjs = Vec::new();
        flatten_and(&init, &mut conjs);
        assert_eq!(ast_init_col_bound(&conjs, 0, vars, sorts), Some(12));
        // Init `hr = 1` ⇒ H = 1 (either orientation).
        let init_eq = Spanned::dummy(AstExpr::Eq(sp(int(1)), sp(ident("hr"))));
        let mut conjs_eq = Vec::new();
        flatten_and(&init_eq, &mut conjs_eq);
        assert_eq!(ast_init_col_bound(&conjs_eq, 0, vars, sorts), Some(1));
        // NEGATIVE-INT HONESTY: a bare `hr ≤ 12` admits negative TLA+ Init values a Nat
        // domain cannot cover ⇒ NO bound (deliberately stricter than the IR lane).
        let init_leq = Spanned::dummy(AstExpr::Leq(sp(ident("hr")), sp(int(12))));
        let mut conjs_leq = Vec::new();
        flatten_and(&init_leq, &mut conjs_leq);
        assert_eq!(ast_init_col_bound(&conjs_leq, 0, vars, sorts), None);
        // Next `hr' = IF hr # 12 THEN hr+1 ELSE 1` over max(R)=12 ⇒ ub = max(12+1, 1) = 13.
        let next = Spanned::dummy(AstExpr::Eq(
            sp(prime("hr")),
            sp(AstExpr::If(
                sp(AstExpr::Neq(sp(ident("hr")), sp(int(12)))),
                sp(AstExpr::Add(sp(ident("hr")), sp(int(1)))),
                sp(int(1)),
            )),
        ));
        let mut nconjs = Vec::new();
        flatten_and(&next, &mut nconjs);
        assert_eq!(ast_next_col_bound(&nconjs, 0, vars, sorts, &[12]), Some(13));
        // An UNPINNED column (`hr' > 0` — no Eq-pin) is unboundable ⇒ None (fail-closed).
        let loose = Spanned::dummy(AstExpr::Gt(sp(prime("hr")), sp(int(0))));
        let mut lconjs = Vec::new();
        flatten_and(&loose, &mut lconjs);
        assert_eq!(ast_next_col_bound(&lconjs, 0, vars, sorts, &[12]), None);
        // A PRIMED rhs (`hr' = hr' + 1`) declines the pin (ub requires prime-free).
        let primed_rhs = Spanned::dummy(AstExpr::Eq(
            sp(prime("hr")),
            sp(AstExpr::Add(sp(prime("hr")), sp(int(1)))),
        ));
        let mut pconjs = Vec::new();
        flatten_and(&primed_rhs, &mut pconjs);
        assert_eq!(ast_next_col_bound(&pconjs, 0, vars, sorts, &[12]), None);
    }

    // ── End-to-end over certificates ─────────────────────────────────────────

    const HC_SRC: &str = "\
---------------------- MODULE HourClock ----------------------
EXTENDS Naturals
VARIABLE hr
HCini  ==  hr \\in (1 .. 12)
HCnxt  ==  hr' = IF hr # 12 THEN hr + 1 ELSE 1
HC  ==  HCini /\\ [][HCnxt]_hr
--------------------------------------------------------------
THEOREM  HC => []HCini
==============================================================
";

    fn hc_config() -> crate::Config {
        crate::Config {
            init: Some("HCini".into()),
            next: Some("HCnxt".into()),
            invariants: vec!["HCini".into()],
            ..Default::default()
        }
    }

    /// A REAL HourClock cert from the certifier (genuine general legs + reachable set).
    fn hourclock_full_cert() -> SafetyCertificate {
        let fp = certify_explicit_state_spec(HC_SRC, &hc_config())
            .expect("HourClock must explicit-state certify");
        let mut cert = crate::cert::build_explicit_fixpoint_certificate(HC_SRC, &hc_config(), fp);
        cert.digest = cert.compute_digest();
        cert
    }

    /// A HANDCRAFTED explicit-state cert: arbitrary spec/config/invariant-name/R. The AST
    /// lane reads only `spec_src` + config names + `reachable`/`sorts`; the stored IR fields
    /// matter only to the recognized lane (the cross-check tests build certifier-real certs).
    fn handmade_cert(
        src: &str,
        invariants: Vec<String>,
        reachable: Vec<Vec<u64>>,
    ) -> SafetyCertificate {
        let fp = ExplicitFixpointCert {
            reachable: reachable.clone(),
            init_values: reachable.clone(),
            image: reachable,
            sorts: vec![ColSort::Int],
            safety_term: Vec::new(),
            init_member_terms: Vec::new(),
            closed_member_terms: Vec::new(),
            next_shape: None,
            next_completeness: None,
            init_shape: None,
            init_completeness: None,
            next_pred: None,
            next_general_completeness: None,
            init_pred: None,
            init_general_completeness: None,
            unbounded_invariant: None,
            safety_pred: None,
            safety_general: Some(vec![0]),
            init_member_reflected: None,
            closed_member_reflected: None,
            deadlock_free: None,
            deadlock_scan: Default::default(),
        };
        let config = crate::Config {
            init: Some("HCini".into()),
            next: Some("HCnxt".into()),
            invariants,
            ..Default::default()
        };
        let mut cert = crate::cert::build_explicit_fixpoint_certificate(src, &config, fp);
        cert.digest = cert.compute_digest();
        cert
    }

    /// POSITIVE: the genuine HourClock cert AST-direct-certifies all three legs with the
    /// expected leg sizes (R=12, D_init={0..12}=13, D_next={0..13}=14, pairs=12×14=168), and
    /// the recognized-IR lane CROSS-CHECKS to the same class (both certify).
    #[test]
    fn ast_direct_certifies_hourclock_and_crosscheck_agrees() {
        let cert = hourclock_full_cert();
        let (verdict, cc) = reflect_check_ast_direct_with_crosscheck(&cert);
        assert_eq!(
            verdict,
            AstDirectVerdict::Certified {
                states: 12,
                init_domain: 13,
                next_domain: 14,
                next_pairs: 168
            }
        );
        assert!(
            matches!(cc, AstCrossCheck::Agree { .. }),
            "recognized lane must agree on HourClock, got {cc:?}"
        );
    }

    /// ATTACK (violated safety): a broken HourClock variant — Next cycles at 13, the
    /// invariant still demands `hr ∈ 1..12`, R honestly includes [13] ⇒ the AST-direct
    /// safety leg reports NOT-SAFE at [13]. Never a certify.
    #[test]
    fn ast_direct_notsafe_on_violating_reachable_state() {
        const BROKEN: &str = "\
---------------------- MODULE HourClock ----------------------
EXTENDS Naturals
VARIABLE hr
HCini  ==  hr \\in (1 .. 12)
HCnxt  ==  hr' = IF hr # 13 THEN hr + 1 ELSE 1
==============================================================
";
        let r: Vec<Vec<u64>> = (1..=13u64).map(|h| vec![h]).collect();
        let cert = handmade_cert(BROKEN, vec!["HCini".into()], r);
        assert_eq!(
            reflect_check_safety_cert_ast_direct(&cert),
            AstDirectVerdict::NotSafe { state: vec![13] }
        );
    }

    /// ATTACK (non-closed R, init-shape): drop [1] from HourClock's R. [1] is BOTH an Init
    /// state and 12's successor; the Init leg (run first) fires ⇒ NOT-INIT-COMPLETE at [1].
    /// Definitive decline — never a certify.
    #[test]
    fn ast_direct_dropped_init_state_not_init_complete() {
        let mut cert = hourclock_full_cert();
        cert.explicit_fixpoint
            .as_mut()
            .unwrap()
            .reachable
            .retain(|t| t != &vec![1]);
        assert_eq!(
            reflect_check_safety_cert_ast_direct(&cert),
            AstDirectVerdict::NotInitComplete { s: vec![1] }
        );
    }

    /// ATTACK (non-closed R, closure-shape): a start-pinned variant (`Init == hr = 1`) whose
    /// R drops the NON-init state [7]: the safety + init legs pass, and the CLOSURE leg
    /// catches the dropped successor — NOT-CLOSED at (s=[6], sp=[7]). The kernel reduced
    /// `Next(6,7) ⇒ 7∈R` to Bool.false; a non-closed R must NEVER certify.
    #[test]
    fn ast_direct_dropped_successor_not_closed() {
        const START: &str = "\
---------------------- MODULE HourClock ----------------------
EXTENDS Naturals
VARIABLE hr
HCini  ==  hr = 1
HCnxt  ==  hr' = IF hr # 12 THEN hr + 1 ELSE 1
HCsafe ==  hr \\in (1 .. 12)
==============================================================
";
        let r: Vec<Vec<u64>> = (1..=12u64).filter(|h| *h != 7).map(|h| vec![h]).collect();
        let cert = handmade_cert(START, vec!["HCsafe".into()], r);
        assert_eq!(
            reflect_check_safety_cert_ast_direct(&cert),
            AstDirectVerdict::NotClosed {
                s: vec![6],
                sp: vec![7]
            }
        );
    }

    /// ATTACK (out-of-fragment): a spec whose Init uses CHOOSE — the AST-direct lane
    /// DECLINES (Inconclusive), never a wrong quote. (The recognized lane independently
    /// handles or declines it; the AST lane's answer must be a decline either way.)
    #[test]
    fn ast_direct_declines_out_of_fragment_spec() {
        const CHOOSY: &str = "\
---------------------- MODULE Choosy ----------------------
EXTENDS Naturals
VARIABLE hr
HCini  ==  hr = CHOOSE x \\in 1..12 : TRUE
HCnxt  ==  hr' = IF hr # 12 THEN hr + 1 ELSE 1
HCsafe ==  hr \\in (1 .. 12)
============================================================
";
        let r: Vec<Vec<u64>> = (1..=12u64).map(|h| vec![h]).collect();
        let cert = handmade_cert(CHOOSY, vec!["HCsafe".into()], r);
        match reflect_check_safety_cert_ast_direct(&cert) {
            AstDirectVerdict::Inconclusive(why) => {
                assert!(
                    why.contains("fragment"),
                    "decline reason should name the fragment, got: {why}"
                );
            }
            other => panic!("out-of-fragment spec must DECLINE, got {other:?}"),
        }
    }

    /// THE QUOTER-MIS-TRANSLATION ATTACK (the pivotal soundness demonstration): a mutated
    /// quoter that translates `<` as `≤` flips a NOT-SAFE spec (`hr < 12` violated at the
    /// reachable [12]) into an apparent certify — and the CROSS-CHECK catches it as a HARD
    /// Divergence, because the recognized-IR lane independently reduces the STORED
    /// `Lt(hr,12)` to Bool.false at [12]. The failure is LOUD, never silent.
    #[test]
    fn attack_mis_translation_lt_as_leq_caught_by_crosscheck() {
        const LT_SRC: &str = "\
---------------------- MODULE HourClockLt ----------------------
EXTENDS Naturals
VARIABLE hr
HCini  ==  hr \\in (1 .. 12)
HCnxt  ==  hr' = IF hr # 12 THEN hr + 1 ELSE 1
HCsafe ==  hr < 12
================================================================
";
        let r: Vec<Vec<u64>> = (1..=12u64).map(|h| vec![h]).collect();
        // Certifier-real general legs (minted under the SAFE `HCini` invariant, which is how a
        // genuine cert for this spec exists at all), then the stored invariant swapped to the
        // spec's own `hr < 12` — so the recognized lane discharges the REAL `Lt` IR and its
        // safety leg conclusively refuses at [12]. The AST lane reads `HCsafe` from the spec.
        let fp = {
            let mint_cfg = crate::Config {
                init: Some("HCini".into()),
                next: Some("HCnxt".into()),
                invariants: vec!["HCini".into()],
                ..Default::default()
            };
            let mut fp = certify_explicit_state_spec(LT_SRC, &mint_cfg)
                .expect("the Lt variant certifies under the safe HCini invariant");
            assert!(
                fp.init_pred.is_some() && fp.next_pred.is_some(),
                "general legs present"
            );
            fp.safety_pred = Some(PredIR::Lt(ValIR::Var(0), ValIR::Lit(12)));
            fp
        };
        let cfg = crate::Config {
            init: Some("HCini".into()),
            next: Some("HCnxt".into()),
            invariants: vec!["HCsafe".into()],
            ..Default::default()
        };
        let mut cert = crate::cert::build_explicit_fixpoint_certificate(LT_SRC, &cfg, fp);
        cert.digest = cert.compute_digest();

        // (a) The GENUINE quoter: NOT-SAFE at [12] (12 < 12 kernel-reduces to Bool.false) —
        // and the recognized lane agrees (both refuse), so the classes are consistent.
        let (genuine, cc_genuine) = reflect_check_ast_direct_with_crosscheck(&cert);
        assert_eq!(genuine, AstDirectVerdict::NotSafe { state: vec![12] });
        assert!(
            matches!(cc_genuine, AstCrossCheck::Agree { .. }),
            "genuine quoter and recognized lane must both refuse, got {cc_genuine:?}"
        );

        // (b) The MUTATED quoter: `Lt ↦ leq` — `deep::leq(var 0, lit 12)` is EXACTLY what a
        // quoter whose Lt arm was mis-edited to `deep::leq` would emit for `hr < 12`. The
        // three legs then (wrongly) all pass…
        let mut prep = prepare_ast_direct(&cert).expect("in-fragment spec must prepare");
        prep.q_safety = deep::leq(deep::var(0), deep::lit(12));
        let doctored = run_ast_direct_legs(&prep, &r);
        assert!(
            matches!(doctored, AstDirectVerdict::Certified { .. }),
            "the mis-quote flips the verdict (that is the attack), got {doctored:?}"
        );
        // …and the CROSS-CHECK catches the contradiction as a HARD Divergence: the
        // recognized-IR lane reduces the stored `Lt` to NOT-SAFE at [12].
        let recognized = crate::reflect_safety_check::reflect_check_safety_cert_full(&cert, false);
        match classify_crosscheck(&doctored, &recognized) {
            AstCrossCheck::Divergence { ast, recognized } => {
                assert!(ast.contains("Certified"), "ast side: {ast}");
                assert!(
                    recognized.contains("NotSafe"),
                    "recognized side: {recognized}"
                );
            }
            other => panic!(
                "a mis-translated quote MUST be a loud Divergence, got {other:?} — silent \
                 acceptance would be a soundness hole"
            ),
        }
    }

    /// THE QUOTER-MIS-TRANSLATION ATTACK, second failure mode ("or the kernel legs fail"): a
    /// mutated quoter that translates `hr + 1` as `hr + 2` on the GENUINE (safe) HourClock
    /// makes the closure leg itself refuse — the kernel reduces the mis-quoted
    /// `Next(11,13) ⇒ 13∈R` to Bool.false ⇒ NOT-CLOSED (and would also diverge from the
    /// recognized lane's Certified). Loud either way.
    #[test]
    fn attack_mis_translation_add_constant_fails_closure_leg() {
        let cert = hourclock_full_cert();
        let r = cert.explicit_fixpoint.as_ref().unwrap().reachable.clone();
        let mut prep = prepare_ast_direct(&cert).expect("HourClock must prepare");
        // `deep::eq_ite(prime 0, neq(var 0, lit 12), add(var 0, lit 2), lit 1)` is exactly the
        // output of a quoter whose Add arm mis-reads the literal (models any off-by-one /
        // wrong-op mis-translation of the update).
        prep.q_next = deep::eq_ite(
            deep::prime(0),
            deep::neq(deep::var(0), deep::lit(12)),
            deep::add(deep::var(0), deep::lit(2)),
            deep::lit(1),
        );
        let doctored = run_ast_direct_legs(&prep, &r);
        assert_eq!(
            doctored,
            AstDirectVerdict::NotClosed {
                s: vec![11],
                sp: vec![13]
            },
            "the mis-quoted Next must fail the closure leg LOUDLY"
        );
        let recognized = crate::reflect_safety_check::reflect_check_safety_cert_full(&cert, false);
        assert!(
            matches!(
                classify_crosscheck(&doctored, &recognized),
                AstCrossCheck::Divergence { .. }
            ),
            "…and the cross-check flags the contradiction with the recognized lane's Certified"
        );
    }

    /// Cross-check classifier edges: recognized-Inconclusive ⇒ Unavailable (surfaced, not an
    /// error); AST-Inconclusive ⇒ Unavailable (the lane declined; never a divergence);
    /// both-refuse with DIFFERENT classes ⇒ Agree (no lane certifies what the other refutes).
    #[test]
    fn crosscheck_classifier_edges() {
        use crate::reflect_safety_check::ReflectFullVerdict as R;
        let cert_verdict = AstDirectVerdict::Certified {
            states: 1,
            init_domain: 1,
            next_domain: 1,
            next_pairs: 1,
        };
        assert!(matches!(
            classify_crosscheck(&cert_verdict, &R::Inconclusive("out of IR fragment".into())),
            AstCrossCheck::Unavailable { .. }
        ));
        assert!(matches!(
            classify_crosscheck(
                &AstDirectVerdict::Inconclusive("out of AST fragment".into()),
                &R::Certified {
                    states: 1,
                    init_domain: 1,
                    next_domain: 1,
                    next_pairs: 1,
                    coverage:
                        crate::reflect_safety_check::ReflectCoverageBasis::ConstructionComplete,
                },
            ),
            AstCrossCheck::Unavailable { .. }
        ));
        assert!(matches!(
            classify_crosscheck(
                &AstDirectVerdict::NotSafe { state: vec![3] },
                &R::NotClosed {
                    s: vec![1],
                    sp: vec![2]
                },
            ),
            AstCrossCheck::Agree { .. }
        ));
        assert!(matches!(
            classify_crosscheck(
                &AstDirectVerdict::NotSafe { state: vec![3] },
                &R::Certified {
                    states: 1,
                    init_domain: 1,
                    next_domain: 1,
                    next_pairs: 1,
                    coverage:
                        crate::reflect_safety_check::ReflectCoverageBasis::ConstructionComplete,
                },
            ),
            AstCrossCheck::Divergence { .. }
        ));
    }

    /// The AST-quoted HourClock predicates kernel-evaluate correctly straight from the PARSED
    /// spec (semantic spot-checks of the whole front-end + quoter + kernel pipeline, both
    /// truth directions).
    #[test]
    fn quoted_hourclock_predicates_kernel_evaluate() {
        let cert = hourclock_full_cert();
        let prep = prepare_ast_direct(&cert).expect("HourClock must prepare");
        // Safety `hr ∈ 1..12`.
        assert_eq!(
            kernel_eval_quoted_pred(&prep.q_safety, &[1], &[1]),
            Some(true)
        );
        assert_eq!(
            kernel_eval_quoted_pred(&prep.q_safety, &[12], &[12]),
            Some(true)
        );
        assert_eq!(
            kernel_eval_quoted_pred(&prep.q_safety, &[0], &[0]),
            Some(false)
        );
        assert_eq!(
            kernel_eval_quoted_pred(&prep.q_safety, &[13], &[13]),
            Some(false)
        );
        // Next `hr' = IF hr # 12 THEN hr + 1 ELSE 1`.
        assert_eq!(
            kernel_eval_quoted_pred(&prep.q_next, &[5], &[6]),
            Some(true)
        );
        assert_eq!(
            kernel_eval_quoted_pred(&prep.q_next, &[5], &[5]),
            Some(false)
        );
        assert_eq!(
            kernel_eval_quoted_pred(&prep.q_next, &[12], &[1]),
            Some(true)
        );
        assert_eq!(
            kernel_eval_quoted_pred(&prep.q_next, &[12], &[13]),
            Some(false)
        );
    }

    // ── Increment 2: UNCHANGED + disjunctive Next (multi-var Int) ─────────────

    /// The `UNCHANGED` quote: `UNCHANGED x` expands to the primed-equality `deep::eq(prime i,
    /// var i)` (reusing this lane's already-tested `eq`/`prime`/`var` ctors — same kernel normal
    /// form `Nat.beq (nth sp i) (nth s i)` as the embedding's `unchanged` ctor), and
    /// `UNCHANGED <<x,y>>` to the left-nested And-fold of per-column equalities. Fail-closed on
    /// a state predicate (no primes), non-variable / non-tuple-of-variables operands, an empty
    /// tuple, and an unknown variable.
    #[test]
    fn quoter_unchanged_expands_to_eq_conjunction() {
        let unch =
            |e: AstExpr| quote_pred_ast(&AstExpr::Unchanged(sp(e)), VARS, SORTS, MAX_R, true);
        assert_eq!(
            unch(ident("hr")),
            Some(deep::eq(deep::prime(0), deep::var(0)))
        );
        assert_eq!(
            unch(ident("m")),
            Some(deep::eq(deep::prime(1), deep::var(1)))
        );
        // A pre-resolved `StateVar` operand maps BY NAME (its embedded index is ignored).
        assert_eq!(
            unch(AstExpr::StateVar("m".into(), 9, NameId::INVALID)),
            Some(deep::eq(deep::prime(1), deep::var(1)))
        );
        // `UNCHANGED <<hr, m>>` ≡ hr'=hr ∧ m'=m — left-nested And-fold.
        let tup =
            |vs: &[&str]| AstExpr::Tuple(vs.iter().map(|v| Spanned::dummy(ident(v))).collect());
        assert_eq!(
            unch(tup(&["hr", "m"])),
            Some(deep::and(
                deep::eq(deep::prime(0), deep::var(0)),
                deep::eq(deep::prime(1), deep::var(1)),
            ))
        );
        // FAIL-CLOSED edges.
        assert_eq!(
            quote_pred_ast(
                &AstExpr::Unchanged(sp(ident("hr"))),
                VARS,
                SORTS,
                MAX_R,
                false
            ),
            None
        ); // state pred
        assert_eq!(unch(int(3)), None); // non-variable operand
        assert_eq!(unch(AstExpr::Add(sp(ident("hr")), sp(int(1)))), None); // compound operand
        assert_eq!(
            unch(AstExpr::Tuple(vec![
                Spanned::dummy(ident("hr")),
                Spanned::dummy(int(2))
            ])),
            None
        ); // a non-variable tuple element poisons the whole quote
        assert_eq!(unch(AstExpr::Tuple(vec![])), None); // empty tuple (degenerate)
        assert_eq!(unch(ident("zzz")), None); // unknown variable
    }

    /// The `UNCHANGED` NEXT-domain pin: `UNCHANGED x_i` pins column i to `max_r[i]` (the frozen
    /// successor equals the current value, whose column max over R is `max_r[i]`); it pins ONLY
    /// the named column(s). `unchanged_cols` is fail-closed identically to the quoter.
    #[test]
    fn unchanged_pins_next_column_to_max_r() {
        let vars = &["x", "y"];
        // Flatten `e`'s conjuncts and take the column-i Next bound over max_r=[2,5].
        let bound = |e: &Spanned<AstExpr>, i: u64| {
            let mut cs = Vec::new();
            flatten_and(e, &mut cs);
            ast_next_col_bound(&cs, i, vars, SORTS, &[2, 5])
        };
        // UNCHANGED y pins column 1 to max_r[1]=5; it does NOT pin column 0.
        let unch_y = Spanned::dummy(AstExpr::Unchanged(sp(ident("y"))));
        assert_eq!(bound(&unch_y, 1), Some(5));
        assert_eq!(bound(&unch_y, 0), None);
        // UNCHANGED <<x,y>> pins BOTH to their column maxes.
        let unch_xy = Spanned::dummy(AstExpr::Unchanged(sp(AstExpr::Tuple(vec![
            Spanned::dummy(ident("x")),
            Spanned::dummy(ident("y")),
        ]))));
        assert_eq!(bound(&unch_xy, 0), Some(2));
        assert_eq!(bound(&unch_xy, 1), Some(5));
        // `unchanged_cols` fail-closed.
        assert_eq!(unchanged_cols(&ident("x"), vars), Some(vec![0]));
        assert_eq!(unchanged_cols(&int(3), vars), None);
        assert_eq!(unchanged_cols(&AstExpr::Tuple(vec![]), vars), None);
    }

    /// The disjunctive-Next domain rule: `flatten_or` splits top-level disjuncts, and each
    /// column's OVERALL bound is the MAX over disjuncts of its per-disjunct pin (a MAX
    /// UNDER-covers nothing; a MIN would be unsound). Here
    /// `∨ (x'=x+1 ∧ UNCHANGED y) ∨ (y'=y+1 ∧ UNCHANGED x)` over max_r=[2,2]: per-disjunct
    /// bounds x=(3,2), y=(2,3); MAX ⇒ x=3, y=3.
    #[test]
    fn disjunctive_next_domain_max_combines_per_disjunct_bounds() {
        let vars = &["x", "y"];
        let max_r = &[2u64, 2u64];
        let d1 = AstExpr::And(
            sp(AstExpr::Eq(
                sp(prime("x")),
                sp(AstExpr::Add(sp(ident("x")), sp(int(1)))),
            )),
            sp(AstExpr::Unchanged(sp(ident("y")))),
        );
        let d2 = AstExpr::And(
            sp(AstExpr::Eq(
                sp(prime("y")),
                sp(AstExpr::Add(sp(ident("y")), sp(int(1)))),
            )),
            sp(AstExpr::Unchanged(sp(ident("x")))),
        );
        let next = Spanned::dummy(AstExpr::Or(sp(d1), sp(d2)));
        let mut disj = Vec::new();
        flatten_or(&next, &mut disj);
        assert_eq!(disj.len(), 2, "top-level Or splits into 2 disjuncts");
        let bound = |d: &Spanned<AstExpr>, i: u64| {
            let mut cs = Vec::new();
            flatten_and(d, &mut cs);
            ast_next_col_bound(&cs, i, vars, SORTS, max_r)
        };
        // disjunct1: x'=x+1 ⇒ 2+1=3 ; UNCHANGED y ⇒ max_r[1]=2
        assert_eq!(bound(disj[0], 0), Some(3));
        assert_eq!(bound(disj[0], 1), Some(2));
        // disjunct2: UNCHANGED x ⇒ max_r[0]=2 ; y'=y+1 ⇒ 3
        assert_eq!(bound(disj[1], 0), Some(2));
        assert_eq!(bound(disj[1], 1), Some(3));
        // MAX-combine (the D_next axis bounds): x=max(3,2)=3, y=max(2,3)=3.
        let col_max = |i: u64| disj.iter().map(|d| bound(*d, i).unwrap()).max().unwrap();
        assert_eq!(col_max(0), 3);
        assert_eq!(col_max(1), 3);
        // A plain conjunctive Next is a SINGLE disjunct (flatten_or is a no-op on non-Or) — so
        // HourClock's conjunctive derivation is byte-identical to the pre-disjunctive path.
        let conj_only = Spanned::dummy(AstExpr::And(
            sp(AstExpr::Eq(sp(prime("x")), sp(ident("x")))),
            sp(AstExpr::Eq(sp(prime("y")), sp(ident("y")))),
        ));
        let mut one = Vec::new();
        flatten_or(&conj_only, &mut one);
        assert_eq!(one.len(), 1);
    }

    /// A handcrafted 2-column explicit-state cert (2 Int columns). Like [`handmade_cert`] the
    /// AST-direct lane reads only `spec_src` + config names + `reachable`/`sorts`.
    fn handmade_cert_2col(
        src: &str,
        init: &str,
        next: &str,
        invariants: Vec<String>,
        reachable: Vec<Vec<u64>>,
    ) -> SafetyCertificate {
        let fp = ExplicitFixpointCert {
            reachable: reachable.clone(),
            init_values: reachable.clone(),
            image: reachable,
            sorts: vec![ColSort::Int, ColSort::Int],
            safety_term: Vec::new(),
            init_member_terms: Vec::new(),
            closed_member_terms: Vec::new(),
            next_shape: None,
            next_completeness: None,
            init_shape: None,
            init_completeness: None,
            next_pred: None,
            next_general_completeness: None,
            init_pred: None,
            init_general_completeness: None,
            unbounded_invariant: None,
            safety_pred: None,
            safety_general: Some(vec![0]),
            init_member_reflected: None,
            closed_member_reflected: None,
            deadlock_free: None,
            deadlock_scan: Default::default(),
        };
        let config = crate::Config {
            init: Some(init.into()),
            next: Some(next.into()),
            invariants,
            ..Default::default()
        };
        let mut cert = crate::cert::build_explicit_fixpoint_certificate(src, &config, fp);
        cert.digest = cert.compute_digest();
        cert
    }

    /// THE DECISIVE SOUNDNESS TEST (ATTACK 1 — unpinned disjunct DECLINES): Next's 2nd disjunct
    /// `x'=x` leaves `y'` UNCONSTRAINED (no `y'=…` Eq-pin, no `UNCHANGED y`), so a successor
    /// from that disjunct could carry ANY `y'` — escaping every product `D_next`. The domain
    /// rule MUST DECLINE (Inconclusive), never certify: MAX-combining over only the *pinning*
    /// disjunct would UNDER-cover column y and pass closure vacuously on the missing successors
    /// ⇒ a false safe. (A real certifier cannot even finitely enumerate this spec — the
    /// unconstrained `y'` blows the state cap — so this handcrafted cert models the fail-closed
    /// guarantee directly at the domain rule.)
    #[test]
    fn ast_direct_unpinned_disjunct_declines() {
        const SRC: &str = "\
---------------------- MODULE Unpinned ----------------------
EXTENDS Naturals
VARIABLES x, y
UInit == x = 0 /\\ y = 0
UNext == \\/ (x' = x + 1 /\\ y' = y)
         \\/ (x' = x)
UInv  == x \\in (0 .. 9) /\\ y \\in (0 .. 9)
============================================================
";
        let r: Vec<Vec<u64>> = vec![vec![0, 0], vec![1, 0]];
        let cert = handmade_cert_2col(SRC, "UInit", "UNext", vec!["UInv".into()], r);
        match reflect_check_safety_cert_ast_direct(&cert) {
            AstDirectVerdict::Inconclusive(why) => assert!(
                why.contains("does not pin column"),
                "the decline must name the unpinned column, got: {why}"
            ),
            other => panic!(
                "an unpinned disjunct MUST DECLINE (a MAX over only the pinning disjunct would \
                 under-cover ⇒ false safe), got {other:?}"
            ),
        }
    }

    /// INCREMENT 5 — LANE-LEVEL ATTACK 1 (truncation ⇒ FALSE SAFE if admitted): a spec whose
    /// Next contains `y' = 3 - x` where the stored `R` lets `x` reach 5 (`max_r[x] = 5 > 3`).
    /// TLA+ `3 - 5 = -2`; the kernel's `Nat.sub 3 5 = 0`. If the quoter admitted this, the kernel
    /// would compute `y' = 0` where TLA+ has `-2` — masking a violated `y ≥ 0` invariant. The AST
    /// lane MUST DECLINE (`Next body outside the AST-direct fragment`), NEVER certify. Compare the
    /// twin below where `x ≤ 3` and the SAME `3 - x` construct is admitted — proving the gate is
    /// EXACTLY `x ≤ c`, not a blanket refusal of subtraction.
    #[test]
    fn ast_direct_truncating_subtraction_declines_lane_level() {
        const SRC: &str = "\
---------------------- MODULE Trunc ----------------------
EXTENDS Naturals
VARIABLES x, y
TInit == x \\in (0 .. 5) /\\ y = 0
TNext == x' = x /\\ y' = 3 - x
TSafe == y \\in (0 .. 3)
==========================================================
";
        // `R` reaches x=5, so max_r[x]=5 > 3 ⇒ `3 - x` could truncate ⇒ the quoter DECLINES.
        let r: Vec<Vec<u64>> = (0..=5u64).map(|x| vec![x, 0]).collect();
        let cert = handmade_cert_2col(SRC, "TInit", "TNext", vec!["TSafe".into()], r);
        match reflect_check_safety_cert_ast_direct(&cert) {
            AstDirectVerdict::Inconclusive(why) => assert!(
                why.contains("Next body outside") || why.contains("fragment"),
                "the truncation decline must be a fragment decline, got: {why}"
            ),
            other => panic!(
                "a `3 - x` with x reaching 5 (bound > 3) MUST DECLINE (Nat.sub would truncate to 0 \
                 where TLA+ goes negative — a false safe), got {other:?}"
            ),
        }
    }

    /// INCREMENT 5 — LANE-LEVEL POSITIVE (the admitted twin of the attack above): the SAME
    /// `c - x` construct CERTIFIES end-to-end when the column is proved `≤ c`. A boolean toggle
    /// `x' = 1 - x` over `x ∈ {0,1}` (max_r[x]=1 ≤ 1): all three kernel legs discharge. This is
    /// the recognizer-free, self-contained proof that bounded subtraction is SOUND and COMPLETE
    /// for its construct (the corpus spec AsynchInterface uses exactly this `1 - x` shape).
    #[test]
    fn ast_direct_certifies_bounded_subtraction_toggle() {
        const SRC: &str = "\
---------------------- MODULE SubToggle ----------------------
EXTENDS Naturals
VARIABLES x, y
SInit == x = 0 /\\ y = 0
SNext == x' = 1 - x /\\ y' = y
SInv  == x \\in (0 .. 1) /\\ y \\in (0 .. 0)
=============================================================
";
        // R = {(0,0),(1,0)} — x toggles 0↔1, y frozen. max_r=[1,0]; `1 - x` admits (1 ≤ 1).
        let r: Vec<Vec<u64>> = vec![vec![0, 0], vec![1, 0]];
        let cert = handmade_cert_2col(SRC, "SInit", "SNext", vec!["SInv".into()], r);
        assert_eq!(
            reflect_check_safety_cert_ast_direct(&cert),
            AstDirectVerdict::Certified {
                states: 2,
                init_domain: 1, // D_init: x pinned to 0, y pinned to 0 ⇒ {(0,0)}
                next_domain: 2, // D_next: x' bound ub(1-x)=1, y' bound max_r[1]=0 ⇒ {(0,0),(1,0)}
                next_pairs: 4,  // |R| × |D_next| = 2 × 2
            }
        );
    }

    // ── End-to-end over a REAL disjunctive-UNCHANGED cert ─────────────────────

    const TWOCOUNTER_SRC: &str = "\
---------------------- MODULE TwoCounter ----------------------
EXTENDS Naturals
VARIABLES x, y
TC_Init == x = 0 /\\ y = 0
TC_Next == \\/ (x < 2 /\\ x' = x + 1 /\\ UNCHANGED y)
           \\/ (y < 2 /\\ y' = y + 1 /\\ UNCHANGED x)
TC_Inv  == x \\in (0 .. 2) /\\ y \\in (0 .. 2)
==============================================================
";

    fn twocounter_cfg() -> crate::Config {
        crate::Config {
            init: Some("TC_Init".into()),
            next: Some("TC_Next".into()),
            invariants: vec!["TC_Inv".into()],
            ..Default::default()
        }
    }

    fn twocounter_full_cert() -> SafetyCertificate {
        let fp = certify_explicit_state_spec(TWOCOUNTER_SRC, &twocounter_cfg())
            .expect("TwoCounter must explicit-state certify");
        let mut cert =
            crate::cert::build_explicit_fixpoint_certificate(TWOCOUNTER_SRC, &twocounter_cfg(), fp);
        cert.digest = cert.compute_digest();
        cert
    }

    /// POSITIVE (increment 2): a genuine 2-var Int cert whose Next is DISJUNCTIVE with UNCHANGED
    /// in each arm AST-direct-certifies all three legs (R=9; D_init={(0,0)}=1;
    /// D_next={0..3}×{0..3}=16 from the MAX-combined bounds [3,3]; pairs=9×16=144), and the
    /// recognized-IR lane cross-checks to the same class.
    #[test]
    fn ast_direct_certifies_twocounter_unchanged_disjunctive() {
        let cert = twocounter_full_cert();
        let (verdict, cc) = reflect_check_ast_direct_with_crosscheck(&cert);
        assert_eq!(
            verdict,
            AstDirectVerdict::Certified {
                states: 9,
                init_domain: 1,
                next_domain: 16,
                next_pairs: 144
            }
        );
        assert!(
            matches!(cc, AstCrossCheck::Agree { .. }),
            "recognized lane must agree on TwoCounter, got {cc:?}"
        );
    }

    /// ATTACK 2 (tamper ⇒ NOT-CLOSED over the disjunctive spec): drop the reachable successor
    /// [1,1] (produced by disjunct 1 from [0,1] and disjunct 2 from [1,0]). The kernel closure
    /// leg reduces `Next([0,1],[1,1]) ⇒ [1,1]∈R` to Bool.false ⇒ NOT-CLOSED — proving the
    /// disjunctive `D_next` actually COVERS that successor (had it under-covered, the leg would
    /// pass vacuously and certify a non-closed R: a false safe).
    #[test]
    fn ast_direct_twocounter_dropped_successor_not_closed() {
        let mut cert = twocounter_full_cert();
        cert.explicit_fixpoint
            .as_mut()
            .unwrap()
            .reachable
            .retain(|t| t != &vec![1, 1]);
        assert_eq!(
            reflect_check_safety_cert_ast_direct(&cert),
            AstDirectVerdict::NotClosed {
                s: vec![0, 1],
                sp: vec![1, 1]
            }
        );
    }

    /// ATTACK 3 (reachably-violated invariant ⇒ NOT-SAFE): a broken TwoCounter whose `x` guard
    /// allows `x` up to 3 while the invariant still demands `x ∈ 0..2`; an honest R that
    /// includes [3,0] ⇒ the AST-direct safety leg reduces the invariant to Bool.false at [3,0].
    /// Never a certify. (`ty check` independently reports the same violation — CLI parity.)
    #[test]
    fn ast_direct_notsafe_on_violating_reachable_state_multivar() {
        const BROKEN: &str = "\
---------------------- MODULE Unsafe ----------------------
EXTENDS Naturals
VARIABLES x, y
UInit == x = 0 /\\ y = 0
UNext == \\/ (x < 3 /\\ x' = x + 1 /\\ UNCHANGED y)
         \\/ (y < 2 /\\ y' = y + 1 /\\ UNCHANGED x)
UInv  == x \\in (0 .. 2) /\\ y \\in (0 .. 2)
==========================================================
";
        // An honest reachable set that includes the invariant-violating successor [3,0].
        let r: Vec<Vec<u64>> = vec![vec![0, 0], vec![1, 0], vec![2, 0], vec![3, 0]];
        let cert = handmade_cert_2col(BROKEN, "UInit", "UNext", vec!["UInv".into()], r);
        assert_eq!(
            reflect_check_safety_cert_ast_direct(&cert),
            AstDirectVerdict::NotSafe { state: vec![3, 0] }
        );
    }

    // ── Increment 3: finite-set membership + enum/model-value atoms ────────────

    fn string(s: &str) -> AstExpr {
        AstExpr::String(s.to_string())
    }
    fn setenum(elems: Vec<AstExpr>) -> AstExpr {
        AstExpr::SetEnum(elems.into_iter().map(Spanned::dummy).collect())
    }
    fn enum_model(labels: &[&str]) -> ColSort {
        ColSort::Enum {
            labels: labels.iter().map(|s| s.to_string()).collect(),
            kind: EnumKind::Model,
        }
    }

    /// The set-membership DESUGAR: `x ∈ {v1,…,vn}` (a LITERAL set over an Int column) quotes to the
    /// left-nested `Or`-fold of equalities `(x=v1) ∨ … ∨ (x=vn)` (the Aristotle `mem_iff_any_eq`),
    /// reusing only `Or`/`Eq`/`var`/`prime`/`lit`. Fail-closed on an empty set, a negative element,
    /// a non-literal domain, and (state pred) a primed column.
    #[test]
    fn quoter_setenum_membership_desugars_to_or_fold() {
        let qp = |e: &AstExpr| quote_pred_ast(e, VARS, SORTS, MAX_R, true);
        let mem = |x: &str, elems: Vec<AstExpr>| AstExpr::In(sp(ident(x)), sp(setenum(elems)));
        // Singleton ⇒ a bare equality (no Or).
        assert_eq!(
            qp(&mem("hr", vec![int(1)])),
            Some(deep::eq(deep::var(0), deep::lit(1)))
        );
        // Two elements ⇒ one Or.
        assert_eq!(
            qp(&mem("hr", vec![int(1), int(3)])),
            Some(deep::or(
                deep::eq(deep::var(0), deep::lit(1)),
                deep::eq(deep::var(0), deep::lit(3)),
            ))
        );
        // Three ⇒ LEFT-nested Or-fold.
        assert_eq!(
            qp(&mem("m", vec![int(0), int(2), int(4)])),
            Some(deep::or(
                deep::or(
                    deep::eq(deep::var(1), deep::lit(0)),
                    deep::eq(deep::var(1), deep::lit(2)),
                ),
                deep::eq(deep::var(1), deep::lit(4)),
            ))
        );
        // A PRIMED column in Next: `m' ∈ {2}` ⇒ `prime 1 = 2`.
        assert_eq!(
            qp(&AstExpr::In(sp(prime("m")), sp(setenum(vec![int(2)])))),
            Some(deep::eq(deep::prime(1), deep::lit(2)))
        );
        // FAIL-CLOSED: empty set, negative element, non-literal domain, primed in a STATE pred.
        assert_eq!(qp(&mem("hr", vec![])), None); // `hr ∈ {}` ≡ FALSE ⇒ decline
        assert_eq!(qp(&mem("hr", vec![int(0), int(-1)])), None); // negative element
        assert_eq!(qp(&AstExpr::In(sp(ident("hr")), sp(ident("m")))), None); // `hr ∈ m` (var domain)
        assert_eq!(
            quote_pred_ast(
                &AstExpr::In(sp(prime("m")), sp(setenum(vec![int(2)]))),
                VARS,
                SORTS,
                MAX_R,
                false // state pred: a primed column declines
            ),
            None
        );
    }

    /// ENUM / model-value equality + membership resolve a label to its CODE (index in the column's
    /// `labels`), KIND-guarded and column-LOCAL. Plus the two decline attacks: an UNKNOWN label
    /// (ATTACK 1) and a WRONG-COLUMN label (ATTACK 2) both DECLINE — never a wrong code.
    #[test]
    fn quoter_enum_label_resolution_and_decline_attacks() {
        // mode/mode2 : {busy,idle} (busy=0, idle=1) ; state : {off,on} (off=0, on=1).
        let vars: &[&str] = &["mode", "state", "mode2"];
        let sorts: &[ColSort] = &[
            enum_model(&["busy", "idle"]),
            enum_model(&["off", "on"]),
            enum_model(&["busy", "idle"]),
        ];
        let max_r: &[u64] = &[1, 1, 1]; // enum tests: subtraction never fires; length matches cols
        let q = |e: &AstExpr| quote_pred_ast(e, vars, sorts, max_r, true);
        let eq = |a: AstExpr, b: AstExpr| AstExpr::Eq(sp(a), sp(b));
        // Label resolved to its code, either orientation.
        assert_eq!(
            q(&eq(ident("mode"), ident("idle"))),
            Some(deep::eq(deep::var(0), deep::lit(1)))
        );
        assert_eq!(
            q(&eq(ident("mode"), ident("busy"))),
            Some(deep::eq(deep::var(0), deep::lit(0)))
        );
        assert_eq!(
            q(&eq(ident("busy"), ident("mode"))),
            Some(deep::eq(deep::lit(0), deep::var(0)))
        );
        assert_eq!(
            q(&eq(ident("state"), ident("on"))),
            Some(deep::eq(deep::var(1), deep::lit(1)))
        );
        // `≠` too.
        assert_eq!(
            q(&AstExpr::Neq(sp(ident("mode")), sp(ident("idle")))),
            Some(deep::neq(deep::var(0), deep::lit(1)))
        );
        // Enum membership over the full label set ⇒ Or-fold of code equalities.
        assert_eq!(
            q(&AstExpr::In(
                sp(ident("mode")),
                sp(setenum(vec![ident("busy"), ident("idle")]))
            )),
            Some(deep::or(
                deep::eq(deep::var(0), deep::lit(0)),
                deep::eq(deep::var(0), deep::lit(1))
            ))
        );
        // Two enum columns of the SAME sort compare index-exactly; DIFFERENT sorts decline.
        assert_eq!(
            q(&eq(ident("mode"), ident("mode2"))),
            Some(deep::eq(deep::var(0), deep::var(2)))
        );
        assert_eq!(q(&eq(ident("mode"), ident("state"))), None); // cross-sort enum=enum
                                                                 // ATTACK 1 (unknown label): `mode = ghost` / `mode ∈ {idle, ghost}` ⇒ DECLINE.
        assert_eq!(q(&eq(ident("mode"), ident("ghost"))), None);
        assert_eq!(
            q(&AstExpr::In(
                sp(ident("mode")),
                sp(setenum(vec![ident("idle"), ident("ghost")]))
            )),
            None
        );
        // ATTACK 2 (wrong-column label): `on` is `state`'s label, NOT `mode`'s ⇒ DECLINE.
        assert_eq!(q(&eq(ident("mode"), ident("on"))), None);
        // An enum column compared to an Int literal is a category error ⇒ DECLINE (never a code).
        assert_eq!(q(&eq(ident("mode"), int(0))), None);
        // An enum-column lhs with an Int-branch `IF` rhs ⇒ DECLINE (the eq_ite composite is
        // Int-value-pinned; comparing a code to an Int conditional would be unsound).
        assert_eq!(
            q(&eq(
                ident("mode"),
                AstExpr::If(sp(AstExpr::Bool(true)), sp(int(0)), sp(int(1)))
            )),
            None
        );

        // KIND guard: a `Str` column resolves a String literal, NOT a model-value Ident.
        let svars: &[&str] = &["p"];
        let ssorts: &[ColSort] = &[ColSort::Enum {
            labels: vec!["a".into(), "b".into()],
            kind: EnumKind::Str,
        }];
        let smax_r: &[u64] = &[1];
        let sq = |e: &AstExpr| quote_pred_ast(e, svars, ssorts, smax_r, true);
        assert_eq!(
            sq(&eq(ident("p"), string("a"))),
            Some(deep::eq(deep::var(0), deep::lit(0)))
        );
        assert_eq!(
            sq(&eq(ident("p"), string("b"))),
            Some(deep::eq(deep::var(0), deep::lit(1)))
        );
        assert_eq!(sq(&eq(ident("p"), ident("a"))), None); // Str col + Ident ⇒ kind mismatch
        assert_eq!(sq(&eq(ident("p"), string("z"))), None); // unknown String label
        assert_eq!(
            sq(&AstExpr::In(
                sp(ident("p")),
                sp(setenum(vec![string("a"), string("b")]))
            )),
            Some(deep::or(
                deep::eq(deep::var(0), deep::lit(0)),
                deep::eq(deep::var(0), deep::lit(1))
            ))
        );
    }

    /// The INIT/NEXT bound rules over enum + literal-set pins: an enum `= label` / `∈ {labels}`
    /// pins to the label code(s); an Int `∈ {ints}` to the max literal; a primed `∈ {…}`/`lo..hi`
    /// bounds the non-deterministic successor. An unknown label yields NO bound (⇒ decline).
    #[test]
    fn enum_and_setenum_bound_rules() {
        let vars: &[&str] = &["mode", "step"];
        let sorts: &[ColSort] = &[enum_model(&["busy", "idle"]), ColSort::Int];
        let init_bound = |e: &Spanned<AstExpr>, i: u64| {
            let mut cs = Vec::new();
            flatten_and(e, &mut cs);
            ast_init_col_bound(&cs, i, vars, sorts)
        };
        let next_bound = |e: &Spanned<AstExpr>, i: u64| {
            let mut cs = Vec::new();
            flatten_and(e, &mut cs);
            ast_next_col_bound(&cs, i, vars, sorts, &[1, 1])
        };
        // Init enum `mode ∈ {busy,idle}` ⇒ max(0,1)=1 ; `mode = idle` ⇒ 1 ; `mode = busy` ⇒ 0.
        let d = Spanned::dummy;
        assert_eq!(
            init_bound(
                &d(AstExpr::In(
                    sp(ident("mode")),
                    sp(setenum(vec![ident("busy"), ident("idle")]))
                )),
                0
            ),
            Some(1)
        );
        assert_eq!(
            init_bound(&d(AstExpr::Eq(sp(ident("mode")), sp(ident("idle")))), 0),
            Some(1)
        );
        assert_eq!(
            init_bound(&d(AstExpr::Eq(sp(ident("mode")), sp(ident("busy")))), 0),
            Some(0)
        );
        // Int literal set `step ∈ {0,1}` ⇒ 1 ; `step ∈ {2,5,3}` ⇒ 5.
        assert_eq!(
            init_bound(
                &d(AstExpr::In(
                    sp(ident("step")),
                    sp(setenum(vec![int(0), int(1)]))
                )),
                1
            ),
            Some(1)
        );
        assert_eq!(
            init_bound(
                &d(AstExpr::In(
                    sp(ident("step")),
                    sp(setenum(vec![int(2), int(5), int(3)]))
                )),
                1
            ),
            Some(5)
        );
        // Next: `mode' ∈ {busy,idle}` ⇒ 1 ; `mode' = idle` ⇒ 1 ; `step' ∈ 0..3` ⇒ 3.
        assert_eq!(
            next_bound(
                &d(AstExpr::In(
                    sp(prime("mode")),
                    sp(setenum(vec![ident("busy"), ident("idle")]))
                )),
                0
            ),
            Some(1)
        );
        assert_eq!(
            next_bound(&d(AstExpr::Eq(sp(prime("mode")), sp(ident("idle")))), 0),
            Some(1)
        );
        assert_eq!(
            next_bound(
                &d(AstExpr::In(
                    sp(prime("step")),
                    sp(AstExpr::Range(sp(int(0)), sp(int(3))))
                )),
                1
            ),
            Some(3)
        );
        // FAIL-CLOSED: an unknown label in the set yields NO usable bound.
        assert_eq!(
            init_bound(
                &d(AstExpr::In(
                    sp(ident("mode")),
                    sp(setenum(vec![ident("busy"), ident("ghost")]))
                )),
                0
            ),
            None
        );
    }

    /// A handcrafted explicit-state cert over ARBITRARY column sorts (the AST-direct lane reads
    /// only `spec_src` + config names + `reachable`/`sorts`).
    fn handmade_cert_sorts(
        src: &str,
        init: &str,
        next: &str,
        invariants: Vec<String>,
        sorts: Vec<ColSort>,
        reachable: Vec<Vec<u64>>,
    ) -> SafetyCertificate {
        let fp = ExplicitFixpointCert {
            reachable: reachable.clone(),
            init_values: reachable.clone(),
            image: reachable,
            sorts,
            safety_term: Vec::new(),
            init_member_terms: Vec::new(),
            closed_member_terms: Vec::new(),
            next_shape: None,
            next_completeness: None,
            init_shape: None,
            init_completeness: None,
            next_pred: None,
            next_general_completeness: None,
            init_pred: None,
            init_general_completeness: None,
            unbounded_invariant: None,
            safety_pred: None,
            safety_general: Some(vec![0]),
            init_member_reflected: None,
            closed_member_reflected: None,
            deadlock_free: None,
            deadlock_scan: Default::default(),
        };
        let config = crate::Config {
            init: Some(init.into()),
            next: Some(next.into()),
            invariants,
            ..Default::default()
        };
        let mut cert = crate::cert::build_explicit_fixpoint_certificate(src, &config, fp);
        cert.digest = cert.compute_digest();
        cert
    }

    /// ATTACK 3 (membership-safety NOT-SAFE, Int literal set): Safety `x ∈ {0,1}` but an honest R
    /// includes `x=2`; the kernel reduces the Or-fold `(x=0)∨(x=1)` to `Bool.false` at [2] ⇒
    /// NOT-SAFE. The desugar is SOUND — a value outside the literal set falsifies membership.
    #[test]
    fn ast_direct_notsafe_on_int_membership_violation() {
        const SRC: &str = "\
---------------------- MODULE IntMem ----------------------
EXTENDS Naturals
VARIABLE x
HCini == x \\in {0, 1}
HCnxt == x' \\in {0, 1}
XSafe == x \\in {0, 1}
==========================================================
";
        let r: Vec<Vec<u64>> = vec![vec![0], vec![1], vec![2]];
        let cert = handmade_cert_sorts(
            SRC,
            "HCini",
            "HCnxt",
            vec!["XSafe".into()],
            vec![ColSort::Int],
            r,
        );
        assert_eq!(
            reflect_check_safety_cert_ast_direct(&cert),
            AstDirectVerdict::NotSafe { state: vec![2] }
        );
    }

    /// ATTACK 3 (membership-safety NOT-SAFE, ENUM label set): Safety `mode ∈ {d1,d2}` (labels
    /// [d1,d2,d3], so `d3` is code 2) with an honest R holding `mode=d3(2)`; the kernel reduces
    /// the code Or-fold `(mode=0)∨(mode=1)` to `Bool.false` at [2] ⇒ NOT-SAFE. Proves the
    /// label→code desugar is faithful: a label OUTSIDE the invariant set is caught.
    #[test]
    fn ast_direct_notsafe_on_enum_membership_violation() {
        const SRC: &str = "\
---------------------- MODULE EnumMem ----------------------
EXTENDS Naturals
VARIABLE mode
EInit == mode \\in {d1, d2}
ENext == mode' \\in {d1, d2}
ESafe == mode \\in {d1, d2}
==========================================================
";
        let r: Vec<Vec<u64>> = vec![vec![0], vec![1], vec![2]];
        let cert = handmade_cert_sorts(
            SRC,
            "EInit",
            "ENext",
            vec!["ESafe".into()],
            vec![enum_model(&["d1", "d2", "d3"])],
            r,
        );
        assert_eq!(
            reflect_check_safety_cert_ast_direct(&cert),
            AstDirectVerdict::NotSafe { state: vec![2] }
        );
    }

    /// ATTACK 1 end-to-end (unknown-label spec DECLINES): the Safety invariant references `ghost`,
    /// a label ABSENT from the column's sort `{d1,d2}` ⇒ the quoter cannot resolve it ⇒
    /// INCONCLUSIVE (never a wrong code, never a certify).
    #[test]
    fn ast_direct_declines_unknown_label_spec() {
        const SRC: &str = "\
---------------------- MODULE Ghost ----------------------
EXTENDS Naturals
VARIABLE mode
EInit == mode \\in {d1, d2}
ENext == mode' \\in {d1, d2}
ESafe == mode \\in {d1, ghost}
=========================================================
";
        let r: Vec<Vec<u64>> = vec![vec![0], vec![1]];
        let cert = handmade_cert_sorts(
            SRC,
            "EInit",
            "ENext",
            vec!["ESafe".into()],
            vec![enum_model(&["d1", "d2"])],
            r,
        );
        match reflect_check_safety_cert_ast_direct(&cert) {
            AstDirectVerdict::Inconclusive(why) => {
                assert!(
                    why.contains("fragment"),
                    "decline should name the fragment, got: {why}"
                )
            }
            other => panic!("an unknown label MUST DECLINE (never a wrong code), got {other:?}"),
        }
    }

    // ── End-to-end over a REAL enum + literal-set-membership cert ──────────────
    //
    // A STRING-enum column (`Enum{Str}`) is the end-to-end vehicle: it certifies via the
    // explicit-state fixpoint route (a reachability-bounded `n` counter defeats the symbolic
    // type-invariant proof, forcing enumeration) AND its atoms are LITERAL — `mode = "idle"`,
    // `mode ∈ {"idle","busy"}` — so they lie IN the increment-3 fragment. (Model-VALUE atoms
    // resolve by the IDENTICAL `resolve_enum_code` path — unit-tested in
    // `quoter_enum_label_resolution_and_decline_attacks`; they cannot be a WHOLE-spec end-to-end
    // vehicle here because a model value must be config-declared as a member of a model-value SET,
    // and `x ∈ ThatSetConstant` is NOT a literal set — `cert_inline` never inlines it — so it is
    // out of the increment-3 literal-set scope, as the ModeMV decline test below documents.)

    const STRCOUNT_SRC: &str = "\
---------------------- MODULE StrCount ----------------------
EXTENDS Naturals
VARIABLES mode, n
Init == mode = \"idle\" /\\ n = 0
SCNext == \\/ (mode = \"idle\" /\\ mode' = \"busy\" /\\ n' = n)
          \\/ (mode = \"busy\" /\\ mode' = \"idle\" /\\ n' = IF n < 2 THEN n + 1 ELSE 0)
Safety == mode \\in {\"idle\", \"busy\"} /\\ n \\in {0, 1, 2}
==========================================================
";

    fn strcount_cfg() -> crate::Config {
        crate::Config {
            init: Some("Init".into()),
            next: Some("SCNext".into()),
            invariants: vec!["Safety".into()],
            ..Default::default()
        }
    }

    fn strcount_full_cert() -> SafetyCertificate {
        let fp = certify_explicit_state_spec(STRCOUNT_SRC, &strcount_cfg())
            .expect("StrCount must explicit-state certify");
        let mut cert =
            crate::cert::build_explicit_fixpoint_certificate(STRCOUNT_SRC, &strcount_cfg(), fp);
        cert.digest = cert.compute_digest();
        cert
    }

    /// POSITIVE (increment 3, end-to-end): a genuine cert with an ENUM column (`mode : Enum{Str}`,
    /// labels [busy,idle], codes 0/1) whose Init/Next/Safety use enum-label EQUALITY
    /// (`mode = "idle"`, an enum guard), literal-STRING-set membership (`mode ∈ {"idle","busy"}`),
    /// and literal-INT-set membership (`n ∈ {0,1,2}`) — all quoted AST-direct — certifies all three
    /// legs (R=6; D_init={idle}×{0}=... the enum Init pin `mode = "idle"` bounds mode to code 1 and
    /// `n = 0` bounds n to 0, D_init = {0,1}×{0} = 2; D_next=8; pairs=48), and the recognized-IR
    /// lane CROSS-CHECKS to the same class.
    #[test]
    fn ast_direct_certifies_strcount_enum_and_membership() {
        let cert = strcount_full_cert();
        // Sanity: the cert really carries a string-enum column (else the test would be vacuous).
        assert!(
            matches!(
                cert.explicit_fixpoint.as_ref().unwrap().sorts.first(),
                Some(ColSort::Enum {
                    kind: EnumKind::Str,
                    ..
                })
            ),
            "StrCount must certify with an Enum{{Str}} first column"
        );
        let (verdict, cc) = reflect_check_ast_direct_with_crosscheck(&cert);
        assert_eq!(
            verdict,
            AstDirectVerdict::Certified {
                states: 6,
                init_domain: 2,
                next_domain: 8,
                next_pairs: 48
            }
        );
        assert!(
            matches!(cc, AstCrossCheck::Agree { .. }),
            "recognized lane must agree on StrCount, got {cc:?}"
        );
    }

    /// ATTACK 3 tamper (dropped successor ⇒ NOT-CLOSED over the enum/membership spec): drop the
    /// reachable NON-init successor [0,0] (mode=busy,n=0), which disjunct 1 produces from the Init
    /// state [1,0] (mode=idle,n=0). Codes are busy=0/idle=1, so [1,0]=(idle,0) is the Init state
    /// and stays in R; the Init leg passes, and the CLOSURE leg reduces `Next([1,0],[0,0]) ⇒
    /// [0,0]∈R` to `Bool.false` ⇒ NOT-CLOSED — proving the enum-eq/membership `D_next` actually
    /// COVERS that successor (had it under-covered, the leg would pass vacuously and certify a
    /// non-closed R: a false safe).
    #[test]
    fn ast_direct_strcount_dropped_successor_not_closed() {
        let mut cert = strcount_full_cert();
        cert.explicit_fixpoint
            .as_mut()
            .unwrap()
            .reachable
            .retain(|t| t != &vec![0, 0]);
        match reflect_check_safety_cert_ast_direct(&cert) {
            AstDirectVerdict::NotClosed { sp, .. } => {
                assert_eq!(
                    sp,
                    vec![0, 0],
                    "the escaping successor must be the dropped [0,0]"
                )
            }
            other => panic!("a dropped successor MUST be NOT-CLOSED, got {other:?}"),
        }
    }

    /// SCOPE BOUNDARY (why AsynchInterface / ModeMV do NOT flip): membership in a model-value SET
    /// CONSTANT `x ∈ Modes` is NOT a literal set — `cert_inline` deliberately never inlines a
    /// model-value set — so the `In` domain is a bare `Ident("Modes")` and the quoter DECLINES
    /// (Inconclusive), fail-closed. (The recognized-IR lane handles it via `mvsets`; the AST-direct
    /// increment-4 win: a MODEL-VALUE-SET CONSTANT domain (`mode ∈ Modes` where the `.cfg` gives
    /// `Modes = {idle, busy}`) now CERTIFIES AST-direct. `resolve_mvset_domains` rewrites the
    /// `In(x, Ident(Modes))` to the literal `In(x, {idle, busy})` — reading the SAME `.cfg`
    /// constant `ty check` reads — after which the proven Or-fold desugar + enum-code resolution
    /// apply. `mode' ∈ Modes` is a bounded nondeterministic successor (domain = the label
    /// universe). (The exact construct AsynchInterface's `val ∈ Data` uses — AsynchInterface itself
    /// additionally needs Nat-safe `1-x` subtraction in its Next, a later increment.)
    #[test]
    fn ast_direct_certifies_modelvalue_set_constant() {
        const MODEMV_SRC: &str = "\
---------------------- MODULE ModeMV ----------------------
EXTENDS Naturals
CONSTANT Modes
VARIABLES mode, step
Init == /\\ mode \\in Modes
        /\\ step \\in {0, 1}
MNext == /\\ mode' \\in Modes
         /\\ step' = step
Safety == /\\ mode \\in Modes
          /\\ step \\in {0, 1}
==========================================================
";
        const MODEMV_CFG: &str = "\
INIT Init
NEXT MNext
INVARIANT Safety
CONSTANT Modes = {idle, busy}
";
        let config = crate::Config::parse(MODEMV_CFG).expect("ModeMV cfg parses");
        let fp = certify_explicit_state_spec(MODEMV_SRC, &config)
            .expect("ModeMV must explicit-state certify");
        let mut cert = crate::cert::build_explicit_fixpoint_certificate(MODEMV_SRC, &config, fp);
        cert.digest = cert.compute_digest();
        // The cert really has a model-value enum column (else the test would be vacuous).
        assert!(matches!(
            cert.explicit_fixpoint.as_ref().unwrap().sorts.first(),
            Some(ColSort::Enum {
                kind: EnumKind::Model,
                ..
            })
        ));
        let (verdict, cc) = reflect_check_ast_direct_with_crosscheck(&cert);
        assert!(
            matches!(verdict, AstDirectVerdict::Certified { .. }),
            "the model-value-set-constant spec must now CERTIFY AST-direct (increment 4), got {verdict:?}"
        );
        assert!(
            !matches!(cc, AstCrossCheck::Divergence { .. }),
            "recognized lane must not DIVERGE on ModeMV, got {cc:?}"
        );
    }

    /// SOUNDNESS of the mvset rewrite: it resolves `x ∈ C` to `x ∈ <C's literal value>` — the
    /// EXACT meaning — so a GENUINE (non-type) membership constraint that a reachable state
    /// VIOLATES must reduce to NOT-SAFE, never a false certify. `x ∈ Allowed` with `Allowed =
    /// {0,1}` but `x` reaching `2` in R.
    #[test]
    fn mvset_rewrite_preserves_meaning_violated_is_not_safe() {
        const SRC: &str = "\
---------------------- MODULE Conx ----------------------
EXTENDS Naturals
CONSTANT Allowed
VARIABLE x
Init == x = 0
XNext == \\/ (x < 2 /\\ x' = x + 1)
         \\/ (x' = x)
Safety == x \\in Allowed
========================================================
";
        const CFG: &str = "\
INIT Init
NEXT XNext
INVARIANT Safety
CONSTANT Allowed = {0, 1}
";
        let config = crate::Config::parse(CFG).expect("cfg parses");
        // R reaches x=2 (∉ {0,1}); the explicit-state certifier still enumerates R (safety is a
        // membership the kernel will refute at x=2), so build the cert and reflect-check it.
        let Some(fp) = certify_explicit_state_spec(SRC, &config) else {
            // If the certifier itself declines (R⊆Safety fails at mint), that is ALSO fail-closed —
            // no cert, no false safe. Acceptable.
            return;
        };
        let mut cert = crate::cert::build_explicit_fixpoint_certificate(SRC, &config, fp);
        cert.digest = cert.compute_digest();
        let verdict = reflect_check_safety_cert_ast_direct(&cert);
        assert!(
            !matches!(verdict, AstDirectVerdict::Certified { .. }),
            "a violated membership constraint (x=2 ∉ {{0,1}}) must NOT certify — got {verdict:?}"
        );
    }

    /// `parse_set_literal` is fail-closed on non-set-literal shapes (a scalar, nested braces, an
    /// empty set, an empty element) and admits exactly a flat bare-atom list.
    #[test]
    fn parse_set_literal_is_fail_closed() {
        assert_eq!(
            parse_set_literal("{a, b, c}"),
            Some(vec!["a".into(), "b".into(), "c".into()])
        );
        assert_eq!(
            parse_set_literal(" {d1,d2} "),
            Some(vec!["d1".into(), "d2".into()])
        );
        assert_eq!(parse_set_literal("3"), None); // scalar
        assert_eq!(parse_set_literal("{}"), None); // empty set
        assert_eq!(parse_set_literal("{a, {b}}"), None); // nested
        assert_eq!(parse_set_literal("{a, }"), None); // empty element
        assert_eq!(parse_set_literal("{a b}"), None); // whitespace-separated (not comma)
    }

    /// SOUNDNESS (cross-kind guard): an ENUM / model-value column in an INTEGER interval
    /// (`mode ∈ 0..1`) or ORDER comparison (`mode < 2`) is a category error — `ty check` reports
    /// the invariant VIOLATED (a model value is never an ordered/interval integer). Quoting it on
    /// the label CODE would falsely certify, so both must DECLINE (fail-closed). Guards the five
    /// arms `Lt`/`Leq`/`Gt`/`Geq`/`In(Range)`. Regression for the skeptic-found lane-level gap.
    #[test]
    fn enum_column_in_int_interval_or_comparison_declines() {
        let vars = ["mode"];
        let sorts = [ColSort::Enum {
            labels: vec!["idle".into(), "busy".into()],
            kind: EnumKind::Model,
        }];
        // `mode ∈ 0..1` — enum column in an Int interval ⇒ decline.
        let in_range = AstExpr::In(
            sp(ident("mode")),
            sp(AstExpr::Range(sp(int(0)), sp(int(1)))),
        );
        let max_r: &[u64] = &[1];
        assert!(
            quote_pred_ast(&in_range, &vars, &sorts, max_r, false).is_none(),
            "`mode ∈ 0..1` (enum in Int interval) must DECLINE"
        );
        // `mode < 2`, `mode <= 1`, `mode > 0`, `mode >= 0` — enum column in an order comparison.
        for cmp in [
            AstExpr::Lt(sp(ident("mode")), sp(int(2))),
            AstExpr::Leq(sp(ident("mode")), sp(int(1))),
            AstExpr::Gt(sp(ident("mode")), sp(int(0))),
            AstExpr::Geq(sp(ident("mode")), sp(int(0))),
            // enum on the RIGHT operand too
            AstExpr::Lt(sp(int(0)), sp(ident("mode"))),
        ] {
            assert!(
                quote_pred_ast(&cmp, &vars, &sorts, max_r, false).is_none(),
                "an order comparison with an enum operand must DECLINE: {cmp:?}"
            );
        }
        // Control: an INT column in the same comparison still quotes (the guard is enum-specific).
        let int_sorts = [ColSort::Int];
        let int_vars = ["x"];
        let int_max_r: &[u64] = &[5];
        let int_cmp = AstExpr::Lt(sp(ident("x")), sp(int(5)));
        assert!(
            quote_pred_ast(&int_cmp, &int_vars, &int_sorts, int_max_r, false).is_some(),
            "`x < 5` on an Int column must still quote (guard is enum-specific, no over-decline)"
        );
    }

    /// SOUNDNESS (S1 root-cause guard): the per-arm enum guards were only skin-deep — an enum
    /// column in a NESTED value position (inside `+`, an `IF` branch, a RANGE BOUND, a compound
    /// comparison operand) reached `quote_val_ast`'s value leaf and was quoted as its Nat CODE,
    /// treated as an ordered integer. `quote_val_ast` now declines any enum column at the value
    /// leaf, closing every nested position uniformly. These are the exact bypasses the audit found.
    #[test]
    fn enum_column_in_nested_value_position_declines() {
        let vars = ["mode"];
        let sorts = [ColSort::Enum {
            labels: vec!["a".into(), "b".into()],
            kind: EnumKind::Model,
        }];
        let max_r: &[u64] = &[1];
        // `mode + 0 < 2` — enum inside `+` inside a comparison.
        let add_cmp = AstExpr::Lt(sp(AstExpr::Add(sp(ident("mode")), sp(int(0)))), sp(int(2)));
        // `y ∈ 0..mode` — enum as a RANGE BOUND (the range subject is a fresh int, mode is the hi).
        let range_bound = AstExpr::In(
            sp(int(1)),
            sp(AstExpr::Range(sp(int(0)), sp(ident("mode")))),
        );
        // `x = IF TRUE THEN mode ELSE 1` — enum as an `IF` branch (via the eq_ite composite).
        let ite_branch = AstExpr::Eq(
            sp(ident("mode")),
            sp(AstExpr::If(
                sp(ident("mode")),
                sp(ident("mode")),
                sp(int(1)),
            )),
        );
        for (label, e) in [
            ("mode + 0 < 2", add_cmp),
            ("y ∈ 0..mode", range_bound),
            ("IF-branch mode", ite_branch),
        ] {
            assert!(
                quote_pred_ast(&e, &vars, &sorts, max_r, false).is_none(),
                "enum column in a NESTED value position must DECLINE: {label}"
            );
        }
        // And the bare value leaf itself declines for an enum column.
        assert!(
            quote_val_ast(&ident("mode"), &vars, &sorts, max_r, false).is_none(),
            "an enum column at the value leaf must DECLINE (its code is not an integer value)"
        );
    }

    // ── Increment 6: the two-pass one-hop column-equality Init pin ──────────────

    /// POSITIVE (the AsynchInterface Init shape, Int projection): `rdy ∈ {0,1} ∧ ack = rdy`. Pass 1
    /// LITERALLY pins `rdy` to 1 and leaves `ack` unpinned; pass 2 transfers `rdy`'s bound to `ack`
    /// via the one-hop column-equality rule — `ack`'s derived bound EQUALS the pinned column's
    /// bound H_j (the Aristotle `init_domain_pin_transfer` bound). Both equality orientations pin.
    #[test]
    fn ast_init_two_pass_one_hop_column_equality_pin() {
        let vars = &["rdy", "ack"];
        let sorts: &[ColSort] = &[ColSort::Int, ColSort::Int];
        let d = Spanned::dummy;
        let pin_via_two_pass = |eq: AstExpr| -> Option<u64> {
            let init = d(AstExpr::And(
                sp(AstExpr::In(
                    sp(ident("rdy")),
                    sp(setenum(vec![int(0), int(1)])),
                )),
                sp(eq),
            ));
            let mut conjs = Vec::new();
            flatten_and(&init, &mut conjs);
            // Pass 1: rdy ⇒ Some(1) (literal), ack ⇒ None (not literally pinned).
            let lit: Vec<Option<u64>> = (0..2u64)
                .map(|i| ast_init_col_bound(&conjs, i, vars, sorts))
                .collect();
            assert_eq!(
                lit,
                vec![Some(1), None],
                "pass 1: only rdy literally pinned"
            );
            // Pass 2: transfer rdy's bound (H_j = 1) to ack. The transferred bound EQUALS H_j.
            let transferred = ast_init_col_eq_pin(&conjs, 1, vars, &lit);
            assert_eq!(
                transferred, lit[0],
                "the pin-transfer bound equals the pinned column's H_j"
            );
            transferred
        };
        assert_eq!(
            pin_via_two_pass(AstExpr::Eq(sp(ident("ack")), sp(ident("rdy")))),
            Some(1)
        ); // ack = rdy
        assert_eq!(
            pin_via_two_pass(AstExpr::Eq(sp(ident("rdy")), sp(ident("ack")))),
            Some(1)
        ); // rdy = ack
    }

    /// THE DECISIVE ONE-HOP SOUNDNESS TESTS. CONTROL (one-hop pins): `y = z ∧ z ∈ {0,1}` — y's only
    /// partner z is LITERALLY pinned ⇒ y pins to z's bound 1. ATTACK 1 (CHAIN — the SECOND hop
    /// declines): `x = y ∧ y = z ∧ z ∈ {0,1}` — only z is literally pinned, so x's partner y is NOT
    /// literally pinned ⇒ x DECLINES (pass 2 reads only pass-1 bounds — never chains through y's
    /// pass-2 bound). ATTACK 2 (UNPINNED RHS): `x = y`, neither pinned ⇒ decline. CYCLE
    /// `x = y ∧ y = x` ⇒ decline. CONFLICT: `w = a ∧ w = b`, a,b pinned to DIFFERENT bounds ⇒
    /// decline (never a guess). The rule is fail-closed on ANY non-literally-pinned partner.
    #[test]
    fn ast_init_col_eq_pin_chain_unpinned_cycle_conflict_decline() {
        let vars = &["x", "y", "z"];
        let sorts: &[ColSort] = &[ColSort::Int, ColSort::Int, ColSort::Int];
        let d = Spanned::dummy;
        let lit_of = |conjs: &[&Spanned<AstExpr>], vs: &[&str]| -> Vec<Option<u64>> {
            (0..vs.len() as u64)
                .map(|i| ast_init_col_bound(conjs, i, vs, sorts))
                .collect()
        };
        // CONTROL (ONE HOP PINS): `y = z ∧ z ∈ {0,1}` — y's ONLY partner z is literally pinned ⇒ y
        // inherits z's bound 1 (`s_y = s_z ⇒ s_y ≤ 1`). This shows the one-hop rule working.
        let one_hop = d(AstExpr::And(
            sp(AstExpr::Eq(sp(ident("y")), sp(ident("z")))),
            sp(AstExpr::In(
                sp(ident("z")),
                sp(setenum(vec![int(0), int(1)])),
            )),
        ));
        let mut oh = Vec::new();
        flatten_and(&one_hop, &mut oh);
        let lit_oh = lit_of(&oh, vars);
        assert_eq!(lit_oh, vec![None, None, Some(1)]);
        assert_eq!(
            ast_init_col_eq_pin(&oh, 1, vars, &lit_oh),
            Some(1),
            "y one-hop pins from z"
        );

        // ATTACK 1 (CHAIN — x is a SECOND hop).
        let chain = d(AstExpr::And(
            sp(AstExpr::And(
                sp(AstExpr::Eq(sp(ident("x")), sp(ident("y")))),
                sp(AstExpr::Eq(sp(ident("y")), sp(ident("z")))),
            )),
            sp(AstExpr::In(
                sp(ident("z")),
                sp(setenum(vec![int(0), int(1)])),
            )),
        ));
        let mut cs = Vec::new();
        flatten_and(&chain, &mut cs);
        let lit = lit_of(&cs, vars);
        assert_eq!(lit, vec![None, None, Some(1)], "only z is LITERALLY pinned");
        // x: its partner y is NOT literally pinned (lit[1] == None) ⇒ DECLINE. THE decisive
        // guarantee — pass 2 never chains through y's pass-2 bound.
        assert_eq!(
            ast_init_col_eq_pin(&cs, 0, vars, &lit),
            None,
            "x (2nd hop) declines"
        );
        // y ALSO declines HERE (unlike the control): it is additionally equated to the UNPINNED x
        // (`x = y`), and the fail-closed rule declines a column equated to ANY non-literally-pinned
        // partner. Either way, the CHAIN lane declines (x is unpinnable).
        assert_eq!(
            ast_init_col_eq_pin(&cs, 1, vars, &lit),
            None,
            "y declines: equated to unpinned x"
        );

        // ATTACK 2 (UNPINNED RHS): x = y, neither pinned.
        let unp = d(AstExpr::Eq(sp(ident("x")), sp(ident("y"))));
        let mut cu = Vec::new();
        flatten_and(&unp, &mut cu);
        let lit_u = lit_of(&cu, vars);
        assert_eq!(lit_u, vec![None, None, None]);
        assert_eq!(ast_init_col_eq_pin(&cu, 0, vars, &lit_u), None);
        assert_eq!(ast_init_col_eq_pin(&cu, 1, vars, &lit_u), None);

        // CYCLE: x = y ∧ y = x, neither literally pinned ⇒ decline both.
        let cyc = d(AstExpr::And(
            sp(AstExpr::Eq(sp(ident("x")), sp(ident("y")))),
            sp(AstExpr::Eq(sp(ident("y")), sp(ident("x")))),
        ));
        let mut cc = Vec::new();
        flatten_and(&cyc, &mut cc);
        let lit_c = lit_of(&cc, vars);
        assert_eq!(ast_init_col_eq_pin(&cc, 0, vars, &lit_c), None);
        assert_eq!(ast_init_col_eq_pin(&cc, 1, vars, &lit_c), None);

        // CONFLICT: w = a ∧ w = b with a ∈ {0,1} (H=1), b ∈ 0..5 (H=5) ⇒ conflicting ⇒ decline.
        let wv = &["w", "a", "b"];
        let conflict = d(AstExpr::And(
            sp(AstExpr::And(
                sp(AstExpr::Eq(sp(ident("w")), sp(ident("a")))),
                sp(AstExpr::Eq(sp(ident("w")), sp(ident("b")))),
            )),
            sp(AstExpr::And(
                sp(AstExpr::In(
                    sp(ident("a")),
                    sp(setenum(vec![int(0), int(1)])),
                )),
                sp(AstExpr::In(
                    sp(ident("b")),
                    sp(AstExpr::Range(sp(int(0)), sp(int(5)))),
                )),
            )),
        ));
        let mut ccf = Vec::new();
        flatten_and(&conflict, &mut ccf);
        let lit_cf = lit_of(&ccf, wv);
        assert_eq!(lit_cf, vec![None, Some(1), Some(5)]);
        assert_eq!(
            ast_init_col_eq_pin(&ccf, 0, wv, &lit_cf),
            None,
            "conflicting equality pins must DECLINE, never a guessed bound"
        );
    }

    const HANDSHAKE_SRC: &str = "\
---------------------- MODULE Handshake ----------------------
EXTENDS Naturals
VARIABLES rdy, ack
HSInit == rdy \\in {0, 1} /\\ ack = rdy
HSNext == \\/ (rdy = ack /\\ rdy' = 1 - rdy /\\ UNCHANGED ack)
          \\/ (rdy # ack /\\ ack' = 1 - ack /\\ UNCHANGED rdy)
HSInv  == rdy \\in {0, 1} /\\ ack \\in {0, 1}
==============================================================
";

    fn handshake_cfg() -> crate::Config {
        crate::Config {
            init: Some("HSInit".into()),
            next: Some("HSNext".into()),
            invariants: vec!["HSInv".into()],
            ..Default::default()
        }
    }

    fn handshake_full_cert() -> SafetyCertificate {
        let fp = certify_explicit_state_spec(HANDSHAKE_SRC, &handshake_cfg())
            .expect("Handshake must explicit-state certify");
        let mut cert =
            crate::cert::build_explicit_fixpoint_certificate(HANDSHAKE_SRC, &handshake_cfg(), fp);
        cert.digest = cert.compute_digest();
        cert
    }

    /// POSITIVE (increment 6, END-TO-END — the AsynchInterface column-equality Init shape distilled
    /// to two Int columns): a genuine cert whose `Init == rdy ∈ {0,1} ∧ ack = rdy` pins `ack` (NOT
    /// literally pinned) to `rdy`'s bound via the one-hop rule, so all three legs discharge and the
    /// recognized-IR lane cross-checks to the same class. R = {(0,0),(0,1),(1,0),(1,1)} = 4;
    /// D_init = {0,1}×{0,1} = 4 (ack pinned to rdy's bound 1); D_next = 4; pairs = 16.
    #[test]
    fn ast_direct_certifies_handshake_column_equality_init() {
        let cert = handshake_full_cert();
        let (verdict, cc) = reflect_check_ast_direct_with_crosscheck(&cert);
        assert_eq!(
            verdict,
            AstDirectVerdict::Certified {
                states: 4,
                init_domain: 4,
                next_domain: 4,
                next_pairs: 16
            }
        );
        assert!(
            matches!(cc, AstCrossCheck::Agree { .. }),
            "recognized lane must agree on Handshake, got {cc:?}"
        );
    }

    /// ATTACK 3 (tamper ⇒ NOT-INIT-COMPLETE over the column-equality Init): drop the Init state
    /// (1,1) from R. (1,1) satisfies Init (rdy=1∈{0,1}, ack=1=rdy) and lies IN the one-hop-derived
    /// D_init, so the Init-completeness leg reduces `Init((1,1)) ⇒ (1,1)∈R` to Bool.false ⇒
    /// NOT-INIT-COMPLETE — proving D_init actually COVERS ack's Init value 1. Had the one-hop rule
    /// derived ack's bound too SMALL (excluding 1), (1,1) would be OUTSIDE D_init and the missed
    /// Init state would pass VACUOUSLY (a false safe) — which is exactly why the rule transfers
    /// `rdy`'s TRUE bound H_j (Aristotle `init_domain_pin_transfer`), never a smaller guess.
    #[test]
    fn ast_direct_handshake_dropped_init_state_not_init_complete() {
        let mut cert = handshake_full_cert();
        cert.explicit_fixpoint
            .as_mut()
            .unwrap()
            .reachable
            .retain(|t| t != &vec![1, 1]);
        // Sanity: the true-bound D_init really contains the tampered Init state (else vacuous).
        let prep = prepare_ast_direct(&cert).expect("Handshake must prepare");
        assert!(
            prep.init_domain.contains(&vec![1, 1]),
            "the one-hop-derived D_init must COVER ack's Init value (else the leg is vacuous)"
        );
        assert_eq!(
            reflect_check_safety_cert_ast_direct(&cert),
            AstDirectVerdict::NotInitComplete { s: vec![1, 1] }
        );
    }

    /// ATTACK 1 END-TO-END (a CHAIN spec DECLINES): `Init == z ∈ {0,1} ∧ y = z ∧ x = y`. Only z is
    /// literally pinned; y is one hop from z, but x = y is a second hop (y is not literally pinned)
    /// ⇒ column x is unpinnable ⇒ the lane DECLINES (Inconclusive), never certifying with a chained
    /// (possibly-wrong) bound. The decline names the un-derivable column.
    #[test]
    fn ast_direct_declines_chain_column_equality_spec() {
        const SRC: &str = "\
---------------------- MODULE Chain ----------------------
EXTENDS Naturals
VARIABLES x, y, z
CInit == z \\in {0, 1} /\\ y = z /\\ x = y
CNext == x' = x /\\ y' = y /\\ z' = z
CInv  == x \\in {0, 1} /\\ y \\in {0, 1} /\\ z \\in {0, 1}
=========================================================
";
        let r: Vec<Vec<u64>> = vec![vec![0, 0, 0], vec![1, 1, 1]];
        let cert = handmade_cert_sorts(
            SRC,
            "CInit",
            "CNext",
            vec!["CInv".into()],
            vec![ColSort::Int, ColSort::Int, ColSort::Int],
            r,
        );
        match reflect_check_safety_cert_ast_direct(&cert) {
            AstDirectVerdict::Inconclusive(why) => assert!(
                why.contains("does not pin column 0") && why.contains('x'),
                "the chain must DECLINE naming the un-derivable column x, got: {why}"
            ),
            other => panic!(
                "a CHAIN `x = y = z` MUST DECLINE (one hop only — x's partner y is not literally \
                 pinned), never certify with a chained bound, got {other:?}"
            ),
        }
    }
}
