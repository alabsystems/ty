// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! `reflect` v1: the **deep-embedding corroboration lane** for scalar kernel obligations.
//!
//! The shallow embedder ([`crate::cleancic::embed_pred_ir`]) is trusted Rust: for every
//! obligation it CONSTRUCTS the kernel term op by op (`Lt` becomes `Nat.ble (a+1) b`, `Equiv`
//! becomes `(a∧b)∨(¬a∧¬b)`, …), so a bug in that construction silently changes what the kernel
//! is asked to prove. This module moves those op choices INTO the kernel:
//!
//! 1. a deep syntax for the scalar `PredIR`/`ValIR` fragment — the inductives
//!    [`TyReflectPExp`](PEXP) and [`TyReflectPPred`](PPRED) — admitted at runtime through
//!    `Environment::add_inductive` (the CHECKED admission path: positivity, universes, and
//!    recursor derivation are all kernel work, not ours);
//! 2. kernel-defined evaluators [`TyReflectEvalV`](EVALV)`: TyReflectPExp → List Nat → List
//!    Nat → Nat` and [`TyReflectEvalP`](EVALP)`: TyReflectPPred → List Nat → List Nat → Bool`,
//!    built from the DERIVED recursors and admitted through `Environment::add_decl` (checked:
//!    each definition body is type-checked against its stated type before registration). Their
//!    op choices mirror the shallow embedder's EXACTLY (see the minor premises below);
//! 3. a QUOTER — [`quote_val`] / [`quote_pred`] / [`quote_state`] — that maps IR nodes to deep
//!    constructors one match arm per node, with NO logic beyond the 1:1 constructor mapping;
//! 4. [`reflect_corroborate`]: for a concrete `(ir, s, sp)` the kernel itself REDUCES
//!    `TyReflectEvalP ⌜ir⌝ ⌜s⌝ ⌜sp⌝` and decides `Eq Bool … Bool.{true|false}` by `Eq.refl`.
//!
//! ## The honest trust story
//!
//! **What leaves the TCB (for reflected obligations):** the shallow embedder's op-by-op Rust
//! term construction. Which kernel term realizes `Lt`, `Implies`, `Unchanged`, … is no longer
//! decided per obligation by Rust code paths; it is decided ONCE, as kernel-checked definition
//! DATA (the recursor minor premises of `TyReflectEvalV`/`TyReflectEvalP`), and applied by
//! kernel reduction.
//!
//! **What remains trusted:**
//! * the quoter — deliberately line-auditable: one constructor per match arm, no arithmetic,
//!   no op selection;
//! * clean-kernel's checked admission path (`add_inductive` / `add_decl`; this module NEVER
//!   calls `add_decl_structural`) and the `with_prelude` base env it extends (see
//!   `kernel_census` for that env's honest axiom inventory);
//! * the clean kernel itself (its type checker and native `Nat`/`Bool` literal reduction);
//! * the CLAIM that the deep evaluator's op choices agree with the shallow embedder's — pinned
//!   here by per-constructor DEFINITIONAL tests and by [`reflect_agrees_with_shallow`], not yet
//!   by a kernel induction proof (the feasibility probe demonstrated genuine `PExp` inductions
//!   check, so v2 can discharge agreement lemmas inside the kernel).
//!
//! **Boundary honesty:** the deep indexer `TyReflectNth` returns the DEFAULT `0` for a column
//! index beyond the state length (a total function; the kernel has no partiality to offer
//! here), which does NOT mirror the shallow embedder (whose Rust `s[i]` would panic). The
//! corroborator therefore DECLINES (`Unavailable`) any IR whose `Var`/`Prime`/`Unchanged`
//! column index is `≥` the state length instead of relying on the default — enforced by
//! `pred_cols_in_bounds` and pinned by test.
//!
//! ## Coverage
//!
//! The deep evaluator covers the scalar `PredIR`/`ValIR` fragment PLUS the bitmask SET fragment that
//! `pred_exact` admits: the set-valued `SetIR` (`Lit`/`Var`/`Prime`/`Cup`/`Cap`/`Digit` — the
//! `SetMask`/`SetMaskRec`/`FuncSetMask` columns), the set predicates
//! (`SetEq`/`SetNeq`/`SetMem`/`SetNotMem`/`SetSubseteq`/`SetUnchanged`), and the counting values
//! `SetCard` (bitmask popcount) / `CountFold` (`Σ boolToNat` over a fixed domain). These realize as
//! `Nat` bitwise ops (`Nat.lor`/`Nat.land`/`Nat.shiftRight` — the three ctors [`PEXP_LOR`] /
//! [`PEXP_LAND`] / [`PEXP_SHR`]) plus truncated-`Nat` arithmetic, EXACTLY mirroring the shallow
//! `embed_set_ir`/`embed_val_ir`/`embed_pred_ir` op choices, so `SetMask`/`FuncSetMask`/`SetMaskRec`/
//! `Cardinality` specs are now semantically cross-checked (were `Uncovered`).
//!
//! Honestly UNCOVERED (the quoter returns `None`, so the cross-check is `Uncovered`, never a false
//! `Corroborate`): the `Seq*` value ops, `SetIR::Filter` set-comprehensions, and the four bounded
//! `SUBSET`/set QUANTIFIER folds (`SetForall`/`SetExists`/`SubsetForall`/`SubsetExists`) — the last
//! two families `pred_exact`/`set_exact` also reject, so no COVERED obligation ever reaches them.
//!
//! ## v2 roadmap
//!
//! * the remaining `Seq*` / `Filter` / quantifier folds via FUEL-BOUNDED folds in the deep syntax (the
//!   shallow embedder unrolls them at embed time; deeply they become `Nat.rec` fuel folds);
//! * agreement lemmas (`∀ ir s sp, TyReflectEvalP ⌜ir⌝ ⌜s⌝ ⌜sp⌝ = shallow(ir,s,sp)`) proved by
//!   kernel induction over the deep syntax, closing the residual op-agreement trust;
//! * replacing the shallow obligations outright, so the per-obligation TCB is quoter + kernel.

use std::sync::OnceLock;

use clean_kernel::expr::BinderInfo;
use clean_kernel::{
    Constructor, Declaration, Environment, Expr, InductiveDecl, InductiveType, Level, Name,
    TypeChecker,
};

use crate::explicit_fixpoint_cert::{PredIR, SetIR, ValIR};

// ===========================================================================
// Names. `TyReflect`-prefixed so nothing collides with the Clean prelude.
// ===========================================================================

/// Deep syntax of scalar VALUE terms (mirrors [`ValIR`]'s scalar fragment).
pub const PEXP: &str = "TyReflectPExp";
const PEXP_LIT: &str = "TyReflectPExp.lit";
const PEXP_VAR: &str = "TyReflectPExp.var";
const PEXP_PRIME: &str = "TyReflectPExp.prime";
const PEXP_ADD: &str = "TyReflectPExp.add";
const PEXP_MUL: &str = "TyReflectPExp.mul";
const PEXP_DIV: &str = "TyReflectPExp.div";
const PEXP_MOD: &str = "TyReflectPExp.mod";
const PEXP_SUB: &str = "TyReflectPExp.sub";
// Bitwise `Nat` ops — the SET-fragment extension (bitmask sets embed as `Nat` bitwise ops, EXACTLY
// mirroring the shallow `embed_set_ir`: `∪`=`Nat.lor`, `∩`=`Nat.land`, and a bit test / popcount via
// `Nat.shiftRight`+`Nat.land`). APPENDED after the arithmetic ctors so the existing indices are
// undisturbed (the recursor minor-premise order is [`PEXP_CTORS`]).
const PEXP_LOR: &str = "TyReflectPExp.lor";
const PEXP_LAND: &str = "TyReflectPExp.land";
const PEXP_SHR: &str = "TyReflectPExp.shr";
const PEXP_REC: &str = "TyReflectPExp.rec";
/// The 11 `TyReflectPExp` constructors, in DECLARATION order (the recursor's minor-premise
/// order — the evaluator definitions below depend on it).
const PEXP_CTORS: [&str; 11] = [
    PEXP_LIT, PEXP_VAR, PEXP_PRIME, PEXP_ADD, PEXP_MUL, PEXP_DIV, PEXP_MOD, PEXP_SUB, PEXP_LOR,
    PEXP_LAND, PEXP_SHR,
];

/// Deep syntax of scalar PREDICATE terms (mirrors [`PredIR`]'s scalar fragment, PLUS the two
/// AST-direct constructors `inRange`/`eqIte` that only the AST quoter
/// ([`crate::reflect_ast_direct`]) emits — see [`PPRED_INRANGE`]/[`PPRED_EQITE`]).
pub const PPRED: &str = "TyReflectPPred";
const PPRED_AND: &str = "TyReflectPPred.and";
const PPRED_OR: &str = "TyReflectPPred.or";
const PPRED_NOT: &str = "TyReflectPPred.not";
const PPRED_IMPLIES: &str = "TyReflectPPred.implies";
const PPRED_EQUIV: &str = "TyReflectPPred.equiv";
const PPRED_BOOLLIT: &str = "TyReflectPPred.boolLit";
const PPRED_EQ: &str = "TyReflectPPred.eq";
const PPRED_NEQ: &str = "TyReflectPPred.neq";
const PPRED_LT: &str = "TyReflectPPred.lt";
const PPRED_LEQ: &str = "TyReflectPPred.leq";
const PPRED_GT: &str = "TyReflectPPred.gt";
const PPRED_GEQ: &str = "TyReflectPPred.geq";
const PPRED_UNCHANGED: &str = "TyReflectPPred.unchanged";
// ── AST-DIRECT fragment ctors (design pivot increment 1, docs/cert/design-pivot-reflect.md) ──
// These two give the AST-direct quoter (`crate::reflect_ast_direct`) 1:1 targets for two TLA+
// AST shapes the IR fragment desugars away: interval membership `x ∈ lo..hi` and the pinned
// conditional update `l = IF c THEN t ELSE f`. APPENDED after the existing ctors so every
// existing constructor index (and recursor minor-premise position) is undisturbed. They have
// NO shallow-embedder counterpart (the IR quoter never emits them): their semantics is decided
// ONCE, as the kernel-checked evaluator definition data below, and pinned by definitional tests.
const PPRED_INRANGE: &str = "TyReflectPPred.inRange";
const PPRED_EQITE: &str = "TyReflectPPred.eqIte";
const PPRED_REC: &str = "TyReflectPPred.rec";
/// The 15 `TyReflectPPred` constructors, in DECLARATION order.
const PPRED_CTORS: [&str; 15] = [
    PPRED_AND,
    PPRED_OR,
    PPRED_NOT,
    PPRED_IMPLIES,
    PPRED_EQUIV,
    PPRED_BOOLLIT,
    PPRED_EQ,
    PPRED_NEQ,
    PPRED_LT,
    PPRED_LEQ,
    PPRED_GT,
    PPRED_GEQ,
    PPRED_UNCHANGED,
    PPRED_INRANGE,
    PPRED_EQITE,
];

/// `TyReflectNth : List Nat → Nat → Nat` — positional state indexing, default `0` out of range.
pub const NTH: &str = "TyReflectNth";
/// `TyReflectEvalV : TyReflectPExp → List Nat → List Nat → Nat`.
pub const EVALV: &str = "TyReflectEvalV";
/// `TyReflectEvalP : TyReflectPPred → List Nat → List Nat → Bool`.
pub const EVALP: &str = "TyReflectEvalP";
/// `TyReflectTupEq : List Nat → List Nat → Bool` — structural tuple equality (a `List.rec` fold
/// with `Nat.beq`, LENGTH-SENSITIVE: lists of different lengths are unequal).
pub const TUPEQ: &str = "TyReflectTupEq";
/// `TyReflectMem : List Nat → List (List Nat) → Bool` — tuple membership in a quoted tuple set
/// (a `List.rec` fold of `Bool.or ∘ TyReflectTupEq` over the set).
pub const MEM: &str = "TyReflectMem";
/// `TyReflectAllMem : List (List Nat) → List (List Nat) → Bool` — every member of `xs` is a
/// member of `ys` (a `List.rec` fold of `Bool.and ∘ TyReflectMem`). The SEMANTIC reference for
/// the reflected `⊆` obligations; O(|xs|·|ys|) kernel reduction work.
pub const ALLMEM: &str = "TyReflectAllMem";
/// `TyReflectSubseq : List (List Nat) → List (List Nat) → Bool` — `xs` is an ORDER-PRESERVING
/// `TyReflectTupEq`-matching subsequence of `ys` (the sorted-merge subset check). `Subseq xs ys =
/// true` IMPLIES every member of `xs` is a member of `ys` (each `xs`-element is matched against a
/// distinct `ys`-element), so a `true` verdict is sound for the `⊆` claim UNCONDITIONALLY; the
/// converse needs `xs` to embed in `ys`'s order, which the callers guarantee by canonicalizing
/// BOTH lists (sorted + deduplicated) — a sorted subset is always a subsequence of its sorted
/// superset. Chosen over [`ALLMEM`] for the LIVE legs because its kernel reduction is
/// O(|xs|+|ys|) (one merge pass), not O(|xs|·|ys|) — at the CoffeeCan scale (|R| ≈ 5.2K) the
/// quadratic fold exceeds the kernel's heartbeat budget and the debug-build wall clock, the merge
/// does not (probed; see the tests).
pub const SUBSEQ: &str = "TyReflectSubseq";

// ===========================================================================
// Term-construction helpers (same conventions as the feasibility probe).
// ===========================================================================

fn nm(s: &str) -> Name {
    Name::from_string(s)
}
fn c(s: &str) -> Expr {
    Expr::const_str(s)
}
/// Const at universe level 1 (`Type` motives / `Eq` at `Type`).
fn c1(s: &str) -> Expr {
    Expr::const_str_levels(s, vec![Level::succ(Level::zero())])
}
fn lam(ty: Expr, body: Expr) -> Expr {
    Expr::lam(BinderInfo::Default, ty, body)
}
#[cfg_attr(not(test), allow(dead_code))] // used by the definitional-equation tests
fn pi(ty: Expr, body: Expr) -> Expr {
    Expr::pi(BinderInfo::Default, ty, body)
}
fn ap(f: Expr, args: impl IntoIterator<Item = Expr>) -> Expr {
    Expr::apps(f, args)
}
fn bv(i: u32) -> Expr {
    Expr::bvar(i)
}
fn nl(v: u64) -> Expr {
    Expr::nat_lit(v)
}
fn nat() -> Expr {
    c("Nat")
}
fn boolc() -> Expr {
    c("Bool")
}
/// The state type: the PRELUDE's `List` (level params `[u]`, `List.{u} : Sort(u+1) → Sort(u+1)`)
/// instantiated at `u=0`, applied to `Nat` — i.e. `List.{0} Nat : Type`.
fn list_nat() -> Expr {
    Expr::app(Expr::const_str_levels("List", vec![Level::zero()]), nat())
}
/// Motive/result type of the value evaluator: `List Nat → List Nat → Nat`.
fn vmot() -> Expr {
    Expr::arrow(list_nat(), Expr::arrow(list_nat(), nat()))
}
/// Motive/result type of the predicate evaluator: `List Nat → List Nat → Bool`.
fn pmot() -> Expr {
    Expr::arrow(list_nat(), Expr::arrow(list_nat(), boolc()))
}

// ===========================================================================
// The deep syntax + evaluator declarations (admitted through CHECKED paths only).
// ===========================================================================

/// `TyReflectPExp` — deep scalar value syntax. Constructor order is [`PEXP_CTORS`].
fn pexp_decl() -> InductiveDecl {
    let e = || c(PEXP);
    let un = |name: &str| Constructor {
        name: nm(name),
        type_: Expr::arrow(nat(), e()),
    };
    let bin = |name: &str| Constructor {
        name: nm(name),
        type_: Expr::arrow(e(), Expr::arrow(e(), e())),
    };
    InductiveDecl {
        level_params: vec![],
        num_params: 0,
        types: vec![InductiveType {
            name: nm(PEXP),
            type_: Expr::type_(),
            constructors: vec![
                un(PEXP_LIT),   // lit   : Nat → PExp     (literal value)
                un(PEXP_VAR),   // var   : Nat → PExp     (current-state column index)
                un(PEXP_PRIME), // prime : Nat → PExp     (next-state column index)
                bin(PEXP_ADD),
                bin(PEXP_MUL),
                bin(PEXP_DIV),
                bin(PEXP_MOD),
                bin(PEXP_SUB),
                bin(PEXP_LOR),  // bitmask ∪  (Nat.lor)
                bin(PEXP_LAND), // bitmask ∩  (Nat.land)
                bin(PEXP_SHR),  // bit test   (Nat.shiftRight)
            ],
        }],
    }
}

/// `TyReflectPPred` — deep scalar predicate syntax. Constructor order is [`PPRED_CTORS`].
fn ppred_decl() -> InductiveDecl {
    let p = || c(PPRED);
    let e = || c(PEXP);
    let pbin = |name: &str| Constructor {
        name: nm(name),
        type_: Expr::arrow(p(), Expr::arrow(p(), p())),
    };
    let cmp = |name: &str| Constructor {
        name: nm(name),
        type_: Expr::arrow(e(), Expr::arrow(e(), p())),
    };
    InductiveDecl {
        level_params: vec![],
        num_params: 0,
        types: vec![InductiveType {
            name: nm(PPRED),
            type_: Expr::type_(),
            constructors: vec![
                pbin(PPRED_AND),
                pbin(PPRED_OR),
                Constructor {
                    name: nm(PPRED_NOT),
                    type_: Expr::arrow(p(), p()),
                },
                pbin(PPRED_IMPLIES),
                pbin(PPRED_EQUIV),
                Constructor {
                    name: nm(PPRED_BOOLLIT),
                    type_: Expr::arrow(boolc(), p()),
                },
                cmp(PPRED_EQ),
                cmp(PPRED_NEQ),
                cmp(PPRED_LT),
                cmp(PPRED_LEQ),
                cmp(PPRED_GT),
                cmp(PPRED_GEQ),
                Constructor {
                    name: nm(PPRED_UNCHANGED),
                    type_: Expr::arrow(nat(), p()),
                },
                // inRange x lo hi — interval membership `x ∈ lo..hi` (AST-direct fragment).
                Constructor {
                    name: nm(PPRED_INRANGE),
                    type_: Expr::arrow(e(), Expr::arrow(e(), Expr::arrow(e(), p()))),
                },
                // eqIte l c t f — the pinned conditional update `l = IF c THEN t ELSE f`
                // (AST-direct fragment). The condition `c` is a RECURSIVE PPred argument
                // (positive position), so the derived recursor supplies its IH.
                Constructor {
                    name: nm(PPRED_EQITE),
                    type_: Expr::arrow(
                        e(),
                        Expr::arrow(p(), Expr::arrow(e(), Expr::arrow(e(), p()))),
                    ),
                },
            ],
        }],
    }
}

/// `TyReflectNth : List Nat → Nat → Nat` — a `List.rec` fold into an index function.
/// `nth [] i = 0` (the honest total-function default — see the module docs boundary note);
/// `nth (h :: t) i = if i == 0 then h else nth t (i∸1)`, the branch via `Bool.rec` on
/// `Nat.beq i 0` (minor premises in constructor order: `false` case first, then `true`).
fn nth_def() -> Declaration {
    let nn = || Expr::arrow(nat(), nat());
    // cons minor: fun (h : Nat) (t : List Nat) (ih : Nat → Nat) (i : Nat) =>
    //   Bool.rec.{1} (fun _ : Bool => Nat) (ih (Nat.sub i 1)) h (Nat.beq i 0)
    // de Bruijn: h=3 t=2 ih=1 i=0.
    let cons_case = lam(
        nat(),
        lam(
            list_nat(),
            lam(
                nn(),
                lam(
                    nat(),
                    ap(
                        Expr::const_str_levels("Bool.rec", vec![Level::succ(Level::zero())]),
                        [
                            lam(boolc(), nat()),
                            ap(bv(1), [ap(c("Nat.sub"), [bv(0), nl(1)])]), // false: ih (i∸1)
                            bv(3),                                         // true:  h
                            ap(c("Nat.beq"), [bv(0), nl(0)]),              // major: i == 0
                        ],
                    ),
                ),
            ),
        ),
    );
    // List.rec levels are [motive_level, elem_level]; Nat motive `Nat → Nat : Type` and
    // `List.{0} Nat` give `List.rec.{1,0}`. Argument order: α, motive, nil, cons, major.
    let value = lam(
        list_nat(),
        ap(
            Expr::const_str_levels("List.rec", vec![Level::succ(Level::zero()), Level::zero()]),
            [
                nat(),
                lam(list_nat(), nn()), // motive: fun _ : List Nat => Nat → Nat
                lam(nat(), nl(0)),     // nil: every index ⇒ default 0
                cons_case,
                bv(0),
            ],
        ),
    );
    Declaration::Definition {
        name: nm(NTH),
        level_params: vec![],
        type_: Expr::arrow(list_nat(), Expr::arrow(nat(), nat())),
        value,
        is_reducible: true,
    }
}

/// `TyReflectEvalV : TyReflectPExp → List Nat → List Nat → Nat` via the derived recursor.
/// Minor premises in [`PEXP_CTORS`] order, mirroring [`crate::cleancic::embed_val_ir`]:
/// `lit n ↦ n`, `var i ↦ nth s i`, `prime i ↦ nth sp i`, and `Nat.{add,mul,div,mod,sub,lor,land,
/// shiftRight}` for the binary nodes (the last three are the SET-fragment bitmask ops).
fn evalv_def() -> Declaration {
    // unary-Nat minors: fun (x : Nat) (s sp : List Nat) => body   (x=2 s=1 sp=0)
    let un = |body: Expr| lam(nat(), lam(list_nat(), lam(list_nat(), body)));
    // binary minors: fun (a b : PExp) (iha ihb : List Nat → List Nat → Nat) (s sp) =>
    //   op (iha s sp) (ihb s sp)          (a=5 b=4 iha=3 ihb=2 s=1 sp=0)
    let bin = |op: &str| {
        lam(
            c(PEXP),
            lam(
                c(PEXP),
                lam(
                    vmot(),
                    lam(
                        vmot(),
                        lam(
                            list_nat(),
                            lam(
                                list_nat(),
                                ap(
                                    c(op),
                                    [ap(bv(3), [bv(1), bv(0)]), ap(bv(2), [bv(1), bv(0)])],
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        )
    };
    let value = lam(
        c(PEXP),
        ap(
            c1(PEXP_REC),
            [
                lam(c(PEXP), vmot()),           // motive
                un(bv(2)),                      // lit n     ↦ n
                un(ap(c(NTH), [bv(1), bv(2)])), // var i     ↦ nth s i
                un(ap(c(NTH), [bv(0), bv(2)])), // prime i   ↦ nth sp i
                bin("Nat.add"),
                bin("Nat.mul"),
                bin("Nat.div"),
                bin("Nat.mod"),
                bin("Nat.sub"),
                bin("Nat.lor"),        // lor  a b ↦ Nat.lor  (bitmask ∪)
                bin("Nat.land"),       // land a b ↦ Nat.land (bitmask ∩)
                bin("Nat.shiftRight"), // shr  a b ↦ Nat.shiftRight (bit test / popcount)
                bv(0),                 // major: e
            ],
        ),
    );
    Declaration::Definition {
        name: nm(EVALV),
        level_params: vec![],
        type_: Expr::arrow(c(PEXP), vmot()),
        value,
        is_reducible: true,
    }
}

/// `TyReflectEvalP : TyReflectPPred → List Nat → List Nat → Bool` via the derived recursor.
/// Minor premises in [`PPRED_CTORS`] order, mirroring [`crate::cleancic::embed_pred_ir`]'s op
/// choices EXACTLY: `Lt(a,b) ↦ Nat.ble (a+1) b`, `Gt` flipped, `Implies ↦ ¬a ∨ b`,
/// `Equiv ↦ (a∧b)∨(¬a∧¬b)`, `Neq ↦ Bool.not ∘ Nat.beq`, `Unchanged i ↦ Nat.beq (nth sp i)
/// (nth s i)`. The two APPENDED AST-direct ctors have NO shallow counterpart (the IR quoter
/// never emits them): `inRange x lo hi ↦ Nat.ble ⟦lo⟧ ⟦x⟧ ∧ Nat.ble ⟦x⟧ ⟦hi⟧` and
/// `eqIte l c t f ↦ Nat.beq ⟦l⟧ (Bool.rec (fun _ => Nat) ⟦f⟧ ⟦t⟧ ⟦c⟧)`.
fn evalp_def() -> Declaration {
    let bnot = |a: Expr| ap(c("Bool.not"), [a]);
    // recursive-binary minors: fun (p q : PPred) (ihp ihq : … → Bool) (s sp) => body
    // (p=5 q=4 ihp=3 ihq=2 s=1 sp=0)
    let pbin = |body: Expr| {
        lam(
            c(PPRED),
            lam(
                c(PPRED),
                lam(pmot(), lam(pmot(), lam(list_nat(), lam(list_nat(), body)))),
            ),
        )
    };
    let ihp = || ap(bv(3), [bv(1), bv(0)]);
    let ihq = || ap(bv(2), [bv(1), bv(0)]);
    // comparison minors: fun (a b : PExp) (s sp) => body   (a=3 b=2 s=1 sp=0; PExp args are
    // non-recursive in PPred, so the recursor supplies no IHs for them)
    let cmp = |body: Expr| {
        lam(
            c(PEXP),
            lam(c(PEXP), lam(list_nat(), lam(list_nat(), body))),
        )
    };
    let va = || ap(c(EVALV), [bv(3), bv(1), bv(0)]);
    let vb = || ap(c(EVALV), [bv(2), bv(1), bv(0)]);
    let inc = |e: Expr| ap(c("Nat.add"), [e, nl(1)]);
    let value = lam(
        c(PPRED),
        ap(
            c1(PPRED_REC),
            [
                lam(c(PPRED), pmot()),                   // motive
                pbin(ap(c("Bool.and"), [ihp(), ihq()])), // and
                pbin(ap(c("Bool.or"), [ihp(), ihq()])),  // or
                // not: fun (p : PPred) (ihp) (s sp) => Bool.not (ihp s sp)   (ihp=2 s=1 sp=0)
                lam(
                    c(PPRED),
                    lam(
                        pmot(),
                        lam(list_nat(), lam(list_nat(), bnot(ap(bv(2), [bv(1), bv(0)])))),
                    ),
                ),
                pbin(ap(c("Bool.or"), [bnot(ihp()), ihq()])), // implies ↦ ¬a ∨ b
                pbin(ap(
                    c("Bool.or"),
                    [
                        ap(c("Bool.and"), [ihp(), ihq()]),
                        ap(c("Bool.and"), [bnot(ihp()), bnot(ihq())]),
                    ],
                )), // equiv ↦ (a∧b)∨(¬a∧¬b)
                // boolLit: fun (b : Bool) (s sp) => b   (b=2)
                lam(boolc(), lam(list_nat(), lam(list_nat(), bv(2)))),
                cmp(ap(c("Nat.beq"), [va(), vb()])),       // eq
                cmp(bnot(ap(c("Nat.beq"), [va(), vb()]))), // neq
                cmp(ap(c("Nat.ble"), [inc(va()), vb()])),  // lt  ↦ a+1 ≤ b
                cmp(ap(c("Nat.ble"), [va(), vb()])),       // leq
                cmp(ap(c("Nat.ble"), [inc(vb()), va()])),  // gt  ↦ b+1 ≤ a
                cmp(ap(c("Nat.ble"), [vb(), va()])),       // geq
                // unchanged: fun (i : Nat) (s sp) => Nat.beq (nth sp i) (nth s i)   (i=2 s=1 sp=0)
                lam(
                    nat(),
                    lam(
                        list_nat(),
                        lam(
                            list_nat(),
                            ap(
                                c("Nat.beq"),
                                [ap(c(NTH), [bv(0), bv(2)]), ap(c(NTH), [bv(1), bv(2)])],
                            ),
                        ),
                    ),
                ),
                // inRange: fun (x lo hi : PExp) (s sp) =>
                //   Bool.and (Nat.ble (EvalV lo s sp) (EvalV x s sp))
                //            (Nat.ble (EvalV x s sp) (EvalV hi s sp))
                // — `x ∈ lo..hi ↦ lo ≤ x ∧ x ≤ hi`, the SAME `Nat.ble` realization the `leq`
                // arm uses (an empty TLA+ interval `lo > hi` is unsatisfiable both ways).
                // de Bruijn: x=4 lo=3 hi=2 s=1 sp=0 (PExp args are non-recursive: no IHs).
                {
                    let evx = || ap(c(EVALV), [bv(4), bv(1), bv(0)]);
                    let evlo = ap(c(EVALV), [bv(3), bv(1), bv(0)]);
                    let evhi = ap(c(EVALV), [bv(2), bv(1), bv(0)]);
                    lam(
                        c(PEXP),
                        lam(
                            c(PEXP),
                            lam(
                                c(PEXP),
                                lam(
                                    list_nat(),
                                    lam(
                                        list_nat(),
                                        ap(
                                            c("Bool.and"),
                                            [
                                                ap(c("Nat.ble"), [evlo, evx()]),
                                                ap(c("Nat.ble"), [evx(), evhi]),
                                            ],
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    )
                },
                // eqIte: fun (l : PExp) (c : PPred) (t f : PExp) (ihc : … → Bool) (s sp) =>
                //   Nat.beq (EvalV l s sp)
                //           (Bool.rec.{1} (fun _ : Bool => Nat) (EvalV f s sp) (EvalV t s sp) (ihc s sp))
                // — `l = IF c THEN t ELSE f ↦ Nat.beq ⟦l⟧ (if ⟦c⟧ then ⟦t⟧ else ⟦f⟧)`, the branch
                // via `Bool.rec` on the condition's IH (minor order: `false` case first, as in `nth`).
                // de Bruijn: l=6 c=5 t=4 f=3 ihc=2 s=1 sp=0 (ctor fields first, then the one IH).
                {
                    let evl = ap(c(EVALV), [bv(6), bv(1), bv(0)]);
                    let evt = ap(c(EVALV), [bv(4), bv(1), bv(0)]);
                    let evf = ap(c(EVALV), [bv(3), bv(1), bv(0)]);
                    let cond = ap(bv(2), [bv(1), bv(0)]);
                    lam(
                        c(PEXP),
                        lam(
                            c(PPRED),
                            lam(
                                c(PEXP),
                                lam(
                                    c(PEXP),
                                    lam(
                                        pmot(),
                                        lam(
                                            list_nat(),
                                            lam(
                                                list_nat(),
                                                ap(
                                                    c("Nat.beq"),
                                                    [
                                                        evl,
                                                        ap(
                                                            Expr::const_str_levels(
                                                                "Bool.rec",
                                                                vec![Level::succ(Level::zero())],
                                                            ),
                                                            [lam(boolc(), nat()), evf, evt, cond],
                                                        ),
                                                    ],
                                                ),
                                            ),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    )
                },
                bv(0), // major: p
            ],
        ),
    );
    Declaration::Definition {
        name: nm(EVALP),
        level_params: vec![],
        type_: Expr::arrow(c(PPRED), pmot()),
        value,
        is_reducible: true,
    }
}

// ===========================================================================
// The quoted-tuple-set membership defs (R2 applied to the fixpoint legs):
// TupEq / Mem / AllMem / Subseq. All CHECKED admissions (`add_decl`).
// ===========================================================================

/// `List.{0} (List.{0} Nat) : Type` — the type of a quoted tuple SET.
fn list_list_nat() -> Expr {
    Expr::app(
        Expr::const_str_levels("List", vec![Level::zero()]),
        list_nat(),
    )
}

/// `List.rec.{1,0}` over element type `alpha` at the CONSTANT `Bool` motive:
/// `List.rec.{1,0} alpha (fun _ : List alpha => Bool) nil_case cons_case major`.
fn list_rec_bool(alpha: Expr, nil_case: Expr, cons_case: Expr, major: Expr) -> Expr {
    let list_alpha = Expr::app(
        Expr::const_str_levels("List", vec![Level::zero()]),
        alpha.clone(),
    );
    ap(
        Expr::const_str_levels("List.rec", vec![Level::succ(Level::zero()), Level::zero()]),
        [alpha, lam(list_alpha, boolc()), nil_case, cons_case, major],
    )
}

/// `TyReflectTupEq : List Nat → List Nat → Bool` — structural equality by a `List.rec` fold:
/// `tupEq [] ys = isNil ys`; `tupEq (x::t) ys = match ys with [] => false | y::t2 =>
/// Nat.beq x y && tupEq t t2`. LENGTH-SENSITIVE by construction (the nil/cons cross cases are
/// `false`), so equal-prefix tuples of different arities never compare equal.
fn tupeq_def() -> Declaration {
    // nil outer case: fun (ys : List Nat) => List.rec (fun _ => Bool) true (fun _ _ _ => false) ys
    let is_nil_nat = lam(
        list_nat(),
        list_rec_bool(
            nat(),
            c("Bool.true"),
            lam(nat(), lam(list_nat(), lam(boolc(), c("Bool.false")))),
            bv(0),
        ),
    );
    // cons outer case: fun (x : Nat) (t : List Nat) (ih : List Nat → Bool) (ys : List Nat) =>
    //   List.rec (fun _ => Bool) false
    //     (fun (y : Nat) (t2 : List Nat) (_ih2 : Bool) => Bool.and (Nat.beq x y) (ih t2)) ys
    // de Bruijn inside the innermost lambda: _ih2=0 t2=1 y=2 ys=3 ih=4 t=5 x=6.
    let cons_case = lam(
        nat(),
        lam(
            list_nat(),
            lam(
                Expr::arrow(list_nat(), boolc()),
                lam(
                    list_nat(),
                    list_rec_bool(
                        nat(),
                        c("Bool.false"),
                        lam(
                            nat(),
                            lam(
                                list_nat(),
                                lam(
                                    boolc(),
                                    ap(
                                        c("Bool.and"),
                                        [ap(c("Nat.beq"), [bv(6), bv(2)]), ap(bv(4), [bv(1)])],
                                    ),
                                ),
                            ),
                        ),
                        bv(0),
                    ),
                ),
            ),
        ),
    );
    // value = fun (xs : List Nat) => List.rec.{1,0} Nat (fun _ => List Nat → Bool) nil cons xs
    let value = lam(
        list_nat(),
        ap(
            Expr::const_str_levels("List.rec", vec![Level::succ(Level::zero()), Level::zero()]),
            [
                nat(),
                lam(list_nat(), Expr::arrow(list_nat(), boolc())), // motive
                is_nil_nat,
                cons_case,
                bv(0),
            ],
        ),
    );
    Declaration::Definition {
        name: nm(TUPEQ),
        level_params: vec![],
        type_: Expr::arrow(list_nat(), Expr::arrow(list_nat(), boolc())),
        value,
        is_reducible: true,
    }
}

/// `TyReflectMem : List Nat → List (List Nat) → Bool` — `mem x [] = false`;
/// `mem x (h::t) = TyReflectTupEq x h || mem x t` (a `List.rec` fold over the SET).
fn mem_def() -> Declaration {
    // cons case: fun (h : List Nat) (t : List (List Nat)) (ih : Bool) =>
    //   Bool.or (TyReflectTupEq x h) ih          (ih=0 t=1 h=2 s=3 x=4)
    let cons_case = lam(
        list_nat(),
        lam(
            list_list_nat(),
            lam(
                boolc(),
                ap(c("Bool.or"), [ap(c(TUPEQ), [bv(4), bv(2)]), bv(0)]),
            ),
        ),
    );
    let value = lam(
        list_nat(),
        lam(
            list_list_nat(),
            list_rec_bool(list_nat(), c("Bool.false"), cons_case, bv(0)),
        ),
    );
    Declaration::Definition {
        name: nm(MEM),
        level_params: vec![],
        type_: Expr::arrow(list_nat(), Expr::arrow(list_list_nat(), boolc())),
        value,
        is_reducible: true,
    }
}

/// `TyReflectAllMem : List (List Nat) → List (List Nat) → Bool` — `allMem [] ys = true`;
/// `allMem (h::t) ys = TyReflectMem h ys && allMem t ys`. The semantic reference for the
/// reflected `⊆` claim (O(|xs|·|ys|) reduction — see [`SUBSEQ`] for the live merge form).
fn allmem_def() -> Declaration {
    // cons case: fun (h : List Nat) (t : List (List Nat)) (ih : Bool) =>
    //   Bool.and (TyReflectMem h ys) ih          (ih=0 t=1 h=2 ys=3 xs=4)
    let cons_case = lam(
        list_nat(),
        lam(
            list_list_nat(),
            lam(
                boolc(),
                ap(c("Bool.and"), [ap(c(MEM), [bv(2), bv(3)]), bv(0)]),
            ),
        ),
    );
    let value = lam(
        list_list_nat(),
        lam(
            list_list_nat(),
            list_rec_bool(list_nat(), c("Bool.true"), cons_case, bv(1)), // major = xs
        ),
    );
    Declaration::Definition {
        name: nm(ALLMEM),
        level_params: vec![],
        type_: Expr::arrow(list_list_nat(), Expr::arrow(list_list_nat(), boolc())),
        value,
        is_reducible: true,
    }
}

/// `TyReflectSubseq : List (List Nat) → List (List Nat) → Bool` — the sorted-merge subset
/// check: recursion on `ys` with the motive `List (List Nat) → Bool` (a function of the
/// REMAINING `xs`):
/// `subseq' [] rem = isNil rem`;
/// `subseq' (y::t) rem = match rem with [] => true | x::xt => if TyReflectTupEq x y then
/// subseq' t xt else subseq' t (x::xt)`.
/// `Subseq xs ys = subseq' ys xs`. A `true` verdict exhibits an order-preserving injection of
/// `xs` into `ys` with `TupEq` matches — every `xs`-member IS a `ys`-member — sound for `⊆`
/// unconditionally; completeness (a genuine subset evaluating `true`) holds whenever `xs`,`ys`
/// are canonicalized to the SAME sort order, which every caller does.
fn subseq_def() -> Declaration {
    // inner match on rem, under (y t ih rem): innermost cons lambda binds x=2 xt=1 _ih2=0;
    // outer slots at that depth: rem=3 ih=4 t=5 y=6.
    let inner_cons = lam(
        list_nat(),
        lam(
            list_list_nat(),
            lam(
                boolc(),
                ap(
                    Expr::const_str_levels("Bool.rec", vec![Level::succ(Level::zero())]),
                    [
                        lam(boolc(), boolc()),
                        ap(bv(4), [bv(3)]), // false: ih rem   (keep x, drop y)
                        ap(bv(4), [bv(1)]), // true:  ih xt    (matched — drop both)
                        ap(c(TUPEQ), [bv(2), bv(6)]), // major: tupEq x y
                    ],
                ),
            ),
        ),
    );
    // ys-cons case: fun (y : List Nat) (t : List (List Nat)) (ih : List (List Nat) → Bool)
    //               (rem : List (List Nat)) => List.rec (fun _ => Bool) true inner_cons rem
    let ys_cons = lam(
        list_nat(),
        lam(
            list_list_nat(),
            lam(
                Expr::arrow(list_list_nat(), boolc()),
                lam(
                    list_list_nat(),
                    list_rec_bool(list_nat(), c("Bool.true"), inner_cons, bv(0)),
                ),
            ),
        ),
    );
    // ys-nil case: fun (rem : List (List Nat)) => isNil rem
    let ys_nil = lam(
        list_list_nat(),
        list_rec_bool(
            list_nat(),
            c("Bool.true"),
            lam(
                list_nat(),
                lam(list_list_nat(), lam(boolc(), c("Bool.false"))),
            ),
            bv(0),
        ),
    );
    // value = fun (xs ys : List (List Nat)) =>
    //   (List.rec.{1,0} (List Nat) (fun _ => List (List Nat) → Bool) ys_nil ys_cons ys) xs
    let value = lam(
        list_list_nat(),
        lam(
            list_list_nat(),
            ap(
                ap(
                    Expr::const_str_levels(
                        "List.rec",
                        vec![Level::succ(Level::zero()), Level::zero()],
                    ),
                    [
                        list_nat(),
                        lam(list_list_nat(), Expr::arrow(list_list_nat(), boolc())), // motive
                        ys_nil,
                        ys_cons,
                        bv(0), // major: ys
                    ],
                ),
                [bv(1)], // applied to xs
            ),
        ),
    );
    Declaration::Definition {
        name: nm(SUBSEQ),
        level_params: vec![],
        type_: Expr::arrow(list_list_nat(), Expr::arrow(list_list_nat(), boolc())),
        value,
        is_reducible: true,
    }
}

// ===========================================================================
// The cached reflect environment + its trust ledger.
// ===========================================================================

/// How each `TyReflect*` name entered the env — the auditable ledger of this lane's entire
/// admission surface. There is deliberately NO variant for a structural/axiom admission:
/// the builder only ever calls `add_inductive` and `add_decl` (both checked paths).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReflectLedgerEntry {
    /// Admitted via `Environment::add_inductive` (positivity/universe checks + recursor
    /// derivation all done BY the kernel).
    Inductive,
    /// Admitted via `Environment::add_decl` as a `Definition` — the body was kernel-checked
    /// against the stated type before registration.
    CheckedDef,
}

/// The prelude env extended with the deep syntax + evaluators, plus the admission ledger.
pub struct ReflectEnv {
    env: Environment,
    /// (dotted name, how it entered). Exactly the `TyReflect*` names; nothing structural.
    pub ledger: Vec<(String, ReflectLedgerEntry)>,
}

static REFLECT_ENV: OnceLock<Option<ReflectEnv>> = OnceLock::new();

/// The process-wide cached env; `None` if any admission failed (every corroboration is then
/// `Unavailable` — fail-closed, never a silently-weaker env).
fn cached_env() -> Option<&'static ReflectEnv> {
    REFLECT_ENV.get_or_init(build_env).as_ref()
}

/// Test/report hook: the admission ledger of the process-wide reflect env.
pub fn env_ledger() -> Option<&'static [(String, ReflectLedgerEntry)]> {
    cached_env().map(|e| e.ledger.as_slice())
}

/// Build the reflect env: prelude + deep syntax + evaluators, every admission through a
/// CHECKED kernel path (`add_inductive` / `add_decl`; never `add_decl_structural`). Any
/// failure yields `None` (fail-closed).
fn build_env() -> Option<ReflectEnv> {
    // Clone the process-wide cached prelude (never mutate the shared build): the checked
    // admissions below extend this private copy only. Clone is faithful — same decls,
    // structurally shared `Arc` children.
    let mut env = crate::cleancic::prelude_env().clone();
    let mut ledger: Vec<(String, ReflectLedgerEntry)> = Vec::new();
    let admit_ind = |env: &mut Environment,
                     ledger: &mut Vec<(String, ReflectLedgerEntry)>,
                     decl: InductiveDecl,
                     type_name: &str,
                     ctors: &[&str]| {
        if let Err(e) = env.add_inductive(decl) {
            debug_assert!(false, "reflect env: `{type_name}` admission failed: {e:?}");
            return None;
        }
        ledger.push((type_name.to_string(), ReflectLedgerEntry::Inductive));
        for ctor in ctors {
            ledger.push(((*ctor).to_string(), ReflectLedgerEntry::Inductive));
        }
        Some(())
    };
    admit_ind(&mut env, &mut ledger, pexp_decl(), PEXP, &PEXP_CTORS)?;
    admit_ind(&mut env, &mut ledger, ppred_decl(), PPRED, &PPRED_CTORS)?;
    // Definition order matters: EvalV mentions Nth, EvalP mentions EvalV; Mem/AllMem/Subseq
    // mention TupEq (Mem before AllMem).
    for (name, decl) in [
        (NTH, nth_def()),
        (EVALV, evalv_def()),
        (EVALP, evalp_def()),
        (TUPEQ, tupeq_def()),
        (MEM, mem_def()),
        (ALLMEM, allmem_def()),
        (SUBSEQ, subseq_def()),
    ] {
        if let Err(e) = env.add_decl(decl) {
            debug_assert!(false, "reflect env: `{name}` definition failed: {e:?}");
            return None;
        }
        ledger.push((name.to_string(), ReflectLedgerEntry::CheckedDef));
    }
    Some(ReflectEnv { env, ledger })
}

// ===========================================================================
// The QUOTER — the only per-obligation trusted piece. Constructor-for-constructor,
// one match arm per IR node, NO logic beyond the 1:1 mapping. Fail-closed (`None`)
// on every out-of-scope arm.
// ===========================================================================

/// Quote a [`SetIR`] as a deep `TyReflectPExp` BITMASK term (a `Nat`), mirroring the shallow
/// [`crate::cleancic::embed_set_ir`]'s op choices EXACTLY: a bitmask Set/SetMask/SetMaskRec column
/// cell is already the bitmask `Nat` (`var i`/`prime i`), `∪`=`lor`, `∩`=`land`, and a `FuncSetMask`
/// `Digit` is the positional digit `mod(div pack place) base` (the same `Nat.div`/`Nat.mod` the
/// `Record`/`Func` value recognizers use). `None` for the out-of-scope `Filter` comprehension (its
/// per-element bound-var fold is not in v2's scope — and `set_exact` already declines `Filter`, so no
/// COVERED obligation reaches this arm).
fn quote_set(ir: &SetIR) -> Option<Expr> {
    Some(match ir {
        SetIR::Lit(m) => ap(c(PEXP_LIT), [nl(*m)]),
        SetIR::Var(i) => ap(c(PEXP_VAR), [nl(u64::try_from(*i).ok()?)]),
        SetIR::Prime(i) => ap(c(PEXP_PRIME), [nl(u64::try_from(*i).ok()?)]),
        SetIR::Cup(a, b) => ap(c(PEXP_LOR), [quote_set(a)?, quote_set(b)?]),
        SetIR::Cap(a, b) => ap(c(PEXP_LAND), [quote_set(a)?, quote_set(b)?]),
        // `f[k]` set-mask DIGIT `(pack / place) mod base` — EXACTLY the shallow `embed_set_ir` op.
        SetIR::Digit { pack, place, base } => ap(
            c(PEXP_MOD),
            [
                ap(
                    c(PEXP_DIV),
                    [quote_val(pack)?, ap(c(PEXP_LIT), [nl(*place)])],
                ),
                ap(c(PEXP_LIT), [nl(*base)]),
            ],
        ),
        SetIR::Filter { .. } => return None,
    })
}

/// The deep bit-`e`-of-`mask` `Nat` value `land(shr(mask, e), 1)` — reduces to `0` or `1` — mirroring
/// the shallow [`crate::cleancic`]'s `set_mem_bit` core `Nat.land(Nat.shiftRight mask e, 1)`.
fn set_bit(mask: Expr, e: u64) -> Expr {
    ap(
        c(PEXP_LAND),
        [
            ap(c(PEXP_SHR), [mask, ap(c(PEXP_LIT), [nl(e)])]),
            ap(c(PEXP_LIT), [nl(1)]),
        ],
    )
}

/// Quote `boolToNat(P)` as a deep `TyReflectPExp` term reducing to EXACTLY `1` (P true) / `0` (P false),
/// via truncated-`Nat` arithmetic IDENTITIES over the SAME deep value/set quoters. This is an
/// INDEPENDENT realization of the shallow `bool_to_nat(embed_pred_ir P)` (which lifts a `Bool` with
/// `Bool.rec`) — the two op-realizations must agree, so a wrong identity here is caught by the
/// exactness cross-check + the definitional agreement tests, not rubber-stamped. Used ONLY by the
/// `CountFold` counting sum `Σ_d boolToNat(P(d))`. `None` on any out-of-fragment leaf (a bounded
/// quantifier), so a `CountFold` carrying such a term stays Uncovered.
///
/// The identities (over truncated `Nat`, `∸` = `Nat.sub`): for a Nat "distance" `d`, `1 ∸ d` is `1`
/// iff `d = 0`. `eqInd(x,y) = 1 ∸ ((x∸y)+(y∸x))` is `1` iff `x = y`; `x ≤ y` iff `x∸y = 0`; `x < y`
/// iff `(x+1)∸y = 0`. Boolean combinators fold the 0/1 indicators (`∧`=`·`, `¬b`=`1∸b`, `∨` by De
/// Morgan). Each indicator is provably `0`/`1`, so the sums count satisfied elements EXACTLY.
fn quote_bool_to_nat(p: &PredIR) -> Option<Expr> {
    let one = || ap(c(PEXP_LIT), [nl(1)]);
    let add = |a: Expr, b: Expr| ap(c(PEXP_ADD), [a, b]);
    let mul = |a: Expr, b: Expr| ap(c(PEXP_MUL), [a, b]);
    let sub = |a: Expr, b: Expr| ap(c(PEXP_SUB), [a, b]);
    let lor = |a: Expr, b: Expr| ap(c(PEXP_LOR), [a, b]);
    // eqInd(x,y) = 1 ∸ ((x∸y)+(y∸x)) — the 0/1 indicator of `x = y`.
    let eq_ind = |x: Expr, y: Expr| sub(one(), add(sub(x.clone(), y.clone()), sub(y, x)));
    // colIdx(i) as (prime, current) column value terms.
    let cols = |i: &usize| -> Option<(Expr, Expr)> {
        let j = u64::try_from(*i).ok()?;
        Some((ap(c(PEXP_PRIME), [nl(j)]), ap(c(PEXP_VAR), [nl(j)])))
    };
    Some(match p {
        PredIR::BoolLit(b) => ap(c(PEXP_LIT), [nl(u64::from(*b))]),
        PredIR::Not(a) => sub(one(), quote_bool_to_nat(a)?),
        PredIR::And(a, b) => mul(quote_bool_to_nat(a)?, quote_bool_to_nat(b)?),
        // ∨ ≡ ¬(¬a ∧ ¬b): 1 ∸ ((1∸ba)·(1∸bb)).
        PredIR::Or(a, b) => sub(
            one(),
            mul(
                sub(one(), quote_bool_to_nat(a)?),
                sub(one(), quote_bool_to_nat(b)?),
            ),
        ),
        // a ⇒ b ≡ ¬(a ∧ ¬b): 1 ∸ (ba·(1∸bb)).
        PredIR::Implies(a, b) => sub(
            one(),
            mul(quote_bool_to_nat(a)?, sub(one(), quote_bool_to_nat(b)?)),
        ),
        PredIR::Equiv(a, b) => eq_ind(quote_bool_to_nat(a)?, quote_bool_to_nat(b)?),
        PredIR::Eq(a, b) => eq_ind(quote_val(a)?, quote_val(b)?),
        PredIR::Neq(a, b) => sub(one(), eq_ind(quote_val(a)?, quote_val(b)?)),
        PredIR::Lt(a, b) => sub(one(), sub(add(quote_val(a)?, one()), quote_val(b)?)),
        PredIR::Leq(a, b) => sub(one(), sub(quote_val(a)?, quote_val(b)?)),
        PredIR::Gt(a, b) => sub(one(), sub(add(quote_val(b)?, one()), quote_val(a)?)),
        PredIR::Geq(a, b) => sub(one(), sub(quote_val(b)?, quote_val(a)?)),
        PredIR::Unchanged(i) | PredIR::SetUnchanged(i) => {
            let (sp_i, s_i) = cols(i)?;
            eq_ind(sp_i, s_i)
        }
        PredIR::SetEq(a, b) => eq_ind(quote_set(a)?, quote_set(b)?),
        PredIR::SetNeq(a, b) => sub(one(), eq_ind(quote_set(a)?, quote_set(b)?)),
        PredIR::SetMem(e, set) => set_bit(quote_set(set)?, *e),
        PredIR::SetNotMem(e, set) => sub(one(), set_bit(quote_set(set)?, *e)),
        PredIR::SetSubseteq(a, b) => {
            let (qa, qb) = (quote_set(a)?, quote_set(b)?);
            eq_ind(lor(qa, qb.clone()), qb)
        }
        PredIR::SetForall { .. }
        | PredIR::SetExists { .. }
        | PredIR::SubsetForall { .. }
        | PredIR::SubsetExists { .. } => return None,
    })
}

/// Quote a [`ValIR`] as a deep `TyReflectPExp` term. Covers the scalar arithmetic fragment PLUS the
/// SET-valued counting nodes — `SetCard` (bitmask popcount) and `CountFold` (`Σ boolToNat` over a
/// fixed domain). `None` for the out-of-scope `Seq*` arms.
pub fn quote_val(ir: &ValIR) -> Option<Expr> {
    Some(match ir {
        ValIR::Lit(v) => ap(c(PEXP_LIT), [nl(*v)]),
        ValIR::Var(i) => ap(c(PEXP_VAR), [nl(u64::try_from(*i).ok()?)]),
        ValIR::Prime(i) => ap(c(PEXP_PRIME), [nl(u64::try_from(*i).ok()?)]),
        ValIR::Add(a, b) => ap(c(PEXP_ADD), [quote_val(a)?, quote_val(b)?]),
        ValIR::Mul(a, b) => ap(c(PEXP_MUL), [quote_val(a)?, quote_val(b)?]),
        ValIR::Div(a, b) => ap(c(PEXP_DIV), [quote_val(a)?, quote_val(b)?]),
        ValIR::Mod(a, b) => ap(c(PEXP_MOD), [quote_val(a)?, quote_val(b)?]),
        ValIR::Sub(a, b) => ap(c(PEXP_SUB), [quote_val(a)?, quote_val(b)?]),
        ValIR::SeqLen { .. } | ValIR::SeqTail { .. } | ValIR::SeqAppend { .. } => return None,
        // Cardinality(S) = Σ_{i<universe} bit_i(mask). Each `bit_i` is already a 0/1 `Nat` (so
        // `boolToNat` is the identity), a left-nested `add` fold. universe=0 ⇒ the literal 0. Mirrors
        // the shallow `Σ boolToNat(set_mem_bit(i,mask))` EXACTLY (a faithful bitmask ⇒ set-bit count = |S|).
        ValIR::SetCard { set, universe } => {
            let mask = quote_set(set)?;
            let mut acc: Option<Expr> = None;
            for i in 0..u64::from(*universe) {
                let leg = set_bit(mask.clone(), i);
                acc = Some(match acc {
                    None => leg,
                    Some(a) => ap(c(PEXP_ADD), [a, leg]),
                });
            }
            acc.unwrap_or_else(|| ap(c(PEXP_LIT), [nl(0)]))
        }
        // Cardinality({d∈D:P(d)}) = Σ_{d∈D} boolToNat(P(d)) — a left-nested `add` fold of the per-element
        // 0/1 indicators. Empty domain ⇒ the literal 0. An out-of-fragment term ⇒ the whole fold declines.
        ValIR::CountFold { terms } => {
            let mut acc: Option<Expr> = None;
            for t in terms {
                let leg = quote_bool_to_nat(t)?;
                acc = Some(match acc {
                    None => leg,
                    Some(a) => ap(c(PEXP_ADD), [a, leg]),
                });
            }
            acc.unwrap_or_else(|| ap(c(PEXP_LIT), [nl(0)]))
        }
    })
}

/// Quote a [`PredIR`] as a deep `TyReflectPPred` term. Covers the scalar Boolean/comparison fragment
/// PLUS the bitmask SET predicates (`SetEq`/`SetNeq`/`SetMem`/`SetNotMem`/`SetSubseteq`/`SetUnchanged`,
/// each realized over the [`quote_set`] bitmask through the EXISTING `eq`/`neq`/`unchanged` evaluator
/// arms — no new `PPred` constructor). `None` for the bounded/`SUBSET` quantifier folds (`SetForall`,
/// …) — which `pred_exact` also rejects, so no COVERED obligation reaches those arms.
pub fn quote_pred(ir: &PredIR) -> Option<Expr> {
    Some(match ir {
        PredIR::And(a, b) => ap(c(PPRED_AND), [quote_pred(a)?, quote_pred(b)?]),
        PredIR::Or(a, b) => ap(c(PPRED_OR), [quote_pred(a)?, quote_pred(b)?]),
        PredIR::Not(a) => ap(c(PPRED_NOT), [quote_pred(a)?]),
        PredIR::Implies(a, b) => ap(c(PPRED_IMPLIES), [quote_pred(a)?, quote_pred(b)?]),
        PredIR::Equiv(a, b) => ap(c(PPRED_EQUIV), [quote_pred(a)?, quote_pred(b)?]),
        PredIR::BoolLit(b) => ap(
            c(PPRED_BOOLLIT),
            [c(if *b { "Bool.true" } else { "Bool.false" })],
        ),
        PredIR::Eq(a, b) => ap(c(PPRED_EQ), [quote_val(a)?, quote_val(b)?]),
        PredIR::Neq(a, b) => ap(c(PPRED_NEQ), [quote_val(a)?, quote_val(b)?]),
        PredIR::Lt(a, b) => ap(c(PPRED_LT), [quote_val(a)?, quote_val(b)?]),
        PredIR::Leq(a, b) => ap(c(PPRED_LEQ), [quote_val(a)?, quote_val(b)?]),
        PredIR::Gt(a, b) => ap(c(PPRED_GT), [quote_val(a)?, quote_val(b)?]),
        PredIR::Geq(a, b) => ap(c(PPRED_GEQ), [quote_val(a)?, quote_val(b)?]),
        PredIR::Unchanged(i) => ap(c(PPRED_UNCHANGED), [nl(u64::try_from(*i).ok()?)]),
        // ── SET fragment: bitmask predicates over the `quote_set` `Nat`, EXACTLY the shallow ops ──
        // S = T ↦ Nat.beq maskS maskT  (via the `eq` arm).
        PredIR::SetEq(a, b) => ap(c(PPRED_EQ), [quote_set(a)?, quote_set(b)?]),
        PredIR::SetNeq(a, b) => ap(c(PPRED_NEQ), [quote_set(a)?, quote_set(b)?]),
        // e ∈ S ↦ Nat.beq (land(shr maskS e) 1) 1  (the `eq` arm over the bit-test term).
        PredIR::SetMem(e, set) => ap(
            c(PPRED_EQ),
            [set_bit(quote_set(set)?, *e), ap(c(PEXP_LIT), [nl(1)])],
        ),
        PredIR::SetNotMem(e, set) => ap(
            c(PPRED_NEQ),
            [set_bit(quote_set(set)?, *e), ap(c(PEXP_LIT), [nl(1)])],
        ),
        // S ⊆ T ↦ Nat.beq (lor maskS maskT) maskT  (every bit of S is a bit of T).
        PredIR::SetSubseteq(a, b) => {
            let (qa, qb) = (quote_set(a)?, quote_set(b)?);
            ap(c(PPRED_EQ), [ap(c(PEXP_LOR), [qa, qb.clone()]), qb])
        }
        // UNCHANGED S ↦ Nat.beq sp[i] s[i]  (bitmask cell equality — the `unchanged` arm, EXACT).
        PredIR::SetUnchanged(i) => ap(c(PPRED_UNCHANGED), [nl(u64::try_from(*i).ok()?)]),
        PredIR::SetForall { .. }
        | PredIR::SetExists { .. }
        | PredIR::SubsetForall { .. }
        | PredIR::SubsetExists { .. } => return None,
    })
}

/// Quote a concrete state tuple as a `List.{0} Nat` cons chain of `Nat` literals.
pub fn quote_state(s: &[u64]) -> Expr {
    let nil = ap(
        Expr::const_str_levels("List.nil", vec![Level::zero()]),
        [nat()],
    );
    s.iter().rev().fold(nil, |acc, v| {
        ap(
            Expr::const_str_levels("List.cons", vec![Level::zero()]),
            [nat(), nl(*v), acc],
        )
    })
}

/// Quote a list of state tuples as a `List.{0} (List.{0} Nat)` cons chain — built ITERATIVELY
/// (a fold, not recursion in `|set|`: at the CoffeeCan scale the chain is thousands deep and a
/// recursive builder would risk the Rust stack; the kernel side is already stack-safe).
pub fn quote_state_set(set: &[Vec<u64>]) -> Expr {
    let nil = ap(
        Expr::const_str_levels("List.nil", vec![Level::zero()]),
        [list_nat()],
    );
    set.iter().rev().fold(nil, |acc, s| {
        ap(
            Expr::const_str_levels("List.cons", vec![Level::zero()]),
            [list_nat(), quote_state(s), acc],
        )
    })
}

// ===========================================================================
// Deep-constructor BUILDERS for the AST-DIRECT quoter (`crate::reflect_ast_direct`).
// Each builder is EXACTLY `ctor arg…` — constructor application, NO other logic —
// so the AST quoter's match arms read 1:1 (one AST node ↦ one deep constructor).
// ===========================================================================

pub(crate) mod deep {
    use super::*;

    pub(crate) fn lit(n: u64) -> Expr {
        ap(c(PEXP_LIT), [nl(n)])
    }
    pub(crate) fn var(i: u64) -> Expr {
        ap(c(PEXP_VAR), [nl(i)])
    }
    pub(crate) fn prime(i: u64) -> Expr {
        ap(c(PEXP_PRIME), [nl(i)])
    }
    pub(crate) fn add(a: Expr, b: Expr) -> Expr {
        ap(c(PEXP_ADD), [a, b])
    }
    /// Bounded subtraction `a - b`, evaluated by the kernel as TRUNCATING `Nat.sub` (see the
    /// `EvalV` `Nat.sub` arm). The AST-direct quoter (increment 5) admits this ctor ONLY for a
    /// literal minuend `c` minus a column subtrahend PROVED `≤ c` on every evaluated state, so
    /// the kernel's `Nat.sub c x` coincides with TLA+ integer `c - x` (Aristotle
    /// `bounded_sub_coincide`: `x ≤ c ⇒ (c:ℤ)-(x:ℤ) = ((c ∸ x : ℕ):ℤ) ≥ 0`).
    pub(crate) fn sub(a: Expr, b: Expr) -> Expr {
        ap(c(PEXP_SUB), [a, b])
    }
    pub(crate) fn and(a: Expr, b: Expr) -> Expr {
        ap(c(PPRED_AND), [a, b])
    }
    pub(crate) fn or(a: Expr, b: Expr) -> Expr {
        ap(c(PPRED_OR), [a, b])
    }
    pub(crate) fn not(a: Expr) -> Expr {
        ap(c(PPRED_NOT), [a])
    }
    pub(crate) fn implies(a: Expr, b: Expr) -> Expr {
        ap(c(PPRED_IMPLIES), [a, b])
    }
    pub(crate) fn equiv(a: Expr, b: Expr) -> Expr {
        ap(c(PPRED_EQUIV), [a, b])
    }
    pub(crate) fn eq(a: Expr, b: Expr) -> Expr {
        ap(c(PPRED_EQ), [a, b])
    }
    pub(crate) fn neq(a: Expr, b: Expr) -> Expr {
        ap(c(PPRED_NEQ), [a, b])
    }
    pub(crate) fn lt(a: Expr, b: Expr) -> Expr {
        ap(c(PPRED_LT), [a, b])
    }
    pub(crate) fn leq(a: Expr, b: Expr) -> Expr {
        ap(c(PPRED_LEQ), [a, b])
    }
    pub(crate) fn gt(a: Expr, b: Expr) -> Expr {
        ap(c(PPRED_GT), [a, b])
    }
    pub(crate) fn geq(a: Expr, b: Expr) -> Expr {
        ap(c(PPRED_GEQ), [a, b])
    }
    /// `inRange x lo hi` — `x ∈ lo..hi`.
    pub(crate) fn in_range(x: Expr, lo: Expr, hi: Expr) -> Expr {
        ap(c(PPRED_INRANGE), [x, lo, hi])
    }
    /// `eqIte l c t f` — `l = IF c THEN t ELSE f`.
    pub(crate) fn eq_ite(l: Expr, cond: Expr, t: Expr, f: Expr) -> Expr {
        ap(c(PPRED_EQITE), [l, cond, t, f])
    }
}

// ===========================================================================
// Quoted-term kernel evaluation for the AST-DIRECT lane: the same kernel legs
// as `reflect_corroborate` / `reflect_implies_mem`, but over a PRE-QUOTED deep
// `TyReflectPPred` term (produced by the AST quoter, not the IR quoter). The
// AST quoter constructs column indices from the spec's own variable list, so
// the IR-level `pred_cols_in_bounds` gate does not apply here; the CALLER
// guarantees every quoted `var i`/`prime i` has `i < |s|` (checked structurally
// at quote time against the state arity).
// ===========================================================================

/// Kernel-reduce `TyReflectEvalP quoted ⌜s⌝ ⌜sp⌝` to a `Bool` constant. `Some(b)` iff the
/// kernel accepts `Eq.refl` at that constant; `None` = env-build failure, arity mismatch, or
/// non-reduction (fail-closed at the caller — NOT a verdict).
pub(crate) fn kernel_eval_quoted_pred(quoted: &Expr, s: &[u64], sp: &[u64]) -> Option<bool> {
    let renv = cached_env()?;
    if s.len() != sp.len() {
        return None;
    }
    let tc = TypeChecker::new(&renv.env);
    kernel_bool_verdict(
        &tc,
        &ap(c(EVALP), [quoted.clone(), quote_state(s), quote_state(sp)]),
    )
}

/// Kernel-reduce the reflected implication `⟦quoted⟧(s,sp) ⇒ mem∈R` — i.e.
/// `Bool.or (Bool.not (TyReflectEvalP quoted ⌜s⌝ ⌜sp⌝)) (TyReflectMem ⌜mem⌝ ⌜R⌝)` — to a
/// `Bool` constant. `Some(false)` is the decisive completeness failure (`⟦quoted⟧` holds but
/// `mem ∉ R`); `None` fail-closes. The `⇒`/`∨`/`¬`/membership are ALL kernel work.
pub(crate) fn kernel_eval_quoted_implies_mem(
    quoted: &Expr,
    s: &[u64],
    sp: &[u64],
    mem: &[u64],
    r_set: &[Vec<u64>],
) -> Option<bool> {
    let renv = cached_env()?;
    if s.len() != sp.len() {
        return None;
    }
    let tc = TypeChecker::new(&renv.env);
    let eval = ap(c(EVALP), [quoted.clone(), quote_state(s), quote_state(sp)]);
    let memb = ap(c(MEM), [quote_state(mem), quote_state_set(r_set)]);
    kernel_bool_verdict(&tc, &ap(c("Bool.or"), [ap(c("Bool.not"), [eval]), memb]))
}

// ===========================================================================
// Column-bounds enforcement (see the module-docs boundary note): the deep `nth`
// TOTALIZES out-of-range indexing to 0, which does not mirror the shallow embedder,
// so the corroborator must DECLINE such IRs rather than evaluate them.
// ===========================================================================

fn val_cols_in_bounds(ir: &ValIR, len: usize) -> bool {
    match ir {
        ValIR::Lit(_) => true,
        ValIR::Var(i) | ValIR::Prime(i) => *i < len,
        ValIR::Add(a, b)
        | ValIR::Mul(a, b)
        | ValIR::Div(a, b)
        | ValIR::Mod(a, b)
        | ValIR::Sub(a, b) => val_cols_in_bounds(a, len) && val_cols_in_bounds(b, len),
        // Out of reflect scope regardless of columns (the quoter already declined).
        ValIR::SeqLen { .. } | ValIR::SeqTail { .. } | ValIR::SeqAppend { .. } => false,
        // SET counting nodes: the columns are those of the underlying set / per-element terms.
        ValIR::SetCard { set, .. } => set_cols_in_bounds(set, len),
        ValIR::CountFold { terms } => terms.iter().all(|t| pred_cols_in_bounds(t, len)),
    }
}

fn set_cols_in_bounds(ir: &SetIR, len: usize) -> bool {
    match ir {
        SetIR::Lit(_) => true,
        SetIR::Var(i) | SetIR::Prime(i) => *i < len,
        SetIR::Cup(a, b) | SetIR::Cap(a, b) => {
            set_cols_in_bounds(a, len) && set_cols_in_bounds(b, len)
        }
        SetIR::Digit { pack, .. } => val_cols_in_bounds(pack, len),
        // Out of reflect scope regardless of columns (the quoter already declined).
        SetIR::Filter { .. } => false,
    }
}

fn pred_cols_in_bounds(ir: &PredIR, len: usize) -> bool {
    match ir {
        PredIR::And(a, b) | PredIR::Or(a, b) | PredIR::Implies(a, b) | PredIR::Equiv(a, b) => {
            pred_cols_in_bounds(a, len) && pred_cols_in_bounds(b, len)
        }
        PredIR::Not(a) => pred_cols_in_bounds(a, len),
        PredIR::Eq(a, b)
        | PredIR::Neq(a, b)
        | PredIR::Lt(a, b)
        | PredIR::Leq(a, b)
        | PredIR::Gt(a, b)
        | PredIR::Geq(a, b) => val_cols_in_bounds(a, len) && val_cols_in_bounds(b, len),
        PredIR::BoolLit(_) => true,
        PredIR::Unchanged(i) | PredIR::SetUnchanged(i) => *i < len,
        // SET predicates: bounds of the underlying bitmask set(s).
        PredIR::SetEq(a, b) | PredIR::SetNeq(a, b) | PredIR::SetSubseteq(a, b) => {
            set_cols_in_bounds(a, len) && set_cols_in_bounds(b, len)
        }
        PredIR::SetMem(_, s) | PredIR::SetNotMem(_, s) => set_cols_in_bounds(s, len),
        // Out of reflect scope regardless of columns (the quoter already declined).
        PredIR::SetForall { .. }
        | PredIR::SetExists { .. }
        | PredIR::SubsetForall { .. }
        | PredIR::SubsetExists { .. } => false,
    }
}

// ===========================================================================
// The public corroboration API.
// ===========================================================================

/// Outcome of a reflected corroboration. Fail-closed semantics mirror `ck0_bridge`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReflectOutcome {
    /// The kernel reduced the reflected obligation to the EXPECTED `Bool` constant and
    /// accepted the `Eq.refl` proof.
    Corroborated,
    /// The reflect lane cannot decide this obligation (env build failure, out-of-scope IR,
    /// out-of-range column index, or a non-reducing term). NOT a verdict.
    Unavailable(String),
    /// DEFINITIVE disagreement: the reflected obligation reduced, but to the OTHER `Bool`
    /// constant. Evidence of a bug somewhere (embedder, quoter, or caller); fail closed.
    Disagree(String),
}

/// Kernel verdict on a closed `Bool` term: `Some(b)` iff the kernel accepts
/// `Eq.refl Bool Bool.b : Eq Bool lhs Bool.b` (i.e. `lhs` REDUCES to that constant —
/// checking BOTH constants is what makes a wrong reduction [`ReflectOutcome::Disagree`]
/// rather than silently `Unavailable`). `None` if neither constant checks.
fn kernel_bool_verdict(tc: &TypeChecker<'_>, lhs: &Expr) -> Option<bool> {
    if !crate::cleancic::kernel_term_within_resource_limits(lhs) {
        return None;
    }
    for expect in [true, false] {
        let konst = if expect { "Bool.true" } else { "Bool.false" };
        let ty = ap(c1("Eq"), [boolc(), lhs.clone(), c(konst)]);
        let pf = ap(c1("Eq.refl"), [boolc(), c(konst)]);
        if !crate::cleancic::kernel_term_within_resource_limits(&ty)
            || !crate::cleancic::kernel_term_within_resource_limits(&pf)
        {
            return None;
        }
        if tc.check_type(&pf, &ty).is_ok() {
            return Some(expect);
        }
    }
    None
}

/// Corroborate a concrete scalar obligation through the DEEP embedding: quote `(ir, s, sp)`,
/// build `Eq Bool (TyReflectEvalP ⌜ir⌝ ⌜s⌝ ⌜sp⌝) Bool.{true|false}` per `expect_true`, and let
/// the kernel decide it by `Eq.refl` (the kernel evaluates the deep evaluator by
/// ι/β/δ-reduction — no Rust computes the verdict).
pub fn reflect_corroborate(
    ir: &PredIR,
    s: &[u64],
    sp: &[u64],
    expect_true: bool,
) -> ReflectOutcome {
    let Some(renv) = cached_env() else {
        return ReflectOutcome::Unavailable("reflect env construction failed".into());
    };
    if s.len() != sp.len() {
        return ReflectOutcome::Unavailable(format!(
            "state arity mismatch: |s|={} but |sp|={}",
            s.len(),
            sp.len()
        ));
    }
    let Some(quoted) = quote_pred(ir) else {
        return ReflectOutcome::Unavailable(
            "IR outside the reflect v1 scalar fragment (Set/quantifier/Seq node)".into(),
        );
    };
    if !pred_cols_in_bounds(ir, s.len()) {
        return ReflectOutcome::Unavailable(format!(
            "column index out of range for state length {}: the deep `nth` would totalize it \
             to the default 0, which does not mirror the shallow embedder — declined",
            s.len()
        ));
    }
    let tc = TypeChecker::new(&renv.env);
    let lhs = ap(c(EVALP), [quoted, quote_state(s), quote_state(sp)]);
    match kernel_bool_verdict(&tc, &lhs) {
        None => ReflectOutcome::Unavailable(
            "reflected obligation did not reduce to a Bool constant".into(),
        ),
        Some(b) if b == expect_true => ReflectOutcome::Corroborated,
        Some(b) => ReflectOutcome::Disagree(format!(
            "kernel reduced the reflected obligation to Bool.{b}, caller expected Bool.{expect_true}"
        )),
    }
}

/// Cross-check the two embeddings on the SAME `(ir, s, sp)`: the kernel reduces both the
/// reflected term (`TyReflectEvalP ⌜ir⌝ ⌜s⌝ ⌜sp⌝`) and the shallow embedder's term
/// ([`crate::cleancic::embed_pred_ir`]) to a `Bool` constant, and the verdicts are compared.
/// `Some(true)` = the lanes agree, `Some(false)` = they DISAGREE (a bug somewhere),
/// `None` = out of scope / out of bounds / non-reduction (no comparison made).
pub fn reflect_agrees_with_shallow(ir: &PredIR, s: &[u64], sp: &[u64]) -> Option<bool> {
    let renv = cached_env()?;
    if s.len() != sp.len() {
        return None;
    }
    // Quote FIRST: it fail-closes the Set/quantifier/Seq arms, which keeps the shallow
    // embedder below on the scalar fragment (where the bounds check makes its `s[i]`
    // indexing safe).
    let quoted = quote_pred(ir)?;
    if !pred_cols_in_bounds(ir, s.len()) {
        return None;
    }
    let tc = TypeChecker::new(&renv.env);
    let deep = kernel_bool_verdict(
        &tc,
        &ap(c(EVALP), [quoted, quote_state(s), quote_state(sp)]),
    )?;
    let shallow = kernel_bool_verdict(&tc, &crate::cleancic::embed_pred_ir(ir, s, sp))?;
    Some(deep == shallow)
}

// ===========================================================================
// The reflected SAFETY-obligation leg: discharge `R ⊆ Safety` for the scalar
// fragment through the DEEP evaluator (NOT the shallow `embed_pred_ir`).
// ===========================================================================

/// Outcome of discharging `R ⊆ Safety` through the reflected (deep-embedding) evaluator.
///
/// SOUNDNESS-CRITICAL: [`Self::NotSafe`] is a DEFINITIVE kernel verdict that a reachable state
/// FALSIFIES the invariant (the kernel reduced its reflected obligation `TyReflectEvalP ⌜Safety⌝
/// ⌜s⌝ ⌜s⌝` to `Bool.false`); it must NEVER be collapsed to [`Self::Safe`]. [`Self::Inconclusive`]
/// is fail-closed (out-of-fragment IR, OOB column, non-reduction, env-build failure) — NOT a safety
/// verdict, and never a false accept.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReflectSafetyOutcome {
    /// The kernel reduced `TyReflectEvalP ⌜Safety⌝ ⌜s⌝ ⌜s⌝` to `Bool.true` for EVERY state `s ∈ R`
    /// (a vacuously-safe empty `R` also lands here).
    Safe,
    /// A reachable state definitively FALSIFIES the invariant under the deep evaluator (the kernel
    /// reduced its reflected obligation to `Bool.false`). Carries the offending state tuple.
    NotSafe {
        /// The reachable tuple whose reflected `⟦Safety⟧(state)` reduced to `Bool.false`.
        state: Vec<u64>,
        /// The corroborator's disagreement detail.
        detail: String,
    },
    /// The reflected lane cannot decide the obligation for ≥1 state (fail-closed): the `Safety` IR
    /// is outside the reflect scalar fragment (Set/quantifier/Seq node), an OOB column index, a
    /// non-reducing term, or the env failed to build. Carries the reason (+ which state, if one).
    Inconclusive(String),
}

/// Discharge `R ⊆ Safety` for a SCALAR STATE-predicate `Safety` (`safety_ir`) through the deep
/// evaluator: for every reachable tuple `s`, the kernel REDUCES `TyReflectEvalP ⌜safety_ir⌝ ⌜s⌝ ⌜s⌝`
/// (a state predicate ⇒ `sp = s`) and decides `Eq Bool`. This is the embedder-free companion to the
/// shallow `safety_general` leg: WHICH kernel op realizes each `Lt`/`Leq`/`And`/… is decided by the
/// kernel-checked [`EVALP`] DEFINITION (recursor minor premises), NOT by
/// [`crate::cleancic::embed_pred_ir`]'s op-by-op Rust construction.
///
/// Fail-closed on the FIRST undecidable state ([`ReflectSafetyOutcome::Inconclusive`]) and on the
/// FIRST state the kernel reduces to `Bool.false` ([`ReflectSafetyOutcome::NotSafe`]) — never a
/// silent skip, never a false `Safe`.
pub fn reflect_safety_over_reachable(
    safety_ir: &PredIR,
    reachable: &[Vec<u64>],
) -> ReflectSafetyOutcome {
    for s in reachable {
        // A safety invariant is a STATE predicate: sp = s (its primed columns, if any, are gated
        // out upstream). The kernel is the arbiter — no Rust re-computes the per-state verdict.
        match reflect_corroborate(safety_ir, s, s, true) {
            ReflectOutcome::Corroborated => {}
            ReflectOutcome::Disagree(msg) => {
                return ReflectSafetyOutcome::NotSafe {
                    state: s.clone(),
                    detail: msg,
                };
            }
            ReflectOutcome::Unavailable(reason) => {
                return ReflectSafetyOutcome::Inconclusive(format!("state {s:?}: {reason}"));
            }
        }
    }
    ReflectSafetyOutcome::Safe
}

// ===========================================================================
// The reflected COMPLETENESS legs (R2 milestone): discharge the two fixpoint
// completeness obligations — Init-completeness (`∀ s∈D: Init(s) ⇒ s∈R`) and
// Next-completeness / closure (`∀ s∈R, sp∈D: Next(s,sp) ⇒ sp∈R`) — through the
// DEEP evaluator (`TyReflectEvalP`) COMPOSED with the deep membership evaluator
// (`TyReflectMem`), so the shallow `embed_pred_ir` is OUT of both legs.
//
// The single per-obligation term is the reflected IMPLICATION
//   `Bool.or (Bool.not (TyReflectEvalP ⌜ir⌝ ⌜s⌝ ⌜sp⌝)) (TyReflectMem ⌜mem⌝ ⌜R⌝)`
// which the KERNEL reduces to `Bool.true` (the implication holds) or `Bool.false`
// (the antecedent holds but `mem ∉ R` — a MISSING init state / an ESCAPING
// successor, i.e. R is NOT closed). No Rust decides the `⇒`: the `Bool.or`/`Bool.not`
// combinators are kernel terms and `TyReflectMem`'s membership fold is a checked
// definition. This is the LOAD-BEARING closure guard: a non-closed R makes the
// Next leg reduce to `Bool.false`, never a false `closed`.
//
// SOUNDNESS of the DOMAIN `D`: the completeness legs prove closure/init-coverage
// RELATIVE to `D`. The residual `D ⊇ Succ(R)` / `D ⊇ {s:Init(s)}` is NOT decided
// here — the CALLER re-derives `D` from the IR by the SAME structural bound rules
// the cert uses (`crate::cleancic::{next,init}_domain_bounds_from_ir`, embedder-free)
// and classifies whether each axis is its column's full universe (coverage by
// construction) or a trusted-Rust bound. This module only reduces the relative
// obligation; it never claims `D` is complete.
// ===========================================================================

/// Reduce the reflected IMPLICATION `⟦ir⟧(s,sp) ⇒ mem∈R` through the kernel:
/// `Bool.or (Bool.not (TyReflectEvalP ⌜ir⌝ ⌜s⌝ ⌜sp⌝)) (TyReflectMem ⌜mem⌝ ⌜R⌝)` reduced to a
/// `Bool` constant. [`ReflectOutcome::Corroborated`] iff it reduces to `Bool.true`;
/// [`ReflectOutcome::Disagree`] iff it reduces to `Bool.false` (`⟦ir⟧` holds but `mem ∉ R` — the
/// decisive NON-CLOSED / missing-init signal); [`ReflectOutcome::Unavailable`] (fail-closed) on an
/// out-of-fragment IR, an OOB column, an arity mismatch, env-build failure, or a non-reduction.
/// The `⇒`, `∨`, `¬`, and membership are ALL kernel work — no Rust computes the verdict.
pub fn reflect_implies_mem(
    ir: &PredIR,
    s: &[u64],
    sp: &[u64],
    mem: &[u64],
    r_set: &[Vec<u64>],
) -> ReflectOutcome {
    let Some(renv) = cached_env() else {
        return ReflectOutcome::Unavailable("reflect env construction failed".into());
    };
    if s.len() != sp.len() {
        return ReflectOutcome::Unavailable(format!(
            "state arity mismatch: |s|={} but |sp|={}",
            s.len(),
            sp.len()
        ));
    }
    let Some(quoted) = quote_pred(ir) else {
        return ReflectOutcome::Unavailable(
            "IR outside the reflect v1 scalar fragment (Set/quantifier/Seq node)".into(),
        );
    };
    if !pred_cols_in_bounds(ir, s.len()) {
        return ReflectOutcome::Unavailable(format!(
            "column index out of range for state length {}: the deep `nth` would totalize it to \
             the default 0, which does not mirror the shallow embedder — declined",
            s.len()
        ));
    }
    let tc = TypeChecker::new(&renv.env);
    let eval = ap(c(EVALP), [quoted, quote_state(s), quote_state(sp)]);
    let memb = ap(c(MEM), [quote_state(mem), quote_state_set(r_set)]);
    // ⟦ir⟧(s,sp) ⇒ mem∈R   ≡   (¬⟦ir⟧(s,sp)) ∨ (mem∈R).
    let lhs = ap(c("Bool.or"), [ap(c("Bool.not"), [eval]), memb]);
    match kernel_bool_verdict(&tc, &lhs) {
        None => ReflectOutcome::Unavailable(
            "reflected implication did not reduce to a Bool constant".into(),
        ),
        Some(true) => ReflectOutcome::Corroborated,
        Some(false) => ReflectOutcome::Disagree(format!(
            "kernel reduced ⟦ir⟧(s,sp) ⇒ mem∈R to Bool.false: the antecedent holds but {mem:?} ∉ R"
        )),
    }
}

/// Outcome of the reflected NEXT-completeness (closure) discharge over `R × D`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReflectClosureOutcome {
    /// Every `(s∈R, sp∈D)` reduced `Next(s,sp) ⇒ sp∈R` to `Bool.true` — `R` is closed under `Next`
    /// RELATIVE to `D` (the caller owns the `D ⊇ Succ(R)` coverage argument). Carries the pair count.
    Closed {
        /// Number of `(s, sp)` pairs the kernel discharged.
        pairs: usize,
    },
    /// A successor ESCAPES `R`: the kernel reduced `Next(s,sp) ⇒ sp∈R` to `Bool.false` — `Next(s,sp)`
    /// holds but `sp ∉ R`. `R` is NOT closed (a genuine non-closed / tampered reachable set).
    NotClosed {
        /// The source state whose successor escapes `R`.
        s: Vec<u64>,
        /// The escaping successor tuple (`sp ∈ D`, `Next(s,sp)`, `sp ∉ R`).
        sp: Vec<u64>,
        /// The kernel disagreement detail.
        detail: String,
    },
    /// Fail-closed: an out-of-fragment `Next_ir`, an OOB column, a non-reduction, or env-build
    /// failure for ≥1 pair. NOT a closure verdict.
    Inconclusive(String),
}

/// Discharge NEXT-completeness (closure) `∀ s∈R, ∀ sp∈D: Next(s,sp) ⇒ sp∈R` through the deep
/// evaluator: for every source `s∈R` and every domain successor `sp∈D`, the kernel reduces the
/// reflected implication [`reflect_implies_mem`] (`mem = sp`). Fail-closed on the FIRST
/// non-reducing/out-of-fragment pair and on the FIRST escaping successor — never a false `Closed`.
///
/// SOUNDNESS: closure is proved RELATIVE to `D`; the obligation is probative for TRUE closure only
/// when `D ⊇ Succ(R)`, which the caller establishes by re-deriving `D` from the IR (never trusting
/// a stored bound). `embed_pred_ir` is NOT called.
pub fn reflect_next_completeness_over_domain(
    next_ir: &PredIR,
    reachable: &[Vec<u64>],
    domain: &[Vec<u64>],
) -> ReflectClosureOutcome {
    let mut pairs = 0usize;
    for s in reachable {
        for sp in domain {
            // `mem = sp`: the successor whose membership closure requires.
            match reflect_implies_mem(next_ir, s, sp, sp, reachable) {
                ReflectOutcome::Corroborated => pairs += 1,
                ReflectOutcome::Disagree(detail) => {
                    return ReflectClosureOutcome::NotClosed {
                        s: s.clone(),
                        sp: sp.clone(),
                        detail,
                    };
                }
                ReflectOutcome::Unavailable(reason) => {
                    return ReflectClosureOutcome::Inconclusive(format!(
                        "s={s:?} sp={sp:?}: {reason}"
                    ));
                }
            }
        }
    }
    ReflectClosureOutcome::Closed { pairs }
}

/// Outcome of the reflected INIT-completeness discharge over `D`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReflectInitOutcome {
    /// Every `s∈D` reduced `Init(s) ⇒ s∈R` to `Bool.true` — every `Init`-satisfying domain state is
    /// in `R` (RELATIVE to `D`; the caller owns `D ⊇ {s:Init(s)}`). Carries the domain size.
    Complete {
        /// Number of domain states the kernel discharged.
        states: usize,
    },
    /// An `Init` state is MISSING from `R`: the kernel reduced `Init(s) ⇒ s∈R` to `Bool.false`
    /// (`Init(s)` holds but `s ∉ R`).
    NotComplete {
        /// The `Init`-satisfying state absent from `R`.
        s: Vec<u64>,
        /// The kernel disagreement detail.
        detail: String,
    },
    /// Fail-closed: out-of-fragment `Init_ir`, OOB column, non-reduction, or env-build failure.
    Inconclusive(String),
}

/// Discharge INIT-completeness `∀ s∈D: Init(s) ⇒ s∈R` through the deep evaluator: for every domain
/// state `s∈D`, the kernel reduces the reflected implication (`Init` is a state predicate ⇒ `sp = s`,
/// `mem = s`). Fail-closed on the first non-reducing pair and on the first missing `Init` state.
/// `embed_pred_ir` is NOT called.
pub fn reflect_init_completeness_over_domain(
    init_ir: &PredIR,
    domain: &[Vec<u64>],
    reachable: &[Vec<u64>],
) -> ReflectInitOutcome {
    for s in domain {
        match reflect_implies_mem(init_ir, s, s, s, reachable) {
            ReflectOutcome::Corroborated => {}
            ReflectOutcome::Disagree(detail) => {
                return ReflectInitOutcome::NotComplete {
                    s: s.clone(),
                    detail,
                };
            }
            ReflectOutcome::Unavailable(reason) => {
                return ReflectInitOutcome::Inconclusive(format!("s={s:?}: {reason}"));
            }
        }
    }
    ReflectInitOutcome::Complete {
        states: domain.len(),
    }
}

// ===========================================================================
// The reflected SUBSET obligation (R2 applied to the fixpoint membership legs).
// ===========================================================================

/// Heartbeat budget for a reflected-subset kernel check, scaled to the obligation: the merge
/// evaluation touches each element of both lists a bounded number of times, so a generous
/// per-element allowance keeps the budget FINITE (a malicious cert cannot spin the checker
/// unboundedly — the obligation itself is rebuilt from re-derived data, never from the cert)
/// while never starving a genuine obligation. Floor at the kernel default.
fn reflect_subset_heartbeats(n_elems: usize) -> u32 {
    let scaled = (n_elems as u64)
        .saturating_mul(4096)
        .saturating_add(1 << 21);
    u32::try_from(scaled).unwrap_or(u32::MAX)
}

/// The reflected `xs ⊆ ys` obligation `(type, proof)`: type
/// `Eq Bool (TyReflectSubseq ⌜xs⌝ ⌜ys⌝) Bool.true`, proof `Eq.refl Bool Bool.true` — the kernel
/// accepts iff its OWN ι/β/δ-reduction of the merge fold over the two quoted lists yields
/// `Bool.true`. Callers pass canonicalized (sorted, deduplicated) lists.
fn reflect_subset_bool_eq(xs: &[Vec<u64>], ys: &[Vec<u64>]) -> (Expr, Expr) {
    let lhs = ap(c(SUBSEQ), [quote_state_set(xs), quote_state_set(ys)]);
    let ty = ap(c1("Eq"), [boolc(), lhs, c("Bool.true")]);
    let pf = ap(c1("Eq.refl"), [boolc(), c("Bool.true")]);
    (ty, pf)
}

/// The acceptance gate for a reflected obligation — the reflect-env mirror of
/// `cleancic::kernel_accepts`: (1) the clean kernel type-checks `term : expected` under the
/// reflect env (prelude + the CHECKED `TyReflect*` admissions); (2) the term's transitive const
/// closure reaches NO trust marker (same Phase-0 gate, walked over a private clone of the
/// reflect env); (3) clean-ck0 is CONSULTED — `List`/`List.rec` are outside its ingest fragment
/// so the expected outcome is `Unavailable` (clean-kernel tier, tallied honestly), but an actual
/// `Rejected` (checker disagreement) fails CLOSED exactly as in the shallow lane.
fn reflect_accepts(renv: &ReflectEnv, term: &Expr, expected: &Expr, heartbeats: u32) -> bool {
    if !crate::cleancic::kernel_term_within_resource_limits(term)
        || !crate::cleancic::kernel_term_within_resource_limits(expected)
    {
        return false;
    }
    let mut tc = TypeChecker::new(&renv.env);
    tc.set_heartbeat_limit(heartbeats);
    if tc.check_type(term, expected).is_err() {
        return false;
    }
    if !crate::cleancic::term_reaches_no_trust_marker(renv.env.clone(), expected, term) {
        return false;
    }
    !matches!(
        crate::ck0_bridge::corroborate(term, expected),
        crate::ck0_bridge::Ck0Corroboration::Rejected(_)
    )
}

/// CERTIFY the reflected subset claim `xs ⊆ ys` (both canonicalized): kernel-check the
/// `Eq.refl` proof at the reflected obligation type and serialize the (tiny, constant-size)
/// proof term on acceptance. `None` (fail-closed) on env-build failure, a kernel rejection —
/// including `Subseq` reducing to `Bool.false` (a genuine non-subset) or heartbeat exhaustion —
/// or a ck0 disagreement. The obligation TYPE is never serialized: verifiers rebuild it from
/// re-derived data ([`reflect_verify_subset`]).
pub fn reflect_certify_subset(xs: &[Vec<u64>], ys: &[Vec<u64>]) -> Option<Vec<u8>> {
    let renv = cached_env()?;
    let (ty, pf) = reflect_subset_bool_eq(xs, ys);
    if reflect_accepts(
        renv,
        &pf,
        &ty,
        reflect_subset_heartbeats(xs.len() + ys.len()),
    ) {
        return crate::cleancic::expr_to_bytes(&pf);
    }
    reflect_check_subset_partitioned(renv, xs, ys, &pf)?;
    crate::cleancic::expr_to_bytes(&pf)
}

/// LHS elements per fallback reflected-subset obligation. Each canonical LHS chunk is checked against
/// the canonical RHS window spanning the same ordered values; the windows are literal slices of `ys`,
/// so proving every `chunk ⊆ window` proves their union `xs ⊆ ys`. This keeps TokenRing's 46,656
/// six-wide tuples below the per-term 2M-node boundary without trusting a membership verdict from Rust.
const REFLECT_SUBSET_CHUNK_ELEMS: usize = 4_096;

fn reflect_check_subset_partitioned(
    renv: &ReflectEnv,
    xs: &[Vec<u64>],
    ys: &[Vec<u64>],
    proof: &Expr,
) -> Option<()> {
    if xs.is_empty() {
        let (ty, _) = reflect_subset_bool_eq(&[], &[]);
        return reflect_accepts(renv, proof, &ty, reflect_subset_heartbeats(0)).then_some(());
    }
    for chunk in xs.chunks(REFLECT_SUBSET_CHUNK_ELEMS) {
        let first = chunk.first()?;
        let last = chunk.last()?;
        let lo = ys.partition_point(|y| y < first);
        let hi = ys.partition_point(|y| y <= last);
        let window = &ys[lo..hi];
        let (ty, _) = reflect_subset_bool_eq(chunk, window);
        if !reflect_accepts(
            renv,
            proof,
            &ty,
            reflect_subset_heartbeats(chunk.len() + window.len()),
        ) {
            return None;
        }
    }
    Some(())
}

/// RE-CHECK a stored reflected-subset proof against the obligation REBUILT from the re-derived
/// `(xs, ys)` — the stored bytes' claimed type is never trusted. Fail-closed on bad bytes, a
/// kernel rejection, or ck0 disagreement.
pub fn reflect_verify_subset(xs: &[Vec<u64>], ys: &[Vec<u64>], bytes: &[u8]) -> bool {
    let Some(renv) = cached_env() else {
        return false;
    };
    let Some(term) = crate::cleancic::expr_from_bytes(bytes) else {
        return false;
    };
    let (ty, _) = reflect_subset_bool_eq(xs, ys);
    if reflect_accepts(
        renv,
        &term,
        &ty,
        reflect_subset_heartbeats(xs.len() + ys.len()),
    ) {
        return true;
    }
    reflect_check_subset_partitioned(renv, xs, ys, &term).is_some()
}

// ===========================================================================
// Tests. The kernel is the arbiter throughout: every assertion is a check_type
// (or an outcome derived from one), never a Rust re-computation of the verdict.
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::explicit_fixpoint_cert::SetIR;

    // IR shorthands (tests only).
    fn band(a: PredIR, b: PredIR) -> PredIR {
        PredIR::And(Box::new(a), Box::new(b))
    }
    fn imp(a: PredIR, b: PredIR) -> PredIR {
        PredIR::Implies(Box::new(a), Box::new(b))
    }
    fn lt(a: ValIR, b: ValIR) -> PredIR {
        PredIR::Lt(a, b)
    }
    fn leq(a: ValIR, b: ValIR) -> PredIR {
        PredIR::Leq(a, b)
    }
    fn eqp(a: ValIR, b: ValIR) -> PredIR {
        PredIR::Eq(a, b)
    }
    fn v(i: usize) -> ValIR {
        ValIR::Var(i)
    }
    fn p(i: usize) -> ValIR {
        ValIR::Prime(i)
    }
    fn l(x: u64) -> ValIR {
        ValIR::Lit(x)
    }
    /// A genuinely OUT-OF-FRAGMENT predicate the quoter declines (`None`) and `pred_exact` rejects: a
    /// bounded set quantifier. (The bitmask `Set*` predicates are now COVERED, so a bare `SetEq` is no
    /// longer a valid out-of-fragment probe — the quantifier folds remain the honest residual.)
    fn out_of_frag() -> PredIR {
        PredIR::SetForall {
            source: SetIR::Lit(1),
            universe: 1,
            bound_col: 1,
            body: Box::new(PredIR::BoolLit(true)),
        }
    }

    /// A 2-column obligation exercising the Lt / Implies / Unchanged arms:
    /// `x < 5 ∧ ((3 ≤ y) ⇒ y < y') ∧ UNCHANGED x`.
    fn two_col_ir() -> PredIR {
        band(
            lt(v(0), l(5)),
            band(imp(leq(l(3), v(1)), lt(v(1), p(1))), PredIR::Unchanged(0)),
        )
    }

    /// (a) The env admits: every ledger entry went through a CHECKED path — the ledger is
    /// EXACTLY the two inductives (with all constructors) plus the three checked defs, and
    /// the `ReflectLedgerEntry` enum has no structural/axiom variant to hide behind.
    #[test]
    fn reflect_env_builds_with_fully_checked_ledger() {
        let ledger = env_ledger().expect("reflect env must build");
        let entry = |n: &str| {
            ledger
                .iter()
                .find(|(name, _)| name == n)
                .map(|(_, e)| e.clone())
        };
        assert_eq!(entry(PEXP), Some(ReflectLedgerEntry::Inductive));
        assert_eq!(entry(PPRED), Some(ReflectLedgerEntry::Inductive));
        for ctor in PEXP_CTORS.iter().chain(PPRED_CTORS.iter()) {
            assert_eq!(entry(ctor), Some(ReflectLedgerEntry::Inductive), "{ctor}");
        }
        for def in [NTH, EVALV, EVALP, TUPEQ, MEM, ALLMEM, SUBSEQ] {
            assert_eq!(entry(def), Some(ReflectLedgerEntry::CheckedDef), "{def}");
        }
        // Nothing else entered: 2 type formers + 8 + 13 constructors + 7 checked defs.
        assert_eq!(ledger.len(), 2 + PEXP_CTORS.len() + PPRED_CTORS.len() + 7);
        // Zero structural admissions, by type: every entry is Inductive-or-CheckedDef.
        assert!(ledger.iter().all(|(_, e)| matches!(
            e,
            ReflectLedgerEntry::Inductive | ReflectLedgerEntry::CheckedDef
        )));
    }

    /// (b) The kernel corroborates a TRUE 2-column obligation through the deep evaluator
    /// (Lt/Implies/Unchanged arms live), and DISAGREES on the negative control — the same
    /// IR made false — rather than rubber-stamping the caller's expectation.
    #[test]
    fn reflect_corroborates_true_and_disagrees_false_two_column() {
        let ir = two_col_ir();
        // s = (x=2, y=3), sp = (x=2, y=4): 2<5 ∧ (3≤3 ⇒ 3<4) ∧ x unchanged ⇒ TRUE.
        assert_eq!(
            reflect_corroborate(&ir, &[2, 3], &[2, 4], true),
            ReflectOutcome::Corroborated
        );
        // Expecting FALSE of a true obligation is a definitive disagreement, not Unavailable.
        assert!(matches!(
            reflect_corroborate(&ir, &[2, 3], &[2, 4], false),
            ReflectOutcome::Disagree(_)
        ));
        // Negative control: sp = (x=9, y=4) violates UNCHANGED x ⇒ the obligation is FALSE.
        assert!(matches!(
            reflect_corroborate(&ir, &[2, 3], &[9, 4], true),
            ReflectOutcome::Disagree(_)
        ));
        assert_eq!(
            reflect_corroborate(&ir, &[2, 3], &[9, 4], false),
            ReflectOutcome::Corroborated
        );
    }

    /// (c) Out-of-scope IR (a Set node) is `Unavailable` — declined by the quoter,
    /// never mis-evaluated.
    #[test]
    fn reflect_unavailable_on_out_of_scope_set_node() {
        let ir = out_of_frag();
        assert!(matches!(
            reflect_corroborate(&ir, &[0], &[0], true),
            ReflectOutcome::Unavailable(_)
        ));
    }

    /// (d) Per-constructor DEFINITIONAL agreement: the evaluator's recursor equations hold by
    /// bare `Eq.refl` UNDER BINDERS (symbolic subterms and states), pinning the deep op
    /// choices to the shallow embedder's (`Nat.add`, `Nat.ble`, the `a+1 ≤ b` Lt shape,
    /// `Bool.and`) as kernel-checked facts, not Rust conventions.
    #[test]
    fn reflect_recursor_equations_definitional() {
        let renv = cached_env().expect("reflect env must build");
        let tc = TypeChecker::new(&renv.env);
        // Π-telescope (a b : PExp) (s sp : List Nat) — a=3 b=2 s=1 sp=0 in the body.
        let ea = || ap(c(EVALV), [bv(3), bv(1), bv(0)]);
        let eb = || ap(c(EVALV), [bv(2), bv(1), bv(0)]);
        let under_pexp = |lhs: Expr, rhs: Expr, ty: Expr| {
            let goal = pi(
                c(PEXP),
                pi(
                    c(PEXP),
                    pi(
                        list_nat(),
                        pi(list_nat(), ap(c1("Eq"), [ty.clone(), lhs, rhs.clone()])),
                    ),
                ),
            );
            let proof = lam(
                c(PEXP),
                lam(
                    c(PEXP),
                    lam(list_nat(), lam(list_nat(), ap(c1("Eq.refl"), [ty, rhs]))),
                ),
            );
            tc.check_type(&proof, &goal)
        };
        // (i) EvalV (add a b) s sp ≡ Nat.add (EvalV a s sp) (EvalV b s sp)
        under_pexp(
            ap(c(EVALV), [ap(c(PEXP_ADD), [bv(3), bv(2)]), bv(1), bv(0)]),
            ap(c("Nat.add"), [ea(), eb()]),
            nat(),
        )
        .expect("add equation must be definitional");
        // (ii) EvalP (leq a b) s sp ≡ Nat.ble (EvalV a s sp) (EvalV b s sp)
        under_pexp(
            ap(c(EVALP), [ap(c(PPRED_LEQ), [bv(3), bv(2)]), bv(1), bv(0)]),
            ap(c("Nat.ble"), [ea(), eb()]),
            boolc(),
        )
        .expect("leq equation must be definitional");
        // (iii) EvalP (lt a b) s sp ≡ Nat.ble (Nat.add (EvalV a s sp) 1) (EvalV b s sp)
        under_pexp(
            ap(c(EVALP), [ap(c(PPRED_LT), [bv(3), bv(2)]), bv(1), bv(0)]),
            ap(c("Nat.ble"), [ap(c("Nat.add"), [ea(), nl(1)]), eb()]),
            boolc(),
        )
        .expect("lt equation must be definitional (the a+1 ≤ b shape)");
        // (iv) EvalP (and p q) s sp ≡ Bool.and (EvalP p s sp) (EvalP q s sp), (p q : PPred).
        let pa = || ap(c(EVALP), [bv(3), bv(1), bv(0)]);
        let pb = || ap(c(EVALP), [bv(2), bv(1), bv(0)]);
        let lhs = ap(c(EVALP), [ap(c(PPRED_AND), [bv(3), bv(2)]), bv(1), bv(0)]);
        let rhs = ap(c("Bool.and"), [pa(), pb()]);
        let goal = pi(
            c(PPRED),
            pi(
                c(PPRED),
                pi(
                    list_nat(),
                    pi(list_nat(), ap(c1("Eq"), [boolc(), lhs, rhs.clone()])),
                ),
            ),
        );
        let proof = lam(
            c(PPRED),
            lam(
                c(PPRED),
                lam(
                    list_nat(),
                    lam(list_nat(), ap(c1("Eq.refl"), [boolc(), rhs])),
                ),
            ),
        );
        tc.check_type(&proof, &goal)
            .expect("and equation must be definitional");
    }

    /// (d′) AST-direct ctor DEFINITIONAL equations (design pivot increment 1): `inRange` and
    /// `eqIte` hold by bare `Eq.refl` UNDER BINDERS — their op realizations are kernel-checked
    /// definition DATA, pinned here per-constructor exactly like the (d) equations. Also pins
    /// closed-instance reductions in BOTH truth directions (the kernel can say no).
    #[test]
    fn reflect_ast_direct_ctor_equations_definitional() {
        let renv = cached_env().expect("reflect env must build");
        let tc = TypeChecker::new(&renv.env);
        // (i) inRange, under (x lo hi : PExp) (s sp : List Nat) — x=4 lo=3 hi=2 s=1 sp=0:
        //     EvalP (inRange x lo hi) s sp ≡ Bool.and (Nat.ble ⟦lo⟧ ⟦x⟧) (Nat.ble ⟦x⟧ ⟦hi⟧).
        {
            let evx = || ap(c(EVALV), [bv(4), bv(1), bv(0)]);
            let evlo = ap(c(EVALV), [bv(3), bv(1), bv(0)]);
            let evhi = ap(c(EVALV), [bv(2), bv(1), bv(0)]);
            let lhs = ap(
                c(EVALP),
                [ap(c(PPRED_INRANGE), [bv(4), bv(3), bv(2)]), bv(1), bv(0)],
            );
            let rhs = ap(
                c("Bool.and"),
                [
                    ap(c("Nat.ble"), [evlo, evx()]),
                    ap(c("Nat.ble"), [evx(), evhi]),
                ],
            );
            let goal = pi(
                c(PEXP),
                pi(
                    c(PEXP),
                    pi(
                        c(PEXP),
                        pi(
                            list_nat(),
                            pi(list_nat(), ap(c1("Eq"), [boolc(), lhs, rhs.clone()])),
                        ),
                    ),
                ),
            );
            let proof = lam(
                c(PEXP),
                lam(
                    c(PEXP),
                    lam(
                        c(PEXP),
                        lam(
                            list_nat(),
                            lam(list_nat(), ap(c1("Eq.refl"), [boolc(), rhs])),
                        ),
                    ),
                ),
            );
            tc.check_type(&proof, &goal)
                .expect("inRange equation must be definitional (lo ≤ x ∧ x ≤ hi)");
        }
        // (ii) eqIte, under (l : PExp) (cnd : PPred) (t f : PExp) (s sp) — l=5 cnd=4 t=3 f=2 s=1 sp=0:
        //      EvalP (eqIte l cnd t f) s sp
        //        ≡ Nat.beq ⟦l⟧ (Bool.rec (fun _ => Nat) ⟦f⟧ ⟦t⟧ (EvalP cnd s sp)).
        {
            let evl = ap(c(EVALV), [bv(5), bv(1), bv(0)]);
            let evt = ap(c(EVALV), [bv(3), bv(1), bv(0)]);
            let evf = ap(c(EVALV), [bv(2), bv(1), bv(0)]);
            let evc = ap(c(EVALP), [bv(4), bv(1), bv(0)]);
            let lhs = ap(
                c(EVALP),
                [
                    ap(c(PPRED_EQITE), [bv(5), bv(4), bv(3), bv(2)]),
                    bv(1),
                    bv(0),
                ],
            );
            let rhs = ap(
                c("Nat.beq"),
                [
                    evl,
                    ap(
                        Expr::const_str_levels("Bool.rec", vec![Level::succ(Level::zero())]),
                        [lam(boolc(), nat()), evf, evt, evc],
                    ),
                ],
            );
            let goal = pi(
                c(PEXP),
                pi(
                    c(PPRED),
                    pi(
                        c(PEXP),
                        pi(
                            c(PEXP),
                            pi(
                                list_nat(),
                                pi(list_nat(), ap(c1("Eq"), [boolc(), lhs, rhs.clone()])),
                            ),
                        ),
                    ),
                ),
            );
            let proof = lam(
                c(PEXP),
                lam(
                    c(PPRED),
                    lam(
                        c(PEXP),
                        lam(
                            c(PEXP),
                            lam(
                                list_nat(),
                                lam(list_nat(), ap(c1("Eq.refl"), [boolc(), rhs])),
                            ),
                        ),
                    ),
                ),
            );
            tc.check_type(&proof, &goal).expect(
                "eqIte equation must be definitional (Nat.beq ⟦l⟧ (if ⟦c⟧ then ⟦t⟧ else ⟦f⟧))",
            );
        }
        // (iii) Closed instances through the quoted-eval helper, BOTH truth directions.
        // inRange: 1 ≤ 5 ≤ 12 true; 13 ∉ 1..12; empty interval 5..3 is unsatisfiable.
        let ir = |x: u64| deep::in_range(deep::lit(x), deep::lit(1), deep::lit(12));
        assert_eq!(kernel_eval_quoted_pred(&ir(5), &[], &[]), Some(true));
        assert_eq!(kernel_eval_quoted_pred(&ir(13), &[], &[]), Some(false));
        assert_eq!(kernel_eval_quoted_pred(&ir(0), &[], &[]), Some(false));
        let empty = deep::in_range(deep::lit(4), deep::lit(5), deep::lit(3));
        assert_eq!(kernel_eval_quoted_pred(&empty, &[], &[]), Some(false));
        // eqIte (HourClock's Next shape): hr' = IF hr ≠ 12 THEN hr+1 ELSE 1.
        let next = deep::eq_ite(
            deep::prime(0),
            deep::neq(deep::var(0), deep::lit(12)),
            deep::add(deep::var(0), deep::lit(1)),
            deep::lit(1),
        );
        assert_eq!(kernel_eval_quoted_pred(&next, &[5], &[6]), Some(true));
        assert_eq!(kernel_eval_quoted_pred(&next, &[5], &[7]), Some(false));
        assert_eq!(kernel_eval_quoted_pred(&next, &[12], &[1]), Some(true));
        assert_eq!(kernel_eval_quoted_pred(&next, &[12], &[13]), Some(false));
    }

    /// (e) Multi-column Var/Prime indexing works through the deep `nth`; an OUT-OF-RANGE
    /// column is DECLINED (`Unavailable`) — and the test also kernel-verifies WHY that
    /// enforcement exists: `TyReflectNth [2,7] 5` genuinely reduces to the default `0`,
    /// which would spuriously satisfy `Var(5) < 1` if the quoter relied on it.
    #[test]
    fn reflect_multi_column_indexing_and_oob_columns_decline() {
        // 3 columns: s[2]=7 and 7 < sp[2]=9.
        let ir = band(eqp(v(2), l(7)), lt(v(2), p(2)));
        assert_eq!(
            reflect_corroborate(&ir, &[1, 4, 7], &[1, 4, 9], true),
            ReflectOutcome::Corroborated
        );
        // Column 5 over 2-column states: declined, NOT defaulted.
        let oob = lt(v(5), l(1));
        assert!(matches!(
            reflect_corroborate(&oob, &[2, 3], &[2, 3], true),
            ReflectOutcome::Unavailable(_)
        ));
        let renv = cached_env().expect("reflect env must build");
        let tc = TypeChecker::new(&renv.env);
        // Quoted states really are `List.{0} Nat` values.
        tc.check_type(&quote_state(&[2, 7]), &list_nat())
            .expect("quoted state must check as List Nat");
        tc.check_type(&quote_state(&[]), &list_nat())
            .expect("quoted empty state must check as List Nat");
        // The boundary, honestly: nth beyond the length reduces to the default 0.
        let lhs = ap(c(NTH), [quote_state(&[2, 7]), nl(5)]);
        let ty = ap(c1("Eq"), [nat(), lhs, nl(0)]);
        let pf = ap(c1("Eq.refl"), [nat(), nl(0)]);
        tc.check_type(&pf, &ty)
            .expect("out-of-range nth must reduce to the default 0 (why we decline OOB IRs)");
    }

    /// PROBE (R2 fixpoint membership, step 1 of the brief): the quoted-list membership defs
    /// ι-reduce correctly on 3-state sets, INCLUDING negative controls — a non-member reduces
    /// to `Bool.false`, and `Eq.refl` at `Bool.true` for it is REJECTED by the kernel (the
    /// fail-closed direction is real, not assumed). Every verdict below is the KERNEL's
    /// (`kernel_bool_verdict` = a `check_type` against both constants), never a Rust re-fold.
    #[test]
    fn reflect_membership_defs_probe_three_state_sets() {
        let renv = cached_env().expect("reflect env must build");
        let tc = TypeChecker::new(&renv.env);
        let verdict = |lhs: Expr| kernel_bool_verdict(&tc, &lhs);
        let tupeq = |a: &[u64], b: &[u64]| verdict(ap(c(TUPEQ), [quote_state(a), quote_state(b)]));
        // TupEq: componentwise Nat.beq, LENGTH-SENSITIVE.
        assert_eq!(tupeq(&[1, 2], &[1, 2]), Some(true));
        assert_eq!(tupeq(&[1, 2], &[1, 3]), Some(false));
        assert_eq!(tupeq(&[1], &[1, 2]), Some(false), "prefix ≠ longer tuple");
        assert_eq!(tupeq(&[1, 2], &[1]), Some(false), "longer tuple ≠ prefix");
        assert_eq!(tupeq(&[], &[]), Some(true));

        // A 3-state quoted set (canonical sorted order, 2 columns).
        let s: Vec<Vec<u64>> = vec![vec![0, 5], vec![1, 4], vec![2, 3]];
        let mem = |x: &[u64]| verdict(ap(c(MEM), [quote_state(x), quote_state_set(&s)]));
        assert_eq!(mem(&[1, 4]), Some(true), "member");
        assert_eq!(mem(&[1, 5]), Some(false), "non-member reduces to false");
        // NEGATIVE CONTROL: Eq.refl at TRUE for the non-member must be REJECTED.
        let bad_ty = ap(
            c1("Eq"),
            [
                boolc(),
                ap(c(MEM), [quote_state(&[1, 5]), quote_state_set(&s)]),
                c("Bool.true"),
            ],
        );
        let refl_true = ap(c1("Eq.refl"), [boolc(), c("Bool.true")]);
        assert!(
            tc.check_type(&refl_true, &bad_ty).is_err(),
            "the kernel must reject Eq.refl-at-true for a non-member"
        );

        // AllMem (the semantic reference) and Subseq (the live merge form) agree on the probe
        // sets — subset, non-subset, and empty.
        let sub: Vec<Vec<u64>> = vec![vec![0, 5], vec![2, 3]];
        let non: Vec<Vec<u64>> = vec![vec![0, 5], vec![9, 9]];
        let empty: Vec<Vec<u64>> = vec![];
        for (xs, expect) in [(&sub, true), (&non, false), (&empty, true), (&s, true)] {
            let am = verdict(ap(c(ALLMEM), [quote_state_set(xs), quote_state_set(&s)]));
            let sq = verdict(ap(c(SUBSEQ), [quote_state_set(xs), quote_state_set(&s)]));
            assert_eq!(am, Some(expect), "AllMem {xs:?}");
            assert_eq!(sq, Some(expect), "Subseq {xs:?}");
        }
        // Supersets are NOT subsets: both defs refute ys ⊆ xs for a strict subset xs.
        assert_eq!(
            verdict(ap(c(ALLMEM), [quote_state_set(&s), quote_state_set(&sub)])),
            Some(false)
        );
        assert_eq!(
            verdict(ap(c(SUBSEQ), [quote_state_set(&s), quote_state_set(&sub)])),
            Some(false)
        );
        // The pinned Subseq incompleteness (why callers canonicalize): an OUT-OF-ORDER xs is
        // declined (false) even though it is a set-subset — fail-closed, never unsound.
        let out_of_order: Vec<Vec<u64>> = vec![vec![2, 3], vec![0, 5]];
        assert_eq!(
            verdict(ap(
                c(SUBSEQ),
                [quote_state_set(&out_of_order), quote_state_set(&s)]
            )),
            Some(false),
            "out-of-order xs must fail closed (sortedness is the completeness precondition)"
        );
        assert_eq!(
            verdict(ap(c(ALLMEM), [quote_state_set(&out_of_order), quote_state_set(&s)])),
            Some(true),
            "AllMem is order-insensitive — the semantic reference the merge form under-approximates"
        );
    }

    /// The reflected-subset certify/verify API round-trips, and every tamper direction from the
    /// brief REJECTS: mutated proof bytes, the R-subset/superset swap, and a non-subset claim.
    #[test]
    fn reflect_subset_certify_verify_and_tampers() {
        let r: Vec<Vec<u64>> = vec![vec![0, 5], vec![1, 4], vec![2, 3]];
        let xs: Vec<Vec<u64>> = vec![vec![0, 5], vec![2, 3]];
        let bytes = reflect_certify_subset(&xs, &r).expect("genuine subset must certify");
        assert!(
            reflect_verify_subset(&xs, &r, &bytes),
            "round-trip verifies"
        );
        // Superset swap: the same bytes cannot prove R ⊆ xs.
        assert!(
            !reflect_verify_subset(&r, &xs, &bytes),
            "superset swap must reject"
        );
        // Mutated bytes: garbage and a WRONG-CONSTANT proof term both reject.
        assert!(!reflect_verify_subset(&xs, &r, b"garbage"));
        let wrong = crate::cleancic::expr_to_bytes(&ap(c1("Eq.refl"), [boolc(), c("Bool.false")]))
            .expect("serialize");
        assert!(
            !reflect_verify_subset(&xs, &r, &wrong),
            "Eq.refl at false must reject"
        );
        // A non-subset never certifies (the kernel reduces Subseq to false — fail-closed).
        let non: Vec<Vec<u64>> = vec![vec![9, 9]];
        assert!(reflect_certify_subset(&non, &r).is_none());
    }

    /// The resource fallback proves each canonical LHS chunk against a literal ordered window of the
    /// RHS. Pin both directions directly: complete chunk coverage accepts, while one foreign tuple is
    /// rejected even though the stored proof token is the same constant `Eq.refl` used by real certs.
    #[test]
    fn reflect_subset_partition_fallback_is_exact() {
        let renv = cached_env().expect("reflect env");
        let ys: Vec<Vec<u64>> = (0..32u64).map(|i| vec![i, 31 - i]).collect();
        let xs: Vec<Vec<u64>> = ys.iter().step_by(2).cloned().collect();
        let (_, proof) = reflect_subset_bool_eq(&[], &[]);
        assert!(
            reflect_check_subset_partitioned(renv, &xs, &ys, &proof).is_some(),
            "every partition is a genuine subset of its RHS window"
        );
        let mut bad = xs;
        bad.push(vec![99, 99]);
        bad.sort_unstable();
        assert!(
            reflect_check_subset_partitioned(renv, &bad, &ys, &proof).is_none(),
            "a foreign tuple must refute its partition"
        );
    }

    /// Required scale regression: the reflected subset legs at the CoffeeCan-100 magnitude —
    /// |R| = 5151 1-wide tuples, |xs| = 5050 — certify + verify within the scaled heartbeat
    /// budget and a debug-build-friendly wall clock. This is the evidence gating the
    /// `DEFAULT_FIXPOINT_STATE_CAP` raise, so it belongs in the normal suite.
    #[test]
    fn reflect_subset_scale_probe() {
        let ys: Vec<Vec<u64>> = (0..5151u64).map(|v| vec![v]).collect();
        let xs: Vec<Vec<u64>> = (0..5151u64)
            .filter(|v| v % 51 != 0)
            .map(|v| vec![v])
            .collect();
        let t0 = std::time::Instant::now();
        let bytes = reflect_certify_subset(&xs, &ys).expect("5K-scale subset must certify");
        let mint = t0.elapsed();
        let t1 = std::time::Instant::now();
        assert!(reflect_verify_subset(&xs, &ys, &bytes));
        let verify = t1.elapsed();
        // Negative control at scale: a single foreign tuple refutes.
        let mut bad = xs.clone();
        bad.push(vec![999_999]);
        bad.sort_unstable();
        let t2 = std::time::Instant::now();
        assert!(reflect_certify_subset(&bad, &ys).is_none());
        let refute = t2.elapsed();
        println!("scale probe: mint={mint:?} verify={verify:?} refute={refute:?}");
    }

    /// The cross-embedding comparator: deep and shallow lanes agree on true AND false
    /// instances of the 2-column obligation, and decline (None) out-of-scope / OOB inputs.
    #[test]
    fn reflect_agrees_with_shallow_embedder() {
        let ir = two_col_ir();
        assert_eq!(
            reflect_agrees_with_shallow(&ir, &[2, 3], &[2, 4]),
            Some(true)
        );
        assert_eq!(
            reflect_agrees_with_shallow(&ir, &[2, 3], &[9, 4]),
            Some(true)
        );
        let set_ir = out_of_frag();
        assert_eq!(reflect_agrees_with_shallow(&set_ir, &[0], &[0]), None);
        assert_eq!(
            reflect_agrees_with_shallow(&lt(v(5), l(1)), &[2, 3], &[2, 3]),
            None
        );
    }

    // ── SET-fragment extension (bitmask Set*/SetCard/CountFold) — the kernel is the arbiter ─────────

    /// The bitmask SET predicates reduce correctly through the DEEP evaluator over a single Set column
    /// (`s[0]` = the mask). Both truth directions are the KERNEL's verdict; a false instance DISAGREES
    /// (never rubber-stamps). Masks: `5 = 0b101 = {0,2}`, `6 = 0b110 = {1,2}`.
    #[test]
    fn reflect_set_predicates_corroborate_and_disagree() {
        let corr = |ir: &PredIR, s: &[u64], want: bool| {
            assert_eq!(
                reflect_corroborate(ir, s, s, want),
                ReflectOutcome::Corroborated,
                "expected {want} for {ir:?} at {s:?}"
            );
        };
        // e ∈ S : bit e of s[0].
        corr(&PredIR::SetMem(0, SetIR::Var(0)), &[5], true); // bit0(5)=1
        corr(&PredIR::SetMem(1, SetIR::Var(0)), &[5], false); // bit1(5)=0
        corr(&PredIR::SetMem(2, SetIR::Var(0)), &[5], true); // bit2(5)=1
        corr(&PredIR::SetNotMem(1, SetIR::Var(0)), &[5], true);
        // S = C / S ≠ C.
        corr(&PredIR::SetEq(SetIR::Var(0), SetIR::Lit(5)), &[5], true);
        corr(&PredIR::SetEq(SetIR::Var(0), SetIR::Lit(4)), &[5], false);
        corr(&PredIR::SetNeq(SetIR::Var(0), SetIR::Lit(4)), &[5], true);
        // S ⊆ T : {0}⊆{0,2} true; {1}⊄{0,2} false.
        corr(
            &PredIR::SetSubseteq(SetIR::Lit(1), SetIR::Var(0)),
            &[5],
            true,
        );
        corr(
            &PredIR::SetSubseteq(SetIR::Lit(2), SetIR::Var(0)),
            &[5],
            false,
        );
        // Cup / Cap: {0}∪{1}={0,1}=3 ; {1,2}∩{0,2}={2}=4.
        corr(
            &PredIR::SetEq(
                SetIR::Cup(Box::new(SetIR::Lit(1)), Box::new(SetIR::Lit(2))),
                SetIR::Lit(3),
            ),
            &[0],
            true,
        );
        corr(
            &PredIR::SetEq(
                SetIR::Cap(Box::new(SetIR::Lit(6)), Box::new(SetIR::Var(0))),
                SetIR::Lit(4),
            ),
            &[5],
            true,
        );
        // UNCHANGED S: mask cell equality (s==sp true; s≠sp false).
        assert_eq!(
            reflect_corroborate(&PredIR::SetUnchanged(0), &[5], &[5], true),
            ReflectOutcome::Corroborated
        );
        assert!(matches!(
            reflect_corroborate(&PredIR::SetUnchanged(0), &[5], &[4], true),
            ReflectOutcome::Disagree(_)
        ));
        // A false membership expected-true is a definitive DISAGREE (can-say-no).
        assert!(matches!(
            reflect_corroborate(&PredIR::SetMem(1, SetIR::Var(0)), &[5], &[5], true),
            ReflectOutcome::Disagree(_)
        ));
    }

    /// `Cardinality(S)` (bitmask popcount) reduces to the true set-bit count through the DEEP evaluator:
    /// `|{0,2}| = 2` for mask `5`, `|∅| = 0` for mask `0`, `|{0,1,2,3}| = 4` for mask `15`.
    #[test]
    fn reflect_setcard_popcount_definitional() {
        let card_eq = |mask: u64, k: u64, want: bool| {
            let ir = PredIR::Eq(
                ValIR::SetCard {
                    set: SetIR::Var(0),
                    universe: 4,
                },
                ValIR::Lit(k),
            );
            assert_eq!(
                reflect_corroborate(&ir, &[mask], &[mask], want),
                ReflectOutcome::Corroborated,
                "|mask {mask}| {} {k}",
                if want { "==" } else { "!=" }
            );
        };
        card_eq(5, 2, true);
        card_eq(5, 3, false);
        card_eq(0, 0, true);
        card_eq(15, 4, true);
        card_eq(15, 3, false);
        // Cardinality inside an ORDERING: |S| ≤ 2 holds for mask 5, fails for mask 7 (=3 bits).
        let leq2 = |mask: u64, want: bool| {
            let ir = PredIR::Leq(
                ValIR::SetCard {
                    set: SetIR::Var(0),
                    universe: 4,
                },
                ValIR::Lit(2),
            );
            assert_eq!(
                reflect_corroborate(&ir, &[mask], &[mask], want),
                ReflectOutcome::Corroborated
            );
        };
        leq2(5, true);
        leq2(7, false);
    }

    /// `CountFold` (set-comprehension counting `Σ_d boolToNat(P(d))`) reduces to the true count through
    /// the DEEP evaluator, via the arithmetic-identity `boolToNat`. `terms = [x0=1, x1=1, x2=1]` counts
    /// the columns equal to 1: `[1,0,1] ↦ 2`, `[1,1,1] ↦ 3`, `[0,0,0] ↦ 0`.
    #[test]
    fn reflect_countfold_counts_definitional() {
        let terms = vec![eqp(v(0), l(1)), eqp(v(1), l(1)), eqp(v(2), l(1))];
        let count_eq = |s: &[u64], k: u64, want: bool| {
            let ir = PredIR::Eq(
                ValIR::CountFold {
                    terms: terms.clone(),
                },
                ValIR::Lit(k),
            );
            assert_eq!(
                reflect_corroborate(&ir, s, s, want),
                ReflectOutcome::Corroborated,
                "count{s:?} {} {k}",
                if want { "==" } else { "!=" }
            );
        };
        count_eq(&[1, 0, 1], 2, true);
        count_eq(&[1, 0, 1], 3, false);
        count_eq(&[1, 1, 1], 3, true);
        count_eq(&[0, 0, 0], 0, true);
    }

    /// FAITHFULNESS PIN (the decisive can-say-no guard): the DEEP set-fragment evaluator and the shallow
    /// embedder must AGREE on the SAME IR over a GRID of masks/states — a wrong quote arm (a boolToNat
    /// identity that mis-computes, a flipped `⊆`) would make the two op-realizations DISAGREE and this
    /// test would fail. Covers every covered set predicate + `SetCard` + `CountFold` across `0..16`.
    #[test]
    fn reflect_agrees_with_shallow_on_set_fragment() {
        // Single Set column: exhaustively over masks 0..16.
        for m in 0..16u64 {
            let preds: Vec<PredIR> = vec![
                PredIR::SetMem(0, SetIR::Var(0)),
                PredIR::SetMem(2, SetIR::Var(0)),
                PredIR::SetNotMem(3, SetIR::Var(0)),
                PredIR::SetEq(SetIR::Var(0), SetIR::Lit(5)),
                PredIR::SetNeq(SetIR::Var(0), SetIR::Lit(6)),
                PredIR::SetSubseteq(SetIR::Lit(5), SetIR::Var(0)),
                PredIR::SetSubseteq(SetIR::Var(0), SetIR::Lit(7)),
                PredIR::SetEq(
                    SetIR::Cup(Box::new(SetIR::Var(0)), Box::new(SetIR::Lit(1))),
                    SetIR::Lit(m | 1),
                ),
                PredIR::SetEq(
                    SetIR::Cap(Box::new(SetIR::Var(0)), Box::new(SetIR::Lit(6))),
                    SetIR::Lit(m & 6),
                ),
                PredIR::Eq(
                    ValIR::SetCard {
                        set: SetIR::Var(0),
                        universe: 4,
                    },
                    ValIR::Lit(2),
                ),
                PredIR::Leq(
                    ValIR::SetCard {
                        set: SetIR::Var(0),
                        universe: 4,
                    },
                    ValIR::Lit(2),
                ),
            ];
            for ir in &preds {
                assert_eq!(
                    reflect_agrees_with_shallow(ir, &[m], &[m]),
                    Some(true),
                    "deep/shallow DISAGREE on {ir:?} at mask {m} — set-embedding exactness bug"
                );
            }
        }
        // CountFold over a 3-column Bool-ish state: agreement across a grid of tuples.
        let terms = vec![
            eqp(v(0), l(1)),
            PredIR::Or(Box::new(eqp(v(1), l(1))), Box::new(eqp(v(2), l(1)))),
            PredIR::Not(Box::new(eqp(v(2), l(1)))),
        ];
        for a in 0..2u64 {
            for b in 0..2u64 {
                for cc in 0..2u64 {
                    for k in 0..4u64 {
                        let ir = PredIR::Eq(
                            ValIR::CountFold {
                                terms: terms.clone(),
                            },
                            ValIR::Lit(k),
                        );
                        assert_eq!(
                            reflect_agrees_with_shallow(&ir, &[a, b, cc], &[a, b, cc]),
                            Some(true),
                            "deep/shallow DISAGREE on CountFold at [{a},{b},{cc}] vs {k}"
                        );
                    }
                }
            }
        }
    }

    /// HourClock-shaped Safety `1 ≤ hr ∧ hr ≤ 12` over R = {1..12}: the reflected discharge is
    /// `Safe` (every reachable `hr` satisfies it) — the embedder-free `R ⊆ Safety` accept.
    #[test]
    fn reflect_safety_over_reachable_hourclock_safe() {
        // `And(Leq(Lit(1), Var(0)), Leq(Var(0), Lit(12)))` — exactly HourClock's recognized IR.
        let safety = band(leq(l(1), v(0)), leq(v(0), l(12)));
        let r: Vec<Vec<u64>> = (1..=12u64).map(|h| vec![h]).collect();
        assert_eq!(
            reflect_safety_over_reachable(&safety, &r),
            ReflectSafetyOutcome::Safe
        );
        // Vacuous: an empty reachable set is `Safe` (no state falsifies the invariant).
        assert_eq!(
            reflect_safety_over_reachable(&safety, &[]),
            ReflectSafetyOutcome::Safe
        );
    }

    /// THE DECISIVE SOUNDNESS TEST: a VIOLATED invariant `hr < 12` over a reachable set that
    /// INCLUDES `hr = 12` must be reported `NotSafe` at that state — the deep evaluator reduces
    /// `12 < 12` to `Bool.false`. It must NOT falsely report `Safe`.
    #[test]
    fn reflect_safety_over_reachable_declines_reachable_violation() {
        let bad = lt(v(0), l(12)); // Safety == hr < 12 — FALSE at the reachable hr=12
        let r: Vec<Vec<u64>> = (1..=12u64).map(|h| vec![h]).collect();
        match reflect_safety_over_reachable(&bad, &r) {
            ReflectSafetyOutcome::NotSafe { state, .. } => assert_eq!(state, vec![12]),
            other => panic!("a reachable invariant violation must be NotSafe, got {other:?}"),
        }
        // Restricting R to the states that DO satisfy `hr < 12` flips it back to Safe — the
        // outcome tracks the actual states, not the predicate alone.
        let r_ok: Vec<Vec<u64>> = (1..=11u64).map(|h| vec![h]).collect();
        assert_eq!(
            reflect_safety_over_reachable(&bad, &r_ok),
            ReflectSafetyOutcome::Safe
        );
    }

    /// The reflected eval and the shallow embedder AGREE on the HourClock safety obligation across
    /// every reachable state (12 in-fragment (ir, s) pairs) PLUS violating states — if they ever
    /// disagreed on an in-fragment case that would be a bug. Pins the `reflect_agrees_with_shallow`
    /// discipline for THIS obligation (the leg's soundness rests on the two lanes agreeing).
    #[test]
    fn reflect_safety_agrees_with_shallow_on_hourclock() {
        let ok = band(leq(l(1), v(0)), leq(v(0), l(12))); // 1 ≤ hr ∧ hr ≤ 12
        let bad = lt(v(0), l(12)); // hr < 12

        // Both the true invariant and the violated one, over every state hr ∈ 1..=13.
        for hr in 1..=13u64 {
            for ir in [&ok, &bad] {
                assert_eq!(
                    reflect_agrees_with_shallow(ir, &[hr], &[hr]),
                    Some(true),
                    "reflected/shallow DISAGREE at hr={hr} on {ir:?} — soundness bug"
                );
            }
        }
    }

    /// FRAGMENT-BOUNDARY probe: an invariant using a form the quoter DECLINES (a Set node) makes
    /// the reflected discharge `Inconclusive` (fail-closed) — never `Safe`, never `NotSafe`.
    #[test]
    fn reflect_safety_over_reachable_inconclusive_out_of_fragment() {
        let set_ir = out_of_frag(); // a bounded quantifier — out-of-fragment
        assert!(matches!(
            reflect_safety_over_reachable(&set_ir, &[vec![0]]),
            ReflectSafetyOutcome::Inconclusive(_)
        ));
        // An out-of-BOUNDS column is also declined (the deep `nth` would totalize to 0).
        let oob = lt(v(3), l(1)); // column 3 over 1-column states
        assert!(matches!(
            reflect_safety_over_reachable(&oob, &[vec![5]]),
            ReflectSafetyOutcome::Inconclusive(_)
        ));
    }

    // ── R2 milestone: the reflected COMPLETENESS legs ────────────────────────────────────────

    /// The reflected IMPLICATION `⟦ir⟧(s,sp) ⇒ mem∈R`: TRUE (antecedent+member), TRUE (vacuous —
    /// antecedent false), FALSE (antecedent true but non-member ⇒ the decisive Disagree), and
    /// Unavailable (out-of-fragment). Every verdict is the KERNEL's, never a Rust re-fold.
    #[test]
    fn reflect_implies_mem_true_vacuous_false_unavailable() {
        let r: Vec<Vec<u64>> = vec![vec![1], vec![2]];
        // Eq(x,2) at s=[2], mem=[2]∈R: antecedent TRUE, member ⇒ Corroborated.
        let ir = eqp(v(0), l(2));
        assert_eq!(
            reflect_implies_mem(&ir, &[2], &[2], &[2], &r),
            ReflectOutcome::Corroborated
        );
        // Eq(x,2) at s=[3], mem=[3]∉R: antecedent FALSE ⇒ implication VACUOUSLY true.
        assert_eq!(
            reflect_implies_mem(&ir, &[3], &[3], &[3], &r),
            ReflectOutcome::Corroborated
        );
        // Eq(x,3) at s=[3], mem=[3]∉R: antecedent TRUE but non-member ⇒ Disagree (Bool.false).
        let ir2 = eqp(v(0), l(3));
        assert!(matches!(
            reflect_implies_mem(&ir2, &[3], &[3], &[3], &r),
            ReflectOutcome::Disagree(_)
        ));
        // Out-of-fragment antecedent (a bounded quantifier) ⇒ Unavailable (fail-closed, never mis-evaluated).
        let set_ir = out_of_frag();
        assert!(matches!(
            reflect_implies_mem(&set_ir, &[1], &[1], &[1], &r),
            ReflectOutcome::Unavailable(_)
        ));
    }

    /// HourClock Init-completeness `∀ s∈D: Init(s) ⇒ s∈R` over D=0..=12: Complete for the full
    /// R={1..12}; NotComplete at [1] once [1] is dropped (Init(1) holds but 1∉R).
    #[test]
    fn reflect_init_completeness_hourclock_complete_and_missing() {
        let init = band(leq(l(1), v(0)), leq(v(0), l(12)));
        let domain: Vec<Vec<u64>> = (0..=12u64).map(|h| vec![h]).collect();
        let r: Vec<Vec<u64>> = (1..=12u64).map(|h| vec![h]).collect();
        assert_eq!(
            reflect_init_completeness_over_domain(&init, &domain, &r),
            ReflectInitOutcome::Complete { states: 13 }
        );
        let r_missing: Vec<Vec<u64>> = (2..=12u64).map(|h| vec![h]).collect();
        match reflect_init_completeness_over_domain(&init, &domain, &r_missing) {
            ReflectInitOutcome::NotComplete { s, .. } => assert_eq!(s, vec![1]),
            other => panic!("a missing Init state must be NotComplete, got {other:?}"),
        }
    }

    /// THE DECISIVE CLOSURE TEST: HourClock Next-completeness `∀ s∈R, sp∈D: Next(s,sp) ⇒ sp∈R` over
    /// D=0..=13. Closed for the full R={1..12}; NotClosed at (s=[12], sp=[1]) once [1] is dropped —
    /// the deep evaluator reduces `Next(12,1) ⇒ 1∈R` to Bool.false (Next(12,1) holds, 1∉R). A
    /// non-closed R must NEVER read as Closed.
    #[test]
    fn reflect_next_completeness_hourclock_closed_and_nonclosed() {
        // Next = IF hr#12 THEN hr'=hr+1 ELSE hr'=1 (the recognized Or-desugaring).
        let next = PredIR::Or(
            Box::new(band(
                PredIR::Neq(v(0), l(12)),
                PredIR::Eq(p(0), ValIR::Add(Box::new(v(0)), Box::new(l(1)))),
            )),
            Box::new(band(
                PredIR::Not(Box::new(PredIR::Neq(v(0), l(12)))),
                PredIR::Eq(p(0), l(1)),
            )),
        );
        let domain: Vec<Vec<u64>> = (0..=13u64).map(|h| vec![h]).collect();
        let r: Vec<Vec<u64>> = (1..=12u64).map(|h| vec![h]).collect();
        assert!(matches!(
            reflect_next_completeness_over_domain(&next, &r, &domain),
            ReflectClosureOutcome::Closed { .. }
        ));
        // Drop [1]: Next(12,1) holds, 1∉R ⇒ NotClosed at (s=[12], sp=[1]).
        let r_missing: Vec<Vec<u64>> = (2..=12u64).map(|h| vec![h]).collect();
        match reflect_next_completeness_over_domain(&next, &r_missing, &domain) {
            ReflectClosureOutcome::NotClosed { s, sp, .. } => {
                assert_eq!(s, vec![12]);
                assert_eq!(sp, vec![1]);
            }
            other => panic!("a dropped successor must be NotClosed, got {other:?}"),
        }
    }
}
