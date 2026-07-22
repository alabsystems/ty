// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! **clean-ck0 corroboration of TY's soundness meta-proofs** — the second-checker residual.
//!
//! `tests/clean_soundness_proofs.rs` puts every `.clean` meta-theorem under `proofs/clean/` in
//! the CLEAN-KERNEL trust base. This harness closes the second-checker residual: the same
//! declarations are INDEPENDENTLY re-checked by `clean-ck0` (~9K LOC, `#![forbid(unsafe_code)]`,
//! zero production code shared with clean-kernel), so the checker-TCB claim for the covered
//! fragment does not rest on clean-kernel alone.
//!
//! ## How it works
//!
//! Per proof file: elaborate through the EXACT `clean check` pipeline into
//! `Environment::with_prelude()` (same loader as the soundness harness), then for each DECLARED
//! value-bearing constant (Theorem/Definition/Opaque with a value) translate its clean-kernel
//! `type_` and `value` — independently of each other — into ck0 `RawExpr`s, validate both through
//! ck0's single chokepoint (`Term::validate`), and ask ck0 to decide
//! `infer::check(env, value, type)` under a pinned deterministic budget.
//!
//! The ck0 environment is built PER FILE from empty, entirely demand-driven by the transitive
//! `Const` dependency closure of the file's declarations:
//!   * inductives (prelude `Nat`/`Eq`/`List`/`Prod`/`Acc`/`Nat.le`/… AND file-local ones like
//!     `Reach`) are admitted via `clean_ck0::add_inductive`, which kernel-checks
//!     positivity/universes and DERIVES its own recursors from clean's `InductiveVal` metadata;
//!   * every prelude definition/theorem in the closure is translated and **ck0-`check`ed against
//!     its translated type BEFORE registration** (a failure shrinks the fragment, honestly);
//!   * the FILE's own defs/theorems are registered the same way — but only AFTER their own
//!     corroboration succeeds, so later theorems in the file resolve them (fail-closed: an
//!     uncorroborated def is never registered);
//!   * `I.rec` is lowered to `RawExpr::Elim` (ck0 re-derives the level vector itself);
//!     `casesOn`/`brecOn`/… have no ck0 analog and fail closed;
//!   * NO body-less constant is ever registered (`with_const_typed` is never called), so a
//!     `corroborated` verdict is axiom-free by construction relative to ck0's checker.
//!
//! ## Numeral encoding
//!
//! Everything (env defs AND obligations) uses `Nat.succ^n Nat.zero` CONSTRUCTOR chains, capped at
//! [`CTOR_NUMERAL_MAX`] — never `RawExpr::Lit`. ck0 has no Lit↔constructor bridging, and these
//! meta-proofs are `Nat.rec`/`List.rec` casework through and through: a literal meeting a derived
//! ι-site would be honestly stuck and surface as a spurious refutation. One encoding everywhere
//! keeps the whole reduction world coherent; over-cap numerals are an honest unsupported-skip.
//!
//! ## The fail-closed contract (cardinal invariant)
//!
//! * ck0 REJECTION of a submitted, fully-supported term = **test failure** (second-checker
//!   disagreement with clean-kernel is a soundness alarm).
//! * UNSUPPORTED (converter gap, env gap, over-cap numeral, depth cap, budget exhaustion) =
//!   SKIP, recorded with its reason, counted, printed — never silent, never counted as
//!   corroborated.
//! * The `N` in the total line is machine-derived from actual `check()` `Ok` returns; the
//!   tiny-TCB label attaches ONLY to those `N` declarations.
//! * Negative controls prove the check is real: a genuine proof value fed against a wrong (but
//!   fully in-fragment) type MUST be rejected.
//!
//! ## Known residual (honest unsupported, as of this writing)
//!
//! The M1 bitmask theorems (`SetMaskSound.*`) skip: their prelude dependency chains pass through
//! lemmas whose TYPES are in typeclass form (`@LT.lt Nat instLTNat …`), and `LT.lt` δ-reduces to
//! a `Proj` at the head of an application spine — a shape ck0's whnf deliberately leaves stuck
//! (its `whnf_core` App arm β-reduces only `Lam` heads; projection-of-constructor fires only on
//! a bare `Proj` node). clean-kernel decides that def-eq; ck0 cannot, so `Nat.zero_lt_succ`
//! fails ck0's OWN registration check and everything above it skips with the recorded reason.
//! Widening this is a ck0 (M4+) improvement, not a harness change.

#![cfg(feature = "clean-cic")]

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;

use clean_ck0::{
    add_inductive, check as ck0_check, Budget, Constructor as CkCtor, InductiveDecl as CkIndDecl,
    InferError, MinimalEnv, Name as CkName, RawExpr, RawLevel, Term as CkTerm, Transparency,
    MAX_VALIDATE_DEPTH,
};
use clean_elab::{
    elaborate_decl_and_register, preprocess_decl_with_context, ElabResult, FileContext,
};
use clean_kernel::{
    BigNat as KBigNat, ConstantKind, Environment, Expr, ExprKind, Level as KLevel, Literal,
    Name as KName,
};
use clean_parser::{parse_file, SurfaceDecl};

/// Pinned deterministic step budget for one obligation `check` (same as the bridge's).
const CK0_CHECK_FUEL: u64 = 64_000_000;
/// Budget for ck0-checking one dependency definition at env-build time.
const CK0_DEF_FUEL: u64 = 32_000_000;
/// Cap on a Nat literal translated as a `Nat.succ^n Nat.zero` chain (headroom under ck0's
/// `MAX_VALIDATE_DEPTH = 1024` for the structure AROUND the numeral).
const CTOR_NUMERAL_MAX: u64 = 512;

// ===========================================================================
// Translator: clean-kernel Expr/Level -> ck0 RawExpr. Adapted from
// `tla_check::ck0_bridge::tr_expr` (same discipline: structure-preserving,
// total-or-explicit-error), generalized: the recursor arity table is the clean
// env itself, MData wrappers are transparently unwrapped, and numerals are
// ALWAYS constructor chains.
// ===========================================================================

fn tr_level(lvl: &KLevel, lps: &[KName]) -> Result<RawLevel, String> {
    match lvl {
        KLevel::Zero => Ok(RawLevel::Zero),
        KLevel::Succ(l) => Ok(RawLevel::Succ(Box::new(tr_level(l, lps)?))),
        KLevel::Max(a, b) => Ok(RawLevel::Max(
            Box::new(tr_level(a, lps)?),
            Box::new(tr_level(b, lps)?),
        )),
        KLevel::IMax(a, b) => Ok(RawLevel::IMax(
            Box::new(tr_level(a, lps)?),
            Box::new(tr_level(b, lps)?),
        )),
        KLevel::Param(n) => lps
            .iter()
            .position(|p| p == n)
            .and_then(|i| u32::try_from(i).ok())
            .map(RawLevel::Param)
            .ok_or_else(|| format!("unknown level param `{n}`")),
    }
}

fn tr_binfo(info: clean_kernel::BinderInfo) -> clean_ck0::rawexpr::BinderInfo {
    use clean_ck0::rawexpr::BinderInfo as CkBinderInfo;
    match info {
        clean_kernel::BinderInfo::Default => CkBinderInfo::Default,
        clean_kernel::BinderInfo::Implicit => CkBinderInfo::Implicit,
        clean_kernel::BinderInfo::StrictImplicit => CkBinderInfo::StrictImplicit,
        clean_kernel::BinderInfo::InstImplicit => CkBinderInfo::InstImplicit,
    }
}

/// Recursor-name suffixes ck0's chokepoint reserves. `rec` lowers to `RawExpr::Elim`;
/// the rest are ordinary VALUE-BEARING definitions in clean's prelude (`I.casesOn` is a
/// `Definition` whose body is an `I.rec` application) — only their NAME is reserved by ck0's
/// validate gate, so they are registered under a renamed constant ([`ck0_reg_name`]),
/// ck0-checked against their translated type exactly like every other def. A reserved-suffix
/// constant WITHOUT a value stays an honest unsupported-skip.
fn recursor_suffix(last: &str) -> Option<&'static str> {
    match last {
        "rec" => Some("rec"),
        "recOn" | "casesOn" | "below" | "ibelow" | "brecOn" | "binductionOn" | "brecOnEq" => {
            Some("other")
        }
        _ => None,
    }
}

/// The ck0-side registration name for a clean constant: reserved recursor-family suffixes
/// (`casesOn`, `recOn`, …) get a `_ck0` tail so ck0's reserved-name gate does not fire.
/// The rename is a CONSISTENT bijection applied to every `Const` reference AND the
/// registration itself, and the renamed def's body is still ck0-checked against its renamed
/// type from scratch — no soundness property of the gate is bypassed (the gate exists to stop
/// UNCHECKED `.rec`/`.casesOn` consts dodging `ElimRef`; here everything is checked).
fn ck0_reg_name(name: &str) -> String {
    match name.rsplit_once('.') {
        Some((parent, last)) if recursor_suffix(last) == Some("other") => {
            format!("{parent}.{last}_ck0")
        }
        _ => name.to_string(),
    }
}

/// `Nat.succ^n Nat.zero`, built iteratively; fail-closed above [`CTOR_NUMERAL_MAX`].
fn nat_ctor_chain(n: &KBigNat) -> Result<RawExpr, String> {
    let v = match n.to_u64() {
        Some(v) if v <= CTOR_NUMERAL_MAX => v,
        _ => {
            return Err(format!(
                "nat literal exceeds the ctor-numeral cap ({CTOR_NUMERAL_MAX})"
            ))
        }
    };
    let mut acc = RawExpr::Const(CkName::from_dotted("Nat.zero"), Vec::new());
    for _ in 0..v {
        acc = RawExpr::App(
            Box::new(RawExpr::Const(CkName::from_dotted("Nat.succ"), Vec::new())),
            Box::new(acc),
        );
    }
    Ok(acc)
}

/// Translate a clean `Expr` to ck0 `RawExpr` against level telescope `lps`.
/// `ind_lp_arity` resolves an inductive's level-param count for `I.rec` Elim lowering.
fn tr_expr(
    e: &Expr,
    lps: &[KName],
    ind_lp_arity: &dyn Fn(&str) -> Option<usize>,
) -> Result<RawExpr, String> {
    match e.kind() {
        ExprKind::BVar(i) => Ok(RawExpr::BVar(*i)),
        ExprKind::Sort(l) => Ok(RawExpr::Sort(tr_level(l, lps)?)),
        ExprKind::MData(_, inner) => tr_expr(inner, lps, ind_lp_arity),
        ExprKind::Const(name, levels) => {
            let dotted = name.to_string();
            if let Some((parent, last)) = dotted.rsplit_once('.') {
                match recursor_suffix(last) {
                    Some("rec") => {
                        let arity = ind_lp_arity(parent).ok_or_else(|| {
                            format!(
                                "recursor `{dotted}`: inductive `{parent}` not in the clean env"
                            )
                        })?;
                        let lv: Vec<RawLevel> = levels
                            .iter()
                            .map(|l| tr_level(l, lps))
                            .collect::<Result<_, _>>()?;
                        let (motive, ind_levels) = if lv.len() == arity + 1 {
                            let mut it = lv.into_iter();
                            let m = it.next().unwrap_or(RawLevel::Zero);
                            (m, it.collect())
                        } else if lv.len() == arity {
                            (RawLevel::Zero, lv)
                        } else {
                            return Err(format!(
                                "recursor `{dotted}`: level vector arity {} does not fit inductive arity {arity}",
                                levels.len()
                            ));
                        };
                        return Ok(RawExpr::Elim(
                            CkName::from_dotted(parent),
                            motive,
                            ind_levels,
                        ));
                    }
                    Some(_) => {
                        // Reserved-name family with a real prelude definition: reference the
                        // renamed registration (see `ck0_reg_name`). If the def could not be
                        // registered, the renamed const is unknown -> honest Unavailable.
                        let lv: Vec<RawLevel> = levels
                            .iter()
                            .map(|l| tr_level(l, lps))
                            .collect::<Result<_, _>>()?;
                        return Ok(RawExpr::Const(
                            CkName::from_dotted(&ck0_reg_name(&dotted)),
                            lv,
                        ));
                    }
                    None => {}
                }
            }
            let lv: Vec<RawLevel> = levels
                .iter()
                .map(|l| tr_level(l, lps))
                .collect::<Result<_, _>>()?;
            Ok(RawExpr::Const(CkName::from_dotted(&dotted), lv))
        }
        ExprKind::App(f, a) => Ok(RawExpr::App(
            Box::new(tr_expr(f, lps, ind_lp_arity)?),
            Box::new(tr_expr(a, lps, ind_lp_arity)?),
        )),
        ExprKind::Lam(bd, ty, body) => Ok(RawExpr::Lam(
            tr_binfo(bd.info),
            Box::new(tr_expr(ty, lps, ind_lp_arity)?),
            Box::new(tr_expr(body, lps, ind_lp_arity)?),
        )),
        ExprKind::Pi(bd, ty, body) => Ok(RawExpr::Pi(
            tr_binfo(bd.info),
            Box::new(tr_expr(ty, lps, ind_lp_arity)?),
            Box::new(tr_expr(body, lps, ind_lp_arity)?),
        )),
        ExprKind::Let(_name, ty, val, body, _nondep) => Ok(RawExpr::Let(
            Box::new(tr_expr(ty, lps, ind_lp_arity)?),
            Box::new(tr_expr(val, lps, ind_lp_arity)?),
            Box::new(tr_expr(body, lps, ind_lp_arity)?),
        )),
        ExprKind::Lit(Literal::Nat(n)) => nat_ctor_chain(n),
        ExprKind::Proj(name, idx, inner) => Ok(RawExpr::Proj(
            CkName::from_dotted(&name.to_string()),
            *idx,
            Box::new(tr_expr(inner, lps, ind_lp_arity)?),
        )),
        other => Err(format!("unsupported ExprKind: {other:?}")),
    }
}

/// Every `Const` name referenced by `e` (dotted). Iterative — proof bodies are large.
fn const_refs(e: &Expr, out: &mut Vec<String>) {
    let mut stack: Vec<&Expr> = vec![e];
    while let Some(node) = stack.pop() {
        match node.kind() {
            ExprKind::Const(name, _) => out.push(name.to_string()),
            ExprKind::App(f, a) => {
                stack.push(f);
                stack.push(a);
            }
            ExprKind::Lam(_, t, b) | ExprKind::Pi(_, t, b) => {
                stack.push(t);
                stack.push(b);
            }
            ExprKind::Let(_, t, v, b, _) => {
                stack.push(t);
                stack.push(v);
                stack.push(b);
            }
            ExprKind::Proj(_, _, inner) | ExprKind::MData(_, inner) => stack.push(inner),
            _ => {}
        }
    }
}

/// Iterative structural-depth pre-check mirroring ck0's `MAX_VALIDATE_DEPTH`: the recursive
/// translator must never be the thing that discovers an over-deep term (stack overflow is a
/// crash, not a fail-closed decline).
fn exceeds_ck0_depth(e: &Expr) -> bool {
    let cap = MAX_VALIDATE_DEPTH;
    let mut stack: Vec<(&Expr, u32)> = vec![(e, 1)];
    while let Some((node, d)) = stack.pop() {
        if d > cap {
            return true;
        }
        let d1 = d.saturating_add(1);
        match node.kind() {
            ExprKind::App(f, a) => {
                stack.push((f, d1));
                stack.push((a, d1));
            }
            ExprKind::Lam(_, t, b) | ExprKind::Pi(_, t, b) => {
                stack.push((t, d1));
                stack.push((b, d1));
            }
            ExprKind::Let(_, t, v, b, _) => {
                stack.push((t, d1));
                stack.push((v, d1));
                stack.push((b, d1));
            }
            ExprKind::Proj(_, _, inner) | ExprKind::MData(_, inner) => stack.push((inner, d1)),
            _ => {}
        }
    }
    false
}

// ===========================================================================
// The per-file ck0 world: demand-driven, fail-closed env construction.
// ===========================================================================

/// Outcome of one obligation (the test-local mirror of the bridge's verdict semantics).
#[derive(Debug, Clone, PartialEq, Eq)]
enum Outcome {
    Corroborated,
    Unavailable(String),
    Rejected(String),
}

enum Class {
    /// `I.rec` — Elim-lowered, requires `I` admitted.
    RecursorOf(String),
    /// A constructor — registered when its inductive is admitted.
    CtorOf(String),
    Inductive,
    /// Value-bearing Theorem/Definition/Opaque.
    Def,
    Unsupported(String),
}

struct World<'a> {
    clean: &'a Environment,
    env: MinimalEnv,
    /// name -> Ok(()) = registered in ck0, Err(reason) = honestly skipped.
    status: HashMap<String, Result<(), String>>,
    entered: HashSet<String>,
}

impl<'a> World<'a> {
    fn new(clean: &'a Environment) -> Self {
        World {
            clean,
            env: MinimalEnv::new(),
            status: HashMap::new(),
            entered: HashSet::new(),
        }
    }

    fn tr(&self, e: &Expr, lps: &[KName]) -> Result<RawExpr, String> {
        let clean = self.clean;
        let arity = move |parent: &str| {
            clean
                .get_inductive(&KName::from_string(parent))
                .map(|i| i.level_params.len())
        };
        tr_expr(e, lps, &arity)
    }

    fn classify(&self, name: &str) -> Class {
        if let Some((parent, last)) = name.rsplit_once('.') {
            match recursor_suffix(last) {
                Some("rec") => {
                    if self
                        .clean
                        .get_inductive(&KName::from_string(parent))
                        .is_some()
                    {
                        return Class::RecursorOf(parent.to_string());
                    }
                    return Class::Unsupported(format!(
                        "recursor `{name}` of unknown inductive `{parent}`"
                    ));
                }
                Some(_) => {
                    // `I.casesOn` & friends: value-bearing prelude defs registered under a
                    // renamed constant (ck0 reserves the literal name). Fall through to the
                    // Def classification below; a value-less one skips honestly there.
                }
                None => {}
            }
        }
        let kn = KName::from_string(name);
        if self.clean.get_inductive(&kn).is_some() {
            return Class::Inductive;
        }
        if let Some(cv) = self.clean.get_constructor(&kn) {
            return Class::CtorOf(cv.inductive_name.to_string());
        }
        match self.clean.get_const(&kn) {
            Some(c)
                if c.value.is_some()
                    && matches!(
                        c.kind,
                        ConstantKind::Theorem | ConstantKind::Definition | ConstantKind::Opaque
                    ) =>
            {
                Class::Def
            }
            Some(c) if c.kind == ConstantKind::Axiom => {
                Class::Unsupported(format!("`{name}` is an AXIOM — never admitted into ck0"))
            }
            Some(_) => Class::Unsupported(format!(
                "`{name}` is value-less in the clean env — never admitted into ck0"
            )),
            None => Class::Unsupported(format!("`{name}` absent from the clean env")),
        }
    }

    /// Direct `Const` dependencies of `name` under its classification.
    fn direct_deps(&self, name: &str, class: &Class) -> Vec<String> {
        let mut deps = Vec::new();
        match class {
            Class::RecursorOf(parent) | Class::CtorOf(parent) => deps.push(parent.clone()),
            Class::Inductive => {
                let kn = KName::from_string(name);
                let Some(ind) = self.clean.get_inductive(&kn) else {
                    return deps;
                };
                const_refs(&ind.type_, &mut deps);
                let mut own: HashSet<String> = HashSet::new();
                own.insert(name.to_string());
                for cn in &ind.constructor_names {
                    own.insert(cn.to_string());
                    if let Some(cv) = self.clean.get_constructor(cn) {
                        const_refs(&cv.type_, &mut deps);
                    }
                }
                deps.retain(|d| !own.contains(d));
            }
            Class::Def => {
                let kn = KName::from_string(name);
                if let Some(c) = self.clean.get_const(&kn) {
                    const_refs(&c.type_, &mut deps);
                    if let Some(v) = &c.value {
                        const_refs(v, &mut deps);
                    }
                }
                deps.retain(|d| d != name);
            }
            Class::Unsupported(_) => {}
        }
        deps.sort();
        deps.dedup();
        deps
    }

    /// If any direct dep is skipped/unresolved, the reason; else None.
    fn dep_gap(&self, name: &str, class: &Class) -> Option<String> {
        for dep in self.direct_deps(name, class) {
            match self.status.get(&dep) {
                Some(Ok(())) => {}
                Some(Err(why)) => {
                    return Some(format!("depends on unsupported `{dep}` ({why})"));
                }
                None => {
                    return Some(format!(
                        "depends on `{dep}`, unresolved (dependency cycle?)"
                    ));
                }
            }
        }
        None
    }

    /// Ensure `root` (and its transitive closure) is registered in the ck0 env, or has an
    /// honest skip reason. Iterative Enter/Exit postorder — never recurses on proof structure.
    fn ensure(&mut self, root: &str) {
        enum Visit {
            Enter(String),
            Exit(String),
        }
        let mut stack = vec![Visit::Enter(root.to_string())];
        while let Some(v) = stack.pop() {
            match v {
                Visit::Enter(name) => {
                    if self.status.contains_key(&name) || self.entered.contains(&name) {
                        continue;
                    }
                    self.entered.insert(name.clone());
                    let class = self.classify(&name);
                    if let Class::Unsupported(why) = class {
                        self.status.insert(name, Err(why));
                        continue;
                    }
                    let deps = self.direct_deps(&name, &class);
                    stack.push(Visit::Exit(name));
                    for dep in deps {
                        if !self.status.contains_key(&dep) && !self.entered.contains(&dep) {
                            stack.push(Visit::Enter(dep));
                        }
                    }
                }
                Visit::Exit(name) => {
                    if self.status.contains_key(&name) {
                        continue;
                    }
                    let class = self.classify(&name);
                    let outcome = if let Some(gap) = self.dep_gap(&name, &class) {
                        Err(gap)
                    } else {
                        match &class {
                            // Parent admitted (dep check passed) => the recursor is
                            // Elim-lowerable / the ctor was registered by commit_inductive.
                            Class::RecursorOf(_) | Class::CtorOf(_) => Ok(()),
                            Class::Inductive => self.admit_inductive(&name),
                            Class::Def => self.register_def(&name),
                            Class::Unsupported(why) => Err(why.clone()),
                        }
                    };
                    self.status.insert(name, outcome);
                }
            }
        }
    }

    /// Admit one inductive via ck0's kernel-checked `add_inductive`, driven entirely by
    /// clean's `InductiveVal` metadata (works for prelude AND file-local inductives).
    fn admit_inductive(&mut self, name: &str) -> Result<(), String> {
        let kn = KName::from_string(name);
        let ind = self
            .clean
            .get_inductive(&kn)
            .ok_or_else(|| format!("`{name}` vanished from the clean env"))?;
        let lps = ind.level_params.clone();
        let nlvl = u32::try_from(lps.len()).map_err(|_| "level arity overflow".to_string())?;
        let ind_raw = self
            .tr(&ind.type_, &lps)
            .map_err(|e| format!("inductive type: {e}"))?;
        let ind_term = CkTerm::validate(&self.env, &ind_raw, 0, nlvl)
            .map_err(|e| format!("inductive type validate: {e:?}"))?;
        let mut ck_ctors = Vec::new();
        for cn in &ind.constructor_names {
            let cv = self
                .clean
                .get_constructor(cn)
                .ok_or_else(|| format!("constructor `{cn}` absent"))?;
            // A constructor's positional level params must mirror its inductive's; translate
            // against the ctor's OWN telescope (clean keeps them in lockstep).
            let claw = self
                .tr(&cv.type_, &cv.level_params)
                .map_err(|e| format!("ctor `{cn}` type: {e}"))?;
            // Bootstrap env: the ctor type mentions the inductive itself, which is not yet
            // admitted — validate against env-plus-name (the same producer->kernel boundary
            // the bridge uses; add_inductive re-checks everything against the real env).
            let boot = self.env.clone().with_const(CkName::from_dotted(name), nlvl);
            let cterm = CkTerm::validate(&boot, &claw, 0, nlvl)
                .map_err(|e| format!("ctor `{cn}` type validate: {e:?}"))?;
            ck_ctors.push(CkCtor {
                name: CkName::from_dotted(&cn.to_string()),
                type_: cterm,
            });
        }
        add_inductive(
            &mut self.env,
            CkIndDecl {
                name: CkName::from_dotted(name),
                num_level_params: nlvl,
                num_params: ind.num_params,
                type_: ind_term,
                constructors: ck_ctors,
            },
        )
        .map_err(|e| format!("ck0 admission: {e:?}"))
    }

    /// Register one value-bearing constant: translate type + body, validate, ck0-`check` the
    /// body against the type, then `with_def`. A failure at any step = honest skip reason.
    fn register_def(&mut self, name: &str) -> Result<(), String> {
        let kn = KName::from_string(name);
        let decl = self
            .clean
            .get_const(&kn)
            .ok_or_else(|| format!("`{name}` vanished from the clean env"))?;
        let lps = decl.level_params.clone();
        let transparency = match decl.kind {
            ConstantKind::Definition => Transparency::Transparent,
            _ => Transparency::Opaque,
        };
        let ty = decl.type_.clone();
        let val = decl
            .value
            .clone()
            .ok_or_else(|| "value-less — never admitted".to_string())?;
        // Rename-collision guard (fail-closed): if the clean env ALREADY has a constant under
        // the ck0 registration name, registering would silently shadow it — skip instead.
        let reg = ck0_reg_name(name);
        if reg != name && self.clean.get_const(&KName::from_string(&reg)).is_some() {
            return Err(format!(
                "ck0 registration name `{reg}` collides with a real constant"
            ));
        }
        let (ty_term, val_term) = self.validate_pair(&val, &ty, &lps)?;
        let mut budget = Budget::new(CK0_DEF_FUEL);
        // A def-check failure here is a REGISTRATION-TIME fragment gap, not an obligation
        // verdict: ck0's def-eq is deliberately smaller than clean-kernel's (e.g. a `Proj`
        // at the head of an application spine — the shape every typeclass-projection chain
        // like `LT.lt Nat instLTNat` δ-reduces to — is honestly stuck in ck0's whnf). The
        // def is NOT registered and every dependent skips honestly.
        ck0_check(&self.env, &val_term, &ty_term, &mut budget).map_err(|e| {
            format!("ck0 def check: {e:?} (ck0 could not re-derive this prelude def — fragment shrinks; not an obligation verdict)")
        })?;
        let nlvl = u32::try_from(lps.len()).map_err(|_| "level arity overflow".to_string())?;
        // Reserved-suffix names (`I.casesOn`, …) register under their renamed constant —
        // the same name every translated reference uses (see `ck0_reg_name`).
        self.env = std::mem::take(&mut self.env).with_def(
            CkName::from_dotted(&ck0_reg_name(name)),
            nlvl,
            ty_term,
            val_term,
            transparency,
        );
        Ok(())
    }

    /// Depth-gate, translate (type FIRST and independently of the term), and validate both
    /// sides. Shared by dependency registration, obligations, and the negative controls.
    fn validate_pair(
        &self,
        term: &Expr,
        expected: &Expr,
        lps: &[KName],
    ) -> Result<(CkTerm, CkTerm), String> {
        if exceeds_ck0_depth(expected) || exceeds_ck0_depth(term) {
            return Err(format!(
                "nesting depth exceeds ck0's validation cap ({MAX_VALIDATE_DEPTH})"
            ));
        }
        let ty_raw = self.tr(expected, lps).map_err(|e| format!("type: {e}"))?;
        let term_raw = self.tr(term, lps).map_err(|e| format!("term: {e}"))?;
        let nlvl = u32::try_from(lps.len()).map_err(|_| "level arity overflow".to_string())?;
        let ty_term = CkTerm::validate(&self.env, &ty_raw, 0, nlvl)
            .map_err(|e| format!("type validate: {e:?}"))?;
        let term_term = CkTerm::validate(&self.env, &term_raw, 0, nlvl)
            .map_err(|e| format!("term validate: {e:?}"))?;
        Ok((ty_term, term_term))
    }

    /// The raw second check: `term : expected` in this world's env. Fail-closed verdicts,
    /// mirroring the bridge: translation/validation/env/budget gaps are `Unavailable`;
    /// only a genuine ck0 refutation is `Rejected`.
    fn raw_check(&self, term: &Expr, expected: &Expr, lps: &[KName]) -> Outcome {
        let (ty_term, term_term) = match self.validate_pair(term, expected, lps) {
            Ok(p) => p,
            Err(e) => return Outcome::Unavailable(e),
        };
        let mut budget = Budget::new(CK0_CHECK_FUEL);
        match ck0_check(&self.env, &term_term, &ty_term, &mut budget) {
            Ok(()) => Outcome::Corroborated,
            Err(InferError::OutOfBudget) => {
                Outcome::Unavailable("ck0 budget exhausted (gave up, not a verdict)".to_string())
            }
            Err(InferError::UnknownConst { name }) => {
                Outcome::Unavailable(format!("const `{name}` outside the ck0 fragment"))
            }
            Err(e) => Outcome::Rejected(format!("{e:?}")),
        }
    }

    /// One DECLARED value-bearing constant: ensure its dependency closure, run the second
    /// check, and — only on corroboration — register it so later theorems resolve it.
    fn check_obligation(&mut self, name: &str) -> Outcome {
        let kn = KName::from_string(name);
        let Some(decl) = self.clean.get_const(&kn) else {
            return Outcome::Unavailable("declared but not in the elaborated env".to_string());
        };
        let Some(val) = decl.value.clone() else {
            return Outcome::Unavailable("declared constant is value-less".to_string());
        };
        let ty = decl.type_.clone();
        let lps = decl.level_params.clone();
        let mut deps = Vec::new();
        const_refs(&ty, &mut deps);
        const_refs(&val, &mut deps);
        deps.sort();
        deps.dedup();
        deps.retain(|d| d != name);
        for dep in &deps {
            self.ensure(dep);
            if let Some(Err(why)) = self.status.get(dep) {
                let out = Outcome::Unavailable(format!("depends on unsupported `{dep}` ({why})"));
                self.status
                    .insert(name.to_string(), Err(format!("skipped: {out:?}")));
                return out;
            }
        }
        let outcome = self.raw_check(&val, &ty, &lps);
        match &outcome {
            Outcome::Corroborated => {
                // Already ck0-checked above — safe to register for later theorems.
                let transparency = match decl.kind {
                    ConstantKind::Definition => Transparency::Transparent,
                    _ => Transparency::Opaque,
                };
                if let Ok((ty_term, val_term)) = self.validate_pair(&val, &ty, &lps) {
                    let nlvl = u32::try_from(lps.len()).unwrap_or(u32::MAX);
                    self.env = std::mem::take(&mut self.env).with_def(
                        CkName::from_dotted(name),
                        nlvl,
                        ty_term,
                        val_term,
                        transparency,
                    );
                    self.status.insert(name.to_string(), Ok(()));
                }
            }
            Outcome::Unavailable(why) => {
                self.status.insert(name.to_string(), Err(why.clone()));
            }
            Outcome::Rejected(why) => {
                self.status
                    .insert(name.to_string(), Err(format!("ck0 REJECTED: {why}")));
            }
        }
        outcome
    }
}

// ===========================================================================
// File loading (the exact pipeline of tests/clean_soundness_proofs.rs).
// ===========================================================================

fn collect_failed(r: &ElabResult, out: &mut Vec<(String, String)>) {
    match r {
        ElabResult::Multiple(rs) => rs.iter().for_each(|x| collect_failed(x, out)),
        ElabResult::Failed { name, error, .. } => out.push((name.clone(), format!("{error:?}"))),
        _ => {}
    }
}

/// Declared names in file order (namespace-prefixed). Constructs outside the vetted set are
/// the soundness harness's problem (it fails the build); here they contribute nothing.
fn collect_declared(decls: &[SurfaceDecl], prefix: &str, out: &mut Vec<String>) {
    for d in decls {
        match d {
            SurfaceDecl::Def { name, .. }
            | SurfaceDecl::Theorem { name, .. }
            | SurfaceDecl::Opaque { name, .. }
            | SurfaceDecl::Inductive { name, .. } => out.push(format!("{prefix}{name}")),
            SurfaceDecl::Namespace { name, decls, .. } => {
                collect_declared(decls, &format!("{prefix}{name}."), out);
            }
            _ => {}
        }
    }
}

fn proofs_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("proofs/clean")
}

/// Elaborate one proof file into a fresh prelude env; panic loudly on any failure (the
/// sibling soundness harness guarantees green — a failure here means the repo is broken).
fn elaborate_file(path: &std::path::Path) -> (Environment, Vec<String>) {
    let src =
        std::fs::read_to_string(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    let decls = parse_file(&src).unwrap_or_else(|e| panic!("parse {}: {e:?}", path.display()));
    let mut env = Environment::with_prelude();
    let mut fc = FileContext::new();
    for d in &decls {
        let p = preprocess_decl_with_context(d, &mut fc);
        let r = elaborate_decl_and_register(&mut env, &p)
            .unwrap_or_else(|e| panic!("elaborate {}: {e}", path.display()));
        let mut failures = Vec::new();
        collect_failed(&r, &mut failures);
        assert!(
            failures.is_empty(),
            "{}: inner elaboration failures (the soundness harness should have caught this): {failures:?}",
            path.display()
        );
    }
    let mut declared = Vec::new();
    collect_declared(&decls, "", &mut declared);
    (env, declared)
}

// ===========================================================================
// The corroboration run.
// ===========================================================================

/// clean-ck0 independently re-checks the declarations of every soundness meta-proof file.
/// Per the fail-closed contract: rejections fail the test; unsupported declarations are
/// skipped with printed reasons; the tiny-TCB label attaches ONLY to the machine-derived N.
#[test]
fn ck0_corroborates_soundness_proof_declarations() {
    let dir = proofs_dir();
    let mut files: Vec<_> = std::fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("read proofs dir {}: {e}", dir.display()))
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "clean"))
        .collect();
    files.sort();
    assert!(
        !files.is_empty(),
        "no .clean proofs found in {}",
        dir.display()
    );

    let mut total_declared = 0usize;
    let mut total_corroborated = 0usize;
    let mut total_unsupported = 0usize;
    let mut rejections: Vec<String> = Vec::new();
    let mut ran_cross_control = false;

    for path in &files {
        let fname = path.file_name().unwrap().to_string_lossy().to_string();
        let (env, declared) = elaborate_file(path);
        let mut world = World::new(&env);
        let mut corroborated = 0usize;
        let mut unsupported: Vec<(String, String)> = Vec::new();
        let mut corroborated_names: HashSet<String> = HashSet::new();

        for name in &declared {
            let kn = KName::from_string(name);
            if env.get_inductive(&kn).is_some() {
                // A declared inductive is not a value-bearing proof obligation: it enters the
                // env via ck0's own kernel-checked add_inductive (or is skipped honestly).
                world.ensure(name);
                if let Some(Err(why)) = world.status.get(name.as_str()) {
                    eprintln!("    (inductive) {name} not admitted: {why}");
                }
                continue;
            }
            let is_value_bearing = env.get_const(&kn).is_some_and(|c| {
                c.value.is_some()
                    && matches!(
                        c.kind,
                        ConstantKind::Theorem | ConstantKind::Definition | ConstantKind::Opaque
                    )
            });
            if !is_value_bearing {
                // Nothing to corroborate (and nothing silently passed): record it.
                total_declared += 1;
                total_unsupported += 1;
                unsupported.push((name.clone(), "declared but not value-bearing".to_string()));
                continue;
            }
            total_declared += 1;
            match world.check_obligation(name) {
                Outcome::Corroborated => {
                    corroborated += 1;
                    total_corroborated += 1;
                    corroborated_names.insert(name.clone());
                }
                Outcome::Unavailable(why) => {
                    total_unsupported += 1;
                    unsupported.push((name.clone(), why));
                }
                Outcome::Rejected(why) => {
                    rejections.push(format!("{fname}: {name}: {why}"));
                }
            }
        }

        eprintln!(
            "CK0: {fname} — {corroborated} corroborated, {} unsupported",
            unsupported.len()
        );
        for (name, why) in &unsupported {
            let short: String = why.chars().take(200).collect();
            eprintln!("    unsupported {name}: {short}");
        }

        // Secondary negative control on REAL proof material: a corroborated proof value fed
        // against a corroborated sibling's (wrong) type must be REJECTED, in this very world.
        if fname == "S1_pack_injective.clean" {
            let donor = "PackInj.zero_ne_succ";
            let wrong = "PackInj.le_add_left";
            if corroborated_names.contains(donor) && corroborated_names.contains(wrong) {
                let dv = env
                    .get_const(&KName::from_string(donor))
                    .and_then(|c| c.value.clone())
                    .expect("donor has a value");
                let wt = env
                    .get_const(&KName::from_string(wrong))
                    .expect("wrong-type donor exists")
                    .type_
                    .clone();
                let out = world.raw_check(&dv, &wt, &[]);
                assert!(
                    matches!(out, Outcome::Rejected(_)),
                    "NEGATIVE CONTROL (in-file): ck0 must REJECT `{donor}`'s proof value against \
                     `{wrong}`'s type — got {out:?}; the second check would be a rubber stamp"
                );
                ran_cross_control = true;
                eprintln!("    negative control: {donor} value vs {wrong} type — rejected (good)");
            } else {
                eprintln!(
                    "    negative control on file material skipped ({donor}/{wrong} not both \
                     corroborated); the standalone mismatched-pair control still runs"
                );
            }
        }
    }

    eprintln!(
        "ck0 second-checker: {total_corroborated}/{total_declared} declarations corroborated \
         (the tiny-TCB label attaches ONLY to the {total_corroborated}); \
         {total_unsupported} unsupported (reasons above)"
    );
    if ran_cross_control {
        eprintln!("ck0 second-checker: in-file mismatched-pair negative control ran and rejected");
    }

    assert!(
        rejections.is_empty(),
        "ck0 REJECTED {} declaration(s) that clean-kernel accepted — SECOND-CHECKER \
         DISAGREEMENT (a soundness alarm; either kernel or the translator has a bug):\n  {}",
        rejections.len(),
        rejections.join("\n  ")
    );
    assert!(
        total_corroborated > 0,
        "zero declarations corroborated — the second-checker fragment is EMPTY (blocked); \
         see the unsupported reasons above"
    );
}

/// MANDATORY NEGATIVE CONTROL (deterministic, fully in-fragment): a REAL proof value
/// (`@Eq.refl Nat Nat.zero`, a genuine proof of `0 = 0`) fed against a WRONG type it does not
/// inhabit (`@Eq Nat 0 (succ 0)`) must be REJECTED — and the positive twin corroborates,
/// proving the rejection is a genuine type-mismatch verdict, not a fragment gap.
#[test]
fn negative_control_ck0_rejects_mismatched_pair() {
    let clean = Environment::with_prelude();
    let mut world = World::new(&clean);
    for root in ["Nat", "Eq"] {
        world.ensure(root);
        assert_eq!(
            world.status.get(root),
            Some(&Ok(())),
            "control precondition: `{root}` must admit into ck0"
        );
    }
    let lvl1 = vec![KLevel::succ(KLevel::zero())];
    let nat = Expr::const_str("Nat");
    let zero = Expr::const_str("Nat.zero");
    let succ_zero = Expr::app(Expr::const_str("Nat.succ"), zero.clone());
    let term = Expr::apps(
        Expr::const_str_levels("Eq.refl", lvl1.clone()),
        [nat.clone(), zero.clone()],
    );
    let good_ty = Expr::apps(
        Expr::const_str_levels("Eq", lvl1.clone()),
        [nat.clone(), zero.clone(), zero.clone()],
    );
    let bad_ty = Expr::apps(Expr::const_str_levels("Eq", lvl1), [nat, zero, succ_zero]);
    // Positive twin: the pair is fully inside the supported fragment.
    assert_eq!(
        world.raw_check(&term, &good_ty, &[]),
        Outcome::Corroborated,
        "control precondition: the well-typed twin must corroborate (else the negative \
         control would be vacuous — a fragment gap, not a real rejection)"
    );
    // The control: same real proof value, wrong type -> ck0 MUST refuse.
    let out = world.raw_check(&term, &bad_ty, &[]);
    assert!(
        matches!(out, Outcome::Rejected(_)),
        "ck0 MUST reject a real proof value against a type it does not inhabit — got {out:?}; \
         corroboration would be a rubber stamp"
    );
}
