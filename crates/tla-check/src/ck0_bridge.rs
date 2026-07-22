// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Phase 1 of `docs/kernel-checked-tla-plan.md`: **clean-ck0 as the mandatory independent
//! second checker** for the Nat/Bool/Int obligation fragment.
//!
//! `clean-ck0` is the genuinely tiny kernel (~8K LOC of `#![forbid(unsafe_code)]` source,
//! deps only num-bigint/num-traits/thiserror) that shares ZERO production code with
//! `clean-kernel`. This module
//!
//! 1. translates the already-clean-kernel-checked proof term AND its expected type —
//!    independently of each other — from `clean_kernel::Expr` into ck0 `RawExpr`
//!    (fail-closed on anything outside the fragment),
//! 2. validates both through ck0's single chokepoint (`Term::validate`), and
//! 3. asks ck0 to decide `check(term, expected)` under a deterministic step budget.
//!
//! ## The trust story (read this before widening anything)
//!
//! The ck0 environment is built from **nothing** (`MinimalEnv::new()` is empty — no
//! built-ins, no axioms, no unchecked admission path):
//!
//! * inductives (`Bool`, `Nat`, `Eq`, `And`, `Or`, `False`, `True`, `Int`, and the
//!   indexed family `Int.NonNeg`) are admitted via `clean_ck0::add_inductive`, which
//!   kernel-checks positivity/universes and DERIVES its own recursors — ty ships no
//!   recursor axioms;
//! * every definition is translated from the Clean prelude artifact and
//!   **ck0-`check`ed against its translated type before registration** — a def that
//!   fails ck0's own check is NOT registered (the fragment shrinks, honestly). The
//!   Nat/Bool core (`Bool.and`, `Nat.beq`, …) is a hand-curated list; the Int fragment
//!   is ingested as the TRANSITIVE DEPENDENCY CLOSURE of [`INT_FRAGMENT_ROOTS`]
//!   (`Int.add`, `Int.le`, `Int.le_trans`, …), registered in dependency postorder.
//!   Prelude `Theorem`s register [`Transparency::Opaque`]: opaque WITH a ck0-checked
//!   body is NOT axiom-shaped (the body was checked against the type at registration;
//!   opacity only bars δ-unfolding, and ck0's Prop proof irrelevance covers their
//!   proof-position uses);
//! * NO body-less constant is ever registered in the final env (`with_const_typed` — the
//!   axiom-shaped builder — is never called), so a ck0-`Corroborated` verdict is
//!   **axiom-free by construction** relative to ck0's checker + its native Nat/Bool
//!   literal reducer (`try_native_nat`, part of the audited ~8K LOC).
//!
//! ## Numeral encodings (why there are two)
//!
//! ck0 deliberately has NO `Lit`↔constructor bridging: its native reducer computes
//! `Nat.{add,sub,beq,…}` on `RawLit::Nat` literals, but its ι-rules fire only on
//! CONSTRUCTOR-form majors — `Nat.rec … (Lit 3)` is honestly stuck, and `Lit 1` is
//! never def-eq to `Nat.succ Nat.zero`. The prelude's Int defs (`Int.subNatNat`,
//! `Int.neg`, …) pattern-match their `Nat` arguments via `Nat.rec`, so an Int-fragment
//! obligation whose numerals were translated as literals would get STUCK mid-reduction
//! and be spuriously refuted. The translator therefore picks ONE encoding per
//! obligation, fail-closed:
//!
//! * no `Int.*` constant mentioned → [`NumeralEnc::Native`]: `RawExpr::Lit`, byte-for-byte
//!   the historical Nat/Bool behavior (native literal arithmetic, any magnitude);
//! * any `Int.*` constant mentioned → [`NumeralEnc::Ctor`]: `Nat.succ^n Nat.zero` chains,
//!   capped at [`CTOR_NUMERAL_MAX`] (over-cap → `Unavailable`, never a wrong verdict), so
//!   every reduction step goes through ck0's own derived ι-rules. The closure-ingested
//!   Int defs are registered under the SAME encoding (their prelude bodies contain no
//!   nat literals — all-constructor — so the two encodings cannot meet mid-reduction).
//!
//! The residual trust in this module itself is the [`tr_expr`] translator (untrusted
//! glue, same discipline as clean's `ck0_ingest_bridge`): it is structure-preserving and
//! total-or-explicit-error, the expected TYPE is translated independently of the proof
//! term, and a mistranslation surfaces as a ck0 rejection (fail-closed), never as a
//! silent accept of something ck0 did not see.
//!
//! ## Verdict semantics (fail-closed, never a false tiny-TCB label)
//!
//! * [`Ck0Corroboration::Corroborated`] — ck0 independently re-checked `term : expected`.
//!   Only this outcome may carry the "tiny auditable kernel" label.
//! * [`Ck0Corroboration::Unavailable`] — the obligation is outside ck0's fragment
//!   (an unregistered constant, an over-[`CTOR_NUMERAL_MAX`] Int-fragment numeral, the
//!   1024 validation-depth cap) or ck0 exhausted its pinned budget. NOT a
//!   disagreement: the verdict keeps the clean-kernel-tier trust base and says so.
//! * [`Ck0Corroboration::Rejected`] — ck0 REFUTED a term clean-kernel accepted. Checker
//!   disagreement is treated as evidence of a bug somewhere (either kernel, or this
//!   translator): the caller must fail the certification closed.

use std::sync::OnceLock;

use clean_ck0::{
    add_inductive, Budget, Constructor as CkCtor, InductiveDecl as CkIndDecl, MinimalEnv,
    Name as CkName, RawExpr, RawLevel, RawLit, Term as CkTerm, Transparency,
};
use clean_kernel::{
    BigNat as KBigNat, BinderInfo, ConstantKind, Environment, Expr, ExprKind, Level as KLevel,
    Literal, Name as KName,
};

/// The pinned deterministic step budget for one ck0 `check` decision. Chosen large
/// enough for the product-domain completeness obligations the P2 leg emits (each
/// conjunct costs a bounded number of native-literal reductions); exhaustion collapses
/// to [`Ck0Corroboration::Unavailable`], never to an accept.
const CK0_CHECK_FUEL: u64 = 64_000_000;

/// Budget for ck0-checking one prelude declaration at env build time. Raised from
/// 4M when the Int fragment landed: the closure-ingested `Theorem` bodies (e.g. the
/// `Int.add_comm`/`Int.add_assoc` case lemmas backing `Int.le_trans`) are large
/// recursor-heavy proof terms, and each is ck0-checked exactly once per process at
/// env build. Exhaustion SKIPS the def (ledger-recorded) — never an unchecked
/// registration.
const CK0_DEF_FUEL: u64 = 32_000_000;

/// Cap on a Nat literal translated as a `Nat.succ^n Nat.zero` constructor chain
/// ([`NumeralEnc::Ctor`], the Int-fragment encoding). Chains add `n` to the term's
/// nesting depth, and ck0's validation chokepoint pins `MAX_VALIDATE_DEPTH = 1024`;
/// 512 leaves headroom for the obligation structure AROUND the numeral. Over-cap
/// numerals fail translation → `Unavailable` (fail-closed, never a wrong verdict).
const CTOR_NUMERAL_MAX: u64 = 512;

/// The outcome of the independent second check. See the module docs for semantics.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Ck0Corroboration {
    /// ck0 independently re-checked `term : expected` from an axiom-free env.
    Corroborated,
    /// Outside ck0's Nat/Bool fragment, or ck0 gave up (budget/depth). Not a verdict.
    Unavailable(String),
    /// ck0 refuted a term clean-kernel accepted — checker disagreement. Fail closed.
    Rejected(String),
}

/// Tally of second-checker outcomes across the `kernel_accepts` calls of one
/// certification/verification run (see [`begin_tally`]/[`take_tally`]).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Ck0Tally {
    /// Obligations ck0 independently re-checked (the tiny-TCB fragment).
    pub corroborated: usize,
    /// Obligations outside ck0's fragment (clean-kernel tier only).
    pub unavailable: usize,
    /// Checker disagreements (each one also failed the certification closed).
    pub rejected: usize,
}

std::thread_local! {
    static CK0_TALLY: std::cell::RefCell<Option<Ck0Tally>> = const { std::cell::RefCell::new(None) };
}

/// Start tallying second-checker outcomes on this thread (idempotent: resets any
/// running tally). Pair with [`take_tally`].
pub fn begin_tally() {
    CK0_TALLY.with(|t| *t.borrow_mut() = Some(Ck0Tally::default()));
}

/// Stop tallying and return the counts since [`begin_tally`], if one was running.
pub fn take_tally() -> Option<Ck0Tally> {
    CK0_TALLY.with(|t| t.borrow_mut().take())
}

/// Whether a tally is running on THIS thread (peek, no reset). The parallel chunk
/// driver ([`crate::cleancic`]'s per-source-state completeness legs) uses this to
/// decide whether worker threads must capture per-worker tallies for the merge.
pub(crate) fn tally_active() -> bool {
    CK0_TALLY.with(|t| t.borrow().is_some())
}

/// Fold `extra` counts into THIS thread's active tally (no-op when none is running).
/// Order-independent sums — the parallel chunk driver merges its workers' per-worker
/// tallies through this, so a run's printed per-obligation counts are IDENTICAL to the
/// sequential loop's (every chunk still tallied exactly once).
pub(crate) fn merge_into_active_tally(extra: Ck0Tally) {
    CK0_TALLY.with(|t| {
        if let Some(tally) = t.borrow_mut().as_mut() {
            tally.corroborated += extra.corroborated;
            tally.unavailable += extra.unavailable;
            tally.rejected += extra.rejected;
        }
    });
}

fn tally(outcome: &Ck0Corroboration) {
    CK0_TALLY.with(|t| {
        if let Some(tally) = t.borrow_mut().as_mut() {
            match outcome {
                Ck0Corroboration::Corroborated => tally.corroborated += 1,
                Ck0Corroboration::Unavailable(_) => tally.unavailable += 1,
                Ck0Corroboration::Rejected(_) => tally.rejected += 1,
            }
        }
    });
}

// ===========================================================================
// The untrusted translator: clean-kernel Expr/Level/Name -> ck0 RawExpr/...
// Structure-preserving; fail-closed (BridgeError) outside the fragment.
// ===========================================================================

#[derive(Debug, Clone, PartialEq, Eq)]
enum BridgeError {
    /// A clean `ExprKind`/`Literal` variant outside the ck0 M0–M3 fragment.
    Unsupported(String),
    /// A universe `Param` name not in the declaration's level telescope.
    UnknownLevelParam(String),
    /// A recursor constant whose inductive is not in the admitted table, or whose
    /// level vector does not decompose as `[motive] ++ ind_levels`.
    Recursor(String),
}

impl std::fmt::Display for BridgeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BridgeError::Unsupported(s) => write!(f, "unsupported: {s}"),
            BridgeError::UnknownLevelParam(s) => write!(f, "unknown level param: {s}"),
            BridgeError::Recursor(s) => write!(f, "recursor lowering: {s}"),
        }
    }
}

/// Level-parameter arity of each inductive the env admits, for recursor lowering
/// (`I.rec.{u?, vs..}` → `Elim(I, u_or_Zero, vs)`). Kept in lockstep with
/// [`build_env`]'s admission list.
const ADMITTED_INDUCTIVE_LEVEL_ARITY: &[(&str, usize)] = &[
    ("Bool", 0),
    ("Nat", 0),
    ("Eq", 1),
    ("And", 0),
    ("Or", 0),
    ("False", 0),
    ("True", 0),
    ("Int", 0),
    ("Int.NonNeg", 0),
];

fn admitted_inductive_level_arity(ind: &str) -> Option<usize> {
    ADMITTED_INDUCTIVE_LEVEL_ARITY
        .iter()
        .find(|(n, _)| *n == ind)
        .map(|(_, a)| *a)
}

fn tr_level(lvl: &KLevel, lps: &[KName]) -> Result<RawLevel, BridgeError> {
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
            .ok_or_else(|| BridgeError::UnknownLevelParam(n.to_string())),
    }
}

fn tr_binfo(info: BinderInfo) -> clean_ck0::rawexpr::BinderInfo {
    use clean_ck0::rawexpr::BinderInfo as CkBinderInfo;
    match info {
        BinderInfo::Default => CkBinderInfo::Default,
        BinderInfo::Implicit => CkBinderInfo::Implicit,
        BinderInfo::StrictImplicit => CkBinderInfo::StrictImplicit,
        BinderInfo::InstImplicit => CkBinderInfo::InstImplicit,
    }
}

fn tr_name(n: &KName) -> CkName {
    CkName::from_dotted(&n.to_string())
}

/// Recursor-name suffixes ck0's chokepoint reserves (rejected in `Const` position).
/// `rec` is lowered to `RawExpr::Elim`; the others have no ck0 analog — fail closed.
fn recursor_suffix(last: &str) -> Option<&'static str> {
    match last {
        "rec" => Some("rec"),
        "recOn" | "casesOn" | "below" | "ibelow" | "brecOn" | "binductionOn" | "brecOnEq" => {
            Some("other")
        }
        _ => None,
    }
}

/// How Nat literals are encoded across the ck0 boundary. See the module docs
/// ("Numeral encodings"): ck0 has no `Lit`↔constructor bridging, so the encoding is
/// chosen ONCE per obligation ([`NumeralEnc::Ctor`] iff any `Int.*` constant is
/// mentioned) and used for the type, the term, and — for the closure-ingested Int
/// defs — the registered definitions, keeping the reduction world coherent.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NumeralEnc {
    /// `RawExpr::Lit` — ck0's native literal reducer does the arithmetic. The
    /// historical Nat/Bool-fragment encoding, unchanged.
    Native,
    /// `Nat.succ^n Nat.zero` constructor chains (≤ [`CTOR_NUMERAL_MAX`]), so the
    /// prelude's `Nat.rec`-matching Int defs can ι-reduce through the numeral.
    Ctor,
}

/// `Nat.succ^n Nat.zero` for a literal, built iteratively (no recursion in `n`).
/// Fail-closed above [`CTOR_NUMERAL_MAX`] — see the const's docs.
fn nat_ctor_chain(n: &KBigNat) -> Result<RawExpr, BridgeError> {
    let v = match n.to_u64() {
        Some(v) if v <= CTOR_NUMERAL_MAX => v,
        _ => {
            return Err(BridgeError::Unsupported(format!(
                "nat literal exceeds the Int-fragment ctor-numeral cap ({CTOR_NUMERAL_MAX})"
            )))
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

/// Convert a clean-kernel arbitrary-precision literal into ck0's (BigUint-backed)
/// representation by folding the little-endian u64 limbs. No fixed-width arithmetic
/// on the VALUE — everything goes through ck0's own `BigNat` ops.
fn tr_bignat(n: &KBigNat) -> clean_ck0::BigNat {
    let half = clean_ck0::BigNat::from_u64(1u64 << 32); // 2^32
    let base = half.mul(&half); // 2^64
    let mut acc = clean_ck0::BigNat::zero();
    for limb in n.limbs().iter().rev() {
        acc = acc.mul(&base).add(&clean_ck0::BigNat::from_u64(*limb));
    }
    acc
}

/// Translate a clean `Expr` into a ck0 `RawExpr` against level telescope `lps`,
/// encoding Nat literals per `enc`. Total over the fragment; fail-closed everywhere
/// else. Recursor constants (`I.rec.{u?, vs..}`) are lowered to
/// `RawExpr::Elim(I, u|Zero, vs)` — ck0 derives and checks the full recursor level
/// vector itself.
fn tr_expr(e: &Expr, lps: &[KName], enc: NumeralEnc) -> Result<RawExpr, BridgeError> {
    match e.kind() {
        ExprKind::BVar(i) => Ok(RawExpr::BVar(*i)),
        ExprKind::Sort(l) => Ok(RawExpr::Sort(tr_level(l, lps)?)),
        ExprKind::Const(name, levels) => {
            let dotted = name.to_string();
            if let Some((parent, last)) = dotted.rsplit_once('.') {
                match recursor_suffix(last) {
                    Some("rec") => {
                        let arity = admitted_inductive_level_arity(parent).ok_or_else(|| {
                            BridgeError::Recursor(format!("`{dotted}`: inductive not admitted"))
                        })?;
                        let lv: Vec<RawLevel> = levels
                            .iter()
                            .map(|l| tr_level(l, lps))
                            .collect::<Result<_, _>>()?;
                        // Lean/Clean recursor levels: `[motive] ++ ind_levels` when the
                        // eliminator is universe-polymorphic, or just `ind_levels` for a
                        // Prop-only eliminator (motive fixed at Prop = Zero).
                        let (motive, ind_levels) = if lv.len() == arity + 1 {
                            let mut it = lv.into_iter();
                            let m = it.next().unwrap_or(RawLevel::Zero);
                            (m, it.collect())
                        } else if lv.len() == arity {
                            (RawLevel::Zero, lv)
                        } else {
                            return Err(BridgeError::Recursor(format!(
                                "`{dotted}`: level vector arity {} does not fit inductive arity {arity}",
                                levels.len()
                            )));
                        };
                        return Ok(RawExpr::Elim(
                            CkName::from_dotted(parent),
                            motive,
                            ind_levels,
                        ));
                    }
                    Some(_) => {
                        return Err(BridgeError::Recursor(format!(
                            "`{dotted}` has no ck0 lowering"
                        )));
                    }
                    None => {}
                }
            }
            let lv: Vec<RawLevel> = levels
                .iter()
                .map(|l| tr_level(l, lps))
                .collect::<Result<_, _>>()?;
            Ok(RawExpr::Const(tr_name(name), lv))
        }
        ExprKind::App(f, a) => Ok(RawExpr::App(
            Box::new(tr_expr(f, lps, enc)?),
            Box::new(tr_expr(a, lps, enc)?),
        )),
        ExprKind::Lam(bd, ty, body) => Ok(RawExpr::Lam(
            tr_binfo(bd.info),
            Box::new(tr_expr(ty, lps, enc)?),
            Box::new(tr_expr(body, lps, enc)?),
        )),
        ExprKind::Pi(bd, ty, body) => Ok(RawExpr::Pi(
            tr_binfo(bd.info),
            Box::new(tr_expr(ty, lps, enc)?),
            Box::new(tr_expr(body, lps, enc)?),
        )),
        ExprKind::Let(_name, ty, val, body, _nondep) => Ok(RawExpr::Let(
            Box::new(tr_expr(ty, lps, enc)?),
            Box::new(tr_expr(val, lps, enc)?),
            Box::new(tr_expr(body, lps, enc)?),
        )),
        ExprKind::Lit(Literal::Nat(n)) => match enc {
            NumeralEnc::Native => Ok(RawExpr::Lit(RawLit::Nat(tr_bignat(n)))),
            NumeralEnc::Ctor => nat_ctor_chain(n),
        },
        ExprKind::Proj(name, idx, inner) => Ok(RawExpr::Proj(
            tr_name(name),
            *idx,
            Box::new(tr_expr(inner, lps, enc)?),
        )),
        other => Err(BridgeError::Unsupported(format!("{other:?}"))),
    }
}

/// Does the obligation mention any `Int`-rooted constant (`Int`, `Int.le`,
/// `Int.NonNeg.mk`, …)? Decides the numeral encoding — see the module docs.
/// Iterative walk; never recurses on untrusted structure.
fn mentions_int(e: &Expr) -> bool {
    mentions_pred(e, &|dotted| dotted == "Int" || dotted.starts_with("Int."))
}

/// Whether the obligation references the `Nat` recursor. Such terms (e.g. the Phase-A
/// successor-bound lemmas' `Nat.rec` casework) must reach ck0 in CONSTRUCTOR-chain numeral
/// form: ck0 has no Lit↔constructor bridging, so a literal-encoded numeral meeting a
/// `Nat.rec` ι-site is undecidable-by-construction for ck0 (a stuck def-eq would surface as
/// a spurious "rejection"). Ctor encoding makes the whole obligation coherent and decidable.
fn mentions_nat_rec(e: &Expr) -> bool {
    mentions_pred(e, &|dotted| dotted == "Nat.rec")
}

fn mentions_pred(e: &Expr, pred: &dyn Fn(&str) -> bool) -> bool {
    let mut stack: Vec<&Expr> = vec![e];
    while let Some(node) = stack.pop() {
        match node.kind() {
            ExprKind::Const(name, _) => {
                let dotted = name.to_string();
                if pred(&dotted) {
                    return true;
                }
            }
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
            ExprKind::Proj(_, _, inner) => stack.push(inner),
            _ => {}
        }
    }
    false
}

// ===========================================================================
// The ck0 environment: built from empty, entirely from kernel-checked admissions
// and ck0-checked translated definitions. Cached once per process.
// ===========================================================================

/// How each name entered the ck0 env — the auditable ledger of the second
/// checker's entire trust surface (MinimalEnv itself has no enumeration API).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LedgerEntry {
    /// Admitted via `clean_ck0::add_inductive` (positivity/universe/recursor
    /// derivation all kernel-checked by ck0 itself).
    Inductive,
    /// Translated from the Clean prelude and ck0-`check`ed against its translated
    /// type BEFORE registration as a transparent definition.
    CheckedDef,
    /// Present in the candidate list but not registered (translation or ck0 check
    /// failed) — obligations mentioning it stay `Unavailable`, never unsound.
    Skipped(String),
}

/// The built second-checker environment plus its trust ledger.
pub struct Ck0Env {
    env: MinimalEnv,
    /// (dotted name, how it entered). No entry is ever a body-less typed constant.
    pub ledger: Vec<(String, LedgerEntry)>,
}

static CK0_ENV: OnceLock<Option<Ck0Env>> = OnceLock::new();

/// The cached env; `None` if construction failed wholesale (every corroboration is
/// then `Unavailable` — fail-closed, certification proceeds at clean-kernel tier).
fn cached_env() -> Option<&'static Ck0Env> {
    CK0_ENV.get_or_init(build_env).as_ref()
}

/// Test/report hook: the ledger of the process-wide second-checker env.
pub fn env_ledger() -> Option<&'static [(String, LedgerEntry)]> {
    cached_env().map(|e| e.ledger.as_slice())
}

fn clean_decl(env: &Environment, name: &str) -> Option<(Expr, Option<Expr>)> {
    let d = env.get_const(&KName::from_string(name))?;
    Some((d.type_.clone(), d.value.clone()))
}

/// Admit one inductive: translate the type-former and constructor types from the
/// Clean prelude, validate them against a bootstrap env that knows the names
/// admitted SO FAR plus this declaration's own names (the producer→kernel
/// boundary — `Int.NonNeg`'s constructor type mentions `Nat`/`Int.ofNat`, so the
/// boot env must resolve earlier admissions), and let ck0 kernel-check the
/// admission itself against the real accumulating env.
fn admit_inductive(
    env: &mut MinimalEnv,
    clean: &Environment,
    ledger: &mut Vec<(String, LedgerEntry)>,
    name: &str,
    num_level_params: u32,
    num_params: u32,
    ctors: &[&str],
) -> Result<(), String> {
    let lps: Vec<KName> = clean
        .get_const(&KName::from_string(name))
        .map(|d| d.level_params.clone())
        .unwrap_or_default();
    let (ind_ty, _) = clean_decl(clean, name).ok_or_else(|| format!("`{name}` absent"))?;
    let mut boot = env
        .clone()
        .with_const(CkName::from_dotted(name), num_level_params);
    for c in ctors {
        boot = boot.with_const(CkName::from_dotted(c), num_level_params);
    }
    // Inductive/ctor TYPES carry no nat literals; Native is the identity encoding here.
    let ind_raw =
        tr_expr(&ind_ty, &lps, NumeralEnc::Native).map_err(|e| format!("`{name}` type: {e}"))?;
    let ind_term = CkTerm::validate(&boot, &ind_raw, 0, num_level_params)
        .map_err(|e| format!("`{name}` type validate: {e:?}"))?;
    let mut ck_ctors = Vec::new();
    for c in ctors {
        let (cty, _) = clean_decl(clean, c).ok_or_else(|| format!("`{c}` absent"))?;
        let claw =
            tr_expr(&cty, &lps, NumeralEnc::Native).map_err(|e| format!("`{c}` type: {e}"))?;
        let cterm = CkTerm::validate(&boot, &claw, 0, num_level_params)
            .map_err(|e| format!("`{c}` type validate: {e:?}"))?;
        ck_ctors.push(CkCtor {
            name: CkName::from_dotted(c),
            type_: cterm,
        });
    }
    add_inductive(
        env,
        CkIndDecl {
            name: CkName::from_dotted(name),
            num_level_params,
            num_params,
            type_: ind_term,
            constructors: ck_ctors,
        },
    )
    .map_err(|e| format!("`{name}` admission: {e:?}"))?;
    ledger.push((name.to_string(), LedgerEntry::Inductive));
    for c in ctors {
        ledger.push(((*c).to_string(), LedgerEntry::Inductive));
    }
    Ok(())
}

/// Register one definition: translate type + body from the Clean prelude (numerals
/// per `enc`), validate, ck0-`check` the body against the type, then register.
/// Prelude `Definition`s register Transparent (they must δ-unfold for reduction);
/// `Theorem`s register Opaque — opaque WITH a ck0-checked body is NOT axiom-shaped
/// (the zero-body-less-consts invariant stands), and ck0's Prop proof irrelevance
/// covers their proof-position uses without unfolding. A failure at any step SKIPS
/// the def (recorded) — never an unchecked registration.
fn register_def(
    env: &mut MinimalEnv,
    clean: &Environment,
    ledger: &mut Vec<(String, LedgerEntry)>,
    name: &str,
    enc: NumeralEnc,
) {
    let decl = clean.get_const(&KName::from_string(name));
    let lps: Vec<KName> = decl.map(|d| d.level_params.clone()).unwrap_or_default();
    let transparency = match decl.map(|d| &d.kind) {
        Some(ConstantKind::Definition) => Transparency::Transparent,
        _ => Transparency::Opaque,
    };
    let num_lvl = u32::try_from(lps.len()).unwrap_or(u32::MAX);
    let skip = |ledger: &mut Vec<(String, LedgerEntry)>, why: String| {
        ledger.push((name.to_string(), LedgerEntry::Skipped(why)));
    };
    let Some((ty, Some(val))) = clean_decl(clean, name) else {
        return skip(ledger, "absent or value-less in the Clean prelude".into());
    };
    let (ty_raw, val_raw) = match (tr_expr(&ty, &lps, enc), tr_expr(&val, &lps, enc)) {
        (Ok(t), Ok(v)) => (t, v),
        (Err(e), _) | (_, Err(e)) => return skip(ledger, format!("translate: {e}")),
    };
    let (ty_term, val_term) = match (
        CkTerm::validate(env, &ty_raw, 0, num_lvl),
        CkTerm::validate(env, &val_raw, 0, num_lvl),
    ) {
        (Ok(t), Ok(v)) => (t, v),
        (Err(e), _) | (_, Err(e)) => return skip(ledger, format!("validate: {e:?}")),
    };
    let mut budget = Budget::new(CK0_DEF_FUEL);
    if let Err(e) = clean_ck0::check(env, &val_term, &ty_term, &mut budget) {
        return skip(ledger, format!("ck0 def check: {e:?}"));
    }
    *env = std::mem::take(env).with_def(
        CkName::from_dotted(name),
        num_lvl,
        ty_term,
        val_term,
        transparency,
    );
    ledger.push((name.to_string(), LedgerEntry::CheckedDef));
}

/// Roots of the ck0 Int fragment. The env ingests the TRANSITIVE dependency
/// closure of these names from the Clean prelude (definitions and theorems with
/// values; inductive formers/ctors are admitted separately and recursors are
/// Elim-lowered on the fly), so the widened P3 legs — `Int.le` consecution via
/// `le_trans`/`add_le_add_left`/`add_zero`, ground `Int.le` initiation via
/// `NonNeg.mk` + closed reduction, `NonNeg` closure via `NonNeg.add` — get
/// second-checker coverage.
const INT_FRAGMENT_ROOTS: &[&str] = &[
    "Int.add",
    "Int.sub",
    "Int.neg",
    "Int.le",
    "Int.lt",
    "Int.zero",
    "Int.subNatNat",
    "Int.NonNeg.add",
    "Int.le_trans",
    "Int.le_self_add_one",
    "Int.add_le_add_left",
    "Int.add_le_add_right",
    "Int.add_zero",
    "Int.le_refl",
    "Eq.subst",
    "Eq.symm",
    "Eq.trans",
];

/// Collect every `Const` name referenced by `e` (dotted form). Iterative walk —
/// prelude proof bodies are large; never recurse on their structure.
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
            ExprKind::Proj(_, _, inner) => stack.push(inner),
            _ => {}
        }
    }
}

/// Is `name` a recursor-family constant (`I.rec`, `I.casesOn`, …)? Those are never
/// registered as defs: `.rec` is Elim-lowered inside [`tr_expr`]; the rest have no
/// ck0 analog and fail translation of any def that mentions them (fail-closed).
fn is_recursor_family(name: &str) -> bool {
    name.rsplit_once('.')
        .is_some_and(|(_, last)| recursor_suffix(last).is_some())
}

/// The registration order for the Int fragment: DFS postorder over the transitive
/// `Const` closure of [`INT_FRAGMENT_ROOTS`] in the Clean prelude (dependencies
/// before dependents, so each def validates against an env that already holds its
/// deps). `skip` prunes names that never register: already-in-ledger (admitted
/// inductives/ctors, the curated Nat/Bool defs) and recursor-family names. A
/// residual cycle (impossible in a well-founded prelude) degrades to a Skipped
/// entry at registration — fail-soft, never wrong.
fn int_closure_postorder(clean: &Environment, skip: &dyn Fn(&str) -> bool) -> Vec<String> {
    enum Visit {
        Enter(String),
        Exit(String),
    }
    let mut order = Vec::new();
    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
    // Reversed so the LIFO stack processes roots in declared order.
    let mut stack: Vec<Visit> = INT_FRAGMENT_ROOTS
        .iter()
        .rev()
        .map(|r| Visit::Enter((*r).to_string()))
        .collect();
    while let Some(visit) = stack.pop() {
        match visit {
            Visit::Enter(name) => {
                if seen.contains(&name) || skip(&name) {
                    continue;
                }
                seen.insert(name.clone());
                let mut deps = Vec::new();
                if let Some(d) = clean.get_const(&KName::from_string(&name)) {
                    const_refs(&d.type_, &mut deps);
                    if let Some(val) = &d.value {
                        const_refs(val, &mut deps);
                    }
                }
                stack.push(Visit::Exit(name));
                for dep in deps {
                    if !seen.contains(&dep) && !skip(&dep) {
                        stack.push(Visit::Enter(dep));
                    }
                }
            }
            Visit::Exit(name) => order.push(name),
        }
    }
    order
}

/// Build the second-checker env from empty. Inductive admissions are hard
/// requirements (a failure yields `None` → everything `Unavailable`); definitions
/// are best-effort with a ledger trail.
fn build_env() -> Option<Ck0Env> {
    // Read-only source of prelude declarations — share the process-wide cached build
    // (this function itself runs once, behind `CK0_ENV`; the shared env is never mutated here).
    let clean: &Environment = crate::cleancic::prelude_env();
    let mut env = MinimalEnv::new();
    let mut ledger = Vec::new();
    // Keep in lockstep with ADMITTED_INDUCTIVE_LEVEL_ARITY. Order matters: each
    // admission's boot env is the env-so-far (`Int.NonNeg`'s ctor mentions `Nat`,
    // `Int`, and `Int.ofNat`, so it must come after both).
    let inductives: &[(&str, u32, u32, &[&str])] = &[
        ("Bool", 0, 0, &["Bool.false", "Bool.true"]),
        ("Nat", 0, 0, &["Nat.zero", "Nat.succ"]),
        ("Eq", 1, 2, &["Eq.refl"]),
        ("And", 0, 2, &["And.intro"]),
        ("Or", 0, 2, &["Or.inl", "Or.inr"]),
        ("False", 0, 0, &[]),
        ("True", 0, 0, &["True.intro"]),
        ("Int", 0, 0, &["Int.ofNat", "Int.negSucc"]),
        // Indexed family over Int (0 params, 1 index): mk : (n:Nat) → NonNeg (ofNat n).
        ("Int.NonNeg", 0, 0, &["Int.NonNeg.mk"]),
    ];
    for (name, nlvl, nparams, ctors) in inductives {
        if let Err(e) = admit_inductive(&mut env, clean, &mut ledger, name, *nlvl, *nparams, ctors)
        {
            debug_assert!(false, "ck0 env: inductive admission failed: {e}");
            return None;
        }
    }
    // Curated Nat/Bool core, dependency order — kept first, byte-identical to the
    // historical fragment (Native numeral encoding). Each is ck0-checked before
    // registration; failures shrink the fragment (Skipped in the ledger), never
    // weaken it.
    for name in [
        "Bool.and",
        "Bool.or",
        "Bool.not",
        "Nat.add",
        "Nat.mul",
        "Nat.sub",
        "Nat.beq",
        "Nat.ble",
        "Nat.div",
        "Nat.mod",
        "Nat.pow",
        "And.left",
        "And.right",
        "Not",
        "False.elim",
    ] {
        register_def(&mut env, clean, &mut ledger, name, NumeralEnc::Native);
    }
    // Int fragment: dependency-closure-driven, registered in postorder under the
    // Ctor numeral encoding (see module docs — the closure's prelude bodies are
    // literal-free today, so this is currently the identity choice, but it pins
    // coherence with the Int-obligation encoding if the prelude ever drifts).
    let in_ledger: std::collections::HashSet<String> =
        ledger.iter().map(|(n, _)| n.clone()).collect();
    let prune = |n: &str| in_ledger.contains(n) || is_recursor_family(n);
    for name in int_closure_postorder(clean, &prune) {
        register_def(&mut env, clean, &mut ledger, &name, NumeralEnc::Ctor);
    }
    Some(Ck0Env { env, ledger })
}

// ===========================================================================
// The second check itself.
// ===========================================================================

/// Independently re-check `term : expected` with clean-ck0. See module docs for
/// the outcome semantics; this function never panics on untrusted input.
pub fn corroborate(term: &Expr, expected: &Expr) -> Ck0Corroboration {
    let outcome = corroborate_inner(term, expected);
    tally(&outcome);
    outcome
}

/// ITERATIVE structural-depth pre-check mirroring ck0's own `check_raw_depth`: any expression
/// deeper than ck0's pinned `MAX_VALIDATE_DEPTH` would end `Unavailable` at validation anyway,
/// and the recursive [`tr_expr`] must never be the thing that discovers it — recursing into a
/// product-domain obligation with tens of thousands of nested conjuncts is a stack overflow
/// (a crash, not a fail-closed decline). Explicit worklist; no recursion.
fn exceeds_ck0_depth(e: &Expr) -> bool {
    let cap = clean_ck0::MAX_VALIDATE_DEPTH;
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
            ExprKind::Proj(_, _, inner) => stack.push((inner, d1)),
            _ => {}
        }
    }
    false
}

fn corroborate_inner(term: &Expr, expected: &Expr) -> Ck0Corroboration {
    let Some(ck0) = cached_env() else {
        return Ck0Corroboration::Unavailable("ck0 env construction failed".into());
    };
    // Depth gate BEFORE the recursive translator (see `exceeds_ck0_depth`): fail-closed to
    // `Unavailable` — semantics-preserving, since ck0's validate would reject the depth anyway.
    if exceeds_ck0_depth(expected) || exceeds_ck0_depth(term) {
        return Ck0Corroboration::Unavailable(format!(
            "nesting depth exceeds ck0's validation cap ({})",
            clean_ck0::MAX_VALIDATE_DEPTH
        ));
    }
    let lps: [KName; 0] = [];
    // Numeral encoding is decided ONCE for the whole obligation (see module docs):
    // an Int-fragment obligation must reach ck0 in constructor-chain form so the
    // Nat.rec-matching Int defs can ι-reduce; everything else keeps the historical
    // native-literal form.
    let enc = if mentions_int(expected)
        || mentions_int(term)
        || mentions_nat_rec(expected)
        || mentions_nat_rec(term)
    {
        NumeralEnc::Ctor
    } else {
        NumeralEnc::Native
    };
    // Translate the TYPE first and independently of the term (the obligation
    // binding lives in the type; a term-only translation bug cannot move it).
    let ty_raw = match tr_expr(expected, &lps, enc) {
        Ok(t) => t,
        Err(e) => return Ck0Corroboration::Unavailable(format!("type: {e}")),
    };
    let term_raw = match tr_expr(term, &lps, enc) {
        Ok(t) => t,
        Err(e) => return Ck0Corroboration::Unavailable(format!("term: {e}")),
    };
    let ty_term = match CkTerm::validate(&ck0.env, &ty_raw, 0, 0) {
        Ok(t) => t,
        Err(e) => return Ck0Corroboration::Unavailable(format!("type validate: {e:?}")),
    };
    let term_term = match CkTerm::validate(&ck0.env, &term_raw, 0, 0) {
        Ok(t) => t,
        Err(e) => return Ck0Corroboration::Unavailable(format!("term validate: {e:?}")),
    };
    let mut budget = Budget::new(CK0_CHECK_FUEL);
    match clean_ck0::check(&ck0.env, &term_term, &ty_term, &mut budget) {
        Ok(()) => Ck0Corroboration::Corroborated,
        Err(clean_ck0::InferError::OutOfBudget) => {
            Ck0Corroboration::Unavailable("ck0 budget exhausted (gave up, not a verdict)".into())
        }
        // An unknown constant surviving validation means an env-fragment gap, not a
        // refutation of the mathematics.
        Err(clean_ck0::InferError::UnknownConst { name }) => {
            Ck0Corroboration::Unavailable(format!("const `{name}` outside the ck0 fragment"))
        }
        Err(e) => Ck0Corroboration::Rejected(format!("{e:?}")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The env builds, every inductive admitted, and the load-bearing defs for the
    /// bool_true_eq obligation fragment are ck0-CHECKED (not skipped). If a prelude
    /// change breaks a def's translation, this pins the regression loudly.
    #[test]
    fn ck0_env_builds_with_checked_core_defs() {
        let ledger = env_ledger().expect("ck0 env must build");
        let entry = |n: &str| {
            ledger
                .iter()
                .find(|(name, _)| name == n)
                .map(|(_, e)| e.clone())
        };
        for ind in ["Bool", "Nat", "Eq", "And", "Or", "False", "True"] {
            assert_eq!(entry(ind), Some(LedgerEntry::Inductive), "{ind}");
        }
        for def in [
            "Bool.and", "Bool.or", "Bool.not", "Nat.beq", "Nat.ble", "Nat.add",
        ] {
            assert_eq!(entry(def), Some(LedgerEntry::CheckedDef), "{def}");
        }
        // THE axiom-freeness invariant: nothing in the ledger is a body-less typed
        // constant (there is no LedgerEntry variant for one, and the builder never
        // calls with_const_typed) — every name is kernel-admitted or ck0-checked.
    }

    /// ck0 independently re-checks a real bool_true_eq-shaped obligation: the
    /// embedded formula reduces to Bool.true INSIDE ck0 (its own native Nat
    /// reduction + its own derived Bool recursor), not inside clean-kernel.
    #[test]
    fn ck0_corroborates_bool_true_obligation() {
        use clean_kernel::Level;
        // C = Bool.and (Nat.beq (Nat.add 2 3) 5) (Bool.not Bool.false)  ~~> Bool.true
        let c = Expr::apps(
            Expr::const_str("Bool.and"),
            [
                Expr::apps(
                    Expr::const_str("Nat.beq"),
                    [
                        Expr::apps(
                            Expr::const_str("Nat.add"),
                            [Expr::nat_lit(2), Expr::nat_lit(3)],
                        ),
                        Expr::nat_lit(5),
                    ],
                ),
                Expr::app(Expr::const_str("Bool.not"), Expr::const_str("Bool.false")),
            ],
        );
        let lvl1 = vec![Level::succ(Level::zero())];
        let ty = Expr::apps(
            Expr::const_str_levels("Eq", lvl1.clone()),
            [Expr::const_str("Bool"), c, Expr::const_str("Bool.true")],
        );
        let term = Expr::apps(
            Expr::const_str_levels("Eq.refl", lvl1),
            [Expr::const_str("Bool"), Expr::const_str("Bool.true")],
        );
        assert_eq!(corroborate(&term, &ty), Ck0Corroboration::Corroborated);
    }

    /// Genuineness: a claim whose embedded formula reduces to Bool.FALSE is
    /// REJECTED by ck0 (the second check is not a rubber stamp).
    #[test]
    fn ck0_rejects_false_bool_obligation() {
        use clean_kernel::Level;
        let c = Expr::apps(
            Expr::const_str("Nat.beq"),
            [Expr::nat_lit(2), Expr::nat_lit(3)],
        );
        let lvl1 = vec![Level::succ(Level::zero())];
        let ty = Expr::apps(
            Expr::const_str_levels("Eq", lvl1.clone()),
            [Expr::const_str("Bool"), c, Expr::const_str("Bool.true")],
        );
        let term = Expr::apps(
            Expr::const_str_levels("Eq.refl", lvl1),
            [Expr::const_str("Bool"), Expr::const_str("Bool.true")],
        );
        assert!(matches!(
            corroborate(&term, &ty),
            Ck0Corroboration::Rejected(_)
        ));
    }

    /// The Int fragment is now INSIDE ck0's env: the `Eq Int (ofNat 0) (ofNat 0)`
    /// refl obligation that was historically `Unavailable` (Int outside the
    /// fragment) is corroborated after the closure-driven widening. This test
    /// REPLACED `ck0_unavailable_for_int_fragment` — its premise (Int legs are
    /// outside the fragment) is exactly what the widening removed;
    /// `ck0_unavailable_outside_widened_fragment` below keeps the honest
    /// `Unavailable` semantics pinned on genuinely-outside shapes.
    #[test]
    fn ck0_corroborates_int_refl_after_widening() {
        use clean_kernel::Level;
        let lvl1 = vec![Level::succ(Level::zero())];
        let x = Expr::app(Expr::const_str("Int.ofNat"), Expr::nat_lit(0));
        let ty = Expr::apps(
            Expr::const_str_levels("Eq", lvl1.clone()),
            [Expr::const_str("Int"), x.clone(), x.clone()],
        );
        let term = Expr::apps(
            Expr::const_str_levels("Eq.refl", lvl1),
            [Expr::const_str("Int"), x],
        );
        assert_eq!(corroborate(&term, &ty), Ck0Corroboration::Corroborated);
    }

    /// Obligations genuinely OUTSIDE the widened fragment stay `Unavailable` —
    /// never corroborated, never rejected: (a) an unregistered constant
    /// (`Int.mul` is not in the ingested closure), (b) an Int-fragment numeral
    /// over the ctor-chain cap.
    #[test]
    fn ck0_unavailable_outside_widened_fragment() {
        use clean_kernel::Level;
        let lvl1 = vec![Level::succ(Level::zero())];
        let x = Expr::apps(
            Expr::const_str("Int.mul"),
            [
                Expr::app(Expr::const_str("Int.ofNat"), Expr::nat_lit(1)),
                Expr::app(Expr::const_str("Int.ofNat"), Expr::nat_lit(1)),
            ],
        );
        let ty = Expr::apps(
            Expr::const_str_levels("Eq", lvl1.clone()),
            [Expr::const_str("Int"), x.clone(), x.clone()],
        );
        let term = Expr::apps(
            Expr::const_str_levels("Eq.refl", lvl1),
            [Expr::const_str("Int"), x],
        );
        assert!(matches!(
            corroborate(&term, &ty),
            Ck0Corroboration::Unavailable(_)
        ));
        // Over-cap numeral in an Int-fragment obligation: translation declines.
        let big = super::CTOR_NUMERAL_MAX + 1;
        let ty = Expr::app(
            Expr::const_str("Int.NonNeg"),
            Expr::app(Expr::const_str("Int.ofNat"), Expr::nat_lit(big)),
        );
        let term = Expr::app(Expr::const_str("Int.NonNeg.mk"), Expr::nat_lit(big));
        assert!(matches!(
            corroborate(&term, &ty),
            Ck0Corroboration::Unavailable(_)
        ));
    }

    /// The widened-fragment ledger: `Int` and the indexed family `Int.NonNeg`
    /// admitted as kernel-checked inductives, and every load-bearing def/lemma of
    /// the widened P3 legs ck0-CHECKED (not skipped) via the dependency-closure
    /// registration pass. Pins regressions loudly if a prelude change breaks a
    /// translation or a ck0 check.
    #[test]
    fn ck0_env_admits_int_fragment() {
        let ledger = env_ledger().expect("ck0 env must build");
        let entry = |n: &str| {
            ledger
                .iter()
                .find(|(name, _)| name == n)
                .map(|(_, e)| e.clone())
        };
        for ind in ["Int", "Int.NonNeg"] {
            assert_eq!(entry(ind), Some(LedgerEntry::Inductive), "{ind}");
        }
        for ctor in ["Int.ofNat", "Int.negSucc", "Int.NonNeg.mk"] {
            assert_eq!(entry(ctor), Some(LedgerEntry::Inductive), "{ctor}");
        }
        for def in [
            "Int.add",
            "Int.sub",
            "Int.le",
            "Int.NonNeg.add",
            "Int.le_trans",
            "Int.le_self_add_one",
            "Int.add_le_add_left",
            "Int.add_zero",
            "Int.le_refl",
        ] {
            assert_eq!(entry(def), Some(LedgerEntry::CheckedDef), "{def}");
        }
    }

    /// ck0 independently re-checks a REAL widened P3 consecution leg (the
    /// `bound > 0` shape cleancic emits): `Π(x:Int). le 2 x → le 2 (x+1)` proved
    /// by `λ x h. le_trans 2 x (x+1) h (le_self_add_one x)`. Exercises the
    /// Ctor numeral encoding end to end (the leg's `1` must be def-eq to the
    /// prelude lemma's `Nat.succ Nat.zero` — under chains it is IDENTICAL).
    #[test]
    fn ck0_corroborates_widened_int_consecution_leg() {
        let int_ty = || Expr::const_str("Int");
        let ofnat = |n: u64| Expr::app(Expr::const_str("Int.ofNat"), Expr::nat_lit(n));
        let ile = |a: Expr, b: Expr| Expr::apps(Expr::const_str("Int.le"), [a, b]);
        let iadd = |a: Expr, b: Expr| Expr::apps(Expr::const_str("Int.add"), [a, b]);
        // TYPE: Π(x:Int). le 2 x → le 2 (add x 1)   (x = bvar1 in the codomain)
        let ty = Expr::pi(
            BinderInfo::Default,
            int_ty(),
            Expr::pi(
                BinderInfo::Default,
                ile(ofnat(2), Expr::bvar(0)),
                ile(ofnat(2), iadd(Expr::bvar(1), ofnat(1))),
            ),
        );
        // TERM: λ(x:Int)(h: le 2 x). le_trans 2 x (add x 1) h (le_self_add_one x)
        let body = Expr::apps(
            Expr::const_str("Int.le_trans"),
            [
                ofnat(2),
                Expr::bvar(1),
                iadd(Expr::bvar(1), ofnat(1)),
                Expr::bvar(0),
                Expr::app(Expr::const_str("Int.le_self_add_one"), Expr::bvar(1)),
            ],
        );
        let term = Expr::lam(
            BinderInfo::Default,
            int_ty(),
            Expr::lam(BinderInfo::Default, ile(ofnat(2), Expr::bvar(0)), body),
        );
        assert_eq!(corroborate(&term, &ty), Ck0Corroboration::Corroborated);
    }

    /// ck0 corroborates the closed ground initiation `NonNeg.mk 3 : le 2 5` — the
    /// REAL reduction test: ck0 must δ-unfold `Int.le` to `NonNeg (sub 5 2)` and
    /// ι-reduce `sub`/`add`/`neg`/`subNatNat` through the translated defs (its own
    /// derived recursors) down to `NonNeg (ofNat 3)`.
    #[test]
    fn ck0_corroborates_int_ground_initiation_by_reduction() {
        let ofnat = |n: u64| Expr::app(Expr::const_str("Int.ofNat"), Expr::nat_lit(n));
        let ty = Expr::apps(Expr::const_str("Int.le"), [ofnat(2), ofnat(5)]);
        let term = Expr::app(Expr::const_str("Int.NonNeg.mk"), Expr::nat_lit(3));
        assert_eq!(corroborate(&term, &ty), Ck0Corroboration::Corroborated);
    }

    /// GENUINENESS: the false ground claim `NonNeg.mk 3 : le 5 2` is REJECTED —
    /// `le 5 2` reduces to `NonNeg (negSucc 2)` and the constructor heads clash.
    /// Must be `Rejected` (a real refutation), not `Unavailable`.
    #[test]
    fn ck0_rejects_false_int_ground_claim() {
        let ofnat = |n: u64| Expr::app(Expr::const_str("Int.ofNat"), Expr::nat_lit(n));
        let ty = Expr::apps(Expr::const_str("Int.le"), [ofnat(5), ofnat(2)]);
        let term = Expr::app(Expr::const_str("Int.NonNeg.mk"), Expr::nat_lit(3));
        assert!(matches!(
            corroborate(&term, &ty),
            Ck0Corroboration::Rejected(_)
        ));
    }
}
