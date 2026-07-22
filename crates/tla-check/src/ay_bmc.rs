// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! AY-based bounded model checking (BMC) for symbolic bug finding.
//!
//! This module wires `tla-ay`'s `BmcTranslator` into `tla-check` so callers can
//! run incremental deepening over a TLA+ `Init`/`Next`/`INVARIANT` spec.

use std::sync::Arc;
use std::time::{Duration, Instant};

use tla_ay::{
    AYError, BmcState, BmcTranslator, SolveResult, StrictProofVerdict, TlaSort, UnknownReason,
    UnsatProofArtifact,
};
use tla_core::ast::{Expr, Module, Unit};
use tla_core::name_intern::NameId;
use tla_core::visit::ExprVisitor;
use tla_core::Spanned;

use crate::ay_pdr::expand_operators_for_chc;
use crate::ay_shared;
use crate::check::CheckError;
use crate::config::Config;
use crate::eval::EvalCtx;
use crate::shared_verdict::{CertificateVerification, SharedVerdict};
use crate::symbolic_explore::AYSolveDecisionProfileEvidence;

/// Result of BMC-based symbolic bug finding.
#[derive(Debug)]
pub enum BmcResult {
    /// A counterexample exists within the requested bound.
    Violation {
        /// Minimal depth bound `k` at which the solver found a violation.
        depth: usize,
        /// Counterexample trace from step 0 through `depth`.
        trace: Vec<BmcState>,
    },
    /// No counterexample exists up to and including `max_depth`.
    BoundReached {
        /// Maximum depth checked exhaustively by BMC.
        max_depth: usize,
    },
    /// A REACHABLE deadlock state was found at `depth`: a concrete state that is
    /// reachable via the Init/Next prefix yet has NO successor under `Next`
    /// (i.e. `~Enabled(Next)` holds there). Like an explicit-state BFS Deadlock,
    /// this is treated as Unsafe by consumers. Additive variant — see
    /// `probe_deadlock_at_depth` for the soundness argument of the encoding.
    Deadlock {
        /// Minimal depth at which a reachable deadlock state was detected.
        depth: usize,
        /// Reachability-prefix trace from step 0 through the deadlocked state.
        trace: Vec<BmcState>,
    },
    /// Solver could not determine a result at `depth`.
    Unknown {
        /// Depth at which the unknown result occurred.
        depth: usize,
        /// Human-readable explanation of the unknown result.
        reason: String,
    },
}

/// BMC result plus the typed AY solve decision/profile boundary evidence.
#[derive(Debug)]
pub struct BmcRunResult {
    /// Legacy BMC result.
    pub result: BmcResult,
    /// Stable evidence row and structured consumer acceptance fields from AY.
    pub solver_decision_profile: AYSolveDecisionProfileEvidence,
}

impl BmcRunResult {
    fn new(result: BmcResult, solver_decision_profile: AYSolveDecisionProfileEvidence) -> Self {
        Self {
            result,
            solver_decision_profile,
        }
    }

    /// Borrow the summarizer-ready AY decision/profile evidence row.
    #[cfg_attr(not(test), allow(dead_code))] // exercised by ay_bmc/tests.rs
    pub fn solver_decision_profile_evidence(&self) -> &str {
        self.solver_decision_profile.evidence_row()
    }

    /// Consume this wrapper and return the legacy BMC result.
    pub fn into_result(self) -> BmcResult {
        self.result
    }
}

/// Construct an inconclusive BMC run when the portfolio did not execute BMC.
pub fn unknown_bmc_run_with_missing_evidence(reason: impl Into<String>) -> BmcRunResult {
    BmcRunResult::new(
        BmcResult::Unknown {
            depth: 0,
            reason: reason.into(),
        },
        AYSolveDecisionProfileEvidence::missing("TLA"),
    )
}

/// Errors specific to BMC checking.
#[derive(Debug, thiserror::Error)]
pub enum BmcError {
    /// Missing Init or Next definition.
    #[error("Missing specification: {0}")]
    MissingSpec(String),
    /// No invariants configured.
    #[error("No invariants configured for BMC checking")]
    NoInvariants,
    /// Failed to translate the TLA+ expression into BMC constraints.
    #[error("BMC translation failed: {0}")]
    TranslationError(String),
    /// Solver setup or execution failed.
    #[error("BMC solver failed: {0}")]
    SolverFailed(String),
    /// General checker error.
    #[error("Check error: {0:?}")]
    CheckError(#[from] CheckError),
}

impl From<AYError> for BmcError {
    fn from(err: AYError) -> Self {
        match err {
            AYError::Solver(inner) => BmcError::SolverFailed(inner.to_string()),
            other => BmcError::TranslationError(other.to_string()),
        }
    }
}

/// Configuration for bounded model checking.
#[derive(Debug, Clone)]
pub struct BmcConfig {
    /// Maximum depth to check with incremental deepening.
    pub max_depth: usize,
    /// Timeout applied to each per-depth solver invocation.
    pub solve_timeout: Option<Duration>,
    /// Enable lightweight debug logging to stderr.
    pub debug: bool,
    /// Use incremental solving: keep one solver instance across all depths,
    /// retaining learned clauses via `push`/`pop` scoping. Part of #3724.
    pub incremental: bool,
    /// Detect reachable DEADLOCK states (states with no `Next` successor) in
    /// addition to invariant violations. Mirrors TLC/explicit-BFS, which treat
    /// a reachable deadlock as a property failure. Default `true`. The probe is
    /// strictly additive (best-effort): it can only turn a wrong `BoundReached`
    /// (Safe) into a correct `Deadlock` (Unsafe), never the reverse, and never
    /// emits `Unknown`. See `probe_deadlock_at_depth`.
    pub check_deadlock: bool,
}

impl BmcConfig {
    /// Construct a config for a specific maximum depth with default timeouts.
    pub fn with_max_depth(max_depth: usize) -> Self {
        Self {
            max_depth,
            ..Self::default()
        }
    }
}

impl Default for BmcConfig {
    fn default() -> Self {
        let timeout_secs: u64 = std::env::var("TY_BMC_TIMEOUT_SECS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(300);

        // Incremental solving (push/pop clause reuse) is the default; it avoids
        // the cold O(k^2) restart of re-solving every prefix from scratch.
        // Env override remains for opt-out: TY_BMC_INCREMENTAL=0/false disables.
        // Part of #3724.
        let incremental = std::env::var("TY_BMC_INCREMENTAL")
            .ok()
            .map(|v| !(v == "0" || v.eq_ignore_ascii_case("false")))
            .unwrap_or(true);

        Self {
            max_depth: 0,
            solve_timeout: Some(Duration::from_secs(timeout_secs)),
            debug: debug_ay_bmc_enabled(),
            incremental,
            check_deadlock: true,
        }
    }
}

debug_flag!(debug_ay_bmc_enabled, "TY_DEBUG_AY_BMC");

/// When the wavefront starvation gap (sent - consumed) exceeds this threshold,
/// BMC drains intermediate wavefronts and skips to the latest one. This prevents
/// unbounded memory growth from wavefronts accumulating faster than BMC can
/// process them under high BFS throughput.
///
/// Part of #4004.
const STARVATION_THRESHOLD: u64 = 4;

/// Number of seeds (wavefronts + frontier samples) processed before the
/// persistent BMC translator is recreated to clear accumulated learned clauses.
/// Without periodic refresh, the translator's internal solver state grows
/// unboundedly, degrading solving performance over time.
///
/// Part of #4006.
const TRANSLATOR_REFRESH_INTERVAL: u64 = 64;

/// Create a BMC translator with the appropriate logic for the given variable sorts.
///
/// Delegates to [`ay_shared::make_translator`] for logic selection.
fn make_translator(
    var_sorts: &[(String, tla_ay::TlaSort)],
    depth: usize,
) -> Result<BmcTranslator, BmcError> {
    Ok(ay_shared::make_translator(var_sorts, depth)?)
}

// ===========================================================================
// Sound inductive interval-bound lemma injection.
//
// PROBLEM: ay symbolic BMC blows up proving UNSAT on SAFE specs with integer
// state vars + ITE/disjunctive Next, because each step's ITE condition (e.g.
// `count < 3`) becomes a free SAT selector the brancher enumerates across k
// steps (~2^k); the LIA theory cannot bound `count` a priori.
//
// FIX: derive a candidate interval bound B (lo <= v /\ v <= hi, conjoined over
// all integer state vars), PROVE B is inductive, then conjoin B at EVERY step
// of the real BMC query so LIA has a propagatable interval.
//
// SOUNDNESS (LOAD-BEARING): B is asserted ONLY if BOTH
//   (1) Init => B                          [Init(s0) /\ ~B(s0) is UNSAT]
//   (2) B /\ Next => B'   [B(s0) /\ Next(s0,s1) /\ ~B(s1) is UNSAT]
// hold. A proven-inductive B is logically implied by Init/Next, so conjoining
// B into the BMC query is equivalence-preserving: same SAT models, same UNSAT
// verdict, same counterexample — it only gives LIA an interval to propagate.
// If EITHER gate check is SAT or Unknown, B is NOT proven inductive and is
// NEVER asserted (behavior unchanged). The gate is the entire soundness
// argument; do not weaken it.
// ===========================================================================

/// Visitor that collects every integer literal appearing in an expression tree.
///
/// Used to seed the candidate interval [gMin, gMax] for the inductive bound.
/// A loose/wrong candidate is harmless: it simply fails the inductiveness gate
/// and is skipped (behavior unchanged).
#[derive(Default)]
struct IntLiteralCollector {
    lits: Vec<i64>,
}

impl ExprVisitor for IntLiteralCollector {
    type Output = ();

    fn visit_node(&mut self, expr: &Expr) -> Option<()> {
        if let Expr::Int(n) = expr {
            // Only track literals representable as i64; oversized literals are
            // simply ignored (a looser candidate just risks failing the gate).
            if let Ok(v) = i64::try_from(n) {
                self.lits.push(v);
            }
        }
        // Always recurse into children.
        None
    }
}

/// Collect the names of state variables that appear inside an arithmetic
/// operator (`+`, `-`, `*`, unary `-`) anywhere in `Next`.
///
/// PURPOSE (performance only — NOT soundness): interval bounds pay off when a
/// variable *accumulates* via arithmetic (e.g. `count' = count + 1`), where LIA
/// otherwise has no a-priori range and the ITE/guard selectors blow up. For
/// variables that only appear in equalities to literals (e.g. a token bit
/// `t1' = 0`), an interval bound is redundant with the spec's own structure and
/// merely adds assertions that slow the solver. Restricting candidate
/// generation to accumulating variables avoids that regression. Soundness is
/// unaffected: every kept candidate still passes the full inductiveness gate.
fn collect_arith_vars(expr: &Expr, acc: &mut std::collections::HashSet<String>) {
    // Helper: collect every variable name referenced anywhere in a subtree.
    fn collect_all_vars(e: &Expr, acc: &mut std::collections::HashSet<String>) {
        match e {
            Expr::Ident(name, _) | Expr::StateVar(name, ..) => {
                acc.insert(name.clone());
            }
            _ => {
                let mut child = |c: &Spanned<Expr>| collect_all_vars(&c.node, acc);
                walk_immediate_children(e, &mut child);
            }
        }
    }

    match expr {
        // Arithmetic context: every variable beneath is an "accumulating" var.
        Expr::Add(a, b) | Expr::Sub(a, b) | Expr::Mul(a, b) => {
            collect_all_vars(&a.node, acc);
            collect_all_vars(&b.node, acc);
        }
        Expr::Neg(a) => collect_all_vars(&a.node, acc),
        _ => {
            let mut child = |c: &Spanned<Expr>| collect_arith_vars(&c.node, acc);
            walk_immediate_children(expr, &mut child);
        }
    }
}

/// Invoke `f` on each immediate `Spanned<Expr>` child of `expr`.
///
/// Covers the structural cases reachable in Init/Next/Safety predicates; other
/// variants simply have no relevant children for arithmetic-var detection.
fn walk_immediate_children(expr: &Expr, f: &mut impl FnMut(&Spanned<Expr>)) {
    use Expr::*;
    match expr {
        And(a, b) | Or(a, b) | Implies(a, b) | Equiv(a, b) | Eq(a, b) | Neq(a, b)
        | Lt(a, b) | Gt(a, b) | Leq(a, b) | Geq(a, b) | In(a, b) | NotIn(a, b)
        | Add(a, b) | Sub(a, b) | Mul(a, b) | Div(a, b) | IntDiv(a, b) | Mod(a, b)
        | Pow(a, b) | Range(a, b) => {
            f(a);
            f(b);
        }
        Not(a) | Neg(a) | Prime(a) | Unchanged(a) => f(a),
        If(c, t, e) => {
            f(c);
            f(t);
            f(e);
        }
        Apply(op, args) => {
            f(op);
            for a in args {
                f(a);
            }
        }
        SetEnum(es) | Tuple(es) => {
            for e in es {
                f(e);
            }
        }
        _ => {}
    }
}

/// Build the candidate interval-bound predicate
/// `AND over int vars v of (lo <= v /\ v <= hi)` as a TLA AST.
///
/// Returns `None` when there are no integer state vars (nothing to bound).
/// Build the single-variable interval predicate `lo <= v /\ v <= hi` as a TLA AST.
fn build_var_interval_expr(name: &str, lo: i64, hi: i64) -> Spanned<Expr> {
    fn int_lit(n: i64) -> Spanned<Expr> {
        Spanned::dummy(Expr::Int(num_bigint::BigInt::from(n)))
    }
    // `Ident` with INVALID NameId — the BMC translator resolves state vars by
    // name from its declared `vars` map, so an unresolved Ident suffices.
    let var_ref = || Spanned::dummy(Expr::Ident(name.to_string(), NameId::INVALID));

    let ge = Spanned::dummy(Expr::Leq(Box::new(int_lit(lo)), Box::new(var_ref())));
    let le = Spanned::dummy(Expr::Leq(Box::new(var_ref()), Box::new(int_lit(hi))));
    Spanned::dummy(Expr::And(Box::new(ge), Box::new(le)))
}

/// Build the single-variable lower-bound predicate `lo <= v` as a TLA AST.
///
/// Used by the FIX-B certificate's strengthening: accumulating vars (`v'=v+1`)
/// are unbounded ABOVE, so an interval `[lo,hi]` is never inductive, but a pure
/// lower bound `lo <= v` often is, and the WHOLE-conjunction lower-bound
/// strengthening (`a>=0 /\ b>=0 /\ c>=0`) is what makes pipeline-style specs
/// inductive (where each var's bound depends on another's via the hypothesis).
fn build_var_lower_bound_expr(name: &str, lo: i64) -> Spanned<Expr> {
    fn int_lit(n: i64) -> Spanned<Expr> {
        Spanned::dummy(Expr::Int(num_bigint::BigInt::from(n)))
    }
    let var_ref = Spanned::dummy(Expr::Ident(name.to_string(), NameId::INVALID));
    Spanned::dummy(Expr::Leq(Box::new(int_lit(lo)), Box::new(var_ref)))
}

/// Conjoin a non-empty list of predicates into a single `And` chain.
fn conjoin(mut exprs: Vec<Spanned<Expr>>) -> Option<Spanned<Expr>> {
    let mut result = exprs.pop()?;
    while let Some(c) = exprs.pop() {
        result = Spanned::dummy(Expr::And(Box::new(c), Box::new(result)));
    }
    Some(result)
}

/// Discharge a single validity check via a fresh scratch translator/solver, so
/// the gate query never pollutes the real BMC query.
///
/// Builds the assertions via `build`, then returns `true` iff the resulting
/// formula is UNSAT (i.e. the implication is valid). SAT or Unknown -> `false`.
fn scratch_check_unsat(
    var_sorts: &[(String, TlaSort)],
    bound_k: usize,
    timeout: Option<Duration>,
    build: impl FnOnce(&mut BmcTranslator) -> Result<(), BmcError>,
) -> Result<bool, BmcError> {
    let mut t = make_translator(var_sorts, bound_k)?;
    t.set_timeout(timeout);
    for (name, sort) in var_sorts {
        t.declare_var(name, sort.clone())?;
    }
    build(&mut t)?;
    match t.try_check_sat()? {
        SolveResult::Unsat(_) => Ok(true),
        // SAT or Unknown: implication NOT proven valid -> not inductive.
        _ => Ok(false),
    }
}

/// A single discharged proof obligation with AY's own re-checkable proof.
///
/// This is the content of the certificate's AY proof leg: AY produced a proof
/// that the UNSAT query holds AND strict-checked it (`strict_verdict ==
/// Verified` means `ay-proof`'s `check_proof_strict` accepted every step). The
/// proof is scoped to the asserted problem (`export_alethe_with_problem_scope`),
/// so re-checking it in-process — re-asserting the obligation, re-solving, and
/// requiring `Verified` — is sound for THIS obligation by construction (no
/// serialized-proof-of-a-different-formula trust).
#[derive(Clone, Debug)]
pub(crate) struct ObligationProof {
    /// `"initiation"`, `"consecution"`, or `"safety"`.
    pub(crate) name: &'static str,
    /// Whether the query was UNSAT (the implication is valid).
    pub(crate) unsat: bool,
    /// Whether AY strict-checked the proof (`check_proof_strict` == Verified).
    pub(crate) strict_verified: bool,
    /// Whether every step is in the clean-supported subset AND strict-verified.
    pub(crate) clean_supported: bool,
    /// Rendered Alethe proof text (problem-scoped), embeddable in a certificate.
    pub(crate) alethe: String,
    /// Whether a replayable LRAT SAT-backbone certificate was emitted.
    pub(crate) lrat_present: bool,
    /// Leg D: serde_json of the portable, checker-only `SerializableProofBundle`
    /// AY exported for THIS UNSAT query (term entries + proof steps + the
    /// obligation assertion ids). `None` when the query was not UNSAT, proofs
    /// were off, or no bundle was exposed (e.g. the structural deadlock marker).
    /// Embedded verbatim in a certificate so the proof can be re-checked offline
    /// without re-running the solver.
    pub(crate) bundle_json: Option<String>,
}

/// Run a scratch UNSAT query WITH proof production and capture AY's proof artifact.
fn scratch_check_unsat_with_proof(
    name: &'static str,
    var_sorts: &[(String, TlaSort)],
    bound_k: usize,
    timeout: Option<Duration>,
    build: impl FnOnce(&mut BmcTranslator) -> Result<(), BmcError>,
) -> Result<ObligationProof, BmcError> {
    let mut t = make_translator(var_sorts, bound_k)?;
    t.set_timeout(timeout);
    t.set_produce_proofs(true);
    for (var_name, sort) in var_sorts {
        t.declare_var(var_name, sort.clone())?;
    }
    build(&mut t)?;
    let unsat = matches!(t.try_check_sat()?, SolveResult::Unsat(_));
    let artifact: Option<UnsatProofArtifact> = if unsat {
        t.export_last_unsat_artifact()
    } else {
        None
    };
    // Leg D: capture AY's portable, checker-only proof bundle for offline
    // re-check. Serialized eagerly so the certificate carries plain JSON, not an
    // AY type. Independent of the (in-process) artifact above.
    let bundle_json: Option<String> = if unsat {
        t.export_last_unsat_bundle()
            .as_ref()
            .and_then(|b| serde_json::to_string(b).ok())
    } else {
        None
    };
    let (strict_verified, clean_supported, alethe, lrat_present) = match artifact {
        Some(a) => (
            matches!(a.strict_verdict, StrictProofVerdict::Verified(_)),
            a.clean_supported,
            a.alethe,
            a.lrat_certificate.is_some(),
        ),
        None => (false, false, String::new(), false),
    };
    Ok(ObligationProof {
        name,
        unsat,
        // A strict-verified proof only counts when the query was actually UNSAT.
        strict_verified: unsat && strict_verified,
        clean_supported,
        alethe,
        lrat_present,
        bundle_json,
    })
}

/// Discharge the FOUR certificate obligations for `j` WITH AY proofs: initiation
/// (`Init => J`), consecution (`J /\ Next => J'`), safety (`J => Safety`), and
/// DEADLOCK-FREEDOM (`J => Enabled(Next)` — every state satisfying `J` has a
/// successor, so the spec cannot deadlock). Each is an UNSAT query carrying AY's
/// strict-checked proof. `enabled` is the enabling predicate `Enabled(Next)` (the
/// conjunction of Next's guards, or TRUE for an unguarded total Next; see
/// [`enabled_of_next`]). Mirrors [`gate_is_inductive`] + the implication checks,
/// but produces proofs and adds the deadlock obligation.
pub(crate) fn discharge_obligations_with_proofs(
    var_sorts: &[(String, TlaSort)],
    init: &Spanned<Expr>,
    next: &Spanned<Expr>,
    safety: &Spanned<Expr>,
    j: &Spanned<Expr>,
    enabled: &Spanned<Expr>,
    timeout: Option<Duration>,
) -> Result<Vec<ObligationProof>, BmcError> {
    let not_j = negate_normalized(j);
    let not_safety = negate_normalized(safety);
    let not_enabled = negate_normalized(enabled);

    // The three SMT obligations are translated through the SINGLE source of
    // truth `build_smt_obligation`, so the producer here (which then solves +
    // proves) and the Leg D re-checker (which re-translates WITHOUT solving to
    // bind the embedded proof to the obligation) translate byte-for-byte
    // identically — the assertion sets can never drift apart.
    let initiation = scratch_check_unsat_with_proof("initiation", var_sorts, 1, timeout, |t| {
        build_smt_obligation(t, SmtObligation::Initiation, init, next, &not_safety, j, &not_j)
            .map(|_| ())
    })?;

    let consecution = scratch_check_unsat_with_proof("consecution", var_sorts, 1, timeout, |t| {
        build_smt_obligation(t, SmtObligation::Consecution, init, next, &not_safety, j, &not_j)
            .map(|_| ())
    })?;

    let safety_ob = scratch_check_unsat_with_proof("safety", var_sorts, 1, timeout, |t| {
        build_smt_obligation(t, SmtObligation::Safety, init, next, &not_safety, j, &not_j)
            .map(|_| ())
    })?;

    // Deadlock-freedom: J /\ ~Enabled(Next) is UNSAT, i.e. J => Enabled(Next), so
    // every reachable state has a successor (no deadlock). For MODULE Dead this is
    // SAT (x=3 satisfies J=x<=3 but not Enabled=x<3) -> NOT strict-verified -> reject.
    //
    // The UNGUARDED case (Enabled == TRUE) is structural: a total Next with no guard
    // always has a successor (proven by analyze_deadlock_freedom). The SMT query is
    // then the degenerate `J /\ FALSE`, whose AY proof is a trivial-conflict TRUST
    // step (not strict-verifiable), so we mark it structurally — sound by the
    // decomposition AND independently cross-checked by the eval oracle (which runs
    // with check_deadlock=true and refutes any reachable deadlock).
    let deadlock_freedom = if matches!(enabled.node, Expr::Bool(true)) {
        ObligationProof {
            name: "deadlock_freedom",
            unsat: true,
            strict_verified: true,
            clean_supported: true,
            alethe: "structural: unguarded total Next => Enabled(Next) == TRUE".to_string(),
            lrat_present: false,
            // Structural marker: no SMT proof, so Leg D does not cover this
            // obligation (it stays cross-checked by the deadlock-aware oracle).
            bundle_json: None,
        }
    } else {
        scratch_check_unsat_with_proof("deadlock_freedom", var_sorts, 1, timeout, |t| {
            let j0 = t.translate_safety_at_step(j, 0)?;
            t.assert(j0);
            let not_enabled0 = t.translate_safety_at_step(&not_enabled, 0)?;
            t.assert(not_enabled0);
            Ok(())
        })?
    };

    Ok(vec![initiation, consecution, safety_ob, deadlock_freedom])
}

/// One of the three SMT obligations covered by the external proof re-check
/// (Leg D). Deadlock-freedom is NOT a member: for the unguarded total-Next case
/// it is structural (the degenerate `J /\ FALSE` is a trivial-conflict trust
/// step, not strict-verifiable), so it carries no SMT proof bundle.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum SmtObligation {
    /// `Init /\ ~J@0` is UNSAT, i.e. `Init => J`.
    Initiation,
    /// `J@0 /\ Next(0,1) /\ ~J@1` is UNSAT, i.e. `J /\ Next => J'`.
    Consecution,
    /// `J@0 /\ ~Safety@0` is UNSAT, i.e. `J => Safety`.
    Safety,
}

impl SmtObligation {
    /// The obligation's stable name (matches the [`ObligationProof::name`]s).
    pub(crate) fn name(self) -> &'static str {
        match self {
            SmtObligation::Initiation => "initiation",
            SmtObligation::Consecution => "consecution",
            SmtObligation::Safety => "safety",
        }
    }
}

/// Assert ONE SMT obligation into `t` and return the asserted top-level terms in
/// assertion order. This is the SINGLE SOURCE OF TRUTH for each obligation's
/// asserted problem, consumed by BOTH the producer
/// ([`discharge_obligations_with_proofs`], which then solves + proves) and the
/// Leg D re-checker ([`retranslate_obligation_canonical`], which re-translates
/// WITHOUT solving). `not_j` / `not_safety` are the caller's already
/// negation-normalized predicates, so the two paths translate identically.
pub(crate) fn build_smt_obligation(
    t: &mut BmcTranslator,
    ob: SmtObligation,
    init: &Spanned<Expr>,
    next: &Spanned<Expr>,
    not_safety: &Spanned<Expr>,
    j: &Spanned<Expr>,
    not_j: &Spanned<Expr>,
) -> Result<Vec<tla_ay::Term>, BmcError> {
    let mut asserted = Vec::new();
    match ob {
        SmtObligation::Initiation => {
            let init_term = t.translate_init(init)?;
            t.assert(init_term);
            asserted.push(init_term);
            let not_j0 = t.translate_safety_at_step(not_j, 0)?;
            t.assert(not_j0);
            asserted.push(not_j0);
        }
        SmtObligation::Consecution => {
            let j0 = t.translate_safety_at_step(j, 0)?;
            t.assert(j0);
            asserted.push(j0);
            let next_term = t.translate_next(next, 0)?;
            t.assert(next_term);
            asserted.push(next_term);
            let not_j1 = t.translate_safety_at_step(not_j, 1)?;
            t.assert(not_j1);
            asserted.push(not_j1);
        }
        SmtObligation::Safety => {
            let j0 = t.translate_safety_at_step(j, 0)?;
            t.assert(j0);
            asserted.push(j0);
            let not_safety0 = t.translate_safety_at_step(not_safety, 0)?;
            t.assert(not_safety0);
            asserted.push(not_safety0);
        }
    }
    Ok(asserted)
}

/// The obligation inputs re-derived from a certificate's spec + invariant text,
/// via TY's front end + translator-preparation pipeline (NOT its solver). Shared
/// by [`certificate_obligation_proofs`] (Leg C re-discharge) and Leg D part-2
/// ([`retranslate_obligation_canonical`]), so both reason about the SAME ASTs.
pub(crate) struct ObligationInputs {
    /// State variables and their inferred sorts.
    pub(crate) var_sorts: Vec<(String, TlaSort)>,
    /// `Init` (operators expanded).
    pub(crate) init: Spanned<Expr>,
    /// `Next` (operators expanded).
    pub(crate) next: Spanned<Expr>,
    /// The conjunction of configured safety invariants (operators expanded).
    pub(crate) safety: Spanned<Expr>,
    /// The candidate inductive invariant `J` (operators expanded).
    pub(crate) j: Spanned<Expr>,
    /// `Enabled(Next)` (the conjunction of Next's guards; `TRUE` if unguarded).
    pub(crate) enabled: Spanned<Expr>,
    /// The conjunction of the module's `ASSUME` declarations that mention a
    /// SYMBOLIC constant (`TRUE` if none) — e.g. `ASSUME N ≥ 1`. Conjoined into
    /// the deadlock-freedom obligation so a spec that admits an empty-domain
    /// `N=0` (where `∃p∈1..N` is unenabled) is certified only over the `N` its
    /// own `ASSUME` admits. DERIVED from the re-parsed spec at BOTH mint and
    /// verify, so the render-binding covers it with no certificate schema change.
    pub(crate) assume: Spanned<Expr>,
}

/// Conjoin the module's `ASSUME` expressions that mention a SYMBOLIC constant
/// (the all-`N` target), so the deadlock obligation can be discharged over
/// exactly the `N` the spec admits. `TRUE` if the spec makes no such assumption.
/// SOUND: this only adds spec-DECLARED facts (never injected), and it is derived
/// identically at mint and verify, so it cannot widen acceptance beyond the spec.
fn extract_symbolic_assume(
    module: &Module,
    symbolic_constants: &std::collections::HashSet<String>,
) -> Spanned<Expr> {
    let mut acc: Option<Spanned<Expr>> = None;
    for unit in &module.units {
        if let Unit::Assume(decl) = &unit.node {
            if symbolic_constants.iter().any(|c| expr_mentions(&decl.expr.node, c)) {
                acc = Some(match acc {
                    None => decl.expr.clone(),
                    Some(prev) => mk_and(prev, decl.expr.clone()),
                });
            }
        }
    }
    acc.unwrap_or_else(|| Spanned::dummy(Expr::Bool(true)))
}

/// Re-derive the obligation inputs from a certificate's spec source + invariant
/// text via TY's front end + translator pipeline (NOT its solver). Returns
/// `None` if the spec cannot be parsed/lowered or `Enabled(Next)` is not cleanly
/// decomposable (in which case the certificate cannot be re-validated).
/// The module's declared `CONSTANT` names that `config` does NOT bind — i.e.
/// the SYMBOLIC (unbound) constants in scope. On the all-`N` lane this is the
/// singleton symbolic target; on every ordinary path it is empty (all constants
/// are config-concretized), so downstream symbolic-domain recognition is inert.
fn symbolic_constant_names(module: &Module, config: &Config) -> std::collections::HashSet<String> {
    let mut set = std::collections::HashSet::new();
    for unit in &module.units {
        if let Unit::Constant(decls) = &unit.node {
            for decl in decls {
                // Operator constants (arity present) are not scalar domain
                // bounds; skip them so only nullary constants are considered.
                if decl.arity.is_none() && !config.constants.contains_key(&decl.name.node) {
                    set.insert(decl.name.node.clone());
                }
            }
        }
    }
    set
}

/// Desugar the EXCEPT self-reference `@` (parsed as `Ident("@")`) to the explicit
/// OLD-VALUE expression at each spec's path: `[r EXCEPT !.a = @ + 1]` becomes
/// `[r EXCEPT !.a = r.a + 1]`; `[f EXCEPT ![i] = @ + 1]` becomes `[f EXCEPT ![i] =
/// f[i] + 1]`. This is the DEFINITIONAL meaning of `@` (TLA+ — `@` is the base's
/// value at that path), so the rewrite is exact and sound. Folded BOTTOM-UP so a
/// nested EXCEPT resolves its own `@` before an enclosing one substitutes (each
/// `@` binds to its nearest enclosing EXCEPT). DETERMINISTIC and applied on the
/// SHARED `rederive_obligation_inputs` path, so mint and verify desugar identically
/// and the render-binding stays symmetric. An expr with no `@` is unchanged.
fn desugar_except_at(e: &Spanned<Expr>) -> Spanned<Expr> {
    use tla_core::ast::{ExceptPathElement, ExceptSpec};
    struct D;
    impl tla_core::ExprFold for D {
        fn fold_expr(&mut self, e: Spanned<Expr>) -> Spanned<Expr> {
            // Fold children first (bottom-up): inner EXCEPTs resolve their own `@`.
            let folded = Spanned {
                node: self.fold_expr_inner(e.node),
                span: e.span,
            };
            let Expr::Except(base, specs) = &folded.node else {
                return folded;
            };
            let new_specs: Vec<ExceptSpec> = specs
                .iter()
                .map(|sp| {
                    // old-value = base with THIS spec's path applied (`r` + `.a` =
                    // `r.a`; `f` + `[i]` = `f[i]`). `base` is already folded, so it
                    // carries no unresolved `@`.
                    let mut old = (**base).clone();
                    for pe in &sp.path {
                        old = match pe {
                            ExceptPathElement::Index(ix) => Spanned::dummy(Expr::FuncApply(
                                Box::new(old),
                                Box::new(ix.clone()),
                            )),
                            ExceptPathElement::Field(f) => {
                                Spanned::dummy(Expr::RecordAccess(Box::new(old), f.clone()))
                            }
                        };
                    }
                    ExceptSpec {
                        path: sp.path.clone(),
                        value: substitute_at(&sp.value, &old),
                    }
                })
                .collect();
            Spanned {
                node: Expr::Except(base.clone(), new_specs),
                span: folded.span,
            }
        }
    }
    let mut d = D;
    tla_core::ExprFold::fold_expr(&mut d, e.clone())
}

/// Substitute every EXCEPT self-reference `@` (`Ident("@")`) by `replacement`.
/// Used only by [`desugar_except_at`], at the level of a single EXCEPT spec (so the
/// `@`s it targets all bind to that one EXCEPT — nested ones are already resolved).
fn substitute_at(e: &Spanned<Expr>, replacement: &Spanned<Expr>) -> Spanned<Expr> {
    struct S<'a> {
        r: &'a Spanned<Expr>,
    }
    impl tla_core::ExprFold for S<'_> {
        fn fold_expr(&mut self, e: Spanned<Expr>) -> Spanned<Expr> {
            if matches!(&e.node, Expr::Ident(n, _) if n == "@") {
                return self.r.clone();
            }
            Spanned {
                node: self.fold_expr_inner(e.node),
                span: e.span,
            }
        }
    }
    let mut s = S { r: replacement };
    tla_core::ExprFold::fold_expr(&mut s, e.clone())
}

/// Expand a structural set membership `elem \in set` into an equivalent predicate
/// the BMC translator can handle, for the two set-constructor shapes a record
/// TypeInvariant / Init uses. Returns `None` for any other `set` (the membership
/// is left verbatim — a range / enum / symbolic set BMC already translates).
/// EXACT + SOUND (each is the definition of the constructor); RECURSIVE, so a
/// record-set domain inside a filter (or a record-of-records) expands fully:
///   * `r \in [f1 : D1, …]`      ==  `⋀ expand(r.fi ∈ Di)`   (record-set)
///   * `x \in {c \in S : P(c)}`  ==  `expand(x ∈ S) /\ P(x)`  (set filter)
fn expand_membership(elem: &Spanned<Expr>, set: &Spanned<Expr>) -> Option<Spanned<Expr>> {
    use tla_core::ast::RecordFieldName;
    match &set.node {
        Expr::RecordSet(fields) => {
            let mut conj: Option<Spanned<Expr>> = None;
            for (fname, domain) in fields {
                let access = Spanned::dummy(Expr::RecordAccess(
                    Box::new(elem.clone()),
                    RecordFieldName::new(fname.clone()),
                ));
                let member = expand_membership(&access, domain).unwrap_or_else(|| {
                    Spanned::dummy(Expr::In(Box::new(access.clone()), Box::new(domain.clone())))
                });
                conj = Some(match conj {
                    None => member,
                    Some(acc) => mk_and(acc, member),
                });
            }
            Some(conj.unwrap_or_else(|| Spanned::dummy(Expr::Bool(true))))
        }
        Expr::SetFilter(bv, pred) => {
            // Only a simple `c \in S` binder (no tuple destructuring); the domain
            // must be present. `x \in {c \in S : P(c)}` == `x \in S /\ P(x)`.
            if bv.pattern.is_some() {
                return None;
            }
            let domain = bv.domain.as_ref()?;
            let dom_member = expand_membership(elem, domain).unwrap_or_else(|| {
                Spanned::dummy(Expr::In(Box::new(elem.clone()), Box::new((**domain).clone())))
            });
            let pred_sub = substitute_ident(pred, &bv.name.node, elem);
            Some(mk_and(dom_member, pred_sub))
        }
        _ => None,
    }
}

/// Rewrite every structural set membership (`\in` a record-set or set-filter) in
/// `e` via [`expand_membership`]. The BMC translator has no record-set /
/// set-filter membership primitive but DOES translate per-field range/enum
/// membership, so this is what lets a record TypeInvariant / Init flow through.
/// Deterministic + on the shared `rederive_obligation_inputs` path ⇒ symmetric
/// across mint and verify. A membership over a plain range / enum is left as-is.
fn expand_set_membership(e: &Spanned<Expr>) -> Spanned<Expr> {
    struct R;
    impl tla_core::ExprFold for R {
        fn fold_expr(&mut self, e: Spanned<Expr>) -> Spanned<Expr> {
            let folded = Spanned {
                node: self.fold_expr_inner(e.node),
                span: e.span,
            };
            if let Expr::In(elem, set) = &folded.node {
                if let Some(expanded) = expand_membership(elem, set) {
                    return expanded;
                }
            }
            folded
        }
    }
    let mut r = R;
    tla_core::ExprFold::fold_expr(&mut r, e.clone())
}

/// Cert-lane spec normalization applied uniformly to every obligation expression
/// after operator expansion: desugar EXCEPT `@` and expand structural set
/// memberships (record-set / set-filter). Both are exact definitional rewrites;
/// running them on the shared [`rederive_obligation_inputs`] path keeps mint and
/// verify byte-identical.
fn normalize_cert_expr(e: &Spanned<Expr>) -> Spanned<Expr> {
    desugar_except_at(&expand_set_membership(e))
}

pub(crate) fn rederive_obligation_inputs(
    spec_src: &str,
    config: &Config,
    j_tla: &str,
) -> Option<ObligationInputs> {
    const CERT_J_OP: &str = "TY__Cert_J";
    // Locate the START of the FIRST module terminator LINE (`====…`). Two hazards:
    //   (1) a plain `rfind("====")` matches the RIGHTMOST 4-char window inside a long
    //       terminator (`====…====`, 70+ chars), stranding the op after a split line;
    //   (2) `rfind("\n====")` in a MULTI-module source (byihive/braf: 3 modules / 3
    //       terminators) picks the LAST module's terminator, so the op lands in the LAST
    //       module — but `tla_core::lower` binds the FIRST, so the op is never in the
    //       bound module ("Operator not found").
    // Anchor on the FIRST `\n====` so the op lands just before the first module's true
    // terminator line — inside exactly the module `tla_core::lower` binds.
    let term_pos = ay_shared::first_module_terminator_pos(spec_src)?;
    let augmented = format!(
        "{}\n{CERT_J_OP} == {j_tla}\n\n{}",
        spec_src[..term_pos].trim_end(),
        &spec_src[term_pos..]
    );
    let tree = tla_core::parse_to_syntax_tree(&augmented);
    let module = tla_core::lower(tla_core::FileId(0), &tree).module?;

    let mut ctx = EvalCtx::new();
    ctx.load_module(&module);

    let symbolic_ctx = ay_shared::symbolic_ctx_with_config(&ctx, config).ok()?;
    let resolved = ay_shared::resolve_init_next(config, &symbolic_ctx).ok()?;
    let init_expr = ay_shared::get_operator_body(&symbolic_ctx, &resolved.init).ok()?;
    let next_expr = ay_shared::get_operator_body(&symbolic_ctx, &resolved.next).ok()?;
    // Desugar EXCEPT `@` after operator expansion (records/functions write
    // `!.f = @ - 1`; the BMC translator has no `@` binding — it must become the
    // explicit old value). Definitional + deterministic; both mint and verify run
    // this SAME path, keeping the render-binding symmetric.
    let init_expanded = normalize_cert_expr(&expand_operators_for_chc(&symbolic_ctx, &init_expr, false));
    let next_expanded = normalize_cert_expr(&expand_operators_for_chc(&symbolic_ctx, &next_expr, true));
    let safety_expr =
        ay_shared::build_safety_conjunction(&symbolic_ctx, &config.invariants).ok()?;
    let safety_expanded = normalize_cert_expr(&expand_operators_for_chc(&symbolic_ctx, &safety_expr, false));
    let j_expr = ay_shared::get_operator_body(&symbolic_ctx, CERT_J_OP).ok()?;
    let j_expanded = normalize_cert_expr(&expand_operators_for_chc(&symbolic_ctx, &j_expr, false));

    let vars = ay_shared::collect_state_vars(&module, &symbolic_ctx);
    // Symbolic (unbound) CONSTANTs: the module's declared CONSTANTs that the
    // config does NOT bind. On the all-`N` lane exactly the symbolic target is
    // unbound (`config_without_constant` removed it); on the ordinary safety
    // cert path every constant is bound so the set is empty (no behavior
    // change). A function-set membership over such a constant's contiguous
    // range is then typed as a map-only `FunctionSym` (see
    // `infer_var_sorts_with_symbolic`).
    let symbolic_constants = symbolic_constant_names(&module, config);
    let var_sorts = ay_shared::infer_var_sorts_with_symbolic(
        &vars,
        &init_expanded,
        &config.invariants,
        &symbolic_ctx,
        &symbolic_constants,
    );

    // Re-derive Enabled(Next) for the deadlock-freedom obligation. If Next is not
    // cleanly decomposable, the certificate cannot be re-validated for
    // deadlock-freedom -> return None (the verifier then rejects, never accepts).
    let var_names: Vec<String> = vars.iter().map(|v| v.to_string()).collect();
    // FunctionSym state vars: map-only TOTAL arrays, so a read `f[k]` is a total
    // value the Enabled derivation may treat as a successor (slice-2 read-valued
    // writes). Empty on non-all-N specs (no FunctionSym var) → inert.
    let funcsym_vars: std::collections::HashSet<String> =
        funcsym_domains(&var_sorts).into_keys().collect();
    let enabled = if string_enum_deadlock_free(&init_expanded, &next_expanded, &var_sorts, &var_names)
    {
        // F1 structural deadlock-freedom: a STRING "clock" variable whose closed
        // reachable literal universe is fully COVERED by unconditional-per-literal
        // actions is deadlock-free on every reachable state (`Enabled ≡ TRUE`
        // there), discharged structurally — the disjunctive enum `~Enabled` is
        // outside AY's strict Farkas fragment, but the coverage+closure argument
        // needs no solver. See [`string_enum_deadlock_free`].
        Spanned::dummy(Expr::Bool(true))
    } else {
        enabled_of_next(&next_expanded, &var_names, &funcsym_vars)?
    };

    let assume = extract_symbolic_assume(&module, &symbolic_constants);

    Some(ObligationInputs {
        var_sorts,
        init: init_expanded,
        next: next_expanded,
        safety: safety_expanded,
        j: j_expanded,
        enabled,
        assume,
    })
}

/// Re-derive JUST the `(measure, Next)` ASTs (operators expanded) for the affine descent kernel
/// leg — independent of the `J`/`P`/`Enabled` machinery, so it succeeds even when the FULL liveness
/// cert is blocked in the SMT layer (e.g. a record-set-membership invariant). `measure_op` is the
/// integer-measure operator name. `None` if the spec cannot be parsed/lowered or the operators are
/// not found.
pub(crate) fn rederive_measure_next(
    spec_src: &str,
    config: &Config,
    measure_op: &str,
) -> Option<(Spanned<Expr>, Spanned<Expr>, Vec<String>)> {
    let tree = tla_core::parse_to_syntax_tree(spec_src);
    let module = tla_core::lower(tla_core::FileId(0), &tree).module?;
    // The DECLARED state variables — the authoritative set a genuine stutter must leave unchanged.
    let vars: Vec<String> = module
        .units
        .iter()
        .flat_map(|u| match &u.node {
            Unit::Variable(decls) => decls.iter().map(|d| d.node.clone()).collect::<Vec<_>>(),
            _ => Vec::new(),
        })
        .collect();
    let mut ctx = EvalCtx::new();
    ctx.load_module(&module);
    let symbolic_ctx = ay_shared::symbolic_ctx_with_config(&ctx, config).ok()?;
    let resolved = ay_shared::resolve_init_next(config, &symbolic_ctx).ok()?;
    let next_expr = ay_shared::get_operator_body(&symbolic_ctx, &resolved.next).ok()?;
    let next_expanded = expand_operators_for_chc(&symbolic_ctx, &next_expr, true);
    let m_expr = ay_shared::get_operator_body(&symbolic_ctx, measure_op).ok()?;
    let m_expanded = expand_operators_for_chc(&symbolic_ctx, &m_expr, false);
    Some((m_expanded, next_expanded, vars))
}

/// Re-derive the liveness obligation inputs from a certificate's spec + the
/// `J`/`P`/`m` TLA texts (operators expanded). Both the producer and the verifier
/// use this so they reason about identical ASTs. `None` if the spec cannot be
/// parsed/lowered or `Enabled(Next)` is not cleanly decomposable.
pub(crate) fn rederive_liveness_inputs(
    spec_src: &str,
    config: &Config,
    j_tla: &str,
    p_tla: &str,
    m_tla: &str,
) -> Option<LiveInputs> {
    const J_OP: &str = "TY__Live_J";
    const P_OP: &str = "TY__Live_P";
    const M_OP: &str = "TY__Live_M";
    // Locate the START of the FIRST module terminator LINE (`====…`). Two hazards, both
    // fixed here (see `rederive_obligation_inputs`): a plain `rfind("====")` splits long
    // `====…====` lines, and `rfind("\n====")` in a MULTI-module source picks the LAST
    // module's terminator while `tla_core::lower` binds the FIRST. Anchor on the FIRST
    // `\n====`. The op NAMES also lead with a letter (`TY__…`, not `__…`): a `_`-leading
    // name directly after a unit ending in `]`/`>>` is eaten by the `[A]_v`/`<A>_v`
    // action-subscript lexer, truncating the module before the injected ops.
    let term_pos = ay_shared::first_module_terminator_pos(spec_src)?;
    let augmented = format!(
        "{}\n{J_OP} == {j_tla}\n{P_OP} == {p_tla}\n{M_OP} == {m_tla}\n\n{}",
        spec_src[..term_pos].trim_end(),
        &spec_src[term_pos..]
    );
    let tree = tla_core::parse_to_syntax_tree(&augmented);
    let module = tla_core::lower(tla_core::FileId(0), &tree).module?;

    let mut ctx = EvalCtx::new();
    ctx.load_module(&module);

    let symbolic_ctx = ay_shared::symbolic_ctx_with_config(&ctx, config).ok()?;
    let resolved = ay_shared::resolve_init_next(config, &symbolic_ctx).ok()?;
    let init_expr = ay_shared::get_operator_body(&symbolic_ctx, &resolved.init).ok()?;
    let next_expr = ay_shared::get_operator_body(&symbolic_ctx, &resolved.next).ok()?;
    let init_expanded = expand_operators_for_chc(&symbolic_ctx, &init_expr, false);
    let next_expanded = expand_operators_for_chc(&symbolic_ctx, &next_expr, true);
    let j_expr = ay_shared::get_operator_body(&symbolic_ctx, J_OP).ok()?;
    let j_expanded = expand_operators_for_chc(&symbolic_ctx, &j_expr, false);
    let p_expr = ay_shared::get_operator_body(&symbolic_ctx, P_OP).ok()?;
    let p_expanded = expand_operators_for_chc(&symbolic_ctx, &p_expr, false);
    let m_expr = ay_shared::get_operator_body(&symbolic_ctx, M_OP).ok()?;
    let m_expanded = expand_operators_for_chc(&symbolic_ctx, &m_expr, false);

    let vars = ay_shared::collect_state_vars(&module, &symbolic_ctx);
    let var_sorts =
        ay_shared::infer_var_sorts(&vars, &init_expanded, &config.invariants, &symbolic_ctx);
    let var_names: Vec<String> = vars.iter().map(|v| v.to_string()).collect();
    let funcsym_vars: std::collections::HashSet<String> =
        funcsym_domains(&var_sorts).into_keys().collect();
    let enabled = enabled_of_next(&next_expanded, &var_names, &funcsym_vars)?;

    Some(LiveInputs {
        var_sorts,
        init: init_expanded,
        next: next_expanded,
        j: j_expanded,
        p: p_expanded,
        m: m_expanded,
        enabled,
    })
}

/// Leg D part-2 (NO-SOLVE): re-translate one SMT obligation into a FRESH
/// translator and render each asserted term canonically (variables by NAME,
/// store-independent). NO `check_sat` is ever called — this binds the embedded
/// proof to the obligation TY recognizes via TY's TRANSLATOR, not its solver.
/// Returns the canonical S-expr strings (one per asserted term), or `None` if
/// re-translation fails (the verifier then treats the obligation as
/// inconclusive). Mirrors the producer's `make_translator(_, 1)` bound so the
/// step-1 indices match.
pub(crate) fn retranslate_obligation_canonical(
    ob: SmtObligation,
    inputs: &ObligationInputs,
) -> Option<Vec<String>> {
    let not_j = negate_normalized(&inputs.j);
    let not_safety = negate_normalized(&inputs.safety);
    let mut t = make_translator(&inputs.var_sorts, 1).ok()?;
    for (var_name, sort) in &inputs.var_sorts {
        t.declare_var(var_name, sort.clone()).ok()?;
    }
    let terms = build_smt_obligation(
        &mut t,
        ob,
        &inputs.init,
        &inputs.next,
        &not_safety,
        &inputs.j,
        &not_j,
    )
    .ok()?;
    // See retranslate_all_n_obligation_canonical: append the side-asserted
    // `\div`/`%` linearization constraints (empty otherwise) so the reconstructed
    // assertion set matches the proof bundle's full solver stack.
    let mut rendered = t.render_terms_canonical(&terms);
    rendered.extend(t.aux_asserted_canonical());
    Some(rendered)
}

// ===========================================================================
// ALL-N (parametric) obligations: a scalar CONSTANT is kept SYMBOLIC and declared
// as a RIGID constant (the SAME SMT term across steps), so the obligations range
// over it as a free variable — a proof holds for ALL its values. `rigid_consts`
// are the symbolic constant names; they are declared INTO each fresh translator
// (after the state vars) before the obligation is built. No `N' = N` equality is
// asserted (rigidity is structural), keeping consecution in the strict
// single-equality Farkas fragment.
// ===========================================================================

/// F1 feature 1 — ∃k-SKOLEMIZATION in antecedents. Rewrite each POSITIVE
/// `\E k \in lo..hi : body` (with a SYMBOLIC range that the bounded translator
/// cannot enumerate) into `lo<=k_sk /\ k_sk<=hi /\ body[k := k_sk]` for a FRESH
/// rigid Int constant `k_sk`, with `natMin`/`natMax` in the bounds eliminated
/// ([`leq_upper_bound`]/[`leq_lower_bound`]). Returns the rewritten predicate and
/// the fresh constant names (declared rigid before translation).
///
/// SOUNDNESS: this is Skolemization of an existential in POSITIVE (antecedent)
/// position — `Γ /\ (\E k φ(k))` is UNSAT iff `Γ /\ φ(k_sk)` is UNSAT for a fresh
/// free `k_sk`. It is applied ONLY to the `Next` used in the consecution
/// ANTECEDENT (asserted positively; never under a negation): the recursion
/// descends `And`/`Or` (polarity-preserving) and STOPS at any other node
/// (crucially `Not`), so a negated existential is left verbatim and never
/// Skolemized. Both the certify and the verify side rewrite the SAME re-derived
/// `Next` AST with the SAME deterministic left-to-right naming, so the render
/// binding matches term-for-term.
fn skolemize_next_antecedent(next: &Spanned<Expr>) -> (Spanned<Expr>, Vec<String>) {
    let mut counter = 0usize;
    let mut consts = Vec::new();
    let rewritten = skolemize_pos(next, &mut counter, &mut consts);
    (rewritten, consts)
}

/// Recurse through POSITIVE `And`/`Or` structure skolemizing bounded
/// existentials; STOP (leave verbatim) at every other node — never descend under
/// a `Not` (soundness rail).
fn skolemize_pos(
    e: &Spanned<Expr>,
    counter: &mut usize,
    consts: &mut Vec<String>,
) -> Spanned<Expr> {
    match &e.node {
        Expr::And(a, b) => Spanned::dummy(Expr::And(
            Box::new(skolemize_pos(a, counter, consts)),
            Box::new(skolemize_pos(b, counter, consts)),
        )),
        Expr::Or(a, b) => Spanned::dummy(Expr::Or(
            Box::new(skolemize_pos(a, counter, consts)),
            Box::new(skolemize_pos(b, counter, consts)),
        )),
        Expr::Exists(bounds, body) => {
            try_skolemize_exists(bounds, body, counter, consts).unwrap_or_else(|| e.clone())
        }
        // Any other node (including `Not`) is left verbatim: an un-skolemized
        // symbolic-range existential then declines at translation (fail-closed).
        _ => e.clone(),
    }
}

/// Attempt to Skolemize one `\E k \in lo..hi : body`. Returns `None` (leave
/// verbatim) for any out-of-shape existential: multiple bound vars, a non-range
/// or concrete-literal range (the latter is handled precisely by the existing
/// enumerator), `k` in a range bound, or a bound whose `natMin`/`natMax`
/// elimination declines. On success the counter/consts are advanced atomically
/// (they are untouched on the `None` paths, keeping the naming deterministic).
fn try_skolemize_exists(
    bounds: &[tla_core::ast::BoundVar],
    body: &Spanned<Expr>,
    counter: &mut usize,
    consts: &mut Vec<String>,
) -> Option<Spanned<Expr>> {
    if bounds.len() != 1 {
        return None;
    }
    let bound = &bounds[0];
    let domain = bound.domain.as_ref()?;
    let Expr::Range(lo, hi) = &domain.node else {
        return None;
    };
    // Concrete literal ranges are enumerated precisely upstream — only intercept
    // the SYMBOLIC ranges the bounded translator cannot handle.
    if matches!(lo.node, Expr::Int(_)) && matches!(hi.node, Expr::Int(_)) {
        return None;
    }
    let k = bound.name.node.as_str();
    if expr_mentions(&lo.node, k) || expr_mentions(&hi.node, k) {
        return None;
    }
    let idx = *counter;
    let sk = format!("__ty_skolem_{idx}");
    let sk_ref = Spanned::dummy(Expr::Ident(sk.clone(), NameId::INVALID));
    // Bound constraints (min/max eliminated). A decline here leaves the counter
    // and consts untouched (we have not committed `idx` yet).
    let lower = leq_lower_bound(lo, &sk_ref)?;
    let upper = leq_upper_bound(&sk_ref, hi)?;
    // Commit this skolem index now that both bounds linearized.
    *counter += 1;
    consts.push(sk);
    // Substitute k := k_sk in the body, then recurse for any nested existentials.
    let subbed = substitute_ident(body, k, &sk_ref);
    let subbed_sk = skolemize_pos(&subbed, counter, consts);
    let inner = Spanned::dummy(Expr::And(Box::new(upper), Box::new(subbed_sk)));
    Some(Spanned::dummy(Expr::And(Box::new(lower), Box::new(inner))))
}

/// Capture-avoiding substitution of the bare identifier `from` by `to`
/// throughout `body` (used to replace a skolemized bound variable).
fn substitute_ident(body: &Spanned<Expr>, from: &str, to: &Spanned<Expr>) -> Spanned<Expr> {
    let subs = std::collections::HashMap::from([(from, to)]);
    let mut sub = tla_core::SubstituteExpr {
        subs,
        span_policy: tla_core::SpanPolicy::Preserve,
    };
    tla_core::ExprFold::fold_expr(&mut sub, body.clone())
}

// ===========================================================================
// POINTWISE-∀ discipline for FUNCTION-STATE all-N (symbolic-domain functions).
//
// A `FunctionSym` state var `f : [lo..N -> T]` is encoded map-only as
// `(Array Int T)`. Its TYPE constraint (`f \in [lo..N -> T]`, or an explicit
// `\A i \in lo..N : P(f[i])`) is a UNIVERSAL over the symbolic domain, which the
// bounded translator cannot enumerate. Slice 1 replaces the native universal
// with a QF discipline applied IDENTICALLY on mint and verify:
//
//   * GOAL position (¬J' / ¬J / ¬Safety, asserted positively as the refutation
//     target): `¬(∀i∈D:P(f[i])) = ∃i∈D:¬P(f[i])` -> SKOLEMIZE to a fresh rigid
//     `i*` with `i*∈D ∧ ¬P(f[i*])`.
//   * HYPOTHESIS position (J / Init / ¬Enabled, asserted positively): a positive
//     universal cannot be skolemized -> INSTANTIATE at the finite index set `S`
//     collected from the obligation (the goal skolem, plus every index at which a
//     `FunctionSym` var is read/written in `Next`/goal), each instance GUARDED by
//     domain membership: `⋀_{s∈S} (s∈D ⟹ P(f[s]))`.
//
// SOUNDNESS: skolemizing a positive existential is equisat-preserving in the
// UNSAT direction; replacing a positive universal by a GUARDED finite subset of
// its instances only WEAKENS the hypothesis (each `s∈D ⟹ P(f[s])` is a logical
// consequence of `∀i∈D:P(f[i])` for ANY term `s`), so `UNSAT(instantiated) ⟹
// UNSAT(true)`. The instantiation set `S` is thus a COMPLETENESS lever only —
// too small an `S` leaves the query SAT and the lane declines, never a false
// accept (the array_ic3 over-approximation argument). Guarding every instance
// makes it sound for arbitrary (even out-of-domain) index terms.
// ===========================================================================

/// `(name -> (lo, hi_const, hi_offset))` for every `FunctionSym` state var.
type FuncDomains = std::collections::HashMap<String, (i64, String, i64)>;

fn funcsym_domains(var_sorts: &[(String, TlaSort)]) -> FuncDomains {
    var_sorts
        .iter()
        .filter_map(|(n, s)| match s {
            TlaSort::FunctionSym {
                domain_lo,
                domain_hi_const,
                domain_hi_offset,
                ..
            } => Some((
                n.clone(),
                (*domain_lo, domain_hi_const.clone(), *domain_hi_offset),
            )),
            _ => None,
        })
        .collect()
}

fn funcsym_present(var_sorts: &[(String, TlaSort)]) -> bool {
    var_sorts
        .iter()
        .any(|(_, s)| matches!(s, TlaSort::FunctionSym { .. }))
}

/// True iff `f` is (a possibly-primed) reference to a `FunctionSym` var.
fn funcexpr_base_is_funcsym(f: &Spanned<Expr>, fd: &FuncDomains) -> bool {
    match &f.node {
        Expr::Ident(n, _) | Expr::StateVar(n, ..) => fd.contains_key(n),
        Expr::Prime(inner) => funcexpr_base_is_funcsym(inner, fd),
        _ => false,
    }
}

/// Recognise `f \in [D -> R]` for a `FunctionSym` var `f`; return `(f, D, R)`.
fn as_funcsym_membership<'a>(
    expr: &'a Spanned<Expr>,
    fd: &FuncDomains,
) -> Option<(String, &'a Spanned<Expr>, &'a Spanned<Expr>)> {
    let Expr::In(elem, set) = &expr.node else {
        return None;
    };
    let name = match &elem.node {
        Expr::Ident(n, _) | Expr::StateVar(n, ..) if fd.contains_key(n) => n.clone(),
        _ => return None,
    };
    let Expr::FuncSet(domain, range) = &set.node else {
        return None;
    };
    Some((name, domain, range))
}

/// Recognise `\A i \in D : body` with a single, un-patterned bound var whose
/// domain is a SYMBOLIC range (upper bound not a literal), so the bounded
/// translator cannot enumerate it; return `(bound_name, D, body)`. Concrete
/// ranges are left to the existing enumerator.
fn as_symbolic_range_forall<'a>(
    expr: &'a Spanned<Expr>,
) -> Option<(&'a str, &'a Spanned<Expr>, &'a Spanned<Expr>)> {
    let Expr::Forall(bounds, body) = &expr.node else {
        return None;
    };
    if bounds.len() != 1 {
        return None;
    }
    let b = &bounds[0];
    if b.pattern.is_some() {
        return None;
    }
    let dom = b.domain.as_ref()?;
    // Symbolic domain: a range whose upper bound is not a plain literal, or any
    // non-range domain built from a symbolic range (SetMinus etc.). We only
    // intercept the range case here; a concrete `1..3` is left verbatim.
    if let Expr::Range(lo, hi) = &dom.node {
        if matches!(lo.node, Expr::Int(_)) && matches!(hi.node, Expr::Int(_)) {
            return None;
        }
    }
    Some((b.name.node.as_str(), dom, body))
}

/// Build a fresh pointwise skolem index `__ty_pw_{n}` (Ident) and record it.
fn fresh_pw_index(counter: &mut usize, consts: &mut Vec<String>) -> Spanned<Expr> {
    let name = format!("__ty_pw_{}", *counter);
    *counter += 1;
    consts.push(name.clone());
    Spanned::dummy(Expr::Ident(name, NameId::INVALID))
}

fn mk_and(a: Spanned<Expr>, b: Spanned<Expr>) -> Spanned<Expr> {
    Spanned::dummy(Expr::And(Box::new(a), Box::new(b)))
}
fn mk_not(a: Spanned<Expr>) -> Spanned<Expr> {
    Spanned::dummy(Expr::Not(Box::new(a)))
}
fn mk_in(elem: Spanned<Expr>, set: Spanned<Expr>) -> Spanned<Expr> {
    Spanned::dummy(Expr::In(Box::new(elem), Box::new(set)))
}
fn mk_implies(a: Spanned<Expr>, b: Spanned<Expr>) -> Spanned<Expr> {
    Spanned::dummy(Expr::Implies(Box::new(a), Box::new(b)))
}
fn mk_func_apply(func: &str, arg: Spanned<Expr>) -> Spanned<Expr> {
    let f = Spanned::dummy(Expr::Ident(func.to_string(), NameId::INVALID));
    Spanned::dummy(Expr::FuncApply(Box::new(f), Box::new(arg)))
}

/// Rewrite a NEGATED (goal) predicate: skolemize every negated pointwise
/// universal — `¬(f∈[D->R])` and `¬(∀i∈D:body)` over a symbolic domain — to a
/// fresh rigid index with `i*∈D ∧ ¬body(i*)`. Descends `And`/`Or` (both sound
/// in a positively-asserted goal) and leaves every other node verbatim.
fn skolemize_pointwise_goal(
    e: &Spanned<Expr>,
    fd: &FuncDomains,
    counter: &mut usize,
    consts: &mut Vec<String>,
) -> Spanned<Expr> {
    match &e.node {
        Expr::And(a, b) => mk_and(
            skolemize_pointwise_goal(a, fd, counter, consts),
            skolemize_pointwise_goal(b, fd, counter, consts),
        ),
        Expr::Or(a, b) => Spanned::dummy(Expr::Or(
            Box::new(skolemize_pointwise_goal(a, fd, counter, consts)),
            Box::new(skolemize_pointwise_goal(b, fd, counter, consts)),
        )),
        Expr::Not(inner) => {
            if let Some((f, dom, range)) = as_funcsym_membership(inner, fd) {
                let sk = fresh_pw_index(counter, consts);
                let member = mk_in(mk_func_apply(&f, sk.clone()), range.clone());
                return mk_and(mk_in(sk, dom.clone()), mk_not(member));
            }
            if let Some((bv, dom, body)) = as_symbolic_range_forall(inner) {
                let sk = fresh_pw_index(counter, consts);
                let subbed = substitute_ident(body, bv, &sk);
                return mk_and(mk_in(sk, dom.clone()), mk_not(subbed));
            }
            e.clone()
        }
        _ => e.clone(),
    }
}

/// Conjoin the GUARDED instances `⋀_{s∈S} (s∈D ⟹ body(s))`. Empty `S` yields
/// `TRUE` (a dropped, hence weakened, hypothesis — sound, declines if needed).
fn conjoin_guarded_instances(
    dom: &Spanned<Expr>,
    index_terms: &[Spanned<Expr>],
    mut body_at: impl FnMut(&Spanned<Expr>) -> Spanned<Expr>,
) -> Spanned<Expr> {
    let mut acc: Option<Spanned<Expr>> = None;
    for s in index_terms {
        let inst = mk_implies(mk_in(s.clone(), dom.clone()), body_at(s));
        acc = Some(match acc {
            None => inst,
            Some(prev) => mk_and(prev, inst),
        });
    }
    acc.unwrap_or_else(|| Spanned::dummy(Expr::Bool(true)))
}

/// Rewrite a POSITIVE (hypothesis) predicate: replace every pointwise universal
/// over a symbolic domain by its guarded instances at `index_terms`. Descends
/// `And` only (a conjunctive hypothesis); other nodes (incl. `Or`) are left
/// verbatim — a symbolic-domain universal under an `Or` then declines at
/// translation (fail-closed).
fn instantiate_pointwise_hypothesis(
    e: &Spanned<Expr>,
    fd: &FuncDomains,
    index_terms: &[Spanned<Expr>],
) -> Spanned<Expr> {
    match &e.node {
        Expr::And(a, b) => mk_and(
            instantiate_pointwise_hypothesis(a, fd, index_terms),
            instantiate_pointwise_hypothesis(b, fd, index_terms),
        ),
        _ => {
            if let Some((f, dom, range)) = as_funcsym_membership(e, fd) {
                return conjoin_guarded_instances(dom, index_terms, |s| {
                    mk_in(mk_func_apply(&f, s.clone()), range.clone())
                });
            }
            if let Some((bv, dom, body)) = as_symbolic_range_forall(e) {
                return conjoin_guarded_instances(dom, index_terms, |s| {
                    substitute_ident(body, bv, s)
                });
            }
            e.clone()
        }
    }
}

/// Collect the GROUND indices at which a `FunctionSym` var is read (`f[e]`) or
/// written (`[f EXCEPT ![e] = _]`) in `e`. Does NOT descend into quantifier /
/// function-definition BODIES (their reads reference a bound variable, not a
/// ground index) — only their domains, which are ground. Deduplicated by
/// span-insensitive structural equality.
fn collect_func_indices(e: &Spanned<Expr>, fd: &FuncDomains, out: &mut Vec<Spanned<Expr>>) {
    let push = |out: &mut Vec<Spanned<Expr>>, idx: &Spanned<Expr>| {
        if !out.iter().any(|existing| eq_ignore_span(existing, idx)) {
            out.push(idx.clone());
        }
    };
    match &e.node {
        Expr::FuncApply(f, arg) => {
            if funcexpr_base_is_funcsym(f, fd) {
                push(out, arg);
            }
            collect_func_indices(f, fd, out);
            collect_func_indices(arg, fd, out);
        }
        Expr::Except(base, specs) => {
            let base_is_fsym = funcexpr_base_is_funcsym(base, fd);
            for spec in specs {
                for pe in &spec.path {
                    if let tla_core::ast::ExceptPathElement::Index(idx) = pe {
                        if base_is_fsym {
                            push(out, idx);
                        }
                        collect_func_indices(idx, fd, out);
                    }
                }
                collect_func_indices(&spec.value, fd, out);
            }
            collect_func_indices(base, fd, out);
        }
        Expr::Domain(f) => collect_func_indices(f, fd, out),
        Expr::FuncSet(d, r) => {
            collect_func_indices(d, fd, out);
            collect_func_indices(r, fd, out);
        }
        // Binders: collect from the DOMAIN(s) only, never the body.
        Expr::Forall(bounds, _) | Expr::Exists(bounds, _) | Expr::FuncDef(bounds, _) => {
            for b in bounds {
                if let Some(d) = &b.domain {
                    collect_func_indices(d, fd, out);
                }
            }
        }
        _ => walk_immediate_children(&e.node, &mut |c| collect_func_indices(c, fd, out)),
    }
}

/// The transformed (QF) pieces of one pointwise all-N obligation, plus the fresh
/// rigid skolem-const names to declare. Deterministic in `(ob, inputs)`, so mint
/// and verify produce byte-identical results.
struct PointwisePieces {
    init: Spanned<Expr>,
    next: Spanned<Expr>,
    not_safety: Spanned<Expr>,
    j: Spanned<Expr>,
    not_j: Spanned<Expr>,
    skolem_consts: Vec<String>,
}

fn transform_all_n_pointwise(ob: SmtObligation, inputs: &ObligationInputs) -> PointwisePieces {
    let fd = funcsym_domains(&inputs.var_sorts);
    let mut counter = 0usize;
    let mut consts: Vec<String> = Vec::new();

    let raw_not_j = negate_normalized(&inputs.j);
    let raw_not_safety = negate_normalized(&inputs.safety);

    // Consecution is the only obligation that translates `Next`; skolemize its
    // symbolic-range existentials (the McCarthy action selectors).
    let (next_sk, next_consts) = if matches!(ob, SmtObligation::Consecution) {
        skolemize_next_antecedent(&inputs.next)
    } else {
        (inputs.next.clone(), Vec::new())
    };

    // GOAL skolemization (position depends on obligation).
    let (not_j, not_safety) = match ob {
        SmtObligation::Initiation | SmtObligation::Consecution => (
            skolemize_pointwise_goal(&raw_not_j, &fd, &mut counter, &mut consts),
            raw_not_safety,
        ),
        SmtObligation::Safety => (
            raw_not_j,
            skolemize_pointwise_goal(&raw_not_safety, &fd, &mut counter, &mut consts),
        ),
    };

    // Collect the hypothesis-instantiation index set S from the transformed Next
    // (consecution) and the transformed goal.
    let mut indices: Vec<Spanned<Expr>> = Vec::new();
    match ob {
        SmtObligation::Initiation => collect_func_indices(&not_j, &fd, &mut indices),
        SmtObligation::Consecution => {
            collect_func_indices(&next_sk, &fd, &mut indices);
            collect_func_indices(&not_j, &fd, &mut indices);
        }
        SmtObligation::Safety => collect_func_indices(&not_safety, &fd, &mut indices),
    }

    // HYPOTHESIS instantiation.
    let init = match ob {
        SmtObligation::Initiation => {
            instantiate_pointwise_hypothesis(&inputs.init, &fd, &indices)
        }
        _ => inputs.init.clone(),
    };
    let j = match ob {
        SmtObligation::Consecution | SmtObligation::Safety => {
            instantiate_pointwise_hypothesis(&inputs.j, &fd, &indices)
        }
        SmtObligation::Initiation => inputs.j.clone(),
    };

    let mut skolem_consts = consts;
    skolem_consts.extend(next_consts);

    PointwisePieces {
        init,
        next: next_sk,
        not_safety,
        j,
        not_j,
        skolem_consts,
    }
}

// ===========================================================================
// BLOCKER-2 CLEAN CLOSE — per-branch McCarthy reduction of the CONSECUTION.
//
// The array (store/select) encoding of `f' = [f EXCEPT ![p*]=v]` forces AY down
// its array→LIA rescue, whose refutation carries trust steps `check_proof_strict`
// refuses (named-store ROW + the case-split master conflict). The pointwise
// discipline ALREADY confines every read of the next-state function to the finite
// skolem set S = {goal index i*} ∪ {write indices p*}, so the store can be
// McCarthy-resolved AWAY at a DECIDED index equality — turning the one
// case-splitting obligation into a CONJUNCTION of per-equality-partition BRANCH
// obligations, each array-free AND case-split-free (single-Farkas) and hence
// STRICT-CHECKABLE with ZERO ay/checker changes.
//
// SOUNDNESS: the partition {i*=p*, i*≠p*} is EXHAUSTIVE, so
// `⋀_branches UNSAT ⟹ consecution UNSAT`. Each branch resolves `f'[i*]` by the
// store DEFINITION at a decided index (`f'[i*]=v` when i*=p*, `f'[i*]=f[i*]` when
// i*≠p* — both exact McCarthy), keeps the write-index bounds, and reuses the same
// guarded-instantiated hypothesis. A shape outside the recognised single-point
// McCarthy fragment returns `None` → the array path runs and declines fail-closed;
// the twins are preserved (a write outside the codomain leaves branch i*=p* SAT →
// decline; whole-function `f=g` bait has no pointwise goal read → None → decline).

/// Recognise a single-point store `f' = [f EXCEPT ![idx] = val]` (FunctionSym
/// `f`); return `(idx, val)`.
fn as_single_point_store(
    lhs: &Spanned<Expr>,
    rhs: &Spanned<Expr>,
    fd: &FuncDomains,
) -> Option<(Spanned<Expr>, Spanned<Expr>)> {
    if !matches!(lhs.node, Expr::Prime(_)) || !funcexpr_base_is_funcsym(lhs, fd) {
        return None;
    }
    let Expr::Except(base, specs) = &rhs.node else {
        return None;
    };
    if !funcexpr_base_is_funcsym(base, fd) || specs.len() != 1 {
        return None;
    }
    let spec = &specs[0];
    if spec.path.len() != 1 {
        return None;
    }
    let tla_core::ast::ExceptPathElement::Index(idx) = &spec.path[0] else {
        return None;
    };
    Some((idx.clone(), spec.value.clone()))
}

/// Replace every recognised single-point store equality by `TRUE` (keeping the
/// surrounding `∧`-structure — i.e. the write-index bounds), recording each
/// captured `(idx, val)`. Descends `And` only (Next is a conjunctive action here).
fn strip_and_capture_store(
    e: &Spanned<Expr>,
    fd: &FuncDomains,
    found: &mut Vec<(Spanned<Expr>, Spanned<Expr>)>,
) -> Spanned<Expr> {
    match &e.node {
        Expr::And(a, b) => mk_and(
            strip_and_capture_store(a, fd, found),
            strip_and_capture_store(b, fd, found),
        ),
        Expr::Eq(lhs, rhs) => {
            if let Some(store) = as_single_point_store(lhs, rhs, fd) {
                found.push(store);
                Spanned::dummy(Expr::Bool(true))
            } else {
                e.clone()
            }
        }
        _ => e.clone(),
    }
}

/// Bare base name of a (possibly-primed) FunctionSym reference.
fn funcsym_base_name(f: &Spanned<Expr>) -> Option<String> {
    match &f.node {
        Expr::Ident(n, _) | Expr::StateVar(n, ..) => Some(n.clone()),
        Expr::Prime(inner) => funcsym_base_name(inner),
        _ => None,
    }
}

/// Match the skolemized pointwise goal `i*∈D ∧ ¬(f[i*] ∈ R)`; return
/// `(f, i*, D, R)`. Any other shape (e.g. an extensionality-bait `f=g` goal) → None.
fn extract_pointwise_goal_read(
    not_j: &Spanned<Expr>,
    fd: &FuncDomains,
) -> Option<(String, Spanned<Expr>, Spanned<Expr>, Spanned<Expr>)> {
    let Expr::And(a, b) = &not_j.node else {
        return None;
    };
    let Expr::In(idx, dom) = &a.node else {
        return None;
    };
    let Expr::Not(inner) = &b.node else {
        return None;
    };
    let Expr::In(read, range) = &inner.node else {
        return None;
    };
    let Expr::FuncApply(f, arg) = &read.node else {
        return None;
    };
    if !funcexpr_base_is_funcsym(f, fd) || !eq_ignore_span(arg, idx) {
        return None;
    }
    let name = funcsym_base_name(f)?;
    Some((name, (**idx).clone(), (**dom).clone(), (**range).clone()))
}

fn conj_all(parts: Vec<Spanned<Expr>>) -> Spanned<Expr> {
    let mut it = parts.into_iter();
    let mut acc = it.next().unwrap_or_else(|| Spanned::dummy(Expr::Bool(true)));
    for p in it {
        acc = mk_and(acc, p);
    }
    acc
}

/// A select index is ATOMIC when it is a bare var/const. A select at an atomic
/// index rebuilds checkably; a select at a COMPUTED (arithmetic) index — e.g.
/// `f[p-1]` — materializes its ASSUME leaf as an un-rebuildable TRUST step
/// (slice-3). Atomizing the index (below) restores checkability.
fn is_atomic_index(e: &Expr) -> bool {
    matches!(e, Expr::Ident(..) | Expr::StateVar(..) | Expr::Int(_))
}

/// Fold that ATOMIZES computed FunctionSym read indices: rewrites each `f[idx]`
/// with a non-atomic `idx` (e.g. `p-1`) to `f[k]` for a fresh rigid const `k`,
/// recording `(idx, k)` in `atoms` (deduped by structural index equality, so the
/// SAME computed index maps to the SAME `k` across all pieces — determinism).
/// The caller conjoins the defining equalities `k = idx`. SOUND BY CONSTRUCTION:
/// `k` is asserted equal to `idx`, so `f[k] ≡ f[idx]` — no range/soundness
/// analysis, only a form change that makes the select's index atomic (checkable).
struct AtomizeReads<'a> {
    fd: &'a FuncDomains,
    atoms: Vec<(Spanned<Expr>, String)>,
}

impl tla_core::ExprFold for AtomizeReads<'_> {
    fn fold_expr(&mut self, e: Spanned<Expr>) -> Spanned<Expr> {
        match &e.node {
            Expr::FuncApply(f, arg)
                if funcexpr_base_is_funcsym(f, self.fd) && !is_atomic_index(&arg.node) =>
            {
                let k = self
                    .atoms
                    .iter()
                    .find(|(idx, _)| eq_ignore_span(idx, arg))
                    .map(|(_, k)| k.clone())
                    .unwrap_or_else(|| {
                        let k = format!("__ty_atom_{}", self.atoms.len());
                        self.atoms.push(((**arg).clone(), k.clone()));
                        k
                    });
                let k_ref = Spanned::dummy(Expr::Ident(k, NameId::INVALID));
                Spanned::dummy(Expr::FuncApply(f.clone(), Box::new(k_ref)))
            }
            _ => Spanned {
                node: self.fold_expr_inner(e.node),
                span: e.span,
            },
        }
    }
}

/// Collect the write-skolem `p`'s integer lower bound and upper-bound expression
/// from the `∧`-structured `next_bounds` (`lo ≤ p` and `p ≤ hi`).
fn collect_leq_bounds(
    e: &Spanned<Expr>,
    p: &str,
    lo: &mut Option<num_bigint::BigInt>,
    hi: &mut Option<Spanned<Expr>>,
) {
    match &e.node {
        Expr::And(a, b) => {
            collect_leq_bounds(a, p, lo, hi);
            collect_leq_bounds(b, p, lo, hi);
        }
        Expr::Leq(l, r) => {
            if let (Expr::Int(n), Expr::Ident(pn, _)) = (&l.node, &r.node) {
                if pn == p {
                    *lo = Some(n.clone());
                }
            }
            if let Expr::Ident(pn, _) = &l.node {
                if pn == p {
                    *hi = Some((**r).clone());
                }
            }
        }
        _ => {}
    }
}

/// SLICE-3 PART 2 soundness gate: is the computed value-read index `idx` (e.g.
/// `p - c`) PROVABLY within `dom`, given the write skolem `p`'s range in
/// `next_bounds`? If so, `k ∈ dom` is ENTAILED and may be asserted directly (so
/// the guarded hypothesis at `k` discharges checkably). Handles the ring shape
/// `p - c` (c ≥ 0) with `lo ≤ p ≤ hi` where `hi` is STRUCTURALLY the domain's
/// upper (the shared symbolic `N`): then `k = p-c ∈ (lo-c)..(hi-c) ⊆ dlo..hi`
/// iff `lo-c ≥ dlo` (and `c ≥ 0`, `hi-c ≤ hi`). FAIL-CLOSED on any other shape —
/// the branch then keeps the guarded form and declines (never a false accept).
fn computed_index_in_domain(
    idx: &Spanned<Expr>,
    next_bounds: &Spanned<Expr>,
    dom: &Spanned<Expr>,
) -> bool {
    let Expr::Sub(base, off) = &idx.node else {
        return false;
    };
    let Expr::Ident(p, _) = &base.node else {
        return false;
    };
    let Expr::Int(c) = &off.node else {
        return false;
    };
    if *c < num_bigint::BigInt::from(0) {
        return false;
    }
    let Expr::Range(dlo_e, dhi_e) = &dom.node else {
        return false;
    };
    let Expr::Int(dlo) = &dlo_e.node else {
        return false;
    };
    let mut lo: Option<num_bigint::BigInt> = None;
    let mut hi: Option<Spanned<Expr>> = None;
    collect_leq_bounds(next_bounds, p, &mut lo, &mut hi);
    let (Some(lo), Some(hi)) = (lo, hi) else {
        return false;
    };
    // hi (the skolem's upper) must be STRUCTURALLY the domain's upper — then
    // `hi - c ≤ hi = dhi` holds for any c ≥ 0 without symbolic arithmetic.
    if !eq_ignore_span(&hi, dhi_e) {
        return false;
    }
    (lo - c) >= *dlo
}

/// Build the per-equality-partition BRANCH obligations for the consecution. Each
/// branch is ONE step-0 conjunction asserted UNSAT (asserted as a single term so
/// a ground-false goal collapses cleanly — trust-free — and assume-coverage is
/// the trivial `{branch} ⊆ {branch}`). The consecution is proved iff EVERY branch
/// is UNSAT + strict. `None` falls back to the array path (declines fail-closed).
/// DETERMINISTIC in `(inputs)` so mint and verify build the same branches.
fn mccarthy_consecution_branches(
    inputs: &ObligationInputs,
    fd: &FuncDomains,
) -> Option<(Vec<Spanned<Expr>>, Vec<String>)> {
    let pt = transform_all_n_pointwise(SmtObligation::Consecution, inputs);
    let mut stores = Vec::new();
    let next_bounds = strip_and_capture_store(&pt.next, fd, &mut stores);
    // Slice 1: exactly one single-point McCarthy update. (0 or ≥2 → decline.)
    if stores.len() != 1 {
        return None;
    }
    let (write_idx, write_val) = stores.into_iter().next().unwrap();
    let (_goal_f, goal_idx, goal_dom, goal_range) = extract_pointwise_goal_read(&pt.not_j, fd)?;

    // Slice-3: ATOMIZE computed FunctionSym read indices (e.g. the read-valued
    // write `f[p-1]`) across ALL pieces with a shared map, so `f[p-1]` becomes
    // `f[k]` (atomic) + a defining equality `k = p-1`. Sound by construction; the
    // select is now atomic-index so its ASSUME leaf rebuilds checkably. Order
    // (write_val → not_j → j) is fixed ⇒ deterministic const naming ⇒ verify's
    // re-derivation is identical ⇒ the render-binding matches term-for-term.
    let mut az = AtomizeReads { fd, atoms: Vec::new() };
    let write_val = tla_core::ExprFold::fold_expr(&mut az, write_val);
    let not_j = tla_core::ExprFold::fold_expr(&mut az, pt.not_j.clone());
    let j = tla_core::ExprFold::fold_expr(&mut az, pt.j.clone());
    let defining_eqs: Vec<Spanned<Expr>> = az
        .atoms
        .iter()
        .map(|(idx, k)| {
            Spanned::dummy(Expr::Eq(
                Box::new(Spanned::dummy(Expr::Ident(k.clone(), NameId::INVALID))),
                Box::new(idx.clone()),
            ))
        })
        .collect();
    let mut skolem_consts = pt.skolem_consts;
    skolem_consts.extend(az.atoms.iter().map(|(_, k)| k.clone()));

    // Slice-3 part 2: for each atomized computed index whose value is PROVABLY in
    // the domain (`computed_index_in_domain`), assert `k ∈ D` directly. `k∈D` is
    // ENTAILED by the defining eq + the write-skolem bounds, so asserting it adds
    // no constraint (sound) but makes the guarded hypothesis at `k` discharge
    // checkably (its guard is now directly asserted, exactly like slice-2's `p∈D`).
    // Indices whose in-domain-ness is NOT statically provable get NO such assert,
    // so their guard stays a trust leaf and the branch declines (fail-closed).
    // Assert the guard in its ORIGINAL computed form `idx ∈ D` (== the guarded
    // hypothesis's guard, since atomization rewrites only the READ `f[idx]→f[k]`,
    // not the guard `In(idx,D)`), so the guard discharges by a direct match (like
    // slice-2's `p∈D`) rather than a trust-materialized Farkas+MP.
    let in_domain: Vec<Spanned<Expr>> = az
        .atoms
        .iter()
        .filter(|(idx, _)| computed_index_in_domain(idx, &next_bounds, &goal_dom))
        .map(|(idx, _)| mk_in(idx.clone(), goal_dom.clone()))
        .collect();

    let eq = Spanned::dummy(Expr::Eq(
        Box::new(goal_idx.clone()),
        Box::new(write_idx.clone()),
    ));
    // Branch A — i* = p*: the store writes the goal cell, so f'[i*] = v; the goal
    // `¬(f'[i*] ∈ R)` becomes `¬(v ∈ R)` (v atomized if it was a computed read).
    let goal_a = mk_and(
        mk_in(goal_idx.clone(), goal_dom.clone()),
        mk_not(mk_in(write_val, goal_range.clone())),
    );
    let mut a_parts = vec![next_bounds.clone(), eq.clone(), goal_a, j.clone()];
    a_parts.extend(defining_eqs.iter().cloned());
    a_parts.extend(in_domain.iter().cloned());
    let branch_a = conj_all(a_parts);

    // Branch B — i* ≠ p*: the store is transparent at i*, so f'[i*] = f[i*] and
    // the goal `¬(f[i*] ∈ R)` (== `not_j`) reads the step-0 array directly.
    let mut b_parts = vec![next_bounds, mk_not(eq), not_j, j];
    b_parts.extend(defining_eqs);
    b_parts.extend(in_domain);
    let branch_b = conj_all(b_parts);

    Some((vec![branch_a, branch_b], skolem_consts))
}

/// Re-translate each consecution branch to its CANONICAL rendered assertions
/// (no solve), for the verifier's per-branch render-binding. `None` when the spec
/// is not the recognised single-point McCarthy shape (mint used the array path,
/// whose single-bundle re-translation the caller handles instead). Mirrors the
/// mint's assertion structure in [`discharge_consecution_branches`] EXACTLY
/// (each piece asserted separately), so the render binding matches term-for-term.
pub(crate) fn retranslate_consecution_branches_canonical(
    inputs: &ObligationInputs,
    rigid_consts: &[String],
) -> Option<Vec<Vec<String>>> {
    let fd = funcsym_domains(&inputs.var_sorts);
    let (branches, sk_consts) = mccarthy_consecution_branches(inputs, &fd)?;
    let mut out = Vec::with_capacity(branches.len());
    for branch in &branches {
        let mut t = make_translator(&inputs.var_sorts, 1).ok()?;
        for (var_name, sort) in &inputs.var_sorts {
            t.declare_var(var_name, sort.clone()).ok()?;
        }
        for c in rigid_consts.iter().chain(sk_consts.iter()) {
            t.declare_rigid_const(c, TlaSort::Int).ok()?;
        }
        let term = t.translate_safety_at_step(branch, 0).ok()?;
        out.push(t.render_terms_canonical(&[term]));
    }
    Some(out)
}

/// Discharge the McCarthy branch obligations and fold them into ONE
/// `consecution` proof (the cert model requires exactly four obligations): UNSAT
/// iff EVERY branch is UNSAT, strict iff EVERY branch strict-verified. The branch
/// bundles are carried as a JSON array for offline re-check.
fn discharge_consecution_branches(
    inputs: &ObligationInputs,
    rigid_consts: &[String],
    sk_consts: &[String],
    branches: &[Spanned<Expr>],
    timeout: Option<Duration>,
) -> Result<ObligationProof, BmcError> {
    let mut all_unsat = true;
    let mut all_strict = true;
    let mut bundles: Vec<String> = Vec::new();
    for branch in branches {
        let p = scratch_check_unsat_with_proof("consecution", &inputs.var_sorts, 1, timeout, |t| {
            for c in rigid_consts.iter().chain(sk_consts.iter()) {
                t.declare_rigid_const(c, TlaSort::Int)?;
            }
            let term = t.translate_safety_at_step(branch, 0)?;
            t.assert(term);
            Ok(())
        })?;
        all_unsat &= p.unsat;
        all_strict &= p.strict_verified;
        if let Some(bj) = p.bundle_json {
            bundles.push(bj);
        }
    }
    Ok(ObligationProof {
        name: "consecution",
        unsat: all_unsat,
        strict_verified: all_unsat && all_strict,
        clean_supported: all_unsat && all_strict,
        alethe: format!(
            "mccarthy-branch-reduced consecution: {} branches, all strict-verified={}",
            branches.len(),
            all_strict
        ),
        lrat_present: false,
        bundle_json: if bundles.is_empty() {
            None
        } else {
            serde_json::to_string(&bundles).ok()
        },
    })
}

/// The pointwise `deadlock_freedom` obligation `J@0 ∧ ¬Enabled@0` UNSAT, with the
/// positive universals in BOTH `J` and `¬Enabled` guard-instantiated at the
/// indices they mention (a hypothesis weakening — sound; SAT ⇒ honest decline).
fn transform_all_n_pointwise_deadlock(inputs: &ObligationInputs) -> (Spanned<Expr>, Spanned<Expr>, Vec<String>) {
    let fd = funcsym_domains(&inputs.var_sorts);
    let not_enabled = negate_normalized(&inputs.enabled);
    let mut indices: Vec<Spanned<Expr>> = Vec::new();
    collect_func_indices(&not_enabled, &fd, &mut indices);
    collect_func_indices(&inputs.j, &fd, &mut indices);
    let j = instantiate_pointwise_hypothesis(&inputs.j, &fd, &indices);
    let ne = instantiate_pointwise_hypothesis(&not_enabled, &fd, &indices);
    // Blocker-3: conjoin the spec's symbolic-constant ASSUME (e.g. `N ≥ 1`) into
    // the hypothesis so `assume ∧ J ∧ ¬Enabled` is UNSAT — vacuously at an
    // excluded `N=0` (empty domain, `∃p∈1..N` unenabled), and by `Enabled` for
    // the admitted `N`. `TRUE` when the spec makes no assumption (byte-unchanged).
    let j = match &inputs.assume.node {
        Expr::Bool(true) => j,
        _ => mk_and(inputs.assume.clone(), j),
    };
    (j, ne, Vec::new())
}

/// Pointwise counterpart of [`discharge_all_n_obligations_with_proofs`] for
/// specs with a `FunctionSym` state var. Applies the goal-skolemize / guarded
/// hypothesis-instantiation discipline per obligation, then reuses
/// [`build_smt_obligation`] on the resulting QF pieces.
fn discharge_all_n_pointwise(
    inputs: &ObligationInputs,
    rigid_consts: &[String],
    timeout: Option<Duration>,
) -> Result<Vec<ObligationProof>, BmcError> {
    let mk = |ob: SmtObligation| -> Result<ObligationProof, BmcError> {
        let pt = transform_all_n_pointwise(ob, inputs);
        scratch_check_unsat_with_proof(ob.name(), &inputs.var_sorts, 1, timeout, |t| {
            for c in rigid_consts.iter().chain(pt.skolem_consts.iter()) {
                t.declare_rigid_const(c, TlaSort::Int)?;
            }
            build_smt_obligation(t, ob, &pt.init, &pt.next, &pt.not_safety, &pt.j, &pt.not_j)
                .map(|_| ())
        })
    };
    let initiation = mk(SmtObligation::Initiation)?;
    // Blocker-2 close: discharge the consecution via per-branch McCarthy reduction
    // (array-free, strict-checkable) when the single-point McCarthy shape is
    // recognised; else fall back to the array path (declines fail-closed).
    let consecution = {
        let fd = funcsym_domains(&inputs.var_sorts);
        match mccarthy_consecution_branches(inputs, &fd) {
            Some((branches, sk_consts)) => {
                discharge_consecution_branches(inputs, rigid_consts, &sk_consts, &branches, timeout)?
            }
            None => mk(SmtObligation::Consecution)?,
        }
    };
    let safety_ob = mk(SmtObligation::Safety)?;

    let deadlock_freedom = if matches!(inputs.enabled.node, Expr::Bool(true)) {
        ObligationProof {
            name: "deadlock_freedom",
            unsat: true,
            strict_verified: true,
            clean_supported: true,
            alethe: "structural: unguarded total Next => Enabled(Next) == TRUE".to_string(),
            lrat_present: false,
            bundle_json: None,
        }
    } else {
        let (j_t, ne_t, dl_consts) = transform_all_n_pointwise_deadlock(inputs);
        scratch_check_unsat_with_proof("deadlock_freedom", &inputs.var_sorts, 1, timeout, |t| {
            for c in rigid_consts.iter().chain(dl_consts.iter()) {
                t.declare_rigid_const(c, TlaSort::Int)?;
            }
            let j0 = t.translate_safety_at_step(&j_t, 0)?;
            t.assert(j0);
            let ne0 = t.translate_safety_at_step(&ne_t, 0)?;
            t.assert(ne0);
            Ok(())
        })?
    };
    Ok(vec![initiation, consecution, safety_ob, deadlock_freedom])
}

/// Pointwise counterpart of [`retranslate_all_n_obligation_canonical`]: re-derive
/// the SAME transformed obligation (identical skolemization + instantiation +
/// naming) WITHOUT solving and render canonically, so the verifier's render
/// binding matches the mint term-for-term.
fn retranslate_all_n_pointwise(
    ob: SmtObligation,
    inputs: &ObligationInputs,
    rigid_consts: &[String],
) -> Option<Vec<String>> {
    let pt = transform_all_n_pointwise(ob, inputs);
    let mut t = make_translator(&inputs.var_sorts, 1).ok()?;
    for (var_name, sort) in &inputs.var_sorts {
        t.declare_var(var_name, sort.clone()).ok()?;
    }
    for c in rigid_consts.iter().chain(pt.skolem_consts.iter()) {
        t.declare_rigid_const(c, TlaSort::Int).ok()?;
    }
    let terms = build_smt_obligation(
        &mut t,
        ob,
        &pt.init,
        &pt.next,
        &pt.not_safety,
        &pt.j,
        &pt.not_j,
    )
    .ok()?;
    Some(t.render_terms_canonical(&terms))
}

/// Pointwise counterpart of [`retranslate_deadlock_obligation_canonical`].
fn retranslate_all_n_pointwise_deadlock(
    inputs: &ObligationInputs,
    rigid_consts: &[String],
) -> Option<Vec<String>> {
    let (j_t, ne_t, dl_consts) = transform_all_n_pointwise_deadlock(inputs);
    let mut t = make_translator(&inputs.var_sorts, 1).ok()?;
    for (var_name, sort) in &inputs.var_sorts {
        t.declare_var(var_name, sort.clone()).ok()?;
    }
    for c in rigid_consts.iter().chain(dl_consts.iter()) {
        t.declare_rigid_const(c, TlaSort::Int).ok()?;
    }
    let j0 = t.translate_safety_at_step(&j_t, 0).ok()?;
    let ne0 = t.translate_safety_at_step(&ne_t, 0).ok()?;
    Some(t.render_terms_canonical(&[j0, ne0]))
}

/// Discharge the four inductive-safety obligations with the `rigid_consts` kept
/// symbolic (all-N). Mirrors [`discharge_obligations_with_proofs`] but declares
/// each rigid constant in every obligation's translator.
pub(crate) fn discharge_all_n_obligations_with_proofs(
    inputs: &ObligationInputs,
    rigid_consts: &[String],
    timeout: Option<Duration>,
) -> Result<Vec<ObligationProof>, BmcError> {
    // Function-state specs (a `FunctionSym` var) route through the pointwise-∀
    // discipline; every other spec keeps the scalar/finite path unchanged.
    if funcsym_present(&inputs.var_sorts) {
        return discharge_all_n_pointwise(inputs, rigid_consts, timeout);
    }
    // Conjoin the spec's symbolic ASSUME into every obligation's HYPOTHESIS
    // (`assume ∧ Init ⇒ J`, `assume ∧ J ∧ Next ⇒ J'`, `assume ∧ J ⇒ Safety`,
    // `assume ∧ J ⇒ Enabled`). A spec-level `ASSUME P(N)` constrains the rigid
    // constant globally, so it is sound in EVERY obligation's antecedent (and
    // NECESSARY when Init/J reference the constant unconditionally — e.g. a record
    // `Init == r = [a |-> 0, b |-> N]` needs `N >= 0` to entail `r.b >= 0`). Absent
    // an ASSUME, `inputs.assume` is `TRUE` and every hypothesis is byte-identical.
    let init_h = conjoin_assume(&inputs.assume, &inputs.init);
    let j_h = conjoin_assume(&inputs.assume, &inputs.j);
    let not_j = negate_normalized(&inputs.j);
    let not_safety = negate_normalized(&inputs.safety);
    let not_enabled = negate_normalized(&inputs.enabled);
    // F1 feature 1: skolemize symbolic-range existentials in the consecution
    // ANTECEDENT `Next`, declaring the fresh skolem constants rigid alongside the
    // symbolic constant(s). Only consecution translates `Next`, so the skolem
    // constants surface in exactly that obligation.
    let (next_sk, skolem_consts) = skolemize_next_antecedent(&inputs.next);
    let declare_rigid = |t: &mut BmcTranslator| -> Result<(), BmcError> {
        for c in rigid_consts.iter().chain(skolem_consts.iter()) {
            t.declare_rigid_const(c, TlaSort::Int)?;
        }
        Ok(())
    };

    let mk = |ob: SmtObligation| -> Result<ObligationProof, BmcError> {
        scratch_check_unsat_with_proof(ob.name(), &inputs.var_sorts, 1, timeout, |t| {
            declare_rigid(t)?;
            build_smt_obligation(t, ob, &init_h, &next_sk, &not_safety, &j_h, &not_j)
                .map(|_| ())
        })
    };
    let initiation = mk(SmtObligation::Initiation)?;
    // Consecution: try the single whole-`J` check first (unchanged for every spec
    // that already passes). If its proof is not offline-STRICT — a disjunctive Next
    // (`⋁ Aᵢ`) over a conjunctive `J` can force the solver into a case-split that
    // materializes as a TRUST step (CoffeeCan's 4-action × 4-conjunct proof) — fall
    // back to the (action × conjunct) case-split, each case a single-Farkas strict
    // proof. The fallback only REPLACES a non-strict single result, so any spec the
    // single path already certifies is byte-identical.
    let consecution = {
        let single = mk(SmtObligation::Consecution)?;
        if single.strict_verified && bundle_offline_strict(&single.bundle_json) {
            single
        } else if let Some(pairs) = consecution_disjunctive_cases(&inputs.next, &inputs.j) {
            let cased = discharge_consecution_disjunctive_cases(
                inputs,
                rigid_consts,
                &init_h,
                &j_h,
                &not_safety,
                &pairs,
                timeout,
            )?;
            if cased.strict_verified {
                cased
            } else {
                single
            }
        } else {
            single
        }
    };
    let safety_ob = mk(SmtObligation::Safety)?;

    let deadlock_freedom = if matches!(inputs.enabled.node, Expr::Bool(true)) {
        ObligationProof {
            name: "deadlock_freedom",
            unsat: true,
            strict_verified: true,
            clean_supported: true,
            alethe: "structural: unguarded total Next => Enabled(Next) == TRUE".to_string(),
            lrat_present: false,
            bundle_json: None,
        }
    } else if let Some(clauses) = deadlock_dnf_clauses(&inputs.enabled) {
        // DISJUNCTIVE coverage (`¬Enabled` is genuinely disjunctive — e.g. an
        // equality guard's `Neq` negation, CoffeeCan's `BeanCount = 1`): case-split
        // into DNF clauses and strict-prove each. Only the SYMBOLIC-constant rigids
        // are declared (deadlock reasons over `J`/`¬Enabled`, not `Next`, so no
        // skolem constants), matching retranslate_deadlock_dnf_cases_canonical.
        discharge_deadlock_dnf_cases(inputs, rigid_consts, &j_h, &clauses, timeout)?
    } else {
        scratch_check_unsat_with_proof("deadlock_freedom", &inputs.var_sorts, 1, timeout, |t| {
            declare_rigid(t)?;
            let j0 = t.translate_safety_at_step(&j_h, 0)?;
            t.assert(j0);
            let ne0 = t.translate_safety_at_step(&not_enabled, 0)?;
            t.assert(ne0);
            Ok(())
        })?
    };
    Ok(vec![initiation, consecution, safety_ob, deadlock_freedom])
}

/// Leg D part-2 binding for an all-N obligation: re-translate WITHOUT solving
/// (with the rigid constants declared) and render canonically.
pub(crate) fn retranslate_all_n_obligation_canonical(
    ob: SmtObligation,
    inputs: &ObligationInputs,
    rigid_consts: &[String],
) -> Option<Vec<String>> {
    if funcsym_present(&inputs.var_sorts) {
        return retranslate_all_n_pointwise(ob, inputs, rigid_consts);
    }
    let not_j = negate_normalized(&inputs.j);
    let not_safety = negate_normalized(&inputs.safety);
    // Mirror the mint-side skolemization (F1 feature 1) so the render binding is
    // term-for-term identical: the SAME re-derived `Next` yields the SAME fresh
    // constants under the same deterministic naming.
    let (next_sk, skolem_consts) = skolemize_next_antecedent(&inputs.next);
    let mut t = make_translator(&inputs.var_sorts, 1).ok()?;
    for (var_name, sort) in &inputs.var_sorts {
        t.declare_var(var_name, sort.clone()).ok()?;
    }
    for c in rigid_consts.iter().chain(skolem_consts.iter()) {
        t.declare_rigid_const(c, TlaSort::Int).ok()?;
    }
    // Mirror the mint-side ASSUME conjunction (see
    // discharge_all_n_obligations_with_proofs) so the render binding is symmetric.
    let init_h = conjoin_assume(&inputs.assume, &inputs.init);
    let j_h = conjoin_assume(&inputs.assume, &inputs.j);
    let terms = build_smt_obligation(
        &mut t,
        ob,
        &init_h,
        &next_sk,
        &not_safety,
        &j_h,
        &not_j,
    )
    .ok()?;
    // Append the definitional constraints side-asserted during translation (the
    // `\div`/`%` Euclidean linearization) so the reconstructed assertion set
    // matches the proof bundle's, which records the FULL solver stack. Empty
    // absent any `\div`/`%` (byte-identical for every existing obligation).
    let mut rendered = t.render_terms_canonical(&terms);
    rendered.extend(t.aux_asserted_canonical());
    Some(rendered)
}

/// Canonical re-translation of the DEADLOCK-FREEDOM obligation (`J@0 /\ ~Enabled@0`
/// UNSAT) for the all-N verifier's render binding. Mirrors the mint-side bundle
/// build in `discharge_all_n_obligations_with_proofs` term-for-term (assert `J@0`,
/// assert `~Enabled@0`), so an embedded deadlock bundle is bound to the RE-DERIVED
/// spec exactly like the three SMT obligations — closing the gap where the verifier
/// previously accepted this leg on the certificate's own `strict_verified` flag.
pub(crate) fn retranslate_deadlock_obligation_canonical(
    inputs: &ObligationInputs,
    rigid_consts: &[String],
) -> Option<Vec<String>> {
    if funcsym_present(&inputs.var_sorts) {
        return retranslate_all_n_pointwise_deadlock(inputs, rigid_consts);
    }
    let not_enabled = negate_normalized(&inputs.enabled);
    let mut t = make_translator(&inputs.var_sorts, 1).ok()?;
    for (var_name, sort) in &inputs.var_sorts {
        t.declare_var(var_name, sort.clone()).ok()?;
    }
    for c in rigid_consts {
        t.declare_rigid_const(c, TlaSort::Int).ok()?;
    }
    let j_h = conjoin_assume(&inputs.assume, &inputs.j);
    let j0 = t.translate_safety_at_step(&j_h, 0).ok()?;
    let ne0 = t.translate_safety_at_step(&not_enabled, 0).ok()?;
    Some(t.render_terms_canonical(&[j0, ne0]))
}

/// Conjoin the spec's symbolic `ASSUME` onto an obligation hypothesis `x`
/// (`assume ∧ x`). `inputs.assume` is `TRUE` (from [`extract_symbolic_assume`])
/// when the spec has no constant-mentioning ASSUME, in which case `x` is returned
/// unchanged so non-ASSUME specs are byte-identical.
fn conjoin_assume(assume: &Spanned<Expr>, x: &Spanned<Expr>) -> Spanned<Expr> {
    match &assume.node {
        Expr::Bool(true) => x.clone(),
        _ => mk_and(assume.clone(), x.clone()),
    }
}

/// FAIL-CLOSED cap on the number of DNF clauses the disjunctive deadlock coverage
/// expands `¬Enabled` into. Beyond it, the multi-case discharge declines — the
/// single strict check already failed, so deadlock-freedom simply stays unproven
/// (never a false accept).
const DEADLOCK_DNF_CAP: usize = 64;

/// Integer-split disequalities so an equality guard's negation becomes a
/// DISJUNCTION the DNF expansion can case-split: for a `Neq(e, c)` with `c` an
/// integer literal, `e ≠ c ⟺ e ≤ c-1 ∨ e ≥ c+1`. SOUND for integers (the QF_LIA
/// cert fragment — every arithmetic term is Int-sorted). A `Neq` with no literal
/// side is left as-is (its clause then won't strict-check ⇒ fail-closed decline).
fn split_int_disequalities(e: &Spanned<Expr>) -> Spanned<Expr> {
    fn split_around(x: &Spanned<Expr>, c: &num_bigint::BigInt) -> Spanned<Expr> {
        let one = num_bigint::BigInt::from(1);
        let lo = Spanned::dummy(Expr::Leq(
            Box::new(x.clone()),
            Box::new(Spanned::dummy(Expr::Int(c - &one))),
        ));
        let hi = Spanned::dummy(Expr::Geq(
            Box::new(x.clone()),
            Box::new(Spanned::dummy(Expr::Int(c + &one))),
        ));
        Spanned::dummy(Expr::Or(Box::new(lo), Box::new(hi)))
    }
    struct S;
    impl tla_core::ExprFold for S {
        fn fold_expr(&mut self, e: Spanned<Expr>) -> Spanned<Expr> {
            let folded = Spanned {
                node: self.fold_expr_inner(e.node),
                span: e.span,
            };
            if let Expr::Neq(a, b) = &folded.node {
                if let Expr::Int(c) = &b.node {
                    return split_around(a, c);
                }
                if let Expr::Int(c) = &a.node {
                    return split_around(b, c);
                }
            }
            folded
        }
    }
    let mut s = S;
    tla_core::ExprFold::fold_expr(&mut s, e.clone())
}

/// DNF-expand `¬Enabled` into conjunctive clauses for the DISJUNCTIVE deadlock
/// coverage (the blocker-1 non-tractable case — e.g. an equality guard whose
/// negation is a disequality). The coverage `assume ∧ J ∧ ¬Enabled` is UNSAT iff
/// EVERY DNF clause `assume ∧ J ∧ Dᵢ` is UNSAT (DNF is an exact propositional
/// identity + the integer disequality split is exact), so it decomposes into
/// per-clause single-Farkas strict proofs. Returns `Some(clauses)` ONLY when the
/// expansion is genuinely multi-clause (`> 1`) and within the cap; a single clause
/// keeps the caller's single-bundle path, and an over-cap expansion returns `None`
/// (fail-closed — deadlock-freedom stays unproven). DETERMINISTIC (left-to-right
/// distribution), so mint and verify re-derive the identical clause list/order.
fn deadlock_dnf_clauses(enabled: &Spanned<Expr>) -> Option<Vec<Spanned<Expr>>> {
    let not_enabled = negate_normalized(enabled);
    let split = split_int_disequalities(&not_enabled);
    let clauses = distribute_dnf(&split, DEADLOCK_DNF_CAP)?;
    (clauses.len() > 1).then_some(clauses)
}

/// Discharge the deadlock-freedom obligation by DISJUNCTIVE case-split: prove each
/// DNF clause `assume∧J ∧ Dᵢ` UNSAT strictly and carry a MULTI-CASE bundle (a JSON
/// array of per-clause proof bundles, exactly like the multi-branch McCarthy
/// consecution). Deadlock-free iff every case is UNSAT + strict; any SAT case (a
/// genuine deadlock witness) leaves `strict_verified = false` ⇒ the lane declines
/// (fail-closed). Mirrors [`discharge_consecution_branches`].
fn discharge_deadlock_dnf_cases(
    inputs: &ObligationInputs,
    rigid_consts: &[String],
    j_h: &Spanned<Expr>,
    clauses: &[Spanned<Expr>],
    timeout: Option<Duration>,
) -> Result<ObligationProof, BmcError> {
    let mut all_unsat = true;
    let mut all_strict = true;
    let mut bundles: Vec<String> = Vec::new();
    for clause in clauses {
        let p = scratch_check_unsat_with_proof("deadlock_freedom", &inputs.var_sorts, 1, timeout, |t| {
            for c in rigid_consts {
                t.declare_rigid_const(c, TlaSort::Int)?;
            }
            let j0 = t.translate_safety_at_step(j_h, 0)?;
            t.assert(j0);
            let d0 = t.translate_safety_at_step(clause, 0)?;
            t.assert(d0);
            Ok(())
        })?;
        all_unsat &= p.unsat;
        all_strict &= p.strict_verified;
        if let Some(bj) = p.bundle_json {
            bundles.push(bj);
        }
    }
    Ok(ObligationProof {
        name: "deadlock_freedom",
        unsat: all_unsat,
        strict_verified: all_unsat && all_strict,
        clean_supported: all_unsat && all_strict,
        alethe: format!(
            "disjunctive deadlock coverage: {} DNF cases, all strict-verified={}",
            clauses.len(),
            all_strict
        ),
        lrat_present: false,
        bundle_json: if bundles.is_empty() {
            None
        } else {
            serde_json::to_string(&bundles).ok()
        },
    })
}

/// Re-translate each disjunctive-deadlock DNF case to its CANONICAL rendered
/// assertions (no solve), for the verifier's per-case render-binding. `None` when
/// mint did NOT use the multi-case path (single clause, or a FunctionSym spec whose
/// pointwise deadlock the caller re-checks separately), so the caller falls back to
/// the single-bundle re-translation. Mirrors the mint assertion order in
/// [`discharge_deadlock_dnf_cases`] (assert `J_h`, then the clause) EXACTLY.
pub(crate) fn retranslate_deadlock_dnf_cases_canonical(
    inputs: &ObligationInputs,
    rigid_consts: &[String],
) -> Option<Vec<Vec<String>>> {
    if funcsym_present(&inputs.var_sorts) {
        return None;
    }
    let clauses = deadlock_dnf_clauses(&inputs.enabled)?;
    let j_h = conjoin_assume(&inputs.assume, &inputs.j);
    let mut out = Vec::with_capacity(clauses.len());
    for clause in &clauses {
        let mut t = make_translator(&inputs.var_sorts, 1).ok()?;
        for (name, sort) in &inputs.var_sorts {
            t.declare_var(name, sort.clone()).ok()?;
        }
        for c in rigid_consts {
            t.declare_rigid_const(c, TlaSort::Int).ok()?;
        }
        let j0 = t.translate_safety_at_step(&j_h, 0).ok()?;
        let d0 = t.translate_safety_at_step(clause, 0).ok()?;
        out.push(t.render_terms_canonical(&[j0, d0]));
    }
    Some(out)
}

/// Split a disjunctive-Next consecution into (action × invariant-conjunct) CASES.
/// A terminating spec's `Next = ⋁ Aᵢ` and a conjunctive invariant `J = ⋀ cⱼ`, so
/// `J ∧ Next ⟹ J'` iff for EVERY action `Aᵢ` and EVERY conjunct `cⱼ`,
/// `J ∧ Aᵢ ⟹ cⱼ'` — each a SINGLE-action, SINGLE-consequent consecution provable
/// by one Farkas step, with NO trust step from resolving the `Aᵢ`/`cⱼ` disjunctions
/// (which is exactly what makes CoffeeCan's 4×4 whole-`J` consecution proof
/// non-offline-strict). Returns the `(Aᵢ, ¬cⱼ)` pairs when the product is genuinely
/// `> 1` and within the cap; `None` otherwise (keep the single-check path). The
/// conjuncts come from the ORIGINAL `J` (not `assume ∧ J`), so `assume` is not
/// spuriously negated as a consequent; the antecedent stays the full `assume ∧ J`.
/// True iff an obligation proof `bundle_json` passes the OFFLINE strict re-check —
/// the AUTHORITATIVE check the verifier runs (`re_check_bundle_strict` + hole-free
/// quality). The in-process `strict_verified` flag can DISAGREE: a serialized proof
/// may reveal a TRUST step the in-process artifact did not surface (CoffeeCan's
/// disjunctive consecution), and the verifier trusts only the offline result — so
/// mint gates its consecution path choice on this SAME check, not the flag.
fn bundle_offline_strict(bundle_json: &Option<String>) -> bool {
    let Some(bj) = bundle_json else {
        return false;
    };
    let Ok(bundle) = serde_json::from_str::<tla_ay::SerializableProofBundle>(bj) else {
        return false;
    };
    match tla_ay::re_check_bundle_strict(&bundle) {
        Ok(r) => r.quality.is_complete(),
        Err(_) => false,
    }
}

fn consecution_disjunctive_cases(
    next: &Spanned<Expr>,
    j: &Spanned<Expr>,
) -> Option<Vec<(Spanned<Expr>, Spanned<Expr>)>> {
    let mut actions: Vec<&Spanned<Expr>> = Vec::new();
    flatten_or(next, &mut actions);
    let mut conjs: Vec<&Spanned<Expr>> = Vec::new();
    flatten_and(j, &mut conjs);
    let total = actions.len().checked_mul(conjs.len())?;
    if total <= 1 || total > DEADLOCK_DNF_CAP {
        return None;
    }
    let mut pairs = Vec::with_capacity(total);
    for a in &actions {
        for c in &conjs {
            pairs.push(((*a).clone(), negate_normalized(c)));
        }
    }
    Some(pairs)
}

/// Discharge a disjunctive-Next consecution by the (action × conjunct) case-split:
/// strict-prove each `assume∧J ∧ Aᵢ ∧ ¬cⱼ'` UNSAT and carry a MULTI-CASE bundle
/// (JSON array), exactly like the multi-branch McCarthy consecution and the
/// disjunctive deadlock coverage. Inductive iff EVERY case is UNSAT + strict; any
/// SAT case leaves `strict_verified = false` ⇒ the caller keeps the (also-declining)
/// single result. Each action is skolemized independently (a fresh translator per
/// case ⇒ no cross-case constant collision).
fn discharge_consecution_disjunctive_cases(
    inputs: &ObligationInputs,
    rigid_consts: &[String],
    init_h: &Spanned<Expr>,
    j_h: &Spanned<Expr>,
    not_safety: &Spanned<Expr>,
    pairs: &[(Spanned<Expr>, Spanned<Expr>)],
    timeout: Option<Duration>,
) -> Result<ObligationProof, BmcError> {
    let mut all_unsat = true;
    let mut all_strict = true;
    let mut bundles: Vec<String> = Vec::new();
    for (action, not_cj) in pairs {
        let (action_sk, skolem) = skolemize_next_antecedent(action);
        let p = scratch_check_unsat_with_proof("consecution", &inputs.var_sorts, 1, timeout, |t| {
            for c in rigid_consts.iter().chain(skolem.iter()) {
                t.declare_rigid_const(c, TlaSort::Int)?;
            }
            build_smt_obligation(
                t,
                SmtObligation::Consecution,
                init_h,
                &action_sk,
                not_safety,
                j_h,
                not_cj,
            )
            .map(|_| ())
        })?;
        all_unsat &= p.unsat;
        all_strict &= p.strict_verified;
        if let Some(bj) = p.bundle_json {
            bundles.push(bj);
        }
    }
    Ok(ObligationProof {
        name: "consecution",
        unsat: all_unsat,
        strict_verified: all_unsat && all_strict,
        clean_supported: all_unsat && all_strict,
        alethe: format!(
            "disjunctive consecution: {} (action x conjunct) cases, all strict-verified={}",
            pairs.len(),
            all_strict
        ),
        lrat_present: false,
        bundle_json: if bundles.is_empty() {
            None
        } else {
            serde_json::to_string(&bundles).ok()
        },
    })
}

/// Re-translate each disjunctive-consecution (action × conjunct) case canonically
/// (no solve), for the verifier's per-case render-binding. `None` when the spec is
/// not the multi-case shape (FunctionSym McCarthy — handled separately — or a
/// single action/conjunct). Mirrors [`discharge_consecution_disjunctive_cases`]
/// EXACTLY (same pair order, same per-action skolemization, same `assume∧J`
/// antecedent), so the render binding matches term-for-term.
pub(crate) fn retranslate_consecution_disjunctive_cases_canonical(
    inputs: &ObligationInputs,
    rigid_consts: &[String],
) -> Option<Vec<Vec<String>>> {
    if funcsym_present(&inputs.var_sorts) {
        return None;
    }
    let pairs = consecution_disjunctive_cases(&inputs.next, &inputs.j)?;
    let j_h = conjoin_assume(&inputs.assume, &inputs.j);
    let not_safety = negate_normalized(&inputs.safety);
    let mut out = Vec::with_capacity(pairs.len());
    for (action, not_cj) in &pairs {
        let (action_sk, skolem) = skolemize_next_antecedent(action);
        let mut t = make_translator(&inputs.var_sorts, 1).ok()?;
        for (name, sort) in &inputs.var_sorts {
            t.declare_var(name, sort.clone()).ok()?;
        }
        for c in rigid_consts.iter().chain(skolem.iter()) {
            t.declare_rigid_const(c, TlaSort::Int).ok()?;
        }
        let terms = build_smt_obligation(
            &mut t,
            SmtObligation::Consecution,
            &inputs.init,
            &action_sk,
            &not_safety,
            &j_h,
            not_cj,
        )
        .ok()?;
        let mut rendered = t.render_terms_canonical(&terms);
        rendered.extend(t.aux_asserted_canonical());
        out.push(rendered);
    }
    Some(out)
}

// ===========================================================================
// Engine-diverse Leg D part-2 binding (closes the TLA->AY translator trust).
//
// The canonical-render binding (`retranslate_obligation_canonical`) re-derives
// the obligation through TY's OWN `BmcTranslator`, so a translation bug is shared
// by both the producer's proof and the binding and stays invisible. The probe
// cross-check below confirms the EMBEDDED AY obligation denotes the spec via a
// DIFFERENT engine — `tla-eval` — so a `BmcTranslator` (or `negate_normalized`)
// bug is CAUGHT: for a set of concrete probe states, the AY obligation is folded
// to a boolean by ground `substitute`+`simplify`, and the SAME predicate is
// evaluated by tla-eval; ANY disagreement rejects. Restricted to scalar
// (Int/Bool) specs; compound-sort specs return `None` (the verifier keeps the
// render-only binding). Bounded — a refutation aid, not a completeness proof —
// so it augments, never replaces, the render equality.
// ===========================================================================

/// A concrete probe value for a scalar state variable.
#[derive(Clone, Copy, Debug)]
enum ProbeVal {
    Bool(bool),
    Int(i64),
}

/// A concrete probe state: a value for each `(variable, step)` in scope.
#[derive(Clone, Debug)]
struct ProbeAssignment {
    vals: std::collections::HashMap<(String, usize), ProbeVal>,
}

impl ProbeAssignment {
    fn value(&self, base: &str, step: usize) -> Option<ProbeVal> {
        self.vals.get(&(base.to_string(), step)).copied()
    }
}

/// Parse a step-indexed BmcTranslator var name `base__step` -> `(base, step)`.
/// The step is the trailing `__<usize>`; `base` may itself contain `__`.
fn parse_step_var(name: &str) -> Option<(String, usize)> {
    let (base, step_str) = name.rsplit_once("__")?;
    let step: usize = step_str.parse().ok()?;
    Some((base.to_string(), step))
}

/// Deterministic, finite, capped probe states. Scalar domains: `Bool`->{f,t};
/// `Int`->{-1,0,1,2}. The cartesian product over every `(var, step)` in `steps`,
/// deterministically strided down to at most `MAX_PROBES` so the gate is
/// reproducible across re-checks. Returns empty for any non-scalar sort.
fn generate_probe_states(var_sorts: &[(String, TlaSort)], steps: &[usize]) -> Vec<ProbeAssignment> {
    const MAX_PROBES: usize = 64;
    let mut keys: Vec<(String, usize)> = Vec::new();
    let mut domains: Vec<Vec<ProbeVal>> = Vec::new();
    for &step in steps {
        for (name, sort) in var_sorts {
            let dom = match sort {
                TlaSort::Bool => vec![ProbeVal::Bool(false), ProbeVal::Bool(true)],
                TlaSort::Int => vec![
                    ProbeVal::Int(-1),
                    ProbeVal::Int(0),
                    ProbeVal::Int(1),
                    ProbeVal::Int(2),
                ],
                _ => return Vec::new(),
            };
            keys.push((name.clone(), step));
            domains.push(dom);
        }
    }
    if keys.is_empty() {
        return Vec::new();
    }
    let mut total: u128 = 1;
    for d in &domains {
        total = total.saturating_mul(d.len() as u128);
    }
    let count = total.min(MAX_PROBES as u128) as usize;
    // Uniform deterministic stride over [0, total): distinct, spread, no RNG.
    let stride = (total / count.max(1) as u128).max(1);
    let mut out = Vec::with_capacity(count);
    for k in 0..count {
        let idx = (k as u128).saturating_mul(stride);
        if idx >= total {
            break;
        }
        let mut rem = idx;
        let mut vals = std::collections::HashMap::new();
        for (key, dom) in keys.iter().zip(domains.iter()) {
            let d = dom.len() as u128;
            let pick = (rem % d) as usize;
            rem /= d;
            vals.insert(key.clone(), dom[pick]);
        }
        out.push(ProbeAssignment { vals });
    }
    out
}

/// AY side: fold the embedded obligation (conjunction of `assertions`) to a bool
/// at `probe` by simultaneous ground `substitute` then `simplify`, reading back
/// `Constant::Bool`. `None` (fail-closed) if any var is uncovered or any
/// conjunct does not fold to a boolean constant.
fn ay_obligation_truth_at_probe(
    store: &mut tla_ay::TermStore,
    assertions: &[tla_ay::TermId],
    varmap: &std::collections::HashMap<(String, usize), tla_ay::TermId>,
    probe: &ProbeAssignment,
) -> Option<bool> {
    let mut from = Vec::with_capacity(varmap.len());
    let mut to = Vec::with_capacity(varmap.len());
    for ((base, step), var_id) in varmap {
        let const_id = match probe.value(base, *step)? {
            ProbeVal::Bool(b) => store.mk_bool(b),
            ProbeVal::Int(i) => store.mk_int(tla_ay::BigInt::from(i)),
        };
        from.push(*var_id);
        to.push(const_id);
    }
    let mut result = true;
    for &a in assertions {
        let grounded = store.substitute(a, &from, &to);
        let folded = store.simplify(grounded);
        match store.get(folded) {
            tla_ay::TermData::Const(tla_ay::Constant::Bool(b)) => result &= *b,
            _ => return None,
        }
    }
    Some(result)
}

/// Build the tla-eval state-value array for `step`, indexed by the var registry.
fn build_probe_state_vec(
    ctx: &EvalCtx,
    var_sorts: &[(String, TlaSort)],
    probe: &ProbeAssignment,
    step: usize,
) -> Option<Vec<crate::eval::Value>> {
    let reg = ctx.var_registry();
    let mut s = vec![crate::eval::Value::Bool(false); reg.len()];
    for (name, _sort) in var_sorts {
        let idx = reg.get(name)?.as_usize();
        s[idx] = match probe.value(name, step)? {
            ProbeVal::Bool(b) => crate::eval::Value::Bool(b),
            ProbeVal::Int(i) => crate::eval::Value::SmallInt(i),
        };
    }
    Some(s)
}

/// Evaluate a single-state predicate `expr` at `state` via tla-eval; `None`
/// (fail-closed) on any non-boolean result or eval error.
fn eval_pred_single(
    ctx: &EvalCtx,
    state: &[crate::eval::Value],
    expr: &Spanned<Expr>,
) -> Option<bool> {
    let mut c = ctx.clone();
    let _g = c.bind_state_array_guard(state);
    match crate::eval::eval(&c, expr) {
        Ok(crate::eval::Value::Bool(b)) => Some(b),
        _ => None,
    }
}

/// Evaluate a two-state (current+primed) predicate `expr` — e.g. `Next` — with
/// `s0` current and `s1` primed; `None` (fail-closed) on non-boolean/error.
fn eval_pred_two_state(
    ctx: &EvalCtx,
    s0: &[crate::eval::Value],
    s1: &[crate::eval::Value],
    expr: &Spanned<Expr>,
) -> Option<bool> {
    let mut c = ctx.clone();
    let _g = c.bind_state_array_guard(s0);
    // Clear any stale HashMap next-state so eval_prime uses the array fast path.
    *c.next_state_mut() = None;
    let _ng = c.bind_next_state_array_guard(s1);
    match crate::eval::eval(&c, expr) {
        Ok(crate::eval::Value::Bool(b)) => Some(b),
        _ => None,
    }
}

/// tla-eval side: the obligation's truth at `probe`, decomposed exactly as
/// [`build_smt_obligation`] (`Init/\~J`; `J/\Next/\~J'`; `J/\~Safety`) — but the
/// negations use tla-eval's OWN boolean logic (NOT `negate_normalized`), so a
/// `negate_normalized` bug is also caught. `None` (fail-closed) on any eval issue.
fn tla_obligation_truth_at_probe(
    ob: SmtObligation,
    inputs: &ObligationInputs,
    ctx: &EvalCtx,
    probe: &ProbeAssignment,
) -> Option<bool> {
    let s0 = build_probe_state_vec(ctx, &inputs.var_sorts, probe, 0)?;
    match ob {
        SmtObligation::Initiation => {
            let init_b = eval_pred_single(ctx, &s0, &inputs.init)?;
            let j_b = eval_pred_single(ctx, &s0, &inputs.j)?;
            Some(init_b && !j_b)
        }
        SmtObligation::Safety => {
            let j_b = eval_pred_single(ctx, &s0, &inputs.j)?;
            let safety_b = eval_pred_single(ctx, &s0, &inputs.safety)?;
            Some(j_b && !safety_b)
        }
        SmtObligation::Consecution => {
            let s1 = build_probe_state_vec(ctx, &inputs.var_sorts, probe, 1)?;
            let j0 = eval_pred_single(ctx, &s0, &inputs.j)?;
            let next_b = eval_pred_two_state(ctx, &s0, &s1, &inputs.next)?;
            let j1 = eval_pred_single(ctx, &s1, &inputs.j)?;
            Some(j0 && next_b && !j1)
        }
    }
}

/// Build a tla-eval `EvalCtx` for probe evaluation from a certificate's spec +
/// invariant text (mirrors [`rederive_obligation_inputs`]'s augmentation), with
/// state VARIABLES registered (`register_vars` — `load_module` alone registers
/// only operators). `None` if the spec cannot be parsed/lowered.
pub(crate) fn build_probe_eval_ctx(
    spec_src: &str,
    config: &Config,
    j_tla: &str,
) -> Option<EvalCtx> {
    const CERT_J_OP: &str = "TY__Cert_J";
    let _ = config;
    // FIRST-terminator + leading-letter anchoring: see `rederive_obligation_inputs`
    // (first `\n====` = the module `lower` binds; a `_`-leading op name would be eaten
    // by the `[A]_v`/`<A>_v` subscript lexer when the prior unit ends in `]`/`>>`).
    let term_pos = ay_shared::first_module_terminator_pos(spec_src)?;
    let augmented = format!(
        "{}\n{CERT_J_OP} == {j_tla}\n\n{}",
        spec_src[..term_pos].trim_end(),
        &spec_src[term_pos..]
    );
    let tree = tla_core::parse_to_syntax_tree(&augmented);
    let module = tla_core::lower(tla_core::FileId(0), &tree).module?;
    let mut ctx = EvalCtx::new();
    ctx.load_module(&module);
    // `module` here is the certificate's fully-flattened single-module spec, so
    // the top-module scan inside `collect_state_vars` (registry still empty) is
    // exactly the right set; register the result so probe evaluation resolves
    // state-var slots.
    let vars = ay_shared::collect_state_vars(&module, &ctx);
    let var_names: Vec<String> = vars.iter().map(|v| v.to_string()).collect();
    ctx.register_vars(var_names);
    Some(ctx)
}

/// Engine-diverse cross-check of ONE SMT obligation (Leg D part-2). For scalar
/// specs, confirms the embedded AY obligation agrees with tla-eval on every probe
/// state. `Some(true)` = all probes agreed; `Some(false)` = a disagreement (a
/// translator bug caught — REJECT); `None` = not probe-checkable (non-scalar sort,
/// uncovered/unfoldable term, or an eval error — the caller keeps the render-only
/// binding, never accepting on `None` alone).
pub(crate) fn probe_check_obligation_engine_diverse(
    ob: SmtObligation,
    embedded_store: &mut tla_ay::TermStore,
    obligation_assertions: &[tla_ay::TermId],
    inputs: &ObligationInputs,
    ctx: &EvalCtx,
    indep: Option<&crate::cert_indep_frontend::IndepSpec>,
) -> Option<bool> {
    // Scalar gate: only Int/Bool vars are probe-cross-checkable (compound sorts
    // are multi-term-encoded and have no single `base__step` Var).
    if !inputs
        .var_sorts
        .iter()
        .all(|(_, s)| matches!(s, TlaSort::Bool | TlaSort::Int))
    {
        return None;
    }
    let steps: &[usize] = match ob {
        SmtObligation::Initiation | SmtObligation::Safety => &[0],
        SmtObligation::Consecution => &[0, 1],
    };
    let var_names: std::collections::HashSet<&str> =
        inputs.var_sorts.iter().map(|(n, _)| n.as_str()).collect();
    // Map (base, step) -> embedded Var TermId by POSITIONAL scan (the empty names
    // map forbids mk_var-by-name lookup). Restrict to genuine state vars at the
    // obligation's steps so internal/aux vars never enter the substitution.
    let mut varmap: std::collections::HashMap<(String, usize), tla_ay::TermId> =
        std::collections::HashMap::new();
    for i in 0..embedded_store.len() {
        let id = tla_ay::TermId::new(i as u32);
        if let tla_ay::TermData::Var(n, _) = embedded_store.get(id) {
            if let Some((base, step)) = parse_step_var(n) {
                if var_names.contains(base.as_str()) && steps.contains(&step) {
                    varmap.insert((base, step), id);
                }
            }
        }
    }
    let probes = generate_probe_states(&inputs.var_sorts, steps);
    if probes.is_empty() {
        return None;
    }
    for probe in &probes {
        // Run-boundary discipline: clear thread-local eval caches between probes
        // so a prior probe's state-var values cannot leak.
        crate::clear_thread_local_eval_caches();
        let ay = ay_obligation_truth_at_probe(embedded_store, obligation_assertions, &varmap, probe)?;
        let tla = tla_obligation_truth_at_probe(ob, inputs, ctx, probe)?;
        if ay != tla {
            return Some(false);
        }
        // THIRD comparand — the fully independent front end (parser+evaluator that
        // shares nothing with tla_core or BmcTranslator). When it can evaluate this
        // probe, its truth must ALSO equal the embedded AY obligation; a disagreement
        // means a parse/lower OR translation bug — caught by a path that shares NO
        // front end. `None` (out-of-fragment / eval issue) skips ONLY the indep gate.
        if let Some(indep) = indep {
            if let Some(indep_truth) = indep_obligation_truth_at_probe(ob, indep, &inputs.var_sorts, probe)
            {
                if ay != indep_truth {
                    return Some(false);
                }
            }
        }
    }
    Some(true)
}

/// Build the independent-front-end state map (var name -> scalar value) for `step`.
fn build_indep_state(
    var_sorts: &[(String, TlaSort)],
    probe: &ProbeAssignment,
    step: usize,
) -> Option<crate::cert_indep_frontend::IState> {
    use crate::cert_indep_frontend::IVal;
    let mut s = crate::cert_indep_frontend::IState::new();
    for (name, _sort) in var_sorts {
        let v = match probe.value(name, step)? {
            ProbeVal::Bool(b) => IVal::Bool(b),
            ProbeVal::Int(i) => IVal::Int(i),
        };
        s.insert(name.clone(), v);
    }
    Some(s)
}

/// The obligation's truth at `probe` per the INDEPENDENT front end, decomposed
/// exactly as [`build_smt_obligation`]. `None` if the front end cannot evaluate
/// this probe (out-of-fragment / missing var) — the caller then skips the indep
/// gate for this probe (never an accept on `None`).
fn indep_obligation_truth_at_probe(
    ob: SmtObligation,
    indep: &crate::cert_indep_frontend::IndepSpec,
    var_sorts: &[(String, TlaSort)],
    probe: &ProbeAssignment,
) -> Option<bool> {
    let s0 = build_indep_state(var_sorts, probe, 0)?;
    match ob {
        SmtObligation::Initiation => indep.initiation_truth(&s0),
        SmtObligation::Safety => indep.safety_truth(&s0),
        SmtObligation::Consecution => {
            let s1 = build_indep_state(var_sorts, probe, 1)?;
            indep.consecution_truth(&s0, &s1)
        }
    }
}

/// One of the FIVE liveness obligations for `<>P` under `WF(Next)`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum LiveObligation {
    /// `Init /\ ~J`.
    Initiation,
    /// `J /\ Next /\ ~J'`.
    Consecution,
    /// `J /\ m < 0` (=> J => m >= 0: the measure is Nat-bounded on the invariant).
    Bounded,
    /// `J /\ ~P /\ Next /\ m' >= m` (=> every fair `~P` step strictly decreases m).
    Decrease,
    /// `J /\ ~P /\ ~Enabled(Next)` (=> `~P` keeps the fair action enabled).
    Enabled,
}

impl LiveObligation {
    pub(crate) fn name(self) -> &'static str {
        match self {
            LiveObligation::Initiation => "initiation",
            LiveObligation::Consecution => "consecution",
            LiveObligation::Bounded => "live_bounded",
            LiveObligation::Decrease => "live_decrease",
            LiveObligation::Enabled => "live_enabled",
        }
    }
    pub(crate) const ALL: [LiveObligation; 5] = [
        LiveObligation::Initiation,
        LiveObligation::Consecution,
        LiveObligation::Bounded,
        LiveObligation::Decrease,
        LiveObligation::Enabled,
    ];
}

/// The well-founded-descent liveness obligation inputs, re-derived from a
/// certificate's spec + J/P/m text. Like [`ObligationInputs`] but for liveness.
pub(crate) struct LiveInputs {
    pub(crate) var_sorts: Vec<(String, TlaSort)>,
    pub(crate) init: Spanned<Expr>,
    pub(crate) next: Spanned<Expr>,
    pub(crate) j: Spanned<Expr>,
    pub(crate) p: Spanned<Expr>,
    pub(crate) m: Spanned<Expr>,
    pub(crate) enabled: Spanned<Expr>,
}

/// Assert ONE liveness obligation into `t` and return the asserted terms — the
/// SINGLE SOURCE OF TRUTH consumed by both the producer
/// ([`discharge_liveness_obligations_with_proofs`]) and the verifier's
/// re-translation binding ([`retranslate_live_obligation_canonical`]), so they
/// translate identically. `m' >= m` is `Geq(Prime(m), m)` — `Prime` translates
/// `m`'s vars at the next step.
pub(crate) fn build_live_obligation(
    t: &mut BmcTranslator,
    ob: LiveObligation,
    inp: &LiveInputs,
) -> Result<Vec<tla_ay::Term>, BmcError> {
    let not_j = negate_normalized(&inp.j);
    let not_p = negate_normalized(&inp.p);
    let not_enabled = negate_normalized(&inp.enabled);
    let zero = Spanned::dummy(Expr::Int(tla_ay::BigInt::from(0i64)));
    let m_lt_0 = Spanned::dummy(Expr::Lt(Box::new(inp.m.clone()), Box::new(zero)));
    let m_prime = Spanned::dummy(Expr::Prime(Box::new(inp.m.clone())));
    let m_not_decrease = Spanned::dummy(Expr::Geq(Box::new(m_prime), Box::new(inp.m.clone())));

    let mut a = Vec::new();
    let assert = |t: &mut BmcTranslator, term: tla_ay::Term, a: &mut Vec<tla_ay::Term>| {
        t.assert(term);
        a.push(term);
    };
    match ob {
        LiveObligation::Initiation => {
            let x = t.translate_init(&inp.init)?;
            assert(t, x, &mut a);
            let x = t.translate_safety_at_step(&not_j, 0)?;
            assert(t, x, &mut a);
        }
        LiveObligation::Consecution => {
            let x = t.translate_safety_at_step(&inp.j, 0)?;
            assert(t, x, &mut a);
            let x = t.translate_next(&inp.next, 0)?;
            assert(t, x, &mut a);
            let x = t.translate_safety_at_step(&not_j, 1)?;
            assert(t, x, &mut a);
        }
        LiveObligation::Bounded => {
            let x = t.translate_safety_at_step(&inp.j, 0)?;
            assert(t, x, &mut a);
            let x = t.translate_safety_at_step(&m_lt_0, 0)?;
            assert(t, x, &mut a);
        }
        LiveObligation::Decrease => {
            let x = t.translate_safety_at_step(&inp.j, 0)?;
            assert(t, x, &mut a);
            let x = t.translate_safety_at_step(&not_p, 0)?;
            assert(t, x, &mut a);
            let x = t.translate_next(&inp.next, 0)?;
            assert(t, x, &mut a);
            let x = t.translate_safety_at_step(&m_not_decrease, 0)?;
            assert(t, x, &mut a);
        }
        LiveObligation::Enabled => {
            let x = t.translate_safety_at_step(&inp.j, 0)?;
            assert(t, x, &mut a);
            let x = t.translate_safety_at_step(&not_p, 0)?;
            assert(t, x, &mut a);
            let x = t.translate_safety_at_step(&not_enabled, 0)?;
            assert(t, x, &mut a);
        }
    }
    Ok(a)
}

/// Discharge the five liveness obligations WITH AY strict-checked proofs (see
/// [`LiveObligation`] for the descent argument).
pub(crate) fn discharge_liveness_obligations_with_proofs(
    inp: &LiveInputs,
    timeout: Option<Duration>,
) -> Result<Vec<ObligationProof>, BmcError> {
    let mut out = Vec::with_capacity(5);
    for ob in LiveObligation::ALL {
        let proof = scratch_check_unsat_with_proof(ob.name(), &inp.var_sorts, 1, timeout, |t| {
            build_live_obligation(t, ob, inp).map(|_| ())
        })?;
        out.push(proof);
    }
    Ok(out)
}

/// Leg-D part-2 binding for ONE liveness obligation: re-translate it into a FRESH
/// translator WITHOUT solving and render the asserted terms canonically, so the
/// verifier can require the embedded obligation to equal what TY independently
/// re-translates. Mirrors [`retranslate_obligation_canonical`].
pub(crate) fn retranslate_live_obligation_canonical(
    ob: LiveObligation,
    inp: &LiveInputs,
) -> Option<Vec<String>> {
    let mut t = make_translator(&inp.var_sorts, 1).ok()?;
    for (var_name, sort) in &inp.var_sorts {
        t.declare_var(var_name, sort.clone()).ok()?;
    }
    let terms = build_live_obligation(&mut t, ob, inp).ok()?;
    Some(t.render_terms_canonical(&terms))
}

/// Re-derive the obligations from a certificate's spec source + invariant text and
/// discharge all four WITH AY proofs (the in-process AY proof-artifact leg, Leg
/// C). Returns the per-obligation proofs, or `None` if they could not be
/// re-derived. Leg D (the external re-check) reuses
/// [`rederive_obligation_inputs`] so both legs reason about the same ASTs.
pub(crate) fn certificate_obligation_proofs(
    spec_src: &str,
    config: &Config,
    j_tla: &str,
) -> Option<Vec<ObligationProof>> {
    let inputs = rederive_obligation_inputs(spec_src, config, j_tla)?;
    let timeout = BmcConfig::default().solve_timeout;
    discharge_obligations_with_proofs(
        &inputs.var_sorts,
        &inputs.init,
        &inputs.next,
        &inputs.safety,
        &inputs.j,
        &inputs.enabled,
        timeout,
    )
    .ok()
}

/// Run the two-part inductiveness gate for a single candidate predicate `b`.
///
/// Returns `true` iff BOTH validity checks pass — each discharged via its own
/// fresh scratch translator/solver so the gate never pollutes the BMC query:
///   (1) Init => b          i.e.  (Init(s0) /\ ~b(s0))            is UNSAT
///   (2) b /\ Next => b'   i.e.  (b(s0) /\ Next(s0,s1) /\ ~b(s1)) is UNSAT
///
/// Negation-normal form of `Not(expr)`: push the negation through comparisons and
/// boolean connectives so the SMT translation emits a DIRECT comparison literal
/// (`Lt`/`Gt`/`Leq`/`Geq`) rather than `Not(comparison)`.
///
/// This is logically equivalent (`Not(x >= 0)` == `x < 0`) so it never changes a
/// verdict — but it is what makes the inductive-safety obligations' proofs
/// STRICT-VERIFIABLE: AY's strict checker reconstructs the Farkas certificate of a
/// direct comparison's theory lemma, but treats `Not(comparison)` as an unverified
/// "trust" step (measured: `Not(x>=0)` -> Rejected, `x<0` -> Verified). Certifying
/// verification needs the proof, so we hand AY the comparison form.
fn negate_normalized(expr: &Spanned<Expr>) -> Spanned<Expr> {
    let node = match &expr.node {
        // Boolean constants: Not(TRUE)=FALSE, Not(FALSE)=TRUE. Keeps the
        // deadlock-freedom obligation of an unguarded Next (Enabled==TRUE) as a
        // propositional contradiction (J /\ FALSE), which strict-verifies trivially
        // instead of demoting a `Not(TRUE)` to a trust step.
        Expr::Bool(b) => Expr::Bool(!b),
        Expr::Geq(a, b) => Expr::Lt(a.clone(), b.clone()),
        Expr::Leq(a, b) => Expr::Gt(a.clone(), b.clone()),
        Expr::Gt(a, b) => Expr::Leq(a.clone(), b.clone()),
        Expr::Lt(a, b) => Expr::Geq(a.clone(), b.clone()),
        Expr::Eq(a, b) => Expr::Neq(a.clone(), b.clone()),
        Expr::Neq(a, b) => Expr::Eq(a.clone(), b.clone()),
        Expr::And(p, q) => Expr::Or(
            Box::new(negate_normalized(p)),
            Box::new(negate_normalized(q)),
        ),
        Expr::Or(p, q) => Expr::And(
            Box::new(negate_normalized(p)),
            Box::new(negate_normalized(q)),
        ),
        // Double negation: Not(Not(p)) == p.
        Expr::Not(inner) => return inner.as_ref().clone(),
        // Shapes we do not normalize keep an explicit `Not` (still sound).
        _ => Expr::Not(Box::new(expr.clone())),
    };
    Spanned::dummy(node)
}

/// Flatten a (possibly nested) `And` into its top-level conjunct list,
/// left-to-right. A non-`And` expression is its own single conjunct. Fully
/// deterministic on the expanded AST, so the all-N certify and verify sides
/// (which both re-derive through the SAME front end) split identically.
pub(crate) fn flatten_conjuncts(e: &Spanned<Expr>) -> Vec<Spanned<Expr>> {
    match &e.node {
        Expr::And(a, b) => {
            let mut v = flatten_conjuncts(a);
            v.extend(flatten_conjuncts(b));
            v
        }
        _ => vec![e.clone()],
    }
}

/// SOUNDNESS: if either check is SAT or Unknown the predicate is NOT proven
/// inductive and `false` is returned, so the caller never asserts an unproven
/// bound. A `true` result means `b` is logically implied by Init/Next, so
/// conjoining it into the BMC query is equivalence-preserving.
fn gate_is_inductive(
    var_sorts: &[(String, TlaSort)],
    init_expanded: &Spanned<Expr>,
    next_expanded: &Spanned<Expr>,
    b: &Spanned<Expr>,
    timeout: Option<Duration>,
) -> Result<bool, BmcError> {
    // ~b, built as a TLA AST negation so it translates like any predicate.
    let not_b = negate_normalized(b);

    // Gate (1): Init => b. If SAT, some initial state violates b.
    let init_implies_b = scratch_check_unsat(var_sorts, 1, timeout, |t| {
        let init_term = t.translate_init(init_expanded)?;
        t.assert(init_term);
        let not_b0 = t.translate_safety_at_step(&not_b, 0)?;
        t.assert(not_b0);
        Ok(())
    })?;
    if !init_implies_b {
        return Ok(false);
    }

    // Gate (2): b /\ Next => b'. If SAT, a transition leaves the bound.
    scratch_check_unsat(var_sorts, 1, timeout, |t| {
        let b0 = t.translate_safety_at_step(b, 0)?;
        t.assert(b0);
        let next_term = t.translate_next(next_expanded, 0)?;
        t.assert(next_term);
        let not_b1 = t.translate_safety_at_step(&not_b, 1)?;
        t.assert(not_b1);
        Ok(())
    })
}

/// Derive a SOUND inductive interval bound for the integer state variables, or
/// `None` if no per-variable candidate is proven inductive.
///
/// Candidate generation: collect all integer literals in Init/Next/Safety; let
/// [gMin, gMax] be their min/max. For EACH integer state var `v` independently,
/// test whether `gMin <= v /\ v <= gMax` is inductive (via [`gate_is_inductive`])
/// and KEEP only the vars whose bound passes. The returned `B` is the
/// conjunction of the surviving per-var bounds.
///
/// SOUNDNESS of the per-var conjunction: each kept `B_v` independently satisfies
/// Init => B_v and B_v /\ Next => B_v'. The conjunction `B = AND_v B_v` therefore
/// also satisfies Init => B (each conjunct holds in every init state) and
/// B /\ Next => B' (the conjoined hypothesis is *stronger* than each B_v alone,
/// so every consequent B_v' still follows). Thus B is inductive and implied by
/// Init/Next — conjoining it into the BMC query is equivalence-preserving. A
/// loose/wrong per-var candidate simply fails its gate and is dropped.
fn derive_inductive_bound(
    var_sorts: &[(String, TlaSort)],
    init_expanded: &Spanned<Expr>,
    next_expanded: &Spanned<Expr>,
    safety_expanded: &Spanned<Expr>,
    timeout: Option<Duration>,
    debug: bool,
) -> Result<Option<Spanned<Expr>>, BmcError> {
    // No integer vars -> nothing to bound.
    if !var_sorts.iter().any(|(_, s)| matches!(s, TlaSort::Int)) {
        return Ok(None);
    }

    // --- Candidate generation: gather int literals from Init/Next/Safety. ---
    let mut collector = IntLiteralCollector::default();
    collector.walk_expr(&init_expanded.node);
    collector.walk_expr(&next_expanded.node);
    collector.walk_expr(&safety_expanded.node);

    let (Some(&g_min), Some(&g_max)) =
        (collector.lits.iter().min(), collector.lits.iter().max())
    else {
        // No integer literals to seed a bound.
        return Ok(None);
    };
    if g_min > g_max {
        return Ok(None);
    }

    // Performance restriction (NOT soundness): only consider variables that
    // accumulate via arithmetic in Next. Bounding equality-only vars (e.g. token
    // bits) is redundant with the spec and only slows the solver. The gate below
    // still fully governs soundness for every candidate we DO consider.
    let mut arith_vars = std::collections::HashSet::new();
    collect_arith_vars(&next_expanded.node, &mut arith_vars);

    // Per-variable inductiveness gate: keep only the vars whose [gMin,gMax]
    // interval is proven inductive. The kept conjuncts are independently sound
    // and their conjunction remains inductive (see doc-comment).
    let mut kept: Vec<Spanned<Expr>> = Vec::new();
    for (name, sort) in var_sorts {
        if !matches!(sort, TlaSort::Int) {
            continue;
        }
        if !arith_vars.contains(name) {
            // Not an accumulating var — skip (performance heuristic).
            continue;
        }
        let b_v = build_var_interval_expr(name, g_min, g_max);
        if gate_is_inductive(var_sorts, init_expanded, next_expanded, &b_v, timeout)? {
            if debug {
                eprintln!(
                    "[ay-bmc] inductive bound {name} \\in [{g_min},{g_max}] PROVEN — injecting"
                );
            }
            kept.push(b_v);
        } else if debug {
            eprintln!("[ay-bmc] inductive bound {name} \\in [{g_min},{g_max}] rejected by gate");
        }
    }

    Ok(conjoin(kept))
}

/// Best-effort wrapper around [`derive_inductive_bound`]: the bound is a pure
/// performance optimization, so an error inside the gate (e.g. a solver quirk on
/// a scratch query) must NEVER abort the real BMC run. On any error we log (in
/// debug) and proceed with no bound — identical to pre-fix behavior. This keeps
/// the optimization strictly additive: it can only ever speed a run up or leave
/// it unchanged, never break one that previously worked.
fn derive_inductive_bound_best_effort(
    var_sorts: &[(String, TlaSort)],
    init_expanded: &Spanned<Expr>,
    next_expanded: &Spanned<Expr>,
    safety_expanded: &Spanned<Expr>,
    timeout: Option<Duration>,
    debug: bool,
) -> Option<Spanned<Expr>> {
    match derive_inductive_bound(
        var_sorts,
        init_expanded,
        next_expanded,
        safety_expanded,
        timeout,
        debug,
    ) {
        Ok(bound) => bound,
        Err(e) => {
            if debug {
                eprintln!(
                    "[ay-bmc] inductive-bound derivation errored ({e:?}); proceeding without bound"
                );
            }
            None
        }
    }
}

// ===========================================================================
// SOUND symbolic deadlock detection (Fix A).
//
// A state s is DEADLOCKED iff Next is NOT enabled at s:
//     deadlock(s) = ~Enabled(Next)(s) = ~(EXISTS s': Next(s, s')).
//
// The outer negation makes s' UNIVERSALLY quantified:
//     deadlock(s) = FORALL s': ~Next(s, s').
// QF_LIA (the BMC theory) cannot express that universal. The naive encoding
// `assert ~translate_next(s, ghost)` is UNSOUND: a single SAT model for
// `~Next(s, ghost)` only witnesses EXISTS s': ~Next(s, s'), which holds for
// ANY non-total Next (e.g. a guarded `x' = x + 1` where the ghost picks a
// non-incrementing value). That would mislabel perfectly live specs as
// deadlocked and flip Safe -> Unsafe — a false-positive machine.
//
// SOUND ENCODING — CONCRETE-STATE ENUMERATION (validated by the prototype):
// Instead of negating an existential symbolically, we ENUMERATE the reachable
// concrete states at the BMC frontier and test each one for successor-freedom
// with a per-state existential query:
//   1. Build a fresh translator at bound max(k,1); declare vars; assert the
//      reachability prefix Init(s0) /\ Next(0..k-1) — IDENTICAL to the BMC
//      query, so any frontier state s_k it yields is genuinely reachable.
//   2. Loop: check_sat. If UNSAT, no (more) reachable frontier states exist =>
//      no deadlock at this depth (return None). If SAT, extract the concrete
//      frontier state s_k from the model.
//   3. Test s_k for a successor in a SEPARATE fresh translator at bound 1:
//      assert_concrete_state(s_k @ step 0) /\ translate_next(next, 0). This is
//      a CONCRETE existential `EXISTS s': Next(s_k, s')`. If UNSAT => s_k has
//      NO successor => REACHABLE DEADLOCK (return the reachability-prefix
//      trace). If SAT => s_k is live; block s_k in the enumeration solver and
//      continue.
// Because step 3 fixes s_k to concrete values, the `EXISTS s'` is a real
// satisfiability query QF_LIA CAN answer, and its UNSAT genuinely certifies
// FORALL s': ~Next(s_k, s'). No universal is ever asserted symbolically.
//
// STRICTLY ADDITIVE / NEVER-CRASH CONTRACT: the enumeration is capped
// (`MAX_DEADLOCK_ENUM`) and every solve respects the BMC timeout. If the cap
// is hit, or any solve returns Unknown, or any translation errors, the probe
// returns None (INCONCLUSIVE => no deadlock claim). It NEVER emits
// BmcResult::Unknown (the cross-val harness panics on Unknown) and NEVER turns
// a genuinely-safe result Unsafe.
// ===========================================================================

/// Upper bound on the number of distinct reachable frontier states enumerated
/// while probing for a deadlock at a single depth. If exceeded, the probe gives
/// up (returns `None`) rather than risk an unbounded loop. A few hundred is
/// ample for the small integer specs BMC targets; larger frontiers fall through
/// to `BoundReached` exactly as before (additive, never wrong).
const MAX_DEADLOCK_ENUM: usize = 256;

/// Per-solve timeout applied to EVERY deadlock-probe solver call, taken as the
/// min of this and the caller's BMC timeout.
///
/// RATIONALE: the probe's enumeration can hit a hard UNSAT ("no more reachable
/// frontier states") whose proof — reasoning across the full ITE/guard prefix —
/// is far more expensive than the main BMC query. Without a tight per-solve cap
/// the probe could burn the entire (300s default) BMC budget on a SAFE spec
/// just to conclude "no deadlock". Capping each solve keeps the probe cheap:
/// a genuine deadlock is found by the FIRST (cheap, SAT) solve, while a hard
/// "exhausted enumeration" proof simply times out => Unknown => None
/// (INCONCLUSIVE, strictly additive — never a false deadlock, never a false
/// "safe"). Soundness is unaffected; only the probe's willingness to keep
/// grinding is bounded.
const DEADLOCK_PROBE_SOLVE_TIMEOUT: Duration = Duration::from_secs(3);

/// TOTAL wall-clock budget for ALL deadlock probing across every depth of one
/// BMC run. Concrete-state enumeration is cheap when a deadlock exists (it is
/// found quickly) but, on a genuinely-safe spec with a large reachable frontier,
/// it can grind through many states at every depth. This budget bounds that cost
/// so deadlock detection never dominates the BMC wall-clock: once exhausted, the
/// probe gives up (returns `None` = no claim). Strictly additive — giving up can
/// only MISS a deadlock (incompleteness), never invent one (unsoundness).
const DEADLOCK_PROBE_TOTAL_BUDGET: Duration = Duration::from_secs(2);

/// Probe for a REACHABLE deadlock state at depth `k` via concrete-state
/// enumeration (see the module-level soundness argument above).
///
/// `deadline` is the shared total-budget cutoff (see `DEADLOCK_PROBE_TOTAL_BUDGET`);
/// once `Instant::now()` passes it the probe bails with `None`.
///
/// Returns `Some((k, prefix_trace))` iff a reachable concrete state at step `k`
/// has NO `Next` successor. Returns `None` for "no deadlock at this depth" AND
/// for every INCONCLUSIVE outcome (cap hit, solver Unknown, translation error,
/// budget exhausted). Best-effort like `derive_inductive_bound_best_effort`: a
/// `None` never blocks the surrounding BMC run, and a `Some` can only correct a
/// wrong Safe.
fn probe_deadlock_at_depth(
    deadline: Instant,
    var_sorts: &[(String, TlaSort)],
    init_expanded: &Spanned<Expr>,
    next_expanded: &Spanned<Expr>,
    inductive_bound: Option<&Spanned<Expr>>,
    k: usize,
    timeout: Option<Duration>,
    debug: bool,
) -> Option<(usize, Vec<BmcState>)> {
    // Cap every probe solve at min(caller timeout, DEADLOCK_PROBE_SOLVE_TIMEOUT)
    // so a hard "enumeration exhausted" UNSAT cannot burn the whole BMC budget.
    // A timed-out solve surfaces as Unknown below => None (inconclusive).
    let probe_timeout = Some(match timeout {
        Some(t) => t.min(DEADLOCK_PROBE_SOLVE_TIMEOUT),
        None => DEADLOCK_PROBE_SOLVE_TIMEOUT,
    });

    // The enumeration solver needs the full reachability prefix up to step `k`,
    // so it must be built at bound at least `k` (and at least 1 so step-0/Next
    // translation has room). Successor-test translators always run at bound 1.
    let enum_bound = k.max(1);

    // --- Build the enumeration solver: Init(s0) /\ Next(0..k-1). ---
    let mut enum_t = make_translator(var_sorts, enum_bound).ok()?;
    enum_t.set_timeout(probe_timeout);
    for (name, sort) in var_sorts {
        enum_t.declare_var(name, sort.clone()).ok()?;
    }
    let init_term = enum_t.translate_init(init_expanded).ok()?;
    enum_t.assert(init_term);
    for step in 0..k {
        let next_term = enum_t.translate_next(next_expanded, step).ok()?;
        enum_t.assert(next_term);
    }
    // Conjoin the already-proven inductive interval bound at every step of the
    // reachability prefix. Equivalence-preserving (the bound is implied by
    // Init/Next), but it hands LIA a propagatable interval so the ITE-selector
    // search does not blow up ~2^k while enumerating the frontier — exactly the
    // optimization the main BMC query relies on. Skipped if no bound was proven.
    if let Some(bound) = inductive_bound {
        for step in 0..=k {
            if let Ok(bound_term) = enum_t.translate_safety_at_step(bound, step) {
                enum_t.assert(bound_term);
            }
        }
    }

    for _ in 0..MAX_DEADLOCK_ENUM {
        // Total-budget cutoff: stop grinding once the shared deadlock-probe
        // budget is spent. Bailing here only forgoes a deadlock we have not yet
        // found (incompleteness), never claims a false one.
        if Instant::now() >= deadline {
            if debug {
                eprintln!("[ay-bmc] deadlock probe at depth {k} hit total budget; inconclusive");
            }
            return None;
        }
        match enum_t.try_check_sat() {
            Ok(SolveResult::Unsat(_)) => {
                // No (more) reachable frontier states => no deadlock at depth k.
                return None;
            }
            Ok(SolveResult::Sat) => { /* fall through to extract + test */ }
            // Unknown / unexpected / solver error => INCONCLUSIVE, never claim.
            _ => return None,
        }

        // Extract the concrete reachable trace; the frontier state is at step k.
        let model = enum_t.try_get_model().ok()?;
        let trace = enum_t.extract_trace(&model);
        // Defensive: `extract_trace` returns steps 0..=enum_bound; the frontier
        // state lives at index k. If the trace is malformed, give up safely.
        let frontier = trace.get(k)?.clone();

        // --- Successor test: EXISTS s': Next(frontier, s') in a fresh solver. ---
        // This concrete existential is answerable in QF_LIA. UNSAT certifies
        // FORALL s': ~Next(frontier, s') => the frontier state is deadlocked.
        let has_successor = {
            let mut succ_t = make_translator(var_sorts, 1).ok()?;
            succ_t.set_timeout(probe_timeout);
            for (name, sort) in var_sorts {
                succ_t.declare_var(name, sort.clone()).ok()?;
            }
            // Pin the frontier state at step 0, then look for any Next(0).
            let frontier_assignments: Vec<(String, tla_ay::BmcValue)> = frontier
                .assignments
                .iter()
                .map(|(name, value)| (name.clone(), value.clone()))
                .collect();
            succ_t.assert_concrete_state(&frontier_assignments, 0).ok()?;
            let next_term = succ_t.translate_next(next_expanded, 0).ok()?;
            succ_t.assert(next_term);
            match succ_t.try_check_sat() {
                Ok(SolveResult::Sat) => true,
                Ok(SolveResult::Unsat(_)) => false,
                // Inconclusive successor test => do not claim deadlock here.
                _ => return None,
            }
        };

        if !has_successor {
            if debug {
                eprintln!(
                    "[ay-bmc] reachable DEADLOCK found at depth {k}: frontier state has no Next successor"
                );
            }
            // Return the reachability prefix (steps 0..=k) as the witness trace.
            let prefix: Vec<BmcState> = trace.into_iter().take(k + 1).collect();
            return Some((k, prefix));
        }

        // Live state: block it and look for a different reachable frontier state.
        if enum_t.block_concrete_state(&frontier).is_err() {
            // Cannot block (e.g. empty state) => stop enumerating safely.
            return None;
        }
    }

    // Enumeration cap hit without a verdict => INCONCLUSIVE => no claim.
    if debug {
        eprintln!(
            "[ay-bmc] deadlock probe at depth {k} hit enumeration cap ({MAX_DEADLOCK_ENUM}); inconclusive"
        );
    }
    None
}

/// Truncate a BMC counterexample trace to exactly the violation depth.
///
/// `extract_trace` returns one state per declared step `0..=bound_k`. In the
/// INCREMENTAL path the translator is built once at `max_depth`, so a violation
/// found at a smaller `depth` would otherwise carry the full `max_depth + 1`
/// states — the steps beyond `depth` are UNCONSTRAINED (no transition asserted
/// for them yet), i.e. arbitrary solver junk. A violation at depth `d` is a path
/// `s0 .. sd` of exactly `d + 1` states, so we keep only those. In the per-depth
/// path `bound_k == depth` already, so this is a harmless no-op there.
fn truncate_trace_to_depth(trace: Vec<BmcState>, depth: usize) -> Vec<BmcState> {
    let mut trace = trace;
    trace.truncate(depth + 1);
    trace
}

// ===========================================================================
// FIX B: SOUND inductive infinite-state SAFETY CERTIFICATE.
//
// PROBLEM: an infinite-state spec whose Next accumulates arithmetic on a state
// var (e.g. `x' = x + 1`) has an UNBOUNDED reachable state space, so explicit
// BFS `check_module` never terminates — it keeps enumerating x=0,1,2,...
// forever even though the spec is provably safe and deadlock-free.
//
// FIX: before BFS enumerates, attempt a complete symbolic PROOF that the spec
// is safe. Return Success(Safe) ONLY when BOTH of these are discharged:
//
//   (A) SAFETY is inductive. For EACH configured invariant S we build an
//       inductive predicate J such that:
//         - J is 1-inductive:  Init => J  and  J /\ Next => J'   (gate_is_inductive)
//         - J => S             (trivial: J is a CONJUNCTION that *includes* S)
//       We first try J = S directly; if that is not inductive we STRENGTHEN to
//       J = S /\ B where B is the proven-inductive integer interval bound from
//       `derive_inductive_bound` (the same candidate generation BMC uses). If
//       neither J=S nor J=S/\B is inductive for some invariant => NO certificate.
//
//       SOUNDNESS: J inductive and J => S together imply S holds in every
//       reachable state (standard inductive-invariant argument). An inductive
//       proof alone does NOT establish liveness/termination — but we are not
//       proving liveness, only SAFETY, which is exactly what BFS would check.
//
//   (B) DEADLOCK-FREEDOM. An inductive SAFETY proof does NOT imply deadlock-
//       freedom: `Next == count < 3 /\ count' = count+1` keeps `count <= 3`
//       inductive yet DEADLOCKS at count=3. Returning Safe there would be
//       UNSOUND (BFS reports Deadlock => Unsafe).
//
//       We establish deadlock-freedom by GUARD EXTRACTION combined with a
//       FINITENESS FILTER. Flatten Next into top-level conjuncts and classify
//       each as either:
//         - a TOTAL assignment of one primed state var: `v' = e` (e total &
//           unprimed), `v' \in S` (S provably non-empty & unprimed), or
//           `UNCHANGED v` / `UNCHANGED <<v1,..>>`; or
//         - an unprimed-only GUARD (no primed var anywhere in it).
//       The certificate proceeds ONLY when EVERY declared state var is totally
//       assigned exactly once and there are NO guards — i.e. Enabled(Next) ==
//       TRUE. That single condition simultaneously discharges (B) (TRUE-enabled
//       Next can never deadlock, independent of config.check_deadlock) AND acts
//       as a finiteness filter: an UNGUARDED self-accumulating Next genuinely
//       diverges (BFS hangs => the certificate is the only terminating path),
//       whereas a GUARDED accumulator (`count < 3 /\ count'=count+1`) is bounded
//       => BFS terminates, so we deliberately decline (fall through) to avoid
//       perturbing that finite run's stats (e.g. states_found). This is strictly
//       MORE CONSERVATIVE than a general `J => AND(guards)` implication proof:
//       the operator_expansion trap (`count<3 /\ Inc`, Safety count<=3) is
//       guarded => declined => BFS keeps Deadlock => Unsafe, exactly as required.
//
//       If Next does NOT decompose to an unguarded total assignment (disjunctive
//       Next, a conditional/partial assignment, ANY guard, a primed var assigned
//       0 or >1 times, an UNCHANGED of a non-state var, a partial RHS, etc.) we
//       CONSERVATIVELY decline the certificate (return None) and let BFS run —
//       never claiming Safe on a structure we cannot prove unconditionally
//       enabled.
//
// VERDICT-PRESERVING BY CONSTRUCTION: the certificate returns Safe ONLY on a
// complete proof (A) /\ (B). On ANY failure — invariant not inductive even after
// strengthening, Next not cleanly decomposable, a guard reachable under J, or
// ANY solver Unknown/error — it returns None and behavior is EXACTLY today's
// BFS. It therefore can NEVER turn an unsafe / deadlocking / not-actually-
// inductive spec into a false Safe.
// ===========================================================================

/// Returns `true` iff `expr` references any primed sub-expression (a `Prime`
/// node or an `Unchanged` node, both of which constrain a NEXT-state var).
fn contains_primed(expr: &Expr) -> bool {
    let mut found = false;
    fn rec(e: &Expr, found: &mut bool) {
        if *found {
            return;
        }
        match e {
            Expr::Prime(_) | Expr::Unchanged(_) => {
                *found = true;
            }
            _ => {
                let mut child = |c: &Spanned<Expr>| rec(&c.node, found);
                walk_immediate_children(e, &mut child);
            }
        }
    }
    rec(expr, &mut found);
    found
}

/// Extract the bare state-var name from a `Prime(Ident/StateVar)` LHS, if it is
/// a simple primed variable reference (`v'`). Returns `None` for anything else
/// (e.g. a primed function application `f'[i]`), which we treat as NOT a clean
/// total assignment.
fn primed_var_name(expr: &Expr) -> Option<String> {
    match expr {
        Expr::Prime(inner) => match &inner.node {
            Expr::Ident(name, _) | Expr::StateVar(name, ..) => Some(name.clone()),
            _ => None,
        },
        _ => None,
    }
}

/// Flatten a top-level `/\` chain into its conjuncts (no normalization beyond
/// splitting `And`). A non-`And` expression yields a single-element vec.
fn flatten_and<'a>(expr: &'a Spanned<Expr>, out: &mut Vec<&'a Spanned<Expr>>) {
    match &expr.node {
        Expr::And(a, b) => {
            flatten_and(a, out);
            flatten_and(b, out);
        }
        _ => out.push(expr),
    }
}

/// Classification of a single top-level Next conjunct for the deadlock-freedom
/// guard-extraction analysis.
enum NextConjunct<'a> {
    /// A total assignment of these primed state var(s): `v' = e`, `v' \in S`,
    /// or `UNCHANGED <<v1,..>>`. Carries `(var name, deterministic WITNESS)`
    /// pairs: the witness is the unprimed successor value the assignment pins
    /// (`e` for `v' = e`, `v` itself for `UNCHANGED v`) and is what a
    /// [`NextConjunct::PrimedGuard`] on the same var substitutes into `Enabled`
    /// (T2 rail: the SAME assignment the disjunct carries — no fresh
    /// existentials). A `v' \in S` assignment has NO deterministic witness
    /// (`None`): a primed guard on such a var declines the analysis.
    Assign(Vec<(String, Option<Spanned<Expr>>)>),
    /// An unprimed-only guard (references no primed var).
    Guard(&'a Spanned<Expr>),
    /// An `\E k \in lo..hi : body` conjunct whose `body` totally assigns these
    /// vars (with `k` free ONLY in the assignment RHSs) and whose extraction
    /// contributes these OWNED enabling guards: the `k`-free inner guards plus
    /// the range-NONEMPTINESS predicate `lo <= hi` (F1 feature 2 + 3). Enabled
    /// iff those guards hold, since every `k` in a non-empty range yields a
    /// total successor. Carries `(assigned vars, synthesized guards)`. The
    /// assigned vars carry NO witness (the RHS may depend on `k`; substituting
    /// it into a guard would leak `k` out of QF — a primed guard on such a var
    /// declines).
    ExistsAssign(Vec<String>, Vec<Spanned<Expr>>),
    /// A conjunct that CONSTRAINS (but does not totally assign) a primed state
    /// var with an Int range/comparison (T2 widening 2): `v' \in lo..hi` whose
    /// non-emptiness is not provable, or `v' <cmp> e` / `e <cmp> v'`. Resolved
    /// at [`analyze_deadlock_freedom`] level AFTER all assignments collect: the
    /// `Enabled` contribution is the guard with `v'` substituted by the SAME
    /// deterministic witness the disjunct's assignment carries. Obligations need
    /// nothing — the conjunct is already inside the Next the translator asserts.
    PrimedGuard(String, PrimedGuardShape<'a>),
    /// Structure we cannot soundly classify (e.g. primed RHS, conditional
    /// assignment, primed function application). Forces the certificate to bail.
    Opaque,
}

/// The recognized shapes of a [`NextConjunct::PrimedGuard`] — Int ranges and
/// comparisons ONLY (a string-enum primed membership `s' \in {..}` would put a
/// heterogeneous disjunction inside `~Enabled`, re-biting the strict wall).
enum PrimedGuardShape<'a> {
    /// `v' \in lo..hi` with unprimed, provably-total Int bounds. Expands BOTH
    /// sides on substitution: `lo <= rhs /\ rhs <= hi` (min/max-eliminating via
    /// [`leq_lower_bound`]/[`leq_upper_bound`]).
    Range(&'a Spanned<Expr>, &'a Spanned<Expr>),
    /// `v' <cmp> e` or `e <cmp> v'` (cmp in `<=,<,>=,>`) with the other side
    /// unprimed and provably-total Int. Carries the WHOLE conjunct; resolution
    /// substitutes the witness for `v'` in place (orientation preserved).
    Cmp(&'a Spanned<Expr>),
}

/// Span-insensitive structural equality of two expressions. The post-expansion
/// `natMin(i,j) == IF i<j THEN i ELSE j` places the SAME operand term at two
/// positions (the comparison LHS and the THEN branch) with DIFFERENT source
/// spans, so the derived `PartialEq` (which compares spans) would miss the
/// match. Normalizing every span to dummy before comparing recognizes it.
fn eq_ignore_span(a: &Spanned<Expr>, b: &Spanned<Expr>) -> bool {
    struct Strip;
    impl tla_core::ExprFold for Strip {
        fn fold_expr(&mut self, e: Spanned<Expr>) -> Spanned<Expr> {
            Spanned {
                node: self.fold_expr_inner(e.node),
                span: tla_core::Span::dummy(),
            }
        }
    }
    use tla_core::ExprFold as _;
    let mut s = Strip;
    s.fold_expr(a.clone()) == s.fold_expr(b.clone())
}

/// Returns `true` iff a bare `Ident`/`StateVar` named `name` occurs anywhere in
/// `expr` (used to enforce the F1 "`k` only in assignments, never in a guard or
/// a range bound" rail).
fn expr_mentions(expr: &Expr, name: &str) -> bool {
    match expr {
        Expr::Ident(n, _) | Expr::StateVar(n, ..) => n == name,
        _ => {
            let mut found = false;
            let mut child = |c: &Spanned<Expr>| {
                if !found && expr_mentions(&c.node, name) {
                    found = true;
                }
            };
            walk_immediate_children(expr, &mut child);
            found
        }
    }
}

/// Recognize the post-expansion `natMin(a,b) == IF a<b THEN a ELSE b` (accepting
/// `<` or `<=`) and return `(a, b)`. A general `IF` (branches not equal to the
/// comparison operands) returns `None` — F1 declines general ITE, only the exact
/// min shape is linearizable in a bound.
fn match_min_ite(e: &Spanned<Expr>) -> Option<(Spanned<Expr>, Spanned<Expr>)> {
    let Expr::If(cond, then_b, else_b) = &e.node else {
        return None;
    };
    let (l, r) = match &cond.node {
        Expr::Lt(l, r) | Expr::Leq(l, r) => (l, r),
        _ => return None,
    };
    (eq_ignore_span(l, then_b) && eq_ignore_span(r, else_b))
        .then(|| ((**then_b).clone(), (**else_b).clone()))
}

/// Dually recognize `natMax(a,b) == IF a<b THEN b ELSE a` and return `(a, b)`.
fn match_max_ite(e: &Spanned<Expr>) -> Option<(Spanned<Expr>, Spanned<Expr>)> {
    let Expr::If(cond, then_b, else_b) = &e.node else {
        return None;
    };
    let (l, r) = match &cond.node {
        Expr::Lt(l, r) | Expr::Leq(l, r) => (l, r),
        _ => return None,
    };
    (eq_ignore_span(l, else_b) && eq_ignore_span(r, then_b))
        .then(|| ((**else_b).clone(), (**then_b).clone()))
}

fn leq(a: Spanned<Expr>, b: Spanned<Expr>) -> Spanned<Expr> {
    Spanned::dummy(Expr::Leq(Box::new(a), Box::new(b)))
}

/// Build the affine predicate for `k <= hi`, eliminating a `natMin` upper bound:
/// `k <= min(a,b)` ⇔ `k<=a /\ k<=b`. A plain (non-`If`) `hi` yields `k<=hi`
/// verbatim. A general `If` that is not the min shape returns `None` (decline).
fn leq_upper_bound(k: &Spanned<Expr>, hi: &Spanned<Expr>) -> Option<Spanned<Expr>> {
    if matches!(hi.node, Expr::If(..)) {
        let (a, b) = match_min_ite(hi)?;
        Some(Spanned::dummy(Expr::And(
            Box::new(leq(k.clone(), a)),
            Box::new(leq(k.clone(), b)),
        )))
    } else {
        Some(leq(k.clone(), hi.clone()))
    }
}

/// Build the affine predicate for `lo <= k`, eliminating a `natMax` lower bound:
/// `max(a,b) <= k` ⇔ `a<=k /\ b<=k`. A plain (non-`If`) `lo` yields `lo<=k`
/// verbatim. A general `If` that is not the max shape returns `None` (decline).
fn leq_lower_bound(lo: &Spanned<Expr>, k: &Spanned<Expr>) -> Option<Spanned<Expr>> {
    if matches!(lo.node, Expr::If(..)) {
        let (a, b) = match_max_ite(lo)?;
        Some(Spanned::dummy(Expr::And(
            Box::new(leq(a, k.clone())),
            Box::new(leq(b, k.clone())),
        )))
    } else {
        Some(leq(lo.clone(), k.clone()))
    }
}

/// The range-NONEMPTINESS predicate for `lo..hi` (`lo <= hi`) with `natMin` in
/// the upper position eliminated: `lo <= min(a,b)` ⇔ `lo<=a /\ lo<=b`. Declines
/// (returns `None`) when `lo` is itself an `If` (a max-in-lower-bound of the
/// non-emptiness test is not part of the recognized glowingRaccoon shape).
fn range_nonempty(lo: &Spanned<Expr>, hi: &Spanned<Expr>) -> Option<Spanned<Expr>> {
    if matches!(lo.node, Expr::If(..)) {
        return None;
    }
    leq_upper_bound(lo, hi)
}

/// Returns `true` iff `e` is a TOTAL expression: it always evaluates to a value
/// for any source state, so an assignment `v' = e` always has a successor.
///
/// SOUNDNESS (load-bearing for deadlock-freedom): we reject any expression that
/// could be PARTIAL — integer division / modulo (undefined on a zero divisor),
/// function application (out of domain), EXCEPT, CHOOSE, etc. Over QF_LIA the
/// realistic partiality sources are `\div` and `%`; anything outside the
/// total-arithmetic fragment below is conservatively rejected. A rejected RHS
/// makes the conjunct Opaque => the certificate declines (never a false Safe).
fn is_total_assignment_rhs(e: &Expr, funcsym_vars: &std::collections::HashSet<String>) -> bool {
    match e {
        // A bare literal/var as the WHOLE rhs is a total assignment to the
        // primed var (`v' = TRUE`, `v' = "s"`, `v' = w` all have a value).
        Expr::Int(_) | Expr::Bool(_) | Expr::String(_) => true,
        Expr::Ident(..) | Expr::StateVar(..) => true,
        // A FunctionSym read `f[k]` is TOTAL: the map-only encoding makes `f` a
        // total `(Array Int T)`, so a value always exists (slice-2 read-valued
        // writes like `[f EXCEPT ![p] = f[q]]`). RESTRICTED to FunctionSym vars —
        // a general/finite-function read could be out-of-domain (partial), and
        // treating it as total here would over-claim a successor → an UNSOUND
        // deadlock-freedom Enabled. `funcsym_vars` is empty on every non-all-N
        // path, so this arm is inert there (byte-identical old behavior).
        Expr::FuncApply(base, _) => {
            funcsym_base_name(base).is_some_and(|n| funcsym_vars.contains(&n))
        }
        // Total integer arithmetic only — and its OPERANDS must be integer-
        // valued. A Bool/String literal inside arithmetic (`v' = w + TRUE`) is a
        // TLA type error with no value (a state could deadlock), so the operands
        // are checked with `is_total_int_operand`, which rejects Bool/String
        // leaves. This makes deadlock-freedom soundness self-contained here,
        // rather than relying on the downstream QF_LIA translator to reject it.
        Expr::Add(a, b) | Expr::Sub(a, b) | Expr::Mul(a, b) => {
            is_total_int_operand(&a.node) && is_total_int_operand(&b.node)
        }
        Expr::Neg(a) => is_total_int_operand(&a.node),
        // `[base EXCEPT !.f = val, ...]` — a record/function UPDATE. It always
        // yields a value (hence a successor exists) when the base is a plain
        // state var and every replacement value is itself total (`@`, `@ ± n`,
        // literals). This is the total-assignment shape a terminating record
        // spec (e.g. CoffeeCan's `can' = [can EXCEPT !.black = @ - 1]`) uses.
        Expr::Except(base, specs) => {
            matches!(&base.node, Expr::Ident(..) | Expr::StateVar(..))
                && specs
                    .iter()
                    .all(|sp| is_total_assignment_rhs(&sp.value.node, funcsym_vars))
        }
        // `x % k` / `x \div k` with a CONCRETE POSITIVE literal divisor `k` are
        // TOTAL: a nonzero divisor cannot divide-by-zero, so the write always has a
        // value (a successor exists). tla-ay linearizes them EXACTLY as
        // `x = k*q + r ∧ 0 ≤ r < k` (#556), a QF_LIA-strict encoding — the range
        // `0 ≤ r < k` is asserted there, so a `written ∈ 0..k-1` consecution goal
        // discharges with no extra fact here.
        Expr::Mod(a, b) | Expr::IntDiv(a, b) => {
            is_total_int_operand(&a.node) && is_positive_int_literal(&b.node)
        }
        // A SYMBOLIC or ≤0 divisor stays PARTIAL (tla-ay only linearizes constant
        // positive divisors; else the CHC path, not strict-checkable) => reject.
        // Pow/Range and everything else (CHOOSE, IF, non-FunctionSym apply, ...) =>
        // reject (fail-closed).
        _ => false,
    }
}

/// Returns `true` iff `e` is a concrete POSITIVE integer literal — the only
/// divisor shape for which `x % k` / `x \div k` is provably total (nonzero) AND
/// strict-linearizable (tla-ay #556 requires a constant positive divisor). A
/// symbolic constant, zero, or negative literal is rejected (fail-closed).
fn is_positive_int_literal(e: &Expr) -> bool {
    matches!(e, Expr::Int(k) if *k > num_bigint::BigInt::from(0))
}

/// Returns `true` iff `e` is a provably integer-valued, total arithmetic
/// operand. Unlike a whole rhs, an operand inside `+`/`-`/`*` may NOT be a
/// Bool/String literal (that would be a type error with no value). A bare
/// `Ident`/`StateVar` is assumed integer here (it appears under arithmetic); a
/// genuinely non-integer var is independently rejected by Obligation (A)'s
/// QF_LIA inductiveness proof, so accepting it cannot yield a false `Safe`.
fn is_total_int_operand(e: &Expr) -> bool {
    match e {
        Expr::Int(_) => true,
        Expr::Ident(..) | Expr::StateVar(..) => true,
        Expr::Add(a, b) | Expr::Sub(a, b) | Expr::Mul(a, b) => {
            is_total_int_operand(&a.node) && is_total_int_operand(&b.node)
        }
        Expr::Neg(a) => is_total_int_operand(&a.node),
        // `x % k` / `x \div k` with a concrete positive divisor are total integer
        // operands (nonzero divisor; see is_total_assignment_rhs).
        Expr::Mod(a, b) | Expr::IntDiv(a, b) => {
            is_total_int_operand(&a.node) && is_positive_int_literal(&b.node)
        }
        // A record field access `r.f` is a TOTAL value (a record is a total map
        // over its fields — the field always has a value). Assumed Int here, like a
        // bare `Ident`; a genuinely non-Int field is independently rejected by the
        // QF_LIA inductiveness proof, so accepting it cannot yield a false `Safe`.
        Expr::RecordAccess(..) => true,
        // Bool/String literal inside arithmetic => type error => not total.
        _ => false,
    }
}

/// Returns `true` iff the set expression `s` is SYNTACTICALLY non-empty, so a
/// membership `v' \in s` always has a witness (a successor exists). Conservative:
/// only a non-empty set enum `{e1,..}` or a literal range `lo..hi` with lo<=hi
/// qualifies; anything else (set comprehension, variable set, `Int`, etc.) is
/// rejected to avoid a false deadlock-free claim on a possibly-empty set.
fn set_is_provably_nonempty(s: &Expr) -> bool {
    match s {
        Expr::SetEnum(elems) => !elems.is_empty(),
        Expr::Range(lo, hi) => match (&lo.node, &hi.node) {
            (Expr::Int(l), Expr::Int(h)) => l <= h,
            _ => false,
        },
        _ => false,
    }
}

/// Collect the bare state-var names named by an `UNCHANGED` argument: a single
/// `UNCHANGED v`, a tuple `UNCHANGED <<v1, v2>>`, or a NESTED tuple
/// `UNCHANGED <<a, <<b, c>>>>` (flattened recursively — `UNCHANGED` of a tuple
/// is the conjunction of member `UNCHANGED`s, so nesting is a pure grouping).
/// Returns `None` if any leaf is not a plain variable reference.
fn unchanged_var_names(arg: &Expr) -> Option<Vec<String>> {
    match arg {
        Expr::Ident(name, _) | Expr::StateVar(name, ..) => Some(vec![name.clone()]),
        Expr::Tuple(elems) => {
            let mut names = Vec::with_capacity(elems.len());
            for e in elems {
                names.extend(unchanged_var_names(&e.node)?);
            }
            Some(names)
        }
        _ => None,
    }
}

/// Classify one top-level Next conjunct. `state_vars` is the set of declared
/// state-variable names (used to reject `UNCHANGED` of a non-state var).
fn classify_next_conjunct<'a>(
    conj: &'a Spanned<Expr>,
    state_vars: &std::collections::HashSet<String>,
    funcsym_vars: &std::collections::HashSet<String>,
) -> NextConjunct<'a> {
    match &conj.node {
        // `v' = e`  — total assignment iff LHS is a primed simple var and the
        // RHS does NOT itself reference a primed var (a primed RHS like
        // `v' = w'` is a relational constraint, not a total assignment).
        Expr::Eq(lhs, rhs) => {
            if let Some(name) = primed_var_name(&lhs.node) {
                // Total assignment iff RHS is unprimed AND a provably-total
                // expression (so a successor value always exists). The RHS is
                // the deterministic WITNESS a primed guard on `name` may
                // substitute into Enabled.
                if state_vars.contains(&name)
                    && !contains_primed(&rhs.node)
                    && is_total_assignment_rhs(&rhs.node, funcsym_vars)
                {
                    return NextConjunct::Assign(vec![(name, Some((**rhs).clone()))]);
                }
                // Primed LHS but suspicious/partial RHS => not provably total.
                // (An Eq guard shape `v' = <non-total e>` is NOT widened: any
                // provably-total-Int `e` already classifies as the assignment,
                // and a partial `e` inside Enabled could over-claim a successor.)
                return NextConjunct::Opaque;
            }
            // No primed var anywhere => unprimed guard.
            if !contains_primed(&conj.node) {
                return NextConjunct::Guard(conj);
            }
            NextConjunct::Opaque
        }
        // `v' \in S` — total assignment (picks some member) iff LHS is a primed
        // simple var, S is unprimed, AND S is PROVABLY NON-EMPTY (otherwise the
        // membership has no witness => no successor => could deadlock). We only
        // accept syntactically non-empty S: a non-empty set enum, or a literal
        // integer range `lo..hi` with lo <= hi. There is no DETERMINISTIC
        // witness (any member may be picked), so the witness is `None`.
        //
        // T2 widening 2: a membership in an Int range whose non-emptiness is
        // NOT provable (`v' \in lo..hi`, symbolic bounds) is a PRIMED GUARD —
        // it constrains rather than provides the successor. It resolves against
        // the disjunct's own deterministic assignment of `v` (or declines).
        Expr::In(lhs, set) => {
            if let Some(name) = primed_var_name(&lhs.node) {
                if state_vars.contains(&name) && !contains_primed(&set.node) {
                    if set_is_provably_nonempty(&set.node) {
                        return NextConjunct::Assign(vec![(name, None)]);
                    }
                    if let Expr::Range(lo, hi) = &set.node {
                        if is_total_int_operand(&lo.node) && is_total_int_operand(&hi.node) {
                            return NextConjunct::PrimedGuard(
                                name,
                                PrimedGuardShape::Range(lo, hi),
                            );
                        }
                    }
                }
                return NextConjunct::Opaque;
            }
            if !contains_primed(&conj.node) {
                return NextConjunct::Guard(conj);
            }
            NextConjunct::Opaque
        }
        // Int comparisons: unprimed => guard; `v' <cmp> e` / `e <cmp> v'` with
        // the OTHER side unprimed, provably-total Int => a primed guard on `v`
        // (T2 widening 2), resolved against the disjunct's own assignment.
        Expr::Leq(a, b) | Expr::Lt(a, b) | Expr::Geq(a, b) | Expr::Gt(a, b) => {
            if !contains_primed(&conj.node) {
                return NextConjunct::Guard(conj);
            }
            let sides = [(a, b), (b, a)];
            for (pv, other) in sides {
                if let Some(name) = primed_var_name(&pv.node) {
                    if state_vars.contains(&name)
                        && !contains_primed(&other.node)
                        && is_total_int_operand(&other.node)
                    {
                        return NextConjunct::PrimedGuard(name, PrimedGuardShape::Cmp(conj));
                    }
                }
            }
            NextConjunct::Opaque
        }
        // `UNCHANGED <<..>>` — total assignment of each named var to its current
        // value (witness: the var itself, unprimed). Reject if any named entity
        // is not a declared state var.
        Expr::Unchanged(arg) => match unchanged_var_names(&arg.node) {
            Some(names) if names.iter().all(|n| state_vars.contains(n)) => NextConjunct::Assign(
                names
                    .into_iter()
                    .map(|n| {
                        let witness = Spanned::dummy(Expr::Ident(n.clone(), NameId::INVALID));
                        (n, Some(witness))
                    })
                    .collect(),
            ),
            _ => NextConjunct::Opaque,
        },
        // `\E k \in lo..hi : body` — a bounded existential assignment (F1
        // feature 2 + 3). Enabled via range-nonemptiness when `k` is confined to
        // the assignment RHSs; see [`classify_exists_assign`].
        Expr::Exists(bounds, body) => {
            classify_exists_assign(bounds, body, state_vars, funcsym_vars)
        }
        // Any other conjunct: a pure unprimed guard is fine; anything that
        // references a primed var but is not one of the assignment shapes above
        // (disjunction, IF/THEN/ELSE assignment, etc.) is opaque.
        _ => {
            if contains_primed(&conj.node) {
                NextConjunct::Opaque
            } else {
                NextConjunct::Guard(conj)
            }
        }
    }
}

/// Classify an `\E k \in lo..hi : body` Next conjunct (F1 feature 2 + 3).
///
/// ENABLED VIA RANGE-NONEMPTINESS: the action `\E k ∈ [lo,hi] : body` has a
/// successor iff SOME `k` in the range makes `body` hold. When (a) `k` binds a
/// single integer range, (b) every conjunct of `body` is either a TOTAL
/// assignment (whose RHS may use `k`) or a `k`-FREE guard, and (c) `k` appears
/// ONLY in assignment RHSs (never a guard or a range bound), then EVERY `k` in a
/// non-empty range yields a total successor, so
///   `Enabled = (k-free inner guards) ∧ (lo <= hi)`
/// — affine, with `natMin`/`natMax` in the bounds eliminated ([`range_nonempty`]).
/// Any other shape (k in a guard, a partial/relational body conjunct, a
/// non-range domain, multiple bound vars) is [`NextConjunct::Opaque`] — decline.
fn classify_exists_assign<'a>(
    bounds: &[tla_core::ast::BoundVar],
    body: &Spanned<Expr>,
    state_vars: &std::collections::HashSet<String>,
    funcsym_vars: &std::collections::HashSet<String>,
) -> NextConjunct<'a> {
    if bounds.len() != 1 {
        return NextConjunct::Opaque;
    }
    let bound = &bounds[0];
    let Some(domain) = &bound.domain else {
        return NextConjunct::Opaque;
    };
    let Expr::Range(lo, hi) = &domain.node else {
        return NextConjunct::Opaque;
    };
    let k = bound.name.node.as_str();
    // The range bounds define the domain and MUST be `k`-free.
    if expr_mentions(&lo.node, k) || expr_mentions(&hi.node, k) {
        return NextConjunct::Opaque;
    }

    let mut inner = Vec::new();
    flatten_and(body, &mut inner);
    let mut assigned: Vec<String> = Vec::new();
    let mut guards: Vec<Spanned<Expr>> = Vec::new();
    for c in inner {
        match classify_next_conjunct(c, state_vars, funcsym_vars) {
            NextConjunct::Assign(pairs) => {
                assigned.extend(pairs.into_iter().map(|(name, _witness)| name));
            }
            NextConjunct::Guard(g) => {
                // A guard mentioning `k` is the k-in-guard shape — decline (the
                // nonemptiness rewrite is unsound when `k` also gates the action).
                if expr_mentions(&g.node, k) {
                    return NextConjunct::Opaque;
                }
                guards.push(g.clone());
            }
            // A nested exists-assign, a primed guard (its witness substitution
            // could leak `k` out of QF — T2 keeps primed guards OUTSIDE ∃), or
            // an opaque conjunct is out of shape.
            NextConjunct::ExistsAssign(..)
            | NextConjunct::PrimedGuard(..)
            | NextConjunct::Opaque => {
                return NextConjunct::Opaque;
            }
        }
    }
    if assigned.is_empty() {
        return NextConjunct::Opaque;
    }
    // Range-nonemptiness (`lo <= hi`, min/max-eliminated) is the enabling guard.
    let Some(nonempty) = range_nonempty(lo, hi) else {
        return NextConjunct::Opaque;
    };
    guards.push(nonempty);
    NextConjunct::ExistsAssign(assigned, guards)
}

/// Outcome of the deadlock-freedom guard-extraction analysis.
enum DeadlockFreedom {
    /// Next cleanly decomposes; `Enabled(Next) == AND(guards)`. Empty guard list
    /// means `Enabled(Next) == TRUE`.
    Decomposed(Vec<Spanned<Expr>>),
    /// Next does NOT cleanly decompose; cannot prove deadlock-freedom.
    Undecomposable,
}

/// A guard slot collected in conjunct order during the classification pass —
/// primed guards resolve only AFTER every assignment's witness is known, but
/// the FINAL guard order must stay the deterministic conjunct order (mint and
/// verify re-derive the same list from the same AST).
enum PendingGuard<'a> {
    Plain(&'a Spanned<Expr>),
    /// Guards synthesized by an inner classification (ExistsAssign).
    Synth(Vec<Spanned<Expr>>),
    Primed(String, PrimedGuardShape<'a>),
}

/// Run the guard-extraction analysis on an (already expanded) Next predicate.
///
/// Returns [`DeadlockFreedom::Decomposed`] with the unprimed guard conjuncts iff
/// EVERY declared state var is totally assigned exactly once, every other
/// top-level conjunct is an unprimed guard, and every PRIMED guard (T2 widening
/// 2) resolves against the deterministic witness of ITS OWN disjunct's
/// assignment of that var. Otherwise [`Undecomposable`].
fn analyze_deadlock_freedom(
    next_expanded: &Spanned<Expr>,
    state_var_names: &[String],
    funcsym_vars: &std::collections::HashSet<String>,
) -> DeadlockFreedom {
    let state_vars: std::collections::HashSet<String> =
        state_var_names.iter().cloned().collect();

    let mut conjuncts = Vec::new();
    flatten_and(next_expanded, &mut conjuncts);

    // Pass 1 — classify every conjunct, recording each assigned var's
    // deterministic WITNESS (None when the assignment picks nondeterministically
    // or under an ∃k) and the guard slots in conjunct order.
    let mut assigned: std::collections::HashMap<String, Option<Spanned<Expr>>> =
        std::collections::HashMap::new();
    let mut pending: Vec<PendingGuard<'_>> = Vec::new();

    for conj in conjuncts {
        match classify_next_conjunct(conj, &state_vars, funcsym_vars) {
            NextConjunct::Assign(pairs) => {
                for (name, witness) in pairs {
                    // A var assigned more than once => not a clean single total
                    // assignment => decline.
                    if assigned.insert(name, witness).is_some() {
                        return DeadlockFreedom::Undecomposable;
                    }
                }
            }
            NextConjunct::ExistsAssign(names, extra_guards) => {
                for name in names {
                    // No witness: the ∃k-RHS is not a deterministic successor
                    // value a primed guard could substitute.
                    if assigned.insert(name, None).is_some() {
                        return DeadlockFreedom::Undecomposable;
                    }
                }
                // The range-nonemptiness + k-free inner guards enable the action.
                pending.push(PendingGuard::Synth(extra_guards));
            }
            NextConjunct::Guard(g) => pending.push(PendingGuard::Plain(g)),
            NextConjunct::PrimedGuard(name, shape) => {
                pending.push(PendingGuard::Primed(name, shape));
            }
            NextConjunct::Opaque => return DeadlockFreedom::Undecomposable,
        }
    }

    // Every declared state var must be assigned exactly once. (Vars in
    // `assigned` are a subset of state_vars by construction.)
    if assigned.len() != state_vars.len() {
        return DeadlockFreedom::Undecomposable;
    }

    // Pass 2 — resolve the guard slots in conjunct order. Each primed guard
    // substitutes THE SAME deterministic witness its disjunct's assignment
    // carries (T2 rail — never a fresh existential): `v' \in lo..hi` becomes
    // `lo <= rhs /\ rhs <= hi` (min/max-eliminating), `v' <cmp> e` becomes
    // `rhs <cmp> e` in place. A guard on a var whose assignment has no
    // deterministic witness (`v' \in S`, ∃k-RHS) or a non-Int witness declines.
    let mut guards: Vec<Spanned<Expr>> = Vec::new();
    for slot in pending {
        match slot {
            PendingGuard::Plain(g) => guards.push(g.clone()),
            PendingGuard::Synth(gs) => guards.extend(gs),
            PendingGuard::Primed(name, shape) => {
                let Some(Some(witness)) = assigned.get(&name) else {
                    return DeadlockFreedom::Undecomposable;
                };
                // The witness must itself be provably-total Int arithmetic —
                // a Bool/String/EXCEPT witness inside an Int range/comparison
                // would be a type error the SMT totalization could mask.
                if !is_total_int_operand(&witness.node) {
                    return DeadlockFreedom::Undecomposable;
                }
                match shape {
                    PrimedGuardShape::Range(lo, hi) => {
                        let Some(lower) = leq_lower_bound(lo, witness) else {
                            return DeadlockFreedom::Undecomposable;
                        };
                        let Some(upper) = leq_upper_bound(witness, hi) else {
                            return DeadlockFreedom::Undecomposable;
                        };
                        guards.push(lower);
                        guards.push(upper);
                    }
                    PrimedGuardShape::Cmp(conj) => {
                        let substituted = substitute_primed_var(conj, &name, witness);
                        // The substitution must eliminate EVERY primed
                        // reference (the classifier guarantees `v` is the only
                        // one; this is the fail-closed re-check).
                        if contains_primed(&substituted.node) {
                            return DeadlockFreedom::Undecomposable;
                        }
                        guards.push(substituted);
                    }
                }
            }
        }
    }

    DeadlockFreedom::Decomposed(guards)
}

/// Replace every `Prime(var)` reference to the named state var by `witness`
/// (an unprimed expression). Used to resolve a [`NextConjunct::PrimedGuard`]
/// against its disjunct's own deterministic assignment.
fn substitute_primed_var(
    expr: &Spanned<Expr>,
    var: &str,
    witness: &Spanned<Expr>,
) -> Spanned<Expr> {
    struct Sub<'a> {
        var: &'a str,
        witness: &'a Spanned<Expr>,
    }
    impl tla_core::ExprFold for Sub<'_> {
        fn fold_expr(&mut self, e: Spanned<Expr>) -> Spanned<Expr> {
            if let Expr::Prime(inner) = &e.node {
                if matches!(&inner.node,
                    Expr::Ident(n, _) | Expr::StateVar(n, ..) if n == self.var)
                {
                    return self.witness.clone();
                }
            }
            Spanned {
                node: self.fold_expr_inner(e.node),
                span: e.span,
            }
        }
    }
    use tla_core::ExprFold as _;
    let mut s = Sub { var, witness };
    s.fold_expr(expr.clone())
}

/// The enabling predicate `Enabled(Next)` for the certificate's deadlock-freedom
/// obligation: the conjunction of `Next`'s guards (TRUE for an unguarded total
/// Next). `Enabled(Next)(s)` holds iff `s` has a successor, so `J => Enabled(Next)`
/// establishes deadlock-freedom.
///
/// Returns `None` when `Next` is not cleanly decomposable into guards + total
/// assignments ([`DeadlockFreedom::Undecomposable`] — disjunctive/opaque Next). The
/// certificate path then REFUSES to certify (fail-closed): we will not claim
/// deadlock-freedom for a Next whose enabling condition we cannot extract.
fn enabled_of_next(
    next_expanded: &Spanned<Expr>,
    state_var_names: &[String],
    funcsym_vars: &std::collections::HashSet<String>,
) -> Option<Spanned<Expr>> {
    // T2 widening 1 + 5: normalize Next to a disjunction of conjunctive actions
    // (DNF distribution of top-level `And` over nested `Or`, plus unprimed-
    // condition ITE lifting), STRICTLY fail-closed at [`DNF_CAP`]. The
    // normalization is used ONLY here, for the Enabled derivation — the
    // obligations keep the ORIGINAL Next AST (see
    // [`discharge_all_n_obligations_with_proofs`]), so there is no blowup or
    // skolem-naming drift on the consecution side.
    //
    // DISJUNCTIVE Next (`⋁ A_i`): `Enabled(⋁ A_i) = ⋁ Enabled(A_i)`, and each conjunctive action
    // `A_i` contributes `Enabled(A_i) = ⋀ (its guards)` (TRUE if unguarded). This is the multi-action
    // generalization of the single-action guard extraction below — a terminating spec's `Next` is a
    // disjunction of actions, so without it the deadlock-freedom / `Enabled` obligation cannot be
    // formed (fail-closed). Each disjunct must itself be a clean conjunctive action (total
    // assignment + unprimed guards); any opaque disjunct declines the whole analysis.
    let disjuncts = normalize_enabled_disjuncts(next_expanded, DNF_CAP)?;
    if disjuncts.len() > 1 {
        let mut enabled_disjuncts = Vec::with_capacity(disjuncts.len());
        for d in &disjuncts {
            match analyze_deadlock_freedom(d, state_var_names, funcsym_vars) {
                DeadlockFreedom::Decomposed(guards) => enabled_disjuncts
                    .push(conjoin(guards).unwrap_or_else(|| Spanned::dummy(Expr::Bool(true)))),
                DeadlockFreedom::Undecomposable => return None,
            }
        }
        // `⋁_i Enabled(A_i)` (an unguarded disjunct makes the whole thing TRUE).
        let mut result = enabled_disjuncts.pop()?;
        while let Some(e) = enabled_disjuncts.pop() {
            result = Spanned::dummy(Expr::Or(Box::new(e), Box::new(result)));
        }
        return Some(result);
    }
    match analyze_deadlock_freedom(disjuncts.first()?, state_var_names, funcsym_vars) {
        DeadlockFreedom::Decomposed(guards) => {
            Some(conjoin(guards).unwrap_or_else(|| Spanned::dummy(Expr::Bool(true))))
        }
        DeadlockFreedom::Undecomposable => None,
    }
}

/// FAIL-CLOSED cap on the number of Enabled-derivation disjuncts produced by
/// DNF distribution + ITE lifting (T2 rail: beyond the cap the analysis returns
/// `None` == Undecomposable — NEVER truncates; a truncated `⋁ Enabled` happens
/// to be conservative for the deadlock leg alone, but any other consumer of a
/// truncated distributed Next would be unsound, so the blanket rule stays).
///
/// Derivation of 64: census max top-level disjunct count is 10 (btree); the
/// worst in-scope distributed shape (Moving_Cat_Puzzle) is 2 (inner or) x 2
/// (ITE lift) x 2 (Observe branches) = 8; 64 = max-observed x ITE-doubling
/// headroom.
const DNF_CAP: usize = 64;

/// Normalize a Next predicate into the disjunct list the Enabled derivation
/// consumes: distribute top-level `And` over nested `Or` (deterministic,
/// left-to-right — a pure propositional identity on the POSITIVE Next), then
/// lift `IF g THEN a ELSE b` nodes with UNPRIMED (state/constant-determined)
/// conditions leftmost-first to fixpoint:
///   `D[ITE(g,a,b)]  ==  (g /\ D[a]) \/ (~g /\ D[b])`
/// — exact if-elimination given `g`'s value is state-determined; the `~g`
/// branch enters Enabled as an ordinary unprimed guard. Primed conditions are
/// NOT lifted (assignment-order semantics) and decline later as Opaque. Both
/// rewrites share the fail-closed `cap`: `None` == Undecomposable, never
/// truncate.
fn normalize_enabled_disjuncts(
    next: &Spanned<Expr>,
    cap: usize,
) -> Option<Vec<Spanned<Expr>>> {
    let mut work: Vec<Spanned<Expr>> = vec![next.clone()];
    loop {
        let mut changed = false;
        let mut out: Vec<Spanned<Expr>> = Vec::with_capacity(work.len());
        for d in work {
            // Distribute And-over-Or on this disjunct (left-to-right).
            let parts = distribute_dnf(&d, cap)?;
            if parts.len() > 1 {
                changed = true;
            }
            for p in parts {
                // Lift the leftmost unprimed-condition ITE, if any.
                if let Some((g, with_then, with_else)) = lift_first_unprimed_ite(&p) {
                    changed = true;
                    let not_g = Spanned::dummy(Expr::Not(Box::new(g.clone())));
                    out.push(Spanned::dummy(Expr::And(
                        Box::new(g),
                        Box::new(with_then),
                    )));
                    out.push(Spanned::dummy(Expr::And(
                        Box::new(not_g),
                        Box::new(with_else),
                    )));
                } else {
                    out.push(p);
                }
                if out.len() > cap {
                    return None;
                }
            }
        }
        work = out;
        if !changed {
            return Some(work);
        }
    }
}

/// Distribute top-level `And` over nested `Or` into a disjunct list —
/// deterministic left-to-right, exact propositional identity. Any node other
/// than `And`/`Or` is an atom (no distribution inside quantifiers, ITE
/// branches, negations, ...). FAIL-CLOSED: more than `cap` disjuncts => `None`.
fn distribute_dnf(e: &Spanned<Expr>, cap: usize) -> Option<Vec<Spanned<Expr>>> {
    match &e.node {
        Expr::Or(a, b) => {
            let mut left = distribute_dnf(a, cap)?;
            let right = distribute_dnf(b, cap)?;
            left.extend(right);
            if left.len() > cap {
                return None;
            }
            Some(left)
        }
        Expr::And(a, b) => {
            let la = distribute_dnf(a, cap)?;
            let lb = distribute_dnf(b, cap)?;
            let mut out = Vec::with_capacity(la.len().saturating_mul(lb.len()));
            for x in &la {
                for y in &lb {
                    out.push(Spanned::dummy(Expr::And(
                        Box::new(x.clone()),
                        Box::new(y.clone()),
                    )));
                    if out.len() > cap {
                        return None;
                    }
                }
            }
            Some(out)
        }
        _ => Some(vec![e.clone()]),
    }
}

/// Folder that replaces the FIRST (deterministic `ExprFold` traversal order)
/// `IF g THEN a ELSE b` node with an UNPRIMED condition by its `then` or `else`
/// branch — NEVER descending under a binder (`∃`/`∀`/`CHOOSE`/`LET`/function
/// constructs), where `g` could mention a bound variable that is NOT
/// state-determined (and whose extraction would leak the binder's scope; this
/// also keeps the F1 `natMin`-in-range recognition untouched). Records the
/// lifted condition `g`. The SAME folder logic runs for both branches, so both
/// rebuilds target the SAME node.
struct LiftFirstIte {
    take_then: bool,
    lifted_cond: Option<Spanned<Expr>>,
}

impl tla_core::ExprFold for LiftFirstIte {
    fn fold_expr(&mut self, e: Spanned<Expr>) -> Spanned<Expr> {
        if self.lifted_cond.is_some() {
            return e;
        }
        match &e.node {
            Expr::If(cond, then_b, else_b) if !contains_primed(&cond.node) => {
                self.lifted_cond = Some((**cond).clone());
                if self.take_then {
                    (**then_b).clone()
                } else {
                    (**else_b).clone()
                }
            }
            // Binder nodes: opaque to the lift (see doc above).
            Expr::Exists(..)
            | Expr::Forall(..)
            | Expr::Choose(..)
            | Expr::SetFilter(..)
            | Expr::SetBuilder(..)
            | Expr::FuncDef(..)
            | Expr::Lambda(..)
            | Expr::Let(..) => e,
            _ => Spanned {
                node: self.fold_expr_inner(e.node),
                span: e.span,
            },
        }
    }
}

/// Lift the first (deterministic traversal order) unprimed-condition ITE out of
/// `e`: returns `(g, e[ITE := a], e[ITE := b])`, or `None` when no liftable ITE
/// exists.
fn lift_first_unprimed_ite(
    e: &Spanned<Expr>,
) -> Option<(Spanned<Expr>, Spanned<Expr>, Spanned<Expr>)> {
    use tla_core::ExprFold as _;
    let mut ft = LiftFirstIte {
        take_then: true,
        lifted_cond: None,
    };
    let with_then = ft.fold_expr(e.clone());
    let g = ft.lifted_cond?;
    let mut fe = LiftFirstIte {
        take_then: false,
        lifted_cond: None,
    };
    let with_else = fe.fold_expr(e.clone());
    debug_assert!(
        fe.lifted_cond.is_some(),
        "the second lift pass must find the same ITE"
    );
    Some((g, with_then, with_else))
}

/// Flatten a top-level `\/` tree into its disjuncts (left-to-right).
fn flatten_or<'a>(expr: &'a Spanned<Expr>, out: &mut Vec<&'a Spanned<Expr>>) {
    if let Expr::Or(a, b) = &expr.node {
        flatten_or(a, out);
        flatten_or(b, out);
    } else {
        out.push(expr);
    }
}

/// How a single action (Next disjunct) constrains a given string state var's
/// NEXT value, for the F1 structural-deadlock closure analysis.
enum StringAssign {
    /// `s' = "lit"` — assigns a concrete literal (contributes to the universe).
    Literal(String),
    /// `UNCHANGED s` (directly or inside a tuple) — keeps the current value.
    Unchanged,
    /// `s'` unconstrained or assigned a non-literal expression — closure of the
    /// literal universe cannot be established, so decline the structural path.
    Bad,
}

/// Extract the string literal `s` is initialized to in `Init` (`s = "lit"`), if
/// any — the seed of the reachable literal universe.
fn init_string_literal(init: &Spanned<Expr>, s: &str) -> Option<String> {
    let mut conjuncts = Vec::new();
    flatten_and(init, &mut conjuncts);
    for c in conjuncts {
        if let Expr::Eq(lhs, rhs) = &c.node {
            let lhs_is_s = matches!(&lhs.node, Expr::Ident(n, _) | Expr::StateVar(n, ..) if n == s);
            if lhs_is_s {
                if let Expr::String(lit) = &rhs.node {
                    return Some(lit.clone());
                }
            }
        }
    }
    None
}

/// Determine how one action (a single Next disjunct) constrains `s'`.
fn action_string_assign(action: &Spanned<Expr>, s: &str) -> StringAssign {
    let mut conjuncts = Vec::new();
    flatten_and(action, &mut conjuncts);
    let mut found: Option<StringAssign> = None;
    for c in conjuncts {
        match &c.node {
            // `s' = "lit"` (or `s' = <non-literal>`).
            Expr::Eq(lhs, rhs) => {
                if primed_var_name(&lhs.node).as_deref() == Some(s) {
                    match &rhs.node {
                        Expr::String(lit) => found = Some(StringAssign::Literal(lit.clone())),
                        _ => return StringAssign::Bad,
                    }
                }
            }
            // `UNCHANGED s` / `UNCHANGED <<.., s, ..>>`.
            Expr::Unchanged(arg) => {
                if let Some(names) = unchanged_var_names(&arg.node) {
                    if names.iter().any(|n| n == s) {
                        found = Some(StringAssign::Unchanged);
                    }
                }
            }
            _ => {}
        }
    }
    // An action that never constrains `s'` leaves it unconstrained — not closed.
    found.unwrap_or(StringAssign::Bad)
}

/// `true` iff `guards` is EXACTLY the single equality `s = "lit"` (either
/// operand order) — i.e. this action is enabled precisely when `s = "lit"`, with
/// no further constraint, so it unconditionally covers that literal.
fn guards_are_exactly_eq(guards: &[Spanned<Expr>], s: &str, lit: &str) -> bool {
    if guards.len() != 1 {
        return false;
    }
    let Expr::Eq(a, b) = &guards[0].node else {
        return false;
    };
    let matches_pair = |var: &Spanned<Expr>, val: &Spanned<Expr>| {
        matches!(&var.node, Expr::Ident(n, _) | Expr::StateVar(n, ..) if n == s)
            && matches!(&val.node, Expr::String(l) if l == lit)
    };
    matches_pair(a, b) || matches_pair(b, a)
}

/// F1 STRUCTURAL deadlock-freedom for a string-enum "clock" variable.
///
/// A spec like glowingRaccoon is driven by a string var `s` (`tee`) whose actions
/// partition on `s = "lit"` and always reassign `s` to another literal. Such a
/// spec is deadlock-free on every REACHABLE state — but the SMT obligation
/// `J /\ ~Enabled` needs enum/disjunctive reasoning over `s`'s finite domain,
/// outside AY's strict single-equality Farkas fragment. This recognizes the
/// structural argument that needs NO solver:
///   1. CLOSURE — every action assigns `s' = "lit"` (or `UNCHANGED s`), so from
///      `Init`'s literal `s` stays within a finite literal universe `L`
///      (inductive by construction, no obligation needed);
///   2. COVERAGE — every literal `ℓ ∈ L` is covered by some action whose guard is
///      EXACTLY `s = ℓ` and which totally assigns EVERY state var (so it is
///      enabled whenever `s = ℓ`, unconditionally).
/// Under (1)+(2) every reachable state (`s ∈ L`) has an enabled action, so the
/// spec is deadlock-free and `Enabled` is discharged as the structural `TRUE`
/// marker. FAIL-CLOSED: any gap (an action assigning `s'` non-literally, an
/// uncovered literal, an undecomposable disjunct) returns `false` and the caller
/// falls back to the SMT `~Enabled` obligation (which then honestly declines).
///
/// SOUNDNESS of the `Enabled ≡ TRUE` marker: it asserts no reachable J-state
/// deadlocks. The safety obligations pin `J` to the reachable states; closure
/// pins `s ∈ L` there; coverage makes every `s ∈ L` state enabled. A
/// non-reachable state with `s ∉ L` is irrelevant to deadlock-FREEDOM (the spec
/// never reaches it). The verify side re-runs this SAME recognition, so a forged
/// structural claim on a non-covered spec is rejected (Enabled ≠ TRUE there).
fn string_enum_deadlock_free(
    init: &Spanned<Expr>,
    next: &Spanned<Expr>,
    var_sorts: &[(String, TlaSort)],
    var_names: &[String],
) -> bool {
    // Per-disjunct guard lists; bail if any disjunct is not cleanly decomposable.
    let mut disjuncts: Vec<&Spanned<Expr>> = Vec::new();
    flatten_or(next, &mut disjuncts);
    let mut disjunct_guards: Vec<Vec<Spanned<Expr>>> = Vec::with_capacity(disjuncts.len());
    for d in &disjuncts {
        match analyze_deadlock_freedom(d, var_names, &std::collections::HashSet::new()) {
            DeadlockFreedom::Decomposed(guards) => disjunct_guards.push(guards),
            DeadlockFreedom::Undecomposable => return false,
        }
    }

    // Try each STRING state var as the enum clock.
    for (name, sort) in var_sorts {
        if *sort != TlaSort::String {
            continue;
        }
        let Some(init_lit) = init_string_literal(init, name) else {
            continue;
        };
        // Closure: collect the reachable literal universe; every action must
        // assign a literal or leave the var UNCHANGED.
        let mut universe: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
        universe.insert(init_lit);
        let mut closed = true;
        for d in &disjuncts {
            match action_string_assign(d, name) {
                StringAssign::Literal(lit) => {
                    universe.insert(lit);
                }
                StringAssign::Unchanged => {}
                StringAssign::Bad => {
                    closed = false;
                    break;
                }
            }
        }
        if !closed {
            continue;
        }
        // Coverage: every literal is unconditionally enabled by some action.
        let all_covered = universe.iter().all(|lit| {
            disjunct_guards
                .iter()
                .any(|guards| guards_are_exactly_eq(guards, name, lit))
        });
        if all_covered {
            return true;
        }
    }
    false
}

/// Build candidate STRENGTHENING predicates for the FIX-B certificate from the
/// integer literals in Init/Next/Safety. Returns a list of whole-state
/// candidate predicates (each an `AND` over int state vars) to try, in
/// increasing strength order; the caller conjoins each with the invariant `S`
/// and tests whether the result is inductive.
///
/// Candidates (each spanning ALL integer state vars together — pipeline-style
/// inductiveness needs every var bounded simultaneously because one var's bound
/// is established from another's via the induction hypothesis):
///   1. lower bounds `gMin <= v` for every int var;
///   2. intervals `gMin <= v /\ v <= gMax` for every int var.
/// A loose/wrong candidate is harmless: it simply fails the inductiveness gate
/// (which the caller still runs on the conjoined J) and the certificate falls
/// through — never an unsound Safe.
fn derive_strengthening_candidates(
    var_sorts: &[(String, TlaSort)],
    init_expanded: &Spanned<Expr>,
    next_expanded: &Spanned<Expr>,
    safety_expanded: &Spanned<Expr>,
) -> Vec<Spanned<Expr>> {
    let int_vars: Vec<&str> = var_sorts
        .iter()
        .filter(|(_, s)| matches!(s, TlaSort::Int))
        .map(|(n, _)| n.as_str())
        .collect();
    if int_vars.is_empty() {
        return Vec::new();
    }

    let mut collector = IntLiteralCollector::default();
    collector.walk_expr(&init_expanded.node);
    collector.walk_expr(&next_expanded.node);
    collector.walk_expr(&safety_expanded.node);
    let (Some(&g_min), Some(&g_max)) =
        (collector.lits.iter().min(), collector.lits.iter().max())
    else {
        return Vec::new();
    };
    if g_min > g_max {
        return Vec::new();
    }

    let mut candidates = Vec::new();

    // Candidate 1: lower bounds for all int vars.
    let lowers: Vec<Spanned<Expr>> = int_vars
        .iter()
        .map(|v| build_var_lower_bound_expr(v, g_min))
        .collect();
    if let Some(c) = conjoin(lowers) {
        candidates.push(c);
    }

    // Candidate 2: full intervals for all int vars (only adds value over (1)
    // when the safe upper bound also matters for inductiveness).
    let intervals: Vec<Spanned<Expr>> = int_vars
        .iter()
        .map(|v| build_var_interval_expr(v, g_min, g_max))
        .collect();
    if let Some(c) = conjoin(intervals) {
        candidates.push(c);
    }

    candidates
}

/// The verdict of an attempted inductive-safety certificate.
pub(crate) enum InductiveSafetyCertificate {
    /// Proven safe AND (when required) deadlock-free. Caller returns Success(Safe).
    Safe,
    /// Could not complete the proof; caller must fall through to BFS.
    FallThrough,
}

/// The serializable, re-checkable CONTENT of a discharged inductive-safety
/// proof: the inductive invariant `J` and the `Init`/`Next`/`Safety` predicates
/// (all as expanded TLA ASTs) plus the inferred variable sorts.
///
/// Today the model checker proves `J`, returns `Safe`, and DISCARDS `J` — the
/// proof exists only as a transient solver fact. Capturing it here is the first
/// step of certifying verification: a third party (or a diverse engine) can
/// re-discharge the three Hoare obligations `Init => J`, `J /\ Next => J'`, and
/// `J => Safety` from this data alone, WITHOUT trusting the checker that
/// produced it.
#[derive(Clone)]
pub(crate) struct InductiveProof {
    /// The proven inductive invariant `J` (`Safety`, or `Safety /\ C`).
    pub(crate) invariant_j: Spanned<Expr>,
    /// `Init` predicate (expanded).
    pub(crate) init: Spanned<Expr>,
    /// `Next` action (expanded).
    pub(crate) next: Spanned<Expr>,
    /// `Safety` = conjunction of all configured invariants (expanded).
    pub(crate) safety: Spanned<Expr>,
    /// Inferred sorts for each state variable (the SMT signature).
    pub(crate) var_sorts: Vec<(String, TlaSort)>,
}

/// Prove inductive safety for CERTIFICATE EMISSION (certifying verification, P2).
///
/// This does NOT require the divergence trigger
/// (unguarded self-accumulating `Next`): it attempts the inductive-safety proof for
/// ANY spec whose safety conjunction is 1-inductive — directly or after interval
/// strengthening — widening the certifiable class well beyond unbounded counters.
/// It proves SAFETY only (not deadlock-freedom), which is exactly what a safety
/// certificate claims, and it is NEVER used to skip a BFS verdict (only to emit a
/// re-checkable certificate).
pub(crate) fn prove_inductive_safety_for_cert(
    ctx: &EvalCtx,
    config: &Config,
    state_var_names: &[Arc<str>],
) -> Option<InductiveProof> {
    try_inductive_safety_certificate_inner(ctx, config, state_var_names, false, true)
        .ok()
        .flatten()
}

/// Re-discharge a STRICT inductive-safety certificate for the cooperative lane's spec,
/// returning [`CertificateVerification::Verified`] iff every Hoare obligation is UNSAT
/// AND strict-verified (the same gate as `cert.rs:495`). FAIL-CLOSED: any failure — the
/// proof is not re-derivable at the AST level, or any obligation is not strict-verified —
/// yields [`CertificateVerification::MissingVerifier`].
///
/// This is what makes a fused symbolic `Satisfied` PROOF-PRODUCING rather than trusted:
/// the cooperative verdict resolves (via `publish_analytical`) ONLY when the lane's proven
/// invariant re-discharges a strict, independently re-checkable certificate. An
/// unverifiable "proof" leaves the slot unresolved, so BFS stays authoritative. Cost: one
/// inductive-safety solve + the bounded per-obligation proof solves — only on the (rare)
/// Safe arm, and only when `symbolic_safety_proof_covers_all_obligations` already holds.
pub(crate) fn strict_safety_certificate_state(
    ctx: &EvalCtx,
    config: &Config,
    vars: &[Arc<str>],
) -> CertificateVerification {
    let Some(proof) = prove_inductive_safety_for_cert(ctx, config, vars) else {
        return CertificateVerification::MissingVerifier;
    };
    // The publish site is already behind `symbolic_safety_proof_covers_all_obligations`
    // (⇒ deadlock-freedom is not an obligation), so the enabled predicate is TRUE and the
    // 4th obligation (J ⇒ Enabled) is the structural Verified marker; the strict gate
    // reduces to the three SMT obligations.
    let enabled = Spanned {
        node: Expr::Bool(true),
        span: proof.invariant_j.span,
    };
    match discharge_obligations_with_proofs(
        &proof.var_sorts,
        &proof.init,
        &proof.next,
        &proof.safety,
        &proof.invariant_j,
        &enabled,
        BmcConfig::default().solve_timeout,
    ) {
        Ok(obs) if !obs.is_empty() && obs.iter().all(|o| o.unsat && o.strict_verified) => {
            // Proof-carrying: emit a trust-ir Module recording the inductive-safety obligations
            // with their ay (Alethe) SMT proofs as evidence, at the honest Discharged tier. This
            // fills the `trust_ir::Module` proof conduit (Verified -> structured, lineage-ready
            // obligations) without changing the verdict gate; reaching the kernel-checkable
            // Certified/CleanCic tier is a future step (de-Bruijn CIC encoding of the
            // clean-supported proofs). See docs/cleancic-dependent-types-wiring-2026-06-24.md.
            // An obligation is reflexive (`φ ⇒ φ`) when its antecedent denotes its consequent:
            // initiation `Init ⇒ J` (reflexive iff Init ≡ J), safety `J ⇒ Safety` (reflexive iff
            // J ≡ Safety, the common 1-inductive case); consecution is never reflexive. The Clean
            // kernel certifies these via the FAITHFUL construction `Π(vars). embed(ante)→embed(cons)`
            // proved by the identity term, accepting it ONLY when `embed(ante) ≡ embed(cons)` by
            // conversion.
            //
            // SEMANTIC reflexivity HINT (CONSTANT-concretized, span-insensitive `pretty_expr`):
            // identifies which obligations to ATTEMPT as `φ ⇒ φ`. It is only a hint — the Clean
            // kernel is the ARBITER: `build_safety_certificate_module` certifies a reflexive
            // obligation by checking the identity at `Π(vars). embed(ante)→embed(cons)`, which the
            // kernel accepts ONLY when `embed(ante) ≡ embed(cons)`. So a `denote_eq` false-positive
            // yields no kernel cert (Discharged), never a false `Certified`.
            let denote_eq = |a: &Spanned<Expr>, b: &Spanned<Expr>| {
                tla_core::pretty_expr(&a.node) == tla_core::pretty_expr(&b.node)
            };
            let reflexive_pairs: Vec<Option<(Expr, Expr)>> = obs
                .iter()
                .map(|ob| match ob.name {
                    "initiation" if denote_eq(&proof.init, &proof.invariant_j) => {
                        Some((proof.init.node.clone(), proof.invariant_j.node.clone()))
                    }
                    "safety" if denote_eq(&proof.invariant_j, &proof.safety) => {
                        Some((proof.invariant_j.node.clone(), proof.safety.node.clone()))
                    }
                    _ => None,
                })
                .collect();
            let certificate =
                build_safety_certificate_module("ty.inductive_safety", &obs, &reflexive_pairs);
            log_safety_certificate_summary(&certificate, &obs);
            CertificateVerification::Verified
        }
        _ => CertificateVerification::MissingVerifier,
    }
}

/// Build a trust-ir proof-carrying [`trust_ir::Module`] for an inductive-safety certificate: the
/// inductive invariant `J` plus the three Hoare obligations (initiation / consecution / safety),
/// each at [`trust_ir::ProofStatus::Discharged`] and (for the obligations) bound to a
/// [`trust_ir::ProofCertificate`] carrying the ay SMT proof (Alethe text) as
/// [`trust_ir::ProofEvidence::SmtProof`].
///
/// This is TY's `Verified` result wired into Trust's proof ladder at the HONEST tier — trust base
/// = the ay SMT solver. It is deliberately NOT the kernel-checkable `Certified` / `CleanCic` tier:
/// that requires re-encoding the (clean-supported) proofs as de-Bruijn CIC terms a small kernel can
/// re-check (the trust-base-shrinking future step in
/// `docs/cleancic-dependent-types-wiring-2026-06-24.md`).
pub(crate) fn build_safety_certificate_module(
    module_name: &str,
    obligations: &[ObligationProof],
    // Per-obligation reflexive `(ante, cons)` formula pair — `Some` iff the obligation is `φ ⇒ φ`
    // (`Init≡J` / `J≡Safety`). The pair is used to FAITHFULLY kernel-bind reflexivity (the kernel
    // checks the identity at `Π(vars). embed(ante)→embed(cons)`) and to bind the lineage to the
    // actual formula (not a spec-independent slot string).
    reflexive_pairs: &[Option<(Expr, Expr)>],
) -> trust_ir::Module {
    use trust_ir::value::ProofId;
    use trust_ir::{
        ObligationKind, ProofCertificate, ProofEvidence, ProofFormula, ProofObligation, ProofStatus,
    };
    let mut module = trust_ir::Module::new(module_name);

    // The inductive invariant J — the witness the three Hoare obligations are about. It is
    // Discharged BECAUSE those obligations discharge (its evidence is the inductive argument
    // itself, i.e. obligations 1..=3), so it carries no separate certificate.
    module.proof_obligations.push(ProofObligation::new(
        ProofId::new(0),
        ObligationKind::LoopInvariant,
        ProofStatus::Discharged,
        "inductive invariant J (re-discharged from Init/Next/Safety on a fresh solver)",
    ));

    // The three Hoare obligations, each carrying its ay (Alethe) SMT proof as evidence.
    for (i, ob) in obligations.iter().enumerate() {
        let id = ProofId::new((i + 1) as u32);
        let smt_discharged = ob.unsat && ob.strict_verified;
        let pair = reflexive_pairs.get(i).and_then(|p| p.as_ref());
        // The formula records WHAT is proven; for a reflexive obligation it includes the ACTUAL
        // `ante ⇒ cons` (rendered), so the CleanCic lineage digest is SPEC-SPECIFIC — a certificate
        // minted for one spec is not lineage-valid for another (closes the slot-string replay hole).
        let desc = match pair {
            Some((ante, cons)) => format!(
                "inductive-safety obligation: {} :: {} => {}",
                ob.name,
                tla_core::pretty_expr(ante),
                tla_core::pretty_expr(cons),
            ),
            None => format!("inductive-safety obligation: {}", ob.name),
        };
        let formula = ProofFormula::new("ty.inductive_safety.v1", desc.clone());

        // Try to PROMOTE a reflexive (`φ ⇒ φ`), SMT-discharged obligation to the kernel-checkable
        // `Certified` tier — but ONLY when the Clean kernel accepts the FAITHFUL proof at
        // `Π(vars). embed(ante)→embed(cons)` (fail-closed; gated behind `clean-cic`). The kernel —
        // not an off-kernel string compare — decides reflexivity, so a misclassification yields no
        // cert.
        // `status`/`clean_cic` are mutated only under `clean-cic` (the kernel-promotion block below).
        #[allow(unused_mut)]
        let mut status = if smt_discharged {
            ProofStatus::Discharged
        } else {
            ProofStatus::Failed
        };
        #[allow(unused_mut)]
        let mut clean_cic: Option<ProofEvidence> = None;
        let _ = pair; // used only under `clean-cic`
        #[cfg(feature = "clean-cic")]
        {
            if smt_discharged {
                if let Some((ante, cons)) = pair {
                    if let Some(bytes) = crate::cleancic::certify_reflexive_faithful(ante, cons) {
                        // NOTE: TY runs the Clean kernel HERE at mint time (certify_reflexive_faithful
                        // → kernel_accepts). But the downstream trust-ir Module validate gate
                        // (`obligation_has_matching_clean_cic`) enforces ONLY lineage-binding + a
                        // non-empty term — it does NOT re-run the kernel. A consumer wanting the
                        // kernel-rechecked guarantee must run a kernel re-checker (TY provides
                        // `cleancic::verify_reflexive_faithful`, driven by cert.rs `verify_reflexive_leg`
                        // / `verify_leg_k`); a validate-clean Module alone is not kernel-re-checked.
                        let candidate = ProofObligation::new(
                            id,
                            ObligationKind::TemporalSafety,
                            ProofStatus::Certified,
                            desc.clone(),
                        )
                        .with_formula(formula.clone());
                        clean_cic = Some(ProofEvidence::CleanCic {
                            term: bytes,
                            context: Vec::new(),
                            lineage: trust_ir::clean_cic_lineage_digest(&candidate),
                            // No named-theorem anchor directive: TY's terms are anonymous
                            // per-obligation proofs re-checked by TY's own Leg-K
                            // (`verify_leg_k` → `verify_reflexive_faithful`), per the NOTE
                            // above — not by trust-ir's fixed-anchor re-checker.
                            kernel_recheck: None,
                        });
                        status = ProofStatus::Certified;
                    }
                }
            }
        }

        module.proof_obligations.push(
            ProofObligation::new(id, ObligationKind::TemporalSafety, status, desc)
                .with_formula(formula),
        );

        // Always attach the literal ay SMT proof — prefer the portable, checker-only bundle
        // (re-checkable OFFLINE without re-running the solver); else the rendered Alethe text.
        let proof_bytes = ob
            .bundle_json
            .clone()
            .unwrap_or_else(|| ob.alethe.clone())
            .into_bytes();
        module.proof_certificates.push(ProofCertificate {
            obligation: id,
            prover: "ty.ay".to_string(),
            evidence: ProofEvidence::SmtProof(proof_bytes),
        });
        // When the kernel certified it, ALSO attach the CleanCic certificate (the Certified tier:
        // trust base = the Clean kernel, not the SMT solver).
        if let Some(ev) = clean_cic {
            module.proof_certificates.push(ProofCertificate {
                obligation: id,
                prover: "ty.clean-cic".to_string(),
                evidence: ev,
            });
        }
    }
    module
}

/// Emit a one-line provenance summary for a freshly-built inductive-safety certificate Module,
/// honestly distinguishing the achieved `Discharged` (SMT) tier from the `Certified` (kernel) tier.
fn log_safety_certificate_summary(module: &trust_ir::Module, obligations: &[ObligationProof]) {
    let clean = obligations.iter().filter(|o| o.clean_supported).count();
    let certified = module
        .proof_obligations
        .iter()
        .filter(|o| o.status == trust_ir::ProofStatus::Certified)
        .count();
    eprintln!(
        "[ty-certificate] inductive-safety proof Module: {} obligation(s), {} certificate(s); \
         {certified} kernel-CERTIFIED (trust base: Clean CIC kernel), the rest Discharged via ay SMT. \
         {clean}/{} clean-supported (CleanCic-ready); non-reflexive obligations await CIC reconstruction.",
        module.proof_obligations.len(),
        module.proof_certificates.len(),
        obligations.len(),
    );
}

/// Attempt the SOUND inductive infinite-state safety certificate (FIX B).
///
/// `state_var_names` are the declared state variables (from the checker's own
/// var list). `check_deadlock` mirrors `config.check_deadlock`.
///
/// Returns [`InductiveSafetyCertificate::Safe`] ONLY when a COMPLETE proof is
/// discharged (every configured invariant is inductive — directly or after
/// interval strengthening — AND, when `check_deadlock`, Next is deadlock-free
/// under the inductive invariant). On any failure, Unknown, or non-decomposable
/// structure it returns [`FallThrough`] and the caller runs unchanged BFS.
///
/// SOUNDNESS: see the module-level comment. This function NEVER returns `Safe`
/// for an unsafe, deadlocking, or not-provably-inductive spec.
pub(crate) fn try_inductive_safety_certificate(
    ctx: &EvalCtx,
    config: &Config,
    state_var_names: &[Arc<str>],
    check_deadlock: bool,
) -> InductiveSafetyCertificate {
    match try_inductive_safety_certificate_inner(ctx, config, state_var_names, check_deadlock, false)
    {
        Ok(Some(_proof)) => InductiveSafetyCertificate::Safe,
        Ok(None) => InductiveSafetyCertificate::FallThrough,
        // Any error in setup/translation is treated as INCONCLUSIVE: fall
        // through to BFS. Never let a certificate error change a verdict.
        Err(_) => InductiveSafetyCertificate::FallThrough,
    }
}

fn try_inductive_safety_certificate_inner(
    ctx: &EvalCtx,
    config: &Config,
    state_var_names: &[Arc<str>],
    check_deadlock: bool,
    for_certification: bool,
) -> Result<Option<InductiveProof>, BmcError> {
    let debug = debug_ay_bmc_enabled();

    // Need invariants to prove anything about, and Init/Next to reason over.
    if config.invariants.is_empty() {
        return Ok(None);
    }

    let symbolic_ctx =
        ay_shared::symbolic_ctx_with_config(ctx, config).map_err(BmcError::MissingSpec)?;
    if state_var_names.is_empty() {
        return Ok(None);
    }

    let resolved =
        ay_shared::resolve_init_next(config, &symbolic_ctx).map_err(BmcError::MissingSpec)?;
    let init_expr = ay_shared::get_operator_body(&symbolic_ctx, &resolved.init)
        .map_err(BmcError::MissingSpec)?;
    let next_expr = ay_shared::get_operator_body(&symbolic_ctx, &resolved.next)
        .map_err(BmcError::MissingSpec)?;

    let init_expanded = expand_operators_for_chc(&symbolic_ctx, &init_expr, false);
    let next_expanded = expand_operators_for_chc(&symbolic_ctx, &next_expr, true);

    // DIVERGENCE / FINITENESS TRIGGER (gates WHEN we attempt the proof; not a
    // soundness obligation). Two conditions must BOTH hold:
    //
    //   (i) Next contains a SELF-ACCUMULATING assignment `v' = v + e` / `v' = v - e`
    //       — the only shape that can make the reachable space unbounded.
    //
    //  (ii) Next decomposes into total assignments with NO guards, i.e.
    //       Enabled(Next) == TRUE (`analyze_deadlock_freedom` => Decomposed with
    //       an EMPTY guard list).
    //
    // Why require (ii) EMPTY guards (the finiteness filter): a guarded
    // accumulator like `count < 3 /\ count' = count + 1` is actually BOUNDED —
    // BFS terminates on it, and firing the certificate would only replace its
    // real `states_found` with the proof's 0 while providing no benefit. An
    // UNGUARDED self-accumulating assignment (`x' = x + 1` with no bounding
    // guard) genuinely diverges, so BFS would hang and the certificate is the
    // only terminating path. Restricting to the no-guard case therefore (a)
    // leaves every finite/guarded spec on the unchanged BFS path with identical
    // stats, and (b) makes deadlock-freedom TRIVIAL: with no guards
    // Enabled(Next) == TRUE, so the spec is deadlock-free unconditionally
    // regardless of `config.check_deadlock`. Disjunctive / conditional / partial
    // Next => Undecomposable => fall through. This is strictly MORE conservative
    // than a general guard-implication proof and is sound by construction.
    // Divergence/finiteness trigger — BFS-SKIP mode ONLY. The certificate only
    // REPLACES a BFS run when BFS would actually diverge (unguarded self-
    // accumulating Next, so Enabled(Next)==TRUE => unbounded AND deadlock-free).
    //
    // In CERTIFICATION mode (`for_certification`) we are EMITTING a re-checkable
    // safety certificate, not skipping a verdict, so we prove inductive SAFETY for
    // ANY spec — guarded, finite, disjunctive — whose safety conjunction is
    // 1-inductive (directly or after interval strengthening). Deadlock-freedom is
    // irrelevant to a safety claim (`J` inductive /\ `J` => Safety establishes
    // Safety on every reachable state regardless of deadlock), so the trigger is
    // skipped. This widens the certifiable class well beyond unguarded counters.
    if !for_certification {
        // A configured state/action CONSTRAINT bounds BFS exploration exactly
        // like an in-Next guard does (TLC ch.14 semantics: constraint-violating
        // states are checked but not expanded), so "Next self-accumulates" no
        // longer implies BFS diverges — the constraint IS the bounding guard,
        // just applied at exploration level instead of inside Next. Leave
        // constrained specs on the unchanged BFS path: that preserves the
        // TLC-parity semantics and stats (`states_found` = explored count) the
        // user explicitly asked for by writing the constraint, per the same
        // rationale as the guarded-accumulator case below. Verdict-neutral:
        // falling through never changes a verdict, and BFS applies the
        // constraint exactly.
        if !config.constraints.is_empty() || !config.action_constraints.is_empty() {
            if debug {
                eprintln!(
                    "[ay-cert] state/action constraints configured => exploration is \
                     user-bounded (TLC parity); fall through to BFS"
                );
            }
            return Ok(None);
        }
        if !next_has_accumulating_arith(&next_expanded.node) {
            return Ok(None);
        }
        let names: Vec<String> = state_var_names.iter().map(|n| n.to_string()).collect();
        match analyze_deadlock_freedom(&next_expanded, &names, &std::collections::HashSet::new()) {
            DeadlockFreedom::Decomposed(guards) if guards.is_empty() => {
                if debug {
                    eprintln!(
                        "[ay-cert] trigger: self-accumulating + unguarded total Next (Enabled==TRUE)"
                    );
                }
            }
            _ => {
                if debug {
                    eprintln!(
                        "[ay-cert] Next is guarded or non-decomposable => finite/uncertain; fall through"
                    );
                }
                return Ok(None);
            }
        }
    }
    let _ = check_deadlock;

    // Build the full safety conjunction once for sort inference and for the
    // interval-bound candidate generation (literals come from Init/Next/Safety).
    let safety_expr = ay_shared::build_safety_conjunction(&symbolic_ctx, &config.invariants)
        .map_err(BmcError::TranslationError)?;
    let safety_expanded = expand_operators_for_chc(&symbolic_ctx, &safety_expr, false);

    let vars: Vec<Arc<str>> = state_var_names.to_vec();
    let var_sorts =
        ay_shared::infer_var_sorts(&vars, &init_expanded, &config.invariants, &symbolic_ctx);

    let timeout = BmcConfig::default().solve_timeout;

    // --- Obligation (A): SAFETY (all configured invariants) must be inductive. ---
    //
    // The induction target is the FULL safety conjunction `safety_expanded` =
    // AND of every configured invariant. Proving an inductive J with J => safety
    // establishes that EVERY configured invariant holds in every reachable state
    // (J inductive => J reachable-invariant; J => safety => each conjunct holds).
    //
    // We try, in increasing strength:
    //   (a) J = safety directly;
    //   (b) J = safety /\ C for each strengthening candidate C (whole-state lower
    //       bounds, then intervals). Since J is a CONJUNCTION that INCLUDES
    //       safety, J => safety is trivially valid — no separate implication
    //       check needed. We only have to prove J itself 1-inductive via the
    //       existing two-part gate (Init => J and J /\ Next => J').
    //
    // The proven J is retained as `inductive_hypothesis` for the deadlock check.
    let direct_ok = gate_is_inductive(
        &var_sorts,
        &init_expanded,
        &next_expanded,
        &safety_expanded,
        timeout,
    )?;

    let inductive_hypothesis: Spanned<Expr> = if direct_ok {
        if debug {
            eprintln!("[ay-cert] full safety conjunction is 1-inductive directly");
        }
        safety_expanded.clone()
    } else {
        // Strengthen with candidates until one proven-inductive J is found.
        let candidates = derive_strengthening_candidates(
            &var_sorts,
            &init_expanded,
            &next_expanded,
            &safety_expanded,
        );
        let mut proven: Option<Spanned<Expr>> = None;
        for cand in candidates {
            let j = Spanned::dummy(Expr::And(
                Box::new(safety_expanded.clone()),
                Box::new(cand),
            ));
            if gate_is_inductive(&var_sorts, &init_expanded, &next_expanded, &j, timeout)? {
                if debug {
                    eprintln!("[ay-cert] safety is inductive after strengthening");
                }
                proven = Some(j);
                break;
            }
        }
        match proven {
            Some(j) => j,
            None => {
                // Safety not inductive even after strengthening => cannot prove
                // => fall through to BFS (which finds the violation if any).
                if debug {
                    eprintln!(
                        "[ay-cert] safety not inductive even after strengthening; fall through"
                    );
                }
                return Ok(None);
            }
        }
    };

    // --- Obligation (B): DEADLOCK-FREEDOM. ---
    //
    // BFS-skip mode reaches here only past the trigger (unguarded total Next =>
    // Enabled(Next) == TRUE), so deadlock-freedom is structural there. CERTIFICATION
    // mode skips that trigger, so it must PROVE deadlock-freedom explicitly:
    // J => Enabled(Next), where Enabled(Next) is the conjunction of Next's guards.
    // If Next is not decomposable, or J does not imply Enabled (the spec can
    // deadlock — e.g. MODULE Dead: J=x<=3 does not imply Enabled=x<3 at x=3),
    // REFUSE to certify. A safety certificate must not be emitted for a spec that
    // can deadlock.
    if for_certification {
        let names: Vec<String> = state_var_names.iter().map(|n| n.to_string()).collect();
        let Some(enabled) = enabled_of_next(&next_expanded, &names, &std::collections::HashSet::new())
        else {
            if debug {
                eprintln!("[ay-cert] cert mode: Next not decomposable for Enabled(Next); declining");
            }
            return Ok(None);
        };
        let not_enabled = negate_normalized(&enabled);
        let j_implies_enabled = scratch_check_unsat(&var_sorts, 1, timeout, |t| {
            let j0 = t.translate_safety_at_step(&inductive_hypothesis, 0)?;
            t.assert(j0);
            let ne0 = t.translate_safety_at_step(&not_enabled, 0)?;
            t.assert(ne0);
            Ok(())
        })?;
        if !j_implies_enabled {
            if debug {
                eprintln!("[ay-cert] cert mode: J does not imply Enabled(Next) (can deadlock); declining");
            }
            return Ok(None);
        }
    }

    if debug {
        eprintln!("[ay-cert] inductive-safety certificate PROVEN — returning Safe");
    }
    // Capture the proof content (J + Init/Next/Safety + sorts) so it can be
    // serialized and independently re-checked, instead of discarding J.
    Ok(Some(InductiveProof {
        invariant_j: inductive_hypothesis,
        init: init_expanded,
        next: next_expanded,
        safety: safety_expanded,
        var_sorts,
    }))
}

/// Performance/divergence TRIGGER for the certificate (NOT soundness): does
/// `Next` contain a SELF-ACCUMULATING assignment `v' = v + e`, `v' = e + v`, or
/// `v' = v - e` — the canonical UNBOUNDED-growth shape that makes explicit BFS
/// non-terminating? Only when this fires (alongside ay + invariants) do we pay
/// the 2-4 SMT solves.
///
/// This is DELIBERATELY narrower than "any arithmetic on a var": a bounded
/// oscillator like `x' = 1 - x` (`Sub` with `x` as the SUBTRAHEND, coefficient
/// -1) is NOT self-accumulating — its reachable space is finite, BFS terminates
/// quickly, and the certificate would only perturb stats (e.g. states_found)
/// without benefit. Requiring `v` on the GROWING side of the operator
/// (`v + e`, `e + v`, `v - e`) keeps the trigger to genuine divergence
/// (`x' = x + 1`, `a' = a + 1`, `c' = c - 1`) and leaves finite specs on the
/// unchanged BFS path with identical stats.
///
/// Soundness is unaffected either way — the certificate proves Safe only via a
/// complete proof regardless of what triggers it; this gate only decides WHEN to
/// attempt it.
pub(crate) fn next_has_accumulating_arith(next_expr: &Expr) -> bool {
    has_self_accumulating_assignment(next_expr)
}

/// Recursively search for a primed assignment whose RHS grows the SAME variable:
/// `v' = v + e`, `v' = e + v`, or `v' = v - e`.
fn has_self_accumulating_assignment(expr: &Expr) -> bool {
    match expr {
        Expr::Eq(lhs, rhs) => {
            if let Some(name) = primed_var_name(&lhs.node) {
                return rhs_grows_var(&rhs.node, &name);
            }
            false
        }
        _ => {
            let mut found = false;
            let mut child = |c: &Spanned<Expr>| {
                if has_self_accumulating_assignment(&c.node) {
                    found = true;
                }
            };
            walk_immediate_children(expr, &mut child);
            found
        }
    }
}

/// Returns `true` iff `rhs` is `v + e`, `e + v`, or `v - e` (so iterating the
/// assignment `v' = rhs` moves `v` monotonically and unboundedly), where `v`
/// matches `var`. `e - v` (v as subtrahend) is NOT accumulating.
fn rhs_grows_var(rhs: &Expr, var: &str) -> bool {
    fn is_var(e: &Expr, var: &str) -> bool {
        matches!(e, Expr::Ident(n, _) | Expr::StateVar(n, ..) if n == var)
    }
    match rhs {
        Expr::Add(a, b) => is_var(&a.node, var) || is_var(&b.node, var),
        // `v - e`: v is the minuend (decreases unboundedly). `e - v` is bounded
        // oscillation territory and is intentionally excluded.
        Expr::Sub(a, _) => is_var(&a.node, var),
        _ => false,
    }
}

/// Run BMC-based symbolic bug finding on a TLA+ spec.
pub fn check_bmc(
    module: &Module,
    config: &Config,
    ctx: &EvalCtx,
    bmc_config: BmcConfig,
) -> Result<BmcResult, BmcError> {
    check_bmc_with_evidence(module, config, ctx, bmc_config).map(BmcRunResult::into_result)
}

/// Run BMC and return typed AY decision/profile boundary evidence.
pub fn check_bmc_with_evidence(
    module: &Module,
    config: &Config,
    ctx: &EvalCtx,
    bmc_config: BmcConfig,
) -> Result<BmcRunResult, BmcError> {
    check_bmc_with_portfolio_and_evidence(module, config, ctx, bmc_config, None)
}

/// Run BMC with portfolio verdict for early termination.
///
/// When `portfolio_verdict` is `Some`, each depth iteration checks whether
/// another lane has already resolved before starting the solver call.
/// If resolved, returns `BmcResult::Unknown` immediately.
///
/// Part of #3717.
pub fn check_bmc_with_portfolio(
    module: &Module,
    config: &Config,
    ctx: &EvalCtx,
    bmc_config: BmcConfig,
    portfolio_verdict: Option<Arc<SharedVerdict>>,
) -> Result<BmcResult, BmcError> {
    check_bmc_with_portfolio_and_evidence(module, config, ctx, bmc_config, portfolio_verdict)
        .map(BmcRunResult::into_result)
}

/// Run BMC with portfolio verdict and preserve typed AY decision/profile evidence.
pub fn check_bmc_with_portfolio_and_evidence(
    module: &Module,
    config: &Config,
    ctx: &EvalCtx,
    bmc_config: BmcConfig,
    portfolio_verdict: Option<Arc<SharedVerdict>>,
) -> Result<BmcRunResult, BmcError> {
    let symbolic_ctx =
        ay_shared::symbolic_ctx_with_config(ctx, config).map_err(BmcError::MissingSpec)?;
    let vars = ay_shared::collect_state_vars(module, &symbolic_ctx);
    if vars.is_empty() {
        return Err(BmcError::MissingSpec(
            "No state variables declared".to_string(),
        ));
    }

    if config.invariants.is_empty() {
        return Err(BmcError::NoInvariants);
    }

    let resolved =
        ay_shared::resolve_init_next(config, &symbolic_ctx).map_err(BmcError::MissingSpec)?;

    let init_expr = ay_shared::get_operator_body(&symbolic_ctx, &resolved.init)
        .map_err(BmcError::MissingSpec)?;
    let next_expr = ay_shared::get_operator_body(&symbolic_ctx, &resolved.next)
        .map_err(BmcError::MissingSpec)?;
    let safety_expr = ay_shared::build_safety_conjunction(&symbolic_ctx, &config.invariants)
        .map_err(|e| BmcError::TranslationError(e))?;

    let init_expanded = expand_operators_for_chc(&symbolic_ctx, &init_expr, false);
    let next_expanded = expand_operators_for_chc(&symbolic_ctx, &next_expr, true);
    let safety_expanded = expand_operators_for_chc(&symbolic_ctx, &safety_expr, false);

    let var_sorts =
        ay_shared::infer_var_sorts(&vars, &init_expanded, &config.invariants, &symbolic_ctx);

    if bmc_config.debug {
        eprintln!(
            "[ay-bmc] checking {} vars up to depth {}{}",
            var_sorts.len(),
            bmc_config.max_depth,
            if bmc_config.incremental {
                " (incremental)"
            } else {
                ""
            }
        );
    }

    if bmc_config.incremental {
        check_bmc_incremental(
            &var_sorts,
            &init_expanded,
            &next_expanded,
            &safety_expanded,
            &bmc_config,
            portfolio_verdict.as_deref(),
        )
    } else {
        check_bmc_per_depth(
            &var_sorts,
            &init_expanded,
            &next_expanded,
            &safety_expanded,
            &bmc_config,
            portfolio_verdict.as_deref(),
        )
    }
}

/// Per-depth BMC: creates a fresh solver for each depth bound.
///
/// This is the original approach. Each depth discards the solver and starts fresh,
/// losing any learned clauses from previous depths.
fn check_bmc_per_depth(
    var_sorts: &[(String, tla_ay::TlaSort)],
    init_expanded: &tla_core::Spanned<tla_core::ast::Expr>,
    next_expanded: &tla_core::Spanned<tla_core::ast::Expr>,
    safety_expanded: &tla_core::Spanned<tla_core::ast::Expr>,
    bmc_config: &BmcConfig,
    portfolio_verdict: Option<&SharedVerdict>,
) -> Result<BmcRunResult, BmcError> {
    let mut last_solver_profile = None;

    // Derive a SOUND inductive interval bound ONCE before deepening. Asserting
    // a proven-inductive B at every step is equivalence-preserving (B is implied
    // by Init/Next) but hands LIA a propagatable interval, taming the ~2^k
    // ITE-selector blowup on UNSAT-safe specs. `None` if no bound is proven
    // inductive (then behavior is unchanged). See `derive_inductive_bound`.
    let inductive_bound = derive_inductive_bound_best_effort(
        var_sorts,
        init_expanded,
        next_expanded,
        safety_expanded,
        bmc_config.solve_timeout,
        bmc_config.debug,
    );

    // Shared total budget for all deadlock probing across the depth loop.
    let deadlock_deadline = Instant::now() + DEADLOCK_PROBE_TOTAL_BUDGET;

    for depth in 0..=bmc_config.max_depth {
        // Portfolio early-exit: another lane resolved (Part of #3717).
        if let Some(sv) = portfolio_verdict {
            if sv.is_resolved() {
                return Ok(BmcRunResult::new(
                    BmcResult::Unknown {
                        depth,
                        reason: String::from("portfolio verdict resolved by another lane"),
                    },
                    AYSolveDecisionProfileEvidence::missing("TLA"),
                ));
            }
        }

        if bmc_config.debug {
            eprintln!("[ay-bmc] depth {}", depth);
        }

        let mut translator = make_translator(var_sorts, depth)?;
        translator.set_timeout(bmc_config.solve_timeout);

        for (name, sort) in var_sorts {
            translator.declare_var(name, sort.clone())?;
        }

        let init_term = translator.translate_init(init_expanded)?;
        translator.assert(init_term);

        for step in 0..depth {
            let next_term = translator.translate_next(next_expanded, step)?;
            translator.assert(next_term);
        }

        // Assert the proven-inductive bound at EVERY step present in the query
        // (0..=depth). Equivalence-preserving by the inductiveness gate.
        if let Some(bound) = &inductive_bound {
            for step in 0..=depth {
                let bound_term = translator.translate_safety_at_step(bound, step)?;
                translator.assert(bound_term);
            }
        }

        let not_safety = translator.translate_not_safety_all_steps(safety_expanded, depth)?;
        translator.assert(not_safety);

        let (sat_result, summary) = translator.try_check_sat_with_decision_profile_summary()?;
        let solver_profile = AYSolveDecisionProfileEvidence::from_summary("TLA", Some(&summary));

        match sat_result {
            SolveResult::Sat => {
                if !solver_profile.accepts_model_for_tla_boundary() {
                    return Ok(BmcRunResult::new(
                        BmcResult::Unknown {
                            depth,
                            reason: format!(
                                "AY SAT result rejected by consumer boundary at depth {}",
                                depth
                            ),
                        },
                        solver_profile,
                    ));
                }
                let model = translator.try_get_model()?;
                let trace = truncate_trace_to_depth(translator.extract_trace(&model), depth);
                return Ok(BmcRunResult::new(
                    BmcResult::Violation { depth, trace },
                    solver_profile,
                ));
            }
            SolveResult::Unsat(_) => {
                // No safety violation at this depth. Before deepening, probe for
                // a REACHABLE deadlock state at depth k (sound concrete-state
                // enumeration; strictly additive — see probe_deadlock_at_depth).
                // A safety Violation already returned above takes priority; this
                // only fires when safety holds so far.
                if bmc_config.check_deadlock {
                    if let Some((dl_depth, dl_trace)) = probe_deadlock_at_depth(
                        deadlock_deadline,
                        var_sorts,
                        init_expanded,
                        next_expanded,
                        inductive_bound.as_ref(),
                        depth,
                        bmc_config.solve_timeout,
                        bmc_config.debug,
                    ) {
                        return Ok(BmcRunResult::new(
                            BmcResult::Deadlock {
                                depth: dl_depth,
                                trace: dl_trace,
                            },
                            solver_profile,
                        ));
                    }
                }
                last_solver_profile = Some(solver_profile);
            }
            SolveResult::Unknown => {
                let reason = match translator.last_unknown_reason() {
                    Some(UnknownReason::Timeout) => {
                        format!("solver timed out at depth {}", depth)
                    }
                    Some(other) => {
                        format!("solver returned unknown at depth {}: {:?}", depth, other)
                    }
                    None => format!("solver returned unknown at depth {}", depth),
                };
                return Ok(BmcRunResult::new(
                    BmcResult::Unknown { depth, reason },
                    solver_profile,
                ));
            }
            _ => {
                return Ok(BmcRunResult::new(
                    BmcResult::Unknown {
                        depth,
                        reason: format!("solver returned unexpected result at depth {}", depth),
                    },
                    solver_profile,
                ));
            }
        }
    }

    Ok(BmcRunResult::new(
        BmcResult::BoundReached {
            max_depth: bmc_config.max_depth,
        },
        last_solver_profile.unwrap_or_else(|| AYSolveDecisionProfileEvidence::missing("TLA")),
    ))
}

/// Incremental BMC: keeps one solver instance across all depths. Part of #3724.
///
/// Uses `push_scope`/`pop_scope` to retract per-depth safety negation queries
/// while retaining Init + accumulated Next transition assertions. This allows
/// the solver to carry forward learned clauses across depth iterations.
///
/// Formula at depth k:
/// ```text
/// [persistent] Init(s0)
/// [persistent] Next(s0,s1), Next(s1,s2), ..., Next(sk-1,sk)
/// [scoped]     ¬Safety(s0) ∨ ... ∨ ¬Safety(sk)
/// ```
fn check_bmc_incremental(
    var_sorts: &[(String, tla_ay::TlaSort)],
    init_expanded: &tla_core::Spanned<tla_core::ast::Expr>,
    next_expanded: &tla_core::Spanned<tla_core::ast::Expr>,
    safety_expanded: &tla_core::Spanned<tla_core::ast::Expr>,
    bmc_config: &BmcConfig,
    portfolio_verdict: Option<&SharedVerdict>,
) -> Result<BmcRunResult, BmcError> {
    // Create a single translator for the maximum depth. All k+1 step variables
    // are declared up front so transitions can be added incrementally.
    let mut translator = make_translator(var_sorts, bmc_config.max_depth)?;
    translator.set_timeout(bmc_config.solve_timeout);

    for (name, sort) in var_sorts {
        translator.declare_var(name, sort.clone())?;
    }

    // Derive a SOUND inductive interval bound ONCE (see `derive_inductive_bound`).
    // A proven-inductive B is implied by Init/Next, so conjoining it as a
    // PERSISTENT assertion at every step is equivalence-preserving while giving
    // LIA an interval to propagate. `None` -> behavior unchanged.
    let inductive_bound = derive_inductive_bound_best_effort(
        var_sorts,
        init_expanded,
        next_expanded,
        safety_expanded,
        bmc_config.solve_timeout,
        bmc_config.debug,
    );

    // Assert Init(s0) once — this persists across all depths.
    let init_term = translator.translate_init(init_expanded)?;
    translator.assert(init_term);

    // Assert the bound at step 0 persistently (alongside Init). Subsequent
    // steps' bounds are asserted as each Next transition is added below, so B
    // holds at every step present in the query.
    if let Some(bound) = &inductive_bound {
        let bound_term = translator.translate_safety_at_step(bound, 0)?;
        translator.assert(bound_term);
    }
    let mut last_solver_profile = None;

    // Shared total budget for all deadlock probing across the depth loop.
    let deadlock_deadline = Instant::now() + DEADLOCK_PROBE_TOTAL_BUDGET;

    for depth in 0..=bmc_config.max_depth {
        // Portfolio early-exit: another lane resolved (Part of #3717).
        if let Some(sv) = portfolio_verdict {
            if sv.is_resolved() {
                return Ok(BmcRunResult::new(
                    BmcResult::Unknown {
                        depth,
                        reason: String::from("portfolio verdict resolved by another lane"),
                    },
                    AYSolveDecisionProfileEvidence::missing("TLA"),
                ));
            }
        }

        if bmc_config.debug {
            eprintln!("[ay-bmc-incr] depth {}", depth);
        }

        // Add the transition for the new step (persistent, not retracted by pop).
        // At depth 0 there is no transition to add.
        if depth > 0 {
            let next_term = translator.translate_next(next_expanded, depth - 1)?;
            translator.assert(next_term);

            // Persistently assert the proven-inductive bound at the new step
            // `depth` (step 0 was asserted after Init). Equivalence-preserving.
            if let Some(bound) = &inductive_bound {
                let bound_term = translator.translate_safety_at_step(bound, depth)?;
                translator.assert(bound_term);
            }
        }

        // Push a scope for the per-depth safety negation query.
        // Use a scoped-result pattern: run the query inside a closure, then
        // ALWAYS pop the scope before propagating errors. Fixes push/pop
        // scope leak on `?` error propagation (same class of bug as #4000).
        translator.push_scope()?;
        let scoped_result: Result<
            (Option<BmcRunResult>, Option<AYSolveDecisionProfileEvidence>),
            BmcError,
        > = (|| {
            let not_safety = translator.translate_not_safety_all_steps(safety_expanded, depth)?;
            translator.assert(not_safety);

            let (sat_result, summary) = translator.try_check_sat_with_decision_profile_summary()?;
            let solver_profile =
                AYSolveDecisionProfileEvidence::from_summary("TLA", Some(&summary));

            match sat_result {
                SolveResult::Sat => {
                    if !solver_profile.accepts_model_for_tla_boundary() {
                        return Ok((
                            Some(BmcRunResult::new(
                                BmcResult::Unknown {
                                    depth,
                                    reason: format!(
                                        "AY SAT result rejected by consumer boundary at depth {} (incremental)",
                                        depth
                                    ),
                                },
                                solver_profile.clone(),
                            )),
                            Some(solver_profile),
                        ));
                    }
                    let model = translator.try_get_model()?;
                    let trace = truncate_trace_to_depth(translator.extract_trace(&model), depth);
                    Ok((
                        Some(BmcRunResult::new(
                            BmcResult::Violation { depth, trace },
                            solver_profile.clone(),
                        )),
                        Some(solver_profile),
                    ))
                }
                SolveResult::Unsat(_) => {
                    // No safety violation at this depth. Probe for a REACHABLE
                    // deadlock state at depth k (sound concrete-state enumeration
                    // in fresh scratch translators; never touches this persistent
                    // solver). Strictly additive — see probe_deadlock_at_depth.
                    if bmc_config.check_deadlock {
                        if let Some((dl_depth, dl_trace)) = probe_deadlock_at_depth(
                            deadlock_deadline,
                            var_sorts,
                            init_expanded,
                            next_expanded,
                            inductive_bound.as_ref(),
                            depth,
                            bmc_config.solve_timeout,
                            bmc_config.debug,
                        ) {
                            return Ok((
                                Some(BmcRunResult::new(
                                    BmcResult::Deadlock {
                                        depth: dl_depth,
                                        trace: dl_trace,
                                    },
                                    solver_profile.clone(),
                                )),
                                Some(solver_profile),
                            ));
                        }
                    }
                    Ok((None, Some(solver_profile)))
                }
                SolveResult::Unknown => {
                    let reason = match translator.last_unknown_reason() {
                        Some(UnknownReason::Timeout) => {
                            format!("solver timed out at depth {} (incremental)", depth)
                        }
                        Some(other) => {
                            format!(
                                "solver returned unknown at depth {} (incremental): {:?}",
                                depth, other
                            )
                        }
                        None => {
                            format!("solver returned unknown at depth {} (incremental)", depth)
                        }
                    };
                    Ok((
                        Some(BmcRunResult::new(
                            BmcResult::Unknown { depth, reason },
                            solver_profile.clone(),
                        )),
                        Some(solver_profile),
                    ))
                }
                _ => Ok((
                    Some(BmcRunResult::new(
                        BmcResult::Unknown {
                            depth,
                            reason: format!(
                                "solver returned unexpected result at depth {} (incremental)",
                                depth
                            ),
                        },
                        solver_profile.clone(),
                    )),
                    Some(solver_profile),
                )),
            }
        })();

        // ALWAYS pop the scope, regardless of whether the inner block
        // succeeded or failed. Prevents scope leak on error propagation.
        translator.pop_scope()?;

        match scoped_result {
            Ok((Some(result), _)) => return Ok(result),
            Ok((None, Some(profile))) => {
                last_solver_profile = Some(profile);
            }
            Ok((None, None)) => { /* No solve ran — continue to next depth */ }
            Err(e) => return Err(e),
        }
    }

    Ok(BmcRunResult::new(
        BmcResult::BoundReached {
            max_depth: bmc_config.max_depth,
        },
        last_solver_profile.unwrap_or_else(|| AYSolveDecisionProfileEvidence::missing("TLA")),
    ))
}

/// Run BMC seeded from BFS frontier states instead of Init.
///
/// Polls `cooperative.frontier_rx` for concrete states and compressed wavefront
/// formulas. Uses a **persistent translator** across all seeds — variable
/// declarations and solver configuration are done once, and each seed
/// (frontier sample or wavefront formula) is wrapped in a `push_scope`/`pop_scope`
/// pair so seed-specific assertions are retracted without rebuilding the solver.
/// This preserves learned clauses across iterations, enabling incremental solving.
///
/// Part of #3766, Epic #3762 (CDEMC).
/// Part of #3834: incremental wavefront BMC with translator reuse.
pub(crate) fn check_bmc_cooperative(
    module: &Module,
    config: &Config,
    ctx: &EvalCtx,
    bmc_config: BmcConfig,
    cooperative: Arc<crate::cooperative_state::SharedCooperativeState>,
) -> Result<BmcResult, BmcError> {
    use std::time::Duration;
    use tla_ay::BmcValue;

    // Same setup as check_bmc_with_portfolio.
    let symbolic_ctx =
        ay_shared::symbolic_ctx_with_config(ctx, config).map_err(BmcError::MissingSpec)?;
    let vars = ay_shared::collect_state_vars(module, &symbolic_ctx);
    if vars.is_empty() {
        return Err(BmcError::MissingSpec(
            "No state variables declared".to_string(),
        ));
    }
    if config.invariants.is_empty() {
        return Err(BmcError::NoInvariants);
    }

    let resolved =
        ay_shared::resolve_init_next(config, &symbolic_ctx).map_err(BmcError::MissingSpec)?;
    let next_expr = ay_shared::get_operator_body(&symbolic_ctx, &resolved.next)
        .map_err(BmcError::MissingSpec)?;
    let safety_expr = ay_shared::build_safety_conjunction(&symbolic_ctx, &config.invariants)
        .map_err(|e| BmcError::TranslationError(e))?;

    let next_expanded = expand_operators_for_chc(&symbolic_ctx, &next_expr, true);
    let safety_expanded = expand_operators_for_chc(&symbolic_ctx, &safety_expr, false);

    // Infer sorts for Init expression (needed for var_sorts even though we don't translate Init).
    let init_expr = ay_shared::get_operator_body(&symbolic_ctx, &resolved.init)
        .map_err(BmcError::MissingSpec)?;
    let init_expanded = expand_operators_for_chc(&symbolic_ctx, &init_expr, false);
    let var_sorts =
        ay_shared::infer_var_sorts(&vars, &init_expanded, &config.invariants, &symbolic_ctx);

    if bmc_config.debug {
        eprintln!(
            "[ay-bmc-coop] waiting for frontier states, {} vars, max depth {}",
            var_sorts.len(),
            bmc_config.max_depth,
        );
    }

    // Lemma cursor: tracks how many learned lemmas BMC has already consumed
    // from the cooperative state. Part of #3835.
    let mut lemma_cursor: usize = 0;
    // Cached expanded lemma expressions, ready for BMC translation.
    let mut expanded_lemmas: Vec<tla_core::Spanned<tla_core::ast::Expr>> = Vec::new();

    // Helper closure: deepen from a seeded translator, checking safety at each
    // depth. Returns Ok(Some(result)) on violation or cooperative resolution,
    // Ok(None) if max_depth exhausted without finding a violation.
    //
    // Part of #3823: extracted so both frontier-sample and wavefront-formula
    // code paths share the same deepening logic.
    //
    // Enhanced: reports depth progress to SharedCooperativeState at each step
    // so that other lanes (BFS, PDR) can observe live BMC deepening activity.
    // Tracks per-seed completion metrics for average depth analysis.
    //
    // NOTE: Lemmas are NOT asserted here. PDR-learned lemmas are universal
    // invariants that hold at every reachable state. They are asserted
    // persistently at the base translator level (outside seed push/pop
    // scopes) so they are shared across all seeds without redundant
    // re-assertion. See `assert_lemmas_persistent` below. Fixes #4003.
    let deepen_from_seed =
        |translator: &mut tla_ay::BmcTranslator,
         cooperative: &crate::cooperative_state::SharedCooperativeState|
         -> Result<Option<BmcResult>, BmcError> {
            // Signal that a new seed is being processed.
            cooperative.bmc_start_seed();

            // Inner helper that does the actual deepening work. Returns
            // `(max_unsat_depth, result)` so the outer closure can pass the
            // accurate depth to `bmc_complete_seed` on ALL exit paths (success
            // or error), fixing #4005. Scope-level cleanup (pop after push) is
            // handled within this helper using a scoped-result pattern, fixing
            // #4000.
            let inner = |translator: &mut tla_ay::BmcTranslator|
         -> (u64, Result<Option<BmcResult>, BmcError>) {
            let mut max_unsat_depth: u64 = 0;

            for depth in 0..=bmc_config.max_depth {
                // Report live depth progress to the cooperative state.
                cooperative.bmc_report_depth_progress(depth as u64);

                if cooperative.is_resolved() {
                    return (max_unsat_depth, Ok(Some(BmcResult::Unknown {
                        depth,
                        reason: String::from("cooperative verdict resolved during BMC deepening"),
                    })));
                }

                if depth > 0 {
                    let next_term = match translator.translate_next(&next_expanded, depth - 1) {
                        Ok(t) => t,
                        Err(e) => return (max_unsat_depth, Err(e.into())),
                    };
                    translator.assert(next_term);
                }

                // Push a scope for the per-depth safety negation query.
                // Use a scoped-result pattern: capture the result of the
                // push/query/check block, then ALWAYS pop before propagating
                // any error. This prevents scope leaks on `?` errors (#4000).
                match translator.push_scope() {
                    Ok(()) => {}
                    Err(e) => return (max_unsat_depth, Err(e.into())),
                }
                let scoped_result: Result<Option<BmcResult>, BmcError> = (|| {
                    let not_safety =
                        translator.translate_not_safety_all_steps(&safety_expanded, depth)?;
                    translator.assert(not_safety);

                    match translator.try_check_sat()? {
                        tla_ay::SolveResult::Sat => {
                            let model = translator.try_get_model()?;
                            let trace = truncate_trace_to_depth(translator.extract_trace(&model), depth);
                            // SOUNDNESS (fail closed): a SAT model comes from the
                            // SMT TRANSLATION and can be spurious. Publishing
                            // `Violated` truncates the racing BFS lane into a
                            // result indistinguishable from a clean Success, so
                            // it may happen ONLY after the explicit-state
                            // evaluator has re-confirmed the counterexample. An
                            // unconfirmed model yields Unknown (no publish) —
                            // BFS keeps running, unharmed.
                            let cv = crate::check::cross_validation::confirm_symbolic_cex_fail_closed(
                                module,
                                config,
                                &trace,
                                crate::check::cross_validation::CrossValidationSource::Bmc,
                            );
                            if cv.engine_agrees {
                                cooperative
                                    .verdict
                                    .publish(crate::shared_verdict::Verdict::Violated);
                                Ok(Some(BmcResult::Violation { depth, trace }))
                            } else {
                                telemetry_eprintln!(
                                    "[ay-bmc-coop] SAT at depth {depth} but the explicit-state \
                                     evaluator did NOT confirm the counterexample ({}) — \
                                     failing closed to Unknown (no verdict published)",
                                    cv.detail
                                );
                                Ok(Some(BmcResult::Unknown {
                                    depth,
                                    reason: format!(
                                        "SAT at depth {depth} but the explicit-state evaluator \
                                         did not confirm the counterexample ({}) — failing closed",
                                        cv.detail
                                    ),
                                }))
                            }
                        }
                        tla_ay::SolveResult::Unsat(_) => Ok(None),
                        _ => Ok(Some(BmcResult::Unknown {
                            depth,
                            reason: format!(
                                "solver returned unexpected result at depth {} (cooperative)",
                                depth
                            ),
                        })),
                    }
                })();

                // ALWAYS pop the scope, regardless of whether the inner block
                // succeeded or failed. This is the key fix for #4000.
                match translator.pop_scope() {
                    Ok(()) => {}
                    Err(e) => return (max_unsat_depth, Err(e.into())),
                }

                match scoped_result {
                    Ok(Some(result)) => return (max_unsat_depth, Ok(Some(result))),
                    Ok(None) => {
                        // Unsat at this depth — record and continue deepening.
                        max_unsat_depth = depth as u64;
                    }
                    Err(e) => return (max_unsat_depth, Err(e)),
                }
            }
            (max_unsat_depth, Ok(None))
        };

            // Run the inner helper and ensure bmc_complete_seed is called on ALL
            // exit paths — both success and error. This fixes #4005.
            let (max_unsat_depth, result) = inner(translator);
            cooperative.bmc_complete_seed(max_unsat_depth);
            result
        };

    // Assert PDR-learned lemmas persistently at the base translator level.
    //
    // PDR lemmas are universal invariants — they hold at every reachable state,
    // not just at specific seeds. Asserting them outside seed push/pop scopes
    // means they persist across all seeds, avoiding:
    // (a) Redundant re-assertion of the same lemmas for every seed
    // (b) The misleading "persistent constraints" comment from the old code
    //     where lemmas were actually scoped to seeds (inside push/pop)
    //
    // Lemmas must be asserted at every BMC step (0..=max_depth) because the
    // invariant holds at each reachable state along the trace.
    //
    // Returns the number of successfully asserted lemma-step pairs.
    //
    // Fixes #4003: lemmas are now truly persistent (base level), not falsely
    // labeled "persistent" while being scoped inside seed push/pop brackets.
    // Part of #3835: PDR lemma sharing to BMC.
    // Part of #4001: log translation failures instead of silently swallowing.
    let assert_lemmas_persistent = |translator: &mut tla_ay::BmcTranslator,
                                    lemmas: &[tla_core::Spanned<tla_core::ast::Expr>],
                                    debug: bool|
     -> usize {
        let mut asserted: usize = 0;
        let mut failures: usize = 0;
        for lemma in lemmas {
            for step in 0..=bmc_config.max_depth {
                match translator.translate_safety_at_step(lemma, step) {
                    Ok(lemma_term) => {
                        translator.assert(lemma_term);
                        asserted += 1;
                    }
                    Err(e) => {
                        failures += 1;
                        if debug {
                            eprintln!(
                                "[ay-bmc-coop] lemma translation failed at step {}: {}",
                                step, e,
                            );
                        }
                    }
                }
            }
        }
        if failures > 0 && debug {
            eprintln!(
                "[ay-bmc-coop] {}/{} lemma-step assertions failed (persistent)",
                failures,
                lemmas.len() * (bmc_config.max_depth + 1),
            );
        }
        asserted
    };

    // Helper: create a fresh persistent translator with variable declarations.
    // Extracted so we can recreate it periodically to clear learned clause
    // accumulation (Part of #4006).
    let create_translator = |var_sorts: &[(String, tla_ay::TlaSort)],
                             bmc_config: &BmcConfig|
     -> Result<tla_ay::BmcTranslator, BmcError> {
        let mut t = make_translator(var_sorts, bmc_config.max_depth)?;
        t.set_timeout(bmc_config.solve_timeout);
        for (name, sort) in var_sorts {
            t.declare_var(name, sort.clone())?;
        }
        Ok(t)
    };

    // Create ONE persistent translator for the cooperative poll loop.
    // Variable declarations and solver configuration are done once. Each seed
    // (frontier sample or wavefront formula) uses push_scope/pop_scope to add
    // sample-specific assertions without rebuilding the translator, preserving
    // any learned clauses across iterations.
    //
    // Part of #3834: incremental wavefront BMC with translator reuse.
    // Part of #4006: periodically recreated to clear accumulated learned clauses.
    let mut translator = create_translator(&var_sorts, &bmc_config)?;
    // Cooperative teardown: let the fused orchestrator abort this solver
    // mid-`check_sat` once another lane resolves the verdict (the per-depth
    // is_resolved polls cannot fire while the thread is inside a solve).
    cooperative.register_solver_interrupt(translator.interrupt_handle());

    // Track seeds processed since last translator refresh for clause eviction.
    // Part of #4006.
    let mut seeds_since_refresh: u64 = 0;

    if bmc_config.debug {
        eprintln!(
            "[ay-bmc-coop] persistent translator created, {} vars declared",
            var_sorts.len(),
        );
    }

    // Poll cooperative channels for concrete frontier states and compressed
    // wavefront formulas. Individual samples are seeded via assert_concrete_state;
    // wavefront formulas are seeded via assert_wavefront_formula, encoding the
    // entire compressed frontier as a single disjunctive constraint.
    //
    // Each seed is wrapped in a push/pop scope so the base translator state
    // (variable declarations, solver config) persists across seeds.
    //
    // Part of #3823: close the feedback loop so wavefront formulas are consumed.
    // Part of #3834: persistent translator with incremental solving.
    // Part of #4004: starvation prevention via backpressure and wavefront skipping.
    let poll_timeout = Duration::from_millis(500);
    let wavefront_timeout = Duration::from_millis(50);
    loop {
        // Early exit: another lane resolved.
        if cooperative.is_resolved() {
            return Ok(BmcResult::Unknown {
                depth: 0,
                reason: String::from("cooperative verdict resolved by another lane"),
            });
        }

        // Early exit: BFS completed without publishing a resolved verdict.
        // Without this check, the cooperative loop spins forever waiting for
        // wavefronts that will never arrive. Part of #4002.
        if cooperative.is_bfs_complete() && !cooperative.is_resolved() {
            return Ok(BmcResult::Unknown {
                depth: 0,
                reason: String::from("BFS completed without resolved verdict, BMC exiting"),
            });
        }

        // Periodic translator refresh: recreate the solver to clear accumulated
        // learned clauses and internal state. Without this, the persistent
        // translator's memory grows unboundedly over long runs.
        // Part of #4006.
        if seeds_since_refresh >= TRANSLATOR_REFRESH_INTERVAL {
            if bmc_config.debug {
                eprintln!(
                    "[ay-bmc-coop] refreshing translator after {} seeds to clear learned clauses",
                    seeds_since_refresh,
                );
            }
            translator = create_translator(&var_sorts, &bmc_config)?;
            // Re-register the fresh solver's interrupt handle (per-instance).
            cooperative.register_solver_interrupt(translator.interrupt_handle());
            seeds_since_refresh = 0;
            cooperative.record_translator_refresh();

            // Re-assert all accumulated lemmas on the fresh translator.
            // Lemmas are persistent base-level constraints that must survive
            // translator refresh. Fixes #4003.
            if !expanded_lemmas.is_empty() {
                let n =
                    assert_lemmas_persistent(&mut translator, &expanded_lemmas, bmc_config.debug);
                if bmc_config.debug {
                    eprintln!(
                        "[ay-bmc-coop] re-asserted {} lemma-step pairs after translator refresh",
                        n,
                    );
                }
            }
        }

        // Poll for new PDR lemmas before processing the next seed.
        // New lemmas are asserted persistently at the base translator level
        // (outside any push/pop scope) so they apply to all future seeds.
        // Fixes #4003: lemmas are no longer re-asserted redundantly inside
        // each seed's push/pop scope.
        let (new_lemmas, new_cursor) = cooperative.poll_learned_lemmas(lemma_cursor);
        if !new_lemmas.is_empty() {
            let mut new_expanded = Vec::with_capacity(new_lemmas.len());
            for lemma in new_lemmas {
                new_expanded.push(expand_operators_for_chc(&symbolic_ctx, &lemma, false));
            }

            // Assert the new lemmas persistently at base solver level.
            let n = assert_lemmas_persistent(&mut translator, &new_expanded, bmc_config.debug);

            if bmc_config.debug {
                eprintln!(
                    "[ay-bmc-coop] consumed {} new PDR lemmas ({} step-assertions, total: {})",
                    new_expanded.len(),
                    n,
                    new_cursor,
                );
            }

            expanded_lemmas.extend(new_expanded);
            lemma_cursor = new_cursor;
        }

        // Starvation detection: if BFS is producing wavefronts much faster
        // than BMC can consume them, drain intermediate wavefronts and skip
        // to the most recent one. This bounds memory growth and keeps BMC
        // working on the current frontier rather than falling permanently behind.
        // Part of #4004.
        let starvation_gap = cooperative.wavefront_starvation_gap();
        if starvation_gap > STARVATION_THRESHOLD {
            let (drained_wf, latest_wf) = cooperative.drain_stale_wavefronts();
            let (drained_fs, _latest_fs) = cooperative.drain_stale_frontier_samples();

            if drained_wf > 0 {
                for _ in 0..drained_wf {
                    cooperative.record_wavefront_dropped_backpressure();
                    // Count drained wavefronts as consumed to keep gap accurate.
                    cooperative.record_wavefront_consumed();
                }
            }
            if drained_fs > 0 {
                for _ in 0..drained_fs {
                    cooperative.record_frontier_sample_dropped_backpressure();
                }
            }

            if bmc_config.debug && (drained_wf > 0 || drained_fs > 0) {
                eprintln!(
                    "[ay-bmc-coop] backpressure: starvation gap {}, drained {} wavefronts + {} samples",
                    starvation_gap, drained_wf, drained_fs,
                );
            }

            // Process the latest wavefront if we got one from the drain.
            if let Some(wavefront) = latest_wf {
                cooperative.record_wavefront_consumed();
                seeds_since_refresh += 1;

                translator.push_scope()?;

                let shared: Vec<(String, BmcValue)> = wavefront
                    .shared
                    .iter()
                    .map(|s| (s.name.clone(), s.value.clone()))
                    .collect();
                let disjuncts: Vec<Vec<(String, BmcValue)>> = wavefront
                    .disjuncts
                    .iter()
                    .map(|d| d.assignments.clone())
                    .collect();
                translator.assert_wavefront_formula(&shared, &disjuncts, 0)?;

                let result = deepen_from_seed(&mut translator, &cooperative)?;

                translator.pop_scope()?;
                // Evict Skolem constants and other temporary variables
                // accumulated during seed translation. Part of #4006.
                translator.clear_temporary_vars();

                if let Some(result) = result {
                    if matches!(result, BmcResult::Violation { .. }) {
                        cooperative.record_wavefront_induced_violation();
                    }
                    return Ok(result);
                }

                continue;
            }
        }

        // Prefer wavefront formulas over individual samples: wavefronts
        // encode many states at once (higher coverage per solver call).
        // Fall back to individual frontier samples when no wavefront is ready.
        //
        // BFS frontier hint prioritization: when the BFS lane has explored
        // several levels, seeds from shallow (already-checked) BFS depths are
        // less likely to find new violations. We still process them but log
        // when a seed is deprioritized for diagnostics.

        // Try compressed wavefront formula from the compressor thread first.
        // Wavefronts encode many frontier states in a single disjunctive formula,
        // giving higher coverage per solver invocation than individual samples.
        // Part of #3823: close the feedback loop so wavefront formulas are consumed.
        if let Some(wavefront) = cooperative.recv_wavefront(wavefront_timeout) {
            let wavefront_depth = wavefront.depth;
            let prioritized = cooperative.should_prioritize_seed(wavefront_depth);

            if bmc_config.debug {
                eprintln!(
                    "[ay-bmc-coop] got wavefront at depth {}, {} shared, {} disjuncts{}",
                    wavefront_depth,
                    wavefront.shared.len(),
                    wavefront.disjuncts.len(),
                    if prioritized { "" } else { " (deprioritized)" },
                );
            }

            if !prioritized {
                cooperative.record_bmc_seed_deprioritized();
            }

            cooperative.record_wavefront_consumed();
            seeds_since_refresh += 1;

            // Push an outer scope for this wavefront seed. Use a scoped-result
            // pattern to ensure pop_scope is always called, even if seed
            // assertion or deepening fails via `?`. Fixes #4000.
            translator.push_scope()?;
            let seed_result: Result<Option<BmcResult>, BmcError> = (|| {
                // Convert WavefrontFormula types to the (String, BmcValue) format
                // expected by assert_wavefront_formula.
                let shared: Vec<(String, BmcValue)> = wavefront
                    .shared
                    .iter()
                    .map(|s| (s.name.clone(), s.value.clone()))
                    .collect();
                let disjuncts: Vec<Vec<(String, BmcValue)>> = wavefront
                    .disjuncts
                    .iter()
                    .map(|d| d.assignments.clone())
                    .collect();
                translator.assert_wavefront_formula(&shared, &disjuncts, 0)?;
                deepen_from_seed(&mut translator, &cooperative)
            })();

            // ALWAYS pop the wavefront seed scope. Fixes #4000.
            translator.pop_scope()?;
            // Evict Skolem constants and other temporary variables
            // accumulated during seed translation. Part of #4006.
            translator.clear_temporary_vars();

            let seed_result = seed_result?;
            if let Some(result) = seed_result {
                // Track wavefront-induced violations for diagnostics.
                if matches!(result, BmcResult::Violation { .. }) {
                    cooperative.record_wavefront_induced_violation();
                }
                return Ok(result);
            }

            // Wavefront processed — skip individual sample polling this iteration
            // to avoid starvation of wavefront consumption.
            continue;
        }

        // Fall back to individual frontier sample when no wavefront is ready.
        if let Some(sample) = cooperative.recv_frontier_sample(poll_timeout) {
            let sample_depth = sample.depth;
            let prioritized = cooperative.should_prioritize_seed(sample_depth);

            if bmc_config.debug {
                eprintln!(
                    "[ay-bmc-coop] got frontier state at depth {}, {} vars{}",
                    sample_depth,
                    sample.assignments.len(),
                    if prioritized { "" } else { " (deprioritized)" },
                );
            }

            if !prioritized {
                cooperative.record_bmc_seed_deprioritized();
            }

            seeds_since_refresh += 1;

            // Push an outer scope for this seed — all seed assertions and
            // deepening transitions will be retracted on pop. Use a
            // scoped-result pattern to ensure pop_scope is always called,
            // even if assertion or deepening fails via `?`. Fixes #4000.
            translator.push_scope()?;
            let seed_result: Result<Option<BmcResult>, BmcError> = (|| {
                let bmc_assignments: Vec<(String, BmcValue)> = sample.assignments;
                translator.assert_concrete_state(&bmc_assignments, 0)?;
                deepen_from_seed(&mut translator, &cooperative)
            })();

            // ALWAYS pop the seed scope. Fixes #4000.
            translator.pop_scope()?;
            // Evict Skolem constants and other temporary variables
            // accumulated during seed translation. Part of #4006.
            translator.clear_temporary_vars();

            let seed_result = seed_result?;
            if let Some(result) = seed_result {
                return Ok(result);
            }
        }
    }
}

#[cfg(test)]
mod tests;
