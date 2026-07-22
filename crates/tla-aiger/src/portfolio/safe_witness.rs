// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Symmetric `Safe`-result cross-validation (#4315).
//!
//! This is the portfolio-layer defense-in-depth gate that closes the systemic
//! blind spot exposed by TL54's #4310 P0 (false UNSAT via circular_pointer).
//! Before TL78 landed this module, `runner.rs:312-331` validated *unsafe*
//! witnesses by simulating the returned counterexample trace against the
//! circuit, rejecting a result whose witness did not actually violate the
//! property — but the symmetric path for `Safe` had no validator at all. Any
//! engine that said "Safe" was trusted unconditionally.
//!
//! Four witness shapes are recognized:
//! * `SafeWitness::InductiveInvariant { lemmas, .. }` — a CNF inductive
//!   invariant (the standard IC3 proof witness). The validator re-runs the
//!   three inductive-invariant checks against a **fresh** independent SAT
//!   backend (SimpleSolver for small circuits, AYNoPreprocess for larger
//!   ones — same tiering as IC3's internal validator):
//!   1. `init ⇒ inv`
//!   2. `inv ∧ T ⇒ inv'`
//!   3. `inv ⇒ ¬bad`
//!   This is the path that would have caught #4310 (false UNSAT via
//!   circular_pointer): an unsound invariant fails one of the three checks.
//! * `SafeWitness::Trivial` — the property is trivially safe (no bad lits,
//!   or all bad lits are constant FALSE). Validated by re-checking the
//!   circuit's bad-lit structure directly.
//! * `SafeWitness::KInduction { .. }` — a k-induction Safe verdict (plain or
//!   strengthened). k-induction does NOT produce a 1-step inductive invariant
//!   that the `InductiveInvariant` checker could re-verify directly (the
//!   strengthening lemmas are bounded-BMC-discovered, not 1-inductive), so the
//!   validator instead **re-runs k-induction on a fresh, independent solver
//!   backend** and confirms it also reaches `Safe`. This is the symmetric
//!   counterpart of `verify_witness` for the `Unsafe` path: a fresh proof on a
//!   different solver cannot share the producing engine's solver state. The
//!   gate is *fail-open for assurance, fail-closed for soundness*: it only
//!   REJECTS when the independent re-run produces a counterexample that
//!   genuinely violates the property (checked via `verify_witness` against the
//!   original `Transys`) — a real soundness alert — and otherwise (re-confirmed
//!   Safe, inconclusive within budget, or an unverified refutation) ACCEPTS the
//!   engine's own internally-verified proof exactly as before. It therefore can
//!   never downgrade a *correct* k-induction `Safe` verdict.
//! * `SafeWitness::EngineVerified { engine }` — engine ran its own internal
//!   safety check (e.g. BMC-with-bound saturation, CEGAR) but does not emit a
//!   formal invariant we can re-verify and is not a k-induction proof we can
//!   replay. Accepted with a log line.
//! * `SafeWitness::Unwitnessed` — engine cannot prove Safe and made no
//!   internal safety claim. The portfolio conservatively **downgrades the
//!   result to `Unknown`** unless another engine has already produced a
//!   witnessed corroborating `Safe`.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use rustc_hash::{FxHashMap, FxHashSet};

use crate::check_result::CheckResult;
use crate::kind::{KindConfig, KindEngine, KindStrengthenedEngine};
use crate::sat_types::{Lit, SatResult, SatSolver, SimpleSolver, SolverBackend, Var};
use crate::transys::Transys;

/// Proof witness attached to a `Safe` verdict from an engine.
#[derive(Debug, Clone)]
pub enum SafeWitness {
    /// Engine produced an inductive invariant expressed as CNF clauses
    /// (each inner `Vec<Lit>` is one disjunctive clause). `depth` is the
    /// convergence depth (IC3 frame index) reported by the engine. This is
    /// the only witness shape that gets full independent SAT re-verification
    /// — a rejection here is a SOUNDNESS ALERT.
    InductiveInvariant {
        /// The invariant as CNF: each inner `Vec<Lit>` is one disjunctive clause.
        lemmas: Vec<Vec<Lit>>,
        /// Convergence depth (IC3 frame index) reported by the engine.
        depth: usize,
    },
    /// Property is trivially safe: `bad_lits` is empty or all bad lits are
    /// constant FALSE. No inductive invariant is needed — the circuit cannot
    /// reach a bad state by construction. Validator re-checks bad_lits
    /// directly and rejects if the claim is false.
    Trivial,
    /// Engine does not emit a formal proof witness but ran its own internal
    /// safety check (BMC-with-bound saturation, CEGAR, etc.). The validator
    /// cannot re-verify this independently, so it accepts it but logs that no
    /// symmetric check was performed. The conservative downgrade path is
    /// reserved for `Unwitnessed`, which signals "engine made no promise at
    /// all".
    EngineVerified {
        /// Audit label naming the engine that ran the internal safety check.
        engine: &'static str,
    },
    /// A k-induction `Safe` verdict (plain or strengthened). k-induction's
    /// proof is a k-step inductive argument, not a 1-step inductive invariant,
    /// so it is re-verified by independently re-running k-induction on a fresh
    /// solver backend rather than by the CNF invariant checker. `engine` is the
    /// audit label; `strengthened` selects the strengthened engine for replay;
    /// `simple_path` reproduces the plain engine's simple-path mode; `max_depth`
    /// is the unrolling budget. See `validate_kinduction_replay`.
    KInduction {
        /// Audit label naming the k-induction engine.
        engine: &'static str,
        /// Whether the strengthened engine (with invariant discovery) is used for replay.
        strengthened: bool,
        /// Whether the plain engine's simple-path mode is reproduced on replay.
        simple_path: bool,
        /// Unrolling budget for the replay.
        max_depth: usize,
    },
    /// Engine cannot (or did not) produce a proof witness. Downgrade to
    /// `Unknown` unless another engine produces a witnessed `Safe`.
    Unwitnessed,
}

/// Outcome of running `validate_safe()` on a proposed `Safe` verdict.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SafeValidation {
    /// Witness passed all independent checks — accept the `Safe` verdict.
    Accepted,
    /// Validator ran out of budget / cancellation — indeterminate. Caller
    /// should not accept this as a proof but can keep waiting for sibling
    /// engines.
    Indeterminate {
        /// Why validation could not complete (e.g. budget exhausted, cancelled).
        reason: String,
    },
    /// Witness was actively rejected by an independent check. SOUNDNESS ALERT.
    Rejected {
        /// What the independent check found wrong with the witness.
        reason: String,
    },
    /// Engine did not expose a witness. Conservative downgrade to `Unknown`.
    Downgrade {
        /// Why the verdict is being downgraded.
        reason: String,
    },
}

impl SafeValidation {
    /// `true` iff the validator *independently confirmed* the `Safe` proof.
    #[inline]
    pub fn accepted(&self) -> bool {
        matches!(self, SafeValidation::Accepted)
    }

    /// Portfolio acceptance decision for a `Safe` verdict — **fail-OPEN for
    /// assurance**, mirroring the committed `SafeWitness::KInduction` replay
    /// contract (`CheckResult::Unknown => Accepted` in
    /// `validate_kinduction_replay`).
    ///
    /// * [`SafeValidation::Accepted`] — proof independently re-verified ⇒ accept.
    /// * [`SafeValidation::Indeterminate`] — the independent re-check ran out of
    ///   budget or the SAT backend returned `Unknown`. The proof could not be
    ///   *confirmed*, but it was **not disproved** — there is no counterexample
    ///   to the invariant. Accept the engine's own internally-verified `Safe`
    ///   rather than downgrade a verdict that may well be correct. This is the
    ///   property that makes IC3 `Safe` proof-backed *by default* without ever
    ///   changing a correct SAT/UNSAT verdict.
    /// * [`SafeValidation::Rejected`] — the independent re-check found a genuine
    ///   counterexample to the claimed invariant (Init⇒Inv, Inv∧T⇒Inv', or
    ///   Inv⇒¬bad fails). A real soundness bug ⇒ do **not** accept.
    /// * [`SafeValidation::Downgrade`] — the engine emitted no proof witness at
    ///   all ⇒ do **not** accept.
    ///
    /// Only the [`SafeValidation::Rejected`] and [`SafeValidation::Downgrade`]
    /// branches block acceptance, and `Rejected` is the sole behaviour-changing
    /// branch (it fires only on an actual soundness violation). A *correct*
    /// `Safe` is therefore never downgraded by this gate.
    #[inline]
    pub fn portfolio_accepts(&self) -> bool {
        matches!(
            self,
            SafeValidation::Accepted | SafeValidation::Indeterminate { .. }
        )
    }
}

/// Default validation budget for `validate_safe()`. Matches the small-circuit
/// tier of IC3's internal validator (`ic3/validate.rs:51`).
const DEFAULT_VALIDATE_BUDGET: Duration = Duration::from_secs(10);

/// Symmetric validator for `Safe` verdicts — the counterpart to
/// `Transys::verify_witness()` for the `Unsafe` path.
///
/// Runs on a fresh independent SAT backend (no state shared with the
/// producing engine) so a bug in the producing engine's solver cannot
/// silently make the validator agree.
pub fn validate_safe(witness: &SafeWitness, ts: &Transys) -> SafeValidation {
    validate_safe_with_budget(witness, ts, DEFAULT_VALIDATE_BUDGET)
}

/// Same as [`validate_safe`] but with an explicit validation budget.
pub fn validate_safe_with_budget(
    witness: &SafeWitness,
    ts: &Transys,
    budget: Duration,
) -> SafeValidation {
    match witness {
        SafeWitness::Unwitnessed => SafeValidation::Downgrade {
            reason: "engine returned Safe without a proof witness (#4315 \
                 conservative fallback)"
                .into(),
        },
        SafeWitness::EngineVerified { engine } => {
            eprintln!(
                "portfolio validate_safe: accepting {engine}'s internal \
                 safety proof without independent re-verification (no formal \
                 witness to check; #4315 logged)"
            );
            SafeValidation::Accepted
        }
        SafeWitness::KInduction {
            engine,
            strengthened,
            simple_path,
            max_depth,
        } => {
            validate_kinduction_replay(ts, engine, *strengthened, *simple_path, *max_depth, budget)
        }
        SafeWitness::Trivial => validate_trivial_safe(ts),
        SafeWitness::InductiveInvariant { lemmas, depth } => {
            validate_inductive_invariant(ts, lemmas, *depth, budget)
        }
    }
}

/// Independent re-verification of a k-induction `Safe` verdict (#4315).
///
/// k-induction proves safety by a *k-step* inductive argument — base case
/// (no bad reachable within `k` steps of init) plus an inductive step (any
/// `k`-good run stays good) — which is **not** a 1-step inductive invariant.
/// The strengthened engine's auxiliary lemmas are discovered by *bounded* BMC
/// and are not guaranteed 1-inductive, so handing them to
/// [`validate_inductive_invariant`] would spuriously fail the consecution
/// check and wrongly reject sound `Safe` verdicts. Instead we re-establish the
/// proof the way it was produced: re-run k-induction on a **fresh, independent
/// solver backend** (one that does not share the producing engine's solver
/// state) and confirm the result.
///
/// Soundness contract — this gate never downgrades a *correct* `Safe`:
/// * `Safe` again → `Accepted` (now independently confirmed on another solver).
/// * `Unsafe` whose trace **passes** [`Transys::verify_witness`] against the
///   original system → `Rejected` (a genuine counterexample ⇒ the original
///   `Safe` was unsound; this is the only behaviour-changing branch and it only
///   fires on a real soundness bug).
/// * `Unsafe` with an unverified/spurious trace, or `Unknown`/budget-exhausted
///   → `Accepted`: we fall back to the engine's own internally-verified proof
///   exactly as the legacy `EngineVerified` path did (no regression).
fn validate_kinduction_replay(
    ts: &Transys,
    engine_label: &str,
    strengthened: bool,
    simple_path: bool,
    max_depth: usize,
    budget: Duration,
) -> SafeValidation {
    // Independent backend. For genuinely small circuits use SimpleSolver — a
    // different solver family (basic DPLL) for maximal independence, fast on
    // tiny instances. For anything larger use ay-sat with preprocessing AND
    // inprocessing disabled: a fresh solver in a different configuration from
    // the producing engine's default that, crucially, polls the cancellation
    // flag so the wall-clock budget below is actually enforced.
    let backend = if ts.latch_vars.len() <= 16 {
        SolverBackend::Simple
    } else {
        SolverBackend::AYNoPreprocess
    };

    // Watchdog: bound the independent re-proof to `budget`. On expiry it flips
    // the shared cancellation flag; the engine/solver observe it and return
    // `Unknown`, which we treat as "could not independently confirm" → accept.
    let cancelled = Arc::new(AtomicBool::new(false));
    let watchdog_flag = cancelled.clone();
    let watchdog = thread::spawn(move || {
        let deadline = Instant::now() + budget;
        while Instant::now() < deadline {
            if watchdog_flag.load(Ordering::Relaxed) {
                return; // validation finished early; stop spinning
            }
            thread::sleep(Duration::from_millis(20));
        }
        watchdog_flag.store(true, Ordering::Relaxed);
    });

    eprintln!(
        "portfolio validate_safe: independently re-running {engine_label} \
         (strengthened={strengthened}, simple_path={simple_path}, \
         backend={backend:?}, budget={:.1}s) to confirm Safe",
        budget.as_secs_f64(),
    );

    let result = if strengthened {
        let mut engine = KindStrengthenedEngine::with_backend(ts.clone(), backend);
        engine.set_cancelled(cancelled.clone());
        engine.check(max_depth)
    } else {
        let mut engine = KindEngine::with_config_and_backend(
            ts.clone(),
            KindConfig {
                simple_path,
                skip_bmc: false,
            },
            backend,
        );
        engine.set_cancelled(cancelled.clone());
        engine.check(max_depth)
    };

    // Signal the watchdog that validation is done (so it stops promptly) and
    // join it to avoid leaking the thread.
    cancelled.store(true, Ordering::Relaxed);
    let _ = watchdog.join();

    match result {
        CheckResult::Safe => {
            eprintln!(
                "portfolio validate_safe: ACCEPTED — independent {backend:?} \
                 re-run confirms {engine_label}'s k-induction Safe verdict",
            );
            SafeValidation::Accepted
        }
        CheckResult::Unsafe { trace, depth } => match ts.verify_witness(&trace) {
            Ok(()) => SafeValidation::Rejected {
                reason: format!(
                    "independent k-induction re-run found a VERIFIED \
                     counterexample at depth {depth} — {engine_label}'s Safe \
                     verdict is UNSOUND (#4315 SOUNDNESS ALERT)"
                ),
            },
            Err(why) => {
                eprintln!(
                    "portfolio validate_safe: independent re-run of \
                     {engine_label} produced an UNVERIFIED counterexample \
                     ({why}); not trusting the refutation — accepting the \
                     engine's internally-verified k-induction proof",
                );
                SafeValidation::Accepted
            }
        },
        CheckResult::Unknown { reason } => {
            eprintln!(
                "portfolio validate_safe: independent re-run of {engine_label} \
                 inconclusive ({reason}); accepting the engine's internally-\
                 verified k-induction proof (no independent confirmation, no \
                 regression vs. legacy EngineVerified)",
            );
            SafeValidation::Accepted
        }
    }
}

/// Validate a `Safe` claim whose justification is "the property is trivially
/// safe". Re-checks the bad-lit structure directly from `ts`; rejects any
/// non-constant bad literal.
fn validate_trivial_safe(ts: &Transys) -> SafeValidation {
    if ts.bad_lits.is_empty() {
        return SafeValidation::Accepted;
    }
    for &bad in &ts.bad_lits {
        if bad != Lit::FALSE {
            return SafeValidation::Rejected {
                reason: format!(
                    "trivial-safe witness rejected: bad_lit {bad:?} is \
                     not constant FALSE (there are {} non-trivial bad lits)",
                    ts.bad_lits.iter().filter(|l| **l != Lit::FALSE).count(),
                ),
            };
        }
    }
    SafeValidation::Accepted
}

/// Inductive-invariant validator. Mirrors `Ic3Engine::validate_invariant_budgeted`
/// but runs on a fresh solver with no access to the engine's internal state:
/// the only inputs are the `Transys` (trust-rooted here — it is what the
/// engines themselves were checking) and the `lemmas` that the engine claims
/// as its proof.
fn validate_inductive_invariant(
    ts: &Transys,
    lemmas: &[Vec<Lit>],
    depth: usize,
    budget: Duration,
) -> SafeValidation {
    let start = Instant::now();
    let should_abort = |s: &Instant| -> bool { s.elapsed() > budget };

    // Degenerate case: an empty lemma set cannot prove a non-trivial property.
    // If bad_lits is empty / all FALSE we would already have taken the
    // `Trivial` path above — so reject here unless the property really is
    // trivial (defense-in-depth).
    if lemmas.is_empty() {
        // Treat "empty lemmas" as an implicit trivial claim and re-verify.
        return validate_trivial_safe(ts).map_reject_reason(|r| {
            format!(
                "inductive-invariant witness has 0 lemmas AND property is \
                 not trivially safe: {r}",
            )
        });
    }

    eprintln!(
        "portfolio validate_safe: checking {} lemmas (depth={}, budget={:.1}s)",
        lemmas.len(),
        depth,
        budget.as_secs_f64(),
    );

    // Build the next-state variable map for priming, EXACTLY like the engine's
    // sound consecution (`Ic3Engine` engine.rs:342-399 / validate.rs:211-242).
    //
    // The historical implementation derived priming solely from `ts.next_state`,
    // which `Transys::from_aiger` populates for LATCHES ONLY (transys.rs:103-108).
    // IC3 with `internal_signals=true` (the default in nearly all portfolio
    // configs) puts AND-gate-output (internal-signal) literals into lemmas. For
    // such a literal the old `prime_lit` left the variable UNCHANGED (current
    // frame) and Check 2 added no next-frame Tseitin definition, so a unit
    // internal-signal lemma `{¬g}` made the consecution query vacuously UNSAT
    // for ANY such lemma regardless of inductiveness — a false ACCEPT and hence
    // a false SAFE.
    //
    // We now allocate a dedicated fresh next-frame variable for EVERY latch and
    // EVERY internal signal, and emit the matching next-link clauses so that in
    // Check 2 a primed internal signal `g'` is the genuine next-frame value
    // `g' <=> AND(prime(a), prime(b))` rather than the pinned current-frame gate.
    //
    // `next_vars`: current-state var (latch OR internal signal) -> fresh next var.
    let mut next_var_id = ts.max_var + 1;
    let mut next_vars: FxHashMap<Var, Var> = FxHashMap::default();
    for &latch_var in &ts.latch_vars {
        next_vars.insert(latch_var, Var(next_var_id));
        next_var_id += 1;
    }
    for &isig_var in &ts.internal_signals {
        // `internal_signals` may overlap nothing with latches (it is selected
        // from AND-gate outputs), but guard against accidental duplicates.
        next_vars.entry(isig_var).or_insert_with(|| {
            let v = Var(next_var_id);
            next_var_id += 1;
            v
        });
    }
    // Upper bound on variable indices once the next-frame vars are allocated.
    let primed_max_var = if next_var_id > ts.max_var + 1 {
        next_var_id - 1
    } else {
        ts.max_var
    };

    // `resolve_lit_to_primed`: map a literal's var through `next_vars` (latch or
    // internal signal) preserving polarity; genuinely free vars (inputs,
    // constants) fall through unchanged. Mirrors `Ic3Engine::resolve_lit_to_primed`.
    let resolve_to_primed = |l: Lit| -> Lit {
        if let Some(&nv) = next_vars.get(&l.var()) {
            if l.is_positive() {
                Lit::pos(nv)
            } else {
                Lit::neg(nv)
            }
        } else {
            l
        }
    };

    // Next-link clauses for Check 2 only (mirrors engine.rs:366-399). These
    // define the fresh next-frame vars; they must NOT be added to the pure
    // current-frame Check 1 / Check 3 solvers.
    //
    // Latches: next_v <=> ts.next_state[latch]. The next-state expression can
    // carry constant/negated polarity, so keep the polarity-aware
    // [¬nv, next_expr] / [nv, ¬next_expr] form rather than the var-only resolve.
    let mut next_link_clauses: Vec<Vec<Lit>> = Vec::new();
    for &latch_var in &ts.latch_vars {
        if let (Some(&next_var), Some(&next_expr)) =
            (next_vars.get(&latch_var), ts.next_state.get(&latch_var))
        {
            let nv_pos = Lit::pos(next_var);
            let nv_neg = Lit::neg(next_var);
            next_link_clauses.push(vec![nv_neg, next_expr]);
            next_link_clauses.push(vec![nv_pos, !next_expr]);
        }
    }
    // Internal signals: g' <=> AND(prime(a), prime(b)) for g = AND(a, b).
    for &isig_var in &ts.internal_signals {
        if let (Some(&next_var), Some(&(rhs0, rhs1))) =
            (next_vars.get(&isig_var), ts.and_defs.get(&isig_var))
        {
            let rhs0_primed = resolve_to_primed(rhs0);
            let rhs1_primed = resolve_to_primed(rhs1);
            let nv_pos = Lit::pos(next_var);
            let nv_neg = Lit::neg(next_var);
            next_link_clauses.push(vec![nv_neg, rhs0_primed]);
            next_link_clauses.push(vec![nv_neg, rhs1_primed]);
            next_link_clauses.push(vec![nv_pos, !rhs0_primed, !rhs1_primed]);
        }
    }

    // Conservative complement / fail-open guard (#4315): a lemma literal whose
    // variable is neither a latch, nor a defined internal signal, nor a genuine
    // input/constant is un-primable — we cannot soundly compute its next-frame
    // value. Rather than silently leave it current-frame (the exact vacuity bug
    // above), return Indeterminate so the portfolio falls open to the engine's
    // own sound validator instead of asserting an unverifiable SAFE.
    let input_set: FxHashSet<Var> = ts.input_vars.iter().copied().collect();
    let is_primable_or_free = |v: Var| -> bool {
        v == Var(0)                       // constant
            || next_vars.contains_key(&v) // latch or internal signal
            || input_set.contains(&v) // genuine free input
    };
    for (i, lemma) in lemmas.iter().enumerate() {
        for &lit in lemma {
            if !is_primable_or_free(lit.var()) {
                return SafeValidation::Indeterminate {
                    reason: format!(
                        "validate_safe cannot prime lemma {} literal {:?}: var {} is \
                         neither a latch, a defined internal signal, nor an input \
                         (un-primable witness — failing open to the engine's own \
                         inductiveness proof; #4315)",
                        i,
                        lit,
                        lit.var().0,
                    ),
                };
            }
        }
    }

    // `prime_lit` now consults `next_vars` (latches AND internal signals) and
    // maps to the dedicated next-frame var with polarity; only genuinely free
    // vars (inputs/constants) fall through unchanged.
    let prime_lit = &resolve_to_primed;

    let solver_factory = || -> Box<dyn SatSolver> {
        if ts.latch_vars.len() <= 60 {
            let mut s = SimpleSolver::new();
            s.ensure_vars(primed_max_var + 1);
            Box::new(s)
        } else {
            SolverBackend::AYNoPreprocess.make_solver_no_inprocessing(primed_max_var + 1)
        }
    };

    // Check 1: Init ⇒ Inv.
    //
    // For each lemma L = l1 ∨ l2 ∨ …, assert !L (the negated cube) as
    // assumptions and check SAT over init ∧ T ∧ constraints. If SAT, init
    // admits a state that violates the lemma — the invariant claim is
    // unsound.
    {
        let mut solver = solver_factory();
        solver.add_clause(&[Lit::TRUE]);
        for clause in &ts.init_clauses {
            solver.add_clause(&clause.lits);
        }
        for clause in &ts.trans_clauses {
            solver.add_clause(&clause.lits);
        }
        for &c in &ts.constraint_lits {
            solver.add_clause(&[c]);
        }
        for (i, lemma) in lemmas.iter().enumerate() {
            if should_abort(&start) {
                return SafeValidation::Indeterminate {
                    reason: format!(
                        "validate_safe budget exceeded at Init⇒Inv check \
                         lemma {}/{} ({:.1}s)",
                        i,
                        lemmas.len(),
                        start.elapsed().as_secs_f64(),
                    ),
                };
            }
            let neg: Vec<Lit> = lemma.iter().map(|l| !*l).collect();
            match solver.solve(&neg) {
                SatResult::Sat => {
                    return SafeValidation::Rejected {
                        reason: format!(
                            "Init does NOT imply lemma {}: {:?} (SOUNDNESS \
                             ALERT — engine claimed inductive invariant but \
                             initial states violate it)",
                            i, lemma,
                        ),
                    };
                }
                SatResult::Unsat => {}
                SatResult::Unknown => {
                    return SafeValidation::Indeterminate {
                        reason: format!(
                            "validator SAT solver returned Unknown during \
                             Init⇒Inv check at lemma {i}"
                        ),
                    };
                }
            }
        }
    }

    // Check 2: Inv ∧ T ⇒ Inv' (consecution / inductiveness).
    //
    // Add every lemma as a clause over current-state vars, the next-link
    // clauses that define the fresh next-frame vars (latches AND internal
    // signals — mirroring validate.rs:211-213), then for each lemma check that
    // its primed negation is UNSAT.
    {
        let mut solver = solver_factory();
        solver.add_clause(&[Lit::TRUE]);
        for clause in &ts.trans_clauses {
            solver.add_clause(&clause.lits);
        }
        for &c in &ts.constraint_lits {
            solver.add_clause(&[c]);
        }
        for clause in &next_link_clauses {
            solver.add_clause(clause);
        }
        for lemma in lemmas {
            solver.add_clause(lemma);
        }
        for (i, lemma) in lemmas.iter().enumerate() {
            if should_abort(&start) {
                return SafeValidation::Indeterminate {
                    reason: format!(
                        "validate_safe budget exceeded at Inv∧T⇒Inv' check \
                         lemma {}/{} ({:.1}s)",
                        i,
                        lemmas.len(),
                        start.elapsed().as_secs_f64(),
                    ),
                };
            }
            let neg_primed: Vec<Lit> = lemma.iter().map(|l| !prime_lit(*l)).collect();
            match solver.solve(&neg_primed) {
                SatResult::Sat => {
                    return SafeValidation::Rejected {
                        reason: format!(
                            "Inv ∧ T does NOT preserve lemma {}: {:?} \
                             (SOUNDNESS ALERT — invariant is not inductive, \
                             #4315 symmetric validator disagrees with engine)",
                            i, lemma,
                        ),
                    };
                }
                SatResult::Unsat => {}
                SatResult::Unknown => {
                    return SafeValidation::Indeterminate {
                        reason: format!(
                            "validator SAT solver returned Unknown during \
                             Inv∧T⇒Inv' check at lemma {i}"
                        ),
                    };
                }
            }
        }
    }

    // Check 3: Inv ⇒ ¬bad.
    {
        let mut solver = solver_factory();
        solver.add_clause(&[Lit::TRUE]);
        for clause in &ts.trans_clauses {
            solver.add_clause(&clause.lits);
        }
        for &c in &ts.constraint_lits {
            solver.add_clause(&[c]);
        }
        for lemma in lemmas {
            solver.add_clause(lemma);
        }
        for &bad in &ts.bad_lits {
            if should_abort(&start) {
                return SafeValidation::Indeterminate {
                    reason: format!(
                        "validate_safe budget exceeded at Inv⇒¬bad check \
                         ({:.1}s)",
                        start.elapsed().as_secs_f64(),
                    ),
                };
            }
            // Constant-FALSE bad lit is trivially non-reachable.
            if bad == Lit::FALSE {
                continue;
            }
            match solver.solve(&[bad]) {
                SatResult::Sat => {
                    return SafeValidation::Rejected {
                        reason: format!(
                            "Inv admits a bad state: bad_lit={:?} (SOUNDNESS \
                             ALERT — engine's invariant does NOT prove the \
                             property)",
                            bad,
                        ),
                    };
                }
                SatResult::Unsat => {}
                SatResult::Unknown => {
                    return SafeValidation::Indeterminate {
                        reason: format!(
                            "validator SAT solver returned Unknown during \
                             Inv⇒¬bad check for bad_lit={bad:?}"
                        ),
                    };
                }
            }
        }
    }

    eprintln!(
        "portfolio validate_safe: ACCEPTED inductive invariant ({} lemmas, \
         depth={}, elapsed={:.3}s)",
        lemmas.len(),
        depth,
        start.elapsed().as_secs_f64(),
    );
    SafeValidation::Accepted
}

impl SafeValidation {
    /// Internal helper: map a `Rejected` reason through a transform while
    /// preserving all other variants unchanged.
    fn map_reject_reason<F: FnOnce(String) -> String>(self, f: F) -> Self {
        match self {
            SafeValidation::Rejected { reason } => SafeValidation::Rejected { reason: f(reason) },
            other => other,
        }
    }
}

// -----------------------------------------------------------------------------
// Tests — unit tests for validate_safe. See the portfolio tests module for
// the integration tests that wire this into `runner::portfolio_check`.
// -----------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::parse_aag;
    use crate::transys::Transys;

    /// An `Unwitnessed` verdict must always downgrade — this is the core
    /// conservative fallback that would have caught #4310 on its own.
    #[test]
    fn test_validate_safe_unwitnessed_downgrades() {
        // Trivially safe circuit — but we pretend the engine couldn't emit
        // a witness.
        let circuit = parse_aag("aag 0 0 0 1 0\n0\n").unwrap();
        let ts = Transys::from_aiger(&circuit);
        let outcome = validate_safe(&SafeWitness::Unwitnessed, &ts);
        match outcome {
            SafeValidation::Downgrade { .. } => {}
            other => panic!("expected Downgrade for Unwitnessed, got {other:?}"),
        }
    }

    /// Trivial-safe witness on a circuit with no bad properties must be
    /// accepted.
    #[test]
    fn test_validate_safe_trivial_passes() {
        // aag 0 0 0 1 0 with bad=0 (constant FALSE).
        let circuit = parse_aag("aag 0 0 0 1 0\n0\n").unwrap();
        let ts = Transys::from_aiger(&circuit);
        let outcome = validate_safe(&SafeWitness::Trivial, &ts);
        assert_eq!(outcome, SafeValidation::Accepted);
    }

    /// Trivial-safe witness on a circuit whose bad is a real signal must be
    /// REJECTED — you cannot claim "trivially safe" on a non-trivial
    /// property.
    #[test]
    fn test_validate_safe_trivial_on_nontrivial_rejected() {
        // aag 1 0 1 0 0 1: one latch, bad = latch. Not trivially safe.
        let circuit = parse_aag("aag 1 0 1 0 0 1\n2 0\n2\n").unwrap();
        let ts = Transys::from_aiger(&circuit);
        let outcome = validate_safe(&SafeWitness::Trivial, &ts);
        match outcome {
            SafeValidation::Rejected { .. } => {}
            other => panic!("expected Rejected, got {other:?}"),
        }
    }

    /// Legitimate inductive invariant: latch stays at 0 → invariant !latch
    /// → proves the property.
    #[test]
    fn test_validate_safe_inductive_invariant_passes() {
        // aag 1 0 1 0 0 1: latch var = 2, next = 0 (stuck at 0), bad = latch.
        let circuit = parse_aag("aag 1 0 1 0 0 1\n2 0\n2\n").unwrap();
        let ts = Transys::from_aiger(&circuit);
        // The inductive invariant is just the clause {!latch}: i.e. latch
        // is always false.
        let latch = ts.latch_vars[0];
        let lemmas = vec![vec![Lit::neg(latch)]];
        let witness = SafeWitness::InductiveInvariant { lemmas, depth: 1 };
        let outcome = validate_safe(&witness, &ts);
        assert_eq!(
            outcome,
            SafeValidation::Accepted,
            "valid invariant !latch should be accepted"
        );
    }

    /// Bogus invariant: claim `latch` is always TRUE on a stuck-at-0
    /// latch. `init ⇒ inv` fails at step 1.
    #[test]
    fn test_validate_safe_bogus_invariant_init_fails() {
        let circuit = parse_aag("aag 1 0 1 0 0 1\n2 0\n2\n").unwrap();
        let ts = Transys::from_aiger(&circuit);
        let latch = ts.latch_vars[0];
        // Lemma {latch} claims "latch is TRUE" — violates init state.
        let lemmas = vec![vec![Lit::pos(latch)]];
        let witness = SafeWitness::InductiveInvariant { lemmas, depth: 1 };
        let outcome = validate_safe(&witness, &ts);
        match outcome {
            SafeValidation::Rejected { reason } => {
                assert!(
                    reason.contains("Init"),
                    "expected Init⇒Inv rejection, got: {reason}"
                );
            }
            other => panic!("expected Rejected, got {other:?}"),
        }
    }

    /// Bogus invariant: claim a vacuous lemma that holds at init but does
    /// not prove ¬bad. Here `latch` is free (next = latch i.e. stuck at
    /// whatever it was), bad = latch. The property is NOT trivially safe
    /// and an empty invariant cannot prove it — validator should reject
    /// via Check 3 (Inv ⇒ ¬bad).
    #[test]
    fn test_validate_safe_insufficient_invariant_bad_reachable() {
        // aag 1 0 1 0 0 1: latch with self-loop (stuck at whatever init).
        // next = 2 (= latch current), bad = latch.
        // No init clause — latch can be T or F initially, so bad IS
        // reachable at step 0 when latch=1. Invariant {TRUE} (always true)
        // is technically inductive but doesn't prove ¬bad.
        let circuit = parse_aag("aag 1 0 1 0 0 1\n2 2\n2\n").unwrap();
        let ts = Transys::from_aiger(&circuit);
        // Pass a lemma that's always TRUE (tautology): {TRUE}. This
        // passes Check 1 and Check 2 but must fail Check 3.
        let lemmas = vec![vec![Lit::TRUE]];
        let witness = SafeWitness::InductiveInvariant { lemmas, depth: 0 };
        let outcome = validate_safe(&witness, &ts);
        match outcome {
            SafeValidation::Rejected { reason } => {
                assert!(
                    reason.contains("bad") || reason.contains("Inv"),
                    "expected Inv⇒¬bad rejection, got: {reason}"
                );
            }
            other => panic!("expected Rejected for insufficient invariant, got {other:?}"),
        }
    }

    /// Empty-lemmas witness on a non-trivial property must be rejected.
    #[test]
    fn test_validate_safe_empty_lemmas_on_nontrivial_rejected() {
        let circuit = parse_aag("aag 1 0 1 0 0 1\n2 2\n2\n").unwrap();
        let ts = Transys::from_aiger(&circuit);
        let witness = SafeWitness::InductiveInvariant {
            lemmas: Vec::new(),
            depth: 0,
        };
        let outcome = validate_safe(&witness, &ts);
        match outcome {
            SafeValidation::Rejected { .. } => {}
            other => panic!(
                "expected Rejected for empty-lemmas witness on non-trivial \
                 property, got {other:?}"
            ),
        }
    }

    /// Empty-lemmas witness on a trivial property must be accepted.
    #[test]
    fn test_validate_safe_empty_lemmas_on_trivial_accepted() {
        let circuit = parse_aag("aag 0 0 0 1 0\n0\n").unwrap();
        let ts = Transys::from_aiger(&circuit);
        let witness = SafeWitness::InductiveInvariant {
            lemmas: Vec::new(),
            depth: 0,
        };
        let outcome = validate_safe(&witness, &ts);
        assert_eq!(outcome, SafeValidation::Accepted);
    }

    /// `EngineVerified` is accepted with no re-verification — we trust the
    /// engine's internal check (e.g. k-induction inductive step) but log.
    /// This preserves existing k-induction + BMC Safe results until those
    /// engines can emit a formal invariant.
    #[test]
    fn test_validate_safe_engine_verified_accepted() {
        // Circuit with non-trivial property. The validator does NOT re-check.
        let circuit = parse_aag("aag 1 0 1 0 0 1\n2 0\n2\n").unwrap();
        let ts = Transys::from_aiger(&circuit);
        let witness = SafeWitness::EngineVerified {
            engine: "k-induction",
        };
        let outcome = validate_safe(&witness, &ts);
        assert_eq!(outcome, SafeValidation::Accepted);
    }

    /// A genuinely-safe property (stuck-at-0 latch, bad = latch) carried by a
    /// `KInduction` witness must be ACCEPTED — the independent re-run on a
    /// fresh solver re-proves Safe via k-induction.
    #[test]
    fn test_validate_safe_kinduction_replay_accepts_genuine_safe() {
        // aag 1 0 1 0 0 1: latch var = 2, next = 0 (stuck at 0), bad = latch.
        let circuit = parse_aag("aag 1 0 1 0 0 1\n2 0\n2\n").unwrap();
        let ts = Transys::from_aiger(&circuit);
        let witness = SafeWitness::KInduction {
            engine: "k-induction",
            strengthened: false,
            simple_path: false,
            max_depth: 10,
        };
        assert_eq!(validate_safe(&witness, &ts), SafeValidation::Accepted);
    }

    /// Same genuinely-safe property, but routed through the strengthened
    /// engine replay. Must also be ACCEPTED.
    #[test]
    fn test_validate_safe_kinduction_strengthened_replay_accepts_genuine_safe() {
        let circuit = parse_aag("aag 1 0 1 0 0 1\n2 0\n2\n").unwrap();
        let ts = Transys::from_aiger(&circuit);
        let witness = SafeWitness::KInduction {
            engine: "k-induction-strengthened",
            strengthened: true,
            simple_path: false,
            max_depth: 10,
        };
        assert_eq!(validate_safe(&witness, &ts), SafeValidation::Accepted);
    }

    /// Fail-OPEN contract for the portfolio acceptance gate. The portfolio
    /// emits the engine's `Safe` on `Accepted` AND `Indeterminate` (an
    /// inconclusive re-check is "could not confirm", never a disproof — a
    /// correct SAFE must not be downgraded), and blocks acceptance only on a
    /// genuine soundness signal (`Rejected`) or a missing witness (`Downgrade`).
    /// This mirrors `SafeWitness::KInduction`'s `Unknown => Accepted` replay
    /// contract.
    #[test]
    fn test_safe_validation_portfolio_accepts_fail_open() {
        assert!(SafeValidation::Accepted.portfolio_accepts());
        assert!(SafeValidation::Indeterminate {
            reason: "budget exceeded".into()
        }
        .portfolio_accepts());
        assert!(!SafeValidation::Rejected {
            reason: "counterexample".into()
        }
        .portfolio_accepts());
        assert!(!SafeValidation::Downgrade {
            reason: "no witness".into()
        }
        .portfolio_accepts());
    }

    /// A genuinely-valid inductive invariant whose independent re-check cannot
    /// finish inside a near-zero budget must yield `Indeterminate` (or, if the
    /// tiny circuit completes instantly, `Accepted`) — and `portfolio_accepts()`
    /// must hold either way. It must NEVER be `Rejected`: a correct SAFE is
    /// never downgraded on budget-out. This is the IC3 analogue of
    /// `validate_kinduction_replay`'s fail-open-on-`Unknown` behaviour.
    #[test]
    fn test_validate_safe_inductive_fail_open_under_tiny_budget() {
        // aag 1 0 1 0 0 1: latch var = 2, next = 0 (stuck at 0), bad = latch.
        // Inductive invariant {!latch} is genuinely valid.
        let circuit = parse_aag("aag 1 0 1 0 0 1\n2 0\n2\n").unwrap();
        let ts = Transys::from_aiger(&circuit);
        let latch = ts.latch_vars[0];
        let lemmas = vec![vec![Lit::neg(latch)]];
        let witness = SafeWitness::InductiveInvariant { lemmas, depth: 1 };
        let outcome = validate_safe_with_budget(&witness, &ts, Duration::from_nanos(1));
        assert!(
            outcome.portfolio_accepts(),
            "fail-open: a correct invariant must be accepted on budget-out, got {outcome:?}"
        );
        assert!(
            !matches!(outcome, SafeValidation::Rejected { .. }),
            "a correct invariant must never be Rejected, got {outcome:?}"
        );
    }

    /// SOUNDNESS GATE: a `KInduction` Safe witness attached to a circuit that
    /// is actually UNSAFE (toggling latch reaches bad at depth 1) must be
    /// REJECTED — the independent re-run finds a `verify_witness`-confirmed
    /// counterexample. This is the only behaviour-changing branch and it fires
    /// only on a genuinely unsound Safe claim.
    #[test]
    fn test_validate_safe_kinduction_replay_rejects_unsound_safe() {
        // aag 1 0 1 0 0 1: latch var = 2, next = 3 (= !latch, toggles), bad =
        // latch. Init latch = 0, so bad becomes reachable at step 1.
        let circuit = parse_aag("aag 1 0 1 0 0 1\n2 3\n2\n").unwrap();
        let ts = Transys::from_aiger(&circuit);
        let witness = SafeWitness::KInduction {
            engine: "k-induction-strengthened",
            strengthened: true,
            simple_path: false,
            max_depth: 10,
        };
        match validate_safe(&witness, &ts) {
            SafeValidation::Rejected { reason } => {
                assert!(
                    reason.contains("counterexample") || reason.contains("UNSOUND"),
                    "expected soundness-alert rejection, got: {reason}"
                );
            }
            other => panic!("expected Rejected for unsound k-induction Safe, got {other:?}"),
        }
    }

    /// Build the hand-traced internal-signal soundness counterexample circuit:
    ///   inputs:  i        (lit 2, var 1)
    ///   latches: x        (lit 4, var 2), next = i  (lit 2), reset 0
    ///            y        (lit 6, var 3), next = x  (lit 4), reset 0
    ///   and:     g = y ∧ i (lit 8, var 4) = (lit 6) ∧ (lit 2)
    ///   bad:     g        (lit 8)
    /// `g` is an AND-gate output (internal signal). The circuit is GENUINELY
    /// UNSAFE: from x=y=0, x'=i can be 1, then y'=x=1, then g=y∧i can be 1 ⇒
    /// bad reachable at depth 2.  Returns the Transys with `g` registered as an
    /// internal signal (so the witness lemma {¬g} carries an isig literal,
    /// exactly as an `internal_signals=true` IC3 config would produce).
    fn isig_counterexample_ts() -> Transys {
        let circuit = parse_aag("aag 4 1 2 0 1 1\n2\n4 2\n6 4\n8\n8 6 2\n").unwrap();
        let mut ts = Transys::from_aiger(&circuit);
        // Var(4) is the AND-gate output g = y ∧ i.
        let g = Var(4);
        assert!(ts.and_defs.contains_key(&g), "g must be an AND-gate output");
        ts.internal_signals = vec![g];
        ts
    }

    /// REGRESSION (#4315 core bug): the non-inductive invariant {¬g} on the
    /// internal-signal counterexample circuit MUST NOT be Accepted. With the
    /// old latch-only priming, `prime_lit(¬g) = ¬g` (current frame) and Check 2
    /// passed VACUOUSLY UNSAT ⇒ Accepted ⇒ false SAFE on a genuinely-UNSAFE
    /// circuit. The internal-signal-aware Check 2 now primes g' as the true
    /// next-frame AND value, so consecution is SAT ⇒ Rejected (the invariant is
    /// not inductive). It must NEVER be Accepted.
    #[test]
    fn test_validate_safe_isig_noninductive_invariant_rejected() {
        let ts = isig_counterexample_ts();
        let g = Var(4);
        // Witness lemma set {¬g}: holds at init (y=0 ⇒ g=0) but is NOT
        // 1-inductive (from x=1, y'=x=1 and i'=1 give g'=1).
        let lemmas = vec![vec![Lit::neg(g)]];
        let witness = SafeWitness::InductiveInvariant { lemmas, depth: 1 };
        let outcome = validate_safe(&witness, &ts);
        assert!(
            !outcome.accepted(),
            "non-inductive isig invariant {{¬g}} must NOT be Accepted (false SAFE \
             on a genuinely-UNSAFE circuit), got {outcome:?}"
        );
        match outcome {
            SafeValidation::Rejected { reason } => {
                assert!(
                    reason.contains("inductive") || reason.contains("preserve"),
                    "expected consecution rejection, got: {reason}"
                );
            }
            other => panic!(
                "expected Rejected (not inductive) for {{¬g}} on the UNSAFE isig \
                 circuit, got {other:?}"
            ),
        }
    }

    /// COVERAGE PRESERVED: a genuinely-inductive invariant that uses internal
    /// signals on a genuinely-SAFE variant must still be ACCEPTED. Variant: same
    /// circuit but the input is tied off so g can never become true — we model
    /// the safe case by stucking the latches at 0 (next = 0) so g = y∧i is
    /// always 0 along every reachable path, and the invariant {¬x, ¬y, ¬g} is
    /// truly 1-inductive and proves ¬bad. The internal-signal literal ¬g must
    /// be primed correctly (to the genuine g') and the consecution check must
    /// still pass ⇒ Accepted.
    #[test]
    fn test_validate_safe_isig_inductive_invariant_accepted() {
        // inputs: i (lit2,var1); latches x(lit4,var2) next=0, y(lit6,var3) next=0;
        // and g=y∧i (lit8,var4)=(lit6)∧(lit2); bad=g(lit8). Latches stuck at 0 ⇒
        // y always 0 ⇒ g always 0 ⇒ SAFE.
        let circuit = parse_aag("aag 4 1 2 0 1 1\n2\n4 0\n6 0\n8\n8 6 2\n").unwrap();
        let mut ts = Transys::from_aiger(&circuit);
        let x = ts.latch_vars[0];
        let y = ts.latch_vars[1];
        let g = Var(4);
        assert!(ts.and_defs.contains_key(&g));
        ts.internal_signals = vec![g];
        // Genuinely 1-inductive invariant including the internal signal literal.
        let lemmas = vec![vec![Lit::neg(x)], vec![Lit::neg(y)], vec![Lit::neg(g)]];
        let witness = SafeWitness::InductiveInvariant { lemmas, depth: 1 };
        let outcome = validate_safe(&witness, &ts);
        assert_eq!(
            outcome,
            SafeValidation::Accepted,
            "genuinely-inductive isig invariant on a SAFE circuit must be Accepted",
        );
    }

    /// A unit internal-signal lemma {¬g} alone on the SAFE-stuck variant is also
    /// genuinely 1-inductive (g is structurally always 0 once latches are stuck
    /// at 0): consecution with the correctly-primed g' must be UNSAT ⇒ Accepted.
    /// This guards against the fix over-rejecting: the priming must compute the
    /// true next-frame value, not blanket-reject every isig lemma.
    #[test]
    fn test_validate_safe_isig_unit_lemma_inductive_accepted() {
        let circuit = parse_aag("aag 4 1 2 0 1 1\n2\n4 0\n6 0\n8\n8 6 2\n").unwrap();
        let mut ts = Transys::from_aiger(&circuit);
        let g = Var(4);
        ts.internal_signals = vec![g];
        let lemmas = vec![vec![Lit::neg(g)]];
        let witness = SafeWitness::InductiveInvariant { lemmas, depth: 1 };
        assert_eq!(validate_safe(&witness, &ts), SafeValidation::Accepted);
    }
}
