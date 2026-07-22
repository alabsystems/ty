// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Independent, disjoint-trust SAFE certifier for bounded-array BTOR2 nets
//! (HWMCC Track 2, design §1.2/§4.2 of `docs/hwmcc/array-ic3-design.md`).
//!
//! # Why
//!
//! [`crate::to_chc::check_btor2_adaptive`] proof-backs a word-level array-PDR
//! `SAFE` verdict by re-checking the discovered inductive invariant with
//! `external_invariant_model_excludes_error` — but that re-check runs through
//! the *same* ay-dpll ARRAY theory that produced the verdict. A bug in that
//! (trusted, non-LRAT-checked) array theory would be a silent false-SAFE that
//! the proof-back cannot catch, because it *is* that stack.
//!
//! This module is a **second, disjoint** gate. For the class of nets whose
//! state arrays are small enough to bit-blast, it re-discharges the invariant's
//! three one-step verification conditions
//!
//! ```text
//!   VC0 (initiation): Init(s)            /\ ¬Inv(s)   is UNSAT
//!   VC1 (consecution): Inv(s) /\ T(s,s') /\ ¬Inv(s')  is UNSAT
//!   VC2 (safety):      Inv(s)            /\ Bad(s)     is UNSAT   (per property)
//! ```
//!
//! as **ground, bit-level SAT queries** — arrays are fully flattened to
//! bit-vectors (2^iw cells of ew bits), so the leaf solver is *pure
//! propositional* ay-sat with **no array theory whatsoever**. Each UNSAT is
//! then re-verified by the separate [`ay_lrat_check`] crate against the exact
//! CNF fed to ay-sat. The two engines share no array reasoning: the only common
//! ancestor is the CNF, which this module builds itself.
//!
//! # Soundness (fail-closed, additive)
//!
//! The certifier NEVER changes a verdict. It only *adds* a certificate:
//!
//! - All three VCs LRAT-verified UNSAT  → [`IndependentCertResult::Certified`].
//! - Any VC is SAT, inconclusive, over the resource cap, uses an op the
//!   bit-blaster does not model, or the invariant is out of scope
//!                                       → [`IndependentCertResult::NotConfirmed`].
//!
//! A wrong/weak invariant makes a VC **SAT** → not confirmed (the gate can say
//! no, proven by tests). A bit-blaster bug can only make a VC spuriously SAT
//! (→ withhold a certificate: sound, a coverage loss) or spuriously UNSAT (→
//! fail to *catch* a Gate-A bug: no worse than the proof-back alone). Neither
//! direction can manufacture a SAFE the PDR did not already claim.
//!
//! The VC expressions come verbatim from [`crate::to_chc::VcComponents`] (the
//! same clause bodies the PDR portfolio solved) and the invariant comes verbatim
//! from the portfolio's `VerifiedInvariant` model, so the negations are faithful
//! by construction rather than re-derived.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use ay_chc::{
    engines, AdaptiveConfig, AdaptivePortfolio, ChcExpr, ChcOp, ChcSort, ChcVar, InvariantModel,
    PdrConfig, VerifiedChcResult,
};

use crate::error::Btor2Error;
use crate::to_chc::{translate_to_chc_with_vc, StateVarEntry, VcComponents};
use crate::types::Btor2Program;

// ---------------------------------------------------------------------------
// Structural caps (shared with the bit-blaster's array expansion limits) — a
// bounded array is one small enough that its 2^iw cells fully expand. These are
// fail-closed structural bounds; the *resource* cap below is the adaptive one.
// ---------------------------------------------------------------------------

/// Max array index width the certifier expands (2^width cells). Matches
/// `bitblast::ARRAY_INDEX_MAX_BITS`.
const ARRAY_INDEX_MAX_BITS: u32 = 12;
/// Max flat expanded array width in bits. Matches `bitblast::ARRAY_FLAT_MAX_BITS`.
const ARRAY_FLAT_MAX_BITS: u64 = 8192;

/// Conservative estimate of the bytes an AIG AND-gate costs once Tseitin-encoded
/// into CNF and handed to ay-sat (3 clauses × ~3 lits × 4-byte ints, plus solver
/// bookkeeping). Used only to convert the resource-derived byte budget into a
/// gate ceiling — not a behavioural magic constant.
const BYTES_PER_GATE: u64 = 128;

/// Fraction of the host's effective-available memory the certifier is willing to
/// spend materializing one VC's CNF. Mirrors the MDD/BDD adaptive-budget
/// discipline (derive the cap from a live probe, then take a safe fraction).
const BUDGET_FRACTION_DIV: u64 = 16;

// ---------------------------------------------------------------------------
// Public result type
// ---------------------------------------------------------------------------

/// Outcome of the independent SAFE certifier. It is a *certificate strength*
/// marker layered on top of an existing verdict — never a verdict itself.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IndependentCertResult {
    /// The PDR `SAFE` was independently re-confirmed: all three one-step VCs of
    /// the discovered invariant were discharged as LRAT-checked UNSAT through a
    /// pure-propositional ay-sat leaf disjoint from ay-dpll's array theory.
    Certified {
        /// Widest state-array index width bit-blasted (0 if no array state).
        index_width_bits: u32,
        /// Total array cells expanded across all bounded state arrays.
        cells: usize,
        /// Number of one-step VCs discharged (2 + #bad-properties).
        vcs_discharged: usize,
    },
    /// The certifier withheld confirmation. The underlying verdict is unchanged;
    /// only its certificate is single-gated (the ay-chc proof-back) rather than
    /// dual-gated. `reason` says why (out of scope, over cap, a VC came back
    /// SAT/inconclusive, or an unmodelled operator).
    NotConfirmed {
        /// Human-readable explanation of why confirmation was withheld.
        reason: String,
    },
}

impl IndependentCertResult {
    /// True only for [`IndependentCertResult::Certified`].
    #[must_use]
    pub fn is_certified(&self) -> bool {
        matches!(self, Self::Certified { .. })
    }

    fn not(reason: impl Into<String>) -> Self {
        Self::NotConfirmed {
            reason: reason.into(),
        }
    }
}

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

/// Run the array-PDR portfolio on `program`, and — only if it returns `SAFE`
/// with an invariant that survives the ay-chc proof-back — independently
/// re-certify that SAFE through the disjoint bit-level LRAT path.
///
/// This is **opt-in and additive**: it re-runs the adaptive portfolio to obtain
/// the invariant, it never emits `SAFE` where the portfolio did not, and it can
/// only return [`IndependentCertResult::Certified`] or
/// [`IndependentCertResult::NotConfirmed`].
///
/// `time_budget` bounds the portfolio solve (the leaf SAT queries are ground and
/// small, so they self-bound). Determinism: the same net yields the same result.
///
/// # Errors
///
/// Returns [`Btor2Error`] only if the initial BTOR2→CHC translation fails
/// (over-wide sort, unparseable constant, undefined node). A translation that
/// succeeds but is out of the certifier's scope yields
/// `Ok(NotConfirmed { .. })`, never an error.
pub fn certify_btor2_safe_independent(
    program: &Btor2Program,
    time_budget: Option<Duration>,
) -> Result<IndependentCertResult, Btor2Error> {
    if program.bad_properties.is_empty() {
        return Ok(IndependentCertResult::not(
            "no bad properties to certify (vacuously safe)",
        ));
    }

    let (translation, components) = translate_to_chc_with_vc(program)?;

    // Run the same adaptive portfolio check_btor2_adaptive uses, to obtain the
    // invariant. We clone the problem for the proof-back (Gate A) mirror.
    let config = match time_budget {
        Some(budget) => AdaptiveConfig::with_budget(budget, false),
        None => AdaptiveConfig::default(),
    };
    let chc_problem = translation.problem.clone();
    let portfolio = AdaptivePortfolio::new(translation.problem, config);

    let inv = match portfolio.solve() {
        VerifiedChcResult::Safe(inv) => inv,
        // The certifier only strengthens an existing SAFE. Anything else is out
        // of scope — it must NEVER promote a non-SAFE to SAFE.
        VerifiedChcResult::Unsafe(_) => {
            return Ok(IndependentCertResult::not(
                "portfolio verdict is UNSAFE — nothing to certify",
            ));
        }
        _ => {
            return Ok(IndependentCertResult::not(
                "portfolio verdict is not SAFE (unknown/budget) — nothing to certify",
            ));
        }
    };

    // Gate A (mirror of check_btor2_adaptive): only a proof-backed SAFE — one
    // whose invariant provably excludes every bad state on independent
    // re-verify — is a real SAFE worth certifying. Fail-closed on Ok(false)/Err.
    match engines::external_invariant_model_excludes_error(
        &chc_problem,
        inv.model(),
        &PdrConfig::default(),
    ) {
        Ok(true) => {}
        Ok(false) => {
            return Ok(IndependentCertResult::not(
                "invariant permits a bad state on ay-chc proof-back — not a SAFE to certify",
            ));
        }
        Err(e) => {
            return Ok(IndependentCertResult::not(format!(
                "ay-chc proof-back could not be completed: {e}"
            )));
        }
    }

    // Gate B: the disjoint bit-level LRAT certifier.
    let invariant = match extract_invariant(inv.model(), &components.state_entries) {
        Ok(i) => i,
        Err(reason) => return Ok(IndependentCertResult::not(reason)),
    };

    Ok(discharge_vcs_lrat(&components, &invariant))
}

// ---------------------------------------------------------------------------
// Invariant extraction (from the portfolio model)
// ---------------------------------------------------------------------------

/// The inductive invariant as a bit-blastable term: the single predicate's
/// formula plus its formal parameters (positional to the state variables).
pub(crate) struct Invariant {
    /// Formal parameters of the `Inv` predicate, in argument order.
    pub(crate) params: Vec<ChcVar>,
    /// The invariant formula over `params`.
    pub(crate) formula: ChcExpr,
}

/// Pull the single-predicate invariant out of the model and validate it is in
/// scope: exactly one predicate, arity matching the state vector, matching
/// per-position sorts, and no free (non-parameter) variables.
fn extract_invariant(
    model: &InvariantModel,
    state_entries: &[StateVarEntry],
) -> Result<Invariant, String> {
    if model.is_empty() {
        return Err(
            "invariant model is empty (trivial/BMC certificate — out of Gate-B scope)".into(),
        );
    }
    if model.len() != 1 {
        return Err(format!(
            "invariant model has {} predicates; certifier scope is single-predicate (Inv)",
            model.len()
        ));
    }
    // Exactly one interpretation.
    let (_, interp) = model.iter().next().expect("len==1");
    let params = interp.vars.clone();
    let formula = interp.formula.clone();

    if params.len() != state_entries.len() {
        return Err(format!(
            "invariant arity {} != state count {}",
            params.len(),
            state_entries.len()
        ));
    }
    for (p, s) in params.iter().zip(state_entries.iter()) {
        if p.sort != s.var.sort {
            return Err(format!(
                "invariant parameter sort {:?} != state sort {:?}",
                p.sort, s.var.sort
            ));
        }
    }
    // Reject free interpretation variables — bit-blasting them would treat a
    // universally-meant variable existentially, which is unsound. Fail-closed.
    use std::collections::HashSet;
    let param_names: HashSet<&str> = params.iter().map(|v| v.name.as_str()).collect();
    for v in formula.vars() {
        if !param_names.contains(v.name.as_str()) {
            return Err(format!(
                "invariant references free variable `{}` (out of Gate-B scope)",
                v.name
            ));
        }
    }
    Ok(Invariant { params, formula })
}

/// Substitute `params[i] -> replacement[i]` throughout `expr` (variable
/// renaming only). Used to instantiate `Inv` at the current vs. next state.
fn subst_vars(expr: &ChcExpr, map: &HashMap<ChcVar, ChcVar>) -> ChcExpr {
    match expr {
        ChcExpr::Var(v) => match map.get(v) {
            Some(target) => ChcExpr::Var(target.clone()),
            None => expr.clone(),
        },
        ChcExpr::Op(op, args) => ChcExpr::Op(
            *op,
            args.iter().map(|a| Arc::new(subst_vars(a, map))).collect(),
        ),
        ChcExpr::ConstArray(sort, v) => {
            ChcExpr::ConstArray(sort.clone(), Arc::new(subst_vars(v, map)))
        }
        // Leaves / unsupported nodes pass through unchanged; the bit-blaster
        // declines on anything it cannot model.
        other => other.clone(),
    }
}

// ---------------------------------------------------------------------------
// VC discharge
// ---------------------------------------------------------------------------

pub(crate) fn discharge_vcs_lrat(
    components: &VcComponents,
    invariant: &Invariant,
) -> IndependentCertResult {
    // Must have at least one bounded array state — else this is the scalar
    // class the bit-level lane already owns, not the certifier's target.
    let mut widest_iw = 0u32;
    let mut total_cells: usize = 0;
    let mut saw_array = false;
    for e in &components.state_entries {
        match array_dims(&e.var.sort) {
            Some((iw, ew, cells, flat)) => {
                saw_array = true;
                if iw > ARRAY_INDEX_MAX_BITS || flat > ARRAY_FLAT_MAX_BITS {
                    return IndependentCertResult::not(format!(
                        "state array `{}` too large to bit-blast (iw={iw}, ew={ew}, flat={flat} bits) — over structural cap",
                        e.var.name
                    ));
                }
                widest_iw = widest_iw.max(iw);
                total_cells = total_cells.saturating_add(cells);
            }
            None => {
                // A non-bitvector-indexed/element array is not expandable.
                if matches!(e.var.sort, ChcSort::Array(_, _)) {
                    return IndependentCertResult::not(format!(
                        "state array `{}` has non-bitvector index/element — not expandable",
                        e.var.name
                    ));
                }
            }
        }
    }
    if !saw_array {
        return IndependentCertResult::not(
            "no bounded array state — scalar nets are out of the array certifier's scope",
        );
    }

    let gate_ceiling = derive_gate_ceiling();

    // Build the current/next substitution maps for Inv.
    let mut curr_map: HashMap<ChcVar, ChcVar> = HashMap::new();
    let mut next_map: HashMap<ChcVar, ChcVar> = HashMap::new();
    for (p, e) in invariant.params.iter().zip(components.state_entries.iter()) {
        curr_map.insert(p.clone(), e.var.clone());
        next_map.insert(p.clone(), e.next_var.clone());
    }
    let inv_curr = subst_vars(&invariant.formula, &curr_map);
    let inv_next = subst_vars(&invariant.formula, &next_map);

    // ---- VC0: initiation — Init(s) /\ ¬Inv(s) UNSAT -------------------------
    {
        let mut b = Blaster::new(gate_ceiling);
        let init_lit = match b.assert_true(&components.init_constraint) {
            Ok(l) => l,
            Err(reason) => return decline_vc("VC0/init", reason),
        };
        let inv_lit = match b.blast_bool(&inv_curr) {
            Ok(l) => l,
            Err(reason) => return decline_vc("VC0/inv", reason),
        };
        b.assert_lit(init_lit);
        b.assert_lit(Blaster::neg(inv_lit)); // ¬Inv
        match b.solve_unsat_lrat() {
            LeafOutcome::VerifiedUnsat => {}
            LeafOutcome::Sat => {
                return IndependentCertResult::not(
                    "VC0 (initiation Init/\\¬Inv) is SAT — invariant is not initiation-sound",
                )
            }
            LeafOutcome::Inconclusive(r) => return decline_vc("VC0", r),
        }
    }

    // ---- VC1: consecution — Inv(s) /\ T(s,s') /\ ¬Inv(s') UNSAT -------------
    {
        let mut b = Blaster::new(gate_ceiling);
        let inv_curr_lit = match b.blast_bool(&inv_curr) {
            Ok(l) => l,
            Err(reason) => return decline_vc("VC1/inv", reason),
        };
        b.assert_lit(inv_curr_lit);
        if let Some(trans) = &components.trans_constraint {
            let t_lit = match b.assert_true(trans) {
                Ok(l) => l,
                Err(reason) => return decline_vc("VC1/trans", reason),
            };
            b.assert_lit(t_lit);
        }
        let inv_next_lit = match b.blast_bool(&inv_next) {
            Ok(l) => l,
            Err(reason) => return decline_vc("VC1/inv'", reason),
        };
        b.assert_lit(Blaster::neg(inv_next_lit)); // ¬Inv'
        match b.solve_unsat_lrat() {
            LeafOutcome::VerifiedUnsat => {}
            LeafOutcome::Sat => {
                return IndependentCertResult::not(
                    "VC1 (consecution Inv/\\T/\\¬Inv') is SAT — invariant is not inductive",
                )
            }
            LeafOutcome::Inconclusive(r) => return decline_vc("VC1", r),
        }
    }

    // ---- VC2_i: safety — Inv(s) /\ Bad_i(s) UNSAT, one per property --------
    for (i, bad_body) in components.bad_bodies.iter().enumerate() {
        let mut b = Blaster::new(gate_ceiling);
        let inv_lit = match b.blast_bool(&inv_curr) {
            Ok(l) => l,
            Err(reason) => return decline_vc("VC2/inv", reason),
        };
        b.assert_lit(inv_lit);
        let bad_lit = match b.assert_true(bad_body) {
            Ok(l) => l,
            Err(reason) => return decline_vc("VC2/bad", reason),
        };
        b.assert_lit(bad_lit);
        match b.solve_unsat_lrat() {
            LeafOutcome::VerifiedUnsat => {}
            LeafOutcome::Sat => {
                return IndependentCertResult::not(format!(
                    "VC2 (safety Inv/\\bad) for property {i} is SAT — invariant does not exclude the bad state"
                ))
            }
            LeafOutcome::Inconclusive(r) => return decline_vc("VC2", r),
        }
    }

    IndependentCertResult::Certified {
        index_width_bits: widest_iw,
        cells: total_cells,
        vcs_discharged: 2 + components.bad_bodies.len(),
    }
}

fn decline_vc(which: &str, reason: String) -> IndependentCertResult {
    IndependentCertResult::not(format!("{which}: {reason}"))
}

/// Resource-derived byte budget shared by the certifier's adaptive caps: a
/// safe fraction of the host's effective-available memory (cgroup/confinement
/// aware), floored at the collective floor. No fixed magic size — it tracks
/// the live memory probe. Denominated in BYTES so it converts directly both
/// into the CNF gate ceiling and into ay-sat's byte-denominated proof
/// bookkeeping meter.
pub(crate) fn derive_byte_budget() -> u64 {
    let avail = tla_resource::platform::effective_available_bytes()
        .map(|b| b as u64)
        .unwrap_or_else(|| tla_resource::collective_floor_bytes() as u64);
    (avail / BUDGET_FRACTION_DIV).max(tla_resource::collective_floor_bytes() as u64)
}

/// Adaptive gate ceiling: the shared byte budget converted to an AND-gate
/// count.
fn derive_gate_ceiling() -> u64 {
    (derive_byte_budget() / BYTES_PER_GATE).max(1)
}

/// `(index_width, element_width, cells, flat_bits)` for a bounded array sort,
/// or `None` if not an array or its index/element are not plain bitvectors.
fn array_dims(sort: &ChcSort) -> Option<(u32, u32, usize, u64)> {
    let ChcSort::Array(idx, elem) = sort else {
        return None;
    };
    let ChcSort::BitVec(iw) = idx.as_ref() else {
        return None;
    };
    let ChcSort::BitVec(ew) = elem.as_ref() else {
        return None;
    };
    let cells = 1usize.checked_shl(*iw)?;
    let flat = (cells as u64).checked_mul(u64::from(*ew))?;
    Some((*iw, *ew, cells, flat))
}

// ===========================================================================
// ChcExpr -> AIG bit-blaster (bounded-array flattening) -> CNF
//
// Gate primitives (mk_and/mk_or/mk_xor/mk_mux) and array/arith algorithms are
// the SAME ones bitblast.rs uses for BTOR2 nodes, ported to operate on ChcExpr
// and to emit into an AIG that is finally Tseitin-encoded to CNF. Reusing the
// proven algorithms keeps the VC's bit-level semantics matched to the trusted
// bit-blaster; the only new surface is the ChcExpr traversal + CNF emission.
//
// AIGER literal encoding: lit = var<<1 | negated. lit 0 = FALSE, lit 1 = TRUE.
// ===========================================================================

/// A bit-blasted value. Bit vectors are LSB-first vectors of AIG literals.
enum Blasted {
    /// A Bool: a single truth literal.
    Bool(u64),
    /// A bitvector: LSB-first literals.
    Bv(Vec<u64>),
    /// A fully expanded array: `flat` is `cells*ew` bits (cell `e` at
    /// `[e*ew, e*ew+ew)`), with `ew` the element width.
    Array { flat: Vec<u64>, ew: usize },
}

/// Terminal leaf outcome of the LRAT-checked ay-sat query.
///
/// `pub(crate)`: the lazy-array k-induction lane ([`crate::array_bmc`])
/// re-discharges its base/step queries through this same disjoint leaf as its
/// independent second trust path (the Gate-B discipline).
pub(crate) enum LeafOutcome {
    /// ay-sat returned UNSAT and ay-lrat-check verified the proof.
    VerifiedUnsat,
    /// ay-sat returned SAT (the VC's negation is satisfiable → VC does not hold).
    Sat,
    /// Neither a verified UNSAT nor a SAT (budget/timeout/unverifiable proof).
    Inconclusive(String),
}

struct Blaster {
    /// Next fresh variable index (var 0 reserved for constant FALSE).
    next_var: u64,
    /// AIG AND gates: (lhs, rhs0, rhs1); lhs always even (a positive literal).
    ands: Vec<(u64, u64, u64)>,
    /// Top-level literals that must be TRUE for the VC's negation to hold.
    asserts: Vec<u64>,
    /// Cache: a variable's blasted signal (so shared vars share bits).
    var_cache: HashMap<ChcVar, Vec<u64>>,
    /// Fail if the AIG exceeds this many AND gates (resource cap).
    gate_ceiling: u64,
    /// Set when the ceiling is hit; the blast then declines.
    over_budget: bool,
}

impl Blaster {
    fn new(gate_ceiling: u64) -> Self {
        Self {
            next_var: 1,
            ands: Vec::new(),
            asserts: Vec::new(),
            var_cache: HashMap::new(),
            gate_ceiling,
            over_budget: false,
        }
    }

    // ---- AIG primitives (copied verbatim from bitblast.rs) -----------------

    fn alloc_var(&mut self) -> u64 {
        let lit = self.next_var << 1;
        self.next_var += 1;
        lit
    }

    #[inline]
    fn neg(lit: u64) -> u64 {
        lit ^ 1
    }

    fn mk_and(&mut self, a: u64, b: u64) -> u64 {
        if a == 0 || b == 0 {
            return 0;
        }
        if a == 1 {
            return b;
        }
        if b == 1 {
            return a;
        }
        if a == b {
            return a;
        }
        if a == Self::neg(b) {
            return 0;
        }
        if self.ands.len() as u64 >= self.gate_ceiling {
            self.over_budget = true;
            // Return a fresh unconstrained var; the caller will decline once it
            // observes `over_budget`, so this value is never trusted.
            return self.alloc_var();
        }
        let lhs = self.alloc_var();
        self.ands.push((lhs, a, b));
        lhs
    }

    fn mk_or(&mut self, a: u64, b: u64) -> u64 {
        if a == 1 || b == 1 {
            return 1;
        }
        if a == 0 {
            return b;
        }
        if b == 0 {
            return a;
        }
        if a == b {
            return a;
        }
        if a == Self::neg(b) {
            return 1;
        }
        Self::neg(self.mk_and(Self::neg(a), Self::neg(b)))
    }

    fn mk_xor(&mut self, a: u64, b: u64) -> u64 {
        if a == b {
            return 0;
        }
        if a == Self::neg(b) {
            return 1;
        }
        if a == 0 {
            return b;
        }
        if a == 1 {
            return Self::neg(b);
        }
        if b == 0 {
            return a;
        }
        if b == 1 {
            return Self::neg(a);
        }
        let and1 = self.mk_and(a, Self::neg(b));
        let and2 = self.mk_and(Self::neg(a), b);
        self.mk_or(and1, and2)
    }

    fn mk_mux(&mut self, sel: u64, a: u64, b: u64) -> u64 {
        if sel == 1 {
            return a;
        }
        if sel == 0 {
            return b;
        }
        if a == b {
            return a;
        }
        let t = self.mk_and(sel, a);
        let f = self.mk_and(Self::neg(sel), b);
        self.mk_or(t, f)
    }

    fn bitwise_and(&mut self, a: &[u64], b: &[u64]) -> Vec<u64> {
        a.iter().zip(b).map(|(&x, &y)| self.mk_and(x, y)).collect()
    }
    fn bitwise_or(&mut self, a: &[u64], b: &[u64]) -> Vec<u64> {
        a.iter().zip(b).map(|(&x, &y)| self.mk_or(x, y)).collect()
    }
    fn bitwise_xor(&mut self, a: &[u64], b: &[u64]) -> Vec<u64> {
        a.iter().zip(b).map(|(&x, &y)| self.mk_xor(x, y)).collect()
    }

    /// Ripple-carry add (mod 2^n), discarding carry-out.
    fn add_signals(&mut self, a: &[u64], b: &[u64]) -> Vec<u64> {
        let n = a.len().min(b.len());
        let mut result = Vec::with_capacity(n);
        let mut carry = 0u64;
        for i in 0..n {
            let axb = self.mk_xor(a[i], b[i]);
            let sum = self.mk_xor(axb, carry);
            // carry' = (a&b) | (carry & (a^b))
            let ab = self.mk_and(a[i], b[i]);
            let cc = self.mk_and(carry, axb);
            carry = self.mk_or(ab, cc);
            result.push(sum);
        }
        result
    }

    /// Two's-complement negation: ~a + 1.
    fn negate_signal(&mut self, a: &[u64]) -> Vec<u64> {
        let not_a: Vec<u64> = a.iter().map(|&l| Self::neg(l)).collect();
        let mut one = vec![0u64; a.len()];
        if !one.is_empty() {
            one[0] = 1;
        }
        self.add_signals(&not_a, &one)
    }

    fn sub_signals(&mut self, a: &[u64], b: &[u64]) -> Vec<u64> {
        let neg_b = self.negate_signal(b);
        self.add_signals(a, &neg_b)
    }

    /// 1-bit: a == b (bitwise, all-equal).
    fn eq_signals(&mut self, a: &[u64], b: &[u64]) -> u64 {
        if a.len() != b.len() {
            // Unequal widths cannot be equal in a well-typed net; treat as a
            // hard mismatch by returning FALSE.
            return 0;
        }
        let mut acc = 1u64;
        for (&x, &y) in a.iter().zip(b) {
            let xnor = Self::neg(self.mk_xor(x, y));
            acc = self.mk_and(acc, xnor);
        }
        acc
    }

    /// 1-bit unsigned less-than: a < b.
    fn ult_signals(&mut self, a: &[u64], b: &[u64]) -> u64 {
        // Compare from MSB down: borrow-chain. lt = OR over i of (a_i<b_i AND
        // higher bits equal). Implement via subtract borrow: a<b iff the
        // (n+1)-bit subtraction a-b borrows.
        let n = a.len();
        // borrow_in starts 0 at LSB; a - b - borrow; final borrow-out = a<b.
        let mut borrow = 0u64;
        for i in 0..n {
            // diff_i = a_i xor b_i xor borrow; borrow' = (!a_i & b_i) | (!(a_i xor b_i) & borrow)
            let axb = self.mk_xor(a[i], b[i]);
            let not_ai = Self::neg(a[i]);
            let b1 = self.mk_and(not_ai, b[i]);
            let not_axb = Self::neg(axb);
            let b2 = self.mk_and(not_axb, borrow);
            borrow = self.mk_or(b1, b2);
        }
        borrow
    }

    /// 1-bit signed less-than: a <_s b.
    fn slt_signals(&mut self, a: &[u64], b: &[u64]) -> u64 {
        // a <_s b  iff  (a<b unsigned) XOR (sign_a != sign_b)
        let n = a.len();
        if n == 0 {
            return 0;
        }
        let ult = self.ult_signals(a, b);
        let sign_a = a[n - 1];
        let sign_b = b[n - 1];
        let signs_differ = self.mk_xor(sign_a, sign_b);
        self.mk_xor(ult, signs_differ)
    }

    /// Array select: one-hot mux of cell `index` of `n` cells, each `ew` bits.
    fn array_read(&mut self, array: &[u64], index: &[u64], ew: usize, n: usize) -> Vec<u64> {
        if let Some(e) = Self::const_index_value(index) {
            return if e < n {
                array[e * ew..(e + 1) * ew].to_vec()
            } else {
                vec![0u64; ew]
            };
        }
        let mut result = vec![0u64; ew];
        for e in 0..n {
            let sel = self.index_eq_const(index, e);
            for b in 0..ew {
                let term = self.mk_and(sel, array[e * ew + b]);
                result[b] = self.mk_or(result[b], term);
            }
        }
        result
    }

    /// Array store: new flat array where cell `e` becomes `(index==e)?value:old`.
    fn array_write(
        &mut self,
        array: &[u64],
        index: &[u64],
        value: &[u64],
        ew: usize,
        n: usize,
    ) -> Vec<u64> {
        if let Some(e) = Self::const_index_value(index) {
            let mut result = array.to_vec();
            if e < n {
                result[e * ew..(e + 1) * ew].copy_from_slice(value);
            }
            return result;
        }
        let mut result = Vec::with_capacity(n * ew);
        for e in 0..n {
            let sel = self.index_eq_const(index, e);
            for b in 0..ew {
                let old = array[e * ew + b];
                let m = self.mk_mux(sel, value[b], old);
                result.push(m);
            }
        }
        result
    }

    fn index_eq_const(&mut self, index: &[u64], val: usize) -> u64 {
        let mut acc = 1u64;
        for (i, &bit) in index.iter().enumerate() {
            let want = if i < usize::BITS as usize {
                (val >> i) & 1
            } else {
                0
            };
            let lit = if want == 1 { bit } else { Self::neg(bit) };
            acc = self.mk_and(acc, lit);
        }
        acc
    }

    fn const_index_value(index: &[u64]) -> Option<usize> {
        let mut val = 0usize;
        for (i, &bit) in index.iter().enumerate() {
            match bit {
                0 => {}
                1 => {
                    if i >= usize::BITS as usize {
                        return None;
                    }
                    val |= 1usize << i;
                }
                _ => return None,
            }
        }
        Some(val)
    }

    // ---- ChcExpr traversal --------------------------------------------------

    /// Blast a Bool-sorted expression to a single truth literal.
    fn blast_bool(&mut self, expr: &ChcExpr) -> Result<u64, String> {
        match self.blast(expr)? {
            Blasted::Bool(l) => Ok(l),
            Blasted::Bv(bits) if bits.len() == 1 => Ok(bits[0]),
            other => Err(format!(
                "expected a Bool/1-bit value, got {}",
                kind_of(&other)
            )),
        }
    }

    /// Blast a Bool expression and register it as a top-level truth assertion.
    fn assert_true(&mut self, expr: &ChcExpr) -> Result<u64, String> {
        self.blast_bool(expr)
    }

    fn assert_lit(&mut self, lit: u64) {
        self.asserts.push(lit);
    }

    fn require_bv(&mut self, expr: &ChcExpr) -> Result<Vec<u64>, String> {
        match self.blast(expr)? {
            Blasted::Bv(bits) => Ok(bits),
            Blasted::Bool(l) => Ok(vec![l]),
            Blasted::Array { .. } => Err("expected a bitvector, got an array".into()),
        }
    }

    fn require_array(&mut self, expr: &ChcExpr) -> Result<(Vec<u64>, usize), String> {
        match self.blast(expr)? {
            Blasted::Array { flat, ew } => Ok((flat, ew)),
            _ => Err("expected an array".into()),
        }
    }

    fn blast(&mut self, expr: &ChcExpr) -> Result<Blasted, String> {
        if self.over_budget {
            return Err("gate ceiling exceeded (resource cap)".into());
        }
        match expr {
            ChcExpr::Bool(b) => Ok(Blasted::Bool(if *b { 1 } else { 0 })),
            ChcExpr::BitVec(v, w) => {
                let w = *w as usize;
                let mut bits = Vec::with_capacity(w);
                for i in 0..w {
                    bits.push(if (v >> i) & 1 == 1 { 1 } else { 0 });
                }
                Ok(Blasted::Bv(bits))
            }
            ChcExpr::Var(v) => self.blast_var(v),
            ChcExpr::ConstArray(key_sort, val) => {
                let ChcSort::BitVec(iw) = key_sort else {
                    return Err("const-array over non-bitvector index".into());
                };
                let cells = 1usize
                    .checked_shl(*iw)
                    .ok_or("const-array index width overflow")?;
                if *iw > ARRAY_INDEX_MAX_BITS {
                    return Err(format!("const-array index width {iw} over cap"));
                }
                let val_bits = self.require_bv(val)?;
                let ew = val_bits.len();
                let mut flat = Vec::with_capacity(cells * ew);
                for _ in 0..cells {
                    flat.extend_from_slice(&val_bits);
                }
                Ok(Blasted::Array { flat, ew })
            }
            ChcExpr::Op(op, args) => self.blast_op(*op, args),
            other => Err(format!("unsupported expression node: {other:?}")),
        }
    }

    fn blast_var(&mut self, v: &ChcVar) -> Result<Blasted, String> {
        if let Some(bits) = self.var_cache.get(v) {
            let bits = bits.clone();
            return Ok(match &v.sort {
                ChcSort::Array(_, elem) => {
                    let ew = bv_width(elem).ok_or("array element not bitvector")? as usize;
                    Blasted::Array { flat: bits, ew }
                }
                ChcSort::Bool => Blasted::Bool(bits[0]),
                _ => Blasted::Bv(bits),
            });
        }
        let blasted = match &v.sort {
            ChcSort::BitVec(w) => {
                let bits: Vec<u64> = (0..*w).map(|_| self.alloc_var()).collect();
                self.var_cache.insert(v.clone(), bits.clone());
                Blasted::Bv(bits)
            }
            ChcSort::Bool => {
                let lit = self.alloc_var();
                self.var_cache.insert(v.clone(), vec![lit]);
                Blasted::Bool(lit)
            }
            ChcSort::Array(idx, elem) => {
                let iw = bv_width(idx).ok_or("array index not bitvector")?;
                let ew = bv_width(elem).ok_or("array element not bitvector")?;
                if iw > ARRAY_INDEX_MAX_BITS {
                    return Err(format!("array index width {iw} over cap"));
                }
                let cells = 1usize.checked_shl(iw).ok_or("array index width overflow")?;
                let flat_bits = (cells as u64)
                    .checked_mul(u64::from(ew))
                    .ok_or("array flat width overflow")?;
                if flat_bits > ARRAY_FLAT_MAX_BITS {
                    return Err(format!("array flat width {flat_bits} over cap"));
                }
                let flat: Vec<u64> = (0..flat_bits).map(|_| self.alloc_var()).collect();
                self.var_cache.insert(v.clone(), flat.clone());
                Blasted::Array {
                    flat,
                    ew: ew as usize,
                }
            }
            other => return Err(format!("unsupported variable sort: {other:?}")),
        };
        Ok(blasted)
    }

    fn blast_op(&mut self, op: ChcOp, args: &[Arc<ChcExpr>]) -> Result<Blasted, String> {
        match op {
            // ---- Boolean connectives --------------------------------------
            ChcOp::Not => {
                let a = self.blast_bool(&args[0])?;
                Ok(Blasted::Bool(Self::neg(a)))
            }
            ChcOp::And => {
                let mut acc = 1u64;
                for a in args {
                    let l = self.blast_bool(a)?;
                    acc = self.mk_and(acc, l);
                }
                Ok(Blasted::Bool(acc))
            }
            ChcOp::Or => {
                let mut acc = 0u64;
                for a in args {
                    let l = self.blast_bool(a)?;
                    acc = self.mk_or(acc, l);
                }
                Ok(Blasted::Bool(acc))
            }
            ChcOp::Implies => {
                let a = self.blast_bool(&args[0])?;
                let b = self.blast_bool(&args[1])?;
                let na = Self::neg(a);
                Ok(Blasted::Bool(self.mk_or(na, b)))
            }
            ChcOp::Iff => {
                let a = self.blast_bool(&args[0])?;
                let b = self.blast_bool(&args[1])?;
                Ok(Blasted::Bool(Self::neg(self.mk_xor(a, b))))
            }

            // ---- Equality (bv / bool / array) -----------------------------
            ChcOp::Eq | ChcOp::Ne => {
                let eq = self.blast_eq(&args[0], &args[1])?;
                Ok(Blasted::Bool(if op == ChcOp::Ne {
                    Self::neg(eq)
                } else {
                    eq
                }))
            }

            // ---- Conditional ----------------------------------------------
            ChcOp::Ite => {
                let cond = self.blast_bool(&args[0])?;
                let then_b = self.blast(&args[1])?;
                let else_b = self.blast(&args[2])?;
                self.blast_ite(cond, then_b, else_b)
            }

            // ---- Bitvector unary ------------------------------------------
            ChcOp::BvNot => {
                let a = self.require_bv(&args[0])?;
                Ok(Blasted::Bv(a.iter().map(|&l| Self::neg(l)).collect()))
            }
            ChcOp::BvNeg => {
                let a = self.require_bv(&args[0])?;
                let r = self.negate_signal(&a);
                Ok(Blasted::Bv(r))
            }

            // ---- Bitvector binary bitwise ---------------------------------
            ChcOp::BvAnd => self.bv_bin(args, |s, a, b| s.bitwise_and(a, b)),
            ChcOp::BvOr => self.bv_bin(args, |s, a, b| s.bitwise_or(a, b)),
            ChcOp::BvXor => self.bv_bin(args, |s, a, b| s.bitwise_xor(a, b)),
            ChcOp::BvNand => {
                let r = self.bv_bin(args, |s, a, b| s.bitwise_and(a, b))?;
                Ok(bv_map_neg(r))
            }
            ChcOp::BvNor => {
                let r = self.bv_bin(args, |s, a, b| s.bitwise_or(a, b))?;
                Ok(bv_map_neg(r))
            }
            ChcOp::BvXnor => {
                let r = self.bv_bin(args, |s, a, b| s.bitwise_xor(a, b))?;
                Ok(bv_map_neg(r))
            }

            // ---- Bitvector arithmetic -------------------------------------
            ChcOp::BvAdd => self.bv_bin(args, |s, a, b| s.add_signals(a, b)),
            ChcOp::BvSub => self.bv_bin(args, |s, a, b| s.sub_signals(a, b)),

            // ---- Bitvector comparisons (return 1-bit Bool) ----------------
            ChcOp::BvULt => self.bv_cmp(args, |s, a, b| s.ult_signals(a, b)),
            ChcOp::BvUGt => self.bv_cmp(args, |s, a, b| s.ult_signals(b, a)),
            ChcOp::BvULe => self.bv_cmp(args, |s, a, b| {
                let gt = s.ult_signals(b, a);
                Self::neg(gt)
            }),
            ChcOp::BvUGe => self.bv_cmp(args, |s, a, b| {
                let lt = s.ult_signals(a, b);
                Self::neg(lt)
            }),
            ChcOp::BvSLt => self.bv_cmp(args, |s, a, b| s.slt_signals(a, b)),
            ChcOp::BvSGt => self.bv_cmp(args, |s, a, b| s.slt_signals(b, a)),
            ChcOp::BvSLe => self.bv_cmp(args, |s, a, b| {
                let gt = s.slt_signals(b, a);
                Self::neg(gt)
            }),
            ChcOp::BvSGe => self.bv_cmp(args, |s, a, b| {
                let lt = s.slt_signals(a, b);
                Self::neg(lt)
            }),
            ChcOp::BvComp => {
                let eq = self.blast_eq(&args[0], &args[1])?;
                Ok(Blasted::Bv(vec![eq]))
            }

            // ---- Bitvector structural -------------------------------------
            ChcOp::BvConcat => {
                // (concat a b) with b the low bits; LSB-first => b ++ a.
                let a = self.require_bv(&args[0])?;
                let b = self.require_bv(&args[1])?;
                let mut r = b;
                r.extend_from_slice(&a);
                Ok(Blasted::Bv(r))
            }
            ChcOp::BvExtract(hi, lo) => {
                let a = self.require_bv(&args[0])?;
                let (hi, lo) = (hi as usize, lo as usize);
                if lo > hi || hi >= a.len() {
                    return Err(format!(
                        "extract [{hi}:{lo}] out of range for width {}",
                        a.len()
                    ));
                }
                Ok(Blasted::Bv(a[lo..=hi].to_vec()))
            }
            ChcOp::BvZeroExtend(n) => {
                let mut a = self.require_bv(&args[0])?;
                a.extend(std::iter::repeat(0u64).take(n as usize));
                Ok(Blasted::Bv(a))
            }
            ChcOp::BvSignExtend(n) => {
                let a = self.require_bv(&args[0])?;
                let sign = *a.last().unwrap_or(&0);
                let mut r = a;
                r.extend(std::iter::repeat(sign).take(n as usize));
                Ok(Blasted::Bv(r))
            }

            // ---- Bitvector shifts (constant amount only) ------------------
            ChcOp::BvShl => self.bv_shift(args, ShiftKind::Left),
            ChcOp::BvLShr => self.bv_shift(args, ShiftKind::LogicalRight),
            ChcOp::BvAShr => self.bv_shift(args, ShiftKind::ArithRight),

            // ---- Array ops ------------------------------------------------
            ChcOp::Select => {
                let (flat, ew) = self.require_array(&args[0])?;
                let index = self.require_bv(&args[1])?;
                let n = if ew == 0 { 0 } else { flat.len() / ew };
                let r = self.array_read(&flat, &index, ew, n);
                Ok(Blasted::Bv(r))
            }
            ChcOp::Store => {
                let (flat, ew) = self.require_array(&args[0])?;
                let index = self.require_bv(&args[1])?;
                let value = self.require_bv(&args[2])?;
                if value.len() != ew {
                    return Err(format!(
                        "store value width {} != element width {ew}",
                        value.len()
                    ));
                }
                let n = if ew == 0 { 0 } else { flat.len() / ew };
                let r = self.array_write(&flat, &index, &value, ew, n);
                Ok(Blasted::Array { flat: r, ew })
            }

            // ---- Everything else: decline (fail-closed) -------------------
            other => Err(format!(
                "unsupported operator {other:?} — certifier declines (fail-closed)"
            )),
        }
    }

    fn blast_eq(&mut self, a: &ChcExpr, b: &ChcExpr) -> Result<u64, String> {
        let ba = self.blast(a)?;
        let bb = self.blast(b)?;
        match (ba, bb) {
            (Blasted::Bool(x), Blasted::Bool(y)) => Ok(Self::neg(self.mk_xor(x, y))),
            (Blasted::Bv(x), Blasted::Bv(y)) => Ok(self.eq_signals(&x, &y)),
            (Blasted::Bool(x), Blasted::Bv(y)) | (Blasted::Bv(y), Blasted::Bool(x))
                if y.len() == 1 =>
            {
                Ok(Self::neg(self.mk_xor(x, y[0])))
            }
            (Blasted::Array { flat: fa, ew: ea }, Blasted::Array { flat: fb, ew: eb }) => {
                if ea != eb || fa.len() != fb.len() {
                    return Err("array equality between mismatched array sorts".into());
                }
                // Exact for fully-expanded bounded arrays: bitwise equality of
                // the flat cell vectors (no extensionality reasoning needed).
                Ok(self.eq_signals(&fa, &fb))
            }
            _ => Err("equality between mismatched kinds".into()),
        }
    }

    fn blast_ite(
        &mut self,
        cond: u64,
        then_b: Blasted,
        else_b: Blasted,
    ) -> Result<Blasted, String> {
        match (then_b, else_b) {
            (Blasted::Bool(t), Blasted::Bool(e)) => Ok(Blasted::Bool(self.mk_mux(cond, t, e))),
            (Blasted::Bv(t), Blasted::Bv(e)) => {
                if t.len() != e.len() {
                    return Err("ite branch width mismatch".into());
                }
                let r = (0..t.len())
                    .map(|i| self.mk_mux(cond, t[i], e[i]))
                    .collect();
                Ok(Blasted::Bv(r))
            }
            (Blasted::Array { flat: t, ew: et }, Blasted::Array { flat: e, ew: ee }) => {
                if et != ee || t.len() != e.len() {
                    return Err("ite array branch mismatch".into());
                }
                let r = (0..t.len())
                    .map(|i| self.mk_mux(cond, t[i], e[i]))
                    .collect();
                Ok(Blasted::Array { flat: r, ew: et })
            }
            _ => Err("ite branches have mismatched kinds".into()),
        }
    }

    fn bv_bin<F>(&mut self, args: &[Arc<ChcExpr>], f: F) -> Result<Blasted, String>
    where
        F: Fn(&mut Self, &[u64], &[u64]) -> Vec<u64>,
    {
        let a = self.require_bv(&args[0])?;
        let b = self.require_bv(&args[1])?;
        if a.len() != b.len() {
            return Err("binary bitvector op width mismatch".into());
        }
        Ok(Blasted::Bv(f(self, &a, &b)))
    }

    fn bv_cmp<F>(&mut self, args: &[Arc<ChcExpr>], f: F) -> Result<Blasted, String>
    where
        F: Fn(&mut Self, &[u64], &[u64]) -> u64,
    {
        let a = self.require_bv(&args[0])?;
        let b = self.require_bv(&args[1])?;
        if a.len() != b.len() {
            return Err("comparison width mismatch".into());
        }
        Ok(Blasted::Bool(f(self, &a, &b)))
    }

    fn bv_shift(&mut self, args: &[Arc<ChcExpr>], kind: ShiftKind) -> Result<Blasted, String> {
        let a = self.require_bv(&args[0])?;
        let amt = self.require_bv(&args[1])?;
        let shamt = Self::const_index_value(&amt)
            .ok_or("dynamic (non-constant) shift amount — certifier declines")?;
        let n = a.len();
        let fill = match kind {
            ShiftKind::ArithRight => *a.last().unwrap_or(&0),
            _ => 0,
        };
        let mut r = vec![fill; n];
        for i in 0..n {
            let src = match kind {
                ShiftKind::Left => {
                    if i >= shamt {
                        Some(i - shamt)
                    } else {
                        None
                    }
                }
                ShiftKind::LogicalRight | ShiftKind::ArithRight => {
                    if i + shamt < n {
                        Some(i + shamt)
                    } else {
                        None
                    }
                }
            };
            r[i] = match src {
                Some(j) => a[j],
                None => fill,
            };
        }
        Ok(Blasted::Bv(r))
    }

    // ---- CNF emission + LRAT-checked ay-sat leaf ---------------------------

    /// Tseitin-encode the AIG to DIMACS CNF text (asserting every top literal
    /// TRUE), or `None` if a top assertion forces the empty clause (trivially
    /// UNSAT).
    fn to_dimacs(&self) -> DimacsCnf {
        let mut clauses: Vec<Vec<i32>> =
            Vec::with_capacity(self.ands.len() * 3 + self.asserts.len());
        // Gate definitions: for lhs = a AND b, (¬lhs∨a)(¬lhs∨b)(lhs∨¬a∨¬b).
        for &(lhs, a, b) in &self.ands {
            let l = lit_to_dimacs(lhs);
            let la = lit_to_dimacs(a);
            let lb = lit_to_dimacs(b);
            clauses.push(vec![-l, la]);
            clauses.push(vec![-l, lb]);
            clauses.push(vec![l, -la, -lb]);
        }
        let mut trivially_unsat = false;
        for &t in &self.asserts {
            match t {
                1 => {} // TRUE: no constraint.
                0 => {
                    trivially_unsat = true; // FALSE asserted: empty clause.
                }
                _ => clauses.push(vec![lit_to_dimacs(t)]),
            }
        }
        DimacsCnf {
            num_vars: (self.next_var.saturating_sub(1)) as usize,
            clauses,
            trivially_unsat,
        }
    }

    /// Discharge the current asserted VC as an LRAT-checked ay-sat query.
    fn solve_unsat_lrat(&self) -> LeafOutcome {
        if self.over_budget {
            return LeafOutcome::Inconclusive("gate ceiling exceeded (resource cap)".into());
        }
        let cnf = self.to_dimacs();
        if cnf.trivially_unsat {
            // A FALSE assertion is structurally UNSAT; no proof needed. This is
            // sound (asserting `false` cannot be satisfied) and rare.
            return LeafOutcome::VerifiedUnsat;
        }
        if cnf.clauses.is_empty() {
            // No constraints ⇒ satisfiable ⇒ the VC's negation holds.
            return LeafOutcome::Sat;
        }
        // The proof bookkeeping meter is byte-denominated, so the same
        // resource-derived byte budget that capped this VC's CNF
        // materialization also caps its proof bookkeeping: runaway proof
        // construction degrades to a fail-closed Inconclusive instead of
        // eating the certify deadline.
        solve_dimacs_unsat_lrat(&cnf, Some(derive_byte_budget()))
    }
}

enum ShiftKind {
    Left,
    LogicalRight,
    ArithRight,
}

/// Bitwise-negate every literal of a blasted bitvector (for Nand/Nor/Xnor).
fn bv_map_neg(b: Blasted) -> Blasted {
    match b {
        Blasted::Bv(bits) => Blasted::Bv(bits.iter().map(|&l| Blaster::neg(l)).collect()),
        other => other,
    }
}

fn kind_of(b: &Blasted) -> &'static str {
    match b {
        Blasted::Bool(_) => "Bool",
        Blasted::Bv(_) => "BitVec",
        Blasted::Array { .. } => "Array",
    }
}

fn bv_width(sort: &ChcSort) -> Option<u32> {
    match sort {
        ChcSort::BitVec(w) => Some(*w),
        _ => None,
    }
}

/// Map an AIG literal (`var<<1 | negated`, with var ≥ 1 here) to a DIMACS int.
fn lit_to_dimacs(lit: u64) -> i32 {
    let var = (lit >> 1) as i32;
    if lit & 1 == 1 {
        -var
    } else {
        var
    }
}

pub(crate) struct DimacsCnf {
    pub(crate) num_vars: usize,
    pub(crate) clauses: Vec<Vec<i32>>,
    pub(crate) trivially_unsat: bool,
}

impl DimacsCnf {
    fn to_text(&self) -> String {
        use std::fmt::Write;
        let mut s = String::new();
        let _ = writeln!(s, "p cnf {} {}", self.num_vars.max(1), self.clauses.len());
        for cl in &self.clauses {
            for lit in cl {
                let _ = write!(s, "{lit} ");
            }
            s.push_str("0\n");
        }
        s
    }
}

/// The disjoint LRAT-checked leaf: solve `cnf` with ay-sat in proof mode; on
/// UNSAT, re-verify the emitted LRAT proof with the *separate* `ay-lrat-check`
/// crate against the identical CNF. Only a re-verified UNSAT is trusted.
///
/// This path is pure propositional SAT — it never enters ay-dpll or any theory
/// solver, so it is disjoint from the array reasoning that produced the SAFE.
///
/// `proof_work_budget` bounds ay-sat's deterministic, byte-denominated
/// search-time proof bookkeeping meter (`None` = unbounded). On exhaustion the
/// solver keeps its verdict but degrades mid-search to no-proof power, so the
/// emitted LRAT stream ends early and lands in the existing
/// proof-absent/unverifiable arms below — the same fail-closed
/// [`LeafOutcome::Inconclusive`] as a missing proof, never a certificate.
pub(crate) fn solve_dimacs_unsat_lrat(
    cnf: &DimacsCnf,
    proof_work_budget: Option<u64>,
) -> LeafOutcome {
    let dimacs = cnf.to_text();

    let formula = match ay_sat::parse_dimacs(&dimacs) {
        Ok(f) => f,
        Err(e) => return LeafOutcome::Inconclusive(format!("internal CNF parse error: {e:?}")),
    };

    // Built directly rather than via `ay_sat::PortfolioSolver::new(1)` because
    // only the raw solver exposes `set_proof_bookkeeping_budget`. Everything
    // else mirrors the portfolio's single-thread proof-mode lane through the
    // same public API it uses internally: an in-memory forward LRAT writer,
    // the default strategy knobs, and the same instance-adaptive feature
    // adjustment before solving.
    let features = ay_sat::SatFeatures::extract(formula.num_vars, &formula.clauses);
    let class = ay_sat::InstanceClass::classify(&features);
    let proof_output =
        ay_sat::ProofOutput::lrat_text(Vec::<u8>::new(), formula.clauses.len() as u64);
    let mut solver = ay_sat::Solver::with_proof_output(formula.num_vars, proof_output);
    solver.set_glucose_restarts(true);
    solver.set_chrono_enabled(true);
    solver.set_chrono_reuse_trail(true);
    solver.set_branch_selector_ucb1(false);
    solver.set_random_seed(0);
    let mut profile = ay_sat::InprocessingFeatureProfile::default();
    ay_sat::adjust_features_for_instance(&features, &class, &mut profile);
    solver.apply_feature_profile(&profile);
    solver.set_proof_bookkeeping_budget(proof_work_budget);
    for clause in &formula.clauses {
        solver.add_clause(clause.clone());
    }
    let result = solver.solve().into_inner();

    if result.is_sat() {
        return LeafOutcome::Sat;
    }
    if !result.is_unsat() {
        return LeafOutcome::Inconclusive("ay-sat returned unknown".into());
    }

    // UNSAT: require a proof and re-verify it with the independent checker.
    let proof_bytes = solver.take_proof_writer().and_then(|w| w.into_vec().ok());
    let Some(bytes) = proof_bytes else {
        return LeafOutcome::Inconclusive("ay-sat UNSAT produced no LRAT proof".into());
    };
    let proof_text = match String::from_utf8(bytes) {
        Ok(t) => t,
        Err(_) => return LeafOutcome::Inconclusive("LRAT proof not UTF-8".into()),
    };
    let steps = match ay_lrat_check::lrat_parser::parse_text_lrat(&proof_text) {
        Ok(s) => s,
        Err(e) => return LeafOutcome::Inconclusive(format!("LRAT parse failed: {e:?}")),
    };
    let cnf_ids = match ay_lrat_check::dimacs::parse_cnf_with_ids(dimacs.as_bytes()) {
        Ok(c) => c,
        Err(e) => return LeafOutcome::Inconclusive(format!("LRAT CNF parse failed: {e:?}")),
    };
    let mut checker = ay_lrat_check::checker::LratChecker::new(cnf_ids.num_vars);
    for (id, clause) in &cnf_ids.clauses {
        checker.add_original(*id, clause);
    }
    if checker.verify_proof(&steps) {
        LeafOutcome::VerifiedUnsat
    } else if let Some(budget) = proof_work_budget {
        // A finite meter truncates the LRAT stream when it exhausts
        // (deterministically), which is indistinguishable here from a genuinely
        // bad proof; both get the identical fail-closed treatment.
        LeafOutcome::Inconclusive(format!(
            "LRAT proof did not verify (possibly truncated by the deterministic \
             proof-work budget of {budget} bytes)"
        ))
    } else {
        LeafOutcome::Inconclusive("LRAT proof did not verify".into())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[allow(clippy::unwrap_used)]
#[cfg(test)]
mod tests;
